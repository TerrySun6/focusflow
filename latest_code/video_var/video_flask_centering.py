#!/usr/bin/env python3
"""
YouTube 视频语义分析器 — 基于 text_baai_ema_flask_centering.py 后端
====================================================================
流程：YouTube URL → 下载音频 → Whisper 转录 → BGE-M3 编码
      → 滑动窗口 EMA → 零均值中心化(Centering)
      → PHATE 降维 → 图聚类 → PELT 语义分段 → 交互式 3D 可视化

复用 text_baai_ema_flask_centering.py 的完整后端管线，
前端调整为 YouTube 视频分析场景，支持直接粘贴 YouTube URL。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template_string, request
from plotly.subplots import make_subplots

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── 依赖检测 ──
_PHATE_AVAILABLE: bool
try:
    import phate  # noqa: F401
    _PHATE_AVAILABLE = True
except ImportError:
    _PHATE_AVAILABLE = False

_HDBSCAN_AVAILABLE: bool
try:
    import hdbscan  # noqa: F401
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

_LEIDEN_AVAILABLE: bool
try:
    import igraph as ig  # noqa: F401
    import leidenalg  # noqa: F401
    _LEIDEN_AVAILABLE = True
except ImportError:
    _LEIDEN_AVAILABLE = False

# ── 配置 ──
@dataclass
class Config:
    input_file_path: str = ""
    doc_title: str = "video"
    sbert_model_name: str = "BAAI/bge-m3"
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
    ema_alpha: float = 0.5
    ema_bidirectional: bool = True
    ema_window_size: int = 5           # 滑动窗口大小（自动取奇数）
    phate_n_components: int = 3
    phate_knn: int = 5
    phate_knn_dist: str = "cosine"
    mds_method: str = "metric"
    pelt_penalty_multiplier: float = 5.0
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 5
    whisper_model: str = "mlx-community/whisper-small-mlx-4bit"
    window_size: int = 0               # 0 = 每个 Whisper 分段独立作为一个语义块
    step_size: int = 1                 # 滑动步长（window_size=0 时不使用）


# ═══════════════════════════════════════════════════════════════
# 模块 1：YouTube 工具（来自 video_qwen.py）
# ═══════════════════════════════════════════════════════════════

REMOTE_COMPONENTS = ["ejs:github"]
JS_RUNTIMES = {"deno": {}}


def sanitize_filename(name: str, max_len: int = 120) -> str:
    if not name:
        return "video"
    safe = "".join(c if (c.isalnum() or c in " ._-") else "_" for c in name)
    safe = " ".join(safe.split()).strip(" ._-")
    return safe[:max_len] or "video"


def get_video_title(url: str) -> str:
    try:
        import yt_dlp
        opts = {"quiet": True, "skip_download": True,
                "remote_components": REMOTE_COMPONENTS, "js_runtimes": JS_RUNTIMES}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info.get("title") or info.get("id") or "video"
    except Exception as e:
        logger.warning("获取视频标题失败: %s", e)
        return "video"


def download_audio(url: str, output_path: str):
    """使用 yt_dlp 下载 YouTube 音频为 wav。"""
    if os.path.exists(output_path):
        logger.info("音频文件已存在: %s", output_path)
        return
    import yt_dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "outtmpl": output_path.replace(".wav", ""),
        "quiet": False,
        "cookiesfrombrowser": ("safari",),
        "remote_components": REMOTE_COMPONENTS,
        "js_runtimes": JS_RUNTIMES,
    }
    logger.info("⬇️ 开始下载音频: %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    logger.info("✅ 音频下载完成: %s", output_path)


def get_script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


# ═══════════════════════════════════════════════════════════════
# 模块 2：Whisper 转录（来自 video_qwen.py）
# ═══════════════════════════════════════════════════════════════

def load_or_transcribe(audio_file: str, output_dir: str, whisper_model: str) -> List[Dict[str, Any]]:
    """Whisper 语音转文字，支持缓存。"""
    import mlx_whisper
    whisper_cache = os.path.join(output_dir, "whisper_segments.json")
    if os.path.exists(whisper_cache):
        with open(whisper_cache, "r", encoding="utf-8") as f:
            segments = json.load(f)
        logger.info("已读取 Whisper 缓存: %s", whisper_cache)
        return segments
    logger.info("🎤 开始 Whisper 转录 (模型: %s)...", whisper_model)
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=whisper_model)
    segments = result["segments"]
    with open(whisper_cache, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)
    logger.info("✅ 转录完成，共 %d 个片段", len(segments))
    return segments


def load_or_build_chunks(segments: List[Dict[str, Any]], output_dir: str,
                          window_size: int = 0, step_size: int = 1) -> List[str]:
    """构建语义块，支持缓存。window_size<=0 时每个 Whisper 分段独立作为一个语义块。"""
    chunks_cache = os.path.join(output_dir, "chunks.json")
    if os.path.exists(chunks_cache):
        with open(chunks_cache, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info("已读取 chunks 缓存: %s", chunks_cache)
        return chunks
    if window_size <= 0:
        chunks = [s["text"].strip() for s in segments]
    else:
        chunks = [
            " ".join([s["text"].strip() for s in segments[i:i + window_size]])
            for i in range(0, len(segments) - window_size + 1, step_size)
        ]
    with open(chunks_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info("已保存 chunks 缓存: %s (共 %d 个)", chunks_cache, len(chunks))
    return chunks


# ═══════════════════════════════════════════════════════════════
# 模块 3：BGE-M3 编码（来自 text_baai_ema_flask_centering.py）
# ═══════════════════════════════════════════════════════════════

def load_or_embed_bge(chunks: List[str], output_dir: str,
                       model_name: str, device: str) -> torch.Tensor:
    """使用 BGE-M3 进行语义编码，支持缓存。"""
    emb_cache = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_cache):
        embeddings = np.load(emb_cache)
        logger.info("读取 embeddings 缓存: %s", emb_cache)
        return torch.tensor(embeddings, dtype=torch.float32)
    try:
        from mlx_embedding_models.embedding import EmbeddingModel
        try:
            from transformers import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
                def _batch_encode_plus(self, batch, **kwargs):
                    return self.__call__(batch, **kwargs)
                PreTrainedTokenizerBase.batch_encode_plus = _batch_encode_plus
        except Exception:
            pass
        if hasattr(EmbeddingModel, "from_pretrained"):
            mlx_model = EmbeddingModel.from_pretrained(model_name)
        else:
            mlx_model = EmbeddingModel.from_registry("bge-m3")
        embeddings = np.array(mlx_model.encode(chunks))
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    except Exception:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        embeddings_tensor = model.encode(chunks, convert_to_tensor=True)
    np.save(emb_cache, embeddings_tensor.detach().cpu().numpy())
    return embeddings_tensor


# ═══════════════════════════════════════════════════════════════
# 模块 4：核心分析器（来自 text_baai_ema_flask_centering.py — 含 Centering）
# ═══════════════════════════════════════════════════════════════

class EMAnalyzer:
    def __init__(self, embeddings: torch.Tensor, cfg: Config):
        self.embeddings_gpu = embeddings
        self.embeddings_cpu = embeddings.cpu().numpy()
        self.embeddings_original = embeddings.clone()
        self.cfg = cfg
        self.results: Dict[str, Any] = {}

    def run_ema(self) -> np.ndarray:
        """滑动窗口 EMA + 策略 B 全局中心化。"""
        emb = self.embeddings_gpu
        n = emb.shape[0]
        alpha = self.cfg.ema_alpha
        window = self.cfg.ema_window_size

        if window <= 0 or n < 2:
            self.embeddings_cpu = emb.detach().cpu().numpy()
            self.embeddings_cpu = self.embeddings_cpu - np.mean(self.embeddings_cpu, axis=0)
            return self.embeddings_cpu

        if window % 2 == 0:
            window += 1
        k = window // 2

        j = torch.arange(-k, k + 1, device=emb.device)
        weights = alpha ** torch.abs(j)

        out = torch.zeros_like(emb)
        for i in range(n):
            left = max(0, i - k)
            right = min(n, i + k + 1)
            w = emb[left:right]
            w_left = left - (i - k)
            w_right = (i + k + 1) - right
            valid_weights = weights[w_left:window - w_right]
            valid_weights = valid_weights / valid_weights.sum()
            out[i] = (w * valid_weights.unsqueeze(1)).sum(dim=0)

        self.embeddings_gpu = out
        self.embeddings_cpu = out.detach().cpu().numpy()
        # 策略 B：零均值中心化
        self.embeddings_cpu = self.embeddings_cpu - np.mean(self.embeddings_cpu, axis=0)
        return self.embeddings_cpu

    def run_phate(self) -> Optional[np.ndarray]:
        if not _PHATE_AVAILABLE:
            return None
        import phate as _phate
        op = _phate.PHATE(
            n_components=self.cfg.phate_n_components,
            knn=self.cfg.phate_knn,
            knn_dist=self.cfg.phate_knn_dist,
            mds=self.cfg.mds_method,
            n_jobs=-1, verbose=False,
        )
        coords = op.fit_transform(self.embeddings_cpu)
        self.results["phate"] = coords
        return coords

    def run_clustering(self) -> Tuple[np.ndarray, np.ndarray]:
        emb = self.embeddings_cpu
        n = len(emb)
        cfg = self.cfg
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

        from sklearn.neighbors import NearestNeighbors
        k = int(np.clip(np.sqrt(max(n, 4)), 6, 18))
        nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
        nn.fit(emb_norm)
        dists, inds = nn.kneighbors(emb_norm)
        dists, inds = dists[:, 1:], inds[:, 1:]
        sims = np.clip(1.0 - dists, 0.0, 1.0)
        outlier_scores = 1.0 - sims.mean(axis=1)

        if _LEIDEN_AVAILABLE:
            import igraph as _ig
            import leidenalg as _la
            edges, weights, seen = [], [], set()
            for i in range(n):
                for j, sv in zip(inds[i], sims[i]):
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    if a == b or (a, b) in seen:
                        continue
                    seen.add((a, b))
                    edges.append((a, b))
                    weights.append(float(sv))
            g = _ig.Graph(n=n, edges=edges, directed=False)
            part = _la.find_partition(g, _la.RBConfigurationVertexPartition,
                                       weights=weights,
                                       resolution_parameter=cfg.leiden_resolution)
            labels = np.array(part.membership, dtype=int)
        elif _HDBSCAN_AVAILABLE:
            import hdbscan as _hdbscan
            clusterer = _hdbscan.HDBSCAN(min_cluster_size=cfg.hdbscan_min_cluster_size)
            labels = clusterer.fit_predict(emb_norm)
            outlier_scores = clusterer.outlier_scores_
        else:
            labels = np.zeros(n, dtype=int)

        self.results["clusters"] = labels
        self.results["outliers"] = outlier_scores
        return labels, outlier_scores

    def run_pelt(self) -> Dict[str, Any]:
        import ruptures as rpt
        emb = self.embeddings_cpu
        n = len(emb)
        clusters = self.results.get("clusters", np.zeros(n, dtype=int))
        orig = self.embeddings_original.detach().cpu().numpy()
        orig_distances = np.linalg.norm(orig[1:] - orig[:-1], axis=1)
        median_dist = float(np.median(orig_distances))
        penalty = median_dist * self.cfg.pelt_penalty_multiplier
        try:
            algo = rpt.Pelt(model="rbf", min_size=3).fit(emb)
            raw_cuts = algo.predict(pen=penalty)
        except Exception:
            algo = rpt.Pelt(model="l2", min_size=3).fit(emb)
            raw_cuts = algo.predict(pen=penalty * 0.1)
        boundaries = [0] + sorted(set(raw_cuts))
        topics = []
        for i in range(len(boundaries) - 1):
            l, r = boundaries[i], boundaries[i + 1]
            vals = clusters[l:r]
            if len(vals) > 0:
                u, c = np.unique(vals, return_counts=True)
                topics.append(int(u[np.argmax(c)]))
            else:
                topics.append(0)
        final_bs = [boundaries[0]]
        final_ts = []
        if topics:
            ct = topics[0]
            for i in range(1, len(topics)):
                if topics[i] != ct:
                    final_bs.append(boundaries[i])
                    final_ts.append(ct)
                    ct = topics[i]
            final_bs.append(boundaries[-1])
            final_ts.append(ct)
        else:
            final_bs, final_ts = boundaries, topics
        seg = {"boundaries": final_bs, "topics": final_ts, "n_segments": len(final_bs) - 1}
        self.results["segments"] = seg
        return seg

    def get_coherence(self, emb: torch.Tensor) -> np.ndarray:
        norm = F.normalize(emb, p=2, dim=1)
        return (norm[:-1] * norm[1:]).sum(dim=1).cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# 模块 5：分析管线（缓存 + 线程池）
# ═══════════════════════════════════════════════════════════════

_analysis_cache: Dict[str, Any] = {}
_analysis_cache_lock = threading.Lock()
_ANALYSIS_CACHE_MAX = 20

_analysis_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="video_analyze")


def _run_video_pipeline(url: str, cfg: Config) -> Dict[str, Any]:
    """完整视频分析管线。"""
    import hashlib

    video_title = get_video_title(url)
    safe_title = sanitize_filename(video_title)
    script_dir = get_script_dir()
    output_dir = os.path.join(script_dir, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    audio_file = os.path.join(output_dir, "audio.wav")

    # Step 1: 下载音频
    logger.info("Step 1/6: 下载音频...")
    download_audio(url, audio_file)

    # Step 2: Whisper 转录
    logger.info("Step 2/6: Whisper 转录...")
    segments = load_or_transcribe(audio_file, output_dir, cfg.whisper_model)

    # Step 3: 构建 Chunks（每个 Whisper 分段独立作为一个语义块）
    logger.info("Step 3/6: 构建语义块 (每个 Whisper 分段独立)...")
    chunks = load_or_build_chunks(segments, output_dir, cfg.window_size, cfg.step_size)

    # Step 4: BGE-M3 编码
    logger.info("Step 4/6: BGE-M3 编码...")
    emb_tensor = load_or_embed_bge(chunks, output_dir, cfg.sbert_model_name, cfg.device)
    logger.info("  Embedding 形状: %s", emb_tensor.shape)

    # Step 5: EMA + Centering
    logger.info("Step 5/6: EMA 平滑 + Centering...")
    analyzer = EMAnalyzer(emb_tensor, cfg)
    analyzer.run_ema()
    emb_orig_cpu = emb_tensor.detach().cpu().numpy()

    # Step 6: PHATE + 聚类 + PELT
    logger.info("Step 6/6: PHATE + 聚类 + PELT...")
    phate_coords = analyzer.run_phate()
    labels, outlier_scores = analyzer.run_clustering()
    seg = analyzer.run_pelt()

    # ── 计算连贯性 ──
    coherence_orig = analyzer.get_coherence(emb_tensor)
    coherence_smooth = analyzer.get_coherence(analyzer.embeddings_gpu)

    # ── 构建结果 ──
    result = _build_video_result(
        chunks, analyzer, coherence_orig, coherence_smooth,
        emb_orig_cpu, video_title, output_dir, cfg, segments,
    )
    return result


def _build_video_result(
    chunks: List[str],
    analyzer: EMAnalyzer,
    coherence_orig: np.ndarray,
    coherence_smooth: np.ndarray,
    emb_orig_cpu: np.ndarray,
    video_title: str,
    doc_dir: str,
    cfg: Config,
    whisper_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构建 API 返回结果（与 text_baai_ema_flask_centering.py 对接）。"""
    import hashlib

    emb_cpu = analyzer.embeddings_cpu
    n = len(chunks)
    n_clusters = len(set(analyzer.results.get("clusters", [])))
    seg_info = analyzer.results.get("segments", {"boundaries": [0, n], "topics": [0]})
    bs = seg_info["boundaries"]
    ts = seg_info["topics"]

    # ── 连贯性图 ──
    coh_fig = go.Figure()
    coh_fig.add_trace(go.Scatter(y=coherence_orig, mode="lines", name="原始", line=dict(color="#888", width=1)))
    coh_fig.add_trace(go.Scatter(y=coherence_smooth, mode="lines", name=f"EMA(α={cfg.ema_alpha})",
                                  line=dict(color="#4a6cf7", width=2)))
    for b in bs[1:-1]:
        b = int(b)
        if 0 < b < len(coherence_smooth):
            coh_fig.add_vline(x=b, line=dict(color="red", width=1, dash="dash"))
    coh_fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", y=1.1), height=250)

    # ── 簇分布柱状图 ──
    cluster_labels = analyzer.results.get("clusters", np.zeros(n, dtype=int))
    cluster_counts = {}
    for cl in cluster_labels:
        cluster_counts[int(cl)] = cluster_counts.get(int(cl), 0) + 1
    cl_fig = go.Figure()
    cl_fig.add_trace(go.Bar(
        x=[str(k) for k in sorted(cluster_counts.keys())],
        y=[cluster_counts[k] for k in sorted(cluster_counts.keys())],
        marker_color=["#4a6cf7", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    ))
    cl_fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="簇", yaxis_title="数量", height=250)

    # ── PHATE 3D ──
    phate_coords = analyzer.results.get("phate")
    phate_chart_json = None
    if phate_coords is not None:
        phate_fig = go.Figure()
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        n_seg = max(0, len(bs) - 1)
        for s_i in range(n_seg):
            l, r = int(bs[s_i]), int(bs[s_i + 1])
            idx = np.arange(l, r, dtype=int)
            if len(idx) <= 0:
                continue
            color = palette[s_i % len(palette)]
            topic = ts[s_i] if s_i < len(ts) else -1
            phate_fig.add_trace(go.Scatter3d(
                x=phate_coords[idx, 0], y=phate_coords[idx, 1], z=phate_coords[idx, 2],
                mode="markers", marker=dict(size=2, color=color, opacity=0.8),
                name=f"段{s_i + 1}|T{topic}",
                text=[chunks[j][:60] for j in idx], hoverinfo="text+name"))
        b_idx = [max(0, min(int(b), len(phate_coords) - 1)) for b in bs]
        phate_fig.add_trace(go.Scatter3d(
            x=phate_coords[b_idx, 0], y=phate_coords[b_idx, 1], z=phate_coords[b_idx, 2],
            mode="lines+markers", line=dict(color="black", width=2.5),
            marker=dict(size=4, color="black", symbol="x"), name="分割点"))
        phate_fig.add_trace(go.Scatter3d(
            x=[phate_coords[0, 0]], y=[phate_coords[0, 1]], z=[phate_coords[0, 2]],
            mode="markers+text", marker=dict(size=8, color="red", symbol="diamond"),
            text=["START"], name="起点"))
        phate_fig.add_trace(go.Scatter3d(
            x=[phate_coords[-1, 0]], y=[phate_coords[-1, 1]], z=[phate_coords[-1, 2]],
            mode="markers+text", marker=dict(size=8, color="green", symbol="circle"),
            text=["END"], name="终点"))
        phate_fig.update_layout(
            title=f"PHATE 3D — {video_title}  (α={cfg.ema_alpha}, window={cfg.ema_window_size})",
            scene=dict(aspectmode="data",
                       xaxis=dict(showbackground=False),
                       yaxis=dict(showbackground=False),
                       zaxis=dict(showbackground=False),
                       camera=dict(eye=dict(x=1.3, y=1.25, z=0.9))),
            template="plotly_white", margin=dict(l=0, r=0, t=40, b=0), height=500)
        phate_chart_json = json.dumps(json.loads(plotly.io.to_json(phate_fig)))

        # 导出交互式 HTML
        phate_html_path = os.path.join(doc_dir, "phate_3d.html")
        phate_fig.write_html(phate_html_path, include_plotlyjs="cdn")
        logger.info("已保存 PHATE 3D HTML: %s", phate_html_path)

    # ── 轨迹统计 ──
    trajectory = {}
    if phate_coords is not None:
        total_dist = float(np.sum(np.linalg.norm(phate_coords[1:] - phate_coords[:-1], axis=1)))
        end_dist = float(np.linalg.norm(phate_coords[0] - phate_coords[-1]))
        trajectory = {
            "total_dist": round(total_dist, 2),
            "end_dist": round(end_dist, 2),
            "ratio": round(total_dist / max(end_dist, 1e-9), 2),
        }

    # ── 连贯性样本 ──
    sample_step = max(1, n // 50)
    coherence_samples = []
    for i in range(0, min(n, len(coherence_orig)), sample_step):
        coherence_samples.append({
            "pos": i + 1,
            "text": chunks[i][:60] if i < len(chunks) else "",
            "orig": f"{coherence_orig[i]:.4f}" if i < len(coherence_orig) else "-",
            "smooth": f"{coherence_smooth[i]:.4f}" if i < len(coherence_smooth) else "-",
        })

    # ── 分段详情 ──
    seg_list = []
    for s_i in range(len(bs) - 1):
        l, r = int(bs[s_i]), int(bs[s_i + 1])
        topic = ts[s_i] if s_i < len(ts) else -1
        l = min(l, len(chunks) - 1)
        r = min(r, len(chunks))
        seg_chunks = chunks[l:r]
        # 获取对应时间戳
        start_time = whisper_segments[l]["start"] if l < len(whisper_segments) else 0
        end_time = whisper_segments[r - 1]["end"] if r - 1 < len(whisper_segments) else 0
        seg_list.append({
            "id": s_i + 1,
            "range": f"[{l},{r})",
            "len": r - l,
            "topic": topic,
            "time_range": f"{_format_time(start_time)} - {_format_time(end_time)}",
            "preview": (seg_chunks[0][:80] + "...") if seg_chunks else "",
        })

    result = {
        "video_title": video_title,
        "n_chunks": n,
        "ema_alpha": cfg.ema_alpha,
        "ema_window_size": cfg.ema_window_size,
        "ema_direction": "滑动窗口 EMA + Centering",
        "n_clusters": n_clusters,
        "n_segments": len(bs) - 1,
        "coherence_before": round(float(np.mean(coherence_orig)), 4),
        "coherence_after": round(float(np.mean(coherence_smooth)), 4),
        "cluster_chart_json": json.dumps(json.loads(plotly.io.to_json(cl_fig))),
        "coherence_chart_json": json.dumps(json.loads(plotly.io.to_json(coh_fig))),
        "coherence_samples": coherence_samples,
        "phate_chart_json": phate_chart_json,
        "trajectory": trajectory,
        "segments": seg_list,
        "phate_html_file": os.path.join(doc_dir, "phate_3d.html"),
        "embedding_shape": list(emb_cpu.shape),
        "output_dir": doc_dir,
    }
    return result


def _format_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS。"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════
# Flask App
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>🎬 视频语义分析器 (Centering 版)</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: #0f0f1a; color: #e0e0f0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { font-size: 1.6rem; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        h1 .badge { font-size: 0.7rem; background: #4a6cf7; color: white; padding: 2px 10px;
                    border-radius: 12px; font-weight: 400; }
        .layout { display: flex; gap: 20px; }
        .sidebar { width: 320px; flex-shrink: 0; }
        .main { flex: 1; min-width: 0; }
        .card { background: #1a1a2e; border-radius: 12px; padding: 16px; margin-bottom: 16px;
                border: 1px solid #2a2a4a; box-shadow: 0 4px 16px rgba(0,0,0,0.3); }
        .card h3 { font-size: 0.95rem; color: #8899cc; margin-bottom: 12px; letter-spacing: 0.5px; }
        label { display: block; font-size: 0.85rem; color: #8899cc; margin-bottom: 4px; margin-top: 10px; }
        input[type="text"], input[type="number"], select {
            width: 100%; padding: 8px 10px; border: 1px solid #2a2a4a; border-radius: 8px;
            font-size: 0.9rem; background: #12122a; color: #e0e0f0; }
        input[type="text"]:focus, input[type="number"]:focus { outline: none; border-color: #4a6cf7; }
        input[type="range"] { width: 100%; margin: 4px 0; accent-color: #4a6cf7; }
        .range-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #667; }
        .btn { width: 100%; padding: 10px; background: #4a6cf7; color: white; border: none;
               border-radius: 8px; font-size: 1rem; cursor: pointer; font-weight: 600;
               transition: background 0.2s; }
        .btn:hover { background: #3a5ce5; }
        .btn:disabled { background: #333; color: #666; cursor: not-allowed; }
        .btn-download { background: #10a37f; }
        .btn-download:hover { background: #0d8c6b; }
        .checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
        .checkbox-row input { width: auto; }
        .chart { width: 100%; height: 500px; }
        .chart-sm { height: 300px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
        .metric { background: #12122a; border-radius: 8px; padding: 12px; text-align: center;
                  border: 1px solid #2a2a4a; }
        .metric .val { font-size: 1.3rem; font-weight: 700; color: #6d8cff; }
        .metric .lbl { font-size: 0.75rem; color: #667; margin-top: 2px; }
        .loading { display: none; text-align: center; padding: 40px; }
        .loading.active { display: block; }
        .spinner { border: 4px solid #2a2a4a; border-top: 4px solid #4a6cf7; border-radius: 50%;
                   width: 40px; height: 40px; animation: spin 0.8s linear infinite;
                   margin: 0 auto 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #2a0a0a; color: #ff6b6b; padding: 12px; border-radius: 8px;
                 margin: 10px 0; border: 1px solid #4a1a1a; }
        .tabs { display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
        .tab { padding: 8px 16px; border-radius: 8px 8px 0 0; cursor: pointer; font-size: 0.85rem;
               background: #1a1a2e; color: #667; border: 1px solid #2a2a4a; border-bottom: none; }
        .tab.active { background: #1a1a2e; color: #6d8cff; font-weight: 600; border-color: #4a6cf7; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #2a2a4a; }
        th { background: #12122a; font-weight: 600; color: #8899cc; }
        .seg-text { max-height: 150px; overflow-y: auto; background: #12122a; padding: 8px;
                    border-radius: 6px; font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap;
                    color: #c0c0d0; }
        .time-badge { background: #2a3a6a; color: #8ab4ff; padding: 2px 8px; border-radius: 12px;
                      font-size: 0.75rem; }
        .progress-bar { width: 100%; background: #2a2a4a; border-radius: 8px; height: 6px;
                        margin: 10px 0; overflow: hidden; }
        .progress-bar .fill { height: 100%; background: #4a6cf7; width: 0%;
                             transition: width 0.5s; border-radius: 8px; }
        .progress-text { font-size: 0.8rem; color: #667; text-align: center; margin: 5px 0; }
        .step-indicator { display: flex; gap: 8px; margin: 10px 0; flex-wrap: wrap; }
        .step { flex: 1; min-width: 80px; text-align: center; padding: 6px 4px; border-radius: 6px;
                font-size: 0.7rem; background: #12122a; color: #667; border: 1px solid #2a2a4a; }
        .step.active { background: #1a2a4a; color: #6d8cff; border-color: #4a6cf7; }
        .step.done { background: #0a2a1a; color: #4caf50; border-color: #2a6a3a; }
        @media (max-width: 900px) { .layout { flex-direction: column; } .sidebar { width: 100%; } }
        .video-header { display: flex; align-items: center; gap: 12px; padding: 8px 0; }
        .video-header .icon { font-size: 2rem; }
        .video-header .info { flex: 1; }
        .video-header .info .title { font-size: 1.1rem; font-weight: 600; }
        .video-header .info .sub { font-size: 0.8rem; color: #667; }
        a { color: #6d8cff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
<div class="container">
    <h1>🎬 视频语义分析器 <span class="badge">Centering</span></h1>
    <div class="layout">
        <div class="sidebar">
            <div class="card">
                <h3>▶️ YouTube 视频</h3>
                <label>YouTube URL</label>
                <input type="text" id="videoUrl"
                       placeholder="https://www.youtube.com/watch?v=..."
                       value="{{ request.args.get('url', '') }}">
                <label>自定义标题（可选）</label>
                <input type="text" id="videoTitle" placeholder="留空自动获取">
                <button class="btn" onclick="analyzeVideo()" style="margin-top:12px;" id="analyzeBtn">
                    🚀 开始分析
                </button>
                <div id="progress" style="display:none;margin-top:12px;">
                    <div class="step-indicator" id="steps">
                        <div class="step" id="s1">⬇️ 下载</div>
                        <div class="step" id="s2">🎤 转录</div>
                        <div class="step" id="s3">🧠 编码</div>
                        <div class="step" id="s4">🔄 EMA</div>
                        <div class="step" id="s5">📊 分析</div>
                    </div>
                    <div class="progress-bar"><div class="fill" id="progressFill"></div></div>
                    <div class="progress-text" id="progressText">准备中...</div>
                </div>
            </div>

            <div class="card" id="paramsCard">
                <h3>🔄 EMA 平滑</h3>
                <label>EMA α (0=不平滑, 1=极端)</label>
                <input type="range" id="emaAlpha" min="0" max="0.95" step="0.05" value="0.5"
                       oninput="document.getElementById('alphaVal').textContent=this.value">
                <div class="range-label"><span>0</span><span id="alphaVal">0.5</span><span>0.95</span></div>
                <label>EMA 窗口大小 (0=全文, >0=局部)</label>
                <input type="number" id="emaWindow" value="5" min="0" max="200" step="1">

                <h3 style="margin-top:16px;">🎯 PHATE</h3>
                <label>KNN 邻居数</label>
                <input type="number" id="phateKnn" value="5" min="2" max="30">
                <label>距离度量</label>
                <select id="phateDist">
                    <option value="cosine">cosine</option>
                    <option value="euclidean">euclidean</option>
                    <option value="manhattan">manhattan</option>
                </select>

                <h3 style="margin-top:16px;">🔗 聚类</h3>
                <label>Leiden 分辨率</label>
                <input type="range" id="leidenRes" min="0.1" max="3.0" step="0.1" value="1.0"
                       oninput="document.getElementById('resVal').textContent=this.value">
                <div class="range-label"><span>0.1</span><span id="resVal">1.0</span><span>3.0</span></div>

                <h3 style="margin-top:16px;">✂️ PELT 分段</h3>
                <label>惩罚系数 (越小段越多)</label>
                <input type="range" id="peltPenalty" min="0.1" max="20.0" step="0.1" value="5.0"
                       oninput="document.getElementById('peltVal').textContent=this.value">
                <div class="range-label"><span>0.1</span><span id="peltVal">5.0</span><span>20.0</span></div>

                <div class="card" style="margin-top:12px;background:#12122a;border-left:3px solid #4a6cf7;padding:10px;">
                    <strong style="font-size:0.85rem;color:#6d8cff;">✨ 策略 B：零均值中心化已启用</strong>
                    <div style="font-size:0.75rem;color:#667;margin-top:4px;line-height:1.5;">
                        EMA 平滑后自动执行全局中心化，<br>消除各向异性偏置，放大局部突变信号。
                    </div>
                </div>
            </div>
        </div>

        <div class="main">
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <div>⏳ 处理中... (下载 → 转录 → 编码 → 分析)</div>
            </div>
            <div id="error" class="error" style="display:none;"></div>
            <div id="results" style="display:none;">
                <div class="video-header" id="videoHeader" style="display:none;">
                    <div class="icon">🎬</div>
                    <div class="info">
                        <div class="title" id="resultVideoTitle"></div>
                        <div class="sub" id="resultVideoSub"></div>
                    </div>
                </div>

                <div class="tabs" id="tabs">
                    <div class="tab active" onclick="switchTab('tab-overview',this)">📊 概览</div>
                    <div class="tab" onclick="switchTab('tab-coherence',this)">🔬 EMA 效果</div>
                    <div class="tab" onclick="switchTab('tab-phate',this)">🌐 3D 轨迹</div>
                    <div class="tab" onclick="switchTab('tab-segments',this)">📋 分段</div>
                    <div class="tab" onclick="switchTab('tab-log',this)">📝 日志</div>
                </div>

                <div id="tab-overview" class="tab-content active">
                    <div class="card"><div class="metric-grid" id="metrics"></div></div>
                    <div class="card"><div id="clusterChart" class="chart-sm"></div></div>
                </div>
                <div id="tab-coherence" class="tab-content">
                    <div class="card"><div id="coherenceChart" class="chart-sm"></div></div>
                    <div class="card"><div id="coherenceTable"></div></div>
                </div>
                <div id="tab-phate" class="tab-content">
                    <div class="card"><div id="phateChart" class="chart"></div></div>
                    <div class="card"><div class="metric-grid" id="trajectoryMetrics"></div></div>
                </div>
                <div id="tab-segments" class="tab-content">
                    <div class="card"><div id="segmentsTable"></div></div>
                    <div class="card"><div id="segmentsText"></div></div>
                </div>
                <div id="tab-log" class="tab-content">
                    <div class="card"><pre id="logContent" style="max-height:400px;overflow:auto;font-size:0.8rem;background:#12122a;padding:12px;border-radius:8px;color:#8ab4ff;"></pre></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
let currentResult = null;
let logBuffer = [];

function log(msg) {
    logBuffer.push(msg);
    document.getElementById('logContent').textContent = logBuffer.join('\n');
}

function switchTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    el.classList.add('active');
    setTimeout(() => {
        document.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
    }, 100);
}

function showError(msg) {
    document.getElementById('error').textContent = msg;
    document.getElementById('error').style.display = 'block';
}

function hideError() { document.getElementById('error').style.display = 'none'; }

function setStep(n, status) {
    const el = document.getElementById('s' + n);
    if (!el) return;
    el.className = 'step ' + status;
}

async function analyzeVideo() {
    const url = document.getElementById('videoUrl').value.trim();
    if (!url) { showError('请输入 YouTube URL'); return; }

    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('analyzeBtn').textContent = '⏳ 处理中...';
    document.getElementById('progress').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    hideError();
    logBuffer = [];
    document.getElementById('logContent').textContent = '';

    // 重置步骤
    for (let i = 1; i <= 5; i++) setStep(i, '');
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressText').textContent = '正在下载音频...';
    setStep(1, 'active');

    try {
        const resp = await fetch('/analyze_video', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                url: url,
                title: document.getElementById('videoTitle').value.trim() || undefined,
                ema_alpha: parseFloat(document.getElementById('emaAlpha').value),
                ema_window_size: parseInt(document.getElementById('emaWindow').value),
                phate_knn: parseInt(document.getElementById('phateKnn').value),
                phate_knn_dist: document.getElementById('phateDist').value,
                leiden_resolution: parseFloat(document.getElementById('leidenRes').value),
                pelt_penalty_multiplier: parseFloat(document.getElementById('peltPenalty').value),
            })
        });
        const data = await resp.json();
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').textContent = '🚀 开始分析';

        if (data.error) {
            showError(data.error);
            document.getElementById('progress').style.display = 'none';
            setStep(1, '');
            return;
        }

        currentResult = data;
        renderResults(data);

        // 全部完成
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = '✅ 分析完成！';
        for (let i = 1; i <= 5; i++) setStep(i, 'done');

        setTimeout(() => { document.getElementById('progress').style.display = 'none'; }, 2000);

    } catch(e) {
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').textContent = '🚀 开始分析';
        showError('分析失败: ' + e.message);
        setStep(5, '');
        document.getElementById('progressText').textContent = '❌ 失败';
    }
}

function renderResults(d) {
    document.getElementById('results').style.display = 'block';

    // ── 视频信息 ──
    document.getElementById('videoHeader').style.display = 'flex';
    document.getElementById('resultVideoTitle').textContent = d.video_title || '视频分析';
    document.getElementById('resultVideoSub').textContent =
        `${d.n_chunks} 个语义块 · ${d.embedding_shape[1]} 维 · Centering ✓`;

    // ── 概览指标 ──
    let metricsHtml = `
        <div class="metric"><div class="val">${d.n_chunks}</div><div class="lbl">语义块</div></div>
        <div class="metric"><div class="val">${d.ema_alpha}</div><div class="lbl">EMA α</div></div>
        <div class="metric"><div class="val">${d.ema_window_size}</div><div class="lbl">窗口</div></div>
        <div class="metric"><div class="val">${d.n_clusters}</div><div class="lbl">聚类簇</div></div>
        <div class="metric"><div class="val">${d.n_segments}</div><div class="lbl">语义段</div></div>
        <div class="metric"><div class="val">${d.coherence_before.toFixed(4)}</div><div class="lbl">平滑前连贯性</div></div>
        <div class="metric"><div class="val">${d.coherence_after.toFixed(4)}</div><div class="lbl">平滑后连贯性</div></div>
        <div class="metric"><div class="val">${(d.coherence_after - d.coherence_before) > 0 ? '+' : ''}${(d.coherence_after - d.coherence_before).toFixed(4)}</div><div class="lbl">Δ 变化</div></div>
    `;
    document.getElementById('metrics').innerHTML = metricsHtml;

    // ── 簇分布 ──
    if (d.cluster_chart_json) {
        Plotly.newPlot('clusterChart', JSON.parse(d.cluster_chart_json).data,
                       JSON.parse(d.cluster_chart_json).layout || {}, {responsive: true});
    }

    // ── 连贯性 ──
    if (d.coherence_chart_json) {
        Plotly.newPlot('coherenceChart', JSON.parse(d.coherence_chart_json).data,
                       JSON.parse(d.coherence_chart_json).layout || {}, {responsive: true});
    }
    if (d.coherence_samples) {
        let tbl = '<table><tr><th>位置</th><th>文本(前)</th><th>原始</th><th>EMA</th></tr>';
        d.coherence_samples.forEach(s => {
            tbl += `<tr><td>${s.pos}</td><td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${s.text}</td><td>${s.orig}</td><td>${s.smooth}</td></tr>`;
        });
        tbl += '</table>';
        document.getElementById('coherenceTable').innerHTML = tbl;
    }

    // ── PHATE 3D ──
    if (d.phate_chart_json) {
        Plotly.newPlot('phateChart', JSON.parse(d.phate_chart_json).data,
                       JSON.parse(d.phate_chart_json).layout || {}, {responsive: true});
    }

    // ── 轨迹指标 ──
    if (d.trajectory) {
        let trHtml = `
            <div class="metric"><div class="val">${d.trajectory.total_dist}</div><div class="lbl">轨迹总长</div></div>
            <div class="metric"><div class="val">${d.trajectory.end_dist}</div><div class="lbl">端点距离</div></div>
            <div class="metric"><div class="val">${d.trajectory.ratio}</div><div class="lbl">弯曲比</div></div>
        `;
        document.getElementById('trajectoryMetrics').innerHTML = trHtml;
    }

    // ── 分段表 ──
    if (d.segments) {
        let tbl = '<table><tr><th>#</th><th>区间</th><th>时间</th><th>长度</th><th>主题</th><th>预览</th></tr>';
        d.segments.forEach(s => {
            const timeStr = s.time_range || '';
            tbl += `<tr><td>${s.id}</td><td>${s.range}</td><td><span class="time-badge">${timeStr}</span></td><td>${s.len}</td><td>T${s.topic}</td><td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${s.preview}</td></tr>`;
        });
        tbl += '</table>';
        document.getElementById('segmentsTable').innerHTML = tbl;
    }

    // ── 日志 ──
    log(`✅ 分析完成: ${d.video_title}`);
    log(`   语义块: ${d.n_chunks} | 簇: ${d.n_clusters} | 段: ${d.n_segments}`);
    log(`   连贯性: ${d.coherence_before} → ${d.coherence_after} (Δ=${(d.coherence_after - d.coherence_before).toFixed(4)})`);
}

// 按 Enter 触发分析
document.getElementById('videoUrl').addEventListener('keydown', e => { if (e.key === 'Enter') analyzeVideo(); });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    """执行完整视频分析管线（带内存缓存）。"""
    import hashlib

    data = request.get_json()
    url = data.get("url", "").strip()
    title = data.get("title", "").strip() or None

    if not url:
        return jsonify({"error": "请提供 YouTube URL"})

    # 配置
    cfg = Config()
    cfg.ema_alpha = float(data.get("ema_alpha", 0.5))
    cfg.ema_window_size = int(data.get("ema_window_size", 5))
    cfg.phate_knn = int(data.get("phate_knn", 5))
    cfg.phate_knn_dist = str(data.get("phate_knn_dist", "cosine"))
    cfg.mds_method = str(data.get("mds_method", "metric"))
    cfg.pelt_penalty_multiplier = float(data.get("pelt_penalty_multiplier", 5.0))
    cfg.leiden_resolution = float(data.get("leiden_resolution", 1.0))

    if title:
        cfg.doc_title = title

    try:
        key = hashlib.md5(f"{url}|{json.dumps(data, sort_keys=True)}".encode()).hexdigest()
        with _analysis_cache_lock:
            if key in _analysis_cache:
                logger.info("⚡ 命中分析结果缓存 (key=%s)", key[:8])
                return jsonify(_analysis_cache[key])

        result = _run_video_pipeline(url, cfg)

        with _analysis_cache_lock:
            _analysis_cache[key] = result
            if len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
                oldest = next(iter(_analysis_cache))
                del _analysis_cache[oldest]

        return jsonify(result)
    except Exception as e:
        logger.exception("视频分析处理失败")
        return jsonify({"error": f"分析失败: {str(e)}"})


if __name__ == "__main__":
    print("=" * 50)
    print("🎬 视频语义分析器 - Flask UI (零均值中心化增强版)")
    print("=" * 50)
    print(f"🔗 打开浏览器访问: http://127.0.0.1:8083")
    print(f"📺 粘贴 YouTube URL 即可开始分析")
    print(f"✨ 流程: 下载音频 → Whisper 转录 → BGE-M3 编码")
    print(f"     → 滑动窗口 EMA → 零均值中心化 → PHATE → 聚类 → PELT")
    print(f"⚡ 性能优化: 内存缓存已启用 | 线程池 active")
    print("=" * 50)
    app.run(debug=False, threaded=True, host="127.0.0.1", port=8083)
