"""
Flask UI for text_baai_ema.py — 零均值中心化 (Centering) 增强版
================================================================
在原有 EMA + PHATE 降维管线基础上，引入策略 B（零均值中心化 Centering）。
在 EMA 平滑之后、进入下游任务（PHATE 降维、Leiden/HDBSCAN 聚类、PELT 分段）
之前，对高维 Embedding 矩阵进行全局中心化消除。
这能有效移去大模型（BGE-M3）因全局背景主题产生的各向异性偏置（Anisotropy Bias），
且完全不破坏宏观逻辑跳跃的绝对方差与物理步长。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Tuple

import requests  # 用于 DeepSeek API 调用

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template, request
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
    doc_title: str = "document"
    sbert_model_name: str = "BAAI/bge-m3"
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
    ema_alpha: float = 0.5
    ema_bidirectional: bool = True  # 保留兼容
    ema_window_size: int = 0  # 0 = 不平滑，>0 = 滑动窗口大小（自动取奇数）
    phate_n_components: int = 3
    phate_knn: int = 5
    phate_knn_dist: str = "cosine"
    mds_method: str = "metric"
    pelt_penalty_multiplier: float = 5.0
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 5

# ── 文本处理 ──
def read_text_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"不支持的文件格式: {ext}")

def split_text_into_segments(text: str) -> List[str]:
    t = text.strip()
    t = re.sub(r'([。！？])(?!["」』》\）\)】\s]*[」』》\）\)】])', r"\1\n", t)
    t = re.sub(r"([.!?])\s+(?=[A-Z\"])", r"\1\n", t)
    t = re.sub(r"\n\s*\n", "\n", t)
    sentences = [s.strip() for s in t.split("\n") if s.strip()]
    if len(sentences) < 3:
        sentences = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return sentences

def load_or_embed(chunks: List[str], output_dir: str, model_name: str, device: str) -> torch.Tensor:
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

# ── 核心分析器 ──
class EMAnalyzer:
    def __init__(self, embeddings: torch.Tensor, cfg: Config):
        self.embeddings_gpu = embeddings
        self.embeddings_cpu = embeddings.cpu().numpy()
        self.embeddings_original = embeddings.clone()  # 保存原始值，供 PELT 计算稳定基值
        self.cfg = cfg
        self.results: Dict[str, Any] = {}

    def run_ema(self) -> np.ndarray:
        """滑动窗口 EMA + 策略 B 全局中心化。

        对每个位置 i，取以 i 为中心的奇数大小窗口 [i-k, i+k]，
        窗口内各 embedding 的权重按到中心的距离指数衰减：w(j) = α^|j|，
        然后做加权平均。步长 = 1，遍历所有位置。

        边界处窗口自动截断并重新归一化权重。

        在返回之前，执行策略 B（零均值中心化）：
        减去全篇均值，将全局背景偏置重置为空间原点，
        使余弦相似度天然向皮尔逊相关系数演变，放大局部突变信号。
        """
        emb = self.embeddings_gpu          # (n, d)
        n = emb.shape[0]
        alpha = self.cfg.ema_alpha
        window = self.cfg.ema_window_size  # 滑动窗口大小（奇数）

        # window <= 0 或 n 太小 → 不平滑，但依然进行中心化消除偏置
        if window <= 0 or n < 2:
            self.embeddings_cpu = emb.detach().cpu().numpy()
            # 策略 B：消除全局背景词偏置，但不改变空间拉伸比例
            self.embeddings_cpu = self.embeddings_cpu - np.mean(self.embeddings_cpu, axis=0)
            return self.embeddings_cpu

        # 确保窗口为奇数
        if window % 2 == 0:
            window += 1
        k = window // 2                     # 单侧半径

        # 预计算权重核：(1-α)^|j|  for  j = -k ... +k
        # α→1：邻居权重极小 → 平滑轻（几乎保留原样）
        # α→0：邻居权重接近 1 → 平滑重（和邻居融合）
        j = torch.arange(-k, k + 1, device=emb.device)
        weights = (1 - alpha) ** torch.abs(j).float()  # (W,)

        # ── 向量化滑动窗口（替换逐句 Python for-loop）──
        # 手动 replicate padding（F.pad replicate 不支持 2D 张量）
        emb_pad = torch.cat([emb[:1].expand(k, -1), emb, emb[-1:].expand(k, -1)], dim=0)
        windows = emb_pad.unfold(0, window, 1).permute(0, 2, 1)       # (n, W, d)

        # 逐位置有效性 mask：标记窗口内哪些位置对应真实 embedding（非 padding）
        j_idx = torch.arange(window, device=emb.device).view(1, window)   # (1, W)
        i_idx = torch.arange(n, device=emb.device).view(n, 1)             # (n, 1)
        pos = i_idx + (j_idx - k)                                          # (n, W) — 原始索引
        valid_mask = ((pos >= 0) & (pos < n)).float()                     # (n, W)

        # 逐位置归一化权重（边界处自动截断并重新归一化）
        norm_weights = weights.view(1, window) * valid_mask               # (n, W)
        norm_weights = norm_weights / norm_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)

        out = (windows * norm_weights.unsqueeze(-1)).sum(dim=1)           # (n, d)

        self.embeddings_gpu = out
        self.embeddings_cpu = out.detach().cpu().numpy()

        # ── 策略 B 核心切入点 ──
        # 减去全篇均值，使余弦相似度天然向皮尔逊相关系数演变，放大局部突变信号
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
                    if a == b or (a, b) in seen: continue
                    seen.add((a, b))
                    edges.append((a, b)); weights.append(float(sv))
            g = _ig.Graph(n=n, edges=edges, directed=False)
            part = _la.find_partition(g, _la.RBConfigurationVertexPartition, weights=weights,
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
        # PELT 在原始 embedding 上运行：EMA 平滑会模糊边界，削弱变点检测信号
        emb = self.embeddings_original.detach().cpu().numpy()
        n = len(emb)
        clusters = self.results.get("clusters", np.zeros(n, dtype=int))
        distances = np.linalg.norm(emb[1:] - emb[:-1], axis=1)
        median_dist = float(np.median(distances))
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
                    final_ts.append(ct); ct = topics[i]
            final_bs.append(boundaries[-1]); final_ts.append(ct)
        else:
            final_bs, final_ts = boundaries, topics
        seg = {"boundaries": final_bs, "topics": final_ts, "n_segments": len(final_bs) - 1, "penalty": penalty}
        self.results["segments"] = seg
        return seg

    @staticmethod
    def get_coherence(emb: torch.Tensor) -> np.ndarray:
        norm = F.normalize(emb, p=2, dim=1)
        return (norm[:-1] * norm[1:]).sum(dim=1).cpu().numpy()

# ── DeepSeek API 摘要 ──

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-6a90467e66aa44ab9991f9e2c26866f3"

def summarize_segments_with_deepseek_stream(
    segment_texts: List[List[str]],
    model: str = "deepseek-v4-flash",
):
    """使用 DeepSeek API 流式对每个语义分段生成摘要。

    全部分段合并为一个 prompt 发送，让模型感知上下文。
    以生成器方式逐 token 产出内容。

    Yields:
        str: SSE 格式的数据行。
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    # 构建请求内容：列出所有分段
    segments_text = ""
    for i, seg_sentences in enumerate(segment_texts):
        combined = "\n".join(seg_sentences)
        segments_text += f"【分段 {i+1}】\n{combined}\n\n"

    prompt = f"""下文共有 {len(segment_texts)} 个语义分段。请逐一为每个分段用一两句话概括核心内容。

{segments_text}

**必须输出全部 {len(segment_texts)} 个分段，每个分段一行，不允许合并、省略或总结。格式如下：**
分段 1: <摘要>
分段 2: <摘要>
...
分段 {len(segment_texts)}: <摘要>"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个专业的文档分析助手，擅长概括和总结文本内容。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "stream": True,
    }

    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True)
        resp.raise_for_status()
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'请求失败: {e}'})}\n\n"
        return

    for line_str in resp.iter_lines(decode_unicode=True):
        if not line_str:
            continue
        if line_str.startswith("data: "):
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': delta})}\n\n"
            except json.JSONDecodeError:
                continue

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── 性能优化：内存缓存 + 线程池 ──

# 分析结果缓存：key = hash(file_path + params_json), value = result dict
_analysis_cache: Dict[str, Any] = {}
_analysis_cache_lock = threading.Lock()
_ANALYSIS_CACHE_MAX = 20  # 最多缓存 20 份结果

def _make_cache_key(path: str, title: str, cfg: Config) -> str:
    """生成分析结果的缓存键。"""
    raw = f"{path}|{title}|{cfg.ema_alpha}|{cfg.ema_bidirectional}|{cfg.ema_window_size}|{cfg.phate_knn}|{cfg.phate_knn_dist}|{cfg.mds_method}|{cfg.pelt_penalty_multiplier}|{cfg.leiden_resolution}|{cfg.hdbscan_min_cluster_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cached_analyze(path: str, title: str, cfg: Config) -> Dict[str, Any]:
    """带缓存的分析管线封装。"""
    key = _make_cache_key(path, title, cfg)

    with _analysis_cache_lock:
        if key in _analysis_cache:
            logger.info("⚡ 命中分析结果缓存 (key=%s)", key[:8])
            return _analysis_cache[key]

    result = _run_analysis_pipeline(path, title, cfg)

    with _analysis_cache_lock:
        _analysis_cache[key] = result
        # 限制缓存大小
        if len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
            oldest = next(iter(_analysis_cache))
            del _analysis_cache[oldest]
            logger.info("缓存已达上限，淘汰最旧条目: %s", oldest[:8])

    return result


# 线程池在 _run_analysis_pipeline 内按需创建，避免模块级资源常驻


# ── Flask App ──
app = Flask(__name__)



def _export_phate_html(phate_fig: go.Figure, out_dir: str, title: str) -> str:
    """将已构建好的 PHATE 3D Plotly 图形导出为自包含交互式 HTML 文件。

    用浏览器打开后可以旋转、缩放、悬停查看每点标注，保持完整的交互性。
    """
    html_path = os.path.join(out_dir, "phate_3d.html")

    # 调整布局：高度自适应、保留 legend、增大点尺寸
    fig = go.Figure(phate_fig)
    fig.update_traces(marker=dict(size=5))  # 点调大
    fig.update_layout(
        title=f"PHATE 3D — {title}",
        height=700,
        template="plotly_white",
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False, gridcolor="lightgray"),
            yaxis=dict(showbackground=False, gridcolor="lightgray"),
            zaxis=dict(showbackground=False, gridcolor="lightgray"),
        ),
        legend=dict(font=dict(size=10)),
    )

    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )
    logger.info("📦 PHATE 3D 交互图已导出 -> %s", html_path)
    return html_path


def _setup_chinese_font() -> None:
    """检测并配置 matplotlib 中文字体，确保图表中文正常显示。

    按优先级尝试 macOS / Windows / Linux 常见中文字体，
    同时修复 CJK 字体下负号显示为方框的问题。
    仅在首次调用时生效；重复调用无副作用（幂等）。
    """
    # 候选字体列表：按系统 + 通用性排序
    _CJK_CANDIDATES = [
        # macOS
        "PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
        "Apple LiGothic", "Arial Unicode MS",
        # Windows
        "Microsoft YaHei", "SimHei", "SimSun", "FangSong", "KaiTi",
        # Linux
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC",
        "Noto Sans SC", "Source Han Sans SC", "Droid Sans Fallback",
    ]

    # 检测系统已安装的字体
    import matplotlib.font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}

    chosen = None
    for name in _CJK_CANDIDATES:
        if name in installed:
            chosen = name
            break

    if chosen:
        plt.rcParams["font.family"] = [chosen, "sans-serif"]
        logger.info("🔤 matplotlib 中文字体: %s", chosen)
    else:
        logger.warning("⚠️ 未检测到中文字体，图表中文可能显示为方框。可用字体: %s",
                      sorted(installed)[:20])

    # 修复 CJK 字体下负号/减号显示异常
    plt.rcParams["axes.unicode_minus"] = False


def _export_pelt_plot(
    embeddings: np.ndarray,
    boundaries: List[int],
    penalty: float,
    doc_dir: str,
    title: str,
    chunks: Optional[List[str]] = None,
) -> str:
    """将 PELT 变点检测结果导出为 matplotlib 静态 PNG 图片。

    图片包含：
    - 上子图：逐句嵌入的 L2 相邻距离（平滑后），叠加 PELT 分割竖线
    - 下子图（可选）：惩罚值阈值参考线

    参数
    ----------
    embeddings : np.ndarray, shape (n, d)
        经过 EMA + 中心化后的高维嵌入矩阵。
    boundaries : list of int
        PELT 检测到的分割边界索引列表（含首尾 0 和 n）。
    penalty : float
        本次 PELT 使用的惩罚系数值。
    doc_dir : str
        输出目录。
    title : str
        文档标题，用于图片标题。
    chunks : list of str, optional
        句子文本列表，用于在分割点标注文本预览。
    """
    # ── 中文字体适配 ──
    _setup_chinese_font()

    n = embeddings.shape[0]

    # ── 计算逐句嵌入的 L2 距离（变点检测依赖的信号） ──
    distances = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)

    # ── 准备图表 ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   constrained_layout=True)

    # ── 上子图：L2 距离信号 + 分割线 ──
    x_positions = np.arange(1, n)  # 距离信号对应 1..n-1 的间隔位置
    ax1.plot(x_positions, distances, color="#4a6cf7", linewidth=0.8, alpha=0.85,
             label="平滑后 L2 嵌入距离")
    ax1.fill_between(x_positions, 0, distances, color="#4a6cf7", alpha=0.08)

    # 中位数距离参考线
    median_dist = float(np.median(distances))
    ax1.axhline(y=median_dist, color="orange", linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"中位数距离 = {median_dist:.4f}")

    # PELT 分割竖线（排除首位的 0 和 n）
    segment_colors = plt.cm.tab10(np.linspace(0, 1, max(len(boundaries) - 1, 1)))
    for i, b in enumerate(boundaries):
        if b <= 0 or b >= n:
            continue
        color = segment_colors[min(i, len(segment_colors) - 1)]
        ax1.axvline(x=b, color=color, linestyle="-", linewidth=2.0, alpha=0.85)
        # 标注分割点编号
        ax1.annotate(
            f"段{i}",
            xy=(b, ax1.get_ylim()[1] * 0.92 if ax1.get_ylim()[1] > 0 else 1),
            xytext=(b + 0.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 1.05),
            fontsize=7, color=color, fontweight="bold",
            rotation=90, va="top", ha="left",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
        )

    ax1.set_xlabel("句子位置", fontsize=11)
    ax1.set_ylabel("嵌入 L2 距离", fontsize=11)
    ax1.set_title(f"PELT 变点检测 — {title}\n"
                  f"分段数={len(boundaries)-1}  |  惩罚系数={penalty:.2f}",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(0, n)

    # ── 下子图：分割区间色块展示 ──
    for i in range(len(boundaries) - 1):
        left = int(boundaries[i])
        right = int(boundaries[i + 1])
        color = segment_colors[i % len(segment_colors)]
        ax2.axvspan(left, right, alpha=0.18, color=color)
        mid = (left + right) / 2
        ax2.text(mid, 0.5, f"段{i+1}", ha="center", va="center",
                 fontsize=7, color=color, fontweight="bold", alpha=0.7,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=color, alpha=0.6))

    ax2.set_xlabel("句子位置", fontsize=11)
    ax2.set_ylabel("分段区间", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlim(0, n)
    ax2.grid(True, axis="x", alpha=0.3, linestyle="--")

    # ── 保存 ──
    png_path = os.path.join(doc_dir, "pelt_changepoints.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("📊 PELT 变点检测图已导出 -> %s", png_path)
    return png_path


# ── Flask Routes ──

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/load", methods=["POST"])
def load_doc():
    """加载文档，缓存 chunks 和 embeddings。"""
    data = request.get_json()
    path = data.get("path", "").strip()
    title = data.get("title", "document")

    if not path or not os.path.isfile(path):
        return jsonify({"error": f"文件不存在: {path}"})

    try:
        text = read_text_file(path)
    except Exception as e:
        return jsonify({"error": f"读取文件失败: {e}"})

    chunks = split_text_into_segments(text)

    # 构建缓存目录
    doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), title.replace(" ", "_"))
    os.makedirs(doc_dir, exist_ok=True)
    with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # 尝试加载或计算 embeddings
    cfg = Config()
    try:
        emb = load_or_embed(chunks, doc_dir, cfg.sbert_model_name, cfg.device)
        has_emb = True
    except Exception as e:
        logger.warning("Embedding 计算失败: %s", e)
        has_emb = False

    return jsonify({"n_chunks": len(chunks), "has_embeddings": has_emb, "doc_dir": doc_dir})


def _run_analysis_pipeline(path: str, title: str, cfg: Config) -> Dict[str, Any]:
    """实际的完整分析管线（不含HTTP逻辑，可被缓存层复用）。"""
    log_capture: List[str] = []
    class LogHandler(logging.Handler):
        def emit(self, record):
            log_capture.append(self.format(record))
    log_handler = LogHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(log_handler)

    try:
        # 读取文本
        text = read_text_file(path)
        chunks = split_text_into_segments(text)

        # 构建缓存目录
        doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), title.replace(" ", "_"))
        os.makedirs(doc_dir, exist_ok=True)
        with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        orig_emb = load_or_embed(chunks, doc_dir, cfg.sbert_model_name, cfg.device)

        # 原始连贯性
        orig_sim = EMAnalyzer.get_coherence(orig_emb)
        coh_before = float(orig_sim.mean())

        # EMA（内部包含 Centering 策略 B）
        analyzer = EMAnalyzer(orig_emb.clone(), cfg)
        smooth_emb = analyzer.run_ema()
        smooth_sim = EMAnalyzer.get_coherence(analyzer.embeddings_gpu)
        coh_after = float(smooth_sim.mean())

        # PHATE 和聚类互相独立，并行执行以减少端到端延迟
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_phate = pool.submit(analyzer.run_phate)
            fut_cluster = pool.submit(analyzer.run_clustering)
            phate_coords = fut_phate.result()
            labels, outliers = fut_cluster.result()

        n_clusters = len(set(labels) - {-1}) if labels is not None and len(labels) > 0 else 0

        # PELT（依赖聚类结果，顺序执行）
        seg = analyzer.run_pelt()

        # ── 构建结果 JSON ──
        result = {
            "n_chunks": len(chunks),
            "ema_alpha": cfg.ema_alpha,
            "ema_direction": "双向" if cfg.ema_bidirectional else "单向",
            "n_clusters": n_clusters,
            "n_segments": seg.get("n_segments", 0),
            "coherence_before": coh_before,
            "coherence_after": coh_after,
            "log": log_capture,
        }

        # 连贯性图
        coh_fig = go.Figure()
        coh_fig.add_trace(go.Scatter(y=orig_sim, mode="lines", name="原始", line=dict(color="rgba(100,100,100,0.6)")))
        coh_fig.add_trace(go.Scatter(y=smooth_sim, mode="lines", name="EMA", line=dict(color="rgba(255,50,50,0.8)")))
        coh_fig.add_trace(go.Scatter(y=smooth_sim - orig_sim, mode="lines", name="Δ",
                                     line=dict(color="green", dash="dot"), yaxis="y2"))
        coh_fig.update_layout(title="语义连贯性对比", xaxis_title="句子位置",
                              yaxis_title="余弦相似度",
                              yaxis2=dict(overlaying="y", side="right", title="Δ"),
                              template="plotly_white", hovermode="x unified",
                              margin=dict(l=40, r=40, t=40, b=40), height=300)
        result["coherence_chart_json"] = plotly.io.to_json(coh_fig)

        # 连贯性样本
        n_show = min(10, len(chunks))
        samples = []
        for i in range(n_show):
            samples.append({
                "pos": f"{i}→{i+1}",
                "text": chunks[i][:60],
                "orig": f"{orig_sim[i]:.4f}" if i < len(orig_sim) else "",
                "smooth": f"{smooth_sim[i]:.4f}" if i < len(smooth_sim) else "",
            })
        result["coherence_samples"] = samples

        # 簇分布柱状图
        if labels is not None and len(labels) > 0:
            unique, counts = np.unique(labels, return_counts=True)
            cluster_fig = go.Figure(data=[go.Bar(x=[f"T{int(u)}" if u != -1 else "噪声" for u in unique],
                                                  y=counts, marker_color="#4a6cf7")])
            cluster_fig.update_layout(title="簇分布", template="plotly_white",
                                      margin=dict(l=40, r=40, t=40, b=40), height=250)
            result["cluster_chart_json"] = plotly.io.to_json(cluster_fig)

        # PHATE 3D 图
        if phate_coords is not None:
            palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                       "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
            phate_fig = go.Figure()
            phate_fig.add_trace(go.Scatter3d(
                x=phate_coords[:, 0], y=phate_coords[:, 1], z=phate_coords[:, 2],
                mode="lines", line=dict(color="rgba(60,60,60,0.4)", width=1.5), name="轨迹"))
            bs = seg["boundaries"]
            ts = seg["topics"]
            for s_i in range(max(0, len(bs) - 1)):
                l, r = int(bs[s_i]), int(bs[s_i + 1])
                idx = np.arange(l, r, dtype=int)
                if len(idx) <= 0: continue
                idx = idx[idx < len(chunks)]
                if len(idx) <= 0: continue
                color = palette[s_i % len(palette)]
                topic = ts[s_i] if s_i < len(ts) else -1
                phate_fig.add_trace(go.Scatter3d(
                    x=phate_coords[idx, 0], y=phate_coords[idx, 1], z=phate_coords[idx, 2],
                    mode="markers", marker=dict(size=3, color=color, opacity=0.8),
                    name=f"段{s_i+1}|T{topic}",
                    text=[chunks[j][:60] for j in idx], hoverinfo="text+name"))
            b_idx = [max(0, min(int(b), len(phate_coords)-1)) for b in bs]
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
                title=f"PHATE 3D — α={cfg.ema_alpha} {'双向' if cfg.ema_bidirectional else '单向'}",
                scene=dict(aspectmode="data",
                           xaxis=dict(showbackground=False),
                           yaxis=dict(showbackground=False),
                           zaxis=dict(showbackground=False),
                           camera=dict(eye=dict(x=1.3, y=1.25, z=0.9))),
                template="plotly_white", margin=dict(l=0, r=0, t=40, b=0), height=500)
            result["phate_chart_json"] = plotly.io.to_json(phate_fig)

            # ── 导出自包含交互式 HTML（后台线程，不阻塞 HTTP 响应）──
            threading.Thread(target=_export_phate_html, args=(phate_fig, doc_dir, title), daemon=True).start()

            # 轨迹统计
            total_dist = float(np.sum(np.linalg.norm(phate_coords[1:] - phate_coords[:-1], axis=1)))
            end_dist = float(np.linalg.norm(phate_coords[0] - phate_coords[-1]))
            result["trajectory"] = {
                "total_dist": total_dist,
                "end_dist": end_dist,
                "ratio": total_dist / max(end_dist, 1e-9),
            }
            result["phate_html_file"] = os.path.join(doc_dir, "phate_3d.html")

        # 分段详情
        bs = seg["boundaries"]
        ts = seg["topics"]
        seg_list = []
        for s_i in range(len(bs) - 1):
            l, r = int(bs[s_i]), int(bs[s_i + 1])
            topic = ts[s_i] if s_i < len(ts) else -1
            l = min(l, len(chunks) - 1)
            r = min(r, len(chunks))
            seg_chunks = chunks[l:r]
            seg_list.append({
                "id": s_i + 1,
                "range": f"[{l},{r})",
                "len": r - l,
                "topic": topic,
                "preview": (seg_chunks[0][:80] + "...") if seg_chunks else "",
                "full_text": seg_chunks,
            })
        result["segments"] = seg_list

        # ── 导出 PELT 变点检测静态图（matplotlib PNG）──
        try:
            pelt_png_path = _export_pelt_plot(
                embeddings=analyzer.embeddings_original.detach().cpu().numpy(),
                boundaries=seg["boundaries"],
                penalty=seg["penalty"],
                doc_dir=doc_dir,
                title=title,
                chunks=chunks,
            )
            result["pelt_plot_file"] = pelt_png_path
        except Exception as e:
            logger.warning("PELT 绘图导出失败: %s", e)

        return result

    except Exception as e:
        logger.exception("分析失败")
        raise
    finally:
        logging.getLogger().removeHandler(log_handler)


@app.route("/analyze", methods=["POST"])
def analyze():
    """执行完整分析管线（带内存缓存）。"""
    data = request.get_json()
    path = data.get("path", "").strip()
    title = data.get("title", "document")

    if not path or not os.path.isfile(path):
        return jsonify({"error": f"文件不存在: {path}"})

    # 配置
    cfg = Config()
    cfg.ema_alpha = float(data.get("ema_alpha", 0.5))
    cfg.ema_bidirectional = bool(data.get("ema_bidirectional", True))
    cfg.ema_window_size = int(data.get("ema_window_size", 0))
    cfg.phate_knn = int(data.get("phate_knn", 5))
    cfg.phate_knn_dist = str(data.get("phate_knn_dist", "cosine"))
    cfg.mds_method = str(data.get("mds_method", "metric"))
    cfg.pelt_penalty_multiplier = float(data.get("pelt_penalty_multiplier", 5.0))
    cfg.leiden_resolution = float(data.get("leiden_resolution", 1.0))
    cfg.hdbscan_min_cluster_size = int(data.get("hdbscan_min_cluster_size", 5))

    try:
        result = _cached_analyze(path, title, cfg)
        return jsonify(result)
    except Exception as e:
        logger.exception("分析路由处理失败")
        return jsonify({"error": f"分析失败: {str(e)}"})


@app.route("/summarize", methods=["POST"])
def summarize():
    """使用 DeepSeek API 流式对每个语义分段生成摘要。"""
    data = request.get_json()
    segment_texts: List[str] = data.get("segments", [])  # 前端传过来的各段文本（每段是一个字符串）

    if not segment_texts:
        return jsonify({"error": "请先运行分析，获取分段结果"})

    # 转为 List[List[str]] 格式（每段内的句子列表）
    segments_for_api: List[List[str]] = [[seg] for seg in segment_texts]

    def generate():
        for sse_data in summarize_segments_with_deepseek_stream(segments_for_api):
            yield sse_data

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("=" * 50)
    print("📄 语义分析器 - Flask UI (零均值中心化增强版)")
    print("=" * 50)
    print(f"🔗 打开浏览器访问: http://127.0.0.1:8082")
    print(f"📂 在输入框中粘贴文本文件的绝对路径后点击「加载文档」或按 Enter")
    print(f"✨ 策略 B: EMA 平滑后自动执行零均值中心化 (Centering)")
    print(f"⚡ 性能优化: 内存缓存已启用 | 线程池 active | threaded=True")
    print("=" * 50)
    app.run(debug=False, threaded=True, host="127.0.0.1", port=8082)
