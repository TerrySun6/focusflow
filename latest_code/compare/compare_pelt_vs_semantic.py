#!/usr/bin/env python3
"""
对比评测：PELT 变点检测 vs SemanticChunker 语义分割。

在同一个 EMA 平滑后的 embedding 序列上，分别用两种方法做分段，
对比分段数量、边界一致性、段内连贯性、段间差异性等指标。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import ruptures as rpt
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 可选依赖检测 ──
_SEMANTIC_CHUNKER_AVAILABLE = False
try:
    from langchain_experimental.text_splitter import (
        SemanticChunker,
        BreakpointThresholdType,
    )
    _SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    logger.warning("langchain_experimental 未安装，SemanticChunker 不可用")


# 为了让没有装 langchain 的人也能跑，我们自己实现一个简化的 SemanticChunker
# 核心逻辑从 text_splitter.py 提取
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度。"""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-9))


def semantic_chunker_split(
    sentences: List[str],
    embeddings: np.ndarray,          # shape: (n, d)
    buffer_size: int = 1,
    threshold_percentile: float = 95,
    min_chunk_size: Optional[int] = None,
) -> List[int]:
    """简化版 SemanticChunker，直接使用已有 embedding。

    Args:
        sentences: 句子列表。
        embeddings: 对应的 embedding 数组，shape (n, d)。
        buffer_size: 上下文缓冲大小。
        threshold_percentile: 断点距离百分位阈值。
        min_chunk_size: 最小 chunk 字符数。

    Returns:
        breakpoint_indices: 分段断点索引列表（不含 0 和 n）。
    """
    n = len(sentences)
    if n < 2:
        return []

    # 1. 构建组合句（每个句子前后 buffer 个句子拼接）
    combined = []
    for i in range(n):
        parts = []
        for j in range(i - buffer_size, i + buffer_size + 1):
            if 0 <= j < n:
                parts.append(sentences[j])
        combined.append(" ".join(parts))

    # 2. 如果外面没有传入 embedding，这里就需要重新编码了
    #    但我们可以用传入的 embedding 近似替代组合句的 embedding
    #    这里用 sentences 自己的 embedding 来做简化
    _emb = embeddings

    # 3. 计算相邻组合句之间的余弦距离
    distances = []
    for i in range(n - 1):
        sim = _cosine_similarity(_emb[i].tolist(), _emb[i+1].tolist())
        distances.append(1.0 - sim)

    # 4. 计算阈值
    threshold = float(np.percentile(distances, threshold_percentile))

    # 5. 找出超过阈值的断点
    indices_above = [i for i, d in enumerate(distances) if d > threshold]

    return indices_above


# ── 自定义的 Embedding 包装器（给 SemanticChunker 用） ──
class NumpyEmbeddings:
    """把已有的 numpy embedding 包装成 langchain Embeddings 接口的假冒版。"""

    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings
        self._dim = embeddings.shape[1]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 这里返回预先计算好的 embedding
        # 但 SemanticChunker 会对组合句重新请求 embedding，这里不做真实编码
        # 所以我们返回预先计算的单句 embedding（简化处理）
        n = len(texts)
        embs = self._embeddings[:n]
        return [embs[i].tolist() for i in range(n)]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self._dim


# ═══════════════════════════════════════════════════════════════
# 核心对比逻辑
# ═══════════════════════════════════════════════════════════════


def split_text_into_sentences(text: str) -> List[str]:
    """按中英文标点分割句子。"""
    t = text.strip()
    # 中文标点
    t = re.sub(r'([。！？])(?!["」』》\）\)】\s]*[」』》\）\)】])', r"\1\n", t)
    # 英文标点
    t = re.sub(r"([.!?])\s+(?=[A-Z\"])", r"\1\n", t)
    t = re.sub(r"\n\s*\n", "\n", t)
    sentences = [s.strip() for s in t.split("\n") if s.strip()]
    if len(sentences) < 3:
        sentences = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return sentences


def run_pelt_segmentation(
    embeddings: np.ndarray,
    penalty_multiplier: float = 5.0,
    min_size: int = 3,
) -> Tuple[List[int], List[int]]:
    """PELT 变点检测。

    Args:
        embeddings: shape (n, d)
        penalty_multiplier: 惩罚系数，越大段越少
        min_size: 最小分段大小

    Returns:
        (boundaries, clusters_for_sections)
        注意：boundaries 包含 0 和 n
    """
    n = len(embeddings)
    if n < min_size * 2:
        return [0, n], [0]

    distances = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
    median_dist = float(np.median(distances))
    penalty = median_dist * penalty_multiplier

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(embeddings)
        raw_cuts = algo.predict(pen=penalty)
    except Exception:
        logger.warning("rbf 核失败，回退 l2 核")
        algo = rpt.Pelt(model="l2", min_size=min_size).fit(embeddings)
        raw_cuts = algo.predict(pen=penalty * 0.1)

    boundaries = [0] + sorted(set(raw_cuts))
    # 确保最后一个边界不超过 n
    if boundaries[-1] < n:
        boundaries.append(n)

    return boundaries, list(range(len(boundaries) - 1))


def compute_segment_coherence(
    sentences: List[str], embeddings: np.ndarray, boundaries: List[int]
) -> Dict[str, Any]:
    """计算分段质量指标。

    Args:
        sentences: 句子列表
        embeddings: embedding 数组 (n, d)
        boundaries: 分段边界 [0, ..., n]

    Returns:
        各指标字典
    """
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    # 1. 段内连贯性（同段内相邻句子的相似度均值）
    intra_sims = []
    for i in range(len(boundaries) - 1):
        l, r = boundaries[i], boundaries[i + 1]
        if r - l < 2:
            continue
        seg_norm = emb_norm[l:r]
        sims = np.sum(seg_norm[:-1] * seg_norm[1:], axis=1)
        intra_sims.append(float(np.mean(sims)) if len(sims) > 0 else 0.0)

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0

    # 2. 段间差异性（相邻段边界处的相似度均值）
    inter_sims = []
    for i in range(len(boundaries) - 2):
        r = boundaries[i + 1]  # 前一段的最后一个句子索引
        if r >= n:
            break
        sim = float(np.sum(emb_norm[r - 1] * emb_norm[r]))
        inter_sims.append(sim)

    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0

    # 3. 段内 embedding 紧致度（段内各点到段中心的平均距离）
    compactness = []
    for i in range(len(boundaries) - 1):
        l, r = boundaries[i], boundaries[i + 1]
        if r - l < 1:
            continue
        center = np.mean(emb_norm[l:r], axis=0)
        center = center / (np.linalg.norm(center) + 1e-9)
        dists = 1.0 - np.sum(emb_norm[l:r] * center, axis=1)
        compactness.append(float(np.mean(dists)))

    mean_compact = float(np.mean(compactness)) if compactness else 0.0

    # 4. 分段数量
    n_segments = len(boundaries) - 1

    # 5. 分段大小统计
    seg_sizes = [boundaries[i+1] - boundaries[i] for i in range(n_segments)]
    size_std = float(np.std(seg_sizes)) if seg_sizes else 0.0

    return {
        "n_segments": n_segments,
        "intra_coherence": mean_intra,     # 越高越好：段内句子相似
        "inter_dissimilarity": 1.0 - mean_inter,  # 越高越好：段间边界差异大
        "intra_minus_inter": mean_intra - mean_inter,  # 综合：越大越好
        "compactness": mean_compact,        # 越低越好：段内密集
        "seg_size_std": size_std,           # 越低越好：分段均匀
    }


def compute_embedding_embedding(
    sentences: List[str],
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    output_dir: str = "",
) -> np.ndarray:
    """编码句子列表（优先用 MLX，缓存 .npy，避免重复下载模型）。

    和 text_baai_ema.py 的策略一致：
    1. 如果 output_dir 里有 embeddings.npy 缓存，直接读取
    2. 优先用 MLX Embedding Models
    3. 回退到 SentenceTransformer
    """
    # 1. 尝试读取缓存
    if output_dir:
        emb_cache = os.path.join(output_dir, "embeddings.npy")
        if os.path.exists(emb_cache):
            embeddings = np.load(emb_cache)
            logger.info("已读取 embeddings 缓存: %s", emb_cache)
            # 检查缓存长度是否匹配
            if len(embeddings) == len(sentences):
                return embeddings
            logger.warning("缓存长度 %d 不匹配句子数 %d，重新计算", len(embeddings), len(sentences))

    # 2. 优先 MLX
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
        embeddings = np.array(mlx_model.encode(sentences))
        logger.info("使用 MLX Embedding Models: %s", model_name)
    except Exception as exc:
        logger.warning("MLX 不可用，回退 SentenceTransformer。原因: %s", exc)
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(sentences, convert_to_tensor=True)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

    # 3. 保存缓存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        logger.info("已保存 embeddings 缓存")

    return embeddings


def run_comparison(
    text: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    ema_alpha: float = 0.5,
    ema_bidirectional: bool = True,
    pelt_penalty: float = 5.0,
    semantic_threshold: float = 95,
    buffer_size: int = 1,
    output_dir: str = "",
) -> Dict[str, Any]:
    """运行完整对比评测。

    Returns:
        包含所有结果的字典。
    """
    # 1. 分句
    sentences = split_text_into_sentences(text)
    logger.info("句子总数: %d", len(sentences))

    if len(sentences) < 3:
        return {"error": "句子太少，无法分段", "n_sentences": len(sentences)}

    # 2. 编码（传入 output_dir 以便利用 .npy 缓存，避免重复下载）
    logger.info("开始编码 (model=%s, device=%s)...", model_name, device)
    t0 = time.time()
    embeddings = compute_embedding_embedding(sentences, model_name, device, output_dir)
    t1 = time.time()
    logger.info("编码完成: %d × %d, 耗时 %.1fs", *embeddings.shape, t1 - t0)

    # 3. EMA 平滑
    emb_t = torch.tensor(embeddings, dtype=torch.float32)
    n = emb_t.shape[0]
    alpha = ema_alpha

    fwd = torch.zeros_like(emb_t)
    fwd[0] = emb_t[0]
    for i in range(1, n):
        fwd[i] = (1 - alpha) * emb_t[i] + alpha * fwd[i - 1]

    if ema_bidirectional and n > 2:
        bwd = torch.zeros_like(emb_t)
        bwd[n - 1] = emb_t[n - 1]
        for i in range(n - 2, -1, -1):
            bwd[i] = (1 - alpha) * emb_t[i] + alpha * bwd[i + 1]
        ema_emb = (fwd + bwd) / 2.0
    else:
        ema_emb = fwd

    ema_np = ema_emb.detach().cpu().numpy()
    logger.info("EMA 平滑完成: alpha=%.2f, bidirectional=%s", alpha, ema_bidirectional)

    # ── A. PELT 分段 ──
    logger.info("--- 运行 PELT 分段 (penalty=%.1f) ---", pelt_penalty)
    pelt_boundaries, pelt_sections = run_pelt_segmentation(
        ema_np, penalty_multiplier=pelt_penalty
    )
    pelt_metrics = compute_segment_coherence(sentences, ema_np, pelt_boundaries)
    logger.info("PELT: %d 段", pelt_metrics["n_segments"])

    # ── B. SemanticChunker 分段 ──
    logger.info("--- 运行 SemanticChunker 分段 (threshold=%s, buffer=%d) ---",
                semantic_threshold, buffer_size)

    semantic_breakpoints = semantic_chunker_split(
        sentences, ema_np,
        buffer_size=buffer_size,
        threshold_percentile=semantic_threshold,
    )
    # 构造 boundaries
    semantic_boundaries = [0] + [b + 1 for b in semantic_breakpoints]
    if semantic_boundaries[-1] < n:
        semantic_boundaries.append(n)
    # 去重、排序
    semantic_boundaries = sorted(set(semantic_boundaries))
    semantic_boundaries = [b for b in semantic_boundaries if b <= n]
    if semantic_boundaries[-1] != n:
        semantic_boundaries.append(n)

    semantic_metrics = compute_segment_coherence(
        sentences, ema_np, semantic_boundaries
    )
    logger.info("SemanticChunker: %d 段", semantic_metrics["n_segments"])

    # ── C. PELT + SemanticChunker 混合（交集/并集） ──
    # 并集：两个方法都检测到的边界才保留
    pelt_breaks = set(pelt_boundaries[1:-1])
    semantic_breaks = set(semantic_boundaries[1:-1])
    intersection_breaks = sorted(pelt_breaks & semantic_breaks)
    union_breaks = sorted(pelt_breaks | semantic_breaks)

    inter_boundaries = [0] + intersection_breaks + [n]
    union_boundaries = [0] + union_breaks + [n]

    inter_metrics = compute_segment_coherence(sentences, ema_np, inter_boundaries)
    union_metrics = compute_segment_coherence(sentences, ema_np, union_boundaries)

    # ── 收集结果 ──
    results = {
        "config": {
            "n_sentences": len(sentences),
            "embedding_dim": embeddings.shape[1],
            "model": model_name,
            "ema_alpha": ema_alpha,
            "ema_bidirectional": ema_bidirectional,
            "pelt_penalty": pelt_penalty,
            "semantic_threshold": semantic_threshold,
            "semantic_buffer_size": buffer_size,
        },
        "pelt": {
            "boundaries": pelt_boundaries,
            "metrics": pelt_metrics,
        },
        "semantic_chunker": {
            "boundaries": semantic_boundaries,
            "metrics": semantic_metrics,
        },
        "intersection": {
            "boundaries": inter_boundaries,
            "metrics": inter_metrics,
        },
        "union": {
            "boundaries": union_boundaries,
            "metrics": union_metrics,
        },
        "embeddings": embeddings,
        "ema_embeddings": ema_np,
        "sentences": sentences,
    }

    return results


def print_comparison_table(results: Dict[str, Any]) -> None:
    """打印对比表格到终端。"""
    if "error" in results:
        logger.error(results["error"])
        return

    cfg = results["config"]
    print("\n" + "=" * 80)
    print("  📊 SemanticChunker vs PELT 分段对比评测")
    print("=" * 80)
    print(f"  句子数: {cfg['n_sentences']}  |  模型: {cfg['model']}")
    print(f"  EMA α={cfg['ema_alpha']}  bidirectional={cfg['ema_bidirectional']}")
    print(f"  PELT penalty={cfg['pelt_penalty']}  |  SC threshold={cfg['semantic_threshold']}%  buffer={cfg['semantic_buffer_size']}")
    print("=" * 80)

    print(f"\n  {'方法':<22} {'段数':<6} {'段内连贯性':<12} {'段间差异':<12} {'综合分':<10} {'紧致度':<10} {'均匀性':<10}")
    print(f"  {'-'*22} {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for method, key in [("PELT", "pelt"),
                         ("SemanticChunker", "semantic_chunker"),
                         ("交集 (Both)", "intersection"),
                         ("并集 (Either)", "union")]:
        m = results[key]["metrics"]
        score = m["intra_minus_inter"]
        print(f"  {method:<22} {m['n_segments']:<6} {m['intra_coherence']:<12.4f} "
              f"{m['inter_dissimilarity']:<12.4f} {score:<10.4f} "
              f"{m['compactness']:<10.4f} {m['seg_size_std']:<10.2f}")

    print("=" * 80)
    print("  指标说明:")
    print("  · 段内连贯性 ↑ = 段内句子语义相似，好")
    print("  · 段间差异 ↑   = 段边界处语义跳变大，好")
    print("  · 综合分 ↑     = 段内连贯 - 段间相似，越大越好")
    print("  · 紧致度 ↓     = 段内点到中心距离均值，越小越密集")
    print("  · 均匀性 ↓     = 分段大小标准差，越小越均匀")
    print("=" * 80)


def generate_comparison_chart(
    results: Dict[str, Any],
    output_path: str = "comparison_report.html",
) -> str:
    """生成交互式对比报告 HTML。"""
    if "error" in results:
        return ""

    sentences = results["sentences"]
    cfg = results["config"]
    n = len(sentences)

    # 构建 subplot（第2行用 polar 类型放雷达图）
    fig = make_subplots(
        rows=4, cols=1,
        specs=[
            [{"type": "xy"}],
            [{"type": "polar"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
        subplot_titles=[
            "📈 段内连贯性对比 (越高越好)",
            "📊 各方法指标雷达图",
            "🔍 分段边界可视化 (灰色竖线 = 分段点)",
            "📋 详细分段结果",
        ],
        vertical_spacing=0.08,
        row_heights=[0.25, 0.30, 0.30, 0.20],
    )

    # ── 1. 段内连贯性 + 段间差异对比 ──
    methods_names = ["PELT", "SemanticChunker", "交集", "并集"]
    methods_keys = ["pelt", "semantic_chunker", "intersection", "union"]
    colors = ["#4a6cf7", "#e74c3c", "#2ecc71", "#f39c12"]

    intra_vals = [results[k]["metrics"]["intra_coherence"] for k in methods_keys]
    inter_vals = [results[k]["metrics"]["inter_dissimilarity"] for k in methods_keys]
    score_vals = [results[k]["metrics"]["intra_minus_inter"] for k in methods_keys]

    # 柱状图：段内连贯性
    fig.add_trace(
        go.Bar(
            x=methods_names, y=intra_vals,
            name="段内连贯性",
            marker_color=colors,
            text=[f"{v:.4f}" for v in intra_vals],
            textposition="auto",
        ),
        row=1, col=1,
    )

    # ── 2. 雷达图 ──
    categories = ["段内连贯性", "段间差异", "综合分", "紧致度(反向)", "均匀性(反向)"]
    # 归一化
    all_compact = [results[k]["metrics"]["compactness"] for k in methods_keys]
    all_std = [results[k]["metrics"]["seg_size_std"] for k in methods_keys]
    max_compact = max(all_compact) if max(all_compact) > 0 else 1
    max_std = max(all_std) if max(all_std) > 0 else 1

    radar_data = []
    for k in methods_keys:
        m = results[k]["metrics"]
        radar_data.append([
            m["intra_coherence"] * 10,          # 放大到 0-10 范围
            m["inter_dissimilarity"] * 10,
            max(0, m["intra_minus_inter"] * 10),
            (1 - m["compactness"] / max_compact) * 10,  # 反向指标
            (1 - m["seg_size_std"] / max_std) * 10,
        ])

    for i, name in enumerate(methods_names):
        fig.add_trace(
            go.Scatterpolar(
                r=radar_data[i] + [radar_data[i][0]],
                theta=categories + [categories[0]],
                name=name,
                line=dict(color=colors[i], width=2),
                marker=dict(size=4),
            ),
            row=2, col=1,
        )

    fig.update_polars(radialaxis=dict(range=[0, 10], showticklabels=True))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])))

    # ── 3. 分段边界可视化（在 embedding 距离曲线上标注断点） ──
    ema = results["ema_embeddings"]
    emb_norm = ema / (np.linalg.norm(ema, axis=1, keepdims=True) + 1e-9)
    sim_seq = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)
    x = list(range(len(sim_seq)))

    fig.add_trace(
        go.Scatter(
            x=x, y=sim_seq,
            mode="lines",
            line=dict(color="gray", width=1.5),
            name="相邻句余弦相似度",
            hovertemplate="位置 %{x}<br>相似度 %{y:.4f}<extra></extra>",
        ),
        row=3, col=1,
    )

    # 在各方法的分段边界处画竖线（用 add_shape 避免 polar subplot 干扰）
    y_min = float(min(sim_seq)) - 0.1
    y_max = float(max(sim_seq)) + 0.1

    for i, (key, name, color) in enumerate(
        zip(methods_keys, methods_names, colors)
    ):
        bounds = results[key]["boundaries"]
        for b in bounds[1:-1]:  # 跳过首尾
            if 0 < b < len(sim_seq):
                fig.add_shape(
                    type="line",
                    x0=b - 0.5, y0=y_min,
                    x1=b - 0.5, y1=y_max,
                    xref="x3", yref="y3",
                    line=dict(width=1.5, dash="dash", color=color),
                    opacity=0.6,
                )
        # 添加图例占位
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"{name} 断点",
            ),
            row=3, col=1,
        )

    fig.update_yaxes(title_text="余弦相似度", row=3, col=1)
    fig.update_xaxes(title_text="句子位置", row=3, col=1)

    # ── 4. 详细分段结果表 ──
    table_data = []
    for key, name in zip(methods_keys, methods_names):
        m = results[key]["metrics"]
        bounds = results[key]["boundaries"]
        seg_details = []
        for i in range(len(bounds) - 1):
            l, r = bounds[i], bounds[i + 1]
            seg_text = " ".join(sentences[l:r])[:50] + "..."
            seg_details.append(f"[{l},{r}) {seg_text}")

        table_data.append(dict(
            method=name,
            n_seg=m["n_segments"],
            intra=f"{m['intra_coherence']:.4f}",
            inter=f"{m['inter_dissimilarity']:.4f}",
            score=f"{m['intra_minus_inter']:.4f}",
            segments="<br>".join(seg_details[:6]),
        ))

    fig.add_trace(
        go.Table(
            header=dict(
                values=["方法", "段数", "段内连贯", "段间差异", "综合分", "分段详情"],
                fill_color="paleturquoise",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=[
                    [d["method"] for d in table_data],
                    [d["n_seg"] for d in table_data],
                    [d["intra"] for d in table_data],
                    [d["inter"] for d in table_data],
                    [d["score"] for d in table_data],
                    [d["segments"] for d in table_data],
                ],
                align="left",
                font=dict(size=11),
                height=30,
            ),
        ),
        row=4, col=1,
    )

    # 整体布局
    fig.update_layout(
        title=dict(
            text=f"📊 PELT vs SemanticChunker 对比 — {cfg['n_sentences']}句, α={cfg['ema_alpha']}, penalty={cfg['pelt_penalty']}, SC_th={cfg['semantic_threshold']}%",
            font=dict(size=16),
        ),
        height=1400,
        template="plotly_white",
        showlegend=True,
        hovermode="x unified",
    )

    pio.write_html(fig, output_path, include_plotlyjs="cdn", auto_open=False)
    logger.info("对比报告已生成: %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════
# 自动化参数扫描（找最佳参数组合）
# ═══════════════════════════════════════════════════════════════

def parameter_scan(
    text: str,
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    ema_alpha: float = 0.5,
    ema_bidirectional: bool = True,
    output_dir: str = "",
) -> List[Dict[str, Any]]:
    """扫描不同参数组合，找出最佳设置。

    扫描参数：
    - PELT penalty: [1, 3, 5, 10, 20]
    - SemanticChunker threshold: [80, 90, 95, 99]
    - SemanticChunker buffer_size: [0, 1, 2]
    """
    sentences = split_text_into_sentences(text)
    if len(sentences) < 5:
        return []

    logger.info("参数扫描: %d 个句子", len(sentences))

    # 编码（一次性，传 output_dir 用缓存）
    embeddings = compute_embedding_embedding(sentences, model_name, device, output_dir)

    # EMA 平滑
    emb_t = torch.tensor(embeddings, dtype=torch.float32)
    n = emb_t.shape[0]
    fwd = torch.zeros_like(emb_t)
    fwd[0] = emb_t[0]
    for i in range(1, n):
        fwd[i] = (1 - ema_alpha) * emb_t[i] + ema_alpha * fwd[i - 1]
    if ema_bidirectional and n > 2:
        bwd = torch.zeros_like(emb_t)
        bwd[n - 1] = emb_t[n - 1]
        for i in range(n - 2, -1, -1):
            bwd[i] = (1 - ema_alpha) * emb_t[i] + ema_alpha * bwd[i + 1]
        ema_np = ((fwd + bwd) / 2.0).detach().cpu().numpy()
    else:
        ema_np = fwd.detach().cpu().numpy()

    results = []

    # PELT 扫描
    for penalty in [1, 2, 3, 5, 8, 10, 15, 20]:
        try:
            pelt_boundaries, _ = run_pelt_segmentation(
                ema_np, penalty_multiplier=penalty
            )
            metrics = compute_segment_coherence(sentences, ema_np, pelt_boundaries)
            results.append({
                "method": "PELT",
                "params": {"penalty": penalty},
                "metrics": metrics,
            })
        except Exception as e:
            logger.warning("PELT penalty=%d 失败: %s", penalty, e)

    # SemanticChunker 扫描
    for threshold in [80, 85, 90, 93, 95, 97, 99]:
        for buffer in [0, 1, 2]:
            try:
                bp = semantic_chunker_split(
                    sentences, ema_np,
                    buffer_size=buffer,
                    threshold_percentile=threshold,
                )
                sc_boundaries = [0] + [b + 1 for b in bp] + [n]
                sc_boundaries = sorted(set(sc_boundaries))
                if sc_boundaries[-1] != n:
                    sc_boundaries.append(n)
                metrics = compute_segment_coherence(sentences, ema_np, sc_boundaries)
                results.append({
                    "method": "SemanticChunker",
                    "params": {"threshold": threshold, "buffer": buffer},
                    "metrics": metrics,
                })
            except Exception as e:
                logger.warning("SC th=%d buffer=%d 失败: %s", threshold, buffer, e)

    # 排序：按综合分降序
    results.sort(key=lambda r: r["metrics"]["intra_minus_inter"], reverse=True)

    return results


def print_scan_results(scan_results: List[Dict[str, Any]], top_k: int = 10) -> None:
    """打印参数扫描结果 Top-K。"""
    print("\n" + "=" * 100)
    print(f"  🔬 参数扫描 Top-{min(top_k, len(scan_results))}")
    print("=" * 100)
    print(f"  {'排名':<4} {'方法':<20} {'参数':<30} {'段数':<6} {'段内连贯':<10} {'段间差异':<10} {'综合分':<10}")
    print(f"  {'-'*4} {'-'*20} {'-'*30} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for rank, r in enumerate(scan_results[:top_k], 1):
        m = r["metrics"]
        params_str = str(r["params"])
        print(f"  {rank:<4} {r['method']:<20} {params_str:<30} "
              f"{m['n_segments']:<6} {m['intra_coherence']:<10.4f} "
              f"{m['inter_dissimilarity']:<10.4f} {m['intra_minus_inter']:<10.4f}")

    print("=" * 100)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    """入口：读取配置中指定的文件，运行对比评测。"""
    import argparse

    parser = argparse.ArgumentParser(description="PELT vs SemanticChunker 对比评测")
    parser.add_argument("--file", "-f", type=str, default="",
                        help="输入文本文件路径")
    parser.add_argument("--title", "-t", type=str, default="对比评测",
                        help="文档标题")
    parser.add_argument("--model", "-m", type=str, default="BAAI/bge-m3",
                        help="SBERT 模型名称")
    parser.add_argument("--device", "-d", type=str, default="",
                        help="计算设备 (cpu/mps/cuda)")
    parser.add_argument("--ema_alpha", type=float, default=0.5,
                        help="EMA 平滑系数 (0-1)")
    parser.add_argument("--no_bidirectional", action="store_true",
                        help="禁用双向 EMA")
    parser.add_argument("--pelt_penalty", type=float, default=5.0,
                        help="PELT 惩罚系数")
    parser.add_argument("--semantic_threshold", type=float, default=95,
                        help="SemanticChunker 百分位阈值")
    parser.add_argument("--semantic_buffer", type=int, default=1,
                        help="SemanticChunker buffer size")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="输出报告路径")
    parser.add_argument("--scan", action="store_true",
                        help="执行参数扫描模式")

    args = parser.parse_args()

    # 自动确定 device
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # 确定缓存目录（复用已有 embeddings.npy）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = args.title.replace(" ", "_")
    output_dir = os.path.join(script_dir, dir_name)
    if not os.path.exists(os.path.join(output_dir, "embeddings.npy")):
        # 从现有项目里找已有缓存
        for candidate in ["document", "my_document", "测试文本", "thelittleprince", "XiaoWangZi"]:
            cand_dir = os.path.join(script_dir, candidate)
            if os.path.exists(os.path.join(cand_dir, "embeddings.npy")):
                output_dir = cand_dir
                logger.info("使用已有缓存目录: %s", output_dir)
                break

    # 读取文件
    file_path = args.file
    if not file_path:
        file_path = os.path.join(script_dir, "XiaoWangZi.txt")
        if not os.path.exists(file_path):
            file_path = os.path.join(script_dir, "Notre-Dame de Paris.txt")
        if not os.path.exists(file_path):
            logger.error("未找到输入文件，请使用 --file 指定")
            sys.exit(1)

    logger.info("读取文件: %s", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info("文本长度: %d 字符", len(text))

    if args.scan:
        # ── 参数扫描模式 ──
        logger.info("=== 参数扫描模式 ===")
        scan_results = parameter_scan(
            text, model_name=args.model, device=device,
            ema_alpha=args.ema_alpha,
            ema_bidirectional=not args.no_bidirectional,
            output_dir=output_dir,
        )
        print_scan_results(scan_results, top_k=20)

        # 生成扫描结果图表
        # (为简化，跳过扫描图表生成)
    else:
        # ── 单次对比模式 ──
        results = run_comparison(
            text,
            model_name=args.model,
            device=device,
            ema_alpha=args.ema_alpha,
            ema_bidirectional=not args.no_bidirectional,
            pelt_penalty=args.pelt_penalty,
            semantic_threshold=args.semantic_threshold,
            buffer_size=args.semantic_buffer,
            output_dir=output_dir,
        )

        # 打印对比表格
        print_comparison_table(results)

        # 生成对比报告
        output_path = args.output or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "comparison_report.html",
        )
        generate_comparison_chart(results, output_path)

        logger.info("✅ 评测完成！报告: %s", output_path)

    # 打印总结建议
    print("\n📌 使用建议:")
    print("   PELT 适合：需要自适应分段数、数据有渐变趋势的场景")
    print("   SemanticChunker 适合：段边界清晰、有明确语义跳变的场景")
    print("   `交集`（两者都检测到的边界）最可靠，适合生产使用")
    print("   `并集`（任一检测到的边界）最敏感，适合细粒度分析")
    print()


if __name__ == "__main__":
    main()
