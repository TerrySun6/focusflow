#!/usr/bin/env python3
"""
PCA vs t-SNE vs PHATE 降维对比工具 — Centering（零均值中心化）增强版
===================================================================
基于 text_baai_ema_flask_centering.py 的完整 pipeline：
  读文件 → 分句 → bge-m3 编码 → 滑动窗口 EMA → 零均值中心化(Centering)
  → [PCA / t-SNE / PHATE] 降维 → 对比图 + 量化指标

策略 B（零均值中心化 Centering）：
  在 EMA 平滑后、降维之前，对高维 Embedding 矩阵进行全局中心化消除。
  这能有效移去大模型（BGE-M3）因全局背景主题产生的各向异性偏置（Anisotropy Bias），
  使余弦相似度天然向皮尔逊相关系数演变，放大局部突变信号。

用法:
  python compare_pca_tsne_phate_centering.py <文档路径>

示例:
  python compare_pca_tsne_phate_centering.py 巴黎圣母院.txt
  python compare_pca_tsne_phate_centering.py 测试文本.txt --window-size 5 --alpha 0.3
  python compare_pca_tsne_phate_centering.py my_document.txt --no-tsne --no-centering

输出:
  - compare_pca_tsne_phate_3d.html          — 交互式 3D 对比（plotly）
  - compare_pca_tsne_phate_2d.html          — 交互式 2D 对比（plotly）
  - compare_pca_tsne_phate_metrics.json     — 量化指标 JSON
  - compare_pca_tsne_phate_comparison.png   — 关键指标柱状图对比
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# ★ 配置区 — 在此修改默认测试文件和参数，无需命令行输入 ★
# ═══════════════════════════════════════════════════════════════
# 直接运行 python compare_pca_tsne_phate_centering.py 即可使用下方配置。
# 如想在命令行覆盖，python ... --alpha 0.3 --window-size 5 仍有效。
DEFAULT_FILE = "/Users/terrysun/Documents/learning/project/focusflow/latest_code/XiaoWangZi.txt"  # ← 填入你的测试文件路径，例如 "巴黎圣母院.txt"
DEFAULT_ALPHA = 0.8               # EMA 衰减系数
DEFAULT_WINDOW_SIZE = 5           # 滑动窗口大小 (0=全文EMA)
DEFAULT_ENABLE_CENTERING = True   # 是否启用零均值中心化
DEFAULT_PHATE_KNN = 5             # PHATE kNN
DEFAULT_PHATE_DIST = "cosine"     # PHATE 距离度量
DEFAULT_TSNE_PERPLEXITY = 30      # t-SNE perplexity
DEFAULT_NO_TSNE = False           # 是否跳过 t-SNE
DEFAULT_COMPARE_CENTERING = False # 是否额外对比有/无 centering
# ═══════════════════════════════════════════════════════════════

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("compare_centering")

# ── 中文字体 ──
for _f in ["PingFang SC", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "SimHei", "Microsoft YaHei", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


# ═══════════════════════════════════════════════════════════════
# 以下代码与 text_baai_ema_flask_centering.py 保持一致
# 核心差异：滑动窗口 EMA + 策略 B 零均值中心化
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    input_file_path: str = ""
    doc_title: str = "document"
    sbert_model_name: str = "BAAI/bge-m3"
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
    ema_alpha: float = 0.5
    ema_bidirectional: bool = True
    ema_window_size: int = 0  # 0 = 全文 EMA, >0 = 滑动窗口（自动取奇数）
    enable_centering: bool = True  # ★ 策略 B：零均值中心化


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


# ═══════════════════════════════════════════════════════════════
# ★ 核心：滑动窗口 EMA + 零均值中心化（Centering）
# ═══════════════════════════════════════════════════════════════

def run_ema_with_centering(
    embeddings: torch.Tensor,
    alpha: float = 0.5,
    window_size: int = 0,
    enable_centering: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑动窗口 EMA + 可选策略 B 零均值中心化。

    参数：
        embeddings: (n, d) tensor
        alpha: 指数衰减系数，窗口内权重 = α^|j|
        window_size: 0 = 全文 EMA（经典双向），>0 = 滑动窗口（取奇数）
        enable_centering: 是否执行零均值中心化

    返回：
        (emb_before_ema_cpu, emb_after_ema_cpu, emb_after_centering_cpu)
        - emb_before_ema_cpu:  原始嵌入（用于 baseline 对比）
        - emb_after_ema_cpu:   EMA 平滑后（centering 前）
        - emb_after_centering_cpu:  EMA + Centering 后（最终结果）
    """
    emb = embeddings  # (n, d)
    n = emb.shape[0]

    # 保存原始值
    emb_orig_cpu = emb.detach().cpu().numpy()

    # 使用滑动窗口 EMA（来自 text_baai_ema_flask_centering.py）
    emb_out = _sliding_window_ema(emb, alpha, window_size)
    emb_ema_cpu = emb_out.detach().cpu().numpy()

    # 策略 B：零均值中心化
    if enable_centering:
        emb_final_cpu = emb_ema_cpu - np.mean(emb_ema_cpu, axis=0)
    else:
        emb_final_cpu = emb_ema_cpu

    return emb_orig_cpu, emb_ema_cpu, emb_final_cpu


def _sliding_window_ema(emb: torch.Tensor, alpha: float, window_size: int) -> torch.Tensor:
    """滑动窗口 EMA（与 text_baai_ema_flask_centering.py 完全一致）。

    对每个位置 i，取以 i 为中心的奇数大小窗口 [i-k, i+k]，
    窗口内各 embedding 的权重按到中心的距离指数衰减：w(j) = α^|j|，
    然后做加权平均。步长 = 1，遍历所有位置。
    边界处窗口自动截断并重新归一化权重。
    """
    n = emb.shape[0]

    # window <= 0 或 n 太小 → 不平滑，返回原始值
    if window_size <= 0 or n < 2:
        return emb.clone()

    # 确保窗口为奇数
    if window_size % 2 == 0:
        window_size += 1
    k = window_size // 2

    # 预计算权重：α^|j|  for  j = -k ... +k
    j = torch.arange(-k, k + 1, device=emb.device)
    weights = alpha ** torch.abs(j)  # (W,)

    out = torch.zeros_like(emb)

    for i in range(n):
        left = max(0, i - k)
        right = min(n, i + k + 1)
        w = emb[left:right]
        w_left = left - (i - k)
        w_right = (i + k + 1) - right
        valid_weights = weights[w_left:window_size - w_right]
        valid_weights = valid_weights / valid_weights.sum()
        out[i] = (w * valid_weights.unsqueeze(1)).sum(dim=0)

    return out


# ═══════════════════════════════════════════════════════════════
# 降维方法
# ═══════════════════════════════════════════════════════════════

def run_pca(X: np.ndarray, n_components: int = 3) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components).fit_transform(X)


def run_tsne(X: np.ndarray, n_components: int = 3,
             perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn import __version__ as sk_ver
    from packaging.version import Version
    kwargs = dict(
        n_components=n_components,
        perplexity=min(perplexity, X.shape[0] - 1),
        random_state=random_state, init="pca",
    )
    if Version(sk_ver) >= Version("1.5"):
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    return TSNE(**kwargs).fit_transform(X)


def run_phate(X: np.ndarray, n_components: int = 3,
              knn: int = 5, knn_dist: str = "cosine") -> np.ndarray:
    import phate
    op = phate.PHATE(
        n_components=n_components, knn=knn,
        knn_dist=knn_dist, mds="metric",
        n_jobs=-1, verbose=False,
    )
    return op.fit_transform(X)


# ═══════════════════════════════════════════════════════════════
# 量化指标
# ═══════════════════════════════════════════════════════════════

def trustworthiness_score(X_high: np.ndarray, X_low: np.ndarray) -> float:
    from sklearn.manifold import trustworthiness
    return float(trustworthiness(X_high, X_low, n_neighbors=15))


def trajectory_smoothness(X_low: np.ndarray) -> float:
    """轨迹平滑度：相邻步向量的平均夹角（弧度），越小越平滑。"""
    diffs = np.diff(X_low, axis=0)
    angles = []
    for i in range(len(diffs) - 1):
        v1, v2 = diffs[i], diffs[i + 1]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return float(np.mean(angles)) if angles else 0.0


def clustering_ari(X_low: np.ndarray, high_labels: np.ndarray) -> float:
    """降维后聚类与高维聚类的 ARI 一致性。"""
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.cluster import HDBSCAN
    low_labels = HDBSCAN(min_cluster_size=3, min_samples=2).fit_predict(X_low)
    mask = (high_labels != -1) & (low_labels != -1)
    if mask.sum() < 5:
        return 0.0
    return float(ari(high_labels[mask], low_labels[mask]))


def compute_trajectory_stats(X_low: np.ndarray) -> Dict[str, float]:
    """轨迹统计：总长度、端点距离、弯曲比。"""
    total_dist = float(np.sum(np.linalg.norm(X_low[1:] - X_low[:-1], axis=1)))
    end_dist = float(np.linalg.norm(X_low[0] - X_low[-1]))
    return {
        "total_distance": round(total_dist, 2),
        "end_to_end_distance": round(end_dist, 2),
        "tortuosity": round(total_dist / max(end_dist, 1e-9), 2),
    }


def compute_coherence_preservation(
    emb_high: np.ndarray, emb_low: np.ndarray
) -> Dict[str, float]:
    """衡量降维前后相邻点相似度保持能力。"""
    n = min(len(emb_high), len(emb_low))
    # 高维相邻相似度
    high_sim = np.array([
        float(np.dot(emb_high[i], emb_high[i+1]) /
              (np.linalg.norm(emb_high[i]) * np.linalg.norm(emb_high[i+1]) + 1e-9))
        for i in range(n - 1)
    ])
    # 低维相邻相似度
    low_sim = np.array([
        float(np.dot(emb_low[i], emb_low[i+1]) /
              (np.linalg.norm(emb_low[i]) * np.linalg.norm(emb_low[i+1]) + 1e-9))
        for i in range(n - 1)
    ])
    # 排名相关性 (Spearman)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(high_sim, low_sim)
    return {
        "high_dim_mean_coherence": round(float(high_sim.mean()), 4),
        "low_dim_mean_coherence": round(float(low_sim.mean()), 4),
        "coherence_spearman_r": round(float(rho), 4),
    }


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

def _make_trace_coords(c: np.ndarray, method: str, sentences: List[str] = None):
    """生成 3D 坐标点的 scatter trace + 轨迹线 + 首尾标记 + hover 文本。"""
    import plotly.graph_objects as go
    n = c.shape[0]
    colors = [f"hsl({(i / n) * 300}, 70%, 50%)" for i in range(n)]
    hover_text = None
    if sentences:
        hover_text = [sentences[i][:80] + "..." if len(sentences[i]) > 80 else sentences[i]
                      for i in range(n)]

    scatter = go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors, opacity=0.7),
        text=hover_text,
        hoverinfo="text" if hover_text else "all",
        name=f"{method.upper()} 点",
        showlegend=False,
    )
    line = go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="lines",
        line=dict(color="gray", width=0.8),
        name=f"{method.upper()} 轨迹",
        showlegend=False,
    )
    start_marker = go.Scatter3d(
        x=[c[0, 0]], y=[c[0, 1]], z=[c[0, 2]],
        mode="markers",
        marker=dict(size=6, color="red", symbol="circle"),
        name="Start",
    )
    end_marker = go.Scatter3d(
        x=[c[-1, 0]], y=[c[-1, 1]], z=[c[-1, 2]],
        mode="markers",
        marker=dict(size=6, color="blue", symbol="square"),
        name="End",
    )

    return scatter, line, start_marker, end_marker


def _make_trace_2d(c: np.ndarray, method: str, sentences: List[str] = None):
    """生成 2D 坐标点 trace + 轨迹线 + 首尾标记 + hover 文本。"""
    import plotly.graph_objects as go
    n = c.shape[0]
    colors = [f"hsl({(i / n) * 300}, 70%, 50%)" for i in range(n)]
    hover_text = None
    if sentences:
        hover_text = [sentences[i][:80] + "..." if len(sentences[i]) > 80 else sentences[i]
                      for i in range(n)]

    scatter = go.Scatter(
        x=c[:, 0], y=c[:, 1],
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.85),
        text=hover_text,
        hoverinfo="text" if hover_text else "all",
        name=f"{method.upper()} 点",
        showlegend=False,
    )
    line = go.Scatter(
        x=c[:, 0], y=c[:, 1],
        mode="lines",
        line=dict(color="gray", width=1),
        name=f"{method.upper()} 轨迹",
        showlegend=False,
    )
    start_marker = go.Scatter(
        x=[c[0, 0]], y=[c[0, 1]],
        mode="markers",
        marker=dict(size=12, color="red", symbol="circle"),
        name="Start",
    )
    end_marker = go.Scatter(
        x=[c[-1, 0]], y=[c[-1, 1]],
        mode="markers",
        marker=dict(size=12, color="blue", symbol="square"),
        name="End",
    )
    return scatter, line, start_marker, end_marker


def visualize_html(
    coords: Dict[str, np.ndarray],
    title: str,
    save_path_3d: str,
    save_path_2d: str,
    sentences: List[str] = None,
):
    """用 plotly 生成交互式 3D + 2D HTML 对比图。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_methods = len(coords)
    method_names = list(coords.keys())

    # ── 3D ──
    fig_3d = make_subplots(
        rows=1, cols=n_methods,
        specs=[[{"type": "scatter3d"} for _ in range(n_methods)]],
        subplot_titles=[m.upper() for m in method_names],
    )
    for idx, (method, c) in enumerate(coords.items(), 1):
        traces = _make_trace_coords(c, method, sentences)
        for t in traces:
            fig_3d.add_trace(t, row=1, col=idx)

    fig_3d.update_layout(
        title=dict(text=title, x=0.5),
        height=550, width=350 * n_methods,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05,
            xanchor="center", x=0.5,
        ),
    )
    fig_3d.write_html(save_path_3d, include_plotlyjs="cdn")
    logger.info("已保存: %s", save_path_3d)

    # ── 2D ──
    fig_2d = make_subplots(
        rows=1, cols=n_methods,
        subplot_titles=[m.upper() for m in method_names],
    )
    for idx, (method, c) in enumerate(coords.items(), 1):
        traces = _make_trace_2d(c, method, sentences)
        for t in traces:
            fig_2d.add_trace(t, row=1, col=idx)

    # 等比缩放
    max_ranges = []
    for c in coords.values():
        ranges = [c[:, i].max() - c[:, i].min() for i in range(2)]
        max_ranges.append(max(ranges))
    global_max = max(max_ranges)
    x_rng = [-global_max * 0.6, global_max * 0.6]
    y_rng = [-global_max * 0.6, global_max * 0.6]

    for i in range(1, n_methods + 1):
        fig_2d.update_xaxes(range=x_rng, row=1, col=i, visible=False)
        fig_2d.update_yaxes(range=y_rng, row=1, col=i, visible=False, scaleanchor="x", scaleratio=1)

    fig_2d.update_layout(
        title=dict(text=title, x=0.5),
        height=500, width=350 * n_methods,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig_2d.write_html(save_path_2d, include_plotlyjs="cdn")
    logger.info("已保存: %s", save_path_2d)


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str,
    title: str,
):
    """生成关键指标的柱状图对比（含/不含 centering 对比）。"""
    methods = list(metrics_dict.keys())

    # 提取三个核心指标
    metric_names = ["trustworthiness", "smoothness", "clustering_ari"]
    metric_labels = ["可信任度↑", "平滑度↓(弧度)", "聚类ARI↑"]

    fig, axes = plt.subplots(1, 3, figsize=(5 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]

    colors = ["#4a6cf7", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, m_name, m_label in zip(axes, metric_names, metric_labels):
        values = [metrics_dict[m].get(m_name, 0) for m in methods]
        bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(m_label, fontsize=11)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("指标对比图已保存: %s", save_path)


# ═══════════════════════════════════════════════════════════════
# Centering 效果对比（额外对比：有/无 centering 的 PHATE 轨迹差异）
# ═══════════════════════════════════════════════════════════════

def compare_centering_effect(
    emb_ema: np.ndarray,
    emb_centered: np.ndarray,
    coords_ema: np.ndarray,
    coords_centered: np.ndarray,
    sentences: List[str],
) -> Dict[str, Any]:
    """对比有/无 centering 的 PHATE 结构和指标差异。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    result = {}

    # 1. 轨迹统计对比
    stats_ema = compute_trajectory_stats(coords_ema)
    stats_centered = compute_trajectory_stats(coords_centered)
    result["trajectory_stats"] = {
        "without_centering": stats_ema,
        "with_centering": stats_centered,
    }

    # 2. 高维余弦相似度分布对比
    n = len(emb_ema)
    sims_ema = np.array([
        float(np.dot(emb_ema[i], emb_ema[i+1]) /
              (np.linalg.norm(emb_ema[i]) * np.linalg.norm(emb_ema[i+1]) + 1e-9))
        for i in range(n - 1)
    ])
    sims_centered = np.array([
        float(np.dot(emb_centered[i], emb_centered[i+1]) /
              (np.linalg.norm(emb_centered[i]) * np.linalg.norm(emb_centered[i+1]) + 1e-9))
        for i in range(n - 1)
    ])
    result["coherence"] = {
        "without_centering_mean": round(float(sims_ema.mean()), 4),
        "without_centering_std": round(float(sims_ema.std()), 4),
        "with_centering_mean": round(float(sims_centered.mean()), 4),
        "with_centering_std": round(float(sims_centered.std()), 4),
        "delta_mean": round(float(sims_centered.mean() - sims_ema.mean()), 4),
    }

    # 3. PHATE 3D 对比图（有/无 centering 并排）
    n_sentences = len(sentences)
    colors = [f"hsl({(i / n_sentences) * 300}, 70%, 50%)" for i in range(n_sentences)]
    hover_text = [s[:60] + "..." if len(s) > 60 else s for s in sentences]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["PHATE (无 Centering)", "PHATE (有 Centering)"],
    )

    for idx, (c, label) in enumerate([
        (coords_ema, "无 Centering"),
        (coords_centered, "有 Centering"),
    ], 1):
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="lines",
            line=dict(color="gray", width=1),
            name=f"{label} 轨迹",
            showlegend=False,
        ), row=1, col=idx)
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.7),
            text=hover_text,
            hoverinfo="text",
            name=f"{label} 点",
            showlegend=False,
        ), row=1, col=idx)
        # 起点
        fig.add_trace(go.Scatter3d(
            x=[c[0, 0]], y=[c[0, 1]], z=[c[0, 2]],
            mode="markers",
            marker=dict(size=6, color="red", symbol="circle"),
            name=f"{label} Start",
        ), row=1, col=idx)
        # 终点
        fig.add_trace(go.Scatter3d(
            x=[c[-1, 0]], y=[c[-1, 1]], z=[c[-1, 2]],
            mode="markers",
            marker=dict(size=6, color="blue", symbol="square"),
            name=f"{label} End",
        ), row=1, col=idx)

    fig.update_layout(
        title=dict(text="Centering 效果对比", x=0.5),
        height=500, width=800,
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )
    save_path = "centering_effect_3d.html"
    fig.write_html(save_path, include_plotlyjs="cdn")
    logger.info("Centering 效果对比图已保存: %s", save_path)
    result["centering_comparison_html"] = save_path

    return result


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 配置缺省值（被配置区的 DEFAULT_* 覆盖） ──
    parser = argparse.ArgumentParser(
        description="PCA vs t-SNE vs PHATE 降维对比 — Centering 增强版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python compare_pca_tsne_phate_centering.py 巴黎圣母院.txt
              python compare_pca_tsne_phate_centering.py 测试文本.txt --window-size 5
              python compare_pca_tsne_phate_centering.py my_document.txt --no-tsne --no-centering
              
            ★ 也可直接运行 python compare_pca_tsne_phate_centering.py，
               此时使用文件顶部配置区的 DEFAULT_FILE 等参数。
        """),
    )
    parser.add_argument("file", type=str, nargs="?", default=DEFAULT_FILE or None,
                        help="输入的 .txt 文档路径（缺省时使用配置区的 DEFAULT_FILE）")
    parser.add_argument("--phate-knn", type=int, default=DEFAULT_PHATE_KNN, help="PHATE kNN")
    parser.add_argument("--phate-dist", type=str, default=DEFAULT_PHATE_DIST,
                        choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--tsne-perplexity", type=int, default=DEFAULT_TSNE_PERPLEXITY, help="t-SNE perplexity")
    parser.add_argument("--no-tsne", action="store_true", default=DEFAULT_NO_TSNE, help="跳过 t-SNE")
    parser.add_argument("--output-prefix", type=str, default="compare_pca_tsne_phate", help="输出前缀")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="Embedding 模型")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="EMA 衰减系数")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="滑动窗口大小，0=全文EMA")
    parser.add_argument("--no-centering", action="store_true", default=not DEFAULT_ENABLE_CENTERING,
                        help="禁用零均值中心化（策略 B）")
    parser.add_argument("--compare-centering", action="store_true", default=DEFAULT_COMPARE_CENTERING,
                        help="额外对比有/无 centering 的效果")
    args = parser.parse_args()

    # 如果命令行没给 file 且 DEFAULT_FILE 也没设，报错提示
    file_path = args.file
    if not file_path:
        logger.error("未指定文件！请在文件顶部配置区设置 DEFAULT_FILE，或在命令行传入路径。")
        logger.error("例如：python compare_pca_tsne_phate_centering.py 巴黎圣母院.txt")
        sys.exit(1)

    enable_centering = not args.no_centering

    if not os.path.exists(file_path):
        logger.error("文件不存在: %s", file_path)
        sys.exit(1)

    doc_name = os.path.splitext(os.path.basename(file_path))[0]
    out = args.output_prefix

    # ── Step 1: 读文件 ──
    logger.info("=" * 60)
    logger.info("Step 1/6: 读取文件 %s", file_path)
    text = read_text_file(file_path)

    # ── Step 2: 分句 ──
    logger.info("Step 2/6: 分句")
    sentences = split_text_into_segments(text)
    logger.info("  共 %d 个句子", len(sentences))

    if len(sentences) < 3:
        logger.error("句子数太少 (%d)，无法进行有意义的分析", len(sentences))
        sys.exit(1)

    # ── Step 3: bge-m3 编码 ──
    logger.info("Step 3/6: bge-m3 编码 (%s)", args.model)
    cache_dir = f".compare_centering_cache_{hashlib.md5(text.encode()).hexdigest()[:12]}"
    os.makedirs(cache_dir, exist_ok=True)

    cfg = Config(
        input_file_path=file_path,
        doc_title=doc_name,
        sbert_model_name=args.model,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        ema_alpha=args.alpha,
        ema_window_size=args.window_size,
        enable_centering=enable_centering,
    )
    emb_tensor = load_or_embed(sentences, cache_dir, args.model, cfg.device)
    logger.info("  Embedding 形状: %s", emb_tensor.shape)

    # ── Step 4: 滑动窗口 EMA + Centering ──
    logger.info("Step 4/6: 滑动窗口 EMA (α=%.2f, window=%d) %s Centering",
                args.alpha, args.window_size,
                "+" if enable_centering else "without")
    emb_orig_cpu, emb_ema_cpu, emb_final_cpu = run_ema_with_centering(
        embeddings=emb_tensor,
        alpha=args.alpha,
        window_size=args.window_size,
        enable_centering=enable_centering,
    )
    logger.info("  EMA 平滑完成 → 最终形状: %s", emb_final_cpu.shape)

    # ── Step 5: 三种降维 ──
    logger.info("Step 5/6: 降维对比")
    n = emb_final_cpu.shape[0]

    # 归一化
    emb_norm = emb_final_cpu / (np.linalg.norm(emb_final_cpu, axis=1, keepdims=True) + 1e-10)

    coords = {}
    logger.info("  [1/3] PCA ...")
    coords["pca"] = run_pca(emb_norm)

    if not args.no_tsne and n >= 5:
        logger.info("  [2/3] t-SNE (perplexity=%d) ...", args.tsne_perplexity)
        coords["t-sne"] = run_tsne(emb_norm, perplexity=args.tsne_perplexity)
    else:
        logger.info("  [2/3] 跳过 t-SNE")

    logger.info("  [3/3] PHATE (knn=%d, dist=%s) ...", args.phate_knn, args.phate_dist)
    coords["phate"] = run_phate(emb_norm, knn=args.phate_knn, knn_dist=args.phate_dist)

    # ── 量化指标 ──
    logger.info("\n" + "─" * 60)
    logger.info("量化指标")
    logger.info("─" * 60)

    from sklearn.cluster import HDBSCAN
    high_labels = HDBSCAN(min_cluster_size=3, min_samples=2).fit_predict(emb_norm)
    n_clusters = len(set(high_labels)) - (1 if -1 in high_labels else 0)
    n_noise = (high_labels == -1).sum()
    logger.info("  高维 HDBSCAN: %d 簇 + %d 噪声点", n_clusters, n_noise)

    metrics = {}
    for method, c in coords.items():
        m = {
            "trustworthiness": round(trustworthiness_score(emb_norm, c), 4),
            "smoothness": round(trajectory_smoothness(c), 4),
            "clustering_ari": round(clustering_ari(c, high_labels), 4),
        }
        # 额外轨迹统计
        m["trajectory"] = compute_trajectory_stats(c)
        # 连贯性保持
        m["coherence"] = compute_coherence_preservation(emb_norm, c)

        metrics[method] = m
        logger.info("  %-8s | trustworthy=%.4f | smooth=%.4f rad | ARI=%.4f",
                     method.upper(), m["trustworthiness"], m["smoothness"], m["clustering_ari"])
        logger.info("          └─ trajectory: %s", m["trajectory"])
        logger.info("          └─ coherence:  %s", m["coherence"])

    # ── Step 6: 可视化 ──
    logger.info("\nStep 6/6: 生成可视化")
    title = (f"降维对比: {doc_name}  ({n}句, {emb_tensor.shape[1]}维, "
             f"α={args.alpha}, window={args.window_size}, "
             f"{'Centering ✓' if enable_centering else 'Centering ✗'})")

    visualize_html(coords, title, f"{out}_3d.html", f"{out}_2d.html", sentences)

    # 指标对比柱状图
    plot_metrics_comparison(metrics, f"{out}_comparison.png", title)

    # ── 可选：Centering 效果对比 ──
    centering_comparison = None
    if args.compare_centering:
        logger.info("\n" + "─" * 60)
        logger.info("Centering 效果对比：用 PHATE 展示有/无 centering 的差异")
        logger.info("─" * 60)

        # 跑一次 无 centering 的 PHATE
        _, _, emb_no_center = run_ema_with_centering(
            embeddings=emb_tensor,
            alpha=args.alpha,
            window_size=args.window_size,
            enable_centering=False,
        )
        emb_no_center_norm = emb_no_center / (np.linalg.norm(emb_no_center, axis=1, keepdims=True) + 1e-10)
        coords_no_center = run_phate(emb_no_center_norm, knn=args.phate_knn, knn_dist=args.phate_dist)

        centering_comparison = compare_centering_effect(
            emb_ema=emb_no_center,
            emb_centered=emb_final_cpu,
            coords_ema=coords_no_center,
            coords_centered=coords["phate"],
            sentences=sentences,
        )

        logger.info("  Centering 效果对比:")
        logger.info("    无 Centering 轨迹: %s", centering_comparison["trajectory_stats"]["without_centering"])
        logger.info("    有 Centering 轨迹: %s", centering_comparison["trajectory_stats"]["with_centering"])
        logger.info("    连贯性变化: %s", centering_comparison["coherence"])

    # ── 保存指标 ──
    metrics_data = {
        "dataset": doc_name,
        "n_sentences": n,
        "n_dimensions": emb_tensor.shape[1],
        "model": args.model,
        "config": {
            "ema_alpha": args.alpha,
            "ema_window_size": args.window_size,
            "enable_centering": enable_centering,
            "phate_knn": args.phate_knn,
            "phate_knn_dist": args.phate_dist,
            "tsne_perplexity": args.tsne_perplexity,
        },
        "high_dim_clustering": {
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
        },
        "metrics": metrics,
        "centering_comparison": centering_comparison,
    }
    with open(f"{out}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info("指标已保存: %s_metrics.json", out)

    # ── 清理缓存 ──
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    # ── 汇总报告 ──
    logger.info("\n" + "=" * 60)
    logger.info("🎉 完成！产出文件:")
    logger.info("  %s_3d.html          (交互式 3D 对比)", out)
    logger.info("  %s_2d.html          (交互式 2D 对比)", out)
    logger.info("  %s_metrics.json     (量化指标)", out)
    logger.info("  %s_comparison.png   (指标柱状图)", out)
    if centering_comparison:
        logger.info("  centering_effect_3d.html (Centering 效果对比)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
