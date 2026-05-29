#!/usr/bin/env python3
"""
PCA vs t-SNE vs PHATE vs UMAP 降维对比工具
===================================
从 text_baai_ema_flask_semantic.py 复用的完整 pipeline：
  读文件 → 分句 → bge-m3 编码 → EMA 平滑 → [PCA / t-SNE / PHATE / UMAP] 降维 → 对比图 + 量化指标

用法:
  python compare_pca_tsne_phate_umap.py <文档路径>

示例:
  python compare_pca_tsne_phate_umap.py Paris圣母院.txt
  python compare_pca_tsne_phate_umap.py 测试文本.txt
  python compare_pca_tsne_phate_umap.py my_document_ema --no-tsne

输出:
  - compare_3d.html      — 交互式 3D 对比（plotly，可旋转缩放）
  - compare_2d.html      — 交互式 2D 对比
  - compare_metrics.json — 量化指标
"""

import argparse, hashlib, json, os, re, sys, textwrap, warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("compare")

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
# 以下代码与 text_baai_ema_flask_semantic.py 保持一致
# ═══════════════════════════════════════════════════════════════

def _cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def semantic_chunker_split(
    sentences: List[str],
    embeddings: np.ndarray,
    buffer_size: int = 1,
    threshold_percentile: float = 95,
    min_chunk_size: Optional[int] = None,
) -> List[int]:
    n = len(sentences)
    if n < 2:
        return []
    combined = []
    for i in range(n):
        parts = []
        for j in range(i - buffer_size, i + buffer_size + 1):
            if 0 <= j < n:
                parts.append(sentences[j])
        combined.append(" ".join(parts))
    _emb = embeddings
    distances = []
    for i in range(n - 1):
        sim = _cosine_similarity_vec(_emb[i], _emb[i + 1])
        distances.append(1.0 - sim)
    threshold = float(np.percentile(distances, threshold_percentile))
    indices_above = [i for i, d in enumerate(distances) if d > threshold]
    return indices_above

def _get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"

@dataclass
class Config:
    input_file_path: str = ""
    doc_title: str = "document"
    sbert_model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    ema_alpha: float = 0.6
    ema_bidirectional: bool = True
    ema_window_size: int = 5
    phate_n_components: int = 3
    phate_knn: int = 5
    phate_knn_dist: str = "cosine"
    mds_method: str = "metric"
    semantic_threshold_percentile: float = 95.0
    semantic_buffer_size: int = 1
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 5

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

class EMAnalyzer:
    def __init__(self, embeddings: torch.Tensor, cfg: Config):
        self.embeddings_gpu = embeddings
        self.embeddings_cpu = embeddings.cpu().numpy()
        self.cfg = cfg
        self.results = {}

    def run_ema(self) -> np.ndarray:
        """滑动窗口 EMA + 策略 B 全局中心化。

        对每个位置 i，取以 i 为中心的奇数大小窗口 [i-k, i+k]，
        窗口内各 embedding 的权重按到中心的距离指数衰减：w(j) = α^|j|，
        然后做加权平均。步长 = 1，遍历所有位置。

        边界处窗口自动截断并重新归一化权重。

        在返回之前，执行策略 B（零均值中心化）：
        减去全篇均值，将全局背景偏置重置为空间原点。
        """
        emb = self.embeddings_gpu          # (n, d)
        n = emb.shape[0]
        alpha = self.cfg.ema_alpha
        window = self.cfg.ema_window_size  # 滑动窗口大小（奇数）

        # window <= 0 或 n 太小 → 不平滑，但依然进行中心化消除偏置
        if window <= 0 or n < 2:
            self.embeddings_cpu = emb.detach().cpu().numpy()
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

        # ── 向量化滑动窗口 ──
        # 手动 replicate padding（F.pad replicate 不支持 2D 张量）
        emb_pad = torch.cat([emb[:1].expand(k, -1), emb, emb[-1:].expand(k, -1)], dim=0)
        windows = emb_pad.unfold(0, window, 1).permute(0, 2, 1)       # (n, W, d)

        # 逐位置有效性 mask
        j_idx = torch.arange(window, device=emb.device).view(1, window)
        i_idx = torch.arange(n, device=emb.device).view(n, 1)
        pos = i_idx + (j_idx - k)
        valid_mask = ((pos >= 0) & (pos < n)).float()

        # 逐位置归一化权重
        norm_weights = weights.view(1, window) * valid_mask
        norm_weights = norm_weights / norm_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)

        out = (windows * norm_weights.unsqueeze(-1)).sum(dim=1)       # (n, d)

        self.embeddings_gpu = out
        self.embeddings_cpu = out.detach().cpu().numpy()

        # ── 策略 B 核心切入点 ──
        self.embeddings_cpu = self.embeddings_cpu - np.mean(self.embeddings_cpu, axis=0)

        return self.embeddings_cpu

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

def run_umap(X: np.ndarray, n_components: int = 3,
             n_neighbors: int = 5, min_dist: float = 0.1,
             metric: str = "cosine", random_state: int = 42) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)

# ═══════════════════════════════════════════════════════════════
# 量化指标
# ═══════════════════════════════════════════════════════════════

def trustworthiness_score(X_high: np.ndarray, X_low: np.ndarray) -> float:
    from sklearn.manifold import trustworthiness
    return float(trustworthiness(X_high, X_low, n_neighbors=15))

def trajectory_smoothness(X_low: np.ndarray) -> float:
    diffs = np.diff(X_low, axis=0)
    angles = []
    for i in range(len(diffs) - 1):
        v1, v2 = diffs[i], diffs[i + 1]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return float(np.mean(angles)) if angles else 0.0

def clustering_ari(X_low: np.ndarray, high_labels: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.cluster import HDBSCAN
    low_labels = HDBSCAN(min_cluster_size=3, min_samples=2).fit_predict(X_low)
    mask = (high_labels != -1) & (low_labels != -1)
    if mask.sum() < 5:
        return 0.0
    return float(ari(high_labels[mask], low_labels[mask]))

# ═══════════════════════════════════════════════════════════════
# 可视化（plotly 交互式 HTML）
# ═══════════════════════════════════════════════════════════════

def _make_trace_coords(c: np.ndarray, method: str):
    """生成坐标点的 scatter trace + 轨迹线 + 首尾标记"""
    import plotly.graph_objects as go
    n = c.shape[0]
    colors = [f"hsl({(i / n) * 300}, 70%, 50%)" for i in range(n)]

    scatter = go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers",
        marker=dict(size=4, color=colors, opacity=0.85),
        name=f"{method.upper()} 点",
        showlegend=False,
    )
    # 轨迹线
    line = go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="lines",
        line=dict(color="gray", width=1),
        name=f"{method.upper()} 轨迹",
        showlegend=False,
    )
    # 起点/终点标记
    start_marker = go.Scatter3d(
        x=[c[0, 0]], y=[c[0, 1]], z=[c[0, 2]],
        mode="markers",
        marker=dict(size=10, color="red", symbol="circle"),
        name="Start",
    )
    end_marker = go.Scatter3d(
        x=[c[-1, 0]], y=[c[-1, 1]], z=[c[-1, 2]],
        mode="markers",
        marker=dict(size=10, color="blue", symbol="square"),
        name="End",
    )
    return scatter, line, start_marker, end_marker

def _make_trace_2d(c: np.ndarray, method: str):
    """2D 坐标点 trace + 轨迹线 + 首尾标记"""
    import plotly.graph_objects as go
    n = c.shape[0]
    colors = [f"hsl({(i / n) * 300}, 70%, 50%)" for i in range(n)]

    scatter = go.Scatter(
        x=c[:, 0], y=c[:, 1],
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.85),
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

def visualize_html(coords: dict, title: str, save_path_3d: str, save_path_2d: str):
    """用 plotly 生成交互式 3D + 2D HTML"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_methods = len(coords)
    method_names = list(coords.keys())

    # === 3D: 每个方法独立子图，水平排列 ===
    fig_3d = make_subplots(
        rows=1, cols=n_methods,
        specs=[[{"type": "scatter3d"} for _ in range(n_methods)]],
        subplot_titles=[m.upper() for m in method_names],
    )
    for idx, (method, c) in enumerate(coords.items(), 1):
        traces = _make_trace_coords(c, method)
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

    # === 2D: 同样水平排列 ===
    fig_2d = make_subplots(
        rows=1, cols=n_methods,
        subplot_titles=[m.upper() for m in method_names],
    )
    for idx, (method, c) in enumerate(coords.items(), 1):
        traces = _make_trace_2d(c, method)
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

# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PCA vs t-SNE vs PHATE vs UMAP 降维对比（完整 Pipeline）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python compare_pca_tsne_phate_umap.py 测试文本.txt
              python compare_pca_tsne_phate_umap.py 巴黎圣母院.txt --phate-knn 10
              python compare_pca_tsne_phate_umap.py my_document_ema.txt --no-tsne
        """),
    )
    parser.add_argument("file", type=str, help="输入的 .txt 文档路径")
    parser.add_argument("--phate-knn", type=int, default=5, help="PHATE kNN (default: 5)")
    parser.add_argument("--phate-dist", type=str, default="cosine", choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--tsne-perplexity", type=int, default=30, help="t-SNE perplexity (default: 30)")
    parser.add_argument("--no-tsne", action="store_true", help="跳过 t-SNE（数据点少时）")
    parser.add_argument("--no-umap", action="store_true", help="跳过 UMAP")
    parser.add_argument("--output-prefix", type=str, default="compare", help="输出前缀 (default: 'compare')")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="Embedding 模型 (default: BAAI/bge-m3)")
    parser.add_argument("--ema-alpha", type=float, default=0.6, help="EMA 平滑系数 (default: 0.6)")
    parser.add_argument("--ema-window", type=int, default=5, help="滑动窗口大小 (default: 5, 奇数)")
    parser.add_argument("--ema-bidirectional", action="store_true", default=True, help="双向 EMA (default: True)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error("文件不存在: %s", args.file)
        sys.exit(1)

    # Step 1: 读文件
    logger.info("=" * 60)
    logger.info("Step 1/5: 读取文件 %s", args.file)
    text = read_text_file(args.file)

    # Step 2: 分句
    logger.info("Step 2/5: 分句")
    sentences = split_text_into_segments(text)
    logger.info("  共 %d 个句子", len(sentences))

    # Step 3: bge-m3 编码
    logger.info("Step 3/5: bge-m3 编码 (%s)", args.model)
    doc_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    cache_dir = f".compare_cache_{doc_hash}"
    os.makedirs(cache_dir, exist_ok=True)

    cfg = Config(
        input_file_path=args.file,
        doc_title=os.path.splitext(os.path.basename(args.file))[0],
        sbert_model_name=args.model,
        device=_get_device(),
        ema_alpha=args.ema_alpha,
        ema_window_size=args.ema_window,
        ema_bidirectional=args.ema_bidirectional,
    )
    emb_tensor = load_or_embed(sentences, cache_dir, args.model, cfg.device)
    logger.info("  embedding 形状: %s", emb_tensor.shape)

    # Step 4: EMA 平滑
    logger.info("Step 4/5: EMA 平滑 (alpha=%.2f)", args.ema_alpha)
    analyzer = EMAnalyzer(emb_tensor, cfg)
    analyzer.run_ema()
    emb_smooth = analyzer.embeddings_cpu

    # Step 5: 三种降维
    logger.info("Step 5/5: 降维对比")
    n = emb_smooth.shape[0]

    # 归一化
    emb_norm = emb_smooth / (np.linalg.norm(emb_smooth, axis=1, keepdims=True) + 1e-10)

    coords = {}
    logger.info("  [1/3] PCA ...")
    coords["pca"] = run_pca(emb_norm)

    if not args.no_tsne and n >= 5:
        logger.info("  [2/3] t-SNE (perplexity=%d) ...", args.tsne_perplexity)
        coords["t-sne"] = run_tsne(emb_norm, perplexity=args.tsne_perplexity)
    else:
        logger.info("  [2/3] 跳过 t-SNE")

    logger.info("  [3/4] PHATE (knn=%d) ...", args.phate_knn)
    coords["phate"] = run_phate(emb_norm, knn=args.phate_knn, knn_dist=args.phate_dist)

    if not getattr(args, 'no_umap', False):
        logger.info("  [4/4] UMAP (n_neighbors=%d) ...", args.phate_knn)
        coords["umap"] = run_umap(emb_norm, n_neighbors=args.phate_knn, metric=args.phate_dist)
    else:
        logger.info("  [4/4] 跳过 UMAP")

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
            "trustworthiness": trustworthiness_score(emb_norm, c),
            "smoothness": trajectory_smoothness(c),
            "clustering_ari": clustering_ari(c, high_labels),
        }
        metrics[method] = m
        logger.info("  %-8s | trustworthy=%.4f | smoothness=%.4f rad | ARI=%.4f",
                     method.upper(), m["trustworthiness"], m["smoothness"], m["clustering_ari"])

    # ── 可视化 ──
    doc_name = os.path.splitext(os.path.basename(args.file))[0]
    title = f"降维对比: {doc_name}  ({n}句, {emb_tensor.shape[1]}维, α={args.ema_alpha})"
    out = args.output_prefix

    visualize_html(coords, title, f"{out}_3d.html", f"{out}_2d.html")

    # ── 保存指标 ──
    metrics_data = {
        "dataset": doc_name,
        "n_sentences": n,
        "n_dimensions": emb_tensor.shape[1],
        "ema_alpha": args.ema_alpha,
        "metrics": metrics,
    }
    with open(f"{out}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info("指标已保存: %s_metrics.json", out)

    # ── 清理缓存 ──
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info("完成！产出:")
    logger.info("  %s_3d.html  (交互式 3D)", out)
    logger.info("  %s_2d.html  (交互式 2D)", out)
    logger.info("  %s_metrics.json", out)


if __name__ == "__main__":
    main()
