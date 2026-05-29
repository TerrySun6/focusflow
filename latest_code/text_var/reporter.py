"""
PHATE 语义拓扑深度逻辑报告生成器

对文本进行语义编码、PHATE 降维、图聚类、PELT 分段，
并生成交互式 Plotly 可视化报告。

典型用法::

    # 直接传入文本分析
    python reporter.py

    # 或作为模块导入
    from reporter import process_text
    process_text("你的文本内容", doc_title="我的文档")
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from mlx_lm import load
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# 依赖检测（可选第三方库）
# ---------------------------------------------------------------------------

try:
    import gtda  # noqa: F401
    from gtda.homology import VietorisRipsPersistence

    TDA_AVAILABLE: bool = True
except ImportError:
    TDA_AVAILABLE = False

try:
    import phate  # noqa: F401

    PHATE_AVAILABLE: bool = True
except ImportError:
    PHATE_AVAILABLE = False

try:
    import hdbscan  # noqa: F401

    HDBSCAN_AVAILABLE: bool = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import igraph as ig  # noqa: F401
    import leidenalg  # noqa: F401

    LEIDEN_AVAILABLE: bool = True
except ImportError:
    LEIDEN_AVAILABLE = False

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

__all__ = [
    "ReporterError",
    "ReporterConfig",
    "split_text_into_segments",
    "load_or_build_chunks",
    "load_or_embed",
    "compute_coherence",
    "LogicAnalyzer",
    "save_modular_report",
    "process_text",
    "main",
]

# ===================================================================
# 异常类
# ===================================================================


class ReporterError(Exception):
    """reporter 模块的基础异常。"""


class EmbeddingError(ReporterError):
    """语义编码过程中发生的错误。"""


class ConfigurationError(ReporterError):
    """配置相关的错误。"""


class SegmentationError(ReporterError):
    """文本分段相关的错误。"""


# ===================================================================
# 配置
# ===================================================================


@dataclass(frozen=True)
class ReporterConfig:
    """全局配置对象。

    Attributes:
        sbert_model_name: SBERT 嵌入模型名称（MLX 格式）。
        window_size: 滑动窗口大小。
        step_size: 滑动步长。
        enable_tda: 是否启用持续同调分析。
        phate_n_components: PHATE 降维目标维度。
        phate_knn: PHATE 的 KNN 参数。
        phate_knn_dist: KNN 距离度量。
        phate_n_pca: PHATE 的 PCA 预降维维度（None 表示跳过）。
        phate_mds: MDS 初始化方法（"metric" 或 "nonmetric"）。
        phate_mds_solver: MDS 求解器。
        outlier_k_min: 离群分析 KNN 最小 K 值。
        outlier_k_max: 离群分析 KNN 最大 K 值。
        leiden_resolution: Leiden 聚类分辨率参数。
        hdbscan_min_cluster_size: HDBSCAN 最小簇大小。
        hdbscan_min_samples: HDBSCAN 最小样本数。
        device: torch 设备（"mps", "cuda", 或 "cpu"）。
        clear_every: 每 N 个 chunk 清理一次 MLX 显存。
    """

    sbert_model_name: str = "mlx-community/Qwen3-Embedding-0.6B-8bit"
    window_size: int = 5
    step_size: int = 1
    enable_tda: bool = False

    # PHATE
    phate_n_components: int = 3
    phate_knn: int = 7
    phate_knn_dist: str = "cosine"
    phate_n_pca: Optional[int] = None
    phate_mds: str = "metric"
    phate_mds_solver: str = "smacof"

    # 图聚类 / 离群
    outlier_k_min: int = 6
    outlier_k_max: int = 18
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 1

    # 输入输出
    default_input_path: str = "/Users/terrysun/XiaoWangZi.txt"
    default_doc_title: str = "小王子"

    # 运行时
    device: str = field(default_factory=lambda: _detect_device())
    clear_every: int = 10


def _detect_device() -> str:
    """自动检测可用的 torch 设备。"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEFAULT_CONFIG = ReporterConfig()
"""模块级默认配置。可通过 ``ReporterConfig(...)`` 创建自定义配置。"""

# ===================================================================
# 工具类
# ===================================================================


class SimpleTimer:
    """简单的上下文管理器计时器。

    Examples::

        >>> with SimpleTimer("编码阶段"):
        ...     time.sleep(0.5)
    """

    def __init__(self, name: str) -> None:
        """初始化计时器。

        Args:
            name: 阶段名称，用于日志显示。
        """
        self.name = name
        self._start: float = 0.0

    def __enter__(self) -> SimpleTimer:
        """进入上下文，记录开始时间并输出日志。"""
        self._start = time.time()
        logger.info("▶️ [开始] %s...", self.name)
        return self

    def __exit__(self, *args: Any) -> None:
        """退出上下文，计算耗时并输出日志。"""
        elapsed = time.time() - self._start
        logger.info("✅ [完成] %s | 耗时: %.2fs", self.name, elapsed)


# ===================================================================
# 模块 1：文本分段
# ===================================================================


def get_script_dir() -> str:
    """获取当前脚本所在目录。

    Returns:
        脚本所在目录的绝对路径。如果运行在交互式环境中，返回当前工作目录。
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def split_text_into_segments(text: str) -> List[Dict[str, str]]:
    """将文本按句子边界分割为 segments 列表。

    同时支持中英文分割：
    - 中文：按 。！？分割
    - 英文：按 ``.!? + 空格 + 大写字母`` 分割
    - 连续换行也作为分割边界

    Args:
        text: 原始文本字符串。

    Returns:
        segments 列表，每个元素为 ``{"text": "..."}`` 格式。
        若分段数少于 3，则按换行符分割作为回退。

    Examples::

        >>> split_text_into_segments("你好。世界！Hello world. Next.")
        [{'text': '你好。'}, {'text': '世界！'}, {'text': 'Hello world.'}, {'text': 'Next.'}]
    """
    t = text.strip()
    if not t:
        return []

    # 第一步：在中文标点后插入换行符作为分割标记
    # 保留引号内的完整性
    t = re.sub(r'([。！？])(?!["」』》\）\)】\s]*[」』》\）\)】])', r"\1\n", t)
    # 第二步：在英文标点 + 空格 + 大写字母前插入换行符
    t = re.sub(r'([.!?])\s+(?=[A-Z"])', r"\1\n", t)
    # 第三步：合并连续换行
    t = re.sub(r"\n\s*\n", "\n", t)

    sentences = [s.strip() for s in t.split("\n") if s.strip()]
    segments = [{"text": s} for s in sentences]

    # 若分段太少（无标点），按换行符分割作为回退
    if len(segments) < 3:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        segments = [{"text": line} for line in lines]

    return segments


# ===================================================================
# 模块 2：嵌入
# ===================================================================


def load_or_build_chunks(
    segments: List[Dict[str, str]],
    output_dir: str,
    config: ReporterConfig = DEFAULT_CONFIG,
) -> List[str]:
    """滑动窗口构建语义块，支持缓存。

    Args:
        segments: 文本段列表（格式 ``{"text": "..."}``）。
        output_dir: 输出目录，用于缓存 chunks.json。
        config: 配置对象。

    Returns:
        chunk 文本列表。
    """
    chunks_cache = os.path.join(output_dir, "chunks.json")
    if os.path.exists(chunks_cache):
        with open(chunks_cache, "r", encoding="utf-8") as f:
            chunks: List[str] = json.load(f)
        logger.info("✅ 已读取 chunks 缓存: %s", chunks_cache)
        return chunks

    w = config.window_size
    s = config.step_size
    chunks = [
        " ".join([seg["text"].strip() for seg in segments[i : i + w]])
        for i in range(0, len(segments) - w + 1, s)
    ]

    with open(chunks_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info("✅ 已保存 chunks 缓存: %s", chunks_cache)
    return chunks


def load_or_embed(
    chunks: List[str],
    output_dir: str,
    config: ReporterConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """对 chunks 进行语义编码，支持缓存。

    针对 Apple Silicon（M3 8GB）优化的 MLX Qwen3 编码器。
    使用 ``mlx_lm`` 模型进行嵌入提取，支持自动显存清理。

    Args:
        chunks: chunk 文本列表。
        output_dir: 输出目录，用于缓存 ``embeddings.npy``。
        config: 配置对象。

    Returns:
        形状为 ``(len(chunks), hidden_dim)`` 的嵌入张量。

    Raises:
        EmbeddingError: 加载模型或编码过程中发生错误。
    """
    emb_cache = os.path.join(output_dir, "embeddings.npy")

    # 检查缓存
    if os.path.exists(emb_cache):
        embeddings = np.load(emb_cache)
        logger.info(
            "✅ 已读取 embeddings 缓存: %s | 形状: %s", emb_cache, embeddings.shape
        )
        return torch.tensor(embeddings, dtype=torch.float32)

    logger.info("🚀 正在加载模型: %s", config.sbert_model_name)

    try:
        model, tokenizer = load(config.sbert_model_name)
    except ImportError as exc:
        msg = "未安装 mlx-lm。请运行: pip install mlx-lm"
        logger.error("❌ %s", msg)
        raise EmbeddingError(msg) from exc
    except Exception as exc:
        msg = f"模型加载失败: {config.sbert_model_name}"
        raise EmbeddingError(msg) from exc

    all_embeddings: List[np.ndarray] = []
    total_chunks = len(chunks)

    logger.info("🧪 开始编码 %d 个语义块...", total_chunks)

    try:
        for i, chunk in enumerate(chunks):
            # 编码文本（加上指令引导以优化逻辑表征）
            full_prompt = (
                f"Represent this passage for logical structure analysis: {chunk}"
            )
            tokens = mx.array(tokenizer.encode(full_prompt))

            # 前向传播：获取隐藏层状态 [1, seq_len, 1024]
            output: Any = model.model(tokens[None])

            # 平均池化：将序列压缩为向量 [1024]
            emb: mx.array = mx.mean(output, axis=1)
            mx.eval(emb)

            # 转换为 numpy
            all_embeddings.append(np.array(emb, dtype=np.float32))

            # 定期清理显存
            if (i + 1) % config.clear_every == 0:
                mx.metal.clear_cache()
                gc.collect()
                logger.info("  进度: %d/%d | 内存已回收", i + 1, total_chunks)

        # 合并所有向量
        embeddings = np.vstack(all_embeddings)

        # 保存结果
        np.save(emb_cache, embeddings)
        logger.info("✅ 编码完成，缓存已保存: %s", emb_cache)

        return torch.tensor(embeddings, dtype=torch.float32)

    except Exception as exc:
        logger.error("❌ 编码过程中发生错误: %s", exc)
        raise EmbeddingError("语义编码失败") from exc

    finally:
        # 释放模型显存
        if "model" in locals():
            del model  # type: ignore[assignment]
        gc.collect()
        mx.metal.clear_cache()


def compute_coherence(
    embeddings_tensor: torch.Tensor,
) -> Tuple[float, np.ndarray]:
    """计算相邻语义块之间的余弦相似度（连贯性）。

    Args:
        embeddings_tensor: 嵌入张量，形状 ``(N, D)``。

    Returns:
        ``(平均连贯性, 逐点连贯性数组)`` 的二元组。
    """
    norm = F.normalize(embeddings_tensor, p=2, dim=1)
    sims = (norm[:-1] * norm[1:]).sum(dim=1).cpu().numpy()
    return (float(np.mean(sims)), sims)


# ===================================================================
# 模块 3：分析
# ===================================================================


class LogicAnalyzer:
    """语义逻辑分析器。

    封装 PHATE 降维、TDA 拓扑分析、图聚类和 PELT 语义分段功能。

    Attributes:
        results: 所有分析结果的字典。
    """

    def __init__(
        self,
        embeddings_tensor: torch.Tensor,
        config: ReporterConfig = DEFAULT_CONFIG,
    ) -> None:
        """初始化 LogicAnalyzer。

        Args:
            embeddings_tensor: 嵌入张量，形状 ``(N, D)``。
            config: 分析配置。
        """
        self.embeddings_gpu: torch.Tensor = embeddings_tensor
        self.embeddings_cpu: np.ndarray = embeddings_tensor.cpu().numpy()
        self.config = config
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # PHATE
    # ------------------------------------------------------------------

    def run_phate(self) -> None:
        """运行 PHATE 降维。

        将高维语义嵌入降至 3 维，保存轨迹坐标至 ``self.results["phate"]``。
        """
        if not PHATE_AVAILABLE:
            logger.warning("⚠️ PHATE 不可用，跳过")
            return

        cfg = self.config
        op = phate.PHATE(
            n_components=cfg.phate_n_components,
            knn=cfg.phate_knn,
            knn_dist=cfg.phate_knn_dist,
            n_pca=cfg.phate_n_pca,
            mds=cfg.phate_mds,
            mds_solver=cfg.phate_mds_solver,
            n_jobs=-1,
            verbose=False,
        )
        self.results["phate"] = op.fit_transform(self.embeddings_cpu)
        self.results["phate_n_pca"] = cfg.phate_n_pca
        self.results["phate_params"] = {
            "n_components": cfg.phate_n_components,
            "knn": cfg.phate_knn,
            "knn_dist": cfg.phate_knn_dist,
            "n_pca": cfg.phate_n_pca,
            "mds": cfg.phate_mds,
            "mds_solver": cfg.phate_mds_solver,
        }
        logger.info("✅ PHATE 降维完成")

    # ------------------------------------------------------------------
    # TDA
    # ------------------------------------------------------------------

    def run_tda(self) -> None:
        """运行持续同调（TDA）分析。

        计算嵌入的余弦距离矩阵并应用 Vietoris-Rips 持续同调。
        结果写入 ``self.results["tda"]``。
        """
        if not TDA_AVAILABLE:
            logger.warning("⚠️ TDA 不可用，跳过")
            return

        norm = F.normalize(self.embeddings_gpu, p=2, dim=1)
        dist_matrix = (
            (1.0 - torch.clamp(torch.mm(norm, norm.t()), -1.0, 1.0)).cpu().numpy()
        )
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.maximum(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        vr = VietorisRipsPersistence(
            metric="precomputed",
            homology_dimensions=[0, 1, 2],
            collapse_edges=True,
            n_jobs=-1,
        )
        dgms = vr.fit_transform(dist_matrix[np.newaxis, :, :])
        self.results["tda"] = {
            "diagrams": dgms[0],
            "betti": {0: 0, 1: 0},
        }
        logger.info("✅ TDA 分析完成")

    # ------------------------------------------------------------------
    # 图聚类
    # ------------------------------------------------------------------

    @staticmethod
    def _knn_graph(
        emb: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建 KNN 图。

        Args:
            emb: 嵌入数组，形状 ``(N, D)``。
            k: 邻居数量。

        Returns:
            ``(距离矩阵, 索引矩阵)`` 的二元组。均不包含自身。
        """
        nn = NearestNeighbors(
            n_neighbors=min(k + 1, len(emb)),
            metric="cosine",
        )
        nn.fit(emb)
        dists, inds = nn.kneighbors(emb)
        return dists[:, 1:], inds[:, 1:]

    def run_graph_clustering(self) -> None:
        """运行图聚类（Leiden / HDBSCAN）和离群分析。

        优先使用 Leiden 算法（如果 ``igraph`` + ``leidenalg`` 可用），
        否则回退到 HDBSCAN。结果写入 ``self.results["clusters"]`` 等。
        """
        emb = self.embeddings_cpu
        n = len(emb)

        # 边界情况：空或极短
        if n == 0:
            self.results["clusters"] = np.array([], dtype=int)
            self.results["clusters_raw"] = np.array([], dtype=int)
            self.results["outliers"] = np.array([], dtype=float)
            self.results["cluster_backend"] = "empty"
            return
        if n < 3:
            labels = np.zeros(n, dtype=int)
            self.results["clusters_raw"] = labels
            self.results["clusters"] = labels
            self.results["outliers"] = np.zeros(n, dtype=float)
            self.results["cluster_backend"] = "short"
            return

        # 归一化并计算离群度
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        cfg = self.config
        k = int(np.clip(np.sqrt(max(n, 4)), cfg.outlier_k_min, cfg.outlier_k_max))
        dists, inds = self._knn_graph(emb_norm, k)
        sims = np.clip(1.0 - dists, 0.0, 1.0)
        outliers = 1.0 - sims.mean(axis=1)

        labels_raw: Optional[np.ndarray] = None
        backend: str

        if LEIDEN_AVAILABLE:
            labels_raw, backend = self._run_leiden(n, inds, sims)
        elif HDBSCAN_AVAILABLE:
            labels_raw, outliers, backend = self._run_hdbscan(emb_norm)
        else:
            labels_raw = np.zeros(n, dtype=int)
            backend = "fallback-single"

        self.results["clusters_raw"] = labels_raw
        self.results["clusters"] = labels_raw
        self.results["outliers"] = outliers
        self.results["cluster_backend"] = backend
        self.results["outlier_k"] = k

        logger.info("✅ 图聚类完成 (backend=%s, k=%d)", backend, k)

    def _run_leiden(
        self,
        n: int,
        inds: np.ndarray,
        sims: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """使用 Leiden 算法进行图聚类。

        Args:
            n: 节点数。
            inds: KNN 索引矩阵。
            sims: 相似度矩阵。

        Returns:
            ``(标签数组, "leiden")``。
        """
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        seen: set = set()

        for i in range(n):
            for j, sim in zip(inds[i], sims[i]):
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                edges.append((a, b))
                weights.append(float(sim))

        g = ig.Graph(n=n, edges=edges, directed=False)
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=self.config.leiden_resolution,
        )
        return np.array(part.membership, dtype=int), "leiden"

    def _run_hdbscan(
        self,
        emb_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """使用 HDBSCAN 进行聚类（回退方案）。

        Args:
            emb_norm: L2 归一化后的嵌入。

        Returns:
            ``(标签数组, 离群分数组, "hdbscan")``。
        """
        cfg = self.config
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
        )
        labels_raw = clusterer.fit_predict(emb_norm)
        outliers = clusterer.outlier_scores_
        return labels_raw, outliers, "hdbscan"

    # ------------------------------------------------------------------
    # PELT 分段
    # ------------------------------------------------------------------

    def run_pelt_segmentation(self) -> None:
        """运行 PELT 变点检测算法进行语义分段。

        流程：
        1. 计算相邻 chunk 的归一化 L2 距离
        2. 根据位移中值+标准差动态计算 penalty
        3. 使用 ``ruptures.Pelt`` 检测变点
        4. 合并相邻且主题相同的碎片段

        结果写入 ``self.results["segments"]``。
        """
        emb = self.embeddings_cpu
        n = len(emb)

        if n < 5:
            self.results["segments"] = {
                "boundaries": [0, n],
                "topics": [0],
                "segment_ids": np.zeros(n, dtype=int),
                "method": "too_short",
            }
            return

        # 1. 归一化计算距离
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        distances = np.linalg.norm(emb_norm[1:] - emb_norm[:-1], axis=1)
        median_dist = float(np.median(distances))
        std_dist = float(np.std(distances))

        # 2. 动态 penalty（保持高敏感度）
        sensitivity_factor = 0.5
        penalty = (median_dist + std_dist) * sensitivity_factor

        logger.info(
            "📊 PELT 分段校准：中值位移=%.4f, 动态 Penalty=%.4f",
            median_dist,
            penalty,
        )

        # 3. PELT 分割
        try:
            import ruptures as rpt

            algo = rpt.Pelt(model="rbf", min_size=3).fit(emb_norm)
            raw_cuts = algo.predict(pen=penalty)
            raw_boundaries = [0] + sorted(set(raw_cuts))
        except Exception as exc:
            logger.warning("⚠️ RBF 模型分割失败，回退到 L2: %s", exc)
            rpt_l2 = rpt.Pelt(model="l2", min_size=3).fit(emb_norm)
            raw_cuts = rpt_l2.predict(pen=penalty * 0.5)
            raw_boundaries = [0] + sorted(set(raw_cuts))

        if raw_boundaries[-1] != n:
            raw_boundaries.append(n)

        # 4. 获取每一小段的主题簇
        clusters = np.array(
            self.results.get("clusters", np.zeros(n, dtype=int)),
        )
        raw_topics: List[int] = []
        for i in range(len(raw_boundaries) - 1):
            left, r = raw_boundaries[i], raw_boundaries[i + 1]
            vals = clusters[left:r]
            u, c = np.unique(vals, return_counts=True) if len(vals) > 0 else ([0], [0])
            raw_topics.append(int(u[np.argmax(c)]))

        # 5. 合并相邻且主题相同的碎片
        final_boundaries = [raw_boundaries[0]]
        final_topics: List[int] = []

        if raw_topics:
            current_topic = raw_topics[0]
            for i in range(1, len(raw_topics)):
                if raw_topics[i] != current_topic:
                    final_boundaries.append(raw_boundaries[i])
                    final_topics.append(current_topic)
                    current_topic = raw_topics[i]
            final_boundaries.append(raw_boundaries[-1])
            final_topics.append(current_topic)
        else:
            final_boundaries = raw_boundaries
            final_topics = raw_topics

        self.results["segments"] = {
            "boundaries": final_boundaries,
            "topics": final_topics,
            "segment_ids": _update_segment_ids(n, final_boundaries),
            "method": "pelt_adaptive_merged",
        }
        logger.info("✅ PELT 分段完成: %d 个段落", len(final_topics))


def _update_segment_ids(
    n: int,
    boundaries: Sequence[Union[int, float]],
) -> np.ndarray:
    """根据分段边界生成段落 ID 数组。

    Args:
        n: 总长度。
        boundaries: 边界列表。

    Returns:
        形状 ``(n,)`` 的整数数组，每个位置标记所属段 ID。
    """
    seg_ids = np.zeros(n, dtype=int)
    for i in range(len(boundaries) - 1):
        left = int(boundaries[i])
        r = min(int(boundaries[i + 1]), n)
        seg_ids[left:r] = i
    return seg_ids


# ===================================================================
# 模块 4：可视化
# ===================================================================


def smooth_3d_curve(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin 角切割算法，用于平滑 3D 轨迹曲线。

    Args:
        points: 原始点序列，形状 ``(N, 3)``。
        iterations: 平滑迭代次数。

    Returns:
        平滑后的点序列。
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or iterations <= 0:
        return pts

    sm = pts.copy()
    for _ in range(iterations):
        out = [sm[0]]
        for i in range(len(sm) - 1):
            p = sm[i]
            q = sm[i + 1]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        out.append(sm[-1])
        sm = np.asarray(out, dtype=np.float64)
    return sm


_PALETTE: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
"""Plotly 颜色调色板，用于段落着色。"""


def save_modular_report(
    chunks: List[str],
    analyzer: LogicAnalyzer,
    coh_data: Tuple[float, np.ndarray],
    output_dir: str,
    embeddings_tensor: torch.Tensor,
    doc_title: str = "text_document",
    report_name: str = "phate_logic_report.html",
) -> str:
    """生成交互式 Plotly HTML 报告。

    报告包含以下部分（按可用性动态调整）：
    - PHATE 3D 逻辑流轨迹
    - 密度离群分析
    - 语义连贯性实时演变
    - 持续性图（TDA，可选）
    - PHATE 轨迹特征量化分析
    - Chunk 文本表
    - 分段主题簇表
    - 参数清单

    Args:
        chunks: chunk 文本列表。
        analyzer: LogicAnalyzer 实例（含结果）。
        coh_data: ``compute_coherence`` 的返回值。
        output_dir: 报告输出目录。
        embeddings_tensor: 嵌入张量。
        doc_title: 文档标题。
        report_name: 报告文件名。

    Returns:
        生成的报告文件绝对路径。
    """
    res = analyzer.results

    emb = embeddings_tensor.detach().cpu().numpy()
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    # 计算主题相关性 R(t)
    phi = 0.7
    c_t = [emb_norm[0]]
    for i in range(1, len(emb_norm)):
        c_t.append(phi * c_t[-1] + (1.0 - phi) * emb_norm[i])
    c_t = np.array(c_t)
    c_t = c_t / (np.linalg.norm(c_t, axis=1, keepdims=True) + 1e-9)
    R = np.sum(emb_norm * c_t, axis=1)

    seg_info = res.get("segments", {})
    seg_boundaries: List[Union[int, float]] = seg_info.get(
        "boundaries", [0, len(chunks)]
    )
    seg_topics: List[int] = seg_info.get("topics", [])

    # 构建子图布局
    specs: List[List[Optional[Dict[str, Any]]]] = [
        [{"type": "scene"}, {"type": "xy"}],
        [{"type": "xy", "colspan": 2}, None],
    ]
    titles: List[str] = [
        "PHATE 逻辑流轨迹 (PELT 分段 + Leiden 段内主题)",
        "密度离群分析",
        "语义连贯性实时演变",
    ]

    if "tda" in res:
        specs.append([{"type": "xy"}, {"type": "domain"}])
        titles.extend(["持续性图 (TDA)", "拓扑逻辑摘要"])
    if "phate" in res:
        specs.append([{"type": "domain", "colspan": 2}, None])
        titles.append("PHATE 轨迹特征量化分析")

    specs.append([{"type": "table", "colspan": 2}, None])
    titles.append("Chunk 文本")
    specs.append([{"type": "table", "colspan": 2}, None])
    titles.append("分段主题簇")
    specs.append([{"type": "table", "colspan": 2}, None])
    titles.append("参数清单")

    fig = make_subplots(
        rows=len(specs),
        cols=2,
        column_widths=[0.6, 0.4],
        specs=specs,
        subplot_titles=titles,
    )
    phate_point_trace_indices: List[int] = []

    # ---- PHATE 3D 轨迹 ----
    if "phate" in res:
        p_coords: np.ndarray = res["phate"]
        smooth_coords = smooth_3d_curve(p_coords, iterations=2)

        # 全局轨迹线
        fig.add_trace(
            go.Scatter3d(
                x=smooth_coords[:, 0],
                y=smooth_coords[:, 1],
                z=smooth_coords[:, 2],
                mode="lines",
                line=dict(color="rgba(60,60,60,0.45)", width=2),
                name="全局轨迹",
            ),
            row=1,
            col=1,
        )

        # 分割点连线
        if len(seg_boundaries) >= 2:
            b_idx = [max(0, min(int(b), len(p_coords) - 1)) for b in seg_boundaries]
            fig.add_trace(
                go.Scatter3d(
                    x=p_coords[b_idx, 0],
                    y=p_coords[b_idx, 1],
                    z=p_coords[b_idx, 2],
                    mode="lines+markers",
                    line=dict(color="rgba(0,0,0,0.6)", width=2.5),
                    marker=dict(size=4, color="black", symbol="x"),
                    name="分割点连线",
                ),
                row=1,
                col=1,
            )

        # 逐段落着色
        n_seg = max(0, len(seg_boundaries) - 1)
        for s in range(n_seg):
            left, r = int(seg_boundaries[s]), int(seg_boundaries[s + 1])
            idx = np.arange(left, r, dtype=int)
            if len(idx) <= 0:
                continue
            color = _PALETTE[s % len(_PALETTE)]
            topic = seg_topics[s] if s < len(seg_topics) else -1
            hover_data = [[int(j + 1)] for j in idx]
            fig.add_trace(
                go.Scatter3d(
                    x=p_coords[idx, 0],
                    y=p_coords[idx, 1],
                    z=p_coords[idx, 2],
                    mode="markers",
                    marker=dict(size=3, color=color, opacity=0.7),
                    customdata=hover_data,
                    hovertemplate="语义块 %{customdata[0]}<extra></extra>",
                    name=f"段 {s + 1} | 主题簇 {int(topic)}",
                ),
                row=1,
                col=1,
            )
            phate_point_trace_indices.append(len(fig.data) - 1)

        # 起止标记
        fig.add_trace(
            go.Scatter3d(
                x=[p_coords[0, 0]],
                y=[p_coords[0, 1]],
                z=[p_coords[0, 2]],
                mode="markers+text",
                marker=dict(size=8, color="red", symbol="diamond"),
                text=["START"],
                name="起点",
            ),
            row=1,
            col=1,
        )
        phate_point_trace_indices.append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter3d(
                x=[p_coords[-1, 0]],
                y=[p_coords[-1, 1]],
                z=[p_coords[-1, 2]],
                mode="markers+text",
                marker=dict(size=8, color="green", symbol="circle"),
                text=["END"],
                name="终点",
            ),
            row=1,
            col=1,
        )
        phate_point_trace_indices.append(len(fig.data) - 1)

    # ---- 离群分析 ----
    fig.add_trace(
        go.Scatter(
            y=res.get("outliers", []),
            mode="markers",
            marker=dict(color="rgba(255,0,0,0.5)", size=3),
            name="离群度",
        ),
        row=1,
        col=2,
    )

    # ---- 连贯性 ----
    fig.add_trace(
        go.Scatter(
            y=coh_data[1],
            mode="lines",
            fill="tozeroy",
            name="连贯性",
        ),
        row=2,
        col=1,
    )

    curr_row = 3

    # ---- TDA ----
    if "tda" in res:
        dg = res["tda"]["diagrams"]
        for dim in [0, 1, 2]:
            pts = dg[dg[:, 2] == dim]
            if len(pts) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        name=f"H{dim}",
                    ),
                    row=curr_row,
                    col=1,
                )
        fig.add_trace(
            go.Table(
                header=dict(values=["维度", "含义"]),
                cells=dict(
                    values=[
                        ["H0", "H1", "H2"],
                        ["主题独立性", "逻辑闭环", "高阶空洞/多主题交错"],
                    ]
                ),
            ),
            row=curr_row,
            col=2,
        )
        curr_row += 1

    # ---- 量化指标 ----
    if "phate" in res:
        cos_sim = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        d_t = np.arccos(cos_sim)
        L_vel = float(d_t.mean()) if len(d_t) > 0 else 0.0
        mu_v = emb.mean(axis=0)
        L_vol = float(np.sqrt(((emb - mu_v) ** 2).sum(axis=1).mean()))
        S_coh = float(np.mean(cos_sim)) if len(cos_sim) > 0 else 0.0
        alpha = 0.5
        beta = 0.5

        z_vel = (L_vel - np.mean(d_t)) / (np.std(d_t) + 1e-9) if len(d_t) > 0 else 0.0
        vol_series = np.sqrt(((emb - mu_v) ** 2).sum(axis=1))
        z_vol = (
            (L_vol - np.mean(vol_series)) / (np.std(vol_series) + 1e-9)
            if len(vol_series) > 0
            else 0.0
        )
        score_load = float(S_coh * (alpha * z_vel + beta * z_vol))

        lam = 0.5
        D_shock = float(np.mean(np.abs(R[1:] - R[:-1]))) if len(R) > 1 else 0.0
        mu_R = float(np.mean(R)) if len(R) > 0 else 0.0
        raw_stab = mu_R - lam * D_shock
        z_stab = (
            float((raw_stab - np.mean(R)) / (np.std(R) + 1e-9)) if len(R) > 0 else 0.0
        )

        fig.add_trace(
            go.Scatter(y=R, mode="lines", name="主题相关性"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["分析指标", "量化数值", "深度解释"],
                    fill_color="darkred",
                    font=dict(color="white"),
                ),
                cells=dict(
                    values=[
                        [
                            "语义推进速率",
                            "语义空间体积",
                            "语义连贯性",
                            "逻辑负载度综合得分",
                            "主题相关性均值",
                            "稳定性得分",
                        ],
                        [
                            f"{L_vel:.4f}",
                            f"{L_vol:.4f}",
                            f"{S_coh:.4f}",
                            f"{score_load:.4f}",
                            f"{mu_R:.4f}",
                            f"{z_stab:.4f}",
                        ],
                        [
                            r"$L_{vel} = \frac{1}{N-1}\sum_{t=1}^{N-1} \arccos(\cos_t)$",
                            r"$L_{vol} = \sqrt{\frac{1}{N}\sum_{t=1}^{N} \lVert v_t-\mu \rVert^2}$",
                            r"$S_{coh} = \frac{1}{N-1}\sum_{t=1}^{N-1} \cos_t$",
                            r"$Score = S_{coh}\cdot(0.5\,Z(L_{vel})+0.5\,Z(L_{vol}))$",
                            r"$\mu_R = \frac{1}{N}\sum_{t=1}^{N} R(t)$",
                            r"$S_{stab} = Z(\mu_R - 0.5\,D_{shock})$",
                        ],
                    ]
                ),
            ),
            row=curr_row,
            col=1,
        )
        curr_row += 1

    # ---- Chunk 文本表 ----
    chunk_idx = list(range(1, len(chunks) + 1))
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Chunk", "文本"],
                fill_color="darkslategray",
                font=dict(color="white"),
            ),
            cells=dict(values=[chunk_idx, chunks], align=["center", "left"]),
        ),
        row=curr_row,
        col=1,
    )
    curr_row += 1

    # ---- 分段主题簇表 ----
    seg_rows = list(range(1, max(1, len(seg_boundaries))))
    seg_ranges: List[str] = []
    seg_topic_vals: List[int] = []
    seg_sizes: List[int] = []
    for i in range(max(0, len(seg_boundaries) - 1)):
        left, r = int(seg_boundaries[i]), int(seg_boundaries[i + 1])
        seg_ranges.append(f"[{left}, {r})")
        seg_topic_vals.append(
            int(seg_topics[i]) if i < len(seg_topics) else -1,
        )
        seg_sizes.append(max(0, r - left))
    fig.add_trace(
        go.Table(
            header=dict(
                values=["段", "区间", "topic_cluster", "长度"],
                fill_color="darkslateblue",
                font=dict(color="white"),
            ),
            cells=dict(
                values=[seg_rows, seg_ranges, seg_topic_vals, seg_sizes],
                align=["center", "center", "center", "center"],
            ),
        ),
        row=curr_row,
        col=1,
    )
    curr_row += 1

    # ---- 参数清单表 ----
    cfg = analyzer.config
    phate_params = res.get("phate_params", {})
    param_names: List[str] = [
        "Input type",
        "SBERT_MODEL_NAME",
        "WINDOW_SIZE",
        "STEP_SIZE",
        "PHATE.n_components",
        "PHATE.knn",
        "PHATE.knn_dist",
        "PHATE.n_pca",
        "PHATE.mds",
        "PHATE.mds_solver",
        "Outlier space",
        "Outlier graph k(actual)",
        "Outlier graph k rule",
        "Cluster backend",
        "Leiden resolution",
        "HDBSCAN.min_cluster_size",
        "HDBSCAN.min_samples",
        "Segmentation method",
        "ENABLE_TDA",
    ]
    param_values: List[str] = [
        "text (direct input)",
        cfg.sbert_model_name,
        str(cfg.window_size),
        str(cfg.step_size),
        str(phate_params.get("n_components", cfg.phate_n_components)),
        str(phate_params.get("knn", cfg.phate_knn)),
        str(phate_params.get("knn_dist", cfg.phate_knn_dist)),
        str(phate_params.get("n_pca", cfg.phate_n_pca)),
        str(phate_params.get("mds", cfg.phate_mds)),
        str(phate_params.get("mds_solver", cfg.phate_mds_solver)),
        "embedding (L2 normalized)",
        str(res.get("outlier_k", "n/a")),
        f"k=clip(sqrt(n), {cfg.outlier_k_min}, {cfg.outlier_k_max})",
        str(res.get("cluster_backend", "unknown")),
        str(cfg.leiden_resolution),
        str(cfg.hdbscan_min_cluster_size),
        str(cfg.hdbscan_min_samples),
        str(seg_info.get("method", "unknown")),
        str(cfg.enable_tda),
    ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["参数", "值"],
                fill_color="darkred",
                font=dict(color="white"),
            ),
            cells=dict(
                values=[param_names, param_values],
                align=["left", "left"],
            ),
        ),
        row=curr_row,
        col=1,
    )

    # ---- 显示/隐藏点按钮 ----
    n_traces = len(fig.data)
    vis_points_on = [True] * n_traces
    vis_points_off = [True] * n_traces
    for idx in phate_point_trace_indices:
        if 0 <= idx < n_traces:
            vis_points_off[idx] = False

    fig.update_layout(
        height=450 * len(specs),
        title=f"📄 PHATE 语义拓扑深度逻辑报告 - {doc_title}",
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.08,
                showactive=True,
                buttons=[
                    dict(
                        label="显示点",
                        method="update",
                        args=[{"visible": vis_points_on}],
                    ),
                    dict(
                        label="隐藏点",
                        method="update",
                        args=[{"visible": vis_points_off}],
                    ),
                ],
            ),
        ],
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False, gridcolor="#d9e2ef"),
            yaxis=dict(showbackground=False, gridcolor="#d9e2ef"),
            zaxis=dict(showbackground=False, gridcolor="#d9e2ef"),
            camera=dict(eye=dict(x=1.3, y=1.25, z=0.9)),
        ),
    )

    report_path = os.path.join(output_dir, report_name)
    fig.write_html(report_path, include_mathjax="cdn")
    logger.info("📊 报告已生成: %s", report_path)
    return report_path


# ===================================================================
# 入口
# ===================================================================


def process_text(
    text: str,
    doc_title: str = "my_document",
    config: Optional[ReporterConfig] = None,
) -> str:
    """处理输入文字，生成语义逻辑分析报告。

    Args:
        text: 要分析的文字内容。
        doc_title: 文档标题，用于输出目录和报告标题。
        config: 配置对象。默认为 ``DEFAULT_CONFIG``。

    Returns:
        生成的报告文件路径。

    Raises:
        ReporterError: 处理过程中发生错误。

    Examples::

        >>> report_path = process_text("你好，世界。", doc_title="test")
        >>> print(report_path)
        /path/to/test/phate_logic_report.html
    """
    cfg = config or DEFAULT_CONFIG
    script_dir = get_script_dir()
    safe_title = (
        doc_title.replace(" ", "_").replace("/", "_").replace("\\", "_")[:120]
        or "text_document"
    )
    output_dir = os.path.join(script_dir, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with SimpleTimer("文字分段"):
            segments = split_text_into_segments(text)
            logger.info("  共分割为 %d 个语义段", len(segments))

        chunks = load_or_build_chunks(segments, output_dir, cfg)

        with SimpleTimer("SBERT 编码"):
            embeddings_tensor = load_or_embed(chunks, output_dir, cfg)

        analyzer = LogicAnalyzer(embeddings_tensor, cfg)

        with SimpleTimer("PHATE 轨迹生成 (高维直投)"):
            analyzer.run_phate()

        if TDA_AVAILABLE and cfg.enable_tda:
            with SimpleTimer("TDA 拓扑分析"):
                analyzer.run_tda()

        with SimpleTimer("图聚类 (Leiden/HDBSCAN)"):
            analyzer.run_graph_clustering()

        with SimpleTimer("PELT 语义分段"):
            analyzer.run_pelt_segmentation()

        with SimpleTimer("连贯性计算"):
            coh_data = compute_coherence(embeddings_tensor)

        return save_modular_report(
            chunks=chunks,
            analyzer=analyzer,
            coh_data=coh_data,
            output_dir=output_dir,
            embeddings_tensor=embeddings_tensor,
            doc_title=doc_title,
        )

    except ReporterError:
        raise
    except Exception as exc:
        raise ReporterError(f"文本处理失败: {exc}") from exc


def main() -> None:
    """CLI 入口。

    支持以下调用方式::

        python reporter.py                     # 使用环境变量 INPUT_FILE_PATH (或有 --ui 启动图形界面)
        python reporter.py /path/to/file.txt   # 直接指定文件路径
        python reporter.py --ui                # 启动图形界面

    环境变量:
        INPUT_FILE_PATH — 输入文件路径（默认取自 ReporterConfig.default_input_path）
        DOC_TITLE — 文档标题（默认取自 ReporterConfig.default_doc_title）
    """
    # 检查是否请求 UI
    if "--ui" in sys.argv:
        try:
            from reporter_ui import main as ui_main

            ui_main()
        except ImportError:
            logger.error("❌ reporter_ui.py 未找到，无法启动图形界面。")
        return

    # 检查是否通过命令行传入了文件路径
    cli_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if cli_args:
        input_path = cli_args[0]
        doc_title = (
            os.environ.get("DOC_TITLE")
            or os.path.splitext(os.path.basename(input_path))[0]
        )
    else:
        input_path = os.environ.get("INPUT_FILE_PATH", DEFAULT_CONFIG.default_input_path)
        doc_title = os.environ.get("DOC_TITLE", DEFAULT_CONFIG.default_doc_title)

    if not os.path.exists(input_path):
        logger.error("❌ 输入文件不存在: %s", input_path)
        logger.error(
            "使用方式: python reporter.py <文件路径>   或设置环境变量 INPUT_FILE_PATH"
        )
        return

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info("✅ 已读取输入文件: %s", input_path)

    process_text(text, doc_title=doc_title)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
