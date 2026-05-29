"""
EMA 增强版文本语义拓扑分析器。

使用指数移动平均（Exponential Moving Average）对单句 embedding 序列做时序平滑，
替代传统滑动窗口（文本拼接）和 GNN 图拓扑两类上下文融合方案。

优势：
- EMA 是轻量级 O(n) 操作，不引入额外参数
- 保留严格的时间顺序，无"花瓣折返"现象
- 语义分段边界更锐利：EMA 让相似主题的句子融合、不同主题的句子区分度放大
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import ruptures as rpt
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Logging 配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# scikit-learn 兼容性补丁
# ---------------------------------------------------------------------------
try:
    import sklearn.utils.validation as _skval

    if getattr(_skval.check_array, "_is_patched", False):
        importlib.reload(_skval)
    if not hasattr(_skval, "_original_check_array"):
        _skval._original_check_array = _skval.check_array

    def _patched_check_array(
        array: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """移除 force_all_finite 参数以兼容新版 numpy."""
        kwargs.pop("force_all_finite", None)
        return _skval._original_check_array(array, *args, **kwargs)

    _patched_check_array._is_patched = True  # type: ignore[attr-defined]
    if _skval.check_array is not _patched_check_array:
        _skval.check_array = _patched_check_array
        logger.info("已应用 scikit-learn 兼容性补丁")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 依赖检测（惰性导入）
# ---------------------------------------------------------------------------
_TDA_AVAILABLE: bool
try:
    import gtda  # noqa: F401
    from gtda.homology import VietorisRipsPersistence  # noqa: F401

    _TDA_AVAILABLE = True
except ImportError:
    _TDA_AVAILABLE = False
    logger.warning("未检测到 gtda，TDA 模块将跳过")

_PHATE_AVAILABLE: bool
try:
    import phate  # noqa: F401

    _PHATE_AVAILABLE = True
except ImportError:
    _PHATE_AVAILABLE = False
    logger.warning("未检测到 phate，PHATE 模块将跳过")

_HDBSCAN_AVAILABLE: bool
try:
    import hdbscan  # noqa: F401

    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    logger.warning("未检测到 hdbscan，离群分析将跳过")

_LEIDEN_AVAILABLE: bool
try:
    import igraph as ig  # noqa: F401
    import leidenalg  # noqa: F401

    _LEIDEN_AVAILABLE = True
except ImportError:
    _LEIDEN_AVAILABLE = False
    logger.warning("未检测到 igraph/leidenalg，将回退 HDBSCAN 聚类")

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """分析器的全部可调参数，集中管理便于实验追踪。

    EMA 模式下 ``window_size=1``（单句），依靠 EMA 时序平滑替代上下文拼接。
    """

    # --- 输入 ---
    input_file_path: str = "/Users/terrysun/Documents/learning/project/focusflow/latest_code/XiaoWangZi.txt"
    doc_title: str = "the_little_prince"

    # --- 编码 ---
    sbert_model_name: str = "BAAI/bge-m3"
    device: str = field(
        default_factory=lambda: "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- 窗口（已禁用，单句模式）---
    window_size: int = 1
    step_size: int = 1

    # --- EMA 时序平滑 ---
    ema_alpha: float = 0.5
    r"""EMA 平滑系数。 

    ``refined[i] = (1 - α) · emb[i] + α · refined[i-1]``

    - α → 0：``refined[i] = emb[i]``，不平滑，保留原始 SBERT 逐句精度
    - α → 1：``refined[i] = refined[i-1]``，极端平滑，序列被卡死在首个值
    - 建议范围 **0.2 ~ 0.6**，兼顾局部细节与全局趋势
    """
    ema_bidirectional: bool = True
    r"""是否使用双向 EMA。

    ``True``（默认）：
        分别执行前向和后向 EMA，然后取均值。
        每个位置同时融合过去和未来的上下文信息，消除单向 EMA 的相位偏移，
        让平滑后的 embedding 在主题切换点更锐利。

    ``False``：
        使用传统单向 EMA（仅前向），等效于低通滤波。
    """


    # --- PHATE ---
    phate_n_components: int = 3
    phate_knn: int = 5
    phate_knn_dist: str = "cosine"
    phate_n_pca: Optional[int] = None
    phate_mds: str = "metric"
    phate_mds_solver: str = "smacof"
    phate_auto_pca_cap: int = 100

    # --- 离群 / 图聚类 ---
    outlier_k_min: int = 6
    outlier_k_max: int = 18
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 1

    # --- 开关 ---
    enable_tda: bool = False


# 全局单例配置
CFG = Config()


# ==================== 工具类 ====================


class SimpleTimer:
    """上下文管理器，用于计时并输出耗时信息。"""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.start: float = 0.0

    def __enter__(self) -> SimpleTimer:
        self.start = time.perf_counter()
        logger.info("▶️ [开始] %s ...", self.name)
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - self.start
        logger.info("✅ [完成] %s | 耗时: %.2fs", self.name, elapsed)


# ==================== 模块 1：文字分段 ====================


def get_script_dir() -> str:
    """返回本脚本所在目录的绝对路径。"""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def split_text_into_segments(text: str) -> List[Dict[str, str]]:
    """将输入文字按句子边界分割为 segment 列表。

    支持中英文混合文本：
    - 中文：按 。！？分割，并尝试保留引号内的完整性
    - 英文：按 ``. ! ?`` + 空格 + 大写字母 分割
    - 连续换行也作为分割边界

    Args:
        text: 原始文本字符串。

    Returns:
        每个元素为 ``{"text": <句子>}`` 的列表。
    """
    t = text.strip()

    t = re.sub(r'([。！？])(?!["」』》\）\)】\s]*[」』》\）\)】])', r"\1\n", t)
    t = re.sub(r"([.!?])\s+(?=[A-Z\"])", r"\1\n", t)
    t = re.sub(r"\n\s*\n", "\n", t)

    sentences = [s.strip() for s in t.split("\n") if s.strip()]
    segments: List[Dict[str, str]] = [{"text": s} for s in sentences]

    if len(segments) < 3:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        segments = [{"text": line} for line in lines]

    return segments


# ==================== 模块 2：核心分析（EMA 增强）====================


class LogicAnalyzer:
    """核心分析器，封装 EMA 平滑 → PHATE → TDA → 聚类 → PELT 全流程。

    Attributes:
        embeddings_gpu: GPU 上的 embedding 张量（EMA 平滑后更新）。
        embeddings_cpu: CPU numpy 副本。
        results: 各步骤输出结果的字典。
    """

    def __init__(self, embeddings_tensor: torch.Tensor) -> None:
        """初始化分析器。

        Args:
            embeddings_tensor: 形状 ``(n, d)`` 的 float32 张量，
                每行是一个单句的原始 SBERT embedding。
        """
        self.embeddings_gpu: torch.Tensor = embeddings_tensor
        clean_cpu = embeddings_tensor.cpu().numpy()
        self.embeddings_cpu: np.ndarray = clean_cpu + np.random.normal(
            0, 1e-5, clean_cpu.shape
        )
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # EMA 时序平滑（核心改动）
    # ------------------------------------------------------------------

    def run_ema_smoothing(self) -> None:
        """指数移动平均时序平滑，替代滑动窗口拼接。

        支持**双向 EMA**（默认启用）：
            ``ema[i] = (forward_ema[i] + backward_ema[i]) / 2``

        其中：
        - **前向 EMA**: ``f[0] = emb[0]``, ``f[i] = (1-α)·emb[i] + α·f[i-1]``
        - **后向 EMA**: ``b[n-1] = emb[n-1]``, ``b[i] = (1-α)·emb[i] + α·b[i+1]``

        **为什么双向 EMA 更好？**
        - 单向 EMA 仅融合过去信息，存在**相位偏移**（smoothed 曲线滞后于原始信号）
        - 双向 EMA 同时融合**过去和未来**上下文，每个位置的信息是对称的
        - 在主题切换边界处，双向 EMA 的**过渡更锐利**，因为前后向在边界处形成"钳制"
        - 保留 O(n) 效率，仅需两次线性扫描

        EMA 的作用：
        - **低通滤波**：高频的语义震荡被衰减，保留全局主题趋势
        - **边界增强**：在主题切换点，EMA 的滞后效应天然放大差异
        - **O(n) 效率**：单次线性扫描，无需图构建或矩阵运算
        """
        emb = self.embeddings_gpu
        n = emb.shape[0]

        if n < 2:
            logger.warning("EMA 平滑：节点数 < 2，跳过")
            return

        cfg = CFG
        alpha = cfg.ema_alpha
        bidirectional = cfg.ema_bidirectional

        # ---- 前向 EMA ----
        fwd = torch.zeros_like(emb)
        fwd[0] = emb[0]
        for i in range(1, n):
            fwd[i] = (1 - alpha) * emb[i] + alpha * fwd[i - 1]

        if bidirectional and n > 2:
            # ---- 后向 EMA（从末尾向前扫描）----
            bwd = torch.zeros_like(emb)
            bwd[n - 1] = emb[n - 1]
            for i in range(n - 2, -1, -1):
                bwd[i] = (1 - alpha) * emb[i] + alpha * bwd[i + 1]

            # ---- 双向平均 ----
            ema = (fwd + bwd) / 2.0

            logger.info(
                "双向 EMA 平滑完成: alpha=%.2f, 节点=%d",
                alpha,
                n,
            )
        else:
            ema = fwd
            logger.info(
                "单向 EMA 平滑完成: alpha=%.2f, 节点=%d%s",
                alpha,
                n,
                " (n≤2，退化为单向)" if n <= 2 else "",
            )

        # 更新内部状态
        self.embeddings_gpu = ema
        self.embeddings_cpu = ema.detach().cpu().numpy()


    # ------------------------------------------------------------------
    # PHATE 降维
    # ------------------------------------------------------------------

    def run_phate(self) -> None:
        """运行 PHATE 降维，将高维 embedding 映射到 3D 流形空间。"""
        if not _PHATE_AVAILABLE:
            logger.warning("PHATE 不可用，跳过")
            return
        import phate as _phate

        cfg = CFG
        op = _phate.PHATE(
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
        self.results["phate_params"] = {
            "n_components": cfg.phate_n_components,
            "knn": cfg.phate_knn,
            "knn_dist": cfg.phate_knn_dist,
            "n_pca": cfg.phate_n_pca,
            "mds": cfg.phate_mds,
            "mds_solver": cfg.phate_mds_solver,
        }

    # ------------------------------------------------------------------
    # TDA 拓扑数据分析
    # ------------------------------------------------------------------

    def run_tda(self) -> None:
        """计算 Vietoris–Rips 持续同调（H0 / H1 / H2）。"""
        if not _TDA_AVAILABLE:
            return
        from tqdm import tqdm

        from gtda.homology import VietorisRipsPersistence

        n = len(self.embeddings_gpu)
        norm = F.normalize(self.embeddings_gpu, p=2, dim=1)
        logger.info("计算 %d×%d 距离矩阵 ...", n, n)

        with tqdm(total=3, desc="TDA 步骤") as pbar:
            dist_matrix = (
                1.0
                - torch.clamp(torch.mm(norm, norm.t()), -1.0, 1.0)
            ).cpu().numpy()
            pbar.update(1)
            pbar.set_description("后处理距离矩阵")
            np.fill_diagonal(dist_matrix, 0.0)
            dist_matrix = np.maximum(dist_matrix, 0.0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
            pbar.update(1)
            pbar.set_description("VietorisRips 持续同调")
            vr = VietorisRipsPersistence(
                metric="precomputed",
                homology_dimensions=[0, 1, 2],
                collapse_edges=True,
                n_jobs=-1,
            )
            dgms = vr.fit_transform(dist_matrix[np.newaxis, :, :])
            pbar.update(1)

        self.results["tda"] = {
            "diagrams": dgms[0],
            "betti": {0: 0, 1: 0, 2: 0},
        }

    # ------------------------------------------------------------------
    # 离群检测 + 图聚类
    # ------------------------------------------------------------------

    def _knn_graph(
        self, emb: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 k-NN 距离和索引。

        Args:
            emb: 形状 ``(n, d)`` 的归一化 embedding。
            k: 邻居数量。

        Returns:
            (distances[:, 1:], indices[:, 1:])，自环被排除。
        """
        nn = NearestNeighbors(
            n_neighbors=min(k + 1, len(emb)), metric="cosine"
        )
        nn.fit(emb)
        dists, inds = nn.kneighbors(emb)
        return dists[:, 1:], inds[:, 1:]

    def run_graph_clustering(self) -> None:
        """执行离群分析 + Leiden / HDBSCAN 聚类。"""
        emb = self.embeddings_cpu
        n = len(emb)
        cfg = CFG

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

        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        k = int(
            np.clip(np.sqrt(max(n, 4)), cfg.outlier_k_min, cfg.outlier_k_max)
        )
        dists, inds = self._knn_graph(emb_norm, k)
        sims = np.clip(1.0 - dists, 0.0, 1.0)
        outlier_scores: np.ndarray = 1.0 - sims.mean(axis=1)

        labels_raw: Optional[np.ndarray] = None
        backend: str = "unknown"

        if _LEIDEN_AVAILABLE:
            import igraph as _ig
            import leidenalg as _la

            edges: List[Tuple[int, int]] = []
            weights: List[float] = []
            seen: set = set()
            for i in range(n):
                for j, sim_val in zip(inds[i], sims[i]):
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    if a == b or (a, b) in seen:
                        continue
                    seen.add((a, b))
                    edges.append((a, b))
                    weights.append(float(sim_val))

            g = _ig.Graph(n=n, edges=edges, directed=False)
            part = _la.find_partition(
                g,
                _la.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=cfg.leiden_resolution,
            )
            labels_raw = np.array(part.membership, dtype=int)
            backend = "leiden"
        elif _HDBSCAN_AVAILABLE:
            import hdbscan as _hdbscan

            clusterer = _hdbscan.HDBSCAN(
                min_cluster_size=cfg.hdbscan_min_cluster_size,
                min_samples=cfg.hdbscan_min_samples,
            )
            labels_raw = clusterer.fit_predict(emb_norm)
            outlier_scores = clusterer.outlier_scores_
            backend = "hdbscan"
        else:
            labels_raw = np.zeros(n, dtype=int)
            backend = "fallback-single"

        self.results["clusters_raw"] = labels_raw
        self.results["clusters"] = labels_raw
        self.results["outliers"] = outlier_scores
        self.results["cluster_backend"] = backend
        self.results["outlier_k"] = k

    # ------------------------------------------------------------------
    # PELT 语义分段
    # ------------------------------------------------------------------

    def run_pelt_segmentation(self) -> None:
        """基于 PELT 变点检测算法对 embedding 序列分段。

        先尝试 ``rbf`` 核，若失败则回退 ``l2`` 核。
        最后合并连续相同主题的分段。
        """
        emb = self.embeddings_cpu
        n = len(emb)

        distances = np.linalg.norm(emb[1:] - emb[:-1], axis=1)
        median_dist = float(np.median(distances))
        penalty = median_dist * 5.0

        logger.info(
            "PHATE 模式校准：中值位移=%.6f, 动态 Penalty=%.6f",
            median_dist,
            penalty,
        )

        try:
            algo = rpt.Pelt(model="rbf", min_size=5).fit(emb)
            raw_cuts = algo.predict(pen=penalty)
            boundaries: List[int] = [0] + sorted(set(raw_cuts))
        except Exception:
            logger.warning("rbf 核失败，回退 l2 核")
            algo = rpt.Pelt(model="l2", min_size=5).fit(emb)
            raw_cuts = algo.predict(pen=penalty * 0.1)
            boundaries = [0] + sorted(set(raw_cuts))

        clusters = np.array(
            self.results.get("clusters", np.zeros(n, dtype=int))
        )

        topics: List[int] = []
        for i in range(len(boundaries) - 1):
            l, r = boundaries[i], boundaries[i + 1]
            vals = clusters[l:r]
            if len(vals) > 0:
                u, c = np.unique(vals, return_counts=True)
                topics.append(int(u[np.argmax(c)]))
            else:
                topics.append(0)

        # 合并连续相同 topic
        final_boundaries: List[int] = [boundaries[0]]
        final_topics: List[int] = []
        if topics:
            current_topic = topics[0]
            for i in range(1, len(topics)):
                if topics[i] != current_topic:
                    final_boundaries.append(boundaries[i])
                    final_topics.append(current_topic)
                    current_topic = topics[i]
            final_boundaries.append(boundaries[-1])
            final_topics.append(current_topic)
        else:
            final_boundaries = boundaries
            final_topics = topics

        self.results["segments"] = {
            "boundaries": final_boundaries,
            "topics": final_topics,
            "segment_ids": self._update_segment_ids(n, final_boundaries),
            "method": "pelt_merged_adaptive",
        }

    @staticmethod
    def _update_segment_ids(
        n: int, boundaries: List[int]
    ) -> np.ndarray:
        seg_ids = np.zeros(n, dtype=int)
        for i in range(len(boundaries) - 1):
            l, r = int(boundaries[i]), int(boundaries[i + 1])
            r = min(r, n)
            seg_ids[l:r] = i
        return seg_ids


# ==================== 模块 3：数据加载 ====================


def load_or_build_chunks(
    segments: List[Dict[str, str]], output_dir: str
) -> List[str]:
    """加载或构建单句 chunk 列表。

    EMA 模式下每个 segment 的文本即为一个独立 chunk，
    不再做滑动窗口拼接。时序上下文由 EMA 平滑在向量空间完成。

    Args:
        segments: ``split_text_into_segments`` 的输出。
        output_dir: 缓存目录。

    Returns:
        单句文本列表。
    """
    chunks_cache = os.path.join(output_dir, "chunks.json")
    if os.path.exists(chunks_cache):
        with open(chunks_cache, "r", encoding="utf-8") as f:
            chunks: List[str] = json.load(f)
        logger.info("已读取 chunks 缓存: %s", chunks_cache)
        return chunks

    chunks = [s["text"].strip() for s in segments]
    with open(chunks_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info("已保存 chunks 缓存: %s", chunks_cache)
    return chunks


def load_or_embed(
    chunks: List[str], output_dir: str
) -> torch.Tensor:
    """加载或计算 SBERT embedding。

    优先从 ``embeddings.npy`` 缓存读取；否则使用 MLX / SentenceTransformer 编码。

    Args:
        chunks: 文本列表。
        output_dir: 缓存目录。

    Returns:
        形状 ``(n, d)`` 的 float32 张量。
    """
    emb_cache = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_cache):
        embeddings = np.load(emb_cache)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        logger.info("已读取 embeddings 缓存: %s", emb_cache)
        return embeddings_tensor

    try:
        from mlx_embedding_models.embedding import EmbeddingModel

        try:
            from transformers import PreTrainedTokenizerBase

            if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):

                def _batch_encode_plus(
                    self: Any,
                    batch_text_or_text_pairs: Any,
                    **kwargs: Any,
                ) -> Any:
                    return self.__call__(batch_text_or_text_pairs, **kwargs)

                PreTrainedTokenizerBase.batch_encode_plus = (  # type: ignore[method-assign]
                    _batch_encode_plus
                )
        except Exception:
            pass

        if hasattr(EmbeddingModel, "from_pretrained"):
            mlx_model = EmbeddingModel.from_pretrained(CFG.sbert_model_name)
        else:
            mlx_model = EmbeddingModel.from_registry("bge-m3")

        embeddings = np.array(mlx_model.encode(chunks))
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        logger.info("使用 MLX Embedding Models: %s", CFG.sbert_model_name)
    except Exception as exc:
        logger.warning(
            "MLX 不可用，回退 SentenceTransformer。原因: %s", exc
        )
        model = SentenceTransformer(CFG.sbert_model_name, device=CFG.device)
        embeddings_tensor = model.encode(chunks, convert_to_tensor=True)

    np.save(emb_cache, embeddings_tensor.detach().cpu().numpy())
    logger.info("已保存 embeddings 缓存: %s", emb_cache)
    return embeddings_tensor


def compute_coherence(
    embeddings_tensor: torch.Tensor,
) -> Tuple[float, np.ndarray]:
    """计算相邻句子间的余弦相似度（连贯性指标）。

    Args:
        embeddings_tensor: 形状 ``(n, d)`` 的 embedding 张量。

    Returns:
        (平均连贯性, 每个相邻对的相似度数组)。
    """
    norm = F.normalize(embeddings_tensor, p=2, dim=1)
    sims = (norm[:-1] * norm[1:]).sum(dim=1).cpu().numpy()
    return (float(np.mean(sims)), sims)


def smooth_3d_curve(
    points: np.ndarray, iterations: int = 2
) -> np.ndarray:
    """对 3D 轨迹曲线做 Chaikin 式平滑。

    Args:
        points: 形状 ``(n, 3)`` 的 3D 点序列。
        iterations: 平滑迭代次数。

    Returns:
        平滑后的点序列（长度可能增加）。
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or iterations <= 0:
        return pts
    sm = pts.copy()
    for _ in range(iterations):
        out: List[np.ndarray] = [sm[0]]
        for i in range(len(sm) - 1):
            p, q = sm[i], sm[i + 1]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        out.append(sm[-1])
        sm = np.array(out, dtype=np.float64)
    return sm


# ==================== 模块 4：报告生成 ====================


def _build_subplot_specs_and_titles(
    res: Dict[str, Any],
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """根据分析结果动态构建 subplot 布局和标题。"""
    specs: List[List[Dict[str, Any]]] = [
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
        titles.append("持续性图 (TDA)")
        titles.append("拓扑逻辑摘要")

    if "phate" in res:
        specs.append([{"type": "domain", "colspan": 2}, None])
        titles.append("PHATE 轨迹特征量化分析")

    specs.append([{"type": "table", "colspan": 2}, None])
    titles.append("Chunk 文本")
    specs.append([{"type": "table", "colspan": 2}, None])
    titles.append("分段主题簇")
    specs.append([{"type": "table", "colspan": 2}, None])
    titles.append("参数清单")

    return specs, titles


def _add_phate_traces(
    fig: go.Figure,
    p_coords: np.ndarray,
    seg_boundaries: Sequence[int],
    seg_topics: Sequence[int],
    phate_point_trace_indices: List[int],
) -> None:
    """添加 PHATE 3D 轨迹相关 trace。"""
    # 按时间戳顺序连接所有点的直线轨迹
    fig.add_trace(
        go.Scatter3d(
            x=p_coords[:, 0],
            y=p_coords[:, 1],
            z=p_coords[:, 2],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.5)", width=1.5),
            name="时间序轨迹",
        ),
        row=1,
        col=1,
    )

    palette = [
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

    if len(seg_boundaries) >= 2:
        b_idx = [
            max(0, min(int(b), len(p_coords) - 1))
            for b in seg_boundaries
        ]

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

    n_seg = max(0, len(seg_boundaries) - 1)

    for s in range(n_seg):
        l, r = int(seg_boundaries[s]), int(seg_boundaries[s + 1])
        idx = np.arange(l, r, dtype=int)
        if len(idx) <= 0:
            continue
        color = palette[s % len(palette)]
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


def save_modular_report(
    chunks: List[str],
    analyzer: LogicAnalyzer,
    coh_data: Tuple[float, np.ndarray],
    output_dir: str,
    embeddings_tensor: torch.Tensor,
    doc_title: str = "text_document",
    report_name: str = "phate_logic_report.html",
) -> None:
    """生成交互式 Plotly 报告并保存为 HTML。

    Args:
        chunks: 文本列表。
        analyzer: 已完成全部分析的 LogicAnalyzer 实例。
        coh_data: ``compute_coherence`` 返回的 (均值, 数组)。
        output_dir: 报告保存目录。
        embeddings_tensor: 原始 embedding 张量（用于 R 序列计算）。
        doc_title: 文档标题。
        report_name: 报告 HTML 文件名。
    """
    res = analyzer.results
    cfg = CFG

    emb = embeddings_tensor.detach().cpu().numpy()
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    phi = 0.7
    c_t = [emb_norm[0]]
    for i in range(1, len(emb_norm)):
        c_t.append(phi * c_t[-1] + (1.0 - phi) * emb_norm[i])
    c_t_arr = np.array(c_t)
    c_t_arr = c_t_arr / (
        np.linalg.norm(c_t_arr, axis=1, keepdims=True) + 1e-9
    )
    R = np.sum(emb_norm * c_t_arr, axis=1)

    seg_info = res.get("segments", {})
    seg_boundaries: List[int] = seg_info.get(
        "boundaries", [0, len(chunks)]
    )
    seg_topics: List[int] = seg_info.get("topics", [])

    specs, titles = _build_subplot_specs_and_titles(res)
    fig = make_subplots(
        rows=len(specs),
        cols=2,
        column_widths=[0.6, 0.4],
        specs=specs,
        subplot_titles=titles,
    )
    phate_point_trace_indices: List[int] = []

    if "phate" in res:
        p_coords = res["phate"]
        _add_phate_traces(
            fig,
            p_coords,
            seg_boundaries,
            seg_topics,
            phate_point_trace_indices,
        )

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

    if "tda" in res:
        dg = res["tda"]["diagrams"]
        for d in (0, 1, 2):
            pts = dg[dg[:, 2] == d]
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    name=f"H{d}",
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
                        [
                            "主题独立性",
                            "逻辑闭环",
                            "高阶空洞/多主题交错",
                        ],
                    ]
                ),
            ),
            row=curr_row,
            col=2,
        )
        curr_row += 1

    if "phate" in res:
        cos_sim = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        d_t = np.arccos(cos_sim)
        L_vel = float(d_t.mean()) if len(d_t) > 0 else 0.0
        mu_v = emb.mean(axis=0)
        L_vol = float(
            np.sqrt(((emb - mu_v) ** 2).sum(axis=1).mean())
        )
        S_coh = float(np.mean(cos_sim)) if len(cos_sim) > 0 else 0.0
        z_vel = (
            (L_vel - np.mean(d_t)) / (np.std(d_t) + 1e-9)
            if len(d_t) > 0
            else 0.0
        )
        vol_series = np.sqrt(((emb - mu_v) ** 2).sum(axis=1))
        z_vol = (
            (L_vol - np.mean(vol_series)) / (np.std(vol_series) + 1e-9)
            if len(vol_series) > 0
            else 0.0
        )
        score_load = float(S_coh * (0.5 * z_vel + 0.5 * z_vol))

        D_shock = (
            float(np.mean(np.abs(R[1:] - R[:-1])))
            if len(R) > 1
            else 0.0
        )
        mu_R = float(np.mean(R)) if len(R) > 0 else 0.0
        raw_stab = mu_R - 0.5 * D_shock
        z_stab = (
            (raw_stab - np.mean(R)) / (np.std(R) + 1e-9)
            if len(R) > 0
            else 0.0
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

    chunk_idx = list(range(1, len(chunks) + 1))
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Chunk", "文本"],
                fill_color="darkslategray",
                font=dict(color="white"),
            ),
            cells=dict(
                values=[chunk_idx, chunks],
                align=["center", "left"],
            ),
        ),
        row=curr_row,
        col=1,
    )
    curr_row += 1

    seg_rows = list(range(1, max(0, len(seg_boundaries) - 1) + 1))
    seg_ranges: List[str] = []
    seg_topic_vals: List[int] = []
    seg_sizes: List[int] = []
    for i in range(max(0, len(seg_boundaries) - 1)):
        l, r = int(seg_boundaries[i]), int(seg_boundaries[i + 1])
        seg_ranges.append(f"[{l}, {r})")
        seg_topic_vals.append(
            int(seg_topics[i]) if i < len(seg_topics) else -1
        )
        seg_sizes.append(max(0, r - l))

    fig.add_trace(
        go.Table(
            header=dict(
                values=["段", "区间", "topic_cluster", "长度"],
                fill_color="darkslateblue",
                font=dict(color="white"),
            ),
            cells=dict(
                values=[
                    seg_rows,
                    seg_ranges,
                    seg_topic_vals,
                    seg_sizes,
                ],
                align=["center", "center", "center", "center"],
            ),
        ),
        row=curr_row,
        col=1,
    )
    curr_row += 1

    phate_params = res.get("phate_params", {})
    param_names: List[str] = [
        "Input type",
        "SBERT_MODEL_NAME",
        "WINDOW_SIZE",
        "STEP_SIZE",
        "EMA_ALPHA",
        "EMA_BIDIRECTIONAL",
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
        str(cfg.ema_alpha),
        str(cfg.ema_bidirectional),
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

    n_traces = len(fig.data)
    vis_points_on = [True] * n_traces
    vis_points_off = [True] * n_traces
    for idx in phate_point_trace_indices:
        if 0 <= idx < n_traces:
            vis_points_off[idx] = False

    fig.update_layout(
        height=450 * len(specs),
        title=f"📄 [EMA] PHATE 语义拓扑深度逻辑报告 - {doc_title}",
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
            )
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
    logger.info("报告已生成: %s", report_path)


# ==================== 文件读取工具 ====================


def read_text_file(path: str) -> str:
    """根据文件扩展名读取文本内容，支持 .txt / .docx / .doc。

    Args:
        path: 文件绝对路径。

    Returns:
        提取出的文本字符串。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".docx":
        try:
            from docx import Document as DocxDocument

            doc = DocxDocument(path)
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs)
        except Exception as exc:
            logger.error("读取 .docx 失败: %s", exc)
            raise

    if ext == ".doc":
        # macOS: 用系统 textutil 将 .doc 转为临时 .txt
        try:
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w"
            ) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                ["textutil", "-convert", "txt", "-output", tmp_path, path],
                check=True,
                capture_output=True,
            )
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
            os.unlink(tmp_path)
            return text
        except Exception as exc:
            logger.error("读取 .doc 失败（需 macOS textutil）: %s", exc)
            raise

    raise ValueError(f"不支持的文件格式: {ext}（仅支持 .txt / .docx / .doc）")


# ==================== 入口 ====================


def process_text(text: str, doc_title: str = "my_document") -> None:

    """全自动语义分析管线（EMA 增强版）。

    流程:
    1. 文字分段 → 2. 单句 chunks → 3. SBERT 编码
    → 4. **EMA 时序平滑（替代窗口化和 GNN）**
    → 5. PHATE 降维 → 6. TDA / 聚类 / PELT → 7. 报告生成

    Args:
        text: 要分析的文字内容。
        doc_title: 文档标题，用于输出目录和报告标题。
    """
    script_dir = get_script_dir()
    safe_title = (
        doc_title.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")[:120]
        or "text_document"
    )
    output_dir = os.path.join(script_dir, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    with SimpleTimer("文字分段"):
        segments = split_text_into_segments(text)
        logger.info("共分割为 %d 个语义段", len(segments))

    chunks = load_or_build_chunks(segments, output_dir)

    with SimpleTimer("SBERT 编码"):
        embeddings_tensor = load_or_embed(chunks, output_dir)

    analyzer = LogicAnalyzer(embeddings_tensor)

    with SimpleTimer("EMA 时序平滑 (替代窗口化)"):
        analyzer.run_ema_smoothing()

    with SimpleTimer("PHATE 轨迹生成 (EMA 增强后降维)"):
        analyzer.run_phate()

    if _TDA_AVAILABLE and CFG.enable_tda:
        with SimpleTimer("TDA 拓扑分析"):
            analyzer.run_tda()

    with SimpleTimer("图聚类 (Leiden/HDBSCAN)"):
        analyzer.run_graph_clustering()

    with SimpleTimer("PELT 语义分段"):
        analyzer.run_pelt_segmentation()

    with SimpleTimer("连贯性计算"):
        coh_data = compute_coherence(embeddings_tensor)

    save_modular_report(
        chunks,
        analyzer,
        coh_data,
        output_dir,
        embeddings_tensor,
        doc_title,
    )


def main() -> None:
    """从 ``CFG.input_file_path`` 读取文件（支持 .txt / .docx / .doc）并运行分析。"""
    input_path = os.path.join(get_script_dir(), CFG.input_file_path)
    if not os.path.exists(input_path):
        logger.error("输入文件不存在: %s", input_path)
        logger.info(
            "请先在 Config 中设置 input_file_path，"
            "或在同级目录下创建输入文件。"
        )
        return

    text = read_text_file(input_path)
    logger.info("已读取输入文件: %s", input_path)
    process_text(text, doc_title=CFG.doc_title)


if __name__ == "__main__":

    main()
