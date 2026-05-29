import os
import time
import json
import warnings
import importlib

try:
    import sklearn.utils.validation as _skval
    if getattr(_skval.check_array, "_is_patched", False):
        importlib.reload(_skval)
    if not hasattr(_skval, "_original_check_array"):
        _skval._original_check_array = _skval.check_array
    def _patched_check_array(array, *args, **kwargs):
        if 'force_all_finite' in kwargs:
            del kwargs['force_all_finite']
        return _skval._original_check_array(array, *args, **kwargs)
    _patched_check_array._is_patched = True
    if _skval.check_array is not _patched_check_array:
        _skval.check_array = _patched_check_array
        print("✅ 已应用 scikit-learn 兼容性补丁")
except ImportError:
    pass
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yt_dlp
import mlx_whisper
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import ruptures as rpt

# ==================== 全局配置 ====================
YOUTUBE_URL = "https://www.youtube.com/watch?v=zIwLWfaAg-8"
WHISPER_MODEL = "mlx-community/whisper-small-mlx-4bit"
SBERT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
WINDOW_SIZE, STEP_SIZE = 5, 1
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ENABLE_TDA = False
PHATE_AUTO_PCA_CAP = 100

# PHATE 参数
PHATE_N_COMPONENTS = 3
PHATE_KNN = 7
PHATE_KNN_DIST = "cosine"
PHATE_N_PCA = None
PHATE_MDS = "metric"
PHATE_MDS_SOLVER = "smacof"

# 图聚类/离群参数
OUTLIER_K_MIN = 6
OUTLIER_K_MAX = 18
LEIDEN_RESOLUTION = 1.0
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 1



REMOTE_COMPONENTS = ["ejs:github"]
JS_RUNTIMES = {"deno": {}}

# ==================== 依赖检测 ====================
try:
    import gtda
    from gtda.homology import VietorisRipsPersistence
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    print("⚠️ 未检测到 gtda，TDA 模块将跳过")

try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False
    print("⚠️ 未检测到 phate，PHATE 模块将跳过")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️ 未检测到 hdbscan，离群分析将跳过")

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("⚠️ 未检测到 igraph/leidenalg，将回退 HDBSCAN 聚类")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== 工具类 ====================
class SimpleTimer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        print(f"▶️ [开始] {self.name}...")
        return self
    def __exit__(self, *args):
        print(f"✅ [完成] {self.name} | 耗时: {time.time() - self.start:.2f}s")

# ==================== 模块 1：抓取 ====================

def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def sanitize_filename(name, max_len=120):
    if not name:
        return "video"
    safe = ''.join(c if (c.isalnum() or c in ' ._-') else '_' for c in name)
    safe = ' '.join(safe.split()).strip(' ._-')
    return (safe[:max_len] or "video")


def get_video_title(url):
    try:
        opts = {
            "quiet": True,
            "skip_download": True,
            "remote_components": REMOTE_COMPONENTS,
            "js_runtimes": JS_RUNTIMES,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info.get("title") or info.get("id") or "video"
    except Exception:
        return "video"


def download_audio(url, output_path):
    if os.path.exists(output_path):
        return
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "outtmpl": output_path.replace(".wav", ""),
        "quiet": False,
        "remote_components": REMOTE_COMPONENTS,
        "js_runtimes": JS_RUNTIMES,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# ==================== 模块 2：处理 ====================

class LogicAnalyzer:
    def __init__(self, embeddings_tensor):
        self.embeddings_gpu = embeddings_tensor
        clean_cpu = embeddings_tensor.cpu().numpy()
        self.embeddings_cpu = clean_cpu + np.random.normal(0, 1e-5, clean_cpu.shape)
        self.results = {}

    def run_phate(self):
        if not PHATE_AVAILABLE:
            return
        # Use full embedding space (no PCA pre-reduction).
        phate_input = self.embeddings_cpu
        op = phate.PHATE(
            n_components=PHATE_N_COMPONENTS,
            knn=PHATE_KNN,
            knn_dist=PHATE_KNN_DIST,
            n_pca=PHATE_N_PCA,
            mds=PHATE_MDS,
            mds_solver=PHATE_MDS_SOLVER,
            n_jobs=-1,
            verbose=False,
        )
        self.results["phate"] = op.fit_transform(phate_input)
        self.results["phate_n_pca"] = PHATE_N_PCA
        self.results["phate_params"] = {
            "n_components": PHATE_N_COMPONENTS,
            "knn": PHATE_KNN,
            "knn_dist": PHATE_KNN_DIST,
            "n_pca": PHATE_N_PCA,
            "mds": PHATE_MDS,
            "mds_solver": PHATE_MDS_SOLVER,
        }

    def run_tda(self):
        if not TDA_AVAILABLE:
            return
        norm = F.normalize(self.embeddings_gpu, p=2, dim=1)
        dist_matrix = (1.0 - torch.clamp(torch.mm(norm, norm.t()), -1.0, 1.0)).cpu().numpy()
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.maximum(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2], collapse_edges=True, n_jobs=-1)
        dgms = vr.fit_transform(dist_matrix[np.newaxis, :, :])
        self.results["tda"] = {"diagrams": dgms[0], "betti": {0: 0, 1: 0}}


    def _knn_graph(self, emb, k):
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb)), metric="cosine")
        nn.fit(emb)
        dists, inds = nn.kneighbors(emb)
        return dists[:, 1:], inds[:, 1:]

    def run_graph_clustering(self):
        emb = self.embeddings_cpu
        n = len(emb)
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

        # Outlier/dispersion is computed in original embedding space (after L2 normalization), not PHATE space.
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        k = int(np.clip(np.sqrt(max(n, 4)), OUTLIER_K_MIN, OUTLIER_K_MAX))
        dists, inds = self._knn_graph(emb_norm, k)
        sims = np.clip(1.0 - dists, 0.0, 1.0)
        outliers = 1.0 - sims.mean(axis=1)

        labels_raw = None
        if LEIDEN_AVAILABLE:
            edges = []
            weights = []
            seen = set()
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
                resolution_parameter=LEIDEN_RESOLUTION,
            )
            labels_raw = np.array(part.membership, dtype=int)
            backend = "leiden"
        elif HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES)
            labels_raw = clusterer.fit_predict(emb_norm)
            outliers = clusterer.outlier_scores_
            backend = "hdbscan"
        else:
            labels_raw = np.zeros(n, dtype=int)
            backend = "fallback-single"

        self.results["clusters_raw"] = labels_raw
        self.results["clusters"] = labels_raw
        self.results["outliers"] = outliers
        self.results["cluster_backend"] = backend
        self.results["outlier_k"] = k

    # 原先名为 run_topictiling_segmentation，已更名为更准确的 run_pelt_segmentation
    # 这个方法使用 PELT 算法进行语义分段，并在最后将结果写回 self.results
    def run_pelt_segmentation(self):
        emb = self.embeddings_cpu
        n = len(emb)

        # 1. [核心逻辑] 针对滑动窗口的“去平滑”补偿
        # 计算相邻 Chunk 的欧式距离序列
        # 因为滑动窗口下，d(i, i+1) 非常小，只有话题真正切换时才会出现一个小峰值
        distances = np.linalg.norm(emb[1:] - emb[:-1], axis=1)
        
        # 自动校准：将惩罚项设为“背景摩擦力”的极低倍数
        # 滑动窗口下，我们要寻找的是“慢速移动”中的“突然加速”
        median_dist = np.median(distances)
        
        # 这是一个经验公式：针对滑动窗口，Penalty 不能根据维度算，要根据平均位移算
        # 调优建议：如果还是 1 段，把 5.0 调小到 2.0；如果段太多，调大到 15.0
        penalty = median_dist * 5.0 
        
        print(f"📊 PHATE 模式校准：中值位移={median_dist:.6f}, 动态 Penalty={penalty:.6f}")

        # 2. [算法选择]
        # 如果用 'l2' 模型容易被平滑数据欺骗，建议改用 'rbf' (径向基函数)
        # RBF 对高维空间的微小局部结构变化更敏感
        try:
            algo = rpt.Pelt(model="rbf", min_size=5).fit(emb)
            raw_cuts = algo.predict(pen=penalty)
            boundaries = [0] + sorted(list(set(raw_cuts)))
        except:
            # 后备方案：如果 rbf 报错，回退到更激进的 l2
            algo = rpt.Pelt(model="l2", min_size=5).fit(emb)
            raw_cuts = algo.predict(pen=penalty * 0.1) # 进一步强行压低惩罚
            boundaries = [0] + sorted(list(set(raw_cuts)))

        # 3. 结果写回（保持与你 700 行代码的其他部分兼容）
        clusters = np.array(self.results.get("clusters", np.zeros(n, dtype=int)))
        topics = []
        for i in range(len(boundaries) - 1):
            l, r = boundaries[i], boundaries[i + 1]
            vals = clusters[l:r]
            u, c = np.unique(vals, return_counts=True) if len(vals) > 0 else ([0], [0])
            topics.append(int(u[np.argmax(c)]))
        final_boundaries = [boundaries[0]]
        final_topics = []

        # --- 同主题合并补丁 ---
        raw_boundaries = boundaries
        raw_topics = topics

        if len(raw_topics) > 0:
            current_topic = raw_topics[0]
            for i in range(1, len(raw_topics)):
                # 核心判断：用 str() 防止类型差异导致判断失败
                if str(raw_topics[i]) != str(current_topic):
                    final_boundaries.append(raw_boundaries[i])
                    final_topics.append(current_topic)
                    current_topic = raw_topics[i]
            # 放入最后一个段落的结束边界和主题
            final_boundaries.append(raw_boundaries[-1])
            final_topics.append(current_topic)
        else:
            final_boundaries = raw_boundaries
            final_topics = raw_topics

        # --- 关键：必须确保把结果写回 self.results ---
        self.results["segments"] = {
            "boundaries": final_boundaries,
            "topics": final_topics,
            "segment_ids": self._update_segment_ids(n, final_boundaries),
            "method": "pelt_merged_adaptive"
        }


    def _update_segment_ids(self, n, boundaries):
        """辅助函数：根据新的边界重新生成段落 ID 数组"""
        seg_ids = np.zeros(n, dtype=int)
        for i in range(len(boundaries) - 1):
            l, r = int(boundaries[i]), int(boundaries[i + 1])
            # 注意边界处理：确保索引不越界
            r = min(r, n)
            seg_ids[l:r] = i
        return seg_ids
    
def load_or_transcribe(audio_file, output_dir):
    whisper_cache = os.path.join(output_dir, "whisper_segments.json")
    if os.path.exists(whisper_cache):
        with open(whisper_cache, "r", encoding="utf-8") as f:
            segments = json.load(f)
        print(f"✅ 已读取 Whisper 缓存: {whisper_cache}")
        return segments
    result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=WHISPER_MODEL)
    segments = result["segments"]
    with open(whisper_cache, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)
    print(f"✅ 已保存 Whisper 缓存: {whisper_cache}")
    return segments


def load_or_build_chunks(segments, output_dir):
    chunks_cache = os.path.join(output_dir, "chunks.json")
    if os.path.exists(chunks_cache):
        with open(chunks_cache, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✅ 已读取 chunks 缓存: {chunks_cache}")
        return chunks
    chunks = [" ".join([s["text"].strip() for s in segments[i:i + WINDOW_SIZE]])
              for i in range(0, len(segments) - WINDOW_SIZE + 1, STEP_SIZE)]
    with open(chunks_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"✅ 已保存 chunks 缓存: {chunks_cache}")
    return chunks


def load_or_embed(chunks, output_dir):
    emb_cache = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_cache):
        embeddings = np.load(emb_cache)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        print(f"✅ 已读取 embeddings 缓存: {emb_cache}")
        return embeddings_tensor

    try:
        from mlx_embedding_models.embedding import EmbeddingModel
        try:
            from transformers import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
                def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
                    return self.__call__(batch_text_or_text_pairs, **kwargs)
                PreTrainedTokenizerBase.batch_encode_plus = _batch_encode_plus
        except Exception:
            pass

        if hasattr(EmbeddingModel, "from_pretrained"):
            mlx_model = EmbeddingModel.from_pretrained(SBERT_MODEL_NAME)
        else:
            mlx_model = EmbeddingModel.from_registry("bge-base-en-v1.5")
        embeddings = np.array(mlx_model.encode(chunks))
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        print(f"✅ 使用 MLX Embedding Models: {SBERT_MODEL_NAME}")
    except Exception as e:
        print(f"⚠️ MLX Embedding Models 不可用或加载失败，回退 SentenceTransformer。原因: {e}")
        model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
        embeddings_tensor = model.encode(chunks, convert_to_tensor=True)

    np.save(emb_cache, embeddings_tensor.detach().cpu().numpy())
    print(f"✅ 已保存 embeddings 缓存: {emb_cache}")
    return embeddings_tensor


def compute_coherence(embeddings_tensor):
    norm = F.normalize(embeddings_tensor, p=2, dim=1)
    sims = (norm[:-1] * norm[1:]).sum(dim=1).cpu().numpy()
    return (np.mean(sims), sims)


def smooth_3d_curve(points, iterations=2):
    """Chaikin corner-cutting for a visually smoother 3D trajectory."""
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


# ==================== 模块 3：报告 ====================

def save_modular_report(
    chunks,
    analyzer,
    coh_data,
    output_dir,
    embeddings_tensor,
    video_title,
    report_name="phate_logic_report.html",
):
    res = analyzer.results

    emb = embeddings_tensor.detach().cpu().numpy()
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    phi = 0.7
    c_t = [emb_norm[0]]
    for i in range(1, len(emb_norm)):
        c_t.append(phi * c_t[-1] + (1.0 - phi) * emb_norm[i])
    c_t = np.array(c_t)
    c_t = c_t / (np.linalg.norm(c_t, axis=1, keepdims=True) + 1e-9)
    R = np.sum(emb_norm * c_t, axis=1)
    seg_info = res.get("segments", {})
    seg_boundaries = seg_info.get("boundaries", [0, len(chunks)])
    seg_topics = seg_info.get("topics", [])

    specs = [[{"type": "scene"}, {"type": "xy"}],
             [{"type": "xy", "colspan": 2}, None]]
    titles = ["PHATE 逻辑流轨迹 (PELT 分段 + Leiden 段内主题)", "密度离群分析", "语义连贯性实时演变"]

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

    fig = make_subplots(rows=len(specs), cols=2, column_widths=[0.6, 0.4], specs=specs, subplot_titles=titles)
    phate_point_trace_indices = []

    if "phate" in res:
        p_coords = res["phate"]
        smooth_coords = smooth_3d_curve(p_coords, iterations=2)
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        # Global trajectory line in time order.
        fig.add_trace(go.Scatter3d(
            x=smooth_coords[:, 0], y=smooth_coords[:, 1], z=smooth_coords[:, 2],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.45)", width=2),
            name="全局轨迹"
        ), row=1, col=1)
        # Segment boundary polyline (adjacent boundaries only).
        if len(seg_boundaries) >= 2:
            b_idx = [max(0, min(int(b), len(p_coords) - 1)) for b in seg_boundaries]
            bx = [p_coords[i, 0] for i in b_idx]
            by = [p_coords[i, 1] for i in b_idx]
            bz = [p_coords[i, 2] for i in b_idx]
            fig.add_trace(go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines+markers",
                line=dict(color="rgba(0,0,0,0.6)", width=2.5),
                marker=dict(size=4, color="black", symbol="x"),
                name="分割点连线"
            ), row=1, col=1)
        n_seg = max(0, len(seg_boundaries) - 1)
        for s in range(n_seg):
            l, r = seg_boundaries[s], seg_boundaries[s + 1]
            idx = np.arange(l, r, dtype=int)
            if len(idx) <= 0:
                continue
            color = palette[int(s) % len(palette)]
            topic = seg_topics[s] if s < len(seg_topics) else -1
            hover_data = [
                [int(j + 1)]
                for j in idx
            ]
            fig.add_trace(go.Scatter3d(
                x=p_coords[idx, 0], y=p_coords[idx, 1], z=p_coords[idx, 2],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.7),
                customdata=hover_data,
                hovertemplate="语义块 %{customdata[0]}<extra></extra>",
                name=f"段 {s + 1} | 主题簇 {int(topic)}"
            ), row=1, col=1)
            phate_point_trace_indices.append(len(fig.data) - 1)

        fig.add_trace(go.Scatter3d(
            x=[p_coords[0, 0]], y=[p_coords[0, 1]], z=[p_coords[0, 2]], mode="markers+text",
            marker=dict(size=8, color="red", symbol="diamond"), text=["START"], name="起点"
        ), row=1, col=1)
        phate_point_trace_indices.append(len(fig.data) - 1)
        fig.add_trace(go.Scatter3d(
            x=[p_coords[-1, 0]], y=[p_coords[-1, 1]], z=[p_coords[-1, 2]], mode="markers+text",
            marker=dict(size=8, color="green", symbol="circle"), text=["END"], name="终点"
        ), row=1, col=1)
        phate_point_trace_indices.append(len(fig.data) - 1)

    fig.add_trace(go.Scatter(y=res.get("outliers", []), mode="markers",
                             marker=dict(color="rgba(255,0,0,0.5)", size=3), name="离群度"), row=1, col=2)
    fig.add_trace(go.Scatter(y=coh_data[1], mode="lines", fill="tozeroy", name="连贯性"), row=2, col=1)
    curr_row = 3
    if "tda" in res:
        dg = res["tda"]["diagrams"]
        for d in [0, 1, 2]:
            pts = dg[dg[:, 2] == d]
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"H{d}"), row=curr_row, col=1)
        fig.add_trace(go.Table(header=dict(values=["维度", "含义"]),
                               cells=dict(values=[["H0", "H1", "H2"], ["主题独立性", "逻辑闭环", "高阶空洞/多主题交错"]])),
                     row=curr_row, col=2)
        curr_row += 1

    if "phate" in res:
        cos_sim = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        d_t = np.arccos(cos_sim)
        L_vel = float(d_t.mean()) if len(d_t) else 0.0
        mu_v = emb.mean(axis=0)
        L_vol = float(np.sqrt(((emb - mu_v) ** 2).sum(axis=1).mean()))
        S_coh = float(np.mean(cos_sim)) if len(cos_sim) else 0.0
        alpha = 0.5
        beta = 0.5
        z_vel = (L_vel - np.mean(d_t)) / (np.std(d_t) + 1e-9) if len(d_t) else 0.0
        vol_series = np.sqrt(((emb - mu_v) ** 2).sum(axis=1))
        z_vol = (L_vol - np.mean(vol_series)) / (np.std(vol_series) + 1e-9) if len(vol_series) else 0.0
        score_load = float(S_coh * (alpha * z_vel + beta * z_vol))

        lam = 0.5
        D_shock = float(np.mean(np.abs(R[1:] - R[:-1]))) if len(R) > 1 else 0.0
        mu_R = float(np.mean(R)) if len(R) else 0.0
        raw_stab = mu_R - lam * D_shock
        z_stab = float((raw_stab - np.mean(R)) / (np.std(R) + 1e-9)) if len(R) else 0.0

        fig.add_trace(go.Scatter(y=R, mode="lines", name="主题相关性"), row=2, col=1)
        fig.add_trace(go.Table(
            header=dict(values=["分析指标", "量化数值", "深度解释"], fill_color="darkred", font=dict(color="white")),
            cells=dict(values=[
                ["语义推进速率", "语义空间体积", "语义连贯性", "逻辑负载度综合得分", "主题相关性均值", "稳定性得分"],
                [f"{L_vel:.4f}", f"{L_vol:.4f}", f"{S_coh:.4f}", f"{score_load:.4f}", f"{mu_R:.4f}", f"{z_stab:.4f}"],
                [r"$L_{vel} = \\frac{1}{N-1}\\sum_{t=1}^{N-1} \\arccos(\\cos_t)$",
                 r"$L_{vol} = \\sqrt{\\frac{1}{N}\\sum_{t=1}^{N} \\lVert v_t-\\mu \\rVert^2}$",
                 r"$S_{coh} = \\frac{1}{N-1}\\sum_{t=1}^{N-1} \\cos_t$",
                 r"$Score = S_{coh}\\cdot(0.5\\,Z(L_{vel})+0.5\\,Z(L_{vol}))$",
                 r"$\\mu_R = \\frac{1}{N}\\sum_{t=1}^{N} R(t)$",
                 r"$S_{stab} = Z(\\mu_R - 0.5\\,D_{shock})$"]
            ])
        ), row=curr_row, col=1)
        curr_row += 1

    chunk_idx = list(range(1, len(chunks) + 1))
    fig.add_trace(go.Table(
        header=dict(values=["Chunk", "文本"], fill_color="darkslategray", font=dict(color="white")),
        cells=dict(values=[chunk_idx, chunks], align=["center", "left"])
    ), row=curr_row, col=1)
    curr_row += 1

    seg_rows = list(range(1, max(0, len(seg_boundaries) - 1) + 1))
    seg_ranges = []
    seg_topic_vals = []
    seg_sizes = []
    for i in range(max(0, len(seg_boundaries) - 1)):
        l, r = int(seg_boundaries[i]), int(seg_boundaries[i + 1])
        seg_ranges.append(f"[{l}, {r})")
        seg_topic_vals.append(int(seg_topics[i]) if i < len(seg_topics) else -1)
        seg_sizes.append(max(0, r - l))
    fig.add_trace(go.Table(
        header=dict(values=["段", "区间", "topic_cluster", "长度"], fill_color="darkslateblue", font=dict(color="white")),
        cells=dict(values=[seg_rows, seg_ranges, seg_topic_vals, seg_sizes], align=["center", "center", "center", "center"])
    ), row=curr_row, col=1)
    curr_row += 1

    phate_params = res.get("phate_params", {})
    param_names = [
        "WHISPER_MODEL",
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
    param_values = [
        WHISPER_MODEL,
        SBERT_MODEL_NAME,
        str(WINDOW_SIZE),
        str(STEP_SIZE),
        str(phate_params.get("n_components", PHATE_N_COMPONENTS)),
        str(phate_params.get("knn", PHATE_KNN)),
        str(phate_params.get("knn_dist", PHATE_KNN_DIST)),
        str(phate_params.get("n_pca", PHATE_N_PCA)),
        str(phate_params.get("mds", PHATE_MDS)),
        str(phate_params.get("mds_solver", PHATE_MDS_SOLVER)),
        "embedding (L2 normalized)",
        str(res.get("outlier_k", "n/a")),
        f"k=clip(sqrt(n), {OUTLIER_K_MIN}, {OUTLIER_K_MAX})",
        str(res.get("cluster_backend", "unknown")),
        str(LEIDEN_RESOLUTION),
        str(HDBSCAN_MIN_CLUSTER_SIZE),
        str(HDBSCAN_MIN_SAMPLES),
        str(seg_info.get("method", "unknown")),
      
        str(ENABLE_TDA),
    ]
    fig.add_trace(go.Table(
        header=dict(values=["参数", "值"], fill_color="darkred", font=dict(color="white")),
        cells=dict(values=[param_names, param_values], align=["left", "left"])
    ), row=curr_row, col=1)

    n_traces = len(fig.data)
    vis_points_on = [True] * n_traces
    vis_points_off = [True] * n_traces
    for idx in phate_point_trace_indices:
        if 0 <= idx < n_traces:
            vis_points_off[idx] = False

    fig.update_layout(
        height=450 * len(specs),
        title=f"🎬 PHATE 语义拓扑深度逻辑报告 - {video_title}",
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.08,
                showactive=True,
                buttons=[
                    dict(label="显示点", method="update", args=[{"visible": vis_points_on}]),
                    dict(label="隐藏点", method="update", args=[{"visible": vis_points_off}]),
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
    print(f"📊 报告已生成: {report_path}")

# ==================== 入口 ====================

def process_video(url):
    script_dir = get_script_dir()
    video_title = get_video_title(url)
    safe_title = sanitize_filename(video_title)
    output_dir = os.path.join(script_dir, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    audio_file = os.path.join(output_dir, "audio.wav")

    with SimpleTimer("YouTube 下载"):
        download_audio(url, audio_file)

    with SimpleTimer("Whisper 转录"):
        segments = load_or_transcribe(audio_file, output_dir)

    chunks = load_or_build_chunks(segments, output_dir)

    with SimpleTimer("SBERT 编码"):
        embeddings_tensor = load_or_embed(chunks, output_dir)

    analyzer = LogicAnalyzer(embeddings_tensor)
    with SimpleTimer("PHATE 轨迹生成 (高维直投)"):
        analyzer.run_phate()
    if TDA_AVAILABLE and ENABLE_TDA:
        with SimpleTimer("TDA 拓扑分析"):
            analyzer.run_tda()
    with SimpleTimer("图聚类 (Leiden/HDBSCAN)"):
        analyzer.run_graph_clustering()
    with SimpleTimer("PELT 语义分段"):
        analyzer.run_pelt_segmentation()

    with SimpleTimer("连贯性计算"):
        coh_data = compute_coherence(embeddings_tensor)

    save_modular_report(chunks, analyzer, coh_data, output_dir, embeddings_tensor, video_title)


def main():
    process_video(YOUTUBE_URL)


if __name__ == "__main__":
    main()
