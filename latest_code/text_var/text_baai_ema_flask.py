"""
Flask UI for text_baai_ema.py
===============================
极简界面：输入文件路径 → 调整参数 → 查看结果。
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
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests  # 用于 DeepSeek API 调用

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
        """滑动窗口 EMA（指数衰减加权平均）。
        
        对每个位置 i，取以 i 为中心的奇数大小窗口 [i-k, i+k]，
        窗口内各 embedding 的权重按到中心的距离指数衰减：w(j) = α^|j|，
        然后做加权平均。步长 = 1，遍历所有位置。
        
        数学公式：
            out[i] = ( Σ_{j=-k}^{k} α^{|j|} · emb[i+j] )
                     / ( Σ_{j=-k}^{k} α^{|j|} )
        
        边界处窗口自动截断并重新归一化权重。
        """
        emb = self.embeddings_gpu          # (n, d)
        n = emb.shape[0]
        alpha = self.cfg.ema_alpha
        window = self.cfg.ema_window_size  # 滑动窗口大小（奇数）

        # window <= 0 或 n 太小 → 不平滑
        if window <= 0 or n < 2:
            self.embeddings_cpu = emb.detach().cpu().numpy()
            return self.embeddings_cpu

        # 确保窗口为奇数
        if window % 2 == 0:
            window += 1
        k = window // 2                     # 单侧半径

        # 预计算权重：α^|j|  for  j = -k ... +k
        j = torch.arange(-k, k + 1, device=emb.device)
        weights = alpha ** torch.abs(j)     # (W,)

        out = torch.zeros_like(emb)

        for i in range(n):
            # 当前窗口边界（截断到有效范围）
            left = max(0, i - k)
            right = min(n, i + k + 1)

            # 窗口内的 embedding
            w = emb[left:right]             # (W_valid, d)

            # 从权重向量中截取对应的片段
            w_left = left - (i - k)          # 左侧被截断的权重数
            w_right = (i + k + 1) - right    # 右侧被截断的权重数
            valid_weights = weights[w_left:window - w_right]  # (W_valid,)

            # 归一化权重 → 加权平均
            valid_weights = valid_weights / valid_weights.sum()
            out[i] = (w * valid_weights.unsqueeze(1)).sum(dim=0)

        self.embeddings_gpu = out
        self.embeddings_cpu = out.detach().cpu().numpy()
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
        emb = self.embeddings_cpu
        n = len(emb)
        clusters = self.results.get("clusters", np.zeros(n, dtype=int))
        # 用原始 embedding（EMA 前的值）计算 median_dist，确保惩罚系数稳定可控
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
                    final_ts.append(ct); ct = topics[i]
            final_bs.append(boundaries[-1]); final_ts.append(ct)
        else:
            final_bs, final_ts = boundaries, topics
        seg = {"boundaries": final_bs, "topics": final_ts, "n_segments": len(final_bs) - 1}
        self.results["segments"] = seg
        return seg

    def get_coherence(self, emb: torch.Tensor) -> np.ndarray:
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


# 线程池：用于并行执行分析管线中的独立步骤
_analysis_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="analyze")


# ── Flask App ──
app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>📄 语义分析器</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: #f5f7fa; color: #1a1a2e; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { font-size: 1.6rem; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        .layout { display: flex; gap: 20px; }
        .sidebar { width: 300px; flex-shrink: 0; }
        .main { flex: 1; min-width: 0; }
        .card { background: white; border-radius: 12px; padding: 16px; margin-bottom: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .card h3 { font-size: 0.95rem; color: #555; margin-bottom: 12px; }
        label { display: block; font-size: 0.85rem; color: #666; margin-bottom: 4px; margin-top: 10px; }
        input[type="text"], input[type="number"], select {
            width: 100%; padding: 8px 10px; border: 1px solid #ddd; border-radius: 8px;
            font-size: 0.9rem; }
        input[type="range"] { width: 100%; margin: 4px 0; }
        .range-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #888; }
        .btn { width: 100%; padding: 10px; background: #4a6cf7; color: white; border: none;
               border-radius: 8px; font-size: 1rem; cursor: pointer; font-weight: 600; }
        .btn:hover { background: #3a5ce5; }
        .btn:disabled { background: #aaa; cursor: not-allowed; }
        .checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
        .checkbox-row input { width: auto; }
        .chart { width: 100%; height: 500px; }
        .chart-sm { height: 300px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
        .metric { background: #f8f9fc; border-radius: 8px; padding: 12px; text-align: center; }
        .metric .val { font-size: 1.3rem; font-weight: 700; color: #4a6cf7; }
        .metric .lbl { font-size: 0.75rem; color: #888; }
        .loading { display: none; text-align: center; padding: 40px; }
        .loading.active { display: block; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #4a6cf7; border-radius: 50%;
                   width: 36px; height: 36px; animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #fee; color: #c33; padding: 12px; border-radius: 8px; margin: 10px 0; }
        .tabs { display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
        .tab { padding: 8px 16px; border-radius: 8px 8px 0 0; cursor: pointer; font-size: 0.85rem;
               background: #eee; color: #666; }
        .tab.active { background: white; color: #4a6cf7; font-weight: 600; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fc; font-weight: 600; color: #555; }
        .seg-text { max-height: 150px; overflow-y: auto; background: #fafafa; padding: 8px;
                    border-radius: 6px; font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap; }
        @media (max-width: 900px) { .layout { flex-direction: column; } .sidebar { width: 100%; } }
    </style>
</head>
<body>
<div class="container">
    <h1>📄 语义分析器</h1>
    <div class="layout">
        <div class="sidebar">
            <div class="card">
                <h3>📂 文档</h3>
                <label>文件绝对路径</label>
                <input type="text" id="filePath" placeholder="/path/to/your/text.txt"
                       value="{{ request.args.get('path', '') }}">
                <label>文档标题</label>
                <input type="text" id="docTitle" placeholder="my_document" value="my_document">
                <button class="btn" onclick="loadDoc()" style="margin-top:12px;">📥 加载文档</button>
            </div>
            <div class="card" id="paramsCard">
                <h3>🔄 EMA 平滑</h3>
                <label>EMA α (0=不平滑, 1=极端)</label>
                <input type="range" id="emaAlpha" min="0" max="0.95" step="0.05" value="0.5"
                       oninput="document.getElementById('alphaVal').textContent=this.value">
                <div class="range-label"><span>0</span><span id="alphaVal">0.5</span><span>0.95</span></div>
                <div class="checkbox-row">
                    <input type="checkbox" id="emaBidirectional" checked>
                    <label style="margin:0;">双向 EMA</label>
                </div>
                <label>EMA 窗口大小 (0=全文, >0=局部)</label>
                <input type="number" id="emaWindow" value="0" min="0" max="200" step="1">

                <h3 style="margin-top:16px;">🎯 PHATE</h3>
                <label>KNN 邻居数</label>
                <input type="number" id="phateKnn" value="5" min="2" max="30">
                <label>距离度量</label>
                <select id="phateDist">
                    <option value="cosine">cosine</option>
                    <option value="euclidean">euclidean</option>
                    <option value="manhattan">manhattan</option>
                </select>
                <label>MDS 方法</label>
                <select id="mdsMethod">
                    <option value="metric">metric (SMACOF)</option>
                    <option value="nonmetric">nonmetric</option>
                </select>

                <h3 style="margin-top:16px;">🔗 聚类</h3>
                <label>Leiden 分辨率</label>
                <input type="range" id="leidenRes" min="0.1" max="3.0" step="0.1" value="1.0"
                       oninput="document.getElementById('resVal').textContent=this.value">
                <div class="range-label"><span>0.1</span><span id="resVal">1.0</span><span>3.0</span></div>
                <label>HDBSCAN 最小簇</label>
                <input type="number" id="hdbscanMin" value="5" min="2" max="20">

                <h3 style="margin-top:16px;">✂️ PELT 分段</h3>
                <label>惩罚系数 (越小段越多, 越大段越少)</label>
                <input type="range" id="peltPenalty" min="0.1" max="20.0" step="0.1" value="5.0"
                       oninput="document.getElementById('peltVal').textContent=this.value">
                <div class="range-label"><span>0.1</span><span id="peltVal">5.0</span><span>20.0</span></div>


                <button class="btn" onclick="runAnalysis()" style="margin-top:18px;">🚀 运行分析</button>
            </div>

            <div class="card">
                <h3>🤖 DeepSeek 摘要</h3>
                <button class="btn" onclick="summarizeSegments()" style="margin-top:4px;background:#10a37f;" id="summarizeBtn">📝 生成分段摘要</button>
                <div style="font-size:0.75rem;color:#888;margin-top:6px;">先运行分析，再点击生成摘要</div>
            </div>
        </div>

        <div class="main">
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <div>分析中... (EMA + PHATE + 聚类 + PELT)</div>
            </div>
            <div id="error" class="error" style="display:none;"></div>
            <div id="results" style="display:none;">
                <div class="tabs" id="tabs">
                    <div class="tab active" onclick="switchTab('tab-overview',this)">📊 概览</div>
                    <div class="tab" onclick="switchTab('tab-coherence',this)">🔬 EMA 效果</div>
                    <div class="tab" onclick="switchTab('tab-phate',this)">🌐 3D 轨迹</div>
                    <div class="tab" onclick="switchTab('tab-segments',this)">📋 分段</div>
                    <div class="tab" onclick="switchTab('tab-summary',this)">🤖 摘要</div>
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
                <div id="tab-summary" class="tab-content">
                    <div class="card"><div id="summaryContent"></div></div>
                </div>
                <div id="tab-log" class="tab-content">
                    <div class="card"><pre id="logContent" style="max-height:400px;overflow:auto;font-size:0.8rem;background:#f8f9fc;padding:12px;border-radius:8px;"></pre></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
let currentResult = null;

function switchTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    el.classList.add('active');
    // 重新布局 Plotly 图表
    setTimeout(() => {
        document.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
    }, 100);
}

// 按下 Enter 加载文档
document.getElementById('filePath').addEventListener('keydown', e => { if (e.key === 'Enter') loadDoc(); });

async function loadDoc() {
    const path = document.getElementById('filePath').value.trim();
    if (!path) { showError('请输入文件路径'); return; }
    showLoading();
    hideError();
    try {
        const resp = await fetch('/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path, title: document.getElementById('docTitle').value})
        });
        const data = await resp.json();
        hideLoading();
        if (data.error) { showError(data.error); return; }
        document.getElementById('results').style.display = 'none';
        alert(`✅ 文档已加载: ${data.n_chunks} 个句子, embeddings: ${data.has_embeddings}`);
    } catch(e) { hideLoading(); showError('加载失败: ' + e.message); }
}

async function runAnalysis() {
    const path = document.getElementById('filePath').value.trim();
    if (!path) { showError('请输入文件路径'); return; }
    showLoading();
    hideError();
    try {
        const resp = await fetch('/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                path,
                title: document.getElementById('docTitle').value,
                ema_alpha: parseFloat(document.getElementById('emaAlpha').value),
                ema_bidirectional: document.getElementById('emaBidirectional').checked,
                ema_window_size: parseInt(document.getElementById('emaWindow').value),
                phate_knn: parseInt(document.getElementById('phateKnn').value),
                phate_knn_dist: document.getElementById('phateDist').value,
                mds_method: document.getElementById('mdsMethod').value,
                leiden_resolution: parseFloat(document.getElementById('leidenRes').value),
                hdbscan_min_cluster_size: parseInt(document.getElementById('hdbscanMin').value),
                pelt_penalty_multiplier: parseFloat(document.getElementById('peltPenalty').value),
            })
        });
        const data = await resp.json();
        hideLoading();
        if (data.error) { showError(data.error); return; }
        currentResult = data;
        renderResults(data);
    } catch(e) { hideLoading(); showError('分析失败: ' + e.message); }
}

function renderResults(d) {
    document.getElementById('results').style.display = 'block';

    // ── 概览指标 ──
    let metricsHtml = `
        <div class="metric"><div class="val">${d.n_chunks}</div><div class="lbl">句子数</div></div>
        <div class="metric"><div class="val">${d.ema_alpha}</div><div class="lbl">EMA α</div></div>
        <div class="metric"><div class="val">${d.ema_direction}</div><div class="lbl">EMA 方向</div></div>
        <div class="metric"><div class="val">${d.n_clusters}</div><div class="lbl">聚类簇</div></div>
        <div class="metric"><div class="val">${d.n_segments}</div><div class="lbl">语义段</div></div>
        <div class="metric"><div class="val">${d.coherence_before.toFixed(4)}</div><div class="lbl">平滑前连贯性</div></div>
        <div class="metric"><div class="val">${d.coherence_after.toFixed(4)}</div><div class="lbl">平滑后连贯性</div></div>
        <div class="metric"><div class="val">${(d.coherence_after - d.coherence_before) > 0 ? '+' : ''}${(d.coherence_after - d.coherence_before).toFixed(4)}</div><div class="lbl">Δ 变化</div></div>
    `;
    document.getElementById('metrics').innerHTML = metricsHtml;

    // ── 簇分布柱状图 ──
    if (d.cluster_chart_json) {
        Plotly.newPlot('clusterChart', JSON.parse(d.cluster_chart_json).data,
                       JSON.parse(d.cluster_chart_json).layout || {}, {responsive: true});
    }

    // ── 连贯性图 ──
    if (d.coherence_chart_json) {
        Plotly.newPlot('coherenceChart', JSON.parse(d.coherence_chart_json).data,
                       JSON.parse(d.coherence_chart_json).layout || {}, {responsive: true});
    }

    // ── 连贯性样本表 ──
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
        let trjHtml = `
            <div class="metric"><div class="val">${d.trajectory.total_dist.toFixed(2)}</div><div class="lbl">总轨迹长度</div></div>
            <div class="metric"><div class="val">${d.trajectory.end_dist.toFixed(2)}</div><div class="lbl">起点→终点距</div></div>
            <div class="metric"><div class="val">${d.trajectory.ratio.toFixed(2)}</div><div class="lbl">弯曲度比</div></div>
        `;
        document.getElementById('trajectoryMetrics').innerHTML = trjHtml;
    }

    // ── 分段表 ──
    if (d.segments) {
        let segTbl = '<table><tr><th>段</th><th>区间</th><th>长度</th><th>主题簇</th><th>起始文本</th></tr>';
        d.segments.forEach(s => {
            segTbl += `<tr><td>${s.id}</td><td>${s.range}</td><td>${s.len}</td><td>T${s.topic}</td><td>${s.preview}</td></tr>`;
        });
        segTbl += '</table>';
        document.getElementById('segmentsTable').innerHTML = segTbl;

        // ── 各段文本 ──
        let segTextHtml = '';
        d.segments.forEach(s => {
            segTextHtml += `<div style="margin-bottom:12px;"><strong>段 ${s.id}</strong> | [${s.range}] | T${s.topic} | ${s.len} 句</div>
                            <div class="seg-text">${s.full_text.join('\n\n')}</div><hr style="margin:12px 0;border:none;border-top:1px solid #eee;">`;
        });
        document.getElementById('segmentsText').innerHTML = segTextHtml;
    }

    // ── 日志 ──
    if (d.log) {
        document.getElementById('logContent').textContent = d.log.join('\n');
    }

    // 切换到概览 tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelector('.tab').classList.add('active');
    document.getElementById('tab-overview').classList.add('active');
}

async function summarizeSegments() {
    if (!currentResult || !currentResult.segments) {
        showError('请先运行分析，获取分段结果');
        return;
    }
    document.getElementById('summarizeBtn').disabled = true;
    document.getElementById('summarizeBtn').textContent = '⏳ 生成中...';
    hideError();

    // 显示原始流式输出
    const summaryDiv = document.getElementById('summaryContent');
    summaryDiv.innerHTML = '<pre id="streamOutput" style="white-space:pre-wrap;font-size:0.85rem;line-height:1.6;max-height:600px;overflow-y:auto;background:#f9f9fa;padding:12px;border-radius:8px;">等待响应...</pre>';
    const streamPre = document.getElementById('streamOutput');
    // 切换到摘要 tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab')[4].classList.add('active');
    document.getElementById('tab-summary').classList.add('active');

    try {
        // 把每个 segment 的 full_text 数组合并成一个字符串发送
        const segTexts = currentResult.segments.map(s => s.full_text.join('\n'));
        const resp = await fetch('/summarize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ segments: segTexts })
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            showError(errData.error || `HTTP ${resp.status}`);
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let rawContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.type === 'chunk') {
                            rawContent += data.content;
                            streamPre.textContent = rawContent;
                        } else if (data.type === 'error') {
                            showError(data.message);
                        }
                    } catch(e) {}
                }
            }
        }
        streamPre.textContent = rawContent || '[无内容]';
    } catch(e) {
        showError('摘要生成失败: ' + e.message);
    } finally {
        document.getElementById('summarizeBtn').disabled = false;
        document.getElementById('summarizeBtn').textContent = '📝 生成分段摘要';
    }
}

function showLoading() { document.getElementById('loading').classList.add('active'); }
function hideLoading() { document.getElementById('loading').classList.remove('active'); }
function showError(msg) { document.getElementById('error').textContent = msg; document.getElementById('error').style.display = 'block'; }
function hideError() { document.getElementById('error').style.display = 'none'; }
</script>
</body>
</html>
"""


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


# ── Flask Routes ──

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


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
        orig_sim = F.cosine_similarity(orig_emb[:-1], orig_emb[1:], dim=1).cpu().numpy()
        coh_before = float(orig_sim.mean())

        # EMA
        analyzer = EMAnalyzer(orig_emb.clone(), cfg)
        smooth_emb = analyzer.run_ema()
        smooth_sim = F.cosine_similarity(analyzer.embeddings_gpu[:-1], analyzer.embeddings_gpu[1:], dim=1).cpu().numpy()
        coh_after = float(smooth_sim.mean())

        # PHATE
        phate_coords = analyzer.run_phate()

        # 聚类
        labels, outliers = analyzer.run_clustering()
        n_clusters = len(set(labels) - {-1}) if labels is not None and len(labels) > 0 else 0

        # PELT
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
        result["coherence_chart_json"] = json.dumps(json.loads(plotly.io.to_json(coh_fig)))

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
            result["cluster_chart_json"] = json.dumps(json.loads(plotly.io.to_json(cluster_fig)))

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
            result["phate_chart_json"] = json.dumps(json.loads(plotly.io.to_json(phate_fig)))

            # ── 导出自包含交互式 HTML ──
            _export_phate_html(phate_fig, doc_dir, title)

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

        logging.getLogger().removeHandler(log_handler)
        return result

    except Exception as e:
        logging.getLogger().removeHandler(log_handler)
        logger.exception("分析失败")
        raise


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
    print("📄 语义分析器 - Flask UI")
    print("=" * 50)
    print(f"🔗 打开浏览器访问: http://127.0.0.1:8080")
    print(f"📂 在输入框中粘贴文本文件的绝对路径后点击「加载文档」或按 Enter")
    print(f"⚡ 性能优化: 内存缓存已启用 | 线程池 active | threaded=True")
    print("=" * 50)
    app.run(debug=False, threaded=True, host="127.0.0.1", port=8080)
