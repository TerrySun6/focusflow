import os
import time
import warnings

try:
    import importlib
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# ==================== 🛠 配置区 ====================
ENABLED_MODULES = ["DOWNLOAD", "WHISPER", "SBERT", "COHERENCE", '''"TDA"''', "PHATE", "HDBSCAN"]

YOUTUBE_URL = "https://www.youtube.com/watch?v=iG9CE55wbtY&t=979s"
WHISPER_MODEL = "mlx-community/whisper-small-mlx-4bit"
SBERT_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
WINDOW_SIZE, STEP_SIZE = 5, 1
DEVICE = "mps"
CHANGE_Z = 2.0

FETCH_TOP_TED = True
TED_CHANNEL_POPULAR_URL = "https://www.youtube.com/@TED/videos?view=0&sort=p&flow=grid"
TED_TOP_N = 10
TED_POPULAR_MODE = "page"  # "page" uses popular order; "view_count" is slower and rate-limit prone
TED_REQUEST_DELAY = 2


class SimpleTimer:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.start = time.time()
        print(f"▶️ [开始] {self.name}...")
        return self
    def __exit__(self, *args):
        print(f"✅ [完成] {self.name} | 耗时: {time.time() - self.start:.2f}s")


def process_video(url, remote_components, js_runtimes):
    script_dir = get_script_dir()
    video_title = get_video_title(url, remote_components, js_runtimes)
    safe_title = sanitize_filename(video_title)
    output_dir = os.path.join(script_dir, safe_title)
    os.makedirs(output_dir, exist_ok=True)

    audio_file = os.path.join(output_dir, "audio.wav")

    if "DOWNLOAD" in ENABLED_MODULES:
        with SimpleTimer("YouTube 下载"):
            download_audio(url, audio_file, remote_components, js_runtimes)

    if "WHISPER" in ENABLED_MODULES:
        with SimpleTimer("Whisper 转录"):
            segments = load_or_transcribe(audio_file, WHISPER_MODEL, output_dir)

    chunks = load_or_build_chunks(segments, WINDOW_SIZE, STEP_SIZE, output_dir)

    if "SBERT" in ENABLED_MODULES:
        with SimpleTimer("SBERT 编码"):
            embeddings_tensor = load_or_embed(chunks, SBERT_MODEL_NAME, DEVICE, output_dir)

    emb_hdb_labels = None
    if "HDBSCAN" in ENABLED_MODULES:
        emb_hdb_labels = run_embedding_hdbscan(embeddings_tensor, min_cluster_size=12, min_samples=8)

    analyzer = LogicAnalyzer(embeddings_tensor)
    analyzer.results['emb_hdb_labels'] = emb_hdb_labels
    if "PHATE" in ENABLED_MODULES:
        with SimpleTimer("PHATE 轨迹生成 (高维直投)"):
            analyzer.run_phate(phate if PHATE_AVAILABLE else None)
    if "TDA" in ENABLED_MODULES and TDA_AVAILABLE:
        with SimpleTimer("TDA 拓扑分析"):
            analyzer.run_tda(gtda)
    if "HDBSCAN" in ENABLED_MODULES:
        with SimpleTimer("HDBSCAN 密度分析"):
            analyzer.run_hdbscan()

    coh_data = (0, [])
    if "COHERENCE" in ENABLED_MODULES:
        with SimpleTimer("连贯性计算"):
            coh_data = compute_coherence(embeddings_tensor)

    save_modular_report(chunks, analyzer, coh_data, output_dir, embeddings_tensor, video_title, change_z=CHANGE_Z)


def main():
    remote_components = ["ejs:github"]
    js_runtimes = {"deno": {}}

    if FETCH_TOP_TED:
        videos = get_top_ted_videos(
            TED_CHANNEL_POPULAR_URL,
            top_n=TED_TOP_N,
            mode=TED_POPULAR_MODE,
            request_delay=TED_REQUEST_DELAY,
            remote_components=remote_components,
            js_runtimes=js_runtimes,
        )
        for v in videos:
            if not v.get("url"):
                continue
            print(f"🎯 处理: {v.get('title')}")
            process_video(v["url"], remote_components, js_runtimes)
    else:
        process_video(YOUTUBE_URL, remote_components, js_runtimes)


if __name__ == "__main__":
    main()
