#!/usr/bin/env python3
"""批量测试脚本：对 test_texts/ 下所有 txt 文件运行语义分析管线，输出对比表格。

用法：
    cd /Users/terrysun/Documents/learning/project/focusflow
    python latest_code/text_var/batch_test_texts.py

参数：
    ema_alpha = 0.8，其他使用 Config 默认值。
"""

import json
import os
import sys

# 确保能 import text_baai_ema_flask_centering 模块
# 模块位于 text_baai_ema_flask_centering/ 子目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(_SCRIPT_DIR, "text_baai_ema_flask_centering")
sys.path.insert(0, _MODULE_DIR)

from text_baai_ema_flask_centering import Config, _run_analysis_pipeline

# ── 配置 ──
TEST_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "..", "test_texts")
TEST_DIR = os.path.abspath(TEST_DIR)

cfg = Config()
cfg.ema_alpha = 0.8          # 用户指定
cfg.ema_window_size = 0      # 默认
cfg.phate_knn = 5
cfg.phate_knn_dist = "cosine"
cfg.mds_method = "metric"
cfg.pelt_penalty_multiplier = 5.0
cfg.leiden_resolution = 1.0
cfg.hdbscan_min_cluster_size = 5

print(f"配置: α={cfg.ema_alpha}, window={cfg.ema_window_size}, knn={cfg.phate_knn}")
print(f"测试目录: {TEST_DIR}")
print()

# ── 收集所有 txt 文件 ──
txt_files = []
for fname in sorted(os.listdir(TEST_DIR)):
    if fname.endswith(".txt"):
        txt_files.append(os.path.join(TEST_DIR, fname))

print(f"找到 {len(txt_files)} 个测试文件\n")

# ── 逐文件运行分析 ──
results = []
for fpath in txt_files:
    fname = os.path.basename(fpath)
    title = os.path.splitext(fname)[0]
    print(f"▶ 分析中: {fname} ...", end=" ", flush=True)

    try:
        result = _run_analysis_pipeline(fpath, title, cfg)

        # 提取关键指标
        metrics = {
            "文件": fname,
            "句数": result.get("n_chunks", 0),
            "连贯性(前)": round(result.get("coherence_before", 0), 4),
            "连贯性(后)": round(result.get("coherence_after", 0), 4),
            "Δ连贯性": round(result.get("coherence_after", 0) - result.get("coherence_before", 0), 4),
            "聚类簇": result.get("n_clusters", 0),
            "分段数": result.get("n_segments", 0),
        }

        # Tortuosity (弯曲度比)
        trj = result.get("trajectory", {})
        if trj:
            metrics["Tortuosity"] = round(trj.get("ratio", -1), 2)
        else:
            metrics["Tortuosity"] = "N/A"

        results.append(metrics)
        print("✓")
    except Exception as e:
        print(f"✗ 失败: {e}")
        results.append({"文件": fname, "句数": "ERR", "连贯性(前)": str(e)[:60]})

# ── 输出表格 ──
print("\n" + "=" * 110)
print("测试结果汇总")
print("=" * 110)

# 手动格式化表格（避免依赖外部库）
HEADERS = ["文件", "句数", "连贯性(前)", "连贯性(后)", "Δ连贯性", "聚类簇", "分段数", "Tortuosity"]
COL_WIDTHS = [40, 6, 12, 12, 10, 8, 8, 12]

# 表头
header_line = ""
for h, w in zip(HEADERS, COL_WIDTHS):
    header_line += f"{h:<{w}}"
print(header_line)
print("-" * sum(COL_WIDTHS))

# 数据行
for r in results:
    row = ""
    for h, w in zip(HEADERS, COL_WIDTHS):
        val = r.get(h, "")
        row += f"{str(val):<{w}}"
    print(row)

print("-" * sum(COL_WIDTHS))

# ── 按逻辑性分类汇总 ──
print("\n--- 按分类汇总 ---")
logical = [r for r in results if "有逻辑" in r["文件"]]
illogical = [r for r in results if "无逻辑" in r["文件"]]
boundary = [r for r in results if "边界" in r["文件"]]

def avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else "N/A"

print(f"有逻辑组 ({len(logical)} 篇): 平均 Tortuosity = {avg(logical, 'Tortuosity')}, 平均分段数 = {avg(logical, '分段数')}")
print(f"无逻辑组 ({len(illogical)} 篇): 平均 Tortuosity = {avg(illogical, 'Tortuosity')}, 平均分段数 = {avg(illogical, '分段数')}")
print(f"边界组   ({len(boundary)} 篇): 平均 Tortuosity = {avg(boundary, 'Tortuosity')}, 平均分段数 = {avg(boundary, '分段数')}")

print("\n✅ 批量测试完成。")
print("提示：Tortuosity 越低表示逻辑越聚焦，分段数在同等句数下越少表示结构越清晰。")
