# FocusFlow — 基于 PHATE 降维的语义流形分析与逻辑可视化

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**FocusFlow** 是一个用于文本/视频语义逻辑分析的 Python 研究项目。它通过 **BGE-M3 嵌入** + **EMA 时序平滑** + **Centering 去偏** + **PHATE 3D 降维** + **Leiden/HDBSCAN 聚类** + **PELT 变点检测** 的完整管线，将文本的逻辑结构映射到三维语义流形上，并进行定量分析和可视化。

---

## ✨ 核心特性

- **语义嵌入**：使用 BGE-M3（1024 维）多语言模型对句子进行编码
- **EMA 平滑**：指数移动平均窗口，减少措辞波动噪声
- **Centering 去偏**：零均值中心化消除大模型的各向异性偏置（Anisotropy Bias）
- **PHATE 降维**：扩散势能距离嵌入，保留局部 + 全局语义结构
- **多方法对比**：PCA / t-SNE / UMAP / PHATE 定量评估（Trustworthiness, Smoothness, ARI, Tortuosity）
- **聚类分段**：Leiden 模块度优化 + HDBSCAN 密度聚类 + PELT 变点检测
- **DeepSeek 摘要**：流式 API 调用，为每个语义分段生成摘要
- **3D 可视化**：Plotly 交互式图表，支持连贯性对比、簇分布、PHATE 3D 流形

---

## 📐 技术管线

```
文本输入 → BGE-M3 嵌入 → EMA 滑动平滑 → Centering 中心化 
→ PHATE 3D 降维 → Leiden/HDBSCAN 聚类 → PELT 变点分段 
→ DeepSeek 摘要 + Plotly 可视化
```

详细数学原理见 [`项目数学原理手册.md`](项目数学原理手册.md)。

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Apple Silicon Mac 推荐（支持 MLX 加速），Intel Mac / Linux 也可运行

### 安装

```bash
git clone https://github.com/TerrySun6/focusflow.git
cd focusflow
pip install -r requirements.txt
```

### 运行

**文本分析（主程序 — Flask Web UI）**：

```bash
cd latest_code/text_var/text_baai_ema_flask_centering
python text_baai_ema_flask_centering.py
```

打开浏览器访问 `http://localhost:5000`，输入文本文件路径或粘贴文本即可分析。

**降维方法对比**：

```bash
cd latest_code/compare
python compare_pca_tsne_phate_centering.py
```

**命令行报告生成**：

```bash
python latest_code/text_var/reporter.py
```

---

## 📂 项目结构

```
focusflow/
├── README.md                         # 项目说明
├── requirements.txt                  # Python 依赖
├── .gitignore                        # Git 忽略规则
├── 项目数学原理手册.md                # 数学原理详解（面试/答辩用）
│
├── latest_code/                      # 核心代码
│   ├── text_var/
│   │   ├── text_baai_ema_flask_centering/  # ★ 主程序：Flask UI + 完整管线
│   │   ├── text_baai_ema_flask_umap/       # UMAP 替代 PHATE 版本
│   │   ├── text_baai_ema_flask_semantic.py  # 语义分析变体
│   │   ├── text_baai_ema.py                # 无 Flask 版本（EMA 核心）
│   │   ├── text_baai.py                    # 基础 BGE-M3 嵌入版本
│   │   ├── text_baai_gnn.py                # GNN 图神经网络实验
│   │   ├── reporter.py                     # 报告生成器（MLX 版）
│   │   └── run_batch_tests.py              # 批量测试脚本
│   │
│   ├── video_var/                    # 视频分析管线
│   │   ├── video_flask_centering.py   # 视频语义分析（含 Whisper）
│   │   └── video_qwen.py             # Qwen 编码器实验
│   │
│   ├── compare/                      # 降维方法定量对比
│   │   ├── compare_pca_tsne_phate_centering.py
│   │   ├── compare_pca_tsne_phate_umap.py
│   │   ├── compare_pelt_vs_semantic.py
│   │   └── compare_pca_tsne_phate_metrics.json
│   │
│   └── tools/                        # 工具脚本
│       └── epub_to_txt.py            # EPUB 转 TXT
│
├── test_texts/                       # 测试文本集（有逻辑/无逻辑/边界）
│   └── README.md                     # 测试集说明
│
├── test.py                           # TED 视频批处理测试
└── reporter_original.py              # 早期版本参考
```

---

## 📊 降维方法对比

基于《小王子》1331 句的对比测试：

| 方法 | Trustworthiness ↑ | Smoothness ↓ | ARI ↑ | Tortuosity ↓ |
|------|:-----------------:|:------------:|:-----:|:------------:|
| PCA  | 0.813             | 1.186        | 0.106 | 875.89       |
| t-SNE| **0.950**         | 1.015        | 0.292 | 99.82        |
| PHATE| 0.818             | **1.009**    | **0.552** | 82.48     |

> PHATE 在聚类一致性（ARI）和平滑度（Smoothness）上表现最优，且最能保持低维轨迹的语义连贯性。

---

## 🔬 关键数学创新

1. **Centering 去偏**：解决大模型 embedding 的各向异性偏置问题，将余弦相似度基线从 ~0.85 降至 ~0.02
2. **EMA 双重加权**：指数衰减权重 + 双向滑动窗口，平滑局部噪声的同时保留宏观逻辑跳跃
3. **PHATE + 轨迹度量**：Tortuosity（弯曲度）作为文本逻辑复杂度的定量指标
4. **PELT + 聚类交叉验证**：分段边界与聚类标签双重确认语义转折点

---

## 📄 许可证

Apache License 2.0 © 2026 TerrySun6

---

## 🙏 致谢

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — BAAI 多语言嵌入模型
- [PHATE](https://github.com/KrishnaswamyLab/PHATE) — 热扩散势能嵌入
- [DeepSeek](https://deepseek.com/) — LLM API 支持
- [Plotly](https://plotly.com/) — 交互式可视化
