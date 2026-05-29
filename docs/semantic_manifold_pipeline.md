# 语义流形数学建模 — 系统流程图

> 基于 `text_baai_ema_flask_centering.py`（已实现）+ 后续数学建模（规划中）

```mermaid
graph TB
    %% ====== Phase 0: 输入 ======
    INPUT["📄 文本输入<br/>（.txt / .epub 等）"]
    SPLIT["✂️ 分句<br/>split_text_into_segments()"]
    INPUT --> SPLIT

    %% ====== Phase 1: 嵌入与预处理 ======
    subgraph PHASE1["🔵 Phase 1: 嵌入与预处理（已实现）"]
        EMBED["🧠 BGE-M3 Embedding<br/>MLX / SentenceTransformer<br/>→ (n, 1024) 高维向量"]
        CACHE{"📦 缓存检测<br/>embeddings.npy ?"}
        SPLIT --> CACHE
        CACHE -->|"命中"| LOAD["⚡ 直接加载"]
        CACHE -->|"未命中"| EMBED
        EMBED --> SAVE["💾 写入缓存"]
        LOAD --> RAW["原始连贯性计算<br/>cosine_similarity(i, i+1)"]
        SAVE --> RAW
    end

    %% ====== Phase 2: 时序平滑与中心化 ======
    subgraph PHASE2["🟢 Phase 2: 时序平滑与中心化（已实现）"]
        EMA["📊 EMA 滑动窗口平滑<br/>run_ema()<br/>权重: w(j)=alpha^abs(j)<br/>向量化 unfold + 批量归一化"]
        CENTER["🎯 策略B：零均值中心化<br/>X ← X − mean(X)<br/>消除各向异性偏置"]
        COH_AFTER["平滑后连贯性计算"]
        RAW --> EMA
        EMA --> CENTER
        CENTER --> COH_AFTER
    end

    %% ====== Phase 3: 降维与聚类 ======
    subgraph PHASE3["🟡 Phase 3: 降维与聚类（已实现）"]
        PHATE["🌐 PHATE 降维<br/>n_components=3<br/>knn=5, cosine<br/>→ 3D 语义流形"]
        CLUSTER["🔗 聚类<br/>Leiden (优先) / HDBSCAN<br/>→ 主题簇标签"]
        PELT["📏 PELT 变点检测<br/>基于 EMA 前原始距离<br/>→ 语义分段边界"]
        COH_AFTER --> PHATE
        COH_AFTER --> CLUSTER
        CLUSTER --> PELT
        PHATE --> MERGE["🧩 分段 + 簇合并<br/>相邻同簇段融合"]
        PELT --> MERGE
    end

    %% ====== Phase 4: 应用层 ======
    subgraph PHASE4["🔴 Phase 4: 应用层（已实现）"]
        SEGS["📋 语义分段<br/>(id, 区间, 主题簇, 文本)"]
        DEEPSEEK["🤖 DeepSeek API<br/>流式分段摘要"]
        PLOTLY["📈 Plotly 可视化<br/>• 连贯性对比图<br/>• 簇分布柱状图<br/>• PHATE 3D 交互图<br/>• 分段详情表"]
        HTML["🌍 导出 HTML<br/>自包含交互式 3D 图"]
        MERGE --> SEGS
        SEGS --> DEEPSEEK
        MERGE --> PLOTLY
        PLOTLY --> HTML
    end

    %% ====== Phase 5: 数学建模 ======
    subgraph PHASE5["🟣 Phase 5: 语义流形数学建模（规划中）"]
        subgraph GEOMETRY["📐 流形几何分析"]
            DIM["本征维数估计<br/>Intrinsic Dimension<br/>(MLE / TLE / TwoNN)"]
            CURV["离散里奇曲率<br/>Ollivier-Ricci Curvature<br/>沿语义轨迹"]
            GEOD["测地线距离谱<br/>Geodesic Distance Spectrum"]
            TORSION["流形挠率分析<br/>Manifold Torsion"]
        end

        subgraph SPECTRAL["🌈 谱分析"]
            DIFFUSE["扩散距离矩阵<br/>Diffusion Distance<br/>(PHATE 中间产物)"]
            ENTROPY["冯·诺依曼熵<br/>von Neumann Entropy<br/>语义复杂度度量"]
            EIGEN["特征谱衰减<br/>Eigenspectrum Decay<br/>→ 有效维度数"]
            LAPLACE["图拉普拉斯谱<br/>Graph Laplacian<br/>连通性分析"]
        end

        subgraph DYNAMICS["⚡ 动力学分析"]
            VELOCITY["语义速度场<br/>v(t) = norm(x(t+1)-x(t))"]
            ACCEL["语义加速度<br/>a(t) = v(t+1) - v(t)<br/>→ 逻辑跳跃检测"]
            PHASE_TRANS["相变检测<br/>速度/曲率突变点<br/>(与 PELT 交叉验证)"]
            INERTIA["语义惯性张量<br/>局部协方差结构"]
        end

        subgraph TOPO["🔮 拓扑分析"]
            PERSIST["持续性同调<br/>Persistent Homology<br/>H₀ / H₁ 特征"]
            BETTI["Betti 数演化<br/>沿文本展开"]
            MAPPER["Mapper 图<br/>拓扑骨架提取"]
        end

        subgraph INFO["📊 信息几何"]
            FISHER["Fisher-Rao 度量<br/>嵌入空间的自然梯度"]
            ANISO["各向异性指数<br/>Anisotropy Index<br/>(centering 前后对比)"]
            KL_DIV["KL 散度流<br/>相邻句分布偏移"]
        end

        PHATE --> GEOMETRY
        PHATE --> SPECTRAL
        PHATE --> DYNAMICS
        PHATE --> TOPO
        CENTER --> ANISO
    end

    %% ====== Phase 6: 综合报告 ======
    subgraph PHASE6["🟠 Phase 6: 综合报告（规划中）"]
        REPORT["📑 数学建模报告<br/>• 流形结构诊断<br/>• 语义转折点交叉验证<br/>• 各向异性消除效果量化<br/>• 与 PELT/聚类的对比<br/>• 可视化大图"]
    end

    PHASE5 --> REPORT
    PHASE4 --> REPORT

    %% 样式 (如遇兼容问题可删除本段)
    classDef c0 fill:#e8f5e9,stroke:#2e7d32
    classDef c1 fill:#e3f2fd,stroke:#1565c0
    classDef c2 fill:#fff3e0,stroke:#ef6c00
    classDef c3 fill:#fce4ec,stroke:#c62828
    classDef c4 fill:#f3e5f5,stroke:#6a1b9a
    classDef c5 fill:#fff8e1,stroke:#f9a825
    class INPUT c0
    class PHASE1 c1
    class PHASE2 c0
    class PHASE3 c2
    class PHASE4 c3
    class PHASE5 c4
    class PHASE6 c5
```

## 管线说明

### 已实现 (Phase 1–4)

| 阶段 | 核心算法 | 数学原理 |
|------|----------|----------|
| **嵌入** | BGE-M3 (1024-d) | Transformer 最后一层 hidden state |
| **平滑** | EMA + Centering | 指数衰减加权滑动平均 + 零均值去偏 |
| **降维** | PHATE (n_components=3) | 扩散势能距离 + MDS |
| **聚类** | Leiden / HDBSCAN | 模块度优化 / 密度聚类 |
| **分段** | PELT (rbf/l2) | 基于原始 embedding 欧氏距离的变点检测 |

### 规划中 (Phase 5–6)

| 分析维度 | 关键方法 | 解决的问题 |
|----------|----------|------------|
| **本征维数** | MLE / TwoNN | 语义流形的真实自由度 |
| **离散曲率** | Ollivier-Ricci | 文本逻辑转折的几何度量 |
| **冯·诺依曼熵** | 扩散算子的谱熵 | 语义复杂度 / 信息密度 |
| **速度/加速度场** | PHATE 坐标差分 | 叙事节奏与逻辑跳跃的定量描述 |
| **持续性同调** | Ripser / GUDHI | 流形拓扑空洞 (H₁) 检测 |
| **各向异性指数** | Centering 前后对比 | 量化全局偏置去除效果 |
| **Fisher-Rao 度量** | 嵌入空间自然梯度 | 捕捉语义流形的信息几何结构 |

### 关键交叉验证

```
PELT 变点 ←→ 相变检测 (速度/曲率突变)
PELT 变点 ←→ 离散曲率峰值
聚类标签 ←→ 持续性同调 H₀ 连通分量
各向异性指数 ←→ Centering 前后的谱熵变化
轨迹弯曲度 ←→ 测地线/欧氏距离比
```
