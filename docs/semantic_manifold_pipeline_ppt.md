# 语义流形分析管线 — PPT 用图

> 精简版：节点标签 ≤ 8 字，结构拍平，适合 16:9 幻灯片。
> 在 VS Code 中 `Cmd+Shift+V` 预览 → 右键图表 → **Copy as PNG** → 粘贴到 PPT。

## 总览图（单页）

```mermaid
graph TB
    A["📄 文本输入"] --> B["🧠 BGE-M3 嵌入"]
    B --> C["📊 EMA 平滑"]
    C --> D["🎯 中心化去偏"]
    D --> E["🌐 PHATE 降维 3D"]
    D --> F["🔗 聚类"]
    F --> G["📏 PELT 分段"]
    E --> H["📋 语义分段输出"]
    G --> H
    H --> I["🤖 DeepSeek 摘要"]
    H --> J["📈 可视化报告"]

    style A fill:#e8f5e9,stroke:#2e7d32
    style B fill:#e3f2fd,stroke:#1565c0
    style C fill:#fff3e0,stroke:#ef6c00
    style D fill:#fce4ec,stroke:#c62828
    style E fill:#f3e5f5,stroke:#6a1b9a
    style F fill:#f3e5f5,stroke:#6a1b9a
    style G fill:#f3e5f5,stroke:#6a1b9a
    style H fill:#e8f5e9,stroke:#2e7d32
    style I fill:#fff8e1,stroke:#f9a825
    style J fill:#fff8e1,stroke:#f9a825
```

## 数学建模扩展（第二页）

```mermaid
graph LR
    PHATE["🌐 PHATE 3D 语义流形"] --> GEO["📐 几何分析"]
    PHATE --> SPEC["🌈 谱分析"]
    PHATE --> DYN["⚡ 动力学"]
    PHATE --> TOPO["🔮 拓扑分析"]
    CENTER["🎯 中心化"] --> INFO["📊 信息几何"]

    GEO --> R1["本征维数<br/>离散曲率<br/>测地线距离"]
    SPEC --> R2["扩散距离<br/>冯诺依曼熵<br/>谱衰减"]
    DYN --> R3["速度场<br/>加速度<br/>相变检测"]
    TOPO --> R4["持续同调<br/>Betti数<br/>Mapper图"]
    INFO --> R5["Fisher度量<br/>各向异性<br/>KL散度"]

    R1 --> REPORT["📑 综合报告"]
    R2 --> REPORT
    R3 --> REPORT
    R4 --> REPORT
    R5 --> REPORT

    style PHATE fill:#f3e5f5,stroke:#6a1b9a
    style CENTER fill:#fce4ec,stroke:#c62828
    style REPORT fill:#fff8e1,stroke:#f9a825
```

## 技术栈一览（第三页）

```mermaid
graph TB
    subgraph 嵌入层
        BGE["BGE-M3<br/>1024维"]
        MLX["MLX / ST<br/>推理加速"]
    end

    subgraph 平滑层
        EMA["滑动窗口<br/>指数衰减"]
        CTR["零均值<br/>中心化"]
    end

    subgraph 降维层
        PH["PHATE<br/>势能距离"]
        MDS["SMACOF<br/>MDS"]
    end

    subgraph 聚类层
        LEI["Leiden<br/>模块度优化"]
        HDB["HDBSCAN<br/>密度聚类"]
    end

    subgraph 分段层
        PELT["PELT<br/>变点检测"]
        MRG["簇合并<br/>边界融合"]
    end

    subgraph 应用层
        DS["DeepSeek<br/>流式摘要"]
        PLT["Plotly<br/>3D可视化"]
    end

    BGE --> EMA --> CTR --> PH --> LEI --> PELT --> MRG --> DS
    CTR --> HDB --> PELT
    PH --> MDS --> PLT

    style 嵌入层 fill:#e3f2fd,stroke:#1565c0
    style 平滑层 fill:#fff3e0,stroke:#ef6c00
    style 降维层 fill:#f3e5f5,stroke:#6a1b9a
    style 聚类层 fill:#e8f5e9,stroke:#2e7d32
    style 分段层 fill:#fce4ec,stroke:#c62828
    style 应用层 fill:#fff8e1,stroke:#f9a825
```

---

## 导出到 PPT 的方法

### 方法 1：VS Code 直接复制（推荐）
1. `Cmd+Shift+V` 打开 Markdown 预览
2. 右键 Mermaid 图表 → **Copy as PNG**（或 Copy as SVG）
3. 在 PPT 中 `Cmd+V` 粘贴

### 方法 2：在线导出（备选）
1. 打开 https://mermaid.live
2. 粘贴 Mermaid 代码
3. 点击右上角下载按钮 → PNG / SVG

### 方法 3：命令行导出（最清晰）
```bash
# 需要先装 mermaid-cli（可选）
npm install -g @mermaid-js/mermaid-cli
mmdc -i input.mmd -o output.png -w 1920 -H 1080
```
