# GridS：可微分网格采样剪枝，让 VLA 用 16 个视觉 Token 看见关键

> **论文**：*See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model*
>
> **作者**：Yixu Feng, Zinan Zhao, Yanxiang Ma, Chenghao Xia, Chengbin Du, Yunke Wang, Chang Xu
>
> **机构**：The University of Sydney、City University of Hong Kong、StellarEdge Robotics
>
> **发布时间**：2026 年 5 月
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2605.11817) | [项目主页](https://fediory.github.io/Grid-Sampler/) | [ICML Poster](https://icml.cc/virtual/2026/poster/65010)
>
> **发表会议**：ICML 2026

---

## 一句话总结

将 VLA Token 剪枝从「离散丢弃固定网格上的 patch」重构为「几何感知的连续重采样」：用全局池化 + 轻量 MLP 预测 $K$ 个 $[0,1]^2$ 浮点坐标，通过**可微双线性插值**从密集特征图中提取亚像素级 Token，端到端与策略联合优化，π₀ 在 LIBERO 上仅用 **16/256 (6.25%)** 个视觉 Token 即可削减 76% FLOPs 同时平均成功率反升 +1.6%，**极限 $K=1$ 时 π₀.₅ 仍保持 96.6%**，SmolVLA 真实世界 OOD 准确率提升 +28.6%。

---

## 一、问题与动机

### 1.1 VLA 视觉 Token 冗余的"剪枝两难"

标准 VLA 用 ViT（SigLIP / DINOv2）把图像编码为 $H \times W = 16 \times 16 = 256$ 个稠密 Token，Transformer 注意力复杂度 $\mathcal{O}(N^2)$ 让这成为推理瓶颈。现有 VLA Token 剪枝陷入根本性 trade-off：

- **激进压缩**（如保留 < 25% Token）会丢弃关键几何细节（接触点、夹爪边缘），成功率严重下降；
- **保守压缩**虽能维持精度但加速效果有限，达不到实时部署要求。

### 1.2 现有方法的共同缺陷：固定网格的离散选择

| 方法路线 | 代表工作 | 缺陷 |
| --- | --- | --- |
| 语义剪枝 | [SparseVLM](https://arxiv.org/abs/2410.04417)、[FastV](https://arxiv.org/abs/2403.06764) | 偏向显著物体本体，忽略语义微弱但几何关键的边缘/接触点 |
| 动态剪枝 | [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025)、[VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) | Training-free 启发式策略，**不可微**，无法与下游任务对齐 |
| 异步执行 | [SmolVLA](https://arxiv.org/abs/2506.01844)、[RTC](/papers/06-embodied-ai/vla/efficient/RTC_2025) | 引入观测延迟（state staleness）→ 动态场景失稳 |

**更深层次的问题**：所有上述方法都被限制在 ViT 输出的**离散 patch 网格**上选择 Token。然而机器人交互通常要求**亚 patch 精度**——例如抓取点恰好落在两个 patch 之间，强制选择整 patch 会带来**量化误差**（quantization error），损失精细空间保真度（如 Fig 2(a)）。

### 1.3 解决思路：把剪枝当作连续重采样而非离散选择

GridS 的核心洞察：**剪枝不应该是从固定网格里挑选，而应该是在连续特征空间上预测任务驱动的浮点坐标然后做插值采样**。这样既能突破网格量化限制（亚 patch 精度），又能让坐标预测网络通过可微插值接收下游任务的梯度信号实现端到端优化。

---

## 二、预备知识

### 2.1 标准 VLA 视觉编码流水线

输入图像 $I \in \mathbb{R}^{3 \times H_R \times W_R}$ 经冻结视觉编码器（如 DINOv2）patch 化后得到稠密特征：

$$T_{\text{dense}} \in \mathbb{R}^{H \times W \times C}, \quad H \times W = 256$$

该 Token 序列直接送入下游 Transformer 与语言/动作 Token 拼接，注意力计算量为 $\mathcal{O}((N_v + N_l + N_a)^2)$。

### 2.2 PyTorch 的 `grid_sample` 算子

PyTorch 内置的 `F.grid_sample(input, grid)` 支持双线性插值采样：给定连续浮点坐标 grid，从特征图四个最近邻整数坐标加权获得插值特征。该操作对**采样坐标可微**，是 STN、Deformable Convolution、DAT 等工作的基础。GridS 本质上把这种连续采样机制首次系统引入到 VLA Token 剪枝中。

### 2.3 Deformable 系列与隐式表征的启发

- **Deformable Convolution**、**DAT**：用学习的偏移让注意力查询离开固定网格中心；
- **LIIF**、**NeRF**：将图像/场景建模为连续函数，允许任意亚像素查询。

GridS 借鉴这些思路但把"连续查询"用作 Token 剪枝目的，而非密集预测或新视角合成。

---

## 三、核心方法

GridS 在视觉编码器之后、下游 Transformer 之前插入一个轻量模块，将 $H \times W \times C$ 的稠密特征压缩为 $K \times C$ 的稀疏 Token（$K \ll H \times W$，论文实验取 $K \in \{1, 4, 16\}$）。整个流程分两个阶段（如 Fig 3(b)）。

### 3.1 阶段一：全局坐标预测

#### 3.1.1 全局上下文聚合

先用 Global Average Pooling 把整张特征图压成一个全局上下文向量 $z \in \mathbb{R}^{C}$：

$$z = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} T_{\text{dense}}^{(i,j)}$$

> **为什么用 GAP 而不是 Conv 下采样**：消融实验显示用参数化的 Conv 下采样会引入对训练集空间偏置的过拟合，反而比无参数的 GAP 低 4.4 个点（表 12）。

#### 3.1.2 轻量 MLP 预测连续坐标

紧接着用一个**轻量** MLP 把全局上下文映射到 $K$ 个二维连续坐标：

$$P = \sigma\bigl(\text{MLP}(z)\bigr) \in [0, 1]^{K \times 2}$$

其中 $\sigma$ 是 Sigmoid，确保坐标位于图像归一化范围内。

> **关键设计**：网络输出**不是**离散网格索引而是**连续浮点坐标**——这是与所有现有剪枝方法的根本分界点。

消融实验中放大 MLP 容量 10× / 100× 仅带来 +0.2% 改进，说明"全局上下文 → $K$ 个坐标"本身是低复杂度任务；用预训练 SAM 引导坐标反而下降 6.2%——**通用语义分割**（"what is the object"）与**任务驱动几何采样**（"where should the gripper interact"）不对齐。

### 3.2 阶段二：可微双线性采样 + 几何注入

#### 3.2.1 双线性插值机制

设预测坐标 $P(x, y)$ 一般不在整数网格上。取四个最近邻整数坐标 $P_1 = (\lfloor x \rfloor, \lfloor y \rfloor)$, $P_2, P_3, P_4$，定义偏移 $\Delta_x = x - \lfloor x \rfloor$, $\Delta_y = y - \lfloor y \rfloor$，四点权重：

$$\omega_1 = (1 - \Delta_x)(1 - \Delta_y), \quad \omega_2 = \Delta_x (1 - \Delta_y)$$

$$\omega_3 = (1 - \Delta_x) \Delta_y, \quad \omega_4 = \Delta_x \Delta_y$$

采样特征：

$$F_{\text{sampled}}(x, y) = \sum_{k=1}^{4} \omega_k \cdot P_k$$

> **用大白话说**：如果一个抓取点恰好落在两个 patch 的边界上，GridS 可以通过对四邻居打平衡权重精确采样该边界点；而离散剪枝只能被迫挑选其中一边，必然损失保真度。

#### 3.2.2 梯度回传：让任务信号驱动坐标移动

由于权重 $\omega_k$ 是 $(x, y)$ 的线性函数，任务损失 $\mathcal{L}$ 可以通过链式法则回传到坐标：

$$\frac{\partial \mathcal{L}}{\partial x} = \sum_{k=1}^{4} \frac{\partial \mathcal{L}}{\partial F_{\text{sampled}}} \cdot P_k \cdot \frac{\partial \omega_k}{\partial x}$$

这意味着 MLP 坐标预测器**不是启发式**，而是由动作预测误差**主动训练**——会自动将采样点移到能最小化任务损失的几何位置。

#### 3.2.3 几何注入：恢复空间结构

稀疏采样会破坏原始网格的 2D 空间结构。GridS 用一个 Coordinate Encoder 把预测坐标 $P$ 映射成位置嵌入 $E_{\text{pos}} \in \mathbb{R}^{K \times C}$，叠加到采样特征上得到最终视觉稀疏 Token：

$$T_{\text{spa}} = F_{\text{spa}} + E_{\text{pos}}$$

消融实验：**移除几何注入掉 3.6%**，把双线性换成最近邻插值掉 4.1%（梯度链路被四舍五入破坏）。

### 3.3 联合优化：从"训练后压缩"到"训练时蒸馏"

> 这是 GridS 与 EfficientVLA / VLA-Cache / SparseVLM 等 Training-free 方法的方法论分界。

GridS 直接嵌在视觉编码器与下游 Transformer 之间，**与 VLA 策略一起在标准微调流程中端到端训练**，只用主任务损失，不需要辅助监督或 ground-truth 注意力图，兼容 auto-regressive (SmolVLA) 与 flow-matching (π₀ / π₀.₅) 两类范式。

由于 VLA 微调本来就是必经步骤，GridS 不增加额外训练阶段；而压缩本身把视觉 Token 在前向极早阶段砍掉，反而**大幅加速训练**（表 4：π₀ 训练步速 3.4×，π₀.₅ 2.9×）。

---

## 四、实验结果

### 4.1 LIBERO 基准

实验在 π₀（256 Token 基线）与 π₀.₅ 上对比 FastV / SparseVLM / VLA-Cache 等 Training-free 剪枝。

| 模型 | Vis. Token | FLOPs(G) | Time(s) | Spatial | Object | Goal | Long | Avg. | ΔSR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π₀ 基线 | 256 | 216.0 | 8.17 | 97.2 | 98.8 | 96.0 | 85.6 | 94.4 | — |
| π₀ + FastV† | 100 | 143.6 | 7.32 | 97.0 | 98.4 | 93.8 | 82.4 | 92.9 | −1.5 |
| π₀ + SparseVLM† | 100 | 150.3 | 7.48 | 93.4 | 98.0 | 91.2 | 76.6 | 89.8 | **−4.6** |
| π₀ + VLA-Cache† | 256 | 188.1 | 7.52 | 95.2 | 97.6 | 96.6 | 85.6 | 93.8 | −0.6 |
| **π₀ + GridS₁₆** | **16** | **51.6** | **6.05** | 98.0 | 99.2 | 96.4 | 90.2 | **96.0** | **+1.6** |
| **π₀ + GridS₄** | **4** | **43.6** | **5.86** | 96.6 | 99.4 | 96.4 | 89.6 | 95.5 | +1.1 |
| π₀.₅ 基线 | 256 | 249.8 | 8.54 | 98.4 | 98.0 | 97.6 | 92.8 | 96.7 | — |
| π₀.₅ + GridS₁₆ | 16 | 83.8 | 6.76 | 98.6 | 98.8 | 98.4 | 95.2 | **97.7** | +1.0 |

†标记为 Training-free post-hoc 剪枝。

**关键观察**：

1. **离散剪枝（FastV / SparseVLM）破坏空间连续性**，分别下降 1.5 / 4.6 pp；
2. **GridS 用 6.25% Token (16/256) 削减 76% FLOPs 并反超基线 +1.6 pp**；
3. **最大增益出现在 Long-horizon（+4.6 pp）**：长程任务最容易因视觉干扰累积误差，GridS 的连续采样有效过滤干扰；
4. **K=4 仅用 4 个 Token 仍能保持 +1.1 pp**——表明 LIBERO 上动作执行所需的视觉信息极度稀疏。

### 4.2 ALOHA 双臂操作

| 方法 | Token | Time(s) | Env Reward↑ | Avg.↑ | Transfer Cube (Scripted/Human) | Insertion (Scripted/Human) |
| --- | --- | --- | --- | --- | --- | --- |
| π₀ 基线 | 256 | 7.04 | 2.44 | 86.3 | 100.0 / 96.9 | 91.4 / 56.7 |
| π₀ + GridS | 16 | 6.32 | 2.38 | **87.0** | 100.0 / 96.9 | 86.9 / **64.2** |

Insertion-Human 任务上 **+7.5 pp**——人类演示自带噪声变化，GridS 的稀疏化反而提供鲁棒性。

### 4.3 真实世界 SO100 机器人（SmolVLA 骨架）

三个真实任务：Pick & Place、Transfer Pen、Stack Cubes，外加 21 种 OOD 场景（杂乱背景、未见物体、未训练空间布局等）。

**Stack Cubes 任务上 GridS₁₆ vs SmolVLA 基线**：

- 标准场景成功率：**61.9% vs 9.5%**（+52.4 pp）
- OOD 场景成功率：**38.1% vs 0%**（+38.1 pp）
- 单步执行时间：减少 3.3 s

**OOD 上整体提升 +28.6 pp**（Fig 1c）——视频对比显示基线像"机械重放训练轨迹"，遇到位置变动直接失败；GridS 则真正"在看物体"，能动态调整轨迹锁定目标。

### 4.4 关键消融

#### 4.4.1 Token 预算 $K$ 的倒 U 形

真实世界 Stack Cubes 上：

| $K$ | Success | OOD Success |
| --- | --- | --- |
| 4 | 0.0% | 0.0% |
| 8 | 28.5% | 19.1% |
| **16** | **61.9%** | **38.1%** |
| 32 | 19.0% | 0.0% |
| 64 (基线) | 9.5% | 0.0% |

- $K$ 太小：几何信息瓶颈；
- $K$ 太大：背景噪声又回流，导致夹爪释放时机判断失败；
- **$K = 16$ 是"信号-噪声比"最优点**——这一现象与 LIBERO 上的高准确率共同支持"信息瓶颈"假说。

#### 4.4.2 极限压缩 $K = 1$

| 模型 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- |
| π₀.₅ 基线 (256 Token) | 98.4 | 98.0 | 97.6 | 92.8 | 96.7 |
| π₀.₅ + GridS ($K=1$) | 98.0 | 99.0 | 97.6 | 91.8 | **96.6** |

**用 1 个 Token 取得 99.6% 压缩率仍保持基线性能**——这是 VLA 视觉剪枝中迄今最极端的结果，强烈暗示当前 VLA 在 LIBERO 上根本不需要稠密视觉表征，而是被迫学到了"空间捷径"。

#### 4.4.3 与其他下采样方式对比

固定 $K = 16$、完整微调：

| 方法 | Avg. SR |
| --- | --- |
| 随机采样 | 87.8 |
| Top-K（按显著度） | 90.5 |
| GridS（双线性） | **96.0** |

随机采样长程任务暴跌至 73.2%，Top-K 仍落后 GridS 5.5 pp——**GridS 的增益不来自"减少 Token 防过拟合"，而来自可微几何采样**。

### 4.5 LIBERO-Plus & RoboTwin 鲁棒性扩展（Appendix E）

- **LIBERO-Plus 零样本 OOD**：基线在 Camera Viewpoints 严重崩溃（67.0%），GridS₃₂ 提升 **+19.4 pp**（86.4%）；Light Conditions 提升 **+12.1 pp**；Sensor Noise +5.6 pp。
- **RoboTwin 2.0 "Place Bread Skillet"**：XVLA 基线 12%，XVLA + GridS **76%**。

### 4.6 计算效率

- **训练加速**：batch 128 下 π₀ 训练步速 **3.4×**，π₀.₅ **2.9×**；
- **推理 FLOPs 削减**：4.2× (π₀) / 3.0× (π₀.₅)；
- **单 batch wall-clock 加速**有限（~1.2×）—— JAX 编译后 dense 基线的内核开销已经成为新瓶颈，需要大 batch 才能完全释放算力红利。

---

## 五、局限性与未来方向

1. **单 batch wall-clock 加速有限**：JAX 编译过的稠密基线主要受 kernel overhead 限制，GridS 的 FLOPs 红利在 batch=1 推理时只能转化为 ~1.2× 加速；
2. **$K$ 静态固定**：消融显示倒 U 形最优点存在但目前是手工选择，无法根据场景复杂度动态调整 Token 预算；
3. **与 PEFT 兼容性差**：仅用 LoRA 微调时 GridS 比稠密基线落后 8.3 pp——稀疏连续采样改变了视觉 Token 分布，冻结的稠密预训练注意力无法借少量低秩适配重新对齐；
4. **失败模式**：Robot Initial States 严重扰动时性能下降（-8 ~ -14 pp），因为复杂恢复轨迹需要完整运动学链可视化，32 Token 把本体感知边界过度平滑。

---

## 六、个人思考

### 6.1 与项目内其他 VLA 高效推理工作的对比

| 维度 | GridS | EfficientVLA | VLA-Cache | VLA-Pruner | SparseVLM |
| --- | --- | --- | --- | --- | --- |
| 训练时机 | 联合训练 | Training-free | Training-free | Training-free | Training-free |
| 选择空间 | **连续** $[0,1]^2$ | 离散 patch | 离散 patch | 离散 patch | 离散 patch |
| Token 预算 | 4 ~ 32 | 56 ~ 112 | 256（缓存复用） | 32 | 100 |
| 对应冗余 | 空间 + 几何量化 | 三维（深度 / 空间 / 时序） | 时间（跨帧） | 双层语义/动作 | 视觉显著性 |
| LIBERO Avg | **96.0**（K=16） | — | 93.8 | — | 89.8 |
| 极限 K | **1** | 56 | — | 32 | — |

GridS 的独特价值在于**首次把可微连续采样引入 VLA 剪枝**，借此突破"固定网格"的根本性 trade-off。EfficientVLA / VLA-Cache 走的是"工程优化"路线（在 Training-free 约束下尽量保性能），GridS 则承认必须重训但通过端到端优化获得更高上限。

### 6.2 "Learning is Forgetting" 视角下的反直觉提升

GridS 在多个设定下**剪枝后反而涨点**：

- LIBERO Long-horizon +4.6 pp
- 真实世界 Stack Cubes +52.4 pp
- OOD 平均 +28.6 pp

论文用 Information Retention Map 给出解释：成功执行时平均检索率维持在 **0.8 ~ 0.9** 而非 1.0，正好对应丢弃 10%-20% 的背景噪声。这呼应了 Conklin 等人的 *"Learning is Forgetting"*（ICLR 2026）观点——**完整信息保留反而让策略学到伪相关 / 空间捷径**，适度的信息瓶颈强迫模型只学因果性的物理几何。

这与 [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) PickCan 任务上"剪 36% 参数涨点"以及 [VLA-Pruner](/papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025) "50% 剪枝率反超原模型"形成系列证据——**当前 VLA 在标准基准上严重过参数化 / 过 Token 化**。

### 6.3 K=1 现象的深层含义

单 Token 保持 96.6% 准确率几乎可以视作 LIBERO 评测体系的一个**警钟**：

- 这暗示 LIBERO 任务的视觉判别空间被"任务序列 + 物体位置"的有限组合主导，根本不需要 256 个 Token 的稠密表征；
- 也意味着我们当前用 LIBERO 评估 VLA 视觉能力时极易产生**虚假饱和**——必须配合 LIBERO-Plus、RoboTwin、真实 OOD 才能反映真实视觉理解能力；
- 这与 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) 论文揭示的「standard LIBERO 已饱和」结论一致。

### 6.4 与连续/可变形机制的方法谱系

GridS 在方法谱系上承袭：
- **Spatial Transformer Networks (2015)**：用可微采样让网络学会几何变换；
- **Deformable Conv / DETR**：让 query 离开固定网格中心；
- **Detr3D / Sparse4D**：3D 检测中已经用 reference points + bilinear sampling；
- **NeRF / LIIF**：把图像视作连续函数允许任意亚像素查询。

但**首次**把这些机制系统化引入 VLA Token 剪枝目的——把"采样点位置"作为可学习的任务参数而非辅助 query。后续工作可能进一步把 GridS 与稀疏 attention（如 [Linear Attention](https://arxiv.org/abs/2006.16236)）或 KV 缓存（[VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025)）组合，形成「空间 + 时间」双维稀疏化。

### 6.5 局限与展望

最显著的瓶颈是**与 PEFT 不兼容**（LoRA 微调差 8.3 pp）——这制约了它在大型 VLA 上的廉价部署，因为 70B 级 VLA 难以全参数微调。一个可行方向：把 GridS 视为 "Vision-LM 接口适配层"，在冻结骨干两侧训练小型适配器恢复对稀疏 Token 的响应能力；或者把 $K$ 做成 token-dependent 的动态预算，配合分组 LoRA 路由。

另一个直觉是：**$K = 1$ 现象本质是「全局上下文 + 单一关键定位点」就足以驱动操作**——这与 RoboPoint / SpatialVLA 的"affordance point"思路异曲同工。GridS 实质提供了一个端到端学习 affordance point 的轻量框架，未来或可拓展到多臂、移动操作等需要多 affordance 的设定。

---

## 参考

- [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024) — Flow Matching VLA，GridS 的主要实验骨架
- [π₀.₅](/papers/06-embodied-ai/vla/foundation/pi05_2025) — 开放世界 VLA，GridS 在此架构上验证极限压缩
- [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) — 三维度结构化 Training-free 加速，与 GridS 形成方法论对照
- [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) — 跨帧时间冗余利用，与 GridS 空间稀疏化正交
- [VLA-Pruner](/papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025) — 双层语义-动作剪枝，同样观察到"剪枝涨点"
- [SmolVLA](https://arxiv.org/abs/2506.01844) — 论文真实机器人实验的 baseline 骨架
- [FastV](https://arxiv.org/abs/2403.06764) — VLM 视觉 Token 剪枝代表方法
- [SparseVLM](https://arxiv.org/abs/2410.04417) — Training-free 跨模态注意力剪枝
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626) — 揭示 VLA 在原始 LIBERO 上过度饱和的扩展基准
