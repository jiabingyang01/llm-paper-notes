# HeiSD：运动学感知的混合推测解码加速 VLA 推理

> **论文**：*HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awareness*
>
> **作者**：Zihao Zheng, Zhihao Mao, Sicheng Tian, Jiayu Chen, Maoliang Li, Xinhao Sun, Zhaobo Zhang, Xuanzhe Liu, Donggang Cao, Hong Mei, Xiang Chen†
>
> **机构**：Peking University, China University of Geosciences, Beijing Normal University
>
> **发布时间**：2026年03月
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.17573)
>
> **分类标签**：`VLA` `推测解码` `混合 SD` `运动学感知` `检索增强` `推理加速` `OpenVLA` `LIBERO`

---

## 一句话总结

提出 **HeiSD** 框架，首次将 **Drafter-based SD** 和 **Retrieval-based SD** 混合用于 VLA 推理加速：通过自适应验证跳过 + 序列级宽松接受优化检索 SD 质量，再用**运动学融合指标**（曲率半径 + 累积位移）自动判定混合边界，LIBERO 上最高 **2.45×** 加速，真实世界 **2.06×-2.41×** 加速，成功率几乎无损。

---

## 一、问题与动机

### 1.1 VLA 推理速度瓶颈

自回归 VLA（如 OpenVLA）每步需解码 7 个动作 token（X, Y, Z, $r_X$, $r_Y$, $r_Z$, G），每个 token 都需一次完整的 LLM 前向传播。OpenVLA 单步推理约 174ms（ViT 8ms + LLM 解码 113ms + 系统开销 53ms），无法满足实时控制需求。

### 1.2 两类推测解码各有缺陷

**Speculative Decoding (SD)** 是一种通用 LLM 加速方法：低成本生成 draft token 序列，再由大模型并行验证。应用于 VLA 时有两种形式：

| 类型 | 原理 | 优势 | 劣势 |
| --- | --- | --- | --- |
| **Drafter-based SD** | 小模型生成 draft | 高质量 draft，长接受序列 | 需维护 draft 模型，有额外计算开销 |
| **Retrieval-based SD** | 从向量数据库检索历史动作 | 无 drafter 开销，理论加速更高 | Draft 质量低，分布不匹配，难通过验证 |

**现有方法（如 SpecVLA）只用单一类型 SD**，无法同时利用两者优势。

### 1.3 关键观察：轨迹重叠规律

作者构建向量数据库，对比检索轨迹与 VLA 推理轨迹：
- **成功案例**：大量轨迹段**高度重叠**（绿色区域），少量偏差（红色区域）
- **失败案例**：重叠极少，端点不匹配

**核心洞察**：应该对重叠轨迹段用检索 SD（速度快），对非重叠段用 drafter SD（质量高），实现混合 SD。

### 1.4 混合 SD 的两大挑战

1. **检索 SD 需要优化**：分布不匹配导致 draft 几乎无法通过严格验证（接受长度仅 0.81-1.03）；且无 drafter 自纠错能力，同一错误被反复检索
2. **如何确定混合边界**：推理时轨迹是逐步生成的，无法预知完整轨迹来判断哪些步用检索、哪些用 drafter

---

## 二、预备知识

### 2.1 VLA 动作生成

VLA 模型自回归预测每个动作维度 token：

$$a_j = \arg\max_{a_j} \left[ P(a_j \mid a_{0:j-1}, \mathbb{O}, \mathbb{P}, \mathbb{W}) \right]$$

其中 $\mathbb{O}$ 为视觉观测，$\mathbb{P}$ 为语言指令，$\mathbb{W}$ 为模型参数。每个动作切片是 7 维向量（位置 X, Y, Z + 关节旋转 $r_X$, $r_Y$, $r_Z$ + 夹爪 G）。

### 2.2 推测解码

**Drafter-based SD**：小模型 $M_D$ 生成 draft，大模型 $M_V$ 并行验证：

$$\text{Draft: } a_j = M_D(f_{1:t}, e_{0:t}, a_{t+1:j-1})$$

$$\text{Verify: } \hat{a}_j = M_V(a_{0:j-1}, \mathbb{P}, \mathbb{W}), \quad \begin{cases} \hat{a}_j = a_j, & \text{Accept} \\ \hat{a}_j \neq a_j, & \text{Discard} \end{cases}$$

**Retrieval-based SD**：从预构建的数据库 $\text{DB}$ 中检索 draft：

$$\text{Draft: } a_j = \{Retri(f_{1:t}, e_{0:t}, a_{t+1:j-1}) \mid \in \text{DB}\}$$

---

## 三、核心方法

### 3.1 向量数据库构建

基于 LIBERO 训练数据构建 Qdrant 向量数据库：
- **向量表征**：DINOv2（1024-dim）+ SigLIP（1152-dim）双编码器 × 双视角（第三人称 + 肘部相机）= **4352 维**联合 embedding，L2 归一化
- **Payload**：每个向量附带当前动作 + 3 步 lookahead 动作序列 + 元数据
- **分片策略**：按任务分片，40 个独立 collection，HNSW 索引
- **性能**：平均检索延迟 5.13ms（远低于 LLM 前向 13.93ms），273k 向量，6.5GB 存储

纯检索完成任务的成功率：LIBERO-Goal 62.0%（VLA 77.0%），速度 3.74×-4.83×。

### 3.2 自适应验证跳过机制 (Adaptive Verify-Skip)

**问题**：检索 draft 的分布与 VLA 推理输出不匹配，即使轨迹重叠也难通过严格验证。

**核心思想**：对于"足够准确"的检索 draft，跳过验证直接接受。

**方法**：

1. **离线阶段**：在数据库历史轨迹上，提取每个轨迹点在 VLA 最后一层（`lm_head`）的输入特征。计算轨迹点之间的特征相似度——距离越近的点相似度越高

2. **关键发现**：特征相似度与轨迹点距离强相关。高相似度的点理论上可以直接接受而无需验证

3. **在线阶段**：复用历史的最小可接受相似度和对应距离，自动判断哪些新任务轨迹点可以跳过验证。根据任务完成反馈信号（成功/失败）动态调整阈值

### 3.3 序列级宽松接受策略 (Sequence-Wise Relaxed Acceptance)

**问题**：检索 SD 无 drafter 自纠错能力，错误 draft 被反复检索导致**持续性错误**。

**解决方案**：

#### Top-K 多样化检索
将 Top-1 检索扩展为 **Top-K 匹配**，增加 draft 多样性。

#### 运动学语义分组
将 7 维动作 token 按运动学相关性分为 3 个**序列**：
- **位置序列**：X, Y, Z（位置相关）
- **角度序列**：$r_X$, $r_Y$, $r_Z$（旋转相关）
- **夹爪序列**：G（独立处理，因为夹爪状态对任务成功率至关重要）

#### 序列级树解码
继承 Eagle-2 的树解码框架，但以**序列**而非 token 为单位构建树。不同检索结果的序列可以跨组合连接（夹爪序列除外），最大化 draft 生成潜力。

#### 序列级宽松接受
深度优先验证每条链。计算每个 draft token 与 verify token 的 index 偏差 $bias_{a_j}$。当：
- 整体序列偏差 $bias_{seq} \leq 30$
- 单 token 偏差 $bias_{a_j} \leq 15$

则**强制接受整个序列**。允许 $bias_{a_j} > bias_{seq}$（单 token 可有较大偏差，只要序列整体偏差小），与现有 token 级宽松接受有本质区别。

**夹爪序列零容忍**：不允许任何偏差，因为夹爪状态对任务成功率至关重要。

### 3.4 运动学融合指标确定混合边界

**问题**：推理时如何自动判断当前步应用检索 SD 还是 drafter SD？

#### 运动学特征观察

轨迹重叠区域（绿色，适合检索 SD）：
- **曲率半径 $\mathcal{R}^{[w]}$ 大**：接近直线运动
- **累积位移 $\mathcal{D}^{[w]}$ 大**：快速移动

轨迹偏差区域（红色，适合 drafter SD）：
- 曲率半径小：弯曲轨迹
- 累积位移小：缓慢精细操作

#### 曲率半径计算

在滑动窗口 $w$ 内，先投影轨迹点到 2D 平面并计算几何中心：

$$(u_i^{[w]}, v_i^{[w]}) = Proj\left(P_{x,y,z}^i - \frac{1}{w}\sum_{i}^{w-1} P_{x,y,z}^i \mid \in \mathcal{T}\right)$$

迭代优化几何中心后计算半径：

$$\mathcal{R}^{[w]} = \frac{1}{w}\sum_{i}^{w-1} Euclid_{2\text{-dim}}\left((u_i^{[w]}, v_i^{[w]}); (\hat{u}_c, \hat{v}_c)\right)$$

#### 累积位移计算

$$\mathcal{D}^{[w]} = \sum_{i}^{w-1} Euclid_{3\text{-dim}}\left(P_{x,y,z}^i \mid \in \mathcal{T}); (P_{x,y,z}^{i+1} \mid \in \mathcal{T})\right)$$

不考虑位移方向（因为机器人可能往返或圆周运动）。

#### 融合指标

$$\mathcal{F}^{[w]} = \alpha \cdot Norm(\mathcal{R}_i^{[w]}) + (1-\alpha) \cdot Norm(\mathcal{D}_i^{[w]})$$

- $\mathcal{F}^{[w]}$ 大（快速直线运动）→ 检索 SD
- $\mathcal{F}^{[w]}$ 小（缓慢弯曲精细操作）→ Drafter SD
- 归一化使用 95 百分位裁剪的 min-max 归一化，每个 LIBERO 子集独立计算
- 默认 $\alpha = 0.5$，$w = 15$

### 3.5 系统实现

**CPU+GPU 异构部署**：
- 向量数据库部署在 **CPU 内存**（避免 GPU OOM，VLA 模型 16.6GB + 数据库 8.8GB 超出消费级 GPU 24GB）
- GPU→CPU 传输 embedding（4352-dim FP16，延迟 0.25ms）
- CPU→GPU 传回 draft（7-dim FP16，延迟可忽略）
- CPU 检索效率实际略优于 GPU（额外加速 1.04×-1.09×）

---

## 四、实验

### 4.1 LIBERO 仿真结果

| 环境 | 方法 | SR | 加速 | AL |
| --- | --- | --- | --- | --- |
| **LIBERO-Goal** | AR w/o SD | 77.0% | 1.00× | – |
| | Pure R-SD | 77.0% | 0.96× | 1.03 |
| | Pure D-SD | 76.2% | 0.87× | 1.68 |
| | SpecVLA (SOTA) | 71.0% | 1.23× | 3.63 |
| | **HeiSD** | **73.0%** | **2.38×** | **4.75** |
| **LIBERO-Object** | AR w/o SD | 71.2% | 1.00× | – |
| | SpecVLA | 62.4% | 1.10× | 3.91 |
| | **HeiSD** | **71.0%** | **2.45×** | **4.94** |
| **LIBERO-Spatial** | AR w/o SD | 82.8% | 1.00× | – |
| | SpecVLA | 80.4% | 1.26× | 3.80 |
| | **HeiSD** | **78.0%** | **1.90×** | **4.83** |
| **LIBERO-Long** | AR w/o SD | 54.4% | 1.00× | – |
| | SpecVLA | 46.2% | 1.13× | 3.63 |
| | **HeiSD** | **47.0%** | **1.79×** | **4.96** |

关键观察：
- **Pure R-SD 和 Pure D-SD 单独使用几乎无法加速**（甚至减速），验证了混合使用的必要性
- HeiSD 相比 SpecVLA 加速 **1.51×-2.22×**，且 SR 更高（SpecVLA SR 下降严重）
- 接受长度 AL 提升到 4.75-4.96（SpecVLA 3.63），主要得益于 verify-skip 和序列级宽松接受

### 4.2 真实世界结果

| 任务类别 | 微调 SR | HeiSD SR | 加速 | AL |
| --- | --- | --- | --- | --- |
| Atomic Grasping | 87.2% | 86.0% | 2.33× | 4.47 |
| Spatial Displacement | 77.3% | 75.1% | 2.41× | 4.39 |
| Composite Sequential | 71.7% | 67.8% | 2.06× | 4.15 |

SR 损失仅 1.2%-3.9%，加速 2.06×-2.41×。

### 4.3 消融实验

| 配置 | SR | 加速 | AL |
| --- | --- | --- | --- |
| Only Hybrid SD (仅融合指标) | 74.0% | 1.05× | 1.05 |
| + Adaptive Verify-Skip | 73.0% ↓1.0% | 2.08× ↑1.03× | 4.04 ↑2.99 |
| + Seq-Wise Relaxed Accept | 73.0% ↑0.0% | 2.38× ↑0.30× | 4.50 ↑0.46 |

- **仅用融合指标**：可定义边界但检索 SD 本身无效（AL=1.05），几乎无加速
- **+Verify-Skip**：AL 跃升到 4.04，是**最大的加速来源**
- **+Seq-Wise Relaxed Accept**：额外 0.30× 加速，不损 SR

---

## 五、局限性

1. **仅适用于自回归 VLA**：SD 本质上是自回归加速方法，不适用于扩散策略（π₀、Diffusion Policy 等）
2. **超参数需手动设定**：$w$ 和 $\alpha$ 需要针对每个环境调参，缺乏自动确定方法
3. **数据库依赖**：检索 SD 需要预构建的演示数据库，对新任务需要重新采集数据
4. **Verify-skip 和序列级宽松接受对输出分布的影响**未分析（论文明确声明超出范围）
5. **数据库扩展不提升质量**：添加异构数据（SimplerEnv）到数据库并不改善 SR

---

## 六、个人思考

### 6.1 混合 SD 的核心洞察

HeiSD 的关键贡献不在于单独优化某一类 SD，而是发现了**轨迹运动学特征与 SD 类型适配性的对应关系**：快速直线运动适合检索（历史轨迹可复用），精细弯曲操作需要 drafter（需要逐步推理）。这是一个从机器人运动学视角重新理解推测解码的有趣角度。

### 6.2 与项目中已有 VLA 加速方法的关系

- **PD-VLA**（Jacobi 并行解码，2.52×）：不改模型不训练，但仅适用于 action chunking VLA
- **RTC**（异步修复执行，π₀.₅ 快 20%）：针对 flow-based VLA
- **VLA-Cache / SD-VLA / LAC**：Token 缓存/剪枝，减少跨帧冗余计算
- **HeiSD**：推测解码路线，正交于 Token 缓存方法，**理论上可与 VLA-Cache 等叠加使用**

HeiSD 的 2.45× 加速已超过大多数 Token 剪枝方法（1.7×-2.26×），但代价是需要额外数据库和 drafter 模型。

### 6.3 序列级宽松接受的设计哲学

现有 SD（如 SpecVLA）都是 token 级验证：每个 token 独立判断接受/拒绝。HeiSD 的序列级设计利用了 VLA 动作的**运动学语义结构**——X/Y/Z 作为位置是高度耦合的，$r_X$/$r_Y$/$r_Z$ 作为旋转也是。单个 token 可以有较大偏差，只要整体序列偏差小即可。这种设计比 token 级宽松接受更合理。

### 6.4 CPU+GPU 异构部署的实用价值

将数据库卸载到 CPU 不仅解决了 GPU 显存不足的问题，还带来了 1.04×-1.09× 的额外加速——CPU 在检索操作上比 GPU 更高效（无需 GPU 针对检索的特殊优化）。这个设计使 HeiSD 可以在 A100 40GB 上运行，而非必须 80GB，提升了实用性。

---

## 七、参考

- **SpecVLA** (Wang et al., 2025) — token 级宽松接受的 VLA 推测解码
- **OpenVLA** (Kim et al., 2024) — 开源 VLA 基线
- **Eagle-2** (Li et al., 2024) — 动态 draft 树解码
- **LIBERO** (Liu et al., 2023) — 操作基准
- **Qdrant** — 高性能向量数据库
- **DINOv2** (Oquab et al., 2023) — 自监督视觉特征
- **SigLIP** (Zhai et al., 2023) — 视觉-语言对齐特征
- **PD-VLA** (2025) — Jacobi 并行解码加速 VLA
- **VLA-Cache** (2025) — 跨帧 Token 缓存加速 VLA
