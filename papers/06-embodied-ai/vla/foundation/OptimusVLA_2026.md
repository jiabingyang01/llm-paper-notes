# OptimusVLA：双记忆增强 VLA 框架——全局先验与局部一致性的统一

> **论文**：*Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation*
>
> **作者**：Zaijing Li, Bing Hu, Rui Shao, Gongwei Chen, Dongmei Jiang, Pengwei Xie, Jianye Hao, Liqiang Nie
>
> **机构**：哈尔滨工业大学（深圳）、鹏城实验室、深圳坪山区研究院、华为诺亚方舟实验室
>
> **发布时间**：2026年2月
>
> **会议**：CVPR 2026
>
> **链接**：[arXiv](https://arxiv.org/abs/2602.20200) | [项目主页](https://cybertronagent.github.io/OptimusVLA.github.io/)
>
> **分类标签**：`VLA` `Flow Matching` `记忆增强` `高效推理` `时序一致性`

---

## 一句话总结

OptimusVLA 提出双记忆机制——Global Prior Memory (GPM) 用检索到的任务级先验替代高斯噪声初始化以缩短 flow matching 的生成路径、Local Consistency Memory (LCM) 用轻量 Mamba 结构建模动作历史注入时序一致性约束——在 LIBERO 上达 98.6%、真实世界推理加速 2.9 倍。

---

## 一、问题与动机

分层式 VLA 模型（VLM 骨架 + 生成式策略）已成为机器人操作的主流范式，但其**动作生成过程**存在两个关键瓶颈：

### 1.1 低推理效率：先验-目标分布鸿沟

标准 flow matching 或 diffusion 策略以**各向同性高斯噪声** $\mathcal{N}(0, I)$ 作为生成起点。然而噪声分布与结构化动作分布之间的跨域变换距离很大，导致：

- 需要大量 NFE（Number of Function Evaluations）才能收敛到高质量动作
- 随机起点频繁落入运动学不可行区域，产生无效采样

一种朴素思路是直接用动作先验作为起点，但这会严重限制多样性——学到的映射退化为 "类似目标" 的确定性函数，丧失泛化能力。

### 1.2 时序鲁棒性差：马尔可夫假设的局限

现有 VLA 模型（如 $\pi_0$、OpenVLA）仅基于当前帧生成动作，属于**马尔可夫假设**。这导致：

- **阶段混淆**：无法区分视觉上相似但任务阶段不同的状态（例如，"抽屉还没打开" vs "抽屉刚被关上"）
- **抖动控制**：缺乏与历史轨迹的一致性约束，生成的动作不平滑

虽然一些工作尝试拼接长序列历史观测，但这会大幅增加推理延迟和显存，并且与 VLA 的单帧预训练分布不一致。也有工作（如 MemoryVLA）使用 working memory 建模观测历史，但每次更新都需要完整的 VLM 前向传播，造成吞吐瓶颈。

### 1.3 OptimusVLA 的核心思路

OptimusVLA 用两个互补的记忆模块解决上述问题：

- **GPM（全局先验记忆）**：将先验初始化视为**记忆检索问题**而非固定噪声设计，从语义相似的历史轨迹中构造任务级先验分布，大幅缩短 flow 的生成路径
- **LCM（局部一致性记忆）**：用轻量结构编码近期动作序列，注入时序一致性约束和进度感知，不需要重复调用 VLM

---

## 二、预备知识：Conditional Flow Matching

Conditional Flow Matching (CFM) 训练一个时间条件化的速度场 $v_\theta(x, t)$，将源分布 $\mathcal{P}_0$（通常为高斯噪声）传输到目标动作分布 $\mathcal{P}_1$。采用最优传输直线路径：

$$x_t = (1 - t)x_0 + tx_1, \quad t \in [0, 1]$$

对应的目标速度场为常数：$u_t(x_t \mid x_0, x_1) = x_1 - x_0$。训练目标：

$$\min_\theta \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x \sim p_t(x)} \left\| v_\theta(t, x) - u_t(x) \right\|_2^2$$

推理时通过求解 ODE $dx_t / dt = v_\theta(t, x_t)$ 生成动作。当 $\mathcal{P}_0 = \mathcal{N}(0, I)$ 而 $\mathcal{P}_1$ 是结构化动作分布时，源-目标差距大，需要多步 NFE 才能生成高质量动作。

**OptimusVLA 的关键洞察**：训练时仍用 $\mathcal{P}_0 = \mathcal{N}(0, I)$ 保证向量场的泛化性；但推理时用 GPM 构造的任务级先验 $\mathcal{P}_\text{re}$ 替换标准噪声起点，将 flow ODE 的起点拉近目标流形。

---

## 三、核心方法

### 3.1 框架总览

OptimusVLA 由四个组件构成：

1. **VLM 骨架**：将观测 $O_t$（多视角图像 + 本体感知）和指令 $\ell$ 编码为多模态表征 $E_\text{emb}$
2. **GPM**：从 $E_\text{emb}$ 提取检索 token，查询记忆库获取任务级先验分布，采样带自适应噪声的初始化
3. **LCM**：编码上一个动作块 $\mathbf{A}_{t-1}$，生成一致性偏置 $\mathbf{B}_t$
4. **Flow Policy**：将 $\hat{X}_t + \mathbf{B}_t$ 通过自适应 NFE 去噪为最终动作块 $a_{t+1:t+H}$

数学流程：

$$E_\text{emb} \leftarrow \text{VLM}(O_t, \ell) \tag{3}$$

$$\mathcal{P}_\text{re} \leftarrow \text{GPM}(z_\text{re}) \tag{4}$$

$$\mathbf{B}_t \leftarrow \text{LCM}(\mathbf{A}_{t-1}) \tag{5}$$

$$\mathbf{X}_t = \hat{X}_t + \mathbf{B}_t \tag{6}$$

$$a_{t+1:t+H} \leftarrow p_\theta(\mathbf{X}_t, N) \tag{7}$$

### 3.2 Global Prior Memory (GPM)

GPM 是一个长期记忆模块，包含三个组件：

#### 3.2.1 Prior Head

一个轻量 MLP，将多模态表征 $E_\text{emb}$ 投影为归一化的检索 token $z_\text{re}$：

$$z_\text{re} = \text{PriorHead}(E_\text{emb})$$

#### 3.2.2 Memory Bank

存储 $M$ 个键值对 $\{z_m, J_m\}_{m=1}^M$，其中 $z_m$ 是任务嵌入，$J_m$ 是对应的完整轨迹。检索过程：

$$\{J_i, s_i\}_{i=1}^k \leftarrow \text{MemoryBank}(z_\text{re})$$

计算 softmax 权重和全局相似度：

$$\alpha_i = \text{softmax}(s_i / \tau_s), \quad \bar{s} = \sum_{i=1}^k \alpha_i s_i$$

对每条检索到的轨迹 $J_i$，通过滑动窗口提取动作块 $C_i \in \mathbb{R}^{H \times A}$。构造任务级先验 $\mathcal{P}_\text{re} = \mathcal{N}(\mu, \text{diag}(\text{Var}))$：

$$\mu = \sum_{i=1}^k \alpha_i C_i, \quad \text{Var} = \sum_{i=1}^k \alpha_i (C_i - \mu)^{\odot 2}$$

其中 $\odot 2$ 表示逐元素平方。

直觉理解：类似的任务（如"抓杯子"和"抓盘子"）共享相似的动作分布。GPM 从语义相似的历史轨迹中构造一个**加权高斯混合**近似，作为 flow 的起点。这比从 $\mathcal{N}(0, I)$ 出发，距离目标近得多。

#### 3.2.3 Prior-Aware Sampler

根据检索置信度 $\bar{s}$ 自适应调整噪声尺度 $\lambda$ 和步数 $N$：

$$\lambda = \lambda_\max - \frac{\bar{s}+1}{2}(\lambda_\max - \lambda_\min)$$

$$N = N_\min + \left(1 - \frac{\bar{s}+1}{2}\right)(N_\max - N_\min)$$

$$\hat{X}_t = \mu + \lambda \left(\epsilon \odot \sqrt{\text{Var}}\right), \quad \epsilon \sim \mathcal{N}(0, I)$$

**自适应逻辑**：

- **检索置信度高**（$\bar{s} \approx 1$）：$\lambda$ 减小（更多依赖检索均值），$N$ 减小（更简单的传输路径）→ 高效推理
- **检索置信度低**（$\bar{s} \approx 0$，新颖场景）：$\lambda$ 增大（更多随机探索），$N$ 增大（需要更多步）→ 优雅退化为标准 flow

这种设计既利用了先验知识加速已见过的任务，又保留了对未见任务的泛化能力。

### 3.3 Local Consistency Memory (LCM)

LCM 是一个轻量工作记忆，包含两个子模块：

#### 3.3.1 Consistency Layer

在时间步 $t$，接收上一个动作块 $\mathbf{A}_{t-1} = [a_{t-H+1}, \dots, a_t] \in \mathbb{R}^{H \times A}$，使用**自注意力机制**建模动作块内部的依赖关系：

$$\hat{\mathbf{B}}_{t-1} \leftarrow \text{ConsistencyLayer}(\mathbf{A}_{t-1})$$

#### 3.3.2 Dynamic Awareness Module

基于 **Mamba** 结构（选择性状态空间模型），捕捉动作块间的时序动态。输入 $\hat{\mathbf{B}}_{t-1}$，更新内部状态，预测下一步的一致性偏置：

$$\mathbf{B}_t \leftarrow \text{DynamicAwareness}(\hat{\mathbf{B}}_{t-1})$$

**为什么选 Mamba？** SSM 的线性复杂度使其能以极低开销建模长程依赖，非常适合实时控制场景。

**LCM 的作用**：将时序一致性转化为加性偏置注入策略输入。用大白话说，LCM 告诉策略 "你之前在做什么，接下来应该顺着来"，从而：
- 避免相似视觉观测导致的阶段混淆
- 产生更平滑的控制轨迹
- 无需修改 VLA 预训练范式，也不需要重复调用 VLM

### 3.4 三阶段训练流程

**Stage 1：VLA 预训练。** 基于 $\pi_{0.5}$ 的架构和协议预训练分层 VLA 模型。此阶段不附加 GPM 和 LCM。

**Stage 2：GPM 训练（Prior Head）。** 冻结所有 VLA 参数，仅训练 Prior Head。使用 InfoNCE 对比损失，使相同语义任务的嵌入聚拢、不同任务的分离：

$$\mathcal{L}_\text{GPM} = -\mathbb{E}_q \left[ \log \frac{\exp(\text{sim}(z_\text{re}, z^+) / \tau_c)}{\sum_{j \in \mathcal{N}(q)} \exp(\text{sim}(z_\text{re}, z_j) / \tau_c)} \right]$$

训练后冻结 Prior Head，用 FAISS IndexFlatIP 构建记忆库索引。使用 **Task-Pair Batch Sampler** 确保每个 batch 至少包含同一任务的两条轨迹。

**Stage 3：LCM 训练。** 冻结 VLM、flow policy 和 GPM，训练 LCM 预测 GPM 先验均值与真实动作之间的残差：

$$\mathbf{B}_t^\star = \mathbf{A}_t^\star - \mu_t$$

$$\mathcal{L}_\text{LCM} = \mathbb{E}_{(\mathbf{A}_{t-1}, \mathbf{A}_t^\star, \mu_t) \sim \mathcal{D}} \left[ \|\mathbf{B}_t - \mathbf{B}_t^\star\|_2^2 \right]$$

训练时采用**冷启动策略**：以概率 $p_\text{cold}$ 将 $\mathbf{A}_{t-1}$ 置零，确保模型在无历史信息（如 episode 首步）时也能正常工作。

| 超参数 | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| 优化器 | AdamW | AdamW | AdamW |
| 学习率 | 5e-5 | 1e-4 | 1e-4 |
| 步数 | 30,000 | 1,000 | 1,000 |
| Batch Size | 512 | 64 | 64 |

总参数量 3.6B，在 8× NVIDIA A800 上训练。

---

## 四、实验结果

### 4.1 仿真基准

#### LIBERO

| 方法 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- |
| DP | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| $\pi_0$-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.4 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **OptimusVLA** | **99.6** | **99.8** | **98.4** | **96.4** | **98.6** |

OptimusVLA 在四个 suite 上全面领先，尤其在 LIBERO-Long（长时域任务）上比 $\pi_{0.5}$ 高出 4.0%，得益于 GPM 的任务级先验锚定生成过程，减少了长 horizon 的误差累积。

#### CALVIN (ABC → D)

| 方法 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. Len |
| --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| $\pi_{0.5}$ | 94.4 | 88.4 | 85.3 | 80.1 | 76.1 | 4.26 |
| VPP | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29 |
| **OptimusVLA** | **97.6** | **93.2** | **88.8** | **85.7** | **78.1** | **4.45** |

在 ABC → D 零样本迁移设置下，$\pi_0$ 依赖任务无关的高斯噪声，对分布偏移脆弱。OptimusVLA 通过 GPM 检索语义相似轨迹作为初始化，通过**变形已有的可行先验**而非从零生成来适应新场景。

#### RoboTwin 2.0 Hard

| 任务 | RDT | ACT | DP | DP3 | $\pi_0$ | $\pi_{0.5}$ | OptimusVLA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Click Bell | 9% | 3% | 0% | 0% | 3% | 28% | **46%** |
| Open Laptop | 32% | 0% | 0% | 7% | 46% | 38% | **48%** |
| Stack Bowls Two | 30% | 0% | 0% | 6% | 41% | 49% | **58%** |
| **Average (8 tasks)** | 20% | 2% | 1% | 11% | 25% | 29% | **38%** |

双臂操作需要高度的时序和臂间一致性。RDT 等方法缺乏显式的双臂协调机制；OptimusVLA 的 LCM 提供必要的一致性约束，强制生成平滑、协调的双臂轨迹。

### 4.2 真实世界评估

在 GALAXEA R1 Lite 双臂机器人（14-DoF）上评估：

**泛化任务**（4 个任务，100-150 条演示/任务，50 次 rollout）：OptimusVLA 平均成功率 **85.0%**，比 $\pi_0$ 高 42.9%（59.5% → 85.0%），对光照和场景变化鲁棒。

**长时域任务**（4 个任务，200-300 条演示/任务，25 次 rollout）：OptimusVLA 平均成功率 **64.0%**，比 $\pi_0$ 高 52.4%，展现了优越的长程动作稳定性和双臂协调。

### 4.3 推理效率

| 方法 | LIBERO NFE | LIBERO 推理时间 (ms) | Real-World NFE | Real-World 推理时间 (ms) |
| --- | --- | --- | --- | --- |
| OpenVLA | — | 552 | — | 254 |
| $\pi_0$ | 10 | 187 | 10 | 69 |
| $\pi_{0.5}$ | 10 | 133 | 10 | 57 |
| **OptimusVLA** | **3.2** | **112** | **3.4** | **39** |

OptimusVLA 在 LIBERO 上实现 **3.1× NFE 减少**和 **6.5× 推理加速**（相对 OpenVLA），在真实世界实现 **2.9× NFE 减少**。GPM 和 LCM 引入的开销极小，性能提升主要来自 NFE 的大幅下降。

### 4.4 消融实验

#### GPM 和 LCM 的贡献

| GPM | LCM | LIBERO-Long | CALVIN | Real-World 泛化 |
| --- | --- | --- | --- | --- |
| ✓ | ✓ | 96.4 | 4.45 | 85.0 |
| ✗ | ✓ | 93.2 (↓3.3%) | 4.28 (↓3.8%) | 77.0 (↓9.4%) |
| ✓ | ✗ | 94.8 (↓1.7%) | 4.38 (↓1.6%) | 79.5 (↓6.5%) |
| ✗ | ✗ | 92.4 (↓4.1%) | 4.26 (↓4.3%) | 75.0 (↓11.8%) |

- 去掉 GPM 在真实世界掉 9.4%，说明任务级先验对跨场景泛化至关重要
- 去掉 LCM 在 LIBERO-Long 掉 1.7%，说明时序一致性对长时域任务不可或缺
- 两者同时去掉掉 11.8%，接近 $\pi_{0.5}$ 的性能

#### 记忆库规模

| Num=6500, k=8 | Num=6500, k=16 | Num=1300, k=1 | Num=1300, k=8 | Num=130, k=1 | Num=130, k=8 |
| --- | --- | --- | --- | --- | --- |
| **96.4** | 94.8 | 92.6 | 95.2 | 92.4 | 93.6 |

- 每个任务仅存 1 条轨迹会导致先验过于确定性
- 检索数 $k$ 太小会过拟合单条轨迹，$k=8$ 允许构造更鲁棒的高斯混合先验

### 4.5 训练效率

在相同初始化权重下，OptimusVLA 在 LIBERO-Goal 上 18,000 步达到 97.6%，而 $\pi_{0.5}$ 需要 26,000 步达到相似水平。GPM 提供的任务级先验将初始化放在目标流形附近，降低了变换复杂度，加速了收敛。

---

## 五、局限性与未来方向

1. **记忆库覆盖率**：GPM 的效果受限于记忆库的覆盖面和质量。当任务或场景显著偏离已存储的经验时，检索到的先验可能产生误导。未来方向是开发**自适应记忆机制**——在线更新、遗忘和不确定性感知检索。

2. **LCM 的局部性**：LCM 聚焦于固定长度动作块的局部一致性。对于需要跨多个阶段推理、存在延迟效应的任务，可能不够充分。

3. **端到端联合训练**：当前三阶段分离训练可能丢失组件间的协同优化潜力。联合训练 GPM、LCM 和 flow policy 是一个有价值的未来方向。

---

## 六、个人思考

### 6.1 与 $\pi_{0.5}$ / $\pi_0$ 的关系

OptimusVLA 直接构建在 $\pi_{0.5}$ 之上，保持了预训练 VLA 的框架不变，仅在**推理阶段的初始化和输入**上做文章。这是一种非常工程友好的增强方式——不需要改变预训练目标，Stage 2 和 Stage 3 各只需 1000 步训练。

### 6.2 GPM 的本质：从"固定分布"到"数据驱动分布"

GPM 的核心贡献不在于"用先验替代噪声"这个想法本身，而在于**如何获取和组合先验**：
- InfoNCE 训练的 Prior Head 确保了语义层面的检索质量
- 滑动窗口对齐和进度标量 $\rho_t$ 确保了时间层面的正确性
- 自适应噪声和 NFE 确保了泛化-效率的平衡

这种设计让 GPM 在已见任务上极为高效（NFE 低至 1-3），在新任务上优雅退化为标准 flow。

### 6.3 LCM vs MemoryVLA

与 MemoryVLA 的关键区别在于 LCM 建模的是**动作历史**而非**观测历史**。这意味着：
- 不需要在每一步调用 VLM，计算开销极低
- 直接以加性偏置注入策略输入，不修改模型结构
- 但也意味着 LCM 无法利用视觉层面的时序信息

### 6.4 NFE 的自适应调度

OptimusVLA 的 NFE 从固定的 10 步降到自适应的 3-4 步，是推理加速的核心。这种基于检索置信度的调度非常巧妙——本质上是让模型"对自己有多确定"来决定"花多少计算"。这与 test-time compute 的思路一脉相承，但方向相反：这里是在确定时**减少**计算。

### 6.5 与 RL 后训练方法的互补性

OptimusVLA 聚焦于推理阶段的效率和鲁棒性优化，与 RL 后训练（如 $\pi_{0.6}^*$、RISE）方向正交。理论上，OptimusVLA 的双记忆机制可以与 RL 微调结合——用 RL 改进策略质量，用 GPM+LCM 加速推理和增强时序一致性。

---

## 参考

- **$\pi_0$ / $\pi_{0.5}$**：OptimusVLA 的 VLA 基线架构和预训练权重来源
- **MemoryVLA**：另一种 VLA 记忆增强方法，使用观测历史的 working memory，OptimusVLA 与之对比并超越
- **Flow Matching (Lipman et al., 2022)**：OptimusVLA 策略的核心生成框架
- **Mamba (Gu & Dao, 2024)**：LCM 的 Dynamic Awareness Module 使用的高效序列建模架构
- **RoboTwin 2.0**：双臂操作的 Hard 设置评估基准
