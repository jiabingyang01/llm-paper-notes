# DiffusionNFT：基于前向过程的扩散模型在线强化学习

> **论文**：*DiffusionNFT: Online Diffusion Reinforcement with Forward Process*
>
> **作者**：Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, Ming-Yu Liu
>
> **机构**：清华大学、NVIDIA、Stanford University
>
> **发布时间**：2025年9月
>
> **发表会议**：ICLR 2026 Oral
>
> **链接**：[arXiv](https://arxiv.org/abs/2509.16117) | [项目主页](https://research.nvidia.com/labs/dir/DiffusionNFT)
>
> **分类标签**：`扩散模型 RL` `前向过程优化` `Negative-aware Fine-Tuning` `CFG-Free` `Flow Matching` `SD3.5`

---

## 一句话总结

DiffusionNFT 提出在**前向加噪过程**（而非反向去噪过程）上做扩散模型在线 RL：将生成样本按奖励拆分为正/负子集，通过隐式参数化技术定义对比式策略改进方向，直接将强化信号嵌入标准 flow matching 训练目标。该方法无需似然估计、不依赖特定采样器、无需 CFG，效率比 FlowGRPO 高 3-25 倍，SD3.5-Medium 的 GenEval 从 0.24 提升到 0.98（1k 步内）。

---

## 一、问题与动机

### 1.1 扩散模型 RL 的核心困境：似然不可解

Policy Gradient 算法（PPO、GRPO）的前提是模型似然可精确计算。自回归模型天然满足这一条件，但扩散模型的似然只能通过概率 ODE 或 SDE 的变分界来近似，计算代价高昂且引入系统性偏差。

近期工作（FlowGRPO、DanceGRPO）通过**离散化反向采样过程**将扩散生成建模为多步 MDP，使相邻步的转移概率成为可解的高斯分布，从而可以套用 GRPO。但这一路线存在三个根本缺陷：

| 问题 | 说明 |
| --- | --- |
| **前向不一致性** | 仅优化反向过程，破坏了对前向扩散过程的忠实建模，模型有退化为级联高斯的风险 |
| **采样器绑定** | 训练损失与一阶 SDE 采样器耦合，无法利用 ODE 或高阶求解器的效率优势 |
| **CFG 集成复杂** | 扩散模型严重依赖 CFG，但 RL 后训练中使用 CFG 需要同时优化条件/无条件两个模型，效率低下 |

### 1.2 核心洞察：在前向过程上做 RL

一个扩散策略只有**唯一的前向（加噪）过程**，但可以有**多种反向（去噪）过程**。既然反向过程 RL 面临诸多困难，能否在前向过程上做强化学习？

DiffusionNFT 给出了肯定答案：通过 flow matching 目标在前向过程上定义策略优化，完全绕开了似然估计和反向过程离散化。

---

## 二、预备知识

### 2.1 Flow Matching 与 Velocity 参数化

前向加噪过程的闭式转移核：$x_t = \alpha_t x_0 + \sigma_t \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

速度参数化 $v_\theta(x_t, t)$ 的训练目标：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0 \sim \pi_0, \epsilon \sim \mathcal{N}(0,I)} \left[ w(t) \| v_\theta(x_t, t) - v \|_2^2 \right]$$

其中目标速度 $v = \dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon$。对于 rectified flow（$\alpha_t = 1-t, \sigma_t = t$），简化为 $v = \epsilon - x_0$。

反向采样通过 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t)$ 完成（Euler 离散化即 DDIM）。

### 2.2 FlowGRPO 的做法及其局限

FlowGRPO 将反向过程写成 SDE 形式并做 Euler 离散化：

$$\pi_\theta(x_{t-\Delta t} \mid x_t) = \mathcal{N}\left(x_t + \left[v_\theta(x_t, t) + \frac{g_t^2}{2t}(x_t + (1-t)v_\theta(x_t, t))\right]\Delta t,\ g_t^2 \Delta t \mathbf{I}\right)$$

这使每步转移为可解高斯，可套用 GRPO。但代价是：必须使用一阶 SDE 采样器、需要存储整条采样轨迹、训练损失与采样器耦合。

---

## 三、核心方法

### 3.1 问题设定：正负样本分裂

给定预训练扩散策略 $\pi^{\text{old}}$，对每个 prompt $c$ 采样 $K$ 张图像并用奖励函数 $r(x_0, c) \in [0,1]$ 评估。将样本按奖励概率软分配到正集 $\mathcal{D}^+$ 和负集 $\mathcal{D}^-$：

$$\pi^+(x_0|c) = \frac{r(x_0, c)}{p_{\pi^{\text{old}}}(\text{o}=1|c)} \pi^{\text{old}}(x_0|c), \quad \pi^-(x_0|c) = \frac{1 - r(x_0, c)}{1 - p_{\pi^{\text{old}}}(\text{o}=1|c)} \pi^{\text{old}}(x_0|c)$$

恒有 $\pi^+ \succ \pi^{\text{old}} \succ \pi^-$（正集分布优于原始分布优于负集分布）。

### 3.2 强化引导方向（Reinforcement Guidance）

仅用正样本做 Rejection Fine-Tuning（RFT）虽然简单，但无法利用负样本信息。DiffusionNFT 的核心思想是**从正负样本的对比中提取策略改进方向**：

$$v^*(x_t, c, t) := v^{\text{old}}(x_t, c, t) + \frac{1}{\beta} \Delta(x_t, c, t)$$

其中 $\Delta$ 是 reinforcement guidance，$\frac{1}{\beta}$ 是引导强度。

**Theorem 3.1（改进方向）**：正/负/原始三个速度模型 $v^+, v^-, v^{\text{old}}$ 之间的差异成比例：

$$\Delta := [1 - \alpha(x_t)] \left[ v^{\text{old}}(x_t, c, t) - v^-(x_t, c, t) \right] = \alpha(x_t) \left[ v^+(x_t, c, t) - v^{\text{old}}(x_t, c, t) \right]$$

其中 $\alpha(x_t) := \frac{\pi_t^+(x_t|c)}{\pi_t^{\text{old}}(x_t|c)} \mathbb{E}_{\pi^{\text{old}}(x_0|c)} r(x_0, c) \in [0,1]$。

用大白话说：改进方向 $\Delta$ 就是"远离负样本分布、靠近正样本分布"的速度场差值，两种表述等价。

### 3.3 隐式参数化策略优化

**Theorem 3.2（策略优化）**：定义训练目标：

$$\mathcal{L}(\theta) = \mathbb{E}_{c, \pi^{\text{old}}(x_0|c), t} \left[ r \| v_\theta^+(x_t, c, t) - v \|_2^2 + (1-r) \| v_\theta^-(x_t, c, t) - v \|_2^2 \right]$$

其中**隐式正策略**和**隐式负策略**分别为：

$$v_\theta^+(x_t, c, t) := (1 - \beta) v^{\text{old}}(x_t, c, t) + \beta v_\theta(x_t, c, t)$$

$$v_\theta^-(x_t, c, t) := (1 + \beta) v^{\text{old}}(x_t, c, t) - \beta v_\theta(x_t, c, t)$$

在无限数据和模型容量下，最优解满足：

$$v_{\theta^*}(x_t, c, t) = v^{\text{old}}(x_t, c, t) + \frac{2}{\beta} \Delta(x_t, c, t)$$

关键巧妙之处：不需要分别训练正策略模型和负策略模型，而是通过**隐式参数化**（将 $v_\theta^+$ 和 $v_\theta^-$ 都表示为 $v_\theta$ 和 $v^{\text{old}}$ 的线性组合），直接优化单一目标策略 $v_\theta$。正样本（$r$ 大）推动 $v_\theta$ 学习正策略方向，负样本（$1-r$ 大）推动 $v_\theta$ 远离负策略方向，两者通过奖励权重自然融合。

### 3.4 前向过程 RL 的四大优势

| 优势 | 说明 |
| --- | --- |
| **前向一致性** | 在前向加噪过程上定义标准扩散损失，保证模型的概率密度遵守 Fokker-Planck 方程，避免退化 |
| **采样器自由** | 训练与采样完全解耦，数据收集可以用任意黑盒求解器（ODE、高阶 ODE、SDE） |
| **隐式引导集成** | 强化信号直接嵌入策略模型，无需维护单独的引导模型，关键是支持在线 RL 的迭代更新 |
| **无似然** | 不依赖似然估计（无 Jensen 不等式近似、无反向过程离散化偏差），本质上是监督学习 |

### 3.5 实用设计

**Optimality Reward 归一化**：将原始奖励转化为 $[0,1]$ 的最优概率：

$$r(x_0, c) := \frac{1}{2} + \frac{1}{2} \text{clip}\left(\frac{r^{\text{raw}}(x_0, c) - \mathbb{E}_{\pi^{\text{old}}(\cdot|c)} r^{\text{raw}}(x_0, c)}{Z_c},\ -1,\ 1\right)$$

**Soft Update**：利用 off-policy 特性，采样策略与训练策略解耦，用 EMA 软更新：

$$\theta^{\text{old}} \leftarrow \eta_i \theta^{\text{old}} + (1 - \eta_i) \theta$$

其中 $\eta_i$ 从小值逐渐增大，平衡收敛速度与稳定性。

**自适应损失加权**：用 $x_0$ 预测器的自归一化回归替代手动调节的时间依赖权重 $w(t)$：

$$w(t) \| v_\theta - v \|_2^2 \leftarrow \frac{\| x_\theta(x_t, c, t) - x_0 \|_2^2}{\text{sg}(\text{mean}(\text{abs}(x_\theta(x_t, c, t) - x_0)))}$$

**CFG-Free**：DiffusionNFT 将 CFG 解释为一种离线强化引导（条件/无条件模型对应正/负信号），因此可以完全丢弃 CFG，仅从条件模型初始化。虽然初始性能很低（GenEval 0.24），但 RL 训练后迅速超越 CFG 基线（0.63）。

---

## 四、实验结果

### 4.1 实验设置

基座模型：SD3.5-Medium（2.5B 参数），512×512 分辨率。LoRA 微调（$\alpha=64, r=32$）。

奖励模型：
- 规则奖励：GenEval（组合生成）、OCR（文字渲染）
- 模型奖励：PickScore、ClipScore、HPSv2.1、Aesthetics、ImageReward、UnifiedReward

### 4.2 多奖励联合训练

从 CFG-free 的 SD3.5-M 出发，联合优化 5 个奖励（GenEval、OCR、PickScore、ClipScore、HPSv2.1），共 1.7k 迭代：

| 模型 | GenEval | OCR | PickScore | ClipScore | HPSv2.1 | Aesthetic | ImgRwd | UniRwd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD-XL (1024) | 0.55 | 0.14 | 22.42 | 0.287 | 0.280 | 5.60 | 0.76 | 2.93 |
| SD3.5-L (1024) | 0.71 | 0.68 | 22.91 | 0.289 | 0.288 | 5.50 | 0.96 | 3.25 |
| FLUX.1-Dev | 0.66 | 0.59 | 22.84 | 0.295 | 0.274 | 5.71 | 0.96 | 3.27 |
| SD3.5-M (w/o CFG) | 0.24 | 0.12 | 20.51 | 0.237 | 0.204 | 5.13 | -0.58 | 2.02 |
| SD3.5-M + CFG | 0.63 | 0.59 | 22.34 | 0.285 | 0.279 | 5.36 | 0.85 | 3.03 |
| SD3.5-M + FlowGRPO | **0.95** | 0.66 | 22.51 | **0.293** | 0.274 | 5.32 | 1.06 | 3.18 |
| **SD3.5-M + Ours** | 0.94 | **0.91** | **23.80** | **0.293** | **0.331** | **6.01** | **1.49** | **3.49** |

关键发现：
- 单一 CFG-free 模型在所有指标上超越 CFG 基线
- 在域外指标（Aesthetic、ImageReward、UnifiedReward）上也大幅提升
- 超越参数量大数倍的 SD3.5-L（8B）和 FLUX.1-Dev（12B）

### 4.3 单奖励 Head-to-Head 对比

| 任务 | DiffusionNFT 效率倍数 | DiffusionNFT 最终分数 | FlowGRPO 最终分数 |
| --- | --- | --- | --- |
| GenEval | **25×** | **0.98** | 0.95 |
| OCR | **24×** | 与 FlowGRPO 持平 | — |
| PickScore | **8×** | 23.80 | ~23.50 |
| HPSv2.1 | **3×** | 0.38 | ~0.37 |

GenEval 上的差距最为惊人：DiffusionNFT 在 ~1k 步内达到 0.98，而 FlowGRPO 需要 >5k 步且额外使用 CFG 才达到 0.95。

### 4.4 消融实验

**负样本损失的关键性**：移除负策略损失 $v_\theta^-$ 后，在线训练中奖励几乎立即崩溃。这与 LLM 中 RFT 仍是强基线的现象不同——在扩散模型 RL 中，负信号至关重要。

**采样器选择**：ODE 采样器优于 SDE（尤其是噪声敏感的 PickScore），二阶 ODE 略优于一阶。这验证了采样器自由的优势——可以选择最优采样器。

**自适应加权**：大 $t$ 时给 flow matching 损失更高权重效果最好，反向策略（$w(t)=1-t$）导致崩溃。自适应加权稳定匹配或超越所有启发式选择。

**Soft Update**：纯 on-policy（$\eta=0$）快但不稳定；过度 off-policy（$\eta=0.9$）收敛太慢。最佳策略是 $\eta_i$ 从小逐渐增大。

**引导强度**：$\beta$ 近 1 表现稳定，实践中取 $\beta=1$ 或 $0.1$。

---

## 五、局限性与未来方向

1. **仅验证了文本到图像**：尚未在视频生成或机器人控制（VLA）等其他扩散策略场景验证
2. **奖励模型依赖**：多奖励联合训练的表现上限受限于奖励模型质量，可能存在 reward hacking
3. **Soft Update 调参**：$\eta$ 的调度策略对稳定性影响大，不同奖励需要不同的 $\eta_{\max}$（如 OCR 需要 0.999）
4. **理论到实践的 gap**：定理假设无限数据和模型容量，有限样本下的收敛保证尚未建立

---

## 六、个人思考

### 6.1 前向 vs. 反向：RL 应该在哪里做？

这是本文最深刻的洞察。FlowGRPO 等方法在反向去噪过程上做 RL，本质上是把扩散生成看作多步决策问题，需要离散化轨迹、估计步级似然。DiffusionNFT 转换视角——在前向加噪过程上做 RL，利用前向过程的闭式性质完全绕开了似然估计。这个转换之所以可行，关键在于 **flow matching 训练目标本身就定义在前向过程上**，只需将 RL 信号（正/负对比）注入这个已有的训练框架。

### 6.2 与 NFT（LLM 版本）的关系

DiffusionNFT 是 LLM 领域 NFT（Negative-aware Fine-Tuning）范式向扩散模型的扩展。两者共享同一思想：通过正/负样本对比定义隐式的策略改进方向，将 RL 信号嵌入监督学习目标。关键区别在于 LLM 的 NFT 在 token 级别操作（离散空间），而 DiffusionNFT 在速度场上操作（连续空间），数学形式更优雅。

### 6.3 对 VLA RL 后训练的启示

这项工作对机器人 VLA 的 RL 后训练有重要启示。当前 VLA RL 方法（如 FPO++、SAC Flow、πRL）大多在反向采样过程上做 Policy Gradient，面临与 FlowGRPO 类似的似然估计和采样器绑定问题。DiffusionNFT 的前向过程 RL 范式**可以直接迁移到 VLA 领域**——对 flow matching VLA（如 π₀ 系列）的动作生成做正/负对比式优化，可能带来类似的效率提升。

### 6.4 CFG-Free 的意义

DiffusionNFT 将 CFG 解释为一种静态的正/负对比引导，并证明这种引导可以通过在线 RL 学习得到。这意味着 CFG 并非扩散模型的必要组件，而是一种可以被 RL 后训练替代的"先验"技巧。这一发现与近期 guidance-free 生成的趋势一致，暗示未来的扩散模型可能不再需要 CFG。

### 6.5 负样本在扩散 RL 中的特殊重要性

消融实验显示，移除负样本损失后扩散模型 RL 立即崩溃，而在 LLM 中 RFT（仅用正样本）仍是强基线。这一差异可能源于：扩散模型的输出空间（像素/潜空间）远比 LLM 的 token 空间高维，仅靠正样本无法提供足够的梯度信号来约束模型在如此高维空间中的行为，负样本提供了关键的"推斥力"防止模型退化。

---

## 参考

- **NFT**（Chen et al., 2025）：LLM 版本的 Negative-aware Fine-Tuning，DiffusionNFT 的概念来源
- **FlowGRPO**（Liu et al., 2025）：在反向 SDE 过程上做 GRPO 的扩散 RL 基线
- **DanceGRPO**（Xue et al., 2025）：另一个反向过程 GRPO 扩展，同样面临采样器耦合问题
- **SD3.5**（Esser et al., ICML 2024）：Rectified Flow Transformer 基座模型
- **Diffusion-DPO**（Wallace et al., CVPR 2024）：用 DPO 对齐扩散模型的离线方法
- **FLAC**（Lv et al., 2026）：本项目中已有的另一篇扩散/flow 策略 RL 方法，用动能正则化做 MaxEnt RL
