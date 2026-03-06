# GigaBrain-0.5M*：基于世界模型强化学习的 VLA——RAMP 方法详解

> 论文：*GigaBrain-0.5M\*: a VLA That Learns From World Model-Based Reinforcement Learning*
>
> 机构：GigaAI
>
> 发布时间：2026年2月
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.12099) | [项目主页](https://gigabrain05m.github.io) | [GigaTrain 代码](https://github.com/open-gigaai/giga-train)
>
> 分类标签：`World Model RL` `VLA` `RAMP` `优势条件化` `Wan2.2`

---

## 一句话总结

提出 **RAMP**（Reinforcement leArning via world Model-conditioned Policy），用视频世界模型同时预测未来状态和价值，将其作为策略的条件输入，结合 human-in-the-loop rollout 迭代训练，在 Laundry Folding、Box Packing、Espresso Preparation 等任务上比 RECAP 基线提升约 **30%**，并从理论上证明 RECAP 是 RAMP 的退化特例。

---

## 一、问题与动机

### 1.1 当前 VLA 的短视性缺陷

主流 VLA 模型（如 π₀.₅、GigaBrain-0 等）在 action chunk 生成时仅依赖**当前观测**，缺乏对未来状态的预见能力：

- **反应式控制**：策略只能看到"现在"，无法预判"接下来会发生什么"
- **长时域任务脆弱**：在需要多步序贯规划的任务（如折叠衣物、制作咖啡）中，缺乏前瞻性导致执行中期频繁失败
- **依赖短视观测**：无法利用"执行当前动作后场景会变成什么样"的信息来辅助决策

### 1.2 RECAP 的局限性

π₀.₆* 提出的 RECAP 通过**二值化优势指标** $I \in \{0, 1\}$ 条件化策略，已经证明了优势条件化的有效性。但 RECAP 存在根本性的信息瓶颈：

- **信号极度稀疏**：只有 1 bit 信息（好/不好），对策略的引导力非常有限
- **隐式平均未来**：策略 $\pi(a|o, I)$ 必须在所有可能的未来状态上做平均，无法针对具体的未来场景做出最优决策

### 1.3 RAMP 的核心洞察

RAMP 的关键创新是引入**世界模型预测的未来状态** $\mathbf{z}$ 作为策略的额外条件输入：

- 世界模型不仅预测价值（用于计算优势），还预测**未来视觉状态的隐表征**
- 策略同时以 $(I, \mathbf{z})$ 为条件生成动作——既知道"做得好不好"，又看到"未来会怎样"
- 这从根本上将策略从"对未来的平均猜测"升级为"针对具体未来的精确规划"

---

## 二、预备知识

### 2.1 GigaBrain-0.5 架构

GigaBrain-0.5 继承了 GigaBrain-0 的端到端 VLA 架构：

- **VLM 骨干**：预训练 PaliGemma-2（Steiner et al., 2024）编码多模态输入
- **架构创新**：Mixture-of-Transformers (MoT) 骨干，分离语言和动作的计算路径
- **动作生成**：Action Diffusion Transformer (DiT) + Flow Matching 预测 action chunk
- **推理增强**：生成 Embodied Chain-of-Thought（子目标语言 + 离散动作 token + 2D 操作轨迹）

联合训练目标：

$$\mathcal{L} = \mathbb{E}_{\mathcal{D}, \tau, \epsilon}\left[-\sum_{j=1}^{n-1} M_{\text{CoT},j} \log p_\theta(x_{j+1}|x_{1:j}) + \|\epsilon - a_{\text{chunk}} - f_\theta(a_{\text{chunk}}^{\tau,\epsilon})\|^2 + \lambda\|\text{GRU}(\hat{\mathbf{t}}_{1:10}) - \mathbf{t}_{1:10}\|^2\right]$$

其中第一项是 CoT 自回归损失，第二项是 flow matching 动作生成损失，第三项是 2D 轨迹预测损失。深度信息和 2D 轨迹被设计为可选状态，使模型适应不同传感器配置。

### 2.2 RECAP 回顾

RECAP 的训练目标：

$$\mathcal{L}(\theta) = \mathbb{E}_D\left[-\log \pi_\theta(a|\mathbf{o}, l) - \alpha \log \pi_\theta(a|I, \mathbf{o}, l)\right]$$

其中 $I = \mathbb{1}(A(\mathbf{o}, l, a) > \epsilon)$ 是二值化优势指标。策略以观测 $\mathbf{o}$、语言指令 $l$ 和优势指标 $I$ 为输入。

RECAP 的问题在于：策略只看到 $I$（1 bit），完全不知道未来状态会是什么样。

---

## 三、核心方法：RAMP

### 3.1 RAMP 的理论基础

RAMP 将状态空间扩展为 $\mathbf{S} = (\mathbf{o}, \mathbf{z}, l)$，其中 $\mathbf{z}$ 是世界模型预测的未来隐状态。在 KL 正则化 RL 框架下，最优策略的闭式解为：

$$\hat{\pi}(a|\mathbf{S}) \propto \pi_{\text{ref}}(a|\mathbf{S}) \exp\left(\frac{A^{\pi_{\text{ref}}}(\mathbf{S}, a)}{\beta}\right)$$

引入二值改进指标 $I$ 并应用贝叶斯定理，最终训练目标为：

$$\mathcal{L}(\theta) = \mathbb{E}_D\left[-\log \pi_\theta(a|\mathbf{o}, \mathbf{z}, l) - \alpha \log \pi_\theta(a|I, \mathbf{o}, \mathbf{z}_t, l)\right]$$

其中 $I = \mathbb{1}(A(\mathbf{o}, \mathbf{z}, l, a) > \epsilon)$。

### 3.2 RECAP 是 RAMP 的特例——理论证明

从概率建模角度，RECAP 的策略 $\pi(a|\mathbf{o}, I)$ 是 RAMP 策略 $\pi(a|\mathbf{o}, \mathbf{z}, I)$ 对隐状态 $\mathbf{z}$ 的边际化：

$$\pi_{\text{RECAP}}(a|\mathbf{o}, I) = \int_{\mathbf{z}} \pi_{\text{RAMP}}(a|\mathbf{o}, \mathbf{z}, I) \, p(\mathbf{z}|\mathbf{o}, I) \, d\mathbf{z}$$

用大白话说：RECAP 学的是一个"平均策略"——它必须在所有可能的未来状态上取平均，做出一个折中的决策。而 RAMP 直接告诉策略"未来会是这样的"（通过 $\mathbf{z}$），策略可以针对这个具体的未来做出精确的动作规划。

从信息论角度：引入 $\mathbf{z}$ 提供了显著的**信息增益**，降低了动作生成的条件熵：

$$H(a|\mathbf{o}, \mathbf{z}, I) \leq H(a|\mathbf{o}, I)$$

RECAP 只有 1 bit 的 $I$ 作为 RL 信号，而 RAMP 有 $I$ + 整个未来状态隐表征 $\mathbf{z}$，信息量差距巨大。

### 3.3 四阶段迭代流程

#### Stage 1：世界模型预训练

世界模型 $\mathcal{W}_\varphi$ 基于 **Wan2.2** 视频生成模型，联合预测**未来视觉状态**和**价值估计**。

**奖励定义**（与 RECAP 相同的极简设计）：

$$r_t = \begin{cases} 0 & \text{if } t=T \text{ 且成功} \\ -C_{\text{fail}} & \text{if } t=T \text{ 且失败} \\ -1 & \text{otherwise} \end{cases}$$

**隐状态构建**：未来观测 $\{o_{t+i}\}_{i \in \{12,24,36,48\}}$ 经预训练 VAE 编码为视觉隐向量 $\mathbf{z}_t \in \mathbb{R}^{H' \times W' \times C'}$，价值 $v_t$ 和本体感受状态 $\mathbf{p}_t$ 通过空间铺排投影 $\Psi(\cdot)$ 对齐空间维度，然后通道拼接：

$$\mathbf{s}_t = [\mathbf{z}_t;\; \Psi(v_t);\; \Psi(\mathbf{p}_t)]$$

**关键设计**：价值信号被编码为额外的 latent frame 直接拼入视觉隐状态，无需修改 DiT 架构。

**训练目标**（Flow Matching）：

$$\mathcal{L}_{\text{WM}} = \mathbb{E}_{\mathcal{D}, \tau, \epsilon}\left[\|\mathcal{W}_\varphi(\mathbf{s}_{\text{future}}^{\tau,\epsilon}) - (\mathbf{s}_{\text{future}} - \epsilon)\|^2\right]$$

其中 $\mathbf{s}_{\text{future}}^{\tau,\epsilon} = \tau \mathbf{s}_{\text{future}} + (1-\tau)\epsilon$，使用 4K 小时真实机器人操作数据训练。

#### Stage 2：世界模型条件化策略训练

从预训练 GigaBrain-0.5 checkpoint 初始化策略，加入世界模型的两种条件输入：

1. **未来状态 token** $\mathbf{z}_{\text{future}}$：经轻量 MLP 投影对齐维度
2. **价值估计** $v_t$：通过 n-step TD 计算优势并二值化

**n-step 优势计算**：

$$A(\mathbf{s}_t, a_t) = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n v_{t+n} - v_t$$

$$I = \mathbb{1}(A(\mathbf{s}_t, a_t) > \epsilon)$$

**防止过度依赖世界模型的两个策略**：

- **单步去噪推理**：世界模型推理时只做 1 步去噪，最小化计算开销
- **随机注意力遮蔽**：训练时以 $p=0.2$ 的概率随机遮蔽世界模型 token，强制策略在世界模型输入缺失时仍能正常工作

这使得部署时有两种模式：高效模式（跳过世界模型）和标准模式（使用世界模型引导）。

#### Stage 3：Human-in-the-Loop Rollout (HILR) 数据收集

部署策略收集混合数据：

- **自主 rollout**：机器人独立执行，动作分布更接近策略的原生分布（比纯遥操作的分布差距小）
- **专家干预**：人类在机器人即将失败时接管纠正

工程细节：开发了专用的 HILR 数据收集软件，自动检测和移除人工干预边界处的时序不连续伪影，确保轨迹数据的时间连贯性。

#### Stage 4：Rollout 数据持续训练

用 HILR 数据集微调策略和世界模型：

- **世界模型**：用 HILR 数据 + 基础数据联合训练，防止优势坍缩到零（$A(\mathbf{s}_t, a_t) \approx 0$）
- **策略**：保持 $p=0.2$ 的随机遮蔽，对 $I$ 和 $\mathbf{z}_{\text{future}}$ 都做 dropout

整个 rollout → 标注 → 训练的循环可迭代执行：策略变强 → rollout 数据质量提升 → 训练效果更好。

### 3.4 推理配置

- 优势指标固定 $I=1$（乐观策略——总是让策略给出它认为最好的动作）
- 两种模式：
  - **高效模式**：遮蔽世界模型 token，纯当前观测推理
  - **标准模式**：世界模型生成 $\mathbf{z}$，策略利用未来状态引导长时域规划

---

## 四、实验结果

### 4.1 GigaBrain-0.5 基础模型性能

**预训练配置**：10,000+ 小时数据（6,000+ 小时世界模型生成 + 4,000+ 小时真实机器人），batch size 3072，100K 步，FSDP v2 训练。

**内部评估**（8 个任务，对比 π₀、GigaBrain-0、π₀.₅）：

| 任务 | GigaBrain-0.5 vs π₀.₅ |
| --- | --- |
| Juice Preparation | 100%（超 GigaBrain-0 的 90%） |
| Paper Towel Preparation | 超 π₀.₅ 约 15% |
| Laundry Folding | 超 π₀.₅ 约 5% |
| Laundry Collection | 超 π₀.₅ 约 10% |
| Box Packing | 超 π₀.₅ 约 10% |
| Espresso Preparation | 超 π₀.₅ 约 20% |

所有 8 个任务上 GigaBrain-0.5 均取得最高成功率。

**RoboChallenge 公开基准**：中间版本 GigaBrain-0.1 以 51.67% 平均成功率排名第一（截至 2026.02.09），比 π₀.₅ 的 42.67% 高出 9%。

### 4.2 价值预测方法对比

| 方法 | 推理时间 (s) | MAE ↓ | MSE ↓ | RMSE ↓ | Kendall ↑ |
| --- | --- | --- | --- | --- | --- |
| VLM-based（π₀.₆* 方式） | 0.32 | 0.0683 | 0.0106 | 0.1029 | 0.7972 |
| WM-based（仅价值） | 0.11 | 0.0838 | 0.0236 | 0.1433 | 0.7288 |
| **WM-based（状态+价值）** | **0.25** | **0.0621** | **0.0099** | **0.0989** | **0.8018** |

关键发现：

- VLM-based 方法受 SigLIP 编码器拖累，推理最慢（0.32s）
- 仅预测价值的世界模型最快（0.11s）但精度最差——说明价值预测需要未来状态的上下文信息
- **联合预测状态+价值**在精度和速度间取得最优平衡

### 4.3 世界模型条件化的多任务泛化

在 4 个任务上对比有/无世界模型条件，分别测试单任务和多任务训练：

- **单任务**：世界模型条件化在所有任务和训练步数上一致优于基线
- **多任务**：性能差距随训练步数增加**不断扩大**，在 Box Packing 20K 步时差距达 ~30%

这表明世界模型条件有效促进了跨任务知识迁移。

### 4.4 RL 方法对比（RAMP 的核心实验）

在三个高难度任务上对比 RAMP 与 AWR、RECAP：

| 方法 | Box Packing | Espresso Preparation | Laundry Folding |
| --- | --- | --- | --- |
| GigaBrain-0.5 Pretrain | ~60% | ~50% | ~70% |
| GigaBrain-0.5 + AWR | ~65% | ~55% | ~75% |
| GigaBrain-0.5 + RECAP | ~65% | ~60% | ~70% |
| **GigaBrain-0.5 + RAMP (Ours)** | **~95%** | **~90%** | **~95%** |

RAMP 在 Box Packing 和 Espresso Preparation 上比 RECAP 提升约 **30%**，在所有三个任务上接近满分。

---

## 五、局限性与未来方向

### 5.1 世界模型的计算开销

Wan2.2 作为世界模型骨干虽然提供了强大的时空推理能力，但训练和推理成本仍然较高。虽然推理时只做单步去噪，但完整训练世界模型需要大量 GPU 资源。

### 5.2 Human-in-the-Loop 的依赖

RAMP 的 Stage 3 仍需人类干预来收集高质量 rollout 数据。如何进一步减少对人工监督的依赖，实现更自主的数据收集闭环，是重要的未来方向。

### 5.3 世界模型生成质量的天花板

世界模型预测的未来状态不可能完美——特别是在罕见场景或极端物理交互下。策略对不准确的 $\mathbf{z}$ 的鲁棒性（通过 20% 随机遮蔽部分解决）仍有改进空间。

### 5.4 论文自述的未来方向

- 更高效地利用 model rollout 数据，最大化合成轨迹的信息价值
- 探索更可扩展的自进化范式：自主数据策展、策略精调、世界模型更新的闭环交互

---

## 六、个人思考

### 6.1 RAMP vs RECAP vs RISE：三条世界模型 + RL 路线

这三篇论文代表了"世界模型辅助 VLA RL 训练"的三种不同思路：

| 维度 | RECAP (π₀.₆*) | RISE | RAMP (GigaBrain) |
| --- | --- | --- | --- |
| 世界模型角色 | 无世界模型，VLM 做价值预测 | 组合式（动力学 + 价值分离） | 统一视频世界模型（状态 + 价值联合） |
| RL 信号 | 二值优势 $I$（1 bit） | 多 bin 优势（~3.3 bits） | 二值优势 $I$ + 未来状态 $\mathbf{z}$（高维） |
| 策略条件输入 | $(\mathbf{o}, I, l)$ | $(\mathbf{o}, A_{\text{bin}}, l)$ | $(\mathbf{o}, \mathbf{z}, I, l)$ |
| 数据来源 | 真实世界 rollout + 专家干预 | 想象空间 rollout（零真实交互） | 真实 HILR rollout + 迭代训练 |
| 世界模型骨干 | — | Genie Envisioner (LTX-Video) | Wan2.2 (DiT) |
| 推理时世界模型 | 不使用 | 不使用 | 可选（高效模式不用） |

RAMP 的独特价值在于：它不仅用世界模型计算优势信号，更关键的是将世界模型的**未来状态预测**直接注入策略。这在理论上（信息增益）和实验上（~30% 提升）都显著优于只用优势信号的 RECAP。

### 6.2 "条件化"范式的持续进化

从 RECAP 的优势条件化，到 RAMP 的优势 + 未来状态条件化，我们看到了一个清晰的趋势：**给策略更丰富的条件输入，比让策略"猜"更好**。

- RECAP：告诉策略"这个动作好不好"（1 bit）
- RAMP：告诉策略"这个动作好不好 + 做完会看到什么"（高维信息）
- 未来可能：告诉策略"做完会看到什么 + 可替代方案各自会怎样"（多样未来 + 对比信息）

### 6.3 随机遮蔽的工程智慧

训练时以 20% 概率遮蔽世界模型 token 是一个精巧的设计，一石三鸟：

1. **防止过拟合**：策略不会过度依赖可能不准确的世界模型预测
2. **灵活推理**：部署时可以按需选择是否启用世界模型
3. **训练-推理一致性**：无论高效模式还是标准模式，策略在训练时都见过对应的输入模式

这个 trick 类似于 NLP 中的 token dropout 和扩散模型的 classifier-free guidance dropout，但应用在世界模型条件化的场景下效果独到。

### 6.4 与 WMPO 的对比

WMPO 在隐空间做 PPO（策略梯度），RAMP 在观测空间做优势条件化（监督学习式）。RAMP 的优势在于：

- 避免了在 flow matching 模型上做策略梯度的不稳定性
- 世界模型预测的未来状态直接可视化可解释（视频帧），而非 WMPO 的 MLP 隐表征
- 支持迭代改进（HILR 闭环），而 WMPO 是纯离线方案

---

## 参考

- **π₀.₆* / RECAP (2025)**：优势条件化离线 RL 框架，RAMP 的直接前身和对比基线
- **GigaBrain-0 (2025)**：GigaBrain 系列 VLA 基础架构，GigaBrain-0.5 的前代
- **RISE (2026)**：组合式世界模型 + 想象空间 RL，与 RAMP 互补的世界模型 + VLA 路线
- **Wan2.2 (Wang et al., 2025)**：RAMP 世界模型的骨干架构
- **WMPO (2025)**：隐空间世界模型 RL，另一种 model-based VLA RL 方案
- **AWR (Peng et al., 2019)**：优势加权回归，RAMP 的对比基线之一
