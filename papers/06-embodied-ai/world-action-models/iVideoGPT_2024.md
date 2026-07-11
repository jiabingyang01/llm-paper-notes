# iVideoGPT：可交互 VideoGPT 是可扩展的世界模型

> **论文**：*iVideoGPT: Interactive VideoGPTs are Scalable World Models*
>
> **作者**：Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, Mingsheng Long（通讯作者）
>
> **机构**：清华大学软件学院 / BNRist、清华大学致理书院、华为诺亚方舟实验室、天津大学智能与计算学部
>
> **发布时间**：2024 年 05 月（arXiv 2405.15223）
>
> **发表状态**：NeurIPS 2024（已发表）
>
> 🔗 [arXiv](https://arxiv.org/abs/2405.15223) | [PDF](https://arxiv.org/pdf/2405.15223)
>
> **分类标签**：`交互式视频预测` `压缩tokenization` `自回归世界模型` `基于模型的强化学习`

---

## 一句话总结

iVideoGPT 用条件 VQGAN 把未来帧压缩到仅 4×4 个 token（渐近约 16 倍压缩），在 140 万条 Open X-Embodiment + Something-Something v2 的人机操作轨迹上预训练一个 GPT-2 规模的自回归 transformer，从而在保持逐帧交互性的同时获得可扩展性；下游实验中它在 RoboNet 256×256 动作条件预测上以 FVD 197.9 大幅超越 MaskViT（211.7），并首次成功把 MBPO 应用到视觉连续控制任务，在 Meta-World 上样本效率显著优于无模型基线、媲美或超过 DreamerV3。

## 一、问题与动机

世界模型的理想形态需要同时具备两个属性：**交互性**（每一步都能根据智能体刚采取的动作条件生成下一步观测，供 MPC/规划/RL 逐步介入）与**可扩展性**（能像 LLM 一样在海量、多样的互联网/机器人视频上做无监督预训练）。但现有两条技术路线各有短板：

- **基于模型的 RL 中的循环世界模型**（Dreamer、MuZero 等）天然支持逐步的动作条件转移，交互性好，但主要在游戏/仿真等简单视觉域验证，循环架构难以扩展到大规模真实世界数据（论文用 DreamerV3-XL 在同一预训练数据上做了对照实验加以验证，见第三节）。
- **互联网规模视频生成模型**（VideoGPT、Stable Video Diffusion 等）可以合成逼真的长视频，但多用非因果的时空联合压缩模块（如 3D 卷积），只能在生成开始时一次性给定整段动作序列，是"轨迹级"而非"步级"交互，无法满足智能体在仿真中逐步试错所需的细粒度介入。

论文提出的核心问题是："如何利用可扩展视频生成模型的进展，构建既交互又可扩展的世界模型？"由此设计了 iVideoGPT——一个把视觉观测、动作、奖励等多模态信号统一编码为 token 序列、以 next-token prediction 逐帧自回归生成的 GPT 式框架。

## 二、核心方法

iVideoGPT 由两部分组成：**压缩式 tokenizer**（把高维视频帧离散化为少量 token）+ **自回归 transformer**（对 token 序列做下一个 token 预测）。

### 2.1 压缩式 tokenization（context-dynamics 解耦）

标准做法是用图像 tokenizer（如 VQGAN）逐帧独立离散化，每帧产生 $N=16\times16$ 个 token，序列长度随帧数线性增长，交互性虽好但序列很快爆炸。iVideoGPT 反其道而行：设计一个**条件 VQGAN**，包含两套编码器-解码器：context 编解码器 $(E_c, D_c)$ 和 prediction 编解码器 $(E_p, D_p)$。

- 初始的 $T_0$ 帧上下文帧信息丰富，独立编码为完整的 $N$ 个 token：$z_t^{(1:N)} = E_c(o_t),\ \hat o_t = D_c(z_t)$，$t=1,\dots,T_0$。
- 由于视频前后帧存在大量时间冗余，未来帧只需编码"动态"信息（物体位置姿态的变化），因此用条件编解码器压到更少的 $n=4\times4$ 个 token：

$$z_t^{(1:n)} = E_p(o_t \mid o_{1:T_0}), \quad \hat o_t = D_p(z_t \mid o_{1:T_0}) \quad \text{for } t = T_0+1,\dots,T$$

**用大白话说**：不需要每帧都从头"画"一张完整的图，只要给模型看过一次"这个场景长什么样"（context 帧），后面每帧只需要告诉它"手臂往哪动了、物体挪到哪了"这类差量信息即可，大部分静态背景直接靠 cross-attention 从 context 特征里"抄"过来。

条件机制通过 context 编码器与 prediction 编码/解码器之间的**多尺度 cross-attention**实现（类似作者此前工作 ContextWM）：在每个下采样/上采样层级，prediction 分支的特征图 $F_p^l$ 通过 cross-attention 从 context 分支的同层特征图 $F_c^l$ 中检索上下文信息再残差融合，仅在特征图分辨率不超过阈值（如 64×64 分辨率下的 16×16）时启用以控制显存。

tokenizer 联合训练目标：

$$\mathcal{L}_{\text{tokenizer}} = \sum_{t=1}^{T_0} \mathcal{L}_{\text{VQGAN}}\big(o_t; E_c(\cdot), D_c(\cdot)\big) + \sum_{t=T_0+1}^{T} \mathcal{L}_{\text{VQGAN}}\big(o_t; E_p(\cdot \mid o_{1:T_0}), D_p(\cdot \mid o_{1:T_0})\big)$$

其中 $\mathcal{L}_{\text{VQGAN}}$ 是 $L_1$ 重建损失、VQ commitment 损失、感知损失（可选对抗损失）的组合。这一设计带来渐近 **16×** 的 token 序列长度缩减（$N=256$ 对比 $n=16$），既加速训练/rollout，也因为把"画背景"这件事从"建模动态"中解耦出来，让 transformer 能更专注地建模物体运动，连带提升了视频质量。

### 2.2 多模态自回归 transformer

Tokenization 后视频被展平为序列

$$x = \big(z_1^{(1)},\dots,z_1^{(N)}, \texttt{[S1]},\ z_2^{(1)},\dots,z_2^{(N)},\dots,\texttt{[S1]},\ z_{T_0+1}^{(1)},\dots,z_{T_0+1}^{(n)},\dots\big)$$

context 帧与未来帧不共享 token id（词表共 16,386 个：8,192 个 context token + 8,192 个 prediction token + 2 个 slot token），插入槽位 token `[S1]`/`[S2]` 划定帧边界，并用于挂载动作/奖励等低维模态：动作通过线性投影后加到槽位 token embedding 上；奖励预测则在每帧最后一个 token 的隐状态上接一个线性头，与视频预测损失联合训练（多任务学习提升控制相关信息的建模精度）。骨干沿用 LLaMA 架构（RMSNorm 预归一化、SwiGLU、RoPE 旋转位置编码），大小对齐 GPT-2（12 层/768 维，138M 参数为主实验配置；另有 24 层/1024 维、436M 的规模化实验配置）。

预训练采用动作无关的 next-token cross-entropy 损失，只对未来帧 token 计算：

$$\mathcal{L}_{\text{pre-train}} = -\sum_{i=(N+1)T_0+1}^{L} \log p(x_i \mid x_{1:i-1})$$

context 帧本身不参与损失计算（模型不需要学会"生成"上下文，只需学会用它做条件）。

**用大白话说**：整套设计就是把视频、动作、奖励拼成一句"话"，用训练语言模型的方式做下一词预测；区别只在于"词表"是视觉 token，而且靠特殊设计把每帧要预测的"词数"从几百压到十几个，这样一步步生成才够快、够能连续几十步做规划 rollout。

### 2.3 预训练数据与下游适配

预训练混合 35 个 Open X-Embodiment（OXE）数据集与 Something-Something v2（SSv2），共 **1,417,954 条轨迹**（约 140 万条，Fractal/Bridge/BC-Z/RoboNet 各占 12.8%，Kuka 8.5%，SSv2 占 15% 权重，其余数十个小数据集各占 0.5%）。下游按 LLM 式两阶段范式适配：视频预测（微调 action-conditioned）、视觉规划（VP²）、视觉 MBRL（Meta-World，构建于 MBPO 之上）。此外验证了**只微调 tokenizer、冻结 transformer**即可零样本迁移到未见过的机器人夹爪类型（BAIR），支持"transformer 学到的是跨具身共享的动力学/物理常识，tokenizer 只需负责域特定的视觉外观对齐"这一假设。

## 三、实验结果

### 3.1 视频预测（BAIR / RoboNet，64×64 分辨率）

| 设置 | 方法 | FVD↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| BAIR action-free | MAGVIT | **62.0** | 19.3 | 78.7 | 12.3 |
| BAIR action-free | **iVideoGPT** | 75.0±0.20 | **20.4**±0.01 | **82.3**±0.05 | **9.5**±0.01 |
| BAIR action-conditioned | MaskViT | 70.5 | - | - | - |
| BAIR action-conditioned | **iVideoGPT** | **60.8**±0.08 | 24.5±0.01 | 90.2±0.03 | 5.0±0.01 |
| RoboNet action-conditioned | FitVid | **62.5** | **28.2** | 89.3 | 2.4 |
| RoboNet action-conditioned | **iVideoGPT** | 63.2±0.01 | 27.8±0.01 | **90.6**±0.02 | 4.9±0.00 |
| RoboNet 256×256 action-conditioned | MaskViT | 211.7 | 20.4 | 67.1 | 17.0 |
| RoboNet 256×256 action-conditioned | **iVideoGPT** | **197.9**±0.66 | **23.8**±0.00 | **80.8**±0.01 | **14.7**±0.01 |

在低分辨率 BAIR/RoboNet 上 iVideoGPT 与 MAGVIT/FitVid 等专用 SOTA 大体持平（各有优劣），但在**高分辨率（256×256）RoboNet** 上以全部四项指标明显超越 MaskViT——作者指出 MaskViT 这类逐帧独立 tokenize 的方法存在时间不一致和闪烁伪影，而 iVideoGPT 的条件式压缩 tokenizer 天然保持上下文一致性。从动作无关切换到动作条件预训练可将 BAIR 的 FVD 改善约 20%。

### 3.2 视觉规划（VP² 基准，MPC）

在 4 个 Robosuite + 7 个 RoboDesk 任务上，iVideoGPT 在 Open Drawer（均值成功率 0.375，第一）、Red Button（0.9222，第一）等任务上大幅领先 FitVid/SVG'/MCVD/MaskViT/Struct-VRNN，整体（除 Flat Block 外）达到与最强基线 SVG' 相当的平均水平；但在 Open Slide 任务上表现欠佳（详见局限性）。

### 3.3 视觉 Model-based RL（Meta-World）

作者基于 MBPO 框架、以 iVideoGPT 作为世界模型、DrQ-v2 作为底层 actor-critic，在 Button Press Topdown Wall、Plate Slide、Hammer、Door Lock、Handle Pull Side、Coffee Push 六个 Meta-World 任务（共 30 组独立 run）上评测。结果显示：该 MBPO 变体不仅显著优于同架构的无模型基线（DrQ-v2）的样本效率，还达到甚至超过了 DreamerV3 的表现——论文称这是 **MBPO 首次成功应用于视觉连续控制任务**。作为对照，把 DreamerV3-XL（200M 参数，与 iVideoGPT 相当）在同一预训练数据上训练，生成的预测明显模糊、无法捕捉真实机器人动力学，在下游 Meta-World 上也无法从这种预训练中获益——印证了"循环架构难以从大规模预训练中受益"的核心论点。

### 3.4 模型分析

- **零样本泛化**：预训练 transformer 不经微调即可在未见过的 BAIR 夹爪上预测出自然但外观不同的运动；仅微调 tokenizer（transformer 冻结）即可达到与全量微调相近的感知质量，支持 context-dynamics 解耦假设。
- **少样本适配**：BAIR 上仅用 1,000 条动作条件轨迹微调即可达到 FVD 82.3；预训练带来的收益在数据稀缺（100/1,000 条）时最显著，数据充足时增益减弱。
- **模型规模化**：138M → 436M 参数，预训练验证集 loss 持续下降且更大模型下降更快，与 LLM 缩放规律一致；但下游控制任务上的规模化收益本文未验证。
- **Context-dynamics 解耦验证**：去掉解码器对 context 帧的 cross-attention 后，仍能重建出正确的运动轨迹，但场景视觉细节几乎完全丢失——直接证明了压缩 tokenizer 确实把"上下文外观"和"动态信息"分离到了不同的 token 分支中。
- **Tokenization 效率**（RoboNet，40G A100 / RTX 4090）：训练速度上，4×4 逐帧 tokenizer 3.10 iter/s（10.6GB），标准 16×16 逐帧 tokenizer 直接 **OOM**，而 iVideoGPT 的条件式 tokenizer 2.62 iter/s（22.3GB，因训练时仍需编码完整 context）；生成阶段（batch=1）16×16 需要 22.5 秒/条，4×4 与 iVideoGPT 均仅需约 1.1 秒/条——生成速度上 iVideoGPT 与纯 4×4 相当，但重建质量（LPIPS 0.089）远优于纯 4×4（0.160），逼近标准 16×16（0.038）。
- **人类偏好研究**：在 BAIR action-free 上收集 9 名参与者共 386 条标注，iVideoGPT 相对 VideoGPT 胜率 50.7%（负 24.7%，平 24.6%），相对 MCVD 胜率 44.7%（负 25.2%，平 30.1%）。

## 四、局限性

论文第 6 节及附录 C.3 明确讨论了以下局限：

1. **预训练数据多样性有限**：公开机器人数据集（OXE 等）规模和多样性仍不及互联网视频，作者认为需要引入更大规模的人类视频（如 Ego4D）来进一步弥合人-机数据的鸿沟。
2. **模态覆盖不全**：当前架构虽支持插入动作/奖励等模态，但预训练本身是单视角、动作无关的，尚未纳入多视角观测、机器人本体感知（proprioception）状态。
3. **压缩 tokenizer 的上下文假设有边界**：其设计假设初始 context 帧足以为后续所有未来帧提供上下文，这对长视频、大幅相机运动的场景会失效（智能体常常需要展望数十步），作者提出关键帧提取（keyframe extraction）作为未来可能的缓解方向。
4. **规模化验证不完整**：模型规模只测到 436M 且仅在预训练验证 loss（perplexity）上验证了 scaling，更大规模模型对下游控制任务的收益本文未观测；下游控制实验均在视觉上较简单的仿真基准（Meta-World、VP²）中进行，未扩展到更复杂的真实机器人任务。
5. **VP² Open Slide 失败案例**：64×64 低分辨率下模型难以判别夹爪是否已接触滑动把手这类细粒度接触线索，导致 MPC 过度自信地选中"看似成功"但实际失败的动作序列；作者对比发现端到端像素空间模型（SVG'、FitVid）在此任务上明显优于包括 iVideoGPT 在内的两阶段（tokenize-then-predict）模型，暗示离散 tokenization 在需要捕捉细粒度接触事件的操作任务上存在信息损失的固有代价。
6. **VP² 基准自身缺陷**：其内置的成功判别器（学习得到的分类器）对分布外的预测帧鲁棒性不足，容易给低质量/不可能成功的轨迹打高分，这是基准本身而非 iVideoGPT 独有的问题，但会干扰跨方法比较的公平性。

## 五、评价与展望

**优点**：iVideoGPT 的核心贡献是给出了一个简洁但工程上很扎实的方案，用条件 VQGAN 的"context 与 dynamics 解耦"思路，在几乎不改变标准 GPT 式自回归框架的前提下，把逐帧交互性和大规模预训练的可扩展性同时做到位——这恰好补上了循环世界模型（Dreamer 系）和非因果视频生成模型（VideoGPT/Stable Video Diffusion）之间的空白。论文的实验设计也相当完整：不仅有标准视频预测指标对比，还专门用 DreamerV3-XL 做了"循环架构 vs 大规模预训练"的对照实验（结果是循环架构从预训练中几乎不获益），并且是文中明确指出的 **MBPO 首次在视觉连续控制上成功应用**的报道，这个结果本身对 model-based RL 社区有一定参考价值：意味着"世界模型只需作为即插即用的 rollout 生成器"这条更简单的路线（相较于 Dreamer 式紧耦合 latent imagination）在有强大预训练世界模型加持时也能奏效。代码和预训练模型均已开源，相较于同期未公开的 UniSim（扩散式）和 Genie（掩码生成式）具备可复现性优势。

**局限/开放问题**：其一，压缩比是靠"context 帧信息足够丰富"这一假设换来的，本质上是一种短时程假设，长视频/大幅度视角变化下的退化是一个尚未解决的问题，作者自己也只是提了 keyframe extraction 这一可能方向而未验证；这与之后一批工作（如引入记忆机制或滚动式重新条件化的世界模型）试图解决的问题相通。其二，论文的下游验证集中在相对"干净"的仿真/半仿真基准（Meta-World、VP² 的 Robosuite/RoboDesk、64×64/256×256 的 BAIR/RoboNet 离线数据），尚未展示在真实机器人闭环部署中的表现，这也是几乎所有这一代"视频世界模型"论文的共同短板。其三，两阶段（离散 tokenize + 自回归预测）架构在需要精细接触判断的任务（如 VP² 的 Open Slide）上明显弱于端到端像素空间方法，这提示离散视觉 token 化本身可能天然不适合对接触/力这类细粒度物理线索敏感的操作任务，后续工作是否需要在视觉 token 之外显式引入触觉/力信号或提高有效分辨率，是一个值得跟进的方向。整体而言，iVideoGPT 提供了一个足够简单、可复现、有开源权重的基线，为"用统一自回归 token 序列联合建模视觉观测/动作/奖励"这一范式在具身智能预训练中的可行性提供了较有说服力的证据。

## 参考

1. Bruce et al. *Genie: Generative Interactive Environments*. ICML 2024.
2. Yang et al. *Learning Interactive Real-World Simulators (UniSim)*. ICLR 2024.
3. Hafner et al. *Mastering Diverse Domains through World Models (DreamerV3)*. arXiv:2301.04104, 2023.
4. Wu et al. *Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning (ContextWM)*. NeurIPS 2023.
5. Micheli et al. *Transformers are Sample-Efficient World Models (IRIS)*. ICLR 2023.
