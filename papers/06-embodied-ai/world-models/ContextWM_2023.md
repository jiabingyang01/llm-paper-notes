# ContextWM：用野外视频预训练上下文化世界模型以服务强化学习

> **论文**：*Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning*
>
> **作者**：Jialong Wu\*, Haoyu Ma\*, Chaoyi Deng, Mingsheng Long（\* 共同一作，Mingsheng Long 为通讯）
>
> **机构**：清华大学软件学院、北京信息科学与技术国家研究中心（BNRist），中国
>
> **发布时间**：2023 年 05 月（arXiv 2305.18499，v2 于 2023 年 10 月）
>
> **发表状态**：NeurIPS 2023（第 37 届 Conference on Neural Information Processing Systems）
>
> 🔗 [arXiv](https://arxiv.org/abs/2305.18499) | [PDF](https://arxiv.org/pdf/2305.18499)
>
> **分类标签**：`world-model` `video-pretraining` `model-based-RL` `context-dynamics-disentangle`

---

## 一句话总结

针对"野外视频背景/纹理繁杂、上下文与动力学纠缠"导致世界模型难以迁移的问题，ContextWM 用一个 context 编码器把时不变的上下文（纹理、形状、颜色）与时变的动力学（位置、布局、运动）显式解耦——图像解码器同时条件于潜在动力学 $z_t$ 和上下文帧 $c$，而潜在动力学的推断刻意**不看** $c$，逼迫 $z_t$ 只编码本质时序变化；在 SSv2 等野外视频上预训练后微调，能显著提升 Meta-world、DMControl Remastered、CARLA 三域 MBRL 的样本效率（Meta-world 六任务中五个超过 plain WM）。

## 一、问题与动机

- **MBRL 仍在从零学（tabula rasa）**。世界模型（world model）用来近似环境的状态转移与奖励，从而在"想象"中做规划或行为学习，是提升视觉控制样本效率的核心。但预训练-微调范式在 CV/NLP 已大获成功，MBRL 却仍以从头训练为主，世界模型的准确性与泛化性受限。
- **已有视频预训练局限于领域内/仿真数据**。APV（Action-free Pre-training from Videos，本文最直接的前作）首次尝试跨域预训练世界模型，但其数据是 RLBench 里脚本策略生成的仿真演示，缺乏多样性与规模。此前直接用真实野外视频的尝试要么欠拟合、要么对下游收益微乎其微。
- **野外视频的复杂性会"吃掉"模型容量**。互联网视频天然带有复杂背景、外观、形状以及纠缠的动力学。若粗糙地把上下文与动力学一起建模，模型会把大量容量浪费在刻画"画面里有什么（what is there）"的低层视觉细节上，从而无法捕捉"正在发生什么（what is happening）"这一可跨场景共享的知识。作者受生物视觉系统启发（视网膜神经节约 80% 的 P 型细胞处理空间细节/颜色、约 20% 的 M 型细胞处理时序变化），主张要把上下文与动力学分开建模。
- **核心问题**：*能否在多样的野外视频上预训练世界模型，从而让下游视觉控制任务学得更快？* 作者把这一范式命名为 **IPV（In-the-wild Pre-training from Videos）**，并提出配套的 **ContextWM** 来落地。

## 二、核心方法

### 2.1 背景：Dreamer 的潜在动力学模型

视觉控制被形式化为 POMDP $\langle \mathcal{O}, \mathcal{A}, p, r, \gamma \rangle$。Dreamer 系列把世界模型写成含四个组件的潜在动力学模型（RSSM）：表示模型 $z_t \sim q_\theta(z_t \mid z_{t-1}, a_{t-1}, o_t)$（后验）、转移模型 $\hat{z}_t \sim p_\theta(\hat{z}_t \mid z_{t-1}, a_{t-1})$（先验）、图像解码器 $\hat{o}_t \sim p_\theta(\hat{o}_t \mid z_t)$、奖励预测器 $\hat{r}_t \sim p_\theta(\hat{r}_t \mid z_t)$，通过最小化 ELBO（负变分下界）联合训练。APV 则先训一个去掉动作条件与奖励的 action-free 变体做视频预测，再在其上"堆叠"一个 action-conditioned 模型微调。

### 2.2 上下文化潜在动力学模型（Contextualized latent dynamics）

核心直觉：观测序列里有两组信息——**时不变的 context $c$**（纹理、形状、颜色等静态概念）与**时变的 dynamics $z_t$**（位置、布局、运动等时序转移）。ContextWM 让**图像解码器**同时条件于二者：

$$
p_\theta(o_t \mid z_t, c)
$$

但对潜在动力学的**变分推断刻意做成 context-unaware**（不引用 $c$）：

$$
q_\theta(z_t \mid z_{t-1}, a_{t-1}, o_t)
$$

> 用大白话说：解码器"作弊"地拿到一张干净的上下文帧 $c$ 去还原纹理背景，于是 $z_t$ 就没必要再去记这些静态细节了；而推断 $z_t$ 时又不给它看 $c$，形成一个信息瓶颈，逼着 $z_t$ 只保留"运动/变化"这类本质时序信息，并把它学到一个高层语义空间里（而不是仅仅在低层像素上和上下文帧作差）。这样学到的表示更可迁移、对繁杂上下文更鲁棒。

其 ELBO（对条件对数似然 $\ln p_\theta(o_{1:T}, r_{1:T} \mid a_{1:T}, c)$，无需建模复杂的 $p(c)$）为：

$$
\mathcal{L}(\theta) \doteq \mathbb{E}_{q_\theta}\Big[\sum_{t=1}^{T}\big(-\ln p_\theta(o_t \mid z_t, c) - \ln p_\theta(r_t \mid z_t) + \beta_z\,\mathrm{KL}\big[q_\theta(z_t \mid z_{t-1}, a_{t-1}, o_t)\,\big\|\,p_\theta(\hat{z}_t \mid z_{t-1}, a_{t-1})\big]\big)\Big]
$$

### 2.3 具体架构

- **上下文的最简选择**：为端到端视觉控制，$c$ 直接取轨迹片段中随机采样的一帧观测 $c \doteq o_{\tilde{t}},\ \tilde{t} \sim \mathrm{Uniform}\{1, 2, \dots, T\}$。假设上下文信息在各帧中均匀存在，随机选帧能让 context 编码器对时序变化鲁棒。
- **多尺度 cross-attention 条件化**：context 编码器用 U-Net 结构，把多尺度上下文特征通过**交叉注意力**注入解码器。作者指出，传统 U-Net 用 concat/相加会强制上下文与重建之间的空间对齐，从而无法处理运动、形变；而 cross-attention 允许跨空间位置的灵活"信息捷径"。设解码器特征 $X \in \mathbb{R}^{c\times h\times w}$、上下文特征 $Z \in \mathbb{R}^{c\times h\times w}$：

$$
Q = \mathrm{Reshape}(X),\quad K = V = \mathrm{Reshape}(Z)
$$

$$
R = \mathrm{Attention}(QW^Q, KW^K, VW^V),\quad X = \mathrm{ReLU}\big(X + \mathrm{BatchNorm}(\mathrm{Reshape}(R))\big)
$$

> 用大白话说：让解码器每个位置去"查询"整张上下文帧里最相关的纹理，而不是只对着正下方那一块像素；这样即便物体移动/变形了，也能从上下文帧的别处把外观搬过来。

- **双奖励预测器（dual reward predictors）**：只把潜在变量 $s_t$ 喂给 actor/critic 会带来一个隐患——$s_t$ 可能通过 U-Net 的跳连"抄近路"，导致它没编码到任务相关信息（如静态目标物体的位置只落在时不变上下文 $c$ 里）；且视频内在探索奖励 $r_t^{\mathrm{int}}$ 会在训练中漂移、扭曲被回归的奖励。于是设两个奖励头：**behavioral** 头回归探索性奖励 $r_t + \lambda r_t^{\mathrm{int}}$ 用于行为学习，**representative** 头回归纯奖励 $r_t$ 用于强化任务相关表示学习。
- **总目标（微调，Eq. 6）**：在 action-free 潜在动力学之上堆叠 action-conditioned 模型，损失包含 context-unaware 潜在推断、上下文化图像损失、behavioral 奖励损失、representative 奖励损失、action-free KL 与 action-conditional KL。预训练时则去掉动作条件与奖励项。行为学习沿用 DreamerV2 的 actor-critic 想象训练。
- 实现细节：新超参 $\beta_r = 1.0$；视觉编码器/解码器为 13 层 ResNet；其余超参与 APV 一致。

## 三、实验结果

**评测设置**：三大视觉控制域——Meta-world（50 任务基准，选与 APV 相同的 6 个操作任务）、DMControl Remastered（DMCR，在 DMControl 上加入复杂图形多样性的运动控制）、CARLA（Town04 高速公路，1000 步内尽量无碰撞地行驶，多种天气）。预训练数据：SSv2（Something-Something-v2，19.3 万条人-物交互视频）、Human3.6M（4 视角、300 万+ 人体姿态）、YouTube Driving（134 条真实驾驶视频、120+ 小时），以及三者合并的 assembled 数据。每任务 8 次独立运行，报告 IQM（四分位均值）+ bootstrap 置信区间。基线：DreamerV2（从零 plain WM）、IPV w/ Plain WM（即 APV，改用野外视频预训练）、IPV w/ ContextWM。

| 域 / 分析 | 关键发现 |
|---|---|
| Meta-world（SSv2 预训练） | IPV 一致地提升样本效率与最终性能；ContextWM 在六任务中**五个超过 plain WM**；Dial Turn 极难，两法都基本无法解 |
| Meta-world 消融（48 runs / 6 任务聚合） | **plain WM 几乎不从预训练获益**（其微弱增益主要来自内在探索奖励），而 ContextWM 借预训练显著提升；去掉 cross-attention（换成 concat）或去掉双奖励头，性能均下降——两组件都有贡献 |
| 数据规模（Meta-world） | 从 SSv2 抽 1.5k / 15k 视频预训练，几乎可匹配全量——因 SSv2 只有 174 类人-物交互，动力学多样性有限，少量数据即可学好 |
| 数据领域（Meta-world） | 更相似域的 RLBench（仿真）预训练优于 SSv2，但仿真缺多样性与规模；assembled 数据能持续改进，是可扩展的替代方案 |
| DMCR（SSv2 预训练） | 即便存在巨大域差，SSv2 预训练仍**显著增强** ContextWM，说明其能有效迁移"分离上下文/动力学"的共享知识；plain WM 也受益但在个别任务（Hopper Stand）挣扎；ContextWM 即使从零训练也具竞争力 |
| DMCR（Human3.6M 预训练） | 出现**负迁移**——Human3.6M 采自实验室环境而非真正"野外"，缺多样性的数据难以帮上世界模型 |
| CARLA（SSv2 预训练） | IPV+ContextWM 在训练早期学得更快、几乎所有天气下最终性能也优于 plain WM；但对 plain WM 优势较小——单帧上下文对驾驶场景可能信息不足 |
| CARLA（数据领域） | YouTube Driving 或 assembled 均优于从零，但**都未显著超过 SSv2 预训练**，尽管 YouTube Driving 与 CARLA 域差更小；推测 YouTube Driving 复杂度更高 |

**定性分析**：(1) 视频预测——ContextWM 能有效捕捉物体形状与运动，plain WM 失败；cross-attention 确能关注上下文帧中变化的空间位置。(2) t-SNE——把"从右往左推/从左往右推"两类视频（预训练**未用任何标签**）的平均池化表示可视化，ContextWM 能按运动方向清晰分开，plain WM 则纠缠。(3) 组合式解码——用另一条轨迹的随机帧替换原上下文、保持动力学不变，ContextWM 能正确地把**新上下文 + 原动力学**组合出来，证明其学到了解耦表示，plain WM 因表示纠缠而预测糟糕。

## 四、局限性

- **单帧上下文信息不足**：随机选的单帧未必能覆盖真实场景（如自动驾驶）的完整上下文，导致在 CARLA 上相对 plain WM 优势有限；需选/融合多帧或多模态上下文。
- **中等规模**：世界模型与预训练数据都还是中等规模，限制了可获得的、广泛适用的知识；缺少针对预训练数据规模的清晰 scaling 规律（作者观察到 SSv2 因动力学多样性不足而"饱和"得早）。
- **生成式目标低效**：以图像重建为目标会把模型容量低效地用在还原繁杂上下文上；用对比学习或 self-prediction 等替代目标，或许能去掉沉重的上下文建模、更专注动力学。
- **数据依赖"真野外"**：Human3.6M 的负迁移说明并非任何视频都有用，实验室采集的低多样性数据可能有害。

## 五、评价与展望

**优点**。(1) 问题切入准：把"野外视频难迁移"的症结归结为上下文/动力学纠缠，并给出一个概率图模型上干净的解耦方案——解码器条件于 $(z_t, c)$、推断 $z_t$ 时 context-unaware 形成信息瓶颈，机制上比单纯堆数据更有说服力。(2) cross-attention 条件化相较 U-Net 直接 concat，恰当地解除了"空间对齐"约束，是处理运动/形变的关键工程点。(3) 三域（操作、运动、驾驶）+ 多预训练数据 + IQM/CI 的评测较扎实，组合式解码与 t-SNE 的可视化对"解耦"的论证很直观。(4) 相较前作 APV 从仿真脚本演示预训练，本文真正打通了"用互联网真实视频预训练世界模型"这条路径。

**不足与开放问题**。(1) 报告以学习曲线为主、缺少统一的最终成功率/回报数值表，跨任务定量比较不够精细。(2) 相对 plain WM 的增益在驾驶域偏小，且 SSv2 早早饱和，说明当前范式尚未展现清晰的正向 scaling——这与同期 LLM 的 scaling 叙事形成反差，是最值得追问的开放问题：换用 Transformer 世界模型（如 TransDreamer/TWM 一系）+ Ego4D 级海量视频，解耦设计能否解锁真正的规模效应？(3) context 只取"单帧"，对长时上下文（光照渐变、多物体场景）表达力有限；多帧/结构化/文本条件的上下文是自然的扩展。(4) 与同期 SWIM 相比，本文不引入视觉 affordance 之类的强结构先验，因而适用视频与下游任务范围更广，但也放弃了人-机器人动作空间对齐带来的直接可操作性——两条路线各有取舍。(5) 生成式重建 vs. 对比/自预测目标之争，作者自己也在展望中点到，是后续减负、聚焦动力学的可行方向。

总体上，ContextWM 是"视频预训练世界模型"这一子方向里机制清晰、实验完整的代表作，其"上下文-动力学解耦 + context-unaware 瓶颈"的思想对后续可扩展世界模型仍有借鉴意义。

## 参考

1. Seo et al. *Reinforcement Learning with Action-Free Pre-training from Videos (APV)*, ICML 2022 —— 本文最直接的前作，首个跨域预训练世界模型（仿真演示）。
2. Hafner et al. *Mastering Atari with Discrete World Models (DreamerV2)*, ICLR 2021 —— 行为学习与 RSSM 骨干。
3. Hafner et al. *Dream to Control (Dreamer)*, ICLR 2020 —— 潜在想象训练的基础框架。
4. Goyal et al. *The "Something Something" Video Database*, ICCV 2017 —— 主力预训练数据集 SSv2。
5. Denton & Birodkar. *Unsupervised Learning of Disentangled Representations from Video*, NeurIPS 2017 —— 视频中上下文/动力学解耦表示的思想来源之一。
