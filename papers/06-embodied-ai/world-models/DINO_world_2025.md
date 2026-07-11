# DINO-world：回到特征——DINO 作为视频世界模型的基础

> **论文**：*Back to the Features: DINO as a Foundation for Video World Models*
>
> **作者**：Federico Baldassarre, Marc Szafraniec, Basile Terver, Vasil Khalidov, Francisco Massa, Yann LeCun, Patrick Labatut, Maximilian Seitzer, Piotr Bojanowski et al.
>
> **机构**：Meta FAIR
>
> **发布时间**：2025 年 07 月（arXiv 2507.19468）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.19468) | [PDF](https://arxiv.org/pdf/2507.19468)
>
> **分类标签**：`world-model` `latent-prediction` `DINOv2` `action-conditioned-planning`

---

## 一句话总结

DINO-world 把视频世界模型建成"在**冻结的 DINOv2 特征空间**里做下一帧预测"的问题：用一个 ViT-g 规模、带 3-轴 RoPE 的 cross-attention 因果预测器，在约 6000 万条未经筛选的网络视频上自监督训练，在 VSPW 语义分割 0.5 秒预测上比最强生成式模型 COSMOS-12B 高 6.3 mIoU，并可通过加入零初始化 action block 微调成动作条件世界模型用于规划。

## 一、问题与动机

世界模型（world model）指"给定历史观测与动作，预测环境未来状态"的网络。作者指出当前主流路线的两难：

- **像素空间生成式路线**（SORA、COSMOS、Wan 等）视觉保真度高，但资源消耗惊人——COSMOS 训练耗费约 2200 万 GPU 小时、模型可达 12B 参数；而且大量算力被浪费在"风中每片叶子的精确运动"这类对下游任务无关的细节上。它们大多局限在自动驾驶、游戏等窄域。
- **联合训练编码器 + 预测器的表征路线**（如 V-JEPA）：把编码器和预测器一起优化，得到的特征适合视频摘要，却**不擅长做准确的未来预测与规划**。

作者的核心主张是把两件事解耦：**在一个冻结的视觉基础模型（DINOv2）的潜空间里，单独训练一个未来预测器**。这样做的好处：

1. **预训练与后训练分离**：无标注视频上学通用动力学，后续只需少量带动作标注的数据做微调，降低对昂贵标注数据的依赖；
2. **潜空间建模而非像素**：绕开了逐像素建模的难题，对多数下游任务足够；
3. **冻结编码器自带强语义/几何先验**，作为引导大幅降低训练复杂度，也避免了编码器-预测器联合训练的技术复杂性。

此外作者批评现有世界模型评测各说各话（有的只看生成质量、有的只看物理理解、有的只看智能体控制），因此本文设计了一套横跨"稠密预测 → 直觉物理 → 规划"的统一评测。

## 二、核心方法

### 2.1 状态定义：DINOv2 patch token 作为"世界状态"

视频是一串带时间戳的 RGB 帧 $\{(v_t, \tau_t)\}_{t=1}^{T}$（显式建模时间戳 $\tau$ 以支持可变帧率）。冻结的 DINOv2 编码器把每帧映射成 patch 特征张量 $x_t = \mathrm{ENCODER}(v_t) \in \mathbb{R}^{H\times W\times D}$。世界模型被形式化为如下映射（Eq. 1）：

$$(\mathbf{X}_{1:t}, \mathcal{T}_{1:t}, (\tau_{t'}, i', j')) \to x_{t',i',j'}$$

即：给定过去所有帧的特征 $\mathbf{X}_{1:t}$ 与时间戳 $\mathcal{T}_{1:t}$，以及一个"未来查询坐标"$(\tau_{t'}, i', j')$（未来某时刻、某空间位置），预测该处的 patch token。

**用大白话说**：世界状态不是像素，而是 DINOv2 眼中的"语义补丁"。模型学的是"这些语义补丁随时间怎么演化"，而不是"每个像素长什么样"。

### 2.2 预测器架构：cross-attention 因果解码器

预测器借鉴机器翻译/图像重建的思路，是 $N$ 个残差 pre-norm cross-attention 块的堆叠。要预测坐标 $(\tau_{t'}, i', j')$ 处的未来 token，先从一个可学习 embedding 初始化 query $q \in \mathbb{R}^{D'}$，每个块里 query 去 cross-attend 所有历史 patch token（key-value），再过 MLP：

$$q \leftarrow q + \mathrm{CrossAttn}\big(\mathrm{LN}(q),\ \{\,x_{t,i,j} \mid \tau_t < \tau_{t'}\,\}\big)$$

$$q \leftarrow q + \mathrm{MLP}(\mathrm{LN}(q))$$

最后一个块后用线性投影得到预测 token $\hat{x}_{t',i',j'} \in \mathbb{R}^{D}$。

**用大白话说**：不像标准 self-attention 让所有 token 互相看（那样很贵），这里让"未来查询"这个小 query 单向地去读历史特征，像翻译里 decoder 读 encoder 输出一样。集合里的条件 $\tau_t < \tau_{t'}$ 保证只能看过去、不能看未来（因果性）。

### 2.3 3-轴 RoPE 位置编码

由于 query 和 context 特征本身不携带位置信息，作者把 rotary position encoding（RoPE）注入注意力：**把 head 维度切成三份，分别编码时间、水平、垂直坐标**。空间坐标定义在 $[-1,+1]^2$ 网格上（这样改变输入分辨率不改变 patch 间相对距离，实现分辨率无关）；时间坐标用**以秒为单位的绝对时间戳**，使模型能区分高低帧率并外推到更长视频。RoPE 周期取在 $[10^{-2}, 10^2]$。

### 2.4 训练目标与可变 FPS

为便于并行，训练用 next-frame 预测目标（$t'=t+1$，teacher forcing）。给定 $T$ 帧，一次并行地为所有 $(i',j')$ 和 $t\in\{1,\dots,T-1\}$ 计算预测——通过堆叠 $(T-1)HW$ 个 query 并用**块-三角注意力掩码**保证时间因果。损失为 smooth L1（Eq. 4）：

$$\min_\theta\ \mathcal{L}\big(x_{t+1,i',j'},\ \mathrm{Predictor}_\theta(\mathbf{X}_{1:t}, \mathcal{T}_{1:t}, (\tau_{t+1}, i', j'))\big)$$

在**所有** $(T-1)HW$ 项上求和。作者强调这与 V-JEPA / DINO-Foresight 的关键区别：后者只在少量 mask token 上算损失，本文对全部 token 算损失。

**可变 FPS 采样**：若直接取连续 $T$ 帧，时间差 $\Delta\tau$ 会偏向短间隔，限制预测视野。作者改为：每条视频均匀采样 $T-1$ 个时间差 $\Delta\tau \sim U[\Delta\tau_{\min}, \Delta\tau_{\max}]$，累加得到 $T$ 个时间戳（加随机起点），再取最近帧及其真实时间戳。这样时间间隔分布均匀，训练出的模型可在任意 $\Delta\tau$ 上预测。

**用大白话说**：不让模型只见过"相邻两帧变化很小"的情况，而是有意识地喂给它"隔 0.1 秒""隔 1 秒"各种时间跨度，让它学会在不同时间尺度上外推。

### 2.5 动作条件微调：零初始化 action block

预训练完成后，为支持规划，在每个块的 MLP 之后插入一个 **action block**，用当前动作 $a_t$ 更新 query：

$$q \leftarrow q + \mathrm{MLP}(\mathrm{LN}([q, a_t]))$$

这些 action block **初始化为恒等映射**，在小规模动作轨迹数据上训练。可选地**冻结整个基座、只训 action block**——既缓解过拟合，又能让同一基座复用到不同任务。作者明确对比 DINO-WM 的做法（把动作 token 交织进 patch token 序列）：那样会使 batching/masking 复杂化、需要额外容量、且被迫全量微调，可能破坏已学到的视频理解。

### 2.6 实现规格

- 编码器：DINOv2 ViT-B/14（带 register），取最后一层 $D=768$（与 DINO-Foresight 同款）。
- 预测器：$N=40$ 块、$D'=1536$、24 头（ViT-g 规模，但用 cross-attention），约 1.1B 参数。
- 训练：AdamW，300k 步、batch 1024 clips、$T=8$、分辨率 $224\times224$；再 50k 步在 $448\times448$ 上；warmup 后 LR 恒为 $10^{-4}$。
- 数据：约 6600 万条未筛选网络视频，时长 5–60 秒、帧率各异。

## 三、实验结果

### 3.1 稠密特征预测（Table 1）

在当前帧上训练分割/深度线性头，再把它应用到世界模型预测的未来特征上。short≈200ms，mid≈0.5s。"Present"是当前帧上界，"Copy Last"是复制最后一帧的平凡基线。

| 方法 | 编码器 | VSPW mIoU↑ Short/Mid | Cityscapes mIoU↑ Short/Mid | KITTI RMSE↓ Short/Mid |
|---|---|---|---|---|
| Copy Last | ViT-B | 47.9 / 42.1 | 53.2 / 39.7 | 3.778 / 4.745 |
| COSMOS-4B | ViT-B | 46.6 / 40.2 | 55.4 / 46.2 | 4.178 / 4.742 |
| COSMOS-12B | ViT-B | 46.6 / 40.7 | 55.6 / 45.9 | 4.157 / 4.617 |
| V-JEPA | ViT-H | 4.9 / 4.6 | 13.3 / 12.2 | 5.458 / 5.785 |
| DINO-Foresight | ViT-B | 44.7 / 37.7 | 64.5 / **57.2** | 3.562 / **3.740** |
| **DINO-world** | ViT-B | **51.6 / 47.0** | **64.7** / 55.1 | **3.214** / 4.268 |

- VSPW 上 DINO-world 全面领先，mid（0.5s）达 47.0，比最强生成式模型 COSMOS-12B 的 40.7 高 **6.3** mIoU。
- V-JEPA 表现极差（因其预测器只是训编码器的"头"，并非为准确未来预测设计）。
- DINO-Foresight 在 Cityscapes/KITTI 上略优于 DINO-world，作者归因于它专门在驾驶域 Cityscapes 上训练；DINO-world 作为通用模型跨域表现更均衡。

### 3.2 直觉物理（Table 2）

用"surprise"分数衡量物理理解：对合理视频预测误差应低、对违反物理（如物体永久性、重力）的视频应高。本文/V-JEPA/DINO-Foresight 用预测与真实特征的平均绝对误差，COSMOS 用自回归预测器的困惑度。指标为跨类别平均相对准确率↑。

| 方法 | 编码器 | 预测器规模 | IntPhys | GRASP | InfLevel |
|---|---|---|---|---|---|
| COSMOS-4B | VAE | 4B | **99.5** | 60.1 | 44.8 |
| V-JEPA | ViT-L | 22M | 92.2 | 67.0 | 58.9 |
| V-JEPA | ViT-H | 22M | 89.4 | 73.0 | 59.9 |
| DINO-Foresight | ViT-B | 193M | 87.8 | 64.9 | 62.8 |
| **DINO-world** | ViT-B | 1.1B | 91.3 | **76.0** | **63.7** |

DINO-world 在 GRASP / InfLevel 上最优，IntPhys 上与 V-JEPA-H 相当（尽管编码器更小）。COSMOS 在较简单的 IntPhys 近乎满分但另两项掉队。作者认为这些任务噪声大，只作 sanity check 而非严格基准。

### 3.3 消融（Table 3）

- **预测器规模**（IntPhys / CS / VSPW）：Base 86M→84.9/47.7/45.4，Large 304M→89.1/51.9/46.4，Giant 1.1B→90.6/53.2/46.8。清晰的缩放趋势——建模时间动力学比建模静态图像更吃容量。
- **训练数据**：仅用 Cityscapes→66.7/45.6/23.1，仅用 SSv2→79.3/44.9/45.2，而 66M 网络视频→90.6/53.2/46.8。**大规模多样数据是通用世界模型的关键**。
- **视觉编码器**：SD3.5 VAE→(–/13.0/1.5) 几乎失效（非为图像理解设计），SigLIP2→80.7/50.5/41.0，DINOv2→90.6/53.2/46.8 最优。验证 DINOv2 是最佳潜空间。

### 3.4 动作条件规划（Table 4）

在三个仿真环境（PushT 推物、Wall、PointMaze）上，用离线 $(v_t,a_t)$ 轨迹训练 25 epoch（$T=4$、224 分辨率），规划器在潜空间 rollout 候选轨迹并迭代逼近目标。512 episode 成功率：

| 模型 | PushT | Wall | PointMaze |
|---|---|---|---|
| Scratch（从零训） | 46.9 | 87.1 | 59.4 |
| Action-only（冻结基座、只训 action block） | 49.4 | 91.1 | 61.6 |
| Fine-tuned（全量微调） | **59.4** | **93.8** | **68.7** |

核心观察：**大规模预训练对规划有正向作用**，全量微调 > 只训 action block > 从零训。作者预期在更接近预训练数据的复杂环境中收益会更明显。

### 3.5 直接预测 vs 自回归（Figure 3）

在 Cityscapes 上：直接预测（一步查询目标时刻）在短时间差上更准，自回归 rollout（拆成小步）在长视野上更稳；但当预测间隔逼近 1 秒时两者都失准——**长视野预测仍是当前模型的局限**。

## 四、局限性

1. **长时预测退化**：定性可视化（Figure 2）与 Figure 3 都显示，时间视野越长不确定性越高、预测越模糊，接近 1 秒即失准。作者指出根因是 L1 回归目标会对多种可能未来做"平均"，缺乏对未来分布的建模。
2. **训练数据不公开**：约 6600 万条视频为私有池，难以复现；开源替代（Cityscapes/SSv2）因规模小、域窄导致性能明显下降。
3. **规划仅在低维仿真环境验证**：PushT/Wall/PointMaze 都是简单玩具环境，尚未在真实机器人或复杂环境验证动作条件世界模型。
4. **不能直接出像素**：潜空间预测无法直接渲染为可视图像，可视化依赖 PCA 投影；对需要像素级输出的应用是短板。
5. **直觉物理评测噪声大**：作者自己承认这些基准分布偏移严重、需要长时上下文，只当 sanity check。
6. **动作空间受限**：action block 只处理低维连续动作向量，尚未扩展到语言等更丰富的条件信号（作者列为未来方向）。

## 五、评价与展望

**优点**：
- **范式清晰且经济**：把"表征"与"动力学"解耦，用冻结 SSL 编码器 + 轻量潜空间预测器，以 <1.1B 参数、约 6600 万视频，在稠密预测上压过 4–12B 的像素级 COSMOS，是对"世界模型必须烧海量算力做像素生成"这一假设的有力反例。
- **评测统一**：把 dense forecasting、intuitive physics、planning 三条平行研究线放进同一框架横向比较，缓解了世界模型评测碎片化的问题，本身是有价值的贡献。
- **工程设计巧妙**：3-轴 RoPE + 绝对时间戳实现分辨率/帧率无关；cross-attention 因果解码器 + 块三角掩码让全 token 并行训练；零初始化 action block 支持"冻结基座、复用到多任务"的即插即用微调，比 DINO-WM 交织动作 token 的做法更干净。

**与公开工作的关系**：
- 相对 **V-JEPA**：同样在潜空间预测，但 V-JEPA 的预测器只是训编码器的辅助头，本文证明"为预测/规划专门训一个大预测器"在 forecasting 上远胜之（VSPW mid 47.0 vs 4.6）。
- 相对 **DINO-Foresight**：都在 DINOv2 空间预测，但 DINO-Foresight 只在窄域 Cityscapes 上、只对 mask token 算损失；本文扩到大规模通用数据、对全 token 算损失，跨域更均衡（但在驾驶域内 DINO-Foresight 仍略优，说明域内专训仍有价值）。
- 相对 **COSMOS**：以两个数量级更小的成本，在下游语义/几何预测上取胜，印证"潜空间世界模型对下游任务更高效"。
- 相对 **DINO-WM**：沿用其规划评测设置，但用外挂 action block 替代交织动作 token，架构更简洁、可复用基座。

**开放问题与可能改进**：
1. **未来的多模态性**：当前 L1 回归天然趋于"平均未来"，导致长时模糊。引入采样式/生成式目标（如在 DINOv2 特征上做扩散或 flow matching、或离散化后做自回归采样）以"采样一个可能未来"而非取平均，可能是突破长视野瓶颈的关键（作者也点了这个方向）。
2. **数据策展**：消融显示数据规模/多样性至关重要，但全量私有；如何用可控的数据策展策略在公开数据上逼近同等性能，是复现与推广的核心。
3. **真实机器人验证**：规划实验止步于玩具仿真，动作条件世界模型能否在真实操作任务上支撑基于模型的规划/强化学习，仍是最有分量的待答问题。
4. **语言条件**：把条件信号从低维动作扩展到语言指令，将其与 VLA/指令跟随打通，是自然的下一步。
5. **可解码性**：潜空间预测虽高效，但缺乏像素级可解释输出；是否可挂一个轻量解码器在需要时还原像素，同时保持训练效率，值得探索。

## 参考

1. Baldassarre et al. *Back to the Features: DINO as a Foundation for Video World Models*. arXiv 2507.19468, 2025.（本文）
2. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR, 2024.（冻结编码器/潜空间来源）
3. Bardes et al. *V-JEPA / Revisiting Feature Prediction for Learning Visual Representations from Video*. 2024.（联合训练编码器+预测器基线）
4. Karypidis et al. *DINO-Foresight: Looking into the Future of Semantic Features*. 2024.（同源潜空间预测、窄域基线）
5. Zhou et al. *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*. 2024.（规划评测设置与动作注入对照）
6. Agarwal et al. *Cosmos World Foundation Model Platform for Physical AI*. arXiv 2501.03575, 2025.（像素级生成式世界模型基线）
