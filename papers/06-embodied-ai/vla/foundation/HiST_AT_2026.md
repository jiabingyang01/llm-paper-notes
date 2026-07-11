# HiST-AT：面向机器人上下文模仿学习的分层时空动作分词器

> **论文**：*A Hierarchical Spatiotemporal Action Tokenizer for In-Context Imitation Learning in Robotics*
>
> **作者**：Fawad Javed Fateh†、Ali Shah Ali†（†共同一作）、Murad Popattia、Usman Nizamani、Andrey Konin、M. Zeeshan Zia、Quoc-Huy Tran
>
> **机构**：Retrocausal, Inc.（美国 Redmond, WA）
>
> **发布时间**：2026 年 04 月（arXiv 2604.15215；本笔记依据当前版本 v3，修订于 2026-05-29）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.15215) | [PDF](https://arxiv.org/pdf/2604.15215)
>
> **分类标签**：`具身智能` `模仿学习` `上下文学习` `动作分词器` `分层向量量化`

---

## 一句话总结

针对机器人上下文模仿学习（in-context imitation learning, ICIL），提出两级分层向量量化（细粒度子簇 Z→粗粒度动作簇 A）并联合重建动作与其时间戳的动作分词器 HiST-AT，在 ICRT 框架下于 RoboCasa 七任务上把平均成功率从此前最优基线 LipVQ-VAE 的 0.530 提升到 0.590，在 ManiSkill 三任务上从 0.617 提升到 0.670。

## 一、问题与动机

ICIL 借鉴 LLM 的 in-context learning 能力，让机器人策略仅凭推理时给定的少量示范（prompt demonstrations）就能完成新任务，无需针对新任务微调（follow ICRT [Fu et al. 2024] 的 next-token 预测式建模）。论文指出该范式的关键瓶颈是动作表示的质量：现有动作分词器（离散分箱、MLP、FAST、VQ-VAE、LFQ-VAE 等）大多采用"flat"（单层）聚类，且只通过重建动作本身来利用空间信息，既没有显式建模动作的分层结构（短时子动作原语组合成长程连贯动作），也常常缺乏时间平滑性。近期的 LipVQ-VAE 用 Lipschitz 正则化增强了动作平滑性，但仍是单层 VQ + 仅空间重建，表达能力受限（图 1(b) 称为"Flat Clustering + Spatial Reconstruction"）。

作者的动机来自两条独立但可结合的研究线：(1) 时序动作分割领域中"时空联合重建"用于表征学习的做法；(2) 无监督动作分割中的分层向量量化（HVQ）。论文将二者结合，提出图 1(c) 所示的"Hierarchical Clustering + SpatioTemporal Reconstruction"路线，即 HiST-AT。

## 二、核心方法

**整体流程（图 2）**：编码器 $f_\theta$ 把输入动作序列 $\mathbf{X}\in\mathbb{R}^{(B\cdot S)\times D_{feature}}$（含相对位置和夹爪角度）映射为潜在表示 $\mathbf{V}$；再经 Lipschitz 条件化网络 $f_\psi$ 得到平滑的潜在表示 $\mathbf{V}'$；随后通过两级向量量化得到量化码 $\mathbf{Q}^Z$、$\mathbf{Q}^A$；最后分别用空间解码器重建动作 $\hat{\mathbf{X}}$、用时间解码器（两层 MLP）预测时间戳 $\hat{\mathbf{T}}$。

**Lipschitz 正则化**（follow LipVQ-VAE）：网络每层权重按行做归一化并乘以可学习的 softplus 界：

$$\mathbf{W}_i^{(\ell)} = \frac{\mathbf{W}_i^{(\ell)}}{\sum_j |\mathbf{W}_{i,j}^{(\ell)}|} \cdot \text{softplus}(c_\ell)$$

大白话：强行把每层的"放大倍数"钳制住，使得输入动作的微小扰动不会在网络里被无限放大，从而保证重建/量化出来的动作轨迹是平滑连续的。

**两级分层向量量化**（受 HVQ 启发）：维护两个码本 $\mathbf{Z}=\{\mathbf{z}_j\}_{j=1}^{\alpha K}$（细粒度子动作原型）和 $\mathbf{A}=\{\mathbf{a}_i\}_{i=1}^{K}$（粗粒度动作簇），其中 $\alpha$ 是比例参数。第一级把每个正则化后的潜在向量 $\mathbf{v}_k'$ 量化到最近的子簇原型：

$$\mathbf{q}_k^Z = \mathbf{z}_{j^*}, \quad j^* = \arg\min_j \|\mathbf{v}_k' - \mathbf{z}_j\|_2$$

第二级再把（经 Lipschitz 平滑后的）子簇码 $\mathbf{q}_k^{Z'}$ 量化到最近的高层簇：

$$\mathbf{q}_k^A = \mathbf{a}_{i^*}, \quad i^* = \arg\min_i \|\mathbf{q}_k^{Z'} - \mathbf{a}_i\|_2$$

大白话：先把每一帧动作归到很多个"细粒度动作字"（比如"轻微下压""小幅旋转"），再把这些细粒度字进一步归并成较少的"高层动作词"（比如"抓取""插入"），构成一个类似语言层级的动作词表，短时子动作可以复用/组合出不同的长程动作。

**时空联合重建**：$\mathbf{Q}^A$ 送入空间解码器重建原始动作 $\hat{\mathbf{X}}$；$\mathbf{Q}^{Z'}$ 送入时间解码器预测时间戳 $\hat{\mathbf{T}}$。二者用 MSE 度量：

$$\mathcal{L}_{spat}=\frac{1}{B\cdot S}\sum_k \|\hat{\mathbf{X}}^{(k)}-\mathbf{X}^{(k)}\|_2^2,\quad \mathcal{L}_{temp}=\frac{1}{B\cdot S}\sum_k \|\hat{\mathbf{T}}^{(k)}-\mathbf{T}^{(k)}\|_2^2$$

大白话：只重建动作数值，模型可能学到"看起来像"但顺序错乱的表示；额外要求模型能从量化码猜出这一帧原本发生在轨迹的第几个时刻，等于强迫码本保留时间/顺序信息，防止不同时间步的动作被混淆到同一个码。

**总损失**：两级 VQ 的承诺损失+码本损失（$\mathcal{L}_{vq_Z}=\mathcal{L}_{commit_Z}+\mathcal{L}_{codebook_Z}$，$\mathcal{L}_{vq_A}$ 同理）与时空重建损失、Lipschitz 正则损失加权求和：

$$\mathcal{L} = \lambda_{vq}(\mathcal{L}_{vq_Z}+\mathcal{L}_{vq_A}) + \lambda_{spat}\mathcal{L}_{spat} + \lambda_{temp}\mathcal{L}_{temp} + \lambda_{reg}(\mathcal{L}_{reg_Z}+\mathcal{L}_{reg_A})$$

码本随机初始化，训练时联合优化编码器/正则化网络/两级码本/两个解码器。

## 三、关键结果

实验在单张 NVIDIA A100 上进行。RoboCasa 上评测 7 个任务、训练 500K 迭代（MimicGen 合成数据）；ManiSkill 上评测 3 个任务、训练 30K 迭代。RoboCasa 用 ICRT 框架搭配不同分词器对比，ManiSkill 用 ACT 框架搭配不同分词器对比。

**RoboCasa（表 1，平均成功率）**：

| 方法 | Pick&Place | Open Doors | Open Drawers | Levers | Knobs | Insertion | Buttons | Average |
|---|---|---|---|---|---|---|---|---|
| BC-Transformer | 0.29 | 0.55 | 0.78 | 0.62 | 0.31 | 0.24 | 0.78 | 0.477 |
| ICRT+MLP | 0.20 | 0.61 | 0.81 | 0.70 | 0.32 | 0.35 | 0.64 | 0.442 |
| ICRT+LipVQ-VAE（此前最优） | 0.32 | 0.80 | 0.84 | 0.68 | 0.41 | 0.41 | 0.59 | 0.530 |
| **ICRT+HiST-AT（本文）** | **0.35** | **0.90** | 0.89 | 0.72 | **0.52** | **0.44** | 0.63 | **0.590** |

**ManiSkill（表 2）**：ACT+HiST-AT 平均成功率 0.670（Pick Cube 0.85、Push Cube 0.78、Stack Cube 0.38），超过此前最优 ACT+LipVQ-VAE 的 0.617 达 5.3 个百分点。

**消融（表 3，RoboCasa）**：以 LipVQ-VAE 基线（0.530）为起点，单独加时空重建到 0.552，单独加分层聚类到 0.573，两者都加即完整 HiST-AT 达到 0.590（约 +6 个百分点），说明两个组件贡献互补，其中分层聚类的单独增益更大。

**其它消融**：码本大小 $(\alpha K, K)$ 在 ManiSkill 上从 (32,16) 增至 (64,16) 有提升，再增至 (64,32) 不再提升甚至略降，最终取 (64,16)；时间重建权重 $\lambda_{temp}$ 在 $[0.002,2]$ 区间内以 $\lambda_{temp}=0.02$（适中权重）效果最好，过大反而损害性能。

**跨数据集迁移（表 4，MimicGen→稀疏/无结构的 Human 数据集）**：ICRT+HiST-AT 平均成功率 0.575，超过第二名 ICRT+LipVQ-VAE（0.525）约 5 个百分点。

**零样本泛化（表 5，训练/测试任务集拆分）**：ICRT+HiST-AT 平均成功率 0.090，超过第二名 LipVQ-VAE（0.052）约 3.8 个百分点（各方法在此设定下绝对成功率均很低）。

**真实机器人（表 6）**：用 UR5e 机械臂完成 Pick Cube、Stack Cube 两个任务，每任务采集 10 条真机遥操作示范，并用 MimicGen 生成 2000 条合成示范，合成+真实数据混合训练；评测时另收集 50 条示范作为上下文提示。Sim-to-real 迁移仍具挑战：ICRT+HiST-AT 成功率 Pick Cube 0.23 / Stack Cube 0.14，优于 ICRT+LipVQ-VAE 的 0.19 / 0.12，但绝对水平仍偏低。

## 四、评价与展望

**优点**：方法概念清晰——用"两级 VQ（分层聚类）+ 时空联合重建"两个正交组件分别解决"动作层级结构未建模"和"仅空间重建丢失时序信息"两个具体问题，且消融实验干净地证明了两者独立有效、组合互补。评测覆盖了 RoboCasa/ManiSkill 两个仿真基准的多任务场景、跨数据集迁移、零样本泛化以及真实 UR5e 机器人，验证链条相对完整，在与同为 ICRT 框架内的多种分词器（MLP、离散分箱、FAST、VQ-VAE、LFQ-VAE、LipVQ-VAE）对比中全面占优。

**局限**：(1) 真实世界成功率绝对值仍然很低（0.14-0.23），真实任务仅 2 个、示范量很小（10 条真机遥操作 + 2000 条合成），本文的主要证据链仍集中在仿真环境，sim-to-real 差距本身未被系统分析；(2) 时间解码器只是一个两层 MLP，回归绝对时间戳这一设计相对简单，论文自陈"更先进的时间目标"留作未来工作；(3) 分层深度固定为两层（子簇→簇），未探讨更深层级或自适应层数/粒度是否有进一步收益；(4) 码本大小与 $\lambda_{temp}$ 的消融只在 ManiSkill 上做，未验证在 RoboCasa 或真实数据上是否有相同规律；(5) 比较对象局限于 ICRT/ACT 这类中小规模 ICIL 框架和任务集，尚未在大规模预训练 VLA（如以 FAST 为动作分词器的 pi0 量级模型）上验证该分层时空 VQ 思路能否 scale。

**与其他公开工作的关系**：本文可视为把动作分割领域的分层向量量化（HVQ）与时空表征学习中的联合时空重建，首次系统迁移到"动作分词器"这一具体子问题，直接建立在 LipVQ-VAE（IROS 2025）的 Lipschitz 平滑机制之上做增量式改进；相比 VLA 主线中常见的 FAST/离散分箱等分词方案，其分层+时序建模思路提供了一个可能的补充方向，但目前的验证规模（单张 A100、任务数个位数、模型量级远小于主流 VLA）与 FAST 论文覆盖的大规模跨本体数据集相比仍有明显差距。

**开放问题**：分层时空 VQ 分词器能否迁移到大规模 VLA 的动作头并在预训练规模下依然有效；时间戳回归是否存在更适合的监督信号形式（如相对进度百分比而非绝对时间戳）；分层聚类的层数与粒度能否自动/端到端确定而非手工设定 $(\alpha K, K)$；如何进一步缩小 sim-to-real 的性能差距。

## 参考

- L. Fu et al. In-context imitation learning via next-token prediction (ICRT), arXiv:2408.15980, 2024
- A. D. Vuong, M. N. Vu, D. An, I. Reid. Action tokenizer matters in in-context imitation learning (LipVQ-VAE), IROS 2025
- F. Spurio, E. Bahrami, G. Francesca, J. Gall. Hierarchical vector quantization for unsupervised action segmentation (HVQ), AAAI 2025
- S. Nasiriany et al. RoboCasa: Large-scale simulation of household tasks for generalist robots, RSS 2024
- S. Tao et al. ManiSkill3: GPU parallelized robotics simulation and rendering for generalizable embodied AI, arXiv:2410.00425, 2024
