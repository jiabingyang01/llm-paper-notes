# PhysGen：从预训练视频生成模型中学习物理规律的多模态连续时序世界交互模型

> **论文**：*Learning Physics from Pretrained Video Models: A Multimodal Continuous and Sequential World Interaction Models for Robotic Manipulation*
>
> **作者**：Zijian Song, Qichang Li, Sihan Qin, Yuhao Chen, Tianshui Chen, Liang Lin, Guangrun Wang（通讯作者）et al.
>
> **机构**：中山大学（Sun Yat-sen University）；广东省大数据分析与处理重点实验室（Guangdong Key Laboratory of Big Data Analysis and Processing）；X-Era AI Lab；广东工业大学（Guangdong University of Technology）
>
> **发布时间**：2026 年 03 月（arXiv 2603.00110；v2 于 2026 年 4 月 23 日更新）
>
> **发表状态**：未录用（预印本）。论文页眉标注为模板占位符 "ACM Conference, 2026"（ACM acmart 模板默认值），正文未见任何具体会议/期刊录用信息。
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.00110) | [PDF](https://arxiv.org/pdf/2603.00110)
>
> **分类标签**：`世界模型` `视频生成预训练迁移` `连续token自回归` `扩散去噪` `VLA`

---

## 一句话总结

PhysGen 将预训练的连续（非量化）视频自回归模型 NOVA 直接复用为机器人策略骨干，用共享的"物理 token"把视频帧与动作联合自回归建模、以扩散过程做连续信号的去 token 化，在**完全不做动作专项预训练**的情况下，以 732M 参数在 LIBERO 上取得 90.8% 平均成功率（超 OpenVLA/WorldVLA 13.8/8.8 个百分点），在 ManiSkill PushCube 上达到 100%，真实机器人透明物体抓取任务上比 π0 高 5 个百分点。

## 一、问题与动机

机器人大规模示教数据的采集成本高、耗时长，因此复用其他模态的基础模型知识成为一条现实路径。当前主流做法是 VLA（Vision-Language-Action），即在预训练 LLM/VLM 上接一个动作头（OpenVLA、π0、SpatialVLA、ThinkAct、MolmoACT 等）。但文本与动作模态之间存在天然鸿沟：文本符号化地描述世界，而操作任务需要精确的时空理解与物理控制，这导致符号推理与物理控制的对齐往往是次优的。

一个更"物理"的替代路线是构建于预训练**视频生成模型**（尤其是自回归式的）之上：这类模型通过反复预测未来观测，隐式地捕捉了世界的物理规律与时序动态，其"逐步预测下一状态"的机制与序贯决策过程天然契合。已有 WorldVLA、UWM、UniMimic 等工作尝试联合建模视频与动作，但它们普遍依赖**离散 token 化**（矢量量化）表示图像/动作，这会给连续信号引入分辨率误差，并可能随轨迹推进而累积漂移。

PhysGen 的核心问题是：能否设计一个**统一的连续 token 空间**，让视频观测与机器人动作共享同一套自回归 + 扩散建模流程，从而更完整地把预训练视频模型中的隐式物理知识迁移到操作策略上，并且不需要昂贵的动作专项预训练？

## 二、核心方法

### 2.1 总体框架：物理 token 与物理自回归

PhysGen 建立在预训练的连续视频自回归模型 **NOVA**（非量化视频自回归模型，采用"帧间时间自回归 + 帧内空间集合自回归"的两级结构）之上。NOVA 的生成过程可写为：

$$p(l,S_1,\dots,S_N)=\prod_{n=1}^{N}p(S_n\mid l,S_1,\dots,S_{n-1})$$

（用大白话说：把一段视频看成一串"帧 token"，像语言模型逐词预测一样逐帧预测下一帧，条件是文本提示 $l$ 和之前所有帧。）

每一帧内部再按 token 集合做空间自回归：

$$p(S'_n,S_{(n,1)},\dots,S_{(n,K)})=\prod_{k=1}^{K}p(S_{(n,k)}\mid S'_n,S_{(n,1)},\dots,S_{(n,k-1)})$$

PhysGen 在此基础上引入 **physical token**：把每一步的视觉观测 token $E_{O,n}$ 与动作 token $E_{A,n}$ 沿序列维度拼接为一个联合 token：

$$P_n=[E_{O,n};E_{A,n}],\qquad P_n\in\mathbb{R}^{(K_O+K_A)\times d}$$

（用大白话说：不再把"看到什么"和"该做什么"当成两条分开建模的信息流，而是把它们焊成同一个 token，让感知和动作在同一个自回归序列里共同演化。）

因为动作相对观测存在一步的时间偏移（动作是"由当前观测决定的下一步输出"），PhysGen 引入一个可学习的 **BOA（Begin of Action）token**，插入到动作序列前对齐两者的时间步长。整个"物理自回归"过程写作标准的下一 token 预测形式：

$$p(E_l,P_0,\dots,P_N)=\prod_{n=0}^{N}p\left(P_n\mid E_l,P_0,\dots,P_{n-1}\right)$$

条件分布由一个复用 NOVA 架构的因果 Transformer（Causal Transformer）参数化。语言指令用冻结的 Phi 小语言模型 token 化，视觉观测用冻结的 3D-VAE token 化为帧 token，动作则通过新引入的动作 tokenizer（MLP）投影进同一连续物理嵌入空间。

### 2.2 连续信号的扩散去 token 化

离散自回归模型（VQ 类）在处理图像/动作这类本质连续的信号时会引入量化误差。PhysGen 延续 MAR（Li et al.）"用扩散去噪计算条件分布，不依赖信号是否离散"的思路，把扩散过程当作**连续信号的去 token 化器**。

标准扩散前向过程为 $q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_tI)$，训练目标是去噪损失：

$$\mathcal{L}(z,x)=\mathbb{E}_{\epsilon,t}\left[\lVert \epsilon-\epsilon_\theta(x_t\mid t,z)\rVert^2\right]$$

其中 $x_t=\sqrt{\bar\alpha_t}x+\sqrt{1-\bar\alpha_t}\,\epsilon$，$z$ 是条件向量。在 PhysGen 中，条件向量取 Transformer 在第 $n$ 步的输出：

$$Z_n=\mathrm{Transformer}(l,P_0,\dots,P_{n-1})$$

再用一个 DiT 去噪网络估计物理 token 的条件分布：

$$\mathcal{L}(P_n,Z_n)=\mathbb{E}_{\epsilon,t}\left[\lVert \epsilon-\epsilon_\theta(P_n\mid t,Z_n)\rVert^2\right]$$

推理时通过反向扩散链逐步采样：

$$P_{n,t-1}=\frac{1}{\sqrt{\alpha_t}}\left(P_{n,t}-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(P_{n,t}\mid t,Z_n)\right)+\sigma_t\delta$$

（用大白话说：Transformer 只负责"根据历史猜一个高层次的条件向量"，真正把这个条件向量还原成具体的像素/动作数值，交给一个轻量扩散网络"去噪"出来——图像走 NOVA 原本的重建范式，动作则由新设计的 **Action-DiT**（用交叉注意力把 $Z_n$ 注入）完成。这样连续信号既保留了扩散模型的生成表达力，又避免了离散量化误差。）

训练总损失是所有物理 token 上观测扩散损失与动作扩散损失之和：

$$loss=\sum_{n=1}^{N}\mathcal{L}(Z_n,P_n)=\sum_{n=1}^{N}\Big(\mathcal{L}_{obs}(Z_n,E_{O,n})+\mathcal{L}_{act}(Z_n,E_{A,n})\Big)$$

### 2.3 三项架构增强设计

1. **因果掩码实现隐式逆运动学（Causal Masking for Implicit IK）**：帧内部分采用分块（chunk-wise）全注意力，同一帧所有 patch 互相可见；动作部分采用时间因果掩码（chunk 内早期动作不能看到后面动作）；关键设计是动作 token 被允许**单向**关注帧 token，即可以关注到（对齐后代表未来结果的）视觉状态，这隐式地让动作规划以"未来视觉状态"为条件，起到类似逆运动学推理的作用。

2. **Lookahead-MTP（L-MTP）**：把"前瞻式规划"与"多 token 并行预测"（MTP）结合，在每个自回归步并行解码 3 个未来 token；训练时对所有预测 token 都加监督（teacher forcing），推理时只执行第一个预测 token，其余作为前瞻信息去调节后续预测，从而拉长规划视野、提升时序一致性。

3. **高效训练与推理**：训练端用全并行的 teacher forcing 一次前向计算所有 token 的损失，并用 LoRA 微调 Transformer 骨干以保留预训练能力；推理端引入 KV-cache 缓存逐层中间特征，实现高效自回归生成。

### 2.4 关键实现细节

动作 chunk 长度 $L=8$；Transformer 最大上下文长度 2096 token（256 语言 token + 5 个物理 token 包，每包含 360 视觉 token + 8 动作 token）；多视角图像被拼接为单图后送入 VAE 与自回归 Transformer，靠模型自身自注意力维持跨视角一致性；使用 RoPE 位置编码，对帧 token 与动作 token 分别设置不同频率。骨干**不在任何大规模动作/操作数据集上做预训练**（论文称为 "no action pretraining"），只在视频生成任务上预训练，再直接在下游操作任务上微调。所有微调实验均在单张 NVIDIA A100-SXM4-80GB 上完成，最长训练不超过 60 GPU 小时；训练时对每张图像/每个动作采样 4 次扩散时间步 $t$（沿用 MAR 的做法）。

## 三、实验结果

### LIBERO（成功率 %，四个任务套件各微调约 400 条示教、评测 500 次 rollout）

| 方法 | 参数量 | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|---|
| DP（无动作预训练） | – | 78 | 93 | 68 | 51 | 72 |
| Octo | 93M | 79 | 86 | 85 | 51 | 75 |
| OpenVLA | 7B | 85 | 88 | 79 | 54 | 77 |
| SpatialVLA | 4B | 88 | 90 | 79 | 56 | 78 |
| ThinkAct | 7B | 88 | 91 | 87 | 71 | 84 |
| Pi0-Fast | 3B | **96** | 97 | 89 | 60 | 86 |
| MolmoACT | 7B | 87 | 95 | 88 | 77 | 87 |
| UniMimic（无动作预训练） | ~200M | 71 | 79 | 67 | 29 | 62 |
| UniMimic | ~200M | 89 | 91 | 85 | 59 | 81 |
| CoT-VLA | 7B | 88 | 92 | 88 | 69 | 84 |
| WorldVLA（无动作预训练） | 7B | 88 | 96 | 83 | 60 | 82 |
| **PhysGen（无动作预训练）** | **732M** | 91.0 | **99.6** | **93.8** | **78.8** | **90.8** |

PhysGen 平均成功率全场最高，超过 WorldVLA 8.8 个百分点、超过 OpenVLA 13.8 个百分点，在长时序任务 LIBERO-Long 上比 WorldVLA 高 18.8 个百分点；唯一逊于 Pi0-Fast 的是 LIBERO-Spatial（91.0% vs 96%），论文将其归因于底层视频模型空间感知能力有限。

### ManiSkill（成功率 %，各任务 1000 条示教微调、125 次 rollout 评测）

| 方法 | PushCube | PickCube | StackCube | Avg |
|---|---|---|---|---|
| ACT | 76 | 20 | 30 | 42 |
| BC-T | 98 | 4 | 14 | 39 |
| DP | 88 | 40 | 80 | 69 |
| ICRT | 77 | 78 | 30 | 62 |
| RDT | **100** | **77** | **74** | **84** |
| Pi0 | **100** | 60 | 48 | 69 |
| **PhysGen（Ours）** | **100** | 73 | 48 | 74 |

PhysGen 在 PushCube 上达到满分 100%，整体平均略逊于用大规模动作数据预训练的 RDT（74% vs 84%），但优于 ICRT（+12pp）和 Pi0（+5pp）。

### 真实机器人（Franka Panda + 2 台 RealSense D415，80–100 条遥操作示教/任务，各 20 次试验）

| 方法 | Pick Cube | Press Button | Stack Cube | Pick Transparency | Avg |
|---|---|---|---|---|---|
| ACT | 40 | 40 | 30 | 10 | 30 |
| OpenVLA | 30 | 25 | 10 | 0 | 16.3 |
| Pi0 | 85 | **85** | **60** | 70 | 75 |
| **PhysGen（Ours）** | 80 | **85** | **60** | **75** | 75 |

PhysGen 在完全不做任何动作预训练、仅用采集到的少量真机数据训练的情况下，平均成功率与经大规模动作预训练的 π0 打平（均为 75%），且在需要处理折射/反光等强物理歧义的 Pick Transparency（透明物体抓取）任务上比 π0 高 5 个百分点。

### 消融实验（LIBERO-Object 任务）

| 变体 | 视频预训练 | Token 表示 | L-MTP | 自回归 | 成功率 |
|---|---|---|---|---|---|
| PhysGen-Zero | 无 | 连续 | 有 | AR | 86.4% |
| PhysGen-Discrete | NOVA | 离散 | 有 | AR | 94.2% |
| PhysGen-NoAR | NOVA | 连续 | 有 | 无AR（端到端单步） | 95.0% |
| PhysGen-STP | NOVA | 连续 | 无 | AR | 96.8% |
| PhysGen-Full | NOVA | 连续 | 有 | AR | **99.6%** |

四项设计各自带来的绝对提升：视频生成预训练 +13.2pp（Zero→Full）、连续 token 表示 +5.4pp（Discrete→Full）、自回归架构 +4.6pp（NoAR→Full）、L-MTP +2.8pp（STP→Full）。其中视频预训练贡献最大，验证了论文的核心假设——视频生成中学到的动态先验能显著提升物理接地的操作能力。

## 四、局限性

- 论文明确承认的短板：在 LIBERO-Spatial 上被 Pi0-Fast 反超（91.0% vs 96%），归因于底层视频模型（NOVA）本身空间感知能力有限，作者将其列为未来改进方向。
- 语言 tokenizer（冻结 Phi）与视觉 tokenizer（冻结 3D-VAE）均不参与微调，只有因果 Transformer 主干通过 LoRA 更新，加上新引入的动作 tokenizer/Action-DiT；模型能学到的"新物理知识"很大程度上受限于预训练视频先验本身的质量与覆盖范围（该先验来自通用视频语料，并非专门的具身/操作视频）。
- 评测场景相对集中：仿真为 LIBERO 四套件 + ManiSkill 三个桌面任务，真实机器人仅在单一 Franka Panda 平台上测试 4 个抓取/按压/堆叠类任务，尚未展示在更多样的机械臂形态或长时序、接触密集型复杂任务上的表现。
- 论文未报告推理延迟或控制频率等指标；每个自回归步都需要跑一次 DiT 反向扩散链去噪出动作，尽管有 KV-cache 加速自回归部分，扩散采样本身的多步迭代成本对实时闭环控制的影响未被量化说明。
- 消融实验只在 LIBERO-Object 单一任务套件上进行，四项设计的收益是否在其他任务套件（尤其是长时序 LIBERO-Long）上同样成立，缺乏进一步验证。

## 五、评价与展望

PhysGen 的核心贡献是把"视频生成模型即物理模拟器"这一直觉落到了一个具体、干净的架构上：用共享的连续物理 token 把感知和动作焊接进同一条自回归链，并用扩散过程统一承担连续信号的生成式去 token 化，从而避免了 WorldVLA 等同类工作依赖离散 VQ 表示带来的分辨率误差。在只有 732M 参数、且完全不做动作专项预训练的情况下于 LIBERO/ManiSkill 上追平甚至超过多个 7B 级、经过大规模动作数据预训练的 VLA 基线，是比较有说服力的数据效率证据；真实机器人上对透明物体的稳健抓取，也提供了一个直观的"物理直觉可从视频预训练迁移"的定性案例。

与最相关的同类工作相比：WorldVLA、UWM、UniMimic 都尝试联合建模视频与动作，但 WorldVLA 走离散 token 路线，UWM 用两个独立的扩散过程分别处理动作和视频，UniMimic 侧重在潜在动作/潜在状态层面做联合预训练；PhysGen 的差异化在于用单一连续 embedding 空间 + 因果跨模态注意力（动作单向可见帧 token）来实现更紧密的双模态交互，实验上也确实以更小参数量取得了更高的平均成功率。不过论文的消融虽然拆解了"是否用视频预训练""连续 vs 离散""是否自回归""是否 L-MTP"四个因素，但并未提供与 WorldVLA/UniMimic 在同一骨干规模、同一训练数据下的严格受控对比，因此"连续表示本身"相对"离散表示"的优势有多少是表示方式带来、多少是训练细节或骨干选择（NOVA vs 各基线自身骨干）带来，仍有一定混杂。

开放问题包括：（1）视频预训练带来的收益中，有多少来自"物理规律"本身、多少只是更好的通用视觉初始化——这是所有"世界模型/视频预训练当具身预训练"路线共有的待厘清问题；（2）扩展性尚未探讨，即该框架在骨干规模进一步放大（或视频预训练数据规模进一步扩大）时收益是否持续，目前 732M 规模已展现较强性能，但同样走视频预训练路线的 UniMimic（~200M）LIBERO 均值仅 81%，规模-性能关系值得进一步系统研究；（3）长时序任务（LIBERO-Long 78.8%）与更短任务（Object 99.6%）之间仍有约 21 个百分点差距，提示因果视频-动作自回归链在长 rollout 上的误差累积问题并未完全解决，Lookahead-MTP 只带来 2.8pp 的边际改善，是否有更强的长视野规划机制（如显式子目标分解、分层规划）值得探索。

## 参考

1. Deng et al. "Autoregressive Video Generation without Vector Quantization"（NOVA，PhysGen 所基于的预训练连续视频自回归骨干）。
2. Li et al. "Autoregressive Image Generation without Vector Quantization"（MAR，PhysGen 扩散去 token 化范式的直接来源）。
3. Cen et al. 2025. "WorldVLA: Towards Autoregressive Action World Model." arXiv:2506.21539（离散联合视频-动作自回归的代表性基线，PhysGen 的主要对比对象）。
4. Chen et al. 2025. "Unifying Latent Action and Latent State for Policy Learning from Videos."（UniMimic，SIGGRAPH Asia 2025；联合隐动作/隐状态预训练的相关工作）。
5. Black et al. 2024. "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164（大规模动作预训练 VLA 代表，PhysGen 真机实验的主要对比基线）。
