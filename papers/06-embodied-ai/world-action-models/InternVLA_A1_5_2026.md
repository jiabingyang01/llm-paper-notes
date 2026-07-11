# InternVLA-A1.5：统一理解、潜在预见与动作以实现组合泛化

> **论文**：*InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization*
>
> **作者**：Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, et al.（通讯作者 Weinan Zhang）
>
> **机构**：上海人工智能实验室（Shanghai AI Laboratory），Intern Robotics / Physical Intelligence Team
>
> **发布时间**：2026 年 07 月（arXiv 2607.04988）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.04988) | [PDF](https://arxiv.org/pdf/2607.04988)
>
> **分类标签**：`World-Action-Model` `VLA` `Latent-Foresight` `Mixture-of-Transformers` `组合泛化`

---

## 一句话总结

InternVLA-A1.5 用 Mixture-of-Transformers 架构把预训练 VLM（Qwen-3.5 2B）与一个轻量**统一专家**（460M 参数）绑定，专家不在像素空间从零学习未来预测，而是用 50 个可学习的**预见 token**向冻结的预训练视频生成模型（WAN2.2-5B）查询任务相关的未来信息，把动力学先验蒸馏成一段紧凑潜码，再用它条件化 flow-matching 连续动作生成；在 1.2M 机器人 episode + 3M 多模态样本上预训练后，六个仿真基准全部取得最优或接近最优（SimplerEnv 80.8%、LIBERO 98.9%、RoboTwin 93.2%），真实世界组合泛化（未见过的"工具-目标"指令绑定）与长时程任务（MOF 化学制备，76.4% vs π_0.5 的 29.3%）上明显领先 π_0.5 和 Motus 基线，且推理时完全丢弃视频生成分支，单步推理约 0.1 秒，保持实时闭环控制。

## 一、问题与动机

具身智能领域近年出现两条并行路线：一条是 Vision-Language-Action（VLA）模型，通过继承预训练 VLM 的语义先验获得跨物体、跨指令的泛化能力；另一条是 Video Action / World Action Model，通过预测未来视觉状态学习物理动力学感知。二者互补，因此涌现出一批尝试用单一统一模型同时兼顾两者的工作（作者自己此前的 InternVLA-A1 即是早期尝试之一，已显示出在动态操作任务上的明显收益）。

但作者指出现有统一模型普遍存在三个根本性张力：

1. **语义漂移**：理解组件（大规模 VQA/语言训练）往往在生成和动作目标叠加之后才被引入，随着训练推进预训练 VLM 的语义知识逐渐被侵蚀，指令跟随能力被削弱；
2. **异质目标互相干扰**：未来视觉隐变量回归、flow-matching 动作预测、语言建模三类目标在形式和尺度上完全不同，联合训练时彼此干扰，减慢甚至损害收敛；
3. **视觉预测从零学起**：视觉预测模块通常从零开始重建未来状态，没有利用大规模预训练视频生成模型中已经蕴含的时空动力学先验，等于把"如何想象世界演化"这件事又学了一遍。

InternVLA-A1.5 要回答的问题是：如何在不牺牲预训练 VLM 语义能力、也不用支付像素级生成全额计算成本的前提下，把预训练视频生成模型里的世界动力学知识注入到动作策略中。

## 二、核心方法

### 2.1 整体架构：Mixture-of-Transformers

模型由两部分组成：(1) 一个作为多模态感知与推理骨干的预训练 VLM（Qwen-3.5 2B，混合注意力结构，每个 block 交替 3 层 Gated DeltaNet 线性注意力 + 1 层标准全注意力，共 6×）；(2) 一个与 VLM 共享同一架构蓝图、但隐藏维度更小的轻量**统一专家**（460M 参数）。两者只在共享的全注意力层交互，各自维护独立的 Gated DeltaNet 层做模态专属处理——这正是 Mixture-of-Transformers（MoT）设计的核心：VLM 与专家既联通又不完全共享参数，避免异质目标直接互相踩踏。

输入侧，每个时间步 $t$ 的多视角观测 $o_t$、语言指令 $l$、离散化的本体感知状态 $q_t$（均匀分箱为每维 256 个区间）以及一个控制模式标记（关节空间 / 末端执行器 / VQA）被编码进统一的 chat-template：Prompt 部分拼接 $K$ 个图像块、任务指令、控制模式与离散状态，Label 部分是子任务描述加上经 FAST tokenizer 编码的离散动作段（机器人数据），或答案文本（VQA 数据）。动作词表大小为 2048，直接拼接到 VLM 原词表后面，与其余 token 共享同一套 embedding 表和语言头。

### 2.2 阶段一目标：把 VLM 变成 VLA 执行器

沿用 π_0.5 的思路，将子任务预测 $\hat{\ell}$ 与动作 chunk $\mathbf{a}_{t:t+H}$ 的联合分布做自回归分解：

$$
\pi_\theta(\mathbf{a}_{t:t+H}, \hat{\ell} \mid o_t, \ell) = \pi_\theta(\mathbf{a}_{t:t+H} \mid o_t, \ell, \hat{\ell})\, \pi_\theta(\hat{\ell} \mid o_t, \ell)
$$

训练目标是标准的下一 token 交叉熵：

$$
\mathcal{L}_{\text{stage1}} = -\mathbb{E}_{(o_t,\ell,y)\sim\mathcal{D}}\left[\sum_{i=1}^{M+N} \log p_\theta(y_i \mid o_t, \ell, y_{<i})\right]
$$

**大白话说**：这一步只训练主干 VLM（不引入统一专家），把机器人数据和 VQA 数据统统塞进同一套"对话模板"，让 VLM 顺带学会预测子任务、离散动作 token，同时不丢掉原本的语言理解能力——这一"prompt 设计"被作者在结论中特别强调为本文最重要的经验之一。

### 2.3 阶段二：预见推理机制（Foresight Reasoning）

阶段二引入统一专家，并把未来预测重新表述为一个**潜在查询（latent-querying）问题**：插入 $M=50$ 个可学习的预见 token $Q^f \in \mathbb{R}^{M\times d}$，它们通过统一专家的共享全注意力层查询当前视觉-语言隐藏状态 $H_t$，产生一段紧凑的"未来相关"表示：

$$
Z_t^f = \Phi_\theta\big([H_t;\, Q^f]\big)_{\mathcal{F}}
$$

这段预见嵌入被投影进冻结的 WAN2.2-5B 视频生成模型的条件空间，替换掉其原本的 T5 文本编码器输入：

$$
C_t^f = P_{\text{WAN}}(Z_t^f)
$$

对每个动作 chunk，均匀采样 $N=4$ 个未来帧作为预测目标，拼接当前帧构成视频片段 $V_t$，经 WAN-VAE 编码为干净潜变量 $x_1$。按 WAN 使用的 flow-matching 目标采样噪声 $x_0\sim\mathcal{N}(0,I)$ 和插值时刻 $s\in[0,1]$，$x_s=(1-s)x_0+sx_1$，目标速度 $v_s=x_1-x_0$，视频监督损失为：

$$
\mathcal{L}_{\text{video}} = \mathbb{E}_{x_0,x_1,C_t^f,s}\left[\left\| u(x_s, C_t^f, s) - v_s \right\|^2\right]
$$

**大白话说**：WAN 视频生成模型的全部参数被冻结，梯度只经由条件通路回传到预见 token 和上游专家层——这意味着预见 token 被训练成"能被视频模型读懂"的查询码，模型只需学会向一个已经很懂物理世界的老师"提问"，而不必自己重新学会画视频。推理时这条视频分支被完全丢弃，不引入任何逐帧生成的延迟。

### 2.4 动作生成：flow-matching

统一专家并行地用 flow-matching 生成连续动作 chunk。给定真实动作 chunk $\mathbf{a}_{t:t+H}$，采样噪声 $\epsilon\sim\mathcal{N}(0,I)$、插值时刻 $\tau\sim\text{Beta}(1.5,1.0)$，构造插值动作：

$$
\mathbf{a}_{t:t+H}^{\tau} = (1-\tau)\epsilon + \tau\,\mathbf{a}_{t:t+H}
$$

动作预测损失为：

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\mathbf{a}_{t:t+H},\epsilon,\tau}\left[\left\| v_\theta^{\text{act}}\big(\mathbf{a}_{t:t+H}^{\tau}, H_t, Q^f\big) - (\mathbf{a}_{t:t+H}-\epsilon) \right\|^2\right]
$$

推理时从高斯噪声出发，用 $K$ 步 Euler 积分求解学到的速度场：

$$
\mathbf{a}_{t:t+H}^{\tau+\Delta\tau} = \mathbf{a}_{t:t+H}^{\tau} + \Delta\tau \cdot v_\theta^{\text{act}}\big(\mathbf{a}_{t:t+H}^{\tau}, H_t, Q^f\big)
$$

阶段二总损失把三部分加权合并：

$$
\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{stage1}} + \alpha\,\mathcal{L}_{\text{video}} + \beta\,\mathcal{L}_{\text{action}}, \quad \alpha=1,\ \beta=10
$$

**大白话说**：预见 token 不仅要向视频模型汇报"未来会怎样"，其产出的表示还会被动作头复用，作为条件之一去噪出连续动作——相当于让"对未来的想象"直接服务于"现在该怎么动"这件事，而不是两条互不通气的支路。

### 2.5 注意力掩码与训练协议

统一专家内部对预见 token 组和噪声动作 embedding 组采用分组因果 + 组内双向注意力：预见 token 只看 VLM 上下文，噪声动作 embedding 可以同时看 VLM 上下文与预见 token，组内互相双向可见（匹配 flow-matching 并行去噪的非自回归性质）。训练时统一专家被显式禁止关注 FAST 离散动作 token，避免离散分支的真值信息泄露给连续动作分支、也避免两个分支的梯度互相干扰。

三阶段训练配置：阶段一预训练 30 万步（batch 1024，恒定学习率 $5\times10^{-5}$）；阶段二预训练 60 万步（同 batch/学习率，引入统一专家和预见机制）；后训练阶段 6 万步（batch 128，学习率余弦从 $5\times10^{-5}$ 衰减到 $5\times10^{-6}$）。预见 token 数量与动作 chunk 长度均固定为 50。

### 2.6 数据配方

机器人操作语料汇聚 6 个来源（1 个合成 + 5 个真实世界：InternData-A1、AgiBotWorld、UMI、DROID、Galaxea、RoboMind 1.0），共 120 万 episode、8.61 亿帧，全部映射进 InternVLA-A1 定义的统一动作空间。多模态协同训练语料来自 InternVLA-M1，约 300 万样本，涵盖通用 QA（63.7 万）、Box QA 指代检测（87.9 万）、Point QA 空间定位（83.2 万）、Trajectory QA 末端执行器路径点预测（68.4 万）四类。采样策略采用两级分组：组内按帧数的 $\gamma$ 次幂采样，组间权重先用 Re-Mix 方法初始化再手工微调，刻意上调小规模真实数据源（DROID、Galaxea、RoboMind）权重、下调占比最大的合成数据源；机器人语料与多模态语料整体按 0.15∶0.85 固定比例混合，保证每个 batch 大部分仍在巩固 VLM 的语义与定位能力。

## 三、实验结果

### 3.1 真实世界实验

四个真实任务：Sort Tubes、Insert Tubes、Move Tubes（三者均设计了"部分绑定训练可见、部分绑定仅测试可见"的组合泛化测试）与 MOF（13 步长时程化学制备流程）。对比基线为 π_0.5 和 Motus，所有实验在单张 RTX 5090 上完成，单步推理约 0.1 秒。

| 任务 | π_0.5 | Motus | InternVLA-A1.5 |
|---|---|---|---|
| Sort Tubes（总体） | 77.8% | 64.8% | 75.9% |
| Insert Tubes（总体） | 51.7% | 44.2% | 72.5% |
| Move Tubes（总体） | 72.7% | 56.2% | 80.5% |
| MOF（长时程） | 29.3% | 0.0% | 76.4% |

组合泛化拆分（Seen / 未见绑定 OOD）：

| 任务 | π_0.5 (Seen/OOD) | Motus (Seen/OOD) | InternVLA-A1.5 (Seen/OOD) |
|---|---|---|---|
| Sort Tubes | 86.0 / 63.6 | 64.6 / 54.5 | 83.0 / 65.0 |
| Insert Tubes | 60.0 / 56.7 | 36.7 / 46.7 | 80.0 / 60.0 |
| Move Tubes | 71.9 / 66.7 | 50.0 / 60.0 | 87.5 / 83.3 |

在未见指令绑定上，InternVLA-A1.5 三项任务均取得最高成功率，说明优势并非来自"背下"训练时见过的绑定；Sort Tubes 总体略逊于 π_0.5，作者归因于该任务只需把管子放进一个开口大箱、精度要求低，π_0.5 在这类简单 pick-and-place 上已经很强。MOF 长时程任务上 Motus 全部失败（0%），π_0.5 仅 29.3%，InternVLA-A1.5 达 76.4%，作者将差距归因于显式子任务预测带来的进度感知能力，以及学到的动力学先验使策略能推理液体倾倒等改变环境状态的动作。

### 3.2 六个仿真基准

| 基准 | 指标 | InternVLA-A1.5 | 最强基线（数值） |
|---|---|---|---|
| SimplerEnv (WidowX) | 平均成功率 | **80.8%** | Xiaomi-Robotics-0 79.2% |
| LIBERO（4 suite 平均） | 平均成功率 | **98.9%** | Xiaomi-Robotics-0 98.7% |
| LIBERO-Plus（零样本） | Total | **84.8%** | π_0.5 84.4% |
| RoboTwin 2.0 | Clean/Rand/Avg | **93.3 / 93.0 / 93.2** | LingBot-VA 92.9/91.5/92.2 |
| DOMINO（零样本 / 微调） | SR | **27.7% / 29.3%** | Qwen-VLA-Instruct 26.6%（零样本对比） |
| EBench | Val-Train/Val-Unseen/Test SR | **43.1 / 32.8 / 35.2** | LingBot-VA 38.3/26.6/30.9 |

六个基准全部拿到最优或并列最优平均分，覆盖单臂（LIBERO 系）、双臂（RoboTwin）、移动操作（EBench）、real-to-sim（SimplerEnv）以及动态物体操作（DOMINO）等不同技能类型；LIBERO-Plus 与 DOMINO 均为零样本泛化测试（分别在 LIBERO 和 RoboTwin 上训练后直接评测），说明预见机制带来的鲁棒性能迁移到分布外扰动和动态场景。

### 3.3 消融与训练效率

| 配置 | LIBERO | LIBERO-Plus | RoboTwin | DOMINO |
|---|---|---|---|---|
| 完整模型 | 98.9 | 84.8 | 93.2 | 27.7 |
| 去掉视频监督损失 | 97.9 | 78.0 | 91.1 | 25.3 |
| 去掉可学习预见 token | 98.6 | 77.9 | 90.2 | 23.8 |

去掉视频损失或去掉预见 token 都导致各基准全面下降，在零样本 LIBERO-Plus 和动态物体的 DOMINO 上跌幅最明显，验证预见 token 是把预训练视频模型的时空动力学先验蒸馏进统一专家的关键接口。另外在 RoboTwin 微调设置下对比 SFT loss 曲线，InternVLA-A1.5 相比 π_0.5 和前作 InternVLA-A1 收敛最快、最终 loss 最低，说明预训练学到的表示为下游微调提供了更有利的优化地形。定性上，作者还展示了预见嵌入条件化的 WAN 模型可以生成物理合理的多步未来 rollout（包括零样本场景下对液面变化等因果状态转移的建模），佐证预见 token 确实学到了可读出的世界模型信息。

## 四、局限性

论文明确指出两点局限：第一，预见监督只覆盖单个动作 chunk 的短时程范围（$N=4$ 帧），策略吸收的是局部动力学先验，尚不具备长时程想象或基于世界模型的显式规划能力；第二，视频生成器（WAN2.2）在全部训练阶段保持冻结且是通用模型，继承的先验因此受限于该生成器预训练时对具身/机器人场景的覆盖程度。作者表示将在后续工作中处理这两点。

## 五、评价与展望

**优点**：本文最有价值的贡献是把"统一模型如何利用预训练视频生成先验"这个问题，从"训练一个像素级生成分支"简化为"训练一小撮可学习查询 token 去读出一个冻结教师模型的知识"，在保留真实闭环推理速度（约 0.1s/步，无需在线生成视频）的同时获得了动力学先验的收益，这是相对于同类 world-action model（需要在推理时做未来想象/生成）的一个明确工程与效率优势。消融实验对视频损失和预见 token 的贡献都给出了清晰的正向证据，六个仿真基准 + 四个真实任务的覆盖面也较为全面，尤其在零样本泛化（LIBERO-Plus、DOMINO）和长时程执行（MOF）上优势最突出，这与"预见机制主要帮助应对分布外扰动和动态场景"的论点自洽。阶段一"把机器人数据和 VQA 数据塞进同一套 chat-template、共享同一个语言头"的朴素设计被证明对稳定训练、保留 VLM 语义能力很有效，是一个值得其他统一模型借鉴的工程经验。

**局限与开放问题**：其一，视频生成器保持完全冻结、且是通用领域预训练（WAN2.2），论文自己也承认继承的先验受限于该模型对具身场景的覆盖——是否微调生成器、或换用专门在机器人视频上预训练的生成模型，会带来多大额外收益尚未探索。其二，预见监督的时间窗口仅为一个动作 chunk（约 4 帧），相比同期一些强调"测试时未来想象"的 world-action model（如同一批文献中的 Fast-WAM，专门讨论"world action model 是否需要测试时未来想象"）形成了鲜明对比——InternVLA-A1.5 选择完全放弃测试时想象换取速度，这一取舍在需要更长时程规划、更强不确定性建模的任务上是否仍然成立，是一个开放问题。其三，50 个预见 token 作为信息瓶颈，论文没有给出这段紧凑潜码具体保留/丢失了哪些未来信息的定量分析（仅有定性 rollout 可视化），其容量是否会在更复杂、更长时程的任务上成为限制值得进一步研究。其四，实验对比的基线（π_0.5、Motus、LingBot-VA、Xiaomi-Robotics-0 等）训练数据规模和构成各不相同，本文用了 120 万 episode + 300 万多模态样本的定制混合语料，架构改进带来的收益与数据规模/配比带来的收益尚未做严格解耦（如同规模数据下的纯架构消融）。

**与其他公开工作的关系**：本文与 Motus、τ_0-wm、GEAR/DreamZero、VLA-JEPA 等一批"world action model"工作同属于把视频预测/世界建模融入 VLA 策略这一方向；相较于直接生成未来像素或在像素/patch 空间做未来预测的思路，本文的差异化贡献在于把预测重新表述为对冻结生成模型的潜在查询问题，并通过 Mixture-of-Transformers 显式隔离离散 FAST token 分支与连续 flow-matching 分支的梯度，避免了作者所述的"异质目标互相干扰"问题；这一设计脉络也延续自作者前作 InternVLA-A1，以及同期借鉴 π_0.5 "先预测子任务再预测动作"分解思路的其他统一模型。

## 参考

1. Intelligence et al. π_0.5: a Vision-Language-Action model with open-world generalization. arXiv:2504.16054, 2025.
2. Cai et al. InternVLA-A1: Unifying understanding, generation and action for robotic manipulation. arXiv:2601.02456, 2026.
3. Bi et al. Motus: A unified latent action world model. arXiv:2512.13030, 2025.
4. Wan et al. Wan: Open and advanced large-scale video generative models. arXiv:2503.20314, 2025.
5. Yuan et al. Fast-WAM: Do world action models need test-time future imagination? arXiv:2603.16666, 2026.
