# AgiBot World Colosseo：面向可扩展智能具身系统的大规模操作平台

> **论文**：*AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems*
>
> **作者**：Qingwen Bu, Guanghui Ren, Chiming Liu, Maoqing Yao, Yu Qiao, Hongyang Li, et al.（Team AgiBot-World）
>
> **机构**：The University of Hong Kong、AgiBot Inc.、Shanghai Innovation Institute、Shanghai AI Lab（论文原文标注）
>
> **发布时间**：2025 年 03 月（arXiv 2503.06669，v4 2025 年 08 月）
>
> **发表状态**：IROS 2025（camera-ready，Change Log 标注）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.06669) | [PDF](https://arxiv.org/pdf/2503.06669)
>
> **分类标签**：`大规模操作数据` `双臂人形` `latent action` `VLA/ViLLA` `data engine`

---

## 一句话总结

用 100+ 台同构双臂人形机器人 + 全流程标准化 + human-in-the-loop 质检，采集了 100 万条真机操作轨迹（217 任务 / 87 技能 / 106 场景）并全部开源，同时提出以 latent action 为中间表征的三阶段生成式策略 GO-1，在真机复杂任务上平均完成分达 0.78（RDT-1B 仅 0.46、π0 0.58），并验证了数据量与策略性能之间 r=0.97 的幂律缩放关系。

## 一、问题与动机

作者的核心判断是：机器人落后于 NLP/CV 的根本原因是缺少「高质量数据」，而现有真机数据集在三条轴上同时受限：

- **规模与多样性不足**：早期集（RoboTurk 2.1k、BridgeData 7.2k）体量小；RT-1 130k、BridgeData V2 60.1k 仍局限于单臂、桌面短任务；DROID 76k 虽然众包扩场景，但缺乏质量保证。
- **聚合式方案的隐患**：Open X-Embodiment（OXE）把众多数据集拼成 1.4M 统一格式，但 embodiment、视角、质量高度不一致，策略只能在短时任务内学习，跨域泛化弱。
- **任务过于「实验室化」**：绝大多数轨迹时长小于 5s，是 pick-and-place 这类单原子技能，无法覆盖真实世界里长程、需灵巧手、需协作的复杂操作。

因此作者提出一个问题：*how could we resolve the real-world complexity effectively by scaling up real-world robot data?* 目标是构建一个「全栈」平台——统一硬件 + 标准化采集 + 质量管控 + 精心设计的任务集 + 配套模型与工具链，用规模化真机数据去桥接受控实验室与真实部署之间的鸿沟。

## 二、核心方法

方法分两大块：**数据平台（AgiBot World）** 与 **策略模型（GO-1）**。

### 1. 硬件与采集：标准化 + human-in-the-loop

- **硬件 AgiBot G1**：双 7-DoF 臂 + 移动底盘 + 可调腰部；末端模块化（标准夹爪或 6-DoF 灵巧手）；配 visuo-tactile 触觉传感器；共 8 个相机（前视 RGB-D、三个前视鱼眼、每个末端一个 RGB-D/鱼眼、后方两个鱼眼）；30 Hz 记录图像与本体状态。
- **两套遥操作**：VR 头显控制（手势映射到末端平移/旋转，再经 IK 转关节角，但灵巧手只能做少数预定义手势）与全身动捕控制（记录手指/躯干/头部，解锁更精细的灵巧操作）。
- **三阶段采集流程**（Fig 2，edge-side/cloud-side 闭环）：① 可行性预采集 + 制定采集标准；② 遥操作者按标准采集，本地做 validity verification（如查漏帧）后上云；③ 云端标注者做后处理、按标准做 quality check，并补全 item/scene/skill(子步骤)/task 四级语言标注。
- **两个「反直觉」的数据决策**：
  - **Failure recovery**：遥操作中掉物等失误若被成功挽回，则保留该轨迹并标注失败原因与时间戳，约占数据集 1%，用于策略对齐与「失败反思」。
  - **Human-in-the-loop 反馈闭环**：小批采集→训策略→部署评测→用评测暴露的问题（如动作起始处长时停顿、过多空闲）反向修订采集协议（如后处理剔除 idle 段），形成持续的数据质量迭代。

### 2. GO-1 / ViLLA：以 latent action 为枢纽的三阶段策略

作者提出 **ViLLA（Vision-Language-Latent-Action）** 框架，区别于把动作直接条件化在视觉-语言上的 VLA：ViLLA 先预测 latent action token，再生成低层动作，从而能用海量「无动作标签」的网络视频/人类视频参与预训练。

**Stage 1 — Latent Action Model（LAM）**：在互联网级异构数据（web 视频、Ego4D 人类视频、跨 embodiment 机器人数据、AgiBot World）上训练一个编码器-解码器，把连续两帧的变化编码为离散 latent action。逆动力学编码器给出

$$\mathbf{I}(z_t \mid I_t, I_{t+H})$$

前向动力学解码器重建未来帧

$$\mathbf{F}(I_{t+H} \mid I_t, z_t)$$

编码器是带因果时序掩码的 spatial-temporal transformer（沿用 Genie 的思路），解码器是以初始帧 + latent token $z_t = [z_t^0,\dots,z_t^{k-1}]$（$k=4$）为输入的 spatial transformer；latent token 用 VQ-VAE 目标量化，码本大小为 $\lvert C \rvert$。

用大白话说：不知道机器人真实动作没关系，只要看「这一帧→H 帧之后」的画面变化，就能反推出一个抽象的「动作意图」编号。这个编号来自人类视频也能学，于是把网络视频的动力学知识搬进了机器人。

**Stage 2 — Latent Planner**：以在 web-scale 视觉-语言数据上预训练的 VLM（**InternVL2.5-2B**）为骨干，多视角图像先经 InternViT 编码再投影到语言空间；Latent Planner 是 24 层 transformer，用 full bidirectional attention 并逐层接收 VLM 骨干的条件信息，预测 latent action token

$$\mathbf{P}(z_t \mid I_t^h, I_t^l, I_t^r, l)$$

监督信号来自 Stage 1 的 LAM 编码器在头部相机上产生的 $z_t := \mathbf{I}(I_t^h, I_{t+H}^h)$。由于 latent 空间比 OpenVLA 那种离散化低层动作小几个数量级，通用 VLM 到机器人策略的适配更高效。

用大白话说：这一步就是让一个「会看图会读指令」的大模型去做长程规划，但它规划的不是关节角，而是那串抽象动作编号，所以既能保留 VLM 的推理能力，又与具体机身解耦（embodiment-agnostic）。

**Stage 3 — Action Expert**：与 Latent Planner 共享架构但目标不同——Latent Planner 做掩码语言建模生成离散 latent token，Action Expert 用 **diffusion（去噪）目标** 回归连续低层动作，并层级化地条件于前面所有模块。它解码动作块

$$A_t = [a_t, a_{t+1}, \dots, a_{t+H}], \quad H=30$$

条件包含本体状态 $p_t$，即 $\mathbf{A}(A_t \mid I_t^h, I_t^l, I_t^r, p_t, l)$。推理时 VLM + Latent Planner + Action Expert 协同：先预测 $k$ 个 latent token，再经去噪产生最终控制信号。

用大白话说：抽象编号还不能直接驱动电机，Stage 3 用扩散模型把「意图」翻译成 30 步高频、连续、灵巧的真实动作序列。

## 三、实验结果

**数据集对比（Table I）**：AgiBot World 是「to date 最大」的真机操作数据集，1M+ 轨迹、87 技能、106 场景，且同时具备 detailed annotation / camera calibration / 双臂 / 灵巧手 / failure recovery / human-in-the-loop 全部维度——这是此前数据集（含 OXE 聚合、DROID 76k、RoboMIND 55k）都不完整具备的。全量 beta 版共 1,001,552 条、2976.4 小时。

| 数据集 | 轨迹数 | 技能 | 场景 | 双/单臂 | 灵巧手 | Failure recovery | Human-in-loop |
|---|---|---|---|---|---|---|---|
| RT-1 | 130k | n/a | 2 | 单 | ✗ | ✗ | ✗ |
| BridgeData V2 | 60.1k | 13 | 24 | 单 | ✗ | ✗ | ✗ |
| DROID | 76k | 86 | 564 | 单 | ✗ | ✗ | ✗ |
| Open X-Embodiment | 1.4M | 217 | 311 | 单+双 | ✗ | ✗ | ✗（聚合） |
| **AgiBot World** | **1M+** | **87** | **106** | **双** | **✓** | **✓** | **✓** |

**GO-1 vs 主流 generalist 策略（Fig 5，6 个真机任务，每任务 10 rollout 取归一化均分）**：所有对比策略都在 AgiBot World beta 上预训练，公平比较模型本身。

| 任务 | RDT-1B | π0 | GO-1 w/o Latent Planner | GO-1 |
|---|---|---|---|---|
| Restock Bag | 0.84 | 0.91 | 0.93 | 0.98 |
| Table Bussing | 0.54 | 0.60 | 0.87 | 1.00 |
| Pour Water | 0.13 | 0.41 | 0.42 | 0.67 |
| **平均** | **0.46** | **0.58** | **0.66** | **0.78** |

要点：GO-1 全面领先；latent planner 平均带来 **+0.12** 完成分，尤其在「Fold Shorts」等复杂任务与「Restock Beverage」的指令跟随/泛化上收益显著。摘要口径为 GO-1 在复杂任务上 >60% 成功率，较 RDT 提升 **32%**。

**数据集是否提升策略性能与泛化（Fig 6，固定用 RDT 模型，只换预训练数据源）**：

| 场景 | OXE 预训练 | AgiBot World (Alpha) | AgiBot World (Beta) |
|---|---|---|---|
| In-distribution 平均 | 0.47 | 0.68 | **0.77** |
| Out-of-distribution 平均 | 0.38 | — | **0.67** |

- 平均完成分：in-domain **+0.30**、OOD **+0.29**；「Table Bussing」性能近乎翻三倍。
- Alpha 版（约 10% 数据、236h）虽远小于 OXE（约 2000h），成功率反而更高，凸显数据质量。
- 摘要另给出：仅用相当于 OXE 1/10 时长的数据子集，预训练策略泛化性即提升 **18%**。

**数据缩放律与质量消融（Fig 7）**：

| 分析 | 关键数字 |
|---|---|
| 数据规模缩放（9.2k→1M 轨迹） | 幂律拟合 $y=12.24x^{0.11}$，Pearson $r=0.97$ |
| 质量过滤（"Wipe Table" 任务） | 全量（All）0.41 → 仅人工核验（Verified only）0.59 |
| 核验 vs 未核验 | 528 条已核验 vs 482 条未核验；已核验数据带来 **+0.18** 完成分 |

结论鲜明：**「数量更大」不必然带来性能提升**，一小批人工核验数据反而更有价值——为 human-in-the-loop 质检提供了直接证据。

## 四、局限性

- **仅真机评测，缺仿真基准**：作者明确指出所有评测都在真实世界进行，仿真环境（与真机对齐以复现部署结果）仍在开发中，导致评测难以快速、可复现。
- **硬件强绑定**：数据与 GO-1 深度依赖 AgiBot G1 这一特定双臂人形本体（8 相机、6-DoF 灵巧手、触觉），跨本体迁移能力未系统验证。
- **VR 遥操作对灵巧手表达受限**：VR 手柄只能触发少数预定义手势，精细灵巧数据主要依赖动捕系统，采集成本与一致性存在权衡。
- **Failure recovery 仅约 1%**：失败/挽回数据占比很低，其对策略对齐的增益规模化后是否成立，论文未深入。
- **数字读取自图表**：多数细粒度结果以柱状图给出，个别任务分项数值精度有限（本文表格只保留可靠可读项与官方文本口径的平均值）。

## 五、评价与展望

**优点**：
- 这是真机操作数据「工程化」的一次系统性示范。相比 DROID 的众包扩场景、OXE 的事后聚合，本文把「统一硬件 + 标准化协议 + 云边闭环质检 + 四级语言标注」做成了可复制的生产管线，且 Fig 7 用可量化证据（r=0.97 幂律 + 质量过滤 +0.18）回答了「规模 vs 质量」这一社区长期争论，说服力强。
- ViLLA/GO-1 把 latent action 作为「通用中间货币」，与 LAPA（latent action pretraining）、Genie（latent action + 世界模型）一脉相承，但补齐了 LAPA 在下游丢失 latent planning 能力的短板，并在 Stage 3 用 diffusion action expert 保证高频灵巧输出，工程完整度高于多数只发数据的工作。
- 全量数据 + 模型 + 工具链以 CC BY-NC-SA 4.0 开源，对社区是实打实的公共品。

**不足与开放问题**：
- **数据质量的「质检标准」本身是隐性的**：human-in-the-loop 依赖标注者主观判断与策略反馈，what makes a trajectory "good" 尚未形式化，难以自动化、难以跨团队对齐。
- **latent action 的可解释性与最优粒度**（$k=4$、码本 $\lvert C \rvert$）缺乏系统消融，latent 空间是否真的「embodiment-agnostic」仍偏经验。
- **与 π0、RDT 的对比略有主场优势**：所有基线都在 AgiBot 数据上预训练，未在各自原生数据/最佳配置下比较，GO-1 的模型级增益与数据级增益部分耦合。
- **改进方向**：① 把仿真-真机对齐补齐，形成可复现评测闭环；② 探索自动化数据质量评分以替代人工核验（可与缩放律联合建模「有效数据量」）；③ 扩大 failure/recovery 与反思数据比例，验证其对长程任务的鲁棒性收益；④ 探索 latent action 到多本体的迁移，检验 ViLLA 的跨 embodiment 上限。

总体而言，本文的真正贡献不在单一模型，而在于把「真机数据 = 高质量、可缩放、可核验的工程系统」这一理念做实，并用缩放律与质量消融给出了可被后续工作复用的方法论。

## 参考

1. Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," ICRA 2024.（大规模聚合数据集，本文主要对照基线）
2. A. Khazatsky et al., "DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset," RSS 2024.（众包真机数据，本文对比其质量管控缺失）
3. J. Bruce et al., "Genie: Generative Interactive Environments," ICML 2024.（latent action + 世界模型，LAM 编码器思路来源）
4. S. Ye et al., "LAPA: Latent Action Pretraining from Videos," ICLR 2025.（latent action 预训练，本文补齐其下游 latent planning 缺失）
5. K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," 2024.（flow-based action expert 与主要策略基线）
