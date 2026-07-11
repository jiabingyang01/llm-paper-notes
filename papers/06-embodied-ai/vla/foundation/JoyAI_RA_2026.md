# JoyAI-RA：面向机器人自主操作的基础模型

> **论文**：*JoyAI-RA 0.1: A Foundation Model for Robotic Autonomy*
>
> **作者**：Tianle Zhang, Zhihao Yuan, Dafeng Chi, Peidong Liu, Dongwei Li, et al.（核心贡献者共20人；通讯作者 Yuzheng Zhuang, Liang Lin）
>
> **机构**：Joy Future Academy, JD（京东旗下研究团队）
>
> **发布时间**：2026 年 04 月（arXiv 2604.20100）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.20100) | [PDF](https://arxiv.org/pdf/2604.20100)
>
> **分类标签**：`VLA基础模型` `多源数据预训练` `统一动作空间` `跨具身迁移` `第一人称人类视频`

---

## 一句话总结

JoyAI-RA 用统一动作空间把网络数据、自建第一人称人类操作视频（EgoLive）、仿真轨迹与真实机器人数据四类异构来源对齐到同一坐标系，再配合三阶段（VLM 预训练→VLA 联合预训练→目标机器人后训练）训练配方，在 RoboTwin 2.0（90.48%/89.28% Easy/Hard）、RoboCasa GR1 Tabletop（63.2%）和真实 AgiBot G1 机器人基准（平均成功率 0.62→0.74）上均取得当时最优表现。

## 一、问题与动机

开放世界机器人自主操作被两个相互耦合的问题卡住脖子：一是数据多样性不足——尽管 RT-1、Open X-Embodiment 等工作已大幅扩展机器人数据规模，但高质量机器人交互数据采集依旧昂贵且受操作条件限制，长尾交互、罕见失败模式、开放场景布局仍严重欠采样，限制了策略习得的行为先验广度；二是具身差异带来的知识迁移障碍——不同机器人本体在运动学结构、动作空间上差异巨大，跨源、跨具身的行为知识难以有效共享。近期 VLA 系统开始尝试更广泛的跨具身预训练与 scaling，但如何在差异显著的具身之间做有效知识迁移仍是核心开放问题。JoyAI-RA 的思路是同时从"数据端"（引入网络、人类视频、仿真、真实机器人四类互补数据）和"表示端"（显式动作空间统一）两头发力来解决这一对耦合问题。

## 二、核心方法

**四类预训练数据源**（Figure 2a 给出的比例：真实机器人数据 31%、跨具身仿真数据 24%、多模态网络数据 12%、人类第一人称数据 33%）：
- 多模态网络数据：Cambrian-10M、RefSpatial、Galaxea、Cosmos-Reason1-SFT 等视觉问答/空间推理语料，不含可执行操作轨迹，主要提供语言与感知先验；
- 自建第一人称人类操作数据集 **EgoLive**：60FPS RGB 拍摄，覆盖 1969 个物体类别、1796 个动作类别，总计逾 10000 个任务（家庭场景 3779、零售场景 3686、物流场景 2518），含细粒度逐帧子任务标注，并通过自研手部位姿估计流水线将人手轨迹 retarget 到 ALOHA、Fourier、AgiBot G1 等多个机器人本体；
- 仿真数据：InternData-A1、GenieSim3.0-Dataset、InternData-M1 的精选子集；
- 真实机器人数据：Open X-Embodiment、AgiBot-World、Galaxea Open-World Dataset，以及自采数据集 **JDAgibot**。

**统一动作空间（Unified Action Space）**是本文核心设计。为把异构具身的动作对齐到同一物理语义空间：(1) Camera-Frame 末端执行器表示——凡是有可靠外参/末端位姿监督的数据（真实机器人、retarget 后的人类轨迹、仿真）都在相机坐标系而非机器人本体坐标系/关节空间下表示末端位姿，6-DoF 位姿被分解为 3 维平移向量 + 3 维轴角旋转向量；这样做的两个好处是动作向量的物理语义与机器人基座位置无关，且与输入 VLM 的图像视角天然对齐，便于视觉 grounded 的动作预测。(2) 统一动作维度——定义一个覆盖左右臂、左右灵巧手/夹爪等全部执行器组的定长动作向量，某具身缺失的自由度维度在损失和梯度中被 mask 掉，从而同一套定长表示可训练从单臂夹爪到双臂灵巧手系统的各种形态。

用大白话说：不同机器人的"关节"和"坐标系"天差地别，直接混着训练会互相打架。JoyAI-RA 把动作都换算成"相机看到的物体要往哪儿挪、转多少角度"，这样人手、仿真臂、真实机械臂说的其实是同一种语言；训练时只需要把每个具身缺失的自由度维度盖住不计梯度即可。

**模型架构**（Figure 3）：一个 4B 参数的预训练 VLM 负责视觉语言理解，输出 spatially grounded 的多模态表征；一个基于 Perceiver 架构、600M 参数的 Perception-Action Expert 通过 latent bottleneck 做多模态融合并生成连续动作，解耦语义理解与底层控制。模型先生成高层语义描述（如子任务文本），再据此和观测生成动作 chunk。Action Expert 用 flow matching 建模条件速度场，输入是拼接后的隐变量：

$$a_{t:t+H}^0 = \text{Concat}(\phi_s(s_t),\, f_{\text{future}},\, \phi_a(\tilde a_{t:t+H},\tau))$$

用大白话说：把当前本体状态编码、一组可学习的"未来动作占位符" token，以及在时间步 $\tau$ 加了噪声的未来动作特征拼接在一起，作为动作专家的输入。

$$v_{t:t+H}^{\text{out}} = f_\theta(z_t, a_{t:t+H}^0, \tau)$$

用大白话说：模型根据视觉语言表征 $z_t$、拼接后的动作特征和当前去噪时间步 $\tau$，预测一个"速度场"——告诉当前带噪声的动作序列应当往哪个方向修正，从而通过 flow matching（类扩散式生成）逐步从纯高斯噪声去噪出真实动作序列。内部由堆叠的 time-aware Perceiver attention 块实现：每层先对视觉语言流做基于去噪时间步的 AdaLN 自适应归一化，再用注意力更新 latent 动作 token，最后经 FFN 精炼，输出速度预测。

**三阶段训练配方**：
1. VLM Co-Pretraining——混合通用 VQA、具身 VQA（点/框/轨迹等空间理解与任务分解数据）、跨具身离散动作数据（FAST tokenization）、人类视频数据，用标准自回归负对数似然损失：

$$\mathcal{L}_{\text{VLM}}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[-\sum_{j=1}^{n-1} M_j\log p_\theta(y_{j+1}\mid x_{1:j})\Big]$$

用大白话说：就是标准的下一词预测损失，只是预测目标里既有正常的 VLM 文本回答 token，也有离散化后的（FAST）动作 token。

2. VLA Co-Pretraining——在自回归损失基础上叠加 flow matching 损失联合优化：

$$\mathcal{L}_{\text{VLA}}(\theta)=\alpha\cdot\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[-\sum_{j=1}^{n-1} M_j\log p_\theta(y_{j+1}\mid x_{1:j})\Big]+\mathbb{E}_{\mathcal{D},\tau,\omega}\big[\|\omega-a_{1:H}-f_\theta^a(a^{\tau,\omega}_{1:H})\|^2\big]$$

用大白话说：一边继续维持语言理解能力（自回归项），一边让连续动作专家学会去噪预测（flow matching 项），两者用系数 $\alpha$ 加权平衡。

3. VLA Post-Training——只用目标机器人本体采集的数据，丢弃自回归辅助目标，仅用 flow matching 损失端到端微调：

$$\mathcal{L}_{\text{Post}}(\theta)=\mathbb{E}_{\mathcal{D}_{\text{target}},\tau,\omega}\big[\|\omega-a_{1:H}-f_\theta^a(a^{\tau,\omega}_{1:H})\|^2\big]$$

用大白话说：最后一步专门针对部署机器人做"临门一脚"式精调，把预训练学到的通用先验落地到具体本体的执行成功率上。

## 三、关键结果

**RoboTwin 2.0**（Table 1，训练集为 2500 条 clean 场景演示 + 25000 条重度随机化场景演示）：

| 方法 | Easy | Hard |
|---|---|---|
| π0 | 65.92 | 58.40 |
| π0.5 | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| LingBot-VLA | 88.56 | 86.68 |
| **JoyAI-RA** | **90.48** | **89.28** |

**RoboCasa GR1 Tabletop**（Table 2，24 个操作任务，单模型全任务训练，每任务 50 次 rollout）：

| 方法 | GR00T-N1.6 | Qwen3PI | TwinBrainVLA | DualCoT-VLA | ABot-M0 | Being-H0.7 | **JoyAI-RA** |
|---|---|---|---|---|---|---|---|
| 成功率(%) | 47.6 | 43.9 | 54.6 | 55.1 | 58.3 | 49.2 | **63.2** |

在 CanToDrawerClose（+16.0）、MilkToMicrowaveClose（+24.0）、TrayToPot（+18.0）等长序列任务上领先幅度尤其明显。

**真实世界 AgiBot G1 基准**（6 项任务：Headphones/Mouse/Cup/Croissant/Food Scraps/Remedy，每任务 20 次 trial）：相较 π0.5，JoyAI-RA 把跨任务平均成功率从 0.62 提升到 **0.74**；去掉 EgoLive 预训练的消融版本仅 0.56。任务级最大增益出现在 Headphones 与 Remedy（目标识别与精确放置是关键），而在 Cup 与 Food Scraps 上 π0.5 仍有优势，说明长程、精细序贯操作对该模型仍具挑战。

**EgoLive 数据规模消融**（Table 3，RoboTwin 2.0）：

| 设置 | No Pretraining | JDAgibot Only | EgoLive(10%)+JDAgiBot | EgoLive(Full)+JDAgiBot |
|---|---|---|---|---|
| 成功率(%) | 81.64 | 77.62 | 81.40 | **87.42** |

仅用机器人数据（JDAgibot Only）反而低于不预训练的基线，而加入完整规模的 EgoLive 后有约 6 个百分点的进一步提升，显示人类视频的收益在数据规模足够大时才明显显现，未见早期饱和。

**训练阶段消融**（Table 5，RoboTwin 2.0 Easy）：Baseline（仅 VLA Post-Training）81.28% → 仅 VLM Co-Pretraining 87.84% → 仅 VLA Co-Pretraining 87.42% → 两阶段联合 90.48%（较基线 +9.2pt）；Stage 2 中去掉仿真数据后从 90.24% 降到 89.10%（-1.14pt），说明仿真数据在对齐视觉分布、丰富跨具身动作多样性方面仍有独立贡献。

## 四、评价与展望

**优点**：JoyAI-RA 没有单纯堆数据规模，而是把"数据多样性不足"与"具身鸿沟"两个耦合问题用统一动作空间 + 分阶段训练配方系统性地捆绑解决，消融实验（Table 3-5）逐层拆解 VLM/VLA 预训练阶段、仿真数据、人类数据各自的边际贡献，量化证据较为扎实。Camera-Frame 的末端位姿参数化是一个简洁但有效的工程选择，把跨具身对齐问题转化为坐标系归一化问题，与近期强调空间表示统一的工作（如 SpatialVLA）方向一致。自建的 EgoLive 数据集在规模（万级任务）与标注粒度（逐帧子任务标注、跨三种机器人 retarget）上是本文的重要贡献，论文用词频分布与 t-SNE 可视化（Figure 8）定量佐证了其相对 EgoDex 更长尾、语义空间更连续的特点，为"人类视频收益主要来自多样性而非单纯规模"提供了支撑。

**局限与开放问题**：与同期基础模型（π0/π0.5、GR00T N1、ABot-M0 等）相比，本文的核心创新集中在数据工程与训练配方层面，模型架构（VLM + Perceiver 动作专家 + flow matching）延续了较成熟的设计范式，架构新意有限。论文对 EgoDex 与 EgoLive 的比较停留在下游任务性能和语言/视觉统计层面，未给出人手到机器人动作 retarget 误差的定量分析，这部分噪声对策略质量的具体影响仍不透明——论文也自陈"大规模数据采集难免引入噪声"。真实世界评测只覆盖 6 个任务、单一人形平台（AgiBot G1），随机化协议描述较简略，统计置信度与仿真基准相比略弱。更值得关注的是 Figure 7 揭示的负面案例：在 Mouse、Food Scraps 两个任务上，加入 in-domain EgoLive 反而不如不加，作者将其归因于人类视频采集场景与评测场景在环境上下文和任务结构上的分布不匹配——这提示"人类视频数据对齐质量"比单纯扩大规模更关键，也是后续工作可以改进的方向（如显式建模或过滤 retarget 误差、更精细的域匹配采样策略）。此外，论文未公开数据集或模型权重（仅提供 project page），复现性依赖后续开源计划。

## 参考

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*. arXiv:2212.06817, 2022.
- Hoque et al. *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video*. arXiv:2505.11709, 2025.
- Padalkar et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. arXiv:2310.08864, 2023.
- Wu et al. *LingBot-VLA: A Pragmatic VLA Foundation Model*. arXiv:2601.18692, 2026.
