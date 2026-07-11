# EgoScaler-VLA：从第一视角视频构建视觉-语言-动作模型

> **论文**：*Developing Vision-Language-Action Model from Egocentric Videos*
>
> **作者**：Tomoya Yoshida, Shuhei Kurita, Taichi Nishimura, Shinsuke Mori
>
> **机构**：Kyoto University；National Institute of Informatics；Institute of Science Tokyo；NII LLMC；Sony Interactive Entertainment
>
> **发布时间**：2025 年 09 月（arXiv 2509.21986）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.21986) | [PDF](https://arxiv.org/pdf/2509.21986)
>
> **分类标签**：`egocentric-video` `VLA-pretraining` `6DoF-trajectory` `data-curation`

---

## 一句话总结

用作者此前的 EgoScaler 框架从**无任何辅助标注（无手部姿态、无深度、无时间戳）**的原始第一视角视频里自动抽取 6DoF 物体操作轨迹,经过滤清洗构建出 45,157 条 episode 的 VLA 预训练数据集,在 π0 架构上预训练后真机 pick-and-place 平均成功率从 scratch 的 0% 提升到 55%,与真机机器人数据集（BC-Z/BridgeData V2/Fractal）持平甚至略优,且与 BridgeData V2 混合后进一步升到 68%。

## 一、问题与动机

VLA 预训练几乎全部依赖**人类遥操作（teleoperation）**采集真机数据,成本高、劳动密集,构成数据稀缺瓶颈。第一视角(egocentric)视频是一条有前景的替代路线:AR/VR 设备与智能眼镜带来大量近距离人-物交互录像,天然蕴含操作运动线索,且场景多样、易获取。

但既有工作有两条路都不够理想:

- **富标注路线**(EgoMimic、EgoVLA、EgoDex 等):依赖手部姿态、动作起止时间戳等**密集辅助录制**,需要多相机系统、深度传感器或 Aria Glasses 之类专用硬件,可扩展性受限。
- **隐式潜动作路线**(LAPA):用 VQ-VAE 从相邻帧对学离散潜动作 token,虽不需辅助标签、可扩展,但潜表征**难以捕捉细粒度运动**——在 push 这类简单动作上表现好,在 pick-and-place 这类复杂技能上只是中等。

因此,能否**直接从原始第一视角视频、不用任何辅助录制**训练出 VLA,仍是开放问题。本文的立场是:抽取**显式动作轨迹**(物体如何平移、旋转)能提供比潜动作更鲁棒、更信息量的监督。作者还观察到一个反直觉现象——在真实多样的 egocentric 视频上训练的策略,搬到简化视觉的仿真器里评测时反而可能掉点(visual domain gap)。

## 二、核心方法

整体流程:用 EgoScaler 从四个大规模 egocentric 视频集抽 6DoF 物体轨迹 → 规则化过滤清洗 → 在 π0 上预训练 → 在小规模具身特定数据上后训练(post-training)。每条 episode 含一张图像序列、一条文本指令、一条 6DoF 物体位姿轨迹。

### 2.1 问题定义

沿用 ALOHA/π0 的 action chunk 建模。在时刻 $t$,给定语言指令 $\ell$、RGB 观测 $v_t$、本体感知状态 $\tau_t$,学习策略

$$\pi(a_{1:H} \mid v_t, \tau_t, \ell)$$

对未来 $H$ 步动作 $\{a_t, \dots, a_{t+H-1}\}$ 建模。预训练阶段 $v_t$ 是单张图,$\tau_t$ 与 $a_t$ 都在**不含 gripper 的末端执行器空间**里近似(由 egocentric 视频推得);后训练/评测阶段 $v_t$ 可为一到多路相机,$\tau_t$、$a_t$ 用机器人原生控制空间。

**用大白话说**：预训练时机器人根本没上场,而是把"视频里物体质心怎么移动、怎么转"当成"末端执行器状态"的替身;推理时再一次性吐出一段未来动作依次执行。人类数据和机器人数据之间的模态差,靠 VLA 本身的容错能力和归一化去弥合。

### 2.2 EgoScaler 轨迹提取(四阶段)

EgoScaler(作者 CVPR 2025 的前作)把一段 egocentric 视频转成一条 6DoF 物体操作轨迹,四步:

1. **动作定位**：用 GPT-4o 识别动作起止时间戳与被操作物体;
2. **位置序列**：用开放词表分割(Grounding DINO + SAM)+ 稠密 3D 点跟踪(SpatialTracker)抽出被操作物体的位置序列;
3. **去除拍摄者运动**：通过点云配准把序列投影到动作起始帧的相机坐标系,消掉戴相机者自身的移动;
4. **旋转序列**：对相邻帧物体点云做 SVD 求相对变换,得到旋转。

合成后得到位姿序列 $\tau = \{\tau_1, \dots, \tau_T\}$,其中

$$\tau_t = (x, y, z, \text{roll}, \text{pitch}, \text{yaw})$$

$(x,y,z)$ 为物体质心的平移,$(\text{roll},\text{pitch},\text{yaw})$ 为旋转。本文把 EgoScaler 从只覆盖 Ego-Exo4D 扩展到**四个数据集**:Ego4D、Ego-Exo4D、HD-EPIC、Nymeria(涵盖 cooking / experiments / crafting / repair 等)。其中 Ego4D **不提供相机内参**,作者先用 COLMAP(SfM)估计内参,若找不到满足特征匹配与几何约束的有效初始像对则丢弃该实例。初步抽出 **124,559 条 episode**。

### 2.3 数据清洗:两条规则过滤 + 平滑

EgoScaler 会因**物体检测错误**和**点云配准错误**产出坏轨迹,作者用两条自动规则去除:

**(1) 行程距离阈值 $\delta_{DT}$(治配准错误)**。轨迹平移分量的累积位移

$$D = \sum_{t=1}^{T-1} \lVert p_{t+1} - p_t \rVert_2, \quad p_t = (x_t, y_t, z_t)$$

配准出错时相邻帧会突变,$D$ 异常大;丢弃 $D > \delta_{DT}$ 的轨迹。

**(2) 背景轨迹相似度阈值 $\delta_{BGTS}$(治检测错误)**。检测错误常发生在未被交互的静止物上,其运动轨迹与背景高度相似。设物体轨迹 $\{o_t\}$、背景轨迹 $\{q_t\}$(均为图像平面 2D 坐标),定义速度向量 $u_t = o_{t+1}-o_t$、$b_t = q_{t+1}-q_t$,背景轨迹相似度(BGTS)取平均余弦相似度:

$$\text{BGTS} = \frac{1}{T-1}\sum_{t=1}^{T-1} \frac{u_t \cdot b_t}{\lVert u_t \rVert \, \lVert b_t \rVert}$$

丢弃 $\text{BGTS} > \delta_{BGTS}$ 的 episode,经仿真实验凭经验取 $\delta_{BGTS} = 0.7$。

**(3) 平滑**：帧间深度不一致导致平移分量抖动,对每个平移向量在以当前帧为中心的 5 帧窗内做均值滤波(序列边界处窗口自适应缩小)。

清洗后最终得到 **45,157 条 episode**。

**用大白话说**：EgoScaler 抽出来的轨迹里混着两类垃圾——一类是配准崩了、物体"瞬移"很远(用总位移 D 太大来抓),一类是跟错了物体、跟到了背景上不动的东西(用"这条轨迹和背景一起动吗"的余弦相似度来抓)。再加一层 5 帧滑窗把深度抖动磨平。三招下来 12.5 万条砍到 4.5 万条。

### 2.4 动作表示与策略训练

用 π0(基于 PaliGemma VLM + flow-matching 动作头)。预训练时动作定义为 6DoF 物体位姿的位移:

$$a_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta \text{rot6D}_t]$$

其中 $\text{rot6D}_t \in \mathbb{R}^6$ 是旋转矩阵 $R_t$ 前两列展平。因为**拿不到 gripper 状态**,动作是 **9 维**向量。本体感知状态用原始轨迹 $\tau$ 并把旋转转成 rot6D 表示。为缓解人-机数据分布差,动作与本体状态在训练中都做归一化。与真机数据混训时,对动作/本体向量做 padding 与归一化以对齐维度,再在拼接数据上联合训练。

训练目标为动作 chunk 上的 MSE:

$$\mathcal{L}_{\text{action}} = \frac{1}{H}\sum_{t=1}^{H} \lVert \hat{a}_t - a_t \rVert_2^2$$

预训练与后训练共用此损失。受 SmolVLA 启发,**冻结 VLM 参数**(预训练与后训练全程),更省显存、更快。

## 三、实验结果

**数据集统计(Table I)**：本文数据规模中等,但**动词/物体多样性最高**。

| 数据集 | #Episodes | #Verbs | #Objects |
|---|---|---|---|
| RoboTurk | 1,796 | 2 | 2 |
| BC-Z | 39,350 | 9 | 17 |
| BridgeData V2 | 53,192 | 270 | 749 |
| Fractal | 87,212 | 6 | 13 |
| DROID | 92,233 | 194 | 907 |
| **Ours** | **45,157** | **313** | **1,217** |

**设置**：统一 π0 架构,**从零预训练**(而非微调公开 checkpoint)。仿真用 SIMPLER 的 BridgeData V2 环境(4 个 pick-and-place 任务,每任务 25 条成功 rollout 做后训练、200 条 rollout 评测);真机用 ALOHA 设计 4 个语言条件 pick-and-place 任务("Pick up the [carrot/onion] and place it into the [pot/bowl]",四物体同时在场,须正确理解指令),每任务采 200 episode(共 800)后训练、10 条 rollout 评测;真机用两段式打分(抓对物体 0.5 分 + 放对位置 0.5 分)。硬件 8×H200,AdamW/bf16,恒定 LR $5\times10^{-5}$,预训练 20,000 步、batch 1,024。

**主结果(Fig 5,平均成功率 %)**：Scratch 在真机为 0%(完全无法 ground 指令),预训练带来大幅提升;LAPA 虽在真机优于 scratch,但在仿真里掉点。SIMPLER 一栏 Ours=26.0 与下表 Table III($\delta_{BGTS}=0.7$)一致,其余为图中读数(近似)。

| 方法 | Real Robot (%) | SIMPLER (%) |
|---|---|---|
| Scratch | 0 | 25 |
| LAPA (SthV2) | ~34 | 21 |
| LAPA (Ours, 无动作标签) | ~38 | 18 |
| **Ours** | **55** | **26** |
| **Ours + BridgeData V2** | **68** | **27** |

**与真机数据集对比(Table II,真机成功次数 / 40)**：Ours 单用即优于 BC-Z、BridgeData V2,略逊于规模大得多的 Fractal;与 BridgeData V2 混合后超过任何单一数据集。

| 预训练数据集 | Carrot-Pot | Carrot-Bowl | Onion-Pot | Onion-Bowl | Total |
|---|---|---|---|---|---|
| BridgeData V2 | 4/10 | 3/10 | 6/10 | 4/10 | 17/40 |
| BC-Z | 5/10 | 5/10 | 4/10 | 5/10 | 19/40 |
| Fractal | 7/10 | 4/10 | 7/10 | 4/10 | 22/40 |
| **Ours** | 4/10 | 6/10 | 7/10 | 4/10 | **21/40** |
| **Ours + BridgeData V2** | 7/10 | 7/10 | 7/10 | 6/10 | **27/40** |

**规模消融(Fig 6)**：真机上随数据量单调涨——5K→40%、20K→44%、45K→55%;仿真上几乎持平(全量甚至比 0.1 子集低约 1%),作者归因于 visual domain gap。

**清洗阈值消融(Table III)**：$\delta_{BGTS}=0.7$ 在规模与质量间取得最佳折中。

| $\delta_{BGTS}$ | #Episodes | SIMPLER | Real Robot |
|---|---|---|---|
| 0.5 | 28,719 | 25.8 ± 5.4 | 53.8 ± 5.2 |
| **0.7** | **45,157** | **26.0 ± 5.9** | **55.0 ± 6.1** |
| 1.0 | 86,427 | 22.5 ± 4.8 | 38.8 ± 4.3 |

阈值放到 1.0 虽保留 8.6 万条,但噪声增多导致真机掉到 38.8%——**质量比数量更关键**。

## 四、局限性

- **仿真几乎不涨**:真机大涨(0→55),但 SIMPLER 只从 25 微升到 26,规模化在仿真里近乎无效。作者归因于 egocentric 视频的丰富视觉线索与简化仿真器之间的 domain gap——这也意味着该数据集的收益是否泛化到更多样真机场景仍待验证,四任务(carrot/onion × pot/bowl)本身较窄。
- **无 gripper / 无抓握时机监督**:动作仅 9 维物体位姿位移,丢失了开合手爪这一关键自由度,依赖后训练阶段的真机数据补上抓握,预训练学不到"何时闭合/张开"。
- **物体位姿 ≈ 末端执行器是强近似**:把被操作物体质心当作末端执行器状态,忽略了手与物体的接触关系、抓取点、以及非刚性/多物体交互;旋转由相邻点云 SVD 求得,对遮挡与深度噪声敏感。
- **管线依赖多个重模型**:GPT-4o + Grounding DINO + SAM + SpatialTracker + COLMAP 串联,任一环节出错都会污染轨迹,清洗只能靠两条粗粒度规则事后补救,124,559→45,157 意味着约 64% 被丢弃,产出率不高。
- **评测规模小**:真机每任务仅 10 条 rollout,统计误差大(标准误 5-6 个百分点),LAPA 各变体的图中读数差异需谨慎看待。

## 五、评价与展望

**优点**。这篇工作的价值在于给出一个干净的"控制变量"证据:在**统一 π0 架构与统一后训练设置**下,只换预训练数据源,证明了**不用任何辅助标注、直接从原始 egocentric 视频抽显式 6DoF 轨迹**就能把真机成功率从 0 拉到 55%,并与真机数据集持平、可与之互补叠加。相较 EgoMimic/EgoVLA/EgoDex 依赖手部姿态与专用硬件,它把数据获取门槛降到"只要 RGB 视频",可扩展性显著更好;相较 LAPA 的隐式潜动作,它用显式轨迹换来了更强的细粒度运动监督,并在真机上明显胜出。两条清洗规则(行程距离 + 背景轨迹相似度)简单有效,BGTS 用"物体是否与背景一起动"判检测错误的直觉很干净。

**缺点与存疑**。① 核心新颖性偏工程整合——轨迹抽取用的是作者前作 EgoScaler,本文主要贡献是"扩到四数据集 + 两条过滤 + 接到 π0",方法层面增量有限。② "与真机数据持平"的结论建立在极窄的四任务、每任务 10 rollout 上,且 Ours 仍逊于 Fractal,说明规模仍是硬约束;更公平的比较应在 OXE 全量或更长程任务上。③ 仿真几乎不涨这一点被轻描淡写为 domain gap,但它也可能暴露"物体轨迹当动作"这一近似的信息损失,值得更深入的诊断(例如仿真里换更真实渲染、或引入 gripper 监督后是否改善)。

**与其他公开工作的关系**。本文正处在"从人类视频学机器人"这一密集赛道:与 LAPA(隐式潜动作)、EgoVLA/EgoDex(富手部标注)、Phantom/Masquerade/ZeroMimic/VidBot(data-editing 或 affordance 蒸馏)形成对照,本文押注"**显式几何轨迹 + 规则清洗**"这一路。它与 UniSkill、EgoZero、MimicPlay 等共享"降低对遥操作依赖"的动机,但独特之处是把物体 6DoF 轨迹直接当 π0 的动作监督,而非中间的高层 plan 或 affordance。

**开放问题与改进方向**。(a) 补回 gripper/接触监督:从视频里估计手-物接触状态或抓取开合时刻,把动作从 9 维扩到含开合,预期能显著缩小与真机数据的差距;(b) 用手部轨迹与物体轨迹的联合信号替代"物体≈末端",更贴近双手/工具操作;(c) 把 domain gap 作为一等公民研究,例如在预训练中做视觉随机化或引入仿真渲染桥接;(d) 用更强的一致性/几何约束替代粗粒度阈值过滤,提高 64% 丢弃率下的数据产出;(e) 在长程、多阶段、双臂任务上验证,而非仅四个单臂 pick-and-place。总体上,这是一篇"把 egocentric 视频用显式轨迹接入主流 VLA"的扎实实证工作,结论(无辅助标注亦可、质量>数量、可与真机数据互补)对数据合成方向有直接参考价值。

## 参考

1. Yoshida et al., *Generating 6DoF Object Manipulation Trajectories from Action Description in Egocentric Vision* (EgoScaler), CVPR 2025 — 本文轨迹抽取的直接前作。
2. Ye et al., *Latent Action Pretraining from Videos* (LAPA), ICLR 2025 — 隐式潜动作的主要对照基线。
3. Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv 2410.24164, 2024 — 使用的策略架构。
4. Yang et al., *EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos*, arXiv 2507.12440, 2025 — 依赖手部姿态的富标注对照路线。
5. Grauman et al., *Ego4D / Ego-Exo4D*, CVPR 2022 / 2024 — 主要 egocentric 视频数据源。
