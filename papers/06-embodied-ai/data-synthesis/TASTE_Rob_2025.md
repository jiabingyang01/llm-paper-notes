# TASTE-Rob：面向任务的手-物交互视频生成推进可泛化机器人操作

> **论文**：*TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction for Generalizable Robotic Manipulation*
>
> **作者**：Hongxiang Zhao, Xingchen Liu, Mutian Xu, Yiming Hao, Weikai Chen, Xiaoguang Han（通讯）
>
> **机构**：SSE, CUHKSZ（香港中文大学深圳）；FNii, CUHKSZ（未来智能网络研究院）
>
> **发布时间**：2025 年 03 月（arXiv 2503.11423，v2 于 2025 年 6 月）
>
> **发表状态**：未录用（预印本，PDF 未标注 venue）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.11423) | [PDF](https://arxiv.org/pdf/2503.11423)
>
> **分类标签**：`手物交互视频` `视频扩散生成` `模仿学习数据` `手姿势精修`

---

## 一句话总结

针对现有 ego-centric 手-物交互（HOI）数据集视角漂移、语言-动作错位的问题，作者自采了 **100,856** 条固定视角、单动作、指令严格对齐的 HOI 视频数据集 TASTE-Rob，并提出"粗生成 → 手姿势精修 → 重生成"的三阶段流水线；相比通用视频扩散模型，FVD 从 48.72 降到 **9.43**，抓取类型误差（GTCE）从 67.8% 降到 **9.7%**，下游 Mujoco 机器人操作成功率从 84% 升到 **96%**。

## 一、问题与动机

机器人模仿学习（IL）可以直接从视频示范中学动作,但传统方式要求执行环境与录制视频几乎一致,泛化性差。一条可扩展的思路是用生成模型（尤其是视频扩散模型 VDM）来"造"示范视频:直接生成机器人-物体交互视频受限于机器人数据规模,于是转向利用海量人类操作数据来生成**手-物交互（HOI）视频** 作为示范。

但现有大规模 ego-centric HOI 数据集（以 Ego4D 为代表,含 83,647 条 ego-centric 交互视频）用于该目标有两个致命缺陷:

1. **视角不一致**:每条 clip 内相机视角会漂移,而 IL 示范通常需要**固定视角**录制。
2. **语言-动作错位**:clip 是从涵盖多任务的长视频里切出来的,部分 clip 只包含与相邻片段重叠的**局部交互**,导致视频与语言指令不对应。

这两点都会拖垮视频生成质量。作者实测发现,即使用强大的通用 VDM,也难以生成高质量、任务理解准确的 HOI 视频。因此需要一个专为该任务定制的数据集 + 一套能保证手姿势物理可行的生成方法。

## 二、核心方法

### 2.1 TASTE-Rob 数据集

作者自采 100,856 对"视频 + 语言指令",三条设计目标:固定相机视角 + 单动作严格对齐指令;环境与任务多样;手姿势多样。关键采集约束:

- 广角镜头拍摄 **1080p** ego-centric 视频,录制中**不发生视角漂移**,且视角对齐 Ego4D 头戴相机设置。
- **受控采集协议**:每条视频严格 **短于 8 秒**、只含单个动作;采集者按"开始录制 → 按指令执行 HOI → 完成即停"的结构化流程,保证动作与指令精确对应。
- 多样性:环境覆盖厨房/卧室/餐桌/办公桌等;任务含 pick/place/push/pour 等;其中 **75,389** 条单手任务、**25,467** 条双手任务。
- 手姿势多样性用 HaMeR 提取手参数分析:掌心朝向分布(Table 1)以朝下(0–180°)为主(0–90° 占 33.82%,90–180° 占 47.7%),并统计了指间夹角、手指曲率分布。

Table 2 中,TASTE-Rob 是唯一同时满足 **Perfect Action Alignment（动作严格对齐指令）** 与 **Determiner（含物体描述限定词）**、且采用 **Static** 相机的数据集(9M 帧 / 100,856 clips / 1080p)。

### 2.2 三阶段姿势精修流水线

给定环境图 $i$ 和任务描述 $\mathcal{T}$,生成的 HOI 视频需满足两点:(1) **Accurate Task Understanding**——正确识别操作哪个物体、怎么操作;(2) **Feasible HOI**——全程维持一致、可行的抓握姿势。作者发现:单个 VDM 能做到任务理解准确,但抓握姿势会随时间"抽搐"(如某帧突然变成捏合手势),缺乏运动连贯性。故拆成三阶段。

**基座——视频扩散(DynamiCrafter)**。设 $\mathcal{T},\ i \in \mathbb{R}^{3\times H\times W},\ v \in \mathbb{R}^{L\times 3\times H\times W}$ 分别为指令、环境图、真值视频帧。在潜空间训练去噪目标:

$$\min_{\theta_V}\ \mathbb{E}_{t,z,\epsilon\sim\mathcal{N}(0,I)}\ \|\epsilon-\epsilon_\theta(z_t,\mathcal{T},i,t,fps)\|_2^2$$

**用大白话说**:把视频压进潜空间,学一个网络在指令+首帧环境图+帧率条件下,从噪声一步步"擦"出连贯视频帧。

**Stage I:粗动作规划器**。在 TASTE-Rob 上微调 DynamiCrafter,只微调 image context projector 和去噪 U-Net 的 spatial layers,得到粗 HOI 视频 $\hat v_c \in \mathbb{R}^{L\times 3\times H\times W}$。此阶段解决"任务理解",但抓握姿势不稳。

**用大白话说**:先让模型看懂"该抓桌上哪个杯子往盘子里放",粗略画出一段动作视频,手指抖不抖先不管。

**Stage II:手姿势序列精修(Image-to-Hand-Params MDM)**。用 MediaPipe 从粗视频里抽出 2D 手关键点序列 $p_c \in \mathbb{R}^{L_p\times N_h\times 2}$（$L_p$ 序列长度,$N_h$ 关键点数,归一化坐标）,再用一个可学习的 Motion Diffusion Model（MDM）$\mathcal{M}$ 精修。MDM 直接预测干净动作序列(而非单步噪声),在原 MDM 上加了一条**环境图像条件分支**（用 CLIP 图像类特征）,目标为:

$$\min_{\theta_\mathcal{M}}\ \mathbb{E}_{t,\mathbf{p},\epsilon\sim\mathcal{N}(0,I)}\ \|p_0-\mathcal{M}(p_t,\mathcal{T},i,t)\|_2^2$$

关键技巧:不从高斯噪声重新生成,而是**用 $p_c$ 作为初始化**,令 $p_{0,N_{rv}}=p_c$,只做 $N_{rv}$ 步去噪。这样既保留 $p_c$ 的空间感知(手在哪),又通过扩散修复运动可行性(手怎么连贯地动),得到精修序列 $\hat p$。

**用大白话说**:把粗视频里那双"抽搐的手"抠出坐标轨迹,交给一个专修人手运动的模型小修一下——不是推倒重来,而是在原轨迹基础上去掉抖动、补上物理合理的连贯动作。$N_{rv}$ 修多少步是关键旋钮:修太少手还抖,修太多手就"飘"离物体位置。

**Stage III:带精修姿势重生成**。仿 ToonCrafter,训练一个逐帧独立的姿势编码器 $\mathcal{S}$（类似 ControlNet 的注入方式）来控制手姿。把 $\hat p$ 可视化为手姿图像序列 $s^i$,逐帧注入特征 $F^i_{inject}=\mathcal{S}(\hat p, s^i, t)$。此阶段**冻结 Stage I 的 $\mathcal{V}$,只训练 $\mathcal{S}$ 的参数 $\eta$**:

$$\min_{\eta}\ \mathbb{E}_{t,s,\mathcal{E}(v),\epsilon\sim\mathcal{N}(0,I)}\ \|\epsilon-\epsilon_\theta^{\mathcal{S}}(z_t;\mathcal{T},i,s,t,fps)\|_2^2$$

**用大白话说**:把修好的手轨迹当作"骨架控制条"再喂回视频模型,让它重画一遍——这次任务理解和手姿势都对了。生成的精细 HOI 视频 $\hat v$ 直接当作 IL 示范,用 Im2Flow2Act 策略模型驱动机器人操作。

## 三、实验结果

**设置**:Stage I 微调 30K 步(bs=16,lr=5e-5);Stage II 训练 MDM 100K 步(bs=64,lr=1e-4);Stage III 微调姿势编码器 30K 步(bs=32,lr=5e-5);推理 50 步去噪,姿势精修 $N_{rv}=10$。评测集为 TASTE-Rob-Test(每动作类别取 2%)与 Im2Flow2Act 提供的 Mujoco 仿真集(50 个任务)。指标:视频质量 FVD/KVD/PIC;抓握一致性 GPV(抓握姿势方差)/GTCE(抓握类型分类误差)/HMDA(手运动方向与目标轨迹的方向误差,以度计,越低越好);机器人操作成功率。

**与通用 VDM 对比(视频质量,TASTE-Rob-Test)**:

| 方法 | KVD↓ | FVD↓ | PIC↑ |
|---|---|---|---|
| ConsistI2V | 0.24 | 65.77 | 0.85 |
| DynamiCrafter | 0.21 | 62.36 | 0.79 |
| Open-Sora Plan | 0.18 | 50.19 | 0.84 |
| CogVideoX | 0.16 | 48.72 | 0.85 |
| **TASTE-Rob（本文）** | **0.03** | **9.43** | **0.90** |

**数据集价值消融(同一 DynamiCrafter,仅训练集不同)**:在 Ego4D 上微调的 Ego4D-Gen 对比在 TASTE-Rob 上微调的 Coarse-TASTE-Rob:

| 方法 | KVD↓ | FVD↓ | PIC↑ |
|---|---|---|---|
| Ego4D-Gen | 0.18 | 52.17 | 0.77 |
| Coarse-TASTE-Rob | **0.04** | **10.85** | **0.88** |

说明数据集本身(固定视角 + 严格对齐)就带来巨大质量跃升。

**姿势精修流水线消融(Coarse=仅 Stage I,TASTE-Rob=完整三阶段)**:

| 指标 | Coarse-TASTE-Rob | TASTE-Rob |
|---|---|---|
| KVD↓ | 0.04 | **0.03** |
| FVD↓ | 10.85 | **9.43** |
| PIC↑ | 0.88 | **0.90** |
| GPV↓ | 0.28 | **0.24** |
| GTCE↓ | 67.8% | **9.7%** |
| HMDA↓ | 26.4° | **11.3°** |
| 操作成功率↑ | 84% | **96%** |

姿势精修对视频质量提升有限,但对**抓握一致性提升极大**（GTCE 67.8%→9.7%,HMDA 26.4°→11.3°）,并把下游机器人操作成功率从 84% 抬到 96%。

**$N_{rv}$ 权衡**:$N_{rv}=0$（不精修）手姿势不一致;$N_{rv}=50$（从高斯噪声生成）空间感知差、手飘离物体。随 $N_{rv}$ 增大,姿势一致性变好但空间感知变差,视频质量先升后降,最终取 $N_{rv}=10$。

## 四、局限性

- 作者明确指出:当物体发生**显著形变/位姿变化**（如旋转、开抽屉/开门）时,生成质量明显下降——这类需要长程物体状态推理的操作是其软肋。
- 手姿势精修依赖 MediaPipe 抽 2D 关键点 + MDM 修复,只建模了 **2D 归一化手关键点**,未显式建模 3D 手-物接触与物体几何,遮挡严重时抽取的姿势本身就不可靠。
- 下游验证仅在 **Mujoco 仿真** 上,通过 Im2Flow2Act 间接评估,缺少真机验证;成功率对比也只在自身数据/流水线内做消融。
- 数据全部由**右手为主** 的采集者录制,掌心朝向分布明显偏斜,可能限制对左手/非常规抓握的泛化。
- 三阶段级联(VDM→MDM→VDM)推理链条长,误差会逐级传递,且没报告推理耗时。

## 五、评价与展望

**优点**:(1) 数据侧的洞察很扎实——把"IL 需要固定视角 + 语言严格对齐"这一朴素但被忽视的需求,做成一个 10 万级、专门定制的 HOI 数据集,Table 4 的消融清楚地证明了"数据即杠杆"(仅换训练集,FVD 就从 52.17 到 10.85)。(2) 三阶段的分工干净利落:把"任务理解"(VDM 擅长)和"手姿势物理可行"(MDM 擅长)解耦,用 MDM 作为轻量"运动修复器",而非重训一个联合大模型,工程上性价比高。(3) 用 $p_c$ 初始化 MDM 去噪、只修 $N_{rv}$ 步的做法,在"保留空间锚点"与"修复运动连贯"之间给出了一个可调的显式旋钮,是本文最漂亮的小技巧。

**与其他公开工作的关系**:与直接生成机器人-物体视频的路线（如 Im2Flow2Act、UniPi 等文本引导视频策略）相比,本文走"人手 HOI 视频→人手示范"的路,数据来源更廉价可扩;与 Track2Act、Im2Flow2Act 这类"从视频抽轨迹/光流再驱动策略"的工作是互补的——TASTE-Rob 负责造高质量示范视频,下游仍需一个 human-to-robot 的动作迁移策略(本文借 Im2Flow2Act)。基座上是 DynamiCrafter + MDM(Tevet 等)+ ToonCrafter/ControlNet 式控制的组合创新,单个组件都不新,但拼装解决 HOI 手姿势抖动的思路是清晰有效的。

**开放问题与可改进方向**:(1) 手-物接触与物体形变缺失是最大短板,可引入 3D 手模型(MANO)、接触/受力约束或物体 6D 位姿条件,直面 limitation 中的旋转/开合类任务;(2) 三阶段级联可探索端到端联合训练或用单个带姿势控制的扩散模型统一,减少误差传递;(3) 评测应补真机迁移与更强的跨环境泛化基准,当前 Mujoco+Im2Flow2Act 的验证偏窄;(4) 双手协同任务(数据里已占 25%)的显式建模与评测尚未展开,是一个自然的扩展点。总体而言,这是一篇"数据 + 轻量流水线"驱动的扎实工作,核心贡献在数据集与解耦式手姿势精修,而非生成模型本身的突破。

## 参考

1. Xing et al., *DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors*, 2024 —— 本文 Stage I/III 的 I2V 扩散基座。
2. Tevet et al., *Human Motion Diffusion Model (MDM)*, ICLR 2023 —— Stage II 手姿势精修所用运动扩散骨架。
3. Grauman et al., *Ego4D: Around the World in 3,000 Hours of Egocentric Video*, 2022 —— 主要对比与被批评的数据集基线。
4. Xing et al., *ToonCrafter*, 2024 —— Stage III 逐帧姿势编码器 $\mathcal{S}$ 的控制注入思路来源。
5. Wang et al., *Im2Flow2Act: Learning Robot Manipulation via ... Flow*, 2024 —— 下游机器人操作与 Mujoco 评测所用策略模型。
