# 3D/4D WM Survey：3D 与 4D 世界建模综述

> **论文**：*3D and 4D World Modeling: A Survey*
>
> **作者**：Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Ziwei Liu, Wei Tsang Ooi, Steven C. H. Hoi, et al.（WorldBench Team）
>
> **机构**：WorldBench Team（论文首页仅标注该项目团队名，未逐一列出各作者所属机构）
>
> **发布时间**：2025 年 09 月（arXiv 2509.07996，本笔记依据 v3，2025 年 12 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.07996) | [PDF](https://arxiv.org/pdf/2509.07996)
>
> **分类标签**：`world-models` `3D/4D` `autonomous-driving` `occupancy` `LiDAR-generation` `survey`

---

## 一句话总结

这是第一篇**专门面向 native 3D/4D 表征**（RGB-D、occupancy grid、LiDAR 点云）的世界模型综述：它把"世界模型"这个混乱的术语正式拆成 generative 与 predictive 两种范式、四种功能角色（数据引擎 / 动作解释器 / 神经模拟器 / 场景重建器），并沿 **VideoGen / OccGen / LiDARGen** 三条模态主线整理了约 137 篇代表方法（63+40+34）、24 个数据集与一整套评测协议，指出当前领域的核心瓶颈是长时程保真、物理一致性、跨模态对齐与标准化评测的缺失。

## 一、问题与动机

世界建模（world modeling）已成为 AI 与机器人的基础任务——让智能体去理解、表示并预测其所处的动态环境。近年 VAE / GAN / diffusion / autoregressive 的进步极大丰富了生成与预测能力，但**绝大多数进展集中在 2D 图像或视频**。而真实世界本质上是 3D 且随时间演化（4D）的，很多场景需要能承载 metric geometry、visibility、motion 的 native 3D/4D 表征。作者提出三点动机：

- **2D 投影丢失了物理作用的坐标系**。native 3D/4D 信号直接编码几何、可见性与运动，是 actionable modeling 的"一等载体"——它天然携带多视一致性、刚体/非刚体运动学、场景级遮挡推理、地图拓扑一致性等约束，这些正是自动驾驶、机器人等 safety-critical 系统所必需的归纳偏置。
- **"world model" 术语高度混乱**。文献中有人把它窄化为传感器数据的生成模型（图像/视频），有人则宽泛到把预测式 forecasting、模拟器、决策框架都算进去，claim 之间互相矛盾、难以比较。
- **已有综述几乎只覆盖 2D 或纯视觉**，native 3D/4D 数据独有的挑战与机会被长期忽视，文献碎片化、缺乏统一框架与 taxonomy。

作者据此给出三点贡献：(1) 为"world model"与"3D/4D world modeling"建立精确定义；(2) 提出以表征模态为主轴的层级化 taxonomy；(3) 系统整理专为 3D/4D 场景定制的数据集与评测协议。作者明确把 native 3D/4D 与相邻的 video/panorama/mesh 世界模型、object-centric 3D 资产生成区分开——后两者提供外观与拓扑资产，而 native 3D/4D 提供几何锚定的动力学与交互。

## 二、核心方法

本综述的"方法"就是一套**定义 + 统一记号 + 分类框架**，下面按其骨架复现。

### 1. 表征与条件的统一记号

四类核心场景表征：

- **Video Streams**：$\mathbf{x}_v \in \mathbb{R}^{T\times H\times W\times C}$，强调几何连贯与时间一致，而非普通 2D 视频。
- **Occupancy Grids**：静态 $\mathbf{x}_o \in \{0,1\}^{X\times Y\times Z}$，时序版 $\mathbf{x}_o^t \in \{0,1\}^{T\times X\times Y\times Z}$；体素化几何天然施加空间约束，适合 physics-consistent 生成。
- **LiDAR Point Clouds**：$\mathbf{x}_l = \{(x_i,y_i,z_i)\}_{i=1}^N$，时序版加时间戳 $t_i$；直接捕捉几何，对纹理/光照/天气鲁棒。
- **Neural Representations**：NeRF（把射线 $\langle \mathbf{r},\mathbf{d} \rangle$ 映到颜色 $\mathbf{c}$ 与密度 $\sigma$）与 3D Gaussian Splatting，时序扩展支持 4D 重建与仿真。

条件（conditions）被归为三组（对应论文 Table 1）：

| 条件组 | 记号 | 典型信号 |
|---|---|---|
| 几何 Geometry | $\mathcal{C}_{geo}$ | camera pose、depth map、BEV/HD map、3D bbox、flow field、past occupancy、LiDAR pattern、partial point cloud、RGB frame、surface mesh |
| 动作 Action | $\mathcal{C}_{act}$ | ego-trajectory / velocity / acceleration / steering / command、route plan、action token、scan path |
| 语义 Semantics | $\mathcal{C}_{sem}$ | semantic mask、text prompt、scene graph、object label、weather tag、material tag |

### 2. 两大范式的形式化定义

**生成式世界模型（Generative）**——从零或从部分观测合成 plausible 场景：

$$\mathcal{G}(\mathbf{x}_i, \mathcal{C}_{geo}, \mathcal{C}_{act}, \mathcal{C}_{sem}) \to \mathcal{S}_g$$

其中 $i \in \{\varnothing, v, o, l\}$（噪声/视频/occupancy/LiDAR），输出 $\mathcal{S}_g$ 是生成的 3D/4D 场景。

用大白话说：给一堆"想要什么样"的条件（相机怎么摆、车怎么开、场景语义是什么），模型"画"出一个符合这些约束的三维世界。

**预测式世界模型（Predictive）**——基于历史观测、在给定动作下 forecast 未来演化：

$$\mathcal{P}(\mathbf{x}_i^{-t:0}, \mathcal{C}_{act}) \to \mathcal{S}_p^{1:k}$$

即用过去 $t$ 步观测加动作条件，预测未来 $k$ 步的场景 $\mathcal{S}_p^{1:k}$。

用大白话说：给模型看"过去几秒发生了什么 + 我接下来打算怎么动"，它推演出"接下来几秒世界会变成什么样"。前者是"想象世界"，后者是"预判世界"。

### 3. 四种功能角色（把"模型消费什么"与"模型做什么"解耦）

作者指出领域的一个常见混淆是把"条件（consumes）"与"功能（does）"搅在一起，于是按功能拆出四类：

1. **Data Engines（数据引擎）**：输入 $\mathcal{C}_{geo}/\mathcal{C}_{act}/\mathcal{C}_{sem}$，输出 $\mathcal{S}_g$；关注 plausibility 与 diversity，用于大规模数据增强与场景构造。
2. **Action Interpreters（动作解释器）**：输入历史 $\mathbf{x}_i^{-t:0}$ 与 $\mathcal{C}_{act}$，输出预测序列 $\mathcal{S}_p^{1:k}$；支持 action-aware forecasting，服务轨迹规划、行为预测、策略评估。
3. **Neural Simulators（神经模拟器）**：输入当前场景状态 $\mathcal{S}_g^t$ 与 agent 策略 $\pi_{agent}$，输出下一状态 $\hat{\mathcal{S}}_g^{t+1}$；支持闭环 policy-in-the-loop 交互仿真。
4. **Scene Reconstructors（场景重建器）**：从 partial/sparse/corrupted 观测 $\mathbf{x}_i^p$ 加 $\mathcal{C}_{geo}$ 补全出完整场景 $\hat{\mathcal{S}}_g$；服务高保真建图、digital twin 修复、事后分析。

### 4. 生成骨干（algorithmic backbone）

综述简明列出四类生成家族及其权衡：**VAE**（ELBO 目标，训练稳、latent 可解释，但样本偏糊）、**GAN**（min-max 博弈，高保真但训练不稳、mode collapse）、**Diffusion**（学习逆噪声过程，稳定且样本质量高，但迭代采样慢）、**Autoregressive**（$p(\mathbf{x})=\prod_i p(x_i \mid x_{<i})$，精确似然、灵活但生成慢）。Diffusion 的训练目标为：

$$\mathbb{E}_{\mathbf{x},\epsilon,t}\big[\lVert \epsilon - \epsilon_\theta(\mathbf{x}_t,t) \rVert^2\big]$$

用大白话说：给干净数据一点点加噪声、再训练网络"把每一步噪声猜回来"，采样时从纯噪声反向去噪即可生成场景。作者强调进入 native 3D/4D 域后，scalability、controllability、multi-modal 融合这些权衡被放大，是构建可靠具身世界模型的关键。

### 5. 三条模态主线的层级 taxonomy（Sec. 3）

这是综述的正文主体，每条主线再按功能分类。以下列代表性方法（均来自论文正文与 Table 2/3/4）：

**VideoGen（视频生成，63 篇）**
- Data Engines：BEVGen、BEVControl、MagicDrive、SyntheOcc、PerLDiff（感知增强）；DiVE、MagicDrive-V2、Cosmos-Drive、Glad、STAGE（长时程）；Delphi、DriveDreamer-2、Nexus、Challenger（规划导向的稀有/corner-case 挖掘）；WoVoGen、SimGen、DrivePhysica、GeoDrive（场景编辑与风格迁移）。
- Action Interpreters：GAIA-1/GAIA-2、GenAD、Vista、GEM、MaskGWM、InfinityDrive、Epona、DrivingWorld（action-guided 生成）；Drive-WM、DriveDreamer、ADriver-I（forecast-driven 规划）；DrivingGPT、Doe-1、VaVAM、ProphetDWM（GPT 式 next-token 联合建模）。
- Neural Simulators：DriveArena、DreamForge、DrivingSphere、UMGen（生成驱动闭环）；StreetGaussian、HUGSIM、UniSim、OmniRe、ReconDreamer、Stage-1（重建中心，NeRF/3DGS）。

**OccGen（occupancy 生成，40 篇）**
- Scene Representers：SSD、SemCity、UrbanDiff、DrivingSphere、UniScene——把 occupancy 当作几何一致的中间体增强感知鲁棒性、并为高保真视频/LiDAR 提供 3D 先验。
- Occupancy Forecasters：Emergent-Occ、UnO、UniWorld、DriveWorld（把 4D occupancy forecasting 当作自监督预训练 pretext）；OccWorld、OccSora、Cam4DOcc、OccLLaMA、Occ-LLM、UniOcc（ego-conditioned 可控预测）。
- AR Simulators：PDD、XCube、InfiniCube、X-Scene（scalable open-world 生成）；OccSora、DynamicCity（长时程 4D，16s 级序列）。

**LiDARGen（LiDAR 生成，34 篇）**
- Data Engines：DUSty/DUSty-v2（GAN）、LiDARGen（首个 Langevin/score-based）、R2DM、R2Flow、LiDM/RangeLDM/3DiSS（latent diffusion）、LiDARGRIT、SDS、SPIRAL、La La LiDAR、Veila（感知增强）；UltraLiDAR、LiDiff、DiffSSC、LiDPM、Distillation-DPO、SuperPC（场景补全）；Text2LiDAR、WeatherGen、OLiDM（稀有条件）；X-Drive（多模态 LiDAR+图像联合）。
- Action Forecasters：Copilot4D（VQ-VAE tokenize + MaskGIT 离散扩散）、ViDAR、BEVWorld、DriveX、HERMES。
- AR Simulators：HoloDrive、LiDARCrafter、LidarDM（mesh-based 长序列 LiDAR 仿真）。

## 三、实验结果

综述用五个视角组织评测（Generation / Forecasting / Planning-Centric / Reconstruction-Centric / Downstream），并汇总了大量 SOTA benchmark（均取自论文 Table 5–14，主流用 nuScenes / SemanticKITTI）。关键数字如下。

**数据集与规模**：Table 5 覆盖 24 个数据集，核心真实数据为 KITTI、nuScenes、Waymo Open、SemanticKITTI、Argoverse 2、Occ3D-nuScenes、OpenOccupancy、NAVSIM，模拟平台 CARLA/CarlaSC 提供可编辑布局与 counterfactual；作者强调"真实数据供 realism、模拟数据供 perfect label 与稀有场景"两者互补。

**VideoGen 感知保真（nuScenes val，FID/FVD 越低越好，Table 6）**：

| 设置 | 方法 | FID ↓ | FVD ↓ |
|---|---|---|---|
| 单视 | MaskGWM | 4.00 | 59.40 |
| 单视 | GeoDrive | 4.10 | 61.60 |
| 单视 | Vista | 6.90 | 89.40 |
| 单视 | DriveDreamer（早期基线） | 14.90 | 340.80 |
| 多视 | DiST-4D | 6.83 | **22.67** |
| 多视 | MiLA | 4.90 | 36.30 |
| 多视 | UniMLVG | 5.80 | 36.10 |
| 多视 | DrivePhysica | 3.96 | 38.06 |

作者总结：早期 BEV 类（BEVControl/BEVGen）FID 普遍 $>20$；引入几何一致与时空推理的 UniScene、DiST-4D 取得最佳平衡，多视 FVD 可低至 22.67，但时间连贯仍是难点。

**OccGen 重建质量（nuScenes val，越高越好，Table 9）**：Triplane 因式分解优势明显——**X-Scene 达 92.40% mIoU / 85.60% IoU**，T³Former 85.50 / 72.07；VAE 类 DOME 83.08% mIoU / 77.25% IoU、UniScene 92.10 / 87.00；而 OccSora 在激进时空压缩下显著退化。结论：latent 表征设计（triplane 强于单纯放大 latent 维度）对重建保真是决定性的。

**OccGen 4D forecasting（mIoU %，@1s，Table 10）**：I²World **47.62**、T³Former 46.32、UniScene 35.37、DOME 35.11；naive 自回归/生成方法在长时程快速退化，triplane 因式分解显著改善空间保真。

**OccGen 规划（nuScenes，L2 误差 m / 碰撞率 %，越低越好，Table 11）**：把 occupancy 世界模型接入规划一致优于纯轨迹方法——Occ-LLM 平均 L2 仅 **0.28 m**（@1s 0.12 m），Drive-OccWorld 达 0.85 m L2 / 0.29% 碰撞，均优于 UniAD（0.69 m）与序列基线 ST-P3（2.11 m）。

**LiDARGen 保真（SemanticKITTI，FRD/FPD/JSD/MMD，越低越好，Table 12）**：WeatherGen 借 Mamba backbone 取得最佳综合（FRD 184.11 / FPD 11.42 / JSD 0.0290 / MMD 3.80×10⁻⁵）；有趣的是 Text2LiDAR 虽文本可控性强，但过度贴合语义 prompt 反而牺牲几何保真（FRD 522.32）。

**LiDARGen 4D 时间一致（nuScenes，TTCE/CTC，越低越好，Table 13）**：LiDARCrafter TTCE 2.65（@3 帧）/ CTC 系列最优，UniScene 与 OpenDWM-DiT 在短时程几何一致上有优势，但固定长度生成限制了长时程外推。

**下游一致结论**：光真实的生成本身不足以提升 detection/segmentation/planning——显式建模几何、时间一致与运动动力学的模型（如 DiST-4D、UniScene）才能同时改善感知与规划安全，且合成数据与真实数据的任务级差距仍显著。

## 四、局限性

作者在 Sec. 6 系统列出领域（而非单一方法）的开放挑战，本身也构成本综述的局限视角：

- **缺乏标准化 benchmark**：各家用不同数据集/ad-hoc 指标，难以公平比较真实性能；作者呼吁统一涵盖 physical plausibility、temporal consistency、controllability 的闭环 + 真实场景协议。
- **长时程高保真难**：短时程尚可，但小误差随序列累积导致行为不真实、场景退化，多智能体与环境因素持续演化更放大此问题。
- **物理保真 / 可控性 / 泛化不足**：常产生非形变碰撞、错误阴影、尺度畸变等 physically implausible 事件；编辑能力粗糙（多限于调整交通参与者位置/外观，难以细粒度控制建筑、路标）；且倾向 overfit 训练数据，难泛化到新城市与稀有目标。
- **计算效率与实时性**：重型架构 + 多步采样带来高延迟与显存开销，制约大规模生成与仿真。
- **跨模态一致性**：视觉/几何/语义联合生成常出现与底层 3D 结构冲突的失配，损害下游可靠性。
- 综述自身层面：覆盖以**自动驾驶**场景为绝对主导，室内操作/通用机器人操纵仅在应用章节 5.2 简述，未深入；且是纯 taxonomy 式梳理，未提供统一再实现的开源 benchmark（仅项目页汇总链接）。

## 五、评价与展望

**优点**：这是该细分方向第一篇成体系的综述，最大价值在于**概念清晰化**——把"world model"从含糊口号拆成 generative/predictive 两范式 + 四功能角色，并用 $\langle \mathcal{C}_{geo},\mathcal{C}_{act},\mathcal{C}_{sem} \rangle$ 条件三元组把"模型消费什么"与"模型做什么"解耦，这一 conditions-vs-functions 的分离比多数只按"输入/架构"堆砌的综述更有解释力。三条模态主线（VideoGen/OccGen/LiDARGen）× 功能的二维网格，配合 Table 2/3/4 的逐方法属性表和 Table 14 巨型指标目录，作为该领域的入门地图与查阅手册相当称职。它也纠正了一个重要认知：**occupancy forecasting 与 LiDAR forecasting 天然是自监督预训练的 pretext task**（UnO、DriveWorld、UniWorld、Copilot4D 都印证这一点），这对几何锚定的表征学习有普适启发。

**不足与开放问题**：(1) 覆盖面严重偏向自动驾驶，与 2D 视频世界模型综述（如 Genie 类、通用 video-as-world-model 一系）以及机器人 manipulation 的 world model（Dreamer 系、iVideoGPT、机器人 latent action 一系）的连接较弱，读者难以据此判断"驾驶 native 3D/4D"经验能否迁移到室内操作。(2) 定量结论多是"谁 SOTA"的横向罗列，缺少对为何 triplane 因式分解普遍胜出、diffusion vs. autoregressive 在 3D 上究竟如何权衡的机制性归因（虽有点到，但未展开）。(3) 未提供可复现的统一评测代码或 leaderboard，"标准化 benchmark"既是它指出的痛点、也是它自己没补上的坑。

**可能的改进方向**（纯学术视角）：把 predictive 世界模型的 forecasting 目标与 policy learning 更紧地耦合（当前 Action Interpreter 与 Neural Simulator 仍偏松散）；用 occupancy/LiDAR 的几何约束显式正则视频生成以缓解 physically implausible 事件；以及建立跨模态（视频⇄occupancy⇄LiDAR）一致性的联合训练与评测协议，让"同一世界的三种观测"真正对齐。总体而言，这是进入 3D/4D 世界模型必读的 map，但要落到具体方法选型，仍需回到它索引的原始论文。

## 参考

1. Zheng et al. *OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving*（ECCV 2024）— OccGen forecasting 代表，联合建模 ego motion 与环境 3D occupancy 演化。
2. Hu et al. *GAIA-1: A Generative World Model for Autonomous Driving*（arXiv 2023）— action-guided 视频世界模型开山之作。
3. Gao et al. *Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability*（NeurIPS 2024）— 高保真、多样可控的驾驶视频世界模型，含 uncertainty-aware reward。
4. Zhang et al. *Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion*（ICLR 2024）— LiDAR 世界模型代表，VQ-VAE tokenize + 离散扩散。
5. Min et al. *DriveWorld / UniWorld* 系列（CVPR 2024）— 把 occupancy forecasting 作为大规模自监督预训练、可微调至检测与规划。
