# EgoHOI：面向照片级真实感手物交互合成的第一人称世界模型

> **论文**：*Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis*
>
> **作者**：Dayou Li, Lulin Liu, Bangya Liu, Shijie Zhou, Jiu Feng, Ziqi Lu, Minghui Zheng, Chenyu You, Zhiwen Fan
>
> **机构**：Texas A&M University、University of Minnesota、University of Wisconsin–Madison、University of California, Los Angeles、University of Texas at Austin、Amazon、State University of New York at Stony Brook
>
> **发布时间**：2026 年 03 月（arXiv 2603.13615）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.13615) | [PDF](https://arxiv.org/pdf/2603.13615)
>
> **分类标签**：`第一人称世界模型` `手物交互合成` `物理先验蒸馏` `视频扩散DiT` `HOT3D评测`

---

## 一句话总结

EgoHOI 是一个不依赖未来物体轨迹等特权信息、仅凭用户动作信号（3D 手部运动学 + 度量级头部相机运动）从首帧因果推演第一人称手物交互视频的世界模型，通过把手部运动学、度量级自运动、物体身份三类"物理先验"蒸馏成轻量 embedding 注入冻结的 Wan-DiT 视频骨干，在自建的 HOT3D 评测集上相较 Wan / Cosmos-2B / Cosmos-14B / Uni3C 基线大幅提升（PSNR 14.78→21.05，手部预测缺失率 MR 19.67%→5.84%，物体位置误差 OPE 0.078～0.168→0.015）。

## 一、问题与动机

第一人称人-物交互（HOI）世界模型旨在从第一人称视角、依据用户控制预测物理合理的交互动态演化，其 rollout 可作为灵巧操作、VLA 模型、时空推理、physical intelligence、数据增强等下游任务的可扩展训练/评测数据源。但建模这类世界模型极具挑战：头部运动剧烈、遮挡严重，且手部高自由度关节会突然改变接触拓扑和物体状态。作者指出，即便是最新的视觉语言模型（如 VLM4D）也难以可靠理解第一人称 HOI 视频中的复杂动态，说明"理解"这类交互已属不易，"生成"满足动作、物理规律、动力学一致性的未来帧则更难。

论文将现有 pipeline 的局限归纳为两点：其一，现有 HOI 视频生成主要研究静态、第三人称视角，不直接支持带动态交互的第一人称 rollout；其二，许多方法严重依赖特权的未来物体信息（如 ground-truth 轨迹、waypoint），显式规定物体将如何运动，本质上把任务变成了 conditional video generation，从而回避了"接触驱动动力学推理"这一核心难题。理想情况下，世界模型应仅凭当前观测和用户指定动作预测未来手物交互与状态，而非重放特权的未来物体轨迹。作者进一步指出，一旦去掉这种未来状态捷径，单纯扩大 RGB 训练数据规模并不足以保证物理准确性，因此需要引入显式的度量与运动学先验来正则化生成过程。

## 二、核心方法

**世界模型的一般形式化**。内部状态 $x_t$ 在动作 $a_t$ 驱动下演化，观测函数把内部状态映射回可见画面：

$$x_{t+1} = \mathbf{f}_\theta(x_t, a_t), \qquad o_t \sim g_\phi(x_t)$$

用大白话说：动作被当作干预（intervention），世界模型要预测动作对世界状态的因果效应，而不只是重建智能体已经走过的运动轨迹。

**Latent diffusion 骨干**。EgoHOI 以 Wan 2.1 14B 视频 DiT 为冻结骨干，用标准的前向加噪 $q(z_t\mid z_{t-1})=\mathcal{N}(z_t;\sqrt{1-\beta_t}\,z_{t-1},\beta_t \mathbf{I})$ 和噪声预测目标训练：

$$\mathcal{L} = \mathbb{E}_{z_0,\varepsilon,t}\left[\left\|\varepsilon-\varepsilon_\theta(z_t,t)\right\|_2^2\right]$$

**第一人称 HOI 的 latent 世界建模**。给定首帧 $\mathbf{I}_1$ 编码得到初始 latent $z_1$，每步动作信号由手部运动学 $\mathbf{H}^t$ 与度量级头部运动 $\mathbf{C}^t$ 构成：

$$z_{t+1} = \mathbf{f}_\theta(z_t, \mathbf{H}^t, \mathbf{C}^t), \qquad t=1,\dots,L$$

用大白话说：不给"物体下一步在哪"这种作弊信息，只给"手怎么动、头怎么动"，逼模型自己推断物体会如何随之变化。

为了在不依赖特权未来物体状态的前提下正则化这一高自由度、多耦合效应的 latent 演化，作者从互补的三个来源蒸馏出三类**物理先验 embedding**：

1. **Hand Kinematic Embeddings（HKE）**：把重建的 3D 手部 mesh 逐帧渲染成 hand-kinematics map $H_t\in\mathbb{R}^{S\times S\times3}$（$S=480$），堆叠 $L=81$ 帧后用时序 3D 卷积栈（SiLU 激活，通道逐步升维到共享的 5120 维 latent token 空间）编码；另设一个 6 层 2D 卷积金字塔构成的 reference-hand 分支，单独编码首帧手部姿态 $\mathbf{P}^1$ 以稳定手部身份、减少时序漂移。密集的网格渲染比稀疏坐标向量能提供更细粒度的接触结构信息。

2. **Ego-Motion Embeddings（EME）**：把内参 $\mathbf{K}$、相对首帧的外参 $\mathbf{E}=[\mathbf{R}:\mathbf{t}]$ 转成逐像素 Plücker 射线场，每像素 embedding 为 6 维向量 $\mathbf{p}_{u,v}=(\mathbf{o}\times\mathbf{d}_{u,v},\mathbf{d}_{u,v})$，其中方向由下式给出：

$$\mathbf{d}_{u,v} = \mathbf{R}\,\mathbf{K}^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix} + \mathbf{t}$$

用大白话说：与其把相机位姿压缩成一个低维姿态向量，不如用一张"每个像素对应哪条 3D 射线"的图来表示视角——这种表示对场景坐标原点选取不变，还保留了空间分辨的度量结构，能对视角演化施加更细粒度的约束。堆叠后的射线场经三层时序因果 3D 卷积下采样，再交替用 2D 残差卷积与沿时间维的自注意力做混合时空处理，最终投影、patchify 成 ego-motion token（$N_{\text{EME}}=21\times30\times30$）。

3. **Object Entity Embeddings（OEE）**：用现成分割模型得到的首帧物体分割图，经冻结的视频 VAE 编码、再用 3D 卷积 patchifier 转成 object-entity token，作为整个 rollout 中不变的"物体身份锚点"，专门缓解快速自运动和部分遮挡下物体颜色、纹理、细节漂移的问题。

**latent 空间集成方式**（均以冻结 Wan-DiT 为主干、外挂轻量 adapter）：HKE 通过可学习门控的残差方式融进主 token 流 $\mathbf{X}_0=\mathbf{T}_{\text{main}}+\gamma_h\mathbf{T}^{\text{HKE}}$，reference-hand 特征则叠加到首帧 latent 上并与噪声 latent 沿通道拼接后再 patchify，把初始手部构型直接绑定进 latent 状态；EME 通过在前 $D$ 个 DiT block 中逐块使用零初始化线性 adapter 生成残差 $\Delta\mathbf{X}_l=\mathbf{U}_l(\mathbf{T}^{\text{EME}})$ 叠加到 block 输入，零初始化保证训练初期不破坏骨干原有能力；OEE 被 patchify 成 token 后用 shifted RoPE 独立锚定空间位置，拼接成扩展序列 $T_{\text{all}}=[X_0;T^{\text{OEE}}]$，在每个 DiT block 的自注意力中只作为 KV（不参与最终解码），提供一个"只被参照、不被生成"的持久物体参考以减少长 rollout 漂移。此外首帧还经 CLIP 图像编码器提取全局 embedding，作为固定 KV 输入 Wan-DiT 的 cross-attention 层，进一步鼓励整体场景结构与外观一致性。

**实现细节**：Wan 2.1 14B 骨干 + LoRA（rank 128、alpha 128），推理 50 步，学习率 $1\times10^{-5}$，Adam + BF16，cfg=1.0；16×NVIDIA H100 训练 8000 步，batch size 1，分辨率 480×480，训练约 1 天；训练与推理帧长均为 81，推理可达 16 FPS。

## 三、关键结果

因不存在公开的第一人称 HOI 世界模型 benchmark，作者从 HOT3D 构建评测集：HOT3D-Clips（WebDataset 格式），每 clip 150 帧、16 FPS，含标定的相机内外参、世界对齐相机轨迹、MANO 手部关节和 6-DoF 物体位姿（相对扫描 mesh），按 clip 级 90/10 切分得到 1364 个训练样本、152 个测试样本；正式对比时从训练场景之外挑选 100 个 clip，用滑窗方式切出多个 81 帧子序列作评测样本。对比基线为 Wan（骨干级视频生成）、Cosmos 2B / 14B（通用世界模型）和 Uni3C（更强相机+人体运动控制），三者均用与 EgoHOI 相同的数据、协议微调；PlayerOne 等更新的第一人称世界模型因提交时未公开、无法在同一协议下复现而被排除对比。

**主表（Table 1，HOT3D 测试集）**：

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Object-CLIP↑ | ATE↓ | RRE↓ | RPE↓ | MR↓ | MPJPE↓ | RMSE↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Wan | 14.78 | 0.46 | 0.52 | 0.86 | 0.124 | 10.687 | 0.036 | 19.67% | 0.049 | 0.083 |
| Cosmos 2B | 15.49 | 0.56 | 0.41 | 0.87 | 0.116 | 9.515 | 0.026 | 16.23% | 0.044 | 0.078 |
| Cosmos 14B | 15.89 | 0.59 | 0.38 | 0.88 | 0.112 | 9.046 | 0.022 | 14.61% | 0.041 | 0.074 |
| Uni3C | 14.21 | 0.50 | 0.55 | 0.82 | 0.168 | 14.209 | 0.026 | 20.50% | 0.073 | 0.089 |
| **EgoHOI（本文）** | **21.05** | **0.65** | **0.27** | **0.92** | **0.084** | **5.192** | **0.021** | **5.84%** | **0.014** | **0.044** |

其中 ATE/RRE/RPE 用 MapAnything 对生成视频与 GT 视频分别估计相机轨迹后计算，衡量自运动一致性；MR/MPJPE/RMSE 用 HaMeR 对生成与 GT 帧分别做手部重建后计算，衡量运动学保真度。放大模型规模（Wan→Cosmos 14B）只带来适度的画质提升，运动相关指标改善有限；Uni3C 虽强化了运动控制，但整体 rollout 反而退化（ATE/RRE/MR 均最差），说明单纯的运动控制不能恢复接触驱动的交互动力学。

**VBench 感知/时序质量（Table 2）**：EgoHOI 在 Subject Consistency（95.51%）、Background Consistency（94.91%）、Aesthetic Quality（52.03%）、Imaging Quality（64.48%）上均最高；但 Dynamic Degree（53.29%）低于 Cosmos 14B（89.50%），作者将其解释为"优先保证手物交互稳定"的设计取舍，而非缺陷——该数据集的 Dynamic Degree 并非越高越好，需结合一致性指标综合判断。

**消融研究**（Base = 无 HKE/OEE/EME 的 Wan 2.1 DiT 骨干；三项各自独立开启，验证单一组件贡献）：

| 维度 | 配置 | 指标1 | 指标2 | 指标3 |
|---|---|---|---|---|
| 运动学保真度 | Base / +HKE / 完整模型 | MR: 28.77% / 6.47% / **5.84%** | MPJPE: 0.576 / 0.015 / **0.014** | RMSE: 0.089 / 0.044 / **0.044** |
| 自运动一致性 | Base / +EME / 完整模型 | ATE: 0.133 / 0.096 / **0.084** | RRE: 15.249 / 6.525 / **5.192** | RPE: 0.039 / 0.023 / **0.021** |
| 物体完整性 | Base / +OEE / 完整模型 | Object-CLIP: 0.81 / 0.83 / **0.92** | OPE: 0.141 / 0.131 / **0.015** | OOE: 27.739 / 24.124 / **9.412** |

三项组件各自主导对应维度、组合后进一步收窄误差，表明三路先验互补而非冗余；补充材料还给出基线方法的 OPE/OOE（Wan 0.078/19.295，Cosmos 2B 0.112/23.823，Cosmos 14B 0.108/23.312，Uni3C 0.083/24.751，均明显劣于 EgoHOI 的 0.015/9.412），以及同一首帧、不同用户输入下模型生成不同但均合理的 rollout，用以佐证模型确实响应动作信号而非简单记忆重放。

## 四、评价与展望

**优点**：论文明确指出并试图纠正当前第一人称 HOI 视频生成中一个容易被忽视的方法论问题——很多工作用未来物体轨迹/waypoint 作条件，实质是 conditional video generation 而非因果世界模型，评测时看似逼真、实则绕开了"从动作推断动力学"的核心难题。EgoHOI 去掉这一捷径后仍保持较高的物理合理性，方法论上更贴近世界模型的本意。三路 embedding（手部运动学 / 度量自运动 / 物体实体）用各自专门的表示（稠密手部渲染、Plücker 射线场、分割 token）蒸馏 3D 先验，再通过零初始化 adapter 和 KV-only 注入等轻量方式挂在冻结骨干上，是一种代价可控、可解释性较好的"把 3D 重建先验蒸馏进 2D 视频扩散模型"范式，与 Plücker/ray-map 表示在可控视频生成、新视角合成中的应用一脉相承。评测协议也较为完整：视觉质量、借助外部模型（MapAnything、HaMeR）反推的自运动一致性与运动学保真度、自定义的物体位姿误差（OPE/OOE），加上逐组件消融，证据链较为完整。

**局限与开放问题**：作者在 Discussion 中坦承，全面评估物理合理性仍是开放问题——当前指标本质上都是通过外部重建模型（HaMeR、SAM、MapAnything）反推得到的间接代理，缺少直接的接触力或物理仿真一致性验证；未来需要更直接的接触感知（contact-aware）、动力学感知（dynamics-aware）评测手段。此外，评测规模有限（仅 100 个测试 clip、81 帧 rollout 上限），完全基于 HOT3D 单一数据集（受控桌面场景），泛化到更复杂真实环境、双手协作、可变形物体等场景尚待验证。由于 PlayerOne 等专为第一人称场景设计的世界模型在提交时未开源，论文的主要对比对象实为通用视频生成/相机可控模型（Wan、Cosmos、Uni3C）的微调版本，而非严格意义上的同赛道强基线，公平性上留有余地。Dynamic Degree 明显低于 Cosmos 14B 也提示模型在物理先验强正则化下可能存在"可控性换动态范围"的取舍，这一现象在许多引入强 3D/运动先验的可控视频生成方法中也有类似体现，其成因（正则化本身 vs. HOT3D 训练数据动态范围有限）值得进一步拆解。最后，驱动 rollout 的手部运动学与头部轨迹信号本身来自 3D 重建/估计得到的"伪 ground truth"，实际部署时如何从更弱的传感或意图信号生成可靠的动作序列，是这类方法走向下游数据合成实用化之前仍需解决的问题。

## 参考

- Agarwal et al. *Cosmos World Foundation Model Platform for Physical AI.* arXiv:2501.03575, 2025.
- Cao et al. *Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation.* arXiv:2504.14899, 2025.
- Wang et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* CoRR, 2025.
- Hassan et al. *GEM: A Generalizable Ego-Vision Multimodal World Model for Fine-Grained Ego-Motion, Object Dynamics, and Scene Composition Control.* CVPR, 2025.
- Tu et al. *PlayerOne: Egocentric World Simulator.* arXiv:2506.09995, 2025.
- Pavlakos et al. *Reconstructing Hands in 3D with Transformers (HaMeR).* CVPR, 2024.
