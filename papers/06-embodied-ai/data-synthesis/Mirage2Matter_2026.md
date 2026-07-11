# Mirage2Matter：一个从视频构建的、物理接地的高斯世界模型

> **论文**：*Mirage2Matter: A Physically Grounded Gaussian World Model from Video*
>
> **作者**：Zhengqing Gao\*、Ziwen Li\*(共同一作)、Xin Wang、Jiaxin Huang、Yandong Guo、Runqi Lin、Tongliang Liu、Kun Zhang、Mingming Gong(通讯) et al.
>
> **机构**：MBZUAI、AI2 Robotics、The University of Sydney、Carnegie Mellon University、The University of Melbourne
>
> **发布时间**：2026 年 02 月（arXiv 2602.00096）
>
> **发表状态**：未录用（预印本；ACM 会议模板占位，作者页仍为 Anonymous 双盲状态）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.00096) | [PDF](https://arxiv.org/pdf/2602.00096)
>
> **分类标签**：`高斯世界模型` `3DGS仿真` `Sim2Real` `VLA数据合成` `机器人操作`

---

## 一句话总结

用一段手持多视角视频重建出 photorealistic 的 3DGS 场景与物体,再借标定板 + 手眼标定把它们统一对齐到 Genesis 物理引擎的机器人基座坐标系,通过"Genesis 出物理正确的机械臂前景 + 3DGS 出真实感背景"的混合渲染批量合成 VLA 训练数据;在真机零样本部署上,抓取成功率把 DISCOVERSE/RoboSimGS 从 60~77% 提到 80~86.7%,逼近纯真机数据训练的 90~96.7% 上界。

## 一、问题与动机

具身智能的可扩展性瓶颈在于真实交互数据稀缺——真机采集贵、慢(熟练操作员约 1 条/分钟),且依赖专用硬件。现有仿真造数据的两条路线各有硬伤:

- **重建式(reconstruction-based)** 世界模型(AI2thor、Habitat、SAPIEN 等传统重建,以及 DISCOVERSE、RoboGSim、Re3Sim、SplatSim、GSWorld 等 3DGS 增强版):视觉可做到 photorealistic,但往往依赖昂贵传感器、精确机器人标定或深度测量,且缺乏对物理规律的内建刻画,需要额外场景搭建与任务专用求解器,难以从普通视频做个性化机器人学习。
- **生成式(generative-based)** 世界模型(DreamGen、Ctrl-World、EmbodiedGen、PhysGen3D、OmniPhysGS、LucidSim 等):自动化、多样、可控,但不为"忠实复刻某个具体真实环境"设计,几何/布局/光照/相机常与目标部署环境偏离,且多在图像或 latent 空间用近似物理,难以保证与真实机器人平台的度量对齐和物理一致性。

核心矛盾:**当前世界模型难以同时做到高保真(visual fidelity)与物理接地(physical grounding)**。本文的目标就是仅凭普通多视角视频,构建一个"视觉忠实 + 物理接地 + 可编辑 + 与现实一致"的世界模型,让纯仿真训练的 VLA 能零样本迁移到真机。

## 二、核心方法

Mirage2Matter 是一条 **reconstruction-based** 管线,由三部分组成:场景与物体重建、跨域对齐、数据生成。关键设计是"渲染用 3DGS、物理用 mesh、两者在同一机器人坐标系下逐像素对齐"。

### 2.1 photorealistic 场景与物体重建

采用 3DGS 作为渲染表征:场景是 $N$ 个各向异性高斯基元的集合,每个基元 $i$ 由均值 $\boldsymbol{\mu}_i$、(以 log-scale 存储的)各向异性尺度 $\ell_i$、旋转 $R_i$、不透明度 $\alpha_i$、球谐系数 $\mathrm{SH}_i$ 参数化。高斯密度与协方差为:

$$
G_i(\mathbf{x}) = \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^\top \Sigma_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right),\qquad \Sigma_i = R_i\,\mathrm{diag}\!\big(\exp(\ell_i)\big)^2\,R_i^\top
$$

- **场景重建(背景)**:采集前先在机器人基座位置(地面/桌面)放一块已知物理尺寸的**标定板**,提供后续度量对齐的锚点;手持视频经 COLMAP 得到稀疏 SfM 点 $P_\text{SfM}$ 与相机位姿,再训练场景 3DGS。用 L1 光度损失 $L_\text{photo} = \sum_t \lVert \hat I_t - I_t\rVert_1$(可加 SSIM)。
- **物体重建(前景)**:用文本提示引导 SAM2 分割物体 mask $M_t^{(o)}$,只在被 mask 的像素上优化 object-specific 3DGS,抑制背景杂波,得到纯物体高斯场。
- **mesh 生成(物理几何)**:3DGS 适合渲染但缺显式表面,碰撞检测/接触需要网格。对每个物体拍几张单视角照片 + 文本描述,交给 **Tripo3D** 生成 canonical 位姿下的 watertight mesh $M_o$。于是形成两条分支——重建分支(COLMAP+3DGS)与生成分支(Tripo3D mesh)。

**用大白话说**:一段随手拍的视频,一路走 3DGS 管线得到"长得像"的高斯点云(负责好看),另一路让 3D 生成模型吐出一个封闭网格(负责能碰、能被抓)。

### 2.2 跨域对齐(把两套东西钉到同一坐标系)

重建与生成的输出天生不共享仿真器坐标系,全部对齐到 Genesis 世界系(定义为机器人基座系)。几何用相似变换,相机/机器人位姿保持刚体变换。

- **场景对齐(SfM → Genesis,在训 3DGS 之前预对齐)**:关键洞察是**训完 3DGS 再做相似变换会全局扭曲已学好的高斯椭球、损害渲染**,所以在 SfM 层面先估相似变换,再在对齐后的坐标系里训 3DGS。在两块语义对应区域间跑 scaled-ICP——(i) Genesis 工作区 mesh 采样点云 $P_r$ 中的机器人基座区,(ii) $P_\text{SfM}$ 中的标定板区,求 $S=(s,R,t)$:

$$
L_\text{ICP} = \sum_i \big\lVert s\,R\,p_i^\text{SfM} + t - p_i^{r}\big\rVert_2^2
$$

  随后把 SfM 点与相机位姿一并按该相似变换更新(点做 $s R p + t$,相机旋转保持、平移随全局相似缩放),再在对齐系里训场景 3DGS,得到天然对齐机器人基座的模型。

- **物体对齐(object 3DGS ↔ mesh,后对齐)**:从 $G_o$ 和 mesh $M_o$ 各采点云,选少量关键点对求初始相似变换 $S_o=(s_o,R_o,t_o)$ 再 ICP 精化。把该变换施加到每个高斯基元:

$$
\tilde{\boldsymbol{\mu}}_i = s_o R_o \boldsymbol{\mu}_i + t_o,\qquad \tilde{\Sigma}_i = s_o^2\, R_o \Sigma_i R_o^\top,\qquad \tilde R_i = R_o R_i,\quad \tilde{\ell}_i = \ell_i + (\log s_o)\mathbf{1}
$$

  即旋转左乘 $R_o$、log-scale 加 $\log s_o$、SH 与不透明度不变,使物体高斯与物理 mesh 在空间上重合。

- **标定与 Sim-to-Real 对齐**:用贴在末端执行器上的 **ChArUco** 标定板,采集"相机检测 vs 正运动学"的配对数据,解经典手眼标定方程(Tsai–Lenz)得到相机相对基座的外参,并注入仿真运动学链——保证机械臂进入真实相机视野时,其在仿真渲染中的投影位置与相对位姿**逐像素一致**(Fig. 5)。

### 2.3 数据生成(混合渲染批量出数据)

- **交互式摆放**:用 SuperSplat 得到摆放变换 $T_\text{place}=(s_p,R_p,t_p)\in \mathrm{Sim}(3)$(例如把面包放到桌上),用与式(9)~(11)相同的规则作用到物体高斯上。
- **合并统一高斯世界**:因所有资产已对齐,直接拼接高斯参数集即可,无需再联合优化:

$$
G_\text{world} = G_\text{scene} \cup \bigcup_{k=1}^{K} G_{\text{obj},k}^{(\text{placed})}
$$

- **Genesis 物理仿真**:加载机器人 URDF(基座=世界原点),按同一 $T_\text{place}$ 加载物体 mesh,使每个 mesh 与对应高斯簇重合;Genesis 提供碰撞检测、接触响应与物理一致运动。
- **运动规划**:解 IK $q^* = \arg\min_q \lVert T_\text{ee}(q) - T_\text{target}\rVert^2$ 得目标关节角,再用关节空间 RRT(OMPL)规划无碰撞轨迹 $P=\{q_t\}_{t=0}^{T}$,在 Genesis 执行。
- **混合渲染**:从 Genesis 抽机械臂前景(mask $M_t^\text{robot}$ 与纯机器人 RGB $I_t^\text{robot}$),用匹配相机渲 3DGS 背景 $I_t^\text{3DGS}$,alpha 合成:

$$
I_t^\text{final} = M_t^\text{robot} \odot I_t^\text{robot} + \big(1 - M_t^\text{robot}\big) \odot I_t^\text{3DGS}
$$

**用大白话说**:机械臂怎么动、会不会撞、能不能抓稳,全交给物理引擎算(保证物理对);画面好不好看、背景像不像真的,全交给 3DGS 渲(保证视觉对);最后把"物理正确的机械臂"抠出来贴到"真实感背景"上,合成的视频既物理正确又照片级真实。

## 三、实验结果

**设置**:VLA backbone 默认 **PoSA-VLA**(也试过 $\pi_{0.5}$、OpenVLA-OFT,趋势相似,因成本高故主实验用 PoSA-VLA);真机为 **AlphaBot 1s**(7-DoF 机械臂 + 头部/腕部 RGB,主输入为头部 egocentric 相机);训练单卡 A100、batch 16、200k step,推理/部署 RTX 4090;任务为 Grasp / Press Button / Push-Pull,每任务每物体 30 次真机试验报成功率。仿真每任务每物体采 **300** 条轨迹,真机上界仅采 **50** 条。基线为 DISCOVERSE、RoboSimGS(同任务定义、同数据量)。评测协议是纯 Sim2Real:仿真训 → 真机零样本部署,不做任何真机微调。

**主结果 · 真机抓取成功率(%,30 次/物体)**:

| 物体 | Real-World(上界) | DISCOVERSE | RoboSimGS | Mirage2Matter(本文) |
| --- | --- | --- | --- | --- |
| Banana | 96.7 | 60.0 | 76.7 | **80.0** |
| Croissant | 90.0 | 66.7 | 76.7 | **86.7** |

在相同仿真数据量下,本文显著优于两条 3DGS 仿真基线。

**任务级 · 对比纯真机训练(%,30 次/任务)**:

| 任务 | Real-World | Mirage2Matter |
| --- | --- | --- |
| Press Button | 96.7 | 93.3 |
| Push/Pull Objects | 83.3 | 73.3 |

纯真机训练最高,但本文在按钮任务上仅差 3.4 个点,整体大幅缩小 gap。

**消融 1 · 物体表征(抓取成功率 %)**:

| 物体表征 | Banana | Croissant | 均值 |
| --- | --- | --- | --- |
| w/o 3DGS(mesh-only) | 76.7 | 80.0 | 78.4 |
| 3DGS(本文) | 80.0 | 86.7 | **83.4** |

把物体从"mesh 渲染"换成"3DGS 渲染"稳定涨点——印证了**物体与背景视觉一致性**的重要性。

**消融 2 · 仿真数据规模(抓取成功率 %)**:

| 演示条数 | Banana | Croissant | 均值 |
| --- | --- | --- | --- |
| 50 | 46.7 | 60.0 | 53.4 |
| 150 | 76.7 | 86.7 | 81.7 |
| 300 | 80.0 | 86.7 | **83.4** |

50→150 提升剧烈(+28.3 均值),150→300 边际收益递减。

**效率**:人工约 1 条真机演示/分钟(受疲劳与失误制约);Mirage2Matter 在单张 RTX 4090 上可达到"每分钟生成相当数量的演示",且可持续、可并行,边际成本更低。

**为何迁移更好(作者分析)**:(1) 更高的视觉保真 + 更强的视觉-物理一致性,使 egocentric 观测的纹理/光照/背景几何统计更接近真机;(2) 环境与物体在**统一 3DGS 世界**内渲染,避免了"mesh 物体 + 3DGS 背景"的系统性视觉不一致——后者会诱导 shortcut learning(模型依赖仿真专属视觉伪影而非交互相关线索);(3) 物理交互由 Genesis 执行 + 碰撞感知运动规划,阻止模型过拟合到"看着合理但物理错误"的行为。

## 四、局限性

- **本质上是特定场景的数字孪生,而非可泛化世界**:所有真机评测都在"当初拍摄建仿真的同一工作区"进行,论文明确称保证 sim/real 一对一对应。这既是干净对照的优点,也意味着未验证对未采集的新环境/新布局的泛化能力。
- **评测规模偏小**:仅 3 类任务、抓取仅 2 个物体、每格 30 次试验;缺少长程/多物体/杂乱场景与统计显著性分析。
- **依赖若干重人工/外部组件**:需预放标定板、做 ChArUco 手眼标定;物体物理几何依赖 Tripo3D 生成的网格,其质量与真实物体的物理属性(质量、摩擦、软硬)一致性未量化,且高保真表面并不等于高保真接触动力学。
- **物理仍是刚体接触为主**:管线围绕 IK+RRT 的抓取/按压/推拉,未涉及可形变、铰接、流体等复杂物理;文中"物理接地"更多指几何/碰撞对齐,而非材料级物理辨识。
- **"每分钟一条"的效率论断偏定性**:未给出端到端吞吐、重建耗时、并行规模等硬指标,3DGS 训练与 COLMAP 的前置成本也未计入。

## 五、评价与展望

**优点**。这篇工作的清晰贡献是把"渲染保真"与"物理接地"这对老矛盾,用一个务实的工程闭环拧到了同一个坐标系:关键创新有两点值得记住——其一是**先在 SfM 层估相似变换、再在对齐系里训 3DGS**,规避了"训完再变换会扭曲高斯"的常见坑;其二是**用标定板 + 手眼标定把 egocentric 相机做到逐像素 sim-real 对齐**,这正是纯生成式世界模型难以保证的度量一致性。消融里"3DGS 物体 vs mesh 物体"的对照,给出了一个有说服力的经验证据:前景-背景视觉一致性本身就是抑制 shortcut learning、提升 Sim2Real 的关键变量,这一点比单纯堆真实感更有洞见。

**与公开工作的关系**。它属于 3DGS 真到仿(real-to-sim)重建式路线,与 SplatSim、RoboGSim/RoboSimGS、Re3Sim、DISCOVERSE、GSWorld、PolaRiS 同源;相对这些工作,卖点在于"仅需普通多视角视频、无需深度/专用传感器"的低门槛与统一高斯世界的一致性。与生成式路线(DreamGen、Ctrl-World、EmbodiedGen、GigaWorld-0)形成互补对照:生成式赢在多样性与新环境想象,本文赢在对特定部署环境的忠实复刻与物理度量对齐——两者是"diversity vs fidelity"的经典权衡的两端。物理侧选择 Genesis + OMPL 而非可微/可辨识物理,属稳妥但保守。

**开放问题与可能改进**。(1) 泛化:当前是单一数字孪生,若要成为"数据引擎",需验证在多环境、随机化布局、跨 embodiment 下的迁移,并把生成式的多样性注入进来(如对高斯世界做光照/材质/布局的可控编辑增强)。(2) 物理辨识:从视频反演质量/摩擦/接触参数,而非仅几何对齐,才能支撑推拉/按压等接触密集任务(现推拉仅 73.3% 也印证接触物理是短板)。(3) 规模化与自动化:去标定板化(如自监督度量恢复)、自动化物体网格质检、报告真实吞吐,才能撑起"可扩展数据引擎"的定位。(4) 更强 backbone 与更大 benchmark:文中已提及 $\pi_{0.5}$/OpenVLA-OFT 趋势相似但未展开,若配以统一公开 benchmark,论证力会明显增强。总体上,这是一篇工程完整、动机清晰、但评测规模与泛化论证仍显单薄的预印本;它的价值更多在于把"逐像素对齐的混合渲染"这一配方讲清楚,而非刷出压倒性的 SOTA 数字。

## 参考

1. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 2023 —— 本文渲染表征的基石。
2. Genesis Authors, *Genesis: A Generative and Universal Physics Engine for Robotics and Beyond*, 2024 —— 本文物理仿真与运动执行后端。
3. Jia et al., *DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments*, arXiv 2507.21981 —— 主要 3DGS 仿真基线之一。
4. Qureshi et al., *SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting*, ICRA 2025 —— 同源 3DGS real-to-sim 路线代表作。
5. Li et al., *PoSA-VLA: Enhancing Action Generation via Pose-Conditioned Anchor Attention*, arXiv 2512.03724 —— 本文默认 VLA backbone。
