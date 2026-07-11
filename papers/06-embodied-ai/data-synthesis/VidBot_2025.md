# VidBot：从野外 2D 人类视频学习可泛化的 3D 动作以实现零样本机器人操作

> **论文**：*VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation*
>
> **作者**：Hanzhi Chen, Boyang Sun, Anran Zhang, Marc Pollefeys, Stefan Leutenegger
>
> **机构**：Technical University of Munich（慕尼黑工业大学）、ETH Zürich（苏黎世联邦理工）、Microsoft
>
> **发布时间**：2025 年 03 月（arXiv 2503.07135）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.07135) | [PDF](https://arxiv.org/pdf/2503.07135)
>
> **分类标签**：`人类视频` `3D affordance` `扩散模型` `零样本操作` `SfM`

---

## 一句话总结

VidBot 用 SfM + 度量深度基础模型从**野外 RGB-only 人类视频**（EpicKitchens-100）重建**度量尺度、时序一致的 3D 手部轨迹** 作为 embodiment-agnostic 的 3D affordance（接触点 + 交互轨迹），再用 coarse-to-fine 框架（heatmap 粗预测接触/目标点 + 扩散模型细化轨迹 + 测试期可微代价引导）学习并零样本部署到机器人，在 13 个仿真家务操作任务上取得 **88.2% 成功率**（比次优的 Octo 69.2% 高近 20 个点），真机（Stretch 3 + Spot）55 次试验 **80.0%** 成功。

## 一、问题与动机

- **具身鸿沟与数据不可扩展**：主流操作策略依赖遥操作/模仿学习采集的物理机器人数据，成本高、耗时、难以覆盖具身×任务×环境的组合爆炸；即便 Open X-Embodiment、DROID 这类大规模遥操作数据集，扩展性依然受限。
- **人类视频是天然可扩展的数据源**，但已有 human-to-robot 迁移方法有明显限制：要么需要静态相机/场景、深度传感器、MoCap（in-lab 设置，缺乏场景/光照/视角多样性），要么把交互动作简化为**像素平面的 2D 向量**（如 VRB、Track2act），缺乏空间感知，难以直接部署到机器人。
- 作者主张：真正统一不同具身、可解释动作意图的是 **3D affordance**——即带空间感知的接触点与交互轨迹。核心难题即本文两个问题：
  1. 如何从原始 RGB-only 人类视频中**提取 3D 可动作知识**？
  2. 如何把提取到的知识**零样本可靠迁移** 到新环境与新具身？

## 二、核心方法

VidBot 分两大块:(A) 一条从野外视频**提取 3D affordance 标签**的几何优化流水线;(B) 一个 **coarse-to-fine** 的 affordance 学习/部署模型。

**affordance 表示**：把 affordance 建模为在观测相机坐标系下、以 RGB-D 帧与语言指令为条件的因子化模型

$$\mathbf{a} = \pi(\{\bar{\mathbf{I}}, \bar{\mathbf{D}}\},\, l) = \{\mathbf{c}, \boldsymbol{\tau}\},\quad \mathbf{c}\in\mathbb{R}^{N_c\times 3},\ \boldsymbol{\tau}\in\mathbb{R}^{H\times 3}$$

其中 $\mathbf{c}$ 是接触点、$\boldsymbol{\tau}$ 是交互轨迹，$H$ 为轨迹步长。深度 $\bar{\mathbf{D}}$ 可来自深度传感器或度量深度基础模型。

> **用大白话说**：模型看一张 RGB-D 图 + 一句话"打开抽屉"，就吐出"手/夹爪该在哪儿抓（接触点）"和"抓完往哪儿动一条 3D 轨迹",且用的是米制真实尺度,和具体是什么机器人无关。

### A. 从人类视频提取 3D affordance（数据流水线）

**数据准备**：给定彩色帧 $\{\tilde{\mathbf{I}}_0,\dots,\tilde{\mathbf{I}}_T\}$ 与语言描述 $l$；用 SfM（COLMAP）估计相机内参 $\mathbf{K}$、逐帧**尺度未知** 位姿与稀疏路标;用度量深度基础模型(Depth Anything / Metric3D / ZoeDepth 系)预测逐帧稠密深度;用手-物检测 + 分割模型(SAM 系)得到手 mask $\mathbf{M}^h$ 与在接触物体 mask $\mathbf{M}^o$;再用视频 inpainting 把手抹掉得到"无手帧"$\{\bar{\mathbf{I}}_0,\dots,\bar{\mathbf{I}}_T\}$。

**一致位姿优化（关键）**：野外视频相机在动、尺度未知，且手-物动态运动会污染 SfM。分两步把位姿/深度校正到统一度量尺度：

第一步，优化全局尺度 $s_g$，让稀疏路标在各帧的重投影深度与预测深度一致：

$$\min_{s_g}\ \sum_{i,j}\bar{\mathbf{M}}_i[\mathbf{u}_{ij}]\,\big\|\hat{\mathbf{D}}_i[\mathbf{u}_{ij}] - s_g\,\mathrm{d}(\mathbf{T}^{-1}_{WC_i}\,{}_w\mathbf{l}_j)\big\|_2^2$$

其中 $\bar{\mathbf{M}}_i=\lnot(\mathbf{M}^h_i\cup\mathbf{M}^o_i)$ 仅取**静态区域**（把手和被抓物排除，因为它们在动、不可靠）。

第二步，进一步精修各帧位姿 $\mathbf{T}_{WC_i}$ 与逐帧尺度 $s_i$，靠**跨视图光度/几何一致性** 补偿 SfM 误差：

$$\min_{\mathcal{T}\setminus\{\mathbf{T}_{WC_0}\},\,\mathcal{S}\setminus\{s_k\}}\ \sum_{i\neq k}\sum_{\mathbf{u}_i}\bar{\mathbf{M}}_i[\mathbf{u}_i]\,\bar{\mathbf{M}}_k[\mathbf{u}_k]\,\mathbf{E}[\mathbf{u}_i]$$

$\mathbf{E}[\mathbf{u}_i]=\big\|s_i^{-1}\mathbf{T}_{C_kC_i}{}_{C_i}\mathbf{X}^n_i[\mathbf{u}_i]-s_k^{-1}{}_{C_k}\mathbf{X}^n_k[\mathbf{u}_k]\big\|_2^2$ 是把第 $i$ 帧反投影的 3D 点变换到参考帧 $k$（与其余帧共视度最高）后的对齐误差，$s_k$ 固定为 $s_g$。

> **用大白话说**：野外视频既没有真实米制尺度、相机又乱动，直接三角化的点云是"橡皮尺量的"。这两步先用一根全局标尺把深度对齐到米制，再逐帧微调让所有视角看到的同一块背景在 3D 里真正重合——只用背景（去掉运动的手和物）当"锚"，从而反解出手在真实 3D 空间里的运动。

**affordance 抽取**：拿每帧手部中心点，用精修后的位姿与尺度统一到首帧坐标系 → 得交互轨迹 $\hat{\boldsymbol{\tau}}$；首帧手点均匀下采样得接触点 $\hat{\mathbf{c}}$，末帧手点得目标点 $\hat{\mathbf{g}}$（用于监督中间预测）。模型输入取首帧无手彩色 $\bar{\mathbf{I}}_0$、深度 $\bar{\mathbf{D}}_0$、被抓物裁剪图与语言 $l$。数据来自 EpicKitchens-100 视频 + EpicFields 的 SfM 结果。

### B. Coarse-to-Fine affordance 学习

把 $\pi$ 拆成粗模型 $\pi_c$ 与细模型 $\pi_f$：粗阶段做高层场景理解、预测目标点与接触点 $\{\mathbf{g},\mathbf{c}\}=\pi_c(\{\bar{\mathbf{I}},\bar{\mathbf{D}}\},l)$；细阶段以粗输出为条件规划细粒度轨迹 $\boldsymbol{\tau}=\pi_f(\{\bar{\mathbf{I}},\bar{\mathbf{D}}\},l,\mathbf{a}_c)$。

**粗预测**：两支 hourglass 网络 $\pi_c^{goal}$ 与 $\pi_c^{cont}$，用视觉编码器出全局上下文特征、RoI Pooling 出物体中心特征、MLP 出 bbox 位置特征、冻结 CLIP 出语言特征，经 Perceiver 融合后解码出**逐像素目标/接触 heatmap**（目标点额外回归深度，接触点直接查表面深度），再用内参 lift 到 3D。

> **用大白话说**：先在 2D 图像上"圈出"该抓哪、终点大概在哪（heatmap 比直接回归 3D 坐标更稳、更抗噪），再借深度把这些点抬到 3D。

**细预测（条件扩散）**：$\pi_f$ 用 1D U-Net，以最高概率的目标点 $\bar{\mathbf{g}}$、接触点 $\bar{\mathbf{c}}$、ViT 物体特征、语言特征为条件 $\mathbf{o}$。为注入空间感知，把 RGB-D 体素化成 TSDF 场 $\mathbf{U}$，经 3D U-Net 编码，对每个 waypoint 三线性插值取空间特征 $\mathbf{f}^k$ 拼进去噪输入。扩散前向/反向过程为

$$q(\boldsymbol{\tau}^k\mid\boldsymbol{\tau}^{k-1})=\mathcal{N}(\boldsymbol{\tau}^k;\sqrt{1-\beta_k}\,\boldsymbol{\tau}^{k-1},\beta_k\mathbf{I}),\qquad p_\phi(\boldsymbol{\tau}^{k-1}\mid\boldsymbol{\tau}^k)=\mathcal{N}(\boldsymbol{\tau}^{k-1};\boldsymbol{\mu}_\phi(\boldsymbol{\tau}^k,k),\boldsymbol{\Sigma}_k)$$

且不预测噪声、而是每步直接预测**去噪后的完整轨迹** $\hat{\boldsymbol{\tau}}^0=\pi_f(\mathbf{x}^k,\mathrm{PE}(k),\mathbf{o})$。

> **用大白话说**：把"生成一条 3D 轨迹"看成扩散去噪：从纯噪声一步步磨成一条合理轨迹，且每步都能拿到当前完整轨迹估计，方便后面用几何代价去"掰"它。

**测试期代价引导（zero-shot 泛化关键）**：把多目标、避碰、接触法向等约束写成可微代价，在去噪过程中用 reconstruction guidance 修正轨迹：

$$\mathcal{J}_{goal}=\min_{\mathbf{g}_n\in\mathbf{g}}\|\mathbf{g}_n-\boldsymbol{\tau}^0_H\|_2^2$$

$$\mathcal{J}_{collide}=\frac{1}{H'N_p}\sum_{h\neq1,i}-\min\!\big(\mathbf{U}[\mathbf{p}_i+\boldsymbol{\tau}^0_h-\boldsymbol{\tau}^0_1],\,0\big),\qquad \mathcal{J}=\lambda_g\mathcal{J}_{goal}+\lambda_c\mathcal{J}_{collide}+\lambda_n\mathcal{J}_{normal}$$

去噪修正为 $\boldsymbol{\tau}^0=\bar{\boldsymbol{\tau}}^0-\boldsymbol{\Sigma}_k\nabla_{\boldsymbol{\tau}^k}\mathcal{J}$。$\mathcal{J}_{goal}$ 把"该落到哪个目标点"当成可选集合中取最近（多目标而非硬绑单一点）；$\mathcal{J}_{collide}$ 从 TSDF 场采样夹爪/物体表面点，惩罚穿入障碍（TSDF 值为负即穿透）；$\mathcal{J}_{normal}$ 约束轨迹方向贴合接触法向。

> **用大白话说**：训练数据有噪、粗阶段的目标点也可能偏，所以部署时不完全信网络——用"别撞、朝法向走、落到合理目标点"这些**当前场景/机器人几何** 算出来的可微代价,在生成轨迹的每一步把它往物理上合理的方向推。这就是它能零样本适配新场景、新夹爪的核心。最后每条轨迹的代价值 $\mathcal{J}$ 还能当分数,用来挑最优执行方案。

**训练**：粗阶段把 $\hat{\mathbf{g}},\hat{\mathbf{c}}$ 投影到像面拟合 GMM 生成 heatmap 真值，用 BCE + 深度回归 + 辅助向量场回归损失 $\mathcal{L}_v$ 监督；细阶段用 $\mathcal{L}_f=\mathbb{E}_{\epsilon,k}[\|\hat{\boldsymbol{\tau}}-\boldsymbol{\tau}^0\|_2^2]$ 直接监督去噪轨迹。

## 三、实验结果

**仿真主结果**：IsaacGym 平台，13 个家务任务取自 FrankaKitchen / PartManip / ManiSkill 三个基准，每任务 3 视角 × 5 条轨迹 = 15 次试验；成功 = 使物体自由度超过阈值且不碰撞。成功率(%)：

| 方法 | 训练数据 | 平均成功率 |
| --- | --- | --- |
| GAPartNet | 仿真铰接资产 | 51.1 |
| Where2Act | 仿真铰接资产 | 58.5 |
| Octo*（微调） | 大规模遥操作 + 自采数据微调 | 69.2 |
| VRB† | 野外人类视频（2D→3D lift） | 59.0 |
| GFlow | 野外人类视频（+GT 深度/位姿/物姿） | 61.0 |
| **VidBot（本文）** | **野外人类视频** | **88.2** |

- VidBot 全面最优,平均比次优的 Octo(69.2%,还多用了大规模遥操作预训练)高约 **19 个点**;比同用 EpicKitchens 数据的 VRB(59.0%)高约 **30 个点**——作者归因于充分利用 3D 先验 + 本文的 affordance 学习策略。
- VRB / GAPartNet / Where2Act 把交互抽象成方向向量,只能应付直线拉/推,遇到"开柜子"这类曲线交互会夹爪打滑失败。

**消融（6 任务子集,成功率%）**：

| 变体 | 平均成功率 |
| --- | --- |
| Full Model | 85.6 |
| w/o 粗 goal 预测（V1） | 57.8 |
| w/o 多目标引导（V2） | 73.3 |
| w/o 接触法向引导（V3） | 76.7 |
| w/o 避碰引导（V4） | 77.8 |
| w/o 代价启发式选择（V5） | 74.5 |

- **粗目标预测最关键**：去掉后 85.6 → 57.8（掉 27.8 点），说明直接从高维观测一步生成细轨迹很难，粗阶段把目标配置当 affordance cue 大大简化了轨迹生成。
- **多目标代价引导** 是提升最大的引导项，整体 +12.3%（单目标条件反而会误导生成）。
- **避碰引导** 对可搬运物体的抓取任务尤其重要（该任务 +26.7%）。
- **用代价值当启发式挑最优方案** 去掉后掉 11.1 个点。

**下游应用**：视觉目标到达（visual goal-reaching）与探索（exploration）两个任务上，VidBot 收敛更快、成功率显著高于 VRB / GFlow / Random。

**真机实验**：两个移动平台 **Hello Robot Stretch 3** 与 **Boston Dynamics Spot**（均载 RGB-D 相机 + 语言指令），在 3 个真实人居环境做推抽屉、开柜、取纸巾等任务，**55 次试验 80.0% 成功**，验证 embodiment-agnostic 与零样本可迁移性。

## 四、局限性

- **数据质量受限于深度基础模型与 SfM 流水线的精度**：尽管有最终一致性优化损失来过滤低质量标签，度量深度与 SfM 的误差仍是标签噪声主来源。作者指出可换用学习式 SfM（MASt3R-SfM / DUSt3R / MonST3R）改善标签质量。
- **只有单目 RGB 手部轨迹这一种模态**：精细/接触力相关任务（如拧瓶盖）目前难以覆盖，作者设想未来用可穿戴设备采多模态 affordance 数据。
- 交互被建模为"接触点 + 一段手中心轨迹",**未显式建模手指/夹爪的抓握姿态与力**,对灵巧、需闭环力控的任务表达力有限。
- 评测的 13 个任务仍以铰接物体开合、推拉、抓取为主,长时序、多步骤组合任务未涉及;成功判据是"自由度超阈值不碰撞",相对宽松。

## 五、评价与展望

**优点**：
- **把 human-to-robot 从 2D 像素向量真正推进到度量尺度 3D**。相比 VRB（2D 接触点 + 方向向量）、General Flow（GFlow，需 GT 深度/位姿）、RAM（检索式迁移），本文的一致位姿优化流水线在**纯 RGB、移动相机、未知尺度**的最难设定下重建出米制 3D 轨迹，这是其能零样本跨具身部署的物理基础。
- **coarse-to-fine + 测试期可微代价引导** 是很实用的组合：heatmap 粗预测抗噪、扩散细化多样、代价引导把训练噪声与具身/场景差异在**部署时**补偿掉，且代价值天然可作为多方案排序的评分——这套"训练不必完美、测试期用几何约束兜底"的思路对野外弱标注数据尤其契合。
- 实证扎实：13 任务 + 6 项消融 + 两类下游任务 + 两个真机平台，把"数据流水线—模型—引导—部署"每一环都做了拆解验证。

**缺点/开放问题**：
- **误差链条长**：SfM→度量深度→inpainting→手-物分割任一环出错都会污染 3D 标签，虽有过滤但缺乏对标签质量的定量评估（如与真值轨迹的误差分布）。
- **接触/力语义缺失**：仅用手中心轨迹，未建模抓握构型，限制了向灵巧手、接触密集任务的推广。
- 与并发的"生成式世界模型/视频扩散造数据"路线相比，本文是**几何重建式造标签**——优点是物理一致、可解释，缺点是受限于现成几何模块上限，且不易生成训练分布外的新交互。二者结合（几何一致性约束 + 生成式多样性）是自然的改进方向。
- 可能的改进：换学习式 SfM 端到端联合优化尺度与深度、把抓握姿态/接触力纳入 affordance 表示、引入长时序多步任务与更严格的成功判据、以及把代价引导从手工几何项扩展为可学习的约束。

## 参考

1. Bahl et al., *Affordances from Human Videos as a Versatile Representation for Robotics (VRB)*, CVPR 2023 —— 同用 EpicKitchens、但把 affordance 简化为 2D 接触点 + 方向向量的最相关前作。
2. Yuan et al., *General Flow as Foundation Affordance for Scalable Robot Learning (GFlow)*, CoRL 2024 —— 从人类视频学 3D 流作为 affordance 的强基线（需 GT 深度/位姿）。
3. Kuang et al., *RAM: Retrieval-Based Affordance Transfer for Generalizable Zero-Shot Robotic Manipulation*, arXiv 2024 —— 本文用其策略把 2D affordance lift 到 3D 做公平对比。
4. Janner et al., *Planning with Diffusion for Flexible Behavior Synthesis (Diffuser)*, ICML 2022 —— 细阶段条件扩散轨迹生成 + guidance 的方法学基础。
5. Tschernezki et al., *EPIC Fields: Marrying 3D Geometry and Video Understanding*, NeurIPS 2023 —— 提供 EpicKitchens 的 SfM 结果，是本文 3D 标签流水线的关键数据依托。
