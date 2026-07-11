# RoboPaint：从人类演示到任意机器人与任意视角

> **论文**：*RoboPaint: From Human Demonstration to Any Robot and Any View*
>
> **作者**：Jiacheng Fan, Zhiyue Zhao, Yiqian Zhang, Chao Chen, Peide Wang, Hengdi Zhang, Zhengxue Cheng et al.
>
> **机构**：Paxini Tech.（柏睿科技）；上海交通大学；浙江大学
>
> **发布时间**：2026 年 02 月（arXiv 2602.05325，v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.05325) | [PDF](https://arxiv.org/pdf/2602.05325)
>
> **分类标签**：`人到机器人数据` `灵巧手retargeting` `触觉感知` `3DGS-real-sim-real` `VLA数据合成`

---

## 一句话总结

RoboPaint 是一条 **Real-Sim-Real** 数据流水线：在标定好的多相机采集间用带触觉手套记录人类灵巧操作（视觉+关节+触觉），经 **Dex-Tactile 几何加力引导 retargeting** 把人手状态映射到灵巧手，再用 **3DGS 重建真实场景 + Isaac Sim 路径追踪渲染**"重绘"成任意机器人、任意视角的可执行训练数据；灵巧手接触点平均误差仅 **3.86 mm**、真机复现成功率 **84%**，仅用合成数据训练的 Pi0.5 在三个任务上达 **80%** 平均成功率（遥操作数据为 100%），而数据采集效率相对遥操作最高提速 **5.33×**。

## 一、问题与动机

VLA 模型的性能上限由训练数据的规模、多样性和质量决定,而高保真机器人数据的采集始终是瓶颈:

- **遥操作(teleoperation)** 如 ALOHA、GELLO、VR/AR、动捕虽保真度高,但可扩展性差、硬件成本高、需专家操作员,且长时/双手/接触密集任务的采集时间随复杂度急剧上升。
- **手持采集设备** 如 UMI、DOBB-E 牺牲了灵巧性,只能训练简单夹爪末端,无法覆盖多指精细操作。
- **网络人类视频** 面临 **embodiment gap(具身差异)**:人手与机器人手在结构、运动学上差异巨大,且视频缺乏触觉信息,难以直接迁移技能。

本文的核心主张是:与其遥操作机器人,不如直接采集**人类**演示——人类操作天然灵巧、连续、高效,再把这些演示"重绘(paint)"到目标机器人本体和目标部署场景上。要打通这条路,必须同时解决三件事:(1) 高保真采集人手的视觉/运动学/**触觉**多模态信号;(2) 跨越 embodiment gap,把人手轨迹映射为物理可行的灵巧手轨迹;(3) 消除视觉域差异,把动作渲染进真实感的部署场景。RoboPaint 分别用**多模态采集间**、**Dex-Tactile retargeting**、**3DGS 数字孪生渲染**回应这三点。

## 二、核心方法

整条流水线分三阶段:多模态人类数据采集 → 跨本体建模(位姿估计 + Dex-Tactile retargeting)→ 场景重建与机器人数据渲染。

### 2.1 多模态人类数据采集

在标定好的采集间中,多相机阵列同步记录:**11 路 RGB 信号**(各 1200×1920)、**3 路 RGB-D 信号**(各 720×1280,Intel RealSense D455)、**15 路触觉信号**(总分辨率 3465×3)以及 **29 路本体关节信号**。手套内嵌高精度磁旋编码器采集各指关节角(亚度级),并用 Hall 效应触觉阵列覆盖掌面与指尖测法向接触力。所有模态由中央时间服务器经硬件触发做微秒级对齐,30 Hz 采样,打包成分层 HDF5:

$$I_i = \{\text{img}_{\text{human}}^t\}_{t=1}^{T}, \quad J_{\text{Glove}} = \{j_{\text{Glove}}^t\}_{t=1}^{T}, \quad \Gamma_{\text{Glove}} = \{\gamma_{\text{Glove}}^t\}_{t=1}^{T}$$

其中 $J_{\text{Glove}}$ 为 20 维手部关节角向量,$\gamma_{\text{Glove}}^t \in \mathbb{R}^M$ 为 $M=32$ 个触觉传感器归一化到 $[0,1]$ 的力读数。采集前所有被操作物体先用结构光 3D 扫描仪扫成高质量数字资产,入库供后续位姿估计与仿真复用。

**用大白话说**:先造一个"专业录音棚",人戴着能测每个指关节角度、还能测每根手指按多大力的手套,在多台相机围观下做各种灵巧活;所有数据(画面、关节、触觉)按时间戳严丝合缝对齐存好,物体也提前扫成 3D 模型备用。

### 2.2 位姿估计

对手腕:在腕带上贴 **ArUco marker**,用亚像素精度检测器恢复位姿,即使部分遮挡也能得到平滑轨迹;对物体:用 **FoundationPose** 在变光照/杂乱背景下做度量精确的 6D 定位。得到手腕位姿 $P_{\text{Glove}}=\{p_{\text{Glove}}^t\}$ 与物体位姿 $P_{\text{Object}}=\{p_{\text{Object}}^t\}$(均在 RGB-D 相机坐标系下)。

### 2.3 Dex-Tactile 联合 retargeting(核心贡献)

人手与灵巧手(目标为 Paxini DexH13)在指长、关节限位、驱动拓扑上差异巨大,不能直接执行手套数据。与只对齐末端几何的既有方法(如 AnyTeleop)不同,本文在 retargeting 中**显式引入触觉反馈作为额外约束**。给定手套输入 $(J_{\text{Glove}}, P_{\text{Glove}}, \Gamma_{\text{Glove}})$,在每一时刻优化灵巧手状态使如下能量最小:

$$\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{tac}}$$

运动学项对齐指尖关键点的 3D 位置 $\mathbf{p}$ 与朝向向量 $\mathbf{d}$:

$$\mathcal{L}_{\text{kin}} = \frac{1}{N}\sum_{i=1}^{N}\left(\lambda_{\text{pos}}\|\mathbf{p}_i^{\text{Glove}} - \mathbf{p}_i^{\text{Dex}}\|_2 + \lambda_{\text{dir}}\|\mathbf{d}_i^{\text{Glove}} - \mathbf{d}_i^{\text{Dex}}\|_2\right)$$

触觉项强制接触一致性,把手套触觉点 $\mathbf{g}_j$ 经预定义解剖对应函数 $\text{NN}_{\text{Dex}}(\cdot)$ 映到灵巧手表面:

$$\mathcal{L}_{\text{tac}} = \frac{1}{M}\sum_{j=1}^{M} w_{\mathbf{g}_j}\|\mathbf{g}_j - \text{NN}_{\text{Dex}}(\mathbf{g}_j)\|_2$$

其中权重由归一化净力 $F_j\in[0,1]$ 经 sigmoid 激活给出,**让真正在用力接触的区域权重更高**:

$$w_{\mathbf{g}_j} = \left[1 + \exp(-20(F_j - 0.5))\right]^{-1}$$

**合成灵巧手触觉信号**:给定 retarget 后的关节角 $J_{\text{Dex}}$,用正运动学算出灵巧手上触觉点位置,对每个手套触觉点 $\mathbf{g}_i$ 找到解剖对应的灵巧手触觉点 $\mathbf{q}_i$,量化二者空间偏差 $\delta_i = \|\mathbf{g}_i - \mathbf{q}_i\|_2$,再用**距离感知衰减函数**调制原始触觉:

$$\hat{\gamma}_i = \gamma_i \cdot \frac{1}{1 + \exp(-\alpha(\delta_i - \beta))}$$

$\alpha>0$ 控制过渡陡度(取 $\alpha=20$),$\beta\ge 0$ 是理想 retargeting 下的期望对齐误差阈值(取 5–10 mm)。**用大白话说**:人手和机械手的指尖对不齐时($\delta_i$ 大),就把这个点上测到的力"打折扣",因为对应位置不准、力也不该照搬;对得越齐,力越原样保留——这样合成出来的机器人触觉才符合物理接触语义(抓握力度、接触时序)。

最终得到 $P_{\text{Dex}}, J_{\text{Dex}}, \Gamma_{\text{Dex}} = \text{Retarget}(P_{\text{Glove}}, J_{\text{Glove}}, \Gamma_{\text{Dex}})$,再用相机到机器人基座的标定外参 $M$ 转到机器人操作空间:

$$P_{\text{RobotTCP}} = M P_{\text{Dex}}, \quad P_{\text{ObjectInRobot}} = M P_{\text{Object}}$$

### 2.4 场景重建与机器人数据渲染

- **场景重建**:把工作台用 **3DGS** 重建成实时真实感的数字孪生。为了让 3DGS 与仿真在尺度/旋转/平移上度量对齐,遵循 **Re³Sim** 协议在桌面放已知尺寸(如 10 cm)的 ArUco marker,求解相似变换 $S=(s,R,\mathbf{t})$:

$$\min_{s,R,\mathbf{t}}\sum_j \|sR\mathbf{x}_j^{\text{3DGS}} + \mathbf{t} - \mathbf{x}_j^{\text{real}}\|_2^2$$

对齐后导出为 USD 资产,导入 **Isaac Sim** 作静态背景。

- **混合渲染**:在 Isaac Sim 5.1 中,静态背景用 3DGS 渲染保真,动态物体与机械臂用高质量 mesh 模型;由 $p_{\text{RobotTCP}}^t$ 经 IK 解出机械臂关节角驱动运动,物体由 $p_{\text{ObjectInRobot}}^t$ 驱动;用 **路径追踪(path-tracing)** 引擎获得真实光影,把动态渲染的机器人/物体叠加到 3DGS 背景上。这套方案支持**任意视角、任意机器人本体**的图像编辑。

- **数据打包**:动作 $a^t = [pos^t, rot^t, j_{Dex}^t]$($pos,rot\in\mathbb{R}^3$ 来自 TCP,$j_{Dex}\in\mathbb{R}^N$ 为灵巧手关节角);触觉按 **ObjTac** 格式处理成触觉图像,最终训练样本 $d^t = [a^t, img_{\text{visual}}^t, (img_{\text{tactile}}^t)]$。

- **数据增强(附录)**:(1) 背景替换——用重建好的多样桌面背景库对同一条数据换背景;(2) 物体材质替换——用开源 3D 生成工具给物体换纹理。

## 三、实验结果

评测分仿真评测(采集精度、retargeting 精度、编辑保真度)与真机 VLA 下游三部分,共两种机器人本体(UR5、Paxini ToRA One)。

**retargeting 与真机复现精度(Fig.4)**:在 10 个不同形状物体上,灵巧手触觉接触点平均误差 **3.86 mm**;UR5+DexH13 真机复现每物体 10 次试验,平均成功率 **84%**,简单几何/稳定接触物体(椰子水瓶、柔顺剂瓶)达 90–100%,复杂物体(塑料杯、相机)仍在 80% 以上。

**VLA 下游三任务(Table 1)**:严格控制两种数据源(Real-Sim-Real 记为 Paint,真机遥操作记为 Tele)演示数量相等,各评 10 个随机初始位置。

| 任务 | DP·Tele | DP·Paint | Pi05(含腕相机)·Tele | Pi05(含腕相机)·Paint | Pi05(无腕相机)·Tele | Pi05(无腕相机)·Paint |
|---|---|---|---|---|---|---|
| Pick and place | 90% | 40% | 100% | 70% | 90% | 40% |
| Push cuboid | 100% | 100% | 100% | 100% | 90% | 70% |
| Pour bottle | 40% | 10% | 100% | 70% | 70% | 30% |
| **Avg** | **76.6%** | **50.0%** | **100%** | **80.0%** | **83.3%** | **46.6%** |

遥操作数据训练的策略在三任务上达 100% 成功率,合成数据(Pi05 含腕相机)达 80%,**仅有约 20% 的成功率下降**——作者据此论证该流水线可作为遥操作的可扩展、低成本替代。

**采集效率(Table 2)**:在 6 个不同难度任务上各采集 100 条成功演示,人类演示全面快于遥操作;越复杂效率优势越大。

| 任务 | 遥操作 | 人类演示 | 效率倍率 |
|---|---|---|---|
| pick and place | ~1h30min | ~35min | 2.57 |
| open box | ~2h | ~30min | 4.00 |
| push cuboid | ~2h | ~30min | 4.00 |
| bagging fruits | ~10h | ~2h20min | 4.28 |
| table bussing | ~12h | ~2h30min | 4.80 |
| fold clothes | ~16h | ~3h | **5.33** |

**部署轨迹质量(附录 Fig.6)**:合成数据训练的 Pi0.5 部署轨迹更平滑,遥操作数据训练的出现明显抖动(jitter)——作者归因于遥操作需同时追踪机器人状态并经中间控制器传递人类意图,累积了跟踪误差、传感噪声与控制延迟,而人类演示天然连续。

## 四、局限性

- **与遥操作仍有约 20% 差距**:合成数据在 pour(70% vs 100%)、pick-and-place(70% vs 100%)这类精度/接触敏感任务上掉点明显,尤其去掉腕部相机后 Pi05 平均仅 46.6%,对腕视依赖强。
- **腕部视角 3DGS 伪影**:Fig.8 明确指出"重绘"的腕部图像有伪影,因腕相机离 3DGS 点云太近导致相关 3D 点被 renderer 剔除(culled),作者称需优化 3DGS 重建算法。
- **重资产依赖**:整条流水线依赖标定好的多相机采集间(11 RGB+3 RGB-D)、定制触觉手套、结构光扫描全部物体资产,以及每个部署场景的 3DGS 重建,前置门槛高、场景泛化需重复搭建。
- **"任意机器人"更多停留在渲染层面**:真机闭环验证仅在 UR5+DexH13 上完成,其余本体(Franka、Agilex、Unitree G1、ToRA 系列)主要以渲染图展示,跨本体动力学可行性与真机成功率未量化。
- **触觉对应靠预定义映射**:$\text{NN}_{\text{Dex}}(\cdot)$ 是预定义的解剖对应函数,依赖人手与特定灵巧手形态的先验匹配,换灵巧手需重新设计。
- **论文内数值口径不完全自洽**:触觉通道在摘要(14)、贡献列表(15 路/3465×3)、公式($M=32$)处描述不一,关节维度在 20 维与 29 路本体信号间亦需读者自行对照。

## 五、评价与展望

**优点**。RoboPaint 最有价值的两点:(1) 把**触觉**纳入 retargeting 的优化目标与合成对象,而非仅做末端几何对齐——$\mathcal{L}_{\text{tac}}$ 与距离感知触觉衰减 $\hat{\gamma}_i$ 让合成数据保留了接触力语义,这是绝大多数"人到机器人"视觉迁移工作(RoviAug、Shadow、EgoMimic、AR2-D2)所缺失的维度,对接触密集/力敏任务(倒水、插入)是正确方向;(2) 用 3DGS 数字孪生 + Isaac Sim 路径追踪做"任意视角/任意本体"渲染,把域差异控制在背景真实感与 IK 可行性上,而非留给策略去弥合。效率数据(最高 5.33×)对长时双手任务尤其有说服力,轨迹平滑性分析也点出了遥操作数据的一个真实痛点。

**与公开工作的关系**。方法上是多条线的组装:retargeting 思路源自 AnyTeleop 并加触觉约束;度量对齐 3DGS 直接沿用 Re³Sim;物体位姿用 FoundationPose;背景表征用 3DGS(Kerbl 2023);下游用 Pi0.5 与 Diffusion Policy;触觉图像格式借 ObjTac。相对 RoviAug/Shadow/EgoMimic 这些**在 2D 图像空间**做人机替换的方案,本文选择**在 3D 空间对齐后再渲染 2D**,物理合理性更高但工程链条更长、更依赖精确标定与扫描。

**开放问题与可能改进**。(1) 腕视 3DGS 剔除伪影是当前最直接的性能天花板,可考虑近距离用 mesh/NeRF 混合或对腕相机做单独的近场重建;(2) 合成-真实之间 20% 的 gap 值得拆解:究竟来自视觉域(渲染)、动作域(retargeting/IK)还是触觉利用不足,论文缺少消融(如去掉 $\mathcal{L}_{\text{tac}}$、去掉触觉衰减对下游成功率的影响),补上会显著增强"触觉有用"的论证;(3) "任意机器人"若能给出跨本体(非 DexH13)的真机成功率,claim 会实至名归;(4) 采集间重资产限制了规模化,能否用更轻的采集(单目+手套)配合更强的重建来放宽是后续可扩展性的关键。总体上,这是一篇工程完整、抓住"触觉+3DGS 重绘"差异化卖点的数据合成系统工作,证据链在真机复现精度(3.86 mm/84%)上扎实,但下游 VLA 的对照与消融尚不足以完全支撑"低成本替代遥操作"的强结论。

## 参考

1. Liu et al. *Re³Sim: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation.* arXiv:2502.08645, 2025.（3DGS 度量对齐与 real-to-sim 协议的直接来源）
2. Wen et al. *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects.* arXiv:2312.08344, 2024.（物体位姿估计）
3. Qin et al. *AnyTeleop: A General Vision-based Dexterous Robot Arm-Hand Teleoperation System.* RSS, 2023.（灵巧手 retargeting 思路来源)
4. Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG, 2023.（场景表征与渲染基座）
5. Physical Intelligence et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization.* arXiv:2504.16054, 2025.（下游 VLA 架构与主要评测模型）
