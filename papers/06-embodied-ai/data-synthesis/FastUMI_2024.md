# FastUMI：一种可扩展且硬件无关的通用操作接口及其数据集

> **论文**：*FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset*
>
> **作者**：Zhaxizhuoma, Kehui Liu, Chuyue Guan, Zhongjie Jia, Ziniu Wu, Xin Liu, Yan Ding, Bin Zhao, Xuelong Li et al.
>
> **机构**：Shanghai AI Lab、Shanghai Jiao Tong University、University of Bristol、Fudan University、The University of Hong Kong、Xi'an Jiaotong-Liverpool University、Institute of AI, China Telecom Corp Ltd
>
> **发布时间**：2024 年 09 月（arXiv 2409.19499，v2 于 2025 年 02 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2409.19499) | [PDF](https://arxiv.org/pdf/2409.19499)
>
> **分类标签**：`通用操作采集接口` `手持式数据采集` `UMI 改进` `模仿学习数据集` `硬件解耦`

---

## 一句话总结

FastUMI 是对 UMI 手持采集接口的一次系统性重构：硬件上用一套 ISO 标准相机安装件与可插拔指尖把采集设备从特定夹爪解耦、软件上用现成的 RealSense T265 追踪模组替代脆弱的 GoPro-VIO/SLAM 定位管线,并配套开源了跨 22 个日常任务的 **10,000 条**真实演示轨迹;在 12 个任务上以 200 条/任务训练 ACT 与 DP,验证了第一人称鱼眼单目采集数据可支撑高成功率策略学习(如 Depth-Enhanced DP 把 Open Ricecooker 成功率从 20% 提到 93.33%)。

## 一、问题与动机

真实世界操作数据稀缺,是机器人操作策略发展的主要瓶颈。现有采集方式各有短板:遥操作精确但昂贵且需真机、依赖非直观的人到机器人映射;纯视觉演示可大规模低成本但缺乏细粒度交互动力学;传感增强接口(以 UMI 为代表)用手持设备直接捕获与机器人机载传感对齐的多模态信号,兼顾保真度与可迁移性,是有前景的折中方案。

但作者指出 UMI 存在两个关键局限:

1. **硬件强耦合**:UMI 与特定组件(如 Weiss WSG-50 夹爪)紧耦合,迁移到新平台需要指定夹爪、机械重设计、传感重标定与代码参数修改,带来显著的人力与财务开销,泛化性差。
2. **软件依赖 VIO/SLAM**:基于 GoPro 的 VIO(视觉惯性里程计)+ 开源 SLAM 定位管线,在长时间遮挡任务(如开合铰链门/抽屉)中视觉信号间歇丢失,鲁棒性差;VIO 对参数敏感、标定复杂、坐标变换繁多,进一步抬高操作复杂度、损害可复现性。

FastUMI 的三个设计目标:通过硬件解耦增强适配性、通过软件驱动的即插即用提升效率、建立保障数据质量与算法兼容的生态。

## 二、核心方法

### 1. 解耦式硬件设计

系统沿三个维度做解耦:**物理解耦**(标准化接口+模块化组件跨平台集成)、**视觉一致性**(手持端与机载端相机视角统一,使人类演示数据可直接迁移)、**操作独立**(自包含追踪与传感,不依赖外部计算框架)。

**手持采集端**三个核心部件:

- **鱼眼相机模组(GoPro Hero9)**:配鱼眼扩展镜,155° 广角 FOV,远大于常用 RealSense D435i 的窄视场,减少遮挡、提供更宽环境上下文,利于特征提取。
- **位姿追踪模组(RealSense T265)**:替代 UMI 的视觉里程计;T265 自带高性能 IMU,在开橱柜/抽屉这类部分遮挡场景下仍能给出稳定位姿,消除了 VIO 的复杂标定。作者还验证了 RoboBaton MINI 作为可持续获得的替代方案(T265 已停产)。
- **顶盖/指尖/Marker 重设计**:把 GoPro 移近指尖并保证顶盖落在鱼眼视场外,实现完整硬件解耦;优化 marker 尺寸与位置以抑制近距离镜头畸变、提升检测精度。

关键分工:T265 专司位姿追踪(哪怕遮挡),GoPro 只负责环境上下文捕获、演示验证与算法输入(不参与追踪),因此 GoPro 可灵活安装以维持视角一致,T265 则置于更受保护位置。

**机载执行端**去掉 T265(仅手持端有),强调广兼容:符合 ISO 标准的 **Flange Plate** 适配多种臂;**可插拔指尖(Plug-in Fingertip)**内部按夹爪形状定制、外部保持统一交互点;基于 Open X-Embodiment 设计 5 款指尖(如 xArm、Robotiq 2f-85),覆盖这些数据集 90%+ 的夹爪;**可调相机安装结构**(GoPro Robotic Mount + 延长臂,横向/纵向可调,最多三节)复现手持视角。

**视觉对齐规则**:"GoPro 鱼眼图像底边与夹爪指尖底边对齐",确保不同部署下观测近乎一致。

### 2. 软件框架:采集、同步与漂移校正

原始采集由三个 ROS 节点完成:camera node(GoPro,1920×1080@60fps)、tracking node(T265,200Hz,位姿 $\langle x,y,z,q_x,q_y,q_z,q_w \rangle$)、storage node(汇聚同步存为 HDF5)。

**子采样与同步**:统一 ROS 时钟 + 多线程缓冲 + 按最大公共频率同步子采样。T265(200Hz)与 GoPro(60Hz)统一下采样到 20Hz,每帧配最近邻 T265 位姿,时间偏差在亚毫秒级(小于 T265 采样周期 1/200 s 的一半)。

**T265 漂移校正** 两策略:①重初始化(在预定静止参考位重启 T265);②回环闭合(桌面上一条蓝色 3D 打印凹槽作视觉参考,重访时重对齐)。

**数据质量评估**:利用 T265 的四级置信度(Failed/Low/Medium/High),要求 ≥95% 位姿达 High;并对速度/加速度/相对朝向设阈值剔除突变。作者观察到光照对 T265 影响显著(暗光→置信度下降、漂移增大)。

### 3. 训练数据准备(三种表示)

从相机局部帧位姿 $\langle \mathbf{p}_i,\mathbf{R}_i \rangle$、机器人 URDF、相机到夹爪偏移 $\Delta_{c2g}$、夹爪中心已知位姿 $\langle \mathbf{p}_{b2g},\mathbf{R}_{b2g} \rangle$ 出发,推导绝对 TCP、相对 TCP 与绝对关节三种轨迹。相机在基座系下的绝对位姿:

$$\mathbf{p}_{\text{cam}}^{(i)} = \mathbf{p}_{b2g} + \mathbf{p}_i - \mathbf{R}_{b2g}\Delta_{c2g}, \qquad \mathbf{R}_{\text{cam}}^{(i)} = \mathbf{R}_{\text{base}}\cdot\mathbf{R}_i$$

再叠加相机-夹爪偏移得绝对 TCP,并由相邻帧作差得相对 TCP:

$$\mathbf{p}_{\text{ee}}^{(i)} = \mathbf{p}_{\text{cam}}^{(i)} + \mathbf{R}_{\text{cam}}^{(i)}\Delta_{c2g}, \qquad \mathbf{p}_{\text{rel}}^{(i)} = \mathbf{p}_{\text{ee}}^{(i+1)} - \mathbf{p}_{\text{ee}}^{(i)}, \qquad \mathbf{R}_{\text{rel}}^{(i)} = (\mathbf{R}_{\text{ee}}^{(i)})^{-1}\cdot\mathbf{R}_{\text{ee}}^{(i+1)}$$

用大白话说:相对 TCP 去掉了对全局参考系的依赖,数据分布更均匀、跨场景泛化更好;绝对关节则由绝对 TCP 经 URDF 逆运动学求得(以上一帧解作初值保连续)。

**连续夹爪开度(marker-based)**:测两枚 ArUco marker 的像素距离 $d$,对最大/最小开口的像素距离 $d_{\max},d_{\min}$ 归一化后乘最大物理开口 $G_{\max}$:

$$W = \frac{d - d_{\min}}{d_{\max} - d_{\min}} \times G_{\max}$$

用大白话说:用两个视觉标记的像素间距反推夹爪张多大,把开度测量从具体机械结构里解耦出来,换夹爪不用改硬件。若只检测到一枚 marker 则按中轴镜像补齐,均检测不到则插补,保证每帧都有有效值。

### 4. 针对第一人称数据的算法适配

FastUMI 数据有三个特点:近距第一人称(看不全整条机械臂,更依赖先验保运动学可行)、几何/场景多变、单目鱼眼深度信息有限。为此:

**Smooth-ACT(局部时序平滑)**:在 Transformer decoder 上加一层 GRU,decoder 输出 $\hat{a}$ 与 GRU 输出 $\hat{a}_{\text{GRU}}$ 都对齐真值:

$$\mathcal{L} = \|\hat{a}-a\|_1 + \|\hat{a}_{\text{GRU}}-a\|_1 + \lambda\,\mathrm{KL}(\mu,\log\sigma^2)$$

用大白话说:Transformer 抓全局时空模式,再让 GRU 把相邻帧动作抹平,减少第一人称下常见的关节突变/非法姿态。

**PoseACT(末端位姿预测)**:把 ACT 的绝对关节预测换成 TCP 表示(含绝对/相对两变体),带来平台无关性与数值稳定性,推理时相对位姿再经运动学映射回关节角。

**Depth-Enhanced DP**:原始 DP 在需精确深度的任务上易过早闭合/够不准。用开源 Depth Anything V2 为每帧补深度图,先裁掉黑边(鱼眼圆内矩形区域)再缩放到 448×448;把单通道深度转 3 通道伪彩色,与 RGB 共用 ViT-Base Patch 16(输入 224×224)CLIP 编码器编码后拼接嵌入;推理在 RTX 4090 上达 20Hz,无需额外深度传感器或多目。

**动态误差补偿(非平行夹爪)**:xArm/Robotiq 等夹爪闭合时 TCP 会沿局部 Z 轴平移约 1cm,造成手持采集与机载执行的视觉/动作错位。按当前开度 $W(i)$ 算补偿距离并沿 TCP 局部 -Z 轴修正位置,再 IK:

$$d(i) = d_{\text{close}} - \frac{d_{\text{close}} - d_{\text{open}}}{W_{\max}}\,W(i), \qquad \mathbf{p}_{\text{ee}}'^{(i)} = \mathbf{p}_{\text{ee}}^{(i)} - d(i)\,\mathbf{z}_{\text{axis}}^{(i)}$$

## 三、实验结果

数据集本身:**10,000** 条演示,22 个任务、19 类物体、12 种操作技能,单条约 6-12s(多数约 9s),由 5 名操作者用 3 台 FastUMI 设备采集,存为 HDF5(可转 Zarr);qpos 形状 $\langle \text{num\_timesteps},7 \rangle$ 对应 $[x,y,z,q_x,q_y,q_z,q_w]$。

**位姿精度(TABLE I,单位 mm,10 条轨迹均值)**:轻遮挡场景 T265 更优,重遮挡时 MINI 更稳。

| 任务 | RealSense T265 | RoboBaton MINI |
|---|---|---|
| Pick Cup(轻遮挡) | ≈10.5 | ≈15.2 |
| Open Container(部分遮挡) | ≈17.7 | ≈11.2 |
| Rearrange Coke(重遮挡) | 最高至 36 | 更稳定 |

**基线策略成功率(TABLE II,12 任务,各 200 条训练、测 15 次、新场景布置)**:DP 用相对 TCP、ACT 用绝对关节。

| 任务 | 类型 | DP(相对 TCP) | ACT(绝对关节) |
|---|---|---|---|
| Open Container | Hinged | 93.33 | 86.67 |
| Open Roaster | Hinged | 80.00 | 66.67 |
| Open Drawer | Hinged | 53.33 | 80.00 |
| Open Suitcase | Hinged | 40.00 | 86.67 |
| Rearrange Coke | Pick-Place | 80.00 | 86.67 |
| Fold Towel | Pick-Place | 93.33 | 73.33 |
| Pick Bear | Pick-Place | 80.00 | 20.00 |
| Unplug Charger | Pick-Place | 86.67 | 86.67 |
| Pick Lid | Pick-Place | 53.33 | 93.33 |
| Pick Pen | Pick-Place | 53.33 | 20.00 |
| Sweep Trash | Pick-Push | 46.67 | 6.67 |
| Open Ricecooker | Button Press | 20.00 | 80.00 |

**算法增强**:

| 增强 | 任务 | 基线 | 增强后 |
|---|---|---|---|
| Depth-Enhanced DP(TABLE III) | Pick Lid | 53.33 | 80.00 |
| Depth-Enhanced DP | Open Ricecooker | 20.00 | 93.33 |
| ACT 变体(TABLE IV) | Pick Bear | ACT 20.00 | Smooth-ACT 60.00 / PoseACT(Abs) 80.00 / PoseACT(Rel) 73.33 |
| ACT 变体 | Sweep Trash | ACT 6.67 | Smooth-ACT 26.67 / PoseACT(Abs) 53.33 / PoseACT(Rel) 60.00 |

**相机构型对比(TABLE V,各 50 条,原始 ACT)**:核心结论是"末端鱼眼单目 ≈ 多视角"。

| 构型 | Pick Bear | Open Container |
|---|---|---|
| D435i(第一人称) | 0 | 0 |
| GoPro 平镜(第一人称) | 6.67 | 93.33 |
| D435i(第一 + 第三人称) | 86.67 | 100.00 |
| GoPro 鱼眼(第一人称) | 80.00 | 100.00 |

**数据规模(TABLE VI,Pick Cup)**:200→20.00%,400→26.67%,800→53.33%,数据越多泛化越强。

## 四、局限性

作者明确列出三条:

1. **感知模态单一**:仅依赖视觉,缺力/触觉反馈,不适合易碎物体等高精度精细交互。
2. **机器人兼容受限**:目前只支持单臂/双臂,尚未适配移动操作臂、全身控制等更复杂形态。
3. **有线传输**:依赖有线连接,限制便携性与野外/独立作业;无线 + 机载计算是待补方向。

此外还有几处工程性隐忧(评审视角补充):T265 已停产,长期供货依赖 MINI 替代;回环闭合依赖桌面预置蓝色凹槽这类人工地标,泛化到任意环境有限;每任务仅 200 条、每任务测 15 次,统计置信度偏弱,部分任务(如 Sweep Trash 多步任务)两基线都难。

## 五、评价与展望

**优点**:①问题定位精准——UMI 的"硬件耦合 + VIO 脆弱"确是社区痛点,用现成 SLAM 相机(T265)直接读位姿,把最难调的定位环节工程化,是务实且可复制的路线;②真正开源了 10,000 条、22 任务的大规模真机演示,数据多样性在 UMI 类工作中领先,本身即有价值的训练资源;③"末端鱼眼单目 ≈ 第一+第三人称多视角"这一发现颇具启发,暗示宽视场单目在成本与性能间是划算的折中;④ marker-based 连续开度、非平行夹爪动态误差补偿等细节,直击手持到机载的 sim-to-real 式 embodiment gap。

**不足与开放问题**:①方法整体偏"系统 + 工程集成",单点算法创新有限——Smooth-ACT/PoseACT/Depth-DP 更像对 ACT、DP 的针对性缝合(GRU 平滑、TCP 表示、Depth Anything 补深度),缺乏对"为何第一人称数据更难"的理论刻画;②表 II 中 DP 与 ACT 互有胜负、缺乏统一 SOTA 对照(如未与原始 UMI 在同任务上直接同台),难以量化"重构"相对原 UMI 的净收益;③精度评估只在 3 个任务上做 mocap 对齐,且置信度筛选/回环闭合的人工介入较重,规模化采集时的自动化程度存疑。

**与公开工作的关系**:与原始 UMI(Chi et al.)是直接改进关系,思路上和 DexCap、DART/DexHub、Grasping in the Wild 等"低成本手持/穿戴式采集 + 视频里学"同属一条线;其"鱼眼单目替代多视角"的结论,可与近年 BridgeData V2、Open X-Embodiment、DROID 等大规模真机数据工作互为补充。可能的改进方向:引入力/触觉模态、无线化 + 机载算力、把回环地标换成更通用的重定位、以及用更强的 VLA(而非 ACT/DP)去吃这批第一人称数据以检验其上限。

## 参考

1. Chi et al. *Universal Manipulation Interface: In-the-Wild Robot Teaching Without In-the-Wild Robots.* arXiv:2402.10329, 2024.(直接前身)
2. Wang et al. *DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation.* arXiv:2403.07788, 2024.(手持/穿戴式采集)
3. Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).* arXiv:2304.13705, 2023.(本文 ACT 基线)
4. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR, 2023.(本文 DP 基线)
5. Yang et al. *Depth Anything V2.* arXiv:2406.09414, 2024.(Depth-Enhanced DP 所用深度估计器)
