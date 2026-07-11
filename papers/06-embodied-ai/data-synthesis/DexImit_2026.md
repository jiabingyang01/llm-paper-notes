# DexImit：从单目人手视频中学习双手灵巧操作

> **论文**：*DexImit: Learning Bimanual Dexterous Manipulation from Monocular Human Videos*
>
> **作者**：Juncheng Mu, Sizhe Yang（共同一作）, Yiming Bao, Hojin Bae, Tianming Wei, Linning Xu, Boyi Li, Huazhe Xu, Jiangmiao Pang（通讯）et al.
>
> **机构**：Shanghai AI Laboratory；Tsinghua University；The Chinese University of Hong Kong；NVIDIA
>
> **发布时间**：2026 年 02 月（arXiv 2602.10105）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.10105) | [PDF](https://arxiv.org/pdf/2602.10105)
>
> **分类标签**：`人手视频转机器人数据` `双手灵巧操作` `具身数据合成`

---

## 一句话总结

DexImit 是一条**免深度、免相机参数**的四阶段自动流水线（4D 重建 → 动作中心调度 → 力闭合抓取合成 → 数据增强），把任意视角的**单目人手操作视频**（含互联网视频与文本生成视频）转成物理合理的**双手灵巧机器人**数据；在 100 个短程任务上重建成功率达 82%（ST2+FoundationPose++），用生成数据训练的 DP3 策略在长程 Pot / Stack Cups 真机任务上零样本成功率 78% / 52%，单条视频约 4 分钟即可处理。

## 一、问题与动机

双手灵巧操作的核心瓶颈是**数据稀缺**：灵巧手自由度高、遥操作困难、硬件昂贵,采集大规模真机数据的代价远高于简单夹爪。人手操作视频天然规模大、任务多样,直接编码了高层任务概念与底层操作动作,是扩展灵巧操作的诱人来源;近期文本到视频生成模型(Wan2.2、Veo3 等)更能按提示批量生成人手操作视频。

但从人手视频学习面临几个硬骨头:

- **具身鸿沟**:把人手当作异构末端直接预训练,视觉观测与动作空间差异会严重约束跨具身迁移。
- **依赖深度/精确重建**:一类工作重建 3D 手–物关键点流或物体轨迹再用运动规划/RL 复现,但大多依赖绝对深度,或对重建精度要求苛刻(否则 RL 训练失败)。
- **难处理复杂场景**:快速运动、遮挡、接触密集(尤其双手协同)时现有方法退化。

DexImit 的目标:**不需要任何额外信息**(深度、相机位姿/内参),把单目、任意视角的人手视频重建成近似公制尺度的手–物轨迹,再合成可零样本部署的双手灵巧机器人数据,覆盖工具使用(切苹果)、长程(调制饮品)、精细(叠杯子)等任务。

## 二、核心方法

整体范式:**Reconstruction → Scheduling → Action-Generation → Augmentation**(见原文 Fig 2)。

### 阶段一:4D 手–物交互轨迹重建

**分割**:用 Qwen3-VL 做视频理解识别涉及物体集合 $S_o$;用 Grounded SAM2 逐帧生成三类掩码——物体掩码 $m_o$、手掩码 $m_h$($h_0$ 左手、$h_1$ 右手)、桌面掩码 $m_t$(用于定世界系)。

**近公制尺度重建(免深度)**:输入 RGB 无深度,先用 SpatialTracker v2(ST2)估计无尺度深度 $D$。关键洞见来自"人手尺寸方差很小"这一先验——用手来估计公制尺度因子。以左手首帧为例,取首帧手点云 $\mathcal{P}^0_{h_0}$,用 Wilor 估计手网格 $\mathcal{M}^0_{h_0}$,通过 **align-render-align**(先对齐中心 → 投射平行光线取无遮挡可见顶点 → 再对齐)计算尺度因子:

$$s = \frac{PCA(\mathcal{M}^0_{h_0})}{PCA(\mathcal{P}^0_{h_0})}$$

把 $s$ 施加到 $D$ 得到公制尺度深度 $\hat D$。物体侧用 SAM3D 做图像到 3D 生成(得到近似位姿与尺度),再重复 align-render-align 把尺度对齐到 $\hat D$。

> **用大白话说**:相机不告诉你真实尺寸,但"人手多大"是几乎固定的常识。用一只已知真实大小的手网格,量一量它在重建点云里"看起来多大",两者之比就是把整段无尺度重建拉回真实公制的放大倍数。

**6D 位姿估计**:物体用 FoundationPose++ 的跟踪变体做逐帧跟踪保持时序一致;手因已有准确朝向只需恢复平移(align-render-align)。最终得到手轨迹 $\{p_h^t\}$ 与物体轨迹 $\{p_o^t\}$。

**世界坐标变换**:视频视角任意,需映射到固定世界系。相机系 $\mathcal{F}_c$ 到世界系 $\mathcal{F}_w$ 的变换 $\mathbf{T}_{c\to w}\in SE(3)$ 由三轴唯一确定:

- $z$ 轴:桌面点云法向,$\mathbf{z}_w = \mathbf{n}_t/\|\mathbf{n}_t\|$;
- $x$ 轴:首帧左右手位置连线的垂直平分方向 $\mathbf{d}_h$,投影到与 $\mathbf{z}_w$ 正交的平面上以保证正交性:

$$\tilde{\mathbf{x}}_w = \mathbf{d}_h - (\mathbf{d}_h^\top \mathbf{z}_w)\mathbf{z}_w,\qquad \mathbf{x}_w = \frac{\tilde{\mathbf{x}}_w}{\|\tilde{\mathbf{x}}_w\|}$$

- $y$ 轴:右手系约束唯一确定;
- **原点**:所有被操作物体轴对齐包围盒的中心,平移到预定工作区($x=0.6$)。

> **用大白话说**:桌面法线定"上",双手连线定"前",剩下一个方向按右手定则补齐,再把物体堆的中心搬到机器人够得着的位置——这样不管视频从哪个角度拍,都能拼到同一张桌子上。

### 阶段二:子任务分解与调度(Action-Centric Scheduling)

对时序长度、双手并发/异步程度、动作类型都不设限。用两个结构描述:

- 任务 $\tau = (\mathcal{E}_\tau, o_\tau, \mathcal{S}_\tau, k_\tau)$:$\langle$具身集合, 关联物体, 有序子动作列表, 当前子动作索引$\rangle$;
- 子动作 $s = (a_s, t_s)$:$\langle$动作类型, 起始帧$\rangle$,其中 $a_s \in \{\texttt{pregrasp}, \texttt{grasp}, \texttt{motion}, \texttt{release}\}$。

用 Qwen3-VL 从视频自动标注上述结构(长程任务可选人工标注提升精度)。**Algorithm 1**(动作中心调度)用优先队列在任意具身数、时域、动作组合下做无冲突的子任务分派,支持单手(抓)、协同双手(共抓一物,如端锅)、独立双手(并行倾倒)。

### 阶段三:源数据生成(力闭合抓取 + 关键帧运动规划)

**抓取合成**:候选生成–选择两阶段。沿用 BODex 的双层优化,以物体网格 $\mathcal{M}_o$、手 $h$、策略相关的活动接触集 $\mathcal{C}$ 建模,决策变量 $\mathbf{g}=\{(\mathbf{t}_h, \mathbf{R}_h, \mathbf{q}_h)\}$ 与接触力 $\{\mathbf{f}_c\}$,抓取图 $\mathbf{G}_c = [\mathbf{I}; (\mathbf{p}_c - \mathbf{m})_\times]\mathbf{O}_c$。给定目标 wrench $\{\mathbf{w}_j\}$ 求解:

$$\min_{\mathbf{g}, \{\mathbf{f}_c\}} \sum_{j=1}^{J} \kappa_w \left\|\lambda \mathbf{w}_j - \sum_{c\in\mathcal{C}} \mathbf{G}_c(\mathbf{g})\mathbf{f}_c \right\|_2^2 + \kappa_{con}\sum_{c\in\mathcal{C}}\psi(d_M(\mathbf{p}_c)) + \kappa_{coll}\Phi_M(\mathbf{g}) + \kappa_{hh}\Phi_{hh}(\mathbf{g})$$

四项分别惩罚:wrench 不平衡、接触点到物面距离、手–物碰撞 $\Phi_M$、双手互穿 $\Phi_{hh}$。VLM(Qwen3-VL)还会预测视频中实际参与接触的手指数 $N$,据此从完整接触点集选 $N$ 个活动接触,让合成抓取的手指数与人手示范一致。

**抓取排序 + 稳定性筛选**:候选按到重建人手位姿 $p_{\mathcal{E}_\tau}^t$ 的距离排序,越接近人手行为越靠前:

$$\mathcal{G}_{o_i}^{sorted} = \mathrm{sort}\left(\{g_j\},\, d(g_j, p_{\mathcal{E}_\tau}^t)\right)$$

其中距离度量(附录)$d(g, p) = \sum_{h\in\mathcal{E}_\tau}\lambda_t\|\Delta\mathbf{t}_h\|_2 + \lambda_r\theta_h$,平移误差 $\Delta\mathbf{t}_h$ 与旋转误差 $\theta_h = \arccos\!\big(\tfrac{\mathrm{trace}(\Delta\mathbf{R}_h)-1}{2}\big)$ 加权。然后按序做仿真 rollout 稳定性评估(采样物面点云,比较目标点云与仿真点云的平均欧氏偏差,低于阈值 $\epsilon$ 即稳定),取首个稳定候选为最终抓取 $g^*$。

**运动生成**:关键帧运动规划。把抓取后的手–物视为刚体,物体从 $t$ 到 $t'$ 的相对变换施加到末端:

$$\mathbf{T}_{o_i}^{t\to t'} = (p_{o_i}^t)^{-1} p_{o_i}^{t'},\qquad p_{ee,\mathcal{E}_\tau}^{t'} = \mathbf{T}_{o_i}^{t\to t'}\, p_{ee,\mathcal{E}_\tau}^{t}$$

> **用大白话说**:抓稳之后,手和物体就"焊"在一起了。人手视频里物体怎么动,机器人末端就照着这个相对位移动——不用重新规划轨迹,直接把物体的运动"翻译"成末端目标位姿,再交给运动规划器补全。

### 阶段四:全面数据增强(为零样本 sim2real)

- **物体位姿**:随机化位置与平移;
- **物体尺度**:对近公制源尺度(记为 1.0)施加 $[0.8, 1.2]$ 缩放。关键设计:缩放后**只调整手指关节角**、保留原抓取与运动,而**不为每个尺度重新生成**抓取(重新生成会在不同示范间引入冲突监督,拖慢甚至破坏 DP3 收敛);
- **相机位姿**:随机化朝向与位置;
- **观测(点云)**:随机删除 30% 物体点(模拟遮挡/缺失),对剩余点法向加 30% 噪声($\sigma=0.015$),复现真实 RGB-D(Azure Kinect)传感器的稀疏与噪声特性。

最终在增强数据上训练 **3D Diffusion Policy(DP3)**。

## 三、实验结果

真机平台:两台 UR5e 机械臂 + XHands 灵巧手 + Microsoft Azure Kinect 深度相机。

### 4D 物体轨迹重建成功率(100 个短程任务)

不同深度估计(TA=Trace-Anything, DA3=Depth-Anything v3, ST2=SpatialTracker v2)与位姿方法(RANSAC / PCR=ColorPCR / FPose=FoundationPose++)的组合:

| 组合 | Success | 组合 | Success |
| --- | --- | --- | --- |
| TA + RANSAC | 38% | DA3 + PCR | 45% |
| TA + PCR | 11% | **ST2 + PCR** | **76%** |
| VGGT + PCR | 32% | **ST2 + FPose** | **82%** |

ST2 深度时序一致性最好、FoundationPose++ 跟踪最准,组合达到最高重建精度 82%。

### 与基线的策略成功率对比(仿真,DP3,每任务 100 条示范)

基线 RigVid(视频估物体位姿 + 抓取合成,扩到双手)、DexMan(用 RL 复现视频动作,官方未开源、按原文复现):

| 方法 | Put Cup | Grapefruit | Fruits | Pour | Pot | Stack Cups |
| --- | --- | --- | --- | --- | --- | --- |
| RigVid | 96 | – | 100 | 50 | – | – |
| DexMan | 94 | 98 | 100 | 100 | – | – |
| **DexImit(本文)** | **100** | **100** | **100** | **100** | **78** | **52** |

RigVid 只能处理短程单手,双手/接触密集场景失败;DexMan 依赖 RL 对逐帧动作噪声极敏感,只在短程可行,更长的多步序列无法成功。DexImit 在短程近乎满分,并能处理长程 Pot(78%)与最难的精细长程 Stack Six Cups(52%)。

### 数据可用性:输入质量 × 任务难度(Fig 3,读自热力图,近似值)

输入质量从低到高:文本生成(Wan2.2 / Veo3)→ 野外随手拍 → 有意识定制拍摄 → 人工校正;任务难度递增:单手短程 → 独立双手 → 协同双手 → 协同双手长程。可用率(%)大致呈"越往下越高、越往右越低"的单调趋势:

| 输入质量 | 单手短程 | 独立双手 | 协同双手 | 协同双手长程 |
| --- | --- | --- | --- | --- |
| Text (Wan2.2) | 80 | 70 | ~30 | ~0 |
| Text (Veo3) | 80 | 80 | 60 | ~60 |
| 野外随手拍 | 90 | 80 | 60 | 10 |
| 定制拍摄 | 100 | 100 | 80 | 50 |
| 人工校正 | 100 | 100 | 90 | 80 |

结论:低复杂度任务下文本生成视频即可提供高可用数据;任务越难、时域越长,重建可用率随之下降;定制拍摄可保证简单任务近乎全可用、长程任务仍有可观成功;人工校正后长程与精细任务(如做饭、多杯堆叠)也能有效处理。

### 消融(Fig 6,四个真机 sim2real 任务)

- **去掉尺度增强(w/o scale aug)**:成功率显著下降——尽管重建已近公制,灵巧操作对空间感知极敏感,训练时必须暴露"公制一致的尺度分布"才能鲁棒真机执行;
- **每尺度重新生成抓取(regen grasp)**:退化剧烈,甚至低于不做尺度增强——不同尺度合成的运动互相矛盾,引入冲突监督害了模仿学习;
- **去掉视觉增强(w/o obj pcd noise / 相机位姿)**:真机性能下降,归因于 Kinect 点云固有噪声。

### 失败来源分解(Fig 7)与运行时(Table III)

四大失败来源:**任务分解 37%**、**6D 位姿估计 31%**、3D 生成 6%、抓取合成 4%(其余归为 Others)。运行时随视频时长增长:5s/10s/20s 视频总耗时约 173 / 201 / 257 秒,其中 3D 生成(约 60s)与抓取合成(11.3s)基本固定,深度/手位姿/分割/子任务分解随时长线性增加;**单条视频约 4 分钟**,可扩展到大规模生成。

## 四、局限性

- **误差沿流水线传播**:多模块串行执行,偶发误差累积会使数据不可用;
- **无法处理复杂 in-hand 操作**:单目视频遮挡严重、可观测性差,未设专门机制;
- **不支持软体/铰接物体**:3D 生成依赖 SAM3D,假设刚性几何;
- **仅桌面场景,不支持移动操作**:扩展需显式建模具身运动与环境动力学;
- **长视频需人工介入**:短程可靠,长程有时需人工修正 VLM 子任务分解或重建伪影。

## 五、评价与展望(学术视角)

**优点**。(1)系统集成度高且"免深度、免相机参数"的定位很实用——用"人手尺寸先验 + align-render-align"把无尺度单目重建拉回近公制,规避了此前工作对绝对深度/精确重建的强依赖,这是把互联网/生成视频真正拉进灵巧数据管线的关键一步。(2)把"抓取合成"与"轨迹复现"解耦:力闭合优化保证物理可行,再用刚体相对变换把物体运动"翻译"成末端目标,避免了 RL 复现(如 DexMan)对逐帧动作噪声的病态敏感——这是它能吃下长程双手任务的根因。(3)数据增强中"缩放只改手指关节、不重合成抓取"的洞见很到位,直接对应 DP3 对一致监督的需求,消融证据扎实。(4)Action-Centric Scheduling 统一了单手/协同/独立双手,是把 pipeline 推向真正双手协同的工程贡献。

**缺点与开放问题**。(1)评测强度偏弱:策略对比只用 DP3、基线只有两个(其一还是自行复现),真机任务数与轨迹条数有限,缺与直接把手当末端预训练(如 Being-H0、EgoVLA 类)以及关键点流方法(Dream2Flow、NovaFlow)的同台比较。(2)整条链路重度依赖多个外部大模型(ST2、SAM3D、FoundationPose++、Wilor、Grounded SAM2、Qwen3-VL),任一模块的失效都会级联——失败分解显示 6D 位姿(31%)+ 任务分解(37%)已占多数,鲁棒性天花板受制于这些组件。(3)"近公制"尺度仍建立在"人手尺寸方差小"的强先验上,对儿童/戴手套/异常视角可能失效,论文未做敏感性分析。(4)只在刚性桌面物体上验证,软体/铰接/in-hand 都被排除,而这些恰是灵巧手的价值高地。

**改进方向**。端到端可微的数据生成(替代串行 pipeline 以抑制误差传播,论文自陈为 future work);引入更强的铰接/可形变 3D 生成以解锁工具与柔性物体;把稳定性筛选从"仿真 rollout"升级为可微接触模型以降本;以及在生成视频质量–任务难度的交叉面上做更系统的 scaling 研究(现有 Fig 3 已给出很好的雏形)。总体看,DexImit 在"视频→双手灵巧数据"这条拥挤赛道上给出了一个工程完整、物理约束清晰、且能零样本落地的方案,主要价值在 pipeline 设计与"免深度近公制"这一实用取舍,而非单点算法创新。

## 参考

1. Hsieh et al. *DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos.* arXiv:2510.08475, 2025.(RL 复现路线,本文主要基线)
2. Patel et al. *RigVid: Robotic Manipulation by Imitating Generated Videos without Physical Demonstrations.* arXiv:2507.00990, 2025.(位姿+抓取合成基线)
3. Chen et al. *BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis via Bilevel Optimization.* ICRA, 2025.(抓取合成基础)
4. Ze et al. *3D Diffusion Policy (DP3): Generalizable Visuomotor Policy Learning via Simple 3D Representations.* arXiv:2403.03954, 2024.(策略骨干)
5. Xiao et al. *SpatialTracker v2: Advancing 3D Point Tracking with Explicit Camera Motion.* ICCV, 2025.(深度/重建核心组件)
