# Video2Sim2Real：从单段人类视频全栈式自主习得灵巧操作技能

> **论文**：*Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single Human Video*
>
> **作者**：Yunhai Han, Jianuo Qiu（共同一作）… Kenneth Shaw, Matthew Gombolay, Zsolt Kira, Harish Ravichandar（共同通讯）et al.
>
> **机构**：Georgia Institute of Technology；University of Pennsylvania；Toyota Research Institute；Carnegie Mellon University
>
> **发布时间**：2026 年 06 月（arXiv 2606.08828）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.08828) | [PDF](https://arxiv.org/pdf/2606.08828)
>
> **分类标签**：`从人类视频学习` `灵巧操作` `Sim-to-Real` `数字孪生` `残差RL`

---

## 一句话总结

只需**一段** RGB-D 人类操作视频、无需任何机器人示教或专家标注,系统就能自动重建仿真数字孪生、用 **object-centric 关键帧精修**修正 retarget 出来的机器人轨迹,再用 **IL 负责几何 gap、残差 RL 负责手指级接触物理 gap 的解耦策略**完成 sim-to-real,七个日常灵巧操作任务上仿真平均成功率 91.4%、真实世界在物体位姿扰动下平均 95.7%(对照 FoundationPose 仅 1.4%)。

## 一、问题与动机

从人类操作视频学习灵巧操作有天然吸引力:视频易采集、天然覆盖多样任务/物体/场景。但把人手灵巧性直接迁到机器人有两大障碍:

1. **感知误差**:操作过程中严重的自遮挡与手-物遮挡会显著劣化人手/物体运动估计。
2. **具身鸿沟(embodiment gap)**:即便 retarget 到机器人关节,由于运动学与接触动力学差异,机器人复现的关节轨迹往往无法诱发与人类相同的物体交互。

作者把已有"精修 retarget 轨迹"的工作按"对 retarget 轨迹的信任程度递减"排成一条谱系:(a) 在 retarget 轨迹上学局部残差;(b) 用人类运动作为奖励引导学新轨迹;(c) 只用 pre-contact 位姿初始化;(d) 完全丢弃人类运动、纯 RL 复现物体运动。实验发现:**(a)(b) 强受限于 retarget 轨迹质量,而 (c)(d) 因高维状态空间探索困难**。

更棘手的是 **sim-to-real gap**:所有精修都在仿真里做,数字孪生重建误差会传导到真实执行,在接触密集的灵巧操作里被放大。为此作者主张:精修阶段用一种既不盲信也不抛弃人类运动的**折中方案**(在稀疏关键帧上用物体信息优化机器人配置、再插值);部署阶段用**显式分工的解耦策略**(IL 管全局几何、残差 RL 管局部接触)。

## 二、核心方法

框架 Video2Sim2Real 由四个模块串成一条全栈流水线。

### 2.0 输入与数字孪生

每个任务只假设可获得:RGB-D 人类操作视频 $\mathcal{V}=\{(I_t,D_t)\}_{t=1}^{T}$、操作前拍的参考场景图 $(I_s,D_s)$、相机内参、以及相机/桌面/机器人基座之间的外参 $\{T_{c\to t},\,T_{c\to r},\,T_{t\to r}\}$。

**数字孪生重建**用现成基础模型完成:Gemini 做场景语义、SAM3 做分割、SAM3D 做三维重建。输出一个可直接进仿真器的元组

$$
\mathcal{S}=\Big(\{(q_i,M_i,\mathcal{G}_i,T^t_i,U_i)\}_{i=1}^{N},\,i_{\mathrm{manip}},\,i_{\mathrm{target}},\,\tau\Big)
$$

其中每个物体带语义描述 $q_i$、掩码 $M_i$、标准网格 $\mathcal{G}_i$、桌面系位姿 $T^t_i$、URDF 资产 $U_i$;还含被操作物索引 $i_{\mathrm{manip}}$、目标物索引 $i_{\mathrm{target}}$ 和任务类型 $\tau$。

**运动估计**:机器人轨迹用 HaMeR 估计人手 3D 关键点(相机系→机器人基座系),再用 Mink 逆运动学 retarget 到机器人关节;物体轨迹用 CoTracker 跟踪 2D flow 点,借深度 lift 到 3D 并转到桌面系。

> 用大白话说:先让一堆现成大模型把"这段视频里桌上有什么、长什么样、被怎么挪动"全自动搭成一个能跑物理的仿真副本,顺手把"人手大概怎么动"翻译成机器人关节的粗糙轨迹。

### 2.1 Object-centric 关键帧精修(核心创新一)

不直接去学"怎么修轨迹",而是**用物体信息在少数关键帧上反解出机器人该处于什么配置**,再把这些配置当锚点插值。

**关键帧识别**:基于物体运动检测三个语义关键帧——
- 接触帧 $T_c$:被操作物 2D flow 质心相对首帧发生首次持续位移;
- 交互帧 $T_i$:被操作物与目标物之间的相对运动出现持续变化;
- 脱离帧 $T_d$:接触后被操作物首次落到贴近桌面的高度阈值以下。

**关键帧精修**(按 $\tau\in\{$抓取, 推, 拉$\}$ 分别定目标):
- 接触帧:先在仿真里回放 retarget 轨迹恢复"手-物"变换 $T^c_{h\to m}$。抓取任务用物体几何配合 Lightning Grasp 生成候选抓取,经仿真过滤选出**物体跟踪误差最小的成功抓取**,得到修正后的 $\hat{T}^c_{h\to m}$;推/拉任务则从 3D 物体 flow 估计期望运动方向,选物体表面接触点,让指尖法向对齐该方向,得到接触帧手位姿 $\hat{T}^c_h$(手指关节不变)。
- 交互帧:用配准的 3D flow 对应做刚体 SE(3) 对齐估计物体位姿,再复合 $\hat{T}^c_{h\to m}$ 得 $\hat{T}^i_h$。
- 脱离帧:精修释放物体时的手位姿。

**轨迹插值**:把精修手位姿转回基座系、解 IK 得关节空间锚点,在锚点与原 retarget 轨迹之间插值生成最终轨迹。

> 用大白话说:与其硬学"人手轨迹哪里错了怎么补",不如抓住"接触、发生交互、松手"这三个决定成败的瞬间,直接问仿真里的物体"要产生这个效果、机器人该以什么姿态/抓法出现",把这几个正确的姿态钉死当锚点,其余部分沿用人类运动插值——既保留了人类动作的时间连贯性,又不被噪声轨迹拖累。

### 2.2 解耦式 sim-to-real 策略(核心创新二)

即便精修轨迹在仿真里成功,直接回放到真机仍会因几何/物理差异失败。作者在几何与物理参数上做 domain randomization,并**显式拆分策略角色**:全局几何 gap 主要交给 IL,局部控制与接触物理 gap 交给手指级残差 RL。

**IL 蒸馏关键帧机器人位姿**:用 mask-aware PointNet 风格残差网络,把原始物体点云直接映射到关键帧机器人手位姿,输出相对物体质心 $c$ 的平移残差 $\Delta\hat{p}$($\Delta p=p-c$)与 6D 旋转 $\hat{r}_{6D}$。监督残差位姿回归损失

$$
\mathcal{L}=\lambda_{\mathrm{pos}}\,\|\Delta\hat{p}-\Delta p\|_2^2+\lambda_{\mathrm{rot}}\,\|R(\hat{r}_{6D})-R(r_{6D})\|_F^2
$$

其中 $R(\cdot)$ 把 6D 表示转成旋转矩阵,$\lambda_{\mathrm{pos}}=\lambda_{\mathrm{rot}}=1$;训练时对输入点云加小坐标抖动与随机点丢弃,并在桌面系内把物体位置/朝向随机化($\pm5$ cm、$\pm10^\circ$)。**部署时 IL 只在机器人进入工作区前查询一次点云**,从而不被机器人自遮挡污染。

> 用大白话说:IL 学的是"看一眼物体点云,就说出机器人该到的关键位姿",专门吸收"真实物体摆放和仿真不一样"这种几何偏差;而且它在机械臂还没伸过去、视野干净时问一次就够,避免手臂挡住相机。

**手指级残差 RL 适配**:用 PPO 学一个 MLP 残差策略,每步在基座轨迹上叠加手指关节残差。观测

$$
o_t=[\,\bar{q}^{\mathrm{cur}}_t,\ \bar{q}_{t+1},\ c_t,\ \Delta x_t,\ \Delta\theta_t\,]
$$

即归一化的当前/下一步参考关节角、当前 3D flow 质心 $c_t$、以及当前物体运动增量 $(\Delta x_t,\Delta\theta_t)$;动作是手指关节残差;奖励是**任务无关的 3D flow 跟踪目标**——比较观测与估计的物体 flow 质心轨迹,外加残差动作惩罚;训练时随机化观测、动作、控制增益、物体网格尺度/质量/摩擦。

> 用大白话说:手臂大动作交给蒸馏好的基轨迹,RL 只负责"手指怎么微调才能抓稳/推动",目标就是让真实物体的运动轨迹跟仿真里期望的对上。分工后 IL 迁移几何、RL 专注接触,各自问题维度都变小,既比纯 RL 稳、又比纯 IL 省数据。

**真机推理四步**:① SAM3+CoTracker 采集并跟踪物体点云;② IL 从初始点云蒸馏关键帧位姿;③ 用 IK+插值调整精修轨迹;④ 每步用真机反馈跑残差 RL 修手指,再下发关节指令。

### 2.3 空间泛化模块(可选)

当物体被摆到离演示位置很远处时,用碰撞感知运动规划器 CuRobo,以数字孪生和精修轨迹为输入,为新配置生成免碰撞的可行机器人运动,支持杂乱多障碍场景下的空间泛化。

## 三、实验结果

**平台**:单个固定 RealSense D455 采 RGB-D;7-DoF Kinova Gen 机械臂 + 16-DoF Leap Hand;IsaacGym 仿真。**七个日常任务**:水果摆放(苹果、桃子)、牛排调味、玩具整理、纸巾盒递交、传书、取托盘。

### 3.1 仿真中的精修策略评估(Table 1)

与五类 RL 精修基线对比:Residual RL(RRL)、Deep-mimic RL(DM)、Pre-Contact Init(PCI)、Opt-Pre-Contact Init(OPCI)、Object-only(Obj);每类含 -F(仅 flow 跟踪奖励)与 -A(flow 跟踪 + approaching/contact/lifting 等辅助奖励)两个变体。每任务 10 次跑、跨任务取均值,指标为成功率、安全率、机械臂 RMS jerk、手 RMS jerk。

| 方法 | Success↑ | Safety↑ | Arm Jerk↓ | Hand Jerk↓ |
|---|---|---|---|---|
| **Ours** | **91.4** | **100.0** | **3.7** | **5.7** |
| RRL-F | 49.5 | 42.9 | 39.9 | 38.2 |
| RRL-A | 52.9 | 29.0 | 58.6 | 55.4 |
| DM-F | 19.5 | 54.8 | 24.6 | 87.4 |
| DM-A | 24.3 | 56.7 | 29.8 | 63.8 |
| PCI-F | 6.2 | 37.1 | 143.6 | 25.4 |
| PCI-A | 19.5 | 14.3 | 169.0 | 42.0 |
| OPCI-F | 1.9 | 33.3 | 321.9 | 22.5 |
| OPCI-A | 14.3 | 16.7 | 300.6 | 28.6 |
| Obj-F | 9.5 | 13.8 | 1153.0 | 19.4 |
| Obj-A | 15.7 | 17.6 | 523.0 | 13.7 |

本方法在四项指标上全部第一:成功率 91.4(次优仅 52.9)、安全率 100.0(次优 56.7)、机械臂/手 jerk 分别低到 3.7/5.7(基线普遍高一到两个数量级)。作者归因于关键帧优化对齐了任务意图、同时靠插值保留了人类运动的平滑性;而 RL 基线要么受噪声 retarget 轨迹拖累、要么困于高维状态空间探索。

### 3.2 Sim-to-Real 迁移评估

**仿真侧(Fig 4,50 组相同随机参数)**对比 Pure IL、Pure RL(finger)、Pure RL(arm+finger)、Ours(解耦 IL+RL):

| 方法 | 总体成功率↑ | 完成帧数↓ |
|---|---|---|
| Pure IL | ≈0.65 | 232.7 |
| Pure RL(finger) | ≈0.71 | 167.6 |
| Pure RL(arm+finger) | ≈0.68 | 179.0 |
| **Ours** | **≈0.85** | **149.1** |

解耦策略拿到最高平均成功率、次优安全率、最短完成时间、最连贯轨迹;且所有 sim-to-real 学习法都产出安全行为,反过来印证了精修策略的有效性。

**真机侧(Table 2,物体位姿扰动下每任务 10 次)**对比 FoundationPose(用重建网格 + FoundationPose 估位姿调整机器人)与本方法:

| 方法 | Apple | Peach | Steak | Toy | Tissue | Book | Tray | 平均 |
|---|---|---|---|---|---|---|---|---|
| FoundationPose | 0/10 | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | 1.4% |
| **Ours** | 10/10 | 9/10 | 10/10 | 8/10 | 10/10 | 10/10 | 10/10 | **95.7%** |

两者用同一手指级残差 RL 策略。结果表明:在接触密集任务里**直接依赖 FoundationPose 位姿估计极其脆弱**,而部署时的蒸馏学习使本方法即便物体位置略偏于视频演示位也能成功执行(局部泛化抓取)。消融另证明:单用 IL 或单用 RL 都不足以完成任务。

## 四、局限性

1. **精修只在稀疏关键帧上做**:对需要复杂手内重定向(in-hand reorientation)的任务不够,扩展到连续调整是重要方向。
2. **关键帧检测靠启发式**(物体运动阈值),对长时程灵巧操作鲁棒性有限,可用预训练的通用运动分析方法替代。
3. **重建流水线仅支持非铰接刚体**,引入更通用的重建(如 URDFormer/Phys-Anything 等)才能覆盖更多物体类型。
4. 系统仍依赖预放 AprilTag 做标定、依赖 RGB-D(未来可用深度估计支持纯 RGB 低成本设置、自动标定去掉标签)。
5. 接触动力学仅靠视觉 flow 反馈,残差 RL 未用触觉,面对严重手-物遮挡时的接触自适应有改进空间。

## 五、评价与展望

**优点**:(1) 真正的"全栈 + 单视频 + 零机器人数据 + 零专家干预"闭环,这是从人类视频学灵巧操作里少见的完整度,把重建、精修、sim-to-real、空间泛化一次性打通;(2) 两处设计都切中痛点——object-centric 关键帧精修回避了"直接精修高维轨迹"的两难,把优化局部化到少数语义帧;IL/RL 解耦把 sim-to-real 的两类误差(几何 vs 接触物理)分派给各自擅长的范式,显著降低了每部分的问题维度;(3) 真机 95.7% vs FoundationPose 1.4% 的对照极具说服力,直指"重建 + 位姿估计回放"这条 real-to-sim-to-real 主流路线在接触密集任务上的脆弱性;(4) flow 跟踪奖励任务无关、系数跨任务一致,工程上省调参。

**缺点与开放问题**:(1) 大量能力外包给现成基础模型(Gemini/SAM3/SAM3D/HaMeR/CoTracker/Mink),整体性能受这些上游模型上限约束,且非铰接刚体假设限制了任务面;(2) 关键帧只三类(接触/交互/脱离),对多阶段、多次抓放或倾倒/工具使用类任务的表达力存疑;(3) Fig 4 的部分定量结果读数于柱状图,精度需以正文/附录复核;(4) 与并行工作(如 X-Sim、SimToolReal、DexMimicGen、Real2Render2Real、Phantom 等 real-to-sim-to-real 与数据生成路线)相比,本文的差异化在"单视频 + 显式角色分工",但缺乏与这些方法的直接同任务对比。

**展望**:把关键帧精修推广到连续时序、用可学习运动分析取代启发式检测、引入触觉与铰接物体重建,是自然的下一步;更根本地,该框架给出了一个可复用的模板——"基础模型自动搭数字孪生 → 语义关键帧局部优化 → 误差按类型解耦迁移",对任何"从少量人类视频规模化生产可执行机器人技能"的方向都有参考价值。

## 参考

1. Y. Chen, C. Wang, Y. Yang, C. K. Liu. *Object-centric dexterous manipulation from human motion data.* arXiv:2411.04005, 2024.（object-centric 精修的直接对照/基线思想来源）
2. I. Guzey, Y. Dai, G. Savva, R. Bhirangi, L. Pinto. *Bridging the human to robot dexterity gap through object-oriented rewards.* ICRA 2025.（残差 RL + object-oriented reward 基线）
3. T. G. W. Lum, O. Y. Lee, C. K. Liu, J. Bohg. *Crossing the human-robot embodiment gap with sim-to-real RL using one human demonstration.* arXiv:2504.12609, 2025.（单示范 pre-contact init 路线）
4. B. Wen, W. Yang, J. Kautz, S. Birchfield. *FoundationPose: Unified 6D pose estimation and tracking of novel objects.* CVPR 2024.（真机对照基线）
5. B. Sundaralingam et al. *cuRobo: Parallelized collision-free minimum-jerk robot motion generation.* arXiv:2310.17274, 2023.（空间泛化所用运动规划器）
