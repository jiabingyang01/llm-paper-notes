# Robotic Telekinesis：观看 YouTube 人类视频学习机器人手模仿器

> **论文**：*Robotic Telekinesis: Learning a Robotic Hand Imitator by Watching Humans on YouTube*
>
> **作者**：Aravind Sivakumar\*, Kenneth Shaw\*, Deepak Pathak（\* 共同一作）
>
> **机构**：Carnegie Mellon University
>
> **发布时间**：2022 年 02 月（arXiv 2202.10448，v2 2022-07-24）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2202.10448) | [PDF](https://arxiv.org/pdf/2202.10448)
>
> **分类标签**：`从人类视频学习` `跨形态 retarget` `灵巧手遥操作`

---

## 一句话总结

用互联网上海量的**无标注**人手视频（Epic Kitchens + 100 Days of Hands，约 2000 万帧）离线训练一个"人手 3D 姿态 → Allegro 灵巧手关节角"的神经网络 retargeter，配合单目身体姿态估计做手臂 retargeting，从而实现只用**一台未标定 RGB 相机**、无手套、无 mocap 标记、任何未经训练的人都能实时（约 25 Hz）遥操作 16-DoF 灵巧手 + xArm6 机械臂完成 10 项灵巧操作任务；在 7/10 任务上成功率超过在线优化基线。

## 一、问题与动机

灵巧手遥操作面临"鸡生蛋"困境：要训练一个**在真实世界任意环境、任意操作者**都能用的遥操作系统，需要成对的"人手姿态—机器人手姿态"对应数据；但要采集这种成对数据，本身又需要一个可用的遥操作系统。

现有方案各有硬伤：动觉示教 / VR 设备 / 触觉手套 / mocap 都需要昂贵硬件、专门工装或专家操作，束缚了手的自然运动。最接近的 DexPilot（Handa et al., ICRA 2020）虽然是无标记视觉遥操作，但依赖**多台标定深度相机**的定制机架、且只能在特定实验室环境用。

作者的目标是把遥操作做到 **in-the-wild**：低成本、单目未标定彩色相机、任意环境、任意未训练操作者。**关键洞察**是：虽然缺成对的人—机器人数据，但互联网上并不缺人手数据——因此在**训练时**利用海量无标注互联网人手视频，学一个理解人手动作并把它 retarget 成机器人手臂轨迹的系统；训练全程只用**被动数据**（passive data），不需要在机器人上做任何主动微调。

## 二、核心方法

系统名为 **Robotic Telekinesis**，分两条支路（Fig 3）：一条做**手** retargeting，一条做**手臂** retargeting；每帧图像被 retarget 成两条指令，分别驱动 Allegro 手和 xArm6 臂。

### 1. 人手 2D 图像 → 3D 人手姿态

对操作者右手做裁剪（OpenPose 检测框），送入 FrankMocap（ResNet50 主干 + MLP 回归头），输出 MANO/SMPL-X 手模型参数：shape $\beta_h\in\mathbb{R}^{10}$、pose $\theta_h\in\mathbb{R}^{45}$、全局朝向 $\phi_h\in\mathbb{R}^3$。这一模块**对任意操作者、任意相机、任意环境**都能直接用、无需微调——泛化性来自它在数百万张互联网图像上训练。

### 2. 3D 人手姿态 → Allegro 手关节角（核心 retargeter）

面临三大挑战：欠约束（Allegro 16-DoF 与人手形态/尺寸/关节结构差异大）、通用性（要对任意人/任意任务有效）、效率（需 >15 Hz 实时）。**由于没有成对标签，无法直接监督回归**。

作者不做回归，而是把 retargeting 表述成一个**可行性目标**（feasibility objective），即"最优的机器人手姿态是最能模仿人手*功能意图*的那个"。沿用 DexPilot 的思路：定义 5 个 hand keypoints（4 个指尖，不含小指 + 手掌中心），枚举所有 keypoint 对得到 10 个 **keyvector**；能量函数度量人手 keyvector 与 Allegro keyvector 之间的**不相似度**：

$$
E\big((\beta_h,\theta_h),\,q_a\big) \;=\; \sum_{i=1}^{10} \big\lVert\, v_i^h - (c_i\cdot v_i^a)\,\big\rVert_2^2
$$

其中 $q_a$ 是 16 维 Allegro 关节角，$v_i^h$ 是人手第 $i$ 条 keyvector，$v_i^a$ 是 Allegro 手对应 keyvector，$c_i$ 是缩放超参（补偿人手与机器人手的尺寸比例；若追求"抓稳"则把指尖类 keyvector 的 $c_i$ 设小于 1，让手指多合拢一点以对物体施力）。

> **用大白话说**：不去逼机器人手指关节角等于某个"标准答案"（根本没有答案），而是逼机器人**指尖之间的相对空间关系**和人手指尖之间的相对关系一致——手比出什么"形状"，机器人手就摆成对应的"形状"。

关键在于：由于 Allegro 手的**前向运动学是关节角的可微函数**，$v_i^a$ 也就对 $q_a$ 可微，于是整个能量 $E$ 是关节角的**完全可微**函数，可以直接当损失来梯度下降训练一个网络。retargeter 网络 $f(\cdot)$ 是一个 MLP（3 个隐层 256/256/128，末端 tanh 把输出压到 $[-1,1]^{16}$ 再线性映射到各关节合法范围），输入 $x=\mathrm{concat}(\beta_h,\theta_h)\in\mathbb{R}^{55}$，输出 $y=q_a\in\mathbb{R}^{16}$。训练目标：

$$
\arg\min_f \; \mathbb{E}_{x\in\mathcal{X}}\big[\,E\big(x,\,f(x)\big)\,\big]
$$

> **用大白话说**：与其在部署时对每一帧在线跑梯度下降优化（慢、易陷局部最优），不如**离线**把"怎么解这个优化"蒸馏进一个网络。训练时可以不计成本地跑昂贵的可微能量，部署时网络一次前向只需约 3 ms（333 Hz），足够实时且平滑。

**训练数据**：从 Epic Kitchens（约 2000 万帧第一人称家务视频）和 100 Days of Hands（YouTube 视频链接集）用同一个 FrankMocap 模块提取（含噪的）人手姿态，再混入干净的 FreiHand（3 万+带真值手姿态）。最终训练集 = 3 万 FreiHand + 3 万 100 Days of Hands 采样。全程只用**源域**（人手）数据，无任何目标域（机器人手）标签。

### 3. 对抗训练做自碰撞规避

只最小化 keyvector 能量有时会输出手指互相穿插 / 戳进手掌的**自碰撞**姿态，而"是否自碰撞"不是关节角的可微函数，无法直接加进能量。做法：先用程序化采样的合法关节角向量 + 一个（不可微的）碰撞检测器生成二值标签，训练一个**自碰撞分类器**（MLP）；然后冻结它，当作"判别器 / 对手"，把自碰撞得分的梯度反传进 retargeter 网络，惩罚会自碰撞的输出（Fig 6）。这与 GAN 的生成器/判别器关系类似，但**判别器预训练后冻结**，避免了 GAN 联合训练的不稳定。Fig 10 的消融显示：自碰撞损失权重越大，自碰撞比例越低，但 keyvector 能量越高（姿态与人手越不像）——存在权衡，中间权重最好用。

### 4. 人体 → 机器人手臂 retargeting

单目"漂浮"相机带来两个问题：无深度（估不准手离相机多远）、无相机—机器人标定（相机坐标系到机器人坐标系无变换）。作者绕开这两点：不去解绝对位姿，而是估计**人的手腕相对于身体某锚点的相对变换**。取人体躯干为原点当作"机器人的躯干"，假设"人右腕相对人躯干的变换 = 机器人腕相对机器人躯干的变换",沿 SMPL-X 运动学链从躯干遍历到腕即可求得。这个简单的对应技巧对移动中的操作者也很鲁棒。之后剔除离群、对腕位姿做低通滤波,再用 SDLS（Selectively Damped Least Squares）IK 求解器（PyBullet 实现）算出 6 个臂关节角。手臂目标位姿还用 EMA 平滑：

$$
P_{\mathrm{EMA}} = \alpha\cdot P_{\mathrm{new}} + (1-\alpha)\cdot P_{\mathrm{EMA}},\qquad \alpha=0.25
$$

### 5. 控制与工程

控制栈（Fig 4）：视觉 retargeting 模块产出原始目标位姿 → IK → 低通滤波 → 插值 → 力/笛卡尔安全裁剪 → 平滑指令下发。硬件：xArm6 + Wonik Allegro Hand（4 个 3D 打印细指尖 + 3M TB641 防滑胶带），单台 Intel Realsense D415 仅用 RGB 流；AMD Ryzen 3960x + 128 GB + 双 RTX 3080Ti。用 ROS 发布订阅式并行数据流图把各节点并行化，把整体运行时从串行的约 3 Hz 提升到约 25 Hz。

## 三、实验结果

**对比基线 DexPilot-Monocular\***：除手 retargeting 改为**在线梯度下降**（Jax 实现，思路源自 DexPilot）外，其余（含单相机设置）与本文系统完全相同。不与完整多相机 DexPilot 比（其为特定机架设计，且代码未开源）。

**任务成功率 / 完成时间（专家操作者，每任务 10 次试验，1 分钟超时，Table I）：**

| 任务 | Ours 成功率 | 基线成功率 | Ours 用时 (s) | 基线用时 (s) |
|---|---|---|---|---|
| Pickup Dice Toy | **0.9** | 0.7 | **8.6** | 13.5 |
| Pickup Dinosaur Doll | **0.9** | 0.6 | **8.2** | 11.0 |
| Box Rotation | **0.6** | 0.3 | 37.2 | **16.3** |
| Scissor Pickup | **0.7** | 0.5 | 28.6 | **27.7** |
| Cup Stack | 0.6 | **0.7** | **21.5** | 22.9 |
| Two Cup Stacking | **0.3** | 0.1 | **27.3** | 45.0 |
| Pouring Cubes onto Plate | **0.7** | 0.5 | 36.8 | **13.8** |
| Cup Into Plate | **0.8** | 0.7 | **10.6** | 13.7 |
| Open Drawer | 0.9 | 0.9 | 23.6 | **14.9** |
| Open Drawer & Pickup Cup | 0.6 | **0.7** | 33.7 | **28.1** |

成功率上本文在 **7/10** 任务优于基线，其余 3 项相当。定性上专家反馈本文系统更流畅、更跟手；在线优化基线常因用上一帧作种子而陷入局部最优、输出手指戳进手掌的不自然姿态。

**retargeter 网络精度（DexYCB 500 序列，40 ms 时间预算）**：以"无限时间预算跑到收敛的 DexPilot-Monocular\*"为伪真值 oracle，本文神经网络 retargeter 相对 oracle 的 RMSE 为 **0.17 rad（约 10°）**，而受 40 ms 预算限制的在线优化基线 RMSE 为 **0.25 rad（约 14°）**——网络在有限算力下更接近最优。

**运行时（Table II，Hz）**：OpenPose 身体 29 / OpenPose 手 29 / FrankMocap 身体 16 / FrankMocap 手 27 / 身体 retargeter 16 / 手 retargeter **24**（在线梯度下降基线仅 10 Hz）。

**通用性人体实验**：10 名**从未用过系统**的操作者，各做 3 个任务（塑料骰子拾取 30 s、开抽屉 30 s、放杯到盘 60 s）各 7 次；每人上手到做完全部任务约 15 分钟。不同操作者之间行为无明显差异，验证了"任意人可用"。主要抱怨集中在拇指 retargeting 误差与手掌正对相机时的跟踪问题（单目深度歧义）。

## 四、局限性

- **单目深度歧义**：手内遮挡、手掌平行于相机时跟踪差；这是单相机设置无法根治的问题。
- **拇指 retargeting 系统性误差**：Allegro 拇指的形状与关节轴与人拇指差异大，能量函数对拇指匹配权重不足，导致复杂手势常复现失败。
- **依赖现成 3D 姿态估计器**：整条链的上限受 FrankMocap 质量约束；对训练集未见的罕见手势会偶发误差（好在这类一次性误差不随时间累积）。
- **手臂 retargeting 靠启发式对应**：把"人躯干→机器人躯干"手工设为锚点、并硬编码一套坐标系对应，缺乏原理性；手的全局朝向 $\phi_h$ 在手 retargeting 中未使用，靠手臂来近似匹配。
- **能量超参需调**：缩放常数 $c_i$ 需针对"美观手势"还是"稳抓"手动权衡。
- **本质是遥操作系统而非自主学习**：作者定位为收集示范、bootstrap 机器人自主学习的工具，本身不产生自主策略。

## 五、评价与展望

**优点**：这是"从被动互联网人类视频学跨形态 retargeting"这一范式的代表性早期工作。最漂亮的一招是把 DexPilot 式的**在线能量优化蒸馏成一个前向网络**——用可微前向运动学把能量当损失、用海量无标注人手数据训练，既保留了 keyvector 功能相似度目标的物理含义，又换来了实时性、平滑性和"不陷局部最优/总输出自然手势"的额外好处；DexYCB 上 0.17 vs 0.25 rad 的对比量化了这一优势。自碰撞分类器"预训练冻结当对手"是对 GAN 不稳定性的务实规避。整套系统只需一台 RGB 相机、任意人 15 分钟上手，显著降低了灵巧手示范采集门槛。

**缺点与开放问题**：(1) 精度天花板受限于单目 + 现成姿态估计器，拇指/深度问题是硬伤，难以支撑高精度接触密集任务。(2) 手臂那条支路的"躯干锚点"对应比手支路粗糙很多，本质是启发式，缺乏对全局位姿的原理性建模。(3) 缩放常数与自碰撞权重都要人工调，泛化到别的机器人手需重调。(4) 系统只解决"人→机器人"的运动映射，不涉及物体接触/力反馈,遥操作者只能靠视频反馈闭环。

**与其他公开工作的关系**：与 DexPilot（多相机 + 在线优化）相比，本文用单相机 + 离线蒸馏网络换取了 in-the-wild 可用性，但放弃了多相机的深度精度；两者共享同一个 keyvector 能量目标。方法上与 Neural Kinematic Networks（Villegas et al., 2018）的 cycle-consistency 动作 retargeting、Xirl（Zakka et al., 2021）的跨形态学习同属"跨 embodiment 迁移"谱系，但本文强调"训练时纯用被动人类数据、无需目标域标签或主动交互"。它也是后续一批"从人类视频/单目遥操作 bootstrap 灵巧操作"工作的先声。**可能改进方向**：引入时序信息或轻量深度先验缓解单目歧义；对拇指做形态感知的能量项或分部件加权；把手支路的可微 retargeter 思想推广到手臂支路，学一个"人体姿态→末端位姿"的网络替代启发式锚点；或把 retargeter 与下游模仿/强化学习联合优化，让"是否有利于任务成功"反过来监督 retargeting。

## 参考

1. Handa et al. *DexPilot: Vision-based Teleoperation of Dexterous Robotic Hand-Arm System.* ICRA 2020 —— keyvector 能量与在线优化基线的思想来源。
2. Rong et al. *FrankMocap: A Monocular 3D Whole-Body Pose Estimation System via Regression and Integration.* ICCV Workshops 2021 —— 本文手/身体 3D 姿态估计模块。
3. Romero et al. *Embodied Hands (MANO).* SIGGRAPH Asia 2017；Pavlakos et al. *Expressive Body Capture (SMPL-X).* CVPR 2019 —— 参数化手/全身模型。
4. Shan et al. *Understanding Human Hands in Contact at Internet Scale (100 Days of Hands).* CVPR 2020；Zimmermann et al. *FreiHand.* ICCV 2019 —— 训练数据来源。
5. Villegas et al. *Neural Kinematic Networks for Unsupervised Motion Retargeting.* 2018；Zakka et al. *Xirl: Cross-embodiment Inverse RL.* 2021 —— 相关跨形态 retargeting / 迁移工作。
