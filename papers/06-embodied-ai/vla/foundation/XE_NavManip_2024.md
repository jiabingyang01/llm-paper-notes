# XE-NavManip：把跨本体学习推向极限——用统一目标条件策略打通导航与操作

> **论文**：*Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation*
>
> **作者**：Jonathan Yang, Catherine Glossop, Arjun Bhorkar, Dhruv Shah, Quan Vuong, Chelsea Finn, Dorsa Sadigh, Sergey Levine
>
> **机构**：Stanford University; UC Berkeley; Google DeepMind
>
> **发布时间**：2024 年 02 月（arXiv 2402.19432）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2402.19432) | [PDF](https://arxiv.org/pdf/2402.19432)
>
> **分类标签**：`跨本体学习` `目标条件模仿学习` `导航-操作联合训练` `diffusion policy` `零样本本体泛化`

---

## 一句话总结

在单一 Transformer + diffusion policy 架构上,把机械臂抓取、轮式/腿式导航、无人机、无人驾驶、移动双臂共 18 个数据集统一成"目标图像条件下的自我中心动作预测"任务,发现导航数据co-train 能给操作任务带来平均**20%**的成功率提升,操作数据 co-train 也能给导航带来**5-7%**的提升,且策略可零样本控制训练中从未出现的移动操作臂(Mobile ALOHA,50% 成功率)。

## 一、问题与动机

已有的跨本体（cross-embodiment）研究大多局限于"相似本体"之间的迁移,比如都是并联夹爪机械臂,或者都用同一套动作空间做参数化;像 Open X-Embodiment（OXE）这样的大规模操作数据集也只在操作域内部混合。本文提出一个更极端的问题:**导航和操作这两个看起来动作表示、观测模态、任务目标完全不同的领域,是否可以共享同一个策略,并从彼此的数据中获益？** 作者称之为 heterogeneous cross-embodiment(异构跨本体),以区别于以往只研究相似机械臂/相似动作空间的 cross-embodiment 工作。

动机来自一个直觉类比:无论是机械臂末端接近目标物体,还是移动机器人朝目标点前进,两者都需要"感知碰撞与几何关系"以及某种形式的 visual servoing——在自我中心（egocentric）观测下,把当前状态导向目标状态。如果操作用腕部相机、导航用前视相机,两者在"图像空间中同一方向的动作会引起同类型的图像变换"这一点上具有近似的等变性（equivariance）。

## 二、核心方法

**1) 统一为目标条件到达（goal-reaching）任务。** 对轨迹 $\tau \in D_{e}$ 中两个时间上接近的观测 $o_i, o_j$,定义"真值动作" $a^*$ 为生成这两帧图像的相机位姿之差,这个 $a^*$ 与本体无关（agnostic to embodiment）。理想情况下学习 $f(o_i,o_j)\to a^*$ 就能把所有导航、操作数据集统一到同一目标函数下。大白话说:不管是机械臂手腕相机还是移动机器人的车载相机,只要把"动作"定义成"相机从当前姿态挪到目标姿态需要走的位移",不同本体的数据就能塞进同一个回归目标里训练。

**2) 动作坐标系对齐。** 现实中不同数据集的相机外参、动作幅度、控制模式（绝对/增量、关节/笛卡尔）都不一致,理想的刚体变换对齐做不到,因此作者退而求其次:先把每个数据集的动作归一化到 $[-1,1]$,再通过人工抽样"观测-动作-下一观测"三元组、观察末端相对姿态变化,手动交换/取反各数据集的动作维度符号,使每个维度指向大致相同的物理方向（例如维度 0 统一代表"沿相机方向前进/后退"）。操作动作空间为 7 维（0-2 位移、3-5 旋转、6 夹爪开合）;导航动作从 $(x,y,\text{yaw})$ 的自我中心 waypoint 映射为 $a=(0, a[1], -a[0], 0,0,0,0)$,即把导航的"前进"对齐到操作的向下 $-z$ 方向,导航的"左"对齐到操作的"左" $+x$。

**3) 网络结构。** 两个 EfficientNet-b5 卷积编码器分别编码观测历史 $o_{t-k:t}$ 和目标图像 $o_g$（沿通道维拼接当前观测与目标观测后编码目标分支）,特征拼接后送入 Transformer,输出两个头:一个 diffusion policy 动作头 $\epsilon_\phi$ 预测未来动作序列 $a_{t:t+k-1}$（用于处理人类演示中的多模态/噪声）,一个 MLP 距离头 $d_\psi$ 预测到目标的时间步距离（仅导航侧使用,用于在拓扑地图上定位和选子目标,操作侧该头训练但不参与推理决策）。总损失:

$$\mathcal{L}(\theta,\phi,\psi) = \mathcal{L}_{\text{diffusion}}(\theta,\phi) + \lambda\, \mathcal{L}_{\text{distance}}(\theta,\psi)$$

其中 $\mathcal{L}_{\text{diffusion}} = \|\epsilon_k - \epsilon_\phi(f_\theta(o_{t-k:t}, o_g), a_t^0+\epsilon_k, k)\|_2^2$,$\mathcal{L}_{\text{distance}} = \|d_\psi(f_\theta(\cdot)) - d_t\|_2^2$,取 $\lambda=0.001$。大白话说:动作头用 diffusion 学"从当前观测和目标观测预测未来一小段动作序列",距离头只是个辅助任务,权重很小,不会干扰主任务。目标图像在训练时从当前帧未来 20-40 步内均匀采样,保证方向可辨识,也让方法能扩展到 OXE 中不带真实"目标"标注的长序列数据。

**4) 数据配比。** 训练数据由自采操作集（WidowX250S/ViperX300S,300/400 条轨迹）、OXE 中 9 个操作数据集（Bridge、Fractal、Taco Play、Jaco Play、RoboTurk、NYU Door Opening、Viola、Berkeley Autolab UR5、Toto）、GNM 系列导航数据集（SACSoN、GO Stanford、SCAND、RECON、Cory Hall、Seattle、TartanDrive）以及 BDD100k 自动驾驶数据组成,导航被上采样至占总配比约 50%,以防止操作域淹没导航域。

## 三、关键结果

**操作任务(5 个真实任务,20 次 rollout 取均值),不同数据混合的成功率：**

| 数据混合 | 均值 | Two-Object | Cluttered | Toy Kitchen | Shelf | Novel Cluttered |
|---|---|---|---|---|---|---|
| Manip-only | 51% | 70% | 65% | 70% | 30% | 20% |
| GNM + Manip | 64% | 80% | 75% | 65% | 50% | 50% |
| GNM + Driving + Manip | 71% | 80% | 80% | 80% | 65% | 50% |

联合训练相对纯操作数据带来约 **20 个百分点**的平均提升,在分布外任务（Novel Cluttered Grasp、Shelf Manipulation）上提升最明显。

**导航任务(4 个本体：DJI Tello、Clearpath Jackal、LoCoBot、Unitree Go1),加入操作数据的效果：** 均值从 GNM-only 的 74% 提升到 GNM+Manip 的 81%（+7%）,GNM+Driving+Manip 为 79%。分本体看,GNM+Manip 相对 GNM-only 在 Jackal/LoCoBot/Go1 上分别 +7%/+15%/+17%;GNM+Driving+Manip 相对 GNM-only 分别 +12%/+7%/+1%。DJI Tello（相机差异大,需换评估场景）在两种配置下均为 95%,未见明显差异。

**零样本迁移到新本体：** 策略从未见过 Mobile ALOHA 数据,直接用 GNM+Driving+Manip checkpoint 分别以腕部相机跑操作头、以底盘相机跑导航头（按动作幅度阈值切换）,在 "Egg Nav/Pick/Place" 任务上取得 **50%** 成功率（桌子和鸡蛋均未出现在训练数据中）。

**其它关键消融：**
- 去掉目标图像条件（无条件基线）后,导航数据带来的操作迁移收益从 35 个百分点（GC: 55% vs 20%）骤降到 5 个百分点（UC: 45% vs 40%）,说明迁移收益强依赖目标图像条件（Table II）。
- 典型相关分析（CCA）显示,联合训练策略的 Transformer 特征与"到目标的真实时间距离"的 $R^2$ 相关系数在 Cluttered/Novel Cluttered/Shelf 上分别为 0.749/0.740/0.644,均高于纯操作策略的 0.733/0.703/0.614,支持"导航数据帮助策略学到更好的目标距离表征"这一假设（Table I）。
- 哪种导航数据更有效：室内/人行道类数据（SACSoN、GO Stanford、SCAND）迁移收益显著,户外小径/越野类数据（Seattle、TartanDrive）几乎无正迁移（Figure 7）。
- 模型规模：27M 参数模型操作/导航均值仅 33%/44%,186M 参数模型达到 68%/55%,呈明显正相关(Table XI/XII)。
- 动作离散化（RT-1 式 256-bin + 交叉熵）方案在 100M 参数以下几乎全部失败（成功率接近 0%）,仅 180M 参数模型有个位数成功率,提示该架构下 diffusion 头对小模型更友好（Table XIII）。
- 腕部+第三人称视角联合训练比纯腕部视角平均高 42 个百分点的成功率（Table VII）。

## 四、评价与展望

**贡献与优点：** 本文是较早明确检验"操作和导航能否共享同一策略并相互迁移"的工作,给出了一个简洁但有物理直觉的统一动作空间构造（自我中心相机位姿差),并通过 1000+ 次真实机器人实验、覆盖机械臂/轮式/腿式/无人机/移动双臂五类本体,提供了相对扎实的实证证据,而不仅是仿真或单一本体上的验证。CCA 相关性分析和目标条件消融尝试为"为什么导航数据有帮助"给出了机制性解释（更好的目标-距离表征),而不只是报告一个成功率数字,这在同类论文中并不常见。

**局限性（作者自陈 + 可观察到的问题）：**
- 框架无法处理自由度差异悬殊的本体（如多指灵巧手无法自然地压缩进这套简化动作空间);四足机器人也只是被当作整体朝向控制,而非关节级控制。
- 目标模态仅限于目标图像,不支持语言指令等更贴近人类使用习惯的任务描述方式。
- 导航侧提升幅度(5-7%)本身不大,作者也承认可能受评估轮次方差影响；跨域收益的因果性主要靠相关性分析（CCA)和消融间接支撑,并非严格的因果推断。
- 坐标系对齐依赖人工抽样观察来手动决定各数据集动作维度的符号/顺序,而非从相机外参自动推导刚体变换,可扩展性和精确性有限,作者也明确指出这是对"理想等变性"假设的近似。
- 模型规模（27M-186M 参数、EfficientNet-b5 视觉编码器）相较同期 RT-2、OpenVLA 等基于预训练 VLM 的十亿级参数 VLA 明显偏小,尚未验证结论在更大规模基础模型上是否依然成立。

**与其他工作的关系：** 方法与作者团队此前的导航基础模型 GNM/ViNT/NoMaD 一脉相承（沿用了 diffusion policy 动作头和拓扑地图目标选择机制),操作侧数据直接复用 Open X-Embodiment(OXE)。相较于 Open X-Embodiment/RT-X、Octo 等只在操作域内部做跨本体的工作,本文的差异化贡献在于把"跨本体"的边界从操作扩展到了导航乃至自动驾驶,是对"数据是否越多样越好"这一大规模机器人基础模型路线的一次跨域压力测试。

**开放问题：** 如何用自动化方式（例如基于相机内外参或自监督）替代手工坐标系对齐；如何把统一动作空间扩展到可变自由度（灵巧手、双臂协同)而不损失表达力；将目标图像条件替换/补充为语言条件后跨域迁移收益是否依然成立；以及在十亿参数级 VLM 骨干下,导航-操作跨域迁移的收益会放大还是被大规模预训练本身的知识覆盖掉,仍是未解之题。

## 参考

- Shah et al. *GNM: A General Navigation Model to Drive Any Robot*, ICRA 2023.
- Sridhar et al. *NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration*, 2023.
- Open X-Embodiment Collaboration et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, arXiv:2310.08864, 2023.
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023.
- Fu, Zhao & Finn. *Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation*, 2024.
