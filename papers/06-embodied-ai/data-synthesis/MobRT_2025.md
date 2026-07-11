# MobRT：面向可扩展移动操作学习的数字孪生框架

> **论文**：*MobRT: A Digital Twin-Based Framework for Scalable Learning in Mobile Manipulation*
>
> **作者**：Yilin Mei, Peng Qiu, Wei Zhang, WenChao Zhang, Wenjie Song†
>
> **机构**：北京理工大学自动化学院（论文仅明确标注通讯作者 Wenjie Song 的所属机构为 School of Automation, Beijing Institute of Technology；其余作者未单独标注）
>
> **发布时间**：2025 年 10 月（arXiv 2510.04592）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.04592) | [PDF](https://arxiv.org/pdf/2510.04592)
>
> **分类标签**：`移动操作` `数字孪生` `仿真数据生成` `全身运动规划` `Flow Matching` `Sim-to-Real`

---

## 一句话总结

MobRT 用数字孪生资产 + 虚拟运动链（VKC）+ 全身运动规划自动生成移动操作机器人的关节-底盘协同示教数据,并配合一个 Transformer/Flow-Matching 策略与少量真实数据联合训练,在真实机械臂上实现"300 条仿真 + 20 条真实"即可将开抽屉任务成功率从 20% 提升到 60%（点云策略）。

## 一、问题与动机

模仿学习依赖大规模高质量示教数据,但移动操作机器人（需要同时协调底盘移动与机械臂操作,处于高维、动态、部分可观的场景）的数据采集尤其困难：传统遥操作(teleoperation)耗时耗力且依赖专用硬件,VR 远程操作虽缓解部分问题但仍需大量人力,可扩展性和泛化性有限。已有的仿真数据生成框架（如 MimicGen、DexMimicGen、RoboTwin）主要聚焦固定底座的桌面操作任务,对移动操作场景支持有限。与之最相关的两个工作：

- **MoMaGen**：将 X-Gen 范式扩展到移动操作,通过回放人类示教生成多样轨迹,但缺乏全身控制(whole-body control),限制了对复杂任务的适用性,且依赖回放示教可能引入偏差,难以发现最优轨迹。
- **RoboTwin**：在双臂桌面精细操作上表现优异,但场景局限于桌面,不涉及移动操作特有的挑战。

此外,先前方法多将底盘移动与机械臂操作分离处理（先移动底盘到预定位置再操作）,这种"分离规划(separate planning)"会产生不自然、不连贯的仿真数据（例如开门时机械臂单独动作而底盘不配合,导致门开不到位）。MobRT 的目标即是用全身控制(whole-body control)自动生成物理一致、动作连贯的大规模移动操作示教数据,并建立可评测的 benchmark。

## 二、核心方法

MobRT pipeline 分三部分：(1)仿真中生成示教数据；(2)少量真实数据用于自适应；(3)混合数据联合训练策略。

**数字孪生资产与标注**：物体资产来自 PartNet-Mobility、UniDoorManip（铰接物体,如柜子、抽屉、洗碗机）以及 RoboTwin-OD（通过 AIGC 从简单 2D RGB 图像生成的刚体物体）；论文对实验场景中的物体采用与 RoboTwin-OD 相同的 AIGC 方法生成数字孪生资产,并用 CoACD 做凸分解以保证物理交互一致性。每个物体标注 6-DoF 抓取位姿；对容器、盘子等部分刚体物体额外标注"功能轴(functional axis)"表示其预期用途（如放置方向）。

**操作动作生成**：区分两类交互——(a) 刚体-刚体交互（如把容器放到盘子上）：通过对齐两物体的功能轴简化为轴对齐问题；(b) 铰接部件操作（如开门/抽屉）：使用 Virtual Kinematic Chains(VKC)框架。末端执行器抓住铰接连杆后与之刚性连接,采样从当前关节值到目标关节值的中间序列,利用铰接运动学模型计算连杆位姿轨迹 $\{T_{link}\}_{t=1}^T$,再映射为末端位姿轨迹：

$$
\{T_{eef}\}_{t=1}^T = \{T_{link}(\theta) \cdot T_{link}^{-1}(\theta_{init}) \cdot T_{eef}^{init}\}_{t=1}^T
$$

用大白话说：把"怎么开门"转化为"末端执行器沿铰接连杆的运动学轨迹走一遍"的轨迹跟踪问题,避免手工设计末端运动的繁琐调参。

**全身运动规划**：为使底盘与机械臂协同运动,定义优化问题

$$
\min_{x_{[1,T]}} \; \mathcal{C}_{eef}(x_T, T_{eef}) + \sum_{t=1}^{T} \big[\mathcal{C}_{smooth}(x(t)) + \mathcal{C}_{base}(x(t))\big] \quad \text{s.t. } x(1)=x_{init},\; x_{min}\le x(t)\le x_{max}
$$

其中 $\mathcal{C}_{eef}$ 驱动末端到达目标位姿,$\mathcal{C}_{smooth}$ 约束控制平滑,$\mathcal{C}_{base}$ 是底盘软约束（如推门时保持底盘朝向固定、拾放时避免与桌子碰撞）。规划结果再经 TOPP-RA 算法做时间最优参数化,保证动作动力学可行且连续性好。此外还支持底盘/机械臂分离规划的运动基元,与抓取开合动作组合可灵活编排多样任务流程。

**环境重置与过滤**：每条轨迹执行后按任务判据（拾放任务看两物体距离阈值,铰接任务看目标关节角度）校验成功与否,失败轨迹被过滤丢弃；重置时对物体位置/朝向/尺寸、机械臂初始关节、地面与物体纹理、光照等做大范围随机化,增强数据多样性与后续策略鲁棒性。

**策略学习**：提出一个新的 baseline 策略——Transformer 骨干 + Diffusion Policy 框架下的 Flow Matching。相比传统扩散模型模拟完整随机过程,Flow Matching 学习一个把噪声传输到专家动作的时间相关向量场,训练用 100 步、推理仅需 10 步。给定动作序列插值 $X_\tau = \tau A + (1-\tau)Z$,损失为

$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau, Z, A}\big\| v(\tau, X_\tau, o_t) - (A - Z) \big\|^2
$$

架构上：RGB 用预训练 ResNet-18 编码,点云先统一到同一坐标系融合后用共享 PointEncoder 编码（沿用 Lift3D 的做法),本体感知与噪声动作各自过 MLP；解码器交替使用自注意力(平滑动作序列)与交叉注意力(与多模态编码交互),扩散时间步经 AdaNorm 注入。

**与真实数据联合训练**：仿真轨迹主要由运动规划生成,难以覆盖真实系统的动力学、传感器噪声、执行延迟,因此用少量真实示教 $D^m_{real}$ 与仿真示教 $D^m_{sim}$ 等概率采样联合训练：

$$
\mathbb{E}_{(o^i,a^i)\sim D^m_{sim}}\big[\mathcal{L}(a^i,\pi^m_\theta(o^i))\big] + \mathbb{E}_{(o^i,a^i)\sim D^m_{real}}\big[\mathcal{L}(a^i,\pi^m_\theta(o^i))\big]
$$

真实数据额外做深度扰动（模拟边缘伪影、随机空洞）以逼近 Intel RealSense 噪声,并对仿真/真实点云统一做体素降采样与统计离群点剔除以对齐分布。仿真环境用 ManiSkill3（GPU 光追,支持 RGB-D/分割数据高效采集）。训练用单张 RTX A6000,batch size 64,训练 100K 步,Adam,weight decay 1e-6,warmup 500 步至峰值 lr 1e-4 后余弦衰减到 1e-6,梯度裁剪范数 10。真实平台为 Galaxea A1 机械臂 + Ranger Mini 3 移动底盘,3 个 Intel RealSense D435i(两个后方全局视角 + 一个手腕近距离)。

## 三、关键结果

**Benchmark 任务**：Open Cabinet Drawer、Container Place、Open Dish Washer 三个代表性任务,分别在 50/100/200 条示教下训练四种 baseline(ACT、DP、DP3、iDP3),每策略 30 次 rollout 评测成功率。

| 任务（200 demos） | ACT | DP | DP3 | iDP3 |
|---|---|---|---|---|
| Open Cabinet Drawer | 60.0% | 50.0% | 53.3% | 43.3% |
| Container Place | 70.0% | 46.7% | 13.3% | 3.3% |
| Open Dish Washer | 50.0% | 40.0% | 33.3% | 23.3% |
| **平均** | **60.0%** | 45.6% | 33.3% | 23.3% |

数据量与成功率呈明显正相关（如 ACT 在 Open Cabinet Drawer 上从 50→200 条示教,成功率 23.3%→60.0%）,验证了 MobRT 生成数据的有效性。

**MobRT 自研策略**（RGB-based / Point Cloud-based）在同等设置下平均成功率显著超过最优 baseline：50 条示教 46.7%/41.1%（较最优 baseline 分别 +26.7/+21.1 pt）,100 条 66.7%/50.0%（+23.4/+6.7 pt）,200 条 76.7%/61.1%（+16.7/+1.1 pt）,尤其在低数据量场景优势更明显。

**点云预处理消融**：DP3/iDP3 在 Container Place 任务上对背景噪声/杂乱高度敏感,加前景分割后成功率大幅提升（如 DP3 在 200 条示教下 13.3%→73.3%,+60 pt；iDP3 3.3%→60.0%,+56.7 pt）,说明其 max-pooling 式点云编码难以应对非桌面场景的杂乱背景,而 MobRT 提出的策略采用 tokenize 点云而非 max pooling,能在无需前景分割的情况下保持鲁棒性。

**真实机器人 Sim+Real 联合训练**（Open Cabinet Drawer 任务,10 次 rollout 评测）：

| 训练数据 | iDP3 | 点云策略(MobRT) |
|---|---|---|
| 仅 20 条真实示教 | 10.0% | 20.0% |
| 20 条真实 + 300 条 MobRT 仿真 | 40.0% | 60.0% |

即少量真实数据配合 MobRT 生成的仿真数据可显著提升真实场景成功率（点云策略 20%→60%),且训练后策略对光照变化、抽屉高度变化表现出一定适应性。

## 四、评价与展望

**优点**：MobRT 针对移动操作这一相对被忽视的场景,系统性地把"全身运动规划"引入自动化示教生成流程,直接对比展示了分离规划(先移动底盘再操作)带来的不自然/不完整开门问题,论证较为直观；标注功能轴 + VKC 的组合把拾放与铰接操作统一简化为末端轨迹跟踪问题,降低了硬编码动作的调参负担；实验设计较完整,既有仿真 benchmark(数据量-成功率相关性)、又有点云预处理消融、也有真实机器人 sim+real 联合训练的验证,形成从数据生成到策略、再到真实迁移的闭环。

**局限与开放问题**：(1)任务种类仍然有限,仅覆盖两类铰接物体（柜子、洗碗机）与容器放置这一种拾放任务,尚未验证在更长时程(long-horizon)、多步骤复合任务上的可扩展性；(2)真实世界评测规模较小（单任务、10 次 rollout、20 条真实示教）,论文自身也承认"real-world performance has yet to be fully explored";(3)全身运动规划的软约束（$\mathcal{C}_{base}$ 中固定朝向、避碰等）需针对任务手工设计,尚未看到通用化或自动学习约束的机制；(4)与 MoMaGen、RoboTwin 等同类工作相比,MobRT 的优势主要体现在"引入全身控制"和"聚焦移动底座",但尚缺少与这些方法在同一 benchmark 上的直接数值对比,目前的比较停留在方法论层面的定性讨论。作者在结论中提出未来将扩展任务范围与时程,并引入强化学习进一步提升真实场景下的鲁棒性与适应性,这也是当前"仿真自动生成示教 + 少量真实数据联合训练"范式在移动操作方向上一个值得跟进的方向。

## 参考

- MoMaGen: Generating demonstrations under soft and hard constraints for multi-step bimanual mobile manipulation (RSS 2025 Workshop)
- RoboTwin: Dual-arm robot benchmark with generative digital twins (CVPR 2025) / RoboTwin 2.0 (arXiv:2506.18088)
- MimicGen: A data generation system for scalable robot learning using human demonstrations (arXiv:2310.17596)
- DexMimicGen: Automated data generation for bimanual dexterous manipulation via imitation learning (arXiv:2410.24185)
- Mobile ALOHA: Learning bimanual mobile manipulation using low-cost whole-body teleoperation (CoRL 2024)
