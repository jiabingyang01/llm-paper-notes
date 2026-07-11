# Mobile ALOHA：面向双臂移动操作的低成本全身遥操作学习

> **论文**：*Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation*
>
> **作者**：Zipeng Fu\*, Tony Z. Zhao\*, Chelsea Finn（\* 为共同第一作者/项目共同负责人）
>
> **机构**：Stanford University
>
> **发布时间**：2024 年 01 月（arXiv 2401.02117）
>
> **发表状态**：未录用（预印本，PDF 页眉仅标注 "January 2024"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2401.02117) | [PDF](https://arxiv.org/pdf/2401.02117)
>
> **分类标签**：`低成本遥操` `双臂移动操作` `模仿学习` `co-training`

---

## 一句话总结

作者把桌面级双臂系统 ALOHA 装到一台差速轮式底盘上，用"操作者腰部拴接底盘、双手操纵 leader 臂、身体反驱动轮子"的全身遥操作方案，以约 \$32k 采到高质量双臂移动操作数据；再把这些少量（每任务 20–50 条）in-domain 数据与已有的 825 条静态 ALOHA 数据做 **co-training**，即可让 ACT/Diffusion Policy 等常规模仿学习方法在 7 个复杂长程家务任务上普遍达到 80% 以上成功率，co-training 相对不联合训练平均带来约 34% 的绝对成功率提升（个别子任务如 Turn On Faucet、Press Button 从 0–5% 提到 80–100%）。

## 一、问题与动机

模仿学习在**桌面级**操作上已很成功，但要做真正有用的家务任务（往柜子里放重锅、坐电梯、扫卫生间）需要同时具备**移动性**与**双臂灵巧性**的全身协同，而这一方向长期受两个瓶颈阻碍：

1. **硬件不可及**。现成的双臂移动机器人（PR2、TIAGo）动辄 20 万美元以上；且缺乏"即插即用"的全身遥操作接口——现有方案要么用动捕重定向（只控单臂）、要么用手柄/键盘（不支持双臂或全身遥操作）、要么需要 VR 头显/力反馈设备。作者指出目前没有低成本方案能采集双臂移动操作的全身专家演示。
2. **学习层面未被验证**。虽然 diffusion、transformer 类高表达策略在精细桌面操作上表现好，但加入底盘自由度后，臂-底盘的耦合更复杂，底盘位姿的小偏差会被放大成末端执行器的大漂移（作者实测 1m 半径 180° 转弯开环回放，底盘平均误差 >10cm），此前没有工作在双臂移动操作上给出既便宜又可信的软硬件方案。

因此本文同时解决硬件（低成本全身遥操作系统 Mobile ALOHA）与学习（用静态数据 co-training 提升数据效率）两个问题。

## 二、核心方法

### 2.1 硬件：Mobile ALOHA 全身遥操作系统

在原 ALOHA（4 臂 leader-follower 双臂遥操作套件）基础上加一台移动底盘。四条设计原则：**移动**（速度接近人步行，约 1.42m/s）、**稳定**（能操纵重锅重柜）、**全身遥操作**（双臂 + 底盘可同时遥操作）、**无线自持**（板载电源与算力）。

关键工程选择：
- **底盘**：AgileX Tracer AGV（差速驱动，最高 1.6m/s，负载 100kg，美国售价约 \$7,000，比同类 AGV 便宜 5 倍以上），底部压低配重防倾覆。
- **全身遥操作机制**：把操作者腰部拴接到底盘。操作者双手已被两条 leader 臂占用，于是靠身体前后移动/侧身来**反驱动（backdrive）**低摩擦的轮子实现底盘遥操作——底盘与双臂由此可**同时**独立控制，且底盘撞到物体时能给操作者粗略的力反馈。不需要 FPV 眼镜或力反馈设备，可连续遥操作数小时（做三道菜、清洁公共卫生间、洗衣）。
- **自持化**：底座放 1.26kWh、14kg 电池（兼作配重），全部计算跑在一台消费级笔记本（Nvidia 3070 Ti / 8GB VRAM，i7-12800H）。3 个 Logitech C922x RGB 摄像头（480×640、50Hz）：两个装在 follower 手腕、一个朝前。
- **规格摘要**：14（双臂）+ 2（底盘）自由度，整机 75kg，垂直可达 65–200cm，可探出底盘 100cm，单臂负载 750g（双臂合力可举 1.5kg），1.5m 高处可施 100N 拉力，电池续航 12 小时。整机预算约 \$32k，与单台工业协作臂（如 Franka Emika Panda）相当，且软硬件全部开源。

### 2.2 学习：与静态 ALOHA 数据 co-training

策略把 16 维动作直接建模：14 维臂动作 $a_{\text{arms}}\in\mathbb{R}^{14}$（含两个连续夹爪动作）拼上底盘的 2 维目标线速度/角速度 $a_{\text{base}}\in\mathbb{R}^{2}$。观测 $o^i$ 为两路手腕 RGB + 一路顶部朝前 RGB + 双臂关节位置。这样几乎不改实现就能直接套用已有深度模仿学习算法。

**核心配方**：借用大量已存在、更易采集的**静态双臂数据**（来自 RT-X release 中的静态 ALOHA 数据集，共 825 条演示，任务如封 Ziploc 袋、撕纸巾、开塑料杯盖、玩乒乓球、装电池、拧螺丝等，与 Mobile ALOHA 任务完全不重叠，且相机架法/背景/朝向都不同）与每个移动任务的少量 in-domain 数据 $D^m_{\text{mobile}}$ 一起训练。训练目标：

$$\mathbb{E}_{(o^i,a^i_{\text{arms}},a^i_{\text{base}})\sim D^m_{\text{mobile}}}\big[L(a^i_{\text{arms}},a^i_{\text{base}},\pi^m(o^i))\big]+\mathbb{E}_{(o^i,a^i_{\text{arms}})\sim D_{\text{static}}}\big[L(a^i_{\text{arms}},[0,0],\pi^m(o^i))\big]$$

**用大白话说**：以相等概率从"静态数据"和"本任务移动数据"里各抽一批一起训。静态数据没有底盘动作，就把底盘那 2 维标签补零 $[0,0]$；静态数据多一路前置相机，就丢掉让两边都是 3 路相机；动作归一化只用移动数据的统计量。就这么朴素——不做任何特殊的图像或双臂动作对齐处理——静态桌面数据里"抓取/接近物体"的运动先验就能迁移到移动任务里，尤其能利用手腕相机带来的不变性。

**实现细节**：batch size 16，静态与移动采样各约 50%。该配方与多种 base 模仿学习方法通用：ACT、Diffusion Policy、VINN 均可。所有方法都用 action chunking（一次预测未来一段动作序列），且作者利用 chunk 处理硬件延迟——底盘速度有延迟而位置控制的臂延迟小，于是对 $d$ 步的底盘延迟，机器人先执行 chunk 中前 $k-d$ 步的臂动作、后 $k-d$ 步的底盘动作。

## 三、实验结果

7 个任务的规模（演示条数 / 每条步数·时长）：Wipe Wine（50 / 1300 步·26s）、Cook Shrimp（20 / 3750 步·75s）、Rinse Pan（50 / 1100 步·22s）、Use Cabinet（50 / 1500 步·30s）、Call Elevator（50 / 2250 步·45s）、Push Chairs（50 / 2000 步·40s）、High Five（20）。每个成功率由 20 次试验计（Cook Shrimp 为 5 次）。

**表 1：ACT 全任务成功率(%)，co-train vs. 无 co-train**

| 任务 | Co-train | No Co-train | 提升 |
|---|---|---|---|
| Wipe Wine | **95** | 50 | +45 |
| Cook Shrimp | **40** | 20 | +20 |
| Rinse Pan | **80** | 0 | +80 |
| Use Cabinet | **85** | 85 | 0 |
| Call Elevator | **95** | 0 | +95 |
| Push Chairs | **80** | 0 | +80 |
| High Five | **85** | 85 | 0 |

7 个任务中 5 个因 co-training 显著提升（+45/+20/+80/+95/+80），另 2 个持平。作者强调 co-training 对"精细操作是瓶颈"的子任务尤其关键，例如：

- Rinse Pan 的 **Turn On Faucet** 子任务：co-train 80% vs. 无 co-train 0%；
- Call Elevator 的 **Press Button**（按 2cm×2cm 的按钮）：co-train 100% vs. 无 co-train 5%。

**表 2：跨方法通用性（Wipe Wine / Push Chairs 全任务成功率 %）**

| 方法 | 训练 | Wipe Wine | Push Chairs |
|---|---|---|---|
| VINN + Chunking | Co-train | 15 | **60** |
| | No Co-train | **20** | 40 |
| Diffusion Policy | Co-train | **65** | 100 |
| | No Co-train | 35 | 80 |
| ACT | Co-train | **95** | 100 |
| | No Co-train | 50 | 100 |

ACT 与 Diffusion Policy 都从 co-training 明显获益（DP 在 Wipe Wine/Push Chairs 分别 +30/+20）；VINN 因只 co-train 视觉表征、动作检索机制无法利用 out-of-domain 静态数据，结果参差（Wipe Wine 反被 -5，Push Chairs +20）。Diffusion Policy 在 Wipe Wine 只到 65%，作者归因于 50 条演示对扩散策略偏少（既往工作常需 250+ 条）。

**其余消融**：
- **数据效率（图 4）**：co-train 用 35 条 in-domain 演示（70%）即可超过无 co-train 用 50 条演示（50%），领先 20%。
- **混合比鲁棒（表 3，Wipe Wine/ACT）**：静态数据占比 30/50/70% 时成功率 95/95/90%，对采样比不敏感，减少了引入新任务时的调参负担。
- **Co-train 优于 Pre-train（表 4，Wipe Wine）**：Co-train 95% > 先在静态数据预训练再微调 40% ≈ 什么都不做 50%；作者认为微调阶段网络会遗忘静态经验。
- **用户研究**：8 名 CS 研究生（5 女 3 男，21–26 岁，4 人无遥操作经验）学操作陌生任务，5 次试验内完成时间即接近专家：Wipe Wine 46s→28s（-39%），Use Cabinet 75s→36s（-52%），说明系统易学易用。

## 四、局限性

作者在结论中自陈：
1. **仅单任务模仿学习**，机器人不能自我提升或主动探索获取新知识。
2. **演示来自两位专家操作者**，数据非高度次优/异质；从次优异质数据学习留作未来工作。
3. **占地面积偏大**（90cm×135cm），对某些狭窄通道过宽。
4. **follower 臂高度固定**，够不到低矮橱柜、烤箱、洗碗机；计划给臂增加升降自由度。
5. **底盘控制误差**：开环回放同一演示（物体复位到原配置）全任务成功率为零，成功必须靠闭环策略纠错；主要误差源是底盘速度控制（1m 半径 180° 转弯平均误差 >10cm）。此外最难的 Cook Shrimp 仅 40%，作者归因于 75 秒长程任务只采了 20 条演示、以及白碗白桌低对比度下的翻虾/倒虾困难。

## 五、评价与展望（学术视角）

**优点**：
- **系统性与可复现性极强**。把"硬件低成本化 + 全身遥操作交互 + 少样本 co-training 学习"打通成一个完整闭环，且软硬件全开源、成本压到工业单臂量级，是低成本双臂移动操作数据采集事实上的开源基线。腰部拴接反驱动底盘的交互方案巧妙地解决了"双手已被 leader 臂占用、还要同时控底盘"的矛盾，且无需 FPV/力反馈外设。
- **co-training 结论朴素而有力**。不做任何特殊的域对齐（甚至相机架法、背景、朝向、任务全不同），仅靠零填充底盘动作 + 等比例采样，就能让 out-of-domain 静态桌面数据显著提升移动任务，且对混合比不敏感——这大大降低了向新任务扩展时的工程与调参成本。"co-train 优于 pre-train→fine-tune"的对照也很有说服力，与灾难性遗忘的直觉一致。

**局限与开放问题**：
- **动作建模朴素**。直接把底盘速度拼进动作向量，未显式建模臂-底盘运动学耦合，导致底盘开环误差大、必须靠闭环纠错；引入 whole-body 逆运动学或以末端位姿为中心的动作空间可能进一步提升精度。
- **静态数据的可迁移边界未厘清**。为何完全不重叠、形态各异的桌面数据仍能正向迁移，论文只给出"运动先验/手腕相机不变性"的假设而缺乏系统分析；何时会负迁移（如 VINN 上的 -5%）尚不清楚。
- **规模有限**。7 个任务、每任务 20–50 条，属小数据 low-data regime；co-training 也被作者用来缓解此时 transformer 的过拟合，说明结论在更大规模/更多任务下是否保持有待验证。
- **与后续工作的关系**。本文的动作分块骨干直接沿用 ACT（Zhao et al. [104]，同组），与 Diffusion Policy（Chi et al. [18]）正交可组合；其"用异构机器人数据 co-training 提升单一平台"的思路，与 Open X-Embodiment/RT-X（[20]）的跨本体大规模联合训练一脉相承，但本文聚焦"少量 in-domain + 大量静态"的实用配方而非泛化到新本体。可能的改进方向包括：引入更大规模异构预训练/VLA 骨干、从次优与人类视频等异质数据学习、以及给策略加入自我改进（RL/交互式纠错）以突破纯模仿的上限。

## 参考

1. Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ALOHA/ACT，本文硬件与动作分块骨干，[104]）。
2. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023（本文 co-train 的 base 方法之一，[18]）。
3. Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, 2023（静态 ALOHA 数据来源、跨本体联合训练思路，[20]）。
4. Pari et al. *The Surprising Effectiveness of Representation Learning for Visual Imitation*（VINN，检索式模仿基线，[63]）。
5. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022（大规模真实数据模仿学习的代表，[12]）。
