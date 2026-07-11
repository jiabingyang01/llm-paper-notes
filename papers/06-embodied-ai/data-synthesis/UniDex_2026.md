# UniDex：从第一视角人手视频学习通用灵巧手控制的机器人基础套件

> **论文**：*UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos*
>
> **作者**：Gu Zhang, Qicheng Xu, Haozhe Zhang, Jianhan Ma, Long He, Yiming Bao, Zhecheng Yuan, Chengbo Yuan, Mingyu Ding, Yang Gao, Hang Zhao, Huazhe Xu, et al.
>
> **机构**：清华大学；上海期智研究院（Shanghai Qizhi Institute）；中山大学；北卡罗来纳大学教堂山分校（UNC Chapel Hill）
>
> **发布时间**：2026 年 03 月（arXiv 2603.22264）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.22264) | [PDF](https://arxiv.org/pdf/2603.22264)
>
> **分类标签**：`人手视频转数据` `灵巧手操作` `3D VLA` `统一动作空间` `具身预训练`

---

## 一句话总结

UniDex 用「指尖 IK 重定向 + 人手掩码后贴入机器人手网格的点云替换」这一 human-in-the-loop 流水线，把四个第一视角人手操作视频数据集转成机器人中心的灵巧手预训练数据 **UniDex-Dataset**（9M 帧、52K 轨迹、8 种手、6–24 DoF），再配合统一的 **Function-Actuator-Aligned Space（FAAS）** 动作空间训练 3D flow-matching VLA，在 5 个真机工具使用任务上平均任务进度达 **81%**（对比 $\pi_0$ 的 38%），并能零样本跨手迁移。

## 一、问题与动机

- **数据瓶颈**：真机遥操作采集贵、难扩展；且现有机器人基础策略几乎都面向平行夹爪，而日常工具使用（用剪刀、喷壶、扫把）离不开多指灵巧手，用夹爪根本做不了。
- **灵巧手三大难点**：(i) 灵巧手数据比夹爪更难采、大规模可用预训练集稀缺；(ii) 不同灵巧手在 DoF、形态、运动学、外观上差异极大，数据与策略跨本体迁移差；(iii) 灵巧控制本质高维，需要表达力强的动作空间与有效学习算法。
- **人手视频是天然富矿**：第一视角人手视频比机器人遥操作数据更便宜、更多样、更易得，且机器人灵巧手本就是模仿人手设计、共享操作模式。核心挑战是弥合人手与机器人手之间的 **kinematic gap（运动学）** 和 **visual gap（视觉）**。
- 现有从人手视频学习的路线：一类只用人手轨迹做规划/控制或 sim-to-real 重定向；一类在 egocentric 视频上预训练但需要预测人手运动，再靠专门的、复杂且脆弱的对齐后训练阶段去接机器人。UniDex 主张**在预训练阶段就直接生成机器人中心的灵巧手监督**，从而去掉微调时的专门对齐技巧，同时保持跨手控制。

## 二、核心方法

整套系统由三部分组成：数据集 UniDex-Dataset、策略 UniDex-VLA（含统一动作空间 FAAS）、便携采集与协同训练方案 UniDex-Cap。

### 2.1 Human-Robot Transformation：把人手视频转成机器人可执行轨迹

数据来自四个 RGB-D 第一视角人手操作数据集：**H2O、HOI4D、HOT3D、TACO**。转换需跨越运动学与视觉两个 gap。

**（a）Kinematic Retargeting（运动学重定向）。** 以指尖为主要接触点，将人手指尖轨迹对齐到机器人手，同时允许一个全局手基座调整以保证接触物理合理。给定人手姿态，提取 $m$ 个指尖目标：

$$X^\star = [x_1^\star, \dots, x_m^\star] \in \mathbb{R}^{3\times m}$$

引入一个 6-DoF 对齐偏置 $T_{\text{offset}}$（插在真实机器人基座前的一个 dummy base 的刚体变换）。指尖 $i$ 的前向运动学为：

$$x_i(q; T_{\text{offset}}) = \mathrm{Trans}\!\left(T^{\text{dummy}}_{\text{world}}\, T_{\text{offset}}\, T_i(q)\right) \in \mathbb{R}^3$$

其中 $T_i(q)$ 是机器人基座到指尖 $i$ 的齐次变换，$T^{\text{dummy}}_{\text{world}} = T_{\text{hand}}$ 固定不变。堆叠指尖残差得到 IK 误差：

$$e(q, T_{\text{offset}}) = \big[\, x_1(q;T_{\text{offset}}) - x_1^\star;\ \dots;\ x_m(q;T_{\text{offset}}) - x_m^\star \,\big] \in \mathbb{R}^{3m}$$

对含 mimic（联动）关节的手（Inspire/Oymotion/Agility），主关节解完后按运动学规范迭代更新从关节 $q_{j_s} = k\, q_{j_m} + c$，反复 $N$ 次直到收敛。

> **用大白话说**：机器人手指头长度、关节耦合都跟人不一样，硬套人手关节角会飞。UniDex 的做法是「只对齐指尖落点」——让机器人手的指尖去够人手指尖去过的位置，用 IK 反解关节角；同时额外给整只手加一个可平移旋转的「垫片」$T_{\text{offset}}$，好让手掌整体挪一挪，保证抓握时接触点是物理上说得通的。

流程是**两阶段 human-in-the-loop**：① 自动阶段用 PyBullet 的多末端执行器 IK 求解器在关节限位/阻尼约束下最小化指尖误差；② 交互阶段用一个 GUI 把 $T_{\text{offset}}$ 的 6 个自由度（3 平移 + 3 旋转）暴露成**滑动条**，人只需拖几下滑条微调，就能在多样姿态下得到鲁棒的指尖对齐与更合理的手-物接触。实际中每个人手数据集+每种手只需做一次基础标定 + 对少量接触帧微调即可覆盖绝大多数轨迹，因而能低人工成本扩到大规模。

**（b）Visual Alignment（视觉对齐）。** 从 RGB-D 算点云；用 **WiLoR + SAM2** 分割并抹掉人手对应的点；把重定向后的机器人手网格放进场景、渲染其几何进点云；最后用针孔相机模型把融合点云重投影回 RGB-D，避免因深度排序错误产生的遮挡穿帮，从而匹配真机微调时的单视角设置。

### 2.2 FAAS：功能-执行器对齐的统一动作空间

对任意有 $n$ 个驱动 DoF 的灵巧手，把每个 actuator（含 mimic 关节等任何可控 DoF）按其**功能角色**映射到 FAAS 索引，而不是按 URDF 特定关节顺序。功能原语包括：拇指-食指捏合、手指绕把手弯曲、侧向内收/外展稳定等。

FAAS 是一个 **82 维** 动作向量：前 18 维编码腕部姿态（每只手 9 维，双手共 18），每 9 维 = 6 维连续旋转表示（局部 $x$、$y$ 轴两个 3 维向量）+ 3 维平移；其余 64 维编码关节命令，每只手 32 个槽位，其中保留 **21 个 base actuator 槽位**跨所有手共享，剩余槽位留给手特有 DoF（如 Shadow 的额外腕关节）和未来的手。索引 $\langle 0,1,3,5,6 \rangle$ 在四种手间对齐。

> **用大白话说**：与其记「Shadow 手第 7 个关节」这种本体私有编号，FAAS 按「这根关节干什么活」给它编号——只要都是「拇指捏」，就放进同一个共享槽位。这样一条「捏合」技能在不同手之间就能对号入座地迁移，是后处理无关（post-processing-free）的，比 EgoVLA 那种在后训练阶段还要做 IK 的表示更不易累积误差。

### 2.3 UniDex-VLA：3D flow-matching 策略

- **观测** $o_t = [P_t, \ell_t, q_t]$：单视角带色点云 $P_t$、语言指令 $\ell_t$、本体感知 $q_t$。建模 $p(A_t \mid o_t)$，$A_t = [a_t, \dots, a_{t+H-1}]$ 为 $H$ 步动作块。$q_t$ 与 $a_t$ 均用 FAAS 表示；腕部在 $q_t$ 用绝对位姿，在动作输出用相对首帧的相对腕位姿（follow UMI）。
- **架构** 基本沿用 $\pi_0$，改点云输入：把 PaliGemma 里的 SigLIP 2D 编码器换成 **Uni3D**（vanilla ViT、由 2D 预训练 ViT 初始化的强 3D 点云编码器），Gemma backbone + Flow Matching 头，条件 flow-matching 目标训练，推理时前向 Euler 积分生成去噪动作块。
- **训练** 在 UniDex-Dataset 上预训练，再用每个任务 50 条真机演示微调。

### 2.4 UniDex-Cap：便携人手采集 + 人-机协同训练

用 Apple Vision Pro 采手/头姿态 + Intel RealSense L515 采高质量 RGB-D，二者用 3D 打印支架刚性固连并标定到同一坐标系；再走 2.1 的同一条转换流水线，加视角变换对齐人/机视角并降采样匹配遥操作速度，把人手数据转成机器人可执行轨迹，与真机数据 **co-train**，以少量真机数据 + 大量人手数据降低采集成本。

## 三、实验结果

**硬件**：7-DoF Franka + 三种末端（Inspire 6 active/12 full DoF、Wuji 20 active DoF、Oymotion 6 active/11 full DoF），Intel RealSense L515 第一视角 RGB-D。**5 个真机工具使用任务**：Make Coffee、Sweep Objects、Water Flowers、Cut Bags、Use Mouse；每任务 50 条演示微调，每方法/任务 20 次试验。指标为 **average task progress（平均任务进度，按任务阶段计）**。

### 3.1 主结果（5 任务聚合，图 11）

| 方法 | 平均任务进度 (%) | 最终成功率 (%) |
| --- | --- | --- |
| Diffusion Policy (DP) | 29.0 ± 19.9 | 22.0 ± 22.5 |
| DP3 | 35.0 ± 17.1 | 30.0 ± 18.7 |
| $\pi_0$（夹爪数据预训练） | 38.0 ± 7.4 | 35.0 ± 10.0 |
| UniDex-VLA（No Pretrain） | 32.5 ± 18.5 | 23.0 ± 12.0 |
| **UniDex-VLA** | **81.0 ± 12.1** | **76.0 ± 17.8** |

分任务平均任务进度（图 11，UniDex-VLA vs. 最好基线）：

| 任务 | DP | DP3 | $\pi_0$ | No Pretrain | **UniDex-VLA** |
| --- | --- | --- | --- | --- | --- |
| Make Coffee | 12.5 | 32.5 | 35.0 | 60.0 | **87.5** |
| Sweep Objects | 37.5 | 55.0 | 50.0 | 40.0 | **82.5** |
| Water Flowers | 35.0 | 50.0 | 32.5 | 12.5 | **85.0** |
| Cut Bags | 20.0 | 17.5 | 32.5 | — | **60.0** |
| Use Mouse | 60.0 | 40.0 | 20.0 | 20.0 | **90.0** |

要点：仅 50 条演示，UniDex-VLA 在这些长时序工具使用任务上大幅领先；预训练带来的增益在最难的 Cut Bags 上尤其明显——相对最好竞争方法平均任务进度提升 **84.6%**，说明 UniDex-Dataset 预训练赋予了强灵巧运动先验。

### 3.2 泛化

- **空间泛化（Make Coffee，水壶/滴滤杯放到训练分布外位置，图 8）**：UniDex-VLA 约 75%，显著高于各基线；配合 DemoGen 式几何编辑增广后可覆盖近满工作空间。
- **物体泛化（图 9，把黑水壶换成更小的紫水壶，颜色/尺寸/把手&壶嘴都变）**：UniDex-VLA 80% vs. No Pretrain 30% / DP3 15% / $\pi_0$ 10% / DP 0%。
- **跨手零样本迁移（图 10）**：在 Inspire 手上训好的 Make Coffee 策略直接零样本部署到 Wuji（20 DoF）和 Oymotion（6 DoF，不同运动学）：

| 目标手 | $\pi_0$ | UniDex-VLA (No Pretrain) | **UniDex-VLA** |
| --- | --- | --- | --- |
| Wuji | 0 | 0 | **40** |
| Oymotion | 10 | 5 | **60** |

FAAS + 预训练确实实现了零样本跨手技能迁移，而基线几乎全为 0。

### 3.3 数据集对比与人-机协同训练

UniDex-Dataset 覆盖 8 个灵巧手平台（Inspire、Leap、Shadow、Allegro、Ability、Oymotion、Xhand、Wuji），active DoF 6–24。与已发布灵巧数据集对比（表 1）：

| 数据集 | 轨迹数 | 手种类 | Language | 多场景 | Pointcloud |
| --- | --- | --- | --- | --- | --- |
| **UniDex-Dataset** | **52K** | **8** | ✓ | ✓ | ✓ |
| ActionNet | 30K | 2 | ✓ | ✗ | ✗（点云极低质） |
| RoboMind | 19K | 1 | ✓ | ✗ | ✗ |
| RealDex | 2K | 2 | ✓ | ✗ | ✓ |

**UniDex-Cap 协同训练（Make Coffee，图 13）** 结论：(i) 重定向人手数据有帮助但**真机数据不可或缺**——完全没有真机数据时成功率几近为零；(ii) **人-机替换率 ≈ 2:1**（约两条人手演示可替代一条真机演示，图中 $r{=}50$ 高性能区边界斜率 $\approx 2$）；(iii) Make Coffee 上人手演示采集比真机快约 **5.2×**，结合 2:1 替换率能显著降低采集成本。

## 四、局限性

- 作者自述：**尚未利用大规模 action-free（或弱标注）的第一视角活动数据集**——目前四个来源都是带手-物姿态标注的 RGB-D 灵巧操作集，把真正海量、无动作标签的日常视频纳入进来是进一步 scale 灵巧预训练的方向。
- 真机评测仅 5 个任务、2 种手做主评（第三种 Oymotion 主要用于跨手迁移），每任务仅 50 条演示、20 次试验，统计方差较大（多项 ±15–22%）。
- Kinematic retargeting 只对齐指尖 + 手基座 6-DoF，未显式建模手掌/近端指节与物体的接触，对包裹式/掌内操作的接触保真度存疑；且仍需 human-in-the-loop 拖滑条，虽轻量但非全自动。
- 视觉对齐把机器人手网格渲染贴回场景，依赖 WiLoR/SAM2 分割与深度重投影质量，分割失败或深度噪声会污染点云监督。
- 缺少与其他人手视频预训练 VLA（如 EgoVLA、Being-H0、EgoMimic 等）在同一基准上的直接对照，主要与 $\pi_0$/DP/DP3 及自身消融比较。

## 五、评价与展望

**优点**：(1) 抓住了灵巧 VLA 最实在的痛点——夹爪数据无法覆盖工具使用，而灵巧真机数据太贵；用人手视频造机器人中心监督是正解方向。(2) **FAAS 的「按功能对齐执行器」思路简洁而有效**，把跨本体迁移从「后训练做 IK」（EgoVLA 路线，易累积高 DoF 误差）前移为「预训练阶段的表示对齐」，换来了 40–60% 的零样本跨手成功率，是本文最有借鉴价值的设计。(3) 3D 点云 + 单视角设置贴合真机部署，且天然支持点云几何编辑做空间/物体增广。(4) UniDex-Cap 给出了「2:1 替换率、5.2× 采集加速」这类可量化的成本结论，对判断人手数据到底能省多少真机数据很有参考价值。

**缺点与开放问题**：(1) 「机器人中心」监督的代价是必须先把每个人手数据集重定向到每种目标手，是 O(数据集 × 手) 的工程量，虽有滑条 GUI 但仍是 human-in-the-loop，能否做到全自动、免标定是 scale 的关键；(2) 只对齐指尖的重定向对接触丰富的掌内操作可能不够，接触监督的物理保真度需要更强验证（如力/触觉）；(3) 与 action-free 大规模视频的结合（作者也承认是主要局限）——如何在没有精确手-物姿态时仍产出可用的机器人监督，是把该范式推向真正「互联网规模」的核心难题；(4) 主结果里 UniDex-VLA(No Pretrain) 常不敌 DP3/$\pi_0$，说明该 3D flow-matching 架构本身的增益有限，81% 主要来自 UniDex-Dataset 预训练——这反而印证了数据侧贡献是主角。

**与公开工作的关系**：数据侧沿 EgoMimic/Being-H0/EgoVLA 等「从第一视角人手视频学习」谱系，但强调预训练即产出机器人中心监督、省去脆弱的对齐后训练；动作空间对标 RDT-1B（保留语义结构）、$\pi_0$（左对齐表示）、EgoVLA（人手参数表示），FAAS 的功能中心统一空间是其区别点；策略骨架是 $\pi_0$ 的 3D 化（Uni3D 换 SigLIP）。可能的改进：引入触觉/接触监督、用自动化重定向替代滑条、把 action-free 视频与 latent action 结合、以及在更多手与更长时序任务上系统评测跨手迁移的边界。

## 参考

1. Black et al. *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.（策略骨架与主要基线）
2. Luo et al. *Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos*. arXiv:2507.15597, 2025.（人手视频预训练对照路线）
3. *EgoVLA*（Liu et al., arXiv:2410.08792, 2024，文中 [60]）：以人手参数为灵巧表示、后训练做 IK 的路线，FAAS 的直接对照对象。
4. Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*. arXiv:2410.07864, 2024.（统一动作空间参考）
5. Kareer et al. *EgoMimic: Scaling Imitation Learning via Egocentric Video*. ICRA 2025.（第一视角视频扩规模模仿学习）
