# RoboVerse：面向可扩展、可泛化机器人学习的统一平台、数据集与基准

> **论文**：*RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning*
>
> **作者**：Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An et al.（多位共同一作；Jitendra Malik、Pieter Abbeel 共同指导，通讯 Haoran Geng）
>
> **机构**：UC Berkeley、Peking University、USC、University of Michigan、UIUC、Stanford、CMU、UCLA、BIGAI
>
> **发布时间**：2025 年 04 月（arXiv 2504.18904）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2504.18904) | [PDF](https://arxiv.org/pdf/2504.18904)
>
> **分类标签**：`统一仿真平台` `合成数据生成` `机器人学习基准`

---

## 一句话总结

RoboVerse 以核心抽象层 **MetaSim** 把 Isaac Lab / Isaac Gym / MuJoCo / SAPIEN / Genesis / Bullet / CoppeliaSim 等异构仿真器统一成"配置系统 + 仿真器无关接口 + Gym 封装"三层架构，从而实现跨仿真器迁移、混合仿真与跨本体迁移；在此之上迁移并合成了一个含 **276 个任务类别、510.5k 条高保真轨迹、5.5k 资产、5000 万+ 状态转移** 的合成数据集，并配套一个四级泛化评测协议，实验证明其数据能提升模仿学习、强化学习、世界模型学习并支撑 sim-to-real 迁移。

## 一、问题与动机

机器人学习相对 NLP / CV 的两大瓶颈在于 **数据难以规模化** 与 **缺乏标准化评测**：

- **真实数据昂贵**：采集示范耗时耗力、依赖硬件、跨场景适应性差；光照、物体摆放、背景在真机上难以复现，导致规模化与可复现评测都不现实。
- **仿真生态碎片化**：仿真器种类繁多、内部架构与外部接口差异巨大，工作流迁移困难；已有合成数据集散落在各自的仿真器里，格式不一，复用与横向比较代价高。现有合成数据往往质量/多样性不足，或过拟合到某个特定仿真器，泛化到真实世界时失效。
- **基准不可复现**：不同基准之间因仿真精度、渲染风格、资产属性差异，结果难以互相复现。

作者主张:仿真提供高效计算、合成数据与全知信息,是规模化数据与一致性评测的可行路径,但前提是要有一个 **统一、可扩展** 的底座把碎片化生态粘合起来。RoboVerse 就是这样一个"平台 + 数据集 + 基准"三位一体的框架。

## 二、核心方法

### 2.1 MetaSim:仿真器之上的统一抽象层

MetaSim 是整个 RoboVerse 的基础设施,一个凌驾于具体仿真实现之上的高层接口,采用 **三层架构**:

1. **通用配置系统 `MetaConfig`**:一个嵌套 dataclass,以仿真器无关的方式抽象一个仿真场景的核心组件——agents(谁执行动作)、objects(环境长什么样)、tasks(做什么,含 instructions / success_metrics / reward_funcs)、sensors(如何被感知)、physics(gravity / collision / friction 等物理参数)。同时允许挂载仿真器专属超参(如 solver type)以保留各仿真器特性。
2. **对齐的仿真器后端(`Handler`)**:不同仿真器各自实现同一个 `Handler` 接口,统一 `launch()`、`get_states()`、`set_states()`、`step()`、`close()` 等覆盖仿真全生命周期的通用操作;上层只跟接口打交道,不关心底层是哪家仿真器。
3. **用户友好的环境封装(Gym `Env`)**:把 `Handler` 包成标准 `gym.Env`(`reset()` / `step()` / `render()` / `close()`),每个方法内部调用对应的 `Handler` 方法。

> 用大白话说:MetaSim 就像给七种"方言各异"的仿真器装了一个通用翻译器 + 统一遥控器——你用同一套配置和 API 描述任务,它翻译成各家仿真器的原生命令,于是同一份任务/资产/轨迹可以在任意后端上跑。

### 2.2 三大能力

基于上述抽象,MetaSim 提供:

- **Cross-Simulator Integration(跨仿真器集成)**:任务与轨迹可在不同仿真器间无缝切换。例如把 Meta-World 任务放到 Isaac Gym 里做快速并行 RL 训练,再把生成的轨迹放到 Isaac Sim 里做高保真渲染。
- **Hybrid Simulation(混合仿真)**:一条命令即可"物理引擎用一家、渲染器用另一家",取长补短。例如 **MuJoCo 的精确物理 + Isaac Sim 的强力渲染器** 组合出更强的数据生成能力。
- **Cross-Embodiment Transfer(跨本体迁移)**:通过重定向末端执行器位姿,把轨迹在不同的夹爪型机器人本体间复用,从而把异源数据整合进统一格式。

### 2.3 数据引擎:迁移 + 遥操作 + AI 生成 + Real-to-Sim

数据分三类——任务、资产、轨迹,主来源是 **从已有仿真环境迁移**:

- **直接迁移**:为任务初始化与评测定义环境配置,转换轨迹与资产格式;先在原仿真器内对齐格式,再自动保证跨仿真器兼容。
- **Motion Planning + RL Rollout**:若基准只给了关键点轨迹或抓取位姿(部分操作数据),用运动规划补全完整轨迹;若没有显式操作数据但有现成/可训策略,则用策略 rollout 采集,并用适配后的 success checker 严格过滤。
- **遥操作系统**:支持 keyboard / joystick / 手机 App(自研 Android/iOS)/ Mocap / VR 多种设备,可控制机械臂、灵巧手、双臂系统,跨仿真器统一接入。
- **AI 辅助任务生成**:用大生成模型学习示范中的空间/语义约束(如物体间避免重叠地铺开),把来自不同基准的物体组装进物理合理的场景,自动输出统一格式配置;并做两步过滤(Format Validation 剔除不合规任务、Feasibility Check 剔除不合理任务)以抑制幻觉。
- **Real-to-Sim 资产重建**:手机多视图 → COLMAP 结构初始化 + Gaussian Splatting 渲染 → VLM 推断物理属性 → surfel splatting + TSDF 重建网格 → 从视频学习物体运动学并细化 URDF(坐标系、朝向、轴对齐、尺度、相对 6-DoF 位姿、PD 参数),把真实视频桥接为可仿真资产。

已迁移的操作基准包括 ManiSkill、RLBench、CALVIN、Meta-World、RoboSuite、MimicGen、GAPartNet、Open6DOR、ARNOLD、LIBERO、SIMPLER、GraspNet、GarmentLab、UniDoorManip、GAPartManip;并覆盖导航(VLN-CE R2R/RxR)、运动与全身控制(HumanoidBench、Humanoid-X、SkillBlender)等更广本体(灵巧手、四足、人形)。

### 2.4 数据增强

- **轨迹增强(Trajectory Augmentation)**:沿用 MimicGen 框架。把一个任务分解成一串 **物体中心子任务** $(S_1(o_{S_1}), S_2(o_{S_2}), \ldots, S_M(o_{S_M}))$,其中第 $i$ 段子任务里机器人的轨迹 $S_i(o_{S_i})$ 定义在单个物体坐标系 $o_{S_i}\in\mathcal{O}$ 下($\mathcal{O}$ 为任务 $\mathcal{M}$ 的物体集合)。因此每条源示范可按预定义子任务顺序切成连续片段 $\{\tau_i\}_{i=1}^{M}$,再针对初始/目标状态分布(物体位姿 $D$、机器人 $R$ 的变化)在仿真中批量生成大量任务变体轨迹。

  > 用大白话说:既然"把杯子放到盘子上"这段动作本质是相对盘子坐标系的,那就把盘子挪到任意新位置,把这段动作按新坐标系刚性搬过去、在仿真里验证成功即可——用几十条人类示范就能滚出成千上万条新轨迹。文中实验用 50 条源示范扩增到 200/1000/3000 条。

- **域随机化(Domain Randomization,在 Isaac Sim handler 内实现)**:四类——① Table/Ground/Wall(桌面约 300 种材质,墙/地约 150 种,取自 ARNOLD + vMaterials);② Lighting(远光 + 圆柱灯阵 $n\times m$,强度与色温随机);③ Camera Poses(59 个候选相机位姿,多数正对机器人);④ Reflection(roughness / specular / metallic 随机)。四类可自由组合模拟室内场景。

### 2.5 四级泛化评测协议

统一按 90% 训练 / 10% 评测切分,逐级加难:

| 级别 | 泛化维度 | 变化内容 |
|---|---|---|
| Level 0 | 任务空间泛化 | 环境固定,仅物体初始化/指令在同一设定内 90/10 切分 |
| Level 1 | 环境随机化 | 相机/材质/光照固定,随机化 house/table/ground 配置 |
| Level 2 | 相机随机化 | 引入标注过的真实相机高度与角度 |
| Level 3 | 光照与反射随机化 | 随机化真实感材质与光照 |

模仿学习基准用专家模型 ACT、Diffusion Policy 与通用模型 OpenVLA、Octo;RL 基准集成 Stable-Baselines3 与 rsl_rl 的 PPO,并扩展支持 TD-MPC2。

## 三、实验结果

评测设定:$256\times256\times3$ RGB 输入,专家模型 9 维关节动作空间从零训练,通用模型微调到 delta 末端位姿空间、夹爪离散为 $\{0,+1\}$;随机抽 10 训练场景 + 10 验证场景,3 seed 平均。cuRobo 做 IK。

### 3.1 模仿学习基线(Table II,各源基准代表任务成功率 %)

| 模型(参数) | PickCube (ManiSkill) | StackCube (ManiSkill) | CloseBox (RLBench) | MoveSliderLeft (CALVIN) | PickChocolatePudding (LIBERO) | NutAssembly (RoboSuite) | 平均 |
|---|---|---|---|---|---|---|---|
| Diffusion Policy (78M) | 52.7 | 53.8 | 51.5 | 76.5 | 50.0 | 7.1 | 48.6 |
| ACT (84M) | 31.7 | 36.7 | 68.3 | 85.0 | 78.3 | 0.0 | 50.0 |

结论:两类主流专家策略在迁移进 RoboVerse 的多源任务上都能训练收敛、数值合理,验证数据集与基准可靠;NutAssembly 这类需精细接触/装配的任务对两者都很难(7.1 / 0.0)。

### 3.2 泛化性能(Table III,Level 0→3)

| 任务 | 模型 | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| MoveSliderLeft | DP | 76.5 | 81.3 | 72.0 | 60.0 |
| MoveSliderLeft | ACT | 85.0 | 83.3 | 43.3 | 16.6 |
| MoveSliderLeft | OpenVLA | 45.0 | 40.0 | 35.0 | 30.0 |
| CloseBox | DP | 51.5 | 42.8 | 20.0 | 10.4 |
| CloseBox | ACT | 68.3 | 73.3 | 0.0 | 20.0 |
| CloseBox | OpenVLA | 0.0 | 0.0 | 0.0 | 0.0 |
| PickCube | DP | 52.7 | 11.1 | 0.0 | 0.0 |
| PickCube | ACT | 31.7 | 30.0 | 6.7 | 3.3 |
| PickCube | OpenVLA | 40.0 | 15.0 | 0.0 | 0.0 |

结论:随视觉泛化难度上升,几乎所有模型都显著掉点(尤其相机/光照随机化的 L2、L3),说明该协议能有效区分策略的鲁棒性;不同任务上没有一个模型全面占优。

### 3.3 VLA 模型(Table IV,%)

| 方法 | PickCube | MoveSliderLeft | Grasp Obj Set1 | Set2 | Set3 |
|---|---|---|---|---|---|
| OpenVLA | 40.0 | 45.0 | 46.0 | 33.3 | 14.4 |
| Octo | 50.0 | 30.0 | 42.0 | 14.4 | 2.2 |

GraspNet 58 个物体按几何难度分三组,越难成功率越低,体现语言条件抓取的难度梯度。

### 3.4 轨迹增强(Fig 10)

在 PickCube/StackCube(ManiSkill)、CloseBox(RLBench)、MoveSliderLeft(CALVIN)上,Diffusion Policy 用 Source-50 vs Generated-200 / 1000 / 3000 训练,成功率随生成数据量单调上升,验证增强 API 的有效性与可扩展性。

### 3.5 世界模型学习(Fig 11)

动作条件世界模型仅用 DROID 50k episodes 训练时,基本能遵守动作条件但难以捕捉夹爪-物体接触的物理交互,接触时物体会"扭曲(warped)";额外加入 RoboVerse 50k 合成 episodes 组成 100k 后,物体几何保持明显改善。单独用 RoboVerse-50K 或 DROID-RoboVerse-100K 训练则生成帧在多数场景中物理上更真实——作者归因于 RoboVerse 丰富的随机化与增强,并指出 DROID 单场景样本覆盖有限、夹爪可见性不全导致迁移困难。

### 3.6 Sim-to-Real 与 RL sim-to-sim-to-real

- **直接 Sim-to-Real(Table V)**:用 GraspNet 适配示范微调后直接迁移真机(满分 10,夹爪接触目标记 0.5,成功抓取记 1)。

| 物体任务 | Octo | OpenVLA |
|---|---|---|
| Pick up Wash Soap | 5.0/10.0 | 7.0/10.0 |
| Lift Mouth Rinse | 3.0/10.0 | 8.0/10.0 |
| Grasp Green Dish | 6.0/10.0 | 5.0/10.0 |

  在 RoboVerse 上微调 OpenVLA 后可零额外微调迁移到真实、操作未见物体。
- **RL sim-to-sim-to-real**:PPO 在 Isaac Lab 上训人形全身控制,跨 Isaac Sim / MuJoCo 收敛一致;下肢 RL 策略 + 上身 PD 控制以 sim-to-sim-to-real 方式实现了在真机上的野外泛化行走。

## 四、局限性

1. **非刚体尚未完全支持**:统一格式对布料/柔性物等非刚体物体的整合尚不完备,留作未来工作(尽管已迁移 GarmentLab)。
2. **未做基础模型预训练**:大规模数据集有预训练操作基础模型的潜力,但因资源限制本文未做此探索。
3. **基线可能次优**:尽管尽力在 RoboVerse 内重实现并优化所有基线,部分实现仍可能未达最优——作者明确说明主要目标是展示系统的全面性与仿真-真实一致性,而非直接横评策略性能。
4. **数字对比有限**:VLA/世界模型/sim-to-real 实验受时间与算力约束,评测场景与轨迹数偏少(如 OpenVLA 统一采样 20 个测试场景),结论更偏"可行性验证"而非充分统计。

## 五、评价与展望

**优点**:
- **抽象层设计是真正的贡献**。把"配置系统 + Handler 接口 + Gym 封装"三层拆开,让任务/资产/轨迹与具体仿真器解耦,是解决社区碎片化的正确工程范式;Hybrid Simulation(物理与渲染分家)尤其巧妙,直接回应了"MuJoCo 物理准但渲染弱、Isaac 渲染强"的长期矛盾。
- **数据规模与来源多样**。510.5k 轨迹 / 276 任务 / 5.5k 资产 / 5000 万+ 状态转移,且来自迁移、遥操作、AI 生成、Real-to-Sim 四条并行管线,广度在同类工作中领先。
- **评测协议有区分度**。四级泛化协议把"任务空间 / 环境 / 相机 / 光照反射"逐层剥离,实验中确实产生了清晰的掉点梯度,比单一成功率更能刻画鲁棒性。

**不足与开放问题**:
- **物理保真度未被量化**。混合仿真号称"物理准 + 渲染真",但论文没有给出跨仿真器物理一致性或与真机的定量对齐指标,更多靠 sim-to-real 的少量成功案例佐证;不同后端间接触动力学差异对轨迹迁移的影响未充分讨论。
- **跨本体迁移局限于夹爪型**。末端位姿重定向对灵巧手/多指本体不成立,而灵巧操作恰是最需要数据的方向。
- **迁移数据的分布偏置**。大量轨迹来自 GraspNet(200k)、RLBench(150k)、RLAfford(40k)、SIMPLER(30k)等少数源,任务类别虽多但轨迹分布高度不均,直接混训可能被大源主导。
- **与相关工作的关系**:相比 Open X-Embodiment(真机数据聚合)与 ManiSkill/RLBench/LIBERO(单一仿真器基准),RoboVerse 的差异化在"仿真器无关的统一底座 + 混合仿真";相比 MimicGen/DexMimicGen(纯轨迹增强)与 RoboGen/GenSim(纯生成式任务合成),它是把这些能力都收编为管线中的一环;相比 Genesis(统一物理引擎),它走的是"不重造引擎、而是统一现有引擎"的互补路线。

**可能的改进方向**:引入跨仿真器物理一致性的定量基准;把非刚体/流体纳入统一格式;真正用该数据集预训练一个操作基础模型以验证数据价值上限;对多源轨迹做分布重加权或课程混合;把跨本体迁移从夹爪扩展到灵巧手。

## 参考

1. Mandlekar et al. *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations.* CoRL 2023.（轨迹增强的物体中心分解基础)
2. Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models.* 2023.（真机跨本体数据聚合的对照)
3. Gu et al. *ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills.* ICLR 2023.（被迁移的重要源基准与 SAPIEN 后端)
4. James et al. *RLBench: The Robot Learning Benchmark & Learning Environment.* RA-L 2020.（贡献 150k 轨迹的核心迁移源)
5. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* 2024.（模仿学习/sim-to-real 的通用模型基线)
