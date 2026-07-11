# InternData-A1：面向通用策略预训练的高保真合成数据引擎

> **论文**：*InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy*
>
> **作者**：Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi（共同一作）, Ning Gao, …, Jia Zeng, Hao Dong, Jiangmiao Pang（通讯）et al.
>
> **机构**：Shanghai AI Laboratory（上海人工智能实验室）、Peking University（北京大学）
>
> **发布时间**：2025 年 11 月（arXiv 2511.16651）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.16651) | [PDF](https://arxiv.org/pdf/2511.16651)
>
> **分类标签**：`合成数据预训练` `VLA` `sim-to-real` `跨本体数据引擎`

---

## 一句话总结

用一条全自动、模块解耦、可组合的高保真仿真流水线合成了 **630k 条轨迹 / 7433.9 小时 / 401.4M 帧**、覆盖 4 本体、18 技能、70 任务、227 场景（含刚体/铰接/可形变/流体）的操作数据集 InternData-A1,并首次证明:**仅用合成数据预训练的 $\pi_0$ 可以匹敌甚至反超在闭源 $\pi$-dataset 上训练的官方 $\pi_0$**(49 个仿真任务 Easy 60.0% vs 55.0%、Hard 26.5% vs 20.0%),其中 10 个任务零样本 sim-to-real 成功率超过 50%。

## 一、问题与动机

VLA 模型的泛化能力主要由预训练数据的规模、多样性与来源决定。目前最强的证据来自 **大规模真机数据**($\pi$-series、AgiBot World 等),但真机采集依赖熟练操作员、专用硬件和大量人力,大多数研究组无法负担,导致社区缺乏系统研究"VLA 预训练到底需要什么样的数据"的能力。

仿真是天然的互补路径:资产库丰富、场景可控、可自动化生成。但已有的仿真数据集(RoboCasa、RoboTwin 2.0、MimicGen、ManiSkill2 等)存在三个通病:

1. **技能面窄**——绝大多数只做 pick-and-place;
2. **物体单一**——几乎只涉及刚体,缺乏铰接/可形变/流体;
3. **很少真正验证大规模 VLA 预训练效果**——多数需要大量人工介入,且没有和最强真机数据正面对比。

由此留下核心开放问题:**当合成数据在本体、场景、技能、物理真实性上都被充分放大后,能否匹配最强真机数据的预训练效果?** 本文用 InternData-A1 给出第一份肯定答案。

## 二、核心方法

### 2.1 数据集构成(与已有仿真数据集对比)

| 维度 | InternData-A1 | 对比(如 RoboTwin 2.0 / RoboCasa) |
|---|---|---|
| 轨迹数 | 630k | RoboTwin 2.0 100k / RoboCasa 77k |
| 时长 / 帧数 | 7433.9 h / 401.4M 帧 | — |
| 本体数 | 4(单臂+双臂) | 1~2 为主 |
| 技能 / 任务 / 场景 | 18 / 70 / 227 | 技能多为 pick-place,场景 1~120 |
| 物体类型 | 刚体+铰接+可形变+流体 | 基本只有刚体 |
| 失败恢复 / 开源 / 采集方式 | ✓ / ✓ / 全自动 | 多数无失败恢复 |

**4 类本体及占比**:Franka Emika Panda(单臂,桌面,23.3%)、AgileX Split Aloha(Piper-100 臂,30.8%)、ARX Lift-2(R5a 臂,37.8%)、AgiBot Genie-1(7.9%),双臂本体支撑丰富的双手协作与并行动作。

**70 个任务的技能分布**:4 个流体、4 个可形变、15 个铰接、47 个刚体任务;其中 **18 个长程任务**每个至少串联 3 个技能,合计 124,789 条轨迹、141,421,619 帧。作者刻意做了配比平衡,使 70 个任务里有 56 个落在 1,000~10,000 条轨迹区间,逼近近似均匀分布(而非像旧数据集那样"换个物体就算一个新任务")。

**资产**:3,185 个刚体(107 类,源自 OmniObject3D / Objaverse)、321 个铰接物体(14 类,融合 GRUtopia / GAPartNet / GenSim2 / Infinite Mobility / ArtVIP)、EinScan 扫描的 20 件真实衣物、来自 GRScenes-100 的 227 个室内场景(厨房/书房/餐厅/客厅)。

### 2.2 四阶段全解耦合成流水线

流水线把资产规格、技能策略、任务组合、渲染四件事彻底解耦:

**① 环境构建**:按任务模板从库中检索机器人、场景、物体。刚体带物理属性+按 AnyGrasp 自动生成抓取位姿;铰接体标注关节轴/部件位姿/阻尼刚度;可形变体用 Vertex Block Descent 仿真;流体用基于粒子的 PBD 动力学 + 等值面渲染。

**② 技能组合(核心)**:每个原子技能是一个模块化脚本策略,输入物体状态、机器人状态、用户约束,输出一串目标末端 6D 位姿(waypoint)。形式化即一个状态到路点的映射:

$$\Phi_{\text{skill}}:\ (s_{\text{obj}},\ s_{\text{robot}},\ c)\ \mapsto\ (W_1,\ W_2,\ \dots,\ W_k),\quad W_i \in SE(3)$$

用大白话说:每个技能就是个"脚本函数",告诉机器人手该依次去哪几个位姿,把"要干什么"的高层逻辑和"怎么动"的底层运动彻底分开;换物体、换空间范围、换场景甚至换本体都不用重写,唯一的人工只是调一下空间范围。

长程任务则是若干技能的顺序/并行组合:

$$\mathcal{T}\ =\ \Phi_{\text{skill}}^{(1)}\ \circ\ \Phi_{\text{skill}}^{(2)}\ \circ\ \cdots\ \circ\ \Phi_{\text{skill}}^{(m)},\quad m \ge 3$$

用大白话说:把 pick、handover、place 这些原子技能像搭积木一样串起来(比如"一只手拿盘子递过去、另一只手放上架子"),长程任务就自动展开成一条完整轨迹。

**③ 域随机化**:相机(主视角+腕部)在 $\Delta R \in [-5°,+5°]$、$\Delta t \in [-5\text{cm},+5\text{cm}]$ 内扰动;构建 174 张环境贴图库随机光温光强;同类物体可替换、桌面与背景随机化;抓取上,自动流水线先生成数百万候选,再按 AnyGrasp 从 top-40 高置信里随机选一个;铰接/可形变物体则把接触区扩成邻域后随机采点。

**④ 生成与存储**:用 **CuRobo** 运动规划器在路点之间插值稠密关节动作,并在物理仿真里验证——只有成功的轨迹才被渲染:

$$\{q_t\}_{t=1}^{T}\ =\ \text{CuRobo}(W_1,\dots,W_k)\quad \text{s.t.}\ \text{无碰撞、物理可行}$$

用大白话说:先规划、先在物理里跑一遍验证,失败的直接丢掉、绝不浪费 GPU 去渲染。最终数据统一转成 LeRobot 格式,含物体元数据、语言指令、多视角 RGB、本体状态与动作标签,可选存深度/grounding/框。

### 2.3 系统级框架优化

作者指出传统"规划+渲染合一"的流水线在大规模下有两大瓶颈:任务越复杂规划成功率越低(却仍白白渲染)、规划是 CPU 串行而渲染是 GPU 并行(异构负载串行执行利用率极差)。为此引入四项优化:Planner/Renderer 解耦并流水线化、动态资源调度(双端并行批处理)、Stack Render 堆叠渲染提吞吐、以及 Balancer + Supervisor 保障集群稳定。整体端到端提速 **2–3×**;成本低至 **每条 episode < 0.003 美元**,在 **8 张 RTX 4090** 上每天产出 **209.7 小时** 机器人数据。

## 三、实验结果

预训练完全沿用 $\pi_0$ 架构(Paligemma VLM + flow-matching 动作专家),从 Paligemma 权重起、仅用 InternData-A1 训练,再与官方 $\pi_0$(在闭源 $\pi$-dataset 上训)做同样的下游微调对比。

### 3.1 对比闭源 $\pi$-dataset(49 个 RoboTwin 2.0 双臂任务,每任务 100 试)

| 方法 | Easy 平均 | Hard 平均 |
|---|---|---|
| $\pi_0$ (Scratch, 仅 Paligemma) | 23.5% | 2.5% |
| $\pi_0$ (官方, 闭源 $\pi$-dataset) | 55.0% | 20.0% |
| $\pi_0$ (**InternData-A1**) | **60.0%** | **26.5%** |

即纯合成预训练相比官方 $\pi_0$ **Easy +5.0、Hard +6.5 个百分点**,相比 Paligemma 裸模型 **Easy +36.5、Hard +24.0**。代表性任务上(Easy/Hard,InternData-A1):Lift Pot 63.5/2.5、Pick Dual Bottles 62.0/19.0、Place Object Stand 48.5/29.5、Shake Bottle 98.0/64.0、Turn Switch 40.5/32.5、Hanging Mug 24.5/20.0,几乎逐项超过官方 $\pi_0$。值得注意 Hard 模式(干净示范训练、域随机化评测)的稳健性提升,说明合成数据的强域随机化带来的泛化即使在纯净微调后仍保留。

### 3.2 9 个真机任务(3 本体,每任务 30 试;单位 %)

| 任务(本体) | Scratch | 官方 $\pi_0$ | $\pi_0$ (InternData-A1) |
|---|---|---|---|
| Place Markpen (Genie-1) | 40 | 73 | **83** |
| Pass Bottle (Genie-1) | 30 | 53 | **63** |
| Heat Sandwich (ARX Lift-2) | 50 | 83 | **90** |
| Sort Rubbish (ARX Lift-2) | 0 | 93 | **97** |
| Sweep Trash (ARX Lift-2) | 17 | 50 | 50 |
| Sort Parts (ARX Lift-2) | 33 | 63 | **74** |
| Unscrew Cap (ARX AC One) | 0 | 73 | **83** |
| Fold Clothes (ARX AC One) | 57 | 57 | 50 |
| Zip Bag (ARX AC One) | 13 | 40 | **47** |

5 个常规任务上 InternData-A1 平均超 $\pi$-dataset **6.2%**;4 个长程灵巧任务(含二者都没见过的新本体 ARX AC One)也做到与 $\pi$-dataset 相当,显示大量长程任务带来的更广动作空间可迁移到新任务与新本体。

### 3.3 对比开源数据集(Table 3)

| 数据集 | 域 | 49 Sim Easy | 49 Sim Hard | Sort Rubbish(真机) | Pass Bottle(真机) |
|---|---|---|---|---|---|
| OXE | Real | 32.5 | 11.0 | 40.0 | 36.7 |
| Agibot World | Real | 52.5 | 12.0 | 53.3 | 56.7 |
| RoboCasa | Sim | 50.0 | 11.0 | 23.3 | 13.3 |
| **InternData-A1** | **Sim** | **60.0** | **26.5** | **90.0** | **60.0** |

RoboCasa 在仿真上仅落后 10%,但真机骤降;InternData-A1 真机平均较 RoboCasa 提升 **57.7%**,作者归因于高度逼真的渲染与充足数据量。

### 3.4 sim-to-real 定量结论

- **纯仿真 vs 纯真机**(4 任务,200~1600 仿真 / 200 真机 episode):基础技能任务(Sort Rubbish、Wipe Stain)**200 条仿真即可追平 200 条真机**;复杂任务(Flip Package、Instructional Pick)需约 **1600 条**仿真——即等效数据比收窄到 **8:1 以内,部分逼近 1:1**。据此可写成成本决策关系:

$$r\ =\ \frac{N_{\text{sim}}}{N_{\text{real}}}\ \le\ 8:1$$

用大白话说:一条真机数据"顶"多少条仿真?最坏 8 条、最好 1 条,而合成一条的时间与金钱成本远低于真机采集。强 sim-to-real 迁移**不要求**背景/光照/纹理精确复刻,只要相机视角与关节动作空间大致对齐即可。

- **500 仿真 episode 的额外 6 任务**(Figure 7):Make Sandwich 50、Close Box 63、Close Microwave 87、Pack 50、Sweep 60、Handover 57,均超 50%。加上前 4 个,**70 个任务里有 10 个实现无任何真机数据的高 sim-to-real 成功率**。

### 3.5 数据成分消融(Table 4,RoboTwin 2.0 Easy/Hard)

| 配置 | Easy / Hard |
|---|---|
| Full | **58.0 / 25.0** |
| w.o. PnP(pick-place, 占 30.61%) | 57.0 / 22.5 |
| w.o. Art(铰接, 11.67%) | 55.5 / 19.5 |
| w.o. Base(基础多技能, 35.95%) | 52.5 / 20.5 |
| w.o. Long(长程, 21.77%) | 54.0 / 19.0 |

两个关键观察:① 尽管 PnP 与 Base 体量最大,**去掉 Base 或 Long 比去掉 PnP 掉得更多**;② 铰接任务规模小、物体少,但去掉它比去掉 PnP 掉得还多(因其关节几何多样、手臂构型分布更广)。作者据此提出假设:**轨迹多样性(而非 pick-place 的绝对体量)才是有效预训练的核心驱动力**。

## 四、局限性

- **物理仿真限制灵巧度**:高精细灵巧任务(系鞋带、穿针)难以仿真,数据集的灵巧覆盖有上限;
- **对比条件受限**:$\pi$-dataset 闭源,只能"重训一个 $\pi_0$ vs 官方 checkpoint"间接比,严格控制变量不足;开源数据集对比因资源限制仅训 500k iters、只测 2 个真机任务;
- **"匹配最强真机数据"的结论边界**:主要在 RoboTwin 2.0 仿真基准与作者自设真机任务上成立,评测任务与合成任务的技能高度同源,存在评测-数据同源带来的乐观偏差;
- **轨迹多样性驱动**仅停留在消融观察与假设,缺乏严格的理论或受控实验证据;域随机化范围(相机 ±5°/±5cm)相对保守,更大 sim-to-real gap 场景下的表现未知。

## 五、评价与展望

**贡献定位**:这是继 GraspVLA(仅单一抓取技能、十亿级合成动作)之后,第一份系统论证"**多样、复杂、长程的合成数据可以整体替代最强真机数据做 VLA 预训练**"的工作。相比 RoboTwin 2.0 主打 benchmark、RoboCasa 主打日常任务库,InternData-A1 的差异化在于:(1)真正把刚体/铰接/可形变/流体四类物理都纳入同一自动流水线;(2)用"技能=状态→路点脚本 + CuRobo 插值 + 物理验证后再渲染"的解耦设计,把人工降到"只调空间范围",并给出 <0.003 美元/episode 的量化成本;(3)Planner/Renderer 流水线化的系统工程细节(异构负载调度、Stack Render、集群稳定)是同类数据论文里少见的、可直接复用的工程贡献。

**最有价值的洞见**是消融得出的"**trajectory diversity > pick-place volume**":这直接反驳了"合成数据只要堆 pick-place 体量就行"的朴素做法,对社区的数据配比策略有指导意义——应主动增配铰接与长程多技能任务。sim-to-real 8:1~1:1 的定量比值也为"真机 vs 合成"的采买/配比提供了可操作的成本锚点。

**存疑与可改进方向**:① 与 $\pi$-dataset 的对比本质是"复现 $\pi_0$ vs 官方 checkpoint",训练配方、数据清洗、超参都无法对齐,"匹敌闭源最强"的结论需谨慎;② 评测集(RoboTwin 2.0 + 自设真机任务)与合成技能同源,更能说明问题的应是完全 held-out 的第三方 benchmark;③ 论文正文与 Table 2 标题在"Easy/Hard 各提升几个点"上口径略有出入(正文 6%/5% vs 表格 5%/6.5%),以表格数字为准;④ 未来若能把"轨迹多样性"形式化为可度量的量(如技能转移图的覆盖/熵)并做受控 scaling law,将比现有假设更有说服力;⑤ 可形变/流体虽已纳入,但真机验证任务仍以刚体+铰接为主,这两类的 sim-to-real 证据偏弱。总体而言,这是一份"数据引擎工程扎实、结论有冲击力、但对照实验受闭源基线掣肘"的高质量合成数据工作。

## 参考

1. Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, RSS 2024 —— 主对照基线与所用架构。
2. Chen et al., *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization*, 2025 —— 主要仿真评测基准与对比数据集。
3. Nasiriany et al., *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*, CVPR 2024 —— 开源仿真数据集对照。
4. Sundaralingam et al., *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*, 2023 —— 流水线中的运动规划器。
5. Fang et al., *AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains*, TRO 2023 —— 自动抓取位姿生成。
