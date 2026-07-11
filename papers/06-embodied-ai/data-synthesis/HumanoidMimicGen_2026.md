# HumanoidMimicGen：面向 Loco-Manipulation 的全身规划数据生成

> **论文**：*HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning*
>
> **作者**：Kevin Lin, Ajay Mandlekar, Caelan Reed Garrett（共同一作）, Nikita Chernyadev, Yu Fang, Runyu Ding, Yuqi Xie, Justin Tran, Linxi Fan, Yuke Zhu（项目负责人）et al.
>
> **机构**：NVIDIA；The University of Texas at Austin
>
> **发布时间**：2026 年 05 月（arXiv 2605.27724）
>
> **发表状态**：未录用（预印本，NVIDIA 版权标注）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.27724) | [PDF](https://arxiv.org/pdf/2605.27724)
>
> **分类标签**：`数据生成` `人形 loco-manipulation` `whole-body planning` `sim-and-real co-training`

---

## 一句话总结

HumanoidMimicGen 把 MimicGen/DexMimicGen 这套"少量人类演示 → 物体中心 skill 复用 → 空间自适应扩增"的数据生成范式,首次搬到**腿式人形的 loco-manipulation**上:通过混合动作空间(上身关节控制 + 下身 RL 步态控制器)与"运动 / 操作解耦"的全身规划,把**单条**遥操作演示自动放大成 1000 条无碰撞轨迹,在自建的 9 任务 G1 基准上把策略平均成功率从 DexMimicGen+ 的 0.33 提到 0.89,并使真机 sim-and-real 联合训练相对纯真机数据的成功率从 0.51 提升到 0.71。

## 一、问题与动机

VLA 范式在**固定站位**的桌面操作上已被反复验证,但人形机器人的核心卖点是"能像人一样边走边操作"。把 VLA 扩到 loco-manipulation 面临两大瓶颈:

- **数据稀缺**:真机遥操作采集人形全身演示费时费力,现成的大规模 loco-manip 数据集几乎不存在。
- **控制困难**:人形是高维复合动作空间(手臂 + 腿 + 躯干),操作时还要动态维持平衡,难以像固定机械臂那样对每条肢体独立做任务空间控制。

现有自动数据生成算法(MimicGen [41]、DexMimicGen [30]、SkillGen [20])都假设动作空间是**任务空间**,用 OSC 之类把末端目标位姿映射为指令。这一假设在腿上不成立——腿必须由全身协调来静态站稳、动态行走。作者的目标:在不做奖励调参(避开纯 RL 路线)的前提下,让一套演示自适应算法能生成**稳定、无碰撞**的人形 loco-manip 数据。

## 二、核心方法

整体流程沿用 MimicGen 家族思路:把演示切成物体中心的 **skill**,在新场景里对每个 skill 做空间自适应回放;创新在于用**混合动作空间**和**全身规划**去替换原来的任务空间执行。

### 2.1 问题形式化与 skill 自适应

把每个 loco-manip 任务建模为 POMDP。机器人构型 $q \in \mathbf{R}^{|\mathcal{J}|}$,关节集 $\mathcal{J}$ 含腿、躯干、双臂、双手。每条演示 $d$ 被切成一组物体中心 skill,单个 skill 记为元组 $\psi = \langle e, f, d^\psi \rangle$,其中 $e$ 是末端执行器、$f$ 是参考物体坐标系、$d^\psi$ 是演示 $d$ 的一段连续子序列。

自适应的关键是把 skill 里相对物体坐标系记录的末端目标位姿,搬到新场景里当前物体位姿下:

$$a'[e] = s'[f]\, s_0^\psi[f]^{-1}\, a^\psi[e]$$

**用大白话说**:$s_0^\psi[f]^{-1} a^\psi[e]$ 先把演示时"末端相对物体"的位姿抠出来(消掉演示时物体在哪),再乘上新场景当前物体位姿 $s'[f]$,于是无论物体挪到哪,末端都按"相对物体不变"的方式复现同一个抓/放动作。这正是 MimicGen 的核心,HumanoidMimicGen 只是把它嵌进人形全身。

skill 之间用**先序对** $\mathcal{P}$(某 skill 必须先于另一个完成)与**并发对** $\mathcal{C}$(两 skill 必须同时开始)标注,合成一张有向无环图 DAG,再用贪心的**在线拓扑排序**把无先序冲突的 skill 分组 $\Psi_i$ 一起执行。以 Table-to-Shelf 为例:两个"抓箱"skill 并发为第一组,两个"放箱"skill 并发为第二组。

### 2.2 混合动作空间

不对腿做独立任务空间控制,而是采用**分层混合动作空间**:所有关节最底层都用关节位置控制;其上暴露高层 API,分两块——(i) 上身(臂、手、躯干)的关节位置指令;(ii) 下身的 base 运动指令。下身直接复用 Homie [4] 的 RL 步态控制器,base 指令为

$$a[l] = [\dot{x}, \dot{y}, \dot{\theta}, z]$$

其中 $\dot{x}, \dot{y}$ 是平面 base 速度、$\dot{\theta}$ 是偏航角速度、$z$ 是期望躯干高度。RL 控制器吃当前/目标臂躯干构型 + base 指令,吐出动态可行的腿部关节位置。**用大白话说**:平衡、接触、迈腿这些脏活全丢给学出来的步态控制器,数据生成只需在"往哪走、躯干多高、上身怎么摆"这个抽象层做规划。

### 2.3 全身数据生成(运动 / 操作解耦)

因为 RL 控制器只能**近似**跟踪速度指令、无法精确跟踪,作者把每个 skill 的执行**解耦**成动态行走段与静态操作段(Algorithm 1):

1. 对当前 skill 组算出各末端目标位姿 $T[e] = s[f]\, s_0^\psi[f]^{-1}\, a^\psi[e]$;
2. 解**全身逆运动学** WHOLE-INV-KINEMATICS,批量求出能同时够到所有末端目标的可达构型 $q''$;
3. 构造 **switch 构型** $q'$:上身关节取当前 $q$、下身关节取目标 $q''$——即"走到位、准备切换到操作"的姿态;
4. 规划 base 从当前构型到 $q'$ 的**行走轨迹** $\tau_l$,用 RL 控制器执行;由于执行不精确,用**实际到达**的 switch 构型替换 $q'$;
5. 规划 $q' \to q''$ 的**操作轨迹** $\tau_m$,用上身关节控制器执行;
6. 最后 ADAPT-SKILL-DEMOS(Algorithm 2)逐点做全身 IK 跟踪自适应后的末端目标 $a^\psi[e]$,完成 skill 回放;手指关节 $a^\psi[J_h]$ **原样回放**不改。执行完用 CHECK-SUCCESS 判成功,只保留成功轨迹,重复采样到凑够 $N$ 条。

**全身 IK 的两个工程点**(Appendix C):(a) 用 cuRobo 做 GPU 加速的批量多链 IK 与碰撞检测;机器人和物体用**变半径球集**近似(对每个 mesh 采点、求切于该点与另一面的最小体积球、半径膨胀 $\epsilon \approx 0.01\text{m}$、再贪心组合优化选 $N$ 个球最大覆盖),握持物体则惰性做球分解。因球近似会过估几何导致初始/目标位形常处于"接触=碰撞"的假不可行,故每次 IK/规划前把正碰撞的球**收缩**到不碰为止。(b) 全身 IK 最小化对当前构型的 $L_0$ 加权距离 $\lVert q'' - q \rVert_0$,按关节组迭代放开自由度,顺序为 $[J_a,\ J_a \cup J_t,\ J_a \cup J_l,\ \mathcal{J}]$(仅臂 → 臂+躯干 → 臂+腿 → 全身),尽量少动躯干、能不迈腿就不迈腿。

### 2.4 扰动增强

两类噪声提策略性能:(i) rollout 时注入**动作噪声** $a' = a + \epsilon,\ \epsilon \sim \mathcal{N}(0, \sigma^2 I)$,但存的标签仍是原始 $a$——让机器人见到偏离标称的状态而学到纠偏;(ii) 随机扰动机器人**初始 base 位姿**,增加初始态多样性。消融显示这两者都很关键。

## 三、实验结果

**基准**:自建 G1 loco-manipulation 基准(robosuite + MuJoCo),9 个任务,沿 base 运动量、操作复杂度(单臂/双臂/全身)、执行时长三个轴变化,均为二值成功判据。

**策略**:微调 GR00T N1.6 VLA 基座 [6],单目 ego 相机 RGB(224×224)+ 本体状态,训 25K 步、每 5k 步在 100 episode 上评、取最优 checkpoint。对每个任务从**同一条**源演示出发,各方法各生成 1000 条。

**主结果(Table 1,策略成功率)**:

| 方法 | Box Lift Floor | Push Button | Box Lift | Push Shelf Fwd | Drill Lift | Drill PnP | Box→Shelf | Pick Drill | Drill Lift Obs | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 条人类演示 | 0.14 | 0.18 | 0.95 | 0.70 | 0.20 | 0.08 | 0.04 | 0.00 | 0.04 | 0.26 |
| 100 条人类演示 | 0.83 | 0.82 | 0.95 | 0.90 | 0.30 | 0.20 | 0.28 | 0.07 | 0.00 | 0.48 |
| DexMimicGen+ | 0.87 | 0.26 | 0.68 | 0.35 | 0.13 | 0.13 | 0.17 | 0.36 | 0.00 | 0.33 |
| **HumanoidMimicGen** | **0.97** | **0.92** | **1.00** | **1.00** | **1.00** | **0.70** | **0.53** | **1.00** | **0.87** | **0.89** |

要点:从**单条**源演示扩增,HumanoidMimicGen 平均 0.89,全面碾压加了 locomotion 但缺规划/碰撞检测的 DexMimicGen+(0.33)、甚至碾压 100 条真人演示(0.48)。长时任务优势尤其明显——Push Shelf Forward(1230 步)1.00 vs DexMimicGen+ 0.35,Box Lift Floor(900 步)0.97 vs 0.87。DexMimicGen+ 因为直线插值 base、无碰撞检测,在需要绕障导航的 Drill Lift Obstacle 上直接归零。

**策略架构消融(Table 4)**:同样 1000 条 HumanoidMimicGen 数据下,VLA 0.89 > Flow Matching 0.86 > Diffusion Policy 0.51,VLA 最优、Flow Matching 接近。

**数据生成消融(Table 5)**:去掉动作噪声平均掉到 0.49,固定初始位姿掉到 0.51——两类扰动缺一不可(vs 完整 0.89)。

**跨本体(Table 3)**:同一套 pipeline 无算法改动地用到浮动基座(无腿)人形,平均 0.90 与腿式 0.89 基本持平,说明方法不绑死在腿式。

**真机(Figure 5,4 任务)**:每任务 1 条仿真演示自动扩增出 500 条,做 sim-and-real 联合训练。

| | ThrowBottle | BoxToCart | PickCanister | PickCanister w/ Obstruction | 平均 |
|---|---|---|---|---|---|
| 仅真机 | 0.60 | 0.35 | 0.50 | 0.60 | 0.51 |
| 联合训练 | **0.75** | **0.60** | **0.75** | **0.75** | **0.71** |

联合训练把平均分从 0.51 提到 0.71(绝对 +0.20,即摘要所称"20%"),四个任务全涨。真机侧:Luxonis OAK-D 相机,上身控制 25 Hz、下身策略推理 50 Hz、底层位置控制 200 Hz。

## 四、局限性

- **重人工标注**:需人工切 skill 段并标注先序/并发约束(与 DexMimicGen、SkillGen 同病);联合训练用的仿真环境、初始态分布、成功判据也都靠手工设计。
- **固定 skill 集与固定序列结构**:方法假设一组固定的物体中心 skill 和固定 skill 序列,无法泛化到需要**新 skill 序列或新高层规划**的任务。
- **依赖刚体坐标变换**:自适应基于物体坐标系刚性变换,不处理大类内几何差异或模糊接触 affordance(作者建议可用 CP-Gen [34] 这类方法补足)。
- **步态控制器上限**:下身完全托管给 Homie RL 控制器,数据质量与可达空间受其跟踪精度与稳定域约束,论文未系统分析控制器失败对生成失败的贡献。

## 五、评价与展望

**优点**。(1) 定位精准:把成熟的 MimicGen 空间自适应范式干净地嫁接到人形,核心贡献是"混合动作空间 + 运动/操作解耦全身规划",绕开了任务空间控制在腿上失效的死结,思路清晰、工程完整(cuRobo GPU IK、自动球分解碰撞近似都很务实)。(2) 数据效率极端:**单条**演示扩增即超越 100 条真人演示,且带真机闭环验证(0.51→0.71),比很多只在仿真里自我论证的数据生成工作更有说服力。(3) 顺带交付了一个 9 任务、按 loco/nav/单臂/双臂/竖直/接触/长时七维标注的可复现 G1 基准,对社区有价值。

**与公开工作的关系**。它是 MimicGen [41] → DexMimicGen [30](双臂灵巧手)→ MoMaGen [33](移动操作 + 软可见性约束)→ WBCMimicGen [35] 这条"自适应数据生成"主线在**腿式人形**上的自然延伸。相较同期人形数据工作,它与 OmniRetarget [54] 形成鲜明路线对比:后者走 RL + 交互保持重定向,本文刻意用 imitation learning 回避奖励调参;两者本质是"演示自适应"与"运动重定向"两种扩增哲学。DexMimicGen+ 作为消融基线的设计也诚实地隔离出了"skill 推理 + 运动规划 + 碰撞检测"三项增量的价值。

**开放问题与可能改进**。(1) skill 与约束标注的自动化是最大瓶颈,作者自己也点名可用 foundation model 来自动构环境与标约束——这是把方法从"专家可扩增"推向"规模化自增殖"的关键。(2) 固定 skill 序列使其难做需要重规划的任务,一个方向是把 skill DAG 生成交给高层 LLM/VLM planner,让先序与并发约束可在线合成。(3) 刚体变换假设在可变形物、铰接物、密集类内几何变化下会失效,结合 CP-Gen 式约束保持或点级表征(如 Point Bridge [23])值得探索。(4) 下身 RL 控制器目前是黑盒依赖,若能把控制器的可达/稳定裕度显式纳入 IK 可行性判据,或对生成失败做归因分析,数据产率与安全性都可能进一步提升。(5) 真机仅 4 任务、每任务 20 episode 评测,统计力度偏弱,长时全身任务的真机迁移仍待更大规模验证。

## 参考

1. Mandlekar et al. *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations.* CoRL 2023. —— 空间自适应扩增范式的源头。
2. Jiang et al. *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning.* arXiv:2410.24185, 2024. —— 双臂灵巧手扩展,本文主要基线 DexMimicGen+ 的原型。
3. Garrett et al. *SkillGen: Automated Demonstration Generation for Efficient Skill Learning and Deployment.* CoRL 2024. —— skill + 规划自适应,本文 skill 表示的直接来源。
4. Ben et al. *Homie: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit.* arXiv:2502.13013, 2025. —— 本文所用的下身 RL 步态控制器。
5. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734, 2025. —— 本文微调所用的 VLA 基座模型。
