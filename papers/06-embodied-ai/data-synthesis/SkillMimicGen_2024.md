# SkillMimicGen：面向高效技能学习与部署的自动化演示生成

> **论文**：*SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning and Deployment*
>
> **作者**：Caelan Garrett\*, Ajay Mandlekar\*, Bowen Wen, Dieter Fox（\* 共同一作）
>
> **机构**：NVIDIA
>
> **发布时间**：2024 年 10 月（arXiv 2410.18907）
>
> **发表状态**：8th Conference on Robot Learning (CoRL 2024), Munich, Germany
>
> 🔗 [arXiv](https://arxiv.org/abs/2410.18907) | [PDF](https://arxiv.org/pdf/2410.18907)
>
> **分类标签**：`技能级演示生成` `模仿学习` `运动规划` `数据合成` `混合控制策略`

---

## 一句话总结

SkillMimicGen（SkillGen）把一条人类演示切成"接触密集的技能段"和"自由空间运动段",数据生成时只把技能段自适应迁移到新场景、用运动规划把它们缝合起来,从 60 条人类演示自动扩增出 24K+ 条演示;配套的 Hybrid Skill Policy（HSP）学习技能的起始/控制/终止三件套,使得测试时可用运动规划把闭环技能串起来执行——仿真数据生成成功率平均 75.4%（vs MimicGen 40.7%),训练出的智能体平均比 MimicGen 高 24%（85.7% vs 59.1%）。

## 一、问题与动机

模仿学习依赖大规模人类遥操作数据,采集昂贵;长程任务需串接多个操作行为更是困难。已有自动数据扩增系统 MimicGen 的做法是:把源人类演示切成以物体为中心的子任务段,针对新场景把这些段逐一做刚体变换后**用线性插值直接拼接**回放。作者指出这种朴素拼接有两个根本缺陷:

- **线性插值不感知场景几何**,拼接段容易与环境/杂物碰撞,导致数据生成失败,也不安全,难以真机部署;
- **插值段与人类段运动风格异质**,策略学习困难。MimicGen 因此陷入一个两难:短插值段利于策略学习但生成成功率低、真机不安全;长插值段生成吞吐高但策略更难学。

关键观察是:**操作任务的控制难度在时间上分布不均**。机器人必须先在自由空间移动到能与世界接触的状态,才谈得上接触操作;自由空间运动对规划器是易事,却给策略学习带来沉重负担。据此本文主张把任务拆成 motion(自由空间)与 skill(接触密集)两类段,数据生成与策略学习都只聚焦技能段。

## 二、核心方法

### 2.1 Skill 抽象（options 形式化）

借用强化学习的 options 框架,一个 skill 定义为四元组

$$\psi = \langle O, \mathcal{I}, \pi, \mathcal{T} \rangle$$

其中 $O$ 是被操作物体, $\mathcal{I}$ 是起始条件(允许启动策略 $\pi$ 的状态集合), $\pi$ 是策略, $\mathcal{T}$ 是终止条件(终止状态集合)。这一抽象贯穿三个阶段:遥操作采集、数据自适应扩增、部署执行。

**用大白话说**:把"抓咖啡包"这类局部动作打包成一个带"什么时候能开始、怎么做、什么时候算做完"的技能积木,任务就是若干积木按顺序拼起来,积木之间的搬运交给运动规划。

前提假设:动作空间为末端执行器(EE)的连续位姿指令 + 离散夹爪指令(A1);任务含一组可操作物体 $\{O_1,\dots,O_k\}$(A2);采集时在机器人接触某物体**之前**可观测/估计其位姿(A3)。任务被建模为 POMDP,策略经行为克隆(BC)学习:

$$\arg\max_\theta \; \mathbb{E}_{(s,o,a)\sim\mathcal{D}}\big[\log \pi_\theta(a \mid o)\big]$$

### 2.2 源演示与切分

源数据集 $\mathcal{D}_{\text{src}}$ 很小(实验每任务仅 10 条)。每条轨迹被标注为交替的运动段与技能段 $\tau = (\tau_{1m},\tau_{1s},\dots,\tau_{Nm},\tau_{Ns})$。本文采用 HITL-TAMP 采集:人只演示各任务的**局部技能段**,其余由 TAMP 规划器完成,切分点因此可自动获得。技能段内每个 EE 位姿动作以技能物体 $O_i$ 的坐标系存储:$T^{A_i}_{O_i} \leftarrow (T^{O_i}_W)^{-1} T^{A_i}_W$,其中 $T^{O_i}_W$ 是该技能开始前观测到的物体位姿。段内首个 EE 位姿即 initiation state,末位姿隐式定义 termination state(后者通过二分类学习)。

### 2.3 数据生成:自适应 + 缝合 + 拒绝采样

给定一个新初始状态,SkillGen 逐技能地把源技能段迁移到新场景:

1. 采样一个参考技能段 $\tau_{is}$(实验中首段随机采样源演示索引 $j\in\{1,\dots,N\}$,其余段沿用同一 $j$);
2. 用新场景中物体 $O_i$ 的位姿 $T^{O'_i}_W$ 与起始状态,算出新技能段应从哪个 EE 位姿开始:$T^{E'_0}_W \leftarrow T^{O'_i}_W\, T^{E_0}_{O_i}$;
3. 把整段动作序列以物体坐标系为锚做刚体变换:$T^{A'_i}_W = T^{O'_i}_{O_i}\, T^{A_i}_W$,该变换**保持了动作相对物体坐标系不变**;
4. 用运动规划(先 IK 转关节配置,再 RRT-Connect 规划无碰路径)把上一技能终止位形连到本技能起始位形,再用 EE 控制器执行技能段;
5. 全部技能执行完后检查任务是否成功,**只保留成功的演示**(trial-and-error 拒绝采样)。

生成时对动作施加加性高斯噪声(将绝对位姿动作转成归一化 delta 动作后加 $\mathcal{N}(0,1)$、幅度 $\sigma=0.05$),夹爪动作原样拷贝不加噪。

**用大白话说**:人类段只教"手相对物体该怎么动",换个场景只要知道物体新位姿,就能把这套动作平移旋转到新位置;技能之间的空中搬运不再靠瞎连直线,而是让规划器绕开障碍走一条安全路径;生成失败的直接丢掉,只留下能完成任务的干净数据。

### 2.4 Initiation Augmentation(起始增广)

测试时闭环技能既要预测起始目标(供规划器),又要执行技能。规划跟踪误差会让策略从分布外状态启动。为此可选地对起始状态加噪 $T^{E'_0}_W$(均匀平移 $\mathcal{U}[-t,t]$、随机轴 + 随机角 $\phi\sim\mathcal{U}[0,r]$ 的旋转,实验取 $t=0.08$ m、$r=80°$),并在技能段前额外规划一段 **recovery segment** 把加噪起点拉回原起点,从而拓宽起始集支撑。代价是这种激进随机化下大量目标位姿不可达/碰撞,生成成功率大幅下降(见实验)。

### 2.5 Hybrid Skill Policy（HSP）

在生成数据上学习参数化技能 $\psi_\theta = \langle O, \mathcal{I}_\theta, \pi_\theta, \mathcal{T}_\theta \rangle$:

- $\mathcal{I}_\theta:\mathcal{O}\to\mathrm{SE}(3)$ 从上一技能的最后观测预测本技能起始位姿;
- $\pi_\theta:\mathcal{O}\to\mathcal{A}$ 用 BC(BC-RNN)学习的闭环控制策略;
- $\mathcal{T}_\theta:\mathcal{O}\to\{0,1\}$ 从最近观测判断技能是否到达终止状态。

部署时(见论文 Fig. 2):对序列中每个技能,先用 $\mathcal{I}_\theta$ 预测起始位姿并用运动规划器执行到达,再滚动 $\pi_\theta$ 直到 $\mathcal{T}_\theta$ 判定终止,然后进入下一技能。**闭环反应式技能与由规划器承担的粗略搬运交替进行**。

三个变体,假设强度递减:

| 变体 | 起始条件学习方式 | 额外假设 |
|---|---|---|
| **HSP-TAMP** | 不学起始/终止,仅把 $\pi_\theta$ 塞进 HITL-TAMP | 假设最多:需 TAMP 系统告知何时启停 |
| **HSP-Class** | 在源演示上做分类(预测生成该 demo 的源演示)再做位姿自适应 | 需已知相关物体序列且可观测物体位姿 |
| **HSP-Reg** | 直接回归 $\mathrm{SE}(3)$ 起始位姿 | 假设最少,与标准 BC 相同(无需物体位姿观测) |

终止分类器与起始网络共享 ResNet-18 + spatial-softmax 观测编码器;起始回归网络用 6D 旋转表示 + GMM 头。为防提前终止,部署时需连续 5 次预测终止才接受。

## 三、实验结果

任务:6 个仿真任务 × 3 种复位分布 = 18 个变体(Square / Threading / Piece Assembly 精细插入; Coffee / Nut Assembly / Coffee Prep 长程串接),4 个真机任务(Pick-Place-Milk / Cleanup-Butter-Trash / Coffee / Nut-Assembly),以及 1 个 sim-to-real 任务。每任务 10 条 HITL-TAMP 源演示,SkillGen 生成 1000 条成功演示。运动规划用 TRAC-IK + RRT-Connect + OSC;真机用 FoundationPose 估计位姿、点云碰撞检测。

### 3.1 数据生成成功率(仿真,Table F.1)

SkillGen 平均 75.4% vs MimicGen 40.7%,复位分布越大优势越明显:

| 变体 | MimicGen | SkillGen |
|---|---|---|
| Square $D_2$ | 31.8 | **87.7** |
| Threading $D_2$ | 21.6 | **74.3** |
| Piece Assembly $D_2$ | 31.3 | **69.3** |
| Coffee $D_2$ | 27.7 | **70.0** |
| Nut Assembly $D_0$ | 53.0 | **98.6** |
| Coffee Prep $D_2$ | **0.0** | **59.9** |

Coffee Prep $D_2$ 中抽屉与马克杯位于桌面两端、与源演示差异极大,MimicGen 完全无法生成数据,SkillGen 仍达 59.9%。**杂物场景(Clutter)** 中放置大障碍物时,SkillGen 达 49.0–72.5%,MimicGen 仅 4.0–16.5%(如 Square $D_1$-Clutter:62.5 vs 4.0)。注意起始增广会显著拉低生成率(+IA 平均仅 14.7%),但生成率与最终策略成功率并无严格正相关。

### 3.2 策略成功率(仿真,Fig. 4)

跨全部任务平均:

| 方法 | 平均成功率 |
|---|---|
| Source(源演示直接训) | — |
| MimicGen (BC-RNN) | 59.1 |
| HSP-Reg | 72.6 |
| HSP-Class | 82.9 |
| HSP-TAMP | 85.7 |

在源任务 $D_0$ 上,用 SkillGen 数据相比直接用源演示训练有巨大提升(Three Piece Assembly 28%→96%, Nut Assembly 22%→100%)。混合控制在难变体上尤为有效:Nut Assembly $D_1/D_2$ 上 HSP 比 MimicGen 高出多达 62%;Coffee Prep $D_2$ 上 HSP-Reg 达 74–84%,而 MimicGen 无法生成该变体的任何数据。HSP-Reg 假设最少却只比 HSP-Class/TAMP 平均低 10–13%。

### 3.3 数据效率与规模缩放

- **等量对比**:200 条 SkillGen 演示与 200 条人类演示训练出的 HSP-TAMP 智能体性能相当(最大偏差 10%),但 SkillGen 只用了 10 条源演示、每任务采集 < 4 分钟(人类采 200 条需 37–71 分钟)。
- **规模缩放**(HSP-Reg):Square $D_2$ 从 1000→5000 条,52%→72%;Threading $D_1$ 60%→76%;200→1000 条时各任务普遍显著提升。

### 3.4 真机与 sim-to-real(Fig. 4 右下)

真机每任务仅 3 条源演示、生成 100 条、训练 HSP-Class:

| 真机任务 | MimicGen (BC-RNN) | SkillGen (HSP) |
|---|---|---|
| Milk-Bin | — | **95.0** |
| Butter-Trash | — | **95.0** |
| Coffee | 14.0 | **65.0** |
| Nut-Assembly [Sim] | 72.0 | **92.0** |
| Square-Assembly(sim-to-real 首插入) | 5.0 | **35.0** |
| Nut-Assembly(sim-to-real 全任务) | 0.0 | **35.0** |

Coffee 任务 SkillGen 达 65%(HITL-TAMP 原文用 100 条人类演示报 74%,而本文只用 3 条且需学抓咖啡包,更难)。零样本 sim-to-real 中,MimicGen 智能体只能以 5% 解决首个插入、从不完成全任务,HSP-Class 则能以 35% 完成整条长程装配——验证了"把任务拆成更易迁移的局部行为序列"对 sim-to-real 的价值。

## 四、局限性

作者在正文与附录 C 列出:

1. **数据生成需预先给定固定的技能序列**(每个技能段要操作哪个物体必须已知);
2. 生成时**假设每个技能段开始处可观测/估计物体位姿**(A3);
3. 仅在**准静态、刚体**任务上验证;
4. **依赖 HITL-TAMP 源数据**:用常规遥操作采集的源演示效果更差(附录 L),如何用更一致的人类标注缩小该差距留待未来;
5. sim-to-real 智能体**观测受限**——仅观测本体感知变化,不用位姿跟踪或视觉观测,动作空间限于仅平移(无旋转),以规避感知与控制器的 sim-real gap,牺牲了通用性。
6. 迁移的技能段可能产生规划器/控制器难以到达的位姿,开环回放的小误差会在插入等高精度动作中累积,造成生成失败。

## 五、评价与展望

**优点**。(1) 诊断精准:把"控制难度时间上不均匀"这一观察转化为 skill / motion 的干净分解,直击 MimicGen 线性插值的碰撞与异质性两大痛点,思路简洁且可解释。(2) 生成与学习协同:同一 options 抽象同时服务数据生成(拒绝采样保证干净)与部署(HSP 让闭环技能与规划搬运交替),而非仅做数据扩增。(3) 数据效率极高:10 条源演示扩增到 1000 条、200 条合成数据 ≈ 200 条人类数据但采集成本降一个数量级,且提供了三档假设强度的 HSP 变体供不同可观测性场景选用,工程实用性强。(4) 真机与 sim-to-real 均有正向验证,长程装配 0%→35% 的对比说服力较强。

**缺点与开放问题**。(1) **强结构先验**:必须预先知道技能序列与每步操作物体、且接触前可观测物体位姿,这在真正开放、非结构化场景(可变形物、动态物、序列未知)下难以满足,通用性弱于纯端到端方法。(2) **对 HITL-TAMP 的依赖**是隐性成本——采集"局部技能段"本身需要一套 TAMP 基础设施与自动切分,常规遥操作源数据效果明显下降,削弱了"仅需少量人类演示"的普适卖点。(3) **技能段仍为开环回放 + 加噪**,高精度插入处误差累积,起始增广虽拓宽支撑却把生成率压到 ~15%,靠并行算力硬扛,采样效率有改进空间(更智能的可达/无碰采样)。(4) sim-to-real 的成功部分建立在"仅本体感知、仅平移动作"的强约束上,与真正的视觉闭环 sim-to-real 仍有距离。

**与相关工作的关系**。本文是 MimicGen(同组前作,以线性插值缝合)的直接演进,核心改进即"插值 → 运动规划 + 技能抽象";与 HITL-TAMP 的关系是把其 TAMP-gated 的启停条件由工程写死改为**学习** $\mathcal{I}_\theta/\mathcal{T}_\theta$;相对 SayCan 等语言组合技能的工作,这里的技能是局部闭环操作、搬运交给规划器。可能的改进方向:引入闭环技能段回放(而非开环 + 噪声)以进一步减小累积误差、用生成式/学习式采样替代激进随机化提升起始增广效率、放宽物体位姿可观测假设(如用类别级或无位姿的接触表征),以及把技能序列本身也做成可学习/可搜索而非人工给定。

## 参考

1. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations*, CoRL 2023 —— 本文直接改进的前作(线性插值缝合)。
2. Mandlekar et al., *Human-in-the-Loop Task and Motion Planning for Imitation Learning* (HITL-TAMP), CoRL 2023 —— 源演示采集与技能段自动切分所依赖的系统。
3. Stolle & Precup, *Learning Options in Reinforcement Learning*, SARA 2002 —— skill 四元组所借用的 options 形式化。
4. Wen et al., *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*, CVPR 2024 —— 真机实验的物体位姿估计。
5. Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*, CoRL 2021 —— BC-RNN 策略架构与评测协议来源。
