# LUCID：从无结构人类视频中学习本体无关的意图模型以实现可扩展的灵巧机器人技能获取

> **论文**：*LUCID: Learning Embodiment-Agnostic Intent Models from Unstructured Human Videos for Scalable Dexterous Robot Skill Acquisition*
>
> **作者**：Harsh Gupta, Guanya Shi, Wenzhen Yuan（Harsh Gupta 为通讯作者，Guanya Shi 与 Wenzhen Yuan 为共同指导）
>
> **机构**：University of Illinois Urbana-Champaign；Carnegie Mellon University
>
> **发布时间**：2026 年 06 月（arXiv 2606.11628，2026-06-10）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.11628) | [PDF](https://arxiv.org/pdf/2606.11628)
>
> **分类标签**：`从人类视频学意图` `意图-控制解耦` `物体流+palm pose接口` `sim-to-real RL` `跨本体迁移`

---

## 一句话总结

LUCID 把机器人技能拆成两块并分开学：一个**本体无关的意图模型** $f_\theta$ 从互联网规模的无结构人类视频里学"接下来场景该怎么变"（短时物体流 + 掌心 palm pose），一个**本体相关的 sensorimotor 策略** $\pi$ 在大规模并行仿真里用 RL 学"这只手/夹爪怎么把它做出来"；两者通过一个共享的短时参考接口对接，同一个意图模型可零改动地驱动灵巧手与平行夹爪。在三个纯 web 视频监督的真实任务上闭环 LUCID 平均成功率 73%，而开环视频生成基线仅 28%。

## 一、问题与动机

机器人操作策略今天主流靠两类数据训练:遥操作机器人示范,或带专门采集装置(动捕、多视角、可穿戴、手持夹爪)的结构化人类示范。这两条路都需要专用采集基建、绑定特定本体、且只能随操作员工时线性扩展。

作者指出另外两类数据**天然可扩展**却各有短板:

1. **无结构人类视频**(互联网视频):物体、场景、策略极其多样,量大,但"没有动作标签"(actionless)——你看不到关节指令。
2. **物理仿真**:能以任意规模产出带动作标签的数据,但每个任务都要手工设计奖励,尤其是高层意图难以定义。

论文的核心论点是:这两类源**互补**,应通过"意图-控制分离"来配对——人类视频提供**本体无关的意图**,仿真提供**任务无关且鲁棒的 sensorimotor 策略**。

对已有工作的批评很具体:
- 直接在从人类视频抽取的轨迹上做模仿的策略,学到的是**轨迹级行为而非任务级意图**,泛化不出被演示的那个场景;
- 把预训练视频模型在初始场景上条件化的**开环规划器**,一旦执行偏离就无法恢复(计划只在 $t=0$ 生成一次,执行中不更新);
- 把视频模型当策略 backbone 复用的路线,仍然需要**每任务、每本体的真机数据**。

而在仿真侧,即便有在大规模域随机化下能跨物体/几何/场景条件泛化的通才 sensorimotor 策略,它们**推理时的参考仍来自外部**(动捕、单视频抽取、或在开局跑一次视频模型)。

LUCID 的两个设计选择正是针对这些痛点:
- **意图与控制解耦**:意图模型预测短时物体流 + palm pose 参考,sensorimotor 策略只负责把参考实现成关节指令;接口共享,故同一意图模型可迁移到不同本体。
- **闭环意图**:部署时 $f_\theta$ 持续从**当前实时场景**重新查询,而非一次性出计划,不需要物体网格、动捕或每本体适配。

## 二、核心方法

系统由两个学习组件组成,通过一个短时参考 $\mathcal{R}$(物体流 + palm pose)通信。

### 2.1 意图模型 $f_\theta$(从人类视频学)

**定义**:操作意图 = 对物体运动 + 粗略掌心位姿的短时预测,跨本体共享;把它落成关节指令的活交给独立的 sensorimotor 策略。

设 $\mathbf{I}_t$ 为截止 $t$ 时刻的 $F$ 帧 RGB-D 堆叠,$\tau=0,1,\dots,T$ 索引当前步与 $T$ 个未来步。对目标物体上的查询点 $n$,$\mathbf{x}^{\text{trk}}_{n,\tau}\in\mathbb{R}^3$ 是它在第 $\tau$ 步的 3D 位置,$(\mathbf{p}^{\text{palm}}_\tau, \mathbf{R}^{\text{palm}}_\tau)\in SE(3)$ 是第 $\tau$ 步的掌心位姿。意图模型从当前观测和当前步值预测未来步值,堆成参考 $\mathcal{R}$:

$$
f_\theta\!\left(\mathbf{I}_t,\ \{\mathbf{x}^{\text{trk}}_{n,0}\}_{n=1}^{N},\ (\mathbf{p}^{\text{palm}}_0, \mathbf{R}^{\text{palm}}_0)\right) = \mathcal{R} = \left(\{\mathbf{x}^{\text{trk}}_{n,\tau}\}_{n=1,\tau=1}^{N,T},\ \{(\mathbf{p}^{\text{palm}}_\tau, \mathbf{R}^{\text{palm}}_\tau)\}_{\tau=1}^{T}\right)
$$

用大白话说:给模型一段最近的画面 + 现在物体表面上那 $N$ 个点在哪 + 手现在大概在哪,它就吐出"未来一小段时间里这些点会怎么动、手该怎么走",而不吐关节角。物体流是 mesh-free 的,能覆盖刚体/铰接/可变形物体,且从互联网视频里现成可抽。palm pose 补上了物体流缺失的"手该在哪接触"这一信息(对多指操作很关键)。

**架构**:把 CoTracker3 改造成一个"点-token transformer"做前向短时预测,三处改动:(1) 用冻结的 DINOv3 patch token 做条件,深度经一个零初始化残差 adapter 融进 patch-token 空间;(2) 从 $\mathbf{I}_t$ **向前预测未来**(标准点跟踪器是对已观测帧估计轨迹);(3) 在 $N$ 个物体 token 旁挂一个**单独的 palm-pose token**,让物体流与 palm pose 联合产出。输入 $256\times256$,$F=2$,$N=16$,$T=8$,预测视界 1 秒;transformer 用 CoTracker3 的 EfficientUpdateFormer(depth 12,hidden 768,12 heads,空间/时间/场景交叉注意力分解)。

**训练**:对流 $\mathcal{L}_{\text{trk}}$、palm 位置 $\mathcal{L}_{\text{palm,p}}$、palm 旋转 $\mathcal{L}_{\text{palm,r}}$ 三路做 MSE,在 $T$ 个未来步上平均并相加 $\mathcal{L}(\theta)=\mathcal{L}_{\text{trk}}+\mathcal{L}_{\text{palm,p}}+\mathcal{L}_{\text{palm,r}}$。关键工程:**大力增强 $\mathbf{I}_t$ 里的人类像素**($p_{\text{human}}=0.6$,用四种外观模式替换人手区域),逼模型学"物体在动"而不依赖演示者的手长什么样——因为部署时画面里可见的是机器人的手而非人手。

**从无结构视频抽监督(数据引擎核心)**:每个任务从公开视频集(Panda-70M、Action100M、Something-Something-V2、EPIC-Kitchens、LVP metadata,主要是野外 YouTube)挖 **20k clip**;对代表性不足的任务改用挂载 iPhone 自采 ~100 段演示。**唯一标注是物体名**(用来 prompt 分割)。clip 重采样到 8Hz、按 stride 2 切滑窗。每窗过四阶段抽取流水线:

| 阶段 | 工具 | 产出 |
|---|---|---|
| a 相机+深度重建 | ViPE(单目 SLAM) | 逐帧内参 $\mathbf{K}_\tau$、外参 $\mathbf{E}_\tau$、稠密深度 $D_\tau$ |
| b 物体+手分割 | SAM 3.1(以物体名 prompt) | 物体掩膜 $M^{\text{obj}}_\tau$、人手掩膜 $M^{\text{hum}}_\tau$;从物体掩膜减去人手掩膜防泄漏 |
| c 物体流轨迹 | DenseTrack3Dv2 | 在物体掩膜内采 $N$ 个查询像素跟踪,经 $D_\tau$、$\mathbf{K}_\tau$ 反投影得 3D 轨迹 $\{\mathbf{x}^{\text{trk}}_{n,\tau}\}$ |
| d 手重建 | WiLoR + MANO 刚性拟合 | 逐帧刚性拟合把 MANO mesh 对齐到人手掩膜内深度,读出 palm pose |

palm-pose lifting(Eq. 1)解一个统一尺度 $s$ 与平移 $\mathbf{t}$,让 MANO 顶点对齐 ViPE 深度:

$$
\min_{s,\mathbf{t}}\ \sum_{i\in\mathcal{V}_\tau}\ \big\| s\,\mathbf{v}^{\mathcal{M}}_{\tau,i} + \mathbf{t} - \text{unproj}(\mathbf{u}_i, \bar{d}_\tau;\ \mathbf{K}_\tau)\big\|_2^2
$$

用大白话说:MANO 出的手是"手局部坐标、尺度任意"的,这一步用画面里手部像素的深度中值把它摆正、缩放到真实世界尺度,然后从掌心顶点读出 palm pose——每帧就是一次 lstsq。

### 2.2 通才 Sensorimotor 策略 $\pi$(仿真里学)

$\pi$ 把板载感知映射到电机指令,在 Isaac Lab 里用**目标条件 RL + 大规模并行仿真**训练,负责在具体本体上实现意图参考 $\mathcal{R}$;每个本体用**同一套配方**单独训一个。

**关键点:参考完全程序化生成,不来自 $f_\theta$ 也不来自人类视频**。每个 episode 加载一个程序化生成的物体(box/sphere/capsule/cylinder/plate 图元的布尔并,覆盖 blob/tool/plate 三类形状,~1k 形状池),随机化尺度($[0.65,1.1]$)、质量、摩擦。参考轨迹 $\mathcal{R}$ 是四段链:**approach(2.5s,full 耦合)→ in-hand motion(0–7.5s,0–2 个随机 waypoint)→ goal(3.0s)→ disengage(2.5s,none 耦合)**;每段的"手-物耦合模式"(full 刚性锁定 / position-only 释放旋转 / none 完全解耦)不同,使同一段物体运动允许多种有效手部策略。这样训出的是一个**任务无关**的策略,部署时跟随 $f_\theta$ 产出的任何 $\mathcal{R}$。

**动作空间**:$\mathbf{a}_t=[\mathbf{a}^{\text{arm}}_t;\mathbf{a}^{\text{eig}}_t;\mathbf{a}^{\text{hnd}}_t]$,即臂关节位置增量 $\mathbb{R}^6$、eigen-grasp 系数 $\mathbb{R}^5$、逐关节手残差 $\mathbb{R}^{16}$。eigen-grasp 基是对 retargeted 人类抓取做 top-5 PCA,让手指沿自然抓取的协调模式探索(偏向稳定抓取),残差再补基表达不了的逐关节运动。动作叠加在上一关节目标上,EMA 平滑、裁剪到关节限位(follow SimToolReal)。

**Teacher–Student 蒸馏**:teacher $\pi^T$ 先用 PPO 在特权信息上训(从**完整物体表面**采样的物体流 + palm-pose 参考 + 本体感知)。student $\pi^S$ 从 $\pi^T$ 蒸馏,把特权采样换成**外部相机可见子集 + 腕部深度图像**,用混合目标(PPO + 蒸馏):

$$
\mathcal{L}_{\text{student}} = \mathcal{L}_{\text{PPO}}(\pi^S) + \lambda_D\,\kappa\,\mathbb{E}\!\left[\|\mu^S(o^S)-\mu^T(o^T)\|_2^2\right]
$$

$\lambda_D$ 在前 1000 epoch 从 1.0 线性退火到 0.1——早期模仿主导,后期 on-policy PPO 接管以补齐 sim-to-real 差距。policy 架构基于 AME(源自足式运动的交叉注意力 map encoding),把查询点各自 token 化(编码器置换不变),student 额外把腕深度图 token 化并入。teacher PPO 用 **20,480 并行环境**,student 蒸馏用 2,048;curriculum 把重力从近零升到全、物体 wrench 扰动增大、成功容差收紧,绑到标量 $\rho\in[0,1]$(follow DextrAH-RGB);再用 CMA-ES 做系统辨识把仿真动力学对齐真机。

### 2.3 真实部署:慢意图环套快控制环

真机用一个固定外部 RGB-D 相机 + 一个腕挂深度相机。**慢意图环**(每周期镜像训练监督流水线):SAM 3.1 刷新物体掩膜,采查询点反投影到当前 3D 位置,以此 + $\mathbf{I}_t$ + 前向运动学的 palm pose 重新查询 $f_\theta$ 产出新的 $\mathcal{R}$。**快控制环**:每一步 $\pi^S$ 消费当前点、$\mathcal{R}$ 的流与 palm-pose 参考、腕深度、本体感知,输出 $\mathbf{a}_t$。两次意图周期之间由一个滑窗 3D 点跟踪器保持点与实时场景一致;lookahead 窗口沿 $\mathcal{R}$ 的流视界推进,到头就重查 $f_\theta$。部署速率(Table 11):SAM 3.1 掩膜 1Hz、$f_\theta$ 1Hz、DenseTrack3Dv2 跟踪 30Hz、$\pi^S$ 50Hz;整栈跑在**单张 RTX 5090**。真机为 UR5e 臂 + LEAP 手。

## 三、实验结果

论文围绕四个问题(Q1 端到端是否可行、Q2 数据 scaling 是否有效、Q3 跨本体/新任务迁移、Q4 策略设计选择)。

### 3.1 Q1:web 视频监督的真实任务(§4.1)

三个任务各由 **20k 人类视频 clip** 监督:stirring(捡勺子在容器里画三圈)、wiping(捡布擦掉表面随机痕迹)、binning(捡起工作区每个物体投进目标容器)。每任务在 3 个场景(共同变化物体实例、桌面布置、外部相机位姿,**均为意图模型的 OOD**)各 10 次试验。基线为**开环视频生成规划器**:Veo 3.1 从初始 RGB 生成一段视频,抽取物体流 + palm pose 参考,驱动**同一个** sensorimotor 策略。

| 任务 | 闭环 LUCID(灵巧手) | 开环规划器(Veo 3.1) |
|---|---|---|
| Stirring | 19/30 ≈ 63% | 7/30 ≈ 23% |
| Wiping | 26/30 ≈ 87% | 13/30 ≈ 43% |
| Binning | 21/30 ≈ 70% | 5/30 ≈ 17% |
| **平均** | **66/90 ≈ 73%** | **25/90 ≈ 28%** |

LUCID 的优势正在**闭环意图**:初次抓取落空或物体中途移位时,意图模型重查场景并重定向策略;开环基线一旦偏离生成计划就用陈旧参考,失败模式不同(如搅拌时误抓把勺子甩飞、Veo 3.1 还会幻觉出根本没擦的区域)。sensorimotor 策略虽只在刚体上仿真训练,却能处理可变形物体(纸巾、毛巾、布)。作者还观察到**涌现的任务组合**:擦完番茄酱后在工作区旁放个 bin,意图模型会捡起脏纸巾投进 bin——**没有任何 binning 专门监督**,归因于 20k 擦拭 clip 池里偶发的投放行为。

### 3.2 Q2:意图数据 scaling(§4.2)

在 binning 任务上把训练语料从 $\{1\text{k},2\text{k},5\text{k},10\text{k},20\text{k}\}$ clip 扫,每个规模点在 3 个 binning 场景各 10 次真机试验(共 30 次)+ 在 1k held-out clip 上算意图损失。结果:held-out 意图损失随语料稳步下降(观测区间内可用幂律 $L(M)=c+aM^{-\alpha}$ 拟合,从 ~3.15 降到 ~2.8),真机成功率随之上升(1k→20k 约从 ~0% 升到 ~73%)。定性:1k–2k 参考很差(策略够到物体但认不出 bin);5k–10k 容器定位涌现但放置对齐弱,释放时常错过 bin。注意:附录 C.3 明确观测区间内多条不同 $\alpha$ 的幂律几乎无法区分,外推到 $1000\times$ 时不确定性很大——**该扫描支持"更多视频有帮助"但钉不出精确的长程 scaling 预测**。

### 3.3 Q3:跨本体意图迁移(§4.3)

两个任务:push-T(把 T 形块推到目标位姿)、cable routing(把线穿过两个夹具),各用 **1 小时自采手机视频**训意图模型。**同一份意图预测**分别喂给两个仿真训练的 sensorimotor 策略:主 LEAP 手策略、以及带少量本体特定改动的平行夹爪变体。每本体每任务 15 次真机试验。

| 任务 | 灵巧手 | 平行夹爪 |
|---|---|---|
| Push-T | 12/15 | 10/15 |
| Cable-routing | 7/15 | 9/15 |
| **合计** | **19/30 ≈ 63%** | **19/30 ≈ 63%** |

两种形态迥异的本体拿到**相同的聚合成功率**。有意思的是 cable routing **反而更偏爱夹爪**(两片对置爪很适合夹细线,而灵巧手很难精确抓住细线);两本体的失败都集中在 OOD 状态,作者归因于仅 1 小时手机语料太小。

### 3.4 Q4:sensorimotor 策略消融(§4.4,仿真,各 3 seed)

- **Teacher**(Fig 7A):① 用 MLP 编码器(拼接所有输入、无 token 化)替换点-token 编码器——仍能定位物体但丢失稳定抓取接触与 in-hand 操作所需的逐点细节,episode reward 明显更低;② **去掉 eigen-grasp 基**(只用逐关节动作)——探索空间大得多,收敛差,几乎学不起来。
- **Student**(Fig 7B):③ 用 DAgger-BC(50/50 teacher/student rollout blend)替换混合蒸馏——是个有竞争力的基线,但因纯模仿 teacher,student 无法充分利用自己独有的输入模态(腕相机);④ **移除腕相机**——物体几何未解析,退化到类似 MLP 编码器的水平。

## 四、局限性

论文自陈三条(§5),都相当诚实:

1. **流水线脆性**:系统过度模块化,SAM 3.1、DenseTrack3Dv2、ViPE、WiLoR 各自在数据抽取和部署时都是一个失败点,感知故障(in-hand 遮挡、无纹理物体)沿链向下游传播。理想方案是把意图模型从原始视频端到端训练,用单一学习表征替换这条链。
2. **任务-条件缺口**:没有可验证终止条件的任务会让意图模型无限循环(搅拌当前需外部强加停止);训练新任务需人工筛语料,难以扩展。作者认为用文本条件 scaling 到上千任务、并配高层规划器串联子任务("搅拌"→"放下勺子")可同时解决二者。
3. **有损的显式接口**:$f_\theta$ 与 $\pi^S$ 之间的"3D 流 + palm pose"接口是手工设计的,丢弃了手指构型、精细接触等信息;即便意图预测很准,也缺策略完整复现人类演示所需的线索。一个跨两个子模型联合优化的**隐式中间表示**或能让接口自适应策略所需信号(引 The Bitter Lesson)。

## 五、评价与展望

**优点**:
- **接口设计是最大亮点**。用"短时物体流 + palm pose"作为跨本体的意图媒介,把"两个可独立扩展的监督源"(免遥操作的人类视频 + 任意规模的仿真)干净地缝在一起——这正切中当前 VLA 路线"动作数据是瓶颈"的痛处。同一意图模型零改动驱动灵巧手与平行夹爪(63% vs 63%)是很有说服力的解耦证据。
- **闭环 vs 开环的对照实验做得干净**:两者共享同一 sensorimotor 策略,只差意图是否持续重查,73% vs 28% 的差距因此几乎可完全归因于"闭环重规划"这一个变量,而非策略强弱。这比很多"整体换掉一套系统"的对比更有诊断力。
- **数据引擎具体可复现**:ViPE→SAM 3.1→DenseTrack3Dv2→WiLoR+MANO 四阶段把野外视频转成 3D 物体流 + palm pose 监督,唯一人工标注是物体名,20k clip/任务的规模现实可达;对人手像素的重度增强是防"依赖演示者外观"的关键 trick。

**缺点与存疑**:
- **意图 scaling 的证据偏弱**。Q2 只在 binning 单任务、且真机每点仅 30 次试验;论文自己也承认幂律外推不确定性巨大。"更多视频→更好"的定性结论成立,但离"scaling law"还很远,标题里的 "scalable" 更多是架构主张而非实证。
- **系统复杂度高、脆性由作者自陈**。四个现成大模型串成的感知链每个都是失败点,真机成功率的天花板很可能被这条链而非策略/意图本身限制(Fig 12 中 perception loss、unrecoverable state 占了相当比例)。
- **palm pose 接口对灵巧手是"有损"的**:丢弃手指构型,对精细 in-hand 操作(如穿线,灵巧手反被夹爪超越)是结构性短板。
- **仿真侧参考完全程序化**:sensorimotor 策略从没见过真实意图分布,四段式程序参考与 $f_\theta$ 实际产出的 $\mathcal{R}$ 之间的分布差可能是部分 OOD 失败的隐因。

**与公开工作的关系**:
- 与"从视频抽轨迹直接模仿"(DexMV、ViViDex 等)相比,LUCID 只用视频监督**意图**而非动作,规避了 human-to-robot 本体 gap;
- 与"视频生成器出开环计划"(Track2Act、Dreamitate、NovaFlow 等)相比,LUCID 用闭环短时重查替代一次性长视频生成,且不承担视频生成 2–3 分钟/次的开销;
- 物体流作为跨域接口延续了 Flow-as-interface(Xu et al.、General Flow、Any-point Trajectory)一脉,但**补上了 palm pose** 这一手部构型信息,针对多指操作是有价值的补全;
- sensorimotor 策略侧则是 DextrAH-RGB / DexterityGen / SimToolReal 那条"仿真通才 + 域随机化 + teacher-student"路线的直接沿用。

**开放问题与可能改进**:
1. 把感知链**端到端化**(作者已点名),用一个从原始视频直接预测物体流 + palm pose 的模型替代四模块串联,既减脆性也可能让 scaling 更平滑;
2. 用**文本/语言条件**统一多任务意图模型,替代"每任务人工筛 20k clip",这才可能把 "scalable" 落到任务维度;
3. 用**联合优化的隐式意图接口**替代手工的 3D 流 + palm pose,让接口自适应下游策略需要的信息(尤其手指精细接触);
4. 在仿真参考里注入更接近真实 $f_\theta$ 产出的分布,缩小 sim 参考与 deploy 参考之间的 gap。

## 参考

1. M. Xu et al. *Flow as the cross-domain manipulation interface.* CoRL 2025.（物体流作为跨域接口,LUCID 意图侧的直接前身)
2. T. G. W. Lum et al. *Crossing the human-robot embodiment gap with sim-to-real RL using one human demonstration.* arXiv:2504.12609, 2025.（单人示范 + sim-to-real RL,retargeting 与 eigen-grasp 来源)
3. R. Singh et al. *DextrAH-RGB: Visuomotor policies to grasp anything with dexterous hands.* arXiv:2412.01791, 2024.（sensorimotor 策略的 curriculum/teacher-student 配方来源)
4. K. Kedia et al. *SimToolReal: An object-centric policy for zero-shot dexterous tool manipulation.* arXiv:2602.16863, 2026.（动作积分与 sim-to-real 配方)
5. N. Karaev et al. *CoTracker3: Simpler and better point tracking by pseudo-labelling real videos.* ICCV 2025.（意图模型 backbone,被改造成前向点-token 预测器)
