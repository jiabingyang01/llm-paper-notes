# GenSim2：用多模态与推理大模型规模化生成机器人数据

> **论文**：*GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs*
>
> **作者**：Pu Hua, Minghuan Liu, Annabella Macaluso, Yunfeng Lin, Weinan Zhang, Huazhe Xu, Lirui Wang（et al.）
>
> **机构**：清华大学交叉信息研究院（IIIS）；UCSD；上海交通大学；MIT CSAIL
>
> **发布时间**：2024 年 10 月（arXiv 2410.03645）
>
> **发表状态**：未录用（预印本，arXiv v1）
>
> 🔗 [arXiv](https://arxiv.org/abs/2410.03645) | [PDF](https://arxiv.org/pdf/2410.03645)
>
> **分类标签**：`仿真数据生成` `LLM任务合成` `关节物体操作` `sim-to-real` `多任务策略`

---

## 一句话总结

GenSim2 让多模态/推理 LLM（GPT-4V、OpenAI o1）自动"提任务 → 写代码 → 用 kPAM 关键点规划器求解 → 采集示范"，规模化生成 100+ 关节物体与长程操作任务的仿真数据；再用点云 + 本体感受的 PPT 策略蒸馏这些数据，实现 zero-shot sim-to-real，并在真机上以"仿真数据 + 少量真机数据"联合训练把 8 个真机任务平均成功率从 0.363 提到 0.575（绝对 +20%、相对 +50%）。

## 一、问题与动机

机器人学习需要海量交互数据,真机采集昂贵。仿真是低成本可规模化的出路,但仍有两大瓶颈:

- **任务复杂度受限**:此前的 LLM 生成仿真任务工作（如 GenSim、RoboGen）主要停留在**桌面自上而下的 pick-and-place**,难以覆盖需要 6-DOF 带接触运动、关节物体操作、以及多步长程任务的复杂场景。
- **求解与迁移困难**:生成了任务还要**自动求解**产出示范数据,并且策略要能从仿真**迁移到真机**;而多数 sim-to-real 方法只针对单一任务,缺乏可扩展性。

GenSim2 的核心动机:把 LLM 从"只用语言/代码知识"升级为"用视觉 + 空间推理知识"(GPT-4V 看渲染图、o1 做思维链),从而生成更真实、更复杂的关节物体任务,并配套一个原生适配 sim-to-real 的点云策略架构,把仿真里"廉价无限"的数据蒸馏成真机可用的多任务策略。

## 二、核心方法

整体是一条 agent 流水线,分四阶段:**任务提议 → 示范生成(求解器创建)→ 多任务训练 → 泛化评测与 sim-to-real 迁移**。

### 2.1 任务提议(Task Proposal)

- **基元任务(primitive)**:从固定的关节资产库 + 一个小的手工示例任务库出发,in-context 提示 LLM 生成新任务(任务描述短语 + 所用资产 + 任务代码),代码编译后交给下阶段生成示范。
- **长程任务(long-horizon)**:在任务提议与代码实现之间插入**任务分解**,把长程任务拆成若干基元子任务。两种策略:
  - **Top-down(自上而下)**:先直接生成一个长程任务,再分解为各带专用求解器的子任务;
  - **Bottom-up(自下而上)**:先造好基元任务建成任务库,再让 LLM 从库里**挑选并组合**成新的长程任务。
  论文发现引入**推理 LLM(o1)** 能改善此阶段的任务提议质量。

### 2.2 示范生成:多模态求解器创建

求解器需满足:通用于 6-DOF 任务、对场景配置鲁棒、执行快、确定性高成功率。核心求解器是**kPAM 关键点运动规划器**(另有 RL 学习器作补充)。

kPAM 把一个操作任务定义为求解一个**作动位姿(actuation pose)** $T$——即抓取/操作目标物所需的齐次变换,通过若干**基于关键点的约束**求解一个优化问题:

$$
T^\ast=\arg\min_{T\in SE(3)}\ \sum_k c_k(T;\,\mathcal{P})\quad \text{s.t.}\quad \Phi(T;\,\mathcal{P})=0
$$

其中 $\mathcal{P}$ 是人工标注的物体关键点集合($\langle$ 关键点, 关键点 $\rangle$ 或 $\langle$ 向量, 向量 $\rangle$ 这类元组),$c_k$ 是代价项,$\Phi$ 是几何约束(如 $p_1,p_2$ coincident 重合、$v_1,v_2$ parallel 平行、$v_3,v_2$ orthogonal 正交)。

> 用大白话说:kPAM 不直接输出"手该走到哪个 xyz",而是先想清楚"要把夹爪摆成一个与物体几何对齐的姿态"(比如夹爪 x 轴要跟箱盖某条边平行、夹爪指尖要落在盖子铰链关键点上),把这些几何关系写成约束去解一个位姿。因为约束都是"物体为中心"的相对关系,换一个同类别、不同位置/朝向的物体它照样成立,天然带类别级泛化。

解出作动位姿后,再围绕它扩展出一条**作动运动(actuation motions)** 轨迹——引入**预作动(pre-actuation)** 与**后作动(post-actuation)** 一串路点(如先沿 $x$ 平移 $-0.05$、沿 $z$ 平移 $-0.15$ 接近,再 Rotate $30°$ 掀盖):

$$
\tau=\{\,T_{\text{pre}},\ T^\ast,\ T_{\text{post}}\,\}
$$

> 用大白话说:光有一个"抓到位"的姿态还不够,得补上"怎么靠近、抓完怎么发力(转、推、拉)"。pre/post 运动就是这条前后串起来的路点序列,合起来才是一条能真正把箱子打开的完整轨迹。

**关键的多模态一环**:生成约束时,先跑一遍任务代码渲染出**场景图像**,把图像 + 物体关键点一起喂给 MLLM(GPT-4V)生成规划器约束;再把作动位姿可视化,回喂 MLLM 生成作动运动;并用**拒绝采样(reject sampling)** 让模型多轮自省、修正之前的输出。

> 用大白话说:纯文本 LLM 不知道"盖子在物体的哪一侧、朝哪开",凭空写约束会幻觉。GenSim2 让模型"边看渲染图边写代码",还允许它反复重试挑一个能跑通的方案——这就是它比纯语言方法强的关键。

**RL 学习器(补充求解器)**:对 kPAM 难以定义约束的任务(薄物体、接触密集),让 LLM 生成奖励函数并用 PPO 训练。但论文优先用 kPAM,因为其输出运动更平滑自然、可泛化,且规划快($\sim 2$ 秒/配置)。

### 2.3 多任务策略:PPT(Proprioception Point-cloud Transformer)

处理三种可从真机获取的观测:**点云 $o_{\text{pcd}}$、本体感受状态 $o_{\text{prop}}$、语言任务描述 $\ell$**。各自编码 + cross-attention token 化,经 transformer 融合进共享隐空间,得到全局条件 token,再由动作头(MLP / transformer decoder / diffusion 均支持)预测动作序列:

$$
z=\mathrm{Transformer}\big(\mathrm{Enc}_{\text{pcd}}(o_{\text{pcd}}),\ \mathrm{Enc}_{\text{prop}}(o_{\text{prop}}),\ \mathrm{Enc}_{\ell}(\ell)\big),\qquad a_{t:t+H}=\pi_{\text{head}}(z)
$$

> 用大白话说:策略只吃"点云 + 关节状态 + 一句话指令"这三样真机也拿得到的东西,故意不用 RGB 颜色。因为点云在仿真和真实之间的差距远小于 RGB 图像,这样训出来的策略天生就好迁真机。

**sim-to-real 关键处理**:训练时丢掉点云颜色,并对仿真点云做数据增强(裁剪、加高斯噪声、随机丢点);真机侧对采集点云做均匀采样 + 最远点采样 + 离群点去除,得到干净点云作为推理输入。整套配合 kPAM 生成数据自带的**物体级 + 空间级随机化**当作 domain randomization。

## 三、实验结果

### 3.1 任务生成质量(vs RoboGen,Table 1)

评测两类成功率:**execution rate**(整条流水线无语法/运行错误跑通)与 **solution rate**(生成任务被成功求解)。GenSim2-B = bottom-up,GenSim2-T = top-down 用 GPT-4,GenSim2-T (o1) = top-down 用 OpenAI o1。

| 任务类型 | 指标 | GenSim2 | RoboGen |
|---|---|---|---|
| 基元(primitive) | Execution | **0.94** | 0.94 |
| 基元 | Solution | **0.78** | 0.58 |

| 长程任务 | GenSim2-B | GenSim2-T | GenSim2-T (o1) | RoboGen |
|---|---|---|---|---|
| Execution | **1.00** | 0.83 | 0.87 | 0.76 |
| Solution | **0.68** | 0.54 | 0.60 | 0.43 |

要点:基元任务 solution rate 78% vs RoboGen 58%(约 +25% 相对提升);长程任务上 bottom-up 组合已建任务优于 top-down 从零分解;o1 的推理能力让 top-down 分解更合理(solution 0.60 > GPT-4 的 0.54)。

### 3.2 生成流水线消融(Fig 5,solution rate)

| 消融维度 | 设置 | Solution Rate |
|---|---|---|
| LLM 类型 | Multi-modal LLM(GPT-4V) | **0.78** |
| | Reasoning LLM | 0.36 |
| | Vanilla LLM | 0.18 |
| 拒绝采样最大迭代 | 1 次 | 0.10 |
| | 3 次 | 0.50 |
| | 5 次 | 0.70 |
| 思维链提示链 | GenSim2(约束→运动分链) | **0.78** |
| | w/o CoT(一次直出整个求解器) | 0.44 |

结论:视觉信息(0.78 vs 0.18)是成败关键——纯文本模型拿不到物体结构/空间关系,约束和运动误差大;多轮拒绝采样的自省显著提分;把求解器生成拆成"先约束、后运动"的提示链比一次直出好 30%+。

### 3.3 多任务仿真训练与泛化(Fig 6,382M 参数策略)

- **任务数扩展**:联合训练 4/10/15/20/24 个任务(低数据,10 demo/task),成功率随任务数**先降后升**(约 0.4 → 0.3 → 0.5),体现规模化收益。
- **模态消融(4 任务)**:PPT 全模态 **0.66**;去语言 0.46;去点云 0.43;去本体感受 0.55——三种模态都有贡献,点云影响最大。
- **物体级泛化**:PPT 在训练/未见实例上 0.49 / 0.46(**掉不到 3%**);RGB 策略 0.29 / 0.19(掉幅大)。点云 + 数据随机化让策略在未见物体实例上几乎不掉点。

### 3.4 真机实验(Table 2,Franka Research 3,8 任务,每任务 100 仿真示范 + 10 真机示范,10 次评测)

| 训练数据 | OpenLaptop | CloseLaptop | OpenSafe | CloseSafe | CloseDrawer | SwingBucket | OpenBox | CloseBox | 平均 |
|---|---|---|---|---|---|---|---|---|---|
| Real-only(仅 10 真机) | 0.5 | 0.0 | 0.2 | 0.4 | 1.0 | 0.5 | 0.2 | 0.1 | 0.363 |
| Sim-only(仅 100 仿真) | 0.7 | 0.5 | 0.1 | 0.3 | 0.8 | 0.5 | **0.5** | 0.0 | 0.425 |
| Combined(仿真 + 真机) | **0.8** | **0.7** | **0.3** | **0.6** | 1.0 | **0.8** | 0.0 | **0.4** | **0.575** |

要点:(a) 纯仿真数据即可支撑**有效的 zero-shot sim-to-real**(0.425 > 仅真机的 0.363);(b) 仿真 + 真机联合训练把平均成功率从 0.363 提到 0.575——**绝对 +20%、相对 +50%**,印证大规模高质量仿真数据能显著减轻真机采集负担。

### 3.5 人力成本(Appendix)

关键点标注每资产 1-3 个、平均 8.2 秒/任务;用户实验(Table 3)显示 GenSim2 造一个任务平均约 4 分钟,比手工设计**省时约 50%**,且 90%+ 时间只是在等 LLM 响应;专家与新手用时差距缩小,门槛更低。资产库(Table 4)覆盖 35 类关节物体(laptop/drawer/safe/microwave/box/bucket 等),含 revolute/prismatic 两种关节。

## 四、局限性

- **基础模型缺"机器人中心"知识**:GPT-4V 的 3D 空间理解仍不足,创建有意义任务、正确写代码时依然会幻觉;拒绝采样也承认 MLLM 在处理 3D 机器人场景时可能给出无关回答。
- **仍需少量人力**:关键点标注、拒绝采样引导等人工介入虽已最小化但未消除。
- **只覆盖 6-DOF、单臂、限点云**:zero-shot sim-to-real 仅在 6-DOF 任务、有限点云观测下验证,未涉及多本体/双臂/灵巧手/接触密集的高难任务。
- **难任务改由 teleop 或 RL 兜底**:极小关节物体("往里放东西"类)normal 随机化下采不到成功示范,需减半噪声;部分复杂长程任务(如 7 子任务的"准备早餐")靠遥操而非规划器求解,说明 kPAM 覆盖面有边界。
- **规模仍偏小**:100 任务、8 真机任务、每任务示范量有限,离真正"无限数据"尚远;真机成功率绝对值(平均 0.575)仍不高。

## 五、评价与展望

**优点**。(1)把"看图 + 推理"引入 LLM 任务生成,是相比 GenSim/RoboGen 的实质升级——用 GPT-4V 的渲染图理解 + o1 的思维链分解,把可生成任务从 top-down pick-place 推进到关节物体与长程操作,任务 solution rate 提升明显。(2)kPAM 关键点规划器天然"物体为中心",输出运动平滑、类别级可泛化,且 $\sim 2$ 秒规划,比 RoboGen"生成奖励 + RL 训"的路线更稳、更快、更省算力——这是它在长程任务上 execution 1.00 的根因。(3)PPT 用点云 + 本体感受 + 语言、刻意弃 RGB,把 sim-to-real 语义鸿沟压小,未见实例掉点 $<3\%$,是很干净的架构选择。(4)完整闭环(生成→求解→训练→迁真机)且开源,工程可复现性强。

**缺点与开放问题**。(1)方法很大程度绑定"可微/可约束的关键点几何 + 已知铰链结构"的关节物体,对可变形物、接触密集、双手协调、非刚体的覆盖是硬边界;难任务退化为遥操,削弱了"全自动"叙事。(2)关键点仍需人标(哪怕 8.2 秒),且约束由 MLLM 生成时的幻觉是成功率天花板的主要来源,规模再上去需更强 3D-aware 基础模型。(3)真机绝对成功率不高、任务数少,"20% 提升"是在低基线上取得,规模化的真正收益曲线尚未展开(Fig 6 左"先降后升"也说明多任务干扰未完全解决)。

**与公开工作的关系**。相较 GenSim(仅语言、top-down pick-place)与 RoboGen(奖励生成 + RL,更脆),GenSim2 的差异化在"多模态求解器 + 关键点规划"；与 RoboCasa(大规模日常任务仿真)相比更聚焦关节物体与 sim-to-real 闭环。求解器 kPAM 承接 Manuelli 等的关键点 affordance 传统,视觉提示环节与 MOKA、ReKep、Keypoint Action Tokens 等"标记式/关键点约束视觉提示"路线同源;PPT 架构与 PerAct、3D Diffusion Policy、异构预训练 transformer(HPT/GNFactor)一脉相承。

**可能的改进方向**:用更强 3D 空间基础模型(或在 3D 场景上微调 MLLM)替代 GPT-4V 以降幻觉;把关键点标注也交给自动检测器实现真正零人工;扩展到双臂/灵巧手与可变形物;把 bottom-up 任务组合做成可自我扩张的课程,并研究多任务负迁移的根因;引入生成式资产/场景合成进一步扩大任务多样性。

## 参考

1. Wang et al. *GenSim: Generating Robotic Simulation Tasks via Large Language Models.* ICLR 2024.（前作,纯语言 top-down 任务生成)
2. Wang et al. *RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation.* 2023.(主要对比基线,奖励生成 + RL)
3. Manuelli et al. *kPAM: KeyPoint Affordances for Category-Level Robotic Manipulation.* ISRR 2019.(核心求解器来源)
4. Wang et al. *Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers.* NeurIPS 2024.(PPT 架构相关)
5. Huang et al. *ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation.* 2024.(关键点约束视觉提示同源工作)
