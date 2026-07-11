# Video-to-Control Survey：从视频到控制——从时序视觉数据学习机器人操作接口的综述

> **论文**：*From Video to Control: A Survey of Learning Manipulation Interfaces from Temporal Visual Data*
>
> **作者**：Linfang Zheng, Zikai Ouyang, Chen Wang, Jia Pan, Wei Zhang（通讯）
>
> **机构**：香港大学（The University of Hong Kong）；南方科技大学（Southern University of Science and Technology, SUSTech）；鹏城实验室（Peng Cheng Laboratory）；LimX Dynamics
>
> **发布时间**：2026 年 06 月（arXiv 2604.04974v3，文献截止 2026-05-25）
>
> **发表状态**：The International Journal of Robotics Research (IJRR)，2026（录用/in press，DOI 待分配）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.04974) | [PDF](https://arxiv.org/pdf/2604.04974)
>
> **分类标签**：`learning-from-video` `manipulation-survey` `video-prediction` `latent-action` `world-action-model`

---

## 一句话总结

这是一篇以"**接口为中心**（interface-centric）"重组"从视频学操作"文献的综述：它不按模型类别（Transformer vs. Diffusion）或生成技术分类,而是追问**视频派生的时序结构在"哪里"、以"何种显式程度"接入机器人控制回路**,由此把全部方法归为三大家族——直接视频-动作策略（direct video–action）、隐动作方法（latent-action）、显式视觉接口（explicit visual interface）——并提出核心论点:三家在模型架构上的差异远小于在"接口如何被表示、grounding 在哪里发生、部署负担如何"这三条耦合设计轴上的差异,而最紧迫的未解难题都汇聚在"**机器人集成层**（robotics integration layer）"——把视频预测落成可靠闭环行为的那层机制。

## 一、问题与动机

机器人操作的根本瓶颈是**数据**:带同步动作标签的机器人轨迹昂贵、难扩展,即便有 Open X-Embodiment（OXE,100 万+ 轨迹、22 种本体、527 项技能）这类协同努力仍受限。与之相对,网络与第一人称设备上存在海量**无动作标注视频**(Ego4D、EPIC-Kitchens 等),它编码了物体如何运动、接触如何展开、场景如何随交互演化——但缺少同步动作标签,且常来自不同本体、视角、传感模态。

核心张力:**最丰富的经验来源(无动作视频)恰恰离最有用的监督(机器人动作)最远。** 弥合这一鸿沟不只是数据问题:从视频抽取的时序结构最终必须闭合一个控制回路、尊重物理约束、并落在具体机器人的本体极限内。

综述界定的统一问题是:*如何用大规模无动作视频(一种可扩展的世界动态观测)来学习支持可靠机器人操作的控制接口?* 它刻意区别于既有综述:VLA/基础模型综述以动作标注轨迹为主要 grounding 信号;视觉感知/世界模型综述多扎根于机器人交互数据或经典传感-控制管线;而本综述把**无动作时序视频作为首要监督信号**,并按"视频动态与机器人动作之间的接口在何处构建"来组织。

**纳入标准**(三条须同时满足):(i) 以视频时序连续性为核心训练信号(视频/子目标预测、点/流轨迹预测、从帧间转移学时序预测隐变量),而非静态单帧目标;(ii) 关键接口从无动作视频学得或预训练(可大规模);(iii) 该接口通过机器人数据/策略/规划/控制回路 grounding 到操作。**排除**:纯静态图 affordance(MOKA、FlowBot3D、KETO、ReKep)、主要从机器人交互数据学策略的自监督表示法(ManipulateBySeeing、DynaMo)、从视频学奖励而无预测接口的方法(VIP、LIV)、以 RL/MPC 仿真为主的通用世界模型(UniSim、PointWorld)、以及主要靠动作标签监督的 VLA(RT-1/RT-2/π₀,仅作动机背景引用)。

## 二、核心方法(接口中心分类法)

### 2.1 两条设计轴与设计空间

综述把代表性方法锚定在二维设计空间(Fig. 3):

- **横轴——接口显式度(interface explicitness)**:方法把"视觉变化 → 控制"这条链暴露得多显式。最左极端,链隐藏在共享表示与动作头内部;向右,引入显式中间变量;最右暴露可解释输出(子目标、视频计划、轨迹、位姿)。
- **纵轴——离机器人动作的距离(distance from robot actions)**:预测/条件的对象与机器人实际执行的对象之间的抽象分离。低=接近电机指令(直接动作预测);高=指定更抽象目标(子目标、物体运动、位姿计划),需要额外控制模块把目标翻译成可执行动作。

标记形状=所属家族,标记大小=典型 grounding 所依赖的**动作标注机器人数据量**(低/中/高)——这一维刻画了"从机器人动作监督中解耦"的程度。

**分配规则(可复现)**:按"在最终 grounding 阶段承担主要控制负担的信号"归类。部署系统直接吐动作、无可检视中间目标 → **direct**;可迁移物是从转移学得的紧凑类动作码 → **latent**;暴露下游控制器要跟踪的视觉/几何目标 → **explicit**。据此解决边界:APV、ContextWM 归为 direct 边界(其隐变量构成预测**状态** 而非类动作接口);UVA、Fast-WAM 保持 direct(测试时消费动作头而非显式视觉目标);SWIM 归为 explicit 边界(暴露的控制对象是 affordance/轨迹式)。

### 2.2 三大家族

**家族 I:直接视频-动作策略(§4)**——接口隐式,时序视频预测塑造共享 backbone 的内部表示,动作从中直接解码,不暴露可检视中间目标。这直接对应新兴的"**世界-动作模型**(world–action model)"概念。三类:

- **联合视频-动作生成器**:GR-1(GPT 式联合 token,Ego4D 预训练)、GR-2(web 级视频+多机器人扩展)、PAD(未来图像与动作**联合去噪**,视频-only 样本对动作分支做掩码协同训练)、UWM(按 timestep 解耦的模态特定扩散,可查询为策略/预测器/正逆动力学)、UVA(共享表示+分离视频/动作头,动作-only 推理更省)、Cosmos Policy(把动作/本体/价值作为额外 latent-frame 插入预训练视频 backbone,支持价值引导规划)、Fast-WAM(仅**训练时** 视频协同训练、测试时关掉视频分支,用注意力掩码防动作分支依赖特权未来帧)。
- **冻结预测视频特征 + 动作头**:VidMan(冻结 Open-Sora 式视频扩散做动力学感知编码器,轻量自注意力 adapter 学逆动力学式映射)、VPP(改造现成视频基础模型 SVD 的预测特征做控制表示)。
- **隐状态世界模型(边界)**:APV、ContextWM(用无动作视频初始化时序预测隐状态,再经 DreamerV2 式基于模型的 RL grounding;视频预测塑造隐状态空间而非直接监督动作头)。

**家族 II:隐动作方法(§5)**——引入从"观测如何变化"学得的紧凑**类动作抽象**,再用较少动作标注数据 grounding 到可执行指令。核心是把 direct 模型中混淆的三个角色解耦:转移动力学模型(从无动作视频学)、隐策略/规划器(在隐动作空间选码)、grounding 模块(把隐码映射到真实动作)。两类:

- **作为独立控制接口**:CLASP(从无动作视频发现连续转移隐变量,用 MPC 式规划朝图像目标搜索,再经小 grounding 模块落地;引入"minimal + composable"两个偏置)。
- **嵌入指令条件策略学习**:LAPA(VQ 隐码作**预训练目标**,换真实动作头微调)、Moto(把单步伪动作扩到序列级自回归隐 token 预测)、UniVLA(**任务中心** 隐动作,在 DINOv2 特征空间学,抑制相机/背景干扰)、ConLA(对比式把码本推向运动相关、远离外观)、RotVLA(SoftVQ + $SO(n)$ 旋转组合约束)、ALAM(加性转移几何:相邻隐码近似相加=更长转移、反转≈抵消)、HiLAM(把短程隐码聚成长程技能)。

**家族 III:显式视觉接口(§6)**——预测结构化、人可读的目标(子目标图、视频计划、轨迹、位姿),由独立下游控制器显式跟踪。训练模块化:视频预训练接口预测器 + 小机器人数据集训练的控制器。两类:

- **帧式接口(frame-based)**:UniPi(密集视频计划+学得逆动力学)、Gen2Act(生成人类视频计划+闭环视频条件策略)、AVDC(视频计划→光流+PnP 几何转 6D 位姿)、RIGVid(多候选 rollout + VLM 选择,再转 6D 物体位姿)、Dreamitate(立体视频计划→工具 6D 位姿,MegaPose)、GVF-TAPE(RGB-D 未来观测→末端执行器位姿)、Dream2Flow(视频计划→3D 物体流)、SuSIE(单张子目标图 + 目标条件策略,InstructPix2Pix 编辑)、CLOVER(RGB-D 子目标序列 + 误差驱动闭环重规划)、V2A(自采 rollout + hindsight relabel)。
- **轨迹式接口(trajectory-based)**:VRB(接触点+接触后 2D 轨迹)、SWIM(affordance + 隐世界模型规划,边界)、MimicPlay(3D 人手轨迹)、ATM(任意点 2D 轨迹,CoTracker 造监督)、Im2Flow2Act(物体中心 2D 点流)、Track2Act(2D 轨迹→刚体 SE(3) 计划)、GeneralFlow(3D 物体点轨迹,SVD 对齐,零样本)、SKIL-H(语义 3D 关键点轨迹)、ZeroMimic(人手腕 SE(3) 抓后接口)。

### 2.3 关键公式:隐动作的正逆动力学分解

隐动作发现通常把它建模为在某个空间上的**正/逆动力学分解**:

$$z_t \sim q_\phi(\cdot \mid o_t, o_{t+H}), \qquad \hat{o}_{t+H} \sim p_\theta(\cdot \mid o_t, z_t)$$

其中 $H$ 通常为 1 或固定预测步长。把 $z_t$ 解释为"动作",则编码器 $q_\phi$ 是**隐逆动力学模型**(推断转移之因),解码器 $p_\theta$ 是**隐正动力学模型**(预测施加 $z_t$ 之果);二者间对 $o_{t+H}$ 与 $\hat o_{t+H}$ 施加带瓶颈约束的重建目标,迫使隐变量捕获"观测间发生了什么变化"而非静态场景内容。

> **用大白话说**:给模型看"前一帧+后一帧",让它猜"中间发生了什么动作";再让它只拿"前一帧+这个动作"去还原后一帧。因为动作这条通道被"卡了脖子"(瓶颈),它只能塞进真正引起画面变化的信息,于是这个隐变量就长得像"动作"了。

**瓶颈机制** 决定 $z_t$ 的形态与容量:β-VAE/信息瓶颈(连续,惩罚随机隐变量携带的信息量)、硬 VQ 码本(离散**动作 token**,支持字典式 grounding 与语言兼容)、SoftVQ(软码本,隐变量=码本原型的加权组合,可连续变化)。综述强调:瓶颈把 $z_t$ 偏向紧凑转移描述子,但"是否真的**可控**(controllable)"还取决于预测空间、辅助约束与 grounding 过程。

### 2.4 控制回路抽象(Fig. 4)与三条论点

- **Direct**:感知与动作坍缩进单次推理调用,执行模式(逐步/分块/滚动时域)决定重观测间的开环长度。类比端到端视觉运动伺服。
- **Latent**:在观测与指令间插入可被规划/解码/保留为 token/与动作协同生成的类动作变量,控制质量取决于隐变量可辨识性与 grounding 对齐。类比压缩动作空间上的学得型 MPC。
- **Explicit**:控制器跟踪预测目标,构成两级层级,给"执行前验证"开了口子但引入跟踪误差。类比"感知+规划→控制器跟踪"的经典分层。

三大**贡献**:(1) 接口中心分类法;(2) 逐家族的控制集成分析(回路如何闭合、执行前能验证什么、失败从哪进入);(3) 跨家族综合 → **机器人集成层** 论点。

## 三、实验结果(综述汇编的关键数字)

综述反复声明:所有数字**不是排行榜**,各论文在机器人/数据规模、本体、评测协议上不可比,只用来指示"证据类型与成熟度"。以下为其汇编的三张 within-paper 快照表的关键项。

**Direct 家族(Table 4)**——CALVIN 长程语言条件多任务(ABC→D,链式 1–5 指令成功率 %,及平均完成任务数 Avg.Len):

| 方法 | SR@1–5 (%) | Avg.Len | 说明 |
|---|---|---|---|
| GR-1 | 85.4 / 71.2 / 59.6 / 49.7 / 40.1 | 3.06 | 100% ABC 训、D 测 |
| VidMan | 91.5 / 76.4 / 68.2 / 59.2 / 46.7 | 3.42 | 冻结视频特征+adapter |
| VPP | 96.5 / 90.9 / 86.6 / 82.0 / 76.9 | 4.33 | 视频基础模型预测特征 |

其他:PAD 在 MetaWorld 50 任务平均 SR 72.5%,VPP 68.2%;LIBERO(Sp/Ob/Go/Lo)Cosmos Policy 98.1/100.0/98.2/97.6、Fast-WAM 98.2/100.0/97.0/95.2;GR-2 在 >100 任务自定义环境平均成功率 74.7%。

**Latent 家族(Table 7)**——LIBERO/CALVIN 与 SIMPLER:

| 方法 | 基准 | 结果 |
|---|---|---|
| UniVLA | LIBERO 平均 (Sp/Ob/Go/Lo) | 95.2(全预训练)/ 92.5(Bridge-V2)/ 88.7(纯人类视频) |
| LAPA | LIBERO 平均 | 65.7 |
| RotVLA | LIBERO 四套件均值 | 98.2 |
| ALAM | LIBERO 四套件均值(π₀+3B flow) | 98.1 |
| Moto | CALVIN ABC→D 链长 | 3.10 |
| LAPA/ConLA/Moto | SIMPLER 平均 SR | 57.3 / 60.4 / 61.4(不同预训练源) |
| RotVLA | RoboTwin2.0(50 双臂任务) | Clean 89.6 / Random 88.5 |

**Explicit 家族(Table 10)**——CALVIN、LIBERO 与真机零样本:

| 方法 | 基准 | 结果 |
|---|---|---|
| UniPi | CALVIN SR@1–5 | 56 / 16 / 08 / 08 / 04 |
| SuSIE | CALVIN SR@1–5 | 87 / 69 / 49 / 38 / 26 |
| CLOVER | CALVIN SR@1–5,Avg.Len | 96 / 84 / 71 / 58 / 45,3.53 |
| ATM | LIBERO (Sp/Ob/Go/Lo),L-90 | 68.5 / 68.0 / 77.8 / 39.3,48.4 |
| GVF-TAPE | LIBERO 三套件均值 | 95.5 / 86.7 / 66.8 |
| GeneralFlow | 真机 18 任务 6 场景(零样本) | 平均 81 |
| ZeroMimic | 真机(EpicKitchens→机器人,零样本) | 71.9(Franka 9 技能)/ 65.0(WidowX) |

综述从这些数字读出的**结构性发现**:在 direct 家族,时序视频预测"更像是训练时的**表示塑造** 机制,而非提供有用的测试时计划"——UVA 架构上就走 action-only 推理、Fast-WAM 消融显示关掉测试时视频生成对下游控制的损害远小于关掉训练时视频协同训练。跨套件比较则被数据规模/本体/任务分布混淆,只在共享套件(CALVIN、MetaWorld、LIBERO 切片)内有意义。

## 四、局限性

作为综述,其自身边界与不足:

1. **时间截点与预印本比重高**。文献截止 2026-05-25;Table 1 用上标标注了大量方法仅为 workshop/arXiv 预印本(如 Cosmos Policy、Fast-WAM、RotVLA、ALAM、ConLA、Dream2Flow、X-WAM、LDA-1B 等),综述明确"把预印本量化结果仅当作新兴趋势的指示,而非跨方法排名的依据",综合结论主要建立在同行评审工作上。
2. **评测碎片化,无法给出排行榜**。方法在机器人-数据预算、观测模态、任务难度、评测协议上差异巨大;共享基准(CALVIN、LIBERO)仅覆盖少数方法,大量方法在不可比的自定义套件上评测。因此三张快照表都带"comparability caveats"标注,只指示证据成熟度。
3. **范围限定**。只覆盖操作(manipulation),不含移动/导航/全身控制;刻意排除静态 affordance、纯奖励学习、通用世界模型、动作标注 VLA 等相邻方向(仅作对比背景),因此不能替代 learning-from-video 的全景。
4. **家族边界本身有主观性**。许多系统是混合体(综述反复承认),按"最终 grounding 阶段主要控制负担"分配虽可复现,但把 UVA/Fast-WAM/APV/ContextWM/SWIM 等硬性归边界,可能掩盖它们跨家族的特性。
5. **真机证据仍稀疏**。多张能力表的"Real-Robot Scope"列有大量"—",综述提示这应读作"评测尚不成熟"的信号——不少视频派生接口至今仍主要靠"表示/仿真机制"验证,而非真机部署。

## 五、评价与展望

**优点。** (1) **视角新颖且实用**:以"接口在哪、多显式、grounding 在哪、部署负担如何"重组文献,比按 Transformer/Diffusion 或"world model"标签分类更能暴露真正决定成败的控制耦合选择——这一点抓得很准,例如它令"UWM/PAD/UVA 表面都是联合视频-动作,但测试时推理路径与可验证性截然不同"变得可见。(2) **控制集成视角罕见而扎实**:Table 12 沿"回路闭合方式、执行前验证程度(direct 低 / latent 中低 / explicit 中高)、主要控制风险、常见缓解"逐家族对齐,把 robotics 而非 ML 的关切摆到中心。(3) **"机器人集成层"论点凝练**:把四大开放难题(执行感知与物理 grounding、鲁棒跨本体 grounding、多模态与接触丰富操作、评测/验证/安全部署)都收敛到"把视频预测落成可靠闭环行为"这层,是一个有组织力的判断。（该论点即前述**机器人集成层** 主张的落点。）(4) **诚实**:反复强调数字不可比、拒绝造排行榜、把预印本降权,学术态度端正。

**缺点与开放问题。** (1) **缺乏定量元分析**:综述放弃了跨方法排名(有其理由),但也因此没有给出"在受控 robot-data 预算下,显式度每提高一档、跨本体迁移收益 vs. 执行脆弱性代价"的量化曲线,读者只能得到定性权衡。若能补一张"同基准同预算"的严格对照(哪怕小),说服力会更强。(2) **"预测性 vs. 可控性"张力被诊断但未给出可操作判据**:综述正确指出视频生成模型最大化视觉似然而非物理可行性,会产出"看着对、机器人够不到/动力学不可行"的目标;但对"如何在训练中注入可行性信号(约束违反、不确定性、轻量物理先验)"只停留在方向性建议。(3) **与并行综述的关系**:它自陈区别于 learning-from-video(McCarthy et al. 2025、Feng et al. 2026 以人为中心)、VLA/动作 tokenization(Zhong et al. 2025)、世界模型(Zhang et al. 2025b、Ai et al. 2025)等综述,定位于"无动作时序视频作为首要监督 + 接口构建位置",这个 niche 站得住,但与"world model"综述的边界在 APV/ContextWM/Cosmos 这类方法上仍显模糊。

**与公开工作的联系与改进方向。** 综述隐含的最有价值判断,是"隐动作/显式接口比 direct 更能利用无动作视频、也更易跨本体,但把 grounding 难题**外移** 而非**消除**"——这与 latent-action 路线(LAPA/UniVLA/Moto)近两年的快速演进、以及 explicit-flow 路线(GeneralFlow/ATM/Track2Act)的零样本跨本体结果一致。可能的改进方向:(a) **多分辨率混合接口**——高层视觉上下文(帧式)+ 短程易验证目标(轨迹式)配对,兼顾 holistic guidance 与 compact precision;(b) **执行感知的接口学习**——把可行性/接触一致性检查从"事后"提到"生成/解码时",这正是综述反复呼吁却指出当前系统"极少集成"的空白;(c) **多模态 grounding**——把力/触觉与视觉派生接口融合以覆盖接触丰富任务(现有方法几乎全是 vision-only、刚体);(d) **统一评测基础设施**——在受控 robot-data 预算、观测模态、任务难度下,增加"扰动鲁棒性、恢复行为、不确定性校准"等超越成功率的指标。

## 参考(最相关)

1. Wu et al., **GR-1**: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation, ICLR 2024.
2. Ye et al., **LAPA**: Latent Action Pretraining from Videos, ICLR 2025.
3. Du et al., **UniPi**: Learning Universal Policies via Text-Guided Video Generation, NeurIPS 2023.
4. Zhu et al., **UWM**: Unified World Models — Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets, RSS 2025.
5. Bu et al., **UniVLA**: Learning to Act Anywhere with Task-centric Latent Actions, RSS 2025.
