# WM-for-Manipulation 综述：面向机器人操作的世界模型综述

> **论文**：*World Models for Robotic Manipulation: A Survey*
>
> **作者**：Fangyuan Wang, Ziyuan Wang（共同一作），Guorui Pei, Mengshi Zhang, Canxi Liang, Jun Hu, Zhongxuan Li, Jinsong Wu, Ning Han, Zeqing Zhang, Jiaming Qi, Hongmin Wu, Shiyao Zhang, Pai Zheng, Jia Pan, David Navarro-Alarcon, Sichao Liu, Peng Zhou（通讯）et al.
>
> **机构**：香港理工大学（The Hong Kong Polytechnic University）；哈尔滨工业大学（深圳）；大湾区大学（Great Bay University）；太原理工大学；香港城市大学（东莞）；广东技术师范大学；香港大学；南洋理工大学（Singapore）；东北林业大学；粤港澳大湾区国家技术创新中心；KTH 皇家理工学院（Sweden）
>
> **发布时间**：2026 年 05 月（arXiv:2606.00113）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.00113) | [PDF](https://arxiv.org/pdf/2606.00113)
>
> **分类标签**：`world-model` `robotic-manipulation` `survey` `VLA` `taxonomy`

---

## 一句话总结

这是一篇以**机器人操作(manipulation)为中心** 的世界模型综述,它拒绝按"架构/模态/损失"给世界模型下定义,而是提出一个**功能性(function-based)操作定义**——世界模型是一个"预测系统",价值取决于它**预测世界的哪些方面** 以及这些预测**如何被机器学习/控制消费**;并用三条正交轴(预测什么表征 / 预测如何连接动作 / 预测在流水线何时被用)统一梳理了强化学习、模仿学习、视频生成、3D/4D、物理建模与 VLA 五大传统,系统归纳了 34 个代表方法、34 个数据集与三层评测协议,核心论点是:操作世界模型正从"任务专用动力学预测器"演化为"可复用的预测式基础设施",而当前最大的开放问题不是预测好不好看,而是**闭环下的评测(evaluation under intervention)**。

## 一、问题与动机

**痛点:术语"world model"已严重歧义化。** 机器人学习从紧凑的任务专用控制器,扩展到大规模、语言条件、通用主义系统后,"世界模型"这个词的外延爆炸式膨胀:在一个社区里它指用于规划的紧凑 latent 转移模型;在另一个社区里它指一个 action-conditioned 视频生成器;还有人用它指学到的模拟器、奖励产生源,或嵌在 VLA 策略里的推理模块。这种模糊对操作任务代价尤其高,因为 **预测保真度(predictive fidelity)与动作效用(action utility)并不总是重合**:一个模型可以生成视觉上很像的未来,却违反接触、物体永久性、力闭合或动作可行性;反过来一个紧凑 latent 模型可能支持高效控制,却把安全部署所需的物理状态隐藏掉了。

**动机:现有综述都留了缺口。** 通用世界模型综述偏重视频生成与自动驾驶,把操作当边缘;具身智能综述把世界模型与物理模拟器并列当"使能技术";3D/4D 综述只关注几何表征、不覆盖 RL/IL 传统;最接近的操作综述要么回避严格定义(按感知/预测/控制组织),要么只盯 VLA、把 model-based RL 与 IL 排除在外。**结果是缺一篇把 RL、IL、视频生成、几何、物理、VLA 用统一"以操作为中心的预测建模"账本连起来的综述。** 本文正是填这个缺口。作者的中心立场(central position):世界模型最好被理解为一个"预测系统",其价值联合取决于它表征了未来的哪些方面、以及这些预测如何被机器人学习或控制消费。

## 二、核心方法

本文不是一个算法,而是一套**分类框架**。其形式化贡献是一个操作定义,加上四条正交的组织轴,和一组评测度量的规范化。

### 1. 世界模型的操作定义(三要件)

> 世界模型是一个预测系统(端到端学习或由学习/解析组件拼装),它估计外部世界中与任务相关的方面如何随时间演化,并且在机器人操作中,预测或评估这一演化在机器人干预下的结果。

它必须同时满足三条:
- **预测性(predictive)**:必须估计未来演化,而非只编码当前——静态视觉编码器、目标检测器、分割/语言感知模块本身都不是世界模型。
- **世界接地(world-grounded)**:必须描述外部场景的某方面(观测、物体、几何、接触、物理状态、affordance、latent 变量)——只输出效用标量的 reward/value 函数本身不算。
- **干预感知(intervention-aware)**:必须支持"如果机器人推/抓/放/插/开会怎样"的反事实推理,这通常意味着显式的 action-conditioning——纯 forward passive predictor 只有在被适配/查询/嵌入到评估机器人动作后果的系统里时才进入范围。

用大白话说:世界模型 = "能想象未来 + 想象的是真实世界 + 想象的是'我这么动之后'的未来",三者缺一不可。这条定义把 inverse model(算动作而非算后果)、policy(映射目标到动作)、孤立的 reward/value 都明确排除在外,却又足够宽到覆盖 pixel/video/latent/flow/3D/4D/physics 各种底物。

### 2. 轴一:表征(What is predicted)——五族,按归纳偏置递增排序

| 表征族 | 空间 | 代表方法 | 强项 / 弱点 |
|---|---|---|---|
| ① 图像与视频 | 像素空间(高保真,低抽象) | UniPi, SuSIE, VLP, GR-1/GR-2, DreamGen, HMA, WorldGym, Cosmos Policy, Genie Envisioner | 接口兼容(策略本就吃图像)、可被人/VLM 检视;但视觉合理 $\neq$ 动作可行,像素目标浪费容量在光照纹理背景 |
| ② 学到的 latent | 紧凑 latent(planning-friendly) | PlaNet, Dreamer, V-JEPA 2, WorldVLA, Fast-WAM, LaST-VLA, Chain of World, AtomVLA | rollout 便宜、利于搜索优化;但可审计性差、易过拟合任务/本体分布 |
| ③ 运动场与场景流 | 运动中心(motion-centric) | FLIP, FlowVLA | 与短时程物理变化对齐、暴露"哪块动了";但丢物体身份/语义、接触未发生时有歧义 |
| ④ 几何与时空(3D/4D) | 3D/4D 结构化世界 | 3D-VLA, OG-VLA, 3D-CAVLA, PointWorld, TesserAct, WristWorld, GWM | 遮挡/视角/接触几何显式化;但基础设施成本高(深度、多视、点云) |
| ⑤ 物理知情动力学 | 物理约束(constraint) | PIN-WM | 满足接触/摩擦/形变约束、外推更强;但接触不连续、摩擦难估、真实场景有未建模顺应性 |

用大白话说:这条轴的本质是"把泛化的负担放在哪":视频模型把负担押在数据规模与生成容量;latent 押在训练目标与下游策略;flow 押在"位移能否代表动作相关变化";几何押在传感与重建;物理押在系统辨识与假设有效性。**没有一族在所有操作场景通吃**,当前趋势是混合(视频被 flow 条件化、latent 建在几何基元上、Gaussian 场景耦合物理约束、模拟器被 verifier 增强)。

### 3. 轴二:预测-动作接口(How prediction connects to action)

核心区分:预测是**内部的(integrated)** 还是**外露的(explicit)**。

- **集成式预测-动作模型**:未来建模融进产生动作的模型本身,部署时不作为独立对象出现。
  - *联合世界-动作生成建模*:把未来观测与动作当**一条生成序列**——GR-1/GR-2、HMA、WorldVLA、RynnVLA-002、PAR、DUST。优点是"想象的"与"能执行的"天然一致;代价是规模化成本。
  - *预测增强的策略学习*:保持策略中心架构,预测只作辅助信号/瓶颈/推理轨迹——Seer、FLARE、CoT-VLA、DreamVLA、VLA-JEPA、FlowVLA、3D-VLA、DIAL、UP-VLA。
  - *想象驱动的策略学习*:Dreamer 式在 latent rollout 里优化策略,DayDreamer 证明可迁移到真机;后续 ReViWo、LaDi-WM、LUMOS、IQ-MPC、FOCUS、SeeX 强化多视一致/视角不变/接触敏感。
- **显式预测式规划器**:预测被暴露成一个"另一个模块必须实现"的中间目标。
  - *子目标预测*:SuSIE、GR-MG、Imagine2Act、MinD——接口最简单、内环快,但时间欠定(一张目标图说不清接触序列)。
  - *轨迹预测*:UniPi、CLOVER、EVA、V-JEPA 2、VPP、TD-MPC2、MoDem——比子目标提供更丰富时序指导,但更易受复合误差(compounding error)。
  - *路点与 latent-plan 预测*:PIVOT-R、PIN-WM——压缩未来成结构化中间变量,搜索高效但透明度低。
  - *分层预测式规划*:VLP、Reflective Planning、NovaPlan、RoboHorizon、TriVLA——多时间尺度堆叠,高层决定"该发生什么"、低层评估局部可行性,风险是误差沿层级传播。

用大白话说:集成式"想到就做,一致性好但难审计";显式式"先画个中间蓝图再让下游去实现,可视化/可替换/可打分,但存在 handoff 脆弱问题(预测对了但控制器实现不了)"。作者判断:最可靠的系统很可能是**紧动作耦合 + 保留一个外露机制来检查想象的未来是否物理可达** 的混合体。

### 4. 轴三 & 生命周期:世界模型作为基础设施(五角色)与三阶段

当交互昂贵、评测有风险、许多物理任务慢/不可逆/难复位时,世界模型越来越被当作**可复用基础设施** 反复查询。五个功能角色(Fig. 4):

1. **合成经验生成(data engine)**:DreamGen、Ctrl-World、WristWorld、GigaWorld-0、Cosmos Policy——扩展 demo 覆盖不到的物体/视角/接触/恢复行为。主要失效模式是 action inconsistency(视频可能显示成功抓取却没编码所需力/接触)。
2. **候选动作过滤与精化(predictive critic)**:GPC(Generative Predictive Control,冻结 BC 策略 + 独立训练的 action-conditioned 视频世界模型重排候选)、flow-matching 变体。受限于 proposal dependence(基座不提就打不出)。
3. **基于搜索的动作评估(evaluation oracle)**:FLIP、Vidar、VLA-Reasoner(MCTS over imagined futures)、WorldPlanner、M³PC、DINO-WM。关键优势是"落子前多次比较想象的未来",风险是搜索会放大模型误差。
4. **策略评估/改进的学习环境(learned simulator)**:Robotic World Model、DINO-WM、World4RL、RISE、GWM、WorldGym、World-Env、WMPO、Prophet、World-Gymnast、World-VLA-Loop、WoVR。中心风险是 **simulator exploitation**(策略学到利用模型 artifact 而非可迁移技能)。
5. **结果打分与可行性验证(judge / verifier)**:WorldGym(VLM-as-reward)、IRL-VLA、NORA-1.5(偏好奖励)、SRPO(latent 作稠密 progress metric)、AtomVLA(给 action chunk 按分解子目标打分)、World Action Verifier(评估预测转移是否 plausible & reachable)。

**生命周期(Fig. 5)** 把同一预测能力按"何时进入流水线"再切一刀:**Pretraining=Prior**(吸收动力学/几何/本体/动作后果的通用规律,如 V-JEPA 2、Seer、Genie Envisioner、TesserAct、PointWorld);**Post-training=Refine**(合成数据/学习模拟器/奖励偏好/幻觉过滤改进已有策略,如 DreamGen、World4RL、WMPO);**Inference=Adapt**(在线搜索、候选重排、test-time training、记忆更新、progress 预测、自我纠错,如 VLA-Reasoner、MinD、AdaPower、Self-Correcting VLA)。同一架构在三阶段的**失效模式随用途改变**——所以不存在单一评测指标。

### 5. 评测度量的规范化(三层)

作者把评测显式分三层,并给出统一公式。挑几条代表:

$$\mathrm{PSNR} = 10 \log_{10} \frac{\mathrm{MAX}^2}{\mathrm{MSE}}$$

用大白话说(直接保真度):像素级重建质量,越高越好——但它只测"像不像",奖励模糊平均、看不出接触/力/物体永久性对不对。

$$\mathrm{LC} = D_{\mathrm{KL}}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

用大白话说(latent 一致性):预测 latent 分布 $P$ 与目标先验/后验 $Q$ 的散度,越低说明 latent rollout 越规整;这是 latent 世界模型绕开"没有可解释图像"时的替代保真度。

$$\mathrm{SR} = \frac{\text{successful episodes}}{\text{total episodes}}, \qquad \mathrm{SE} = \inf\{\, N \mid \mathrm{SR}_N \ge \tau \,\}$$

用大白话说(下游任务):成功率与样本效率——世界模型真正的价值证据是"用了它策略更好/更省数据",而不是 rollout 更真实。

$$\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2-1)}, \qquad \mathrm{HR} = \frac{\text{infeasible transitions}}{\text{total transitions}}, \qquad \mathcal{D} = \frac{1}{T}\sum_{t=1}^{T} \lVert \mathbf{s}_t - \hat{\mathbf{s}}_t \rVert_2^2$$

用大白话说(模拟器可靠性):$\rho$ 是"模拟里的策略排名与真机排名的 Spearman 相关"(高才说明这个学到的环境是可信的策略选择代理);$\mathrm{HR}$ 是幻觉率(生成转移里违反物理/任务约束的比例,如物体穿模、瞬移、消失);$\mathcal{D}$ 是 rollout 漂移(想象状态多快偏离真实状态)。**作者反复强调:三层没有一层单独够用**——被当学习环境用的世界模型需要 $\rho$/$\mathrm{HR}$ 这类可靠性检验,而不是 PSNR。

## 三、实验结果

作为综述,其"结果"是对领域的**结构化覆盖与量化梳理**:34 个方法(Table I)按功能角色 × 表征 × 预测信号 × 生命周期阶段归类;34 个数据集(Table II)按功能角色分六组;三层评测(Table III)。

**方法在生命周期上的分布(Table I 摘要,共 34 项)**:

| 功能角色 | 阶段主导 | 代表方法(节选) |
|---|---|---|
| Predictive prior | Pre / Multi | V-JEPA 2(latent), Genie Envisioner(video), PointWorld(3D), TesserAct(4D) |
| Integrated | Train | DreamerV3, Seer, GR-1/GR-2, WorldVLA, RynnVLA-002, DUST, CoT-VLA, DreamVLA, VLA-JEPA, DIAL, FlowVLA, 3D-VLA |
| Planner | Infer | PlaNet, UniPi, SuSIE, PIN-WM, MinD |
| Infrastructure | Post / Infer | DreamGen, GigaWorld-0, WristWorld, World4RL, WorldGym, World-Env, WMPO, NORA-1.5, SRPO, AtomVLA, WoVR, GPC, WorldPlanner, VLA-Reasoner, World Action Verifier |

**数据集景观(Table II 摘要,六组共 34 个,均为原文标注真实数字)**:

| 组别 | 代表数据集(年份 / 规模 / 任务 / 本体) |
|---|---|
| A 视频预测先驱 | BAIR pushing(2017 / 44K traj / 1 Sawyer);RoboNet(2019 / ~162K traj / 7 robots) |
| B 任务中心仿真 benchmark | Meta-World(2019 / 50 tasks);RLBench(2020 / 100);CALVIN(2022 / 34);LIBERO(2023 / 130);ManiSkill3(2025 / 20+ robots);RoboTwin 2.0(2025 / 100K+ traj / 50 tasks / 5 dual-arm);RoboCasa365(2026 / ~655K real+2227h / 300 pretrain tasks) |
| C 模仿/策略学习 benchmark | RoboTurk(2018 / 2.2K demos);Robomimic(2022 / ~2.8K demos+MG);Language-Table(2022 / ~600K);FurnitureBench(2023);RoboSet MT-ACT(2023 / 7.5K) |
| D 大规模真机预训练语料 | RT-1(2022 / ~130K ep / 700+ instr);RH20T(2023 / 110K seq / 147);BridgeData V2(2023 / ~60K);Open X-Embodiment(2023 / 1M+ traj / 22 本体);DROID(2024 / 76K);AgiBot World(2025 / 1M+ / 217);Open-H-Embodiment(2026 / 124K ep / 770h / 20) |
| E 多模态与接触丰富 | Robo360(2023 / 2K traj);ManiWAV(2024 / 音视觉);REASSEMBLE(2025 / 4.6K demos / 事件相机+力矩+音频) |
| F 自主数据范式 | MimicGen(2023 / 50K 生成 demos);PlayWorld(2026 / 30h 自主 play) |

**三层评测(Table III)**:直接保真度(PSNR/SSIM/LPIPS/LC)、下游任务性能(SR/SE/ACS)、模拟器可靠性($\rho$/HR/$\mathcal{D}$),各配公式与解读。作者点出**"缺失的保真度指标是 action alignment"**——应该比较预测与实际的物体位移、接触起始、affordance 变化、逆动力学/控制器下的可执行性,而这类指标目前远不如图像相似度标准化。

## 四、局限性

- **综述本身没有实证再评测**:所有数字来自被引论文原文,作者未在统一协议下复现比较各方法(这在快速演化领域几乎不可避免,但意味着表格的"role/stage"归类是作者的主观映射,存在争议空间)。
- **快照易过时**:领域每月刷新,截稿到 2026 年中的 34 方法/34 数据集只是切片;许多 2026 引用(如 Open-H-Embodiment、PlayWorld、World-VLA-Loop、WoVR)本身还很新、结论未沉淀。
- **四条轴并非完全正交**:同一方法(如 GR-2)同时是"视频表征 + 联合世界-动作 + train 阶段",作者也承认"categories are not mutually exclusive",表格只报"最被强调"的角色,对读者定位单个方法的完整定位有损。
- **评测部分给了公式却未给横向数据**:三层度量是规范化倡议,但没有一张"各方法在统一 benchmark 上的 SR/ρ/HR"对照表——恰恰是作者自己指出的领域缺口(metric fragmentation),综述能指出却无法弥补。

## 五、评价与展望

**优点。** (1) 定义层面的贡献扎实:用"功能而非架构"来划界,干净地把 world model 与 inverse model / policy / reward-value / 经典模拟器区分开,这条 function-based 定义比多数综述"按感知-预测-控制列清单"更有解释力,能统一 RL/IL/视频/3D/物理/VLA 六个原本割裂的传统。(2) 三轴(what/how/when)+ 基础设施五角色 + 生命周期三阶段的框架,确实抓住了操作(而非驾驶/通用视频)特有的张力:**预测保真 ≠ 动作效用**,并把它贯穿到评测三层。(3) 覆盖面广且时新,把 GR 系、Dreamer 系、V-JEPA 2、SuSIE、UniPi、PIN-WM、DreamGen、WorldGym 等散落在不同会议的工作放进同一坐标系,对入门者是好地图。

**缺点与开放问题(纯学术视角)。** (1) 与已有综述的差异主要在"组织视角"而非"新证据",框架价值高但不可证伪,读者难以判断某方法归类是否恰当。(2) 最有价值的论断——"闭环下评测是核心未解问题"——只停在倡议层面(呼吁 held-out 物理测试场景、对抗扰动、infeasible-transition 检查、rollout-depth 压力测试、把 policy+world-model 当耦合系统评测),没有给出可落地的统一 benchmark 或 leaderboard,而这正是领域最缺的。(3) 对 action alignment 度量只指出缺失、未提出具体形式(如"预测物体位移与真实位移的 chamfer / 接触起始时刻的时间对齐误差"这类可操作定义),留白较大。(4) 物理知情一族(仅 PIN-WM 一个代表)与 flow 一族(FLIP/FlowVLA)覆盖偏薄,相对视频/latent 系的展开不成比例,反映当前领域重心,但也弱化了"物理约束"作为独立设计维度的论证。

**与其他公开工作的关系。** 本文明确对标并区分于:通用世界模型/驾驶综述、具身智能物理模拟器综述、3D/4D 建模综述、以及只盯 VLA agents 的综述——定位为"以操作为中心、连接六大传统"的独家生态位。其核心张力(fidelity ≠ utility、simulator exploitation、hallucination control)与 model-based RL 里 Dreamer 系的经典教训一脉相承,又把它推广到 VLA 的联合世界-动作建模。**未来最值得推进的方向**:统一的 action-conditioned 校准 benchmark、把"想象的未来是否物理可达"变成可训练的显式约束/verifier、以及在真机上做"policy+world model 作为耦合系统"的闭环评测协议。

## 参考

1. Ha & Schmidhuber, *Recurrent World Models Facilitate Policy Evolution*, NeurIPS 2018.（world model 概念在深度生成设定下的普及,本文的定义起点）
2. Hafner et al., *Dream to Control (Dreamer)*, ICLR 2020 / *DreamerV3*, Nature 2025.（latent 想象驱动策略优化,想象式一族的范式基石）
3. Assran et al., *V-JEPA 2: Self-supervised video models enable understanding, prediction and planning*, 2025.（预测式 latent 学习的 predictive prior 代表）
4. Black et al., *SuSIE: Zero-shot robotic manipulation with pretrained image-editing diffusion*, ICLR 2024.（子目标图像式显式规划器代表)
5. Bharadhwaj et al. / NVIDIA, *DreamGen*, 2025.（视频世界模型作合成数据引擎的代表)
