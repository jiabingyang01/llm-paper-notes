# Genie Sim 3.0：面向人形机器人的高保真综合仿真平台

> **论文**：*Genie Sim 3.0: A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot*
>
> **作者**：Chenghao Yin, Da Huang, Di Yang, Jichao Wang, Nanshu Zhao, Chen Xu（共同一作）+ et al.（通讯作者 Qian Wang, Maoqing Yao）
>
> **机构**：AgiBot（智元机器人，代码仓库 org 为 AgibotTech）
>
> **发布时间**：2026 年 01 月（arXiv 2601.02078，v3 更新于 2026 年 06 月）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.02078) | [PDF](https://arxiv.org/pdf/2601.02078)
>
> **分类标签**：`仿真平台` `合成数据` `3DGS重建` `LLM场景生成` `Sim2Real`

---

## 一句话总结

Genie Sim 3.0 把"LLM 对话生成场景 + 3DGS/扩散重建高保真环境 + 遥操作与自动双通道采数 + LLM-VLM 自动出题打分"串成一条全周期机器人操作仿真流水线,开源 5,140 个资产、200 个任务超 10,000 小时合成数据与 100,000+ 评测场景;核心实证是:π0.5 用 1500 条合成轨迹微调即可在真机零样本上**全面超过** 500 条真机数据训练的基线(四任务真机成功率如 Recognize Size 0.94 vs 0.75),且仿真-真机成功率线性相关 $R^2=0.924$。

## 一、问题与动机

VLA 模型的进步高度依赖大规模、高质量、多样化的交互数据与可复现的评测基准,而真机采数成本高、难扩展,真机评测又费时、主观、不可复现。作者指出现有仿真方案有三个瓶颈:

1. **构建高保真环境需要大量人工**（3D 建模与物理仿真专家),限制了训练场景的规模与多样性。
2. **自动/程序化生成缺乏细粒度可控性**:既要足够多样以支撑泛化,又要能精确复现、系统性地变动单一变量以做消融——二者存在张力,导致难以调试失败、做受控实验。
3. **评测依赖固定的手工指标**（如成功率),无法刻画任务完成质量;人在环评测低效、主观、不可扩展。

Genie Sim 3.0 的目标是用一个统一平台同时解决环境重建、场景泛化、数据采集与自动评测,并首次把 LLM 引入到"自动出评测题 + VLM 自动判分"这一环。

## 二、核心方法

平台分五个模块:场景生成、评测生成、环境重建、数据生成、闭环评测。

### 2.1 场景生成:Genie Sim Generator（LLM 工具链）

由两个紧耦合模块组成——**Assets Index**（LLM/VLM/RAG 增强的 Isaac Sim 就绪资产库,提供结构化 API)与 **Scene Generator**（受 The Scene Language [37] 启发,通过多轮对话捕捉用户意图,翻译成可执行 Python,再编译成 Isaac Sim 的 scene graph)。整条流水线四阶段:意图解析 → 资产检索 → DSL 代码生成 → 结果组装。

- **Intention Interpreter**：CoT 提示的 LLM 把开放自然语言解析成结构化任务请求,输出为 JSON schema,含语义物体类别、几何约束(尺寸/颜色/形状)与成对空间关系("on/adjacent/aligned");含糊指令用预训练世界知识做推理消解,约束冲突则回到用户澄清。

- **Assets Index（RAG 检索)**：对全部 5,140 个物体抽取外观/几何/用途语义描述,用 QWEN text-embedding-v4 编码成 2048 维向量存入 ChromaDB。运行时把场景描述里的关键词(如 "yellow cube")编码到同一空间,按余弦相似度取 top-k:

$$
\mathrm{sim}(q, a_i) = \frac{q \cdot a_i}{\lVert q \rVert \, \lVert a_i \rVert}, \qquad \mathcal{R} = \operatorname{top\text{-}k}_i \; \mathrm{sim}(q, a_i)
$$

  召回结果连同元数据(USD 路径、碰撞外壳、质量属性、纹理变体)注入 LLM 上下文,保证后续代码只引用已验证可用的资产;整个检索通常在 200 ms 内完成。

  **用大白话说**：把每个 3D 资产写成一句"长什么样、能干嘛"的说明并变成一串数字指纹;你说"来个黄方块",系统就找指纹最像的几个货架物件塞给 LLM,后者只能从这批真实存在的货里挑,不会凭空捏造资产。

- **DSL Code Generator + Results Assembler**：预训练 LLM 依据 scene language 的语法生成场景规格(双精度浮点、可迭代编辑),再由组装器实例化。为引入随机化,DSL 程序内嵌随机函数扰动物体位姿、布局模式与物体选择。最终生成一个层级化 **Scene Graph** $G = \langle V, E \rangle$:节点 $V$ 编码 asset id、语义、尺寸、位姿、任务标签,边 $E$ 编码空间关系(on/in/adjacent/aligned/stacked),经 OpenUSD Schema 与 Isaac Sim API 合成可仿真的 USD 文件。不同于以往只保留任务相关物体的做法,系统保留场景完整性(货架、储物架等复杂布局也支持)。

  **用大白话说**：把一个场景写成"哪些东西(点)+它们谁在谁上面(边)"的关系图,再自动翻译成仿真器认识的文件;每次实例化时随机挪一挪位置、换一换物体,就能几分钟内从一个模板衍生出成千上万个不重样的场景。

### 2.2 评测生成:LLM 出题 + VLM 判分

用 **Action Domain Rule（ADER)** 系统结合 LLM,针对给定场景自动批量生成合理指令与可执行的评测配置文件,克服传统基准"指令空间单一、扩展成本高"的问题。执行时把形式化任务规格与任务执行过程中记录的时序视觉观测一并交给 VLM,由 VLM 判断任务要求是否满足并给出基于证据的判分理由(见原文 Fig.3 的清桌面例子)。这是作者主张的"首个把 LLM 用于自动评测"的基准贡献。

### 2.3 环境重建:3DGS + 扩散补视角

用 3D Gaussian Splatting（3DGS [38])做真实感渲染与表面重建。采集端用 SkylandX 的 MetaCam 手持三维激光扫描仪(鱼眼图 + 每帧位姿 + 稠密点云)。因为 3DGS 对相机位姿精度极敏感,作者在相机位姿优化模块里用 **SuperPoint [39] + LightGlue [40]** 替换 COLMAP-PCD 里的 DSP-SIFT 特征;再用 LiDAR SLAM 先验位姿做三角化、联合 BA 优化;用 gsplat [43] 训练 3DGS。针对大场景视角覆盖不足,用 **Difix3D+ [44]** 扩散模型渲染外插视角补足高质量新视图,再结合 LiDAR 点云用 **PGSR [45]** 做表面重建得到高精度网格。

### 2.4 数据生成:遥操作 + 全自动双通道

- **Teleoperation**：PICO VR 头显作主输入,采集末端执行器目标动作信号送中央主机,由基准模块处理、运动控制器在仿真里执行规划轨迹;含真实物理(碰撞/摩擦),整条交互序列(关节状态、视觉观测、物体位姿)全程记录。用于复杂长时程任务的高质量拟人示范。

- **Automated Collection**：以 GPU 加速运动规划器 **cuRobo [46]** 为核心。任务生成靠 LLM 资产检索 + 预定义原子技能;数据采集阶段,候选关键 waypoint 取自 GraspNet [47] 标注的抓取位姿,按运动学可达性、避碰、拟人可行性打分;每个动作生成多条候选序列,逐条在仿真里执行并由专门的轨迹评估模块打分,失败则**状态回滚** 再试下一条候选,并对物体网格做简化以提效。

  **用大白话说**：自动采数就像"多路预案 + 打分 + 失败重来":每一步都准备好几条抓取/移动方案,谁在仿真里跑得通就留谁,跑不通就把场景状态倒回去换下一条,靠这套过滤和重试机制把自动采数的成功率顶上去。

### 2.5 闭环评测

仿真与模型推理服务解耦,经 HTTP 通信:仿真器发观测图像与本体感知状态,模型返回控制指令并在仿真里执行,周期性判定任务完成,超时则终止。支持接入 π0.5、GO-1、GR00T、UniVLA、RDT、X-VLA 等主流 VLA;支持多机型(Genie G1/G2)、多末端(omnipicker、omnihands、INSPIRE skillhands、zhixing gripper)、本地/分布式推理与自动多维评测。

数据集沿三个正交轴组织(原文 Fig.5):**操作技能**（pick/place/pull/push/open/close 等原子动作)、**认知理解**（空间推理/属性理解/逻辑推断/常识推理)、**任务复杂度**（规划时程、协同控制)。合成集共 200 个代表性任务、超 10,000 小时,用 Agibot G1/G2 双平台生成,并在任务布局、初始机器人位姿、光照、场景配置、相机噪声、指令措辞等维度做系统性变化。

## 三、实验结果

五个闭环评测套件:GenieSim-Sim2Real / Instruction / Robust / Manipulation / Spatial。Sim2Real 套件全部用 **π0.5** 作基座策略,真机端用 Agibot G1,每个配置在真机与仿真各跑 50 次试验。

### 3.1 合成数据 scaling（Table I,π0.5,单元格为 仿真 / 真机 成功率）

| 训练配置 | Select Color | Recognize Size | Grasp Targets | Organize Objects |
|---|---|---|---|---|
| 200 条真机 | 0.45 / 0.53 | 0.50 / 0.56 | 0.34 / 0.39 | 0.25 / 0.30 |
| 500 条真机 | 0.75 / 0.73 | 0.75 / 0.75 | 0.54 / 0.58 | 0.45 / 0.40 |
| 500 条合成 | 0.53 / 0.60 | 0.50 / 0.63 | 0.29 / 0.33 | 0.39 / 0.35 |
| **1500 条合成** | **0.86 / 0.85** | **0.93 / 0.94** | **0.72 / 0.71** | **0.52 / 0.60** |

结论:成功率随数据量单调上升(符合 scaling law);**同等 500 条规模下真机数据优于合成**（物理保真度更高);但把合成扩到 1500 条后,四个任务的真机成功率**全面反超所有真机基线**,说明系统性 domain randomization(纹理/光照/物理参数/任务变体)在规模上有效弥合了 sim-to-real gap。仿真与真机成功率的定量相关分析得 $R^2 = 0.924$、斜率约 $1.045$,即仿真榜单能可靠预测真机趋势。

### 3.2 Sim-to-Real 迁移（Table II,π0.5,8 任务,合成 500~1500 条 vs 真机 500 条)

| 任务 | 仿真环境·合成(s2s) | 仿真环境·真机(r2s) | 真机环境·合成(s2r) | 真机环境·真机(r2r) |
|---|---|---|---|---|
| Select Color | 0.86 | 0.75 | 0.85 | 0.73 |
| Recognize Size | 0.93 | 0.75 | 0.94 | 0.75 |
| Grasp Targets | 0.72 | 0.54 | 0.71 | 0.58 |
| Organize Items | 0.48 | 0.45 | 0.60 | 0.40 |
| Pack in Supermarket | 0.94 | 1.00 | 0.95 | 0.95 |
| Sort Fruit | 0.90 | 0.90 | 1.00 | 1.00 |
| Place Block into Drawer | 0.80 | 0.90 | 0.85 | 0.90 |
| Bimanual Chip Handover | 0.80 | 0.70 | 0.73 | 0.71 |
| **平均** | **0.80** | 0.75 | **0.83** | 0.75 |

双向可迁移:纯合成训练的真机平均 0.83 优于纯真机的 0.75(sim-to-real 成立);反过来真机训练在仿真里也拿到 0.75(real-to-sim 成立),说明仿真环境忠实刻画了真实的物理结构与任务动态,可作可靠评测台。

### 3.3 四套件横向对比（各套件平均分)

| 套件（任务数) | π0.5 | ACoT-VLA | GR00T-N1.7 | π0 |
|---|---|---|---|---|
| Instruction（10,语言条件) | 0.72 | **0.73** | 0.61 | 0.35 |
| Robust（5 类扰动) | **0.497** | 0.493 | 0.443 | 0.231 |
| Manipulation（10,复杂操作) | **0.58** | 0.48 | 0.44 | 0.35 |
| Spatial（8,空间推理) | 0.30 | **0.36** | 0.25 | 0.04 |

π0.5 与 ACoT-VLA 是两个最强策略,呈互补:π0.5 在 Robust、Manipulation 领先,ACoT-VLA 在 Instruction、Spatial 领先;GR00T-N1.7 稳居第三;π0 因缺强语言/视觉骨干全线垫底(如常识题 pick_common_sense 仅 0.06)。

**Robust 扰动细节(Table IV,Δ 为相对无扰动 Reference 的变化):** Reference 分 π0.5=0.720 / ACoT-VLA=0.730 / GR00T-N1.7=0.610 / π0=0.350。外观与语言级扰动(Background、Instruction)几乎免费(各模型最多掉约 0.05),而几何与本体级扰动主导退化:仅 Robot Pose 与 Camera Position 就从两个领先模型各扣掉 0.40~0.47,把分数压到 0.26~0.31 区间。π0 名义退化最小(平均 Δ=−0.119),但这是"地板效应"——它 Reference 本就低(0.350),扰动后绝对分(0.231)仍最差;π0.5/ACoT-VLA 吸收了更大名义跌幅却仍保持最高绝对分,说明"干净任务能力强"才是可部署鲁棒性的决定因素。长时程任务(如 Sorting Packages Continuous、Clean the Desktop)四个模型几乎全线失败(0.00~0.10),是当前能力边界。

## 四、局限性

- **文中已承认的差距**:合成与真机仍存在接触动力学、随机扰动上的失配,只是整体趋势一致;斜率 $\approx1.045$ 说明相关强但非完美对齐(且原文正文 $R^2=0.924$ 与 Fig.7 图例标注 $R^2=0.94$、斜率 1.025 略有出入)。
- **保真度上限来自采集硬件**:高保真环境依赖 MetaCam 激光扫描 + 3DGS/扩散补视角,重建成本与覆盖度受采集流程限制,3DGS 对位姿误差极敏感需要专门的 SuperPoint/LightGlue + LiDAR-BA 补救。
- **评测依赖 VLM 判分**:LLM 出题 + VLM 判分虽可扩展,但判分本身的可靠性、系统性偏差与幻觉未做独立量化(未给 VLM 判分与人工判分的一致性数字)。
- **实验只在自家数据/机型上闭环**:Sim2Real 的所有强证据都基于 π0.5 + Agibot G1/G2,缺少在第三方公开机型/公开真机基准上的交叉复现;每配置 50 次试验样本量偏小。
- **长时程与灵巧操作仍是硬骨头**:连续分拣、清桌面等任务几乎全 0,说明平台"能出题"但当前 SOTA 策略"答不上来",这既是数据集的判别价值也是覆盖盲区。

## 五、评价与展望

**优点**:(1)工程完整度高——把场景生成、重建、采数、评测四段真正打通成"全周期"闭环,并开源了资产/数据/评测/代码,规模(5,140 资产 / 10,000+ 小时 / 100,000+ 场景)在同类平台里可观。(2)最有分量的科学结论是 scaling 曲线 + 双向迁移 + $R^2=0.924$ 的仿真-真机相关性,给"用合成数据替代真机采数"提供了较硬的定量支撑,而非仅演示性截图。(3)"LLM 出题 + VLM 判分"把评测从固定成功率推向可解释、可扩展的多维判分,方向前瞻。

**与公开工作的关系**:场景生成直接构建在 The Scene Language [37] 之上,理念上与 Gen2Sim [8]、RoboVerse [15] 一脉相承(LLM/生成模型驱动的仿真扩展);数据规模与双臂任务对标 RoboTwin 2.0 [23] 与 Agibot World [20],但强调更强的 domain randomization 与更高渲染保真;3DGS 重建路线(gsplat+Difix3D++PGSR)是把 CV 侧最新 real2sim 组件工程化集成。相比 RoboCasa/BEHAVIOR-1K 等偏程序化或偏家居的基准,本文卖点是"自然语言可控 + 高保真 + 自动评测"三合一。

**开放问题与可改进方向**:(a)VLM 判分需要与人工判分做一致性/校准报告,否则"自动评测可信"缺闭环证据;(b)$R^2$ 相关性只在 4~8 个较简单任务上成立,长时程/灵巧/接触密集任务上的 sim2real 可预测性尚未验证;(c)domain randomization 目前是启发式随机,能否学一个"最优随机分布"(如对抗式或基于真机反馈的自适应 randomization)以更省合成数据达到同等真机分,是自然的下一步;(d)平台对第三方开放机型与公开真机基准的可复现性,决定了它能否成为社区通用底座而非单厂内测台。

## 参考

1. Yunzhi Zhang et al. *The Scene Language: Representing Scenes with Programs, Words, and Embeddings.* arXiv:2410.16770, 2025.（场景生成的语言表示基础)
2. Tianxing Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization.* arXiv:2506.18088, 2025.（最直接对标的双臂合成数据基准)
3. Qingwen Bu et al. *AgiBot World Colosseo: A Large-scale Manipulation Platform.* IROS, 2025.（同源大规模真机操作平台/数据)
4. Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization.* arXiv:2504.16054, 2025.（Sim2Real 全部实验的基座策略)
5. Pushkal Katara, Zhou Xian, Katerina Fragkiadaki. *Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models.* ICRA, 2024.（生成模型驱动仿真扩展的先驱)
