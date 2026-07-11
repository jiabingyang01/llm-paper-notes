# SceneSmith：面向仿真就绪室内场景的智能体式生成

> **论文**：*SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes*
>
> **作者**：Nicholas Pfaff, Thomas Cohn, Sergey Zakharov, Rick Cory, Russ Tedrake
>
> **机构**：Massachusetts Institute of Technology；Toyota Research Institute
>
> **发布时间**：2026 年 02 月（arXiv 2602.09153）
>
> **发表状态**：ICML 2026（PMLR Vol. 306，正式录用）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.09153) | [PDF](https://arxiv.org/pdf/2602.09153)
>
> **分类标签**：`室内场景生成` `Agentic AI` `具身仿真` `Text-to-3D资产生成` `物理仿真就绪`

---

## 一句话总结

SceneSmith 用 designer-critic-orchestrator 三智能体在 layout→furniture→wall/ceiling→manipuland 五级层级树上迭代构建场景,并把 text-to-3D 按需生成、铰接资产检索、物理属性估计与碰撞/稳定性后处理直接打进流水线,使生成场景在物体密度(71.1 vs 基线 11-23 个/房间)、碰撞率(1.2% vs 3-29%)、静态稳定性(95.6% vs 8-61%)上全面超越现有方法,205 人用户研究中真实感/提示保真度平均胜率达 92.2%/91.5%。

## 一、问题与动机

现有仿真室内环境普遍稀疏、物体多样性有限、且多为静态场景,与真实家庭中密集杂乱、带铰接家具、可物理交互的实际分布严重不匹配,而这种差距正是机器人操作策略训练/评测的核心瓶颈。既有工作要么是"资产中心"(只做单个物体的重建/合成,如 Pfaff et al., 2025a/b),要么是"场景中心"(在固定资产库上做布局/摆放,如 Holodeck、HSM),两者割裂;此外多数场景合成方法(程序化/数据驱动/LLM-based)主要优化家具级布局与视觉真实感,把小物体、铰接资产、物理属性视为次要,这与机器人操作所需的密集可操作物体排布、层级支撑关系、物理有效构型严重不对齐。SceneSmith 的目标是从自然语言 prompt 出发,联合生成资产与场景,端到端产出可直接用于仿真的室内环境。

## 二、核心方法

**场景表示与层级构建**。场景 $\mathcal{S}=\{\mathcal{R}_j\}$ 由房间 $\mathcal{R}_j=(\mathcal{G}_j,\mathcal{O}_j)$ 组成,$\mathcal{G}_j$ 为墙/门/窗等建筑几何,$\mathcal{O}_j=\{(\mathcal{A}_i,\mathcal{X}_i)\}$ 为(资产,$SE(3)$ 位姿)对。构建过程组织成一棵树:根阶段处理全局 prompt $\mathcal{T}$ 生成 $M$ 个房间的建筑布局;每个房间用派生的房间级 prompt $\mathcal{T}_j$ 依次跑家具、墙面物体、天花板物体三个阶段;被选中的支撑实体(家具台面、墙架、地面区域)再各自派生实体级 prompt $\mathcal{T}_{j,k}$,触发可操作小物体(manipuland)分支。这种层级化 prompt 细化让局部决策独立进行同时保持与场景整体意图一致。

**Designer-Critic-Orchestrator 三元组**。每个阶段都是三个 VLM 智能体的交互:designer 通过结构化工具(状态观察/视觉观察/场景修改/资产获取/可行性验证)提出修改;critic 只做评估,给出 0-10 分类评分和自然语言反馈;orchestrator 管理迭代循环,维护 checkpoint,当评分下降时回滚到上一版本并让 designer 尝试新策略,并根据分数阈值/迭代上限决定终止。相比单智能体规划(SceneWeaver 的 reason-act-reflect)或双角色资产级精修(LL3M),SceneSmith 把这一模式扩展到场景级,并允许每轮无限次工具调用、支持 handoff 前的自我校验。摆放专用工具还包括吸附对齐(椅子贴桌、柜子贴墙)和朝向查询(判断物体是否面向另一物体/建筑元素)。

**资产生成与路由**。资产路由器把复合请求拆解为可独立操作的原子资产(如"水果碗"拆为碗+若干水果,便于机器人分别抓取),并为每个资产选择获取策略:(1)**生成**——标准静态物体走 text-to-image(GPT Image 1.5)→ SAM3 前景分割 → SAM3D 单图重建纹理网格的 text-to-3D 流水线,按需生成避免了训练数据污染;(2)**检索**——带活动部件的铰接物体(柜子、抽屉、家电)从 ArtVIP 库中做两阶段 CLIP(ViT-H-14)检索(按类型过滤→相似度排序→按目标尺寸 L1 距离重排),因为作者发现当前 text-to-3D 尚不能可靠生成适合机器人仿真的铰接结构与运动学;(3)**薄覆盖物**——地毯/桌布/海报等扁平装饰件用程序化几何+PBR 材质(优先从 ambientCG 检索,检索失败退回 AI 生成纹理)。所有资产经 VLM 校验(物体类型、风格一致性、单物体、完整性、比例合理、门/抽屉须闭合)才能入库,失败则在预算内重生成或换路由。

**物理仿真就绪**。VLM 读取 6 个多视角渲染(顶/底+4 个 30°侧视图)估计主导材质类别(19 类之一,决定摩擦系数)、带置信区间的质量估计 $[m_{\min}, m_{\max}]$(供域随机化用)、以及规范朝向(区分家具的"人面朝向"与墙饰的"装饰面")。假设均匀密度计算惯性张量：

$$\rho = \frac{m}{V_{\text{mesh}}}, \quad I_B = \rho \cdot I_{\text{unit}}$$

大白话说:先用估计质量除以网格体积得密度,再乘单位密度惯性张量,把纯几何的"形状惯性"换算成物理真实的"质量惯性",铰接物体则按连杆分别估计再约束求和等于总质量。碰撞几何用 V-HACD 做凸分解(家具 128 片、墙面/manipuland 64 片、天花板 16 片),相比 CoACD 更快(1-2 个数量级)且不会出现物体堆叠时因几何膨胀而"悬浮"的问题。生成/摆放之后不保证物体严格无穿透或静态稳定,因此引入轻量后处理:先用非线性优化把物体位置投影到最近的无碰撞构型(保持朝向不变),再把整个场景丢进 Drake 物理引擎做重力仿真,让不稳定物体自然沉降到静态平衡构型。

**布局生成算法**。用最佳优先回溯搜索沿已放置房间的边缘采样候选位置(每条边 11 个候选,含原始与旋转 90° 两种朝向),局部打分靠邻接奖励与形心距离惩罚引导搜索方向,完整布局则用两个全局指标评估:紧凑度 $S_{\text{compact}}=A_{\text{rooms}}/A_{\text{bbox}}$(空间利用率)和稳定度 $S_{\text{stable}}=\sum_r \exp(-\lVert \mathbf{p}_r-\mathbf{p}_r^{\text{prev}}\rVert_2/2)$(编辑前后房间中心尽量不动)。大白话说:紧凑度鼓励房间填满外包框、少留死角;稳定度让"改窄一条走廊"这类局部编辑只牵动必要的相邻房间微调,而不是整栋楼推倒重来,从而支持增量式迭代精修。

**应用:机器人策略评测闭环**。给定自然语言任务描述,先用 LLM 生成多样场景 prompt,再由 SceneSmith 批量生成对应场景,策略在其中执行(演示用基于模型的 pick-and-place 策略),最后由一个 evaluator agent 联合查询物体位姿的符号状态与渲染视觉观测来判定任务是否完成——不依赖手工设计的成功判据,代价是牺牲了确定性。

## 三、关键结果

评测覆盖 210 个房间/房屋级 prompt(五类:SceneEval-100、房型多样性、物体密度、主题场景、房屋级多房间),对比 5 个外部基线(HSM、Holodeck、SceneWeaver、I-Design、LayoutVLM 的 curated/Objaverse 两版)。

**用户研究**(205 名众包参与者,3051 次有效两两比较,均 $p<0.001$):

| 对比对象 | 真实感胜率 | 保真度胜率 |
|---|---|---|
| vs HSM | 88.5% | 85.2% |
| vs Holodeck | 88.6% | 90.6% |
| vs SceneWeaver | 91.7% | 92.9% |
| vs I-Design | 94.1% | 90.6% |
| vs LayoutVLM (Curated) | 94.9% | 95.8% |
| vs LayoutVLM (Objaverse) | 95.4% | 93.9% |
| 房间级平均 | **92.2%** | **91.5%** |
| 房屋级 vs Holodeck(唯一多房间基线) | 80.3% | 84.7% |

**自动化指标**(SceneEval 系列 + 物理指标,房间级,均值 $\pm$ 95% CI):SceneSmith 物体数 71.1$\pm$13.0(基线 11.2-23.0),OOR(物体-物体关系)比最强基线提升 2.2x,碰撞率 COL 1.2%$\pm$0.6(基线 3.1%-25.9%,平均剩余穿透深度仅 3.8mm、比基线浅 3-12x),静态稳定 STB 95.6%$\pm$1.7(基线 8.1%-60.8%)。房屋级:SceneSmith 214.1$\pm$60.9 个物体 vs Holodeck 81.3$\pm$18.3(2.6x),碰撞 0.9% vs 3.8%,稳定 79.8% vs 17.9%。代价是可达性 ACC 与可导航性 NAV 略低于部分基线,论文认为这是密度大幅提高、可用空间随之减少的自然结果。

**消融**(相对完整版 SceneSmith 的胜率):NotGenerated(改用 HSSD 固定资产库而非生成)63.8%/67.0%,NoAssetValidation(去资产校验)63.0%/62.2%,NoObserveScene(去视觉观测)61.5%/53.2%,三者均显著,说明生成式资产获取、资产校验、视觉反馈都是关键贡献点;而 NoCritic、NoSpecializedTools、NoAgentMemory 影响较小(51-55%,未达显著性),其中 NoCritic 在保真度上甚至和完整版几乎打平,但成本降低约 70%、物体数减少约 24%——提示 critic 循环在"质量-成本"权衡上并非严格必要。

**机器人策略评测**:在 100 个生成场景 x 4 类 pick-and-place 任务上,标准策略成功率 16% vs 人为退化(运动约束放松)策略 12%,验证了该评测闭环能区分策略优劣;300 次(100 场景 x 3 状态)evaluator 判定与人工标注一致率 99.7%,唯一分歧是水果落在盘子边缘的模糊案例。此外用 RB-Y1 人形机器人做了柜子开关的遥操作演示,以及一个未见过 SceneSmith 数据训练的先验策略(Lin et al., 2026)在生成场景中零样本完成"把苹果从碗里放到砧板上"的定性展示。

## 四、评价与展望

**优点**:(1)首次把资产生成与场景生成端到端打通,摆脱了此前场景合成方法(Holodeck、HSM 等)依赖固定资产库的限制,实现真正开放词表的按需生成;(2)"仿真就绪"不是口号——碰撞几何、质量/惯性/摩擦全部落实,并用真实物理引擎(Drake)重力仿真做静态可行性校验,而非仅靠启发式检查;(3)评测扎实,大规模用户研究、自动化指标、系统性消融、下游机器人策略评测闭环一应俱全,且 evaluator 判定与人工标注做了一致性验证,说明这套"用 VLM agent 评测任务完成"的方法本身也具备一定可信度。

**局限与开放问题**:生成阶段本身并不强制满足物理约束,仍需依赖后处理(投影去穿透+重力沉降)才能达到高稳定性,残余碰撞虽浅但非零;铰接物体仍依赖固定资产库(ArtVIP)检索而非生成,论文明确指出这是当前 text-to-3D 技术的能力边界,一旦库外类别需求出现就会退化为"最接近匹配";物体密度大幅提升的同时可导航性/可达性指标下降,密集杂乱与机器人自由通行/抓取空间之间存在张力,如何联合优化二者是开放问题;整套系统重度依赖高推理强度的商用 VLM(GPT-5.2)做多轮多智能体调用,NoCritic 消融显示的"低成本几乎不掉点"现象也提示当前 critic 循环的设计可能存在优化空间;论文聚焦于用生成场景做策略**评测**,尚未展示用这类场景做大规模策略**训练**并验证 sim-to-real 迁移效果,这是自然的后续方向。

**与其他工作的关系**:方法论上继承 HSM(Pun et al., 2026)的层级分解与支撑面检测思路,并扩展出层级化 prompt 细化(全局 prompt 派生房间/实体级局部约束)与联合面级填充(同一房间内相关台面协同摆放,如"一层架子放书、另一层放绿植");智能体交互范式上相对 SceneWeaver 的单智能体 reason-act-reflect 和 LL3M 的资产级 designer-critic,把 critic-designer 分离的思路系统性搬到多阶段场景级构建,并加入 checkpoint 回滚机制以对抗迭代精修中的质量退化。

## 参考

- Pun, H. I. D. et al. HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation. 3DV, 2026.
- Yang, Y. et al. Holodeck: Language Guided Generation of 3D Embodied AI Environments. CVPR, 2024.
- Yang, Y. et al. SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent. NeurIPS, 2025.
- Lu, S. et al. LL3M: Large Language 3D Modelers. arXiv:2508.08228, 2025.
- Tam, H. I. I. et al. SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis. 2025.
