# InternVLA-M1：以空间引导为核心的通用机器人视觉-语言-动作框架

> **论文**：*InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy*
>
> **作者**：Yilun Chen、Ning Gao、Jiangmiao Pang、Bolun Wang、Fangjing Wang、Jinhui Ye、Junqiu Yu、Jinyu Zhang、Yangkun Zhu 等（核心贡献者，按姓氏字母序排列；全部 29 位贡献者含 Jiaya Jia、Yu Qiao、Bowen Zhou 等，同样按字母序列出）
>
> **机构**：上海人工智能实验室（Shanghai AI Laboratory）Intern Robotics 团队
>
> **发布时间**：2025 年 10 月（arXiv 2510.13778，页眉标注提交日期 2025-10-15）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.13778) | [PDF](https://arxiv.org/pdf/2510.13778)
>
> **分类标签**：`VLA` `空间定位预训练` `双系统架构` `Spatial Prompting` `长时程规划` `仿真数据引擎`

---

## 一句话总结

InternVLA-M1 用"空间定位预训练（回答'在哪做'）+ 空间引导的动作后训练（回答'怎么做')"两阶段配方，把 3M+ 多模态数据中 2.3M+ 的点/框/轨迹空间推理数据蒸馏进一个 4.1B 参数的双系统 VLA（Qwen2.5-VL-3B 规划器 + DiT 扩散动作头），在 SimplerEnv WidowX 上比不带空间引导的同架构基线高 17.0 个百分点，LIBERO 均值达 95.9，真实杂乱场景对未见物体/新配置提升 20.6%，长时程推理任务上超越现有工作 10% 以上。

## 一、问题与动机

大型多模态基础模型（VLM）已经在网络规模的视觉-语言语料上学到了强大的泛化能力，但要把这种能力延伸到物理世界，机器人不仅要理解指令"是什么意思"，还要确定"在哪里做"和"怎么做"。文本抽象只能间接地承载空间线索，而真实世界的动作需要连续的、具身的交互，这类数据在 VLM 训练语料中极度稀缺。远程操作数据集（如 DROID、Open X-Embodiment）能提供有价值的监督，但其规模和多样性相比大规模指令跟随语料仍然有限。

作者认为，**具身无关的空间先验**（object recognition、affordance grounding、visual trajectory reasoning、relative localization 等）是连接文本指令与具身动作指令的天然桥梁：这类先验一旦建立，具身相关的细节（机械臂关节、末端轨迹、人形运动、移动导航）可以留给下游适配去学习。

已有两条技术路线各有局限：
- **层级式机器人系统**（如 VoxPoser、ReKep、SayPlan 等）显式地用基础模型编码空间先验，但依赖规则化任务分解和人工设计的规划启发式，难以端到端地扩展到更复杂多样的任务。
- **数据驱动的端到端 VLA**（RT-2、OpenVLA、π0、Helix 等）直接在遥操作数据上学习控制，但往往过拟合细粒度运动行为，对涉及绝对/相对位置的高层语言指令欠拟合，未能把空间先验充分融入执行。

InternVLA-M1 的目标就是在一个统一的端到端框架里，既保留数据驱动 VLA 的可扩展性，又把显式的空间定位能力系统性地注入训练流程。

## 二、核心方法

### 2.1 双系统架构

InternVLA-M1 是一个双系统、端到端 VLA 框架：

- **System 2（VLM Planner）**：以 Qwen2.5-VL-3B-Instruct 为多模态编码器，负责捕捉空间先验、做子任务规划和潜在规划（latent planning）。
- **System 1（DiT Actor）**：采用 Diffusion Policy 风格的扩散动作专家（约 86M 参数）作为快速执行器，建立在 DINOv2 视觉编码器（21M）与一个轻量状态编码器（0.4M）之上，负责生成具身相关的电机指令。

全模型合计约 **4.1B** 参数，推理时单张 RTX 4090（约 12GB 显存）即可运行；配合 FlashAttention，VLM 部分推理速度约 **10 FPS**，动作执行还可通过 chunking 与 KV cache 进一步加速。

### 2.2 潜在规划与 Spatial Prompting

VLM Planner 与动作专家之间通过一个轻量的**查询 Transformer**（约 8.7MB）连接：它把变长的输入 token 映射为固定数量的可学习 query token，通过 $k$ 层交叉注意力选择性地关注 VLM 的 $k$ 个中间层（$k=1$ 时仅关注最后一层）。

为了显式激活 Stage 1 学到的空间感知能力，作者引入 **spatial prompting**：在任务指令后追加简单提示，例如把"把葡萄放进篮子里"扩展为"Figure out how to execute it, then locate the key object needed."，抽取出的特征嵌入为 Planner 提供显式的空间线索，即使 VLM 并不真的对这句提示生成文本回答。

受 GR00T N1（Bjorck et al., 2025）、π0.5-KI（Dreiss et al., 2025）、Zhou et al.（2025b）等工作"动作梯度直接回传会扭曲多模态知识"的观察启发，查询 Transformer 中引入了**梯度衰减因子**：动作专家反传到 VLM 的梯度被按固定比例（如 0.5）衰减：

$$\nabla_{\text{VLM}}\mathcal{L} \;\leftarrow\; \lambda \cdot \nabla_{\text{VLM}}\mathcal{L}_{\text{action}} \;+\; \nabla_{\text{VLM}}\mathcal{L}_{\text{grounding}}, \quad \lambda \approx 0.5$$

**用大白话说**：动作头想怎么调 VLM 的参数就随它调，但力度打个五折——既让动作学习能反过来微调规划器，又不至于把它辛苦学来的空间常识"带偏"。

### 2.3 两阶段训练配方

**Stage 1：空间定位预训练（Spatial Grounding Pre-training）**——只优化 VLM。目标不是通用的图文对齐，而是机器人所需的空间推理与规划能力。训练数据把网络级多模态语料与机器人专属数据集（RefCOCO、RoboRefIt、A0、MolmoAct、Pixmo-Points 等）统一改写成 QA 格式，覆盖 bounding-box 检测、轨迹预测、affordance 识别与思维链推理。

**Stage 2：空间引导的动作后训练（Spatially Guided Action Post-training）**——VLM 与动作专家联合优化，两个策略配合使用：
- **Spatial prompting**：预测动作前先给指令前置一段空间线索提示（如上文示例），激发关于物体关系与任务约束的结构化推理；
- **与空间定位数据协同训练（co-training）**：训练在机器人轨迹数据与空间定位数据之间交替进行——轨迹数据上，VLM 主干与动作专家都通过预测噪声与真实噪声之间的 L2 损失联合优化；空间定位数据上，只有 VLM 主干通过 next-token prediction 更新。

扩散动作头遵循标准 Diffusion Policy 目标，直觉上可写作：

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{a,\epsilon,t}\left[\left\| \epsilon - \epsilon_\theta(a_t, t, z_{\text{VLM}}) \right\|^2\right]$$

其中 $z_{\text{VLM}}$ 是查询 Transformer 从 VLM Planner 提取的潜在规划条件。**用大白话说**：动作专家学的是"往噪声动作里加了多少噪声、该怎么把它减回去"，而这个减噪过程时刻被 VLM 给出的"空间地图"牵着走。

在 Post-Pre-Training 阶段（Section 3.2），作者还额外用一个仿真合成的 InternData-M1 数据集（244K 闭环 pick-and-place 样本，建立在 GenManip + Isaac Sim 之上）来初始化动作头，弥合 VLM 预训练与 VLA 微调之间的鸿沟。

### 2.4 数据构建

InternVLA-M1 使用超过 3M 条多模态训练样本，其中 2.3M+ 属于空间推理数据，分四类：

| 类别 | 规模 | 主要来源 |
|---|---|---|
| General QA | 约 637K | LLaVA-OneVision、InternVL3 |
| Box QA | 约 879K | RefCOCO、ASv2、COCO-ReM、InternData-M1、RoboRefIt |
| Trajectory QA | 约 684K | A0 ManiSkill 子集、InternData-M1 waypoint、MolmoAct |
| Point QA | 约 832K | Pixmo-Points、RoboPoint、RefSpatial、InternData-M1 |

此外，作者在 GenManip + Isaac Sim 基础上搭建了一套**可扩展的合成数据引擎**：资产库含 14,716 个标注物体、200+ 张桌面、80+ 组灯光、1,676 种材质纹理；场景图求解器随机生成场景布局，配合抓取生成、运动规划（cuRobo/MPLib）与物理回放，只有同时通过"物理可执行"和"场景图目标达成"验证的轨迹才被保留。相机内外参通过 ArUco 标定与真实相机对齐，渲染阶段解耦于物理规划以提高效率，副产物包括 2D 物体框、2D 轨迹与 2D 点标注，可直接复用为空间定位监督信号。

## 三、实验结果

### 3.1 SimplerEnv（公开基准）

**Google Robot（Table 1）**：InternVLA-M1 相比论文自建的 Vanilla VLA（同样以 Qwen2.5-VL-3B + DiT 头搭建，但不做空间引导训练）：

| 设定 | InternVLA-M1 | Vanilla VLA | Δ | 此前最优公开基线 |
|---|---|---|---|---|
| Visual Matching 均值 | 80.7 | 66.1 | +14.6 | CogACT 74.8（+5.9） |
| Variant Aggregation 均值 | 76.0 | 63.5 | +12.5 | SpatialVLA 70.7（+5.3） |

**WidowX（Table 2）**：InternVLA-M1 均值 71.7 vs Vanilla VLA 54.7（Δ+17.0），相比此前最优基线 GR00T N1.5（61.9）提升 +9.8。

### 3.2 LIBERO（Franka，Table 4）

| 任务集 | InternVLA-M1 | Vanilla VLA | π0 | GR00T N1 | π0.5-KI |
|---|---|---|---|---|---|
| Spatial | 98.0 | **98.8** | 96.8 | 94.4 | 98.0 |
| Object | 99.0 | 98.0 | 98.8 | 97.6 | 97.8 |
| Goal | 93.8 | 81.4 | 95.8 | 93.0 | 95.6 |
| Long | 92.6 | 88.0 | 85.2 | 90.6 | 85.8 |
| Avg | **95.9** | 91.6 | 94.2 | 93.9 | 94.3 |

值得注意的是 Spatial 子任务上 Vanilla VLA（98.8）反而略高于 InternVLA-M1（98.0），但在难度更高的 Goal 与 Long-horizon 子集上空间引导训练带来明显增益（+12.4、+4.6），整体均值取得四个对比方法中的最优。

### 3.3 消融：空间引导对多模态理解/空间定位/操作三方面的联合影响（Table 3）

| 模型 | MME | MMVet | POPE | RefCOCO-g BoxIoU | Where2place PointAcc | Google Robot VM/VA | WidowX VM |
|---|---|---|---|---|---|---|---|
| Vanilla VLA | – | – | – | – | – | 66.1/63.5 | 54.7 |
| Vanilla co-train | 1106 | 19.2 | 78.0 | 47.1 | 21.4 | 70.2/66.5 | 61.1 |
| InternVLA-M1 | **1411** | **23.3** | **86.2** | **71.2** | **25.5** | **80.7/76.0** | **71.7** |

作者还用 SVD 计算了空间定位目标与操作目标之间的 Projection-space Similarity（PSS，Raghu et al. 2017 方法）：直接协同训练（vanilla co-train）梯度子空间对齐度 PSS = 0.25，而空间引导训练把 PSS 提升到 **0.42**，对应更快的收敛速度与更好的空间感知能力保留（Figure 5）。

### 3.4 大规模仿真 Pick-and-Place（200 任务 / 3K+ 物体，Isaac-Sim）

四种泛化设定（In-distribution / Unseen Object / New Background / Unseen Instruction）下，InternVLA-M1（w/ InternData-M1 中训练）相比 GR00T N1.5 平均提升 **+6.2%**，且始终优于 π0 与未做中训练的自身变体（w/o mid-train）。

### 3.5 真实世界杂乱场景 Pick-and-Place（Franka + Robotiq 2F-85）

在自建的 23 个已见物体 / 5 个已见容器 / 27 个未见物体 / 6 个未见容器基准上，InternVLA-M1 在同分布设定下相比 GR00T N1.5、π0 均有明显优势；引入仿真数据 InternData-M1 协同训练后，在未见物体和新配置（新背景、未见位置/朝向、未见指令）上平均提升 **+20.6%**，同分布设定下提升 **7.3%**（原文摘要口径）。

### 3.6 长时程与推理密集型操作

在整理抽屉、组装三明治、按语义分类的桌面整理、数学计算选按钮、按价格标签购物等任务上（含"物理干扰"与"任务中途重新规划"两种压力测试），InternVLA-M1 相比 GR00T N1.5、π0 在多数设定下领先 10 个百分点以上。此外，作者单独评测了 VLM Planner 的任务调度能力（Table 5），把微调后的 3B Planner（"Ours-3B"）与未经任务专属后训练的 Gemini-2.5 Pro、GPT-5、GPT-4o、Qwen2.5-VL-72B/3B 对比：

| 模型 | Sort into Drawers | Make Sandwiches | Desktop Sorting | Math calc. | Goods Purchase |
|---|---|---|---|---|---|
| GPT-5 | 75 | 67 | 62 | 79 | 82 |
| Gemini-2.5 Pro | 57 | 62 | 83 | 53 | 61 |
| Ours-3B（post-trained） | **90** | **91** | **91** | **93** | **92** |

需要注意，这一对比并非公平的零样本对零样本设定：Ours-3B 经过了针对这些具体长时程任务的后训练，而 GPT-5 / Gemini 等为零样本调用，差距更多反映"任务专属后训练"而非底座模型推理能力本身的差距。

## 四、局限性

论文正文没有设置专门的"Limitations"章节，以下基于方法设计与实验设置进行梳理：

1. **数据工程负担重**：2.3M+ 空间定位数据来自十余个异构数据集（RefCOCO、RoboRefIt、A0、MolmoAct、Pixmo-Points、RoboPoint、RefSpatial 等）的统一改写，加上自建的合成数据引擎（InternData-M1，14,716 物体资产），复现该配方的数据工程成本相当可观。
2. **真机验证的具身范围有限**：核心真实世界实验集中在单臂 Franka Research 3（+ Robotiq 2F-85 夹爪），仅"Goods Purchase"任务使用了双臂 ARX LIFT2；论文强调"具身无关的空间先验+具身相关的下游适配"这一设计理念，但并未在人形机器人、移动操作等差异更大的具身形态上给出实证。
3. **长时程任务依赖人工分段的示教标注**：长时程推理实验用了 22 小时、约 500 条/任务的遥操作示教，并需要人工把轨迹切分为子任务片段（segment）并插入零动作向量做过渡标注，标注成本较高，且该子任务边界的定义方式是否能自动化仍是开放问题。
4. **VLM Planner 与通用闭源大模型的比较口径不完全对等**（如上节所述），Table 5 中 Ours-3B 是针对具体任务后训练的，而对比的 GPT-5/Gemini-2.5 Pro 是零样本调用，结论"post-training 对长时程任务调度至关重要"是成立的，但不能直接得出"3B 后训练模型的通用推理能力超过 GPT-5"这样更强的结论。
5. **消融止步于宏观指标**：论文用 MME/MMVet/RefCOCO-g 等标准 benchmark 和 PSS 梯度相似度证明了空间引导训练"确实有效"，但没有给出细粒度的失败案例分析（例如哪类空间关系、哪类遮挡场景仍会失败），对于理解方法边界的信息量有限。

## 五、评价与展望

**优点**：InternVLA-M1 的核心洞察——"空间定位是连接语言指令与具身动作的可迁移中间层"——并不是全新的观点（VoxPoser、ReKep、RoboPoint、RoboRefer 等此前已从不同角度触及"空间先验"这一主题），但本文的贡献在于把这一先验通过一套具体、可复现的两阶段训练配方，端到端地整合进一个统一的双系统 VLA，而不是像层级式系统那样依赖独立的、规则驱动的规划模块。这与 ECoT（Zawalski et al., 2024）、RT-H（Belkhale et al., 2024）等"生成中间文本推理"的思路形成对照：InternVLA-M1 选择用 spatial prompting + 潜在规划 token 的方式把推理"隐式化"到潜空间，声称可以省去显式生成推理文本带来的额外推理开销，这是一个务实的工程取舍，但代价是可解释性和可调试性会弱于显式思维链方法。

Table 3 的 PSS（梯度子空间相似度）分析是本文比较有说服力的机制性证据：它直接量化了"空间定位目标"和"操作目标"梯度方向的对齐程度（0.25 → 0.42），为"co-training 有效"这一现象提供了一个比单纯的下游指标更接近原因层面的解释，这一分析思路也可推广到其他多任务/多目标联合训练的 VLA 工作中。

**与其他公开工作的关系**：在基线选择上，本文与 GR00T N1.5、π0/π0-FAST、CogACT、SpatialVLA、OpenVLA 等主流开源 VLA 做了较为全面的对比，覆盖 SimplerEnv 与 LIBERO 两个社区认可度较高的公开基准，且给出了自建的 Vanilla VLA（同架构无空间引导）作为受控消融对象，这种"同架构、单变量"的对比设计比单纯堆砌 SOTA 数字更有说服力。相较 GR00T N1.5、π0.5-KI 同样采用的"梯度隔离/衰减"机制，本文的查询 Transformer + 梯度衰减因子设计思路接近，说明"防止动作梯度污染语言-视觉表示"已成为这一代双系统 VLA 的共识性设计模式。

**开放问题与可能的改进方向**：(1) 论文用固定的梯度衰减系数（如 0.5），未探讨该系数是否需要随训练阶段或任务动态调整，这是一个可以做进一步消融的方向；(2) 空间 grounding 的监督形式停留在点/框/轨迹三类 2D 表示，尚未系统性纳入 3D/深度信息或多视角一致性约束，而这是近期不少空间推理 VLA 工作（如 SpatialVLA、RoboRefer）关注的另一条演进路线，两者结合可能进一步提升遮挡、堆叠等复杂场景下的定位精度；(3) 长时程任务的子任务边界目前依赖人工标注切分，若能通过无监督或自监督的方式自动发现子任务边界，将显著降低该类数据的构建成本；(4) 论文尚未报告模型在训练数据分布之外的全新任务类别（而非仅仅"未见物体/未见指令"）上的表现，这类"新任务零样本泛化"的评测将是检验空间先验是否真正具有可迁移性的更严格试金石。

## 参考

1. Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.
2. Bjorck, J. et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734, 2025.
3. Li, Q. et al. *CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation.* arXiv:2411.19650, 2024.
4. Qu, D. et al. *SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model.* arXiv:2501.15830, 2025.
5. Zawalski, M. et al. *Robotic Control via Embodied Chain-of-Thought Reasoning.* arXiv:2407.08693, 2024.
