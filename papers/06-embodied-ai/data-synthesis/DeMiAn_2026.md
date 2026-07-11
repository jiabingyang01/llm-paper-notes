# DeMiAn：如何给机器人下指令——密集语言标注赋能机器人策略学习

> **论文**：*How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning*
>
> **作者**：Bosung Kim, Ruiyi Wang（共同一作）, David Acuna, Jaehun Jung, Alexander Trevithick, Brandon Cui, Yejin Choi, Prithviraj Ammanabrolu
>
> **机构**：University of California, San Diego；NVIDIA
>
> **发布时间**：2026 年 05 月（arXiv 2605.17077）
>
> **发表状态**：未录用（预印本，标注 "Preprint"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.17077) | [PDF](https://arxiv.org/pdf/2605.17077)
>
> **分类标签**：`密集语言标注` `数据再标注` `VLA` `世界-动作模型` `具身预训练`

---

## 一句话总结

把"语言密度"当作一个廉价杠杆：用一个 VLM 对**已有** 机器人/人类演示沿 physical_motion / scene_composition / arm_pose / reasoning **四个互补维度** 做密集再标注（DeMiAn），再训练一个**异步 instructor** 在部署时为每个任务挑选最合适的那种标注；在 RoboCasa365 上把 π0.5 VLA 成功率从 44% 提到 49%（逼近 52% 的 per-task oracle），并在 1M 规模 MolmoBot 上以约 **62% 更少的算力** 匹配无标注基线。

## 一、问题与动机

现代机器人策略学习的主导范式是"规模化"——更多演示、更多环境、更多本体。但采集演示极其昂贵：现代规模的数据集需要在专用机器人硬件上投入数千小时熟练操作员的遥操作；人类第一视角数据集同样昂贵（如 EgoVerse 50K 是在约 1500 小时线下录制中采集的）。

作者的关键观察是：**与这些演示配对的语言并不面临同样的瓶颈**。一段演示典型地只配一句短指令（如 "open the drawer"），把像素和动作里已经隐含的空间关系、接触转换、本体状态、子目标结构统统留成了隐式。对已有数据集做更丰富的再标注，其成本比采集新演示低几个数量级——一次 VLM 调用能在数秒内、以亚美分成本、无需人工，产出多句描述。因此作者把 **语言密度（language density）** 视为从固定演示语料中榨取更多监督信号的廉价杠杆。

与以往密集语言工作的区别：现有工作往往锁定**单一** 的 caption 风格（RT-H 的 "language motions"、ECoT 的逐步 chain-of-thought、CLIPort 的场景/物体 grounding、Inner Monologue 的环境状态独白）。本文反过来问：**哪一种密集语言对哪一类任务最有帮助,又如何在部署时把它送到位?** 核心发现是——caption 的**类型** 和它的**存在与否** 同等重要,不存在一种固定标注规则能达到 per-task oracle。

## 二、核心方法

DeMiAn（Dense Multi-aspect Annotation）在不采集任何新数据的前提下,通过两阶段放大语言信号。

### 1. 训练时:四维度密集再标注流水线

作者把各流派的单一 caption 风格整合为四个互补的**标注维度**（aspect,见下表）,每个维度捕获一条被一句话任务标签所省略的结构信息:

| 标注维度 | 动机 | 标注信号 |
|---|---|---|
| `physical_motion` | 时序动作与接触 grounding | 手/末端执行器运动:reach、grasp、lift、place、open、close 及涉及的物体 |
| `scene_composition` | 空间 grounding | 工作区、可见物体、干扰物、与任务相关的空间关系 |
| `arm_pose` | 本体与接触状态 grounding | 段边界处的手/臂姿态、reach 方向、接触状态 |
| `reasoning` | 因果与子目标 grounding | 段的用途（preparation / main manipulation / cleanup）及对相邻段的依赖 |

**流水线细节**:一个 segment 是数据集元数据关联到单条一句话标签的连续子区间(如一段标为 "open the drawer" 的帧区间)。对每个 segment,均匀采样至多 $F=10$ 帧,对四个维度 $k\in\mathcal{K}=\{$physical_motion, scene_composition, arm_pose, reasoning$\}$ **各发一次 VLM 调用**。每次调用以采样帧 + 一个上下文块(任务描述、场景描述符、物体列表、相邻段标签)为条件,prompt 强制严格 JSON schema 和两句话长度上限。较长的多基元演示会先被切成单基元 clip(如 `OpenDoubleDoor` 拆成两段单门 clip)。标注模型为 **Qwen3-VL-30B-A3B-Instruct**。该流水线应用于三个已有数据集且不采集新演示:**RoboCasa 365(11K clip)、MolmoBot(1M clip)、EgoVerse(50K 人类第一视角 clip)**,共产出约 100 万条标注。对 EgoVerse 用了改写为人类第一视角措辞的 prompt 版本。

### 2. 推理时:学习式 instructor(异步注入)

由于不同维度帮助不同任务,一个"对所有任务用同一固定维度"的部署会跑输给"为每个任务挑维度"的 oracle。作者训练一个小的 **instructor**(Qwen3.5-2B),把任务描述 + 初始场景快照(3 帧初始观测 RGB)映射到一个任务合适的标注。

训练采用 **reward-weighted target sampling(SFT)**:先用动作策略在所有训练任务上分别跑四种 GT 固定维度标注,记录每任务验证 SR,构造奖励表 $w(\tau,k)$。对每个训练 episode,目标维度按对 $w(\tau,\cdot)$ 的 softmax 采样:

$$
k \sim \mathrm{softmax}\!\left(\frac{w(\tau,\cdot)}{T}\right),\quad T=2,\ \text{top-3 截断}
$$

采到的那个维度对应的流水线生成 caption 作为 SFT 目标。**若某任务下每个维度都跑输无标注基线,则赋空目标,教 instructor 学会"弃权(abstain)"**。作者也试过 Top-1 SFT 和两种 DPO 变体,均未超过默认 SFT(DPO 因长度混淆的偏好对倾向漂移到过长输出而反噬动作策略)。

> 用大白话说:先做一张"打分表"记下每个任务配哪种标注涨点最多,再让一个小模型看着任务描述和开场画面去模仿这张打分表——学会"这个任务该给运动描述、那个任务该给场景描述,有些任务干脆别多嘴"。

**异步注入(Async Injection)**:部署时生成的标注与任务描述拼进同一 prompt 槽。instructor **异步** 与动作策略并行运行——机器人先仅凭任务描述开始动作,等标注就绪后在下一个 action-chunk 边界插入(见论文 Figure 2)。这把约 1.87s 的生成延迟藏在了 rollout 之后,保持动作服务器正常控制节奏。

### 3. 两种策略骨干

- **DeMiAn VLA**:以开源 **openpi 0.5(π0.5)** 为骨干(PaliGemma 视觉语言骨干 + flow-matching 动作专家)。相较原始 openpi,额外加一个小的标注级 LM 交叉熵作为**正则项**(非生成目标),防止双向解码器把标注压成粗糙全局摘要:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{FM}}+\lambda_{\mathrm{LM}}\,\mathcal{L}_{\mathrm{cap}},\qquad
\mathcal{L}_{\mathrm{cap}}=-\frac{1}{\sum_{i=1}^{N-1} m_i}\sum_{i=1}^{N-1} m_i\log p_\theta(c_{i+1}\mid h_i)
$$

其中 $\lambda_{\mathrm{LM}}=0.1$,$m_i$ 是标注 token 掩码。由于前缀是双向注意力,$h_i$ 能看到后续标注 token,该 $\mathcal{L}_{\mathrm{cap}}$ 并非因果似然,只作表征 grounding 正则。**推理时标注由外部 instructor 提供,模型自身不生成标注,故无 train-test 失配。**

> 用大白话说:这个辅助语言损失不是让模型学会"写标注",而是像一根"拴绳",逼着标注 token 的隐状态保持可解码,免得双向编码器偷懒把丰富标注揉成一团模糊的全局向量,丢掉物体身份和空间关系。

- **DeMiAn WAM**:视频式世界-动作模型,采 **Cosmos-Predict 2.5 视频 DiT** 骨干(prefix-suffix 设计,风格类 GR00T-N1),用 Cosmos Reason 1 作冻结前缀编码器。动作头是 4 层 cross-attention 栈($K=4$ 层在均匀深度各接一个 DiT block)。视频目标不变,动作从学到的视频特征确定性预测。所有标注消融中模型架构、视觉数据、动作目标固定,只改与每段演示配对的语言。

## 三、实验结果

评测基准:**RoboCasa365**(17 个原子厨房任务,held-out target split,SR 由仿真成功检查器判定)与 **MolmoSpaces bench-v2**(9 个基准,四大任务族 Pick / Pick+Place / NextTo / Color)。

### RQ1:哪种标注对哪个任务最好?(the oracle gap)

密集标注有帮助但**不均匀**。DeMiAn VLA 在 RoboCasa 上,`physical_motion` 带来大幅单任务提升:

| 任务 | 无标注基线 | 加 Physical Motion |
|---|---|---|
| SlideDishwasherRack | 38% | **75%**(+37) |
| CoffeeSetupMug | 29% | **60%**(+31) |
| CloseToasterOvenDoor | 60% | **74%**(+14) |

在 MolmoSpaces 上,`scene_composition` 在 Pick 任务加 **13 个百分点**,`reasoning` 在 NextTo 任务比基线加 **8 个百分点**。由于没有单一固定维度处处最优,per-task oracle(逐列取四维度最大)在 RoboCasa 达到 **52% 平均 SR**,比基线高 8 个点、比最好的固定维度(Physical Motion 46%)高 6 个点。DeMiAn WAM 在更低绝对 SR 上呈现同样形状(最好固定维度 18%,oracle 20%)。

注意力可视化(论文 Figure 4)显示:基线策略对 prompt token 的注意力几乎全塌在 `<bos>` token 上(把 prompt 当位置锚点而非语言在读);而 Physical-Motion 标注训练的策略,注意力分散到标注中被语言 grounding 的单元——交互物体(oven door)、动词(push、retracts)、方向副词(inward、away),说明动作专家确实在**读** 密集 caption。

### RQ2:学习式 instructor 能逼近 per-task oracle 吗?

instructor 把 RoboCasa 与 MolmoSpaces 平均 SR 从 44% 提到 49%,落在 per-task oracle(51–52%)的 2–3 个点内。RoboCasa Dev 集详解(共享同一动作策略 checkpoint,只改推理指令):

| 推理时指令 | SR | 相对基线 Δ |
|---|---|---|
| Baseline(无标注) | 44.3% | — |
| GT physical_motion(固定) | 46.1% | +1.8 |
| GT scene_composition(固定) | 48.4% | +4.1 |
| GT arm_pose(固定) | 47.4% | +3.1 |
| GT reasoning(固定) | 50.1% | +5.8 |
| Random 每-episode 随机维度 | 46.6% | +2.3 |
| **SFT Instructor(异步,本文)** | **50.4%** | **+6.1** |
| Oracle(per-task-best) | 52.4% | +8.1 |

与随机每-episode 路由(46.6%)相比,instructor 多加 **+3.8 点**,说明增益来自**学到的逐任务选择** 而非启发式路由。

**异步部署零策略延迟**(Table 3):异步在 SR 上与同步只差零点几个点(**49.0% vs 49.5%**);同步在第 1 步就注入但阻塞首动作,异步在第 21.9/23 步注入,把约 1.87s/1.86s 生成延迟藏在前约 22 个动作步之后。

### RQ3:能否泛化到训练分布之外的新指令?

策略在 RoboCasa365 原子任务上训练,测试到 18 个组合任务(各由训练见过的原子任务链成)。两种 prompt 策略:`-fix`(整段喂一句组合任务描述)与 `-dynamic`(子目标驱动状态机,一旦宽松的仿真内触发器 fire 就把 prompt 换成当前阶段的分布内原子指令)。

| 指标 | Baseline `-fix` | DeMiAn-VLA `-fix` | Baseline-GT `-dynamic` | DeMiAn-VLA-GT `-dynamic` | DeMiAn-VLA(instr) |
|---|---|---|---|---|---|
| Phase-1 SR | 50% | 52% | 57% | **65%** | 61% |
| Phase-2 SR | 28% | **32%** | 26% | 31% | 30% |
| Full-task SR | 13% | 15% | 19% | **22%** | 18% |

要点:(1) OOD `-fix` 下 DeMiAn-VLA 比任务-only 基线 +2 全任务点(15% vs 13%),密集标注带来适度鲁棒性;(2) `-dynamic` 下 DeMiAn-VLA-GT 最强(22% vs 19%),密集标注策略比基线更受益于子目标分解;(3) 部署时用 instructor 换掉 GT 原子 prompt,全任务 SR 降到 18%,居于两个 GT 配置之间、约比 baseline-GT 低 1 点——差距集中在最后阶段,是留待未来工作的开放问题。

### RQ4:密集标注是算力高效的杠杆吗?

作者把生成标注的算力(FLOPs)**计入预算** 做等算力对比。在 50K EgoVerse clip 上 mid-train DeMiAn WAM(无动作头,纯视频预测目标),下游 RoboCasa365 评测:小的前期标注成本之后,带标注的 WAM 在相近算力预算下达到更高下游 SR。在 1M 轨迹 MolmoBot 上 post-train:带标注策略更早达到更强 MolmoSpaces SR 且峰值更高;在 **MolmoSpaces NextTo 与 Color 任务上,DeMiAn 以约 62% 更少的算力匹配无标注基线,节省约 $1.3\times10^{20}$ FLOPs**。

**标注成本**:单次 Qwen3-VL-30B 调用处理约 8.2K 输入 token(帧+prompt)、约 150 输出 token,$F_{\mathrm{call}}\approx 2P_{\mathrm{active}}(X_{\mathrm{in}}+X_{\mathrm{out}})\approx 5.0\times10^{13}$ FLOPs;对 1M-clip 语料做一个维度的一遍标注约 $5.0\times10^{19}$ FLOPs,按托管价约 **$1.1K/维度**,四维度约 4 倍。

## 四、局限性

- **单一冻结 VLM 的偏差**:所有标注由一个冻结 VLM 产出,继承其偏差与幻觉倾向。
- **逐任务选型未解决**:instructor 只闭合了 per-task oracle 差距的大部分而非全部;在部分任务上**每个固定维度都跑输任务-only 基线**,即密集标注并非处处有益。
- **仅仿真**:实验只在 RoboCasa 365 与 MolmoSpaces 上做,无真机评测;语言密度杠杆能否越过这些基准的视觉/动力学分布仍待验证。
- **四维度 schema 固定且启发式选定**:维度集人工选定、按数据集(而非按 segment)混合;组合任务最后阶段的差距未闭合。
- **instructor 与动作策略解耦**:instructor 只模仿离线奖励表,未与动作策略联合优化;作者指出联合训练(让 instructor 直接对下游动作成功率优化)是自然的下一步。

## 五、评价与展望

**优点**:(1) 立意清晰且实用——把"语言类型"提升为一等设计选择,而非固定 prompt 格式;"密集再标注比采新数据便宜几个数量级"的成本不对称性是坚实的现实动机。(2) oracle gap 的刻画(不存在单一固定维度达到 per-task oracle,且逐任务/逐族异质性很大)是有信息量的负结果,直接为 instructor 的存在提供理由。(3) 算力计入预算的等算力对比比"忽略标注成本"的常见做法更诚实,62% 省算力的结论有说服力。(4) 异步注入是简洁的工程点子,把 LLM 延迟藏在动作 rollout 后,几乎零 SR 损失。(5) 同一杠杆跨 VLA(π0.5)与 WAM(Cosmos-Predict)两种主流范式验证,增强了普适性论证。

**缺点/存疑**:(1) 绝对 SR 偏低(RoboCasa 44→49,WAM 更低到 20% 量级),密集标注是"锦上添花"而非量变。(2) 全部仿真,真机迁移是最大未知。(3) instructor 训练依赖一张需要"跑遍四种 GT 标注 × 全部任务"才能构造的奖励表,这本身有不小的前期算力/工程成本,论文对这部分成本的会计不如标注成本透明。(4) instructor 在组合任务上换掉 GT 原子 prompt 后反而略低于 baseline-GT,说明学到的选型在分布外仍脆弱。

**与其他公开工作的关系**:相较 RT-H(在任务与动作间插 "language motions")、ECoT(逐步具身 chain-of-thought)锁定单一子指令风格,本文的贡献是**系统地比较四种风格并学习按任务投递**;相较 ROSIE(用文生图做视觉数据增强)、R3M/Voltron/LIV(用语言做表征预训练),本文把密集 per-step caption 用作 VLA 微调期的**辅助协同训练信号**,且推理时仅在 prompt 里以文本形式注入。

**开放问题与可能改进方向**:(a) 把标注维度从"按数据集"细化到"按 segment"混合,甚至让 instructor 输出多维度融合;(b) instructor 与动作策略端到端联合训练,用下游 SR 而非离线奖励表作目标;(c) 真机验证语言密度杠杆是否跨本体/跨视觉分布迁移;(d) 组合/长程任务最后阶段的选型崩塌需要更强的子目标追踪或在线反馈闭环。

## 参考

1. Belkhale et al., *RT-H: Action Hierarchies Using Language*, arXiv 2403.01823 —— 在任务与动作间插入 "language motions" 的细粒度子指令语言,本文的直接对照。
2. Zawalski et al., *Robotic Control via Embodied Chain-of-Thought Reasoning (ECoT)*, arXiv 2407.08693 —— 动作前发射逐步推理的具身 CoT,`reasoning` 维度的思想来源。
3. Black et al., *π0/π0.5: A Vision-Language-Action Flow Model for General Robot Control*, arXiv 2410.24164 —— DeMiAn VLA 的 openpi 骨干。
4. Yu et al., *Scaling Robot Learning with Semantically Imagined Experience (ROSIE)*, arXiv 2302.11550 —— 把语言当数据增强接口,与本文"语言当协同训练信号"形成对照。
5. Nasiriny et al., *RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots*, arXiv 2603.04356 —— 主评测基准之一。
