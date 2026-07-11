# RoboEvolve：面向有限数据机器人操作的规划器-模拟器协同进化框架

> **论文**：*RoboEvolve: Co-Evolving Planner-Simulator for Robotic Manipulation with Limited Data*
>
> **作者**：Harold H. Chen, Sirui Chen, Yingjie Xu, Wenhang Ge, Ying-Cong Chen（Harold H. Chen 与 Sirui Chen 共同一作,Ying-Cong Chen 通讯）
>
> **机构**：The Hong Kong University of Science and Technology (Guangzhou);The Hong Kong University of Science and Technology
>
> **发布时间**：2026 年 05 月（arXiv 2605.13775）
>
> **发表状态**：未录用（预印本,v1）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.13775) | [PDF](https://arxiv.org/pdf/2605.13775)
>
> **分类标签**：`视频生成世界模型` `规划器-模拟器协同进化` `自进化数据合成` `GRPO+DPO`

---

## 一句话总结

RoboEvolve 把一个 VLM 规划器（planner $\mathcal{P}$）和一个视频生成模型模拟器（VGM simulator $\mathcal{S}$）耦合成一个相互强化的协同进化闭环,仅用**无标注种子图像**、通过"白天在线探索（GRPO）+ 夜晚离线巩固（DPO,挖掘 near-miss 失败）"的类脑双阶段机制自动合成任务对齐的交互数据,在 BridgeData V2 上把模拟器相对成功率平均提升约 48%、在 EB-ALFRED / EB-Habitat 上把基座规划器平均抬高 30 个绝对点,并以 500 张无标注种子图（相对 25K 人工标注轨迹缩减 50×）超越全监督基线。

## 一、问题与动机

机器人操作的核心瓶颈是**可扩展、任务对齐的交互数据与监督信号极度稀缺**。真机轨迹采集昂贵、耗时,尤其在需要精细标注或人类演示时。现有两条自动造数据路线各有硬伤:

- **VLM 作规划器**:语义理解强,可当"大脑"生成高层计划,但其空间-物理推理被压在文本潜空间里,常产生逻辑自洽却**物理不可行**的计划(semantic-spatial / textual-spatial misalignment),要落地必须靠昂贵的人工核验。
- **VGM 作模拟器**:能合成大规模交互视频,是替代真机采集的可扩展途径,但因缺乏任务对齐的训练数据,普遍存在**物理幻觉**(physical hallucination)——画面看似合理却物理不可行(如物体突然消失)。

作者提出的核心假设是:**VLM 与 VGM 可以互相辅助**——VLM 给 VGM 提供多样且语义扎实的任务提示与判据,引导其生成更有意义的轨迹;VGM 则模拟物理可行性、把物理反馈回灌给 VLM 修正规划。据作者称此前没有工作直接研究这一协同问题;而且现有自博弈/自进化系统几乎只做语言域,并且**只盯着成功轨迹、把失败当作无信息噪声丢弃**。由此凝练出核心研究问题:如何设计一个耦合 VLM 规划器与 VGM 模拟器、能同时利用成功与失败、从有限无标注数据自我进化的协作系统?

灵感来自认知科学的 **Complementary Learning Systems (CLS)** 理论——有效学习源于探索性过程与巩固性过程的交替。

## 二、核心方法

RoboEvolve 建立一个把"自主发现"与"知识巩固"交替进行的自进化闭环,分四步。

### 1. 场景锚定的任务初始化（Scene-Grounding Task Initialization）

从无标注种子图 $I$ 出发,规划器 $\mathcal{P}$ 抽取结构化场景表示 $S(I)$,含三要素:物体 $\{o_k\}$、空间关系(on/in/near)、affordance 先验(pickable/openable)。为抗感知误差与幻觉,用 **self-consistency voting**:采样 $m=8$ 份独立解析,只保留多数一致的实体与关系。随后把 $S(I)$ 映射到 BridgeData V2 taxonomy 的 **13 个基础任务模板**(pick-and-place、stacking 等),并将原子技能组合成层级化复合任务(如 `pick(bowl) → place(bowl, rel=on(table)) → push(spoon, rel=in(cabinet))`)。

每个计划 $\pi = \langle a_1,\dots,a_n \rangle$ 被分解为原子动作序列,任务难度定义为各原子动作单位代价的累加:

$$D(\tau \mid I) = \sum_{a_i \in \pi} c(a_i)$$

> 用大白话说:一张图先被 VLM"读懂"成有哪些物体、谁在谁上面、哪些能抓能开,再拼成从简单到复杂的任务;每个任务打一个"难度分",这个难度分就是后面课程自动升级的状态变量。

### 2. 白天学习:在线探索（Daytime Learning,GRPO）

**模拟器白天训练**:对每个任务 $\tau$,在同一提示下从 $\mathcal{S}$ 采样 $K$ 条视频轨迹,用 GRPO 组内相对优势优化,提升物理保真度:

$$\mathcal{J}_{\text{Daytime}}(\mathcal{S}) = \mathbb{E}_{\tau\sim\mathcal{D},\{V_k\}\sim\mathcal{S}}\left[\frac{1}{K}\sum_{k=1}^{K}\text{clip}\left(\frac{\mathcal{S}(V_k\mid\pi)}{\mathcal{S}_{\text{old}}(V_k\mid\pi)},\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_k\right]$$

优势 $\hat{A}_k = R(V_k) - \frac{1}{K}\sum_{j=1}^{K}R(V_j)$,其中奖励 $R(V)$ 由规划器 $\mathcal{P}$ 依据语义与任务对齐度给出。

**规划器白天训练**:当 $\mathcal{S}$ 在难度 $D$ 上稳定后,让 $\mathcal{P}$ 去探索更长程任务 $\mathcal{T}_{\text{high}}$(复杂度 $(D, 2D)$,如"做汉堡")。为防止过度依赖 $\mathcal{S}$ 导致的幻觉,采用 **selective simulation strategy**:self-consistency 投票选出最一致的计划 $\pi^*$,只在难度 $\le D$ 的分段内(segment-wise)用 $\mathcal{S}$ 验证。规划器同样用 GRPO 优化,奖励为:

$$\hat{R}(\pi,\tau) = \mathbb{1}[\pi = \pi^*]\cdot\big(1 + \eta\cdot R(\mathcal{S}(\pi^*))\big)$$

其中 $\mathbb{1}[\pi=\pi^*]$ 是共识计划的二值门,$R(\mathcal{S}(\pi^*))$ 是来自物理反馈的 reward shaping 项。乘性二值门既让 $\mathcal{P}$ 不采纳不可执行的逻辑,又隔离了 $\mathcal{S}$ 潜在幻觉。

> 用大白话说:模拟器先在当前难度把"物理"练扎实(GRPO 组内比谁生成得更物理);规划器则大胆往更难的任务想,但只在模拟器"信得过"的难度区间去验证,靠"和多数投票的计划一致才给分"这道闸门防止被模拟器的幻觉带偏。

**语义可控多粒度奖励(Semantic-controlled Multi-Granular Reward)**:这是把物理保真拆细的关键设计。

$$R(V) = \mathbb{1}_{\text{sem}}\cdot\big(s_F + w_s s_S + s_E\big)$$

- $\mathbb{1}_{\text{sem}}$:语义对齐指示子。$\mathcal{P}$ 不是从头重写提示,而是当"评审"检查视频 $V$、只修改原目标 $G$ 中冲突的部分得到修订提示 $G'$,用 $\text{Sim}(G, G')$ 作为 $\mathbb{1}_{\text{sem}}$——一旦语义偏离,物理分被成比例压制。
- $s_F$（Frame-level Consistency）:惩罚不连续,仅当全帧的物体持久性与空间平滑性都保持时 $s_F=1$。
- $s_S$（Segment-level Execution）:$s_S = \frac{1}{M}\sum_{i=1}^{M}\mathbb{1}[a_i \in Seg_i]$,权重 $w_s = 1/M$,做动作级二值检测。
- $s_E$（Episode-level Success）:整段任务是否达成。

物理奖励全部离散化为二值信号以保证数值稳定、杜绝 reward-hacking,并为夜晚学习提供清晰的失败判据。

> 用大白话说:一段视频好不好,先看它"有没有跑题"(语义门),没跑题再从帧级(画面连不连贯)、段级(每个动作有没有做出来)、集级(最终有没有成功)三个粒度打分;语义一偏,物理分就被打折,防止模型学会"画得漂亮但答非所问"。

### 3. 夜晚学习:离线巩固（Nighttime Learning,DPO,挖掘 near-miss 失败）

白天探索有广度但夹带物理幻觉和规划谬误,夜晚像皮层巩固一样把原始经验转成高价值偏好对。

**模拟器夜晚训练**:构造视频偏好对 $(V^+, V^-)$——正样本是累积奖励高($s_E=1$ 且 $\mathbb{1}_{\text{sem}}=1$)的物理一致轨迹;负样本刻意选**hard negatives**:至少满足一条有效性准则($s_F=1$ 或 $w_s s_S=1$)但整体任务失败($s_E=0$)的"near-miss"。用 DPO 目标让 $\mathcal{S}$ 学会区分"看似合理却物理无效"与"物理正确":

$$\mathcal{L}_{\text{Nighttime}}(\mathcal{S}) = -\mathbb{E}_{(V^+,V^-)}\left[\log\sigma\left(\beta\log\frac{\mathcal{S}(V^+\mid\pi)}{\mathcal{S}_{\text{ref}}(V^+\mid\pi)} - \beta\log\frac{\mathcal{S}(V^-\mid\pi)}{\mathcal{S}_{\text{ref}}(V^-\mid\pi)}\right)\right]$$

$\mathcal{S}_{\text{ref}}$ 是上一轮迭代的模拟器,保证 $\mathcal{S}$ 逐步贴近物理真实的流形。

**规划器夜晚训练——层级化偏好优化(hierarchical preference optimization)**:跨三个认知维度抽取偏好信号:

- **Planning-level $\mathcal{D}_P$**(高层目标与分解):偏好多数投票的共识计划 $\pi^*$ 而非次优候选。
- **Understanding-level $\mathcal{D}_U$**(场景与物体理解):给定高奖励视频 $V^+$,偏好被验证的目标 $G$ 而非错误的反向翻译,纠正感知误判。
- **Transition-level $\mathcal{D}_T$**(动作可行性与动力学):给定状态对 $(f_1, f_T)$,正确推断出计划 $\pi^*$ 而非被误认的意图,内化因果。

累积目标:

$$-\sum_{k\in\{P,U,T\}}\mathbb{E}_{(\mathbf{c},\pi^+,\pi^-)\sim\mathcal{D}_k}\left[\log\sigma\left(\beta\log\frac{\mathcal{P}(\pi^+\mid\mathbf{c})}{\mathcal{P}_{\text{ref}}(\pi^+\mid\mathbf{c})} - \beta\log\frac{\mathcal{P}(\pi^-\mid\mathbf{c})}{\mathcal{P}_{\text{ref}}(\pi^-\mid\mathbf{c})}\right)\right]$$

> 用大白话说:白天失败的那些"差一点点成功"的样本不再被丢掉,而是当作最有信息量的负样本——模拟器学会不再把它们生成得"以假乱真",规划器则从"目标/理解/动力学"三个层面分别改错,把想象和现实的缝隙缝上。

### 4. 双阶段课程进化（Dual-Phase Curriculum Evolution）

按难度分 $D(\pi\mid I)$ 把难度空间离散成 $B$ 个 bin,跟踪每个 bin 成功率 $S(b)$,定义学习进展 $P_k(b) = S_k(b) - S_{k-\Delta}(b)$,用 UCB 上置信界策略挑下一阶段的上限难度:

$$b_k^* = \arg\max_b\left(P_k(b) + \lambda\sqrt{\frac{\log\sum_j n_k(j)}{n_k(b)+1}}\right)$$

当简单任务成功率饱和($P_k(b)\to 0$),采样预算自动转向更高复杂度前沿,实现无人工干预的持续能力扩张。

## 三、实验结果

**设置**:模拟器 $\mathcal{S}$ = Wan2.2-TI2V-5B,规划器 $\mathcal{P}$ = Qwen3-VL-4B(均选轻量部署尺寸)。$\mathcal{S}$ 用 Flow-Factory + Flow-DPO 训练,$\mathcal{P}$ 用 TRL 库。GRPO rollout $K=16$,reward shaping $\eta=0.2$,课程 $\lambda=0.1$,核心评测最大难度 $D=3$。**500 张无标注种子图** → 877 个 $D{=}1$ 原子任务 → 组合成 3,363 个($D{=}2$)+ 9,228 个($D{=}3$)复合任务。全部在 NVIDIA A800 上完成。模拟器在 BridgeData V2 测试集评测(VBench 视觉质量 / Gemini-2.5-Pro 判定的 Success / User Preference),规划器在 EB-ALFRED 与 EB-Habitat 评测。

**模拟器(BridgeData V2,任务成功率 Success)**——难度越高,协同进化相对增益越大:

| 方法 | L1 Success | L2 Success | L3 Success |
|---|---|---|---|
| Wan2.2-TI2V(base) | 0.477 | 0.395 | 0.324 |
| + SFT(cold start) | 0.491 | 0.409 | 0.341 |
| Wow-1-Wan(最强外部基线) | 0.519 | 0.439 | 0.364 |
| + 冻结 Planner $\mathcal{P}$ | 0.640 | 0.561 | 0.483 |
| **+ RoboEvolve(本文)** | **0.668** | **0.591** | **0.505** |
| 相对 base 增益 Δ% | **+40.0%** | **+49.6%** | **+55.9%** |

VBench 上 RoboEvolve 也取得各级最优(L1 0.852 / L2 0.841 / L3 0.828);User Preference 从 base 的 0.076/0.056/0.026 提升到 0.198/0.288/0.360。三级相对成功增益平均约 48%,且随复杂度单调放大(L1→L3:40.0%→55.9%)。

**规划器(EB-ALFRED / EB-Habitat,平均分 Avg)**:

| 方法 | EB-ALFRED Avg | EB-Habitat Avg |
|---|---|---|
| Qwen3-VL(base) | 25.3 | 29.7 |
| RoboAgent(域内 SOTA 参考) | 67.0 | 22.3 |
| WAP(参考) | 62.7 | — |
| + 冻结 Simulator $\mathcal{S}$ | 55.7 | 49.0 |
| **+ RoboEvolve(本文)** | **61.7** | **54.3** |
| Δ 相对 base | **+36.4** | **+24.6** |

RoboEvolve 无需域内专门训练,就把通用 Qwen3-VL 平均抬高 EB-ALFRED +36.4、EB-Habitat +24.6 个绝对点(两者均值约 30 点),在空间与长程推理子项上尤为突出,并逼近/超越域内专家。

**消融与扩展**(Figure 5-6):

- 双阶段缺一不可:Daytime-only 因未纠正的物理幻觉累积而迅速饱和;顺序基线 "D+N"(先跑完全部白天探索再夜晚巩固)因不可逆的策略退化而成功率显著更低——紧密交替的睡-醒循环才是稳定器。夜晚对 near-miss 的激进惩罚是防止连续探索中模型崩塌的关键。
- 持续学习:课程扩展到 $D=4$ 仍能掌握更复杂的组合任务,对简单原子动作无灾难性遗忘,各指标单调上升。
- 数据效率:仅从 300 张种子图起步就合成约 7.6K 轨迹(不足原始 BridgeData ~25K 人工标注的 1/3),却完整超越 $\mathcal{S}$ 的 L3 成功率与 $\mathcal{P}$ 的 EB-ALFRED 表现;500 张种子相对 25K 标注约 50× 缩减。

## 四、局限性

论文未设独立局限性小节,以下为客观评述:

1. **"操作"停留在视频层面,未闭环到真机/真策略**:模拟器是视频生成模型,产出的是"看似执行任务的视频",Success 由 Gemini-2.5-Pro 作 VLM-as-judge 打分,而非真机执行或动作头输出。合成视频对训练下游 VLA 策略的实际增益、以及视频世界模型的 sim-to-real 迁移,论文均未验证,是最关键的开放缺口。
2. **循环监督/reward-hacking 风险**:奖励里的语义门 $\mathbb{1}_{\text{sem}}$ 与共识计划 $\pi^*$ 都由规划器 $\mathcal{P}$ 自己生成又自己评判,规划器既当运动员又当裁判;虽用二值离散化与 self-consistency 缓解,但两模型互相打分的闭环仍有共同漂移(mutual collusion / 崩塌)的隐患,论文主要用夜晚巩固经验性地稳住,缺乏理论保证。
3. **任务空间受 13 个模板约束**:所有可发现行为都源自 BridgeData V2 taxonomy 的 13 个基础模板与其组合,行为多样性上限受限,难以涌现模板外的新技能。
4. **强依赖 VLM 场景解析质量**:整个闭环从 $S(I)$ 的正确解析起步,$m=8$ 投票能降噪但无法根除感知偏差,种子图分布外场景表现未知。
5. **算力成本高**:对视频生成模型反复做 GRPO + DPO 迭代(A800)开销可观,评测上限 $D=3/4$,更长程任务的可扩展性尚待检验。

## 五、评价与展望

**优点**:(1) 把 VLM 规划器与 VGM 模拟器显式塑造成**相互监督的协同进化系统**,而非把 VGM 当固定 oracle,是相对已有 VGM-based RL(WoW、Wow-1、DreamDojo、RoboDreamer)与纯 LLM 自博弈的明确差异化;(2) **near-miss 失败挖掘**把"差一点成功"的样本作为最有信息量的 hard negative,针对性纠正了自博弈文献普遍"只学成功、丢弃失败"的偏差,是本文最扎实的贡献点;(3) CLS 启发的"白天 GRPO 探索 / 夜晚 DPO 巩固"结构清晰,消融证明交替(而非顺序)是稳定关键;(4) 数据效率论断有力——无标注种子图 + 自动课程,300~500 张即超越 25K 人工标注。

**与公开工作的关系**:方法学上把 GRPO(策略探索)与 DPO(偏好巩固)分别绑到"广度/深度"两个角色,并用多粒度二值奖励桥接高层语义与低层物理,思路上介于"VGM as world model / 神经模拟器"与"VLM planner 自我改进"两条线之间;规划器评测沿用 MIND-V 协议、对标 RoboAgent/WAP 等,模拟器对标 Wow-1-Wan/DreamDojo 等。

**开放问题与可能改进**:① 最重要的是补上**合成视频→真机策略**的下游验证,否则"造数据"的价值仅证明到视频与规划评测层;② 为互评闭环给出防崩塌的理论或诊断指标(如引入独立第三方判据、周期性真值校准);③ 打破 13 模板的封闭任务空间,探索开放词表/开放物体的任务发现;④ 将语义门与物理奖励从"规划器自评"解耦,降低 reward-hacking;⑤ 把课程 UCB 从难度标量扩展到多维能力画像,支撑更细粒度的能力增长追踪。总体是一篇在"用世界模型 + 规划器自造具身操作数据"方向上机制设计相当完整、失败利用视角新颖的工作,主要待补证据在于对真实操作策略的下游可迁移性。

## 参考

1. Ebert et al. *Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets.*（BridgeData V2,种子图与模拟器评测基准来源）
2. Wan et al. *Wan 2.2* 系列（Wan2.2-TI2V-5B,本文模拟器骨干)与 Bai et al. *Qwen* 技术报告(Qwen3-VL-4B,规划器骨干)。
3. Rafailov et al. *Direct Preference Optimization (DPO).*（夜晚巩固的偏好优化基础）与 Guo et al. *GRPO*（白天在线探索的策略优化基础）。
4. Yang et al. *EB-ALFRED / EB-Habitat*(规划器评测基准)与 Zhang et al. *MIND-V*(评测协议)。
5. Chen et al. *Hierarchical Fine-Grained Preference Optimization for Physically Plausible Video Generation* (NeurIPS 2026);Kumaran et al./McClelland et al. *Complementary Learning Systems (CLS)* 理论(双阶段设计的认知科学依据)。
