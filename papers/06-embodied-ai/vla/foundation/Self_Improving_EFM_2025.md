# Self-Improving EFM：自我提升的具身基础模型

> **论文**：*Self-Improving Embodied Foundation Models*
>
> **作者**：Seyed Kamyar Seyed Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Igor Mordatch
>
> **机构**：Generalist；Google DeepMind（论文脚注注明项目主体工作于 2024 年 4 月完成于 Google DeepMind，一作后转入 Generalist）
>
> **发布时间**：2025 年 09 月（arXiv 2509.15155）
>
> **发表状态**：NeurIPS 2025（论文首页标注 "Appearing in the Conference on Neural Information Processing Systems (NeurIPS 2025)"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.15155) | [PDF](https://arxiv.org/pdf/2509.15155)
>
> **分类标签**：`具身基础模型` `Self-Improvement` `在线强化学习` `steps-to-go奖励塑形` `PaLI/RT-2` `行为泛化`

---

## 一句话总结

借鉴 LLM"预训练 → SFT → RL"的后训练范式,本文给具身基础模型(PaLI-3B/RT-2 架构)加了第二阶段在线 Self-Improvement：用同一个"steps-to-go"(离目标还差几步)回归头同时导出奖励塑形函数和成功检测器,免去人工奖励工程,让一名操作员即可监督多台机器人自主刷任务;真实 LanguageTable 上仅追加约 3% 的自采数据就把成功率从 62–63% 推到 87–88%,并且是首个证明"预训练基础模型 + 在线 Self-Improvement"组合能让策略学会训练集里完全没见过的新技能(BananaTable 任务,63%→85%)的工作。

## 一、问题与动机

- 现状：把基础模型直接微调成低层控制策略(RT-2、Octo、OpenVLA、π0 等)已成为具身基础模型(Embodied Foundation Model, EFM)的主流路线,但这些工作的后训练几乎清一色停留在行为克隆(behavioral cloning, BC)这一监督学习阶段。
- 对照 LLM 领域：LLM 后训练通常分两步——监督微调(SFT)之后再做强化学习(RLHF/RLVR),且 RL 阶段已被反复证明能显著超越 SFT 的天花板。机器人领域至今没有类比物。
- 阻碍机器人 RL 落地的核心是奖励工程：手工设计奖励需要反复试错和"打补丁"式修正来堵住 reward hacking 的漏洞;即便设计出了完美的奖励定义,在真实世界里*测量*这个奖励(reward instrumentation)本身也需要大量传感器/位姿估计工程。当任务集合越铺越广,人工奖励设计完全不可扩展。
- 核心思路：与其手工设计奖励,不如从基础模型自身的知识里"蒸馏"出一个数据驱动的奖励函数与成功检测器,使其继承基础模型在网络规模预训练中获得的鲁棒性与泛化性,从而把这套 RL 后训练机制变成通用、免奖励工程的流程。

## 二、核心方法

整体框架分两阶段(Figure 1),底座统一用 PaLI-3B 视觉语言模型(Chen et al., 2022/2023),动作按 RT-2(Brohan et al., 2023)方式离散化为 token 序列。

### 2.1 Stage 1：监督微调(SFT)

给定模仿学习数据集 $\mathcal{D}$,可采样元组 $(o_t, a_t, g_{t'}) \sim \mathcal{D}$,其中 $g_{t'}$ 是同一条轨迹里未来某时刻($t \le t'$)达成的目标/事件。两个训练目标联合优化：

$$\mathcal{L}_{\text{BC}}(\text{EFM}) = -\mathbb{E}_{(o_t,a_t,g_{t'})\sim\mathcal{D}}\big[\log p^{\text{EFM}}_{\text{action}}(a_t \mid o_t, g_{t'})\big]$$

$$\mathcal{L}_{\text{steps-to-go}}(\text{EFM}) = -\mathbb{E}_{(o_t,a_t,g_{t'})\sim\mathcal{D}}\big[\log p^{\text{EFM}}_{\text{steps-to-go}}(t'-t \mid o_t, g_{t'})\big]$$

**用大白话说**：第一项就是标准的目标条件行为克隆——看图看目标,学人类怎么动。第二项是新增的辅助任务——让模型同时学会"看当前画面和目标,预测还要走多少步才能完成",这个能力单独训练时看似无用,但正是打开 Stage 2 大门的钥匙。

### 2.2 Stage 2：Self-Improvement(在线 RL)

先定义"期望剩余步数"值函数

$$d(o,g) := \mathbb{E}_{p^{\text{EFM}}_{\text{steps-to-go}}(\text{steps-to-go}\mid o,g)}[\text{steps-to-go}]$$

在线 RL 用的奖励函数直接取相邻两步 $d$ 值之差：

$$r(o_t,a_t,o_{t+1},g) := d(o_t,g) - d(o_{t+1},g) \qquad (2)$$

**用大白话说**：如果这一步动作让模型觉得"离目标更近了"(预测的剩余步数变少了),就给正奖励;如果动作让情况变糟(剩余步数变多),就给负奖励。完全不需要人工定义任何任务专属的奖励,标量信号直接从模型自己对"进度"的理解中产生。

成功检测同样复用 $d(o,g)$，不需要额外训练二分类头：

$$\text{success}(o,g) := \mathbb{1}[d(o,g) \le s] \qquad (3)$$

其中 $s$ 是一个很小的步数阈值。论文发现这个基于回归的成功判据在低数据量下也非常鲁棒,比显式训练一个成功二分类目标更可靠。

**Self-Improvement 循环**(Algorithm 1)：固定一份 Stage 1 checkpoint 专门用来算奖励和判断成功,策略本身从 Stage 1 checkpoint 初始化后开始迭代——采样指令 $g$、执行当前策略采集轨迹(直到成功检测器判真、达到最大步数或人工终止)、按式 (2) 计算蒙特卡洛回报 $R_t=\sum_{i=t}^{T}\gamma^{i-t}\cdot r(o_i,a_i,o_{i+1},g)$、把 $(o_t,a_t,g,R_t)$ 存入 replay buffer,凑够 $N\times B$ 条数据后做 $N$ 步 REINFORCE 更新：

$$-c \cdot R_t \cdot \log p^{\text{EFM}}_{\text{action}}(a_t \mid o_t, g) \qquad (4)$$

更新完清空 buffer,进入下一轮。全程使用同策略(on-policy)、无数据复用,论文的解释是这样能同时避开"死亡三角"(deadly triad)里的两个顶点——自举(bootstrapping)和离策略学习(off-policy),只留下函数逼近这一个顶点,从而获得更稳定的训练。实现细节：$\gamma=0.9$，$c=5\text{e-}2$，每轮 $N=16$ 次策略更新，Stage 2 batch size 为 Stage 1 的一半(64 vs 128)且冻结 ViT 部分只微调其余参数。

### 2.3 为什么这个奖励是"天然塑形"的:理论直觉

设 $\mu$ 为产生模仿数据集的策略(如遥操作时的"人类策略"),定义一个虚拟的稀疏奖励 $-\mathbb{1}[o_t \text{ 满足 } g]$，其对应的(未折扣)价值函数恰好是 $V^\mu(o,g) = -d(o,g)$。把它代入式 (2) 可以做望远镜式(telescoping)展开：

$$r(o_t,a_t,o_{t+1},g) = V^\mu(o_{t+1},g) - V^\mu(o_t,g) = \underbrace{(1-\gamma)\cdot V^\mu(o_{t+1},g)}_{\text{core reward}} + \underbrace{\big[\gamma\cdot V^\mu(o_{t+1},g) - V^\mu(o_t,g)\big]}_{\text{reward shaping}} \qquad (5)$$

**用大白话说**：这正是 Ng et al. (1999) 意义下的 potential-based reward shaping——奖励里天然内嵌了"在数据集策略 $\mu$ 擅长的状态下给更高奖励"的成分。也就是说,Self-Improvement 训出来的策略会朝着"比 $\mu$ 更高效地达成目标"的方向优化,但同时被隐式地正则化,倾向于停留在 $\mu$ 熟悉、擅长的状态空间区域内——这是一种和常见的 KL-to-reference 正则化机制不同、但效果类似的"别跑太偏"约束。

进一步利用式 (5) 简化蒙特卡洛回报的望远镜求和：

$$R_t = \Big[(1-\gamma)\sum_{i=t}^{T}\gamma^{i-t}V^\mu(o_{i+1},g)\Big] - \underbrace{V^\mu(o_t,g)}_{\text{baseline}}$$

**用大白话说**：$V^\mu(o_t,g)$ 项相当于 REINFORCE 里天然自带的一个 baseline,能降低策略梯度估计的方差。当 $\gamma\to0$ 时 $R_t \approx V^\mu(o_{t+1},g)-V^\mu(o_t,g)$，近似单步策略提升;当 $\gamma\to1$ 时则鼓励策略走一条全程都在 $\mu$ 高价值区域内的轨迹。

## 三、实验结果

统一底座为 PaLI-3B,动作头等价于 RT-2 策略。共两种机器人本体(LanguageTable 2D 平面推方块、Aloha 双臂 14 自由度精细插拔)、真实与仿真场景。

**1)仿真 LanguageTable**(数据集 181,020 条人采轨迹、78,623 条不同指令;子采样 10%/20%/80%,Stage 2 只在 Block2Block 子任务上做)：

| 数据集规模 | Stage 1 (BC) | + Stage 2 Self-Improvement | 额外数据量 |
|---|---|---|---|
| 10% | 基线 | 提升 ≥1.5× | < 2% 额外 episode |
| 20% | 基线 | 提升 ≥1.5× | < 2% 额外 episode |
| 80% | 基线 | 提升 ≥1.5× | < 2% 额外 episode |

10% 数据 + 1% Self-Improvement 额外 episode 训出的策略,显著超过用 20% 甚至 80% 数据单纯做 BC 的策略。摘要中给出的示例数字：同一 LanguageTable 设定下,追加 10% 机器人时间做 Self-Improvement 把成功率从 45% 提到 75%,而把模仿数据量扩大 8 倍只能把成功率从 45% 提到 60%。

**2)真实世界 LanguageTable**(20% 与 80% 两种数据规模,4 个机器人工位,全程仅 1 名人类操作员负责复位,不提供任何标签或成功信号,每组真实实验约耗时 20 小时)：

| 设定 | Stage 1 (BC) 成功率 | + Stage 2 (追加约 3% Block2Block episode) | 对比 |
|---|---|---|---|
| 20% 数据 | ~62–63% | ~87–88% | 超过 80% 数据纯 BC 策略 |
| 80% 数据 | ~62–63% | ~87–88% | —— |

即"20% 模仿数据 + 3% Self-Improvement 数据"的总经验量,效果远超"80% 模仿数据"纯 BC 训练的策略,体现了 1 对多(1 人监督多机)相对遥操作 1 对 1 采集的巨大效率优势。

**3)仿真 Aloha 双臂插拔任务**(70 维动作、精细插销入孔;模仿数据集 5K/10K/15K episode,Stage 2 只在 5K、10K 上做)：5K(模仿)+2.5K(Self-Improvement)组合训出的策略,超过 10K 纯模仿数据(RT-2 等价基线),并可比肩 15K 纯模仿数据的策略。

**4)预训练消融**(仅在仿真 LanguageTable 上做,对比三种奖励模型初始化)：

| 奖励模型初始化 | 表现排序 | 备注 |
|---|---|---|
| PaLI(完整多模态联合预训练) | 最优 | 在 20% 数据下即优于 Uni-PaLI 在 80% 数据下的表现 |
| Uni-PaLI(视觉、语言模型分别单模态预训练,无联合微调) | 中等 | 低数据量时差距更明显 |
| Scratch(随机初始化) | 最差 | 跨随机种子方差极大,低数据(10%/20%)下几乎无法带来有意义提升 |

同一套消融也验证了 Stage 1 的 BC 策略本身：从 Scratch/Uni-PaLI 初始化训出的 BC 策略也显著弱于 PaLI 初始化,故消融聚焦在 Stage 2(策略仍用 PaLI Stage 1 checkpoint 初始化,只替换奖励计算所用的 checkpoint)。

**5)泛化能力**(全文最具特色的部分)：

- Real2Sim 域迁移：用 80% 真实 LanguageTable 数据训 Stage 1,Stage 2 却在*仿真* LanguageTable 环境里做 Self-Improvement。仅追加约 3% 目标域(仿真)episode,成功率从 ~22% 提升到 ~59%,相当于在目标域用 80% 模仿数据做 BC 的水平;换成 Uni-PaLI 奖励模型则迁移明显变慢,再次印证联合多模态预训练是迁移能力的关键。
- BananaTable(全新技能获取)：把 LanguageTable 桌面上的方块全部替换成一根香蕉,要求策略把香蕉推到棋盘的 8 个方位之一(如"move the banana to the top center of the board")。模仿数据集里从未出现过香蕉,也没有"桌上没有方块"的场景,策略、奖励模型、成功检测器完全靠 PaLI 基础模型自身的泛化能力驱动。用 2 个机器人工位跑约 8 小时 Self-Improvement,成功率从 ~63% 提升到 ~85%;定性上策略学会了从香蕉中段或两端推(而不是随意位置推),以避免因香蕉细长几何形状导致的原地打转。作者强调这不同于 RT-2 展示的"语义泛化"(新语境下执行相同动作模式),而是真正的"行为泛化"(behavioral generalization)——学会了训练数据中不存在的新动作策略。

## 四、局限性

- **技能边界标注是瓶颈**：steps-to-go 目标天然适合扩展为分层控制(用同一个进度估计器给子任务分段打密集奖励),但可扩展的 episode/子技能边界标注非常昂贵,论文只在 Future Work 部分讨论,未做实验验证。
- **OOD 失败状态缺乏监督**：奖励模型完全基于成功轨迹为主的模仿数据集训练,对失败后的恢复轨迹(不在数据集支撑集内)缺乏监督,可能给出错误的塑形奖励;论文承认这是数据驱动奖励方法的通病。
- **底座 VLM 从未见过机器人数据的预训练阶段**：本文所用的 PaLI 主干在预训练期完全没接触过具身/机器人数据,如何设计兼顾"保留视觉-语言泛化知识"与"注入物理推理先验"的预训练课程仍是开放问题。
- **On-policy REINFORCE 无数据复用,计算效率低**：为了训练稳定性主动放弃了离策略方法的数据复用优势,如何扩展到更大模型规模上的高效离策略变体是明确的未来方向。
- **观测到过优化会导致性能倒退**：论文明确提到把 Self-Improvement 训练超过性能峰值后成功率会下降,提示需要更好的停止准则或自适应正则化机制,但本文未给出解决方案。
- **真实世界 Aloha 上的 Self-Improvement 从未完整跑通**：受限于 Stage 2 远程推理服务器的网络延迟无法满足 Aloha 平台所需的 10Hz 控制频率(附录 J),虽然验证了 Stage 1 BC 策略在真实机器人上表现合理,但完整的真实世界 Self-Improvement 实验因项目时间限制(一作中途转往 Generalist)未能完成。
- **"全指令"泛化实验被排除**：附录中提到曾尝试把 Self-Improvement 从 Block2Block 子任务扩展到 LanguageTable 全部指令类型,观察到正向迁移(单任务上的 Self-Improvement 能带动其它任务成功率),但因部分机器人工位在实验期间被操作员误设置,数据被污染,且时间受限未能重跑,故未计入正式结果。

## 五、评价与展望

**优点**：本文最大的巧思是用同一个 steps-to-go 回归头"一鱼两吃"——同时导出奖励塑形函数和成功检测器,从而绕开机器人 RL 里最痛的两个工程难题(奖励设计、成功判定)。式 (5) 的理论推导把这个看似启发式的奖励设计严格证明为相对于数据集策略 $\mu$ 的 potential-based reward shaping,这不仅给出了直觉解释,也说明了该正则化机制比常见的 KL-to-reference 更通用(可套用到任意价值函数,不局限于 steps-to-go)。消融实验(Scratch vs. Uni-PaLI vs. PaLI)设计干净,清楚地把"预训练"这个笼统说法拆解为"是否经过联合多模态微调"这一具体变量,量化贡献做得比同类工作扎实。BananaTable 实验是全文说服力最强的证据,首次让人信服"预训练基础模型 + 在线自我提升"这个组合能够产生真正意义上的行为泛化,而不只是把已学动作迁移到新语境。

**与其他公开工作的关系**：相比 Code-as-Rewards 一类方法(Eureka、L2R 等用 LLM 写奖励代码),本文是纯数据驱动而非规则驱动,优势是不需要一个 ground-truth 的成功检测器去闭环指导 LLM 修改奖励代码,但代价是奖励质量完全依赖 steps-to-go 回归的标定质量与数据集覆盖率。相比 RoboCat(Bousmalis et al., 2023)用 hindsight relabeling + BC 自举式改进策略,本文采用真正的 on-policy RL(REINFORCE)加学习到的奖励,机制上更接近传统 RL,但论文自己也引用 Ghugare et al. (2024) 指出"把 rollout relabeling 当作策略提升"存在已知失败模式,提示两条路线各有隐患,谁更稳健仍待更大规模的横向对比。方法论上也直接承接了 Hartikainen et al. (2019) 的 dynamical distance learning 和 Hejna et al. (2023) 的 offline steps-to-go + 加权 BC,本文的增量贡献在于把这套"学习到步数距离"的思路整体搬进 online RL + 大规模预训练基础模型的框架,并第一次系统验证了预训练规模对该框架成败的决定性作用。

**开放问题与可能的改进方向**：(1) 长时程、多阶段任务上的层级化 Self-Improvement(利用 steps-to-go 做技能切分)仍是空白,自动化的技能边界发现是关键难点;(2) 如何在不牺牲训练稳定性的前提下引入数据复用(off-policy)以降低真实机器人时间成本,是走向更大规模部署的必经之路;(3) 论文观测到的"过优化导致性能倒退"现象目前缺乏理论刻画,自适应正则化或早停准则的研究可能是重要的后续方向;(4) 真实世界高自由度双臂平台(如 Aloha)上完整的 Self-Improvement 闭环尚未被验证,其对控制频率、推理延迟的工程要求可能是复现该方法的现实门槛。

## 参考

- Brohan et al., 2023 — *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*(本文 Stage 1 BC 策略架构直接对标的基线)
- Chen et al., 2022/2023 — *PaLI / PaLI-X*(本文所用的 3B 参数视觉语言基础模型底座)
- Zhao et al., 2023; Aldaco et al., 2024 — *ALOHA / ALOHA 2*(本文第二个机器人本体实验平台)
- Lynch et al., 2023 — *Interactive Language*(LanguageTable 任务与数据集来源)
- Ma et al., 2023 — *Eureka: Human-Level Reward Design via Coding Large Language Models*(Code-as-Rewards 代表工作,本文在 Related Work 中对比的另一条奖励获取路线)
- Bousmalis et al., 2023 — *RoboCat: A Self-Improving Foundation Agent for Robotic Manipulation*(同样冠以"self-improving"之名、但基于 hindsight relabeling + BC 的对比工作)
