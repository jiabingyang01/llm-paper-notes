# CUPID：用影响函数策展你的机器人所钟爱的数据

> **论文**：*CUPID: Curating Data your Robot Loves with Influence Functions*
>
> **作者**：Christopher Agia, Rohan Sinha, Jingyun Yang, Rika Antonova, Marco Pavone, Haruki Nishimura, Masha Itkina, Jeannette Bohg
>
> **机构**：Stanford University；University of Cambridge；NVIDIA Research；Toyota Research Institute
>
> **发布时间**：2025 年 06 月（arXiv 2506.19121，v2 于 2025 年 09 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.19121) | [PDF](https://arxiv.org/pdf/2506.19121)
>
> **分类标签**：`data-curation` `influence-function` `imitation-learning` `diffusion-policy` `closed-loop-attribution`

---

## 一句话总结

CUPID 把机器人模仿学习的数据策展重新定义为"哪些示范对策略闭环期望回报 $J(\pi_\theta)$ 贡献最大"的因果归因问题，用影响函数推导出一个可由少量 rollout 估计的 REINFORCE 式"性能影响"（performance influence）分数来给示范打分，据此过滤有害示范或挑选高价值新示范；在 RoboMimic 上用不到 33% 的策展数据就训出 SOTA 扩散策略，在真机 Franka 任务上把成功率相对基线提升约 38%，并能识别鲁棒策略、剥离虚假相关。

## 一、问题与动机

机器人模仿学习的策略性能与示范数据的质量与构成紧密耦合，但"单条示范究竟如何影响下游闭环成功/失败"一直缺乏精确刻画。既有的机器人数据策展主要依赖两类启发式：

- **任务无关的质量度量**（如 DemInf 用互信息离线估计示范质量）——隐含假设"人为设计的质量指标"与"策略下游表现"对齐，但作者反复观察到二者并不一致，甚至比随机选样更差；
- **性能相关的启发式**（如 Demo-SCORE 训练分类器区分成功/失败 rollout 的状态）——需要同时观测成功与失败，且易被 rollout 分布中的虚假相关误导，无法在训练数据与策略行为间建立强因果联系。

核心难点在于**目标错配（objective mismatch）**：策略通过监督式行为克隆（BC）训练，但评估却在闭环环境交互中进行——任务成功取决于一长串顺序决策，且测试时没有真值动作标签。因此作者主张：要正确策展，必须把训练数据**因果地**连接到策略的闭环回报，而非只看 BC 训练损失或表面质量。

作者形式化了两类策展任务：

- **Task 1（Filter-$k$，数据过滤）**：从训练集 $\mathcal{D}$ 中移除 $k$ 条冗余/有害示范以最大化性能，$S^\star = \arg\max_{S\in\Xi_k^-} J(\pi_\theta)\ \text{s.t.}\ \theta=\arg\min_{\theta'}\mathcal{L}_{\text{bc}}(\theta';\mathcal{D}\setminus S)$；
- **Task 2（Select-$k$，数据选择）**：从新采集/预采集的候选集 $\mathcal{H}$ 中挑选 $k$ 条最有价值示范加入 $\mathcal{D}$。

## 二、核心方法

### 2.1 从影响函数到"性能影响"

标准影响函数（Koh & Liang）估计:对训练样本 $z$ 的损失加一个无穷小权重 $\epsilon$ 后，某测试性能度量 $f$ 的一阶变化：

$$\Psi_{\text{inf}}(\hat{z},z) := \frac{df(\hat{z};\theta)}{d\epsilon}\bigg|_{\epsilon=0} = -\nabla_\theta f(\hat{z};\theta(\mathcal{D}))^\top H_\theta^{-1}\nabla_\theta \ell(z;\theta(\mathcal{D}))$$

其中 $H_\theta$ 是训练损失的 Hessian。除以 $-\tfrac{1}{n}$ 即得留一（leave-one-out）影响、加一（add-one-in）影响。

**用大白话说**：影响函数不用真的重训模型，就能预测"如果我把某条数据的权重稍微调大/调小，我关心的那个指标会往哪个方向变、变多少"，本质是拿 Hessian 逆把"训练损失的梯度"和"测试指标的梯度"对齐点乘。

作者把这一思想搬到闭环性能上。先把 BC 损失按轨迹分组 $\ell_{\text{traj}}(\xi;\pi_{\theta'}) := \tfrac{1}{H}\sum_{(s,a)\in\xi}\ell(s,a;\pi_{\theta'})$，然后定义一条示范 $\xi$ 的**性能影响（Definition 1）**为期望回报 $J(\pi_\theta)$ 对该示范权重 $\epsilon$ 的导数：

$$\Psi_{\pi\text{-inf}}(\xi) := \frac{dJ(\pi_\theta)}{d\epsilon}\bigg|_{\epsilon=0} = -\nabla_\theta J(\pi_\theta)^\top H_{\text{bc}}^{-1}\nabla_\theta \ell_{\text{traj}}(\xi;\pi_\theta)$$

**用大白话说**：这就是在问反事实——"如果训练时把这条示范的分量抬高一点点，我的策略实际跑起来的成功率会涨还是跌？"它把测试度量从"某个标注预测的损失"换成了"策略的闭环期望回报"。

### 2.2 关键难点与分解（Proposition 1）

直接算 $\Psi_{\pi\text{-inf}}$ 有两个障碍：(1) 性能影响归因的是一整条时间序列示范的"结果"，而经典影响函数只对单个标注预测；(2) $J(\pi_\theta)$ 依赖未知的转移动力学与奖励，无法直接对其求梯度。

作者先定义**动作影响（Definition 2）**——训练对 $(s,a)$ 对某测试状态-动作对 $(s',a')$ 的对数似然 $\log\pi_\theta(a'\mid s')$ 的影响：

$$\Psi_{a\text{-inf}}((s',a'),(s,a)) := -\nabla_\theta \log\pi_\theta(a'\mid s')^\top H_{\text{bc}}^{-1}\nabla_\theta \ell(s,a;\pi_\theta)$$

这个量给定策略权重与训练示范即可直接算。随后用策略梯度（REINFORCE）的对数导数技巧证明 **Proposition 1**：性能影响可分解为各动作影响按轨迹回报 $R(\tau)$ 加权的期望和：

$$\Psi_{\pi\text{-inf}}(\xi) = \mathbb{E}_{\tau\sim p(\tau\mid\pi_\theta)}\left[\frac{R(\tau)}{H}\sum_{(s',a')\in\tau}\sum_{(s,a)\in\xi}\Psi_{a\text{-inf}}\big((s',a'),(s,a)\big)\right]$$

**用大白话说**：一条示范对成功率的贡献，等于"它对策略在真实 rollout 里每一步动作的影响"逐步累加，再乘上那条 rollout 到底成功（$R{=}1$）还是失败（$R{=}-1$）。成功 rollout 里被这条示范强烈支持的动作，就给它加分；失败 rollout 里被它支持的动作，就给它减分。这样只需跑几条 rollout、看结果、算动作影响，就能估出性能影响（Algorithm 1）。

### 2.3 策展规则

用一阶 Taylor 近似把留一/加一影响写成 $\Psi^{\text{out}}_{\pi\text{-inf}}(\xi) := -\tfrac{\Psi_{\pi\text{-inf}}(\xi)}{|\mathcal{D}|}$、$\Psi^{\text{in}}_{\pi\text{-inf}}(\xi) := \tfrac{\Psi_{\pi\text{-inf}}(\xi)}{|\mathcal{D}|}$，则最优策展就是排序取 top-$k$：

- **Task 1（Filter）**：$S^\star_{\text{out}} = \arg\text{top-}k\{\Psi^{\text{out}}_{\pi\text{-inf}}(\xi^i):\xi^i\in\mathcal{D}\}$
- **Task 2（Select）**：$S^\star_{\text{in}} = \arg\text{top-}k\{\Psi^{\text{in}}_{\pi\text{-inf}}(\xi^i):\xi^i\in\mathcal{H}\}$

这本质上构造了一个线性 datamodel。此外作者补充一个**奖励无关的质量分数（Eq. 6）** $\Psi_{\text{qual}}$，用动作影响分数中的离群/噪声程度惩罚"含噪声运动"的示范；取凸组合 $\alpha\Psi_{\pi\text{-inf}}+(1-\alpha)\Psi_{\text{qual}}$，$\alpha{=}1$ 即 **CUPID**，$\alpha{=}1/2$ 即 **CUPID-QUALITY**。

### 2.4 扩散策略的实现细节

对扩散策略（Diffusion Policy），$\nabla_\theta\log\pi_\theta$ 因迭代去噪不可直接求。作者遵循 Zheng et al. 的做法，用一个**标注无关的平方替代损失** $\ell_{\text{square}}(s,a;\pi_\theta):=\mathbb{E}_{\epsilon^i,i}[\lVert\epsilon_\theta(\sqrt{\bar\alpha_i}a+\sqrt{1-\bar\alpha_i}\epsilon^i,s,i)\rVert^2]$ 同时替换对数似然与损失项，并用 Gauss-Newton 近似 Hessian $H_{\text{square}}$；再用 **TRAK**（随机投影 $d{=}4000$ + Gauss-Newton Hessian，单 checkpoint $C{=}1$）高效估计动作影响，把高维动作输出坍缩为标量以省算力。这套估计框架对影响估计技术的选择是解耦的，可即插即用更精确的数据归因方法。

## 三、实验结果

**设置**：3 个仿真 RoboMimic 多人（MH）任务（Lift/Square/Transport，各 300 条示范）+ 3 个真机 Franka FR3 任务（Figure-8 打结、TuckBox 塞箱、Bookshelf 抽书，各 120–160 条）。策略统一用卷积版 Diffusion Policy，真机额外用 $\pi_0$（Physical Intelligence 的 VLA，LoRA 微调）。Filter-$k$ 从随机 $\sim2/3$ 到 $\sim1/3$，Select-$k$ 从随机 $\sim9/10$ 到 $\sim4/10$。仿真用 $m{=}100$ rollout、报 500 次 rollout 的成功率；真机用 $m{=}25$ rollout。基线：DemInf（互信息）、Demo-SCORE（分类器）、Success Similarity（自定义相似度）、Random、Oracle（享有真值质量标签的上界）。

**仿真 RoboMimic（关键发现）**：DemInf 按真值质量策展出的数据集"质量最高"，但 CUPID 训出的策略成功率却持续匹配或超过 DemInf——**人类感知的示范质量并不对应下游成功**。CUPID-QUALITY 在 5 个设置中的 3 个超过 Oracle；在最难的 Transport MH 上，用不到原始 300 条示范的 33%、且模型参数量仅 10%，成功率甚至高于官方 Diffusion Policy。

**真机 Franka 过滤（Task 1，成功率来自 Fig. 5 / Fig. 12 饼图）**：

| 任务（过滤比例） | Demo-SCORE | DemInf | CUPID-QUALITY | CUPID | Oracle |
|---|---|---|---|---|---|
| Figure-8（filter 66%，混合质量） | 56% | 64% | **80%** | 72% | 84% |
| TuckBox（filter 66%，多策略） | N/A | 0% | 4% | **84%** | 88% |
| Bookshelf（filter 50%，虚假相关） | 44% | 36% | 20% | **84%** | 96% |

**真机 Franka 选择（Task 2，成功率来自 Fig. 13 / Fig. 14 饼图）**：

| 任务（选择比例） | Demo-SCORE | CUPID-QUALITY | CUPID | Oracle |
|---|---|---|---|---|
| Figure-8（select 33%） | 36% | **76%** | 72% | 84% |
| TuckBox（select 33%） | N/A | 16% | **88%** | 92% |

- **Figure-8（混合质量）**：CUPID 相对基线策略平均（过滤+选择）提升约 38%；此设置下质量项有用，CUPID-QUALITY 进一步增强。
- **TuckBox（多策略/分布漂移）**：数据集含 2:1 的"滑动:抓放"策略，测试时改变箱子质量分布使滑动失效。基线策略只学到滑动，漂移下 100% 失败。Demo-SCORE 因需同时观测成功/失败而失效；DemInf 与 CUPID-QUALITY 误把高方差的"抓放"当低质量而过滤掉，退化为脆弱的滑动策略。CUPID 无需观测成功，只把失败因果归因到影响它们的示范，保留鲁棒的抓放策略，成功率 84%–88%，逼近 Oracle。
- **Bookshelf（虚假相关）**：白背景与"水平抽书"运动虚假共现，基线仅 44% 成功（错误地在白背景下水平抽书导致上方书本掉落）。CUPID 识别出"因果驱动失败"的示范（白背景下的水平拉动）并再平衡数据集，达 84%，而 Demo-SCORE 把失败误归因到 rollout 中的相关物（叠放的书）而非因果因素。

**跨架构迁移到 VLA $\pi_0$（Fig. 7 / Fig. 15）**：用扩散策略策展的数据集去微调 $\pi_0$，成功率显著高于用全量未策展数据微调：

| 设置 | $\pi_0$ 微调(全量) | CUPID | CUPID-QUALITY |
|---|---|---|---|
| PI-0 Figure-8, Filter 66% | 48% | **92%** | 88% |
| PI-0 Figure-8, Select 33% | 20% | **84%** | 80% |
| PI-0 TuckBox, Filter 66% | 36% | **64%** | – |
| PI-0 TuckBox, Select 33% | 36% | **80%** | 20% |

说明策展分数在架构间可迁移，且**大规模多任务预训练并不自动让 VLA"忽略"低质量/脆弱行为，后训练仍需数据策展**。作者据此提出"用小型单任务策略策展数据、再喂给大型多任务模型"以降低算力的方向。

**消融（§7）**：Lift MH 仅需约 15% 数据即可最大化性能（高度冗余），Square MH 几乎移除任何示范都掉分（全都重要），Transport MH 策展收益最大——**策展收益取决于数据与任务的性质**。rollout 数量上，Lift/Square MH 在 $m\in[25,50]$ 收敛，Transport MH 需 $m\in[50,100]$，真机仅用 $m{=}25$，说明 top-$k$ 排序对 rollout 保真度要求低于精确估计。

## 四、局限性

- **策展量 $k$ 未定**：Filter/Select 都假定给定 $k$，如何自动确定该保留/剔除多少条仍是开放问题。
- **贪心选择忽略交互**：Eq. 4/5 的 top-$k$ 独立排序不考虑被选集合内示范间的相互作用（多样性/覆盖），策展集较大时收益会打折，需更高阶近似（类主动学习）。
- **大规模数据算力**：估计全数据集的性能影响，其算力与策略训练同量级；需借助 group effects、随机采样或粗粒度启发式缩小估计范围。
- **估计器方差**：REINFORCE 式估计在 rollout 很少时方差大，需引入方差缩减技术。
- **可解释性有限**：方法只策展已有示范，尚不能反过来指导后续采集（如告诉数据采集者该补什么）。
- **非完全模型无关**：同一批示范对不同架构的模型影响不同；$\pi_0$ 在为扩散策略策展的数据上表现只与扩散策略持平或略差。

## 五、评价与展望

**优点**：(1) 概念上把"数据质量"从任务无关的启发式，纠正为"对闭环回报的因果影响"，直击模仿学习的目标错配，理论优雅（影响函数 + 策略梯度分解，Proposition 1 是漂亮的桥接）；(2) 只需单个 checkpoint 与少量 rollout（真机 $m{=}25$）即可用，工程可行性高；(3) 同时覆盖过滤与选择两类任务，且能天然处理"无需观测成功、只从失败归因"的场景，这是相对 Demo-SCORE 的本质优势；(4) TuckBox/Bookshelf 两个真机案例把"识别鲁棒策略"和"剥离虚假相关"讲得很干净，实验说服力强；(5) 跨架构迁移到 $\pi_0$ 的结果，为"小策略策展、大模型消费"提供了实证。

**缺点与开放问题**：(1) 影响函数在深度非凸网络上的保真度本就有争议（作者引用的 Basu、Bae 等工作都质疑其脆弱性），扩散策略上还叠加了 $\ell_{\text{square}}$ 替代损失与 TRAK 单 checkpoint 的双重近似，理论上的性能影响与实际留一重训之间的误差缺乏定量验证；(2) 需要在线 rollout 才能拿到 $R(\tau)$，对真机是不小的成本，且与纯离线的 DemInf、并行工作 DataMIL 相比牺牲了"零 rollout"的便利；(3) CUPID 与 CUPID-QUALITY 谁更好高度依赖任务——混合质量任务质量项有益、多策略/虚假相关任务质量项有害，实践中如何先验地选 $\alpha$ 没有给出规则；(4) 所有真机成功率来自 25 次 rollout，统计置信区间偏宽。

**与公开工作的关系**：相较 DemInf（互信息、离线、只做过滤）与 Demo-SCORE（分类器、需成功/失败对比），CUPID 的差异化在于"用单一 checkpoint、对 rollout 分布的虚假相关鲁棒、且能自然扩展到选择新数据"。它把 NLP 数据归因（datamodels、TRAK、influence functions、LESS）系统性引入机器人策展，与并行的 DataMIL（用 datamodel 从多任务大数据里选数据）形成互补——后者聚焦离线多任务、前者聚焦单任务闭环回报。未来值得探索的方向包括：把贪心 top-$k$ 升级为考虑集合多样性的次模/组效应优化、用方差缩减降低 rollout 预算、以及把归因信号反哺到主动数据采集闭环。

## 参考

1. Koh & Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017.（影响函数的机器学习基石）
2. Hejna et al. *Robot Data Curation with Mutual Information Estimators (DemInf).* arXiv 2502.08623, 2025.（离线互信息质量策展，主要对比基线）
3. Chen, Lessing, Liu, Finn. *Curating Demonstrations using Online Experience (Demo-SCORE).* arXiv 2503.03707, 2025.（分类器式在线策展基线）
4. Park et al. *TRAK: Attributing Model Behavior at Scale.* ICML 2023.（本文影响估计的核心工具）
5. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR 2023.（本文统一使用的策略架构）
