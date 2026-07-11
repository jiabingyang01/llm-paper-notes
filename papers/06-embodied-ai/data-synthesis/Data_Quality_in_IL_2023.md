# Data Quality in IL：模仿学习中的数据质量(动作散度与转移多样性)

> **论文**：*Data Quality in Imitation Learning*
>
> **作者**：Suneel Belkhale, Yuchen Cui, Dorsa Sadigh
>
> **机构**：Stanford University
>
> **发布时间**：2023 年 06 月（arXiv 2306.02437）
>
> **发表状态**：未录用（预印本）——PDF 页脚标注 "Preprint. Under review."
>
> 🔗 [arXiv](https://arxiv.org/abs/2306.02437) | [PDF](https://arxiv.org/pdf/2306.02437)
>
> **分类标签**：`imitation-learning` `data-curation` `distribution-shift`

---

## 一句话总结

本文首次从**分布偏移(distribution shift)**的视角把模仿学习的"数据质量"形式化为两个基本属性——**动作散度(action divergence)** 与 **转移多样性(transition diversity)**,理论上证明分布偏移被"按剩余时长加权的动作散度之和"上界约束(Thm 4.1),实证发现:在专家数据里注入**系统噪声(转移多样性)** 能显著提升 BC 成功率(Square 上以 $\sigma_s{=}0.2$ 采集比低噪声更鲁棒,评测噪声下仍达 ~82%),而注入**策略噪声(拉大动作散度)** 在小数据下严重伤害性能;并揭示业界奉为圭臬的"状态多样性"其实与成功率不相关(robomimic 真人数据 Table 1)。

## 一、问题与动机

- 在监督学习里,模型越来越大越能"吞下"脏数据,数据质量与筛选被淡化;但在机器人离线学习尤其是**模仿学习(IL)** 里,根本没有互联网级数据,高质量数据集是刚需。
- IL 策略在测试时会因动作预测的**复合误差(compounding errors)** 陷入分布偏移:一步小误差把策略带到训练里没见过的状态,策略无法恢复,最终任务失败。以往两条路线:
  - **算法中心(algorithm-centric)**:改算法去容忍分布偏移(model-based IL、更强的状态/动作表征、时序抽象动作等)。
  - **数据中心(data-centric)**:改数据采集过程,但几乎都只盯着"最大化**状态覆盖/状态多样性**"(shared control 的 DAgger 系、噪声注入的 DART、主动查询等)。
- 作者的核心观察:**同一个 IL 算法在不同数据集上表现差异巨大**,说明缺一套"数据质量"的形式化度量。而以往"只看状态覆盖"的做法忽略了**动作(即专家本身)** 在数据质量中的角色。本文主张:好数据的完整定义应当是"让学到的策略在测试时尽量留在分布内",即最小化分布偏移,这需要同时刻画状态分布与动作分布,以及二者随时间的相互作用。

## 二、核心方法

### 预备:把 BC 写成前向 KL

数据集 $\mathcal{D}_N=\{\tau_1,\dots,\tau_N\}$ 含 $N$ 条专家轨迹,每条是状态-动作对序列。标准行为克隆(BC)最大化似然:

$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}_N|}\sum_{(s,a)\in\mathcal{D}_N}\log\pi_\theta(a\mid s)$$

它等价于在专家状态分布 $\rho_{\pi_E}$ 下最小化前向 KL:$\mathbb{E}_{s\sim\rho_{\pi_E}}[D_{KL}(\pi_E(\cdot\mid s),\pi_\theta(\cdot\mid s))]$(常数项是专家熵)。策略 $\pi$ 的状态访问分布逐步展开为

$$\rho_\pi^t(s') = \int_{s,a}\pi(a\mid s)\,\rho(s'\mid s,a)\,\rho_\pi^{t-1}(s)\,ds\,da$$

**用大白话说**:BC 只在"专家走过的状态"上拟合动作;但测试时策略走的是"自己会走到的状态" $\rho_{\pi_A}$,这两个分布不一致就是分布偏移的病根。而下一时刻走到哪(上式)由三样东西决定:策略动作 $\pi(a\mid s)$、环境转移 $\rho(s'\mid s,a)$、以及上一步的状态分布 $\rho_\pi^{t-1}$——这正对应下面三个因子。

### 数据质量的定义(Eq. 6)

作者把数据集质量定义为"用算法 $A$ 学到的策略 $\pi_A$ 相对专家的负分布偏移":

$$Q(\mathcal{D}_N;\pi_E,A) = -D_f\big(\rho_{\pi_A}(s),\,\rho_{\pi_E}(s)\big),\qquad \pi_A = A(\mathcal{D}_N)$$

**用大白话说**:数据质量不能孤立地谈,它同时取决于**专家 $\pi_E$、数据量 $N$、学习算法 $A$** 三者。同一批状态,不同专家动作/不同算法会得到完全不同的最终策略访问分布,因而质量不同。把"改专家/筛数据以适配算法"称为 **data curation**。

### 属性一:动作散度(Action Divergence)

定义为学到策略与专家策略的距离 $D_f(\pi_A(\cdot\mid s),\pi_E(\cdot\mid s))$,来源包括算法动作表征不匹配、样本不足,以及**专家本身**(多峰、暂停、抖动等次优性)。

**Theorem 4.1(核心界)**:在离散状态/动作、horizon $H$ 下,

$$D_{KL}(\rho_{\pi_A},\rho_{\pi_E}) \;\le\; \frac{1}{H}\sum_{t=0}^{H-1}(H-t)\,D_{KL}^{\,s\sim\rho_{\pi_A}^t}\big(\pi_A(\cdot\mid s),\pi_E(\cdot\mid s)\big)$$

**用大白话说**:整条轨迹的分布偏移,被"逐步动作散度按剩余时长 $(H-t)$ 加权求和"卡住上界。越早期(t 小)的动作误差权重越大,会在后续被复合放大——这与 Ross & Bagnell 的复合误差直觉一致。关键陷阱:界里的期望取在 $\rho_{\pi_A}^t$(**策略自己访问的状态**)上,而 BC 只在 $\rho_{\pi_E}^t$ 上压散度,二者错位正是 BC 的软肋。推论:更有表现力的动作空间/更一致的专家动作 → 更低动作散度 → 更高质量。

### 属性二:转移多样性(Transition Diversity)

即"某状态-动作下,下一状态转移的多样性"(系统噪声)。

**Lemma 4.2**:若在专家支撑集内动作散度被 $\beta$ 控制,则策略访问分布下的期望散度被拆成"支撑集内 $\le\beta$"与"支撑集外(无界项)"两部分。含义:好算法($\beta$ 小)下,应尽量让**策略访问的状态与专家数据支撑重叠**(压掉无界项),但**不能以抬高动作散度为代价**去做。与其盲目最大化状态多样性,不如让专家去调另外两个能扩大访问分布的因子:**系统噪声**与**初始状态方差**。

**Definition 1 + Theorem 4.3(有限数据下的覆盖)**:定义"下一状态覆盖概率" $P_S(s;N,\epsilon)$。若转移为高斯 $\rho(s'\mid s,a)=\mathcal{N}(\mu(s,a),\sigma^2 I)$,则

$$P_S(s;N,\epsilon) = 1 - \Big(1 - \operatorname{erf}\!\big(\tfrac{\epsilon}{2\sigma}\big)^{d}\Big)^{N}$$

($d$ 为状态维数)。**用大白话说**:即便策略完美,系统噪声 $\sigma$ 越大,覆盖概率越低;但**加数据量 $N$ 对覆盖的正向作用远强于降 $\sigma$**——所以"噪声稀释覆盖"只在数据量很小时才是问题。**Theorem A.1** 进一步松弛"完美策略"假设,证明系统噪声带来的覆盖增益,反而能**抵消/盖过**学到策略噪声的坏处,即"适度加系统噪声可对学习有益"(这为 DART 的噪声注入现象提供了理论解释)。

### 对数据采集的启示(Sec. 4.3)

- **动作一致性(Eq. 7)**:压低专家动作熵以减动作散度 $\min_{\pi_E}\mathbb{E}_{s\sim\rho_{\pi_A}}[\mathcal{H}(\pi_E(\cdot\mid s))]$——采一致、别多峰。
- **系统噪声(Eq. 8)**:鼓励高系统熵路径但把总状态熵压在阈值 $\gamma$ 下:$\max_{\pi_E}\mathbb{E}[\mathcal{H}(\rho(\cdot\mid a,s))]\ \text{s.t.}\ \mathcal{H}(\rho_{\pi_E}(s))\le\gamma$。
- **状态多样性**:是动作散度与系统噪声的**下游产物**,不是越大越好;若以牺牲动作一致性换覆盖,反而更差。
- **horizon 长度**:同样是下游量,真正该压的是"跨时间聚合的动作散度 + 转移多样性"。

## 三、实验结果

两组实验:(1) **Data Noising** 在脚本策略(scripted policy,人手设计保证成功)上加高斯噪声,消融各属性;(2) **Data Measuring** 在真人数据上量化度量。环境:**PMObstacle**(2D 质点绕障到点)、**Square**(robomimic 的方形螺母套桩 3D 机械臂),BC 用 MLP。系统噪声 $\sigma_s$ 加到动力学、策略噪声 $\sigma_p$ 加到专家动作。

### 实验一:噪声消融的三条结论

| 现象 | 高数据区 | 低数据区 |
|---|---|---|
| **系统噪声(转移多样性)** | 训练时加系统噪声普遍产出**最好**模型,但评测时噪声越大越差(Square 超 $\sigma_s{\approx}0.3$ 后因覆盖太稀而下滑) | 效果同向但更夸张 |
| **策略噪声(拉大动作散度)** | 因附带状态多样性,仅**略差**于同量系统噪声 | **显著更差**——少量噪声样本无法还原无偏专家,动作散度真正伤人 |
| **系统 + 策略噪声** | 固定 $\sigma_s{=}0.03$ 后,模型对追加策略噪声变得**很鲁棒** | 同样鲁棒:转移多样性可抵消次优专家带来的动作散度 |

代表性数字(成功率 %,括号为标准误):

| 设置 | 低训练噪声行 | 中等训练噪声行 |
|---|---|---|
| **PMObstacle 1000 demos, 系统噪声**(Table 2) | scripted 对角 100 / 100 / 99 / 96 | 训练 $\sigma_s{=}0.04$:100 / 100 / 99.3 / 96.7(评测噪声下仍高) |
| **Square 200 ep, 系统噪声**(Table 5) | 训练 $\sigma_s{=}0.05$:55.7 → 7.3(评测噪声下崩) | 训练 $\sigma_s{=}0.2$:67.9 / 68.7 / **82.0** / 74.3 / 50.3(最鲁棒) |
| **Square 50 ep, 策略噪声**(Table 8) | $\sigma_p{=}0.005$:32.7 / 30.3 / 18.0 / 7.0 / 5.7(低数据下策略噪声毁灭性) | $\sigma_p{=}0.01$:61.3 / 59.0 / 48.3 / 29.7 / 19.3 |

### 实验二:真人数据的度量(Table 1,robomimic Square/Can)

度量指标:**Action Variance**(Eq. 9,近邻状态聚类后每维动作方差,越高=动作越不一致)、**Horizon Length $H$**(越长=轨迹越拖)、**State Similarity**(Eq. 10,平均簇大小占比,是状态多样性的反面)。数据分档:Proficient Human(PH)与多真人 Better/Okay/Worse。

| 指标(Square) | PH | Better | Okay | Worse |
|---|---|---|---|---|
| 成功率 | 58 | 36 | 12 | 2 |
| 数据量 $N$ | 200 | 100 | 100 | 100 |
| Action Variance | 0.073 | 0.062 | 0.099 | 0.061 |
| Horizon $H$ | 150 | 190 | 250 | 350 |
| State Similarity | 8.2e-5 | 1.8e-4 | 1.7e-4 | 1.2e-4 |

| 指标(Can) | PH | Better | Okay | Worse |
|---|---|---|---|---|
| 成功率 | 96 | 56 | 40 | 22 |
| Action Variance | 0.051 | 0.066 | 0.079 | 0.063 |
| Horizon $H$ | 115 | 140 | 180 | 300 |
| State Similarity | 1.0e-4 | 2.1e-4 | 2.4e-4 | 2.0e-4 |

要点:成功率下降常伴随**动作方差↑与 horizon 长度↑**;但 Worse 档动作方差反而偏低,原因是其多峰行为(抓螺母不同部位)是"长期"效应,**单步动作方差度量抓不住**(印证 Thm 4.1 需长时视角);而 **State Similarity(状态多样性)始终与成功率不相关**,再次否定"多样性=质量"。

## 四、局限性

- **度量不完整,作者自承**:单步 Action Variance 无法刻画长时/多峰的动作散度,需要 Thm 4.1 意义下的跨时间聚合视角;真实数据集的数据质量"综合度量"仍很难。
- **理论假设强**:转移高斯、KL 散度、部分定理假设学到策略完美/确定性、线性简化动力学等,与真实高维视觉观测差距大。
- **实验规模小**:仅 2D PMObstacle 与 robomimic 仿真(Square/Can),无真机、无视觉大模型/多任务大数据验证。
- **可控性理想化**:结论建立在"专家能在轨迹级控制系统噪声、能降低动作熵"上,但现实里人类演示者 $\pi_E$ 很难按需调控;只能靠 prompting/反馈或事后 filtering 间接影响。
- 聚类度量依赖阈值 $\epsilon$ 的选择,不同 $\epsilon$ 下"覆盖/一致性"结论可能漂移。

## 五、评价与展望

**优点**:①**首次**把 IL 数据质量从"只看状态覆盖"升级为"状态×动作×时间"的分布偏移统一框架,概念贡献扎实;②Thm 4.1 的 $(H{-}t)$ 加权把"早期动作误差被复合放大"讲清楚,与 Ross & Bagnell 的复合误差、DAgger 的 covariate shift 一脉相承;③给出反直觉且可操作的结论——**注入系统(转移)噪声有益、注入策略噪声有害**,恰好统一解释了 DART(在专家动作上注噪反而鲁棒,本质是增加状态覆盖而非拉大动作散度)与 DAgger(on-policy 纠偏)的经验;④"状态多样性并非越大越好"对当下盲目扩数据的风气是有益提醒。

**缺点与开放问题**:①理论与实证之间存在断层——理论建议"筛/调专家 $\pi_E$",但论文没给出可落地的自动 curation 算法;②提出的可测度量被自己证伪(抓不住多峰),真正需要的"长时动作散度度量"仍空缺;③规模停留在仿真,VLA/大规模多任务时代下这些属性是否成立、如何高效估计,未验证。可能的改进方向:把动作散度沿轨迹做多步/子序列级度量(而非单步方差)、把该框架接入 behavior retrieval / 数据筛选(如 Du et al. 的 Behavior Retrieval、Nasiriny et al. 的检索式技能学习)、用更有表现力的动作表征(Diffusion Policy、动作分块)直接压低动作散度,以及研究"系统噪声上界 $\gamma$"如何随数据量自适应设定。

## 参考

1. Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger, AISTATS 2011) — 分布偏移/复合误差的经典分析,本文界的思想来源。
2. Laskey et al., *DART: Noise Injection for Robust Imitation Learning* (CoRL 2017) — 在专家动作上注入噪声,本文用"转移多样性/覆盖"给出理论解释。
3. Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation* (robomimic, 2021) — 本文 Square/Can 真人数据与度量实验的数据来源。
4. Du et al., *Behavior Retrieval: Few-Shot Imitation Learning by Querying Unlabeled Datasets* (RSS 2023) — 通过筛选无标注数据做 curation,与本文"筛专家/数据"主张互补。
5. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (RSS 2023) — 更有表现力的动作空间以捕捉多峰专家、降低动作散度。
