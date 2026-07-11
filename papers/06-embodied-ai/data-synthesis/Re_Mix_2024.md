# Re-Mix：面向大规模模仿学习的数据配比优化

> **论文**：*Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning*
>
> **作者**：Joey Hejna, Chethan Bhateja, Yichen Jiang, Karl Pertsch, Dorsa Sadigh
>
> **机构**：Stanford University；UC Berkeley（Karl Pertsch）
>
> **发布时间**：2024 年 08 月（arXiv 2408.14037）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2408.14037) | [PDF](https://arxiv.org/pdf/2408.14037)
>
> **分类标签**：`数据配比` `DRO鲁棒优化` `Open-X-Embodiment` `模仿学习预训练`

---

## 一句话总结

把语言模型里的 group DRO 数据配比方法（DoReMi）迁移到机器人模仿学习，通过"策略最小化、配比最大化"的 min-max 优化,自动学出各数据集（domain）的采样权重;为解决机器人数据的动作空间异构、连续损失、易过拟合三大障碍,引入 per-domain 高斯归一化 + 动作离散化 + 激进早停;在 Open-X（RT-X 子集）上学到的权重比均匀采样平均高 38%、比人工专家配比平均高 32%,且仅用 25% 数据即可几乎无损保留性能。

## 一、问题与动机

大规模机器人数据集（如 Open-X-Embodiment,60+ 数据集、200 万+ 轨迹）正快速增长,但如何为通用机器人策略预训练**加权组合**这些异构子集,几乎没有系统方法。现状是纯靠人工直觉:

- RT-X 系列只从 OpenX 中挑了 12 个数据集,按专家经验手工赋权;
- Octo、OpenVLA 也是靠作者主观的"interestingness"选数据集和采样权重。

这类 ad hoc 策略需要大量领域知识和人工检查,无法随数据规模扩展。视觉/NLP 里成熟的数据筛选手段又难以直接搬来:n-gram 过滤不适用;CLIP/视觉 embedding 捕捉不到 episodic robot data 的时序与**动作**信息;基于检索的方法要预先知道目标任务,不适合"无目标任务"的通用预训练。

作者的核心问题:能否**自动**地为大规模机器人数据集学出各 domain 的混合权重,以最大化下游通用策略的表现?他们借鉴 DoReMi(LLM 领域用 group DRO 优化训练数据配比),但发现直接照搬到机器人上根本不 work——这正是本文要解决的。

## 二、核心方法

### 问题设定

给定演示数据集 $\mathcal{D}=\{\tau_1,\dots,\tau_n\}$,拆成 $k$ 个异构 domain $\mathcal{D}_1,\dots,\mathcal{D}_k$(可以是不同 embodiment 的数据集,也可以细到单条轨迹或单个场景)。标准行为克隆（BC）损失为:

$$\mathcal{L}_{\text{BC}}(\pi_\theta, \mathcal{D}) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[-\log \pi_\theta(a\mid s)\right]$$

目标是学一个权重向量 $\alpha \in \Delta^k$(k 维单纯形上的概率分布),使得按 $\alpha$ 加权训练出的模型在**所有 domain 上**都表现好。

> 用大白话说:就是给每个数据子集分配一个"采样多少比例"的旋钮,拧到一组值,让最终模型不偏科、每个来源的技能都学得会。

### 从朴素 DRO 到 excess loss

直接套用 group DRO(分布鲁棒优化）的目标是最小化最坏 domain 的损失:

$$\min_{\theta} \max_{\alpha \in \Delta^k} \sum_{i=1}^{k} \alpha_i \,\mathcal{L}_{\text{BC}}(\pi_\theta, \mathcal{D}_i)$$

> 用大白话说:$\alpha$ 专挑当前损失最高（最难拟合）的 domain 加码,逼着策略去啃硬骨头。

但纯用 BC 损失有个致命问题:损失绝对值高的 domain 不一定"值得多学"——可能只是里面有次优/噪声动作,本来就拟合不了。为此作者改用**excess loss**(超额损失）:把 $\mathcal{L}_{\text{BC}}(\pi_\theta,\mathcal{D}_i)$ 替换为策略相对一个参考模型 $\pi_{\text{ref}}$ 的差值 $\mathcal{L}_{\text{BC}}(\pi_\theta,\mathcal{D}_i) - \mathcal{L}_{\text{BC}}(\pi_{\text{ref}},\mathcal{D}_i)$。参考模型是在均匀采样（按 domain 大小加权）上训到收敛的。

> 用大白话说:excess loss 衡量的是"策略比参考模型还有多大进步空间"。既压低已经学好的 domain,也压低那些又难学又参考模型也学不好的 domain（说明数据本身有问题）;只有那些"策略还能追上参考模型"的 domain 才会被 $\alpha$ 加码。

### 机器人场景下 DRO 的三大障碍与对策

**1. 损失量纲不平衡（Unbalanced Losses）**。不同数据集动作空间、控制频率、单位（英寸 vs 米）差异巨大,导致某些 domain 损失量级天然偏大,被错误加码。对策:对**每个 domain 单独**做高斯归一化(而非 bounds/[-1,1] 归一化,后者不对齐各分布的矩）。作者用一个 toy 实验验证(Table 1):构造一个把 Bridge 动作换成随机噪声的 noise domain 和一个正常 bridge domain,噪声 domain 理论上无法拟合、应拿到大权重。结果 bounds 归一化下 $\alpha_{\text{noise}}=0.943$、$\alpha_{\text{bridge}}=0.057$(被 bridge 动作幅度偏小干扰,反了);高斯归一化下 $\alpha_{\text{noise}}=0.158$、$\alpha_{\text{bridge}}=0.842$(正确对齐后才合理)。

**2. 连续损失（Continuous Losses）**。DRO 大多用在离散分类的交叉熵上;而策略常用 L1/L2 连续损失,既无法表达多模态动作,又极易被动作 outlier 拉高损失估计。对策:对每个动作维度**离散化分箱（binning）**,参考模型和 DRO 阶段都用离散动作 + 交叉熵——足以估计"某 domain 动作可预测性",且对 outlier 鲁棒。

**3. 过拟合（Overfitting）**。机器人数据集小(约 10–100k 条,单个数据集甚至只有 100 条),高容量策略容易在每个数据点上把训练损失打到近 0。一旦参考模型对某 domain 过拟合、参考损失≈0,excess loss 退化成普通损失,$\alpha$ 失去意义。对策:对参考模型和鲁棒模型都做**激进早停**——选那个尚未对**任何** domain 过拟合(用训练/留出验证损失差衡量)的最新 checkpoint。

### Re-Mix 的四阶段流程

最终目标(离散策略 $\pi_\theta$):

$$\min_{\theta} \max_{\alpha \in \Delta^k} \sum_{i=1}^{k} \alpha_i \left[\frac{1}{|\mathcal{D}_i|} \sum_{(s,a)\in\mathcal{D}_i}\left(-\log \pi_\theta(a\mid s) + \log \pi_{\text{ref}}(a\mid s)\right)\right]$$

> 用大白话说:内层策略努力把每个 domain 的超额损失压低;外层 $\alpha$ 专给"还追得上参考模型"的 domain 加码,自动过滤掉平凡（如无 gripper 动作、动作只有几种取值）和无法拟合的 domain。注意这是**直接基于动作**筛选,不同于视觉/语言里只看 embedding 的方法。

- **Stage 1 动作预处理**:每个 domain 单独高斯归一化 + 分箱离散化。
- **Stage 2 参考模型训练**:在按大小加权的均匀混合上训练离散参考模型 $\pi_{\text{ref}}$,按验证损失早停选 checkpoint。
- **Stage 3 Group DRO**:用离散策略优化上式。$\alpha$ 用指数梯度上升更新一步,$\theta$ 按 domain 加权梯度下降一步,交替进行;训练步数与参考模型相同。
- **Stage 4 用权重训最终策略**:取整个训练过程中 $\alpha$ 的平均值 $\bar\alpha$,用它重新加权或**子采样**数据。关键设计:Re-Mix 只输出权重 $\alpha$,与策略训练**解耦**——DRO 阶段用离散动作,而最终策略换成**扩散头（diffusion policy）**,权重可复用于不同类型/更大规模的策略。

## 三、实验结果

**设置**:两个大数据集——Bridge V2(约 4.5 万条 WidowX 6-DoF 演示,按场景切成 32 个 domain）与 RT-X 用的 OpenX 子集(约 35 万条、11 个第三人称相机数据集）。最终策略用 ResNet-50 编码器 + 扩散头,训 40 万步;为让预训练策略能零样本迁移,额外用少量域内数据（3 个代表任务各 25 条,占 5% 权重）co-train。所有评测在真机上进行,每任务 10 次试验,累计 500+ 次真机试验。

**RT-X 混合数据上的主结果(Fig 1,WidowX + Franka 六个 OOD 任务成功率)**:

| 任务 | Uniform | Human（RT-X 专家配比） | Re-Mix |
|---|---|---|---|
| 平均 | 0.42 | 0.48 | **0.80** |
| Carrot to Rack | 0.3 | 0.7 | 1.0 |
| Fork to Rack | 0.3 | 0.4 | 0.8 |
| OOD Cup | 0.1 | 0.0 | 0.7 |
| Cube to Plate | 0.9 | 0.8 | 0.8 |
| Flip Bowl | 0.3 | 0.4 | 0.9 |
| Pen in Cup | 0.6 | 0.6 | 0.6 |

Re-Mix 比均匀采样平均高 38%、比人工专家配比高 32%(相对提升）。在 Bridge 全量数据上（Fig 2,Uniform 0.9 vs Re-Mix 1.0）差异不明显——作者认为在数据够全时配比不那么关键。

**学到的权重(Table 2,RT-X 混合内各 domain 的 $\alpha$)**,选取几个有代表性的:

| domain | Uniform | Human | Re-Mix |
|---|---|---|---|
| RT-1 | 40.9% | 26.8% | 42.5% |
| Kuka | 24.9% | 25.1% | **12.1%**（下调） |
| Bridge | 22.7% | 27.5% | 19.9% |
| Toto | 3.42% | 4.13% | **16.3%**（上调 >4×） |
| Cable Routing | 0.43% | 1.56% | 0.20% |
| Jaco | 0.81% | 1.95% | 0.39% |

Re-Mix 大幅**下调 Kuka**(该数据集是自动采集后按成功筛选的,动作质量可能偏低）,下调易拟合的小 domain（Cable Routing 无 gripper 动作、Jaco 只有 3 种动作取值);同时**上调 Toto** 逾 4 倍(其动作分布高度多模态、偏离高斯,较难拟合）。

**子采样能力（Fig 3,子采样到 25% 规模后训练的平均成功率）**:

| 数据集 | Uniform | 基线 | Re-Mix |
|---|---|---|---|
| RT-X 25% 子集 | 0.38 | 0.33（Human） | **0.80** |
| Bridge 25% 子集 | 0.50 | 0.22（SSP） | **0.99** |

用 Re-Mix 权重子采样到 25% 数据,RT-X 上仅掉 2.5% 成功率,而 Human 权重掉了 10% 以上;Bridge 上视觉聚类基线 SSP(Self-Supervised Prototypes,k-means 去冗余）表现很差(0.22),因为机器人轨迹对 CLIP 类视觉模型是 OOD,聚类与数据多样性不相关。子采样时 Re-Mix 保留 Berkeley UR5 的 72%、Kuka 仅 12%,而 Human 只保留 UR5 的 30%、Kuka 的 24%——人工权重不会主动剔除"无趣/无用"的数据。Bridge 更激进的 10% 子采样(Fig 5)下 Re-Mix 0.47 vs Uniform 0.13 vs SSP 0.37。

**消融(Fig 4,25% 子集设置下)**:

| 消融项 | 配置 A | 配置 B |
|---|---|---|
| 参考模型早停 | 早停 150K:**0.73** | 过拟合 200K:0.50 |
| 动作离散化 | 离散:**0.87** | 连续:0.10 |

参考模型晚停 50K 步就掉 15%+ 成功率;用连续动作代替离散动作估 $\alpha$ 会灾难性下降(0.87→0.10),因为连续损失拟合不了 outlier 和多模态动作。

**仿真验证(Table 3–4,RoboMimic NutAssemblySquare,300 条多操作员人类演示按 better/okay/worse 分 6 个 domain,用 Conditional UNet Diffusion Policy）**:

| 方法 | 50% 子采样 | 25% 子采样 |
|---|---|---|
| Re-Mix | **77/100** | **59/100** |
| Uniform | 53/100 | 39/100 |

Re-Mix 学到的权重把两个"better"操作员上调(Better1 从均匀的 9.6% 升到 22.8%),把"worse"下调（Worse2 从 23.7% 降到 12.7%),印证其确实在**按动作质量**做筛选。

## 四、局限性

- **评测覆盖窄**:真机评测只做了 WidowX 和 Franka 两种臂,受限于真机试验成本,难以在更多 embodiment/setup 上穷尽验证泛化。
- **异常动作分布被误上调**:Re-Mix 会给动作分布反常（如 Toto,严重多模态且偏斜）的数据集加码,虽然最终混合表现好,但这种上调未必是"想要"的,鲁棒优化对分布不规整的数据仍不够稳健。
- **计算成本高**:需要在全量数据上训策略**两遍**（一次参考模型、一次 Group DRO),才能算出权重;理想是"边训边配比"一遍出结果。
- **规模上限**:只在 Bridge V2 与 RT-X 两个数据集上验证,尚未扩到完整 OpenX(>200 万条）这种更大规模。
- 方法**解耦**带来复用便利,但也意味着 DRO 阶段用离散/简化策略估出的权重,不保证对最终扩散策略是最优的(存在 proxy 模型与目标模型的错配隐患）。

## 五、评价与展望

**优点**。第一,把 DoReMi 的 group DRO 配比范式**成功迁移到机器人模仿学习**,并非简单照搬,而是精准诊断出机器人数据特有的三大障碍（量纲不平衡、连续损失、小数据过拟合）并逐一给出对策——per-domain 高斯归一化、动作离散化、激进早停,这套"工程细节"才是让 DRO 在机器人上真正 work 的关键,消融也证明缺一不可（尤其离散化 0.87→0.10）。第二,excess-loss 的设计巧妙地把"值得学"从"损失高"中解耦出来,能自动识别并剔除低质/平凡数据(无 gripper、动作取值极少、成功筛选后的次优动作),这是纯视觉/语言 embedding 筛选做不到的——它直接看**动作**。第三,权重与策略解耦、可复用于扩散策略和更大规模,加上"25% 数据几乎无损"的子采样能力,实用价值高。

**与其他公开工作的关系**。它是 DoReMi(Xie et al., NeurIPS 2023)在机器人上的对应物,理论根基是 group DRO(Sagawa et al.);面向的正是 RT-X、Octo、OpenVLA 这些靠人工配比的通用策略——并直接击败了 RT-X 的专家配比(平均 +32%）。相较视觉里的 coreset/主动学习和 SSP 聚类去冗余,本文强调机器人数据对视觉预训练模型是 OOD,聚类筛选失效,凸显"按动作筛"的必要性。

**开放问题与改进方向**。(1) 双遍训练成本高,能否做成单遍在线配比(on-the-fly)是最直接的效率改进。(2) proxy 模型(离散 MLP）与目标模型（扩散头）错配:离散化估权重在动作精度要求高、多模态强的任务上是否会失真,值得研究更贴近目标策略的权重估计方式。(3) 对 Toto 这类异常分布的误上调暴露了高斯归一化假设的局限,或可引入更灵活的分布对齐（如按分位数/流模型归一化）或对"难度"与"质量"进一步解耦。(4) 权重是数据集/场景级的粗粒度;推进到轨迹级甚至(s,a)级的细粒度加权,可能进一步提升数据效率。(5) 论文只验证到 35 万条量级,扩到 200 万+ 的完整 OpenX、并与 embodiment 泛化联合分析,是最有价值的后续。

## 参考

1. Xie et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023.（本文直接借鉴的 LLM 配比方法）
2. Sagawa et al. *Distributionally Robust Neural Networks for Group Shifts.* arXiv 2019.（group DRO 理论基础）
3. Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models.* 2023.（数据来源与人工配比基线）
4. Walke et al. *BridgeData V2: A Dataset for Robot Learning at Scale.* CoRL 2023.（另一主实验数据集与策略架构来源）
5. Oren et al. *Distributionally Robust Language Modeling.* EMNLP-IJCNLP 2019.（DRO 用于配比的更早工作）
