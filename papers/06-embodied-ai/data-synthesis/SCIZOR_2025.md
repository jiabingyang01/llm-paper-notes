# SCIZOR：一种面向大规模模仿学习的自监督数据清洗方法

> **论文**：*SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning*
>
> **作者**：Yu Zhang, Yuqi Xie, Huihan Liu, Rutav Shah, Michael Wan, Linxi "Jim" Fan, Yuke Zhu（Yu Zhang 与 Yuqi Xie 为共同一作）
>
> **机构**：The University of Texas at Austin；NVIDIA Research
>
> **发布时间**：2025 年 05 月（arXiv 2505.22626，v2 于 2025 年 9 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.22626) | [PDF](https://arxiv.org/pdf/2505.22626)
>
> **分类标签**：`数据清洗` `模仿学习` `自监督` `VLA` `去重`

---

## 一句话总结

SCIZOR 是首个**transition 级**（单个 state-action 对粒度）的自监督数据清洗框架:用"自监督任务进度预测器"识别并删除缺乏任务进展的**suboptimal（次优）** 帧,用"视觉+动作联合表征上的去重模块"删除**redundant（冗余）** 帧,无需任何人工标注即可扩展到百万级 Open-X-Embodiment 数据集,在 RoboMimic / Sirius-Fleet / OXE 上相比全量训练平均提升 15.4%,其中真机 Sirius-Fleet 绝对提升高达 32.9%(46.7%→79.6%)。

## 一、问题与动机

大规模模仿学习(尤其是训练 Octo、OpenVLA 这类 VLA 模型)依赖多机构、多硬件众包采集的海量轨迹,数据质量参差不齐。作者将低质量数据归纳为两类互补来源:

- **suboptimal(次优)** 数据:含碰撞、抖动、抓空、来回乱动等错误动作,会把错误行为"复制"进策略;
- **redundant(冗余)** 数据:同一技能在近乎相同场景下被反复采集,会稀释稀有但信息量大的样本,损害泛化。

已有机器人数据清洗方法存在两大不足:(1)多依赖昂贵的人工标注(如 [Mandlekar et al.] 对高/低质量打标),规模化不可行;(2)粒度太粗——要么在**dataset level(数据集级)** 重加权(Re-Mix、Octo 的 Magic Soup),要么在**trajectory level(轨迹级)** 整条丢弃(DemInf、Demo-SCORE),都忽视了单个 state-action 对的贡献差异。作者的关键观察是:一条轨迹里往往"失败抓取 + 成功恢复"混杂,理想的清洗应只切掉无信息/错误的片段(失败抓取),保留有用片段(成功恢复)。因此需要一个**免标注、可扩展、transition 粒度** 的清洗方法。

## 二、核心方法

模仿学习目标是最大化专家数据集 $\mathcal{D}_{expert}$ 上动作对数似然:

$$\theta^{\star} = \arg\max_{\theta}\ \mathbb{E}_{(s,a)\sim\mathcal{D}_{expert}}\big[\log \pi_{\theta}(a\mid s)\big]$$

SCIZOR 的目标是提炼 $\mathcal{D}_{expert}$——过滤掉次优与冗余样本。两个模块对每条轨迹**并行** 执行,各自删一部分,取并集得到清洗后的数据集。

### 2.1 基于进度估计的次优帧删除

核心直觉:**任务进度应随时间稳定增长**。由于没有真值进度标签,作者用"两帧之间真实经过的时间"当作进度的代理量,训练一个轻量模型去预测这个进度。

定义进度函数 $f: S_{i:i+T}\to T_p$,输入从 $i$ 到 $i+T$ 的子轨迹,预测其向任务完成推进的进度 $T_p$(以秒计,衡量任务被向前推进了多少)。把它与实际经过时间 $T$ 相比,子轨迹的次优分数为:

$$V_{i:i+T} = T - T_p$$

**用大白话说**:如果一段 2 秒的操作,模型判断"任务其实只前进了 0.3 秒的量",说明机器人在原地磨蹭/犯错(落后于进度表),这段就是次优。若 $T_p\approx T$ 则说明这段在踏实推进任务。

**从子轨迹分数到单样本分数**:为让每个 transition 拿到自己的分数,把子轨迹分数 $V_{i:i+T}$ 均摊到它包含的 $T$ 个 transition 上(各得 $\tfrac{1}{T}V_{i:i+T}$),再对覆盖某 transition 的所有子轨迹求和,得到聚合样本分数:

$$\hat{V}_i = \sum_{t=i-T}^{i}\frac{1}{T}\,V_{t:t+T}$$

**时间折扣**:让当前分数也反映"它将导向的未来次优",且影响随时间衰减:

$$V_i = \sum_{t=i}^{T}\gamma^{\,t-i}\,\hat{V}_t,\quad \gamma\in[0,1]$$

**用大白话说**:某个 transition 本身也许还行,但它把机器人带进后面一连串糟糕状态,那它也该被扣分;折扣因子 $\gamma$ 控制"往后看多远"。

**融合轨迹级质量**:最终清洗分数把局部分数 $V_i$ 与整条轨迹 $N$ 个 transition 的均值加权:

$$V_i^{\text{final}} = \alpha\cdot V_i + (1-\alpha)\cdot\frac{1}{N}\sum_{j=1}^{N}V_j$$

$\alpha\in[0,1]$(实验取 0.5)平衡"局部影响"与"整条轨迹的全局质量"——非专家操作者录的整条轨迹会因均值项被整体压低。训练时对每个样本算 $V_i^{\text{final}}$,清洗时删除分数高于阈值 $\epsilon_s$ 的 transition。

**进度预测器实现**:把每条轨迹切成 5 个等长时间 bin,进度预测被建成对时间 bin 的**分类**(比回归更鲁棒),bin 边界为 $[0,0.5),[0.5,1.0),[1.0,2.0),[2.0,5.0),[5.0,+\infty)$ 秒。网络用**冻结的 DINO-V2** 独立编码 $i$ 与 $i+T$ 两帧,取特征差(delta feature,强调任务相关变化、丢弃静态冗余)拼上 CLS token,过若干自注意力 Transformer 块,由 CLS token 经分类头预测 bin。全程用固定 2 秒时间窗(不同数据集控制频率不同,对应不同 transition 数)。

### 2.2 基于相似度的 state-action 去重

关键洞察:有些片段**视觉相似但动作不同**,反之亦然;纯视觉去重(如 SemDeDup)不适合机器人这类序列决策任务,必须**同时考虑视觉观测与动作**。

流程:把数据切成不重叠、跨度固定为 $T$(2 秒)的 state-action chunk,每个 chunk 均匀采 $N=8$ 帧 RGB,用预训练视频编码器 **Cosmos** 抽取 1D 视频特征 $z_v$,再把动作(delta end-effector pose)拼接上去,得到联合表征 $z_{v+a}$。对所有 chunk 做 **K-means 聚类**;在每个簇内计算两两余弦距离,每个 chunk 的相似度分定义为它到簇内任意其他 chunk 的**最小距离**;若某 chunk 的最大相似度超过阈值 $\epsilon_d$(即至少与另一个样本高度相似),标记为重复并过滤。

**用大白话说**:先按语义把相似片段聚到一堆,堆内那些"既看着像、动作也像"的近重复只留代表、删掉多余副本,从而拉平过采样的常见模式,给稀有样本让位。

**统一阈值**:在 RoboMimic 与 OXE_Magic 上做超参搜索得 $\epsilon_s=0.58,\ \epsilon_d=0.99$,并直接迁移到真机 Sirius-Fleet。这套统一阈值在各数据集上的实际删除比例:RoboMimic 29.6%、Sirius-Fleet 7.9%、OXE_Magic 15.8%、OXE_RT-X 15.8%、OXE_RT-1 9.7%。作者强调整条流程无需任何显式质量标签,策略架构与超参完全沿用各数据集公开参考实现,是**即插即用(plug-and-play)** 的。

## 三、实验结果

评测三类基准:大规模众包(OXE,Octo 训练,SIMPLER 中的 Pick Coke Can / Move Near)、专家水平不齐的仿真(RoboMimic MH,Can/Square,BC 策略)、人机协作混合的真机(Sirius-Fleet,Mutex 设置,1500 条 rollout,8 个任务,BC-Transformer-GMM)。基线含 Uniform(等量随机删)、DemInf(轨迹级互信息)、Re-Mix(数据集级 DRO 混合权重)。

**主结果(成功率 %)**:

| 数据集 | No Deletion | Uniform | DemInf | Re-Mix | SCIZOR |
|---|---|---|---|---|---|
| RoboMimic | 56.9 | 55.1 | **65.1** | — | 62.3 |
| Sirius-Fleet(真机) | 46.7 | 45.4 | 60.4 | — | **79.6** |
| OXE Magic Soup | 20.0 | 21.2 | — | — | **28.1** |
| OXE RT-X | — | — | — | 27.8 | **31.3** |

相比全量训练,绝对提升 RoboMimic +5.4、OXE_Magic +8.1、Sirius-Fleet +32.9,平均 **+15.4%**;比 Uniform 平均高 16.1%,比 Re-Mix 平均高 3.5%。真机 Sirius-Fleet 上比 DemInf 高 19.2%,作者认为混合来源(自主策略动作 + 人工纠正)导致质量分布不均,细粒度清洗优势明显;而 RoboMimic 因数据已被显式划分为三档轨迹质量,轨迹级 DemInf(65.1)反而更占优。

**消融:两模块的贡献**(成功率 %,删同等数量数据):

| 变体 | RoboMimic | Sirius-Fleet | OXE_Magic |
|---|---|---|---|
| 仅次优删除 | 60.9 ± 1.8 | 64.2 ± 2.6 | 25.3 ± 2.9 |
| 仅去重 | 48.3 ± 0.8 | 63.3 ± 6.9 | 22.1 ± 0.9 |
| SCIZOR(全) | **62.3 ± 1.6** | **79.6 ± 1.4** | **28.1 ± 3.3** |

单用任一模块均优于全量基线,但都不足以达到组合效果;次优删除通常比去重更有效,二者组合收益最大。

**消融:次优打分策略**(成功率 %):

| 变体 | RoboMimic Can | RoboMimic Square | OXE_RT-1 Pick | OXE_RT-1 Move |
|---|---|---|---|---|
| 无 transition/轨迹分混合 | 81.3 ± 0.6 | 36.0 ± 1.4 | 21.8 ± 7.9 | 12.4 ± 4.6 |
| 无时间折扣 | 79.6 ± 1.4 | 31.5 ± 5.5 | 20.7 ± 6.4 | 9.4 ± 1.4 |
| SCIZOR(全) | **87.3 ± 0.7** | **37.2 ± 2.5** | **30.9 ± 8.4** | **17.5 ± 1.0** |

去掉混合或折扣都会全面掉点,说明"把次优向前传播 + 局部/全局融合"两个设计缺一不可。

**与简单基线对比**(RoboMimic,%):速度过滤(删最慢 k% 的 end-effector 速度)Can 75.7/Square 30.1、零样本 VLM(GPT-4.1-small 打三档丢最低档)Can 80.4/Square 32.1、无清洗 79.0/34.8、SCIZOR **87.3/37.2**。VLM 在 Can 上有帮助但在需要精密装配的 Square 上反而有害,暴露其空间理解不足。

**质性分析**:随机抽 100 条被标次优的演示人工归类,主导失败模式为 Lagging(迟滞)、Manipulation Failure(抓取失败/掉物)、Pause/Stalled(停顿),表明进度分类器抓到的是语义上有意义的错误而非噪声。可控误差分析中 13/25 条人工标注的真实失败被成功预测。冗余端与保真端:随机抽 50 条完美专家演示,SCIZOR 均**未** 标出次优段;对恢复行为,54% 完全保留、31% 部分保留、仅 15% 被删,说明其能保住高质量与有用的"失败后恢复"数据。

## 四、局限性

- **去重表征较简单**:目前只是把动作特征与状态特征拼接,未来可用更具表达力或可学习的联合表征。
- **依赖大多数演示是好的**:方法建立在"一条轨迹里多数数据质量尚可"的假设上;若低质量演示占主导则失效,如何反过来利用低质量数据中的有用片段尚未解决。
- **线性进度假设**:假定任务进度线性、无停顿/重复;但真实任务(反复搅拌、等待烹饪)天然含此类子行为,需要更长历史、分层进度建模或多时间尺度进度模型。
- **阈值需搜索**:虽做到跨数据集统一阈值,但仍靠在两个仿真集上的超参搜索确定,缺乏自适应删除比例机制(作者也将其列为 future work)。

## 五、评价与展望

**优点**:(1)把 NLP/CV 里成熟的"数据清洗"范式(SemDeDup、DoReMi、Data Filtering Networks)真正落到机器人 transition 粒度,填补了此前只有 dataset/trajectory 级清洗的空白;(2)进度预测这个"用时间当进度代理"的自监督信号非常巧妙且免标注,配合 DINO-V2/Cosmos 这类现成表征几乎零额外标注成本,能扩到百万级 OXE;(3)真机 Sirius-Fleet 上 32.9 个点的绝对提升相当可观,且是即插即用、不改策略架构。

**缺点与开放问题**:(1)"次优 = 进度落后"的定义与其"线性进度假设"强绑定,对含合理停顿/往复的长程任务会误伤,这也是最根本的方法论局限;(2)去重用 K-means + 固定阈值,簇数、chunk 跨度(固定 2 秒)等对不同控制频率数据的敏感性缺乏系统研究;(3)与轨迹级 DemInf 的对比呈现"数据分布决定谁赢"的现象——RoboMimic 上 DemInf(65.1)优于 SCIZOR(62.3),说明细粒度清洗并非在所有分布上都占优,何时该用哪种粒度仍是开放问题;(4)Re-Mix 只在 OXE_RT-X 一个设置上比,证据面偏窄。

**展望**:与并行的机器人数据质量工作(DemInf 的互信息估计、Demo-SCORE 的在线 rollout 评分、DataMIL 的 datamodels)相比,SCIZOR 的独特价值在于"免在线交互 + transition 粒度"。一个自然的改进方向是把次优检测从"线性进度分类"升级为能显式建模子任务的分层/多尺度进度模型;把去重表征换成任务感知的可学习联合嵌入;以及研究自适应阈值以自动决定最佳删除比例。作为 VLA 预训练数据配方的一环,这类免标注、可组合的清洗器很可能与生成式数据增广、混合权重优化等手段叠加使用。

## 参考

1. Hejna et al., *Robot Data Curation with Mutual Information Estimators (DemInf)*, arXiv:2502.08623 — 最直接的轨迹级免标注清洗对比对象。
2. Hejna et al., *Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning*, arXiv:2408.14037 — 数据集级混合权重清洗基线。
3. Abbas et al., *SemDeDup: Data-Efficient Learning at Web-Scale through Semantic Deduplication*, arXiv:2303.09540 — 本文去重思路的纯视觉前身。
4. Octo Team, *Octo: An Open-Source Generalist Robot Policy*, arXiv:2405.12213 — 本文在 OXE 上的策略骨干与 Magic Soup 权重来源。
5. Chen et al., *Curating Demonstrations using Online Experience (Demo-SCORE)*, arXiv:2503.03707 — 依赖在线 rollout 的轨迹级清洗,与本文"免在线"形成对照。
