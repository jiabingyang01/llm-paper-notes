# HumanScale：自我中心人类视频在具身预训练中可超越真机数据

> **论文**：*HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining*
>
> **作者**：Juncheng Ma\*, Jianxin Bi\* , Yufan Deng, Xuanran Zhai, Bingyi Kang, Enze Xie, Wojciech Matusik, Tat-Seng Chua, Daquan Zhou†（\* 共同一作，† 通讯）et al.
>
> **机构**：PKU、NUS、MIT、UCSB、NVIDIA
>
> **发布时间**：2026 年 06 月（arXiv 2606.20521）
>
> **发表状态**：未录用（预印本）；代码将开源于 github.com/DAGroup-PKU/HumanNet
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.20521) | [PDF](https://arxiv.org/pdf/2606.20521)
>
> **分类标签**：`具身预训练` `自我中心人类视频` `数据源对比` `world-action model` `scaling law`

---

## 一句话总结

在架构、数据规模、后训练数据与评测协议**全部固定** 的受控对比下,本文首次给出结论:以自我中心(egocentric)人类视频作为具身预训练数据,在下游真机动作预测上不仅能替代、甚至能**超越** 等量的遥操作真机数据；同等 5000 小时预训练下 OOD 验证损失比无预训练基线低约 24%,真机执行的 in-distribution / OOD 成功率达 92.5% / 90.0%(而无预训练基线为 40.0% / 0.0%)。

## 一、问题与动机

- **具身基础模型想复刻 LLM 的数据-模型-算力联合 scaling,但数据瓶颈远比语言/视觉严峻。** 语言与视觉可从互联网被动采集,而具身数据必须由真实机器人、人类操作员、设计任务和受控环境主动生产。
- **主流预训练源是遥操作(teleoperated)真机轨迹。** 其优点是动作监督精确、与部署本体运动学对齐好；缺点是采集成本高(低成本 ALOHA 平台约 $20k/站,成本由持续人力与硬件主导,规模只随车队和预算线性增长)、公开总量受限(所有公开遥操作数据聚合仅约 $2\times10^4$ 小时)、场景与行为多样性窄。
- **自我中心人类视频提供了天然的"覆盖度"补充。** 第一人称日常活动视频规模大(可及约 $10^6$ 小时)、采集近乎零边际成本、天然覆盖富接触手-物交互与开放世界长尾行为；但它缺少动作标签,且与机器人存在 embodiment gap。
- **此前工作只把人类视频当作"视觉表征/运动重定向/跨本体先验"来间接利用,从未在匹配规模、匹配后训练协议下与真机数据做正面 head-to-head 比较。** 于是本文回答一个基本经验问题:在匹配规模(matched-scale)下,人类视频的多样性优势能否压过真机数据的运动学对齐优势?

## 二、核心方法

### 2.1 概念框架:coverage 由预训练供给,alignment 由后训练补齐

作者把预训练-后训练范式拆成两个正交诉求:预训练看重 **coverage**(对场景、物体、交互、行为的广覆盖,支撑可迁移表征),后训练看重 **alignment**(与目标本体、相机、任务分布对齐)。因此选预训练数据源的关键不是"哪种模态本质更好",而是"哪种模态在规模上供给更多覆盖度"。据此沿四个轴对比 egocentric 人类视频与遥操作真机数据:可及规模、采集成本、获取难度、多样性。得到的"分工"结论是:egocentric 视频统治预训练奖励的诸轴(规模、边际成本、运动/交互/场景多样性),其唯一短板 embodiment gap 恰好正是后训练用少量运动学对齐的真机数据就能修补的。

### 2.2 统一的 World-Action Model(WAM)

预训练与评测都基于一个自回归 world-action 模型,用 Mix-of-Transformers(MoT)架构把**视频动力学预测** 与**动作推断** 统一:给定文本与观测,预测未来视频与后续动作。其中 video expert 用 *Wan 2.2* 初始化,action expert 用插值初始化。全程固定后训练数据、算力预算与评测协议,使**预训练数据源成为唯一变量**。

### 2.3 两阶段协议(受控 distributional shift)

- **Stage 1 预训练(两个 5000 小时匹配集)**：
  - *Egocentric*：从 HumanNet 的自我中心部分筛选；用手部姿态重定向(hand-pose retargeting)估计每段的末端执行器位姿与夹爪状态,作为**伪动作标签**,放到与机器人相同的动作空间;提供背景、物体、交互技能的开放世界覆盖。
  - *Real-robot*：多本体轨迹,含精确末端执行器位姿与夹爪状态,聚合自多个真机数据集;运动学对齐但场景/任务受限。
- **Stage 2 后训练**：从 AgiBot World 精选 15 个操作任务,每任务 100 条专家示范,共 1500 条轨迹;背景与物体实例比预训练真机数据更丰富。
- **评测协议**：在 held-out 的 Stage-2 真机数据上报告 flow-matching 动作损失,分两个 split:
  - *Seen*（in-distribution）：15 个后训练任务的 held-out 轨迹,任务语义已见但物体实例/变体未见,测同分布内鲁棒性。
  - *Unseen*（out-of-distribution）：25 个**未参与后训练** 的任务,是主 OOD 泛化指标。

### 2.4 log-linear scaling law

对 egocentric 预训练小时数 $D$ 拟合下游最优动作损失:

$$\mathcal{L} = a - b\ln(D)$$

拟合结果:Seen split $\mathcal{L}=0.0094-0.0003\ln(D)$（$R^2=0.86$）,Unseen split $\mathcal{L}=0.0273-0.0008\ln(D)$（$R^2=0.94$）。

用大白话说:每把 egocentric 预训练数据翻一倍,下游动作损失就稳定下一个台阶,且到 5000 小时斜率仍明显为负(还没饱和)。Unseen 的斜率 $b$ 更大,说明预训练覆盖度对 OOD 泛化的杠杆比对同分布任务更大。

### 2.5 匹配规模下 egocentric 的信息密度更高

在匹配采样时长(各取 2 小时子集统计)下,egocentric 视频在多样性各轴全面碾压真机数据:

| 多样性指标 | 人类视频 | 真机数据 | 方向 |
|---|---:|---:|---|
| 动作空闲时间 idle fraction | 约 1% | 约 25% | 越低越好 |
| 归一化 jerk(运动平滑度) | 更低 | 更高 | 越低越好(人类轨迹更平滑) |
| 交互词汇量(unique verb-object pairs) | 2744 | 107 | 越高越好 |
| 视觉场景覆盖(unique scene terms) | 361 | 156 | 越高越好 |
| workspace / inter-session 空间分布 | 更宽、跨 demo 方差更大 | 集中于固定工位 | 越宽越好 |

由此推论:按小时匹配其实**低估** 了 egocentric 的优势——1 小时人类视频含远多、更干净的轨迹。作者在 100 小时配方下量化:egocentric 约含 45,000 条轨迹,而真机约 8,000 条(遥操作被长空闲和慢速手臂运动拖慢)。

## 三、实验结果

两个基线:**Wan 2.2**（无具身预训练）与 **LingBot-VA**（在 20k 小时真机数据上微调 Wan 2.2,强具身预训练基线）。所有损失取后训练全程验证曲线的最小值。

### Q1：egocentric 预训练随数据 scaling(动作损失,越低越好)

| 预训练小时 | Seen(ID)损失 | Unseen(OOD)损失 |
|---:|---:|---:|
| 100 h | 0.0080 | 0.0234 |
| 5000 h | 0.0067 | 0.0204 |
| vs Wan2.2 无预训练 | 低约 35% | 低约 24% |

损失随规模单调下降,log-linear 拟合 $R^2$ 达 0.86 / 0.94,到 5000 小时未饱和。

### Q2：egocentric vs real-robot 预训练(5000 小时匹配,动作损失)

| 预训练源 | Seen(ID)@5000h | Unseen(OOD)@5000h |
|---|---:|---:|
| Real-robot | 0.0071 | 约 0.0254(各规模几乎持平) |
| Egocentric | 0.0067 | 0.0204 |
| 相对差 | 基本相当 | egocentric 低约 20% |

关键现象:真机预训练在 Seen 上随规模稳步改善、逼近 egocentric,但在 Unseen 上**基本不随规模改善**（损失卡在约 0.025)。作者刻意让真机预训练任务与后训练/评测任务不相交以排除泄漏,因而这条平线反映的是"实验室受限采集的操作任务对真正未见任务迁移差",而非数据泄漏假象。

### 真机执行实验(AgiBot 双臂平台,3 任务:放杯、分拣果蔬、盖章)

平均成功率(vs Wan 2.2 无预训练基线):

| 预训练 | In-distribution | Out-of-distribution |
|---|---:|---:|
| Wan2.2(baseline) | 40.0% | 0.0% |
| Egocentric(ours) | **92.5%** | **90.0%** |

egocentric 预训练在分布迁移下仅掉 2.5 个百分点,而基线从 40% 崩到 0%(掉 40 点)。摘要中"52.5% 和 90% 更高的成功率"即指 ID 92.5%−40.0%=52.5 个百分点、OOD 90.0%−0.0%=90 个百分点的绝对差。此外在果蔬分拣任务上,egocentric 初始化收敛到的动作损失约为无预训练基线的 $1/2.4$（即低约 2.4 倍),video loss 亦更低。

## 四、局限性

- **规模封顶于 5000 小时。** 为做"匹配规模"对比,egocentric 预训练被人为限制到与公开真机数据可及量对齐的 5000 小时,远未触及人类视频约 $10^6$ 小时的真实规模;更大规模下的行为尚属外推。
- **只在 world-action model(WAN2.2 骨干)上验证。** 未在主流 VLA 家族与更广本体上确认结论,作者明确说 VLA 上的 scaling 仍在评估中,当前结论"encouraging but still preliminary"。
- **伪动作标签质量依赖手部重定向。** egocentric 的 end-effector/gripper 动作由 hand-pose retargeting 估计,其误差如何影响下游、以及与真机精确标签的系统性偏差,文中未做消融。
- **真机执行仅 3 个任务、单一双臂平台。** 成功率样本量小(如 40 次量级),OOD 只覆盖物体实例替换,未涉及更强的场景/技能级 OOD。
- **未拆解"是视频动力学预测还是伪动作监督在起作用"。** WAM 同时学视频与动作,论文未消融去掉视频 co-training 后 egocentric 优势还剩多少。

## 五、评价与展望

**优点。** (1) 问题干净、控制严格:把架构/规模/后训练/评测全部钉死,只留预训练数据源为变量,这类"matched-scale head-to-head"正是此前 EgoMimic、EgoScale、Being-H0 等工作缺失的一环,结论因此更可信。(2) 用 seen/unseen 双 split 分离 in-distribution 拟合与 OOD 泛化,揭示了一个有价值的非对称现象——真机预训练能提升同分布拟合却几乎不带来 OOD 泛化,而 egocentric 的收益主要落在 OOD,这与"coverage 决定泛化"的直觉自洽。(3) 首次给出 world-action 模型上 egocentric 预训练的 log-linear scaling 曲线($R^2$ 高、未饱和),对"该不该在昂贵真机采集前先评估数据质量"给出可操作指引。

**局限与存疑。** 5000 小时的天花板使"人类视频超越真机"的结论只在中等规模成立,真机数据规模一旦上到 π0 corpus(>10k h,专有)或 Being-H0.5(约 35k h)量级,对齐优势会否反超尚不可知。匹配"小时"而非匹配"信息量/轨迹数"对真机偏保守,这既是作者论点的加分项(说明差距被低估),也意味着比较口径本身可争议。伪动作标签噪声、单平台小样本真机评测,都让 90% OOD 成功率这类强数字需要更多复现。

**与公开工作的关系。** 本文站在 world-action model(WAM)一支:相较 LingBot-VA(先生成未来视频再自回归解码动作)、DreamZero(视频与动作单扩散联合去噪)、Fast-WAM(保留视频 co-training 但推理时跳过未来生成),本文不改架构而是把数据源当作研究对象。数据侧则延续 HumanNet(百万小时人类视频)、EgoScale(2 万小时、报告灵巧操作 log-linear scaling)、Being-H0/H0.5(手轨迹先验蒸馏)与 HumanEgo(几分钟人类示范可替代更长遥操作)的脉络,把"人类视频有用"推进到"在匹配规模上人类视频优于真机"。

**开放问题与可能改进。** (1) 把结论迁到 VLA 与多本体,验证优势是否在 foundation-model 规模持续;(2) 对伪动作标签质量做消融,并探索联合优化 retargeting 与策略;(3) 混合配比研究:egocentric 预训练 + 少量真机对齐的最优比例与课程;(4) 更强 OOD(新技能、新场景)与更多任务下的真机复现;(5) 拆解视频动力学 co-training 与伪动作监督各自的贡献。

## 参考

1. Deng & Zhou. *HumanNet: Scaling Human-Centric Video Learning to One Million Hours.* arXiv 2605.06747, 2026.（本文预训练数据来源)
2. Li et al. *Causal World Modeling for Robot Control (LingBot-VA).* arXiv 2601.21998, 2026.（主要具身预训练基线)
3. Zheng et al. *EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data.* arXiv 2602.16710, 2026.（egocentric log-linear scaling 先例)
4. Luo et al. *Being-H0 / H0.5: Vision-Language-Action Pretraining from Large-Scale Human Videos.* arXiv 2507.15597 / 2601.12993.（人类视频动作先验)
5. Wang et al. *HumaNego: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos.* arXiv, 2025.（多样性统计与"少量人类示范可替代遥操作"参照)
