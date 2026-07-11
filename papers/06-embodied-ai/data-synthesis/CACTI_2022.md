# CACTI：面向可扩展多任务多场景视觉模仿学习的框架

> **论文**：*CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning*
>
> **作者**：Zhao Mandi、Homanga Bharadhwaj、Vincent Moens、Shuran Song、Aravind Rajeswaran、Vikash Kumar
>
> **机构**：Columbia University；Carnegie Mellon University；Meta AI（第一作者工作在 Meta AI 完成）
>
> **发布时间**：2022 年 12 月（arXiv 2212.05711，v2 于 2023 年 2 月）
>
> **发表状态**：未录用（预印本）；项目页 cacti-framework.github.io
>
> 🔗 [arXiv](https://arxiv.org/abs/2212.05711) | [PDF](https://arxiv.org/pdf/2212.05711)
>
> **分类标签**：`视觉模仿学习` `生成式数据增广` `多任务多场景`

---

## 一句话总结

CACTI 把"如何低成本扩展机器人操作学习"拆成 **Collect–Augment–Compress–TraIn** 四个独立阶段：少量专家采集 + 用 Stable Diffusion 图像 inpainting 做语义级数据增广 + 冻结的预训练视觉表征压缩 + 在压缩表征上做单一多任务 BC 策略；真机 10 任务平均成功率约 30%（去掉 inpainting 增广仅约 11%），仿真 18 任务 100 布局下对未见布局的泛化成功率随增广规模从 14% 提升到 47%。

## 一、问题与动机

机器人操作要像 NLP/CV 那样靠大规模数据驱动,面临两个核心矛盾:

- **数据采集贵**:真机 teleoperation 费时费力、有安全与硬件成本;端到端 RL 需要在真实硬件上边探索边学,部署次优策略采数据既慢又危险。
- **多样性难覆盖**:要让单一策略泛化到大量物体、布局、场景,需要海量的物理资源去人工布置每一种变体。

作者的观点是:不要用一个单体(monolithic)的端到端 pipeline 硬扛,而是把"扩展机器人学习"这件事**按开销分解成可管理的模块**,让每个模块各自用最合适、最便宜的技术。由此提出 CACTI 框架——名字正是四阶段首字母:

- **Collect**:用任务专属专家采集**少量**域内(in-domain)演示;
- **Augment**:用**域外**(out-of-domain)生成式大模型把数据的视觉/语义多样性成倍放大;
- **Compress**:用预训练视觉骨干把原始图像压成低维 latent,便于高效训练;
- **TraIn**:在冻结的压缩表征上训练**单一**多任务多场景视觉策略。

关键 insight 是:数据增广发生在**数据集层面**,不需要额外的机器人操作工时;而"压缩"让策略训练可以用大 batch、短训练循环,从而把多样性转化为泛化能力。

## 二、核心方法

CACTI 在真机(Franka Emika Panda,平行夹爪,8 维关节位置控制,12.5 Hz)与仿真厨房(基于 Franka Kitchen 类环境)两套系统上分别实例化。四阶段如下。

### 2.1 Collect：小规模域内专家采集

- **真机**:人类通过 kinesthetic teaching(手把手拖动机械臂)完成任务,记录每一步的关节位姿与末端信息。好处是"录一次可回放多次"——正文称每个任务采 5 条演示、每条回放 20 次并每次重排非目标物体(附录另处报告为每任务 8 条、共 80 条专家轨迹)。10 个任务,例如拖动马克杯、抓放物体、开合烤箱。
- **仿真**:用 on-policy RL 算法 NPG(Natural Policy Gradient),基于状态输入 $s_t=\{robot, object\}_{pos,vel}$ 为**每个任务的每个布局**单独训练一个专家策略。18 个任务 × 100 个布局 = 1800 个策略(附录),再用 **90% 成功率阈值**筛掉未收敛者。这些专家训练便宜且高度可并行。

### 2.2 Augment：语义场景增广

目标是把原始数据集 $\mathcal{D}_\tau$ 放大成增广集 $\mathcal{D}'_\tau$,分**视觉增广**(改颜色/纹理/光照)与**语义增广**(改物体位置朝向、甚至加入新的人工物体)。

- **真机**:用开源 **Stable Diffusion** 做 zero-shot **inpainting**——指定图像中一块 mask 区域 + 文本 prompt(如 "a coffee mug on a white tabletop"),让扩散模型在该区域生成逼真的新物体,其余场景保持不变。由此在**零额外机器人工时**下获得大量语义变化的演示。此外还叠加 color jitter、回放时的 action noise(缓解 covariate shift)、以及手工重排 distractor 物体。
- **仿真**:由于专家 RL 策略只用状态输入、不看图像,可以在**回放同一条轨迹**时自由改变渲染的视觉属性(颜色/纹理/光照)而**不改变策略的物理效果**;并随机化 distractor 物体位姿、加入 action noise。对每个"任务×布局"组合渲染 50 条 episode(每条 50 步),最终得到 **45,000 条 episode** 的训练集。作者据此开源了 **CACTI-Sim-100** 基准(18 基础任务 × 100 语义变体 × 50 轨迹,附带仿真器代码可无限生成视觉增广)。

用大白话说:与其真的搬来一堆新杯子重新采数据,不如让扩散模型"P 图"把杯子画进去——动作标签不变、像素变了,数据集就便宜地膨胀出了语义多样性。

### 2.3 Compress：在互联网/域内数据上预训练的表征

把图像观测编码成低维 embedding,既降低下游策略学习难度,又便于跨场景泛化,还解耦了表征学习与策略学习。两条路线:

- **域外(out-of-domain)**:用 **R3M**(Nair et al. 2022,在 Ego4D 第一视角人类视频上用时序对比 + 视频语言对齐损失预训练)。真机上直接用其冻结表征,或在阶段 2 的域内数据上微调。
- **域内(in-domain)**:用 **MoCo** 自监督训练一个 ResNet-50。仿真数据量大足以从头训 MoCo;真机数据太少,R3M 从头训不动,故用域外 R3M 再拿域内数据微调。

### 2.4 TraIn：多任务多场景视觉策略学习

用阶段 3 的视觉骨干,在全部增广数据上训练**单一**目标条件策略 $\pi$(一个 MLP)。时刻 $t$ 的图像观测 $o_t$ 与任务名字符串分别嵌成 $z_t$ 与 $z_g$(真机用与 R3M 相同的 tokenizer 得到 768 维文本 embedding),拼接后送入策略 MLP,输出高斯动作分布:

$$\hat{a}_t = \pi_\theta([z_t, z_g]) + \exp(\sigma)\cdot z, \quad z \sim \mathcal{N}(0, \mathbf{1})$$

$$\mathcal{L}_{\text{BC}} = \|\hat{a}_t - a_t\|^2$$

用大白话说:策略就是"看一眼当前图像压缩向量 + 读一下任务描述向量",然后回归出专家当时的动作;$\exp(\sigma)\cdot z$ 只是给输出加一点可学习方差的高斯噪声,训练目标就是最朴素的均方 behavior cloning 损失。

仿真里任务条件用手工设计的 **43 维 context embedding**(含目标物体位姿 + 布局排布信息);拼上 1024 维观测 + 1024 维目标 + 机器人关节速度/位姿,得到 **2064 维**输入,输出 8 维动作分布。作者指出:把四阶段解耦、让视觉 IL 作为最后一环,相比单体式在线多任务 RL 更容易学目标条件行为——因为可以直接把每条增广轨迹的最后一帧当作 goal image。

## 三、实验结果

**真机 10 任务平均成功率(Fig 5)**：核心结论是 inpainting 增广至关重要。

| 变体 | 平均成功率 |
|---|---|
| Frozen MoCo（有增广） | ≈ 30% |
| Frozen R3M（有增广） | ≈ 27% |
| Frozen MoCo w/o Aug | ≈ 11% |
| FineTune MoCo w/o Aug | ≈ 15% |

去掉 inpainting 增广后,即便用域内数据微调也无法追平——增广带来约 **15–20 个百分点**的绝对提升,验证了"语义增广优于传统 color-jitter/random-crop 像素级增广"这一假设。作者还提到:完全冻结的域外表征几乎与域内冻结/微调表征打平,说明可以直接借用海量互联网图像/人类日常视频来预训练机器人视觉表征。

**仿真 18 任务(Table 1)**：对比在 10/50/100 个布局上训练,分别在"训练见过布局(Train)"与"未见布局(Heldout)"上评测成功率(%)。

| 表征 | Sim10 Train | Sim10 Heldout | Sim50 Train | Sim50 Heldout | Sim100 Train | Sim100 Heldout |
|---|---|---|---|---|---|---|
| State-based | 81.2±2.2 | 14.1±3.1 | 83.3±2.2 | 31.6±4.4 | 91.3±1.1 | 47.2±4.5 |
| Out-domain | 51.3±4.3 | 9.5±2.0 | 64.7±3.6 | 30.4±5.6 | 62.0±3.8 | 33.1±4.3 |
| In-domain | 88.7±1.2 | 18.8±2.9 | 72.1±2.8 | 30.2±5.2 | 75.9±2.6 | 38.4±4.7 |

核心趋势:**布局增广规模越大,对未见布局的泛化越强**。State-based 的 Heldout 从 14.1→31.6→47.2;In-domain 从 18.8→30.2→38.4;Out-domain 从 9.5→30.4→33.1。作者据此主张 Augment 阶段的强语义增广对分布外泛化是关键。作为对照,端到端 RL(Chen et al. 2021 的 randomized ensemble double Q-learning,吃像素输入跨所有任务/布局)**无法收敛,成功率 0%**;CACTI 在 18 任务 100 变体上训练集成功率约 62%(Out-domain Train)。

## 四、局限性

- **增广只改像素、不改物理**:inpainting/换纹理生成的"新物体"只存在于图像里,不参与仿真或真实的接触动力学;动作标签沿用原轨迹,存在"视觉变了但动作没变"的潜在语义错配,难以增广真正需要不同抓取/接触策略的物体几何。
- **绝对成功率仍低**:真机约 30%、仿真训练集约 62%、未见布局仅 30–47%,离可用还很远。
- **规模有限**:真机仅 10 任务、少量演示;仿真任务(18)与物体(8 类厨房物件)种类不算丰富。
- **人工成分**:真机 inpainting 的 mask 需人工指定、distractor 手工重排;仿真任务条件是手工设计的 43 维向量,不易迁移到新任务。
- **数字口径略有出入**:正文与附录在"演示条数""专家策略数(900 vs 1800)"上表述不完全一致,读者需以附录 + 45,000 episode 反推。

## 五、评价与展望

**优点**:CACTI 最有价值的贡献是把"用**冻结的域外生成式/表征大模型**放大域内机器人小数据"这条路线,在同一个框架里同时用于**数据增广**(Stable Diffusion inpainting)与**表征压缩**(R3M/MoCo),并首次验证了 zero-shot 用 SD inpainting 生成机器人训练数据在完全未见的实验室场景下依然可用。四阶段解耦的工程范式清晰,便于各模块独立替换升级;把最后一帧作为 goal image 复用增广轨迹的做法也很自然。它与同期 DALL-E-Bot(Kapelyukh et al. 2022,用 DALL-E 生成重排目标图)是"扩散模型进入机器人"的两条并行早期尝试,但 CACTI 关注的是**扩充训练数据**而非生成子目标。

**与其它公开工作的关系**:相比 RT-1/BC-Z 等靠海量真机遥操作堆数据的路线,CACTI 走的是"小数据 + 生成式放大"的省钱路线;相比纯 domain randomization,它用扩散模型带来的是**语义级**(换物体/换布局)而非仅参数级(换颜色/光照)的多样性;表征侧则延续了 R3M、Parisi et al.(2022)"预训练视觉模型对控制出奇有效"的发现,并进一步证明冻结域外表征可与域内表征打平。

**开放问题与可能改进**:(1) 当前增广是"改图不改物理",后续世界模型/可微仿真类工作可以让增广的视觉变化与动力学一致,从而生成需要不同动作的新样本;(2) 用现代文本到图像/视频扩散(可控生成、深度/几何约束的 inpainting)替代 2022 版 SD,并自动化 mask 与 prompt 生成,能进一步降人工;(3) 把 BC 换成更强的策略(diffusion policy、序列 Transformer)或引入少量在线交互纠错,有望突破 30% 的成功率天花板;(4) 把仿真手工 43 维条件替换为语言/图像目标,才能真正规模化到开放任务集。总体上,CACTI 是"生成式模型为具身数据引擎服务"这一方向早期、思路完整但结果尚粗糙的代表作,其框架价值大于其绝对指标。

## 参考

1. Nair et al., *R3M: A Universal Visual Representation for Robot Manipulation*, 2022(arXiv:2203.12601)——本文域外冻结表征来源。
2. Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*(Stable Diffusion), CVPR 2022——inpainting 增广所用生成模型。
3. Gupta et al., *Relay Policy Learning*, 2019(arXiv:1910.11956)——仿真厨房环境基础。
4. Parisi et al., *The Unsurprising Effectiveness of Pre-trained Vision Models for Control*, 2022(arXiv:2203.03580)——MoCo 表征用于控制的依据。
5. Kapelyukh et al., *DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics*, 2022(arXiv:2210.02438)——同期把扩散模型引入机器人的并行工作。
