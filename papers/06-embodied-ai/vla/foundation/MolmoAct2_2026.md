# MolmoAct2：面向真实世界部署的动作推理模型

> **论文**：*MolmoAct2: Action Reasoning Models for Real-world Deployment*
>
> **作者**：Haoquan Fang, Jiafei Duan（共同一作）et al.（含 Zhongzheng Ren、Joyce Chai、Ali Farhadi、Dieter Fox、Ranjay Krishna 等；项目 PI 为 Ranjay Krishna）
>
> **机构**：Allen Institute for AI（Ai2）、University of Washington、National University of Singapore、University of Pennsylvania、Johns Hopkins University、Amazon、Cortex AI、University of Michigan、University of North Carolina at Chapel Hill
>
> **发布时间**：2026 年 05 月（arXiv 2605.02881，v2 于 2026 年 5 月 8 日更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.02881) | [PDF](https://arxiv.org/pdf/2605.02881)
>
> **分类标签**：`VLA` `动作推理模型` `flow matching` `逐层KV条件化` `自适应深度推理` `开源双臂数据集`

---

## 一句话总结

MolmoAct2 在具身推理 VLM 骨干 Molmo2-ER 之上,用**逐层 KV 条件化** 把离散 token 骨干与 flow-matching 连续动作专家对接,并提出仅对场景变化区域重新解码深度 token 的自适应推理变体 MolmoAct2-Think;在 DROID 真实机器人零样本部署达到 87.1% 成功率(超第二名 MolmoBot 38.7 个百分点)、LIBERO 均分 97.2%(Think 版 98.1%)、Molmo2-ER 在 13 个具身推理基准上以 63.8% 均分超过 GPT-5 与 Gemini Robotics-ER 1.5 Thinking,并开源了目前最大的双臂操作数据集(720 小时、34,500 条轨迹)。

## 一、问题与动机

论文指出当前 VLA 落地部署存在四重矛盾:(1) 前沿模型(π 系列、Gemini Robotics 等)训练数据、配方、权重均不公开,难以复现或改进;(2) 少数开放权重模型(如 π0.5)绑定在昂贵/特定机器人平台上,超出大多数学术实验室可及范围;(3) 引入"推理"过程(预测深度、目标图像、点轨迹或完整世界模型 rollout)虽能提升动作质量和可解释性,但要在动作生成前吐出数百 token 甚至整帧预测,极大拖慢闭环控制的推理延迟;(4) 即便做任务特定微调,现实任务成功率仍达不到可靠部署门槛。作者的前作 MolmoAct(Lee et al., 2025)已尝试用深度 token 做具身推理,但每一步都重新生成几乎相同的静态场景表示,推理开销仍然过高。

MolmoAct2 的目标是同时满足"**完全开放**(权重+代码+数据)+**开箱可部署**+**快速可解释推理**"三个条件,沿五个轴改进前作:更强的具身推理 VLM 骨干(Molmo2-ER)、新的开源机器人数据集、开源多具身动作分词器(MolmoAct2-FAST Tokenizer)、新的 VLA 架构(逐层 KV 条件化的 flow-matching 动作专家)、自适应深度推理范式(MolmoAct2-Think)。

## 二、核心方法

### 2.1 Molmo2-ER:面向具身推理的 VLM 骨干

通用 VLM 很少针对机器人所需的度量距离、自由空间、跨视角物体追踪、场景几何等能力训练或测试。作者在 Molmo2(Clark et al., 2026)基础上,用一个约 330 万样本的具身推理语料(Table 1,涵盖单图具身 QA、指点/检测、视频具身 QA、多图/ego-exo 对应、抽象具身推理六大能力支柱,数据来源包括 SAT、RoboPoint-QA、RefSpatial、VST-P、VSI-590K、SIMS-VSI、RoboVQA、SenseNova-SI、CLEVR、GRiD-3D 等)进行**specialize-then-rehearse** 两阶段训练:

- **Stage 1(具身专化)**:从 Molmo2-4B 中训练检查点出发,在具身推理语料 + 8% Tulu-3 文本数据上微调 20K 步,序列长度 4,200,快速把模型移动到具身数据流形上。
- **Stage 2(联合精修)**:继续训练 1.5K 步,将具身语料与 Molmo2 原始多模态中训练数据(通用 VQA、caption、学术基准、追踪、指点)按比例 $p$ 混合,扫描 $p\in\{0.30,0.50,0.70,0.90\}$ 发现 $p=0.5$ 在具身推理与 Molmo2 通用基准间取得最佳帕累托权衡。

**用大白话说**:先让模型"专精"到机器人需要的空间理解能力上,再把这部分能力和原有的通用视觉语言能力按比例"揉"回去,避免为了学空间推理而遗忘掉通用视觉语言能力。

### 2.2 三个新数据集

- **MolmoAct2-BimanualYAM Dataset**:自建双臂 YAM(Yet Another Manipulator)平台采集,覆盖 28+ 类真实任务(叠衣、理线、收桌、扫描杂货、分装药品等),34,500 条演示、共 720 小时,是目前最大的开源双臂操作数据集(对比 AIST Bimanual 10,000 条、BiPlay 7,000 条、RDT-1B 微调集 6,000 条、闭源 ALOHA Unleashed 26,000 条)。采集设备总成本控制在 6,000 美元以内(RealSense D435/D405 + IKEA 桌 + YAM 双臂)。
- **MolmoAct2-SO100/101 Dataset**:从社区贡献的 1,660 个 LeRobot 数据集中,用四阶段过滤(结构有效性 → 剔除评测风格数据 → 许可/代码合规 → TOPReward 质量门限)筛出 1,222 个高质量数据集(377 名贡献者、38,059 条演示、19.8M 帧、约 184 小时)。
- **MolmoAct2-DROID Dataset**:基于 DROID 官方补充标注(扩展语言标注 + 空闲帧过滤),保留 74,604 条有效、成功、无长时间停顿的片段,共 17,758,044 帧。

三套数据集都用开源 VLM(Qwen3.5-27B)重新标注语言指令,将数据集整体唯一指令占比从 22% 提升到 46%(如 BimanualYAM 从 0.1% 提升到 35%),缓解"lerobot_test"之类占位标注和重复模板指令的问题。

### 2.3 MolmoAct2-FAST Tokenizer

沿用 FAST(Pertsch et al., 2025)的频域压缩(DCT + BPE)分词原理,但完全公开训练数据混合(Table 2):在 100 万条动作序列上训练,YAM/SO-100/101/DROID 各占 30%,Fractal/BC-Z/Bridge 各占 3.33%,覆盖绝对关节控制与增量末端执行器控制两种"方言",词表 2048。所有连续动作先统一填充到 32 维、用 1–99 百分位统计归一化(夹爪单独处理为二值/窄范围开合信号),再分词,使同一分词器可跨平台复用。

### 2.4 架构:逐层 KV 条件化的 flow-matching 动作专家

MolmoAct2 采用三阶段训练管线:预训练(离散 next-token)→ 后训练(接入连续动作专家)→ 具身特定微调。

**预训练** 将 Molmo2-ER 适配为离散自回归机器人策略,新增状态 token 和动作 token 流,用与基础 VLM 相同的 next-token 目标训练,不引入独立的连续动作头。

**后训练** 给预训练骨干接上一个 DiT 风格、L=36 层(与骨干同深)的 flow-matching 动作专家。给定归一化目标动作块 $a$、高斯噪声 $\epsilon$、采样时间 $t\in[0,1]$,构造插值

$$x_t = (1-t)\epsilon + ta, \qquad u^\star = a-\epsilon,$$

专家 $f_\theta$ 从带噪轨迹 $x_t$、时间嵌入、VLM 上下文 $c$ 预测目标速度 $u^\star$,损失为

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{a,\epsilon,t}\Big[\big\|m\odot\big(f_\theta(x_t,t,c)-u^\star\big)\big\|_2^2\Big],$$

其中 $m$ 屏蔽掉填充的时间步和填充的动作维度。**用大白话说**:这就是标准的 flow-matching/整流流训练——让网络学会"从纯噪声一步步流向真实动作轨迹"的速度场,而不是像扩散模型那样反复去噪很多步。

关键架构选择在于动作专家如何获取 VLM 上下文。作者没有像多数已有 VLA(如 π0)那样只在专家的每一层用 VLM 最终隐藏状态做浅层投影条件化,而是让专家**每一层** 都从骨干**对应层** 的自注意力 key/value 取条件,即

$$\tilde K_\ell = \mathrm{reshape}(P_K K_\ell^{\text{vlm}}), \qquad \tilde V_\ell = \mathrm{reshape}(P_V V_\ell^{\text{vlm}}),$$

$$\mathrm{CA}(Q_\ell,\tilde K_\ell,\tilde V_\ell) = \mathrm{softmax}\!\left(\frac{Q_\ell \tilde K_\ell^{\top}}{\sqrt{d_h}}\right)\tilde V_\ell,$$

专家每个 block 依次做动作自注意力(带 AdaRMS 时间调制)、对骨干对应层 KV 的交叉注意力、再过 MLP。**用大白话说**:与其把 VLM 压缩成一条"扁"的向量喂给动作专家,不如让动作专家在第 $\ell$ 层直接"偷看"骨干在第 $\ell$ 层用的注意力状态——浅层看到的更偏底层视觉/几何特征、深层看到的更偏语义,这样动作专家能拿到骨干内部逐层演化的层次化视觉语义特征,而不是只看最后一层的"摘要"。后训练阶段该条件化通路对 VLM 是**detach** 的(knowledge insulation),即 flow 损失只更新专家和适配投影,不会反向传播进 VLM 本身。

后训练同时保留离散动作 token 预测,总损失为

$$\mathcal{L}_{\text{post}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{flow}},$$

且对每个动作块采样 $K$(默认 4)个独立的 $(\epsilon_i,t_i)$ 做多重 flow 样本平均:

$$\mathcal{L}_{\text{flow}}(a,c) = \frac{1}{K}\sum_{i=1}^{K}\big\|m\odot\big(f_\theta(x_{t_i},t_i,c)-(a-\epsilon_i)\big)\big\|_2^2.$$

**用大白话说**:同一个视觉语言上下文,多采几个噪声/时间点算 loss 再平均,相当于花更少的前向次数换取更稠密的监督信号。

### 2.5 MolmoAct2-Think:自适应深度推理

前作 MolmoAct 在动作生成前预测一个 $10\times10$ 深度 token 网格(每格取 128 个学习到的深度码之一)作为显式几何 grounding,但每一步都重新预测全部 100 个格子,而机器人轨迹里场景大部分区域帧间几乎不变。MolmoAct2-Think 通过逐帧比较 $32\times32$ patch 的余弦相似度,只在变化区域重新解码深度码,未变化区域复用缓存:

$$m_{t,i} = \mathbf{1}\big[\cos(x_{t,i},x_{t-1,i}) < 0.996\big], \qquad b_{t,i}=\begin{cases}d_{t,i}, & m_{t,i}=1\\ b_{t-1,i}, & m_{t,i}=0\end{cases}$$

**用大白话说**:把深度 token 当成一段可增量更新的"缓存",只对画面里真正发生变化的格子重新推理,静止的格子直接照抄上一帧的结果,latency 与场景变化比例成正比而非固定 100 token。此外还引入一个逐层学习门控 $g_\ell=\sigma(w_\ell^\top c_\ell+b_\ell)$(初始化偏置为 -4,即默认贴近标准动作条件化路径),按需调节每层专家对深度前缀 KV 的依赖强度;训练时对 10% 的深度 token 输入注入噪声,模拟推理阶段深度预测本身不完美的情形。

### 2.6 模型规模

骨干总参数约 5.06B:视觉编码器(SigLIP2 ViT)380M + V/L connector 57M + LLM 4.0B + 动作专家 621M(8 头、宽度 768,与骨干同为 36 层)。预训练 200K 步(64×H100,约 5,760 GPU 小时),后训练 100K 步(64×H100,约 2,304 GPU 小时),各具身微调约 1,152–2,304 GPU 小时。

## 三、实验结果

### 3.1 Molmo2-ER:13 个具身推理基准(Table 3)

| 模型 | 类型 | Overall Avg |
|---|---|---|
| GR-ER 1.5 Thinking | API | 61.3 |
| GPT-5 | API | 57.9 |
| Gemini 2.5 Pro | API | 57.1 |
| Qwen3-VL-8B | 开放权重 | 61.0 |
| InternVL3.5-8B | 开放权重 | 52.4 |
| Molmo2(基座) | 开放权重/数据 | 46.8 |
| **Molmo2-ER(本文)** | 开放权重/数据 | **63.8** |

Molmo2-ER 在 9/13 个基准上超过所有基线,总均分 63.8%,比第二名 Gemini-ER 1.5 Thinking(61.3)高 2.5 分,比基座 Molmo2 提升 17 个百分点。

### 3.2 开箱部署(Sec 6.2)

**仿真(MolmoSpaces,Table 4)**:MolmoAct2-DROID 在 Pick/Pick&Place/Open/Close 四类任务上均分 37.7%,超过最强基线 π0.5-DROID 的 34.5%(+3.2),Pick(+7.3)、Pick&Place(+13.1)增益最大,Open 项仍落后为待改进方向。

**真实世界零样本部署**(相机位姿随机初始化、物体全未见、场景域外):

| 平台 | 基线最佳 | MolmoAct2 | 提升 |
|---|---|---|---|
| DROID(5 任务, 15 trials/任务) | MolmoBot 48.4% | **87.1%** | +38.7pt |
| SO-100/101(5 任务,部分计分) | π0-SO100/101 45.3% | **56.7%** | +11.4pt |

### 3.3 高效微调(Sec 6.3)

| 基准 | 次优基线 | MolmoAct2 | MolmoAct2-Think |
|---|---|---|---|
| LIBERO(4 套件均分) | π0.5 96.9% | 97.2% | **98.1%** |
| RoboEval(8 双臂任务) | π0.5 40.5%(+3.8 差距) | **44.3%** | — |
| 真实世界 Bimanual YAM(8 任务) | OpenVLA-OFT 35.5% | **50.6%**(+15pt) | — |

LIBERO 上 MolmoAct2 单项 LIBERO-Object 达 100.0%,总均分 97.2%,超越前作 MolmoAct-7B-D(86.6%)达 10.6 个百分点。

### 3.4 MolmoAct2-Think 的增益(Sec 6.4)与 OOD 鲁棒性(Table 9)

LIBERO 上 Think 版在 3/4 套件上优于标准版,提升集中在最难的 Long 套件(93.2%→95.4%,其余套件已接近饱和),整体 2,000 次 rollout 均分从 97.2% 提升到 98.1%(+0.9,方向一致)。

在 4 类分布外扰动(空间位置、光照、语言复述、干扰物)下,MolmoAct2-Think 综合均分 50.69%,较次优 OpenVLA-OFT(39.89%)高约 10.8 个百分点;优势在 Distractor 上最窄(仅 +5.8),Spatial Variance 上绝对分数最低(26.25%),说明细粒度空间泛化仍是短板。

### 3.5 消融(Sec 6.7, LIBERO)

| 消融维度 | 结果 |
|---|---|
| 骨干替换(Molmo2→Molmo2-ER,纯离散动作) | 77.6%→83.6%(+6.0pt) |
| VLM→专家条件化来源 | 逐层 KV 95.9% > 逐头逐层 KV 94.8% > 隐藏状态条件化 94.0% |
| flow 采样数 $K$ | $K{=}1$ 94.15% → $K{=}8$ 95.90%(非严格单调,但整体趋势向好) |
| 微调设计 | 全参数微调+离散动作联合训练+不做 knowledge insulation 最佳(97.20%);仅调动作专家最差(93.05%) |
| Think 深度微调设计 | 混合训练+噪声注入+深度门控三者齐全最佳(98.10%),纯净版仅 97.50% |

### 3.6 推理速度(Sec 6.8, 单张 H100,action horizon=10)

| 优化路径 | MolmoAct2 | MolmoAct2-Think |
|---|---|---|
| 原始实现 | 23.02 Hz | 8.04 Hz |
| + 缓存复用 | 27.39 Hz | 9.72 Hz |
| + CUDA Graph | **55.79 Hz**(2.42×) | **12.71 Hz**(1.58×) |

连续动作路径比离散自回归解码路径快 3.94×(MolmoAct2)/1.86×(Think),因此论文将连续路径设为默认部署选项。

## 四、局限性

论文在附录 E 明确列出两点:

1. **动作块与缺乏实时重新分块(re-chunking)**:MolmoAct2 预测固定视野的动作块(YAM/SO-100/101 为 30Hz 下 30 步,DROID 为 15Hz 下 15 步,LIBERO 为 10Hz 下 10 步)后开环执行,再重新查询策略。这带来两个问题:(a) flow-matching 专家独立去噪每个块,块与块之间没有连续性损失约束运动平滑性,块边界处可能出现速度/加速度不连续(尤其当 VLM 上下文在两次查询间发生变化时);(b) 策略要执行完整个块才能观察后果,无法在块内对扰动、接触事件或自身跟踪误差做出反应——第 6.8 节报告的 55.79 Hz 是均摊的块吞吐量,并非真正的闭环反应速率。
2. **具身特定的零样本部署边界**:目前"开箱部署"的三个平台(BimanualYAM、SO-100/101、DROID Franka)之所以能零样本工作,直接源于数据采集/整理策略集中在这三个平台上;MolmoAct2 并非能零样本迁移到任意新具身(其他双臂平台、灵巧手、移动操作、类人机器人等)的通用控制器,部署到集合之外的机器人仍需要在目标具身演示上做微调。作者将其定位为"可扩展的基础"而非"完成品"。

此外从实验结果本身也能看出一些隐含局限:Open(开门类关节物体交互)在 MolmoSpaces 上仍落后于最强基线;细粒度空间泛化(Spatial Variance 扰动下 26.25%)是四类 OOD 扰动中表现最弱的一项;真实世界评测样本量总体有限(如真实 DROID 每任务仅 15 trials),置信区间较宽。

## 五、评价与展望

**优点**:MolmoAct2 是目前公开程度最完整的 VLA 工作之一——不仅开放权重,还开放训练代码、完整训练数据(含新采集的 720 小时双臂数据集)与分词器,并给出了迄今为止开放 VLA 中最全面的实证研究(7 个仿真+真实基准、13 个具身推理基准、8 项微调任务、系统性消融)。其架构贡献——用**逐层** 而非**末层** KV 条件化连接离散骨干与连续 flow-matching 专家——是对当前"VLM-动作专家桥接"这一开放问题(论文 Related Work 中明确点出,π0/RDT/OpenVLA-OFT 等多用末层隐藏状态条件化)的一个具体、消融验证过的改进,且消融显示其相对隐藏状态条件化在 Object/Goal/Long 套件上均有稳定优势。MolmoAct2-Think 的自适应深度 token(仅重算变化区域)相较同期把整张 RGB 观测都做视觉追踪/掩码的 PEEK(Zhang et al., 2025)等方法,在保持深度 grounding 收益的同时明显降低了 token 消耗与延迟,是"推理增强 VLA down to 实时可用"路线上的一个值得关注的设计。

**局限与开放问题**:(1) 论文自己承认的块级开环执行问题,使得该模型本质仍是"高频重规划"而非严格意义上的闭环反应式控制,与近来一些强调因果/流式 world model rollout 的工作(如 π0.5 系列后续、Cosmos Policy)相比,对突发扰动的鲁棒性上限受限于块粒度;(2) 零样本部署能力被清楚限定在三个训练时覆盖的具身范围内,论文没有报告在这三者之外(如人形机器人、灵巧手)不经微调的表现,其"通用性"主张更多体现在架构和训练配方的可迁移性,而非当前 checkpoint 的即插即用范围;(3) 细粒度空间位置泛化(OOD Spatial Variance 26.25%)相对其他扰动类型明显偏弱,提示深度 token 提供的几何 grounding 可能还没有充分转化为对新颖空间布局的鲁棒性,这是后续工作(例如更细粒度的深度/点云表示,或显式的空间数据增广)可以着力的方向;(4) 自适应深度机制依赖一个基于像素余弦相似度的启发式阈值(0.996)判定"场景是否变化",这一固定阈值在不同光照/相机噪声条件下的稳健性未被单独消融,是一个可能的脆弱点;(5) 与同为"推理增强"路线但强调隐式迭代计算的 Recurrent-Depth VLA(Tur et al., 2026)、显式世界模型 rollout 的统一世界模型(Zhu et al., 2025)相比,MolmoAct2-Think 选择了"轻量离散深度 token + 时间稀疏更新"这一相对折中的设计点,其代价(几何表达力弱于稠密世界模型)与收益(推理延迟低、可解释)之间的权衡是否是长期最优尚待社区进一步比较。

## 参考

1. Lee, J. et al. MolmoAct: Action Reasoning Models that can Reason in Space. arXiv:2508.07917, 2025.（前作)
2. Clark, C. et al. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding. arXiv:2601.10611, 2026.（VLM 骨干基座)
3. Pertsch, K. et al. FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747, 2025.（本文动作分词器所依据的框架)
4. Intelligence, P. et al. π0.5: A Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054, 2025.（贯穿全文最主要的强基线)
5. Khazatsky, A. et al. DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset. arXiv:2403.12945, 2024.（本文 DROID 子集与真实世界评测所依据的原始数据集)
