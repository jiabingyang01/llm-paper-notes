# CoT-VLA：面向视觉-语言-动作模型的视觉思维链推理

> **论文**：*CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models*
>
> **作者**：Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin
>
> **机构**：NVIDIA、Stanford University、MIT
>
> **发布时间**：2025 年 03 月（arXiv 2503.22020）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.22020) | [PDF](https://arxiv.org/pdf/2503.22020)
>
> **分类标签**：`VLA` `视觉思维链` `子目标图像生成` `动作分块` `混合注意力`

---

## 一句话总结

CoT-VLA 在预测动作之前先自回归生成一张未来第 n 帧的子目标图像作为"视觉思维链"中间推理步骤,再基于当前观测与该子目标图像预测一段动作块;模型基于统一多模态生成理解基座 VILA-U(7B)构建,配合"图像/文本用因果注意力、动作块用全注意力"的混合注意力设计,在 LIBERO 上平均成功率 81.13% 超过 OpenVLA(76.5%)和 Diffusion Policy(72.4%),在真实 Franka-Tabletop 平台上预训练带来约 46.7% 的相对性能提升(53.7%→78.8%)。

## 一、问题与动机

现有 VLA 大多直接把 VLM 微调为"观测+指令→动作"的直接映射,没有显式的中间推理步骤,缺少语言 CoT 那样逐步思考带来的时间规划/推理能力。已有的具身 CoT 工作尝试用语言描述、关键点或 bounding box 作为中间表示,但这些抽象表示通常需要额外的标注或预处理管线。作者观察到"子目标图像"这种中间推理形式天然存在于机器人示范视频里,几乎不需要额外标注,同时它还能解锁大量无动作标注的人类活动视频(如 EPIC-KITCHEN、Something-Something V2)用于增强模型的视觉推理与动力学理解能力——这是本文的核心切入点。

## 二、核心方法

**两阶段建模**。记机器人示范数据 $D_r=\{(l,\mathbf{a}_{1...T},\mathbf{s}_{1...T})\}$、无动作视频数据 $D_v=\{(l,\mathbf{s}_{1...T})\}$。传统 VLA 直接学习

$$\hat{\mathbf{a}}_t \sim P_\theta(\mathbf{a}_t \mid \mathbf{s}_t, l)$$

CoT-VLA 拆成"先想后做"两步：先预测 $n$ 帧之后的子目标图像 $\hat{\mathbf{s}}_{t+n}\sim P_\theta(\mathbf{s}_{t+n}\mid \mathbf{s}_t,l)$,再基于当前观测和子目标图像预测长度为 $m$ 的动作块 $\{\hat{\mathbf{a}}_t,...,\hat{\mathbf{a}}_{t+m}\}\sim P_\theta(\cdot\mid \mathbf{s}_t,l,\hat{\mathbf{s}}_{t+n})$。大白话说：模型先在脑子里"画"出完成任务后大致长什么样,再照着这张想象图去规划怎么动手,而不是看一眼就直接出手。子目标生成阶段同时用 $D_r$ 和 $D_v$ 训练(不需要动作标注),动作生成阶段只用 $D_r$ 训练。

**基座与 tokenizer**：模型建立在统一多模态基座 VILA-U 之上,该模型用残差量化(类 RQ-VAE 的 depth transformer)把 $256\times256$ 图像编码为 $16\times16\times4$ 的离散视觉 token,可同时做图文理解与生成。视觉 token 训练目标为逐位置自回归预测 $D=4$ 层残差 token：

$$\mathcal{L}_{\text{visual}} = -\sum_j\sum_{d=1}^{D}\log P_\delta(k_{jd}\mid k_{j,<d})$$

即每个空间位置的粗到细残差编码逐级自回归解码,大白话说就是"先定大致颜色块,再逐层补细节"。

**动作 token 化与混合注意力**：每个动作维度独立离散化为 256 个 bin(边界由训练数据 1~99 百分位均匀划分,复用文本 tokenizer 中最不常用的 256 个 token 作为动作 bin token,做法沿用 OpenVLA);7 自由度动作用 7 个 token 表示。动作预测损失为

$$\mathcal{L}_{\text{action}} = -\sum_{i=1}^{m}\log P_\theta(\mathbf{a}_t...\mathbf{a}_{t+m}\mid l, \mathbf{s}_t, \mathbf{s}_{t+n})$$

总损失 $\mathcal{L}=\mathcal{L}_{\text{action}}+\mathcal{L}_{\text{visual}}$。区别于逐 token 因果预测,动作块内部采用**全注意力**(而非因果注意力),让一个动作块内的所有 token 互相可见、支持并行解码,图像/文本 token 生成仍用标准因果注意力——这就是论文强调的"混合注意力"设计。动作块长度 $m=10$。

**训练与部署**：预训练阶段在 VILA-U(7B)基座上联合训练 OpenX 机器人示范子集(沿用 OpenVLA 的预处理,单臂末端执行器 7-DoF、第三人称视角)与无动作视频(EPIC-KITCHEN-100、Something-Something V2),预测子目标的时间跨度 $n$ 按数据集分别设定上下界(如 Bridge 为 5~10 帧,TOTO 为 20~24 帧)。下游任务在目标机器人平台的少量示范上继续微调 LLM 主干、projector 与 depth transformer(视觉塔冻结)。部署时闭环执行：采样子目标图像→采样动作块→执行→采集新观测→重复。预训练用 12 节点×8 卡 A100(共 96 卡)训练,总计约 11K A100 GPU 小时;下游微调在单节点 A100 上 10~24 小时。

## 三、关键结果

**LIBERO 仿真基准**(500 episode/任务套件,3 个随机种子,均值±标准误):

| 方法 | Average | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| Diffusion Policy | 72.4±0.7 | 78.3±1.1 | **92.5±0.7** | 68.3±1.2 | 50.5±1.3 |
| Octo (fine-tuned) | 75.1±0.6 | 78.9±1.0 | 85.7±0.9 | 84.6±0.9 | 51.1±1.3 |
| OpenVLA (fine-tuned) | 76.5±0.6 | 84.7±0.9 | 88.4±0.8 | 79.2±1.0 | 53.7±1.3 |
| **CoT-VLA-7B（本文）** | **81.13±0.6** | **87.5±1.4** | 91.6±0.5 | **87.6±0.6** | **69.0±0.8** |

**Bridge-V2 真实机器人**(WidowX 6-DoF,四类泛化能力,每类 10 次试验,部分分制打分):

| 类别 | SUSIE | Octo | OpenVLA | CoT-VLA |
|---|---|---|---|---|
| Visual（杂乱背景干扰物） | 30% | 35% | **75%** | 65% |
| Motion（高度变化） | 10% | 10% | 45% | **60%** |
| Semantic（未见语言描述物体） | 20% | 0% | 40% | **50%** |
| Language（指令跟随） | 40% | 40% | **75%** | 70% |

论文指出 CoT-VLA 在 Visual/Language 两项略低于 OpenVLA,主要归因于动作分块导致的抓取失败,而非视觉推理本身的错误。

**Franka-Tabletop 真实机器人**(6 个任务,10~150 条示范/任务)：单指令任务上 Diffusion Policy 表现最好(如"把玉米放进碗里"),但在多指令、需要语言 grounding 的任务上明显退化;经过 OpenX 预训练的 Octo/OpenVLA/CoT-VLA 在多指令任务上适应更好。图 4 柱状图显示 CoT-VLA 六任务平均成功率约 78.8%,高于 Diffusion Policy(约 51%)、Octo(约 42%)、OpenVLA(约 67%)(读自柱状图,存在像素级误差,78.8% 数值与消融实验中"预训练后"结果一致)。

**消融实验**（Fig.6a,LIBERO-Spatial / LIBERO-Goal,四个变体依次叠加组件）：

| 变体 | Spatial | Goal |
|---|---|---|
| VLA（vanilla, 单步动作） | 67.5 | 54.9 |
| + 动作分块 | 73.3 | 74.9 |
| + 混合注意力 | 81.8 | 79.7 |
| + 视觉 CoT（完整版） | **87.5** | **87.6** |

预训练消融(Fig.6b,Franka-Tabletop)：不做 OpenX+无动作视频预训练直接微调 VILA-U 基座为 53.7%,加入预训练阶段后提升到 78.8%,相对提升 46.7%。

**更优视觉推理是否直接转化为更优动作**（Table 3,两个分布外长时程子任务,各 5 次试验)：用生成的子目标图像 vs. 用示范轨迹里的真值目标图像对比——子任务 1 从 20%→60%,子任务 2 从 0%→40%,两项均绝对提升 40 个百分点,说明子目标图像质量对最终动作成功率有直接、显著的因果影响,同时也暴露出当前子目标生成在分布外场景下质量还不够。

**局限性**（原文第 5 节明确列出）：(1) 推理时需先生成 256 个图像 token 再生成动作 token,在动作块长度为 10 时导致平均约 7 倍的推理减速,是当前系统的主要瓶颈;(2) 自回归图像生成的视觉质量低于当前最优的扩散式生成模型;(3) 动作分块会在块与块之间引入不连续动作,且缺乏高频反馈;(4) 受算力限制,当前模型对全新分布外任务的视觉推理泛化能力仍然有限。

## 四、评价与展望

CoT-VLA 的核心贡献是把"子目标图像"作为一种几乎零额外标注成本的具身 CoT 中间表示,并证明它能同时联通机器人示范数据与海量无动作视频数据的预训练——这一点相比此前用 bounding box/关键点/语言描述作为中间推理步骤的具身 CoT 工作(如 Embodied Chain-of-Thought Reasoning、RT 系列的显式规划变体)更贴近"视频即通用中间表示"的思路,也和 SuSIE 这类"目标图像编辑+目标条件策略"两阶段方法同源,但本文用单一统一自回归模型(VILA-U)同时完成图像生成与动作生成,而非像 SuSIE 那样依赖独立的扩散先验模型。实验中一个值得关注的反差是：SuSIE 依赖扩散先验生成的目标图像视觉质量更高,但在 Bridge-V2 上的成功率却明显低于 CoT-VLA 和 OpenVLA,说明"目标图像视觉质量"与"目标图像对策略有用"并不完全等价,统一 token 化表示可能在与动作预测的耦合效率上更有优势,这是一个值得后续工作深挖的方向。

方法的主要开放问题在于计算效率与生成质量的权衡：自回归逐 token 生成 256 个图像 token 带来约 7 倍的推理开销,这对需要较高控制频率的真实机器人闭环控制是较严重的实际约束,论文也坦承这是当前系统的主要瓶颈,并寄望于更快的图像生成/LLM 推理技术(如推测解码、一致性模型)或替换为扩散式/掩码式并行生成方案。另一个开放问题是分布外泛化：Table 3 显示子目标图像质量对动作成功率有近乎线性的直接影响,而当前预训练规模下模型在全新长时程任务上的子目标生成质量仍不理想,这与更大规模视频生成模型/世界模型的能力上限直接相关,提示未来提升空间很大程度上取决于底层视频生成基座的进步而非动作头设计本身。此外,消融实验将"动作分块""混合注意力""视觉 CoT"三个改动依次叠加得到的增益(LIBERO-Spatial 从 67.5% 到 87.5%),其中动作分块和混合注意力两项本身就贡献了大部分提升(67.5→81.8),视觉 CoT 这一核心创新点的增量贡献(81.8→87.5,约 5.7 个百分点)相对温和,这提示该框架下"预测子目标图像"这一步骤的边际价值仍有进一步量化和消融的空间(例如是否可以用更轻量的中间表示替代完整图像 token 序列)。

## 参考

- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024 — 本文核心基线与动作离散化方案的直接来源。
- Wu et al. *VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation*, arXiv:2409.04429, 2024 — CoT-VLA 所构建的统一多模态基座模型。
- Black et al. *Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models (SuSIE)*, arXiv:2310.10639, 2023 — 目标图像生成+目标条件策略的两阶段对比基线。
- Zawalski et al. *Robotic Control via Embodied Chain-of-Thought Reasoning*, arXiv:2407.08693, 2024 — 相关的具身 CoT 路线(语言/关键点中间推理)。
- Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, arXiv:2306.03310, 2023 — 本文主要仿真评测基准。
