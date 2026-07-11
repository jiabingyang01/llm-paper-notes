# InstructVLA：视觉-语言-动作指令微调——从理解到操作

> **论文**：*Vision-Language-Action Instruction Tuning: From Understanding to Manipulation*
>
> **作者**：Shuai Yang*、Hao Li* 、Bin Wang、Yilun Chen、Yang Tian、Tai Wang、Hanqing Wang、Feng Zhao、Yiyi Liao、Jiangmiao Pang（*为共同一作）
>
> **机构**：University of Science and Technology of China；Zhejiang University；Shanghai Artificial Intelligence Laboratory
>
> **发布时间**：2025 年 07 月（arXiv 2507.17520，2026 年 03 月发布 v2 修订版）
>
> **发表状态**：ICLR 2026（Published as a conference paper）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.17520) | [PDF](https://arxiv.org/pdf/2507.17520)
>
> **分类标签**：`VLA` `指令微调` `MoE-LoRA` `latent action` `flow matching` `具身推理`

---

## 一句话总结

InstructVLA 用 **MoE-LoRA** 门控让一个 1.5B 参数的 VLM 骨干在"文本推理专家"与"隐动作（latent action）专家"之间逐 token 动态混合，再由独立的 flow-matching 动作专家把隐动作解码为连续控制，从而在 SimplerEnv 上比 SpatialVLA 高 33.3%、在新提出的 80 任务指令泛化基准 SimplerEnv-Instruct 上比 OpenVLA+GPT-4o 高 31.7%（相对提升），同时多模态理解分数（MMMU/MME/VQA 等）与同规模纯 VLM 基座基本持平，没有出现其余 VLA 常见的灾难性遗忘。

## 一、问题与动机

现有 VLA 模型在"保留 VLM 的通用多模态推理"与"学到精准的动作生成"之间往往二选一：直接在动作数据上微调会侵蚀预训练视觉语言能力（catastrophic forgetting）；而把具身推理硬编码进结构化格式（如 ECoT 的子任务拆解、Emma-X 的 grounding 框）虽然能保留一部分链式推理，但这类格式与特定动作架构强绑定，表达力有限，作者的消融也显示这类两阶段方法仍存在"具身场景的领域缺口"。

论文将三大障碍归纳为：(1) 任务干扰导致的灾难性遗忘；(2) 带丰富多模态监督的高质量操作数据稀缺；(3) 缺乏把多模态推理有效转化为动作生成的训练范式与机制。核心问题因此被表述为：**如何在不侵蚀 VLM 多模态推理能力的前提下习得操作技能，并让推理反过来增强操作？**

## 二、核心方法

**整体流程（三步生成）**：(1) VLM 自回归文本推理 → (2) 隐动作（latent action）生成 → (3) flow-matching 动作专家解码为连续动作。骨干为轻量高效的 Eagle2-2B（LLM 部分约 1.5B 参数）。

**1. 隐动作接口。** 引入 $N$ 个可学习 action query $Q \in \mathbb{R}^{N\times D}$，通过 attention 从 VLM 隐状态中提取任务相关隐动作 $C \in \mathbb{R}^{N\times D}$，作为高层 VLM 规划与低层动作专家之间的解耦接口。消融显示 $N$ 从 16 扫到 128，取 64 是行为多样性与训练效率的较优折中——太少限制行为多样性，太多降低训练效率。

**2. MoE-LoRA 适配。** 为了让骨干在"语言推理"与"动作规划"之间无缝切换，作者用 LoRA 模块作为 LLM 内部的专家：

$$h = W_0x + \sum_{i=0}^{K} B_iA_ix \cdot \alpha_i \cdot \lambda_i$$

其中 $W_0$ 是原始权重、$A_i\in\mathbb{R}^{r\times d}$、$B_i\in\mathbb{R}^{d\times r}$ 为 LoRA 参数、$\alpha_i$ 为 LoRA 缩放因子、$\lambda_i$ 是一个 4 层 MLP"标量头"（scalar head，$2048\to128\to128\to128\to2$，ReLU）对每个 token 预测出的门控系数。

**用大白话说**：不管当前 token 是文本还是隐动作 query，都会同时经过"语言 LoRA"和"动作 LoRA"两条低秩旁路，标量头像个交通协管员，实时判断这个 token 更该听语言专家的还是动作专家的，从而让同一套骨干网络分时复用来同时处理两类任务，而不需要整体切换权重或维护两套独立网络。

**3. Flow-matching 动作专家。** 动作生成由一个独立、轻量（134M 参数）的 12 层 transformer（hidden size 768）承担，输入 DINOv2 视觉特征（用 FiLM 按隐动作调制）、隐动作及可选本体感知，输出动作 chunk $A\in\mathbb{R}^{H\times7}$（$H=16$ 步、7 维含夹爪）。训练目标为标准 flow matching 损失：

$$\mathcal{L}_{FM} = \mathbb{E}\left[\lVert V_\theta(A^\tau, q_t) - (\epsilon - A) \rVert^2\right]$$

**用大白话说**：像扩散模型一样，让网络学会"从纯噪声出发，一步步把它去噪成一条合理动作轨迹"所需的速度场，但用 flow matching 比标准扩散更简洁、采样步数更少（推理时仅需 10 步前向欧拉积分 $A^{\tau+1/N}=A^\tau+\frac1N V_\theta(A^\tau,q_t)$，从 $A^0\sim\mathcal N(0,I)$ 出发）。

**4. 两阶段训练配方。**
- **Stage 1（Action Pretraining）**：用异构操作数据（RT-1 类 Google Robot 数据 + Bridge）训练 VLM 预测隐动作，并用文本形式的 "language motion"（对低层动作的文字描述）做交叉熵监督，总损失 $\mathcal{L}=\mathcal{L}_{LM}+\mathcal{L}_{FM}$；此阶段只训练隐动作 embedding 与动作 LoRA 适配器（650M 参数），得到 "Expert" 模型。
- **Stage 2（Vision-Language-Action Instruction Tuning, VLA-IT）**：在已能跟随隐动作的动作专家基础上，加入语言 LoRA + 标量头组成完整 MoE 适配模块（仅 220M 可训练参数），与自建的 650K 样本 VLA-IT 数据集（场景描述 caption、QA、指令改写 command rewriting、上下文创建 context creation 四类，由 GPT-4o 基于三帧图像 + ground-truth 指令标注）及额外通用多模态语料联合、交替训练，多模态:动作训练比例取 1:7（是 ECoT/ChatVLA 所用 1:3 的两倍，以降低维持多模态能力所需的额外算力），得到 "Generalist" 模型。

**5. 推理加速。** 文本响应用贪心解码到第一个 action query token 出现即停止自回归，剩余 action query 在单次前向内并行解码；并支持跨动作步缓存语言响应与隐动作（Latent Action Caching）。单张 A100（BF16）下三种设置的推理频率：带语言推理 2.51 Hz、仅动作 3.50 Hz、启用隐动作缓存 4.96 Hz。

## 三、实验结果

**SimplerEnv（原子指令，Google Robot + WidowX Robot 平均成功率 %）与 SimplerEnv-Instruct（新提出，80 个零样本指令泛化任务，含 Task Aggregation 50 个 + Situated Reasoning 30 个，共 1.1K trial）：**

| 方法 | #参数 | SimplerEnv Avg | SimplerEnv-Instruct Avg |
|---|---|---|---|
| RT-1-X | - | 26.8 | - |
| OpenVLA-7B | 7B | 27.2 | 14.2 |
| SpatialVLA-3B | 3B | 45.9 | 16.5 |
| $\pi_0$-3B（S.） | 3B | 41.7 | 12.0 |
| GR00T-N1.5-3B（S.） | 3B | 36.0 | - |
| Magma-8B（sampling） | 8B | 43.6 | 23.8 |
| OpenVLA(FT&GPT-4o) | 7B | - | 35.6 |
| **InstructVLA-Expert** | 1.5B+134M | **50.9** | **17.3** |
| **InstructVLA-Expert（S.）** | 1.5B+134M | **61.2** | **20.7** |
| **InstructVLA-Generalist** | 1.5B+134M | **49.7** | **46.2** |
| **InstructVLA-Generalist（S.）** | 1.5B+134M | **54.9** | **46.9** |

（S. 表示加入本体感知状态输入；InstructVLA 结果为 3 个随机种子平均。）

**多模态理解（Table 1，节选，1.5B–8B 规模基线对照）：**

| 方法 | #参数 | MMMU | MM-Vet | MMStar | TextVQA | ChartQA |
|---|---|---|---|---|---|---|
| PaliGemma | 2B | 34.9 | 33.1 | 48.3 | 68.1 | 33.1 |
| Eagle2（基座） | 1.5B | 43.1 | 53.8 | 56.4 | 79.1 | 82.3 |
| OpenVLA(FT) | 7B | 26.0 | 9.1 | 28.2 | 2.5 | 1.4 |
| Magma | 8B | 38.8 | 34.1 | 41.3 | 66.5 | 61.8 |
| **InstructVLA-Generalist** | 1.5B | **44.2** | **51.7** | **56.2** | **77.7** | **81.7** |

InstructVLA-Generalist 与其多模态基座 Eagle2 基本持平（部分指标略优），而 OpenVLA 微调后多项指标近乎归零，说明动作专用微调造成了严重的多模态能力损失。

**LIBERO 基准（Table 10，四个任务套件成功率 %）：**

| 方法 | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| OpenVLA-7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT-7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| $\pi_0$-2B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_{0.5}$+KI（from generalist） | 98.0 | 97.8 | 95.6 | 85.8 | 94.3 |
| **InstructVLA-1.5B** | 97.3 | 99.6 | 96.5 | 89.8 | **95.8** |

InstructVLA 的动作专家仅 134M 参数（$\pi_0$ 为 300M），在未使用大规模操作数据预训练的情况下取得与 $\pi_{0.5}$+知识隔离（knowledge insulation）相当的成绩。

**真实机器人实验（WidowX-250 零样本 + Franka Research 3 少样本）：** 原子指令上 InstructVLA 比 OpenVLA 提升 23.3%；对名人识别/OCR/工具推理等推理型任务，少样本设置提升 41.7%、零样本设置提升 46.7%；数学推理任务上比 $\pi_0$（表现接近随机猜测）提升 2.5 倍。

**关键消融（Table 3，WidowX+Google 平均成功率 %）：** 去掉动作专家的 DINOv2 视觉输入（w/o DINO）成绩从 52.9 掉到 23.0（-50.0%）；去掉 FiLM 调制（w/o FiLM）为 45.9（-15.3%）；去掉 language motion 文本监督（w/o Lang.）为 48.4（-9.3%）。此外，四种多模态-操作协同训练范式对比中，InstructVLA 的 MoE + 两阶段训练比 AR 范式的 Magma 在 SimplerEnv 上高 12.5%；数据消融显示为 VLA-IT 语料加入 QA 与 captioning 标注可再带来 10.8% 的泛化提升。

## 四、局限性

- 现有任务仍局限于开/关、抓取/放置等基础操作原语，受限于所用数据集（RT-1、Bridge）与仿真器能力；相比标准 VLM benchmark 动辄数千种任务，操作任务集合仍然有限。
- 当前是单轮"指令 → 推理 → 动作"闭环，未支持涉及用户中途干预、多轮纠错的长程交互式任务。
- 未整合深度、触觉等模态，物理交互场景下的安全性与鲁棒性尚未系统验证。
- 数据标注依赖 GPT-4o，而论文自身也指出即使是 SOTA 多模态大模型在具身场景下也会出错，标注质量与真实指令之间存在性能落差。
- 作者声明大模型仅用于文字润色，研究思路、实验设计、数据分析与结论均由作者独立完成。

## 五、评价与展望

**优点**：InstructVLA 用参数高效的 MoE-LoRA + 逐 token 标量门控这一相对轻量的机制，而非维护两套独立网络，直接回应了"VLA 做动作微调会遗忘多模态能力"这一长期痛点；相较已发表的同类思路——Magma（自回归联合训练但迁移有限）、ECoT（只做文本 CoT、缺乏 QA 能力）、OpenVLA 外接 GPT-4o 做系统二（仍难以准确把自由指令转译为原子技能）——在指令泛化与多模态双指标上取得了更均衡的结果。论文同时贡献了 SimplerEnv-Instruct 这一公开评测资产（80 任务、系统控制 OOD 物体/环境/干扰物比例，覆盖 task aggregation 与 situated reasoning 两个维度），填补了此前 SimplerEnv/LIBERO 只评测原子指令、缺少高层指令泛化基准的空白。消融链条也较完整：视觉编码器（DINOv2/FiLM）、language motion 监督、隐动作 query 数量、四种协同训练范式、数据标注多样性均有量化对照。

**局限与开放问题**：其"推理"仍停留在单轮文本 CoT + 隐动作层面，并非 Hi Robot、$\pi_{0.5}$ 一类支持多轮交互式纠错的分层规划；与依赖闭源数据的 Helix、$\pi_{0.5}$ 等工作相比，InstructVLA 采用公开数据与代码更利于复现，但尚未在双臂协作、可变形物体等更灵巧的长程操作上得到验证。MoE 门控目前只在语言/动作两个专家间路由，且标量头没有额外辅助损失约束，在专家数量更多、任务更异构的场景下是否仍然稳定尚待验证。隐动作作为高层规划与低层控制之间的解耦接口，其可解释性与跨具身迁移能力也未与显式轨迹/关键点类中间表示（如 RT-Trajectory、Hamster）做直接对比。在 LIBERO 上，InstructVLA（95.8）与采用知识隔离的 $\pi_{0.5}$+KI（94.3）、OpenVLA-OFT（97.1）处于同一梯队但动作专家参数量明显更小，说明其参数效率具有优势，但尚未验证更大规模数据/模型下的 scaling 行为，也未覆盖人形等更多具身形态。

## 参考

- Qu et al., *SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model*, arXiv:2501.15830, 2025.
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024.
- Yang et al., *Magma: A Foundation Model for Multimodal AI Agents*, arXiv:2502.13130, 2025.
- Zawalski et al., *Robotic Control via Embodied Chain-of-Thought Reasoning (ECoT)*, arXiv:2407.08693, 2024.
- Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, arXiv:2307.15818, 2023.
