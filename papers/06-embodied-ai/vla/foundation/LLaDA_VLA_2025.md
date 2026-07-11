# LLaDA-VLA：视觉-语言-扩散动作模型

> **论文**：*LLaDA-VLA: Vision Language Diffusion Action Models*
>
> **作者**：Yuqing Wen, Hebei Li, Kefan Gu, Yucheng Zhao, Tiancai Wang, Xiaoyan Sun et al.
>
> **机构**：University of Science and Technology of China（中国科学技术大学）、Nanjing University（南京大学）、Dexmal
>
> **发布时间**：2025 年 09 月（arXiv 2509.06932）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.06932) | [PDF](https://arxiv.org/pdf/2509.06932)
>
> **分类标签**：`VLA` `掩码扩散模型` `离散扩散动作生成` `LLaDA` `机器人操作`

---

## 一句话总结

LLaDA-VLA 是首个构建在预训练掩码扩散视觉语言模型（d-VLM，基座为 LLaDA-V）之上的 VLA 模型，通过"局部化特殊 token 分类"和"层级化动作结构解码"两项设计把通用掩码扩散范式适配到机器人动作生成，在 SimplerEnv WidowX（55.5% vs. OpenVLA 4.2%/CogACT 51.3%）、CALVIN ABC-D（平均连续完成任务数 4.01 vs. OpenVLA 3.27）和真实 WidowX 机器人（58% vs. π0 35%/CogACT 30%，OOD 泛化 40% vs. π0 15%）上全面超越对比基线。

## 二、问题与动机

自回归视觉语言动作模型（ARM-based VLA，如 RT-2、OpenVLA、CogACT、π0）依赖顺序化的 token 生成，本质上是单向的，在复杂多模态机器人任务中效率和灵活性受限。近期以 LLaDA 为代表的大语言掩码扩散模型（masked diffusion model, MDM）在文本生成上展现出可与自回归模型媲美的性能和 scaling 特性；LLaDA-V、MMaDA 等工作进一步把这一范式拓展到视觉语言领域，形成能与主流 ARM-VLM 性能相当的扩散式 VLM（d-VLM）。但 d-VLM 能否用于机器人策略学习仍未被探索。作者指出直接迁移面临两个挑战：其一，d-VLM 预训练于富含高层语义的大规模通用数据，而 VLA 需要解读低层视觉线索来生成精确动作，二者之间存在显著的域差异（domain gap）；其二，掩码扩散原生的解码策略平等对待所有输出 token，未考虑动作序列固有的结构依赖（同一动作内部各维度之间、以及连续多步动作之间的依赖），难以生成合理连贯的动作轨迹。论文提出 LLaDA-VLA 来应对这两个挑战。同期工作 Discrete Diffusion VLA（论文中标注为 concurrent work）思路上有部分相似，但仍是在 ARM-VLM 基础上加装扩散动作解码头，与本文"直接以预训练 d-VLM 为骨干"的路线不同。

## 三、核心方法

### 3.1 背景：掩码扩散模型

MDM 定义一个离散状态上的前向-反向扩散过程：正向过程中，长度为 $L$ 的序列 $x_0$ 里每个 token 独立地以概率 $t\in(0,1)$ 被替换为特殊 mask token `[M]`；反向过程由一个 Transformer 参数化的 mask predictor $p_\theta$ 学习，训练目标是仅在被 mask 的位置上计算交叉熵：

$$\mathcal{L}(\theta) \triangleq -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^L \mathbf{1}[x_t^i=\text{[M]}]\log p_\theta(x_0^i \mid x_t)\right]$$

用大白话说：训练时随机盖住序列里一部分 token（盖住比例由随机采样的扩散时间步 $t$ 决定），模型要根据没被盖住的上下文把被盖住的内容猜回来；推理时从一个完全被盖住的序列出发，反复"预测—按置信度保留一部分—把低置信度的重新盖住—再预测"，逐步迭代解出完整序列。

### 3.2 模型架构与动作 tokenization

LLaDA-VLA 由三部分组成：语言骨干（LLaDA，权重来自 LLaDA-V）、视觉编码器（SigLIP-2）、以及连接两者的 MLP projector。输入为语言指令和前视 RGB 图像；视觉特征经 projector 投影后与文本 token 拼接，一起送入大语言扩散模型生成动作 token。

动作 tokenization：把连续动作值离散化为若干 bin，词表扩充 $\mathcal V_a$ 个特殊动作 token $\mathcal S=\{s_0,\dots,s_{\mathcal V_a-1}\}$（$\mathcal V_a\ll\mathcal V$，总词表 $\mathcal V_{total}=\mathcal V+\mathcal V_a$，实验中新增 32 个特殊 token）。每个时间步的动作由 $D=7$ 个特殊动作 token 表示（3 个位置位移、3 个旋转变化、1 个夹爪开合状态）。为生成多步轨迹，模型一次预测跨 $K$ 个连续时间步的动作 chunk，产生 $K\times D$ 个特殊动作 token（主实验 $K=5$，即 35 个 token 一次性生成），可反 tokenize 还原为连续动作执行。

### 3.3 局部化特殊 token 分类（Localized Special-token Classification, LSC）

预训练 d-VLM 的原始训练目标是在整个词表 $\mathcal V_{total}$ 上做全词表分类，直接沿用会不必要地放大域差异、增大动作生成的适配难度。LSC 把分类空间收窄到特殊动作 token 子集 $\mathcal S$：训练时把原始 token 标签映射为局部类别索引，

$$l_i=\begin{cases}\text{map}(y_i), & y_i\in\mathcal S\\ -100, & \text{否则（该位置忽略 loss）}\end{cases}$$

只取特殊动作 token 对应的 logits $z_i=\text{logits}[i,\mathcal S]\in\mathbb{R}^{\mathcal V_a}$，在被 mask 的位置集合 $M$ 上计算 token 级交叉熵：

$$L_{\text{token}}=\frac{1}{|M|}\sum_{i\in M}\text{CE}(z_i,l_i)$$

用大白话说：与其让模型在几万甚至十几万词的全词表里去猜"这个动作 token 到底是哪个词"，不如只让它在 32 个动作专用 token 里做选择题——把一个开放的大词表分类问题收窄成一个封闭的小规模分类问题，从而降低预训练 d-VLM 迁移到机器人领域时的学习难度、提升训练效率。

### 3.4 层级化动作结构解码（Hierarchical Action-structured Decoding, HAD）

原始 LLaDA 的解码策略对所有输出 token 一视同仁：从全 mask 序列出发，每步预测全部 mask 位置并保留高置信度 token，将剩余低置信度 token 重新 mask，如此迭代。但这种策略无法区分动作 chunk 内部的结构依赖——同一个动作的多个维度 token 之间（intra-action）以及连续多个动作之间（inter-action）实际上存在强相关性。HAD 分两个层级显式建模这种依赖：先计算每个动作的动作级置信度，即该动作内所有 token 置信度之和，

$$C_a^{(i)}=\sum_{j=1}^D c_{i,j}$$

用大白话说：不再把一个动作的 7 个 token 当作 7 个互不相干的决定，而是用它们置信度之和衡量"这一步动作整体靠不靠谱"。

具体解码流程：（1）动作级——对 chunk 内所有动作按 $C_a^{(i)}$ 排序，选出置信度最高的动作予以部分保留，其余动作全部重新 mask（action-level remask）；（2）token 级——在被选中的动作内部，再按 token 级置信度排序，只保留高置信度的子集 token，其余 token 重新 mask（token-level remask）。被 remask 的 token 在后续扩散步骤中重新生成。这样逐动作、逐 token 地迭代生成，使轨迹按"先确定哪一步动作最可信、再细化该动作内部各维度"的方式展开，兼顾了轨迹的结构完整性与局部细化。

训练与推理细节：基座为 LLaDA-V 开源 d-VLM 权重；微调 3 个 epoch，学习率 2e-5，batch size 128；因采用固定长度输出设置移除了 EOS token；主实验 action chunk size 设为 5，模型预测 delta action；推理阶段用 10 步迭代扩散（每个动作 2 次迭代，对应 chunk 内 5 个动作），并采用 dllm-cache 方法加速解码。

## 四、关键结果

**SimplerEnv（WidowX，Visual Matching 设置，4 任务成功率 %）**：

| 方法 | Put Spoon on Towel | Put Carrot on Plate | Stack Green on Yellow | Put Eggplant in Basket | 平均 |
|---|---|---|---|---|---|
| RT-1-X | 0.0 | 4.2 | 0.0 | 0.0 | 1.1 |
| Octo-Base | 15.8 | 12.5 | 0.0 | 41.7 | 17.5 |
| OpenVLA | 4.2 | 0.0 | 0.0 | 12.5 | 4.2 |
| Cog-ACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| DiscreteDiffusionVLA | 37.5 | – | 20.8 | 29.2 | 29.2 |
| **LLaDA-VLA** | 56.9 | 76.3 | 30.6 | 58.3 | **55.5** |

**CALVIN ABC-D（连续 5 任务成功率链 %，Avg. Len. 为平均连续完成任务数）**：

| 方法 | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|---|---|---|---|---|---|---|
| RoboFlamingo | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47 |
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| 3D Diffusor Actor | 92.2 | 78.7 | 63.9 | 51.2 | 41.2 | 3.27 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| **LLaDA-VLA** | 95.6 | 87.8 | 79.5 | 73.9 | 64.5 | **4.01** |

**真实 WidowX 机器人（4 个 in-domain 任务，成功率 %）**：

| 方法 | Banana on Plate | Strawberry in Bowl | Starfruit on Plate | Banana&Strawberry in Bowl | 平均 |
|---|---|---|---|---|---|
| π0 | 50 | 30 | 40 | 20 | 35 |
| CogACT | 40 | 30 | 30 | 20 | 30 |
| **LLaDA-VLA** | 50 | 70 | 70 | 40 | **58** |

**真实机器人 OOD 泛化（未见物体/容器/干扰物，成功率 %）**：

| 方法 | Cube on Plate | Strawberry in Box | Cube in Box | Banana&Strawberry in Bowl(干扰) | 平均 |
|---|---|---|---|---|---|
| π0 | 30 | 20 | 10 | 0 | 15 |
| **LLaDA-VLA** | 50 | 60 | 50 | 0 | **40** |

**消融（CALVIN ABC-D，逐步加入两个设计）**：baseline（无 LSC 无 HAD）Avg. Len. 2.64；+LSC 提升至 3.43（+0.79）；+HAD 进一步提升至 4.01（+0.58）。两个设计都带来实质性增益，其中 LSC 贡献了更大的绝对提升，说明降低分类空间是适配 d-VLM 到机器人域最关键的一步；HAD 在此基础上进一步显式建模层级依赖带来额外增益。

**Action chunk size 消融（CALVIN）**：chunk=5 时 Avg. Len. 最高（4.01）；chunk=3 时为 3.90，chunk=8/10 时分别降至 3.53/3.36。说明 chunk 过小会牺牲轨迹平滑性，chunk 过大则因单次需要预测的 mask token 数增多而使预测任务本身变难，需要在轨迹平滑度与预测准确率之间权衡。

## 五、评价与展望

优点：LLaDA-VLA 首次系统性地把"整个 VLM 骨干本身即为掩码扩散模型"这条路线引入 VLA（区别于 π0/CogACT 那类"ARM-VLM 骨干 + 扩散/flow-matching 动作头"的主流做法），验证了 d-VLM 本身可以直接作为 VLA 骨干并取得有竞争力的性能，是对 VLA 骨干选择空间的一次有意义拓展。两项设计（LSC 收窄分类空间、HAD 显式建模动作层级依赖）思路简单、消融充分，其中 LSC 的"收窄输出空间以降低域迁移难度"的思路具有一定通用性，可能迁移到其他 d-VLM 下游任务的适配场景；HAD 对 intra-/inter-action 依赖的显式建模，也从另一个角度呼应了 ACT 的 action chunking、π0 的 flow matching 等工作对轨迹平滑性/连贯性的强调。真机实验（含 4 个 OOD 泛化任务）提供了不错的实证支持。

局限与开放问题：（1）评测规模有限——SimplerEnv 仅覆盖 WidowX 的 4 个任务，真实机器人 in-domain 与 OOD 各仅 4 个任务、每任务 10 次 trial，样本量偏小，未见 SimplerEnv Google Robot 设置或 LIBERO 等更大规模基准上的结果；（2）动作表示仍是离散化（32 bins）token，论文未对轨迹平滑度、连续动作精度做与 π0 类连续动作头方法的直接定量对比，只比较了任务成功率；（3）推理效率未见量化报告——虽引入 dllm-cache 加速，但 10 步迭代扩散加 action chunk 的多轮解码相对 ARM 单次前向或扩散动作头，实际推理延迟/控制频率没有给出具体数字，这对实时闭环控制是重要的工程指标；（4）扩散步数（10 步）与 chunk size（5）更多是经验选择，论文只对 chunk size 做了消融，未见扩散步数-精度的系统权衡曲线；（5）与并发工作 Discrete Diffusion VLA 路线接近（都用离散扩散做动作解码），二者的实质性差异（d-VLM 骨干 vs. ARM-VLM+扩散头）是否在更大规模、更多任务上依然成立，仍需后续工作验证。

## 参考

- Nie et al. *LLaDA: Large Language Diffusion Models*, arXiv:2502.09992 — 本文语言扩散骨干的原始来源。
- You et al. *LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning*, arXiv:2505.16933 — 本文直接使用的预训练 d-VLM 权重。
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246 — 主要 ARM-VLA 基线。
- Li et al. *CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action*, arXiv:2411.19650 — 扩散动作头 VLA 基线。
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164 — flow matching 动作 VLA 基线。
- Liang et al. *Discrete Diffusion VLA*, arXiv:2508.20072 — 同期工作，同样探索离散扩散动作解码。
