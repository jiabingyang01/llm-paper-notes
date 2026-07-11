# VideoVLA：视频生成模型可以成为可泛化的机器人操作器

> **论文**：*VideoVLA: Video Generators Can Be Generalizable Robot Manipulators*
>
> **作者**：Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, Baining Guo
>
> **机构**：IAIR（西安交通大学人工智能与机器人研究所）、Microsoft Research Asia、复旦大学
>
> **发布时间**：2025 年 12 月（arXiv 2512.06963）
>
> **发表状态**：NeurIPS 2025（第 39 届神经信息处理系统大会）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.06963) | [PDF](https://arxiv.org/pdf/2512.06963)
>
> **分类标签**：`VLA` `视频生成即操作` `Diffusion Transformer` `Video-Action联合去噪` `CogVideoX` `跨embodiment泛化`

---

## 一句话总结

把预训练的视频生成大模型（CogVideoX-5B）直接改造成"视频-动作"联合扩散 Transformer，用同一套 DiT 权重在同一去噪过程里同时预测未来视觉帧和 7 维动作序列，凭借视频生成模型自带的强泛化先验，在 SIMPLER 仿真新物体/新技能泛化上大幅超过 π0、CogACT、SpatialVLA 等主流 VLA（新技能任务平均成功率 48.6% vs 次优 CogACT 的 20.4%），并观察到"想象画面质量"与"实际执行成功率"存在强相关。

## 一、问题与动机

当前主流 VLA（Vision-Language-Action）模型（OpenVLA、CogACT、π0、SpatialVLA 等）普遍以预训练的视觉-语言**理解**模型（如 CLIP、SigLIP、InternVL、Qwen2-VL）为骨干，再在机器人动作数据上微调。这类模型虽然借助大规模图文预训练提升了任务成功率，但对未见物体、未见技能、未见 embodiment 的真正泛化能力依然有限。

与此同时，大规模视频生成模型（Stable Video Diffusion、CogVideoX、HunyuanVideo、Wan 等）在条件生成新文本/新图像时展现出惊人的物理合理性和泛化能力——这源于其在海量真实世界视频上学到的物理动态知识。作者观察到：视频生成模型处理"新文本+新图像条件"的能力，与机器人操作模型处理"新指令+新观测"的能力，在本质上是同构的；而"预测动作执行后世界如何变化"这件事本身，也正是视频生成模型所擅长的"预测未来帧"。

由此提出核心问题："大规模视频生成器能否被无缝改造为可泛化的机器人操作器？" 论文的关键挑战在于：如何在视频生成模型中加入动作模态作为新的输出，同时保证生成视频所代表的"视觉想象"与实际执行的动作在语义和物理上保持一致，从而把视频生成域学到的强泛化能力迁移到动作域。

## 二、核心方法

### 2.1 问题形式化

给定文本指令 $\mathcal{T}$ 和当前视觉观测 $\mathcal{O}$，模型联合预测：

1. 动作块 $\mathcal{A} = \{a_i \in \mathbb{R}^7\}_{i=1}^K$：每个 $a_i$ 是 7 维向量（3 维手腕旋转 + 3 维手腕平移 + 1 维夹爪开合状态，0 表示闭合、1 表示打开）；
2. 未来视频片段 $\mathcal{F} = \{F_j\}_{j=1}^N$：描述执行 $\mathcal{A}$ 后环境会如何变化的 $N$ 帧画面。实现上并不直接预测像素帧，而是预测其在 VAE 潜空间的表示，兼顾效率。

推理是闭环的：模型预测出动作块后，机器人执行其中若干步，获得新观测，再重复上述预测过程（receding horizon 控制），直至任务完成。$\mathcal{A}$ 与 $\mathcal{F}$ 的帧率可以不同——一个动作可对应视频中多帧。

### 2.2 模型架构：Video-Action 联合 DiT

VideoVLA 的骨架是一个 DiT（Diffusion Transformer）风格网络，直接由预训练视频生成模型 CogVideoX-5B 初始化，将动作作为新增输出模态，与语言、视频统一进多模态 token 序列联合去噪：

- **文本编码器**：T5，把指令编码为定长 226 个 token，记作 $\boldsymbol{T}$；
- **视频编码器**：CogVideoX 的 3D-causal VAE，把视频片段 $\mathcal{F}=\{F_j\}_{j=1}^N$ 编码为潜变量序列 $\mathcal{V}=\{V_j\}_{j=1}^n$。由于是因果设计，第一个潜变量 $V_1$ 仅编码首帧 $F_1$，即当前观测 $\mathcal{O}$ 的潜表示；推理时只需编码当前观测得到 $V_1$，训练时则编码整段视频得到 $V_1$ 和未来帧潜变量 $\{V_j\}_{j=2}^n$；
- **动作**：不做任何 tokenization，直接使用 7 维连续向量表示，与视频/语言 token 一起投影到统一 embedding 维度。

把每个视觉潜变量在空间维度按光栅序展平为一维序列（当前帧记 $V'_1$，未来帧记 $\{V'_j\}_{j=2}^n$），构造统一多模态输入序列：拼接 $\boldsymbol{T}$、$V'_1$（条件，不加噪）、$\{V'_j\}_{j=2}^n$（待去噪目标）、$\mathcal{A}$（待去噪目标）。骨干由自注意力块堆叠而成，同时建模跨模态与跨时间的交互；扩散时间步通过 adaptive LayerNorm（沿用 DiT 做法）注入。

训练采用 DDPM，噪声调度策略同时施加于未来帧潜变量和动作：

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t,\,\epsilon}\Big[\big\| \epsilon - \epsilon_\theta\big(\{V_j'^{\,t}\}_{j=2}^n,\ \mathcal{A}^t,\ t \;\big|\; \boldsymbol{T},\ V_1'\big) \big\|_2^2\Big]$$

**用大白话说**：把"预测未来会发生什么画面"和"预测该怎么动"这两件事，捏进同一个去噪任务里训练——网络看着指令和当前这一帧，同时对"加了噪声的未来视频潜变量"和"加了噪声的动作序列"做去噪，一次前向、共享同一套注意力权重。这样"脑子里想的画面"和"手上做的动作"被迫用同一套内部表征生成，天然被约束成互相一致，而不是两个独立的、可能各说各话的模块。

推理时用 DDIM 采样（默认 50 步）对噪声输入迭代去噪，同时得到"想象视频"的潜变量和待执行动作；视频解码器（VAE decoder）是**可选**模块，只在需要可视化想象画面时才使用，动作执行本身不依赖解码。

### 2.3 训练与实现细节

- 预训练数据：Open X-Embodiment（OXE），沿用 Octo/OpenVLA/CogACT 的子集，约 2250 万帧，覆盖 22 种 embodiment；
- 真实机器人数据：自采集 5824 条样本，覆盖 pick / stack / place 三类任务，通过遥操作在配备 7-DoF 机械臂+夹爪的 Realman 机器人上采集；
- 骨干：CogVideoX-5B，预训练 10 万迭代，真机微调 1.5 万迭代；
- 硬件与优化：32 张 AMD MI300X GPU，batch size 256，AdamW，学习率 1e-5，weight decay 1e-4；
- 推理配置：默认每次预测 13 个未来帧潜变量（对应 49 帧视频）与 6 步动作，仅执行前 3 步；真实机器人为提效改为预测 4 个潜变量（对应 13 帧）；DDIM 采样步数在真机部署中降为 10 步，单次推理约 1.1 秒（H100 GPU），对应约 3Hz 的有效控制频率。

## 三、实验结果

评测分仿真（SIMPLER，含 Google Robot 与 WidowX Robot）与真实世界（Realman 机器人）两大类，各自再分"域内（in-domain）"与"泛化（新物体 / 新技能 / 跨 embodiment 技能迁移）"。对比基线：RT-1-X、RT-2-X、Octo、OpenVLA、SpatialVLA、π0、CogACT。

### 3.1 仿真域内评测（SIMPLER）

在 WidowX（Visual Matching）与 Google Robot（VM/VA）共 12 项任务上，VideoVLA 取得 WidowX-VM 平均最高、Google-VA 平均最高、Google-VM 平均次高，**全部 12 项任务总平均最高（63.0，次优 CogACT 62.6）**。

### 3.2 仿真泛化评测

**新物体**（Google Robot，10 个来自 YCB/GSO 的未见物体，如 wrench、strawberry、tennis ball 等）：

| 方法 | 平均成功率 |
|---|---|
| OpenVLA | 6.4% |
| SpatialVLA | 50.8% |
| π0 | 28.8% |
| CogACT | 42.4% |
| **VideoVLA（Ours）** | **65.2%**（10 个物体中 8 个取得最优） |

**新技能**（把 WidowX 训练数据中有、但 Google Robot 训练集中没有的 8 项技能迁移到 Google Robot，如 Put Spoon on Towel、Flip Cup、Slide to \{L,R,U,B\} 等）：

| 方法 | 平均成功率 |
|---|---|
| OpenVLA | 6.2% |
| SpatialVLA | 18.9% |
| π0 | 18.3% |
| CogACT | 20.4% |
| **VideoVLA（Ours）** | **48.6%**（超第二名 CogACT 达 28.2 个百分点） |

### 3.3 真实世界评测（Realman 机器人）

**域内**（Pick Up / Stack / Place 三类任务总平均，Task All）：OpenVLA 9.7%，SpatialVLA 22.9%，π0 50.7%，CogACT 58.4%，**VideoVLA 64.6%（最优）**。

**新物体**（12 个未见物体，Pick up后放到指定颜色盘子，平均成功率）：OpenVLA 9.6%，SpatialVLA 14.1%，π0 21.8%，CogACT 26.9%，**VideoVLA 50.6%**——其余四个基线中有近半数物体成功率为 0%，而 VideoVLA 在全部 12 个物体上均取得非零成功率，最低 16.7%，最高 83.3%。

**跨 embodiment 新技能迁移**（把仅在 WidowX 上出现过的 Move/Grab/Topple/Take Out/Wipe 等技能迁移到 Realman，物体在 Realman 训练中已见过，仅技能未见过）：

| 方法 | 平均成功率 |
|---|---|
| OpenVLA | 8.3% |
| SpatialVLA | 13.5% |
| π0 | 28.5% |
| CogACT | 35.1% |
| **VideoVLA（Ours）** | **58.0%**（对 Topple、Wipe 这类差异较大的新技能仍能部分完成） |

### 3.4 消融实验

- **预训练骨干的重要性**（SIMPLER-Google-VM，4 任务平均）：OpenSora-1.1 骨干 50.2%；CogVideoX-5B 从零训练（无预训练）12.6%；CogVideoX-5B 预训练权重初始化 **80.4%**——说明生成质量更高的预训练视频模型、以及"预训练"本身，均对下游操作性能起决定性作用。
- **预测时域长度**：预测 13/25/49 帧（对应 4/7/13 个潜变量）平均成功率分别为 75.2% / 77.4% / 80.4%，时域越长越好，说明更长的未来预测有助于模型更准确地"预判"动作后果。
- **双预测策略（视频+动作 vs 仅动作）**：默认联合预测视频与动作 domain 内平均 80.4%（新物体 65.2% / 新技能 48.6%）；只用动作去噪损失但仍联合建模视频（No video loss）domain 内平均骤降至约 27%（新物体 12.7% / 新技能 4.4%）；完全不预测视频、只做动作去噪（Action only）domain 内平均约 25.5%（新物体 11.3% / 新技能 2.1%）。可见联合预测视频对域内表现和（尤其是）泛化能力都至关重要。
- **想象-执行一致性分析**（Section 4.4）：用 SIFT 关键点 + SAM 分割前景 + SAM-PT 跟踪，抽取"想象视频"与"实际执行"轨迹并做匈牙利匹配计算余弦相似度，Figure 3 显示：成功执行对应的想象-实际运动相似度显著高于失败执行（Google Robot 与 WidowX Robot 上均如此）。人工评估想象视频质量（Table 10）：新物体设定下想象成功率 84.0% vs 实际执行成功率 65.2%；新技能设定下想象成功率 63.4% vs 实际执行成功率 48.6%——想象质量始终高于实际执行，符合"想得到不一定做得到"的直觉，但两者高度正相关。
- **双向注意力 vs 因果掩码**（附录）：默认双向注意力（动作 token 与视频 token 互相可见）平均 80.4%；施加因果掩码（视频 token 看不到动作 token）后降至 75.5%。
- **同步 vs 异步扩散调度**（附录）：视频与动作共享同一扩散时间步（默认）平均 80.4%；训练异步/推理同步降至 73.8%；训练与推理均异步（先去噪视频再据此去噪动作，即两阶段）降至 71.0%。

## 四、局限性

论文附录明确指出的核心局限是**推理速度**：真实部署中预测 4 个未来潜变量（13 帧视频）+ 6 步动作，用 10 步 DDIM 去噪，单次推理约 1.1 秒（H100 GPU），有效控制频率仅约 3Hz，明显慢于传统 VLA（如 OpenVLA/CogACT 通常可达数十 Hz）。这一瓶颈根源在于依赖大规模预训练视频生成器（CogVideoX-5B，5B 参数量级）。作者提出的改进方向包括：预训练一个专为机器人场景设计、参数量更小的视频生成器，采用如 ShortCut 模型等一步去噪技术，以及模型蒸馏，但这些均未在本文中实现，留作未来工作。

此外，从实验设计本身看还存在以下未讨论或未覆盖的局限：（1）仅验证了单臂 7-DoF 抓取类操作，未涉及双臂协同或灵巧手等更复杂的动作空间；（2）真实世界预训练/微调数据规模有限（5824 条样本，仅 pick/stack/place 三类任务），真实世界的泛化评测任务类型也相对单一（均为"拾取放置"范式）；（3）未来帧以 VAE 潜变量形式预测而非像素，虽提升效率，但潜变量层面的"视觉想象"质量与真实像素解码质量之间的关系未做系统分析；（4）消融显示模型对预训练视频生成器的质量高度敏感（OpenSora-1.1 骨干平均分骤降至 50.2%），意味着方法的泛化优势很大程度上"借用"了 CogVideoX 这一特定强模型的先验，可迁移性依赖于视频生成基座本身的持续进步。

## 五、评价与展望

**优点**：VideoVLA 提出了一条与主流 VLA 范式（依赖视觉-语言理解模型）正交的路径——直接复用视频生成模型的生成式先验，并用一个统一 DiT 架构把"预测未来视觉后果"与"预测动作"绑定在同一去噪过程中训练，设计简洁（无需额外的逆动力学模块或分阶段规划-执行流水线）。实验证据链条比较完整：域内、新物体、新技能、跨 embodiment 迁移、仿真与真机均一致地领先 π0、CogACT、SpatialVLA 等强基线，尤其在新技能迁移（+28.2 点）和真实世界新物体（+23.7 点对比次优）上优势明显；并用可解释的运动相似度分析和人工评估，直接验证了"想象质量与执行成功率相关"这一核心假设，而不只是端到端黑箱地报告成功率数字。

**与相关公开工作的关系**：论文将自己定位为该方向的先驱，明确对比了同期最相关的两个工作——UVA（Unified Video Action Model）与 VPP（Video Prediction Policy），指出区别在于：（1）VideoVLA 更充分地利用了大规模预训练视频生成器的能力；（2）系统性验证了对新物体、新技能的强泛化；（3）揭示了预测动作与视觉想象之间的强相关性；（4）在性能上对齐甚至超越了 π0、CogACT 这类新近的强 VLA 基线。相较于另一类"视频生成 + 逆动力学"模块化方案（如 UniPi、RoboDreamer、GR-2、VidMan 等，先用视频模型做视觉规划，再单独提取或预测动作），VideoVLA 走的是端到端联合训练路线，动作与视频共享同一权重和注意力，这类设计在直觉上更容易保证"想"与"做"一致，但也让整个系统的推理成本与视频生成器强绑定。

**开放问题与可能改进方向**：（1）3Hz 的控制频率对高动态、接触密集型任务（如插拔、翻转类精细操作）可能不够用，如何在保持视频先验泛化能力的同时大幅压缩推理延迟（小型专用视频生成器、少步/一步扩散、蒸馏）是最直接的后续工作；（2）双臂、灵巧手等更高自由度动作空间下，"视频想象"是否仍能提供同等程度的泛化收益尚待验证；（3）当前仅在潜空间层面做想象-动作一致性约束，是否可以引入显式的像素级或结构化（如关键点、深度）监督进一步加强物理一致性；（4）该范式对预训练视频生成器质量的强依赖，意味着其上限被基座视频模型的物理合理性所约束——随着 Wan、HunyuanVideo 等更强视频生成模型的出现，VideoVLA 这类"生成器即操作器"的范式收益预计将持续提升，这也是与依赖理解模型的传统 VLA 范式相比更具长期潜力（但也更依赖生成式基座竞赛）的地方。

## 参考

1. Yang et al. *CogVideoX: Text-to-video diffusion models with an expert transformer.* arXiv:2408.06072, 2024.（VideoVLA 的预训练视频生成骨干）
2. Li et al. *CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation.* arXiv:2411.19650, 2024.（主要对比基线之一）
3. Black et al. *π0: A vision-language-action flow model for general robot control.* arXiv:2410.24164, 2024.（主要对比基线之一）
4. Li, Gao, Sadigh, Song. *Unified Video Action Model.* arXiv:2503.00200, 2025.（UVA，同期最相关的视频-动作联合建模工作）
5. O'Neill et al. *Open X-Embodiment: Robotic learning datasets and RT-X models.* arXiv:2310.08864, 2023.（预训练所用 OXE 数据集）
