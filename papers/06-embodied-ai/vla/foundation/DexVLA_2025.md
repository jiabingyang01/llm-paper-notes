# DexVLA：面向通用机器人控制的即插式扩散专家视觉-语言-动作模型

> **论文**：*DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control*
>
> **作者**：Junjie Wen, Yichen Zhu et al.
>
> **机构**：Midea Group；East China Normal University；Shanghai University
>
> **发布时间**：2025 年 02 月（arXiv 2502.05855）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2502.05855) | [PDF](https://arxiv.org/pdf/2502.05855)
>
> **分类标签**：`扩散专家VLA` `跨embodiment预训练` `具身课程学习` `子步推理` `长时程灵巧操作`

---

## 一句话总结

DexVLA 把动作专家从常见的百万参数级 Diffusion Policy 扩大到 **10 亿参数**的多头扩散 Transformer，配合"跨embodiment预训练 → embodiment 对齐 → 任务后训练"三阶段**具身课程学习**，并让 VLM 自己生成子步推理token 取代外部高层规划器（如 π0 依赖的 SayCan），使模型在仅约 100 小时数据、单张 A6000 上 60Hz 推理的条件下，在叠衣、装箱、清桌等长时程灵巧任务上全面超过 OpenVLA、Octo、Diffusion Policy，并在部分任务上超过 π0。

## 一、问题与动机

作者指出当前 VLA 模型面临两个瓶颈：(1) **数据稀缺**——OpenVLA、Octo 依赖 Open-X Embodiment（4000 小时）甚至 π0/π0.5 用到的万小时级数据，人类示教采集成本极高；(2) **架构失衡**——现有工作普遍把参数堆在 VLM 侧（OpenVLA 用 7B VLM，π0 用 3B VLM），而负责真正与物理世界交互的动作表示/动作专家部分相对薄弱，未与 VLM 的视觉-语言理解能力形成对等匹配。此外，π0 这类模型在处理叠衣服、连续 bin-picking 等长时程、强接触任务时，仍需借助 SayCan 等外部高层策略每隔固定时间（如 2 秒）重新给出子指令，整个系统并非端到端。

DexVLA 的目标是：用一个**可插拔的十亿参数扩散专家**承担跨 embodiment 的底层运动技能学习，同时让 VLM 骨干通过自生成的子步推理直接承担高层规划职能，从而在数据效率和长时程任务完成度上同时取得提升。

## 二、核心方法

### 2.1 整体架构

DexVLA 以 **Qwen2-VL（2B）**为 VLM 骨干，图像观测经图像编码器投影进与语言相同的 token 空间；VLM 同时输出**推理 token**和**动作 token**两类内容。动作 token 经过一个由两层线性+LayerNorm 组成的投影模块（类比 LLaVA 的 connector）映射到扩散专家的输入空间；推理 token 则通过 **FiLM 层**对扩散专家内部投影层的参数做 scale-and-shift 调制，把 VLM 自主生成的高层推理注入到动作生成过程中。

动作专家基于 **ScaleDP**（Diffusion Policy 的 Transformer 变体）搭建，规模扩大到 **1B 参数**（32 层，隐藏维度 1280，16 个注意力头）。由于原始 Diffusion Policy 不支持跨 embodiment 训练，DexVLA 借鉴 Octo 的做法为**每个 embodiment 分配独立的 MLP 输出头**（multi-head），使同一套骨干网络能在动作维度、控制频率各异的机器人之间共享底层参数。

训练目标为扩散损失与下一token 预测损失的加权和：

$$L = L_{diff} + \alpha L_{ntp}$$

用大白话说：$L_{diff}$ 是扩散专家去噪重建动作序列的损失，$L_{ntp}$ 是 VLM 自回归生成推理文本 token 的语言建模损失。全部实验中取 $\alpha=1$，因为作者观察到 $L_{ntp}$ 在训练早期就已收敛，之后梯度主要来自动作损失，所以简单等权相加即可让模型把重心放在学习"依据推理与指令预测动作"上。

### 2.2 三阶段具身课程学习（Embodied Curriculum Learning）

类比人类先学通用动作、再适应自身身体、最后打磨专项技能的学习过程，DexVLA 设计三阶段训练：

- **Stage 1：跨embodiment预训练**。只训练扩散专家，暂时与 VLM 解耦——用随机初始化的 ResNet-50 作图像编码器、DistilBERT 作语言编码器，通过 FiLM 层融合语言与视觉特征。因为不涉及 VLM，扩散专家可以吃遍所有 embodiment 的数据，学习通用底层运动技能，并借助 multi-head 结构避免不同 embodiment 之间的动作空间冲突。
- **Stage 2：embodiment 特定对齐**。只用单一目标 embodiment 的数据，联合训练 VLM、投影层与扩散专家（冻结 VLM 视觉编码器），把 VLM 的抽象视觉-语言表示对齐到目标机器人的物理约束上。论文强调该阶段单独即可让模型完成叠衣服、bin-picking 等域内任务，无需第三阶段。
- **Stage 3：任务特定后训练**。针对折叠皱缩衣物、连续清桌等强上下文依赖、长时程任务，用高质量专家演示做进一步后训练，使模型学会稳定流畅地完成复杂任务。

### 2.3 子步推理（Substep Reasoning）：把 SayCan 内化进模型

关键设计是把长时程任务（如叠衣服、清桌，单条轨迹常超过 2 分钟）标注为若干子步指令（如"抚平褶皱""对齐袖子""固定折痕"），并在 Stage 2、Stage 3 中**把子步推理作为模型的中间语言输出而非输入**——即模型在生成动作的同时自主生成子步描述，充当隐式高层规划器。这与 π0 依赖外部 SayCan、每固定 2 秒更新一次指令的方式形成对比：SayCan 的固定频率更新可能在某些场景下产生冗余或重复的子目标，而 DexVLA 的子步推理随任务状态自适应切分，作者的消融显示这种隐式规划显著优于替换为 SayCan 的版本。

子步标注数据的获取采用了两套流水线：对物体级任务（bin picking、sorting、清桌）用 Grounding-DINO + DINOv2 检测物体与夹爪包围盒，通过夹爪-物体 IoU 判断抓取是否成功；对长时程单物体任务（叠衣服）先人工列出候选子步列表，再用 Google Gemini 2.0 对视频做分段并从列表中选取对应推理描述（每个子步最短约 5 秒，避免过度切分）。仅 Stage 3 数据经过人工复核。

## 三、实验结果

评测覆盖 4 种 embodiment（单臂 Franka+夹爪、Franka+灵巧手、双臂 UR5e、双臂 AgileX），预训练数据约 100 小时、91 个任务（AgileX ARX 臂占 42.7%、AgileX PIPER 臂 4.4%、单臂 UR5e 18.2%、单臂 Franka 34.7%）。所有分数为 10 次试验的归一化平均分（0–1）。

**无任务特定后训练（仅 Stage 2）**：

| 任务（embodiment） | Diffusion Policy / Octo / OpenVLA | DexVLA |
|---|---|---|
| Shirt Folding（AgileX 双臂） | 均接近 0（三者均未能完成任一步骤） | **0.92** |
| Bin Picking easy（Franka+夹爪） | 均较低（<0.3） | **0.77** |
| Bussing Table easy（AgileX 双臂） | 均较低（<0.3） | **0.72** |

**新 embodiment 上学习灵巧技能（<100 条演示，微调 Stage 2 权重）**：

| 任务（embodiment） | Diffusion Policy（从零训练） | Octo / OpenVLA | DexVLA |
|---|---|---|---|
| Packing（双臂 UR5e） | 0.35 | 0 / 0 | **0.95** |
| Drink Pouring（Franka+灵巧手） | 0 | 0 / 0 | **0.85** |
| 两任务平均 | – | – | **0.90** |

**长时程任务、直接语言 prompt（含 Stage 3 后训练）**：

| 任务（AgileX 双臂） | Octo / OpenVLA | $\pi_0$ | DexVLA |
|---|---|---|---|
| Laundry Folding | 0 / 0 | 0.2 | **0.4** |
| Bussing Table Hard | 0 / 0 | 0.63 | **0.70**（超出 $\pi_0$ 0.08 分） |
| Dryer Unloading | 0 / 0 | – | **0.8**（对比 Octo/OpenVLA） |

**消融——三阶段训练必要性（Table 2，均为 Shirt Folding / Laundry Folding 分数）**：

| Stage1 | Stage2 | Stage3 | Shirt Folding | Laundry Folding |
|---|---|---|---|---|
| ✓ | – | – | 0.0 | 0.0 |
| – | ✓ | – | 0.0 | 0.0 |
| ✓ | ✓ | – | 0.92 | 0.0 |
| ✓ | ✓ | ✓ | 0.92 | **0.4** |

去掉 Stage 1 或 Stage 2 中任一阶段都会导致完全失败（0 分），说明二者不可互相替代；Stage 3 对于攻克长时程任务是必要的（Laundry Folding 从 0 提升到 0.4）。

**消融——扩散专家规模（Table 3，Shirt Folding 分数）**：UNet（93M）**0.17** ＜ Diff Expert（410M）**0.63** ＜ Diff Expert（1B，DexVLA 采用）**0.92**，验证了把动作专家扩到十亿参数级确有必要，作者认为小模型在多任务参数空间中更容易产生干扰。

**消融——子步推理（Table 7/8，Shirt Folding / Bussing Table Hard）**：Stage1+Stage2 均不用子步推理（纯直接 prompt）时叠衣分数从 0.92 骤降到 **0**；仅 Stage1 用子步、Stage2 不用则为 **0.07**。用 SayCan 替代 DexVLA 隐式子步推理，清桌(hard)分数从 0.70 降到 **0.58**。

**其他结果**：LIBERO 基准（Table 5）DexVLA 取得 Spatial/Object/Goal/Average = 97.2/99.1/95.6/**97.3**，略高于 $\pi_0$（96.8/98.8/95.8/97.1）与 $\pi_0$-FAST（93.9）。训练成本（Table 6）：只训扩散专家（Stage 1）的吞吐量为 0.89 epoch/小时，比训练整个 VLA（0.32 epoch/小时）快 **2.78 倍**。零样本跨形态迁移（Appendix A.5）：把 Stage 2 训好的 Franka+夹爪模型直接换装 Inspire 灵巧手（限制为单自由度）部署到 30 个训练时未见过的物体上做 bin-picking，无需任何再训练，成功率 **60%**（原夹爪配置为 67%）。视觉泛化（Table 4）：Shirt Folding 在新物体/新场景/新物体+新场景下分别为 0.78/0.78/0.56；Drink Pouring 为 0.83/0.67/0.67。

## 四、局限性

- **对子步标注流水线的重度依赖**：消融显示去掉子步推理数据会让长时程任务分数从 0.92/0.4 直接跌到 0，说明方法的长时程能力高度依赖 Grounding-DINO+DINOv2 与 Gemini-2.0 构成的自动标注管线，且 Stage 3 数据仍需人工复核，标注成本和工程复杂度不低，难以直接扩展到互联网规模数据。
- **零样本跨形态迁移仅验证了受限自由度场景**：灵巧手在零样本迁移实验中被人为约束为单自由度（模拟夹爪开合），并未测试真正高自由度（如原生 12DoF）灵巧操作的零样本迁移，成功率也从 67% 降到 60%，作者自陈视角偏移、手部高度变化等因素带来性能损失。
- **评测规模有限**：所有真实机器人实验均为每任务 10 次试验的均值，未报告方差/置信区间；预训练涉及 91 个任务，但主文用于对比基线的任务仅约 10 个，且数据集为自建的私有 100 小时数据集，难以被第三方直接复现或与其他工作在同一数据源下公平对比。
- **LIBERO 对比未覆盖全部子集**：Table 5 只报告了 Spatial/Object/Goal 三个子集，未包含常用于衡量长时程能力的 LIBERO-Long 子集。
- **扩散专家规模的扩展规律仍是外推**：消融只测试到 93M/410M/1B 三个规模点，尚不清楚性能是否会在更大规模上继续提升或出现饱和。

## 五、评价与展望

DexVLA 的核心贡献可以概括为两点被证明都必要的设计：把动作专家做大（scale 到 1B 的 ScaleDP 多头架构）与把课程做细（三阶段具身课程学习 + 自生成子步推理）。相比 OpenVLA 把参数集中在 7B VLM、动作头相对轻量的路线，以及 Octo 用 93M 小型扩散策略跨 embodiment 训练的路线，DexVLA 提供了"VLM 轻量化（2B）+ 动作专家重量化（1B）"的另一种参数配置思路，与该团队此前的 TinyVLA、Diffusion-VLA 工作一脉相承，是同一系列思路（用扩散/去噪动作头替代或增强自回归动作 token 化）在跨 embodiment 与长时程任务上的延伸。与 π0 相比，最有价值的对比在于高层规划方式：π0 依赖外部 SayCan 定频重新规划，DexVLA 把这一职能内化为 VLM 自生成的子步推理 token，在 Bussing Table Hard 上以内化规划反超了显式调用 SayCan 的版本（0.70 vs 0.58），为"高层规划是否需要外部符号化模块"这一开放问题提供了一个正面证据。

开放问题与可能的改进方向：(1) 子步推理标注目前依赖检测器+多模态大模型的组合管线并需人工复核，如何用更弱监督（例如仅靠语言模型自我反思、或从大规模无标注视频中自动挖掘子步边界）替代这一环节，是把该框架推向更大规模数据的关键；(2) 论文的多头架构本质上仍要求预先知道 embodiment 类别以分配对应输出头，对训练时完全未见过的新 embodiment（而非仅新任务/新物体）如何做到真正零样本泛化，零样本迁移实验中仍需手动约束自由度这一点说明该问题尚未解决；(3) 扩散专家 1B 参数是否是效率-性能的最优点、更大的动作专家配合更小或更大的 VLM 骨干如何权衡，仍缺乏系统的扩展规律研究；(4) 评测任务集与基线（π0、OpenVLA、Octo）均在作者自建平台上完成，与 RoboTwin、LIBERO 之外更多公开、可复现的真实机器人基准对齐，将有助于该方向的公平横向比较。

## 参考

1. Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. Kim, M. J. et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.
3. Octo Model Team. *Octo: An Open-Source Generalist Robot Policy*. RSS, 2024.
4. Zhu, M. et al. *Scaling Diffusion Policy in Transformer to 1 Billion Parameters for Robotic Manipulation*（ScaleDP）. arXiv:2409.14411, 2024.
5. Wen, J. et al. *TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation*. arXiv:2409.12514, 2024.
