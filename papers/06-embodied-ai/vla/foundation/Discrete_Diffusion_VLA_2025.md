# Discrete Diffusion VLA：将离散扩散引入视觉-语言-动作策略的动作解码

> **论文**：*Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies*
>
> **作者**：Zhixuan Liang、Yizhuo Li、Tianshuo Yang、Chengyue Wu、Sitong Mao、Liuao Pei、Tian Nian、Shunbo Zhou、Xiaokang Yang、Jiangmiao Pang、Yao Mu、Ping Luo（通讯作者：Ping Luo, Yao Mu）
>
> **机构**：The University of Hong Kong；Shanghai AI Laboratory；Shanghai Jiao Tong University；Huawei Cloud Computing Technologies Co., Ltd.；Ola Dimensions
>
> **发布时间**：2025 年 08 月（arXiv 2508.20072）
>
> **发表状态**：已录用，*Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*, Seoul, South Korea, PMLR 306
>
> 🔗 [arXiv](https://arxiv.org/abs/2508.20072) | [PDF](https://arxiv.org/pdf/2508.20072)
>
> **分类标签**：`VLA` `离散扩散` `动作解码` `并行解码` `视觉语言先验保留`

---

## 一句话总结

在统一 transformer 骨干内部用与 VLM 相同的 token 级交叉熵目标做**离散扩散**式动作解码（bidirectional attention + 置信度自适应解码顺序 + 二次 re-masking 纠错），在 LIBERO 达到 96.4% 平均成功率（离散化方法中 SOTA）、SimplerEnv-Fractal 71.2% visual matching / 64.1% overall、SimplerEnv-Bridge 54.2% overall 均取得跨离散/连续方法的 SOTA，且 OOD 场景下语言退化仅 0.8%（vs 并行解码 8.0%、连续扩散 2.4%）、视觉退化 20.4%（vs 连续扩散 29.0%），推理 NFE 从 AR 的 56 降到 12（4.7 倍加速）。

## 一、问题与动机

当前 VLA 的动作生成主要有两条路线，各有明显缺陷：

1. **自回归（AR）离散 token 解码**（如 OpenVLA、π0-FAST）：沿用 GPT 式从左到右固定顺序逐 token 预测。问题是——顺序固定导致 compounding error 逐步放大；每个 chunk 需要 $L=H\times D_{act}$ 次串行前向（LIBERO 中 $L=56$），推理延迟随动作 horizon 线性增长；且无法利用同一 chunk 内"后面"动作 token 的上下文信息来修正"前面"的预测。
2. **独立连续扩散/flow-matching 动作头**（如 π0、SmolVLA、Diffusion Policy）：通常挂在 VLM 输出的隐向量之外，用 MLP 或独立 diffusion 网络把隐向量映射成连续动作轨迹。即便像 Transfusion（π0 中的一种实现）把 diffusion 集成进同一个架构，训练时仍然保留 diffusion 专属的损失函数和优化目标，与 VLM 自身的 token 预测目标存在**梯度竞争**，被作者指出是导致预训练视觉-语言先验退化的关键原因——已有工作（Yang et al., 2026b；Liu et al., 2025）也观察到连续扩散头的 VLA 在分布外（OOD）场景下"过度依赖视觉"、语言指令泛化能力弱。

本文要解决的核心问题是：**能否设计一种动作解码机制，既能像连续扩散一样并行、迭代地生成高精度动作，又能与 VLM 骨干共享同一套优化目标（避免梯度竞争），从而保留预训练视觉-语言先验？**

作者借鉴语言/多模态生成领域近年的离散扩散与离散流匹配进展（D3PM、MaskGIT、DiffusionBERT、LLaDA 等），把动作 chunk 离散化为 token 序列，用 masked-token 扩散范式在同一个 transformer 内联合建模视觉、语言、动作。

## 二、核心方法

### 1. 动作 tokenization 与 chunk 化

沿用 RT 系列 / OpenVLA 的离散化方案：每个控制维度用 256-bin 分位数方案离散化（取 1–99 百分位以避免异常值影响），gripper 状态单独作为二元 token。单个时间步的动作产生 $D_{act}=7$ 个 token（3 平移 + 3 旋转 + 1 gripper），$H$ 个连续时间步组成固定长度的 chunk，共 $L=H\times D_{act}$ 个离散动作 token。这种定长 chunk 天然适配离散扩散的分块并行生成范式（LIBERO / SimplerEnv-Fractal 用 chunk size 8，SimplerEnv-Bridge 用 chunk size 3）。

### 2. 离散扩散建模（前向加噪）

设动作 chunk $\mathbf{a}_0=(a_{0,1},\dots,a_{0,L})$，每个 $a_{0,i}\in\{1,\dots,K\}$。引入特殊 mask token 后词表大小为 $V=K+1$。前向过程是逐 token 独立腐化的 Markov 链，转移矩阵 $\mathbf{Q}_t$ 以概率 $\beta_t$ 把 token 变为 $\mathrm{[MASK]}$、以概率 $1-\beta_t$ 保持不变：

$$
\mathbf{Q}_t\, \mathbf{e}_{a_{t,i}} = (1-\beta_t)\, \mathbf{e}_{a_{t,i}} + \beta_t\, \mathbf{e}_{\mathrm{M}}
$$

复合转移矩阵 $\bar{\mathbf{Q}}_t=\mathbf{Q}_t\cdots\mathbf{Q}_1$ 后，$t$ 时刻各位置独立地按类别分布腐化：

$$
q(\mathbf{a}_t\mid \mathbf{a}_0)=\prod_{i=1}^{L}\mathrm{Categorical}\bigl(a_{t,i}\mid \bar{\mathbf{Q}}_t\, \mathbf{e}_{a_{0,i}}\bigr)
$$

**用大白话说**：训练时随机把动作 chunk 里的一部分 token 换成"未知"占位符，网络的任务就是把这些占位符猜回原来的值——这跟 BERT 的 masked language modeling 是同一套数学。

### 3. 反向去噪与训练目标

反向条件 $p_\theta(a_{t-1,i}\mid a_{t,i},\mathbf{c})$（$\mathbf{c}$ 为视觉+语言等多模态条件）在 mask 腐化下可化简为：已揭示的 token 恒等映射，仍处于 $\mathrm{[MASK]}$ 的位置由模型预测分布采样：

$$
p_\theta(a_{t-1,i}\mid a_{t,i},\mathbf{c}) =
\begin{cases}
\delta(a_{t-1,i},a_{t,i}), & a_{t,i}\neq \mathrm{M} \\
\mathrm{Categorical}\bigl(a_{t-1,i}\mid \pi_\theta(i\mid\mathbf{c})\bigr), & a_{t,i}= \mathrm{M}
\end{cases}
$$

训练时把多步链坍缩为单一的 masked-token 预测目标：采样 mask 比例 $\gamma\sim$ schedule（如 cosine），将 $\gamma L$ 个动作位置替换为 $\mathrm{[MASK]}$ 得到 $\tilde{\mathbf{a}}_t$，只在被 mask 的位置计算交叉熵：

$$
\mathcal{L}_{CE}(\theta) = -\sum_{i\in\mathcal{M}_{\gamma_t}} \log p_\theta(a_{0,i}\mid \tilde{\mathbf{a}}_t,\mathbf{c})
$$

关键点在于：这个损失与 VLM 骨干本身的语言建模交叉熵**同源同构**（同为在共享词表空间上的 token 分类），视觉/语言 token 只参与 attention、不参与该 loss。因而不需要引入额外的 diffusion 专属损失项，从根本上避免了动作生成目标与 VLM 语言建模目标之间的梯度竞争。

### 4. 架构：统一 transformer + 双向注意力

以 OpenVLA（Prismatic-7B VLM，Llama2 backbone）为基座，把原本对动作 token 的**因果注意力**替换为**双向注意力**，使每个动作位置可以关注全部视觉、语言与其它动作 token，实现完整的跨模态融合；所有 token 经统一 transformer 后，动作位置的隐状态通过共享分类头投影到 256 类词表。视觉输入（主视角必选，腕部视角可选）由 SigLIP+DINOv2 ViT 编码后投影进 Llama2 embedding 空间；若提供本体感知状态，用轻量 MLP 编码后与视觉 token 拼接。

### 5. 推理：自适应解码 + 二次 re-masking

推理从全 mask 的动作 chunk $\mathbf{a}_1=\mathrm{M}^L$ 出发，按单调递减的 cosine schedule $\gamma_t$ 迭代 $T$ 步。第 $t$ 步只 commit 当前置信度最高的 $(1-\gamma_t)L$ 个位置，其余保持 mask：

$$
a_{t-1,i} =
\begin{cases}
\text{sample } p_\theta(\cdot\mid \mathbf{a}_t,\mathbf{c})_i, & \text{if } i \text{ in top } (1-\gamma_t)L \\
\mathrm{[MASK]}, & \text{otherwise}
\end{cases}
$$

置信度打分有两种候选指标：

$$
s_{t,i}=\max_k p_\theta(k\mid \mathbf{a}_t,\mathbf{c}) \quad(\text{Max Confidence})
$$

$$
g_{t,i}=p_\theta(k_{(1)}\mid\cdot)-p_\theta(k_{(2)}\mid\cdot) \quad(\text{Confidence Gap})
$$

被选中的高置信度位置用带温度的 tempered Gumbel-max 采样（而非纯 argmax）以保留一定探索性，温度随步数线性衰减。**用大白话说**：模型每一轮先把"最有把握"的动作 token 定下来，剩下"没把握"的留到下一轮，参考已经确定的动作再猜——这是一种由置信度驱动、逐实例自适应的解码顺序，而不是固定从左到右。

除了满足目标 mask 比例外，还引入**二次 re-masking**：对已经 commit 的 token 做阈值检查，若其置信度低于随迭代步单调递增的阈值 $\eta_t^{abs}$，则重新置为 $\mathrm{[MASK]}$，留到下一步重新预测：

$$
\mathcal{R}_t^{abs} = \{\, i\in\mathcal{K}_t : s_{t,i} < \eta_t^{abs} \,\}
$$

这一机制赋予模型"反悔"能力，用于防止早期误判在后续步骤中持续传播，实现鲁棒纠错。

## 三、实验结果

评测平台：Franka Panda 臂上的 LIBERO（Spatial/Object/Goal/Long 四套件，各 10 任务 × 50 rollouts）、Google Robot 上的 SimplerEnv-Fractal（Visual Matching / Variant Aggregation）、WidowX 臂上的 SimplerEnv-Bridge，以及 AgileX Cobot Magic 双臂真机。均只用 RGB + 末端位姿本体感知，不用深度/力觉。

**LIBERO 同分布结果（Table 1，离散化方法内 SOTA）**

| 方法 | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| Diffusion Policy (scratch) | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| MDT (scratch) | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| OpenVLA (离散 AR) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0+FAST（离散） | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| OpenVLA-OFT (Discrete) | 96.2 | 98.2 | 95.6 | 92.5 | 95.6 |
| **Discrete Diffusion VLA** | **97.2** | **99.4** | **96.8** | **92.2** | **96.4** |
| OpenVLA-OFT (L1，连续，overall SOTA) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |

在所有离散化动作方法中最优（+0.9pt 超过 OpenVLA-OFT Discrete），仅比连续动作表示的整体 SOTA（OpenVLA-OFT L1）低 0.7 个百分点——这一差距被作者归因于 bin 离散化固有的量化误差。

**LIBERO-Goal / LIBERO-Spatial OOD 鲁棒性（Table 2、3，语言复述 + 视觉外观扰动）**

| 方法 | Goal-Original | Goal-语言增强(退化) | Goal-视觉增强(退化) |
|---|---|---|---|
| OpenVLA-OFT (Discrete) | 95.6% | 87.6% ($\downarrow$8.0%) | 73.0% ($\downarrow$22.6%) |
| OpenVLA-OFT (Diffusion) | 96.0% | 93.6% ($\downarrow$2.4%) | 67.0% ($\downarrow$29.0%) |
| OpenVLA-OFT (L1) | 97.9% | 94.7% ($\downarrow$3.2%) | 74.7% ($\downarrow$23.2%) |
| **Discrete Diffusion VLA** | 96.8% | **96.0%** ($\downarrow$0.8%) | **76.4%** ($\downarrow$20.4%) |

Spatial 套件呈现同样趋势（Discrete Diffusion VLA 语言退化仅 $\downarrow$1.0%、视觉退化仅 $\downarrow$0.8%，均为四者中最小）。这一优势直接来自与预训练 VLM 相同交叉熵目标 + 双向注意力对全模态上下文的建模，而非额外正则化。

**SimplerEnv-Fractal（Google Robot）**：Visual Matching 71.2%（超过 π0 的 58.8%、π0-FAST 的 61.9%、OpenVLA-OFT 的 63.0%），Variant Aggregation 56.9%（与 RT-2-X 64.3%、π0-FAST 59.0% 同一水平），综合两指标 overall 64.1%，为跨离散/连续所有方法最高。

**SimplerEnv-Bridge（WidowX）**：overall 54.2%，SOTA，超过 π0（40.1%，+14.1pt）、GR00T-N1（49.5%，+4.7pt）、π0-FAST（48.3%，+5.9pt）。

**推理效率（Table 6，H800 单卡）**

| 方法 | 延迟 (ms) | 频率 (Hz) | NFE |
|---|---|---|---|
| OpenVLA (AR) | 136.2 | 7.34 | 56 |
| OpenVLA-OFT (并行解码，单步) | 31.1 | 32.14 | 1 |
| OpenVLA-OFT (Diffusion, 12步) | 67.1 | 14.91 | 12 |
| **Discrete Diffusion VLA (12步)** | 68.8 | 14.53 | 12 |

相比 AR 的 56 次前向，离散扩散只需 12 次（4.7 倍减少），延迟与相同步数的连续扩散基本持平，比纯 AR 快约 2 倍。

**消融**：解码策略上，Max Confidence（96.8%）> Confidence Gap（96.6%）> Random Order（95.8%）> 一次性 Parallel Decoding（95.6%），自适应解码相对一次性并行带来 +1.2pt；温度调度上线性衰减（96.8%）优于固定温度（96.4%）和硬采样/argmax（96.2%）；在不依赖 OpenVLA 机器人预训练、改用纯 VLM 骨干（Qwen2.5-VL）从零训练动作头的对照实验中，离散扩散在 LIBERO 各套件仍全面优于 AR、FAST、并行解码、连续扩散，说明该范式优势并非来自 OpenVLA 特定预训练，而是方法本身的架构增益。

**真机（AgileX Cobot Magic 双臂，click the bell / place cup on coaster，各 15 trials）**：Discrete Diffusion VLA 在 click bell 上 66.7%（OpenVLA-OFT Discrete 33.3%，π0 53.3%），在 place cup 上与 π0 打平（均 40.0%，OpenVLA-OFT Discrete 20.0%），控制频率 9.69Hz（RTX 4090），低于 π0 的 24.5Hz 与 OpenVLA-OFT Discrete 的 34.3Hz，作者说明这更多是工程实现差异而非架构必然限制。

## 四、局限性

论文第 5 节明确指出：

1. 多步迭代解码在设计上天然慢于单步并行解码（虽仍显著快于 AR）；真机部署中的控制频率（9.69Hz）明显低于其它基线。
2. 定长 chunk 的离散化 tokenization 与 π0-FAST 一类**变长**（如 DCT+BPE 压缩）动作 tokenization 方案不兼容，限制了进一步压缩 token 数、提升效率的空间。
3. 在 LIBERO 同分布评测中，相对连续动作表示的整体 SOTA（OpenVLA-OFT L1）仍有 0.7 个百分点的精度差距，源于 bin 离散化引入的量化误差，尚未被完全消除。
4. 输入模态只覆盖 RGB + 末端位姿，未引入深度/力觉/affordance 等辅助信息；真机验证仅在两个简单桌面任务、单一双臂平台上进行，长时程、精细操作或双臂协作能力尚缺乏更广泛检验。

## 五、评价与展望

**优点**：本文的核心贡献在于把"离散扩散"和"VLA 统一骨干"两者第一次真正做到了同一套优化目标下——与 Transfusion 等把 diffusion loss 直接嫁接进统一 transformer、但仍保留 diffusion 专属训练目标的做法不同，本文让动作生成复用与语言建模完全相同的 masked-token 交叉熵，理论上从根源上避免了梯度竞争，这也是其 OOD 语言/视觉退化显著更小的直接原因，实验数据（0.8% vs 2.4%/8.0% 语言退化，20.4% vs 29.0% 视觉退化）为这一论点提供了有说服力的证据。相比 AR 解码，双向注意力 + 自适应置信度顺序天然解决了固定从左到右顺序无法利用同 chunk 后续信息、以及 compounding error 的问题；二次 re-masking 提供了一种轻量的自我纠错机制，且几乎不增加计算开销（<1ms）。

**与其它公开工作的关系**：与连续扩散/流匹配动作头路线（π0、Diffusion Policy、GR00T-N1）相比，本文验证了"离散化 + 扩散"同样能达到甚至超越连续方法在 OOD 泛化上的表现，为动作表示的选择提供了新的证据（并非"连续动作表示天然更优"）；与同期探索"扩散 + 自回归混合"统一架构的 HybridVLA、以及大规模离散扩散语言模型 LLaDA/DiffuLLaMA 等工作相比，本文把该范式落地到动作模态并给出了完整的自适应解码+纠错机制设计,是较早的系统性尝试之一。

**开放问题与可能的改进方向**：
1. 能否设计出既保持并行 refinement 优势、又支持变长/自适应分辨率的动作 tokenizer（借鉴 π0-FAST 的 DCT-BPE 思路），从而进一步缩小与连续方法的精度差距、同时降低 token 数量？
2. 论文的可视化分析显示解码顺序高度依赖训练集中动作 token 的出现频率（高频 token 更早被解出），这种"频率驱动"的隐式先验在长尾/罕见动作模式（如少见的抓取姿态、紧急避障动作）上是否会出现解码顺序失配、进而影响精度，值得进一步研究。
3. 当前验证的动作 chunk 与任务时长相对有限（LIBERO/SimplerEnv 均为短时程桌面操作），该框架能否扩展到更长 horizon、更高自由度（如全身或双臂协同）的复杂操作任务，以及是否能与图像/视频生成的离散扩散联合建模、形成真正意义上的多模态统一生成模型，是值得关注的后续方向。
4. 真机推理频率（9.69Hz）与部分连续方法（π0 24.5Hz）仍有差距，如何在保留自适应解码/纠错优势的同时进一步压缩推理步数或做工程优化（如缓存复用、蒸馏更少步数的解码器），是走向高频闭环控制场景的实际工程问题。

## 参考

1. Kim, M. J. et al. OpenVLA: An Open-Source Vision-Language-Action Model. CoRL 2024.
2. Black, K. et al. π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164, 2024.
3. Pertsch, K. et al. FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747, 2025.
4. Austin, J. et al. Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM). NeurIPS 2021.
5. Chang, H. et al. MaskGIT: Masked Generative Image Transformer. CVPR 2022.
