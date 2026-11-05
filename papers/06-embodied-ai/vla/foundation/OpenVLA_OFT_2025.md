# OpenVLA-OFT：兼顾极速推理与高成功率的 VLA 优化微调范式

> **论文**：*Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*
>
> **作者**：Moo Jin Kim, Chelsea Finn, Percy Liang
>
> **机构**：Stanford University（斯坦福大学）
>
> **发布时间**：2025 年 2 月（`arXiv:2502.19645`）
>
> **发表状态**：`RSS 2025`
>
> **链接**：[arXiv](https://arxiv.org/abs/2502.19645) | [PDF](https://arxiv.org/pdf/2502.19645) | [项目主页](https://openvla-oft.github.io) | [代码](https://github.com/moojink/openvla-oft)
>
> **分类标签**：`VLA Fine-Tuning Recipe` `Parallel Decoding` `Action Chunking` `Continuous L1 Regression` `FiLM Conditioning` `Bimanual Manipulation`

## 一句话总结

针对自回归离散 VLA（如 OpenVLA）在下游微调时面临的**串行解码极慢（3–5 Hz）**、**无法高效支持动作分块（Action Chunking）** 以及**在高频双臂操作上性能不佳**三大痛点，系统研究了动作生成机制、动作表征与训练目标三个维度的微调设计空间，提出**优化微调配方（Optimized Fine-Tuning, OFT）**：将自回归因果解码替换为**双向注意力并行解码 + 动作分块（PD&AC）**、将 256-bin 离散 Token 替换为 **4 层 MLP 连续动作头**、并采用简洁的 **$L_1$ 回归损失**（在多视角双臂任务上进一步结合 **FiLM** 语言调制構成 **OFT+**），在不重做大规模预训练的前提下，将 OpenVLA 在 LIBERO 四大套件上的平均成功率从 **76.5% 跃升至 97.1%**（超越 $\pi_0$ 的 94.2%），同时实现 **26$\times$（单臂 109.7 Hz）至 43$\times$（ALOHA 双臂 77.9 Hz）** 的动作吞吐量加速。

## 核心贡献

1. **解耦预训练与微调范式，确立三合一 OFT 核心配方**：证明即使基座 VLA（OpenVLA）在预训练阶段采用的是“单视角 + 自回归因果掩码 + 离散 Token 交叉熵”，在下游微调阶段直接切换为 **（1）双向掩码并行解码与动作分块（PD&AC）+（2）连续动作表征 +（3）$L_1$ 回归目标**，不仅不会破坏预训练先验（相比无 OpenX 预训练仍高出 +5.2%），反而能彻底释放 7B 模型的表征容量。
2. **揭示“简单 $L_1$ 回归 + 大容量 VLA 骨干”媲美扩散策略的实证规律**：在严格控制变量的对比中发现，在 7B OpenVLA 骨干上，单次前向的 **$L_1$ 回归（Cont-L1）** 与需要 50 步迭代去噪的**条件扩散策略（Cont-Diffusion）** 在 LIBERO 上取得完全相当的成功率（`95.3%` vs `95.4%`），但 $L_1$ 回归训练收敛快一倍以上，且推理延迟仅为 50 步扩散的 **$1/26$**（`0.0729 s` vs `1.9070 s`），无需承受 DDIM 少步采样带来的成功率折损。
3. **提出面向多视角双臂控制的 FiLM 空间无关通道调制（OFT+）**：发现当输入扩展为多视角图像（第三人称 + 左右腕部相机）时，策略极易走捷径过拟合腕部相机的局部视觉虚假相关性，导致语言指令跟随能力退化至随机盲猜水平（33%）；通过在 SigLIP 与 DINOv2 每层自注意力之后、FFN 之前注入基于平均语言嵌入的 **FiLM（Feature-wise Linear Modulation）** 调制，使僅在单臂数据上预训练的 OpenVLA 成功跨本体迁移至 25Hz 真机 **ALOHA 双臂平台**，综合表现超越原生经双臂预训练的 **$\pi_0$** 与 **RDT-1B**。

## 1. 问题与动机：为什么原版自回归 VLA 难以胜任高频灵巧控制？

原版 OpenVLA 将 $D=7$ 维连续动作离散化为 256 个桶，并使用因果自注意力（Causal Mask）逐个 Token 自回归生成。这一范式在实际部署中存在三重结构性障碍：
1. **动作分块（Action Chunking）下的串行延迟爆炸**：动作分块（一次预测未来 $K$ 步动作序列 $\mathbf{A}_t = [\mathbf{a}_t, \dots, \mathbf{a}_{t+K-1}]$ 并开环执行）是克服马尔可夫复合误差、提升轨迹平滑度的关键技术。然而在自回归生成下，预测 $K$ 步 $D$ 维动作需要串行执行 $K \times D$ 次 Transformer 解码器前向传播（例如单臂 $K=8, D=7$ 需串行 56 步；双臂 $K=25, D=14$ 需串行 **350 步**！），单次查询耗时高达数秒，完全无法满足 25–50 Hz 的实时控制需求。
2. **分箱量化误差限制精细操作**：256-bin 均匀量化损失了亚毫米级连续控制精度；若单纯增加分箱数量，又会导致单个动作 Token 在训练语料中的出现频率稀释，损害泛化性。
3. **单视角无本体状态输入的局限**：原版 OpenVLA 仅接收单张第三人称图像，不支持腕部相机（Wrist Camera）与机器人本体关节状态（Proprioceptive State）输入。

## 2. 核心方法：OpenVLA-OFT 与 OFT+ 架构设计

### 2.1 三大核心微调设计维度的重构

OpenVLA-OFT 在保留原版 OpenVLA 预训练权重的基础上，在 LoRA（$r=32$）微调阶段对模型结构与目标函数进行了六项模块化改造：

1. **并行解码与动作分块（Parallel Decoding & Action Chunking, PD&AC）**：
   - 摒弃自回归解码中的右移真实动作输入（Teacher Forcing），改为在解码器输入序列末尾拼接 $K \times D$ 个**空动作查询嵌入（Empty Action Embeddings）** $\mathbf{E}_{\text{empty}} \in \mathbb{R}^{KD \times d_{\text{LLM}}}$（这些嵌入向量本身全为零或共享常数，仅通过各自的位置编码区分时间步 $k \in \{1,\dots,K\}$ 与动作维度 $d \in \{1,\dots,D\}$）；
   - 将解码器的因果下三角注意力掩码（Causal Mask）替换为**全双向注意力掩码（Bidirectional Attention Mask）**，使所有空动作查询位置能够在**单次前向传播**中同时互相关注并汇聚前序视觉与语言上下文，一次性并行输出全部 $K \times D$ 个动作隐状态 $\mathbf{H}_{\text{act}} \in \mathbb{R}^{K \times (D \cdot d_{\text{LLM}})}$。
2. **连续动作表征与 $L_1$ 回归目标（Continuous Action + $L_1$ Regression）**：
   - 移除 Llama-2 解码器顶部的词表线性分类头（LM Head），替换为一个 **4 层 ReLU 激活的 MLP 动作头** $f_{\text{act}}$，直接将解码器最后一层对应动作分块的隐藏状态映射为归一化到 $[-1, +1]$ 的连续动作序列 $\hat{\mathbf{A}}_t = f_{\text{act}}(\mathbf{H}_{\text{act}}) \in \mathbb{R}^{K \times D}$；
   - 直接最小化预测动作分块与专家真实动作分块之间的平均 **$L_1$ 损失**：

$$
\mathcal{L}_{\text{OFT}}(\theta) = \frac{1}{K D} \sum_{k=0}^{K-1} \sum_{d=1}^{D} \left\| \hat{a}_{t+k, d} - a_{t+k, d} \right\|_1
$$

   - 作为对比变体，作者还实现了**条件去噪扩散动作头（Cont-Diffusion）**：采用相同结构的 4 层 MLP 作为噪声预测网络 $\boldsymbol{\epsilon}_\theta(\mathbf{A}_t^\tau, \tau, \mathbf{H}_{\text{act}})$，配合余弦噪声调度（Squared Cosine Beta Schedule）与 $T_{\text{train}}=50$ 步 DDIM 采样。
3. **多视角图像与本体状态灵活扩展（Multi-View & Proprioception）**：
   - 每路相机图像（第三人称相机 + 1 至 2 路腕部相机）分别通过共享的 SigLIP-DINOv2 双视觉编码器提取 256 个 Patch 嵌入，经共享的 3 层 GELU MLP 映射为视觉 Token；
   - 机器人低维本体状态 $\mathbf{q}_t$（如关节角与夹爪状态）通过独立的 **2 层 GELU MLP** 映射为 1 个额外的状态 Token $\mathbf{h}_{\text{prop}} \in \mathbb{R}^{1\times d_{\text{LLM}}}$；
   - 最终输入解码器的多模态序列为 $\big[\mathbf{H}_{\text{vis}}^{(1)}, \dots, \mathbf{H}_{\text{vis}}^{(V)},\; \mathbf{h}_{\text{prop}},\; \mathbf{H}_{\text{lang}},\; \mathbf{E}_{\text{empty}}\big]$。

### 2.2 OFT+：基于 FiLM 的空间无关视觉特征语言调制

在包含双腕部相机的 ALOHA 双臂任务中，由于腕部相机画面随机械臂移动已经强烈暗示了正在接近的目标物体，模型在微调时极易忽略文本指令 Token $\mathbf{H}_{\text{lang}}$。为此，**OpenVLA-OFT+** 在双路视觉编码器（SigLIP 与 DINOv2）的**每一个 ViT Transformer Block 内部**（自注意力层之后、FFN 层之前）插入了 **特征线性调制（Feature-wise Linear Modulation, FiLM）**：

1. **平均语言嵌入提取与仿射投影**：对任务指令的所有语言 Token 嵌入取均值得到全局语言表征 $\bar{\mathbf{x}} \in \mathbb{R}^{d_{\text{LLM}}}$，在每个 ViT 第 $l$ 层通过独立的线性层映射出缩放向量 $\boldsymbol{\gamma}^{(l)} = f_l(\bar{\mathbf{x}}) \in \mathbb{R}^{D_{\text{ViT}}}$ 与平移向量 $\boldsymbol{\beta}^{(l)} = h_l(\bar{\mathbf{x}}) \in \mathbb{R}^{D_{\text{ViT}}}$；
2. **空间无关（Spatially-Agnostic）通道调制公式**：将 $\boldsymbol{\gamma}^{(l)}, \boldsymbol{\beta}^{(l)}$ 广播作用于该层全部 256 个图像 Patch 特征 $\mathbf{F}^{(l)} \in \mathbb{R}^{256 \times D_{\text{ViT}}}$ 的对应隐藏通道维上（初始化时 $\boldsymbol{\gamma}^{(l)}, \boldsymbol{\beta}^{(l)} \approx \mathbf{0}$，故乘以 $1 + \boldsymbol{\gamma}^{(l)}$ 以保证微调初始时刻不破坏预训练视觉激活）：

$$
\mathrm{FiLM}\!\left(\mathbf{F}^{(l)} \;\middle|\; \boldsymbol{\gamma}^{(l)}, \boldsymbol{\beta}^{(l)}\right) = \hat{\mathbf{F}}^{(l)} = \left(\mathbf{1} + \boldsymbol{\gamma}^{(l)}\right) \odot \mathbf{F}^{(l)} + \boldsymbol{\beta}^{(l)}
$$

作者特别指出：**必须沿隐藏通道维 $D_{\text{ViT}}$ 对所有 Patch 进行统一调制（模拟 CNN 中对整张特征图通道进行缩放平移）**；若改为对每个 Patch 位置独立调制，语言接地效果会大幅退化。

## 3. 实验结果与定量分析

### 3.1 LIBERO 仿真基准：设计决策消融与 SOTA 对比

表 1 与表 2 展示了在 LIBERO 四大任务套件（每个套件 10 项任务 $\times$ 500 次评估回合，动作分块 $K=8$）上的逐步消融、与主流基线的对比及 NVIDIA A100 上的推理效率实测：

| 输入设定与方法 | Spatial (%) | Object (%) | Goal (%) | Long (%) | **Average SR (%) $\uparrow$** | 吞吐量 (Hz) $\uparrow$ | 单次延迟 (s) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **单视角输入（3rd-person img + lang，过滤全零与失败轨迹）** | | | | | | | |
| Diffusion Policy (Scratch) | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 | — | — |
| Octo (Fine-tuned) | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 | — | — |
| DiT Policy (Fine-tuned) | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 | — | — |
| **OpenVLA（原版自回归离散微调）** | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 4.2 Hz | 0.2396 s |
| **OpenVLA + PD（仅并行解码，$K=1$）** | — | — | — | — | — | 15.9 Hz | **0.0629 s** |
| **OpenVLA + PD&AC（并行解码 + 分块 $K=8$，离散）** | 91.3 | 92.7 | 90.5 | 86.5 | 90.2 (+13.7) | 108.8 Hz | 0.0735 s |
| **OpenVLA + PD&AC + Cont-Diffusion（50 步 DDIM）** | **96.9** | 98.1 | 95.5 | **91.1** | **95.4** (+18.9) | 4.2 Hz | 1.9070 s |
| **OpenVLA-OFT（PD&AC + Cont-L1，单视角）** | 96.2 | **98.3** | **96.2** | 90.7 | 95.3 (+18.8) | **109.7 Hz (26$\times$)** | 0.0729 s |
| **多视角 + 本体状态输入（3rd-person + wrist img + proprio + lang）** | | | | | | | |
| $\pi_0$ + FAST (Fine-tuned) | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 | — | — |
| $\pi_0$ (Flow Matching Fine-tuned) | 96.8 | **98.8** | 95.8 | 85.2 | 94.2 | — | — |
| **OpenVLA-OFT（未过滤原始数据集）** | 95.2 | 94.2 | 95.2 | 93.2 | 94.5 | 71.4 Hz | 0.1120 s |
| **OpenVLA-OFT（完整版，过滤后数据集）** | **97.6** | 98.4 | **97.9** | **94.5** | **97.1** | 71.4 Hz | 0.1120 s |

**关键实验洞察**：
1. **PD&AC 对长程任务的质变提升**：仅将自回归解码改为并行解码加 8 步动作分块（仍保持离散动作），平均成功率就从 `76.5%` 暴涨至 `90.2%`（+13.7%），其中最难的长视野套件 `LIBERO-Long` 从 **`53.7%` 飙升至 `86.5%`（+32.8%）**，证明动作分块有效消除了单步自回归的复合误差与静止抖动。
2. **连续 $L_1$ 回归的极致性价比**：改用连续动作后，成功率再提升 +5.1% 达到 `95.3%`。当把扩散策略的测试步数从 50 步压缩至 10 步 / 5 步 / 2 步 / 1 步时（表 2），其 `LIBERO-Long` 成功率依次从 `91.1%` 降至 `91.0% -> 90.0% -> 85.7% -> 0.0%`，而单步前向的 `Cont-L1` 在 `0.0729 s` 延迟下直接达到 `90.7%`。
3. **加入腕部相机与本体状态刷新 SOTA**：得益于并行解码省下的计算余量，即使视觉 Token 数量翻倍（256 $\to$ 512），OpenVLA-OFT 仍保持 **71.4 Hz** 的高吞吐量，并将四套件平均成功率推高至 **97.1%**（其中 `LIBERO-Long` 达 **94.5%**，超越 $\pi_0$ 的 `85.2%` 达 **+9.3%**）。此外，将四个套件合并训练单一通才策略 `OpenVLA-OFT (All-in-One)` 同样取得了与分套件微调持平的高成功率。

### 3.2 真机双臂 ALOHA（25Hz）跨本体迁移实验

在包含两只 ViperX 300 S 机械臂（14 维绝对关节角动作，$K=25$，25Hz 控制频率）、3 路相机（1 顶视 + 2 腕部）的真机 **ALOHA** 平台上，评测 4 项高难度灵巧任务：`fold shorts`（叠短裤，20 条演示）、`fold shirt`（长程叠 T 恤，30 条演示）、`scoop X into bowl`（左臂移碗、右臂持金属勺舀取指定食材，45 条演示）以及 `put X into pot`（左臂开锅盖、右臂放入指定蔬菜并盖盖，含分布内与 OOD 测试）。对比从头训练的 **ACT（84M）**、**Diffusion Policy（157M）** 以及预训练过大量双臂数据的 **RDT-1B（1.2B）** 与 **$\pi_0$（3.3B）**：

1. **任务完成度与闭环纠错能力（图 4、图 6）**：
   - **OpenVLA-OFT+** 在全部 4 项任务上取得最高平均得分，超越从头训练的 ACT / Diffusion Policy 达 **15%** 以上，并超越原生双臂预训练的 $\pi_0$ 与 RDT-1B；
   - 定性分析发现：**RDT-1B** 过度依赖本体状态历史而忽视实时视觉反馈（例如左臂漏抓碗后，右臂依然对着空气倾倒食材）；**$\pi_0$** 具备良好的视觉闭环重抓能力但语言指令跟随略逊；而 **OpenVLA-OFT+** 兼具闭环视觉响应与最高语言接地准确率。
2. **FiLM 消融实验（图 5）**：在需要语言区分目标物体的 `scoop X into bowl` 和 `put X into pot`（各含 3 种候选目标）上，完整 **OpenVLA-OFT+** 的语言指令跟随率接近 **90%–100%**；一旦移除 FiLM（退化为无调制的 OpenVLA-OFT），两项任务的语言跟随成功率双双暴跌至 **33%**（完全等于 $1/3$ 随机盲猜！），证实了多视角微调中用 FiLM 打破视觉虚假相关性的必要性。
3. **双臂推理效率对比（表 3，A100 实测，3 张图 + 14D 状态，$K=25$）**：
   - 原版 **OpenVLA**（$K=1$）：吞吐量仅 **1.8 Hz**，延迟 `0.543 s`（无法驱动 25Hz 双臂）；
   - **OpenVLA-OFT+**（7.5B 参数）：吞吐量跃升至 **77.9 Hz（提速 $43.3\times$）**，单次分块延迟 `0.321 s`，以 $6\times$ 参数量逼近需 5 步去噪的 **RDT-1B**（`84.1 Hz / 0.297 s`）。

## 4. 局限性与深层启示

1. **多模态专家轨迹下的中位数折中风险**：$L_1$ 回归本质上拟合条件分布的**中位数模态（Median Mode）**。在演示策略高度一致的数据集上，$L_1$ 回归能有效平滑采集噪声；但如果训练集中针对同一状态存在多条截然不同但同样正确的绕障路径（强多峰分布），纯 $L_1$ 回归可能输出两条路径的平均轨迹，此时流匹配（Flow Matching）或扩散头仍具理论优势。
2. **核心启示——“微调配方有时比预训练数据覆盖更关键”**：原版 OpenVLA 从未在预训练中见过任何双臂机器人、多视角相机或绝对关节角控制，却仅凭 OFT+ 微调配方就在 ALOHA 上击败了预训练过 6,000 条 / 8,000 小时双臂数据的 RDT-1B 与 $\pi_0$。这证明了大模型底层的跨模态表征潜力远未被传统自回归离散微调榨干。
