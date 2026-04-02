# MMaDA-VLA：基于原生离散扩散的统一多模态指令与生成 VLA

> **论文**：*MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Unified Multi-Modal Instruction and Generation*
>
> **作者**：Yang Liu, Pengxiang Ding, Tengyue Jiang, Xudong Wang, Minghui Lin, Wenxuan Song, Hongyin Zhang, Zifeng Zhuang, Han Zhao, Wei Zhao, Siteng Huang, Jinkui Shi, Donglin Wang
>
> **机构**：西湖大学、浙江大学、华东理工大学、华为 Celia Team、香港科技大学（广州）、OpenHelix Robotics
>
> **发布时间**：2026年3月
>
> **链接**：[arXiv](https://arxiv.org/abs/2603.25406)
>
> **分类标签**：`VLA` `离散扩散` `统一多模态` `目标观测生成` `混合注意力` `LIBERO 98.0%` `CALVIN 4.78`

---

## 一句话总结

MMaDA-VLA 基于原生预训练的离散扩散大模型 MMaDA-8B，将语言、图像和机器人动作统一到离散 token 空间，通过 masked token denoising 目标**并行生成**未来目标观测和动作块（action chunk），结合混合注意力机制（模态内双向 + 模态间因果），在 LIBERO 达 98.0%、CALVIN ABC→D 达 4.78 平均任务长度，全面刷新 SOTA。

---

## 一、问题与动机

### 1.1 现有 VLA 范式的三大瓶颈

| 范式 | 代表方法 | 问题 |
| --- | --- | --- |
| **层级式（Hierarchical）** | VLA-Adapter、OpenVLA-OFT | 在预训练 VLM 上外接策略头，引入额外架构复杂度和训练成本，模块间信息保真度下降 |
| **自回归离散化** | OpenVLA、RT-2 | 将动作离散为 token 逐个自回归生成，动作维度之间本无序关系却被强加顺序，导致时序不一致和长时域误差累积 |
| **世界模型辅助** | GEVRM、Seer、VPP | 需要额外的视觉生成模块或逆动力学模块来建模环境动态，多阶段训练/推理引入 mismatch |

共性问题：**缺乏在单一框架内同时完成多模态理解和生成（包括动作生成与未来视觉预测）的能力**。

### 1.2 近期离散扩散 VLA 的局限

Discrete Diffusion VLA 和 LLaDA-VLA 等方法用 mask token 预测替代自回归解码，但它们**从自回归模型微调而来**，训练与推理范式不一致（pretraining 用自回归、fine-tuning 改扩散），限制了性能上界。

### 1.3 MMaDA-VLA 的核心思路

MMaDA-VLA 的关键突破在于：直接使用**原生预训练的离散扩散模型**（MMaDA-8B）作为骨干，从根本上消除训练-推理不一致问题。同时，将目标观测图像和动作块的生成统一为**并行迭代去噪**过程——模型在每一步去噪中同时利用所有已确定的 token 来推断剩余未知 token，实现全局无序优化（global, order-free refinement），天然适配动作维度无固有顺序的特点。

---

## 二、预备知识

### 2.1 离散扩散语言模型

与自回归模型逐 token 从左到右生成不同，离散扩散模型定义一个**前向加噪过程**（通常用 mask token 替换）和一个**反向去噪过程**（恢复被 mask 的 token）。代表工作 LLaDA 在文本上展示了这种范式的可行性。

核心优势：
- **并行预测**：每步可同时预测多个 token，不受自回归的顺序约束
- **全局优化**：迭代去噪允许模型在多步中反复修正，实现全局一致性
- **双向建模**：每个 token 可利用双向上下文信息

### 2.2 MMaDA 基座模型

MMaDA-8B-Base 是一个**原生**预训练的多模态离散扩散模型（非从自回归模型转换），同时支持文本理解、图像理解和图像生成。它使用：
- 文本分词器来自 LLaDA
- 图像量化器来自 MAGVIT-v2（Show-o 采用的版本）

MMaDA-VLA 在此基座上引入机器人动作 token，扩展为视觉-语言-动作统一框架。

---

## 三、核心方法

### 3.1 问题建模

传统 VLA 建模为条件动作预测：$\hat{a}_t \sim \pi_\theta(a_t \mid o_t, l)$。

MMaDA-VLA 将其扩展为**并行生成目标观测和动作块**：

$$(\hat{o}_{t'};\ \hat{a}_{t:t'-1}) \sim \pi_\theta(o_{t'},\ a_{t:t'-1} \mid o_t, l)$$

其中 $t' = t + k$，$k$ 为 action chunk 大小。模型不仅预测"怎么做"（动作），还预测"做完后会看到什么"（目标观测），将世界模型的概念**内化**到 VLA 框架中。

### 3.2 统一离散 Token 化

所有模态被映射到同一离散 token 空间：

| 模态 | 分词器 | 说明 |
| --- | --- | --- |
| 文本 | LLaDA 文本分词器 | 语言指令编码 |
| 图像 | MAGVIT-v2 | 将 $256 \times 256$ 图像编码为离散 token |
| 动作 | 256-bin 均匀量化 | 7D 连续动作每维独立量化为 256 个 bin |

所有 token 共享一个大小为 $\mathcal{V}$ 的词表，使得**单一学习目标**（masked token prediction）可以跨模态统一训练。

### 3.3 多模态序列格式

输入序列按照指令-生成二分结构组织：

$$x = \underbrace{[\text{SOO}]\ \tilde{o}_t\ [\text{EOO}]\ [\text{SOL}]\ \tilde{\ell}\ [\text{EOL}]}_{\text{Instruction}} \quad \underbrace{[\text{SOO}]\ \tilde{o}_{t'}\ [\text{EOO}]\ [\text{SOA}]\ \tilde{a}_{t:t'-1}\ [\text{EOA}]}_{\text{Generation}}$$

其中 $[\text{SOX}]$、$[\text{EOX}]$ 是标记模态起止的特殊 token。训练时生成部分的 token 会被随机 mask；推理时生成部分全部替换为 $[\text{M}]$。

### 3.4 混合注意力机制

这是 MMaDA-VLA 的关键设计。模型采用**模态内双向全注意力 + 模态间因果注意力**的混合策略：

- **模态内**（intra-modal）：同一模态的 token 之间使用**双向全注意力**，实现全局信息交换。这对动作 token 尤为重要——7-DoF 动作向量的各维度本质上无序，双向注意力避免了自回归强加的虚假顺序依赖
- **模态间**（inter-modal）：不同模态之间使用**因果注意力**，信息从指令部分单向流向生成部分。这将目标图像生成和动作生成有效解耦，防止跨模态信息泄露

用大白话说：观测图像的 token 之间、动作 token 之间可以"互相看"，但动作 token 只能"看"前面的指令和目标图像，不能反向影响它们。

关键好处：并行去噪允许动作预测在每一步迭代中**持续利用目标图像生成的中间特征**，而不是等图像完全生成后才开始预测动作。这比"先生成图像再预测动作"的串行方案更高效且误差更小（消融实验证实了这一点）。

### 3.5 学习目标

预训练和微调都使用同一个**masked token prediction**目标。训练时按余弦调度采样 mask 比例，将生成部分的 token 随机替换为 $[\text{M}]$，然后优化：

$$\mathcal{L}(\theta) = -\mathbb{E}_{t, x, x^m} \left[ \frac{1}{N} \sum_{i=1}^{n} \mathbf{1}[x_i^m = [\text{M}]] \log \pi_\theta(x_i \mid x^m) \right]$$

其中 $N$ 是被 mask 的 token 数，$x^m$ 是 mask 后的序列。这个目标在所有模态上是统一的——无论是图像 token、文本 token 还是动作 token，都用同一个交叉熵损失。

### 3.6 推理：迭代去噪

推理时执行 $D$ 步迭代去噪（默认 $D=24$）：

> 1. 初始化：将生成部分所有 token 设为 $[\text{M}]$
> 2. 在每一步 $d$（从 $D$ 到 $0$）：
>    - 模型估计所有位置的 token 分布，通过贪心解码得到 $\hat{x}^{(0)}$
>    - 根据 mask 调度函数计算当前步应保留的 token 数 $\beta = \lceil \gamma(d/D) \cdot n' \rceil$
>    - **基于置信度重新 mask**：置信度低于第 $\beta$ 小值的 token 重新设为 $[\text{M}]$，其余保留
> 3. 最终步 ($d=0$) 输出完整序列，分别解码得到目标观测图像和动作

### 3.7 KV Cache 加速

为满足机器人实时控制的延迟要求，MMaDA-VLA 采用**无训练缓存框架**：

- **指令部分**：在整个去噪过程中固定不变，其 KV 可以**完整缓存**，跨步复用
- **生成部分**：仅选择性刷新——对每层，只更新当前值向量与缓存值向量**余弦相似度最低**的 $\lfloor \rho n' \rfloor$ 个 token（$\rho$ 为自适应更新比例）
- 缓存每 $\lambda=6$ 步全局刷新一次

---

## 四、实验结果

### 4.1 训练流程

**预训练**：在 28 个跨具身数据集上训练，共 6100 万步数据，DROID 占比最大（49.94%）。使用 8 节点 × 8 H800 GPU，约 30 小时完成。

**微调**：在下游基准上微调，LIBERO 微调 20-40 epochs（action chunk = 5），CALVIN 微调 2 epochs（action chunk = 10）。

### 4.2 LIBERO

| 方法 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| Discrete Diffusion VLA | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| VLA-Adapter | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| UniVLA | 95.4 | 98.8 | 93.5 | 94.0 | 95.5 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 93.0 | 96.3 |
| **MMaDA-VLA** | **98.8** | **99.8** | **98.0** | **95.2** | **98.0** |

MMaDA-VLA 在所有四个子集上均达到最高，整体 98.0% 超越连续动作方法 VLA-Adapter（97.3%）和世界模型方法 MM-ACT（96.3%）。值得注意的是，这是**离散动作方法首次全面超越连续动作方法**。

### 4.3 CALVIN ABC→D

| 方法 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. Len. |
| --- | --- | --- | --- | --- | --- | --- |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| π₀ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| LLaDA-VLA | 95.6 | 87.7 | 79.5 | 73.9 | 64.5 | 4.01 |
| VLA-Adapter | 99.1 | 94.6 | 88.8 | 82.8 | 76.5 | 4.42 |
| DreamVLA | 98.2 | 94.6 | 89.5 | 83.4 | 78.1 | 4.44 |
| UniVLA | 98.9 | 94.8 | 89.0 | 82.8 | 75.1 | 4.41 |
| **MMaDA-VLA** | **99.8** | **98.6** | **96.3** | **93.5** | **89.7** | **4.78** |

MMaDA-VLA 大幅刷新 CALVIN SOTA：
- 平均任务长度 4.78，超越此前最佳 DreamVLA 的 4.44（+0.34）
- 第 5 个子任务成功率从此前最佳 ~78% 跃升至 **89.7%**，提升超 11 个百分点
- 这说明 MMaDA-VLA 不仅单步执行更准确，长时域一致性也显著更强

### 4.4 真实世界

使用 AgileX Piper 6-DoF 机械臂 + 1-DoF 夹爪，涵盖四类任务：

| 任务 | MMaDA-VLA | GR00T N1.6 |
| --- | --- | --- |
| Pick-and-Place | 86.7% | 63.3% |
| Stacking | 93.3% | 70.0% |
| Storage（开抽屉→放物→关抽屉） | 93.3% | 66.7% |
| Organizing（整理 5 件餐具） | 83.3% | 56.7% |

所有任务均超 80%，全面超越 GR00T N1.6 约 20+ 个百分点。

### 4.5 消融实验（CALVIN，无预训练）

| 变体 | Avg. Len. | 相对完整模型变化 |
| --- | --- | --- |
| **MMaDA-VLA（w/o Pre-Training）** | **4.56** | — |
| w/o World-Model（不生成目标图像） | 4.08 | **-0.48** |
| w/o Parallel Denoising（先生成图像再生成动作） | 4.38 | -0.18 |
| w/ Causal Attention（全因果注意力） | 4.49 | -0.07 |
| w/ Bidirectional Attention（全双向注意力） | 4.52 | -0.04 |

关键发现：
- **目标观测生成最为关键**（-0.48）：移除世界模型建模导致最大性能下降，证实了预测未来视觉的重要性
- **并行去噪优于串行**（-0.18）：先完整生成图像再预测动作会丢失中间特征的利用机会，且确定性生成的目标图像可能引入累积误差
- **混合注意力优于纯因果或纯双向**：因果注意力限制了模态内信息交互；全双向注意力导致模态间信息泄露引入噪声

### 4.6 预训练效果

| 配置 | LIBERO Avg. SR | CALVIN Avg. Len. |
| --- | --- | --- |
| MMaDA-VLA | **98.0%** | **4.78** |
| w/o Pre-Training | 94.5% | 4.56 |

预训练在 LIBERO 上带来 +3.5%，在 CALVIN 上带来 +0.22 的一致提升。

---

## 五、局限性与未来方向

论文未显式列出局限性，但从分析中可以识别：

1. **图像生成保真度有限**：由于使用紧凑的 token 表示，生成的目标观测图像像素级精度较低（夹爪细节、小物体模糊）。不过论文认为高层语义一致性已足以辅助动作预测
2. **推理延迟**：24 步迭代去噪带来额外计算开销，尽管 KV Cache 有所缓解，但相比自回归单步前向传播仍更慢
3. **仅验证了单臂操作**：真实世界实验限于 6-DoF 单臂夹爪，未验证灵巧手或双臂场景
4. **Action chunk 大小固定**：LIBERO 用 5、CALVIN 用 10，不同任务可能需要不同的 chunk 大小

---

## 六、个人思考

### 6.1 原生扩散 vs. 自回归转扩散：训练一致性是关键

MMaDA-VLA 与 Discrete Diffusion VLA、LLaDA-VLA 的核心区别在于**骨干模型是否原生预训练为扩散模型**。后两者从自回归模型微调为扩散模型，存在 pretraining/fine-tuning 的范式不一致。MMaDA-VLA 直接使用 MMaDA-8B 这个原生扩散基座，训练和推理全程使用 masked token denoising，消除了这一 mismatch。LIBERO 上的 98.0% vs. Discrete Diffusion VLA 的 96.3%（+1.7%）证实了这种一致性的价值。

### 6.2 与 DreamVLA 的对比：两种世界模型内化路线

MMaDA-VLA 和 DreamVLA 都在 VLA 中引入了世界模型的概念，但实现路线截然不同：

| | MMaDA-VLA | DreamVLA |
| --- | --- | --- |
| **预测目标** | 完整的未来 RGB 图像（离散 token） | 三类结构化知识（动态区域/深度/语义） |
| **生成方式** | 并行离散扩散去噪 | 自回归 `<dream>` 查询 + DiT 动作头 |
| **骨干** | MMaDA-8B（原生扩散） | GPT-2 Medium（自回归） |
| **动作与视觉的关系** | 并行共同去噪，动作持续利用图像中间特征 | 串行：先 dream 再 act |
| **CALVIN Avg. Len.** | **4.78** | 4.44 |

DreamVLA 的优势在于世界知识有明确的物理含义（动态/深度/语义），且使用 block-wise 结构化注意力解耦不同类型的知识。MMaDA-VLA 的优势在于**并行去噪**允许动作和图像在生成过程中持续相互促进，且原生扩散基座保证了训练-推理一致性。

### 6.3 离散扩散首次超越连续动作方法

此前在 LIBERO 和 CALVIN 上，连续动作方法（VLA-Adapter、π₀、OpenVLA-OFT）一直优于离散动作方法。MMaDA-VLA 首次打破这一格局，说明**离散化 + 扩散去噪的范式在充分预训练的基座上可以弥补连续表示的精度优势**。关键因素可能是：
- 并行无序去噪避免了自回归离散化的误差累积
- 大规模跨具身预训练提供了强大的先验

### 6.4 混合注意力的设计哲学

MMaDA-VLA 的混合注意力（模态内双向 + 模态间因果）在消融中仅带来微小提升（+0.04~0.07），但设计思想值得注意：它在扩散模型的双向建模需求和 VLA 的模态间信息流方向之间找到了平衡。与 DreamVLA 的 block-wise attention 类似，这再次说明**在多模态生成任务中，精心设计的注意力模式比简单的全注意力或全因果更优**。

### 6.5 KV Cache 的必要性

24 步迭代去噪是离散扩散 VLA 面临的实际部署挑战。MMaDA-VLA 的 KV Cache 策略（指令缓存 + 生成部分选择性刷新）是一个实用的工程方案，但论文未报告具体的推理延迟数据，这在实际部署中可能是一个关键限制。

---

## 参考

- **MMaDA**（Yang et al., NeurIPS 2025）：MMaDA-VLA 的基座模型，原生预训练的多模态离散扩散大语言模型
- **LLaDA**（Nie et al., NeurIPS 2025）：离散扩散语言模型，提出 masked token denoising 的文本生成范式
- **Discrete Diffusion VLA**（Liang et al., 2025）：首个将离散扩散引入 VLA 的工作，但从自回归模型微调
- **LLaDA-VLA**（Wen et al., 2025）：基于 LLaDA 的 VLA，同样存在训练-推理不一致
- **DreamVLA**（Zhang et al., NeurIPS 2025）：预测三类世界知识辅助动作生成，CALVIN 4.44
- **UniVLA**（Bu et al., RSS 2025 / ICLR 2026）：统一视觉-语言-动作模型，自回归范式
- **VLA-Adapter**（Wang et al., AAAI 2026）：小参数量 VLA 适配器，连续动作方法在 LIBERO/CALVIN 上的强基线
- **MAGVIT-v2**（Yu et al., ICLR 2024）：图像离散量化器，为 MMaDA-VLA 提供视觉 token 化能力
