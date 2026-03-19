# Token Pruning in MLLMs：我们真的在解决正确的问题吗？

> **论文**：*Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?*
>
> **作者**：Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, Linfeng Zhang
>
> **机构**：上海交通大学、上海人工智能实验室、中山大学
>
> **发布时间**：2025年2月（arXiv），**ACL 2025 Findings** 录用
>
> **分类标签**：`MLLM` `Token Pruning` `Visual Token Compression` `Efficiency` `Analysis`

---

## 一句话总结

系统性分析 MLLM 视觉 token 剪枝的五个核心问题——**位置偏差**导致精心设计的方法不如随机剪枝、**语言引导**仅在文本强关联任务有效、**重要性 vs. 冗余性**需按任务类型自适应平衡、**FLOPs 不等于真实延迟**、**训练感知压缩**远优于推理阶段剪枝——为未来 token 剪枝方法设计提供系统性指导。

---

## 一、问题与动机

### 1.1 MLLM 推理瓶颈

多模态大语言模型面临严重的推理开销问题。以视觉-语言模型为例：
- LLaVA-1.5 单张图像产生 **576 个** visual token
- LLaVA-NeXT 双倍分辨率下产生 **2880 个** visual token，远超文本 prompt 长度
- 视觉 token 数量多、空间冗余度高、信息密度低

**Token 剪枝**因无需训练即可应用于现有模型而备受关注，号称可剪枝 70%+ token 且精度损失可接受。

### 1.2 一个令人震惊的发现

论文发现一个反直觉的现象：

> **在多数 benchmark 上，随机 token 选择和简单平均池化竟然优于 FastV、SparseVLM 等精心设计的 token 剪枝方法。**

| 方法 | GQA | MMB | MMB-CN | MME | POPE | SQA | VQA$^{\text{Text}}$ | VizWiz | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Vanilla（576 tokens）** | 61.9 | 64.7 | 58.1 | 1862 | 85.9 | 69.5 | 58.2 | 50.0 | 100% |
| **保留 144 tokens（↓ 75%）** | | | | | | | | | |
| Random | 59.0 | **62.2** | 54.1 | 1736 | 79.4 | 67.8 | 51.7 | **51.9** | 95.0% |
| Pooling | 59.1 | **62.5** | **55.2** | **1763** | **81.4** | **69.1** | 53.4 | **51.9** | **96.4%** |
| Vanilla FastV | 56.5 | 59.3 | 42.1 | 1689 | 71.8 | 65.3 | **53.6** | 51.3 | 89.8% |
| SparseVLM | 55.1 | 59.5 | 51.0 | 1711 | 77.6 | **69.3** | **54.9** | 51.4 | 93.5% |

**Random 和 Pooling 在近 2/3 的 benchmark 上超过了精心设计的方法。** 这说明现有方法对"重要 token"的理解可能存在根本性偏差。

### 1.3 五个被忽视的核心问题

论文围绕以下五个问题展开系统研究：

1. 为什么很多方法连随机选择都不如？
2. 基于注意力的评分机制是否足以可靠识别冗余 token？
3. 语言信息在 token 剪枝中是否真正有用？
4. token 重要性和重复性之间如何权衡？
5. 当前评估协议是否全面且无偏？

---

## 二、实验设置

### 2.1 模型

- **LLaVA-1.5-7B**：CLIP + LLaMA，MLP 连接器
- **LLaVA-Next-7B**：动态分辨率 + 层次化特征集成
- **Qwen2-VL 系列**（7B/72B）：训练阶段内置 token 合并（4 patch → 1 token）

### 2.2 数据集

| 类型 | 数据集 |
| --- | --- |
| 视觉理解 | GQA、MMBench、MME、POPE、ScienceQA、VQA V2、TextVQA |
| 物体定位 | RefCOCO/RefCOCO+/RefCOCOg |
| 物体检索 | Visual Haystack |

### 2.3 剪枝方法

- **FastV**：第 2 层后按最后一个 token 的注意力分数选择 visual token
- **SparseVLM**：文本引导的跨模态注意力 token 选择（无训练）
- **MustDrop**：视觉编码 + prefill + 解码全生命周期多阶段剪枝
- 基线：**Random**（随机选择）、**Pooling**（平均池化）

---

## 三、核心发现

### 3.1 发现一：空间均匀性 > 位置偏差

**现象。** FastV 利用最后一个 text token 对 visual token 的注意力分数来评估重要性。论文在 POPE 数据集 8,910 个样本上统计 FastV 保留的 visual token 分布，发现：

> 位于 visual token 序列末尾（对应图像底部）的 token 获得显著更高的注意力分数，被保留的频率远高于其他位置。

这意味着注意力评分天然存在**位置偏差 (position bias)**——靠后位置的 token 不一定更重要，但注意力分数系统性偏高。

**验证。** 论文提出 **Window FastV**：引入滑动窗口机制，在每个局部窗口内按注意力分数选择固定数量的 token，从而保证保留 token 的**空间均匀分布**。

> **算法：Window FastV**
>
> 1. 前 $K-1$ 层正常计算，记录全局注意力分数 $\alpha = \text{mean}(A)[s:e]$
> 2. 第 $K$ 层将图像区域 reshape 为 2D 网格 $\Gamma \in \mathbb{R}^{h \times w}$
> 3. 将网格划分为局部窗口 $\{W_{ij}\}$
> 4. 在每个窗口内计算局部注意力分数并选 top-$k$
> 5. 聚合所有窗口索引，构建保留序列

**结果。** 75% 剪枝率下 Window FastV 比 Vanilla FastV 平均性能衰减少 **3.4%**；88.9% 剪枝率下差距扩大到 **9%**。

**空间定位验证。** 在 RefCOCO 系列 grounding 任务上，所有方法性能严重下降（↓ 76%–95%），但空间均匀方法（Window FastV、Random、Pooling）显著优于空间非均匀方法（Vanilla FastV、SparseVLM）：

| 方法 | RefCOCO Avg. |
| --- | --- |
| Vanilla（无剪枝） | 100% |
| SparseVLM | 4.8%（↓ 95.2%）|
| Vanilla FastV | 18.8%（↓ 81.2%）|
| Random | 23.2%（↓ 76.8%）|
| Window FastV | 20.2%（↓ 79.8%）|
| Pooling | 22.7%（↓ 77.3%）|

> **Summary 1**：保留 token 分布的位置偏差是现有方法不如 Random/Pooling 的关键原因。设计 token 剪枝策略时应确保保留 token 的**空间均匀性**。

### 3.2 发现二：语言引导何时有效？

**假设。** Token 剪枝方法分两类：文本引导（FastV、SparseVLM、MustDrop）和纯视觉（FasterVLM）。两类方法在常见 benchmark 上表现相当——但这是否因为常见 benchmark 缺少文本信息至关重要的任务？

**实验。** 选择 **Visual Haystack** 任务——一个强文本依赖场景：模型需从多张干扰图像中根据锚词选择正确图像，然后判断目标物体是否存在。

论文将 FastV 改为不使用文本信息的 **FastV$_{\text{VIS}}$**（用最后一个 visual token 替代 text token 计算注意力），对比结果：

| 方法 | Oracle | 2 imgs | 3 imgs | 5 imgs | 10 imgs |
| --- | --- | --- | --- | --- | --- |
| LLaVA-1.5-7B（无剪枝） | 86.5 | 70.0 | 66.2 | 58.3 | 53.5 |
| SparseVLM | **81.3** | **66.1** | **66.5** | **58.2** | **54.0** |
| FastV | 76.3 | 61.2 | 58.3 | 53.4 | 52.1 |
| FastV$_{\text{VIS}}$ | 71.9 | 61.6 | 55.8 | 52.7 | 52.8 |
| Random | 75.2 | 62.1 | 55.6 | 51.3 | 50.8 |

- FastV$_{\text{VIS}}$ 显著下降，说明文本引导在强文本依赖任务中**至关重要**
- SparseVLM 在 77.8% 压缩率下几乎保持原模型精度
- 但在常见 VQA benchmark 上，纯视觉方法反而更优

> **Summary 2**：文本引导仅在任务强依赖语言信息时有效。剪枝方法应根据任务需求自适应调整是否利用语言信息。

### 3.3 发现三：重要性 vs. 冗余性的 $\alpha$ 困境

这是论文最具理论深度的部分。Token 剪枝面临一个根本性张力：**应优先移除冗余 token 以保持结构完整性，还是移除不重要 token 以保持预测能力？**

#### 信息论视角

**冗余准则（任务无关）**：最大化原始 token $X$ 与保留 token $X'$ 之间的互信息：

$$\max_{\mathcal{P}} I(X; X') = H(X) - H(X|X')$$

这等价于信息瓶颈原理的压缩阶段，保持结构完整性。

**重要性准则（任务导向）**：保留对预测输出 $Y$ 关键的 token：

$$I(X'; Y) \geq I(X; Y) - \epsilon$$

由链式法则展开：

$$\underbrace{I(X; Y)}_{\text{原始}} = \underbrace{I(X'; Y)}_{\text{保留}} + \underbrace{I(X \setminus X'; Y | X')}_{\text{丢弃}}$$

两者的权衡由信息平面上的 rate-distortion 函数控制：

$$\mathcal{R}(\beta) = \max_{X'} \left[ I(X'; Y) - \beta^{-1} I(X; X') \right]$$

#### 自适应评分机制

论文提出可调参数 $\alpha$ 的统一评分：

$$\text{Score}(x_i) = \alpha \cdot \underbrace{I(x_i; Y | x_{\setminus i})}_{\text{预测关键性}} + (1 - \alpha) \cdot \underbrace{[1 - I(x_i; X_{\setminus i})]}_{\text{模式独特性}}$$

实践中，重要性由 FastV 注意力分数衡量，冗余性由 visual token 与 last token 的余弦相似度（取反）衡量，两者均经 min-max 归一化后加权。

#### 实验结果

| Benchmark | Vanilla | $\alpha$=0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | **0.8** | **0.9** | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **MME** | 1862 | 1707 | **1714** | 1711 | 1706 | 1711 | 1699 | 1680 | 1688 | 1689 |
| **POPE** | 85.9 | **82.8** | 82.6 | 82.4 | 81.9 | 81.6 | 79.7 | 77.9 | 75.6 | 71.8 |
| **SQA** | 69.5 | 64.8 | 65.2 | 65.2 | 65.1 | 65.3 | 65.2 | 65.5 | **65.7** | 65.3 |
| **VQA$^{\text{Text}}$** | 58.2 | 53.6 | 53.8 | **54.8** | 54.0 | 54.3 | 54.5 | 54.4 | 54.2 | 53.6 |

两个关键发现：

- **感知主导任务**（MME、POPE）：$\alpha = 0.0 \sim 0.1$ 最优，偏向**冗余优先**剪枝，保持结构完整性（$\uparrow I(X; X')$）
- **知识密集任务**（SQA、VQA$^{\text{Text}}$）：$\alpha = 0.8 \sim 0.9$ 最优，偏向**重要性优先**剪枝，增强语义连贯性（$\uparrow I(X'; Y)$）

> **Summary 3**：应按任务类型调整剪枝策略。感知任务用冗余优先保持结构保真度，知识推理任务用重要性优先保持预测能力。

### 3.4 发现四：FLOPs ≠ 真实加速

**现象。** 相同剪枝设定下（均保留 320 tokens），三种方法的 FLOPs 相近但实际延迟差异巨大：

| 方法 | Tokens ↓ | 延迟 | FLOPs ↓ | KV Cache ↓ | POPE |
| --- | --- | --- | --- | --- | --- |
| Vanilla LLaVA-Next-7B | 2880 | 36:16 | 100% | 1512.1 MB | 86.5 |
| + FastV | 320 | **18:17** | 12.8% | 168.0 MB | 78.3 |
| + SparseVLM | 320 | 23:11 | 15.6% | 168.0 MB | 82.3 |
| + MustDrop | 320 | 23:40 | 11.5% | 168.0 MB | 82.1 |

SparseVLM 的 FLOPs 仅比 FastV 高 **2.8%**，但实际延迟高出 **26.8%**。

**原因分析。**

1. **Flash Attention 不兼容**：三种方法都需要完整 attention map 来选择 token，无法使用 Flash Attention
2. **剪枝层数差异**：FastV 仅在 1 层剪枝，SparseVLM 和 MustDrop 在 4 层剪枝——更多层被迫使用 $O(N^2)$ 内存的传统注意力
3. **运行时开销**：逐层剪枝的复杂 token 选择操作可能抵消序列缩短带来的加速
4. **深层剪枝收益递减**：在网络深层剪枝 token 对整体加速贡献有限

> **Summary 4**：FLOPs 不是评估加速效果的可靠指标，应以实际延迟为准。Token 剪枝应在浅层用简单操作完成，并确保与 Flash Attention 兼容。

### 3.5 发现五：训练感知压缩的被忽视优势

新一代 MLLM（如 Qwen2-VL）在训练阶段就内置了 token 合并策略（4 个相邻 patch 合并为 1 个 visual token）。这些模型产生的 visual token **信息密度更高**，同样数量的 token 剪枝会导致更大的信息损失。

论文定义了训练感知的 Token Reduction Rate（TRR）：

$$\text{TRR}(\text{FastV}^{\dagger}) \triangleq \underbrace{\text{TACR}}_{\text{训练感知}} \times \underbrace{\text{TFRR}}_{\text{推理阶段}}$$

其中 Qwen2-VL 的 TACR = 4。FastV$^{\dagger}$ 表示考虑训练阶段压缩的 FastV。

**实验结果。** Qwen2-VL-7B 上：

| 方法 | GQA | MMB | MME | POPE | SQA | VQA$^{\text{Text}}$ | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 62.2 | 80.5 | 2317 | 86.1 | 84.7 | 82.1 | 100% |
| FastV（↓ 66.7%） | 58.0 | 76.1 | 2130 | 82.1 | 80.0 | 77.3 | 94.0% |
| FastV$^{\dagger}$（↓ 66.7%） | 61.9 | 80.9 | 2296 | 86.2 | 84.6 | 81.7 | **99.8%** |
| FastV（↓ 88.9%） | 51.9 | 70.1 | 1962 | 76.1 | 75.8 | 60.3 | 84.0% |
| FastV$^{\dagger}$（↓ 88.9%） | 61.9 | 81.1 | 2289 | 86.2 | 84.4 | 81.3 | **99.6%** |

考虑训练感知压缩后，即使 88.9% 的名义剪枝率下性能仍近乎无损（99.6%）！这表明 Qwen2-VL 的训练阶段 PatchMerger 已经有效压缩了大部分冗余信息。

> **Summary 5**：训练感知 token 压缩技术值得更多研究关注——它提供远优于推理阶段剪枝的性能保障。

---

## 四、局限性

1. **模型覆盖有限**：实验主要在 LLaVA 和 Qwen2-VL 上进行，未扩展到更多架构（如 InternVL、MiniCPM-V 等）
2. **缺少不同模型规模的系统对比**：结论是否在更大或更小的模型上仍然成立需要验证
3. **未探讨 token 剪枝 vs. token 合并**：两者在不同场景下的优劣尚未系统比较
4. **OCR 场景未涉及**：富文本 OCR 图像上的 token 剪枝效果未被评估

---

## 五、个人思考

### 5.1 与 VLA-Pruner 的关联

VLA-Pruner 提出的双层 token 剪枝策略（语义级 prefill + 动作级 decode 注意力时序平滑）实际上已经隐含了本文的多个洞察：
- VLA-Pruner 的 **mRMR 选择策略**正是在重要性和冗余性之间取平衡——与本文 $\alpha$ 困境的分析完全吻合
- VLA-Pruner 发现 50% 剪枝率**反超原模型**，这可能恰好是因为去除了冗余 token 带来的噪声

### 5.2 "Random 不如"才是真问题

本文最有价值的贡献不是提出新方法，而是**指出问题**：如果一个精心设计的方法连随机选择都不如，说明我们对"什么 token 重要"的理解可能从根本上就是错的。位置偏差的分析非常有说服力——注意力分数并不等于视觉重要性。

### 5.3 训练感知压缩的启示

FastV$^{\dagger}$ 在 Qwen2-VL 上 88.9% 剪枝率仍保持 99.6% 性能，这个结果极其震撼。它暗示：**与其在推理阶段费尽心思设计剪枝策略，不如在训练阶段就学会压缩**。这与近期 Qwen2-VL、MiniCPM-V 等模型的设计趋势一致——训练时内置高效的视觉 token 压缩模块。

### 5.4 信息论框架的价值

$\alpha$ 困境的信息论分析虽然在实际使用中不直接可操作（需要知道任务类型才能选 $\alpha$），但提供了一个**理解 token 剪枝本质**的清晰框架：感知任务需要空间完整性（保留独特 token），推理任务需要语义关键性（保留重要 token）。这对未来设计自适应剪枝方法很有指导意义。

### 5.5 Flash Attention 兼容性

FLOPs vs. 延迟的分析揭示了一个实用但常被忽视的问题：如果剪枝方法需要完整 attention map，就无法利用 Flash Attention，反而可能导致**负加速**。这对工业部署尤为重要——方法论文中报告的理论加速比在实际硬件上可能完全不成立。

---

## 六、参考

- **FastV** (Chen et al., 2024)：An Image is Worth 1/2 Tokens After Layer 2
- **SparseVLM** (Zhang et al., 2024)：Visual Token Sparsification for Efficient Vision-Language Model Inference
- **MustDrop** (Liu et al., 2024)：Multi-Stage Vision Token Dropping
- **ToMe** (Bolya et al., 2023)：Token Merging: Your ViT but Faster
- **FasterVLM** (Zhang et al., 2024)：[CLS] Attention is All You Need for Training-Free Visual Token Pruning
- **VLA-Pruner** (项目中已有)：双层 Token 剪枝策略用于 VLA 高效推理
