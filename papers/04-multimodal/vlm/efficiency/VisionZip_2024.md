# VisionZip：基于注意力主导 Token 选择与上下文 Token 合并的视觉 Token 压缩

> **论文**：*VisionZip: Longer is Better but Not Necessary in Vision Language Models*
>
> **作者**：Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia
>
> **机构**：CUHK、HKUST、HITSZ
>
> **发布时间**：2024年12月（**CVPR 2025**）
>
> 🔗 [arXiv](https://arxiv.org/abs/2412.04467) | [代码](https://github.com/dvlab-research/VisionZip)
>
> **分类标签**：`视觉 Token 压缩` `Dominant Token Selection` `Token Merging` `Text-Agnostic` `Training-Free`

---

## 一句话总结

提出 VisionZip——一种 text-agnostic 的视觉 token 压缩方法：在视觉编码器内部利用 CLS token（CLIP）或平均注意力（SigLIP）选出聚合了大量信息的 **dominant token**，再将剩余 token 基于 key 相似度合并为 **contextual token**，从而在 LLM 输入前大幅削减视觉 token 数量。在 LLaVA-1.5 上仅保留 64/576 token（↓88.9%）即达 94% 原始性能（大幅超越 FastV 18.4%、SparseVLM 8.2%），LLaVA-NeXT 上实现 **8× prefilling 加速**，甚至使 13B 模型推理速度超过 7B 同时性能更优。

---

## 一、问题与动机

### 1.1 视觉 Token 的计算瓶颈

VLM 的视觉 token 数量远超文本 token：LLaVA-1.5 为 576 个，LLaVA-NeXT 高达 2880 个，而文本 token 通常仅几十到百余个。这些冗长的视觉序列在 LLM 中引发巨大的计算和内存开销，序列长度 $n$ 的计算复杂度为：

$$\text{Total FLOPs} = T \times (4nd^2 + 2n^2d + 2ndm)$$

其中 $n = n_{\text{sys}} + n_{\text{img}} + n_{\text{question}}$，$n_{\text{img}}$ 往往是其他部分的 20 倍以上。因此，**减少 $n_{\text{img}}$ 是提升 VLM 效率的关键**。

### 1.2 视觉 Token 的冗余性观察

通过对 CLIP 和 SigLIP 视觉编码器的注意力分布进行可视化分析，发现一个关键现象：

- 注意力高度集中在极少数 token 上，大量 token 的注意力权重接近零
- 随着 encoder 层数加深，注意力从均匀分布逐渐收敛到少数 "代理 token"（proxy tokens），在第 23 层（LLM 所用的倒数第二层）达到峰值聚集
- 这种 "注意力汇聚" 现象类似于 LLM 中的 Attention Sink

根本原因在于 softmax 的梯度特性：

$$\frac{\partial \text{softmax}(z_i)}{\partial z_i} = \text{softmax}(z_i) \cdot (1 - \text{softmax}(z_i))$$

当 $z$ 较大时梯度呈指数上升，当 $z$ 较小时梯度几乎为零——这使得高注意力区域越来越突出，低注意力区域越来越被忽略，最终信息 "捷径式" 地聚集到少数 dominant token 中。

### 1.3 Text-Relevant 方法的根本缺陷

现有方法（FastV、SparseVLM）在 LLM forward 过程中基于文本-视觉 token 间的注意力来选择 token。但这种 text-relevant 策略存在 **特征错位**（feature misalignment）问题：

视觉编码器已经将信息 "预聚合" 到少数 dominant token 中，这些 token 往往**位于图像的边缘或背景区域**而非主体上。当 text-relevant 方法选择语义上与文本相关的 token（如图中的 "人" 或 "车"）时，实际选到的是**信息含量较低**的 token，因为真正的信息已被聚合到背景区域的 dominant token 中。

---

## 二、核心方法

VisionZip 在视觉编码器输出后、投入 LLM 之前进行 token 压缩，分为两步：

### 2.1 Dominant Token Selection（主导 Token 选择）

**目标**：从视觉 token 中选出聚合了最多信息的少数 dominant token。

**对于有 CLS Token 的编码器（如 CLIP）**：利用 CLS token 的注意力分数，选取被 CLS token 关注最多的 $K$ 个视觉 token：

> 1. 从视觉编码器的 SELECT_LAYER 层提取注意力矩阵 $\boldsymbol{S} \in \mathbb{R}^{B \times H \times S \times S}$
> 2. 计算 CLS token 对各 patch 的接收注意力：$\text{attn\_rec} = \sum_h \boldsymbol{S}[:, h, \text{cls}, \text{cls}+1:]$
> 3. 按 $\text{attn\_rec}$ 取 top-$K$，与 CLS token 拼接，得到 dominant token

**对于无 CLS Token 的编码器（如 SigLIP）**：计算每个 token 从所有其他 token 接收到的平均注意力：

$$\text{attn\_rec}_j = \frac{1}{H \cdot S} \sum_h \sum_i S_{h,i,j}$$

按此分数选取 top-$K$ 作为 dominant token。

### 2.2 Contextual Token Merging（上下文 Token 合并）

**目标**：避免丢失小而重要的细节信息。对 dominant token 之外的剩余 token，基于语义相似度合并为 contextual token。

> 1. 从剩余 token 中均匀采样 $M$ 个作为 **target token**，其余为 **merge token**
> 2. 利用自注意力中的 key 向量计算相似度：$\text{similarity} = \text{bmm}(\boldsymbol{K}_{\text{merge}}, \boldsymbol{K}_{\text{target}}^\top)$
> 3. 将每个 merge token 分配给最相似的 target：$\text{assign} = \text{argmax}(\text{similarity}, \text{dim}=2)$
> 4. 对同组 token 取均值合并，生成 contextual token

**Token 数量配置**（以 LLaVA-1.5 为例）：

| 保留总数 | Dominant Token | Contextual Token |
| --- | --- | --- |
| 192 | 162 | 30 |
| 128 | 108 | 20 |
| 64 | 54 | 10 |

用大白话说：dominant token 是图像中的 "信息枢纽"，聚合了绝大部分视觉信息；contextual token 则像 "补充材料"，通过合并语义相近的剩余 token，保留住那些可能被遗漏的细节。

### 2.3 Efficient Tuning（可选微调）

直接用压缩后的 token 输入 LLM 会造成轻微的模态空间错位（LLM 训练时接收的是完整 token）。VisionZip 提供一种极轻量的微调方案：

- **仅微调 projector 层**，其他组件冻结
- 使用 **1/10 的 LLaVA-1.5 数据集**
- 8 × A800 上仅需 **30 分钟**（也可在 3090 上完成）

消融实验表明，使用更匹配的数据集（如 LLaVA-NeXT 数据微调 LLaVA-NeXT 模型）增益不超过 0.5%，说明性能提升来自于**适应 token 数量变化**而非额外知识注入。

---

## 三、实验结果

### 3.1 LLaVA-1.5 图像理解

在 576 → 64 token（↓88.9%）极端压缩下，11 个基准测试平均性能（相对 vanilla 的百分比）：

| 方法 | 保留 192 (↓66.7%) | 保留 128 (↓77.8%) | 保留 64 (↓88.9%) |
| --- | --- | --- | --- |
| FastV (ECCV'24) | 88.2% | 83.5% | 75.6% |
| SparseVLM | 96.4% | 93.4% | 85.8% |
| **VisionZip** | **98.5%** | **97.6%** | **94.0%** |
| **VisionZip‡** | **99.1%** | **98.4%** | **95.2%** |

关键观察：
- 64 token 下 VisionZip 超越 FastV **18.4%**、SparseVLM **8.2%**
- VisionZip‡ 仅用 64 token（原始 1/9）即达 95.2% 性能
- 在 MMMU、MMVet 等基准上，减少 token 反而**提升**了性能（如 MMVet 从 31.1 提升到 32.6），说明冗余 token 可能作为噪声干扰模型判断

### 3.2 LLaVA-NeXT 高分辨率

2880 → 160 token（↓94.4%）的极端压缩下：

| 方法 | 保留 640 (↓77.8%) | 保留 320 (↓88.9%) | 保留 160 (↓94.4%) |
| --- | --- | --- | --- |
| SparseVLM | 96.1% | 93.3% | 86.4% |
| **VisionZip** | **97.6%** | **95.0%** | **92.0%** |
| **VisionZip‡** | **98.9%** | **97.9%** | **95.5%** |

VisionZip‡ 在仅保留约 5% token 时仍达 95.5%，超越 SparseVLM 9.1%。

### 3.3 视频理解

Video-LLaVA 上将 2048 视频 token 压缩到 136（每帧 256→17）：

| 方法 | TGIF | MSVD | MSRVTT | ActivityNet | Avg |
| --- | --- | --- | --- | --- | --- |
| Video-LLaVA | 47.1 | 69.8 | 56.7 | 43.1 | 100% |
| FastV | 23.1 | 38.0 | 19.3 | 30.6 | 52.1% |
| SparseVLM | 44.7 | 68.2 | 31.0 | 42.6 | 86.5% |
| **VisionZip** | **42.4** | **63.5** | **52.1** | **43.0** | **93.2%** |

VisionZip 超越 SparseVLM 6.7%，在 MSRVTT 上优势高达 **37.2%**（52.1 vs. 31.0）。更重要的是，VisionZip 使模型可以在相同内存下编码 **10 倍帧数**。

### 3.4 效率分析

LLaVA-NeXT 7B 上（POPE 数据集，单卡 A800）：

| 方法 | Token 数 | 总时间 | Prefilling 时间 | 总加速 | Prefill 加速 |
| --- | --- | --- | --- | --- | --- |
| Vanilla | 2880 | 2293s | 218ms | - | - |
| FastV | 160 | 1792s | 119ms | 1.3× | 1.8× |
| SparseVLM | 160 | 1895s | 128ms | 1.2× | 1.7× |
| **VisionZip** | **160** | **756s** | **27.8ms** | **3.0×** | **7.8×** |

VisionZip 在 prefilling 上实现 **7.8× 加速**——因为它在 token 进入 LLM 之前就完成了压缩，而 FastV/SparseVLM 需要先将所有 token 送入 LLM 浅层再逐步裁剪。

### 3.5 13B 模型超越 7B

通过 VisionZip 压缩视觉 token 后，LLaVA-NeXT 13B 在 TextVQA 上：

| 配置 | 推理时间 | TextVQA |
| --- | --- | --- |
| 7B Vanilla | 1714s | 61.3 |
| 13B Vanilla | 2516s | 64.3 |
| **13B + VisionZip** | **1246s** | **62.2** |

13B + VisionZip 比 7B Vanilla **更快**（1246s vs. 1714s）且**更准**（62.2 vs. 61.3）。结合 4-bit 量化后 13B 模型仅需 10GB 显存。

### 3.6 多轮对话优势

Text-relevant 方法（如 SparseVLM）在第一轮对话中基于当前问题选择视觉 token 并存入 KV Cache，但这些 token 与后续问题可能无关，导致多轮对话性能急剧下降。VisionZip 以 text-agnostic 方式选择信息量最大的 token，在多轮对话中天然具有优势。

---

## 四、局限性与未来方向

### 4.1 依赖特定层的注意力模式

VisionZip 的 dominant token 选择依赖于视觉编码器倒数第二层的注意力聚合现象。如果未来的视觉编码器设计减少了这种 "注意力汇聚"（如通过 register token 等机制），方法的有效性可能减弱。

### 4.2 Contextual Token 合并策略的局限

均匀采样 target token + key 相似度合并是一种启发式策略，缺乏理论上的最优性保证。当图像中存在多个空间分散的小目标时，均匀采样可能无法覆盖所有关键区域。

### 4.3 视觉编码器冗余的根本问题

论文最后指出一个更深层的方向：与其在下游压缩冗余 token，不如从源头设计**低冗余的视觉编码器**。当前 CLIP/SigLIP 的注意力汇聚现象本身就是编码效率低的体现——大量参数和计算被浪费在生成 "无信息" 的 token 上。

---

## 五、个人思考

### 5.1 Text-Agnostic vs. Text-Relevant 的根本分歧

VisionZip 与 FastV/SparseVLM 的核心分歧在于 **"在哪里做 token 选择"**：

| 维度 | Text-Relevant (FastV/SparseVLM) | Text-Agnostic (VisionZip) |
| --- | --- | --- |
| 选择位置 | LLM 内部（需 forward 浅层） | 视觉编码器输出后、LLM 之前 |
| 选择依据 | 文本-视觉跨模态注意力 | 视觉编码器内部注意力 |
| 计算开销 | 仍需处理全量 token 的浅层计算 | 零 LLM 计算开销 |
| 多轮对话 | KV Cache 锁定第一轮选择 | 天然兼容 |
| 理论假设 | 文本相关的 token = 重要的 token | 高注意力的 token = 信息聚合的 token |

VisionZip 的成功揭示了一个反直觉的事实：**语义相关性 ≠ 信息量**。视觉编码器的预聚合机制导致信息与语义在空间上发生了错位。

### 5.2 与 DART 的比较

VisionZip 和 DART 都是 text-agnostic 的视觉 token 压缩方法，但切入角度不同：

| 维度 | VisionZip | DART |
| --- | --- | --- |
| 核心思路 | 选择 "信息最丰富" 的 token | 移除 "最冗余重复" 的 token |
| Token 选择 | 注意力 top-K（importance-based） | 余弦相似度阈值（duplication-based） |
| 剩余 token | 合并为 contextual token（信息不丢失） | 直接丢弃（依赖理论保证） |
| FlashAttention | 需要注意力分数（不兼容） | 完全兼容（可用 K/V-norm） |
| 额外微调 | 可选 30min projector tuning | 无 |

有趣的是，DART 的研究直接挑战了 VisionZip 的 importance-based dominant token 选择思路——DART 证明了随机选 pivot 仅比最优策略低 1.2%，暗示 "哪些 token 注意力高" 可能并非最优选择标准。两篇论文的共同点是都证实了 **视觉 token 中存在大量结构化冗余**，但对冗余的处理策略截然不同（选最好的 vs. 去最差的）。

### 5.3 与 Elastic Cache 的关系

VisionZip 压缩的是 prefill 阶段的视觉 token 输入，Elastic Cache 压缩的是 decode 阶段的 KV Cache。两者在流水线的不同阶段工作，可以串联使用：VisionZip 先在 encoder 输出端压缩 token 数量，Elastic Cache 再在 LLM 生成时压缩 KV Cache 大小，实现双重加速。

### 5.4 对 Token Pruning Survey 的回应

Token Pruning Survey（同一作者 Zichen Wen）指出的 5 个核心问题中，VisionZip 的设计恰好规避了其中几个：

- **位置偏差**：VisionZip 用 CLS 注意力而非层间注意力，减轻了位置偏差问题
- **FLOPs ≠ 延迟**：VisionZip 在 LLM 之前完成压缩，避免了 text-relevant 方法 "FLOPs 低但实际不快" 的问题
- **训练感知压缩**：VisionZip 的 efficient tuning 模式本质上就是一种轻量的训练感知压缩

---

## 参考

- **FastV**（Chen et al., ECCV 2024）：基于 LLM 层间注意力的视觉 token 剪枝——VisionZip 的主要基线，被大幅超越
- **SparseVLM**（Zhang et al., 2024）：利用文本-视觉跨模态注意力稀疏化视觉 token——另一基线，特征错位导致信息选择不佳
- **ToMe**（Bolya et al., ICLR 2023）：Token Merging，基于相似度合并 ViT token——VisionZip 的 contextual token merging 受其启发
- **DART**（Wen et al., EMNLP 2025）：基于 token 重复度的视觉 token 压缩——同为 text-agnostic 方法，但从 "去冗余" 角度切入
- **Elastic Cache**（Liu et al., ECCV 2024）：KV Cache 压缩——与 VisionZip 在流水线不同阶段工作，互补使用
- **Token Pruning Survey**（Wen et al., ACL 2025 Findings）：系统分析 token pruning 五大问题——VisionZip 的设计规避了其中多个问题
