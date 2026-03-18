# MemVR：记忆空间视觉回溯缓解多模态大模型幻觉

> **论文**：*Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models*
>
> **作者**：Xin Zou*, Yizhou Wang*（共同一作）, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang, Jia Liu, Chang Tang, Xuming Hu
>
> **机构**：香港科技大学（广州）、蚂蚁集团、香港科技大学、悉尼科技大学、华中科技大学
>
> **发布时间**：2025年5月（**ICML 2025**）
>
> 🔗 [arXiv](https://arxiv.org/abs/2410.03577) | [代码](https://github.com/1zhou-Wang/MemVR)
>
> **分类标签**：`视觉幻觉` `FFN Key-Value Memory` `视觉回溯` `不确定性触发` `Training-Free` `Plug-and-Play`

---

## 一句话总结

受"看过的图片记忆模糊时会再看一遍"这一认知直觉启发，MemVR 在 MLLM 推理时**将视觉 token 作为补充证据通过 FFN 的 key-value memory 机制重新注入**中间触发层，当模型不确定性超过阈值时动态触发"look-twice"，无需训练即插即用，在 POPE 上提升 +7.0%、CHAIR$_I$ 改善 15.6%，同时推理延迟仅为贪心解码的 1.04×。

---

## 一、问题与动机

### 1.1 MLLM 中的视觉"失忆"

多模态大语言模型（MLLM）在推理过程中存在一个根本性问题：**文本解码器对视觉模态的"遗忘"**。具体表现为：

- 图像的信息密度远高于文本，LLM 天生更擅长理解文本而非视觉
- 自回归解码过程中，模型越来越依赖文本上下文 $y_{<t}$ 和查询 $x$，对视觉输入 $v$ 的依赖逐渐降低
- 从浅层到深层，注意力逐渐偏向文本 token，视觉 token 在深层几乎不影响输出

这种模态失衡导致了类似"失忆"的现象——模型在生成过程中逐渐"忘记"了图像内容。

### 1.2 现有方法的局限

| 方法类别 | 代表方法 | 局限 |
| --- | --- | --- |
| 对比解码（CD） | VCD、ICD | 需要多轮推理获取对比 logits，延迟翻倍；对比分布可能引入噪声 |
| 注意力干预 | OPERA、EAH、CCA | 推理延迟极高（3.66×）；内存开销大 |
| 微调/RAG | RLHF-V、LURE | 需要额外训练数据和计算资源 |

**关键问题**：CD 方法通过扰动输入（加噪声、修改文本等）来放大语言先验进行对比，但这种扰动：(1) 需要多轮推理，延迟翻倍；(2) 对比分布与视觉/指令的细微差异无关，可能引入噪声而非放大幻觉。

### 1.3 核心洞察

论文通过三组实验验证了关键假设：

**实验 1：模态失衡验证**（Figure 4 左）
- 等比放大图像特征导致的性能下降**大于**等比放大文本特征
- 等比缩小文本特征导致的性能下降**远大于**等比缩小视觉特征
- 结论：LLM 更依赖文本、更难理解视觉模态

**实验 2：刷新视觉记忆的效果**（Figure 4 右）
- 分别对文本/图像/文本+图像执行 look-twice 策略
- **仅补充图像信息**时效果最佳
- 结论：刷新视觉记忆可以有效缓解幻觉

**实验 3：幻觉的不确定性模式**（Figure 5）
- 幻觉 token（如"pomegranate"）在中间层和深层呈现高不确定性
- 简单 token（如功能词"image""with"）从中间层开始不确定性就很低
- 结论：不确定性可以作为触发视觉回溯的信号

---

## 二、预备知识

### 2.1 FFN 即 Key-Value Memory

论文的核心 insight 建立在 Geva et al. (2021) 的发现之上：**Transformer 的 FFN 层可以被理解为 key-value 记忆存储**。

标准 FFN 的形式为：

$$\text{FFN}(\boldsymbol{x}) = \phi(\boldsymbol{x}\boldsymbol{W}_1)\boldsymbol{W}_2^\top$$

其中 $\boldsymbol{W}_1, \boldsymbol{W}_2 \in \mathbb{R}^{d \times D}$，$D = 4d$。将权重矩阵按列分解：

$$\boldsymbol{W}_1 = (\boldsymbol{k}_1, \boldsymbol{k}_2, \ldots, \boldsymbol{k}_D), \quad \boldsymbol{W}_2 = (\boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_D)$$

则 FFN 可以重写为：

$$\text{FFN}(\boldsymbol{x}) = \sum_i \phi(\langle \boldsymbol{x}, \boldsymbol{k}_i \rangle) \cdot \boldsymbol{v}_i$$

直觉上，FFN 用输入 $\boldsymbol{x}$ 作为 query，与 key $\boldsymbol{k}_i$ 计算相似度，然后按相似度加权聚合对应的 value $\boldsymbol{v}_i$。这就是一个**无参数的注意力/检索过程**。

### 2.2 不确定性量化

参考 DoLa (Chuang et al., 2023)，在每一层 $l$ 通过词表头 $\varsigma(\cdot)$ 计算下一个 token 的概率分布，然后用归一化熵衡量不确定性：

$$u^{(l)} = \frac{\sum -p_i^{(l)} \log p_i^{(l)}}{\log N}$$

其中 $\{p_i\}_{i=1}^N$ 是词表上的概率分布，$N$ 为词表大小。$u \in [0, 1]$，越高表示越不确定。

---

## 三、核心方法

### 3.1 Visual Retracing（VR）——视觉回溯

MemVR 的核心操作是将视觉 token 重新注入 FFN 层，利用 FFN 的 key-value memory 机制刷新视觉信息。

给定隐状态 $\boldsymbol{x} \in \mathbb{R}^d$ 和维度对齐的视觉 token $\boldsymbol{z}_v = (\boldsymbol{z}_{v,1}, \ldots, \boldsymbol{z}_{v,N_v}) \in \mathbb{R}^{d \times N_v}$，在第 $l$ 层的 FFN 中执行视觉回溯：

$$\text{FFN}^{(l)}(\boldsymbol{x} \propto \boldsymbol{z}_v) = \alpha \Delta + (1 - \alpha) \text{FFN}^{(l)}(\boldsymbol{x})$$

其中 $\alpha \in [0, 1]$ 是注入比例，$\Delta$ 是视觉回溯操作：

$$\Delta(\boldsymbol{z}_v \mid \boldsymbol{x}) = \sum_{i=1}^{N_v} \phi(\langle \boldsymbol{x}, \boldsymbol{z}_{v,i} \rangle) \cdot \boldsymbol{z}_{v,i}$$

**直觉理解**：从 FFN key-value memory 的视角看，VR 将隐状态 $\boldsymbol{x}$ 作为 query，将 $\langle \boldsymbol{z}_{v,i} : \boldsymbol{z}_{v,i} \rangle$ 作为新的 key-value 条目（视觉证据），补充 hidden states 中的视觉信息。本质上是在 FFN 的记忆空间中**添加了视觉相关的新记忆条目**。

**计算开销极低**：FFN 中原有的 key-value memory 大小为 $D$（如 LLaMA-7B 中 $D = 11008$），而视觉 token 数量 $N_v$（如 ViT-L/14 中 $N_v = 256$）远小于 $D$，因此 VR 操作的计算量可以忽略不计。

### 3.2 动态触发策略（MemVR-dynamic）

VR 应该在什么时候触发、在哪一层触发？MemVR 利用不确定性动态决定：

> **算法：Dynamic Triggered MemVR**
>
> 1. 在每个解码步 $t$，设 $\text{trigger} = \text{TRUE}$
> 2. 对 $l = 1$ 到 $L-1$：
>    - 计算第 $l$ 层的不确定性 $u^{(l)} = \sum -p_\theta^{(l)} \log p_\theta^{(l)} / \log N$
>    - 若 $\text{trigger} = \text{TRUE}$ 且 $u^{(l)} > \gamma$：
>      - 执行视觉回溯 $\Delta(\boldsymbol{z}_v \mid \boldsymbol{h}_t^{(l+1)})$
>      - 选择 $\text{FFN}^{(l+1)}(\boldsymbol{h}_t^{(l+1)} \propto \boldsymbol{z}_v)$
>      - 设 $\text{trigger} = \text{FALSE}$（仅 look-twice 一次）
> 3. MLLM 解码，得到当前 token $\hat{y}_t$

关键设计：
- **仅触发一次**：每个 token 最多执行一次 VR，避免过度干扰
- **从浅到深扫描**：选择第一个不确定性超过阈值 $\gamma$ 的层执行 VR
- **不触发也无开销**：如果所有层的不确定性都低于 $\gamma$，说明模型足够自信，不触发 VR

### 3.3 静态触发策略（MemVR-static）

另一种更简单的方式：在验证集上穷举搜索所有可能的层，选择平均性能最佳的层作为固定触发层。局限在于：(1) 需要更多超参数调优；(2) 最优层对数据分布敏感。

### 3.4 动态注入比例变体（MemVR†）

为减少超参数 $\alpha$，可以将注入比例与不确定性绑定：

$$\alpha = 2(u - \gamma)$$

当不确定性刚超过阈值时 $\alpha$ 接近 0，不确定性越高注入越多。实验表明 MemVR† 与固定 $\alpha$ 的 MemVR 性能接近。

---

## 四、理论分析

论文从信息论角度提供了三个定理，解释 MemVR 为什么有效。

### 定理 1：MemVR 增强互信息

设 $\boldsymbol{x}$ 为 FFN 的隐状态，$\hat{\boldsymbol{x}}$ 为注入视觉证据 $\boldsymbol{z}_v$ 后的隐状态，则：

$$I(\hat{\boldsymbol{x}}; \boldsymbol{z}_v) \geq I(\boldsymbol{x}; \boldsymbol{z}_v)$$

**证明思路**：根据数据处理不等式（DPI），随着 Transformer 层数加深，隐状态与视觉特征的互信息单调不增。MemVR 通过在中间层直接注入 $\boldsymbol{z}_v$，打破了这一 Markov 链，确保 $\hat{\boldsymbol{x}}$ 保留了 $\boldsymbol{x}$ 的所有信息并额外包含了来自 $\boldsymbol{z}_v$ 的信息。

### 定理 2：MemVR 降低条件熵

若 $\boldsymbol{x}$ 与 $\boldsymbol{z}_v$ 的互信息增加，则目标输出 $\boldsymbol{y}$ 的条件熵降低：

$$H(\boldsymbol{y} \mid \hat{\boldsymbol{x}}) \leq H(\boldsymbol{y} \mid \boldsymbol{x})$$

直觉上，隐状态包含更多视觉信息→对输出的不确定性更低→幻觉概率更低。

### 定理 3：Information Bottleneck 优化

在 IB 框架下，MemVR 优化了目标函数 $\mathcal{L}(\hat{\boldsymbol{x}}) \leq \mathcal{L}(\boldsymbol{x})$，其中 $\mathcal{L}(\boldsymbol{x}) = I(\boldsymbol{x}; \boldsymbol{c}) - \beta I(\boldsymbol{x}; \boldsymbol{y})$。当 $\beta$ 足够大时（即更重视预测准确性），MemVR 通过增强 $I(\hat{\boldsymbol{x}}; \boldsymbol{y})$ 来降低 IB 损失。

---

## 五、实验结果

### 5.1 实验设置

- **模型**：LLaVA-1.5-7B、Qwen-VL-Chat、GLM-4V-9B、LLaVA-Next
- **基线**：VCD、OPERA、ICD、DoLa
- **Benchmark**：POPE（3 数据集 × 3 设置）、CHAIR、HallusionBench、MME、MMBench、MM-Vet、LLaVA-Bench、VizWiz
- **默认超参数**：$\gamma = 0.75$，贪心解码

### 5.2 幻觉基准结果

**POPE（LLaVA-1.5-7B）**：

| 数据集 | 方法 | Random Acc↑ | Popular Acc↑ | Adversarial Acc↑ | Average Acc↑ |
| --- | --- | --- | --- | --- | --- |
| MSCOCO | LLaVA-1.5 | 83.49 | 79.98 | 76.03 | 79.83 |
| | OPERA | 87.53 | 84.21 | 80.88 | 84.21 |
| | VCD | 86.84 | 82.65 | 77.31 | 82.27 |
| | **MemVR** | **88.50** | **87.10** | **85.20** | **86.93** |
| A-OKVQA | LLaVA-1.5 | 83.45 | 79.90 | 74.04 | 79.13 |
| | **MemVR** | **91.10** | **87.33** | **80.20** | **86.21** |

MemVR 在 POPE 上平均准确率提升 **+7.0%**，在最具挑战性的 Adversarial 设置上提升高达 **+9.2%**。

**CHAIR（LLaVA-1.5-7B）**：

| 方法 | CHAIR$_S$↓ | CHAIR$_I$↓ | Recall↑ |
| --- | --- | --- | --- |
| LLaVA-1.5 | 50.0 | 15.4 | 77.1 |
| OPERA | 47.8 | 14.6 | 76.8 |
| VCD | 48.6 | 14.9 | 77.3 |
| ICD | 56.2 | 16.3 | 16.3 |
| **MemVR** | **46.6** | **13.0** | **80.8** |

MemVR 在降低幻觉（CHAIR$_S$ -6.8%、CHAIR$_I$ -15.6%）的同时**提高了 Recall**（+3.7），说明不是通过生成更少内容来降低幻觉。

### 5.3 通用基准结果

**关键发现**：CD 方法在缓解幻觉的同时**普遍导致通用能力下降**，而 MemVR 是唯一同时在幻觉和通用基准上取得正向提升的方法。

| 基准 | LLaVA-1.5 | +OPERA | +VCD | +ICD | **+MemVR** |
| --- | --- | --- | --- | --- | --- |
| MME Overall | 1864.7 | 1784.3 ↓80 | 1872.9 ↑8 | 1594.8 ↓270 | **1896.7 ↑32** |
| LLaVA-Bench | 64.8 | 64.3 ↓0.5 | 63.2 ↓1.6 | 56.9 ↓7.9 | **65.2 ↑0.4** |
| MM-Vet | 31.1 | 32.0 ↑0.9 | 30.2 ↓0.9 | 25.9 ↓5.2 | **32.4 ↑1.3** |
| VizWiz | 50.0 | 50.8 ↑0.8 | 44.9 ↓5.1 | 37.6 ↓12.4 | **51.5 ↑1.5** |
| MMBench | 62.8 | 62.8 ↑0.0 | 54.2 ↓8.6 | 39.8 ↓23.0 | **63.8 ↑1.0** |

### 5.4 推理效率

| 方法 | 延迟（ms/token） | 吞吐量（token/ms） | 80 tokens 总耗时 | 显存（MB） |
| --- | --- | --- | --- | --- |
| Greedy | 65.71 (×1.00) | 0.015 (×1.00) | 5256.6 (×1.00) | 14257 (×1.00) |
| OPERA | 240.59 (×3.66) | 0.004 (×0.27) | 19247.2 (×3.66) | 21300 (×1.49) |
| VCD | 144.62 (×2.20) | 0.007 (×0.47) | 11569.3 (×2.20) | 14967 (×1.05) |
| **MemVR** | **68.32 (×1.04)** | **0.015 (×1.00)** | **5545.5 (×1.06)** | **14345 (×1.01)** |

MemVR 的延迟仅为贪心解码的 **1.04×**，远低于 VCD（2.20×）和 OPERA（3.66×），因为 VR 操作仅在 FFN 中添加少量视觉 token 的检索计算，且不需要多轮推理。

### 5.5 消融实验

**阈值 $\gamma$ 的影响**：$\gamma \in [0.6, 0.95]$ 范围内均有正向效果，最优值约 0.75。过低的阈值会在过早的层触发 VR，过高则难以触发。

**注入比例 $\alpha$ 的影响**：$\alpha \in [5\%, 35\%]$ 为正向效果，超过 35% 为负向效果。说明视觉记忆的补充存在上限——过多注入会破坏原有的语言建模能力。

**动态 vs 静态策略**：

| MME | Static-7 | Static-15 | Static-23 | Static-$\phi$ | **Dynamic** |
| --- | --- | --- | --- | --- | --- |
| Total | 1847.6 | 1881.2 | 1858.1 | 1889.2 | **1896.7** |

动态策略在无需验证集的情况下超越了静态最优层，验证了不确定性引导的有效性。

### 5.6 多模型泛化

MemVR 在 Qwen-VL-Chat 和 GLM-4V-9B 上同样有效：

| 模型 | MME Regular → +MemVR | LLaVA-Bench Regular → +MemVR |
| --- | --- | --- |
| Qwen-VL-Chat | 1784.9 → 1821.0 (+36) | 68.5 → 69.5 (+1.0) |
| GLM-4V-9B | 2160.5 → 2170.2 (+10) | 75.3 → 76.7 (+1.4) |

相比之下，VCD 和 ICD 在 Qwen-VL 和 GLM-4V 上通常导致严重的性能下降（LLaVA-Bench 降幅 5-15 分）。

---

## 六、局限性与未来方向

1. **超参数调优**：注入比例 $\alpha$ 和阈值 $\gamma$ 的最优值依赖于具体模型和任务，虽然 MemVR† 通过绑定 $\alpha$ 与不确定性缓解了这一问题，但仍需进一步简化
2. **多模态扩展**：虽然论文聚焦于视觉模态，但 MemVR 的框架理论上可以扩展到 listen-twice（音频）、scan-twice（空间感知）、check-twice（fMRI）等
3. **失败案例**：当原始视觉特征已经足够推理时，重新注入的 token 可能反而干扰推理（Type 1 failure）；当图像过于复杂或模型本身知识不足时，VR 也无法修复（Type 2 failure）

---

## 七、个人思考

### 7.1 与项目内其他幻觉缓解工作的对比

| 方法 | 干预阶段 | 干预位置 | 是否需要训练 | 推理延迟 | 通用能力影响 |
| --- | --- | --- | --- | --- | --- |
| **MemVR** | 解码时 | FFN hidden states | 否 | 极低（1.04×） | 正向提升 |
| HALC | 解码时 | Logits（FOV 对比） | 否 | 极高（2.4×） | 无影响 |
| DLC | 解码时 | Logits（CLIP 校准） | 否 | 低 | 无影响 |
| VisFlow | 解码时 | Attention matrix | 否 | 低 | 需关注 |
| HIME | 推理前 | MLP 权重编辑 | 否 | 零 | 轻微下降 |
| SENTINEL | 训练时 | 偏好学习 C-DPO | 是 | 零 | 正向提升 |

MemVR 的独特优势在于：(1) 干预 hidden states 而非 logits，避免了多轮推理；(2) 同时提升幻觉和通用能力，说明视觉信息的补充是一种"正和"操作。

### 7.2 FFN-as-Memory 视角的启示

MemVR 将 FFN 重新解读为 key-value memory 并利用其检索机制注入新信息，这一视角非常优雅。值得注意的是，这与 Embodied AI 领域的 UAOR（同在本项目中）思路惊人地相似——UAOR 也是在 VLA 的 FFN 层重新注入观测特征来缓解动作预测中的不确定性。两篇论文从不同领域（VLM 幻觉 vs VLA 推理）独立发现了同一个 insight：**FFN 的 key-value memory 机制可以作为信息再注入的天然接口**。

### 7.3 "不确定性驱动的自适应干预"范式

MemVR 使用层级不确定性来决定是否触发干预，这种**按需干预**的思想值得关注：对自信的 token 不做任何修改，仅在模型"犹豫"时提供额外证据。这比无差别干预（如 VCD 对每个 token 都做对比解码）更高效也更安全——避免了对正确预测的不必要干扰。

### 7.4 与对比解码方法的根本区别

MemVR 操作的是 hidden states（前向传播中间），而 CD 方法操作的是 logits（前向传播末端）。这意味着 MemVR 的修正信息可以被后续所有层**进一步加工和整合**，而 CD 方法的修正是"一锤子买卖"。从信息论角度看，在中间层注入信息允许后续层通过非线性变换将视觉信息与文本信息**深度融合**，而非简单地在概率分布上做加减。

---

## 参考

- **VCD**（Leng et al., 2024）：视觉对比解码——MemVR 的主要对比基线，通过噪声图像做对比但需 2× 推理
- **OPERA**（Huang et al., 2024a）：注意力过度信任惩罚——注意力干预范式的代表，推理延迟 3.66×
- **DoLa**（Chuang et al., 2023）：层间对比解码——MemVR 借鉴了其层级不确定性量化的思路
- **ICD**（Wang et al., 2024）：指令对比解码——修改文本输入做对比，但导致通用能力严重下降
- **Geva et al. (2021)**：Transformer FFN 是 Key-Value Memory——MemVR 的理论基础
- **UAOR**（2026）：VLA 领域的类似工作，同样利用 FFN memory 重注入观测特征缓解不确定性
