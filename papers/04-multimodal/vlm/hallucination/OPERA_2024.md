# OPERA：通过过度信任惩罚与回溯分配缓解多模态大模型幻觉

> **论文**：*OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation*
>
> **作者**：Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, Nenghai Yu
>
> **机构**：University of Science and Technology of China (USTC)、Shanghai AI Laboratory、The Chinese University of Hong Kong (CUHK)
>
> **发布时间**：2023年10月（**CVPR 2024**）
>
> 🔗 [arXiv](https://arxiv.org/abs/2311.17911) | [代码](https://github.com/shikiw/OPERA)
>
> **分类标签**：`对象幻觉` `解码策略` `注意力分析` `Beam Search` `Training-Free`

---

## 一句话总结

发现 MLLM 幻觉与自注意力矩阵中的**知识聚合模式**（少数 summary token 呈柱状注意力吸引后续所有 token 注意力）高度共现，提出在 Beam Search 解码中引入**过度信任 logit 惩罚**（列乘积度量聚合强度）和**回溯重分配策略**（检测到持续聚合时回滚到 summary token 重选），无需额外训练/数据/知识，在 InstructBLIP/MiniGPT-4/LLaVA-1.5/Shikra 上 CHAIR$_S$ 平均降低 20%+。

---

## 一、问题与动机

### 1.1 MLLM 幻觉：严重且普遍

多模态大语言模型（MLLM）在图像描述中频繁产生**对象幻觉**——描述图中不存在的物体、错误的属性或关系。例如 LLaVA-1.5 对一碗燕麦的图片描述中凭空生成了"bottle"、"cup"、"book"等不存在的物体。

### 1.2 现有方法的代价

| 方法类型 | 代表 | 局限 |
| --- | --- | --- |
| 训练阶段 | LRV-Instruction | 需要大量额外标注数据 |
| 后处理 | Woodpecker | 依赖外部更强模型（ChatGPT） |
| 外部知识 | VIGC | 引入额外模型和知识源 |

**共同缺陷**：都引入了显著的额外成本（数据、计算、外部依赖）。

### 1.3 核心观察：幻觉与"过度信任"注意力模式共现

论文通过可视化自注意力矩阵发现了一个关键现象——**partial over-trust**（部分过度信任）：

1. **柱状注意力模式**：在自注意力矩阵中，某些 token（通常是句号、引号等低信息量 token）会呈现**柱状注意力**——即后续所有 token 都将大量注意力分配给它们
2. **这些 token 是 summary token**：它们在浅层聚合了前文的关键知识，在深层被后续 token 过度依赖
3. **幻觉在柱状模式后出现**：统计 100 张图片发现，大部分幻觉内容在知识聚合模式出现后的 10 个 token 内开始

这一观察与 NLP 领域的 **anchor token** 现象一致：LLM 倾向在浅层将信息聚合到少数 anchor token 上，在深层基于这些 anchor 预测下一个 token。

### 1.4 为什么聚合模式导致幻觉？

MLLM 的视觉 token 通常放在序列开头。随着生成变长：

- summary token 越来越多
- 视觉信息在 summary token 间传递时逐渐**衰减**（单个 summary token 无法记住全部上下文的密集信息）
- 后续 token 忽略开头的图像 token，转而**过度信任**距离更近的 summary token
- 模型基于文本偏差而非视觉内容生成，导致幻觉（如从"road"联想出"cars"）

**实证验证**：将 MLLM 的长回复按 summary token 位置分段，分别计算 CHAIR 分数。结果显示 CHAIR 分数与分段编号呈**正相关**——越靠后（经过越多 summary token），幻觉越多。

---

## 二、核心方法

OPERA 基于 Beam Search 解码，包含两个核心组件：**Over-Trust Logit Penalty**（过度信任惩罚）和 **Retrospection-Allocation Strategy**（回溯分配策略）。

### 2.1 MLLM 生成流程回顾

MLLM 的生成可分为三步：

1. **输入构造**：视觉 token $\mathbf{x}^v = \{x_0, \ldots, x_{N-1}\}$（$N$ 个）+ 文本 token $\mathbf{x}^p = \{x_N, \ldots, x_{M+N-1}\}$，拼接为完整序列 $\{x_i\}_{i=0}^{T-1}$，$T = N + M$
2. **模型前向**：$\mathbf{h} = \text{MLLM}(\mathbf{x}_i)$，再通过词表头 $\mathcal{H}$ 得到下一个 token 的概率：$p(x_t | x_{<t}) = \text{Softmax}[\mathcal{H}(h_t)]_{x_t}$
3. **解码**：Beam Search 维护 $N_{\text{beam}}$ 条候选序列，逐步扩展并选择累积得分最高的假设

### 2.2 Over-Trust Logit Penalty（过度信任惩罚）

**核心思想**：在 beam score 中引入累积惩罚项，使包含知识聚合模式的候选序列得分降低，从而被淘汰。

**Step 1：提取局部窗口注意力。** 在最后一层自注意力中，取多头的最大值，裁剪出**仅覆盖生成 token** 的局部窗口（不包含图像和 prompt token）：

$$\mathbf{W}_{t-1}^k = \{\mathbf{w}^i\}_{i=t-k}^{t-1}, \quad \text{s.t. } \mathbf{w}^i = \{\omega_{i,j}\}_{j=t-k}^{i}$$

其中 $k$ 是窗口大小，$\omega_{i,j}$ 是第 $j$ 个 token 对第 $i$ 个 token 的注意力权重。窗口起始位置满足 $t - k \geq N + M$，确保只关注生成的 token。

**Step 2：缩放与列乘积。** 将窗口注意力值乘以缩放因子 $\sigma$（默认 50），使聚合模式处的注意力值 > 1、弱注意力处 < 1：

$$\mathbf{W}_{t-1}^k \triangleq \{\mathbf{w}^i\}_{i=t-k}^{t-1}, \quad \text{s.t. } \mathbf{w}^i = \{\sigma \omega_{i,j}\}_{j=t-k}^{t-1}$$

上三角补零（因果掩码），然后对下三角做**列乘积**——对每一列 $j$，将该列所有非零值相乘。直觉：如果某个 token 被后续很多 token 都高度关注（柱状模式），其列乘积就会非常大。

**Step 3：取最大列乘积作为惩罚值。** 选择列乘积最大的那一列作为聚合模式的强度度量：

$$\phi(\omega_{<t}) = \prod_{i=c}^{t-1} \sigma \omega_{i,c}, \quad \text{s.t. } c = \arg\max_{t-k \leq j \leq t-1} \prod_{i=j}^{t-1} \sigma \omega_{i,j}$$

**Step 4：融入 beam score。** 从每个 beam 的 logit 中取 top-$N_{\text{can}}$（默认 5）组成候选集 $\mathcal{Y}$，修改 token 选择概率：

$$p(x_t | x_{<t}) = \text{Softmax}[\mathcal{H}(h_t) - \alpha \phi(\omega_{\leq t})]_{x_t}, \quad x_t \in \mathcal{Y}$$

其中 $\alpha = 1$ 是惩罚权重。**用大白话说**：如果当前候选序列的注意力矩阵中出现了强烈的柱状模式，$\phi$ 值会很大，所有候选 token 的 logit 都被减去这个惩罚值，使该序列在 beam 竞争中处于劣势。

### 2.3 Retrospection-Allocation Strategy（回溯分配策略）

**问题**：惩罚项有**滞后性**——知识聚合模式需要生成几个后续 token 后才能被观察到，此时幻觉可能已经发生。极端情况下，所有 beam 的候选都已包含幻觉。

**核心思想**：当检测到持续的聚合模式时，**回滚**到 summary token 位置，排除之前的选择，重新选择后续 token。

**Step 1：收集位置坐标。** 对最近 $l$ 个 token（默认 $l = k$），分别计算各自局部窗口的最大列乘积位置 $c$，组成位置集合：

$$\mathcal{C} = \{c \mid c = \arg\max_{t-k \leq j \leq z} \prod_{i=j}^{z} \sigma \omega_{i,j},\ z \in [t-l, t-1]\}$$

**Step 2：检查位置重叠。** 计算集合 $\mathcal{C}$ 中众数 $s = \text{Mode}(\mathcal{C})$ 的出现次数：

$$N_{\text{overlap}} = \sum_{c \in \mathcal{C}} \mathbb{1}_{c=s}$$

如果 $N_{\text{overlap}} \geq r$（默认 $r = 15$），说明连续多个 token 都指向同一个 summary token $x_s$，聚合模式持续存在。

**Step 3：回滚与重选。** 将序列回滚到 $\{x_0, \ldots, x_s\}$，在候选集 $\mathcal{Y} \setminus \{x_{s+1}\}$ 中重新选择 $x_s$ 之后的 token。

**约束条件**：
- 回滚位置 $s$ 必须**单调不递减**（防止无限回滚）
- 每个位置最多回滚 $\beta$ 次（默认 5），超过则再向前回滚一步到 $x_{s-1}$

### 2.4 超参数设置

| 超参数 | 含义 | 默认值 |
| --- | --- | --- |
| $N_{\text{beam}}$ | Beam 大小 | 5 |
| $N_{\text{can}}$ | 每个 beam 的候选数 | 5 |
| $\sigma$ | 注意力缩放因子 | 50 |
| $\alpha$ | 惩罚权重 | 1 |
| $k$ | 局部窗口大小 | (论文未明确指定统一值) |
| $r$ | 回溯触发阈值 | 15 |
| $\beta$ | 最大回滚次数 | 5 |

所有超参数在四个 MLLM 上**统一设置**，无需逐模型调优。

---

## 三、实验结果

### 3.1 CHAIR 评估（MSCOCO 500 张图）

**长描述（max new tokens = 512）：**

| 方法 | InstructBLIP $C_S$↓ | $C_I$↓ | MiniGPT-4 $C_S$↓ | $C_I$↓ | LLaVA-1.5 $C_S$↓ | $C_I$↓ | Shikra $C_S$↓ | $C_I$↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 58.8 | 23.7 | 31.8 | 9.9 | 45.0 | 14.7 | 55.8 | 15.4 |
| Nucleus | 54.6 | 24.8 | 32.6 | 10.7 | 48.8 | 14.2 | 55.6 | 15.4 |
| Beam Search | 55.6 | 15.8 | 30.6 | 9.5 | 48.8 | 13.9 | 50.4 | 13.3 |
| DoLa | 48.4 | 15.9 | 32.2 | 10.0 | 47.8 | 13.8 | 55.8 | 15.1 |
| **OPERA** | **46.4** | **14.2** | **26.2** | **9.5** | **44.6** | **12.8** | **36.2** | **12.1** |

在 Shikra 上，$C_S$ 从 DoLa 的 55.8 降至 36.2（**~35% 改进**）。

**短描述（max new tokens = 64）：**

| 方法 | InstructBLIP $C_S$↓ | $C_I$↓ | MiniGPT-4 $C_S$↓ | $C_I$↓ | LLaVA-1.5 $C_S$↓ | $C_I$↓ | Shikra $C_S$↓ | $C_I$↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 30.0 | 14.5 | 24.2 | 8.2 | 20.6 | 6.2 | 22.0 | 7.0 |
| Beam Search | 21.4 | 7.2 | 23.6 | 7.8 | 18.8 | 5.9 | 20.2 | 6.4 |
| DoLa | 22.2 | 7.1 | 24.2 | 8.2 | 20.4 | 6.3 | 20.2 | 6.3 |
| **OPERA** | **16.6** | **6.8** | **22.6** | **8.2** | **14.2** | **5.2** | **14.2** | **5.9** |

短描述和长描述下均一致优于所有基线。

### 3.2 GPT-4 辅助评估（VG-100K）

| 指标 | 含义 | OPERA vs Greedy | OPERA vs DoLa |
| --- | --- | --- | --- |
| HSR↓ | 幻觉句子比例 | ~30.4% 改进 | ~15.4% 改进 |
| HWR↓ | 幻觉词比例 | 显著降低 | 显著降低 |
| SPI↑ | 每图句子数 | 略有减少 | — |

OPERA 在减少幻觉的同时仅轻微减少输出长度，可能是因为去掉了多余的幻觉内容。

### 3.3 GPT-4V 辅助评估（MSCOCO 500 张图）

| 方法 | InstructBLIP C/D | MiniGPT-4 C/D | LLaVA-1.5 C/D | Shikra C/D |
| --- | --- | --- | --- | --- |
| Beam Search | 5.52 / 5.26 | 5.29 / 5.06 | 5.53 / 5.15 | 5.25 / 5.08 |
| **OPERA** | **6.26** / 5.27 | **6.87** / 5.08 | **6.32** / 5.16 | **6.29** / 5.26 |

正确性（C）最高提升 **27.5%**（MiniGPT-4），详细度（D）基本持平。

### 3.4 POPE 评估

| 方法 | InstructBLIP | MiniGPT-4 | LLaVA-1.5 | Shikra |
| --- | --- | --- | --- | --- |
| Greedy | 80.0 | 58.5 | 82.2 | 81.1 |
| Beam Search | 84.4 | 70.3 | 84.9 | 82.5 |
| DoLa | 83.4 | 72.8 | 83.2 | 82.1 |
| **OPERA** | **84.8** | **73.3** | **85.4** | **82.7** |

POPE 提升边际但一致。论文坦诚指出：POPE 的 Yes/No 回答太短，知识聚合模式不容易出现——OPERA 的优势主要体现在**长序列生成**中。

### 3.5 文本质量与通用基准

| 指标 | Greedy | Beam Search | DoLa | OPERA |
| --- | --- | --- | --- | --- |
| PPL$_1$↓ | 12.72 | **11.11** | 12.89 | 11.67 |
| PPL$_2$↓ | 10.27 | **8.89** | 10.40 | 9.31 |
| Grammar↑ | **9.58** | 9.54 | 9.31 | 9.54 |
| Fluency↑ | 9.01 | **8.95** | 8.89 | 8.93 |

| 基准 | Greedy | Beam | DoLa | **OPERA** |
| --- | --- | --- | --- | --- |
| MMBench | 64.3 | 64.4 | 63.8 | **64.4** |
| MME | 1510.7 | 1504.3 | 1480.1 | **1515.4** |

OPERA 不仅不损害文本质量和通用能力，甚至在 MME 上略有提升。

---

## 四、局限性与未来方向

### 4.1 依赖 Beam Search

OPERA 基于 Beam Search 解码，相比 greedy 解码速度更慢（$N_{\text{beam}} = 5$ 意味着约 5× 前向计算量）。对实时应用不够友好。

### 4.2 短序列效果有限

POPE 实验表明，对于简短回答（Yes/No），知识聚合模式不易出现，OPERA 的优势不明显。核心收益集中在长文本生成场景。

### 4.3 仅验证 7B 模型

所有实验均在 7B 级别模型上进行。更大模型的注意力模式可能不同，OPERA 的有效性和超参数泛化性有待验证。

### 4.4 缩放因子的敏感性

$\sigma = 50$ 的设定是为了让聚合模式处的注意力值经缩放后 > 1，但不同模型的注意力分布差异可能需要不同的 $\sigma$。论文虽然在 4 个模型上统一使用 $\sigma = 50$，但更广泛的泛化性未知。

---

## 五、个人思考

### 5.1 与项目内其他幻觉缓解工作的对比

| 方法 | 干预阶段 | 核心机制 | 额外成本 | 长序列优势 |
| --- | --- | --- | --- | --- |
| **OPERA** | 解码时 | 注意力柱状模式检测 + beam 惩罚 + 回滚 | Beam Search（~5× 前向） | 强 |
| HALC | 解码时 | FOV 对比 + 视觉匹配 beam search | 多次前向 + 外部检测器 | 强 |
| VisFlow | 推理时 | 双层注意力干预（token + head 级） | 几乎零开销 | 中 |
| DLC | 解码时 | CLIP 探针动态校准 logits | CLIP 前向 | 中 |
| HIME | 推理前 | 模型编辑 MLP 权重 | 零推理开销 | — |
| MemVR | 推理时 | FFN key-value 视觉回溯 | 1.04× 延迟 | 中 |
| SENTINEL | 训练时 | 句子级 C-DPO | 训练成本 | — |
| CSR | 训练时 | CLIP 校准自奖励 + 迭代 DPO | 训练成本 | — |

OPERA 的独特价值在于它**从注意力机制的内部视角揭示了幻觉的成因**，而非仅仅从输出端修正。这一观察启发了后续大量工作（如 VisFlow 对注意力头的分类分析）。

### 5.2 "Summary Token"观察的深远影响

OPERA 发现的 summary token / anchor token 现象后来被 NLP 社区广泛验证：

- **Attention Sink**（StreamingLLM）发现第一个 token 总是获得大量注意力——这也是一种聚合模式
- **VisFlow** 进一步将注意力头分为 Visual Sink Head / System Prompt Head / Text Following Head，并发现抑制后两者可以缓解幻觉

OPERA 可以说是**最早将这一注意力现象与 MLLM 幻觉建立因果联系**的工作之一。

### 5.3 列乘积度量的巧妙之处

用列乘积（而非列求和或列最大值）来度量聚合模式强度是一个精巧的设计：
- 列**求和**无法区分"一个 token 被所有后续高度关注"（柱状）和"一个 token 被少数后续极度关注"（尖刺）
- 列**最大值**丢失了"持续性"信息
- 列**乘积**要求该 token 被**连续多个**后续 token 都有一定关注度，才能产生大值——这恰好是柱状模式的定义

### 5.4 回溯策略的局限

回溯策略虽然直觉合理，但有一个潜在问题：回滚后重新选择的 token 仍然基于同一个模型的同一组注意力权重——如果模型本身的知识偏差很强（如"road"→"cars"），排除一个候选后选出的下一个候选可能同样是幻觉。这也解释了为什么 OPERA 在 POPE 这种简单场景下提升有限。

---

## 参考

- **DoLa**（Chuang et al., 2023）：层间对比解码缓解 LLM 幻觉——OPERA 的基线之一，DoLa 从"层"维度寻找对比信号，OPERA 从"注意力模式"维度检测幻觉
- **Anchor Token**（Wang et al., 2023）：NLP 中发现 LLM 将信息聚合到少数 anchor token 的现象——OPERA 的核心观察与之一致
- **HALC**（Chen et al., 2024）：自适应 FOV 对比解码——另一种解码阶段方法，从视觉上下文角度缓解幻觉，与 OPERA 从注意力模式角度互补
- **VCD**（Leng et al., 2023）：视觉对比解码，添加噪声视觉输入做对比——比 OPERA 更简单但效果也更弱
- **StreamingLLM**（Xiao et al., 2023）：发现 Attention Sink 现象——与 OPERA 的 summary token 观察密切相关
