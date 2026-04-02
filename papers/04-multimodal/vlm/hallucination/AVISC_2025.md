# AVISC：通过注意力视觉校准缓解大视觉-语言模型幻觉

> **论文**：*Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models*
>
> **作者**：Sangmin Woo*, Donguk Kim*, Jaehyuk Jang*, Yubin Choi, Changick Kim
>
> **机构**：KAIST
>
> **发布时间**：2024年5月（arXiv），**ACL 2025 Findings** 录用
>
> [arXiv](https://arxiv.org/abs/2405.17820) | [代码](https://github.com/sangminwoo/AvisC)
>
> **分类标签**：`LVLM` `Hallucination` `Decoding Strategy` `Training-Free` `Attention` `Contrastive Decoding`

---

## 一句话总结

发现 LVLM 中存在**盲 token (blind tokens)**——少数图像 token 垄断注意力权重却不携带物体判别信息，提出 AVISC 在解码阶段通过层选择、盲 token 识别和对比解码三步动态校准注意力分布，training-free 即插即用，POPE 上 InstructBLIP 平均 Accuracy 提升 6%+、AMBER 得分达 85.95。

---

## 一、问题与动机

### 1.1 盲 token 现象

论文深入分析了 LVLM 的注意力权重分布，发现了一个关键现象：

> **即使图像不包含任何与 query 相关的内容（如纯色图像），LVLM 的注意力仍然集中在少数几个固定的图像 token 上。** 这些 token 被称为 blind tokens——它们垄断了注意力资源，却不承载有意义的物体判别信息。

具体实验验证（Fig. 2）：
- **置零高注意力 token（blind tokens）**：对预测 logits 几乎没有影响（Yes/No 概率变化 < 0.2）
- **置零低注意力 token（non-blind tokens）**：logits 剧烈变化，概率趋近 50:50，丧失判别能力

> **直觉**：真正携带物体判别信息的 token 反而是那些注意力权重较低的 token。盲 token 就像"噪声放大器"——占据了模型大量注意力资源，却对最终判断贡献甚微。

### 1.2 现有方法的不足

| 方法类别 | 代表方法 | 局限 |
| --- | --- | --- |
| 训练级 | RLHF-V, HACL | 需要额外训练数据和计算资源 |
| 外部模型引导 | HALC, CFG-based | 依赖 CLIP/DETR 等外部模型 |
| 自反馈 | Volcano | 需要多次自推理，开销大 |
| 对比解码 | VCD, M3ID | 使用噪声/空图像作为对比基准，未利用模型内部注意力模式 |

**关键差距**：VCD 通过向图像添加扩散噪声构造"坏"分布，M3ID 对比有图/无图分布——两者都不直接针对注意力偏差的根因。AVISC 直接从注意力模式出发，精准识别并抑制盲 token 的影响。

---

## 二、预备知识

### 2.1 LVLM 框架

标准 LVLM 处理流程：视觉编码器（如 CLIP）产生 $N$ 个视觉 token $\mathcal{V} = \{\nu_0, \nu_1, \ldots, \nu_{N-1}\}$，文本 tokenizer 产生 $M$ 个文本 token $\mathcal{Q} = \{\sigma_N, \sigma_{N+1}, \ldots, \sigma_{N+M-1}\}$，拼接后送入 LLM 自回归生成：

$$\ell_t = \log p(\xi_t \mid \mathcal{V}, \mathcal{Q}, \xi_{<t}; \theta)$$

其中 $\ell_t$ 为第 $t$ 步的 logits，$\xi_t$ 为待预测 token，$\xi_{<t}$ 为已生成的前缀。

### 2.2 对比解码

对比解码的核心思想是构造一个"差"分布，从原始分布中减去它以抑制不良倾向。VCD 使用噪声图像 $\mathcal{V}_{\text{noise}}$，M3ID 使用无图像条件：

$$\xi_t \sim \text{Softmax}\left((1+\alpha)\ell_t^{\text{orig}} - \alpha \ell_t^{\text{bad}}\right)$$

AVISC 的创新在于：不使用外部构造的"坏"输入，而是**从模型自身的注意力模式中提取盲 token 构造有针对性的对比基准**。

---

## 三、核心方法

AVISC 在每个解码步执行三个操作：**层选择 → 盲 token 识别 → 对比解码**。

### 3.1 层选择 (Layer Selection)

不同 LVLM 的注意力分布模式差异显著：InstructBLIP 的图像注意力集中在后层，LLaVA-1.5 集中在前层。因此需要自适应选择与图像 token 相关度高的层。

**第 $i$ 层的注意力权重矩阵**：

$$\mathbf{A}_i = \left[\mathbf{a}^i_{h,q,k}\right]_{(h,q,k)=(1,1,1)}^{(H, N+M, N+M)}$$

其中 $\mathbf{a}^i_{h,q,k}$ 表示第 $i$ 层第 $h$ 个注意力头中 query $q$ 对 key $k$ 的注意力权重。

**图像注意力占比**：计算最后一个 token（即当前生成位置 $N+M$）分配给图像 token 的注意力在所有层中的相对占比：

$$AP_i^{\text{layer}} = \frac{\sum_h \sum_{k=1}^{N} \mathbf{a}^i_{h,(N+M),k}}{\sum_{i,h} \sum_{k=1}^{N} \mathbf{a}^i_{h,(N+M),k}}$$

> **直觉**：$AP_i^{\text{layer}}$ 衡量"第 $i$ 层在所有层的图像注意力总量中占了多少比例"。占比越高的层，越可能包含有意义的图像-文本交互信号。

按 $AP_i^{\text{layer}}$ 降序排列所有层，使用 **top-P 采样**（累积占比达到阈值 $\gamma$）选择层集合：

$$\{\text{Selected Layers}\} = \text{top-P}(\{AP_i^{\text{layer}}\}_{i=1}^{L}, \gamma)$$

实验中 $\gamma = 0.5$，即选择图像注意力占比累积达 50% 的前若干层。

### 3.2 盲 token 识别 (Blind Token Identification)

在选定层内，计算每个图像 token 的平均注意力权重：

$$AP^{\text{image}} = \frac{\sum_{i \in \{\text{Selected Layers}\}} \sum_{h=1}^{H} \mathbf{a}^i_{h,(N+M),[1:N]}}{|\{\text{Selected Layers}\}| \times H}$$

$AP^{\text{image}}$ 是一个 $N$ 维向量，$AP_j^{\text{image}}$ 表示第 $j$ 个图像 token 在所有选定层和注意力头上的平均注意力权重。

计算均值 $\mu$ 和标准差 $\sigma$，将注意力权重超过 $\mu + \lambda\sigma$ 的 token 识别为盲 token：

$$\{\text{Blind Token Indices}\} = \{j \mid AP_j^{\text{image}} > \mu + \lambda\sigma\}$$

> **直觉**：$\mu + \lambda\sigma$ 阈值筛选出注意力分布中的"离群值"——那些远高于平均水平的 token。$\lambda$ 越大，只有注意力越极端的 token 才会被标记。实验中 $\lambda = 1$。

### 3.3 对比解码 (Contrastive Decoding)

构造偏置视觉输入 $\mathcal{V}^*$：**只保留盲 token，将所有非盲 token 置零**：

$$\mathcal{V}^* = \bigcup_{j=1}^{N} \mathbb{1}_{\{j \in \text{Blind Token Indices}\}}(j) \cdot \nu_j$$

分别用原始输入和偏置输入计算 logits：

$$\ell_t = \log p(\xi_t \mid \mathcal{V}, \mathcal{Q}, \xi_{<t}; \theta), \quad \ell_t^* = \log p(\xi_t \mid \mathcal{V}^*, \mathcal{Q}, \xi_{<t}; \theta)$$

对比解码：

$$\xi_t \sim \text{Softmax}\left((1+\alpha)\ell_t - \alpha \ell_t^*\right)$$

> **直觉**：$\ell_t^*$ 是模型"只看盲 token"时的预测——它捕获了模型对无信息 token 的偏见。从原始预测中减去这个偏见，等价于**增强那些在原始分布中高概率但在偏置分布中低概率的 token**——即那些真正依赖非盲 token（携带判别信息）的预测。

$\alpha$ 控制对比强度：InstructBLIP 使用 $\alpha = 3$，LLaVA-1.5 使用 $\alpha = 2.5$。

### 3.4 截断采样

遵循 VCD 的实践，使用 cut-off sampling 过滤低概率 token：

$$\mathcal{H}(\xi_{<t}) = \{\xi_t \in \mathcal{V}_{\text{vocab}} : p(\xi_t \mid \mathcal{V}, \mathcal{Q}, \xi_{<t}; \theta) \geq \beta \cdot \max_w p(w \mid \mathcal{V}, \mathcal{Q}, \xi_{<t}; \theta)\}$$

实验中 $\beta = 0.1$，仅保留概率不低于最大概率 10% 的 token 参与采样。

---

## 四、实验结果

### 4.1 POPE 判别性评估

在 POPE 上跨 MS-COCO、A-OKVQA、GQA 三个子集、三种设置（Random/Popular/Adversarial）的综合表现：

**InstructBLIP 7B**（$\alpha = 3$）：

| 子集 | 设置 | base Acc/F1 | VCD Acc/F1 | M3ID Acc/F1 | **AVISC Acc/F1** |
| --- | --- | --- | --- | --- | --- |
| MS-COCO | Random | 82.27/82.11 | 83.37/83.24 | 84.37/84.31 | **88.73/88.03** |
| MS-COCO | Popular | 77.77/79.02 | 78.00/79.19 | 77.30/78.71 | **83.90/84.53** |
| MS-COCO | Adversarial | 73.13/75.46 | 75.87/77.36 | 76.03/77.79 | **81.57/81.92** |
| A-OKVQA | Random | 81.00/82.06 | 81.73/82.66 | 82.33/83.66 | **88.47/88.59** |
| GQA | Random | 80.00/81.02 | 81.73/82.45 | 80.57/81.85 | **86.47/86.57** |

InstructBLIP 上 AVISC 的提升极为显著，Random 设置下 Accuracy 平均提升 **6-7 个百分点**。

**LLaVA-1.5 7B**（$\alpha = 2.5$）：

| 子集 | 设置 | base Acc/F1 | VCD Acc/F1 | M3ID Acc/F1 | **AVISC Acc/F1** |
| --- | --- | --- | --- | --- | --- |
| MS-COCO | Random | 84.47/84.72 | 84.80/85.20 | 86.00/86.18 | **87.93/87.88** |
| MS-COCO | Popular | 82.23/82.95 | 82.27/83.15 | 82.83/83.72 | **84.33/84.96** |
| A-OKVQA | Random | 82.73/84.26 | 81.30/83.23 | 83.57/85.09 | **84.60/85.88** |
| GQA | Random | 82.40/83.99 | 82.27/84.22 | 82.83/84.62 | **85.00/86.45** |

LLaVA-1.5 上提升相对温和（1-3%），在 Popular/Adversarial 设置下部分场景接近持平，但整体依然一致性最优。

### 4.2 MME 幻觉评估

| 模型 | 方法 | Existence | Count | Position | Color | **Total** |
| --- | --- | --- | --- | --- | --- | --- |
| InstructBLIP | base | 170.19 | 89.52 | 67.62 | 114.76 | 442.09 |
| InstructBLIP | VCD | 172.62 | **98.33** | 71.90 | 117.14 | 459.99 |
| InstructBLIP | **AVISC** | **184.76** | 82.85 | **74.76** | **131.43** | **473.80** |
| LLaVA-1.5 | base | 173.57 | 110.00 | 100.47 | 125.24 | 509.28 |
| LLaVA-1.5 | VCD | 172.14 | **117.14** | 103.33 | 119.52 | 512.14 |
| LLaVA-1.5 | **AVISC** | **189.29** | 104.76 | **106.19** | 127.86 | **528.09** |

AVISC 在 Existence 和 Color 上优势显著，Total Score 两个模型分别提升 31.7 和 18.8。Count 指标上 VCD 更优——这暗示盲 token 可能在计数任务中包含有用信息。

在 **MME-Fullset**（14 个类别）上，AVISC 在 InstructBLIP 的 7/14 类别、LLaVA-1.5 的 11/14 类别中取得最佳，说明注意力校准不仅缓解幻觉，还整体提升了视觉理解能力。

### 4.3 AMBER 综合评估

| 指标 | InstructBLIP base | InstructBLIP AVISC | LLaVA base | LLaVA AVISC |
| --- | --- | --- | --- | --- |
| CHAIR↓ | 8.40 | **6.70** | 7.95 | **6.25** |
| Hal↓ | 31.10 | **28.00** | 31.00 | **25.60** |
| Acc.↑ | 68.20 | **72.60** | 67.00 | **70.70** |
| F1↑ | 74.60 | **78.60** | 71.10 | **75.45** |
| **AMBER↑** | 83.10 | **85.95** | 81.58 | **84.60** |

AVISC 在生成式和判别式任务上均取得最高 AMBER 得分，验证了方法的综合优势。

### 4.4 消融实验

**非盲 token 去活化方式**（POPE-COCO-Random）：

| 方式 | InstructBLIP Acc/F1 | LLaVA Acc/F1 |
| --- | --- | --- |
| Zeros（默认） | 88.50/87.86 | **87.87/87.83** |
| Ones | 82.50/84.62 | 79.97/82.94 |
| Noise | 86.77/87.15 | 88.47/87.80 |
| Mask | **88.53/88.30** | 84.77/84.44 |

Zeros 在两个模型上综合表现最稳定。Ones 效果最差——将非盲 token 设为全 1 向量引入了错误的正向信号。

**超参数消融**（MME-Hallucination）：
- $\alpha$（对比强度）：InstructBLIP 在 $\alpha = 3$、LLaVA 在 $\alpha = 2.5$ 达到最优 Total Score；$\alpha$ 越大整体越好，但 Count 类别可能下降
- $\lambda$（盲 token 阈值）：$\lambda$ 从 0 增加到 1 时性能持续提升（$\lambda = 0$ 即将所有 token 视为盲 token，Total Score 仅 430 vs $\lambda = 1$ 的 478），说明精准定位少数极端盲 token 比泛化抑制更有效

---

## 五、局限性与未来方向

1. **Count 任务性能下降**：AVISC 在 MME Count 和 AMBER Numbers 上表现不佳。论文推测在计数场景中盲 token 可能承载了重要的空间布局信息，简单抑制它们会损害计数能力。
2. **需要双次前向传播**：每步解码需分别对原始和偏置输入计算 logits，推理延迟约为 2×。
3. **仅验证 7B 模型**：实验仅在 InstructBLIP 和 LLaVA-1.5（均为 Vicuna 7B 骨干）上验证，未测试更大规模模型或更新架构。
4. **静态阈值**：$\lambda$ 在所有 token 生成步中保持固定，未考虑不同解码阶段可能需要不同程度的校准。

---

## 六、个人思考

### 6.1 与项目中其他论文的联系

**与 VCD 的关系（同为对比解码，对比基准不同）**：VCD 通过向图像添加扩散噪声构造"坏"视觉输入，AVISC 从模型自身注意力模式中提取盲 token 构造"坏"输入。AVISC 的优势在于对比基准是数据驱动的（基于当前输入的注意力分布），而非固定的噪声策略，因此理论上对不同输入的适应性更强。

**与 OPERA 的对比（都分析注意力模式）**：OPERA 发现"注意力柱状聚合"与幻觉共现，通过 beam search 惩罚聚合 token；AVISC 发现"盲 token 垄断注意力"并通过对比解码抑制。两者分析的注意力异常模式不同——OPERA 关注注意力在序列维度的聚合（attention sink），AVISC 关注注意力在图像 token 间的不均匀分配。

**与 VisFlow 的对比（双层注意力干预 vs 对比解码）**：VisFlow 在模型内部做 token 级 + head 级注意力干预，直接修改注意力权重；AVISC 不修改模型内部，而是在输入-输出层面做对比。VisFlow 更精细但需要更深的模型理解（head 分类），AVISC 更简洁且模型无关。

**与 TAF 的对比（同为注意力分析驱动）**：TAF 区分 phantom token（文本→视觉异常高影响）和 anchor token（关键视觉证据不足），在注意力 logits 层面做非对称过滤。AVISC 的盲 token 概念与 TAF 的分析角度不同——AVISC 关注的是"注意力权重高但判别力低"的 token，TAF 关注的是跨模态影响的方向性。两者或可互补。

### 6.2 盲 token 现象的深层含义

盲 token 的存在揭示了一个重要现象：**LVLM 的注意力权重与 token 的信息量之间存在严重脱节**。这与近期 attention sink 研究（如 StreamingLLM）中发现的 BOS token 吸收大量注意力但不携带语义信息的现象高度一致。AVISC 可以看作是将 attention sink 的发现从纯文本领域扩展到多模态领域，并提供了一种实用的解码时校准方案。

### 6.3 方法简洁性

AVISC 的三步流程（层选择 → 盲 token 识别 → 对比解码）设计简洁，且三个超参数（$\gamma$、$\lambda$、$\alpha$）的语义清晰、可解释。特别是 $\lambda$ 的消融实验表明 $\lambda = 0$（将所有 token 视为盲 token）效果极差，验证了"精准打击"优于"全面抑制"的直觉。

---

## 参考

- **VCD (CVPR 2024)**：扩散噪声视觉对比解码，AVISC 的主要对比基线
- **M3ID (2024)**：多模态幻觉控制，基于有图/无图分布对比
- **OPERA (CVPR 2024)**：注意力聚合模式惩罚 + beam search 回溯
- **AGLA (CVPR 2025)**：GradCAM 增强图像 + 全局-局部 logit 融合，与 AVISC 同为 training-free 解码策略但机制互补
- **TAF (2026)**：Phantom/Anchor token 非对称注意力过滤，同为注意力分析驱动
- **VisFlow (2025)**：双层注意力干预（token 级 + head 级），在模型内部修改注意力
