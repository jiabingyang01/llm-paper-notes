# FarSight：注意力因果解码缓解 MLLM 幻觉

> **论文**：*Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding*
>
> **作者**：Feilong Tang*, Chengzhi Liu*, Zhongxing Xu*, Ming Hu, Zile Huang, Haochen Xue, Ziyang Chen, Zelin Peng, Zhiwei Yang, Sijin Zhou, Wenxue Li, Yulong Li, Wenxuan Song, Shiyan Su, Wei Feng, Jionglong Su, Minquan Lin, Yifan Peng, Xuelian Cheng, Imran Razzak†, Zongyuan Ge†
>
> **机构**：Monash University, MBZUAI, XJTLU, Northwestern Polytechnical University, Shanghai Jiaotong University, Fudan University, University of Minnesota, Cornell University
>
> **发布时间**：**CVPR 2025**
>
> 🔗 [项目页](https://mllms-farsight.github.io/)
>
> **分类标签**：`MLLM` `Hallucination` `Causal Mask` `Attention Register` `Positional Encoding` `Training-Free` `Plug-and-Play` `Image+Video`

---

## 一句话总结

提出 FarSight，通过在因果掩码的上三角区域引入**注意力寄存器**吸收分散到 outlier token 的多余注意力，并设计**渐减遮蔽率**编码绝对位置信息缓解 RoPE 长程衰减导致的视觉信息丢失，仅优化因果掩码即可 training-free 即插即用缓解图像和视频 MLLM 的初始幻觉与雪球幻觉，LLaVA-1.5 CHAIR$_S$ 降低 6.4 pp、POPE-R 提升 3.5 pp。

---

## 一、问题与动机

### 1.1 两种幻觉类型

论文将 MLLM 幻觉分为两类：

| 类型 | 含义 | 示例 |
| --- | --- | --- |
| **初始幻觉（Initial Hallucination）** | 模型缺乏充分信息，凭空描述图像中不存在的对象 | 图中无桥，但生成"一座小木桥" |
| **雪球幻觉（Snowball Hallucination）** | 模型为维持与前序幻觉的一致性，产生连锁错误描述 | 基于虚构的桥继续描述"桥上的栏杆雕有松果和树叶" |

关键发现：现有对比解码方法（VCD、OPERA 等）在减少初始幻觉方面有效，但**对雪球幻觉的抑制不足**——特别是在视频字幕任务中，雪球幻觉比例仍然很高（Fig. 2）。

### 1.2 幻觉的两个根因

论文通过分析解码过程中的注意力图，识别出两个导致幻觉的关键问题：

**(i) 注意力坍塌（Attention Collapse）**

> 模型倾向于将不成比例的高注意力分配给信息量低的 token——如视觉背景 token 和文本标点符号。这些 **outlier token** 虽然语义贡献极低，但由于 softmax 要求所有注意力权重非负且求和为 1，它们会吸走大量注意力资源，阻碍有用 token 之间的信息传播。

这与 OPERA 发现的"summary token"现象类似：随着生成文本增长，视觉和文本信息的传递逐渐衰减。

**(ii) 位置信息衰减（Positional Information Decay）**

> 由于 RoPE 的长程衰减特性，随着生成进行，文本 token 对视觉 token 的注意力逐步下降。视觉 token 的信息流在上下文交互中逐渐枯竭，导致后续生成越来越依赖语言先验而非视觉信息。

### 1.3 现有方法的不足

| 方法类别 | 代表方法 | 局限 |
| --- | --- | --- |
| 外部知识检索 | Wiki-LLaVA | 额外检索开销大 |
| 指令微调 | RLHF-V, HalluciDoctor | 需要额外训练数据和计算 |
| 对比解码 | VCD, ICD | 需额外前向传播，未分析 token 交互根因 |
| 注意力惩罚 | OPERA | 需 Beam Search，增加解码步骤 |
| 位置编码修改 | FixVPE, EDVT | 仅处理位置问题，未解决注意力坍塌 |

**FarSight 的核心优势**：仅修改因果掩码，**同时解决注意力坍塌和位置衰减两个问题**，不需要额外前向传播、外部模型或 Beam Search，计算开销几乎为零。

---

## 二、预备知识

### 2.1 MLLM 生成范式

MLLM 接收视觉 token $\mathbf{x}^v = \{x_0, \ldots, x_{N-1}\}$ 和文本 token $\mathbf{x}^t = \{x_N, \ldots, x_{M+N-1}\}$ 的拼接作为输入，自回归生成：

$$y_t \sim p_\theta(y_t | \mathbf{x}, y_{<t}) \propto \text{logit}_\theta(y_t | \mathbf{x}, y_{<t})$$

### 2.2 因果自注意力

标准因果自注意力中，注意力权重通过 softmax 和因果掩码计算：

$$O = \text{SoftMax}(\omega) \cdot V, \quad \omega = \frac{Q \cdot K^\top}{\sqrt{d_l}} + M$$

其中因果掩码 $M \in \mathbb{R}^{n \times n}$ 确保每个 token 只能关注它及之前的 token：

$$\omega_i = [\omega_{i1}, \omega_{i2}, \ldots, \omega_{ii}, 0, \ldots, 0]_n$$

### 2.3 注意力坍塌的形式化

**命题 3.1**：设输入采样自数据分布 $q(x_1, \ldots, x_N)$，在第 $l$ 层中：

$$\sum_{n=1}^{N} \sum_{j \leq i} \omega_{n,j}^l > \frac{I(x_{\leq i}; x_{n+1})}{I(x_{\leq n}; x_{n+1})} \sum_{n=1}^{N} \sum_{j \leq n} \omega_{n,j}^l + o(1)$$

其中 $I(A; B)$ 为互信息。这说明某些 token 的注意力权重远超其信息贡献——它们吸收了大量注意力但对预测无实质帮助。

### 2.4 RoPE 位置衰减

RoPE 通过旋转矩阵编码相对位置：

$$\tilde{\omega}_{ij} = \frac{q_i \cdot R_{j-i} \cdot k_j^\top}{\sqrt{d_l}}$$

$R_{j-i}$ 的长程衰减导致 $\tilde{\omega}_{ij}$ 随相对距离 $|j - i|$ 增大而减小——视觉 token 与后续生成的文本 token 距离越远，信息传递越弱。

---

## 三、核心方法

FarSight 的核心思想是**仅通过优化因果掩码实现有效的 token 信息传播**，包含两个组件：注意力寄存器（Section 4.1）和位置感知编码（Section 4.2）。

### 3.1 上三角注意力寄存器

为缓解注意力坍塌，在因果掩码的上三角区域构造**注意力寄存器** $\mathcal{P} \in \mathbb{R}^{n \times n}$：

$$\mathcal{P}_i = [\underbrace{0, 0, \ldots, 0}_{i}, \underbrace{\mathcal{P}_{i,i+1}, \mathcal{P}_{i,i+2}, \ldots, \mathcal{P}_{i,n}}_{n-i}]_n$$

每行分配 $n - i$ 个寄存器注意力分数，用于吸收多余的注意力值。寄存器分数定义为：

$$\mathcal{P}_{i,j} = -(j - i) \cdot \sigma, \quad \forall j > i$$

其中 $\sigma$ 是衰减率超参数。这确保寄存器分数符合注意力的自然衰减模式——距离越远的"未来"位置，寄存器吸收的注意力越少。

将寄存器整合到注意力矩阵中：

$$W = \omega \cdot C + \mathcal{P}, \quad C = \text{tril}(\mathbf{1}_{n \times n})$$

其中 $C$ 是下三角全 1 矩阵，确保原始注意力只保留因果部分。最终注意力计算：

$$\tilde{W} = \text{SoftMax}(\underbrace{\omega \cdot C + \mathcal{P}}_{W}) \cdot C$$

> **关键设计**：SoftMax 内部的 $\mathcal{P}$ 提供了注意力"吸收槽"——多余的注意力被分散到上三角的寄存器位置；SoftMax 外部的 $C$ 将上三角注意力概率清零，保证因果解码性质不被破坏——不会泄露未来 token 信息。

用大白话说：原来 softmax 要求注意力和为 1，outlier token 被迫吸收大量"无处安放"的注意力。现在在上三角开辟了"垃圾桶"位置，多余注意力被导入这些位置后立即清零，有效注意力（下三角部分）的和**不再等于 1**，从而允许模型将更多注意力集中在真正有用的 token 上。

### 3.2 位置感知编码

FarSight 的另一个创新是通过渐减遮蔽率在因果掩码中编码**绝对位置信息**。

设第 $i$ 行的实际注意力分数和寄存器分数满足：

$$\sum_{j \leq i} \text{SoftMax}_{j=1}^n(\omega_{i,j}) + \sum_{j > i} \text{SoftMax}_{j=1}^n(\mathcal{P}_{i,j}) = 1$$

对于相同输入 token 的序列，实际注意力分数在每行内是均匀的。随着行索引 $i$ 增大：
- 因果部分（下三角）可见 token 增多 → 实际注意力分数的指数和累积增大
- 寄存器部分（上三角）位置减少且衰减 → 寄存器分数的指数和减小

因此有效注意力呈**单调递增**趋势：

$$\sum_{j \leq i} \text{SoftMax}_{j=1}^n(\omega_{i,j}) < \sum_{j \leq i+1} \text{SoftMax}_{j=1}^n(\omega_{i+1,j})$$

即 $\tilde{W} \cdot V = \sum_{i=1}^n \beta_i v_i$，满足 $\beta_1 < \beta_2 < \cdots < \beta_n = 1$。

> **直觉**：后续位置的 token 自然聚合更多历史上下文，保持有序的信息流动。这种渐进式分配让模型在生成后期仍能对早期 token（特别是视觉 token）保持强关注，增强长程位置感知，有效对抗 RoPE 的长程衰减。

### 3.3 伪代码

> 1. 构造上三角寄存器矩阵 $\mathcal{P}$（ALiBi-style 线性衰减偏置，翻转后取上三角）
> 2. 计算 $Q, K, V$ 投影和注意力分数 $\omega = QK^\top / \sqrt{d_l}$
> 3. 融合：$\text{scores} = \omega \cdot C \cdot \sigma + \mathcal{P}$
> 4. 应用 SoftMax 后乘以 $C$ 清零上三角：$\text{scores} = \text{SoftMax}(\text{scores}) \cdot C$
> 5. 输出 $O = \text{scores} \cdot V$

### 3.4 超参数设置

- 序列长度 $\text{seq} = 256$
- 衰减率 $\sigma = \log_\alpha(\text{seq})$，其中 $\alpha = 1024$（典型最大 token 限制）
- 支持 Greedy、Sampling 和 Beam Search 三种解码策略

---

## 四、实验结果

### 4.1 图像基准评估

**综合对比**（Table 2）：

| 方法 | MMBench↑ | LLaVA$^W$↑ | MM-Vet↑ | VizWiz↑ | SQA↑ | CHAIR$_S$↓ | CHAIR$_I$↓ | POPE-R↑ | POPE-P↑ | POPE-A↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaVA-1.5 | 64.3 | 72.5 | 30.5 | 48.5 | 64.5 | 48.0 | 13.9 | 87.0 | 82.8 | 76.6 |
| +ICD | 63.1 | 69.7 | 30.4 | 46.9 | 62.8 | 47.7 | 13.6 | 87.9 | 84.0 | 80.2 |
| +VCD | 63.9 | 70.9 | 29.5 | 43.4 | 63.3 | 46.8 | 13.2 | 87.0 | 83.5 | 78.1 |
| +OPERA | 64.4 | 72.0 | 31.4 | 50.0 | 64.9 | 45.2 | 12.7 | 88.8 | 82.8 | 79.2 |
| **+FarSight** | **66.0** | **74.7** | **32.5** | **50.8** | **67.4** | **41.6** | **13.2** | **90.5** | **86.1** | **80.4** |

关键亮点：
- CHAIR$_S$：48.0 → 41.6（**-6.4 pp**），在所有对比方法中降幅最大
- POPE-R：87.0 → 90.5（**+3.5 pp**）
- 综合任务（MMBench/LLaVA$^W$/MM-Vet/VizWiz/SQA）平均提升 **+2%**，说明幻觉缓解的同时不损害通用能力

**跨模型泛化**：InstructBLIP、Video-LLaVA、Chat-UniVi 均获得一致提升，CHAIR$_S$ 分别改善 +3.8、+5.4、+3.4 pp。

### 4.2 视频基准评估

**Zero-Shot 视频问答**（Table 3）：

| 方法 | MSVD-QA Acc↑ | ActivityNet-QA Acc↑ | Cr.↑ | Cs.↑ | De.↑ | Ct.↑ | Te.↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VILA | 72.6 | 50.2 | 3.14 | 3.40 | 2.71 | 3.43 | 2.58 |
| +FarSight | **74.5** (+1.9) | **51.4** (+1.2) | 3.18 | 3.52 | 2.73 | 3.45 | 2.60 |
| Video-LLaMA2 | 70.9 | 49.9 | 3.13 | 3.23 | 2.70 | 3.42 | 2.45 |
| +FarSight | **73.8** (+2.9) | **50.4** (+0.5) | 3.26 | 3.32 | 3.21 | 3.50 | 2.47 |

FarSight 在视频任务上同样有效，MSRVTT-QA 平均准确率提升 +3%，验证了位置感知编码对长序列视频理解的价值。

### 4.3 位置编码消融

| 方法 | CHAIR$_S$↓ | CHAIR$_I$↓ | POPE-R↑ | POPE-P↑ |
| --- | --- | --- | --- | --- |
| LLaVA-1.5 (RoPE) | 48.0 | 13.9 | 87.0 | 82.8 |
| + FixVPE | 47.3 | 13.4 | 87.5 | 84.7 |
| + EDVT | 46.8 | 14.5 | 87.8 | 85.4 |
| **+ FarSight** | **41.6** (+6.4) | **13.2** (+0.7) | **90.5** (+3.5) | **86.1** (+3.3) |

FarSight 大幅超越仅修改位置编码的方法（FixVPE/EDVT），因为它同时解决了注意力坍塌问题。

### 4.4 注意力寄存器消融

不同上三角注意力值的对比（Fig. 5）：

| 上三角值策略 | 效果 |
| --- | --- |
| $-\infty$（标准因果掩码） | 长距离依赖不稳定，准确率下降 |
| $0$（零填充） | 无法有效吸收多余注意力，幻觉风险增加 |
| $10^{-3}$（固定小值） | 中等吸收效果，但不如自适应衰减 |
| **FarSight（线性衰减）** | CHAIR$_S$ 提升 +6.4%/+5.4%，显著优于其他策略 |

### 4.5 序列长度敏感性

最佳序列长度 $\text{seq} = 256$，过短（衰减过快限制远程上下文）或过长（注意力分散导致信息冗余）都会降低性能。

### 4.6 文本质量评估

GPT-4o 辅助评估 600 张 MSCOCO 图像的生成文本质量（Fig. 8）：FarSight 在 PPL、Grammar、Fluency、Naturalness 四个维度上与 Greedy 解码持平或更优，**缓解幻觉的同时不损害文本质量**。

---

## 五、局限性与未来方向

1. **超参数依赖**：$\sigma$ 和 $\text{seq}$ 需要针对不同模型和任务调优，尽管论文给出了基于最大 token 限制的简洁公式。
2. **模型规模验证**：实验主要在 7B 级别 MLLM 上进行，未验证更大规模模型（30B+）是否有相同的注意力坍塌和位置衰减问题。
3. **与 Flash Attention 的兼容性**：FarSight 修改了因果掩码的上三角区域，可能需要适配 Flash Attention 等高效注意力实现。
4. **Prefill 阶段的影响**：论文主要讨论了生成阶段的注意力优化，Prefill 阶段的 token 交互是否也存在类似问题值得探索。

---

## 六、个人思考

### 6.1 与项目内其他幻觉缓解工作的对比

| 方法 | 干预位置 | 核心机制 | 是否需训练 | 额外开销 | 图像+视频 |
| --- | --- | --- | --- | --- | --- |
| **FarSight** | 因果掩码 | 上三角注意力寄存器 + 渐减遮蔽率 | 否 | 几乎为零 | 是 |
| AGLA | 输入+输出 | GradCAM 增强图像 + logit 融合 | 否 | 低（一次 GradCAM） | 否 |
| OPERA | 输出（Beam Search） | 注意力柱状聚合惩罚 + 回溯 | 否 | 较高（多 beam） | 否 |
| VCD | 输出（logits） | 噪声图像对比解码 | 否 | 中（额外前向） | 否 |
| VisFlow | 注意力权重 | Token/Head 双层注意力操控 | 否 | 极低 | 否 |
| HIME | 模型权重 | 零空间投影编辑 MLP | 否 | 零 | 否 |

**FarSight vs OPERA**：两者都关注注意力层面的问题。OPERA 在输出端检测注意力柱状模式并回溯重分配，需要 Beam Search 增加解码步骤；FarSight 在注意力计算的因果掩码层面直接干预，通过寄存器**预防**注意力坍塌而非事后**修复**，更轻量且适用于所有解码策略（Greedy/Sampling/Beam Search）。

**FarSight vs VisFlow**：两者都在注意力层面做干预，但机制互补。VisFlow 识别具体的 sink/salient token 和功能性注意力头，做定向增强/抑制；FarSight 通过全局的因果掩码修改提供统一的注意力"溢出通道"。VisFlow 需要在 Prefill 阶段做一次头分类，FarSight 完全无需额外计算。理论上两者可以叠加。

**FarSight vs AGLA**：AGLA 在输入侧构造增强视图弥补注意力缺陷，FarSight 在注意力机制本身解决问题。AGLA 依赖外部匹配模型（BLIP-ITM），FarSight 完全自足。两者正交——AGLA 让模型"看到更好的输入"，FarSight 让模型"更好地处理注意力"。

### 6.2 注意力寄存器的巧妙之处

FarSight 的核心 insight 可以用一句话概括：**softmax 的归一化约束是注意力坍塌的根源**。当所有注意力必须加和为 1 时，低信息 token 不得不"被迫"接收注意力。通过在上三角开辟寄存器位置，FarSight 相当于给注意力机制提供了一个"泄压阀"——多余的注意力被导入后立即清零，使得有效注意力总和可以小于 1。

这个思路与 Vision Transformer 中的 Register Tokens（Darcet et al., 2024）异曲同工——后者在输入中添加可学习的寄存器 token 吸收注意力伪影。FarSight 的优势是无需训练、无需修改输入，仅通过掩码实现相同效果。

### 6.3 对视频任务的独特价值

FarSight 是本项目幻觉缓解论文中**唯一同时在图像和视频任务上验证有效**的方法。位置感知编码对视频任务尤为重要：视频 MLLM 的输入序列通常更长（多帧视觉 token），RoPE 的位置衰减问题更加严重。FarSight 的渐减遮蔽率让模型在生成长文本时仍能关注到早期帧的视觉信息，这是其他方法未覆盖的场景。

### 6.4 初始幻觉 vs 雪球幻觉的区分视角

FarSight 明确区分了初始幻觉和雪球幻觉，并展示了其方法对两类幻觉的不同缓解机制：
- **初始幻觉**：注意力寄存器释放被 outlier token 占据的注意力资源，让模型能聚焦于真正相关的视觉 token → 减少无中生有
- **雪球幻觉**：位置感知编码维持视觉 token 的长程可达性，让模型在生成后段仍能"回头看"图像 → 阻断错误累积链

这种双管齐下的思路为理解和缓解不同类型的幻觉提供了清晰的框架。

---

## 参考

- **OPERA (CVPR 2024)**：注意力过度信任惩罚 + 回溯分配——FarSight 在同一问题上提出了更轻量的掩码级解决方案
- **VCD (CVPR 2024)**：视觉对比解码——FarSight 不需要额外的扰动图像前向传播
- **ICD (2024)**：指令对比解码——FarSight 从注意力机制而非 logit 分布角度缓解幻觉
- **EDVT / Vista-LLaMA (CVPR 2024)**：仅处理位置编码的视觉 token 等距策略——FarSight 的位置感知编码效果显著优于 EDVT
- **VisFlow (2025)**：Token/Head 双层注意力操控——FarSight 与之互补，一个从掩码全局干预，一个从注意力权重局部干预
- **Register Tokens (Darcet et al., ICLR 2024)**：ViT 中的可学习寄存器 token——FarSight 将寄存器思想无训练地应用到因果解码中
