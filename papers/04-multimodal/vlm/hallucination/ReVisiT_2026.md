# ReVisiT：揭示并引用视觉 Token 内隐语义以引导 LVLM 无训练解码

> **论文信息**：Beomsik Cho, Jaehyung Kim (Yonsei University)
> **发表期刊/会议**：ACL 2026 (`arXiv:2506.09522`, `doi:10.18653/v1/2026.acl-long.262`)
> **代码仓库**：[bscho333/ReVisiT](https://github.com/bscho333/ReVisiT)

## 1. 核心思想要览

主流训练无关（Training-Free）的多模态幻觉抑制方法（如 VCD、M3ID、SID、CODE、AGLA）大多通过构造扰动视图（加噪、掩码、移除图像）并执行**双路或多路解码前向传播（Multi-Pass Forwarding）** 来抑制语言先验，导致推理延迟普遍翻倍（$\sim 2\times$ Greedy 耗时）。另一类单次前向层对比方法（如 DoLa）虽无需第二次前向，但仅对比文本解码位置在不同层的隐状态，忽略了大视觉-语言模型（LVLM）预填充阶段缓存的**视觉 Token 隐状态（Vision Token Hidden States）** 本身所蕴含的丰富文本语义，且仍会带来 $14\% \sim 31.5\%$ 的逐步投影开销。

**ReVisiT（Referencing Vision Tokens）** 从可解释性探针（Logit Lens）视角出发，首次揭示了一个关键事实：**即便在模型发生幻觉的解码步，真实物体（Ground-Truth Object）的语义依然编码在图像视觉 Token 的隐状态中；只要将词表投影限制在与当前上下文语义相容的候选子集内，视觉 Token 在语言空间中的物体语义便会清晰浮现。** 基于这一发现，ReVisiT 提出了一种极其轻量的单次前向解码算法：

1. **预填充一次性视觉 Logits 缓存（Prefill-Time Caching）**：在首个生成步（Prefill 阶段），将候选解码层集合 $\mathcal{J}$ 中所有视觉 Token 的隐状态 $h_i^j$ 通过语言模型输出头（LM Head）一次性投影到完整词表 $\mathcal{V}$，缓存在显存矩阵 $\mathbf{l}_{\texttt{vision}} \in \mathbb{R}^{|\mathcal{J}| \cdot |\mathbf{v}| \times |\mathcal{V}|}$ 中，后续自回归解码步**零额外前向传播、零重复矩阵乘法**；
2. **上下文感知自适应词表子集约束（Context-Aware Vocabulary Subset）**：在每个解码步 $t$，利用自适应合理性约束（APC）根据原始输出分布筛选出合理的局部候选词表子集 $\mathcal{V}_{\texttt{cons}}^t$，滤除全词表下干扰视觉投影的函数词与高频噪声；
3. **约束子集上的最小 JSD 视觉 Token 选择（Constrained Min-JSD Selection）**：通过简单的张量切片与掩码，在子集 $\mathcal{V}_{\texttt{cons}}^t$ 上计算原始输出分布与缓存的各视觉 Token 投影分布之间的 Jensen-Shannon 散度（JSD），选出与当前生成上下文最契合的视觉参考对 $(i^*, j^*)$；
4. **KL 正则化专家乘积对数空间融合（Product-of-Experts Log-Space Fusion）**：将原始约束分布与所选视觉 Token 约束分布在对数空间直接相加并重归一化，以仅 **$+0.6\% \sim +2.0\%$** 的微弱延迟开销（相比 VCD/SID 提速近 $2\times$），在 CHAIR、POPE、HallusionBench、VQAv2 与 MMMU 五大基准、8 个主流模型（7B 至 32B）上取得领先或极具竞争力的抗幻觉表现。

## 2. 先导实证分析：视觉语义在幻觉发生时去哪了？

作者在 Qwen2.5-VL-7B（词表规模 $|\mathcal{V}| = 151,665$）与 MS-COCO 2014 验证集随机抽取的 500 张图像（共包含 $N = 173,173$ 个动态分辨率视觉 Token）上开展了三项层递式的定量实证研究。为保证 Token 与物体类别的一一对应，作者从 MS-COCO 的 403 个同义词集合 $\mathcal{V}_{\texttt{coco}}$ 中筛选出分词后恰好为单个 Token 的子集：

$$
\mathcal{V}_{\texttt{obj}} = \bigl\{\, w \in \mathcal{V}_{\texttt{coco}} \;:\; \texttt{len}(\texttt{tokenize}(\text{`` }w\text{''})) = 1 \,\bigr\}
$$

该集合共包含 **176 个单 Token 词项**，覆盖 80 个 COCO 类别中的 **62 个类别**。在默认 Greedy 解码生成的 500 条图像描述中，CHAIR 共检出 275 处幻觉词，其中幻觉词与至少一个同图真实（GT）物体同义词均为单 Token 的对齐位置共有 $n = 190$ 个幻觉解码步。

### 2.1 观察一：幻觉发生时，真实物体仍在高概率候选前列

在 $n = 190$ 个发生对象幻觉的解码步 $t$ 上，考察最后一层输出分布 $p_{\theta}(h_{T+t-1}^L, \mathcal{V})$ 的 Top-$k$ 候选中是否至少包含一个真实物体同义词（At-Least-One GT Recall@$k$）：

| Top-$k$ 阈值 | 占完整词表比例（$k / 151,665$） | 幻觉步真实物体召回率 At-Least-One Recall@$k$（95% Wilson CI） |
| :--- | :---: | :---: |
| **Top-50** | $0.033\%$ | **63.2%** $[56.0\%, 70.0\%]$ |
| **Top-100** | $0.066\%$ | **73.7%** $[66.9\%, 79.6\%]$ |
| **Top-500** | $0.330\%$ | **95.8%** $[91.6\%, 98.1\%]$ |

这表明：**绝大多数幻觉并非源于模型“完全没看见真实物体”，而是真实物体的排名在局部竞争中略逊于受语言先验驱动的幻觉词。**

### 2.2 观察二：视觉 Token 持续向解码分布注入正向接地增益

记解码步 $t$ 在物体词表 $\mathcal{V}_{\texttt{obj}}$ 上的真实同义词集为 $\mathcal{G}_t$，非真实物体词集为 $\mathcal{H}_t = \mathcal{V}_{\texttt{obj}} \setminus \mathcal{G}_t$，定义每步最大真实物体概率 $m_t^{\mathrm{GT}} = \max_{w \in \mathcal{G}_t} p_{\theta}(w \mid h_{T+t-1}^L, \mathcal{V}_{\texttt{obj}})$ 与最大幻觉物体概率 $m_t^{\mathrm{Hal}} = \max_{w \in \mathcal{H}_t} p_{\theta}(w \mid h_{T+t-1}^L, \mathcal{V}_{\texttt{obj}})$。对比保留图像（$\mathrm{with}$）与移除全部视觉 Token 并用 Teacher Forcing 对齐位置（$\mathrm{w/o}$）下的差值变化 $\Delta_t = (m_t^{\mathrm{GT}} - m_t^{\mathrm{Hal}})_{\mathrm{with}} - (m_t^{\mathrm{GT}} - m_t^{\mathrm{Hal}})_{\mathrm{w/o}}$：
- 在全部 $88,706$ 个解码位置上，Wilcoxon 符号秩检验显示 $\mathrm{median}(\Delta) = +0.715$（$p_{\text{Holm}} < 10^{-300}$），说明视觉 Token 在全局范围内显著拉升了真实物体相对于幻觉词的概率优势；
- 但在发生幻觉的 $n = 190$ 个局部步骤上，$\mathrm{median}(d^{\mathrm{with}}) = -8.281$ 且 $\mathrm{median}(\Delta) = -0.180$（$p_{\text{Holm}} = 0.95$），即隐式注意力传递在这些步骤上未能扭转 $\mathrm{Hal} > \mathrm{GT}$ 的逆序，亟需在解码端显式调用视觉 Token 语义予以矫正。

### 2.3 观察三：子集约束能将视觉 Token 的显式语义召回率提升 20 倍

将 500 张图像全部 $N = 173,173$ 个视觉 Token 的最后一层隐状态 $h_i^L$ 直接通过 LM Head $\phi(\cdot)$ 投影到词表空间（Logit Lens）：

| 评测范围 | 全词表投影 $\mathcal{V}$（$151,665$ 词） | 语义相关物体子集投影 $\mathcal{V}_{\texttt{obj}}$（$176$ 词） |
| :--- | :---: | :---: |
| **Top-1 GT Recall** | $2.03\%$ $[1.96, 2.10]$ | **40.44%** $[40.21, 40.67]$（**提升 $\approx 20\times$**） |
| **Top-5 GT Recall** | $7.87\%$ $[7.74, 8.00]$ | **69.56%** $[69.34, 69.78]$ |
| **Top-10 GT Recall** | $12.38\%$ $[12.23, 12.54]$ | **79.78%** $[79.59, 79.97]$ |
| **Top-30 GT Recall** | $21.72\%$ $[21.53, 21.91]$ | **89.24%** $[89.09, 89.39]$ |

**核心洞见**：在完整词表 $\mathcal{V}$ 上直接做 Logit Lens 投影时，视觉 Token 的概率质量被大量通用功能词稀释（Top-1 仅 $2.03\%$ 命中 GT 物体）；但一旦将投影空间收窄到语义连贯的候选子集，同一个视觉 Token 隐状态的 Top-1 真实物体命中率飙升至 **40.44%**，Top-30 达到 **89.24%**。

## 3. 方法详解：ReVisiT 三步解码机制

设输入序列由 $|\mathbf{v}|$ 个视觉 Token $\mathbf{v} = \{v_0, \dots, v_{|\mathbf{v}|-1}\}$ 与 $|\mathbf{x}|$ 个文本提示 Token $\mathbf{x} = \{x_0, \dots, x_{|\mathbf{x}|-1}\}$ 拼接而成，总预填充长度为 $T = |\mathbf{v}| + |\mathbf{x}|$。模型共有 $L$ 个解码层，在生成步 $t$，最后一层隐状态 $h_{T+t-1}^L$ 经 LM Head $\phi(\cdot)$（权重矩阵 $W_{\texttt{proj}}$）映射为原始词表分布：

$$
p_{\theta}(h_{T+t-1}^L, \mathcal{V}) = \mathrm{softmax}_{\mathcal{V}}\!\left(\phi(h_{T+t-1}^L)\right)
$$

### 3.1 预填充阶段的视觉 Logits 一次性缓存

在首个生成步（$t = 0$，即完成图像与 Prompt 的 Prefill 前向传播后），给定候选解码层集合 $\mathcal{J} \subseteq \{1, \dots, L\}$：
- 对每一候选层 $j \in \mathcal{J}$ 与每一个视觉 Token 下标 $i \in \{0, \dots, |\mathbf{v}| - 1\}$（注意**仅遍历视觉 Token**，不遍历文本 Token），计算其投影 Logits 向量：

$$
l_i^j = W_{\texttt{proj}}^\top h_i^j \in \mathbb{R}^{|\mathcal{V}|}
$$

- 将所有候选层与视觉 Token 的投影结果拼装并缓存为矩阵：

$$
\mathbf{l}_{\texttt{vision}} \in \mathbb{R}^{(|\mathcal{J}| \cdot |\mathbf{v}|) \times |\mathcal{V}|}
$$

由于视觉 Token 位于序列最前端，在因果掩码（Causal Attention Mask）下，其隐状态 $h_i^j$ 在后续自回归解码步 $t \ge 1$ 中**完全保持不变**。因此 $\mathbf{l}_{\texttt{vision}}$ 在整条序列生成过程中只需计算一次，后续每步仅需按索引切片。

### 3.2 步骤一：上下文感知的自适应词表子集构建

在每个解码步 $t$，为消除全词表投影中的无关语法词干扰，受自适应合理性约束（APC）启发，根据当前原始输出分布动态截取合理候选词表子集 $\mathcal{V}_{\texttt{cons}}^t \subseteq \mathcal{V}$：

$$
\mathcal{V}_{\texttt{cons}}^t = \left\{ w \in \mathcal{V} \;:\; p_{\theta}(w \mid h_{T+t-1}^L, \mathcal{V}) \ge \alpha \cdot \max_{w' \in \mathcal{V}} p_{\theta}(w' \mid h_{T+t-1}^L, \mathcal{V}) \right\}
$$

其中 $\alpha \in (0, 1)$ 控制候选子集的松紧度（默认在验证集上从 $\{10^{-1}, 10^{-2}, \dots, 10^{-6}\}$ 对数网格中选取；无需按任务调参时默认取 $\alpha = 10^{-5}$）。随后，将原始输出限制在子集 $\mathcal{V}_{\texttt{cons}}^t$ 上重新做局部 Softmax 归一化，得到：

$$
p_{\text{base}}^{\texttt{cons}}(w) = p_{\theta}(w \mid h_{T+t-1}^L, \mathcal{V}_{\texttt{cons}}^t) = \mathrm{softmax}_{\mathcal{V}_{\texttt{cons}}^t}\!\left(W_{\texttt{proj}}^\top h_{T+t-1}^L\right)
$$

### 3.3 步骤二：约束空间内的投影与最小 JSD 视觉 Token 选择

对缓存矩阵 $\mathbf{l}_{\texttt{vision}}$ 中的每个候选层 $j \in \mathcal{J}$ 和视觉 Token $i \in \{0, \dots, |\mathbf{v}| - 1\}$，仅截取 $\mathcal{V}_{\texttt{cons}}^t$ 对应的列并执行局部 Softmax 归一化，得到视觉 Token 在当前上下文候选空间内的投影分布：

$$
p_{\theta}(h_i^j, \mathcal{V}_{\texttt{cons}}^t) = \mathrm{softmax}_{\mathcal{V}_{\texttt{cons}}^t}\!\left(\mathbf{l}_{\texttt{vision}}[j, i]\right)
$$

接着，计算每个候选视觉 Token 分布与当前原始约束输出分布之间的 Jensen-Shannon 散度（JSD），并联合选取散度**最小**（即语义最契合当前生成语境）的视觉 Token 与解码层对 $(i^*, j^*)$：

$$
(i^*, j^*) = \arg\min_{i \in \{0, \dots, |\mathbf{v}|-1\},\; j \in \mathcal{J}} \mathrm{JSD}\!\left(p_{\theta}(h_{T+t-1}^L, \mathcal{V}_{\texttt{cons}}^t) \;\parallel\; p_{\theta}(h_i^j, \mathcal{V}_{\texttt{cons}}^t)\right)
$$

> **与 DoLa 的本质区别**：DoLa 在同一文本位置对比不同层，选取 JSD **最大**（$\arg\max$）的早熟层做减法（Contrastive Subtraction）以剥离浅层语法偏置；而 ReVisiT 是在 $|\mathcal{J}| \times |\mathbf{v}|$ 个视觉 Token 候选中搜寻与当前解码意图语义同质（处于同一候选子集流形内）的视觉锚点，因此必须取 **$\arg\min \mathrm{JSD}$** 并做正向融合。消融实验证明，若改为 $\arg\max \mathrm{JSD}$，CHAIR 上的物体 F1 会从 $81.16$ 崩溃至 $0.67$。

### 3.4 步骤三：基于 KL 正则化专家乘积（PoE）的分布修正

记选中的最优视觉参考分布为 $p_{\text{ref}}^{\texttt{cons}}(y_t) = p_{\theta}(y_t \mid h_{i^*}^{j^*}, \mathcal{V}_{\texttt{cons}}^t)$。从理论上，寻找最优修正分布 $q^* \in \Delta(\mathcal{V}_{\texttt{cons}}^t)$ 可以建模为一个最大化视觉参考似然并受原始输出分布 KL 散度约束的变分优化问题：

$$
q^* = \arg\max_{q \in \Delta(\mathcal{V}_{\texttt{cons}}^t)} \; \mathbb{E}_{y \sim q}\!\left[\log p_{\text{ref}}^{\texttt{cons}}(y)\right] - \gamma \, \mathrm{KL}\!\left(q \;\parallel\; p_{\text{base}}^{\texttt{cons}}\right)
$$

该优化问题具有显式闭式解（Product-of-Experts, PoE）：

$$
q^*(y_t) \propto p_{\text{base}}^{\texttt{cons}}(y_t) \cdot \left(p_{\text{ref}}^{\texttt{cons}}(y_t)\right)^{1/\gamma}
$$

取默认等权重设置 $\gamma = 1$，在对数空间表现为简洁的无参相加：

$$
l_{\texttt{ReVisiT}}(y_t) = \log p_{\theta}(y_t \mid h_{T+t-1}^L, \mathcal{V}_{\texttt{cons}}^t) + \log p_{\theta}(y_t \mid h_{i^*}^{j^*}, \mathcal{V}_{\texttt{cons}}^t)
$$

最终输出分布在子集 $\mathcal{V}_{\texttt{cons}}^t$ 内通过 Softmax 归一化，并将子集外的词项概率硬置为零：

$$
p_{\texttt{fin}}(y_t) = \begin{cases} \mathrm{softmax}_{\mathcal{V}_{\texttt{cons}}^t}\!\left(l_{\texttt{ReVisiT}}(y_t)\right), & y_t \in \mathcal{V}_{\texttt{cons}}^t, \\ 0, & \text{otherwise}. \end{cases}
$$

在确定性解码（Greedy, $\text{temperature}=0$）下，直接取 $y_t = \arg\max_{y \in \mathcal{V}_{\texttt{cons}}^t} l_{\texttt{ReVisiT}}(y)$。

## 4. 实验评估与核心结果

### 4.1 评测基准与架构设置

- **评测模型（涵盖 3 大家族 8 个模型规模）**：主实验评测 `LLaVA-1.5-7B`、`Qwen2.5-VL-7B`、`InternVL3-8B`；扩展实验涵盖 `LLaVA-1.5-13B`、`InternVL2.5-8B/26B`、`InternVL3-14B`、`Qwen2.5-VL-32B`。
- **候选层池 $\mathcal{J}$ 配置**：对 `LLaVA-1.5` 与 `InternVL3` 默认仅取最后一层（`last`，即 $\mathcal{J} = \{L\}$）；对 `Qwen2.5-VL` 默认取所有偶数解码层（`all`）。
- **对比基线**：Greedy、DoLa、VCD、M3ID、CODE、SID（全部统一在 $\text{temperature}=0$、最大生成长度 512 tokens 下评测）。

### 4.2 CHAIR 开放式图像描述抗幻觉结果（500 张 COCO Val 图像）

| 方法 | LLaVA-1.5-7B $\text{CHAIR}_S \downarrow$ | LLaVA-1.5-7B $\text{CHAIR}_I \downarrow$ | LLaVA-1.5-7B $\text{F1} \uparrow$ | Qwen2.5-VL-7B $\text{CHAIR}_S \downarrow$ | Qwen2.5-VL-7B $\text{CHAIR}_I \downarrow$ | Qwen2.5-VL-7B $\text{F1} \uparrow$ | InternVL3-8B $\text{CHAIR}_S \downarrow$ | InternVL3-8B $\text{CHAIR}_I \downarrow$ | InternVL3-8B $\text{F1} \uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy** | 53.8 | 14.66 | 82.33 | 35.2 | 8.43 | 79.85 | 29.4 | 7.91 | 81.11 |
| **DoLa** | 53.6 | 14.45 | 82.27 | 31.0 | 7.63 | 79.33 | 28.8 | 8.49 | 81.37 |
| **VCD** | 52.8 | 15.61 | 81.84 | 35.0 | 9.07 | 80.10 | 30.8 | 8.43 | 81.56 |
| **M3ID** | 57.0 | 16.57 | 82.06 | 33.8 | 9.80 | 78.80 | 29.2 | 8.02 | 81.81 |
| **CODE** | **45.2** | **13.13** | 81.48 | **17.0** | 9.35 | 48.50 | 29.0 | 8.25 | 81.32 |
| **SID** | 50.6 | 13.58 | **83.34** | 37.6 | 8.84 | 80.40 | 33.0 | 9.58 | 81.82 |
| **ReVisiT (Ours)** | 50.6 | 13.43 | 83.17 | 29.8 | **7.04** | **81.16** | **27.6** | **7.27** | **82.29** |

> 注：CODE 在 Qwen2.5-VL-7B 上虽然 $\text{CHAIR}_S$ 降至 $17.0$，但物体 F1 暴跌至 $48.50$（因过度缩减有效生成内容）；而 ReVisiT 在三个骨干上均实现 $\text{CHAIR}_S / \text{CHAIR}_I$ 显著下降与 F1 稳步提升的双赢。在大尺寸模型（LLaVA-1.5-13B、InternVL3-14B、InternVL2.5-26B、Qwen2.5-VL-32B）上，ReVisiT 同样一致降低 $\text{CHAIR}_I$ 并提升 F1。

### 4.3 POPE、HallusionBench、VQAv2 与 MMMU 综合评测

1. **POPE（MS-COCO / A-OKVQA / GQA $\times$ 3 种采样共 9 个子集平均）**：
   - `LLaVA-1.5-7B`：Accuracy 从 Greedy 的 $79.47\%$ 提升至 **81.80%**，Precision 从 $74.48\%$ 提升至 **78.03%**，F1 达 **83.45%** （全场第一，超越 SID 的 $82.91\%$ 与 VCD 的 $80.97\%$）；
   - `InternVL3-8B`：Accuracy 达 **89.21%**，F1 达 **89.44%** （全场最高）。
2. **HallusionBench（Visual Dependent 子集 591 题，GPT-4 评判）**：
   - 在 `LLaVA-1.5-7B` 上，问题对准确率（qAcc）从 Greedy 的 $11.55\%$ 大幅跃升至 **20.22%**，图像准确率（fAcc）从 $22.61\%$ 升至 **26.52%**，总准确率（aAcc）从 $48.22\%$ 升至 **51.44%**；在 `Qwen2.5-VL-7B`（aAcc **61.76%**）与 `InternVL3-8B`（qAcc **33.21%**、fAcc **39.13%**）上同样全面领先。
3. **通用视觉问答与多学科推理（VQAv2 & MMMU）**：
   - 许多对比解码基线在通用问答上出现明显退化（如 VCD 在 InternVL3-8B 上 VQAv2 从 $68.73\%$ 跌至 $53.60\%$，SID 跌至 $55.27\%$）；
   - ReVisiT 在 `LLaVA-1.5-7B`（VQAv2 **70.40%**，MMMU **35.13%**）、`Qwen2.5-VL-7B`（VQAv2 **67.60%**，MMMU **51.18%**）和 `InternVL3-8B`（MMMU **54.14%**）上均刷新最高分，证明其不会牺牲通用多模态理解与复杂推理能力。

### 4.4 推理延迟与消融实验（NVIDIA RTX A6000 48GB）

| 方法 | 前向传播次数 | LLaVA-1.5-7B 延迟（ms/token） | 相对开销 | Qwen2.5-VL-7B 延迟（ms/token） | 相对开销 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greedy** | $1\times$ | $26.0 \pm 0.2$ | $1.00\times$ | $72.3 \pm 1.5$ | $1.00\times$ |
| **DoLa** | $1\times$（逐步中间层投影） | $34.2 \pm 0.4$ | $+31.5\%$ | $82.6 \pm 0.4$ | $+14.2\%$ |
| **VCD** | $2\times$ | $51.7 \pm 0.4$ | $+98.8\%$ | $143.4 \pm 1.1$ | $+98.3\%$ |
| **M3ID** | $2\times$ | $49.5 \pm 0.2$ | $+90.4\%$ | $143.3 \pm 13.2$ | $+98.2\%$ |
| **SID** | $2\times$ | $51.9 \pm 0.4$ | $+99.6\%$ | $146.5 \pm 1.3$ | $+102.6\%$ |
| **CODE** | $2\times$ | $81.0 \pm 6.3$ | $+211.5\%$ | $176.2 \pm 45.1$ | $+143.7\%$ |
| **ReVisiT (Ours)** | **$1\times$（Prefill 一次性投影）** | **$26.5 \pm 0.2$** | **$+1.92\%$** | **$72.7 \pm 0.4$** | **$+0.55\%$** |

**关键消融结论（Qwen2.5-VL-7B on CHAIR）**：
- **去掉词表子集约束（Full Vocabulary + min-JSD）**：模型迅速陷入单 Token 无限重复退化，物体 F1 从 **81.16** 暴跌至 **1.52**，证实上下文子集约束 $\mathcal{V}_{\texttt{cons}}^t$ 是防止全词表噪声主导视觉投影的承重墙；
- **选择准则对比**：`min-JSD`（F1 $81.16$）显著优于 `min-KL`（F1 $79.09$）与 `Cosine Similarity`（F1 $80.16$），而 `Random`（F1 $17.63$）与 `max-JSD`（F1 $0.67$）完全失效。

## 5. 复现审计与关键实现细节

1. **缓存张量 $\mathbf{l}_{\texttt{vision}}$ 的显存管理**：
   - $\mathbf{l}_{\texttt{vision}}$ 的维度为 $(|\mathcal{J}| \cdot |\mathbf{v}|) \times |\mathcal{V}|$。在 `LLaVA-1.5-7B` 上（$|\mathcal{J}| = 1$ 层，$|\mathbf{v}| = 576$，$|\mathcal{V}| = 32,000$），FP16 仅占 $\sim 36.8\text{ MB}$；
   - 但在 `Qwen2.5-VL-7B` 上（`all` 偶数层 $|\mathcal{J}| = 14$，动态分辨率视觉 Token $|\mathbf{v}|$ 可达 $1,300+$，词表 $|\mathcal{V}| = 151,665$），若以 FP32 缓存将占用 $> 11\text{ GB}$ 显存！工程实现中必须使用 `bfloat16`/`float16` 存储并在每条样本生成结束后立即释放缓存。
2. **伪代码下标修正**：论文 Algorithm 1 中写有 `Given {h_i^j} for i = 0, ..., T-1`，但紧接着的内层循环明确为 `for i = 0 to |v| - 1`。由于视觉 Token 拼接在文本 Token 之前（下标 $0 \dots |\mathbf{v}| - 1$），切片时务必只取视觉 Token 区间，切勿混入文本提示 Token。
3. **阈值 $\alpha$ 的任务特性**：开放式长文本描述（如 CHAIR）适合较宽松的候选子集（$\alpha = 10^{-5}$），以保留丰富的实词候选；而短答案视觉问答（如 VQAv2）更适合较紧的候选子集（$\alpha \in [10^{-2}, 10^{-1}]$），避免引入无关干扰项。
