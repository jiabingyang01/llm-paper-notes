# VisFlow：双层注意力干预缓解视觉幻觉

> 论文：*Not All Tokens and Heads Are Equally Important: Dual-Level Attention Intervention for Hallucination Mitigation*
>
> 作者：Lexiang Tang, Xianwei Zhuang, Bang Yang, Zhiyuan Hu, Hongxiang Li, Lu Ma, Jinghan Ru, Yuexian Zou*
>
> 机构：Peking University
>
> 发布时间：2025年6月（**AAAI 2026**）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.12609)
>
> 分类标签：`视觉幻觉` `注意力干预` `Token 级别` `Head 级别` `Training-Free`

---

## 一句话总结

通过系统分析 LVLM 解码器中的注意力分布，识别出三种病理性注意力模式（弱视觉 grounding、语言先验主导、系统提示冗余），提出双层注意力干预框架 VisFlow：Token 级别（TAI）增强视觉显著 token 注意力、Head 级别（HAI）抑制系统提示头和文本跟随头的过度注意力，无需训练即插即用，LLaVA-1.5 上 CHAIR$_S$ 从 6.9 降至 3.8，POPE Adversarial F1 从 73.54 提升至 84.35。

---

## 一、问题与动机

### 1.1 LVLM 中的三种病理性注意力模式

论文从注意力流（information flow）的视角系统分析了视觉幻觉的成因，发现了三种关键模式：

| 模式 | 含义 | 后果 |
| --- | --- | --- |
| **弱视觉 grounding** | 对视觉 token 的注意力不足且分配错误，过度聚焦于视觉上无信息量的 token 或图像尾部区域 | 生成内容缺乏视觉锚定 |
| **语言先验主导** | 对最近生成的 response token 过度集中注意力 | 自回归模式强化，多模态对齐受损 |
| **系统提示冗余** | 大量注意力头对 system prompt token 分配异常高的权重 | 阻碍视觉、指令和响应内容的整合 |

### 1.2 RoPE 引起的视觉注意力偏差

由于 RoPE 位置编码的长距离衰减特性，**靠近文本 token 的视觉 token（即图像尾部区域）获得不成比例的高注意力**，而真正包含语义信息的视觉区域反而被忽视。论文通过实验验证：在无图像输入的情况下，LLaVA-1.5 生成的描述与有图像输入时几乎完全相同（包含相同的幻觉），说明模型严重依赖语言先验而非视觉信息。

### 1.3 现有方法的不足

| 方法 | 思路 | 局限 |
| --- | --- | --- |
| VCD | 视觉对比解码（噪声图像对比） | 需要额外前向传播，语义漂移风险 |
| OPERA | 注意力过度信任惩罚 + 回溯分配 | 需 beam search，增加解码步数 |
| DoLa | 层间对比解码 | 非针对视觉幻觉设计 |
| HALC | 自适应 FOV 对比解码 | 依赖外部检测器，推理开销大 |

**VisFlow 的核心优势**：不重排输出、不增加解码步骤、不需要外部工具，直接操作注意力权重，计算开销几乎可忽略。

---

## 二、预备知识：信息流分析

### 2.1 Saliency 度量

为分析 token 间的信息流，论文采用 Taylor 展开计算每个注意力矩阵元素的显著性分数：

$$I_l = \left| \sum_h \mathbf{A}_{h,l} \odot \frac{\partial \mathcal{L}(x)}{\partial \mathbf{A}_{h,l}} \right|$$

其中 $\mathbf{A}_{h,l}$ 是第 $l$ 层第 $h$ 个头的注意力矩阵，$\mathcal{L}(x)$ 是任务损失。$I_l(i,j)$ 反映了 token $j$ 对 token $i$ 在第 $l$ 层的贡献。

### 2.2 方向性信息流度量

定义 token 组之间的方向性信息流：

$$S_{ab} = \frac{\sum_{(i,j) \in C_{ab}} I(i,j)}{|C_{ab}|}, \quad C_{ab} = \{(i,j) : i \in \mathcal{A},\ j \in \mathcal{B}\}$$

其中 $\mathcal{A}$、$\mathcal{B}$ 是不同的 token 类型集合（系统、视觉、文本）。通过这个度量可以定量分析不同模态 token 之间的信息传递强度。

论文的逐层分析表明：**系统→视觉的信息流在各层中异常突出**，说明系统提示 token 占据了过多的注意力资源，挤压了视觉信息的有效传递。

---

## 三、核心方法

VisFlow 包含两个互补的干预机制：Token 级别注意力干预（TAI）和 Head 级别注意力干预（HAI）。

### 3.1 Token 级别注意力干预（TAI）

#### 3.1.1 Visual Sink 与 Salient Token 识别

对每个视觉 token $j$，计算其**接收分数**——来自其他视觉 token 的注意力总量：

$$R(j) = \frac{1}{H} \sum_{h=1}^{H} \sum_{i \in \mathcal{I}_{\text{vis}} \setminus \{j\}} \mathbf{A}_\ell^{(h)}[i, j]$$

基于接收分数的相对阈值将视觉 token 分为两类：

$$\mathcal{I}_{\text{thres}}(\tau) = \left\{ j \in \mathcal{I}_{\text{vis}} \;\middle|\; R(j) > \tau \cdot \max_{k \in \mathcal{I}_{\text{vis}}} R(k) \right\}$$

| Token 类型 | 阈值 $\tau$ | 含义 |
| --- | --- | --- |
| **Visual Sink Token** | $\tau = \frac{1}{2}$ | 吸收大量注意力但缺乏语义贡献的 token，类似 LLM 中的 attention sink 现象 |
| **Visual Salient Token** | $\tau = \frac{1}{20}$ | 与有意义的视觉区域对齐、对 grounding 至关重要的 token |

用大白话说：Visual Sink Token 是"注意力黑洞"——其他视觉 token 都在看它，但它本身不包含有用的视觉信息（通常是图像中的纯色背景或边角区域）。Visual Salient Token 则是真正承载关键视觉语义的 token。

#### 3.1.2 增强显著 Token 注意力

对指令 token $i \in \mathcal{I}_{\text{ins}}$ 到视觉 token $j \in \mathcal{I}_{\text{vis}}$ 的注意力权重进行调制：

$$\mathbf{A}_{i,j}^{\ell,h} = \begin{cases} k \cdot \mathbf{A}_{i,j}^{\ell,h}, & \text{if } j \in \mathcal{I}_{\text{salient}}^\ell \\ \delta \cdot \mathbf{A}_{i,j}^{\ell,h}, & \text{if } j \in \mathcal{I}_{\text{sink}}^\ell \\ \mathbf{A}_{i,j}^{\ell,h}, & \text{otherwise} \end{cases}$$

其中 $k > 1$ 放大对显著 token 的注意力，$\delta < 1$ 抑制对 sink token 的注意力。调整后重新归一化：

$$\mathbf{A}_{i,j}^{l,h} = \frac{\mathbf{A}_{i,j}^{l,h}}{\sum_j \mathbf{A}_{i,j}^{l,h}}, \quad \text{if } i \in \mathcal{I}_{\text{txt}}$$

**设计直觉**：通过"加强信号、压制噪声"的方式，纠正 RoPE 导致的位置偏差——让模型真正关注图像中有语义的区域，而非机械地关注位置上靠近文本的视觉 token。

### 3.2 Head 级别注意力干预（HAI）

#### 3.2.1 三类注意力头的识别

论文将注意力头分为三种功能类型：

**Visual-sensitive heads**（视觉敏感头）：

$$\mathcal{H}_{\text{vis}}^\ell = \left\{ h \;\middle|\; A_{\text{vis}}^{\ell,h} > \mu + \lambda_{\text{vis}} \cdot \sigma \right\}$$

其中 $A_{\text{vis}}^{\ell,h} = \sum_{i \in \mathcal{I}_{\text{txt}}} \sum_{j \in \mathcal{I}_{\text{vis}}} \mathbf{A}^{\ell,h}[i,j]$，$\mu$ 和 $\sigma$ 是该层所有头的均值和标准差。

**Text-following heads**（文本跟随头）：

$$\mathcal{H}_{\text{txt}}^\ell = \left\{ h \;\middle|\; A_{\text{txt}}^{\ell,h} > \lambda_{\text{txt}} \right\}$$

其中 $A_{\text{txt}}^{\ell,h} = \sum_{i \in \mathcal{I}_{\text{txt}}} \sum_{j \in \mathcal{I}_{\text{txt}}} \mathbf{A}^{\ell,h}[i,j]$。

**System-prompt dominant heads**（系统提示主导头）：

$$\mathcal{H}_{\text{sys}}^\ell = \left\{ h \;\middle|\; A_{\text{sys}}^{\ell,h} > \lambda_{\text{sys}} \right\}$$

其中 $A_{\text{sys}}^{\ell,h} = \sum_{i \in \mathcal{I}_{\text{txt}}} \sum_{j \in \mathcal{I}_{\text{sys}}} \mathbf{A}^{\ell,h}[i,j]$。

论文通过 zeroing out 实验验证了这三类头的功能角色：
- 屏蔽 visual heads → CHAIR 大幅上升（22.1），确认其对视觉 grounding 至关重要
- 屏蔽 system heads → CHAIR 微降（6.7 vs 6.9），说明存在冗余
- 屏蔽 text heads → CHAIR 略升（7.2），但影响有限

#### 3.2.2 统一抑制机制

对 system heads 和 text heads 的注意力权重进行衰减：

$$\tilde{\mathbf{A}}^{l,h}(i,j) = \begin{cases} (1 - \alpha_{\text{txt}}) \cdot \mathbf{A}^{l,h}(i,j), & \text{if } h \in \mathcal{H}_{\text{text}},\ j \in \mathcal{I}_{\text{text}} \\ (1 - \alpha_{\text{sys}}) \cdot \mathbf{A}^{l,h}(i,j), & \text{if } h \in \mathcal{H}_{\text{sys}},\ j \in \mathcal{I}_{\text{sys}} \\ \mathbf{A}^{l,h}(i,j), & \text{otherwise} \end{cases}$$

然后重新归一化：

$$\hat{\mathbf{A}}^{l,h}(i,:) = \frac{\tilde{\mathbf{A}}^{l,h}(i,:)}{\sum_j \tilde{\mathbf{A}}^{l,h}(i,j)}$$

**关键设计**：
- System heads 的抑制应用于**所有层**（因为系统提示冗余贯穿整个模型）
- Text heads 的抑制仅应用于**浅层（0-7 层）**（因为浅层是跨模态融合的关键阶段，过度关注文本会阻碍视觉信息整合）

### 3.3 效率设计

注意力头类型的识别**仅在 prefill 阶段进行一次**，后续解码阶段直接复用。TAI 从第 2 层开始应用（第 1 层保留原始注意力以维持基本 token 交互）。整个过程无需额外前向传播或外部模型调用。

---

## 四、实验结果

### 4.1 CHAIR 评估（MSCOCO）

| 方法 | LLaVA-1.5 CHAIR$_i$↓ | CHAIR$_s$↓ | Recall↑ | MiniGPT-4 CHAIR$_i$↓ | CHAIR$_s$↓ | Recall↑ | mPLUG-Owl2 CHAIR$_i$↓ | CHAIR$_s$↓ | Recall↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 20.1 | 6.9 | 59.0 | 25.0 | 9.2 | 58.7 | 23.0 | 9.6 | 54.4 |
| Beam Search | 20.0 | 6.5 | 57.0 | 24.0 | 9.2 | 56.7 | 18.0 | 6.4 | 53.0 |
| OPERA | 17.0 | 6.3 | 56.7 | 20.0 | 8.2 | 58.1 | 16.0 | 5.8 | 54.0 |
| VCD | 20.0 | 6.9 | 57.0 | 23.0 | 8.9 | 56.4 | 18.0 | 6.4 | 53.0 |
| DoLa | 19.0 | 6.5 | 57.0 | 19.0 | 8.1 | 56.3 | 18.0 | 6.1 | 53.0 |
| **VisFlow** | **15.0** | **3.8** | **63.1** | **18.0** | **7.8** | 56.3 | **16.0** | **4.9** | **53.5** |

在 LLaVA-1.5 上，VisFlow 将 CHAIR$_s$ 从最佳基线 6.3（OPERA）降低至 **3.8（降低 40%）**，同时 Recall 从 59.0 提升至 **63.1**。这是非常罕见的——降低幻觉的同时还提高了物体召回率。

### 4.2 POPE 评估

| 方法 | LLaVA-1.5 Random↑ | Popular↑ | Adversarial↑ | MiniGPT-4 Random↑ | Popular↑ | Adversarial↑ | mPLUG-Owl2 Random↑ | Popular↑ | Adversarial↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 81.54 | 76.53 | 73.54 | 77.56 | 67.50 | 69.11 | 83.90 | 77.30 | 74.82 |
| Beam Search | 82.64 | 79.34 | 78.15 | 78.54 | 70.20 | 71.62 | 87.33 | 81.42 | 78.95 |
| OPERA | 79.50 | 76.63 | 75.88 | 78.35 | 69.65 | 71.42 | 87.03 | 80.29 | 77.92 |
| VCD | 82.51 | 79.33 | 78.17 | 78.61 | 69.95 | 71.62 | 87.36 | 81.42 | 78.95 |
| DoLa | 82.81 | 79.47 | 78.36 | 80.23 | 73.00 | 73.23 | 87.90 | 81.53 | 79.18 |
| **VisFlow** | **89.55** | **87.09** | **84.35** | **80.86** | **73.61** | **73.94** | **88.72** | **82.19** | **80.17** |

在 LLaVA-1.5 上提升尤为显著：Adversarial 子集 F1 从 78.36（DoLa）提升至 **84.35（+6.0 pp）**，Random 子集从 82.81 提升至 **89.55（+6.7 pp）**。

### 4.3 消融实验

| 设置 | CHAIR$_s$↓ | CHAIR$_i$↓ | Recall↑ |
| --- | --- | --- | --- |
| Greedy（基线） | 20.1 | 6.9 | 59.0 |
| w/o TAI（仅 HAI） | 16.0 | 4.8 | 56.7 |
| w/o HAI（仅 TAI） | 16.0 | 5.3 | 58.4 |
| w/o HAI for Txt. Heads | 18.0 | 6.4 | 61.1 |
| w/o HAI for Sys. Heads | 12.0 | 4.0 | 60.1 |
| **完整 VisFlow** | **15.0** | **3.8** | **63.1** |

几个关键发现：
- TAI 和 HAI 都独立有效，但**结合后效果最佳**（CHAIR$_i$：4.8/5.3 → 3.8）
- 去掉 System Heads 的 HAI 反而 CHAIR$_s$ 最低（12.0），但 Recall 下降——说明过度抑制系统头会导致模型忽略指令
- 完整 VisFlow 在幻觉和 Recall 之间取得最佳平衡

### 4.4 超参数敏感性

对显著 token 的增强因子 $k$：
- CHAIR$_i$ 随 $k$ 呈 **U 形趋势**
- $k = 20$ 时幻觉最低
- 过度放大（$k = 24$）或不放大（$k = 1$）都效果不佳

### 4.5 推理效率

| 方法 | TPS（tokens/sec）↑ |
| --- | --- |
| Greedy | 33.6 |
| Beam Search | 30.3 |
| OPERA | 18.8 |
| VCD | 14.2 |
| DoLa | 3.6 |
| **VisFlow** | **28.6** |

VisFlow 的推理速度接近 Beam Search（28.6 vs 30.3 TPS），远快于 VCD（2×）和 DoLa（8×）。这是因为 VisFlow 不需要额外前向传播或多轮解码。

---

## 五、局限性与未来方向

### 5.1 模型特异性

注意力头的分类（阈值 $\lambda_{\text{vis}}$、$\lambda_{\text{txt}}$、$\lambda_{\text{sys}}$）以及 TAI 的作用层范围是基于 LLaVA-1.5 等特定模型架构调优的。对于视觉 token 经过语义压缩的模型（如 MiniGPT-4 的 Q-former、mPLUG-Owl2），TAI 不适用（论文中对这两个模型不应用 TAI）。

### 5.2 评估范围

实验仅在 CHAIR 和 POPE 两个基准上进行，缺少对更复杂推理任务（如 MMBench、LLaVA-Bench）的评估，难以判断 VisFlow 是否影响模型的通用多模态理解能力。

### 5.3 超参数数量

VisFlow 引入了较多超参数（$k$、$\delta$、$\lambda_{\text{vis}}$、$\lambda_{\text{sys}}$、$\lambda_{\text{txt}}$、$\alpha_{\text{sys}}$、$\alpha_{\text{txt}}$），虽然论文给出了建议值，但在新模型或新任务上可能需要重新调优。

---

## 六、个人思考

### 6.1 与项目内其他幻觉缓解工作的对比

| 方法 | 干预阶段 | 是否需要训练 | 核心机制 | 推理开销 | 主要优势 |
| --- | --- | --- | --- | --- | --- |
| **VisFlow** | 解码时（注意力） | 否 | Token/Head 双层注意力操控 | 极低（~Beam Search） | 速度快、无外部依赖 |
| HALC | 解码时（输出分布） | 否 | FOV 对比 + 视觉匹配 beam search | 较高（~1.35×） | 处理三种幻觉 |
| DLC | 解码时（logits） | 否 | CLIP 探针 + 相对视觉优势 | 低 | 轻量级 logit 校准 |
| HIME | 推理前（权重编辑） | 否 | 零空间投影编辑 MLP | 零 | 无推理时开销 |
| SENTINEL | 训练时 DPO | 是 | 句子级 C-DPO 早期干预 | 零 | 从训练阶段根治 |

VisFlow 和 HIME 代表了两种互补的思路：HIME 通过**永久修改权重**消除幻觉倾向（一劳永逸但不可逆），VisFlow 通过**动态操控注意力**在推理时纠正（灵活但每次推理都需要）。理论上两者可以叠加使用。

### 6.2 Visual Sink Token 与 Attention Sink 的联系

VisFlow 发现的 Visual Sink Token 现象与 LLM 中的 Attention Sink（StreamingLLM, Xiao et al. 2023）高度类似——序列中的某些 token 充当"注意力垃圾桶"，吸收大量注意力但本身不携带语义信息。这是 Softmax 归一化的副作用：注意力权重必须加和为 1，当模型"不知道该看哪里"时，就把多余的注意力分配给这些 sink token。VisFlow 的 TAI 本质上是在视觉 token 内部做 attention redistribution，将被 sink 吸走的注意力重新导向真正有用的显著 token。

### 6.3 从信息瓶颈的视角理解

VisFlow 的三个发现（弱视觉 grounding、语言先验主导、系统提示冗余）可以统一为一个信息瓶颈问题：**在有限的注意力带宽中，系统提示和语言先验占据了过多资源，挤压了视觉信息的传递通道**。HAI 通过抑制 system/text heads 来"释放带宽"，TAI 通过增强 salient token 来"提高信号强度"，两者协同扩大了视觉信息的有效通量。

### 6.4 方法的可扩展性思考

VisFlow 的方法论框架（识别注意力模式 → 分类功能角色 → 定向干预）具有很好的通用性。类似的思路可以应用于：
- **视频 VLM** 中的时序幻觉（temporal hallucination），通过分析帧间注意力流来干预
- **多轮对话** 中的上下文遗忘，通过增强历史 token 的注意力来保持一致性
- **RAG 系统** 中的检索信息忽视，通过增强检索文档 token 的注意力来提高 faithfulness

---

## 参考

- **VCD**（Leng et al., 2024）：视觉对比解码，通过噪声图像做全图对比——VisFlow 不需要额外的对比图像
- **OPERA**（Huang et al., 2024）：注意力过度信任惩罚 + 回溯分配——VisFlow 直接操控注意力，不需要回溯
- **DoLa**（Chuang et al., 2023）：层间对比解码——VisFlow 的 HAI 也利用了层间差异，但以注意力头而非层为粒度
- **CCA-LLaVA**（Xing et al., 2024）：同心因果注意力缓解 RoPE 位置偏差——VisFlow 通过 TAI 从另一个角度（salient/sink token 分类）解决相同问题
- **HALC**（Chen et al., 2024）：自适应 FOV 对比解码——HALC 在输出分布层面干预，VisFlow 在注意力权重层面干预，后者更轻量
