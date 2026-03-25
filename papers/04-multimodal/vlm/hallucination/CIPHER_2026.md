# CIPHER：扩散引导的反事实图像扰动用于 LVLM 幻觉抑制

> **论文**：*Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression*
>
> **作者**：Hamidreza Dastmalchi, Aijun An, Ali Cheraghian, Hamed Barzamini
>
> **机构**：York University (加拿大)、Macquarie University (澳大利亚)、Northern Illinois University (美国)
>
> **发布时间**：2026年3月（arXiv）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.10470) | [项目主页](https://hamidrezadastmalchi.github.io/cipher-cvpr2026/)
>
> **分类标签**：`LVLM` `Hallucination` `Feature-Level Intervention` `Counterfactual` `Diffusion Model` `SVD` `Training-Free`

---

## 一句话总结

提出 CIPHER，通过 Stable Diffusion 生成 25K 反事实图像（语义篡改但结构保留）构造 OHC-25K 数据集，对比反事实与真实图像的 LVLM 隐藏表示差异经 SVD 提取**视觉诱导幻觉子空间**，推理时将隐藏状态投影到该子空间的正交补空间，training-free 单次前向传播零额外开销，LLaVA-1.5 CHAIR$_S$ 降至 13.05%（vs Nullu 15.20%），首次从**视觉模态**角度提取幻觉方向。

---

## 一、问题与动机

### 1.1 视觉诱导幻觉被忽视

LVLM 幻觉的根源不仅来自 LLM 的生成偏差，也来自弱视觉基础、模态错位和偏差训练数据。然而，现有 training-free 方法**主要针对文本诱导的幻觉**（如 Nullu 通过文本扰动提取幻觉方向），对视觉模态引起的幻觉关注不足。

### 1.2 现有方法的分类与不足

| 方法类别 | 代表方法 | 局限 |
| --- | --- | --- |
| 训练方法 | InternVL, LRV-Instruction | 需要昂贵标注、重训练或架构修改 |
| 后处理 | Woodpecker, LURE | 依赖外部模型（GPT），部署复杂 |
| 解码干预 | VCD, OPERA, HALC, DoLa | 多次前向传播，推理开销大（OPERA 0.10 items/s, HALC 0.05 items/s） |
| 特征干预 | Nullu | 仅针对**文本**扰动提取幻觉方向，忽略视觉诱导幻觉 |

### 1.3 核心洞察

论文通过线性探测实验（Linear Probing）验证：
- **文本扰动**产生的隐藏表示偏移在各层仅有**中等且不稳定的可分性**（Accuracy: 0.73–0.80, F1: 0.74–0.78）
- **视觉扰动**（扩散反事实）产生的偏移在所有层都有**一致的高线性可分性**（Accuracy: 0.86–0.89, F1: 0.86–0.89）

这说明视觉扰动引入了**更强、更结构化、层间更稳定的表示偏移**，从视觉模态提取的幻觉子空间比从文本模态提取的更有效。

---

## 二、预备知识

### 2.1 Nullu：基于文本扰动的幻觉方向提取

Nullu (CVPR 2025) 是 CIPHER 的前置工作。其思路是：
1. 用 GPT 对文本 caption 进行扰动（注入错误对象）
2. 对比扰动前后的 LVLM 隐藏表示，提取差异向量
3. SVD 分解得到幻觉子空间
4. 推理时将隐藏状态投影到该子空间的正交补

Nullu 的局限：**仅扰动文本，只能捕获语言诱导的幻觉方向**。

### 2.2 Stable Diffusion 的受控编辑

给定图像 $I$，通过 VAE 编码器 $\mathcal{E}$ 映射到潜在空间 $z_0 = \mathcal{E}(I)$，施加前向扩散引入噪声，再以修改后的 caption 为条件进行反向去噪，可以生成**结构保留但语义改变**的反事实图像。

---

## 三、核心方法

CIPHER 分为两个阶段：离线阶段（构建反事实数据集 + 提取幻觉子空间）和推理阶段（投影抑制幻觉）。

### 3.1 离线阶段：构建 OHC-25K 反事实数据集

从 MSCOCO 训练集随机选取 $M = 5000$ 对图像-caption 对 $\{(I_i, C_i)\}_{i=1}^M$。

**Step 1：生成幻觉 caption。** 使用 GPT-3.5 对每个真实 caption $C_i$ 进行扰动，注入看似合理但实际不存在的对象，生成幻觉 caption $\tilde{C}_i$。

**Step 2：生成反事实图像。** 对原始图像进行"往返扩散"——先前向加噪再以幻觉 caption 为条件反向去噪：

$$z_0 = \mathcal{E}(I_i)$$

$$\tilde{z}_{t_h} = \sqrt{\bar{\alpha}_{t_h}} z_0 + \sqrt{1 - \bar{\alpha}_{t_h}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$\tilde{z}_{t-1} = f_\theta(\tilde{z}_t, t, \tilde{C}_i), \quad t = t_h, \ldots, 1$$

$$\tilde{I}_{i,j} = \mathcal{D}(\tilde{z}_0)$$

其中 $t_h = 0.5T$（总扩散步数的一半），每张原始图像生成 $B = 5$ 个反事实变体（不同噪声种子）。

用大白话说：加一半噪声保留图像结构，再以包含错误对象的描述引导去噪，得到"看起来差不多但内容被篡改"的图像。例如原图有黄瓜和番茄，反事实图可能多出了不存在的香蕉和葡萄。

**最终数据集**：

$$\text{OHC-25K} = \{(\tilde{I}_{i,j}, C_i) \mid i = 1, \ldots, M, \; j = 1, \ldots, B\}$$

注意每个反事实图像仍配对**原始真实 caption**，制造语义冲突。

### 3.2 离线阶段：提取幻觉子空间

对每对真实 $(I_i, C_i)$ 和反事实 $(\tilde{I}_{i,j}, C_i)$，通过冻结的 LVLM 提取第 $\ell$ 层的隐藏表示。

**Token 级池化。** 对 caption token 进行均值池化：

$$h^{(i)}_\ell = \frac{1}{N} \sum_{k=1}^N h^{(i)}_{\ell,k}, \quad \tilde{h}^{(i,j)}_\ell = \frac{1}{N} \sum_{k=1}^N \tilde{h}^{(i,j)}_{\ell,k}$$

**跨变体聚合。** 对 $B$ 个反事实变体取平均：

$$\bar{\tilde{h}}^{(i)}_\ell = \frac{1}{B} \sum_{j=1}^B \tilde{h}^{(i,j)}_\ell$$

**差异向量。** 幻觉方向定义为：

$$\delta^{(i)}_\ell = \bar{\tilde{h}}^{(i)}_\ell - h^{(i)}_\ell$$

**SVD 分解。** 将所有样本的差异向量堆叠并进行 SVD：

$$\Delta_\ell = [\delta^{(1)}_\ell; \delta^{(2)}_\ell; \cdots; \delta^{(M)}_\ell] \in \mathbb{R}^{M \times d}$$

$$\Delta_\ell = U_\ell \Sigma_\ell V_\ell^\top$$

取前 $r$ 个右奇异向量 $V_{\ell,r} = [v_{\ell,1}, \ldots, v_{\ell,r}]$ 作为第 $\ell$ 层的**幻觉基向量库 (Hallucination Basis Bank)**。

### 3.3 推理阶段：幻觉零化投影

在每个解码步骤 $k$ 和选定层 $\ell$（默认层 16–32），将隐藏状态投影到幻觉子空间的正交补：

$$h^{\text{clean}}_{\ell,k} = h^{\text{test}}_{\ell,k} - \sum_{j=1}^r \langle h^{\text{test}}_{\ell,k}, v_{\ell,j} \rangle v_{\ell,j}$$

等价地，使用投影矩阵 $P_\ell = I - V_{\ell,r} V_{\ell,r}^\top$：

$$h^{\text{clean}}_{\ell,k} = P_\ell \; h^{\text{test}}_{\ell,k}$$

用大白话说：预先算好"幻觉长什么样"（用反事实图像标定），推理时把隐藏表示中与幻觉方向对齐的分量减掉，只保留正交于幻觉方向的成分。这个操作只是一个矩阵乘法，**零额外推理开销**。

---

## 四、实验结果

### 4.1 CHAIR 基准

在 3 种 LVLM 上的图像描述幻觉评估：

| 方法 | LLaVA-1.5 CHAIR$_S$↓ | LLaVA-1.5 CHAIR$_I$↓ | LLaVA-1.5 BLEU↑ | MiniGPT-4 CHAIR$_S$↓ | MiniGPT-4 CHAIR$_I$↓ | MiniGPT-4 BLEU↑ | mPLUG-Owl2 CHAIR$_S$↓ | mPLUG-Owl2 CHAIR$_I$↓ | mPLUG-Owl2 BLEU↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 20.40 | 7.08 | 15.72 | 32.40 | 12.20 | 14.57 | 22.90 | 8.62 | 15.01 |
| OPERA | 17.50 | 6.07 | 16.02 | 29.70 | 11.96 | 14.82 | 20.07 | 7.18 | 15.41 |
| VCD | 20.30 | 7.28 | 14.53 | 29.00 | 12.64 | 14.42 | 22.80 | 8.68 | 15.14 |
| HALC | 16.90 | 5.72 | 16.02 | 25.20 | 9.42 | 14.91 | 18.80 | 7.00 | 15.33 |
| Nullu | 15.20 | 5.30 | 15.69 | 21.40 | 8.99 | 14.81 | 15.60 | 5.77 | 15.45 |
| **CIPHER** | **13.05** | **4.53** | 15.82 | **18.48** | **8.33** | 15.10 | **13.60** | **4.92** | **16.25** |

CIPHER 在所有模型上均取得 CHAIR$_S$/CHAIR$_I$ 最优，同时 BLEU 保持或提升。相比 Nullu：
- LLaVA-1.5: CHAIR$_S$ -2.15%, CHAIR$_I$ -0.77%
- MiniGPT-4: CHAIR$_S$ -2.92%, CHAIR$_I$ -0.66%
- mPLUG-Owl2: CHAIR$_S$ -2.00%, CHAIR$_I$ -0.85%

### 4.2 OPOPE 基准

| 方法 | LLaVA-1.5 Acc↑ | LLaVA-1.5 Prec↑ | LLaVA-1.5 F↑ | MiniGPT-4 Acc↑ | MiniGPT-4 Prec↑ | MiniGPT-4 F↑ | mPLUG-Owl2 Acc↑ | mPLUG-Owl2 Prec↑ | mPLUG-Owl2 F↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nullu | 79.52 | 93.46 | 91.79 | 71.92 | 95.96 | 92.07 | 77.09 | 92.83 | 90.80 |
| **CIPHER** | **80.05** | **93.72** | **92.11** | **72.25** | **96.50** | **92.58** | **77.87** | **92.93** | **90.95** |

在已趋饱和的性能区间内仍取得一致提升。

### 4.3 推理效率

| 指标 | Greedy | OPERA | HALC | Nullu | CIPHER |
| --- | --- | --- | --- | --- | --- |
| CHAIR$_S$↓ | 20.40 | 17.50 | 16.90 | 15.20 | **13.05** |
| 吞吐量 (items/s)↑ | 0.70 | 0.10 | 0.05 | **0.70** | **0.70** |

CIPHER 与 Greedy/Nullu 吞吐量相同（0.70 items/s），远超 OPERA（7×）和 HALC（14×）。幻觉投影只是一个预计算的矩阵乘法，**零额外推理延迟**。

### 4.4 消融实验

**扩散时间步 $t_h$**：$t_h = 0.5T$ 效果最佳（CHAIR$_S$ 13.05%）。过少（$0.25T$：15.62%）语义改变不足，过多（$T$：15.81%）结构破坏过大。中间值在"保留结构"和"改变语义"之间取得最佳平衡。

**子空间秩 $r$**：LLaVA-7B 上 $r = 8$ 最优，MiniGPT-4 上 $r = 64$，mPLUG-Owl2 上 $r = 32$。不同模型最优秩差异较大，需要针对性调优。

**幻觉来源**：

| 文本扰动 | 图像扰动 | CHAIR$_S$↓ | CHAIR$_I$↓ | BLEU↑ |
| --- | --- | --- | --- | --- |
| ✓ | ✗ | 15.20 | 5.30 | 15.69 |
| ✗ | ✓ | **13.05** | **4.53** | **15.82** |
| ✓ | ✓ | 15.71 | 5.32 | 15.66 |

单独图像扰动效果最好；联合文本+图像扰动反而略有下降——两种幻觉方向可能存在干扰。

**噪声鲁棒性**：在不同高斯噪声水平 $\sigma \in [0, 1]$ 下，CIPHER 始终优于原始模型，且在严重噪声下差距更大。

---

## 五、局限性与未来方向

1. **离线数据集依赖**：需要预先构建 OHC-25K 反事实数据集，涉及 GPT-3.5 caption 扰动和 Stable Diffusion 图像生成，离线成本不低
2. **固定投影矩阵**：当前使用全局固定的投影矩阵，无法根据输入图像内容动态调整。论文提出未来方向为**输入自适应的动态投影**
3. **子空间秩需逐模型调优**：LLaVA ($r=8$) vs MiniGPT-4 ($r=64$) vs mPLUG-Owl2 ($r=32$)，最优秩差异高达 8 倍，缺乏自动选择机制
4. **仅针对对象幻觉**：实验主要评估对象级幻觉（CHAIR/POPE），对属性、关系、计数等其他类型幻觉的效果从 MMHal 结果看有改善但不均匀
5. **文本+视觉联合反而更差**：消融实验显示两种幻觉方向联合使用性能下降，说明当前方法无法有效融合两种信息源

---

## 六、个人思考

### 6.1 与 Nullu 的关系——视觉 vs 文本幻觉方向

CIPHER 本质上是 Nullu 的"视觉版"——两者共享相同的"提取幻觉子空间 → SVD → 推理时投影"框架，唯一区别在于**扰动来源**：Nullu 扰动文本（caption），CIPHER 扰动图像（扩散编辑）。论文通过线性探测实验有力地证明了视觉扰动产生的表示偏移更强且更稳定，从而解释了 CIPHER 的性能优势。

### 6.2 与 TAF 的互补性

TAF（同在本站收录）从**注意力机制**角度干预——识别 phantom/anchor token 并调制注意力 logits。CIPHER 从**特征空间**角度干预——投影隐藏状态到幻觉子空间的正交补。两者操作在不同层面（注意力 vs 隐藏状态），理论上可以组合使用。

### 6.3 与 HIME 的联系

HIME（同在本站收录）也是特征级干预方法，通过 HIS 量化层敏感度后投影编辑 MLP 权重。CIPHER 的不同在于：(1) 投影的是**隐藏状态**而非**权重**；(2) 通过**反事实数据**而非**幻觉敏感度分数**来定义幻觉方向。CIPHER 的优势是更贴近幻觉的因果机制（通过反事实推理），HIME 的优势是不需要额外数据集。

### 6.4 反事实思维的巧妙之处

论文最关键的洞察是：**想要消除幻觉，不如先故意制造幻觉**。通过扩散模型生成"结构保留但语义篡改"的反事实图像，可以精准标定幻觉在特征空间中的方向。这种反事实推理思路值得其他领域借鉴——例如在 VLA 领域，是否可以通过反事实视觉扰动来提取导致策略失败的特征方向？

### 6.5 关于文本+视觉联合更差

消融实验中联合使用反而更差，可能的解释是：文本幻觉方向和视觉幻觉方向在特征空间中并不正交，联合 SVD 后的子空间"稀释"了各自的信号。未来可以考虑分别提取后正交化处理。

---

## 参考

- **Nullu** (Yang et al., CVPR 2025)：文本扰动提取幻觉子空间 + 投影抑制，CIPHER 的直接前置工作
- **OPERA** (Huang et al., CVPR 2024)：注意力柱状模式 + 过度信任惩罚 + 回溯重分配
- **VCD** (Leng et al., CVPR 2024)：对比正常与干扰图像的输出分布差异缓解幻觉
- **HALC** (Chen et al., ICML 2024)：自适应 FOV 对比解码 + 视觉匹配 beam search
- **LURE** (Zhou et al., ICLR 2024)：GPT 生成幻觉 caption 用于后处理修正
- **Stable Diffusion** (Rombach et al., CVPR 2022)：潜在扩散模型，CIPHER 用于反事实图像生成
