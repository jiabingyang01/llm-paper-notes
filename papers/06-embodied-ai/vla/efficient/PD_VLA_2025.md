# PD-VLA：基于并行解码加速带 Action Chunking 的 VLA 模型

> **论文**：*Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding*
>
> **作者**：Wenxuan Song*, Jiayi Chen*, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, Haoang Li
>
> **机构**：HKUST (Guangzhou)、Westlake University、Zhejiang University、Monash University
>
> **发布时间**：2025 年 3 月
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2503.02310)
>
> **发表会议**：IROS 2025
>
> **分类标签**：`VLA` `并行解码` `Action Chunking` `Jacobi 迭代` `Training-Free`

---

## 一句话总结

将 VLA 模型的自回归动作解码重新建模为非线性方程组，通过 Jacobi 不动点迭代实现**并行解码**，在不修改模型架构、不需要额外训练的前提下，为带 action chunking 的 VLA 模型实现 **2.52× 执行频率提升**。

---

## 一、问题与动机

### 1.1 Action Chunking 引入的推理瓶颈

Action Chunking 是近年 VLA 模型的关键技术——预测并执行多步动作序列（而非单步），能显著提升动作的一致性和稳定性。然而，对于采用**自回归（AR）解码**的 VLA 模型，action chunking 使推理开销线性增长：

- 7-DoF 机械臂单步动作 = 7 个 token
- Chunk size $m = 5$ → 需生成 $7 \times 5 = 35$ 个 action token
- AR 解码逐 token 生成，推理时间与 token 数成正比

这一矛盾使得 VLA 模型难以满足实时控制对高频推理的需求。

### 1.2 现有加速方法的局限

| 方法类型 | 代表工作 | Model-redesign-free | Training-free | Modification-free |
| --- | --- | --- | --- | --- |
| 轻量架构 | TinyVLA、RoboMamba | ✗ | - | - |
| 量化 / 动态推理 | QAIL、DeeR-VLA | ✓ | ✗ | ✗ |
| Token 剪枝 / 缓存 | SparseVLM、FastV、VLA-Cache | ✓ | ✓ | ✗ |
| **PD-VLA（本文）** | — | **✓** | **✓** | **✓** |

- 轻量架构方法需要**重新设计和训练**模型
- 量化/动态推理方法需要**训练或修改**预训练模型
- Token 剪枝/缓存方法虽 training-free，但需要**添加辅助组件**（如缓存机制、剪枝模块）

PD-VLA 是首个同时满足三个"free"的 VLA 加速方法——**不重设计、不训练、不修改模型**，仅改变解码过程。

### 1.3 核心洞察

AR 解码的顺序依赖并非硬性约束——可以将其重新建模为一个**非线性方程组**，用 Jacobi 不动点迭代并行求解。对于 VLA 的动作 token，由于动作空间结构化（如夹爪状态只有开/关两个值），许多 token 可以在前驱 token 尚未正确时就"提前猜对"（**fixed token 现象**），从而加速收敛。

---

## 二、预备知识

### 2.1 VLA 模型架构

PD-VLA 构建在 LLaVA-VLA 上，包含：

- **Vision Encoder** $f_{\text{encoder}}$：CLIP-ViT-Large-Patch14-336，编码静态图像 $I_{\text{static}}$ 和夹爪图像 $I_{\text{gripper}}$ 为视觉 token $h_{\text{img}}$
- **Text Tokenizer** $\mathcal{T}$：将指令 $S$ 编码为 $h_S$
- **LLM**：Vicuna-7B-v1.5，自回归生成 action token $h_{\text{act}}$
- **Action De-Tokenizer**：将 token 还原为 7 维连续动作

整体流程：

$$a = \text{Detokenize}(\text{LLM}(f_{\text{encoder}}(I_{\text{static}}, I_{\text{gripper}}), \mathcal{T}(S)))$$

### 2.2 Action Tokenization

将连续动作 $a = [X, Y, Z, \phi, \theta, \psi, G]$（3-DoF 平移 + 3-DoF 旋转 + 1-DoF 夹爪）离散化为 **256 个均匀 bin**，用 LLM 词表中最低频的 256 个 token 表示。各维度 token 以空格拼接为文本字符串作为训练标签。

### 2.3 Action Chunking

在时间步 $t$，给定 chunk size $m$，模型预测动作序列：

$$A_t = [a_t, a_{t+1}, \ldots, a_{t+m-1}]$$

本文设 $m = 5$，则每次推理需生成 $l = 7m + 2 = 37$ 个 token（含起始和结束 token）。AR 解码需要 37 次前向传播，成为实时控制的瓶颈。

### 2.4 标准 AR 解码

在 greedy 策略下，给定 prompt $\boldsymbol{x}$（含视觉和文本），AR 逐步生成：

$$y_i = \arg\max_y \; p(y \mid \mathcal{Y}_i, \boldsymbol{x}), \quad i = 1, \ldots, n$$

其中 $\mathcal{Y}_i = \{y_1, \ldots, y_{i-1}\}$。生成 $n$ 个 token 需要 $n$ 次前向传播。

---

## 三、核心方法

### 3.1 从 AR 解码到非线性方程组

**Jacobi 解码**的核心思想：将 AR 推理过程重新表述为求解非线性方程组。

定义 $f(y_i, \mathcal{Y}_i, \boldsymbol{x}) := y_i - \arg\max_y \; p(y \mid \mathcal{Y}_i, \boldsymbol{x})$，则 AR 解码等价于求解：

$$f(y_i, \mathcal{Y}_i, \boldsymbol{x}) = 0, \quad i = 1, \ldots, n$$

这是一个包含 $n$ 个未知数 $y_i$ 的 $n$ 元非线性方程组。标准 Jacobi 不动点迭代为：

$$\begin{cases}
y_1^{(j+1)} = \arg\max_y \; p(y \mid \boldsymbol{x}) \\
y_2^{(j+1)} = \arg\max_y \; p(y \mid \mathcal{Y}_1^{(j)}, \boldsymbol{x}) \\
\vdots \\
y_n^{(j+1)} = \arg\max_y \; p(y \mid \mathcal{Y}_n^{(j)}, \boldsymbol{x})
\end{cases}$$

其中 $\mathcal{Y}_i^{(j)}$ 表示第 $j$ 次迭代时的前 $i-1$ 个 token 估计值。

### 3.2 PD-VLA 的并行解码

PD-VLA 对标准 Jacobi 解码做了关键修改——将 causal attention mask 替换为 **bidirectional attention mask**，使每个 token 的更新基于**当前迭代的所有 token**而非仅前驱：

$$\begin{cases}
y_1^{(j+1)} = \arg\max_y \; p(y \mid \mathcal{Y}^{(j)}, \boldsymbol{x}) \\
y_2^{(j+1)} = \arg\max_y \; p(y \mid \mathcal{Y}^{(j)}, \boldsymbol{x}) \\
\vdots \\
y_n^{(j+1)} = \arg\max_y \; p(y \mid \mathcal{Y}^{(j)}, \boldsymbol{x})
\end{cases}$$

**流程**：

> 1. **随机初始化**：生成与解码长度 $n$ 等长的随机 action token 序列 $\mathcal{Y}^{(0)} = \{y_1^{(0)}, \ldots, y_n^{(0)}\}$
> 2. **并行前向传播**：将 prompt $\boldsymbol{x}$ 和当前 token 序列 $\mathcal{Y}^{(j)}$ 一起输入 LLM，**一次前向传播同时预测所有 $n$ 个 token 的更新值**
> 3. **收敛检查**：若 $\mathcal{Y}^{(k)} = \mathcal{Y}^{(k-1)}$（所有 token 不再变化），则达到不动点 $\mathcal{Y}^* := \mathcal{Y}^{(k)}$，终止迭代
> 4. **输出**：将 $\mathcal{Y}^*$ 反 tokenize 为连续动作序列

**加速来源**：每次迭代通过一次前向传播更新**所有** $n$ 个 token，总迭代次数 $k \leq n$。当 $k \ll n$ 时，相比 AR 的 $n$ 次前向传播，实现显著加速。

### 3.3 Decoding Horizon 的选择

Decoding horizon $n$ 是并行解码的核心超参数，决定每次 Jacobi 迭代中并行预测的 token 数量：

- **$n < l$（分段解码）**：每次迭代解码 $n$ 个 token，然后滑动到下一段，相当于多个 Jacobi 解码 + Gauss-Seidel 步的组合
- **$n = l = 37$（全局解码）**：一次 Jacobi 解码覆盖所有 action token，保留原始动作分布建模能力

论文选择 $n \in \{7, 16, 37\}$ 进行对比分析：

| Decoding Horizon $n$ | 设计直觉 |
| --- | --- |
| 7 | 对齐单步动作维度（7-DoF），每段独立解码 |
| 16 | 2 的幂次，硬件友好 |
| 37 | 覆盖完整 action chunk（$7 \times 5 + 2$） |

### 3.4 Fixed Token 现象与加速机理

PD-VLA 在解码过程中观察到 **fixed token 现象**：即使前驱 token 尚未收敛，某些位置的 token 已经预测正确且在后续迭代中保持不变。这些 token 被称为 **fixed token**。

**产生原因**：

- VLA 的动作空间高度结构化，某些维度（尤其是夹爪状态 $G$，仅 0/1 两个值）对上下文的依赖较弱
- 同一 chunk 内相邻时间步的动作具有时间连续性，多个位置的 token 值相近

**加速效果**：fixed token 的存在使得不连续的正确 token 可以在序列中同时扩展，加速整体收敛。Decoding horizon 越大，fixed token 越多：

| $n$ | Fixed Token 数量 |
| --- | --- |
| 7 | 5.17 |
| 16 | 6.75 |
| 37 | **8.75** |

### 3.5 与现有加速方法的协同

PD-VLA 仅作用于**解码过程**，不改变视觉编码或模型参数，因此可以与以下方法无缝叠加：

- **Token 剪枝**（FastV、SparseVLM）：减少 prefill 阶段的视觉 token
- **KV 缓存**（VLA-Cache）：跨帧复用静态 token 的 KV
- **量化**（QAIL、BitVLA）：压缩模型权重

---

## 四、实验结果

### 4.1 实验设置

- **仿真环境**：CALVIN 基准（34 个任务，4 个环境 A/B/C/D，ABCD→D 设置）
- **评估指标**：5 个子任务链的逐步成功率、平均完成长度、推理速度（Token/s）、执行频率（Hz）
- **基座模型**：LLaVA-VLA（Vicuna-7B + CLIP-ViT-Large），8× NVIDIA H100 训练，约 10 小时
- **PD-VLA 无额外训练开销**

### 4.2 与基线模型对比

| 方法 | 输入 | 数据 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | 平均长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCIL | RGB | ALL | 37.3 | 2.7 | 0.2 | 0.0 | 0.0 | 0.40 |
| HULC | RGB | ALL | 89.2 | 70.1 | 54.8 | 42.0 | 33.5 | 2.90 |
| RT-1 | RGB | LANG | 84.4 | 61.7 | 43.8 | 32.3 | 22.7 | 2.45 |
| LLaVA-VLA | RGB | LANG | 72.0 | 29.0 | 12.0 | 6.0 | 1.9 | 1.20 |
| **PD-VLA** | RGB | LANG | **94.1** | **80.0** | **68.3** | **61.4** | **50.5** | **3.55** |

PD-VLA 在所有子任务上大幅超越基线 LLaVA-VLA（平均长度 1.20 → 3.55），这得益于 action chunking 带来的动作一致性提升，以及并行解码使高频推理成为可能。

### 4.3 消融实验

| 方法 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | 平均长度 | 速度 (Token/s) | 频率 (Hz) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaVA-VLA | 72.0 | 29.0 | 12.0 | 6.0 | 1.9 | 1.20 | 39.56 | 1.81 |
| w/o AC（无 Action Chunking） | 71.0 | 25.0 | 8.0 | 6.0 | 2.0 | 1.12 | 39.86 | 1.82 |
| w/o PD（仅 Action Chunking） | 91.8 | 82.4 | 71.0 | 62.8 | 52.6 | 3.61 | 41.44 | 3.60 |
| w/o PD, w/ FastV | 90.1 | 77.2 | 62.4 | 55.4 | 46.5 | 3.31 | 28.69 | 2.54 |
| w/o PD, w/ SparseVLM | 83.2 | 63.2 | 46.0 | 36.0 | 26.4 | 2.55 | 32.43 | 2.83 |
| **PD-VLA（AC + PD）** | **94.1** | **80.0** | **68.3** | **61.4** | **50.5** | **3.64** | **52.84** | **4.56** |

**关键发现**：

1. **Action Chunking 的贡献**：平均长度从 1.12 提升到 3.61（+2.24），通过减少推理次数将频率从 1.82 Hz 提升至 3.60 Hz（**1.98×**）
2. **Parallel Decoding 的贡献**：解码速度从 41.44 提升到 52.84 Token/s（**1.28×**），频率从 3.60 Hz 提升至 4.56 Hz
3. **两者互补**：AC 提升动作一致性 + 减少推理次数，PD 加速单次推理过程，合计实现 **2.52× 执行频率**
4. **FastV / SparseVLM 反效果**：FastV 因额外 mask 开销反而更慢（28.69 Token/s < 41.44），SparseVLM 的 token 剪枝/合并/回收开销导致速度和性能双降

### 4.4 Decoding Horizon 分析

| Decoding Horizon $n$ | Fixed Token 数量 | 平均长度 | 速度 (Token/s) | 频率 (Hz) |
| --- | --- | --- | --- | --- |
| 7 | 5.17 | 3.24 | 41.48 | 3.60 |
| 16 | 6.75 | 3.19 | 48.74 | 3.25 |
| 37 | **8.75** | **3.64** | **52.84** | **4.56** |

- **$n = 37$ 全面最优**：速度最高、性能最强，因为整体预测保留了原始动作分布的完整建模
- **$n = 7$ 优于 $n = 16$**：$n = 7$ 对齐单步 7-DoF 结构，解码效率高于不对齐的 $n = 16$
- **$n = 16$ 频率反而最低**：由于 $l = 37$ 不是 16 的倍数，存在冗余 token 预测的浪费
- **Fixed token 随 $n$ 增大而增多**：$n = 37$ 时 fixed token 达 8.75，贡献了最大加速

最大推理速度方面，$n = 37$ 的峰值速度约为 AR 和 $n = 7$ 的 **2 倍**。

### 4.5 真实世界实验

在 6-DoF Unitree Z1-Pro 机械臂 + ORBBEC Femto Mega 相机上验证，3 个任务各 50 条演示、10 次评估：

| 方法 | Push Button | Lift Block | Pour Water |
| --- | --- | --- | --- |
| LLaVA-VLA | 60% | 40% | 10% |
| **PD-VLA** | **80%** | **70%** | **60%** |

- PD-VLA 在所有任务上显著领先，尤其在需要灵巧操控的 **Pour Water** 任务上提升 50pp
- Pour Water 要求动作高度连续和一致——非刚性塑料瓶的抓握和倾倒过程中，任何不一致都会导致瓶子掉落。PD-VLA 凭借 action chunking + 高频推理保证了动作流畅性

---

## 五、局限性与未来方向

1. **收敛不保证最优迭代次数**：Jacobi 迭代的收敛速度取决于动作分布的结构，最坏情况下 $k = n$，退化为 AR 解码，无加速收益

2. **Bidirectional Attention 的理论偏差**：将 causal mask 替换为 bidirectional mask 改变了 LLM 的条件分布，理论上不再严格等价于 AR 解码。论文通过实验验证了实际影响可控，但缺乏严格的误差界分析

3. **仅验证了单一基座模型**：实验仅在 LLaVA-VLA（Vicuna-7B）上进行，未验证对 π₀（Flow Matching）、OpenVLA（LLaMA-2）等其他 VLA 架构的适用性

4. **仿真-真实差距**：CALVIN 仿真中的高成功率（94.1%）与真实世界的 60-80% 之间存在明显差距，且真实世界仅评估 10 次

5. **未探索与可学习加速方法的协同**：论文声称可与 Token 缓存等方法叠加，但未提供实际实验验证

---

## 六、个人思考

### 6.1 与项目中其他 VLA 加速方法的对比

| 维度 | PD-VLA | [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) | [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) | [VLA-Pruner](/papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025) |
| --- | --- | --- | --- | --- |
| 加速策略 | 并行解码（Jacobi 迭代） | 跨帧 KV 缓存 | LLM 层剪枝 + Token 选择 | 双层 Token 剪枝 |
| 作用阶段 | 解码过程 | Prefill + 解码 | Prefill + 解码 | Prefill |
| Training-Free | ✓ | ✓ | ✓ | ✓ |
| Modification-Free | ✓ | ✗ | ✗ | ✗ |
| 加速倍率 | 2.52× 频率 | 1.7× 延迟 | 1.93× | 1.8× |
| 评估基准 | CALVIN | LIBERO / SIMPLER | LIBERO | LIBERO |

PD-VLA 的独特优势在于**完全不修改模型**——其他方法都需要添加缓存机制或剪枝逻辑，而 PD-VLA 仅改变解码策略。但代价是需要 bidirectional attention，这在某些推理框架中可能需要特殊处理。

### 6.2 并行解码适用的"甜蜜点"

PD-VLA 的加速收益高度依赖 action chunking 带来的长 token 序列。当 chunk size $m = 1$（无 chunking）时，仅需生成 7+2=9 个 token，并行解码的收益微乎其微。这意味着 PD-VLA 天然适配**大 chunk size 的 VLA 模型**（如 π₀ 的 flow matching 架构生成更长的动作序列），但对单步 AR-VLA 价值有限。

### 6.3 Fixed Token 现象的启示

Fixed token 现象揭示了 VLA 动作空间的一个有趣性质：**不同维度的条件依赖强度差异很大**。夹爪状态（二值）几乎不依赖其他维度，而位置/旋转维度之间存在较强的物理约束。这一观察可能启发未来工作：

- 对不同维度采用**不同解码策略**——强独立维度（夹爪）直接并行，强依赖维度（位置-旋转）保留 AR 或使用更少迭代
- 利用 fixed token 分布设计**自适应终止条件**——当 fixed token 比例达到阈值时提前终止

### 6.4 Bidirectional Attention 的隐忧

PD-VLA 将 causal mask 替换为 bidirectional mask 是一个大胆但存在隐患的设计。预训练的 LLM（Vicuna-7B）在训练时使用的是 causal attention，推理时切换为 bidirectional 会导致**注意力分布偏移**。论文在 VLA 场景下验证了影响可控，但这可能是因为动作 token 的分布远比自然语言简单（仅 256 个离散 bin）。对于更复杂的输出空间，这一策略的鲁棒性有待验证。

### 6.5 2.52× 加速的分解

仔细分析数据会发现，PD-VLA 的 2.52× 频率提升中，**action chunking 贡献了约 1.98×**（减少推理次数），**并行解码仅贡献约 1.27×**（加速单次推理）。这意味着并行解码本身的加速收益相对温和，核心加速来自 action chunking。但论文的价值在于证明了：在 action chunking 拉长单次推理时间的情况下，并行解码能有效缓解这一副作用。

---

## 参考

- [Jacobi Decoding for LLM](https://arxiv.org/abs/2305.10427)（Santilli et al., ACL 2023）— 首次将 Jacobi 迭代用于 LLM 并行解码
- [CLLMs](https://arxiv.org/abs/2403.00835)（Kou et al., ICML 2024）— 一致性大语言模型，训练 LLM 适配并行解码
- [ACT](https://arxiv.org/abs/2304.13705)（Zhao et al., RSS 2023）— Action Chunking with Transformers，action chunking 的奠基工作
- [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024)（Black et al., 2024）— Flow Matching VLA，使用 action chunking 架构
- [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025)（Xu et al., NeurIPS 2025）— 跨帧 Token 缓存加速 VLA，PD-VLA 论文中的对比方法
- [FAST](https://arxiv.org/abs/2501.09747)（Pertsch et al., 2025）— 基于 DCT 的动作压缩 tokenization
