# TOPReward：Token 概率作为零样本机器人奖励信号

> **论文**：*TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics*
>
> **作者**：Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan†, Dieter Fox†, Ranjay Krishna†
>
> **机构**：University of Washington, Allen Institute for AI, Amazon, UNC Chapel Hill
>
> **发布时间**：2026年2月
>
> **🔗** [arXiv](https://arxiv.org/abs/2602.19313) | [项目主页](https://topreward.github.io/webpage/)

---

## 一句话总结

提出 TOPReward，不让 VLM 生成数值进度估计，而是直接从 VLM 内部的 token logits 中提取 `True` 令牌的对数概率作为任务完成度信号。在开源 Qwen3-VL-8B 上实现 **0.947 VOC**（ManiRewardBench）和 **0.857 VOC**（OXE），大幅超越 GVL 基线，同时提出包含 130+ 真实任务的 **ManiRewardBench** 基准。

---

## 一、问题与动机

### 1.1 VLA 的奖励瓶颈

VLA 模型（OpenVLA、π₀ 等）的预训练已取得快速进展，但用 RL 进一步改进仍受制于**稀疏奖励**和**低样本效率**。手工设计的奖励不可扩展，学习的奖励模型（如 RoboDopamine、RoboReward）需要大量领域特定训练数据，且跨具身体泛化有限。

### 1.2 GVL 的局限：文本生成的表示瓶颈

当前最先进的 training-free 方法 GVL (Ma et al., 2024) 将进度估计转化为视觉问答：给 VLM 一批打乱的帧，要求输出每帧的 0-1 进度值。然而：

1. **仅在闭源 VLM 上有效**：GVL 在 Gemini/GPT-4 上表现良好（0.541 VOC），但在开源模型上崩溃——Qwen3-VL-8B 仅 0.194，Molmo2-8B 为 −0.016。
2. **根本原因不是视觉理解不足**：论文假设开源 VLM 的失败源于**文本生成的表示瓶颈**——指令遵循不一致 + 数值 token 的内在偏差（LLM 生成精确数字的校准很差）。

### 1.3 核心洞察

> VLM 不是不懂进度，而是不会"说"进度。如果绕过文本生成，直接读取模型内部的"信念"，开源 VLM 其实具备强大的进度估计能力。

---

## 二、预备知识

### 2.1 进度估计作为时序价值函数

进度估计本质上是一个**时序价值函数**：给定指令 $x$ 和视频轨迹 $\tau_{1:T} = (I_1, \ldots, I_T)$，为每个前缀 $\tau_{1:t}$ 生成标量 $p_t \in [0, 1]$，要求随任务完成度单调递增。这与 Universal Value Function (Schaul et al., 2015) 的原则一致。

### 2.2 GVL 的做法

GVL 将进度估计框架为视觉问答：把帧打乱后输入 VLM，要求模型为每帧输出数值进度分数。依赖模型的指令遵循能力和数值生成精度。

### 2.3 Value-Order Correlation (VOC)

$$\text{VOC} = \text{rank-correlation}\big(\text{argsort}(s_{t_1}, s_{t_2}, \ldots, s_{t_K}),\; (t_1, t_2, \ldots, t_K)\big)$$

即预测值排序与时间顺序的 Spearman 秩相关，范围 $[-1, 1]$，1 表示完美对齐。

---

## 三、核心方法

### 3.1 Token 概率作为奖励（核心思想）

**关键转变**：不要求 VLM 生成数值，而是问一个**二元完成判断**，从模型内部的 token logits 提取答案。

构造 prompt：

> `<|video|> The above video shows a robot manipulation trajectory that completes the following task: {INSTRUCTION}. Decide whether the above statement is True or not. The answer is: {a}`

然后计算 `True` token 的对数概率：

$$r_t = \log p_\theta(a \mid c(\tau_{1:t}, u))  \tag{1}$$

其中 $a$ = "True"，$c(\tau_{1:t}, u)$ 是视频条件化的文本上下文。

**为什么选 True？** 论文对比了多种候选 token（True、False、Yes、No 等），发现 `True` token 在成功/失败轨迹之间展现出**最大的绝对概率差异**（差距约为 1.00×10⁻⁶），远超其他候选。

**为什么不用 chat template？** 消融实验表明，添加 chat template 会导致 Qwen3-VL-8B 的 VOC 从 0.945 暴跌至 0.500（−47.1%）。论文假设这是因为进度估计任务更接近**预训练的下一个 token 预测目标**，而非对话场景。

### 3.2 前缀采样与归一化

**前缀采样**：在轨迹上均匀选取 $K$ 个前缀长度 $\{t_k\}_{k=1}^K$，对每个前缀做一次模型前向传播，得到 $\{r_{t_k}\}$。

**Min-Max 归一化**：由于 $\log p \in (-\infty, 0]$，需映射到 $[0, 1]$：

$$s_{t_k} = \frac{r_{t_k} - \min_j r_{t_j}}{\max_j r_{t_j} - \min_j r_{t_j} + \varepsilon} \tag{2}$$

### 3.3 密集奖励用于下游任务

当需要逐步奖励（如用于 behavior cloning 加权）时，使用进度增量构造密集信号：

$$\Delta_{t_k} = \text{clip}\big(\tau \cdot \exp(s_{t_k} - s_{t_{k-1}}),\; \min=0,\; \max=\delta_{\max}\big) \tag{3}$$

其中 $\tau$ 控制好/坏动作的权重差异，$\delta_{\max}$ 防止某些动作获得过大权重。

### 3.4 替代公式的探索

论文还尝试了另一种方案——评估生成**整个指令文本**的概率：

$$r_t = \sum_i \log p_\theta(\text{inst}_i \mid c(\tau_{1:t}, u, \text{inst}_{<i})) \tag{6}$$

但效果不如 `True` token 方案。原因是模型会对视频中出现的实体赋予高概率（如看到苹果就给 "apple" 高概率），无论任务是否完成，从而破坏进度估计。

---

## 四、ManiRewardBench

论文提出 **ManiRewardBench**，一个用于评估机器人操作奖励模型的基准：

| 数据集 | Episodes | Tasks |
| --- | --- | --- |
| Lerobot | 150 | 22 |
| Lerobot failed | 156 | 23 |
| Franka | 150 | 51 |
| Bimanual YAM | 97 | 20 |
| Single-arm YAM | 100 | 20 |
| **Total** | **653** | **130 unique** |

**关键特性**：
- **4 种机器人平台**（Franka、SO-100/101、单臂 YAM、双臂 YAM）
- **子任务级时序标注**：每个 episode 标注了子任务的 start/end 时间，支持细粒度进度评估
- **包含失败轨迹**：23 个任务的 156 条 episode 包含成功和失败两类
- **任务多样性**：涵盖多步推理、精细操作、柔性物体处理、抽象/符号任务

---

## 五、实验结果

### 5.1 大规模零样本进度估计

#### Open X-Embodiment（39 个数据集，每个 20 episodes）

| 方法 | Molmo2-8B | Qwen3-VL-8B | Gemini-2.5-Pro |
| --- | --- | --- | --- |
| GVL | −0.016 | 0.194 | **0.541** |
| TOPReward | **0.417** | **0.857** | 0.433 |

**Qwen3-VL 上提升 +0.663 VOC**，将开源模型从"不可用"推到"接近 Gemini GVL"。

#### ManiRewardBench（113 tasks，497 episodes）

| 数据集 | Qwen3-VL GVL | Qwen3-VL TOPReward | Gemini GVL | Gemini TOPReward |
| --- | --- | --- | --- | --- |
| Lerobot | 0.332 | **0.954** | 0.620 | 0.578 |
| Franka | 0.242 | **0.942** | 0.695 | 0.448 |
| Bimanual YAM | 0.164 | **0.947** | 0.566 | 0.546 |
| Single-arm YAM | 0.544 | **0.945** | 0.752 | 0.488 |

Qwen3-VL + TOPReward 在所有 4 个数据集上达到 **0.942–0.954**，跨平台一致性极高。Gemini 上 TOPReward 反而不如 GVL，原因是 Gemini API 强制添加 chat template。

### 5.2 成功检测

**VOC 的失败模式**：VOC 只衡量排序一致性，无法区分"做完"与"做到一半就停下"。一条 30% 完成度后停滞的轨迹，只要预测值保持有序，VOC 仍可达 ≥0.85。

| 方法 | Qwen3-VL-8B | Gemini-2.5-Pro |
| --- | --- | --- |
| GVL (VOC) | 0.519 | 0.823 |
| TOPReward (log-prob) | **0.654** | **0.826** |

GVL 在 Qwen3-VL 上的成功检测几乎是随机的（0.519 AUC），TOPReward 提升 +0.135。这是因为 TOPReward 直接测量指令被满足的概率，失败轨迹自然获得更低分数。

### 5.3 真实世界 Advantage-Weighted Behavior Cloning

在 SO-100 单臂机器人上，用 TOPReward 计算优势权重进行 AWR 微调（$\tau=2.0$, $\delta_{\max}=2.0$）：

$$\mathcal{L}_{\text{AWR}} = \mathbb{E}_{p(a|o), q(a_t|a)}\left[\Delta_t \cdot \|v_\theta(a_t, t \mid o) - (a - \epsilon)\|^2\right] \tag{5}$$

| 任务 | Pretrained | BC | TOP-AWR |
| --- | --- | --- | --- |
| Place toy car in box | 1 | 2 | **3** |
| Stack red cube on green cube | 1.33 | 1 | **2.33** |
| Put pen into cup | 1.67 | 5.67 | **6.33** |
| Place doll in box | 0 | 7 | **10** |
| Pick up cube | 4 | 7 | **10** |
| Put cube in cup | 4 | 6 | **9** |

分数为部分成功率（10 次试验中完成的子任务比例之和，满分 10）。TOP-AWR 在所有 6 个任务上一致超越 BC。

### 5.4 Chat Template 消融

| 数据集 | Qwen3-VL Base | +Chat | Molmo2 Base | +Chat |
| --- | --- | --- | --- | --- |
| Bimanual YAM | 0.947 | 0.269 | 0.570 | 0.408 |
| Franka | 0.943 | 0.528 | 0.696 | 0.615 |
| Single-arm YAM | 0.946 | 0.703 | 0.691 | 0.546 |
| **Mean** | **0.945** | **0.500** | **0.652** | **0.523** |
| **Δ** | | **−47.1%** | | **−19.8%** |

Chat template 对 logit-based 方法有灾难性影响，这解释了 Gemini API（强制 chat template）上效果不佳的原因。

---

## 六、局限性与未来方向

1. **视觉感知继承自 VLM**：精细空间推理任务（精确对齐、小物体操作）可能获得嘈杂的进度估计。
2. **Min-Max 归一化是 per-episode 的**：无法直接跨轨迹比较绝对进度值，限制了 dataset-level 排序和过滤的精度。
3. **对 prompt 格式敏感**：chat template 导致性能暴跌 47%，意味着方法高度依赖 prompt 与预训练分布的对齐，换模型可能需要重新调试。
4. **计算成本**：每个前缀需要一次完整的 VLM 前向传播，K=20 的前缀采样意味着 20 次推理。
5. **仅验证了 behavior cloning 场景**：AWR 实验使用 50 条演示 + TOPReward 加权，但未在在线 RL 闭环中验证。
6. **ManiRewardBench 访问受限**：为防止数据泄漏，数据集不公开，仅通过受控评估协议访问。

---

## 七、个人思考

### 7.1 与 RoboReward/LRM/Robo-Dopamine/ROBOMETER 的定位对比

TOPReward 在奖励模型谱系中占据独特位置：**零样本、无训练、logit-based**。

| 维度 | TOPReward | RoboReward | LRM | Robo-Dopamine |
| --- | --- | --- | --- | --- |
| 训练需求 | **零** | 45K 样本微调 | LoRA 微调 | 3400+ 小时数据 |
| 奖励粒度 | 前缀级连续 | Episode 级离散 (1-5) | 帧级三维度 | 步级多视角 |
| 跨具身泛化 | **4+ 平台零样本** | 14 种具身体（训练分布内） | 零样本仿真 | 需 one-shot 适配 |
| 核心依赖 | VLM logits 访问 | 负样本数据管线 | 时序单调假设 | 大规模操作数据 |

TOPReward 的核心优势是**零数据、零训练**，但代价是在 Gemini 等不开放 logits 或强制 chat template 的 API 上受限。

### 7.2 "内部表征 > 文本输出"的更深启示

这篇论文的核心发现——VLM 的内部 logits 比其生成的文本更可靠——与 NLP 领域的一系列工作（Kadavath et al., 2022; Burns et al., 2022）一脉相承。用大白话说：**模型比自己"知道"的多**。在机器人领域，这意味着我们不应该仅仅将 VLM 当作"会看图的聊天机器人"来使用（让它输出数字），而应该将其视为**隐式的世界模型**，通过合适的探针（如 token 概率）提取其内部知识。

这也暗示了一个更广泛的研究方向：除了 `True` token 的概率，VLM 的中间层激活、注意力模式等内部表征中可能蕴含更丰富的任务进度信息。

### 7.3 Chat template 敏感性的隐忧

Chat template 导致 47% 的性能下降是一个值得警惕的信号。这说明 TOPReward 本质上依赖于 VLM 在**预训练（而非指令微调）**阶段学到的分布特征。随着开源社区越来越多地发布经过 RLHF/DPO 对齐的模型，预训练权重的原始行为可能被覆盖。方法的长期可用性取决于社区是否继续发布保留原始预训练行为的 base model。

### 7.4 VOC 失败模式的发现很有价值

论文指出 VOC 无法区分"有序但不完整"的轨迹，这对 GVL 等基于排序的方法是根本性的质疑。一条 30% 完成后随机漂移的失败轨迹也能获得高 VOC——这意味着如果只用 VOC 评估，可能会高估奖励模型的实际效用。TOPReward 通过直接测量完成概率绕过了这个问题，同时也为未来的进度估计评估提供了新的 metric 设计思路。

---

## 参考

- **GVL (Ma et al., 2024)**：当前 SOTA 的 training-free 进度估计，将进度估计转化为 VLM 视觉问答，TOPReward 的主要对比基线
- **RoboReward (Lee et al., 2026)**：通用 VLM 奖励模型，反事实重标注 + 时序裁剪训练 episode 级离散进度分
- **Robo-Dopamine (Tan et al., 2025)**：过程奖励模型，3400+ 小时操作数据训练，多视角融合
- **LRM (2026)**：帧级在线奖励模型，三维度设计（对比/进度/完成），同样使用 Qwen3-VL-8B
- **OpenGVL (Budzianowski et al., 2025)**：GVL 的开源基准，揭示开源 VLM 在进度预测上远逊闭源模型
- **AWR (Peng et al., 2019)**：Advantage-Weighted Regression，TOPReward 下游应用所用的离线 RL 框架
- **π₀ (Black et al., 2026)**：Flow Matching VLA 基础模型
- **OpenVLA (Kim et al., 2024)**：开源 VLA 基础模型
