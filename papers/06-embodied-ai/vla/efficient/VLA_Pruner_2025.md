# VLA-Pruner：面向 VLA 双系统特性的时序感知双层视觉 Token 剪枝

> **论文**：*VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference*
>
> **作者**：Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, Bo Zhao
>
> **机构**：Shanghai Jiao Tong University、University of Science and Technology of China、Harbin Institute of Technology (Shenzhen)、BAAI
>
> **发布时间**：2025 年 11 月
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2511.16449)

---

## 一句话总结

现有 VLM Token 剪枝方法仅依赖语义级注意力（prefill attention）选择 token，忽视了 VLA 模型内在的"高层语义理解 + 低层动作执行"**双系统特性**，导致丢失对动作生成至关重要的视觉信息。VLA-Pruner 提出**双层重要性准则**（语义级 prefill 注意力 + 动作级 decode 注意力，后者通过时序平滑估计）和**双层 token 选择策略**（最大相关性池化 + 最小冗余过滤），在 50% 剪枝率下甚至**提升**模型性能，87.5% 剪枝率下仍保持 88.9% 相对准确率，全面超越所有 training-free 基线。

---

## 一、问题与动机

### 1.1 VLA 模型的视觉 Token 冗余

VLA 模型处理连续视觉流时，视觉 token 数量（通常 256×n/帧，n 为相机数）远大于文本 token（~30-50）和动作 token（~7-56），且自注意力复杂度与 token 数平方成正比，**视觉 token prefilling 是主要计算瓶颈**。

### 1.2 现有 VLM 剪枝方法的根本缺陷

FastV、SparseVLM、DivPrune 等 VLM token 剪枝方法的核心问题：**仅使用语义显著性指标**（prefill 注意力、text-to-vision 交叉注意力等）来选择 token。

VLA-Pruner 通过注意力分析揭示了一个关键发现：

| 对比对象 | Top-50% 重叠率 | Top-25% 重叠率 | Top-12.5% 重叠率 |
| --- | --- | --- | --- |
| Prefill vs. Action Decode | ~72% | ~59% | ~52% |
| 连续时间步 Action Decode | **~95%** | **~93%** | **~89%** |

**Prefill 注意力和 Action Decode 注意力的关注区域仅约 50-72% 重叠**，且经常低于 30%。Prefill 注意力呈广泛语义分布，而 Action Decode 注意力呈局部聚焦分布。这意味着仅依赖语义级注意力的剪枝方法会**丢弃对动作生成至关重要但语义显著性低的 token**，特别是在高剪枝率下性能急剧下降。

### 1.3 时序连续性的利用

上表同时揭示了另一个重要规律：**连续时间步的 Action Decode 注意力高度重叠**（89-95%）。这种时序连续性使得可以用历史 action attention 估计当前 action attention，解决了 action decode attention 在 prefill 阶段不可用的难题。

### 1.4 VLA-Cache 的局限

VLA-Cache 虽然利用了时序连续性做跨帧缓存，但其 token 选择仍依赖 text-to-vision 注意力，属于**粗粒度的语义级选择**，在高压缩比下效果有限。且缓存机制本身比直接剪枝效率更低。

---

## 二、预备知识

### 2.1 VLA 推理的两阶段分解

VLA 模型推理可分解为两个阶段：

**(1) Vision-Language Prefill 阶段**：查询来自文本和视觉 token，$\mathbf{Q}_{\text{vl}} \in \mathbb{R}^{(N+M) \times d_k}$，注意力矩阵为：

$$\mathbf{A}_{\text{vl}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{vl}} \mathbf{K}_{\text{vl}}^\top}{\sqrt{d_k}}\right)$$

每个视觉 patch 的语义重要性分数：

$$\mathcal{S}_{\text{vl}}[m] = \frac{1}{M+N} \sum_{i=1}^{M+N} \mathbf{A}_{\text{vl}}[i, m], \quad m = 1, \ldots, M$$

**(2) Action Decode 阶段**：查询来自动作 token，$\mathbf{Q}_{\text{act}} \in \mathbb{R}^{K \times d_k}$（自回归 $K=1$，chunk-based $K=\hat{N}$），注意力矩阵为：

$$\mathbf{A}_{\text{act}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{act}} \mathbf{K}_{\text{vl}}^\top}{\sqrt{d_k}}\right)$$

每个视觉 patch 的动作重要性分数：

$$\mathcal{S}_{\text{act}}[m] = \frac{1}{\hat{N}} \sum_{i=1}^{\hat{N}} \mathbf{A}_{\text{act}}[i, m], \quad m = 1, \ldots, M$$

$\mathcal{S}_{\text{vl}}$ 量化语义理解的重要性，$\mathcal{S}_{\text{act}}$ 量化动作执行的重要性——**两者关注不同的视觉区域**。

### 2.2 问题形式化

给定 $M$ 个视觉 token $\mathbf{E}_v$，目标是选择 $\tilde{M} < M$ 个 token 的子集 $\tilde{\mathbf{E}}_v$，使剪枝前后模型输出的差异最小：

$$\min_f \mathcal{L}(\mathcal{P}, \tilde{\mathcal{P}}), \quad \text{s.t.} \; |\tilde{\mathbf{E}}_v| = \tilde{M}$$

其中 $\mathcal{P} = P(A \mid \mathbf{E}_\tau, \mathbf{E}_v)$，$\tilde{\mathcal{P}} = P(A \mid \mathbf{E}_\tau, f(\mathbf{E}_v))$。将 $\mathcal{P}$ 分解为 prefill $\mathcal{P}_{\text{vl}}$ 和 decode $\mathcal{P}_{\text{act}}$ 两部分，仅优化单一层级目标（$\mathcal{L}(\mathcal{P}_{\text{vl}}, \tilde{\mathcal{P}}_{\text{vl}})$ 或 $\mathcal{L}(\mathcal{P}_{\text{act}}, \tilde{\mathcal{P}}_{\text{act}})$）会导致次优剪枝。

---

## 三、核心方法

### 3.1 双层 Token 重要性准则

VLA-Pruner 使用两个维度的重要性度量：

1. **语义级**：当前帧的 prefill 注意力分数 $\mathcal{S}_{\text{vl}}^t$（Eq. 2）
2. **动作级**：当前帧的 action decode 注意力分数 $\mathcal{S}_{\text{act}}^t$（Eq. 4）

关键挑战：**action decode attention 在 prefill 阶段不可用**（因为 decode 还没开始）。

**解决方案：时序平滑估计。** 利用机器人操控的时序连续性，从历史 action attention 估计当前值。

#### 衰减窗口平均（Decaying Window Average）

论文提出比 EMA 更适合 VLA 短时上下文的衰减窗口平均：

$$\hat{\mathcal{S}}_{\text{act}}^t = \sum_{i=1}^{w} \gamma^i \mathcal{S}_{\text{act}}^{t-i}$$

其中 $w$ 为窗口大小（默认 3），$\gamma \in [0, 1]$ 为衰减率（默认 0.8）。

用大白话说：用最近 3 步的 action attention 做加权平均来估计当前步，越近的步权重越大。这比 EMA 更直觉，且对 VLA 操控中短时上下文更关键的场景效果更好。

> **Remark 1**：时序平滑在目标切换等突变场景下可能失效（如从"放汤"切换到"放番茄酱"时，上一阶段的 action attention 无法定位新目标）。这一根本挑战通过下面的双层选择策略来解决。

### 3.2 双层 Token 选择策略

一个直觉的做法是将 $\mathcal{S}_{\text{vl}}$ 和 $\hat{\mathcal{S}}_{\text{act}}$ 归一化后加权求和，取 Top-$\tilde{M}$。但这有三个问题：
- 引入敏感的权重超参数
- 忽略阶段内和阶段间的冗余（如机械臂中间段在两个阶段都显著，但对两者都非必需）
- 隐式假设 $\hat{\mathcal{S}}_{\text{act}}$ 始终可靠

VLA-Pruner 借鉴信息论中的 **mRMR（最小冗余-最大相关性）** 原则，提出 **patch-wise combine-then-filter** 三阶段策略：

#### 阶段 1：双层 Top-k 选择

分别从两个维度选出 Top-$\tilde{M}$ 候选集：

$$\mathcal{C}_{\text{vl}} = \text{Top-}\tilde{M}\left(\{\mathcal{S}_{\text{vl}}[i]\}_{i=1}^M\right), \quad \mathcal{C}_{\text{act}} = \text{Top-}\tilde{M}\left(\{\hat{\mathcal{S}}_{\text{act}}[i]\}_{i=1}^M\right)$$

#### 阶段 2：最大相关性池化（Max-Relevance Pooling）

取两个候选集的**并集**：

$$\mathcal{C}_{\text{dual}} = \mathcal{C}_{\text{vl}} \cup \mathcal{C}_{\text{act}}$$

由于两阶段注意力的分歧（Motivation 2），$|\mathcal{C}_{\text{dual}}|$ 通常大于 $\tilde{M}$，确保了对两个层级任务相关性的最大覆盖。

#### 阶段 3：最小冗余过滤（Min-Redundancy Filtering）

在候选池 $\mathcal{C}_{\text{dual}}$ 中求解 **Max-Min Diversity Problem (MMDP)**，找到大小为 $\tilde{M}$ 的子集使最小成对距离最大：

$$\tilde{\mathcal{C}} = \arg\max \left[\min_{i,j \in \mathcal{C}} d(v_i, v_j) : \forall \mathcal{C} \subset \mathcal{C}_{\text{dual}}, |\mathcal{C}| = \tilde{M}\right]$$

距离函数为余弦距离：

$$d(v_i, v_j) = 1 - \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}$$

使用贪心算法求解：以最大次近距离的 token 初始化，迭代添加与已选集最小距离最大的 token，直到达到 $\tilde{M}$ 个。

> **为什么不直接用 DivPrune？** DivPrune 只做多样性最大化（Min-Redundancy），不做相关性最大化（Max-Relevance）。它在全局 token 集上做多样性选择，无法识别对任务关键的 token。VLA-Pruner 先通过阶段 1-2 确保候选池包含双层级关键 token，再在此基础上做多样性过滤，兼顾相关性和多样性。

> **Remark 1 的解决**：当时序平滑失效（如目标切换）时，$\mathcal{C}_{\text{act}}$ 可能不准确，但 $\mathcal{C}_{\text{vl}}$ 会通过 prefill 注意力"兜底"覆盖新目标，加上多样性最大化进一步保留这些 token。

### 3.3 实现细节

- **剪枝层**：在 Transformer 第 $K=3$ 层执行剪枝
- **注意力来源**：prefill 使用最后一层注意力；action attention 使用后半层平均（降噪）
- **预热**：前 $w$ 步不剪枝，积累足够的 action attention 历史
- **适用架构**：所有具有 action-to-vision 交叉注意力的 VLA 架构，包括：
  - 自回归策略（OpenVLA）
  - Action-chunk 解码器（OpenVLA-OFT）
  - 扩散头策略（π₀，对 flow matching 步取平均 action attention）

### 3.4 计算复杂度

剪枝后 FLOPs 比：

$$r_{\text{FLOPs}} = \frac{(K-1)C(\mu) + (T-K+1)C(\tilde{\mu})}{T \cdot C(\mu)}$$

其中 $\mu = N + M$ 为原始序列长度，$\tilde{\mu} = N + \rho M$ 为剪枝后序列长度，$\rho$ 为保留比例。由于注意力的平方项，FLOPs 降低与 $\rho^2$ 近似成正比。

额外开销：时序平滑 $O(wM)$/步，贪心 MMDP $O((\rho M)^2)$/prefill，均远小于 Transformer 主体计算。

---

## 四、实验结果

### 4.1 实验设置

- **VLA 模型**：OpenVLA（7B，自回归）、OpenVLA-OFT（action-chunk）、π₀（flow matching）
- **评估基准**：LIBERO（Spatial/Object/Goal/Long）、SIMPLER（Move Near/Pick Coke Can/Open-Close Drawer）、真实机器人（6-DoF xArm6，4 个任务）
- **基线**：FastV、SparseVLM、DivPrune、VLA-Cache
- **硬件**：NVIDIA RTX 4090

### 4.2 LIBERO 主实验（OpenVLA）

| 方法 | 保留率 | Spatial | Object | Goal | Long | 相对准确率(%) | FLOPs(T) | 加速比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | 100% | 87.6 | 84.6 | 78.6 | 52.2 | 100.0 | 1.906 | 1.00× |
| FastV | 50% | 86.2 | 81.6 | 77.2 | 50.6 | 97.43 | 1.136 | 1.37× |
| VLA-Cache | 50% | 87.1 | 80.7 | 78.6 | 51.8 | 98.48 | 1.384 | 1.23× |
| **VLA-Pruner** | **50%** | **88.2** | **85.8** | **79.4** | **56.4** | **102.45** | **1.139** | **1.33×** |
| FastV | 25% | 81.6 | 69.6 | 71.6 | 43.8 | 87.62 | 0.757 | 1.67× |
| VLA-Cache | 25% | 78.1 | 73.2 | 70.2 | 45.5 | 88.08 | 0.961 | 1.44× |
| **VLA-Pruner** | **25%** | **85.4** | **82.5** | **78.4** | **51.8** | **98.48** | **0.758** | **1.63×** |
| FastV | 12.5% | 62.0 | 58.5 | 55.8 | 18.8 | 63.08 | 0.568 | 1.88× |
| VLA-Cache | 12.5% | 52.5 | 50.1 | 52.0 | 15.1 | 54.79 | 0.710 | 1.62× |
| **VLA-Pruner** | **12.5%** | **80.2** | **78.4** | **69.0** | **42.8** | **88.90** | **0.571** | **1.83×** |

**关键发现**：

1. **50% 剪枝率下性能反超原模型**：VLA-Pruner 达到 102.45% 相对准确率，LIBERO-Long 提升 4.2pp（52.2→56.4），说明精确去除冗余 token 反而有利于决策
2. **87.5% 剪枝率下仍保持 88.9% 准确率**，超越次优基线 34.39pp
3. **25% 保留率的 VLA-Pruner 超越 50% 保留率的所有基线**，同时更快且 FLOPs 更低

### 4.3 OpenVLA-OFT 结果

VLA-Pruner 在 OpenVLA-OFT（512 visual tokens）上表现一致：50% 保留率达 101.05% 相对准确率，12.5% 保留率仍有 88.27%，1.99× 加速。

### 4.4 π₀ 结果

| 方法 | 保留率 | Avg. 成功率(%) | 相对准确率(%) | 加速比 |
| --- | --- | --- | --- | --- |
| Vanilla | 100% | 94.03 | 100.0 | 1.00× |
| VLA-Pruner | 50% | 95.07 | 101.10 | 1.51× |
| VLA-Pruner | 25% | 92.56 | 98.44 | 1.73× |
| VLA-Pruner | 12.5% | 83.08 | 88.36 | 1.95× |

在 flow matching 架构上同样有效，25% 保留率仍超越所有基线在 50% 保留率的性能。

### 4.5 SIMPLER 结果

75% 剪枝率下，VLA-Pruner 整体保留 96.8% 性能（50.4% vs 52.1%），远超 FastV（73.1%）和 VLA-Cache（77.2%），展示跨环境泛化能力。

### 4.6 真实机器人结果

在 6-DoF xArm6 上 75% 剪枝率下，VLA-Pruner 在 4 个任务（Can Stack、Cup Pour、Cube Place、Bottle Place）上均取得最高的相对准确率，验证了真实世界部署的可行性。

### 4.7 消融实验

**双层设计验证**（OpenVLA，LIBERO 平均）：

| 变体 | 策略 | 75% 剪枝率性能 | 87.5% 剪枝率性能 |
| --- | --- | --- | --- |
| Prefill-only | 仅语义注意力 | ~67 | ~55 |
| Action-only | 仅动作注意力 | ~63 | ~48 |
| Score-fusion | 加权求和 | ~65 | ~52 |
| **VLA-Pruner** | **双层选择** | **~75** | **~68** |

- Prefill-only 保留高层语义但缺失低层控制细节
- Action-only 改善短时域控制但牺牲语义理解和任务规划
- Score-fusion 甚至不如 Prefill-only
- 只有双层选择策略在两个维度都能保持性能

**窗口大小消融**：$w=3$ 最优，$w=1$（仅用上一步）性能下降明显，验证了短时序平滑的必要性。$w=4$ 与 $w=3$ 接近，说明过远的历史帮助有限。

**衰减率消融**：$\gamma \in \{0.6, 0.7, 0.8, 0.9\}$ 结果稳定，$\gamma=0.8$ 最优。$\gamma=0.0$（无衰减，等权平均）性能下降。

**剪枝层消融**：$K=3$ 最优（74.53%），$K=2$ 过早剪枝信息不足（58.48%），$K=4, 5$ 略差且 FLOPs 更高。

---

## 五、局限性与未来方向

### 5.1 动态场景下的局限

时序平滑依赖短期时序连续性假设。在高度动态场景（如腕部相机的自我中心视角、传送带环境）中，快速视角变化和物体运动可能违反这一假设，模糊 action attention 的急剧变化。

### 5.2 未来方向

- **自适应预测模块**：根据视觉动态幅度或 action attention 方差自适应调整有效窗口
- **轻量时序注意力网络**：学习从历史直接预测每个 token 的重要性
- **与其他加速方法正交组合**：VLA-Pruner 的 token 剪枝可与层剪枝、扩散步缓存等方法叠加

---

## 六、个人思考

### 6.1 "双系统"洞察的价值

VLA-Pruner 最核心的贡献不是某个具体的技术组件，而是**揭示并量化了 VLA 推理的双系统特性**——prefill 和 decode 关注不同的视觉区域。这一发现解释了为什么所有 VLM 剪枝方法在 VLA 上效果不佳（特别是高剪枝率），也为后续 VLA 加速工作提供了基本的设计原则。

### 6.2 与 VLA-Cache 的对比

| 维度 | VLA-Cache | VLA-Pruner |
| --- | --- | --- |
| 核心策略 | 缓存静态 token 的 KV | 直接剪枝冗余 token |
| Token 选择信号 | text-to-vision 注意力（语义级） | prefill + action attention（双层级） |
| 时序利用 | 像素级 patch 相似度 | action attention 时序平滑 |
| 效率 | 缓存机制本身有额外开销 | 直接剪枝更高效 |
| 高压缩比表现 | 12.5% 保留率准确率 54.79% | 12.5% 保留率准确率 **88.90%** |

VLA-Pruner 的核心优势在于**双层级信号**，使其在高压缩比下大幅优于仅用语义级信号的 VLA-Cache（差距 34pp）。两者理论上可以互补：VLA-Pruner 做 token 剪枝，VLA-Cache 做剩余 token 的跨帧 KV 缓存。

### 6.3 "剪枝反而提性能"现象

VLA-Pruner 在 50% 剪枝率下性能超过原模型（102.45%），与 [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) 在真实机器人上的观察一致。这进一步证实：**冗余视觉 token 不仅浪费算力，还可能干扰模型决策**。通过精确去除无关 token，模型能更专注于任务关键区域。这一现象在 LIBERO-Long（长时域多阶段任务）上最为明显（+4.2pp），说明冗余 token 在复杂任务中的干扰效应更大。

### 6.4 与 EfficientVLA 的互补性

[EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) 是多维度协同加速（层剪枝 + token 选择 + 扩散步缓存），但其 token 选择仍基于语义注意力。VLA-Pruner 可以直接替换 EfficientVLA 的 token 选择模块，在不改变其他加速策略的前提下获得更好的 token 剪枝效果。论文中也展示了在 25% 保留率下 VLA-Pruner 仅靠 token 剪枝就超越了 EfficientVLA 完整框架（50% token + 层剪枝）的性能，且 FLOPs 更低。

---

## 参考

- [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) — 训练无关跨帧 Token 缓存，VLA 加速基线
- [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) — 多维度结构化 VLA 加速框架
- [LAC](/papers/06-embodied-ai/vla/efficient/LAC_2026) — 可学习自适应 Token 缓存
- [DivPrune](https://arxiv.org/abs/2412.15108) — 基于多样性的 VLM Token 剪枝（CVPR 2025）
- [FastV](https://arxiv.org/abs/2403.06764) — 基于早期层注意力的 VLM Token 剪枝（ECCV 2024）
- [SparseVLM](https://arxiv.org/abs/2410.04417) — 基于 text-to-vision 注意力的 VLM Token 稀疏化（ICML 2025）
- [OpenVLA](https://arxiv.org/abs/2406.09246) — 开源 VLA 基座模型
- [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024) — Flow Matching VLA 基础模型
