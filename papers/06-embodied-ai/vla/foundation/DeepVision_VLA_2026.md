# DeepVision-VLA：视觉基础表征增强——在 VLA 深层恢复视觉敏感性

> **论文**：*Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models*
>
> **作者**：Luo et al.
>
> **机构**：北京大学（Peking University）、香港中文大学（CUHK）
>
> **发布时间**：2026年3月
>
> **arXiv**：[2603.15618](https://arxiv.org/abs/2603.15618)
>
> **状态**：未录用（Preprint）
>
> **分类标签**：`VLA 基础模型` `视觉表征` `Mixture-of-Transformers` `Token 剪枝` `视觉敏感性衰减`

---

## 一句话总结

通过 Grad-CAM 分析发现 VLA 模型深层的**视觉敏感性逐层衰减**（视觉 token 的任务相关度在深层 Transformer 中下降），提出 **DeepVision-VLA**：以 VL-MoT（Vision-Language Mixture-of-Transformers）将冻结的 DINOv3 视觉专家与深层 LLM 共享 QKV 注意力，并配合 AGVP（Action-Guided Visual Pruning）用浅层动作-视觉注意力图选择高质量 Top-K 视觉 token，在 RLBench 10 任务上达 83%（vs HybridVLA 74%、QwenVLA-OFT 69%），真实世界 91.7%（vs π₀.₅ 84.2%）。

---

## 二、问题与动机

### 2.1 视觉敏感性衰减现象

现有 VLA 方法（OpenVLA、π₀、QwenVLA-OFT 等）的标准架构是：用 Vision Encoder 提取视觉 token，拼接语言 token 后送入大语言模型（LLM）主干，再由动作头解码输出动作。视觉信息仅在**输入端注入**，而 LLM 的深层 Transformer 层本质上是为语言推理设计的——语言 token 会逐渐主导注意力。

本文通过两类实验定量证明了这一问题：

1. **Grad-CAM 分析**：对多个 VLA 模型（OpenVLA、π₀、QwenVLA-OFT）进行逐层 Grad-CAM 可视化，发现模型对任务相关视觉区域（如目标物体、操作区域）的响应在中深层开始显著下降。

2. **ROI 掩码 MSE 实验**：对视觉输入的关键兴趣区域（ROI）应用随机掩码，测量不同层输出的 MSE 变化量。浅层对掩码高度敏感（MSE 变化大），深层接近不敏感——说明视觉信息到了深层已被"稀释"。

### 2.2 为什么深层视觉信息重要

动作预测最终由动作头或最后几层隐状态驱动，**视觉敏感性在深层的衰减直接导致动作解码时视觉信息不足**，尤其在需要精细空间感知的任务（如对准、扫取等）中表现明显。

现有的缓解方案（如多视角输入、3D 点云、中间层视觉监督 SF 等）要么增加输入复杂度，要么引入额外训练目标，并未从根本上解决深层视觉退化的问题。

---

## 三、预备知识

### 3.1 QwenVLA-OFT 基线

DeepVision-VLA 的骨架基于 **QwenVLA-OFT**：以 Qwen3-VL（4B 参数）为 VLM 主干，采用**并行动作解码**（动作头与语言解码并行运行）和 L1 回归损失直接预测连续动作。相比自回归 token 化方法，QwenVLA-OFT 推理延迟更低、动作精度更高。

### 3.2 DINOv3 视觉基础模型

DINOv3 是 DINOv2 的升级版，参数规模约 0.8B，通过自监督视觉预训练在 patch 级别学习丰富的局部特征。其 patch token 对空间结构、纹理和实例边界保持高敏感性，这正是 VLA 深层所缺失的能力。

### 3.3 Mixture-of-Transformers（MoT）

MoT 架构（已在 GR-3 等 VLA 中使用）的核心思想是：让两个不同来源的 Transformer 序列在**共享的注意力计算**中交互，而非通过独立的交叉注意力模块——这样两个序列的 token 能在同一个注意力矩阵中相互"看到"彼此。

---

## 四、核心方法

### 4.1 整体架构

DeepVision-VLA 的改动集中在 QwenVLA-OFT 主干的**深层**：

> 1. 对浅层（前 $L - n$ 层）：不做修改，视觉 token 和语言 token 正常经过 LLM 层。
>
> 2. 在浅层输出处，利用浅层动作-视觉注意力图（AGVP）对视觉 token 剪枝，选出 Top-K 个最重要的视觉 token 位置，作为 DINOv3 视觉专家的输入候选。
>
> 3. 对深层（后 $n$ 层）：DINOv3 视觉专家的 token（$E$）与 VLA 主干的 token（$Z$）合并后，通过 VL-MoT 共享 QKV 注意力联合处理。
>
> 4. 动作头接收深层处理后的动作 token 隐状态，输出连续动作。

### 4.2 VL-MoT：深层视觉-语言 Mixture-of-Transformers

VL-MoT 的关键思想是，在 LLM 深层让 VLA 主干 token 和 DINOv3 视觉专家 token 共享同一套 QKV 线性投影，从而在同一个注意力空间中互相感知。

设 $Z \in \mathbb{R}^{N_Z \times d}$ 为第 $l$ 层（深层）VLA 主干 token，$E \in \mathbb{R}^{N_E \times d}$ 为 DINOv3 视觉专家 token（已过专家自身的前 $n$ 层），两者拼接后计算联合注意力：

$$Q = [Q_E;\ Q_Z], \quad K = [K_E;\ K_Z], \quad V = [V_E;\ V_Z]$$

其中 $Q_E = EW_Q$，$Q_Z = ZW_Q$（共享同一套投影矩阵 $W_Q, W_K, W_V$）。联合自注意力输出后，$E$ 和 $Z$ 的子序列分别继续流向视觉专家和 VLA 主干各自的 FFN。

**大白话**：在深层，VLA 的 token 和 DINOv3 的 token 坐在同一个"会议室"里开注意力会议——VLA 的动作 token 可以直接"看到"并询问 DINOv3 的视觉 patch token，而 DINOv3 也能感知 VLA 的上下文，形成双向视觉-语言增强。

**为什么只耦合最后 $n$ 层？** 浅层 LLM 主要做多模态融合和基本语义理解，视觉信息尚未严重退化；深层才是动作解码的关键路径，针对性地在深层引入视觉专家效益最大，同时避免了全层耦合导致的计算开销翻倍。

### 4.3 AGVP：动作引导视觉剪枝

DINOv3（0.8B）的 patch token 数量较多，若不加筛选地全部注入深层注意力，计算开销显著增加。AGVP 利用浅层已有的注意力图来判断哪些视觉 token 对动作预测最有价值。

**构建注意力图**：在浅层（第 $1$ 到 $L-n$ 层），提取所有**动作 token 对视觉 token 的注意力权重**，在动作 token 维度和层维度上做平均：

$$\tilde{m} = \frac{1}{(L-n) \cdot N_a} \sum_{l=1}^{L-n} \sum_{i=1}^{N_a} \alpha_{l,i}^{\text{act} \to \text{vis}}$$

其中 $\alpha_{l,i}^{\text{act} \to \text{vis}} \in \mathbb{R}^{N_v}$ 是第 $l$ 层第 $i$ 个动作 token 对 $N_v$ 个视觉 token 的注意力权重。

**插值与选择**：$\tilde{m}$ 的分辨率与 VLA 原始视觉 token 对齐，将其插值到 DINOv3 的 patch 分辨率后，选取最高注意力的 Top-K 个位置：

$$S_K = \text{TopK}(\tilde{m},\ K)$$

DINOv3 视觉专家仅处理 $S_K$ 对应的 patch，大幅减少进入深层 VL-MoT 的 token 数量。

**大白话**：浅层的动作 token 天然知道"我在看哪里"——把这些注意力图平均起来，就能知道哪些视觉 patch 对预测动作最重要。只把这些"热点"patch 交给深层的 DINOv3 专家处理，既保证质量又节省计算。

### 4.4 训练策略

- **DINOv3 视觉专家冻结**：保留预训练视觉表征的通用性，只训练 VL-MoT 的 QKV 共享投影（以及正常的 QwenVLA-OFT 动作头）。
- **预训练数据**：Open X-Embodiment + DROID + RoboMIND，共 400K+ 条机器人操作轨迹。
- **动作监督**：与 QwenVLA-OFT 相同，使用 L1 回归损失在并行动作解码分支上监督。

---

## 五、实验结果

### 5.1 RLBench 仿真评估（10 任务）

| 方法 | 平均成功率 |
| --- | --- |
| OpenVLA-OFT | ~55% |
| QwenVLA-OFT（基线） | 69% |
| HybridVLA | 74% |
| **DeepVision-VLA** | **83%** |

DeepVision-VLA 比强基线 QwenVLA-OFT 提升 14 个百分点，比 HybridVLA 高 9 个百分点，在 10 个任务上均有改善，尤其在需要精细视觉对准的任务上差距更大。

### 5.2 真实世界评估

| 方法 | 成功率 |
| --- | --- |
| π₀.₅ | 84.2% |
| **DeepVision-VLA** | **91.7%** |

真实世界多任务评估（Franka 机械臂，多种操作原语），DeepVision-VLA 超过 π₀.₅ 约 7.5 个百分点。

### 5.3 典型任务分析

**Sweep to Dustpan（扫入簸箕）** 任务是对视觉空间感知要求极高的任务——需要精确感知扫帚与簸箕的相对位置并连续调整动作。DeepVision-VLA 在该任务上相比 QwenVLA-OFT 基线提升约 **+80%**，直观验证了深层视觉增强对精细操作的价值。

### 5.4 消融实验

| 配置 | RLBench 平均 |
| --- | --- |
| QwenVLA-OFT 基线 | 69% |
| + VL-MoT（无 AGVP，全量视觉 token） | 78% |
| + VL-MoT + AGVP（完整 DeepVision-VLA） | **83%** |
| 仅 AGVP（无 VL-MoT） | 71% |

关键发现：
1. VL-MoT 贡献最大（+9%），是深层视觉恢复的核心机制
2. AGVP 在 VL-MoT 基础上再增 5%，同时降低计算开销
3. AGVP 单独使用提升有限（+2%），说明视觉 token 的质量比数量更重要

---

## 六、局限性与未来方向

1. **视觉专家规模**：DINOv3（0.8B）与 4B VLA 主干相比规模不小，在计算受限场景下引入额外视觉专家可能不实用；未来可探索更轻量的专家设计
2. **超参数 $n$ 和 $K$ 的敏感性**：耦合层数 $n$ 和 Top-K 值的选择对最终性能有影响，不同任务最优值可能不同，自适应选择机制值得研究
3. **长时程任务**：当前评估集中在单步/短时程操作任务，视觉敏感性衰减在长时程任务（涉及多步骤规划和状态追踪）中的影响尚待验证
4. **仅分析视觉退化**：深层 LLM 中视觉退化是问题之一，语言指令追踪在深层的状态也值得类似分析

---

## 七、个人思考

### 7.1 诊断先于设计

DeepVision-VLA 的论文结构值得借鉴：先通过严格的诊断实验（Grad-CAM + ROI 掩码 MSE）**证明问题存在**，再针对性地设计解决方案，而非直觉驱动地"堆积组件"。这种诊断-设计的路径在 VLA 领域的工作中并不多见。

### 7.2 VL-MoT 与 GR-3 的异同

GR-3 的 MoT 架构是 Qwen2.5-VL 与 Action DiT 共享注意力层，目的是让语言推理流和动作流互相增强。DeepVision-VLA 的 VL-MoT 则是 LLM 深层与 DINOv3 视觉专家共享 QKV，目的是在深层维持视觉信息的高保真度。两者都用 MoT 解决了"两种异构信号流在深层分离"的问题，但针对的问题截然不同：前者是语言-动作解耦，后者是视觉退化。

### 7.3 AGVP 与 VLA-Pruner 的比较

VLA-Pruner 在**推理**阶段对视觉 token 剪枝以加速，剪枝信号来自于语义级和动作级注意力的组合。AGVP 同样利用动作-视觉注意力，但目的不同：它是在**训练阶段**为 DINOv3 视觉专家筛选高价值输入。两者的注意力剪枝思路相近，但一个服务于推理加速，一个服务于视觉质量提升。

### 7.4 视觉退化是普遍问题还是架构特异的问题

论文在 OpenVLA、π₀、QwenVLA-OFT 三种不同架构上都观察到了视觉敏感性衰减，说明这不是某一个特定模型的问题，而是**将视觉 token 输入到语言模型主干这一通用设计范式的固有缺陷**。未来若有更系统的理论解释（如为什么 LLM 层会稀释视觉信息），将有助于设计更根本的解决方案。

### 7.5 SF 的互补视角

SF（Spatial Foundation）通过监督中间层视觉 embedding 与 VGGT 3D 表征对齐来增强视觉感知，强调的是"引导视觉 token 学到什么"。DeepVision-VLA 则关注"视觉信息在深层是否还在"，强调的是持续供给。两种方法从不同角度缓解同一类问题，结合使用可能带来进一步提升。

---

## 参考

- **QwenVLA-OFT** / **Qwen3-VL**：DeepVision-VLA 的直接骨架基线，并行动作解码框架
- **DINOv3**（Meta AI）：高容量视觉基础模型，提供丰富局部 patch 特征
- **GR-3**（ByteDance, 2025）：VLA 中 MoT 架构的成功先例（Qwen2.5-VL + Action DiT）
- **SF**（2025）：通过中间层视觉 embedding 与 VGGT 对齐增强 VLA 视觉感知，LIBERO 98.5%
- **VLA-Pruner**（2025）：双层 Token 剪枝（语义级 + 动作级）加速 VLA 推理
- **HybridVLA**：RLBench 评估基线之一，与 DeepVision-VLA 直接对比
- **π₀.₅**（Physical Intelligence, 2025）：真实世界评估的主要对比对象
