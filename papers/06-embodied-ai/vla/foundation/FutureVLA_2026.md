# FutureVLA：联合视觉-运动预测建模——解耦双流先验增强 VLA

> **论文**：*FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model*
>
> **作者**：Xiaoxu Xu, Hao Li, Jinhui Ye, Yilun Chen, Jia Zeng 等
>
> **机构**：北京航空航天大学、上海 AI Lab、中国科学技术大学、香港中文大学
>
> **发布时间**：2026年3月
>
> **🔗** [arXiv](https://arxiv.org/abs/2603.10712) | [PDF](https://arxiv.org/pdf/2603.10712)
>
> **分类标签**：`VLA` `未来预测` `视觉-运动解耦` `3D-VAE` `潜在对齐`

---

## 一句话总结

FutureVLA 提出**联合视觉-运动预测建模**（JVPM）框架，通过结构性解耦视觉流和运动流的监督目标、门控交叉注意力做条件化交互，从连续多帧视频中提取物理接地的联合视觉运动嵌入，再通过潜在嵌入对齐将时序先验迁移给下游 VLA，推理时无需未来帧输入，SimplerEnv Google Robot 平均成功率达 80.1%，真实机器人超越 π₀ 达 26.7%。

---

## 一、问题与动机

### 1.1 VLA 缺乏物理接地的未来预测能力

当前 VLA 模型主要依赖当前观测和语言指令来预测动作，缺乏对**未来环境动态**的理解。为弥补这一不足，近期工作尝试为 VLA 引入"未来指导"，大致分两条路线：

- **显式未来指导**（WorldVLA、DreamVLA）：直接预测未来视频帧或结构化视觉表示，再用作条件化信号
- **隐式未来指导**（LAPA、UniVLA、Villa-X）：学习未来观测的潜在向量，用于增强当前决策

### 1.2 两个根本缺陷

FutureVLA 指出这两条路线各有一个核心问题：

**问题一：视觉主导（Visual Dominance）**

显式和隐式方法都严重依赖视觉重建损失。由于视觉信号的信息密度远高于动作信号，联合训练时视觉重建目标会**主导**嵌入空间，模型把大量容量浪费在任务无关的视觉细节上（背景纹理、光照变化），静态场景上下文和动态动作意图纠缠不清。

用大白话说：模型花了 90% 的精力去记忆"场景长什么样"，只花 10% 的精力理解"该怎么动"。

**问题二：时序不连续（Temporal Discontinuity）**

隐式方法通常只使用稀疏帧对（如首尾两帧）来提取未来向量。但 VLA 输出的是连续多步的 action chunk（如 16 步动作），稀疏帧对**无法与动作块的时间粒度对齐**——中间时刻的运动信息完全丢失。

### 1.3 FutureVLA 的核心思路

FutureVLA 同时解决这两个问题：

| 问题 | 现有方法 | FutureVLA |
| --- | --- | --- |
| 视觉主导 | 单一重建目标，视觉和运动纠缠 | **双流解耦监督** + **门控条件化交互** |
| 时序不连续 | 稀疏帧对 | **连续 17 帧视频**经 3D-VAE 编码 |

---

## 二、预备知识

### 2.1 3D-VAE 时序视频编码

FutureVLA 使用 WAN（视频生成模型）的 **3D-VAE** 作为视觉分词器。与传统 2D-VAE 逐帧编码不同，3D-VAE 在时间维度上也进行卷积压缩，能捕获帧间的运动连续性。

**输入约束**：帧数必须满足 $4N+1$ 的形式（如 5, 9, 13, 17 帧），这是 3D-VAE 的时间下采样架构决定的。FutureVLA 选择 $N=4$，即 17 帧。

编码后得到时序 token 矩阵 $V \in \mathbb{R}^{T' \times H' \times W' \times d}$，展平为 $V \in \mathbb{R}^{1960 \times 48}$（5 个时间步 × 14 × 28 个空间位置 × 48 维通道）。

### 2.2 Action Chunk

VLA 通常不是逐帧预测单步动作，而是一次性预测未来多步的动作序列 $\mathbf{a}_{t:t+H}$，称为 action chunk。这与 3D-VAE 编码的时间跨度天然对齐——17 帧对应一个完整的 action chunk 时间窗口。

### 2.3 两种 Action Head

FutureVLA 支持两种动作解码风格：

- **OFT-style**：基于 ResNet 的回归头，使用 MAE loss 直接预测动作值
- **GR00T-style**：基于条件流匹配（Conditional Flow Matching）的生成式动作头，将噪声经多步去噪映射为动作

---

## 三、核心方法

FutureVLA 采用两阶段训练范式：**联合视觉运动预训练 → 潜在嵌入对齐后训练**。

### 3.1 阶段一：Joint Visuomotor Pretraining（JVPM）

这是 FutureVLA 的核心贡献——学习一个同时编码未来视觉信息和运动意图的联合表示。

#### 3.1.1 视觉分词

给定未来连续 17 帧视频 $\{I_t, I_{t+1}, \dots, I_{t+16}\}$，经冻结的 WAN 3D-VAE 编码为紧凑的时序 token 序列：

$$V = \text{3D-VAE}(I_{t:t+16}) \in \mathbb{R}^{1960 \times 48}$$

#### 3.1.2 Transformer 编码与双流分裂

将 $V$ 送入 Transformer 编码器，经过若干层自注意力后，将 token 序列**均分为两组**：

- **视觉 token** $V_n \in \mathbb{R}^{980 \times d}$：负责捕获静态场景几何
- **运动 token** $M_n \in \mathbb{R}^{980 \times d}$：负责捕获动态运动意图

#### 3.1.3 Joint Visuomotor Gating（核心机制）

分裂后的两组 token 进入一个**迭代交互**过程（共 3 轮），每轮包含：

**视觉流**：

$$V_n' = V_n + \text{SelfAttn}(V_n)$$

纯自注意力更新，保持视觉表征的独立性。

**运动流**：

$$M_n' = M_n + \text{SelfAttn}(M_n)$$

先做自注意力提取运动模式，再通过**门控交叉注意力**查询视觉 token 获取空间约束：

$$M_{n+1} = \sigma(r) \odot \text{CrossAttn}(Q{=}M_n', K{=}V_f, V{=}V_f) + M_n'$$

其中：
- $r$ 是**可学习标量参数**（初始化为 0）
- $\sigma(\cdot)$ 是 sigmoid 函数
- $V_f$ 是视觉流的最终输出

**设计直觉**：$r$ 初始为 0 意味着 $\sigma(0) = 0.5$，门控适度开放；训练过程中模型自动学习视觉约束应对运动预测贡献多少。这避免了视觉信号一开始就淹没运动信号（防止 Visual Dominance）。

经过 3 轮迭代，得到最终的**联合视觉运动嵌入** $M_f$。

#### 3.1.4 双监督解耦

视觉流和运动流使用**完全独立的监督目标**：

$$\mathcal{L}_{\text{JVPM}} = \mathcal{L}_I + \mathcal{L}_A$$

- **视觉损失** $\mathcal{L}_I$：视觉 token $V_f$ 经解码器重建**第一帧**的潜在嵌入（而非所有帧）
- **动作损失** $\mathcal{L}_A$：运动 token $M_f$ 经 action head 预测 action chunk

**为什么重建第一帧而非末帧？** 消融实验表明（Table 6），重建首帧（71.9%）显著优于重建末帧（68.8%）。首帧作为**静态几何锚点**，为运动流提供稳定的空间参考，不会引入末帧中可能出现的视角变化或遮挡噪声。

### 3.2 阶段二：Latent Embedding Alignment Post-training

阶段一训练好的 JVPM 模型作为 teacher，为下游 VLA 提供时序先验。

#### 3.2.1 VLA 骨干

下游 VLA 使用 Qwen3-VL-4B 作为视觉语言骨干。输入当前观测（单帧）和语言指令，输出中间层特征 $F_r$。

#### 3.2.2 轻量对齐

$F_r$ 经一个**轻量 Transformer adapter** 映射为 $F_a$，使其对齐到 JVPM 产生的联合嵌入 $M_f$：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_A + \beta \|M_f - F_a\|_2$$

其中 $\beta$ 使用 **cosine decay** 调度——训练前期强对齐保证先验注入，后期逐渐松弛让模型自主优化动作预测。

#### 3.2.3 推理时无需未来帧

关键优势：推理时 JVPM 模型完全不参与。VLA 已通过对齐将时序先验内化到自身参数中，输入仍然是**单帧观测 + 语言指令**，不增加任何推理开销。

### 3.3 方法总结

> 1. **视觉分词**：未来 17 帧 → 冻结 3D-VAE → 1960 个时序 token
> 2. **双流分裂**：Transformer 编码 → 均分为 980 视觉 token + 980 运动 token
> 3. **门控交互**（×3 轮）：运动 token 经门控交叉注意力查询视觉 token
> 4. **双监督解耦**：视觉 token → 重建首帧；运动 token → 预测动作块
> 5. **潜在对齐**：冻结 JVPM，用 MSE 将 VLA 中间表征对齐到运动嵌入 $M_f$
> 6. **推理**：只用 VLA，单帧输入，零额外开销

---

## 四、实验结果

### 4.1 主实验

#### SimplerEnv Google Robot（Visual Matching）

| 方法 | Drawer | Pick Coke Can | Move Near | Place Apple | Avg. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA-OFT | 30.3 | 79.3 | 58.7 | 21.7 | 47.5 |
| π₀ | 72.0 | 64.7 | 37.3 | 36.7 | 52.7 |
| **FutureVLA-GT** | **81.7** | **93.3** | **64.0** | **81.3** | **80.1** |

FutureVLA-GT 平均成功率 **80.1%**，较 π₀ 提升 27.4%，较 OpenVLA-OFT 提升 32.6%。

#### SimplerEnv WidowX

| 方法 | Put Carrot | Stack Blocks | Put Eggplant | Avg. |
| --- | --- | --- | --- | --- |
| GR00T-N1.5 | 72.0 | 23.2 | 90.4 | 61.9 |
| **FutureVLA-GT** | **89.6** | **32.8** | **93.2** | **71.9** |

#### LIBERO

| 方法 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- |
| π₀ | 91.7 | 96.7 | 85.0 | 93.3 | 94.2 |
| UniVLA | 98.3 | 95.0 | 91.7 | 95.8 | 95.2 |
| **FutureVLA-GT** | **99.2** | **99.2** | **96.7** | **98.3** | **98.3** |

#### 真实机器人（Franka）

| 任务 | π₀ | OpenVLA-OFT | FutureVLA-GT |
| --- | --- | --- | --- |
| 4 任务平均 | 43.3% | 50.0% | **70.0%** |

真实世界平均成功率超越 π₀ 达 **+26.7%**。

#### LIBERO-Plus（鲁棒性）

FutureVLA-GT 总分 **79.7%**，显著超越 OpenVLA-OFT（69.6%），表明 JVPM 先验有助于应对干扰变化。

### 4.2 消融实验

#### Joint Visuomotor Gating 逐步消融（Table 5, WidowX）

| 设置 | Put Carrot | Stack | Eggplant | Avg. |
| --- | --- | --- | --- | --- |
| 无解耦（所有 token 混合预测动作） | 56.0 | 24.0 | 95.2 | 58.4 |
| 解耦但无交互（运动 token 独立） | 68.8 | 31.2 | 96.8 | 65.6 |
| 解耦 + 交叉注意力（无门控） | 82.4 | 25.6 | 94.4 | 67.5 |
| **完整门控交互** | **89.6** | **32.8** | **93.2** | **71.9** |

关键发现：
- 不解耦反而**更差**（58.4 < 无 JVPM 的 62.5），证明混合 token 的未来信息是噪声而非先验
- 门控比裸交叉注意力高 4.4%，说明需要控制视觉信号的流入强度

#### 时序密度消融

| 帧数 | 2 | 5 | 9 | 17 |
| --- | --- | --- | --- | --- |
| Avg. | 64.1 | 65.3 | 68.0 | **71.9** |

性能随帧数**单调递增**，验证了连续时序建模的必要性，2 帧（稀疏帧对）是最差的。

#### 重建目标消融

| 目标 | Avg. |
| --- | --- |
| 重建首帧 | **71.9** |
| 重建末帧 | 68.8 |

首帧作为静态几何锚点，为运动流提供更稳定的空间参考。

#### 视觉扰动鲁棒性

| 方法 | Embedding MSE ↓ |
| --- | --- |
| LAPA | 0.3047 |
| Villa-X | 0.0188 |
| **FutureVLA** | **0.0054** |

FutureVLA 的嵌入对视觉扰动最不敏感，说明解耦成功地将运动信息与视觉噪声隔离。

### 4.3 PAAC 指标

论文提出 **Physics-Aware Action Consistency（PAAC）** 指标：基于 DTW（动态时间规整）对齐预测与真实动作序列，再用 RBF 核度量逐步差异。PAAC 比简单余弦相似度更能反映物理一致性，且与下游任务成功率高度相关。

---

## 五、局限性与未来方向

### 5.1 3D-VAE 约束

冻结的 WAN 3D-VAE 要求输入帧数必须为 $4N+1$，分辨率固定 224×224。预训练 VAE 的质量直接约束时序 token 的信息量，灵活性受限。

### 5.2 均分策略缺乏自适应性

视觉 token 和运动 token 的数量固定为各 980 个（50%/50%）。不同任务可能需要不同比例——精密操作可能需要更多运动 token，场景理解任务需要更多视觉 token。

### 5.3 仅视觉模态

作者在 Limitations 中承认，对于接触密集型任务（如擦白板），仅靠视觉约束不足，需要触觉、力矩等额外模态。

### 5.4 预训练规模

15.6M 帧的预训练数据规模相比 π₀ 等工业级模型偏小，跨具身泛化的上限可能受限。

### 5.5 OFT-style 表现不稳定

Variant Aggregation 设置下 FutureVLA-OT 的 Open/Close Drawer（31.7%）和 Put in Drawer（23.8%）明显低于 GT 版本，说明 OFT-style action head 在某些场景下适配性有限。

### 5.6 β 调度缺乏消融

对齐损失的权重 $\beta$ 采用 cosine decay，但论文未对此调度策略进行消融实验。

---

## 六、个人思考

### 6.1 "解耦再交互"是正确的信息流设计

与 LAPA（直接重建未来帧的潜在向量）和 Villa-X（多帧拼接单一嵌入）相比，FutureVLA 的**先强制解耦、再门控交互**策略更加干净。消融实验（Table 5）最有说服力的一点：不解耦反而比没有未来指导更差，说明混合的未来信息确实是噪声而非先验。解耦是把噪声变成信号的关键一步。

### 6.2 连续帧 vs 稀疏帧对：与 MemoryVLA 的对话

MemoryVLA 通过记忆库存储历史帧来建模时序依赖（回顾过去），FutureVLA 通过 3D-VAE 编码未来帧来提供时序先验（展望未来）。两者方向互补——一个问题是：能否结合两者，既回顾过去又展望未来？

### 6.3 推理零开销的潜在对齐范式

FutureVLA 的两阶段设计——训练时用未来帧、推理时不需要——与 SF 的思路类似（SF 用 VGGT 3D 表征做隐式对齐，推理时也不需要 3D 输入）。这种**"训练时借力、推理时卸力"**的潜在对齐范式正在成为 VLA 领域的一个趋势。

### 6.4 与世界模型方法的对比

WorldVLA 和 DreamVLA 走显式路线，在推理时也需要生成未来帧，计算开销大。FutureVLA 证明了：通过合适的训练范式，未来信息可以被**内化**到 VLA 参数中，不需要在推理时实际生成。这与 RISE、WoVR 等世界模型 RL 方法的"想象空间训练"形成有趣对比。

### 6.5 门控标量 r 的可解释性

可学习标量 $r$ 控制视觉对运动的约束程度，但论文没有分析训练后 $r$ 的收敛值。如果 $\sigma(r)$ 趋近 0，说明运动流不太需要视觉约束；如果趋近 1，说明视觉约束是必要的。分析 $r$ 的值和梯度变化可以提供更多关于视觉-运动交互的洞察。

---

## 参考

- [LAPA: Latent Action Pretraining from Videos](https://arxiv.org/abs/2410.11758) — 隐式未来指导的代表方法，FutureVLA 的主要对比对象
- [UniVLA: Unified Video-Language-Action Model](https://arxiv.org/abs/2412.03066) — 通过稀疏帧对的隐式引导方法
- [OpenVLA-OFT: Open Vision-Language-Action Model with Optimal Fine-Tuning](https://arxiv.org/abs/2501.01339) — OFT-style action head 的来源
- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) — Flow Matching VLA 基础模型
- [GR00T N1.5](https://arxiv.org/abs/2503.14734) — GR00T-style 条件流匹配 action head 的来源
- [WAN: Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314) — 3D-VAE 的来源
- [WorldVLA: World Model Enhanced VLA](https://arxiv.org/abs/2412.12246) — 显式未来指导的代表方法
