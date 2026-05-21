# MemoryVLA：认知-记忆-动作框架——感知-认知双流记忆赋能长时域操作

> **论文**：*MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation*
>
> **作者**：Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, Gao Huang
>
> **机构**：清华大学、Dexmal、旷视科技（MEGVII）、天津大学、哈尔滨工业大学、StepFun
>
> **发布时间**：2025年8月
>
> **会议**：ICLR 2026
>
> **链接**：[arXiv](https://arxiv.org/abs/2508.19236)
>
> **分类标签**：`VLA` `时序建模` `记忆机制` `扩散策略` `长时域操作`

---

## 一句话总结

MemoryVLA 借鉴认知科学的工作记忆与情景记忆双系统，设计感知-认知记忆库（PCMB）同时存储低层视觉细节和高层语义摘要，通过跨注意力检索、门控融合和合并压缩三步流程为 VLA 注入时序依赖建模能力，在 SimplerEnv-Bridge 上超 CogACT **+14.6**、LIBERO 达 **96.5%**、真实世界长时域任务超 CogACT **+26**。

---

## 一、问题与动机

### 1.1 操作任务的非马尔可夫性

机器人操作本质上是非马尔可夫的——当前最优决策依赖于历史状态和动作。典型例子：Push Buttons 任务中，按钮按下前后视觉外观几乎无区别，仅靠当前帧无法判断"是否已经按过"。主流 VLA 模型（OpenVLA、$\pi_0$）仅基于**单帧观测**决策，在这类任务上表现很差。

### 1.2 朴素时序建模的局限

一种朴素的解决方案是将连续多帧拼接输入 VLM，但存在两个关键问题：

1. **计算瓶颈**：自注意力的二次复杂度严重限制可用时间上下文长度
2. **分布偏移**：多帧输入与模型的单帧预训练分布不一致，导致性能退化

已有尝试包括：RoboFlamingo 用 LSTM 压缩视觉-语言表征为单一 latent token（过于粗糙，丢失细粒度感知历史）；TraceVLA 将历史状态绘制为当前帧上的轨迹线（丢失丰富语义细节）；UniVLA 将过去动作拼入文本提示（仅起 CoT 作用，无法有效利用历史信息）。

### 1.3 MemoryVLA 的认知科学启发

认知科学研究表明，人类通过**双记忆系统**处理操作任务：

- **工作记忆**（前额叶神经活动）：缓冲短期表征，支持即时决策
- **情景记忆**（海马体）：以**逐字表征**（verbatim，精确细节）和**要义表征**（gist，抽象语义）两种形式编码过去经验

MemoryVLA 据此设计：VLM 输出感知 token 和认知 token 构成工作记忆；PCMB 对应海马体，存储低层视觉细节（逐字表征）和高层语义（要义表征），支持长程时序依赖建模。

---

## 二、预备知识

### 2.1 VLA 策略形式化

给定当前 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和语言指令 $L$，策略输出未来动作序列：

$$\mathcal{A} = (a_1, \dots, a_T) = \pi(I, L)$$

每个动作 $a_t = [\Delta x, \Delta y, \Delta z, \Delta\theta_x, \Delta\theta_y, \Delta\theta_z, g]^\top$，包含相对平移、欧拉角旋转和二值夹爪状态 $g \in \{0, 1\}$。

### 2.2 分层 VLA 架构

MemoryVLA 延续 CogACT 等分层架构：VLM 作为骨架编码视觉和语言，扩散/flow 模型作为动作专家生成连续控制。这里的关键创新不在 VLM 或动作专家本身，而在它们之间引入的**记忆模块**。

---

## 三、核心方法

### 3.1 框架总览

MemoryVLA 是端到端的 Cognition-Memory-Action 框架，包含四个核心组件：

1. **Vision-Language Cognition Module**：将当前观测编码为感知 token 和认知 token，构成工作记忆
2. **Perceptual-Cognitive Memory Bank (PCMB)**：存储历史感知细节和认知语义
3. **Memory Retrieval → Fusion → Consolidation**：检索相关历史、自适应融合、合并压缩
4. **Memory-conditioned Diffusion Action Expert**：基于记忆增强的表征生成动作序列

### 3.2 Vision-Language Cognition Module

基于 7B 参数的 **Prismatic VLM**（Karamcheti et al., 2024），在 Open-X Embodiment 大规模跨构型数据集上进一步预训练。

**视觉编码**：并行使用 DINOv2 和 SigLIP 两个视觉编码器处理第三人称 RGB 图像 $I$，拼接特征得到原始视觉 token。通过 **SE-Bottleneck 感知压缩模块**（基于 Squeeze-and-Excitation 注意力）压缩为紧凑的感知 token：

$$p \in \mathbb{R}^{N_p \times d_p}, \quad N_p = 256$$

**认知编码**：原始视觉 token 经线性投影映射到语言嵌入空间，与分词后的指令拼接输入 LLaMA-7B。取 **EOS 位置**的输出作为认知 token：

$$c \in \mathbb{R}^{1 \times d_c}$$

认知 token 代表高层语义摘要（"做什么"），感知 token 保留细粒度视觉细节（"看到什么"）。两者合并构成短期**工作记忆**：

$$M_\text{wk} = \{p \in \mathbb{R}^{N_p \times d_p},\; c \in \mathbb{R}^{1 \times d_c}\}$$

### 3.3 Perceptual-Cognitive Memory Bank (PCMB)

工作记忆仅反映当前时刻，缺乏时序依赖。PCMB 受海马体启发，维护两个并行流：

$$M_\text{pcmb} = \{m^x \mid x \in \{\text{per}, \text{cog}\}\}$$

$$m^x = \{m^x_i \in \mathbb{R}^{N_x \times d_x}\}_{i=1}^L, \quad x \in \{\text{per}, \text{cog}\}$$

- **感知流** $m^\text{per}$：存储 $L$ 个历史时刻的感知 token（细粒度视觉细节）
- **认知流** $m^\text{cog}$：存储 $L$ 个历史时刻的认知 token（高层语义摘要）

每个流最多维护 $L$ 个条目。

#### Memory Retrieval（记忆检索）

当前工作记忆作为双查询，通过**带时间步位置编码的跨注意力**从 PCMB 中检索决策相关历史信息。每个记忆条目附加正弦时间步嵌入 $\text{TE}(\cdot)$：

$$K^x = [m^x_1 + \text{TE}(t_1);\; \dots;\; m^x_L + \text{TE}(t_L)]$$

$$V^x = [m^x_1;\; \dots;\; m^x_L]$$

$$\hat{H}^x = \text{softmax}\left(\frac{q^x (K^x)^\top}{\sqrt{d_x}}\right) V^x, \quad q^x \in \{p, c\},\; x \in \{\text{per}, \text{cog}\}$$

注意力后接 FFN 构成一个 Transformer 层，堆叠 **2 层**得到最终检索嵌入 $H^p$ 和 $H^c$。

**时间步位置编码的作用**：让模型知道每条记忆"是多久以前的"，从而优先检索时间上更相关的历史。消融实验显示去掉时间步 PE 成功率从 71.9% 降到 69.8%。

#### Memory Gate Fusion（门控融合）

用可学习门控自适应地融合检索到的历史信息与当前工作记忆：

$$g^x = \sigma\left(\text{MLP}(\text{concat}[x, H^x])\right)$$

$$\tilde{x} = g^x \odot H^x + (1 - g^x) \odot x$$

其中 $\sigma$ 是 sigmoid，$\odot$ 是逐元素乘法。当历史信息有价值时门控倾向开放（更多使用 $H^x$），否则保留当前表征 $x$。

**直觉理解**：门控机制让模型自适应决定"在多大程度上参考历史"。对于无时序依赖的简单任务，门控可以接近关闭；对于 Push Buttons 这类强时序依赖任务，门控大幅打开。消融显示门控融合（71.9%）显著优于简单加法（67.7%）。

#### Memory Consolidation（记忆合并）

融合后的表征 $\tilde{p}$ 和 $\tilde{c}$ 被同时送往动作专家和更新回 PCMB。当存储条目超过 $L$ 时，在每个流内计算相邻条目的余弦相似度，合并最相似的一对：

$$i^*_x = \arg\max_{i=1,\dots,L-1} \cos(\tilde{x}_i, \tilde{x}_{i+1})$$

$$m^x_{i^*_x} \leftarrow \frac{1}{2}(\tilde{x}_{i^*_x} + \tilde{x}_{i^*_x + 1}), \quad x \in \{\text{per}, \text{cog}\}$$

**设计直觉**：时间上相邻且语义相似的帧往往包含冗余信息（例如机器人静止等待），合并它们不会丢失关键信息。而关键转折点（如"抓取完成→开始放置"）的相邻帧差异大，不会被合并。消融显示 Token Merge（71.9%）远优于 FIFO（66.7%），因为 FIFO 无差别丢弃最旧条目，可能丢失关键的早期决策信息。

### 3.4 Memory-conditioned Diffusion Action Expert

利用记忆增强的工作记忆 $\{\tilde{p}, \tilde{c}\}$，动作专家预测 $T=16$ 步的未来动作序列。采用基于 DiT（Diffusion Transformer）的扩散策略，使用 DDIM 进行 **10 步**去噪。

具体而言，每个去噪步中：

1. 噪声动作 token 注入去噪时间步的正弦编码，并与认知表征 $\tilde{c}$ 拼接
2. **Cognition Attention 层**：以认知 token 提供高层语义引导
3. **Perception Attention 层**：补充感知 token 的细粒度视觉细节
4. FFN 精炼得到当前步的去噪输出
5. 最终通过 MLP 生成连续 7-DoF 动作

训练使用 MSE 损失，推理使用 **Classifier-Free Guidance (CFG)**，引导尺度 1.5。

### 3.5 双注意力设计的合理性

动作专家中的**双注意力**结构（先认知、后感知）体现了层级化决策逻辑：

- Cognition Attention 回答"接下来做什么"（语义层面）
- Perception Attention 回答"具体怎么做"（视觉-空间层面）

这与人类决策过程类似：先确定意图（"把杯子放到盘子上"），再根据视觉细节精确执行。

---

## 四、实验结果

### 4.1 SimplerEnv-Bridge（WidowX 机器人）

| 方法 | Spoon on Towel | Carrot on Plate | Stack Cube | Eggplant in Basket | Avg. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 4.2 | 0.0 | 0.0 | 12.5 | 4.2 |
| SpatialVLA | 16.7 | 25.0 | 29.2 | 100.0 | 42.7 |
| CogACT-Large | 58.3 | 45.8 | 29.2 | 95.8 | 57.3 |
| $\pi_0$-Beta* | 84.6 | 55.8 | 47.9 | 85.4 | 68.4 |
| **MemoryVLA** | **75.0** | **75.0** | **37.5** | **100.0** | **71.9 (+14.6)** |

在 Carrot on Plate 上比 CogACT 高出 29.2%，因为该任务需要精确感知胡萝卜和盘子的相对位置变化，PCMB 的感知流提供了关键的位置历史。

### 4.2 SimplerEnv-Fractal（Google Robot）

| 方法 | VM Avg. | VA Avg. | Overall |
| --- | --- | --- | --- |
| CogACT | 74.8 | 61.3 | 68.1 |
| $\pi_0$-Beta* | 71.4 | – | – |
| **MemoryVLA** | **77.7** | **67.7** | **72.7 (+4.6)** |

在 Visual Aggregation（更强 OOD 测试）下增益更大（+6.4），说明记忆机制增强了模型对环境变化的鲁棒性。Open/Close Drawer (VA) 上提升 +24.9，因为开关抽屉的多阶段控制强依赖时序上下文。

### 4.3 LIBERO（Franka 机器人）

| 方法 | Spatial | Object | Goal | Long | LIBERO-90 | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 73.5 | 75.9 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| $\pi_0$* | 96.8 | 98.8 | 95.8 | 85.2 | – | 94.2 |
| **MemoryVLA** | **98.4** | **98.4** | **96.4** | **93.4** | **95.6** | **96.5 (+3.3)** |

在 Long suite 上提升最为显著（+4.6 vs CogACT，+8.2 vs $\pi_0$），验证了时序记忆对长时域任务的核心价值。值得注意的是 MemoryVLA 仅使用第三人称 RGB，**不用腕部相机或本体感知**（$\pi_0$-FAST 使用了额外输入）。

### 4.4 真实世界评估

| 方法 | General Avg. | Temporal Avg. |
| --- | --- | --- |
| OpenVLA | 31 | 9 |
| $\pi_0$ | 72 | 52 |
| CogACT | 76 | 57 |
| **MemoryVLA** | **85 (+9)** | **83 (+26)** |

**长时域时序任务**是 MemoryVLA 的杀手应用场景：

- **Seq. Push Buttons**：+43（CogACT 仅 15%，MemoryVLA 达 58%）——需要记住按钮按压顺序
- **Change Food**：+38——需要记住"已经移走了什么"
- **Guess Where**：+32——需要记住遮盖动作的执行状态
- **Clean Table & Count**：+17——每清理一件物品后需按计数器，强依赖历史计数

### 4.5 消融实验

#### 记忆类型与长度

| 变体 | Avg. |
| --- | --- |
| 仅认知记忆 | 63.5 |
| 仅感知记忆 | 64.6 |
| **双流记忆** | **71.9** |
| 记忆长度 4 | 67.7 |
| **记忆长度 16** | **71.9** |
| 记忆长度 64 | 67.7 |

- 双流记忆比单流高出 7-8%，说明感知和认知信息互补不可替代
- 长度 16 是最优，过短（4）缺乏足够历史，过长（64）引入噪声和冗余

#### 记忆模块设计

| 组件 | 变体 | Avg. |
| --- | --- | --- |
| Retrieval | w/o Timestep PE | 69.8 |
| Retrieval | **w/ Timestep PE** | **71.9** |
| Fusion | Add | 67.7 |
| Fusion | **Gate** | **71.9** |
| Consolidation | FIFO | 66.7 |
| Consolidation | **Token Merge** | **71.9** |

每个组件的消融都显示了显著差距，设计选择合理且一致。

### 4.6 鲁棒性与泛化

在真实世界的 Pick Place Order 和 Clean Restaurant Table 任务上测试 6 种 OOD 变体：

- **Pick Place Order**：Base 100% → 最低 89%（unseen object），occlusion 96%
- **Clean Restaurant Table**：Base 96% → 最低 86%（unseen distractors）

在所有 OOD 条件下均维持 86%+ 的成功率，说明 PCMB 的记忆机制不仅增强了时序能力，也提升了对环境扰动的鲁棒性。

---

## 五、训练与推理细节

| 超参数 | 值 |
| --- | --- |
| 硬件 | 8× NVIDIA A100 |
| 分布式策略 | PyTorch FSDP |
| 全局 Batch Size | 256（32/GPU × 8） |
| 学习率 | $2 \times 10^{-5}$ |
| 图像分辨率 | 224 × 224 |
| 动作块长度 | 16 步 |
| 感知 token 数 | 256 |
| DDIM 采样步 | 10 |
| CFG 引导尺度 | 1.5 |
| 扩散训练重复步 | 4 |
| VLM 参数量 | ~7B |
| 动作专家参数量 | ~300M |
| 记忆长度 $L$ | 16（General）/ 256（Long-horizon Temporal） |

数据加载器设计为流式队列：每个 episode 作为帧序列推入，batch 内帧尽量来自同一 episode。数据增强包括随机裁剪（90% 面积）、亮度（0.2）、对比度/饱和度（[0.8, 1.2]）、色调（±0.05）。

---

## 六、局限性与未来方向

1. **感知记忆的存储开销**：每个时刻存储 256 个感知 token，当记忆长度 $L$ 增大时 PCMB 的存储和检索开销增长显著。论文中真实世界长时域任务使用 $L=256$，意味着检索时 key 矩阵维度为 $256 \times 256 = 65536$，可能成为部署瓶颈。

2. **单帧视觉编码的上限**：尽管 PCMB 在表征层面建模了时序依赖，但视觉编码器仍然是逐帧独立处理的。对于需要跨帧光流或运动信息的任务（如判断物体运动方向），纯表征级别的记忆可能不够。

3. **记忆合并的信息损失**：相邻条目的简单均值合并虽然有效，但不可避免地丢失部分信息。对于需要精确回忆特定历史细节的任务，可能产生偏差。

4. **相机视角敏感性**：仿真鲁棒性实验显示，unseen camera view 下 Pick Coke Can 从 92% 降到 42%，说明感知记忆对相机视角变化较为敏感。

5. **未来方向**：论文提出 (i) **记忆反思**——将长期记忆对齐到 LLM 输入空间实现嵌入空间 CoT 推理；(ii) **终身记忆**——通过生物启发的合并机制将频繁使用的经验蒸馏为永久表征。

---

## 七、个人思考

### 7.1 与 OptimusVLA 的对比

OptimusVLA 也引入了记忆增强 VLA，但两者的设计哲学截然不同：

- **MemoryVLA**：建模**观测历史**的感知-认知表征，需要每步调用 VLM 更新工作记忆，但信息更丰富
- **OptimusVLA**：GPM 建模**跨任务的轨迹先验**（检索相似任务的历史轨迹初始化 flow），LCM 建模**动作历史**（用 Mamba 编码最近动作块），计算开销更低

两者在 LIBERO-Long 上的表现接近（MemoryVLA 93.4% vs OptimusVLA 96.4%），但 OptimusVLA 额外获得了推理加速（自适应 NFE），而 MemoryVLA 在真实世界长时域任务上的优势更大（+26 vs CogACT）。

### 7.2 感知-认知分离的认知合理性

MemoryVLA 最核心的设计洞察是将记忆分为感知流和认知流。消融实验清楚地表明双流（71.9%）远优于任一单流（63.5/64.6%），增量约 7%。这符合认知科学中 verbatim-gist 理论：

- 感知记忆保留"在哪里看到了什么"→ 支持精确的空间-运动推理
- 认知记忆保留"正在做什么阶段"→ 支持任务进度跟踪和阶段判断

### 7.3 合并策略 vs FIFO 的差距

Token Merge 比 FIFO 高出 5.2%（71.9% vs 66.7%），这是一个非常大的差距。原因在于 FIFO 总是丢弃最旧的条目，但在长时域任务中，最早的观测可能包含关键的初始状态信息（如"物品的初始位置"）。Token Merge 通过合并**相似**而非**最旧**的条目，有效保留了关键转折点的记忆。

### 7.4 与 CogACT 的关系

MemoryVLA 的 VLM 骨架和动作专家架构直接继承自 CogACT（同一团队的前序工作），核心贡献全在中间的记忆模块。这种"保持两端不变、只改中间"的设计使得：(a) 可以直接复用 CogACT 的预训练权重；(b) 消融实验的公平性有保证；(c) 改进的归因非常清晰。

### 7.5 真实世界 +26 的来源

General 任务上 +9，Temporal 任务上 +26，差距约 3 倍。这个比例直接量化了"时序建模"对长时域任务的贡献度。有趣的是，$\pi_0$（无时序建模）在 General 上与 CogACT 相当（72 vs 76），但在 Temporal 上差距拉大（52 vs 57），说明即使是较强的基线策略，长时域也是其核心短板。

---

## 参考

- **CogACT (Li et al., 2024)**：MemoryVLA 的 VLM 骨架和动作专家基线架构
- **Prismatic VLM (Karamcheti et al., 2024)**：7B VLM 基础模型，DINOv2 + SigLIP 双编码器
- **OptimusVLA (Li et al., 2026)**：另一种记忆增强 VLA，建模动作历史而非观测历史
- **$\pi_0$ / $\pi_{0.5}$ (Black et al., 2024/2025)**：Flow Matching VLA 基线
- **DiT (Peebles & Xie, 2023)**：扩散 Transformer，动作专家的基础架构
- **Baddeley & Hitch (1974)**：工作记忆理论，MemoryVLA 的认知科学基础
- **Fuzzy-trace Theory (Reyna & Brainerd, 1995)**：逐字-要义双表征理论，感知-认知分流的理论来源
