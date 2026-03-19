# BitVLA：首个 1-bit VLA 模型

> **论文**：BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation
>
> **作者**：Hongyu Wang, Chuyan Xiong, Ruiping Wang, Xilin Chen
>
> **机构**：Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences; University of Chinese Academy of Sciences
>
> **发布时间**：2025年6月
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2506.07530) | [GitHub](https://github.com/ustcwhy/BitVLA)
>
> **分类标签**：`1-bit 量化` `VLA` `蒸馏感知训练` `三值化` `边端部署`

---

## 一句话总结

首个全参数三值化（$\{-1, 0, 1\}$）的 VLA 模型，基于 BitNet b1.58 LLM 骨架 + 蒸馏感知训练将视觉编码器也量化到 1.58-bit，在 LIBERO 上无需大规模机器人预训练即达 94.8% 平均成功率（匹配 OpenVLA-OFT INT4），显存仅 1.4GB（29.8%）。

---

## 一、问题与动机

### 1.1 VLA 部署的显存瓶颈

VLA 模型（如 OpenVLA 7.5B、OpenVLA-OFT 7.7B）在机器人操控上展现了强大泛化能力，但模型体量过大：

- OpenVLA-OFT 需要 **15.4GB** 显存，远超消费级 GPU（如 RTX 3050 Ti 4GB）的预算
- 实际机器人平台通常内存和算力受限，无法承载 7B+ 的全精度模型
- 即使 4-bit PTQ 也需 4.7GB，仍超出许多边端设备能力

### 1.2 1-bit LLM 的成功与空白

1-bit（实为 1.58-bit）LLM 近期取得了突破性进展：

- **BitNet b1.58** 证明 3B 规模下三值化 LLM 可匹配全精度模型性能
- **bitnet.cpp** 实现了 1-bit 模型在 CPU 上的高效推理
- 三值参数空间支持加法替代乘法，硬件效率极高

但这些工作**全部停留在纯语言领域**——1-bit 模型向多模态和机器人控制的延伸尚未被探索。

### 1.3 视觉编码器的显存占比

VLA 中 LLM 已有原生 1-bit 方案（BitNet b1.58 2B4T），但视觉编码器仍是全精度：

- SigLIP-L 虽然参数量相对小，但全精度仍占 **0.8GB**
- 将 ViT 量化到 1.58-bit 可进一步压至 **0.1GB**（8× 压缩）
- 难点在于 ViT 的量化感知训练缺乏现成方案，且量化后视觉表征质量如何保持是核心挑战

---

## 二、预备知识

### 2.1 BitNet b1.58 量化

BitNet b1.58 将权重限制为三值 $\{-1, 0, 1\}$，使用 **absmean 量化器**：

$$Q_w(W) = \alpha \cdot \text{RoundClip}\left(\frac{W}{\alpha}, -1, 1\right), \quad \alpha = \frac{1}{nm}\|W\|_1$$

其中 $W \in \mathbb{R}^{m \times n}$ 是线性层权重，$\alpha$ 为 L1 范数的均值。

激活值使用 **per-token absmax 量化器** 量化到 INT8：

$$Q_a(x) = \frac{\beta}{127} \cdot \text{RoundClip}\left(\frac{127x}{\beta}, -128, 127\right), \quad \beta = \|x\|_\infty$$

$$\text{RoundClip}(x, a, b) = \max(a, \min(b, \text{round}(x)))$$

用大白话说：权重被「粗暴地」量化到 $\{-1, 0, 1\}$ 三个值，用 L1 均值做缩放；激活值量化到 8-bit 整数，用每个 token 的最大绝对值做缩放。三值化的好处是矩阵乘法可以用加法和减法代替，硬件效率极高。

### 2.2 直通估计器（STE）

量化操作不可微，训练时使用 STE 近似梯度：

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial Q_w(W)}, \quad \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Q_a(X)}$$

梯度直接「穿过」量化函数传递，优化器状态保持全精度以维持训练稳定性。

### 2.3 OpenVLA-OFT 微调范式

OpenVLA-OFT 通过三项关键改进优化 VLA 微调：

1. **并行解码**：用双向注意力掩码替代因果掩码，一次前向传播生成完整动作轨迹
2. **动作分块（Action Chunking）**：每次生成 $K=8$ 步动作，执行完整块后再重规划
3. **连续动作建模**：MLP 动作头将隐表征映射到连续机器人动作空间，用 L1 损失训练

---

## 三、核心方法

BitVLA 的训练分为**四个阶段**：VLM 的视觉对齐 + 指令微调（Stage I/II）、蒸馏感知训练量化 ViT（Stage III）、机器人微调（OFT）。

### 3.1 模型架构

| 组件 | 选择 | 精度 |
| --- | --- | --- |
| LLM | BitNet b1.58 2B4T（2B 参数） | 1.58-bit 权重 + 8-bit 激活 |
| 视觉编码器 | SigLIP-L（224×224） | Stage I/II: BF16 → Stage III: 1.58-bit |
| 连接器 | 2 层 MLP + GeLU | 全精度（参数量可忽略） |

选择 SigLIP-L 而非更高分辨率版本是为了生成更短的视觉 token 序列（256 tokens），以提升计算效率。

### 3.2 Stage I & II：VLM 训练

沿用 LLaVA 范式，使用 1-bit LLM + 全精度 ViT：

**Stage I — 视觉对齐**：
- 仅训练连接器，LLM 和 ViT 冻结
- 数据：LLaVA 1.5-558k 图像描述数据集
- 步数：25k，学习率 1e-3

**Stage II — 指令微调**：
- 训练 LLM + 连接器，ViT 冻结
- 数据：MammoTH-VL 的 10M 样本子集（单图像）
- 步数：40k，学习率 3e-4
- 使用两阶段权重衰减（0.1 → 0）

### 3.3 Stage III：蒸馏感知训练

这是 BitVLA 的核心创新——将全精度 ViT 量化到 1.58-bit 同时保持视觉表征质量。

**初始化**：从 Stage II 的全精度 ViT 权重初始化 1.58-bit ViT 的潜在权重。

**教师模型**：全精度 ViT 作为教师，冻结不参与训练。

**训练目标**由两部分组成：

**语言建模损失**——维持任务能力：

$$\mathcal{L}_\text{LM} = -\sum_{\text{token}_i \in \mathcal{T}_\text{ans}} \log \Pr(\mathcal{Y}^i \mid \mathcal{V}_\text{1.58-bit}, \mathcal{T}^{[:i-1]})$$

其中 $\mathcal{V}_\text{1.58-bit}$ 是 1.58-bit ViT 提取的视觉 token，损失仅在回答部分计算。

**表征对齐损失**——约束 1.58-bit ViT 逐层对齐教师模型：

$$\mathcal{L}_\text{aux} = \frac{1}{n}\sum_{l=1}^{L} \left\|h^l_\text{bf16} - h^l_\text{1.58-bit}\right\|^2$$

其中 $h^l_\text{bf16}$ 和 $h^l_\text{1.58-bit}$ 分别是全精度和 1.58-bit ViT 第 $l$ 层的输出，$n$ 是隐藏维度，$L$ 是层数。

**总训练目标**：

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{LM} + \gamma \cdot \mathcal{L}_\text{aux}$$

其中 $\gamma = 0.1$。仅 ViT 可训练，LLM 和连接器冻结。

用大白话说：一边让 1.58-bit ViT 学会做任务（语言建模损失），一边强制它每一层的输出都尽量模仿全精度教师（对齐损失）。双重约束确保极端量化下视觉表征不崩坏。

**关键发现**：与 LLM 的 1.58-bit 预训练不同，ViT 的量化感知训练在有教师蒸馏的情况下**极其数据高效**——仅需约 10B token 即可保持大部分性能。

**量化范围**：对 ViT 中所有线性层施加量化，但排除输入和输出 embedding 层。

### 3.4 机器人微调（OFT）

使用 OpenVLA-OFT 的微调范式：

- 并行解码 + 动作分块（$K=8$）
- MLP 动作头映射到连续动作空间
- L1 损失：$\mathcal{L} = \|a_\text{pred} - a_\text{gt}\|_1$
- 处理多视角视觉输入（腕部摄像头 + 外部摄像头）+ 本体感觉信号
- 全参数微调（包括 1-bit LLM、1-bit ViT、连接器、动作头）

---

## 四、实验结果

### 4.1 LIBERO 机器人操控主实验

LIBERO 评估四个维度：空间泛化（Spatial）、物体泛化（Object）、目标泛化（Goal）、长时序推理（Long），每个维度 10 个任务 × 500 条演示。

**与有大规模机器人预训练的方法对比**：

| 模型 | 参数量 | 显存 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenVLA | 7.5B | 15.1GB (10.79×) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 4.2B | 8.5GB (6.07×) | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 8.0B | 16.2GB (11.57×) | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| NORA-Long | 3.8B | 7.5GB (5.36×) | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| π₀ | 3.5B | 7.0GB (5.00×) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT（预训练） | 7.7B | 15.4GB (11.00×) | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** |

**无机器人预训练对比**：

| 模型 | 参数量 | 显存 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenVLA-OFT（无预训练） | 7.7B | 15.4GB (11.00×) | 94.3 | 95.2 | 91.7 | 86.5 | 91.9 |
| **BitVLA** | **3.0B** | **1.4GB (1.00×)** | **97.4** | **99.6** | **94.4** | **87.6** | **94.8** |

**核心数字**：

- BitVLA 平均 **94.8%**，超越无预训练的 OpenVLA-OFT（91.9%）**2.9pp**
- 显存仅 **1.4GB**，是 OpenVLA-OFT 的 **29.8%**（1/11）
- 超越有预训练的 π₀（94.2%），比 NORA-Long（87.9%）高 6.9pp
- 仅 LIBERO-Long 上略逊于有大规模机器人预训练的 OpenVLA-OFT（87.6 vs 94.5），差距来自预训练数据

### 4.2 与 PTQ 方法对比

| 模型 | 显存 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| OpenVLA INT8 | 7.4GB (5.29×) | 86.4 | 85.2 | 77.2 | 58.8 | 76.9 |
| OpenVLA-OFT INT8 | 7.7GB (5.50×) | 98.8 | 98.0 | 96.6 | 94.0 | 96.7 |
| OpenVLA INT4 | 4.4GB (3.14×) | 83.0 | 84.0 | 72.0 | 51.6 | 72.7 |
| OpenVLA-OFT INT4 | 4.7GB (3.36×) | 98.2 | 98.2 | 97.2 | 93.8 | 96.9 |
| **BitVLA** | **1.4GB (1.00×)** | **97.4** | **99.6** | **94.4** | **87.6** | **94.8** |

BitVLA 以**不到 OpenVLA-OFT INT4 三分之一的显存**（1.4 vs 4.7GB），达到可比的性能（94.8 vs 96.9），且 Object 维度 **99.6%** 超越所有方法。

### 4.3 VQA 零样本评估

评估蒸馏感知训练对视觉理解能力的影响：

| 模型 | MMMU | SeedBench | SeedBench2+ | MMStar | AI2D | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| BitVLA w/ 16-bit ViT | 37.4 | 70.6 | 45.0 | 43.6 | 68.6 | 53.0 |
| BitVLA w/ 1.58-bit ViT | 35.4 | 69.3 | 43.7 | 41.5 | 67.6 | 51.5 |

1.58-bit ViT 仅造成 **1.5% 平均精度下降**，同时将 ViT 显存从 0.8GB 压缩到 0.1GB（**8× 压缩**）。

### 4.4 消融实验

**表征对齐损失的作用**：

| 训练 Token 数 | $\mathcal{L}_\text{aux}$ | VQA Avg. | LIBERO Avg. |
| --- | --- | --- | --- |
| 10B | ✓ | **51.5** | **94.8** |
| 5B | ✓ | 50.8 | 93.6 |
| 5B | ✗ | 42.4 | 92.9 |

- 对齐损失在 VQA 上提供 **8.4pp** 的巨大增益（42.4 → 50.8）
- 在 LIBERO 上增益较小（92.9 → 93.6），因为下游 fine-tuning 部分弥补了差距
- LIBERO-Goal 上对齐损失提供 2.4pp 增益

**数据量的影响**：

- 10B token 比 5B token 在 VQA 上提升 0.7pp、LIBERO 上提升 1.2pp
- 说明蒸馏感知训练的数据效率很高——5B token 已接近上限

### 4.5 失败案例分析

论文细致分析了 LIBERO 上的三类失败模式：

1. **空间定位偏差**（最常见，占比 71-100%）：抓取位姿不精确、放置位置偏移、重心不稳物体（酒瓶）处理失败
2. **目标误解**（占比 7-21%）：错误交互非目标物体后触发新任务 rollout，视觉-本体感觉信号在目标切换时的主导性问题
3. **轨迹规划失败**（占比 6-17%）：运动碰撞（如机械臂撞到抽屉面板），需更好的前瞻性子目标规划

---

## 五、局限性与未来方向

### 5.1 缺乏大规模机器人预训练

BitVLA 受限于资源未在 Open X-Embodiment 等大规模数据集上预训练，LIBERO-Long 上与有预训练的 OpenVLA-OFT 差距 6.9pp。在大规模预训练数据上训练 1-bit VLA 可能进一步释放性能。

### 5.2 训练成本较高

三阶段 VLM 训练 + 蒸馏感知训练共需 8×A100 训练 14 天，加上下游 OFT 微调。虽然推理极高效，但训练代价不小。

### 5.3 仅验证仿真环境

实验仅在 LIBERO 仿真环境上评估，未涉及真实机器人部署。1-bit 模型在边端硬件上的实际推理速度和控制频率还需验证。

### 5.4 空间定位能力不足

失败案例分析显示空间定位偏差是最主要的瓶颈（Long 套件 93.5%），可能需要更强的空间感知模块（如 3D 信息）来解决。

---

## 六、个人思考

### 6.1 原生 1-bit vs 后训练量化：两条路线的碰撞

BitVLA 和 [RLRC](/papers/06-embodied-ai/vla/efficient/RLRC_2025) 代表了 VLA 压缩的两条截然不同的路线：

| 维度 | BitVLA（原生 1-bit） | RLRC（PTQ + 恢复） |
| --- | --- | --- |
| 量化策略 | 从头训练 1-bit 模型 | 先全精度训练再剪枝/量化 |
| 精度 | 1.58-bit 权重 + 8-bit 激活 | 90% 剪枝 + 4-bit 量化 |
| 显存 | 1.4GB | 1.772GB |
| 性能恢复 | 蒸馏感知训练 | SFT + RL |
| 基座依赖 | BitNet b1.58 2B4T（原生 1-bit） | OpenVLA 7.5B（全精度剪枝） |
| 训练代价 | 14 天 8×A100 | SFT 10k 步 + RL 0.6M 步 |

RLRC 更灵活——可以对任何现有 VLA 施加，且 RL 恢复甚至能超越原始模型。BitVLA 更极致——显存更低、原生支持高效硬件执行，但需要专用 1-bit LLM 基座。两者的最终显存接近（1.4 vs 1.772GB），说明不同路线在「压缩极限」上趋于收敛。

### 6.2 蒸馏的数据效率令人意外

BitVLA 在 Stage III 仅用 5-10B token 就能完成 ViT 的 1.58-bit 量化感知训练，且性能损失极小（VQA 仅降 1.5pp）。这与 LLM 的 1-bit 预训练需要海量数据形成对比。原因可能是：

1. ViT 参数量远小于 LLM，需要的数据量相应更少
2. 全精度教师模型提供了强大的正则化——对齐损失将 VQA 从 42.4% 拉到 50.8%，贡献了绝大部分的性能保持
3. ViT 的权重分布可能天然更适合三值化——视觉特征的冗余度高于语言特征

这暗示一个有趣的方向：**对 VLA 的不同组件施加不同精度的量化**可能比统一量化更高效。例如 LLM 需要 1.58-bit 从头训练，但 ViT 可以从全精度快速蒸馏。

### 6.3 「无预训练也能很强」的启示

BitVLA 没有在 Open X-Embodiment 上做大规模机器人预训练，却在 LIBERO 上达到 94.8%，超越了有预训练的 π₀（94.2%）和 NORA-Long（87.9%）。这说明：

1. **OFT 微调范式本身非常强大**——并行解码 + 动作分块 + L1 连续动作可能比预训练数据量更重要
2. **VLM 阶段的多模态理解能力已经足够**——BitNet b1.58 2B4T 虽然是 1-bit 模型，但在 10M 指令微调后已具备足够的视觉-语言理解
3. 预训练的主要价值可能集中在**长时序推理**上——LIBERO-Long 是 BitVLA 唯一明显落后的维度

### 6.4 与 VLA Token 压缩方法的正交性

BitVLA 压缩的是**模型参数**（权重三值化），而 [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025)、[VLA-Pruner](/papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025)、[EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) 等方法压缩的是**推理计算**（Token 缓存/剪枝）。两者完全正交——理论上可以在 BitVLA 的 1-bit 模型上再叠加 Token 级优化，进一步提升推理效率。这可能是边端 VLA 部署的最终形态：**极致参数量化 + 动态 Token 压缩**。

### 6.5 Object 维度为何特别强？

BitVLA 在 LIBERO-Object 上达到了惊人的 **99.6%**，超越所有方法（包括有预训练的 OpenVLA-OFT 的 98.4%）。Object 维度测试的是对**未见物体类别**的泛化——这恰好是 VLM 预训练最擅长的方面。BitVLA 虽然权重是 1-bit，但通过 SigLIP + BitNet 的多模态训练，保留了强大的语义理解能力。这进一步印证了 VLA 中「视觉-语言理解」和「精细操控」是两个相对独立的能力维度。

---

## 参考

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — 1.58-bit LLM 预训练（BitVLA 的 LLM 骨架来源）
- [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285) — 2B 参数三值化 LLM
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) — VLA 微调优化（BitVLA 的微调策略来源）
- [LLaVA](https://arxiv.org/abs/2304.08485) — 视觉指令微调范式（BitVLA 的 VLM 训练范式来源）
- [RLRC](/papers/06-embodied-ai/vla/efficient/RLRC_2025) — 后训练剪枝+量化 VLA 压缩（另一种压缩路线）
- [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) — Token 缓存加速（正交的推理加速方法）
- [SigLIP](https://arxiv.org/abs/2303.15343) — Sigmoid Loss 视觉-语言预训练（BitVLA 的视觉编码器）
