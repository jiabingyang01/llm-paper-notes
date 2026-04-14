# FocusVLA：聚焦任务相关视觉信息的 VLA 模型

> **论文**：*FocusVLA: Focused Visual Utilization for Vision-Language-Action Models*
>
> **作者**：Yichi Zhang*, Weihao Yuan*†, Yizhuo Zhang, Xidong Zhang, Jia Wan
>
> **机构**：Harbin Institute of Technology (Shenzhen)、DaiMon Robotics、Nanjing University、Renmin University of China
>
> **发布时间**：2026年3月（还未中稿）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.28740)
>
> **分类标签**：`视觉利用效率` `注意力机制` `Cascaded Attention` `Token 选择` `自回归策略` `训练效率`

---

## 一句话总结

通过系统性实验揭示自回归 VLA 策略的性能瓶颈**不在视觉表征质量、而在视觉信息利用方式**（三大瓶颈：结构性捷径偏置、信息过载、任务无关噪声），提出 **FocusVLA**——用 Modality Cascaded Attention 消除混合注意力中的捷径路径，用 Focus Attention（Patch-level 剪枝 + Channel-level 门控）控制视觉 token 的数量和质量，仅 0.5B 参数在 LIBERO 上达到 98.7%（multi-weight）/97.0%（single-weight），超越 7B 级模型，训练收敛速度提升 1.5$\times$（LIBERO-Spatial 上 5$\times$）。

---

## 一、问题与动机

### 1.1 自回归 VLA 的视觉利用困境

当前自回归 VLA 策略（OpenVLA-OFT、VLA-Adapter 等）虽然具有强 in-domain 拟合能力，但在精细操作任务上表现不佳。论文认为核心瓶颈**不是视觉编码器的表征能力，而是策略如何利用视觉信息**。通过 Fig. 1 的对比分析：

- **OpenVLA-OFT**：采用并行解码加速，但动作生成阶段**完全忽略视觉特征**，直接用 MLP 从 action queries 解码动作
- **VLA-Adapter**：引入混合注意力（Mixed Attention）将 VLM 信息注入策略，但产生了**结构性捷径**——模型倾向从易学习的 action query 通道获取信息，绕过视觉 token 中的空间细节；其单参数门控因子 $g$ 训练后收敛到接近零（$\approx 10^{-3}$），进一步抑制了视觉信号

### 1.2 三大系统性瓶颈

通过控制实验（在 LIBERO-Long 上评估四种架构 $\times$ 三种视觉表征），论文识别出三个根本性问题：

**瓶颈一：结构性偏置（Architectural Bias）**

混合注意力中，action latent 同时对 vision tokens、action queries 和自身做注意力计算。由于 action queries 是可学习的且维度远小于视觉 token，模型自然倾向走这条"捷径"——从 action queries 获取粗糙的任务信号，跳过对视觉细节的精细提取。

**瓶颈二：信息过载（Information Overload）**

视觉 token 数量过多（如 512 个 patch token），导致注意力分布被稀释，策略难以集中关注操作关键区域。

**瓶颈三：任务无关噪声（Task-Irrelevant Noise）**

大量背景信息造成低信噪比，任务相关的有效信号被环境噪声淹没。

### 1.3 关键发现

论文在 Fig. 4 中系统对比了四种架构变体（Vanilla / Pooling / 1-param gate / Cascaded-attn）$\times$ 三种视觉表征（DS=DINOv2+SigLIP / VLM / VGGT），得出三个核心发现：

> **Key Finding 1**：仅通过简单的 2$\times$2 pooling 减少 token 数量或用单参数门控抑制视觉信号强度就能显著提升性能，说明 VLA 策略同时受**数量失衡**和**低信噪比**的困扰
>
> **Key Finding 2**：从混合注意力切换到级联注意力产生了**最大的性能增益**（93.6% → 97.0%），注意力分布从分散转为聚焦，说明**聚焦是有效视觉利用的核心驱动力**
>
> **Key Finding 3**：不同视觉表征（DS、VLM、VGGT）在朴素使用时差异不大，但一旦视觉利用被合理调控，**所有表征都获得显著提升**。这表明 VLA 性能主要受限于视觉信息的利用方式，而非表征本身的质量

---

## 二、预备知识：VLA-Adapter 的混合注意力

FocusVLA 基于 VLA-Adapter 构建，先理解其混合注意力机制。VLA-Adapter 将两种 VLM 条件——视觉特征 $C^V_t$ 和 action queries $C^{AQ}_t$——注入 action latent $A_t$（$t$ 表示第 $t$ 层）。注意力权重计算：

$$W_t = \text{Softmax}\left(\frac{[S_V \odot \text{Tanh}(g),\; S_{AQ},\; S_A]}{\sqrt{d}}\right)$$

其中 $g$ 是单参数门控，$S_V$、$S_{AQ}$、$S_A$ 分别是 action latent 对 vision、action query、自身的注意力分数：

$$S_V = (\sigma_q(A_t))(\sigma_k(C^V_t))^\top, \quad S_{AQ} = (\sigma_q(A_t))(\sigma_k(C^{AQ}_t))^\top, \quad S_A = (\sigma_q(A_t))(\sigma_k(A_t))^\top$$

问题出在 **Softmax 的竞争机制**——三种来源的注意力分数拼接后一起做归一化。由于 action queries 数量少（64 个）且可学习，其分数容易在 Softmax 竞争中胜出，导致视觉 token（512 个，大多是背景噪声）的注意力权重被压缩到极低。加上门控 $g$ 训练后收敛到接近零，视觉信号被双重抑制。

用大白话说：混合注意力让模型"自由选择看哪里"，结果模型学会了走捷径——看容易的 action query 就够了，干嘛费劲去从几百个视觉 patch 里找有用信息？

---

## 三、核心方法

### 3.1 整体架构

FocusVLA 的策略网络由 N 层 Cascaded-Attn Block 组成，每层包含三个子模块：

1. **Focus-Attn**：视觉条件注入（含 Patch-level 和 Channel-level 聚焦）
2. **Cross-Attn**：action query 条件注入
3. **Self-Attn**：action latent 自注意力

VLM 的第 $t$ 层输出作为第 $t$ 层策略 block 的条件输入，逐层注入。

### 3.2 Modality Cascaded Attention

核心创新一：**将混合注意力拆解为级联的单模态注意力**，消除结构性捷径。

在每层，action latent $A_t$ 依次独立地与各模态交互：

$$H_A = \text{Attn}(A_t, A_t), \quad H_{AQ} = \text{Attn}(A_t, C^{AQ}_t), \quad H_V = \text{Attn}(A_t, C^V_t)$$

三个输出通过融合 MLP 合并：

$$\hat{A_t} = \sigma_{\text{fusion}}([H_A,\; H_{AQ},\; H_V])$$

最后经残差 FFN 更新：

$$A_{t+1} = \text{FFN}(\hat{A_t}) + A_t$$

用大白话说：级联设计强制模型**单独面对视觉 token**——不能再依赖 action query 走捷径了，因为视觉注意力和 action query 注意力是在独立的 Softmax 中计算的，视觉 token 之间不再需要与 action queries 竞争注意力权重。

**效果**：如 Fig. 2 所示，注意力分布从分散的全图模式转变为高度聚焦于操作目标和接触区域的模式。成功率从 93.6% 提升到 97.0%（+3.4%）。

### 3.3 Focus Attention

核心创新二：在级联注意力的视觉分支 $H_V = \text{Attn}(A_t, C^V_t)$ 中，进一步从 **Patch 级别**（控制数量）和 **Channel 级别**（控制质量）两个层面优化视觉信息。

#### 3.3.1 Patch-level Focus（数量调控）

与现有在 VLM 端做 token 选择以加速推理的方法不同，FocusVLA 的 Patch-level Focus **应用在策略端**，目标是提升动作质量而非计算效率。

原理：VLM 经过大规模预训练，能自然地将图像区域与语义概念对齐；但策略网络从头训练、参数量小得多，难以一致地关注任务相关区域，导致视觉条件充满噪声。

具体做法——基于交叉注意力分数的 TopK 选择：

$$W_V = \text{Softmax}\Big(\text{TopK}\big((\sigma_q(A_t))(\sigma_k(C^V_t))^\top\big)\Big)$$

只保留 top-$K$ 分数的视觉 token 用于信息传播，其余被 mask 掉。然后计算视觉注意力输出：

$$H_V = W_V \cdot (\sigma_v(C^V_0))^\top$$

**关键设计**：Keys 使用深层 VLM 特征 $C^V_t$（语义丰富，适合判断"哪些 token 与任务相关"），Values 使用浅层原始视觉特征 $C^V_0$（保留细粒度空间细节）。

用大白话说：用深层特征来"选人"，用浅层特征来"干活"——因为深层特征知道哪里重要，浅层特征保留了精确的空间信息。

**最优 K 值**：消融实验（Tab. 2）显示 $K=256$（从 512 中选一半）效果最优。$K=128$ 过于激进丢失关键信息，$K=512$（无约束）信息过载。

#### 3.3.2 Channel-level Focus（质量调控）

替换 VLA-Adapter 的单参数门控为**逐元素自适应门控**（来自 Gated Attention）：

$$H'_V = H_V \odot \sigma_g(A_t)$$

其中 $\sigma_g$ 是门控 MLP，$\odot$ 是逐元素乘法。每个通道维度有独立的门控值，可以精细地抑制任务无关通道、保留任务相关通道。

为什么不能继续用单参数门控？因为 Cascaded Attention 已经实现了聚焦的注意力模式，此时**均匀抑制所有视觉信息**反而会误伤已经聚焦到的有用信号。需要更细粒度的控制——按通道区分哪些是噪声、哪些是信号。

**额外发现**：Channel-level Focus 还增强了指令跟随能力。如 Fig. 8b 所示，没有门控时策略可能因噪声信息偏离指令生成错误动作；加入门控后任务无关信号被抑制，动作更精确且与指令一致。

---

## 四、实验结果

### 4.1 LIBERO Benchmark

| 方法 | 参数量 | Spatial | Object | Goal | Long | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| **Multi-weights（单任务单策略）** | | | | | | |
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| UniVLA | 7B | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| Spatial Forcing | 7B | 99.4 | 99.6 | **98.8** | 96.0 | 98.5 |
| X-VLA | 0.9B | 98.2 | 98.6 | 97.8 | **97.6** | 98.1 |
| VLA-Adapter-Pro | 0.5B | **99.6** | 99.6 | 98.2 | 96.4 | 98.5 |
| **FocusVLA** | **0.5B** | **99.6** | **100** | **98.8** | 96.2 | **98.7** |
| **Single-weight（四任务共享策略）** | | | | | | |
| Pi0.5 | 3B | 98.8 | 98.2 | **98.0** | **92.4** | 96.9 |
| NORA-1.5 | 3B | 98.0 | 96.0 | 95.4 | 90.5 | 95.0 |
| EVO-1 | 0.5B | 92.7 | 97.7 | 96.3 | 92.3 | 94.8 |
| VLA-Adapter-Pro | 0.5B | 98.8 | 96.2 | 95.6 | 91.6 | 95.6 |
| **FocusVLA** | **0.5B** | **99.2** | **98.4** | 97.0 | **92.4** | **97.0** |

关键发现：

1. **小而强**：0.5B 参数的 FocusVLA 在 multi-weight 设置下达到 98.7%，超越 7B 的 OpenVLA-OFT（97.1%）和 Spatial Forcing（98.5%）
2. **Object 子集达到 100%**：10 个任务全部完美成功
3. **Single-weight 优势更明显**：97.0% vs. EVO-1 的 94.8%（+2.2%）和 VLA-Adapter 的 95.6%（+1.4%），说明聚焦视觉信息对多任务统一策略的价值更大

### 4.2 RoboTwin 2.0 Benchmark

| 任务 | DP | $\pi_0$ | VLA-Adapter | FocusVLA |
| --- | --- | --- | --- | --- |
| | Easy / Hard | Easy / Hard | Easy / Hard | Easy / Hard |
| Beat Block | 42 / 0 | 43 / 21 | 89 / 43 | **93 / 54** |
| Ranking Blocks | 0 / 0 | 19 / 5 | 35 / 8 | **43 / 11** |
| Click Alarmclock | 61 / 5 | 63 / 11 | 44 / 4 | **81 / 20** |
| Hanging Mug | 8 / 0 | 11 / 3 | 8 / 1 | **18 / 5** |
| Move Pad | 1 / 0 | 21 / 1 | 10 / 1 | **20 / 2** |
| Place Basket | 15 / 0 | 16 / 2 | 12 / 0 | **28 / 4** |
| Stack Blocks | 7 / 0 | 42 / 1 | 26 / 0 | **61 / 10** |
| **AVG** | 19 / 1 | 31 / 6 | 32 / 8 | **58 / 15** |

在需要精细操作的任务（如 Hanging Mug、Stack Blocks）上优势尤为显著。Hanging Mug 的 Easy 成功率 VLA-Adapter 仅 8%，FocusVLA 达到 18%（+125%），说明聚焦视觉细节对精细操作至关重要。

### 4.3 训练效率

FocusVLA 在所有 LIBERO 子集上的训练收敛速度均优于 VLA-Adapter：

- **LIBERO-Spatial**：5k steps vs. 25k steps，**5$\times$ 加速**
- **整体平均**：26k steps vs. 40k steps，**1.5$\times$ 加速**

原因：VLA-Adapter 过度依赖 action queries 承载全部动作信息，而 FocusVLA 将一部分信息负担分担到视觉特征上，有效分配了信息带宽。

### 4.4 消融实验（Tab. 2）

| 注意力模式 | Patch-level | Channel-level | 视觉表征 | Avg. |
| --- | --- | --- | --- | --- |
| Mixed | 512 | w/o gate | VLM | 93.6 |
| Cascaded | 512 | w/o gate | VLM | 97.0 |
| Cascaded | 256 | w/o gate | VLM | 98.0 |
| Cascaded | 256 | Element-wise | VLM | 98.2 |
| Cascaded | 256 | Element-wise | VLM+DS | **98.7** |

逐步增益：

1. Mixed → Cascaded：**+3.4%**（消除结构性偏置，最大单项增益）
2. 512 → 256 tokens：**+1.0%**（减少信息过载）
3. w/o gate → Element-wise gate：**+0.2%**（抑制通道噪声）
4. VLM → VLM+DS：**+0.5%**（语义+空间互补）

### 4.5 真实世界实验

在 Realman 平台（7-DoF 机械臂 + 1-DoF 夹爪，顶部+腕部双相机）上评估三类任务：

| 任务 | ACT | VLA-Adapter | FocusVLA |
| --- | --- | --- | --- |
| Grasp Fruit（背景变化） | 33 | 47 | **61** |
| Stack Cups（空间变化） | 57 | 71 | **87** |
| Place Left Block（目标变化） | 63 | 69 | **75** |

每个任务仅用 50 条演示训练。FocusVLA 在所有变化条件下均取得最高成功率，尤其在背景变化和空间变化场景下优势显著。

### 4.6 视觉表征对比

- **VLM**（Qwen2.5-0.5B 输出）：语义丰富但可能过度抽象丢失空间细节，单独使用 98.2%
- **DS**（DINOv2+SigLIP 原始特征）：保留空间结构但缺乏任务语义，单独使用 98.4%
- **VGGT**（隐式 3D 信息）：空间建模强但只能注入策略端、梯度弱且不稳定，仅 96.8%
- **VLM+DS**：语义+空间互补达到最优 **98.7%**

---

## 五、局限性与未来方向

1. **仅关注策略端视觉利用**：VLM 端的视觉利用同样影响动作生成，本文未涉及
2. **规模化验证不足**：仅在 0.5B 模型上验证，更大规模模型和更多训练数据的效果未知
3. **真实世界数据有限**：每任务仅 50 条演示，实际部署场景的鲁棒性有待进一步验证
4. **Token 选择是硬截断**：TopK 选择是不可微的硬 mask，可能无法进行端到端的选择策略优化
5. **双臂协调问题**：RoboTwin 实验中观察到双臂冲突导致的失败，增大 action chunk size 可缓解但未根本解决

---

## 六、个人思考

### 6.1 "利用比表征更重要"的启示

这篇工作最有价值的贡献是实验验证了一个反直觉的结论：**VLA 性能的瓶颈不在视觉编码器的质量，而在策略如何利用视觉信息**。这意味着社区大量关于"用更好的视觉编码器提升 VLA"（PointVLA、DepthVLA、SpatialVLA 等）的努力，可能在利用机制未解决的情况下收益递减。先把利用做好，再谈表征质量，可能是更高效的研究路径。

### 6.2 与 VLA-Adapter / OpenVLA-OFT 的关系

FocusVLA 直接基于 VLA-Adapter 改进，核心修改就是将 Mixed Attention 换成 Cascaded Attention + Focus Attention。有趣的是，OpenVLA-OFT 走了另一个极端——直接不用视觉 token（MLP 解码），却也取得了不错的性能（97.1%）。这似乎暗示：**在混合注意力框架下，噪声的视觉信息可能比没有视觉信息更糟**——因为噪声信息会主动干扰动作生成。FocusVLA 的方案是"精选后利用"，OpenVLA-OFT 是"干脆不用"，两者殊途同归地说明了**视觉噪声是自回归 VLA 的核心痛点**。

### 6.3 与 VLA-Pruner / VLA-Cache 的区别

VLA-Pruner 和 VLA-Cache 也做视觉 token 选择/剪枝，但其动机是**推理加速**——在 VLM 端减少 token 数量以降低计算成本。FocusVLA 的 Patch-level Focus 则是在**策略端**做选择，动机是**提升动作质量**——即使保留所有 token 做 VLM 推理（计算成本不变），策略端的选择性关注本身就能带来性能提升。这是一个重要的区分，说明 token 选择的价值不仅在效率，更在于信息质量的聚焦。

### 6.4 Cascaded Attention 的通用性

Cascaded Attention 的设计思想——"不同模态的信息应该独立查询、再融合"——可能具有超越 VLA 的通用价值。在任何多模态融合场景中，如果观察到某一模态被另一模态"抢占"注意力，级联设计都是一个值得尝试的方案。这与 Transformer 中 cross-attention 和 self-attention 分离的经典设计一脉相承。

### 6.5 Key-Value 分离的巧妙设计

Patch-level Focus 中用深层特征做 Key（选择）、浅层特征做 Value（提取）的设计很有洞察力。VLM 的深层特征经过大量文本-视觉对齐训练，确实更擅长判断"哪些区域与任务相关"；而浅层特征保留了更多的空间细节（边缘、纹理、位置），是动作生成所需的精细信息。这种"分工协作"的 KV 设计在其他视觉-动作桥接场景中也可能有效。

---

## 参考

- **VLA-Adapter（2025）**：FocusVLA 的基线模型，提出了混合注意力策略架构，FocusVLA 在其基础上将混合注意力替换为级联注意力
- **OpenVLA-OFT（2025）**：自回归 VLA 微调方案，采用并行解码但放弃了视觉 token 利用，与 FocusVLA 形成对比
- **Gated Attention（2025）**：FocusVLA 的 Channel-level Focus 直接采用其逐元素门控机制
- **VGGT（2025）**：Visual Geometry Grounded Transformer，FocusVLA 评估的三种视觉表征之一，提供隐式 3D 信息
- **VLA-Pruner（2026）**：同期工作，在 VLM 端做时间感知双层视觉 token 剪枝以加速推理，与 FocusVLA 在策略端做选择的动机不同
