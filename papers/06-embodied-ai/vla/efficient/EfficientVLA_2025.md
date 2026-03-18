# EfficientVLA：训练无关的结构化 VLA 推理加速框架

> **论文**：*EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models*
>
> **作者**：Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, Linfeng Zhang
>
> **机构**：Shanghai Jiao Tong University、Harbin Institute of Technology、Xi'an Jiaotong University、UESTC
>
> **发布时间**：2025 年 6 月
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2506.10100)
>
> **发表会议**：NeurIPS 2025

---

## 一句话总结

针对 Diffusion-based VLA 的三大冗余源（语言模块深度冗余、视觉 token 冗余、扩散动作头时序冗余），提出结构化 training-free 加速框架 EfficientVLA，协同执行 LLM 层剪枝 + 任务感知视觉 token 选择 + 扩散步缓存，在 CogACT 上实现 **1.93× 加速、FLOPs 降至 28.9%**，成功率仅降 0.6%。

---

## 一、问题与动机

### 1.1 Diffusion-based VLA 的推理瓶颈

Diffusion-based VLA（如 CogACT、π₀、DexVLA）通常由三个模块组成：

| 模块 | 参数量 | 推理时间 | FLOPs |
| --- | --- | --- | --- |
| Vision Module（DINOv2 + SigLIP） | 802.3M | 24.9ms | 405.50G |
| Language Module（LLaMA2-7B） | 6738.9M | 134.5ms | 3726.55G |
| Action Module（DiT, 10 步去噪） | 89.0M | 51.5ms | 57.96G |

**语言模块**占据绝大部分计算量和延迟，而**扩散动作头**的迭代去噪也贡献了显著开销。

### 1.2 现有方法的碎片化困境

已有 VLA 加速工作存在一个根本问题：**只优化单一模块，无法解决全局瓶颈**。

- **TinyVLA、DeeR-VLA**：需要定制架构 + 重新训练，非通用加速框架
- **Mole-VLA**：处理 LLM 层冗余但需要代价高昂的重训练，且忽略其他管道阶段
- **VLA-Cache**：缓存帧间静态 token，但受限于 LLM 内存瓶颈和动作头计算量，加速有限（仅 1.38×）
- **FastV**：仅剪枝视觉 token，在 LLM memory-bound 的情况下加速效果饱和（仅 1.21×）

### 1.3 三重冗余的系统性发现

EfficientVLA 通过系统分析识别了三类冗余：

**① 深度冗余（Language Module）**：相邻层隐状态的余弦相似度极高（特别是深层），说明许多层的变换 $f(\boldsymbol{x}^{(\ell)}, \theta^{(\ell)})$ 效果微乎其微。

**② 视觉 token 冗余**：大量 token 与任务无关或信息重复。但单纯减少 token 数量的收益很快饱和——当 token 减少到一定程度后，系统从 computation-bound 转变为 **memory-bound**，瓶颈转移到 LLM 的参数访存上。

**③ 时序冗余（Action Head）**：扩散去噪的相邻步之间，Attention 和 MLP 的中间特征高度相似，存在大量近乎静态的重复计算。

> 核心洞察：**优化单一模块只是把瓶颈转移到别处，必须协同优化三个模块才能实现真正的加速。**

---

## 二、预备知识

### 2.1 Diffusion-based VLA 架构

标准 Diffusion-based VLA 的推理流程：

1. **Vision Module**：视觉编码器（DINOv2 + SigLIP）将图像 $O_{\text{img}}$ 编码为特征 $F_V$
2. **Language Module**：LLM（如 LLaMA2-7B）融合 $F_V$ 和语言指令，输出任务表征 $F_{VL}$
3. **Action Module**：DiT 以 $F_{VL}$ 为条件，通过 $T$ 步去噪生成 7-DoF 动作序列

### 2.2 Transformer 层的残差结构

每层 $\ell$ 的计算可表示为残差变换：

$$\boldsymbol{x}^{(\ell+1)} = \boldsymbol{x}^{(\ell)} + f(\boldsymbol{x}^{(\ell)}, \theta^{(\ell)})$$

当 $f(\boldsymbol{x}^{(\ell)}, \theta^{(\ell)}) \approx \boldsymbol{0}$ 时，该层对表征的变换贡献极小，可以被安全移除。

---

## 三、核心方法

EfficientVLA 由三个协同策略组成：**(1) LLM 层剪枝**、**(2) 任务感知视觉 token 选择**、**(3) 扩散动作头特征缓存**。

### 3.1 Language Module 层剪枝

#### 3.1.1 基于相似度的层重要性评分

定义第 $\ell$ 层的重要性分数 $I^{(\ell)}$——衡量该层对输入隐状态的变换程度：

$$I^{(\ell)} = 1 - \frac{1}{|\mathcal{D}|}\sum_{i=1}^{|\mathcal{D}|}\left(\frac{1}{L}\sum_{j=1}^{L}\frac{\boldsymbol{x}^{(\ell)}_{i,j}\cdot\boldsymbol{x}^{(\ell+1)}_{i,j}}{\|\boldsymbol{x}^{(\ell)}_{i,j}\|_2\|\boldsymbol{x}^{(\ell+1)}_{i,j}\|_2}\right)$$

其中 $\boldsymbol{x}^{(\ell)}_{i,j}, \boldsymbol{x}^{(\ell+1)}_{i,j} \in \mathbb{R}^d$ 分别是第 $\ell$ 层对样本 $i$ 位置 $j$ 的输入和输出隐状态。

**直觉**：如果某层的输入和输出高度相似（余弦相似度接近 1），说明该层几乎没有改变表征，$I^{(\ell)}$ 就接近 0，即"不重要"。

#### 3.1.2 非连续层剪枝

对 $N$ 层 LLM 的所有层计算 $I^{(\ell)}$，按升序排列得到 $\mathcal{L}_{\text{ranked}} = [\ell_{(1)}, \ell_{(2)}, \ldots, \ell_{(N)}]$（$I^{(\ell_{(1)})} \leq I^{(\ell_{(2)})} \leq \cdots$），然后**移除前 $n$ 个最不重要的层**。

与连续剪枝（如 ShortGPT 只删尾部）不同，EfficientVLA 采用**非连续剪枝**——可以从任意位置移除冗余层。例如从 32 层剪到 22 层（移除 10 层，参数量 ↓41%）。

此外还配合 PruneNet 对剩余层的 MLP 施加 25% 稀疏度，进一步压缩。

### 3.2 任务感知视觉 Token 选择

从初始 $N_{\text{total}} = 256$ 个视觉 token 中选出 $K_{\text{final}}$（如 56）个，分三步完成：

#### 3.2.1 量化任务相关性

利用 VLM 层的交叉注意力分数衡量每个视觉 token $v_i$ 对任务的相关性。$A^{(h)}_{i,j}$ 是视觉 token $v_i$ 对第 $j$ 个上下文 token（语言指令）在第 $h$ 个注意力头的注意力值：

$$r_i = \sum_{j=1}^{L_{\text{ctx}}}\left(\frac{1}{H}\sum_{h=1}^{H}A^{(h)}_{i,j}\right)$$

然后通过 min-max 归一化得到标准化分数 $s_i \in [0,1]$。

#### 3.2.2 选择核心任务 Token（$V_{\text{key}}$）

选取任务相关性最高的 $K_{\text{key}}$（经验值 4~8）个 token 作为核心集合：

$$V_{\text{key}} = \{v_i \in V \mid s_i \text{ 在所有分数中排名前 } K_{\text{key}}\}$$

这些 token 无条件保留，构成最关键的视觉线索基底。

#### 3.2.3 增量选择：任务相关 + 多样性平衡

还需补充 $K_{\text{aug}} = K_{\text{final}} - K_{\text{key}}$ 个 token，通过比例 $\alpha$ 控制两种选择策略的分配：

**任务驱动增量**：从剩余候选中按 $s_i$ 降序再选 $K_{\text{task}} = \lfloor\alpha \cdot K_{\text{aug}}\rfloor$ 个。

**多样性驱动增量**：剩余 $K_{\text{div}} = K_{\text{aug}} - K_{\text{task}}$ 个 token 通过最大化与 $V_{\text{key}}$ 的特征差异来选择：

$$\text{Diversity}(v_j, V_{\text{key}}) = 1 - \max_{v_k \in V_{\text{key}}} \frac{v_j \cdot v_k}{\|v_j\|_2\|v_k\|_2}$$

选择差异度最大的 $K_{\text{div}}$ 个 token 组成 $V_{\text{div}}$，确保最终集合不会过度特化。

**最终保留集合**：

$$V_{\text{pruned}} = V_{\text{key}} \cup V_{\text{task}} \cup V_{\text{div}}$$

> 用大白话说：先选出"任务最关键的"视觉区域（如目标物体），再补充"任务相关但排名靠后的"区域，最后补充"虽然任务相关性不高但信息量独特的"区域（如场景边缘的参照物），从而在精简 token 数量的同时保留充分的视觉信息。

### 3.3 扩散动作头特征缓存

#### 3.3.1 DiT 的时序特征冗余

DiT 在每个去噪步 $t$ 中对输入特征 $\boldsymbol{z}_t$ 计算：

$$\boldsymbol{h}^{\text{attn}}_t = \text{Self-Attn}(\boldsymbol{z}_t)$$

$$\boldsymbol{h}^{\text{mlp}}_t = \text{MLP}(\boldsymbol{h}^{\text{attn}}_t + \boldsymbol{z}_t)$$

实验观察到 $\boldsymbol{h}^{\text{module}}_t \approx \boldsymbol{h}^{\text{module}}_{t-1}$，即相邻去噪步的中间特征高度相似。

#### 3.3.2 静态 N 步缓存

设缓存间隔为 $N$。在初始步 $t = T_{\text{start}}$ 计算并缓存 $\mathcal{C}_{\text{attn}}, \mathcal{C}_{\text{mlp}}$。后续仅当 $t \bmod N = 0$ 时重新计算并更新缓存：

$$\mathcal{C}_{\text{attn}} \leftarrow \text{Self-Attn}(\boldsymbol{z}_t), \quad \mathcal{C}_{\text{mlp}} \leftarrow \text{MLP}(\mathcal{C}_{\text{attn}} + \boldsymbol{z}_t)$$

其余步直接复用缓存值：

$$\boldsymbol{h}^{\text{attn}}_t \leftarrow \mathcal{C}_{\text{attn}}, \quad \boldsymbol{h}^{\text{mlp}}_t \leftarrow \mathcal{C}_{\text{mlp}} \quad (\text{当 } t \bmod N \neq 0)$$

以 $N=5$ 为例，原本 10 步去噪中只需完整计算 2 步（$t=10$ 和 $t=5$），其余 8 步直接复用，**去噪步数等效降低 80%**。

### 3.4 方法总览

| 策略 | 目标冗余 | 关键机制 | 效果 |
| --- | --- | --- | --- |
| LLM 层剪枝 | 深度冗余 | 基于余弦相似度的重要性评分 + 非连续剪枝 | 参数 ↓41%、FLOPs ↓78% |
| 视觉 Token 选择 | Token 冗余 | 任务相关性 + 多样性双重驱动 | 256→56 token（↓78%） |
| 扩散步缓存 | 时序冗余 | 静态 N 步 Attn/MLP 缓存复用 | 去噪步 10→2 等效（↓80%） |

---

## 四、实验结果

### 4.1 实验设置

- **基座模型**：CogACT（DINOv2 + SigLIP 视觉编码、LLaMA2-7B 语言模块、DiT 动作头）
- **评估环境**：SIMPLER（桌面操控仿真，Visual Matching + Variant Aggregation 两种配置）
- **对比方法**：Random Dropping、FastV、VLA-Cache
- **硬件**：NVIDIA A40 GPU

### 4.2 主实验结果

**Visual Matching 配置**：

| 方法 | Training-free | PickCan | MoveNear | Drawer | DrawerApple | 平均 | FLOPs↓ | 加速↑ | 参数(B) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CogACT（基线） | - | 91.3% | 85.0% | 71.8% | 50.9% | 74.8% | 100% | 1.00× | 7.63 |
| Random Dropping | ✓ | 9.7% | 20.4% | 53.5% | 0.0% | 20.9% | 58.5% | 1.20× | 7.63 |
| FastV | ✓ | 92.6% | 81.4% | 69.8% | 52.4% | 74.1% | 42.0% | 1.21× | 7.63 |
| VLA-Cache | ✓ | 92.0% | 83.3% | 70.5% | 51.6% | 74.4% | 80.1% | 1.38× | 7.63 |
| **EfficientVLA** (L=28,T=112) | ✓ | **95.3%** | 83.3% | 70.3% | **56.5%** | **76.4%** | 45.1% | 1.59× | 5.87 |
| **EfficientVLA** (L=22,T=56) | ✓ | 93.3% | 81.3% | 68.2% | 53.8% | 74.2% | **28.9%** | **1.93×** | **4.86** |

**关键发现**：

1. **剪掉 36% 参数反而涨点**：PickCan 任务上 91.3% → 94.0%（L=22, T=112），印证 VLA 模型存在显著参数冗余
2. **随机丢 token 灾难性崩溃**：平均成功率暴跌至 20.9%，说明视觉 token 的选择策略至关重要
3. **单一优化的天花板**：FastV 仅优化 token 只获得 1.21× 加速，VLA-Cache 仅缓存静态 token 只获得 1.38× 加速——都受限于未被优化的其他模块
4. **最激进配置（L=22, T=56）**：FLOPs 降至 28.9%、1.93× 加速，成功率仅降 0.6pp

### 4.3 可扩展性分析

| 模型 | Action-Params | CogACT 成功率 | EfficientVLA 成功率 | CogACT 延迟 | EfficientVLA 延迟 |
| --- | --- | --- | --- | --- | --- |
| CogACT-Small | 13M | 73.3% | 72.6% | 0.2156s | 0.1173s |
| CogACT-Base | 89M | 74.8% | 74.2% | 0.2342s | 0.1213s |
| CogACT-Large | 308M | 76.7% | 76.1% | 0.2628s | 0.1312s |

模型越大，加速效果越显著（Large 模型达 2.0× 加速），且性能损失始终在 1% 以内。

### 4.4 Token 数量与缓存间隔的影响

**视觉 Token 数量**（PickCan 任务）：

| 保留 Token | 56 | 72 | 96 | 112 | 256（全部） |
| --- | --- | --- | --- | --- | --- |
| 成功率 | 95.0% | 95.3% | 95.0% | 96.0% | 91.3% |
| 推理时间(s) | 0.1866 | 0.1870 | 0.1889 | 0.1956 | 0.2342 |

减少到 56 个 token 仍保持 95.0% 成功率（甚至比 256 全量更高！），但推理速度收益在 token 减少到一定程度后趋于饱和——这正是 memory-bound 瓶颈的体现。

**缓存间隔 N**：

| 间隔 N | 1（无缓存） | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 成功率 | 91.3% | 94.0% | 93.7% | 90.3% | 93.7% |
| 推理时间(s) | 0.2342 | 0.2031 | 0.1987 | 0.1953 | 0.1909 |

缓存间隔增大可渐进降低延迟，且在 $N=2$ 和 $N=5$ 时成功率反而提升，说明适度复用有正则化效果。

### 4.5 消融实验

| 配置 | Layer | MLP | Token | Cache | 成功率 | 推理时间 | 加速 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ex0（基线） | ✗ | ✗ | ✗ | ✗ | 91.3% | 0.2342s | 1.00× |
| Ex1（仅 Token） | ✗ | ✗ | ✓ | ✗ | 95.6% | 0.1866s | 1.25× |
| Ex2（仅 Cache） | ✗ | ✗ | ✗ | ✓ | 93.7% | 0.1909s | 1.23× |
| Ex4（Layer+MLP） | ✓ | ✓ | ✗ | ✗ | 92.3% | 0.1638s | 1.43× |
| Ex6（Token+Cache） | ✗ | ✗ | ✓ | ✓ | 95.3% | 0.1592s | 1.47× |
| **Ex7（全部）** | ✓ | ✓ | ✓ | ✓ | **93.3%** | **0.1213s** | **1.93×** |

- 单独优化 token 或 cache 各只获得约 1.2× 加速
- 模型压缩（Layer+MLP）贡献 1.43× 加速
- **三者协同才能达到 1.93×**，且成功率还提升了 2pp

---

## 五、局限性与未来方向

1. **Training-free 的天花板**：固定的缓存间隔 $N$ 无法自适应调整，未来可探索 adaptive caching
2. **模型覆盖有限**：目前仅在 CogACT 上验证，尚未扩展到 π₀ 等 Flow Matching 架构
3. **剪枝策略的泛化性**：基于余弦相似度的层重要性评分在不同 LLM 骨架上的表现有待验证

---

## 六、个人思考

### 6.1 与项目中已有 VLA 高效推理论文的对比

| 维度 | EfficientVLA | VLA-Cache | LAC | SD-VLA | RLRC |
| --- | --- | --- | --- | --- | --- |
| 核心策略 | 层剪枝 + Token 选择 + 扩散缓存 | 跨帧 KV 缓存 | 可学习 Token 缓存 | 静态/动态 Token 解耦 | 结构化剪枝 + RL 恢复 |
| 优化维度 | 三维度（LLM + Vision + Action） | 单维度（Vision Token） | 单维度（Vision Token） | 单维度（Vision Token） | 单维度（LLM 压缩） |
| Training-free | ✓ | ✓ | ✗ | ✗ | ✗ |
| 冗余类型 | 深度 + 空间 + 时序 | 时间（跨帧） | 时间（跨帧） | 时间（跨帧） | 参数冗余 |
| 加速倍率 | 1.93× | 1.38× | 1.76× | 2.26× | 2.3× |
| 性能变化 | -0.6pp | -0.4pp | +1.9pp | -0.6pp | -2.5pp |

EfficientVLA 的**独特价值在于"结构化"**——它第一次系统地分析了 VLA 全流水线的冗余并提出协同解决方案。其他方法要么只看 token（VLA-Cache、LAC、SD-VLA），要么只看模型压缩（RLRC），都存在瓶颈转移的问题。

### 6.2 "剪枝反而涨点"的洞察

PickCan 任务上 91.3% → 95.3%（L=28, T=112），这一反直觉结果与 VLA-Cache 在真实机器人上的观察（82.1% → 84.6%）形成呼应。共同指向一个重要结论：**当前 VLA 模型存在大量冗余计算，这些冗余不仅浪费算力，还可能干扰决策**。

从信息论角度看，过多的低信息量 token 会稀释注意力权重，降低模型对关键区域的聚焦能力。适度剪枝反而起到了类似 Dropout 的正则化效果。

### 6.3 Memory-Bound 瓶颈的深刻启示

EfficientVLA 最重要的分析贡献是明确揭示了 VLA 推理中 computation-bound 到 memory-bound 的转变——当视觉 token 减少到一定程度后，瓶颈从"算不完"变成"搬不动"（LLM 参数的显存带宽成为限制因素）。这解释了为什么 FastV 只剪 token 只能加速 1.21×，以及为什么必须同时压缩 LLM 本身。

这一洞察对 VLA 加速社区有普遍指导意义：**未来的加速方法不应孤立地优化单一模块，而应基于 roofline 分析判断当前的实际瓶颈在哪里**。

### 6.4 与 VLA-Cache 的互补性

VLA-Cache 利用的是**帧间时间冗余**（连续帧之间静态区域的 KV 可复用），而 EfficientVLA 利用的是**帧内冗余**（单帧内的 token 选择 + LLM 深度压缩）以及**扩散步间冗余**。两者在不同维度工作，理论上可以叠加——先用 EfficientVLA 压缩模型和选择 token，再用 VLA-Cache 在帧间复用剩余 token 的 KV，有望实现更大幅度的加速。

---

## 参考

- [CogACT](https://arxiv.org/abs/2411.19650) — Diffusion-based VLA 基座模型，本文的实验平台
- [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) — 训练无关跨帧 Token 缓存加速
- [LAC](/papers/06-embodied-ai/vla/efficient/LAC_2026) — 可学习自适应 Token 缓存加速
- [FastV](https://arxiv.org/abs/2403.06764) — VLM 视觉 token 剪枝加速
- [Mole-VLA](https://arxiv.org/abs/2503.20384) — 动态层跳跃 VLA
- [PruneNet](https://arxiv.org/abs/2501.15296) — 免校准模型压缩
