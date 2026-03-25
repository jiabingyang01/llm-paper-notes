# DeepVision-VLA：视觉基础模型表征增强的 VLA 框架

> **论文**：*Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models*
>
> **作者**：Yulin Luo, Hao Chen, Zhuangzhe Wu, Bowen Sui, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Qiuxuan Feng, Jiale Yu, Shuo Gu, Peng Jia, Pheng-Ann Heng, Shanghang Zhang
>
> **机构**：Peking University, Simplexity Robotics, The Chinese University of Hong Kong
>
> **发布时间**：2026年3月
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.15618)
>
> **发表会议**：暂未中稿（Under review）

---

## 一句话总结

DeepVision-VLA 发现 VLA 模型的 LLM 骨干在深层对任务相关视觉 token 的敏感度逐渐衰减，为此提出 VL-MoT 框架将视觉基础模型（DINOv3）的多层特征通过共享注意力注入到 VLA 深层，并引入 AGVP 策略利用浅层动作注意力剪枝无关视觉 token，RLBench 仿真 83%（+9% SOTA），真实世界 91.7%（+7.5%）。

---

## 一、问题与动机

### 1.1 视觉信息在深层的衰减

标准 VLA 将视觉特征一次性注入 LLM 的第一层，随后视觉信息随 Transformer 层堆叠而逐渐消散。作者通过两种互补的分析方法验证了这一现象：

**定性分析（Grad-CAM）**：对 OpenVLA（32 层）、π₀（18 层）、QwenVLA-OFT（36 层）三种 VLA 架构逐层可视化动作 token 对视觉 token 的贡献分布。结果一致显示：浅层中贡献分布集中于任务相关物体（被操控物体、机械臂）；深层中贡献分布弥散到背景区域，视觉定位能力显著退化。

**定量分析（逐层视觉 token 屏蔽实验）**：对每一层 $\ell$，将 ROI 区域（由 Grounding-DINO 自动检测）对应的视觉 token 以比例 $r$ 随机置零，测量最终动作预测的 MSE 变化。实验在 BridgeV2 随机采样的 1500 条轨迹上进行。结果显示：在浅层屏蔽 ROI token 会导致 MSE 大幅上升；而在深层即便完全屏蔽 ROI token，MSE 变化极小——说明深层动作预测已基本不依赖任务相关视觉信息。

### 1.2 根本原因

当前 VLA 的串行架构使视觉信息仅在第一层注入后单向向深传播，在多层 self-attention 和 FFN 的处理过程中，视觉信号逐渐被稀释和覆盖。与语言 token 和动作 token 的交互积累进一步削弱了任务相关视觉区域对动作生成的影响。

---

## 二、预备知识

### 2.1 基准模型 QwenVLA-OFT

作者自建的 baseline，基于 Qwen3-VL（4B）骨干，采用并行动作解码（OpenVLA-OFT 范式）+ $\ell_1$ 回归目标，与 OpenVLA-OFT 的区别在于：仅对动作 token 使用双向注意力，prompt token 保留因果注意力，以更好保留 Qwen3-VL 的预训练行为。

视觉编码器为 SigLIP2-Large（0.3B），使用 2D-RoPE + 插值绝对位置编码，相邻 $2 \times 2$ token 合并后投影到 LLM 隐维度 $d$，得到 $\mathbf{V} = E_\text{vis}(I) \in \mathbb{R}^{N_v \times d}$。

动作解码器为两层 MLP，将最终动作嵌入 $\mathbf{Z}_A^L$ 映射到动作空间 $\mathbf{a} \in \mathbb{R}^n$。

### 2.2 DINOv3

视觉基础模型，提供空间细粒度的图像表征。与 SigLIP 的图文对齐预训练目标不同，DINOv3 侧重空间结构和语义一致性，更适合需要精确空间感知的操作任务。

---

## 三、核心方法

### 3.1 整体框架

DeepVision-VLA 在 QwenVLA-OFT 基础上引入两个核心组件：

$$\text{DeepVision-VLA} = \text{QwenVLA-OFT} + \underbrace{\text{VL-MoT}}_{\text{视觉专家深层注入}} + \underbrace{\text{AGVP}}_{\text{动作引导视觉剪枝}}$$

整体推理流程：先通过浅层 VLA 计算动作到视觉的注意力图（AGVP），利用该图剪枝 Vision Expert 的 token，再将剪枝后的多层 Vision Expert 特征通过 VL-MoT 注入到 VLA 深层，最后动作头解码得到动作序列。

### 3.2 VL-MoT（Vision-Language Mixture-of-Transformers）

**设计思路**：不在输入层拼接 Vision Expert 特征，而是让 Vision Expert 的中间层与 VLA 深层在注意力层面直接共享 QKV——相当于让两个 Transformer 在深层"合并计算"，而非串联处理。

**耦合机制**：

设 Vision Expert 第 $k$ 层特征为 $\mathbf{E}^k \in \mathbb{R}^{M \times d_e}$，VLA 第 $\ell$ 层 token 为 $\mathbf{Z}^\ell \in \mathbb{R}^{N \times d}$，分别计算各自的 QKV：

$$Q_E = \mathbf{E}^k W_Q^E,\quad K_E = \mathbf{E}^k W_K^E,\quad V_E = \mathbf{E}^k W_V^E$$

$$Q_Z = \mathbf{Z}^\ell W_Q^Z,\quad K_Z = \mathbf{Z}^\ell W_K^Z,\quad V_Z = \mathbf{Z}^\ell W_V^Z$$

由于两者隐维度可能不同，Vision Expert 的 QKV 经可学习线性层对齐到 LLM 维度后拼接：

$$Q = [Q_E;\, Q_Z],\quad K = [K_E;\, K_Z],\quad V = [V_E;\, V_Z]$$

$$A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),\quad H = AV$$

输出 $H = [H_E;\, H_Z]$ 后分拆回两路，各自继续剩余的 Transformer 操作。Vision Expert 保持双向注意力，VLA 保持原有的混合因果/双向注意力策略（见 3.1 节基准）。

**特征选择策略**：Vision Expert 选用**最后 $n$ 层**（而非最前 $n$ 层或均匀采样 $n$ 层）与 VLA 最后 $n$ 层耦合。消融实验证实这是最优策略——DINOv3 深层编码高层语义、类感知的空间表征，与 VLA 动作条件化特征最兼容。

**大白话**：浅层 VLA 已经能聚焦任务相关区域，不需要额外帮忙；深层 VLA 已经"忘记看"了，在这里插入视觉专家的眼睛重新聚焦任务区域。

### 3.3 AGVP（Action-Guided Visual Pruning）

**动机**：Vision Expert 使用高分辨率（512×512）输入以获取细粒度特征，但直接将全量高分辨率 token 注入 VLA 会引入大量冗余背景。AGVP 利用浅层 VLA 的视觉定位能力筛选出与操作任务相关的 token，只将这部分 token 传递给 Vision Expert 和 VLA 深层共享注意力。

**注意力图计算**：

设 $\mathbf{A}^\ell \in \mathbb{R}^{N_a \times N_v}$ 为第 $\ell$ 层的动作-视觉注意力图，先在动作 token 维度取均值：

$$\mathbf{m}^\ell = \frac{1}{N_a} \sum_{i=1}^{N_a} \mathbf{A}^\ell_{i,:}$$

再在浅层集合 $\mathcal{L}_s$ 上取平均（实验中取第 4–19 层）：

$$\mathbf{m} = \frac{1}{|\mathcal{L}_s|} \sum_{\ell \in \mathcal{L}_s} \mathbf{m}^\ell$$

**Token 选择**：将 $\mathbf{m}$ 插值到 Vision Expert 分辨率得到 $\tilde{\mathbf{m}} \in \mathbb{R}^{N_d}$，选取重要性最高的 Top-K token：

$$\mathcal{S}_K = \text{TopK}(\tilde{\mathbf{m}},\, K), \quad \bar{\mathbf{E}}^k = \mathbf{E}^k[\mathcal{S}_K]$$

**三种引导信号的对比**：

| 引导方式 | 成功率 |
| --- | --- |
| DINOv3 CLS token | 65.5% |
| 指令-视觉注意力 | 84.0% |
| **动作-视觉注意力（AGVP）** | **88.0%** |

CLS token 只有全局语义；指令注意力对机械臂和臂-物体交互区域感知不足；动作 token 的注意力天然编码了任务意图和动作条件化的视觉定位，是最有效的剪枝引导信号。

### 3.4 训练流程

- **预训练数据**：Open X-Embodiment + DROID + RoboMIND，过滤后共 400K 条轨迹，训练 1 epoch
- **仿真微调**：RLBench 各任务 100 条轨迹，AdamW，8 NVIDIA H20，300 epochs
- **双分辨率输入**：VLA 分支使用 256×256，Vision Expert 使用 512×512
- **耦合层数**：DINOv3-H 最后 16 层 ↔ QwenVLA-OFT 最后 16 层；剪枝比例 0.5

---

## 四、实验结果

### 4.1 仿真实验（RLBench，10 任务）

| 方法 | 参数量 | Close Box | Sweep Dustpan | Wine Rack | 平均 S.R. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 7B | 0.60 | 0.20 | 0.20 | 0.40 |
| SpatialVLA | — | 0.80 | 0.15 | 0.15 | 0.46 |
| CogACT | 7B+300M | 0.90 | 0.50 | 0.30 | 0.61 |
| CoT-VLA | — | 0.95 | 0.50 | 0.55 | 0.66 |
| π₀.₅ | — | 0.90 | 0.05 | 0.75 | 0.65 |
| HybridVLA | — | 0.85 | 0.50 | 0.50 | 0.74 |
| QwenVLA-OFT | ~4B | 0.95 | 0.65 | 0.65 | 0.69 |
| **DeepVision-VLA** | **~5B** | **1.00** | **0.75** | **0.85** | **0.83** |

在 10 项任务中 8 项达到最高成功率。对比自身基线 QwenVLA-OFT，Sweep to Dustpan 提升最大（0.65→0.95，+46%）。

### 4.2 真实世界实验（Franka Research 3，4 任务）

| 方法 | Stack Coke Cans | Write 'S' | Pick Fruit (S1→S2) | Pour Coke (S1→S2) | 平均 |
| --- | --- | --- | --- | --- | --- |
| π₀.₅ | 0.65 | 0.95 | 0.75→1.00 | 1.00→0.70 | 0.842 |
| OpenVLA-OFT | 0.50 | 0.85 | 0.80→0.80 | 0.75→0.70 | 0.717 |
| QwenVLA-OFT | 0.50 | 0.80 | 0.85→0.90 | 0.70→0.70 | 0.742 |
| **DeepVision-VLA** | **0.65** | **0.95** | **0.95→0.95** | **1.00→1.00** | **0.917** |

Pour Coke to Bottle 第二步（精确倒水对准）达到 100%，而 π₀.₅ 在该步骤降至 70%，体现了细粒度视觉定位的价值。

### 4.3 消融实验

#### VL-MoT 融合范式

| 范式 | 成功率 |
| --- | --- |
| QwenVLA-OFT（无视觉专家） | 65.5% |
| Early Fusion（输入层拼接 DINOv3） | 73.0% |
| Mid Align（中间层对齐到冻结 DINOv3） | 67.0% |
| **Deep MoT（本文方法）** | **88.0%** |

Mid Align（即 [Don't blind your VLA](../foundation/SF_2025.md) 类路线）仅 67%，低于 Early Fusion，说明对齐到冻结通用特征不如直接利用其多层表征。

#### Vision Expert 特征选择

| 策略 | 成功率 |
| --- | --- |
| 首 16 层 | 61.5% |
| 均匀采样 16 层 | 85.0% |
| **最后 16 层（本文）** | **88.0%** |
| SigLIP 均匀 16 层 | 77.0% |

DINOv3 最后 16 层 > SigLIP 均匀 16 层，说明预训练目标（空间细粒度 vs. 图文对齐）对操作任务有显著影响。

#### AGVP 参考层分析

| 配置 | 成功率 |
| --- | --- |
| 第 4 层 | 85.0% |
| 第 8 层 | 69.0% |
| 第 12 层 | 82.5% |
| 第 16 层 | 87.5% |
| **第 4–19 层平均（本文）** | **88.0%** |

多层平均最优，说明单层注意力存在噪声，跨层平均能有效抑制对无关区域的随机注意力。

### 4.4 零样本泛化实验

在 Pick Fruit 任务上测试未见背景和光照变化：

| 场景 | QwenVLA-OFT（Step 1） | DeepVision-VLA（Step 1） |
| --- | --- | --- |
| 原始 | 0.85 | 0.95 |
| 新背景 | 0.70（−18%） | **0.90（−5%）** |
| 新光照 | 0.70（−18%） | **0.80（−16%）** |

Visual Expert 的视觉增强使模型对环境扰动更鲁棒，性能衰减幅度更小。

---

## 五、局限性与未来方向

1. **计算开销**：引入 DINOv3-H（0.8B）显著增加参数量和推理计算，实用部署需要进一步加速
2. **Vision Expert 的选择**：本文固定使用 DINOv3，未探索其他视觉专家（如 SAM、DepthPro）
3. **仅在 RLBench 和少量真实任务验证**：没有在 LIBERO、CALVIN 等标准基准上与更多工作对比
4. **耦合层数的确定依赖手动分析**：需要通过 Grad-CAM 实验确定浅/深层边界，增加了流程复杂性

---

## 六、个人思考

### 6.1 「深层视觉衰减」的诊断是核心贡献

本文最重要的贡献不是方法，而是这个系统性的观察本身。通过在三种不同架构（自回归的 OpenVLA、flow-based 的 π₀、L1 回归的 QwenVLA-OFT）上重现了一致的衰减曲线，说明这是 VLA 架构的结构性问题，而非某个特定设计的缺陷。这个诊断为后续的大量改进工作提供了理论依据。

### 6.2 与 SF（Spatial Foundation）的对比

项目中已有的 [SF](SF_2025.md) 也利用 VGGT 视觉基础模型增强 VLA，但路线完全不同：

- **SF**：将 VLA 中间层视觉 embedding 与冻结 VGGT 特征对齐（知识蒸馏风格），推理零开销
- **DeepVision-VLA**：在深层运行时注入 DINOv3 多层特征（共享注意力），引入额外参数和计算

SF 的消融实验中 Mid-level Align 方案（类似 DeepVision-VLA 的 Mid Align 基线）也仅有有限提升，与本文消融中 Mid Align 67% 的结论吻合——两篇论文从不同角度证明了「中间层对齐到冻结特征」的上限较低。DeepVision-VLA 的「深层运行时注入」是更激进但更有效的路线。

### 6.3 AGVP 与 [VLA-Pruner](../efficient/VLA_Pruner_2025.md) 的对比

两者都使用动作-视觉注意力进行 token 剪枝：

- **VLA-Pruner**：剪枝目标是 VLA 本身的视觉 token，用于加速推理（效率导向）
- **AGVP**：剪枝目标是 Vision Expert 的 token，用于引导高质量特征注入（性能导向）

两者设计动机完全不同，但都印证了浅层动作注意力作为视觉重要性信号的有效性。

### 6.4 深层注入 vs 浅层注入的设计哲学

本文消融发现 Early Fusion（输入层拼接）73% < Deep MoT 88%，证明「在哪里注入」比「注入什么」更关键。背后逻辑：浅层 VLA 已具备良好视觉定位，不缺视觉信号；深层才是视觉信息被稀释的地方，需要在那里补充。这与本文的诊断一脉相承，方法论上有内在一致性。

---

## 参考

- **Qwen3-VL**（Bai et al., 2025）：QwenVLA-OFT 基础 VLM 骨干
- **DINOv3**（Siméoni et al., 2025）：视觉专家，提供细粒度空间表征
- **OpenVLA-OFT**（Kim et al., 2025）：并行动作解码基准
- **HybridVLA**（Liu et al., 2025）：仿真对比基线
- **π₀.₅**（Physical Intelligence, 2025）：真实世界对比基线
- **Grad-CAM**（Selvaraju et al., 2017）：视觉 token 贡献分析工具
