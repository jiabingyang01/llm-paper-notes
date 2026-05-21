# DAM-VLA：基于动态动作模型的 VLA 框架——手臂运动与夹爪操作的解耦与协调

> 论文：*DAM-VLA: A Dynamic Action Model-Based Vision-Language-Action Framework for Robot Manipulation*
>
> 作者：Xiongfeng Peng, Jiaqian Yu, Dingzhe Li 等
>
> 机构：Samsung R&D Institute China-Beijing (SRCB)、Samsung AI Center
>
> 发布时间：2026年3月
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.00926)
>
> 分类标签：`VLA 基础模型` `扩散动作头` `动作路由` `双尺度加权` `手臂-夹爪解耦`

---

## 一句话总结

观察到手臂运动（粗动作）和夹爪操作（精细动作）在路径约束、视觉注意力和数据分布上有本质差异，提出 **DAM-VLA**：用 VLM 推理驱动**动作路由**选择专用的扩散动作模型（手臂用全局 class token、夹爪用局部 register token），配合**双尺度动作加权**（轨迹级非对称高斯 + chunk 级指数衰减）协调两模型训练，SIMPLER 平均成功率 78-83% 超过 CogACT/OpenVLA，真实世界 pick-and-place 达 86.8%。

---

## 一、问题与动机

### 1.1 当前 VLA 的统一动作预测瓶颈

现有 VLA 方法（无论是自回归离散化如 OpenVLA、RT-2，还是扩散动作头如 $\pi_0$、CogACT）都用**同一个动作模型**预测所有类型的动作。但机器人操作任务中存在两种本质不同的动作阶段：

| 特征 | 手臂运动（Arm Movement） | 夹爪操作（Gripper Manipulation） |
| --- | --- | --- |
| **路径约束** | 弱约束，多条路径可达 | 强约束，需精确抓取姿态 |
| **视觉注意力** | 全局场景理解 | 局部精细聚焦 |
| **数据分布** | episode 中占比多 | 占比少但对成功至关重要 |

用同一模型处理这两种截然不同的动作类型，相当于让一个"通才"同时精通粗定位和精细操作——这天然存在冲突。

### 1.2 现有扩散 VLA 的局限

$\pi_0$、CogACT、TinyVLA 等方法在 VLM 后接扩散头，但存在以下问题：

- **仅依赖 VLM 提取的高层特征**：丢失了低层视觉细节，不利于精细操作
- **单一扩散头处理所有动作**：无法针对不同动作阶段做专门优化
- **CoT 类方法**（如 ECoT）推理 token 过多（350 vs OpenVLA 的 7），严重降低控制频率

---

## 二、核心方法

DAM-VLA 由三个核心组件构成：VLM 骨架、动作路由、动态动作模型。

### 2.1 整体架构

给定语言指令 $l$ 和视觉观测 $o_t$，模型预测动作序列 $[a_t, a_{t+1}, \ldots, a_{t+N}] = \pi(l, o_t)$，其中 $a_t = [\delta x, \delta\theta, s^{grip}]$ 是 7-DoF 动作（平移、旋转、夹爪状态）。

### 2.2 Vision-Language Model

**视觉编码器**：同时使用 DINOv2 和 SigLIP，输出三种 token：

- $f^{vis}$：视觉 token 序列——送入 LLM 做多模态融合
- $f^{cls}$：DINOv2 的 class token——**全局**视觉表征，用于手臂运动模型
- $f^{reg}$：DINOv2 的 register token——**局部**视觉表征，用于夹爪操作模型

**LLM 骨干**：LLaMA-2，将 $f^{vis}$ 和语言 token $f^{lan}$ 拼接后因果注意力处理，输出：

- $f^{rea}$：**推理隐向量**（第 2 层 Transformer 输出）→ 用于动作路由
- $f^{cog}$：**认知隐向量**（最后一层输出）→ 用于动作预测

**为什么从不同层提取？** 浅层保留更多直觉判断信息（适合快速路由决策），深层融合了完整的多模态推理（适合精确动作预测）。

### 2.3 动作路由机制（Action Routing）

利用 VLM 的推理能力判断当前处于手臂运动还是夹爪操作阶段：

$$\mathcal{L}_{class} = \| -(\hat{w} \log(w) + (1-\hat{w}) \log(1-w)) \|_1$$

其中 $w$ 是路由预测权重（$f^{rea}$ 经 FC + 归一化），$\hat{w} \in \{0, 1\}$ 是标签（基于夹爪状态变化：开→闭或闭→开时从 0 切换为 1）。

**测试时**：$w < 0.5$ 时执行手臂运动模型，$w \geq 0.5$ 时执行夹爪操作模型。

### 2.4 动态动作模型（Dynamic Action Model）

两个专用的 DiT（Diffusion Transformer）扩散模型并行训练：

**手臂运动模型**：
- 条件输入：$f^{cog}$（高层认知）+ $f^{cls}$（全局视觉）
- 预测全局范围的粗动作

**夹爪操作模型**：
- 条件输入：$f^{cog}$（高层认知）+ $f^{reg}$（局部视觉）
- 预测精细的夹爪动作

两个模型的损失分别为：

$$\mathcal{L}_{move} = \| n^{move}_i - \hat{n}^{move} \|^2_{\sum \hat{w}^{move}}$$

$$\mathcal{L}_{mani} = \| n^{mani}_i - \hat{n}^{mani} \|^2_{\sum \hat{w}^{mani}}$$

其中 $\| \cdot \|_{\sum}$ 是马氏距离，用标签权重 $\hat{w}^{move}$ 和 $\hat{w}^{mani}$ 对误差项做加权。

**总损失**：

$$\mathcal{L} = w_1 \cdot \mathcal{L}_{move} + w_2 \cdot \mathcal{L}_{mani} + w_3 \cdot \mathcal{L}_{class}$$

其中 $w_1 = w_2 = 1.0$，$w_3 = 0.0001$。

### 2.5 双尺度动作加权（Dual-Scale Action Weighting）

如何确定 $\hat{w}^{move}$ 和 $\hat{w}^{mani}$？论文设计了双尺度加权机制：

**轨迹级权重 $w^t$（全局视角）：** 根据夹爪状态变化将轨迹分为运动/操作阶段，在每个操作阶段 $k$ 的夹爪状态转换点处放置**非对称高斯分布**：

$$w^t_k \sim \{N(u, \sigma_l^2),\ N(u, \sigma_r^2)\}$$

- $u$：夹爪状态转换的时间点
- $\sigma_l = 6$（前沿，较宽）：状态变化前需要更多准备
- $\sigma_r = 2$（后沿，较窄）：状态变化后很快回到运动阶段

汇总：$w^t = \text{Norm}(\sum_k w^t_k)$。

**Action-chunk 级权重 $w^a$（局部视角）：** 在每个预测 chunk 内，靠近当前时刻的动作更重要：

$$w^a_j = \gamma^j, \quad \gamma = 0.8$$

**多尺度融合：**

$$w^{move} = (1 - w^t) \odot w^a, \quad w^{mani} = w^t \odot w^a$$

路由标签：$\hat{w} = 1$ if $w^t > 0.5$，否则 $\hat{w} = 0$。

---

## 三、实验结果

### 3.1 训练配置

- 预训练数据：Open X-Embodiment 中的 Fractal + BridgeDataV2
- 硬件：8×H100 GPU，约 2 天
- 学习率 $2 \times 10^{-5}$，batch size 256
- 微调：FurnitureBench 500 条专家轨迹 / 真实世界 50 条遥操作轨迹

### 3.2 SIMPLER 仿真评估

**Google Robot（Visual Matching）：**

| 方法 | PCC | MN | OCD | ODPA | **平均** |
| --- | --- | --- | --- | --- | --- |
| RT-2-X | 79% | 78% | 25% | 4% | 46% |
| OpenVLA | 14% | 51% | 48% | 0% | 28% |
| CogACT | 92% | 82% | 75% | 39% | 72% |
| $\pi_0^*$ | 89% | 81% | 55% | 53% | 70% |
| **DAM-VLA** | **96%** | **84%** | **75%** | **78%** | **83%** |

在需要精细操作的 ODPA（Open Drawer and Place Apple）任务上，DAM-VLA (78%) 大幅超过 CogACT (39%) 和 $\pi_0^*$ (53%)。

**WidowX Robot（Visual Matching）：**

| 方法 | SoT | CoP | GoY | EiB | **平均** |
| --- | --- | --- | --- | --- | --- |
| CogACT | 63% | 50% | 25% | 71% | 52% |
| $\pi_0^*$ | 62% | 59% | 24% | 81% | 57% |
| **DAM-VLA** | **88%** | **71%** | 25% | **100%** | **71%** |

DAM-VLA 在 WidowX 上平均 71%，超过 $\pi_0^*$ 的 57% 和 CogACT 的 52%。

### 3.3 FurnitureBench 长时程评估

One-Leg 组装任务（5 步，50 次试验）：

| 方法 | Step 1 | Step 2 | Step 3 | Step 4 | Step 5（最终） |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 96% | 94% | 78% | 53% | 29% |
| CogACT | 98% | 96% | 96% | 56% | 42% |
| **DAM-VLA** | **100%** | **100%** | **100%** | **62%** | **56%** |

DAM-VLA 在前 3 步达 100%，最终成功率 56%（vs CogACT 42%、OpenVLA 29%）。在接触丰富的拧螺丝步骤（Step 5）的提升验证了双模型设计对精细操作的价值。

### 3.4 真实世界评估

Franka 机器人 pick-and-place（80 次试验）：

| 方法 | 分布内 | 分布外 | **平均** |
| --- | --- | --- | --- |
| CogACT | 65.7% | 60.0% | 62.9% |
| **DAM-VLA** | **91.4%** | **82.2%** | **86.8%** |

### 3.5 消融实验

| VT | ACW | TW | DAM | DL | Google VM | Google VA | WidowX VM | **平均** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | 64% | 61% | 50% | 58% |
| ✓ | ✓ | - | - | - | 78% | 68% | 53% | 66% |
| ✓ | ✓ | ✓ | ✓ | ✓ | **83%** | **81%** | **71%** | **78%** |
| - | - | - | ✓ | ✓ | 66% | 64% | 49% | 60% |

关键发现：
1. 仅加 Visual Tokens + Action Chunk Weights 已提升 8%（58→66%）
2. 完整框架达 78%，比基线提升 20%
3. 去掉 VT 和双尺度加权后（保留 DAM+DL），仅 60%——双尺度加权是关键

---

## 四、局限性与未来方向

1. **任务覆盖有限**：目前主要验证 pick-and-place 和家具组装，更多任务类型（如工具使用、柔性物体）有待验证
2. **GoY 任务提升不足**：Stack Green Block on Yellow Block 成功率仅 25%，与基线持平——精细协调和稳定性仍有改进空间
3. **仅两种动作模型**：当前仅区分手臂运动和夹爪操作，未来可扩展到更多动作类型（如双臂协调、工具使用等）

---

## 五、个人思考

### 5.1 手臂-夹爪解耦的核心洞察

DAM-VLA 的核心观察——手臂运动和夹爪操作有本质差异——虽然直觉上显而易见，但此前的 VLA 工作几乎都忽略了这一点。这个观察与 CogACT 将认知和动作解耦的思路互补：CogACT 解耦的是"想"和"做"，DAM-VLA 解耦的是"做"中的"粗"和"细"。

### 5.2 class token vs register token 的精妙设计

DINOv2 的 class token 捕捉全局语义，register token 则更关注局部结构——这恰好对应手臂运动（需要全局场景理解）和夹爪操作（需要局部精细信息）的视觉需求差异。这种利用预训练视觉模型内部不同 token 的策略，避免了为两种动作模型训练不同视觉编码器。

### 5.3 双尺度加权的设计哲学

非对称高斯分布（前沿 $\sigma_l=6$、后沿 $\sigma_r=2$）反映了一个经验性观察：夹爪状态变化前（如接近物体准备抓取）的动作比变化后（如抓住后开始移动）更需要精确控制。这与人类操作直觉一致。

### 5.4 与 $\pi_0$ 系列的对比

$\pi_0$ 用单一 flow matching 头处理所有动作，依赖大规模数据和模型容量来隐式学习粗/细动作的差异。DAM-VLA 通过显式解耦用更小的模型（LLaMA-2 7B vs $\pi_0$ 的 3B PaliGemma）在 SIMPLER 上取得更好结果。但 $\pi_0$ 的优势在于真实世界大规模多任务泛化——DAM-VLA 尚未在这个维度做充分验证。

### 5.5 动作路由的可扩展性

当前路由仅区分 2 种动作类型。如果扩展到多种类型（如推、拉、旋转、插入等），路由机制需要从二分类变为多分类或层次化路由——这可能是一个有趣的研究方向，类似于 MoE 中的专家选择机制。

---

## 参考

- **CogACT**（Li et al., 2024）：认知-动作解耦的 VLA，DAM-VLA 的主要对比基线
- **$\pi_0$**（Black et al., 2024）：Flow Matching VLA 基础模型
- **OpenVLA**（Kim et al., 2024）：开源自回归 VLA
- **Diffusion Policy**（Chi et al., 2023）：将扩散模型引入机器人策略学习
- **DINOv2**（Oquab et al., 2023）：自监督视觉 Transformer，提供 class/register token
- **FurnitureBench**（Heo et al., 2023）：长时程接触丰富操作基准
