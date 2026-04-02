# SpatialVLA：探索空间表征增强视觉-语言-动作模型

> 论文：*SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model*
>
> 作者：Delin Qu*, Haoming Song*, Qizhi Chen*, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, Jiayuan Gu, Bin Zhao, Dong Wang, Xuelong Li
>
> 机构：Shanghai AI Laboratory, ShanghaiTech, TeleAI
>
> 发布时间：2025年1月
>
> 🔗 [arXiv](https://arxiv.org/abs/2501.15830) | [项目主页](https://spatialvla.github.io)
>
> 分类标签：`3D 空间感知` `自适应动作离散化` `跨构型预训练` `VLA 基础模型`

---

## 一句话总结

SpatialVLA 通过 **Ego3D Position Encoding**（将深度信息以自我中心 3D 编码注入视觉特征）和 **Adaptive Action Grids**（基于高斯分布自适应离散化动作空间，仅需 3 个 token 表征 7D 动作），在 PaliGemma 2 骨架上用 1.1M 真实机器人数据预训练，实现了 SOTA 的零样本泛化和高效后训练迁移能力，推理速度达 20 Hz。

---

## 一、问题与动机

### 1.1 VLA 模型缺乏 3D 空间理解

人类操作物体时会本能地构建丰富的 3D 空间心理模型——物体在哪、距离多远、方向如何。然而现有 VLA 模型（RT-2、OpenVLA 等）的输入仅为 2D RGB 图像，**缺乏对 3D 物理世界的精确感知和理解**，导致在需要空间推理的任务（如"把离机器人最近的毛绒玩具放到车上"）中表现不佳。

### 1.2 跨构型的两大核心挑战

构建具有 3D 空间智能的通才机器人策略面临两个根本性障碍：

**观测层面：3D 观测不对齐**
- 不同机器人的相机传感器种类各异，安装位置不同（腕部 / 第三人称视角）
- 不同观测构成了非标定的 3D 观测空间，无法直接统一

**动作层面：动作特征异构**
- 不同机器人自由度、运动控制器、工作空间配置、任务复杂度各不相同
- 导致跨构型学习可泛化的空间动作极为困难

### 1.3 现有动作离散化的不足

以 RT-2 和 OpenVLA 为代表的方法将每个动作维度均匀离散化为 256 个 bin，存在两个问题：

1. **均匀分箱浪费精度**：动作分布集中在中心附近（接近高斯），但均匀分箱在低频区域（极端动作）和高频区域（常见动作）分配了相同的精度
2. **token 数量多**：7D 动作需要 7 个 token 逐一自回归生成，推理速度慢

---

## 二、预备知识

### 2.1 VLA 模型范式

VLA 模型通过微调预训练 VLM 来生成机器人动作。标准流程：

$$A_t = \mathcal{F}(o_t, L)$$

- $o_t = \{I_t^1, \dots, I_t^n\}$：图像观测
- $L$：自然语言任务指令
- $A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}]$：预测的 $H$ 步动作序列

### 2.2 动作离散化与自回归生成

RT-2 和 OpenVLA 将连续动作离散化为 token 后用自回归交叉熵目标训练：

$$\mathfrak{L}(\theta) = \mathbb{E}_{p(A_t|o_t)} \mathcal{L}(a_t, \tilde{a}_t)$$

其中 $a_t$ 是真实动作 token，$\tilde{a}_t$ 是预测的动作 token。关键区别在于**如何将连续动作映射为 token**——这正是 SpatialVLA 的核心创新。

### 2.3 极坐标表示

SpatialVLA 将平移动作 $(x, y, z)$ 转换为极坐标 $(\phi, \theta, r)$，将**运动方向**（$\phi, \theta$）和**运动距离**（$r$）解耦。这一转换使得模型可以对方向分配更多精度（方向变化通常更精细），对距离分配较少精度。

---

## 三、核心方法

### 3.1 模型整体架构

SpatialVLA 基于 PaliGemma 2 VLM 构建，包含三个核心组件：

| 组件 | 功能 | 来源 |
| --- | --- | --- |
| SigLIP 视觉编码器 | 提取 2D 语义视觉特征 $X \in \mathbb{R}^{d \times h \times w}$ | PaliGemma 2 预训练 |
| Ego3D Position Encoding | 将 3D 空间上下文注入 2D 特征 | 本文创新 |
| Adaptive Action Grids | 将 7D 连续动作映射为 3 个空间 token | 本文创新 |
| Gemma2 LLM 骨架 | 自回归生成空间动作 token | PaliGemma 2 预训练 |
| ZoeDepth 深度估计 | 从 RGB 预测深度图 | 预训练冻结 |

输入为 $224 \times 224$ 的 RGB 图像和语言指令，输出为 $T=4$ 步 action chunk（共 12 个空间动作 token）。

### 3.2 Ego3D Position Encoding

**核心思想**：在自我中心相机坐标系中构建 3D 位置编码，无需机器人-相机外参标定，天然适用于任意机器人构型。

**Step 1 — 深度估计与反投影**：

使用 ZoeDepth 从 RGB 图像估计深度图 $D$，结合相机内参 $K$ 通过反投影 $\pi^{-1}$ 获取每个像素的自我中心 3D 位置：

$$p = \{x, y, z\} = \pi^{-1}(D, K)$$

得到 3D 位置图 $P \in \mathbb{R}^{3 \times h \times w}$。

**Step 2 — 位置编码融合**：

对 3D 位置施加正弦频率编码 $\gamma(\cdot)$ 后通过可学习 MLP 映射为与视觉特征同维度的位置嵌入，再与 SigLIP 提取的 2D 语义特征相加：

$$O_{3d} = X + P' = X + \text{MLP}(\gamma(P))$$

其中 $O_{3d} \in \mathbb{R}^{d \times h \times w}$ 是融合了 3D 空间信息的最终视觉表征。

**为什么用自我中心坐标系？** 自我中心坐标系以相机为原点，消除了对机器人-相机外参标定的依赖。不同机器人安装的相机位置不同，但自我中心坐标系下的几何关系（如物体在相机前方 30cm、偏右 10cm）是通用的。

### 3.3 Adaptive Action Grids

这是 SpatialVLA 最精巧的设计。核心思想是：**根据数据分布自适应地划分动作空间为 3D 网格，每个网格对应一个可学习的 token 嵌入**。

#### 3.3.1 动作分解

单臂机器人的 7D 动作被分解为三部分：

$$a = \{a_{\text{trans}}, a_{\text{rot}}, a_{\text{grip}}\}$$

- $a_{\text{trans}} = \{x, y, z\} \rightarrow (\phi, \theta, r)$：平移→极坐标
- $a_{\text{rot}} = \{\text{roll}, \text{pitch}, \text{yaw}\}$：旋转
- $a_{\text{grip}} = \{\text{grip}\}$：夹爪开/关（2 个离散 token）

#### 3.3.2 基于高斯分布的自适应网格划分

对预训练数据集中每个动作变量进行归一化到 $[-1, 1]$ 后，拟合高斯分布 $\mathcal{N}(\mu^a, \Sigma^a)$，然后按**等概率原则**划分为 $M$ 个区间：

$$a_2, \dots, a_M = \arg\min_{a_2, \dots, a_M} \left| \int_{a_i}^{a_{i+1}} f(x)dx - \frac{1}{M} \right|, \quad i = 1, \dots, M$$

其中 $f(x)$ 是高斯分布的概率密度函数。

**直觉理解**：数据密集的区域（分布中心，对应常见的小幅动作）被划分为更多、更精细的网格；数据稀疏的区域（分布尾部，对应极端动作）则分配较少网格。这就像是让模型在"常走的路"上看得更仔细。

#### 3.3.3 3D 空间网格与 token

平移和旋转各自构成 3D 空间网格：

- **平移空间**：$M_{\text{trans}} = M_\phi \cdot M_\theta \cdot M_r$ 个离散网格（$\phi$ 取 32 bins，$\theta$ 取 16 bins，$r$ 取 8 bins → 4096 个网格）
- **旋转空间**：$M_{\text{rot}} = M_{\text{roll}} \cdot M_{\text{pitch}} \cdot M_{\text{yaw}}$ 个离散网格（各 16 bins → 4096 个网格）

总 token 数 $V = M_{\text{trans}} + M_{\text{rot}} + 2 = 8194$。每个网格对应一个可学习的 $d$ 维嵌入向量：

$$E_a = \{E_{\text{trans}} \in \mathbb{R}^{d \times M_{\text{trans}}}, E_{\text{rot}} \in \mathbb{R}^{d \times M_{\text{rot}}}, E_{\text{grip}} \in \mathbb{R}^{d \times 2}\}$$

这些 token 嵌入与 LLM 的文本嵌入共享参数空间，线性化后无缝接入自回归训练流程。

#### 3.3.4 编码-解码流程

> **编码**（训练时）：
> 1. 将连续动作归一化后转换为极坐标
> 2. 用 `digitize` 将连续值映射到 3D 网格的离散索引
> 3. 将 3D 索引线性化为 1D 索引，查表获取 token 嵌入

> **解码**（推理时）：
> 1. 模型自回归预测 3 个 token 索引（trans, rot, grip）
> 2. 将 1D 索引还原为 3D 网格坐标（gridification）
> 3. 反归一化还原为连续动作

**关键优势：每步动作仅需 3 个 token**（平移 + 旋转 + 夹爪），而 RT-1/RT-2/OpenVLA 需要 7 个 token，推理速度提升显著。

### 3.4 预训练与后训练

#### 3.4.1 预训练

- **数据**：1.1M 真实机器人 episodes，由 OXE 子集和 RH20T 组成，共 28 个数据集
- **骨架**：从 PaliGemma 2 初始化
- **训练设置**：64×A100 GPU，10 天，batch size 2048，AdamW，lr=2e-5
- **两阶段**：
  - Stage 1：全数据集训练 160k 步（达 90% 准确率）
  - Stage 2：移除 DROID 数据集后继续训练 40k 步（达 95%+ 准确率）
- **关键设计**：冻结文本 token 嵌入 $E_{\text{text}}$，保留 VLM 的通用世界知识，仅训练空间动作 token 嵌入、MLP 和 VLM 骨架参数

#### 3.4.2 后训练：Spatial Embedding Adaption

这是 Adaptive Action Grids 带来的独特后训练方式。核心思路是**在新数据上重新拟合高斯分布，重新划分网格，并用三线性插值初始化新 token 嵌入**：

1. 对新数据集拟合新的高斯分布 $\mathcal{N}(\mu_{\text{new}}, \Sigma_{\text{new}})$
2. 根据新分布创建新的自适应网格 $G_{\text{new}}$
3. 对新网格中第 $i$ 个 token，找到预训练网格中的相邻网格 $G^{\text{adj}} = \{G^1, \dots, G^K\}$
4. 用三线性插值初始化新嵌入：

$$e^i_{a_{\text{new}}} = \sum_{j=1}^K w_j e^j$$

其中 $w_j$ 由新旧网格质心距离的归一化权重决定。

**直觉理解**：这相当于把预训练学到的"空间动作知识"按照新机器人的动作特征重新排布，实现了知识的软迁移。

---

## 四、实验结果

### 4.1 零样本控制（SimplerEnv）

在 Google Robot 和 WidowX 两个仿真设定上评估：

| 模型 | Google Robot Visual Matching | Google Robot Variant Agg. | WidowX Overall |
| --- | --- | --- | --- |
| RT-2-X (55B) | 60.7% | 64.3% | — |
| OpenVLA | 27.7% | 39.8% | 1.0% |
| RoboVLM (zero-shot) | 56.3% | 46.3% | 13.5% |
| **SpatialVLA (zero-shot)** | **71.9%** | **68.8%** | **34.4%** |
| **SpatialVLA (fine-tuning)** | **75.1%** | **70.7%** | **42.7%** |

SpatialVLA (3.5B) 零样本超越 55B 的 RT-2-X，Visual Matching +11.2%，Variant Aggregation +4.5%。

### 4.2 真实世界 WidowX 零样本

在 7 个任务 × 11 次试验中，SpatialVLA 展现出：
- **运动干扰鲁棒性**：在物体被人手动移动的情况下仍能成功抓取（#3、#4 任务）
- **指令跟随能力**：准确区分"把绿色杯子放到白色盘子上"vs"放到粉色布上"
- **平均成功率最高**：超越 Octo、RT-1-X、OpenVLA、RoboVLM

### 4.3 LIBERO 仿真微调

| 模型 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 78.3% | 92.5% | 68.3% | 50.5% | 72.4% |
| Octo | 78.9% | 85.7% | 84.6% | 51.1% | 75.1% |
| OpenVLA | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| TraceVLA | 84.6% | 85.2% | 75.1% | 54.1% | 74.8% |
| **SpatialVLA** | **88.2%** | **89.9%** | 78.6% | **55.5%** | **78.1%** |

SpatialVLA 在 LIBERO-Spatial 上取得 88.2%，充分验证了其空间理解能力。

### 4.4 Franka 真实机器人微调

跨三种评估场景的综合表现：

| 场景 | Diffusion Policy | Octo | OpenVLA | SpatialVLA |
| --- | --- | --- | --- | --- |
| 单任务 | 81% | 67% | 73% | **82%** |
| 指令跟随 | 26% | 33% | 47% | **59%** |
| 多任务 | 27% | 36% | 42% | **57%** |
| 综合 | 46% | 46% | 53% | **65%** |

SpatialVLA 在指令跟随和多任务上显著优于 Diffusion Policy（缺乏语言理解）和 OpenVLA。

### 4.5 空间理解能力评估

专门设计的空间任务评估（含高度变化、距离判断）：

| 模型 | Franka 微调：最近玩具 | WidowX 零样本：杯子高度变化 | WidowX 零样本：盘子高度变化 |
| --- | --- | --- | --- |
| OpenVLA | 45.5% | 27.3%/45.5% | 54.5% |
| **SpatialVLA** | **63.6%** | **72.7%/81.8%** | **63.6%** |

SpatialVLA 在空间理解任务上显著领先，验证了 Ego3D 编码的有效性。

### 4.6 消融实验

在 Google Fractal + BridgeData V2 混合数据集上的预训练消融：

| 消融设定 | Pick Coke Can (VA) | Move Near (VA) | 关键发现 |
| --- | --- | --- | --- |
| SpatialVLA（完整） | 81.6% | 79.2% | — |
| 替换为线性 256 bins | 40.7% | 47.1% | 自适应网格 vs 均匀分箱：**+36.5%/+32.1%** |
| 替换为均匀分布网格 | 77.9% | 64.2% | 高斯自适应 vs 均匀网格：+3.7%/+15.0% |
| 降低分辨率至 1026 | 74.4% | 59.1% | 高分辨率 8194 vs 低分辨率：+7.2%/+20.1% |
| 移除 Ego3D 编码 | 68.9% | 66.7% | 3D 编码：+12.7%/+12.5% |
| 不冻结 LLM embedding | 70.2% | 63.1% | 冻结文本嵌入：+11.4%/+16.1% |

后训练消融中，Spatial Embedding Adaption 在 LIBERO 小数据集上带来 +4.6% ~ +5.4% 的提升。

### 4.7 推理速度

| 模型 | 推理速度 (Hz) |
| --- | --- |
| RT-2-X | 5.0 |
| OpenVLA | 5.2 |
| TraceVLA | 6.5 |
| **SpatialVLA** | **20.1** |

SpatialVLA 每步仅需 3 个 token（vs OpenVLA 的 7 个），在单张 RTX 4090 上达 20 Hz，仅需 8.5GB 显存。

---

## 五、局限性与未来方向

### 5.1 高斯分布拟合的局限

论文自身指出高斯分布建模可能不是最优的：
- 极端场景（如单轴运动）下，高斯拟合会导致网格在某些坐标轴上过度集中，丧失其他轴的运动能力
- 数据噪声会扭曲网格分布
- 未来方向：结合 VAE 等隐式分布建模与显式网格划分

### 5.2 长时域能力不足

模型仅依赖当前帧观测和历史 token 预测动作，在 LIBERO-Long 等长时域任务上表现相对较弱（55.5%）。未来需设计高效的历史信息感知机制。

### 5.3 自回归 vs 扩散

尽管 SpatialVLA 通过减少 token 数实现了 21 Hz，但自回归方式仍慢于扩散解码（如 π₀ 的 flow matching 可一次性生成 50 步动作）。论文提出未来可以将扩散解码与空间网格动作表征结合。

### 5.4 数据质量

OXE 数据集质量参差不齐（如 FMB 数据导致机器人右偏、Kuka 缺乏清晰提示词），未来需要更好的数据筛选和组合策略。

---

## 六、个人思考

### 6.1 与其他 3D VLA 的定位对比

SpatialVLA 与项目中已有的多篇 3D VLA 形成有趣的对比：

- **3D-CAVLA**：用 PointNet 编码深度点云作为额外输入流，是显式的 3D 模块添加
- **SF**：用 VGGT 3D 表征做中间层监督对齐，推理零开销
- **BridgeVLA**：正交投影 3D→2D 热力图对齐输入输出
- **SpatialVLA**：深度信息通过位置编码加法融合到 2D 特征，属于轻量级但直接的 3D 注入

SpatialVLA 的 Ego3D 编码方案在简洁性和有效性之间取得了较好平衡，不需要额外的 3D 编码器网络（如 PointNet），仅用 MLP + 正弦编码即可。

### 6.2 Adaptive Action Grids 的深层意义

这个设计不仅仅是"动作离散化的一种改进"，它本质上是在说：**动作空间的拓扑结构应该被模型显式学习**。

- 传统 256 bins 是将每个维度独立对待，忽略了维度间的空间关系
- Adaptive Action Grids 将 $(\phi, \theta, r)$ 和 $(\text{roll}, \text{pitch}, \text{yaw})$ 各自作为 3D 空间处理，保留了维度间的结构
- token 嵌入在这个 3D 空间中是"有位置的"——相邻的网格在物理空间中也相邻，这使得模型可以学习到平滑的空间动作流形

### 6.3 Spatial Embedding Adaption 是一种新的 Transfer Learning 范式

论文提出的后训练方式（重新拟合分布→重新划分网格→三线性插值初始化）提供了一种在动作空间层面做迁移学习的新思路。这与传统的 LoRA/全参数微调互补——前者迁移的是模型内部的"认知能力"，后者迁移的是"动作空间的结构性知识"。消融实验也证实两者结合效果最好。

### 6.4 局限性思考

- 对 ZoeDepth 深度估计的依赖：如果深度估计不准确（如透明物体、反光表面），3D 位置编码的质量会下降
- 8194 个 action token 的词表大小不算小，可能在极端场景下存在 out-of-vocabulary 的问题
- 论文的 SimplerEnv 结果非常强，但需注意这是仿真环境，与真实世界 gap 仍然存在

---

## 参考

- **RT-2** (Brohan et al., 2023)：首个将 VLM 扩展为 VLA 的工作，开创了动作离散化为 token 的范式
- **OpenVLA** (Kim et al., 2024)：开源 VLA 基准线，采用均匀 256-bin 离散化
- **π₀** (Black et al., 2024)：用 flow matching 替代自回归的 VLA 方案，与 SpatialVLA 的自回归+高效 token 路线形成对比
- **Octo** (Team et al., 2024)：灵活的 Transformer 架构统一跨构型预训练
- **3D-VLA** (Zhen et al., 2024)：关注 3D 世界理解和预测，但忽略了动作空间的 3D 特征
- **PaliGemma 2** (Steiner et al., 2024)：SpatialVLA 的 VLM 骨架
- **ZoeDepth** (Bhat et al., 2023)：零样本深度估计模型，为 Ego3D 编码提供深度信息
