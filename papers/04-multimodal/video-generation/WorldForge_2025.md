# WorldForge：用训练无关的推理时引导解锁视频扩散模型中的 3D/4D 生成能力

> **论文**：*WorldForge: Unlocking Emergent 3D/4D Generation in Video Diffusion Model via Training-Free Guidance*
>
> **作者**：Chenxi Song, Yanming Yang, Tong Zhao, Ruibo Li, Chi Zhang
>
> **机构**：西湖大学 AGI Lab、南洋理工大学
>
> **发布时间**：2025年9月（CVPR 2026）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.15130) | [项目主页](https://worldforge-agi.github.io)
>
> **分类标签**：`Video Diffusion` `3D/4D Generation` `Training-Free` `Trajectory Control` `Inference-Time Guidance`

---

## 一句话总结

WorldForge 提出了一个完全 training-free 的推理时引导框架，通过三个协同模块（IRR、FLF、DSG）将精确的轨迹控制注入预训练视频扩散模型，实现了从单张图像生成 3D 场景以及从视频进行 4D 轨迹控制重渲染，在轨迹精度、几何一致性和感知质量上均达到 SOTA。

---

## 一、问题与动机

### 1.1 视频扩散模型的空间智能潜力与瓶颈

近年来，视频扩散模型（VDMs）如 Wan 2.1、SVD、CogVideoX 等在大规模视频数据上训练，编码了丰富的时空先验，展现出用于空间智能任务（3D/4D 理解、重建、生成）的巨大潜力。然而，VDMs 在应用于 3D/4D 任务时面临三大根本性限制：

1. **可控性不足**：难以遵循精确的运动约束（如 6-DoF 相机轨迹），导致新视角合成和轨迹控制中空间一致性差
2. **时空一致性差**：生成的帧之间缺乏几何连贯性
3. **场景-相机动态耦合**：改变视角时会引起非预期的物体变形和场景不稳定

### 1.2 现有方法的不足

为了解决这些问题，前人工作沿两条路线展开：

**路线一：微调（Fine-tuning）**——在带运动标注的数据上微调 VDM（如 LoRA、ControlNet 适配器），问题在于：
- 计算成本高昂
- 泛化能力差
- 存在破坏预训练先验的风险

**路线二：Warp-and-Repaint**——沿新相机路径重投影帧，再用生成模型填充遮挡区域，问题在于：
- 预训练模型难以处理扭曲的 OOD（Out-of-Distribution）输入，容易产生伪影和碎片化几何
- 动态训练数据的偏置会导致静态场景中出现幻觉运动

**核心矛盾**：如何在保留 VDM 宝贵先验的同时，注入精确的轨迹控制？

### 1.3 WorldForge 的核心思路

WorldForge 选择了一条优雅的中间路线：**不改变模型权重，而是在推理时对去噪过程进行精细引导**。具体来说，它沿用 warp-and-repaint 的流程作为基础，但在 repaint（去噪生成）阶段引入三个互补的推理时引导机制，分别解决轨迹注入、运动-外观解耦、以及质量校正三个问题。

---

## 二、预备知识

### 2.1 扩散模型与去噪采样

在 DDIM 采样框架下，给定噪声预测网络 $\epsilon_\theta(\mathbf{x}_t, t)$，从当前状态 $\mathbf{x}_t$ 估计干净样本：

$$\hat{\mathbf{x}}_0(\mathbf{x}_t, t) = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

然后混合 $\hat{\mathbf{x}}_0$ 和噪声得到下一步 $\mathbf{x}_{t-1}$：

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{\mathbf{x}}_0(\mathbf{x}_t, t) + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta(\mathbf{x}_t, t)$$

WorldForge 的核心操作点正是 $\hat{\mathbf{x}}_0^{(t)}$——每一步去噪得到的中间估计。通过修改这个中间变量，可以在不改变模型本身的情况下施加轨迹控制。

论文同时证明了 Flow Matching（如 Wan 2.1 使用的框架）是扩散模型在线性噪声调度 $\alpha_t = 1 - t, \sigma_t = t$ 下的特例，因此框架对两种范式通用。

### 2.2 Classifier-Free Guidance（CFG）

CFG 通过调整条件和无条件预测的插值来提升生成的条件保真度：

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t) = \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) + \omega_{\text{CFG}} \cdot \left[\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \phi)\right]$$

WorldForge 的 DSG 模块正是受 CFG 启发，但针对引导路径和非引导路径之间更大的角度差异进行了关键改进。

### 2.3 基于深度的 Warp 轨迹控制

整个流程的外层结构如下：

1. 深度预测网络从输入图像/视频估计相机位姿 $\mathbf{P}_q$ 和深度图 $\mathbf{D}_q$
2. Warp 算子 $\mathcal{W}$ 将源帧 $\mathbf{I}_{src}$ 从 $\mathbf{P}_{src}$ 投影到目标位姿 $\mathbf{P}_{tar}$，得到部分目标视图和有效像素掩码：

$$(\mathbf{I}'_{tar}, \mathbf{M}_{tar}) = \mathcal{W}(\mathbf{I}_{src}, \mathbf{D}_{src}, \mathbf{P}_{src}, \mathbf{P}_{tar})$$

warp 得到的帧通过 VAE 编码为 latent $\mathbf{Z}_{traj}$，掩码 $\mathbf{M}$ 指示哪些区域有有效观测。这些作为引导信号输入三个核心模块。

---

## 三、核心方法

WorldForge 的三个模块形成一个从粗到细的协同系统：

### 3.1 Intra-Step Recursive Refinement（IRR）——轨迹注入

**解决的问题**：如何在每个去噪步骤中持续注入轨迹控制信号？

**核心思想**：在每个去噪步骤内部嵌入一个微型 "预测-校正" 循环。

具体流程：给定一步去噪的中间估计 $\hat{\mathbf{x}}_0^{(t)}$，IRR 将其与轨迹 latent $\mathbf{Z}_{traj}$ 融合，然后加入高斯噪声重新进入去噪调度：

$$\mathbf{x}'_t = (1 - w(\sigma))\, \mathbf{F}(\hat{\mathbf{x}}_0^{(t)}, \mathbf{Z}_{traj}) + w(\sigma) \cdot \epsilon$$

其中融合操作 $\mathbf{F}$ 的定义为：

$$\mathbf{F}(\hat{\mathbf{x}}_0^{(t)}, \mathbf{Z}_{traj}) = \mathbf{M} \cdot \mathbf{Z}_{traj} + (1 - \mathbf{M}) \cdot \hat{\mathbf{x}}_0^{(t)}$$

用大白话说：在掩码标记的可见区域，用 warp 得到的"正确答案"替换模型自己的预测；在不可见区域（遮挡、新暴露的区域），保留模型的生成。然后加噪后重新送入网络去噪，让轨迹信号在每一步都得到注入。

$w(\sigma)$ 是一个噪声调度器，控制重新加噪的强度，$\epsilon = \mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$。修正后的 $\mathbf{x}'_t$ 替代原始 $\mathbf{x}_t$ 用于下一步采样。

### 3.2 Flow-Gated Latent Fusion（FLF）——运动与外观解耦

**解决的问题**：IRR 中的全通道融合（即公式中的 $\mathbf{F}$）会降低视觉质量，因为 VAE latent 的不同通道编码了不同的信息——有些通道专门负责外观，有些专门负责运动。

**核心洞察**：不应该一视同仁地覆盖所有通道，而应该只修改与运动高度相关的通道，保留外观通道不变。

**如何识别运动通道？** 用光流相似度打分：

> 1. 对 latent $\hat{\mathbf{x}}_0^{(t)}$ 的每个通道 $c$，计算相邻帧之间的光流 $\mathcal{F}_{pred}^{(t,c)}$（使用 Farneback 算法）
> 2. 对轨迹 latent $\mathbf{Z}_{traj}$ 同样计算 GT 光流 $\mathcal{F}_{gt}^{(t,c)}$
> 3. 在掩码有效区域内，用三个光流指标衡量两者的相似度

三个指标分别是：
- **M-EPE**（Masked End-point Error）：预测与 GT 流向量的欧氏距离
- **M-AE**（Masked Angular Error）：角度差异
- **Fl-all**（Outlier Percentage）：不可靠像素的比例

归一化后合成运动相似度得分：

$$S^{(t,c)} = \sum_{k \in \{E, A, F\}} \gamma_k \left(1 - \text{Norm}_k^{(t,c)}\right)$$

其中 $(\gamma_E, \gamma_A, \gamma_F) = (0.4, 0.3, 0.3)$。得分越高说明该通道的光流越符合目标轨迹，即与运动更相关。

**动态阈值选择**：使用 $\delta^{(t)} = \mu_S^{(t)} - \lambda^{(t)} \sigma_S^{(t)}$ 来确定哪些通道被选中。关键设计是 $\lambda^{(t)}$ 随时间变化——**早期步骤选择更多通道**（定义大结构），**后期选择更少通道**（保护细节），这与扩散模型从粗到细的生成过程一致。

最终的选择性融合：

$$\mathbf{FLF}(\hat{\mathbf{x}}_0^{(t)}, \mathbf{Z}_{traj}) = \begin{cases} \mathbf{M}^{(c)} \cdot \mathbf{Z}_{traj}^{(c)} + (1 - \mathbf{M}^{(c)}) \cdot \hat{\mathbf{x}}_0^{(t,c)}, & \text{if } S^{(t,c)} \geq \delta^{(t)} \\ \hat{\mathbf{x}}_0^{(t,c)}, & \text{otherwise} \end{cases}$$

用大白话说：只在那些"主要负责运动"的通道上注入轨迹信息，那些"主要负责外观"的通道保持原样，从而在控制轨迹的同时保护视觉细节。

### 3.3 Dual-Path Self-Corrective Guidance（DSG）——质量校正

**解决的问题**：warp 得到的轨迹 latent $\mathbf{Z}_{traj}$ 不可避免地包含深度误差、遮挡和错位带来的噪声和伪影，直接注入会降低生成质量。

**核心思想**：受 CFG 启发，维护两条并行的去噪路径，利用它们的差异来自适应校正。

IRR 在每一步产生两个速度场：
- **非引导速度** $\mathbf{v}_t^{ori}$：从原始 latent $\mathbf{x}_t$ 得到，保持高保真但忽略轨迹（"坏的方向"——不遵守控制）
- **引导速度** $\mathbf{v}_t^{traj}$：从校正后的 $\mathbf{x}'_t$ 得到，遵循轨迹但可能有噪声（"好的方向"——遵守控制但质量有瑕疵）

**为什么不能直接用 CFG 公式？**

论文实验发现了一个关键差异：标准 CFG 中条件和无条件预测的余弦相似度接近 1（角度接近 0°），而 WorldForge 的两条路径的余弦相似度仅在 0.3--0.6 之间（角度 50°--70°）。如此大的角度差异意味着直接套用 CFG 公式会产生严重伪影。

**DSG 的解决方案**：只使用"好方向"中与"坏方向"**正交**的分量来做校正，避免大角度差异带来的不良效果：

$$\mathbf{v}_t^{corr} = \mathbf{v}_t^{traj} + \rho \cdot \beta_t \left(\mathbf{v}_t^{traj} - \alpha_t \cdot \mathbf{v}_t^{ori}\right)$$

其中：
- $\alpha_t = \frac{\mathbf{v}_t^{traj} \cdot \mathbf{v}_t^{ori}}{\|\mathbf{v}_t^{traj}\| \cdot \|\mathbf{v}_t^{ori}\|}$ 是余弦相似度
- $\beta_t = \sqrt{1 - \alpha_t^2}$ 是对应的正弦值
- $\rho$ 控制引导强度

$\beta_t$ 的自适应缩放效果：
- 当两条路径**分歧大**时（$\alpha_t$ 低，$\beta_t$ 高），施加更强的校正——这说明 warp 引入了更多噪声
- 当两条路径**一致**时（$\alpha_t$ 高，$\beta_t$ 低），减弱校正——模型自身的预测已经足够好

---

## 四、实验结果

### 4.1 实验设置

- **主要骨干**：Wan 2.1 I2V-14B，分辨率最高 1280x720
- **验证骨干**：SVD（U-Net 架构，RTX 4090 上 25 帧推理）
- **深度估计**：支持 VGGT、UniDepth、Mega-SaM、DepthCrafter 等多种模型
- **数据集**：LLFF、Tanks and Temples、MipNeRF 360、互联网/真实/AI 生成图像
- **指标**：
  - 生成质量：FID、CLIPsim（静态）；FVD、CLIP-Vsim（动态）
  - 轨迹精度：ATE、RPE-T、RPE-R

### 4.2 主要定量结果

| 方法 | FID ↓ | CLIPsim ↑ | FVD ↓ | CLIP-Vsim ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|---|---|---|---|---|---|---|---|
| See3D | 123.26 | 0.941 | – | – | 0.091 | 0.089 | 0.250 |
| ViewCrafter | 117.50 | 0.930 | – | – | 0.236 | 0.315 | 0.728 |
| ViewExtrapolator | 125.50 | 0.930 | 108.48 | 0.913 | 0.183 | 0.260 | 0.882 |
| TrajectoryAttention | 122.37 | 0.920 | 106.94 | 0.911 | 0.159 | 0.238 | 0.532 |
| TrajectoryCrafter | 111.49 | 0.910 | 97.31 | 0.923 | 0.090 | 0.152 | 0.267 |
| NVS-Solver | 118.64 | 0.937 | – | – | 0.224 | 0.268 | 1.056 |
| **WorldForge (Ours)** | **96.08** | **0.948** | **93.17** | **0.938** | **0.077** | **0.086** | **0.221** |

关键发现：
- WorldForge 在**所有指标**上取得最优或次优结果
- FID 从 TrajectoryCrafter 的 111.49 大幅提升至 96.08（降低 13.8%），说明生成图像质量显著提高
- ATE 0.077 为最低，轨迹跟踪精度最高
- 作为 training-free 方法，甚至超越了 See3D、ViewCrafter 等需要训练/微调的方法

### 4.3 消融实验

| 配置 | 效果 |
|---|---|
| 移除 IRR | 完全失去轨迹控制，退化为无引导的自由生成 |
| 移除 FLF | 运动和外观耦合，产生不自然的输出 |
| 移除 DSG | warp 噪声传入生成过程，出现伪影，视觉质量下降 |
| 完整模型 | 最佳保真度和控制精度 |

三个模块缺一不可，且协同工作效果最佳。

### 4.4 效率分析

| 方法 | 帧数 | 分辨率 | 推理时间(min) | Training-Free |
|---|---|---|---|---|
| See3D | 25 | 576x1024 | 1.7 | 否 |
| TrajectoryCrafter | 25 | 384x672 | 1.7 | 否 |
| NVS-Solver | 25 | 576x1024 | 9.3 | 是 |
| ReCamMaster | 81 | 480x832 | 14.6 | 否 |
| **WorldForge (720P)** | 25 | 720x1280 | 17.3 | **是** |
| **WorldForge (480P)** | 25 | 480x832 | 6.8 | **是** |
| **WorldForge (SVD)** | 25 | 576x1024 | 1.3 | **是** |

推理时间相比基础 VDM 增加约 40--50%，主要来自 IRR 模块。在 SVD 骨干上甚至比多数基线更快。

### 4.5 模型无关性验证

- **VDM 骨干消融**：将框架迁移到 U-Net 架构的 SVD 上，仍在 SVD 系方法中达到 SOTA，证明引导策略与模型架构无关
- **深度模型消融**：在 VGGT、UniDepth、Mega-SaM、DepthCrafter 四种深度估计器上均保持高性能，VDM 的世界先验能有效补偿 warp 伪影

---

## 五、局限性与未来方向

1. **深度估计失败场景**：当深度估计严重不准确时（如前景-背景完全纠缠、主体被压平），框架无法有效校正
2. **细粒度控制不足**：全局引导对小物体和精细细节的控制有限
3. **推理开销**：IRR 模块引入约 40--50% 的额外推理时间
4. **未来方向**：集成细粒度控制机制，应用于更强大的生成模型

---

## 六、个人思考

### 与世界模型的联系

WorldForge 本质上是在利用 VDM 作为隐式的世界模型——模型在大规模视频数据上学到的时空先验使其能够"想象"未见过的视角。这与 Embodied AI 中的世界模型思路一脉相承，但 WorldForge 更强调的是**如何从预训练模型中"提取"这种能力而不是重新训练**。

### 三个模块的设计哲学

三个模块的设计体现了对扩散模型生成过程的深入理解：
- **IRR** 利用了扩散模型"逐步去噪"的特性，在每步注入校正信号
- **FLF** 利用了 VAE latent 空间中运动和外观的自然解耦
- **DSG** 则是对 CFG 的巧妙推广，解决了高角度差异下引导失效的问题

### DSG 的关键贡献

DSG 中发现的"引导路径与非引导路径余弦相似度仅 0.3--0.6"这一经验观察非常重要。它说明在 warp-and-repaint 场景下，引导信号远比标准 CFG 中的条件/无条件差异要大，需要专门的数学处理。使用正交分量投影来解决这个问题是一个优雅的几何直觉。

### Training-Free 范式的优势

WorldForge 不需要任何训练或微调，直接 plug-and-play 到不同的 VDM 骨干上。这意味着随着基础视频模型的进步（如更强的 Wan 3.0 或更大规模的模型），WorldForge 可以立即受益，而微调方法则需要重新适配。

---

## 参考

- **Wan 2.1**（Wan et al., 2025）：WorldForge 的主要骨干模型，开源的大规模视频生成模型
- **SVD**（Blattmann et al., 2023）：Stable Video Diffusion，U-Net 架构的视频扩散模型，用于验证模型无关性
- **CFG**（Ho & Salimans, 2021）：Classifier-Free Guidance，DSG 模块的灵感来源
- **TrajectoryCrafter**（Yu et al., 2025）：主要对比基线之一，基于 CogVideoX 的轨迹控制方法
- **NVS-Solver**（You et al., 2025）：另一个 training-free 基线，基于 SVD 的零样本新视角合成
- **ViewCrafter**（Yu et al., 2024）：基于微调的高保真新视角合成方法
- **VGGT**（Wang et al., 2025）：Visual Geometry Grounded Transformer，用于深度和位姿估计
