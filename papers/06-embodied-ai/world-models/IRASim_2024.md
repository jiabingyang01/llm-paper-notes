# IRASim：面向机器人操作的细粒度世界模型

> **论文**：*IRASim: A Fine-Grained World Model for Robot Manipulation*
>
> **作者**：Fangqi Zhu, Hongtao Wu (Project Lead), Song Guo (Corresponding), Yuxiao Liu, Chilam Cheang, Tao Kong
>
> **机构**：香港科技大学（HKUST）；ByteDance Seed
>
> **发布时间**：2024 年 06 月（arXiv 2406.14540）
>
> **发表状态**：未录用（预印本，v2 于 2025 年 7 月更新）
>
> 🔗 [arXiv](https://arxiv.org/abs/2406.14540) | [PDF](https://arxiv.org/pdf/2406.14540)
>
> **分类标签**：`world-model` `trajectory-to-video` `diffusion-transformer`

---

## 一句话总结

IRASim 用 Diffusion Transformer 学习 trajectory-to-video 世界模型：给定历史观测帧与机械臂动作轨迹，逐帧生成机器人执行该轨迹的高保真视频；核心创新是在每个 transformer block 内加入 **frame-level action-conditioning** 模块，把每个动作与对应帧显式对齐，从而精确模拟细粒度的机器人-物体交互。在 Push-T 上作为动力学模型做 model-based planning，可把普通 diffusion policy 的 IoU 从 0.637 提升到 0.961，且策略评估结果与真值 Mujoco 仿真器的 Pearson 相关系数达 0.99。

## 一、问题与动机

世界模型（world model）通过预测动作的视觉后果，为机器人提供两大能力：一是通过在模型中"想象"不同动作提案并择优执行来改进策略（model-based planning），二是作为真实世界评估的低成本替代来做可扩展的策略评估。

但把世界模型用于机器人操作有两个特殊难点：

1. **交互细腻**。操作任务对机器人-物体接触极其敏感,细微偏差就会导致任务失败，世界模型必须忠实捕捉这些精细交互。
2. **动作-帧的精确对齐**。现代操作策略普遍采用 action chunking，一次输出一整段动作轨迹而非单步动作。作者把"给定历史观测 + 完整动作轨迹，生成机器人执行该轨迹的视频"这一任务称为 **trajectory-to-video**。它与 text-to-video 有本质区别：文本只提供高层语境线索，而轨迹里每个动作都精确规定了对应视频帧中机器人的运动。已有的 text-to-video 类方法（把轨迹编码成单一 embedding 条件整段视频）忽视了这种逐帧对齐，无法准确建模。

形式化地，trajectory-to-video 任务定义为：

$$\mathbf{I}^{t+1:t+n+1} = f(\mathbf{I}^{t-h:t}, \mathbf{a}^{t:t+n})$$

其中 $h$ 为历史帧数，$n$ 为动作数，$\mathbf{a}^i \in \mathbb{R}^d$ 为第 $i$ 步动作。机械臂动作空间为 7 DoF（3 维平移 + 3 维旋转 + 1 维夹爪），并统一转成 relative delta action $\langle \Delta x, \Delta y, \Delta z, \Delta\alpha, \Delta\beta, \Delta\gamma, g \rangle$。

**用大白话说**：作者要造一个"给机械臂喂一条动作曲线，就吐出这条曲线执行过程视频"的模拟器。关键在于让视频的第 5 帧严格对应轨迹里的第 5 个动作，而不是像文本生成视频那样只保证整体氛围对。

## 二、核心方法

IRASim 是一个在 VAE 隐空间上运行的条件扩散模型，backbone 采用 Diffusion Transformer（DiT）。

### 2.1 隐空间扩散与条件构成

沿用 latent diffusion 思路，用 SDXL 预训练 VAE（训练全程冻结）把每帧 $\mathbf{I}^t$ 压缩为隐表示 $\mathbf{z}^t = \text{Enc}(\mathbf{I}^t)$，扩散在低维隐空间进行以省算力。条件 $\mathbf{c}$ 由两部分组成：历史帧隐表示 $\mathbf{z}^{t-h:t}$ 与动作轨迹 $\mathbf{a}^{t:t+n}$；扩散目标是后续 $n$ 帧的隐表示。训练用标准 DDPM 噪声预测目标：

$$\mathcal{L}_{\text{simple}}(\theta) = \|\epsilon_\theta(\mathbf{x}_t, t) - \epsilon_t\|^2$$

**历史帧条件**的注入方式很巧妙：把历史帧当作输入 token 序列中的"真值部分"，训练时只对待预测帧加噪、历史帧 token 保持干净，且扩散损失只在预测帧上计算。这样预测帧通过注意力机制与干净的历史帧交互，保证生成结果与历史观测一致。

为省算力，transformer block 里用 memory-efficient 的 spatial-temporal 分离注意力（交替的 Spatial Attention Block 与 Temporal Attention Block），避免对全部 token 做二次复杂度的 MHA。

### 2.2 两种轨迹条件注入：Video-Ada vs Frame-Ada（核心）

作者对比了两种通过 adaptive layer normalization（AdaLN）注入轨迹条件的方式：

**Video-Level Adaptation（Video-Ada，baseline 式）**：模仿 text-to-video，用一个线性层把整条轨迹编码成**单一** embedding，加到扩散 timestep embedding 上，回归出所有 spatial/temporal block 共享的 scale $\gamma,\alpha$ 与 shift $\beta$。

**Frame-Level Adaptation（Frame-Ada，本文核心）**：轨迹是比文本细得多的描述——每个动作规定了对应帧中机器人如何运动。因此对**每一帧单独用其对应动作**做条件。具体地，第 $i$ 个动作经线性层编码成独立 embedding，加上 timestep embedding 得到该帧的条件 $\mathbf{c}_S^i$，再回归出该帧在 spatial block 内专属的调制参数。spatial block 的计算变为逐帧：

$$\mathbf{x}^i = \mathbf{x}^i + (1 + \alpha_1^i)\times \text{MHA}(\gamma_1^i \times \text{LayerNorm}(\mathbf{x}^i + \beta_1^i))$$

$$\mathbf{x}^i = \mathbf{x}^i + (1 + \alpha_2^i)\times \text{FFN}(\gamma_2^i \times \text{LayerNorm}(\mathbf{x}^i + \beta_2^i))$$

其中 $\alpha_1^i,\gamma_1^i,\beta_1^i,\alpha_2^i,\gamma_2^i,\beta_2^i$ 均由该帧条件 $\mathbf{c}_S^i$ 回归得到。temporal block 仍沿用 video-level 的共享条件（因为它跨帧建模时序一致性）。

**用大白话说**：Video-Ada 是"给整段视频贴一张统一的动作标签"，而 Frame-Ada 是"给每一帧单独贴上它那一步该做的动作"。后者让"第 $i$ 帧"与"第 $i$ 个动作"在网络内部被强制绑定，动作-帧对齐因此更精确，机器人-物体交互也更真实。

### 2.3 长视频与推理

要生成完成整个任务的长视频，IRASim 以 **autoregressive** 方式滚动：把上一段生成的最后一帧作为下一段的历史条件帧，逐段拼接并保持时序一致。推理用 PNDM 50 步采样。IRASim-XL 生成 16 帧（约 4 秒）视频在单张 A100 上约需 30 秒、仅占 8GB 显存。

## 三、实验结果

在四个真实机器人数据集上验证：RT-1、Bridge、Language-Table、RoboNet。RT-1/Bridge/Language-Table 用 1 历史帧 + 15 动作预测后 15 帧；RoboNet 用 2 历史帧 + 10 动作预测后 10 帧。分辨率最高 288×512，长视频可 >150 帧。主要指标为 Latent L2（隐空间重建 L2）与 PSNR。

### 短轨迹视频生成（Table 1，节选主指标）

| 数据集 | 方法 | PSNR↑ | SSIM↑ | Latent L2↓ |
|---|---|---|---|---|
| RT-1 | VDM | 13.762 | 0.554 | 0.4983 |
| RT-1 | LVDM | 25.041 | 0.815 | 0.2244 |
| RT-1 | Video-Ada | 25.446 | 0.823 | 0.2191 |
| RT-1 | **Frame-Ada** | **26.048** | **0.833** | **0.2099** |
| Bridge | LVDM | 23.546 | 0.810 | 0.2155 |
| Bridge | **Frame-Ada** | **25.275** | **0.833** | **0.1947** |
| Language-Table | LVDM | 28.254 | **0.889** | 0.1704 |
| Language-Table | **Frame-Ada** | **28.818** | 0.888 | **0.1660** |

Frame-Ada 在三个数据集的 Latent L2 与 PSNR 主指标上全面领先，也优于 Video-Ada，验证了逐帧条件的价值。RoboNet 上（Table 2）IRASim PSNR 24.6 / SSIM 81.1，超过 iVideoGPT（23.8 / 80.8）与 MaskViT（20.4 / 67.1）；值得注意的是 iVideoGPT 在 OpenX-Embodiment 上做过大规模预训练，而 IRASim 仅在 RoboNet 上训练便更优。

### 长轨迹（Table 3，与最强 baseline LVDM 比）

| 数据集 | 方法 | Latent L2↓ | PSNR↑ |
|---|---|---|---|
| RT-1 | LVDM | 0.2567 | 23.573 |
| RT-1 | **Frame-Ada** | **0.2408** | **24.615** |
| Bridge | LVDM | 0.2534 | 21.792 |
| Bridge | **Frame-Ada** | **0.2306** | **23.260** |
| Language-Table | LVDM | 0.1776 | 26.215 |
| Language-Table | **Frame-Ada** | **0.1730** | **26.773** |

长轨迹平均长度 42.5 / 33.4 / 23.7 帧（RT-1/Bridge/Language-Table），autoregressive 滚动生成，Frame-Ada 在三数据集 Latent L2 上一致领先。

**Human preference**（5 名参与者，图 4）：Frame-Ada vs VDM 三数据集均 100% 胜；vs LVDM 胜率 60% / 72% / 68%；vs Video-Ada 胜率 38% / 36% / 66%（Language-Table 上明显更优）。

**Scaling**（图 5）：模型从 33M（S）到 679M（XL），随模型规模与训练步数增大，Latent L2 持续下降，扩展性良好。

### 策略评估（Table 4，LIBERO）

用 IRASim 替代真值 Mujoco 仿真器评估 4 个不同训练步的 diffusion policy，各跑 50 次成功率：

| 评估器 | 模型1 | 模型2 | 模型3 | 模型4 |
|---|---|---|---|---|
| 真值 Mujoco | 0.18 | 0.50 | 0.80 | 1.00 |
| **IRASim** | 0.28 | 0.48 | 0.74 | 0.96 |

两者 **Pearson 相关系数 0.99**，表明 IRASim 可作为可扩展的真实策略评估器。为此实验用 OpenSora 预训练权重初始化，并在 expert demo + post-trained rollouts（含成功与失败）上训练，以便同时模拟成功与失败（图 6 中甚至能模拟碗从夹爪滑落）。

### Model-based planning（Table 5，Push-T）

排序式 planning：从策略采样 $K$ 条轨迹，各自在 IRASim 中滚动，用 ResNet50 预测末帧 IoU 作为 value，选最高者执行。$P$ 为训练世界模型所用 post-trained rollout 数。

| 方法 | P | K=1 | K=5 | K=10 | K=50 |
|---|---|---|---|---|---|
| GPC-RANK | N/A | 0.642 | - | - | 0.698 |
| GPC-RANK+OPT | N/A | 0.642 | 0.824 | 0.882 | - |
| IRASim | 200 | 0.637 | 0.866 | 0.916 | 0.912 |
| IRASim | 1000 | 0.637 | 0.886 | **0.945** | **0.961** |

$K=1$ 即无 planning 的原始策略（IoU 0.637，与 GPC 论文 0.642 相当）。加 planning 后，$K=50,P=1000$ 时把 IoU 从 0.637 提升到 **0.961**，超过 GPC 两个变体，且性能随测试期计算量（$K$）与数据量（$P$）同步扩展，指向 robot manipulation 的 test-time scaling。真机 planning（Table 6，三任务）中 IRASim（MSE cost）成功率 0.87 / 0.80 / 0.87，远超随机基线 0.20 / 0.07 / 0.13，MSE cost 也普遍优于 ResNet50 特征 cosine cost。

### 灵活可控性

可用键盘（Language-Table 2D 平移）或 VR 手柄（RT-1 3D）实时"操控"数据集里的虚拟机器人；即便轨迹分布偏离原数据集、或物理不可行（如命令机器人穿透桌面），IRASim 也能鲁棒生成物理合理的视频（机器人停在桌面上），并能处理同一初始帧不同轨迹的多模态生成。

## 四、局限性

- **非实时**。生成 16 帧约需 30 秒，无法实时；作者计划用 diffusion distillation 加速。
- **planning 受 cost function 制约**。MSE cost 并非总最优，有时选错预测视频导致任务失败；策略评估/planning 的表现同时受视频预测精度与 cost 函数精度影响。
- **仅排序式规划**。当前 model-based planning 只是简单采样-排序，采样策略较朴素（如从球面采点连轨迹），未与更强的 action chunking 策略或梯度优化结合。
- **依赖预训练初始化**。策略评估与 planning 实验依赖 OpenSora 预训练权重来加速收敛，纯从头训练在这些下游任务上的表现未充分展示。
- **评估指标取舍**。作者发现 FID/FVD 与人类偏好相关性差（因 trajectory-to-video 本质是重建任务而非分布匹配），故主指标退回 Latent L2/PSNR，长视频甚至不算 FID/FVD——反映当前缺乏公认的世界模型保真度度量。

## 五、评价与展望

**优点**：

1. **问题定义清晰且切中要害**。把 action-chunking 时代的世界模型需求提炼为 trajectory-to-video，并指出其与 text-to-video 的核心差异在逐帧动作对齐，动机扎实。
2. **Frame-Ada 是简洁而有效的归纳偏置**。仅在 AdaLN 里把"整段共享条件"换成"逐帧专属条件"，改动很小却带来一致提升，且与 DiT 的 spatial-temporal 结构天然契合（spatial 块逐帧、temporal 块共享），设计上很自洽。
3. **闭环验证充分**。不止刷视频质量指标，还把世界模型真正用于策略评估（与 Mujoco 相关 0.99）和 model-based planning（Push-T IoU 0.637→0.961，含真机），这类"世界模型是否真有用"的下游证据比单纯生成质量更有说服力。
4. **可控性 demo（键盘/VR）** 直观展示了细粒度动作可控性与对 OOD/物理不可行轨迹的鲁棒性。

**与公开工作的关系**：IRASim 属于 action-conditioned video prediction 谱系，与用文本条件 text-to-video 的 UniSim、VLP 不同，它以精确轨迹逐帧条件；与 iVideoGPT（autoregressive token 预测）、MaskViT 相比走 DiT + latent diffusion 路线并取得更好重建。planning 部分对标同期 GPC（generative predictive control，autoregressive 逐帧生成 + 排序/梯度优化），IRASim 一次并行生成整段轨迹视频、排序即超过 GPC 变体。可控视频生成方向上，相比 stroke/bbox/pose 条件的方法，它建模的是随时间演化的完整 2D/3D 机器人动作。

**开放问题与可能改进**：

1. **实时性**是落地 planning 的最大瓶颈，diffusion distillation、consistency model 或更少步采样值得跟进。
2. **闭环 RL/MPC**。作者已提出把 IRASim 当动力学模型、在其中用 RL 改进策略；当前仅做了开环排序，闭环滚动会放大 autoregressive 累积误差，长时程一致性与误差纠正是关键。
3. **cost/value 函数**。planning 成功率强依赖 value model，如何学习更可靠、可与生成联合训练的 value（乃至可微 planning）是自然延伸。
4. **物理一致性度量**。论文承认 FID/FVD 不合适，如何量化世界模型的"物理正确性"（接触、遮挡、articulation）仍是开放难题。
5. **跨本体泛化**。四数据集各自训练，是否能训练统一跨本体世界模型、并零样本迁移到新机械臂/新动作空间，值得探索。

## 参考

1. Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)*, ICCV 2023 — backbone 架构。
2. Yang et al., *Learning Interactive Real-World Simulators (UniSim)*, ICLR 2024 — 文本/动作条件的交互式世界模拟器对照。
3. Wu et al., *iVideoGPT: Interactive VideoGPTs are Scalable World Models*, 2024 — autoregressive 世界模型基线（RoboNet 对比）。
4. Qi et al., *Strengthening Generative Robot Policies through Predictive World Modeling (GPC)*, 2025 — model-based planning 主要对标方法。
5. Zheng et al., *Open-Sora*, 2024 — 提供预训练权重，用于策略评估/planning 初始化。
