# VLA-RFT：世界模型驱动的 Verified Reward 强化微调——原理详解

> 论文：*VLA-RFT: Vision-Language-Action Reinforcement Fine-Tuning with Verified Rewards in World Simulators*
> 机构：Westlake University、Zhejiang University、OpenHelix Team、Fudan University 等
> 作者：Hengtao Li、Pengxiang Ding、Runze Suo、Yihao Wang、Zirui Ge 等
> 发布时间：2025年10月
> [arXiv](https://arxiv.org/abs/2510.00406)

---

## 一句话总结

用数据驱动的视频预测世界模型充当可控模拟器，将 VLA 输出的动作在世界模型中 rollout 得到视觉轨迹，通过与目标参考轨迹对比计算 verified reward（像素级 MAE + 感知级 LPIPS），再用 GRPO 端到端更新 VLA——仅需 400 步微调即超越 150K 步 SFT 基线。

---

## 一、问题与动机

### 1.1 模仿学习的瓶颈

当前 VLA 模型主要依赖模仿学习（行为克隆），天然存在两个问题：

- **误差累积（compounding error）**：一旦策略偏离专家轨迹，小偏差逐步放大，驱使策略进入从未见过的状态
- **分布偏移下鲁棒性差**：对物体位置、目标位置、机器人初始状态的微小扰动，策略性能急剧下降

### 1.2 RL 后训练的三条路线及各自困境

| 路线 | 代表方法 | 核心瓶颈 |
|:---|:---|:---|
| 在线 RL（真实世界） | VLA-RL、RLVLA | 数据昂贵、安全风险、交互速度慢 |
| 在线 RL（模拟器） | TGRPO、ConRFT | 需百万级交互、sim-to-real gap |
| 离线 RL | ARFM、RWR | 无法从自身动作后果中学习、受限于数据分布 |

### 1.3 VLA-RFT 的思路

**核心洞察**：用一个**数据驱动的世界模型**（learned world model）充当"可控模拟器"——

1. 世界模型在离线数据上预训练，学习 $(o_t, a_t) \to o_{t+1}$ 的动力学
2. VLA 提出动作 → 世界模型生成视觉轨迹 → 与参考轨迹对比得到 verified reward
3. 用 GRPO 做策略优化

这条路线兼具离线训练的安全性和在线交互的学习信号丰富性。

### 1.4 与 WMPO/RISE/WoVR 的关键区别

| 维度 | VLA-RFT | WMPO | RISE | WoVR |
|:---|:---|:---|:---|:---|
| 世界模型空间 | **像素空间**（视频预测） | 隐空间 | 隐空间 | 像素空间 |
| 奖励来源 | Verified reward（MAE+LPIPS） | 隐空间 reward model | 想象轨迹 reward | VLM 判别 |
| 策略优化 | GRPO | PPO | PPO | PPO |
| 策略参数化 | SDE-Policy（Flow+Sigma） | Flow Matching | Diffusion | Diffusion |
| 训练步数 | **400** | ~数千 | ~数千 | ~数千 |

VLA-RFT 的最大特色是在**像素空间**做世界模型预测，配合 verified reward 实现极高的样本效率。

---

## 二、预备知识

### 2.1 Flow Matching 动作头

VLA 策略被分解为两部分：

$$\hat{a}_{i:i+T-1} \sim \pi_\theta(\cdot | o_i, l_i, s_i) = \pi_{\theta_\text{fm}}(\cdot | z_i, s_i), \quad z_i = f_\text{VLM}(o_i, l_i)$$

- $f_\text{VLM}$：视觉-语言编码器，将图像和指令编码为隐表征 $z_i$
- $\pi_{\theta_\text{fm}}$：flow-matching 动作头，生成 $T$ 步动作 chunk

### 2.2 GRPO（Group Relative Policy Optimization）

GRPO 的核心思想是对同一起始状态采样 $N$ 条 rollout，用组内平均奖励作 baseline 计算优势：

$$\bar{R}_\text{group} = \frac{1}{N}\sum_{j=1}^{N} R_j, \quad \text{Adv}_n = R_n - \bar{R}_\text{group}$$

然后用 clipped policy ratio 做优化，避免 critic 网络的引入。

### 2.3 LPIPS 感知距离

LPIPS（Learned Perceptual Image Patch Similarity）使用预训练深度网络的特征衡量图像间的感知相似度，比像素级 L1/L2 更符合人类对"相似"的判断。$d_\text{LPIPS}$ 越小表示越相似。

---

## 三、核心方法：VLA-RFT

### 3.1 两阶段训练框架

**Stage I: 预训练**——世界模型 + VLA 策略分别在离线数据上初始化

**Stage II: 强化微调**——VLA 与世界模型交互，通过 verified reward 优化策略

### 3.2 Stage I: 世界模型预训练

世界模型是一个基于 LLaMA 架构的轻量级自回归 Transformer（138M 参数，GPT-2 small 规模），由 **VQGAN 编码器**（将图像转为离散 token）和 **动作 tokenizer**（连续动作离散化）组成。

训练目标为最大似然：

$$\mathcal{L}_\text{MLE}^\text{WM}(\phi) = -\mathbb{E}\Big[\log p_\phi(o_{i+1}|o_i, a_i) + \sum_{t=1}^{T-1}\log p_\phi(o_{i+t+1}|o_{i:i+t}, a_{i:i+t})\Big]$$

其中 $p_\phi$ 是世界模型的预测分布。给定初始帧和动作序列，世界模型自回归地生成未来视觉帧。

### 3.3 Stage I: VLA 预训练

用标准 flow matching MSE loss 在专家数据上初始化 VLA：

$$\mathcal{L}_\text{MSE}^\text{VLA}(\theta) = \mathbb{E}_{(a,o,l,s)\sim\mathcal{D}}\Big[\|v_\theta(o,l,s,a^\tau) - u_\tau\|_2^2\Big]$$

其中 $\tau \sim \text{Beta}(\alpha,\beta)$ 是 flow matching 时间步，$a^\tau = \tau a + (1-\tau)\epsilon$ 是噪声扰动的动作，$u_\tau = a - \epsilon$ 是目标流场。

### 3.4 SDE-Policy：从确定性 ODE 到随机 SDE

**问题**：Flow matching 本质上是确定性 ODE 过程，难以直接计算 log-likelihood（RL 需要）。

**解决方案**：引入 **Sigma Net**（与 flow-matching head 同构的网络），为每个 denoising step 输出方差向量 $\sigma^k_\psi$，将确定性 ODE 扩展为随机 SDE：

$$\mu_k = a^{k\delta}_{i:i+T-1} + \delta \cdot v_\theta(o_i, l_i, s_i, a^{k\delta}_{i:i+T-1})$$

$$a^{k\delta}_{i:i+T-1} \sim \mathcal{N}(\mu_k, \Sigma_k), \quad \Sigma_k = (\sigma^k_\psi)^2$$

这里 $\delta = 0.1$，$K=10$ 步积分。**均值由 flow head 提供，方差由 Sigma Net 提供**，两者共同定义高斯条件分布。

**对数似然计算**：在一次 rollout 中，对 $K$ 步 denoising 的 step-wise log-likelihood 取平均：

$$\bar{\ell}_{\theta,\psi} = \frac{1}{K}\sum_{k=1}^{K}\log p^{(k)}_{\theta,\psi}(a^{k\delta} | a^{(k-1)\delta}, z_i, s_i)$$

**策略比率**：

$$r = \exp(\bar{\ell}_{\theta,\psi} - \bar{\ell}_\text{old})$$

用大白话说：Sigma Net 赋予了 flow policy 随机性（exploration 能力），同时使 log-likelihood 有了显式解析形式，让 GRPO 可以直接计算策略比率。

### 3.5 Verified Reward：世界模型内的轨迹对比

给定 VLA 输出的动作 chunk $a^K_{i:i+T-1}$，世界模型自回归生成视觉轨迹：

$$\text{Traj} = [o_i, a^{K\delta}_i, \hat{o}_{i+1}, \ldots, a^{K\delta}_{i+T-1}, \hat{o}_{i+T}]$$

**关键设计**：奖励不是将生成帧与真实图像直接对比，而是让**同一个世界模型**分别用策略动作和专家动作生成两条轨迹，在同一生成空间内对比——消除了世界模型生成质量偏差：

$$R = -\sum_{t=0}^{T-1}\Big[\lambda_1 \cdot L_1(\hat{o}_{i+t+1}, o_{i+t+1}) + \lambda_\text{lp} \cdot \text{LPIPS}(\hat{o}_{i+t+1}, o_{i+t+1})\Big]$$

- $L_1$：像素级绝对误差，衡量低层匹配度
- LPIPS：感知距离，衡量语义级匹配度
- $\lambda_1, \lambda_\text{lp}$：权重系数

### 3.6 GRPO 优化目标

$$\mathcal{L}_\text{GRPO}^\text{VLA}(\theta,\psi) = -\mathbb{E}\Big[\text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot \text{Adv}\Big] + \lambda_\text{mse}\mathcal{L}_\text{MSE}^\text{VLA}(\theta) - \alpha\mathbb{H}(\pi_{\theta,\psi})$$

三项组成：

| 项 | 作用 |
|:---|:---|
| Clipped surrogate | 策略梯度优化，限制更新幅度 |
| $\lambda_\text{mse}\mathcal{L}_\text{MSE}^\text{VLA}$ | 辅助 flow matching loss，防止 action head 遗忘 |
| $-\alpha\mathbb{H}(\pi)$ | 熵正则，鼓励探索 |

### 3.7 三种 Reward 设计对比

论文对比了三种 verified reward：

| 类型 | 描述 | 效果 |
|:---|:---|:---|
| Reward Type 1 | 策略动作 vs 数据动作的 L1 距离（无世界模型） | +1.1 |
| Reward Type 2 | 世界模型生成帧 vs 真实图像的 MAE+LPIPS | +0.5 |
| **Reward Type 3**（本文） | 同一世界模型分别渲染策略/数据动作轨迹后对比 | **+4.5** |

Type 3 在同一生成空间做对比，消除了世界模型生成质量偏差，效果远优于直接用真实图像对比。

---

## 四、实验设置

### 4.1 基准与指标

- **基准**：LIBERO benchmark（4 个子 suite：Spatial、Object、Goal、Long）
- **指标**：成功率（SR, %）
- **扰动测试**：物体位置、目标位置、机器人状态、组合扰动

### 4.2 基线策略

- **VLA-Adapter**（Wang et al., 2025）：轻量级 VLA，上层 VLM（Qwen2.5-0.5B + DINOSigLIP）+ 下层 DiT flow matching action head
- VLM 用 LoRA（rank 64）高效微调

### 4.3 世界模型配置

- 架构：LLaMA 风格，12 层 Transformer，hidden 768，FFN 3072，138M 参数
- 输入：VQGAN 编码的图像 token + 离散化的动作 token
- 预训练：LIBERO 数据集，150K 步

### 4.4 训练细节

| 超参数 | 值 |
|:---|:---|
| 优势估计 | GRPO |
| 学习率 | $1 \times 10^{-6}$ |
| Sigma Net 学习率 | $1 \times 10^{-5}$ |
| MSE loss 系数 | 0.01 |
| 熵系数 | 0.003 |
| 训练步数 | **400** |
| Batch size | 16 |
| Rollout 次数 | 16 |
| 框架 | VERL（4x A800 GPU） |

---

## 五、实验结果

### 5.1 世界模型质量

| 指标 | Spatial | Object | Goal | Long | 平均 |
|:---|:---|:---|:---|:---|:---|
| MSE ↓ | 0.0039 | 0.0036 | 0.0024 | 0.0056 | 0.0039 |
| PSNR ↑ | 24.98 | 25.13 | 26.99 | 23.83 | 25.23 |
| SSIM ↑ | 0.896 | 0.913 | 0.929 | 0.885 | 0.906 |
| LPIPS ↓ | 0.067 | 0.054 | 0.040 | 0.074 | 0.059 |

世界模型在像素保真度和感知质量上均表现优异，验证了其作为"可控模拟器"的可行性。

### 5.2 标准设置性能

| 策略 | Spatial | Object | Goal | Long | 平均 |
|:---|:---|:---|:---|:---|:---|
| Base (3w) | 82.4 | 84.8 | 85.4 | 57.2 | 77.5 |
| Base (15w) | 88.4 | 88.0 | 92.8 | 77.2 | 86.6 |
| **VLA-RFT (400)** | **94.4** | **94.4** | **95.4** | **80.2** | **91.1** |
| $\Delta$ vs Base (15w) | +6.0 | +6.4 | +2.6 | +3.0 | **+4.5** |

**仅 400 步 RFT 即超越 150K 步 SFT 基线 +4.5 个百分点**，样本效率提升约 375 倍。

### 5.3 扰动鲁棒性

| 扰动类型 | Base (15w) | VLA-RFT | $\Delta$ |
|:---|:---|:---|:---|
| Object Position (±2.5cm) | 69.3 | 73.5 | +4.2 |
| Object Position (±5cm) | 48.0 | 52.5 | +4.5 |
| Goal Position (±2.5cm) | 74.5 | 79.0 | +4.5 |
| Goal Position (±5cm) | 44.8 | 51.5 | +6.7 |
| RoboState (±20) | 73.0 | 76.5 | +2.5 |
| RoboState (±50) | 63.5 | 67.0 | +3.5 |
| Combined (±2.5/2.5/20) | 63.5 | 70.0 | +6.5 |
| Combined (±5/5/50) | 34.0 | 37.0 | +3.0 |

VLA-RFT 在所有扰动条件下均优于基线，**组合扰动下提升最显著（+6.5%）**。动作分布可视化显示 RFT 策略的动作覆盖更广，SFT 策略集中在窄区域。

### 5.4 与其他 RL 方法对比

| 类型 | 方法 | 基线 SR | 微调后 SR | $\Delta$ | 训练步数 |
|:---|:---|:---|:---|:---|:---|
| Online | VLA-RL | 76.5 | 81.0 | +4.5 | 10,000 |
| Offline | ARFM | 88.1 | 92.1 | +4.0 | 40,000 |
| Offline | RWR | 88.1 | 90.8 | +2.7 | 40,000 |
| Offline | ReinboT | 88.1 | 91.2 | +3.1 | 40,000 |
| **Ours** | **VLA-RFT** | 86.6 | **91.1** | **+4.5** | **400** |

VLA-RFT 以 400 步达到与 online RL（10K 步）、offline RL（40K 步）相当甚至更优的提升，数据效率优势巨大。

---

## 六、类比总结

把 VLA-RFT 想象成一个"考试模拟器"：

- **Stage I**：老师（世界模型）先学会"出题+批改"的能力，学生（VLA）先学会基本功
- **Stage II**：学生做模拟卷（rollout），老师批改并打分（verified reward），学生根据分数调整答题策略（GRPO），不需要真正参加考试（真实环境交互）
- **Reward Type 3 的精妙之处**：老师不是拿标准答案直接对比，而是把标准答案也用同样的出题方式重新"翻译"一遍再对比，消除了翻译偏差

---

## 七、局限性

### 7.1 奖励仍依赖专家数据

Verified reward 本质上是与专家轨迹的相似度，策略无法发现超越专家的策略。未来可引入 learned reward model（如 VLAC）提供更任务相关的反馈。

### 7.2 世界模型容量瓶颈

138M 的轻量世界模型在 LIBERO 上表现良好，但扩展到更复杂的真实场景时，模型容量可能不足，需要更大规模数据和更大模型。

### 7.3 未集成规划

当前世界模型仅作为 reward 提供者，未用于前向规划（look-ahead planning），未充分发挥其动力学预测能力。

### 7.4 策略架构限制

当前框架针对 flow-matching 策略设计（特别是 SDE-Policy 的 Sigma Net），扩展到自回归离散 token VLA 或 diffusion policy 需要额外适配。

---

## 八、个人思考

### 8.1 与 WMPO 的互补视角

WMPO 在**隐空间**做世界模型 + PPO，VLA-RFT 在**像素空间**做世界模型 + GRPO。两者形成了有趣的互补：

- 隐空间方案计算快但信息有损，像素空间方案保真度高但计算重
- VLA-RFT 的 verified reward 比 WMPO 的 learned reward model 更可验证、更稳定
- VLA-RFT 的 400 步极端高效可能得益于像素级 dense reward 的信号丰富度

### 8.2 SDE-Policy 的通用性

将 flow matching ODE 扩展为 SDE 的技巧（引入 Sigma Net）是一个通用方法，可以为所有基于 flow matching 的 VLA 提供 RL 所需的 log-likelihood。这与 FPO++ 用 CFM loss 差值近似 likelihood ratio 的思路形成对比——SDE-Policy 更"正统"但引入了额外网络，FPO++ 更轻量但是近似。

### 8.3 Verified Reward 的设计哲学

Type 3 reward 的核心洞察——在同一生成空间内对比而非跨空间对比——是一个值得推广的设计原则。在所有涉及生成模型的评估场景中，都应考虑消除生成质量偏差的影响。

### 8.4 样本效率的上限在哪

400 步就能提升 4.5 个百分点，这暗示 SFT 基线仍有大量"低垂果实"可被 RL 摘取。但随着基线变强，RFT 的边际收益是否会快速递减？论文中 Base (3w) → Base (15w) 的 SFT 阶段增量已经很小，这可能意味着 RFT 的收益空间有限。

---

## 参考

- **WMPO**（Sun et al., 2025）：隐空间世界模型 + PPO 的离线 RL 后训练 VLA，本站有[笔记](./WMPO_2025.md)
- **RISE**（2026）：组合式世界模型 + 想象空间 RL，本站有[笔记](./RISE_2026.md)
- **WoVR**（2026）：幻觉感知世界模型 RL，本站有[笔记](./WoVR_2026.md)
- **FPO++**（2026）：CFM 损失差值近似似然比的 flow policy RL，本站有[笔记](./FPO_2026.md)
- **VLA-RL**（Lu et al., 2025）：在线 PPO 微调自回归 VLA，本站有[笔记](./VLA_RL_2025.md)
- **RLVLA**（Liu et al., 2025）：RL 在语义和执行维度提升 VLA 泛化，本站有[笔记](./RLVLA_2025.md)
- **ReinFlow**（Zhang et al., 2025）：flow matching + 在线 RL 的基础方法
- **VLA-Adapter**（Wang et al., 2025）：轻量级 VLA adapter 范式，本文基线
- **VERL / HybridFlow**（Sheng et al., 2025）：分布式 RL 训练框架
- **iVideoGPT**（Wu et al., 2024）：交互式视频预测世界模型
