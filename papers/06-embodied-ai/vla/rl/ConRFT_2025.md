# ConRFT：基于一致性策略的 VLA 模型强化微调

> 论文：*ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy*
>
> 作者：Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, Dongbin Zhao
>
> 机构：中国科学院自动化研究所（CASIA）、中国科学院大学
>
> 会议：RSS 2025（Robotics: Science and Systems）
>
> 发布时间：2025.02
>
> :link: [arXiv](https://arxiv.org/abs/2502.05450) | [项目页面](https://cccedric.github.io/conrft/)

---

## 一句话总结

ConRFT 提出统一的一致性策略（consistency policy）训练目标，将离线阶段（Cal-ConRFT：BC + Cal-QL）与在线阶段（HIL-ConRFT：人机协作 RL）无缝衔接，在 8 个真实世界操作任务上仅需 45–90 分钟在线微调即达到 96.3% 平均成功率，相比 SFT 提升 144%，episode 长度缩短 1.9 倍。

---

## 一、论文要解决什么问题？

### 1.1 SFT 微调 VLA 的瓶颈

预训练 VLA 模型通过 SFT 适配下游任务是主流做法，但面临核心问题：

- **数据质量不一致**：人类遥操作收集的演示数据不可避免地包含次优动作和不一致行为
- **数据量有限**：真实世界数据采集成本高，通常只有 20–30 条演示
- **接触密集型任务表现差**：插轮子、挂中国结等精细操作，SFT 难以学到稳健策略

### 1.2 现有 RL 微调方法的不足

- **PA-RL**（Policy Agnostic RL）：通过策略无关的 Q 函数优化动作再用 SFT 蒸馏，但在接触密集任务中 Q 函数泛化差，无法处理精密操作
- **HG-DAgger**：通过人类修正进行监督微调，但人类修正本身不一致（如挂中国结时插入角度差异大），引入噪声
- **RLDG**：用 RL 策略生成演示再 SFT，数据质量更好但仍是间接优化
- **HIL-SERL**：从头训练 RL 策略，需要大量交互和频繁人类干预，相同时间内仅达 31.9%

### 1.3 关键 insight

**离线阶段引入 Q-learning 为在线 RL 提供良好初始化**。纯 SFT 离线训练虽然可以模仿演示行为，但缺乏价值函数估计，转入在线 RL 时会经历严重的策略遗忘。Cal-ConRFT 在离线阶段就同时训练策略和 Q 函数，使在线阶段能快速适应而非从头学习价值估计。

---

## 二、预备知识

### 2.1 一致性策略（Consistency Policy）

一致性策略是一种基于扩散模型的策略，核心思想是学习将高斯噪声映射到专家动作分布。相比标准扩散策略需要多步去噪，一致性策略可以**一步生成**动作，推理效率高。

将扩散区间 $[\epsilon, K]$（$\epsilon = 0.002$）离散为 $M$ 个子区间，边界为 $k_1 = \epsilon \leq k_2 \leq \cdots \leq k_M = K$。一致性策略定义为：

$$\pi_\psi(a|s) = f_\psi(a^k, k | E_\phi(s))$$

其中 $f_\psi$ 是一致性网络，$a^k \sim \mathcal{N}(0, kI)$ 是带噪声的动作，$E_\phi(s)$ 是预训练 VLA 编码器输出的状态表征。

用大白话说：给定当前观测，模型把一个随机噪声动作"一步拉回"到合理的专家动作空间。

### 2.2 Calibrated Q-Learning（Cal-QL）

Cal-QL 是一种校准的保守 Q-learning 方法，核心目标是让离线训练的 Q 函数对 OOD 动作给出保守估计，同时对数据集内动作保持准确。Critic 损失为：

$$\mathcal{L}_Q^{\text{offline}}(\theta) = \alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot|s)} [\max(Q_\theta(s,a), V^\mu(s))] - \mathbb{E}_{s,a \sim \mathcal{D}} [Q_\theta(s,a)] \right) + \frac{1}{2} \mathbb{E}_{(s,a,s') \sim \mathcal{D}} [(Q_\theta(s,a) - \mathcal{B}^\pi \bar{Q}_{\bar{\theta}}(s,a))^2]$$

- 第一项：保守正则——惩罚 OOD 动作的 Q 值（但不低于参考策略价值 $V^\mu(s)$），保留数据集内动作的 Q 值
- 第二项：标准 TD 误差
- $\alpha$ 控制保守惩罚力度

### 2.3 MDP 形式化

每个机器人任务建模为 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \rho, \gamma)$。奖励设计简单：任务完成 $+10$，每步 $-0.05$。通过预先训练的二值分类器（正/负演示训练）判断任务是否完成。

---

## 三、核心方法

ConRFT 由两个阶段组成，共享统一的一致性策略训练目标。

### 3.1 Stage I：离线微调（Cal-ConRFT）

#### 为什么不能只用 SFT 或只用 Cal-QL？

- **纯 SFT**：没有 Q 函数初始化，在线转换时策略遗忘严重（实验观察到干预率飙升）
- **纯 Cal-QL**：只有 20–30 条演示时，状态覆盖太稀疏，Q 值估计不准，所有任务 0% 成功率
- **Cal-ConRFT**：BC 损失确保学到演示行为，Q 损失同时初始化价值函数

#### 统一训练目标

$$\mathcal{L}_\pi^{\text{offline}}(\psi) = \beta \mathcal{L}_\pi^{\text{BC}} + \eta \mathcal{L}_\pi^Q$$

其中：

**BC 损失**——让策略生成的动作接近演示：

$$\mathcal{L}_\pi^{\text{BC}} = \mathbb{E}_{(s,a) \sim \mathcal{D}, m \sim \mathcal{U}[1, M-1]} \left[ d(f_\psi(a + k_m z, k_m | E(s)), a) \right], \quad z \sim \mathcal{N}(0, I)$$

用大白话说：在演示动作上加噪声到第 $m$ 步的噪声水平，让模型把它还原回去，训练模型的去噪能力。

**Q 损失**——让策略倾向于选择高 Q 值的动作：

$$\mathcal{L}_\pi^Q = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\psi} [Q(s, a)]$$

$\beta$ 和 $\eta$ 分别控制两项权重。离线阶段 $\beta = 1.0, \eta = 0.1$（以 BC 为主）。

### 3.2 Stage II：在线微调（HIL-ConRFT）

#### 数据管理

- **Demo Buffer $\mathcal{D}$**：保留离线阶段的演示 + 人类在线干预数据
- **Replay Buffer $\mathcal{R}$**：存储策略在线交互产生的 transition
- **对称采样**：每个 batch 从 $\mathcal{D}$ 和 $\mathcal{R}$ 各采一半

#### 在线 Critic 更新

不再需要保守正则（分布偏移随在线交互自然减少）：

$$\mathcal{L}_Q^{\text{online}}(\theta) = \mathbb{E}_{(s,a,s') \sim (\mathcal{D} \cup \mathcal{R})} [(Q_\theta(s,a) - \mathcal{B}^\pi Q(s,a))^2]$$

#### 在线 Actor 更新

保持与离线阶段**完全相同的损失结构**，仅扩大采样范围：

$$\mathcal{L}_\pi^{\text{online}}(\psi) = \beta \mathcal{L}_\pi^{\text{BC}} + \eta \mathcal{L}_\pi^Q$$

其中 BC 和 Q 损失现在从 $\mathcal{D} \cup \mathcal{R}$ 采样。在线阶段通常降低 $\beta$（0.5）、提高 $\eta$（1.0），让策略更多地受 Q 值引导而非仅模仿演示。

#### 为什么在线阶段仍保留 BC 损失？

两个关键原因：

1. **防止策略崩溃**：BC 损失确保策略不会偏离演示行为太远，在接触密集任务中尤为重要
2. **稳定探索**：RL 在高维状态-动作空间中探索不稳定，BC 损失提供正则化效果

### 3.3 Human-in-the-Loop 机制

在线阶段引入人类操作员通过遥操作工具（如 SpaceMouse）实时干预：

- **安全保障**：当机器人即将碰撞或施力过大时，人类接管控制
- **脱困辅助**：策略陷入局部最优时，人类引导跳出
- **数据质量**：人类干预数据加入 Demo Buffer $\mathcal{D}$，提供高质量示范

与 HG-DAgger 的区别：ConRFT 的人类干预数据通过 **Q-learning 优化**策略，而非仅用于监督学习。这避免了人类修正不一致带来的噪声。

---

## 四、实验结果

### 4.1 实验设置

- **机器人**：7-DoF Franka Emika
- **VLA 骨干**：Octo-small + 一致性策略动作头
- **观测**：腕部相机（128×128）+ 侧面相机（256×256）+ 本体感知
- **动作空间**：6D 末端执行器 delta 位姿（无抓取）或 7D（含 1D 夹爪）
- **控制频率**：10Hz
- **8 个任务**：Pick Banana、Put Spoon、Open Drawer、Pick Bread、Open Toaster、Put Bread、Insert Wheel、Hang Chinese Knot

### 4.2 主结果

| 方法 | 类型 | 平均成功率 | 平均 Episode 长度 |
| --- | --- | --- | --- |
| SFT | 离线 | 48.8% | 59.9 |
| Cal-ConRFT | 离线 | 39.4% | 57.5 |
| HG-DAgger | 在线（SFT 起点） | 65% (+65%) | 56.3 (1.1x) |
| PA-RL | 在线（SFT 起点） | 71.3% (+81%) | 51.1 (1.2x) |
| **HIL-ConRFT** | **在线（Cal-ConRFT 起点）** | **96.3% (+144%)** | **30.7 (1.9x)** |

关键发现：

- **ConRFT 全面领先**：96.3% vs PA-RL 71.3% vs HG-DAgger 65%
- **Episode 长度大幅缩短**：RL 优化让策略学会更高效地完成任务（30.7 步 vs SFT 59.9 步）
- **接触密集任务优势最大**：Insert Wheel 80%（vs PA-RL 30%），Hang Chinese Knot 100%（vs HG-DAgger 50%）

### 4.3 vs 从头训练 RL（HIL-SERL）

| 方法 | 平均成功率 | 平均 Episode 长度 |
| --- | --- | --- |
| HIL-SERL（从头训练） | 0% → 31.9% | 45.7 |
| **HIL-ConRFT（VLA 微调）** | **39.4% → 96.3%** | **30.7** |

HIL-SERL 在相同训练时间内远不如 ConRFT，因为从头训练需要大量探索，且早期频繁出现破坏性行为。

### 4.4 消融分析

**Cal-ConRFT vs SFT 作为在线起点**：

虽然两者离线成功率接近，但从 SFT 起点在线训练时**干预率显著更高**，说明 SFT 训练的策略在 RL 初期经历严重遗忘。Cal-ConRFT 的 Q 函数初始化使在线转换更平滑。

**增加演示数量能替代 RL 吗？**

| 方法 | 演示数量 | Put Spoon | Put Bread | Insert Wheel | 平均 |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 150 | 60% | 30% | 35% | 41.7% |
| SFT | 150 | 70% | 65% | 40% | 58.3% |
| RLDG | 150（RL 策略收集） | 100% | 100% | 50% | 83.3% |
| **HIL-ConRFT** | **20 + 80–120 在线** | **100%** | **100%** | **80%** | **93.3%** |

增加 7.5 倍人类演示（20→150）仍不及 ConRFT，尤其在接触密集任务（Insert Wheel）上差距明显。

### 4.5 跨 VLA 骨干验证

在 RoboVLM 上测试两种 VLM 骨干（仅微调动作头）：

| 骨干 | Pick Banana | Put Spoon | Hang Chinese Knot | 平均 |
| --- | --- | --- | --- | --- |
| Kosmos-2 (1.6B) | 60% → 100% | 55% → 100% | 45% → 100% | 53.3% → 100% |
| PaliGemma (3B) | 65% → 100% | 30% → 100% | 60% → 100% | 51.7% → 100% |

ConRFT 可应用于任何具有动作头的 VLM 架构。

---

## 五、局限性与未来方向

1. **奖励工程敏感性**：二值分类器奖励存在分布偏移风险，可能导致 reward hacking（机器人学会触发假阳性的特定位置）。稀疏反馈也限制学习速度
2. **冻结编码器和 Transformer 主干**：当前仅微调动作头，限制了感知和表征模块的在线适应能力。引入 LoRA 等参数高效微调可能进一步提升性能
3. **可扩展性**：当前仅验证了单任务微调场景，多任务同时在线 RL 训练尚未探索

---

## 六、个人思考

### 6.1 与项目中其他论文的联系

- **vs VLAC**（统一 Actor-Critic）：两者都做真实世界 RL 微调 VLA，但路线不同。VLAC 用 VLM 训练通用 pairwise progress critic，ConRFT 用传统 Cal-QL 训练任务特定 Q 函数。ConRFT 的优势是无需大规模预训练通用奖励模型，20–30 条演示即可启动；VLAC 的优势是 critic 跨任务迁移
- **vs RL-Co**（sim-real co-training）：RL-Co 在仿真中做 RL + 真实数据 SFT 正则，ConRFT 完全在真实世界做 RL。RL-Co 避免了真实世界安全风险但需要高保真仿真器，ConRFT 通过 HIL 解决安全问题但依赖人类在场
- **vs PLD**（残差 RL 专家）：PLD 冻结 VLA 主干训练轻量残差策略，ConRFT 冻结编码器但微调动作头。两者都认识到不应全参数微调大模型，但 PLD 在仿真中训练再蒸馏回 VLA，ConRFT 直接在真实世界优化
- **vs TwinRL**（数字孪生 RL）：TwinRL 用数字孪生作为探索放大器，20 分钟逼近 100%。ConRFT 需要 45–90 分钟但不依赖数字孪生环境的构建
- **vs GR-RL**（多阶段 VLA 特化）：两者都做真实世界在线 RL，但 GR-RL 侧重隐空间 RL + 形态对称增强，ConRFT 侧重一致性策略的统一离线-在线训练目标

### 6.2 一致性策略的选择

这是本文的一个亮点。相比扩散策略的多步去噪（推理慢），一致性策略可以一步生成动作，这对 10Hz 控制频率的真实世界系统至关重要。同时，一致性策略天然兼容 BC 损失和 Q-learning 损失——BC 损失对应"从加噪动作恢复"的训练目标，Q 损失对应"选择高价值动作"的策略梯度目标。这种统一目标使得离线→在线转换无缝衔接。

### 6.3 离线阶段的必要性

一个有趣的实验发现：纯 Cal-QL（不加 BC 损失）在所有任务上都是 0% 成功率，纯 SFT 则有 48.8%。但加了 BC 损失的 Cal-ConRFT（39.4%）虽然离线成功率低于 SFT，在线阶段却远优于从 SFT 起点训练。这说明**离线阶段的 Q 函数初始化比离线策略性能更重要**——它为在线 RL 提供了关键的"价值方向感"。

---

## 参考

- **CPQL**（Chen et al., AAMAS 2024）：一致性策略 + Q-learning 的原始方法，ConRFT 的理论基础
- **Cal-QL**（Nakamoto et al., NeurIPS 2023）：校准的离线 RL 方法，ConRFT 离线阶段的 Q-learning 基础
- **HIL-SERL**（Luo et al., 2024）：从头训练的真实世界 RL 系统，ConRFT 的主要对比基线
- **PA-RL**（Mark et al., 2024）：策略无关的 Q 函数 + SFT 蒸馏，间接利用 RL 的代表性方法
- **Octo**（Ghosh et al., RSS 2024）：ConRFT 使用的预训练 VLA 骨干
