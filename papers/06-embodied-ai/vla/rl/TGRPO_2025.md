# TGRPO——轨迹级组相对策略优化微调 VLA

> 论文：*TGRPO: Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization*
> 机构：吉林大学人工智能学院
> 发布时间：2025年6月（arXiv:2506.08440v3）
> 🔗 [arXiv](https://arxiv.org/abs/2506.08440)

---

## 一句话总结

提出 TGRPO——一种无 Critic 的在线 RL 框架，通过 LLM 自动生成多阶段稠密奖励，并将 GRPO 的组归一化从"仅轨迹级"扩展为"步级 + 轨迹级"双层优势融合，在 LIBERO 四类任务上平均成功率 80.7%，超越 SFT 4.2%、超越 GRAPE 0.5%。

---

## 一、问题与动机

### 1.1 SFT 的根本局限

VLA 模型（OpenVLA、π₀ 等）的主流训练范式是在人类演示上做行为克隆（SFT）。这种方式有两个根本缺陷：

1. **只从成功中学**：训练数据全是成功轨迹，模型从未见过失败情形，缺乏自纠错能力
2. **复合误差**：部署时偏离训练分布后误差逐步累积，在 OOD 场景下性能急剧下降

用大白话说——SFT 训出来的机器人是"照本宣科"的，一旦遇到没见过的情况就手足无措。

### 1.2 RL 微调面临的三重挑战

RL 通过试错交互来优化策略，理论上能解决上述问题，但在长时域机器人任务中面临三重挑战：

| 挑战 | 具体表现 |
| --- | --- |
| **稀疏奖励** | 真实机器人任务通常只有 episode 结束时的 0/1 成功信号，中间步骤无反馈 |
| **高方差** | 长 horizon 任务包含多个阶段，不同子目标的奖励尺度不一致，梯度方差大 |
| **不稳定优化** | PPO 需要训练额外的 Critic 网络，增加计算开销；GRPO 的组归一化在机器人非平稳 MDP 中不够稳定 |

### 1.3 现有方法的不足

- **PPO**：需要额外 Critic 网络，显存和计算开销翻倍
- **DPO / GRAPE**：依赖离线偏好数据或人工干预，训练周期长
- **标准 GRPO**：只在轨迹级做组归一化，忽略了步级的细粒度信号，对长 horizon 任务的 credit assignment 不充分

### 1.4 核心洞察

> **长 horizon 机器人任务需要同时捕获"全局任务进展"和"局部动作质量"两个层面的优化信号**——仅靠轨迹级归一化容易忽视步级差异，仅靠步级归一化又难以感知全局进度。

---

## 二、预备知识

### 2.1 问题建模：MDP

每个语言条件机器人任务建模为 MDP：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{V}, \mathcal{L})$$

其中 $\mathcal{V}$ 是第三人称相机观测空间，$\mathcal{L}$ 是自然语言指令集。在每个时间步 $t$，VLA 模型根据观测 $v \in \mathcal{V}$ 和指令 $l \in \mathcal{L}$ 输出动作 $a \in \mathcal{A}$，并获得奖励 $r_t$。

优化目标为最大化累积轨迹奖励：

$$R = \sum_{t=1}^{M} r_t$$

其中 $M$ 为多条轨迹中的最短长度（所有环境同时终止）。

### 2.2 GRPO 回顾

Group Relative Policy Optimization（GRPO）最早在 DeepSeekMath 中提出，核心思想是**用组内采样估计基线，不需要显式 Critic**。对 $N$ 条采样结果，优势计算为组内归一化回报：

$$\hat{A}_i = \frac{R_i - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$$

GRPO 在 LLM 领域（DeepSeek-R1）表现出色，但直接迁移到机器人的问题在于：它只在**轨迹级**做归一化，把整条轨迹的累积奖励作为单一信号，无法区分同一轨迹中不同步骤的贡献差异。

### 2.3 OpenVLA 架构

本文基于 OpenVLA 作为基础 VLA 模型：

- **视觉编码器**：SigLIP + DINOv2 双流融合
- **语言骨架**：Llama-2 7B
- **动作表示**：离散化 action token $[\Delta x, \Delta\theta, \Delta \text{Grip}]$
- **微调方式**：LoRA

---

## 三、核心方法

TGRPO 包含两个核心模块：**LLM 自动生成的多阶段奖励** 和 **轨迹-步双层组相对优势估计**。

### 3.1 多阶段稠密奖励设计

#### 动机

如果只使用终端 0/1 奖励，把同一个 reward 传播到所有步骤，会忽略不同动作的异质贡献。例如一条失败的轨迹中，前几步可能已经成功完成了若干子目标——这些"局部成功"不应被一刀切地标记为失败。

#### 设计方法

利用 Claude 3.7 Sonnet 自动分解任务为多个子阶段。以 LIBERO-Object 中"把番茄酱瓶放进篮子"为例：

| 阶段 | 描述 |
| --- | --- |
| 1. Approaching | 末端执行器接近目标物体 |
| 2. Grasping | 抓取目标物体 |
| 3. Moving | 携带物体移向目标位置 |
| 4. Placing | 放置到目标容器 |

定义关键物体位置 $P_{\text{object}} \in \mathbb{R}^3$ 和从成功演示中提取的参考关键姿态 $\{P^1_{\text{pose}}, P^2_{\text{pose}}, \dots, P^j_{\text{pose}}\}$，$P^k_{\text{pose}} \in \mathbb{R}^3$。

每步奖励由两部分组成：

$$R_t = f_1(P_{\text{object}}(t), P^k_{\text{pose}}) + f_2(P^k_{\text{pose}}, s_t)$$

- $f_1(\cdot)$：**基于关键物体的阶段奖励**——根据物体和目标的空间关系判断当前所处阶段，给予阶段奖励
- $f_2(\cdot)$：**基于末端执行器姿态的 shaping 信号**——计算当前末端执行器与参考成功姿态的欧氏距离，鼓励策略对齐专家轨迹

LLM 根据任务描述 + 物体位置 + 参考姿态自动生成奖励函数代码，无需人工 reward engineering。

### 3.2 轨迹-步双层组相对优势估计

这是 TGRPO 的核心算法创新。假设我们有 $N$ 条轨迹，每条长度 $M$，第 $i$ 条轨迹第 $t$ 步的奖励为 $R_{i,t}$。

#### 步级优势（Step-level Advantage）

将同一时间步 $t$ 的 $N$ 条轨迹的奖励组成一组，做组内 z-score 归一化：

$$A^{\text{step}}_{i,t} = \frac{R_{i,t} - \frac{1}{N}\sum_{i=1}^{N} R_{i,t}}{\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\left(R_{i,t} - \frac{1}{N}\sum_{i=1}^{N} R_{i,t}\right)^2}}$$

直觉：在时间步 $t$，哪条轨迹的即时奖励相对更好？这提供了**局部动作质量**的信号。

#### 轨迹级优势（Trajectory-level Advantage）

将 $N$ 条轨迹的累积奖励 $R_i = \sum_{t=1}^{M} R_{i,t}$ 做组内归一化：

$$A^{\text{traj}}_i = \frac{R_i - \frac{1}{N}\sum_{i=1}^{N} R_i}{\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\left(R_i - \frac{1}{N}\sum_{i=1}^{N} R_i\right)^2}}$$

直觉：在整个 episode 中，哪条轨迹的整体表现相对更好？这提供了**全局任务进展**的信号。

#### 双层融合

加权合并两个层面的优势：

$$\text{Adv}_{i,t} = \alpha_1 A^{\text{step}}_{i,t} + \alpha_2 A^{\text{traj}}_i$$

实验中最优设置为 $\alpha_1 = 0.3$，$\alpha_2 = 0.7$——**轨迹级信号占主导**（提供稳定性），步级信号作为补充（提供细粒度指导）。

### 3.3 优化目标

定义重要性采样比：

$$\rho_{i,t} = \frac{\pi_\theta(a_{i,t} | s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} | s_{i,t})}$$

最终优化目标采用 GRPO 风格的 clipped surrogate + KL 正则：

$$\mathcal{J}_{\text{TGRPO}}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left\{\min\left[\rho_{i,t}\text{Adv}_{i,t},\;\text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\text{Adv}_{i,t}\right] - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right\}\right]$$

KL 散度使用无偏估计器：

$$D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(a_{i,t}|s_{i,t})}{\pi_\theta(a_{i,t}|s_{i,t})} - \log\frac{\pi_{\text{ref}}(a_{i,t}|s_{i,t})}{\pi_\theta(a_{i,t}|s_{i,t})} - 1$$

### 3.4 训练流水线

完整的在线 RL 后训练流程：

1. **并行环境采样**：$N$ 个环境从相同初始状态出发，VLA 逐步采样动作，直到某个环境完成任务或全部达到最大步数——所有轨迹同时终止，长度一致，便于步级对齐
2. **多阶段奖励计算**：每步根据物体位置和末端执行器姿态计算稠密奖励
3. **双层优势计算**：按 Eq. (2)(3)(4) 计算步级和轨迹级优势并融合
4. **策略更新**：按 Eq. (6) 计算 TGRPO 损失并更新 LoRA 参数

---

## 四、实验

### 4.1 实验设置

- **基础模型**：OpenVLA + LoRA 微调
- **优化器**：AdamW，学习率 $1 \times 10^{-5}$
- **测试平台**：LIBERO 机器人仿真基准，包含四个任务套件：
  - **Spatial**（10 tasks）：空间位置泛化
  - **Object**（10 tasks）：物体类别泛化
  - **Goal**（10 tasks）：任务目标泛化
  - **Long**（10 tasks）：长 horizon 复杂任务
- **并行环境数**：4（即 group size $N=4$）
- **评估**：每任务 50 个测试 episode，报告平均成功率
- **硬件**：单卡 NVIDIA A100

### 4.2 主实验结果

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| Octo | 77.6 | 84.9 | 82.9 | 50.3 | 73.9 |
| OpenVLA-SFT | 84.7 | 88.4 | 79.2 | 51.1 | 76.5 |
| OpenVLA-DPO | 84.2 | 88.6 | 79.5 | 52.6 | 76.2 |
| GRAPE | 88.5 | 92.1 | **83.1** | 57.2 | 80.2 |
| **TGRPO** | **90.4** | **92.2** | 81.0 | **59.2** | **80.7** |

关键发现：

- **vs SFT**：TGRPO 平均高出 4.2%；在 LIBERO-Long 上高出 8.1%，说明在线 RL 在长 horizon 任务上优势尤为明显
- **vs DPO**：TGRPO 全面超越，DPO 的离线偏好学习在机器人领域效果有限
- **vs GRAPE**：平均略优 0.5%；TGRPO 在 Spatial、Object、Long 上胜出，但 Goal 上不如 GRAPE（81.0 vs 83.1）

### 4.3 消融实验（LIBERO-Goal 套件）

| 方法 | 平均成功率 |
| --- | --- |
| SFT | 88.4 |
| TGRPO w/o Trajectory-level Adv. | 80.2 |
| TGRPO w/o Step-level Adv. | 86.8 |
| **TGRPO (完整)** | **92.2** |

三个关键结论：

1. **去掉步级优势**（仅用轨迹级）→ 86.8%：损失 5.4%——细粒度的步级反馈对 credit assignment 至关重要
2. **去掉轨迹级优势**（仅用步级）→ 80.2%：损失 12.0%——甚至不如 SFT！说明缺乏全局信号的步级优化极不稳定
3. **双层融合**是必要的，两个层面互补不可替代

### 4.4 超参数分析

#### $\alpha_1$ 的影响（$\alpha_1 + \alpha_2 = 1$）

| $\alpha_1$ | 平均成功率 |
| --- | --- |
| 0.1 | 75.2% |
| 0.3 | **81.0%** |
| 0.5 | 79.0% |
| 0.7 | 77.4% |
| 0.9 | 77.0% |

$\alpha_1$ 过小（0.1）：步级信号被忽略，细粒度指导不足；$\alpha_1$ 过大（0.7+）：轨迹级信号被淹没，训练不稳定。最优点在 $\alpha_1 = 0.3$，即**轨迹级信号占 70%**。

#### Group Size $N$ 的影响

| Group Size | 平均成功率 |
| --- | --- |
| 2 | 76.2% |
| 4 | **81.0%** |
| 6 | 79.4% |
| 8 | 80.4% |

$N=2$ 时组内只有两条轨迹，相对比较信号不可靠；$N \geq 4$ 后差异不大；$N=4$ 在准确性和效率间取得最佳平衡。

---

## 五、局限性

1. **仅在仿真中验证**：所有实验都在 LIBERO 仿真器上进行，缺乏真实世界验证
2. **单任务训练**：每次只为单个任务做 RL 微调，未展示多任务/跨任务能力
3. **依赖仿真器状态信息**：奖励函数需要物体位置和末端执行器姿态等特权信息，在真实世界中难以直接获取
4. **基础模型局限**：仅在 OpenVLA（自回归 VLA）上验证，未测试 Flow Matching VLA（如 π₀）
5. **与更强 baseline 缺乏对比**：未与 PPO、RLinf-VLA 等已有 VLA+RL 框架直接比较
6. **LIBERO-Goal 上不如 GRAPE**：81.0 vs 83.1，说明 TGRPO 在某些任务结构上仍有改进空间

---

## 六、个人思考

### 6.1 与项目中其他论文的联系

TGRPO 属于 GRPO 在机器人领域的适配工作，与本项目多篇笔记有密切关联：

- **RLVLA**：同样在 LIBERO 上对比了 PPO/GRPO/DPO 三种算法，结论是 PPO 最优。RLVLA 认为 GRPO 在机器人 POMDP 中不稳定，而 TGRPO 试图通过双层优势融合来修复 GRPO 的这个缺陷——但从结果来看，TGRPO 的 80.7% 仍低于 RLVLA 中 PPO 在 ManiSkill 上的表现，说明这种修补可能不够彻底
- **RLinf-VLA**：同时支持 PPO 和 GRPO，且在 LIBERO-130 上单模型达到 98.11%，远超 TGRPO 的 80.7%。差距主要来自 RLinf-VLA 的系统级优化（Hybrid Pipelining、多轮训练），但也侧面说明 GRPO 变体的上限可能不如 PPO
- **SRPO**：同样使用 GRPO 作为基础优化算法，但通过自参照机制 + 世界模型隐表征奖励进行扩展，思路与 TGRPO 的"双层优势"有异曲同工之处——都在"如何给 GRPO 提供更好的优势估计"这个方向发力

### 6.2 LLM 自动生成奖励函数的价值

TGRPO 用 Claude 3.7 Sonnet 自动分解任务并生成稠密奖励函数，这个设计虽然在本文中只是辅助模块，但可能比算法本身更有实用价值：

- **降低 reward engineering 门槛**：传统 RL 中手工设计奖励函数是最大的瓶颈
- **可扩展到新任务**：只需提供自然语言任务描述，LLM 就能自动生成合理的阶段奖励
- **但受限于仿真器**：需要仿真器提供物体位置等特权信息，在真实世界中需要额外的感知模块

### 6.3 "轨迹级主导"的深层含义

消融实验和超参数分析一致表明：$\alpha_2 = 0.7$（轨迹级权重）远大于 $\alpha_1 = 0.3$（步级权重）是最优的。这意味着在长 horizon 任务中，**全局进展信号比局部动作质量更重要**。

换个角度理解：如果步级信号权重太高，优化可能过度关注"每一步都做到局部最优"，反而忽略全局协调——就像下棋只顾吃子不顾大局。轨迹级信号起到"定海神针"的作用，确保优化方向与任务完成目标一致。

### 6.4 与 TACO 的互补性

有趣的是，TGRPO 和已有笔记中的 TACO 解决的是完全不同维度的问题：

- **TGRPO**：改进**训练阶段**的优化算法，用双层优势估计提升 RL 训练效率
- **TACO**：改进**推理阶段**的动作选择，用伪计数器过滤 OOD 动作

两者理论上可以组合——先用 TGRPO 训练 VLA，再用 TACO 做推理时的 test-time scaling。

### 6.5 方法的简洁性与局限性

TGRPO 的最大优点是方法简洁：不需要 Critic 网络，不需要离线偏好数据，单卡 A100 就能训练。但简洁也带来了局限——80.7% 的平均成功率在 VLA+RL 领域只能算中等水平，与 RLinf-VLA 的 98.11% 有显著差距。这提示我们：**无 Critic 方法在机器人领域的天花板可能确实低于有 Critic 的 PPO**，双层优势融合只是缓解而非解决了 GRPO 在非平稳 MDP 中的根本局限。

---

## 参考

- [RLVLA (Guo et al., 2025)](RLVLA_2025.md) — PPO vs GRPO vs DPO 系统性对比
- [RLinf-VLA (Zhong et al., 2025)](RLinf_VLA_2025.md) — 统一 VLA+RL 训练框架
- [SRPO (Xu et al., 2025)](SRPO_2025.md) — 自参照策略优化 + GRPO 扩展
- [TACO (Yang et al., 2025)](TACO_2025.md) — 推理时反探索 test-time scaling
- [VLA-RL (Liu et al., 2025)](VLA_RL_2025.md) — 在线 PPO 微调自回归 VLA
