# SimpleVLA-RL：用在线 RL 扩展 VLA 训练——原理详解

> **论文**：*SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning*
>
> **作者**：Haozhan Li*, Yuxin Zuo*, Jiale Yu*, Yuhao Zhang* 等
>
> **机构**：清华大学、上海 AI Lab、上海交通大学、北京大学、香港大学、Nature Will、Frontis.AI
>
> **发布时间**：2025 年 9 月（ICLR 2026）
>
> **链接**：[arXiv](https://arxiv.org/abs/2509.09674) | [代码](https://github.com/PRIME-RL/SimpleVLA-RL)

---

## 一句话总结

SimpleVLA-RL 基于 veRL 框架构建面向 VLA 的端到端在线 RL 训练系统，通过**二元结果奖励 + GRPO + 三种探索增强策略**（Dynamic Sampling、Clip Higher、高温 Rollout），在 LIBERO 上将 OpenVLA-OFT 从 91% 提升到 **99.1%** 成功率，仅用 1 条演示的 RL 就超越全量 SFT（96.9% vs 91.0%），在 RoboTwin 2.0 上相对提升 80%（38.3% → 68.8%），并发现 RL 涌现出演示数据中不存在的新行为模式"pushcut"。

---

## 一、问题与动机

### 1.1 SFT 范式的数据瓶颈

当前 VLA 模型的训练范式是"大规模预训练 + SFT"。核心瓶颈有二：

1. **数据稀缺**：高质量机器人轨迹数据采集成本极高，需要精心设计场景、多样化物体和熟练操作员
2. **泛化薄弱**：在有限场景数据上 SFT 的模型倾向于记忆模式而非学习可泛化技能，面对分布偏移（未见物体、新环境）时误差级联，尤其在组合式长时域任务中问题严重

### 1.2 传统 RL 的困境

传统机器人 RL 需要为每个任务手工设计奖励函数，不具备可扩展性。

### 1.3 核心洞察：LRM 的启示

大语言推理模型（如 DeepSeek-R1）的突破表明：**仅用稀疏的结果奖励，RL 就能显著增强模型的逐步推理能力**。SimpleVLA-RL 将这一洞察迁移到机器人领域，提出关键问题：

> **结果驱动的 RL 能否提升 VLA 模型的长时域逐步动作规划能力？**

### 1.4 VLA RL 的技术挑战

将 RL 应用于 VLA 面临三个独特挑战：

| 挑战 | LLM RL | VLA RL |
| --- | --- | --- |
| 轨迹生成 | 开环文本生成 | 闭环环境交互，需持续视觉反馈 |
| 探索效率 | 文本采样自然多样 | 高维动作空间 + 稀疏奖励，探索困难 |
| 基础设施 | 成熟的 LLM-RL 框架 | 缺乏 VLA 特定的推理+渲染并行基础设施 |

---

## 二、预备知识

### 2.1 VLA 的 RL 建模

与 LLM RL 的关键区别在于 **闭环交互**：

- **状态**：$s_t = (o_t^{vis}, o_t^{prop}, l_{task})$，包含视觉观测、本体感受和语言指令
- **动作**：通过离散 action tokenizer 生成动作 token 序列，解码为连续控制指令 $a_t \in \mathbb{R}^d$
- **Rollout**：每步生成 action chunk $(a_t, \ldots, a_{t+k-1})$，执行后获取新状态 $s_{t+k}$，循环直到任务完成或达到最大步数

### 2.2 GRPO

Group Relative Policy Optimization (GRPO) 通过组内归一化计算优势，**无需价值函数**：

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

其中 $G$ 是组内轨迹数。优势直接由同一初始状态下多条轨迹的奖励对比得出。

---

## 三、核心方法

### 3.1 交互式 VLA Rollout

LLM 的 rollout 是自回归生成文本直到终止；VLA 的 rollout 是**闭环交互**——每步动作改变环境，后续动作必须基于实时视觉反馈。

SimpleVLA-RL 选择 **token-based 动作表示**（如 OpenVLA-OFT 的 action tokenizer），因为它天然提供动作分布，既支持温度采样实现探索，又兼容 PPO/GRPO 的策略梯度计算。

### 3.2 二元结果奖励

遵循 DeepSeek-R1 的简洁设计，使用最简单的二元奖励：

$$R(a_{i,t} | s_{i,t}) = \begin{cases} 1, & \text{is\_successful}[\text{traj}_i(a_i, s_i)] \\ 0, & \text{otherwise} \end{cases}$$

轨迹级奖励均匀传播到每个 action token：成功轨迹中所有 token 获得奖励 1，失败轨迹为 0。

**优势**：简单、可扩展、跨环境通用、无需复杂的过程奖励设计。

### 3.3 三种探索增强策略

VLA 模型由于训练轨迹的同质性，倾向于收敛到狭窄的解空间，严重限制 RL 效率。SimpleVLA-RL 提出三种探索增强：

#### (1) Dynamic Sampling

无 Critic 的 RL 算法（如 GRPO）在组内所有轨迹奖励相同时优势估计为零，梯度消失。Dynamic Sampling 在 rollout 时过滤掉全成功或全失败的组，只保留混合结果的组：

$$0 < |\{\text{traj}_i(a_i, s_i) \mid \text{is\_successful}[\text{traj}_i(a_i, s_i)]\}| < G$$

持续采样直到 batch 全部由混合结果组构成，确保非零优势估计和稳定梯度流。

**效果**：LIBERO-Long 上 +15%。

#### (2) Clip Higher

标准 GRPO 的裁剪范围 $[1 - \epsilon, 1 + \epsilon]$ 限制了低概率 token 的概率增长，抑制探索。参照 DAPO，将上限从 1.2 提升到 **1.28**（非对称裁剪 $[0.8, 1.28]$），允许有潜力但当前低概率的动作被更大幅度强化。

**效果**：+10%。

#### (3) Higher Rollout Temperature

将采样温度从 1.0 提升到 **1.6**，生成更多样的轨迹。

**效果**：+15%。

### 3.4 训练目标

去除 KL 散度正则化（参照 DAPO），既节省内存（无需加载参考模型），又解放探索空间。最终目标：

$$\mathcal{J}(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D}, \{a_t\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|s_t)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \min\left(r_{i,t}(\theta)\hat{A}_i, \; \text{clip}(r_{i,t}(\theta), 1-\epsilon_L, 1+\epsilon_H)\hat{A}_i\right) \right]$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}$，$\epsilon_L = 0.2$，$\epsilon_H = 0.28$。

---

## 四、实验结果

### 4.1 实验设置

| 项目 | 配置 |
| --- | --- |
| 基础模型 | OpenVLA-OFT（修改版：单视图、LLaMA2 output head 生成 action token） |
| 仿真基准 | LIBERO（4 套件，40 任务）、RoboTwin 1.0（17 双臂任务）、RoboTwin 2.0（12 任务，4 个时域级别） |
| RL 算法 | GRPO（去 KL、非对称裁剪、Dynamic Sampling） |
| 硬件 | 8×A800 80GB |
| 训练时间 | LIBERO 1-2 天，RoboTwin 12-24 小时 |

### 4.2 LIBERO 主结果

| 模型 | Spatial | Object | Goal | Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT (SFT) | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| **OpenVLA-OFT + SimpleVLA-RL** | **99.4** | **99.1** | **99.2** | **98.5** | **99.1** |

- 平均提升 +8.1%，LIBERO-Long 提升最大 +12.0%
- 超越 π₀（94.2%）和 UniVLA（95.2%）

### 4.3 RoboTwin 2.0 主结果

| 时域级别 | π₀ | RDT | OFT (SFT) | OFT + RL | Δ |
| --- | --- | --- | --- | --- | --- |
| Short（112-130 步） | 45.5 | 24.5 | 21.3 | **64.9** | +43.6 |
| Medium（150-230 步） | 64.8 | 47.8 | 47.1 | **72.5** | +25.4 |
| Long+Extra Long（280-650 步） | 47.8 | 27.8 | 46.5 | **69.0** | +22.4 |
| **Overall** | 52.7 | 33.3 | 38.3 | **68.8** | **+30.5** |

相对提升约 80%，全面超越 π₀ 和 RDT。

### 4.4 数据效率：1 条演示的 RL 超越全量 SFT

| 设置 | Spatial | Object | Goal | Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| One-Traj SFT | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| One-Traj SFT + **RL** | **98.2** | **98.7** | **98.8** | **91.7** | **96.9** |
| Full-Traj SFT | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| Full-Traj SFT + RL | 99.4 | 99.1 | 99.2 | 98.5 | 99.1 |

**每个任务仅 1 条演示 + RL 就达到 96.9%，超越全量 SFT 的 91.0%**。One-Traj RL 与 Full-Traj RL 的差距仅 2.2%。LIBERO-Long 从 17.3% 跃升至 91.7%（+430%）。

### 4.5 泛化分析

在 LIBERO 的 Spatial/Object/Goal 三个维度上，保留 1 个任务作为 unseen，用 9 个任务训练：

- **SFT** 在训练任务达 90%+ 后，unseen 任务常常**灾难性遗忘**至 0%
- **RL** 在训练任务提升的同时，unseen 任务也**持续改善**（Spatial +28.5%、Object +36.5%、Goal +5-15%）

RL 训练使 VLA 保持已有能力的同时学习可迁移的技能。

### 4.6 Sim-to-Real

| 任务 | RDT | OFT (SFT) | OFT + RL | Δ |
| --- | --- | --- | --- | --- |
| Stack Bowls | 60.0 | 38.0 | **70.0** | +32.0 |
| Place Empty Cup | 4.0 | 2.0 | **10.0** | +8.0 |
| Pick Bottle | 10.0 | 0.0 | **14.0** | +14.0 |
| Click Bell | 20.0 | 30.0 | **60.0** | +30.0 |
| **平均** | 23.5 | 17.5 | **38.5** | **+21.0** |

全程仅用仿真数据训练，无真实世界演示。RL 训练在 sim-to-real 迁移中持续有效。

---

## 五、关键发现

### 5.1 "Pushcut"现象：RL 涌现新策略

在 RoboTwin 2.0 的 move_can_pot 任务中，所有演示数据都是 grasp-move-place 策略。RL 训练后，模型自主发现了**直接推物体到目标位置**的更高效策略——这是演示数据中完全不存在的行为。

这与 DeepSeek-R1 的 "Aha Moment" 类似：**RL 驱动的探索能发现 SFT 数据中不存在的新解**。二元结果奖励的设计是关键——抓取和推的方式完成任务都得到同等奖励，避免了程序性约束，释放了探索自由度。

### 5.2 失败模式分析

| 条件 | 结果 |
| --- | --- |
| 基础模型 0% 成功率（无 SFT） | RL 完全失败，仍为 0% |
| 初始成功率 < 5% | RL 改善微乎其微 |
| 初始成功率 ~10%（100 条 SFT） | 平均 7.3% → 25.4% |
| 初始成功率 ~28%（1000 条 SFT） | 平均 28.2% → 50.4% |

**模型先验是 RL 有效性的决定性因素**。存在一个"能力阈值"——低于此阈值，稀疏结果奖励下的探索无法产生有意义的改进。

---

## 六、局限性与未来方向

1. **仅适用于 token-based VLA**：当前框架依赖离散 action token 的概率分布进行策略梯度计算，不适用于 diffusion/flow matching 架构的 VLA（如 π₀）
2. **稀疏结果奖励的局限**：对初始能力太弱的模型无效，无法为 0% 成功率的模型提供学习信号
3. **仿真依赖**：在线 RL 需要大量环境交互，目前仅在仿真器中高效可行，直接在真实世界做大规模在线 RL 仍然困难
4. **超参数敏感**：学习率高一个数量级直接崩溃，需精心调参

---

## 七、个人思考

### 7.1 与项目中其他 VLA-RL 论文的关系

SimpleVLA-RL 与本项目中已有的 [VLA-RL](/papers/06-embodied-ai/vla/rl/VLA_RL_2025) 在研究对象高度重合（都是自回归 VLA + 在线 RL），但设计哲学截然不同：

- **VLA-RL** 采用 PPO（需要 Critic）+ Robotic PRM（密集过程奖励）+ 课程选择策略
- **SimpleVLA-RL** 采用 GRPO（无 Critic）+ 纯二元结果奖励 + 探索增强策略

结果证明**极简设计同样甚至更加有效**：SimpleVLA-RL 在 LIBERO 上 99.1% 远超 VLA-RL 的 81.0%。这暗示对于 VLA RL，精心设计的过程奖励可能不如简单结果奖励 + 充分探索来得有效——与 LLM 领域 DeepSeek-R1 的经验一致。

### 7.2 探索增强是关键

三种探索策略合计贡献了 30-40% 的提升，是 SimpleVLA-RL 成功的核心。Dynamic Sampling 解决了 GRPO 在机器人场景中的梯度消失问题（成功率太低或太高时全组奖励相同），高温采样和非对称裁剪则拓宽了探索空间。这些策略借鉴自 LLM-RL（DAPO、Polaris），说明 **LLM RL 的探索技巧可以直接迁移到机器人领域**。

### 7.3 "Pushcut"的启示

"Pushcut"现象是 VLA RL 领域最令人兴奋的发现之一。它证明 RL 不仅能在已知策略空间中优化，还能**发现全新的解题策略**。这与 [TACO](/papers/06-embodied-ai/vla/rl/TACO_2025) 的反探索（anti-exploration）理念形成有趣对比——TACO 约束策略在已知支撑集内，而 SimpleVLA-RL 鼓励跳出已知空间。

### 7.4 1 条演示 + RL 的意义

每个任务仅 1 条演示就能通过 RL 达到 96.9%，这几乎消除了对大规模人类演示的依赖。结合 [RLinf-USER](/papers/06-embodied-ai/vla/rl/RLinf_USER_2026) 和 [TwinRL](/papers/06-embodied-ai/vla/rl/TwinRL_2026) 等真实世界 RL 系统，可以设想一种新范式：**极少量演示引导 + 大规模仿真 RL + sim-to-real 迁移**。

---

## 参考

- [OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models](https://arxiv.org/abs/2502.19645)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948)
- [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476)
- [veRL (HybridFlow): A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2310.07365)
- [RoboTwin 2.0: A Scalable Data Generator and Benchmark](https://arxiv.org/abs/2506.18088)
