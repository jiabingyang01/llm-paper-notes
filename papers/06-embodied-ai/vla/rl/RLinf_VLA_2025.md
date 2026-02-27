# RLinf-VLA：统一高效的 VLA+RL 训练框架——原理详解

> 论文：*RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training*
> 机构：清华大学、中关村科学院、Infinigence AI、北京大学、UC Berkeley、哈工大、中科院自动化所
> 发布时间：2025年10月
> 🔗 [arXiv](https://arxiv.org/abs/2510.06710) | [PDF](https://arxiv.org/pdf/2510.06710) | [代码](https://github.com/RLinf/RLinf) | [模型](https://huggingface.co/RLinf)

---

## 一句话总结

RLinf-VLA 提出了一个**统一**且**高效**的 VLA+RL 训练框架：通过三种灵活的 GPU 分配模式（colocated / disaggregated / 新提出的 **hybrid fine-grained pipelining**）解决仿真器-生成-训练的资源竞争问题，统一支持多种 VLA 架构（OpenVLA、OpenVLA-OFT）、多种 RL 算法（PPO、GRPO）和多种仿真器（ManiSkill、LIBERO），单一模型首次在 **130 个 LIBERO 任务上达到 98.11%** 成功率，并总结出一套 VLA+RL 的 best practices。

---

## 一、问题与动机

### 1.1 SFT 训练 VLA 的根本局限

当前大多数 VLA 模型仍采用 **SFT（行为克隆）** 进行训练。SFT 的核心缺陷在于 **distribution shift（分布偏移）**：

- 训练时只看到专家的正确轨迹
- 部署时一旦偏离专家分布，误差会逐步积累（compounding error）
- 这从根本上限制了 VLA 的泛化能力和鲁棒性

**RL** 通过与环境交互直接优化任务奖励，可以学到超越专家数据的纠错策略，是 SFT 的互补方案。

### 1.2 现有 VLA+RL 研究的碎片化困境

尽管 RL 微调 VLA 的前景广阔，现有研究面临严重的碎片化问题：

| 维度 | 碎片化表现 |
| --- | --- |
| 模型 | 不同工作用不同 VLA（OpenVLA、RT-2 等），结果无法比较 |
| 算法 | PPO vs GRPO 各有尝试，缺乏统一评估 |
| 仿真器 | ManiSkill（GPU 并行）vs LIBERO（CPU 并行）资源特性完全不同 |
| 规模 | 多数工作仅在单个任务组上训练评估 |

### 1.3 GPU 资源竞争：VLA+RL 独有的系统挑战

与 LLM 的 RL 后训练不同，VLA+RL 需要与**仿真器**反复交互。仿真器本身也消耗大量 GPU 资源（特别是 GPU 并行仿真器如 ManiSkill），这导致**仿真、推理生成、训练**三者之间存在严重的 GPU 资源竞争——现有框架无法高效处理这一问题。

---

## 二、核心方法

### 2.1 三种 GPU 分配策略

RLinf-VLA 框架的核心在于针对 VLA+RL 训练中三个组件（**Simulator** / **Generation** / **Training**）的 GPU 资源分配，提出了三种灵活的模式：

#### (1) Colocated 模式（共置）

所有组件共享全部 GPU：

- **优点**：最大化数据并行度，无 GPU 空闲
- **缺点**：组件之间内存竞争；需要频繁 offload/onload，特别是 Generation 和 Simulator 在 rollout 期间的交替切换开销很大
- **适用场景**：CPU 并行仿真器（如 LIBERO），GPU 主要用于推理和训练

#### (2) Disaggregated 模式（分离）

每个组件分配到独立的 GPU 分区，彼此不重叠：

- **优点**：每个组件可以充分利用分配的资源
- **缺点**：组件间存在依赖等待，产生 **GPU bubble**（如 Simulator 等待 Generation 输出动作时，Simulator 的 GPU 空闲）
- **适用场景**：GPU 资源充足时

#### (3) Hybrid + Fine-grained Pipelining 模式（新提出）

这是本文的核心系统创新。核心思路是：

**组件灵活分配 GPU + 流水线并行消除气泡**。

具体做法：Generation 和 Simulator 分配到不同 GPU 分区，Training 可以使用全部 GPU。在此基础上引入**细粒度流水线**：

将一个 GPU 上的仿真器实例拆分为 $k$ 个子仿真器 $S^{(1)}, S^{(2)}, \ldots, S^{(k)}$，然后交错执行：

1. $t=0$: $S^{(1)}$ 生成初始观测 $o_0^{(1)}$ → 送给 Generation 生成动作 $a_0^{(1)}$
2. **同时**: $S^{(2)}$ 并行生成 $o_0^{(2)}$
3. $a_0^{(1)}$ 就绪后，喂回 $S^{(1)}$ 生成 $o_1^{(1)}$；**同时** Generation 处理 $o_0^{(2)}$ 生成 $a_0^{(2)}$

这种交错调度让 Simulator 和 Generation **并发执行**，有效消除等待气泡。

**三种模式通过 YAML 配置切换**——用户只需指定每个组件的 GPU ID 和 offload 开关：

```yaml
cluster.component_placement.env: [0-3]      # Simulator
cluster.component_placement.rollout: [4-7]   # Generation
cluster.component_placement.actor: [0-7]     # Training
rollout.pipeline_stage_num: 2                # 流水线阶段数 k
```

### 2.2 统一的模型支持

框架通过统一接口支持不同的 VLA 架构：

| 模型 | 参数量 | 动作表示 | 关键特性 |
| --- | --- | --- | --- |
| **OpenVLA** | ~7B | 离散 token | VLM 骨架 + 自回归动作生成 |
| **OpenVLA-OFT** | ~7B | 连续空间 + L1 回归 | 并行解码 + Action Chunking，推理速度 10× 以上提升 |

同时支持 **LoRA** 微调——消融实验表明 LoRA 本身不显著影响性能，但通常需要不同的超参数设置。

### 2.3 统一的仿真器接口

针对两类仿真器提供一致的 Gym 风格接口：

- **ManiSkill**（GPU 并行）：渲染、物理仿真、推理全在 GPU 上，吞吐量高但资源竞争严重
- **LIBERO**（CPU 并行）：仿真逻辑分布在 CPU worker 上，GPU 主要用于渲染和推理

关键接口扩展：

- **`auto_reset`**：子环境终止后自动重置，避免空闲
- **`chunk_step`**：专门处理 action chunk 的环境步进，支持 chunk 内终止时立即重置或延迟重置
- **`use_fixed_reset_state_ids`**：GRPO 要求同一组内所有环境共享初始状态

### 2.4 多算法支持：PPO 与 GRPO

#### PPO

标准的 PPO 使用 GAE（Generalized Advantage Estimation）：

$$\hat{A}_t = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \big(r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})\big)$$

优化目标使用 clipped surrogate：

$$J^{\text{PPO}}(\theta) = \mathbb{E}_t \Big[\min\big(\rho_t(\theta)\hat{A}_t,\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]$$

其中重要性采样比率 $\rho_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_\text{old}}(a_t|o_t)}$。

**Critic 设计**：为避免额外维护一个完整的 Value 模型，actor 和 critic 共享参数，仅在 LLM 最后一层 Transformer block 上挂一个**轻量 3 层 MLP value head**，取第一个 action token 位置的隐藏表示 $h^0$ 预测状态值。

#### GRPO

GRPO 的核心特点是**不需要 Value 函数**，通过组内轨迹的相对比较估计优势：

$$\hat{A}^{(i)} = \frac{\mathcal{R}^{(i)} - \text{mean}(\{\mathcal{R}^{(j)}\}_{j=1}^G)}{\text{std}(\{\mathcal{R}^{(j)}\}_{j=1}^G)}$$

其中 $\mathcal{R}^{(i)}$ 是轨迹 $\tau^{(i)}$ 的总奖励，$G$ 是组大小。同一组内的轨迹必须对应**相同任务和相同初始状态**。

---

## 三、算法设计细节

### 3.1 Action Chunk 下的优势估计

VLA-OFT 使用 action chunking（一次预测一组未来动作），这引入了如何将优势与 chunk 内动作关联的问题。设第 $t$ 个 chunk 为 $c_t = (a_{t,1}, a_{t,2}, \ldots, a_{t,C})$，两种方式：

| 粒度 | 定义 | 效果 |
| --- | --- | --- |
| **Chunk-level** | 整个 chunk 作为宏动作，奖励 $r_t = \sum_{j=1}^C r_{t,j}$ | 更粗糙 |
| **Action-level** | 每个 $a_{t,j}$ 独立有自己的奖励和优势 | 消融实验表明**更优** |

对于 PPO，action-level 对应的 value head 输出 $C$ 维向量（每个原子动作一个值），而 chunk-level 输出标量。

### 3.2 Log-Probability 粒度

框架支持三种 log-probability 计算粒度：

- **Token-level**: $\pi(d_{t,i,j}|o_t, d_{t,i,:j-1})$ — 每个离散 token
- **Action-level**: $\pi(a_{t,i}|o_t, a_{t,:i-1}) = \prod_{j=1}^M \pi(d_{t,i,j}|o_t, d_{t,i,:j-1})$ — 一个原子动作
- **Chunk-level**: $\pi(c_t|o_t) = \prod_{i=1}^C \pi(a_{t,i}|o_t, a_{t,:i-1})$ — 整个 chunk

当 log-prob 粒度细于 advantage 粒度时，使用**广播机制**（如 chunk-level advantage 分配给 chunk 内每个 token）。

### 3.3 GRPO 的关键设计选择

#### Valid Action Mask

在 rollout 中环境可能在 `max_episode_steps` 之前就完成任务。两种策略：

- 使用完整轨迹（对应 `success_at_end` 目标）
- 只使用任务完成前的步骤（对应 `success_once` 目标）—— 即 **Valid Action Mask**

在 LIBERO 中，Valid Action Mask 显著提升 GRPO 性能。

#### 轨迹长度归一化

与 LLM 任务不同，机器人中成功轨迹通常并不比失败轨迹更长。为确保不同长度的轨迹对梯度贡献均衡：

$$J^{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{T_i^{\text{succ}}} \sum_{t=1}^{T_i^{\text{succ}}} \min\Big(\rho_{i,t}(\theta)\hat{A}_i,\ \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\Big)\right]$$

其中 $T_i^{\text{succ}}$ 是轨迹 $\tau_i$ 的有效步数（mask 后）。

#### Success Rate Filter

借鉴 DAPO 的动态采样策略，丢弃组内所有轨迹全部成功或全部失败的组（因为此时组内优势为零，不提供有效梯度信号）。在 OpenVLA + ManiSkill 设置中，这一机制有效防止了训练崩溃。

---

## 四、实验结果

### 4.1 高性能：ManiSkill

在 25 个 pick-and-place 任务上，RL 微调带来 **45%–70%** 的成功率提升：

| 模型 | In-Distribution | OOD Vision | OOD Semantic | OOD Execution | OOD Avg. |
| --- | --- | --- | --- | --- | --- |
| OpenVLA (Base) | 53.91% | 38.75% | 35.94% | 42.11% | 39.10% |
| OpenVLA (RLinf-PPO) | **96.09%** | 82.03% | **78.35%** | **85.42%** | **81.93%** |
| OpenVLA-OFT (Base) | 28.13% | 27.73% | 12.95% | 11.72% | 18.29% |
| OpenVLA-OFT (RLinf-PPO) | **97.66%** | **92.11%** | 64.84% | 73.57% | **77.05%** |

关键发现：**PPO 在 ManiSkill 中一致优于 GRPO 且更稳定**。Base model 的质量对最终泛化性能有决定性影响。

### 4.2 高性能：LIBERO-130

首次用**单一模型**在 130 个 LIBERO 任务（5 个子集）上训练和评估：

| 子集 | Spatial | Object | Goal | 10 | 90 | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| OpenVLA-OFT (Base) | 72.18% | 71.48% | 64.06% | 48.44% | 70.97% | 65.43% |
| OpenVLA-OFT (RLinf-GRPO) | **99.40%** | **99.80%** | **98.79%** | **93.95%** | **98.59%** | **98.11%** |
| $\Delta$ | +27.22 | +28.32 | +34.73 | +45.51 | +27.62 | **+32.68** |

成功率从 ~65% 提升到 **98.11%**，平均提升 32.68 个百分点。

### 4.3 高效率：GPU 分配模式对比

不同仿真器适合不同的分配模式：

| 仿真器类型 | 最优模式 | 相对加速 |
| --- | --- | --- |
| ManiSkill + OpenVLA（GPU 并行） | Hybrid (pipe=2) | **1.61×–1.88×** vs disaggregated |
| ManiSkill + OpenVLA-OFT（GPU 并行 + action chunk） | Colocated / Hybrid (pipe=1) | action chunk 使 simulator 执行时间远超生成时间（~15:1），pipelining 收益减弱 |
| LIBERO + OpenVLA-OFT（CPU 并行） | Colocated | **1.34×–2.27×** vs SimpleVLA-RL |

核心结论：**不同类型的仿真器需要不同的分配策略，不同的交互特性进一步需要自适应配置**。

### 4.4 消融实验总结：Best Practices

#### PPO Tips

1. **Action-level value estimation 优于 chunk-level**：更细粒度的值估计在 ManiSkill 和 LIBERO 上都一致更好
2. **Partial Reset 显著提升采样效率**：成功的子环境立即重置并收集新轨迹，相比 Fixed Episode Length 收敛更快

#### GRPO Tips

1. **轨迹长度归一化 (w/ Norm)** 在 LIBERO 上带来显著提升
2. **Valid Action Mask** 在 LIBERO 上有效，但在 ManiSkill 上效果不明显（任务依赖）
3. **Success Rate Filter** 在 OpenVLA + ManiSkill 中防止训练崩溃

#### 通用 Tips

1. **更大的 rollout batch size** 一致带来更高成功率
2. **LoRA 不直接影响性能**，但需要不同的超参数——例如 GRPO 中 LoRA 用 lr=1e-4 正常，无 LoRA 则需降到 lr=1e-5 才稳定

### 4.5 真实世界部署

在 Franka Panda 机器人上测试 6 种未见物体的 pick-and-place 任务（每种 5 次试验）：

| 指标 | OpenVLA (SFT) | OpenVLA (RLinf-PPO) |
| --- | --- | --- |
| Pick 成功 | 3/30 | **13/30** |
| 完整成功 | 0/30 | **8/30** |

SFT 策略倾向于过冲（overshooting），而 RL 策略虽有轻微振荡但能迭代修正末端执行器姿态，展现出更强的零样本泛化能力。

---

## 五、局限性与未来方向

1. **VLA 架构有限**：目前仅支持 OpenVLA 和 OpenVLA-OFT（均 ~7B），未集成 π₀、π₀.₅ 等 Flow Matching VLA（已规划中）
2. **仅 on-policy 算法**：目前只支持 PPO 和 GRPO，缺少 off-policy 算法（SAC 等），后者在样本效率上可能更有优势
3. **真实世界实验初步**：仅在简单 pick-and-place 上测试了 6 种物体，未涉及长时域或灵巧操作
4. **仿真到真实迁移**：未使用任何 sim-to-real 技术（domain randomization 等），8/30 的成功率仍有很大提升空间
5. **仿真器覆盖不足**：未集成 RoboTwin、IsaacLab 等（已在规划中）

---

## 六、个人思考

### 6.1 与 RLinf 通用框架的关系

RLinf-VLA 基于同一团队的 [RLinf](RLinf_2025.md) 通用 RL 训练系统，但专门针对 VLA+RL 场景做了深度适配。RLinf 提出的 M2Flow 宏-微流变换提供了底层调度框架，而 RLinf-VLA 在此基础上解决了**仿真器资源竞争**这一 VLA 独有的系统挑战。

### 6.2 Hybrid Pipelining 的适用条件

消融实验揭示了一个重要的 insight：pipelining 的收益取决于 **Generation 和 Simulator 执行时间的比例**。当两者接近 1:1 时（如 OpenVLA 每步生成一个动作），pipeline 效果显著（1.88×）；但使用 action chunking 后（一次生成多步，simulator 需要执行多步），比例变为 ~15:1，pipeline 收益大幅缩减。这提示在设计系统优化时，需要根据模型的推理特性动态选择分配策略。

### 6.3 GRPO vs PPO 的适用场景差异

实验中 PPO 在 ManiSkill（25 tasks）上一致优于 GRPO，但在大规模 LIBERO-130 上只展示了 GRPO 结果。结合 GRPO 不需要 Value 模型的轻量优势，一个合理推测是：**在大规模多任务设定下 GRPO 可能更实用**（GPU 开销更低，调参更简单），而 PPO 在中等规模下更稳定高效。

### 6.4 Best Practices 的实践价值

本文最有价值的贡献之一是系统化的消融实验和 best practices 总结。对于想要在自己的 VLA 上做 RL 微调的研究者，这些指导非常实用：action-level value estimation、partial reset、trajectory length normalization、valid action mask 的选择，比单纯的 SOTA 数字更有工程指导意义。

---

## 参考

- **RLinf**：同一团队的通用 RL 训练系统，提供底层 M2Flow 调度框架
- **RL4VLA (VLA-RL)**：首个在线 PPO 微调自回归 VLA 的工作，RLinf-VLA 复现并超越了其结果
- **OpenVLA / OpenVLA-OFT**：框架支持的两种 VLA 架构
- **DAPO**：Success Rate Filter 的灵感来源
- **SimpleVLA-RL**：基于 VeRL 的 VLA+RL 框架，作为效率对比基线
- **VeRL (HybridFlow)**：LLM RL 训练框架，GRPO 组内设计参考
