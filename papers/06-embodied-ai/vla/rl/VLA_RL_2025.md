# VLA-RL：用可扩展的在线强化学习提升自回归 VLA——原理详解

> 论文：*VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning*
> 机构：清华大学深圳国际研究生院、南洋理工大学
> 发布时间：2025年5月
> 🔗 [arXiv](https://arxiv.org/abs/2505.18719) | [PDF](https://arxiv.org/pdf/2505.18719) | [代码](https://github.com/GuanxingLu/vlarl)

---

## 一句话总结

VLA-RL 将机器人操作轨迹建模为**多模态多轮对话**，用 PPO 在线强化学习微调自回归 VLA（OpenVLA-7B），配合自动生成伪奖励标签训练的 **Robotic Process Reward Model** 解决稀疏奖励问题，在 LIBERO 基准 40 个任务上超越最强 SFT 基线 4.5%，并展现出机器人领域推理时扩展定律（inference scaling law）的早期信号。

---

## 一、问题与动机

### 1.1 模仿学习的天花板

当前 VLA 模型（如 OpenVLA、RT-2、π₀-FAST）的核心训练范式是**模仿学习（Imitation Learning）**：从人类专家演示中学习策略。这种"利用（exploitation）"离线数据的方式有一个根本局限：

- **状态覆盖有限**：专家演示只覆盖了状态空间的一小部分——专家走的都是"正确路径"
- **OOD 脆弱性**：一旦执行中偏离了专家轨迹（哪怕一点点），模型就进入了从未见过的状态分布，可能雪崩式失败
- **无纠错能力**：从未见过"犯错后如何恢复"的数据

这个问题在 LLM 领域已经被充分认识到：**纯预训练（利用离线数据）的收益正在触及天花板，RL 后训练成为突破口**（如 DeepSeek-R1、DAPO 等）。

### 1.2 那为什么不直接对 VLA 做 RL？

传统机器人 RL 面临几个棘手问题：

| 挑战 | 传统 RL | VLA-RL 的应对 |
| --- | --- | --- |
| 数据效率低 | 从头学需要海量交互 | 从预训练 VLA 出发，搜索空间大幅缩小 |
| 奖励工程复杂 | 需要精心设计每个任务的奖励函数 | 用 Robotic PRM 自动化奖励密集化 |
| 模型规模小 | MLP、小网络 | 直接在 7B 参数的 VLA 上做 RL |
| 单任务 | 每个任务单独训练 | 40 个任务联合训练 |

### 1.3 核心洞察：从 LLM-RL 到 Robot-RL 的类比

VLA-RL 的关键洞察是：**自回归 VLA 本质上就是一个 LLM**（只不过输出的是动作 token 而非文字 token），因此可以直接复用 LLM 领域已经成熟的 RL 训练框架（PPO/GRPO + Process Reward Model）。

| LLM-RL | VLA-RL |
| --- | --- |
| 文本 prompt → 文本回复 | 图像+指令 → 动作 token 序列 |
| 多轮对话 | 多步操作轨迹 |
| Process Reward Model（评价推理过程） | Robotic PRM（评价操作过程） |
| 推理时扩展（更多 token → 更好推理） | 测试时优化（更多训练 → 更好操作） |

---

## 二、预备知识

### 2.1 OpenVLA 回顾

OpenVLA-7B 是当前领先的开源自回归 VLA：

- **骨架**：Llama-2-7B
- **视觉编码器**：双流（SigLIP + DinoV2）
- **动作表示**：将机器人 7 维动作（末端执行器位姿 + 夹爪）的每一维离散化为 256 档，用 7 个 token 表示一步动作
- **生成方式**：标准自回归 next-token prediction

### 2.2 PPO 基础

PPO（Proximal Policy Optimization）是一种 on-policy 策略优化算法：

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\left[\min\left(\frac{\pi_\theta(a_t|o_t, v_t^{in})}{\pi_{\theta_{old}}(a_t|o_t, v_t^{in})}A_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|o_t, v_t^{in})}{\pi_{\theta_{old}}(a_t|o_t, v_t^{in})}, 1-\epsilon, 1+\epsilon\right)A_t\right)\right]$$

核心思想是通过裁剪（clipping）限制新旧策略的偏离程度，防止过大更新导致策略崩溃。优势函数 $A_t$ 通过 GAE（Generalized Advantage Estimation）计算。

### 2.3 Process Reward Model

Process Reward Model（PRM）是 LLM 推理领域的重要工具：不是只在最终结果给奖励（Outcome RM），而是在**推理过程的每一步**给出奖励信号。这解决了稀疏奖励导致的信用分配问题——模型能知道哪一步做对了、哪一步做错了。

---

## 三、核心方法

### 3.1 机器人操作 = 多模态多轮对话

VLA-RL 最优雅的设计在于将机器人操作重新建模为**多模态多轮对话**：

**状态空间**：$\mathcal{S} = \mathcal{O} \times \mathcal{V}^m$（图像空间 × 文本输入空间）

**动作空间**：$\mathcal{V}^n$（VLA 输出的 token 序列）

**一步交互的对应关系**：

| 多轮对话 | 机器人操作 |
| --- | --- |
| 用户输入（文本） | 观测图像 $o_t$ + 任务指令 $v_t^{in}$ |
| 模型输出（文本） | 动作 token 序列 $v_t^{out}$ |
| 一轮对话 | 一步控制 |
| 整段对话 | 一条操作轨迹 |

**动作对数概率的分解**：

$$\log \pi_\theta(\mathbf{a}_t | o_t, v_t^{in}) = \sum_{i=1}^{|A|} \log \pi_\theta(v_{t,i}^{out} | o_t, v_t^{in})$$

其中 $|A| = 7$ 是 OpenVLA 的动作自由度。每一步动作的概率等于 7 个 token 概率的乘积（对数空间下为求和）。

**为什么这个建模很重要？** 它让我们可以直接复用 LLM-RL 的整套基础设施（PPO 训练、token-level 概率计算、GAE 优势估计等），无需为机器人 RL 专门设计新的算法框架。

### 3.2 Robotic Process Reward Model（RPRM）

这是 VLA-RL 解决稀疏奖励问题的核心组件。

#### 3.2.1 问题：环境只给稀疏奖励

大多数机器人任务只在**最终成功时**给 +1 奖励，过程中全是 0。对于一个需要几十甚至上百步的操作任务，模型根本不知道中间哪些步骤是有价值的。

#### 3.2.2 解法：将奖励建模为 next-token prediction

既然 VLA 本身就是一个自回归模型，那就用另一个 VLM 来"评估"VLA 的动作序列是否在朝正确方向前进。

RPRM 的训练目标：

$$\mathcal{L}_{\text{rprm}}(\phi) = -\mathbb{E}_t\left[\sum_j \log p_\phi(v_{t,j}^{rprm} | v_{t,<j}^{out}, o_t, v_t^{in})\right]$$

本质上就是训练一个 VLM，让它学会：给定当前观测和动作历史，预测这个动作序列有多大概率导向成功。

#### 3.2.3 自动伪奖励标签生成

不需要人工标注！流程完全自动化：

**第一步：里程碑分割（Milestone Segmentation）**
- 收集成功的专家轨迹
- 根据**夹爪开合状态的显著变化**自动将轨迹分割为子任务段（夹爪开/关通常标志着一个功能步骤的完成）

**第二步：进度标注（Progress Labeling）**
- 在每个子任务段内，找到末端执行器**速度接近零**的关键帧（通常对应稳定状态或精细动作的完成）
- 将导向这些关键帧的动作序列标记为正伪奖励

**最终奖励**：

$$r_t = r_t^{sparse} + r_t^{rprm}$$

环境的稀疏金标准奖励 + RPRM 的密集预测奖励。

### 3.3 VLA-RL 系统设计

RL 训练高度依赖工程实现。VLA-RL 提出了几个关键工程设计：

#### 3.3.1 课程选择策略（Curriculum Selection）

不是均匀随机选择训练任务，而是自适应地优先选择"刚好在能力边界"的任务：

$$P(task_j) \propto \exp\left((0.5 - s_j) / \tau\right)$$

其中 $s_j$ 是任务 $j$ 的当前成功率，$\tau$ 控制探索程度。这个公式的效果是：
- 成功率 ~50% 的任务（能力前沿）被高概率选中
- 太简单（接近 100%）和太难（接近 0%）的任务被降权
- 兼顾已掌握任务（防遗忘）和困难任务（保持挑战性）

#### 3.3.2 Critic Warmup

值函数网络（Critic）从头训练时，初始的价值估计非常不准确。如果直接用来计算优势函数 $A_t$，会误导策略更新。

解法：先用 SFT 预训练策略收集一批轨迹，**单独训练 Critic 若干轮**，再开始联合 policy-value 优化。实验表明这一步将成功率从 80.0% 提升到 90.2%。

#### 3.3.3 GPU 均衡的向量化环境

- 每个训练 GPU 分配一组环境（避免单 GPU 内存爆炸）
- 用 `all_reduce` 跨 GPU 同步环境状态给推理引擎
- 1 个 GPU 专门做推理（vLLM 加速），其余 GPU 做训练

#### 3.3.4 基础设施细节

- **推理加速**：将 OpenVLA 实现为 vLLM 插件（原始 HuggingFace 的 `generate` 函数在大 batch 下有 bug）
- **训练并行**：PyTorch FSDP
- **精度**：bfloat16
- **参数高效微调**：LoRA（Rollout 时将 LoRA 合并回主模型后广播给推理引擎）

### 3.4 完整算法流程

每一轮迭代分为两个阶段：

**Rollout 阶段**：
1. 合并 LoRA 权重 → 广播给推理引擎
2. N 个并行环境同时运行 M 步
3. 每步：观测 → VLA 生成动作 token → 解码为动作 → 环境执行 → 记录 token log-prob、值估计、RPRM 奖励
4. 收集完整的 batch 轨迹数据

**Learning 阶段**：
1. 用 GAE 计算优势函数 $A^{GAE}$
2. 用 PPO 目标函数更新策略网络（LoRA 参数）
3. 同时更新值网络

---

## 四、实验结果

### 4.1 实验设置

- **基准**：LIBERO——4 个任务套件（Spatial、Object、Goal、Long），共 40 个操作任务
- **基础模型**：OpenVLA-7B（SFT 微调后的 checkpoint）
- **评估**：每个套件 500 个 episode
- **训练成本**：48 GPU 小时

### 4.2 主要结果

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | 平均 | 平均排名 |
| --- | --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 78.3% | **92.5%** | 68.3% | 50.5% | 72.4% | 4.0 |
| Octo (SFT) | 78.9% | 85.7% | **84.6%** | 51.1% | 75.1% | 3.5 |
| OpenVLA (SFT) | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% | 3.5 |
| GRAPE (DPO) | 87.6% | 91.2% | 82.2% | 55.8% | 79.2% | 2.3 |
| **VLA-RL** | **90.2%** | 91.8% | 82.2% | **59.8%** | **81.0%** | **1.5** |
| π₀-FAST（参考） | 96.4% | 96.8% | 88.6% | 60.2% | 85.5% | - |

**关键发现**：
1. VLA-RL 超越最强 SFT 基线 4.5%，超越 DPO 基线 1.8%
2. 仅 48 GPU 小时 RL 训练，OpenVLA-7B 就**接近** π₀-FAST（商业模型、高质量数据训练）的水平
3. 在 LIBERO-Long（最具挑战性的长时域任务）上提升最大（53.7% → 59.8%）

### 4.3 测试时扩展（Inference Scaling）

每 2500 步评估一次：成功率在所有四个任务套件上**持续稳步提升**，没有出现饱和。这是机器人领域**推理时扩展定律**的早期信号——更多的 RL 训练步数（测试时计算）持续带来性能提升。

与 LLM 领域的 inference scaling（更长的 CoT → 更好的推理）不同，机器人的 "scaling" 表现为：更多的在线探索 → 更鲁棒的策略。

### 4.4 训练动态分析

| 指标 | 观察到的趋势 | 含义 |
| --- | --- | --- |
| Episode 长度 | 逐渐减少 | 模型学会了更高效的动作序列（与 LLM-RL 相反——LLM 越训越长） |
| 奖励 | 稳步增长，周期性平台期 | 平台期对应课程切换（任务难度跳跃） |
| Rollout 熵 | 先高后逐渐降低 | 初期探索充分，后期收敛到有效策略 |
| 时间分布 | 训练占比最大 | 环境和推理已被优化，瓶颈转到训练阶段 |

### 4.5 消融实验（LIBERO-Spatial）

| 配置 | 成功率 |
| --- | --- |
| **完整 VLA-RL** | **90.2%** |
| 去除 RPRM | 85.8%（↓4.4%） |
| 去除课程策略 | 88.0%（↓2.2%） |
| 温度 1.5→1.0 | 85.8%（↓4.4%） |
| Critic Warmup 5→0 轮 | 80.0%（↓10.2%） |
| 学习率 2e-5→2e-4 | 0.2%（崩溃） |

**每个组件都不可或缺**：
- **Critic Warmup 影响最大**：没有预热，初始噪声值估计直接毁掉策略
- **RPRM 和温度同等重要**：稀疏奖励不够用，探索不足也不行
- **学习率极其敏感**：高 10 倍直接崩溃（这在大模型 RL 中很常见）

### 4.6 RL vs SFT：为什么 RL 更鲁棒？

**动作覆盖分析**：可视化 SFT 和 RL 采集的动作分布发现——
- SFT 的动作高度集中在空间中心附近（专家偏好的"安全区域"）
- RL 的动作均匀分布在整个动作空间（探索的结果）

**案例分析**：在 "拿起碗并放到盘子上" 的任务中，SFT 策略在抓取时偏移了目标点导致失败，而 VLA-RL 策略能精确抓取。RL 训练特别有助于解决**接触丰富任务**中的对齐问题和过早关合夹爪的问题。

---

## 五、核心原理类比

### 类比：学开车

**SFT（模仿学习）= 看教练开车的录像**

你观看了 100 小时的专家驾驶录像，学会了正常情况下怎么转弯、怎么变道。但当你真正上路，方向盘稍微打歪了一点，你就进入了录像里从未出现过的情况——于是慌了。

**VLA-RL（在线强化学习）= 实际上路练车**

从看过录像后（SFT 预训练）开始，你真正坐上驾驶座开始练习。你会犯错——方向打偏了、油门踩大了——但正是这些错误和纠正的经历，让你学会了如何从偏差中恢复。

**RPRM（过程奖励模型）= 副驾座的教练**

如果只有最终考试的通过/不通过，你很难知道中间哪个环节出了问题。RPRM 就像一个坐在副驾座的教练，能在你每一个操作后给反馈："这个转弯不错"、"变道时机太晚了"——让你更快学会。

**课程选择策略 = 先练简单路段，再上高速**

不是一开始就把你扔到最复杂的路况里，而是先在你"刚好能应付"的难度上练习，等熟练了再升级。

---

## 六、局限性与未来方向

### 6.1 伪奖励标签的启发式局限

当前用夹爪开合和末端速度来分割子任务和标注关键帧，对于更精细的灵巧操作（如旋转、插入等），这种启发式可能不够。

### 6.2 仅验证于仿真

所有实验在 LIBERO 仿真基准中完成，尚未在真实机器人上验证。Sim-to-real gap 可能带来新挑战。

### 6.3 仅适用于自回归 VLA

当前框架假设 VLA 是自回归 token 生成模型。对于基于 diffusion/flow matching 的 VLA（如 π₀），需要不同的 RL 适配方案（作者在论文中也指出这是未来方向）。

### 6.4 RL 训练的工程脆弱性

消融实验表明，学习率高一个数量级就直接崩溃。大模型 RL 的稳定训练仍然高度依赖精心调参。

---

## 七、个人思考

### 7.1 LLM-RL → Robot-RL 的范式迁移正在发生

VLA-RL 最有价值的贡献可能不是具体的算法设计，而是**证明了"LLM RL 后训练"这条路线在机器人领域同样可行**。它直接把 PPO + PRM + 课程学习这套 LLM 领域的成熟工具搬到了机器人领域，这种范式迁移的意义可能比单个方法更深远。

### 7.2 自回归 VLA 的 RL 可行性 vs Flow-based VLA

VLA-RL 能直接复用 LLM-RL 框架，恰恰是因为 OpenVLA 用的是**自回归离散 token** 表示动作。如果底层是 flow matching/diffusion（如 π₀），token-level 的 PPO 就不再适用，需要发展连续空间的 RL 方法。这是两种 VLA 架构路线的一个有趣权衡：自回归架构虽然动作精度和推理速度不如 flow matching，但天然兼容 LLM-RL 框架。

### 7.3 48 GPU 小时就能逼近 π₀-FAST

这个结果非常惊人。π₀-FAST 用了大量高质量数据训练，而 VLA-RL 仅靠 48 小时的在线 RL 就让一个开源 7B 模型接近其性能。这暗示了一种可能性：**RL 后训练可能是比收集更多专家数据更高效的提升策略**。当然需要注意这只是在仿真环境中的结果。

### 7.4 Episode 长度变短 vs LLM 推理变长

VLA-RL 训练中 episode 长度逐渐缩短（模型学会更高效的操作路径），而 LLM-RL 训练中推理链通常变长（模型学会更详细的推理步骤）。这个反差很有趣：物理世界的"聪明"是高效简洁，语言世界的"聪明"是详尽推理。

### 7.5 Critic Warmup 的启示

消融实验中 Critic Warmup 影响最大（去掉直接降 10 个百分点），这验证了一个直觉：对大模型做 RL 时，**值函数的初始化质量决定了训练能否起步**。未来工作可能需要更好的值函数初始化方案，比如从 SFT 模型直接蒸馏值估计。

---

## 参考

- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Let's Verify Step by Step (Process Reward Models)](https://arxiv.org/abs/2305.20050)
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2310.07365)
- [π₀-FAST: Efficient Action Tokenization for VLA Models](https://arxiv.org/abs/2501.09747)
