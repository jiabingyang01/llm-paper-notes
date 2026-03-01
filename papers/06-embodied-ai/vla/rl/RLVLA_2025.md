# What Can RL Bring to VLA Generalization?——VLA 泛化性的系统性实证研究

> 论文：*What Can RL Bring to VLA Generalization? An Empirical Study*
> 机构：清华大学、上海期智研究院、中关村学院
> 发布时间：2025年5月（NeurIPS 2025）
> 🔗 [arXiv](https://arxiv.org/abs/2505.19789) | [项目主页](https://rlvla.github.io)

---

## 一句话总结

围绕"RL 到底能给 VLA 泛化性带来什么"这一问题，构建了覆盖视觉、语义、执行三个维度的系统性泛化基准，发现 PPO 是 VLA 最有效的 RL 算法（优于 DPO 和 GRPO），并给出了一套高效 PPO 微调 recipe（共享 actor-critic 骨架 + 暖身 + 单 epoch），RL 在语义理解和执行鲁棒性上显著优于 SFT，在视觉鲁棒性上两者持平。

---

## 一、问题与动机

### 1.1 SFT 的根本局限

VLA 模型的主流训练范式是**监督微调（SFT）**——对专家演示做行为克隆。这种方式存在一个根本问题：**复合误差（compounding error）**。

- 专家演示只覆盖了"正确"的轨迹
- 部署时一旦偏离训练分布，模型进入从未见过的状态，误差逐步积累
- 最终表现为对 OOD 场景的脆弱性

Ross & Bagnell (2010) 证明了这种训练-测试分布不匹配会导致后悔值随时序 horizon **二次增长**。

### 1.2 RL 的泛化优势：从 LLM 到 VLA

在 LLM 领域，RL 后训练的泛化优势已被广泛验证：

- "SFT memorizes, RL generalizes" (Chu et al., 2025)：SFT 倾向于记忆训练分布，而 RL 能学到更通用的能力
- DeepSeek-R1 等工作表明 RL 能解锁推理泛化

VLA 本质上继承了 LLM/VLM 的规模和结构，但 RL 对 VLA **具体能带来哪些泛化收益**，缺乏系统性的研究。现有工作（如 FLaRe）展示了 PPO 微调 VLA 的可行性，但**泛化能力的细粒度分析不是其重点**。

### 1.3 本文要回答的核心问题

> **RL 微调相比 SFT 能给 VLA 的泛化性带来哪些独特收益？在视觉、语义、执行三个维度上各有什么表现？**

---

## 二、预备知识

### 2.1 问题建模：POMDP

每个语言条件机器人任务建模为部分可观测马尔可夫决策过程（POMDP）：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \mathcal{O}, \mathcal{L}, P(s_0), \gamma)$$

策略 $\pi_\theta$ 根据最近 $H$ 步观测和语言指令输出动作：

$$a_t \sim \pi_\theta(a_t \mid o_{t-H+1:t}, l)$$

### 2.2 SFT 与 RL 的形式化差异

**SFT** 最小化专家演示上的损失：

$$\mathcal{L}_{\text{SFT}}(\theta) = \sum_{(\tau^{(i)}, l^{(i)}) \in \mathcal{D}_T} \sum_{t=0}^{K_i - 1} \ell_{\text{SFT}}(\hat{a}_t^{(i)}, a_t^{(i)})$$

**RL** 最大化与环境交互的累积回报：

$$\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{M-1} \gamma^t R(s_t, l)\right]$$

关键区别：SFT 只"模仿"，RL 通过试错"优化"。SFT 的训练信号完全来自离线数据，而 RL 的训练信号来自策略自身与环境的实时交互。

### 2.3 OpenVLA 架构

本文基于 OpenVLA (Kim et al., 2024)，一个 7B 参数的开源 VLA：

- **视觉编码器**：SigLIP + DINOv2 双流融合
- **语言骨架**：Llama-2 7B
- **动作表示**：RT-2 式离散化，每维连续动作映射到 256 个 bin，产生 $d_a$ 个 action token
- **训练目标**：仅在 action token 上计算 next-token cross-entropy loss
- **观测**：单帧 RGB 图像（$H = 1$）

---

## 三、RL 算法对比：PPO vs GRPO vs DPO

### 3.1 三种算法在 VLA 上的适配

所有方法均使用 **LoRA (rank=32)** 微调，保持计算可行性。

**PPO**：标准 actor-critic，使用 clipped importance ratio 和 GAE 计算优势函数：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t,\; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]$$

其中 $\pi_\theta(a_t|s_t)$ 是所有 action token 概率的**乘积**，优势 $\hat{A}_t$ 由 GAE 给出：

$$\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l[r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})]$$

同时测试了 PPO-ORZ 变体（$\gamma=1, \lambda=1$，即禁用 GAE），灵感来自 Open-Reasoner-Zero。

**GRPO**：用组内采样估计基线，不需要显式价值函数。每组 8 条轨迹从同一初始状态采样，优势计算为组内归一化回报：

$$\hat{A}_t^i = \frac{r^i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

**DPO**：使用 Trajectory-wise Preference Optimization (TPO)，从稀疏奖励推断轨迹级偏好对：

$$\mathcal{L}_{\text{TPO}} = -\mathbb{E}_{(\zeta_w, \zeta_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta\left(\log\frac{\pi_\theta(\zeta_w)}{\pi_{\text{ref}}(\zeta_w)} - \log\frac{\pi_\theta(\zeta_l)}{\pi_{\text{ref}}(\zeta_l)}\right)\right)\right]$$

### 3.2 算法对比结论

PPO **一致性地显著优于** GRPO 和 DPO。论文给出了两个假说：

**GRPO 失败的原因**：NLP 任务中，同一 prompt 的不同回复之间相对独立；但在机器人的 POMDP 中，每个动作**不可逆地改变环境状态**，具有强烈的非平稳性，导致 GRPO 的组内相对优势估计不稳定。用大白话说——语言模型的多次采样是"重新开始重新说"，而机器人的多次采样虽然从同一状态出发，但后续状态分布迅速发散。

**DPO 失败的原因**：
1. **稀疏奖励**使得区分轨迹质量的信号极弱（很多轨迹都失败了，彼此难以比较）
2. **分布偏移**——离线收集的对比数据与当前策略的执行分布相距甚远

---

## 四、PPO 高效微调 Recipe

### 4.1 共享 Actor-Critic 骨架

PPO 需要 actor（策略）和 critic（价值函数）两个网络。天真做法是各用一个 7B 模型，但显存开销翻倍。

本文的方案：**actor 和 critic 共享整个 Transformer 骨架**，只在最后一层 Transformer block 的输出上接一个三层 MLP value head。

关键设计：**value head 接收第一个 action token 位置的 hidden vector $h^0$**。对比实验表明：
- $h^0$（第一个 action token）：最高且最稳定的回报
- $h^n$（最后一个 action token）：较差
- $[h^0, \dots, h^n]$（所有 action token 拼接）：次优
- 独立 Critic backbone：回报相当，但**训练慢 35%，显存多 83%**（81.3 GB vs 44.4 GB）

直觉解释：第一个 action token 的 hidden state 已经编码了完整的视觉和语言信息（因为 causal attention 的特性，这个位置能看到所有图像和语言 token），是最合适的状态表示。

### 4.2 VLA 暖身（Warm-up）

直接从 OpenVLA 预训练权重开始 RL 训练是可以的，但初始性能太差导致探索效率低。先用 **140 条演示轨迹做 SFT 暖身**，可以将 RL 收敛所需的环境交互步数**减少约 50%**，且最终渐近性能相当。

### 4.3 最小化 PPO Epoch

PPO 中的 epoch 参数控制每批数据的梯度更新次数。实验发现：

| PPO Epoch | 样本效率 | 墙钟时间 |
| --- | --- | --- |
| 1 | 基准 | 基准 |
| 2 | 无提升 | ~2x |
| 5 | 无提升 | ~5x |

**epoch=1 是最佳选择**——增加 epoch 不提升性能，只增加训练时间。

### 4.4 训练效率

使用上述 recipe，整个 PPO 训练在**单张 NVIDIA A100 GPU 上约 42 小时**即可收敛。

---

## 五、泛化性评估基准

### 5.1 任务设计

基于 ManiSkill 仿真器，使用 8-DoF WidowX-250S 机械臂执行 pick-and-place 任务。

**奖励设计**：稀疏奖励——
- 抓取正确物体并持续握住：+0.1
- 成功放置：+1.0

**训练域随机化**：16 种桌面外观 $\times$ 16 种物体 $\times$ 位置扰动

### 5.2 三维泛化测试

#### Vision（视觉鲁棒性）

| 测试场景 | 描述 |
| --- | --- |
| Unseen Table | 5 种训练中未见过的桌面 |
| Dynamic Texture (weak/strong) | 前景（物体+机械臂）叠加随机纹理，透明度 0.3/0.5 |
| Dynamic Noise (weak/strong) | 全图叠加随机纹理，透明度 0.3/0.5 |

#### Semantics（语义理解）

| 测试场景 | 描述 |
| --- | --- |
| Unseen Objects | 9 种未见物体 |
| Unseen Receptacles | 16 种未见容器 |
| Unseen Instruction Phrasings | 16 种未见指令模板（"Place the $O$ on the $R$"等） |
| Multi-Object (IND/OOD) | 桌上两个物体，需选正确的一个 |
| Distractive Receptacle | 桌上有一个干扰容器 |
| Multi-Receptacle (OOD) | 两个未见容器，需放入正确的一个 |

#### Execution（执行鲁棒性）

| 测试场景 | 描述 |
| --- | --- |
| Unseen Position | 物体和容器位置超出训练分布 |
| Unseen Robot Init Pose | 机器人初始关节角度随机化 |
| Mid-Episode Object Reposition | 执行过程中物体被瞬移到新位置 |

---

## 六、核心实验结果

### 6.1 SFT 的数据缩放规律

SFT 在 500~64k 条演示轨迹上训练至收敛：
- 分布内和 OOD 性能都在 **16k 条轨迹处饱和**
- 继续增加数据到 64k 不再有显著提升

因此选择 16k-SFT 作为对比基线。

### 6.2 RL vs SFT 的整体对比

| 维度 | RL vs SFT | 核心数据 |
| --- | --- | --- |
| **Vision** | 持平 | 两者性能下降幅度接近，RL 无明显优势 |
| **Semantics** | RL 显著胜出 | RL 平均 OOD 成功率 0.728 vs SFT 0.599，RL 性能下降 -22.4% vs SFT -30.3% |
| **Execution** | RL 大幅胜出 | RL 平均 OOD 成功率 0.783 vs SFT 0.462，RL 性能下降 -16.5% vs SFT -47.8% |

### 6.3 细分任务结果

以下表格整理自 Table 1（成功率，3 seeds 平均）：

#### Vision 维度

| 场景 | SFT | RL |
| --- | --- | --- |
| IND (训练分布) | 0.781 | **0.938** |
| Unseen Table | 0.719 | **0.844** |
| Texture-w | 0.719 | **0.833** |
| Texture-s | 0.557 | **0.630** |
| Noise-w | 0.708 | **0.854** |
| Noise-s | 0.505 | **0.667** |

#### Semantics 维度

| 场景 | SFT | RL |
| --- | --- | --- |
| Unseen Obj. | 0.453 | **0.714** |
| Unseen Recep. | 0.615 | **0.750** |
| Unseen Instruct | 0.672 | **0.891** |
| Multi-Obj. (IND) | 0.615 | **0.750** |
| Multi-Obj. (OOD) | 0.297 | **0.578** |
| Disturb Recep. | 0.672 | **0.812** |
| Multi-Recep. | 0.458 | **0.599** |

#### Execution 维度

| 场景 | SFT | RL |
| --- | --- | --- |
| Obj. Pos. | 0.568 | **0.807** |
| Robot Pose | 0.339 | **0.797** |
| Mid-Episode Obj. Rep. | 0.286 | **0.745** |

最后一项最为震撼：SFT 仅 28.6%，RL 达到 74.5%——**2.6 倍提升**。

### 6.4 Sim-to-Real 初步验证

将训练好的策略零样本迁移到 Franka Panda 真实机器人（不同于训练用的 WidowX）：

| 指标 | SFT | RL |
| --- | --- | --- |
| 抓取成功率 | 0.10 | **0.43** |
| Pick-and-Place 成功率 | 0.00 | **0.27** |

SFT 在 sim-to-real 迁移中几乎完全失败，RL 虽然绝对性能不高但显著更好。

### 6.5 Action Chunking 验证

使用 OpenVLA-OFT（chunk size=4）重复实验，RL 的优势在更强的架构下依然成立。

---

## 七、为什么 RL 泛化性更好？深层分析

### 7.1 轨迹覆盖对比

可视化训练轨迹分布（Fig. 8）揭示了关键差异：

- **SFT 轨迹**：紧密聚集在 motion planner 生成的固定路径上，工作空间覆盖窄、末端执行器旋转角度范围小
- **RL 轨迹**：覆盖更广的工作空间和更丰富的末端执行器朝向

这种更广泛的状态覆盖是 RL 在 Execution 维度上泛化优势的根本原因。

### 7.2 四个典型场景的定性分析

| 场景 | SFT 行为 | RL 行为 |
| --- | --- | --- |
| Vision/强噪声 | 抓起物体后反复掉落，无法定位容器 | 成功连续抓取并放置 |
| Semantics/未见物体 | 对已抓住的物体反复尝试抓取，陷入死循环 | 正确提起并完成任务 |
| Execution/未见初始姿态 | 到错误位置抓取失败后无法调整，卡住 | 失败后调整位置重新抓取成功 |
| Execution/物体瞬移 | 朝原位置运动，将物体扫落桌面 | 重新定位物体并成功完成 |

核心差异归结为一个能力：**纠错（error recovery）**。RL 通过大量试错，学会了"失败后怎么办"的策略，而 SFT 从未见过失败场景。

### 7.3 各维度泛化差异的假说

**Vision 持平**：无论 SFT 还是 RL，视觉鲁棒性主要来自训练域的视觉随机化（16 种桌面），两种方法都受益于此，也都不超越随机化边界。RL 的试错探索不会"发明"新的视觉输入。

**Semantics RL 胜出**：通过试错，RL 学到了更通用的"抓取"技能，不过度依赖物体类型——它学会了"怎么抓东西"而非"怎么抓特定的茄子"。

**Execution RL 大幅胜出**：RL 的在线交互天然覆盖更多样的状态，包括偏离标准路径的情况。SFT 严格沿 motion planner 路径，一旦偏离就手足无措。

---

## 八、局限性与未来方向

1. **SFT 数据来源单一**：仅使用 motion planner 生成的演示，未包含人类演示数据（可能低估了 SFT 的上限）
2. **任务范围有限**：仅测试了 pick-and-place；更复杂的长时域多任务场景有待验证
3. **仅在仿真中测试**：虽有初步 sim-to-real 验证，但系统性的真实世界实验缺失
4. **单一 VLA 骨架**：仅在 OpenVLA 上验证，是否对 flow matching VLA（如 π₀）同样成立？

---

## 九、个人思考

### 9.1 与项目中其他论文的联系

本文是 VLA+RL 领域的**系统性基准工作**，与本项目已有笔记有多处交叉：

- **VLA-RL**：同样验证了 PPO 微调自回归 VLA 的有效性，但 VLA-RL 侧重 Robotic PRM 解决稀疏奖励问题，而本文侧重三维泛化分析
- **RISE & WoVR**：用世界模型做想象空间 RL，绕开了在线交互的成本；本文的 PPO recipe 需要在线交互，但给出了更直接的泛化分析
- **π₀.₆\***：使用离线 RL（优势条件化）而非在线 PPO；本文对 DPO 的负面结论提示离线 RL 在机器人领域需要更精巧的设计（如 π₀.₆\* 的分布式价值函数）
- **RLinf-VLA**：同样对比了 PPO/GRPO，结论一致（PPO 优于 GRPO），互为验证

### 9.2 "SFT memorizes, RL generalizes" 的边界

本文最重要的结论是 RL 的泛化优势**不是均匀的**：
- Vision 维度几乎无优势——RL 无法"创造"新的视觉经验
- Semantics 维度有中等优势——RL 的试错学到了更通用的物体交互技能
- Execution 维度优势最大——RL 的在线探索天然拓展了状态覆盖

这提示我们：**RL 的泛化优势本质上来自"更广的状态覆盖"**。在 RL 探索能有效拓展覆盖的维度上（执行），优势最大；在探索无法改变输入分布的维度上（视觉），则没有优势。

### 9.3 PPO > GRPO 的深层含义

GRPO 在 LLM 领域大放异彩（DeepSeek-R1），但在机器人领域表现不佳。本文和 RLinf-VLA 的一致结论暗示：**机器人控制与语言生成的 MDP 结构有根本差异**。语言生成更接近 bandit（每次生成相对独立），而机器人控制是真正的长 horizon POMDP，状态转移的非平稳性使得无 critic 的方法难以稳定。

### 9.4 高效 PPO Recipe 的实用价值

共享骨架 + 暖身 + epoch=1 的 recipe 极其实用：
- 单卡 A100 + 42 小时就能训完
- 相比独立 actor-critic 节省 83% 显存
- 这为更多团队在有限资源下尝试 VLA+RL 降低了门槛

---

## 参考

- [VLA-RL (Liu et al., 2025)](VLA_RL_2025.md) — 在线 PPO 微调自回归 VLA
- [RISE (Liu et al., 2026)](RISE_2026.md) — 想象空间 RL 自改进 VLA
- [RLinf-VLA (Zhong et al., 2025)](RLinf_VLA_2025.md) — 统一 VLA+RL 训练框架
- [π₀.₆\* (Intelligence et al., 2025)](pi06star_2025.md) — RECAP 离线 RL
- [WoVR (Li et al., 2026)](WoVR_2026.md) — 幻觉感知世界模型 RL
