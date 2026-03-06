# RL-VLA 综述：VLA 模型的强化学习后训练全景图

> 论文：*A Survey on Reinforcement Learning of Vision-Language-Action Models for Robotic Manipulation*
>
> 机构：NTU、BUPT、清华大学
>
> 作者：Haoyuan Deng*, Zhenyu Wu*, Haichao Liu* 等
>
> 发布时间：2025年12月
>
> [TechRxiv](https://doi.org/10.36227/techrxiv.176531955.54563920/v1)

---

## 一句话总结

首篇系统梳理 RL 如何后训练 VLA 模型的综述，沿"架构—训练范式—部署—评测"四个维度建立分类体系，覆盖 60+ 篇 RL-VLA 工作，从动作建模、奖励设计、转移建模到在线/离线/测试时 RL 范式，再到 sim-to-real 和真实世界 RL 部署的完整生命周期。

---

## 一、为什么需要这篇综述

### 1.1 VLA 的现状与瓶颈

VLA 模型（如 OpenVLA、$\pi_0$、$\pi_{0.5}$）通过大规模预训练获得了通用的视觉-语言-动作先验，展现出跨物体、跨任务的零样本/少样本泛化能力。但其核心训练范式——**行为克隆（BC/SFT）**——存在根本局限：

- 只能模仿，无法超越专家数据的质量天花板
- 缺乏失败恢复的演示数据，OOD 场景表现脆弱
- 纯模仿目标阻止了策略探索更优的未见策略

### 1.2 RL 后训练的必要性

RL 通过试错式自探索和奖励驱动优化，能弥补 BC 的不足：

- **突破数据天花板**：策略可以发现演示数据中未覆盖的更优行为
- **OOD 泛化**：在线交互使策略适应分布外场景
- **闭环优化**：从开环推理转向闭环反馈控制

实证表明：RL 微调后的 VLA 在 LIBERO 等基准上相比 SFT 有显著泛化提升（RLVLA [42]）。

### 1.3 综述的定位

尽管 RL 在 LLM（DeepSeek-R1 等）和机器人领域分别有综述，但**专门针对 RL + VLA 的系统性综述此前不存在**。本文填补了这一空白。

---

## 二、分类体系总览

综述按以下四维度组织：

```
RL-VLA
├── Architecture（架构）
│   ├── Action（动作建模）
│   │   ├── 自回归模型（VLA-RL, TGRPO, SimpleVLA-RL 等）
│   │   ├── 生成式模型（πRL, FPO, ARFM 等）
│   │   └── 双系统模型（Hume 等）
│   ├── Reward（奖励设计）
│   │   ├── 内在奖励（PBRS, RND, SASR 等）
│   │   └── 外在奖励（VLAC, Eureka, RoboCLIP 等）
│   └── Transition Modeling（转移建模）
│       ├── 物理仿真器（Isaac Sim, Gazebo）
│       └── 学习型世界模型（Dreamer, WMPO, VLA-RFT 等）
├── Training Paradigms（训练范式）
│   ├── 在线 RL-VLA
│   ├── 离线 RL-VLA
│   └── 测试时 RL-VLA
├── Deployment（部署）
│   ├── Sim-to-Real（域随机化、数字孪生）
│   └── 真实世界 RL（人在环、自主恢复、安全探索）
└── Benchmarks（评测）
    ├── 仿真基准（LIBERO, Meta-World, ManiSkill 等）
    └── 真实世界基准（LeRobot, SERL, FMB 等）
```

---

## 三、架构维度：动作、奖励、转移建模

### 3.1 动作建模

三类 VLA 动作架构对 RL 优化有不同影响：

| 架构 | RL 优化层级 | 核心优势 | 核心挑战 | 代表方法 |
| --- | --- | --- | --- | --- |
| **自回归** | Token 级 | 直接提供动作预测概率，PPO/GRPO 可直接使用 | 离散 token 损失精细控制；细粒度化则降低 token 区分度 | VLA-RL, TGRPO, SimpleVLA-RL, CO-RFT |
| **生成式**（Diffusion/Flow） | 序列级 | 输出连续轨迹，时域一致性好 | 无显式动作概率，需近似密度或损失代理 | $\pi_{\text{RL}}$, FPO/FPO++, ARFM |
| **双系统** | Bridge 级 | 高层推理 + 低层控制解耦 | 两层价值估计异质性导致联合训练不稳定 | Hume |

**个人观察**：自回归 VLA 是当前 RL 后训练的主流（Table I 中 ~70% 方法），因为其天然提供动作概率。生成式 VLA 的 RL 后训练是前沿方向，FPO++ 和 $\pi_{\text{RL}}$ 代表两条不同路径（损失代理 vs. 噪声注入）。

### 3.2 奖励设计

| 类别 | 子类 | 核心思路 | 代表方法 |
| --- | --- | --- | --- |
| **内在奖励** | 基于势函数的奖励整形（PBRS） | 用辅助势函数密化奖励，不改变最优策略 | PBRS, 距离启发式 |
| | 探索驱动奖励 | 好奇心、RND、计数法鼓励探索新状态 | RND, SASR |
| **外在奖励** | 人类对齐奖励 | RLHF、偏好学习、人在环奖励整形 | DemPref, Sirius |
| | 模型生成奖励 | LLM/VLM 生成奖励代码或直接评估 | Eureka, RoboCLIP, GVL, VLAC |

**关键洞察**：模型生成奖励正在取代手工奖励——Eureka 用 LLM 迭代生成奖励代码，在多项操作任务上超越人类设计的奖励。VLAC 通过对比学习 + 负样本增强了奖励的可解释性。

### 3.3 转移建模（世界模型）

世界模型正在成为 RL-VLA 的关键组件：

| 类型 | 建模空间 | 代表方法 | 优劣 |
| --- | --- | --- | --- |
| **状态级** | 隐空间 | Dreamer, WMPO | 计算高效，但丢失视觉细节 |
| **观测级** | 像素空间 | iVideoGPT, GWM, EmbodiedDreamer | 保留视觉保真度，但计算昂贵 |
| **VLA 专用** | 集成到 VLA pipeline | VLA-RFT, World-Env, WMPO | 语言条件化推理 + 物理理解统一 |

---

## 四、训练范式

### 4.1 在线 RL-VLA

在线 RL-VLA 是当前最活跃的方向，沿 5 个子方向推进：

**（1）策略优化**

| 算法家族 | 代表方法 | 关键创新 |
| --- | --- | --- |
| PPO 变体 | FLaRe, RLRC, VLA-RL, RIPT-VLA | Robotic PRM 稠密奖励、RLOO 优势估计 |
| GRPO | SimpleVLA-RL, TGRPO, DeepThinkVLA | 无 Critic 更新、轨迹级优势、CoT 对齐 |
| Flow 策略专用 | FPO/FPO++, $\pi_{\text{RL}}$ | CFM 损失代理、Flow-Noise/Flow-SDE |
| 偏好对齐 | GRAPE | 轨迹级偏好优化 + VLM 代价函数 |

RLVLA [42] 的实证对比：PPO > GRPO > DPO 在在线微调场景下。

**（2）样本效率**

关键策略：演示预训练 warmup（iRe-VLA）、VLM 内置 Actor-Critic（VLAC）、人类干预加速（DAFT）、自参照成功轨迹作奖励（SRPO）。

**（3）训练稳定性**

- 动态 Rollout 采样（RIPT-VLA）
- 轨迹级优势估计降低方差（TGRPO）
- 世界模型合成 rollout 减少真实交互方差（World-Env, VLA-RFT）

**（4）主动探索**

- LLM 任务分解引导探索（Plan-Seq-Learn）
- 流形约束探索（SOE）
- 探索性采样生成 OOD 数据（RESample）

**（5）基础设施**

RLinf / RLinf-VLA 提出统一的在线 RL-VLA 训练框架，支持 PPO/GRPO 等多种算法。

### 4.2 离线 RL-VLA

离线 RL 从静态数据集学习，适合高风险或资源受限场景：

| 方向 | 核心思路 | 代表方法 |
| --- | --- | --- |
| **数据利用** | 定制表征增强奖励 + 保守约束防偏移 | ReinboT（RTG 密化）、$\pi_{0.6}^*$（优势条件化）、ConRFT（Cal-QL + BC） |
| **目标修改** | 架构感知目标设计 + 数据驱动目标适配 | Q-Transformer、ARFM（flow 目标）、RL-100（保守门控） |

**核心挑战**：离线数据集通常来自 IL/SFT pipeline，缺乏完整的 MDP 元组（特别是丰富的奖励和失败数据），限制了价值估计和策略泛化。

### 4.3 测试时 RL-VLA

不更新模型参数，在部署时适应新任务：

| 机制 | 核心思路 | 代表方法 |
| --- | --- | --- |
| **价值引导** | 预训练价值函数重排候选动作 | V-GPS, Hume |
| **记忆缓冲引导** | 检索历史经验辅助决策 | STRAP, RA-DT, ReSA |
| **规划引导适应** | MCTS 搜索或价值监控 | VLA-Reasoner, BGR |

---

## 五、真实世界部署

### 5.1 Sim-to-Real

| 方法 | 核心思路 | 代表方法 |
| --- | --- | --- |
| **域随机化** | 随机化仿真参数逼近真实多样性 | SimpleVLA-RL（零样本迁移）、Continual DR |
| **数字孪生** | 创建真实系统的同步虚拟副本 | Real-Is-Sim、RialTo、DREAM |

**关键发现**：即使 SimpleVLA-RL 在仿真中性能优异，sim-to-real 迁移后成功率仍显著下降——仿真单独不足以支撑可靠的真实部署。

### 5.2 真实世界 RL

三大子方向：

**人在环 RL**：
- 矫正干预（HIL-SERL, ConRFT, DAFT）
- 恢复辅助（Generalist, VLAC）
- 课程任务设计（VLA-RL, MT-Opt, CurricuLLM）

**自主恢复**：
- 无重置学习（LNT, R3L, MEDAL）
- 功能可逆性（Recovery RL, PAINT）
- 语义感知恢复（RECOVER, PaLM-E）

**安全探索**：
- 保守安全评价器（Recovery RL, SLAC）
- 结构化任务分解（GRAPE）
- 实时安全执行（SafeVLA 的 CMDP 框架）

---

## 六、基准与评测

### 6.1 仿真基准

| 基准 | 任务数 | 机器人 | 特色 |
| --- | --- | --- | --- |
| LIBERO | 130 | Franka | 多子基准，RL-VLA 最常用 |
| Meta-World | 50 | Sawyer | 多任务元学习 |
| ManiSkill3 | 100+ | Franka | GPU 加速，接触丰富 |
| BEHAVIOR | 1k | Galaxea | 大规模日常任务 |
| RoboVerse | 1k+ | 多种 | 统一平台 |
| SIMPLER | 8 | Google/WidowX | sim-real 对齐校准 |
| RoboTwin 2.0 | 50 | 多种双臂 | 双臂操作 + 域随机化 |

### 6.2 LIBERO 基准上各方法对比

按平均成功率排序（从综述 Fig. 3）：

| 方法 | 类型 | LIBERO 平均 |
| --- | --- | --- |
| SimpleVLA-RL | 在线 GRPO | 最高 |
| Hume | 测试时价值引导 | 第二梯队 |
| $\pi_{\text{RL}}$ | 在线 Flow RL | 第二梯队 |
| RLinf-VLA | 在线 PPO/GRPO | 中上 |
| RIPT-VLA | 在线 LOOP | 中上 |
| ARFM | 离线 Flow RL | 中等 |
| VLA-RFT | 世界模型 GRPO | 中等 |

### 6.3 评测指标

| 指标 | 含义 | 引入者 |
| --- | --- | --- |
| 成功率 | 任务完成比例 | 通用 |
| 平均回合回报 | 累积奖励 | 通用 |
| 安全代价 | 约束违反程度 | SafeVLA |
| 周期时间 | 数据采集-更新-部署的循环速度 | RLDG, CO-RFT |
| 干预率 | 人类监督介入频率 | ConRFT |

---

## 七、开放挑战与未来方向

### 7.1 长时域任务扩展

当前 RL-VLA 仅监督最终动作，缺乏对中间推理过程的引导。有前景的方向：Chain-of-Action-Thought 监督 + 记忆检索机制（STRAP, RA-DT）。

### 7.2 Model-based RL for VLA

当前方法大量依赖仿真 rollout，样本效率低。世界模型（VLA-RFT, World-Env, WMPO）提供了更高效的路径：学习环境动力学来生成信息丰富的奖励和合成状态。

### 7.3 高效可扩展的真实机器人训练

受限于并行化不足和人工监督依赖。有前景方向：推理 agent 自动处理失败、反应式安全探索、多机器人共享训练 + real-to-sim rollout。

### 7.4 可靠可复现的 RL-VLA

RL-VLA 对超参数、设计选择和随机环境动力学高度敏感。需要一致的训练 pipeline、受控评测环境和标准化的算法设置报告。

### 7.5 安全与风险感知

不完美感知、延迟控制和有限监督下的安全探索仍是核心挑战。结合预测性风险建模 + 约束策略优化 + 语言条件化安全推理是有前景的方向。

---

## 八、个人思考

### 8.1 RL-VLA 的"LLM RLHF 时刻"

这篇综述清晰地展示了一个趋势：VLA 正在重走 LLM 的路径——从大规模预训练（GPT-3 ↔ $\pi_0$）到 RL 后训练（RLHF/DeepSeek-R1 ↔ RL-VLA）。但 VLA 面临的挑战更大：

| 维度 | LLM RL 后训练 | VLA RL 后训练 |
| --- | --- | --- |
| 环境 | 文本空间，生成即交互 | 物理世界，需要模拟器或真实机器人 |
| 奖励 | 人类偏好相对易标注 | 任务成功信号稀疏且延迟 |
| 安全 | 输出有害文本可过滤 | 动作不可逆，可能造成物理损害 |
| 样本效率 | 可大规模并行采样 | 真实交互昂贵，仿真有 gap |

### 8.2 技术路线的收敛

从综述的 Table I 可以观察到几个收敛趋势：

1. **算法收敛**：PPO 和 GRPO 占据主导，DPO 在 VLA 场景下表现不如 PPO（RLVLA 的实证结论）
2. **基座收敛**：OpenVLA（及其 OFT 变体）是最常用的基座模型，$\pi_0$ 在 flow 策略方向占据主导
3. **评测收敛**：LIBERO 成为事实标准仿真基准

### 8.3 世界模型是关键瓶颈

综述将世界模型列为五大未来方向之首。从已有工作看：

- **WMPO**：隐空间 MLP，简洁但表达力有限
- **VLA-RFT**：用 GRPO 在世界模型 rollout 上优化
- **World-Env**：VLM 语义反射 + LOOP 优化
- **RISE/WoVR**：组合式/幻觉感知世界模型

世界模型的核心矛盾是"保真度 vs. 计算效率"——像素级世界模型保真但昂贵，隐空间世界模型高效但丢信息。这一矛盾尚未根本解决。

### 8.4 对站内已有笔记的关联

本综述覆盖了站内已有的多篇论文笔记，可作为导航索引：

- **在线 RL**：VLA-RL, SimpleVLA-RL, TGRPO, RLVLA, GRAPE, FPO++, $\pi_{\text{RL}}$（SAC Flow 相关）, RLinf/RLinf-VLA/RLinf-USER
- **离线 RL**：$\pi_{0.6}^*$, SRPO
- **世界模型 RL**：RISE, WoVR, WMPO
- **测试时 RL**：TACO（反探索）

---

## 九、参考

- Deng et al., "A Survey on Reinforcement Learning of Vision-Language-Action Models for Robotic Manipulation," TechRxiv, 2025.
- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv 1707.06347, 2017.
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning," arXiv 2402.03300, 2024.
- Black et al., "$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control," 2024.
- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv 2406.09246, 2024.
- Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning," NeurIPS, 2023.
