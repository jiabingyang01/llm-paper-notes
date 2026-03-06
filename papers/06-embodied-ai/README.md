# 06 Embodied AI

具身智能：VLA 模型、世界模型、机器人策略 RL 训练、模仿学习等。

---

## 🤖 VLA（Vision-Language-Action）

### 基础模型

π₀ 系列、RT 系列、OpenVLA、GR00T 等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [π₀](vla/foundation/pi0_2024.md) | Flow Matching VLA、VLM 骨架 + Action Expert、跨构型预训练、预训练/后训练范式 | 2024 |
| [π₀.₅](vla/foundation/pi05_2025.md) | 异构多源数据协同训练、分层推理（子任务预测 + 低层控制）、开放世界泛化 | 2025 |

### 高效推理

VLA 模型推理加速，包括 Token 缓存/剪枝、动态计算、量化等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [LAC](vla/efficient/LAC_2026.md) | 可学习自适应 Token 缓存、光流运动先验、Gumbel-Softmax 端到端优化、1.76× 加速 | 2026 |
| [SD-VLA](vla/efficient/SD_VLA_2026.md) | 静态-动态 Token 解耦、多级缓存层次、可学习重缓存门、长时程建模、2.26× 加速 | 2026 |
| [RLRC](vla/efficient/RLRC_2025.md) | 结构化剪枝 + SFT/RL 恢复 + 4-bit 量化、90% 剪枝率、8× 显存压缩、2.3× 加速 | 2025 |
| [VLA-Cache](vla/efficient/VLA_Cache_2025.md) | 训练无关跨帧 Token 缓存、注意力驱动任务相关性过滤、层自适应复用策略、1.7× 加速 | 2025 |

### RL 后训练

用强化学习微调或改进机器人策略，包括 VLA + RL 自改进、Flow/Diffusion Policy + RL 等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [ConRFT](vla/rl/ConRFT_2025.md) | 一致性策略、Cal-QL + BC 离线训练、HIL 在线 RL、真实世界 96.3% 成功率 | 2025 |
| [DiffRL Data](vla/rl/DiffRL_Data_2025.md) | 扩散策略 + PPO 数据生成、BC Warm-Start、低方差轨迹、LIBERO-130 | 2025 |
| [FPO++](vla/rl/FPO_2026.md) | CFM 损失差值代理似然比、逐样本裁剪、非对称信任域 ASPO、sim-to-real | 2026 |
| [GigaBrain-0.5M*](vla/rl/GigaBrain_2026.md) | 世界模型 RL、RAMP、优势+未来状态条件化、Wan2.2、HILR 迭代训练 | 2026 |
| [GRAPE](vla/rl/GRAPE_2025.md) | 轨迹级 DPO（TPO）、VLM 代价函数自动生成、多元对齐目标、plug-and-play | 2025 |
| [GR-RL](vla/rl/GR_RL_2025.md) | 数据过滤 + 形态对称增强 + 隐空间在线 RL，通才 VLA 特化为精密操作专家 | 2025 |
| [MoRE](vla/rl/MoRE_2025.md) | Mixture of LoRA Experts、自回归 Q-learning、混合质量数据、四足多任务 VLA | 2025 |
| [π₀.₆*](vla/rl/pi06star_2025.md) | RECAP 优势条件化离线 RL、分布式价值函数、VLA 吞吐量翻倍 | 2025 |
| [PLD](vla/rl/PLD_2026.md) | 残差 RL 专家、基础策略探针、混合轨迹蒸馏、VLA 自改进、LIBERO 99% | 2026 |
| [RISE](vla/rl/RISE_2026.md) | 组合式世界模型、想象空间 RL、VLA 自改进 | 2026 |
| [RL-Co](vla/rl/RL_Co_2026.md) | Sim-Real RL Co-Training、SFT 正则防遗忘、OpenVLA / $\pi_{0.5}$ 双验证 | 2026 |
| [RLinf](vla/rl/RLinf_2025.md) | M2Flow 宏-微流变换、弹性流水线、上下文切换、RL 训练系统 | 2025 |
| [RLinf-USER](vla/rl/RLinf_USER_2026.md) | 统一硬件抽象、云-边通信、全异步流水线、持久化缓冲区、真实世界在线学习 | 2026 |
| [RLinf-VLA](vla/rl/RLinf_VLA_2025.md) | Hybrid Fine-grained Pipelining、统一 VLA+RL 框架（PPO/GRPO）、LIBERO-130 达 98.11% | 2025 |
| [RL-VLA Survey](vla/rl/RL_VLA_Survey_2025.md) | 综述：RL-VLA 架构、在线/离线/测试时训练范式、sim-to-real 部署、评测基准 | 2025 |
| [RLVLA](vla/rl/RLVLA_2025.md) | PPO 优于 DPO/GRPO、共享 Actor-Critic、RL 在语义和执行维度显著优于 SFT | 2025 |
| [RPD](vla/rl/RPD_2025.md) | VLA→RL 策略蒸馏、PPO + MSE 引导、稀疏奖励加速、视角变化鲁棒、ManiSkill3 | 2025 |
| [SAC Flow](vla/rl/SAC_Flow_2026.md) | Flow 策略 × 序列模型、GRU/Transformer 重参数化、off-policy RL | 2026 |
| [SC-VLA](vla/rl/SC_VLA_2026.md) | 稀疏世界想象、残差 SAC 在线修正、内生密集奖励、Flow Matching | 2026 |
| [SRPO](vla/rl/SRPO_2025.md) | 自参照策略优化、世界模型隐空间 progress-wise 奖励、V-JEPA 2、GRPO 扩展 | 2025 |
| [TACO](vla/rl/TACO_2025.md) | Test-Time Scaling、Anti-Exploration、轻量 CFN 伪计数器选择 in-support 动作 | 2025 |
| [TGRPO](vla/rl/TGRPO_2025.md) | 无 Critic 在线 RL、LLM 自动生成多阶段稠密奖励、步级/轨迹级双层组相对优势 | 2025 |
| [TwinRL](vla/rl/TwinRL_2026.md) | 数字孪生探索放大器、探索空间扩展、Sim-to-Real 引导探索、HiL | 2026 |
| [VLAC](vla/rl/VLAC_2025.md) | 统一 Actor-Critic、Pairwise Progress Delta、真实世界 RL、分级人机协作 | 2025 |
| [VLA-RFT](vla/rl/VLA_RFT_2025.md) | 视频世界模型充当模拟器、Verified Reward（MAE+LPIPS）、SDE-Policy、GRPO 400 步微调 | 2025 |
| [VLA-RL](vla/rl/VLA_RL_2025.md) | 在线 PPO 微调自回归 VLA、Robotic Process Reward Model、多模态多轮对话建模 | 2025 |
| [WMPO](vla/rl/WMPO_2025.md) | 隐空间世界模型、Imagination Rollout + PPO、离线 RL 后训练 VLA | 2025 |
| [World-VLA-Loop](vla/rl/World_VLA_Loop_2026.md) | 闭环联合优化、SANS 近成功数据、Cosmos-Predict 2 + 奖励预测头、迭代 RL | 2026 |
| [WoVR](vla/rl/WoVR_2026.md) | 幻觉感知世界模型 RL、关键帧初始化 Rollout、策略-模型协同进化（PACE） | 2026 |

---

## 🌍 World Models

视频世界模型、动力学预测、可控生成、想象与规划。

> 暂无笔记，敬请期待。
