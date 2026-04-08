# 06 Embodied AI

具身智能：VLA 模型、世界模型、机器人策略 RL 训练、模仿学习等。

---

## 🤖 VLA（Vision-Language-Action）

### 基础模型

π₀ 系列、RT 系列、OpenVLA、GR00T 等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [3D-CAVLA](vla/foundation/3D_CAVLA_2025.md) | 3D 深度感知、CoT 指令分解、ROI 检测、零样本泛化、LIBERO | 2025 |
| [3D-MIX](vla/foundation/3D_Mix_2026.md) | VGGT 3D 融合、9 种策略对比、语义条件化门控、即插即用、GR00T/π-style、SIMPLER +7.0% | 2026 |
| [AimBot](vla/foundation/AimBot_2025.md) | 瞄准线 + 准星视觉空间线索、EE 位姿/朝向/夹爪编码到像素空间、<1 ms 模型无关、LIBERO-Long +5.8、真实世界 27→43/50 | 2025 |
| [AnchorVLA4D](vla/foundation/AnchorVLA4D_2026.md) | 首帧锚帧 + 冻结 Any4D 空间编码器、遮挡遗忘缓解、早期重试、SimplerEnv 64.6%（+13.6%）、真实世界 80% | 2026 |
| [BridgeVLA](vla/foundation/BridgeVLA_2025.md) | 输入-输出 2D 对齐、正交投影、2D 热力图预训练、RLBench 88.2%、3 条轨迹 95.4% | 2025 |
| [ChatVLA](vla/foundation/ChatVLA_2025.md) | Spurious Forgetting 分析、Phased Alignment Training、MoE 双专家（共享 Attention 隔离 MLP）、2B 参数统一理解+控制、MMMU 37.4 | 2025 |
| [CoWVLA](vla/foundation/CoWVLA_2026.md) | 结构-运动解耦 Video VAE、潜在运动链、Chain-of-World、终端帧预测、LIBERO 95.6%、SimplerEnv 76.0%、CALVIN 4.21 | 2026 |
| [CronusVLA](vla/foundation/CronusVLA_2026.md) | Feature Chunking 多帧特征聚合、DiT 跨帧解码器、多帧正则化、SimplerEnv 70.9%、LIBERO 97.0%、SimplerEnv-OR R-Score 86.9 | 2026 |
| [DAM-VLA](vla/foundation/DAM_VLA_2026.md) | 动作路由、双扩散动作头（手臂 class token + 夹爪 register token）、双尺度加权 | 2026 |
| [DeepVision-VLA](vla/foundation/DeepVisionVLA_2026.md) | 视觉敏感性衰减诊断、VL-MoT DINOv3 视觉专家深层耦合、AGVP 动作引导 Token 剪枝、RLBench 83%、真实世界 91.7% | 2026 |
| [DreamVLA](vla/foundation/DreamVLA_2025.md) | 综合世界知识预测（动态区域/深度/语义）、Block-Wise 结构化注意力、DiT 动作头、GPT-2 Medium、CALVIN 4.44 SOTA、LIBERO 92.6% | 2025 |
| [Dexbotic](vla/foundation/Dexbotic_2025.md) | VLA Toolbox、统一模块化框架（VLM + AE）、DexboticVLM（Qwen2.5）、Exp 脚本实验开发、SimplerEnv +46.2% | 2025 |
| [FAST](vla/foundation/FAST_2025.md) | DCT + BPE 频域压缩动作 tokenization、高频灵巧任务、π₀-FAST 匹配扩散 π₀ 训练 5× 加速、FAST+ 通用 tokenizer | 2025 |
| [FutureVLA](vla/foundation/FutureVLA_2026.md) | 联合视觉运动预测（JVPM）、3D-VAE 连续 17 帧编码、双流解耦监督、门控交叉注意力、潜在嵌入对齐、SimplerEnv 80.1%、真实机器人 +26.7% | 2026 |
| [GR-3](vla/foundation/GR3_2025.md) | MoT 架构（Qwen2.5-VL + Action DiT）、VL 协同训练、VR 人类轨迹少样本适配、Task Status 辅助监督、全面超越 π₀ | 2025 |
| [MoH](vla/foundation/MoH_2025.md) | 多 Horizon 动作块并行融合、轻量门控（2k 参数）、跨 Horizon 共识动态推理、Plug-and-Play、LIBERO 99% | 2025 |
| [MemoryVLA](vla/foundation/MemoryVLA_2025.md) | 感知-认知双流记忆库（PCMB）、跨注意力检索 + 门控融合 + 合并压缩、长时域操作、SimplerEnv-Bridge +14.6、LIBERO 96.5%、真实世界时序 +26 | 2025 |
| [MMaDA-VLA](vla/foundation/MMaDA_VLA_2026.md) | 原生离散扩散、统一多模态 token、并行去噪、混合注意力、目标观测生成、LIBERO 98.0%、CALVIN 4.78 | 2026 |
| [OptimusVLA](vla/foundation/OptimusVLA_2026.md) | 双记忆增强（GPM 任务级先验检索 + LCM Mamba 时序一致性）、自适应 NFE、LIBERO 98.6%、2.9× 推理加速 | 2026 |
| [OTTER](vla/foundation/OTTER_2025.md) | 冻结 CLIP、文本感知视觉特征提取、ClearCLIP $X_{\text{attn}}$、余弦相似度 Softmax 选择、零样本泛化 | 2025 |
| [π₀](vla/foundation/pi0_2024.md) | Flow Matching VLA、VLM 骨架 + Action Expert、跨构型预训练、预训练/后训练范式 | 2024 |
| [π₀.₅](vla/foundation/pi05_2025.md) | 异构多源数据协同训练、分层推理（子任务预测 + 低层控制）、开放世界泛化 | 2025 |
| [SF](vla/foundation/SF_2025.md) | 隐式空间表征对齐（VGGT）、中间层视觉 embedding 监督、推理零开销、3.8× 训练加速、LIBERO 98.5% | 2025 |
| [SpatialVLA](vla/foundation/SpatialVLA_2025.md) | 3D 空间感知、Ego3D 位置编码、自适应高斯动作网格、3 token/step、20 Hz | 2025 |
| [SPR](vla/foundation/SPR_2026.md) | 进度感知空间子目标规划、See-Plan-Rewind 闭环、自主错误恢复、LIBERO 91.8%、LIBERO-Plus OOD ↓18.8% | 2026 |
| [TCoT](vla/foundation/TCoT_2026.md) | 全局/局部轨迹思维链、GLSR 失败检测与策略切换恢复、跨任务知识共享、LIBERO 83.3%（Multi）、真实世界 +28% | 2026 |
| [TGM-VLA](vla/foundation/TGM_VLA_2026.md) | 关键帧采样优化、颜色反转投影、跨任务/任务内 Mixup、RLBench 90.5%、COLOSSEUM 68.8% | 2026 |
| [UniVLA](vla/foundation/UniVLA_2025.md) | 任务中心潜在动作解耦（VQ-VAE + DINOv2 + 语言引导两阶段分离）、跨具身无标注视频预训练、1/20 算力超越 OpenVLA、LIBERO 95.2%、真实世界 81.7% | 2025 |
| [VP-VLA](vla/foundation/VP_VLA_2026.md) | 双系统架构、视觉提示接口（十字准星+边框）、事件驱动任务分解、视觉接地辅助损失、RoboCasa +5%、SimplerEnv +8.3% | 2026 |

### 高效推理

VLA 模型推理加速，包括 Token 缓存/剪枝、动态计算、量化等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [BitVLA](vla/efficient/BitVLA_2025.md) | 1-bit 量化、蒸馏感知训练、三值化 VLA、LIBERO 94.8%、显存 1.4GB | 2025 |
| [EfficientVLA](vla/efficient/EfficientVLA_2025.md) | LLM 层剪枝、任务感知 Token 选择、扩散步缓存、Training-Free、1.93× 加速 | 2025 |
| [HeiSD](vla/efficient/HeiSD_2026.md) | 混合推测解码（Drafter + Retrieval SD）、运动学融合指标、Verify-Skip、序列级宽松接受、2.45× 加速 | 2026 |
| [LAC](vla/efficient/LAC_2026.md) | 可学习自适应 Token 缓存、光流运动先验、Gumbel-Softmax 端到端优化、1.76× 加速 | 2026 |
| [PD-VLA](vla/efficient/PD_VLA_2025.md) | Jacobi 并行解码、Action Chunking、Training-Free、Modification-Free、2.52× 加速 | 2025 |
| [SD-VLA](vla/efficient/SD_VLA_2026.md) | 静态-动态 Token 解耦、多级缓存层次、可学习重缓存门、长时程建模、2.26× 加速 | 2026 |
| [RLRC](vla/efficient/RLRC_2025.md) | 结构化剪枝 + SFT/RL 恢复 + 4-bit 量化、90% 剪枝率、8× 显存压缩、2.3× 加速 | 2025 |
| [RTC](vla/efficient/RTC_2025.md) | 异步动作块修复执行、ΠGDM 引导 + 软掩码、Training-Free、π₀.₅ 快 20%、300ms+ 延迟鲁棒 | 2025 |
| [VLA-Cache](vla/efficient/VLA_Cache_2025.md) | 训练无关跨帧 Token 缓存、注意力驱动任务相关性过滤、层自适应复用策略、1.7× 加速 | 2025 |
| [VLA-Pruner](vla/efficient/VLA_Pruner_2025.md) | 双层 Token 剪枝（语义级 + 动作级注意力）、时序平滑估计、mRMR 双层选择、Training-Free、1.8× 加速 | 2025 |

### 推理增强

VLA 推理阶段的 training-free 增强方法，无需修改模型权重即可提升性能。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [UAOR](vla/inference/UAOR_2026.md) | Action Entropy、观测重注入、FFN-as-Memory、Training-Free、Plug-and-Play | 2026 |

### RL 后训练

用强化学习微调或改进机器人策略，包括 VLA + RL 自改进、Flow/Diffusion Policy + RL 等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [ARM](vla/rl/ARM_2026.md) | Tri-state 优势标注、MIMO Transformer、双头（区间分类 + 完成）、长度自适应 AW-BC、叠毛巾 99.4% | 2026 |
| [ConRFT](vla/rl/ConRFT_2025.md) | 一致性策略、Cal-QL + BC 离线训练、HIL 在线 RL、真实世界 96.3% 成功率 | 2025 |
| [DiffRL Data](vla/rl/DiffRL_Data_2025.md) | 扩散策略 + PPO 数据生成、BC Warm-Start、低方差轨迹、LIBERO-130 | 2025 |
| [FPO++](vla/rl/FPO_2026.md) | CFM 损失差值代理似然比、逐样本裁剪、非对称信任域 ASPO、sim-to-real | 2026 |
| [GigaBrain-0.5M*](vla/rl/GigaBrain_2026.md) | 世界模型 RL、RAMP、优势+未来状态条件化、Wan2.2、HILR 迭代训练 | 2026 |
| [GRAPE](vla/rl/GRAPE_2025.md) | 轨迹级 DPO（TPO）、VLM 代价函数自动生成、多元对齐目标、plug-and-play | 2025 |
| [GR-RL](vla/rl/GR_RL_2025.md) | 数据过滤 + 形态对称增强 + 隐空间在线 RL，通才 VLA 特化为精密操作专家 | 2025 |
| [LRM](vla/rl/LRM_2026.md) | 三维度帧级在线奖励（时序对比/绝对进度/任务完成）、Qwen3-VL-8B LoRA、24 源数据、零样本 PPO | 2026 |
| [MoRE](vla/rl/MoRE_2025.md) | Mixture of LoRA Experts、自回归 Q-learning、混合质量数据、四足多任务 VLA | 2025 |
| [π₀.₆*](vla/rl/pi06star_2025.md) | RECAP 优势条件化离线 RL、分布式价值函数、VLA 吞吐量翻倍 | 2025 |
| [π-StepNFT](vla/rl/pi_StepNFT_2026.md) | SDE 探索、逐步监督、对比排序损失、无 Critic 无似然在线 RL | 2026 |
| [πRL](vla/rl/piRL_2025.md) | Flow-Noise 可学习噪声联合似然、Flow-SDE ODE→SDE 两层 MDP、PPO 微调 π₀/π₀.₅、LIBERO 97.6%/98.3% | 2025 |
| [PLD](vla/rl/PLD_2026.md) | 残差 RL 专家、基础策略探针、混合轨迹蒸馏、VLA 自改进、LIBERO 99% | 2026 |
| [PTR](vla/rl/PTR_2026.md) | Posterior-Transition Reweighting、无奖励 identification 评分、保守权重裁剪、跨构型选择性迁移、Being-H0.5 | 2026 |
| [ReWiND](vla/rl/ReWiND_2025.md) | 语言条件化奖励、Video Rewind、进度预测、Open-X 多样化、零演示泛化 | 2025 |
| [RISE](vla/rl/RISE_2026.md) | 组合式世界模型、想象空间 RL、VLA 自改进 | 2026 |
| [Robo-Dopamine](vla/rl/RoboDopamine_2025.md) | 通用过程奖励模型、Hop-based 进度归一化、多视角融合、策略不变奖励塑形、One-shot 适配 | 2025 |
| [ROBOMETER](vla/rl/ROBOMETER_2026.md) | 通用奖励模型、帧级进度 + 轨迹偏好比较、失败数据利用、21 种具身泛化 | 2026 |
| [RoboReward](vla/rl/RoboReward_2026.md) | 通用奖励模型、反事实重标注、时序裁剪、RoboRewardBench、Episode 级离散进度奖励 | 2026 |
| [RL-Co](vla/rl/RL_Co_2026.md) | Sim-Real RL Co-Training、SFT 正则防遗忘、OpenVLA / $\pi_{0.5}$ 双验证 | 2026 |
| [RLinf](vla/rl/RLinf_2025.md) | M2Flow 宏-微流变换、弹性流水线、上下文切换、RL 训练系统 | 2025 |
| [RLinf-USER](vla/rl/RLinf_USER_2026.md) | 统一硬件抽象、云-边通信、全异步流水线、持久化缓冲区、真实世界在线学习 | 2026 |
| [RLinf-VLA](vla/rl/RLinf_VLA_2025.md) | Hybrid Fine-grained Pipelining、统一 VLA+RL 框架（PPO/GRPO）、LIBERO-130 达 98.11% | 2025 |
| [RL-VLA Survey](vla/rl/RL_VLA_Survey_2025.md) | 综述：RL-VLA 架构、在线/离线/测试时训练范式、sim-to-real 部署、评测基准 | 2025 |
| [RLVLA](vla/rl/RLVLA_2025.md) | PPO 优于 DPO/GRPO、共享 Actor-Critic、RL 在语义和执行维度显著优于 SFT | 2025 |
| [RPD](vla/rl/RPD_2025.md) | VLA→RL 策略蒸馏、PPO + MSE 引导、稀疏奖励加速、视角变化鲁棒、ManiSkill3 | 2025 |
| [SAC Flow](vla/rl/SAC_Flow_2026.md) | Flow 策略 × 序列模型、GRU/Transformer 重参数化、off-policy RL | 2026 |
| [SC-VLA](vla/rl/SC_VLA_2026.md) | 稀疏世界想象、残差 SAC 在线修正、内生密集奖励、Flow Matching | 2026 |
| [SimpleVLA-RL](vla/rl/SimpleVLA_RL_2025.md) | 在线 GRPO、二元结果奖励、Dynamic Sampling + Clip Higher + 高温采样、pushcut 涌现行为 | 2025 |
| [SRPO](vla/rl/SRPO_2025.md) | 自参照策略优化、世界模型隐空间 progress-wise 奖励、V-JEPA 2、GRPO 扩展 | 2025 |
| [TACO](vla/rl/TACO_2025.md) | Test-Time Scaling、Anti-Exploration、轻量 CFN 伪计数器选择 in-support 动作 | 2025 |
| [TGRPO](vla/rl/TGRPO_2025.md) | 无 Critic 在线 RL、LLM 自动生成多阶段稠密奖励、步级/轨迹级双层组相对优势 | 2025 |
| [TOPReward](vla/rl/TOPReward_2026.md) | Token 概率零样本奖励、VLM logits 进度估计、ManiRewardBench、VOC 0.947 | 2026 |
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

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [BridgeV2W](world-models/BridgeV2W_2025.md) | Embodiment Mask（URDF + 相机参数）、ControlNet 像素空间动作注入、光流运动损失、跨构型统一、视角鲁棒 | 2025 |
| [Fast-WAM](world-models/FastWAM_2026.md) | 视频协同训练 vs. 测试时未来想象受控拆解、MoT 架构（Wan2.2-5B + Action DiT）、训练-推理解耦、190 ms 延迟、RoboTwin 91.8%、LIBERO 97.6% | 2026.03 |
| [Kinema4D](world-models/Kinema4D_2026.md) | 4D Pointmap 运动学控制、DiT 联合 RGB+Pointmap 合成、Robo4D-200k、构型无关、零样本真实世界迁移、PSNR 22.50、F-Score 0.4733 | 2026 |
| [WorldVLA](world-models/WorldVLA_2025.md) | 自回归统一动作+世界模型、Chameleon 骨架、Action Attention Mask 阻断误差累积、VQ-GAN 离散 token、LIBERO 81.8% | 2025 |

---

## 🎓 Imitation Learning

从视频演示学习操作策略，包括无动作标注学习、光流策略、轨迹模仿等。

| 论文 | 关键词 | 年份 |
| --- | --- | --- |
| [EC-Flow](imitation-learning/EC_Flow_2025.md) | 具身中心光流、目标图像辅助对齐、URDF 运动学动作计算、无动作标注、DiT 扩散、遮挡 +62%、柔性 +45%、非位移 +80% | 2025 |
