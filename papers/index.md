# 📚 论文索引

[LLM Paper Notes](https://llm-paper-notes.jiabingyang.cn/) 是一个开源的论文精读笔记站，聚焦大语言模型及相关领域。每篇笔记包含问题动机、前置知识、方法拆解、公式推导、实验分析和个人思考。

---

## 分类导航

| | 分类 | 覆盖方向 |
| :---: | --- | --- |
| 🏗️ | [Foundation Models](/papers/01-foundation-models/) | GPT、LLaMA、Mamba、Scaling Laws、MoE 预训练 |
| 🛡️ | [Alignment & Safety](/papers/02-alignment-and-safety/) | RLHF、DPO、RLAIF、Constitutional AI |
| 💡 | [Reasoning](/papers/03-reasoning/) | CoT、ToT、o1/o3、数学推理、Test-time Compute |
| 🖼️ | [Multimodal](/papers/04-multimodal/) | GPT-4V、LLaVA、视频理解、语音模型 |
| 🤖 | [Agents](/papers/05-agents/) | ReAct、Toolformer、WebAgent、SWE-Agent |
| 🦾 | [Embodied AI](/papers/06-embodied-ai/) | VLA、世界模型、机器人 RL、模仿学习 |
| ⚡ | [Efficiency](/papers/07-efficiency/) | GPTQ、AWQ、LoRA、Speculative Decoding |
| 🔍 | [RAG & Knowledge](/papers/08-rag-and-knowledge/) | Dense Retrieval、RAPTOR、GraphRAG |
| 📊 | [Evaluation](/papers/09-evaluation-and-benchmarks/) | MMLU、HumanEval、Arena、LLM-as-Judge |

---

## 全部论文

### 🛡️ Alignment & Safety — LLM RL 训练

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [R³L](/papers/02-alignment-and-safety/R3L_2026) | 反思-重试合成成功轨迹 + 关键点信用分配只更新分歧后缀 + 正向放大确保正信号主导，Agentic 和数学推理任务相对 GRPO 提升 5%–52% | GRPO 改进、语言引导探索、Pivotal Credit、Positive Amplification | 2026.01 |

### 🖼️ Multimodal — VLM 幻觉缓解

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [CSR](/papers/04-multimodal/vlm/hallucination/CSR_2024) | 句子级 beam search + CLIP 视觉校准奖励迭代构造自生成偏好数据 + DPO 微调，三轮迭代 10 个基准平均提升 7.62%，CHAIR$_S$ 降低 57% | 校准自奖励、CLIP Score、迭代 DPO、模态对齐、Self-Rewarding | 2024.05 |
| [DLC](/papers/04-multimodal/vlm/hallucination/DLC_2025) | 解码时用 CLIP 逐步评估候选 token 的相对视觉优势 (RVA)，相对动态历史基线自适应调整 logits，无需额外前向传播高效缓解语义漂移幻觉 | 动态 Logits 校准、CLIP 探针、相对视觉优势、自适应引导、Training-Free | 2025.06 |
| [HALC](/papers/04-multimodal/vlm/hallucination/HALC_2024) | 自适应 FOV 采样 + JSD 双向对比解码修正局部幻觉 + 视觉匹配 beam search 全局保障，无训练即插即用，CHAIR$_S$ 在 MiniGPT-4 上降低 36% | FOV 对比解码、JSD 选择、视觉匹配 Beam Search、Plug-and-Play | 2024.03 |
| [HIME](/papers/04-multimodal/vlm/hallucination/HIME_2026) | 提出 HIS 量化每层幻觉敏感度，层自适应加权投影编辑 MLP 权重，无训练/无额外参数/无推理开销平均降低 61.8% 对象幻觉 | HIS、层自适应模型编辑、零空间投影、Training-Free | 2026.02 |
| [MemVR](/papers/04-multimodal/vlm/hallucination/MemVR_2025) | 将视觉 token 通过 FFN key-value memory 机制重注入中间层，不确定性超阈值时动态触发 look-twice，POPE +7.0%、CHAIR$_I$ -15.6%，推理仅 1.04× 延迟且通用能力同步提升 | FFN Key-Value Memory、视觉回溯、不确定性触发、Training-Free、Plug-and-Play | 2025.05 |
| [SENTINEL](/papers/04-multimodal/vlm/hallucination/SENTINEL_2025) | 域内自举采样 + 检测器交叉验证构建句子级偏好数据，C-DPO 在幻觉首次出现处早期干预，Object HalBench 幻觉率降低 92% 且通用能力不降反升 | 句子级早期干预、域内偏好学习、C-DPO、交叉验证 | 2025.07 |
| [VisFlow](/papers/04-multimodal/vlm/hallucination/VisFlow_2025) | Token 级别增强视觉显著 token + Head 级别抑制系统提示头和文本跟随头，双层注意力干预无训练缓解幻觉，LLaVA-1.5 CHAIR$_S$ 降低 40%、POPE Adversarial F1 +10.8 pp | 双层注意力干预、Visual Sink/Salient Token、Head 分类抑制、Training-Free | 2025.06 |

### 🦾 Embodied AI — VLA 基础模型

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [BridgeVLA](/papers/06-embodied-ai/vla/foundation/BridgeVLA_2025) | 3D 点云正交投影为多视图 2D 图像 + 预测 2D 热力图对齐输入-输出，热力图预训练赋予 VLM 空间定位能力，RLBench 88.2%、3 条轨迹达 95.4% | 输入-输出对齐、2D 热力图、正交投影、样本效率 | 2025.06 |
| [DAM-VLA](/papers/06-embodied-ai/vla/foundation/DAM_VLA_2026) | VLM 推理驱动动作路由选择手臂/夹爪专用扩散模型（class token 全局 + register token 局部）+ 双尺度加权协调训练，SIMPLER 平均 78-83%、真实世界 86.8% | 动作路由、双扩散头、class/register token、双尺度加权 | 2026.03 |
| [DreamVLA](/papers/06-embodied-ai/vla/foundation/DreamVLA_2025) | 感知-预测-动作闭环：预测三类综合世界知识（动态区域/深度/语义）+ block-wise 结构化注意力防止跨类泄露 + DiT 动作头，CALVIN ABC-D 4.44 SOTA、LIBERO 92.6%、真实世界 76.7% | 世界知识预测、结构化注意力、DiT 动作头、CoTracker/Depth Anything/DINOv2 | 2025.07 |
| [Dexbotic](/papers/06-embodied-ai/vla/foundation/Dexbotic_2025) | 开源 VLA 工具箱：统一模块化框架（VLM + Action Expert）+ 基于 Qwen2.5 的更强预训练模型 + 实验驱动 Exp 脚本开发，SimplerEnv 最高 +46.2%、CALVIN +0.81 | VLA Toolbox、统一框架、DexboticVLM、实验驱动开发 | 2025.10 |
| [FutureVLA](/papers/06-embodied-ai/vla/foundation/FutureVLA_2026) | 联合视觉-运动预测建模（JVPM）：3D-VAE 编码连续 17 帧 + 双流解耦监督 + 门控交叉注意力条件化交互，潜在嵌入对齐迁移时序先验，推理零开销，SimplerEnv 80.1%、真实机器人超 π₀ 达 26.7% | 联合视觉运动预测、双流解耦、门控交叉注意力、潜在对齐 | 2026.03 |
| [GR-3](/papers/06-embodied-ai/vla/foundation/GR3_2025) | 4B VLA（Qwen2.5-VL + Action DiT），机器人轨迹 + VL 数据协同训练实现 OOD 指令零样本泛化，VR 人类轨迹 10-shot 适配新物体（57.8%→86.7%），Task Status 辅助监督 + DiT RMSNorm 强化指令跟随，全面超越 π₀ | MoT 架构、VL 协同训练、人类轨迹少样本适配、Task Status、ByteMini 双臂移动 | 2025.07 |
| [MoH](/papers/06-embodied-ai/vla/foundation/MoH_2025) | 多 horizon 动作块在共享 Action Transformer 中并行处理 + 轻量门控融合（2k 参数）+ 跨 horizon 共识动态推理，plug-and-play 适用于 flow/regression 策略，$\pi_{0.5}$+MoH LIBERO 99% SOTA | 多 Horizon 融合、门控融合、动态推理、Action Chunking | 2025.11 |
| [MemoryVLA](/papers/06-embodied-ai/vla/foundation/MemoryVLA_2025) | 借鉴认知科学双记忆系统设计感知-认知记忆库（PCMB），同时存储低层视觉细节和高层语义，跨注意力检索 + 门控融合 + 合并压缩建模长时域依赖，SimplerEnv-Bridge +14.6、LIBERO 96.5%、真实世界时序任务 +26 | 感知-认知记忆、时序建模、扩散策略、长时域操作 | 2025.08 |
| [OptimusVLA](/papers/06-embodied-ai/vla/foundation/OptimusVLA_2026) | 双记忆增强 VLA：GPM 用检索到的任务级先验替代高斯噪声缩短 flow 生成路径 + LCM 用 Mamba 建模动作历史注入时序一致性，LIBERO 98.6%、真实世界 2.9× 推理加速 | 双记忆、任务级先验检索、时序一致性、自适应 NFE | 2026.02 |
| [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024) | 用 Flow Matching 替代自回归生成动作，构建首个能完成高频灵巧操作的通用 VLA 基础模型 | Flow Matching VLA、Action Expert、跨构型预训练 | 2024.10 |
| [π₀.₅](/papers/06-embodied-ai/vla/foundation/pi05_2025) | 通过异构多源数据协同训练和分层推理，首次实现端到端 VLA 在全新家庭环境中执行长时域灵巧操作 | 异构协同训练、分层推理、开放世界泛化 | 2025.04 |
| [SF](/papers/06-embodied-ai/vla/foundation/SF_2025) | 将 VLA 中间层视觉 embedding 与 VGGT 3D 表征做余弦对齐，无需 3D 输入、推理零开销，LIBERO 98.5% 超越所有 2D/3D VLA，训练 3.8× 加速、数据 5.9× 高效 | 隐式空间对齐、VGGT、表征监督、训练/数据效率 | 2025.10 |
| [TGM-VLA](/papers/06-embodied-ai/vla/foundation/TGM_VLA_2026) | 优化关键帧采样（存储 -85%、训练 5× 加速）+ 颜色反转投影分支 + 任务引导点云 Mixup，RLBench 90.5% SOTA、COLOSSEUM 68.8% | 关键帧采样优化、颜色反转、跨任务/任务内 Mixup、3D VLA | 2026.02 |

### 🦾 Embodied AI — VLA 高效推理

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) | 结构化 training-free 加速：LLM 层剪枝 + 任务感知视觉 token 选择 + 扩散步缓存，三维度协同消除冗余，FLOPs 降至 28.9%、1.93× 加速 | LLM 层剪枝、任务感知 Token 选择、扩散步缓存、Training-Free、1.93× 加速 | 2025.06 |
| [LAC](/papers/06-embodied-ai/vla/efficient/LAC_2026) | 可学习自适应 Token 缓存，光流运动先验 + Gumbel-Softmax 端到端优化，1.76× 加速 | 可学习 Token 缓存、光流运动先验、1.76× 加速 | 2026.01 |
| [SD-VLA](/papers/06-embodied-ai/vla/efficient/SD_VLA_2026) | 静态-动态 Token 解耦 + 多级缓存层次 + 可学习重缓存门，长时程 VLA 2.26× 加速 | 静态-动态解耦、多级缓存、2.26× 加速 | 2026.02 |
| [RLRC](/papers/06-embodied-ai/vla/efficient/RLRC_2025) | 三阶段 VLA 压缩流水线（结构化剪枝 + SFT/RL 恢复 + 量化），8× 显存压缩、2.3× 加速 | 结构化剪枝、RL 恢复、量化、8× 压缩 | 2025.06 |
| [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) | 训练无关跨帧 Token 缓存 + 注意力驱动任务相关性过滤，1.7× 加速 | 跨帧 Token 缓存、注意力过滤、1.7× 加速 | 2025.02 |
| [VLA-Pruner](/papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025) | 双层重要性准则（语义级 prefill + 动作级 decode 注意力时序平滑）+ mRMR 双层选择策略，50% 剪枝率反超原模型，87.5% 剪枝率保持 88.9% 准确率 | 双层 Token 剪枝、时序平滑、mRMR 选择、Training-Free、1.8× 加速 | 2025.11 |

### 🦾 Embodied AI — VLA 推理增强

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [UAOR](/papers/06-embodied-ai/vla/inference/UAOR_2026) | 用 Action Entropy 检测高不确定性层，通过注意力检索将观测特征重注入 FFN，无训练即插即用一致提升多种 VLA | Action Entropy、观测重注入、FFN-as-Memory、Training-Free | 2026.02 |

### 🦾 Embodied AI — VLA / RL 后训练

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [ConRFT](/papers/06-embodied-ai/vla/rl/ConRFT_2025) | 一致性策略统一离线 BC+Q-learning 与在线 HIL RL，8 个真实任务 45–90 分钟达 96.3%，比 SFT 提升 144% | 一致性策略、Cal-QL、离线-在线统一目标、HIL | 2025.02 |
| [DiffRL Data](/papers/06-embodied-ai/vla/rl/DiffRL_Data_2025) | 轻量扩散策略 + PPO 生成高质量低方差轨迹训练 VLA，纯合成数据超越人类演示 +5.3% | 扩散 RL、数据生成、BC Warm-Start、LIBERO-130 | 2025.09 |
| [FPO++](/papers/06-embodied-ai/vla/rl/FPO_2026) | 用 CFM 损失差值近似似然比绕开 flow 策略密度计算，逐样本裁剪 + 非对称信任域实现稳定 on-policy RL 训练 | Flow Policy Gradient、CFM 代理似然比、ASPO、sim-to-real | 2026.02 |
| [GigaBrain-0.5M*](/papers/06-embodied-ai/vla/rl/GigaBrain_2026) | 用视频世界模型联合预测未来状态+价值条件化 VLA 策略（RAMP），理论证明 RECAP 是其退化特例，比 RECAP 提升约 30% | 世界模型 RL、RAMP、优势条件化、未来状态条件化 | 2026.02 |
| [GRAPE](/papers/06-embodied-ai/vla/rl/GRAPE_2025) | 轨迹级偏好优化（TPO）+ VLM 自动生成代价函数，plug-and-play 提升 VLA 泛化性并支持多元对齐目标 | 轨迹级 DPO、VLM 代价函数、多元对齐、偏好合成 | 2024.11 |
| [GR-RL](/papers/06-embodied-ai/vla/rl/GR_RL_2025) | 多阶段流水线（离线数据过滤 + 形态对称增强 + 隐空间在线 RL）将通才 VLA 特化为穿鞋带专家，83.3% 成功率 | 数据过滤、分布式 Critic、隐空间 RL、形态对称增强 | 2025.12 |
| [LRM](/papers/06-embodied-ai/vla/rl/LRM_2026) | 将 Qwen3-VL-8B 适配为三维度帧级在线奖励引擎（时序对比/绝对进度/任务完成），24 源数据训练后零样本驱动 PPO，30 轮迭代超越 RoboReward 和 ROBOMETER | 帧级在线奖励、三维度奖励分解、VLM-as-Reward、PPO 闭环 | 2026.03 |
| [MoRE](/papers/06-embodied-ai/vla/rl/MoRE_2025) | Fuyu 8B 上构建 Mixture of LoRA Experts + 自回归 Q-learning 离线 RL，从混合质量数据学习，四足 6 任务平均成功率 44%→60% | MoE-LoRA、自回归 Q-learning、混合质量数据、四足 VLA | 2025.03 |
| [π₀.₆*](/papers/06-embodied-ai/vla/rl/pi06star_2025) | 通过 RECAP（优势条件化离线 RL）整合自主 rollout、专家干预和演示数据，VLA 吞吐量翻倍、失败率减半 | 优势条件化、离线 RL、分布式价值函数、RECAP | 2025.11 |
| [π-StepNFT](/papers/06-embodied-ai/vla/rl/pi_StepNFT_2026) | 无 Critic 无似然在线 RL：SDE 拓宽探索 + 逐步监督 + 对比排序损失，ManiSkill OOD 超 PPO 11.1% | SDE 探索、逐步监督、对比排序、无 Critic | 2026.03 |
| [PLD](/papers/06-embodied-ai/vla/rl/PLD_2026) | 冻结 VLA 主干训练轻量残差 RL 专家探索失败区域，基础策略探针 + 混合轨迹蒸馏实现 VLA 自改进，LIBERO 达 99% 成功率 | 残差 RL、基础策略探针、混合数据蒸馏、VLA 自改进 | 2026.01 |
| [ReWiND](/papers/06-embodied-ai/vla/rl/ReWiND_2025) | 从少量演示训练语言条件化奖励模型（Video Rewind + Open-X + 仅首帧位置编码），无需新演示即可语言引导 RL 学新任务，仿真超基线 2×、真实世界提升 5× | 语言条件化奖励、Video Rewind、进度预测、零演示泛化 | 2025.05 |
| [RISE](/papers/06-embodied-ai/vla/rl/RISE_2026) | 用组合式世界模型在想象空间做 RL，让 VLA 不靠真实交互就能自我改进 | 世界模型、Imagination RL、VLA 自改进 | 2026.02 |
| [Robo-Dopamine](/papers/06-embodied-ai/vla/rl/RoboDopamine_2025) | 35M 多视角数据训练步感知 GRM + Hop-based 进度归一化 + 多视角融合 + 策略不变奖励塑形，One-shot 适配新任务 150 次交互达 95% 成功率 | 通用过程奖励模型、Hop-based 进度归一化、多视角融合、策略不变奖励塑形 | 2025.12 |
| [ROBOMETER](/papers/06-embodied-ai/vla/rl/ROBOMETER_2026) | 帧级进度预测 + 轨迹间偏好比较双目标训练通用机器人奖励模型，有效利用失败数据，下游 RL 策略成功率提升 2.4–4.5× | 通用奖励模型、轨迹偏好比较、失败数据利用 | 2026.03 |
| [RoboReward](/papers/06-embodied-ai/vla/rl/RoboReward_2026) | 反事实重标注 + 时序裁剪合成负样本，微调 Qwen3-VL 为 episode 级离散进度奖励模型（1-5分），22 个 VLM 排名第一，真实 RL 大幅超越 Gemini Robotics-ER 1.5 | 通用奖励模型、反事实重标注、时序裁剪、RoboRewardBench | 2026.01 |
| [RL-Co](/papers/06-embodied-ai/vla/rl/RL_Co_2026) | 两阶段 sim-real 协同训练：SFT 混合初始化 + 仿真 RL 微调并加真实数据 SFT 正则防遗忘，OpenVLA +24%、$\pi_{0.5}$ +20% | Sim-Real Co-Training、RL + SFT 正则、数据效率、通用框架 | 2026.02 |
| [RLinf](/papers/06-embodied-ai/vla/rl/RLinf_2025) | 提出 M2Flow 宏-微流变换范式，通过弹性流水线和上下文切换实现灵活高效的大规模 RL 训练 | M2Flow、弹性流水线、RL 训练系统 | 2025.09 |
| [RLinf-USER](/papers/06-embodied-ai/vla/rl/RLinf_USER_2026) | 将机器人视为一等硬件资源，通过统一硬件抽象、云-边通信、全异步流水线构建真实世界在线策略学习系统 | 真实世界 RL、统一硬件抽象、云-边协同、异步训练 | 2026.02 |
| [RLinf-VLA](/papers/06-embodied-ai/vla/rl/RLinf_VLA_2025) | 统一高效的 VLA+RL 训练框架，三种 GPU 分配模式 + PPO/GRPO，单一模型 LIBERO-130 达 98.11% | Hybrid Pipelining、PPO/GRPO、统一 VLA+RL 框架 | 2025.10 |
| [RL-VLA Survey](/papers/06-embodied-ai/vla/rl/RL_VLA_Survey_2025) | 首篇系统综述 RL 后训练 VLA 的全景图：架构（动作/奖励/世界模型）、在线/离线/测试时训练范式、sim-to-real 部署与评测 | 综述、RL-VLA 分类体系、训练范式、部署 | 2025.12 |
| [RLVLA](/papers/06-embodied-ai/vla/rl/RLVLA_2025) | 系统性实证研究 RL 对 VLA 泛化性的收益：PPO 优于 DPO/GRPO，RL 在语义和执行维度显著优于 SFT | PPO、泛化基准、共享 Actor-Critic、SFT vs RL | 2025.05 |
| [RPD](/papers/06-embodied-ai/vla/rl/RPD_2025) | PPO + MSE 蒸馏项将 VLA 通才知识蒸馏为紧凑 RL 专家策略，稀疏奖励和视角变化下大幅优于 vanilla PPO | Policy Distillation、PPO + BC、VLA→RL 专家、ManiSkill3 | 2025.03 |
| [SAC Flow](/papers/06-embodied-ai/vla/rl/SAC_Flow_2026) | 把 Flow Policy 重新理解为序列模型，用 GRU/Transformer 重参数化解决 RL 梯度不稳定问题 | Flow Policy、序列建模、SAC、off-policy RL | 2026.01 |
| [SC-VLA](/papers/06-embodied-ai/vla/rl/SC_VLA_2026) | 稀疏世界想象（预测进度 + 状态增量）+ 残差 SAC 在线修正，内生密集奖励无需外部奖励模型 | 稀疏世界想象、残差 RL、内生奖励、Flow Matching | 2026.02 |
| [SimpleVLA-RL](/papers/06-embodied-ai/vla/rl/SimpleVLA_RL_2025) | 基于 veRL 的端到端在线 RL 框架：二元结果奖励 + GRPO + 三种探索增强，LIBERO 达 99.1%，1 条演示 RL 超越全量 SFT，发现 pushcut 涌现行为 | 在线 GRPO、Dynamic Sampling、探索增强、pushcut | 2025.05 |
| [SRPO](/papers/06-embodied-ai/vla/rl/SRPO_2025) | 自参照策略优化：用模型自身成功轨迹 + 世界模型隐表征为失败轨迹提供 progress-wise 奖励，消除外部演示依赖 | 自参照、隐空间进度奖励、V-JEPA 2、GRPO 扩展 | 2025.11 |
| [TACO](/papers/06-embodied-ai/vla/rl/TACO_2025) | 将 offline RL 反探索原则应用于 VLA 推理阶段，用轻量 CFN 伪计数器选择最 in-support 的动作，无需改参数即提升成功率 | Test-Time Scaling、Anti-Exploration、Pseudo-Count | 2025.12 |
| [TGRPO](/papers/06-embodied-ai/vla/rl/TGRPO_2025) | 无 Critic 在线 RL 框架：LLM 自动生成多阶段稠密奖励 + 步级/轨迹级双层组相对优势融合微调 VLA | GRPO 扩展、双层优势、LLM 奖励设计 | 2025.06 |
| [TwinRL](/papers/06-embodied-ai/vla/rl/TwinRL_2026) | 用高保真数字孪生作为探索放大器和引导器，三阶段流程（探索空间扩展 + 孪生在线 RL + sim-to-real 引导）四任务平均 20 分钟逼近 100% 成功率 | 数字孪生、探索空间扩展、Sim-to-Real 引导、HiL | 2026.02 |
| [VLAC](/papers/06-embodied-ai/vla/rl/VLAC_2025) | 基于 InternVL 构建统一 Actor-Critic 模型，pairwise progress delta 提供通用稠密奖励，配合异步真实世界 RL 和分级人机协作，200 episode 内成功率 30%→90% | 统一 Actor-Critic、Pairwise Progress、真实世界 RL、Human-in-the-Loop | 2025.09 |
| [VLA-RFT](/papers/06-embodied-ai/vla/rl/VLA_RFT_2025) | 数据驱动视频世界模型充当模拟器，verified reward（MAE+LPIPS）+ GRPO 端到端微调 VLA，400 步超越 150K 步 SFT | 视频世界模型、Verified Reward、SDE-Policy、GRPO | 2025.10 |
| [VLA-RL](/papers/06-embodied-ai/vla/rl/VLA_RL_2025) | 将机器人操作建模为多模态多轮对话，用 PPO 在线 RL 微调自回归 VLA，配合 Robotic PRM 解决稀疏奖励 | 在线 PPO、Robotic PRM、自回归 VLA + RL | 2025.05 |
| [WMPO](/papers/06-embodied-ai/vla/rl/WMPO_2025) | 在隐空间世界模型中做 imagination rollout + PPO，无需在线交互即可 RL 后训练 VLA | 隐空间世界模型、Imagination RL、PPO、离线后训练 | 2025.12 |
| [World-VLA-Loop](/papers/06-embodied-ai/vla/rl/World_VLA_Loop_2026) | 视频世界模型与 VLA 策略闭环联合优化：SANS 近成功数据 + 内嵌奖励预测头 + 迭代 RL 后训练，两轮迭代真实世界成功率 13.3%→50.0% | 闭环联合优化、SANS 数据集、奖励预测头、迭代 RL | 2026.02 |
| [WoVR](/papers/06-embodied-ai/vla/rl/WoVR_2026) | 通过三级幻觉控制（稳定世界模型 + 关键帧初始化 Rollout + 策略-模型协同进化），在想象空间中可靠地 RL 后训练 VLA | 世界模型 RL、幻觉感知、KIR、PACE | 2026.02 |

### 🦾 Embodied AI — World Models

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [BridgeV2W](/papers/06-embodied-ai/world-models/BridgeV2W_2025) | 将坐标空间动作通过 URDF + 相机参数渲染为像素对齐 Embodiment Mask，经 ControlNet 注入预训练视频生成模型，辅以光流运动损失，统一解决动作-视频鸿沟、视角敏感性和跨构型架构不统一三大问题 | Embodiment Mask、ControlNet、光流运动损失、跨构型统一 | 2025 |
| [WorldVLA](/papers/06-embodied-ai/world-models/WorldVLA_2025) | 基于 Chameleon 将 VLA 动作模型与世界模型统一到单个自回归框架，共享权重混合训练实现双向增强，提出 Action Attention Mask 阻断 Action Chunking 误差累积 | 自回归统一模型、Action Attention Mask、Chameleon、双向增强 | 2025 |

### 🦾 Embodied AI — Imitation Learning

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [EC-Flow](/papers/06-embodied-ai/imitation-learning/EC_Flow_2025) | 将光流预测从物体中心转为具身中心（预测机器人上采样点轨迹），配合目标图像辅助对齐和 URDF 感知运动学动作计算，仅用 5 条无动作标注 RGB 视频学习操作策略，在遮挡（+62%）、柔性物体（+45%）和非位移操作（+80%）场景大幅超越物体中心方法 | 具身中心光流、目标图像对齐、URDF 运动学、无动作标注、DiT 扩散 | 2025.07 |
