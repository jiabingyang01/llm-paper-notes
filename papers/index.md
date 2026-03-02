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

### 🦾 Embodied AI — VLA 基础模型

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024) | 用 Flow Matching 替代自回归生成动作，构建首个能完成高频灵巧操作的通用 VLA 基础模型 | Flow Matching VLA、Action Expert、跨构型预训练 | 2024.10 |
| [π₀.₅](/papers/06-embodied-ai/vla/foundation/pi05_2025) | 通过异构多源数据协同训练和分层推理，首次实现端到端 VLA 在全新家庭环境中执行长时域灵巧操作 | 异构协同训练、分层推理、开放世界泛化 | 2025.04 |

### 🦾 Embodied AI — VLA / RL 后训练

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [RISE](/papers/06-embodied-ai/vla/rl/RISE_2026) | 用组合式世界模型在想象空间做 RL，让 VLA 不靠真实交互就能自我改进 | 世界模型、Imagination RL、VLA 自改进 | 2026.02 |
| [RLinf](/papers/06-embodied-ai/vla/rl/RLinf_2025) | 提出 M2Flow 宏-微流变换范式，通过弹性流水线和上下文切换实现灵活高效的大规模 RL 训练 | M2Flow、弹性流水线、RL 训练系统 | 2025.09 |
| [RLinf-USER](/papers/06-embodied-ai/vla/rl/RLinf_USER_2026) | 将机器人视为一等硬件资源，通过统一硬件抽象、云-边通信、全异步流水线构建真实世界在线策略学习系统 | 真实世界 RL、统一硬件抽象、云-边协同、异步训练 | 2026.02 |
| [RLinf-VLA](/papers/06-embodied-ai/vla/rl/RLinf_VLA_2025) | 统一高效的 VLA+RL 训练框架，三种 GPU 分配模式 + PPO/GRPO，单一模型 LIBERO-130 达 98.11% | Hybrid Pipelining、PPO/GRPO、统一 VLA+RL 框架 | 2025.10 |
| [SAC Flow](/papers/06-embodied-ai/vla/rl/SAC_Flow_2026) | 把 Flow Policy 重新理解为序列模型，用 GRU/Transformer 重参数化解决 RL 梯度不稳定问题 | Flow Policy、序列建模、SAC、off-policy RL | 2026.01 |
| [VLA-RL](/papers/06-embodied-ai/vla/rl/VLA_RL_2025) | 将机器人操作建模为多模态多轮对话，用 PPO 在线 RL 微调自回归 VLA，配合 Robotic PRM 解决稀疏奖励 | 在线 PPO、Robotic PRM、自回归 VLA + RL | 2025.05 |
| [WoVR](/papers/06-embodied-ai/vla/rl/WoVR_2026) | 通过三级幻觉控制（稳定世界模型 + 关键帧初始化 Rollout + 策略-模型协同进化），在想象空间中可靠地 RL 后训练 VLA | 世界模型 RL、幻觉感知、KIR、PACE | 2026.02 |
| [GR-RL](/papers/06-embodied-ai/vla/rl/GR_RL_2025) | 多阶段流水线（离线数据过滤 + 形态对称增强 + 隐空间在线 RL）将通才 VLA 特化为穿鞋带专家，83.3% 成功率 | 数据过滤、分布式 Critic、隐空间 RL、形态对称增强 | 2025.12 |
| [π₀.₆*](/papers/06-embodied-ai/vla/rl/pi06star_2025) | 通过 RECAP（优势条件化离线 RL）整合自主 rollout、专家干预和演示数据，VLA 吞吐量翻倍、失败率减半 | 优势条件化、离线 RL、分布式价值函数、RECAP | 2025.11 |
| [RLVLA](/papers/06-embodied-ai/vla/rl/RLVLA_2025) | 系统性实证研究 RL 对 VLA 泛化性的收益：PPO 优于 DPO/GRPO，RL 在语义和执行维度显著优于 SFT | PPO、泛化基准、共享 Actor-Critic、SFT vs RL | 2025.05 |
| [SRPO](/papers/06-embodied-ai/vla/rl/SRPO_2025) | 自参照策略优化：用模型自身成功轨迹 + 世界模型隐表征为失败轨迹提供 progress-wise 奖励，消除外部演示依赖 | 自参照、隐空间进度奖励、V-JEPA 2、GRPO 扩展 | 2025.11 |
| [TACO](/papers/06-embodied-ai/vla/rl/TACO_2025) | 将 offline RL 反探索原则应用于 VLA 推理阶段，用轻量 CFN 伪计数器选择最 in-support 的动作，无需改参数即提升成功率 | Test-Time Scaling、Anti-Exploration、Pseudo-Count | 2025.12 |
| [TGRPO](/papers/06-embodied-ai/vla/rl/TGRPO_2025) | 无 Critic 在线 RL 框架：LLM 自动生成多阶段稠密奖励 + 步级/轨迹级双层组相对优势融合微调 VLA | GRPO 扩展、双层优势、LLM 奖励设计 | 2025.06 |
| [GRAPE](/papers/06-embodied-ai/vla/rl/GRAPE_2025) | 轨迹级偏好优化（TPO）+ VLM 自动生成代价函数，plug-and-play 提升 VLA 泛化性并支持多元对齐目标 | 轨迹级 DPO、VLM 代价函数、多元对齐、偏好合成 | 2024.11 |
| [FPO++](/papers/06-embodied-ai/vla/rl/FPO_2026) | 用 CFM 损失差值近似似然比绕开 flow 策略密度计算，逐样本裁剪 + 非对称信任域实现稳定 on-policy RL 训练 | Flow Policy Gradient、CFM 代理似然比、ASPO、sim-to-real | 2026.02 |
