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
| [SAC Flow](/papers/06-embodied-ai/vla/rl/SAC_Flow_2026) | 把 Flow Policy 重新理解为序列模型，用 GRU/Transformer 重参数化解决 RL 梯度不稳定问题 | Flow Policy、序列建模、SAC、off-policy RL | 2026.01 |
| [VLA-RL](/papers/06-embodied-ai/vla/rl/VLA_RL_2025) | 将机器人操作建模为多模态多轮对话，用 PPO 在线 RL 微调自回归 VLA，配合 Robotic PRM 解决稀疏奖励 | 在线 PPO、Robotic PRM、自回归 VLA + RL | 2025.05 |
| [WoVR](/papers/06-embodied-ai/vla/rl/WoVR_2026) | 通过三级幻觉控制（稳定世界模型 + 关键帧初始化 Rollout + 策略-模型协同进化），在想象空间中可靠地 RL 后训练 VLA | 世界模型 RL、幻觉感知、KIR、PACE | 2026.02 |
