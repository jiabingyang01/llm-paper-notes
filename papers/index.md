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

### 🖼️ Multimodal — VLM 幻觉缓解

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [HIME](/papers/04-multimodal/vlm/hallucination/HIME_2026) | 提出 HIS 量化每层幻觉敏感度，层自适应加权投影编辑 MLP 权重，无训练/无额外参数/无推理开销平均降低 61.8% 对象幻觉 | HIS、层自适应模型编辑、零空间投影、Training-Free | 2026.02 |
| [SENTINEL](/papers/04-multimodal/vlm/hallucination/SENTINEL_2025) | 域内自举采样 + 检测器交叉验证构建句子级偏好数据，C-DPO 在幻觉首次出现处早期干预，Object HalBench 幻觉率降低 92% 且通用能力不降反升 | 句子级早期干预、域内偏好学习、C-DPO、交叉验证 | 2025.07 |

### 🦾 Embodied AI — VLA 基础模型

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024) | 用 Flow Matching 替代自回归生成动作，构建首个能完成高频灵巧操作的通用 VLA 基础模型 | Flow Matching VLA、Action Expert、跨构型预训练 | 2024.10 |
| [π₀.₅](/papers/06-embodied-ai/vla/foundation/pi05_2025) | 通过异构多源数据协同训练和分层推理，首次实现端到端 VLA 在全新家庭环境中执行长时域灵巧操作 | 异构协同训练、分层推理、开放世界泛化 | 2025.04 |

### 🦾 Embodied AI — VLA 高效推理

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [LAC](/papers/06-embodied-ai/vla/efficient/LAC_2026) | 可学习自适应 Token 缓存，光流运动先验 + Gumbel-Softmax 端到端优化，1.76× 加速 | 可学习 Token 缓存、光流运动先验、1.76× 加速 | 2026.01 |
| [SD-VLA](/papers/06-embodied-ai/vla/efficient/SD_VLA_2026) | 静态-动态 Token 解耦 + 多级缓存层次 + 可学习重缓存门，长时程 VLA 2.26× 加速 | 静态-动态解耦、多级缓存、2.26× 加速 | 2026.02 |
| [RLRC](/papers/06-embodied-ai/vla/efficient/RLRC_2025) | 三阶段 VLA 压缩流水线（结构化剪枝 + SFT/RL 恢复 + 量化），8× 显存压缩、2.3× 加速 | 结构化剪枝、RL 恢复、量化、8× 压缩 | 2025.06 |
| [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) | 训练无关跨帧 Token 缓存 + 注意力驱动任务相关性过滤，1.7× 加速 | 跨帧 Token 缓存、注意力过滤、1.7× 加速 | 2025.02 |

### 🦾 Embodied AI — VLA / RL 后训练

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [ConRFT](/papers/06-embodied-ai/vla/rl/ConRFT_2025) | 一致性策略统一离线 BC+Q-learning 与在线 HIL RL，8 个真实任务 45–90 分钟达 96.3%，比 SFT 提升 144% | 一致性策略、Cal-QL、离线-在线统一目标、HIL | 2025.02 |
| [DiffRL Data](/papers/06-embodied-ai/vla/rl/DiffRL_Data_2025) | 轻量扩散策略 + PPO 生成高质量低方差轨迹训练 VLA，纯合成数据超越人类演示 +5.3% | 扩散 RL、数据生成、BC Warm-Start、LIBERO-130 | 2025.09 |
| [FPO++](/papers/06-embodied-ai/vla/rl/FPO_2026) | 用 CFM 损失差值近似似然比绕开 flow 策略密度计算，逐样本裁剪 + 非对称信任域实现稳定 on-policy RL 训练 | Flow Policy Gradient、CFM 代理似然比、ASPO、sim-to-real | 2026.02 |
| [GigaBrain-0.5M*](/papers/06-embodied-ai/vla/rl/GigaBrain_2026) | 用视频世界模型联合预测未来状态+价值条件化 VLA 策略（RAMP），理论证明 RECAP 是其退化特例，比 RECAP 提升约 30% | 世界模型 RL、RAMP、优势条件化、未来状态条件化 | 2026.02 |
| [GRAPE](/papers/06-embodied-ai/vla/rl/GRAPE_2025) | 轨迹级偏好优化（TPO）+ VLM 自动生成代价函数，plug-and-play 提升 VLA 泛化性并支持多元对齐目标 | 轨迹级 DPO、VLM 代价函数、多元对齐、偏好合成 | 2024.11 |
| [GR-RL](/papers/06-embodied-ai/vla/rl/GR_RL_2025) | 多阶段流水线（离线数据过滤 + 形态对称增强 + 隐空间在线 RL）将通才 VLA 特化为穿鞋带专家，83.3% 成功率 | 数据过滤、分布式 Critic、隐空间 RL、形态对称增强 | 2025.12 |
| [MoRE](/papers/06-embodied-ai/vla/rl/MoRE_2025) | Fuyu 8B 上构建 Mixture of LoRA Experts + 自回归 Q-learning 离线 RL，从混合质量数据学习，四足 6 任务平均成功率 44%→60% | MoE-LoRA、自回归 Q-learning、混合质量数据、四足 VLA | 2025.03 |
| [π₀.₆*](/papers/06-embodied-ai/vla/rl/pi06star_2025) | 通过 RECAP（优势条件化离线 RL）整合自主 rollout、专家干预和演示数据，VLA 吞吐量翻倍、失败率减半 | 优势条件化、离线 RL、分布式价值函数、RECAP | 2025.11 |
| [PLD](/papers/06-embodied-ai/vla/rl/PLD_2026) | 冻结 VLA 主干训练轻量残差 RL 专家探索失败区域，基础策略探针 + 混合轨迹蒸馏实现 VLA 自改进，LIBERO 达 99% 成功率 | 残差 RL、基础策略探针、混合数据蒸馏、VLA 自改进 | 2026.01 |
| [RISE](/papers/06-embodied-ai/vla/rl/RISE_2026) | 用组合式世界模型在想象空间做 RL，让 VLA 不靠真实交互就能自我改进 | 世界模型、Imagination RL、VLA 自改进 | 2026.02 |
| [RL-Co](/papers/06-embodied-ai/vla/rl/RL_Co_2026) | 两阶段 sim-real 协同训练：SFT 混合初始化 + 仿真 RL 微调并加真实数据 SFT 正则防遗忘，OpenVLA +24%、$\pi_{0.5}$ +20% | Sim-Real Co-Training、RL + SFT 正则、数据效率、通用框架 | 2026.02 |
| [RLinf](/papers/06-embodied-ai/vla/rl/RLinf_2025) | 提出 M2Flow 宏-微流变换范式，通过弹性流水线和上下文切换实现灵活高效的大规模 RL 训练 | M2Flow、弹性流水线、RL 训练系统 | 2025.09 |
| [RLinf-USER](/papers/06-embodied-ai/vla/rl/RLinf_USER_2026) | 将机器人视为一等硬件资源，通过统一硬件抽象、云-边通信、全异步流水线构建真实世界在线策略学习系统 | 真实世界 RL、统一硬件抽象、云-边协同、异步训练 | 2026.02 |
| [RLinf-VLA](/papers/06-embodied-ai/vla/rl/RLinf_VLA_2025) | 统一高效的 VLA+RL 训练框架，三种 GPU 分配模式 + PPO/GRPO，单一模型 LIBERO-130 达 98.11% | Hybrid Pipelining、PPO/GRPO、统一 VLA+RL 框架 | 2025.10 |
| [RL-VLA Survey](/papers/06-embodied-ai/vla/rl/RL_VLA_Survey_2025) | 首篇系统综述 RL 后训练 VLA 的全景图：架构（动作/奖励/世界模型）、在线/离线/测试时训练范式、sim-to-real 部署与评测 | 综述、RL-VLA 分类体系、训练范式、部署 | 2025.12 |
| [RLVLA](/papers/06-embodied-ai/vla/rl/RLVLA_2025) | 系统性实证研究 RL 对 VLA 泛化性的收益：PPO 优于 DPO/GRPO，RL 在语义和执行维度显著优于 SFT | PPO、泛化基准、共享 Actor-Critic、SFT vs RL | 2025.05 |
| [RPD](/papers/06-embodied-ai/vla/rl/RPD_2025) | PPO + MSE 蒸馏项将 VLA 通才知识蒸馏为紧凑 RL 专家策略，稀疏奖励和视角变化下大幅优于 vanilla PPO | Policy Distillation、PPO + BC、VLA→RL 专家、ManiSkill3 | 2025.03 |
| [SAC Flow](/papers/06-embodied-ai/vla/rl/SAC_Flow_2026) | 把 Flow Policy 重新理解为序列模型，用 GRU/Transformer 重参数化解决 RL 梯度不稳定问题 | Flow Policy、序列建模、SAC、off-policy RL | 2026.01 |
| [SC-VLA](/papers/06-embodied-ai/vla/rl/SC_VLA_2026) | 稀疏世界想象（预测进度 + 状态增量）+ 残差 SAC 在线修正，内生密集奖励无需外部奖励模型 | 稀疏世界想象、残差 RL、内生奖励、Flow Matching | 2026.02 |
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
