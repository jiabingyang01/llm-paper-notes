# VLA Foundry：面向视觉-语言-动作模型的统一训练框架

> **论文**：*VLA Foundry: A Unified Framework for Training Vision-Language-Action Models*
>
> **作者**：Jean Mercat, Sedrick Keh, Kushal Arora, Isabella Huang, Paarth Shah, Haruki Nishimura, Shun Iwase, Katherine Liu et al.
>
> **机构**：Toyota Research Institute
>
> **发布时间**：2026 年 04 月（arXiv 2604.19728）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.19728) | [PDF](https://arxiv.org/pdf/2604.19728)
>
> **分类标签**：`VLA训练框架` `开源代码库` `LLM-VLM-VLA统一预训练` `flow-matching动作头` `双臂桌面操作仿真评测`

---

## 一句话总结

VLA Foundry 是 Toyota Research Institute 开源的统一训练框架,把 LLM、VLM、VLA 三阶段预训练整合进同一套 YAML 配置化代码库,并放出两条路线训出的模型:从零训练的 Foundry-VLA-1.7B 在其 16 任务双臂桌面操作仿真基准上与作者此前闭源的 LBM 多任务模型基本打平,而把骨干换成预训练 Qwen3-VL 2B 的 Foundry-Qwen3VLA-2.1B-MT 平均成功率领先约 23 个百分点。

## 一、问题与动机

- 现有开源 VLA 代码库(OpenVLA、OpenPi、GR00T、LeRobot/SmolVLA 等)大多只聚焦"动作训练"这一段,把上游的 LLM/VLM 预训练当作固定或超出范围的外部产物,不同来源、互不兼容的预训练管线被拼接在一起(例如 OpenVLA 建在 Prismatic VLM 之上、OpenPi 复用 π0 checkpoint),研究者难以对语言预训练到动作专家微调的整条管线做系统性、可控的消融。
- 机器人交互数据相对语言/视觉数据仍极度稀缺(多样性、信号密度都更低),这意味着非机器人数据在预训练阶段的配比、时机等决策会直接影响下游操作策略性能,但当前框架割裂了 LLM/VLM 训练与 VLA 训练两个阶段。
- 目标是构建单一代码库、共享数据加载与训练循环,贯穿 LLM→VLM→VLA 全流程的框架,让研究者可以在任意阶段介入(换骨干、换数据配比、换架构),同时原生支持从 Hugging Face 直接加载现成预训练权重(PaliGemma、Qwen-VL、SmolVLM 等),兼容"从零训练"与"接入现成骨干"两条路线。

## 二、核心方法

框架遵循四条设计原则——模块化组合、可魔改互操作(不套重框架如 PyTorch Lightning/HF Trainer,训练循环保持"薄")、性能(在 128 GPU/16 节点规模上做过吞吐测试)、可复现性(确定性逐 rank RNG 种子、dataloader 状态可 checkpoint、冻结 dataclass 防止运行时静默改配置)。

- **配置系统**：基于 Draccus 的冻结 dataclass,参数按"命令行 > YAML 预设 > 嵌套 include > 代码默认值"四级优先级解析;预设可继承组合,跨模块共享参数(如 hidden dim、序列长度)只解析一次并在 dataclass 树里传播,避免配置文件、日志、实际运行三者不一致。
- **注册表机制**：新增模型只需一个冻结参数 dataclass + 一个 `@register_model` 装饰的工厂函数;同一模型类型(LLM/VLM/DP-VLA)共用一个 batch handler(负责 batching、loss 构造、输出归约)与同一个训练循环,新的训练范式（如新增 diffusion policy）只需注册新 handler 而不改主循环。新增数据集经 Ray 并行预处理转为 WebDataset tar shard,同时产出归一化统计量。
- **机器人数据专用处理**（RoboticsNormalizer）：支持全局/逐时间步、基于 t-digest 的百分位数归一化;动作可用绝对世界坐标或相对锚点末端位姿表示,旋转采用 6D 连续表示,相对位姿在 SE(3) 中复合;动作按可配置窗口做 chunk（未来部分作监督目标,过去部分作输入),本体感知观测因果地限制在过去与当前步。
- **分布式训练**：FSDP2（可选 CPU offload）+ 混合精度 + 梯度检查点 + `torch.compile` + 梯度累积,在 AWS SageMaker P5 节点（每节点 8×H100）上测得 LLM/VLM/VLA 三阶段吞吐随 GPU 数近线性扩展至 128 GPU;在论文所用的模型规模（1.2B LLM / 1.1B VLM / 1.5B VLA）下单卡即可放下全部权重,FSDP 相比 DDP 无明显收益,VLM 阶段扩展性反而更弱。

**VLA 架构**（Foundry-VLA-1.7B）：在 VLM 词表中新增一个 observation token；VLM 输入序列由多张图像 patch embedding、文本 token embedding 与 observation token 拼接而成；取 VLM 最后 $N$（实验中 $N=4$）层里与 observation token 对齐位置的隐藏特征,作为条件驱动一个 flow transformer 对动作序列去噪。动作头是一个 325M 参数的 Transformer（与 LLM 同架构但词表大小为 0），输入为 VLM 隐藏特征、可选的本体感知线性编码、带噪动作序列线性编码三者拼接,用 flow-matching 目标训练。

用大白话说：把"看图/读文字得到一个隐藏摘要 token"和"把摘要 token 喂给专门去噪机械臂动作序列的小 Transformer"接起来,从语言预训练到动作专家全流程塞进同一份可配置代码,换骨干、换数据、换动作头只改 YAML 而不用换框架。

两个发布模型共享同一动作头架构：**Foundry-VLA-1.7B** 沿 LLM（1.2B, DCLM 1T token）→VLM（+86M 随机初始化 ViT, DataCompDR-1B 200M 样本 caption 训练）→VLA（+325M flow-transformer 动作头）全部从零训练；**Foundry-Qwen3VLA-2.1B-MT** 把 VLM 骨干直接换成预训练的 Qwen3-VL 2B,复用相同动作头与训练配方。训练数据混合仿真（42 个任务）与真实遥操作双臂桌面操作数据（361 个真实任务，39 个任务在仿真/真实中有对应站点复制版本），数据管线沿用作者此前 LBM 工作的采集方式,未使用 OXE 或 UMI 等公开数据。

评测配套开源仿真基准 lbm_eval_oss（基于 Drake 物理引擎的双臂桌面操作任务集）与统计分析工具 STEP（Bayesian 成功率估计小提琴图 + Compact Letter Display,5% 族错误率显著性判定），并强调多任务聚合对比时须按每任务最小 rollout 数截断以避免有偏聚合（指出此前 LBM 闭源工作未严格执行这一点）。

## 三、关键结果

LLM 阶段（Foundry-LLM-1.2B，标准 few-shot 多选基准，衰减前 800B token vs 衰减后 1T token）：

| 训练量 | HellaSwag | MMLU | ARC-e | ARC-c | PIQA | WinoGrande | OBQA | BoolQ |
|---|---|---|---|---|---|---|---|---|
| 800B tokens | 64.3 | 26.0 | 70.3 | 37.0 | 75.8 | 60.9 | 40.0 | 63.2 |
| 1T tokens | 66.7 | 26.6 | 71.7 | 39.3 | 77.5 | 62.6 | 40.8 | 65.4 |

VLM 阶段（Foundry-VLM-1.3B，COCO_VAL captioning，不同 caption 样本量）：

| 样本量 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|
| 165M | 57.25 | 37.12 | 23.23 | 14.44 | 37.13 | 50.17 |
| 200M | 58.64 | 38.62 | 24.49 | 15.57 | 38.17 | 55.14 |

VLA 阶段闭环仿真评测（lbm_eval，双臂桌面操作，聚合 16 个 seen 任务成功率，每模型 200 rollout）：

- 与此前闭源多任务模型 LBM-MT 对比（闭源仿真环境 lbm_eval_cs）：Foundry-VLA-1.7B-MT-sim 与 LBM-MT 在聚合成功率上统计打平；真实+仿真混合训练的 Foundry-VLA-1.7B-full 反而是四者中最弱；Foundry-Qwen3VLA-2.1B-MT 显著领先其余三者，论文结论中报告平均超出约 **23 个百分点**。
- 训练阶段对比（单任务 ST → 多任务 MT → 单任务微调 FT）：Foundry-Qwen3VLA-2.1B-MT 序列里 MT 优于 ST、FT 在 MT 基础上进一步提升，单调改善；Foundry-VLA-1.7B 序列里则相反，MT 和 FT 在聚合上统计显著劣于 ST，论文将这一差异归因于骨干强弱（骨干越强，多任务训练收益越大）。
- 未见任务零样本泛化（3 个 held-out 任务）：两个多任务模型都有非零零样本成功率；Foundry-Qwen3VLA-2.1B-MT 微调后聚合优于其单任务版本，Foundry-VLA-1.7B 则没有这一优势。
- 数据来源消融（仅 Foundry-VLA-1.7B）：纯仿真（MT-sim）、纯真实（MT-real）、真实+仿真混合（MT）三个多任务变体在同等计算量下训练；纯真实数据模型在仿真评测中成功率接近 0（分布外，符合预期）；纯仿真变体聚合表现最好，略优于真实+仿真混合版本，论文推测可能是欠训练或模型容量被真实/仿真任务分摊所致，留作未来工作。

## 四、评价与展望

- **优点**：填补了"LLM/VLM 预训练"与"VLA 动作训练"之间的代码库断层——此前多数开源 VLA 工作（OpenVLA、OpenPi、LeRobot/SmolVLA、VLA-Scratch、StarVLA 等）要么把上游 VLM 当作固定输入，要么必须跨代码库拼接；VLA Foundry 用单一配置驱动的训练循环打通了从纯文本预训练到闭环仿真评测的整条链路，并以 Foundry-LLM/VLM/VLA 三级 checkpoint 的形式全部开源，配套统计显著性分析（STEP）与仿真基准 dashboard，便于社区在统一协议下横向比较策略。
- **局限**（论文自陈）：报告仅覆盖 lbm_eval 闭环仿真结果，未给出真实机器人硬件上的数字；动作头目前只验证了 flow-matching 一种（代码库中已有 diffusion policy 等其他头但未做实验）；框架虽支持跨阶段概率化多模态数据混合，但论文没有系统刻画"最优数据配方"是什么，也未涉及安全、对齐或失败检测。
- 实验本身也暴露出开放问题：从零训练的 Foundry-VLA-1.7B 在多任务/微调设置下反而不如单任务版本，与用更强 Qwen3-VL 骨干的模型规律相反，论文归因于骨干能力强弱但未给出更细粒度机制分析（容量瓶颈还是预训练数据/目标不匹配尚不清楚）；纯仿真数据变体略优于真实+仿真混合变体的现象同样只给出了两个未经验证的假设。这说明"用同一套框架做可控消融"和"框架本身把这些消融的最优解找出来"是两回事，后续工作可借助其可配置性系统性回答。
- **与同类公开工作的关系**：相比 LeRobot/SmolVLA 侧重廉价硬件与社区易用性、VLA-Scratch/StarVLA 侧重解耦骨干与动作头的模块化、Dexbotic 侧重跨平台复现已有 VLA，VLA Foundry 的差异化定位是贯穿 LLM→VLM→VLA 全流程的单一训练栈，更贴近需要从预训练阶段就做数据/架构联合消融的研究场景，而非单纯的动作策略微调工具。其闭环评测仍局限于作者自建的双臂桌面操作仿真基准（lbm_eval），尚未接入 LIBERO、SimplerEnv、RoboCasa 等社区常用基准，真实世界结果的缺失也使其"训练框架有效性"的证据链目前止步于仿真。

## 参考

- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", arXiv:2406.09246, 2024
- Karamcheti et al., "Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models", ICML 2024
- TRI LBM Team et al., "A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation", arXiv:2507.05331, 2025
- Cadene et al., "LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch", 2024
- StarVLA Community, "StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing", arXiv:2604.05014, 2026
