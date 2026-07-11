# StarVLA：面向 VLA 模型开发的“乐高式”开源代码库

> **论文**：*StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing*
>
> **作者**：Jinhui Ye, Ning Gao, Yilun Chen†, Weiyu Guo, Zixuan Wang, Yuxing Chen, Fangjing Wang, Senqiao Yang, Chengyao Wang, Yuqi Liu, Meng Chu, Changsheng Lu, Pengguang Chen, Shu Liu†, Jiaya Jia† et al.（† 通讯作者）
>
> **机构**：StarVLA Community；Von Neumann Institute, HKUST
>
> **发布时间**：2026 年 04 月（arXiv 2604.05014）
>
> **发表状态**：未录用（预印本，持续更新中的技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.05014) | [PDF](https://arxiv.org/pdf/2604.05014)
>
> **分类标签**：`VLA开源框架` `Backbone-ActionHead解耦` `多benchmark统一评测` `跨具身co-training` `系统效率剖析`

---

## 一句话总结

StarVLA 提出一个"backbone-action head"解耦的统一抽象，把 VLM-based 与 World-Model-based 两大 VLA 范式（FAST 自回归、OFT 并行回归、π0 流匹配、GR00T 双系统）纳入同一代码库、同一数据接口与同一评测协议，在 LIBERO 上用约 1/6 的训练步数（30K vs 175K）匹配甚至反超 OpenVLA-OFT 的 97.1% 均值，并在 RoboCasa-GR1 的 24 任务联合泛化设置下把成功率从最佳 specialist 的 48.8% 提升到 57.3%。

## 一、问题与动机

论文指出当前 VLA 研究存在三层"巴别塔"式碎片化：

- **架构层**：不同方法（RT 系列、OpenVLA、π0、GR00T N1.5、Lingbot-VLA 等）采用互不兼容的动作解码方式，从自回归 token 化到 diffusion/flow-matching，难以做受控对比；
- **系统层**：各代码库对模型结构、数据处理、训练流程做了紧耦合假设，组件无法跨项目复用；
- **评测层**：各论文只在互不重叠的 benchmark 子集上报数字，且预处理/协议细节不一致，公平比较基本不可行。

作者将根源归结为"缺乏统一的 VLA 系统抽象"：已有开源代码库（如 OpenPI、Isaac GR00T、OpenVLA-OFT、Dexbotic、X-VLA）大多是方法特定的，不支持 (i) 跨动作解码范式的模块化组合、(ii) 跨异构数据源的可复用训练、(iii) 跨 benchmark 的标准化评测与部署。StarVLA 的目标是提供这样一个统一研究平台。

## 二、核心方法

### 2.1 policy-centric 的统一 VLA 形式化

论文把任意 VLA 系统抽象为一个策略，将视觉-语言观测映射到未来动作序列与可选辅助输出：

$$\pi(\mathbf{a}_{t:t+k}, \mathbf{y}_{\text{aux}} \mid \mathbf{x}_{\le t}, \ell), \qquad (1)$$

其中 $\mathbf{x}_{\le t} = \{o_{\le t}^{\text{vis}}, o_{\le t}^{\text{depth}}, o_{\le t}^{\text{tactile}}, \ldots\}$ 为多模态观测历史，$\ell$ 为语言指令，$\mathbf{a}_{t:t+k}$ 为 $k$ 步动作 chunk，$\mathbf{y}_{\text{aux}}$ 为可选辅助输出（如未来帧预测、子任务规划语言 token 等）。

训练目标统一写成

$$\mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{aux}}, \qquad (2)$$

**用大白话说**：不管是"VLM 硬用来做动作"还是"world model 顺带做动作"，本质都是同一个策略函数，区别只在于有没有辅助 loss、辅助 loss 是语言型的（子任务规划/空间 grounding）还是视觉型的（未来帧预测）。把 $\mathcal{L}_{\text{aux}}$ 设为 0 就是纯 Direct VLA Modeling；引入语言辅助目标就是 VLM-based VLA；引入未来观测预测（显式或隐式）就是 WM-based VLA。这一视角被作者称为"generalized VLA perspective"——差异被下沉为归纳偏置的形式选择，而不是根本不同的建模范式。

### 2.2 系统抽象：统一 I/O 接口 + backbone-action head 组合

所有框架模块继承同一基类，暴露两个统一接口：

- `forward({raw images, str, ...}) → {loss dict}`：训练入口，接收原始多模态样本（多视角 RGB、语言指令、动作 chunk），返回 loss；
- `predict_action({raw images, str, ...}) → {normalized_actions, ...}`：推理入口，接收与部署时完全相同的原始观测格式，返回动作 chunk。

**用大白话说**：训练和真实机器人部署喂给模型的观测格式必须完全一致，这样就不会出现"训练时用了预处理过的 dataloader tensor，部署时忘了复现同样的预处理"这种常见的隐蔽性能掉点问题。

内部进一步将任意 VLA 方法拆成两个通过标准化表示契约连接的组件：**VL backbone**（如 Qwen2.5-VL/Qwen3-VL、Cosmos-Predict2 等视频世界模型）消费原始多模态观测并输出隐藏状态 $z$；**可插拔 action head** 读取该隐藏状态并转换为动作。二者都通过 YAML 声明式配置，backbone 与 head 可以互相独立替换（"双向模块化"）。

### 2.3 四种代表性范式的统一实现

在该抽象下，StarVLA 实现了横跨动作解码主要家族的四种范式，共享同一 VL backbone、同一 base class 与同一 forward/predict_action 契约，仅在"如何从 backbone 表示中抽取动作"上不同：

- **StarVLA-FAST**（对应 $\pi_{\text{fast}}$）：在 VL backbone 后接 FAST tokenizer，以自回归 next-token prediction 的方式在 LLM 自身词表空间里生成离散动作 token；
- **StarVLA-OFT**：接一个轻量 MLP，读取预定义动作 token 的隐藏状态，用 L1 loss 并行回归连续动作（复现 OpenVLA-OFT，是最简单的可插拔 head）；
- **StarVLA-π**（对应 π0）：引入逐层 cross-attention 的 DiT flow-matching action expert，以多层 VL 隐藏状态为条件，通过迭代去噪预测连续动作；
- **StarVLA-GR00T**：采用双系统设计，VL backbone 作 System 2（慢推理），基于 DiT 的 flow-matching 模块作 System 1（快动作生成），对齐 GR00T N1.5。

四种范式覆盖了从"VLM 原生解码（自回归 token 化、并行回归）"到"生成式模型解码（迭代 flow-matching 去噪、双系统推理）"的完整谱系；新增范式只需实现并注册一个新的 action head，backbone、训练循环与评测流水线保持不变。

### 2.4 训练与评测基础设施

训练支持四类 paradigm-agnostic（而非方法专属）的可复用配方：(i) 标准 SFT 行为克隆；(ii) 多目标 co-training——用双 dataloader（VLA 动作数据 + VLM 网络多模态数据）在每步优化中分别做前向反向，语言建模 loss 权重通过 `loss_scale.vlm` 控制，用于缓解动作专精微调导致的多模态能力遗忘；(iii) 跨具身 co-training——通过 `LeRobotMixtureDataset` 按 (数据集名, 采样权重, 机器人类型) 三元组混合异构机器人数据集；(iv) 强化学习微调——论文写明这是与 RLinf 项目协作、正在集成中的功能，当前公开代码库尚未包含。

评测侧采用统一的 server-client 架构：模型侧作为轻量 WebSocket policy server，各 benchmark 官方评测器作为 client，通过 msgpack 序列化的观测字典（image、lang、state 等）与服务器通信，benchmark 特有差异（分辨率缩放、动作反归一化、sticky gripper 等）被隔离在轻量适配器文件（如 `model2libero_interface.py`）中，核心 policy server 保持 benchmark-agnostic；同一套 client-server 契约同时支持仿真评测与真实机器人部署，checkpoint 可跨环境直接复用。

StarVLA 集成了 7 个 benchmark：**LIBERO**（130 任务、约 6.5K 轨迹）、**LIBERO-Plus**（10,030 个含 7 类扰动因子的测试任务）、**SimplerEnv**（WidowX/Google Robot 两套协议）、**RoboCasa-GR1**（24 任务、约 24K 轨迹）、**RoboTwin 2.0**（50 双臂任务 × clean/random 两种设置、共 27.5K 轨迹、1 万评测 episode）、**BEHAVIOR-1K**（1000 项日常活动、9000+物体）、**CALVIN**（ABC→D 长时程语言条件设置）。论文 Table 1 对比 OpenPI、Isaac GR00T N1.5、OpenVLA-OFT、Dexbotic、X-VLA 等已有开源框架后指出，StarVLA 是唯一同时具备"模块化动作头 + 模块化 VLM backbone + 模块化 world-model backbone + 内建混合 dataloader + 开源多模态 co-training + 开源跨具身 co-training + 多 benchmark 联合训练"全部能力的平台（对比框架的 #Bench 为 1–6，仅 StarVLA 达到 7 且支持 multi-bench co-train）。

## 三、实验结果

**LIBERO（单策略联合训练 4 个 suite，10 任务 × 50 episode/suite，共 500 trial/suite）**

| 方法 | Steps | Epochs | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| π0+FAST | – | – | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| OpenVLA-OFT | 175K | 223 | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | – | – | 96.8 | 92.0 | 95.8 | 85.2 | 94.1 |
| GR00T-N1.5 | 20K | 203 | – | – | – | – | 86.5 |
| StarVLA-FAST（Qwen3-VL-4B） | 30K | 9.54 | 97.3 | 97.4 | 96.3 | 90.6 | 95.4 |
| StarVLA-OFT（Qwen3-VL-4B） | 30K | 9.54 | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| StarVLA-π（Qwen3-VL-4B） | 30K | 9.54 | 98.8 | 99.6 | 95.8 | 88.4 | 95.7 |
| StarVLA-GR00T（Qwen3-VL-4B） | 30K | 9.54 | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |
| StarVLA-OFT（Cosmos-Predict2-2B） | 30K | 9.54 | 98.6 | 97.6 | 95.0 | 91.8 | 95.8 |

StarVLA 用不到 1/6 的训练步数（30K vs 175K）、约 1/23 的 epoch 数即可逼近 OpenVLA-OFT 报告的 97.1% 均值；替换 VL backbone 为 Cosmos-Predict2-2B 后各 head 精度基本持平（均 ≥95.2%），说明该 pipeline 对 VLM-based 与 World-Model-based backbone 都具良好数据效率。

**SimplerEnv（5 次评测取均值）**：WidowX Visual Matching 上 StarVLA-GR00T（Qwen3-VL-4B）取得 65.3% 的最高均值（对比 GR00T N1.5 论文报告 61.9%、SpatialVLA 42.7%、π0 48.3%）；Cosmos-Predict2-2B backbone 下最高 61.6%。Google Robot 上 StarVLA-OFT 取得 Visual Matching 76.0%、Variant Aggregation 70.2%，均高于 CogACT（74.8/61.3）、SpatialVLA（75.1/70.7）等强基线。

**RoboCasa-GR1（24 任务均值，single-benchmark 训练）**

| 方法 | Avg(%) | 方法 | Avg(%) |
|---|---|---|---|
| π0.5 | 37.0 | GR00T-N1.6 | 47.6 |
| StarVLA-FAST | 39.0 | StarVLA-π | 43.9 |
| StarVLA-GR00T | 47.8 | **StarVLA-OFT** | **48.8** |

**RoboTwin 2.0（50 任务 × clean/random，各 100 episode/任务）**

| 方法 | Clean | Random |
|---|---|---|
| π0 | 65.9 | 58.4 |
| X-VLA | 72.9 | 72.8 |
| π0.5 | 82.7 | 76.8 |
| StarVLA-FAST | 72.5 | 83.2 |
| Lingbot-VLA | 88.6 | 86.7 |
| StarVLA-OFT | 88.2 | 88.3 |
| StarVLA-GR00T | 88.0 | 88.5 |
| **StarVLA-π** | 88.1 | **88.8** |

**多模态 co-training（Table 8，引自基于 StarVLA 的后续研究 ST4VLA）**：相较 Vanilla VLA（纯动作微调，RefCOCO-g 20K 步内退化到近随机水平），"+co-training"已能同时提升操作与多模态能力（Google Robot VM +4.1pt、WidowX +6.4pt）；"+spatially guided"进一步达到 Google Robot VM/VA 84.6/75.9%、WidowX 73.2%，同时保持 RefCOCO-g IoU@0.5 71.2%（原始 grounding 能力的 ~70%）。

**跨 benchmark generalist（Table 9）**：单个模型联合在 LIBERO+SimplerEnv+RoboTwin+RoboCasa 的合并训练集上训练（统一 32 维 padding 动作向量，无 benchmark 专属微调），在多数 benchmark 上与逐一训练的 specialist 相当，并把 RoboCasa-GR1 从最佳 specialist 的 48.8% 提升到 57.3%。

**系统效率（Table 10/11，以 StarVLA-GR00T + Qwen3-VL-4B 在 RoboCasa-GR1 数据上、A100 80GB 为准）**：单节点 8×A100 上，per-GPU batch 从 2 增至 24 时，step 延迟由 0.703s 增至 2.404s，但样本吞吐从 22.7 samples/s 升至 79.9 samples/s（GPU 利用率 74%→96%）；多节点弱扩展在固定 per-GPU batch=8 下，8→256 GPU 的 step 延迟由 0.735s 升至 0.93s 后趋于平台，并行效率稳定在 79–80%，样本吞吐从 87 到 2200 samples/s 接近线性扩展。

## 四、局限性

- **系统贡献而非算法贡献**：四个 paradigm（FAST/OFT/π/GR00T）均为已发表方法的复现与统一封装，论文核心创新是系统抽象与工程整合，而非新的建模算法。
- **RL 微调尚未交付**：强化学习微调被列为"正在进行的、与 RLinf 项目协作的集成工作"，当前公开代码库只覆盖 SFT 与 co-training。
- **部分实验数字来自外部研究**：Section 6 的多模态 co-training 数字引自另一篇基于 StarVLA 构建的独立后续论文（ST4VLA），并非本报告自身的消融实验，论文自身对该机制的第一手验证有限。
- **跨具身动作表示较粗糙**：跨 benchmark generalist 训练采用简单的零填充策略把低自由度动作统一 pad 到 32 维向量，对差异悬殊的形态（如移动底盘、灵巧手）是否是最优表示未做讨论。
- **已集成 benchmark 缺实测**：BEHAVIOR-1K、CALVIN、LIBERO-Plus 在 Section 4 中被列为已集成 benchmark，但报告当前版本未给出这三者的实测结果，作者也明确表示"we will update this report as project evolves"，说明这是一份持续演进中的技术报告而非定稿结果集。
- **高成本评测未做全 head 对比**：Google Robot（SimplerEnv）由于评测成本高，仅报告 StarVLA-OFT 一个代表配置，未覆盖 FAST/π/GR00T 在该 benchmark 上的完整对比。
- **效率剖面覆盖面窄**：Section 8 的 scaling 分析只测了一个特定组合（GR00T-N1.5 head + Qwen3-VL-4B backbone + RoboCasa-GR1 数据），不必然代表其余 backbone/head 组合的扩展行为。

## 五、评价与展望

**优点**：StarVLA 是较早尝试把 VLM-based 与 World-Model-based 两条 VLA 技术路线纳入同一 backbone-action head 抽象、同一数据接口与同一评测协议的开源平台，使得四种代表性范式可以在完全相同的训练资源与数据条件下直接对比，减少了跨论文比较中常见的协议不一致问题。工程侧给出的 baseline 已经具备较强说服力：LIBERO 上用远少于原论文的训练步数即匹配甚至反超已发表结果；RoboTwin 2.0、RoboCasa-GR1 等更具挑战性的 benchmark 上也取得有竞争力甚至最优的数字。单节点/多节点效率剖析（Table 10/11）为社区提供了直接可用的工程扩展指导。

**与其他公开工作的关系**：相较 OpenPI（仅覆盖 π0 一路 paradigm）、Isaac GR00T（GR00T-only）、Dexbotic、X-VLA 等现有开源框架，StarVLA 在模块化范围（backbone 与 head 均可插拔、支持 world model 作为 backbone）、benchmark 覆盖（7 个）与训练配方复用性（co-training、跨具身混合作为一等公民而非方法专属附加项）上更为完整（见原文 Table 1），其定位类似机器人学习界的"统一 backbone-action head Transformers 库"角色，与近期 InternVLA-A1、Lingbot-VLA、X-VLA 等强调跨具身/跨范式统一的工作方向一致，但更侧重可复现的系统基础设施而非单点算法性能。

**开放问题**：(1) 统一抽象长期是否会限制引入全新输入模态或全新 loss 结构的新范式的可扩展性，还有待观察；(2) 32 维零填充的跨具身动作表示是权宜之计，对足式、移动底盘等差异悬殊的形态是否需要更结构化的统一表示是开放问题；(3) generalist 单模型在 RoboCasa 上明显超越 specialist，但在 SimplerEnv/RoboTwin 上仅为"competitive"而非全面超越，多 benchmark 联合训练中的任务间正/负迁移机制仍需更细致的消融；(4) RL 微调集成完成后，能否继续保持"backbone/head 独立可换"的双向模块化契约，将是检验该抽象长期有效性的关键节点。

## 参考

- Black, K., Brown, N., Driess, D., et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164.
- Kim, M. J., Finn, C., Liang, P. (2025). *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*（OpenVLA-OFT）. arXiv:2502.19645.
- Bjorck, J., Castañeda, F., Cherniadev, N., et al. (2025). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734.
- Chen, D., Zhang, J., Mu, T., et al. (2025). *RoboTwin 2.0: Towards General Robot Policies with Active Data Generation*. arXiv:2504.13059.
- Ye, J., Wang, F., Gao, N., et al. (2026). *ST4VLA: Spatially Guided Training for Vision-Language-Action Models*. arXiv:2602.10109.
