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
| [ACPO](/papers/04-multimodal/vlm/hallucination/ACPO_2026) | 定义视觉锚点崩塌（chosen 似然随 DPO 训练下降导致放弃视觉证据），引入长度自适应优势目标 τ=δ(|yw|+|yl|) 缩放目标间距，stop-gradient 的非对称系数 α 仅作用于 rejected 梯度，InternVL3-14B POPE 89.22、MM-IFEval 0.570，8B 同样多基准 SOTA | 似然位移、视觉锚点崩塌、非对称标定系数、长度自适应目标 | 2026.03 |
| [AGLA](/papers/04-multimodal/vlm/hallucination/AGLA_2025) | GradCAM 驱动的 Image-Prompt Matching 生成增强图像，融合原始图像全局生成特征与增强图像局部判别特征进行校准解码，training-free 即插即用，POPE 平均提升 5.5% Accuracy / 5.1% F1 | GradCAM、全局-局部注意力组装、自适应遮蔽、Training-Free | 2024.06 |
| [AVISC](/papers/04-multimodal/vlm/hallucination/AVISC_2025) | 发现 LVLM 中盲 token 垄断注意力却不携带判别信息，三步解码校准（层选择 + 盲 token 识别 + 对比解码）动态抑制注意力偏差，InstructBLIP POPE Accuracy +6%、AMBER 得分 85.95 | Blind Token、层选择注意力校准、对比解码、Training-Free | 2024.05 |
| [CIPHER](/papers/04-multimodal/vlm/hallucination/CIPHER_2026) | 扩散模型生成 25K 反事实图像（结构保留 + 语义篡改），对比真实/反事实隐藏表示差异经 SVD 提取视觉诱导幻觉子空间，推理时投影到正交补，training-free 零推理开销，LLaVA-1.5 CHAIR$_S$ 降至 13.05%（vs Nullu 15.20%） | 扩散反事实图像、SVD 幻觉子空间、特征投影、Training-Free | 2026.03 |
| [CSR](/papers/04-multimodal/vlm/hallucination/CSR_2024) | 句子级 beam search + CLIP 视觉校准奖励迭代构造自生成偏好数据 + DPO 微调，三轮迭代 10 个基准平均提升 7.62%，CHAIR$_S$ 降低 57% | 校准自奖励、CLIP Score、迭代 DPO、模态对齐、Self-Rewarding | 2024.05 |
| [DLC](/papers/04-multimodal/vlm/hallucination/DLC_2025) | 解码时用 CLIP 逐步评估候选 token 的相对视觉优势 (RVA)，相对动态历史基线自适应调整 logits，无需额外前向传播高效缓解语义漂移幻觉 | 动态 Logits 校准、CLIP 探针、相对视觉优势、自适应引导、Training-Free | 2025.06 |
| [EFUF](/papers/04-multimodal/vlm/hallucination/EFUF_2024) | CLIP 分数自动区分幻觉/非幻觉对象，细粒度梯度上升遗忘幻觉子句 + 正向训练保留正确子句 + 句子损失维持流畅度，无需配对数据仅 3 GPU 小时，4 个 MLLM 上 CHAIR$_S$ 平均降低 ~15% 且生成质量同步提升 | 细粒度遗忘、CLIP 数据筛选、梯度上升、三重损失 | 2024.02 |
| [FLB](/papers/04-multimodal/vlm/hallucination/FLB_2026) | 存储首 token logit 以指数递增权重叠加到后续解码步骤，通过直接视觉锚定和 "The" 效应双机制缓解长程视觉衰减，单次前向零推理开销，LLaVA-1.5 AMBER CHAIR 11.5→6.1、CHAIR$_S$ 57.5→43.5，全面超越 VCD/ICD/M3ID | 首 Token Logit、视觉锚定、长程衰减、"The" 效应、Training-Free、单次前向 | 2026.04 |
| [FarSight](/papers/04-multimodal/vlm/hallucination/FarSight_2025) | 在因果掩码上三角引入注意力寄存器吸收 outlier token 多余注意力 + 渐减遮蔽率编码绝对位置信息对抗 RoPE 衰减，仅优化因果掩码即可 training-free 缓解图像和视频 MLLM 幻觉，LLaVA-1.5 CHAIR$_S$ -6.4 pp、POPE-R +3.5 pp | 注意力寄存器、因果掩码优化、位置感知编码、Training-Free、Image+Video | 2025 |
| [HALC](/papers/04-multimodal/vlm/hallucination/HALC_2024) | 自适应 FOV 采样 + JSD 双向对比解码修正局部幻觉 + 视觉匹配 beam search 全局保障，无训练即插即用，CHAIR$_S$ 在 MiniGPT-4 上降低 36% | FOV 对比解码、JSD 选择、视觉匹配 Beam Search、Plug-and-Play | 2024.03 |
| [HIME](/papers/04-multimodal/vlm/hallucination/HIME_2026) | 提出 HIS 量化每层幻觉敏感度，层自适应加权投影编辑 MLP 权重，无训练/无额外参数/无推理开销平均降低 61.8% 对象幻觉 | HIS、层自适应模型编辑、零空间投影、Training-Free | 2026.02 |
| [IBD](/papers/04-multimodal/vlm/hallucination/IBD_2024) | 修改注意力权重构造图像偏置模型做对比解码，统计发现 CD score 对内容词有效/功能词无效，设计 $I_{sim} \times I_{con}$ 双指标动态调节 + prompt tuning 仅 74K 参数，4 个 LVLM 上全面超越 VCD/OPERA/Woodpecker，LLaVA-1.5 CHAIR$_S$ 降至 12.7 | 图像偏置注意力、内容词/功能词动态调节、Prompt Tuning、74K 参数 | 2024.02 |
| [HIO](/papers/04-multimodal/vlm/hallucination/HIO_2024) | 反转 Bradley-Terry 模型训练"Evil LVLM"精准放大幻觉 token，多幻觉同时诱导 + logit 级约束，推理时对比解码消除幻觉，POPE Accuracy +3.5%、CHAIR$_I$ 降至 2.24 | 反转 BT 模型、Evil LVLM 对比解码、多幻觉诱导、Logit 约束 | 2024.05 |
| [ICD](/papers/04-multimodal/vlm/hallucination/ICD_2024) | 发现指令前加角色前缀会放大 LVLM 幻觉，据此提出指令对比解码——标准指令分布减去扰动指令分布剥离幻觉概念 + 自适应截断约束，training-free 模型无关，POPE 平均提升 10.5%/6.0%，MME 幻觉子集 +80/+88 | 指令对比解码、多模态对齐不确定性、自适应截断、Training-Free | 2024.03 |
| [LessIsMore](/papers/04-multimodal/vlm/hallucination/LessIsMore_2024) | 发现 LVLM 内在具备基于视觉感知评估文本完整性来决定终止生成的 EOS 决策能力，但被过度详细的训练数据抑制；Selective EOS Supervision 在非 EOS 位置排除 EOS 参与 softmax + Scoring EOS Supervision 过滤有害数据，LLaVA-1.5 CHAIR$_S$/CHAIR$_I$ 分别降低 26.4%/26.6% | EOS 决策、Selective EOS Supervision、数据过滤、训练目标修改 | 2024.02 |
| [LogicCheckGPT](/papers/04-multimodal/vlm/hallucination/LogicCheckGPT_2024) | 逻辑闭环探测（对象→属性→对象）检测 LVLM 逻辑一致性差异，无训练即插即用黑盒后处理，mPLUG-Owl POPE 准确率提升超 30% | 逻辑闭环、逻辑一致性、Training-Free、Plug-and-Play | 2024.02 |
| [LPOI](/papers/04-multimodal/vlm/hallucination/LPOI_2025) | 首次将列表级偏好优化引入 VLM 幻觉缓解：对象检测定位关键物体 → 渐进遮蔽生成硬负样本图像序列 → Plackett-Luce 列表级排序损失按对象可见度排列偏好，无需额外标注，Object HalBench CHAIR$_S$ 从 mDPO 30.7 降至 24.3 | 列表级偏好优化、对象遮蔽、渐进插值、硬负样本、Visual Prompting | 2025.05 |
| [LURE](/papers/04-multimodal/vlm/hallucination/LURE_2024) | 统计分析揭示幻觉三因素（共现/不确定性/位置），GPT-3.5 构造针对性幻觉数据训练轻量修正器，[IDK] 占位符引导重评估可疑对象，后处理兼容任意 LVLM，6 模型 CHAIR$_S$ 平均降低 50%+ | 共现/不确定性/位置统计分析、GPT-3.5 幻觉数据、[IDK] 修正器、Post-hoc | 2023.10 |
| [mDPO](/papers/04-multimodal/vlm/hallucination/mDPO_2024) | 发现多模态 DPO 存在无条件偏好问题（移除图像效果不变），增加图像条件偏好对比 + 锚定奖励正则化，3B+10K 数据媲美 7B+80K，CHAIR$_S$ 降低 37% | 条件偏好优化、图像对比偏好、奖励锚定、无条件偏好问题 | 2024.06 |
| [MemVR](/papers/04-multimodal/vlm/hallucination/MemVR_2025) | 将视觉 token 通过 FFN key-value memory 机制重注入中间层，不确定性超阈值时动态触发 look-twice，POPE +7.0%、CHAIR$_I$ -15.6%，推理仅 1.04× 延迟且通用能力同步提升 | FFN Key-Value Memory、视觉回溯、不确定性触发、Training-Free、Plug-and-Play | 2025.05 |
| [MMHalSnowball](/papers/04-multimodal/vlm/hallucination/MMHalSnowball_2024) | 首次系统研究多模态幻觉雪球效应（前轮幻觉误导后续回答），提出 MMHalSnowball 评估框架（4,973 样本）+ 残差视觉解码（RVD）自适应强调视觉信息，缓解 24%+ 雪球幻觉 | 幻觉雪球效应、残差视觉解码、自适应分布混合、Training-Free | 2024.07 |
| [OPERA](/papers/04-multimodal/vlm/hallucination/OPERA_2024) | 发现幻觉与注意力柱状聚合模式共现，Beam Search 中引入列乘积过度信任惩罚 + 回溯重分配策略，无训练/无数据/无外部知识，Shikra CHAIR$_S$ 降低 ~35% | 注意力聚合模式、Over-Trust Penalty、Beam Search 回溯、Training-Free | 2023.10 |
| [REVERIE](/papers/04-multimodal/vlm/hallucination/REVERIE_2024) | 在视觉指令微调中引入正向/负向 rationale 的反思学习，构建 115k 指令 × 254k 三元组的 REVERIE 数据集，多轮对话解耦 rationale 与回答预测，POPE +12.7、MME +348 | 反思微调、正负 Rationale、细粒度推理监督、REVERIE 数据集 | 2024.07 |
| [RFI](/papers/04-multimodal/vlm/hallucination/RFI_2026) | 利用 Rectified Flow 从正负样本隐藏状态差异学习线性轨迹，推理时动态预测输入特定干预向量 + SVD 去噪后注入解码器隐藏层，仅 1.09x 延迟开销，LLaVA-v1.5 POPE 平均 F1 +7.59%，9 个子集全面 SOTA | Rectified Flow、动态干预向量、输入自适应、SVD 去噪 | 2026 |
| [SENTINEL](/papers/04-multimodal/vlm/hallucination/SENTINEL_2025) | 域内自举采样 + 检测器交叉验证构建句子级偏好数据，C-DPO 在幻觉首次出现处早期干预，Object HalBench 幻觉率降低 92% 且通用能力不降反升 | 句子级早期干预、域内偏好学习、C-DPO、交叉验证 | 2025.07 |
| [SIMA](/papers/04-multimodal/vlm/hallucination/SIMA_2024) | LVLM 自生成候选响应 + 三视觉指标引导的上下文自评估构造偏好对，DPO 微调无需外部模型或数据，14 个基准平均提升 7.5%、CHAIR$_S$ -19.5% | 自生成响应、上下文自评估、三视觉指标、DPO、Self-Improvement | 2024.05 |
| [STIC](/papers/04-multimodal/vlm/hallucination/STIC_2024) | 两阶段自训练——good/bad prompt + 图像腐蚀构造偏好数据正则化 DPO 提升图像描述能力，再将自生成描述注入 SFT 数据提升推理，仅 6k 无标签图像 + 5k 复用 SFT 数据，LLaVA-v1.6 7B 七基准平均 +4.0% | 自训练、图像理解、描述注入微调、正则化 DPO、Self-Training | 2024.05 |
| [TAF](/papers/04-multimodal/vlm/hallucination/TAF_2026) | Token 级分析发现 phantom token（文本→视觉异常高影响）和 anchor token（关键视觉证据不足），在视觉活跃层非对称过滤注意力 logits——隔离幻影 + 强调锚点，training-free 即插即用，POPE 全面 SOTA（LLaVA-1.5 Adversarial F1 86.21），CHAIR$_S$ 降至 42.5 | Phantom Token 隔离、Anchor Token 强调、非对称注意力过滤、Training-Free | 2026 |
| [VACoDe](/papers/04-multimodal/vlm/hallucination/VACoDe_2024) | Softmax L2 距离自适应选择最具对比性的图像增强（7 种候选）用于对比解码，无需训练或外部模型，MME/VQAv2/MMBench 跨 3 种模型一致超越 VCD 和单一增强 | 视觉增强选择、Softmax Distance、对比解码、Training-Free、Plug-and-Play | 2024 |
| [VCD](/papers/04-multimodal/vlm/hallucination/VCD_2026) | 高斯噪声放大视觉不确定性暴露语言先验和统计偏差，原始/噪声图像输出分布对比解码 + 自适应可信度约束，training-free 无需外部工具，POPE 最高 +5.8 Acc / +7.4 F1，MME 幻觉子集 +18% | 视觉对比解码、高斯噪声扰动、统计偏差、语言先验、Training-Free | 2023.11 |
| [VGA](/papers/04-multimodal/vlm/hallucination/VGA_2024) | Referent Method 在数据中嵌入坐标/颜色/形状视觉指代 + FAC 两阶段微调（Foundation 聚焦图像 → Advanced 对齐意图），63.8k GUI VQA 数据集上训练，GUI 理解 benchmark 以 90.83 分超越 GPT-4V (81.82) 和 GPT-4o (80.75) | GUI 理解、Referent Method、两阶段微调、Image-Centric | 2024.06 |
| [VisFlow](/papers/04-multimodal/vlm/hallucination/VisFlow_2025) | Token 级别增强视觉显著 token + Head 级别抑制系统提示头和文本跟随头，双层注意力干预无训练缓解幻觉，LLaVA-1.5 CHAIR$_S$ 降低 40%、POPE Adversarial F1 +10.8 pp | 双层注意力干预、Visual Sink/Salient Token、Head 分类抑制、Training-Free | 2025.06 |

### 🖼️ Multimodal — VLM Token 压缩

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [DART](/papers/04-multimodal/vlm/efficiency/DART_2025) | 揭示 importance-based token pruning 在 2/3 场景不如随机剪枝，转而基于 token 重复度选择少量 pivot 并移除高重复视觉 token，兼容 FlashAttention，LLaVA-1.5-7B 88.9% 剪枝率保持 93.7% 性能实现 1.99× 加速 | Token Duplication、Pivot Token、FlashAttention 兼容、Training-Free | 2025.02 |
| [Elastic Cache](/papers/04-multimodal/vlm/efficiency/ElasticCache_2024) | 指令编码阶段用注意力驱动的 importance-driven cache merging 合并冗余 KV 向量，生成阶段用固定截断点淘汰策略，training-free 即插即用，0.2 KV Budget 实现 78% 加速且全面超越 H2O/StreamingLLM | KV Cache 压缩、Cache Merging、两阶段策略、Training-Free | 2024.07 |
| [Token Pruning Survey](/papers/04-multimodal/vlm/efficiency/TokenPruningSurvey_2025) | 系统性分析 MLLM 视觉 token 剪枝五大核心问题：注意力位置偏差导致精心设计方法不如随机剪枝、语言引导仅在文本强关联任务有效、重要性 vs. 冗余性需按任务自适应平衡、FLOPs 不等于真实延迟、训练感知压缩远优于推理阶段剪枝 | Token Pruning、位置偏差、重要性 vs. 冗余性、训练感知压缩、评估方法论 | 2025.02 |
| [VisionZip](/papers/04-multimodal/vlm/efficiency/VisionZip_2024) | Text-agnostic 视觉 token 压缩：CLS 注意力选出 dominant token + key 相似度合并 contextual token，LLM 输入前完成压缩，LLaVA-1.5 仅 64/576 token 达 94% 性能（超 FastV 18.4%），LLaVA-NeXT 实现 8× prefilling 加速，13B 推理速度超 7B | Dominant Token Selection、Token Merging、Text-Agnostic、Training-Free | 2024.12 |

### 🖼️ Multimodal — 视频生成

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [WorldForge](/papers/04-multimodal/video-generation/WorldForge_2025) | 完全 training-free 的推理时引导框架，通过 IRR（步内递归校正）、FLF（光流门控融合）、DSG（双路径自校正引导）三模块将精确轨迹控制注入预训练视频扩散模型，单图 3D 场景生成 FID 96.08、ATE 0.077 均 SOTA | Video Diffusion、3D/4D Generation、Training-Free、Trajectory Control、Inference-Time Guidance | 2025.09 |

### 🦾 Embodied AI — VLA 基础模型

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [3D-CAVLA](/papers/06-embodied-ai/vla/foundation/3D_CAVLA_2025) | 在 OpenVLA-OFT 上集成 CoT 叙事指令 + PointNet 轻量深度编码器 + 任务感知 ROI 检测，2D→3D 输入升级，LIBERO 双相机 98.1%，自建 LIBERO-Unseen 零样本基准上比 OpenVLA-OFT 绝对提升 8.8% | 3D 深度感知、CoT 指令分解、ROI 检测、零样本泛化、LIBERO | 2025.05 |
| [3D-MIX](/papers/06-embodied-ai/vla/foundation/3D_Mix_2026) | 系统对比 9 种 VGGT 融合策略，提出语义条件化门控融合即插即用模块，GR00T/π-style 双架构一致提升，9 个 GR00T-style 变体 SIMPLER 平均 +7.0% | VGGT 3D 融合、Gated Fusion、即插即用、SimplerEnv、LIBERO | 2026.03 |
| [AimBot](/papers/06-embodied-ai/vla/foundation/AimBot_2025) | 在 RGB 图像上叠加瞄准线（shooting line）和准星（scope reticle）编码 EE 位姿/朝向/夹爪状态到像素空间，<1 ms 无需改架构，π₀ 真实世界 27→43/50、LIBERO-Long +5.8，超越 TraceVLA/RoboPoint | 视觉空间线索、2.5D 空间编码、模型无关、深度可见性 | 2025.08 |
| [AnchorVLA4D](/papers/06-embodied-ai/vla/foundation/AnchorVLA4D_2026) | 首帧作为锚帧保留初始场景上下文 + 冻结 Any4D 空间编码器联合处理锚帧与当前帧提取 3D 几何特征，缓解遮挡遗忘和空间失定向，SimplerEnv 64.6%（+13.6%），真实世界 80% | 锚帧机制、冻结空间编码器、遮挡遗忘、早期重试 | 2026.03 |
| [BridgeVLA](/papers/06-embodied-ai/vla/foundation/BridgeVLA_2025) | 3D 点云正交投影为多视图 2D 图像 + 预测 2D 热力图对齐输入-输出，热力图预训练赋予 VLM 空间定位能力，RLBench 88.2%、3 条轨迹达 95.4% | 输入-输出对齐、2D 热力图、正交投影、样本效率 | 2025.06 |
| [ChatVLA](/papers/06-embodied-ai/vla/foundation/ChatVLA_2025) | 系统分析 VLA 的 spurious forgetting 与 task interference，Phased Alignment Training 先控制后理解 + MoE 双专家隔离 MLP，2B 参数 MMMU 37.4（ECoT 的 6 倍）、25 项真实任务超越 OpenVLA | Spurious Forgetting、MoE、Phased Alignment Training、多模态理解+控制统一 | 2025.02 |
| [CoWVLA](/papers/06-embodied-ai/vla/foundation/CoWVLA_2026) | Video VAE（VidTwin）显式解耦结构-运动潜变量，预训练推断潜在运动链 + 终端帧预测，协同微调联合建模关键帧与动作，统一世界模型时序推理与潜在动作紧凑性，LIBERO 95.6%、SimplerEnv 76.0%、CALVIN 4.21 | 结构-运动解耦、潜在运动链、Video VAE、Chain-of-World、终端帧预测 | 2026.03 |
| [CronusVLA](/papers/06-embodied-ai/vla/foundation/CronusVLA_2026) | 两阶段单帧预训练→多帧后训练：Feature Chunking 特征层聚合历史帧 + DiT 跨帧解码器 + 多帧正则化解耦骨干与时序建模，SimplerEnv 70.9% SOTA、LIBERO 97.0%，提出 SimplerEnv-OR 鲁棒性基准 R-Score 86.9 | Feature Chunking、多帧正则化、跨帧解码器、观测鲁棒性、SimplerEnv-OR | 2026 |
| [DAM-VLA](/papers/06-embodied-ai/vla/foundation/DAM_VLA_2026) | VLM 推理驱动动作路由选择手臂/夹爪专用扩散模型（class token 全局 + register token 局部）+ 双尺度加权协调训练，SIMPLER 平均 78-83%、真实世界 86.8% | 动作路由、双扩散头、class/register token、双尺度加权 | 2026.03 |
| [DeepVision-VLA](/papers/06-embodied-ai/vla/foundation/DeepVisionVLA_2026) | 诊断 VLA 深层视觉敏感性衰减（Grad-CAM + ROI 掩码 MSE），VL-MoT 将 DINOv3 视觉专家与深层 LLM 共享 QKV 注意力，AGVP 用浅层动作-视觉注意力图选取 Top-K 视觉 token，RLBench 83%（vs QwenVLA-OFT 69%）、真实世界 91.7%（vs π₀.₅ 84.2%） | 视觉敏感性衰减、Mixture-of-Transformers、DINOv3 视觉专家、动作引导 Token 剪枝 | 2026.03 |
| [DreamVLA](/papers/06-embodied-ai/vla/foundation/DreamVLA_2025) | 感知-预测-动作闭环：预测三类综合世界知识（动态区域/深度/语义）+ block-wise 结构化注意力防止跨类泄露 + DiT 动作头，CALVIN ABC-D 4.44 SOTA、LIBERO 92.6%、真实世界 76.7% | 世界知识预测、结构化注意力、DiT 动作头、CoTracker/Depth Anything/DINOv2 | 2025.07 |
| [Dexbotic](/papers/06-embodied-ai/vla/foundation/Dexbotic_2025) | 开源 VLA 工具箱：统一模块化框架（VLM + Action Expert）+ 基于 Qwen2.5 的更强预训练模型 + 实验驱动 Exp 脚本开发，SimplerEnv 最高 +46.2%、CALVIN +0.81 | VLA Toolbox、统一框架、DexboticVLM、实验驱动开发 | 2025.10 |
| [FAST](/papers/06-embodied-ai/vla/foundation/FAST_2025) | DCT + BPE 频域压缩 tokenization 解决自回归 VLA 高频灵巧任务训练瓶颈，π₀-FAST 匹配扩散 π₀ 性能但训练 5× 加速，FAST+ 通用 tokenizer 覆盖多构型 | DCT、BPE、动作 tokenization、高频控制、通用 tokenizer | 2025.01 |
| [FocusVLA](/papers/06-embodied-ai/vla/foundation/FocusVLA_2026) | 揭示自回归 VLA 视觉利用三大瓶颈（结构性捷径、信息过载、任务无关噪声），提出 Modality Cascaded Attention + Focus Attention（Patch-level 剪枝 + Channel-level 门控），0.5B 参数 LIBERO 98.7% 超越 7B 模型 | Cascaded Attention、视觉利用效率、Token 选择、门控机制 | 2026.03 |
| [FutureVLA](/papers/06-embodied-ai/vla/foundation/FutureVLA_2026) | 联合视觉-运动预测建模（JVPM）：3D-VAE 编码连续 17 帧 + 双流解耦监督 + 门控交叉注意力条件化交互，潜在嵌入对齐迁移时序先验，推理零开销，SimplerEnv 80.1%、真实机器人超 π₀ 达 26.7% | 联合视觉运动预测、双流解耦、门控交叉注意力、潜在对齐 | 2026.03 |
| [GR-3](/papers/06-embodied-ai/vla/foundation/GR3_2025) | 4B VLA（Qwen2.5-VL + Action DiT），机器人轨迹 + VL 数据协同训练实现 OOD 指令零样本泛化，VR 人类轨迹 10-shot 适配新物体（57.8%→86.7%），Task Status 辅助监督 + DiT RMSNorm 强化指令跟随，全面超越 π₀ | MoT 架构、VL 协同训练、人类轨迹少样本适配、Task Status、ByteMini 双臂移动 | 2025.07 |
| [MoH](/papers/06-embodied-ai/vla/foundation/MoH_2025) | 多 horizon 动作块在共享 Action Transformer 中并行处理 + 轻量门控融合（2k 参数）+ 跨 horizon 共识动态推理，plug-and-play 适用于 flow/regression 策略，$\pi_{0.5}$+MoH LIBERO 99% SOTA | 多 Horizon 融合、门控融合、动态推理、Action Chunking | 2025.11 |
| [MemoryVLA](/papers/06-embodied-ai/vla/foundation/MemoryVLA_2025) | 借鉴认知科学双记忆系统设计感知-认知记忆库（PCMB），同时存储低层视觉细节和高层语义，跨注意力检索 + 门控融合 + 合并压缩建模长时域依赖，SimplerEnv-Bridge +14.6、LIBERO 96.5%、真实世界时序任务 +26 | 感知-认知记忆、时序建模、扩散策略、长时域操作 | 2025.08 |
| [MMaDA-VLA](/papers/06-embodied-ai/vla/foundation/MMaDA_VLA_2026) | 基于原生离散扩散大模型 MMaDA-8B，将语言/图像/动作统一到离散 token 空间，并行去噪生成目标观测和动作块 + 混合注意力（模态内双向 + 模态间因果），LIBERO 98.0%、CALVIN 4.78 全面 SOTA | 原生离散扩散、统一多模态 token、并行去噪、混合注意力、目标观测生成 | 2026.03 |
| [OptimusVLA](/papers/06-embodied-ai/vla/foundation/OptimusVLA_2026) | 双记忆增强 VLA：GPM 用检索到的任务级先验替代高斯噪声缩短 flow 生成路径 + LCM 用 Mamba 建模动作历史注入时序一致性，LIBERO 98.6%、真实世界 2.9× 推理加速 | 双记忆、任务级先验检索、时序一致性、自适应 NFE | 2026.02 |
| [OTTER](/papers/06-embodied-ai/vla/foundation/OTTER_2025) | 冻结预训练 CLIP 编码器，利用 ClearCLIP 的 $X_{\text{attn}}$ 特征 + 余弦相似度 softmax 选择性提取与语言指令对齐的视觉 patch 特征，仅训练温度参数和轻量策略网络（~12M），真实机器人 4 种原语零样本泛化 77%，Octo/OpenVLA 几乎为 0% | 冻结 CLIP、文本感知视觉特征提取、ClearCLIP、零样本泛化 | 2025.03 |
| [ProgressVLA](/papers/06-embodied-ai/vla/foundation/ProgressVLA_2026) | OXE 预训练视觉-语言进度估计器（残差 0.07）+ 世界模型想象未来视觉状态 + 进度梯度 classifier guidance 引导扩散采样朝最大进度方向 + KL 正则化在线 RL 微调，CALVIN 3.73、LIBERO 84.5%、真实世界 76%（Octo 23%），步数减少 47% | 进度估计、Classifier Guidance、潜在动作空间、世界模型、扩散策略 | 2026.03 |
| [π₀](/papers/06-embodied-ai/vla/foundation/pi0_2024) | 用 Flow Matching 替代自回归生成动作，构建首个能完成高频灵巧操作的通用 VLA 基础模型 | Flow Matching VLA、Action Expert、跨构型预训练 | 2024.10 |
| [π₀.₅](/papers/06-embodied-ai/vla/foundation/pi05_2025) | 通过异构多源数据协同训练和分层推理，首次实现端到端 VLA 在全新家庭环境中执行长时域灵巧操作 | 异构协同训练、分层推理、开放世界泛化 | 2025.04 |
| [SF](/papers/06-embodied-ai/vla/foundation/SF_2025) | 将 VLA 中间层视觉 embedding 与 VGGT 3D 表征做余弦对齐，无需 3D 输入、推理零开销，LIBERO 98.5% 超越所有 2D/3D VLA，训练 3.8× 加速、数据 5.9× 高效 | 隐式空间对齐、VGGT、表征监督、训练/数据效率 | 2025.10 |
| [SpatialVLA](/papers/06-embodied-ai/vla/foundation/SpatialVLA_2025) | Ego3D 位置编码注入 3D 信息 + 自适应高斯动作网格仅需 3 token 表征 7D 动作，1.1M 数据预训练 3.5B 零样本超越 55B RT-2-X，20 Hz 推理 | 3D 空间感知、自适应动作离散化、跨构型预训练 | 2025.01 |
| [SPR](/papers/06-embodied-ai/vla/foundation/SPR_2026) | 将任务分解为带 2D 坐标的空间子目标序列，See-Plan 闭环进度监控 + Rewind 自主回退恢复，LIBERO 91.8%（+5% over MolmoAct），LIBERO-Plus OOD 仅 18.8% 退化（SOTA） | 进度感知、空间子目标、Rewind 错误恢复、OOD 鲁棒性 | 2026.03 |
| [TCoT](/papers/06-embodied-ai/vla/foundation/TCoT_2026) | 全局/局部层次轨迹作为 VLA 的中间任务规划层（CoT），GLSR 算法统一失败检测与全局-局部策略切换恢复，多任务促进跨任务知识共享，LIBERO 83.3%（Multi），真实世界 +28% 超越 OpenVLA | 轨迹思维链、层次规划、GLSR 失败恢复、跨任务知识共享 | 2026 |
| [TGM-VLA](/papers/06-embodied-ai/vla/foundation/TGM_VLA_2026) | 优化关键帧采样（存储 -85%、训练 5× 加速）+ 颜色反转投影分支 + 任务引导点云 Mixup，RLBench 90.5% SOTA、COLOSSEUM 68.8% | 关键帧采样优化、颜色反转、跨任务/任务内 Mixup、3D VLA | 2026.02 |
| [UniVLA](/papers/06-embodied-ai/vla/foundation/UniVLA_2025) | 语言引导两阶段解耦任务中心潜在动作（DINOv2 特征空间 + VQ-VAE 离散化），跨具身无标注视频预训练通才策略，1/20 算力超越 OpenVLA，LIBERO 95.2%、CALVIN 3.80、R2R 47.1%、真实世界 81.7% | 任务中心潜在动作、跨具身无标注预训练、VQ-VAE、DINOv2 | 2025.05 |
| [VP-VLA](/papers/06-embodied-ai/vla/foundation/VP_VLA_2026) | 双系统解耦架构，VLM 事件驱动任务分解 + SAM3 生成十字准星/边框视觉提示作为空间锚点，辅以视觉接地损失，RoboCasa +5.0%、SimplerEnv +8.3%，超越 $\pi_{0.5}$ 和 GR00T-N1.6 | 视觉提示、双系统架构、事件驱动分解、空间接地、OOD 泛化 | 2026.03 |

### 🦾 Embodied AI — VLA 高效推理

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [BitVLA](/papers/06-embodied-ai/vla/efficient/BitVLA_2025) | 首个全参数三值化 VLA，BitNet b1.58 LLM + 蒸馏感知训练将 ViT 量化至 1.58-bit，无需大规模机器人预训练 LIBERO 平均 94.8%（匹配 OpenVLA-OFT INT4），显存仅 1.4GB（29.8%） | 1-bit 量化、蒸馏感知训练、三值化、边端部署 | 2025.06 |
| [EfficientVLA](/papers/06-embodied-ai/vla/efficient/EfficientVLA_2025) | 结构化 training-free 加速：LLM 层剪枝 + 任务感知视觉 token 选择 + 扩散步缓存，三维度协同消除冗余，FLOPs 降至 28.9%、1.93× 加速 | LLM 层剪枝、任务感知 Token 选择、扩散步缓存、Training-Free、1.93× 加速 | 2025.06 |
| [HeiSD](/papers/06-embodied-ai/vla/efficient/HeiSD_2026) | 混合推测解码（Drafter SD + Retrieval SD），运动学融合指标自动切换，自适应验证跳过 + 序列级宽松接受，LIBERO 最高 2.45× 加速，真实世界 2.06×-2.41× | 混合推测解码、运动学感知、Verify-Skip、序列级宽松接受、2.45× 加速 | 2026.03 |
| [LAC](/papers/06-embodied-ai/vla/efficient/LAC_2026) | 可学习自适应 Token 缓存，光流运动先验 + Gumbel-Softmax 端到端优化，1.76× 加速 | 可学习 Token 缓存、光流运动先验、1.76× 加速 | 2026.01 |
| [PD-VLA](/papers/06-embodied-ai/vla/efficient/PD_VLA_2025) | AR 解码重建为非线性方程组，Jacobi 不动点并行迭代，不修改模型不训练，action chunking VLA 2.52× 频率提升 | Jacobi 并行解码、Action Chunking、Training-Free、2.52× 加速 | 2025.03 |
| [SD-VLA](/papers/06-embodied-ai/vla/efficient/SD_VLA_2026) | 静态-动态 Token 解耦 + 多级缓存层次 + 可学习重缓存门，长时程 VLA 2.26× 加速 | 静态-动态解耦、多级缓存、2.26× 加速 | 2026.02 |
| [RLRC](/papers/06-embodied-ai/vla/efficient/RLRC_2025) | 三阶段 VLA 压缩流水线（结构化剪枝 + SFT/RL 恢复 + 量化），8× 显存压缩、2.3× 加速 | 结构化剪枝、RL 恢复、量化、8× 压缩 | 2025.06 |
| [RTC](/papers/06-embodied-ai/vla/efficient/RTC_2025) | 将动作块异步执行建模为修复问题：冻结已执行前缀 + ΠGDM 引导修复 + 软掩码指数衰减约束跨块一致性，无需重训练即可让 flow-based VLA 实时执行，π₀.₅ 真实世界比同步推理快 20%，300ms+ 延迟下吞吐量无下降 | 异步动作块执行、Flow Inpainting、软掩码、Training-Free | 2025.06 |
| [VLA-Cache](/papers/06-embodied-ai/vla/efficient/VLA_Cache_2025) | 训练无关跨帧 Token 缓存 + 注意力驱动任务相关性过滤，1.7× 加速 | 跨帧 Token 缓存、注意力过滤、1.7× 加速 | 2025.02 |
| [VLA-Pruner](/papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025) | 双层重要性准则（语义级 prefill + 动作级 decode 注意力时序平滑）+ mRMR 双层选择策略，50% 剪枝率反超原模型，87.5% 剪枝率保持 88.9% 准确率 | 双层 Token 剪枝、时序平滑、mRMR 选择、Training-Free、1.8× 加速 | 2025.11 |

### 🦾 Embodied AI — VLA 推理增强

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [UAOR](/papers/06-embodied-ai/vla/inference/UAOR_2026) | 用 Action Entropy 检测高不确定性层，通过注意力检索将观测特征重注入 FFN，无训练即插即用一致提升多种 VLA | Action Entropy、观测重注入、FFN-as-Memory、Training-Free | 2026.02 |

### 🦾 Embodied AI — VLA / RL 后训练

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [ARM](/papers/06-embodied-ai/vla/rl/ARM_2026) | 相对优势奖励建模：Tri-state（Progressing/Regressing/Stagnant）轻量标注 + MIMO Transformer 双头（区间分类 + 任务完成）+ 长度自适应 AW-BC，长程叠毛巾 99.4% 成功率 | Tri-state 标注、MIMO Transformer、AW-BC、长程操作、99.4% 成功率 | 2026.04 |
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
| [πRL](/papers/06-embodied-ai/vla/rl/piRL_2025) | 首个 flow-based VLA 在线 RL 框架，Flow-Noise（可学习噪声链联合似然）和 Flow-SDE（ODE→SDE 两层 MDP + 混合采样），PPO 微调 π₀/π₀.₅，LIBERO 57.6→97.6%/77.1→98.3%，ManiSkill 4352 组合 38.4→78.8% | Flow-Noise、Flow-SDE、PPO、log-likelihood 估计、ODE-SDE 转换 | 2025.11 |
| [PLD](/papers/06-embodied-ai/vla/rl/PLD_2026) | 冻结 VLA 主干训练轻量残差 RL 专家探索失败区域，基础策略探针 + 混合轨迹蒸馏实现 VLA 自改进，LIBERO 达 99% 成功率 | 残差 RL、基础策略探针、混合数据蒸馏、VLA 自改进 | 2026.01 |
| [PTR](/papers/06-embodied-ai/vla/rl/PTR_2026) | 无奖励保守离线后训练：post-action identification posterior 评分 + 指数化裁剪权重重缩放 SFT 损失，兼容 diffusion/flow 动作头，三构型 12 任务 Generalist +13.8 pp | Posterior-Transition Reweighting、InfoNCE identification、保守加权、跨构型迁移 | 2026.03 |
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
| [TOPReward](/papers/06-embodied-ai/vla/rl/TOPReward_2026) | 绕过 VLM 文本生成，直接从 token logits 提取 True 概率作为零样本任务进度信号，Qwen3-VL-8B 达 0.947 VOC 大幅超越 GVL，提出 130+ 真实任务的 ManiRewardBench | Token 概率奖励、零样本进度估计、ManiRewardBench、VOC | 2026.02 |
| [TwinRL](/papers/06-embodied-ai/vla/rl/TwinRL_2026) | 用高保真数字孪生作为探索放大器和引导器，三阶段流程（探索空间扩展 + 孪生在线 RL + sim-to-real 引导）四任务平均 20 分钟逼近 100% 成功率 | 数字孪生、探索空间扩展、Sim-to-Real 引导、HiL | 2026.02 |
| [ViVa](/papers/06-embodied-ai/vla/rl/ViVa_2026) | 把预训练视频扩散 DiT（Wan2.2）当价值模型用，通过 latent injection 联合预测未来本体感知 + 当前标量价值，替换 RECAP 的 VLM value 后 box assembly 成功率 58%→73% | 视频生成价值模型、Wan2.2、Latent Injection、RECAP、未来本体感知 | 2026.04 |
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
| [Fast-WAM](/papers/06-embodied-ai/world-models/FastWAM_2026) | 通过受控变体实验拆解 WAM 的两个因素，证明训练时视频协同目标（而非测试时未来想象）是性能主因；提出跳过测试时视频生成、单次前向传播动作预测，190 ms 延迟、RoboTwin 91.8%、LIBERO 97.6%，无具身预训练 | 视频协同训练、MoT 架构、训练-推理解耦、Flow Matching、测试时加速 | 2026.03 |
| [Kinema4D](/papers/06-embodied-ai/world-models/Kinema4D_2026) | 将仿真解耦为运动学确定性 4D 机器人轨迹（URDF + FK/IK → pointmap）和生成式环境响应（DiT 联合预测 RGB+pointmap），Robo4D-200k 训练，PSNR 22.50、FVD 98.5、F-Score 0.4733，首次零样本真实世界迁移 | 4D Pointmap 控制、运动学-生成解耦、联合 RGB+Pointmap 合成、构型无关、零样本迁移 | 2026.03 |
| [WorldVLA](/papers/06-embodied-ai/world-models/WorldVLA_2025) | 基于 Chameleon 将 VLA 动作模型与世界模型统一到单个自回归框架，共享权重混合训练实现双向增强，提出 Action Attention Mask 阻断 Action Chunking 误差累积 | 自回归统一模型、Action Attention Mask、Chameleon、双向增强 | 2025 |

### 🦾 Embodied AI — Imitation Learning

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [EC-Flow](/papers/06-embodied-ai/imitation-learning/EC_Flow_2025) | 将光流预测从物体中心转为具身中心（预测机器人上采样点轨迹），配合目标图像辅助对齐和 URDF 感知运动学动作计算，仅用 5 条无动作标注 RGB 视频学习操作策略，在遮挡（+62%）、柔性物体（+45%）和非位移操作（+80%）场景大幅超越物体中心方法 | 具身中心光流、目标图像对齐、URDF 运动学、无动作标注、DiT 扩散 | 2025.07 |

### 📊 Evaluation & Benchmarks

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [MME](/papers/09-evaluation-and-benchmarks/MME_2024) | 首个 MLLM 综合评测基准：14 子任务覆盖感知（粗/细粒度识别、OCR）与认知（常识推理、计算、翻译、代码），Yes/No 人工指令对 + ACC/ACC+ 双指标实现精确量化，30 模型评测揭示指令跟随失败、感知缺陷、推理断裂、目标幻觉四大共性问题 | MLLM 评测、感知与认知、Yes/No 指令、手工标注、14 子任务、30 模型对比 | 2024.03 |

### 🎯 Reinforcement Learning

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [DiffusionNFT](/papers/10-reinforcement-learning/DiffusionNFT_2025) | 在前向加噪过程上做扩散模型在线 RL：正/负样本对比定义隐式策略改进方向，嵌入 flow matching 目标，无需似然估计/CFG/特定采样器，效率比 FlowGRPO 高 3-25 倍，SD3.5-M GenEval 0.24→0.98 | 前向过程 RL、Negative-aware Fine-Tuning、隐式引导集成、CFG-Free、Flow Matching | 2025.09 |
| [FLAC](/papers/10-reinforcement-learning/FLAC_2026) | 将 MaxEnt RL 建模为 Generalized Schrödinger Bridge 问题，用速度场动能作为无似然的熵代理，上界终端分布散度，NFE=2 超越 DIME（NFE=16） | Generalized Schrödinger Bridge、Kinetic Energy、Flow/Diffusion 策略、无似然 MaxEnt RL | 2026.02 |

