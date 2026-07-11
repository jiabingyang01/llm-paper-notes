# MolmoAct：能够在空间中推理的动作推理模型

> **论文**：*MolmoAct: Action Reasoning Models that can Reason in Space*
>
> **作者**：Jason Lee、Jiafei Duan、Haoquan Fang（三人共同一作，排名不分先后）et al.（含 Dieter Fox、Ali Farhadi、Ranjay Krishna 等研究顾问，Ranjay Krishna 为项目 PI）
>
> **机构**：Allen Institute for AI (Ai2)、University of Washington
>
> **发布时间**：2025 年 08 月（arXiv 2508.07917，v4 于 2025 年 9 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2508.07917) | [PDF](https://arxiv.org/pdf/2508.07917)
>
> **分类标签**：`VLA` `动作推理模型` `空间推理` `深度感知token` `视觉推理轨迹` `可操控性steering`

---

## 一句话总结

MolmoAct 把 VLA 的"感知→规划→控制"显式拆成三段可独立解码、可视化的自回归推理链——深度感知 token、2D 视觉推理轨迹、动作 token——用仅 2630 万样本预训练（不到 π0 预训练数据量的十分之一），就在 SimplerEnv（71.6%）、LIBERO（86.6%）和真实单臂/双臂 Franka 任务（较 π0-FAST 分别提升 10%/22.7% 的任务进度）上全面超越 π0、π0-FAST、GR00T N1.5、ThinkAct 等基线,并首次支持用户直接在相机图像上涂画轨迹线来实时 steering 机器人动作。

## 一、问题与动机

- 大语言模型和视觉语言模型的能力提升在机器人领域并未同步兑现：现有 VLA（π0、OpenVLA、RT-2 等）虽然共享 VLM backbone，但依然脆弱、不透明——难以解释为何在某一状态下选择了某个动作，也难以跨任务、跨场景、跨具身泛化。
- 作者认为差距不只是数据量不够，而是**缺乏结构**：语言/视觉任务可以吃到网络规模的松散标注数据,而机器人交互数据昂贵、歧义大、难以规模化；与此同时语言模型已经从"暴力scaling"转向"结构化学习"（CoT、STaR 等），机器人领域应当效仿。
- 已有的"先思考后行动"工作（ECoT、CoT-VLA 等）大多只做**高层语言推理**（把长程任务拆解为语言子任务），忽视了两个关键点：深度感知与精确的运动规划。把复杂的 3D 轨迹硬蒸馏成语言描述,往往损失掉空间和时序信息。
- MolmoAct 的核心主张：不通过语言做中间推理,而是让模型**在空间中推理**——先预测深度感知 token 重建 2.5D 场景理解,再预测图像平面上的 2D 视觉推理轨迹作为中层运动规划,最后才落到底层动作 token,每一步都能被独立解码、可视化,从而兼具可解释性和可操控性（steerability）。

## 二、核心方法

### 2.1 VLM 骨干与动作离散化

MolmoAct 复用开源 VLM **Molmo**（ViT 视觉编码器 + 两层 MLP 连接器 + decoder-only LLM）作为骨干,提供两个变体：MolmoAct-7B-D（SigLIP2 ViT-SO400M/14 384px + Qwen2.5-7B,主力/最佳模型）与 MolmoAct-7B-O（OpenAI CLIP ViT-L/14 336px + OLMo2-7B,训练数据/权重/代码全部公开的"最开放"版本）。

动作预测沿用先前工作的做法,把每个动作维度按数据集分位数离散化为 256 个等宽 bin（取 1–99 百分位以降低离群值影响）。区别于以往从词表尾部随机取 256 个语言 token 表示这些 bin（这些 token 之间彼此在语义上无关,是较差的初始化）,MolmoAct 取 Qwen2 tokenizer 最后 256 个 token 对应的 byte-level BPE 符号,并**按大小单调地** 把它们映射到 256 个 bin,使相邻 bin 对应字符相近的 token——这种保序初始化让动作 codebook 的优化起点更平滑,实践中显著减少了训练时间。

### 2.2 三阶段动作推理链（Action Reasoning）

**深度感知 token。** 定义辅助词表

$$V_{\text{depth}} = \{\langle \text{DEPTH\_START} \rangle, \langle \text{DEPTH\_END} \rangle\} \cup \{\langle \text{DEPTH}\_k \rangle\}_{k=1}^{N}, \quad N=128$$

每张输入图像对应的目标深度串为

$$\mathbf{d} = \big(\langle \text{DEPTH\_START} \rangle\ \langle \text{DEPTH}\_z_1^{\text{depth}} \rangle\ \cdots\ \langle \text{DEPTH}\_z_M^{\text{depth}} \rangle\ \langle \text{DEPTH\_END} \rangle\big) \in V_{\text{depth}}$$

其中 $M=100$，每个 $z_i^{\text{depth}} \in \{1,\ldots,N\}$ 是 VQVAE 码本中的索引。用大白话说：先单独训练一个"深度专家"——把 Depth Anything V2 在 1000 万张桌面操作图像上生成的深度图,用 VQVAE 压成固定 128 维码本、每张图 100 个离散 token；再把这串深度 token 当作监督标签,让 VLA 直接从原始 RGB 自回归地"背"出这串深度描述,相当于把专家模型的 3D 感知能力**蒸馏** 进通才模型,无需真实深度传感器。

**视觉推理轨迹（Visual Reasoning Trace）。** 定义末端执行器在图像上的运动轨迹为一条折线

$$\boldsymbol{\tau} = (p_1,\ldots,p_L),\quad 1\le L\le 5,\quad p_i=(u_i,v_i)\in\{0,\ldots,255\}^2$$

$p_1$ 是末端执行器在当前帧的位置,其余点从未来帧中沿时间均匀采样直到episode终止帧。用大白话说：与其用语言描述"先抓起来再放到左边",不如直接在图像上画出 1–4 段折线勾出手爪未来会怎么走,这比语言指令更精确、也更容易被模型和人类同时理解。真实轨迹标签并非人工标注,而是用 Molmo 自带的 2D pointing 能力对每一帧提示"point to the robot gripper"自动生成（双臂场景左右手分别问）,这是一种类似 NLP 中"用现成模型自动生成弱监督数据"的策略。

**完整推理过程。** 给定图像观测 $I$ 与语言指令 $T$,模型按顺序自回归生成深度 token 序列 $\mathbf{d}$、视觉推理轨迹 $\boldsymbol{\tau}$、动作 token $\mathbf{a}=(a_1,\ldots,a_D)$,联合分布分解为

$$p(\mathbf{d}, \boldsymbol{\tau}, \mathbf{a} \mid I, T) = \prod_{i=1}^{M+2} p\big(d_i \mid I, T, \mathbf{d}_{\lt i}\big) \times \prod_{j=1}^{L} p\big(\tau_j \mid I, T, \mathbf{d}, \boldsymbol{\tau}_{\lt j}\big) \times \prod_{k=1}^{D} p\big(a_k \mid I, T, \mathbf{d}, \boldsymbol{\tau}, \mathbf{a}_{\lt k}\big)$$

用大白话说：模型先"想清楚场景的 3D 结构",再"规划手怎么走",最后才"决定具体的关节/末端指令",每一段都以前一段为条件,保证最终动作在空间上是有依据的。

### 2.3 通过轨迹涂鸦实现的可操控性（Steerability）

作者认为纯语言 steering 有三个问题：(i) 需要大量高质量的语言-动作配对数据才能学到可靠的 grounding；(ii) 自然语言在描述幅度、尺度、终点时天生模糊；(iii) 微调后的模型往往对分布外措辞很脆弱。MolmoAct 转而允许用户直接在相机图像上画一条轨迹 $\boldsymbol{\tau}$,与原图叠加得到增强观测 $I^+ = I \oplus \boldsymbol{\tau}$,测试时以此为条件自回归生成下一步动作：

$$p(\mathbf{a} \mid I^+, T) = \prod_{k=1}^{D} p\big(a_k \mid I^+, T, \mathbf{a}_{\lt k}\big)$$

在每个时间步重复这一过程即可实现精确、可编辑、对分布外措辞更鲁棒的闭环交互式 steering。

### 2.4 训练配方与数据

三阶段训练：**预训练**（OXE 子集 RT-1/BridgeData V2/BC-Z 转成动作推理格式，共 10.5M 样本，加上辅助深度数据 1.5M、辅助轨迹数据 1.5M、轨迹条件动作数据 10.5M、以及 2M 多模态网络数据，总计 26.3M 样本；256×H100，100k 步，batch 512，约 9,728 GPU 小时——相比 GR00T N1.5 的 50,000 GPU 小时降低 5 倍以上）→ **中训练**（在自建 MolmoAct Dataset 上用 1M 动作推理样本 + 1M 轨迹条件动作样本，128×H100，50k 步，2,304 GPU 小时，侧面视角生成推理链、腕部视角仅作辅助信息）→ **后训练**（每个新任务采集 30–50 条遥操作演示,用 rank=32/alpha=16 的 LoRA 做参数高效微调，并引入动作分块 chunk size $N=8$）。

**MolmoAct Dataset**（中训练用,自建、随论文开源）：10,689 条单臂 Franka 轨迹，93 个独立操作任务，覆盖家庭环境（7,730 条，73 个任务，20 个动词，厨房/浴室/卧室/客厅）与桌面环境（2,959 条，20 个原子任务），平均轨迹长度 112 步，由 5 名全职操作员历时两个月采集。

## 三、实验结果

**SimplerEnv（Google Robot，Table 1，均值列）**

| 模型 | Visual Matching Avg | Variant Aggregation Avg |
|---|---|---|
| RT-2-X | 53.4% | 64.3% |
| π0（微调） | 58.7% | 54.8% |
| π0-FAST（微调） | 61.9% | 59.0% |
| GR00T N1.5（微调） | 52.4% | 43.7% |
| Magma | 68.4% | 62.6% |
| MolmoAct-7B-D-Pretrain（零样本） | 70.5% | 59.3% |
| **MolmoAct-7B-D（微调）** | **71.6%** | **72.1%** |

MolmoAct-7B-D-Pretrain 仅用约 2630 万样本预训练（比 π0 系列所用的至少 9.03 亿样本小一个数量级）即取得 70.5% 的零样本 Visual Matching 成功率,超过 GR00T N1.5、π0、π0-FAST、Magma 等闭源/开源基线。

**LIBERO（Table 2，四类任务成功率）**

| 模型 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.5 | 69.0 | 83.9 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| **MolmoAct-7B-D** | 87.0 | 95.4 | 87.6 | **77.2** | **86.6** |

MolmoAct 在四类任务上均值最高，尤其在长程（Long-horizon）任务上比第二名 ThinkAct 高 6.3 个百分点。

**真实机器人后训练（Figure 5，25 trials/任务，任务进度分，节选自附录逐次表）**

| 任务 | 手臂配置 | MolmoAct | π0-FAST | OpenVLA |
|---|---|---|---|---|
| Put Bowl in Sink | 单臂 | 0.826 | 0.708 | 0.250 |
| Wipe Table | 单臂 | 1.000 | 0.817 | 0.265 |
| Fold Towel | 双臂 | 0.80 | 0.52 | 0.32 |
| Lift Tray | 双臂 | 1.00 | 0.74 | 1.00 |
| Set Table | 双臂 | 0.77 | 0.24 | 0.30 |

整体上 MolmoAct 后训练在单臂任务上平均超过 π0-FAST 10 个百分点，在双臂任务上超过 22.7 个百分点。

**分布外泛化（真实世界，Table 21 四类扰动的算术平均，据原文数据汇总）**

| 类别 | OpenVLA | π0-FAST | MolmoAct |
|---|---|---|---|
| In-Distribution | 0.375 | 0.646 | 0.792 |
| Language Variation | 0.229 | 0.292 | 0.708 |
| Spatial Variation | 0.396 | 0.458 | 0.542 |
| Distractors | 0.292 | 0.542 | 0.750 |
| Novel Objects | 0.292 | 0.292 | 0.646 |

原文摘要将其总结为：MolmoAct 相较 π0-FAST 平均任务进度提升 23.3%。此外，中训练消融显示 MolmoAct-7B-D（含 MolmoAct Dataset 中训练）相较不做中训练的版本平均高出约 5.5%（close_lid / rotate_pot / pour_tea 三任务），且即便不做中训练也已分别超过 π0-FAST 和 OpenVLA 14.8% 和 10.9%。

**语言指令跟随与轨迹 steering 的人类偏好评估**

- 开放式指令跟随（SimplerEnv，100 名标注者、1,500+ 票）：MolmoAct-7B-D-Pretrain 的 Elo 评分超过 SpatialVLA 109 分,两两胜率对 SpatialVLA 为 58%、对 OpenVLA 为 81%。
- 轨迹生成人类偏好（Line Steerability，Figure 7）：MolmoAct 的 Elo 显著高于 Gemini-2.5-Flash、GPT-4o、专门做轨迹生成的 HAMSTER（95% 置信区间不重叠）。
- 模糊语言场景下的 steering 成功率（pick_up_bowl 任务，Figure 9）：视觉轨迹 steering（MolmoAct）0.75，开放式语言 steering（MolmoAct）0.42，开放式语言 steering（π0-FAST）0.13——即视觉轨迹比语言指令高 33 个百分点，MolmoAct 的语言 steering 比 π0-FAST 高 29 个百分点。

## 四、局限性

论文附录 G 明确列出以下局限与潜在改进方向：

1. **末端执行器遮挡问题**：后训练阶段空间推理主要依赖前视相机,若末端在前视画面中被遮挡,视觉轨迹预测（进而整体性能）会退化。作者建议改用广角/鱼眼相机配合 SLAM,把纯空间推理换成时序推理来缓解。
2. **轨迹 steering 的鲁棒性依赖两点**：(i) 预训练/中训练阶段轨迹标注的质量与多样性——基于检测框（如 Detectron）的轨迹标注点会向框中心塌缩、跨具身迁移差,而基于 VLM point 标注（如 Molmo、RoboPoint）更准确、非退化；(ii) 后训练数据需覆盖足够丰富的动作组合,否则模型学不到图像空间轨迹与实际动作之间的丰富对应关系。
3. **轨迹 steering 只在 2D 图像平面上生效**：由于线索是纯 2D 的，模型缺乏显式的深度概念，在图像平面内的运动（in-plane）跟随较好，但沿相机深度轴（out-of-plane）的运动容易出现无意的偏差。作者提出未来可复用/条件化已预测的深度感知 token 把轨迹提升到 3D 来缓解，但本文尚未实现。
4. **动作推理链的推理速度**：与许多现有 VLA 类似,模型的控制推理频率与数据采集时的控制频率存在差距,部分原因是服务器-机器人通信延迟以及预测更多推理 token 带来的额外时间开销。
5. **深度感知 token 的精度有限**：沿用 Bigverdi et al. (2025) 的做法固定用 100 个 token 表示每张图的深度,而精细操作可能需要更高分辨率的深度估计；增加深度 token 数量是潜在改进方向。

## 五、评价与展望

**优点。** MolmoAct 最突出的贡献是把"空间推理"做成了三段可独立解码、可视化的显式表示（深度 token / 2D 轨迹 / 动作 token），而不是像 ECoT、CoT-VLA、RAD、ThinkAct 那样把推理压缩进难以 grounding 的语言子目标或潜变量；相比同样做"轨迹式中层表示"的 Emma-X（仅在预测的手爪 2D 位置上推理、未利用完整 3D 场景上下文）和 HAMSTER（轨迹 steering 局限于高层 VLM 输出、执行侧仍由针对固定任务集训练的独立低层策略负责，语义泛化性受限），MolmoAct 用同一个端到端模型完成感知-规划-执行,并将 steering 泛化到新的空间配置、未见物体乃至模糊语言指令上。此外论文在预训练数据规模远小于 π0/GR00T N1.5（相差一个数量级）、GPU 训练小时数减少 5 倍以上的情况下仍取得更优或相当的结果，是结构化中间监督换取样本效率的一个有说服力的证据；全量开源模型权重、训练代码及三类中间监督数据集（含 MolmoAct Dataset）也提升了该工作对社区的可复现价值,这在当前多闭源 VLA（π0、GR00T）的生态里是稀缺的。

**局限与开放问题。** (1) 深度感知 token 依赖单一深度专家（Depth Anything V2）在有限桌面操作场景上蒸馏得到的伪标签,其几何精度和对形变/透明/反光物体的鲁棒性未被单独评估，蒸馏误差是否会在长程任务中累积尚不清楚；(2) 视觉推理轨迹标签本身来自 Molmo 的 2D pointing 能力自动生成，其准确性依赖上游 VLM 的 grounding 质量，论文未报告这一自动标注管线的误差率或对下游 steering 精度的影响；(3) 正如作者自陈，轨迹 steering 停留在图像平面，深度轴上的可控性仍是开放问题，是否可以像深度 token 一样把轨迹升级为显式 3D 关键点序列、并联合监督，是一个自然的后续方向；(4) 评测主要集中在 Franka 单臂/双臂和 SimplerEnv 的 Google Robot/WidowX，对更多样的形态（移动底盘、人形上肢）尚未验证结构化空间推理是否同样有效；(5) 三段式自回归推理增加了每步的 token 生成量,论文承认存在推理频率与数据采集频率不匹配的问题,如何在不牺牲显式推理带来的可解释性/可控性前提下压缩推理延迟，是走向真实部署的关键工程问题。总体而言，MolmoAct 为"结构化空间推理换取样本效率与可解释性/可操控性"这一路线提供了目前较完整的开源实现和消融证据，其数据自动标注（VLM point 生成轨迹、深度专家蒸馏）与图像涂鸦式 steering 的思路对后续 VLA 数据构建和人机交互接口设计具有参考价值。

## 参考

- Black, K. et al. π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164, 2024.
- NVIDIA et al. GR00T N1.5: An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734, 2025.
- Huang, C.-P. et al. ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning. arXiv:2507.16815, 2025.
- Zhao, Q. et al. CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models. CVPR, 2025.
- Sun, Q. et al. Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning. arXiv:2412.11974, 2024.
- Li, Y. et al. HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation. arXiv:2502.05485, 2025.
- Deitke, M. et al. Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models. arXiv:2409.17146, 2024.
