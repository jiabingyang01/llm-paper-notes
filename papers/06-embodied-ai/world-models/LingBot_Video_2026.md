# LingBot-Video：面向具身智能的稀疏混合专家视频预训练

> **论文**：*Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence*
>
> **作者**：Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang（共同一作）, Chaoran Feng, Zijing Hu, Chong Bao, Nan Xue, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, Ka Leong Cheng（Project Lead）et al.
>
> **机构**：Ant Group（论文封面标注机构 logo 为 ANT GROUP，产品线品牌 Robbyant，项目主页 technology.robbyant.com/lingbot-video）
>
> **发布时间**：2026 年 07 月（arXiv 2607.07675）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.07675) | [PDF](https://arxiv.org/pdf/2607.07675)
>
> **分类标签**：`视频基础模型` `Mixture-of-Experts` `Diffusion Transformer` `具身世界模型` `RLHF后训练` `动作条件生成`

---

## 一句话总结

LingBot-Video 是首个面向具身智能开源的大规模 MoE 视频基础模型：在单流 DiT 架构上用 DeepSeekMoE 式细粒度稀疏专家 + 无辅助损失负载均衡实现容量-算力解耦，配合 70,000+ 小时具身数据注入与六维度奖励的 GRPO/RealNFT 后训练，其 RBench 平均分 0.620 超过对比中的全部开源模型，也超过 Wan 2.6（0.607）、Seedance 1.5 pro（0.584）、Veo 3（0.563）等闭源商业模型。

## 一、问题与动机

现有视频生成大模型（扩散或自回归）主要为内容创作优化，追求视觉保真度与创意，而非物理正确性与可控性：它们不显式约束接触稳定性、刚体动力学或长时程状态一致性下的物理交互。这使得视频基础模型难以直接迁移为具身智能所需的"世界模拟器"（支持策略评估、想象式规划、机器人数据合成）。作者指出现有工作在三个耦合维度上存在短板：

- **架构**：主流 dense DiT 对所有 token/时间步统一激活全部参数，推理成本随分辨率/时长急剧上升，扩展性受限；LLM 领域已证明的稀疏 MoE 在视频生成中应用有限。
- **数据**：训练语料以互联网视频为主，缺乏机器人本体先验与精确交互动力学，物理接地弱。
- **训练目标**：现有对齐策略主要优化美学质量与文本-视频对应关系，未显式纳入物理合理性、任务完成度、长时程一致性等信号。

LingBot-Video 试图同时解决这三点，将视频生成模型定位为具身智能的"数据引擎、策略评估器、动作规划器"三重角色。

## 二、核心方法

### 1. 任务统一单流 Diffusion Transformer + 稀疏 MoE

架构采用级联设计：base generator（低分辨率）+ refiner（超分）。base generator 是单流 DiT，将视觉 latent patch 与多模态条件 token（Qwen3-VL-4B 提取，Wan2.1-VAE 压缩视觉 latent）拼接为统一 token 序列，用 Multi-Modal 3D RoPE 区分条件与视觉 token 的时空坐标，配合 QK-Norm、AdaLN-Single 时间步调制。

MoE 层替换单流块中的 dense FFN 分支，token 级路由到共享专家（始终激活）与 top-$K_r$ 路由专家：

$$m(\mathbf{u}_t) = \sum_{i=1}^{N_s} E_i^{(s)}(\mathbf{u}_t) + \sum_{j \in \mathcal{R}_b(\mathbf{u}_t)} g_{t,j} E_j^{(r)}(\mathbf{u}_t)$$

用大白话说：每个 token 除了走一条"通用物理常识"的共享专家路径外,还会挑几个"专精"路由专家一起处理,总参数量可以做得很大但每个 token 实际算力开销不变。

路由亲和度用 sigmoid 而非 softmax，并采用 DeepSeek 式分组限制路由与在线纠偏（bias-free load balancing）：

$$\alpha_{t,j} = \text{Sigmoid}(\mathbf{u}_t^\top \mathbf{r}_j), \qquad b_j \leftarrow b_j - \eta\, \text{sign}(n_j - \bar{n})$$

用大白话说：每步训练结束后,给"太忙"的专家降权、"太闲"的专家加权,不需要额外损失函数就能让专家负载均衡，避免路由塌缩到少数专家。此外还引入 DeepSeek-V3 式**序列级**辅助均衡损失，防止批级统计掩盖单个长视频序列内部的路由不均衡。

架构消融（固定 1.4B 激活参数）表明：专家总数从 64→128 收益明显，128→256 收益边际，故选 $E=128$ 为默认；在固定 13B 总参数下，细粒度路由（top-8/128）稳定优于粗粒度路由（top-4/64），验证了 DeepSeekMoE 细粒度专家分割思路在视频扩散上同样成立。

### 2. Cascaded Refiner：条件 rectified flow 而非从纯噪声去噪

Refiner 不从高斯噪声起步，而是从退化的低分辨率条件 latent 出发学习一段有限区间的整流流轨迹（$t \in [0,\tau]$，$\tau \sim \text{Uniform}(0.85, 0.95)$），把去噪能力集中在高频细节修复而非全局语义/运动重建，降低算力开销的同时避免破坏 base 阶段已确定的运动与语义。

### 3. 数据体系：Data Profiling Engine + World-Knowledge Topological Graph

自建数据画像引擎沿结构、语义、运动、镜头、质量五维标注每条样本；用语义树（50,000 细粒度叶概念、经 LLM Discover-Classify-Consolidate 归并为 25 个顶层簇）与动作树组织长尾覆盖，作为分布感知采样与"难例上采样、饱和易例降采样"的控制面。Stage 2 阶段注入超过 **70,000 小时**具身导向素材（真机/仿真/开源、人形与四足、导航、第一人称）。密集结构化 caption（借鉴 FIBO）为图像/视频/VLA/第一人称四类数据统一 JSON schema，并配二阶段 Caption Rewriter（Expand→Map）弥合训练-推理 prompt 分布差异。五阶段渐进训练课程按 192p→192p→480p→480p→1080p 分辨率与数据源比例逐步演化。

### 4. 六维度奖励 + GRPO/RealNFT 后训练

后训练用六个专用奖励模型（Vision Quality/HPSv3、Text-Video Alignment/时序 VQA、Dynamic Degree、Motion Coherence、Human-Motion Consistency、Physical Plausibility）分别打分并按组内归一化融合优势：

$$\hat{A}^{(i)} = \sum_r w_r \frac{R_r(\mathbf{x}_0^{(i)}, c) - \mu_r}{\sigma_r + \delta}$$

GRPO 沿用 Flash-GRPO 的"单步随机探索"范式（每个 rollout 组仅在一个共享的去噪步注入随机性），配合 Coefficients-Preserving Sampling（DDIM 式而非 SDE 转换）避免噪声爆炸，并按 transition gain 的倒数对不同时间步的策略梯度重新加权，训练严格 on-policy、无 KL 惩罚。另设计 Negative-Aware Finetuning（基于 DiffusionNFT 思路）：真实视频作为 chosen、模型生成视频作为 rejected 构成偏好对，用 active policy 与 EMA old policy 的正/负隐式策略混合来抑制生成分布向"生成味"漂移，并加一个对冻结基座模型的 KL 正则防止过度偏离。

### 5. 动作条件世界模型 LingBot-Video-A2V

在预训练 backbone 上，将未来 $4T$ 帧的逐帧动作转为相对动作，拉平后经可学习 ActionEmbedder 映射为与视觉 latent 时间对齐的动作 latent（首帧补零动作），作为残差信号连同时间嵌入一起调制各 Transformer 块。用 Fourier GR-1 后训练数据集（重写 caption 仅描述初始状态，做未来观测泄漏检查）微调 8k 步（batch 64，lr 1e-5），得到条件于初始帧+动作序列生成未来视觉轨迹的具身世界模型。

## 三、关键结果

**扩展性实验（训练/验证 loss，非下游任务指标）**：MoE 13B-A1.4B 在等激活参数下持续优于 Dense 1.3B；MoE 13B-A1.4B、MoE 30B-A3B 均以约 2 倍激活参数量优于对应 dense 基线，MoE 30B-A3B 逼近 Dense 14B 的性能；早期扩展到 120B 总参数（13B→30B→60B→120B，激活 1.4B→3B→6B→11B）时，模型规模越大训练 loss 越低，符合可预测的 scaling law 趋势（未训到完全收敛）。推理效率上，在 1M token 序列长度处 MoE 30B-A3B 与同激活规模的 Dense-3B 近乎同速（0.97×），但相对 Dense-6B/14B/30B 分别快 1.50×/2.59×/3.18×。

**内部基准（T2V / TI2V，含 Motion/Prompt/Consistency/Aesthetic 通用质量 与 Human Interaction/Physical Simulation/Robotics/Egocentric/Navigation 具身域）**：对比 NVIDIA Cosmos 3、Wan 2.2 A14B、LongCat-Video、HunyuanVideo 1.5、LTX-2.3。TI2V 设定下 LingBot-Video 在通用质量与具身域两项均排名第一（开源模型中）；T2V 设定下通用质量排名第二，但具身域得分仍稳定超过 Cosmos 之外的其他开源基线，说明即便没有首帧图像条件，模型也具备较强的内在物理先验。

**公开基准 RBench（Table 1，650 条文本-图像 prompt，覆盖 5 类任务导向场景与 4 类机器人形态）**：

| 模型 | 类型 | Avg | Manip. | Spatial | Multi-entity | Long-hor. | Reasoning |
|---|---|---|---|---|---|---|---|
| LingBot-Video | 开源 | **0.620** | 0.578 | 0.643 | 0.444 | **0.634** | 0.505 |
| Wan 2.6 | 闭源 | 0.607 | 0.546 | 0.656 | 0.479 | 0.514 | 0.531 |
| Seedance 1.5 pro | 闭源 | 0.584 | 0.577 | 0.495 | **0.484** | 0.570 | 0.470 |
| Cosmos 3 Super | 开源 | 0.581 | 0.487 | 0.642 | 0.444 | 0.591 | 0.395 |
| Veo 3 | 闭源 | 0.563 | 0.521 | 0.508 | 0.430 | 0.530 | 0.504 |
| Wan 2.2 A14B | 开源 | 0.507 | 0.381 | 0.454 | 0.373 | 0.501 | 0.330 |
| HunyuanVideo 1.5 | 开源 | 0.460 | 0.442 | 0.316 | 0.312 | 0.438 | 0.364 |
| LongCat-Video | 开源 | 0.437 | 0.372 | 0.310 | 0.220 | 0.384 | 0.186 |

LingBot-Video 在四足（0.758）与总体平均分上均为全部 8 个对比模型（含 3 个闭源商业模型）中最高。

**Physics-IQ Verified（I2V，66 组真实物理实验/396 条视频）**：LingBot-Video 40.4 分，开源模型中排名第一，微弱领先 Cosmos 3（39.5），明显领先 HunyuanVideo 1.5（33.4）与 Wan 2.2 A14B（32.2）。

**GSB 人评**：T2V 上对 Wan 2.2 5B/A14B、LongCat-Video、LTX-2.3 的 Good 率明显高于 Bad 率，对 HunyuanVideo 1.5、Cosmos 3 竞争更接近；TI2V 上对全部开源基线 Good>Bad，但仍落后于 Kling-V3、Seedance 2.0 等闭源商业模型。

**LingBot-Video-A2V（动作条件后训练）**：在 EgoDex Eval 与 DreamDojo-HV Eval（未见于训练集的分布外物体/动作）上，论文仅给出定性对比（图 18），显示相较 DreamDojo 基线在物体状态保持（如苹果不消失）与动作跟随精度（抓取轨迹与目标物姿态更一致）上更优；**未报告量化指标**。

## 四、评价与展望

**优点**：(1) 把 DeepSeekMoE 的细粒度专家分割 + 无辅助损失负载均衡等 LLM 侧成熟做法系统性迁移到视频 DiT，扩展实验（专家数、粗细粒度、等激活对比 dense、120B 早期规律）做得较完整，证据链清晰；(2) 数据体系（Profiling Engine + World-Knowledge Topological Graph + Dense Structured Caption）覆盖长尾语义/动作分布，是工程量较大但可复用的基础设施贡献；(3) 六维度奖励拆解（尤其 Physical Plausibility 与 Motion Coherence 针对"慢动作幻觉""物体穿模/消失"等视频扩散典型失效模式）比单一标量奖励更细粒度，且用 CPS+单步探索的 Flash-GRPO 变体缓解了视频扩散 RL 的信用分配难题；(4) RBench 上超过 Wan 2.6 / Seedance 1.5 pro / Veo 3 等闭源模型，是开源视频基础模型少见的对闭源商业模型的正面对比结果。

**局限与开放问题**：(1) A2V（动作条件世界模型）部分只有定性图示，缺少量化的动作跟随误差/物理一致性指标，也未与更多同类工作（如 Genie Envisioner、WorldVLA 类模型）做量化对比；(2) 论文未展示世界模型输出直接用于下游策略训练/评估（closed-loop policy improvement）的实验，"Policy Evaluator""Action Planner"更多是愿景陈述而非实证；(3) 120B 规模的扩展实验明确说明"未训练至收敛"，其外推到生产级规模是否仍成立有待观察；(4) RBench、Physics-IQ Verified 等评测本身仍是"用另一个模型/规则打分"的自动化指标，与人类偏好、下游任务成功率之间的相关性缺乏专门验证；(5) 稀疏 MoE 带来的路由/专家并行通信开销、以及在小规模部署（非 1M token 长序列）场景下相对 dense 模型的实际收益边界，论文的推理效率对比集中在长序列区间，短序列/低分辨率场景下的效率优势未展示。整体上，这是一份偏"系统工程报告"性质的技术报告，核心创新是把 LLM MoE 成熟技术栈、结构化数据治理与多维度 RL 奖励整合到统一的视频基础模型训练流水线中，而非单点算法突破。

## 参考

- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, ACL 2024（细粒度专家分割与共享专家隔离的原始设计）
- Liu et al. *DeepSeek-V3 Technical Report*, arXiv:2412.19437（无辅助损失负载均衡、序列级均衡损失来源）
- He et al. *Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization*, arXiv:2605.15980（单步随机探索范式）
- Zheng et al. *DiffusionNFT: Online Diffusion Reinforcement with Forward Process*, ICLR 2026（Negative-Aware Finetuning 的前向过程优化框架来源）
- Gao et al. *DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos*, arXiv:2602.06949（A2V 定性对比基线）
