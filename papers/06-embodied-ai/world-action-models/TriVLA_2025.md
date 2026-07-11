# TriVLA：基于情景世界建模的三系统统一视觉-语言-动作模型，面向通用机器人控制

> **论文**：*TriVLA: A Triple-System-Based Unified Vision-Language-Action Model with Episodic World Modeling for General Robot Control*
>
> **作者**：Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Yanwei Fu, Xiangyang Xue, Yu-Gang Jiang et al.
>
> **机构**：复旦大学（Fudan University）、上海创新研究院（Shanghai Innovation Institute）
>
> **发布时间**：2025 年 07 月（arXiv 2507.01424，v3 版更新于 2025 年 10 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.01424) | [PDF](https://arxiv.org/pdf/2507.01424)
>
> **分类标签**：`VLA` `世界模型` `三系统架构` `视频扩散模型` `情景记忆` `flow matching`

---

## 一句话总结

TriVLA 提出"情景世界模型"（episodic world model）概念，用三个子系统——预训练 VLM 做情景语义感知（System2）、微调的视频扩散模型做情景动态预测（System3）、DiT flow-matching 策略做低层控制（System1）——取代传统双系统 VLA 的静态单帧表征，在 CALVIN ABC→D、LIBERO、MetaWorld 三个仿真基准上分别取得 4.37 平均完成长度、87.0% 平均成功率、0.714 平均成功率，同时保持 34-36Hz 的实时控制频率。

## 一、问题与动机

- 现有 VLA 大多采用"双系统"架构（System2 VLM 做高层推理 + System1 策略做低层动作生成），但通常只依赖一到两帧瞬时观测，缺乏对时间上下文的建模，导致短视野、反应式行为，难以支撑长时程规划与开放式意图理解。
- 论文受认知神经科学中 Tulving 的情景记忆（episodic memory）理论启发：情景记忆使个体能够在时空情境中编码、存储、检索经验，支撑"心理时间旅行"——既能回溯过去也能预演未来。作者据此提出让具身智能体拥有内部"情景世界模型"：一个持续积累、回忆、并预测未来动态的统一表征系统。
- 视频扩散模型（VDM）天然具备建模时间连续性和物理动态的能力，契合情景记忆对时序上下文的需求，但直接用于策略学习面临两个工程难题：①完整视频序列去噪计算量大，且容易带来开环控制问题；②生成的原始像素级视频含大量与决策无关的冗余信息，去噪耗时还会拖慢控制频率。
- TriVLA 的目标是把 VLM（对应情景记忆的"语义/情境编码"部分）和 VDM（对应"预测性想象"部分）统一进一个三系统框架，为下游策略提供既含语义又含时序动态线索的联合情景表征。

## 二、核心方法

**System 2：情景多模态感知（Episodic Multimodal Perception）**。采用预训练的 NVIDIA Eagle-2 VLM（由 SmolLM2 语言模型 + SigLIP-2 图像编码器构建），图像以 224×224 分辨率编码，经 pixel shuffle 每帧产出 64 个图像 token，与文本指令一起送入 LLM。作者实验发现从 LLM 第 12 层（而非最后一层）抽取视觉-语言 token $Q_{vl}$ 能同时带来更快推理速度和更高的策略成功率。机器人状态（关节位置/速度、末端位姿、底盘位置等）经 embodiment-specific MLP 投影到统一嵌入空间，得到状态 token $Q_s$。

**System 3：情景动态感知（Episodic Dynamics Perception）**。微调一个 1.5B 参数的 Stable Video Diffusion (SVD) 模型，训练目标是标准扩散重建损失：

$$\mathcal{L}_D = \mathbb{E}_{x_0\sim D,\epsilon,t}\|V_\theta(x_t, l_{emb}, s_0) - x_0\|^2, \quad x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$$

用大白话说：给未来视频序列 $x_0$ 加噪声得到 $x_t$，训练模型以当前帧 $s_0$ 和 CLIP 语言特征 $l_{emb}$ 为条件把噪声还原回真实未来视频——本质是训练一个"以语言指令为条件、以当前观测为起点"的未来视频生成器。

工程关键 trick：完整去噪整段视频太慢，还会带来开环控制问题，作者发现——即使只做扩散的第一步前向（不产生清晰视频，只是很粗糙的轨迹雏形），也已携带物体和机械臂运动的关键线索，足以指导下游决策（Figure 6/7/8/9 的对比可视化验证了这一点；但论文也在图注中明确承认"该表征能提供有价值的物理动态信息，但纹理和细节不够精确"）。具体做法是把当前帧 $s_0$ 与终态噪声潜变量 $q(x_{t'}\mid x_0)$（近似纯高斯噪声）拼接后送入 SVD 做**单次**前向，取多个上采样层的中间特征：

$$L_m = V_\theta(x_{t'}, l_{emb}, s_0)_{(m)}, \quad L_m\in\mathbb{R}^{T\times C_m\times W_m\times H_m}$$

再统一插值到公共分辨率并沿通道拼接，自动聚合多层特征，免去手工选层：

$$L'_m = \mathrm{Interpolation}(L_m),\ L'_m\in\mathbb{R}^{T\times C_m\times W_p\times H_p}, \qquad F_p=\mathrm{concat}(L'_0,L'_1,\dots,L'_m,\dim=1)$$

用大白话说：扩散第一步就是"半成品预测"，虽然模糊但已包含未来趋势线索；把 VDM 内部多层上采样特征都捞出来拼在一起，比只用某一层更全面。对配备第三人称+腕部相机的机器人，两个视角分别独立预测，得到 $F_p^{static}$ 与 $F_p^{wrist}$。

**System 1：策略学习（Policy Learning）**。用一组固定长度 $T\times L$ 的可学习 token $Q_{[0:T,0:L]}$ 对高维预测特征做时空注意力压缩——先对每帧做空间注意力，再做跨帧时间注意力：

$$Q' = \{\mathrm{Spat\text{-}Attn}(Q[i], (F_p^{static}[i], F_p^{wrist}[i]))\}_{i=0}^{T}, \qquad Q_p = \mathrm{FFN}(\mathrm{Temp\text{-}Attn}(Q'))$$

用大白话说：VDM 输出的特征维度太高（时间×空间×多视角），不能直接喂给策略网络，所以先学一组"摘要 token"，通过注意力把海量特征浓缩成固定大小的表征 $Q_p$。

最终，System2 的语义 token $Q_{vl}$ 与 System3 压缩后的预测 token $Q_p$ 一起作为条件，通过 cross-attention 注入 DiT（diffusion transformer）动作头，用类 flow-matching 的扩散损失训练：

$$\mathcal{L}_{diff}(\psi;A)=\mathbb{E}_{a_0,\epsilon,k}\|A_d(D_\psi(a_k,Q_{vl},Q_p))-a_0\|^2, \quad a_k=\sqrt{\beta_k}a_0+\sqrt{1-\beta_k}\epsilon$$

用大白话说：策略头本质是一个去噪器，在语义+预测双重条件下把加噪动作 $a_k$ 逐步还原为真实动作 $a_0$；末端用 embodiment-specific MLP 解码器 $A_d$ 把统一动作空间映射回具体机器人的关节/位姿空间。System1 每次前向预测 10 步动作 chunk，System2 每个 chunk 只跑一次（而非逐步都跑），二者配合使整体控制频率达到 34-36Hz。

训练规模：System3（视频扩散模块）微调使用 Something-Something V2 人类操作视频（193,690 条）+ Open X-Embodiment 机器人轨迹（179,074 条），并补充 CALVIN ABC、MetaWorld 及自采真实数据，采用类 Octo 的数据集采样比例，在 8×H100 上训练 2-3 天；策略头（System1+System2 联调）在 4×H100 上训练 5-9 小时。System3 微调完成后被冻结，此后仅作特征提取器使用。

## 三、实验结果

三个仿真基准（CALVIN、LIBERO、MetaWorld）+ 真实机器人定性实验（KINOVA 机械臂，RealSense D455 眼在手外配置）。

**CALVIN ABC→D 零样本长时程**（Avg Len，满分 5，100% ABC 标注数据训练）：

| 方法 | 1 | 2 | 3 | 4 | 5 | Avg Len |
|---|---|---|---|---|---|---|
| RT-1 | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| Diffusion Policy | 0.402 | 0.123 | 0.026 | 0.008 | 0.000 | 0.56 |
| Robo-Flamingo | 0.824 | 0.619 | 0.466 | 0.331 | 0.235 | 2.47 |
| RoboUniview（3D 法） | 0.942 | 0.842 | 0.734 | 0.622 | 0.507 | 3.65 |
| GR-1 | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| Vidman | 0.915 | 0.764 | 0.682 | 0.592 | 0.467 | 3.42 |
| Seer | 0.963 | 0.916 | 0.861 | 0.809 | 0.740 | 4.28 |
| VPP（此前最强） | 0.965 | 0.909 | 0.866 | 0.820 | 0.769 | 4.33 |
| **TriVLA（本文）** | **0.968** | **0.924** | **0.868** | **0.832** | **0.818** | **4.37** |

数据效率实验（仅 10% ABC 标注数据）：GR-1 Avg Len 1.41，VPP 3.25，TriVLA 3.46——用十分之一数据即逼近 VPP 用全量数据（4.33）的水平，差距明显小于 GR-1 在同等数据量下的表现。

**LIBERO**（Spatial / Object / Goal / Long 四套件，各 10 任务、500 次试验、3 个随机种子）：

| 方法 | Average | Spatial | Object | Goal | Long |
|---|---|---|---|---|---|
| Diffusion Policy | 72.4±0.7% | 78.3±1.1% | 92.5±0.7% | 68.3±1.2% | 50.5±1.3% |
| Octo | 75.1±0.6% | 78.9±1.0% | 85.7±0.9% | 84.6±0.9% | 51.1±1.3% |
| OpenVLA | 76.5±0.6% | 84.7±0.9% | 88.4±0.8% | 79.2±1.0% | 53.7±1.3% |
| **TriVLA（本文）** | **87.0±0.7%** | **91.2±0.8%** | **93.8±0.7%** | **89.8±0.9%** | **73.2±0.5%** |

**MetaWorld**（60 个任务，Easy / Middle / Hard 三档）：

| 方法 | Easy | Middle | Hard | Avg |
|---|---|---|---|---|
| RT-1 | 0.603 | 0.030 | 0.014 | 0.331 |
| Diffusion Policy | 0.433 | 0.072 | 0.089 | 0.299 |
| Susie | 0.542 | 0.213 | 0.244 | 0.420 |
| GR-1 | 0.695 | 0.337 | 0.448 | 0.582 |
| VPP | 0.822 | 0.507 | 0.519 | 0.679 |
| **TriVLA（本文）** | **0.857** | **0.528** | **0.563** | **0.714** |

**子系统消融**（CALVIN，单卡 H100 测延迟）：

| System2（EMP） | System3（EDP） | Avg Len | 延迟 | 参数量 |
|---|---|---|---|---|
| — | — | 3.68 | 29.29ms | 0.53B |
| — | 有 | 4.06 | 115.19ms | 1.87B |
| 有 | 有 | **4.37** | 142.69ms | 3.39B |

可见 System3（视频扩散动态预测）贡献了从 3.68 到 4.06 的主要增益，System2（VLM 语义感知）在此基础上再贡献 4.06 到 4.37，但代价是延迟从 29ms 涨到 143ms、参数从 0.53B 涨到 3.39B——性能增益相对算力开销呈边际递减。

真实世界实验为定性案例研究，未给出量化成功率表：短时程任务（叠粉色毛巾、抓橙子放盘子、倒水、移杯子）与长时程任务（"取饮料罐→倒入黄杯→插吸管→推杯子"等多步序列）均展示了成功执行轨迹的截图对比。

## 四、局限性

- 视频扩散模块（System3）为规避完整去噪的高延迟和开环问题，只做单次前向（扩散第一步），论文在多处图注（Fig 6/7/8/9）中自己承认"该表征能提供有价值的物理动态信息，但纹理和细节不够精确"，即牺牲了 VDM 本应具备的精细未来预测能力，只保留粗糙轨迹线索。
- 三系统架构在消融实验中显示出明显的算力代价：从纯 System1（0.53B、29ms）到完整三系统（3.39B、143ms），Avg Len 从 3.68 提升到 4.37，参数量和延迟增加约 5-6 倍，边际收益递减，性价比边界未被进一步讨论。
- 真实机器人实验只有定性案例展示（数个短/长时程任务的成功轨迹截图），缺乏量化成功率、试验次数统计和失败案例分析，难以评估真实世界鲁棒性的统计显著性。
- 语言条件路径存在一定冗余：System2（Eagle-2 内的 SmolLM2）和 System3（视频扩散模块用 CLIP 提取 $l_{emb}$）使用了两套独立的语言编码器，增加了系统复杂度但论文未讨论是否必要。
- 对比基线集中在 RT-1、Diffusion Policy、GR-1、VPP、Seer、Robo-Flamingo 等，未与相关工作部分提及的 GR00T N1、Hi Robot、$\pi_0$ 等更新的认知启发式或通用 VLA 基座模型做直接数值对比，难以判断相对当前最优 VLA 基座的真实位置。
- 复现性声明中承诺"发表后开源代码与权重"，提交时尚未公开，当前无法独立复现结果。
- 论文正文与附录对真实机械臂型号描述不一致（正文称 KINOVA GEN2，附录 F 称 Kinova Gen3），细节上存在编辑疏漏。

## 五、评价与展望

TriVLA 的核心贡献是把"情景记忆"这一认知科学概念较为形式化地映射到 VLA 架构设计上：用 System2 对应情景记忆的"语义/情境编码"、System3 对应"预测性想象"，再统一喂给低层策略 System1。这一设计延续了 GR00T N1、Hi Robot 等"认知启发双系统"VLA 的思路，并把 Video Prediction Policy（VPP）、Seer 等"用视频生成/预测特征辅助策略学习"这条技术路线与 VLM 语义路线做了显式融合，区别于 VPP/Seer 只用视频侧信息、或 OpenVLA、RT-2 等纯 VLM-based VLA 完全依赖静态观测的做法。

优点：①用单步扩散前向替代完整视频去噪，是在"预测质量"与"控制实时性"之间的一个务实折中，并辅以自动多层特征聚合，工程实现比较干净；②在 CALVIN/LIBERO/MetaWorld 三个基准上相对此前最强基线（VPP、OpenVLA、GR-1 等）均有提升，且在 10% 标注数据下仍展现出较好的数据效率；③对于接入了大型 VLM+VDM 的三系统方案而言，34-36Hz 的实时控制频率是有意义的工程成果，说明"高频重跑 VLM、低频跑 VDM"的解耦调度是可行的。

局限与开放问题：①"单步扩散前向=粗糙未来预测"这一权衡是否必要，还是可以用更轻量的时序预测头（如离散 latent action 或更小的动力学模型）达到相近效果而不必依赖 1.5B 参数的完整 SVD，论文未在这一维度做消融；②三系统架构相对纯 System1 基线带来的参数与延迟开销增长明显，而消融表显示收益边际递减，系统性价比的最优配置仍不清楚；③"情景记忆"这一认知框架目前更多体现为架构隐喻而非严格的记忆读写机制——论文没有实现显式的情景存储/检索（如外部记忆库、检索增强），System2/3 仍是前馈式的单次感知-预测，与"回忆过去经验"这一情景记忆的核心能力仍有距离，是可探索的后续方向；④真实世界评测量化不足是当前版本最明显的证据缺口，若后续版本能补充带成功率统计和失败分析的真实机器人基准，将显著增强结论的说服力。

## 参考

1. Hu et al. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations (VPP). arXiv:2412.14803, 2024.
2. Tian et al. Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation (Seer). arXiv:2412.15109, 2024.
3. Bjorck et al. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734, 2025.
4. Shi et al. Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models. arXiv:2502.19417, 2025.
5. Blattmann et al. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv:2311.15127, 2023.
