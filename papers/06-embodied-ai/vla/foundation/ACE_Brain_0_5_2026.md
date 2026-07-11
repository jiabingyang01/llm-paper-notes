# ACE-Brain-0.5：面向物理智能体的统一具身基础模型

> **论文**：*ACE-Brain-0.5: A Unified Embodied Foundational Model for Physical Agentic AI*
>
> **作者**：Ziyang Gong, Haoming Gu, Zehang Luo, Zhi Hou, Xue Yang, Dacheng Tao, Xiaogang Wang et al.（ACE-Brain Team）
>
> **机构**：ACE Robotics（ACE-Brain Team；论文正文未列出学术/公司隶属关系，仅以 ACE Robotics 品牌及 ACE-Brain Team 署名，项目页 ace-brain-team.github.io）
>
> **发布时间**：2026 年 07 月（arXiv 2607.04426，落款日期 2026-07-07）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.04426) | [PDF](https://arxiv.org/pdf/2607.04426)
>
> **分类标签**：`统一具身基础模型` `模型合并` `空间智能` `自我提升` `进度奖励模型`

---

## 一句话总结

ACE-Brain-0.5 用单个 8B mixture-of-transformer 模型，通过 SSR+（Scaffold–Specialize–Reconcile–Reactivate）四阶段训练策略，把空间感知、决策规划、具身交互（导航+操作）、自监控（进度估计）、自我提升五大认知功能统一进同一套权重：在 18 个空间感知/定位基准中 14 个超越前代 ACE-Brain-0，LIBERO 平均成功率 98.2%，RBM-EVAL 进度估计 VOC 达 0.94–0.96（优于 Robometer-4B 等专用奖励模型）。

## 一、问题与动机

作者观察到具身智能范式经历了三个阶段：经典模块化 Sense-Plan-Act 管线（可解释但泛化差）；端到端 VLA/World-Action 模型（如 π 系列、GR00T、QwenVLA，动作生成强但空间推理/长程规划/执行评估支持有限）；多模型协作的 Robot-Agent 系统（如 QwenRobot、ABot，靠工具编排扩展自主性但不学习统一的机器人表征）。三条路线各自发展了具身智能的不同侧面却彼此割裂（见论文 Table 1 的能力对比矩阵），没有一个系统能在单一模型内同时具备物体感知、空间理解、任务规划、导航、操作、执行监控与自我提升七项能力。

论文提出"统一具身基础模型"（Unified Embodied Foundation Model）范式，将机器人智能组织为五个紧耦合的认知功能：Spatial Perception（感知与记忆）、Decision Making（长程推理与规划）、Embodied Interaction（导航与操作执行）、Self Monitoring（执行进度追踪与失败反馈）、Self Improvement（从部署经验中持续适配）。空间智能被视为贯穿导航、操作、指令跟随的共享表征基础。工作建立在前作 ACE-Brain-0（arXiv 2603.03198，一个以理解为中心、建立跨异构机器人平台空间智能共享脚手架的模型）之上，ACE-Brain-0.5 是该"统一机器人大脑"范式的首个实例。

## 二、核心方法

**整体架构**：全能视觉编码器（Omni-Vision Encoder）统一处理单视角图像、多视角观测与视频，投影进 LLM token 空间；语言指令与视觉 token 一起送入 LLM Decoder，生成共享具身状态

$$s_t = F_\theta(\ell, o_t, q_t),$$

其中 $\ell$ 为指令、$o_t$ 为视觉观测、$q_t$ 为可选本体状态。大白话说：这一步就是把"看到的场景+要做的事+机器人当前姿态"揉成一个统一的内部语义表示，后续所有输出头都从这个表示解码。

为满足实时闭环控制的低延迟要求，另设一条 Fast Vision Pathway：轻量 DINOv3 编码器实时处理最新多视角观测 $z_t = E_{\text{fast}}(o_t)$，与 LLM Decoder 输出的 $s_t$ 一起送入以流匹配（flow-matching，沿用 $\pi_0$ 的 action-expert 设计）实现的 Action Expert，生成可执行动作块 $a_t = \text{ActionExpert}(s_t, z_t)$。这是一个"双时间尺度"接口：LLM Decoder 以较低频率算高层语境，Fast Vision Pathway 以控制频率提供最新感知特征，训练时冻结 VLM 主干只更新 Fast Vision 和 Action Expert，避免扰动主干的空间推理与语言 grounding 能力。四类任务共享该 backbone 但各自走专属解码路径：空间感知输出 bounding box/mask/point，决策规划自回归生成子目标文本，具身交互输出导航离散动作或操作连续动作块，自监控输出归一化进度序列 $\hat p_t \in [0,1]$ 及成对轨迹偏好判断。

**SSR+ 训练策略**：核心创新是在 ACE-Brain-0 的 SSR（Scaffold-Specialize-Reconcile）基础上新增 Reactivate 阶段，解决"混合监督信号导致跨接口干扰"的问题（QA 用文本生成、grounding 用结构化坐标、导航用离散动作序列、操作用连续动作块、进度估计用时序对齐标量——直接混合 SFT 会互相污染）。

1. **Scaffold**：以 ACE-Brain-0 checkpoint $\theta_0$（从 Qwen3-VL-8B-Instruct 初始化）作为空间理解的共享起点。
2. **Specialize**：从 $\theta_0$ 独立训练四个任务专精 checkpoint：$\theta_{qa}$（空间问答+思维链规划）、$\theta_{grd}$（2D/3D grounding）、$\theta_{nav}$（导航动作预测）、$\theta_{prog}$（帧级进度估计+成对偏好）。
3. **Reconcile**：通过任务向量（$\tau_i = \theta_i - \theta$）合并把四个专家权重space融合回单一参数空间，逐层最小化合并模型与各专家在其数据分布上的输出差异：

$$\theta_{\text{merge}}^{*} \approx \theta_{\text{pre}} + \arg\min_{\tau_{\text{merge}}}\sum_{i=1}^{K}\frac{1}{\|\tau_{i}\|_F^2}\left\|(\tau_{\text{merge}} - \tau_i)\tau_i^\top\right\|_F^2,$$

  在 FusionBench 框架下用 Adam 跑 1000 步 data-free 优化。大白话说：把四个"偏科生"的知识差量投影到一组共同权重里，让合并后的模型对每个任务的中间表征都尽量贴近对应专家，而不是简单平均导致互相抵消。
4. **Reactivate**（新增阶段）：合并后的 $\theta_{\text{merge}}$ 不直接作为最终模型——论文的关键经验发现是，合并模型已经学到了各任务的语义能力（定位物体、预测导航动作、估计进度），丢失的只是"接口惯例"（grounding 用结构化坐标 vs. 导航用离散 token vs. 进度用标量数值，这些输出格式在权重平均时被暂时打乱）。因此只需在一个小型混合 SFT 数据集 $\mathcal{D}_{\text{mix}}$ 上做少量步数微调 $\theta_{0.5} = \text{SFT}(\theta_{\text{merge}}, \mathcal{D}_{\text{mix}})$，即可重新同步各接口的输出惯例，所需步数远少于 Specialize 阶段的冷启动训练。论文附录给出两条理论支撑：一步干扰界（Theorem 1，证明跨任务梯度内积为负时联合训练会增大风险，故 Stage 2 需要隔离）与脚手架迁移界（Theorem 2，证明 ACE-Brain-0 的广覆盖空间预训练能降低下游任务的几何分布偏移，从而降低 Reactivate 所需代价）。

**自我提升框架**：以外部执行状态 $\mathcal{H}$（任务 schema、空间记忆、失败恢复案例）为载体，每次 rollout 存储经验元组 $\xi=(\ell,\tau,p_1,a_1,e_1)$，用更新函数 $\mathcal{H}_{k+1}=\mathcal{U}(\mathcal{H}_k,\{\xi_i\})$ 增量更新，无需重训整模型即可部署时适配。论文以导航为具体实例（"Navigation Evolving"）：冷启动策略做闭环 rollout，oracle 导航教师检测偏差（距目标距离增大、路径对齐度下降、提前停止或累积局部误差），一旦触发即接管完成剩余轨迹，生成修正性导航经验 $\mathcal{D}_{\text{evo}}$，按目标完成度与路径效率过滤后与原始示范混合重训。

## 三、关键结果

**空间感知与定位**（对比 ACE-Brain-0 及 GPT-5.4/Gemini-2.5-Pro/Claude-Sonnet-4.6 等闭源模型，18 个基准中 14 个提升）：

| 基准 | ACE-Brain-0.5 | ACE-Brain-0 | 备注 |
|---|---|---|---|
| VSI（视频空间智能） | 62.2% | 63.1% | 略降 0.9pp，但超 GPT-5.4 52.6% |
| MindCube（空间心智建模） | 86.3% | 82.1% | 超 RynnBrain-8B 56.6% |
| Multi3DRef（3D多目标grounding） | 72.4% | 55.9% | +16.5pp |
| RefSpatial（多步空间指代） | 55.6% | 26.0% | 大幅提升，超 Cosmos3-Nano 53.1% |
| RoboAfford（可供性定位） | 75.1% | 56.5% | +18.6pp |
| ShareRobot-Traj（轨迹预测误差，越低越好） | 0.32 | 0.46 | 优于 Gemini-2.5-Pro 0.34 |

自动驾驶决策基准上 ACE-Brain-0.5 相对 ACE-Brain-0 有"温和下滑"（论文明确说明不追求驾驶专精），如 MAPLM 仍达 71.3%（超 GPT-5.4 57.9%）。

**导航（VLN-CE, Table 3）**：R2R Val-Unseen 上统一模型 SR 57.4% / SPL 51.7% / NE 4.8，专精变体 ACE-Brain-0.5-Specialist 达 SR 62.2%/SPL 56.2%/NE 4.2，为对比方法中最佳 NE；RxR Val-Unseen 统一模型 SR 63.8%（超过 NavFoM、RynnBrain-Nav、Qwen-VLA-Instruct 等全部开源基线）。自我提升消融（Table 7）显示 Navigation Evolving 相对纯静态示范模仿在 R2R SR 上 +8.8pp（48.6%→57.4%）、RxR SR 上 +10.5pp（53.3%→63.8%）。

**操作**：LIBERO 平均成功率 98.2%（Spatial/Object 均 100.0%，Goal 96.0%，Long 97.0%），超过 OpenVLA-OFT 97.1%、GR00T N1.6 97.0%、$\pi_{0.5}$ 96.9%；SimplerEnv-Bridge 上 ACE-Brain-0.5-VLA 变体平均 82.3%，超过 GTA-VLA 81.2%、X-VLA 76.0%。

**自监控（进度估计，Table 6）**：标准 RBM-EVAL-ID/OOD 上 VOC 分别为 0.94/0.96；论文新构造的 RBM-EVAL-Refined（把成功轨迹视频帧序反转作为负例，专门检验时序方向理解而非静态状态识别）上仍保持 0.80/0.88，优于 Robometer-4B 的 0.78/0.81，而通用 VLM（Qwen3-VL-8B 0.28/0.30）和多数专用奖励模型（VLAC-8B、RoboDopamine-8B）在反转设置下大幅退化。

## 四、评价与展望

**优点**：本文最有价值的贡献是 SSR+ 的 Reactivate 阶段及其背后的经验发现——模型合并后损失的是"接口输出惯例"而非"任务能力"，因此只需极少步数的轻量微调即可恢复，这为持续扩展多任务统一模型（而非每次新增能力都全量重训）提供了一条实用路径，并有理论分析（一步干扰界、脚手架迁移界）支撑设计选择。RBM-EVAL-Refined 这一反转轨迹负例控制变量的设计也是对"进度奖励模型是否真正理解时序方向"这一常被忽视问题的一次严谨补充评测。整体上，论文用 15+ 个基准较全面地验证了"感知-规划-交互-监控"闭环可以在单模型内相互增益而非互相拖累。

**局限与开放问题**：(1) 自我提升机制目前仅在导航场景做了完整实例化（基于 oracle 教师检测偏差），论文自陈这是"轻量且领域特定的"机制，尚未扩展到操作或跨任务的模型级自我进化，通用化路径留待未来工作；(2) 空间感知/定位 18 个基准中仍有 4 个相对前代下降（如 VSI），说明多任务统一并非无代价的帕累托改进；(3) 操作能力的验证主要在 LIBERO 与 SimplerEnv-Bridge 两个仿真/半仿真基准上，真实机器人端到端操作的量化结果未见于正文（仅附录一张 grounding+运动控制的定性演示图）；(4) 外部执行状态 $\mathcal{H}$ 的更新函数 $\mathcal{U}$ 依赖启发式过滤（目标完成度、路径效率阈值），尚未形成端到端可学习的自我提升信号。与同期公开的统一具身模型（如 Cosmos 3、RynnBrain、Pelican-Unified）相比，ACE-Brain-0.5 在 Table 1 列出的能力矩阵上是唯一覆盖全部七项能力（含 execution monitoring 与 self-improving）的系统，但其自我提升仍属"部署时经验回放+轻量适配"层面，距离真正的闭环模型自主进化（模型权重级持续学习）仍有较大差距，这也是论文在结论中明确指出的未来方向。

## 参考

- ACE-Brain-0（arXiv 2603.03198）——本文直接基础，建立跨异构机器人平台的空间智能共享脚手架，SSR 训练策略的原始来源。
- $\pi_0$（arXiv 2410.24164）与 $\pi_{0.5}$（arXiv 2504.16054）——Action Expert 流匹配设计与开放世界泛化 VLA 的对比基线。
- RoboMeter / RBM-1M（arXiv 2603.02115）——进度奖励数据与 RBM-EVAL 评测协议的来源，本文在此基础上构造 RBM-EVAL-Refined。
- Qwen3-VL 技术报告（arXiv 2511.21631）——ACE-Brain-0/0.5 的多模态主干初始化来源。
- StreamVLN（arXiv 2507.05240）——VLN-CE 导航数据组织格式（step-level 指令跟随）的参照标准。
