# HoloBrain-0：显式注入相机参数与运动学先验的具身感知 VLA 技术报告

> **论文**：*HoloBrain-0 Technical Report*
>
> **作者**：Xuewu Lin, Tianwei Lin, Yun Du, Hongyu Xie, Yiwei Jin, Jiawei Li, Shijie Wu, Qingze Wang, Mengdi Li, Mengao Zhao, Ziang Li, Chaodong Huang, Hongzhe Bi, Lichao Huang, Zhizhong Su
>
> **机构**：Horizon Robotics
>
> **发布时间**：2026 年 02 月（arXiv 2602.12062）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.12062) | [PDF](https://arxiv.org/pdf/2602.12062)
>
> **分类标签**：`具身先验` `跨本体VLA` `空间增强` `关节图注意力` `扩散策略` `异步推理基础设施`

---

## 一句话总结

HoloBrain-0 把相机内外参和机器人 URDF 运动学链作为**显式**输入注入 VLA 架构（而非让模型隐式学习），配合关节 6D 位姿状态表示、混合相对动作空间、迭代测试驱动数据策略和零开销的 SimpleRTC 异步推理，使 0.2B 参数的轻量模型在 RoboTwin 2.0 随机化设定下达到 90.8% 成功率、在 LIBERO-Plus 零样本 OOD 基准上以 74.0% 刷新 SOTA，并开源了配套的全栈基础设施 RoboOrchard。

## 一、问题与动机

论文观察到当前 VLA 研究面临两个系统性瓶颈：（1）高质量、专家级示教数据的采集成本高昂；（2）大模型低延迟实时控制的部署困难。更深层的架构问题是：传统 VLA 通常直接学习“视觉到动作”的映射，依赖复合动作空间（如 RT-1、π0）、异构编码器或纯文本提示来处理跨本体（cross-embodiment）数据，这**完全忽略了显式的结构先验**——包括相机几何参数和机器人运动学链——迫使模型在没有物理先验的情况下硬拟合跨本体差异。

作者将问题形式化为条件动作生成：给定多视角 RGB 图像 $I\in\mathbb{R}^{N\times H\times W\times3}$、深度图 $D\in\mathbb{R}^{N\times H\times W\times1}$、机器人本体感知关节状态 $S\in\mathbb{R}^{N_j}$、语言指令 $T$，**额外引入**相机参数 $C$（内外参）与机器人运动学先验 $E$（URDF），构成完整输入空间：

$$\{I\in\mathbb{R}^{N\times H\times W\times3}, D\in\mathbb{R}^{N\times H\times W\times1}, S\in\mathbb{R}^{N_j}, T\}\cup\{C,E\}$$

模型输出未来 $t_{out}$ 步的动作块 $S_{out}\in\mathbb{R}^{N_j\times t_{out}}$。用大白话说：与其让网络在训练数据里"猜"出每个机器人的坐标系怎么摆、关节怎么转，不如直接把相机怎么架、机械臂怎么连这些"物理常识"喂给模型，让它专注于学习操作策略本身。

## 二、核心方法

整体架构由三个核心模块组成：VLM 语义骨干（两个变体：GroundingDINO Tiny 或 Qwen2.5-VL-3B）、Spatial Enhancer（空间增强器）、Embodiment-Aware Action Expert（具身感知动作专家）。此外还配套 SimpleRTC + Teacher Forcing 训练/推理策略。

### 2.1 Perspective-aware Spatial Enhancer

沿用作者前作 BIP3D 和 SEM 的空间增强器：利用相机内外参把多视角 2D 图像特征沿各自相机视锥投影到统一 3D 坐标系，预测离散深度分布（可接入深度传感器输入）并聚合生成深度感知的 3D 位置编码，再与图像特征融合，得到跨视角一致的几何感知表示。

关键改动：把 3D 投影坐标系从机器人本地基座坐标系**改为中心固定相机坐标系**（如第三人称视角或头戴视角）。这样做的两个好处：（1）消除了不同机器人基座定义不一致带来的学习干扰，利于跨本体泛化；（2）能自然容纳没有固定机器人"基座"、只有头戴视角的第一人称人类数据（EgoDex）。

### 2.2 Embodiment-Aware Action Expert

动作专家不沿用 LLM 结构，而是基于关节图（joint-graph）注意力的 Transformer，包含机器人状态编码器和动作解码器，采用 diffusion 的 x-prediction 建模动作分布。核心改动在输入状态表示和输出动作空间：

**输入状态**：只保留每个关节的 6D 位姿，屏蔽关节角（除夹爪开合度外）：

$$s_i=\begin{cases}[-1]\oplus[x,y,z,q_w,q_x,q_y,q_z] & \text{if } m_i=1\\ [\theta]\oplus[x,y,z,q_w,q_x,q_y,q_z] & \text{otherwise}\end{cases}\quad 1\le i\le N_j$$

用大白话说：关节角的零位定义、旋转方向、URDF 约定在不同机器人之间千差万别，直接喂给模型会造成歧义；而关节的笛卡尔 6D 位姿是统一的几何参照系，跨本体更一致。

**输出动作空间**：混合相对变换——同时预测关节角残差和连杆位姿位移：

$$a_t=\{[\Delta\theta,\Delta x,\Delta y,\Delta z,\Delta q_w,\Delta q_x,\Delta q_y,\Delta q_z]_i \mid 1\le i\le N_j\}$$

好处有二：（1）同时兼容底层关节位置控制和高层末端位姿控制两种硬件接口；（2）可以在没有关节角标注的异构数据（如人类视频）上训练。

### 2.3 训练目标

四项损失联合优化：

$$L=\alpha_\tau(\lambda_1 L_{joint}+\lambda_2 L_{pose}+\lambda_3 L_{pose}^{fk})+\lambda_4 L_{depth}$$

其中 $L_{joint}$、$L_{pose}$ 分别是预测关节角/6D 位姿与真值的距离（用 smooth L1 替代标准 L2，缓解噪声样本导致的训练不稳定）；$L_{pose}^{fk}$ 是**正运动学位姿损失**——先用预测的关节角重新计算正运动学得到 6D 位姿，再与真值比较，用大白话说就是"强迫关节角预测和位姿预测互相对得上账"；$L_{depth}$ 是空间增强器深度分布的交叉熵损失。$\alpha_\tau=T/(\tau+1)$（$T=1000$）是随扩散时间步变化的系数，噪声越大权重越小。

此外引入 **winner-takes-more** 策略：每个训练样本生成 $N$ 条候选轨迹，动态提高预测误差最小的"获胜"轨迹的损失权重、压低其余轨迹权重，防止模型在多峰动作分布上收敛到"平均模式"（mode averaging）。

### 2.4 SimpleRTC 与 Teacher Forcing：零开销异步推理

同步推理（Synchronous）逐步执行完整动作块，延迟可达约 1s，导致动作停顿；异步推理（Asynchronous）解耦推理与执行，帧率可达 5–10 FPS，但连续动作块之间的不一致会造成机械抖动。作者在 Black et al. 提出的 Real-Time Chunking（RTC，推理时梯度引导，开销大）和 Training-time RTC（训练时固定长度前缀，需重训且结构僵化）基础上，提出两段式方案：

**推理时策略 SimpleRTC**：零梯度、零重训的软约束 inpainting。设 $\hat A_{0\mid\tau}$ 为扩散第 $\tau$ 步的 x-prediction 输出，$A_{prev}$ 为上一动作块尚未执行的剩余部分，融合动作为：

$$\tilde A_{0\mid\tau}=\mathbf{w}\odot A_{prev}+(\mathbf{1}-\mathbf{w})\odot \hat A_{0\mid\tau}$$

权重向量 $\mathbf{w}$ 由归一化变量 $\rho_t$ 决定（$d$ 为推理延迟步数，$L$ 为过渡窗口长度）：

$$\rho_t=\begin{cases}1 & t\in[0,d]\\ 1-\frac{t-d}{L} & t\in(d,d+L)\\ 0 & t\in[d+L,H]\end{cases}$$

并给出线性、二次、指数三种衰减曲线：$w^{lin}_t=\rho_t$，$w^{quad}_t=\rho_t^2$，$w^{exp}_t=\rho_t\frac{e^{\rho_t}-1}{e-1}$。用大白话说：离当前执行点越近的历史动作，越强制沿用旧轨迹；越往未来，越信任新预测，中间用一段平滑过渡窗口衔接，避免生硬跳变。

**训练时策略 Teacher Forcing**：用真值动作 $A_{gt}$ 覆盖噪声输入的前 $N_{prefix}$ 步（$N_{prefix}\sim\text{Poisson}(\lambda)$），以小概率 $\gamma$（如 25%）混合应用，缩小训练时"全噪声输入"与推理时"前缀已知"之间的分布差异。

## 三、实验结果

### 3.1 模型规模

| 模型 | Vision Encoder | LM | Spatial Enhancer | Action Expert | Trainable | 总参数量 |
|---|---|---|---|---|---|---|
| HoloBrain-0-GD | 29.64M | 130.80M | 2.28M | 20.79M | 74.81M | **183.70M (≈0.2B)** |
| HoloBrain-0-QW | 668.68M | 388.24M | 2.09M | 20.79M | 412.17M | **1080.86M (≈1.1B)** |

GD 用 GroundingDINO Tiny 骨干（冻结 BERT 语言模块，但**不冻结**视觉编码器）；QW 用 Qwen2.5-VL-3B，只保留其 LLM 的第一层、丢弃其余层，同时冻结视觉编码器。有意思的是，GD 虽参数量小得多，但因未冻结视觉编码器，训练显存反而与 QW（约 30GB/卡）相当。

### 3.2 真机实验（Dual-arm AgileX Piper，10 项任务，各任务仅 200 条示教）

| 模型 | Fold towel | Place empty cup | Stack blocks three | Grasp anything | Fold clothes | Fold paper box | 平均 progress | 平均成功率 |
|---|---|---|---|---|---|---|---|---|
| π0 | 61.58/31.58 | 48.5/30.00 | 70.00/13.33 | 87.50/87.50 | 33.33/15.00 | 86.00/80.00 | 68.34 | 45.39 |
| π0.5 | 61.58/63.16 | 99.50/95.00 | 80.00/26.67 | 98.40/98.40 | 60.95/50.00 | 81.50/65.00 | 84.46 | 69.16 |
| HoloBrain-0-GD (0.2B) | 95.26/84.21 | 89.50/85.00 | 81.11/40.00 | 93.50/93.50 | 67.62/55.00 | 82.00/75.00 | **88.07** | 74.81 |
| HoloBrain-0-QW (1.1B) | 84.74/84.21 | 89.50/70.00 | 83.33/46.67 | 95.00/95.00 | 81.43/95.00 | 99.50/95.00 | 87.32 | **77.18** |

（每格 "progress score / success rate"，20 次试验取均值。）HB-GD / HB-QW 的平均成功率分别比 π0.5 高 5.65 和 8.02 个百分点。两个长时任务提升尤其明显：fold clothes 成功率从 π0.5 的 50.0% 提升到 HB-QW 的 95.0%；fold paper box 从 65.0% 提升到 HB-GD/HB-QW 的 75.0%/95.0%。Grasp Anything 任务上 HoloBrain-0-QW 在训练时见过的物体类别上成功率 93.5%，未见过的新物体上反而达 97.5%，作者认为这说明抓取能力已经泛化到与"是否见过"基本无关。

### 3.3 仿真基准

**RoboTwin 2.0**（50 任务多任务联合训练，各任务 50 条 clean + 500 条 randomized 示教，共 1 万次 rollout）：

| 方法 | 参数量 | Clean | Randomized |
|---|---|---|---|
| π0.5 | 3B | 82.74 | 76.76 |
| X-VLA | 0.9B | 72.80 | 72.84 |
| Lingbot-VLA | 4B | 88.56 | 86.68 |
| Motus | 8B | 88.66 | 87.02 |
| HoloBrain-0-GD | 0.2B | 91.30 | 90.80 |
| HoloBrain-0-QW | 1.1B | **91.90** | **92.30** |

0.2B 的 HoloBrain-0-GD 已超越所有对比模型（包括 8B 的 Motus）。

**LIBERO / LIBERO-Plus**（标准 LIBERO 训练后，直接在 7 维扰动的 LIBERO-Plus 上零样本评估）：

| 方法 | 参数量 | LIBERO Avg | LIBERO-Plus Avg |
|---|---|---|---|
| OpenVLA-OFT | 7B | 97.1 | 69.6 |
| X-VLA | 0.9B | 98.1 | 69.7 |
| HoloBrain-0-QW | 1.1B | **97.4** | 72.6 |
| HoloBrain-0-GD | 0.2B | 96.7 | **74.0** |

LIBERO-Plus 上 HoloBrain-0-GD 以 74.0% 刷新已知最优（此前最好为 OpenVLA-OFT 的 69.6%），作者将其归因于模型学到了可迁移的操作基元而非记住了训练分布。

**GenieSim 2.2**（Agibot Challenge 2025 基准，人形机器人 Agibot G1 上半身操作，10 项任务 ×25 次试验，仅测 QW）：

| 方法 | RDT | UniVLA | X-VLA | HoloBrain-0-QW |
|---|---|---|---|---|
| Total Score | 2.434 | 2.795 | 4.541 | **4.685** |

### 3.4 消融实验

- **Grasp Anything 联合训练**：把 Grasp Anything 数据与 7 项基础任务联合训练，7 项任务平均 progress 从 85.32 升至 89.19、平均成功率从 72.40 升至 75.00，说明技能聚焦型辅助任务数据能有效提升核心任务的泛化。
- **SimpleRTC + Teacher Forcing**（cloth folding 任务上）：同步基线 progress/成功率为 47.6/20.0，耗时 115s；SimpleRTC（TF=0%）提升到 77.9/65.0，耗时降到 82s；TF=5% 反而下降到 66.9/50.0（低比例有害）；TF=75% 时最优，达到 95.2/85.0，耗时 88s。结论：SimpleRTC 大幅降低执行时间，Teacher Forcing 比例需要适中偏高才能发挥增益，二者互补消除了异步推理的动作停顿与抖动。

## 四、局限性

1. **强依赖精确的 URDF 与相机标定**：架构的核心优势（显式几何/运动学先验）同时也是硬约束——新增机器人本体必须提供精确 URDF 和标定好的多视角相机参数，数据清洗阶段还需用 3D 一致性重投影校验（图 3）过滤标注误差样本，工程门槛不低。
2. **仍是纯模仿学习**：论文 Future Work 明确指出下一代计划引入离线强化学习和价值模型，说明当前版本的能力上限仍受示教数据质量约束，缺乏从自身 rollout 中自我改进的机制。
3. **指令跟随能力未被充分评测**：作者自陈"精确指令跟随，尤其是易混淆指令场景"是当前 VLA 研究中被低估的能力，现有基准也未覆盖，留作未来工作。
4. **真机评测局限于单一本体**：10 项真机任务全部基于固定底座的双臂 AgileX Piper 桌面操作，跨本体能力主要靠预训练数据混合比例和仿真基准（RoboTwin 多本体）验证，缺少大规模真机跨本体迁移测试。
5. **长时可变形物体操作仍有明显失败模式**（附录 C.4）：进度倒退（Progress Reversion，从纠缠状态无法局部恢复、整体回退到初始展平阶段）、状态混淆（部分折叠状态与褶皱状态视觉相似导致模型误判阶段并"状态振荡"）、对丝绸等低摩擦材质的泛化不足（易出现"空抓"或打滑）、以及展平效率低于人类演示者的隐式技巧（张力控制、策略性抓取点选择）。
6. **SimpleRTC 的超参数选择较为经验化**：融合窗口长度 $L$、衰减曲线类型、Teacher Forcing 比例 $\gamma$ 均通过单一任务（cloth folding）网格式消融确定（如 $\gamma=5\%$ 反而有害），未给出跨任务的原则性选择方法。
7. **大规模预训练语料的构建成本**：156.66M 帧、3547.4 小时、53 万余条轨迹、覆盖 7 种本体的数据清洗（含逐样本 3D 重投影一致性校验）本身即是一项高成本工程，"可复现"的说法更多针对代码与基础设施层面，而非数据规模本身。

## 五、评价与展望

HoloBrain-0 的核心贡献在于把"显式几何/运动学先验"这一在传统机器人学（正运动学、多视角几何）里的常识重新带回端到端 VLA 架构，与 π0/π0.5（flow-matching 动作专家但无显式相机-运动学输入）、X-VLA（用软提示 token 学习跨本体兼容性，属隐式方案）、SpatialVLA（引入 3D 位置编码增强空间推理，但未显式利用运动学链）形成鲜明对比。其"关节 6D 位姿输入 + 混合相对动作空间 + 正运动学损失"的组合设计具有一定的架构简洁性和可解释性，且在 0.2B 参数规模下于 RoboTwin 2.0、LIBERO-Plus 等多个基准取得优于更大模型（甚至 7B/8B 级别基线）的成绩，为低延迟边缘部署提供了有说服力的证据。

与同为 Horizon Robotics 团队前作 BIP3D、SEM 一脉相承，Spatial Enhancer 把坐标系从机器人基座改为中心固定相机系是一个不起眼但具体有效的工程改动，体现了作者对"什么该显式建模、什么该留给数据驱动学习"这一权衡的清晰判断。SimpleRTC 相对于 Black et al. 的 RTC / Training-time RTC 提供了一个更工程友好的折中方案（零梯度开销、无需重训、灵活长度），但其理论基础仍是启发式的软 inpainting，缺乏对融合引导下去噪过程收敛性的严格分析，目前仅在单一任务（cloth folding）上做了较充分的消融验证，跨任务、跨骨干的普适性有待进一步检验。

开放问题包括：（1）显式运动学先验方案能否扩展到移动底盘、多指灵巧手、腿式机器人等结构差异更大的本体，目前验证止步于单/双臂机械臂和人形机器人上半身；（2）Grasp Anything 辅助任务提升核心任务表现的现象提示了"技能聚焦型数据的可迁移性"这一更一般的问题，与 X-VLA、GR00T 等工作强调的数据混合策略研究方向相通，但目前仍缺乏系统性方法论指导哪类辅助技能能迁移到哪类目标任务；（3）RoboOrchard 基础设施（MCAP 记录 + Apache Arrow/DuckDB 存储 + 自包含 Model Artifact + 解耦客户端-服务器部署）相对 LeRobot 等已有开源生态，其差异化价值主要体现在高吞吐实时记录与训练/部署的解耦，长期是否能形成社区级标准仍需时间验证。

## 参考

- Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- P Intelligence, Black, K. et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054, 2025.
- Zheng, J. et al. *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model*. arXiv:2510.10274, 2025.
- Lin, X. et al. *SEM: Enhancing Spatial Understanding for Robust Robot Manipulation*. arXiv:2505.16196, 2025.（作者团队前作，Spatial Enhancer 与 Action Expert 的基础）
- Black, K., Galliker, M. Y., Levine, S. *Real-Time Execution of Action Chunking Flow Policies*. arXiv:2506.07339, 2025.（RTC，SimpleRTC 的对比基线）
- Chen, T. et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation*. arXiv:2506.18088, 2025.
