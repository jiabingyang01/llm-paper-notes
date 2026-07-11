# Rethink-VLA：重新思考 VLA 模型规模化——物理对齐、数据混合与正则化

> **论文**：*Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization*
>
> **作者**：Ye Wang, Sipeng Zheng, Hao Luo, Wanpeng Zhang（以上四位并列一作）, Haoqi Yuan, Chaoyi Xu, Haiweng Xu, Yicheng Feng, Mingyang Yu, Zhiyu Kang, Zongqing Lu, Qin Jin†（通讯作者）
>
> **机构**：Renmin University of China；BeingBeyond；Peking University
>
> **发布时间**：2026 年 02 月（arXiv 2602.09722）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.09722) | [PDF](https://arxiv.org/pdf/2602.09722)
>
> **分类标签**：`VLA规模化` `Mixture-of-Transformers` `flow matching` `跨具身数据混合` `动作空间对齐` `双盲真机评测`

---

## 一句话总结

这是一篇不提新架构、只做受控变量消融的系统性实证研究：作者用一个 Mixture-of-Transformers + flow matching 的代表性 VLA testbed，配合自研的 Grouped Blind Ensemble 双盲真机评测协议，得出三条对规模化直觉的反例——EEF-relative 动作空间是跨具身鲁棒迁移的可靠默认选择；naive 地混合异构机器人数据往往带来负迁移而非正迁移（RoboCasa 上门/抽屉类任务成功率相对纯 OXE 预训练倒退 10.0%）；sensory dropout、多阶段训练课程等常见正则化手段在大规模预训练下并不稳定地带来收益，直接端到端联合优化反而更好（85.8% vs 两阶段课程 84.5%）。

## 一、问题与动机

VLA 模型延续了视觉-语言模型"scale data"就能获得更强泛化的信念，但机器人数据在具身、传感模态、控制频率与动作空间上天然异构，与文本/图像数据的同质性截然不同。作者提出三个尚无定论的核心问题：

1. **物理对齐**：什么样的动作表征（坐标系、相对/绝对）最能对齐不同运动学结构的机器人？
2. **具身混合**：跨机器人混合数据到底是带来正迁移还是引入干扰？
3. **训练正则化**：sensory dropout、多阶段课程等常见做法，在大规模预训练下是否仍然有效？

论文进一步指出真机评测本身存在实验者偏差（操作员对策略身份的熟悉度、执行中的细微调整会污染结果），因此在做规模化结论之前必须先解决评测可信度问题——这是提出 Grouped Blind Ensemble 协议的直接动机。全文的立场是：不提出新架构，而是把训练流水线本身当作研究对象，在匹配条件下系统性地消融关键设计选择。

## 二、核心方法

### 2.1 Mixture-of-Transformers 骨干与 flow matching 动作生成

Testbed 采用双专家（Semantic Expert $\mathcal{E}_{\text{sem}}$ 与 Action Expert $\mathcal{E}_{\text{act}}$）并行的 Mixture-of-Transformers（MoT）架构：$\mathcal{E}_{\text{sem}}$ 由预训练 VLM（InternVL-3.5-2B，隐藏维度 2048）初始化以保留视觉-语言先验，$\mathcal{E}_{\text{act}}$ 是随机初始化、深度相同但隐藏维度更小（0.7B 参数，隐藏维度 1024）的 transformer decoder，专职控制。输入被切分为视觉 token、指令 token、本体感觉 token、动作潜变量 token 四组，两条流通过逐层共享的因果自注意力交互：

$$\mathbf{T}^{(l)} = \left[\mathbf{T}_{\text{sem}}^{(l)}\ \mathbf{T}_{\text{act}}^{(l)}\right], \quad \mathbf{T} \in \{\mathbf{Q}, \mathbf{K}, \mathbf{V}\}, \qquad (1)$$

**用大白话说**：两个专家各自算自己的 Q/K/V，但在做注意力之前把两条流的 Q/K/V 沿序列维度拼在一起做统一的因果自注意力，这样 Action Expert 能直接看到未经文本 embedding 压缩的完整视觉-语义上下文，而不用像很多方法那样先把语义信息压成离散 token 再喂给控制头。

动作生成用条件 flow matching 建模动作 chunk 的分布 $p(\mathbf{a}_{t:t+H} \mid \mathbf{o}_t)$，训练时用插值点 $\mathbf{x}_\tau = (1-\tau)\mathbf{x}_0 + \tau \mathbf{x}_1$（$\mathbf{x}_0 \sim \mathcal{N}(0,I)$，$\mathbf{x}_1 \approx \mathbf{a}_{t:t+H}$）回归速度场：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau, \mathbf{x}_0, \mathbf{x}_1}\left[\left\|v_\theta(\mathbf{x}_\tau, \tau, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\right\|^2\right], \qquad (2)$$

其中 $\mathbf{c}$ 是 Semantic Expert 特征与本体感觉的联合条件。推理时用显式 Euler 解 ODE 生成动作序列。**用大白话说**：不直接回归动作数值，而是学一个"从噪声到动作"的速度场，一步步用欧拉法把随机噪声"喷"成一条平滑的动作轨迹，这是目前 π0 系列等主流做法的标准配方，论文本身对此不做创新，只是把它当作固定的骨干去研究训练配方。

### 2.2 物理对齐的统一动作空间与四种坐标系消融

为了让不同运动学结构（单臂夹爪到双臂灵巧手）的机器人数据能投影到同一个空间，论文定义一个物理意义明确的超集动作空间：

$$\mathcal{A}_{\text{uni}} = \mathcal{S}_{\text{eef}} \oplus \mathcal{S}_{\text{joint}} \oplus \mathcal{S}_{\text{gripper}} \oplus \mathcal{S}_{\text{hand}} \oplus \mathcal{S}_{\text{aux}}, \qquad (3)$$

每个具身 $r$ 通过嵌入映射 $\phi_r: \mathcal{A}_r \to \mathcal{A}_{\text{uni}}$ 把自己的原生动作放进对应语义子空间，未使用的维度用二值掩码禁用。**用大白话说**：不是简单地把所有机器人的动作向量拼接或补零对齐，而是先按物理含义（末端位姿、关节角、夹爪开合、灵巧手指、辅助量）分好槽位，同一物理含义的动作永远落在同一槽位里，这样模型才有可能在跨机器人间学到"共享的物理先验"而不是被无意义的维度错位干扰。

在这个统一空间之上，论文进一步定义了四种坐标系参数化，用来研究末端位姿目标 $\mathbf{T}_\tau \in SE(3)$ 应该如何相对表达：

$$\Psi(\mathbf{T}_\tau) \in \begin{cases} \mathbf{T}_\tau \ominus \mathbf{T}_0 & \text{(World-Rel)} \\ \mathbf{T}_\tau \ominus \mathbf{T}_{\tau-1} & \text{(World-Delta)} \\ \mathbf{T}_0^{-1} \circ \mathbf{T}_\tau & \text{(EEF-Rel)} \\ \mathbf{T}_{\tau-1}^{-1} \circ \mathbf{T}_\tau & \text{(EEF-Delta)} \end{cases} \qquad (4)$$

**用大白话说**：World/EEF 区分"目标是相对世界坐标系表达还是相对末端自身坐标系表达"，Rel/Delta 区分"目标是相对 chunk 起点的绝对位移还是相对上一步的逐帧增量"。四种组合覆盖了当前 VLA 文献里常见坐标系选择的完整谱系，论文把它当作第一个需要严格消融的自由变量。

### 2.3 Grouped Blind Ensemble 双盲真机评测协议

为消除真机实验中的操作员偏差，论文提出算法化的双盲协议（Algorithm 1）：把待评测模型池随机划分为不重叠的小组（组大小通常 4–8），每组内模型被匿名化为别名并随机打乱执行顺序，操作员只知道要执行哪个匿名策略、不知道其真实身份，逐条记录二值成败结果，直到全部匿名试验完成后才反匿名化统计聚合结果。**用大白话说**：把"跑策略的人"和"知道策略是什么的人"彻底分开，操作员变成纯执行者，不能靠"我知道这是哪个 checkpoint 所以我更耐心/更小心"来无意识地影响结果；分组还顺带给操作员提供了组间休息点，缓解真机评测中的疲劳效应。

### 2.4 预训练数据与实现细节

训练语料汇总了跨具身、跨域（真机/仿真）、跨控制空间（末端/关节）的四类来源：真机末端位姿数据（OXE 中的 DROID/Bridge/BC-Z/Language Table/Fractal/Kuka 子集，以及大规模私有的 Agibot 夹爪/灵巧手数据和 RoboMind 数据，覆盖 Franka、UR5、AgileX 等平台）、仿真末端/关节数据（InternData）、真机关节数据（SO-100 与人形机器人数据）。原始帧数达 6.585 亿，但各数据源规模极不均衡（数亿到数十万不等），论文用逐数据集的 `frame_step_size` 做动态降采样（例如 Agibot Gripper 步长 7，SO-100 步长 1）平衡梯度贡献，最终有效语料约 1.824 亿帧。预训练分两阶段：Stage 1 冻结 VLM backbone、只训练 Action Expert 共 40k 步；Stage 2 解冻全模型训练 200k 步；全程 batch size 256，8 张 NVIDIA A800。同时引入两种随机正则化：以 $p=0.2$ 概率整体置零本体感觉输入，以 $p=0.2$ 概率独立丢弃每个相机视角（保证至少保留一个视角）。

## 三、实验结果

### 3.1 动作空间消融（物理对齐）

LIBERO 5-shot（Table I，Frozen VLM 设置下预训练模型的平均成功率）：

| 动作空间 | Scratch | Pretrain | 提升 |
|---|---|---|---|
| World-Relative | 69.1% | 74.7% | +5.6% |
| World-Delta | 66.7% | 74.8% | +8.1% |
| **EEF-Relative** | 66.9% | **75.1%** | **+8.2%（最高）** |
| EEF-Delta | 66.1% | 72.0% | +5.9% |

RoboCasa 50-shot（Table II）呈现更明显的分化：EEF-Relative 从 45.1%（scratch）提升到 **50.0%**（+4.9%），是唯一"稳定"提升的表征；World-Delta 几乎无收益（+0.4→-0.1% 量级），EEF-Delta 在 Doors/Drawers 子任务上预训练反而倒退 7.0%。作者将其解释为：世界坐标系依赖固定相机外参与有界工作空间，容易过拟合单一环境的绝对位置规律，而 EEF 坐标系在跨相机、跨机器人基座变化时泛化性更好。

真机双盲评测（Fig. 4，四任务：Stack Bowls/Pick-to-Drawer/Wipe Board/Water Plant）进一步显示：**所有 Delta（逐帧增量）动作在真机上都出现原地抖动（jittering），成功率为 0%**；World-Relative 与 EEF-Relative 两种绝对/相对 chunk 表征平均成功率相当接近（约 44%–45%），无显著差异——这与仿真结果（EEF-Relative 明显占优）有出入，提示仿真-真机之间坐标系选择的最优解并不完全一致。

### 3.2 跨具身数据混合消融

固定动作空间为 EEF-Relative，采用累积式数据混合协议：D1（仅 OXE）→ D2（+真机末端数据）→ D3（+仿真末端数据）→ D4（+关节空间数据）。

LIBERO 5-shot，Frozen VLM（Table III）：

| 混合方案 | Avg 成功率 | 相对上一步变化 |
|---|---|---|
| Scratch | 66.9% | — |
| D1: OXE Only | 77.3% | +10.4%（相对 scratch） |
| D2: D1 + Real EEF | 73.8% | -3.5% |
| D3: D2 + Sim EEF | 72.1% | -1.7% |
| D4: D3 + Joint | 75.1% | +3.0%（仍低于 D1） |

RoboCasa 50-shot（Table IV）呈现同样但更剧烈的模式：D1 达到 54.7% 的最高水位，D2 立即跌至 48.8%（-5.9%），D3/D4 仅小幅回升至 49.6%/50.0%，最终比 D1 低约 4.7 个百分点，在 Doors/Drawers 这类关节式物体任务上更是倒退 **10.0%**。真机双盲评测（Fig. 5）同样观察到：D3（加入仿真末端数据）阶段在 Stack Bowls、Pick-to-Drawer、Wipe Board 等任务上出现不同程度的性能下降。作者的结论是：结构上差异较大的机器人数据集混合会引入破坏性干扰（destructive interference），而非带来正迁移——这与语言模型"数据越多越好"的规模化直觉相反。

### 3.3 训练正则化消融

LIBERO 5-shot（Table V）：balanced 基线（本体感觉掩码 $p=0.2$、视角丢弃 $p=0.2$、两阶段课程）平均成功率 84.5%；关闭本体感觉掩码（$p_{\text{state}}=0$）反而提升到 85.2%；关闭视角丢弃（$p_{\text{view}}=0$）提升到 85.6%；而**跳过两阶段课程、直接对全模型做单阶段端到端优化（"Stage 2 Only"）取得全表最高的 85.8%**，优于两阶段课程的 84.5%。结论是：这两种常被默认采用的正则化/课程学习策略，在异构大规模预训练下并不稳定地带来收益，甚至可能因为预训练语料本身的多样性已经起到了隐式正则化作用而显得多余。

### 3.4 与代表性 generalist VLA 的对比

在匹配的 50-shot 微调协议下（Table VI），testbed 本身（未做任何针对基准的专门优化）与 GR00T-N1、$\pi_0$、$\pi_{0.5}$ 对比：

| 模型 | LIBERO Avg | RoboCasa Avg |
|---|---|---|
| GR00T-N1 | 93.9% | 36.0% |
| $\pi_0$ | 94.4% | 42.4% |
| $\pi_{0.5}$ | 96.9% | 41.4% |
| **本文 testbed（Base）** | **97.9%** | **50.0%** |

说明该 testbed 是一个具有竞争力的高性能基座，前述消融结论并非建立在弱基线之上。

## 四、局限性

论文正文没有单列"Limitations"小节，但结合方法与实验设计可以归纳出以下客观局限：

- **单一架构范式**：所有结论都建立在 MoT + flow matching 双专家这一个 testbed 上，是否可推广到自回归离散 token 化（如 FAST/OpenVLA）或其他动作解码范式的 VLA 尚未验证。
- **累积式而非全排列消融**：D1→D4 的数据混合是单向累加协议，无法完全解耦"是真机 EEF 数据本身有害"还是"任意新增异构源都会稀释已学到的表示"，也没有测试其他添加顺序或留一法（leave-one-out）以定位负迁移的具体来源。
- **真机评测规模有限**：真机双盲实验仅在单一 Franka Panda 平台、4 个任务、每任务 10 次试验上进行，样本量偏小，双盲协议消除了操作员偏差但未消除策略间因执行风格差异（如抖动）本身泄露身份信息的问题。
- **仿真与真机结论不完全一致**：仿真中 EEF-Relative 明显最优，真机中 World-Relative 与 EEF-Relative 表现相当，论文未深入解释这一差异的成因，只停留在现象描述层面。
- **正则化消融的搜索空间较窄**：仅对 dropout 概率和两阶段/单阶段两种课程做了少量取值的消融，未覆盖更广泛的正则化设计（如渐进式课程、embodiment-specific adapter）。

## 五、评价与展望

**优点**：这是 VLA 规模化研究中少见的"控制变量实证研究"而非"新架构论文"，方法论价值可能超过具体数字——尤其是 Grouped Blind Ensemble 协议，为长期被诟病"操作员偏差污染真机对比"的具身 AI 评测提供了一个可复用、可算法化的双盲范式，具有超出本文具体结论的方法论意义。论文的核心发现（naive 跨具身数据混合带来负迁移）对当前"大力出奇迹式"收集海量异构机器人数据、直接联合训练的行业惯性是一个有价值的警示，其结论方向与 Open X-Embodiment、Octo、CrossFormer 等强调需要显式跨具身对齐机制的工作一脉相承，也与同一作者群体此前的 Being-H0（人类中心跨具身规模化学习）形成明显的研究延续性。

**与其他公开工作的关系**：论文选取的三个 baseline（$\pi_0$、$\pi_{0.5}$、GR00T-N1）都是 2024–2025 年公开发布的代表性 generalist VLA，对比设置匹配（相同 50-shot 微调协议），使结论具有一定的横向可信度。但需要注意的是，Open X-Embodiment 原始论文报告的跨具身联合训练在大规模、大模型容量下总体是正迁移的，本文在中等规模 testbed（0.7B 动作专家）上观测到的负迁移，可能与模型容量、数据配比策略（`frame_step_size` 启发式是否最优）、以及是否做了本文强调的"物理对齐"预处理有关——即负迁移未必是跨具身数据混合的必然属性，而可能是"混合了却没有做好物理对齐"或"模型容量不足以吸收异构监督信号"的产物，论文自身也承认 EEF-Relative 对齐虽然显著缓解但仍未完全消除这一负迁移。

**开放问题**：(1) 负迁移的根因究竟是共享参数空间中的梯度冲突，还是当前统一动作空间/坐标系设计仍不够精细；(2) 是否存在轻量级的 embodiment-specific 路由或适配器机制，能够在共享 backbone 收益与跨具身干扰之间取得更好平衡（例如 MoE 式的动作头、per-embodiment LoRA）；(3) Grouped Blind Ensemble 协议本身在策略执行风格差异明显（如本文观测到的 Delta 动作抖动）时如何进一步降低身份泄露；(4) 该研究的结论是否能外推到更大参数规模、更大数据规模的 VLA 预训练——论文的规模仍然是百级参数 M 到十亿参数级别，与业界最新的更大规模基座相比仍有距离，规模继续增大后"负迁移"现象是否会像语言模型中的许多能力一样被规模本身"抹平"，是一个值得后续工作验证的开放问题。

## 参考

- Black, K. et al. *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.
- Physical Intelligence. *$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization.* arXiv:2504.16054, 2025.
- NVIDIA (Bjorck, J. et al.). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734, 2025.
- O'Neill, A. et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models.* ICRA, 2024.
- Luo, H. et al. *Being-H0: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization.* arXiv:2601.12993, 2026.
