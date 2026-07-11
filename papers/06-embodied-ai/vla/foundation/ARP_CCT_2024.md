# ARP：面向机器人操作的自回归动作序列学习

> **论文**：*Autoregressive Action Sequence Learning for Robotic Manipulation*
>
> **作者**：Xinyu Zhang, Yuhan Liu, Haonan Chang, Liam Schramm, Abdeslam Boularias
>
> **机构**：Rutgers University 计算机科学系
>
> **发布时间**：2024 年 10 月（arXiv 2410.03132）
>
> **发表状态**：IEEE Robotics and Automation Letters (RA-L)，2025 年 2 月录用
>
> 🔗 [arXiv](https://arxiv.org/abs/2410.03132) | [PDF](https://arxiv.org/pdf/2410.03132)
>
> **分类标签**：`自回归策略` `动作分块` `causal transformer` `多模态动作空间` `RLBench/ALOHA/Push-T`

---

## 一句话总结

提出 Chunking Causal Transformer（CCT）：把标准 causal transformer 的"逐 token 预测"扩展为"单步预测可变长度 token 块"，并据此构建统一架构 Autoregressive Policy（ARP），用同一套模型在 Push-T、ALOHA、RLBench 三种控制模式迥异的环境中都匹配或超过各自的专用 SOTA（Diffusion Policy / ACT / RVT-2），且计算量和参数量更小。

## 一、问题与动机

- 语言模型的自回归生成（下一个 token）已被 Decision Transformer / Trajectory Transformer / Gato / VIMA / ManipLLM 等引入机器人控制，但这些方法把动作视为像单词一样的离散 token 序列，一次只生成一个端点式（end-effector waypoint）动作，天然只适合低频控制任务。
- 机器人动作与自然语言本质不同：动作是异构的（离散抓手状态、连续关节角、2D 像素坐标、6DoF 末端位姿混杂），且高频控制（如 ALOHA 50Hz 关节控制）需要动作在时间上平滑，这是语言建模里不存在的约束。
- Action Chunking Transformer（ACT）已经引入"分块预测"思路，但 ACT 是单步、固定块大小（一次性预测全部动作），不具备自回归结构，也不能对不同动作类型使用不同块大小。
- 核心问题：能否设计一个同时具备（a）自回归的因果建模能力、（b）分块预测的多步/高频适配能力、（c）可变块大小以混合高层稀疏动作与低层稠密动作的统一架构？

## 二、核心方法

### 2.1 动作序列建模

不同任务的动作没有统一"词表"：Push-T 用 2D 像素坐标点，ALOHA 用 14 维关节角，RLBench 用 6DoF 末端位姿 + 离散抓手状态。ARP 为每类动作设计专门的嵌入 / 解码器：

- 离散动作：查表得到嵌入，解码为类别分布（线性层 + softmax）；
- 连续动作：线性层嵌入，解码为高斯混合模型（GMM）参数；
- 像素坐标动作：在视觉特征图上按坐标取点特征作为嵌入；解码时用嵌入与视觉特征图做乘法后经 RAFT 风格的上采样算子得到 2D 热力图分布。

### 2.2 Chunking Causal Transformer（CCT）

标准 causal transformer 每步只把"最后一个 token"变成"下一个 token"（单 token 预测）。CCT 的关键改动：在输入动作 token 序列 $a_1,\dots,a_n$ 后拼接一组"空 token" $e_1,\dots,e_k$（代表待预测的未来动作），通过三种注意力组合让一次前向传播同时预测多个未来 token（chunking autoregression）：

1. 空 token 之间做**双向自注意力**；
2. 空 token 到动作 token 做**因果注意力**（只能看已知的过去动作）；
3. 动作 token 之间做**因果自注意力**（与标准 causal transformer 相同）。

大白话：把"要预测的位置"先占好几个空位，让模型在一次计算里同时把这几个空位填上，而不是像 GPT 那样一个字一个字蹦出来；空位之间互相看得到，但都不能偷看还没揭晓的历史。

### 2.3 训练技巧：Attention Interleaving

Teacher-forcing 训练标准 causal transformer 只需一次前向传播即可算出所有位置的 loss（$a_1,a_2\to a_2$，$a_1,a_2,a_3\to a_3$ 等可以在一个序列内并行完成）。但朴素地训练 CCT 需要为每个 chunk 单独跑一次前向（如预测 $a_4$ 用 $a_1,e_2,e_3\to a_4$，预测 $a_2,a_3$ 又要单独跑 $a_1\to a_2,a_3$），显著增加训练开销：对 $N$ 个 chunk、chunk size $K$，朴素方法的 MAC 总量为 $\sum_{n=1}^N (nK)^2$。

Attention Interleaving 通过拆分并复用计算——先算好所有动作 token 间的因果注意力并缓存，因为"动作 token 的因果注意力不依赖未来的空 token"——把 MAC 降到

$$2(NK)^2 + NK^2$$

用大白话说：动作 token 之间的相互关注结果和后面要预测哪几个空位无关,算一次存起来重复用,就不用为每一种块划分方式都重新算一遍全部注意力；论文测算当 chunk 数 $N=5$ 时新方法的 MAC 只有朴素方法的约 6%。该技巧只在训练时使用，不增加推理成本。

### 2.4 可变块大小与混合动作序列设计

CCT 允许不同动作类型使用不同 chunk size，从而把高层规划信号和低层执行信号编排进同一条自回归序列：

- **Push-T**：先自回归预测稀疏的高层 2D waypoint，再预测连接这些 waypoint 的稠密低层像素坐标序列，类比分层规划；
- **ALOHA**：先预测关节角，再预测末端 waypoint（类比正运动学）；
- **RLBench**：直接预测目标末端位姿的粗定位 + 精定位 + 旋转 + 抓手状态；
- **真机拧螺母任务**：先预测离散的高层指令（reach / adjust / screw），再预测对应连续动作值。

### 2.5 整体架构与推理

ARP = 视觉 backbone（Push-T / ALOHA 用 ResNet50，RLBench 用 Multi-View Transformer / MVT，与对比的 SOTA 方法一致）+ CCT。CCT 与视觉特征做交叉注意力，序列内部自注意力对输入动作是因果的、对空 token 是双向的。推理时按 Model Predictive Control 方式滚动：自回归采样生成一个 chunk 的未来动作分布并采样，执行后更新观测再重新预测。

## 三、实验结果

三个环境架构相同，仅替换动作序列设计和视觉 backbone；均使用与对应 SOTA 方法相同的训练数据量、episode 数、优化器配置与评测频率。

### 3.1 与各环境专用 SOTA 对比（成功率 %）

| 方法 | Push-T | ALOHA Cube Transfer | ALOHA Insertion | RLBench (avg) |
|---|---|---|---|---|
| Diffusion Policy | 78.8 | 10 | 1.6 | – |
| ACT | 77.5 | 80.8 | 20.8 | 69.8 |
| RVT-2 | – | – | – | 81.4 |
| **ARP (Ours)** | **87.1** | **94** | **24.8** | **81.2** |

计算效率对比（THOP 测算的 MACs / 参数量）：

| 环境 | 方法 | 成功率 | MACs | 参数量 |
|---|---|---|---|---|
| Push-T | Diffusion Policy | 78.8 | 6.8G | 25.5M |
| Push-T | ARP | 87.1 | 3.7G | 23.5M |
| ALOHA | ACT | 20.8 / 80.8 | 17.8G | 50.9M |
| ALOHA | ARP | 24.8 / 94 | 17.8G | 47.6M |
| RLBench | RVT-2 | 81.4 | 57.1G | 72.1M |
| RLBench | ARP / ARP+ | 81.6 / 84.9 | 57.4G | 71.9M / 73.8M |

ARP+（层数更多的变体）在 RLBench 18 个任务上的平均排名（rank）为 1.61，优于 RVT-2 的 2.22 和 ARP 的 1.89。

### 3.2 消融：性能提升来自哪里

在 Push-T 和 ALOHA 上对比三种动作预测范式（仅改变 chunk size 设置，其余不变）：

| 策略 | Push-T | ALOHA Cube Transfer | ALOHA Insertion |
|---|---|---|---|
| SoTA（各环境专用方法） | 78.8 | 80.8 | 20.8 |
| Action Chunking（chunk=全序列长度） | 77.6 | 81.2 | 21.2 |
| 单 token 自回归（chunk=1） | 82.4 | 46 | 6.8 |
| **Chunking 自回归（ARP，可变 chunk）** | **87.1** | **94** | **24.8** |

结论：单纯加大 chunk（等价于 ACT 式一次性分块）或单纯逐 token 自回归都不如"可变块大小的分块自回归"，说明性能提升的关键因素是 chunking autoregression 本身而非额外输入信息（论文专门验证了 Push-T / ALOHA 的高层信息是从示教轨迹自动抽取的，并非额外监督）。

### 3.3 Chunk size 的影响

- ALOHA insertion 任务：关节位置的 chunk size 从 1→100，成功率从 6.8% 单调升到 24.8%，说明高频关节控制任务从更大 chunk 中显著受益；
- Push-T 低层轨迹：chunk size 从 1 到 16 成功率在 85.9%–87.1% 间小幅波动，说明低层 chunk size 相对不敏感（可用较小 chunk 结合"仅执行前几步再重新推理"降低误差累积）；
- Push-T 高层 waypoint：chunk size 从 1→4，成功率从 82.5%→87.1% 单调上升；
- 结论：最优 chunk size 依赖任务和动作序列设计，因此 CCT 支持可变 chunk size 是性能最大化的关键。

### 3.4 真机实验：拧螺母任务

用 Kuka LBR iiwa + 单个 RealSense D415（480×640 RGB-D），MVT 视觉 backbone，70 条示教（示教标签来自使用 FoundationPose 的专家策略）。任务要求螺母与扳手对齐公差 2mm，ARP 先预测高层指令（reach / adjust / screw）再预测对应低层连续动作值。

| 方法 | 成功率 |
|---|---|
| 专家策略（无人工干预） | 3/10 |
| **ARP (Ours)** | **8/10** |

鲁棒性测试：在插入位姿法平面方向加 $-5\text{mm}\sim5\text{mm}$ 均匀噪声后，ARP 仍成功 6/10，平均每次 trial 需要 1.6 次调整动作。

### 3.5 附加能力：似然估计与人类引导条件生成

利用自回归的乘法法则 $P(a_1,\dots,a_n)=\prod_{i=2}^n P(a_i \mid a_1,\dots,a_{i-1})P(a_1)$，ARP 可为任意给定轨迹估计似然，总体上给有效轨迹分配更高似然，但作者也承认部分似然排序不完全符合直觉（不能保证是可靠的轨迹质量代理）。此外 ARP 支持条件于人类手绘初始轨迹继续生成：给定正确引导时能顺利完成 / 纠正错误，但给定错误引导时会出现放大误差的分布外行为（因训练数据只包含成功示教）。

## 四、局限性

1. **动作序列格式与 chunk size 需人工设计**：机器人动作缺乏统一"词表"，每个环境 / 任务都要手工决定动作序列排列（哪些先预测、哪些后预测）以及每类动作的 chunk size，论文在讨论部分明确指出这是主要限制，并提出未来需要一种通用机器人动作语言来降低这一设计成本。
2. **似然估计不总是可解释**：部分轨迹的似然排序与其实际任务表现（是否更接近目标）不一致，作为主动学习 / 样本筛选信号的可靠性有限。
3. **分布外脆弱性**：由于训练数据仅包含成功示教，条件于错误人类引导时模型倾向于放大误差而非稳健纠正。
4. **不显式做状态 / 环境层面的规划**：当前 ARP 只对动作序列做自回归，未对未来状态（图像 / 点云）建模，缺乏显式的前瞻规划能力，作者将其列为未来工作方向（结合近期的混合自回归架构，如 diffusion forcing，生成未来状态）。
5. RLBench 上的低层轨迹执行仍依赖内置 RRT 规划器，ARP 本身只预测关键末端位姿，并非端到端连续控制。

## 五、评价与展望

**优点**：CCT 的"分块自回归"思路简洁但抓住了一个真实痛点——现有方法要么像 GPT 一样每步只出一个动作 token（低频、推理慢），要么像 ACT 那样固定块一次性输出（无法建模块内块间不同粒度）。论文用同一套架构、同一套代码在控制模态迥异的三个 benchmark（2D 低维 / 高频推子、14 维双臂高频关节、6DoF 离散关键位姿）都匹配或超过各自专用 SOTA，是"一个架构打天下"路线的有力证据，且计算 / 参数效率优于对比方法。附录中对 RVT-2 训练数据加载器中 timestep 泄漏问题的排查也是有价值的工程贡献，纠正了 RLBench 社区长期沿用的一个隐性缺陷。

**与其他公开工作的关系**：相比 Decision Transformer / Trajectory Transformer / VIMA / Gato 等早期自回归控制方法，ARP 的关键区别是把 chunk size 从"1"泛化为"可变的多"；相比 ACT，ARP 把"一次性分块"泛化为"自回归式分块串联"，因而能表达 ACT 无法表达的层级化（高层稀疏点→低层稠密轨迹）动作设计；相比 RVT-2，ARP 用单一自回归策略网络替代了 RVT-2 手工设计的两阶段粗精定位网络和 Location Conditioned Rotation 模块，架构更统一但精度相当略优。

**开放问题与可能的改进方向**：（1）如何自动化 / 学习动作序列的分块结构和排列顺序，而非人工设计，是论文自己指出的最直接的后续工作；（2）似然估计用于主动学习 / 数据筛选的可靠性有待改进，可能需要结合不确定性估计或与任务奖励对齐的校准；（3）将自回归扩展到同时生成未来状态（视觉 / 点云）与动作，实现真正的"世界模型式"前瞻规划，是作者点出但尚未实现的方向；（4）分布外时的误差放大问题提示需要引入失败 / 纠错型示教或在线交互式学习（作者在讨论部分提及但未实验验证）；（5）论文的效率对比基于 MACs / 参数量这类静态指标，未报告实际推理延迟（wall-clock），在真实机器人高频闭环控制中这一点更具决定性，值得后续工作补充。

## 参考

1. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS 2023.
2. Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）. RSS 2023.
3. Goyal et al. *RVT-2: Learning Precise Manipulation from Few Demonstrations*. arXiv:2406.08545, 2024.
4. James et al. *RLBench: The Robot Learning Benchmark & Learning Environment*. IEEE RA-L, 2020.
5. Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*. arXiv:2407.01392, 2024.
