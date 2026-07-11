# General Flow：将“通用流”作为可扩展机器人学习的基础可供性

> **论文**：*General Flow as Foundation Affordance for Scalable Robot Learning*
>
> **作者**：Chengbo Yuan, Chuan Wen, Tong Zhang, Yang Gao（通讯作者）et al.
>
> **机构**：清华大学交叉信息研究院（IIIS, Tsinghua University）；上海期智研究院（Shanghai Qi Zhi Institute）；上海人工智能实验室（Shanghai AI Laboratory）
>
> **发布时间**：2024 年 01 月（arXiv 2401.11439，v2 于 2024 年 09 月）
>
> **发表状态**：CoRL 2024（第 8 届 Conference on Robot Learning，Munich, Germany）
>
> 🔗 [arXiv](https://arxiv.org/abs/2401.11439) | [PDF](https://arxiv.org/pdf/2401.11439)
>
> **分类标签**：`3D flow` `human video` `zero-shot transfer` `affordance` `cross-embodiment`

---

## 一句话总结

把“物体上一批 3D 点在未来若干时刻的轨迹”（General Flow / 通用流）当作一种跨本体、跨物类的通用可供性预测目标，直接在大规模 RGBD 人类视频上训练一个语言条件的 3D flow 预测模型 ScaleFlow，再配一个基于 SVD 对齐的启发式闭环策略，**无需任何域内微调**即可在 6 个场景、18 个任务上实现 81%（146/180）的零样本 human-to-robot 技能迁移成功率。

## 一、问题与动机

作者的出发点是把 LLM 的成功范式搬到机器人操作上：LLM 之所以泛化强，靠的是（1）推理域间隙极小的海量训练数据（互联网文本）与（2）一个基础性的预测任务（next-token）。机器人要复刻这两点，必须回答两个问题：

- **数据从哪来？** 真机数据采集昂贵且难扩展，而仿真存在 sim-to-real 域间隙。人类视频恰好是海量、真实、富含物理交互的资源，且与机器人操作行为高度对齐，推理域间隙小。
- **预测什么目标？** 需要一个对所有物类通用、又能为操作提供可执行指引的“可供性”（affordance，源自 Gibson 理论）。作者聚焦 **抓取后的运动（post-grasp motion）** 这一长期难点——功能性抓取已被大量工作解决，但抓取之后该往哪动、怎么动仍缺乏通用方法。

作者提出用 **General Flow**（物体上关注点的未来 3D 轨迹）作为这个基础可供性。以“开保险箱”为例：门上点的通用流描绘出门板未来的旋转轨迹，机器人只要跟随门板的 flow 就能得到一个鲁棒的开门运动原语。相比此前从仿真或真机提取 flow 的工作（FlowBot3D、ToolFlowNet、Flowbot++ 等），General Flow 的优势被归纳为三点：**(1) scalability**（利用真实、可扩展的跨本体人类数据）；**(2) wide application**（覆盖刚体、铰接体、软体多种物类）；**(3) stable skill transfer**（推理域间隙小，足以支撑零样本执行）。

## 二、核心方法

### 2.1 General Flow 的定义

给定任意本体的感知观测 $S$（本文用 RGBD 相机流重建的场景点云）与任务指令 $I$，对 $N_q$ 个 3D 查询点 $Q \in \mathbb{R}^{N_q \times 3}$，通用流 $F \in \mathbb{R}^{N_q \times T \times 3}$ 表示这些点在未来 $T$ 个时刻的轨迹。第 $i$ 个查询点的轨迹记为 $F^i \in \mathbb{R}^{T \times 3}$，其在时刻 $t$ 的绝对位置为 $F_t^i \in \mathbb{R}^3$，初始 $F_0^i$ 即输入点 $p^i$。场景点云 $P_s \in \mathbb{R}^{N_s \times 6}$（XYZ+RGB）。

**用大白话说**：不预测机器人动作、也不预测像素，而是预测“物体上这些点接下来会飘到哪里”，这是一种既与本体无关、又与物类无关的中间表征——人手推还是机械臂推，门板点的运动轨迹是一样的。

### 2.2 标签抽取管线（数据侧）

从两类跨本体数据里自动提取 flow 标签：

- **HOI4D 数据集**（刚体+铰接体）：16 类、800 个物体、44.4 小时录制，自带 3D 分割、6D 位姿、相机参数。在活动物体 mask 内随机采点，用真值位姿+相机参数直接算出未来位置。
- **自采 RGBD 视频**（软体，“叠衣服”任务）：D455 采集 6 类衣物、30 条轨迹、605 段 clip。先做 HOI 检测+FastSAM 分割得到活动物体 mask，在 mask 内采 1024 点，用 Tracking-Any-Point（co-tracker）追踪 2D 轨迹，再反投影回 3D。

统一切成 1.5s 的 clip、0.15s 间隔、$T=3$ 步。数据划分 80/10/10，得到 51693/6950/6835 段 clip，且训练/验证/测试集之间无相同物体实例。

### 2.3 Scale-Aware 预测模型（ScaleFlow）

模型（图 3）用 CLIP 编码语言 → MLP 降维到 $d_I$ 对齐点特征，与场景点云 XYZ+RGB 及查询点拼成 $P_M \in \mathbb{R}^{(N_s+N_q)\times(3+3+d_I)}$，送入带分割头的 **PointNeXt** backbone 抽几何特征；查询点特征作为条件变量喂给一个 **条件 VAE**，以刻画“同一任务同一场景下人可能有多种运动方式”的多模态性。

关键设计是**尺度感知**。作者发现预测相对位移 $\Delta p_t^i = F_t^i - F_{t-1}^i$ 比预测绝对位置更好。为应对不同查询点轨迹长度差异巨大（“开保险箱”门板点比门轴附近点位移大得多），引入 **Total Length Normalization（TLN）**：先定义每个点的轨迹尺度 $L^i = \sum_{t=1}^{T}\lVert\Delta p_t^i\rVert$，再归一化轨迹形状：

$$
\Delta n_t^i = \frac{\Delta p_t^i}{L^i}
$$

VAE 分别预测尺度 $L^i$ 与归一化轨迹 $\Delta n_t^i$。**用大白话说**：把“往哪个方向走、走成什么形状”和“走多远”解耦——形状归一化后大家分布一致好学，长度另开一个头单独回归，避免长轨迹的点主导了短轨迹的点。消融显示 TLN 优于 TDN（按绝对位移归一）与 SDN（每步归一到 1）。

训练总损失（尺度损失权重 $\beta_1=25$，$\beta_2,\beta_3=1$）：

$$
\mathcal{L} = \mathcal{L}_{traj} + \beta_1 \mathcal{L}_{scale} + \beta_2 \mathcal{L}_{KL} + \beta_3 \mathcal{L}_{acc}
$$

其中 $\mathcal{L}_{traj}=\frac{1}{N_q}\sum_{i,t}\lVert\Delta\hat{n}_t^i-\Delta n_t^i\rVert^2$（归一化轨迹）、$\mathcal{L}_{scale}=\frac{1}{N_q}\sum_{i,t}\lVert\hat{L}^i-L^i\rVert^2$（尺度回归）、$\mathcal{L}_{KL}$ 为 VAE 的 KL 项、$\mathcal{L}_{acc}=\frac{1}{N_q}\sum_{i,t}\lVert\hat{F}_t^i-F_t^i\rVert^2$（对累积复原的绝对轨迹做 MSE，抑制累积误差）。**用大白话说**：形状、长度、多模态正则、累积一致性四项一起管，尤其重视尺度（权重 25），因为尺度错了跟随距离就错了。

### 2.4 三个鲁棒性设计（迁移侧）

- **Scale Rebalance**：数据里静止点（如保险箱箱体）占多数，直接训会偏向预测静止。用 K-Means 按尺度 $L^i$ 把点聚成 $N_r$ 类（默认 4），除最大类外做重采样，重采样比例按温度 $\tau$（默认 1）的 softmax 平滑：

$$
\bar{r}_i = \frac{e^{r_i/\tau}}{\sum_{i=1}^{N_r} e^{r_i/\tau}}
$$

**用大白话说**：把“会动的点”过采样一下，别让一堆不动的背景点把模型带偏成“预测啥都不动”。

- **Hand Mask Augmentation（HMA）**：训练时人手会遮挡、部署时机械臂会遮挡。以 $p_{h1}=0.5,p_{h2}=0.2,p_{h3}=0.3$ 三种规则随机处理手部点（全删/全留/只留距锚点 12cm 外的点），模拟无遮挡/全遮挡/部分遮挡。
- **Query Points Sampling（QPS）**：以 $p_{s1}=0.7,p_{s2}=0.3$ 在“完全随机采样”与“锚点附近采样”间切换，让模型适应下游不同的查询点分布。

### 2.5 下游启发式闭环策略

用一台 RealSense D455 在 Franka-Emika 臂后方拍 ego-view。以机器人静止基座作 prompt 让 FastSAM 分割出机器人 → 重建场景点云 → 在夹爪附近 10cm 内选查询点（当作一个微型刚体）→ ScaleFlow 预测 flow → 用 **SVD** 求解把夹爪位姿对齐到预测 flow 的 SE(3) 变换（一个加权 ICP 问题，权重与查询点到夹爪距离成反比）：

$$
w_i = \frac{1/(d_i+\beta)}{\sum_{j=1}^{N} 1/(d_j+\beta)}, \quad \hat{R},\hat{T} = \arg\min_{R,T} w_i \lVert k_{t+1}^i-(R\cdot k_t^i+T)\rVert^2
$$

再用 Deoxys 阻抗控制器执行，闭环循环（ROS 20Hz）。**用大白话说**：flow 只是告诉你“物体点该怎么动”，把夹爪当成一小块刚体、用 SVD 反解出“夹爪该怎么平移旋转才能跟上这团点”，就得到了一个不需要学策略网络的动作。抓取起点仍由人工摆放（可被自动抓取方法替代）。

## 三、实验结果

### 3.1 Flow 预测精度（测试集，单位 cm；PM=参数量百万）

| Model | ADE↓ | FDE↓ | Param(M) |
|---|---|---|---|
| ResNet18 | 7.54 | 10.71 | 13.2 |
| R3M (frozen) | 7.55 | 10.56 | 11.9 |
| R3M (finetune) | 7.54 | 10.69 | 11.9 |
| VAT-MART | 7.16 | 12.20 | 1.6 |
| ViT-B-224 | 6.81 | 9.48 | 86.6 |
| PointNeXt-B | 3.96 | 5.37 | 4.1 |
| PointNeXt-L | 3.83 | 5.16 | 15.6 |
| **ScaleFlow-S** | 3.74 | 5.01 | **0.9** |
| **ScaleFlow-B** | 3.58 | 4.77 | 5.6 |
| **ScaleFlow-L** | **3.55** | **4.70** | 17.1 |

结论：2D 模型（ResNet/R3M/ViT）远逊于 3D 点云模型，凸显 3D 几何信息的重要性；ScaleFlow 在所有指标上均优于最强的 3D baseline，且参数更少（ScaleFlow-S 仅 0.9M 就超过 86.6M 的 ViT）。

### 3.2 零样本真机操作（Franka-Emika，单模型跑全部任务）

| Object | Action-1 | SR-1 | Action-2 | SR-2 | Action-3 | SR-3 |
|---|---|---|---|---|---|---|
| Mug | pickup | 10/10 | putdown | 9/10 | – | – |
| Toy Car | pickup | 10/10 | putdown | 10/10 | push | 5/10 |
| Clothes | fold | 8/10 | – | – | – | – |
| Safe | open | 9/10 | close | 10/10 | – | – |
| Box | open | 10/10 | close | 10/10 | – | – |
| Drawer | open(pull) | 4/10 | open(grasp) | 3/10 | close | 10/10 |
| Refrigerator | open(pull) | 7/10 | open(grasp) | 9/10 | close | 10/10 |
| Laptop | open | 5/10 | close | 7/10 | – | – |
| **平均** | | | | | | **81% (146/180)** |

除 Toy Car、Safe 外均为训练未见的新实例；环境背景随机搭建，均为训练时未见。作者强调这是（据其所知）首个基于 flow 达到此级别真机零样本迁移的工作。有趣的是“Box”开门成功率（100%）反而高于普通“Safe”（90%），因为 Box 结构允许更大轨迹偏差而不脱手。

### 3.3 消融（Flow 精度，Test-ADE/FDE，数值与 Table 1 一致，单位 cm）

| 变体 | Test-ADE↓ | Test-FDE↓ |
|---|---|---|
| Full | **3.58** | **4.77** |
| w/o Text EarlyFusion | 3.70 | 4.95 |
| w/o Scale Norm (relative) | 3.76 | 5.04 |
| w/o Scale Norm (absolute) | 3.81 | 5.12 |
| w/ TDN Scale Norm | 3.74 | 5.00 |
| w/ SDN Scale Norm | 3.77 | 5.10 |
| w/ $\beta_1=1$ | 3.68 | 4.93 |
| w/o central crop | 3.99 | 5.38 |
| w/o Scale Rebalance | 3.59 | 4.78 |
| w/o HMA | 3.66 | 4.88 |
| w/o QPS | 3.58 | 4.77 |

关键观察：去掉中心裁剪（central crop，即不把点云裁到物体附近 $80^3\,\mathrm{cm}^3$）掉点最多（3.99/5.38），因为活动物体点只占约 2%；scale normalization 与 early fusion 都有明显贡献；三个鲁棒性增广（Rebalance/HMA/QPS）在**基准精度**上几乎无损甚至 HMA 还略降 in-domain 误差——但它们真正的价值体现在真机。

### 3.4 真机消融（成功率）

| 变体 | open Safe | close Drawer | Avg |
|---|---|---|---|
| full | 9/10 | 10/10 | **95%** |
| w/o Scale Rebalance | 7/10 | 8/10 | 75% |
| w/o HMA | 8/10 | 6/10 | 70% |
| w/o QPS | 5/10 | 4/10 | 45% |

这张表是全文最有说服力的证据之一：三个增广在 in-domain flow 误差上几乎看不出差别，却把真机零样本成功率从 45% 拉到 95%——说明**分布内测试精度与真机迁移鲁棒性严重脱节**，QPS 影响最大（因为 PointNeXt 把查询点特征抽取与场景耦合了）。

### 3.5 涌现性质与推理开销

大规模训练涌现出四种性质并在真机验证：**语言语义可控**（同一点云给“open”/“close”得到相反 flow）、**对标签噪声鲁棒**、**尺度空间自适应**（putdown Mug 时高位平均预测 19.04cm、低位 5.94cm）、**对手部遮挡鲁棒**。推理延迟 405.7ms（2.5Hz），其中 **FastSAM 分割占 347.6ms（85.7%）** 是瓶颈；去掉分割后仅 58.1ms（17.5Hz），flow 预测本身仅 22.1ms。

## 四、局限性

作者在附录 I 坦诚列出：

1. **数据多样性与体量仍有限**，复杂任务的引导不足；未来可用更大 RGBD 数据集或“RGB+深度估计”扩展。
2. **CVAE 表达力不足**，面对更大更多样数据可能不够，Diffusion 是潜在替代。
3. **抓取起点靠人工摆放**，虽可被自动抓取方法替代，但当前 pipeline 未闭环。
4. **ICP/SVD 启发式策略在零样本下限制了 contact-rich 任务**（如高摩擦抽屉、薄笔记本盖导致点云残缺）；成功率 <60% 的任务（push Toy Car 50%、open Drawer 30% grasp、open Laptop 50%）多因语义理解不足、点云质量差或结构混淆（HOI4D 里 revolute/prismatic 混合致预测偏旋转）。
5. **PointNeXt 把查询点特征与场景耦合**，导致对 QPS 增广的强依赖，需未来解耦架构解决。
6. 对短于 5cm 的轨迹，Deoxys 操作空间控制器精度不足，需把所有步合并放大到 5cm 的工程 workaround（约 25% 的预测触发），说明控制层还较粗糙。

## 五、评价与展望

**优点。** 这篇工作最本质的贡献是把“3D flow”确立为一个**兼具可扩展性与可执行性**的中间可供性表征，且第一次真正做到直接从大规模 RGBD 人类视频端到端训练、零样本上真机。它精准地卡在两个极端之间：一端是纯表征预训练（R3M/MVP/VIP 等，泛化到操作有限），另一端是仿真可供性（FlowBot3D/Where2Act，受 sim-to-real 限制）；3D flow 既避免了 sim 域间隙，又比 2D flow（如 ATM、Track2Act）多了深度信息因而能直接反解 SE(3) 闭环动作。TLN 的尺度-形状解耦、以及“in-domain 精度几乎无差别但真机成功率差 2 倍”的消融，是很有价值的经验教训——它提醒后续工作，**flow 预测的离线指标（ADE/FDE）并不能预测真机迁移鲁棒性**。

**缺点与开放问题。** (1) 策略层是全文最弱环节：SVD 对齐本质是把物体 flow 硬套到夹爪刚体上，只适合“抓稳后跟随刚性运动”的任务，对形变、滑动、力控几乎无能为力；后续的 Track2Act、ATM、以及各类 flow-conditioned policy（如把 flow 作为 diffusion policy 的条件）正是沿这条线补齐。(2) 抓取仍靠人工摆放，使“端到端零样本”的说法打了折扣。(3) 评测规模偏小（每任务 10 trial、单臂单场景族），81% 这个数字需在更大规模上复核。(4) CVAE 的多模态在真机里其实没被充分利用（策略只取一条），多模态价值存疑。

**与后续工作的关系。** General Flow 可视为“human video → 中间可供性 → 机器人策略”这条路线的一个奠基点，与同期/后续的 Track2Act（2D 点轨迹）、ATM（any-point trajectory + 需域内微调）、以及各类以 flow/点轨迹为预训练目标的 VLA 形成清晰谱系。可能的改进方向包括：用 diffusion 替换 CVAE 建模多模态、把 SVD 启发式换成 flow-conditioned 的学习型策略（few-shot 模仿或把 flow 当 RL 约束）、把物体中心 flow 扩展到通用关键点/稠密场景 flow、以及用更快的开放词表分割替换 FastSAM 以突破 2.5Hz 瓶颈。

## 参考

1. Eisner et al. *FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects.* arXiv 2205.04382, 2022.（3D flow 可供性的仿真前身）
2. Seita et al. *ToolFlowNet: Robotic Manipulation with Tools via Predicting Tool Flow from Point Clouds.* CoRL 2023.（本文视为其“数据与应用双重推广”的对象）
3. Bahl et al. *Affordances from Human Videos as a Versatile Representation for Robotics (VRB).* CVPR 2023.（唯一可比的真机人类视频可供性 baseline）
4. Wen et al. *Any-point Trajectory Modeling for Policy Learning (ATM).* arXiv 2401.00025.（2D 点轨迹预训练目标，需域内微调）
5. Liu et al. *HOI4D: A 4D Egocentric Dataset for Category-level Human-Object Interaction.* CVPR 2022.（主训练数据源）
