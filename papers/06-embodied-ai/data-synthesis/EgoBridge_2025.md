# EgoBridge：面向第一人称人类数据可泛化模仿的域适应框架

> **论文**：*EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data*
>
> **作者**：Ryan Punamiya, Dhruv Patel, Patcharapong Aphiwetsa, Pranav Kuppili, Lawrence Y. Zhu, Simar Kareer, Judy Hoffman, Danfei Xu（后三人为 equal advising）
>
> **机构**：Georgia Institute of Technology（佐治亚理工学院）
>
> **发布时间**：2025 年 09 月（arXiv 2509.19626）
>
> **发表状态**：NeurIPS 2025（39th Conference on Neural Information Processing Systems）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.19626) | [PDF](https://arxiv.org/pdf/2509.19626)
>
> **分类标签**：`人机跨本体` `域适应` `Optimal-Transport` `第一人称人类数据` `模仿学习`

---

## 一句话总结

EgoBridge 把"人类演示 vs 机器人演示"建模成一个**域适应(domain adaptation)** 问题,用 **Optimal Transport (OT)** 对齐 policy 隐空间特征与动作的**联合分布**(而非仅对齐观测边缘分布),并用 **Dynamic Time Warping (DTW)** 动作轨迹距离来塑造 OT 代价、构造行为伪配对;在三个真实操作任务上相对"人类增广"跨本体基线取得最高 **44% 绝对成功率** 提升,并能把**只在人类数据里出现过的** 物体/场景/动作迁移到机器人执行,而基线在这些设定下几乎全军覆没。

## 一、问题与动机

用可穿戴设备(智能眼镜/XR)采集的第一人称人类经验数据(embodied human experience)既便宜又可规模化,而且同时包含**观测**(egocentric RGB)和**动作**(手部位姿),因此比无结构的互联网视频更有价值——原文强调这让人类数据与机器人数据可被当作"同一连续演示谱系"来统一学习。

但人机之间存在多重 **domain gap**:视觉外观、传感器模态(机器人有腕部相机而人类数据没有)、运动学差异、以及人类往往比机器人快 2-3 倍造成的**时序错位**。作者指出:

- 天真地把人机数据混在一起 co-train(如 EgoMimic 那样做 visual masking + 数据归一化)并不能自动带来有效知识迁移;近期研究(引用 [2] Wei et al. 2025)也表明简单跨域 co-train 不保证正迁移。
- 标准域适应方法(adversarial training、MMD)只对齐观测的**全局边缘分布**,会丢弃对策略学习至关重要的**动作相关信息**——而在模仿学习中观测与动作是时序耦合的,协变量偏移(covariate shift)会随时间累积。

因此本文把问题形式化为:人机数据是两个带标注的分布,观测端因本体差异存在显著 **covariate shift** ($\mu_H \neq \mu_R$),需要在对齐隐空间的同时**保留动作相关信息**。目标有两级泛化:(1) **observation generalization**——两域都有的任务要跨越视觉/传感器 gap;(2) 更进一步的 **behavior generalization**——让机器人执行**只在人类数据里见过** 的新行为(如目标位姿的空间变化)。

## 二、核心方法

### 2.1 把跨本体模仿写成联合分布 OT 对齐

设共享编码器 $f_\phi: \mathcal{O}^H \cup \mathcal{O}^R \to \mathcal{Z}$ 把人、机观测都投到共享隐空间 $\mathcal{Z}$,策略 $\pi_\theta$ 把 $z$ 映到动作。基础的 co-train 目标是聚合数据上的 BC 损失:

$$\mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) = \mathbb{E}_{(o,a)\sim D_H \cup D_R}\big[\mathcal{L}_{\text{BC}}(\pi_\theta(f_\phi(o)), a)\big]$$

**用大白话说**:先假装人机数据都是同一批人的示范,直接学"看到 $o$ 就输出 $a$"。但它默认了隐空间会自动域不变,现实中并不成立(作者实验里 co-train 的隐空间会形成人机分离的两团簇)。

EgoBridge 的核心是:不去对齐观测边缘 $P(f_\phi(O))$,而是对齐**特征-动作联合分布** $P(f_\phi(O), A)$。给定人、机的 mini-batch,用可微 Sinkhorn OT 计算联合 OT 损失:

$$\mathcal{L}_{\text{OT-joint}}(\phi) = \sum_{i,j} (T_\epsilon^*)_{ij}\cdot\mathcal{C}\Big((f_\phi(o_i^H), a_i^H),\, (f_\phi(o_j^R), a_j^R)\Big)$$

其中 $T_\epsilon^*$ 是 Sinkhorn 熵正则 OT 的最优传输计划(耦合矩阵),$\mathcal{C}$ 是衡量两个"(隐特征, 动作)联合实体"差异的代价函数。熵正则化 OT 目标为:

$$T_\epsilon^* = \arg\min_{T\in\Pi(\mu_S,\mu_T)} \mathbb{E}_{(x^S,x^T)\sim T}\big[\mathcal{C}(x^S,x^T)\big] - \epsilon H(T)$$

**用大白话说**:OT 找一个"最省搬运成本"的方案,把人类样本这堆"土"搬到机器人样本那堆"坑"里。因为代价函数里含了动作,所以传输计划倾向于把**行为相似** 的人机样本配到一起;梯度回传给 $f_\phi$,就逼编码器把行为相似的人机观测编到隐空间的邻近处。熵正则让这个问题严格凸、可用 Sinkhorn 快速且可微地解。

### 2.2 用 DTW 塑造代价:动作感知的软监督

难点在 $\mathcal{C}$ 的设计:要对人机固有差异(时序快慢、SE(3) 末端动作空间内的轻微运动学差)鲁棒。作者用 **DTW** 在**动作序列** 上找行为相似对。对两条等长 $T$ 的动作序列:

$$\text{DTW}(\mathbf{a}^H, \mathbf{a}^R) = \min_{\pi\in\mathcal{A}(T)} \sum_{(i,j)\in\pi} \|a_i^H - a_j^R\|^2$$

$\mathcal{A}(T)$ 是从 $(1,1)$ 到 $(T,T)$ 的单调对齐路径集合,允许小的局部平移以吸收时序变化。对一个 batch 计算 DTW 代价矩阵 $A \in \mathbb{R}^{B\times B}$,对每个机器人样本 $j$ 取行最小得到**最像它的人类伪配对** $i^*(j) = \arg\min_i A_{ij}$。

但作者强调:直接把 DTW 代价当 OT 代价太"noisy",DTW 更适合当**相对配对的强信号**。于是在标准欧氏隐距离 $D_{ij} = \|f_\phi(o_i^H) - f_\phi(o_j^R)\|^2$ 基础上定义**软监督的联合代价**:

$$\tilde{C}_{ij} = \begin{cases} D_{ij}\cdot\lambda & \text{if } i = i^*(j) \\ D_{ij} & \text{otherwise} \end{cases}$$

其中 $0 < \lambda \ll 1$ 是小标量。

**用大白话说**:如果 DTW 认定人类样本 $i$ 是机器人样本 $j$ 的"行为孪生",就把它俩在隐空间里的搬运代价打个大折扣(乘 $\lambda$),强烈激励 OT 把这对配到一起;其余对照常按隐距离计价。这样动作信息不是硬编进 OT 代价(会太吵),而是以"折扣券"的软方式引导对齐往行为相关的对应关系上走。

### 2.3 整体目标与架构

联合优化编码器 $f_\phi$ 与策略 $\pi_\theta$,OT 损失只作用于编码器,BC 损失端到端作用于全网:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) + \alpha\,\mathcal{L}_{\text{OT-joint}}(\phi)$$

架构是 DETR 风格 transformer(受 [33] HPT、[34] DETR 启发):

- **编码器 $f_\phi$**:模态专用 stem + 共享 trunk。一个**共享 vision stem** 处理人机共同的第一人称 RGB($I_{ego}$),强制视觉对齐;机器人独有的腕部相机走单独 wrist stem;proprio stem 处理本体感受。各 stem 用 cross-attention 把原始观测 token 化为 $L$ 个 query token。trunk 前面 prepend $M$ 个可学习 context token,OT 损失就作用在 trunk 输出的前 $M$ 个 token($z$)上。
- **策略 $\pi_\theta$**:multi-block transformer decoder head,用 $T$(即 $k$)个可学习 action token 通过交替 self/cross-attention 注入 context 生成动作 chunk。

**统一的传感与本体**:人机都用 **Meta Project Aria 智能眼镜** 采集第一人称 RGB;机器人把 Aria 以"类人头部"方式安装以缩小相机设备 gap。机器人平台是受 EgoMimic "Eve" 启发的双臂 ViperX 300(6-DoF),leader-follower 遥操(WidowX 作 leader),腕部装 RealSense D405。人类动作用 Aria MPS 的手部追踪(双手 SE(3)),并借鉴 EgoMimic 构造"伪参考帧"把未来 $k$ 帧手位姿投到当前设备帧,得到相对稳定参考系的动作 chunk。人机动作 chunk 均为 $k=100$;人类 30Hz 取 10 帧、插值到长度 100(约 0.9s),机器人取 100 连续步(约 2s)——**印证了"人快机慢 2-3 倍"需要 DTW 吸收时序错位**。均做 per-embodiment z-score 归一化。

## 三、实验结果

### 3.1 仿真 Push-T(可复现的玩具级验证)

把 "human" 源域设为蓝色圆形 pusher + 紫色背景 + 镜像 T(需新动作把 T 塞入位),"robot" 目标域为三角 pusher + 白背景 + 标准 T;背景色对应"人类数据有新场景",镜像 T 对应"人类演示了新动作"。用 ResNet-UNet Diffusion Policy,OT 作用在 ResNet 特征输出。100 个固定 seed,报告成功率(IoU $\geq 0.9$)。相对各成功率下降幅度(越小越好),EgoBridge 在最难的"镜像 T + 背景色"情形只掉 14%,是所有方法里最低,全面优于 Target-BC / MMD / Standard-OT / Co-train。

### 3.2 真实世界三任务(Table 1)

三个任务:**Drawer**(6×4 抽屉阵,取玩具→放入已开抽屉→关闭;人类覆盖全 4 象限,机器人只覆盖 3 象限,测第 4 象限的行为泛化)、**Scoop Coffee**(单臂勺咖啡豆倒入目标;人类含新研磨器目标+新场景,测观测泛化)、**Laundry**(双臂叠衬衫,精细双手协调)。

Scoop Coffee 成功率(%):

| 方法 | In-Dist | Obj Gen | Scene+Obj Gen |
|---|---|---|---|
| Robot-only BC | 33 | 40 | 7 |
| Co-train | 53 | 46 | 0 |
| EgoMimic | 60 | 53 | 0 |
| MimicPlay | 33 | 27 | 0 |
| ATM | 47 | 33 | 0 |
| **EgoBridge** | **67** | **60** | **27** |

Drawer(Total Pts $\mid$ SR / Place Toy SR / Beh Gen SR,%)与 Laundry(Pts $\mid$ SR,%):

| 方法 | Drawer Pts $\mid$ SR | Place Toy | Beh Gen | Laundry Pts $\mid$ SR |
|---|---|---|---|---|
| Robot-only BC | 38 $\mid$ 9 | 28 | 0 | 38 $\mid$ 28 |
| Co-train | 55 $\mid$ 22 | 42 | 0 | 41 $\mid$ 33 |
| EgoMimic | 49 $\mid$ 14 | 39 | 0 | 38 $\mid$ 33 |
| MimicPlay | 33 $\mid$ 14 | 22 | 0 | 32 $\mid$ 28 |
| ATM | 56 $\mid$ 6 | 17 | 8 | 35 $\mid$ 28 |
| **EgoBridge** | **77 $\mid$ 47** | **72** | **33** | **48 $\mid$ 72** |

关键读数:

- **In-domain(H1)**:全部 in-domain 任务上相对基线绝对成功率提升 7-44%(如 Laundry SR 72% vs Robot-only BC 28%,即 44 个百分点的头条数字)。
- **观测泛化(H2)**:Scoop Coffee 换新研磨器时 EgoBridge 60% 领先所有基线;换"新研磨器 + 新场景"时保持 27%,而**几乎所有基线跌到 0%**。
- **行为泛化(H2)**:Drawer 里机器人数据从未见第 4 象限,EgoBridge 达 33% 成功率把人类动作迁移过去,其余方法基本全灭。
- **隐空间对齐(H3)**:TSNE + Wasserstein-2 距离显示 EgoBridge 的人机隐特征重叠最高(in-distribution 下 W2 距离最低),KNN 配对语义最相近(人机处于同一任务阶段),与成功率/泛化高度相关。

### 3.3 消融(Table 2,Drawer)

| 方法 | Drawer SR | Beh Gen SR |
|---|---|---|
| **EgoBridge** | **47** | **33** |
| MSE(用 MSE 代替 DTW 代价) | 14 | 17 |
| Standard-OT(仅边缘对齐) | 33 | 17 |
| Co-train(无对齐损失) | 22 | 0 |

结论:把 DTW 塑造的代价换成朴素 MSE 造成**最大** 的 in-distribution 掉分(说明"构造语义相似伪配对"最关键);去掉联合改用边缘 Standard-OT 也大幅下滑(边缘对齐无法有效搬运人类知识);完全去掉对齐损失的 Co-train 在行为泛化上归零。

## 四、局限性

- **DTW 代价只适合单任务**:作者在结论明确指出,DTW 动作对齐代价在**多任务** 联合域适应下可能失效——不同任务的动作轨迹相似度不再能可靠指示"行为相似"。这是本方法通用性的主要软肋。
- **依赖对称的传感与本体设计**:人机都用 Aria 眼镜、机器人以类人头位安装,且用 Eve 式双臂逼近人类工作空间——很大程度是把 domain gap "在硬件上先关掉一半",迁移到差异更大的本体上效果未知。
- **需要有标注(带动作)的人类演示**:依赖 Aria MPS 的手部追踪构造动作 chunk,尚不能利用无动作标注的互联网人类视频(作者列为 future work)。
- **单本体、单任务的迁移验证**:未展示扩展到多本体、以及大规模多任务;评估轮次偏少(如 Laundry 18 次、Scoop Coffee 每目标 15 rollout),真机成功率方差可能较大。
- **数字口径**:正文称 MSE 消融使 in-dist 从 47% 掉到 17%,而 Table 2 中 MSE 的 Drawer SR 为 14%、Beh Gen 为 17%,存在轻微不一致(本笔记以表格为准)。

## 五、评价与展望(纯学术视角)

**优点**:(1) 把跨本体模仿明确形式化为 **joint distribution domain adaptation**,并给出"OT 对齐联合分布 + DTW 软监督塑造代价"的干净可微方案,概念上比 EgoMimic 的显式 masking/归一化更有原则性;OT 只作用编码器、BC 端到端的解耦也合理。(2) 用 DTW"折扣券"而非把动作硬塞进 OT 代价,是个务实且有效的工程判断,消融验证了其重要性。(3) **behavior generalization** 设定(机器人执行只在人类数据出现过的动作/位置)是本文最有说服力的贡献——多数人类数据 co-train 工作只做到观测泛化,而这里在 Drawer/Scoop Coffee 上把 baseline 从 0% 拉到 27-33%。

**与公开工作的关系**:相对 **EgoMimic**(同组 Kareer 等,靠 visual masking + 共享末端位姿头对齐)是"隐式表征对齐 vs 显式输入对齐"的路线升级,且在同硬件上直接超越;相对 **MimicPlay**(KL 对齐 high-level 隐 marginal)与 **ATM**(2D point track 作中间表征)体现了"联合分布对齐 > 边缘对齐/中间表征"的一致优势;方法根源接续 Courty 等的 Joint Distribution OT for DA(JDOT/DeepJDOT [26,28])与 Cuturi 的 Sinkhorn [29],可视为把经典分类域适应的联合 OT 思想**引入到带时序耦合的机器人策略学习**,并用 DTW 补上"动作序列如何配对"这一环。

**开放问题与可能改进**:(1) 多任务/大规模下 DTW 代价失效——作者自己提出的方向是改用 VLM 的自然语言 embedding 距离或基础模型视觉特征作更通用的对齐代价,值得跟进。(2) 联合 OT 的 batch 内配对本质是"局部"的,能否结合记忆库/更大有效 batch 或分层配对以稳定训练。(3) 扩展到**多本体** 联合适应,以及利用**无动作标注** 的互联网人类视频(需与 inverse dynamics / latent action 方案结合)。(4) OT 权重 $\alpha$ 与折扣 $\lambda$ 的敏感性、以及在更大 domain gap(不同相机、不同手/夹爪构型)下的鲁棒性尚缺系统分析。

## 参考

1. Kareer et al. *EgoMimic: Scaling Imitation Learning via Egocentric Video*, 2024 —— 同组前作,本文主要对标基线与硬件蓝本。
2. Courty et al. *Joint Distribution Optimal Transportation for Domain Adaptation*, NeurIPS 2017;Damodaran et al. *DeepJDOT*, ECCV 2018 —— 联合分布 OT 域适应的理论源头。
3. Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances*, NeurIPS 2013 —— 可微熵正则 OT 的算法基础。
4. Wang et al. *MimicPlay: Long-Horizon Imitation Learning by Watching Human Play*, CoRL 2023;Wen et al. *Any-point Trajectory Modeling (ATM)*, 2023 —— 从人类数据学习的两条代表性对照路线。
5. Sakoe & Chiba. *Dynamic Programming Algorithm Optimization for Spoken Word Recognition*, 1978 —— DTW 原始工作,用于塑造 OT 代价。
