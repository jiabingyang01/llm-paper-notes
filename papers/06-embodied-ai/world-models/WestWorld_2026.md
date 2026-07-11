# WestWorld：面向多样机器人系统的知识编码可扩展轨迹世界模型

> **论文**：*WestWorld: A Knowledge-Encoded Scalable Trajectory World Model for Diverse Robotic Systems*
>
> **作者**：Yuchen Wang, Jiangtao Kong（共同一作）, Sizhe Wei, Xiaochang Li, Haohong Lin, Hongjue Zhao, Tianyi Zhou, Lu Gan, Huajie Shao（通讯作者）et al.
>
> **机构**：William & Mary；Georgia Institute of Technology；Carnegie Mellon University；University of Illinois at Urbana-Champaign；Mohamed bin Zayed University of Artificial Intelligence
>
> **发布时间**：2026 年 03 月（arXiv 2603.14392）
>
> **发表状态**：已录用 — ICML 2026（PMLR 第 306 卷）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.14392) | [PDF](https://arxiv.org/pdf/2603.14392)
>
> **分类标签**：`轨迹世界模型` `Mixture-of-Experts` `运动学结构先验` `跨具身泛化` `模型预测控制`

---

## 一句话总结

WestWorld 用"系统感知 MoE（Sys-MoE）+ 运动学结构编码（KNEE）"两件套，在 89 个仿真/真实环境（UniTraj 80 个仿真环境 + Open X-Embodiment 9 个真实机械臂数据集）上联合预训练一个纯低层状态-动作轨迹世界模型,零样本预测误差相较最强基线 TrajWorld 平均降低约 20%-30%,且规模扩展到 89 个环境时误差几乎不增长（TrajWorld 则从约 3×10⁻² 恶化到约 11.5×10⁻²）,并在 Unitree Go1 上完成蒸馏后的真机部署。

## 一、问题与动机

轨迹世界模型（trajectory world model）直接对低层本体感知状态-动作序列建模,是基于模型的规划与控制的核心组件。作者指出跨多种机器人本体联合训练这类模型面临两个尚未解决的问题：

- **可扩展性（scalability）**：现有方法（如 TDM、TrajWorld）用单一稠密 Transformer 联合拟合多种机器人动力学,不同本体之间存在梯度冲突与负迁移,机器人种类越多、性能退化越明显。
- **泛化性（generalization）**：这些方法把状态-动作只当作离散 token 序列建模,完全忽略机器人的运动学结构（关节拓扑）信息,缺乏支撑零样本迁移到新本体的物理归纳偏置。

论文的核心假设：结构上相似（连接模式相近）的机器人往往共享相似的高层动力学行为（如 SLIP 弹簧倒立摆式的腿式运动）,因此把运动学树结构显式编码进轨迹表示,能作为有效的零样本泛化归纳偏置。

## 二、核心方法

**问题设定**。世界模型学习转移分布 $p_\theta(\boldsymbol{s}_{t+1} \mid \boldsymbol{s}_{1:t}, \boldsymbol{a}_{1:t})$,给定 $h$ 步历史状态-动作,预测未来 $k$ 步状态。每个状态/动作维度被当作一个标量"通道",做逐通道 min-max 归一化后离散化为 $K$ 桶的类别向量,再经可学习投影得到 token 嵌入,并叠加时间步、通道序号、模态（状态/动作）三类嵌入。

WestWorld 由两大模块堆叠而成：

**(1) 知识编码嵌入模块（KNEE）**。把每个铰接体（articulated object）的运动学树通过 left-child-right-sibling（LCRS）变换转成二叉树,为每个刚体节点 $j$ 计算先序、中序、后序遍历秩 $(\pi^{i,j}_{\text{pre}}, \pi^{i,j}_{\text{in}}, \pi^{i,j}_{\text{post}})$（$i$ 为多物体场景下的物体编号）,构造结构嵌入

$$\boldsymbol{p}^{(i,j)} = \text{Concat}\big(e_{\text{obj}}(\pi^{i}_{\text{obj}}),\, e_{\text{pre}}(\pi^{i,j}_{\text{pre}}),\, e_{\text{in}}(\pi^{i,j}_{\text{in}}),\, e_{\text{post}}(\pi^{i,j}_{\text{post}})\big)$$

用大白话说：给运动学树里的每根"骨头"发一张身份证（先序/中序/后序遍历号码 + 物体号）,把这张身份证的向量表示直接加到该关节对应的状态/动作/query token 嵌入上,这样模型天然知道"这是哪条腿的哪个关节",不同机器人只要连接方式相近就能共享这套编号语义,从而具备零样本迁移能力。

**(2) 系统感知 MoE 块（Sys-MoE）**。先做通道间自注意力聚合状态相关性,再用交叉注意力把动作信息注入状态特征（式 4-5）;随后引入一个**可学习的系统嵌入** $e$,与聚合后的状态特征拼接,经 Mamba 风格选择性状态空间模型（SSM）传播,得到系统级动力学表示 $U_{L+1}$;用它经 softmax 路由器产生对 $P$ 个专家（每个专家是一个 MLP）的混合权重

$$\boldsymbol{w} = \text{Softmax}(\text{Router}(\boldsymbol{U}_{L+1})) \in \mathbb{R}^{P}, \qquad \boldsymbol{Y}^{(m)}_{1:L} = \sum_{p=1}^{P} w_p\, E_p(\boldsymbol{U}^{(m)}_{1:L})$$

用大白话说：不再让所有机器人共用同一套稠密参数（会打架、互相拖后腿）,而是先"猜"出当前是哪一类系统动力学,再用一组可复用的"基础动力学模块"按系统相关的权重组合出该系统专属的动力学预测——复杂动力学被近似成若干"基元动力学"的系统相关线性组合。

**训练目标**。堆叠多个 Sys-MoE 块后,线性解码头把隐状态映射为 $K$ 类 logits,用下一 token 交叉熵损失训练状态通道的类别预测（式 11）。推理时以 query 嵌入一次性并行产生 $k$ 步预测（seq-to-seq,非严格自回归逐步展开）。

## 三、关键结果

**零样本泛化**（Walker2d、Hopper、来自真实移动 Franka 数据集,50 步历史→100 步预测,误差 ×10⁻²）：

| 方法 | Walker2d MAE/MSE | Hopper MAE/MSE | Franka MAE/MSE |
|---|---|---|---|
| MLPEnsemble | 26.01 / 12.03 | 19.99 / 7.22 | 12.16 / 4.27 |
| TDM | 20.12 / 6.43 | 17.63 / 5.08 | 23.69 / 8.44 |
| TrajWorld | 22.26 / 8.62 | 17.39 / 5.44 | 13.10 / 5.13 |
| **WestWorld** | **16.35 / 5.06** | **13.73 / 3.37** | **7.74 / 2.54** |

**少样本适应**（Cassie 双足跳跃、Unitree A1 四足行走、UR5 桌面操作,仅 10 条轨迹微调,×10⁻²,3 个随机种子）：WestWorld 在三个系统上全面最优,例如 Cassie MAE/MSE 为 5.32/0.81,显著优于最强基线 TrajWorld 的 7.83/1.70。

**规模扩展性**：预训练环境数 $N \in \{1,2,5,10,20,30,50,60,89\}$ 时,TrajWorld 的长程 MAE 从约 3×10⁻² 单调恶化到约 11.5×10⁻²,而 WestWorld 全程保持在 2×10⁻²-3.5×10⁻² 区间,基本不随 $N$ 增长而退化。Sys-MoE 路由权重可视化显示,Walker/Fish/Cheetah 三种系统在 6 层专家上呈现稀疏且系统相关的专家激活模式,支持"基元动力学组合"的假设。

**下游模型预测控制（MPPI,累计回报,越高越好）**：

| 方法 | Walker2d（预训练前→后） | Hopper（前→后） | Go1（前→后） |
|---|---|---|---|
| MLPEnsemble | 119.2 → 190.5 | 147.4 → 200.6 | -1.07 → -0.31 |
| TDM | 122.7 → 207.6 | 242.3 → 154.6 | -0.72 → 0.03 |
| TrajWorld | 395.2 → 1933.5 | 366.3 → 534.3 | 0.05 → 0.49 |
| **WestWorld** | **707.6 → 2134.6** | **554.9 → 2253.5** | **0.43 → 2.20** |

预训练在几乎所有方法/系统上都提升控制表现,WestWorld 在预训练前后两种设定下均全面最优。真机部署方面,将 WestWorld 蒸馏为两层学生模型并接入 MPPI,在 Unitree Go1 上实现稳定直线行走,而同样蒸馏流程下的 TrajWorld 无法可靠站立/前行。

**消融**（零样本设定,×10⁻²）：去掉 Sys-MoE（替换为稠密 SSM,参数量对齐）在 Walker2D/Hopper/Franka 上误差从 16.35/13.73/7.74 升至 18.71/15.98/9.39;去掉结构嵌入（KNEE）升至 21.16/16.23/7.90。两个模块中,结构嵌入对形态更复杂的 Walker2D/Hopper 收益更大,Sys-MoE 对三个系统均有明显贡献,二者缺一不可。

## 四、评价与展望

**优点**：把机器人运动学树的先序/中序/后序遍历编号直接作为归纳偏置注入轨迹 token,是一个简洁且可解释的跨本体共享机制,不依赖图神经网络式的显式邻接矩阵传播;系统感知 MoE 用一个可学习的系统嵌入驱动路由,而非像语言模型 MoE 那样直接对 token 路由,更契合"同一机器人族群应激活同一组专家"的先验,消融和路由权重可视化都提供了直接证据。规模曲线（$N$ 从 1 到 89 误差几乎不变,TrajWorld 单调恶化）是本文最有说服力的卖点,直接回应了多具身联合训练中梯度干扰这一常见痛点。真机 Go1 部署 + 与 TrajWorld 蒸馏后对比,补上了"仿真结果能否落地"的证据链。

**局限**：作者自陈模型目前只处理低层本体感知轨迹（state-action 序列）,不包含视觉观测,后续计划扩展为融合视觉与轨迹信号的多模态世界模型。零样本评测所用的 Walker2d/Hopper/Franka 严格来说是"与预训练数据结构相似的未见环境",并非完全异构本体（如足式→机械臂之间的跨形态零样本尚未验证,少样本实验里 UR5/A1/Cassie 也只用 10 条轨迹微调而非零样本）。下游控制实验中控制器与动力学模型是分离评估的（MPPI 采样规划,不联合优化策略/控制器）,论文明确将"策略联合优化"列为单独问题,未纳入本工作评估范围。此外,LCRS 遍历编号本质上仍是一种手工设计的结构先验,对树状运动学（如平面/腿式机器人）适配良好,但对存在闭环运动链（parallel linkage）或非树状约束的机构是否仍然有效未做讨论。

**与其他工作的关系**：相较 TDM（基于 Gato 架构、把状态动作展平为一维序列做自回归）和 TrajWorld（temporal-variate attention 的异构轨迹世界模型）,WestWorld 的核心差异在于显式引入形态结构先验与系统感知专家路由,而非单纯扩大 Transformer 容量;这与 TD-MPC2 等追求"单一稠密模型规模化"的路线形成对比,呼应了近期对 MoE 缓解多任务梯度冲突的通用发现（如语言模型 MoE 文献）在机器人动力学建模场景的迁移应用。一个开放问题是：结构先验编码与 Sys-MoE 路由的组合能否进一步扩展到跨本体的零样本"新骨架"泛化（而不仅是同族机器人内插值）,以及能否与视觉世界模型（如 Cosmos、Wow 等生成式视频世界模型）结合,把低层轨迹动力学与高层视觉动力学统一到同一预测框架下,是作者明确指出的下一步方向。

## 参考

- Yin et al. *Trajectory world models for heterogeneous robots*（TrajWorld,ICML 2026）— 本文最强基线,temporal-variate attention 异构轨迹世界模型。
- Schubert et al. *A generalist dynamics model for control*（arXiv 2305.10912）— TDM 所基于的 Gato 式生成式动力学模型。
- Chua et al. *Deep reinforcement learning in a handful of trials using probabilistic dynamics models*（NeurIPS 2018）— MLP Ensemble 基线来源（PETS）。
- Hansen et al. *TD-MPC2: Scalable, robust world models for continuous control*（ICLR 2024）— 单一稠密模型规模化的世界模型代表工作,对照 Sys-MoE 路线。
- Gu & Dao. *Mamba: Linear-time sequence modeling with selective state spaces*（2024）— WestWorld 中 SSM 层的骨干架构来源。
