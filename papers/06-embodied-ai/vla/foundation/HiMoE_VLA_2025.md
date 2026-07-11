# HiMoE-VLA：面向通用 VLA 策略的分层混合专家动作模块

> **论文**：*HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies*
>
> **作者**：Zhiying Du、Bei Liu、Yaobo Liang、Yichao Shen、Haidong Cao、Xiangyu Zheng、Zhiyuan Feng、Zuxuan Wu、Jiaolong Yang、Yu-Gang Jiang
>
> **机构**：Fudan University；Microsoft Research Asia；Xi'an Jiaotong University；Tsinghua University
>
> **发布时间**：2025 年 12 月（arXiv 2512.05693，v2 修订于 2026-07）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.05693) | [PDF](https://arxiv.org/pdf/2512.05693)
>
> **分类标签**：`VLA` `Mixture-of-Experts` `异构数据联合训练` `负迁移` `flow matching` `跨具身泛化`

---

## 一句话总结

HiMoE-VLA 用一个按深度分层的 MoE 动作模块——边界层用 Action-Space MoE（AS-MoE）专门隔离不同动作空间（如关节角 vs. 末端位姿）的计算、相邻内层用 Heterogeneity-Balancing MoE（HB-MoE）为具身/视角/场景等残余异质性提供均衡稀疏容量、中间保留稠密 Transformer 融合共享表示，并配合两个路由正则项（AS-Reg 对比正则化 + HB-Reg 负载均衡），把强基线在异构多源机器人数据联合训练下出现的负迁移逆转为正迁移；CALVIN（D→D）达到 3.98（连续完成子任务数之和），LIBERO 平均成功率 98.0%，真机 xArm7/ALOHA 平均成功率分别达 75.0%/63.7%。

## 一、问题与动机

通用 VLA 策略通常在 Open X-Embodiment（OXE）等大规模异构机器人数据混合上预训练，但这些数据在具身、动作空间、状态表示、控制频率、相机视角、采集协议上差异巨大。现有方法（RT-2、OpenVLA、π0、UniVLA 等）大多仍用一套共享的稠密动作模块吸收全部这些差异；RDT-1B 统一了双臂操作的状态/动作表示，HPT 用数据集专属的 stem/head，SpatialVLA、OpenVLA-OFT、GR00T 系列引入更强的空间/动作/具身感知设计，但都没有在动作模块内部显式区分"动作空间不兼容"与"其余异质性"这两类性质不同的差异来源。作者通过受控的联合训练实验（对应正文 Table 5、7）证明：当动作空间或观测配置在数据源之间不同时，用共享稠密动作模块简单混合额外异构数据反而会造成负迁移（负迁移不是理论假设,而是本文对基线实测出的现象）。核心问题因此变为：如何设计动作模块，使不兼容的因素被分离、可迁移的结构仍被共享？

## 二、核心方法

**整体结构**：策略 $\pi_\theta(l, q_t, o_t) \mapsto A_t$，输入语言指令 $l$、本体感知 $q_t$、多视角 RGB 观测 $o_t=[I_t^1,\dots,I_t^n]$，输出动作 chunk $A_t=[a_t,\dots,a_{t+H-1}]$。VLM 主干用 PaliGemma 初始化（follow π0 的设计），编码指令与图像；动作专家通过层级 KV 交叉注意力接入 VLM 各层中间特征（而非仅最后一层），接收本体感知向量、加噪动作 chunk 与 flow 时间步，预测去噪速度场。不同数据源的状态/动作先映射进统一向量接口（含 padding 与合法性掩码），同时保留一个类别化的动作空间/具身身份标签 $c$ 供路由正则化使用。

**分层 MoE（HiMoE）动作模块**（对应 Fig. 2）：动作专家的前馈子层按深度替换为三段结构——最外侧（输入/输出边界）用 **AS-MoE** 专注于动作空间层面的差异（如关节角 vs. 末端执行器控制）；相邻内层用 **HB-MoE** 为具身、视角、场景等残余变化提供均衡的稀疏容量；中间保持稠密 Transformer block，整合已分离信息为共享表示。每个 MoE block 采用 top-$K$ 路由（$N$ 个专家）并额外配一个对所有 token 并行生效的共享专家（沿用 DeepSeekMoE 设计），其输出与路由专家输出相加，让路由专家专注于源特定变化、共享专家承担与异质性无关的计算。

**训练目标**：

$$
\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_{\text{AS}} \mathcal{L}_{\text{AS}} + \lambda_{\text{HB}} \mathcal{L}_{\text{HB}}
$$

flow-matching 损失负责动作生成：

$$
A_t^\tau = \tau A_t + (1-\tau)\epsilon,\quad \epsilon\sim\mathcal N(0,I),\quad \mathcal{L}_{\text{flow}} = \mathbb{E}\big[\|v_\theta(A_t^\tau,\tau,o_t,l,q_t) - (A_t-\epsilon)\|_2^2\big]
$$

用大白话说：把真实动作序列和高斯噪声按比例 $\tau$ 混合成"半噪声"输入，模型学习预测把它推回真实动作的方向场，推理时从纯噪声沿这个方向场积分一步步"擦"出动作。

AS-Reg（式 4-5）对 AS-MoE 的路由分布施加监督对比目标：设 $c_u$ 为 token $u$ 的动作空间/具身身份，$\hat r_u$ 为其 $\ell_2$ 归一化路由概率向量，正样本集 $P(u)$ 为同一动作空间的其余 token、锚点排除集 $A(u)$ 排除自身：

$$
\mathcal{L}_{\text{AS}} = \frac{1}{U_+}\sum_{u=1}^{U}\mathbb{1}[|P(u)|>0]\frac{-1}{|P(u)|}\sum_{p\in P(u)}\log\frac{\exp(\hat r_u^\top \hat r_p/\beta)}{\sum_{v\in A(u)}\exp(\hat r_u^\top \hat r_v/\beta)}
$$

用大白话说：让同一种动作空间的 token 路由到相似的专家组合、不同动作空间的 token 路由模式互相拉开，强迫 AS-MoE 在边界层就按动作空间"分家"。

HB-Reg（式 6）沿用 DeepSeekMoE 的负载均衡损失：$f_i=\frac{N}{KU}\sum_u r_{i,u}$（专家 $i$ 的实际负载占比,停梯度）、$P_i=\frac{1}{U}\sum_u s_{i,u}$（平均路由 softmax 概率），$\mathcal{L}_{\text{HB}}=\sum_i f_i P_i$，均衡配置下 $\mathcal{L}_{\text{HB}}=1$。用大白话说：防止 HB-MoE 的专家池"偷懒"塌缩到只用少数几个专家,把概率质量从过载专家推向欠载专家,维持这一层应有的均衡容量。

**规模与实现**：4B 参数，16 张 A100 + DeepSpeed 训练；预训练混合 OXE + 公开 ALOHA 类数据（含 Mobile ALOHA、RDT-1B 数据集），共 24.1M 帧；单third-person view + 双腕部视角；$N=32$ 专家、top-$K=4$。

## 三、关键结果

**CALVIN（D→D，连续完成子任务数之和，满分 5）**：

| 方法 | Sum |
|---|---|
| Octo | 1.97 |
| OpenVLA | 1.41 |
| RDT-1B | 2.04 |
| MDT | 3.72 |
| π0 | 3.76 |
| **HiMoE-VLA** | **3.98** |
| FLOWER（原稠密动作专家） | 4.35 |
| FLOWER + HiMoE（替换动作模块） | **4.49** |

**LIBERO（成功率 %，四套件平均）**：OpenVLA-OFT 97.1、π0.5 96.8、**HiMoE-VLA 98.0**（Spatial 98.2、Object 99.4、Goal 98.6、Long 95.8，仅 Spatial 略逊于 π0.5 的 98.8）。

**真机评测（平均成功率 %）**：xArm7 单臂（Fruit-to-Plate / Cup-in-Cup / Block-on-Block）：Octo-Base 19.3、OpenVLA 21.2、CogACT 61.5、π0 62.5、**HiMoE-VLA 75.0**；ALOHA 双臂（Cup-Handover / Scoop / Fold-Shorts）：ACT 20.9、RDT-1B 47.5、π0 54.2、**HiMoE-VLA 63.7**。泛化评测（未见干扰物/新物体，成功率 %）：单臂 π0 55.9 → HiMoE-VLA 67.6；双臂 π0 33.4 → HiMoE-VLA 50.0。

**异构联合训练的负迁移/正迁移逆转（CALVIN Sum）**：

| 方法 | 仅 D | ABC+D 混合后 | 变化 |
|---|---|---|---|
| π0 | 3.806 | 3.547 | **-0.259**（负迁移） |
| 去掉 MoE 的对照 | 3.819 | 3.777 | -0.042 |
| 完整 HiMoE-VLA | 3.826 | 4.012 | **+0.186**（正迁移） |

同样地，在共享末端执行器动作空间下把 LIBERO 并入 CALVIN 训练（仅观测/场景异质，非动作空间异质）：π0 出现 -0.272 的负迁移，标准 MoE 仅 +0.054，完整 HiMoE 达 **+0.147**，说明分层设计的收益不只是"稀疏容量"本身，而是层级化分离带来的额外增益。

**组件消融（CALVIN ABC+D Sum）**：完整模型 4.012；去掉 AS-MoE 3.873；去掉 HB-MoE 3.836；去掉全部正则 3.835；单层非分层 MoE+正则 3.813；去掉 MoE 3.777——移除任一层级都明显掉分，验证了"边界specialize、内层balance、中间共享"三段分工缺一不可。此外，与人工具身条件化方案（Separate Heads、GR00T 式具身指示符）对比，HiMoE 路由学习效果最优（4.012 vs. 3.856 / 3.827），且专家数 $N=32$、top-$K=4$ 最优，$K=8$ 出现不稳定；引入分层 MoE 的训练成本开销约 7%，推理延迟约 0.195s/动作。

## 四、评价与展望

**优点**：本文的核心贡献不是又一次"加更强的 MoE 做稀疏扩容"，而是明确把"动作空间不兼容"和"其余异质性（视角/场景/具身）"当作性质不同的两类问题，用深度上的层级位置（边界 vs. 内层 vs. 中间）分别处理，这是对近期 VLA 异构数据联合训练普遍采用"共享稠密模块硬吃全部差异"或"人工具身/数据集专属 head"两种极端做法的一个中间路线。实验设计上的亮点是给出了受控的"负迁移 → 正迁移"对照实验（Table 5、7）而不只是端到端跑分，并做了较完整的组件、正则项、专家数、共享专家、mask 流程、参数量对齐的消融，比较扎实地支撑了"分层结构本身有效"而非单纯"参数量更大"这一论点。与 RDT-1B（统一双臂动作表示）、HPT（数据集专属 stem/head）相比，HiMoE-VLA 保持共享的 VLM 主干与统一动作接口，把异质性处理下沉到动作模块内部的路由层级，是对这两条路线的互补而非替代。

**局限与开放问题**：作者自陈的局限包括——方法假设每条样本都能映射进统一状态/动作接口，标注缺失或移动操作等更异构场景下可能失效；真机验证仅覆盖 xArm7 单臂与 ALOHA 双臂两个平台，尚未在更大规模、更多本体的混合（如移动操作、人形机器人）上验证 Fig. 1 所展示的"跨具身"能力；长时程分布漂移下的安全性/校准未量化；路由与跨注意力的额外开销尚待优化。从评述角度看，可以进一步追问：(1) 动作空间身份标签 $c$ 依赖数据集元数据人工标注/规则划分，当数据源数量从论文中的 2-3 种（关节角/EEF/双臂）扩展到几十种真正异构的具身时，AS-Reg 的对比目标和 AS-MoE 的容量是否仍够用，论文未给出扩展性证据；(2) 4B 参数、24.1M 帧、16 卡 A100 的训练规模相对 π0.5、GR00T N1 等工业级 VLA 仍偏小，分层 MoE 的收益能否在更大规模预训练下保持或进一步放大是一个开放问题；(3) HiMoE 与 FLOWER 结合后的正迁移提升（4.35→4.49）说明该动作模块具备一定的架构可插拔性，但论文只验证了一种"宿主"策略，能否作为通用组件迁移到其他 VLA 框架仍需更多实证。

## 参考

1. K. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. S. Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*. arXiv:2410.07864, 2024.
3. D. Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. arXiv:2401.06066, 2024.
4. O. Mees, L. Hermann, E. Rosete-Beas, W. Burgard. *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks*. RA-L, 2022.
5. B. Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. arXiv:2306.03310, 2023.
