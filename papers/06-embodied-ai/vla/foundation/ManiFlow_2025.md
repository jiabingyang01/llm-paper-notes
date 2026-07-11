# ManiFlow：基于一致性流训练的通用机器人操作策略

> **论文**：*ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training*
>
> **作者**：Ge Yan, Jiyue Zhu, Yuquan Deng, Shiqi Yang, Ri-Zhao Qiu, Xuxin Cheng, Marius Memmel, Ranjay Krishna, Ankit Goyal, Xiaolong Wang, Dieter Fox et al.
>
> **机构**：University of Washington、UC San Diego、Nvidia、Allen Institute for Artificial Intelligence (AI2)
>
> **发布时间**：2025 年 09 月（arXiv 2509.01819）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.01819) | [PDF](https://arxiv.org/pdf/2509.01819)
>
> **分类标签**：`flow matching` `一致性训练` `灵巧操作` `多模态Transformer` `真机双臂/人形评测`

---

## 一句话总结

ManiFlow 在标准 flow matching 损失之外联合训练一个连续时间的自一致性(consistency)目标(不依赖任何预训练教师模型),并配合新的 DiT-X 多模态条件化架构(把 AdaLN-Zero 的缩放/偏移也施加到 cross-attention 的输入输出上),使策略仅用 1-2 步 ODE 积分即可生成高质量高维灵巧动作;在 12 个仿真灵巧任务、48 个 MetaWorld 语言条件多任务和 3 种真机平台(单臂/双臂/人形)共 8 个真实任务上一致超过 Diffusion Policy、Flow Matching Policy 与 3D Diffusion Policy(DP3),真机平均成功率相对 DP3 提升 98.3%,在 RoboTwin 2.0 的 4 个双臂任务上相对微调后的 π0 提升 58%。

## 一、问题与动机

- Diffusion Policy 在灵巧操作中表现出色,但迭代去噪导致推理慢;flow matching 提升了训练/推理效率,但现有 flow matching 策略在复杂灵巧操作、真实场景鲁棒性与泛化性上仍受限,面临三个具体挑战:(1) 多指交互的高维复杂性,(2) 动作序列时间一致性的维持,(3) 现有架构对视觉、语言、本体感觉等多模态输入建模不充分。
- 已有的少步加速方案(如 Consistency Policy)依赖预训练扩散教师模型做蒸馏,训练流程繁琐、需要多阶段训练;ManiFlow 希望单阶段端到端联合优化,不依赖额外教师模型。
- 已有的多模态条件化架构(如 MDT 的朴素 cross-attention)只在自注意力/前馈层做自适应条件化,cross-attention 本身缺乏细粒度调制能力,难以精确处理低维信号(timestep、本体感觉)与高维视觉/语言信号的交互。

## 二、核心方法

**Flow matching 预备知识**：给定数据点 $x_1 \sim D$、噪声点 $x_0 \sim \mathcal N(0,I)$、时间 $t \sim \mathcal U[0,1]$,定义线性插值 $x_t=(1-t)x_0+t x_1$,速度目标为 $v_t=x_1-x_0$。Flow matching 损失为

$$\mathcal{L}_{FM}(\theta)=\mathbb{E}_{x_0,x_1\sim D}\left[\|v_\theta(x_t,t)-(x_1-x_0)\|^2\right]$$

用大白话说:让模型在噪声到动作的插值路径上任意一点预测出恒定的"指向真实动作"的速度场,训练好后从纯噪声出发沿预测速度做一步或几步 ODE 积分即可生成动作。

**连续时间一致性训练(核心创新一)**：给流模型额外加一个步长参数 $\Delta t$,记为 $v_\theta(x_t,t,\Delta t)$。训练时采样 $t$ 和 $\Delta t \sim \mathcal U[0,1]$,得到下一时间点 $t_1=t+\Delta t$(裁剪保证落在 $[0,1]$ 内)。用模型的 EMA 版本 $\theta^-$ 预测 $x_{t_1}$ 处朝向更远时间点 $t_2=t_1+\Delta t'$ 的速度 $v_{t_1}=v_{\theta^-}(x_{t_1},t_1,\Delta t')$,据此外推终点估计 $\tilde x_1 = x_{t_1} + (1-t_1)\cdot v_{t_1}$,再反推从 $x_t$ 到 $\tilde x_1$ 的平均速度目标 $\tilde v_{target} = (\tilde x_1 - x_t)/(1-t)$。一致性损失为

$$\mathcal{L}_{CT}(\theta)=\mathbb{E}_{t,\Delta t\sim \mathcal U[0,1]}\left[\|v_\theta(x_t,t,\Delta t)-\tilde v_{target}\|^2\right]$$

用大白话说:强迫模型在同一条噪声到动作的轨迹上,无论从哪个中间点出发、迈多大步子,反推出来的终点估计都要相互吻合(自洽),从而把整条轨迹"拉直",使得少数几步(甚至一步)积分也能命中真实动作——且不需要额外的预训练教师模型,这是与 Consistency Policy 等蒸馏类加速方法的关键区别。

**联合训练**：总损失把 $\Delta t=0$ 的 flow matching 项(此时估计的是局部瞬时速度)与连续 $\Delta t$ 的一致性项相加,训练 batch 中 75% 样本用于 flow matching(时间 $t$ 用 Beta$(\alpha=1,\beta=1.5,s=0.999)$ 采样,偏重接近 $t=0$ 的高噪声区)、25% 样本用于一致性训练(时间 $t$ 从离散均匀网格采样、$\Delta t$ 连续均匀采样)。EMA 模型 $\theta^-=\mu\theta^-+(1-\mu)\theta$ 为一致性目标提供缓慢演化、稳定的监督信号,避免用当前快速变化的模型自举导致训练震荡。

**时间步采样策略消融**：论文系统比较了 5 种 $t$ 采样方式——Uniform、Logit-Normal、Mode、CosMap、Beta,发现偏向低 $t$(高噪声区)的 Beta 分布在机器人动作预测上稳定优于其余方案,论文将其归因于"含噪观测能提供更强的动作约束、因此高噪声区的学习价值更大"。

**DiT-X 架构(核心创新二)**：相比 DiT(仅自注意力+AdaLN-Zero 条件化)和 MDT(加入朴素 cross-attention,但条件化只作用于自注意力/前馈层),DiT-X 把 AdaLN-Zero 生成的缩放/偏移参数 $(\alpha,\gamma,\beta)$(以 timestep 和 $\Delta t$ 为条件)也施加到 cross-attention 层的输入和输出上,而不仅是自注意力和前馈层。这使得低维信号能动态调制动作 token 与视觉/语言 token 之间的交互强度,实现更细粒度的 token 级对齐。10 个 MetaWorld 语言条件任务的消融显示,DiT-X 相比"无 cross-attention AdaLN-Zero 条件化"的版本收敛更快、最终成功率更高;论文也坦言这一设计带来"适度的计算开销"。

**感知与动作编解码**：3D 视觉编码器沿用 3D Diffusion Policy(DP3)的轻量 PointNet,但去掉 max pooling 以保留逐点几何特征;实验发现 SE3 空间数据增广反而损害性能,故未采用,颜色 jitter(0.2 概率)则对真机泛化有益。点云密度按场景自适应:规整裁剪场景 128-256 点即足够,杂乱或第一视角场景需 2048-4096 点。语言指令经冻结 T5 编码为 512 维再投影到 token 维度;本体感觉经 2 层 MLP 编码,并以一定概率随机 mask 以防止策略"抄近路"过度依赖本体感觉而非视觉;动作经 2 层 MLP 解码,动作视界因任务而异(短程仿真任务 4 步,RoboTwin 双臂 16 步,真机 64 步并配合时间集成)。

## 三、实验结果

**仿真主结果**(Table 1,12 个灵巧任务,3 个 benchmark:RoboTwin 1.0 五任务、Adroit 三任务、DexArt 四任务):

| 输入 | 方法 | RoboTwin(5) | Adroit(3) | DexArt(4) | Average |
|---|---|---|---|---|---|
| Img(2D) | Diffusion Policy | 28.8±2.1 | 38.1±2.9 | 53.6±2.1 | 39.4±2.3 |
| Img(2D) | Flow Matching Policy | 27.1±2.7 | 39.0±2.2 | 53.3±2.4 | 38.8±2.5 |
| Img(2D) | **ManiFlow** | **46.1±2.7** | **74.3±1.9** | **56.3±2.3** | **56.5±2.4** |
| PC(3D) | 3D Diffusion Policy(DP3) | 42.7±3.3 | 77.8±2.4 | 60.6±0.7 | 57.4±2.2 |
| PC(3D) | 3D Flow Matching Policy* | 48.1±6.3 | 77.1±3.3 | 61.7±1.1 | 59.9±2.8 |
| PC(3D) | **ManiFlow** | **61.9±2.5** | **78.6±2.3** | **63.2±2.7** | **66.5±2.5** |

相对 Diffusion Policy / Flow Matching Policy,2D 输入平均提升 43.4% / 45.6%,3D 输入平均提升 15.9% / 11.0%。

**MetaWorld 48 任务语言条件多任务**(仅 3D 输入,10 demo/任务):ManiFlow 平均成功率 78.1±2.0% vs 3D Diffusion Policy 59.4±3.5%、3D Flow Matching Policy* 57.9±0.5%,相对提升 31.4% / 34.9%,且在 Easy/Medium/Hard/Very Hard 各难度档均领先,Very Hard 档相对提升高达 125% / 73.6%。

**RoboTwin 2.0 域随机化鲁棒性与效率对比**(4 个双臂任务,各仅 50 条 demo,点云输入,对比大规模预训练后微调的 π0):

| 任务 | π0 | ManiFlow |
|---|---|---|
| Lift Pot | 24.3 | 64.7 |
| Pick Dual Bottles | 18.0 | 55.5 |
| Put Object Cabinet | 41.0 | 55.0 |
| Open Laptop | 70.0 | 66.7 |

平均相对提升 58%(注:Open Laptop 一项 ManiFlow 反而略逊于 π0)。数据规模消融(lift pot 任务,10→500 demo):ManiFlow 50 条示范即达 64.7%,100 条约 90%,200 条 97.7%;π0 需 500 条才达到 94.0%,仍低于 ManiFlow 200 条时的水平;ManiFlow 在 500 条时进一步达到 99.7%。

**真机实验**(Table 2,8 个任务 × 3 平台:Unitree H1 人形、双臂 xArm+PSYONIC Ability 灵巧手、单臂 Franka,均为点云输入,对比 DP3):

| 平台 | 任务 | ID(DP3) | ID(ManiFlow) | Unseen(DP3) | Unseen(ManiFlow) |
|---|---|---|---|---|---|
| 人形 | Grasp & Place | 7/40 | 23/40 | 3/20 | 12/20 |
| 人形 | Pouring | 4/20 | 13/20 | 2/20 | 12/20 |
| 双臂 | Handover | 14/30 | 22/30 | 9/20 | 12/20 |
| 双臂 | Pouring | 21/40 | 30/40 | 12/20 | 15/20 |
| 双臂 | Toy Grasping | 17/50 | 37/50 | 7/30 | 20/30 |
| 双臂 | Sorting | 7/10 | 8/10 | 5/10 | 7/10 |
| 单臂 | Cap Hanging | 4/10 | 7/10 | 2/5 | 4/5 |
| 单臂 | Pouring | 5/10 | 9/10 | 2/10 | 9/10 |
| — | 平均成功率 | 37.6% | 71.0% | 31.1% | 67.4% |

即分布内相对提升 88.8%、未见物体相对提升 116.7%、整体相对提升 98.3%。双臂 Handover 任务上 ManiFlow 22/30(73%)明显优于 DP3 14/30(47%),体现了在双手协调抓取交接这类高难度任务上的优势。

**少步推理效率**(Table 4,RoboTwin 5 个双臂任务):ManiFlow 仅用 1 步 / 2 步推理即达 63.7% / 64.5% 平均成功率,优于 3D Diffusion Policy 与 3D Flow Matching Policy* 用 10 步推理达到的 42.7% / 48.1%。

**生成目标消融**(Table 5,7 个 MetaWorld 困难任务,统一 encoder + DiT-X):ManiFlow 平均 78.0±2.8,优于 DDIM(77.2±3.3)、Consistency-FM(76.3±3.8)、Shortcut Model(76.2±3.6)、Rectified Flow(75.7±4.4)等同样不依赖教师模型的生成式训练目标。

**作为策略头插入 3D Diffuser Actor**(Table 6,CALVIN 零样本长程评测):单步 ManiFlow(平均完成链长 3.67)相对 25 步 DDPM(3.35)提速 25 倍且效果更好;10 步 ManiFlow 进一步达到 4.03,连续完成 5 条指令的成功率为 65.7% vs DDPM-25 的 41.2%,同时超过 RoboFlamingo(2.48)、SuSIE(2.69)、GR-1(3.06)等基线。

## 四、局限性

论文第 6 节及附录 A.4 明确列出以下局限:

- 真机表现高度依赖示范数据的质量与多样性;作者认为把 ManiFlow 纳入强化学习框架有望降低对示范数据量的依赖,但本文未做相关实验验证,留作未来工作。
- 架构设计虽面向灵巧操作,作者推测理论上可扩展到导航、移动操作等任务,但论文未提供任何实验支撑这一推测,仍属未验证的展望。
- 多模态条件化目前只验证了视觉(2D/3D)、语言、本体感觉三种模态,触觉信息或基于 VLM 的条件化(点、轨迹、bounding box)等尚未探索。
- 显式失败案例分析:ManiFlow 在需要精细接触信息和力反馈的任务(如精密装配、柔顺插入)上会失败,原因是其设计聚焦运动学控制(kinematic control)而非基于力的交互,缺乏触觉感知和力控能力。
- 在 RoboTwin 2.0 的 Open Laptop 任务上,从零训练的 ManiFlow(66.7%)略逊于大规模预训练后微调的 π0(70.0%),说明数据高效的从零训练策略并非在所有任务上都能全面超越"大规模预训练+微调"范式。
- DiT-X 对 cross-attention 的双向 AdaLN-Zero 调制带来"适度的计算开销"(论文原话),但论文未报告具体的推理延迟或参数量对比数字,难以量化这一开销的实际代价。

## 五、评价与展望

**优点**：

- 将一致性训练直接内嵌进单阶段联合训练,而非像 Consistency Policy、ManiCM 那样依赖预训练教师模型做蒸馏,训练流程更简洁,同时仍能实现 1-2 步高质量动作生成,是对"少步生成加速"这条技术路线的一种实用化简化。
- 对 5 种时间步采样策略做了系统消融并给出可解释的结论(偏向高噪声区的 Beta 分布更适合机器人控制),补上了 flow matching 训练细节里长期被忽视的一个工程问题,该结论具备较强的可迁移参考价值。
- 评测覆盖面较广:2D/3D 双模态输入 × 4 个仿真 benchmark(RoboTwin 1.0/2.0、Adroit、DexArt、MetaWorld)× 3 种真机平台(单臂、双臂、人形灵巧手),尤其是人形与双臂灵巧手的真机结果在同类 flow/diffusion 策略论文中较为少见,提供了有意义的实证数据。
- DiT-X 作为策略头可即插即用地嵌入已有架构(如 3D Diffuser Actor),并在 CALVIN 长程任务上得到验证,说明其贡献不局限于 ManiFlow 自身的训练 pipeline,具备一定的架构通用性。

**局限与开放问题**：

- 与 π0 的对比本质上是"小规模从零训练"对比"大规模预训练后小样本微调",两者起点不同。虽然论文以此凸显数据效率优势,但也回避了一个更根本的问题——如果 ManiFlow 的架构与训练目标也用于大规模预训练,是否仍能保持相对优势,这仍是开放问题。
- 一致性目标的效果验证局限于与 Rectified Flow、Shortcut Model、Consistency-FM 等同样不依赖教师模型的方法比较(Table 5),未与显式教师-学生蒸馏方法(如 Consistency Policy)在同等推理步数下做直接头对头的效果/训练成本比较,横向覆盖仍有限。
- 论文明确指出接触力反馈是短板,这与近期强调触觉/力控融合的操作策略工作形成互补,是自然的后续扩展方向。
- 3D 编码器仍是轻量 PointNet,论文自己在附录也承认其对复杂场景语义理解能力有限,并提出未来可接入预训练 3D 基础模型或从 2D 视觉-语言模型中提升语义先验——这与当前"视觉基础模型语义 lifting 到 3D 表征"这一具身操作研究中的公开趋势相一致。

## 参考

- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR 2023.
- Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023.
- Prasad et al. *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation.* RSS 2024.
- Ze et al. *3D Diffusion Policy.* RSS 2024.
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.
