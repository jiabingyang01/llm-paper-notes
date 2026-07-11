# STARRY：面向机器人操作的时空联合、动作中心化世界建模

> **论文**：*STARRY: Spatial-Temporal Action-Centric World Modeling for Robotic Manipulation*
>
> **作者**：Yuxuan Tian, Yurun Jin, Bin Yu, Yukun Shi, Hao Wu, Chi Harold Liu, Kai Chen, Cong Huang 等
>
> **机构**：Beijing Institute of Technology；Zhongguancun Academy；Zhongguancun Institute of Artificial Intelligence；University of Science and Technology of China；Harbin Institute of Technology；East China Normal University；DeepCybo
>
> **发布时间**：2026 年 04 月（arXiv 2604.26848）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.26848) | [PDF](https://arxiv.org/pdf/2604.26848)
>
> **分类标签**：`世界模型` `VLA` `时空联合扩散` `几何引导注意力` `双臂操作` `RoboTwin 2.0`

---

## 一句话总结

STARRY 用一个统一扩散过程联合去噪"未来时空隐变量"与"未来动作序列"，并引入 GASAM（Geometry-Aware Selective Attention Modulation）把预测深度与末端位姿转成 token 级几何权重去偏置动作到视觉的注意力，在 RoboTwin 2.0 的 50 个双臂任务上取得 Clean/Randomized 平均 **93.82% / 93.30%**（超过 LingBot-VA 的 92.93%/91.55%、Motus 的 88.66%/87.02%），真机三任务上把平均成功率从 π0.5 的 **42.5%** 提升到 **70.8%**。

## 一、问题与动机

VLA（Vision-Language-Action）模型已成为通用具身智能体的主流范式，但多数策略仍是"反应式"的：直接把当前或短历史观测映射到动作，没有显式建模未来的机器人-物体交互状态。像悬挂水杯、递物、容器摆放这类任务，需要预判物体几何、接触区域与末端轨迹，局部关系判断失误会导致抓取不稳、碰撞或放置失败。

近期工作把世界模型引入策略学习（预测未来观测或隐式视频状态），但作者指出：**未来预测的视觉合理性不等于对控制有用**——预测出的"看起来对"的未来画面，未必暴露出物体把手、接触面、开口、障碍物、末端邻域这些动作相关的空间约束。作者归纳出两个具体局限：

1. 现有预测式表征通常为感知/时间一致性而优化，而非动作相关性，造成"预测"与"控制"之间的错配；
2. 空间信息通常通过全局共享表征融合，难以把决策关键区域从背景语境中区分出来。

核心洞察：面向操作的世界模型应当同时是 **action-centric**（以动作为中心）和 **geometry-grounded**（几何落地）的——不仅要预测场景如何演化，还要预测未来交互发生在哪里。

## 二、核心方法

### 2.1 整体架构与问题形式化

在语言条件的具身操作设定下，agent 在时刻 $t$ 观测 $\mathbf{o}_t=\{\mathbf{I}_t,\mathbf{D}_t,\mathbf{c}_t,\mathbf{p}_t,\mathbf{l}\}$（多视角 RGB-D、相机参数、当前末端位姿、语言指令），目标是生成未来动作序列 $\mathbf{a}_{t+1:t+H}$。STARRY 引入两个内部结构：未来时空隐变量 $\mathbf{z}_{t+1:t+H}$ 与几何感知调制权重 $\mathbf{w}_{t+1:t+H}$，联合建模为

$$\pi_\theta(\mathbf{a}_{t+1:t+H}, \mathbf{z}_{t+1:t+H} \mid \mathbf{o}_t).$$

**用大白话说**：策略不是直接从观测蹦到动作，中间先"脑补"一段带几何结构的未来（$\mathbf{z}$），再用一份"哪里重要"的权重图（$\mathbf{w}$）去引导动作生成往关键区域看。

架构含四个模块（图 1）：*Understanding Expert*（语义理解，由 Qwen-VL 初始化）、*ST World Model*（时空世界模型，由 Wan 视频扩散模型初始化）、*Geometry Expert*（几何专家）、*Action Expert*（动作专家，与 ST World Model 同构但参数独立）。四者通过 Multi-modal Joint Attention 耦合：ST World Model 与 Action Expert 联合去噪未来时空隐变量与动作；Geometry Expert 预测未来几何状态，经 GASAM 转成 token 对齐权重，只选择性调制动作注意力分支。

### 2.2 ST World Model：时空联合隐变量

给定时间窗口 $[t_0,t]$ 内的多视角 RGB 图像 $\{\mathbf{I}_t^c\}_c$、深度观测 $\{\mathbf{D}_t^c\}_c$ 与三维末端轨迹 $\{\mathbf{e}_\tau^m\}_{\tau\le t,m}$，先用相机内外参 $K,T$ 把轨迹投影进各相机视角，再融合外观、几何、运动为统一表示：

$$\mathbf{u}_\tau^{c,m} = \Pi(K_t^c,T_t^c,\mathbf{e}_\tau^m), \qquad \mathbf{x}_t = \Phi(\{\mathbf{I}_t^c\}_c,\{\mathbf{D}_t^c\}_c,\{\mathbf{u}_\tau^{c,m}\}_{\tau\le t,c,m}).$$

**用大白话说**：把每个时刻的多视角 RGB、深度图、投影到画面里的末端轨迹点"拼"成一张统一的 RGB-D 布局图，作为世界模型的输入。

编码得到视频 token $\mathbf{v}_{t_0:t}$，结合历史动作 $\mathbf{a}_{\le t}$ 预测未来隐变量：

$$\mathbf{z}_{t+1:t+H} = f_\theta^{\mathrm{ST}}(\mathbf{v}_{t_0:t}, \mathbf{a}_{\le t}),$$

其中 $f_\theta^{\mathrm{ST}}$ 是一个扩散模型。因为输入里显式融合了外观、轨迹、几何三种信息，$\mathbf{z}_{t+1:t+H}$ 天然携带场景演化、末端运动趋势与几何约束，为后续动作生成提供有效条件。

### 2.3 GASAM：几何感知选择性注意力调制

标准联合注意力用 query-key 点积在 token 空间里匹配动作 token 与 2D 视觉 token，等于让策略隐式推断透视和深度关系——当视觉相似性与真实度量距离不一致时，容易造成较大的三维控制偏差。GASAM 的思路是把 2D 视觉 token 显式"提升"到度量三维空间，让视觉观测与物理控制对齐。

**几何预测**：由于 $\mathbf{z}_{t+1:t+H}$ 在去噪过程中是含噪的潜变量，不适合直接做深度反投影这类几何运算，因此单独训练一个 Geometry Expert $g_\phi$，以 $\mathbf{o}_t$、视频 token $\mathbf{v}_{t_0:t}$、历史动作 $\mathbf{a}_{\le t}$ 以及视频/动作两支的扩散时间步 $(\tau_v,\tau_a)$ 为条件，预测未来深度序列与末端位置：

$$(\hat{\mathbf{D}}_{t+1:t+H}, \hat{\mathbf{p}}_{t+1:t+H}) = g_\phi(\mathbf{v}_{t_0:t}, \mathbf{a}_{\le t}, \mathbf{o}_t, \tau_v, \tau_a).$$

**几何权重构造**：用预测深度 $\hat{\mathbf{D}}_t$ 反投影出三维点 $\hat{\mathbf{P}}_{t,j}$，计算其与预测末端位置的距离，再单调递减映射并采样/聚合到视频 token 网格：

$$d_{t,j} = \lVert \hat{\mathbf{p}}_t - \hat{\mathbf{P}}_{t,j} \rVert_2, \qquad \mathbf{w}_{t+1:t+H} = \mathcal{T}\big(\rho(\{d_{t,j}\}_{t,j})\big).$$

**用大白话说**：先"脑补"末端要去哪、周围深度长什么样，再算每个视觉 token 对应的三维点离预测末端有多远——越近权重越高，这张权重图告诉动作分支"该看哪里"。

调制只作用在动作到视觉的注意力（action-to-video attention）上，把 $\log(\mathbf{w}+\epsilon)$ 作为加性偏置在 softmax 前注入：

$$\mathrm{Attn}_{\mathrm{GASAM}}^{a\leftarrow v} = \mathrm{Softmax}\Big(\frac{\mathbf{Q}^a(\mathbf{K}^v)^\top}{\sqrt{d}} + \lambda\log(\mathbf{w}_{t+1:t+H}+\epsilon)\Big)\mathbf{V}^v,$$

其中 $\lambda$ 控制调制强度，$\epsilon$ 防止数值不稳定。调制只施加于动作分支，保留了原始时空建模与视觉-语言理解分支不被干扰。

### 2.4 训练目标与三阶段流程

模型以速度场形式的扩散目标联合建模时空观测与动作：

$$\mathcal{L}_{\mathrm{obs}} = \mathbb{E}\big[\lVert \mathbf{v}_\theta^o-(\boldsymbol\epsilon_o-\mathbf{x}_{t+1:t+H})\rVert^2\big], \quad \mathcal{L}_{\mathrm{action}} = \mathbb{E}\big[\lVert \mathbf{v}_\theta^a-(\boldsymbol\epsilon_a-\mathbf{a}_{t+1:t+H})\rVert^2\big],$$
$$\mathcal{L}_{\mathrm{diff}} = \lambda_o\mathcal{L}_{\mathrm{obs}} + \lambda_a\mathcal{L}_{\mathrm{action}}.$$

观测与动作两支使用不同的扩散时间步，以适配模态差异。几何专家额外由深度、末端位姿、以及"权重一致性"三项监督：

$$\mathcal{L}_{\mathrm{geo}} = \lambda_d\mathcal{L}_{\mathrm{depth}} + \lambda_p\mathcal{L}_{\mathrm{pose}} + \lambda_w\mathcal{L}_{\mathrm{weight}},$$

其中权重目标由真实末端-场景点距离构造：$w^*_{t,j}=\varphi(\lVert \mathbf{p}_t-\mathbf{P}_{t,j}\rVert_2)$，$\mathcal{L}_{\mathrm{weight}}=\lVert \mathbf{w}_{t+1:t+H}-\mathbf{w}^*_{t+1:t+H}\rVert_2^2$。Geometry Expert 先独立训练，再联合微调，通过 GASAM 影响动作生成，而不是直接改动扩散目标本身。

训练分三阶段：Stage 1（时空预训练）在大规模视频与多模态数据上训练 ST World Model 与 Understanding Expert；Stage 2（动作与几何学习）引入 Geometry Expert 与 Action Expert，在扩散目标下联合训练，学习从时空表征到带几何约束动作的映射；Stage 3（联合微调）端到端引入 GASAM，所有模块联合优化。数据按 L1–L6 六级层次组织：L1 网络规模视频、L2 第一视角视频（Ego4D、Ego-Dex）、L3 合成/仿真几何数据、L4 交互数据（EmbodiedMAE）、L5 多机器人真机轨迹（DROID、BridgeData V2）、L6 目标机器人数据（用于任务级微调）。

## 三、实验结果

### 3.1 RoboTwin 2.0 仿真基准（50 个双臂任务）

所有任务的演示数据被池化联合训练（而非逐任务单独训练），Clean 设置 50 条演示、Randomized 设置 500 条演示，batch size 256、训练 40k 步。

| 方法 | Clean 平均成功率 | Randomized 平均成功率 |
|---|---|---|
| π0.5 | 62.86% | 60.30% |
| X-VLA | 72.80% | 72.84% |
| Motus | 88.66% | 87.02% |
| LingBot-VA | 92.93% | 91.55% |
| **STARRY（本文）** | **93.82%** | **93.30%** |

（附录完整 50 任务表另含 GO-1 37.8%/36.24%、π0 65.92%/58.4% 两个更弱基线。）平均分差距被高成功率任务的饱和效应"稀释"，作者强调 STARRY 的优势在判别性更强的任务上更明显：*Handover Mic*（双臂精细递交时序协调）STARRY 达 100%/99%，显著超过 Motus 的 78%/63%；*Hanging Mug*（依赖局部几何结构、接触位置）STARRY 达 69%/72%，比此前最好结果高出 30 多个百分点；*Turn Switch*、*Press Stapler* 分别为 85%/89%、100%/100%。

### 3.2 真机实验

在 ARX R5 双臂平台上评测三个代表性双阶段任务：*Hand Over Vegetables*（递菜）、*Tidy Up Room*（整理房间/开盒放物）、*Wash Baby Bottle*（洗奶瓶）。每任务收集 50 条真机演示，每方法评测 20 次 rollout，与 π0.5 对比：

| 任务 | 方法 | Stage 1 | Stage 2 |
|---|---|---|---|
| Hand Over Vegetables | π0.5 / STARRY | 60% / **85%** | 40% / **70%** |
| Tidy Up Room | π0.5 / STARRY | 55% / **75%** | 35% / **65%** |
| Wash Baby Bottle | π0.5 / STARRY | 40% / **70%** | 25% / **60%** |

六项评测总体平均成功率 STARRY 达 **70.8%**，π0.5 为 **42.5%**；更具挑战性的 Stage 2（完整多步任务）上提升幅度更大（+31.7 个百分点），表明优势不仅来自短时程动作稳定，也来自更强的多步时空建模能力。

### 3.3 消融实验（RoboTwin 2.0）

对比三种预测表征强度（Act. 纯动作去噪、App. 仅外观预测、ST 完整时空建模）与 GASAM 开关：

| 表征 \ 设置 | Randomized（GASAM 关→开） | Clean（GASAM 关→开） |
|---|---|---|
| Act.（纯动作） | 64.96% → 75.88%（+10.92） | 63.42% → 72.50%（+9.08） |
| App.（仅外观） | 85.80% → 86.96%（+1.16） | 86.64% → 87.86%（+1.22） |
| ST（完整时空） | 88.82% → **93.30%**（+4.48） | 90.40% → **93.82%**（+3.42） |

不引入 GASAM 时，从纯动作去噪切到仅外观预测，Randomized 上就有 64.96%→85.80% 的大幅提升，说明未来外观预测本身已能提供有用的时间上下文；进一步引入完整时空建模再提升到 88.82%/90.40%，说明轨迹演化和空间几何的显式建模仍能带来额外约束。GASAM 在所有表征变体上都带来一致增益，且在"纯动作去噪"设置下增益最大（+10.92%/+9.08%），说明几何权重即便在没有完整未来预测时也能提供有效的空间先验；与完整时空建模叠加后取得最佳整体结果 93.30%/93.82%。

## 四、局限性

论文附录 C 明确列出三点：

1. 真机评测仅在有限数量的双臂任务上进行，尚需在更多机器人本体和开放环境下做更广泛验证；
2. GASAM 依赖预测深度与末端几何的准确性，在观测高度模糊的情况下，几何预测不可靠会削弱调制信号的有效性；
3. STARRY 建立在较大的视频与视觉-语言骨干网络之上，相比更小的反应式策略，训练成本更高（微调 40k 步、batch 256 在 8×A100-80G 上约需一周）。

## 五、评价与展望

**优点**：STARRY 把"世界模型预测未来"与"策略生成动作"这两条通常割裂的支路，通过共享 Multi-modal Joint Attention 和统一扩散时间步机制真正耦合在同一个去噪过程里，而不是像多数 world-model-enhanced 策略那样先预测未来帧/隐变量再喂给下游策略（如 F1 的"预测引导的逆动力学"范式）。GASAM 的设计亮点在于它没有试图让隐变量本身承担"精确几何"的角色（隐变量在扩散过程中始终含噪，直接做深度反投影并不稳定），而是单独训练一个显式的 Geometry Expert 产出可解释的深度/位姿，再以 softmax 加性偏置的形式温和地注入注意力，这种"隐式隐变量 + 显式几何引导"的解耦设计是本文相对于纯隐式世界模型（如 Motus、WoW）路线的一个务实折衷。消融实验里"纯动作去噪 + GASAM"仍有大幅提升，也从侧面印证几何先验和时空预测是两条互补而非冗余的信息通路。

**不足与开放问题**：（1）RoboTwin 2.0 上的评测采用 50 任务联合池化训练，这与不少基线论文报告的单任务或少任务设置可能存在协议差异，跨论文横向比较需谨慎，最公平的判断仍应看论文内部同协议下的相对排序；（2）真机验证仅 3 个任务、20 次 rollout/方法，样本量偏小，且都在同一双臂平台（ARX R5）上完成，尚未展示跨本体（如轮式/移动操作、单臂）泛化能力；（3）GASAM 的几何权重完全依赖预测深度的准确度，论文未给出深度预测误差与最终操作成功率之间的定量敏感性分析，这层"几何预测——注意力调制——动作成功率"的因果链条目前主要靠消融实验的整体数字佐证，缺少更细粒度的诊断（例如遮挡、透明/反光物体等深度预测困难场景下 GASAM 是否退化）；（4）作者认为四专家（Understanding/ST World Model/Geometry/Action）+ 联合注意力的架构相比反应式策略显著增加了训练与推理开销，但论文未报告推理延迟或控制频率，这对实机部署（尤其是高频闭环控制）的可行性评估是一个缺口；（5）与同期工作相比，STARRY 延续了"世界模型信息应经过筛选/加权后再输给动作"的思路（类似 GeoPredict 用预测运动学和 3D 高斯几何做精确 VLA 操作的取向），但把这一思想系统化为一个显式的注意力偏置项并给出闭式训练目标，是较为清晰的工程贡献；未来值得探索的方向包括把 GASAM 的几何先验扩展到力/触觉等更多模态，以及在几何预测不确定时引入置信度自适应的调制强度。

## 参考

1. Black et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization.* CoRL 2025.（本文主要真机与仿真基线）
2. Bi et al. *Motus: A Unified Latent Action World Model.* arXiv:2512.13030, 2025.（本文最强的隐式世界模型类基线，仿真平均成功率 88.66%/87.02%）
3. Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation.* arXiv:2506.18088, 2025.（本文核心评测基准，50 个双臂任务）
4. Lv et al. *F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions.* arXiv:2509.06951, 2025.（预测引导逆动力学范式的代表性对比工作）
5. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR 2024.（动作扩散建模的基础工作）
