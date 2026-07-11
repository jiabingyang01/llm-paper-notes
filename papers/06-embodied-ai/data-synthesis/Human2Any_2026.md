# Human2Any：基于约束感知组合式规划的人到机器人迁移

> **论文**：*Human2Any: Human-to-Robot Transfer via Constraint-Aware Compositional Planning*
>
> **作者**：Shuo Cheng, Chuye Zhang, Alfred Cueva, Caelan Garrett\*, Ajay Mandlekar\*, Danfei Xu（\* 表示共同贡献）
>
> **机构**：Georgia Institute of Technology；NVIDIA Corporation
>
> **发布时间**：2026 年 06 月（arXiv 2606.28813）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.28813) | [PDF](https://arxiv.org/pdf/2606.28813)
>
> **分类标签**：`从人类视频学习` `object-centric` `diffusion steering` `motion planning` `跨本体迁移`

---

## 一句话总结

Human2Any 把操作任务表示为**物-物相对运动**（tool-object 相对 target-object 的 SE(3) 轨迹),从人类视频里学到与本体无关的 object-object 交互先验,再在部署时用扩散粒子滤波的**约束感知 steering** 把这些先验与机器人特定的抓取先验、运动规划可行性联合组合,从而在**完全不用真机器人演示**的情况下,让同一份人类知识迁移到 Franka 与 RBY-1 人形机器人,在仿真 OOD 上把成功率从 baseline 的 0.00 拉到 0.71,真机三任务平均 0.80。

## 一、问题与动机

行为克隆(BC)的泛化严重依赖高质量、与本体对齐的机器人演示,而这类数据靠遥操作采集,成本高、难扩展。人类视频量大、多样,天然记录了丰富的物体交互,是极具吸引力的监督来源;但直接把人类演示迁移到机器人面临三重障碍:

1. **本体失配**(embodiment mismatch):人手运动无法直接映射为机器人动作;
2. **场景变化**(scene variation):同一条轨迹在一个布局下可行,换个布局可能因可达性、关节限位、碰撞而不可行;
3. **机器人特定可行性约束**:抓取方式、运动学、避障都与具体本体绑定。

作者的核心洞见是:**在人类视频里真正可迁移的不是人手运动本身,而是"交互结构"**——哪些物体参与、接触何时形成/断开、物体状态如何演化。因此他们提出以 object-centric motion generation 的视角建模:把操作过程表示为一个显式分离 agent-object 与 object-object 交互的因子图。object-object 因子刻画从人类视频学到的、与执行者无关的运动分布;agent-object 因子与可行性项承担抓取、可达、避障等本体特定约束。测试时在机器人的 in-context 约束下采样并 refine 图变量,产生机器人可执行、且保持视频中目标达成结构的交互序列。

## 二、核心方法

### 2.1 交互中心的任务表示

任务由高层 skeleton 指定:

$$\mathcal{S} = [s^1, \dots, s^K], \qquad s^k = \langle a^k, \mathcal{O}^k \rangle$$

其中每个 phase $s^k$ 由交互类型 $a^k$ 和涉及物体集合 $\mathcal{O}^k$ 组成。共三种交互类型:**free-space motion**(自由空间移动,连接各交互阶段)、**agent-object interaction**(建立机器人与工具物体的接触,即抓取)、**object-object interaction**(工具物体相对目标物体如何运动)。

关键的可迁移表示是 **object-object 交互轨迹**:

$$\tau^{\text{rel}} = \{\mathbf{T}^{\text{rel}}(h)\}_{h=0}^{H-1}, \qquad \mathbf{T}^{\text{rel}}(h) \in SE(3)$$

$\mathbf{T}^{\text{rel}}(h)$ 是 $h$ 时刻工具物体相对目标物体的位姿。**用大白话说**:不管是人手还是机械爪、也不管场景怎么摆,"杯子相对碗怎么倾倒、勺子相对锅怎么放进去"这套相对运动是任务的本质,把人手和手臂运动统统丢掉,只留下这条相对位姿轨迹。本工作聚焦 prehensile(抓持式)操作,机器人抓住工具后保持刚性连接,因此执行可以由 object-object 运动加一个抓取位姿恢复出来。

### 2.2 从人类视频学 object-object 先验

对每条演示,假设可得工具物体与目标物体点云 $\mathbf{P}^A, \mathbf{P}^B \in \mathbb{R}^{N\times 3}$ 及相对轨迹 $\tau^{\text{rel}}$。实现上从 RGB-D 视频用现成分割与跟踪工具估计:给定跟踪到的 3D 关键点 $\xi^{3d}\in\mathbb{R}^{H\times M\times 3}$,用 **Kabsch 算法 + RANSAC** 估计 $\tau^{\text{rel}}$,并按任务相关的距离阈值(约 0.15–0.2 m,靠近交互段)裁剪轨迹。得到数据集 $\mathcal{D}_{\text{os}}=\{(\mathbf{P}_i^A,\mathbf{P}_i^B,\tau_i^{\text{rel}})\}$。

### 2.3 机器人特定的 agent-object 先验

object-object 先验只说物体该怎么互动,没说机器人该怎么抓。于是为**每个机器人本体**用仿真抓取数据训练一个 agent-object 先验:每个样本含工具物体点云 $\mathbf{P}^A$ 与机器人末端到达并抓取物体的轨迹 $\tau^e$,即 $\mathcal{D}_{\text{ro}}=\{(\mathbf{P}_i^A,\tau_i^e)\}$。先验 $p_{\theta_{\text{ro}}}(\tau^e \mid \mathbf{P}^A)$ 刻画本体特定的抓取行为。**这种分离让物体级交互知识(来自人类视频)可跨本体复用,而机器人特定执行由 agent-object 先验和测试时可行性推理处理。**

### 2.4 扩散轨迹先验训练

object-object 与 agent-object 先验都实例化为**条件扩散模型**。前向过程:

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon, \qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

训练目标是预测注入噪声:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\Big[\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \tilde c)\|_2^2\Big]$$

其中 object-object 先验的条件 $\tilde c=(\mathbf{P}^A,\mathbf{P}^B)$,agent-object 先验的条件 $\tilde c=\mathbf{P}^A$。训练时对点云和轨迹同步施加刚体变换以增强对物体位姿、场景布局的鲁棒性。

### 2.5 约束感知组合(核心创新)

部署时机器人观察到 context $c$(机器人模型、场景点云、任务 skeleton)。每个先验能生成多样但**独立看似合理**的候选运动段,但独立组合起来未必可执行。作者把测试时执行表述为**约束感知组合**问题。

**轨迹装配**:令 $\mathbf{x}^{1:K}=\{\mathbf{x}^1,\dots,\mathbf{x}^K\}$ 为 $K$ 个阶段的采样运动段。装配算子 $\Gamma(\mathbf{x}^{1:K},c)$ 把这些段转成完整机器人轨迹并评估可行性。刚性抓取假设下,object-object 段转末端目标为:

$$\mathbf{T}^e(h) = \mathbf{T}^{\text{rel}}(h)\,\mathbf{T}^e(0)$$

$\mathbf{T}^e(0)$ 是抓住工具物体后的末端位姿。装配后的轨迹再按运动学可行性、避障、任务约束检查。

**为什么不用 rejection sampling**:独立采样各段、只接受 $\Gamma$ 可执行的组合,期望试验次数为

$$\mathbb{E}[N_{\text{trial}}] = 1/q(c), \qquad q(c)=\Pr(\mathcal{E}=1 \mid c)$$

当可行组合只占联合样本空间一小块时(抓取、运动学、碰撞约束一紧就如此),这种做法极其低效。**用大白话说**:两个采样器各自都给出局部合理的样本,但只有一小撮组合同时满足全局约束(比如"倒的方向要平行于某参考方向"),rejection 只在样本全部生成后才检查,大量算力浪费在不兼容组合上。

**约束感知 steering**:目标分布定义为

$$p(\mathbf{x}^{1:K} \mid c, \mathcal{E}=1) \propto p(\mathcal{E}=1 \mid \mathbf{x}^{1:K}, c)\prod_{k=1}^{K} p_{\theta_k}(\mathbf{x}^k \mid \tilde c^k)$$

可行性似然为软性(soft)打分:

$$p(\mathcal{E}=1 \mid \mathbf{x}^{1:K}, c) \propto \exp\big(\beta\,\mathcal{S}(\Gamma(\mathbf{x}^{1:K}, c), c)\big)$$

$\mathcal{S}$ 对装配轨迹在当前 in-context 约束下打分,$\beta$ 控制 steering 强度。用**粒子滤波**近似该后验:在去噪步 $t$ 维护 $M$ 个粒子 $\{\mathbf{x}_{t,i}^{1:K}\}$,先用 Tweedie 公式估计干净段轨迹:

$$\hat{\mathbf{x}}_{0,i}^k = \frac{1}{\sqrt{\bar\alpha_t}}\Big(\mathbf{x}_{t,i}^k - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta(\mathbf{x}_{t,i}^k, t, \tilde c^k)\Big)$$

再由 $\Gamma$ 装配并按权重重采样:

$$w_{t,i} \propto \exp\big(\beta\,\mathcal{S}(\Gamma(\hat{\mathbf{x}}_{0,i}^{1:K}, c), c)\big)$$

**用大白话说**:不是等去噪完成再一票否决,而是在去噪过程中每一步都用可行性分数(IK 残差、碰撞裕度、运动规划可行性等连续量)引导,把粒子逐步搬向能满足约束的区域;因为分数是软的,早期去噪步无需完全可执行,steering 渐进地把算力重新分配给"去噪估计更接近约束满足"的粒子。

**实现**:$\Gamma$ 用 **MPLib** 合成自由空间运动,并直接在观测到的场景点云上验证运动学与碰撞约束,规划时**无需物体网格模型**。找到可行轨迹后:Franka 用 OSC 以 20 Hz 跟踪末端目标;RBY-1 的躯干、双臂、灵巧手用关节位置控制,移动底盘用 PD 控制跟踪。

## 三、实验结果

设计六个假设 H1–H6。仿真为三个基于 MuJoCo 的 MimicLab 域(POURINBOWL / HANGMUGTREE / PREPARETABLE),每域含两个 in-distribution 变体(S1、S2)与一个 OOD 变体(布局、物体排布、交互 context 显著不同,物体额外随机化约 0.1 m 平移、30° 旋转);训练用两个变体,第三个留作 OOD。baseline 含 **DP3**(域内 BC,仿真每任务 100 条真机演示、真机实验 50 条)、**Im2Flow2Act**(从人类视频预测 2D object flow 再用仿真数据转机器人动作)、以及**去掉 steering 的 Rejection Sampling** 消融。

### 仿真基准(成功率,Table 1)

| 方法 | PIB S1 | PIB S2 | PIB OOD | HMT S1 | HMT S2 | HMT OOD | PT S1 | PT S2 | PT OOD | 总 ID | 总 OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DP3 | 0.46 | 0.50 | 0.00 | 0.18 | 0.10 | 0.00 | 0.00 | 0.10 | 0.00 | 0.22 | 0.00 |
| Im2Flow2Act | 0.10 | 0.04 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| Reject Sampling(无 steering) | 0.72 | 0.56 | 0.52 | 0.72 | 0.42 | 0.40 | 0.46 | 0.66 | 0.72 | 0.59 | 0.55 |
| **Ours** | **0.86** | 0.56 | **0.80** | 0.72 | **0.50** | **0.60** | **0.60** | 0.56 | 0.72 | **0.63** | **0.71** |

关键读数:两个 BC/flow baseline 在 OOD 上**全线归零**;DP3 用了 100 条真机演示总 ID 仍只有 0.22,说明直接动作回归难以应付多样物体形状/位姿/布局与长程误差累积。Human2Any 总 OOD 0.71,大幅领先,支持 H1(无本体对齐数据也能学难任务)与 H2(无 in-context 数据也能泛化到新 context)。对比 Reject Sampling(同样组件、仅关掉 steering),Ours 在总 ID(0.63 vs 0.59)与总 OOD(0.71 vs 0.55)均更优,支持 H5。**值得注意**:并非处处占优——PrepareTable S2(0.56 vs 0.66)、以及个别 in-distribution 格子上 Reject Sampling 反超,steering 的收益主要体现在 OOD 与整体一致性上。

### 真机 tabletop(Franka,10 次/任务,Table 2)

| 方法 | PourInBowl | HangMugTree | SortUtensils | 平均 |
|---|---|---|---|---|
| DP3 | 4/10 | 4/10 | 0/10 | 0.27 |
| Im2Flow2Act | 1/10 | 2/10 | 0/10 | 0.10 |
| **Ours** | **8/10** | **7/10** | **9/10** | **0.80** |

无真机训练数据下 Human2Any 平均 0.80,支持 H3。

### 跨本体(RBY-1 人形移动机器人)

在与 Franka 运动学差异巨大的 RBY-1(带灵巧手)上部署:**PourCup 0.6、UseRoller 0.7**,说明学到的交互先验不绑定特定机器人形态,通过测试时解算本体约束即可跨平台复用,支持 H4。

### 消融与扩展性

- **H5(steering 提升质量与效率)**:粒子可行性分数随去噪步上升且显著高于 No Steering;在所有仿真域取得最高的 task-level 吞吐(一小时预算内含推理/规划/执行的成功试验数),即 steering 减少了花在不可行候选上的算力。
- **H6(随交互运动数据量提升)**:在 PourInBowl 上改变 object-centric 交互轨迹的数量与多样性,成功率随数据增多而提升(Ours 约从 0.66 升到 0.85),由于这些轨迹与从人类视频抽取的运动同形,间接佐证了用更大规模人类视频源扩展的潜力。

## 四、局限性

1. **仅限抓持式操作**:假设末端与工具物体刚性连接,从固定抓取位姿+物体相对运动恢复控制目标;放宽此假设(引入学习动力学模型)才能覆盖非抓持技能如 in-hand 灵巧操作。
2. **依赖给定 task skeleton**:阶段序列需人为提供;若与 task planning 或语言引导的分解结合,才是更完整的 task-and-motion planning 系统。
3. **缺在线失败检测与重规划**:当前实现没有 online replanning,面对感知噪声、物体滑动、意外场景变化(尤其长程执行)鲁棒性受限。
4. **感知与估计链路脆弱**:$\tau^{\text{rel}}$ 依赖现成分割/跟踪 + Kabsch/RANSAC,论文未量化跟踪误差对下游成功率的传播;仿真规模较小(每任务数百次评估),真机每任务仅 10 次,统计噪声不可忽略。

## 五、评价与展望

**优点**:(1)问题拆解干净——把"可迁移的物-物交互结构"与"本体特定的抓取/可行性"显式解耦为因子图,是 learning-from-human-video 里一个概念清晰且工程可落地的抽象,规避了 flow-based 方法(如 Im2Flow2Act)"直接预测机器人动作却不显式强制可行性"的痛点。(2)**约束感知 steering 是本文最有价值的技术贡献**:把 diffusion steering / 引导采样与 TAMP 式可行性打分结合,用粒子滤波在去噪过程中而非之后满足全局组合约束,对比 rejection sampling 在紧约束下的指数级低效有清晰的理论动机($\mathbb{E}[N_{\text{trial}}]=1/q(c)$)。(3)零真机演示即可跨 Franka/RBY-1 迁移,且在点云上直接做碰撞检查、无需物体网格,实用性强。

**缺点与开放问题**:(1)steering 并非在所有 setting 都占优(PrepareTable S2 被 Reject Sampling 反超),说明软打分 $\mathcal{S}$ 的设计与 $\beta$ 调节对结果敏感,论文未做 $\beta$ 或粒子数 $M$ 的敏感性分析。(2)"刚性抓取假设"把 object-object 到末端的映射简化为一次左乘 $\mathbf{T}^e(0)$,这对倾倒/放置够用,但对需要 regrasp 或接触状态切换的任务是硬约束。(3)与并行工作的关系:相较 object flow 类(Track2Act、Im2Flow2Act)用 2D/3D flow 作迁移接口,本文用 SE(3) 相对位姿轨迹 + 显式规划,牺牲了端到端的简洁性换取可行性可控性;相较联合优化 grasp+trajectory 或用机器人数据做 alignment 的工作,本文的卖点是"物体级知识纯从人类视频、执行级约束测试时解算"。(4)可改进方向:把 task skeleton 的产生交给 VLM/语言分解、引入闭环失败检测与重规划、以及真正在大规模野外人类视频(如 EgoDex、Egomimic 这类 egocentric 数据)上验证 H6 的扩展曲线,是把该框架推向"可扩展数据引擎"的自然下一步。

## 参考

1. M. Xu, et al. *Flow as the Cross-Domain Manipulation Interface* (Im2Flow2Act). arXiv:2407.15208, 2024.（flow-based 迁移接口,本文主要 baseline）
2. H. Bharadhwaj, et al. *Track2Act: Predicting Point Tracks from Internet Videos Enables Generalizable Robot Manipulation.* ECCV 2024.（point track 作跨域接口的代表)
3. Y. Ze, et al. *3D Diffusion Policy* (DP3). RSS 2024.（域内 BC baseline)
4. C. Chi, et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023.（扩散轨迹生成的基础)
5. V. Saxena, et al. *What Matters in Learning from Large-Scale Datasets for Robot Manipulation* (MimicLab). ICLR 2025.（仿真基准来源)
