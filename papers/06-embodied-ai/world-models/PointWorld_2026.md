# PointWorld：面向真实开放场景机器人操作的可扩展 3D 世界模型

> **论文**：*PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation*
>
> **作者**：Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei
>
> **机构**：Stanford University；NVIDIA
>
> **发布时间**：2026 年 01 月（arXiv 2601.03782）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.03782) | [PDF](https://arxiv.org/pdf/2601.03782)
>
> **分类标签**：`3D世界模型` `点流(point-flow)` `机器人操作` `具身预训练` `MPC规划`

---

## 一句话总结

PointWorld 把"状态"和"动作"统一表示为共享 3D 空间中的 **point flow(点流)**——状态是 RGB-D 得到的整场景点云,动作是由 URDF 前向运动学生成的、与本体无关(embodiment-agnostic)的机器人稠密 3D 点流——并用 PointTransformerV3 骨干在约 200 万条轨迹上预训练一个"动作条件下的整场景 3D 点流预测器";单个 1B 预训练 checkpoint 配合 MPPI 即可让真实 Franka 在野外场景仅凭一张 RGB-D 图、无需任何演示或微调,完成刚体推动、可变形/铰接物体操作、工具使用等任务(推拉抽屉 90%、推纸巾盒 70%、叠围巾 80% 等零样本成功率)。

## 一、问题与动机

通用机器人需要一个"世界模型":给定当前所见与打算做的动作,预测世界如何演化。已有路线各有短板:物理仿真器精确但有 sim-to-real gap 且需逐场景建模;学习式动力学模型往往依赖强领域先验(完全可观、objectness 先验、材料指定);大规模视频生成模型逼真但缺乏显式动作条件,常常违反物理一致性。

作者的核心哲学是 **为可扩展而统一表示(unification for scaling)**:把 state 与 action 放进同一种模态——3D 物理空间中的点。state 用 RGB-D 重建的整场景点云表示,action 用机器人自身本体(已知 URDF)前向运动学实例化出的稠密 3D 轨迹表示。在这种表示下,3D 世界建模等价于**在一段机器人动作点序列的扰动下,预测整场景 3D 点流(full-scene 3D point flow)**。这样做的好处:(1) 用点流而非本体专属动作空间(如关节角),动作直接以机器人的物理几何为条件,天然跨本体、跨任务共享;(2) 从任意 RGB-D 采集即可获得,支持部分可观、无需场景重建;(3) 用 L2 位移回归即可稳定训练,无需排列匹配;(4) 表达力足以刻画多样的细粒度接触动力学。文中强调这类似于对 3D 空间与时间做"next-token 预测"。

## 二、核心方法

### 状态与动作:都是 3D 点流

动力学被建模为神经网络 $\mathcal{F}_\theta: \mathbf{S}\times\mathbf{A}\to\mathbf{S}$。不同于逐步更新 $s_{t+1}=\mathcal{F}_\theta(s_t,a_t)$,作者采用 **chunked(分块)** 公式,一次前向预测未来 $H$ 步:

$$\mathcal{F}_\theta^{H}:(s_t,\,a_{t:t+H-1})\to s_{t+1:t+H}$$

用大白话说:模型不是"走一步看一步",而是"给我未来 10 步的机器人动作,我一口气把这 10 步里整个场景怎么动都算出来"。取 $H=10$、每步 0.1 秒,一次前向即覆盖 1 秒,既保证时间一致性又摊薄了计算。

**状态**:时刻 $t$ 的点流 $s_t=\{(\mathbf{p}_{t,i},\mathbf{f}_i^{S})\}_{i=1}^{N_S}$,含 $N_S$ 个点的位置 $\mathbf{p}_{t,i}\in\mathbb{R}^3$ 与时不变特征 $\mathbf{f}_i^{S}$。从一或几张标定 RGB-D 视图得到,借前向运动学掩掉机器人像素、反投影其余像素成静态场景点云。

**动作**:给定一段关节配置序列 $\{\mathbf{q}_{t+k}\}_{k=0}^{H}$,在 $t$ 时刻采样机器人表面点并按 link 用前向运动学传播,得到有序的 $N_R$ 个机器人点 $\{(\mathbf{r}_{t+k,j},\mathbf{f}^{R}_{t+k,j})\}$,作为 $a_{t+k}$。用大白话说:动作不是抽象的关节数,而是"机器人自己身体表面这几百个点在未来每一步会移动到哪里"——因为是想象出来的、且用了 URDF,所以即便机械臂在遮挡区(如抱着大箱子)也是**完全可观**的,这正是"跨本体"的关键。实践中为高效只从 gripper 采样(每个夹爪几百个点)。

### 骨干与特征

把静态场景点与时间堆叠的机器人点拼成**一个**点云,送入点云骨干。场景点用**冻结的 DINOv3** 投影到 2D 视图取特征(提供 objectness 先验),机器人点用时间嵌入。骨干选 **PointTransformerV3 (PTv3)**——其点序列化 + U-net 层次注意力兼顾局部与长程,并能扩到十亿参数;骨干输出后接一个共享 MLP head,一次前向预测 chunk 内每步每个场景点的位移。实时延迟约 0.1 秒/前向,远快于扩散式像素预测(通常秒级)。

### 训练目标:三重稳健化

真实数据两大难点:(i) 机器人只操作场景一小部分,绝大多数点静止,直接 L2 会训练信号极稀疏;(ii) 真实数据噪声大易过拟合。完整损失为:

$$\frac{1}{2}\sum_{k,i}^{H,N_S} w_{k,i}\Big(\underbrace{\rho_\delta(\hat{\mathbf{P}}_{t+k,i}-\mathbf{P}_{t+k,i})}_{\text{Huber loss on 3D residual}}\cdot \underbrace{e^{-s_{k,i}}}_{\text{uncertainty weight}}+\underbrace{s_{k,i}}_{\text{uncertainty reg}}\Big)$$

其中:
- **movement weighting**:软移动似然 $m_{k,i}=\sigma\big(\kappa(\delta_{k,i}-\tau)\big)$($\delta_{k,i}$ 为真值位移模长,$\sigma$ sigmoid),归一化后得 $w_{k,i}$。用大白话说:把损失重心压到"真的在动"的点上,别让海量静止背景点淹没学习信号。
- **aleatoric uncertainty regularization**:模型自己预测每点每步的对数方差 $s_{k,i}$,残差按 $e^{-s_{k,i}}$ 加权、再加 $s_{k,i}$ 正则。用大白话说:让模型对"真值本就不可靠"的点(如布料边缘,受物体物理性质影响运动本就随机)自动降权,防止过拟合噪声。有趣的是,这个不确定性无监督地"涌现"出对动作条件下物理不确定性的刻画。
- **Huber loss** $\rho_\delta$:对 3D 残差用 Huber 而非纯 L2,抑制离群噪声。

### 用于操作的 MPC

预训练模型即插进 **MPPI** 采样式规划器,在 $\mathrm{SE}(3)$ 中规划 $T$ 个末端位姿目标。用时间相关(cubic-spline)噪声扰动名义末端轨迹,采样 $K$ 条,转成机器人点流动作 $a^{(\ell)}_{1:T}$,用 PointWorld rollout 出场景点流并累计轨迹代价,按 $\omega_\ell\propto\exp(-J^{(\ell)}/\beta)$ 迭代精化。全局轨迹优化:

$$\arg\min_{\mathbf{E}_{0:T}}\sum_{k=1}^{T}\big[c_{\text{task}}(s_k)+c_{\text{ctrl}}(\mathbf{E}_k)\big]\quad\text{s.t. } s_{1:T}=\mathcal{F}_\theta^{T}(s_0,a_{1:T}),\ \mathbf{E}_0=\mathbf{E}_{\text{measured}}$$

任务代价定义在模型状态空间中:对任务相关场景点集 $\mathcal{I}_{\text{task}}$ 及其目标位置 $\mathbf{g}_i$,$c_{\text{task}}(s_k)=\frac{1}{|\mathcal{I}_{\text{task}}|}\sum_{i\in\mathcal{I}_{\text{task}}}\|\mathbf{p}_{k,i}-\mathbf{g}_i\|_2^2$。用大白话说:人(或 VLM)在图上点几个"要移动的点"和"移到哪",规划器就在世界模型脑内模拟哪条动作能把这些点推到目标,选最优执行——目标以点为单位,对刚体/可变形/铰接物体统一适用。

### 大规模 3D 数据集与标注管线

数据来自 **DROID**(真实、单臂 Franka)与 **BEHAVIOR-1K**(仿真、双臂全身人形),共约 **200 万条轨迹、约 500 小时**。真实数据的 3D 标注是重头戏,采用三阶段无标记离线管线:(1) 用 **FoundationStereo** 立体估计取代原传感器深度(近距工作区更准);(2) 以 **VGGT** 初始化相机位姿,再用"渲染机器人网格对齐观测深度"的优化精化外参;(3) 用 **CoTracker3** 做 2D 点跟踪并借精化后的深度/位姿抬升到 3D,并用可见性掩掉遮挡点。精化外参相对"最优 1% 场景"的中位平移/旋转误差为 **1.8 cm / 1.9°**,最终为超 60% 的 DROID(约 200 小时原始遥操)恢复了可靠 3D 点流。

## 三、实验结果

评测指标为预测视界内 **逐点、逐步的 $\ell_2$ 距离(米)**,并聚焦"移动点"($\ell_2$ mover,用移动似然过滤真正在动的点)与"静止点"($\ell_2$ static);真实域再用独立 expert 模型按不确定性过滤,保留置信度前 80% 的点。因评测集庞大(约 4 万条轨迹 ×1 万点/条),标准误 $\le 10^{-5}$ m,故只报均值。

### 扩展 3D 世界模型的路线图(DROID 测试集,$\ell_2$ mover ↓)

| 改进环节 | 设置 | $\ell_2$ mover |
|---|---|---|
| Backbone | GBND(图神经动力学基线) | 0.0386 |
| Backbone | PointNet | 0.0370 |
| Backbone | **PTv3**(选用) | 0.0348 |
| Training Obj. | +Huber | 0.0342 |
| Training Obj. | +Movement Weighting | 0.0350 |
| Training Obj. | +Uncertainty | 0.0348 |
| Image Feat. | +DINOv3 ViT-L(多层) | 0.0331 |
| Model Scale | 132M → 411M | 0.0324 → 0.0315 |
| Model Scale | **→ 1B**(最终) | **0.0312** |

用大白话说:换骨干、稳目标、加冻结 DINOv3 特征、把模型放大到 1B,每一杠杆都带来一致提升,从 GBND 基线 0.0386 一路降到 0.0312。

### 骨干对比(Table 1,部分)

| Backbone | Params(×GBND) | Latency(ms) | $\ell_2$ mover ↓ | $\ell_2$ static ↓ |
|---|---|---|---|---|
| GBND | 1.00× | 13.46 | 0.0390 | 0.0066 |
| PTv3-50M | 49.14× | 59.60 | 0.0331 | 0.0067 |
| PTv3-411M | 398.67× | 102.47 | 0.0315 | 0.0059 |
| **PTv3-1B** | **957.71×** | **123.65**(≈0.12s) | **0.0312** | **0.0056** |

PTv3 可放大到 GBND 的 957 倍参数,而显存与延迟仅温和增长,仍保持实时。数据与模型规模扩展都呈近 **log-linear** 增益(数据 5%–100%、模型 50M–1B),提示可预期的持续收益。

### 泛化与迁移(Table 2,$\ell_2$ mover ↓,D=DROID/B=B1K/H=留出真实实验室)

| 设置 | In-Domain D→D | Cross-Domain D→B | Held-Out Real D→H | From Scratch(专家) |
|---|---|---|---|---|
| Zero-Shot | 0.0315 | 0.1460 | 0.0305 | 0.0293 |
| Finetuned | — | 0.0107 | 0.0271 | 0.0293 |

用大白话说:预训练模型能域内泛化到未见轨迹;能**零样本迁移到未见的真实新环境**(D→H 零样本 0.0305,与专家 0.0293 接近);仅用 **1/20(20×更少)** 的更新量微调即超过从头训练的专家(0.0271 < 0.0293);真实+仿真联合训练还带来轻微更好的零样本迁移(D+B→H 零样本 0.0300 优于 D→H)。仿真-only 预训练无法零样本迁移到真实。

### 真实机器人零样本 MPC(Figure 8,成功率)

在移动底盘 Franka + 单个 RealSense D435 上,仅凭一张野外 RGB-D、无演示无微调,单个预训练 checkpoint 经 MPPI 完成:

| 任务类型 | 任务 | 成功率 |
|---|---|---|
| 刚体推动 | 纸巾盒 / 书 | 70% / 20% |
| 可变形 | 叠围巾 / 放枕头 | 80% / 40% |
| 铰接 | 开关微波炉门 / 抽屉 | 30% / 90% |
| 工具使用 | 掸子 / 扫帚 | 60% / 60% |

覆盖刚体、可变形、铰接、工具四类,表明预训练模型隐式学到了跨物体的接触推理、铰接/形变推断与物体间(工具)动力学。

### 关键消融

- **动作表示**(Fig 11):gripper-only 点流(300–500 点/夹爪)在有效性、效率与跨本体正迁移间最佳;稠密全身点流因惰性点稀释信号且增算力而落后,低维(末端位姿、关节角)在双臂 B1K 上明显更差。
- **chunked 预测**(Fig 12):训练与推理都用 chunk→chunk 的漂移最小,显著优于自回归/滑窗,且只需一次前向。
- **部分可观**(Fig 13):训练时随机相机数最鲁棒;推理时相机越多误差越低,亚厘米级。

## 四、局限性

作者在附录坦诚列出:
- **静态初始状态**:输入无历史帧/速度,默认观测瞬间世界静止,难处理已在运动的初始条件。
- **无光度动力学**:只预测几何位移,不建模外观(如灯光/屏幕开关),需与 Gaussian Splatting/NeRF 等外观模型结合才能覆盖光度变化任务。
- **刚体机器人假设**:本体建为刚性运动树,忽略柔性/腱驱/顺应结构(如 fin-ray 夹爪)的形变;接触改变的是场景而非机器人自身几何。
- **准静态、已实现动作假设**:把机器人轨迹当作"已知且完全实现"的关节序列,建模的是"若机器人走这条路径环境如何响应",而非"控制器/驱动限制/接触是否真能实现该路径";强接触、重载或欠驱动关节下会失真。
- **相关而非因果**:纯数据驱动,捕捉的是动作-场景演化的相关性;存在机器人不可控的外生因素时无法解耦真实因果。
- **细小物体与标定噪声**:细线/笔/线缆等 3D 标注困难,误配准会污染训练。
- **无显式物理先验**:不含牛顿力学/守恒律约束,是外插/极端情形的潜在软肋。

## 五、评价与展望

**优点**。(1) 表示统一是本文最锋利的思想:把 state 与 action 都压进 3D 点流,使动作直接以机器人物理几何为条件、天然跨本体,又把世界建模变成稳定的 L2/Huber 位移回归——避开了排列匹配与本体专属动作空间,这对"为规模化服务"是干净利落的选择,与用 3D flow 做接触推理的 Toolflownet/Im2Flow2Act 一脉相承但推到了大规模预训练。(2) 系统性的 scaling recipe(骨干、目标、特征、规模、数据混合、部分可观、chunked)做得扎实,近 log-linear 曲线为"3D 世界模型也遵循 scaling law"提供了迄今较有说服力的经验证据。(3) 数据标注管线(FoundationStereo+VGGT 精化+CoTracker3 抬升)本身是可复用的工程贡献,把 DROID 这类只有粗糙深度/外参的野外数据"3D 化",并承诺开源数据/权重。(4) 单一预训练 checkpoint 零样本驱动真实多类操作,是很强的定性结果。

**缺点与开放问题**。(1) 评测主要停留在 $\ell_2$ point-flow 精度,数值差异非常小(0.0386→0.0312,均约几厘米量级),作者也承认这类稠密度量与任务成功率的相关性并不总是明确;真实机器人成功率只有 8 个任务、单次统计、且推书 20%/微波炉 30% 偏低,统计力度有限,难判断相对已有 VLA/扩散策略的净优势。(2) 准静态、刚体、已实现动作三条假设叠加,意味着它更像"运动学响应模型"而非真正的动力学模型——强接触、柔性手、动态物体场景是硬伤,这也解释了为何目标代价需人工/VLM 指定任务点、且规划仍靠外部 MPPI。(3) 无光度动力学限制了它作为通用世界模型的适用面,与大视频世界模型(可生成外观但缺物理一致)恰好互补,如何融合两者是自然的下一步。(4) 与粒子/图式神经动力学(如 ParticleFormer、AdaptiGraph、Particle-Grid Neural Dynamics)相比,PointWorld 放弃了显式物理先验换规模,外插能力存疑;引入物理信息正则或混合仿真器可能兼得。(5) reward/cost 目前靠手工点选,自动化(VLM 指定或从演示逆强化学习)、以及把世界模型当环境做 RL 学参数化策略,是提升自主性的关键方向。总体上,这是一篇"用统一 3D 点流表示 + 大规模预训练把学习式动力学模型推向 in-the-wild"的高质量系统工作,思想清晰、证据扎实,但作为"世界模型"在动力学保真度与任务级评测上仍有明显留白。

## 参考

- Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.* RSS 2024.(真实数据与 3D 标注对象)
- Li et al. *BEHAVIOR-1K: A Human-Centered Embodied AI Benchmark.*(仿真双臂全身数据来源)
- Wu et al. *Point Transformer V3: Simpler, Faster, Stronger.* 2023.(骨干)
- Wen et al. *FoundationStereo: Zero-Shot Stereo Matching.* CVPR 2025;Wang et al. *VGGT: Visual Geometry Grounded Transformer.* CVPR 2025;Karaev et al. *CoTracker3.* ICCV 2025.(3D 标注管线三件套)
- Ai et al. *A Review of Learning-Based Dynamics Models for Robotic Manipulation.* Science Robotics 2025.(学习式动力学模型综述与基线出处)
