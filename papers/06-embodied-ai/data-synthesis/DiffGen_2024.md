# DiffGen：基于可微分物理仿真、可微分渲染与视觉语言模型的机器人演示生成

> **论文**：*DiffGen: Robot Demonstration Generation via Differentiable Physics Simulation, Differentiable Rendering, and Vision-Language Model*
>
> **作者**：Yang Jin、Jun Lv（共同一作）、Shuqiang Jiang、Cewu Lu（共同通讯/指导）
>
> **机构**：上海交通大学；中国科学院大学；中国科学院计算技术研究所
>
> **发布时间**：2024 年 05 月（arXiv 2405.07309）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2405.07309) | [PDF](https://arxiv.org/pdf/2405.07309)
>
> **分类标签**：`可微分物理仿真` `可微分渲染` `视觉语言模型` `演示数据生成` `梯度下降优化`

---

## 一句话总结

**DiffGen** 把可微分物理仿真（NimblePhysics）、可微分渲染（Redner）和预训练视觉语言模型（LIV）串成一条端到端可求梯度的链路，用"文本指令嵌入 vs. 渲染观测嵌入"的负余弦相似度作为损失,直接对动作序列做梯度下降,从而**不训练策略网络、不需要人工设计奖励函数**即可自动生成机器人操作演示;在 Cube-Selection 任务上仅用 2 万步优化即达到 0.85 的成功率,而对照的 PPO-LLM（Eureka 式奖励生成 + RL）用 500 万步才到 0.40,PPO-CosSim/PPO-InfoNCE 用 1000 万步成功率仅 0.06/0.00。

## 二、问题与动机

仿真中生成机器人演示数据分两步:场景构建和专家轨迹生成。场景构建已有 LLM/mesh 生成模型自动化的工作（RoboGen、GenSim、Gen2Sim 等）,但轨迹生成这一步,主流做法要么是训练强化学习策略 rollout（样本效率低、可解释性差）,要么是近年兴起的可微分仿真优化轨迹（gradSim、SAM-RL、Diff-LFD 等）——但后者仍严重依赖工程师手写奖励函数,劳动密集且难以规模化。

作者的核心洞察是:可微分物理仿真建模"动作→状态"的映射,可微分渲染建模"状态→观测"的映射,视觉语言模型（LIV）建模"观测→与语言指令的对齐程度",三者串联恰好覆盖了从任务指令到机器人控制信号的完整可微路径。只要每一环都可微,就能用梯度下降代替强化学习的采样式探索,把"设计奖励函数"的人力成本转移给通用预训练 VLM 的语义先验。

形式化目标(与标准目标条件 RL 一致的表述,但本文不用 RL 求解):

$$\max_{a_t} \; \mathbb{E}\!\left[\sum_{t=0}^{T} \gamma^t \mathcal{R}(s_t, a_t; \mathcal{G})\right], \quad \text{s.t. } s_{t+1} = \mathcal{T}(s_t, a_t)$$

大白话说:要找一串机械臂动作,让环境状态一步步演化后能达成由语言描述的目标 $\mathcal{G}$,DiffGen 的做法是把这个"找动作"问题变成一个可以直接求导、梯度下降求解的优化问题,而不是训练一个策略网络去试错。

## 三、核心方法

**流程（对应论文 Fig. 2）**:动作序列 → 可微分物理仿真前向推演状态 → 可微分渲染生成图像观测 → VLM 编码图像与指令 → 余弦相似度损失 → 梯度反传回动作序列 → 更新动作,循环直至收敛。

1. **可微分物理仿真**:$s_{t+1} = f(s_t, a_t)$,用 NimblePhysics 作为仿真后端——它把接触处理形式化为线性互补问题(LCP)求解,并能解析给出雅可比 $\partial s_{t+1}/\partial s_t$ 与 $\partial s_{t+1}/\partial a_t$。大白话说:仿真器不仅能"往前走一步",还能告诉你"这一步的结果对上一步的状态/动作有多敏感",这是后续梯度反传的基础。

2. **可微分渲染**:$I_t = g(s_t)$,用 Redner 的快速 deferred 渲染模式,把 NimblePhysics 输出的关节空间状态转换为各刚体的世界系变换矩阵（转换过程本身也保持可微,避免断梯度）,再渲染出 RGB 图像,同时可得 $\partial I/\partial s_t$。大白话说:把"物体现在摆成什么样"转成一张图片,并且知道图片每个像素对物体姿态变化有多敏感。

3. **视觉语言模型目标指定**:用预训练于 EpicKitchen（人类第一视角厨房视频,带自然语言标注）并在约 40 万条机器人数据上微调过的 LIV（ResNet-50 视觉编码器 + DistilBERT 语言编码器,权重从 CLIP 初始化;微调时只更新视觉编码器,学习率 1e-5,batch size 64,语言编码器冻结）分别编码指令和观测:

$$z^l = h(l), \qquad z^I = h(I)$$

损失为负余弦相似度:

$$\mathcal{L}(z^l, z^I) = -\frac{z^l \cdot z^I}{\lVert z^l\rVert \lVert z^I\rVert}$$

大白话说:让"渲染出来的画面"在 VLM 的语义空间里尽量贴近"指令这句话",用一个预训练好的通用跨模态相似度当"奖励",不用人再去写"夹爪离目标多远给多少分"这种规则。

4. **梯度下降生成动作**:借助链式法则把损失一路反传到动作序列:

$$\frac{\partial \mathcal{L}}{\partial \{a_t\}_{t=0}^{T}} = \frac{\partial \mathcal{L}}{\partial z^I}\frac{\partial z^I}{\partial I}\frac{\partial I}{\partial s_T}\frac{\partial s_T}{\partial \{a_t\}_{t=0}^{T}}$$

再用 AdamW（多数任务学习率 1e-2,Cup-Placing 因需精细操作降到 1e-3）做梯度下降更新动作。

5. **分段（episodic）优化策略**:直接对整条长轨迹反传梯度容易梯度消失/爆炸、陷入局部极小(误差沿状态转移序列累积)。作者的解法是把长时程轨迹切成若干短时程 episode,逐段优化——每段用更高学习率、更少优化步数快速探索,累积轨迹不断向后扩展,直到目标函数收敛到极值或达到迭代上限才停止。这是一个无理论收敛保证的启发式策略,但消融实验证明其对成功率和优化步数都有显著贡献(见下节)。

**任务设置**（均在 Franka Emika Panda 机械臂 + 方形平台上仿真,随机初始化末端位置）:
- **Cube-Selection**:平台上随机放置红/蓝/绿三个方块,按指令(如 "grasp the red cube")选中目标色方块,夹爪距目标方块 10cm 内判定成功;优化 200 步,固定时程 50 步。
- **Cup-Placing**:机械臂已抓握一个杯子,需按 "put the cup on the dish" 把杯子放到随机位置的碟子上,杯底距碟面 5cm 内判定成功;优化 200 步,时程 70 步,学习率降为 1e-3 以保证稳定性。
- **Obstacle-Crossing**:一堵 30cm×40cm×10cm 的墙作为障碍物,按指令("move to the left/right side of the wall" 或 "move to the back of the wall")把方块绕过障碍送到 40cm×30cm 的目标区域;优化 300 步,时程 100 步。

## 三、关键结果

**与 RL 基线对比（Cube-Selection 任务,100 次随机初始化环境取均值）**:基线包括 VLM-reward 驱动 PPO(参照 RoboCLIP 思路,把 S3D 换成微调后的 LIV,分别用余弦相似度 PPO-CosSim 和 InfoNCE 损失 PPO-InfoNCE 作奖励)以及 LLM 生成奖励代码驱动 PPO(参照 Eureka 的 prompt 设计和奖励反思迭代,PPO-LLM 跑三轮迭代)。

| 方法 | 成功率 | 偏差 | 总优化/训练步数 |
|---|---|---|---|
| Ours(best traj.) | 0.85 | 0.082 | 20k |
| Ours(last traj.) | 0.81 | 0.095 | 20k |
| PPO-CosSim | 0.06 | 0.446 | 10M |
| PPO-InfoNCE | 0.00 | 0.259 | 10M |
| PPO-LLM(第 1 轮) | 0.25 | 0.274 | 5M |
| PPO-LLM(第 2 轮) | 0.37 | 0.173 | 5M |
| PPO-LLM(第 3 轮) | 0.40 | 0.224 | 5M |

即便让 LLM 迭代三轮打磨奖励函数,PPO-LLM 用 500 万步也只能到 0.40 成功率,不到 DiffGen 用 2 万步(约 250 倍更少交互)所达到的 0.85。直接把 VLM 相似度当奖励喂给 PPO(PPO-CosSim/InfoNCE)则几乎学不到有效策略,说明该相似度信号本身对无梯度的 RL 采样并不友好,但对梯度下降却是有效的优化信号。

**零样本目标指定（Obstacle-Crossing,VLM 仅在人类 EpicKitchen 视频上预训练、未做任何机器人数据微调）**:

| 指令 | 成功率 | 位移 | 最终位置 |
|---|---|---|---|
| Move Left | 0.978 | -0.205 | -0.217 |
| Move Right | 0.704 | 0.049 | 0.050 |

模型仅从人类视频学到的"左/右"语义就能迁移指导机器人轨迹优化,验证了跨人类数据到机器人任务的可迁移性(尽管方向不对称,"move right" 成功率明显低于 "move left")。

**跨具身与场景泛化（Cup-Placing,VLM 用 Franka 数据微调后直接测 UR5e,以及测未见过的杯子颜色）**:

| 场景 | 成功率 | 偏差 |
|---|---|---|
| 同具身,已见颜色(基线) | 0.87 | 0.0287 |
| 同具身,未见颜色 | 0.53 | 0.0553 |
| 新具身(UR5e),已见颜色 | 0.84 | 0.0535 |

跨具身迁移的性能损失(0.87→0.84)小于颜色泛化的损失(0.87→0.53),说明模型学到的更多是"put on"这类与具身无关的语义关系,而对物体表观变化(颜色)更敏感。

**消融（Obstacle-Crossing）**:

| 方法 | 成功率 | 总优化步数 |
|---|---|---|
| Ours | 0.72 | 23.9k |
| Ours(w/o episodic optimization) | 0.41 | 30k |
| Handcraft(w/o 可微渲染,用特权信息手写奖励) | 0.19 | 30k |

去掉分段优化策略、退化为定时程单次优化,成功率从 0.72 掉到 0.41(步数还更多);去掉可微渲染 + VLM、改用基于特权信息的手写奖励,成功率进一步掉到 0.19,说明 VLM 提供的语义反馈比手写奖励更能帮助跳出局部极小。

## 四、评价与展望

**优点**:DiffGen 的核心贡献是把"可微分仿真+可微分渲染"这条此前主要用于系统辨识/参数估计(gradSim、RISP)或仍需手写奖励的轨迹优化(SAM-RL、SAGCI、Diff-LFD)的技术路线,与预训练跨模态奖励模型(LIV)结合,首次让"语言指令→动作序列"这条链路整体可微、可端到端梯度下降,从而摆脱了 RL rollout 的采样低效和奖励工程的人力成本。与同样想省掉手工奖励的两条路线相比,它比"VLM-reward + RL"(RoboCLIP 一路)采样效率高出两个数量级以上,也比"LLM 生成奖励代码 + RL"(Eureka 一路)省去了对大模型代码生成能力和奖励反思迭代的依赖。零样本目标指定和跨具身实验也初步表明,VLM 承载的语义先验确实能部分替代人工奖励设计所依赖的领域知识。

**局限性(原文 Section V 及可观察到的实验设计局限)**:作者明确指出,方法性能受限于所用 VLM(LIV)的质量,当前仍需用无动作标注的机器人视频微调以缩小人类数据预训练与机器人任务间的域差,论文认为这一微调需求有望随更强 VLM 的出现而减少。此外从实验设计看:(1)仅在 3 个高度简化的刚体任务(选方块、放杯子、绕障碍)上验证,均在仿真中完成,未做真实机器人部署或 sim-to-real 验证;(2)底层物理引擎 NimblePhysics 基于 LCP 求解刚体接触,天然更适合刚体、低自由度场景,论文未展示该框架在可变形物体、多物体复杂接触等场景下的适用性(相较之下 DexDeform、DiffSkill 等工作专门处理可变形物体的可微分仿真);(3)分段(episodic)优化策略是缓解长时程梯度爆炸/局部极小的启发式方案,没有收敛性保证,其超参数(每段步数、学习率切换)依赖任务复杂度手工设定;(4)成功判据均基于简单的欧氏距离阈值(如 10cm/5cm),对更复杂的多步骤、序列化任务(如需要顺序满足多个子目标)的适用性未被验证。

**开放问题与可能方向**:一是将该范式扩展到可变形物体、绳索、布料等接触更复杂的场景,需要更鲁棒的可微分仿真器;二是分段优化策略若能替换为更有理论支撑的信赖域或课程式(curriculum)时程调度,可能进一步提升长时程任务的稳定性;三是当前每条轨迹都要独立跑一次梯度下降优化(不产出可复用的策略网络),后续工作可以探索把优化得到的轨迹反哺蒸馏成策略网络(类似把 DiffGen 当作数据引擎,下游用模仿学习训练策略),从而真正服务于"scale up 机器人预训练数据"这一在引言中被强调的最终目标;四是零样本方向实验中 "move right" 明显弱于 "move left" 的不对称现象,提示 VLM 对空间关系语言的编码可能存在系统性偏置,值得进一步分析。

## 参考

- Ma et al. *LIV: Language-Image Representations and Rewards for Robotic Control*. ICML 2023.(本文使用的核心 VLM 奖励/表征模型)
- Werling et al. *Fast and Feature-Complete Differentiable Physics for Articulated Rigid Bodies with Contact* (NimblePhysics). arXiv:2103.16021.(可微分物理仿真后端)
- Li et al. *Differentiable Monte Carlo Ray Tracing through Edge Sampling* (Redner). SIGGRAPH Asia 2018.(可微分渲染器)
- Sontakke et al. *RoboCLIP: One Demonstration is Enough to Learn Robot Policies*. NeurIPS 2023.(VLM-reward + RL 对照基线来源)
- Ma et al. *Eureka: Human-Level Reward Design via Coding Large Language Models*. ICLR 2024.(LLM 生成奖励代码 + RL 对照基线来源)
