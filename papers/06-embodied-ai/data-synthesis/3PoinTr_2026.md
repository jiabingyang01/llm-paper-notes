# 3PoinTr：用于从无约束人类视频中学习操作的 3D 点轨迹

> **论文**：*3PoinTr: 3D Point Tracks for Learning Manipulation from Unconstrained Human Videos*
>
> **作者**：Adam Hung, Bardienus Pieter Duisterhof, Jeffrey Ichnowski
>
> **机构**：Carnegie Mellon University（Pittsburgh, Pennsylvania）
>
> **发布时间**：2026 年 3 月（arXiv 2603.08485，v2 为 2026 年 6 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.08485) | [PDF](https://arxiv.org/pdf/2603.08485)
>
> **分类标签**：`learning-from-human-video` `3D-point-track` `sample-efficient-BC`

---

## 一句话总结

3PoinTr 用一个 visibility-aware transformer 从**去除人手/机械臂之后** 的场景点云预测未来 dense 3D 点轨迹，再用 Perceiver-IO 把这些轨迹压成少量 token、以 global FiLM + 逐块残差 cross-attention 两路注入 flow-matching 动作头，训练闭环策略；由此可以直接从**无约束**（人可以随意抓取、不必模仿机器人运动学）的人类视频里预训练操作先验。仅用每任务 20 条带动作机器人 demo，真实任务平均成功率比最强 baseline（DP3）高 **25.0** 个百分点、仿真高 **29.6** 个百分点。

## 一、问题与动机

从人类视频学操作能大幅降低对昂贵遥操作数据的依赖，但存在两大痛点：

1. **Embodiment gap（本体鸿沟）**：人手与并爪夹持器运动学差异巨大，人类视频里的抓取姿态、轨迹对机器人往往不可行、低效甚至危险。为了缩小这个鸿沟，已有 3D 方法（如 General Flow、Point Policy）通常依赖被"编排过"的人类演示——要求演示者的手尽量模仿并爪夹持器（如捏成爪形），或依赖人到机器人的固定 keypoint 映射、人工标注、预定义抓取点。这类约束限制了数据规模。
2. **2D vs 3D 与开环 vs 闭环**：多数 point-track 方法停留在 2D，而 3D 状态表征已被证明更省样本；同时"预测开环轨迹并直接执行"缺乏对执行误差/扰动的反应性，而"每步重新预测轨迹"的闭环方法又会遭遇人到机器人的 distribution shift，且预处理时被遮挡的点被直接剔除、丢失监督。

3PoinTr 的目标是回答一个问题：*完成这个任务时，3D 场景会如何演化？* 它对场景点云预测 dense 3D 点轨迹（本体点已移除），让**学到的 query 自主发现任务相关特征**，从而在保留闭环反应性的同时避免上述偏置与分布偏移，让人类演示者可以用任何自然的抓取、手势、轨迹。

## 二、核心方法

整体分两阶段：先训练一个"从单帧场景观测预测未来 dense 3D 点轨迹"的模型，再训练一个以这些轨迹为条件的闭环策略。

### 2.1 阶段一：3D 点轨迹预测

给定初始点云 $P_0 \in \mathbb{R}^{N\times 3}$（相机坐标系下 $N$ 个可见场景点，**所有本体点已删除**），学习

$$f_\theta: P_0 \mapsto X, \qquad X = \{\, X^i \mid i \in [0,N),\ t \in [0,T) \,\} \in \mathbb{R}^{N\times T\times 3}$$

即每个初始点在未来固定 horizon $T$ 步内的 3D 位置轨迹。网络用 3 层 MLP（输出维 256、GELU）把可见非本体点 token 化，过 2 个标准 self-attention decoder block（4 头、head dim 64、FFN expansion 4），再用线性头输出每点在整段 horizon 上的 3D 轨迹。训练时随机下采样 2048 点、加 $\sigma=0.01$ 高斯噪声做增强。

关键设计是 **visibility-aware 训练**：假设可拿到每点每步的可见性 $m^i_t \in \{0,1\}$，损失只对可见的 point-timestep 对计算 L1 的 3D 位置误差：

$$\mathcal{L} = \sum_{i,t} m^i_t \,\big\| \hat{X}^i_t - X^i_t \big\|_1$$

> **用大白话说**：General Flow 这类方法在预处理时会把"中途被遮挡的整条轨迹"直接扔掉，于是像"抛纸"这种物体被手/夹爪挡住的关键运动就完全没有监督。3PoinTr 反过来——轨迹全保留，只是对"这一帧看不见的那些点"这一格不算损失，从而对被短暂遮挡的**任务关键物体点** 仍保留监督。

### 2.2 阶段二：轨迹条件闭环策略

冻结上面的轨迹预测器。给定预测轨迹 $X$ 与当前观测 $o_t=(P_t,q_t)$（$P_t$ 为去本体的当前可见点云、$q_t$ 为机器人构型），学习

$$\pi_\phi:(X, o_t)\mapsto A, \qquad A \in \mathbb{R}^{H\times D_A}$$

预测 $H$ 步动作 chunk。轨迹 $X$ **在 episode 开头由初始观测 $P_0$ 算一次并全程复用**（充当开环 plan），而 $o_t$ 提供闭环信号：每执行完前 $H_{\text{exec}}\le H$ 步就用更新后的 $o_t$ 重新查询策略。

> **用大白话说**：把"怎么完成任务"（物体该往哪动）一次性算好当作路线图（开环，避免逐步重预测带来的分布偏移与遮挡问题），再让策略每一步看当前场景实时纠偏（闭环，保留反应性）。这是本文对"开环 vs 闭环"两难的核心折中。

**轨迹 token 压缩（Perceiver-IO）**：每条预测轨迹 $X^i\in\mathbb{R}^{T\times 3}$ 先嵌成 token $F^i\in\mathbb{R}^{D_F}$，得 $F\in\mathbb{R}^{N\times D_F}$；一组 $M=4$ 个可学习 query $Q\in\mathbb{R}^{M\times D_F}$ 对 $F$ 做 cross-attention，输出紧凑的轨迹 token $Z\in\mathbb{R}^{M\times D_F}$。这里**不引入人到机器人 keypoint 映射或物体特定特征**，让 query 自主挑出任务相关的运动。

**两路条件注入**：

- **全局 FiLM 条件**：当前点云 $P_t$ 经 DP3 式 PointNet 编为 64 维 $f_{\text{pc}}$，机器人构型 $q_t$ 经 MLP 编为 32 维 $f_{\text{conf}}$，与展平投影后的轨迹 token 特征 $f_{\text{track}}$（128 维）拼接，再经浅层 MLP $\psi$ 得全局条件向量

$$c = \psi\big(f_{\text{track}}, f_{\text{pc}}, f_{\text{conf}}\big) \in \mathbb{R}^{D_c}$$

- **残差轨迹 token cross-attention**：在 flow-matching 动作头（Chi et al. 的条件 1D U-Net）的每个残差块（下采样、中间、上采样路径）里插入残差 cross-attention adapter——U-Net 特征作 query、轨迹 token $Z$ 作 key/value，输出加回 U-Net 且**零初始化**，故训练之初退化为纯 FiLM 模型。

> **用大白话说**：光靠一个全局向量喂条件，动作头容易只盯着原始点云、忽视预测的物体运动。逐块 cross-attention 给动作头一条"直连预测物体运动"的通路——每个动作步都能去查它最相关的那几个运动 token，从而更简单地把"物体该怎么动"映射成"机器人该怎么动"。

动作头用 conditional flow-matching 输出 $A\in\mathbb{R}^{H\times D_A}$，$D_A=10$（3 维末端位置 + 6 维连续旋转表示 + 1 维夹爪），全部在数据采集/轨迹预测所用的相机坐标系下表达。

### 2.3 数据获取

- **仿真**：UFactory xArm 7 + 并爪，程序化生成 demo；靠跟踪刚体位姿得到真值 3D 点轨迹（机器人点排除）。
- **真实**：Xbox 手柄遥操作采机器人 demo；人做**无约束**演示视频。ZED Mini 相机、eye-to-hand 标定、静止相机。3D 轨迹提取管线：2D 点跟踪（CoTracker3）得 2D 轨迹 + 可见性 → stereo-to-depth（FoundationStereo）得深度 → 把 2D 轨迹 lift 到 3D；用 SAM3 以文本 prompt "human arm" 分割本体点并标为不可见。轨迹按归一化时间重采样成定长离散：粗轨迹 $\sim$10–16 步作预测器训练目标，细轨迹 $\sim$30–50 步作 BC 训练目标。真实任务中轨迹预测网络**只在人类视频上训练**（作者发现这样给策略的条件更一致）。

## 三、实验结果

评测覆盖仿真 4 任务（Block Stack / Right Glass / Pot Lid / Open Microwave）与真实 4 任务（Open Drawer / Right Glass / Throw Away Paper / Fold Sock）。真实任务的机器人与人类演示仅共享初/末状态，夹爪与人手的朝向、抓取点、轨迹均不同（如 Right Glass 任务人抓杯身、机器人则把夹爪插入杯内抬起杯沿使其绕底旋转）。

### 3.1 3D 点轨迹预测质量（真实任务，ADE / 5% ADE，单位 mm，越低越好）

5% ADE 指"运动幅度最大的 5% 点"的误差，用以反映任务关键运动（总 ADE 被静止背景点主导）。

| 任务 | GF ADE | GF 5% ADE | 3PoinTr ADE | 3PoinTr 5% ADE |
|---|---|---|---|---|
| Open Drawer | 2.54 | 16.37 | **2.18** | **13.19** |
| Right Glass | 3.34 | 30.38 | **2.29** | **18.55** |
| Throw Away Paper | 2.40 | 22.87 | **1.47** | **8.17** |
| Fold Sock | 1.56 | 10.09 | **1.14** | **4.68** |

3PoinTr 在每个真实任务的两个指标上都胜过 General Flow，平均误差降低 **28%**（ADE）与 **44%**（5% ADE）。优势主要来自它能从 General Flow 忽略的数据里学习：真实点常被短暂遮挡，General Flow 预处理时剔除含不可见点的轨迹，而 3PoinTr 保留全部轨迹、仅按可见性 mask 损失（如 Throw Away Paper 里纸张到末帧已全被遮挡，General Flow 对纸张运动毫无监督）。附录中同一份数据、所有点全程可见的仿真设置下，3PoinTr 用单 transformer decoder 架构就能与 General Flow 更复杂的 PointNeXt encoder-decoder + 条件 VAE 打成平手。

### 3.2 仿真策略成功率（%，每任务 200 rollout，20 demos + 100 无动作视频）

| 任务 | AMPLIFY | DP | ATM | DP3 | 3PoinTr |
|---|---|---|---|---|---|
| Block Stack | 0.5 | 17.0 | 11.0 | 25.5 | **66.0** |
| Right Glass | 2.0 | 16.5 | 8.0 | 21.0 | **51.0** |
| Pot Lid | 3.0 | 42.5 | 27.5 | 51.5 | **67.5** |
| Open Microwave | 7.0 | 17.5 | 70.0 | 43.0 | **75.0** |
| **平均** | 3.1 | 23.4 | 29.1 | 35.3 | **64.9** |

20 demos 下 3PoinTr 在**每个仿真任务** 上均最高，平均比最强 baseline DP3 高 **29.6** 个百分点。随 demo 增多仍领先：3PoinTr 在 20/50/100 demos 上分别为 Block Stack 66.0/84.5/94.0、Right Glass 51.0/77.0/93.0、Pot Lid 67.5/91.0/93.0、Open Microwave 75.0/78.0/89.0，50 与 100 demos 下平均成功率均最高。

### 3.3 真实策略成功率（20 机器人 demo + 50 人类视频/任务）

| 任务 | ATM | DP3 | 3PoinTr |
|---|---|---|---|
| Open Drawer | 6/20 | 14/20 | **20/20** |
| Right Glass | 3/20 | 18/20 | **20/20** |
| Throw Away Paper | 0/20 | 9/20 | **18/20** |
| Fold Sock | 7/20 | 14/20 | **17/20** |
| **平均** | 20.0% | 68.75% | **93.75%** |

3PoinTr 在全部四个真实任务上均最高，平均比最强 baseline DP3 高 **25.0** 个百分点。分析指出：因机器人点占据场景很大一部分，ATM/AMPLIFY 这类保留本体点的方法会过度依赖机器人轨迹预测、对底层场景任务编码更少，且 ATM 逐步重预测在真实中还额外遭受人到机器人分布偏移与遮挡（Throw Away Paper 里夹爪抓取后几乎遮住整个物体，ATM 得 0%，而 3PoinTr 只在初始预测一次、被遮点仍留在条件里，得到 DP3 两倍的成功率）。

### 3.4 消融（仿真 20 demos + 100 视频，平均成功率）

| 消融变体 | 平均成功率 |
|---|---|
| No Perceiver-IO（改为轨迹 token mean pooling） | 33.5 |
| 2D Point Tracks（用 2D 轨迹 + DP 图像 encoder 替代 3D） | 35.8 |
| No Extra Videos（仅用 20 条机器人 demo，无额外视频） | 51.5 |
| No U-Net Xattn（去掉残差轨迹 token cross-attention） | 57.4 |
| **3PoinTr（完整）** | **64.9** |

结论：Perceiver-IO 编码与"用 3D（而非 2D）轨迹"影响最大；No Extra Videos 表现"意外地好"，说明**学习预测 3D 点轨迹本身** 就是一个有用的归纳偏置（不只是为了能吃无动作视频），但接入额外无动作数据后完整模型仍最高；去掉 U-Net 内残差 cross-attention 会在所有任务上掉点。

## 四、局限性

- **仍限于实验室一致条件**：当前实验为获得可控的样本效率，依赖相对一致的实验室拍摄条件；把预训练与 BC 数据都扩到互联网规模、跨视角/环境/更大任务变化泛化仍是未解方向。
- **只用点轨迹作条件**：作者认为 3D 点轨迹已能完整表达任务规范，故放弃彩色图像、文本 prompt 等额外模态；但承认某些任务组合可能仍需这些特征。
- **固定均匀时间下采样**：BC 训练时按均匀时间抽状态-动作对，未把时间分辨率集中到动态接触/运动段；自适应选择是未来方向。
- **依赖精确 2D 点跟踪**：数据采集期依赖 2D 点跟踪，在透明物体、反光物体、阴影下会失败——这是整条 3D 轨迹管线的上游软肋。
- 真实评测每任务规模较小（20 rollout 计成功率），结论的统计置信度有限。

## 五、评价与展望

**优点**。(1) 真正切中"无约束人类视频"这一稀缺设定：相较需要人手模仿并爪、固定 keypoint 映射或人工标注的 General Flow / Point Policy / Motion Tracks，本文让演示者自由行动，data collection 门槛显著降低，是把该方向推向 in-the-wild 的实际一步。(2) "开环一次性轨迹 plan + 闭环策略纠偏"的折中优雅地同时回避了直接执行开环轨迹的无反应性、与 ATM/Im2Flow2Act 逐步重预测的分布偏移和遮挡问题，且在真实 Throw Away Paper 上以 18/20 对 ATM 0/20 给出了强有力的对照。(3) visibility-aware 训练（保留轨迹、按可见性 mask 损失）是相对 General Flow"剔除被遮轨迹"的一个简单但关键的改进，5% ADE 降 44% 印证了对被遮任务关键点保留监督的价值。(4) 去本体点 + Perceiver-IO 让学习到的 query 自主发现任务特征，消融显示这两点是主要增益来源，思路干净。

**缺点与开放问题**。(1) 泛化范围保守：只有 4+4 个任务、单机位静止相机、实验室条件，尚未验证跨视角/场景与真正 in-the-wild 视频，而这恰是"从人类视频学操作"的核心卖点；作者自陈这是最大局限。(2) 整条真实管线串联 CoTracker3 + FoundationStereo + SAM3 三个外部大模型，误差会级联，且透明/反光/阴影场景直接失效，鲁棒性存疑。(3) 开环轨迹"算一次全程复用"对长程、需要多次重规划或接触状态频繁切换的任务可能不够——一次性 plan 何时该刷新是明显的开放问题。(4) 只用点轨迹、丢弃颜色/语义/语言，虽在当前任务够用，但对语义歧义任务（"拿红色那个"）会失去可指令性，与当前 VLA 主流方向存在张力。(5) 与生成式方法（Dream2Flow、NovaFlow 等从生成视频提 actionable flow）相比，本文走的是"预测判别式轨迹"路线，二者在数据来源与可扩展性上的取舍值得后续系统比较。

**展望**。最自然的下一步是把预测器扩到互联网规模人类视频并验证跨域泛化；引入自适应时间采样以覆盖接触密集段；在轨迹条件之外可选地融合语言/图像以恢复可指令性；以及把 visibility-aware 监督与生成式 flow 先验结合，进一步榨取被遮区域的信号。

## 参考

1. C. Yuan, C. Wen, T. Zhang, Y. Gao. *General Flow as Foundation Affordance for Scalable Robot Learning.*（本文轨迹预测主对比 baseline [12]）
2. C. Wen, X. Lin, J. So, K. Chen, Q. Dou, Y. Gao, P. Abbeel. *Any-point Trajectory Modeling for Policy Learning.* arXiv:2401.00025（ATM，2D 逐步重预测轨迹条件策略 [4]）
3. J. A. Collins et al. *AMPLIFY: Actionless Motion Priors for Robot Learning from Videos.* arXiv:2506.14198（潜空间视频动力学 + 逆动力学 baseline [6]）
4. Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, H. Xu. *3D Diffusion Policy.* arXiv:2403.03954（DP3，点云条件扩散策略，最强 BC baseline [17]）
5. A. Jaegle et al. *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* arXiv:2107.14795（轨迹 token 压缩用的编码器 [3]）
