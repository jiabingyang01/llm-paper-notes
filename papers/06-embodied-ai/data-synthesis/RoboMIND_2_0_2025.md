# RoboMIND 2.0：面向可泛化具身智能的多模态双臂移动操作数据集

> **论文**：*RoboMIND 2.0: A Multimodal, Bimanual Mobile Manipulation Dataset for Generalizable Embodied Intelligence*
>
> **作者**：Chengkai Hou, Kun Wu, Jiaming Liu（三位共同一作/核心）, Zhengping Che（项目负责人）, Di Wu, Fei Liao, ..., Shanghang Zhang（通讯）, Jian Tang（通讯）et al.
>
> **机构**：北京人形机器人创新中心（Beijing Innovation Center of Humanoid Robotics）、北京大学计算机学院多媒体信息处理国家重点实验室（State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University）
>
> **发布时间**：2025 年 12 月（arXiv 2512.24653，v3 于 2026-02-27）
>
> **发表状态**：未录用（预印本；前作 RoboMIND 1.0 为 RSS 2025）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.24653) | [PDF](https://arxiv.org/pdf/2512.24653)
>
> **分类标签**：`双臂操作数据集` `移动操作` `触觉多模态` `数字孪生/Sim-to-Real` `离线 RL 后训练`

---

## 一句话总结

RoboMIND 2.0 是一个以"标准化遥操作 + 严格质检"为主线采集的大规模真机双臂数据集：跨 6 种异构本体（Franka、UR5e、AgileX、ARX、天工 Tien Kung、天轶 Tian Yi）收集约 **310K** 条双臂轨迹、覆盖 **759** 个任务 / **129** 类技能 / **1139** 种物体、累计 1000+ 小时，其中含 **20K** 条移动操作轨迹、**12K** 条带触觉的序列，并额外开源 **20K** 条与真实任务一一对齐的 Isaac Sim 数字孪生轨迹；论文同时提出双系统控制器 **MIND-2**（慢规划 VLM + 快执行 VLA，后者用 IQL 离线 RL 后训练），在 AgileX 移动操作 4 任务上把最优 VLA 基线（XR-1)的 0.4/0.2/0.4/0.3 提升到 0.5/0.8/0.4/0.7，并验证了触觉融合、数字孪生混训、跨物体泛化三项收益。

## 一、问题与动机

数据驱动的模仿学习已重塑机器人操作,但真机演示数据仍是瓶颈,尤其在**长程双臂协作**与**陌生环境下的移动操作**两类任务上泛化不足。作者梳理了现有数据集的三类结构性缺陷：

1. **维度单一**。Open X-Embodiment、RH20T、DROID、RoboMIND 1.0 等主流集合大多是单臂、固定基座,缺乏双臂协调样本;AgiBot World、Galaxea Open-World 虽引入了丰富双臂与高分辨率触觉,却几乎只绑定单一(人形)本体,难以研究跨本体泛化。RoboCOIN 用多平台扩了本体多样性并带数字孪生,但每本体任务覆盖稀疏,不利于训练长程时序策略。

2. **触觉缺失**。几乎所有大规模基准只记录视觉观测与基础本体状态,忽略了对接触、打滑、精细操作至关重要的触觉反馈。

3. **仿真资产不开放**。多数数据集只放真机数据,不提供可复现的数字孪生资产,阻碍低成本仿真扩数据。

RoboMIND 2.0 的目标是"一次性"在**本体形态、环境、任务语义、失败模式、多模态传感**五个维度上同时覆盖,并把它定位为一个可同时评测单任务模仿学习与多任务 VLA 的综合基准。

## 二、核心方法

本文有两条主线：**(A) 数据集构建**（贡献主体）与 **(B) MIND-2 双系统模型**（用来验证长程移动操作数据的价值）。

### A. 数据集：标准化采集 + 十二类失败模式质检 + 分层语言标注

**跨本体遥操作。** 六个本体用三类互补的人在环接口采集，以刻意制造"遥操作模态多样性"：

- **HACTS 主从系统**（低成本、Dynamixel 舵机 + 3D 打印 PLA 框架 + 脚踏切换自主/人控）用于 Franka、UR5e、天工，实现关节级双边同步。
- **VR 遥操作**用于 ARX：头显捕捉上肢运动、VR 手柄指令移动底盘。
- **物理引导（推底盘)**记录 AgileX、天轶移动底盘的线速度/角速度；天工人形另用动捕服记录关节。

数据统一存为 HDF5(多视角 RGB-D + 本体状态 + 末端状态 + 遥操作者身体状态；AgileX 额外含左右臂触觉)。运动学上既有 Franka/UR5e 平行对称布局，也有肩部偏置不同的"类人"非对称布局与天工/天轶全人形上身，制造跨臂泛化难度。

**十二类失败模式质检。** 每条轨迹按 12 条质量准则逐一核查（Unintended Contact、Unsmooth Motion、Re-grasping、Pre-grasp Collision、Data Anomalies、Failed Placement、Trajectory Abnormality、Excessive Speed、Visual Artifacts、Robot Arm Abnormality、Non-standard Operation、Target Displacement），走三阶段流程：初检(扫大问题)→细检(逐帧看 12 类失败)→过滤与记录(标时间戳+描述，决定丢弃/修正/重采)。

**分层语言标注。** 对每条移动操作轨迹按"导航↔操作"的语义边界切成子任务段：导航动作定义为 *"Go to [location]"* / *"Stop in front of [object]"*，两次导航之间的所有操作合并为一个子任务单元。标注由 **Gemini 2.5 Pro** 自动生成(用人工标注样例作 few-shot 示范)，动作类型打上 *Self / Others / Move* 标签，供双系统 VLA 训练。

**触觉与数字孪生。** 触觉用 Tashan 传感器，AgileX 平行夹爪每指一枚高分辨率触觉、每枚含两个感知模块，实时记录法向力、切向力及切向力方向。仿真资产遵循 ArtVIP 规范做成 Assembly–Module–Mesh 三级结构，在 Isaac Sim 用 RTX PBR 渲染，凸包+凸分解+精细碰撞网格保证物理精度，关节引入位置相关刚度+速度相关摩擦并以 0.1 mm 光学动捕校验；据此在 Franka 双臂与天工人形上采集 20K 条与真实任务结构、语言、物体配置一致的仿真轨迹。

### B. MIND-2：慢 VLM 规划 + 快 VLA 执行（IQL 离线后训练）

**MIND-2-VLM（慢脑）** 由 InternVL3-8B 微调而来，做时序任务定位。输入含三部分：多视角视觉(front/left-wrist/right-wrist 三个 `<image>` token)、任务上下文(该 episode 全部子任务枚举列表)、执行上下文(7-DoF 臂关节位置 + 底盘 twist 速度 + 上一步推断的任务索引)。监督输出为确定性字符串 `Task Index: xx / Task Progress: xx`，其中进度按段内线性插值：

$$\text{progress} = \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} \in [0,1]$$

以交叉熵端到端训练。严格的输出格式保证下游策略可直接解析。

**用大白话说**：慢脑不直接控制机器人,只回答两句话——"现在做到第几个子任务了""这个子任务干到了百分之几",据此决定何时把下一条高层指令交给快脑。

**MIND-2-VLA（快脑）** 用 **Implicit Q-Learning(IQL)** 做离线 RL 后训练——因为遥操作会天然产生大量失败数据,这些失败并非无用,而能教模型"避坑"。先给所有轨迹赋带折扣的稠密回报：成功轨迹末步 +1、前 $t$ 步 $r_t=\gamma^{T-t}$($\gamma=0.999$);失败轨迹末步 $-1$、前步 $r_t=-\gamma^{T-t}$,使通向失败的动作获得越来越负的回报($T$ 为轨迹长度)。

IQL 只在固定数据集 $\mathcal{D}=\{(s,a,r,s')\}$ 上学三件套。先用非对称分位数回归学状态值函数(避免对 OOD 动作过度乐观)：

$$\mathcal{L}_V(\phi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\rho_\tau\!\left(Q_{\text{target}}(s,a) - V_\phi(s)\right)\right]$$

其中非对称分位数损失 $\rho_\tau(u) = u\cdot\bigl(\tau - \mathbb{1}(u<0)\bigr)$,$\tau\in(0,1)$。再用标准 TD 更新 Q：

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}}\left[\tfrac{1}{2}\Bigl(Q_\theta(s,a) - \bigl(r(s,a) + \gamma^{rl} V(s')\bigr)\Bigr)^2\right]$$

最后用优势加权回归(AWR)更新策略,只强化优于该状态平均值的动作：

$$\mathcal{L}_\pi(\psi) = -\mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\exp\!\bigl(\beta\cdot A(s,a)\bigr)\cdot\log\pi_\psi(a\mid s)\right]$$

其中隐式优势 $A(s,a)=Q(s,a)-V(s)$,$\beta>0$ 为温度。

**用大白话说**：IQL 全程不与环境交互、只从"混质量"的成功+失败数据里学。它通过对比高回报与低回报序列，学会区分"看起来合理但会失败(如虚握、错位)"和"真正能成功"的动作，让快脑不只是模仿成功、还主动规避已知失败模式。

## 三、实验结果

评测覆盖 6 本体、38 个真机任务(每任务测 10 次记成功率)、4 个单任务模仿学习方法(ACT、Dense Policy、DP3、UVA)与 4 个 VLA(π0、π0.5、HybridVLA、XR-1)。

**基准总览(Table 2 要点)。** 3D 方法(DP3、Dense Policy)整体优于 2D(ACT、UVA)，印证双臂协调需要更强空间建模;DP3 在固定臂 FR/UR 上强、但在移动/人形本体上骤降，Dense Policy 跨本体更稳。VLA 中 **XR-1 跨本体泛化最强**(Franka、天工多任务达 0.7–0.8)，π0 偏弱(AgileX-MV、天轶近乎全 0)，π0.5 较 π0 有提升(UR-Task1 达 0.8、AgileX-Task1 达 0.6);HybridVLA 与 π0 成绩相近，说明在同等数据/视觉编码器下，动作生成头(flow / diffusion / 自回归)本身不是决定性因素，数据多样性与长程监督更关键。

**MIND-2 在 AgileX 移动操作(Table 3)。**

| 方法 | Task1 | Task2 | Task3 | Task4 |
|---|---|---|---|---|
| π0 | 0.1 | 0.0 | 0.0 | 0.1 |
| π0.5 | 0.3 | 0.0 | 0.3 | 0.1 |
| XR-1 | 0.4 | 0.2 | 0.4 | 0.3 |
| **MIND-2** | **0.5** | **0.8** | **0.4** | **0.7** |

**多机协作长程任务(Table 4，超市/工业/化学实验室三场景，训练用 200 条成功轨迹)。**

| 变体 | 超市 | 工业 | 化学实验室 |
|---|---|---|---|
| MIND-2(Post Training) | 0.6 | 0.6 | 0.4 |
| MIND-2(Full-scale Training) | 0.8 | 0.7 | 0.4 |
| **MIND-2(Offline RL)** | **0.9** | **0.8** | **0.6** |

即：全量移动操作数据预训练带来一档提升，再叠加 IQL 离线 RL 后训练又拿到一档，验证长程移动数据与离线 RL 的双重价值。

**成功:失败配比消融(Table 5)。** IQL 中失败轨迹并非越多越好，成败 **1:1** 时最佳(三任务达 1.0 / 1.0 / 0.8)，2:1、1:2 均略逊，说明适度而非海量的失败样本最利于"避坑"。

**触觉融合(Table 6，AgileX-MV 4 任务)。** 把触觉并入本体输入后 π0.5 与 XR-1 均涨，XR-1 收益更明显：

| 模型 | 触觉 | Task1 | Task2 | Task3 | Task4 |
|---|---|---|---|---|---|
| π0.5 | ✗ | 0.3 | 0.0 | 0.3 | 0.1 |
| π0.5 | ✓ | 0.4 | 0.1 | 0.5 | 0.2 |
| XR-1 | ✗ | 0.4 | 0.2 | 0.4 | 0.3 |
| **XR-1** | ✓ | **0.6** | **0.4** | **0.6** | **0.4** |

**跨物体泛化(Table 7)。** 训练 π0.5 / XR-1 后用"功能等价但视觉/几何不同"的物体替换测试：颜色/形状改变基本不掉点(UR-Task1-a 仍 0.8/0.8)，但几何改变(改成锥形碗)与材质改变(木块→泡沫/磁性块)会明显掉点(UR-Task5-b 磁性块降至 0.4/0.3)，说明几何与材质仍是泛化难点。

**数字孪生 Sim-to-Real(Table 8 + 混训)。** 纯仿真训练+仿真评测下 XR-1 在天工 4 仿真任务取得 48/50、39/50、46/50、31/50(优于 ACT、Diffusion Policy)，佐证孪生数据内部一致、可作独立基准。真机混训实验中，把仿真占比从 1:1 加到 **1:5** 仍能继续提升真机成绩(Diffusion Policy Task3 0.6→0.8、Task4 0.3→0.5;XR-1 Task3 0.8→0.9、Task4 0.5→0.7)，表明是有效 Sim-to-Real 迁移而非过拟合仿真伪影(但 ACT 在接触密集的 Task3/4 始终 0.0，暴露方法自身局限)。

## 四、局限性

- **绝对成功率整体偏低。** 多数真机任务成功率在 0.1–0.8 区间、长程协作与人形任务尤其吃力(Table 2 大量 0.0–0.3)，说明这批双臂/移动/人形任务对当前 SOTA 仍很难，数据集的"高难度"是把双刃剑——利于暴露短板，但也意味着现成模型开箱即用效果有限。
- **MIND-2 评测规模小。** 关键增益(Table 3/4/5)只在 4 个 AgileX 移动任务与 3 个协作任务上、每任务 10 次试验测得，样本量小、方差大;Offline RL 仅用 200 条成功轨迹训练，结论的统计稳健性有待更大规模复现。
- **触觉/孪生覆盖不均衡。** 触觉仅在 AgileX 一个本体上采集(12K 序列)，数字孪生仅覆盖 Franka 与天工两本体，尚未做到全本体的多模态齐备。
- **文档数字存在小的不一致。** 摘要写"739 complex tasks"，而引言、Table 1、正文第 4.2 节与结论均为"759 tasks"，读者需以正文/表格为准。
- **VLM 规划器依赖闭源模型。** 语言标注用 Gemini 2.5 Pro 自动生成、慢脑基于 InternVL3-8B，标注质量与规划能力对外部大模型有一定依赖。

## 五、评价与展望

**优点。** (1) 这是公开数据集里少见地**同时**支持双臂协调、移动操作、灵巧手与高分辨率触觉四要素、并附数字孪生与 6 本体的集合，相较 RoboMIND 1.0(107K/单臂为主/4 本体/无触觉)在轨迹规模(310K)、技能广度(38→129)、模态维度上都是实质跃升,填补了"双臂×移动×触觉"交叉区的空白。(2) 把"十二类失败模式质检 + 导航/操作语义分段 + 失败轨迹作负样本"这套**质量优先**的数据工程流程讲得相当完整、可复现,是数据集论文里难得的方法学贡献,而不只是堆量。(3) 用 IQL 显式利用失败数据、并给出"成败 1:1 最优"的经验配比,对"如何盘活遥操作副产物"这个被长期忽视的问题给了可操作答案。

**不足与开放问题。** (1) MIND-2 的验证偏"点到为止":规模小、缺与更强双系统基线(如分层 VLA、Hi Robot、Fast-in-Slow 等文中引用工作)的正面对比,难判断增益主要来自数据还是来自 IQL。(2) 慢脑仅做"任务索引+进度"两标量的时序定位,规划粒度较粗,未展示对未见任务的零样本分解能力。(3) 与 AgiBot World / Galaxea 这类超大单本体集合相比,本数据集"广而不深"——每本体任务密度仍有限(天工/天轶各仅 49/46 任务),跨本体迁移训练是否真受益于这种广度、需要更系统的 scaling 研究。

**展望。** 数据集本身的价值大概率会随 VLA 社区对"双臂+移动+触觉"数据的需求上升而释放;更值得跟进的开放方向包括：把触觉扩到全本体、验证数字孪生混训在更强 VLA 上的 scaling 上限、以及把"失败轨迹→离线 RL"从后训练推进到大规模预训练阶段。作者也预告会在同一质检协议下持续扩充力矩、音频等新模态。

## 参考

1. K. Wu et al. *RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation.* RSS 2025.（前作，单臂多本体基准）
2. Q. Bu et al. *AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems.* arXiv:2503.06669, 2025.（超大单本体双臂+触觉对照）
3. S. Wu et al. *RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation.* arXiv:2511.17441, 2025.（多平台双臂+数字孪生对照）
4. I. Kostrikov, A. Nair, S. Levine. *Offline Reinforcement Learning with Implicit Q-Learning.* arXiv:2110.06169, 2021.（MIND-2-VLA 后训练算法）
5. S. Fan et al. *XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations.* arXiv:2511.02776, 2025.（本文最强 VLA 基线）
