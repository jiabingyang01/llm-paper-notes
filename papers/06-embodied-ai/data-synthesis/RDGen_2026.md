# RDGen：基于强化学习的高质量机器人演示生成

> **论文**：*RDGen: Demonstration Generation for High-Quality Robot Learning via Reinforcement Learning*
>
> **作者**：Zijian Zhu*, Menglin Zou*, Zhuang Li, Yaojie Tu, Xinhai Sun（*等贡献）
>
> **机构**：Synthoid.ai
>
> **发布时间**：2026 年 05 月（arXiv 2605.30957）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.30957) | [PDF](https://arxiv.org/pdf/2605.30957)
>
> **分类标签**：`RL生成演示` `sim-to-real` `VLA训练数据` `SAC` `VLM任务解析` `pick-and-place`

---

## 一句话总结

RDGen 把训练好的 RL 策略（而非部署时的最终控制器）重新定位为**轨迹数据生成引擎**：用 VLM（Qwen3-VL）做任务解析、Grounding DINO 做语言引导的 3D 目标定位、SAC 训练的 sim-to-real 策略执行动作,只保留成功且平滑的 rollout 作为下游 VLA 的干净监督;在抓取-放置任务上,RDGen 生成的轨迹平均 jerk 比人工遥操作低约 6~10 倍,用其训练的 π0 成功率相对遥操作数据提升 20~30 个百分点。

## 一、问题与动机

VLA 模型的性能瓶颈正从模型架构转向数据本身:机器人动作数据无法像文本/图像那样在网络上大规模抓取,每条轨迹都必须由真实或仿真智能体实际执行、绑定具体任务、并配上有效的动作监督。当前主流数据来源人工遥操作(labor-intensive、成本高、难扩展)和第一视角人类视频(embodiment gap 更大,手部追踪噪声、相机自运动、遮挡、缺失力/接触信号导致难以恢复出干净的机器人可执行动作标签)都存在共同缺陷:所得轨迹本身带噪声,难以作为清晰的动作监督信号。

论文的核心洞察是:一旦 RL 策略在某任务上训练收敛,它产生的轨迹天然是目标导向、物理可执行、时序一致、并按奖励结构优化过的;配合合适的奖励塑形,还能比人类演示明显更平滑。这促使作者把 RL 的角色从"被部署的控制器"重新框定为"能持续供给高质量轨迹的数据引擎"。

## 二、核心方法

RDGen 整体流程(对应论文 Figure 1)分两条线汇合:

**视觉-语言理解线**:输入指令 → VLM(Qwen3-VL)任务解析器生成物体查询 → Grounding DINO 目标检测 → 结合深度图做 3D 定位。

**仿真控制学习线**:Isaac Sim 环境 → SAC 强化学习训练低层技能策略 π → sim-to-real 迁移到真实机器人执行。

两线交汇后,成功的 rollout 经过轨迹清洗/后处理管线,形成干净的轨迹数据集,用作下游 VLA 训练的监督信号。

**1. VLM Agent 任务理解(3.2 节)**。将语言驱动的长时程视觉操作建模为闭环 agentic 过程:给定指令 $l$ 与观测 $o_t$,agent 反复规划下一个可执行动作、等待对应 RL 策略执行、验证结果、更新任务状态。Agent 维护结构化记忆 $m_t$,记录原始指令、已完成动作、剩余计划和最近的验证反馈。动作表示为

$$a_t = \left(s_t, q_t^{\text{obj}}, q_t^{\text{tar}}\right),$$

其中 $s_t$ 为技能类型、$q_t^{\text{obj}}$ 为被操作物体的语言描述、$q_t^{\text{tar}}$ 为目标区域的可选语言描述。系统内部拆成 Planner(聚焦任务相关视觉证据、忽略机械臂/夹爪本身,生成下一步的技能+物体查询+可选目标查询)和 Verifier(检查动作执行后的观测,判断预期效果是否达成,据此更新或保持记忆、决定重试还是继续)两个解耦角色,像"任务经理"一样不断布置原子任务并核验完成情况,以对抗长时程操作中的误差累积和状态漂移。

**2. Grounding DINO 目标定位(3.3 节)**。VLM 分解出的每个原子动作对应的语言查询 $q$ 送入 Grounding DINO,在 RGB 图像 $I_t$ 上预测候选框及置信度,取置信度最高的框:

$$B_q = \arg\max_{B \in f_{\text{DINO}}(I_t, q)} \text{Conf}(B).$$

选中的 2D 框结合对齐后的深度图与标定好的相机内外参被提升为机器人坐标系下的 3D 位置 $\mathbf{p}_r^q$,最终把符号动作接地为度量层面的机器人指令 $(s_t, q_t^{\text{obj}}, q_t^{\text{tar}}) \mapsto (s_t, \mathbf{p}_r^{\text{obj}}, \mathbf{p}_r^{\text{tar}})$。这一模块是仿真到真实的轻量接地接口:仿真里物体/目标坐标可直接从环境状态读取,真实世界则靠语言引导检测+RGB-D 估计,且不依赖 CAD 模型或物体级点云,因而对任意语言查询的物体都有较强泛化性。

**3. 低层策略学习(3.4 节)**。将抓取-放置技能建模为 RL 问题,在 Isaac Sim 中用 SAC 训练。

观测空间是 14 维向量

$$\mathbf{s}_t = \left[\mathbf{p}_{\text{target},t}, \mathbf{p}_{\text{ee},t}, \mathbf{q}_{\text{ee},t}^{+}, \mathbf{q}_{\text{ee},t}^{-}\right],$$

同时包含原始四元数 $\mathbf{q}_{\text{ee},t}^{+}$ 及其相反数 $\mathbf{q}_{\text{ee},t}^{-} = -\mathbf{q}_{\text{ee},t}^{+}$,用于消解四元数正负号歧义(即 $q$ 与 $-q$ 代表同一姿态)。动作空间为末端位姿增量

$$\mathbf{a}_t = [\Delta x_{\text{ee}}, \Delta y_{\text{ee}}, \Delta z_{\text{ee}}, \Delta \phi_{\text{ee}}, \Delta \theta_{\text{ee}}, \Delta \psi_{\text{ee}}]^{\top} \in \mathbb{R}^6,$$

经逆运动学转换为关节目标执行。

奖励函数为 $r_t = r_{\text{task}} + r_{\text{shape}} - r_{\text{penalty}}$,其中惩罚项

$$r_{\text{penalty}} = \lambda_{\text{still}}\mathbb{I}_{\text{still}} + \lambda_{\text{step}} + \lambda_{\text{IK}}\mathbb{I}_{\text{IK\_fail}} + \lambda_{\text{collision}}\mathbb{I}_{\text{collision}}$$

分别惩罚原地不动、按步计费、IK 求解失败和碰撞;塑形奖励 $r_{\text{shape}} = r_{\text{progress}} + r_{\text{center}} + r_{\text{path}}$ 由三部分构成——进度奖励 $r_{\text{progress}} = \lambda_{\text{progress}}(d_{t-1} - d_t)$ 鼓励末端逐步逼近目标;分轴对中奖励对 $x,y,z$ 三个方向分别加权移动量;路径偏离惩罚 $r_{\text{path}} = -\lambda_{\text{path}}\|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{proj}}\|_2$ 约束末端尽量贴合起点到目标的直线路径($\mathbf{p}_{\text{proj}}$ 为投影点)。这套奖励塑形不仅要求任务成功,还显式地"拉直"运动轨迹,是生成轨迹比人工遥操作更平滑的关键设计。回合在成功、异常接触、目标物异常移动、反复 IK 失败或超时时立即终止重置,避免训练被无效交互数据主导。

**4. Sim-to-real 适配与仿真扩展采集(3.5-3.6 节)**。训练时向末端位姿注入噪声(位置 $x,y,z$ 方向各 5mm 扰动、四元数各分量 0.1 扰动后重新归一化)以匹配真实传感器噪声。针对四元数符号歧义($q$ 与 $-q$ 代表同一姿态),采用双四元数表示,观测中同时包含原始四元数与其相反数并按 $w$ 分量符号固定排列顺序,得到最终 14 维观测 $\mathbf{o}_t = [\mathbf{p}_{\text{target}}, \mathbf{p}_{\text{ee}}, q_{\text{ee}}, -q_{\text{ee}}]$,避免同一物理姿态在观测空间中出现不连续表示导致训练不稳定。除真实机器人部署外,通过变化光照、物体位姿、相机视角与场景配置,RDGen 还能以极低边际成本在仿真中批量生成多样化成功轨迹,作为下游 VLA 预训练数据。

**平滑度评价指标**。借助最小 jerk 轨迹原理,jerk 定义为加速度对时间的导数 $\mathbf{j}(t) = d\mathbf{a}(t)/dt = d^3\mathbf{x}(t)/dt^3$,平滑轨迹对应更小的平方 jerk 积分 $J = \int_0^{t_f}\|\mathbf{j}(t)\|^2\,dt$。实验中对每条轨迹计算平均 jerk 幅值,再取所有轨迹均值作为平滑度指标,数值越低说明运动越平滑。

## 三、关键结果

实验平台:机械臂 Marvin M6CCS + 灵巧手 Ruiyan RY-H2 + 相机 Gemini 435Le,GPU 为 NVIDIA RTX PRO 6000;仿真环境用 Isaac Sim 按同一配置搭建,仿真与真实共享统一坐标系。SAC 训练超参:学习率 $1\times10^{-4}$、batch size 256、每步梯度更新 16 次、温度系数 0.2。下游 VLA 训练直接复用 π0 的默认超参配置。

在"抓取灰色方块并放入纸箱"任务上,迁移到真机的 RL 策略首次尝试成功率达到 100%(采集的 20 条 rollout 全部无需人工干预即成功,如论文 Figure 3 所示)。

论文进一步对比了 RDGen 生成数据与人工遥操作数据在轨迹平滑度(20 条轨迹的平均 jerk)和下游 VLA(π0,10 次随机物体位置试验)成功率上的差异:

| 任务 | 数据来源 | 平均 Jerk (m/s³) ↓ | 下游 VLA 成功率 ↑ |
|---|---|---|---|
| Cube | 人工遥操作 | 2.68 ± 0.41 | 60% |
| Cube | RDGen | 0.47 ± 0.02 | 80% |
| Cola | 人工遥操作 | 5.59 ± 2.44 | 70% |
| Cola | RDGen | 0.57 ± 0.06 | 100% |

RDGen 在两个任务上的平均 jerk 都比人工遥操作低约 6~10 倍,方差也显著更小,说明生成的运动更平滑、噪声更少;与之对应,用 RDGen 数据训练的 π0 在两个任务上分别取得 80%、100% 的成功率,均高于遥操作数据训练的对照组(60%、70%)。作者据此认为轨迹平滑度的改善直接转化为了下游策略学习的可学性提升。

## 四、评价与展望

**优点**:RDGen 提出的核心框架——"训练好的 RL 策略不只是最终控制器,还可以是可持续产出干净轨迹的数据引擎"——是一个简洁但实用的视角转换,与近年 sim-to-real RL 通常直接把策略部署为最终控制器的路线不同。方法把 VLM 任务解析(Planner/Verifier 双角色 + 结构化记忆)、开放词表目标检测(Grounding DINO)与底层 RL 技能解耦,使系统对新物体的语言查询有较好泛化性,而不依赖任务专属的 CAD 模型或点云配准,这与 BridgeData V2、DROID、Open X-Embodiment 等偏重"多样但含噪"的数据采集范式形成互补而非替代关系:RDGen 更像是为特定任务批量补充"干净、动作分布规整"的监督信号的手段。奖励塑形中显式的路径偏离惩罚($r_{\text{path}}$)是让生成轨迹比遥操作更平滑的直接机制,这个设计思路具备一定的可迁移性。

**局限性(作者自述)**:第一,当前框架主要针对抓取、放置一类相对粗粒度的操作,难以直接扩展到叠衣物、拧螺丝等需要毫米级精度、且需要精心设计奖励函数的精细技能。第二,训练管线尚未做到完全任务通用——例如抓取任务中用弹簧机制随机化物体位置这类设计是任务特化的,未必能直接迁移到其他操作技能。

**开放问题与可能的改进方向**:(1)实验仅覆盖单一 pick-and-place 任务的两个物体变体(Cube、Cola),规模较小(20 条轨迹、10 次评测试验),尚未验证方法在更多任务类型、双臂或灵巧手精细操作上的可扩展性;(2)RL 策略生成的演示分布可能相对单一(动作模式集中于奖励函数塑造出的"直线路径"附近),相比人类演示可能牺牲动作多样性,对下游 VLA 的分布外泛化能力是一把双刃剑,论文未做相关消融;(3)3.6 节提出的仿真批量采集仅作为设想提及,未给出对应的规模化实验或量化收益,是后续工作值得补齐的部分。

## 参考

- Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* RSS 2025.
- Zitkovich, B. et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.* CoRL 2023.
- Haarnoja, T. et al. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML 2018.
- Walke, H.R. et al. *BridgeData V2: A Dataset for Robot Learning at Scale.* CoRL 2023.
- Khazatsky, A. et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.* arXiv:2403.12945, 2024.
