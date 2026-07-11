# SafeMimic：面向移动操作的安全、自主的人到机器人模仿

> **论文**：*SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation*
>
> **作者**：Arpit Bahety, Arnav Balaji, Ben Abbatematteo, Roberto Martín-Martín
>
> **机构**：The University of Texas at Austin（Robot Interactive Intelligence Lab, RobIn）
>
> **发布时间**：2025 年 06 月（arXiv 2506.15847）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.15847) | [PDF](https://arxiv.org/pdf/2506.15847)
>
> **分类标签**：`human-to-robot` `mobile-manipulation` `safe-exploration` `learning-from-video`

---

## 一句话总结

SafeMimic 只用**一段第三人称人类视频** 就让移动操作机器人自主学会多步任务：先用 body/hand tracker + 接触检测 + VLM 把视频拆成语义段并翻译成机器人第一视角动作，再用一组在仿真中预训练的 **safety Q-function** 在真实世界中围绕人类动作采样、验证并以 receding-horizon 方式安全探索,不安全时**回溯（backtracking）** 并换抓取模式,最终在七个多步任务上取得至少 40%（最高 100%）成功率,同时把不安全动作率从基线的 ~13% 压到 **0.6%**,且**全程无人监管、无人复位**。

## 一、问题与动机

让家用机器人像"看人做一遍就学会"那样掌握新的移动操作技能,能绕开昂贵、耗时的遥操作数据采集。但从**单段第三人称人类视频**学多步移动操作面临几重挑战:

1. **理解 what 与 how**:既要抽取任务的高层语义变化(环境里发生了什么),也要抽取造成这些变化的低层人手/身体运动;
2. **第三视角→第一视角**:人类动作是第三方视角,机器人要用自身传感器执行并监控,需要坐标系转换;
3. **形态差异(morphology gap)**:人手与机械臂的运动学、抓取能力不同,直接照搬会碰撞、超关节限位、抓空;
4. **安全+自主**:真实世界试错很容易损坏机器人或环境。以往从人类视频"播种"策略再在真机 fine-tune 的方法(如 Human-to-robot in the wild、VideoDex、DEFT、ScrewMimic)大多局限于**短程技能**,且需要人类**持续监督**——不断复位任务、判断成功、盯着别出危险。自主复位类工作(reset-free RL)又普遍**回避安全问题**。

SafeMimic 的目标:让机器人能在**预测危险发生之前** 就规避它,并在探索失败时回溯换策略,从而**安全且自主**地从单段人类视频学会多步移动操作。

## 二、核心方法

框架分三部分:(A) 解析并翻译人类视频得到初始策略;(B) 用 safety Q-function + backtracking 在真机上安全自主地适配探索;(C) 用 policy memory 从过往成功里学习、减少未来探索。

### A. 人类视频的分解、解析与翻译(what & how)

对一段 RGB-D 人类视频:
- **粗分段**:用 body/hand visual tracker 跟踪人体运动,按帧间身体平移量阈值判断当前是"导航"还是"操作";
- **细分段+语义标注**:导航段被视为最细粒度,用 VLM 抽取导航目标物体/位置;操作段再结合 VLM 与**接触检测器**,按"是否与物体接触"进一步拆成以抓取动作或接触界面变化(如开始擦拭)为起点的子段。最终得到一串带单一语义目标的段,如 `navigate_to`、`reach_for_and_grasp`、`open` 等(完整列表见附录 B),可被机器人**自监督地** 逐段优化;
- **翻译到机器人参考系**:语义变化(what)与视角无关;运动(how)则需转到第一视角。导航段假设机器人起点"足够接近"人类初始位置(无需标定),把人类导航动作序列转成 base pose 的**相对变化**;操作段先把手部位姿转成相对于人体的坐标,再算相邻帧间手的相对运动;抓取则从机器人可用的候选抓取里挑一个与人类演示**最接近** 的。

这一步产出一个初始策略,但因形态差异与跟踪误差,**直接执行几乎必然失败**(实验中 0% 成功)。

### B. 安全、自主的真机适配

**问题形式化**:标准 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, R, T, \gamma)$,$R$ 是基于语义目标达成的稀疏奖励。借用 Safe RL 框架,定义不安全状态集,用指示函数 $\mathcal{I}(s)$ 标记(碰撞、超力矩、超关节限位、抓取丢失等多种失败模式的组合):

$$\mathcal{I}(s) = \max_i \mathcal{I}_i(s)$$

机器人目标是在**保持安全** 的前提下最大化任务奖励:

$$\max_\pi \sum_{t=0}^{T} \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\big[R(s_t, a_t)\big] \quad \text{s.t.} \quad \mathbb{E}_{s_t\sim\rho_\pi}\big[\mathcal{I}(s_t)\big] = 0$$

> 用大白话说:$\mathcal{I}$ 就是"出没出事"的报警器,把碰撞/超力/掉物等所有失败模式取最大(任一触发就算不安全);目标是"完成任务的同时,一路上一次险都别出"。

**Safety Q-function**:预测在状态 $s_t$ 执行 $a_t$ 会导致失败的**概率**:

$$Q_{\text{safe}}(s_t, a_t) = \mathcal{I}(s_t) + \big(1 - \mathcal{I}(s_t)\big)\, \mathbb{E}_{s_{t+1}\sim T(\cdot\mid s_t, a_t)}\big[\mathcal{I}(s_{t+1})\big]$$

并按失败类型拆成**集成(ensemble)**,每种失败模式一个 Q,再取最大:

$$Q_{\text{safe}}(s_t, a_t) = \max_i Q_{\text{safe},i}(s_t, a_t)$$

> 用大白话说:这是一个"这一步会翻车吗"的评分器。当前已经出事就是 1;否则等于"下一步大概率会不会出事"。多个专门评分器(专管碰撞、专管超力……)各打分,取最悲观的那个作为总分。

**在仿真中预训练**:直接在真机里学"什么动作会出事"太危险,于是在 **OmniGibson** 仿真里做 domain randomization,覆盖铰接物体交互、刚体抓放、base 导航等场景,用 motion planner 生成随机及加噪的任务相关动作来采集失败/安全样本训练这组 Q。状态表示用**点云 + 本体感知**(而非 RGB),因为点云的 sim2real gap 更小,可**零样本** 迁移到真机。

**安全探索(receding horizon)**:适配时围绕人类演示动作采样候选——抓取动作在抓取生成器给出的**离散集合(本文 3 个抓取)** 里探索,优先最接近人类演示的那个;连续运动从**以解析出的人类运动为均值、方差放大以鼓励探索的高斯分布** 里采样。每步选 $Q_{\text{safe}}$ 最低(最安全)的动作执行,之后用解析出的语义目标 + VLM 判定该段是否完成,未完成就再采一批。

**回溯机制(backtracking)**:若当前状态下所有采样动作都不安全,即对所有 $a_i$ 有

$$Q_{\text{safe}}(s, a_i) > \epsilon$$

则**回溯一步**——执行上一动作的逆动作(反向运动);与物体交互时(开门/开抽屉、移动被抓物)这会把环境带回上一状态,从而换个分支继续探索。若单步回溯迭代若干次(本文 50 个动作)仍无路可走,就一路回溯到抓取段、**换一种抓取模式** 再探索。作者强调"抓取模式探索 + 轨迹级探测"的组合对把人类多步策略适配成机器人可行策略至关重要。

### C. 从过往成功中学习(Policy Memory)

为避免同一任务重复从头探索,SafeMimic 训练一个**动作预测策略网络**,把点云 $P$ 与语言任务描述 $l$ 映射到成功过的抓取模式 $g \in SE(3)$ 及抓取后动作序列 $(a_0, \dots, a_T)$。架构:**PointNet** 编码视觉 + **SentenceTransformer** 编码任务描述 + MLP head,用成功试验数据的**几何增强**(旋转、平移)训练,从而能在不同视角下复用成功经验、偏置(bias)未来探索。

**硬件**:PAL-Robotics **Tiago++** 移动机械臂;单臂末端位姿用 IK 控制,base 用相对位置+偏航指令;头部 Orbbec Astra S RGB-D 相机;腕部 ATI mini45 力矩传感器。考虑的失败模式:臂碰撞、base 碰撞、关节限位、力矩超限、抓取丢失、掉物。

## 三、实验结果

在 **7 个多步移动操作任务** 上评测:`boxing`、`shelving`、`store_in_drawer`、`erase_whiteboard`、`refrigerating`、`fill_pot`、`load_oven`。每任务尝试 5 次。真机 fine-tuning 平均**每导航段 5 分钟、每操作段 15 分钟**,成功由 VLM 判定。

**基线**:Direct Execution(w/o SQF,直接执行不验证)、Direct Execution(with SQF,直接执行但用 Q 验证)、Exploration(w/o SQF,即去掉 SQF 的 SafeMimic)、IL(all safe actions,BC-RNN+PointNet,在所有未违规样本上训)、IL(successful episodes,仅在成功轨迹上训)。注:IL 基线只在 "Place/Open/Close" 段评测,导航与 pick 段直接给成功解。

### Q1 能否从单段第三人称演示完成多步任务

| 方法 | 七任务最终成功率 |
|---|---|
| **SafeMimic** | **最低 40%,探索适配最高达 100%** |
| Direct Execution(w/o SQF) | 7 个任务全 **0%** |
| Direct Execution(with SQF) | 0%(且部分段因 Q 假阳性反而低于 w/o SQF) |
| IL(两种) | 有零星成功,但无法稳定完成 |

- Direct Execution(w/o SQF)因潜在不安全动作,所需**人工干预比 with SQF 多 82%**。
- **7 个任务中有 6 个** 必须靠**抓取模式适配** 才成功:如 `fill_pot`,人类式抓锅会撞到水槽边缘,机器人回溯到 pick 段改用**自上而下(top-down)抓取** 才放进水槽;`store_in_drawer` 人类式抓法会触发关节限位,机器人换抓取才拉开抽屉。作者据此指出,仿真里生成的少量带噪数据足以训好 SafeMimic 的 SQF,但训练稳健的 IL 策略对数据数量与质量要求更高。

### Q2 安全性(unsafe action rate = 不安全动作 / 总执行动作,跨七任务)

| 方法 | Unsafe action rate |
|---|---|
| Exploration(w/o SQF) | 14.2% |
| Direct Execution(w/o SQF) | 13.4% |
| IL(all safe actions) | 10.8% |
| IL(successful episodes) | 9.5% |
| Direct Execution(with SQF) | **0.5%** |
| **SafeMimic** | **0.6%** |

加入 SQF 把安全违规**减少了 13.6 个百分点**(从 ~14% 降到小于 1%),验证了仿真预训练的 SQF 能可靠预测真机中的不安全动作(如以错误方向拉抽屉、放物时会撞到旁边瓶子)。

### Q3 policy memory 能否减少未来探索

用成功试验训练 policy memory 后,重评几个任务的抓取/开启段,**探索的 waypoint 数量在所有任务上显著下降,最高减少 67%**(Fig. 7)。对放置类任务,学到成功抓取模式后大幅减少后续执行的探索;`store_in_drawer` 中 policy memory 让机器人后续能直接成功操作抽屉。

### Q4 对不同用户/环境的鲁棒性

- **3 个不同用户** 演示 `shelving`:SafeMimic 都能恢复、解析并翻译;有趣的是三人都偏好 top-down 抓,但那样放不进架子,机器人都自主适配了抓取。
- **同一演示者跨 3 个环境** 做 `shelving`:成功率 **100% / 100% / 66%**,展示了泛化能力。

## 四、局限性

1. **仿真数据需覆盖真机场景**:SQF 的仿真训练场景需与真机探索遇到的相似,仍需一定的仿真任务工程;作者认为可用大规模、多任务/多资产的仿真预训练缓解。
2. **失败模式需先验枚举**:只研究了有限几种(碰撞、超力、超限位、抓取丢失、掉物),扩展到更多类型的安全违规/任务失败是开放问题。
3. **依赖初始人-机位姿对应**:把动作表示成相对运动才能得到相似轨迹,需在开始时估计人类在地图坐标系的初始位姿并把机器人导航过去。
4. **依赖抓取生成器**:从人类演示抓取准确推断机器人可行抓取仍是开放挑战。
5. **回溯无法处理不可逆事件**:能复位开门/开抽屉等,但掉物等不可逆失败无法恢复,需集成自主复位框架。
6. **SQF 不从真机数据更新**:目前仅仿真预训练;从真机偶发失败中在线改进 SQF 可进一步提升鲁棒性。

## 五、评价与展望

**优点**:
- **问题闭环完整**:把"解析人类视频→跨视角/跨形态翻译→安全探索→回溯纠错→经验复用"串成一个真正**无人监督、无人复位**的自主学习回路,这在从人类视频学**多步移动操作** 的工作里是少见的完整度。相比 ScrewMimic、Human-to-robot in the wild、VideoDex、DEFT 等偏短程、需人监督的前作,是实质推进。
- **安全的抽象干净且可迁移**:把安全建模成"每种失败模式一个 Q、取最大"的 ensemble,用点云+本体感知做状态、仿真预训练零样本迁移真机,把不安全率压到小于 1%,证明 sim-to-real 的安全评估比 sim-to-real 的策略迁移**更易实现**——这是一个有普适价值的洞见。
- **抓取模式探索 + 轨迹探测的组合**被实验证明是跨越人-机形态差异的关键(6/7 任务依赖),回溯到抓取段换模式的设计朴素但有效。

**缺点与开放问题**:
- **评测规模有限**:每任务仅 5 次、七任务、单一机器人(Tiago++),成功率区间跨度大(40%~100%),统计功效偏弱;跨环境第三个场景掉到 66% 说明泛化仍脆。
- **安全全靠先验失败模式**:一旦真机出现未建模的失败类型,SQF 无从预警;而 SQF 不在线更新,系统缺乏"从真实事故学习"的能力,与其"安全"定位存在张力。作者也承认 SQF 假阳性会误伤成功段(Direct Execution with SQF 部分段反而更低)。
- **对上游模块强依赖**:人体/手部 tracker、接触检测、VLM 判定、抓取生成器任一失效都会传导;VLM 判成功的可靠性、每操作段 15 分钟的真机探索成本,都限制了向长程/大规模任务扩展。
- **回溯的可逆性假设**:一旦掉物即无法恢复,对真实家庭里大量不可逆操作是硬约束。

**可能的改进方向**:把 SQF 升级为可在线更新的安全评判器(结合真机偶发失败做保守更新);用大规模多资产仿真预训练一个"通用安全先验"以摆脱逐任务仿真工程;将 policy memory 从"逐任务复用"扩展为跨任务/跨物体的可组合技能库;引入自主复位以处理不可逆失败。总体看,SafeMimic 是"从单段人类视频安全自主学移动操作"方向上一份工程完整、洞见清晰的推进工作,其**安全评估比策略更易 sim-to-real** 的思路值得后续沿用。

## 参考

1. Bahety et al. *ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection.* RSS 2024.(同组前作,双臂人到机器人模仿)
2. Bahl, Gupta, Pathak. *Human-to-robot imitation in the wild.* 2022.(从野外人类视频播种策略再真机微调)
3. Shaw, Bahl, Pathak. *VideoDex: Learning dexterity from internet videos.* CoRL 2023.
4. Kannan et al. *DEFT: Dexterous fine-tuning for hand policies.* CoRL 2023.(人类先验 + 真机试错微调)
5. Mendonca, Bahl, Pathak. *Structured world models from human videos.* 2023.(从人类视频学移动操作的世界模型)
