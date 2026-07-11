# FieldGen：从遥操作前置操作轨迹到场引导的数据生成

> **论文**：*FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation*
>
> **作者**：Wenhao Wang, Kehe Ye, Xinyu Zhou, Tianxing Chen（共同一作）+ Cao Min, Qiaoming Zhu, Xiaokang Yang, Ping Luo, Yao Mu（通讯）et al.
>
> **机构**：上海交通大学 AI Institute（MoE 人工智能重点实验室）、AgiBot、香港大学、Lumina Group、上海人工智能实验室、苏州大学计算机学院
>
> **发布时间**：2025 年 10 月（arXiv 2510.20774v2）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.20774) | [PDF](https://arxiv.org/pdf/2510.20774)
>
> **分类标签**：`真实世界数据生成` `场引导轨迹合成` `半自动采集` `reward标注` `机械臂操作`

---

## 一句话总结

把操作拆成"可随意变化的 pre-manipulation 到达段"与"需要专家精度的 fine-manipulation 接触段",仅用少量遥操作标注一个 manipulation pose,再用手工设计的 cone（位置）+ spherical（姿态）吸引场,为真实采集的 RGB 观测**自动合成到达段的动作标签**;等时预算下比纯遥操作平均成功率高约 35–46 个百分点,采集吞吐 2.11×、有效采集时间占比 2.47×,配合 FieldGen-Reward 还能把 DP 在 12 分钟数据上的平均成功率从 70.8% 拉到 95.9%。

## 一、问题与动机

训练鲁棒的机器人操作策略高度依赖大规模、多样、高质量的数据,但现有三条采集路线各有硬伤:

- **遥操作**:质量高,但一人一机、认知疲劳、即便被要求"多样化"操作也会不自觉收敛到刻板运动模式。作者指出大规模遥操作数据集呈现明显的多峰分布,反而给策略学习带来负担。
- **仿真生成**（RoboTwin / MimicGen / DexMimicGen 等):可大规模随机化,但存在持久的 sim-to-real gap,且程序化轨迹缺乏真实接触交互的行为多样性。
- **半自动**（PATO / Genie Centurion):仍依赖预训练策略自动化"已被人演示过的子任务",无法跳出人类演示的多样性约束。

本文的关键观察:操作天然可分为两个需求迥异的阶段——**pre-manipulation（reach/approach)阶段**只要求轨迹最终收敛到有效的操作构型,路径本身可以任意多样;**fine-manipulation(接触密集)阶段**才真正需要专家精度。于是把"多样性"交给可自动化的到达段,把"精度"交给少量人类演示,二者解耦。

## 二、核心方法

整体是一个半自动 pipeline,分两阶段采集 $\langle \text{obs}, \text{action}\rangle$ 对:

1. **Fine-manipulation 阶段**:用遥操作标注 manipulation pose(目标位置 $p_G$ 与目标姿态 $R_G$),据此构造 pre-manipulation field $\mathcal{F}_{Gen}$——一个从少量演示轨迹抽象/外推出来的场。
2. **FieldGen-based reach 阶段**:自动脚本物理驱动机械臂到各种随机 end-effector 位姿,**记录真实 RGB 观测**;对每个采样位姿,根据其与场的空间关系 + 逆运动学(IK)算出朝目标收敛的动作序列作为标签。于是"观测是真实的、动作标签是场算出来的",绕开了 sim-to-real gap。

场 $\mathcal{F}_{Gen}$ 分解为位置的 cone field $\mathcal{F}_{pos}$ 与姿态的 spherical field $\mathcal{F}_{ori}$。

### Cone Field(位置场)

目标位置 $p_G$,轴向 $\hat{u}$(与夹爪闭合轴相反),锥半角 $\theta$。对采样点 $Q$(位置 $p_Q$)分解出沿轴分量与径向分量:

$$
a = \hat{u}^{\top}(p_Q - p_G), \qquad r = \left\| (p_Q - p_G) - a\hat{u} \right\|
$$

锥面为 $r = \tan\theta \cdot a,\ a \ge 0$。若 $Q$ 在锥内($r \le \tan\theta\, a$),沿由 $(G, Q, \hat{u})$ 张成平面内的半摆线(half-cycloid)收敛:

$$
x(t) = \mu(t - \sin t), \qquad y(t) = \nu(1 - \cos t), \qquad t \in [0, \pi]
$$

其中 $\mu, \nu$ 使曲线起于 $Q$ 止于 $G$。若 $Q$ 在锥外,先沿 $\hat{u}$ 投影到锥面上的 $P$,再走内侧摆线 $P \to G$。最终平移增量:

$$
\Delta p = \mathrm{Curve}(p_Q \to p_G \mid \hat{u}, \theta)
$$

**用大白话说**:把目标点想成一个"零重力水槽",锥内的场线像水流一样平滑汇向目标,锥外则先被"拨"进锥口再顺流而下——模仿人手"先对准、再插进去"的自然 reach-and-align 策略。选摆线而非 Bezier,是因为摆线来自"圆无滑动滚动"的纯几何约束,曲率变化天然平滑,机械臂更好执行。

### Spherical Field(姿态场)

目标姿态 $R_G \in SO(3)$,采样姿态 $R_Q$,相对旋转与其 axis-angle 表示:

$$
R_{\Delta} = R_G^{\top} R_Q, \qquad \omega = \log(R_{\Delta}) \in \mathbb{R}^3, \qquad \Delta R = -K_R\, \omega
$$

**用大白话说**:姿态误差换算成一个"转多少度、绕哪个轴"的旋转向量,再乘一个负增益,得到把夹爪平滑摆正到目标朝向的角速度修正——roll/pitch/yaw 三轴的场线都指向球心。

得到轨迹后,用参数 $\beta$(相邻帧间的平均笛卡尔位移)把轨迹按 $\text{长度}/\beta$ 离散成动作序列,再截取一个 chunk-size 片段;不足则用末点补齐。$\beta$ 太小会导致反复微调、抖动、耗时;太大则可能在离目标还远时就发出闭爪指令导致失败,实验取 $\beta = 0.0025$。

### FieldGen-Reward(带 reward 的多质量轨迹)

对遥操作得到的成功终点 $P_O$,以半径 $R$ 画球,在球内随机采一个新终点 $P_N$,朝它生成一条轨迹,并用距离打连续 reward:

$$
d = |P_O P_N|, \qquad reward = 1 - d/R
$$

**用大白话说**:同一个抓取任务,可以刻意生成"差一点点/差不少"的轨迹,并给每条明确标注质量分(完美=1,越偏越接近 0)。理论上能造出无限条带 reward 的轨迹,让策略从"好中差"的全谱数据里学到行为因果(behavioral causality),而不是只见成功样本。实测同一采集时间预算下 reward 版生成 10× 轨迹、reward 在 0–1 均匀分布,DP-R 训练时额外以 reward 为条件输入。

## 三、实验结果

**平台**:Agibot G1 机械臂,NVIDIA Orin 推理;所有策略**从零训练**(去除预训练数据影响)。策略 backbone:small RDT(170M,SigLIP 冻结编码器)、DP(288M,编码器+头联合训练)、ACT(成功率一致偏低,表中略去)。观测为腕部 RGB,动作为 end-effector delta pose,action chunk 输出 30 步。

**① 等时数据有效性**（4 任务:Pick / Rotate Pick / Transparent Pick / Affordance Pick;每 4 分钟一个 checkpoint,12 个随机物体摆位,训 50 epoch)。各 checkpoint FieldGen 平均超出遥操作 41.7 / 44.8 / 45.8 / 41.7 / 35.5 个百分点。平均成功率(%):

| 采集时长(min) | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| DP · Teleop | 29.2 | 39.6 | 52.1 | 60.4 | 64.6 |
| DP · FieldGen | 68.8 | 87.5 | 95.8 | 93.8 | 97.9 |
| RDT-small · Teleop | 16.7 | 33.3 | 41.7 | 41.7 | 56.2 |
| RDT-small · FieldGen | 60.4 | 75.0 | 89.6 | 91.7 | 93.8 |

20 分钟后 FieldGen 在所有设置上均 >80%,DP 在 4 个任务里有 3 个到 100%。

**② 等数据量泛化**（3 泛化任务:起始 EE 位姿 / 物体位置 / 未见同类物体;数据量 4k/8k/12k 帧)。平均成功率(%):

| 训练帧数 | DP · Teleop | DP · FieldGen | RDT · Teleop | RDT · FieldGen |
|---|---|---|---|---|
| 4000 | 50.0 | 91.7 | 58.3 | 80.6 |
| 8000 | 52.8 | 94.4 | 50.0 | 88.9 |
| 12000 | 55.6 | 100 | 69.5 | 91.7 |

DP 仅用 4000 条 FieldGen 样本就在"起始 EE 位姿泛化"和"物体泛化"两任务上达 100%。

**③ 轨迹多样性 / 空间覆盖**（覆盖率 = 轨迹最小包围立方体中被穿过体素占比):

| 多样性等级 | 空间覆盖率 | DP | RDT-small |
|---|---|---|---|
| Low(同起终点遥操作) | 9.04% | 0% | 0% |
| Middle(变起点遥操作) | 15.44% | 66.7% | 41.7% |
| High(FieldGen) | 18.14% | 83.3% | 83.3% |

三档下游平均成功率分别为 0% / 54.2% / 83.3%——更广的空间覆盖直接转化为更强的策略。

**④ FieldGen-Reward 消融**（DP vs DP-R,平均成功率 %):

| 采集时长(min) | DP | DP-R |
|---|---|---|
| 4 | 50.0 | 79.2 |
| 8 | 66.7 | 85.4 |
| 12 | 70.8 | 95.9 |

仅 4 分钟数据 DP-R 就到 79.2%(比 DP 高 29.2 个百分点),12 分钟达 95.9%。

**⑤ 曲线类型消融**:Cycloid 相对 Bezier,DP +16.7、RDT +8.4 个百分点(Cycloid 达 DP 75% / RDT 66.7%)。

**⑥ 长时采集省力**（2 小时采集):FieldGen 有效采集时间占比 66.73% vs 遥操作 27.07%(2.47×);帧吞吐 1203.14 frames/min vs 569.10(2.11×),大部分时间转为脚本自动执行、人只做低强度监看。

## 四、局限性

- **任务范围窄**:全部 4 个评测任务都是 pick 变体,且只覆盖**单步、单臂**的到达-抓取;论文自陈未来才扩展到多步任务。cone+sphere 场本质是"reach 到单一目标位姿"的几何先验,无法处理真正接触密集的操作段。
- **fine-manipulation 仍靠遥操作**:每个 episode 人类只标注一个 manipulation pose,精细接触动作并未被自动化,系统只是把"到达段"这部分工作量卸给脚本。
- **动作标签是几何构造而非真实执行**:reach 动作来自场+IK,而非真人真机执行的轨迹;虽观测真实,但动作与真实动力学/接触的一致性依赖场设计是否合理,泛化到更复杂形态(如柔性、遮挡对准)时未验证。
- **reward 纯几何**:$reward = 1 - d/R$ 只反映终点空间偏差,不反映任务真实成败或接触质量,可能与下游真实回报错配。
- **场为手工设计、超参敏感**:$\theta$、$\beta$、$K_R$、$R$ 等需人工调;$\beta$ 消融显示过大过小都会显著掉点(0.0045 时 RDT 掉到 8.3%)。
- **观测生成仍需真机运动**:自动脚本要物理驱动机械臂遍历随机位姿采 RGB,尚未做到论文展望里"3DGS/NeRF 合成观测"的全自动。

## 五、评价与展望

**优点**:核心洞见——"reach 段要多样、接触段要精度"的相位解耦——干净且实用,把最贵的人力集中到最需要专家的接触瞬间,其余交给几何场自动扩样。"真实观测 + 场算动作标签"的组合巧妙地既绕开 sim-to-real,又拿到了远超遥操作的空间覆盖(9%→18%),而且工程门槛低:不需要预训练策略、不需要仿真器、几十行脚本即可跑。等时/等量两条对照设计都很扎实,数字提升幅度大且跨 DP/RDT 两个 backbone 一致,长时采集的吞吐/省力量化也有说服力。

**与公开工作的关系**:相较 MimicGen / DexMimicGen / RoboTwin(仿真域内做数据增广)本文直接在真实世界生成,牺牲了"接触段也能自动合成"的能力换取真实性;相较 PATO / Genie Centurion(自动化人已演示的子任务)本文在更底层做相位切分并主动生成人类演示之外的多样到达轨迹,理论上突破了人类行为多样性上限;与 DemoGen(合成 demo)思路相近但这里是显式几何场而非学习式插值。可视为"MimicGen 的真实世界、单相位版本 + reward 谱"。

**开放问题与改进方向**:
- 把场从"reach 到单一 6-DoF pose"推广到**多关键帧/多步**任务(如插拔、倒水),需要在接触段之间串接多个场并处理阶段切换。
- 用**学习式场**替代手工 cone/sphere,让吸引场从少量演示中自适应估计目标流形,减少超参调优并支持非凸目标区域。
- reward 从纯几何距离升级为**任务结果或接触质量**驱动,使 reward-conditioning 与真实回报对齐,更接近 offline-RL 式的质量分层利用。
- 结合论文展望的 **3DGS/NeRF 观测合成**,把"物理驱动机械臂采 RGB"也自动化,进一步逼近零人工的闭环数据引擎。
- 目前只在腕部单目 RGB + delta-pose 动作空间验证,向多视角/深度/双臂协同迁移时场的构造与 IK 求解是否仍稳健有待检验。

## 参考

1. Mandlekar et al. *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations*, arXiv:2310.17596, 2023.（人类演示驱动的自动数据生成,思想最相关的对照)
2. Dass et al. *PATO: Policy Assisted Teleoperation for Scalable Robot Data Collection*, arXiv:2212.04708, 2022.(半自动遥操作 baseline)
3. Wang et al. *Genie Centurion: Accelerating Scalable Real-World Robot Training with Human Rewind-and-Refine Guidance*, arXiv:2505.18793, 2025.(真实世界长程半自动采集)
4. Mu et al. *RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins*, ECCV 2024.(仿真数据生成对照)
5. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, IJRR 2023.(下游主力策略 backbone)
