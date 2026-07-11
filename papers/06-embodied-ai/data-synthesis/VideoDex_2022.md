# VideoDex：从互联网视频中学习灵巧操作

> **论文**：*VideoDex: Learning Dexterity from Internet Videos*
>
> **作者**：Kenneth Shaw\*, Shikhar Bahl\*, Deepak Pathak（\* 共同一作,顺序由抛硬币决定）
>
> **机构**：Carnegie Mellon University（卡内基梅隆大学）
>
> **发布时间**：2022 年 12 月（arXiv 2212.04498，v1 于 2022-12-08）
>
> **发表状态**：CoRL 2022（6th Conference on Robot Learning, Auckland, New Zealand）
>
> 🔗 [arXiv](https://arxiv.org/abs/2212.04498) | [PDF](https://arxiv.org/pdf/2212.04498)
>
> **分类标签**：`人到机器人视频迁移` `灵巧操作` `动作先验` `模仿学习` `Neural Dynamic Policies`

---

## 一句话总结

VideoDex 把互联网人手视频（Epic Kitchens）当作"伪机器人经验",通过**手-臂重定向（retargeting）**将人手 3D 轨迹映射到 16-DoF LEAP 灵巧手 + xArm6 机械臂,从而同时提取 visual / action / physical 三类先验来预训练策略;仅需每任务 120-175 条真机遥操演示微调,在 pick、place、open 等 7 个真机灵巧任务上测试物体成功率普遍超过不用动作先验的基线(如 place 测试集 0.70 vs BC-NDP 0.35)。

## 一、问题与动机

灵巧操作面临"先有鸡还是先有蛋"的困境:要在真实世界安全采集数据,机器人本身就得先具备经验;而高自由度灵巧手的动作空间巨大,直接在真机上学习样本效率极低。互联网上存在海量人手交互视频,以往工作大多只把它们当作**视觉预训练**的素材(仅学 visual prior,如 R3M、MVP),用来初始化视觉编码器。

作者的核心洞见是:视频里不仅有"世界长什么样"(visual prior),更蕴含"人如何运动、意图如何"(action prior)。灵巧手与人手形态相似,使得从人类视频学习动作尤为可行。难点在于要理解 3D 场景、恢复人手意图,并跨越"人-机器人本体差异(embodiment gap)"完成迁移。为此作者提出把三种先验统一进单个开环(open-loop)策略:

- **Visual prior**:场景外观的语义表示;
- **Action prior**:人类为完成某类任务的典型动作(以网络初始化形式编码);
- **Physical prior**:用二阶动力系统(Neural Dynamic Policies)约束轨迹平滑、安全,避免过拟合到含噪的人手检测。

## 二、核心方法

整体流程(Algorithm 1):先在**重定向后的人手轨迹**上预训练策略网络(得到动作先验参数 $\theta_h$),再在真机遥操演示上微调。

### 2.1 物理先验:Neural Dynamic Policies (NDP)

NDP 以 Dynamic Movement Primitive (DMP) 方程作为网络"骨干",输出任意长度的平滑开环轨迹:

$$\ddot{y} = \alpha\big(\beta(g-y) - \dot{y}\big) + f_w(x, g)$$

其中 $y$ 是机器人坐标系下的位置,$g$ 是目标点,$f_w$ 是径向基强迫函数(forcing function),$x$ 为时间变量,$\alpha,\beta$ 为全局常数。一个小 MLP 从场景特征回归出目标 $g$ 与强迫函数形状参数 $w$,再由前向积分器展开成轨迹。

**用大白话说**:与其让网络逐帧硬输出动作(容易抖、容易过拟合噪声),不如让网络只决定"往哪个目标点去、路径大致什么形状",再交给一个物理弹簧-阻尼系统把轨迹积分出来,天然平滑、可展开成任意时长,正好匹配变长的人类视频。

### 2.2 动作先验:人手→机器人重定向

**手部姿态重定向**:沿用 Robotic Telekinesis (Sivakumar et al.) 的方法,在人手/机器人手的掌心与指尖间人工定义关键向量 $v_i^h, v_i^r$,构造能量函数最小化两者差异:

$$E_\pi\big((\beta_h, \theta_h),\, q\big) = \sum_{i=1}^{10} \big\| v_i^h - (c_i \cdot v_i^r) \big\|_2^2$$

其中 $(\beta_h,\theta_h)$ 是 MANO 模型的手形/姿态参数,$q$ 是机器人手关节,$c_i$ 是缩放系数。训练一个 MLP $H(\cdot)$ 隐式最小化该能量,把人手姿态映射为 LEAP 手关节 $x_r = H(x_h)$。

**用大白话说**:人手和机器人手指头数量、比例都不一样,直接抄关节角没意义;所以只对齐"指尖相对掌心的方向向量",让机器人手摆出功能上等价的抓握姿态。

**手腕/相机轨迹重定向**:这是本文相对前作的关键工程。人类第一视角视频里相机是移动的,需要补偿。步骤:① 用 OpenPose 裁剪手部 → FrankMocap 得到 3D MANO 参数;② 用 PnP(相机内参由 COLMAP 估计)算出手腕在相机系的位姿 $M_{C_t}^{Wrist}$;③ 用单目 SLAM(ORBSLAM3)估计首帧相机 $C_1$ 到当前帧 $C_t$ 的相机运动 $M_{C_1}^{C_t}$;④ 估计重力方向(用 Detic 检测桌面/地面等平行地面的物体 + AdaBins 单目深度 → 表面法向),据此求初始 pitch/roll:

$$\text{pitch} = \tan^{-1}\!\Big(x_{Acc} \big/ \sqrt{y_{Acc}^2 + z_{Acc}^2}\Big), \qquad \text{roll} = \tan^{-1}\!\Big(y_{Acc} \big/ \sqrt{x_{Acc}^2 + z_{Acc}^2}\Big)$$

最终把人手腕轨迹变换到机器人系(启发式地重缩放并旋转以对齐机器人起始位姿):

$$M_{Robot}^{Wrist} = T_{Robot}^{World} \cdot M_{World}^{C_1} \cdot M_{C_1}^{C_t} \cdot M_{C_t}^{wrist}$$

**用大白话说**:要把"人的手在世界里怎么动"搬到机器人坐标系,得先把移动相机的自身运动扣掉(SLAM),再把整个视频"摆正"到和直立机器人一致的重力方向(用检测到的桌/地面法向当重力参照),最后缩放平移对齐机器人工作空间。整套只需 2D 人类视频,不依赖陀螺仪。

### 2.3 视觉先验与两流策略

视觉编码器 $E_\phi$ 采用 R3M(在 Ego4D 上用视觉-语言对齐 + 时序一致性损失训练的 ResNet-18)。策略把首帧场景图 $I$ 编码后,分别喂给**手腕**和**手**两个独立 NDP($f_{wrist}, f_{hand}$),即两流(two-stream)结构,以解耦手臂运动与手内抓握。训练损失为 L1:

$$\mathcal{L} = \sum_k \text{Loss}_{L1}\Big(\tau_R - \big[\,f_{hand}(E_\phi(I_k)),\; f_{wrist}(E_\phi(I_k))\,\big]\Big)$$

**用大白话说**:同一个抓握姿势可以用在很多不同位置的物体上,同一段手臂移动也可对应不同抓握,所以把"手臂去哪儿"和"手怎么抓"拆成两条支路分别学,泛化更好。

### 2.4 训练配置

每任务用 Epic Kitchens 中 500-3000 段同类人类视频做动作先验预训练,再用真机 120-175 条遥操演示微调。全流程约 10 小时 / 单张 2080Ti。网络为 R3M-ResNet18 + 3 层 MLP(隐层 512)+ 2 个 NDP。硬件用 16-DoF LEAP 手 + xArm6;另在 1-DoF xArm 二指夹爪上单独训练动作先验做对比。

## 三、实验结果

7 个真机灵巧任务(pick / rotate / open / cover / uncover / place / push),报告 0-1 成功率,分训练物体 / 测试(未见)物体。

**主结果(Table 1,train / test):**

| 方法 | Pick | Rotate | Open | Cover | Uncover | Place | Push |
|---|---|---|---|---|---|---|---|
| BC-NDP | 0.64 / 0.38 | 0.94 / 0.56 | 0.90 / 0.60 | 0.78 / 0.58 | 0.88 / 0.82 | 0.70 / 0.35 | 1.00 / 0.71 |
| BC-Open | 0.50 / 0.44 | 0.72 / 0.38 | 0.80 / 0.40 | 0.44 / 0.58 | 1.00 / 0.91 | 0.40 / 0.25 | 1.00 / 0.93 |
| BC-RNN | 0.56 / 0.31 | 0.78 / 0.50 | 0.90 / 0.50 | 0.56 / 0.42 | 0.88 / 0.75 | 0.70 / 0.50 | 1.00 / 1.00 |
| **VideoDex** | **0.83 / 0.77** | 0.85 / 0.71 | 0.80 / 0.80 | 0.75 / 0.63 | 0.96 / 0.92 | **0.89 / 0.80** | 1.00 / 1.00 |

在**未见物体** 上 VideoDex 优势最明显(如 Pick 测试 0.77 vs 基线 ≤0.44、Place 测试 0.80 vs ≤0.50),说明人类视频动作先验带来的泛化增益是关键。

**place 任务消融(Table 4,Train / Test):**

| 组别 | 方法 | Train | Test |
|---|---|---|---|
| 基线 | BC-NDP | 0.70 | 0.35 |
| | BC-Open | 0.40 | 0.25 |
| | BC-RNN | 0.70 | 0.50 |
| | CQL(离线 RL) | 0.40 | 0.20 |
| 去物理先验 | VideoDex-BC-Open | 0.50 | 0.50 |
| | VideoDex-Single(单流) | 0.50 | 0.30 |
| 视觉先验替换 | VideoDex-VGG | 0.20 | 0.20 |
| | VideoDex-MVP | 0.40 | 0.20 |
| 少数据 | VideoDex-Const-5(每变体5条) | 0.80 | 0.60 |
| | VideoDex-Const-10 | 0.50 | 0.30 |
| — | **VideoDex(完整)** | **0.90** | **0.70** |

要点:① **动作先验 > 视觉先验**——带动作先验的 VideoDex-BC-Open(0.50/0.50)明显强于 BC-Open(0.40/0.25);视觉编码器 R3M > MVP > VGG。② **物理先验(NDP)通常有帮助但非每任务都提升**。③ **两流结构** 优于单流(解耦手臂与手)。④ 即便每变体仅 5 条演示,未见物体仍有约 30%(0.60)成功,样本效率优于纯 BC。⑤ 离线 RL(CQL)表现最差,执行不平滑、不安全。

**1-DoF 二指夹爪(Table 2,Place/Open/Pick):** VideoDex 0.69 / 0.82 / 0.77 均高于 1-DOF BC-Open 0.62 / 0.69 / 0.71——闭合人手→闭合夹爪的手腕轨迹先验对普通夹爪同样有效。

**初始 pitch 估计消融(Table 3,测试物体 Place/Cover/Uncover):** 用物体表面法向估重力的 VideoDex(0.80/0.63/0.92)优于固定值 VideoDex-Fixed(0.55/0.50/0.77)、随机 VideoDex-Random(0.45/0.63/0.85)与用 IMU 的 VideoDex-IMU(0.70/0.67/0.90);作者推测 IMU 传感器噪声大,从视觉估法向更稳。此外 LEAP 手比 Allegro 手平均高 7-12%。

## 四、局限性

- **仅用精选数据集**:虽声称面向"互联网视频",实际只用了带动作标注的 Epic Kitchens 以加速开发;真正从野生视频筛选任务片段是未来工作。
- **依赖现成人手检测**:FrankMocap 等模块在手-物交互时 6D 姿态检测常出错,是噪声主来源。
- **重定向需按本体重算**:动作先验绑定于特定机械臂轨迹 + 手部重定向,换机器人本体就要整套重算,迁移成本高。
- **开环、不可反应**:整套是开环行为克隆,无法对环境变化做出反应;而闭环 BC 在真机难以保证安全、闭环 RL 更难保证安全,故本文回避。
- 手腕重定向中"缩放系数取 1.0、启发式重缩放到工作空间"等简化,以及依赖单目 SLAM/深度估计,链条较长,任一环节失败都会污染动作先验。

## 五、评价与展望

**优点**:① 明确把"从视频学视觉表示"推进到"从视频学动作先验",并给出可复现的人手→机器人手臂完整重定向管线,这是本文最具工程价值的贡献——尤其是用移动第一视角相机(SLAM 补偿相机自运动 + 视觉估重力对齐直立)把人手世界轨迹搬到机器人系。② 用 NDP/DMP 作为物理先验,把含噪人手检测约束成平滑安全轨迹,是应对"人类动作标签很脏"的务实设计。③ 消融清晰地论证了 action prior 比 visual prior 更重要,对后续"人类数据用途"之争是有力经验证据。

**缺点与开放问题**:① 开环执行是根本短板,无法闭环纠错,面对物体扰动/接触不确定性(灵巧操作恰恰需要)天花板明显;后续 human-to-robot 方向(如同作者组的 Human-to-Robot Imitation in the Wild、以及 MimicPlay、DexMV/DexVIP 等)与近年的闭环 VLA 都在补这块。② 每任务仍需上百条真机演示微调,谈不上"零/少样本迁移";动作先验主要起到"初始化/引导到正确区域"的作用,而非直接可用策略。③ 重定向管线与本体强耦合、依赖多个易错的现成模块,规模化到多本体/大规模视频尚有距离。④ 评测规模偏小(每任务几十次 rollout),数值方差不容忽视。

**与公开工作的关系**:视觉侧建立在 R3M、MVP 之上;重定向借用 Robotic Telekinesis 与 DexPilot 的能量函数思路;物理先验来自作者组自己的 Neural Dynamic Policies。相较 DexMV(在仿真中用人手视频做灵巧模仿)与 DexVIP(学抓握 affordance 先验做 RL 初始化),VideoDex 的差异在于直接把视频当"伪机器人经验"预训练**真机开环策略**并覆盖多任务。可能的改进方向:引入闭环/触觉反馈以突破开环上限;用更强的手-物姿态估计降低动作先验噪声;探索跨本体共享的动作先验表示以摊薄重定向成本。

## 参考

1. S. Bahl, M. Mukadam, A. Gupta, D. Pathak. *Neural Dynamic Policies for End-to-End Sensorimotor Learning*. NeurIPS 2020.（物理先验 NDP 来源）
2. S. Nair, A. Rajeswaran, V. Kumar, C. Finn, A. Gupta. *R3M: A Universal Visual Representation for Robot Manipulation*. arXiv:2203.12601, 2022.（视觉先验编码器）
3. A. Sivakumar 等. *Robotic Telekinesis: Learning a Robotic Hand Imitator by Watching Humans on YouTube*. RSS 2022（手部重定向能量函数）。
4. D. Damen 等. *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset*. ECCV 2018.（人类视频数据源）
5. Y. Qin, Y.-H. Wu, S. Liu 等. *DexMV: Imitation Learning for Dexterous Manipulation from Human Videos*. arXiv:2108.05877, 2021.（相关灵巧模仿工作）
