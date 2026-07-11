# EgoEngine：从第一视角人类视频到高保真灵巧机器人示教

> **论文**：*EgoEngine: From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations*
>
> **作者**：Yangcen Liu, Shuo Cheng, Xinchen Yin, Woo Chul Shin, Alfred Cueva, Yiran Yang, Zhenyang Chen, Chuye Zhang, Danfei Xu
>
> **机构**：Georgia Institute of Technology；Tsinghua University
>
> **发布时间**：2026 年 06 月（arXiv 2606.12604）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.12604) | [PDF](https://arxiv.org/pdf/2606.12604)
>
> **分类标签**：`human-to-robot` `数据合成` `灵巧操作` `real2sim2real` `zero-shot`

---

## 一句话总结

EgoEngine 是一个把第一视角人类操作视频批量转成机器人示教的**双分支数据引擎**：视觉分支把画面中的人换成机器人、动作分支通过物体中心轨迹优化生成可执行动作,并用 MCTS 式的 Replay/MPC/RL 自适应模式切换控制求解成本;在四个真机灵巧任务上实现了**无任何真机示教**的零样本策略(平均 SR 0.51,Hammer 任务 0.60 反超真机遥操作的 0.25),同时动作生成成本比纯 RL 精修低约 22%。

## 一、问题与动机

灵巧操作(dexterous manipulation)受限于大规模机器人示教的采集成本:真机遥操作贵、接口复杂,又要处理高自由度的接触密集控制,导致很多灵巧策略只能在窄分布的机器人数据上训练。第一视角人类视频(egocentric human videos)天然丰富、覆盖多样场景与接触行为,Aria 眼镜等可穿戴设备更让高保真采集变得可扩展,是一个诱人的监督来源。

但**人类视频不是机器人示教**,直接使用要跨越两道鸿沟:

- **视觉鸿沟(visual gap)**:画面里是人的手臂和手,遮挡场景且与机器人本体外观差异巨大;
- **动作鸿沟(action gap)**:人与机器人在形态、运动学、驱动方式、接触动力学上都不同,直接把人手轨迹重定向到机器人往往物理上不可执行。

已有工作(EgoMimic、Mimicplay 等)大多把人类视频当作预训练/协同训练的辅助信号,但下游策略仍主要依赖真机遥操作示教。作者主张的核心是:**把人类视频直接转成既有机器人视角观测、又有可执行机器人动作的成对示教**,从而彻底摆脱真机示教。

## 二、核心方法

给定一段第一视角人类 RGB 视频,EgoEngine 先重建一个**物体中心的数字孪生(digital twin)**,再跑两条并行分支——动作分支把人手运动转成可执行机器人动作,视觉分支把人类画面转成机器人视角观测——最终产出成对示教 $\langle \tilde{o}_t, \tilde{a}_t \rangle$ 用于训练下游视觉运动策略。

### 2.1 人类视频到仿真(数字孪生)

用 Aria Gen2 眼镜采集,提供同步 RGB 帧和逐帧 3D 手部姿态(21 个手部关键点)。在此基础上:用 FoundationStereo 从 RGB 估计绝对深度图;用 SAM2 生成两类 mask——手部关键点 prompt 得到"手臂-手"的 mask 用于抹除演示者,首帧点 prompt 追踪任务物体的 mask;用 FoundationPose 估计时序一致的 6D 物体轨迹 $\{T_o^t\}_{t=1}^T$。相机几何、深度、mask、手姿、物体网格与物体轨迹一起构成数字孪生,供两条分支消费。

**用大白话说**:先把这段人类视频"逆向工程"成一个可仿真的小世界——知道相机在哪、物体是什么形状、每一帧物体在哪、人的手在哪,后面才能在这个世界里换上机器人重演一遍。

### 2.2 动作生成

**(a) 人类中心重定向(Human-Centric Retargeting)。** 先把人手运动重定向成一条机器人参考轨迹。给定 5 根指尖的位姿 $\{(p_{\text{tip},k}^t, R_{\text{tip},k}^t)\}_{k=1}^5$ 和手腕朝向 $R_{\text{wrist}}^t$,用 MINK 求解逆运动学:

$$
q_t^* = \arg\min_{q\in\mathcal{Q}} \mathcal{L}_{\text{tip}}(q;t) + \lambda_w \mathcal{L}_{\text{wrist}}(q;t)
$$

其中 $\mathcal{L}_{\text{tip}}$、$\mathcal{L}_{\text{wrist}}$ 是把机器人指尖对齐人指尖、机器人手腕对齐人手腕朝向的 L2 损失,$\mathcal{Q}$ 是受关节限位与自碰撞约束的可行构型空间。得到的 $\tau^{\text{ref}} = \{q_t^*\}_{t=1}^T$ 模仿了人的运动,作为后续精修的初始运动先验。

**用大白话说**:先让机器人手"照着人手的样子摆",指尖对指尖、手腕对手腕,得到一条外形上像样的初始轨迹,但还不保证真能干成活。

**(b) 物体中心轨迹优化(Object-Centric Trajectory Optimization)。** 仅靠重定向不够,原因有二:(1) 直接 replay 常因本体差异(运动学错配、接触动力学差异)而失败;(2) 人类视频给的是可观测的**本体感知轨迹**而非动作指令,存在 proprio-to-action 鸿沟(要施力、维持接触需要真正的动作命令)。于是在仿真中以**物体运动**为任务级目标精修轨迹。定义物体位姿追踪误差:

$$
e^t = \sqrt{\lambda_p\, d_p\!\big(\text{trans}(\hat{T}_o^t), \text{trans}(T_o^t)\big)^2 + \lambda_R\, d_R\!\big(\text{rot}(\hat{T}_o^t), \text{rot}(T_o^t)\big)^2}
$$

$d_p$ 是 $\mathbb{R}^3$ 上的欧氏距离,$d_R$ 是 $SO(3)$ 上的测地距离;$T_o^t$ 为人类视频里追踪到的物体位姿,$\hat{T}_o^t$ 为仿真中机器人执行后的物体位姿。采用带**早停**的阈值化追踪目标:一旦 $e^t$ 超过阈值 $C$ 就终止 episode;在有效区间内 $(e^t \le C)$ 奖励为 $r_{\text{obj}}^t = C - e^t$,误差越小奖励越高。再叠加接触、动作平滑、human-mimetic 正则等辅助项。

**用大白话说**:不管机器人手长得像不像人手,只要它能把**物体推到/搬到人视频里那个物体该去的地方**,就算干对了——用物体的运动而不是手的运动当"标准答案",绕开了本体差异。

**(c) 三种求解器 + MCTS 式自适应模式切换。** 把长时程轨迹切成若干时间 chunk,逐块渐进优化。每块可用三种能力递增的求解器:

- **Replay**:直接执行参考轨迹,最快;
- **MPC**:在参考轨迹附近搜索短时程动作样本,做局部修正,中等成本;
- **RL**:为困难 chunk 训练一个**手部残差策略**,观测手状态、物体位姿和参考重定向指令,输出残差 $\delta a_t \sim \pi_\phi(\cdot \mid s_t)$,最终动作 $a_t = a_t^{\text{base}} + \delta a_t$,用 PPO 优化。

所谓 **MCTS-style** 是一种轻量启发式的渐进搜索(并非完整 MCTS):每块都从 Replay 起步,只有当当前模式无法产生可行/足够改进的 rollout 时才升级到 MPC,MPC 仍不够才回退到 RL。为避免逐块孤立优化陷入局部最优,用**两块联合优化窗口**(同时求解当前块和下一块,但只执行当前块),既省算力又免去维护全轨迹 RL 策略。

**用大白话说**:能用最便宜的办法解决就绝不上贵的——大部分简单片段 Replay/MPC 就搞定,只有接触密集的难片段才动用 RL,把算力花在刀刃上。

### 2.3 视觉生成

**人类抹除。** 用 SAM2 的手臂-手 mask 圈出人的区域,用 Inpaint-Anything v2 修补,恢复被演示者遮挡的场景与物体内容,得到无演示者帧 $\tilde{I}_t$。

**遮挡感知融合(Occlusion-aware blending)。** 在第一视角渲染机器人得到 $R_t$,通过**两遍差分渲染**恢复遮挡感知 mask:一遍物体几何不透明、机器人全透明得到 $I_{\text{bg}}^t$,一遍机器人不透明得到 $I_{\text{rob}}^t$;逐像素计算

$$
\tilde{M}_r^t(p) = \mathbb{1}\!\left[\, \| I_{\text{rob}}^t(p) - I_{\text{bg}}^t(p)\| > 0 \,\right]
$$

由于两遍都含物体,被物体遮挡的机器人像素会被自动剔除。最终观测:

$$
\tilde{o}_t^{(r)} = \tilde{M}_r^t \odot R_t + (1 - \tilde{M}_r^t) \odot \tilde{I}_t
$$

**用大白话说**:先把画面里的人"擦掉"并补全背景,再把渲染的机器人贴回去;贴的时候还专门算清楚哪些机器人像素被物体挡住了,避免机器人"穿模"盖在本该在前面的物体上。

### 2.4 策略蒸馏

每段人类视频被转成一条(同步观测+动作)机器人示教,聚合成合成数据集 $\mathcal{D}_{\text{robot}} = \{(\tilde{o}, \tilde{a})\}$($\tilde{o}$ 含本体感知),用 HPT 训练视觉运动策略 $\pi_\theta$,损失为 L2 动作回归:

$$
\min_\theta \; \mathbb{E}_{(\tilde{o},\tilde{a})\sim\mathcal{D}_{\text{robot}}} \big[\, \|\pi_\theta(\tilde{o}) - \tilde{a}\|_2^2 \,\big]
$$

两条分支被蒸馏进一个闭环控制器,从机器人观测映射到动作。

## 三、实验结果

**设置。** 两个数据源:TACO(2,500 段视频序列)用于视觉与动作评测;自采 Aria 数据集用 Aria Gen2 眼镜采 200 段真实第一视角人类视频,覆盖四个任务;另外采 200 条真机遥操作示教作对比(EgoEngine 本身不用)。仿真用双臂 RB-Y1(两条 7-DoF 臂 + 两只 12-DoF XHand);真机用单臂 RB-Y1 + 一只 XHand。仿真评测取 TACO 中 16 对示教;真机四任务为 Mustard(放芥末瓶到目标盘)、Drawer(开抽屉放方块)、Hammer(拿锤子敲钉)、Flower(拿水瓶浇花),覆盖抓放、工具使用、精细接触。

**视觉保真(Table 1,Fréchet Distance,越低越好)。**

| 方法 (FD↓) | ResNet18 | VGG16 | DINOv2 |
|---|---|---|---|
| Human Video | 764.5 | 670.2 | 602.9 |
| EgoMimic | 830.5 | 812.1 | 579.6 |
| VACE (WAN2.1) | 713.6 | 745.3 | 488.0 |
| Phantom | 620.0 | 650.8 | 470.6 |
| **EgoEngine** | **614.7** | **644.2** | 473.1 |

EgoEngine 在 ResNet18/VGG16 上与真机观测特征距离最近;DINOv2 上与 Phantom 基本持平(473.1 vs 470.6)。

**动作保真(Table 2,SR/Step/Reward 越高越好,Cost 越低越好)。** 对比 Mink(Replay)、Spider(MPC)、H2S2R(RL)。

| 方法 | TACO SR | TACO Reward | TACO Cost | Aria SR | Aria Reward | Aria Cost |
|---|---|---|---|---|---|---|
| Mink (Replay) | 0.17 | 0.29 | 1.00 | 0.10 | 0.62 | 1.00 |
| Spider (MPC) | 0.25 | 0.39 | 7,923 | 0.20 | 0.65 | 4,382 |
| H2S2R (RL) | **0.83** | **0.70** | 73,675 | **0.90** | **0.85** | 20,237 |
| **EgoEngine** | **0.83** | 0.67 | **34,842** | **0.90** | 0.83 | **16,560** |

EgoEngine 在成功率上追平最强的纯 RL,但成本几乎减半(TACO 34,842 vs 73,675;Aria 16,560 vs 20,237)。在 Aria 上生成效率提升 **22.0%**,单张 RTX 4090(无并行)从 RL 的 2.36 demos/hour 提到 2.88 demos/hour;TACO 因是双臂长时程数据(平均 327.5 步,约为 Aria 的 2.39 倍),效率收益更大。

**下游策略蒸馏(Table 3,真机四任务 SR)。**

| 方法 | Mustard | Drawer | Flower | Hammer |
|---|---|---|---|---|
| Human Video(直接重定向) | 0.00 | 0.10 | 0.00 | 0.00 |
| Phantom | 0.00 | 0.05 | 0.00 | 0.00 |
| Real Robot(真机遥操作) | **0.80** | **0.80** | 0.70 | 0.25 |
| **EgoEngine** | 0.40 | 0.35 | **0.70** | **0.60** |

Human Video 与 Phantom 几乎全线归零(主要卡在抓取姿态),说明**仅补视觉鸿沟不足以支撑灵巧策略**;EgoEngine 首次实现无真机示教的零样本真机灵巧操作,且在 Flower(0.70 持平)与 Hammer(0.60 反超真机 0.25)两个任务上匹配或超过真机示教。

**消融(Table 4,四任务平均 SR)。**

| 配置 | SR↑ |
|---|---|
| Human Videos | 0.03 |
| + Visual branch | 0.05 |
| + Action branch | 0.43 |
| **EgoEngine(全)** | **0.51** |

去掉动作分支导致最大性能崩塌(仅剩 0.05),证明**可执行动作生成是下游性能的首要因素**;视觉分支带来额外增益(0.43→0.51),与"策略以物体为中心、可容忍中等本体外观差异"的既有观察一致。

## 四、局限性

作者自陈三条:

1. **质量**:视觉分支是基于融合(blending)的合成而非完全学习式的照片级真实;动作生成仍受接触建模误差与 sim-to-real gap 影响。
2. **可扩展性**:数字孪生的构建是瓶颈——高质量物体资产获取、严重遮挡下的物体状态估计、可形变物体的处理都仍困难。
3. **效率**:基于仿真的轨迹优化在超大规模下仍慢(尽管轨迹可并行),未来可用预训练模型加速。

## 五、评价与展望

**优点。** (1) 问题拆解干净:把 human-to-robot 明确分解为"视觉鸿沟 + 动作鸿沟"两条正交分支,并用消融量化证明动作分支才是下游关键——这个结论对整条 learn-from-human 路线有校准价值,反驳了"只做视觉改写就够"的乐观假设。(2) **物体中心目标**是全文的核心巧思:用物体运动而非人手运动当监督,天然绕过本体形态差异,是把不可执行的人手轨迹变成可执行机器人动作的关键锚点。(3) **MCTS 式自适应模式切换**在工程上很实用——按 chunk 难度动态分配 Replay/MPC/RL,拿到接近纯 RL 的成功率却省一半算力,对大规模数据生成的成本敏感场景有直接意义。(4) 真机结果扎实:零真机示教下 Hammer 反超真机遥操作,是本文最有说服力的证据。

**缺点与开放问题。** (1) 规模仍偏小:真机只测 4 个任务、每任务 200 段视频,离"可扩展数据引擎"的宣称还有距离,数字孪生瓶颈让人怀疑能否真正 scale 到数千任务。(2) 强依赖一套现成感知栈(FoundationStereo/SAM2/FoundationPose),任一环节在遮挡/透明/可形变物体上失效都会级联污染整条示教,论文未量化这种误差传播。(3) 视觉分支是渲染+融合而非生成式,真实感受限于机器人资产与渲染质量,与近期物理/运动条件视频生成(如 VACE 这类基于 WAN2.1 的方案)相比,泛化到复杂光照/新场景的能力存疑。(4) 动作侧残余失败集中在不稳定抓取(pinch grasp)与接触时机错配(Table 3 里 Mustard/Drawer 仅 0.40/0.35),说明物体位姿追踪奖励对**接触力/时序**的约束仍不足。

**与公开工作的关系。** 相比 EgoMimic、EgoBridge、Mimicplay 等把人类视频当预训练/协同训练信号、仍需真机示教作目标域的做法,EgoEngine 走的是**完全把人类视频转成成对机器人示教**的更激进路线,与 EgoZero、Zeromimic 的零样本方向同属一脉,但把设定推到了高自由度灵巧手 + 接触密集任务。视觉侧沿用 Phantom 那类 inpainting-rendering-blending 管线并进一步用数字孪生增强物理一致性;动作侧则与 Real2Sim2Real 的 MPC/RL 精修工作(如 H2S2R 这类残差 RL)相承,创新点在于**自适应地在多种求解器间切换**而非一刀切上最强求解器。

**可能的改进方向。** 用生成式视频模型替代融合式渲染以提升真实感与泛化;在物体中心奖励里显式引入接触力/接触时序项以救回 pinch grasp 与时序错配的失败;把感知栈的不确定性反馈进轨迹优化(遮挡区域降权);以及探索无需逐视频重建数字孪生的更轻量 real2sim 路径以突破可扩展性瓶颈。

## 参考

1. Kareer et al. *EgoMimic: Scaling Imitation Learning via Egocentric Video*. arXiv:2410.24221, 2024.(视觉分支基线与 learn-from-human 直接对标)
2. Lepert, Fang, Bohg. *Phantom: Training Robots without Robots using only Human Videos*. CoRL 2025.(最接近的视觉改写基线)
3. Liu et al. *EgoZero: Robot Learning from Smart Glasses*. arXiv:2505.20290, 2025.(零样本 human-to-robot 同路线)
4. Wang et al. *Mimicplay: Long-horizon Imitation Learning by Watching Human Play*. arXiv:2302.12422, 2023.(人类视频用于高层策略规划)
5. Wang et al. *HPT: Heterogeneous Pre-trained Transformers*(下游策略蒸馏骨干)。
