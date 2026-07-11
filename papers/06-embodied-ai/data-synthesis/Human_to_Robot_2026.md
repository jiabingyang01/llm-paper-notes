# Human-to-Robot：从视频演示学习实现机器人模仿

> **论文**：*Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation*
>
> **作者**：Thanh Nguyen Canh, Thanh-Tuan Tran（共同一作）, Haolan Zhang, Ziyan Gao, Nak Young Chong, Xiem HoangVan（通讯）
>
> **机构**：日本北陆先端科学技术大学院大学（JAIST）信息科学学院；越南国家大学河内工程技术大学（VNU-UET）；韩国汉阳大学机器人系
>
> **发布时间**：2026 年 02 月（arXiv 2602.19184）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.19184) | [PDF](https://arxiv.org/pdf/2602.19184)
>
> **分类标签**：`human-to-robot` `video-to-command` `TSM+TD3` `imitation-learning`

---

## 一句话总结

把"从人类视频学操作"这件事**显式解耦** 成两级流水线——先用 TSM+VLM 把无结构演示视频翻译成"动作+物体"的无语法执行指令(video-to-command),再用 TD3+分层奖励的强化学习在关节空间把指令落地成机器人动作;在改造版 Something-Something V2 上动作分类 Top-1 达 89.97%、标准物体 BLEU-4 达 0.351(相对最强基线 +76.4%)、新物体零样本 BLEU-4 0.265(+128.4%),仿真四动作平均成功率 87.5%,真机 reach/pick 分别 100%/80%。

## 一、问题与动机

从演示学习(Learning from Demonstration, LfD)是机器人技能获取的重要范式,但作者指出既有做法各有瓶颈:

- **传统 LfD**(动觉示教 Kinesthetic Teaching、遥操作 Teleoperation、被动观测 MoCap)依赖专用硬件(动捕服、力反馈设备),难以规模化、难以覆盖多样任务,且存在人-机身体不匹配的 correspondence problem(embodiment mapping)。
- **video-to-command 端到端方法**:通用视频描述模型(video captioning)偏向全局场景特征、忽略被操作物体,精度不足以驱动机器人;早期 LSTM 架构因描述重复性高而过拟合、泛化差;即便拿到正确文本指令,把"scoop""fold"这类语义映射到机器人自身身体的亚厘米级空间动作(embodiment mapping / affordance)仍很难。

作者的核心主张:人类是**先"看懂"(watching)再"模仿"(imitating)**,把"理解演示"与"学习技能"这两件事分开,比在单一端到端框架里直接把视觉映射到策略更鲁棒、更易泛化。由此提出模块化 "Human-to-Robot" 流水线,两级分别对应 "What"(人在做什么)与 "How"(机器人怎么做)。

## 二、核心方法

整体是两阶段架构:**阶段一 Video Understanding**(把视频 $\mathcal{V}$ 翻译成结构化指令 $\mathcal{S}$),**阶段二 Robot Imitation**(用 DRL 把指令执行出来)。

### 2.1 阶段一:Video Understanding(video-to-command)

原始视频帧序列 $\mathbf{F}=\{\mathbf{f}_1,\dots,\mathbf{f}_n\}$ 先降采样到更小的子集 $\hat{\mathbf{F}}=\{\hat{\mathbf{f}}_1,\dots,\hat{\mathbf{f}}_m\}\ (m<n)$,再兵分两路并行:

**(a) Action Understanding Module（AUM）——识别动作。** 用预训练 CNN(ResNet backbone)离线抽帧特征 $\mathbf{X}=\{\mathbf{x}_1,\dots,\mathbf{x}_m\}$,并在残差分支内插入 **Temporal Shift Module (TSM)**:沿时间维随机移位一部分特征通道,得到时移特征 $\tilde{\mathbf{X}}$,让 2D CNN 以近乎零额外参数/算力获得"伪 3D"的时空建模能力,用于区分 10 类细粒度操作动作(如 open vs. close)。TSM 必须放在残差分支内(residual shift),否则 naive in-place 移位会破坏 2D backbone 的空间特征。

用大白话说:把相邻帧的一部分"记忆"沿时间轴前后串一串,一个便宜的 2D 网络就能"看出"动作的时间演化,而不用上昂贵的 3D 卷积。

**(b) Interacted-Objects Understanding Module（IOUM）——识别被交互物体。** 先做关键帧抽取,只保留"发生了显著运动"的帧:

$$
\tilde{\mathbf{F}}_k = T_G\big(\mathbf{f}_s(\mathbf{f}_t(\hat{\mathbf{F}}))\big)
$$

其中 $T_G$ 为灰度变换,$\mathbf{f}_s$ 为相邻帧相减($\hat{\mathbf{f}}_i-\hat{\mathbf{f}}_{i-1}$,假定亮度变化不显著),$\mathbf{f}_t$ 为阈值滤波(只留显著视觉变化)。关键帧送入 HOI(human-object interaction)算法检测与手交互的物体,经 2D conv + ROI pooling 得到高置信物体特征集 $\mathbf{I}=\{\mathbf{i}_1,\dots,\mathbf{i}_h\}$。

随后 **Object Selection 算法**:用 tracker 跨帧追踪物体质心轨迹,依据运动模式(抛物/振荡轨迹)把物体分成 **Pickable**($\mathbf{P}_1$,被拿起的)与 **Placeable**($\mathbf{P}_2$,作为放置目标的)两组;再用两个滤波器精选质量最高的实例送入 VLM——一是**模糊度检测**(Laplacian 卷积),二是**手-物重叠最小化**:

$$
\mathbf{B}(i,j) = \sum_{k=-1}^{1}\sum_{l=-1}^{1}\mathbf{i}_t(i+k,\,j+l)\cdot \mathbf{L}(k+1,\,l+1)
$$

其中 $\mathbf{L}=\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$ 为拉普拉斯核,$3\times3$ 窗口滑动。用大白话说:选那些**最清晰、又没被手挡住**的物体截图交给 VLM,避免模糊/遮挡把物体识别带偏。最后 VLM 负责物体类别识别与零样本泛化(对训练时没见过的新物体也能命名)。

**(c) Captioning Scheme——生成指令。**融合 AUM 的动作向量 $\mathcal{A}_i$ 与 IOUM 的物体类别向量 $\mathcal{I}_i$,经 softmax 得到最终指令:

$$
\mathcal{S}_i = \mathcal{A}_i + \mathcal{I}_i,\quad i=1,\dots,m
$$

刻意强制**无语法(grammar-free)、面向执行** 的紧凑格式(如 "place apple on plate"、"Picking the blue block up"),而非 "A man is picking the blue block up on the table" 这类自然语言长句,以消除歧义、便于下游执行(输出最多约 8 个词,不足补空词)。训练目标遵循标准 video captioning,最大化动作-物体对 $\mathcal{P}$ 的对数似然 $\theta=\arg\max_\theta\sum_{(\mathcal{V},\mathcal{P})}\log p(\mathcal{P}\mid \mathcal{V},\theta)$。

### 2.2 阶段二:Robot Imitation（TD3 + 分层奖励）

作者选择**在关节空间**(而非笛卡尔空间)直接控制,理由:更自然地刻画运动动力学、避开末端控制的运动学奇异、缩小语义指令到底层电机控制的鸿沟。采用 **TD3**(Twin Delayed DDPG)最大化累计折扣回报 $J(\pi_\theta)=\mathbb{E}\big[\sum_{t=0}^{\infty}\gamma^t r(\mathbf{s}_t,\mathbf{a}_t)\big]$。选 TD3 是因为:双 critic 缓解 Q 学习高估、确定性策略适配高维连续控制、延迟策略更新降方差(尤其配合强域随机化)。

**状态表示(Table 1,含域随机化)。** 观测 $\mathcal{O}$ 含本体感受(关节角 $\mathbf{j}_t$ 6D、关节速度 $\mathbf{j}_t'$ 6D)、空间关系(末端-物体相对位 $\bar{\mathbf{p}}_{eo}$、物体-目标相对位 $\bar{\mathbf{p}}_{og}$、物体位姿 $\mathbf{p}_o/\mathbf{R}_o$、目标位 $\mathbf{p}_g$)、交互态(吸盘二值读数 $\mathbf{p}_e^s$ 2D)、时序上下文(上一动作 $\mathbf{a}_{t-1}$ 7D)。除 $\mathbf{a}_{t-1}$ 外各分量加 $\mathcal{U}(\pm0.005)$ 噪声,保持 100% 随机化。用 Welford 在线算法做归一化。动作 $\mathbf{a}_t$ = 各关节连续速度 + 一个 Bernoulli 离散吸盘信号,关节位置由速度积分 $\mathbf{j}_t=\mathbf{j}_{t-1}+\mathbf{j}_t'\cdot dt$ 得到,再经底层 PD 控制器约束速度/关节限位。因状态转移依赖当前与上一动作 $P(\mathbf{s}_{t+1}\mid \mathbf{a}_t,\mathbf{s}_t,\mathbf{a}_{t-1})$,严格说超出标准 MDP。

**分层奖励(Table 2,核心贡献之一)。** 总奖励为 3 项引导奖励减 6 项安全惩罚:

$$
r_t(\mathbf{s}_t,\mathbf{a}_t)=\sum_{i=0}^{2}w_i\, r_t^i(\mathbf{s}_t,\mathbf{a}_t)-\sum_{j=1}^{6}w_j\, c_t^j(\mathbf{s}_t,\mathbf{a}_t)
$$

三项奖励:**Approach**($r_t^e$,指数距离塑形引导末端带正确姿态靠近物体)、**Interaction**($r_t^i$,由 `has_contact` 门控,接触后奖励把物体运向目标,$\lambda_3\gg\lambda_4$)、**Alignment**($r_t^a$,物体速度方向与"物体→目标"单位向量点积,仅在物体被移动 $\hat{\mathbf{p}}_o'\cdot\hat{\mathbf{p}}_{og}>0$ 时激活)。六项惩罚:碰撞(地面/自碰 3.0、撞物 1.5)、超步数、末端倾斜、动作抖动($\|\mathbf{a}_t-2\mathbf{a}_{t-1}+\mathbf{a}_{t-2}\|$ 二阶差分)、物体倾斜、越界。任务成功时冻结 $r_t^i$、置 $r_t^e=0$ 并给 $r_t^a$ 三倍满分。用大白话说:把"先靠近→抓住→朝目标搬→对准放下"拆成有序子目标各给一份塑形奖励,并用一堆安全惩罚保证过程不撞、不抖、不越界,从而在杂乱场景里学到亚厘米精度。

**训练(Algorithm 1)。** 两阶段:先对全部演示视频跑 Video Understanding 生成指令并填入回放缓冲;再跑 TD3 强化学习(critic 用双目标网络取 min 防高估、target policy smoothing 加裁剪噪声、actor 每 $d$ 步延迟更新 + 软更新)。每 episode 随机生成含 19 个物体的场景;reach 只需导航末端到目标位,pick/move/put 需完成"抓对物体 + 搬到指定目标"两个子目标;满足成功/碰撞/关节越限/越界/超时($T_{max}=100$)之一即终止。

## 三、实验结果

**实验设置。** 仿真:PyBullet + UR5e 机械臂 + 吸盘,$60\times60$ cm 桌面,RGB $640\times480$ 无传感器噪声;策略网络在 V100-SXM2-32GB 与 RTX 5080-16GB 上训练,35,000 episodes、约 350 万环境步、约 24 小时。视频理解数据:改造版 Something-Something V2,10 类动作,训练 118,562 段 / 测试 12,480 段,另采集 135 段自录视频作真实世界测试;物体识别分标准集(12 类:apple/pan/bottle/orange/kettle/egg/plate/box/spoon/spatula/knife/cup)与新物体集(9 类:carrot/block/grape/chilly/banana/pressure/lemon/pot/strawberry)。真机:6-DoF UF850 + 真空吸盘。评测 10 次独立试验。

**动作分类(TSM,Table 4)。** 8-frame × 3-crop 采样:

| 测试集 | Backbone | Top-1 Acc. | Top-5 Acc. |
|---|---|---|---|
| Test set 1（标准基准) | ResNet-101 | **89.97** | **96.82** |
| Test set 2（自录真实) | ResNet-101 | **71.11** | **95.56** |

标准集到真实自录集 Top-1 掉约 19 个点(域偏移),但 Top-5 始终 >94%,说明正确动作几乎总在前 5。

**视频→指令 BLEU(标准物体,Table 5;ResNet-101 backbone)。**

| 方法 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Video2Command | 0.319 | 0.183 | 0.175 | 0.148 |
| V2CNet | 0.324 | 0.163 | 0.153 | 0.131 |
| Watch-and-Act | 0.339 | 0.171 | 0.155 | 0.134 |
| Ours w/o keyframe | 0.577 | 0.465 | 0.363 | 0.298 |
| Ours w/o object sel. | 0.524 | 0.411 | 0.303 | 0.237 |
| **Ours** | **0.618** | **0.511** | **0.403** | **0.351** |

BLEU-4 0.351,相对最强基线(Watch-and-Act 用 InceptionV3 的 0.199)提升 **76.4%**。消融:去关键帧抽取 0.351→0.298($-15.1\%$),去物体选择 →0.237($-32.5\%$),说明物体选择更关键。

**视频→指令 BLEU(新物体零样本,Table 6;ResNet-101)。** Ours BLEU-4 **0.265** vs. Watch-and-Act 0.094(其 InceptionV3 最强基线 0.116),提升 **128.4%**;标准→新物体的退化幅度(24.5%)远小于 Watch-and-Act(41.7%),体现 VLM 带来的语义泛化。

**机器人操作成功率(Table 7,%,不同位置误差阈值)。**

| 动作 | 1 cm | 2 cm | 3 cm | 4 cm |
|---|---|---|---|---|
| Reach | 100 | 100 | 100 | 100 |
| Pick | 70 | 80 | 100 | 90 |
| Move | 90 | 80 | 70 | 90 |
| Put | 90 | 90 | 70 | 90 |

reach 全阈值 100%;pick 平均 87.5%;四动作总平均约 87.5%,即便 1 cm 严阈也保持较高成功率,说明学到的是精确操作而非"松标准蒙对"。DRL 对比:TD3 收敛最快、均值最高、方差最低——reach 约 500 episodes 收敛到正回报,复杂动作(put)约 2000 episodes 稳定为正;SAC 次优但 move 上方差大;on-policy 的 PPO/Asym-PPO 在强随机化环境里因样本效率低而垫底。

**真机(Table 8,UF850,10 次/动作)。**

| 动作 | 成功率(%) | 执行时间(s) | 位置精度(mm) |
|---|---|---|---|
| Reach | 100 | 30 ± 6 | 25 ± 6 |
| Pick | 80 | 33 ± 5.4 | 20 ± 4 |

真机在不同背景/光照下 reach 100%、pick 80%,验证了流水线从训练数据到真实场景的泛化。

## 四、局限性

- **动作词表窄、单臂**:机器人执行仅支持 reach/pick/move/put 四个基本原语、单臂,尚不能做双手或工具使用类复杂任务(视频理解侧虽标 10 类,但落地执行只有 4 类)。
- **仿真训练带来 sim-to-real gap**:策略在 PyBullet 训练,迁移到真机的差距需进一步研究;真机也只做了 reach/pick 两个动作的初步验证。
- **物体选择算法假设物体间视觉可分**:在高度遮挡场景下追踪/分组会失效。
- 未与端到端 VLA(如 diffusion/transformer 策略)在同一操作基准上正面对比;BLEU 作为"指令正确性"代理指标,与真实执行成功率之间并非严格对应;成功率报告基于 10 次试验,统计置信区间较宽。

## 五、评价与展望

**优点。**(1) "看懂-模仿"两级解耦的工程主张清晰,规避了成对人-机数据与显式运动学对应,这与 video-to-command 一脉(Nguyen 等 Video2Command/V2CNet、Yang 等 Watch-and-Act)相承而在物体侧显著加强:用轨迹分组 + 模糊/遮挡过滤 + VLM,把"物体识别"从端到端策略里拆出来单独做,换来了对新物体的可观零样本泛化,这是对既有 LSTM/Mask R-CNN 路线(如 Watch-and-Act 的 Visual Change Maps)的合理改进。(2) 分层奖励设计细致,把长程操作拆成有序子目标并配安全惩罚,是本文可复现性较好的部分。(3) TSM+TD3 的组合务实、算力友好,并给出真机验证。

**缺点与开放问题。**(1) 技术上偏"成熟组件的系统集成"(TSM、TD3、VLM 均为现成),新意主要在流水线组织与物体选择启发式,方法论新颖度有限。(2) 与当下主流的端到端 VLA / 大规模模仿学习(如 OpenVLA、RT-2 一类)缺乏对照,难判断解耦范式相对"数据驱动端到端"的真实优劣;分类词表仅 10 类、执行 4 类,离开放世界操作尚远。(3) 关节空间 RL + 分层奖励虽精度高,但奖励项与权重(Table 2 手工设定的十余个 $w,\lambda$)工程调参负担重、跨任务迁移性存疑;域随机化下的仿真策略真机泛化仅在两动作上小样本验证。(4) BLEU 评估语义指令、成功率评估执行,两者未打通成端到端"看视频→做对任务"的闭环成功率,是最值得补的实验。

**可能改进方向。** 把 VLM 升级为可直接输出可执行结构化指令的 VLA 骨干、以更大动作词表与更强遮挡鲁棒的物体分割替换启发式选择、以真机数据做域适应或离线 RL 微调收窄 sim-to-real、引入 LLM 做多步任务规划以支持"高层自然语言→多步操作序列"。作者在结论中亦点出扩展动作(pouring/stirring/folding)、加入触觉/力控、域适应真机部署、LLM 任务规划等后续方向。

## 参考

1. Yang, S. et al. *Watch and Act: Learning Robotic Manipulation from Visual Demonstration.* IEEE T-SMC 53(7), 2023. —— 最直接的 video-to-command 基线,Visual Change Maps + Mask R-CNN。
2. Nguyen, A. et al. *Translating Videos to Commands for Robotic Manipulation with Deep Recurrent Neural Networks (Video2Command).* ICRA 2018. —— video-to-command 早期代表。
3. Nguyen, A. et al. *V2CNet: A Deep Learning Framework to Translate Videos to Commands.* arXiv:1903.10869, 2019. —— 增加动作分类分支的改进。
4. Lin, J. et al. *TSM: Temporal Shift Module for Efficient Video Understanding.* ICCV 2019. —— 本文动作理解模块的核心时序建模组件。
5. Fujimoto, S. et al. *Addressing Function Approximation Error in Actor-Critic Methods (TD3).* ICML 2018. —— 本文机器人模仿阶段采用的 DRL 算法。
