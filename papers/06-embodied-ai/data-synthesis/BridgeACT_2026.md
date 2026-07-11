# BridgeACT：用统一的工具-目标可供性把人类演示桥接到机器人动作

> **论文**：*BridgeACT: Bridging Human Demonstrations to Robot Actions via Unified Tool-Target Affordances*
>
> **作者**：Yifan Han\*, Jianxiang Liu\*, Haoyu Zhang, Yuqi Gu, Yunhan Guo, Wenzhao Lian†（\* 共同一作,† 通讯作者）
>
> **机构**：中国科学院自动化研究所;上海交通大学人工智能学院;华南理工大学
>
> **发布时间**：2026 年 04 月（arXiv 2604.23249,v2 于 2026 年 5 月更新）
>
> **发表状态**：未录用（预印本,cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.23249) | [PDF](https://arxiv.org/pdf/2604.23249)
>
> **分类标签**：`从人类视频学习` `tool-target affordance` `3D motion flow` `diffusion` `无机器人演示数据`

---

## 一句话总结

BridgeACT 把机器人操作抽象成"抓哪里(where to grasp)"和"怎么动(how to move)"两个子问题,提出**可执行的工具-目标可供性(executable tool-target affordance)**——一种与本体无关、角色条件化的 3D 交互表征,仅用人类视频(无任何机器人演示数据)训练一个 diffusion 运动生成器,再经 SVD 刚性配准映射到 SE(3) 闭环控制;在 Franka+UMI 真机上 6 类操作任务达到 43/60 的成功率,显著超过 General Flow(31/60)、ReKep 与 Track2Act。

## 一、问题与动机

从人类视频学操作很有吸引力:相比机器人数据,人类视频采集便宜、规模大、场景与交互模式更丰富。但人类视频**不是** 机器人演示——人手、夹爪、灵巧手在运动学、接触策略、控制空间上都不同,直接迁移人手或物体的**绝对轨迹** 在数学上是病态(ill-posed)的。尤其在 egocentric(第一视角)视频里,相机运动与人头运动耦合,物体的绝对像素轨迹极不稳定。

作者的核心论断:人类演示里**可迁移的信号不是绝对运动,而是解释任务的交互结构**——谁是工具(tool)、谁是被作用的目标(target)、接触发生在哪里、工具相对目标如何运动。已有方法只捕获了这一结构的一部分:

- 很多机器人学习方法仍需机器人演示做策略训练或下游适配(RT-1、π0.5、OpenVLA 等),限制了从人类视频扩展的能力;
- 感知级 affordance(如 UAD)能定位"哪里可交互",但预测的视觉线索不能直接执行为 3D 机器人运动;
- flow-based 方法(General Flow、Track2Act、ATM 等)预测物体或点轨迹,但常常是 object-centric、只关注抓后(post-grasp)运动,不显式建模"谁作用于谁";
- VLM/planner 方法(ReKep、VLM-see-robot-do)产出符号任务计划或关键点约束,但不从演示里学**稠密的交互动力学**。

因此缺一个能**同时** 捕获任务相关接触区域、工具-目标角色、可执行 3D 运动的表征。

## 二、核心方法

BridgeACT 把一次操作用"工具作用于目标"的功能关系来参数化。角色定义在单物体与物体-物体(object-to-object, O2O)两种情形下统一:

- **单物体任务**:target = 被操作物体,tool = 执行者(训练时是人手,部署时是机器人夹爪);
- **O2O 任务**:tool = 交互物体(如刀),target = 被作用物体(如水果)。

一个可执行工具-目标可供性联合编码三要素:任务条件化的可操作区域(where)、功能性工具-目标角色(who acts on whom)、未来 3D 交互动力学(how)。与仅定位交互区域的感知级 affordance 不同,该表征锚定在**3D query anchors** 上并预测它们的角色条件化运动,天然可对接下游执行。整个 pipeline 分三块。

### (a) 从人类视频自动构造 motion point-flow 数据集

数据来源两类:大规模公开 HOI 数据集(HOI4D + Epic-Kitchens)与作者自采视频,**不用任何机器人数据**。公开集取 6 类 affordance:open、close、pickup、place、push、pull;自采数据补充 4 类:pour、press、hang-on、cut,共**10 类** affordance。图 1 标注公开集约 HOI4D 40K + Epic-Kitchen 10K(83%/17%)。

关键设计:**每个样本同时保留 tool 与 target 两组 query**,而不是只建模被操作物体的运动——因为 egocentric 视频里相机随人头晃动,单个物体的绝对运动不是稳定的监督目标;物理上有意义的信号是 tool 与 target 的**相对运动**。

预处理:原始视频切成 1.5 秒定长 clip,滑窗内下采样到 4 帧,相邻 clip 时间步长 0.5 秒。最终得到**超过 400 个场景、3,000 段视频、60K 个 clip**。三阶段处理:

1. **VLM 任务标注**:用 Qwen3-VL 对每个任务均匀采 10 帧,推断任务语义,抽取被操作实体与动作类别,统一成 tool-target 文本模板(单物体时以人手作默认执行者补全角色);
2. **物体级 mask 与轨迹**:有标注则直接用,否则用 SAM3 自动分割,mask 内均匀采点,用 CoTracker3 跨帧跟踪,得到 tool/target 的稠密 2D 轨迹;
3. **3D 重建与过滤**:有可靠 3D 标注/物体位姿则直接恢复 3D 轨迹,否则用估计深度+相机几何把 2D 轨迹 lift 到 3D。清洗步骤包括:把人手近似成夹爪状结构、去掉孤立点簇与漂浮离群点、只保留中心交互区的有效点集。

每个处理后样本表示为 $\{P_{scene}, Q_{tool}, Q_{target}, l\}$,其中 $P_{scene}$ 是场景点云,$Q_{tool}/Q_{target}$ 是从工具/目标点云采样的 object-centric query 点集,$l$ 是语言指令。

> 用大白话说:先让 VLM 看几帧读懂"用刀切水果",然后 SAM3 把刀和水果抠出来,CoTracker3 把它们表面的点在时间上追踪一遍,再抬到 3D——于是一段人类视频就变成了"工具点云怎么相对目标点云运动"的监督信号,全程没碰机器人。

### (b) 任务条件化的可供性 Grounding Agent

部署时需把学到的表征落到当前场景。该轻量模块把抽象/欠定的任务描述转成场景里的定位化 affordance 区域(而非直接产动作):(i) MLLM 先把原始指令规范成更明确的任务表述;(ii) 解析出物体描述、任务相关 affordance 区域描述、分割 prompt;(iii) 用 SAM3 按 prompt 分割候选区域,得 2D mask $M_{tool}, M_{target}$;(iv) **affordance 校验**——用 vision-language verifier 判断 mask 是否在功能上对该交互有效(而不只是跟文本视觉一致);(v) **一步恢复(one-step recovery)**——若首轮不准但可救,最多再做一次带更鲁棒 fallback prompt 的重定位,通常把不稳定的细粒度部件级描述放宽成物体级或更大空间区域。最终校验后的 mask 投影到场景点云得到 3D query 区域。

### (c) 与本体无关的可供性表征 + diffusion 运动生成

与把 query 点仅当作 latent condition 的常规点云管线不同,BridgeACT **把 tool/target query 全程当作显式几何锚点** 保留。两点动机:query 点既是输入上下文、又是预测目标(其未来位置定义了运动可供性),过早坍缩成 latent 会削弱输入锚点与预测点流的逐点对应;且输入异质——场景点带几何+外观,query 点只带空间锚。

场景点 $(x_j,c_j)$ 用 3D 坐标+RGB;query 点 $q$ 用 3D 坐标+位置编码 $\gamma(q)$、**不带 RGB**。拼接后送入 PointNeXt encoder-decoder:

$$H = f_{dec}(f_{enc}(P, Q_{tool}, Q_{target}))$$

再抽出 query 对应特征 $H_Q = H\mid_{Q_{tool}\cup Q_{target}}$ 作几何条件。语言指令用 CLIP 编码得 $z_l$,**late fusion** 融合:$C = \mathrm{Fuse}(H_Q, z_l)$。

以 $C$ 为条件,用 transformer-based diffusion 预测每个 query 点的相对 3D 位移序列。实现上把条件投影成单个 conditioning token $c_{cond}$,把加噪位移序列 reshape 成 step tokens,拼成 $[c_{cond}; x^{1:m}_k]$ 用 Transformer encoder 处理——即**prefix token conditioning** 而非 cross-attention。去噪目标为噪声预测:

$$\mathcal{L}_{diff} = \mathbb{E}_{k,\epsilon}\left[\alpha(k)\,\lVert \epsilon - \epsilon_\theta(\Delta \tilde{Q}^{1:m}(k), k, C)\rVert_2^2\right]$$

其中 $\alpha(k)$ 是 min-SNR 重加权因子,抑制信噪比过大的时间步。此外两个附加设计:

- **motion-aware 加权损失**:很多 query 点(尤其交互区外)几乎静止,为减小对"零运动平凡样本"的偏置,给 GT 位移幅度大的点更大权重;
- **累积位移损失 $\mathcal{L}_{acc}$**:缓解时间积分带来的漂移。令 $s_i^t = q_i^t - q_i^0$、$\hat{s}_i^t = \hat{q}_i^t - \hat{q}_i^0$ 为相对首帧的 GT 与预测位移,$r_i = \rho(\hat{s}_i^{1:m}-s_i^{1:m})$(鲁棒回归罚),$w_i = g(\frac{1}{m}\sum_t \lVert \Delta q_i^t\rVert)$($g$ 单调增)。

$$\mathcal{L}_{acc} = (1-\lambda)\frac{1}{N_q}\sum_{i=1}^{N_q} r_i + \lambda \frac{\sum_i w_i r_i}{\sum_i w_i}$$

配合逐步损失 $\mathcal{L}_{step}$(约束相邻帧位移差),总目标 $\mathcal{L}(\theta) = \lambda_{diff}\mathcal{L}_{diff} + \lambda_{step}\mathcal{L}_{step} + \lambda_{acc}\mathcal{L}_{acc}$。

> 用大白话说:模型不去回归"这个点跑到哪个绝对坐标",而是学"这个点未来每一步往哪挪多少"的位移序列,用扩散模型采样出来。为防止大量不动的点把 loss 拉平、以及一步步累加导致轨迹越飘越远,分别加了"动得多的点权重高"和"看总位移别漂"两个正则。

### (d) 动作生成与真机部署

可供性是显式几何形式,天然兼容轨迹优化器/模型控制器等下游模块。**抓取**:直接用现成 grasping 模块给出可行的 pre-contact 抓取位姿;**运动**:用一个轻量隐式策略把预测的 3D 点流转成机器人动作——把被操作区的 query 点当作局部刚体集,估计使预测点流与末端执行器运动最佳对齐的刚性变换,formulate 成 ICP 式刚性配准、用**SVD** 高效求解,恢复 SE(3) 变换。执行时闭环:每步从更新观测重新估计运动。作者称这个简单映射已能提供可靠真机执行。

## 三、实验结果

真机设置:Franka 机械臂 + UMI 夹爪,侧装 Intel RealSense D435,用 RoboEngine 分割掉机械臂;运动生成器训 2,000 epoch,8× NVIDIA H100 上约 1 天;推理时从 tool/target 区采 128 个 query 点,tool:target = 3:1。评测 6 类 motion affordance 任务(烤箱开/关、杯子抓/放、倒水、切水果),每任务 10 次。

**表 1:六类 motion affordance 成功率(次/10)**

| 方法 | Pickup | Place | Open | Close | Pour | Cut | 合计 |
|---|---|---|---|---|---|---|---|
| ReKep | 5 | 4 | – | – | 2 | 0 | 11 |
| Track2Act | 4 | 1 | 0 | 0 | 0 | 0 | 5 |
| General Flow | 10 | 6 | 7 | 6 | 0 | 2 | 31 |
| **Ours** | **10** | **8** | **8** | **9** | **4** | **4** | **43** |

其中 General Flow 因不显式建模抓取,评测时人工把夹爪放到物体上闭合、从抓后阶段起评;Track2Act 用 2D flow、直接投到机器人无微调。BridgeACT 在需要旋转轴推理的开/关和接触密集的 pour/cut 上优势最明显。

**表 2:泛化(Cross-Object / Cross-Scene,次/10)**

| 方法 | Pick(CO) | Open(CO) | Cut(CO) | Pick(CS) | Open(CS) | Cut(CS) |
|---|---|---|---|---|---|---|
| ReKep | 5 | – | 2 | 4 | – | 1 |
| Track2Act | 3 | 0 | 0 | 2 | 0 | 0 |
| General Flow | 10 | 6 | 1 | 9 | 4 | 0 |
| **Ours** | **10** | **7** | **3** | **10** | **6** | **2** |

跨物体、跨场景背景下 BridgeACT 均一致更优,说明 affordance 表征捕获了可迁移的交互结构。

**表 3:消融(3D ADE/FDE,单位 cm;越低越好)**

| 配置 | ADE↓ | FDE↓ | 参数量↓ |
|---|---|---|---|
| PTv3 + Early Fusion | 0.0596 | 0.0829 | 51.25M |
| PTv3 + Late Fusion | 0.0445 | 0.0614 | 51.25M |
| PointNeXt + Early Fusion | 0.0380 | 0.0504 | 8.89M |
| PointNeXt + Late Fusion | 0.0380 | 0.0504 | 8.89M |
| PointNeXt + Early + WLoss | 0.0354 | 0.0472 | 8.89M |
| **PointNeXt + Late + WLoss** | **0.0350** | **0.0468** | **8.89M** |

结论:(i) backbone 上 PointNeXt(8.89M)反而略优于更重的 Point Transformer v3(51.25M),且 PTv3 优化更稳、收敛更快、随机种子方差更小,两者同训 2000 epoch;(ii) late fusion 略优于 early fusion——点云 backbone 擅长几何建模、不擅长直接文本-语义对齐,高层融合更有效;(iii) motion-aware 加权损失(WLoss)带来一致提升。

定性对比(图 4):in-domain 的 close 动作,General Flow 出现"内侧轨迹长、外侧短"的反向空间模式,BridgeACT 保持正确几何(外长内短);out-of-domain 的 pickup,General Flow 甚至恢复错运动方向(产出横向 flow 而非上提),BridgeACT 仍与交互语义一致。作者归因于 object-centric 运动表征 + diffusion(比 General Flow 的 CVAE 更具表达力、分布偏移下更鲁棒)。

## 四、局限性

作者明确指出两点:(1) **旋转运动** 从人类演示稳健推断本身就更难;(2) **精细操作** 对接触建模与控制精度要求更严。当前方法在这两类场景仍受限。此外从审稿视角可补充:真机评测仅 6 个任务、每任务 10 次(总量小,统计置信度有限);运动映射假设被操作区为**局部刚体** 并用 SVD 单刚体配准,对可形变、多铰接或需手内调整的物体不成立;抓取完全依赖现成 grasping 模块,失败模式与整体成功率的耦合未拆解;grounding agent 高度依赖 Qwen3-VL/SAM3/CoTracker3 等外部大模型,误差传播未量化;论文未给出与训练数据规模的 scaling 曲线,"可扩展到互联网级视频"仍是愿景。

## 五、评价与展望

**优点。** (1) 问题拆解干净——把"从人类视频学操作"归约到 where-to-grasp 与 how-to-move,并用统一的 tool-target 角色把单物体与 O2O 两类任务纳入同一接口,概念上比 object-centric flow 更接近"任务的因果结构";(2) 抓住了 egocentric 视频的真实痛点——绝对轨迹不稳、相对(tool 相对 target)运动稳定,这个观察对 HOI 数据的监督设计很有价值;(3) 工程闭环完整:自动数据构造 + diffusion 生成 + SVD 刚性配准 + 闭环 SE(3),真机零机器人数据可跑通,且参数量仅 8.89M、部署轻。

**与公开工作的关系。** 本文与 General Flow(3D 点流 affordance)、Track2Act(2D 点轨迹)、ATM(any-point trajectory)同属"结构化 flow 表征"一脉,差异在于**显式 tool-target 角色 + 保留 query 为几何锚 + diffusion 替代 CVAE + 显式建模抓取阶段**;与 ReKep 等 VLM/keypoint-planner 路线相比,它学的是稠密交互动力学而非符号约束。数据侧重度依赖 2026 年前后的基础模型栈(Qwen3-VL、SAM3、CoTracker3、RoboEngine),这既是其能"自动化标注"的底气,也是其复现门槛与误差来源。

**开放问题与可能改进方向。** (a) 单刚体 SVD 映射是最脆弱一环,面向铰接/形变物体应考虑多段刚体或非刚性配准、或直接学 SE(3) 流;(b) 旋转 affordance 可引入显式旋转轴/螺旋运动(screw motion)表征或等变网络来缓解;(c) 数据规模仅 60K clip,尚未验证真正 internet-scale 下的 scaling,与大规模 HOI/egocentric 语料(Ego4D 等)结合是自然下一步;(d) 当前是开环预测+闭环重估计的"隐式策略",若把 affordance 作为中间监督去蒸馏一个端到端 VLA/扩散策略,可能在长程与接触密集任务上更稳;(e) grounding agent 的 verifier/one-step recovery 是很实用的鲁棒化技巧,但缺定量消融,其贡献值得单独评估。总体上,这是一篇工程完成度高、思路清晰的"无机器人数据从人类视频学操作"工作,主要说服力来自真机对照实验而非理论,规模与统计强度是其后续需要补强的地方。

## 参考

1. C. Yuan et al. *General Flow as Foundation Affordance for Scalable Robot Learning.* arXiv:2401.11439, 2024.(最直接对比 baseline,3D 点流 affordance)
2. H. Bharadhwaj et al. *Track2Act: Predicting Point Tracks from Internet Videos Enables Generalizable Robot Manipulation.* ECCV 2024.(2D flow affordance baseline)
3. W. Huang et al. *ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation.* arXiv:2409.01652, 2024.(VLM/keypoint-planner baseline)
4. C. Wen et al. *Any-Point Trajectory Modeling for Policy Learning.* arXiv:2401.00025, 2023.(2D 点轨迹表征先驱)
5. Y. Liu et al. *HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction.* CVPR 2022.(主要训练数据来源)
