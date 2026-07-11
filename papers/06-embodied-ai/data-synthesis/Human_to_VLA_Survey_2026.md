# Human-to-VLA Survey：从人类视频到机器人操作——面向可扩展 VLA 学习的人本数据综述

> **论文**：*From Human Videos to Robot Manipulation: A Survey on Scalable Vision-Language-Action Learning with Human-Centric Data*
>
> **作者**：Zhiyuan Feng, Qixiu Li, Huizhi Liang, Rushuai Yang, Yichao Shen, Zhiying Du, Zhaowei Zhang, Yu Deng, Jiaolong Yang, Baining Guo et al.
>
> **机构**：Tsinghua University、HKUST、Xi'an Jiaotong University、Fudan University、Microsoft Research Asia、Peking University、Microsoft Zurich
>
> **发布时间**：2026 年 05 月（arXiv 2606.00054）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.00054) | [PDF](https://arxiv.org/pdf/2606.00054)
>
> **分类标签**：`VLA 综述` `Human-Centric Data` `Representation Bridge`

---

## 一句话总结

这是一篇聚焦"如何把人类视频转化为 VLA 可用知识"的综述,提出以**中间表征桥接（representation bridge）** 为核心的信号导向分类法,把现有工作归为 **Latent Action / World Model / Explicit 2D / Explicit 3D** 四条路线,并系统梳理了两大类共 17 个人本视频数据集(从 SSv2 220K 片段到 HowTo100M 136M 片段),最后凝练出 episodization、异构对齐、部署导向评测三大开放接口。

## 一、问题与动机

VLA(Vision-Language-Action)大模型的泛化能力被**机器人示教数据的稀缺** 卡住:真机遥操作数据昂贵、受安全约束、且与特定本体/传感器/控制频率强耦合。即便有 Open X-Embodiment、DROID、AgiBot World 等规模化努力,机器人数据仍在长程任务结构、罕见失败模式、部署分布漂移上严重欠代表。

与之相对,**人类视频**(YouTube/TikTok 采集的 HowTo100M、可穿戴设备录制的 Ego4D/EPIC-KITCHENS 等)规模大数个量级,且蕴含丰富的语义与物理线索(第一人称物体操作、多视角流程装配、常识物理交互)。但它们并非"机器人可执行的动作数据":缺少对齐的、机器人可执行的动作标签与本体感知,人手运动也无法直接映射到机器人运动学与控制接口。

本文因此提出贯穿全篇的中心问题:**当人类视频进入 VLA 训练管线时,它被转化成了什么类型的信息,这些信息又如何与策略学习对接?** 综述沿着"处理管线中构造出何种表征"这一信号视角组织整个快速演化的领域,并明确聚焦端到端 VLA(把语言、视觉、动作统一进单一生成/预测框架),而非把人类视频仅当作 RL/模块化系统的辅助增强。

作者列出三点贡献:(1)**信号导向分类法**——以管线为轴的表征桥接分类,横向比较四条路线的表征形式、可扩展性与 grounding 需求;(2)**数据集地图**——沿"是否有显式几何/3D 信号"与"脚本化 vs 野外采集"两轴系统整理人本视频数据集;(3)**三个接口的挑战**——野外视频的可扩展 episodization、本体/视角失配下的异构 grounding、部署导向的迁移效率评测。

## 二、核心方法

### 2.0 统一形式化:VLA 与人本数据的接口

VLA 学习即把多模态观测映射为可执行动作。每一时刻 $t$,智能体获得视觉观测 $o_t$、本体状态 $s_t$(关节角/夹爪状态/末端位姿)与语言指令 $l$。现代 VLA 采用序列建模,基于观测与状态历史预测未来动作块 $a_{t:t+H}$(chunk,horizon $H$),以最大似然模仿学习在机器人轨迹集 $\mathcal{D}_{\text{robot}}$ 上优化:

$$\max_{\theta} \sum_{\tau \in \mathcal{D}_{\text{robot}}} \sum_{t} \log \pi_{\theta}\big(a_{t:t+H} \mid o_{\leq t}, s_{\leq t}, l\big)$$

其中动作 $a_t$ 被实例化为物理可执行控制目标:SE(3) 末端运动(6-DoF 位姿或 twist)、夹爪开合(二值或连续)、关节指令,以及灵巧手的高维手部位姿参数(手指关节角)。

**用大白话说**:VLA 就是"看图 + 听指令 → 输出一串机器人动作",训练靠模仿真机数据。人本数据的困境在于——上式里唯一能从人类视频廉价获得的只有观测 $o_t$ 那一项,而昂贵的 $a_t$(真机动作标签)和 $s_t$(本体感知)恰恰缺失。四条"表征桥接"路线本质就是:各自发明一种中间信号,去补上或绕开这块缺口。

### 2.1 路线一:Latent Action(潜在动作抽象)

自监督地从视频中抽取紧凑动作表征,规避对显式动作标注的依赖。典型范式:取相邻两时刻 $o_t$ 与 $o_{t+H}$,推断潜在向量 $z_t$,使得 $o_{t+H}$ 能由 $z_t$ 与 $o_t$ 重建。通过强信息瓶颈(如 VQ-VAE 式离散化),$z_t$ 被迫只编码与动作相关的帧间变化,充当式(1)中动作变量的代理。

**用大白话说**:不知道人做了什么动作没关系——只要能从"这一帧 + 一个小编码"猜出"下一帧",那个小编码就抓住了动作的本质。瓶颈越窄,越逼它丢掉无关的画面细节、只留下"动作"。

自 LAPA 引入以来,两大挑战是:(i)学到跨本体/跨数据源可泛化的紧凑动作表征;(ii)把潜在动作预测有效嵌入 VLA 训练。代表性做法:

- **IGOR**:编解码时用非对称裁剪增广,抑制绝对位置信息。
- **UniVLA**:用 DINOv2 特征替代像素级监督以避开噪声细节,并借任务指令做两阶段离散化,解耦任务无关运动(如相机运动)。
- **ViPRA**:重建时加光流一致性损失,鼓励物理合理运动编码;**Motus**:直接以帧间光流为输入,削弱外观变化影响。
- **villa-X**:在本体上下文条件下联合学习人类与机器人潜在动作空间,用有标注机器人数据把潜空间锚定到动作条件动力学;**CLAP**:用对比学习把人类视频潜变量与量化的机器人潜空间对齐;**LAWM**:用带稀疏约束的连续动作以适配野外动力学。
- 集成进 VLA 训练的另一视角:**Moto**(自回归预训练下一潜在动作,微调时插入 query token 解码机器人动作)、**GO-1**(分层设计,潜在规划器桥接 VL 主干与 flow-based 动作专家)、**GR00T N1**(潜在动作作伪目标,与异构真机动作标签联合训练)。

开放问题:高度压缩的潜在动作 token 能否充分表达复杂高 DoF 灵巧操作。

### 2.2 路线二:World Model(预测式世界建模)

用视频作预测监督,学习环境如何随交互演化。形式化为预测未来环境状态:

$$p\big(S_{t+1:t+H} \mid o_{\leq t}, l\big)$$

其中 $S_{t+1:t+H}$ 是未来环境状态(如原始像素或视觉特征)。该过程可经无动作监督的视频生成学习,类似潜在动作但对复杂动力学更具表达力。

**用大白话说**:让模型学会"想象未来会发生什么"——如果它能准确预测下一秒的画面,说明它已内隐地理解了动作及其后果,再把这份"预知"蒸馏进策略即可。

- **GR-1**:GPT 式世界模型,基于语言与视觉自回归预测未来帧,先在第一人称人类视频预训练,再在机器人数据上加下一帧预测目标微调;**GR-2**:用 web 级人类数据 + VQGAN 视觉 token 扩展。
- **FLARE**:联合训练动作解码器与隐式世界建模,在无动作标签人类视频上对解码器内部特征与未来视觉特征施加表征对齐(REPA);**VPP**:把预训练视频生成模型(SVD)当冻结特征提取器,用其"预测式视觉表征"做下游策略条件;**Mimic-Video**:用视频扩散模型部分去噪步骤的中间表征增强动作预测。
- **Gen2Act**:把视频生成当作显式规划接口——先由预训练视频生成模型从静态首帧"幻想"出一段类人视频计划,再训练视频条件策略把生成计划映射为机器人动作。

代价:世界模型训练开销大(需生成大量视觉 token),且如何把世界模型知识蒸馏进动作预测仍是难题。

### 2.3 路线三:Explicit 2D(显式 2D 线索)

用现成 2D 视觉工具抽取可解释的图像平面信号(关键点、边界框、2D 点轨迹、掩码、光流)作为中间表征与额外监督。

- **ATM**:密集点轨迹作类动作监督,训练 transformer 预测未来空间运动,提供动态空间先验;**Magma**:把轨迹渲染成 "Trace-of-Mark" 视觉提示,变成 in-context 控制线索;**Gemini Robotics**:在标注了 2D 基元(框、关键点、轨迹)的人类视频上训练具身推理主干。
- **A0**:建模物体中心的接触点与接触后轨迹作为本体无关的交互表征;**Masquerade**:编辑野外人类视频使其视觉上像机器人示教,提供显式 2D 机器人关键点监督以缩小本体差距。

局限:2D 线索会继承底层视觉模型的标注噪声、常需额外过滤;且因缺乏深度与遮挡推理,在表达真实三维交互(复杂操作)上存在根本限制。

### 2.4 路线四:Explicit 3D(显式 3D 结构)

恢复 3D 结构(位姿/轨迹)并重定向到机器人兼容动作空间;参数化手模型(MANO)是连接人手与机器人的规范紧凑 3D 表征。方法差异主要在:利用何种视频、如何估计 3D 标签、以何机制桥接人机动作空间。

- **EgoVLA**:借多视角优化或 RGB-D SLAM 获高质量 3D 手部动作标注,仅用腕部位姿 + 指尖位置统一人机动作预测,并以逆运动学从人类动作空间求解机器人动作;**H-RDT**:以手为中心表征,利用 AR/VR 设备采集的手部标注视频;**In-N-On**:AR 采集人类视频并把头部位姿纳入统一动作表征;**Being-H0**:扩展到更大人类视频语料,把连续 MANO 手部轨迹离散成运动 token 词表。
- **VidBot**:结合 SfM 与深度基础模型,从无脚本野外视频恢复 3D affordance 轨迹;**VITRA**:自动化管线,把大量无脚本真实视频转成机器人对齐的原子动作段 + MANO 手部标注 + 语言标注;**Yoshida et al.**:从第一人称视频重建 3D 物体位姿变化作监督。
- **MotionTrans**、**Kareer et al.**:用可穿戴设备采集带 3D 动作标注的人类示教,在人机数据上联合训练,提升对未见任务的泛化。

优点:3D 表征提供坐标一致接口(SE(3) 轨迹、参数手、物体位姿),减少视角依赖与深度歧义,与 VLA 动作预测更天然对齐。开放问题:精确 3D 标签获取困难;野外场景须靠视觉模型重建 3D,引入比 2D 更大的误差;直接在 3D 空间预测须从 2D 投影反推三维世界结构——究竟 3D 预测是否终究优于 2D 图像空间预测,仍是开放问题。

## 三、实验结果

作为综述,本文的"结果"是分类法与数据集地图。下表复现原文核心归纳。

**四条表征桥接路线对比(源自 Sec.3 与 Table 1)**

| 路线 | 中间信号 | 代表方法 | 主要短板 |
|---|---|---|---|
| Latent Action | VQ/连续潜在动作 token | LAPA、IGOR、UniVLA、villa-X、GO-1、GR00T N1 | 高压缩 token 或难表达高 DoF 灵巧操作 |
| World Model | 生成式视频先验/预测视觉特征 | GR-1、GR-2、VPP、FLARE、Gen2Act | 训练开销大、知识蒸馏进动作难 |
| Explicit 2D | 点轨迹/关键点/框/掩码 | ATM、Magma、Gemini Robotics、A0 | 继承标注噪声、缺深度与遮挡推理 |
| Explicit 3D | 6-DoF 腕位姿/MANO/物体位姿 | EgoVLA、Being-H0、H-RDT、VITRA | 精确 3D 标签难获取、野外重建误差大 |

**代表方法数据规模(源自 Table 1,"–"为原文未报告)**

| 方法 | 路线 | 数据源 | 规模 | 末端执行器 |
|---|---|---|---|---|
| LAPA | Latent | SSv2 | 220k clips | Gripper |
| IGOR | Latent | SSv2/Epic/Ego4D/EGTEA | 2m frames | Gripper |
| villa-X | Latent | Ego4D/EGTEA/Epic/HOI4D/SSv2 等 | 3.6m frames | Gripper + Dex Hand |
| GR-1 | World | Ego4D | 800k / 667h | Gripper |
| GR-2 | World | Ego4D/SSv2/Epic/HowTo100M/Kinetics | 38m frames | Gripper |
| ATM | 2D | Self-collected | 0.3k | Gripper |
| Being-H0 | 3D | UniHand | 2.5m / 1155h | Gripper |
| H-RDT | 3D | EgoDex | 338k / 829h | Gripper |
| VITRA | 3D | Ego4D/Epic/EgoExo4D/SSv2 | 1m | Dex Hand |

**人本视频数据集分类(源自 Table 2)**

| 数据集 | 规模 | 视角 | 几何信号 | 脚本化 |
|---|---|---|---|---|
| SSv2 | 220K clips | 3rd | RGB | 是 |
| EPIC-KITCHENS | 100 hours | Ego | RGB | 否 |
| Ego4D | 3.6K hours | Ego | RGB | 否 |
| EgoExo4D | 1.3K hours | MV+Ego | RGB(calib) | 否 |
| HowTo100M | 136M clips | Mix | RGB | 否 |
| Egocentric-100K | 100K hours | Ego | RGB(无语言) | 否 |
| RH20T | 110K clips | MV+Ego | RGB(calib) | 是 |
| HOI4D | 2.4M frames | Ego | RGB-D | 是 |
| EgoDex | 300K episodes | Ego | RGB | 是 |
| ARCTIC | 2.1M frames | MV+Ego | MV(calib) | 是 |
| TACO | 5.2M frames | MV+Ego | MV(calib) | 是 |
| HoloAssist | 166 hours | MV+Ego | RGB-D | 是 |
| HOT3D | 3.7M images | Ego | MV(calib) | 是 |

原文按两轴组织数据集:**是否含显式 metric 3D 动作标签**(Category 1 无 / Category 2 有)与**采集是否脚本化**。核心权衡是——无 3D 标签的大规模野外语料(Ego4D、HowTo100M)行为多样但只能靠视觉状态转移间接推断动作;有 3D 标签的语料(HOI4D、EgoDex、ARCTIC 等)监督直接、更接近模仿学习,但规模较小且多为实验室/仪器化脚本采集,牺牲了多样性与代表性。

## 四、局限性

作为一篇综述,其局限主要在于:

1. **缺横向定量基准**:因四条路线各自在不同数据集、不同机器人、不同评测协议上报告,综述未能给出统一实验横评,读者难以判断路线间的真实优劣——这恰是作者自己在 Sec.5.3 指出的领域痛点,但本文亦未提供实证解答。
2. **时效性与覆盖偏向**:引用大量 2025–2026 极新预印本(部分为 workshop 论文),这些方法尚未经充分同行验证,归类与结论有随时间失效风险;且方法覆盖偏向 Latent Action 与 3D 两条线,World Model 与 2D 的深度相对略浅。
3. **分类边界并不互斥**:GO-1、GR00T N1、FLARE 等实际是"潜在动作 + 世界模型"或"世界建模 + 动作解码"的混合体,单一路线归类是一种简化;论文未充分讨论多路线融合这一现实趋势。
4. **人本数据独有价值未被隔离论证**:综述反复强调"须证明人类视频预训练本身的增益",但受综述体裁所限,无法给出"有/无人类视频、控制真机数据预算与算力"的受控消融证据。

## 五、评价与展望

**优点**。本文最大价值在于其**信号导向(pipeline-based)的组织视角**:不按机器人任务或模型架构分类,而追问"人类视频最终被转成了什么信号、如何接入策略",这一提问方式比常见的按方法罗列更本质,能把 LAPA、GR-1、ATM、EgoVLA 等表面迥异的工作放进同一坐标系。Sec.5 把挑战收敛到 **episodization / 异构对齐 / 部署导向评测** 三个接口,尤其"用语义与交互驱动的分段(围绕操作阶段与物体状态变化)取代固定时间窗"、"把动作表征锚定到交互结果(物体状态变化)而非视角相关外观"两点,是相当有洞见的具体方向。

**与其他公开工作的关系**。相较早期把人类视频当作表征学习(R3M、MVP、VIP)或 affordance 提取(Bahl 等)的辅助信号的综述视角,本文明确聚焦"端到端 VLA 内部的生成/预测式整合",与 HERMES 提出的多源人类运动统一 RL 框架、EMMA 的移动操作 co-training 形成互补但立场不同的对照(它有意排除把人类视频仅作 RL 增强的路线)。其分类法与 Open X-Embodiment/DROID 侧重的"真机数据规模化"是正交且互补的两条思路。

**开放问题与改进方向**。(1)**统一评测缺失** 是全领域最大瓶颈——LIBERO/CALVIN/SIMPLER 都欠代表互联网视频的多样性与长程结构,亟需"以真机数据预算为自变量报告性能 + 对新物体/场景 held-out + 量化视角/本体漂移鲁棒性"的协议;(2)**本体不变监督**(交互意图、物体中心效应)相较逐点轨迹模仿更可能跨形态迁移,是缓解 embodiment mismatch 的可行路径;(3)**跨视角对齐**——野外视频以第三人称为主(接触弱观测、遮挡多),第一人称也因头部转动快速切换视角,如何显式对齐人机观测是硬骨头;(4)3D vs 2D 空间预测孰优、高压缩潜在 token 能否承载高 DoF 灵巧手,均是尚未有定论的核心科学问题。总体而言,这是一篇视角清晰、时效性强、对入门与选题都有较高参考价值的综述,主要遗憾是受体裁所限缺乏实证横评。

## 参考

1. Ye et al. *LAPA: Latent Action Pretraining from Videos.* ICLR 2025.(潜在动作路线奠基工作)
2. Wu et al. *GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation.* ICLR 2024.(世界模型路线代表)
3. Wen et al. *ATM: Any-point Trajectory Modeling for Policy Learning.* RSS 2024.(显式 2D 点轨迹代表)
4. Yang et al. *EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos.* arXiv:2507.12440, 2025.(显式 3D 手部路线代表)
5. Luo et al. *Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos.* arXiv:2507.15597, 2025.(MANO 运动 token 化,规模化 3D)
