# Robots Need More：机器人需要的不止是 VLA 与世界模型

> **论文**：*Robots Need More Than VLAs & World Models*
>
> **作者**：Elis Karcini, Faisal Mehrban, Quang Nguyen, Mac Schwager, Arash Ajoundani, César Cadena, Jan Peters, Marco Hutter, Haitham Bou-Ammar
>
> **机构**：Motoniq.ai；Stanford University；Istituto Italiano di Tecnologia；ETH Zurich；Technical University of Darmstadt；UCL Centre for AI
>
> **发布时间**：2026 年 06 月（arXiv 2606.06556）
>
> **发表状态**：Position Paper（NeurIPS 模板，预印本，未标注录用）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.06556) | [PDF](https://arxiv.org/pdf/2606.06556)
>
> **分类标签**：`position-paper` `data-interface` `embodied-autolabelling` `world-model` `cross-embodiment`

---

## 一句话总结

这是一篇立场/综述文：作者主张通用机器人的核心瓶颈**不是**策略规模（更多演示 + 更大 VLA），而是缺少把世界上海量"未标注行为数据"（人类动作、互联网视频、仿真 rollout、可穿戴传感）转成"机器人可学监督"的**grounding 机制**；据此提出下一代机器人所缺的四根支柱——**physical data engine（具身自动标注）、task-preserving retargeting（跨本体保任务重定向）、physics-grounded world model（物理接地的后果预测）、self-improving deployment loop（自改进部署闭环）**，并把 VLA 重新定位为更大"物理智能栈"中的一层策略接口。

## 一、问题与动机

作者的出发点是一个**数据不对称**：语言/视觉领域有"互联网"——文本和图像天然数字化、密集配有人类监督；而机器人没有。世界里确实充满行为数据（人操作物体、工厂运转、家庭活动、仿真 rollout），它们蕴含任务、目标、接触、失败、物理约束等丰富信息，但**几乎无法被机器人策略直接使用**，因为它们缺三样东西：

1. **embodiment-specific action labels**（本体相关的动作/力信号）；
2. **task semantics**（任务语义、阶段、成功条件）；
3. **reward structure**（奖励/进度结构）。

当前主流 pipeline 是 **robot-data-centric（以机器人数据为中心）**：采真机演示 → 贴动作/语言标签 → 训策略 → 硬件评测 → 循环。作者指出这条路的根本代价：机器人数据集不同于文本语料——**每条轨迹必须物理可执行、每个动作绑定某个具体本体、每次失败都可能损坏硬件或环境**，因此"可用真机监督量"相比世界里已存在的物理行为量小到可以忽略。

因此作者把中心问题从"how to collect **more** robot data"改写为"how to make **broader** sources of physical experience usable for robot learning"，并主张未来 pipeline 应是 **grounding-centric（以接地为中心）**：从广义物理经验（人类动作、互联网视频、机器人交互、仿真、触觉、语言）出发，经由 grounding 机制产出机器人可用的动作、接触、物体状态、任务阶段、目标与奖励。核心命题：**机器人领域缺的不是又一个策略架构，而是一组把物理经验转成机器人可用监督的组件。**

## 二、核心方法

这是立场文，"方法"指作者给出的**形式化框架 + 四支柱议程**。作者刻意不按"数据源/算法族"组织综述，而按"每条研究线暴露的监督瓶颈（supervision bottleneck）"组织。

### 2.1 观测 vs 可执行动作：接地问题的形式化

区分"被观测的物理变化"与"可执行动作"。视频只给观测序列 $\mathbf{o}_{1:T} = \langle o_1,\dots,o_T \rangle$，而模仿学习需要动作序列 $\mathbf{a}_{1:T} = \langle a_1,\dots,a_T \rangle$；从视频学习只能得到"类动作"的潜变量：

$$z_t \sim q(\cdot \mid o_t, o_{t+1}, L_t, L_{t+1})$$

> 用大白话说：视频里能看到"东西怎么动了"，但看不到"机器人该发什么指令"；只能先从相邻两帧（配语言 $L$）里挖出一个"解释这次变化"的潜动作 $z_t$，它还没绑定到任何具体机器人本体，得后续再翻译成真指令。

### 2.2 异构异步经验的统一表示（physical data engine 的输入）

一段"物理经验" episode 可能来自真机 rollout、人类演示、可穿戴、互联网视频、仿真或部署轨迹，**各模态采样频率不同、时间戳不对齐**：

$$\mathbf{x} = \{\, (v_i,\tau_i^{(v)})_{i=1}^{T_v},\ (m_j,\tau_j^{(m)})_{j=1}^{T_m},\ (h_k,\tau_k^{(h)})_{k=1}^{T_h},\ (r_l,\tau_l^{(r)})_{l=1}^{T_r},\ \mathbf{L} \,\}$$

其中 $v$ 为带时间戳视频帧、$m$ 为动捕/可穿戴/体姿测量、$h$ 为触觉/力/接触/手部传感、$r$ 为原始机器人日志（本体感知、部署元数据）、$\mathbf{L}$ 为该 episode 关联的语言（指令、字幕、任务描述、人类纠正）。不是每个 episode 都含全部模态：互联网视频可能只有 $v$ 和弱字幕，可穿戴 episode 可能有 $v,m,h,\mathbf{L}$。

> 用大白话说：把一切"物理经历"塞进一个大杂烩容器——视频、动捕、触觉、机器人日志、语言，各自带时间戳、各自频率。数据引擎的第一件事就是面对这堆异步、不齐、缺模态的原始流。

### 2.3 事件时间线对齐（第一个隐藏推断问题）

因为各流异步，第一个隐藏对象是"原始观测 → 公共物理时间线"的对齐。引入 latent event timeline $\zeta \in \{1,\dots,Z\}$ 与对齐变量 $\mathcal{A}$：

$$\mathcal{A}:\ \{\tau_i^{(v)}, \tau_j^{(m)}, \tau_k^{(h)}, \tau_l^{(r)}\} \to \{1,\dots,Z\}$$

作者举例：`video-frames: 30-55`、`motion-readings: 102-180`、`tactile-spike @ 1.8s` 可能都映射到同一潜事件 $\zeta=2:\texttt{contact-begins}$。**时间对齐不是预处理细节，而是具身自动标注问题的一部分。**

### 2.4 每个事件的潜物理结构 + 数据引擎作为推断模型

对每个潜事件 $\zeta$，恢复其潜物理结构：

$$\mathbf{z}_\zeta = [\mathbf{s}_\zeta,\ \mathbf{c}_\zeta,\ \phi_\zeta,\ \mathbf{u}_\zeta,\ \mathbf{r}_\zeta]$$

$\mathbf{s}_\zeta$ 物体中心物理状态、$\mathbf{c}_\zeta$ 接触/交互标签、$\phi_\zeta$ 任务阶段、$\mathbf{u}_\zeta$ 潜物理动作/转移码、$\mathbf{r}_\zeta$ 任务条件奖励；episode 级还推断目标 $\mathbf{g}$ 与结果标签 $\mathbf{y}$（成功/失败/部分/不安全）。完整隐藏解释 $\mathbf{z} = [\mathbf{z}_{1:Z}, \mathbf{g}, \mathbf{y}]$。于是 **physical data engine 就是一个推断模型**：

$$q_\theta(\mathbf{z}, \mathcal{A} \mid \mathbf{x})$$

它不只是感知模型，而要**联合**求解时间对齐、事件切分、物体状态估计、接触推断、阶段识别、潜动作发现、奖励接地、结果预测。作者强调关键难点：这些标签**互相耦合**（接触改变物体状态、状态定义任务进度、奖励只相对推断目标才有意义），所以必须在一个统一表示里联合推断，而非独立打标。

> 用大白话说：给每个物理事件补上"缺失说明书"——碰到没碰、碰到啥、这是任务第几步、内在动作是什么、算不算进步；再在整段上判断"目标是什么、最后成没成"。这些字段像一张互相牵连的网，动一个牵全身，所以要一次性联合猜出来，这才是"具身自动标注"，区别于普通视频字幕（字幕只会说"一个人把杯子放到托盘上"，数据引擎要恢复能被重定向/仿真/奖励/训策略的物理事件序列）。

### 2.5 支柱二：task-preserving retargeting（跨本体保任务重定向）

推断出事件序列不等于产出策略——存在 **embodiment gap**：人手、平行夹爪、灵巧手、移动操作臂、四足、人形运动学/动力学/接触面/失败模式各异。核心不是"如何复制人的运动"，而是"如何在换一个身体执行时保留**任务相关的物理效果**"。对本体 $e$：

$$\mathbf{a}_\zeta^{(\text{embodied})} = f_\psi(\mathbf{u}_\zeta,\ \mathbf{s}_\zeta,\ \text{embodiment})\quad \text{s.t.}\quad \Delta_{\mathbf{g}}(\mathbf{s}_\zeta,\ \mathbf{a}_\zeta^{(\text{embodied})}) \approx \Delta_{\mathbf{g}}(\mathbf{s}_\zeta,\ \mathbf{u}_\zeta)$$

$\Delta_{\mathbf{g}}$ 是朝目标 $\mathbf{g}$ 的任务相关效果变化（开抽屉=抽屉位移、放置=物体位姿、装配=相对对齐、抓取=接触状态）。作者给出**重定向不变量层级**（从弱到强）：pose（位姿）$\to$ contact（接触时刻/表面）$\to$ object-state transition（物体状态转移）$\to$ intent/skill（意图/技能，机器人可用完全不同的运动完成同一任务）。通用机器人需要沿此层级上移，从"保位姿模仿"走向"保任务效果重定向"。

> 用大白话说：别照抄人的关节角，要抄"效果"。抽屉最后开了多少、杯子最后立没立稳才是要对齐的东西；用什么手、什么轨迹去实现可以完全不同。可穿戴/传感演示的真正价值，就是暴露出接触、力相关事件、物体状态变化、任务阶段这些"比关节角更可迁移、比视频字幕更有信息"的中间变量。

### 2.6 支柱三：physics-grounded world model（后果预测,而非像素预测）

机器人的世界模型区别于通用视频生成器：它要预测"动作会造成什么**物理**变化及为何"，应在结构化物理变量（位姿、空间关系、约束、速度、力、可变形状态、摩擦/质量/刚度/柔顺）上运行。抽象写作 consequence prediction：

$$\mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta,\ \mathbf{u}_\zeta,\ \mathbf{g})\qquad \mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta,\ \mathbf{a}_\zeta^{(\text{embodied})},\ \text{embodiment},\ \mathbf{g})$$

第一式支持任务级推理（"pull/insert/place 会发生什么"），第二式支持本体级规划（"这个机器人以这个控制器执行会怎样"）。作者的关键判据：**评价世界模型的标准不是"未来看起来真不真实"，而是"预测是否保留了决定成功/失败的物理后果"**——一个忽略接触、质量、摩擦、物理可行性但视觉合理的 rollout，对表征学习或许有用，但**不是可靠的机器人监督**。作者还强调世界模型需对自身预测有**校准的置信度**（否则用于规划时会陷入"幻觉→坏决策→更偏离训练分布→更多幻觉"的恶性循环）。

> 用大白话说：机器人要的世界模型是"如果我换个点推、换个角度插、松手后会不会掉"的物理后果引擎，不是画得像的视频。而且要 task-conditioned——开抽屉只需预测抽屉位移和把手接触，桌面纹理无所谓；倒水就得预测液体状态和容器位姿。

### 2.7 支柱四：self-improving deployment loop（自改进部署闭环）

部署后关键问题不再是"发生了什么"，而是"发生的**有没有用**"。这需要 task-conditioned reward grounding：

$$\mathbf{r}_\eta(\mathbf{s}_\zeta,\ \mathbf{g},\ \phi_\zeta)$$

同一物理状态在不同目标下含义不同（杯子放桌上：对"放下杯子"是成功、对"拿起杯子"是失败、对"开抽屉"无关）。闭环为：**deploy policy → observe outcome → 推断 task-conditioned 进度/成功/失败 → 解释失败或纠正 → 把接地监督回灌 data engine → 更新 reward model / world model / retargeting / policy → redeploy**。作者强调需**component-level credit assignment**：动作差就更策略、后果预测错就更世界模型、物理效果没保住就更重定向、成功/失败判错就更奖励模型——否则"知道 rollout 失败了却不知道该改什么"。这是"只会执行训练好策略的机器人"与"能在真实任务上复利改进的学习系统"的根本区别。

> 用大白话说：每次部署都别只留个"成/败"记录，而要变成一段带标签的物理 episode——失败暴露缺了哪次接触、哪个物体状态错了、抓取为何不稳；人类纠正则是高价值信号。把这些回灌进数据引擎，机器人才能在它真正面对的任务上越用越强，而不是停在出厂那一刻。

## 三、关键论据与 survey 数字

这是立场文，**无自研实验**；其"结果"是对现有工作的系统梳理与四支柱议程。以下数字均为原文引用的公开数据集/系统规模，用以论证"robot-native 监督虽强但受限、grounding 层缺失"。

### robot-native 数据集/通用策略（§2.1，暴露"数据已在机器人坐标系里"的瓶颈）

| 数据集/系统 | 规模 | 备注 |
| --- | --- | --- |
| RoboNet | 1500 万帧、7 平台 | 早期多机器人共享，支持视频预测/逆模型 |
| BridgeData V2 | ~6 万操作轨迹 | 低成本平台、24 环境 |
| DROID | ~7.6 万演示 / 350 小时 / 数百场景 | 地理分布众包采集 |
| RH20T | >11 万接触丰富序列 + 对应人类演示视频 | 多模态（视觉/力/音频/动作） |
| RT-1 | ~13 万真机 episode / 13 机器人 / 700+ 任务 | Transformer 语言条件 |
| Open X-Embodiment / RT-X | >100 万真机轨迹 / 22 embodiment | 跨本体聚合、公共格式 |
| Octo | 80 万轨迹预训练 | 开源通用策略，适配新观测/动作空间 |
| OpenVLA | 97 万真机演示 / 7B 参数 | 开源 VLA |
| π0 | flow-matching 建在预训练 VLM 上 | 连续动作、继承互联网语义 |
| SpatialVLA / RDT-1B | ~110 万 / >100 万 多机器人 episode | 显式空间表征 / diffusion transformer 双臂 |

### 生成物理经验（§2.3，暴露"仿真设计者已预设状态/动作/成功条件"的瓶颈）

| 系统 | 规模 | 备注 |
| --- | --- | --- |
| MimicGen | <200 seed 演示 → >5 万演示 / 18 任务 | 从少量人类演示自动合成大规模数据 |
| RoboCasa365 | 365 日常任务 / 2500 厨房场景 / >2000 小时交互 | 含人类演示 + MimicGen 合成 |
| RLBench / Meta-World / ManiSkill / CALVIN / LIBERO | 100 / — / — / — / 130 语言条件任务 | 可复现仿真基准 |

### 从弱接地物理观测学习（§2.2）：作者归纳被动视频能提供的**四类信号**

| 信号类型 | 代表工作 | grounding 缺口 |
| --- | --- | --- |
| 视觉表征 | R3M、VIP、MVP、VC-1 | 编码外观/可供性，但不含接触动力学与力 |
| 潜动作码（latent action） | LAPA、UniVLA | 潜动作≠机器人指令，需本体条件解码器才能落地 |
| 任务进度/奖励信号 | PROGRESSOR、Adapt2Reward、ReWiND、TimeRewarder、SARM | 进度信号≠新本体的奖励 |
| 行为先验（物体使用/可供性/接触/时序结构） | 各类 IfO：TCN、AVID、XIRL | 人类策略未必对机器人可执行 |

作者反复回到的核心判据（贯穿全文的 Takeaway）：**弱监督不消除接地需求，只是把它挪位**——潜动作不是指令、进度信号不是奖励、人类策略未必可执行；视频扩大了物理经验来源，却让 grounding 问题**不可回避**。同理，仿真/世界模型的价值不在视觉真实度，而在**是否保留了决定控制的物理变量**（几何、接触、力、稳定性、约束、材料响应）。

### 四支柱汇总

| 缺失支柱 | 作用 | 形式化 |
| --- | --- | --- |
| Physical Data Engine（具身自动标注） | 异构异步经验 → 结构化标签（物体状态/接触/阶段/潜动作/成功失败） | $q_\theta(\mathbf{z}, \mathcal{A} \mid \mathbf{x})$ |
| Task-preserving Retargeting | 跨本体保"任务相关物理效果"地映射到可执行动作 | $\mathbf{a}^{(\text{emb})} = f_\psi(\mathbf{u}, \mathbf{s}, \text{emb})$ |
| Physics-grounded World Model | 预测动作的物理后果（非仅像素） | $\mathbf{s}_{\zeta+1} \sim p_\omega(\cdot \mid \mathbf{s}_\zeta, \mathbf{u}_\zeta, \mathbf{g})$ |
| Self-improving Deployment Loop | 部署失败/纠正回灌为结构化监督，闭环复利改进 | deploy $\to$ observe $\to$ ground $\to$ update $\to$ redeploy |

## 四、局限性

作为立场文，其局限主要在"论证形态"而非"数值失误"：

- **无任何实证验证**：四支柱与所有形式化（$q_\theta$、$f_\psi$、$p_\omega$、$\mathbf{r}_\eta$）都停留在概念/符号层面，未给出任何最小可行实现或消融，也未定量说明"grounding 层缺失"到底让下游策略损失多少。命题很有说服力，但可证伪性弱。
- **形式化偏"记号化叙事"**：$\mathbf{z}_\zeta = [\mathbf{s},\mathbf{c},\phi,\mathbf{u},\mathbf{r}]$、对齐变量 $\mathcal{A}$ 等把一个极难的联合推断问题写成了一个漂亮的后验，但**如何训练** $q_\theta$（监督从哪来、如何避免退化解、如何评估标签正确性）几乎未触及——而这正是把议程落地的真正难点。
- **各支柱本身即开放难题**：跨本体重定向的 $\Delta_{\mathbf{g}}$ 需要"任务相关效果度量"，其定义本身依赖对任务的先验；物理接地世界模型的表征选择（像素/物体中心/点云/Gaussian/力学）作者也只能列举而无定论。
- **作者利益相关**：核心作者来自 Motoniq.ai，议程与该方向商业叙事一致，需注意"综述—议程"的选择性。
- **survey 覆盖偏 2024–2026 且以英文 arXiv 预印本为主**，部分被引工作尚未同行评审，规模数字以原论文自报为准。

## 五、评价与展望（纯学术视角）

**优点**。这篇立场文最大的贡献是提供了一个**清晰、可操作的组织框架**：把散落的"从视频学习 / 跨本体数据集 / 世界模型 / 奖励建模 / 仿真"等研究线，统一到"每条线暴露的 supervision bottleneck"这一视角下，并用一组连贯符号（观测 vs 动作、异步 episode $\mathbf{x}$、事件时间线 $\zeta$、潜结构 $\mathbf{z}_\zeta$、重定向 $f_\psi$、后果预测 $p_\omega$、奖励 $\mathbf{r}_\eta$）把"data engine → retargeting → world model → deployment loop"串成一条闭环流水线。相比很多只堆参考文献的综述，它给出了**明确的规范性主张**（VLA 只是物理智能栈中的一层策略接口），并把"评价世界模型/仿真的标准应是物理后果保真而非视觉真实"这一判据讲得很透，对领域有校准作用。

**与其他公开工作的关系**。四支柱并非凭空：支柱一延续 LAPA/UniVLA 的 latent-action 与 embodied autolabelling 思路，但把它从"学动作词表"推广为"异步多模态联合接地"；支柱二把 XIRL/TCN 式跨本体不变量学习抽象成"保任务效果层级"；支柱三与 V-JEPA 2、FOCUS、PointWorld、ParticleFormer、PIN-WM、ContactGaussian-WM 等"物理接地世界模型"一脉相承，并呼应了近期把世界模型当零样本策略的 World-Action-Model 路线；支柱四则对应 RoboCat 式自改进与部署闭环。可以说本文是把这些散点**编织成一张议程网**，价值在"编织"而非"新组件"。

**开放问题与可能改进方向**（均为客观学术判断）：

1. **联合接地的可训练性**：$q_\theta(\mathbf{z},\mathcal{A}\mid\mathbf{x})$ 里时间对齐 + 事件切分 + 接触/状态推断相互耦合，天然缺 ground truth。一个务实方向是先在"仿真里有全状态、真机里只有部分标签"上做半监督/弱监督，用仿真提供 $\mathbf{s},\mathbf{c}$ 的强标签、真机提供 $\mathbf{y}$，再对齐分布——但这引入 sim-to-real gap，与作者自己对"物理保真"的判据形成张力，值得专门研究。
2. **任务相关效果度量 $\Delta_{\mathbf{g}}$ 的可学习化**：能否从大量"成功/失败"episode 里反推出 $\Delta_{\mathbf{g}}$ 而不手工指定，是把重定向从"保位姿"推到"保意图"的关键。
3. **标签质量的可验证性**：具身自动标注若标错（比如把"物体被误落"标成失败而非"可复用的放置技能"），会污染下游一切组件；需要一套**标签置信度 + 部署反证**机制，这恰与支柱三"校准不确定性"、支柱四"component-level credit assignment"可以合流。
4. **评测协议缺位**：作者在结论里提了一串评测问题（能否推断接触/状态转移？能否保效果重定向？世界模型能否预测决定成败的后果？部署失败能否更新到正确组件？），但没给基准。领域下一步真正需要的，可能不是又一个 VLA，而是一个**衡量"grounding 质量"的基准套件**。

总体而言，这是一篇观点鲜明、组织有力、对研究方向有导航价值的立场文；其形式化更像"研究蓝图"而非"可复现方法"，读者应把它当作**问题分解与议程设定**来用，而非拿来即用的算法。

## 参考（3-5 篇最相关）

1. Ye et al., *Latent Action Pretraining from Videos (LAPA)*, arXiv:2410.11758, 2024 —— 支柱一"从视频学潜动作词表再落地"的直接前身。
2. Bu et al., *UniVLA: Learning to Act Anywhere with Task-centric Latent Actions*, arXiv:2505.06111, 2025 —— 跨本体/视角的任务中心潜动作，呼应支柱一/二。
3. O'Neill et al., *Open X-Embodiment & RT-X*, ICRA 2024 —— 支柱二跨本体聚合的代表性 robot-native 基座（>100 万轨迹 / 22 本体）。
4. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning*, arXiv:2310.17596, 2023 —— §2.3 生成物理经验路线代表（<200 seed → >5 万演示）。
5. Assran et al., *V-JEPA 2: Self-supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985, 2025 —— 支柱三"物理接地世界模型 + 零样本机器人控制"的最新公开链接。
