# FastUMI-100K：用大规模 UMI 风格数据集推进数据驱动的机器人操作

> **论文**：*FastUMI-100K: Advancing Data-driven Robotic Manipulation with a Large-scale UMI-style Dataset*
>
> **作者**：Kehui Liu, Zhongjie Jia, Yang Li, Zhaxizhuoma（四人共同一作）+ et al.
>
> **机构**：上海人工智能实验室、西北工业大学、上海交通大学、同济大学、西交利物浦大学、苏州 OneStar Robotics、中国电信人工智能研究院（TeleAI）
>
> **发布时间**：2025 年 10 月（arXiv 2510.08022）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.08022) | [PDF](https://arxiv.org/pdf/2510.08022)
>
> **分类标签**：`UMI 数据集` `具身操作` `数据采集`

---

## 一句话总结

用手持、硬件解耦的 FastUMI 采集装置（RealSense T265 做位姿追踪 + GoPro 鱼眼做视觉）批量采集了 **10 万条以上** UMI 风格演示轨迹（约 600 小时、54 类家务任务、上百种物体，含单臂/双臂），配套双层语言标注与多传感器亚毫秒对齐工具链;实验表明该数据在 DP/ACT 单任务模仿、跨本体（Xarm6↔Flexiv Rizon4，仅调坐标映射）、以及 $\pi_0$ VLA 微调上都能取得较高成功率。

## 一、问题与动机

数据驱动的模仿学习是通用操作策略的核心,但演示数据采集是规模化的最大瓶颈。作者把现有采集方式归为三类,并逐一指出短板:

- **遥操作类**（GELLO、ALOHA/ALOHA 2、Bunny-VisionPro）:保真度高,但全程手动远程控制,人力/时间成本高、采集时**硬件耦合**,换机器人需重新做设备特定的重映射,且多集中在简单 pick-place。
- **视觉演示类**:便携,但缺机器人动作标签,也捕捉不到机器人与物体交互时的细粒度动态(力反馈、接触状态)。
- **传感器增强接口类**(UMI 及其后续):把人类演示以高保真直接转成机器人可执行数据。UMI 用手持夹爪 + 腕部相机 + 本体无关接口;前作 FastUMI 进一步用 **RealSense T265** 替换 GoPro 的 VIO/离线 SLAM 流程,把手持硬件与机器人末端解耦,直接记录 6-DoF 末端位姿,避免重型离线 SLAM 或动捕基础设施。

即便如此,机器人数据量相比 CV/NLP 领域仍极小,且长程、细粒度、双臂协作等复杂任务的数据稀缺。本文的目标就是用 FastUMI 系统把 UMI 风格采集**规模化到 10 万条量级**,并补齐双臂与语言标注。

## 二、核心方法

FastUMI-100K 不是一个新模型,而是一套**采集系统 + 数据集 + 处理工具链**。核心贡献在于"如何高效、可复用地把人手演示批量转成本体无关的训练数据"。

### 2.1 硬件与规模化采集

- 硬件沿用 FastUMI 协议:**T265 追踪位姿**,**GoPro 鱼眼拍宽角高分辨率 RGB**;标准化即插即用指尖附件,可装到多种夹爪上。双臂系统把两个 T265 + 两个 GoPro 鱼眼集成到两只夹爪末端,以 **20Hz** 同步记录。
- 组织形式:10 名采集员 + 3 名技术支持,5 个标准化采集环境,5 个两人小组并行采集;分**采集前/采集中/采集后**三阶段,含任务设计、语音引导、实时可视化、场景复位等 SOP。
- **在线质检剔除坏数据**:T265 出问题通常表现为轨迹突跳或全局尺度畸变。系统计算相邻轨迹点间的线速度/角速度,一旦超过阈值就判为 SLAM 漂移异常,自动标记为无效并删除,倒逼采集员即时纠正动作。

### 2.2 多设备多传感器时间对齐

四路传感器采样率不同(2×GoPro 60Hz、2×T265 200Hz),用统一 ROS 时钟打时间戳,再用 ROS 近似时间同步器对齐两路 RGB,允许的最大对齐误差设为 GoPro 采样周期的一半:

$$\Delta t_{\max} = \tfrac{1}{2}\cdot\tfrac{1}{f_{\text{GoPro}}} = \tfrac{1}{2}\cdot\tfrac{1}{60}=\tfrac{1}{120}\ \text{s}$$

用大白话说:两只手臂的两路鱼眼图像只要时间差不超过 1/120 秒就算"同一时刻";随后把视频统一降到 20Hz,每帧再配最近时刻的 T265 位姿,最终双臂多传感器对齐做到**亚毫秒级**,给下游学习一份稳定同步的数据。

### 2.3 双层语言标注(duel-level)

参照 RT-H 的动作层级思想,把"任务目标→动作细节"建成完整语义链,共产出 **15,000 条**文本标注:

- **子任务级(Subtask-Level)**:先用 **GPT-4o** 分析整段视频、自动切分子任务,再用自研 GUI 人工抽取/切分关键帧并与子任务文本对齐。因为第一人称是**自我中心坐标系**,标注刻意只描述目标位置(如 move towards the cup)而非相对方位(如 move left towards the cup)以减歧义;流程是"GPT-4o 预标注 + 人工分割 + 多人交叉校验"。
- **动作级(Motion-Level)**:对每一帧与其后第 10 帧,在三维空间里算相对平移、旋转角、夹爪开合变化,再拼成 move forward/backward/left/right、rotate clockwise/counterclockwise、gripper open/close 等原子语义词。可形式化为对第 $i$ 帧与第 $i+10$ 帧求相对位姿:

$$\Delta \mathbf{p}_i = \mathbf{p}_{i+10} - \mathbf{p}_i,\qquad \Delta \mathbf{R}_i = \mathbf{R}_i^{\top}\,\mathbf{R}_{i+10}$$

用大白话说:不看"末端在哪",而看"从这一帧到十帧后末端往哪挪了、转了多少、夹爪怎么变",把这些量化后翻译成人类能读的运动指令,给语言条件策略更细的监督信号。

### 2.4 数据规模化带来的"高信息密度"

手持直接操作没有遥操作的指令延迟和冗余往返,轨迹更紧凑。作者给出的经验比值是同一任务耗时约为遥操作的五分之一:

$$t_{\text{FastUMI}} \approx \tfrac{1}{5}\, t_{\text{teleop}}$$

用大白话说:叠衣服任务 AgiBot 遥操作要 50 秒,FastUMI 只要 10 秒;单位时间信息密度更高,轨迹也更平滑、更接近真人运动。作者用 Fig.4(d/e) 对比 FastUMI-100K 与 AgiBot 五类双臂任务的平均线速度/角速度,认为前者更贴近真人运动特性。

### 2.5 数据集构成

100K+ 轨迹(含前作 FastUMI 数据),54 类任务分 6 大组:Basic / Composite / Hinged(铰接)/ Flexible Object(柔性)/ Fine(精细)/ Dual-arm Collaborative。每条轨迹 120–500 帧(6–25 秒,60Hz 采集后降到 20Hz)。物体品类分布约为:厨房电器与器具 31%、餐具与饮具 18%、食品饮料 12%、个护清洁 12%、服饰配饰 11%、家具收纳 10%、其他 6%。模态含单/双臂末端状态、多视角腕部鱼眼图、文本标注。声称本体无关,跨本体只需简单坐标映射即可复用。

## 三、实验结果

评测覆盖 16 个任务,每任务重复 15 次,报成功率。主平台为 Xarm6 与 Flexiv Rizon4。

**表 1 · Diffusion Policy 单任务成功率(长程任务拆成子任务)**

| 任务 | 子任务 | 类型 | Flexiv Rizon4 | Xarm6 |
|---|---|---|---|---|
| Make Sandwich | 放生菜叶到盘 | Pick-Place+Rotation | 60.00% | 53.33% |
| Make Sandwich | 放面包片到盘 | Pick-Place+Rotation | 60.00% | 60.00% |
| Place Tableware | 叉子放入筷筒 | Pick-Place+Rotation | 40.00% | 33.33% |
| Heat Food | 打开微波炉门 | Hinged | 66.67% | 66.67% |
| Pour Water | 瓶中水倒入杯 | Rotation | — | 66.67% |
| Storage Shoes | 鞋放入开着的柜 | Pick-Place | — | 40.00% |
| Wash Clothes | 衣物放入洗衣机 | Pick-Place+Rotation | 33.33% | — |

**表 2 · ACT 变体对比(数据源自前作 FastUMI,Joint vs TCP)**

| 任务 | ACT(Joint) | Smooth-ACT(Joint) | PoseACT(TCP-绝对) | PoseACT(TCP-相对) |
|---|---|---|---|---|
| Pick Bear | 20.00% | 60.00% | 80.00% | 73.33% |
| Sweep Trash | 6.67% | 26.67% | 53.33% | 60.00% |

结论:用 **TCP(末端)位姿**训练远优于关节空间,印证了本体无关、以末端状态为动作表示的 UMI 风格更能捕捉轨迹形状与动态特征。

**表 3 · $\pi_0$-base 微调后短程任务成功率(Xarm6)**

| 任务 | 类型 | 成功率 |
|---|---|---|
| Open Drawer | Hinged | 80.00% |
| Open Roaster | Hinged | 86.67% |
| Open Container | Hinged | 73.33% |
| Rearrange Coke | Pick-Place | 93.33% |
| Unplug Charger | Pick-Place | 93.33% |
| Pick Bear / Lid / Cup | Pick-Place | 80.00 / 80.00 / 93.33% |
| Hotdog in Roaster | Hinged+Pick-Place | 80.00% |

短程各类任务均取得高成功率,说明**腕部鱼眼视角**足以捕捉物体空间状态与上下文,UMI 风格数据可与常规非 UMI 数据在 VLA 里联合训练而不损性能。

**表 4 · $\pi_0$ 微调后长程任务分阶段成功率(Flexiv Rizon4)**

| 任务 | 子任务 | 类型 | 成功率 |
|---|---|---|---|
| Make Sandwich | 放生菜到盘 | P-P+Rotation | 100.00% |
| Make Sandwich | 放面包到盘 | P-P+Rotation | 73.33% |
| Wash Clothes | 抓衣入洗衣机 | P-P+Rotation | 93.33% |
| Wash Clothes | 关洗衣机门 | Hinged | 60.00% |
| Heat Food | 开微波炉门 | Hinged | 100.00% |
| Heat Food | 放面包入微波炉 | P-P+Rotation | 0.00% |
| Heat Food | 关微波炉门 | Hinged | 0.00% |

长程任务前阶段高、后阶段掉:Make Sandwich/Wash Clothes 因推理误差累积略降;**Heat Food 在第二阶段直接 0%**——在"推开微波炉门"与"抓面包"这类相似观测间发生局部混淆而卡死,无法继续。

**跨平台部署**:单任务 DP 训练的模型不额外微调、仅调末端坐标映射,即可迁到 Xarm6 与 Flexiv Rizon4。因人手采集不受特定本体物理约束,可能超出小工作空间机械臂的可达范围;作者用"按目标机器人额定工作空间构 3D 包围盒、主轨迹越界则过滤"的策略解决(如 Wash Clothes 完全超出 Xarm6 工作空间)。

## 四、局限性

1. **无固定第三人称/全局视角**:纯依赖腕部第一人称鱼眼,长程任务里不同阶段的相似观测会误导策略推理——Heat Food 长程第二阶段成功率直接归零,是这一缺陷最直接的证据。
2. **长程推理误差累积**:$\pi_0$ 微调下各长程任务后阶段普遍掉点。作者也承认需要更强历史/序列建模的 VLA 才能缓解。
3. **缺真实力/触觉信号**:引言批评视觉演示法"抓不到力反馈、接触状态",但本数据本身仍是视觉 + 位姿,正文提到的"immediate tactile feedback"并无真实触觉传感器支撑,与 ViTaMIn/FreeTacMan/3D-ViTac 那条视触融合路线相比缺失接触模态,这一点与其对视觉演示法的批评存在一定张力。
4. **人手数据超出机械臂工作空间**:需包围盒过滤,意味着一部分采集轨迹在特定本体上不可用,"本体无关可直接复用"要打折扣。
5. **数据披露不充分**:100K 含前作 FastUMI 旧数据,论文未给出去重后规模、每任务轨迹数、有效比例等关键统计;15,000 条文本标注相对 10 万轨迹覆盖有限;GPT-4o 预标注可能引入噪声。
6. **对比与评测偏弱**:与其他 UMI 风格/大规模数据集仅在采集速度上与 AgiBot 定性比较,缺乏训练同一策略在不同数据集上的受控泛化对照;评测每任务仅 15 次重复,统计力度有限。

## 五、评价与展望

**优点**:方向踏实。它把 UMI/FastUMI 这条"手持、硬件解耦、末端位姿为动作"的采集范式真正推到 10 万条量级,并补上**双臂配置**与**双层语言标注**两块此前 UMI 风格数据集普遍缺的拼图;跨本体只调坐标映射、采集效率约为遥操作 1/5,是很实用的工程价值;数据、代码承诺开源。表 2 中 TCP 相对/绝对位姿显著优于关节空间,为"用末端状态作动作表示"提供了干净的经验支撑。

**与公开工作的关系**:它站在 UMI(Chi et al.)与前作 FastUMI 肩上做规模化,可与 DROID(76k)、Open X-Embodiment(1M+/22 本体)、AgiBot World、BridgeData V2、RoboMind 等大规模语料互补——UMI 风格的独特价值在于保留部署时的腕部视角、无需实验室基础设施、天然本体无关,利于复用;而在接触密集任务上,它与 ViTaMIn/FreeTacMan/3D-ViTac 的视触路线是互补而非替代。语言标注借鉴 RT-H 的动作层级,策略侧对接 DP/ACT/$\pi_0$。

**开放问题与可改进方向**:①第一人称长程"相似观测卡死"是 UMI 风格的系统性痛点,需要带记忆/历史建模的策略,或引入辅助第三人称视角来提供全局状态;②缺接触力/触觉,限制了精细与接触密集任务的上限,后续若能在指尖附件上集成低成本触觉会显著提升数据价值;③需要更透明的数据统计(去重规模、每任务分布、有效率)与更严格的 held-out 泛化对照,才能让"100K"这一卖点站得更稳;④跨本体过滤当前靠工作空间包围盒,较粗,可探索基于逆运动学可达性/动力学约束的更细过滤。整体上,这是一份工程扎实、方向正确但披露与评测仍偏薄的规模化数据集工作。

## 参考

1. Chi et al., *Universal Manipulation Interface: In-the-wild Robot Teaching without In-the-wild Robots*, arXiv:2402.10329, 2024.（UMI,范式起点）
2. Liu et al., *FastUMI: A Scalable and Hardware-independent Universal Manipulation Interface with Dataset*, arXiv:2409.19499, 2024.（本文前作/采集系统）
3. Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.（VLA 微调基座）
4. Khazatsky et al., *DROID: A Large-scale In-the-wild Robot Manipulation Dataset*, arXiv:2403.12945, 2024.（大规模数据对照）
5. Belkhale et al., *RT-H: Action Hierarchies Using Language*, arXiv:2403.01823, 2024.（动作级标注范式来源）
