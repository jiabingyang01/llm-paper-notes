# MolmoBot：大规模仿真实现零样本操作

> **论文**：*MolmoBot: Large-Scale Simulation Enables Zero-Shot Manipulation*
>
> **作者**：Abhay Deshpande, Maya Guru, Rose Hendrix, Snehal Jauhri（四位共同一作,按字母序）；Dieter Fox, Ali Farhadi, Georgia Chalvatzaki, Dhruv Shah, Ranjay Krishna 等 et al.
>
> **机构**：Allen Institute for AI (Ai2)、University of Washington、University of California Los Angeles、Technische Universität Darmstadt、Princeton University
>
> **发布时间**：2026 年 03 月（arXiv 2603.16861，v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.16861) | [PDF](https://arxiv.org/pdf/2603.16861)
>
> **分类标签**：`仿真数据引擎` `零样本 sim-to-real` `VLA` `程序化生成` `移动操作`

---

## 一句话总结

用程序化生成的 **1.7M 条**、跨 94.2k 个环境的纯仿真轨迹（无照片级渲染、无域自适应、无任何真机微调）训练操作策略,即可零样本迁移到真机:桌面 pick-and-place 上 MolmoBot 达到 **79.2%** 真机成功率,大幅超过用 10k+ 小时真机数据训练的 $\pi_{0.5}$（约 39.2%），从而挑战"仿真数据不足以支撑真机操作"的主流假设。

## 一、问题与动机

机器人基础模型（GR00T、$\pi_0/\pi_{0.5}$、Gemini Robotics）大多由少数资源雄厚的工业实验室基于大规模**真机**数据构建,其数据配比、采集流程、过滤与训练配方往往只部分公开,导致"如何从零搭一个机器人基础模型"的知识高度集中。与此同时,机器人领域普遍持有一个假设:**仿真本身不足以支撑操作**,sim-to-real gap 只有在引入一定量真机数据做适配后才可控——仿真只能用于预训练、bootstrap 或压力测试,而非产出稳健真机策略的充分底料。

作者延续其导航工作 SPOC 的思路（在数十万程序化生成房屋上模仿最短路径专家即可零样本迁移导航到真实环境）,追问:**规模化的仿真数据能否同样支撑操作任务的零样本迁移?** 结论是可以——只要仿真在环境、本体、articulated 资产、任务四个维度上被"激进地"放大。

## 二、核心方法

整体分两部分:数据引擎 **MolmoBot-Engine**（产出 **MolmoBot-Data**）与三类策略模型。

### 2.1 MolmoBot-Engine 数据引擎

核心洞见:**操作策略从物体/构型/视角的多样性中获益,远多于从照片级渲染中获益**。因此在 MuJoCo 里对程序化生成的 MolmoSpaces 场景（一个含 232k 环境、48k 可操作物体、8 类任务的开放生态）做大量域随机化,以远低于真机采集的成本批量生成演示。

**环境构造**:从 20 万+ 预建 MolmoSpaces 场景中取一个,布局/家具/静态物固定,但为每个任务从物体池采样并放置任务相关物体（充当 receptacle、pickup 目标或干扰物）。刚体来自 iTHOR 与 Objaverse,按可抓取尺寸过滤（receptacle 包围盒边长 $<50$ cm、pickup 物 $xy$ 对角线小于 receptacle 且竖直尺寸 $\le 15$ cm,且要求水密碰撞网格）。

**域随机化**（三轴 + 训练期图像增广）:

- **光照**:$\langle 1, N \rangle$ 盏点光源/平行光,随机位置、强度、颜色、阴影。
- **纹理**:程序化纹理 + AI2THOR 真实贴图。
- **动力学**:摩擦系数、物体质量、关节阻尼在合理范围内采样。
- **位姿**:可操作资产以随机 6-DoF 位姿放置,满足碰撞与可达约束。

**初始关节随机化** 采用"分级"策略——近端关节扰动小、远端关节扰动大:

$$q_0 + \delta,\quad \delta_i \sim \mathcal{U}(-r_i, r_i)$$

其中 Franka 臂用 $\mathbf{r}_{\text{arm}} = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]$ rad（经 Jacobian 加权启发式选取,把 TCP 位移界定在 $\le 10$ cm）。**用大白话说**:让机器人每回合的起始姿态都不一样,而且越靠近手腕的关节抖得越狠,这样策略不会记死一条轨迹;之所以按 Jacobian 缩放,是保证关节角的随机不会把末端甩出去太远。

**动作噪声注入**（防止策略过拟合到精确的动作回放）:噪声与指令位移成正比。臂在 TCP 空间加噪再经 Jacobian 伪逆映回关节空间:

$$\Delta \mathbf{x} = J \Delta \mathbf{q},\quad \sigma_{\text{pos}} = \alpha \lVert \Delta \mathbf{x}_{\text{pos}} \rVert,\quad \sigma_{\text{rot}} = 0.1\,\sigma_{\text{pos}}$$

$\alpha = 0.1$,位置噪声截断到 $\pm 2$ cm、旋转到 $\pm 0.1$ rad,再通过最小二乘解 $J \boldsymbol{\epsilon}_q = \boldsymbol{\epsilon}_{\text{tcp}}$ 投影回关节。**用大白话说**:静止不动的指令不加噪,大幅移动的指令按比例多加噪——模拟真实执行的抖动,又不至于把精细动作（如抓取瞬间）搞坏。

**专家规划器**:脚本化 demonstrator 迭代采样抓取、验可行性、执行运动。Franka FR3 用 IK 插值,移动机器人 RB-Y1 用 **CuRobo**（GPU 加速、碰撞感知的轨迹优化）协调多自由度。抓取采样-过滤分三步:候选加载与排序（按 TCP 邻近度、旋转相似度、竖直对齐、到质心距离的加权代价排序）→ 碰撞过滤（MuJoCo 广相碰撞检测,批量至多 128）→ IK 可达性检查（批量逆运动学至多 256）。轨迹按固定 phase 分解（如 pick-and-place: Pregrasp→Grasp→Lift→Preplace→Place→Postplace→Stow）逐段规划。**Retry 行为**:执行中若检测到失误（物体脱手/抓取失败）则重置到当前 phase 首帧重试,超过 3 次则丢弃该回合——这一显式重试机制让策略习得从扰动中恢复的能力。

**任务定义**:4 类刚体任务（Pick、Pick-and-place、Pick-and-place-next-to、Pick-and-place-color）+ 2 类 articulated 任务（Open 开橱柜/抽屉/烤箱/洗碗机、Open-door 开铰链门,门任务按机器人相对位姿条件化为"推开"或"拉开"）。语言指令的指代表达每回合采样,用 CLIP 相似度 + softmax（温度 $\tau=0.02$）在相似度-margin 上抽取,既多样又无歧义。

### 2.2 三类策略模型

- **MolmoBot**（旗舰）:基于 Molmo2-4B（SigLIP2 视觉编码 + Qwen3-VL LLM）的多帧 VLM,外接 **DiT 流匹配动作头**。动作头与 VLM **逐层耦合**——每个 DiT 层 cross-attend 到对应 LLM 层的隐状态（含视觉与语言）。冻结视觉编码器与 projector,只训动作头与 LLM。每视角吃 $F=3$ 帧（当前、约 0.5 s 前、约 1 s 前，$D=8$ 步）,每图编码为 192 token;预测长度 $H=16$ 的动作块,执行前 8 步后重查询。训练时对每个样本并行采样多个去噪时间步 $T$（默认 $T=8$),提升收敛。动作在**关节空间**,支持绝对关节位置与关节增量两种参数化。可选 2D 点条件（把目标物体/放置点的归一化坐标以特殊 token 注入指令流,用于空间 grounding）。
- **MolmoBot-Pi0**:**完全复刻 $\pi_0$ 架构**（Paligemma 3B VLM + 流匹配动作专家,用 openpi 代码库）,仅把训练数据换成 MolmoBot-Data、从 Paligemma 权重训起。用途是**在架构不变的前提下** 做对照,隔离出"数据"本身的贡献。绝对关节位置监督。
- **MolmoBot-SPOC**:轻量非 VLA transformer（SigLIP2-Base 视觉 + SigLIP 文本 + 并行动作解码器）,动作用分位数分箱（每维 256 个 bin,按 1st/99th 百分位归一化后按经验分位切分）转成分类问题,交叉熵训练;并行解码器用 $D \times T$ 可学习 query 一次前向出整块动作。面向边缘部署与后续仿真内 RL 微调。

## 三、实验结果

**评测本体**:Franka FR3（DROID 配置,15 Hz,桌面 pick-and-place）与 Rainbow Robotics RB-Y1 移动机器人（开门/抽屉/橱柜/移动抓放）。所有策略**仅在仿真训练,零真机数据、零任务特定微调**。

### 数据引擎规模（MolmoBot-Data，Table 1/2）

| 指标 | 数值 |
| --- | --- |
| 总回合数 | 1.7M（295.2M 帧） |
| 目标物 + receptacle 资产 | 11.4k + 9.4k |
| 独立环境数 | 94.2k |
| 累计时长 | 5,704 小时 |
| 平均回合长 | 11.7 s |
| 生成吞吐 | 100 张 A100 上约 660 成功回合/GPU-时,>88 小时机器人经验/墙钟小时 |

对比 DROID（76k 真机）、Open X-Embodiment（1M+）、AgiBot-World（1M+）、RoboMimic（约 1k）、MimicGen（50k+）、InternData-A1（630k）、RoboCasa-365（500k）,MolmoBot-Data 在回合数（1.7M）与环境多样性（94.2k）上均显著更大,且同时覆盖固定臂与移动操作两种本体。全量数据约 6,500 GPU-时生成,相对真机采集约 **2.6×** 数据吞吐。

### 真机 pick-and-place（DROID 平台,4 环境 × 10 任务 × 3 次 = 120 次/策略）

| 策略 | 真机成功率 |
| --- | --- |
| $\pi_0$-DROID（10k+ 小时真机） | 约 9% |
| $\pi_{0.5}$-DROID（10k+ 小时真机,SOTA） | 约 39.2% |
| MolmoBot-Pi0（同 $\pi_0$ 架构,仿真数据） | 46.7% |
| MolmoBot-Img（单帧） | 约 72.5% |
| **MolmoBot (F=2)** | **79.2%** |

关键读数:MolmoBot-Pi0 与 $\pi_0$ **架构完全相同** 却把真机成功率从 ~9% 提到 46.7%——差异**只能由数据解释**,直接证明 MolmoBot-Data 的多样性足以匹敌甚至超过等量真机数据。（注:Table 6 将 $\pi_{0.5}$ 零样本真机均值记为 31.3%,与摘要/Fig 7 的 39.2% 略有出入,应为不同聚合口径。）

### 仿真留出环境评测（Table 6，每任务 1000 回合，oracle / end 双指标）

| 策略 | Pick MSProc | Pick&Place | PnP Next-To | PnP Color | 仿真均值 |
| --- | --- | --- | --- | --- | --- |
| $\pi_{0.5}$ 零样本 | 18.1 | 11.7 / 7.6 | 8.2 / 6.2 | 10.4 / 6.7 | 10.0 |
| $\pi_{0.5}$-Finetune（15k 步 sim 微调） | 48.0 | 43.5 / 29.7 | 28.4 / 14.7 | 48.3 / 38.9 | 36.0 |
| MolmoBot-Pi0 | 66.2 | 44.7 / 38.2 | 24.7 / 13.3 | 46.2 / 40.0 | 41.5 |
| MolmoBot-Img | 92.2 | 63.0 / 55.0 | 21.0 / 16.4 | 67.8 / 60.3 | 61.6 |
| **MolmoBot (F=2)** | **93.5** | 66.4 / 57.7 | 26.4 / 20.2 | 67.8 / 60.0 | **64.1** |
| MolmoBot (F=3) | 91.3 | 65.4 / 55.6 | 28.3 / 22.6 | 66.1 / 57.3 | 62.4 |

StereoVLA、LAP-VLA、X-VLA 等 VLA 零样本在多数任务上成功率低于 7%（几乎全崩）。在受限单相机（fixed-shoulder）设定下（Table 7）,MolmoBot 变体仿真 91–93%、真机厨房 Pick 达 86.6%（MolmoBot-Img）。

### RB-Y1 仿真（Table 8）

| 策略 | Pick | Pick&Place | Open | Door-Open |
| --- | --- | --- | --- | --- |
| MolmoBot Multitask | 44.8 | 22.5 | 25.2 | 70.2 |
| MolmoBot Door Specialist | – | – | – | 77.7 |
| MolmoBot-SPOC Rigid | 10.5 | 1.8 | – | – |
| MolmoBot-SPOC Articulated | – | – | 21.8 | 58.8 |

移动开门真机 9 次试验中 4 次成功抓把手、2 次成功开门,失败多因把手在门右侧的构型在数据中欠采样,以及硬件 e-stop 触发。

### 数据与模型消融（Fig 8/9）

- **轨迹规模**（10k→25k→50k，固定 5k 房屋/12.4k 物类）:真机与仿真都**单调提升**,尤其真机。
- **物体多样性**（10→100 类）:提升仿真但**对真机几乎无增益**（推测真机评测物体少且语义常见如 apple/cup）。
- **环境多样性**（50→50k 房屋，固定 50k 轨迹）:两侧**几乎无影响**——对 pick 而言,性能由交互数据总量而非背景多样性驱动。
- **去噪采样步 $T \in \{1,2,4,8\}$**:仿真随 $T$ 增单调上升、$T=8$ 最佳;真机 $T=4$ 峰值。
- **动作表示**:绝对关节位置在**真机上显著优于** 增量表示,仿真两者相当——是 sim-to-real 的关键设计选择。

## 四、局限性

1. **受限于仿真器能力**:只覆盖刚体与 articulated 操作,contact-rich（insertion、peg-in-hole）、可形变物（布/绳/食物）、流体/颗粒动力学等现代仿真器保真度不足的任务未涉及,作者明确列为开放挑战。
2. **物体/环境多样性收益递减**:消融显示物体与环境多样性对真机几乎无帮助,说明当前"多样性驱动泛化"的论断在评测覆盖有限时并不完全成立,真机评测物体集偏窄可能掩盖了物体多样性的真实价值。
3. **真机评测样本量小、方差大**:桌面每策略 120 次、开门仅 9 次,且开门受硬件 e-stop 干扰;文内 $\pi_{0.5}$ 真机数字（31.3% vs 39.2%）本身存在口径不一致。
4. **articulated/移动任务绝对成功率仍偏低**:RB-Y1 Pick&Place 仅 22.5%、SPOC 刚体 1.8%,离可靠部署尚远;开门依赖脚本 phase + 点条件,把手右置等 OOD 构型脆弱。
5. **仿真-真机的量化对比受限**:因 RB-Y1 缺乏通用基线,移动操作只有定性真机结果;$\pi$ 系模型的预训练数据不公开,限制了严格复现。

## 五、评价与展望

**优点**。这是"仿真数据规模化 → 零样本操作 sim-to-real"这一命题迄今最有说服力的公开证据之一。最锋利的证据是 MolmoBot-Pi0:与 $\pi_0$ **架构逐比特相同**,仅换纯仿真数据,真机就从 ~9% 跳到 46.7%——干净地把"数据 vs 架构"的贡献拆开,直接反驳了"仿真操作必须靠真机微调补齐"的行业默认。而且它不依赖照片级渲染、不做显式域自适应,只靠 MuJoCo + 大量域随机化,这与 GraspVLA、RoboCasa、Infinigen-Articulated 等程序化仿真路线一脉相承,但把规模（1.7M/94.2k 环境）和本体覆盖（固定臂 + 移动）都推到了新高度,并且**全栈开源**（引擎、数据、三类模型、训练代码）。

**与公开工作的关系**。数据引擎建立在 MolmoSpaces 生态与作者前作导航工作 SPOC 之上,把"程序化房屋 + 最短路径专家 → 零样本导航"的方法论迁移到操作;模型侧 MolmoBot 复用 Molmo2 VLM + DiT 流匹配头,与 $\pi_0$ 的 layerwise 耦合思路相通。相比 MimicGen/RoboCasa 侧重"从少量人示范增广",本文更彻底地走"脚本专家 + CuRobo/IK 全自动生成"的纯合成路线;相比 InternVLA 家族的合成预训练,本文强调**零真机微调** 下的直接迁移。

**开放问题与改进方向**。(1) 消融揭示的"环境/物体多样性对真机无增益"是一个反直觉且重要的信号——很可能是真机评测覆盖太窄导致的度量假象,后续需要更大、更 OOD 的真机 benchmark 来证伪或证实"多样性驱动泛化"。(2) 引擎瓶颈在物理保真度,把 contact-rich 与可形变纳入,可能需要耦合可微/生成式 world-model 仿真器,作者在结论中也点到这一方向。(3) 动作噪声、retry、抓取过滤等"生成侧的数据质量工程"对最终成功率的边际贡献缺乏系统消融,值得单独量化——这些恰是脚本专家路线区别于人示范增广的关键变量。(4) SPOC 分支为"仿真内 on-policy RL 微调"预留了轻量本体,但本文尚未展示 RL 收益,是一个自然的下一步。

## 参考

1. Black et al., *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*, arXiv 2024 —— MolmoBot-Pi0 直接复刻其架构做对照,核心基线。
2. Black et al., *$\pi_{0.5}$: A VLA Model with Open-World Generalization*, CoRL 2025 —— 真机 SOTA 基线,被 MolmoBot 大幅超越。
3. Eftekhar/Ehsani et al., *SPOC: Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World*, CVPR 2024 —— 方法论前身,把"仿真最短路径专家 → 零样本迁移"迁到操作。
4. Kim et al., *MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation*, 2026 —— 数据引擎所依赖的环境/资产/抓取生态。
5. Clark et al., *Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding*, arXiv 2026 —— MolmoBot 的 VLM 骨干。
