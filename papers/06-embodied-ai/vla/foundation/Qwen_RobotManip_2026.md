# Qwen-RobotManip：对齐解锁机器人操作基础模型的规模化

> **论文**：*Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models*
>
> **作者**：Qwen Team（核心贡献者 Haoqi Yuan、Zhixuan Liang、Anzhe Chen、Ye Wang、Haoyang Li、Pei Lin、Yiyang Huang、Zixing Lei、Tong Zhang 等；通讯 Xiong-Hui Chen，含 An Yang、Fei Huang、Junyang Lin、Dayiheng Liu、Jingren Zhou、Chenfei Wu、Jiazhao Zhang 等）
>
> **机构**：阿里巴巴 Qwen 团队 + 具身智能合作方（致谢 National Pilot Base for Embodied Intelligence、AgileX Robotics，及 Hao Dong、Yao Mu 教授）
>
> **发布时间**：2026 年 6 月（arXiv 2606.17846，v2）
>
> **发表状态**：技术报告（arXiv 预印本，cs.RO；未见会议/期刊录用）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.17846) | [PDF](https://arxiv.org/pdf/2606.17846) | [Blog](https://qwen.ai/blog?id=qwen-robotmanip) | [代码](https://github.com/QwenLM/Qwen-RobotManip)
>
> **分类标签**：`机器人操作基础模型` `跨本体对齐` `人到机器人数据合成` `camera-frame delta pose` `统一状态-动作表征` `DiT flow matching` `OOD 评测`

---

## 一句话总结

Qwen-RobotManip 把大模型"对齐异构数据 → 规模化训练"的配方搬到机器人操作，核心主张是 **对齐是规模化的前提而非可选项**（alignment first, then scale）：先用**统一状态-动作表征 + 相机系相对位姿 + in-context 行为适配**三层对齐消除跨本体数据的表示冲突，再用一条**人到机器人合成流水线**把 1,933 小时 egocentric 人类视频渲染成跨 15 个平台的 24,808 小时机器人轨迹，最终只靠开源数据 + 人类视频拼出 ~38,100 小时语料（零私有数据），在一批 OOD 基准上全面超 π₀.₅、在 RoboChallenge Table30-v1 generalist 赛道排名第一（相对提升 20%），并涌现出零样本指令跟随、跨本体迁移和失败自恢复能力。

---

## 一、问题与动机

### 1.1 大模型的 scaling recipe 能否搬到操作

语言/多模态基础模型之所以泛化强，靠的是两点：异构数据能在**统一表述**下对齐，海量低成本数据让不同训练信号在规模上**相互增强**。本文要回答：这套配方能不能用到机器人操作、换来真正的泛化？困难在于——操作数据天生异构、采集昂贵、多样性窄，"对齐"和"规模"很难同时拿到。

### 1.2 现有 VLA 的泛化大多是"表面的"

作者抛出一个尖锐观察并用实验坐实：在 LIBERO / RoboTwin 这类 **in-distribution 基准**上，**从头训练的模型能追平甚至超过大规模预训练的模型**。因为训练与评测同分布，靠记忆反复出现的视觉/行为模式就能刷高分，基准根本区分不了"真泛化"与"模式记忆"。既有工作报告的 OOD 大多只是轻微视觉扰动、保持同一本体/任务结构，一旦超出这个窄范围性能就崩。

### 1.3 预训练先验迁移不动的两个根因

1. **数据集中在窄 teleop 设置**：本体和任务多样性远不足以让 scaling 生效；
2. **多样性本身不够，还缺对齐**：当不同本体的观测与动作表示不兼容时，堆数据带来的是**干扰而非协同**——同一物理运动在不同本体上数值不一致，额外数据无法转化为能力。

由此得到本文的中心命题：**对齐不是独立的工程选择，而是数据规模化本身的前置条件**。Qwen-RobotManip 围绕"表征、运动、行为"三个维度做统一对齐，让大规模多源训练变得协同而非冲突。

---

## 二、预备知识

- **动作的坐标系问题**：末端动作可以表示为机器人 base 系的绝对位姿、末端局部系的相对位姿，或世界系增量。不同数据集选择不同 → 同一技能在动作空间里数值不一致，模型被迫花容量去"调和坐标系"而非学操作本身。本文用**相机系相对位姿**解决（见 §3.3）。
- **Flow matching 动作专家**：给定动作块 $\mathbf{a}$，采样时间步 $t\sim\text{Beta}(1,1.5)$ 与噪声 $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})$，构造插值 $\mathbf{x}_t=(1-t)\boldsymbol{\epsilon}+t\,\mathbf{a}$，训练网络预测速度场 $\mathbf{v}=\mathbf{a}-\boldsymbol{\epsilon}$；推理时 4 步 Euler 积分即可出动作，低延迟。这是 π₀ 以来 VLA 动作头的主流做法。

---

## 三、核心方法

### 3.1 三维对齐框架总览

Qwen-RobotManip = **Qwen3.5-4B 视觉语言骨架 + DiT flow-matching 动作专家**，围绕三层对齐设计：

1. **表征对齐**：统一 80 维 canonical 状态-动作向量 + 逐维二值 mask，把不同形态塞进同一模板；
2. **运动对齐**：末端动作用相机系相对位姿参数化，让"视觉上相似的运动在动作空间也数值相近"；
3. **行为对齐**：in-context policy adaptation，读同一 episode 的执行历史作隐式本体标识，部署时不改参数即可适配。

配合**双流协同训练**（操作数据 + VL 数据同训）保住 VLM 的感知与推理能力。

### 3.2 表征对齐：统一状态-动作向量

一个 **80 维 canonical 向量**：两个 29 维的单臂块 + 22 维预留。每个单臂块按语义分组：

- **关节位置**（7 维）；
- **末端位姿**（9 维：3 位置 + 6D 连续旋转表示）；
- **夹爪状态**（1 维：平行夹爪开合）；
- **灵巧手关节**（12 维：多指手）。

尾部 22 维预留给移动底盘速度等额外自由度。状态用绝对坐标；动作里关节是绝对值、末端是相对当前状态的增量（末端朝向增量用 3 维 axis-angle 而非 6D）。不同本体只填自己拥有的子集（如 7-DoF 单臂夹爪只填一侧的关节+末端+夹爪），其余置零。

**关键**：零填充维通过**逐维二值 mask 排除出损失**，梯度只流过语义上真实存在的自由度，避免对结构性缺失的维度施加虚假监督。

### 3.3 运动对齐：相机系相对位姿

不在 base 系表示末端运动，而在**相机系**表示相对位姿增量。核心性质：**图像里看起来相似的运动，在动作空间里数值也相近**——直接把动作表示与视觉观测对齐，从而促成跨本体迁移。记相机系为 $c$、当前末端系为 $e$、目标末端系为 $e^*$，预测动作的位姿分量本质是把"末端相对运动"投影进相机坐标；论文给出一个更紧凑的等价式：

$$\mathbf{a}_p = {}^{c}_{e^*}\mathbf{T}\,\left({}^{c}_{e}\mathbf{T}\right)^{-1}$$

但其平移分量与相机到末端偏移耦合、对标定误差更敏感，故实现上采用展开的旋转/平移分块形式（对长尾分布与标定误差更鲁棒）。代价是训练/推理都需要标定好的相机内外参。

**Camera-aware 位置编码（CaPE）**：把相机位姿经 Camera Positional Encoding 注入 DiT 交叉注意力（占 64 维注意力头的一半，另一半留给 RoPE 做时序索引），仿 GTA/PRoPE 不仅加到 K/Q 还加到 V 和输出，强化几何一致性；相机内参则把图像 patch 归一化坐标线性投影后加到对应视觉 token。因为 CaPE 是旋转型编码，全局世界原点在点积注意力里代数抵消，只留下视觉 token 与状态/动作 token 之间的相对位姿。

此外 DiT 还经 AdaLN 额外条件于两个信号：**末端类型 embedding**（单臂/双臂左/双臂右/头戴/移动底盘）和**辅助标志 embedding**（是否有标定相机参数——在相机系 delta 模式与 base 系相对模式间切换）。

### 3.4 行为对齐：in-context 策略适配

把同一 episode 内的**执行历史**当作隐式"本体身份证"：每个 decision step 观测当前视觉+本体状态，预测 $K$ 步动作块，于是一个 context chunk 定义为三元组 $(\mathbf{o}_h,\mathbf{s}_h,\mathbf{a}_h)$（观测、本体状态、该 chunk 执行的动作序列），$H$ 个这样的 chunk 给策略一个可直接推理的近期行为窗口，部署时无需任何参数更新即可适应新机器人/新环境的运动学。历史帧前置进 VLM 单次前向；本体状态与动作块经轻量 MLP 投影进 VLM 隐空间。采用 **unified 模式**（context 拼到 VLM 输入端，与语言/视觉一起做因果自注意力）而非仅注入 DiT。

**两个关键坑**：

- **action-copy 捷径**：naive 地总喂最近 $H$ 个 chunk，模型会学会"抄最近一个动作"——训练 loss 很低但成功率差。用 **stochastic context sampling**（训练时从 episode 随机位置采历史 chunk）打破，逼模型学整段 episode 的行为风格而非利用时间邻近性。
- **去噪步数**：in-context 的收益要 **10+ 去噪步**才解锁（4 步下反而因动作分布变复杂而抖动）。

### 3.5 人到机器人合成流水线（scaling 引擎）

把 1,933 小时 egocentric 人类视频渲染成**跨 15 个双臂平台、24,808 小时**的机器人轨迹，分两阶段（Fig 1）。

**动作对齐（retarget + 平滑）**：用 MANO 3D 手关键点定义虚拟手指，把人手映射到平行夹爪：

$$\mathbf{k}_{vf} = 0.7\,\mathbf{k}_{index} + 0.3\,\mathbf{k}_{middle}, \qquad \mathbf{p} = \tfrac{1}{2}\left(\mathbf{k}_{thumb} + \mathbf{k}_{vf}\right), \qquad w = \|\mathbf{k}_{thumb} - \mathbf{k}_{vf}\|_2$$

末端位置 $\mathbf{p}$ 取拇指与虚拟手指中点，夹爪宽度 $w$ 取两者距离。朝向构造右手正交系 $[\mathbf{x}\ \mathbf{y}\ \mathbf{z}]$：抓取轴 $\mathbf{z}$ 沿拇指→虚拟手指（乘符号 $s=\pm1$ 做左右手翻转，保证左右手映射到同一夹爪系），$\mathbf{y}$ 为抓取平面法向，$\mathbf{x}$ 为接近方向。逐帧检测的高频噪声用 Savitzky-Golay 滤位置/宽度、高斯加权 SLERP 平滑朝向。

**视觉对齐（换手 + 渲染 + 合成）**：

1. **SAM3** 文本提示分割人手臂得二值 mask；
2. **ProPainter** 光流引导 inpainting 抹掉人手，得干净背景序列 $\{\hat{I}_t\}$；
3. **base 位置搜索**：egocentric 轨迹没有物理机器人底座参考，把底座放置建模为优化——在轨迹质心附近网格搜索、按每个形态最大 reach $r_{\max}$ 约束，最大化 IK 可行的关键帧数量：

$$\mathbf{T}^*_{base} = \arg\max_{\mathbf{T}_{base}} \frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}} \mathbb{1}\!\left[\text{IK}\!\left(\mathbf{T}_{base}^{-1}\mathbf{T}^{ee}_k\right)\ \text{可行}\right]$$

15 个形态各搜各的（臂长/关节各异）；
4. **MuJoCo IK** 渲染机器人图 $I^{robot}_t$ 与深度 $D^{robot}_t$；**Depth Anything v3** 估场景深度 $D_t$，用遮挡 mask 做深度引导合成：

$$M^{occ}_t = \mathbb{1}\!\left[D^{robot}_t \le D_t\right], \qquad I^{syn}_t = M^{occ}_t \odot I^{robot}_t + (1 - M^{occ}_t)\odot \hat{I}_t$$

每条人类演示渲染成 15 种双臂机器人配置（Panda、UR5e、ARX-L5、xArm7、Sawyer、Kinova Gen3、IIWA、Jaco、FR3、UR10e、ViperX、WidowX、Piper、YAM、AgileX ALOHA）。

**动作速度对齐**：人手操作比机器人 teleop 快得多，按源逐帧下采样匹配机器人速度（EgoDex → 60%、EgoVerse → 45%、VITRA → 25%）。

### 3.6 数据清洗（5 阶段信号过滤 + 3 项跨模态检查）

跨本体/仿真/采集管线聚合会引入异构噪声。**五阶段状态-动作过滤**：

1. **突变检测**：级联中值 + SG 平滑得趋势，用残差/加速度/jerk 三重偏差旗标异常帧；
2. **状态-动作趋势对齐**：动作应与状态变化时序一致（因果不变量），用滞后对齐一阶差分的方向一致性 DA 打分，低于阈值剔除——**发现 RoboMIND UR 型 81% 的 episode 没通过、整体剔除**；
3. **极值过滤**：按本体统计 $q_1/q_{99}$，超出 $[q_1-\alpha(q_{99}-q_1),\ q_{99}+\alpha(q_{99}-q_1)]$ 的帧剔除（夹爪维双峰分布豁免）；
4. **FK 一致性**（偏数据修正而非过滤）：Pinocchio 算 FK 对比记录末端位姿，修 TCP 定义常量偏移、把肩相对双臂位姿转世界系——发现同一机器人型号在不同数据集里关节符号约定不同，进一步佐证需要统一表征；
5. **base 系与朝向对齐**：逐数据集旋转校正，保证 $+x$ 一致对应机器人正前方。

**三项跨模态检查**：C1 指令一致性（三段式 VLM：子任务切分 → 推理引导打分 → 多模型交叉裁决）；C2 视频-状态一致性（渲 URDF + SAM3 mask 算 IoU，低于阈值先优化相机参数、否则剔除）；C3 视频质量过滤（黑帧/损坏/模糊/长静止段）。

### 3.7 架构与 embodiment prompt

- **VLM 骨架**：Qwen3.5-4B（原生多模态、早期视觉语言融合，ViT 动态分辨率合并的视觉 token 与文本 token 交织），输出末层隐状态供动作专家 cross-attention。
- **DiT 动作专家**：$N=10$ 层、隐维 768、12 头；每块对拼接的状态-动作 token 自注意力，再 cross-attention 到 VLM 隐状态（交替 attend 偶数块的**视觉** token 与奇数块的**语言** token）；本体状态经两层 MLP 前置进噪声动作序列。
- **结构化 embodiment prompt**：`embodiment`（如 `robot_aloha`）、`instruction`、`speed`（500 步分桶）、`fps`、`camera view direction`（`arm side`/`opposite side`）。训练时以 15% 概率随机丢弃 embodiment/speed/fps 字段，鼓励部分信息缺失时的鲁棒性。

### 3.8 训练与部署

- **双流协同训练**：VLA 流（全多源操作语料）+ VL 流（~28M 视觉语言监督，6 类含 general VQA / 空间推理 / OCR / 多模态知识 / 指令跟随 / 具身中心数据——后者含 ECoT、egocentric 视频理解、2D 轨迹预测；ECoT 用 Qwen3.6-Plus thinking 模式合成）。两流为**互斥的独立 batch**，比例 **9:1**（机器人:VL）。
- **损失**：带 mask 的 flow matching（逐维 slot mask ∧ step 有效性 mask ∧ 逐手有效性 mask，AND 合并，逐样本按有效项平均，防止填充维多的本体主导优化）+ VLM 自回归 next-token（权重 $\lambda=0.1$）。每样本重复 8 次扩散步摊薄 VLM 前向开销。
- **后训练**：domain-specific SFT 走 generalist 范式（域内所有任务合成一个训练集训一个模型），只用 flow matching、关掉 VL 损失与数据清洗、加 color jitter；可选 **mixed post-training**（混入 VL + 预训练 VLA 数据）以防"VLA→VA 退化"（模型退化成忽略语言的视觉-动作模式匹配器）。
- **部署**：远端服务器 + WiFi，用 Real-Time Chunking（RTC）边执行当前 chunk 边异步生成下一 chunk，藏往返延迟。

---

## 四、实验结果

### 4.1 标准基准不够用 → 转向 OOD

在 LIBERO / RoboTwin 上，**从头训练的模型（StarVLA、Ours-scratch）追平甚至超过 π₀.₅ 等预训练模型**（LIBERO 98.0~99.2 一片、RoboTwin 也接近），因为同分布评测靠模式记忆即可。切到 OOD（LIBERO-Plus、RoboTwin-Clean2Rand）差距立刻拉开：StarVLA 从 RoboTwin-Easy 的 85.7% 崩到 Clean2Rand 的 10.6%。结论：**OOD 才是衡量基础模型真泛化的 north star**。为此作者用了 LIBERO-Plus / RoboTwin-Clean2Rand / RoboCasa365 / EBench，并新提两个基准：**RoboTwin-IF**（指令跟随：同场景换指令，五个子套件考目标接地/空间关系/多步序列/动词辨别）和 **RoboTwin-XE**（零样本跨本体迁移）。

### 4.2 主结果：OOD 全面领先

| OOD 维度 | π₀.₅ | Qwen-RobotManip | 增益 |
| --- | --- | --- | --- |
| LIBERO-Plus（总） | 84.4 | 89.0（-Context 91.4） | +7.0 |
| RoboTwin-C2R Hard | 47.9 | 69.4 | +21.5 |
| EBench（总） | 27.1 | 45.6 | +18.5 |
| RoboCasa365（总） | 16.9 | 35.9 | +19.0 |
| RoboTwin-IF（指令跟随均值） | 49.6 | 72.2 | +22.6 |

- **跨本体 RoboTwin-XE**：只在 AgileX ALOHA 上训、零样本迁移到 ARX/UR5/Franka，相机系末端动作达 **23.9%**，是 π₀.₅（7.5%）的 **3.2×**；关节空间迁移几乎随机（<5%）——直接坐实"相机系动作"是跨本体的关键。
- **RoboChallenge Table30-v1 generalist 赛道排名第一**：45% 成功率 / 59.83 process，超次优 DM0（37% / 48.43）约 8 个点。在需要紧密双臂协调的 8 个任务上均值 40%（π₀.₅ 21.2%），且是**唯一**在 *pour fries into plate* 上非零（30%）的模型。

### 4.3 真机验证

- **CobotMagic ALOHA**：7 个 ID 任务均值 88.6%（π₀.₅ 42.9%、StarVLA 20%）；4 个 OOD 任务（杂乱背景/未见物体/光照扰动）均值 87.5%（π₀.₅ 37.5%、StarVLA 0%）。
- **ARX ALOHA**：少样本适配 5 任务 4/5 胜出；跨本体技能迁移（6K CobotMagic + 130 ARX 联训、4 个零演示新任务）均值 55.0%，是最优消融变体的 4×。
- **涌现行为**：反复观察到**自发重试/失败恢复**（抓取滑脱后自主重抓而非僵住），作者归因于预训练数据里天然含"失败后纠正"的片段。

### 4.4 消融：对齐才能 scale（核心证据）

- **数据 scaling law**：用统一表征（UnifiedEEF、Ours）时，验证 MSE 随数据量 1%→100% 近似 **log-linear 下降**；去掉统一空间（w/o UnifiedSpace）曲线抖动、不 scale。→ **对齐是把"更多数据"转成"更强能力"的开关**。
- **相机系末端（camera-frame EEF）三重收益**：in-distribution 控制（RoboTwin-C2R EEF 模式 72.5/56.6，且是唯一 EEF 模式反超关节模式的变体）→ 技能组合（新任务迁移 55.0% vs 次优 12.5%）→ 零样本跨本体（23.9% vs 关节 14.5%），逐级放大。
- **H2R 合成**：robot-only 54.7 →+raw ego 55.0 →**+H2R 58.7**（RoboTwin-C2R Hard），在相机视角/光照维度增益最大；LIBERO-Plus 87.1→89.0。
- **VL 协同训练**：去掉后越难的基准掉得越多（RoboTwin-C2R hard −8.2、RoboTwin-IF −7.0）。
- **UnifiedEEF 是 mixed post-training 的独占前提**：没有它，想混 VL+VLA 做后训练会**直接崩到 0.0%**；有它才能从 71.6% 推到 **75.8%**。
- **架构**：VLM 末层隐状态作 KV 的 **last-layer cross-attention** 最优（LIBERO-Plus 87.5）且最省算力。
- **in-context**：需 10+ 去噪步解锁（均值 70.9，+5.0 over structure prompt），stochastic sampling 必需。

---

## 五、局限性与未来方向

- **H2R 合成的分布 gap**：retarget 近似误差 + inpainting artifact 限制了合成数据的有效质量上限；
- **OOD 评测仍以仿真为主**，需要更广的真实世界部署评测；
- **固定 action chunk + 推理延迟**限制了对亚秒级反应式控制的适用性；
- 未来方向：纳入更多本体/任务域、用更精确的手-机器人 retarget 和物理接地渲染提升合成保真、引入 agentic 系统做更长时程推理。

---

## 六、个人思考

- **真正的洞见是"对齐是 scaling 的开关"这一因果判断，而非某个单点技巧**。§4.4 的 scaling law 消融最有说服力：同样加数据，有统一表征就 log-linear 涨、没有就不涨。这把"VLA 要不要大规模预训练"的争论从"要/不要"重构成"先对齐、才谈得上 scale"，比单纯报 SOTA 更有方法论价值。它与 [LDA-1B](../../world-action-models/LDA_1B_2026.md) 的判断异曲同工（LDA 用统一世界模型 + DINO 潜在解锁 UWM 的 scaling），只是 Qwen-RobotManip 把重心放在**动作/状态表示的跨本体对齐**而非潜在空间。
- **camera-frame delta pose 是这篇最可复用的设计**。它把动作表示与视觉观测强绑定，让"看起来像的动作数值也像"，从而跨本体可迁移——这条线和 *Grounding actions in camera space*（观测中心 VLA）、以及本仓库 [SF](../perception/SF_2025.md) 那种"对齐视觉表征"的思路同源。RoboTwin-XE 上 3.2× 的零样本迁移是硬证据。任何做跨本体/人到机器人的工作，如果还在用 base 系或世界系动作，这是最该改的地方。
- **与相邻路线的坐标**：[π₀.₅](pi05_2025.md) 也强调异构多源协同训练，但仍在 BC 框架、要动作标注；Qwen-RobotManip 用 H2R 合成把"无动作人类视频"变成 24,808 小时带动作的机器人数据，这是它敢喊 38,100 小时的底气，思路上更接近 Being-H0/EgoDex 这条"人类视频预训练"线，但落点是**渲染成真实机器人像素+动作**而非潜在动作。
- **它对 benchmark 的批判呼应了 [WAM 综述](../../world-action-models/WAM_Survey_2026.md) 的评测挑战**：in-domain 指标系统性高估能力、掩盖模式记忆。两篇都在推动"评测方法必须与模型一起进化"。Qwen-RobotManip 更进一步造了 RoboTwin-IF/XE 两个诊断性基准，把"指令跟随是否真接地""动作能否跨本体"单拎出来测，这比笼统的 OOD 成功率更有区分度。
- **一个值得追问的点**：H2R 合成是它的 scaling 引擎，但作者也承认 retarget + inpainting 的分布 gap 是有效质量的天花板。消融里 +H2R 相对 +raw ego 只多 +3.7（Hard），说明合成数据的边际收益已不算大——**合成保真度**很可能是这条路线下一个瓶颈，也正是人到机器人数据合成这类工作值得深耕的方向（更准的手-夹爪 retarget、物理接地的接触/遮挡渲染、更真实的动作速度分布）。

---

## 参考

- **π₀.₅**（Black et al., 2025，CoRL / arXiv 2504.16054）：异构多源协同训练 VLA，本文最强对比基线
- **π₀**（Black et al., 2024，RSS / arXiv 2410.24164）：Flow Matching 动作专家的开山，本文动作头范式来源
- **GR00T-N1**（Bjorck et al., 2025，arXiv 2503.14734）：开源人形基础模型，OOD 对比之一
- **LDA-1B**（Lyu et al., 2026，RSS / arXiv 2602.12215）：统一世界模型 + 通用数据摄入的 scaling，"对齐/统一空间解锁 scale"的同源判断
- **Grounding actions in camera space**（Zhang et al., 2026，AAAI）：观测中心 VLA，相机系动作表示的直接来源
- **EgoDex / Being-H0**（Hoque et al., 2025；Luo et al., 2025）：egocentric 人类数据与人类视频预训练，H2R 数据源与思路参照
- **RoboTwin 2.0**（Chen et al., 2025，arXiv 2506.18088）：RoboTwin-IF / RoboTwin-XE 两个新基准的底座
- **SAM3 / ProPainter / Depth Anything v3**：H2R 视觉对齐三件套（分割 / inpainting / 深度合成）
