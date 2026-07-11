# PRISM：基于图像的场景与运动合成的个性化机器人数据集生成

> **论文**：*PRISM: Personalized Robotic Dataset Generation via Image-based Scene and Motion Synthesis*
>
> **作者**：Dogyu Ko\*, Haneul Kim\*, Chanyoung Yeo\*, Dowoon Lee, Taeho Park, Hyoseok Hwang†（\* 共同一作，† 通讯作者）
>
> **机构**：Kyung Hee University（庆熙大学）
>
> **发布时间**：2026 年 07 月（arXiv 2607.04880）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.04880) | [PDF](https://arxiv.org/pdf/2607.04880)
>
> **分类标签**：`数据合成` `数字表亲` `real-to-sim-to-real` `VLM-TAMP` `VLA`

---

## 一句话总结

PRISM 从**单张 RGB-D 图像 + 一句自然语言指令** 出发,自动生成一批与用户目标环境"语义/几何对齐、但实例级多样"的 digital cousin（数字表亲）场景,再用 VLM-TAMP 无需人工遥操作地合成可执行示范,并用"轨迹不变、只换外观"的重放扩增数据;在 LIBERO / LIBERO-Plus 与三个真实操作任务上,用其数据训练的策略普遍优于 RoboTwin 2.0 与 X-Sim 基线,真实任务成功率最高达**100%**,示范生成成功率**93.46%**。

## 一、问题与动机

大规模预训练 VLA(vision-language-action)模型有很强的零样本泛化,但一旦部署到用户特定环境,性能往往急剧下降——因为策略的能力由其训练轨迹所来自的经验分布决定,而这一分布很少覆盖某个用户的具体环境。因此,弥合这一 gap 需要针对目标环境采集"对齐的数据"。作者把现有采数据的三条路线及各自短板归纳为:

1. **遥操作**:在目标环境里由人直接操作,数据与目标高度对齐,但成本高、难以规模化。
2. **LLM/VLM 合成场景**(RoboGen / GenSim2 等):可自由规模化,但场景来自"与目标无关"的通用分布,丢失了对用户环境的对齐。
3. **数字孪生**(digital twin,如 X-Sim):从传感器重建目标环境恢复了对齐,但把分布**塌缩为单一实例**,没有泛化所需的变化。

作者的核心洞见:个性化数据集生成不应被理解为"采集更多完美对齐的数据",而应被重构为——**构造一个以目标为条件的场景分布**,这些场景共享目标的结构(物体类别、空间关系),但在实例层面(具体物体外观、布局)有变化,并且每个场景都配有能完成任务的示范。这样才能"目标对齐"与"实例多样"二者兼得,而现有仿真方法只能取其一。

## 二、核心方法

PRISM 是一条端到端流水线,分三个阶段(见其 Figure 2)。

### 2.1 阶段一:场景生成(Scene Generation,受 GAIA 启发)

- **物体检测与分割**:先用 VLM 推断场景中物体名称 $\mathcal{O}=\{\mathbf{o}_1,\dots,\mathbf{o}_N\}$,再用 Grounded-SAM 得到实例级 mask。若有 RGB-D 与内参就直接用;若只有单张 RGB,则用 Depth Anything v2 估深度、Perspective Fields 估内参。
- **物体检索(两阶段)**:类别级——用 CLIP embedding 找出与 $\mathbf{o}_i$ 最近的 top-$n$ 类别;实例级——在每个类别内,把分割出的物体图与 3D 资产的渲染图在 DINOv2 特征空间比对取 top-$m$ 近邻,再让 VLM 从中挑出视觉最相似的 top-$k$,得到候选资产集 $\mathcal{C}_i=\{\mathbf{a}_i^{(1)},\dots,\mathbf{a}_i^{(k)}\}$。
- **场景构造**:从深度图 + mask 重建物体点云估计其位置,在该位置放置从 $\mathcal{C}_i$ 中**随机采样** 的一个资产 $\mathbf{a}_i^{(j)}$;可选地由 VLM 补齐指令需要但图中缺失的物体,并加入 distractor、墙面、地板等增加场景级多样性。实验中每个物体检索 8 个候选资产,构造**4 个** digital cousin 场景。

> 用大白话说:先"认出"用户桌上有哪些东西、在哪里,然后不照抄原物,而是从资产库里挑几个"长得像的表亲"随机组合出好几套场景——空间骨架照搬目标,具体是哪一款牛奶盒、哪一个篮子则随机换,从而制造受控的实例多样性。

### 2.2 阶段二:示范生成(Demonstration Generation,受 VLM-TAMP 启发)

- **随机化场景初始化**:轻微扰动物体位置、随机偏移机器人初始关节位形、可选地独立缩放物体尺度,避免示范特化到单一构型。
- **指令解析(两阶段查询)**:第一阶段,给 VLM 输入自然语言指令 $\ell$、标注了 bounding box 的场景图、检测物体列表,让它产出自然语言动作序列 $\hat\pi^{\text{eng}}$;第二阶段,VLM 把它翻译成部分 grounded 的动作序列 $\hat\pi=\{\hat a_1,\dots,\hat a_K\}$,每个 $\hat a_k$ 取自预定义原语动作集 $\mathcal{A}_{\text{prim}}$(如 `pick`、`place`)。
- **轨迹生成**:把问题写成 TAMP 三元组 $\langle \mathcal{O},\mathcal{I},\hat\pi \rangle$($\mathcal{O}$ 为场景物体,$\mathcal{I}$ 为当前状态,$\hat\pi$ 为待细化的 plan skeleton),TAMP planner 为每个部分 grounded 动作搜索抓取位姿、机器人位形、无碰撞路径等连续参数。
- **Motion-Aware Grasp Selection(运动感知抓取选择)**:TAMP 需在解臂部轨迹前先选抓取位姿。PRISM 不随机采样,而是把候选抓取按其相对一组"正面朝向"的规范末端姿态 $\mathcal{R}_c\subset \mathrm{SO}(3)$ 的旋转距离打分。对候选抓取旋转 $R_g$ 与规范姿态 $R_c$,用 $\mathrm{SO}(3)$ 上的测地距离:

$$d(R_g, R_c) = \arccos\left(\frac{\mathrm{tr}\!\left(R_c^{-1} R_g\right) - 1}{2}\right)$$

选出对齐误差最小(即手腕重定向最少)的候选交给 planner。候选抓取由物体轴对齐包围盒的 6 个面枚举得到(碗类再加 4 个从上方接近的抓取)。

> 用大白话说:同一个物体有很多种抓法,随机抓可能导致手腕拧成奇怪角度、轨迹别扭难学。这个公式衡量"某个抓法和机器人自然朝下正视的默认手势差多少度",专挑差得最少的那种,于是生成的手臂动作更平滑、更好被模仿学习复现。

### 2.3 阶段三:数据集构造(Dataset Construction)

- **Trajectory-Preserving Visual Randomization(轨迹保持型视觉随机化)**:传统随机化对每条轨迹配不同随机外观,容易让策略记住"某轨迹—某视觉条件"的捆绑。PRISM 反其道——**固定动作轨迹与全部物理状态,只改光照、背景纹理等视觉因子**,把同一条示范重放多次。这样让策略聚焦于任务相关的状态/动作信息,而对任务无关的外观保持不变。实现上:每个场景合成 20 条候选轨迹、保留最短的 10 条(短轨迹更稳、更不易累积误差),每条再 ×10 视觉随机化重放 → 每场景 100 条、每任务共**400** 条示范,背景纹理取自 RoboTwin 2.0 的纹理库。
- **数据采集**:重放时记录观测、执行动作、任务成功信号(第三人称 + 腕部 RGB,256×256),最后只保留成功示范。

> 用大白话说:合成一条能完成任务的轨迹很贵,合成好后就"换皮不换动作"地重放十遍,一份劳动榨出十份视觉多样的数据,既省算力又逼策略学"外观无关"的技能。

## 三、实验结果

评估用两种策略:LoRA 微调 $\pi_{0.5}$(PaliGemma + Gemma-300M flow-matching action expert)、从零训练 Diffusion Policy(DP)。每任务 400 条轨迹。基线为**RoboTwin 2.0** 与**X-Sim**。In-Domain 为各方法在其原生仿真器内评测,out-of-domain 用 LIBERO(基准)与 LIBERO-Plus(加光照/背景/噪声/布局扰动)。

### 3.1 Sim-to-Sim(成功率 %,其 Table 1)

| 任务 | 策略 | 方法 | In-Domain | LIBERO | LIBERO-Plus |
|---|---|---|---|---|---|
| Put milk in basket | $\pi_{0.5}$ | RoboTwin 2.0 | 74.0 | 14.0 | 21.9 |
| | | X-Sim | **96.0** | 48.0 | 35.8 |
| | | **Ours** | 72.0 | **98.0** | **67.6** |
| | DP | RoboTwin 2.0 | 84.0 | 2.0 | 33.7 |
| | | X-Sim | 84.0 | 80.0 | 2.8 |
| | | **Ours** | **95.0** | **94.0** | **35.6** |
| Put wine bottle on cabinet | $\pi_{0.5}$ | RoboTwin 2.0 | 76.0 | 16.0 | 3.3 |
| | | X-Sim | 94.0 | 82.0 | **54.5** |
| | | **Ours** | **98.0** | **98.0** | 52.0 |
| | DP | RoboTwin 2.0 | 78.0 | 34.0 | 27.2 |
| | | X-Sim | 40.0 | 44.0 | 0.6 |
| | | **Ours** | **100.0** | **56.0** | **28.8** |

关键观察:PRISM 在 out-of-domain(LIBERO/LIBERO-Plus)几乎全面领先,且从 In-Domain 到 LIBERO 的性能跌落在多数设置下**最小**——作者归因于用"结构/语义相似但非精确孪生"的场景训练,提高了数据多样性。X-Sim 的 In-Domain 常最高(它是精确孪生),但跨仿真器迁移时掉得更多。

### 3.2 Real-to-Sim-to-Real(真实,10 次试验成功率 %,其 Figure 4)

| 任务 | RoboTwin 2.0 | X-Sim | **Ours** |
|---|---|---|---|
| Lift cup | 70.0 | 80.0 | **100.0** |
| Box into basket | 80.0 | 50.0 | **80.0** |
| Stack bowls | 70.0 | 40.0 | **80.0** |

真实平台为 Franka Research 3 + RealSense L515。PRISM 在三个真机任务上一致优于或持平最优基线。

### 3.3 关键消融

**数字表亲 vs 数字孪生**(Box into basket,其 Table 2):

| 数据 | 目标环境 | 变体环境 |
|---|---|---|
| PRISM-Twin(单一孪生) | **100.0** | 30.0 |
| PRISM-Cousin(多样表亲) | 80.0 | **80.0** |

孪生在原环境最强但换物体/背景后骤降,表亲在两种环境都稳定——说明表亲以少量保真度换来了防止过拟合单一场景的泛化。

**Motion-Aware Grasp Selection**(其 Table 3):

| 策略 | 随机抓取 | 运动感知抓取 |
|---|---|---|
| $\pi_{0.5}$ | 56.0 | **98.0** |
| DP | 52.0 | **56.0** |

**轨迹保持视觉随机化**(40 条轨迹,变每条重放次数,其 Figure 5):×1 成功率仅 2.0、×5 达 78.0、×10 达 94.0,同时每样本生成时间从 1.00 降到 0.43(归一化)——多重放既提性能又降单样本成本。

**流水线效率**:示范生成成功率 RoboTwin 2.0 = 46.46% / X-Sim = 79.20% / **Ours = 93.46%**(其 Figure 6)。全流程把单张 RGB-D 图变成 400 条示范约需每任务 **1.7–6.6 小时全自动算力**、无任何人工遥操作;五个任务共 2,000 条示范、总体成功率 93.5%。LIBERO-Plus 分维度(Light/Background/Noise/Layout)上 PRISM 亦普遍最优(如 milk 任务 $\pi_{0.5}$:Light 96.0 / Background 90.6,总 67.6)。

## 四、局限性

- **本体与物体受限**:目前仅适配 Franka 单臂与刚体;可变形物体(布、绳、食物)与更多本体留待未来。
- **单图重建的脆弱性**:场景由单张图重建,重度遮挡会导致点云不准 → 物体缩放/放置错误 → 示范生成退化。作者的失败案例分类还指出:资产库中若某资产本身未被校正为直立朝向,会以错位姿态生成;若任务相关物体被遮挡,VLM 据 bounding box 生成的动作序列可能出错,导致轨迹生成失败。
- **依赖多个现成大模型**:VLM(检测/规划/接地)、Grounded-SAM、DINOv2、CLIP、Depth Anything v2 串联,任一环节出错都会传导,且 digital cousin matching 是耗时主项(酒瓶任务约 1.2 小时)。
- **短平任务**:评测多为 pick-place 级、单次抓放的桌面任务(平均 ~127 步、末端路径 ~1.11 m),长程/多阶段/接触丰富任务未充分验证。

## 五、评价与展望

**优点**。(1) 把"个性化数据集"重新表述为"以目标为条件的场景分布"是一个干净且有说服力的问题框架,精准点出孪生(塌缩到单点)与通用合成(丢失对齐)的互补短板,digital cousin 恰好落在中间。(2) 三个设计都务实:motion-aware grasp 用一个 $\mathrm{SO}(3)$ 测地距离就把"易模仿"注入 TAMP;轨迹保持随机化把"合成贵、重放便宜"这一算力结构显式利用起来,消融显示其对成功率和成本的双赢十分显著。(3) 全流程零遥操作、零 per-task RL,对比 X-Sim(需逐任务 PPO 训练奖励模型)在可扩展性上有明确优势;真机 100%/80% 的结果具备说服力。

**不足与开放问题**。(1) 概念上 digital cousin 直接承接 ACDC 与作者自己的 GAIA,场景生成阶段的原创增量主要在"多表亲采样 + 与下游示范/随机化的整合",单看场景生成的新意有限。(2) 基线选择偏窄——只比 X-Sim 与 RoboTwin 2.0,未与 MimicGen/DemoGen/DexMimicGen 等"少量种子示范扩增"路线,以及 DreamGen 等视频世界模型数据路线正面比较;而后两类正是与之竞争的主流数据引擎。(3) In-Domain 上 X-Sim 常更高,暗示"精确对齐"在同分布评测里仍占优,PRISM 的收益本质是"用轻微失配换泛化",这在何种任务/分布偏移强度下划算,缺乏系统刻画。(4) 每任务仅 4 个场景 × 400 条的规模较小,digital cousin 的多样性上限、以及规模扩大后是否边际递减,未有 scaling 分析。(5) 依赖 VLM 语义接地,长程与含遮挡任务的失败率(酒瓶任务示范成功率仅 40.98%、bowls 仅 77.5%)提示接地与 TAMP 可行性是瓶颈。

**可能改进方向**(纯学术视角):把 digital cousin 的实例采样从"随机"升级为"覆盖度/难度感知"的主动采样;引入多视角或视频输入缓解单图遮挡;将轨迹保持随机化与生成式世界模型的外观重渲染结合,进一步拉大视觉分布同时保物理一致;以及给出"对齐-多样性"权衡随分布偏移强度变化的定量曲线,使该框架的适用边界更清晰。

## 参考

1. D. Ko, C. Yeo, D. Kim, J. Kim, H. Hwang. *GAIA: Generating Task Instruction Aware Simulation Grounded in Real Contexts using Vision-Language Models*. IEEE RA-L, 2025.（本文场景生成的直接基础)
2. T. Dai, J. Wong, Y. Jiang, et al. *Automated Creation of Digital Cousins for Robust Policy Learning*(ACDC). CoRL, 2024.（digital cousin 概念来源)
3. P. Dan, K. Kedia, A. Chao, et al. *X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real*. CoRL, 2025.（数字孪生对照基线)
4. T. Chen, Z. Chen, B. Chen, et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization*. arXiv:2506.18088, 2025.(通用合成对照基线)
5. Z. Yang, C. Garrett, D. Fox, L. P. Kaelbling. *Guiding Long-Horizon Task and Motion Planning with Vision Language Models*(VLM-TAMP). ICRA, 2025.（示范生成的规划基础)
