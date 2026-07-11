# CRAFT：用于双臂机器人数据生成的视频扩散

> **论文**：*CRAFT: Video Diffusion for Bimanual Robot Data Generation*
>
> **作者**：Jason Chen, I-Chun Arthur Liu, Gaurav S. Sukhatme, Daniel Seita
>
> **机构**：University of Southern California（Thomas Lord Department of Computer Science）
>
> **发布时间**：2026 年 04 月（arXiv 2604.03552）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.03552) | [PDF](https://arxiv.org/pdf/2604.03552)
>
> **分类标签**：`视频扩散造数据` `双臂操作` `跨本体迁移` `数据增强` `Sim2Real`

---

## 一句话总结

CRAFT 用 Canny 边缘控制视频 + 真机参考图 + 语言指令三路条件化一个视频扩散模型（Wan2.1-Fun-Control），把少量真机双臂遥操作演示在**一条统一管线** 里扩增成覆盖物体位姿/光照/物体颜色/背景/相机视角/腕部与第三人称多视图/跨本体七个维度的照片级演示；跨本体从 UR5 迁到 Franka 时仅用 1000 条**纯生成、零目标机真机** 的数据,把 ACT 仿真成功率从 11.3→82.6% / 6.0→89.3% / 21.6→86.0%,真机做到 17/15/16(满分 20),反超在目标机上实采 100 条演示的上界。

## 一、问题与动机

双臂(bimanual)操作的模仿学习受制于真机数据的**采集成本** 与**视觉多样性狭窄**:一批遥操作演示往往只覆盖固定的光照、背景、物体配置、相机视角和单一机器人本体,导致策略在部署时对分布偏移不鲁棒、泛化差。

现有生成式数据增强大多只攻**某一个子维度**:有的只增强第三人称视图、有的只增强腕部相机、有的只做跨本体迁移,彼此割裂,缺乏一条能同时覆盖多维视觉变化的统一管线。作者据此提出问题:能否用一个视频扩散模型,在保持双臂**协调约束与接触动力学** 的前提下,把一小撮真机演示统一扩增成大规模、视觉高度多样、且**带动作标签** 的照片级演示?

关键洞见:用 **Canny 边缘** 作为视频扩散的控制信号,恰好在"保留机械臂与物体的显著结构轮廓"与"抹掉仿真渲染的低层细节、给外观留出自由度"之间取得平衡——从而既锚定运动结构、又能任意改写视觉外观。

## 二、核心方法

CRAFT(Canny-guided Robot Data Generation using Video Diffusion Transformers)分三个阶段。

**问题设定。** 策略 $\pi_\theta$ 从第三人称 RGB / 腕部相机 / 或二者拼接的观测学习。给定 $M$ 条真机演示 $\mathcal{D}^{\text{real}} = \{\tau_1^{\text{real}}, \dots, \tau_M^{\text{real}}\}$,每条演示是观测-动作序列

$$
\tau_i^{\text{real}} = (I_1^g, a_1^l, a_1^r, \dots, I_T^g, a_T^l, a_T^r),
$$

其中 $a_t = \langle a_t^l, a_t^r \rangle$ 分别是左/右臂的目标关节位置与夹爪动作。目标是合成一个 $\mathcal{D}^{\text{gen}}$,满足 $|\mathcal{D}^{\text{gen}}| \gg |\mathcal{D}^{\text{real}}|$,其中每条合成演示的观测 $I_t^d$ 视觉上逼近真机图像,最终在 $\mathcal{D}^{\text{real}} \cup \mathcal{D}^{\text{gen}}$ 上训练策略。

> 用大白话说:先有一小把真机遥操作数据,想变出一大堆"看起来是真机拍的、但场景/视角/机器人各不相同"的假数据,而且每帧都配好了左右手该怎么动的标签,拿去一起训练。

**阶段一:轨迹扩展(Trajectory Expansion)。** 先用数字孪生(digital twin)管线把真机演示搬进仿真:用 AprilTags 定位物体、用 RoboTwin 已知物体网格,在仿真里回放每条真机轨迹 $\tau_i^{\text{real}}$,得到源视频 $\mathbf{V}^s$ 和对应仿真轨迹 $\tau_i^{\text{sim}}$,规模与原数据相同。为扩大规模,借鉴 DexMimicGen:把每条轨迹按物体为中心切成若干子任务段,对某段施加一个变换算子 $\mathcal{T}$ 生成与新采样场景配置一致的候选轨迹 $\mathcal{T}(\tau_i^{\text{real}})$,在仿真里执行并**只保留任务成功的** 轨迹,以此扩充 $\mathcal{D}^{\text{sim}}$。成功轨迹渲染成源视频后,通过滤出显著结构边缘提取出 Canny 边缘控制视频 $\mathbf{V}^c$。

**阶段二:视频生成(Video Generation)。** 视频扩散学习条件分布

$$
p_\phi(\mathbf{V}^d \mid I^{\text{ref}}, \mathbf{V}^c, \ell),
$$

其中 $\mathbf{V}^c$ 是 Canny 边缘控制视频、$I^{\text{ref}}$ 是真机参考图、$\ell$ 是语言指令,骨干用 **Wan2.1-Fun-Control**(支持 Canny 边缘/深度/骨架三种控制)。作者刻意选 Canny 而非深度或骨架:骨架只编码机械臂结构、丢掉物体信息;深度保留过多场景细节、削弱外观改写自由度;Canny 边缘两者兼顾。**在固定 $\mathbf{V}^c$(即固定运动结构)的前提下,只改变 $I^{\text{ref}}$ 与 $\ell$,就能合成外观各异但动作一致的目标视频 $\mathbf{V}^d$**,并用 LLM 自动生成 $\ell$ 的语义变体。

> 用大白话说:把仿真视频"抽成简笔画"(Canny 边缘)当骨架,再喂一张真实照片当"上色参考"和一句话描述,让扩散模型照着简笔画重画一遍——骨架不动保证机器人怎么动不变,换参考图/换描述就能换光照、换背景、换颜色。

**阶段三:增强数据集构建(七个增强维度)。**

- **物体位姿**:阶段一里对目标物体施加从可行工作空间均匀采样的随机平移/旋转;并发现用"夹爪-物体接触"的参考图能显著提升生成视频的接触保真度。
- **光照**:用图像生成模型 Veo3 对参考图 $I^{\text{ref}}$ 提示合成不同环境光(蓝/绿等)的变体,保留阴影与表面反射(优于简单色彩抖动)。
- **物体颜色**:用**无任何物体的空桌参考图**,由 Canny 控制视频提供物体轮廓、由 $\ell$ 指定颜色;若参考图里含物体会把颜色锚死。
- **背景**:干脆**去掉参考图** $I^{\text{ref}}$,改由 $\ell$ 描述背景,用 LLM 批量生成多样背景描述。
- **跨本体(cross-embodiment)**:对源机器人每帧用正运动学取末端位姿 $p_t^l = \text{FK}(a_t^l),\ p_t^r = \text{FK}(a_t^r)$,再用逆运动学求目标机器人关节 $\hat{a}_t^l = \text{IK}(p_t^l),\ \hat{a}_t^r = \text{IK}(p_t^r)$,并保留源轨迹的夹爪动作;重定向后的演示在仿真里回放生成 $\mathbf{V}^c$,再合成目标本体视频。
- **相机视角**:在仿真里加最多 $N \le 4$ 个相机,把多视图拼成一张 tiled 源图 $I_t^{s,\text{tile}}$ 再提 Canny,参考图同样拼接,一次并行合成多视角且保持各 tile 空间独立。
- **腕部 + 第三人称**:把左腕、右腕、第三人称(外部)相机拼成一张图、第四格留空,提 Canny 后合成。

> 用大白话说:七种增强其实是同一套"简笔画 + 参考图 + 一句话"的排列组合——想换颜色就给空桌照片、想换背景就不给照片、想换机器人就先用 FK/IK 把动作翻译到新机械臂再重画。跨本体那一步的巧妙在于:动作从关节空间经末端位姿桥接过去,夹爪开合直接照抄,不用在目标机上采一条真数据。

## 三、实验结果

下游策略统一用 **ACT**(Action Chunking with Transformers)。仿真基准为 RoboTwin(改造);真机为双臂 Franka Research 3 + GELLO 遥操 + Intel RealSense D435i。视频生成**零样本** 使用 Wan2.1-Fun-Control 1.3B;ACT 训练用单张 RTX 4090,小规模生成用单张 RTX 5090,大规模生成分布在 3 张 RTX 6000 上;输入图像中心裁剪补边缩放到 $512 \times 512$。所有方法在 3 个随机种子、相同 train/val/test 划分下比较。

**跨本体迁移(表 I):UR5 → Franka Panda,仿真三任务 Lift Pot(LP)/Place Cans in Plasticbox(PC)/Stack Bowls(SB)。** CRAFT(Ours)用 1000 条纯生成数据、**零目标机真机**;Collected Target 是在目标机采 100 条演示的上界。

| 方法 | 仿真 LP | 仿真 PC | 仿真 SB | 真机 LR | 真机 PC | 真机 SB |
|---|---|---|---|---|---|---|
| Collected Target(上界,采 100 条) | 55.0% | 69.0% | 59.0% | 5/20 | 2/20 | 3/20 |
| Shadow(分割掩码编辑) | 2.0% | 2.3% | 6.0% | 2/20 | 1/20 | 1/20 |
| CRAFT (Target)(仅重定向,无增强) | 11.3% | 6.0% | 21.6% | 4/20 | 1/20 | 2/20 |
| **CRAFT (Ours)(1000 条生成)** | **82.6%** | **89.3%** | **86.0%** | **17/20** | **15/20** | **16/20** |

CRAFT(Ours)不仅碾压 Shadow,还在**从未采集任何目标机数据** 的情况下反超实采上界,说明多样化物体位姿的数据生成可作为目标机采数的可扩展替代。

**Canny 消融(表 II):Stack Bowls,150 条,无任何数据增强,只隔离 Canny 条件对生成质量的影响。**

| 变体 | Stack Bowls 成功率 |
|---|---|
| Collected Demos(上界) | 59.0% |
| CRAFT w/o Canny(直接用原始仿真图) | 10.3% |
| **CRAFT w/ Canny** | **21.6%** |

Canny 条件把成功率翻了近一倍——原始仿真图保留过多低层细节,让扩散难以对齐夹爪-物体接触等显著结构,导致合成退化。

**真机五维增强(表 III,满分 20):Lift Roller(LR)/PC/SB。** 每个维度都在"只变该维度、其他视觉因素固定"的测试条件下评估;各维度基线:光照=Color Jitter(RoboSplat)、背景=RoboEngine、相机视角=Fine-Tuned VISTA、物体颜色=SAM3。CRAFT(Ours)用 1000 条生成 + 原始真机演示(实采基数:LR 100 / PC 200 / SB 150)。

| 维度 | 方法 | LR | PC | SB |
|---|---|---|---|---|
| 光照 | ACT w/o Aug | 3 | 1 | 0 |
| 光照 | ACT w/ Color Jitter | 13 | 10 | 8 |
| 光照 | **CRAFT (Ours)** | **17** | **14** | **12** |
| 背景 | ACT w/o Aug | 4 | 0 | 0 |
| 背景 | ACT w/ RoboEngine | 4 | 5 | 6 |
| 背景 | **CRAFT (Ours)** | **18** | **5** | **10** |
| 相机视角 | ACT w/o Aug | 6 | 3 | 2 |
| 相机视角 | ACT w/ Fine-Tuned VISTA | 14 | 8 | 6 |
| 相机视角 | **CRAFT (Ours)** | **19** | **18** | **18** |
| 物体颜色 | ACT w/o Aug | 2 | 0 | 1 |
| 物体颜色 | ACT w/ SAM3 | 15 | 9 | 11 |
| 物体颜色 | **CRAFT (Ours)** | **18** | **18** | **17** |
| 腕部+第三人称 | ACT w/o Aug(100 条真腕相机,参考) | 15 | 11 | 13 |
| 腕部+第三人称 | **CRAFT (Ours)** | **20** | **19** | **20** |

CRAFT 在全部五个维度都稳定领先各自的专用基线。腕部+第三人称一栏无合适公开基线:此处 ACT w/o Aug 是在目标相机配置下实采 100 条腕相机数据的参考,而 CRAFT Pose-Only 用 100 条生成数据即逼近它、无需真采腕相机数据,扩到 1000 条(Ours)后在 LR 与 SB 上做到满分 20/20。

## 四、局限性

作者在附录 I 明确列出六条:

1. 与所有基于视频生成模型的方法一样,合成视频可能含视觉伪影或时序不一致,妨碍下游策略学习。
2. 第三人称相机必须**靠近** 机器人与物体;远景会让 Canny 边缘变噪、在夹爪-物体接触处退化生成质量。
3. 虽是零样本生成,但要得到高质量结果仍**高度依赖精心的 prompt 工程与信息充分的参考图**。
4. 轨迹扩展需要**仿真器 + 物体网格** 来搭数字孪生(与 DexMimicGen 同样的假设),限制了对难以精确仿真的任务/物体的适用性。
5. 假设任务能被分解为**以物体为中心的子任务** 才能做轨迹扩展。
6. **未在可变形物体上验证**(附录提到可借 SoftMimicGen 方向扩展)。

补充:跨本体的 xArm7→Franka 重定向用的是"实验室内部、当前不公开"的 MuJoCo 重定向工具;评测规模均为每任务 20 试次的小样本,统计噪声不可忽视。

## 五、评价与展望(学术视角)

**优点。** (1)**统一性** 是最大卖点:七个此前被割裂研究的增强维度(位姿/光照/颜色/背景/视角/多相机/跨本体)被收进同一套 Canny + 参考图 + 语言的条件框架,工程上高度模块化、按需组合。(2)**Canny 作为控制信号的选择有说服力**:消融显示它在"保运动结构"与"放外观自由"间的折中确实关键(成功率翻倍),这是相对深度/骨架条件的清晰经验结论。(3)**跨本体零目标机数据反超实采上界** 是很强的结果,把"换机器人本体"从"重新采数"降级为"FK/IK 重定向 + 重画",对数据经济性意义显著。(4)不需要 AnchorDream 那类显式仿真 rollout 之外的额外世界模型,也不需要 Real2Sim2Real 的最后 Sim2Real 一步,管线相对轻。

**缺点与开放问题。** (1)**强依赖数字孪生**(仿真器 + 物体网格 + AprilTags)与 DexMimicGen 式子任务分解,这既是 Sim2Real 保真度的来源,也是适用性天花板——对可变形物体、难仿真物体、非物体中心任务基本失效。(2)下游只验证了 **ACT** 一种策略、仅 3 个任务、每任务 20 试次,是否能推广到 diffusion policy / VLA 大模型、更长程与更强接触任务尚未知;统计功效偏弱。(3)Wan2.1-Fun-Control **1.3B** 属小模型且零样本,论文自承需要精细 prompt 与参考图,鲁棒性/可复现性对使用者门槛不低。(4)与同期把视频扩散用于机器人数据合成的工作(AnchorDream、VISTA、RoboEngine、RoboSplat、SoftMimicGen 等)相比,CRAFT 的差异化主要在"统一多维 + 双臂 + Canny 控制",但缺少与这些方法在**同一数据预算/同一策略** 下的严格对照,表 III 只是各维度对各自专用基线。**可能的改进方向**:引入接触/物理一致性的显式约束或后验校验来抑制伪影;把控制信号从纯 2D Canny 升级为 Canny+稀疏深度的混合以兼顾远景;在更大 VLA 骨干与更长程任务上检验生成数据的边际收益是否随规模衰减。

## 参考

1. Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314, 2025.(骨干视频扩散 Wan2.1-Fun-Control)
2. Jiang et al. *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation.* ICRA 2026.(轨迹扩展/子任务分解的思想来源)
3. Ye et al. *AnchorDream: Repurposing Video Diffusion for Embodiment-Aware Robot Data Synthesis.* ICRA 2026.(最相关的视频扩散造数据对比工作)
4. Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).* RSS 2023.(下游策略)
5. Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation.* arXiv:2506.18088, 2025.(仿真基准与物体网格来源)
