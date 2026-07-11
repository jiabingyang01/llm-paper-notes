# MoMaGen：面向多步双臂移动操作的软硬约束演示生成

> **论文**：*MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation*
>
> **作者**：Chengshu Li*, Mengdi Xu*, Arpit Bahety*, Hang Yin* et al.（*同等贡献）
>
> **机构**：Stanford University, The University of Texas at Austin, Amazon
>
> **发布时间**：2025 年 10 月（arXiv 2510.18316）
>
> **发表状态**：ICLR 2026（论文页眉标注 "Published as a conference paper at ICLR 2026"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.18316) | [PDF](https://arxiv.org/pdf/2510.18316)
>
> **分类标签**：`双臂移动操作` `演示数据生成` `约束优化` `可见性约束` `仿真到真实`

---

## 一句话总结

MoMaGen 把双臂移动操作的演示数据生成形式化为统一的约束优化问题——用**硬约束**（任务成功、运动学可行、无碰撞、操作前物体可见）与**软约束**（导航时物体可见、操作后回收位形）显式建模此前 X-Gen 系列方法未处理的"底座放哪里"与"相机往哪看"两个问题，仅用 1 条人类源演示即可为 4 项多步双臂移动操作任务批量生成数据，D0 随机化下平均数据生成成功率约 63%，生成数据训练的策略经 40 条真实演示微调后可实现 sim-to-real 部署（π0 达到 60% 真实成功率，纯真实数据基线为 0%）。

## 一、问题与动机

双臂移动操作（既要控制移动底座导航，又要协调两条高自由度机械臂）遥操作代价极高——论文统计单个任务的人类演示中底座运动约占 45% 的时长。已有的 X-Gen 系列自动化数据生成方法（MimicGen、SkillMimicGen、DexMimicGen、DemoGen、PhysicsGen）能用少量人类演示批量合成静态双臂桌面操作数据，但在移动操作场景下失效，原因有二：(1) 移动底座引入**可达性（reachability）**问题——直接复用源演示的底座轨迹，遇到随机化后位于工作空间之外的目标物体就会失败；(2) 车载相机是主动/可移动的，引入**可见性（visibility）**问题——朴素的运动规划或轨迹回放并不保证任务相关物体始终在视野内，视觉运动策略训练所需的图像观测可能整段缺失目标物体。

论文 Table 1 对比 MimicGen/SkillMimicGen/DexMimicGen/DemoGen/PhysicsGen 在双臂、移动、障碍物、底座随机化、主动感知维度上的支持情况：只有 MoMaGen 同时支持全部五项，并显式声明硬约束（Succ, Kin, C-Free, Temp, Vis）与软约束（Vis, Ret）。

## 二、核心方法

MoMaGen 把演示生成建模为一个带约束的轨迹优化问题（每个任务是一个 MDP，可拆成 N 个子任务，每个子任务标注目标物体 $o_i$、抓握物体、预抓取时刻 $t_{pregrasp}$、结束时刻 $t_{end}$、回收类型 $r$），核心公式为：

$$
\arg\min_{a_t \in [T]} \mathcal{L}(\cdot) \quad \text{s.t.}\quad
\begin{cases}
s_{t+1} = f(s_t, a_t), & \forall t \in [T] \\
\mathcal{G}_{\mathrm{kin}}(s_t, a_t) \le 0, & \forall t \in [T] \\
\mathcal{G}_{\mathrm{coll}}(s_t, a_t) \ge 0, & \forall t \in [T] \\
\mathcal{G}_{\mathrm{vis}}(s_t, a_t, o_{i(t)}) \le 0, & \forall t \in [T] \\
s_t \in D_{\mathrm{success}}, & \exists t \in [T]
\end{cases}
$$

**用大白话说**：$\mathcal{L}(\cdot)$ 是用户自定义的软性代价（比如更短的轨迹、更少的抖动）；四条不等式分别是系统动力学、关节可行性/无碰撞、可见性这些必须严格满足的硬约束；最后一条要求轨迹在某个时刻进入"成功状态集合"。除此之外论文还有一条物体—末端相对位姿保持约束：生成的新演示中，末端在接触时刻相对目标物体的位姿要与源演示中记录的相对位姿保持一致，这正是 MimicGen 系列"以物体为中心迁移末端轨迹"思想的形式化表达。MoMaGen 在此基础上把底座位姿 $\mathbf{T}^{base}$、相机位姿 $\mathbf{T}^{cam}$ 也纳入待求变量，而不再假定它们由源演示直接给定。

针对双臂移动操作，论文额外引入三类关键约束：**可达性作为硬约束**（不再直接沿用源演示底座轨迹，而是要求采样出的底座位姿能让所需的末端轨迹全程落在双臂工作空间内）；**操作时物体可见性为硬约束**（采样的底座+相机位姿必须保证任务相关物体在头部相机视野内，必要时联动躯干关节调整视角）；**导航时物体可见性**与**操作后回收到紧凑位形**则作为软约束优化（后者降低后续导航的碰撞风险）。

具体生成流程（Algorithm 1）：对每个子任务片段，先检查抓握物体是否仍在手中（否则提前中止）；用新物体位姿对源演示末端轨迹做物体中心变换得到新末端目标；用当前底座/相机位姿检查可见性并尝试求解末端轨迹逆运动学（IK）；若不可见或无解，则循环采样新的底座位姿与相机位姿，求解躯干+手臂 IK，并在软可见性约束下规划从当前底座到新底座的导航轨迹；到达后规划手臂到预抓取位姿的运动，用任务空间控制回放接触阶段动作；最后尝试回收到中性位形。底座采样在目标物体周围的环形区域内进行，运动规划与 IK 均由 cuRobo 完成。论文归纳了四点新颖性：(1) **全身运动**——同时求解末端、头部相机、底座三种位姿（此前工作只处理末端位姿）；(2) **可见性保证**——操作前硬约束强制可见、导航中软约束鼓励可见；(3) **工作空间扩展**——围绕目标物体采样底座位姿并规划跨房间的底座运动；(4) **高效生成**——优先用代价低的 IK 检测做快速筛选再做完整运动规划，并将躯干/手臂配置空间解耦以实现高效条件采样（类似任务-运动规划思路）。整个方法仅需采集与标注 **1 条源演示**（$N_{src}=1$）。

## 三、关键结果

实验在 OmniGibson 仿真（场景设计参考 BEHAVIOR-1K）中的 4 个多步双臂移动操作任务上进行：Pick Cup（导航到桌前取杯子）、Tidy Table（把杯子从台面搬到水槽，长距离移动）、Put Dishes Away（双臂独立把两个盘子摞到架子上）、Clean Frying Pan（双臂协调用刷子清洁平底锅，接触密集）。域随机化分三档：D0（±15cm/±15° 局部扰动）、D1（家具上任意位置任意朝向）、D2（D1 基础上额外加操作/导航障碍物）。基线是扩展了底座轨迹回放的 SkillMimicGen 与 DexMimicGen。

数据生成成功率（D0 随机化）：

| 方法 | Pick Cup | Tidy Table | Put Dishes Away | Clean Frying Pan |
|---|---|---|---|---|
| MoMaGen | 0.86 | 0.80 | 0.38 | 0.51 |
| SkillMimicGen | 1.00 | 0.69 | 0.38 | 0.40 |
| DexMimicGen | 1.00 | 0.72 | 0.38 | 0.35 |

D1/D2 下两条基线因无法自适应底座运动，成功率降为 0（论文中直接省略），MoMaGen 仍可生成数据：D1 下四任务分别为 0.60/0.64/0.34/0.20，D2 下为 0.47/0.22/0.07/0.16。

物体可见性对比（导航期间目标物体可见帧比例，D0）：MoMaGen 在四个任务上分别为 1.00/0.86/0.79/0.69，普遍高于两条基线（Tidy Table 上 0.86 对 0.40/0.39，接近翻倍）；去除软/硬可见性约束的消融实验显示可见性显著下降，验证了软硬约束设计的必要性。D1/D2 下 MoMaGen 仍能维持 75% 以上的可见性。

策略学习（用 WB-VIMA 与 π0 两种模仿学习方法，各 1000 条生成演示）：Pick Cup(D0) 上 MoMaGen 与基线打平（约 0.75）；Tidy Table(D0) 上 MoMaGen 明显超过基线（基线因过拟合源演示中冗长不平滑的底座轨迹回放，成功率仅约 0.10）；在随机化范围更大的 Pick Cup(D1)（1.3m×0.8m）上，只有 MoMaGen 数据训练出的 WB-VIMA 达到 0.25 成功率，用 D0 基线数据训练的策略完全失败（0）；π0 微调在各任务上的成功率与 WB-VIMA 相当（D1/D2 下约 0.78 ~ 0.94）。

真实机器人部署（Pick Cup 任务，40 条真实演示微调）：WB-VIMA 用 1000 条 MoMaGen 合成数据预训练+真实数据微调达到 10% 成功率，纯真实数据训练基线为 0%；π0 效果更明显，预训练+微调达到 60% 成功率，基线为 0%（基线策略虽有抓取尝试但常因精度不足脱手）。

失败模式分析：跨三档随机化，仿真不稳定性（控制器误差、随机性）贡献约 35% 的失败；手臂级运动规划失败占比最大（平均约 40%），高于底座级规划失败（平均约 26%）；D2 随机化下因地面障碍物增多，导航相关失败显著上升。计算成本上，仿真执行耗时远超规划耗时本身（例如底座运动规划平均 18 秒，对应动作在仿真中执行需约 100 秒）。

## 四、评价与展望

MoMaGen 的核心贡献是把此前分散、各自为战的自动化演示生成方法（MimicGen 系列）统一到"硬约束保证任务成功与物理可行、软约束优化数据质量"的约束优化框架下，并第一次系统性地把"移动底座放哪里"和"相机往哪看"这两个移动操作特有的问题显式建模为可达性与可见性约束，而不是简单沿用源演示的底座轨迹或依赖被动的固定视角相机。论文指出该框架可以还原出 MimicGen/SkillMimicGen/DexMimicGen/DemoGen/PhysicsGen 等已有方法（仅约束集合不同），为后续双臂移动操作的数据合成方法提供了较为通用的设计语言。

局限也比较明显：(1) 数据生成过程假设可获得仿真中的全量真值场景信息（精确物体位姿），这是仿真到真实迁移的关键瓶颈，论文承认真实场景需要额外引入 SAM2 等视觉模型做物体位姿估计，这一部分尚未在文中验证；(2) 目前只处理"导航—操作"交替的任务结构，尚未覆盖如开门这类需要全身协同的操作（whole-body manipulation），论文将其列为未来工作；(3) 数据生成成功率随随机化难度上升明显下降（Put Dishes Away 在 D2 下仅 7%），且仿真不稳定性贡献了三分之一以上的失败，说明该方法对底层运动规划器/物理仿真鲁棒性有较强依赖；(4) 依赖 cuRobo 等 GPU 加速运动规划器，单条演示生成的仿真执行耗时（百秒量级）远高于规划耗时，大规模数据合成的算力成本不容忽视；(5) 真实世界验证目前只在单一任务（Pick Cup）、单一底盘上完成，绝对成功率（10%/60%）与真实数据本身的稀缺性交织在一起，更像是"低数据区域下有效先验"的初步证据，尚不足以充分说明合成数据能替代大规模真实演示。

总体而言，MoMaGen 延续了 X-Gen 系列"少量人类源演示 + 仿真中约束满足式轨迹合成"的技术路线，其增量价值集中在把这一路线从静态桌面双臂操作扩展到了需要主动视觉与自由底座放置的移动双臂操作场景，是该数据生成范式向真实家庭环境任务（BEHAVIOR-1K 风格）迁移的一次扎实但仍处早期阶段的尝试，后续开放问题包括去除真值位姿依赖、扩展到全身操作、以及降低仿真执行的计算开销。

## 参考

- Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*, CoRL 2023.
- Garrett et al., *SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning and Deployment*, CoRL 2024.
- Jiang et al., *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning*, 2025.
- Jiang et al., *Behavior Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities*（WB-VIMA 策略来源）, 2025.
- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.
