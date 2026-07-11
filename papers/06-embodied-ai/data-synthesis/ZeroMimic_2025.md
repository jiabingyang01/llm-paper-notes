# ZeroMimic：从网络视频中蒸馏机器人操作技能

> **论文**：*ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos*
>
> **作者**：Junyao Shi\*, Zhuolun Zhao\*, Tianyou Wang, Ian Pedroza†, Amy Luo†, Jie Wang, Jason Ma, Dinesh Jayaraman（\* 与 † 分别表示同等贡献）
>
> **机构**：University of Pennsylvania
>
> **发布时间**：2025 年 03 月（arXiv 2503.23877）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.23877) | [PDF](https://arxiv.org/pdf/2503.23877)
>
> **分类标签**：`网络视频蒸馏` `零样本操作策略` `第一人称视频`

---

## 一句话总结

ZeroMimic 把一段纯"野外"第一人称厨房视频数据集（EpicKitchens）系统性地蒸馏成 9 个**可零样本直接部署**的图像目标条件（image goal-conditioned）技能策略：抓取阶段用 affordance 模型定位接触区域 + 现成抓取模型生成夹爪抓取，后抓取阶段用 HaMeR + COLMAP 把手腕运动重建成世界坐标系下的 3D 6D 轨迹并训练 ACT 模仿策略，真实世界 9 技能平均成功率 **71.0%**、仿真 **73.8%**，且无需任何域内机器人数据、可跨 Franka / WidowX 两种本体部署。

## 一、问题与动机

- **痛点**：当前大量机器人操作的成功依赖模仿学习，而模仿学习又依赖一种"极难获取"的示范——必须用**同一台机器人、同样的物体、同样的场景**采集演示，测试时策略才能工作。这类"逐场景采集"无法规模化，是通往通用机器人的直接绊脚石。
- **机会**：野外第一人称人类视频（web videos）本身已海量存在、涵盖丰富多样的操作场景，且一段视频可被多个机器人复用。原则上可以把这些数据"离线"蒸馏成一个可复用的机器人技能策略库，无需任何额外的机器人专属演示或探索。
- **三大挑战**（作者明确列出）：① 机器人与人类在**本体、动作空间、硬件能力**上差异巨大；② 单条网络视频往往不完整地呈现一个任务的所有细节（遮挡、物体出框、相机抖动）；③ 野外视频分布差异极大、难以处理。
- **与最接近的前作 H2R 的区别**：作者指出，在"真正野外视频上做零样本策略"这一格子里，此前唯一的工作是 H2R（同样用 EpicKitchens 训 6D 后抓取策略）。ZeroMimic 的关键改进是：引入 structure-from-motion 得到相机运动、在 3D 中重建更高质量的手腕轨迹；并把"学习到的预接触交互 affordance"与"学习到的后接触动作策略"组合成完整系统。作者将其定位在图 2 的右上角——"源视频泛化性 = 野外网络视频" × "知识迁移层级 = 零样本策略"。

## 二、核心方法

**总体设计**：把可分解为两阶段的操作技能作为对象——**抓取阶段（grasping phase）**：合适地接近并抓住目标物体；**后抓取阶段（post-grasp phase）**：稳定夹持后对物体做刚性操作（pick&place、开合平移、开合铰接、pouring、cutting、stirring）。系统限定于：静态两指夹爪机械臂 + 从任意第一人称视角观察工作区的 RGB-D 相机。核心动作抽象是：把人手和标准两指夹爪机械臂都抽象为一个 6D 位姿，从而实现"粗粒度的动作迁移"，再用现成视觉运动原语补足精细控制。

**(A) 抓取阶段：人类 affordance 引导的抓取**

- 用 **VRB**（在 EpicKitchens 上预训练的 affordance 模型）从 RGB 图像 + 自然语言任务描述（如 "open drawer"）生成一个 3D 的**意图接触点**及像素级抓取位置。
- 用 **AnyGrasp**（在机器人数据上预训练的抓取生成模型）在该接触区域附近为两指夹爪选出一个具体抓取姿态。
- 规划一段直线自由空间末端运动去执行该抓取。
- 用大白话说：affordance 模型负责回答"该去哪儿抓（人类经验）",抓取模型负责回答"用夹爪具体怎么抓（机器人经验）"——两个现成模型各取所长，回避了人手与夹爪本体差异带来的抓取姿态不可迁移问题。

**(B) 后抓取阶段：人类运动驱动的后抓取策略**

- **从视频提取人类手腕轨迹**：对 EpicKitchens（20M 帧 / 100 小时厨房日常）跑 **HaMeR**（SOTA 预训练手部追踪模型）做 3D 手部重建，得到所有关节相对规范手（canonical hand）的位姿，以及对应相机参数（平移 $t \in \mathbb{R}^3$）。相机参数来自 COLMAP structure-from-motion（由 EPIC-Fields 提供），据此把像素坐标的手部姿态转换到世界 3D 坐标。只取手腕关节，得到一段 $T$ 帧片段在世界坐标系下的 6D 手腕轨迹：

$$\{h_t = (x_t, y_t, z_t, \alpha_t, \beta_t, \gamma_t)\}_{t=1}^{T}$$

  用大白话说：把抖动、移动的第一人称视频"稳"回一个静止的世界坐标系里，人手手腕就变成了一条干净的 3D 位姿轨迹——这正是"引入 SfM 相机运动"相对 H2R 的关键增量。

- **策略学习**：人类演示高度多模态（同一观察下有多种操作方式），因此用 **ACT（action chunking transformer）** 这一生成式动作序列模型建模。输入为当前图像 $I_t$、目标图像 $I_g$、当前手腕位姿 $h_t$；输出未来手腕位姿块：

$$\{h_i\}_{i=t+1}^{t+n}, \quad n = 10$$

  模型预测的是**相对（relative）6D 手腕位姿**，块大小 $n=10$。每个技能训练一个模型,共得到 9 个技能策略。每策略训练 1000 epoch，在一张 NVIDIA RTX 3090 上约 18 小时。

- **重定向到机器人（部署）**：用一张"人类完成目标状态"的图像作为该任务所有试验的 goal image。给策略当前 RGB 观察 + 相机系下当前夹爪位姿；因测试时相机静止，把所有当前/未来手腕位姿用各帧相机外参变换到当前帧相机系，从而免去让模型预测相机参数的负担。模型在相机系中预测 6D 轨迹，再转到机器人系执行,机器人整块执行完再进入下一轮推理。

**关键设计选择（消融证实）**：动作用**相对表示**（平移 + 朝向都相对）显著优于绝对表示——作者假设是因为朝向从人手到夹爪存在分布偏移，且朝向空间在 $-\pi$ 到 $\pi$ 的不连续使绝对朝向难学。

## 三、实验结果

**评测设置**：真实世界在 UPenn 校园 3 个真实厨房 + 2 台机器人（Franka Emika Panda 7-DOF + Robotiq 两指夹爪 + Zed 2 立体相机；Trossen WidowX 250 S 6-DOF + 两指夹爪 + Intel RealSense D435）。真实评测覆盖 18 个物体类别、30 个真实场景（6 个厨房场景）；仿真在 RoboCasa 评 4 个技能策略、各 20 次随机化试验、12 种厨房风格。训练数据中不含任何评测用到的物体实例或场景。

**总体零样本成功率**

| 环境 / 本体 | 平均成功率 |
| --- | --- |
| Franka（真实世界，9 技能） | 71.9% |
| WidowX（真实世界，4 技能） | 65.0% |
| Franka（仿真 RoboCasa，4 技能） | 73.8% |
| 真实世界总体 | 71.0% |

**Franka 真实世界逐技能成功率**

| 技能 | 成功率 |
| --- | --- |
| Hinge Opening | 70.0% |
| Hinge Closing | 75.0% |
| Slide Opening | 90.0% |
| Slide Closing | 80.0% |
| Pouring | 66.7% |
| Picking | 56.7% |
| Placing | 70.0% |
| Cutting | 80.0% |
| Stirring | 66.7% |

（WidowX 上 Hinge Opening 90%、Pouring 70%，但 Stirring 仅 30%——作者归因于 WidowX 工作空间小、搅拌需大范围运动。Picking 因需先抓住物体、复杂度高于 Placing 而偏低。）

**消融一：抓取方法（Table I，Franka）**

| 抓取任务 | Ours | w/o interaction affordance | w/o grasp model |
| --- | --- | --- | --- |
| Drawer Handle | 8/10 | 0/10 | 0/10 |
| Cupboard Handle | 7/10 | 4/10 | 6/10 |

去掉 affordance（直接用 AnyGrasp 分数选最优抓取）会在无关场景区域乱抓；去掉抓取模型（把 VRB 2D 接触点抬升到 3D 后直接闭合夹爪）则因夹爪朝向错误、接触预测不精而失败。

**消融二：后抓取策略（Table II，Franka）**

| 任务 | Ours | Ours w/o SfM | VRB |
| --- | --- | --- | --- |
| Open drawer | 10/10 | 4/10 | 2/10 |
| Open cupboard | 10/10 | 6/10 | 0/10 |

"w/o SfM"即加强版 H2R（去掉相机内外参、手腕仅由像素坐标 + 手大小表示）；VRB 只给 2D 像素后接触轨迹、需随机采样深度并固定朝向。二者均显著劣于完整 ZeroMimic，证明 SfM 相机信息与"预测超越像素的完整维度"至关重要。

**对比零样本系统 ReKep（Table III，Franka）**

| 任务 | ZeroMimic | ReKep |
| --- | --- | --- |
| Open Drawer | 8/10 | 0/10 |
| Close Drawer | 6/10 | 6/10 |
| Place Pasta Bag into Drawer | 8/10 | 4/10 |
| Pour Food from Bowl into Pan | 8/10 | 0/10 |

ReKep（VLM 生成关键点约束）的失败主要来自：视觉模块生成不准确关键点 / 错误关联目标物体、VLM 因空间推理不足生成错误约束（如把抽屉外拉轴弄反、把倒食物所需倾角低估为 45°）。

**其他消融**：策略架构 ACT vs Diffusion Policy 表现相近（如 Pour water：ACT 7/10、DiffPo 9/10；Open drawer：ACT 10/10、DiffPo 8/10），全文其余实验统一用 ACT。相对/绝对动作表示在 Pour water 上：absT+absO 1/10、absT+relO 3/10、relT+absO 2/10、relT+relO 7/10。

**失败分析**：87 次真实失败中，31.1% 发生在 AnyGrasp 阶段（点云传感失败：小/反光物体、暗光）、24.1% 在 VRB 阶段（大家具接触点难预测、依赖 Grounded SAM 语言分割）、44.8% 在后抓取策略阶段（对相机-机器人相对构型敏感、人类视频动作重建本身有噪声）。

## 四、局限性

- **结构过度简化**：采用简化的"预抓取 / 后抓取"两段式，直接把人类手腕运动重定向到机器人，**未建模人机形态差异**。
- **能力边界**：不学习任何**手内操作（in-hand）、非抓握式交互（non-prehensile）、以及夹爪释放（gripper release）**；不处理需**双臂**的任务。
- **依赖上游现成模型**：整体性能被抓取生成、交互 affordance、深度感知、手部检测等现成模型的能力所限——这些模型进步会直接抬升本方法上限（同时也意味着上游误差会逐级传递，失败分析已证实约 55% 失败源自抓取 + affordance 两个上游模块）。
- **数据规模有限**：仅在约 100 小时的 EpicKitchens 上训练；作者指出扩展到 Ego-4D 等更大数据集有望得到更全面、更强的网络蒸馏技能库。
- **评测规模**：每技能/场景多为 10 次试验，样本量偏小；单帧 goal image 条件对目标态刻画有限（如 stirring 因目标图信息不足最难学）。

## 五、评价与展望

**优点**
- **定位清晰、"零域内数据"承诺兑现**：真正做到了从纯野外第一人称视频离线蒸馏出可零样本部署的策略库，且跨 Franka / WidowX 两种本体、覆盖 9 类常见技能，71.0% 的真实平均成功率在"无任何机器人演示"设定下相当可观。
- **模块化组合而非端到端硬训**：把 affordance（VRB）、抓取（AnyGrasp）、手部重建（HaMeR）、相机 SfM（COLMAP / EPIC-Fields）、模仿策略（ACT）这些成熟组件按"预接触 / 后接触"合理拼装，工程可复现性强，也自然地"搭上"了各上游领域的进步红利。
- **消融扎实**：SfM 相机信息、affordance + 抓取模型的必要性、相对动作表示的优越性都有明确对照实验支撑，且与最接近前作 H2R 做了公平的"加强版对照"。

**缺点 / 存疑**
- **上游误差累积是根本瓶颈**：整条链路是串行的（affordance → grasp → post-grasp policy），任一环节失败即整体失败，失败分析显示 55% 以上失败来自前两个模块，这是模块化范式固有的脆弱性。
- **技能被硬编码为"每技能一个模型 + 一段目标图"**：9 个独立策略、每个由语言指定技能类别、目标由单帧图给定，离"一个通用策略"仍远；技能之间无共享、无长程组合能力。
- **动作抽象过粗**：仅迁移 6D 手腕位姿 + 二值夹爪，丢弃了手内自由度与非抓握交互，天然无法覆盖灵巧操作与柔性物体。

**与其他公开工作的关系**
- 相对 **H2R**：核心增量是 SfM 相机运动 + 3D 手腕重建质量 + affordance 与后接触策略的组合，消融（Table II）量化证明了这些增量的价值。
- 相对 **VRB / Track2Act / RAM** 等 affordance / 轨迹类工作：ZeroMimic 不止停在"表示 / 奖励 / affordance"，而是直接产出**可执行的 6D 策略**。
- 相对 **ReKep**（并行工作，VLM 关键点约束）：二者都免域内训练，但 ZeroMimic 在开抽屉、倒食物等需精细空间推理的任务上大幅领先，暴露了 VLM 零样本空间推理的短板。
- 相对 **DITTO / R+X / OKAMI**：这些方法通常要求人类演示分布与测试环境足够接近、或依赖真值相机深度 / 启发式人手到夹爪映射，难以吃真正野外非结构化视频。

**开放问题与可能改进方向**
- 用更大更多样的第一人称数据（Ego-4D 量级）扩展技能覆盖与泛化。
- 将 9 个独立技能策略合并为语言 / 目标条件的**单一多技能策略**，并探索技能的长程组合。
- 建模人机形态差异（而非直接重定向手腕），并把手内操作、非抓握交互、夹爪释放纳入动作空间。
- 用不确定性 / 重试机制缓解串行模块的误差累积（例如抓取失败可回退到 affordance 重采样）。
- 用视频 + 语言之外的更丰富目标规格（多帧 / 轨迹 / 语言）替代单帧 goal image，尤其改善 stirring 这类目标态信息稀疏的技能。

## 参考

1. Bharadhwaj et al. *Zero-shot robot manipulation from passive human videos*, 2023.（H2R，最直接对照的前作）
2. Bahl et al. *Affordances from human videos as a versatile representation for robotics*, 2023.（VRB，抓取阶段 affordance 组件）
3. Fang et al. *AnyGrasp: Robust and efficient grasp perception in spatial and temporal domains*, T-RO 2023.（抓取生成组件）
4. Pavlakos et al. *Reconstructing hands in 3d with transformers*, 2023.（HaMeR，手部 3D 重建）
5. Huang et al. *ReKep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation*, 2024.（零样本对照系统）
