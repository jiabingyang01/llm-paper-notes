# RoboTwin：面向双臂机器人的生成式数字孪生基准

> **论文**：*RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins*
>
> **作者**：Yao Mu、Tianxing Chen、Zanxin Chen、Shijia Peng、Zhiqian Lan、Zeyu Gao、Zhixuan Liang、Mingyu Ding、Ping Luo et al.（Yao Mu / Tianxing Chen / Zanxin Chen / Shijia Peng 共同一作,Mingyu Ding、Ping Luo 为通讯作者）
>
> **机构**：香港大学(HKU)、Agilex Robotics、上海人工智能实验室、深圳大学(SZU)、中科院自动化所(CASIA)、UNC-Chapel Hill、GDIIST、HKU-Shanghai ICRC、上海交通大学(SJTU)
>
> **发布时间**：2025 年 04 月（arXiv 2504.13059）
>
> **发表状态**：未录用（预印本；PDF 正文未标注录用信息,参考文献 [57] 将其早期版本 arXiv 2409.02920 标为 "early version"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2504.13059) | [PDF](https://arxiv.org/pdf/2504.13059)
>
> **分类标签**：`数据合成` `生成式数字孪生` `双臂操作` `Sim2Real` `LLM 代码生成`

---

## 一句话总结

RoboTwin 提出一条 "单张 RGB 图像 → 多样化 3D 数字孪生 → 空间标注 → LLM 生成无碰撞专家轨迹代码" 的全自动数据生成流水线,并配套 15 个真实/仿真对齐的双臂任务基准;实验显示,用 300 条 RoboTwin 仿真数据预训练 + 20 条真机数据微调,相比仅用 20 条真机数据,单臂任务成功率从 1.2% 提升到 72%(提升 70%+)、双臂任务从 20% 提升到 62%(提升 40%+)。

## 一、问题与动机

具身操作学习面临两大瓶颈:**多样、高质量演示数据稀缺**,以及**与真实世界对齐的评测基准缺失**。作者梳理了三类现有路线的不足:

- **人类遥操作采集**(RT-1、ALOHA、Mobile-ALOHA 等):数据可靠但极其昂贵、耗时,难以覆盖真实部署中的多样场景。
- **仿真轨迹生成器 / 自动数据放大**(MimicGen、RoboCasa 等):能从少量人类演示放大出大规模仿真专家数据,但都运行在**固定场景与预定义配置**下,依赖固定 3D 资产,交互物体的形状/外观多样性受限,难以泛化到新场景。
- **双臂基准缺位**:现有基准要么聚焦单臂(RLBench、ManiSkill2),要么是分离双臂(Peract2)或依赖 VR 遥操作的类人全身操作(HumanoidBench、BiGym),无法刻画**一体化双臂协同**(交接、协调避碰、狭小空间协作)的复杂性。

核心诉求:一条**只需一张真实 RGB 图**就能自动造出多样 3D 物体、并自动生成专家演示的可扩展流水线,加上一个**真实与仿真硬件/环境严格对齐**的双臂评测平台。

## 二、核心方法

RoboTwin 由三个模块串起:多样化数字资产生成 → 3D 资产空间标注 → LLM 专家数据生成。

### 2.1 多样化数字资产生成（Real-to-Sim）

从一张真实物体 RGB 图出发,重建可用于物理仿真的 3D 资产:

1. 用 **GPT-4V** 生成物体文字描述,再用 LLM 改写成 "视觉不同但功能一致" 的多条变体描述;
2. 用 **SDXL-Turbo** 依据这些描述生成同一物体类别的多样 2D 图像;
3. 用**图像条件 3D 生成模型**(Deemos 的 Rodin 平台)把这批 2D 图升维为完整 3D 模型,输出几何、表面法向、线框与纹理;
4. **双重质量校验**:UCLIP-I 相似度做定量评估 + GPT-4V 做定性视觉校验,低于阈值的资产自动打回重生成;
5. **物理属性赋值**:GPT-4V 对物体材质分类并赋予对应物理参数,再叠加 $\pm 5\%$ 随机扰动增强鲁棒性。

> 用大白话说:先让多模态大模型 "看图说话" 讲出这个物体长什么样、干什么用,再让它 "换着花样描述" 造出一堆同类但不同款的图,最后升成 3D 并自动挑次品重做——从而在同一类物体上凭空扩出大量形状/外观各异的可仿真实例。

### 2.2 3D 资产的空间标注框架

为让 LLM 能理解 "工具怎么用",给每个 3D 资产半自动标注**关键点**与**主轴**:

- **关键点**:①功能点(Point for Function,如锤子的敲击面),定义工具的作用原点;②接触点(Point for Contact,如抓握位),刻画人机/夹爪交互位置。
- **三主轴**:①功能轴(Function Axis,工具主作用方向)、②接近轴(Approach Axis,工具靠近目标物的方向)、③侧轴(Lateral Axis,与前两者正交,补全三维坐标系并描述旋转)。

为避免同类物体逐个重标,作者用 **Stable Diffusion 编码器提取扩散特征做特征点匹配**:给定源图像 $I_s$、目标图像 $I_t$、源点 $p_s$,在 $I_t$ 中找到扩散特征与 $p_s$ 相似度最高的像素作为对应点 $p_t$,从而把关键点跨模型迁移到同类新实例上。

> 用大白话说:标注一次 "锤子敲哪、握哪、朝哪敲",就能靠扩散模型的语义对应能力自动搬到成百上千个不同款锤子上,省掉重复人工标注。

### 2.3 LLM 专家数据生成

以标注好的数字孪生为基础,LLM 把复杂任务拆解为子任务并生成**无碰撞可执行代码**,支撑三种双臂能力:①螺旋运动插值(screw motion interpolation)实现双臂同步 + 夹爪协调;②独立单臂操作;③通过持续调整两臂间安全中间位实现动态避碰。运动生成分三阶段:空间约束推断 → LLM 把约束翻译成调用 **MPlib** 轨迹优化库的可执行代码 → 执行校验;并有**自纠错机制**把执行报错回灌给 LLM 重新生成。

每个子任务的行为生成本质是一个带约束优化问题——求解最优关节轨迹 $\theta(t)$:

$$\min_{\theta(t)} \; J(\theta(t))$$

约束条件为:

$$
\begin{cases}
\mathbf{T}_{ee} = f_{\mathrm{FK}}(\theta(t)) & \text{(运动学约束)}\\[2pt]
\mathbf{P}_{ee} = \mathbf{P}_o - d\cdot \vec{a}_o & \text{(位置对齐)}\\[2pt]
\vec{n}_{ee} = \vec{a}_o & \text{(朝向对齐)}\\[2pt]
\theta(t)\in\mathcal{C},\; \forall t\in[t_0,t_f] & \text{(避碰约束)}
\end{cases}
$$

其中 $J(\theta(t))$ 是综合能耗/耗时/运动平滑度的代价函数;$f_{\mathrm{FK}}$ 是正运动学;$\mathbf{P}_o$、$\vec{a}_o$ 为物体接触点与接近轴;$d$ 为沿接近方向的预留距离;$\mathcal{C}$ 是无碰撞配置空间。

> 用大白话说:以敲钉子为例——先由标注推出 "锤头功能点要对准钉子、沿正确方向砸下去",再让优化器在 "手臂运动学能达到 + 末端位姿对齐物体接触点和接近轴 + 全程不撞" 三重约束下,求一条又省力又平滑的关节轨迹,LLM 负责把这套约束写成能直接跑的代码。

整套专家数据生成按六步执行:场景初始化 → 任务分解 → 约束推断 → 机器人行为生成(调用预定义 API 写代码)→ 成功判定 → 迭代精修。附录给出了完整的 Prompt(任务信息、可用 API 如 `get_grasp_pose_to_grasp_object` / `get_grasp_pose_from_goal_point_and_direction` / `get_avoid_collision_pose`、函数示例)与 Blocks-Stack-Hard 的完整生成代码。

## 三、实验结果

**平台与设置**:基准含 15 个任务,物理引擎为 **ManiSkill3**(基于 SAPIEN)。真机为开源 **Cobot Magic** 平台(4 臂 + 4 个 Intel RealSense D-435 RGBD 相机,Tracer 底盘,30Hz)。每任务预采 100 条仿真 + 20 条真机数据。基线为 2D Diffusion Policy(DP)与 3D Diffusion Policy(DP3,分 XYZ 与 XYZ+RGB 两种输入),在 14 个任务上以 20/50/100 条演示、3 个随机种子评测。

### 仿真基准（D435 设置,成功率 %,节选）

| 任务 | 方法 | 20 | 50 | 100 |
|---|---|---|---|---|
| Dual Bottles Pick (Easy) | DP | 1.7 | 38.3 | **85.7** |
| Dual Bottles Pick (Easy) | DP3 (XYZ+RGB) | 36.7 | 74.7 | 75.7 |
| Empty Cup Place | DP | 0.0 | 25.0 | **87.7** |
| Block Handover | DP3 (XYZ+RGB) | 86.0 | **94.0** | 85.3 |
| Container Place | DP3 (XYZ) | 52.7 | 77.7 | **85.3** |
| Pick Apple Messy | DP3 (XYZ+RGB) | 6.0 | 31.0 | 54.0 |
| Dual Shoes Place | DP3 (XYZ) | 4.0 | 7.7 | 12.0 |

关键观察:①**DP3 少样本更强**(20 条时地板效应更小、几何先验足),但可扩展性有限,数据加到 100 条时提升甚微甚至下降;②**DP 起点极低但可扩展性好**,100 条时在若干任务反超 DP3(如 Dual Bottles Pick Easy 从 1.7% → 85.7%);③**RGB + 点云的融合收益不稳定**——在 Pick Apple Messy 等杂乱场景大幅改善,却在 Container Place 等任务反而掉点,说明当前双臂方法在 RGB 语义与点云 3D 信息融合上仍是开放难题;④**协同复杂度决定难度**:简单任务可达 85%+,而需高度协同的 Dual Shoes Place 所有方法均 < 15%,凸显现有模仿学习在双臂协同上的局限。

### 真机 Sim2Real（每任务 50 次试验,20real vs 300Sim+20Real）

单臂:

| 任务 | 20 real | 300Sim+20Real |
|---|---|---|
| Bottle Pick (Easy) | 0/50 | 42/50 |
| Bottle Pick (Hard) | 0/50 | 16/50 |
| Container Place | 0/50 | 49/50 |
| Cup Place | 1/50 | 39/50 |
| Hammer Beat | 2/50 | 37/50 |
| **平均** | **1.2%** | **72%** |

双臂:

| 任务 | 20 real | 300Sim+20Real |
|---|---|---|
| Dual Bottle Pick (Easy) | 0/50 | 31/50 |
| Dual Bottle Pick (Hard) | 0/50 | 11/50 |
| Container Place | 25/50 | 44/50 |
| Cup Place | 0/50 | 26/50 |
| Sweep Ball | 25/50 | 43/50 |
| **平均** | **20%** | **62%** |

即:仿真数据带来单臂 **+70%**、双臂 **+40%** 的绝对提升。Sim2Real 采用两阶段策略:先用 300 条 RoboTwin 仿真数据预训练 DP,再用 20 条真机数据微调(学习率 1e-4 → 5e-5),并对偏暗的真机图像做亮度校正(`cv2.convertScaleAbs, alpha=1.5`)。Figure 7 的缩放实验进一步表明:**300 条仿真 + 20 条真机** 的表现可与 **300 条纯真机数据**相当(单臂 Bottle Pick 与双臂 Cup Place 均验证),这也是选 300 条作为超参的依据。

## 四、局限性

- **依赖闭源基础模型**:资产生成重度依赖 GPT-4V、SDXL-Turbo 与 Deemos Rodin 平台,可复现性、成本与生成分布受制于外部服务。
- **专家代码生成的可靠性天花板**:仍需 LLM 的自纠错与偶尔人工介入(论文明确提到 "minimal human oversight for complex cases"),复杂长程/接触丰富任务的代码正确率未系统报告(仅 Figure 5 给出生成代码成功率柱状图,未给全表数值)。
- **双臂协同任务本身仍难**:即便有大量仿真数据,高协同任务(Dual Shoes Place、Dual Bottle Pick Hard)真机成功率仍很低(11/50 等),作者也承认现有模仿学习算法针对双臂协同不足。
- **RGB + 点云融合不稳定**:DP3(XYZ+RGB)收益因任务而异,缺乏统一的多模态 3D 表征。
- **资产物理保真靠启发式**:材质/物理参数由 GPT-4V 分类 + $\pm5\%$ 扰动赋值,未与真实物理测量对齐,接触动力学的 Sim2Real 差距未量化。
- **策略侧仅评测 DP/DP3**:未涉及大规模 VLA/ACT 等更前沿策略,基准对新一代方法的区分度有待验证。

## 五、评价与展望

**优点**:①把 "生成式 3D 资产 + 扩散特征跨实例迁移标注 + LLM 约束优化代码生成" 组合成一条端到端流水线,相比 MimicGen/RoboCasa 等**固定资产、固定场景**的数据放大方案,在物体多样性与场景自动化上更进一步,真正打通了 "一张 RGB 图 → 可训练仿真数据" 的 real-to-sim 环节;②基准强调**真实与仿真硬件/环境严格对齐**并同时提供真机数据,使 Sim2Real 结论更可信,填补了一体化双臂协同评测的空白;③开源(Cobot Magic + ManiSkill3),对社区复用价值高——RoboTwin 后续也演化出更大规模的版本(如 RoboTwin 2.0),显示该基准框架的延展性。

**与其他公开工作的关系**:数据生成思路上是 MimicGen / RoboCasa 的 "资产多样化 + 场景自动构建" 增强版;空间约束推理与关键点/主轴的用法呼应 ReKep、CoPa 等 "空间约束驱动的操作" 工作,但把约束落到了可执行代码与轨迹优化;策略与仿真侧直接站在 Diffusion Policy、DP3、ManiSkill3 的肩膀上。相较之下,RoboTwin 的独特性在于把**生成式数字孪生**作为数据引擎的第一性来源。

**开放问题与可能改进**:①用开源 3D 生成与开源多模态模型替换闭源依赖,降低复现成本并让生成分布可控;②对生成资产做**物理保真度**校准(接触参数、质量惯量与真实测量对齐),量化接触动力学的 Sim2Real gap;③把 LLM 代码生成从 "启发式 + 自纠错" 推进到带形式化约束/可验证的轨迹合成,提高长程接触丰富任务的一次通过率;④在基准上评测更强策略(VLA、ACT、3D-VLA),并研究统一的 RGB-点云 3D 表征以稳定多模态收益;⑤把 "生成资产多样性" 与 "策略泛化提升" 之间的因果关系做更细粒度的消融(哪一维多样性最有用),指导数据引擎的采样预算分配。

## 参考

1. Yao Mu et al. *RoboTwin: Dual-arm robot benchmark with generative digital twins (early version)*. arXiv:2409.02920, 2024.（本文的早期 CoRL workshop 版本）
2. Ajay Mandlekar et al. *MimicGen: A data generation system for scalable robot learning using human demonstrations*. arXiv:2310.17596, 2023.（固定场景下的仿真数据放大对照)
3. Soroush Nasiriany et al. *RoboCasa: Large-scale simulation of everyday tasks for generalist robots*. arXiv:2406.02523, 2024.（大规模仿真任务生成对照)
4. Yanjie Ze et al. *3D Diffusion Policy*. arXiv:2403.03954, 2024.（主力基线 DP3)
5. Wenlong Huang et al. *ReKep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation*. arXiv:2409.01652, 2024.（空间关键点约束推理的相关工作)
6. Stone Tao et al. *ManiSkill3: GPU parallelized robotics simulation and rendering for generalizable embodied AI*. arXiv:2410.00425, 2024.（底层物理引擎)
