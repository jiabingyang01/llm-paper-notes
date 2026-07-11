# RoboTwin：面向双臂机器人的生成式数字孪生基准（早期版本）

> **论文**：*RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)*
>
> **作者**：Yao Mu、Tianxing Chen（共同一作） + Shijia Peng、Zanxin Chen、Zeyu Gao、Yude Zou、Lunkai Lin、Zhiqiang Xie、Ping Luo（通讯：Ping Luo、Yao Mu）et al.
>
> **机构**：The University of Hong Kong；AgileX Robotics；Shanghai AI Laboratory；Shenzhen University；Institute of Automation, Chinese Academy of Sciences
>
> **发布时间**：2024 年 09 月（arXiv 2409.02920，本文档为 v3，2025 年 04 月更新）
>
> **发表状态**：未录用（预印本）；标题自标注为 "early version"
>
> 🔗 [arXiv](https://arxiv.org/abs/2409.02920) | [PDF](https://arxiv.org/pdf/2409.02920)
>
> **分类标签**：`数字孪生 real2sim` `双臂操作` `AIGC 造数据 + LLM 专家轨迹`

---

## 一句话总结

RoboTwin 用"单张 RGB 图 → AIGC 生成带纹理 3D 网格 → 标注功能坐标轴 → GPT-4V/GPT-4 自动写代码算抓取位姿并调用轨迹规划器"这条 real-to-sim 流水线，把真实工具零成本复刻成仿真数字孪生并**自动**生成专家演示数据；配套一个含真实遥操数据与合成数据的双臂基准（17 个任务，其中 9 个工具使用、5 个人机交互、6 个双臂），用 3D Diffusion Policy 验证时 6 个任务在 50 条演示下成功率可达 64%–98%（如 Block Handover 从 10 条的 50% 提升到 50 条的 98%）。

## 一、问题与动机

- **双臂协作 + 工具使用**是机器人走向真实工厂/医疗/家庭场景的关键能力，但这类技能高度定制化、难以标准化，在常规数据集中几乎没有覆盖，**专门化高质量训练数据极度稀缺**是核心瓶颈。
- 现有数据来源各有缺陷：
  - **真实遥操采集**（RT-1、BridgeData、RoboTurk 等）质量高但成本高、耗时长；
  - **仿真中的算法化轨迹生成器**（RLBench、ManiSkill2、VIMA 等）高效但依赖特权信息与人工设计启发式，且往往**难以产出贴近真机的高保真专家数据**；
  - MimicGen、RoboCasa 等虽能用少量人类演示合成仿真专家数据，但**仍严重依赖预定义场景与交互物体**。
- 同时，把真实场景搬进仿真做数字孪生，传统方法依赖昂贵的高精度扫描/传感器，限制了普及。
- 因此作者提出 RoboTwin，目标是：**只用一张真实 RGB 图就能把目标物体与场景搬进仿真**，并**用大模型自动生成同类任务/同类物体的专家演示代码与数据**，从而大幅减少对持续人工干预的依赖。

三点核心贡献：1）RoboTwin 基准（真实遥操数据 + 对应的高保真合成数据）；2）只需单张真实图像的便捷 real-to-sim 流水线；3）用 LLM 结合仿真环境信息自动生成"造专家数据"的代码。

## 二、核心方法

整条流水线分为两大模块：**生成式数字孪生系统（real2sim）** 与 **LLM 专家数据生成**，再加上**真机数据采集**与**基准**。

### 2.1 生成式数字孪生系统（Sec. 3.1）

- 用 AIGC 从单张 2D RGB 图重建 3D 模型，底层依赖 **Deemos 的 Rodin 平台**（text/image 到 3D 生成模型）。
- 流水线（Fig. 2）：单张真实图 →（自动）分割 + 文本描述 → 3D 生成（几何 Geometry、表面法线 Surface Normal、线框 Wireframe）→ 纹理生成 → 得到带纹理、可进物理引擎的 3D 重建物体。
- **功能坐标轴标注（Fig. 3）**：在重建物体的功能部件上赋予坐标轴。例如锤子：一条轴对齐锤头（功能部件 point for function），另一条轴指示接近方向（approach direction）；把手上标接触点（point for contact）。这一步为自动算抓取位姿服务。

把物体的几何语义抽象成一个**功能坐标系元组**：

$$\mathcal{F}_{\text{obj}} = \langle\, \mathbf{p}_{\text{func}},\ \mathbf{p}_{\text{contact}},\ \mathbf{a}_{\text{func}},\ \mathbf{a}_{\text{app}}\,\rangle$$

其中 $\mathbf{p}_{\text{func}}$ 是功能点（锤头）、$\mathbf{p}_{\text{contact}}$ 是接触/抓取点（把手）、$\mathbf{a}_{\text{func}}$ 指向功能部件、$\mathbf{a}_{\text{app}}$ 为接近方向。

论文文字描述"抓取位姿计算为：**垂直于功能部件表面法线、沿指定接近方向轴**"，可形式化为在接触点 $\mathbf{p}_{\text{contact}}$ 处、以局部表面法线 $\mathbf{n}(\mathbf{p}_{\text{contact}})$ 与接近轴 $\mathbf{a}_{\text{app}}$ 构造抓取姿态 $\mathbf{R}_{\text{grasp}}$：

$$\mathbf{R}_{\text{grasp}} = \big[\ \mathbf{a}_{\text{app}},\ \ \mathbf{n}(\mathbf{p}_{\text{contact}}) \times \mathbf{a}_{\text{app}},\ \ \mathbf{n}(\mathbf{p}_{\text{contact}})\ \big]$$

> **用大白话说**：先让 AI 把一张照片"脑补"成一个能进仿真的带纹理 3D 模型；然后像给工具贴标签一样，指出"哪头是干活的、从哪个方向去抓、抓哪里"。有了这几个方向轴和法线，抓取姿态就能被几何地算出来——夹爪贴着表面法线的垂直方向、沿着接近方向去合上，不用人一个一个去标抓点。

### 2.2 专家数据生成（Sec. 3.2）

- 借助 **GPT-4V 的推理能力写代码**，计算关键位姿与物体功能坐标轴之间的关系：GPT-4V 分析任务需求，生成与需求对齐的**一串位姿序列**，保证任务能被精确执行。
- 再用 **GPT-4 生成代码**，基于算好的位姿去**调用轨迹规划工具**，从而把编程与部署自动化，实现对任务的 zero-shot 专家数据生成。

可形式化为：给定任务指令 $\ell$ 与所涉物体的功能坐标系集合，先由 GPT-4V 生成子目标末端位姿序列，再由 GPT-4 产出的代码调用规划器 $\Pi$ 补出稠密轨迹：

$$\{\mathbf{g}_1,\dots,\mathbf{g}_K\} = \mathrm{GPT4V}\big(\ell,\ \{\mathcal{F}_{\text{obj}}^{(i)}\}\big),\qquad \tau = \Pi(\mathbf{g}_1,\dots,\mathbf{g}_K)$$

> **用大白话说**：把"任务要求 + 工具功能轴"喂给 GPT-4V，让它像写脚本一样输出"先到哪、再到哪"的关键位姿；GPT-4 再写一段调用运动规划器的代码，把这些关键点连成完整轨迹。于是"造一条专家演示"这件事变成了让大模型自动写代码跑规划，几乎不用人工示教。

### 2.3 基准与真机数据（Sec. 4–5）

- **基准（Sec. 4）**：面向双臂机器人多场景，设计 **17 个任务**（Appendix A.3），其中 **9 个强调工具使用、5 个涉及人际/人机交互、6 个为双臂任务**；每任务采集 **30 条轨迹**（采集时把任务拆成多阶段，对需要精细操作的关键子段放慢采集以提升轨迹细节）。每个任务提供**支持无限可变场景（不同物体摆放/环境条件）的专家数据生成 API**，并附**离线数据集**便于离线训练与算法对比。
- **真机平台（Sec. 5, Fig. 5）**：AgileX Robotics 的 **Cobot Magic** 平台，配 4 个 AgileX 机械臂 + 4 个 Intel Realsense D-435 RGBD 相机，装在 Tracer 底盘上。相机布局：1 个高位（大视野）、2 个腕部、1 个低位（可选）。前/左/右三路相机以 **30Hz** 同步采集，每帧含 3 路 RGB+深度图（分辨率 **640×480**），并记录主/从（master/slave）配置下左右臂的关节位姿与末端位姿。数据对齐借助 **ARIO Data Alliance** 工具，存储/格式统一遵循 ARIO 标准。

## 三、实验结果

实验目的不在于比较不同策略网络的设计，而是验证：a）Cobot Magic 平台设置的合理性；b）**自动生成的专家数据是否有效**。作者用 **3D Diffusion Policy（DP3）** 在 6 个任务上，分别用 **10 / 20 / 50 条**专家数据训练，测成功率（Table 1）：

| 任务 Task | 10 条 | 20 条 | 50 条 |
|---|---|---|---|
| Block Hammer Beat | 24% | 56% | 80% |
| Empty Cup Place | 10% | 60% | 96% |
| Dual-Bottles Pick | 10% | 42% | 74% |
| Block Sweep | 28% | 70% | 86% |
| Apple Cabinet Storage | 30% | 57% | 64% |
| Block Handover | 50% | 90% | 98% |

关键读数：

- **专家演示数量与成功率强正相关**——随着合成专家数据从 10 条增到 50 条，6 个任务成功率整体大幅提升，说明"自动生成的专家数据"确实有效。
- 提升最显著的是 **Block Handover**：10 条时 50% → 50 条时 **98%**；**Empty Cup Place** 从 10% → **96%**；**Dual-Bottles Pick** 从 10% → 74%。
- 提升最"温和"的是 **Apple Cabinet Storage**（30% → 64%），暗示涉及柜体开合的长程/接触复杂任务更难，仅靠加数据收益递减。

> 注意：表中所有数字均为**仿真环境内**的成功率；论文未给出把仿真训练策略部署到真机的成功率，也未与"等量真实遥操数据"训练做对照。

## 四、局限性

- **无 sim-to-real 量化**：全部成功率来自仿真，未报告用合成/数字孪生数据训练的策略在真机上的表现，real2sim 的落地价值缺少直接证据。
- **重建保真度无度量**：单图 AIGC 重建的几何/物理一致性（碰撞、质量、摩擦）没有任何定量评估，抓取启发式对复杂几何是否稳健未验证。
- **专家数据"产率"未报告**：GPT-4V 出位姿、GPT-4 写代码调规划器，这套 zero-shot 生成的**成功率/重试次数/失败模式**没有统计，"自动"的实际代价不清楚。
- **实验覆盖窄**：17 个任务只测了 6 个，策略只用了 DP3 一种，缺少与其他策略、与真实演示同量对照的横向比较。
- **依赖闭源组件**：real2sim 依赖 Deemos Rodin，专家数据依赖 GPT-4V/GPT-4，可复现性与长期成本受第三方 API 制约。
- **规模偏小**：真机每任务仅 30 条轨迹；抓取位姿的几何启发式偏向"工具类功能物体"，对无明确功能轴的物体适配性存疑。
- 本文明确是 "early version"，方法与实验都较初步。

## 五、评价与展望

**优点**：

- **切中痛点且工程实用**：用"单张 RGB 图 + AIGC"替代昂贵三维扫描做数字孪生，把 real2sim 的门槛压到很低；再用"功能坐标轴 + LLM 写代码调规划器"把造专家数据从人工示教变成自动化流程，这条"零示教造演示"的组合拳在造数据方向上很有启发性。
- **数据形态完整**：同时提供真实遥操数据与对应合成数据，且附带可无限扩增场景的 API 与离线数据集，作为基准的可用性较好；采用 ARIO 统一数据格式利于跨平台复用。
- 实验虽简单，但"随专家数据量单调提升"的曲线足以支撑核心论点——自动生成的专家数据能有效驱动策略学习。

**与其他公开工作的关系**：

- 与 **MimicGen**、**RoboCasa** 相比，后两者从少量人类演示在**预定义场景/资产**上做数据增广，RoboTwin 的差异点在于**从单张真实图直接重建资产**并用 **LLM 代码生成**产出专家轨迹，理论上对"新物体/新任务"的扩展性更强。
- 与 **RLBench / ManiSkill2 / VIMA** 等仿真基准相比，RoboTwin 强调**功能语义（functional axis）驱动的工具使用**与双臂协作，并强调真机对齐。
- 策略侧直接采用 **3D Diffusion Policy（DP3）** 作为验证器，与 **Diffusion Policy** 一脉相承。

**开放问题与可能的改进方向**：

- **闭 sim2real gap**：引入物理一致的三维重建（碰撞/材质/质量估计）、可微仿真与域随机化，并用真机成功率来校准数字孪生的价值；
- **量化生成流水线的可靠性**：报告专家数据一次生成成功率、规划失败率与自动过滤机制，才能支撑"可规模化"的主张；
- **扩展任务多样性与接触丰富度**：LLM 生成的关键位姿 + 规划器对接触密集/柔性物体任务是否泛化，值得进一步验证；
- 该早期版本后续被扩展为更完整的 RoboTwin 基准工作，本文可视为其思想原型（单图数字孪生 + LLM 造专家数据 + 双臂工具使用基准）。

## 参考

1. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations*, arXiv:2310.17596 (2023) — 少量人类演示合成大规模仿真专家数据的代表作，RoboTwin 的直接对比对象。
2. Nasiriany et al., *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*, arXiv:2406.02523 (2024) — 大规模日常任务仿真数据平台，与 RoboTwin 同属"造数据/基准"路线。
3. Ze et al., *3D Diffusion Policy*, arXiv:2403.03954 (2024) — 本文用于验证合成专家数据有效性的策略网络。
4. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv:2303.04137 (2023) — DP3 的思想源头，扩散式视觉运动策略。
5. Gu et al., *ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills*, arXiv:2302.04659 (2023) — 仿真操作基准代表，作为 RoboTwin 在"仿真轨迹生成器"方向上的参照。
