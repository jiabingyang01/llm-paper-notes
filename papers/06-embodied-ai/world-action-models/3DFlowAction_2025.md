# 3DFlowAction：基于3D流世界模型的跨具身操作学习

> **论文**：*3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model*
>
> **作者**：Hongyan Zhi, Peihao Chen, Siyuan Zhou, Yubo Dong, Quanxi Wu, Lei Han, Mingkui Tan et al.
>
> **机构**：华南理工大学（South China University of Technology）、腾讯 Robotics X（Tencent Robotics X）、香港科技大学（Hong Kong University of Science and Technology）、琶洲实验室（Pazhou Laboratory）
>
> **发布时间**：2025 年 06 月（arXiv 2506.06199）
>
> **发表状态**：未录用（预印本，Preprint. Under review.）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.06199) | [PDF](https://arxiv.org/pdf/2506.06199)
>
> **分类标签**：`3D光流` `世界模型` `跨具身操作` `闭环规划` `视频扩散模型`

---

## 一句话总结

用视频扩散模型在自建 110k 规模的 3D 光流数据集 ManiFlow-110k 上预训练一个以物体为中心、跨具身的 3D flow world model，再结合 GPT-4o 闭环校验的渲染检验机制、任务感知抓取生成与基于关键点距离的约束优化，把预测的 3D 光流转成末端执行器动作序列，在四项复杂操作任务上取得 70% 总成功率，显著超过 AVDC（20%）、Rekep（20%）、Im2Flow2Act*（25%）等世界模型基线以及 PI0（50%）等模仿学习基线，并展现出跨物体、跨背景、跨机械臂平台的强泛化能力。

## 一、问题与动机

操作任务的一个关键瓶颈是缺乏统一、大规模的跨具身动作数据集：不同机器人数据集记录动作的空间（关节角 vs. 末端位姿，不同 base 坐标系）不统一，难以学到统一且鲁棒的动作表征。作者观察到，人类理解操作任务时依赖"物体在 3D 空间中应该如何运动"这一线索，这个线索是**embodiment-agnostic** 的，对人类和不同机器人都适用。

已有的 video world model 方法（如 UniPi、AVDC）通常不是 object-centric 的，需要一并建模无关的背景内容，泛化性差；而且预测被局限在 2D 图像平面上，难以准确表示物体在 3D 空间中的运动，尤其是旋转以及垂直于相机方向的位移。已有的 2D 光流方法（Im2Flow2Act、General-Flow 等）同样受限于 2D 表示，无法完整刻画 3D 运动，且训练数据规模有限导致泛化能力不足。

基于此，本文的核心思路是：从人类和机器人操作视频中学习一个 **3D flow world model**，预测被操作物体的未来 3D 光流轨迹，作为动作策略的运动线索；再用 flow-guided rendering 机制配合 GPT-4o 做闭环校验；最终把预测的 3D 光流当作约束，通过一个不需要机器人动作标签的优化 policy 求解出具体动作序列。

## 二、核心方法

**1. ManiFlow-110k 数据集构建（moving object detection pipeline）**

现有检测模型在杂乱背景、存在相似干扰物的操作视频中表现欠佳，为此作者设计了一条自动化的运动物体检测流水线：先用 Grounding-SAM2 在视频首帧分割出机械爪（gripper）掩码，在整幅图像上采样点并排除落在爪掩码内的点；再用 2D 跟踪模型 Co-tracker3 跟踪这些点的运动，选出运动幅度显著的点，取其最大外接框作为被操作物体的位置；随后再次用 Co-tracker3 提取该物体的 2D 光流并去除相机自身运动的影响；最后用 DepthAnythingV2 做深度估计，把 2D 光流投影到 3D，得到最终的 3D 光流标签。该 pipeline 在 BridgeV2 数据集上验证的运动物体检测准确率超过 80%。汇总 BridgeV2（27%）、RH20T-Human（27%）、RT-1（18%）、AgiWorld（8%）、DROID（13%）、Libero（4%）、HOI4D（3%）等多个开源人类/机器人操作数据集，构建出包含 110k 条 3D 光流实例的 **ManiFlow-110k**。

**2. Flow World Model（基于视频扩散）**

流生成器 $G$ 遵循 Im2Flow2Act 的整体框架，以 AnimateDiff（SD v1.5 主干 + Motion Module）为骨干。输入为初始 RGB 观测、语言任务指令（经 CLIP 编码）以及初始点集 $\mathcal{F}_0$（正弦位置编码）；输出时变 3D 光流 $\mathcal{F} \in \mathbb{R}^{T\times H\times W\times 4}$，前两通道为图像空间 2D 坐标，第三通道为深度，第四通道为可见性。与 Im2Flow2Act 不同的是，作者发现 Stable Diffusion 的 image VAE 即便微调后也难以有效编码深度信息，因此该模型绕过 VAE，直接把 3D 光流送入 U-Net；Motion Module 从头训练，而 SD 主干只插入 LoRA 层，以保留预训练获得的生成能力。

**3. Flow-guided 动作生成（三个组件）**

(a) **闭环 3D 光流生成的渲染检验机制**：设首末两个时间步的光流点集分别为 $P_1$、$P_2$，用 SVD 估计二者间的刚体变换

$$T = \mathrm{SVD}(P_2, P_1)$$

用大白话说：把光流首末两帧对应点集对齐，就能反解出物体从初始位置到预测终态经历的一个刚体旋转+平移。把 $T$ 作用到物体初始点云上得到预测的目标状态，与当前场景点云一起重投影为 2D 图像，连同任务指令一并输入 GPT-4o 判断预测是否与指令语义一致；若不一致则触发重新预测，从而形成闭环。

(b) **任务感知抓取位姿生成**：先用 GPT-4o 判断该抓取物体的哪个部位，再用 AnyGrasp 在该部位周围生成一批候选抓取位姿（AnyGrasp 本身是 task-unaware 的，随机选择可能导致不可达）；利用同一个变换 $T$ 把候选抓取位姿映射到预测的目标位置，再结合机械臂逆运动学（IK）排除不可达的抓取目标，从而选定任务相关且可达的抓取位姿。

(c) **基于流约束的优化动作生成**：用最远点采样（farthest point sampling）在物体表面选取 $N$ 个关键点及其对应的 3D 光流轨迹，通过一个约束优化程序（而非如 Rekep 那样用 GPT 生成 code-based 约束，作者指出后者难以描述旋转等复杂 3D 轨迹）在每个时间步 $t$ 最小化初始关键点与预测流关键点之间的欧氏距离：

$$f^{(t)}(k_{\text{initial}}) = \min \sum_{i=1}^{N} \left\| k_{\text{initial}}^{i} - k_{\text{pred}}^{i}(t) \right\|_2^2$$

用大白话说：把"物体沿预测的 3D 光流轨迹运动"这件事，转成"让抓在物体上的一组关键点尽量贴着预测轨迹走"的最小二乘问题；再叠加 IK 与碰撞检测约束，最终解出末端执行器在 SE(3) 中的一串位姿序列，即为下发执行的动作。整个过程不需要任何机器人动作标签或遥操作数据——对一个新的下游任务，只需采集 10~30 段人类手部演示（每任务约 10 分钟）即可微调该 flow world model。

## 三、关键结果

实验平台为 Dobot Xtrainer，感知用一台 Femto Bolt 相机提供第三人称视角；四项基础任务为：倒茶入杯（保持壶身水平并对准杯口）、插笔入笔筒（需竖直插入并完成复杂旋转）、挂杯上架（需对齐杯柄与挂钩）、开抽屉（避免拖拽卡死），每个设置 10 次试验、随机化物体位姿。

**表 1：与其他世界模型基线对比**（AVDC 用 2D 光流提取器+视频扩散；Rekep 用 LVLM+GPT-4o 生成 code 约束；Im2Flow2Act* 为把原版可学习动作策略替换成优化程序的 2D 光流基线）

| 任务 | AVDC | Rekep | Im2Flow2Act* | 3DFlowAction |
|---|---|---|---|---|
| 倒茶入杯 | 1/10 | 2/10 | 2/10 | 6/10 |
| 插笔入笔筒 | 2/10 | 1/10 | 2/10 | 7/10 |
| 挂杯上架 | 0/10 | 3/10 | 0/10 | 5/10 |
| 开抽屉 | 5/10 | 2/10 | 6/10 | 10/10 |
| **总成功率** | 20.0% | 20.0% | 25.0% | **70.0%** |

**表 2：跨具身实验**（Franka 与 XTrainer 两个平台，均无硬件专属微调）

| 任务 | Franka | XTrainer |
|---|---|---|
| 倒茶入杯 | 7/10 | 6/10 |
| 插笔入笔筒 | 7/10 | 7/10 |
| 挂杯上架 | 4/10 | 5/10 |
| 开抽屉 | 9/10 | 10/10 |
| **总成功率** | 67.5% | 70.0% |

**表 3：与模仿学习方法对比**（PI0、Im2Flow2Act 均用同样 30 条遥操演示微调）

| 任务 | PI0 | Im2Flow2Act | 3DFlowAction |
|---|---|---|---|
| 倒茶入杯 | 5/10 | 4/10 | 6/10 |
| 插笔入笔筒 | 5/10 | 2/10 | 7/10 |
| 挂杯上架 | 4/10 | 5/10 | 5/10 |
| 开抽屉 | 6/10 | 5/10 | 10/10 |
| **总成功率** | 50.0% | 27.5% | **70.0%** |

**表 4：域外物体 / 域外背景零样本泛化**

| 任务 | AVDC(物体) | PI0(物体) | 3DFlowAction(物体) | AVDC(背景) | PI0(背景) | 3DFlowAction(背景) |
|---|---|---|---|---|---|---|
| 倒茶入杯 | 0/10 | 3/10 | 4/10 | 0/10 | 4/10 | 4/10 |
| 插笔入笔筒 | 2/10 | 6/10 | 6/10 | 0/10 | 1/10 | 4/10 |
| 挂杯上架 | 0/10 | 2/10 | 4/10 | 0/10 | 3/10 | 4/10 |
| 开抽屉 | 4/10 | 5/10 | 8/10 | 0/10 | 5/10 | 8/10 |
| **总成功率** | 15.0% | 40.0% | **55.0%** | 0.0% | 32.5% | **50.0%** |

**表 5：消融——闭环渲染检验机制 与 大规模预训练**

| 变体 | 大规模预训练 | 渲染检验机制 | 总成功率 |
|---|---|---|---|
| Variant1（去闭环） | ✓ | — | 50.0% |
| Variant2（去预训练） | — | ✓ | 30.0% |
| 3DFlowAction（完整） | ✓ | ✓ | **70.0%** |

去掉闭环渲染检验机制，平均成功率下降 20 个百分点；去掉在 ManiFlow-110k 上的大规模预训练，平均成功率下降 40 个百分点——说明大规模 3D 流预训练是性能的主要来源，闭环校验是次要但仍显著的增益项。

论文自陈的局限：3D 光流对柔性/可变形物体（因严重遮挡和复杂形变）难以建模，非刚体形变会导致下游约束优化无法输出有效动作，因此方法目前局限于刚体/半刚体类操作任务。

## 四、评价与展望

**优点**：把光流表示从 2D 升级到 3D，并做成 object-centric、embodiment-agnostic 的中间表征，理论上比 2D 光流方法（Im2Flow2Act、General-Flow）能表达更完整的旋转与深度方向运动，也比端到端 video world model（AVDC、UniPi、Uni-Sim 一类方法）更省算力、更能抗背景噪声干扰。闭环设计（SVD 估计刚体变换 + GPT-4o 语义校验）把一次性开环流预测转成可自我纠错的规划过程，是一个不需要额外训练验证器的轻量级 test-time verification 思路。下游动作生成完全不依赖机器人遥操作数据或动作标签，仅需 10~30 条人类手部演示视频即可微调，数据获取成本远低于遥操作采集，这也是其跨具身泛化（Franka/XTrainer 无专属训练分别达到 67.5%/70.0%）的关键原因。

**局限与开放问题**：（1）评测规模有限——4 个任务、每设置仅 10 次试验、平台数量少，样本量偏小、未报告置信区间，后续需要在更大规模基准（如 RoboTwin、CALVIN 一类标准化评测集）上验证结论是否稳健；（2）ManiFlow-110k 的 3D 光流标签本质是"检测（Grounding-SAM2）+ 跟踪（Co-tracker3）+ 单目深度估计（DepthAnythingV2）"拼接出的伪标签管线，深度估计误差与遮挡下的跟踪漂移都会传导进 flow world model 的训练信号，论文未报告伪标签本身的 3D 误差量级；（3）与 Rekep 等 code-based 约束方法相比，本文用连续光流约束替代离散关键点距离约束，回避了"prompt engineering 与约束生成不准确"的问题，但抓取阶段仍依赖 GPT-4o（部位选择）与 AnyGrasp（候选位姿）两个外部现成模块的准确性，构成误差传播链上的薄弱环节；（4）该工作与同期的 TesserAct（4D world model，联合预测 RGB/depth/normal）、Gen2Act/FLIP（flow-guided video generation）代表了同一方向下不同的建模粒度权衡——用稀疏光流而非稠密像素/体素预测未来状态，换取更好的物体中心性与跨具身泛化，但也牺牲了对场景整体动态（如障碍物、非目标物体交互）的建模能力；如何在光流约束的轻量性与场景级约束的完整性之间取得更好折中，是值得后续工作探索的开放问题。

## 参考

- Xu et al. *Flow as the cross-domain manipulation interface* (Im2Flow2Act). arXiv:2407.15208, 2024.
- Ko et al. *Learning to act from actionless videos through dense correspondences* (AVDC). arXiv:2310.08576, 2023.
- Huang et al. *ReKep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation*. arXiv:2409.01652, 2024.
- Black et al. *π0: A vision-language-action flow model for general robot control*. arXiv:2410.24164, 2024.
- Yuan et al. *General Flow as foundation affordance for scalable robot learning*. arXiv:2401.11439, 2024.
