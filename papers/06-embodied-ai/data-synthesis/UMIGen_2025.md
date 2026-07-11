# UMIGen：面向第一视角点云生成与跨本体机器人模仿学习的统一框架

> **论文**：*UMIGen: A Unified Framework for Egocentric Point Cloud Generation and Cross-Embodiment Robotic Imitation Learning*
>
> **作者**：Yan Huang, Shoujie Li（共同一作）, Xingting Li, Wenbo Ding（通讯作者）
>
> **机构**：清华大学深圳国际研究生院（Shenzhen International Graduate School, Tsinghua University）
>
> **发布时间**：2025 年 11 月（arXiv 2511.09302）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.09302) | [PDF](https://arxiv.org/pdf/2511.09302)
>
> **分类标签**：`数据合成` `第一视角点云` `手持数据采集` `跨本体模仿学习` `DemoGen`

---

## 一句话总结

UMIGen 用一个免视觉 SLAM 的手持点云采集设备 Cloud-UMI（RealSense L515 深度相机 + T265 位姿追踪 + 软体夹爪）采集第一视角点云-动作对,再用可见性感知优化（VAO，基于相机内外参做视锥筛选 + 最远点采样）把 DemoGen 的数据增强扩展到腕部第一视角场景;仿真中腕部视角策略成功率与全局视角相当（如 Close Drawer 均为 100%、Can 为 93.3% vs 90.0%）,4 种机械臂间可零样本跨本体迁移,真实世界消融显示加入 VAO 后各泛化点成功率均一致提升。

## 一、问题与动机

数据驱动的机器人操作策略高度依赖大规模、高质量示教数据,但采集成本高、依赖专用硬件、空间泛化能力有限。Universal Manipulation Interface（UMI）用手持设备降低了采集门槛,但只能记录 RGB 图像,丢失了许多任务依赖的三维几何信息；反过来,已有的第一视角三维数据采集方案（如 DROID 的固定臂标定 rig）虽有深度,却牺牲了跨平台可扩展性,需要昂贵机器人、复杂视觉 SLAM 或大量人工标定。

另一条路是数据增强：DemoGen 证明对少量源示教做空间变换即可合成大量任务变体,但它假设"全场景可见 + 固定外部视角",这一假设在腕部相机的第一视角设置下并不成立——可见区域是局部且随末端执行器姿态实时变化的。

论文因此提出两个具体问题：(1) 如何在不依赖固定机器人平台的前提下采集第一视角点云示教；(2) 如何把这类局部可见的示教扩展为任务多样的轨迹集合。UMIGen 分别用 Cloud-UMI 硬件和 VAO 增强机制回答这两个问题。

## 二、核心方法

UMIGen 由两部分组成：数据采集端 Cloud-UMI 和数据生成端 VAO（对 DemoGen 的扩展）。

**Cloud-UMI 硬件。** 基于 UMI/FastUMI 的手持设计,集成 Intel RealSense L515 深度相机与 T265 追踪相机,并采用软体夹爪模块。选择 L515 而非鱼眼相机是因为鱼眼虽视场大但难以获得高精度深度；作者进一步把 L515 位置后移,以扩大可视工作空间并缓解近距离物体的深度缺失问题。T265 沿用 FastUMI 的设计替代了 UMI 原有的视觉里程计模块,免去复杂标定,代价是长时间使用会有 IMU 漂移——论文指出这一缺陷与点云策略的高数据效率（所需示教数更少）恰好互补。软体夹爪用柔性材料贴合物体表面,缓解刚性夹爪在不规则表面上的打滑问题。

设备支持两种观测模式：一是把相机坐标系点云 $P_t^{\text{cam}}$ 经标定链转换到机器人基座系,先转到位姿估计的本地世界系

$$
P_t^{\text{pose-initial}} = T_t^{\text{pose}}\, T_{\text{pose}\leftarrow\text{cam}}\, P_t^{\text{cam}},
$$

再经离线标定的刚体变换转到机器人基座系

$$
P_t^{\text{robot}} = T_{\text{robot}\leftarrow\text{pose-initial}} \cdot P_t^{\text{pose-initial}}.
$$

用大白话说：这两步就是把"相机看到的局部点云"依次乘上"相机到追踪设备"和"追踪设备到机器人底座"两个已知刚体变换矩阵,拼成一个统一坐标系下的点云。二是直接保留相机本地坐标系下的原始点云,用于 IDP3 这类以相机系点云为条件的策略。由于相机与末端执行器刚性固连,追踪相机在机器人基座系下的位姿 $T_t^{\text{robot}} = T_{\text{robot}\leftarrow\text{pose-initial}} \cdot T_t^{\text{pose}}$ 可以直接当作末端执行器目标位姿使用,即动作标签是"顺带"得到的,不需要额外后处理或标定。

**可见性感知优化（VAO）。** DemoGen 把源示教 $\mathcal{D}_{s_0}=\{(o_t,a_t)\}_{t=0}^{L-1}$ 按物体的 SE(3) 变换 $\Delta s_0=\{(T_0^O)^{-1}T_0'^O\}_{O=1}^K$ 扩展为新构型下的示教：接触密集的技能片段随对应物体一起做刚体变换,连接片段则重新规划；动作分为随物体变换的手臂位姿 $a_t^{\text{arm}}$ 与保持不变的手部指令 $a_t^{\text{hand}}$。这一流程假设全局静态视角可见整个场景,但腕部相机存在两类遮挡：物体导致的遮挡（大物体挡住相机视线,周边区域不可见）和视角导致的遮挡（腕部相机在任务不同阶段本身视野有限,关键元素会移出画面）。

VAO 的做法是对生成流程输出的变换后点云 $\hat{P}_t$ 中每个点 $p$,用相机内参 $K$ 和当前相机位姿 $T_t^{\text{cam}}$ 做透视投影

$$
u = \Pi\!\left(K\,(T_t^{\text{cam}})^{-1} p\right),
$$

只保留投影像素落在图像边界内的点,构成可见点云

$$
\hat{P}_t^{\text{visible}} = \left\{\, p \in \hat{P}_t \;\middle|\; u\in[0,W),\ v\in[0,H) \,\right\},
$$

再用最远点采样固定点数：$\hat{P}_t^{\text{final}} = \mathrm{FPS}(\hat{P}_t^{\text{visible}}, N)$。用大白话说：给每个生成的点做一次"能不能被这一帧的腕部相机看见"的检验,看不见的点直接扔掉,再重新采样成固定大小——这样合成出来的点云才和真实腕部相机实际能拍到的东西对得上,不再是"上帝视角"式的完整场景。

## 三、关键结果

**仿真（robosuite 风格基准，5 任务 × 4 机械臂）。** 对比三种配置：Global-View DP3（G-DP3，正视全局点云转基座系）、Wrist-View DP3（W-DP3，腕部点云转基座系）、Wrist-View IDP3（W-IDP3，相机系直接输入）。示教数：Lift/Close Drawer 各 50、Stack 80、Can/Square 各 150。

| 任务 | W-DP3 | G-DP3 | W-IDP3 |
|---|---|---|---|
| Lift | 100.0% | 100.0% | 100.0% |
| Close Drawer | 100.0% | 100.0% | 93.3% |
| Can | 93.3% | 90.0% | 86.7% |
| Stack | 86.7% | 80.0% | 83.3% |
| Square | 66.7% | 73.3% | 66.7% |

结论：视野受限的腕部观测在成功率上与全局观测基本相当，且不做基座系转换、直接用相机系点云的 IDP3 也能维持有竞争力的表现，说明显式全局坐标变换并非高效视觉运动策略学习的必要条件。作者也指出点云策略对离群点（反光面、杂乱场景产生的伪深度点）敏感，IDP3 实现中加入了按距离阈值过滤深度点的步骤来缓解。

**跨本体零样本迁移。** 仅用 Panda 上采集的腕部观测训练 DP3，直接部署到 UR5e、Kinova3、IIWA（30 次试验计数）：

| 任务 | Panda | UR5e | Kinova3 | IIWA |
|---|---|---|---|---|
| Lift | 30/30 | 21/30 | 28/30 | 30/30 |
| Close Drawer | 30/30 | 24/30 | 24/30 | 29/30 |
| Can | 27/30 | 17/30 | 25/30 | 24/30 |
| Stack | 25/30 | 16/30 | 14/30 | 22/30 |
| Square | 21/30 | 13/30 | 17/30 | 19/30 |

不同本体间成功率有明显差异（如 Stack 从 Panda 的 25/30 降到 Kinova3 的 14/30），但策略在所有本体上均保持非零、多数任务过半的成功率，验证了共享腕部视角带来的跨本体可迁移性。

**真实世界（UR5 单臂平台 + Cloud-UMI）。** 4 个任务：Kiwi（单物体单阶段、无遮挡）、Open-Drawer（单物体单阶段、有遮挡）、Mug-Rack、Pick-Place（双物体多阶段、有遮挡）。每个评估位置引入 $(\pm1.5\text{cm})\times(\pm1.5\text{cm})$ 随机扰动生成 9 条示教，总生成示教数 = 源示教数 × 评估点数 × 9（如 Kiwi：3 源示教 × 8 评估点 × 9 = 216 条合成示教，对应 Fig.1 中"约 100 倍生成"的宣传口径）。真实实验结果以成功率热力图（分档 0%/>0%/>40%/>60%/>80%）呈现：Kiwi、Open-Drawer 在多数泛化点成功率超过 80%；多阶段/遮挡密集的 Mug-Rack、Pick-Place 也维持了较稳健的表现。关键消融是"有无 VAO"对比：加入 VAO 后所有泛化点的成功率相对无 VAO 均有提升，验证了视锥筛选对生成数据真实性的作用。作者还观察到工作空间中心的成功率高于边缘，归因于（1）距源示教欧式距离越远、观测差异越大，（2）机械臂运动学极限在边缘区域更容易导致执行失败。

## 四、评价与展望

**优点。** UMIGen 精准补上了"UMI 系手持采集丢失几何信息"与"固定臂 3D 采集牺牲可扩展性"之间的空白：Cloud-UMI 用消费级深度+追踪传感器免 SLAM 采集点云-动作对,动作标签因相机与末端刚性固连而"顺带"获得,省去了单独的动作标定环节；VAO 把视锥剔除这一简单几何操作嫁接到 DemoGen 上，直接针对性地解决了"合成数据默认全局可见"与"真实腕部相机只见局部"之间的分布差距，且真实世界的有/无 VAO 消融给出了一致方向的因果证据。跨本体实验用 4 种不同运动学结构的机械臂做零样本部署，是该类工作中较为完整的验证。

**不足与开放问题。** 其一，任务规模仍偏简单：仿真 5 任务和真实 4 任务均为单臂、桌面级、短时序操作，未覆盖双臂或更长时序任务，方法的可扩展性有待验证。其二，真实世界结果仅以成功率分档热力图呈现，未给出具体试验次数与精确数值，也没有与原版 DemoGen（全场景假设）或 UMI/FastUMI（纯 RGB）在同一真实任务上的直接对比，VAO 的收益证据局限于内部消融。其三，方法依赖深度相机与追踪相机之间、以及相机与末端执行器之间的精确标定，作者自陈标定偏差会引入局部噪声并限制快速/动态动作；选用 L515 而非鱼眼相机也是精度换视场的取舍，限制了大工作空间任务的覆盖率。其四，点云的离群点问题（反光、遮挡边缘的伪深度）目前靠手工设定的距离阈值过滤，是一个启发式而非系统性的解法。其五，论文中"跨本体"的成立前提是不同机械臂共享相近的腕部视角（Fig.1 标注为"sharing the same wrist viewpoint"），并非完全与本体无关的泛化，这是该框架跨本体主张的实际边界。

与同类公开工作相比，UMIGen 处在 MimicGen 系列（依赖真实机器人执行来生成数据、成本高）与 DemoGen（全合成但假设全局静态视角）之间的位置，把 DemoGen 的合成效率与 UMI/FastUMI 式的低成本采集结合，并用可见性约束弥合两者的分布差距。后续可能的方向包括：把 VAO 简单的"视锥剔除+丢弃不可见点"替换或补充为生成式点云补全（如扩散式补全遮挡区域），从而在保真度和数据多样性之间取得更好平衡；考虑末端执行器/夹爪自身造成的自遮挡而不仅是外部物体遮挡；以及在可形变或铰接物体场景下，验证 DemoGen 逐物体刚体变换假设在第一视角局部观测下是否仍然成立。

## 参考

- Xue et al. *DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning*, arXiv:2502.16932, 2025.
- Chi et al. *Universal Manipulation Interface: In-the-Wild Robot Teaching without In-the-Wild Robots*, arXiv:2402.10329, 2024.
- Wu et al. *FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface*, arXiv:2409.xxxxx, 2024.
- Ze et al. *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations*, arXiv:2403.03954, 2024.
- Mandlekar et al. *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*, arXiv:2310.17596, 2023.
