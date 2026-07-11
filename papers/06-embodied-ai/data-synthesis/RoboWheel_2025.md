# RoboWheel：从真实人类演示出发的跨本体机器人学习数据引擎

> **论文**：*RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning*
>
> **作者**：Yuhong Zhang、Zihan Gao（共同一作），Ling-Hao Chen、Xiao Lin、Junjia Liu 等，Haoqian Wang（通讯）et al.
>
> **机构**：清华大学、Synapath、香港中文大学（CUHK）、香港大学（HKU）、香港理工大学（PolyU）；两位一作在 Synapath Research 实习期间完成
>
> **发布时间**：2025 年 12 月（arXiv 2512.02729）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.02729) | [PDF](https://arxiv.org/pdf/2512.02729)
>
> **分类标签**：`人到机器人数据引擎` `HOI重建` `跨本体重定向` `仿真数据增强` `VLA预训练`

---

## 一句话总结

RoboWheel 是一条"单目 RGB(-D) 人手-物交互视频 → 物理合理的 HOI 重建 → 跨本体重定向 → Isaac Sim 域随机化增强 → 可训练监督"的端到端数据引擎，其重建质量（物体 Chamfer 5.1 cm、手 W-MPJPE 7.81 mm）大幅优于 HORT/HOLD/DiffHOI，并首次给出定量证据：仅用它产出的 HORA 数据（约 15 万条轨迹）预训练即可让 Pi0/RDT 等 VLA 在真机任务上媲美甚至超过遥操作数据（5k 预训练后困难任务平均成功率 58.8% vs 遥操作 40.0%）。

## 一、问题与动机

具身智能体最有效的监督来自"人如何与世界交互"，但获得机器人可用的规模化监督一直很难。现有数据采集主要有两条路，各有硬伤：

- **遥操作 / 动捕棚采集**：需要专用硬件与人工精调，成本高、绑定特定本体，动作行为的多样性和跨本体可迁移性都受限。
- **纯仿真数据生成**：便宜但无法反映真实世界的感知分布与接触分布，sim-to-real 有本质鸿沟。

与此同时，海量的人手-物交互（HOI）视频蕴含丰富的真实操作策略，却因为重建噪声、物理不合理、本体不匹配三大问题，很少被转化成机器人可训练的监督。作者主张：一条实用的处理管线必须同时满足三点要求——(i) 在真实操作空间中大规模、连续地获取"机器人可用"轨迹并强制物理合理；(ii) 支持向异构机器人本体（乃至跨域）灵活重定向且保持交互语义；(iii) 通过数据增强策略的有效组合保持可扩展性。RoboWheel 就是为了同时满足这三点而设计的。

## 二、核心方法

整条管线是 video → reconstruction → retargeting → augmentation → data 的闭环。

### 1）从 RGB(-D) 视频重建 HOI

对帧序列 $\{I_t\}_{t=1}^{T}$，恢复度量一致的手与物轨迹。时刻 $t$ 的手状态记为

$$\mathbf{h}_t = (\theta_h(t),\ \mathbf{R}_h^w(t),\ \mathbf{t}_h^w(t))$$

其中 $\theta_h(t)$ 是手（MANO）关节姿态，$\mathbf{R}_h^w, \mathbf{t}_h^w$ 是手腕在世界系下的全局旋转与平移；物体状态为刚体 6D 位姿 $\mathbf{p}_t = T_o^w(t) \in SE(3)$。

- **手/全身运动恢复**：先判定片段是"仅手"还是"全身"。仅手时用 Pavlakos 等的方法逐帧估 $\mathbf{h}_t$；全身时用 Motion-X++ 估 SMPL-H 参数，直接得到身体姿态与形状。
- **物体重建与位姿**：用分割掩码 $m_t$ 与深度 $D_t$（RGB-D 直接给，或用 UnidepthV2 预测），先用多视图 3D 生成器 Hunyuan3D 2.0 生成无尺度纹理网格，再把掩码内深度反投影成点集 $\mathcal{P}$，用点云与网格包围盒对角线之比 $s_o$ 恢复度量尺度，最后用 FoundationPose 做基于对应关系的位姿跟踪。
- **视角对齐**：用 DROID-SLAM 估相机内参 $K$ 与相机到世界变换 $T_c^w=(R_{wc},t_{wc})$，把所有重建统一到世界系，消除视角相关的不一致。

> 用大白话说：先分头把"手怎么动""物体是什么形状、怎么动"各自估出来，再靠 SLAM 把它们放进同一个世界坐标系，这样不同视频里的动作才能拼到一起、后面才谈得上重定向。

### 2）物理合理性的两段式优化

重建出来的手物轨迹通常有穿模、接触不稳、抖动等问题。作者用"SDF 优化打底 + 残差 RL 精修"两段来修：

- **SDF 阶段**：设物体的水密截断有向距离场 $\phi_o(\mathbf{x};t)$（外部为正），手顶点集 $V_h$。先优化手腕平移 $\mathbf{t}_h^w$ 使掌心顶点 $V_h^{palm}$ 的穿透 $\phi_o^2$ 最小，再联合优化手腕旋转/平移 $(\mathbf{R}_{wrist}^w, \mathbf{t}_{wrist}^w)$ 消除手物互穿，得到"无碰撞 HOI 初始化"。
- **残差 RL 阶段**：在无碰撞初始化基础上引入残差 RL 策略（沿用 ManipTrans 思路），在物理仿真里同时精修手与物的轨迹，鼓励精确跟踪并在手物距离低于阈值后促成物理一致的接触，同时保证机器人可达。RL 状态为 $s_t=(h_t, p_t, \dot h_t, \dot p_t, \mathcal{C}_t)$（含接触力 $\mathcal{C}_t$），奖励为

$$r_t = \lambda_{\text{geo}}\Phi_{\text{geo}}\big(\|\Delta h_t\| + \|\Delta p_t\|\big) + \lambda_{\text{dyn}}\Phi_{\text{dyn}}\big(\|\Delta \dot h_t\| + \|\Delta \dot p_t\|\big) + \lambda_{\text{con}}\Phi_{\text{con}}(\mathcal{C}_t)$$

其中 $\Phi$ 是奖励整形函数，$\Delta$ 是当前状态与目标状态之差，三项分别管几何跟踪、动力学平滑、接触一致。

> 用大白话说：先用"距离场"硬把手从物体里拔出来别穿模，再让一个 RL 策略在物理引擎里"抖一抖手腕"把动作修得既贴合原视频、又抓得稳、又不会让机器人够不到。

### 3）规范动作空间与跨本体重定向

- **规范动作空间 $\mathcal{A}$**：以身体关节（左右髋、肩）位置构造参考帧——$z$ 轴对齐重力/地面法向，$y$ 轴取主交互方向（平均手→物接近向量），$x$ 轴右手定则补齐，再把动作轨迹平移到首个显著帧的物体位置，得到 $T_w^{\mathcal{A}}$。这样异构来源的交互轨迹才能统一、可重定向。
- **机械臂重定向**：把 3D 手关节映射为平行夹爪的末端位姿 $\{T_g(t), g(t)\}$。用 kNN 分类区分"整手抓（palm-involved）"与"指尖抓（finger-only）"两类手势，分别用掌面法向或食指-拇指连线定义夹爪朝向（附录给出两套伪代码）；用 CoTracker 跟踪物体关键点位移判定夹爪开合。末端 6D 轨迹经 cuRobo 的 GPU 并行逆运动学映射到关节空间：

$$q_t = \arg\min_q \mathcal{C}_{\text{goal}}(T_g(t), q),\quad \text{s.t.}\ q_{\min}\preceq q \preceq q_{\max},\ \mathcal{C}_{\text{coll}}(q)\le 0$$

并以上一帧解 $q_{t-1}$ 作为 IK 种子保证时序连续。作者在 UR5/UR5e、Franka Panda、KUKA iiwa 7、Kinova Gen3、Rethink Sawyer 五款 6/7-DoF 臂上实例化。

- **灵巧手 / 人形**：灵巧手按运动学相似性与接触保持约束重定向到目标手关节空间；人形借助全身 SMPL-H，通过 IK 与动力学感知优化适配到人形关节树（论文称仅为初步验证）。

### 4）仿真中的数据增强（数据飞轮）

在 Isaac Sim 里、在规范空间 $\mathcal{A}$ 内做大量增强以丰富视觉与轨迹多样性：

- **物体检索与替换**：建大型物体库（Hunyuan3D 生成 + 自采扫描），按融合相似度检索 top-K 替代物

$$\mathcal{S}(M_o, \tilde M) = \alpha\,\text{CD}(\hat M_o, \hat{\tilde M}) + \beta\,(1-\text{IoU}_{\text{AABB}}) + \gamma\,\langle \phi_{\text{sem}}(M_o), \phi_{\text{sem}}(\tilde M)\rangle$$

（对称 Chamfer 形状 + 包围盒粗匹配 + 文本-形状语义嵌入），并对齐主轴、绑定相同最大 AABB，使原有末端运动计划与接触几何在替换物体上仍可直接回放（如杯↔带柄马克杯）。
- **轨迹增强**：把轨迹按接触状态 $c^{(k)}\in\{\text{hold, open}\}$ 切成物体中心的段落。交互段对每个路点施加物体系刚体变换 $\bar T_g(t)=T_o T_g(t)$；非交互段线性重映射平移路径（式 5）以保持连续、可 IK 执行。
- **本体变体 / 背景纹理随机 / 桌面杂乱化 / 手部镜像** 等。
- **自动回放评估与语言标注**：仿真回放可能因精度或非物理碰撞失败，用 Qwen2.5-VL 做二分类成功/失败判定，成功轨迹再用 GPT-4o 生成细粒度语言指令。

### 5）HORA 数据集

用 RoboWheel 构建大规模多模态 HOI-机器人数据集 **HORA**（Hand-Object to Robot Action），三个来源共约 15 万条轨迹：

| 子集 | 轨迹数 | 物体信息粒度 |
|---|---|---|
| HORA（Recordings，RGB-D 采集） | 23,560 | 6-DoF 物体位姿 + 资产 |
| HORA（Mocap，自建多视角动捕+触觉手套） | 63,141 | 6-DoF 位姿 + 资产 + 触觉图 |
| HORA（Public Dataset，公开 HOI 数据） | 66,924 | — |
| 合计 | 约 150,000 | 占比：Public 45% / Mocap 40% / Recording 15% |

自建动捕系统用 3 台 Intel RealSense D455 RGB-D 相机 + 8 台额外 RGB 相机（最多 11 视角），手套为 Paxini 的 EVT2（29 个磁编码器测关节 + 16 个 Gen3 触觉传感器测法向力），并以触觉引导的多约束优化拟合 MANO；公开子集取自 GRAB、HO3D v3、DexYCB、HO-Cap、TACO。

## 三、实验结果

### HOI 重建质量（在 HO-Cap 上评测，相同相机参数与网格公平对比）

| 方法 | Object CD(cm)↓ | F5(%)↑ | F10(%)↑ | Hand jitter↓ | W-MPJPE(mm)↓ | Rel. Trans(cm)↓ | Rel. Rot(deg)↓ |
|---|---|---|---|---|---|---|---|
| HORT | 8.9 | 55.0 | 83.0 | 3.35 | 19.92 | 3.54 | - |
| DiffHOI | 7.2 | 59.6 | 78.1 | 4.59 | 20.21 | 4.51 | - |
| HOLD | 7.5 | 53.2 | 77.9 | 3.47 | 20.59 | 2.44 | - |
| **RoboWheel** | **5.1** | **63.4** | **89.1** | **0.92** | **7.81** | **0.26** | **1.9** |

重建质量全面领先：物体 CD 从 7~9 cm 降到 5.1 cm，手 W-MPJPE 从约 20 mm 降到 7.81 mm，手抖动降到 0.92，手物相对位姿一致性（跨帧标准差）平移仅 0.26 cm、旋转 1.9°。

### 下游 VLA/IL 训练（8 个家用任务，按难度分组，各难度组内平均成功率 %）

评测 ACT、DP、RDT、Pi0 四种算法，三种训练方式：10 条遥操作微调（tele.）、10 条 HORA 微调（HORA）、5k HORA 预训练 + 10 条 HORA 微调（仅 RDT/Pi0 支持预训练）。

| 难度 | 训练方式 | ACT | DP | RDT | Pi0 |
|---|---|---|---|---|---|
| Easy | 10 条遥操作 | 12.5 | 30.0 | 66.3 | 68.8 |
| Easy | 10 条 HORA | 0.0 | 18.8 | 47.5 | 58.8 |
| Easy | 5k HORA 预训练 + 微调 | — | — | **75.0** | **76.3** |
| Hard | 10 条遥操作 | 0.0 | 1.3 | 35.0 | 40.0 |
| Hard | 10 条 HORA | 0.0 | 6.3 | 25.0 | 31.3 |
| Hard | 5k HORA 预训练 + 微调 | — | — | **47.5** | **58.8** |

结论：5k HORA 预训练带来的提升在困难任务上尤为明显（Pi0 从遥操作 40.0% → 预训练 58.8%）；即使只用等量（10 条）的 HORA 数据从头训练，也能达到与遥操作可比的水平，尽管存在 sim-to-real 差距。

### 增强的鲁棒性（RDT，仅 HORA vs HORA-aug，10 次/项）

| 分布偏移场景 | HORA | HORA-aug |
|---|---|---|
| Unseen Object（未见物体） | 3.75/10 | 4.25/10 |
| Clutter Objects（杂乱场景） | 4.00/10 | 4.50/10 |
| Unseen Background（未见背景） | 1.50/10 | 4.00/10 |

背景改变时无增强模型灾难性退化，而增强后未见背景成功率提升约 25%（1.50→4.00），表明增强显著提升对视觉变化的鲁棒性。

### 真机直接回放对比（重定向到二指夹爪，6 项任务宏平均成功率 %）

| 方法 | Macro avg SR(%) |
|---|---|
| **RoboWheel** | **91.7** |
| YOTO | 66.7 |
| GAT-Grasp | 50.0 |

在相同手序列输入下，RoboWheel 的夹爪位姿映射（尤其朝向的精修）在所有任务上都更稳，直接回放宏平均成功率 91.7%，远高于 YOTO（66.7%）与 GAT-Grasp（50.0%）。同一套映射跨 UR5/Gen3/iiwa7/Sawyer/Franka 五臂在 flip_milk/place_milk/pour_cola 上多为 100%（个别 70%），显示重定向的跨本体可扩展性。

## 四、局限性

- **真机跨本体验证规模有限**：真机实验仅在 UR5 二指夹爪、8 个任务、每项 10~20 次试验上开展；灵巧手/人形的重定向仅为仿真中的初步验证（人形借 ManipTrans/仿真），作者亦自陈"real-world cross-embodiment 尤其灵巧手与人形仍受限"。
- **困难任务绝对成功率偏低**：即便最优的 Pi0+5k HORA，困难任务平均也只有 58.8%，距离实用仍有差距。
- **强结论依赖预训练**：仅 10 条 HORA 从头训练时，ACT/DP 常常接近 0，"媲美遥操作"的说法主要建立在 5k 预训练之上；HORA 单独训练在多数配置下弱于遥操作。
- **重建评测面较窄**：重建质量仅在 HO-Cap 上对比三种基线。
- **流水线较重**：逐片段 RL 物理精修 + 多视图 3D 生成 + SLAM 的计算开销不小，扩展到超大规模时的成本有待评估。

## 五、评价与展望

**优点**：这是把"人手-物视频"系统性转化为"机器人可训练监督"的一次较完整的工程化尝试，端到端覆盖重建→物理合理→重定向→增强→数据，并首次给出定量证据表明 HOI 模态可作为有效的上游监督。重建数字（CD 5.1、W-MPJPE 7.81、相对位姿一致性 0.26 cm）确实明显优于 HORT/HOLD/DiffHOI；"物理合理性 = SDF 打底 + 残差 RL 精修"、"规范动作空间统一异构来源"、"整手/指尖两类手势分而治之的夹爪朝向构造"都是可复用的模块化设计；15 万条含触觉的多模态数据集有独立价值。

**与其他公开工作的关系**：与 DemoGen（从真机遥操作做轨迹增强）相比，RoboWheel 的源头是任意人类视频，覆盖面更广、更省硬件；与 Video2Policy（互联网视频驱动仿真任务）相比，它更强调物理合理的手物接触重建与跨本体重定向；相对纯仿真生成器（如 RoboTwin 系）它保留了真实感知/接触分布；相对遥操作规模采集（DROID、低成本 ALOHA）它以牺牲部分精度换取本体无关与低成本。夹爪映射上直接对标并优于 YOTO、GAT-Grasp。它与 UniVLA 的"任务中心 latent action"是缓解本体错配的两条不同路线——一条在数据侧显式重定向，一条在模型侧学习可迁移动作空间。

**开放问题与可能的改进方向**：(1) 把灵巧手/人形从"仿真初步"推进到成规模的真机验证，是这条路线能否兑现"跨本体"承诺的关键；(2) 逐片段 RL 精修的成本较高，若能用可微接触模型或 amortized 策略替代，data flywheel 才真正可规模化；(3) 下游评测任务与试验次数偏少，需要更大、更标准的 benchmark 来支撑"HOI 可媲美遥操作"的结论；(4) 触觉模态目前只在数据集中释放、尚未在下游策略中被利用，接触力监督如何反哺 VLA 是个自然的后续方向；(5) 重建质量对遮挡、手离物阶段、低分辨率视频仍敏感，稳健性有提升空间。

## 参考

1. K. Li et al. *ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning*. CVPR 2025 —— 物理合理性精修所用的残差 RL 思路来源。
2. B. Wen et al. *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*. CVPR 2024 —— 物体位姿跟踪基石。
3. Y. Zhang et al. *Motion-X++: A Large-scale Multimodal 3D Whole-body Human Motion Dataset*. 2025 —— 全身 SMPL-H 运动估计。
4. Z. Xue et al. *DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning*. 2025 —— 轨迹增强策略的直接启发。
5. H. Zhou et al. *YOTO: You Only Teach Once — Learn One-shot Bimanual Robotic Manipulation from Video Demonstrations*. 2025 —— 夹爪重定向的对比基线。
