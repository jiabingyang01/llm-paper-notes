# DexCanvas：连接人类演示与机器人学习的灵巧操作数据集

> **论文**：*DexCanvas: Bridging Human Demonstrations and Robot Learning for Dexterous Manipulation*
>
> **作者**：Xinyue Xu\*, Jieqiang Sun\*, Jing (Daisy) Dai\*, Siyuan Chen, Ke Sun, Yiwen Lu† et al.（\* 共同一作，† 通讯作者）
>
> **机构**：DexRobot Co. Ltd.；University of Michigan；Shanghai Jiao Tong University；Chongqing University；East China University of Science and Technology
>
> **发布时间**：2025 年 10 月（arXiv 2510.15786，v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.15786) | [PDF](https://arxiv.org/pdf/2510.15786)
>
> **分类标签**：`灵巧手数据集` `real-to-sim` `接触力标注` `MANO` `强化学习` `data-synthesis`

---

## 一句话总结

DexCanvas 用 70 小时真人 mocap 演示做种子，通过「一策略一物体-操作对」的 PPO 残差控制在 IsaacGym 里复现物体运动，从而由物理引擎**反解出**每帧接触点与接触力，再通过扰动物体尺寸/位姿/材质把数据放大 100 倍，最终得到 7,000 小时、覆盖 Cutkosky 分类学 21 类操作、带**物理一致连续力标注**的混合真实-合成灵巧操作数据集（约 30 亿帧）。

## 一、问题与动机

灵巧操作（高自由度拟人手）需要大规模、覆盖多样策略且带**物理精确接触动力学与力信息**的数据，但现有数据源各有硬伤：

- **遥操作机器人数据**：直接在目标硬件上采集，但带宽低、成本高，尤其高自由度灵巧手会给操作者带来沉重认知负荷，常产生不自然的手指运动。
- **人类视频/自我中心数据集**（如 EgoDex 829 小时、OpenEgo）：规模大但缺乏系统的技能覆盖与质量保证，3D 位姿估计有自遮挡和噪声，且**无法测量产生物体运动的力**。
- **mocap 数据集**（GRAB、ARCTIC）：毫米级精度但只捕捉几何运动，缺失对接触操作至关重要的力信号。
- **合成抓取数据集**（DexGraspNet 1.32M / 2.0 的 4.27 亿 / Dex1B 的十亿级）：规模无限但常产生**生物力学上不合理**的抓取，且绑定特定机器人形态。
- **接触/力数据集**（ContactDB/ContactPose 热成像、PressureVision）：只能给出二值接触掩码或近似压力，无法提供连续力幅值。

作者指出核心矛盾：**从观测反解力本质上是欠约束的**——几何上相同的接触可能对应差异巨大的力（轻握 vs. 用力握），只靠视觉/几何无法区分。要么用侵入式力传感器（会改变被测接触力学），要么引入物理仿真来强制运动与力的一致性。DexCanvas 选择后者。

## 二、核心方法

整体是一条 **real-to-sim** 流水线：真人 mocap → MANO 拟合 → RL 追踪复现 → 物理引擎读出力 → 扰动增广。

### 2.1 采集与分类体系

- **分类学**：从 Cutkosky 抓取分类学（Feix et al. 2016）系统派生出 **21 类基本操作**，分 4 大类：Power（全手稳定抓）、Intermediate（力量与精度过渡）、Precision（指尖精细控制）、In-hand Manipulation（滚动/滑动/手指换位/旋转）；再按拇指位置（外展/内收）和参与手指数量细分。
- **物体集**：30 个物体，含多尺寸几何原语、加重复制品（标 "H" 用于探测力调制）以及 YCB 物体（电钻、水壶等）。
- **采集**：5 名操作者对每个可行的「操作-物体」对做 50 次重复，共 **12,000 条序列 / 70 小时**（剔除掉落或严重遮挡的试次）。
- **硬件**：22 个红外相机做毫米级光学 mocap + 2 个同步 RGB-D 相机（30 Hz）；右手贴 14 个反光标记，每个物体贴 4 个。关键做法：物体用 CAD 模型 3D 打印、标记安装位置直接刻进几何，使追踪位姿与仿真用的 URDF 精确对齐。

### 2.2 MANO 拟合

先用 HaMeR 回归器估计每个操作者的 MANO 形状参数 $\beta \in \mathbb{R}^{10}$ 并固定，再逐帧优化全局手腕变换 $T_t$ 和姿态 $\theta_t$，最小化标记点到 MANO 表面的距离：

$$
\min_{T_t,\, \theta_t} \sum_{i=1}^{N} \left\| x_{i,t} - p_i\!\left(T_t, \hat{\beta}, \theta_t\right) \right\|_2^2 + \lambda_{\text{pose}}\, \psi(\theta_t)
$$

其中 $p_i(\cdot)$ 是第 $i$ 个虚拟标记在 MANO 表面上的 3D 位置，$\psi$ 是软关节限位先验。

用大白话说：把一只可变形的虚拟手往真实贴的反光点上「套」，套得越贴合越好，同时别把关节掰到不合理的角度。得到的是 45 维手指关节 + 手腕位姿的干净手部轨迹。mocap 世界坐标系通过固定变换 $T_{MW}$ 对齐到 MANO 世界坐标系：$T_H^M(t) = T_{MW}\, T_H^W(t)$。

### 2.3 用 RL 反解接触力（核心创新）

对每个「物体-操作」对**单独训一个策略** $\pi_\theta$，建成 MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma \rangle$，目标是最大化折扣回报：

$$
\max_{\theta} J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^{t} R(s_t, a_t)\right]
$$

关键设计——策略是**残差控制器**：动作是加在已拟合 MANO 关节角上的**小修正量**（不是绝对指令），用来补偿追踪误差、维持稳定接触。动作空间为手腕平移(3) + 手腕旋转(3) + 手指关节($n_f$)，共 $6+n_f$ 维，且经指数加权累积平滑：

$$
u_t = \tau \cdot u_{t-1} + a_t, \qquad \tau \in [0.8, 0.95]
$$

奖励为「追踪物体轨迹」与「贴近人类手势」的双目标：

$$
r_t = r_{\text{dist}} + r_{\text{rot}} - c_{\text{act}}
$$

其中 $r_{\text{dist}} = \exp(-60 \cdot \text{goal\_dist})$、$r_{\text{rot}} = \exp(-10 \cdot \text{rot\_error})$、$c_{\text{act}} = \|a_t\|_2^2$。观测里包含**未来物体位姿**（非因果的特权信息，只在训练/复现阶段用，不需真机部署）。

用大白话说：让仿真里的手「照着录像重演」——手指在人类演示姿态附近微调，把物体推到 mocap 里记录的下一个位置。**力不是网络预测出来的，而是重演成功后由物理引擎直接测量出来的**（每帧接触点、力向量、关节力矩、6 维物体 wrench）。这就把「从观测反解力」这个欠约束反问题，转化成了由仿真器强制物理一致性的「追踪控制」正问题。

### 2.4 100 倍数据增广

策略训好后，通过扰动物体尺寸、初始位姿、材质属性、MANO 形状参数来 rollout，从单条演示种子生成多样但物理有效的变体，把 70 小时 mocap 扩成 **7,000 小时**（100×）。所有模态时间对齐：RGB-D 30 Hz，mocap/仿真 120 Hz（4× 时间分辨率）。数据字段含 MANO 手运动学（wrist [T,3]+[T,3]、finger_pose [T,45]）、6-DoF 物体位姿、接触信息（contact_points [T,N,3]、force_vectors [T,N,3]、object_wrench [T,6]）。

## 三、实验结果

论文本身**不含下游机器人任务评测**（作者明说 cross-dataset applicability 和 downstream task evaluation 留待未来版本），实验只验证「流水线本身是否有效」与「力标注质量」。

**与代表性人-物交互数据集对比（Table 1，选取关键行）**：

| 数据集 | 力/接触 | 操作范围 | 物理有效 | 规模(帧) | 模态 | 标注来源 |
|---|---|---|---|---|---|---|
| EgoDex (2025) | ✗ | General | ✗ | 90M | RGB+3D | Vision Pro |
| OpenEgo (2025) | ✗ | General | ✗ | 119.6M | Multimodal | 6 数据集融合 |
| GRAB (2020) | ✗ | Grasp | ✗ | 1.6M | Mocap | Markers |
| ContactPose (2020) | Binary | Grasp | ✗ | 2.9M | Thermal | 热成像 |
| **DexCanvas (本文)** | **Continuous** | **General** | **✓** | **3.0B** | **RGB-D+Mocap** | **Simulation** |

DexCanvas 是表中**唯一**同时具备「连续力标注 + 物理有效」的数据集。

**流水线有效性（32 个代表性物体-操作对，每对仿真 rollout 100 次，成功=不触发终止即完成演示轨迹，位置误差超 5cm 判失败）**：

| 条件 | 成功率 |
|---|---|
| 名义条件（nominal） | **80.15%** |
| 初始物体位姿扰动至物体尺寸的 20% | **62.54%** |
| 扰动带来的下降 | 仅 17.61 个百分点 |

作者论点：在大幅扰动下仍只中等程度下降，说明每个策略能从**单条演示种子**生成多样且物理有效的训练数据，支撑了 100× 扩展。

**力标注质量（Figure 6/7，定性）**：逐指接触力随时间平滑、物理一致（幅值可达约 55 N 量级）；不同操作类型呈现区分明显的力「签名」——power 抓握多关节均匀受力，precision 操作则将力集中在特定指尖，与 21 类分类学吻合。

## 四、局限性

- **无下游验证**：论文未展示任何真机/仿真机器人策略学习或跨数据集迁移结果，「数据集有用」目前只是间接论证（流水线复现成功率 + 力可视化），种子只有 70 小时真机演示。
- **一策略一「物体-操作」对**：需训练成千上万个专用策略，扩展性差；作者自己提出未来应做条件化于物体/任务编码的**统一 MANO 控制模型**。
- **只发布人手（MANO）数据**：尚未做到向真实机器人手（欠驱动到全拟人）的跨形态 retarget，只是声称力标注为此打下基础。
- **特权观测**：复现依赖未来物体位姿等非因果信息，这在数据合成阶段没问题，但意味着该 RL 策略本身不能直接部署。
- **物体与操作范围窄**：主要是几何原语 + 少量 YCB 物体、基本操作原语，缺少长程、多技能组合、双手操作。
- **RGB-D 与语言标注基本未开发**：视觉模态包含但未利用，语言标注很少。
- **力的「真值」是仿真近似**：力由 IsaacGym 读出，其正确性取决于仿真接触模型与摩擦系数，并非真实力传感器测量，仍可能与真实世界存在系统偏差。

## 五、评价与展望

**优点**：（1）问题定位准——直击「人类演示缺力、合成数据缺自然性」的鸿沟，把「从几何反解力」这个欠约束反问题巧妙转成「仿真追踪 + 物理引擎读力」的正问题，这个 real-to-sim 力反解思路**可迁移到任何已有 mocap 数据集**（如 GRAB/ARCTIC），是本文最有普适价值的方法贡献。（2）系统性覆盖——基于 Cutkosky 分类学的 21 类操作，比多数只覆盖 power grasp + 简单 pinch 的数据集更全面。（3）残差控制 + 特权未来观测的组合，是让 RL 稳定复现人类演示的务实工程选择。

**缺点/存疑**：（1）作为「数据集论文」，最该有的下游价值证明（用它训出的策略在真机上更好）完全缺席，当前更像是「流水线技术报告 + 数据发布公告」。（2）「7,000 小时 / 30 亿帧」的规模主要靠对 70 小时种子做参数扰动 rollout 得到，其**信息多样性**远不及帧数暗示的量级，与 Dex1B（十亿级演示）、DexGraspNet 2.0（4.27 亿抓取）这类合成数据在「有效多样性」上如何比较，缺乏量化分析。（3）成功率定义相当宽松（不掉、不超 5cm 即算成功），80.15% 究竟对应多高质量的力标注，没有与真实力传感器的定量对照。

**与其他公开工作的关系**：相较 EgoDex/OpenEgo 走「规模优先、无力」路线，与 DexGraspNet 系列/Dex1B 走「合成优先、静态抓取、绑定机器人形态」路线，DexCanvas 定位在**「真人种子 + 物理反解连续力 + 形态无关 MANO 桥接」**这一交叉点，其独特卖点是连续力标注与物理有效性，而非绝对规模。与 Chen et al. (2024) 的 object-centric dexterous manipulation from human motion data 在「用人类运动数据 + 仿真」的思路上相近，但 DexCanvas 更强调力的显式提取。

**开放问题与改进方向**：（1）把 per-pair 策略换成条件化统一策略（如条件于 object/task embedding 的单一 MANO 控制器）以真正规模化。（2）补齐向真实灵巧手的 retarget 与真机验证，用力标注做 reward shaping 或接触感知的 VLA 预训练监督。（3）引入真实力/触觉传感器对仿真力做标定，量化 sim-to-real 力偏差。（4）拓展到长程、双手、可变形物体，并挖掘目前闲置的 RGB-D 与语言模态。

## 参考

1. Feix, Romero, Schmiedmayer, Dollar, Kragic. *The GRASP Taxonomy of Human Grasp Types.* IEEE T-HMS, 2016.（21 类操作的分类学来源）
2. Romero, Tzionas, Black. *Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO).* SIGGRAPH Asia / TOG, 2017.（手部参数化模型）
3. Makoviychuk et al. *Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning.* arXiv:2108.10470, 2021.（力反解所用仿真器）
4. Ye et al. *Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation.* arXiv:2506.17198, 2025.（合成灵巧数据规模对照）
5. Hoque, Huang, Yoon, Zhang. *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.* arXiv:2505.11709, 2025.（大规模人类演示、无力标注的对照）
