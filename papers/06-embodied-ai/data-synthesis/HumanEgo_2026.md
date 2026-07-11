# HumanEgo：从数分钟人类第一视角视频中零样本学习机器人操作

> **论文**：*HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos*
>
> **作者**：Zhi (Leo) Wang, Botao He, Kelin Yu, Seungjae Lee, Ruohan Gao, Furong Huang, Yiannis Aloimonos
>
> **机构**：University of Maryland
>
> **发布时间**：2026 年 05 月（arXiv 2605.24934）
>
> **发表状态**：未录用（预印本），作者开源框架
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.24934) | [PDF](https://arxiv.org/pdf/2605.24934)
>
> **分类标签**：`人类视频学习` `具身差距` `Interaction-Centric Tokens` `flow matching` `零样本迁移`

---

## 一句话总结

HumanEgo 把每任务仅约 30 分钟的 Aria 眼镜第一视角人类视频，通过"手臂 inpaint + Interaction-Centric Tokens（ICT）"转成实体级、视角/机体无关的手-物交互表征，再用带三个稠密辅助目标的 flow matching 策略学习双臂动作，在 4 个真实任务上零样本迁移到机器人，平均成功率 92.5%（15 分钟时 75%），比同等采集时长的机器人 teleop（ACT 51.2%）高 41 个百分点，且无需任何机器人数据或大规模预训练。

## 一、问题与动机

人类第一视角视频天然记录了大量操作示范，且采集廉价（一个人戴头显在任何地方几分钟即可采）。但把这些技能迁移到机器人面临 embodiment gap：人与机器人在**视觉外观** 与**运动学** 上都不同。作者指出现有两条路线各有硬伤：

- **Co-training**（EgoMimic/EgoBridge/H0 等）用人类视频补充机器人数据，但每个新任务仍需相当量机器人示范，只是减负而非消除。
- **大规模预训练**（EgoVLA/EgoScale/EgoDex 等）从海量语料学 embodiment-agnostic 表征，但需要巨量算力，且仍需针对机器人的 post-training 才能部署。

作者追求更直接的目标：**只用数分钟人类第一视角示范，不用任何机器人数据、不做互联网级预训练，直接学出可部署的操作策略**。这暴露两个根本挑战：

1. **表征挑战（弥合 embodiment gap）**：视觉侧，retargeting 方法（Phantom/Masquerade/EmbodiSwap）合成类机器人图像但对形态/视角脆弱；point-tracking 方法丢弃交互周边的丰富视觉语境。运动学侧，hierarchical 方法仍需机器人数据训低层控制器；object-centric 方法只跟被操物、丢掉"手如何接近/抓取/释放"的信息。作者的核心主张：机器人**不该模仿人体，而应恢复任务相关的 interaction geometry**（手-物之间的空间关系），这才跨机体可迁移。

2. **学习挑战（从极少数据学习）**：干净且带精确动作标签的片段稀缺，数分钟示范带来 multi-modality（同一任务多种有效策略）与 signal sparsity（每条轨迹信号丰富，但先前方法只用其一小部分，如单一 2D track 或视觉 foresight）两个难题。作者主张：**快速生成式策略 + 多类型稠密监督** 是从数分钟人类视频高效学习的关键——从每条示范里"榨取"尽可能多的监督信号。

## 二、核心方法

HumanEgo 分四阶段（Fig. 2）：Aria 眼镜采集 → 视觉观测预处理（弥合视觉 gap）→ 空间观测预处理 ICT（弥合运动学 gap）→ flow matching 策略生成双臂动作。

### 1. 数据采集

戴 Aria Gen1 眼镜在任意环境执行任务，无需特定工作台/标定；每任务约采 30 分钟人类示范（30 Hz，正文与附录一处记为 40 分钟总时长，单条示范约 30–40 秒）。依赖 Aria 的 Machine Perception Services（MPS）提供：闭环 SLAM 6-DoF 轨迹（每 RGB 帧的相机外参）+ 每帧 21 点/手的标定 3D 手骨架（每手五指指尖 + 腕，直接给到世界系）。

### 2. 视觉观测预处理

两步把畸变校正后的第一视角帧变成 embodiment-agnostic RGB：① 用 SAM2 分割人手/臂并用 LaMa inpaint 抹除，消除视觉 embodiment；② 在 inpaint 后的图上渲染一个虚拟 gripper 与被跟踪物体关键点（均由空间观测导出），把 6D 位姿信息隐式编码为视觉线索。此过程轻量，不需昂贵的域适配或图像翻译。

### 3. 空间观测预处理：Interaction-Centric Tokens（ICT）

把每个物体和两只手都当作一个 entity，跟踪其 6-DoF 位姿，再编码为 ICT。

- **手→gripper retarget**：从 21 点手骨架取 5 个解剖稳定点，构造虚拟平行夹爪的 $SE(3)$ 末端位姿 $T_{ee}$ 与标量抓取 $g$。位置取拇指尖-食指尖中点 $\mathbf{p}_{ee}=(\mathbf{p}_{thumb}+\mathbf{p}_{index})/2$；朝向不用腕系（与真实夹爪动作轴不齐），也不用"腕→指尖中点"（抓合瞬间两指尖重合导致 jaw 轴退化），而是在 MCP 关节上用 Gram–Schmidt 构造（thumb MCP→index MCP 为 x 轴），避免抓取退化。
- **物体跟踪与位姿**：Grounding DINO 文本提示检测 + SAM2 分割 + CoTracker3 跨帧跟 2D 轮廓点 + 三角化抬升到 3D，取质心消三角化噪声，用 Orient-Anything V2 估朝向。抓取时物体被手遮挡，故用 **kinematic latching**：从抓取起点 $t_0$ 起把物体位姿刚性绑到手上。

对每个实体 $k$，ICT 定义为 $\mathbb{R}^{29}$ 向量：

$$
\mathrm{ICT}_k = \big[\; \tau \;\Vert\; {}^{\mathrm{REF}}T_E \;\Vert\; {}^{E}T_{LH} \;\Vert\; {}^{E}T_{RH} \;\Vert\; g \;\big]
$$

其中 $\tau$（1 维）是实体类型（手/物），${}^{\mathrm{REF}}T_E$（9 维）是该实体在共享参考系 REF（静态相机系）下的位姿，${}^{E}T_{LH}$、${}^{E}T_{RH}$（各 9 维）是左右手在**该实体局部系 $E$** 下的位姿，$g$（1 维）是抓取状态。每个 $SE(3)$ 展平为 9D（3D 归一化平移 + 6D 旋转表示）。

**用大白话说**：ICT 不描述"手长什么样"，只描述"手相对每个物体在哪、抓没抓"。因为坐标是相对场景实体而非相对相机表达的，同一个动作从不同视角看会得到几乎相同的 token，这正是跨机体/跨视角零样本迁移的关键；变长接口也天然支持不同数量物体的场景。

### 4. 带稠密辅助目标的 Flow Matching 策略

策略输入场景状态 $s_t$（ICT tokens + RGB 图），输出 $K$ 步双臂动作轨迹 $\mathbf{a}\in\mathbb{R}^{K\times D_a}$（每片拼两手 6-DoF 位姿 + 二值抓取）。用 conditional flow matching 参数化速度场 $v_\theta$（transformer decoder），主损失：

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\big[ w_p \Vert\Delta\mathbf{p}\Vert^2 + w_r \Vert\Delta\mathbf{r}\Vert^2 + w_g \Vert\Delta g\Vert^2 \big], \quad \mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1
$$

其中 $\Delta(\cdot)=v_\theta(\mathbf{x}_t,t,s_t)-(\mathbf{x}_1-\mathbf{x}_0)$ 是速度预测误差，$\mathbf{x}_0\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 为高斯先验，$\mathbf{x}_1$ 是真实双臂动作；权重 $w_p=5,\,w_r=1,\,w_g=10$。推理时用固定步长 Euler 积分 ODE（20 步，$K=50$）。

**用大白话说**：flow matching 学一条从随机噪声流向真实动作的"速度场"，比 diffusion 少很多去噪步、推理快，又比确定性回归能表达"多种有效策略"的多峰分布。

三个**稠密辅助目标**（共享 context encoder）从每条示范榨取更多监督：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{OM}}\mathcal{L}_{\mathrm{OM}} + \lambda_{\mathrm{2D}}\mathcal{L}_{\mathrm{2D}} + \lambda_{\mathrm{LC}}\mathcal{L}_{\mathrm{LC}}
$$

- **Object motion（OM）**：预测被操物未来 6-DoF 轨迹，逼编码器建模手下物体的动力学；
- **2D trace（2D）**：回归实体轨迹未来 2D 投影，把表征 grounding 到视觉观测；
- **Latent consistency（LC）**：预测 $K$ 步后的 ICT 状态，逼编码器捕捉场景动态。

**用大白话说**：三个目标都在不同空间（3D 物理、2D 视觉、latent）里"预测未来会怎样演化"，等于给共享编码器装了一个轻量级手-物交互 world model，同时作为多任务正则抑制过拟合——数据越少收益越大。此外还有两个小技巧：region attention（用可学高斯 spotlight 把图像 cross-attention 偏向当前操作 anchor）与 state-noise injection（训练时给手 token 加噪，让策略对部署期感知噪声鲁棒）。

## 三、实验结果

平台：Trossen WidowX AI 双臂 + 顶置 RealSense D405，每任务 40 次随机初始位置试验。四个真实任务：Serve Bread（pick-and-place）、Downstack Cups（长程多步，拆叠三只嵌套杯，约 1 cm 容差）、Water Flowers（接触密集双臂协作，严格时序）、Adjust Table（持续旋转控制，转柄三整圈不松手）。

### 主结果（Fig. 4，成功率 %）

| 方法 | 数据 | Overall | Serve Bread | Downstack Cups | Water Flowers | Adjust Table |
|---|---|---|---|---|---|---|
| **HumanEgo-30** | 30min 人类 | **92.5** | ~92.5 | 87.5 | 95 | ~92.5 |
| HumanEgo-15 | 15min 人类 | 75.0 | — | — | — | — |
| ACT | 30min 机器人 teleop | 51.2 | — | — | — | — |
| 5 个人类视频基线 | 30min 人类 | 1.9–45.0 | — | — | — | — |

五个人类视频零样本基线为 SPOT、ZeroMimic、Track2Act、PointPolicy、EgoZero，整体在 1.9%–45.0% 之间，无一超过 45%。HumanEgo 是唯一在**每个** 任务都最高的方法，尤其在需要精确协调与空间推理的 Downstack Cups（87.5%，其余 $\le$45%）和 Water Flowers（95%，最好基线 45%）上大幅领先。**仅用一半数据（15 分钟）的 HumanEgo 即达 75%，超过用 30 分钟机器人 teleop 训的 ACT（51.2%）**。

### 数据效率（Serve Bread，Fig. 5–6）

| 人类数据 | ~7min | 8min | 18min | 30min |
|---|---|---|---|---|
| HumanEgo 成功率 | 50% | 57.5% | ~95% | 95% |

- **8 分钟人类数据（57.5%）即超过 30 分钟机器人 teleop 训的 ACT（52.5%），采集成本约降 3.75×**。
- 辅助目标在低数据区收益最大：8 分钟时带辅助 57.5% vs 不带 37.5%；18 分钟后两者收敛到 95%。
- 单条示范采集时间：人类 30–40 秒 vs teleop 60–70 秒（约 2×）。附录 Fig. 6 佐证人类数据 SNR 更高、轨迹更平滑、几乎无空闲、空间/轨迹多样性更大。

### 零样本跨条件泛化（Fig. 7，Serve Bread + Downstack Cups，各 40 次）

全部训练用 Aria 眼镜采，推理却对部署硬件无关。9 个 OOD 条件成功率均在 85%–91.25%：cross-embodiment（UR10 87.5%、Franka 88.75%）、cross-environment（distractors / novel background 均 91.25%）、cross-setup（novel camera 90%、novel viewpoint 87.5%、novel height 86.25%），以及 novel objects 85%、novel lighting 91.25%。训练与部署硬件（相机 RealSense/ZED、臂 Trossen/Franka/UR10）无任何共同点却能无缝迁移。

### 关键消融

**表征研究（Fig. 9，Water Flowers）**：纯视觉方法无论怎么弥合视觉 gap 都卡在 32.5%——Human RGB 7.5% → +关键点渲染与臂 inpaint 20% → 完全无视觉 mismatch 的 Robot RGB 也仅 32.5%。**加 ICT 后 Human RGB 从 7.5% 跳到 85%，全系统 95%（+52.5 pp）**。结论：显式空间表征而非视觉保真才是弥合 embodiment gap 的关键。

**辅助目标研究（Fig. 10，15min）**：无辅助 50%；单加 object motion +17.5 pp（贡献最大）、latent consistency +12.5 pp、2D trace +5 pp；三者合起来 +25 pp（达 75%）。

**手跟踪研究（附录 E.1，Fig. 15，Serve Bread）**：只换手跟踪模块，其余不变——**Aria-MPS（stereo+IMU）95% → WiLoR（单目）45% → HaMeR（单目）32.5% → MediaPipe（单目）0%**。立体深度对下游成功具决定性：单目网络沿深度轴有 5–11 cm 系统偏移，直接污染 ICT 参考系导致学不出一致抓取。作者强调"投资感知前端"是最高杠杆的升级。

**人-机共训（附录 E.2，Fig. 16）**：固定 30min 总时长变人类占比 0→25→50→75→100%，成功率单调 65→72.5→77.5→90→95%，**无 sweet spot，纯人类数据即全局最优**；即便只把 25% teleop 换成人类视频也 +7.5 pp——人类的每分钟信息密度高于机器人 teleop。

**参考系研究（附录 E.3，Fig. 17）**：低数据（4min）anchor frame（27.5%）> camera frame（22.5%），因 anchor frame 提供"抓取先验"的强归纳偏置；大数据（40min）camera frame（95%）反超 anchor frame（87.5%），因其扎根原始传感器、不受上游感知误差污染。主实验用 camera frame；但 anchor frame 独有部署期相机位姿不变性。

## 四、局限性

- **强依赖 Aria 立体手跟踪**：换单目手跟踪真实成功率骤降（95→45→32.5→0%），呼唤能恢复绝对深度的更强单目手位姿估计。
- **逐帧物体检测而非实时跟踪**：涉及遮挡或快速运动的 in-hand manipulation 等动态场景需要在线、抗遮挡跟踪器。
- **流水线级联脆弱**：串联多个 off-the-shelf 感知模块（Grounding DINO / SAM2 / CoTracker3 / Orient-Anything / Aria MPS），任一失败都会级联进策略，作者建议更强或联合训练的前端。
- **精度天花板**：少样本学习在约 1 cm 精度处 plateau；接触密集任务要到亚厘米精度很可能需要 RL 精修或仿真微调。作者自定位为"全零样本、无机器人数据的起点"。

## 五、评价与展望

**优点**：
1. **表征洞见清晰且被消融强力支撑**。"机器人应恢复 interaction geometry 而非模仿人体"这一主张，被 Fig. 9 那个反直觉的对照（完全无视觉 mismatch 的 Robot RGB 也只有 32.5%，加 ICT 才跳到 85%+）钉死——明确把瓶颈从"缩小视觉/像素差距"转到"显式空间关系"，对整条 human-to-robot 迁移路线有校正意义。
2. **数据效率数字硬**。8 分钟人类视频超过 30 分钟机器人 teleop、共训消融显示纯人类数据全局最优，是对"human video 只是廉价替代品"叙事的有力反驳——它主张人类视频可能是**信息密度更高** 的数据源。
3. **诚实的负面消融**（手跟踪 0%、单目骤降）比很多只报正面结果的工作更可信，也精确地指出了整个范式的真实瓶颈在感知前端而非策略。

**缺点与开放问题**：
1. **对 Aria MPS 立体硬件的深度依赖削弱了"随处可采"的卖点**——真正 in-the-wild 的单目网络视频（EgoVLA/EgoDex/EgoScale 所依赖的语料）在此范式下会崩，与其"从数分钟视频学习"的普适愿景存在张力。
2. **仅 4 个真实任务、双臂但均为准静态/低速**，Water Flowers 与 Adjust Table 虽接触密集但动作缓慢；kinematic latching 假设抓取后物体刚性跟手，对可形变物、滑移、精细 in-hand 重定向不成立。
3. **与相邻工作的边界**：与 EgoZero/PointPolicy（point-based）、SPOT（object-centric SE(3) 轨迹）、Track2Act/ZeroMimic（goal-conditioned 2D/3D track）相比，HumanEgo 的差异化正是"手-物 relative pose"这一 interaction token；但它并未与需要少量机器人数据的 co-training 方法（EgoMimic/H2R）在同等真实任务上做数据-性能前沿对比，因此"完全不碰机器人数据是否总更优"仍未被充分证伪。
4. **改进方向**：把级联感知换成联合/端到端可训的前端（作者自己也指出）；引入单目自监督深度或时序立体以摆脱 Aria 依赖；把 latent-consistency 辅助头扩成真正可 rollout 的 world model 用于规划或 RL 精修突破 1 cm 天花板；以及在更动态、更多物体的任务上验证 ICT 变长接口的可扩展性。

总体上，这是一篇**表征主张鲜明、消融扎实、数字有说服力** 的工作，其"interaction-centric 空间 token + flow matching + 稠密辅助监督"的组合，为"少量人类视频→可部署机器人策略"提供了一个强 baseline 与清晰的失败边界。

## 参考

1. Kareer et al. *EgoMimic: Scaling imitation learning via egocentric video.* ICRA 2025.（co-training 代表，HumanEgo 主要对照的另一范式）
2. Yang et al. *EgoVLA: Learning vision-language-action models from egocentric human videos.* arXiv 2507.12440, 2025.（大规模预训练范式代表）
3. Liu et al. *EgoZero: Robot learning from smart glasses.* arXiv 2505.20290, 2025.（point-based 零样本基线，本文对照方法之一）
4. Hsu et al. *SPOT: SE(3) pose trajectory diffusion for object-centric manipulation.* arXiv 2411.00965, 2024.（object-centric 基线，本文对照方法之一）
5. Lipman et al. *Flow matching for generative modeling.* ICLR 2023.（策略生成式主干的理论基础）
