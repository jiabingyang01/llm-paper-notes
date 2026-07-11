# RoboSnap：单张 RGB 图一次成型的 Real-to-Sim 场景生成，服务可泛化机器人学习与评测

> **论文**：*RoboSnap: One-Shot Real-to-Sim Scene Generation for Generalizable Robot Learning and Evaluation*
>
> **作者**：Shujie Zhang、Jingkun Yi（共同一作）、Weipeng Zhong、Xudong Xu（通讯）、Weinan Zhang、Chunhua Shen 等 + et al.
>
> **机构**：Shanghai AI Laboratory、Shanghai Jiao Tong University、Zhejiang University、Tsinghua University
>
> **发布时间**：2026 年 07 月（arXiv 2607.06699）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.06699) | [PDF](https://arxiv.org/pdf/2607.06699)
>
> **分类标签**：`real2sim` `单图场景生成` `VLA数据合成` `策略仿真评测`

---

## 一句话总结

RoboSnap 用一张 RGB 图就重建出「可交互物理层 + 可重渲染视觉层」分离的仿真就绪场景：前景物体经 SAM 3D 网格化并用 SDF-物理交替优化把悬浮/穿模的独立位姿"落实"到重力一致、接触稳定的布局,背景用 3D Gaussian Splatting 保真;由此在 564 个 DROID 真实场景上构建 DROID-Sim 伴生数据集,并证明重建场景能支撑轨迹回放(5/5)、任务数据合成把 π₀.₅ 真机成功率从 32.7% 提到 41.7%、以及作为评测代理给出 sim-real 皮尔逊相关 r=0.887、MMRV=0.0066。

## 一、问题与动机

机器人基础模型把操作建模为「视觉 + 语言 + 本体感觉 → 动作」的条件生成,规模化的瓶颈是**大规模训练数据**与**可复现评测**。真机采集贵、慢、受硬件约束;仿真本可提供可扩展的数据合成与可重复评测,但要求仿真场景既**物理可信(physically plausible)**又**视觉保真(visually faithful)**。

现有工作只满足其一:

- **程序化/生成式场景合成**(Infinigen、PhyScene、RoboCasa 等)能规模化造出多样可仿真环境,但目标是"造新场景"而非"复刻某个真实野外场景"。
- **重建式 real-to-sim**(digital twin/replica)对齐度高,但通常需要多视角采集或人工精修,难以轻量复用。
- **近期单图系统**(Digital Cousins、CAST、RoLA 等)降低了采集负担,但输出端更窄:或做资产检索、或只做静态重建、或只合成演示,**普遍不能恢复出可重渲染、可编辑、可从新视角复用的仿真世界**,且在下游机器人学习/评测中的有效性验证不足。

作者的核心问题:**能否从单张 RGB 图重建出物理可信、视觉保真、且"仿真就绪"的环境?** RoboSnap 借助单目几何(VGGT)与图像条件 3D 资产生成(SAM 3D、Gaussian Splatting)的最新进展给出正面回答,并强调 real-to-sim 的价值"不止在高保真重建,更在于把真实环境变成可复用的训练与评测基础设施"。

## 二、核心方法

整体分三块:**(1) 单图分层场景重建 → (2) 仿真就绪精修 → (3) 数据生成与评测**。关键设计是**分层(layered)**:把"物理临界的交互区"与"周边视觉上下文"解耦,前者做碰撞感知的可交互资产、后者做保真的可重渲染背景。

### 2.1 单图分层场景重建

给定单张 RGB 图 $I \in \mathbb{R}^{H_0 \times W_0 \times 3}$,先重建初始分层场景 $\mathcal{S}^{(0)}$,再精修为仿真就绪的 $\mathcal{S}^\*$。所有内容注册到规范世界系 $W$——原点取支撑平台质心,重力方向为 $-\hat{e}_z$。

**交互物理层(Interactive Physical Layer)。** 流程:VLM(GPT-4V)解析交互区、给出含支撑平台在内的物体名 $\{\ell_i\}_{i=1}^N$ → SAM 3 抽实例掩码 $\{M_i\}$ → SAM 3D 把每个物体重建成带纹理网格 $\mathcal{M}_i$(含初始位姿与尺度)。为把资产配准到场景几何,用 VGGT 预测相机几何、置信度与稠密点图 $\mathcal{X}_V$;对每个掩码取高置信前景点,用**掩码引导配准**精修 SAM 3D 位姿——在采样的网格表面点与前景点云之间做**粗到细、固定尺度的 ICP**,并拒绝旋转/平移过大的更新,得到 $T_{i \to V}^{\text{init}}$。

**规范对齐与机器人底座。** 从主支撑平台的 $\mathcal{X}_V$ 采点,RANSAC 拟合平面 + 最小二乘精修得到 $T_{V \to W}$(把平面法向对齐到 $\hat{e}_z$、平台质心置于原点),物体位姿抬升到 $W$:

$$T_{i \to W}^{\text{init}} = T_{V \to W}\, T_{i \to V}^{\text{init}}.$$

用大白话说:先在相机系里把每个物体摆好,再找出桌面这个"基准平面",把整个坐标系旋正、让桌面水平、桌心当原点,最后一乘就把所有物体搬进这个"重力向下、桌面水平"的世界里。对 DROID 这类已标定数据,机器人底座直接用相机系底座位姿 $T_{B \to W} = T_{V \to W} T_{B \to V}$ 放置。

**视觉上下文层(Visual Context Layer)。** 前景抠掉后,用 VLM 引导的提示词(Gemini-2.5-flash-image)inpaint 缺失区域得到"空背景",再喂给生成式世界模型(World Labs Marble)产出 Gaussian-splat 背景 $\mathcal{G}_M$(位于 $F_M$ 系),用稀疏对应 + ICP 对齐到 $V$ 再抬升到 $W$。这样背景可在新视角下重渲染,与前景物理层解耦。

**铰接物体(Articulated Objects)。** 用基于点的部件分割(P3-SAM)把重建网格分成语义部件,按估计的部件框切分,再挂到从铰接物体数据集检索来的**类别级运动学参数**上。

初始分层场景记为:

$$\mathcal{S}^{(0)} = \bigl(\{(M_i, \ell_i, \mathcal{M}_i, T_{i \to W}^{\text{init}})\}_{i=1}^N,\ (\mathcal{G}_M, T_{F_M \to W}),\ W,\ T_{B \to W}\bigr).$$

### 2.2 仿真就绪精修(Simulation-ready Refinement)

痛点:逐物体独立估位姿会**悬浮、互相穿模、接触不稳**。作者抽出**物理场景图**,再用 **SDF-物理交替**过程精修位姿。

**场景图抽取。** 借鉴 CAST,用 Set-of-Mark 提示 + VLM 推断成对物理关系:GPT-4V 在 $K=5$ 组随机 SoM 叠加上预测关系,多数投票得到有向 **Support 边**与双向 **Contact 边**,构成 $\mathcal{G}_{\text{phys}} = (\mathcal{V}, \mathcal{E}_{\text{sup}} \cup \mathcal{E}_{\text{con}})$。只支撑别人、不被支撑的物体固定为根 $\mathcal{R}$(如桌面)。

**SE(3) 残差优化。** 只优化非根物体的残差位姿:

$$T_{i\to W} = \Delta T_i\, T_{i\to W}^{\text{init}},\qquad \Delta T_i = \begin{bmatrix}\exp([\Delta r_i]_\times) & \Delta t_i \\ 0^\top & 1\end{bmatrix},\quad i \notin \mathcal{R}.$$

**交替两相(Algorithm 1)。** 一轮内先 **SDF 梯度相**($N_{\text{sdf}}=15$ 步),用预计算的 $128^3$ SDF 网格与表面采样点最小化四项损失;再 **物理落定相**:把网格用 V-HACD 分解成凸碰撞壳,在 SAPIEN 里根固定为 kinematic、其余为 dynamic,前 $N_{\text{damp}}=100$ 步把 XY 速度钳到近零、z 速度限到 0.01 m/s,让物体在重力下**只往下慢慢落定、不横向漂**($N_{\text{sim}}=200$ 步)。落定后的位姿初始化下一轮 SDF,最多 $N_{\text{round}}=20$ 轮,最大位移 $< \varepsilon=10^{-4}$ 收敛,最终得 $\mathcal{S}^\*$。

四项 SDF 损失(以 $\Phi_i$ 为物体 $i$ 的带号距离场,内负外正;$\mathcal{S}_i$ 为表面采样点):

- **穿透损失**:把 $j$ 的表面点变到 $i$ 局部系,惩罚落进 $i$ 内部(SDF 为负)的点:

$$\ell_{ij}^{\text{pen}} = \frac{1}{M}\sum_{s \in \mathcal{S}_j}\max\bigl(0,\ -\Phi_i(T_{i\to W}^{-1}T_{j\to W}\,s)\bigr),\qquad \mathcal{L}_{\text{pen}} = \sum_{i\neq j}(\ell_{ij}^{\text{pen}}+\ell_{ji}^{\text{pen}}).$$

- **支撑损失**(Support 边 $s\to t$):第一项把被支撑物 $s$ 最近表面点拉向支撑者 $t$ 的零等值面(实现"贴上去");第二项惩罚穿透。$t$ 的位姿被 detach,梯度只回流到 $s$。
- **接触损失**(Contact 边 $i,j$):对称双向惩罚,一方向防分离、另一方向防穿透。
- **正则**:对残差的 $\ell_2$ 惩罚防止漂离初始估计,$\mathcal{L}_{\text{reg}} = \sum_{i\notin\mathcal{R}}(\|\Delta t_i\|_2^2 + \lambda_r\|\Delta r_i\|_2^2)$,$\lambda_r=5$。

用大白话说:SDF 相像"一个爱干净的人",把互相插进去的物体推开、把该放桌上的东西按到桌面上;物理相则像"松手让它在重力下自己坐稳",两相反复直到既不穿模又站得住。

### 2.3 数据生成与评测

**分层渲染。** 对查询相机 $Q$,Isaac Sim 渲染物理层得 $(I_{\text{fg}}, D_{\text{fg}}, \alpha_{\text{fg}})$,视觉层从同相机变换到 Gaussian-splat 系渲染得 $(I_{\text{bg}}, D_{\text{bg}})$,按**深度合成**:

$$I_{\text{out}}(u) = m(u)\,I_{\text{fg}}(u) + (1-m(u))\,I_{\text{bg}}(u),\quad m(u) = \mathbb{1}\bigl[\alpha_{\text{fg}}(u) > 0 \ \wedge\ D_{\text{fg}}(u) \leq D_{\text{bg}}(u)\bigr].$$

即前景不透明且比背景近时用前景,否则露出背景。

**轨迹式数据生成。** 把 $\mathcal{S}^\*$ 装进基于 Isaac Sim 的数据引擎(InternDataEngine / InternData-A1),抓取技能用 AnyGrasp 初始化候选,技能模块输出末端 6D 路点,cuRobo 转成碰撞感知的稠密关节动作;经优化(过滤不可行 IK 抓取、正则化放置位姿)可在 8×RTX 4090 上达约 **1 万条/天**任务特定轨迹。

**策略评测指标。** 设 $R$、$R_S$ 为 $N$ 个任务/checkpoint 上的真机与仿真成功率向量,报皮尔逊相关 $r(R,R_S)$ 与均值最大秩违背 MMRV(衡量排序一致性)。$r$ 越高、MMRV 越低表示 sim-real 越对齐。

## 三、实验结果

围绕 Q1–Q5 五问展开。定量评测取 DROID-Sim 中 10 个固定场景(覆盖多样桌面布局与物类),完整 DROID-Sim 覆盖 564 个 DROID 场景。全自动流水线单场景约 12–25 分钟(5–10 个物体平均约 17 分钟)。

### Q1 视觉对齐(10 场景均值,与单帧法 RoLA 同设定)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | SIFT-MR↑ | RGB-EMD↓ | Gabor-L1↓ |
|---|---|---|---|---|---|---|
| RoLA | 13.40 | 0.4521 | 0.4996 | 0.0664 | 28.5806 | 0.001817 |
| **RoboSnap** | 13.25 | **0.4907** | **0.4958** | **0.1226** | **11.4795** | **0.000741** |

因 RoboSnap 重建整个交互区(而非保留大片原背景),PSNR/LPIPS 差异不大(PSNR 甚至略低),但在结构 SSIM、局部特征 SIFT-MR、颜色分布 RGB-EMD、纹理频率 Gabor-L1 上全面更优。

### Q1 仿真稳定性(Table 1,10 场景 Isaac Sim 跑 300 步后)

| 方法 | Falling↓ | Collision↓ | Trans MSE↓ | Mean disp (m)↓ | Quat MSE↓ |
|---|---|---|---|---|---|
| SAM 3D | 0.5640 | 0.3590 | 0.1079 | 0.3284 | 0.1560 |
| SAM 3D + FoundationPose | 0.5900 | 0.1538 | 0.1921 | 0.4383 | 0.1703 |
| RoLA | 0.3810 | 0.3226 | 0.0736 | 0.2713 | 0.1093 |
| RoboSnap w/o refinement | 0.4320 | 0.2982 | 0.0977 | 0.3126 | 0.1255 |
| **RoboSnap** | **0.1026** | **0.0256** | **0.0022** | **0.0474** | **0.0178** |

精修带来数量级提升:掉落率 0.43→0.10、碰撞 0.30→0.026、平移漂移 MSE 0.098→0.0022,直接证明 SDF-物理交替精修是稳定性的关键。

### Q2 轨迹回放(5 个随机场景,回放原始 DROID 夹爪轨迹)

| 方法 | 成功场景 |
|---|---|
| **RoboSnap** | **5 / 5** |
| RoLA | 2 / 5 |

回放成功=夹爪抓到目标物并移到目标区、无严重穿模/碰撞。说明恢复的布局与机器人底座足够准,能复现原演示的关键接触事件。

### Q3 数据合成 + 真机微调(Table 2,四真实设置,3×10 次真机试验均值%)

微调 π₀.₅ 与 π₀,对比 real-only 基线与三种数据混合流(真实演示 / RoboSnap 生成演示 / 仿真增强演示的比例):**R1=(0.2,0.4,0.4)、R2=(0.6,0.2,0.2)、R3=(0,0.5,0.5)**,每任务 30 条真实演示。

| 策略 | Real | R1 | R2 | R3 |
|---|---|---|---|---|
| π₀.₅(平均) | 32.7 | 35.7 | **41.7** | 17.3 |
| π₀(平均) | 29.3 | 31.0 | **42.7** | 15.0 |

Ratio 2 最优:π₀.₅ 32.7→41.7%,π₀ 29.3→42.7%。即便 R3 完全不用真实演示(纯合成)也仍有非零真机成功率,说明合成数据本身可迁移。

### Q4 扰动鲁棒性(Table 3,Avg 行,30 次真机试验%)

六种扰动:物体位姿(±10cm)、背景物、光照、桌面纹理、相机位姿、机器人初始状态。

| 设置 | Orig | Obj. | BG | Lt. | Tex. | Cam. | Arm |
|---|---|---|---|---|---|---|---|
| Real-only | 32.7 | 16.7 | 29.0 | 26.3 | 25.7 | 13.3 | 5.66 |
| Real-sim 共训 (R2) | 41.7 | 33.0 | 39.0 | 37.3 | 35.0 | 31.7 | 23.3 |

Real-sim 共训把平均相对退化从 **13% 降到 8%**,在相机位姿、机器人初始状态两类分布漂移下增益尤为明显(Cam 13.3→31.7、Arm 5.66→23.3)。

### Q5 生成式评测代理(Figure 6)

把 real-only 微调的 π₀.₅ 放到对应生成场景里跑,对比真机与仿真成功率:**皮尔逊 r = 0.887,MMRV = 0.0066**。附录 A.5 进一步用末端位移轨迹与各关节增量分布证明仿真诱发的低层控制行为与真机一致,支持把 RoboSnap 场景当作真机部署的评测代理。

## 四、局限性

1. **输入质量**:仅靠单张 RGB;严重遮挡、极端光照、材质歧义会拉低重建质量与生成演示可靠性。
2. **物体与物理域**:只针对有明确支撑/接触结构的刚体与铰接物;不覆盖可形变、颗粒、流体等刚体仿真难以刻画的对象。
3. **物理参数估计**:没有专门的自动物理参数管线——摩擦、质量靠 VLM 先验;铰接关节参数从标准数据集中同类物体检索得到,精度受限。
4. **进一步验证**:输出的是仿真级场景(而非策略特定数据/标签),原则上可支持不同操作策略(含 video/world-model 结构),但受训练与评测预算所限,论文只在自身设置与模型上验证了该接口,更广的框架验证留待未来。

## 五、评价与展望

**优点。**

- **分层解耦是干净的工程抽象**:把"物理临界的交互区"与"只需保真的背景"分开,前者用碰撞网格做真实仿真、后者用 Gaussian Splatting 保视觉真,回避了"整场重建要么物理烂要么视觉烂"的两难,也让场景可编辑、可复用、可换新视角。
- **仿真就绪精修的消融很有说服力**:Table 1 显示不做精修时掉落/碰撞仍很高,而 SDF-物理交替把稳定性指标压了一个数量级——这正是许多单图 real-to-sim 工作(SAM 3D、SAM 3D+FoundationPose、RoLA)真正落地仿真时的短板,本文用"SDF 推开穿模 + 物理落定重力"这一对互补机制系统性地解决。
- **验证闭环完整**:不止比重建指标,而是一路打通"回放(Q2)→合成数据涨真机点(Q3)→抗扰动(Q4)→当评测代理(Q5)"。r=0.887 的 sim-real 相关与把 π 系列真机成功率提约 9–13 个点,是比一般 real-to-sim 论文更硬的下游证据。
- **DROID-Sim(564 场景)可溯源**:每个重建场景回链到 DROID 原始 identifier,把已有真实数据集"升级"为可复用仿真基础设施,方向对路。

**缺点与开放问题。**

- **视觉指标喜忧参半**:PSNR 甚至略低于 RoLA(13.25 vs 13.40),作者归因于重建整个交互区而非保留原像素;这也暗示前景生成资产的绝对纹理保真仍有差距,长尾/反光/透明材质下 SAM 3D 资产质量存疑(限制 1 已自认)。
- **依赖一长串强外部模块**:GPT-4V + SAM 3 + SAM 3D + VGGT + Gemini inpaint + World Labs Marble + P3-SAM,每一环的失败都会级联污染下游;流水线 12–25 分钟/场景,且多为闭源/收费 API,复现与规模化成本不低。
- **物理参数是 VLM 先验拍脑袋**:摩擦/质量/铰接关节全靠先验或检索,而 Q3/Q4 的接触密集任务对这些参数敏感;缺乏 real-to-sim 系统常配的**可微/系统辨识式参数标定**(对比 TWIN-Aligner、Scalable Real2Sim 等以物理感知资产生成为核心的近期工作)。
- **评测样本偏小**:Q1/Q2 用 10/5 个场景、真机 30 次试验,虽已算扎实,但 sim-real 相关 r=0.887 的置信区间在此规模下应谨慎;Table 3 中 Arm 扰动绝对成功率仍很低(5.66%→23.3%),说明机器人初始状态泛化远未解决。
- **与并行工作的关系**:本文与同期 real-to-sim 评测线(Polaris、Eval-in-sim)、单图重建线(CAST、RoLA、Digital Cousins)、生成式数据增强线(ReBot、RoboEngine、Real2Render2Real)高度交错;其差异化在于"分层 + 仿真就绪精修 + 五问全链验证",而非某单点 SOTA。未来若把参数辨识、可形变物体、以及从视频多视角(附录 D 的 GUI 已支持视频)纳入,潜力更大。

**总体判断**:这是一篇工程完成度高、下游验证充分的单图 real-to-sim 系统论文。真正的技术贡献不在"单图重建"本身(那更多是拼装现有基础模型),而在**把不稳定的独立重建"落实"成仿真可用布局的 SDF-物理交替精修**,以及**把 real-to-sim 当作可复用训练/评测基础设施的完整实证闭环**。对做仿真数据合成与策略评测的人有直接参考价值。

## 参考

1. Yao et al. *CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image.* ACM TOG 2025.（场景图 + Set-of-Mark 关系推断的直接思想来源）
2. Zhao et al. *RoLA: Robot Learning from Any Images.* CoRL 2025.（本文视觉对齐/回放的主要单图 baseline）
3. Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.* arXiv 2024.（DROID-Sim 的真实数据底座）
4. SAM 3D Team. *SAM 3D: 3Dfy Anything in Images.* 2025;Wang et al. *VGGT: Visual Geometry Grounded Transformer.* CVPR 2025.(前景网格化与单目几何配准的两大基座)
5. Li et al. *Evaluating Real-World Robot Manipulation Policies in Simulation.* arXiv 2024.(MMRV 与 sim-real 相关评测协议的来源)
