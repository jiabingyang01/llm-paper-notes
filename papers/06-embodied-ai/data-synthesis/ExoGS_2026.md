# ExoGS：面向可扩展操作数据采集的 4D 真实-仿真-真实框架

> **论文**：*ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection*
>
> **作者**：Yiming Wang, Ruogu Zhang, Minyang Li, Hao Shi, Junbo Wang, Deyi Li, Jieji Ren, Wenhai Liu, Weiming Wang, Hao-Shu Fang
>
> **机构**：Shanghai Jiao Tong University（上海交通大学）
>
> **发布时间**：2026 年 01 月（arXiv 2601.18629）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.18629) | [PDF](https://arxiv.org/pdf/2601.18629)
>
> **分类标签**：`real2sim2real` `3DGS 数据增强` `外骨骼演示采集`

---

## 一句话总结

用一套与目标机器人运动学同构的被动外骨骼 AirExo-3（末端定位误差 <1 mm、成本约 $400）在无真机情况下采集人类接触密集演示,将机器人、物体、环境重建为可解耦、可编辑的 3D Gaussian Splatting 资产,通过 forward kinematics 驱动 4D 重放并做视角/光照/背景/位姿四类几何一致增强(数据可扩到原始的 20 倍);再用轻量 Mask Adapter 把实例级语义注入 ViT 策略以缓解残余 sim-to-real gap——在"新物体 pick-and-place"任务上纯合成数据训练达 76% 成功率而遥操作数据为 0%,增强后策略在颜色/背景/光照扰动下大幅超过真机演示训练的策略。

## 一、问题与动机

模仿学习的效果高度依赖演示数据的规模与质量,而两条主流的扩量路线各有硬伤:

- **纯仿真数据合成** 可扩展,但存在众所周知的 sim-to-real gap——几何表征、视觉外观、物理交互三方面的偏差,尤其 RGB 策略对视觉保真度敏感。
- **既有 Real-to-Sim-to-Real(R2S2R)管线**(基于 NeRF / 3DGS 的神经场重建)确实能靠 photorealistic 渲染缩小视觉差距,但**几乎都停留在静态场景重建**,交互数据仍要靠在仿真里跑强化学习获得。对接触密集(contact-rich)任务而言,在纯仿真中获取物理有效、高保真的交互轨迹既困难又低效,且往往需要昂贵真机硬件。

作者的核心洞见:**交互本身应当在真实世界被捕获,而不是在仿真里重新学**。为此提出 ExoGS——一个低成本、robot-free 的 4D R2S2R 框架,用户可捕获包含 3D 环境与"物体-机器人时空交互"的 4D 序列,把真实交互直接搬进仿真做几何一致的大规模增强。三点贡献:(1)4D、robot-free 的 R2S2R 框架,把真实资产与操作序列重建成可编辑 3DGS 资产及其动力学;(2)AirExo-3,一款开源、低成本、高精度、耐用的演示采集设备;(3)Mask Adapter,把语义 mask 注入策略以把注意力引向交互相关区域。

## 二、核心方法

整体流程:AirExo-3 采演示 → 多视图 3DGS 重建 + FoundationPose 物体位姿跟踪 → forward kinematics 驱动 4D 重放 → 四类增强扩量 → Mask Adapter 做 sim-to-real 迁移。

### 2.1 硬件:AirExo-3 被动外骨骼

一条与目标机器人几何匹配的串联连杆-关节链,共 7 个关节 + 一个可更换手持并联夹爪,**共享与真机完全相同的运动学参数、关节限位、夹爪开合范围**,以保证 workspace 一致。全部用玻纤增强尼龙 3D 打印(轻量耐用),关节 4 可选装弹簧悬吊以减负。关节模块用 12 位微型旋转编码器 + cross-roller 轴承(高刚度、高旋转精度);8 个编码器挂在共享总线上,可 ~300 Hz 同步采关节状态。采用 [23](AirExo-2)的标定评测协议,**平均末端定位误差 <1 mm**。相比手持式 UMI 等设备,它靠编码器 forward kinematics 而非视觉,对视觉条件不敏感且强制满足机器人一致的运动学约束。整机成本约 $400。目标平台为 Flexiv Rizon 4s + Xense Aurora Lite 夹爪。

### 2.2 演示表示与 forward kinematics 驱动

一条演示轨迹表示为关节角 + 夹爪开合的离散时间序列:

$$
\tau = \{(q_t, g_t)\}_{t=1}^{H}, \quad q = [q_1, \dots, q_n]^\top, \; g \in [0,1]
$$

用大白话说:每一帧就记两样东西——机械臂各关节的角度、夹爪开多少。因为外骨骼与真机运动学同构,这些关节角 $q_t$ 可直接喂 forward kinematics,在真实与仿真里都能高保真复现同一条轨迹。多台标定过的 Intel RealSense D415 同步采多视图 RGB-D 观测 $I = \{I_t^{(k)}\}$,统一到公共世界坐标系为后续重建/位姿估计提供几何约束。

每个连杆的 6D 位姿由真机 URDF 的 forward kinematics 得到:

$$
T_{\ell,t} = \mathrm{FK}_\ell(q_t), \quad \ell = 1, \dots, L, \; t = 1, \dots, H
$$

用大白话说:拿真机的 URDF 模型,把录到的关节角一转,就知道每个连杆此刻该摆在哪;把这些连杆位姿和物体位姿一起塞进 Gaussian 渲染管线,就能多视角渲染出"机器人-物体"同框画面训策略。

### 2.3 3DGS 资产化与物体位姿

- **资产生成**:采用标准 3DGS,场景(机器人、物体、环境)表示为一堆带位置/协方差/不透明度/球谐的 3D Gaussians。走 "capture-reconstruct-assetize" 流程——多视图图像先过 COLMAP 恢复相机位姿,再以此初始化优化 Gaussian 参数,损失是渲染视图与真实视图之间加权的 L1 + SSIM 光度损失。产出的是**解耦的**高保真资产(机械臂 / 被操作物体各自独立),从而能单独操纵、几何一致地重放。
- **物体位姿**:多视图 RGB-D 序列过 FoundationPose 做 6D 位姿跟踪,得 $T_{o,t}^{(k)} \in SE(3)$;多视图融合策略为"取主相机的旋转做全局朝向 + 跨视图平移取平均",得到机器人基坐标系下统一的 $\{T_{o,t}\}$。
- **PoseProcess 模块**:对物体位姿序列做归一化与重组。其中 fix 操作把物体位姿约束到末端执行器坐标系,实现操作中的刚性附着;通过替换物体模型,**同一条位姿序列可直接迁移到不同物体**,无需额外采集即可生成多样任务实例。

### 2.4 高斯渲染下的四类数据增强

依托可编辑 3DGS 表示做四类增强:(1)**相机视角**——扰动外参渲染,模拟相机摆位变化;(2)**颜色与光照**——随机缩放 Gaussian 颜色属性及全局/局部亮度,补外观与光照差;(3)**背景**——在几何一致的高斯前景后合成多样真实图像做背景,鼓励背景无关学习;(4)**物体位姿**——扰动物体位姿/尺度,或替换为 affordance 兼容的替代物,实现轨迹复用。物体替换可把数据扩到原始的 10 倍;四类增强合起来可扩到 **20 倍**。

### 2.5 策略模块:Mask Adapter(两阶段)

骨干是增强版 ACT + DINOv3 ViT 编码器 + LoRA 微调。ViT 产出 patch tokens $x$ 与基础位置编码 $p$。Mask Adapter 分两阶段训练。

**Stage 1(分割预训练)**:用 3DGS 管线免费生成的像素级监督微调视觉编码器 + 一个 ASPP 风格轻量多尺度分割头 $H_{\text{mask}}$,预测像素 logits $S$;再把像素预测聚合为 patch 级标签(背景 / 机械臂 / 物体):

$$
\ell_n = \arg\max_{c \in \{0,\dots,C-1\}} \frac{1}{|\Omega_n|} \sum_{u \in \Omega_n} \mathrm{softmax}(S_u)_c
$$

用大白话说:一个 patch 盖住一小片像素,把这片像素的分类概率求平均再取最大类,就给这个 patch 贴上"背景/臂/物体"的身份标签。分割损失是(可选类加权的)像素交叉熵 $\mathcal{L}_{seg}$。

**Stage 2(mask 引导策略)**:用 patch 标签同时做两件事。其一,增强位置编码:

$$
\tilde{p} = p + E_{\text{label}}(\ell)
$$

用大白话说:在每个 token 的位置编码上,再加一个"你属于背景还是臂还是物体"的可学习 embedding,让策略天生知道每个 patch 的语义身份。其二,构造标签关系集 $\mathcal{R}$ 与加性注意力掩码:

$$
A_{ij} = \begin{cases} 0, & (\ell_i, \ell_j) \in \mathcal{R} \\ -\infty, & (\ell_i, \ell_j) \notin \mathcal{R} \end{cases}
$$

用大白话说:只放行语义上相关的 token 对(如 物体-物体、物体-臂、臂-物体)做注意力,把"背景-物体"等无关组合用 $-\infty$ 屏蔽掉,避免策略学到"看背景猜动作"的捷径。Stage 2 联合优化动作损失与分割:

$$
\mathcal{L}_{stage2} = \mathcal{L}_{act} + \lambda \mathcal{L}_{seg}
$$

无真值 mask 时,可只优化 $\mathcal{L}_{act}$,同时用 Stage-1 的预测提供 $\ell$ 做位置增强与注意力约束。该模块只需 ViT tokens、位置编码、注意力接口,可低改动地插进大多数 ViT-based 2D 模仿学习策略。

## 三、实验结果

设置:Flexiv Rizon 4s + Xense Aurora Lite 夹爪(位置控制)+ 头顶 RealSense D415。三任务:Pick and Place、Pick Place Close(放完再合盖)、Unscrew Bottle Cap(拧开固定在桌面的瓶盖,接触密集)。每任务 AirExo-3 采 60 条原始演示、另用遥操作采 60 条对比;策略评测每任务 25 次试验。采集效率实验招募 10 名无机器人背景志愿者,每人约 10 分钟培训、每任务录 6 条有效演示,只用成功演示算指标。

**采集效率(Table I:采集成功率)**——AirExo-3 全面高于遥操作,任务越难差距越大,拧瓶盖最悬殊(遥操作大量失败、易碰撞损物):

| 任务 | AirExo-3 | Teleoperation |
|---|---|---|
| Pick and place | 100% | 92.3% |
| Pick place close | 100% | 83% |
| Unscrew bottle cap | 87% | 17% |

采集耗时(Fig. 6)上 AirExo-3 也更快,简单任务相当、复杂任务优势拉大,且用户间方差更小(一致性更好)。

**无增强的策略性能(Table II:策略成功率)**——原始合成演示因渲染-真实的残余视觉差,多数任务略逊遥操作;但拧瓶盖任务反超(遥操作轨迹太噪);**关键亮点是"新物体"设定**——纯靠数字资产替换生成、物理采集中从未出现过的物体:

| 任务 | ExoGS | Teleop |
|---|---|---|
| Pick and place | 50% | 72% |
| Pick place close | 48% | 64% |
| Unscrew bottle cap | 24% | 8% |
| Pick and place（新物体） | **76%** | **0%** |

即通过低成本资产替换把数据扩到 10 倍,策略可逼近真机数据水平,而在遥操作根本没有对应新物体数据时(0%)优势碾压。

**带增强的泛化性能(Fig. 8a)**——用四类增强把数据扩到 20 倍后,在视觉扰动下增强策略持续优于遥操作、甚至优于真机数据训练的策略,尤其在颜色/背景/光照变化下(此时纯真机数据训练常常完全失败)。以 Pick(and place)为例的成功率对比(增强 AirExo vs 遥操作):相机视角 88% vs 32%、颜色 96% vs 8%、背景 80% vs 4%、彩色光照 72% vs 0%。拧瓶盖任务增益有限——瓶盖螺纹耦合带来的滑移/卡死等**运动学约束**才是瓶颈,而非视觉,受操作者熟练度限制。

**增强策略消融(Fig. 8b,在 Pick Place Close 上)**——建三个各 10 倍的数据集:A=视角、B=外观(背景+光照随机 + 大范围 color jitter,color jitter 因与背景外观耦合不单列)、C=物体位姿。结论符合直觉:视角变化与 color jitter 最能扩训练域、增益最大(color jitter 因合成-真实持续存在的色/光差而整体最强),如 Dataset A 在"相机视角变化"下达 72%,Dataset B 在颜色/背景/彩光变化下达约 84–88%;而**位姿增强收益微弱**——采集时物体位姿本已足够多样,且位姿扰动不针对主要的视觉域差。

**Mask Adapter 效果(Fig. 10,仅用无增强数据)**——Mask Adapter 直接在原始数据上工作、与增强兼容,是比"堆 20 倍合成数据"更省存储/算力的替代路线。在原始场景下 Pick and place 达 88%、Pick place close 达 78%,**得益于 AirExo-3 更高质量轨迹,甚至超过遥操作数据训练的策略**;对物体颜色变化有一定鲁棒性(颜色变化下 44%/40%);但在剧烈背景变化(20%/24%)、彩色光照(12%/16%)等会破坏分割模型的严苛条件下仍明显退化。Fig. 9 显示分割模型仅用"容器 + 绿块"训练,却能对未见新物体/背景稳健分割。

## 四、局限性

- **刚体假设**:整套依赖 3DGS 的刚体假设,难以建模几何可变的**可变形物体**,作者明确将此列为主要限制。
- **接触密集任务的瓶颈在物理而非视觉**:拧瓶盖类任务失败主要来自螺纹耦合的滑移/卡死等运动学约束,数据增强与 Mask Adapter 都难改善;且质量受操作者熟练度制约。
- **残余视觉差仍在**:无增强时合成数据在多数任务上仍逊于真机遥操作,渲染-真实差距未被完全消除。
- **Mask Adapter 依赖分割稳定性**:剧烈背景/彩光会先击垮分割模型进而拖垮策略,鲁棒性上限受分割头制约。
- **规模有限**:任务仅 3 个、每任务 25 次试验、单臂单相机,统计力度与任务多样性偏弱;自动化程度仍需人工介入(作者将"提升管线自动化"列为未来工作)。

## 五、评价与展望

**优点**。(1)问题切口精准:把 R2S2R 从"静态环境重建 + 仿真里 RL 学交互"转向"真实世界直接捕获交互 + 仿真里几何一致重放",绕开了接触密集交互在纯仿真中最难合成的部分,这一 4D、robot-free 定位是相对 SplatSim、RoboGSim、Real2Render2Real 等渲染中心/RL 中心管线的清晰差异化。(2)硬件-软件协同扎实:同构外骨骼 + forward kinematics 让"人手演示"天然是 embodiment-consistent 的伪机器人数据,<1 mm 精度与 $400 成本兼顾,叠加解耦 3DGS 资产,使"物体替换零成本扩量"这一条(新物体 76% vs 0%)成为最有说服力的证据。(3)Mask Adapter 思路优雅:用 3DGS 免费产出的像素级 mask 做语义先验,以注意力掩码抑制背景捷径,是把生成管线的副产物反哺策略鲁棒性的巧妙闭环,且即插即用。

**缺点与开放问题**。(1)刚体假设是硬伤,可变形物体、流体、绳索类任务无法覆盖;可探索与 DeformGS、PhysGaussian 等动态/物理先验 3DGS 的结合。(2)Mask Adapter 的注意力掩码为硬 $\{0, -\infty\}$ 二值,关系集 $\mathcal{R}$ 由固定语义类人工设定,缺乏对"背景中偶尔相关"情形的柔性,软掩码或可学习关系权重是自然改进。(3)接触密集任务未真正解决——外骨骼虽给出高质量轨迹,但缺乏力/触觉信号闭环,螺纹类装配仍受挫;引入力反馈或触觉传感(其夹爪已是 Xense 触觉夹爪,却未见触觉进入策略)是明显缺口。(4)评测规模与统计严谨性偏弱,25 次试验的百分比数字方差大,缺置信区间;与 SplatSim/Real2Render2Real 等同类 R2S2R 方法缺少直接对照,目前基线只有遥操作。(5)DINOv3 + LoRA + ACT 的组合较重,论文未报训练/推理开销与实时性。

**展望**。该工作把"低成本外骨骼采集"与"可编辑 3DGS 增强"两条既有线索(AirExo 系列 + 3DGS 机器人世界模型)缝合得相当完整,给"真实交互驱动的可扩展操作数据引擎"提供了一个可复现(已开源代码与硬件)的范式。后续最有价值的方向:柔性/可变形建模、把力-触觉纳入 4D 表示与策略、以及在更大任务集与更强 VLA 骨干上验证增强数据的可迁移性。

## 参考

1. Fang et al., *AirExo: Low-cost Exoskeletons for Learning Whole-arm Manipulation in the Wild*, ICRA 2024（AirExo-3 的前身与低成本外骨骼采集范式）。
2. Fang et al., *AirExo-2: Scaling up Generalizable Robotic Imitation Learning with Low-cost Exoskeletons*, Robot Learning Workshop 2025（本文采用其运动学精度评测协议）。
3. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM ToG 2023（3DGS 表示与渲染基础）。
4. Qureshi et al., *SplatSim: Zero-shot Sim2Real Transfer of RGB Manipulation Policies using Gaussian Splatting*, CoRL 2024 Workshop（最直接的 3DGS-based R2S2R 对照工作）。
5. Wen et al., *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects*, CVPR 2024（本文物体位姿跟踪所用方法）。
