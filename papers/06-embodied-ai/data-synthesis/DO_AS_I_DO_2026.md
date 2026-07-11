# DO AS I DO：从日常人类视频中提取灵巧操作数据

> **论文**：*Do as I Do: Dexterous Manipulation Data from Everyday Human Videos*
>
> **作者**：Bhawna Paliwal\*, Haritheja Etukuru\*, William Liang\*, Pieter Abbeel, Nur Muhammad "Mahi" Shafiullah, Jitendra Malik（\* 表示共同一作）
>
> **机构**：UC Berkeley
>
> **发布时间**：2026 年 06 月（arXiv 2606.19333）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.19333) | [PDF](https://arxiv.org/pdf/2606.19333)
>
> **分类标签**：`人手视频retarget` `灵巧操作数据合成` `hand-object重建` `dynamics-aware retargeting`

---

## 一句话总结

DO AS I DO 用视觉基础模型从**单目 RGB 日常人类视频**里重建 4D 手-物交互轨迹，再用采样式动力学感知 retargeting 把它迁移到 22-DoF 多指灵巧手上，端到端产出**可在真机上回放**的操作数据;在 DexYCB/HOI4D 上重建 SOTA(F-5 0.71/0.72),retarget 在 in-the-wild 重建轨迹上把成功率从 25% 提到 71%,并给出"只有约 5% 的互联网 clip 真正可用"的数据过滤实证。

## 一、问题与动机

机器人学习的数据几乎全是**经验数据**(experiential):真机遥操或仿真探索采集,受操作者专业度、装置成本、环境/奖励设计所限,难以规模化;而人类天然产生的是海量**观察数据**(observational)——单目 RGB 视频。把观察数据转成机器人可用的经验数据,是 "Do as I Do"(照我做)这一古老命题的核心。

已有工作要么依赖专门硬件(depth、3D 关键点、smart glasses)或强先验(仅 pick-and-place、仅 3D 扫描过的物体、已知物体类目),要么其重建质量在真实"野外"视频上崩掉、retargeting 又假设有干净的 MoCap 参考。本文瞄准**最一般也最难**的设定:以单目 RGB 视频为一等公民的**主要数据源**,不假设抓取先验、不限物体类别,支持任意刚体、支持第一/第三人称、甚至支持生成式视频模型的输出。

作者的关键洞察是两条相邻领域的最新进展让这件事第一次变得可行:(1) 3D 视觉基础模型(单目估深度 MoGe、单图生 3D 网格 SAM 3D、单目重建手 HaWoR)可从纯 RGB 恢复 4D 手-物状态;(2) GPU 并行物理仿真器(MuJoCo Warp、Isaac)让采样式优化能在几分钟内从 4D 手-物状态里反推出灵巧手动作。两者对问题结构假设都极少,恰好可拼成从"互联网视频 → 真机灵巧手 rollout"的完整管线(据作者称为业界首个)。

## 二、核心方法

DO AS I DO 分两步:**重建**(3.1,恢复并跟踪 3D 手与物)与 **retargeting**(3.2,把轨迹迁移到机器人本体,产出动力学可行的动作)。所用机器人为 22-DoF Sharpa Wave 灵巧手,真机为双臂 UR3e + Sharpa Wave,50 Hz 指令。

### 2.1 重建:用引导扩散把单图生成模型变成视频物体跟踪器

管线并行跑三个基础模型:SAM 3 做手/物分割、MoGe 估 metric depth 与相机内参、SAM 3D 从单帧生成物体 3D 网格;手的跟踪直接用 HaWoR(对遮挡、运动模糊、低分辨率鲁棒)。难点在**物体的时序位姿跟踪**:6-DoF tracker(FoundationPose、Any6D)在野外视频里一旦视觉证据退化就容易丢失锁定、漂移、无法重新捕获。

作者的做法是把图生 3D 模型 **SAM 3D 改造成视频物体跟踪器**。SAM 3D 学的是给定单张 2D 图与掩码时形状与位姿的联合分布 $p_\theta(x^s, x^p \mid c)$,逐帧独立跑没有时序一致性。关键观察是**形状与位姿共享同一潜空间**,于是在锚定帧固定形状 $\bar{x}^s$,再给定上一帧位姿 $x^p_{k-1}$,只预测当前帧位姿 $x^p_k$——跟踪退化为从 $p_\theta(x^p_k \mid \bar{x}^s, c_k)$ 采样并偏置向 $x^p_{k-1}$。由于在 6-DoF 连续位姿空间上对所有位姿做边缘化不可行,作者利用 flow matching 推理:沿线性路径 $x_t=(1-t)x_0 + t x_1$ 积分 ODE,每个 Euler 步先做模型的 free Euler 更新,再把每个 block 朝**目标插值物**(target interpolant)混合:

$$x_t^s = \underbrace{(1-\alpha_s)(x_{t-\Delta}^s + \Delta v_\theta^s)}_{\text{denoising}} + \underbrace{\alpha_s\, z_{\text{ref}}^s(t)}_{\text{blending}}, \qquad x_t^p = (1-\alpha_p)(x_{t-\Delta}^p + \Delta v_\theta^p) + \alpha_p\, z_{\text{ref}}^p(t)$$

其中 $z_{\text{ref}}^s(t)=(1-t)\epsilon^s + t\,\bar{x}^s$、$z_{\text{ref}}^p(t)=(1-t)\epsilon^p + t\,x^p_{k-1}$ 是目标插值物,$\alpha_s,\alpha_p\in[0,1]$ 是引导强度。

**用大白话说**:一边让扩散模型自由地"想象"这帧物体长啥样、朝哪,一边不断把它往"形状要和锚定帧一致、位姿要接近上一帧"这两个锚拽回来。$\alpha$ 越大越听参考的话、越不容易乱跳。

**自适应引导强度**。形状固定引导 $\alpha_s\in[0.9,1]$ 即可。位姿的 $\alpha_p$ 若固定则要么太僵、要么出现虚假翻转,故从数据里推:用 BootsTAPIR 在物体掩码内采 20 个点做点跟踪,对相邻帧的点集用 SVD 拟合 2D 刚性变换,得到面内旋转量 $\Delta\theta_k$,再取

$$\alpha_p(k) = \max\!\big(0.1,\; 0.7 - 0.09\,|\Delta\theta_k|\big)$$

即物体转得快时放松引导让模型跟上、转得慢时收紧引导抗噪。

**逐帧位姿选择(采样-聚类而非算 log 密度)**。引导采样是随机的,每帧抽 $N$ 个候选 $\{x^p_{k,i}\}$ 需选一个。原则上按模型条件 log 密度排序最优,但通过瞬时变量替换公式(Eq. 2)精确算 log 密度,单帧要约 8.7k 次扩散主干前向+反向(约 $NT(1+D_p)$),比生成本身贵两个数量级,视频尺度下不可承受。作者改用**加权 SE(3) 距离**下的聚类启发式:

$$d(x_i^p, x_j^p) = w_t\,\|t_i - t_j\|_2 + w_r\cdot 2\arccos|\langle q_i, q_j\rangle|$$

对 $N=25$ 个位姿聚类,丢弃过小簇(视为离群),再用各簇的 2D 轮廓与输入掩码的 IoU 排序。直觉是**置信采样会聚到同一位姿模态,而估计噪声在 SE(3) 上散开**,共识过滤+mask-IoU 就能挑出 mode-best 位姿,完全不必再调用扩散主干,比 log 密度法快至多 30×。

**手-物对齐**。手与物独立重建、尺度可能不同,以手的重建尺度为准去缩放物体平移。用 MoGe pointmap 算手/物质心 $\mathbf{c}^M_{\text{hand}},\mathbf{c}^M_{\text{obj}}$ 与 HaWoR 手网格可见部分质心 $\mathbf{c}^H_{\text{hand}}$,尺度比 $k = z^H_{\text{hand}}/z^M_{\text{hand}}$,目标物体位置

$$\mathbf{obj}_{\text{target}} = \mathbf{c}^H_{\text{hand}} + k\,(\mathbf{c}^M_{\text{obj}} - \mathbf{c}^M_{\text{hand}})$$

再固定 SAM 3D 预测的网格朝向、用一维最小二乘沿视线求平移尺度 $s$,最后用 GeoCalib 对齐重力方向,得到 metric 4D 手-物轨迹。

### 2.2 Retargeting:面向噪声参考的动力学感知采样优化

重建出的手-物参考不完整:人手与机器人形态不同、缺接触与力信息、还带时序断裂与手物错位。作者在 SPIDER(Pan et al. [15])的框架上做**动力学感知 retargeting**——在物理仿真里 MPPI 式采样优化,核在迭代与预测 horizon 上退火(先粗探索后局部精化)。仿真用 MuJoCo Warp(200 Hz,0.005s 步长),物体网格 CoACD 凸分解并加厚 2mm 以稳定接触;每 0.5s(2 Hz)在 3s horizon 上规划,每步 1024 samples、32 iterations。针对噪声参考,作者加了三项创新:

1. **Warmup Steps**:初始首帧可能进入无法恢复的状态(如物体没被抓住)。在参考前面前置 $H$ 个 warmup 步——warmup 期间物体被 weld 固定在原地、机器人手可自由运动,之后撤掉 weld 正常仿真。这让优化器在开始跟踪前先调整好手的位姿(避免掉物),且**不假设任何抓取先验或启发式**,复用原有优化流程。这是三项里贡献最大的一项。
2. **Random Force Perturbation**:rollout horizon 可能困在"短暂跟上但无法恢复"的局部最优(如物体在指尖上勉强平衡)。借鉴 sim-to-real,在采样 rollout 时施加随机力,逼出对扰动鲁棒的控制;通用、不假设高保真参考(区别于 SPIDER 的 contact guidance)。
3. **Transition Reward**:物体在"静置↔在手"之间的转换是轨迹关键拐点,但噪声参考下仅靠 tracking reward 太不精确。对失败转换加常数惩罚:(1) 静置参考步却缺物-地接触;(2) 在手参考步却缺手-物接触。转换阶段由参考手物距离低于阈值 $\epsilon$ 判定。

## 三、实验结果

### 3.1 重建(手-物跟踪)

在 DexYCB(160 段有标注视频)与 HOI4D(12 段)上用 F-5、F-10、Chamfer distance(CD)评物体级重建/跟踪(输入真值手位姿以隔离物体性能)。DO AS I DO 在两个数据集上均刷新 SOTA:

| 方法 | DexYCB F-5↑ | DexYCB F-10↑ | DexYCB CD↓ | HOI4D F-5↑ | HOI4D F-10↑ | HOI4D CD↓ |
|---|---|---|---|---|---|---|
| HO | 0.24 | 0.48 | 4.76 | 0.28 | 0.51 | 3.86 |
| IHOI | – | – | – | 0.42 | 0.70 | 2.7 |
| HORSE | 0.23 | 0.42 | 6.97 | 0.26 | 0.45 | 6.69 |
| MCC-HO | 0.36 | 0.60 | 3.74 | 0.52 | 0.78 | 1.36 |
| G-HOP | 0.31 | 0.49 | 8.11 | 0.69 | **0.91** | 0.63 |
| FoundationPose | 0.69 | 0.89 | 0.89 | 0.71 | **0.91** | 0.49 |
| Any6D | 0.69 | 0.88 | 0.97 | 0.71 | **0.91** | 0.50 |
| **Ours** | **0.71** | **0.93** | **0.66** | **0.72** | **0.91** | **0.49** |

在 150 段 in-the-wild(互联网/第一人称/生成视频)基准上无真值,改用人评:每段展示原视频+两个重投影(本方法 vs FoundationPose 的物体位姿投回 2D),问哪个跟踪更一致。人评偏好本方法 **67%**、FPose 18%、平局 15%(非平局胜率 79%);75% 视频三名评者一致,Fleiss $\kappa=0.65$(实质一致)。

物体跟踪消融(Table 5):自适应位姿引导相对固定引导稳定提升(HOI4D CD 0.50→0.49、F-5 0.69→0.72);聚类式候选选择与 log-likelihood 选择基本持平(DexYCB 0.71 vs 0.72),但速度快至多 30×;随机选候选则明显更差(HOI4D F-5 掉到 0.62)。

### 3.2 Retargeting

在 655 段 in-the-wild 重建参考、以及 OakInk2(1352 段干净双手 MoCap 轨迹)上评。成功判据:平均位置误差 $E_{\text{pos}}<0.1\,\text{m}$ 且平均旋转误差 $E_{\text{rot}}<0.5\,\text{rad}$。以 SPIDER(即 Annealed Sampling)为基线逐项累加三个组件:

| 方法 | 重建-成功率↑ | 重建-Pos↓ | 重建-Rot↓ | OakInk2-成功率↑ | OakInk2-Pos↓ | OakInk2-Rot↓ |
|---|---|---|---|---|---|---|
| Annealed Sampling(SPIDER) | 0.25 | 0.08 | 0.40 | 0.72 | 0.08 | 0.32 |
| + Warmup | 0.66 | 0.06 | **0.28** | 0.77 | 0.06 | 0.25 |
| + Perturbation | 0.67 | 0.06 | 0.30 | 0.79 | **0.03** | **0.14** |
| + Transition Reward | **0.71** | **0.05** | **0.28** | **0.81** | **0.03** | 0.15 |

在噪声大的 in-the-wild 重建参考上,成功率从 25% 提到 71%,**Warmup 是主要贡献者**(单项即拉到 66%);在干净的 OakInk2 上从 72% 提到 81%,说明该 retargeting 虽为不完美参考设计,对干净 MoCap 也有增益,并能扩到 1000+ 双手任务。

### 3.3 真机部署与数据过滤实证

管线共产出 **500 段人工核验过的高质量灵巧操作轨迹**,来源为互联网 53%、第一人称 31%、生成视频 16%;挑 10 个不同抓取类别(书写三脚架、power、ventral、平行延展抓取等)的动作在真机双臂上回放成功(whisking、pouring、dusting 等)。

**人类数据过滤 playbook**(在 100DOH 上的实证):从 100DOH 采 2000 段 10 秒 clip(已按手-物交互过滤过),只有 187 段(9%)有实际有意义的手-物交互;其中 41 段手/物越界、29 段无活动或活动跨镜头边界、14 段因相机运动失败、10 段因 SAM 3D 失败、10 段因其他原因丢失;最终只有 **83 段(4%)** 通过重建质量检查,最好情形约 107 段(5%)真正与灵巧操作学习相关。结论:不做恰当预处理/过滤地喂互联网视频,相当于 **约 20× 的数据浪费**。

## 四、局限性

- **假设刚体 + 半准确 metric depth**:两条假设任一不成立就可能失败;单目还带真实手-物距离的歧义,难以区分真实物理接触与单纯遮挡。
- **只重建手与物、不重建场景**:无法推理障碍、铰接等环境约束;而人类意图不只体现在手-物交互、也体现在手-场景交互,缺场景级推理即便有完美参考也受限。
- **仿真只近似真实动力学**:当前物理仿真器对真实世界建模仅近似,给可达到的真机性能设了上界。
- 每段视频还需一次离线点跟踪(为自适应引导)、重建仅逐段独立,尚未做长时/多段拼接。

## 五、评价与展望

**优点**。(1) 把 SAM 3D 这一**单图生 3D 模型免训练改造成视频 6-DoF 跟踪器**是很漂亮的一招——利用 flow matching 的"目标插值物混合"在推理时注入时序先验,不需重训、且形状/位姿共享潜空间的观察用得很巧;逐帧的采样-聚类替代 log 密度排序,是把生成模型当跟踪器时"如何选样本"的一个实用范式,30× 提速有工程价值。(2) retargeting 的三项创新(尤其 warmup)都刻意**不引入抓取启发式**,把"从无法恢复的初态里救回来"交给优化器本身,契合面向噪声参考的目标,25%→71% 的跃升说明抓取初始化确实是野外 retargeting 的主瓶颈。(3) 100DOH 上"5% 可用、20× 浪费"的实证很有价值,给"堆互联网视频"的乐观叙事泼了必要的冷水。

**局限与开放问题**。(1) 全流程串联多个基础模型(SAM 3 / MoGe / HaWoR / SAM 3D / BootsTAPIR / GeoCalib),误差会级联,论文对端到端失败率的刻画主要落在数据过滤统计上,单模块出错如何传播缺少定量分解。(2) 500 段数据、10 个真机动作的规模,离"让人类视频成为机器人学习一等公民"的目标尚远,且真机只做了**开环轨迹回放**、未训练闭环策略,故 downstream policy 收益还是空白——这是与 DexMimicGen、EgoDex/EgoZero、Being-H0 等"人类视频预训练"路线对比时最需要补的一环。(3) 与同期工作的定位:相较 DexMan(TRELLIS+FPose)、DexImit(SAM 3D+FPose++)、H2Sim2Robot(LiDAR 扫描)、VideoManip,本文的差异化在于**同时吃 self/generated/ego/internet 四类来源**且不依赖 depth 硬件或类目先验(Table 1),但代价是对基础模型质量高度敏感。(4) retargeting 建在 SPIDER 之上,采样式优化每段几分钟,规模化到百万级 clip 的算力成本未讨论。

**可能的改进方向**:引入场景级重建(缓解无法推理障碍/铰接的局限)、把逐段重建升级为跨段/长时一致跟踪、以及最关键的——用这批数据实际训练闭环灵巧策略并给出 policy 层面的 scaling 曲线,才能验证"观察数据转经验数据"这条路的真实斜率。此外单目 metric 尺度歧义可考虑接入多视角或生成式深度先验来收紧。

## 参考

1. C. Pan et al. *SPIDER: Scalable Physics-Informed Dexterous Retargeting.* arXiv:2511.09484, 2026.(本文 retargeting 的基础框架)
2. SAM 3D Team. *SAM 3D: 3Dfy Anything in Images.* arXiv:2511.16624, 2025.(被改造成视频跟踪器的图生 3D 模型)
3. J. Zhang et al. *HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos.* CVPR 2025.(手跟踪骨干)
4. R. Wang et al. *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details.* arXiv:2507.02546, 2025.(metric depth/pointmap)
5. B. Wen et al. *FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects.* 2024.(主要物体跟踪基线)
