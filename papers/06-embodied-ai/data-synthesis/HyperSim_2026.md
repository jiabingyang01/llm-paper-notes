# HyperSim：面向鲁棒机器人操作的整体式 Sim-to-Real 框架

> **论文**：*HyperSim: A Holistic Sim-To-Real Framework For Robust Robotic Manipulation*
>
> **作者**：Junyi Dong, Haotian Luo（共同一作）, Ziwei Xu, Shengwei Bian, Heng Zhang, Sitong Mao, Yao Mu, Ping Luo, Shunbo Zhou, Xiaodong Wu et al.
>
> **机构**：CloudRobo Lab, Huawei Cloud Computing Technologies Co., Ltd.（华为云）；上海交通大学；香港大学
>
> **发布时间**：2026 年 05 月（arXiv 2605.26638）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.26638) | [PDF](https://arxiv.org/pdf/2605.26638)
>
> **分类标签**：`sim-to-real` `3D Gaussian Splatting` `对抗轨迹` `sim-real 协同训练` `数据合成`

---

## 一句话总结

HyperSim 把「高保真环境合成 + 对抗式轨迹生成 + 仿真-真实协同训练」三根支柱拼成一个从数据生成到真机部署的闭环框架，仅需一次环境扫描和几十条演示，就在 Galaxea R1 深料箱分拣任务上让 π0 达到 75% 零样本、95% 少样本成功率，并在外部物理扰动下把首次成功率从 25% 提升到 60%。

## 一、问题与动机

端到端操作策略的训练依赖海量「观测-动作」配对轨迹,而真机采集受制于硬件成本和人力。合成数据是绕开数据瓶颈的自然选择,但作者指出现有仿真数据管线存在三条硬伤:

1. **环境过度简化**:典型仿真只是「悬浮桌面 + 空背景」,无法反映真实世界的杂乱与非结构化;
2. **数据多样性受限**:启发式场景配置(如固定物体朝向来人为降低难度)+ 只保留成功轨迹,严重压缩了 state-action 流形的覆盖;
3. **视觉与动力学失配**:sim 与 real 之间持续存在的域差(domain gap)直接拖垮真机表现。

作者的核心洞察是:真机泛化的脆弱性来自**环境复杂度、数据分布、视觉与物理差异**这几类因素的叠加,因此不应逐项孤立处理,而应在一个整体框架内协同解决。这正是 HyperSim 命名中 "Holistic"(整体式)的由来。

## 二、核心方法

HyperSim 采用**两层架构**:base layer 是标准的「场景生成→轨迹生成→训练→部署」数据管线;enhancement layer 是可插拔的增强模块,承载三项高级 sim-to-real 技术。三个模块分别对准三类域差:视觉保真、数据覆盖、跨域表征。

### 2.1 高保真环境合成(对准视觉域差)

采用**前景/背景解耦的混合合成**策略。

**前景(可交互操作区)**:摒弃纯随机场景生成,改用**约束感知的布局生成器**。作者设计了一个含 **18 个 solver** 的空间关系约束库,分三组:unary geometric priors(scale、orientation range)、explicit pairwise relations(place_on_surface / place_left_edge / place_behind 等 13 种)、implicit multi-object formations(random_placement、no_overlapping、with_obstacles)。任务需求被翻译成约束子集来生成合法布局,再用资产库或 text-to-3D 模型填充高保真、可物理交互的 3D 资产。

**背景**:采集同步的多模态数据(RGB、LiDAR、IMU),用 **GPGS**(几何先验 3DGS,作者团队自家 IROS 2025 工作)重建成 Gaussian 基元。每个 3D 高斯由位置 $p_k$、协方差 $\Sigma$、不透明度 $o_k$、球谐系数 $c_k$ 参数化:

$$\mathcal{G}(p) = \exp\left(-\frac{1}{2}(p-p_k)^{T}\Sigma^{-1}(p-p_k)\right)$$

体渲染时投影为 2D 高斯 $\mathcal{G}^{2D}$,深度排序后 alpha 混合得到像素颜色:

$$c(x) = \sum_{k=1}^{K} c_k o_k \mathcal{G}^{2D}(x)_k \prod_{j=1}^{k-1}\left(1 - o_j \mathcal{G}^{2D}(x)\right)$$

**用大白话说**:第一式就是把每个高斯团当成一个空间中的「模糊椭球」,离中心 $p_k$ 越远越透明;第二式是把一条视线上的所有椭球按前后顺序叠色,前面的椭球挡住后面的(那个连乘项就是「前面还剩多少光透过来」)。关键创新不在这两式(它们是标准 3DGS),而在于**用 LiDAR 融合的几何先验约束高斯与真实表面对齐**,从而解决弱纹理/复杂几何区域纯图像重建的尺度模糊问题。

重建后再用 **TSDF** 融合渲染的彩色图与深度图,生成一张与高斯表示结构对齐的**彩色 mesh**。这套「高斯-网格」混合表示天然兼容物理仿真器:高斯负责逼真渲染,严格对齐的 mesh 作为碰撞与接触动力学的后端。

### 2.2 对抗式轨迹生成(对准数据覆盖)

**分段轨迹生成**:把操作任务拆成一串物体中心(object-centric)子任务。定义物体坐标系 $\mathcal{F}_A$(原点 $O_A$ 在目标质心)和工具坐标系 $\mathcal{F}_T$(原点 $O_T$ 在 TCP)。引入 **bottleneck pose**(瓶颈位姿)概念:即 TCP 进入以 $O_A$ 为心、半径 $d$ 的小半球时的构型。它把每个子任务切成两段——从初始态到瓶颈位姿的 **approaching primitive**,和随后处理接触与操作的 **interaction primitive**。各段用运动规划器、逆运动学求解器、夹爪控制器分段合成。

**对抗扰动与恢复**:在夹爪到达瓶颈位姿时,**突然对目标状态(平移 + 旋转)注入扰动**,迫使运动规划器动态计算通往「新瓶颈位姿」的恢复轨迹,end-effector 必须不断重定位、重定向。这个「扰动-恢复」循环有双重收益:① 大幅拓宽目标状态分布的空间覆盖;② 在恢复阶段让外部传感器(尤其腕部相机)见到高度多样的视角。为平衡多样性与轨迹稳定性,每条轨迹**最多注入 3 次干预**。

**用大白话说**:普通合成数据只会「一次到位」抓取,策略容易死记初始状态、不看实时画面。这里故意在快抓到时把物体「挪一下/转一下」,逼着机器人临时改路线去追,于是数据里就包含了大量「看到偏差→纠正」的闭环视觉-运动对齐样本,策略因此学会实时用视觉反馈纠错。

### 2.3 仿真-真实协同训练(对准跨域表征)

不做显式的 sim/real 匹配,而是把仿真数据 $\mathcal{D}_s$ 与真实数据 $\mathcal{D}_r$($|\mathcal{D}_s| \gg |\mathcal{D}_r|$)当作可一起采样的通用数据,用行为克隆损失联合优化:

$$\mathcal{L}_{\mathcal{D}_\alpha} = \alpha\,\mathcal{L}_{\mathcal{D}_s} + (1-\alpha)\,\mathcal{L}_{\mathcal{D}_r}$$

其中 $\alpha \in [0,1]$ 是协同训练比例(从 $\mathcal{D}_s$ 抽样的概率)。$\alpha = 1$ 即纯仿真训练,对应零样本部署;$\alpha$ 略小于 1 则是少样本策略(掺入少量真机数据)。

**用大白话说**:不去逐参数对齐 sim 和 real,而是把两边数据混在一起喂,让网络自己被迫学出「跨域不变」的特征——仿真提供海量覆盖,少量真机数据提供物理接地(physical grounding)。

## 三、实验结果

**设置**:Galaxea R1 人形机器人,头部 + 腕部 RGB 相机 10 Hz。任务模拟工业分拣——把目标物(如红色插头)从中央**深料箱**转移到两侧料箱之一;深箱抓取存在严格运动学约束和撞壁风险,远难于平面操作。对抗扰动:目标 2D 位置每维在 $[0.02, 0.2]\,\text{m}$ 均匀采样,朝向在 $[-180°, 180°]$ 全范围采样。固定 20 条评测 trial(含视觉干扰物),两策略(ACT / π0)共 **400+ 次真机试验**。仿真引擎为 O3DE(原生支持 3DGS 与 ROS2,亦兼容 IsaacSim/Sapien)。少样本用 **35 条**人工演示。

**三种细粒度指标**:TAR(目标对齐率,EE 从初始态成功到达瓶颈位姿)、SR1(首次尝试成功率,单次连续完成不重试)、SR3(总体成功率,3 次尝试内完成)。

### 零样本迁移(Table I)

数据集消融 BaseSim → ADSim(加对抗)→ 3DGS-ADSim(再加高保真渲染,即 HyperSim 的零样本配置):

| 训练数据 | 策略 | TAR | SR1 | SR3 |
|---|---|---|---|---|
| BaseSim | ACT | 10% | 5% | 5% |
| ADSim | ACT | 45% | 10% | 15% |
| 3DGS-ADSim | ACT | 55% | 20% | 25% |
| BaseSim | π0 | 45% | 45% | 55% |
| ADSim | π0 | 75% | 60% | 70% |
| **3DGS-ADSim** | **π0** | **80%** | **60%** | **75%** |

要点:① 对抗机制(ADSim)显著抬升 TAR(π0:45%→75%),说明策略学会了闭环视觉-运动对齐;② 加 3DGS 渲染再涨最多 10%(各指标),验证高保真渲染缩小视觉域差;③ 同等条件下 **π0 全面碾压 ACT 达 25–55%**,凸显大规模预训练基座与高质量合成数据的协同。

### 少样本协同训练(Table II)

掺入 35 条真机演示后:

| 训练数据 | 策略 | TAR | SR1 | SR3 |
|---|---|---|---|---|
| ADSim | ACT | 45% | 10% | 15% |
| Real35&ADSim | ACT | 85% | 65% | 75% |
| Real35&3DGS-ADSim | ACT | 85% | 65% | **80%** |
| ADSim | π0 | 75% | 60% | 70% |
| Real35&ADSim | π0 | 90% | 65% | 85% |
| **Real35&3DGS-ADSim** | **π0** | **95%** | **75%** | **95%** |

对低容量的 ACT,协同训练靠真机数据补足接触动力学的物理接地,SR1/SR3 相对零样本齐涨 35%+;对已有强零样本能力的 π0,协同训练把各指标再抬 15%+,SR3 达 95%。且协同训练策略普遍**超过纯真机基线**(Table III:Real35 下 ACT SR3 60%、π0 SR3 70%),说明合成数据有效增广了有限演示。

### 动态鲁棒性(Table IV)

在线推理时由人手突然改变目标状态,考察即时应对:

| 训练数据 | 策略 | TAR | SR1 |
|---|---|---|---|
| Real35&BaseSim | π0 | 30% | 25% |
| Real35&ADSim | π0 | 80% | 60% |
| Real35&3DGS-ADSim | π0 | 80% | 60% |

用对抗轨迹训练的策略在扰动下 SR1 达 **60%**,而非对抗对照(Real35&BaseSim)仅 25%——**35 个百分点**的鲁棒性提升直接来自对抗机制把动态不确定性写进了训练数据。

### 数据缩放(附录 Table VI)

固定 400 条仿真轨迹、变化真机演示数(Real10/20/35),协同训练策略始终优于同等真机量的纯真机基线,如 Real35&ADSim(π0)SR3 85% vs Real35 纯真机 70%,印证合成数据作为互补源可防止过拟合到少量真机演示。

## 四、局限性

- **单任务、单体**:受深箱操作的硬件安全约束与真机评测成本所限,验证仅覆盖一个工业分拣任务和单一人形本体(Galaxea R1),泛化广度未证;
- **评测规模偏小**:核心结论建立在 20 条固定 trial 上,20 次试验下 5% 的粒度使单点数字方差较大(如 SR1 从 60% 到 75% 仅差 3 次);
- **对自家组件依赖**:背景重建的 GPGS、瓶颈位姿/分段思路均沿用已有工作,方法层面的新增主要是「瓶颈位姿处注入对抗扰动」这一巧思,单项模块的原创性有限;
- **对抗扰动是几何/运动学层面**:仅扰动目标位姿并重规划,未触及光照、纹理、动力学系数等其他域差维度;干预上限硬编码为 3 次,缺乏自适应准则;
- **协同比例 $\alpha$ 未系统扫描**:论文只区分 $\alpha=1$ 与「略小于 1」,未给出 $\alpha$ 的敏感性分析;
- 数据集与管线承诺开源但尚未释放。

## 五、评价与展望

**优点**:这是一篇「系统整合」味道很浓的工程论文,价值不在单点新奇而在把三条正交的 sim-to-real 手段(视觉保真 / 数据覆盖 / 表征对齐)用统一框架串起来并给出干净的消融证据链——BaseSim→ADSim→3DGS-ADSim→+Real35 的阶梯式递进把「每个模块各贡献多少」拆得很清楚,这在 sim-to-real 文献里是稀缺的。三条细粒度指标(TAR/SR1/SR3)也比单一成功率更能刻画失败模式,值得借鉴。前景 mesh + 背景高斯的混合表示是务实之选:既拿到 3DGS 的照片级渲染,又保留网格做接触动力学后端,回避了纯 3DGS「几何噪声不利接触仿真」的老问题。

**与公开工作的关系**:轨迹生成上属于 RoboTwin/RoboGen/InternData-A1 一脉的 piece-wise primitive 分解,而非 MimicGen 式的演示增广;瓶颈位姿沿袭 Johns 等的 coarse-to-fine 模仿学习;协同训练直接采用 RSS 2025 的 sim-and-real co-training 配方(Maddukuri 等)。真正的差异化是**在瓶颈位姿注入对抗扰动诱导恢复行为**,把「主动纠错」样本注入数据分布——这与「用 3DGS 生成新演示做单样本鲁棒操作」(Yang 等)是互补而非重叠的思路。

**开放问题与改进方向**:① 对抗扰动可从纯几何扩展到视觉/动力学联合扰动,并把「注入几次、注入多大」做成基于策略不确定性的自适应课程;② 20 trial 的评测应扩到多任务、多本体以支撑「数据保真与覆盖正相关于迁移」这一强结论;③ 混合高斯-网格表示对动态物体、透明/反光物体的重建鲁棒性待验证;④ 协同比例 $\alpha$ 与真机数据量之间存在明显的数据配比问题,值得一个更系统的 scaling law 式研究。总体上,HyperSim 更像一份「如何把现有 sim-to-real 组件正确拼装并逐项验证」的高质量实证报告,对工业落地参考价值高于方法学新意。

## 参考

1. Maddukuri et al., *Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation*, RSS 2025 —— 本文协同训练损失的直接来源。
2. Chen et al., *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization*, arXiv:2506.18088 —— piece-wise primitive 轨迹生成与域随机化基线。
3. Tian et al., *InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy*, arXiv:2511.16651 —— 同期高保真合成数据预训练工作。
4. Xu et al., *GPGS: Geometric Priors for 3D Gaussian Splatting in Structural Environments*, IROS 2025 —— 本文背景重建所用的几何先验 3DGS。
5. Yang et al., *Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation*, arXiv:2504.13175 —— 用 3DGS 做操作数据生成的相关思路。
