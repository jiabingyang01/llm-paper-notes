# ComSim：基于组合式仿真的可扩展真机机器人数据生成

> **论文**：*Building Scalable Real-World Robot Data Generation via Compositional Simulation*
>
> **作者**：Yiran Qin, Jiahua Ma, Li Kang, Wenzhan Li, Yihang Jiao, Xin Wen, Xiufeng Song, Heng Zhou, Jiwen Yu, Zhenfei Yin, Xihui Liu, Philip Torr, Yilun Du, Ruimao Zhang（通讯）et al.
>
> **机构**：CUHK-Shenzhen（香港中文大学深圳）、Sun Yat-sen University（中山大学）、Shanghai Jiao Tong University（上海交通大学）、USTC（中国科学技术大学）、The University of Hong Kong（香港大学）、University of Oxford（牛津大学）、Harvard University（哈佛大学）
>
> **发布时间**：2026 年 04 月（arXiv 2604.11386）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.11386) | [PDF](https://arxiv.org/pdf/2604.11386)
>
> **分类标签**：`组合式仿真` `sim2real` `神经仿真器` `视频生成数据增强` `具身操作`

---

## 一句话总结

ComSim 提出"经典仿真 + 神经仿真"的**组合式仿真(Compositional Simulation)** 范式：先在经典仿真器里大规模生成动作精确对齐的轨迹-视频,再用一个 real-sim-real 训练出来的**神经仿真器** 把仿真画面"翻译"成真实风格的伪真机数据(Pseudo Real Data),从而以极少真机演示(10 条)撬动大规模、覆盖布局/物体/背景变化的训练集;在真机 Diffusion Policy 上,10 真 + 200 伪真机的配比把 Shake Bottle 域内成功率从 20 真的 17/30 提到 28/30、OOD 从 1/30 提到 12/30,新物体泛化上 Move Playing-Card Away 从 2/30 提到 21/30。

## 一、问题与动机

具身操作的数据瓶颈来自三类数据源各有短板(见原文 Fig.1):

- **真人采集**:质量高,但**无法规模化**、难以覆盖真实世界的多样分布,成本高。
- **经典仿真器**(MuJoCo / Isaac Lab / SAPIEN 系):可无限量生成、动作-视频天然精确对齐(omniscient view),但**外观与物理存在 sim2real gap**,直接联合训练常反而拖垮真机性能。
- **神经仿真器**(基于视频生成模型):画面接近真实、缓解外观物理差距,但存在幻觉、3D 一致性差、**动作条件控制不准**——生成的动作与视频对不上,数据无法用于策略训练。

作者的洞察:上述三者在"可扩展性 / 外观 / 物理 / 动作一致性"四个维度上互补。经典仿真的**强项恰是动作一致性**(动作与画面严格配对),而**弱项是外观真实度**;神经仿真反之。于是把两者**组合**:用经典仿真保证动作精确、用神经仿真补齐真实外观,得到既动作精确又画面逼真的动作-视频对。

## 二、核心方法

整体是一条 **real-sim-real 闭环数据增强流水线**,分两阶段。

### 2.1 目标形式化

记经典仿真数据 $\mathcal{D}_{\text{sim}}=\{(v_i,a_i)\}_{i=1}^{N}$、真机数据 $\mathcal{D}_{\text{real}}=\{(v'_j,a'_j)\}_{j=1}^{M}$,目标是学一个神经仿真函数 $\mathcal{N}(\cdot)$:

$$\mathcal{N}(\mathcal{D}_{\text{sim}}) \approx \mathcal{D}_{\text{real}}$$

约束是映射后动作保持不变,即 $a_i \approx a'_j$,同时保证 3D 场景一致与视频质量。

用大白话说:把整批"仿真视频 + 动作"喂进 $\mathcal{N}$,输出一批"看起来像真机拍的视频 + 原封不动的动作",既换了皮又没改动作标签。

### 2.2 Real2Sim 数据采集(训练神经仿真器的配对数据)

要训 $\mathcal{N}$,需要三元组 $\langle \mathcal{V}_{\text{sim}}, \mathcal{V}_{\text{real}}, \mathcal{A} \rangle$——同一段动作序列 $\mathcal{A}$ 分别在经典仿真器和真机执行,得到严格配对的仿真视频 $\mathcal{V}_{\text{sim}}$ 与真机视频 $\mathcal{V}_{\text{real}}$。为此搭一个与真机采集平台严格对齐的**数字孪生仿真环境**,做三级对齐:

1. **背景与物体对齐**:仿真里的桌面/背景颜色、资产尺寸与真机一致(数字孪生 + 真实尺度匹配);
2. **相机标定与对齐**:仿真相机内外参与真机一致;
3. **物体位置对齐**:任务初始化时把真机场景中物体位置严格搬进仿真器。

做法上:先采少量真机轨迹(视频 $\mathcal{V}_{\text{real}}$ + 动作 $\mathcal{A}$),再把这些真机动作在数字孪生仿真器里**回放**,渲染出配对的 $\mathcal{V}_{\text{sim}}$。共采 **10 个任务、200 对**数据用于训练。

### 2.3 Sim2Real 组合式动态视频生成(神经仿真器)

神经仿真器用 **Compositional Dynamic Video Generation** 训练,骨干是 DiT。它同时以两类"动态"作为条件:

- **Control Dynamics(动作)**:决定机械臂怎么动;
- **Visual Dynamics(仿真观测)**:决定场景外观、视角、物体位置。

DiT 在扩散去噪时分别估计两路条件得分 $\epsilon_a(x_t\mid a_t)$(动作条件)与 $\epsilon_v(x_t\mid v_t)$(视觉条件),并在采样阶段把两路得分**组合(compose)** 起来,做 Dynamic Guidance。其组合式引导的一般形式(论文以 Fig.2 图示"Composing Dynamic scores",未给显式权重系数,此处为其 compositional diffusion 机制的通用刻画)可写作:

$$\tilde{\epsilon}(x_t)=\epsilon_\varnothing+w_a\big(\epsilon_a(x_t\mid a_t)-\epsilon_\varnothing\big)+w_v\big(\epsilon_v(x_t\mid v_t)-\epsilon_\varnothing\big)$$

用大白话说:去噪每一步同时"听两个指挥"——动作分支管"这一帧手该动到哪、夹爪开合时机",视觉分支管"画面里桌子物体长啥样、在哪";把两路预测拼起来,既保证画面真实又保证动作对得上,这正是把 compositional diffusion 的分数组合思想用到 sim2real 视频风格迁移上(作者之一 Yilun Du 是组合式生成模型代表工作作者)。

因为 $\mathcal{V}_{\text{sim}}$ 与 $\mathcal{V}_{\text{real}}$ 的动作已严格对齐,训练只需最小化视频差异:

$$\mathcal{L}_{\text{sim2real}}=\mathcal{L}_{\text{video}}\big(f_\mathcal{N}(\mathcal{V}_{\text{sim}},\mathcal{A},\theta),\ \mathcal{V}_{\text{real}}\big)$$

其中 $\theta$ 是神经仿真器参数,$\mathcal{L}_{\text{video}}$ 度量生成视频与真机视频的差距。

### 2.4 规则驱动的大规模轨迹生成(经典仿真扩数据)

规模化阶段用 **RoboTwin(RoboTwin 2.0,基于 SAPIEN 的双臂操作仿真)** 作为经典仿真器,利用其丰富数字资产与多样轨迹分布。方法要点:

- 定义一套交互规则即 **action primitives**(原子:grasp / push / align;高阶:stack blocks);
- 用 **GPT-5** 合成由这些 primitive 组合的可执行代码,并引入**组合式约束(compositional constraints)** 保证语义正确与物理可行;
- 通过系统性改变环境条件、物体初始状态、agent 动作,构造覆盖动作空间的轨迹集合 $\tau_s$,遍历行为分布(不同物体初始化、异质物体类别);
- 得到时序同步、逐帧动作/状态严格对齐的仿真观测流 $v_s$,再过神经仿真器 $\mathcal{N}$ 精修视觉外观、保留动作一致性,产出 **Pseudo Real Data**。

### 2.5 真机部署与联合训练

把大批 $(\mathcal{V}_{\text{sim}},\mathcal{A})$ 经 $\mathcal{N}$ 变成 $(\mathcal{V}_{\text{real}},\mathcal{A})$ 即 Pseudo Real Data(比原始仿真数据显著更接近真实、domain gap 更小),再与少量真机数据联合训练策略(Algorithm 1):

$$D_{\text{combined}}=\alpha\cdot D_{\text{real}}+(1-\alpha)\cdot P_{\text{pseudo}}$$

其中 $\alpha$ 为真机数据配比。用大白话说:伪真机数据管"广"(覆盖各种布局/物体/背景),少量真机数据管"准"(锚定真实分布),两者混合训练兼得性能与泛化。

## 三、实验结果

### 3.1 神经仿真器的 sim2real 视频质量(消融)

8 个任务(Shake Bottle、Stack Blocks Two、Move Playing-Card Away 及其 Cluttered/Colored Background 变体、Place Mouse Pad 变体、Handover Bottle 等)。对比:Sim(原始经典仿真)、Baseline(Stable Diffusion 1.5 视频到视频 + FRESCO 时序后处理)、Zero-Shot(骨干不做 sim2real 微调)、Ours-CD(仅动作条件)、Ours-VD(仅视觉条件)、Ours-Full(动作+视觉联合)。

| 方法 | PSNR ↑ | SSIM ↑ | CLIP ↑ | LPIPS ↓ | FID ↓ | FVD ↓ |
|---|---|---|---|---|---|---|
| Sim | 16.443 | 0.4342 | 0.7564 | 0.3629 | 187.40 | 61.048 |
| Baseline | 16.849 | 0.5129 | 0.7526 | 0.3494 | 254.59 | 50.369 |
| Zero-Shot | 13.093 | 0.5487 | 0.7308 | 0.4756 | 219.74 | 163.83 |
| Ours-CD | 8.464 | 0.1486 | 0.7216 | 0.8130 | 434.44 | 239.13 |
| Ours-VD | 18.153 | 0.5916 | 0.7884 | 0.2813 | 153.12 | 22.311 |
| **Ours-Full** | **19.577** | **0.6484** | **0.8102** | **0.2647** | **147.90** | **15.765** |

关键结论:**仅动作条件(Ours-CD)效果崩塌**(PSNR 8.46、FVD 239,甚至差于原始仿真),说明在其设定下**视觉动态占主导**——场景结构主要由仿真画面提供,仅凭动作条件会让视角/物体/背景完全由模型先验瞎编。Ours-Full 在全部 6 项指标上最优(FVD 15.765 vs Sim 61.048),证明动作+视觉联合引导才能同时保证高感知真实度与动作一致性。

### 3.2 真机执行(Diffusion Policy,每格 n/30 次成功)

六种数据配比:10 Real、20 Real、200 Sim Pretrain + 10 Real、10 Real + 200 Sim(联合)、**10 Real + 200 Pseudo Real(本文)**、200 Pseudo Real(零真机,策略上界)。下表节选代表任务(In Domain = 域内,OOD = 分布外空间):

| 任务 / 分布 | 20 Real | 10R+200 Sim | **10R+200 Pseudo(本文)** | 200 Pseudo(ZS) |
|---|---|---|---|---|
| Shake Bottle · In | 17/30 | 6/30 | **28/30** | 10/30 |
| Shake Bottle · OOD | 1/30 | 0/30 | **12/30** | 5/30 |
| Move Playing-Card Away · In | 24/30 | 6/30 | **29/30** | 18/30 |
| Move Playing-Card Away · OOD | 2/30 | 1/30 | **17/30** | 9/30 |
| Place Mouse Pad (Cluttered) · In | 15/30 | 7/30 | **19/30** | 8/30 |
| Place Mouse Pad (Cluttered) · OOD | 0/30 | 1/30 | **10/30** | 5/30 |
| Handover Bottle (Cluttered) · In | 8/30 | 1/30 | **13/30** | 11/30 |
| Handover Bottle (Cluttered) · OOD | 0/30 | 0/30 | **5/30** | 3/30 |

要点:(1)只有 10 真机时 DP 很差,增真机(20 真)稳步提升,说明真机经验必要;(2)加经典仿真数据(200 Sim Pretrain / 10R+200 Sim)受制于 sim2real gap,增益有限甚至更差;(3)**伪真机数据显著缩小 gap**,10R+200 Pseudo 在几乎所有任务的域内/OOD 上都最好;(4)**纯 200 Pseudo(零真机)** 也能拿到非平凡成功率(如 Move Playing-Card Away OOD 9/30),说明伪真机数据本身质量高。

### 3.3 泛化(新空间分布 / 新物体)

- **新空间分布**:真机演示中物体初始位置被限制在一个预设小区域,推理时把物体挪到未见区域。纯真机训练的 DP **几乎零泛化**;RoboTwin 仿真自带的空间多样性也因 sim2real gap 失效、无可测提升;伪真机数据在各挪动配置上一致提升。
- **新物体(Table 3)**:真机演示用 Fanta 瓶 + 蓝色扑克牌,推理时换成 Coca-Cola / Sprite / 农夫山泉东方树叶等瓶子 + 红色扑克牌:

| 任务 | 10 Real | 20 Real | 200 Sim Pre+10R | 10R+200 Sim | **10R+200 Pseudo** | 200 Pseudo(ZS) |
|---|---|---|---|---|---|---|
| Shake Bottle | 0/30 | 0/30 | 0/30 | 0/30 | **15/30** | 9/30 |
| Move Playing-Card Away | 1/30 | 2/30 | 1/30 | 0/30 | **21/30** | 11/30 |

经典仿真数据对新物体泛化**零增益**,伪真机数据带来清晰提升,表明该流水线保留了真实世界属性并支持向未见物体迁移。

## 四、局限性

- **仅桌面操作**:实验限于 tabletop manipulation,尚未验证移动操作/轮式机器人等更复杂本体(作者列为 future work)。
- **依赖数字孪生的严格对齐**:Real2Sim 阶段要求背景/物体/相机/物位三级对齐,搭建数字孪生成本不低,换场景/换本体需重做对齐与配对采集(10 任务 × 200 对)。
- **视觉动态占主导带来的隐忧**:消融显示场景结构基本由仿真画面决定,神经仿真器更像"高质量风格迁移 + 局部精修"而非真正从动作重建物理;若经典仿真的物体运动/接触本身失真,伪真机数据会继承该误差。
- **动作条件的实际贡献偏弱**:Ours-CD 崩塌、Ours-VD 已接近 Ours-Full,说明"动作一致性"更多靠"回放仿真已对齐的画面"保证,而非视频模型对动作的强控制;论文也承认视频生成的动作控制(开合时机、运动模糊匹配)仍有非平凡瑕疵。
- **规模化仅到 200 伪真机**:相较真正"scalable"标题,实验规模(每任务 200 条量级)偏小,未展示上千/上万条时的 scaling 曲线。
- **依赖 GPT-5 生成 primitive 组合代码**:轨迹多样性上限受 LLM 代码合成与组合约束设计影响,可复现性与稳定性未充分评估。

## 五、评价与展望

**优点**:(1)问题拆解干净——把"可扩展/外观/物理/动作一致性"四维显式列出,并论证经典仿真与神经仿真在这四维上互补,是全文最有说服力的动机框架;(2)**组合式思路一石二鸟**:既指扩散采样中动作分数与视觉分数的 score composition(compositional diffusion),又指轨迹层面用 primitive + LLM 组合约束扩数据,命名自洽;(3)真机实验设计扎实,In-Domain / OOD / 新物体三条泛化轴 + 六配比对照,直接回答"伪真机数据到底有没有用",结论清楚(经典仿真联训无益、伪真机显著有效);(4)Ours-CD 的失败案例诚实地暴露了"视觉动态占主导"这一非平凡发现。

**缺点与开放问题**:(1)与既有真到仿到真(URDFormer、RialTo)、动作重定向合成(MimicGen、DemoGen)、以及世界模型/视频仿真(UniSim、RoboDreamer、IRASim、Cosmos)相比,ComSim 的新意主要在"经典仿真轨迹 + 神经风格迁移"的组合与 score composition 的具体用法,而非全新范式;论文缺少与 DemoGen / MimicGen 这类同样"少量真机撬动大规模"方法的直接头对头对比;(2)"视觉动态主导 + 动作条件弱"意味着物理正确性其实主要来自经典仿真器,神经仿真器承担的是外观域适配,离"神经仿真器学到真实物理"还有距离;(3)只在 DP 上验证,未见 VLA/流模型等更强策略上的结论,泛化到大模型策略是否仍成立存疑;(4)数字孪生对齐的人工成本与该流水线"可扩展"定位存在张力。

**可能改进方向**:引入 unpaired 数据放松数字孪生的严格配对要求(作者已提及);在采样组合中引入自适应权重 $w_a,w_v$ 而非固定组合,针对接触/夹爪开合等动作敏感时刻加大动作分支权重以修复其弱控制;补充 scaling law 分析(伪真机条数 vs 成功率)与对 sim 物理失真的敏感性分析;以及在更强策略骨干上复验伪真机数据的边际收益。

## 参考

1. Chen T. et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation.* arXiv:2506.18088 (2025) — 本文经典仿真器与数字资产来源。
2. Mandlekar A. et al. *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations.* CoRL (2023) — 少量真机演示合成大规模数据的代表工作。
3. Xue Z. et al. *DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning.* arXiv:2502.16932 (2025) — 合成演示扩数据的直接可比对象。
4. Yang M. et al. *Learning Interactive Real-World Simulators (UniSim).* arXiv:2310.06114 (2023) — 神经/视频世界仿真器路线。
5. Qin Y. et al. *Exploring Embodied Agent Collaboration with Compositional Constraints.* arXiv:2503.16408 (2025) — 本文轨迹生成所用组合式约束来源。
