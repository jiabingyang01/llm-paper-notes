# HAMSTER：面向开放世界机器人操作的分层动作模型

> **论文**：*HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation*
>
> **作者**：Yi Li*、Yuquan Deng*、Jesse Zhang*、Joel Jang、Marius Memmel、Raymond Yu、Caelan Garrett、Fabio Ramos、Dieter Fox、Anqi Li†、Abhishek Gupta†、Ankit Goyal† et al.（*共同一作，†共同通讯/指导）
>
> **机构**：NVIDIA、University of Washington、University of Southern California
>
> **发布时间**：2025 年 02 月（arXiv 2502.05485，v4 于 2025 年 5 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2502.05485) | [PDF](https://arxiv.org/pdf/2502.05485)
>
> **分类标签**：`分层VLA` `2D路径中间表征` `跨域数据利用` `模仿学习` `VLM微调`

---

## 一句话总结

HAMSTER 用一个在廉价、易得的离域数据（网络 VQA、仿真轨迹、异构机器人数据）上微调的高层 VLM 预测粗粒度 2D 末端执行器路径,再把该路径作为条件交给一个在少量本体真机数据上训练的低层 3D 策略(RVT-2 / 3D Diffuser Actor)执行,在 7 个泛化维度、74 个真实任务(222 次评测)上相对 OpenVLA 平均成功率提升约 20 个百分点(相对提升约 50%),相对非 VLM 的 3D 策略提升约 3 倍。

## 一、问题与动机

大型基础模型在视觉与语言领域已展现出很强的开放世界泛化能力,但机器人操作策略远未达到同等水平,根本瓶颈是**机器人本体数据昂贵**——高质量的观测-动作对通常只能通过遥操作在目标机器人上采集,即便近年社区级数据集(Open X-Embodiment、DROID 等)持续扩充,规模、质量、多样性仍远不及视觉/语言语料。

论文把现有两类路线的取舍摆在一起对比:

- **单体(monolithic)VLA**(如 RT-2、OpenVLA、$\pi_0$):直接微调预训练 VLM 输出机器人动作。依赖大规模本体数据,且推理频率受限于大模型前向速度(如 OpenVLA-7B 在 RTX 4090 上仅 6Hz),难以支持高频、灵巧操作。
- **较小的模仿学习策略**(如 RVT-2、3D Diffuser Actor):在 3D 感知输入上训练,具备不错的灵巧性和局部鲁棒性,但对场景/语义的大幅变化很脆弱,也难以有效利用与真实场景视觉差异巨大的仿真数据。

核心问题(Q1–Q5,对应实验设计):hierarchical VLA 能否 1)对显著视觉/语义变化的未见场景泛化;2)比单体架构获得更强的跨域泛化;3)促进非抓取(non-prehensile)与长时序任务的学习;4)展现更高的示教效率;5)因层级设计与 VLM 微调获得更强的视觉+语义推理能力。

核心洞察:与其让 VLM 直接输出动作,不如让它输出一个**中间表征**——只要该表征 1)可以从图像序列中廉价获取;2)基本与本体无关(embodiment-agnostic);3)对动力学的细微变化足够鲁棒——VLM 就可以完全用离域数据(无动作的视频、手绘草图、仿真数据)来微调,而不需要在部署场景中采集本体动作数据。

## 二、核心方法

HAMSTER(**H**ierarchical **A**ction **M**odels with **S**epara**T**Ed Path **R**epresentations)由两个互联模型组成。

### 2.1 中间表征:2D 路径

给定单目 RGB 图像 $\text{img}$ 与语言指令 $z$,高层 VLM 预测一条粗粒度 2D 路径

$$p = [(x_t, y_t, \texttt{gripper\_open}_t)]_t,\quad x_t, y_t \in [0,1]$$

其中 $(x_t,y_t)$ 是末端执行器(或人手)在图像平面上的归一化像素坐标,$\texttt{gripper\_open}_t$ 是夹爪开合的二值状态。**大白话说**:VLM 不再预测精确的机器人动作,只画一条"在哪张图上、手该往哪走、什么时候开合夹爪"的粗线条路线图,把"做什么、去哪"的语义决策与"具体怎么动"的精细控制分开。

这类路径可以从多种廉价来源自动/半自动获取:action-free 视频上的点跟踪(TAPIR/CoTracker)、物理仿真的正向运动学投影、人类手绘草图、以及本体感知(proprioception)投影。

### 2.2 高层 VLM:离域数据微调

基座为 VILA-1.5-13B(130 亿参数,预训练于图文交错数据与视频描述)。微调目标是标准的负对数似然监督损失:

$$\mathbb{E}_{(\text{img}_i, z_i, \text{ans}_i)\sim \mathcal{D}_{\text{off}}}\big[\log \text{VLM}(\text{ans}_i \mid \text{img}_i, z_i)\big]$$

**大白话说**:就是普通的"看图回答"式微调,只是"回答"被约定成一串坐标点。离域数据集 $\mathcal{D}_{\text{off}}$ 由三类数据混合、等权重随机采样构成:

1. **像素点预测**(RoboPoint 数据集,77 万样本):形如"定位被标记的物体"的开放词汇物体定位 QA,答案为像素点/框列表,训练模型建立像素-物体语义关联;
2. **仿真机器人数据**(RLBench,约 32 万条):从 103 个任务中挑出 81 个前置摄像头可见性良好的任务,每任务 1000 条轨迹、约 4 条语言指令变体,ground-truth 路径通过正向运动学 + 相机参数投影得到;
3. **真实机器人数据**(Bridge 约 1 万条轨迹 + DROID 约 4.5 万条轨迹,约 2.2 万条唯一轨迹×2 相机视角):均**不来自部署测试环境**,路径由本体感知投影提取,DROID 中相机外参质量差的轨迹被过滤。

另外混入 66 万样本的 VQA 数据集(LLaVA 指令微调数据)联合训练,以保留 VLM 的世界知识和通用视觉问答能力,避免"窄化"为专用轨迹预测器。

由于原始路径可能长达上百步,训练前用 **Ramer-Douglas-Peucker(RDP)** 算法化简折线(容差 $\epsilon=0.05$),把短时程任务的路径压缩到约 2–5 个关键点,让 VLM 在"高层"尺度上推理,而不是逐帧输出。附录消融显示:对较小的 VILA1.5-3B 骨干,RDP 化简后的稀疏路径显著优于固定 20 点采样;骨干扩大到 13B 后两种表示都能很好工作,说明化简能帮模型把注意力集中在关键的抓/放点上。

训练细节:8×A100(单卡约 65GB 显存),约 30 小时,有效 batch size 256,学习率 $1\times10^{-5}$,包含视觉编码器在内的全参数微调。

### 2.3 低层策略:路径引导的 3D 控制

低层策略 $\pi_\theta(a \mid s, o, z, p)$ 额外接收 VLM 预测的路径 $p$、本体感知状态 $s$、以及 3D 感知观测 $o=(\text{img}, \text{pointcloud})$(即 RGB-D/点云,VLM 无法原生处理的模态)。训练目标同样是标准模仿学习最大似然:

$$\mathbb{E}_{(s_i,o_i,z_i,p_i,a_i)\sim\mathcal{D}_{\text{path}}}\big[\log \pi_\theta(a_i \mid s_i,o_i,z_i,p_i)\big]$$

路径条件化的实现方式是把带时间梯度色彩的折线**直接叠画在 RGB 图像**上(颜色由蓝到红代表时间推进,圆圈标出开/合夹爪的时刻),这样可以兼容任意通道数的策略骨干而无需改动输入接口。论文研究了两种低层骨干:**RVT-2**(Goyal et al., 2024)与 **3D Diffuser Actor / 3D-DA**(Ke et al., 2024)。对 RVT-2 还测试了 "Concat" 变体——把路径画在单独通道并与原图拼接成 6 通道输入(而非直接覆盖叠画),效果更好,因为 RVT-2 的虚拟重投影步骤会把叠画在图上的路径打碎,而独立通道能完整保留路径信息;3D-DA 因图像编码器要求 3 通道输入,不支持该变体。训练时对路径 $(x,y)$ 加入 $\mathcal{N}(0,0.01)$ 的小幅高斯噪声以提升对 VLM 预测误差的鲁棒性(夹爪开合标签不加噪声)。

真机低层策略训练数据总计 320 条遥操作演示(pick-and-place 220 条、knock-down 50 条、press-button 50 条),规模远小于典型单体 VLA 微调所需数据量。

### 2.4 推理与频率解耦

单体 VLA 每个动作步都要查询大模型(如 OpenVLA 只能到 6Hz),限制了灵巧/动态任务的可行性。HAMSTER 的高层 VLM 每个 episode 只需查询一到几次生成路径,随后低层策略沿路径连续执行多步,因此可以把高层骨干换成更大的 VLM 而不担心整体推理速度。

## 三、实验结果

**真实机器人总体评估**(Franka Panda,7 个泛化轴:Basic、Object-and-Goal、Visual、Language、Spatial、Novel Object、Multiple;共 74 个任务、222 次评测,对比 OpenVLA(同数据 LoRA 微调)、RVT-2、3D-DA):HAMSTER 相对 OpenVLA 平均成功率提升约 20 个百分点(相对提升约 50%),相对单体 VLA 整体领先超过 2 倍,相对未加路径引导的非 VLM 3D 策略领先超过 3 倍。

按任务类型细分成功率(Table 7):

| 任务类型 | RVT2 | 3DDA | OpenVLA | HAMSTER+RVT2 | HAMSTER+3DDA |
|---|---|---|---|---|---|
| pick and place | 0.28 | 0.19 | 0.46 | 0.79 | 0.78 |
| press button | 0.13 | 0.16 | 0.25 | 0.50 | 0.63 |
| knock down | 0.17 | 0.03 | 0.41 | 0.47 | 0.66 |

**RLBench 数据对单体 VLA 无效的对照实验**:用同一份用于训练 HAMSTER-VLM 的 RLBench 数据(81 任务×1000 episode,仅取前置摄像头可见性良好的样本,训练至 token 准确率 >90%)微调 OpenVLA,真机 6 个 Basic 任务上平均成功分数为 0.54,反而低于未用 RLBench 数据微调的 0.58——说明单体 VLA 因动作空间/观测空间与仿真数据不匹配,难以从这类廉价异构数据获益,而 HAMSTER 的层级解耦可以。

**Colosseum 仿真评估**(RLBench 之上的系统化视觉扰动基准,14/20 任务,5 个随机种子,vanilla 3D-DA vs HAMSTER+3D-DA):

| 指标 | 3D-DA | HAMSTER+3D-DA |
|---|---|---|
| 平均成功率(跨全部视觉变化) | 0.35 ± 0.04 | 0.46 ± 0.04 |

平均相对提升约 31%,涵盖背景纹理、相机位姿、干扰物、光照、物体颜色/尺寸、桌面纹理等变化。

**示教效率**(Colosseum 5 任务子集:SLIDE_BLOCK_TO_TARGET、PLACE_WINE_AT_RACK_LOCATION、INSERT_ONTO_SQUARE_PEG、STACK_CUPS、SETUP_CHESS):

| 方法 | 成功分数 |
|---|---|
| 3D-DA(100% 数据) | 0.18 ± 0.10 |
| HAMSTER+3D-DA(50% 数据) | 0.36 ± 0.04 |
| HAMSTER+3D-DA(100% 数据) | 0.43 ± 0.05 |

仅用一半数据,HAMSTER 就达到标准 3D-DA(满量数据)约 2 倍的成功分数。

**相机视角不变性**(新视角下 10 次试验,6 个训练物体/3 个训练容器):

| 方法 | 原相机 Success | 原相机 Complete | 新相机 Success | 新相机 Complete |
|---|---|---|---|---|
| OpenVLA | 0.60 | 0.30 | 0.23 | 0.00 |
| HAMSTER+RVT2 | 0.83 | 0.70 | 0.73 | 0.40 |
| HAMSTER+RVT2(Concat) | 1.00 | 1.00 | 0.98 | 0.90 |

**高层路径质量的人类排序评测**(Table 6,秩越低越好,1=最优):

| 方法 | 微调数据 | 平均秩(全部样本) |
|---|---|---|
| RT-Trajectory,GPT-4o 零样本 | — | 3.47 |
| RT-Trajectory,GPT-4o + Code-as-Policies | — | 3.41 |
| HAMSTER-VILA(去掉真实 RLBench 仿真数据) | 部分离域数据 | 2.13 |
| HAMSTER-VILA(完整) | 全部离域数据 | **1.40** |

说明微调开源 VLM(尤其纳入仿真路径数据)显著优于闭源 VLM 零样本生成路径的路线(如 RT-Trajectory)。

**通用视觉语言能力保留**(15 个 VQA/鲁棒性基准,对比基座 VILA1.5-13B 与 HAMSTER):核心 VQA 任务平均分基本持平(VILA 82.8 vs HAMSTER 82.9),鲁棒性/探针类基准也大体相当(如 MME 1569.6 vs 1588.4,MMB 74.9 vs 75.3),表明 HAMSTER 依然是通用 VLM 而非退化为窄域专用轨迹预测器。

**失败模式分析**(Figure 15):HAMSTER+RVT2 的失败中 72% 源于低层模型未能贴合预测路径执行(trajectory adherence),28% 属于执行失败;而 HAMSTER+3D-DA 恰好相反,仅 10% 是路径贴合失败,90% 是执行失败。论文推测这是因为 RVT-2 的虚拟重投影步骤会打碎叠画在图上的 2D 路径,而 3D-DA 的视觉塔直接处理原始 2D 图像、路径解读更简单。

## 四、局限性

论文在结论部分明确列出的局限:

1. **仅生成 2D 空间中的点,不做原生 3D 预测**——高层 VLM 并不具备真正的三维空间理解能力,深度/前后关系需要靠低层 3D 策略补足;
2. **2D 路径接口带宽有限**——只能传递位置轨迹与二值夹爪状态,无法表达力度、旋转等更丰富的动作语义;
3. 失败分析中还暴露:路径在任务开始时一次性生成,若执行过程中环境发生显著变化,模型缺乏动态调整或重新定位目标物体的能力;2D 投影固有的深度歧义(点在物体前方还是后方)会导致轨迹贴合失败。
4. 作者指出的未来方向:探索可学习的中间接口(而非手工设计的 2D 路径),以及直接从大规模人类视频数据训练此类层级 VLM。

## 五、评价与展望

**贡献与优点**:HAMSTER 把"用廉价离域数据扩展 VLA 泛化能力"这一命题,转化为一个具体、可操作的架构选择——用 2D 路径作为高层语义规划与低层精细控制之间的接口。相比同样使用轨迹式中间表征的 RT-Trajectory(Gu et al., 2023),其差异化贡献在于证明了**微调**开源 VLM 比依赖闭源 VLM(GPT-4o)零样本 / Code-as-Policies 生成轨迹更准确、更具跨域泛化性(Table 6 的人类排序差距明显);相比同样把轨迹预测作为辅助任务的单体 VLA LLARVA(Niu et al., 2024),HAMSTER 的层级解耦使低层策略可以接入 VLM 原生不支持的富 3D/本体感知输入。实验设计也较扎实:七个正交的真实世界泛化轴 + Colosseum 系统化视觉扰动基准 + 消融(RLBench 数据对单体 VLA 无效、路径表示的 RDP 消融、失败模式归因)相互印证了"层级解耦本身是收益来源"而非仅仅是骨干模型更强。

**局限与开放问题**:1)2D 路径本质上是位置+开合的低带宽通道,难以扩展到需要力控、旋转姿态、双臂协调的复杂操作,这也是作者自己承认的局限;2)高层路径在任务开始时一次性生成、执行期间不再更新,对动态/可推理干扰的场景(遮挡、物体移动)鲁棒性有限,是一个偏"开环"的层级设计,后续工作可以探索闭环重规划或周期性重新查询 VLM 的机制;3)当前的低层策略仍需要几十到几百条本体遥操作演示,示教效率虽优于纯单体 VLA/3D 策略,但离"零样本迁移到新机器人"仍有距离;4)路径-低层策略接口是手工设计的(叠画/拼接图像通道),论文自己也提出"可学习中间接口"作为更通用的方向,这与后续一些工作探索的隐式/连续 latent plan 表征(而非显式几何路径)是可比较的开放方向;5)失败分析显示不同低层骨干(RVT-2 vs 3D-DA)对同一路径接口的"贴合"难度差异很大,提示路径表示与具体策略架构的重投影/编码方式存在耦合,接口设计本身仍有优化空间。整体而言,HAMSTER 提供了一个开源、可复现的层级 VLA 基线,其"用视频/仿真/异构机器人数据微调高层语义规划器"的思路,对如何绕开真机数据瓶颈具有较强的参考价值。

## 参考

1. Kim et al., 2024 — *OpenVLA: An Open-Source Vision-Language-Action Model*(主要单体 VLA 基线对比对象)
2. Gu et al., 2023 — *RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches*(最相关的轨迹式中间表征前作)
3. Goyal et al., 2024 — *RVT-2: Learning Precise Manipulation from Few Demonstrations*(低层 3D 策略之一)
4. Ke et al., 2024 — *3D Diffuser Actor: Policy Diffusion with 3D Scene Representations*(低层 3D 策略之二)
5. Niu et al., 2024 — *LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning*(最相关的单体 VLA + 轨迹辅助任务对比工作)
