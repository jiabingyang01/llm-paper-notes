# SimFoundry：面向策略学习与评测的模块化自动场景生成

> **论文**：*SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation*
>
> **作者**：Nadun Ranawaka, Josiah Wong（共同一作）, Wei-Lin Pai, Tianyuan Dai, Linxi Fan, Danfei Xu, Li Fei-Fei, Bowen Wen, Ajay Mandlekar, Yuke Zhu 等 et al.
>
> **机构**：NVIDIA；Georgia Institute of Technology；Stanford University；The University of Texas at Austin；University of Toronto
>
> **发布时间**：2026 年 06 月（arXiv 2606.28276，v2 2026-07-04）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.28276) | [PDF](https://arxiv.org/pdf/2606.28276)
>
> **分类标签**：`Real2Sim` `场景生成` `Digital Cousins` `策略评测` `Sim2Real`

---

## 一句话总结

SimFoundry 用一段真实视频零样本重建出可交互、物理就绪（sim-ready）的数字孪生场景，并在 object / scene / task 三个轴上自动繁衍"数字表亲"（digital cousins）来放大数据多样性；其仿真评测与真机表现强相关（7 任务 × 5 策略，mean Pearson r=0.911、MMRV=0.018，比 SOTA 基线 PolaRiS 的相关性高 0.59 以上），且用其合成数据训练的策略可零样本迁移真机（Pot on Stove 达 99%、Stack Dishware 达 100%），三类表亲分别带来真机平均 17% / 21% / 40% 的成功率提升。

## 一、问题与动机

- 机器人基础模型依赖大规模真机操作数据,而这类数据靠遥操作采集,耗时数月乃至数年;在真机上系统评测策略同样昂贵,需要跨任务上千次试验才能做严谨对比。
- 仿真是可扩展的替代方案:自动数据生成工具能合成大批多样、高质量演示;基于仿真的评测也被证明能与真机结果强相关,提供省时省钱的物理基准替代。但**手工搭建**能在视觉、几何、动力学上对齐真实场景的仿真环境仍很难。
- Real-to-Sim 场景构建借助 3D 重建与生成模型,以最小人工搭出"sim-ready"环境。但现有系统割裂:一类只重建 sim-ready 3D 场景,缺物理交互 / 任务定义 / 数据生成机制,无法闭合 sim-to-real 的策略学习环路;另一类为仿真评测设计,却假设场景手工调好、只关注短程原子操作,不支持自动繁衍多样的物体、场景和任务。
- SimFoundry 的目标是把三种此前分散的能力**统一**进一个模块化系统:①重建 sim-ready 数字孪生;②把重建结果扩展为多样训练环境;③用这些仿真同时对策略做基准评测和训练。模块化设计使各基础模型可随更优版本即插即换,无需重构整条流水线。

## 二、核心方法

### 形式化设定

给定输入视频,将真实场景 $\mathcal{S}_{real}$ 转成仿真场景 $\mathcal{S}_{sim}$,即一组物体网格、尺度与位姿的三元组

$$\mathcal{S}_{sim}=\{(\mathcal{M}_i,\ \mathbf{s}_i,\ \mathbf{p}_i)\}_{i=1}^{N}=V_*(\text{video}),$$

其中 $V_*$ 表示一组现成基础模型(深度估计、分割、位姿、网格生成等)。策略记为 $\pi_\theta:\mathcal{O}\to\mathcal{A}$。论文区分两个关键概念:**digital twin(数字孪生)** 是对真实场景几何与物体布局的严格复制;**digital cousin(数字表亲)** 是保留语义与几何 affordance、但不显式复制原场景的虚拟变体,充当一种 object instance randomization。

> 用大白话说:先把一段视频"拆解 + 复刻"成一堆带位姿的 3D 物体拼成仿真场景(孪生);再在此基础上"换一批长得不同但能干同样活的物体/换布局/换任务"造出一大家子表亲,专门用来喂数据、涨泛化。

### 流水线三阶段(Extraction → Generation → Augmentation)

**① Extraction(逐物体信息抽取)**:从原始 RGB 视频取代表帧 $\mathbf{I}_s$,用现成深度模型 $V_{im2depth}$(Depth Anything 3 等)估深度 $\mathbf{D}_s$;结合相机内参 $\mathbf{K}$ 把 RGB-D 反投影为点云 $\mathbf{P}_s$,再用分割模型 $V_{seg}^{image}$(SAM3)提取地平面、对齐到仿真世界坐标系。场景理解 VLM $V_{scene}$ 检测物体,$V_{seg}^{image}$ 迭代分割前景物体 $o_1,\dots,o_n$;每提取一个物体的掩码 $m_i$ 与对应 RGB / 深度像素 $(p_i^{rgb},p_i^{depth})$ 后,用图像 + 深度 inpainting 将其从 RGB-D 中抹去,循环直到无前景物体。输出:逐物体 RGB-D crop 与掩码。

> 用大白话说:像"剥洋葱"一样一个个把桌面物体抠出来,抠一个就把它 P 掉,再抠下一个,直到桌子空了,顺便把地面和相机对齐好。

**② Generation(网格生成、对齐、标注)**:对每个物体 crop 用 $V_{image}$ 超分,再用 2D-to-3D 网格模型 $V_{mesh}$ 生成视觉网格 $\mathcal{M}_i$;结合场景 RGB-D、掩码与点云几何估计并精修物体位姿 $\mathbf{p}_i$,并用 $V_{pose}$(FoundationPose)进一步对齐。铰接物体(柜子、抽屉)交由单独的 articulation 模块:检测可动部件、分割网格、用 $V_{articulation}$ 与既有铰接生成方法产出关节参数。碰撞几何用 CoACD 生成,质量 / 摩擦等物理属性向 $V_{scene}$ 查询;最后在 PyBullet 里组装、消解物体穿模得到稳定构型,导出到 IsaacLab 等下游仿真器。

**③ Augmentation(数字表亲繁衍)**:把重建孪生扩成一"家族"保 affordance 的变体,沿三轴变化:

- **Object cousins**:生成保留原物体 affordance、但几何/拓扑/外观不同的新实例(如一个杯子/抽屉/盘子换成不同形状、把手、纹理、比例的可信替代)。做法是图像扰动 + 2D-3D 重生成,提供实例级多样性而不破坏任务相关功能。
- **Scene cousins**:用语义空间谓词(如 **OnTop**、**RightOf**)改变物体空间排布——不是简单随机扰位姿,而是产生有意义的替代布局(把物体从容器旁移到容器里/上),还可从 sim-ready 资产库加入可控 distractor,引入结构化几何多样性。
- **Task cousins**:基于场景中可用物体与 affordance 提出额外可行的操作任务,转成仿真兼容的 goal specification(输出 BDDL),支持在同一重建场景上做多任务演示采集(含共享物体/目标/中间状态的相关任务),并可加入 user-specified 与 robot hardware 约束。

**背景重建**:前述流水线产出的是物理接地的前景网格;为拿到照片级背景,SimFoundry 可将重建物体与 3D Gaussian Splat 背景融合。提供两条路线:*automatic*(只用同一段原始视频,经 prompted video segmentation + 两遍 inpainting 去前景,恢复度量深度与相机位姿,训练深度监督的 splat 并桥接进仿真世界)与 *manual*(额外拍一段已物理移除前景的空场景视频再训 splat、交互对齐);两者产出相同资产结构,在采集成本与纹理/轮廓保真度间权衡。机器人实验中也用 Scaniverse 之类 app 做网格背景重建。

### 真实-仿真相关性度量

用两个指标衡量仿真评测与真机的一致性:**Pearson 相关系数** $r$(度量真/仿任务成功率的线性相关,理想 $r\to 1$)与 **Mean Maximum Rank Violation(MMRV)**(度量策略在仿真中相对真机的最坏排序违反的平均值,理想 $\to 0$)。任务成功按 end-to-end 二值 0/1 计。

> 用大白话说:光看"仿真成功率数值"没用,关键是"仿真里谁强谁弱的排名"要和真机一致——$r$ 看数值关系,MMRV 看排名有没有被翻车。

## 三、实验结果

评测覆盖两种本体:DROID(单臂 Franka)与 YAM 工作台(双臂);7 个任务(Cup in Bowl / Serve Fruits / Marker in Cup / Stack Dishware / Store Marker / Throw Away Trash / Pot on Stove 等),涵盖短程 pick-and-place、双臂协调、多步与铰接物体交互;5 类策略:$\pi_0$、$\pi_{0.5}$、GR00T N1.6、GR00T N1.7、DreamZero。

**(1) 重建保真度**——12 个重建场景,零样本 F1 0.81–0.92;每物体额外约 3 分钟人工调优后升到 0.93–0.99。相比 SOTA 的 SAM3D:

| 指标 | SimFoundry(全自动) | SAM3D |
|---|---|---|
| F1 score | 0.81–0.92 | 0.66–0.71 |
| Chamfer 距离 | 更低 | 较高 |
| 位置误差 | 更低 | 较高 |

**(2) Real-to-Sim 策略评测**——仿真评测与真机强相关,且完全零样本(策略不对任一仿真框架做适配/微调):

| 指标 | SimFoundry | 说明 |
|---|---|---|
| mean Pearson $r$ | **0.911** | 7 任务 × 5 策略 |
| MMRV | **0.018** | 排序几乎不违反 |
| 相对 PolaRiS 的 $r$ 优势 | **+0.59 以上** | PolaRiS 靠浅层微调获相关性,SimFoundry 纯零样本 |
| sub-task 评测后的 $r$ | 0.90 → **0.95** | 按子任务粒度评测进一步提升多步任务相关性 |

评测还能揭示模型专长:GR00T N1.7 在精细抓取(Marker in Cup)更强,$\pi_{0.5}$ 在语言跟随(Serve Fruits)更强。

**(3) Sim-to-Real 策略训练**——用 SimFoundry 合成数据训练的策略可零样本上真机:YAM 上 **Pot on Stove 达 99%**(从零训练的简单 flow-matching 策略),DROID 上 **Stack Dishware 达 100%**(微调 $\pi_{0.5}$)。加少量真机数据 co-train 进一步提升:**Store Marker 从 60% 升到 92%**;$\pi_0$ 在 Throw Away Trash 上获 36% 仿真成功率提升。

**(4) 三类表亲的增益**:

| 表亲类型 | 关键增益 | 真机平均提升 |
|---|---|---|
| Object cousins | held-out Pot on Stove 物体上 **+50 分** 真机;Throw Away Trash 最高 +20 分 | +17% |
| Scene cousins | Throw Away Trash 仿真 **+28 分**;Store Marker 表亲场景达 16%(twin-only 为 0%) | +21% |
| Task cousins | 13 个 task cousins 使 Throw Away Trash **+60%**、Store Marker **+40%** | +40% |

**(5) 多任务泛化(Table 2,成功率 %)**——重建杂乱场景、用 $V_{scene}$ 提任务、全程在仿真里采集演示微调:

| 设置 | $\pi_{0.5}$-DROID | $\pi_{0.5}$-FT | $\pi_{0.5}$-DROID-FT |
|---|---|---|---|
| Sim | 30 | 51 | **61** |
| Sim – held out | 37 | 45 | 33 |
| Real | 28 | 45 | **46** |
| Real – held out | 26 | **29** | 26 |

SimFoundry 微调策略比 base DROID checkpoint 仿真最高 **+31%**、真机 **+18%**;$\pi_{0.5}$-FT 在无任务专属演示下于 held-out 真机任务达 **29%**。固定演示总量时,把部分目标任务数据替换为相关 task-cousin 演示,在更难任务上收益更大。

## 四、局限性

- **强依赖现成基础模型**:模块化换来即插即换的灵活性,但也天然继承每个底层模型的失败模式(深度、分割、位姿、网格生成任一出错都会传导)。
- **平面桌面假设**:当前流水线针对 tabletop 布局做了若干假设;放宽到多层 / 非平面环境是明确的未来方向。
- **人工介入仍在环**:虽宣称全自动,但达到最高保真(F1 0.93–0.99)需每物体约 3 分钟操作员调优,规模化到成百上千场景的真实人力成本未充分量化。
- **背景 splat 的权衡**:自动背景路线在纹理稀疏表面、物体轮廓上保真度受限,manual 路线又需额外拍空场景视频。
- **相关性 vs 绝对值**:仿真评测强在"保排名"(高 $r$、低 MMRV),但仿真绝对成功率与真机仍有系统性偏差,评测面向"选模型 / 定方向"而非"预测真机绝对性能"。

## 五、评价与展望

**优点**。SimFoundry 最有价值之处是把 real-to-sim 领域此前割裂的三件事——重建、繁衍、评测/训练——收进一条模块化流水线,并同时打通了评测环(高 $r$、低 MMRV)与训练环(零样本上真机)。相较只做评测的 PolaRiS(需浅层微调适配)、只做重建的 3D 场景生成工作,以及主要面向短程原子任务的 SIMPLER 类基准,本文在任务复杂度(多步、铰接、双臂)与本体覆盖(DROID + YAM)上都更进一步。digital cousins 沿 object/scene/task 三轴的显式分解也比笼统的域随机化更可控——尤其 task cousins 用 VLM 提任务 + BDDL 落地,把"多任务数据生成"这个通常最费人力的环节自动化了。

**与公开工作的关系**。方法论上直接站在 ACDC / digital cousins(Dai et al. [17])肩上,把"数字表亲"从单物体检索推广到 object/scene/task 三轴繁衍;评测协议沿用 Li et al.(SIMPLER,[47])提出的 $r$ 与 MMRV;数据生成思路与 MimicGen / DexMimicGen([37])的 subtask 拼接互补(后者拼轨迹、前者造环境);背景走 3D Gaussian Splatting([38])+ 前景网格的"混合表征",这与近来 splat-only 渲染路线(牺牲接触物理)形成对照——SimFoundry 坚持前景可交互网格以保物理接地。

**开放问题与可改进方向**。(1)误差传导缺乏系统刻画:整条链上任一基础模型出错如何影响下游成功率与相关性,论文未给敏感性分析;(2)相关性会不会"过拟合"到 NVIDIA 自家策略族( $\pi$ / GR00T / DreamZero )值得警惕,跨机构第三方策略上的相关性是更硬的检验;(3)cousins 的多样性目前靠谓词与图像扰动启发式生成,缺少"多样性是否覆盖真机分布"的定量度量,存在"生成一堆但都偏离真机"的风险;(4)从 tabletop 扩到长程移动操作、非平面/多层场景是自然但非平凡的下一步;(5)3 分钟/物体的人力在大规模建库时会累积,如何用主动学习只在高不确定物体上花人力是可落地的优化点。总体是一篇工程完成度高、闭环打得实的系统论文,主要价值在"可复用的 real-to-sim 基础设施"而非单点算法创新。

## 参考

1. Dai, Wong, Jiang, et al. *Automated Creation of Digital Cousins for Robust Policy Learning (ACDC)*. arXiv:2410.07408, 2024. — digital cousins 概念源头。
2. Jain, Zhang, Arora, ..., Levine, Finn, et al. *PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies*. arXiv:2512.16881, 2025. — 本文主要评测基线。
3. Xuanlin Li, Hsu, Gu, et al. *Evaluating Real-World Robot Manipulation Policies in Simulation (SIMPLER)*. arXiv:2405.05941, 2024. — $r$ 与 MMRV 相关性评测协议来源。
4. Jiang, Xie, Lin, ..., Fan, Zhu. *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning*. arXiv:2410.24185, 2024. — 互补的合成演示生成路线。
5. Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG, 2023. — 背景重建所用渲染表征。
