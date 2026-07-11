# RoboScape：物理感知的具身世界模型

> **论文**：*RoboScape: Physics-informed Embodied World Model*
>
> **作者**：Yu Shang, Xin Zhang, Yinzhou Tang, Lei Jin, Chen Gao, Wei Wu, Yong Li（通讯）et al.
>
> **机构**：Tsinghua University；Manifold AI
>
> **发布时间**：2025 年 06 月（arXiv 2506.23135）
>
> **发表状态**：未录用（预印本，Preprint / Under review）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.23135) | [PDF](https://arxiv.org/pdf/2506.23135)
>
> **分类标签**：`世界模型` `具身智能` `物理感知视频生成`

---

## 一句话总结

RoboScape 在一个自回归 Transformer 世界模型里,用**时序深度预测** 和**自适应关键点动力学** 两个物理感知辅助任务替代外挂物理引擎,让生成的机器人操作视频既清晰又符合物理:在 AgiBotWorld 上 PSNR 21.85 / LPIPS 0.126 / ΔPSNR 3.34 全面超过 IRASim、iVideoGPT、Genie、CogVideoX;用它合成的数据训练 Diffusion Policy 在 Robomimic Lift 上 200 条合成数据即达 91%(真机 200 条为 92%),训 π0 在 LIBERO 上 800 条合成数据平均 79.1% 反超真机 200 条的 65.2%;作为策略评估器与真仿真器的成功率 Pearson 相关达 0.953。

## 一、问题与动机

世界模型(world model)通过"给定历史观测与动作预测未来观测"充当可交互的仿真器,是缓解机器人真实数据采集昂贵这一瓶颈的有力手段。但作者指出当前具身世界模型的通病:训练目标几乎只盯着 RGB 像素拟合,**缺乏物理感知**——尤其在涉及可形变物体(如布料)的接触密集(contact-rich)操作中,生成视频常出现物体不自然的形变、穿模、运动不连续等 artifact。哪怕是微小的物理不一致,也会严重损害下游策略学习的效果。

作者把根因归结为"过度依赖视觉 token 拟合而无视物理约束"。已有的把物理知识注入视频生成的三条路线都各有硬伤:

- **物理先验正则化**(如对 Gaussian Splatting / 3D 点云施加局部刚性、旋转约束):只适用于人体运动、刚体动力学等窄域,难泛化到多样的机器人场景;
- **基于物理仿真器的知识蒸馏**(用物理引擎抽运动信号/语义图去条件化视频生成):级联管线计算开销大,难以实际部署;
- **材料场建模**:局限于物体级(object-level),难以扩展到场景级(scene-level)生成。

因此本文要解决的核心挑战是:**如何在一个统一且计算高效的框架里,把物理知识整合进世界模型学习,而不需要复杂的模型级联或额外训练管线**。此外作者提到近期也有工作(如 Aether、TesserAct)联合预测 RGB-深度,但它们停留在整帧(whole-image)层面,捕捉不到细粒度的运动动力学与物体形变;而且往往以牺牲 RGB 保真度为代价换取 3D 感知。RoboScape 想同时抓住"全局空间结构(靠时序深度)"和"局部形变与运动(靠时序关键点追踪)"。

## 二、核心方法

### 2.1 问题形式化

世界模型 $f_\theta$ 是一个动力学函数,给定过去观测 $\mathbf{o}_{1:t}$ 与机器人动作 $\mathbf{a}_{1:t}$ 预测下一帧观测:

$$
\mathbf{o}_{t+1} \sim f_\theta(\mathbf{o}_{t+1} \mid \mathbf{o}_{1:t}, \mathbf{a}_{1:t})
$$

其中 $\mathbf{o}\in\mathbb{R}^{H\times W\times 3}$ 是视频帧,$\mathbf{a}\in\mathbb{R}^{k}$ 是 $k$ 维连续动作控制向量(实验里由末端位置、末端朝向、夹爪位置拼接而成)。

**用大白话说**:模型就是一个"看着过去几帧画面+我给的机械臂指令,预测下一帧画面长啥样"的函数,可以一帧帧滚动生成一整段可交互视频。

### 2.2 带物理先验标注的数据处理管线

学习物理感知世界模型需要高分辨率 RGB+深度序列、控制机器人的动作序列、机器人执行的状态序列。作者基于 AGIBOT-World 数据集构建了一条五阶段管线:

1. **Data Collecting**:采集 RGB、动作、状态序列;
2. **Physical Property Annotating**:用 **Video Depth Anything** 生成深度图序列,用 **SpatialTracker** 采样并追踪关键点轨迹(这两类物理先验都能用现成预训练模型高效抽取,保证泛化性与实用性);
3. **Video Slicing**:用 **TransNetV2** 做镜头边界检测、用 **Intern-VL** 生成动作语义,把原视频切成属性归一、运动连续、无镜头跳变、单一动作语义的片段;
4. **Clip Filtering**:用 **FlowNet** 过滤掉运动不明显/杂乱的片段,用 **Intern-VL** 标注关键帧并剔除与关键帧无明确关系的帧;
5. **Clip Categorization**:按动作难度与场景对片段分类重组,支撑由易到难的 **curriculum learning**。

### 2.3 RoboScape 主体

骨架是自回归 Transformer,逐帧地根据历史帧与当前动作预测下一帧,实现帧级动作可控的可交互未来预测。在标准 RGB 预测之外,加两个物理感知辅助任务:时序深度预测、自适应采样的关键点动力学。

**视频 tokenization**:用 **MAGVIT-2** 把原始 RGB 帧 $\mathbf{o}_{1:T}\in\mathbb{R}^{T\times H\times W\times 3}$ 压成离散隐 token $\mathbf{s}_{1:T}\in\mathbb{R}^{T\times H'\times W'\times D}$($H'=H/\alpha$,$W'=W/\alpha$,$\alpha$ 为下采样因子);同样把深度图 $\mathbf{d}_{1:T}$ tokenize 成 $\mathbf{z}_{1:T}$。

**双分支协同自回归 Transformer(DCT)**:RGB 分支 $\mathcal{F}_{\text{RGB}}$ 与深度分支 $\mathcal{F}_{\text{Depth}}$ 并行,各自由堆叠的 Spatial-Temporal Transformer(ST-Transformer)块组成——时间注意力层用因果注意力保证生成因果性,空间注意力层用双向注意力做全局上下文建模。在时刻 $t$,两分支基于历史 token、动作嵌入 $\mathbf{c}_{1:t-1}=\mathcal{E}_a(\mathbf{a}_{1:t-1})$ 与位置嵌入 $\mathbf{e}_{1:t-1}$ 做预测:

$$
\hat{\mathbf{s}}_t = \mathcal{F}_{\text{RGB}}(\mathbf{s}_{1:t-1}\oplus\mathbf{c}_{1:t-1}\oplus\mathbf{e}_{1:t-1}), \qquad
\hat{\mathbf{z}}_t = \mathcal{F}_{\text{Depth}}(\mathbf{z}_{1:t-1}\oplus\mathbf{c}_{1:t-1}\oplus\mathbf{e}_{1:t-1})
$$

其中 $\oplus$ 为带广播的逐元素相加;作者发现这种简单的加性融合就能有效实现动作控制并保持模型效率。

**用大白话说**:一路专门画彩色画面、一路专门画深度图,两路都听同一个动作指令,谁也不许偷看未来的时间帧,但同一帧内可以左右上下随便看。

**几何一致性增强(时序深度)**:把深度分支的中间特征作为几何约束反哺 RGB。每个 ST-Transformer 块 $l$ 上,把深度特征 $\mathbf{h}^l_{\text{depth}}$ 经可学习线性投影 $\mathcal{W}^l$ 后加进对应 RGB 特征:

$$
\mathbf{h}^l_{\text{RGB}} = \mathbf{h}^l_{\text{RGB}} + \mathcal{W}^l(\mathbf{h}^l_{\text{depth}})
$$

两分支都用 token 交叉熵优化:

$$
\mathcal{L}_{\text{RGB}} = -\sum_{t=1}^{T}\mathbf{s}_t\log p(\hat{\mathbf{s}}_t), \qquad
\mathcal{L}_{\text{Depth}} = -\sum_{t=1}^{T}\mathbf{z}_t\log p(\hat{\mathbf{z}}_t)
$$

**用大白话说**:深度图里藏着"哪儿远哪儿近"的 3D 结构,把这份结构信息一层层塞回彩色分支,RGB 生成就不会画出几何上说不通的画面。这一步的直觉是:帧间深度变化本身就编码了关键的 3D 结构信息,与其只拟合 2D 像素,不如让模型隐式学到 3D 场景重建的先验。

**隐式材料理解(关键点动力学)**:核心洞见是——物理材料理解可以从"对接触驱动的关键点做自监督追踪"中涌现(比如机器人把苹果放进塑料袋,准确捕捉袋子形变关键点的运动,就隐式抓住了材料属性)。对每段视频用 **SpatialTracker** 在首帧密集采样 $N_0$ 个关键点并跨 $T$ 帧追踪,得到 $\mathcal{T}_{dense}=\{(\mathbf{p}^1_i,\dots,\mathbf{p}^T_i)\}_{i=1}^{N_0}$,$\mathbf{p}^t_i\in\mathbb{R}^2$ 是其在 token 特征图上的坐标。作者观察到"信息量最大"的关键点往往运动幅度大,于是**自适应** 地按运动幅度选 top-$K$ 最活跃的点(而非依赖昂贵的分割 mask):

$$
\mathcal{M}_i = \sum_{t=1}^{T-1}\lVert \mathbf{p}^{t+1}_i - \mathbf{p}^t_i\rVert_2, \quad \forall i\in\{1,\dots,N_0\}
$$

得到采样轨迹集 $\mathcal{T}_{sample}=\{(\mathbf{p}^1_i,\dots,\mathbf{p}^T_i)\}_{i=1}^{K}$。然后把所有帧的关键点视觉 token 对齐到首帧($t=1$)以强化时序一致性:

$$
\mathcal{L}_{\text{Keypoint}} = \frac{1}{(T-1)K}\sum_{i=1}^{K}\sum_{t=2}^{T}\lVert \hat{\mathbf{s}}_t(\mathbf{p}^t_i) - \hat{\mathbf{s}}_1(\mathbf{p}^1_i)\rVert_2^2
$$

其中 $\hat{\mathbf{s}}_t(\mathbf{p}^t_i)\in\mathbb{R}^D$ 是第 $t$ 帧第 $i$ 个关键点所在位置的预测 token。

**用大白话说**:挑出画面里动得最猛的一小撮点(通常是机器人和被操作物体),要求它们无论运动/形变到哪一帧,生成出来的那块视觉 token 都得跟第一帧对得上——相当于逼模型学会"这块布还是同一块布、这个苹果还是同一个苹果",材料软硬、形变规律就被隐式编码进去了。这样绕开了显式材料场建模,又能自然嵌入视频生成框架、保持强泛化性。

**关键点引导注意力**:活跃关键点区域因运动复杂 token 误差更高,于是给这些时空位置的重建 loss 加权。定义时空注意力图 $\mathbf{A}\in\mathbb{R}^{T\times H'\times W'}$:

$$
\mathbf{A}_{t,x,y} = \begin{cases}\gamma & \text{if } (t,x,y)\in\mathcal{T}_{sample}\\ 1 & \text{otherwise}\end{cases}
$$

$$
\mathcal{L}_{\text{Attention}} = -\sum_{t=1}^{T}\mathbf{A}_t\odot\mathbf{s}_t\log p(\hat{\mathbf{s}}_t)
$$

$\gamma$ 是控制重要性权重的超参。**用大白话说**:在关键点轨迹扫过的地方,把 RGB 重建 loss 放大 $\gamma$ 倍,逼模型在"最难画、最易错"的运动剧烈区多下功夫。

**总损失**:

$$
\mathcal{L} = \mathcal{L}_{\text{RGB}} + \lambda_1\mathcal{L}_{\text{Depth}} + \lambda_2\mathcal{L}_{\text{Keypoint}} + \lambda_3\mathcal{L}_{\text{Attention}}
$$

## 三、实验结果

**实验设置**:AgiBotWorld-Beta 数据集抽 50,000 段视频(覆盖 147 任务、72 技能);预处理成 16 帧、2Hz 采样的片段,约 6.5M 训练片段;训 5 epoch,$\lambda_1=1,\lambda_2=0.01,\lambda_3=1,\gamma=5$;在 32 张 NVIDIA A800-SXM4-80GB 上约 24 小时训完;推理时以首帧为条件自回归预测后续 15 帧。评测三个维度六个指标:外观保真(PSNR↑、LPIPS↓)、几何一致(AbsRel↓、$\delta_1$↑、$\delta_2$↑)、动作可控(ΔPSNR↑,对动作条件的敏感度,越高越能被动作控制)。

**视频生成质量对比(Table 1,基线 IRASim、iVideoGPT 为具身世界模型,Genie、CogVideoX 为通用世界模型)**:

| 方法 | LPIPS↓ | PSNR↑ | AbsRel↓ | $\delta_1$↑ | $\delta_2$↑ | ΔPSNR↑ |
|---|---|---|---|---|---|---|
| IRASim | 0.6674 | 11.57 | 0.6252 | 0.5013 | 0.7020 | 0.0269 |
| iVideoGPT | 0.4963 | 16.12 | 0.7586 | 0.3480 | 0.5795 | 0.1144 |
| Genie | 0.1683 | 19.76 | 0.4425 | 0.5435 | 0.7736 | 1.9871 |
| CogVideoX | 0.2180 | 17.52 | 0.5243 | 0.6046 | 0.7599 | —（不支持动作条件）|
| **RoboScape** | **0.1259** | **21.85** | **0.3600** | **0.6214** | **0.8307** | **3.3435** |

RoboScape 在全部六个指标上都居首。作者分析:CogVideoX 能生成高质量视频但跟不上动作指令,导致未来帧严重偏离;两个具身世界模型在长时生成时运动建模差、指标很低;而 RoboScape 靠关键点动力学同时拿到高保真视觉与更强动作可控性。

**消融(Table 2)**:

| 变体 | LPIPS↓ | PSNR↑ | AbsRel↓ | $\delta_1$↑ | $\delta_2$↑ | ΔPSNR↑ |
|---|---|---|---|---|---|---|
| whole model | 0.1259 | 21.85 | 0.3600 | 0.6214 | 0.8307 | **3.3435** |
| w/o depth | 0.1249 | 21.95 | 0.3921 | 0.5788 | 0.8277 | 3.4863 |
| w/o keypoint | 0.1264 | 21.71 | **0.3417** | **0.6497** | **0.8673** | 2.9462 |
| w/o depth & keypoint | 0.1299 | 21.49 | 0.3565 | 0.6248 | 0.8129 | 1.9871 |

结论:两个组件互补——深度学习主要保住运动物体的几何一致(去掉后 AbsRel 0.36→0.39、$\delta_1$ 0.62→0.58 明显变差);关键点学习对视觉保真与动作可控都关键(去掉后 ΔPSNR 3.34→2.95);两个都去掉时 ΔPSNR 跌到 1.99,退化最严重。值得注意的是去掉深度分支反而让 LPIPS/PSNR/ΔPSNR 略好——印证了 intro 里提到的"联合 RGB-深度会以牺牲 RGB 保真为代价"的张力,RoboScape 是在几何一致与像素保真之间取一个较优平衡点。

**下游策略学习——合成数据训练(Table 3)**:

*Diffusion Policy 在 Robomimic Lift 任务(DP 训 10k 步)*:

| 合成数据量 | 成功率 |
|---|---|
| 50 | 40% |
| 100 | 77% |
| 150 | 84% |
| 200 | 91% |
| 真机(200)| 92% |

仅用 200 条纯合成数据即逼近真机 200 条,且成功率随合成数据量单调上升。

*π0 在 LIBERO 任务(用 200 条真数据做 warm-up)*:

| 合成数据量 | Spatial | Object | Goal | Long(10)| 平均 |
|---|---|---|---|---|---|
| 200 | 77.6 | 81.8 | 71.0 | 36.0 | 66.6 |
| 400 | 79.4 | 85.2 | 74.6 | 46.2 | 71.4 |
| 600 | 81.6 | 86.0 | 78.0 | 51.8 | 74.4 |
| 800 | 84.6 | 89.0 | 82.8 | 60.0 | **79.1** |
| 真机(200)| 77.2 | 79.8 | 68.8 | 34.8 | 65.2 |

800 条合成数据在四个子集上全面超过真机 200 条(平均 79.1% vs 65.2%),说明合成数据即使在多物体、杂乱场景、长时序等更难的 LIBERO 上也能生成物理合理的轨迹持续增益。

**策略评估器能力(Sec 3.4)**:让世界模型充当环境,接收策略动作、滚动预测观测,再靠人工判断预测视频里的成功率。在 Robomimic Lift 上训练 DP、每 250 epoch 存档,在真仿真器与世界模型里各跑 100 次评估,算两者成功率的相关性。RoboScape 的 Pearson 相关 **0.953**($R^2=0.908$),而 IRASim 为 -0.134($R^2=0.018$)、iVideoGPT 为 0.195($R^2=0.038$),表明 RoboScape 可当作可靠的策略评估器。

**Scaling(附录 C)**:模型侧 RoboScape-S(34M)/M(131M)/L(544M)三档,六个指标随容量增大全面提升,呈现清晰 scaling law;数据侧 1M/3M/6M 片段,视觉质量与动作可控随数据增长稳步改善,但几何指标改善微弱甚至略降——作者解释为小数据集会过拟合到条件输入的末帧,虚高了几何指标而没真正生成有意义的时序动态。

## 四、局限性

- **物理先验受限于现成模型**:深度与关键点分别来自 Video Depth Anything 与 SpatialTracker,监督信号质量的上限由这两个预训练模型决定,其误差会作为噪声标签传入世界模型。
- **几何一致与像素保真存在真实张力**:消融显示加深度分支反而略降 LPIPS/PSNR/ΔPSNR,说明"物理约束"与"视觉保真"并非全然共赢,当前是折中而非帕累托改进。
- **关键点损失的对齐假设偏强**:$\mathcal{L}_{\text{Keypoint}}$ 把每帧关键点 token 都对齐到首帧,隐含"关键点区域外观跨帧基本不变"的假设;当关键点区域发生大遮挡、大形变或光照剧变时,该约束可能与真实动力学冲突。
- **评估依赖人工判定**:策略评估实验里成功与否需人工看预测视频判断,主观且难规模化;策略学习的"合成数据增益"也只在 Robomimic Lift、LIBERO 两个仿真基准上验证。
- **未在真机验证**:论文只做仿真闭环,作者在结论中明确把"结合真实机器人测试"列为 future work;15 帧自回归预测的长时序复合误差也未系统分析。
- **数据来源单一**:训练数据全部来自 AgiBotWorld-Beta,跨数据集/跨本体(embodiment)的泛化未评测。

## 五、评价与展望

**优点**。① 定位清晰:把"物理感知"落到两个可用现成模型高效抽取的信号(时序深度、关键点轨迹)上,作为辅助监督任务塞进世界模型自身,彻底避开物理引擎级联的算力与部署负担,这套"辅助任务即正则"的思路简洁且工程可行。② 关键点动力学的自监督对齐是本文最有新意的一笔——用"追踪最活跃点并做跨帧 token 一致性"隐式逼近材料/形变属性,比显式材料场建模更易扩展到场景级、可形变物体。③ 评测闭环完整:不止刷生成指标,还打通了"合成数据训策略"和"世界模型当评估器"两条下游链路,0.953 的评估相关性尤其能说明生成结果的物理可用性,而非仅仅好看。④ scaling 与 curriculum 的加入,让方法有向大规模推进的空间。

**不足与存疑**。① 六指标全胜的对比中,基线是否都在同等数据/分辨率/训练量下公平复现值得关注(Genie 用开源复现实现),尤其 IRASim/iVideoGPT 的 PSNR 低至 11–16、AbsRel 高达 0.63–0.76,与其原论文表现落差较大,可能存在设置不利于基线的情况。② 深度分支"帮几何却略伤 RGB"的现象没有被进一步优化(例如用门控/自适应权重而非固定加性融合),留下改进余地。③ 关键点损失在 token 空间做 L2 对齐,与离散 token 的交叉熵目标混用,几何直觉上是否最优、是否会与 MAGVIT-2 码本冲突,论文未深究。④ 15 帧短时窗、单数据源,离"通用可交互仿真器"仍有距离。

**与公开工作的关系**。RoboScape 与同期联合 RGB-深度的具身世界模型 Aether、TesserAct 属同一条"给世界模型加 3D/4D 感知"的赛道,差异在于后者停在整帧层面,而本文额外用关键点抓局部形变与运动,并明确对比声称在不牺牲 RGB 保真的前提下拿到更细的动力学;骨架上继承 Genie 式 masked 自回归 + ST-Transformer 与 MAGVIT-2 tokenizer;物理注入路线上与 PhysGen、PhysMotion、PhysDreamer 等"物理引擎/材料场"路线形成对照,主打"轻量、隐式、可扩展"。下游用法上,"世界模型合成数据训 DP/π0"和"世界模型当策略评估器"与 IRASim、iVideoGPT、UniSim 一脉相承。

**开放问题与可能改进**。(a) 用更强或自训练的深度/追踪模型闭环迭代,减轻噪声标签;(b) 把固定加性深度融合换成可学习门控或跨分支注意力,尝试做到几何与保真的帕累托改进;(c) 关键点对齐引入遮挡/可见性掩码与显式流场约束,缓解大形变下的对齐冲突;(d) 拉长自回归时窗并系统研究复合误差,配合 memory 机制做长时一致;(e) 跨本体、跨数据集与真机闭环验证,并把人工判定的策略评估替换为可自动化的成功检测器。

## 参考

1. Zhu et al. *IRASim: Learning Interactive Real-Robot Action Simulators.* arXiv:2406.14540, 2024.（具身世界模型基线）
2. Wu et al. *iVideoGPT: Interactive VideoGPTs are Scalable World Models.* NeurIPS 2024.（可交互自回归世界模型基线）
3. Bruce et al. *Genie: Generative Interactive Environments.* ICML 2024.（masked 自回归世界模型骨架来源）
4. Aether Team. *Aether: Geometric-Aware Unified World Modeling.* arXiv:2503.18945, 2025.（联合 RGB-深度、几何感知世界模型,同期对照）
5. Zhen et al. *TesserAct: Learning 4D Embodied World Models.* arXiv:2504.20995, 2025.（4D 具身世界模型,同期对照)
