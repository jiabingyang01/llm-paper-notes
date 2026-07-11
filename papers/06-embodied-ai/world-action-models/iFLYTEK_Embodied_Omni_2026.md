# iFLYTEK-Embodied-Omni：统一视觉、语言、视频与动作的"大脑-小脑"具身基础模型

> **论文**：*iFLYTEK-Embodied-Omni Technical Report*
>
> **作者**：Yuan Zhang, Jingfei Ni et al.
>
> **机构**：iFLYTEK（科大讯飞）；LindenBot；University of Science and Technology of China
>
> **发布时间**：2026 年 07 月（arXiv 2607.02542）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.02542) | [PDF](https://arxiv.org/pdf/2607.02542)
>
> **分类标签**：`世界动作模型` `统一多模态基础模型` `VLA` `视频生成` `RoboTwin 2.0`

---

## 一句话总结

iFLYTEK-Embodied-Omni 用一套共享的 Omni 多模态自注意力把视觉-语言模型（VLM）、视频生成模型（VGM）与动作生成模型（AGM）整合进单一 Transformer——VLM+VGM 构成负责语义规划与未来视觉预测的"大脑"，AGM 是把规划直接转译为动作块的"小脑"，不经过级联的视频生成再逆动力学的中间接口；配合四阶段渐进-联合训练，在 LIBERO-Plus 零样本泛化基准取得 89.6% 平均成功率，在 RoboTwin 2.0 的 Clean/Rand 设置上分别取得 93.68%/93.16%，均为论文汇报中的当期最优。

## 一、问题与动机

论文观察到两类现有路线的互补缺陷：

- **VLA 类模型**（OpenVLA、π0 等）继承预训练视觉-语言表征，具备较强的指令理解、语义推理和高层任务规划能力，但通常把动作生成学成从观测/指令到控制的直接映射，**没有显式建模动作如何改变环境**，因而捕捉物理动力学和预判未来视觉状态的能力有限。
- **WAM 类模型**通过基于视频的世界建模学习物体运动、交互结果与时空动态，为预测环境演化提供了更强基础，但通常**专注于未来视觉预测和低层动作生成**，对语言驱动的语义推理、组合式任务理解和长时序规划支持有限。此外，传统 WAM 常采用"先生成未来视觉观测、再用逆动力学模型（IDM）反推动作"的级联管线，这种显式中间接口会让视觉生成误差传播到动作预测，且视频生成模块无法直接为动作预测塑造表征。

因此核心挑战是：如何在不同语义抽象层级和时间粒度上对齐视觉（图像/视频）、语言、动作等异质模态，同时把 VLA 的语义智能与 WAM 的动力学建模能力结合起来，而不是简单地合并这些模态或依赖级联接口。

## 二、核心方法

### 2.1 Omni 架构与大脑-小脑分工

模型由三个模态专精组件构成：VLM（继承自预训练视觉语言模型，负责语言-图像理解、语义推理、任务规划）、VGM（继承自预训练视频扩散模型，负责未来视觉状态预测）、AGM（从 VGM 的视频扩散权重初始化，再在机器人轨迹上微调，从而复用时序生成先验并适配可执行的机器人控制）。

处理流程：语言-图像、视频、动作三路模态流各自先经过 $M$ 层模态内自注意力 Transformer block；随后拼接为统一 token 序列，送入 $L$ 层 Omni 多模态自注意力层，实现跨模态对齐；之后再分离，经过 $N$ 层模态专属输出 block。这一结构自然形成"大脑-小脑"协作：VLM 与 VGM 组成负责任务分解、时空/语义推理、未来视觉状态预测与进度追踪的高层"大脑"；AGM 则是利用大脑产出的子目标信息与共享 Omni 上下文，生成可执行动作块的低层"小脑"。

论文用统一记号刻画了模型支持的六种预测模式（表 1）：

| 模型 | 预测目标 |
|---|---|
| VLM | $p(\ell_{goal} \mid \mathbf{o}_t, \ell)$ |
| VLA | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ |
| WM | $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{a}_{t+1:t+k})$ |
| IDM | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_{t:t+k})$ |
| VGM | $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ |
| Joint Video-Action | $p(\mathbf{o}_{t+1:t+k}, \mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ |

用大白话说：同一个模型骨架，通过更换注意力 mask 和条件输入，既能当纯语言模型用（VLM 目标追踪/规划），也能当世界模型用（WM：给动作猜观测），也能当逆动力学用（IDM：给观测猜动作），也能纯生成未来视频（VGM），还能视频-动作联合生成——six-in-one 由统一的 Omni 上下文支撑。

### 2.2 跨模态注意力 mask 与多视角编码

由于不同模态生成顺序和预测目标不同，论文没有对所有 token 用统一注意力模式，而是为每种预测模式设计**模态感知的注意力 mask**：图像-语言建模中，同一图像内的图像 token 用双向注意力保持全局视觉上下文，语言 token 在文本序列内用自回归因果 mask；视频建模中，VGM 在帧内和跨帧都用双向注意力，联合建模空间结构与时间动态；动作建模中，同一动作块内所有动作 token 用双向注意力，使 AGM 能生成时序一致的低层控制。每种预测模式只暴露该目标所需的有效条件上下文，防止目标 token 泄露不该看到的未来信息。

多视角视频输入方面，论文提出 **View-aware Latent Fusion**：每个相机视角先独立经 VAE 编码，再把各视角的 latent 拼接作为该时间戳的视频 token 表示；在每个图像 latent 内部用 RoPE 编码空间位置，并为不同相机视角分配独立的可学习视角嵌入加到对应 latent token 上。这一设计与直接在像素空间拼接多视角图像再统一编码（"Picture-Merge"）的朴素基线相对照，在消融中被验证更优（见第三节）。

### 2.3 闭环推理与去噪加速

推理时模型接收当前语言指令、视觉观测和机器人状态，通过迭代去噪生成可执行动作块：VLM 把指令和观测编码为语义规划上下文，VGM 提供视觉动力学上下文用于未来状态推理，AGM 在共享 Omni 上下文条件下把带噪动作 latent 精炼为低层控制；每个动作块执行完毕后，新观测以 teacher-forcing 方式替换此前预测的视觉状态，反馈进模型内部上下文，实现闭环重规划。

为降低迭代去噪的计算成本，论文引入两项加速机制：

**(1) DiT velocity cache**：利用连续去噪步预测的时间平滑性。设 $\mathbf{u}^{(s)}$ 为去噪步 $s$ 时 DiT 模块预测的速度场（可以是视频速度或动作速度），计算相邻两步预测的余弦一致性

$$\rho_s = \frac{\langle \mathbf{u}^{(s)}, \mathbf{u}^{(s-1)} \rangle}{\|\mathbf{u}^{(s)}\|_2 \|\mathbf{u}^{(s-1)}\|_2}$$

当 $\rho_s$ 超过阈值 $\gamma$ 时，跳过接下来 $c$ 次去噪评估，直接复用最近一次的速度：

$$\hat{\mathbf{u}}^{(s+r)} = \mathbf{u}^{(s)}, \quad r = 1,\dots,c$$

用大白话说：如果这一步和上一步的"去噪方向"几乎没变，说明后面几步的方向大概率也差不多，直接抄上一次结果、跳过前向计算，省掉大量重复推理，缓存在每次调用/每个动作块开始时清空，只在当前采样轨迹内局部复用。

**(2) V2A 分阶段推理**：对于用非对称"视频到动作"注意力训练的模型（视频 token 不 attend 动作 token，但动作 token 可以 attend 语言和视觉上下文），先跑一段较短的联合去噪前缀（$N$ 步），随后冻结视频 latent，只继续做动作端的去噪：

$$\left(\mathbf{z}_v^{(s+1)}, \mathbf{z}_a^{(s+1)}\right) = \begin{cases} \Phi_{joint}\left(\mathbf{z}_v^{(s)}, \mathbf{z}_a^{(s)}, s\right), & s < N \\ \left(\mathbf{z}_v^{(N)}, \Phi_{act}\left(\mathbf{z}_a^{(s)}; \mathbf{z}_v^{(N)}, s\right)\right), & s \ge N \end{cases}$$

用大白话说：视频分支的去噪一旦稳定就"冻住"不再重复算，之后多步去噪只更新动作 token，复用缓存好的视觉-语言 KV，从而砍掉视频分支在采样后期的冗余计算。两项优化结合，使 iFLYTEK-Embodied-Omni 在单张 RTX 4090 GPU 上实现 **3 Hz 推理速度**，同时维持 **30 Hz 机器人控制频率**。

### 2.4 四阶段渐进-联合训练

**Stage I（VLM 微调）**：用通用图文、具身感知、空间推理、任务规划数据以自回归语言建模目标微调 VLM：

$$\mathcal{L}_{VLM} = -\mathbb{E}_{(\mathbf{o}_t,\ell,\mathbf{y})}\left[\sum_{i=1}^{T} \log p_{\theta_V}(y_i \mid y_{<i}, \mathbf{o}_t, \ell)\right]$$

建立指令理解、具身感知、空间推理、任务分解与视觉-语言生成能力。

**Stage II（VGM 训练，冻结 VLM）**：以 flow matching 在未来视频 latent 上训练：

$$\mathbf{z}_v^{\sigma_v} = (1-\sigma_v)\mathbf{z}_v + \sigma_v \boldsymbol{\epsilon}_v, \qquad \mathbf{u}_v^{*} = \boldsymbol{\epsilon}_v - \mathbf{z}_v$$

$$\mathcal{L}_{VGM} = \mathbb{E}\left[\left\| \mathbf{v}_{\theta_G}^{v}(\mathbf{z}_v^{\sigma_v}, \mathbf{o}_t, \ell, \sigma_v) - \mathbf{u}_v^{*} \right\|_2^2\right]$$

用大白话说：让 VGM 在冻结的 VLM 语义上下文条件下学习预测未来视频，本质是把"预测未来会发生什么"变成一个 flow matching 生成问题，从而获取物体运动、交互结果、场景演化的先验。

**Stage III（AGM 训练，冻结 VLM 和 VGM）**：AGM 从 VGM 权重初始化，在动作标注轨迹上做 flow matching，并引入 mask 处理不同机器人本体间填充/无效的动作维度：

$$\mathbf{z}_a^{\sigma_a} = (1-\sigma_a)\mathbf{z}_a + \sigma_a \boldsymbol{\epsilon}_a, \qquad \mathbf{u}_a^{*} = \boldsymbol{\epsilon}_a - \mathbf{z}_a$$

$$\mathcal{L}_{AGM} = \mathbb{E}\left[\frac{\left\| \mathbf{m}_a \odot \left(\mathbf{v}_{\theta_A}^{a}(\mathbf{z}_a^{\sigma_a}, \mathbf{o}_t, \ell, \sigma_a) - \mathbf{u}_a^{*}\right) \right\|_2^2}{\|\mathbf{m}_a\|_1 + \varepsilon}\right]$$

冻结大脑组件，防止动作学习梯度扰乱已经学到的语义和动力学表征。

此外，论文为视频和动作设计了**不同的信噪比偏移调度**（modality-specific noise sampling）：

$$\sigma_m = \frac{s_m r_m}{1+(s_m-1)r_m}, \qquad r_m \sim \mathcal{U}(0,1),\ m \in \{v,a\}$$

其中视频取偏移因子 $s_v=6$（更多训练样本落在高噪声区间），动作取 $s_a=1$（噪声等级均匀分布）。用大白话说：视频比动作更"娇贵"，需要在强噪声区间多花训练力气才能学好复原，而动作不需要这种照顾，均匀采样即可；这一非对称调度还使得 AGM 在推理时能把采样步数从 50 步压到 30 步而不掉性能。

**Stage IV（联合微调）**：解冻 VLM、VGM、AGM 及共享 Omni 层，联合优化：

$$\mathcal{L}_{joint} = \lambda_{vlm}\mathcal{L}_{VLM} + \lambda_{vgm}\mathcal{L}_{VGM} + \lambda_{agm}\mathcal{L}_{AGM}$$

让语义推理、未来视觉状态预测与动作生成通过共享 Omni 多模态自注意力相互适配，完成大脑与小脑之间的最终对齐。

### 2.5 训练数据构成

训练数据由四类互补来源混合而成（占比见论文 Fig. 3）：动作标注机器人轨迹占 **26.88%**（OXE、AgiBot、Droid、RoboMIND、Galaxea、Bridge V2），提供直接的控制监督；无动作人类操作视频占 **18.95%**（HoloAssist、Ego4D、EgoDex、HOI4D、Something-Something V2、EgoVid），虽无机器人动作标签，但富含物体交互、手-物运动和时间变化，用于训练 VGM 的视觉动力学与未来状态预测能力；通用图文/规划数据占 **10.75%**（自建 General VQA 132K、Understanding QA 578K、Planning QA 35K）；具身空间推理/感知数据占 **43.42%**（自建 2D Trajectory 698K、3D Trajectory 150K、2D Grounding 100K、3D Grounding 632K、Space Pointing 625K、Object Pointing 401K、Affordance 393K）。

## 三、实验结果

论文在 LIBERO-Plus（零样本七维分布偏移基准）、RoboTwin 2.0（Clean/Rand 两种设置）、RoboTwin 2.0 长时序七任务子集三个层面评测，并做了两组消融。

**LIBERO-Plus 零样本成功率（%）**（部分代表性方法，Ave. 为全七项分布偏移的平均成功率）：

| 方法 | 类别 | Camera | Robot | Language | Light | Background | Noise | Layout | Ave. |
|---|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | VLA | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| ACoT | VLA | 72.6 | 82.6 | 87.5 | 97.7 | 96.5 | 87.8 | 88.1 | 86.6 |
| HoloBrain-0 | WAM | 65.5 | 58.2 | 78.7 | 88.1 | 90.3 | 66.9 | 79.5 | 74.0 |
| CKT-WAM | WAM | 77.4 | 71.4 | 86.7 | 98.2 | 90.2 | 94.8 | 88.5 | 86.1 |
| **iFLYTEK-Embodied-Omni** | WAM | **84.8** | 85.5 | 84.6 | 98.4 | 94.3 | 92.4 | **91.2** | **89.6** |

iFLYTEK-Embodied-Omni 以 89.6% 的平均成功率超过最强 VLA 基线 ACoT（86.6%）3.0 个百分点，超过最强 WAM 竞品 CKT-WAM（86.1%）3.5 个百分点，在 Camera 和 Layout 两项上取得最佳单项成绩（84.8%/91.2%），在 Robot 和 Light 上排名第二（85.5%/98.4%），体现了跨七种偏移类型的均衡表现，而非只在单一变化类型上取胜。

**RoboTwin 2.0（Fig. 4）**：Clean 设置下 iFLYTEK-Embodied-Omni 达 93.68%，超过第二名 LingBot-VA（92.93%）0.75 个百分点；Rand 设置下达 93.16%，超过第二名 HoloBrain-0（92.30%）0.86 个百分点。Clean→Rand 仅下降 0.52 个百分点，显示环境随机化下策略保持稳定。相比最强 VLA 基线，Clean 上超过 π0.5 达 10.94 个百分点，Rand 上超过 ACoT 达 14.44 个百分点。

**RoboTwin 2.0 长时序子集**（7 个多阶段任务：Put Bottles in Dustbin、Open Microwave、Rank Blocks by Size、Stack Three Blocks、Stack Three Bowls、Hang Mug、Handover Block，覆盖物体分类、连续堆叠、关节物体交互、容器操作等）：

| 设置 | iFLYTEK-Embodied-Omni | 第二名（HoloBrain-0） | 差距 |
|---|---|---|---|
| Clean 平均 | 88.3% | 87.0% | +1.3 pp |
| Rand 平均 | 89.0% | 85.6% | +3.4 pp |

最大优势出现在 Hang Mug 任务：Clean 78%、Rand 82%，两个设置均领先第二名 Fast-WAM 达 20 个百分点；Stack Three Bowls 的 Rand 成功率 88%，超过 X-VLA 2 个百分点；Open Microwave 达 93%/90%，Put Bottles in Dustbin 达 90%/94%，Handover Block 达 90%/84%（Clean/Rand）。值得注意的是 Rand 平均反而比 Clean 高 0.7 个百分点，说明环境随机化并未导致这组长时序任务整体性能下降。

**消融研究**（均在 LIBERO-Plus 上进行，Table 3）：

| 消融维度 | 变体 | Ave. | 差距 |
|---|---|---|---|
| MoT 架构 | Two-branch MoT | 85.9% | — |
| MoT 架构 | Three-branch MoT（本方法） | **89.6%** | +3.7 pp |
| 多视角编码 | Picture-Merge | 88.8% | — |
| 多视角编码 | View-aware Latent Fusion（本方法） | **89.6%** | +0.8 pp |

将 VLM/VGM/AGM 拆成三条独立分支（而非把视频-动作合并为单一分支的 Two-branch MoT）在全部七项分布偏移上均有提升，其中 Camera（+4.7 pp）、Noise（+4.5 pp）、Language（+4.0 pp）增益最大；论文用注意力图可视化（Fig. 6）佐证：Three-branch MoT 在多阶段操作任务中的注意力更集中于目标物体、末端执行器和交互区域，而 Two-branch 变体的注意力在背景区域更分散。View-aware Latent Fusion 相对 Picture-Merge 的增益集中在 Camera 和 Language（各 +1.3 pp）、Noise（+1.2 pp）、Layout（+0.8 pp）。

## 四、局限性

论文在结论部分自陈两点局限：

1. **推理成本仍有优化空间**：尽管 DiT velocity cache 与 V2A 分阶段去噪已把推理速度做到单张 RTX 4090 上 3 Hz、同时维持 30 Hz 机器人控制频率，但 Three-branch MoT 架构仍需执行并协调 VLM、VGM、AGM 三条计算路径，在更低延迟、更高频控制或资源受限机器人平台上的部署仍有改进空间；作者提出模型压缩、动态专家路由、更激进的中间计算复用作为未来方向。
2. **动作块长度是固定超参数**：当前动作块长度不能根据子目标的时长或复杂度自适应，短操作可能产生冗余控制，长/困难子任务可能需要更频繁的重规划；作者提出未来研究"子目标条件的自适应动作分块"，根据任务上下文和环境反馈动态确定预测时域。

此外，论文在展望中提到计划将语音作为额外模态纳入 Omni 框架，说明当前版本尚不支持语音交互这一形式的输入/反馈。

## 五、评价与展望

**优点**：该工作用统一的 Omni 多模态自注意力架构，避免了传统 WAM 中"先生成未来视觉观测、再用逆动力学模型反推动作"的级联管线所带来的接口瓶颈和误差传播问题，这是相对于 cascaded video-then-inverse-dynamics 类工作（如论文引用的 Video Prediction Policy、mimic-video 等）的一个结构性改进。三分支（VLM/VGM/AGM 各自独立 modality-specific block、再共享 Omni 自注意力）相对 Two-branch 的消融验证较为扎实（全七项分布偏移均正向提升 3.7 个百分点），并辅以注意力图可视化提供一定可解释性佐证。工程层面给出了较完整的推理加速方案（DiT velocity cache + V2A 非对称注意力分阶段去噪），把一个三分支大模型压到单卡消费级 GPU 上可用的闭环控制频率，这类系统级优化在同类 WAM 技术报告中着墨不多，体现了从算法设计到部署可用性的完整链路。四阶段渐进式训练（先分别训练 VLM/VGM/AGM，再联合微调）是应对异质学习目标相互干扰的常见且有效策略，AGM 从 VGM 权重初始化（复用时序生成先验）是一个实用的工程细节。

**局限与开放问题**：作为技术报告，论文对标的基线（ACoT、CKT-WAM、HoloBrain-0、LingBot-VA、Fast-WAM、World Pilot、X-VLA 等）多为同期 arXiv 上的 WAM/VLA 工作，不少同样是 2026 年上半年才出现的技术报告或预印本，缺乏更长期沉淀的第三方独立复现或公开排行榜验证，数字的可比性和可持续性有待观察。实验仅覆盖 LIBERO-Plus 和 RoboTwin 2.0 两个仿真基准及其长时序子集，未见真实机器人上的定量结果，而摘要与引言部分反复强调面向"开放环境""物理世界"的执行能力，这部分论证在当前版本中略显不足。消融只在 LIBERO-Plus 单一基准上进行，对 RoboTwin 2.0 及长时序任务的架构消融尚未见报告。此外，训练数据中占比最大的具身空间推理/感知数据（43.42%）均为未开源的 in-house 标注（2D/3D Trajectory、Grounding、Pointing、Affordance 等），标注流程与质量控制细节未详述，限制了外部复现性。

与同方向公开工作相比，本文的大脑-小脑框架更强调 VLM 语义规划与 VGM/AGM 世界建模、动作生成之间的模块解耦，但三者又通过共享 Omni 自注意力交互，属于"模态专精+部分统一"的折中路线，与完全端到端联合视频-动作扩散建模（如 Zhu et al. 的 Unified World Models）、以及强调 WAM 本身即可作为零样本策略使用的路线（如 Ye et al. 的 World Action Models are Zero-shot Policies）构成一个有趣的设计谱系。后续工作值得系统比较这几种路线在长时序规划、跨本体泛化、以及推理效率上的取舍；固定长度动作块与推理速度/控制频率之间的权衡，也是当前多数 WAM 共有的工程妥协，自适应动作分块仍是一个尚待解决的开放问题。

## 参考

[1] Kim et al. OpenVLA: An open-source vision-language-action model. arXiv:2406.09246, 2024.

[2] Black et al. π0: A vision-language-action flow model for general robot control. arXiv:2410.24164, 2024.

[3] Ye et al. World action models are zero-shot policies. arXiv:2602.15922, 2026.

[4] Zhu et al. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. arXiv:2504.02792, 2025.

[5] Hu et al. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv:2412.14803, 2025.
