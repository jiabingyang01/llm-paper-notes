# RoboDreamer：面向机器人想象的组合式世界模型

> **论文**：*RoboDreamer: Learning Compositional World Models for Robot Imagination*
>
> **作者**：Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, Chuang Gan
>
> **机构**：Hong Kong University of Science and Technology；Massachusetts Institute of Technology；University of California, San Diego；Google Research；University of Massachusetts Amherst；MIT-IBM Watson AI Lab
>
> **发布时间**：2024 年 04 月（arXiv 2404.12377）
>
> **发表状态**：未录用（预印本，PDF 中未标注会议/期刊录用信息）
>
> 🔗 [arXiv](https://arxiv.org/abs/2404.12377) | [PDF](https://arxiv.org/pdf/2404.12377)
>
> **分类标签**：`组合式世界模型` `视频扩散规划` `语言组合泛化` `多模态目标条件` `逆动力学动作提取`

---

## 一句话总结

RoboDreamer 把自然语言任务指令用句法解析器拆成"动词短语（动作）+介词短语（物体空间关系）"等若干组分，为每个组分训练一个独立的扩散评分函数，推理时以乘积专家（product-of-experts）方式把各组分评分组合成完整视频扩散模型，从而让文本到视频的机器人世界模型具备对**未见过的语言组合**的零样本泛化能力，并可进一步与目标图像、目标草图等多模态条件组合；在 RT-1 数据集上的人类评测显示，对未见指令组合的执行正确率从基线 AVDC 的 46.9% 提升到 81.3%（本文方法 90.1%），在 RLBench 74 项任务的子集上平均成功率 49.3%，超过 UniPi（41.0%）、Hiveformer（44.2%）等基线。

## 一、问题与动机

文本到视频生成模型近年被广泛用作机器人决策的"世界模型"：给定语言指令生成未来观测视频，再用逆动力学模型从视频中提取可执行动作（UniPi 一脉的做法）。但现有方法普遍存在一个泛化缺陷——它们只能合成与训练时见过的语言指令**相似**的视频，一旦指令描述了训练中未出现过的物体组合或空间关系（例如 "move pepsi can near plastic bottle"），模型往往无法准确表达指令中精确的物体空间关系，如论文 Figure 1 所示，基线 AVDC 会把 pepsi can 错误地放到 green can 旁边而非 plastic bottle 旁边。

作者指出，机器人任务的自然语言指令天然具有**组合结构**：全局运动描述 + 物体间的精确空间摆放关系，而现有强化学习数据集中语言指令本身覆盖度有限、分布偏斜，导致模型难以泛化到训练时未见过的动作—物体—关系组合。同时，语言本身是抽象的，难以传达运动的精细细节，因此还需要能引入目标图像、目标草图等更丰富多模态条件的能力。这两点动机共同引出 RoboDreamer：一个通过**分解视频生成过程**、利用语言天然可组合性来实现组合泛化，并进一步支持多模态目标规约的世界模型。

## 二、核心方法

### 2.1 视频生成即规划的形式化（UPDP）

沿用 UniPi（Du et al., 2023b）提出的 Unified Predictive Decision Process（UPDP）formalism，定义元组 $\mathcal{G}=\langle \mathcal{X}, \mathcal{C}, H, \rho \rangle$：$\mathcal{X}$ 为图像观测空间，$\mathcal{C}$ 为文本任务描述空间，$H$ 为规划视野长度，$\rho(\cdot \mid x_0, c): \mathcal{X}\times\mathcal{C}\to\Delta(\mathcal{X}^H)$ 是给定初始观测 $x_0$ 和文本描述 $c$ 的条件视频生成器。学习 $\rho$ 即把"决策"问题转化为"给定语言合成未来图像序列"的文本到视频生成问题，再用轨迹—任务条件策略 $\pi(\cdot \mid \{x_h\}_{h=0}^H, c)$ 从合成视频中推断可执行动作序列。

**用大白话说**：规划不再直接输出动作，而是先"想象"一段未来的画面，再从画面里反推出该怎么动。

### 2.2 文本解析与组合式生成

给定语言指令 $L$，用预训练句法解析器（Kitaev et al., 2018 的 self-attention constituency parser）加规则法把指令拆解为一组语言组分 $\{l_i\}_{i=1:N}$：任务的动作通常对应动词短语（verb phrase），物体空间关系对应动词短语后的介词短语（prepositional phrase）。例如 "place water bottle into bottom drawer" 被拆成动作组分 "place water bottle" 和关系组分 "into bottom drawer"；含 "and" 的复合句（如 "pick orange from bottom drawer and place on counter"）会被进一步拆成多个独立动作—关系子句。

在此基础上，把视频生成的概率模型定义为各组分条件密度的乘积专家：

$$p_\theta(\tau \mid L) \propto \prod_{i=1:N} p_\theta(\tau \mid l_i)^{\frac{1}{N}} \tag{1}$$

**用大白话说**：一段视频"合理"当且仅当它同时满足每一个语言组分各自的要求，各组分各打一票、取几何平均。只要每个组分 $l_i$ 本身在训练分布内，这个乘积形式就能自然泛化到训练时未见过的**新组合**。

利用扩散模型与基于能量的模型（EBM）之间的联系，为每个组分学习一个评分函数 $\epsilon(\tau,t\mid l_i)$，乘积密度对应的评分函数是各分量评分的平均 $\sum_i \frac{1}{N}\epsilon(\tau,t\mid l_i)$，直接用标准去噪目标训练：

$$\mathcal{L}_{\text{MSE}} = \left\| \frac{1}{N}\sum_i \epsilon(\tau_t,t\mid l_i) - \epsilon \right\|^2 \tag{2}$$

但式 (2) 只约束"组合后"的评分正确，并不强制每个单独的 $\epsilon(\tau,t\mid l_i)$ 准确建模 $p_\theta(\tau\mid l_i)$ 本身，因此额外加入单组分去噪目标：

$$\mathcal{L}_{\text{MSE}} = \left\| \epsilon(\tau_t,t\mid l_i) - \epsilon \right\|^2 \tag{3}$$

最终训练目标是二者的混合：给定组分全集 $S=\{l_i\}_{i=1:N}$，每步随机采样一个大小为 $M$ 的子集 $S'$：

$$\mathcal{L}_{\text{MSE}} = \left\| \frac{1}{M}\sum_i \epsilon(\tau_t,t\mid l_{S_i}) - \epsilon \right\|^2 \tag{4}$$

**用大白话说**：训练时既要练"单科"（单个短语单独去噪），也要练"组合科"（随机抽几个短语一起去噪），这样推理时才能灵活地把任意新组合拼在一起还依然可靠。

### 2.3 多模态组合（目标图像/目标草图）

除语言组分外，还引入多模态条件集合 $M=\{m_i\}_{i=1:K}$（如目标图像、目标草图），把似然进一步扩展为语言与多模态条件的联合乘积专家：

$$p_\theta(\tau \mid L, M) \propto \prod_{i=1:N} p_\theta(\tau \mid l_i)^{\frac{1}{N+K}} \prod_{i=1:K} p_\theta(\tau \mid m_i)^{\frac{1}{N+K}} \tag{5}$$

对应训练目标为两部分子采样评分的混合：

$$\mathcal{L}_{\text{MSE}} = \left\| \frac{1}{2M}\sum_i \epsilon(\tau_t,t\mid l_{S_i}) + \frac{1}{2M}\sum_j \epsilon(\tau_t,t\mid M_{S_j}) - \epsilon \right\|^2 \tag{6}$$

**用大白话说**：目标图像/草图和语言短语被同等看待，都是"给评分函数投票"的一个条件源，训练时随机丢弃部分条件、迫使每种条件都能独立发挥作用，这样推理时可以自由挑选任意数量、任意种类的条件组合（哪怕训练时没见过这种搭配）。

### 2.4 推理：组合式引导采样与逆动力学执行

推理时用类似 classifier-free guidance 的组合式引导采样（Algorithm 2）：先算无条件评分 $\epsilon_{\text{uncond}}=\epsilon_\theta(\tau_t,t)$ 和各条件评分 $\epsilon_i=\epsilon_\theta(\tau_t,t\mid l_i)$，再合成 $\tilde\epsilon = \epsilon_{\text{uncond}} + \sum_i w(\epsilon_\theta(\tau_t,t\mid l_i)-\epsilon_{\text{uncond}})$（$w$ 为引导权重），代入标准 DDPM 反向更新迭代出干净视频 $\tau^0$。

得到合成视频计划 $\tau=[x_1,\dots,x_H]$ 后，沿用 UniPi 的做法，训练一个独立的逆动力学模型 $\pi(\cdot)$：输入相邻两帧观测 $x_t,x_{t+1}$，输出应执行的动作 $a$，从 $x_1$ 顺序推到 $x_{H-1}$ 得到整段可执行动作序列；为消除逆动力学估计误差的累积，采用闭环重规划——周期性地基于最新观测重新生成视频计划。

### 2.5 实现细节

视频扩散主干基于 AVDC（Ko et al., 2023）代码库改造，并借鉴 Imagen（Ho et al., 2022）的三级级联超分辨率扩散；U-Net 每个 ResNet block 内用时空卷积层+条件交叉注意力层，编码器最后一 block 与解码器第一 block 加时间注意力层；base channel 128，channel multiplier [1,2,4,8]；256 batch size、5e-5 学习率、约 100 张 V100 训练；先在 $8\times64\times64$ 分辨率训练 base 模型，再逐级上采样到 $8\times128\times128$ 和 $8\times256\times256$。文本编码用冻结的 T5-XXL；目标图像/草图用 Stable Diffusion 预训练 VQVAE 编码器提取特征，经 PerceiverSampler 架构统一为多模态 token 后通过交叉注意力融入 U-Net。逆动力学模型用 ResNet18 主干 + MLP，Adam 优化器、学习率 1e-4，训练 10K 步。

## 三、实验结果

**数据与设置**：主实验用 RT-1 数据集（Brohan et al., 2022），约 7 万条演示、约 500 个任务，每 5 帧采样一次，平均轨迹长度 44。基线为 AVDC（视频生成+动作提取）、HiP（Ajay et al., 2023，组合多个专家基础模型的分层规划）、以及消融版本 RoboDreamer w/o（去掉文本解析组合、退化为单体模型）。

**未见指令组合的人类评测（Table 1，Seen/Unseen 两栏，0/1 打分取均值 × 100）**：

| 模型 | Seen | Unseen |
|---|---|---|
| AVDC | 63.1 | 46.9 |
| HiP | 70.3 | 50.1 |
| RoboDreamer w/o（去组合消融） | 85.5 | 68.8 |
| **RoboDreamer** | **90.1** | **81.3** |

RoboDreamer 相比最强基线 HiP 在 Unseen 组合上高 31.2 个百分点；相比自身去组合消融版本（w/o）也高 12.5 个百分点，说明组合式解析+组合训练本身即带来实质性泛化收益，而不只是模型容量或数据量的差异。

**多模态条件生成质量（Table 2，语言 t / +草图 s / +目标图像 i）**：

| 模型 | Human ↑ | FVD ↓ |
|---|---|---|
| AVDC | 46.9 | 517.1 |
| RoboDreamer (t) | 81.3 | 487.8 |
| RoboDreamer (t+s) | 94.7 | 454.7 |
| RoboDreamer (t+i) | **95.8** | **444.3** |

附录 Table 4 额外报告用预训练 GroundingDino 检测目标物体框、计算 IoU 的 IMO 指标，与人类评测趋势一致：RoboDreamer (t)/(t+s)/(t+i) 的 IMO 分别为 63.5/72.5/78.1，即引入目标图像/草图后空间关系对齐显著改善。

**RLBench 机器人规划（Table 3，成功率 %，仅用单前视相机 RGB、不加目标图像条件）**：

| 模型 | lamp off | lamp on | stack blocks | lift block | take shoes | close box | Average |
|---|---|---|---|---|---|---|---|
| Image-BC | 60.1 | 47.0 | 0 | 0 | 0 | 82.4 | 31.6 |
| Hiveformer | 81.2 | 53.2 | 10.6 | 28.2 | 1.0 | 90.8 | 44.2 |
| UniPi | 70.6 | 47.1 | 7.1 | 23.3 | 3.8 | 94.1 | 41.0 |
| **RoboDreamer** | **96.3** | 51.9 | **18.5** | 22.2 | **10.5** | **96.3** | **49.3** |

RLBench 实验采用 Franka Panda 机械臂（7 自由度、8 维动作空间含夹爪状态），使用 74 项任务中覆盖工具使用、拾放、长时程规划等类别的子集，遵循前人工作以宏观步（macro-step）方式执行。在 "stack blocks" "take shoes" 这类长时程任务上，Image-BC 和 Hiveformer 几乎失败（0–1%），RoboDreamer 分别达到 18.5% 与 10.5%，UniPi 因视频生成与指令对齐差而表现不佳。人类评测每个样本由至少 3 名评审打 0/1 分，共评测约 128 个样本、覆盖 20 余条文本指令。

## 四、局限性

论文 Conclusion 部分明确列出三点局限：

1. **仅支持单相机视角**：许多机器人任务需要多视角信息以获得精确的 3D 场景理解，而 RoboDreamer 目前无法融合多相机输入，限制了其在此类任务上的适用性；作者提出未来可探索引入 3D 归纳偏置来支持多相机信息。
2. **对真实世界图像的泛化能力弱**：论文承认 RoboDreamer 在许多未训练分布内的真实世界图像上泛化较差，作者将其归因于现有机器人数据集多样性有限，提出未来可探索在机器人数据与 YouTube 等网络视频上联合训练以提升泛化。
3. **不支持移动相机场景**：包括本文方法在内的视频生成类世界模型，在相机运动的场景下能力受限，需要额外的稳定化方法来解决。

此外从实验设计本身看，RLBench 上的绝对成功率仍然偏低（平均 49.3%，多个任务个位数到二十几个百分点），组合泛化的评测主要依赖人类主观打分（0/1 二值制），缺乏更细粒度、可复现的自动化指标（附录引入的 IMO 只是辅助验证，仍依赖检测器精度）。

## 五、评价与展望

**优点**：RoboDreamer 的核心贡献在于把"组合泛化"这一在组合式图像生成（Liu et al., 2022 等 composable diffusion models）与组合式文本生成中已有先例的思路，系统性地迁移到文本到视频的机器人世界模型上，并给出了完整的训练（乘积专家+子集采样混合目标）与推理（多条件组合引导采样）方案。相比同期最相近的 UniPi（视频生成→逆动力学的两阶段管线，但未做语言组分解耦，泛化到新指令组合能力弱）和 HiP（依靠拼接多个独立专家基础模型完成分层规划，而非在同一分布分解层面做组合），RoboDreamer 用一套统一的评分函数分解框架同时覆盖了两者的能力边界，且进一步自然地扩展到目标图像、目标草图等多模态条件的即插即用组合，这是其相对新颖之处。RoboDreamer w/o 消融（无组合解析，Table 1 中 Unseen 68.8% vs 完整版 81.3%）也提供了直接证据说明性能提升主要来自组合式建模而非单纯扩大模型/数据规模。

**局限与开放问题**：(1) 论文的组合泛化实验只在 RT-1 这一单一数据集、单一机器人本体（RT-1 移动操作臂）上验证，尚未在更多样化的机器人平台或跨本体设置下检验组合泛化能力是否可迁移；(2) 文本解析依赖基于句法成分（动词短语/介词短语）的规则切分，这一设计隐含假设了任务指令服从"动作+空间关系"这种相对简单的语法结构，对于更复杂的时序性指令（如包含条件、否定、多步因果的自然语言）解析方案的适用性未做讨论，也未与更通用的大语言模型指令解析方案做对比；(3) 组合式引导采样在多个条件同时启用时的引导权重 $w$ 如何设置、组分数目增多后是否会出现评分冲突或生成质量下降，论文未做系统的敏感性分析；(4) RLBench 实验刻意限制为单前视相机、不加目标图像，与作者自己指出的"缺乏多相机支持"这一局限相互印证，绝对成功率（49.3%）与同期基于流匹配/大规模预训练的策略学习方法（如 Diffusion Policy、$\pi_0$ 一类工作）相比仍有明显差距，这类视频生成中介的规划范式在样本效率与最终精度上能否追平直接端到端策略学习，仍是开放问题；(5) 逆动力学模型误差的闭环重规划策略缓解了误差累积，但论文未报告重规划频率对整体推理延迟与成功率的权衡曲线，这类视频扩散规划管线固有的"生成慢、需要多步去噪"的推理效率问题在 RoboDreamer 中未被专门讨论或优化。

## 参考

1. Du, Y., Yang, M., Dai, B., Dai, H., Nachum, O., Tenenbaum, J. B., Schuurmans, D., Abbeel, P. *Learning Universal Policies via Text-Guided Video Generation (UniPi)*. arXiv:2302.00111, 2023（UPDP 形式化与视频生成+逆动力学执行范式的直接来源，也是最相近的对比基线）。
2. Ko, P.-C., Mao, J., Du, Y., Sun, S.-H., Tenenbaum, J. B. *Learning to Act from Actionless Videos through Dense Correspondences (AVDC)*. 2023（本文视频扩散主干代码库基础，也是主要对比基线）。
3. Ajay, A., Han, S., Du, Y., Li, S., Gupta, A., Jaakkola, T., Tenenbaum, J., Kaelbling, L., Srivastava, A., Agrawal, P. *Compositional Foundation Models for Hierarchical Planning (HiP)*. arXiv:2309.08587, 2023（组合多个专家基础模型做长时程规划的对比方法，与本文"在同一学习分布内做组合"路线形成对照）。
4. Liu, N., Li, S., Du, Y., Torralba, A., Tenenbaum, J. B. *Compositional Visual Generation with Composable Diffusion Models*. ECCV 2022（乘积专家式扩散组合、评分函数可加性的理论基础，本文式 (1)(2) 的直接技术源头）。
5. James, S., Ma, Z., Arrojo, D. R., Davison, A. J. *RLBench: The Robot Learning Benchmark & Learning Environment*. IEEE Robotics and Automation Letters, 2020（本文机器人规划实验的评测基准）。
