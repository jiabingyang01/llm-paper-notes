# FlowVLA：基于视觉思维链的运动推理视觉-语言-动作模型

> **论文**：*FlowVLA: Visual Chain of Thought-based Motion Reasoning for Vision-Language-Action Models*
>
> **作者**：Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Tianran Zhang, Wenxuan Song, Jiayi Chen, Xinhu Zheng, Hesheng Wang, Haoang Li
>
> **机构**：香港科技大学（广州）（HKUST(GZ)）；上海交通大学
>
> **发布时间**：2025 年 08 月（arXiv 2508.18269，v3 于 2025 年 10 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2508.18269) | [PDF](https://arxiv.org/pdf/2508.18269)
>
> **分类标签**：`VLA` `世界模型` `视觉思维链` `光流` `样本效率`

---

## 一句话总结

FlowVLA 把 VLA 世界模型预训练中的"下一帧直接预测" $v_t \to v_{t+1}$ 改写为显式的"运动先行"思维链 $v_t \to f_t \to v_{t+1}$（$f_t$ 为光流）,用同一个 VQ-GAN 词表把外观帧与光流图统一 token 化,由单一自回归 Transformer 交替预测,在 LIBERO 上均分达到 88.1%（较此前最强的 UniVLA 84.0% 提升约 4 个点,长时序子集提升尤为显著）,在 SimplerEnv-WidowX 上均分 74.0%,并且在真机数据受限（50% 数据)场景下峰值成功率比基线高 55%(0.48 对 0.31)。

## 一、问题与动机

当前 VLA 常见的世界模型预训练范式是直接建模 $P_\theta(v_{t+1} \mid v_t, L)$,即用一个大自回归 Transformer 在海量视频上学习"下一帧长什么样"。论文指出这一范式存在根本缺陷:

1. **"像素复制陷阱"**：把世界建模变成高维像素空间的回归问题,优化目标存在大量平凡局部最优——模型很容易走捷径,直接复制输入帧中的静态背景像素来降低重建误差,导致长时序预测模糊、不一致、物理上不合理。
2. **缺乏显式因果结构**：$v_t \to v_{t+1}$ 的直接映射学到的只是像素配置随时间的相关性,而非真正的物理因果关系,导致模型在视觉线索变化但物理规律不变的分布外场景中泛化能力差。
3. **预训练/微调之间的域鸿沟**：世界模型预训练学到的是被动的观测知识,而策略微调需要主动的、面向动作的知识,这种错位导致下游微调收敛慢、样本效率低。

受大语言模型中 Chain-of-Thought 提示能通过生成中间推理步骤提升推理能力的启发,论文提出**视觉思维链(Visual Chain of Thought, Visual CoT)**：不再让模型从 $v_t$ 到 $v_{t+1}$ 做单步、无推理的跳跃,而是强制模型先显式预测中间的物理动态——描述每个像素如何运动的稠密光流场 $f_t$,再以此为条件生成最终的未来帧。

## 二、核心方法

**Visual CoT 的概率分解。** 用链式法则把联合分布 $P(v_{t+1}, f_t \mid v_t, L)$ 分解为

$$P(v_{t+1}, f_t \mid v_t, L) = \underbrace{P(v_{t+1} \mid f_t, v_t, L)}_{\text{外观生成}} \times \underbrace{P(f_t \mid v_t, L)}_{\text{运动推理}}.$$

大白话说:模型先想清楚"东西会怎么动"(光流),再据此把"动完之后画面长什么样"画出来,而不是一步到位直接猜下一帧——这给模型引入了一个强的归纳偏置,让预测过程带有物理因果结构,且天然更贴近下游动作生成所需的"运动导向"知识。

**统一的外观-运动 token 化。** 为了不引入任何任务专用架构组件,FlowVLA 把光流也编码成图像格式,与 RGB 帧共用同一个预训练 VQ-GAN tokenizer。具体做法是把 2 通道的光流位移场 $(u,v)$ 按照 VideoJAM 的方案转成 3 通道 RGB 色轮图：运动方向映射到色相 $\alpha=\arctan2(v,u)$,运动幅度 $m=\sqrt{u^2+v^2}$ 映射到饱和度/明度,并做非线性归一化以避免大范围位移下细微运动被压制:

$$m_{\text{norm}} = \min\!\left(1.0,\ \frac{m}{\sigma \cdot \sqrt{H^2+W^2}}\right), \qquad \sigma = 0.15.$$

大白话说:把"每个像素往哪个方向、跑多快"编码成一张彩色图（方向定颜色、速度定深浅),这样光流图和普通照片长得一样,就能塞进同一个图像 tokenizer,不用给模型额外造一个"光流专用模块"。光流本身由预训练的 RAFT 离线计算得到。

**Stage 1：世界模型的 Visual CoT 预训练。** 输入序列交替排列语言指令、外观帧与光流 token：$S_{\text{wm}} = \{L_{\text{instr}}, v_0, f_0, v_1, f_1, \dots, v_T, f_T\}$。用标准 decoder-only Transformer（骨干沿用 8.5B 参数的 Emu3 与 UniVLA 架构)做下一 token 预测,损失是运动推理步与外观生成步交叉熵之和：

$$\mathcal{L}_{\text{WM}} = \sum_{t=0}^{T-1} \Big( \mathcal{L}_{\text{CE}}\big(f_t \mid S_{<v_{t+1}}\big) + \lambda \cdot \mathcal{L}_{\text{CE}}\big(v_{t+1} \mid S_{<v_{t+1}}, f_t\big) \Big), \quad \lambda=1.0.$$

大白话说:模型在每一步都先被要求"预测运动"再被要求"预测画面",两部分损失都算,确保中间的"思考步骤"$f_t$ 真的被监督成物理正确的光流,而不是任由模型生成任意的中间表示。

**Stage 2：动作预测微调。** 用 Stage 1 预训练权重初始化,输入序列改为交替的观测与动作 $S_{\text{policy}} = \{L_{\text{instr}}, v_0, a_0, v_1, a_1, \dots\}$,动作用 FAST tokenizer 离散化,微调损失只计算在动作 token 上。这样 Stage 1 学到的物理动态知识可以直接迁移给动作预测,而不需要重新学习底层动力学。

选择光流而非 3D 姿态、边界框等以物体为中心的表示,原因有二：一是获取这类表示通常依赖专门的、在人工标注数据上训练的上游检测/姿态估计模型,限制了可扩展性；二是稀疏表示无法捕捉非刚体运动或复杂交互动态,而光流是稠密、通用、独立于物体检测器的连续运动信号。

## 三、关键结果

**LIBERO(4 个子集均分成功率 %,表 1)**：FlowVLA 在 Spatial/Object/Goal/Long 四个子集上全面超越已发表方法，均分 **88.1**（Spatial 93.2、Object 95.0、Goal 91.6、Long 72.6),对比同为"带世界模型"类别的 UniVLA（84.0)、CoT-VLA（81.1)、WorldVLA（79.1),以及不带世界模型的 pi0-FAST（85.5)、ThinkAct（84.4)、OpenVLA（76.5)。提升在 Long-horizon 子集上最明显(72.6 对 CoT-VLA 的 69.0),印证显式运动推理对长时序规划的收益。

**SimplerEnv-WidowX(域偏移鲁棒性,表 2)**：FlowVLA 均分 **74.0**（Put Spoon 70.8、Put Carrot 62.5、Stack Block 62.5、Put Eggplant 100.0),显著超过 UniVLA 官方权重复现结果（65.6)、Embodied-R1（56.2)、ThinkAct（43.8)、SpatialVLA（42.7)等基线。

**真机实验(AgileX Cobot 双臂平台,表 3)**：四项单臂/双臂任务（Stack Bowls、Place Vegetable、Place Bottles、Lift Pot,各评测 25 次)均分 **44.0**,对比 UniVLA（31.0)、OpenVLA（20.0)、ACT（19.0),在需要精细协调的双臂任务（Placing two cola cans into a box、Lifting a pot using both arms）上提升最明显。

**样本效率(图 6)**：与基线 UniVLA 对照,在全量数据下 FlowVLA 仅用三分之一训练步数（2k vs. 6k）即达到基线的峰值成功率(0.64),并最终收敛到更高的 0.73；在 50% 数据的低数据域,优势进一步放大——FlowVLA 峰值成功率比基线高 55%(0.48 对 0.31),且仅用 1k 步就超过基线用 3k 步才达到的峰值。

**消融(LIBERO-10,表 4)**：完整模型 73.0%。去掉整个 Visual CoT 结构（退化为 UniVLA 基线)降至 64.0%(掉 9 个点,证明显式"先想动作再想画面"的推理过程是主要增益来源)；保留交替序列结构但去掉光流监督损失(允许中间表示任意坍缩)降至 69.5%(说明直接监督对防止中间表示退化至关重要)；把交替序列改为"先排完所有帧、再排完所有光流"的分组序列(v0,v1,...,f0,f1,...)则性能严重崩溃至 49.4%,因为模型不再能以因果、前瞻的方式用预测出的 $f_t$ 指导 $v_{t+1}$ 生成，证实了逐步交替的因果链结构本身是有效性的关键。

此外,论文在 Bridge V2 数据集上做了定性对比：不带运动推理的基线世界模型会出现机械臂突然消失、物体运动不连贯等"物理不合理"失败,以及预测画面看似合理但实际运动方向与语言指令不符的"语义不一致"失败；FlowVLA 通过先预测光流再生成画面,同时缓解了这两类问题。

## 四、评价与展望

**优点**：FlowVLA 的核心贡献是把"运动先行"的直觉转化成一个概率上严格的因果分解（式 2),并用一个几乎不增加架构复杂度的方案落地——不新增专用运动编码器或分支,而是复用同一个 VQ-GAN 词表把光流也"伪装"成图像 token,使单一自回归 Transformer 就能吃下外观和运动两种模态。这一设计选择本身即是论文的一个亮点，也是消融实验（分组序列相比交替序列性能腰斩)所验证的关键。在实验设置上,论文严格对齐了 UniVLA、WorldVLA 等同类"带世界模型"基线的两阶段训练范式,使得 Table 1/2 中的对比相对公平；样本效率分析（尤其是低数据域峰值成功率提升 55%）为"预训练学到的动力学知识能显著降低微调阶段样本需求"这一核心论点提供了较有说服力的直接证据。

**与同类工作的关系**：FlowVLA 与 CoT-VLA（同样引入"视觉思维链"概念，但走的是预测关键帧/子目标图像的路线)、WorldVLA（直接联合预测下一帧与下一动作,未显式解耦运动与外观)构成同一技术脉络下的不同实现；相比 MolmoAct 等做深度图/2D 末端执行器轨迹的"中层几何推理",以及 ECoT、ThinkAct 等做文本子目标的"高层语义推理",论文将 FlowVLA 定位为更底层的"像素级物理推理"，二者本质上是互补而非竞争关系。光流方案本身继承自光流估计（RAFT）与 VideoJAM 的运动-外观联合表示思路，将其系统性地引入 VLA 世界模型预训练/微调的两阶段流程是本文的主要贡献。

**开放问题**：论文正文未设置专门的局限性讨论,但从方法与实验设计可以看出几点值得关注的边界。第一,交替排列外观帧与光流帧使自回归序列长度相较纯下一帧预测翻倍,论文未报告这对推理延迟/生成速度的具体影响,而这对策略执行的实时性可能是不可忽视的开销。第二,光流依赖 RAFT 离线预计算,预训练质量上限受限于光流估计器本身在遮挡、大位移、透明/反光物体等场景下的误差，论文未讨论光流误差如何传导到下游动作预测。第三,骨干基于 8.5B 参数的 Emu3，门槛较高，未讨论 Visual CoT 范式在更小规模模型上的可迁移性。第四,SimplerEnv 评测只覆盖 WidowX 一种本体，真机实验的成功率（28%-60%）虽全面超越基线，但绝对数值仍有较大提升空间，长时序双臂任务（Lift Pot 仅 28%）显示复杂协调场景仍是瓶颈。这些都是该范式进一步扩展和验证的可能方向。

## 参考

- Wang et al., *UniVLA: Unified Vision-Language-Action Model*, arXiv:2506.19850, 2025 —— FlowVLA 复用的基础架构与主要对比基线
- Cen et al., *WorldVLA: Towards Autoregressive Action World Model*, arXiv:2506.21539, 2025 —— 同类"世界模型 + VLA"路线的直接竞品，联合预测下一帧与下一动作但未解耦运动/外观
- Zhao et al., *CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models*, CVPR 2025 —— 视觉思维链概念的另一实现，走关键帧预测路线
- Chefer et al., *VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models*, arXiv:2502.02492, 2025 —— FlowVLA 光流转 RGB 色轮编码方案的直接来源
- Teed and Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020 —— 用于离线预计算训练数据光流的模型
