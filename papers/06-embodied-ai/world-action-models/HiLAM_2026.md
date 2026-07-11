# HiLAM：层级潜在动作模型

> **论文**：*Hierarchical Latent Action Model*
>
> **作者**：Hanjung Kim, Lerrel Pinto, Seon Joo Kim
>
> **机构**：Yonsei University、New York University
>
> **发布时间**：2026 年 03 月（arXiv 2603.05815）
>
> **发表状态**：ICLR 2026 Workshop（已录用为 workshop 论文）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.05815) | [PDF](https://arxiv.org/pdf/2603.05815)
>
> **分类标签**：`层级隐动作模型` `Latent Action Model` `技能发现` `H-Net动态分块` `LIBERO`

---

## 一句话总结

HiLAM 在预训练的帧级 Latent Action Model（IDM/FDM,复用自 UniSkill）之上叠加一层 H-Net 式动态分块(dynamic chunking),把逐帧隐动作序列无监督地聚合成变长、语义连贯的高层隐技能(latent skill),再据此训练"高层选技能、低层出动作"的分层策略;在 LIBERO-Long 上仅用 10% 专家演示微调即可达到 45% 成功率(BAKU 基线仅 23%,接近翻倍),50% 演示即追平 BAKU 用满 100% 数据的水平(84%),100% 演示时达到 94%(BAKU 87%)。

## 一、问题与动机

Latent Action Model(LAM)通过反向/正向动力学模型(IDM/FDM)从无动作标注的观测视频中提取隐动作,已被用于 VLA 预训练(LAPA)、跨具身动作迁移(UniSkill)、游戏世界模型交互(Genie)等场景。但现有 LAM(LAPO、Genie、LAPA、UniVLA、UniSkill、CLAM 等)几乎都局限于短时帧间转移,只捕捉低层运动模式,忽略了无标注视频中普遍存在的长时程、高层技能结构。

已有的技能发现工作提供了两条思路,但都有局限:

- **固定窗口/预定义技能集**(SPiRL、SkiLD 用自编码器从定长动作片段学技能先验,BUDS 基于预定义技能原语对无分段演示做聚类):真实数据中同一任务的执行速度、技能时长差异很大,把技能强行塞进固定长度窗口,会让两条表达相同行为的轨迹被映射成截然不同的技能表示。
- **用语言定义技能边界**(Hi Robot 用高层 VLM 把用户指令拆成子指令):语言提供的是任务分解信号,而非从运动本身发现的边界,是与运动建模互补而非替代的信息源。

核心问题:能否在不依赖动作标签、不预设技能时长或技能集合数量的前提下,从无标注视频中以数据驱动、长度自适应的方式发现语义连贯的高层技能?HiLAM 的答案是把 H-Net(为语言/DNA 序列设计的动态分块架构)迁移到隐动作序列上。

## 二、核心方法

### 2.1 问题建模

给定长度为 $T$ 的无标注视频 $V$,先用预训练 IDM 提取相隔固定时间间隔 $k$、两帧 $I_t$ 与 $I_{t+k}$ 之间的逐帧隐动作序列 $\{z^l_1,\dots,z^l_{T-k}\}$,再把该序列压缩为长度可变的高层隐技能序列 $\{z^h_1,\dots,z^h_S\}$($S < T-k$,且 $T$、$S$ 均可变,适配不同时长的轨迹与技能)。下游控制时用分层策略:高层策略在每个决策步观察当前状态 $o_t$ 与任务指令 $l$,预测目标隐技能 $z^h_t \sim \pi^h(z^h_t \mid o_t,l)$;低层策略在此技能条件下生成可执行动作 $a_t \sim \pi^l(a_t \mid o_t,z^h_t)$。

大白话:高层策略先决定"现在该做哪个技能",低层策略再把这个技能翻译成一步步的具体动作,类似人做家务时先决定"洗碗"这个子任务,再具体到每一次手部移动。

### 2.2 动态分块机制(改造自 H-Net)

HiLAM 直接采用 H-Net(Hwang et al., 2025)的动态分块结构,把隐动作序列自动切成不定长片段。第 $s$ 层输入序列 $\mathbf{z}^s=(z^s_1,\dots,z^s_{L^s})$,编码器 $\mathcal{E}^s$ 把每个 token 映射为特征 $h^s_t \in \mathbb{R}^d$,由归一化后的 query/key 特征 $\hat{\mathbf{q}}^s_t,\hat{\mathbf{k}}^s_t$ 计算边界概率并二值化为边界指示 $b^s_t$:

$$
p^s_t = \begin{cases} 1, & t=1 \\ \dfrac{1}{2}\big(1-\hat{\mathbf{q}}^{s\top}_{t-1}\hat{\mathbf{k}}^s_t\big), & t>1 \end{cases}
\qquad\qquad
b^s_t = \mathbb{1}_{\{p^s_t \ge 0.5\}}
$$

大白话:把相邻两个 token 的归一化特征做点积当余弦相似度,越不相似(点积越接近 −1、$p^s_t$ 越接近 1)就越可能是一个新技能的起点,越相似就越可能仍在同一技能内部——完全靠特征本身的连续性判边界,不需要人为设定技能时长或数量。

取边界索引集合 $\mathcal{I}^s=\{t\mid b^s_t=1\}$(按升序排列),对特征做 Chunk(降采样,只保留边界处的 token 作为片段摘要),主网络 $\mathcal{M}^s$ 处理变短的 chunk 级序列,解码器 $\mathcal{D}^s$ 再 DeChunk 依同一边界模式还原回原长度:

$$
h^s=\mathcal{E}^s(z^s), \quad z^{s+1}=\mathrm{Chunk}(h^s;b^s), \quad \hat z^{s+1}=\mathcal{M}^s(z^{s+1}), \quad \hat z^{s}=\mathrm{DeChunk}\big(\mathcal{D}^s(\hat z^{s+1});b^s\big)
$$

堆叠多层这样的 encode–chunk–main–dechunk,层数越高处理的序列越短、对应时间跨度越长,由此自然形成一个层级表示。HiLAM 默认用两层 H-Net,取第 2 层(stage $s=2$)的表示 $\mathbf{z}^2$ 作为最终的隐技能表示 $\mathbf{z}^h$。

### 2.3 训练目标

在最底层(stage 0),HiLAM 以 next-token prediction 方式预测下一个隐动作,总损失为三项加权和:

$$
\mathcal{L} = \mathcal{L}_{\text{latent}} + \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\text{ratio}}\mathcal{L}_{\text{ratio}}
$$

- $\mathcal{L}_{\text{latent}}$：预测隐动作与目标隐动作之间的逐元素 $\ell_1$ 损失,对应 next-latent 预测任务;
- $\mathcal{L}_{\text{rec}}$：用预训练 FDM(同样来自 UniSkill)把预测出的隐动作解码为未来帧,与真实未来帧计算重建误差,以约束隐动作保留"动作应有"的运动语义;
- $\mathcal{L}_{\text{ratio}}$：H-Net 自带的分块正则项,防止边界模式退化(如全 0 或全 1),控制平均片段长度。

大白话:单纯做 next-latent 预测容易让模型学出与真实物理动态无关的捷径表示,再加一条"能否用预测出的隐动作重建下一帧画面"的约束,逼模型把有实际运动含义的信息保留在隐技能表示里。

### 2.4 隐技能提取与分层策略学习

训练完成后,把第 $s$ 层降采样表示通过"展开边界" $\bar b^s \in \{0,1\}^T$ 映射回原始帧率:每个时间步的片段 ID 由边界的累加和确定,$k^s_t=\sum_{\tau=1}^t \bar b^s_\tau$,$\bar z^s_t=z^s_{k^s_t}$,同一片段内所有帧共享同一个隐技能表示,记 $\bar{\mathbf z}^s=z^h$ 用于下游策略学习。

随后分两阶段训练高层策略 $\pi^h$ 与低层策略 $\pi^l$(二者均基于 BAKU 架构,T5 作语言编码器):

- **预训练**：在大规模无动作视频上,以 HiLAM 抽取出的 $z^h_t,z^l_t$ 为伪标签,高层策略预测隐技能 $\hat z^h_t \sim \pi^h(z^h_t\mid o_t,l)$,低层策略在预测技能条件下预测隐动作 $\hat z^l_t \sim \pi^l(z^l_t\mid o_t,\hat z^h_t)$。因为目标都来自视频自身提取的伪标签,预训练可以自由混用人类视频与机器人视频。
- **微调**：冻结高层策略 $\pi^h$,只在目标域带真实动作标签的数据上微调低层策略,把隐动作空间映射到真实动作空间,输出可执行动作 $\hat a_t \sim \pi^l(a_t\mid o_t,\hat z^h_t)$。

由于复用预训练 LAM 做隐动作抽取(而非端到端训练整个层级结构),长时程轨迹的编码在训练算力上是高效的。

## 三、实验结果

**数据与设置**：人类视频用 Something-Something V2(短片段、物体中心动作),机器人视频用 DROID(Franka)与 BridgeV2(WidowX-250)。隐动作 tokenizer(IDM)与 FDM 直接复用 UniSkill 的预训练权重。下游评测用 LIBERO 基准的四个子集——LIBERO-Spatial(空间推理)、LIBERO-Object(物体泛化)、LIBERO-Goal、LIBERO-Long(最难,多阶段长时程任务),每个子集 10 个任务、每任务 50 条专家演示;预训练与微调默认各跑 100k 梯度步。

**四子集整体表现**：HiLAM 在 Spatial、Object、Goal、Long 四个子集上均一致优于 BAKU 基线(Haldar et al., 2024),验证了从无标注视频中学到的时序延展技能对下游策略学习普遍有效,其中 LIBERO-Long 上的提升最明显。

**LIBERO-Long 数据效率(核心结果)**：

| 微调演示比例 | BAKU 成功率 | HiLAM 成功率 |
|---|---|---|
| 10% | 23% | 45% |
| 50% | — | 84%(与 100% 数据训练的 BAKU 相当) |
| 100% | 87% | 94% |

即 HiLAM 仅用 10% 演示就把成功率从 23% 提升到 45%,接近翻倍;仅用 50% 演示就追平 BAKU 用满全部演示的水平,体现出预训练所得高层技能显著提高了下游微调的数据效率。

**消融实验(Table 1,均在 LIBERO-Long 上)**：

| 方法 | 预训练数据 | 技能条件层 | 动作条件层 | 成功率 |
|---|---|---|---|---|
| BAKU(非层级) | Robot | – | $\bar z^0$ | 0.87 |
| BAKU(非层级) | Robot | – | $\bar z^2$ | 0.81 |
| BAKU(非层级) | Human | – | $\bar z^0$ | 0.91 |
| BAKU(非层级) | Human | – | $\bar z^2$ | 0.87 |
| HiLAM(无预训练) | – | – | – | 0.67 |
| HiLAM | Robot | $\bar z^1$ | $\bar z^0$ | 0.90 |
| HiLAM | Robot | $\bar z^2$ | $\bar z^0$ | 0.90 |
| HiLAM | Robot | $\bar z^2$ | $\bar z^1$ | 0.87 |
| HiLAM | Human | $\bar z^1$ | $\bar z^0$ | 0.89 |
| **HiLAM**(默认) | **Human** | $\bar z^2$ | $\bar z^0$ | **0.94** |
| HiLAM | Human | $\bar z^2$ | $\bar z^1$ | 0.89 |

关键结论：

- 人类视频预训练总体略优于机器人视频预训练;
- 用第 2 层(stage $s=2$)特征做高层技能条件、第 0 层(原始逐帧隐动作)做低层动作条件效果最好——第 2 层由更深编码器产生,语义聚类更充分,能提供更长时程的上下文;
- 把隐动作条件直接接到非层级的 BAKU 上(flat policy)也能带来提升(0.81~0.91),但仍明显落后于层级策略的 0.94,说明收益不能只归因于隐动作条件本身,层级结构是必要的;
- 只在目标任务上训练、不做大规模预训练,成功率骤降到 0.67,说明收益主要来自大规模无标注视频预训练,而非策略架构本身。

**定性结果**：技能边界发现(Figure 4)中,在一段"移向碗→拿起碗→放下碗"的操作视频上,HiLAM 在完全无监督(无技能标签、无真实动作)条件下自动切出与人类语义标注一致的连续技能片段;未来帧预测(Figure 5)中,即使预测出的隐动作是从历史隐动作序列递归预测出来的,用预训练 FDM 解码后仍能生成与真实运动一致的未来帧,验证预测出的隐动作保留了有意义的运动信息。

## 四、局限性

论文在 Conclusion and Limitations 中明确指出：

- 实验主要在仿真环境(LIBERO)中进行,尚未在真实机器人上做验证性实验,真实世界实验会进一步证明方法的有效性;
- 为保证时间维度建模的计算效率,HiLAM 复用了一个冻结的预训练 IDM 来抽取底层隐动作,而非端到端训练整个层级架构;端到端联合训练底层运动模式与高层技能可能带来更深的联合理解,但本文未验证这一点;
- 当前方法只从隐动作序列本身发现技能,未引入语言信号;论文认为运动线索与语言指令是互补而非替代关系(模仿运动更适合家具组装这类复杂任务,语言指令能以更少约束提升泛化性),融合二者是有前景但尚未开展的未来工作。

## 五、评价与展望

**优点**：

- 把 H-Net 的动态分块机制从语言/DNA 序列建模迁移到隐动作序列上,是一个简洁但切中要害的架构选择,直接解决了此前技能发现工作(SPiRL/SkiLD 依赖定长动作片段、BUDS 依赖预定义技能原语)必须预设技能时长或数量的痛点——分块边界完全由特征相似度驱动,理论上能随视频内容自适应伸缩。
- 复用预训练 LAM(UniSkill 的 IDM/FDM)做底层隐动作抽取,而非从头端到端训练整套层级结构,是长视频训练在计算上可负担的务实折中。
- 消融设计比较完整,同时验证了预训练数据来源、条件层选择、层级 vs 非层级策略、有无预训练四个维度,清楚拆分出"层级结构的收益"与"隐动作条件本身的收益"。
- LIBERO-Long 上的数据效率结果(10% 演示下 23%→45%,接近翻倍)是本文最有说服力的证据,直接支持了"高层技能能为长时程任务提供更好的时序结构先验"这一核心主张。

**不足与开放问题**：

- 这是一篇 ICLR 2026 workshop 论文,篇幅和实验规模都偏小:下游评测只在 LIBERO 一个仿真基准上进行(四个子集共 40 个任务,每任务 50 条演示),尚未在真实机器人或更大规模基准(如 CALVIN、RoboCasa)、更大规模数据集部署上验证,论文自己在 Limitations 中也承认了这一点。
- 部分结果(如四子集整体对比、50% 数据下 BAKU 的具体成功率)论文正文只给出图表和定性描述,未给出精确数字,量化说服力打了折扣。
- 高层策略 $\pi^h$ 在微调阶段被完全冻结,只微调低层策略——若预训练技能空间与下游任务的技能分布存在较大差异,模型缺乏机制去纠正高层的技能选择错误,论文未讨论这类失配情形下的失败模式。
- 与同期公开工作的关系:UniVLA(Bu et al., 2025)用任务中心化方式过滤隐动作中的任务无关信息,LAOM(Nikulin et al., 2025)引入监督信号解决"分心物体"(distractor)问题,二者关注的是隐动作表示"质量"维度;HiLAM 解决的是正交的另一个维度——时间尺度上的层级性。三者理论上可以叠加组合,但本文未做交叉验证,是一个自然的后续方向。
- 边界发现机制继承自 H-Net 的"归一化特征点积 + 0.5 阈值"启发式,论文没有讨论该阈值对不同任务/数据集的敏感性,也未报告分块出的平均技能长度分布,在可解释性和调参鲁棒性方面留有开放问题。
- 论文自己指出的"语言 + 运动线索融合"方向是合理的下一步:纯运动驱动的技能发现容易把执行速度、路径的微小差异也切分成不同技能,而语言标注的任务分解可以提供更稳定的语义锚点,二者的有效融合仍是未解决的问题。

## 参考

- Hwang, S., Wang, B., Gu, A. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling. arXiv:2507.07955, 2025.(H-Net,HiLAM 分块机制的直接来源)
- Kim, H., Kang, J., Kang, H., Cho, M., Kim, S. J., Lee, Y. UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations. CoRL 2025.(HiLAM 复用其 IDM/FDM 做隐动作抽取)
- Ye, S. et al. Latent Action Pretraining from Videos (LAPA). ICLR 2025.
- Bu, Q. et al. Learning to Act Anywhere with Task-Centric Latent Actions (UniVLA). RSS 2025.
- Haldar, S., Peng, Z., Pinto, L. BAKU: An Efficient Transformer for Multi-Task Policy Learning. NeurIPS 2024.(HiLAM 分层策略的基座架构)
