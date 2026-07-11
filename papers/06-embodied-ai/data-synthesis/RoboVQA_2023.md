# RoboVQA：面向机器人的多模态长时程推理

> **论文**：*RoboVQA: Multimodal Long-Horizon Reasoning for Robotics*
>
> **作者**：Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Karol Hausman, Brian Ichter, Yuan Cao, et al.
>
> **机构**：Google DeepMind
>
> **发布时间**：2023 年 11 月（arXiv 2311.00899）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2311.00899) | [PDF](https://arxiv.org/pdf/2311.00899)
>
> **分类标签**：`可扩展数据采集` `跨具身数据` `长时程规划` `视频语言模型` `干预率评测`

---

## 一句话总结

提出自下而上、跨具身（机器人 / 人体单臂 / 人体+抓取工具）的长时程任务数据采集方案，用事后众包分割标注和零成本的任务增强自动生成 10 类 VQA 问答对，采集吞吐相对传统自上而下逐步采集提升 2.2 倍（机器人）到 13.8 倍（人体长时程 vs 机器人逐步）；发布 829,502 条(视频,文本)对的 RoboVQA 数据集，并用认知/物理干预率协议证明微调后的 383M 视频语言模型 RoboVQA-VideoCoCa 比零样本 SOTA VLM（PaLM-E）认知干预率低 46%。

## 一、问题与动机

- 高层具身推理（long-horizon planning/QA）在真实环境中仍是难题：论文在 3 栋真实办公楼里对 SayCan+PaLM(540B)、Grounded Decoding+PaLM(540B)、PaLM-E(12B, zero-shot) 等 SOTA 方法做实测，规划任务的认知干预率普遍高达 81%~99%，说明现有 VLM 在真实部署分布下远未可用。
- 传统数据采集是自上而下的：研究者预先固定一张小任务清单，机器人逐步（step-by-step）执行并反复复位场景/机器人，成本高、任务分布窄，与真实用户请求的分布不匹配。
- 核心问题：能否设计一种自下而上、众包驱动、跨具身的采集范式，在同等预算下获得更高吞吐、更高多样性的 grounded 数据；以及给定采集预算，机器人和(更便宜的)人体数据该如何配比。

## 二、核心方法

**采集流程（四步）**：(1) 众包/远程操作员提出长时程用户请求（如"帮我冲杯咖啡"）；(2) 人类操作员遥操作机器人、或直接用人体（单臂 / 手持抓取工具）连续完成整个任务并录像，过程中不做任务级复位；(3) 事后由众包标注员对录像做时序切分，给每个 medium-horizon 片段配任务指令（hindsight labeling），标注过程可并行、不占用采集预算内的时间；(4) 任务增强（免费）：利用已知的过去/未来片段序列，在时间轴不同语义位置自动采样，生成 10 类问答对（planning、planning with context、remaining steps、future prediction、past description、success 正/负例、discriminative affordance 正/负例、generative affordance），思路上类似 chain-of-thought，把长时程目标拆成显式的中间步骤监督。

**吞吐优势**：由于不需要按步骤复位场景/机器人，长时程连续采集比逐步采集更高效；标注和视觉问答生成都能后置且免费获得。

**数据规模**：3 栋办公楼、3 种具身，共 238 小时视频（约 10 天），5,246 条长时程 episode（平均 102 秒）+ 92,948 条 medium-horizon episode（平均 14 秒），产出 829,502 条(视频,文本) VQA 对、29,520 条独立指令（26,798 条 medium-horizon + 2,722 条 long-horizon）。语言侧统计（LLM 自动抽取，含冗余）约 2,862 个物体、680 个动作/技能、3,322 个位置、901 个属性描述。

**模型 RoboVQA-VideoCoCa**：基于 VideoCoCa（在 CoCa 上扩展到视频，编码器-解码器架构，融合类 CLIP 的对比预训练与类 SimVLM 的生成式预训练），383M 参数，先用图文 caption 任务的原始 checkpoint 初始化，再在 RoboVQA 视频文本数据上微调；推理时最多以 16 帧视频作为条件输入。

**评测协议——干预率（intervention rate）**：把真实部署拆成认知干预率（cognitive，高层文本/规划域，人工在模型给出错误答案时介入纠正）与物理干预率（physical，底层运动执行域），二者平均得到单一可比指标；这样可以让任务始终被纠正到完成，同时量化模型在真实闭环里"离能用还差多远"。

用大白话说：与其花高成本雇机器人反复摆拍一小撮预设任务，不如放开让真人提各种长任务需求，用更便宜的人体/遥操作机器人一次性录完整个任务过程，回头再用众包把视频切段配文字标签；这样同样预算能采到更多、更贴近真实分布的数据，还能顺手免费"薅"出十种不同格式的监督信号，而不需要额外采集成本。

## 三、关键结果

**(1) 采集吞吐量对比**（Fig.2，秒/medium-horizon 步骤，越低越好）：

| 采集方式 | 每步耗时 (s) | 相对 Robot+Step-by-step 加速比 |
|---|---|---|
| Robot + Step-by-step | 49.5 | 1x（基线） |
| Robot + Long-horizon | 22.8 | 2.2x |
| Human + Step-by-step | 7.1 | 6.9x |
| Human + Long-horizon | 3.6 | 13.8x |

**(2) 规划基准 + 干预率**（Fig.5，评测一：854 个规划步骤，机器人/人体各占 50%，离线预录制视频）：

| 模型 | 规模 | 帧数 | 认知干预率 | 平均干预率 |
|---|---|---|---|---|
| SayCan+PaLM | 540B | 1 | 98.8% | 99.4% |
| Grounded Decoding+PaLM | 540B | 1 | 95.5% | 97.8% |
| PaLM-E (zero-shot) | 12B | 1 | 81.4% | 90.7% |
| RoboVQA-VideoCoCa (本文) | 383M | 16 | **44.0%** | 72.0% |

评测二（10 个真实机器人在线长时程任务，人遥操作作为底层策略执行）：PaLM-E zero-shot 认知干预率 78.2%±7.6%，RoboVQA-VideoCoCa 47.67%±9.1%（参数量不到前者 3%，认知干预率反而更低）。评测三（策略 X 全自主控制、1 个真实任务，5 步）：RoboVQA-VideoCoCa 认知干预率 40.0%，物理干预率 0%（narrow/easy 域），平均 20.0%。

**(3) 视频条件 vs 单帧条件**：video VLM 相对 single-image VLM 在全部 VQA 任务上平均错误率降低 19%；分任务看降幅差异很大——discriminative affordance −48%、past description −36%、future prediction −35%、success classification −17%、generative affordance −17%、planning with context −18%、planning −10%、remaining 5 steps 仅 −2%，说明视频条件对需要"看动作/变化过程"而非仅"看静态终态"的任务收益最大。

**(4) 免费任务增强的价值**（Fig.7）：只训练 planning 任务时错误率 77.2%；联合训练全部 10 类任务后 planning 错误率降到 70.9%，尽管 all-tasks 模型能见到的 planning 样本量只有 planning-only 模型的约 1/8，说明零额外采集成本的任务增强能直接提升下游性能。

**(5) 跨具身数据混合的性价比**（Sec.V-B/Fig.13，分别在机器人-only 与人体-only 测试集上评测）：机器人:人体采集成本比 = 1:1、总预算相同时，"机器人250k+人体250k"混合数据在机器人测试集上错误率(62.4%)与全 500k 机器人专属数据(62.7%)基本持平，但在人体测试集上显著更低(53.9% vs 67.0%)；成本比 = 4:1 时，"机器人62k+人体250k"混合数据在机器人测试集上错误率(65.3%)接近机器人专属 125k 数据集(63.5%)，人体测试集则明显更低(51.1% vs 68.7%)。结论：混合廉价人体数据不损害机器人域表现，还能大幅扩展模型的跨具身通用性；论文据此认为掺入人体数据是"零成本改善"策略。

## 四、评价与展望

**优点**：数据采集范式的方法论贡献扎实——把"自上而下固定任务表"翻转为"自下而上众包长时程请求 + 事后分割标注 + 免费任务增强"，用吞吐/成本数字（2.2x / 6.9x / 13.8x）量化了范式切换的收益；提出的认知/物理干预率协议填补了"长时程规划模型真实部署到底多可用"这一评测空白，比纯 offline VQA 准确率更贴近工程落地；跨具身成本-性能权衡分析（Fig.13）是相对同期工作（SayCan、PaLM-E）少见的亮点，直接回答了"给定预算该采多少人体数据、多少机器人数据"这一工程问题。

**局限**（原文第七节 Limitations 明确指出）：为避免重复/过简任务，过滤掉了超过 5 个相同 medium-horizon 步骤的 episode，可能引入采样偏差；论文未与纯人体数据集/基准（如 Ego4D、EpicKitchens）做正面效果对比，人体到机器人迁移的边界仍未定量刻画；采集场景局限在 3 栋办公楼室内环境，多样性相对家庭/工业场景有限；RoboVQA-VideoCoCa 本身只输出高层文本（规划/问答），不直接输出动作，需搭配底层策略才能落地，端到端全自主评测（评测三）目前也只覆盖 1 个任务的 narrow/easy 域，通用性未充分验证；认知干预率依赖人工在环判定，存在标注者主观性和人力成本，指标本身的可扩展性有待讨论。

**与其他公开工作的关系**：RoboVQA 与 SayCan、PaLM-E、Grounded Decoding 属于同一条"用（大）语言/视觉语言模型做机器人高层规划"的技术路线，但本文定位更偏"数据与评测基础设施"——把这些方法当基线，指出其零样本在真实环境下干预率普遍在 80%~99%，进而论证还需要更多、更贴近部署分布的 grounded 数据。相较 Ego4D、EpicKitchens 等纯人类第一视角数据集，RoboVQA 的差异化在于同一环境下同时采集人类和机器人两种具身的数据，直接面向缩小 human-to-robot 域差这一目标，方向上与后续大量利用人类视频/遥操作数据做跨具身预训练的工作一脉相承。开放问题包括：如何量化人体到机器人迁移的上限、能否把该采集范式扩展到家庭等非结构化场景、以及干预率协议能否进一步自动化以降低评测的人力成本。

## 参考

[1] Driess et al., *PaLM-E: An Embodied Multimodal Language Model*, arXiv:2303.03378, 2023.
[2] Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (SayCan), arXiv:2204.01691, 2022.
[3] Huang et al., *Grounded Decoding: Guiding Text Generation with Grounded Models for Robot Control*, arXiv:2303.00855, 2023.
[4] Yan et al., *VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners*, arXiv:2212.04979, 2022.
[5] Grauman et al., *Ego4D: Around the World in 3,000 Hours of Egocentric Video*, CVPR 2022.
