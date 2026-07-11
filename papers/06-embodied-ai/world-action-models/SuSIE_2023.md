# SuSIE：借助预训练图像编辑扩散模型实现零样本机器人操作

> **论文**：*Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models*
>
> **作者**：Kevin Black, Mitsuhiko Nakamoto（共同一作）, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, Sergey Levine
>
> **机构**：加州大学伯克利分校（UC Berkeley）、斯坦福大学、Google DeepMind
>
> **发布时间**：2023 年 10 月（arXiv 2310.10639）
>
> **发表状态**：未录用（预印本，本笔记依据的 PDF 版本未标注录用信息）
>
> 🔗 [arXiv](https://arxiv.org/abs/2310.10639) | [PDF](https://arxiv.org/pdf/2310.10639)
>
> **分类标签**：`图像编辑扩散模型` `子目标生成` `分层控制` `InstructPix2Pix` `目标条件策略` `CALVIN`

---

## 一句话总结

SuSIE（SUbgoal Synthesis via Image Editing）把机器人控制拆成两层：用一个在 InstructPix2Pix 基础上微调的图像编辑扩散模型根据当前图像和语言指令"想象"出下一个子目标画面，再用一个仅依赖视觉的目标条件低层策略执行动作抵达该画面；在 CALVIN 零样本长时序基准上链式完成 5 条指令的成功率达 0.26（此前最优 AugLC 仅 0.05），真实机器人实验中仅用 6 万条轨迹就全面超过用 110 万条以上轨迹训练的 550 亿参数 RT-2-X。

## 一、问题与动机

通用机器人需要像人一样识别、推理从未见过的物体和场景语义（如"拿那支特大号橙色蜡笔"），但机器人操作数据集规模有限，不可能覆盖所有物体和场景组合。已有做法把预训练视觉语言模型直接接入策略主干（如 RT-2），但作者在实验中发现这类方法虽然带来了高层语义理解，却未必能提升低层执行精度——策略仍常出现定位不准、抓取失败等问题。另一类做法（UniPi、HiP 等）用视频生成模型规划完整未来帧序列再提取逆动力学动作，但要求生成的每一帧都物理一致，开源视频模型常出现时序不一致的伪影，反而会干扰低层控制。SuSIE 的出发点是：能否只让生成式模型产出一个"够近、能被低层策略执行到，又够远、能体现语义进展"的单帧子目标，把语义推理留给预训练扩散模型，把精细的视觉—动作对应关系留给一个只需处理局部场景的轻量目标条件策略。

## 二、核心方法

SuSIE 将数据分为三类：仅含语言标注视频片段的 $\mathcal D_l$（无动作，如人类操作视频）、含语言标注和动作的机器人轨迹 $\mathcal D_{l,a}$、仅含动作的无标注机器人数据 $\mathcal D_a$。

**阶段一：子目标生成模型 $p_\theta(\mathbf s_{\text{edited}}\mid \mathbf s_{\text{orig}}, l)$。** 以 InstructPix2Pix 预训练权重为初始化（把机器人子目标预测视为对当前图像按语言指令做"编辑"），在 $\mathcal D_l \cup \mathcal D_{l,a}$ 上微调，训练目标是：

$$\max_\theta\ \mathbb E_{(\tau^n,l^n)\sim \mathcal D_l\cup\mathcal D_{l,a};\ s_i^n\sim\tau^n;\ j\sim q(j\mid i)}\big[\log p_\theta(\mathbf s_j^n\mid \mathbf s_i^n, l^n)\big],$$

其中子目标帧的采样分布 $q(j\mid i)=U\big(j;[i+k_{\min}, i+k_{\max}]\big)$ 从轨迹中均匀采样未来 $k_{\min}$ 到 $k_{\max}$ 步内的帧作为监督目标。用大白话说：让扩散模型学会"看当前画面 + 语言指令，想象几步之后大概会变成什么样子"，而不是生成整段视频，这样既继承了 InstructPix2Pix 的互联网级图像编辑先验，又避免了逐帧生成视频要求的严格物理一致性。

**阶段二：目标条件低层策略 $\pi_\phi(\mathbf a\mid \mathbf s_i,\mathbf s_j)$。** 用标准目标条件行为克隆（GCBC）在 $\mathcal D_{l,a}\cup\mathcal D_a$ 上训练，让策略学会仅根据当前观测和一个邻近的目标画面推断动作，不涉及语言：

$$\max_\phi\ \mathbb E_{\tau^n\sim\mathcal D_{l,a}\cup\mathcal D_a;\ (\mathbf s_i^n,\mathbf a_i^n)\sim\tau^n;\ j\sim U([0,k_{\max}+k_\delta])}\big[\log \pi_\phi(\mathbf a_i^n\mid \mathbf s_i^n,\mathbf s_j^n)\big].$$

用大白话说：低层策略只需要解决"图像 A 怎么走到图像 B"这种局部视觉—动作对应问题，不需要理解任务语义，因此训练简单、样本效率高；$k_\delta$ 给策略预留一点余量，应对生成子目标偶尔超出 $k_{\max}$ 步可达范围的情况。

**测试时闭环执行（Algorithm 1）。** 给定初始状态和语言指令，每隔 $k_{\text{test}}$ 步用 $p_\theta$ 重新生成一个新子目标 $\hat{\mathbf s}_+$，期间低层策略 $\pi_\phi$ 以该子目标为条件连续执行 $k_{\text{test}}$ 步，如此循环直至完成或超时，类似人先做高层规划再交给"肌肉记忆"执行的两阶段模式。实现上，扩散模型和策略都在 $256\times256$ 图像上运作，对语言和图像分别做无分类器引导（沿用 InstructPix2Pix 的做法）；低层策略采用 Diffusion Policy，一次预测 4 步动作 chunk 并做时序平均。真实机器人数据来自 BridgeData V2（6 万条轨迹，其中 4.5 万条带语言标注作为 $\mathcal D_{l,a}$，其余 1.5 万条作为 $\mathcal D_a$），视频协同训练数据来自过滤后的 Something-Something 人类操作视频（约 7.5 万条，作为 $\mathcal D_l$）。

## 三、关键结果

**CALVIN 零样本长时序（train A,B,C → test D，链式指令数 1→5 的成功率）：**

| 方法 | 1 条 | 2 条 | 3 条 | 4 条 | 5 条 |
|---|---|---|---|---|---|
| HULC | 0.43 | 0.14 | 0.04 | 0.01 | 0.00 |
| MCIL | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| AugLC | 0.69 | 0.43 | 0.22 | 0.09 | 0.05 |
| LCBC | 0.67 | 0.31 | 0.17 | 0.10 | 0.06 |
| UniPi（复现） | 0.56 | 0.16 | 0.08 | 0.08 | 0.04 |
| **SuSIE（本文）** | **0.87** | **0.69** | **0.49** | **0.38** | **0.26** |

**真实机器人（WidowX250，3 个场景，平均成功率）：** Scene A（训练分布内）LCBC 0.63 / MOO 0.47 / UniPi 0.00 / RT-2-X 0.43 / **SuSIE 0.87**；Scene B（新背景+干扰物，需抓取又轻又薄的塑料青椒）LCBC 0.20 / MOO 0.05 / RT-2-X 0.00 / **SuSIE 0.50**（唯一能稳定抓起青椒的方法）；Scene C（新桌面纹理+新旧物体混合）LCBC 0.18 / MOO 0.10 / RT-2-X 0.75 / **SuSIE 0.88**。RT-2-X 是 550 亿参数、用超过 110 万条轨迹训练的模型，数据量比 SuSIE 多一个数量级以上，SuSIE 仍全面胜出。

**关键消融：** (1) 与 Oracle GCBC（低层策略结构相同，但测试时直接给真实最终目标图像这一"作弊"信息）对比，SuSIE 在 CALVIN 四个子任务上平均成功率 0.95 远高于 Oracle 的 0.66，在 Scene B 上 0.50 对 0.05——说明生成子目标之所以有效，不只是因为语义正确，更因为它提供了比遥远最终目标更"够近、够可达"的中间引导，直接改善了低层执行精度；(2) 子目标生成模型的视频协同训练消融：仅用 BridgeData 训练 vs. BridgeData + Something-Something 人类视频协同训练，Scene B 平均成功率从 0.30 升到 0.50，Scene C 从 0.80 升到 0.88，验证跨域人类视频数据能提升分布外场景的子目标质量；(3) 定性对比（Figure 4）显示，不用 InstructPix2Pix 预训练权重、从随机初始化训练的子目标模型生成质量明显更差，常出现伪影或误解任务，证明互联网级图像编辑先验是零样本泛化的关键来源。

## 四、评价与展望

SuSIE 的核心贡献是提出了一个简洁的"生成单帧子目标 + 目标条件低层策略"分层框架，只需微调 InstructPix2Pix 即可把互联网图像编辑先验注入机器人控制，相比 UniPi/HiP 等要求生成完整未来视频序列再提取逆动力学动作的路线，规避了开源视频生成模型时序不一致、拖慢推理的问题；相比直接用 VLM 初始化端到端策略（如 RT-2 系）的路线，Oracle GCBC 对比实验提供了一个有意思的证据：语义理解与低层执行精度可能是两个相对独立的能力维度，子目标引导对后者的提升比单纯扩大语言模型更直接有效。这一"高层生成图像目标、低层视觉伺服执行"的思路是后续多篇结合视频/图像生成模型做分层机器人规划工作的重要参照点。

局限性方面，作者在讨论部分明确指出：扩散子目标模型与低层策略是分开训练的，子目标模型并不知道低层策略的实际能力边界，只是假设"数据里出现过的转移都是可达的"，这导致方法的整体表现常常被低层策略的能力瓶颈所限制——论文自身也观察到在 Scene B 抓取轻薄塑料青椒等精细操作上仍会失败。此外，子目标生成本质上仍是 2D 图像编辑，不具备显式的三维空间/几何理解，对于需要精确接触几何推理的任务（如插拔、装配）是否有效未在本文验证；子目标生成与低层执行之间是固定间隔（$k_{\text{test}}$）的开环再规划，论文提到若计算预算允许更频繁地重新生成子目标理论上会更鲁棒，但未做相应实验验证。开放问题包括：如何让子目标生成模型感知并适配低层策略的真实可达能力（联合训练或反馈闭环）；能否将单帧子目标扩展为更丰富的中间表征（如带深度或关键点信息）以支持精细几何操作；以及该范式在更大规模、多机器人本体上的可扩展性。

## 参考

- Brooks et al. *InstructPix2Pix: Learning to Follow Image Editing Instructions*, CVPR 2023.
- Du et al. *Learning Universal Policies via Text-Guided Video Generation (UniPi)*, arXiv:2302.00111.
- Ajay et al. *Compositional Foundation Models for Hierarchical Planning (HiP)*, arXiv:2309.08587.
- Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, 2023.
- Mees et al. *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks*, 2022.
