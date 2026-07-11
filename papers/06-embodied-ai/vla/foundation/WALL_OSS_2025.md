# WALL-OSS：点燃视觉-语言模型迈向具身空间

> **论文**：*Igniting VLMs toward the Embodied Space*
>
> **作者**：Andy Zhai, Hao Wang（通讯作者）, Lucy Liang（项目负责人）et al.
>
> **机构**：X Square Robot
>
> **发布时间**：2025 年 09 月（arXiv 2509.11766）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.11766) | [PDF](https://arxiv.org/pdf/2509.11766)
>
> **分类标签**：`VLA` `MoE架构` `Chain-of-Thought` `Flow Matching` `具身基础模型`

---

## 一句话总结

WALL-OSS 用一个紧耦合的 Mixture-of-Experts 架构（共享自注意力、按模态拆分 Vision-Language FFN 与 Action FFN、用静态路由而非可学习路由分流特征）搭配两阶段训练课程（Inspiration 阶段用离散 FAST token 注入粗粒度动作先验并强化具身 VQA，Integration 阶段切换为 flow matching 做连续动作精修），把"指令→推理（CoT）→子任务→连续动作"统一进单一可微模型（Uni-CoT），在自建 Embodied VQA 基准上将 Object Grounding 从基座 Qwen2.5-VL-3B 的 46.1% 提升到 91.6%，在六个真实机械臂长时程/推理任务上全面超过 π0 与 Diffusion Policy 两个基线，尤其在需要文本推理的 Block-Spell 任务上指令跟随准确率（字母类）达到 87%，远超 π0 的 9%。

## 一、问题与动机

论文指出，把强 VLM 骨干迁移到具身动作空间存在三个根本性错配（gap）：

1. **模态与数据规模的 gap**：文本/2D 图像有海量互联网数据支撑跨模态对齐，而机器人动作是连续、跨越 3D 时空的信号，配对的 VLA 数据稀缺且异构，语义对齐难度远高于文本-图像。
2. **预训练分布的 gap**：VLM 在通用网络语料上训练充分，但具身场景（第一人称视角、鱼眼成像、自遮挡等）与互联网图像分布差异很大，现有 VLM 在空间推理、具身场景理解、进度追踪等能力上普遍薄弱。
3. **训练目标的 gap**：LLM/VLM 用离散序列上的 next-token 似然做训练目标，而动作轨迹是连续、高频信号，更适合用 diffusion/flow matching 这类条件生成目标建模；直接把生成式目标嫁接到 VLM 上会放大 tokenization 鸿沟，破坏原有的语言-动作对齐。

现有两类迁移范式各有缺陷：（a）**统一设计**（如 RT-2、OpenVLA），直接在原 VLM 上叠加离散或连续动作头做 next-token 训练，动作监督会显著扰动 VLM 原有权重分布，导致对动作过拟合、损害指令跟随和泛化；（b）**解耦设计**（如 π0），用独立分支通过交叉注意力从 VLM 提取信息生成动作，语言/视觉只是辅助信号，绑定较松，指令跟随能力弱。WALL-OSS 试图用 Mixture-of-Experts 架构在二者之间找到"紧耦合但不互相破坏"的折中点。

## 二、核心方法

**骨干与整体架构**：以 QwenVL2.5-3B 为主干，输入为第一人称/腕部相机视觉与文本指令，共享一套 Self-Attention，但 FFN 层拆分为 Vision-Language FFN 和 Action FFN 两个专家，二者通过**静态路由**（而非可学习的 softmax/top-k 路由）分流：VQA、CoT、子任务描述以及（Inspiration 阶段的）离散动作走 VL FFN + LM Head；连续动作走 Action FFN + Flow Head。这一设计对应论文 Figure 2 中与"统一设计"（共享 FFN，易过拟合动作、破坏 VL 先验）和"解耦设计"（独立分支，语言只是辅助信号）相区分的第三条路径。

**两阶段训练课程**：

*Inspiration 阶段*：复用预训练 VLM 原有 FFN，用具身 VQA、图文/视频对比学习、指令跟随、时序因果建模等任务增强空间推理能力；同时引入离散动作目标，把连续轨迹 $\mathbf{a}$ 通过 FAST（DCT → 量化 → BPE，借鉴 π0-FAST）离散化为 token 序列 $z_{1:K}=\mathrm{FAST}(\mathbf{a})$，与文本 token 对齐训练：

$$\mathcal{L}_{\text{Inspiration}} = \lambda_{\text{VQA}} \sum_t -\log p_\theta(\tau_t \mid \tau_{<t}, \mathbf{c}) + \lambda_{\text{D}} \sum_k -\log p_\theta(z_k \mid z_{<k}, \mathbf{c})$$

用大白话说：这一阶段先不追求动作精度，而是让模型"看得懂"具身场景（VQA）并对动作有个粗粒度的、离散化的文本级认知（就像先学会说"往左移一格"而不是精确输出连续坐标），为后续连续动作打好语义地基。

*Integration 阶段*：把离散动作预测换成 flow matching 连续建模，分两个 phase 递进。Phase 1 冻结 VLM，只训练 Action FFN 之下的 Flow Head，用带噪声插值样本回归速度场：

$$x_t = (1-\rho(t))\,x_0 + \rho(t)\,\epsilon,\qquad \mathcal{L}_{\text{Integration}} = \lambda_{\text{C}}\,\mathbb{E}\Big[w(t)\,\big\|v_\phi(x_t,\mathbf{h},t) - (\epsilon - x_0)\big\|_2^2\Big]$$

其中 $x_0$ 是干净动作样本，$\epsilon$ 是高斯噪声，$\rho(t)$ 是噪声调度函数，$v_\phi$ 是速度场网络。大白话说：这就是标准的 flow matching / rectified flow 训练——把"干净动作"和"纯噪声"之间连一条直线路径，训练网络学会在路径上任意时刻预测应该往哪个方向走，从而在推理时通过若干步积分从噪声生成动作；Inspiration 阶段学到的跨模态注意力为这一步提供了稳定、语义对齐好的初始化，避免从零训练 flow head 时的语言先验丢失。Phase 2 解冻 VLM，联合优化 VL 与 Action 两个分支（梯度对 $\theta,\phi$ 均非零），让模型能整合多模态信息完成精细动作输出，进一步收紧跨模态对齐。

**Unified Cross-Level CoT（Uni-CoT）**：论文把传统"文本内 step-by-step 推理"意义上的 CoT 推广为跨越"指令→推理→子任务规划→连续动作"整个语义-感知运动谱系的广义 CoT，用单一可微模型学习跨抽象层级的前向任意映射，而不是像 Hi Robot、GR00T N1 那样用独立规划器 + 独立控制器的 pipeline（非可微接口、误差逐级累积）。训练目标为一个允许"路径丢弃"（path-drop）的联合目标：

$$\min_\theta\;\mathbb{E}_{(v,x,c,a)}\Big[\ell_{\text{act}}\big(F_\theta(v,x,c),\,a_{1:T}\big) + \lambda\,\ell_{\text{VQA}}\big(H_\theta(v,x),\,y\big)\Big]$$

其中 $v$ 为视觉输入，$x$ 为语言指令，$c$ 为可选的中间 CoT，$a_{1:T}$ 为目标动作轨迹，$H_\theta$ 是具身 VQA 头。大白话说：模型既可以走"指令→CoT→子任务→动作"的完整链条，也可以在任务简单时直接跳过中间推理、一步到位从指令映射到动作——训练时随机丢弃中间监督环节，让模型学会按任务难度自适应选择推理深度；推理时甚至可以异步交织推理与执行（边完成子任务边继续想下一步），支持更灵活的人机实时交互。

**数据构成**：语料总时长超过 1 万小时（4.4 节又称"超过数万小时"，原文两处表述略有出入），由三部分组成：自采机器人动作数据占 57.5%（覆盖桌面臂、移动底盘、轮式双臂、轮式人形等平台，居家清洁/整理、移动抓取、装配等场景）、开源动作数据占 33.1%（汇总 DROID、BC-Z、RH20T、BridgeData V2、Furniture-Bench、Fractal、UMI-biarm、AgiBotWorld 等约 20 余个公开数据集）、多模态 VQA 数据占 9.4%（含 CapsFusion、Cambrian、PixMo、RoboPoint、Robo2VLM、COCO、VQAv2 等通用 VQA，以及 VQASynth-SpaceLLaVA、SpaceThinker、OpenSpaces 系列等空间推理数据）。开源动作数据统一了坐标系/单位、跨形态 DoF 模板、多视角内外参与时间戳对齐、控制频率重采样等规范以降低跨源摩擦。

## 三、实验结果

**Embodied VQA 基准（Table 2，人工评测）**：基于自采轨迹随机采样帧构建，覆盖 Object Grounding（2D 坐标定位）、Scene Captioning（场景描述）、Action Planning（给定高层指令规划下一步动作）三项。

| 模型 | Object Grounding | Scene Captioning | Action Planning |
|---|---|---|---|
| Qwen2.5-VL-3B（基座） | 46.1% | 57.7% | 59.8% |
| WALL-OSS | 91.6% | 87.6% | 69.0% |

**零样本指令跟随**：pick-and-place 任务下，ID（预训练中见过的物体/容器）平均任务进度 85%，OOD（全新物体）平均任务进度 61%，失败案例多为抓取/摆放位姿的小幅偏差而非语义误解。

**动作精度与泛化（Collect-Waste / Pick-Place-Cup）**：数据充足（Collect-Waste，1000 条演示）时，WALL-OSS 与 π0 在 ID 场景均达 100% 成功率，未经预训练的 Diffusion Policy 仅 80%；数据稀缺、任务更复杂时（Pick-Place-Cup，500 条演示），WALL-OSS/π0 仍保持 90% 以上成功率，DP 跌破 20%；OOD 场景（Collect-Waste 换新环境）下 DP 成功率从 80% 跌至 0%，WALL-OSS 与 π0 均维持 80% 以上。

**长时程任务（Set-Table / Tidy-Bedroom）与推理任务（Block-Spell）**：这两类任务不在预训练数据中出现，专门用来测泛化。WALL-OSS 在 Figure 7 的 ID/OOD 柱状对比中于 place-by-color、block-spell、set-table、tidy-bedroom、collect-waste、pick-place-cup 六项及平均值上全面超过 π0 与 Diffusion-Policy 两个基线（各配 Flat 与 GPT4-Subtask 两种指令范式）；长时程任务中基线常见"原地重复放置""目标漏检"等因缺乏任务进度感知导致的失败模式，而 WALL-OSS 凭子任务生成保持连贯的阶段推进。

**指令跟随细粒度消融（Table 3，Block-Spell 任务，选中正确字母/数字积木的准确率）**：

| 积木类型 | WALL-OSS（多模态协同训练） | WALL-OSS（仅动作训练） | π0（仅动作训练） |
|---|---|---|---|
| Letter | 87% | 26% | 9% |
| Number | 95% | 80% | 35% |

多模态协同训练（联合训练动作生成、CoT+子任务生成、2D referring grounding）相对于仅用动作监督微调带来巨大提升，证明 VQA/grounding 协同训练是细粒度指令跟随的关键来源，而不仅是动作头本身的能力。

## 四、局限性

- 论文在 Discussion 中坦承：WALL-OSS 相对 Diffusion Policy 有更强的 OOD 泛化，但在纯动作精度上，"**π0 在精细操作上仍然更优**"（"Pi-0 remains superior for precise manipulation"），即紧耦合 MoE 换来的语言-动作绑定优势并未完全弥补动作头本身在精度上的差距。
- 训练语料中自采数据占比最高（57.5%），这部分数据不公开（仅开源代码与模型权重），第三方难以完全复现预训练语料；而唯一能提供语言/空间语义强监督的多模态 VQA 数据占比仅 9.4%，论文自己也承认这部分信号"较弱地关联到操作本身，主要是正则化 VL 骨干而非动作头"。
- 主实验的评测任务（六个机械臂任务）与 Embodied VQA 基准均为作者自建，未使用 LIBERO、CALVIN、SimplerEnv 等社区公开基准，虽然采用了盲测第三方评测协议以保证客观性，但跨论文横向可比性有限。
- 主要对比基线只有 π0 与 Diffusion Policy 两个方法，未与 OpenVLA、RT-2、GR00T N1 等同期公开 VLA/具身基础模型做直接数值对比。
- 静态路由（而非可学习路由）依赖训练时预先标定每条数据的模态类型（VL 或 Action）来分流 token，这一设计虽稳定，但相比可学习 MoE 路由损失了根据输入自适应分配专家的灵活性，其在更复杂多模态混合场景下的可扩展性未被验证。
- 论文在 Future Directions 中开放式讨论了"是否需要引入中间表征（未来帧预测或 3D 感知）来降低 VL→动作映射难度"，这本身说明当前端到端方案尚未彻底解决语言/视觉推理与低层空间感知-控制之间的对齐问题，仍是未闭合的研究问题。

## 五、评价与展望

**优点**：WALL-OSS 最主要的方法论贡献是用共享自注意力 + 分专家 FFN 的 MoE 架构，同时兼顾"统一设计"（RT-2/OpenVLA 式，紧耦合但易破坏 VL 先验）和"解耦设计"（π0 式，保留 VL 能力但语言-动作绑定弱）两条路线各自的短板，用较小的额外参数量（仅新增 Action FFN 与 Flow Head）实现了"同时提升 VL 理解与动作生成、而非二者零和博弈"这一具体、可复现的架构论证，Table 2/3 的对比数字为这一论点提供了直接证据。Uni-CoT 的路径丢弃训练目标把"要不要显式推理"变成模型可学习的行为而非人工规则，这一点比 SayCan/Code-as-Policies 式显式规划器-控制器 pipeline 更灵活，也不同于 Hi Robot/GR00T N1 的两阶段级联结构。

**与其他公开工作的关系**：动作生成路线上，WALL-OSS 延续"离散先验→连续控制"的两段式（先用 π0-FAST 式离散 token 注入粗粒度动作先验，再用 flow matching 精修），与 π0/π0.5 的持续动作头思路一脉相承，但通过静态路由的 MoE 结构比 π0 的交叉注意力解耦设计绑定更紧；相比 RT-2/OpenVLA 直接在共享 FFN 上做离散 token next-token 训练，WALL-OSS 用分专家 FFN 避免了动作监督直接污染语言权重的问题；相比 Hi Robot、GR00T N1 的规划器-控制器分离式分层架构，WALL-OSS 坚持单模型端到端可微，规避了非可微接口带来的误差累积，但也因此更依赖精心设计的训练课程（两阶段、path-drop）来保证不同抽象层级之间学到一致的语义。

**开放问题**：其一，论文自陈 π0 在精细操作精度上仍占优，说明"紧耦合架构提升语言-动作绑定"与"动作头本身的控制精度"是两个相对独立的能力维度，如何在同一框架内同时逼近两者的上限仍待探索；其二，评测体系高度依赖自建基准与不公开的自采数据，后续工作若想验证方法的架构学论证是否具有普适性，需要在 LIBERO/CALVIN/SimplerEnv 等公开基准上补充结果；其三，论文明确把"是否需要引入未来帧预测、3D 感知等中间表征"作为悬而未决的问题抛给社区，这与近期一些通过世界模型或视频预测提供额外监督信号的工作（如生成式数据增强、3D-VLA 等）形成呼应，但 WALL-OSS 本身尚未给出结论性答案；其四，静态路由的模态分流规则更像是工程先验而非学习得到的结构，其在训练分布之外（例如动作与语言高度纠缠、难以预先归类的输入）的鲁棒性值得进一步压力测试。总体而言，这是一篇偏工程体系化、以架构和训练课程设计为核心贡献的技术报告，其"紧耦合 MoE 缓解 VL-动作零和博弈"的论证思路对具身基础模型的架构设计具有参考价值，但受限于闭源数据和自建评测，其结论的外部可比性有待社区在公开基准上进一步检验。

## 参考

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Policies*, arXiv:2410.24164 — flow matching 动作头的代表工作，也是本文核心对比基线。
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, arXiv:2501.09747 — 本文 Inspiration 阶段离散动作 token 化方法的直接来源。
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246 — "统一设计"范式（共享 FFN 直接叠加动作头）的代表，本文架构对比对象之一。
- Zitkovich et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, ICML/CoRL 系列 — 离散动作 next-token 预测范式的奠基工作。
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv:2303.04137 — 无 VLM 初始化的扩散策略基线，本文实验中的非预训练对照组。
