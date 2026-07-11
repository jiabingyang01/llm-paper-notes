# Pelican-Unify：面向理解、推理、想象与动作的统一具身智能模型

> **论文**：*Pelican-Unify 1.0: A Unified Embodied Intelligence Model (UEI) for Understanding, Reasoning, Imagination and Action*
>
> **作者**：Yi Zhang, Yinda Chen, Che Liu, Zeyuan Ding, Jin Xu, Shilong Zou 等（通讯作者 Jian Tang, Xiaozhu Ju；技术负责人 Yong Dai）
>
> **机构**：Beijing Innovation Center of Humanoid Robotics（北京人形机器人创新中心，X-Humanoid）, WFM System Group
>
> **发布时间**：2026 年 05 月（arXiv 2605.15153）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.15153) | [PDF](https://arxiv.org/pdf/2605.15153)
>
> **分类标签**：`统一具身模型` `VLM` `Chain-of-Thought` `世界模型` `Flow Matching` `VLA`

---

## 一句话总结

Pelican-Unify 1.0 把统一理解、语言化链式推理（CoT）与"想象未来视频 + 生成低层动作"三个阶段串成同一条可训练的闭环：VLM 把推理轨迹压缩成一个稠密 loop state $z$，再由一个从 Wan2.2 初始化的共享 DiT 通过条件 flow matching 同时对未来视频和未来动作两路噪声去噪，语言、视频、动作三个 loss 联合反传进同一份共享表示；单一 checkpoint 在 8 个 VLM 基准（均分 64.7）、RoboTwin 50 任务双臂操作（clean/randomized 平均 93.5%）、WorldArena 世界建模（EWM 66.03，第一）三条赛道上同时保持接近或超过各自专用模型的水平。

## 一、问题与动机

具身智能模型的发展长期沿着"碎片化专家"路线推进：视觉-语言模型（如 Gemini Robotics ER、Pelican-VL）具备强语义理解和时空推理，但不能直接执行动作，也无法用物理后果检验自己的推理；视觉-语言-动作模型（RT-2、$\pi_0$、$\pi_{0.5}$、OpenVLA、Helix）把感知和语言接到运动指令上，但动作大多是模仿学习出的映射，缺乏显式的未来想象，导致在未见过的技能组合、长时程任务和接触密集操作上泛化能力有限；世界模型 / 视频生成器（Cosmos-Predict、LeWorldModel 等）能够想象未来视觉状态，但这种想象往往是隐式的像素级预测，难以被任务逻辑或语言化的人类知识引导，也难以解释；World Action Model 一类方法（如 Motus）把想象的未来和动作连接了起来，但由于缺少统一的显式推理，rollout 过程仍难以纠错，容易累积长程误差。

论文的核心论点是：理解、推理、想象、动作不应被当作可以独立训练再拼接输出的模块，而应被视为同一个物理智能闭环里相互制约的维度。这里的"统一"被明确定义为三点，而非简单的模块级拼接：**统一理解**——场景、指令、动作历史、世界状态被嵌入同一个共享语义空间；**统一推理**——推理是语言化、可监督、直接约束想象与执行的过程，而不是与执行脱节的"语言独白"；**统一生成**——未来想象和低层动作由同一个以推理 latent 为条件的扩散解码过程共同产生，使动作具备"后果意识"，想象具备"任务导向性"，推理受"可想象、可执行"的边界约束。

## 二、核心方法

**总体形式化**。Pelican-Unify 是一个复合映射，在每个控制步 $t$ 同时输出推理轨迹、想象视频和可执行动作块：

$$(\tau_t,\ \hat v_{t:t+H},\ \hat a_{t:t+H}) = \mathcal{M}_\Theta(a_{<t},\ l,\ o_{\le t},\ s_{\le t})$$

用大白话说：给定动作历史、语言指令、当前及历史观测帧、机器人本体状态，模型一次前向同时吐出"怎么想""接下来会看到什么""接下来该怎么动"三样东西，而且三者共享同一套内部表示。

**阶段一：统一理解**。多模态上下文 $c_t=(a_{<t},\ l,\ o_{\le t},\ s_{\le t})$ 中，视频帧 $o_{\le t}$ 由 3D video VAE $\mathcal{E}_v$ 编码，动作历史 $a_{<t}$ 由轻量 MLP $\mathcal{E}_a$ 编码，语言指令走文本 tokenizer，机器人状态经小型线性投影，所有 token 拼接后送入以 Qwen3-VL 初始化的 VLM 主干：

$$H_t = \mathrm{VLM}_\phi(c_t)$$

$\mathcal{E}_v,\mathcal{E}_a$ 被显式复用给下游生成模块，使理解与生成共享同一套模态嵌入空间，无需额外的对齐模块。

**阶段二：统一推理**。VLM 以 $H_t$ 为条件自回归地解码链式推理轨迹：

$$p_\phi(\tau_t\mid c_t)=\prod_{i=1}^{|\tau_t|} p_\phi(\tau_{t,i}\mid c_t,\ \tau_{t,<i})$$

轨迹交织两类语言：**Video CoT**（场景将如何演变，例如哪些物体会移动、如何形成接触、工作空间如何重组）与**Action CoT**（要调用哪个子技能、目标末端执行器路点等运动程序分解）。二者被放进同一条因果序列，迫使模型在一次前向里同时想清楚"接下来会发生什么"和"应该怎么做"。轨迹末尾的隐状态经学习投影压缩为稠密 loop state：

$$z = P_\phi(h_{\tau_t})$$

$z$ 是理解/推理通向下游生成的唯一接口，同时被语言建模损失和下游视频/动作生成损失塑造，因此必须同时编码语义、可预测性和可执行性信息。

**阶段三：统一未来生成**。$z$ 被送入一个从 Wan2.2 初始化的共享去噪 Transformer（DiT），通过连续时间 flow matching 同时对未来视频 $v_{t:t+H}$ 和未来动作块 $a_{t:t+H}$ 去噪。两路目标先用与理解阶段相同的嵌入器映射进生成器的 token 空间：$x^v=\mathcal{E}_v(v_{t:t+H})$，$x^a=\mathcal{E}_a(a_{t:t+H})$。共享去噪骨干加两个轻量的模态专属头 $d_v, d_a$ 输出速度预测：

$$(\hat u_s^v,\ \hat u_s^a) = f_\theta(x_s^v,\ x_s^a,\ z,\ s)$$

$z$ 通过 cross-attention 注入，视频和动作 token 在共享 self-attention 中相互交互，模态专属参数只出现在输入输出两端，中间的重计算部分是共享的。

动作流走标准 flow matching：

$$x_s^a=(1-s)x^a+s\,\epsilon^a,\qquad u_s^a=\epsilon^a-x^a$$

视频流额外引入观测帧前缀条件：用二值掩码 $M_{\text{cond}}, M_{\text{fut}}$ 分别选中前缀区域和未来区域，前缀部分保持干净、只对未来区域加噪：

$$x_s^v = M_{\text{cond}}\odot x^v + M_{\text{fut}}\odot\big((1-s)x^v+s\,\epsilon^v\big),\qquad u_s^v = M_{\text{fut}}\odot(\epsilon^v-x^v)$$

用大白话说：视频这一路是"给定已经看到的帧，只去噪还没发生的未来帧"，动作这一路是标准的从噪声到目标动作的流匹配，两路共享同一套 Transformer 计算和同一个推理条件 $z$，因此想象出的未来画面和生成出的动作在结构上被绑定在一起。

三个训练信号联合优化：视频损失只在未来区域计算

$$\mathcal{L}_{\text{video}}=\mathbb{E}_{s,\epsilon^v}\big[\|M_{\text{fut}}\odot(\hat u_s^v-u_s^v)\|_2^2\big]$$

动作损失在有效动作维度 $M_a$ 上做鲁棒回归

$$\mathcal{L}_{\text{action}}=\mathbb{E}_{s,\epsilon^a}\big[M_a\odot\mathrm{SmoothL1}(\hat u_s^a,\ u_s^a)\big]$$

语言损失是标准自回归负对数似然

$$\mathcal{L}_{\text{text}}=-\sum_{i=1}^{|\tau_t|}\log p_\phi(\tau_{t,i}\mid c_t,\ \tau_{t,<i})$$

总目标为加权和 $\mathcal{L}=\lambda_{\text{text}}\mathcal{L}_{\text{text}}+\lambda_{\text{video}}\mathcal{L}_{\text{video}}+\lambda_{\text{action}}\mathcal{L}_{\text{action}}$，三者都经由共享 loop state $z$ 和共享嵌入器 $\mathcal{E}_v,\mathcal{E}_a$ 反传，使理解、推理、想象、动作在训练时形成互相塑造梯度的一个整体，而不是四个先分别训练再拼接的模块。

## 三、实验结果

论文把统一模型拆开放进三条独立评测赛道分别检验，逻辑是"先证明统一不牺牲专精能力"。

**（1）理解能力**（Table 1，8 个基准均分）：Pelican-Unify 1.0 在通用推理基准（MMMU/MMBench/MMStar/InfoVQA/ChartQA）与具身导向基准（Where2Place/PhyX/RefSpatial）上取得最高均分。

| 方法 | MMMU | MMBench | MMStar | InfoVQA | ChartQA | Where2Place | PhyX | RefSpatial | 均分 |
|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | 26.3 | – | – | – | – | – | – | – | 3.3 |
| MolmoAct | 28.4 | 55.1 | 1.2 | 41.9 | 55.9 | 8.2 | 29.7 | – | 27.5 |
| $\pi_{0.5}$ | 24.0 | 6.8 | 21.7 | 7.7 | 5.1 | – | 16.2 | – | 10.2 |
| Gemma3-4B-IT | 39.3 | 68.6 | 37.1 | 40.9 | 50.3 | 7.5 | 17.2 | 2.2 | 32.9 |
| Qwen3-VL-4B-Instruct（基座） | 52.6 | 84.5 | 62.9 | 78.4 | 81.1 | 17.0 | 41.1 | 48.0 | 58.2 |
| **Pelican-Unify 1.0** | **53.0** | **84.9** | **63.3** | 78.4 | **81.5** | **45.2** | **61.7** | **49.3** | **64.7** |

相对基座模型 Qwen3-VL-4B-Instruct，通用推理基准基本持平甚至略优，而具身导向基准提升显著：Where2Place +28.2 分，PhyX +20.6 分，说明统一训练带来的增益主要体现在空间接地与物理理解上，并没有以牺牲通用语义能力为代价。

**（2）动作能力**（Table 2，RoboTwin 50 任务双臂操作，clean/randomized 均值）：

| 类型 | 方法 | Clean | Randomized | 均值 |
|---|---|---|---|---|
| VLA | $\pi_0$ | 65.9 | 58.4 | 62.2 |
| VLA | $\pi_{0.5}$ | 82.7 | 76.8 | 79.8 |
| VLA | starVLA | 88.2 | 88.3 | 88.3 |
| World Model | LingBot-VA | 92.9 | 91.6 | 92.3 |
| World Model | AIM | 94.0 | 92.1 | 93.1 |
| World Model | MotuBrain | 95.8 | 96.1 | **95.9**（第一） |
| Unified | **Pelican-Unify 1.0** | 93.6 | 93.3 | **93.5**（第二） |

Pelican-Unify 1.0 取得对比方法中的次高均值，优于 starVLA、LingBot-VA、AIM 等大多数专用 VLA / World Model 方法，但被专用 world action model MotuBrain（95.9%）反超。50 个任务中 31 个达到 $\ge$95% 成功率，39 个达到 $\ge$90%，15 个 100% 满分；失败集中在需要精细对齐或持续接触的长时程 / 几何敏感任务（如挂马克杯、垃圾桶插入）。

**（3）想象能力**（Table 3，WorldArena，EWM 综合分及子维度，节选，完整 15 个基线见原文）：

| 排名 | 模型 | EWM 均分 | 3D Accuracy | Motion Quality | Visual Quality | Physics Adherence | Controllability |
|---|---|---|---|---|---|---|---|
| 1 | **Pelican-Unify 1.0** | **66.03** | **98.13** | **62.69** | 63.43 | 61.51 | 59.28 |
| 2 | WorldScape v0.2 | 64.24 | 96.28 | 42.34 | 62.65 | 66.92 | 59.38 |
| 4 | MotuBrain | 64.07 | 91.64 | 60.69 | 60.69 | 61.18 | 57.35 |
| 12 | Wan2.6 | 59.80 | 84.68 | 45.92 | 61.44 | 64.00 | 62.66 |
| 15 | Veo3.1 | 57.77 | 86.96 | 30.26 | 57.44 | 68.34 | 46.43 |

Pelican-Unify 1.0 综合排名第一，在 3D Accuracy 和 Motion Quality 两个对空间一致性、物理合理性要求最高的维度上分别排名第一。论文进一步做了盲人评（Table 4，0–2 分×4 维），因为自动指标可能奖励"画面干净但任务无关"的 rollout：

| 模型 | Task Success | Controllability | Temporal Consistency | Physical Plausibility | 均值 |
|---|---|---|---|---|---|
| **Pelican-Unify 1.0** | **1.81** | **2.00** | **2.00** | 1.23 | **1.76** |
| Seedance2.0（API） | 1.21 | 1.87 | 1.98 | 1.15 | 1.55 |
| Happyhorse-1.0 | 1.65 | 1.81 | 2.00 | 0.13 | 1.40 |
| EnerVerse-AC | 0.00 | 1.84 | 2.00 | **1.64** | 1.37 |

Pelican-Unify 1.0 在人评中排名第一，比第二名 Seedance2.0 高 0.21，尤其是 Task Success（1.81）和 Controllability（满分 2.00）领先明显，说明其想象出的 rollout 更"言出必行"（把首帧条件当作要完成的操作目标而非仅需保持画面稳定）。

**（4）真机评测**：在 UR5e 机械臂 + Tienkung 人形机器人平台上做了两类测试。组合泛化实验中，插 RJ45 网线（技能 A）与做防水处理（技能 B）分别单独训练，从未见过链式演示，测试时模型需在一段连续 episode 内完成"A 后接 B"的组合指令；失败主要集中在 A 结束、B 开始的过渡时刻，而基线 VLA 方法在该过渡处失败并非因为无法重新感知环境，而是其动作分布不携带"A 完成后应该发生什么"的表示。零样本泛化实验中，在 Tienkung 平台上对 5 个 seen 任务（约 300 段视频-动作/任务）和 3 个 unseen 任务（每任务仅 50 段视频序列）做联合训练，统一模型在未见任务上表现出良好的域外迁移能力。

## 四、局限性

论文本身没有单列"Limitations"小节，但从正文和实验设计可以归纳出以下几点：

- **动作单项能力非最优**：在 RoboTwin 上 Pelican-Unify 1.0（93.5%）被专用 world action model MotuBrain（95.9%）反超，说明统一训练带来的是"跨三条赛道的均衡竞争力"而非单项最优，论文自己的措辞也是"次优"（second-best）而非第一。
- **失败模式集中于长时程 / 高精度接触任务**：挂马克杯、垃圾桶插入等需要紧密对齐或持续接触力控制的任务成功率明显偏低，说明统一范式尚未解决精细接触操作这一 VLA 领域的共性难题。
- **缺少同架构受控消融**：论文的全部证据都是"与其他论文各自训练的专用模型逐一比较"，没有提供在相同架构 / 相同数据下"联合训练 vs. 三个模块分别训练后拼接"的直接对照实验，因此无法精确量化"联合梯度回传"本身贡献了多少增益，"统一不牺牲专精能力"这一核心论点目前更多是跨论文横向比较得出的存在性证据，而非受控消融结论。
- **规模与工程细节不透明**：论文未披露训练数据规模、算力预算、VLM/DiT 具体参数量（仅可从对比基座 Qwen3-VL-4B-Instruct 推测量级），联合监督所需的"loop-closed data"（视频 + 动作 + CoT 三重标注）的构建成本和规模也未详细说明。
- **未报告推理效率**：扩散式的 Unified Future Generator 在真机闭环控制中的推理延迟 / 控制频率未给出定量数据，而这是决定该范式能否用于高频闭环控制的关键工程指标。
- **作者名单仍是占位状态**：致谢前的贡献者章节注明"最终公开版本将在内部审批后用实名替换群体占位符"，从侧面说明这是一份仍在推进中的技术报告而非经过完整同行评审定稿的论文。
- **自动化世界模型指标的已知缺陷**：论文自己指出 WorldArena 的自动指标可能奖励"画面干净但任务无关"的 rollout（如部分视频扩散基线在 Temporal Consistency 上接近满分但 Task Success 接近 0），因此不得不额外引入人评来验证可用性——这也提示当前世界模型评测体系本身仍不成熟。

## 五、评价与展望

**优点**。论文提出的"统一"定义相对清晰——不是模块拼接而是共享表示 + 联合梯度 + 单一去噪过程——并且用三条相互独立的评测赛道（VLM 基准、RoboTwin 操作、WorldArena 世界建模）分别单独检验，这种"拆开测"的评测设计本身是有说服力的方法论选择,能较好地回应"统一是否以牺牲专精能力为代价"这一核心质疑。视频与动作共享同一去噪骨干、仅在输入输出端分叉的设计（前缀条件 mask 处理观测历史,cross-attention 注入推理 latent $z$）是一个简洁的工程实现,复用 Qwen3-VL 和 Wan2.2 两个现成的强预训练模型作为初始化,降低了从零训练统一大模型的成本。Fig.1 展示的"标准 VLA 微调会削弱基座 VLM 的 grounding / attention 能力,而统一训练保留了这些能力"的现象，与已有文献中 VLA 微调导致视觉能力遗忘的观察相印证，是一个有价值的旁证。

**与其他公开工作的关系**。相较于纯 VLA 路线（$\pi_0$、$\pi_{0.5}$、OpenVLA、RT-2），Pelican-Unify 1.0 显式引入了未来想象和语言化 CoT 作为中间表示，理论上更贴合"预测编码 / 具身认知"这类认知科学动机（论文引用了 Clark 的 predictive brain 理论和 Hesslow 的 simulation-of-behaviour 理论作为哲学基础）；相较于纯世界模型路线（Cosmos-Predict、LeWorldModel），它把想象直接接到了可执行动作上；相较于同属 World Action Model 谱系的 Motus、AIM、LingBot-VA、MotuBrain，论文的核心区分点在于"推理"被显式语言化并通过监督信号约束（Video CoT + Action CoT），而不仅是想象-动作的隐式耦合——但从 RoboTwin 结果看，这个额外的推理层并未在纯操作成功率上体现出对 MotuBrain 的优势，統一推理的价值目前更多体现在跨任务组合泛化（Fig.3/4 的组合插拔实验）和零样本迁移上，而非单任务成功率上限。

**开放问题**。第一,统一范式在更大规模骨干（当前对比基座为 4B 量级 Qwen3-VL）下是否仍然保持"不牺牲专精"的 scaling 行为尚未验证。第二,论文没有给出联合训练相对于分阶段训练的直接增益量化,一个自然的后续实验是在相同数据和架构下做"是否联合反传梯度"的开关消融。第三,想象和动作虽然共享去噪骨干,但两者的时间尺度和不确定性结构本质不同（视频是高维稠密预测,动作是低维精确控制）,当前仅用不同的 loss（MSE vs. SmoothL1）和轻量头区分,这种耦合强度是否是最优设计还有讨论空间。第四,论文的真机评测局限于两个平台上的少量组合技能,更大规模、更多机器人本体的验证会让"统一带来更强组合泛化"这一论断更有说服力。

## 参考

- Black et al. $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164, 2024.
- Zitkovich et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. CoRL 2023.
- Bi et al. Motus: A Unified Latent Action World Model. 2025.
- Bai et al. Qwen3-VL Technical Report. arXiv:2511.21631, 2025.
- Wan Team. Wan: Open and Advanced Large-scale Video Generative Models. arXiv:2503.20314, 2025.
- Shang et al. WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models. arXiv:2602.08971, 2026.
