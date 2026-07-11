# Knowledge Insulation：知识隔离的视觉-语言-动作模型——训练快、推理快、泛化更好

> **论文**：*Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better*
>
> **作者**：Danny Driess、Jost Tobias Springenberg、Brian Ichter、Lili Yu、Adrian Li-Bell、Karl Pertsch、Allen Z. Ren、Homer Walke、Quan Vuong、Lucy Xiaoyang Shi、Sergey Levine
>
> **机构**：Physical Intelligence
>
> **发布时间**：2025 年 05 月（arXiv 2505.23705）
>
> **发表状态**：未录用（预印本，PDF 页脚标注 "Preprint. Under review."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.23705) | [PDF](https://arxiv.org/pdf/2505.23705)
>
> **分类标签**：`VLA` `知识隔离` `梯度停止` `flow matching` `π0` `联合训练` `动作专家`

---

## 一句话总结

针对"给 VLM 骨干加装连续动作专家（diffusion/flow matching）会用其梯度污染、拖慢预训练表示"这一问题，Physical Intelligence 提出**知识隔离（knowledge insulation）**训练配方：用离散化 FAST 动作 token 的下一 token 预测损失训练 VLM 骨干、与此同时用 flow matching 训练一个随机初始化的动作专家产生连续动作，但通过修改注意力算子对动作专家到骨干方向的梯度做**停止梯度**，使骨干只被离散动作损失和 VLM 数据更新、不被动作专家的梯度污染；真实机器人多任务评测中该方法训练收敛速度与纯离散的 π0-FAST 相当（比 π0 快约 7.5 倍训练步数)，推理仍走轻量动作专家保持高频控制，且在语言跟随、任务成功率、LIBERO-90/LIBERO-Spatial 等指标上全面超过 π0、π0-FAST、OpenVLA-OFT、HybridVLA、Transfusion 等基线。

## 一、问题与动机

VLA 模型的核心卖点是把 web 规模 VLM 预训练的语义知识迁移到机器人控制中，但真实控制需要高频、连续、精确的动作输出，这与 VLM 天然的离散 token 自回归解码相悖。为此近期 VLA（如 π0、GR00T、HybridVLA、Transfusion 等）普遍给 VLM 骨干加装专用的连续输出模块——扩散/flow matching "动作专家"——这些模块通常是随机初始化并"嫁接"到预训练骨干上的。论文提出一个此前未被系统研究的问题：**这些连续动作适配器到底在多大程度上真正继承并受益于 web 规模预训练？**

作者在 Sec.4 用实证给出三点诊断：

1. **自回归 VLA 慢**：离散 next-token 解码把连续动作粗粒度化为离散 bin，且大模型自回归解码本身慢——π0-FAST 在 RTX4090 上预测 1 秒动作 chunk 约需 750ms，控制频率仅 1.3Hz，远低于高频控制需求。
2. **机器人专用模块不能充分吃到 VLM 预训练红利**：π0 之类架构的动作专家参数量远小于骨干（可达 10Hz 控制），但由于是随机初始化，naive 联合训练时其梯度回传骨干会明显损害模型的语言跟随能力（推测源于梯度干扰）。
3. **冻结骨干不可行**：直接冻结预训练 VLM、只训练新增的机器人专用模块，看似是保留知识最简单的方案，但现有 VLM 从未见过机器人数据，冻结表示不足以支撑高性能策略（实验 Fig.4a、Fig.8 中 frozen backbone 均是最差组之一，甚至 0% 性能）。

由此引出核心方法论问题：**如何在训练时既让骨干持续吸收动作监督信号以学到适合机器人控制的表示，又不让新初始化的连续动作模块的梯度反过来破坏骨干已有的语言/视觉知识？**

## 二、核心方法

论文在 π0 架构（PaliGemma VLM 骨干 + 独立权重的 flow-matching 动作专家，二者通过自注意力交互）基础上，提出三项彼此配合的改动（Sec.5）：

1. **联合训练（joint-training）**：让模型同时输出离散 FAST 动作 token（自回归，作为骨干的表示学习目标）和连续动作（flow matching，作为动作专家的推理目标）；训练时两个目标同时优化，推理时只用（更小、更快的）动作专家做少步 flow 积分生成连续动作，自回归离散目标仅在训练期充当"表示学习"信号。
2. **VLM 数据协同训练（co-training）**：训练集混入通用视觉-语言数据（图像描述、VQA、目标检测）以及机器人规划数据，减轻模型在只用动作数据微调时对预训练知识的遗忘。
3. **知识隔离（知识隔离的梯度停止）**：修改注意力层，使动作专家到骨干方向的梯度被显式截断，让骨干权重完全由离散动作损失 + VLM 数据损失更新，不受 flow matching 损失干扰。

### 2.1 标准 VLA 训练目标

自回归损失（用于纯离散动作 VLA，如 π0-FAST）：

$$
\mathcal{L}_{\text{AR-VLA}}(\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[-\log p_\theta(y_{1:n}\mid x_{1:n})\Big]=\mathbb{E}_{(x,y)\sim\mathcal{D}}\Big[-\sum_{j=1}^{n-1}M_j\log p_\theta(y_{j+1}\mid x_{1:j})\Big]
$$

flow matching 损失（用于连续动作专家，如 π0）：

$$
\mathcal{L}_{\text{FLOW-VLA}}(\theta)=\mathbb{E}_{\mathcal{D},\tau,\omega}\Big[\big\|\omega-a_{1:H}-f^{a}(a_{1:H}^{\tau,\omega})\big\|^2\Big]
$$

其中 $a_{1:H}^{\tau,\omega}=\tau a_{1:H}+(1-\tau)\omega,\ \omega\sim\mathcal{N}(0,I)$，$\tau\in[0,1]$ 为 flow 时间步，动作专家学习预测把噪声 $\omega$ 拉回真实动作 chunk 的速度场。**用大白话说**：前者是"预测下一个离散动作 token 的分类交叉熵"；后者是"给动作加噪声，让网络学会把噪声一步步'纠正'回真实动作轨迹的速度方向"。

### 2.2 联合训练+协同训练的统一目标

论文把两种目标合并成一个损失，允许语言 token、离散 FAST 动作 token、连续动作 chunk 三种数据混合训练：

$$
\mathcal{L}_{\text{CO-VLA}}(\theta)=\mathbb{E}_{\mathcal{D},\tau,\omega}\Big[-\sum_{j=1}^{n-1}M_j^{\ell}\log p_\theta(\hat y_{j+1}\mid x_{1:j})+\alpha M^{\text{act}}\big\|\omega-a_{1:H}-f_\theta^{a}(a_{1:H}^{\tau,\omega})\big\|^2\Big]
$$

其中 $\alpha$ 是语言/离散动作损失与 flow matching 损失的权衡系数，$M^{\ell}$ 是语言（含离散动作 token）损失掩码，$M^{\text{act}}$ 是标记该样本是否需要预测连续动作的掩码。关键设计是设置注意力 mask，使得**离散 FAST 动作 token 与连续动作 token 互不可见**（互相不能 attend），从而可以自由混搭 VLM-only 数据、纯动作数据、以及"语言描述+动作预测"组合数据（三类模态的自由拼接）。**用大白话说**：一条训练样本里可能同时有"这是一张图，请描述它"的纯语言监督、"预测这段离散动作 token"的表示学习监督、以及"生成这段连续动作"的控制监督，三者共享同一个 transformer 但通过 mask 保证彼此不会读到对方的原始信息，只通过骨干的共享表示间接联系。

### 2.3 知识隔离：注意力层的梯度停止

动作专家的梯度若能回传骨干，会因动作专家权重随机初始化而在训练早期产生不稳定/破坏性的梯度信号。为此论文在**单头注意力**层面显式改写计算：

标准形式为 $P=\text{softmax}\big(Q(X)K(X)^T+A\big)=\begin{pmatrix}P_{bb}&0\\P_{ab}&P_{aa}\end{pmatrix}$，其中 $X$ 是该层输入，$Q,K$ 为 query/key 投影，$A$ 为注意力 mask（下标 $b$ 表骨干 token、$a$ 表动作专家 token）。

插入 stop-gradient 后：

$$
\begin{pmatrix}P_{bb}&0\\P_{ab}&P_{aa}\end{pmatrix}=\text{softmax}\left(\begin{pmatrix}Q_b(X_b)K_b(X_b)^T&0\\Q_a(X_a)\,\text{sg}\big(K_b(X_b)\big)^T&Q_a(X_a)K_a(X_a)^T\end{pmatrix}+A\right)
$$

对应的 value 聚合为：

$$
E=\begin{pmatrix}E_b\\E_a\end{pmatrix}=\begin{pmatrix}P_{bb}V_b(X_b)\\P_{ab}\,\text{sg}\big(V_b(X_b)\big)+P_{aa}V_a(X_a)\end{pmatrix},\qquad \text{attn}(X)=PE
$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient 算子。**用大白话说**：动作专家的 token 在前向传播时仍然可以正常"看到"骨干算出来的 key/value（获得骨干学到的良好表示作为条件），但反向传播时这条链路被剪断——动作专家反传的梯度到骨干的 $K_b,V_b$ 处会被清零，骨干权重只会被自己的自回归离散损失以及骨干内部的注意力路径（$P_{bb}V_b$）更新。这样一来，新初始化的动作专家不会"污染"预训练骨干，同时因为 diffusion/flow loss 只作用在一组独立的动作专家权重上，论文也顺势可以简单地取 $\alpha=1$（不再需要精调两个损失的权衡系数）。

模型细节（Appendix B）：骨干沿用 π0 的 PaliGemma 初始化，骨干宽度 2048、深度 18 层（Fig.1 中标注整体 VLM 骨干约 3B 参数，含视觉编码器），动作专家宽度 1024、约 300M 参数，动作 chunk 长度 $H=50$；骨干与动作专家仅通过自注意力交互，且注意力被设计为单向信息流：骨干 embedding 不 attend 动作专家，动作专家可 attend 骨干（但反向梯度被截断）；flow 时间步 $\tau$ 采用 Beta 分布偏重低时间步采样（沿用 π0 的做法）而非均匀采样。

## 三、实验结果

评测覆盖真实机器人多任务（静态单臂/双臂 + 移动双臂操作机器人）、开源基准 DROID、LIBERO 仿真基准，并同时评测"单一具身专用模型"和"跨 12 种具身配置 + OXE + VLM 数据联合训练的通用模型"两种设定。基线包括 π0、π0-FAST、OpenVLA-OFT、Transfusion、HybridVLA，以及去掉 stop-gradient 的 `joint-training`、去掉协同训练数据的 `joint-training w/o VLM data`、用朴素离散化替代 FAST 的 `naive tokenization` 等消融。

**真实任务 "items in drawer"**（静态单臂，未见环境，满分 5）：本文方法平均任务完成度约 95%，显著优于全部基线（对 joint-training p=0.049，对 π0 p<0.001，对 π0-FAST p=0.030，对 HybridVLA p<0.001，对 Transfusion p=0.013，对 frozen backbone p<0.001）；语言跟随率约 83%，同样是各基线中最高。该任务上冻结骨干或 HybridVLA 均不可行。

**真实任务 "table bussing"**（静态单臂，12 个物体，满分 12）：本文方法性能最高、推理延迟低、语言跟随好；π0-FAST 语言跟随同样不错但完成任务耗时约为本文方法的 2 倍（推理慢）；π0 语言跟随较差；OpenVLA-OFT 语言跟随好、推理快，但整体任务完成度最低。

**真实任务 "shirt folding"**（静态双臂，满分 5）：本文方法显著优于 joint-training（p=0.003）、π0-FAST（p=0.002）、π0 与冻结 OpenVLA-OFT 骨干（均 p<0.001）；对比 naive tokenization 消融差异不显著（p=0.765），说明 FAST 相对朴素离散化的增益主要体现在训练速度和其他任务上而非该任务的最终性能。

**DROID 基准**（真实世界桌面操作，与 π0-FAST 论文同一套任务）：本文方法平均得分 0.55±0.09，π0 为 0.49±0.09，π0-FAST 为 0.45±0.09。

**LIBERO 基准**（成功率 %）：

| 方法 | Spatial | Object | Goal | 10 (Long) | 90 |
|---|---|---|---|---|---|
| Baku | – | – | – | 86.0 | 90.0 |
| MoDE | – | – | – | 94.0 | 95.0 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | **94.5** | – |
| π0 | 96.8 | **98.8** | 95.8 | 85.2 | – |
| π0-FAST | 96.4 | 96.8 | 96.0 | 60.2 | – |
| 本文方法（从零训练） | 96.6 | 97.2 | 94.6 | 84.8 | 92.7 |
| 本文方法（从通用模型微调） | **98.0** | 97.8 | 94.8 | 85.8 | **96.0** |

本文方法在 LIBERO-90 和 LIBERO-Spatial 上取得新 SOTA，但在 LIBERO-10（Long，长程任务）上仍不及 OpenVLA-OFT（85.8 vs 94.5）。

**训练收敛速度**（Fig.6b，通用模型 "table bussing" 任务）：本文方法收敛速度与 π0-FAST 相当，而纯 flow-matching 的 π0 需要约 **7.5 倍**的训练步数才能达到接近的性能。

**通用模型消融**（Fig.6a、Fig.7）：`joint-training`（无 stop-gradient）会明显拉低任务完成度；去掉 VLM 协同训练数据（`ours w/o VLM data`）会小幅降低任务完成率，且对语言跟随率的负面影响在 `joint-training` 设定下尤其显著（作者推测 VLM 数据对避免"灾难性干扰预训练表示"更关键）；在移动机械臂对未见物体的 OOD 泛化实验中，VLM 数据协同训练对 OOD 语言跟随率的提升尤为明显，验证了"VLM 预训练知识确实迁移进了动作策略"这一假设。

**其他消融**：用朴素逐维离散化替代 FAST 作为表示学习目标，性能优于纯连续训练但不及 FAST；稀疏采样（stride 5）优于稠密朴素离散化。状态表示消融（Appendix C，Fig.10）显示本文方法在文本状态和连续状态两种输入形式下都表现良好，而 π0 在两种状态表示下均较差，说明本文方法与 π0 的差距并非源于状态表示的选择。

## 四、局限性

作者在 Sec.7 明确指出：

- 同时训练离散和连续两种输出会使单步训练计算量增加约 **20%**；但因为收敛速度显著更快，综合墙钟时间仍比纯 diffusion 的 π0 之类模型更快。
- 语言跟随虽有改善但仍远非完美，作者认为这很可能源于训练数据中语言指令与视觉/动作之间的虚假相关性，导致模型有时仍会忽略语言指令。
- 论文未讨论该方法在更大规模骨干（超过所用 ~2-3B 参数量级）或更长动作 chunk/更高自由度机器人上的可扩展性；LIBERO-10 长程任务上仍落后 OpenVLA-OFT，说明该配方对长程任务的优势不如短程任务明显。

## 五、评价与展望

**优点**：论文用干净的消融把"联合训练""VLM 数据协同""梯度停止"三个变量拆开单独验证，证据链完整（真实机器人多任务 + DROID + LIBERO 三线印证），并给出了具体可复现的注意力算子改写方案（公式 5-6），工程可操作性强。其核心洞察——"用离散 token 目标做表示学习、用连续输出做推理执行"——把 π0.5（同机构前作，两阶段：先 FAST 离散预训练、再动作专家后训练）中隐含的直觉，系统化为一个**单阶段**且理论上更干净（显式阻断污染梯度而非依赖训练顺序）的训练配方，是该机构 π0 系列在训练方法论上的一次自然延伸。

**与其他公开工作的关系**：与 HybridVLA 相比，本文的关键区别在于用 attention mask 阻止离散/连续两路 token 互相 attend、并额外引入 stop-gradient，消融显示这两点分别都能带来实质提升（HybridVLA 风格的联合训练在 "items in drawer" 任务上仍明显落后）；与 Transfusion 相比，本文动作专家是独立权重而非共享骨干投影层，实验显示 Transfusion 语言跟随优于 π0（因其新增参数仅为投影层）但整体任务表现不及本文方法；与 OpenVLA-OFT（并行解码 + 双向注意力）相比，本文方法在语言跟随和推理速度相近的情况下任务成功率更高，但 OpenVLA-OFT 在 LIBERO-10 长程任务上仍占优，提示并行解码类方法在长时序建模上可能有本文尚未完全吃透的优势。

**开放问题**：(1) stop-gradient 只切断了"动作专家→骨干"方向，骨干→动作专家方向的信息仍然充分传递，这一不对称设计的最优程度（例如是否该允许有限的、经过缩放/裁剪的梯度回传）未被系统扫描，只做了"全阻断 vs 完全不阻断（joint-training）"的二元对比，中间地带（如梯度缩放系数）值得后续研究；(2) 语言跟随不完美的问题作者归因于训练数据的虚假相关，但未给出针对性的数据去偏或对抗性数据增强方案；(3) 论文没有测试知识隔离配方在纯 VLM 能力（如通用图文问答基准）上是否真的做到"零遗忘"，只用了机器人任务的语言跟随率作为代理指标，更严格的灾难性遗忘度量（在标准 VQA/captioning benchmark 上直接对比微调前后的 VLM 能力）会是更有力的验证；(4) 该方法目前仅验证于 π0 系列的 PaliGemma 骨干 + flow-matching 动作专家组合，能否推广到扩散头（如 RDT-1B、Diffusion Policy 类架构）或更大规模 VLM 骨干仍待验证。

## 参考

- Black et al. *π0: A vision-language-action flow model for general robot control.* arXiv:2410.24164, 2024.
- Pertsch et al. *FAST: Efficient action tokenization for vision-language-action models.* RSS 2025.
- Physical Intelligence et al. *π0.5: A vision-language-action model with open-world generalization.* arXiv:2504.16054, 2025.
- Kim, Finn, Liang. *Fine-tuning vision-language-action models: Optimizing speed and success (OpenVLA-OFT).* arXiv:2502.19645, 2025.
- Liu et al. *HybridVLA: Collaborative diffusion and autoregression in a unified vision-language-action model.* arXiv:2503.10631, 2025.
- Zhou et al. *Transfusion: Predict the next token and diffuse images with one multi-modal model.* arXiv:2408.11039, 2024.
