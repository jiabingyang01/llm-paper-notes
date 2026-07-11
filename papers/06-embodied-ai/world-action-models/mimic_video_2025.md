# mimic-video：面向可泛化机器人控制的视频-动作模型（超越 VLA）

> **论文**：*mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs*
>
> **作者**：Jonas Pai*, Liam Achenbach*, Victoriano Montesinos, Benedek Forrai, Oier Mees†, Elvis Nava† et al.（*为核心贡献者，†为共同指导）
>
> **机构**：mimic robotics、Microsoft Zurich、ETH Zurich、ETH AI Center、UC Berkeley
>
> **发布时间**：2025 年 12 月（arXiv 2512.15692，v2 于 12 月 19 日提交）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.15692) | [PDF](https://arxiv.org/pdf/2512.15692)
>
> **分类标签**：`视频动作模型(VAM)` `Flow Matching` `逆动力学模型` `Cosmos-Predict2` `样本效率` `部分去噪`

---

## 一句话总结

冻结一个预训练视频扩散骨干（Cosmos-Predict2），只在其中间层的"带噪"latent 表征上训练一个轻量 Flow Matching 逆动力学解码器（IDM）来生成动作，把机器人策略的物理动力学建模完全外包给视频预训练；在 SIMPLER-Bridge 上从零训练即以 46.9%（任务级调参后 56.3%）的平均成功率超过同数据规模训练的 π0.5-style VLA 基线（35.4%）及多个已发表 SOTA，同时相对 VLA 架构实现约 10 倍样本效率、2 倍收敛速度提升。

## 一、问题与动机

主流 Vision-Language-Action（VLA）模型建立在图文预训练的 VLM 骨干之上：VLM 从海量但**静态**的网络图文对中学到丰富的语义先验，但图文数据本身不包含时间维度上的物理因果信息（物体如何运动、形变、相互作用）。因此 VLA 在后训练阶段必须**从零学习**动力学与时序依赖，这一负担几乎完全压在稀缺、昂贵的人类遥操作机器人数据上，构成了数据效率的瓶颈——策略的物理常识越少，需要的机器人示范就越多，扩展性因此受限。

已有工作尝试用视频派生的辅助信号（语言子计划、affordance、关键点等）增强 VLA 训练，但把稠密的视频动力学压缩成稀疏表征会造成信息瓶颈，丢失细粒度物理细节。另一条路线是让视频模型联合建模视频与动作的联合分布，但这类方法通常需要在每个控制步完整生成未来视频帧才能提取动作，推理代价过高、且容易被视频生成的伪影带偏（brittle to heuristic tracking）。

论文的核心论点：视频这种模态本身天然编码了动态、程序性的知识（"事情是怎么做成的"），如果能把机器人策略直接扎根于一个预训练视频模型的**latent 表征**，而不是先完整生成视频再解析动作，就能把物理先验的学习完全交给大规模视频预训练，下游只需学习一个简单得多的、单模态、非因果的逆动力学问题。为验证这一假设，论文先做了一个"oracle"案例研究（第三节），再据此提出正式方法 mimic-video。

## 二、核心方法

### 2.1 案例研究：视频生成质量如何影响策略性能（motivating experiment）

作者设计了一个对照实验（Fig. 2）：在预训练视频模型（灰色，未在机器人数据上微调）和微调后视频模型（橙色）两种骨干上，分别训练一个动作解码器，并对比解码器分别接受 (a) 该骨干**预测**得到的视频 latent 和 (b) 直接由**真实（ground-truth）未来视频帧**编码得到的"oracle" latent 作为条件时的成功率。

结果显示：无论骨干是否在机器人数据上微调，用 oracle latent 条件解码器都能达到接近满分的成功率；而用预测 latent 时，微调后骨干明显优于未微调骨干。这说明：（1）控制性能本质上被"预测未来视频"这一步的质量所限制；（2）一旦视频预测足够准确，用一个在极少量低层动作数据上训练的解码器就足以近乎完美地还原动作——即高质量预训练视频骨干本身已经提供了极其丰富的表征，策略学习的负担从"低层动作解码"整体转移到了"视频骨干预训练/微调"。

### 2.2 预备知识：Flow Matching

mimic-video 用 Conditional Flow Matching（CFM）分别训练视频与动作两个生成分量，采用条件最优传输路径：

$$x^\tau = (1-\tau)x^0 + \tau\varepsilon,\qquad \tau\in[0,1] \tag{1}$$

大白话：把干净数据 $x^0$ 和高斯噪声 $\varepsilon$ 沿一条直线插值，$\tau$ 从 0（纯数据）走到 1（纯噪声），模型学习的是这条直线对应的速度场。

训练目标是回归条件生成向量场 $u_\tau(x^\tau\mid x^0) := \varepsilon - x^0$：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\mathcal{T}(\tau),\, p_0(x^0),\, p_\tau(x^\tau\mid x^0)}\left\|v_\theta(x^\tau,\tau) - u_\tau(x^\tau\mid x^0)\right\|^2 \tag{2}$$

大白话：训练一个网络 $v_\theta$，在给定任意噪声程度 $\tau$ 时预测"该往哪个方向走才能回到干净数据"，本质就是去噪速度预测。

推理时从 $\tau=1$（纯噪声）反向积分到 $\tau=0$ 恢复干净样本：

$$\hat{x}^0 = \varepsilon + \int_1^0 v_\theta(\hat{x}^\tau,\tau)\, d\tau \tag{3}$$

关键点：由于 $\tau$ 是连续时间参数，可以在任意中间时刻 $\tau>0$ **停止积分**，得到一个"部分去噪"（partial denoising）的中间态——这是全文方法设计的核心。

### 2.3 架构：两个耦合的 Flow Matching 模型

策略被形式化为 $\pi(A_t\mid o_t, l)$，预测动作序列 $A_t=[a_t,\dots,a_{t+H_a-1}]$，观测 $o_t$ 包含多帧 RGB 图像、语言指令 $l$ 和本体状态 $q_t$。模型由两部分组成：

- **视频模型**：$v_\phi(z_{\text{past}}^0, z_{\text{future}}^{\tau_v}, l, \tau_v)$，诱导 $p_\phi(z_{\text{future}}^0\mid z_{\text{past}}^0, l)$，即给定历史 latent 和语言，预测未来视频 latent 的分布。
- **动作策略**：$\pi_\theta(A_t^{\tau_a}\mid q_t, h_t^{\tau_v}, \tau_a, \tau_v)$，诱导 $p_\theta(A_t^0\mid q_t, h_t^{\tau_v}, \tau_v)$，其中条件表征 $h^{\tau_v} = v_\phi^{(k)}(z_{\text{past}}^0, z_{\text{future}}^{\tau_v}, l, \tau_v)$ 是视频模型在噪声输入 $z_{\text{future}}^{\tau_v}$ 上前向传播时，第 $k$ 层之后抽取的隐藏状态。

视频与动作两条 Flow Matching 支路使用**独立**的流时间 $\tau_v,\tau_a$ 和独立训练调度，使二者可以分别设计学习问题。

**视频骨干**：Cosmos-Predict2（NVIDIA），一个开源的 2B 参数、基于预训练 3D-tokenizer 的 latent Diffusion Transformer（DiT）。输入是干净的上下文前缀 patch embedding（取 5 帧历史）与待生成未来帧的噪声 patch 的拼接；每层交替做全序列自注意力、对 T5 编码语言指令的交叉注意力、两层 MLP。

**动作解码器**：一个较小的 DiT，将本体状态 $q_t$ 与待预测动作序列分别经两条 MLP 编码后拼接为序列维度，用可学习绝对位置编码加时序信息（训练时随机将本体状态 token 替换为掩码 token 以防过拟合低维观测）。每层包含：(1) 对视频模型中间表征 $h^{\tau_v}$ 的交叉注意力，(2) 动作序列自注意力，(3) 两层 MLP；每个子模块带残差连接，并由 AdaLN 调制（AdaLN 的输入是视频/动作两个流时间 $\tau_v,\tau_a$ 的低秩双线性仿射编码）。

### 2.4 部分去噪与动作采样（Algorithm 1）

为了实时控制，推理时**不对视频完整去噪**：从高斯噪声沿视频流场积分到某个中间流时间 $\tau_v$（而非 $\tau_v=0$），得到只保留结构语义、未确定像素细节的"部分去噪"latent $z_{\text{future}}^{\tau_v}$；将其送入视频模型前 $k$ 层得到条件表征 $h^{\tau_v}$；动作解码器再对该条件做完整的动作流积分生成动作块 $A_t^0$。

特别地，实验发现 $\tau_v$ 的最优取值经验上**接近 1**（高噪声端）。当 $\tau_v=1$ 时，视频骨干只需一次前向传播即可提取条件（Algorithm 1 第 3 行的积分退化为恒等），既是性能最优点也是推理速度最快点，二者恰好重合。

### 2.5 两阶段训练（Algorithm 2）

- **第一阶段**（视频骨干对齐）：用 LoRA 在机器人视频数据集上微调预训练视频骨干，使其获得任务域特定语义，同时保留预训练的时序推理能力。
- **第二阶段**（动作解码器训练，骨干冻结）：从零训练动作解码器回归动作流场，条件是从冻结骨干抽取的 $h^{\tau_v}$；训练时对视频流时间 $\tau_v$ 采用 logit-normal 分布采样（与视频预训练一致），动作流时间 $\tau_a$ 采用 $\mathcal{T}_a(\tau_a)\propto\sqrt{\tau_a-0.001}$（沿用 π0.5 的调度），使解码器对各种噪声水平的条件都具备鲁棒性。

## 三、实验结果

评测覆盖三类场景：SIMPLER-Bridge（Widow-X 仿真基准，评测对未见任务的视觉域泛化）、LIBERO（Panda 单臂仿真，Goal/Object/Spatial 三个子集，每任务 50 条示范）、真实世界双臂灵巧手操作（两台 Franka Panda + mimic 16-DoF 灵巧手）。对比基线包括已发表 SOTA（OpenVLA、Octo、ThinkAct、FLOWER、OpenVLA-OFT、Diffusion Policy）以及作者自建的"π0.5-style VLA（Knowledge-Insulating）"——用 PaliGemma-3B 做骨干、动作解码器与 mimic-video 完全相同，专门用来做"视频预训练 vs. 视觉语言预训练"的公平对照。

**SIMPLER-Bridge（平均成功率 %）**：

| 模型 | Carrot on Plate | Spoon on Towel | Stack Blocks | Eggplant | 平均 SR |
|---|---|---|---|---|---|
| OpenVLA（finetuned） | 4.2 | 8.3 | 0.0 | 45.8 | 14.6 |
| Octo（finetuned） | 8.3 | 12.5 | 0.0 | 43.1 | 16.0 |
| ThinkAct（pretrained） | 37.5 | 58.3 | 8.7 | 70.8 | 43.8 |
| FLOWER（finetuned） | 13.0 | 71.0 | 8.0 | 88.0 | 45.0 |
| π0.5-style VLA（scratch） | 25.0 | 29.2 | 20.8 | 66.7 | 35.4 |
| mimic-video（scratch） | 37.5 | 37.5 | 12.5 | 100.0 | 46.9 |
| mimic-video（scratch，任务级 τv 调优） | 54.2 | 41.7 | 29.2 | 100.0 | 56.3 |

mimic-video 从零训练即取得四任务中最强平均成功率，且仅用单一 workspace 相机视角就超过了使用多视角输入的 DiT-Block Policy 变体（见下表），验证生成式视频先验能弥补遮挡带来的视觉不确定性。

**LIBERO（Spatial / Object / Goal / 平均 %）**：

| 模型 | Spatial | Object | Goal | 平均 |
|---|---|---|---|---|
| Diffusion Policy（scratch） | 78.3 | 92.5 | 68.3 | 79.7 |
| Octo（finetuned） | 78.9 | 85.7 | 84.6 | 83.1 |
| DiT Policy（finetuned） | 84.2 | 96.3 | 85.4 | 88.6 |
| OpenVLA（finetuned） | 84.7 | 88.4 | 79.2 | 84.1 |
| OpenVLA-OFT（finetuned） | 96.2 | 98.3 | 96.2 | 96.9 |
| π0.5-style VLA（scratch） | 79.2 | 94.0 | 84.4 | 85.9 |
| mimic-video（scratch） | 94.2 | 96.8 | 90.6 | 93.9 |

尽管 mimic-video 是纯 scratch（仅任务本身的动作数据、无机器人动作预训练），成绩已超过大多数在通用模型上微调而来的方法，且明显高于同架构的 π0.5-style VLA 基线。

**真实世界双臂灵巧操作（成功率 %）**：动作解码器训练数据极稀缺——分拣任务仅 1 小时 33 分（512 条）、收纳任务仅 2 小时 14 分（480 条），视频骨干在更大的 200 小时语料上微调。

| 模型 | Packing | Package handover |
|---|---|---|
| DiT-Block Policy | 11.0 | 30.0 |
| DiT-Block Policy（+ 手腕相机） | 42.6 | 74.1 |
| mimic-video | 72.0 | 93.0 |

**样本效率与收敛速度**：在 LIBERO 上按训练数据量做 sweep，mimic-video 达到 VLM 条件解码器所需最大成功率时，只需其 10% 的数据量；仅用每任务 1 条示范（动作数据减少 98%）仍能达到 77% 的平均成功率，与用 2% 动作数据训练的 Diffusion Policy 基线相当。收敛曲线显示，相同 batch size（128）下 mimic-video 收敛更快、渐近成功率更高，即使 VLA 基线的骨干额外见过 FAST token 化的动作数据预训练。

**τv 的反直觉行为**：SIMPLER-Bridge 上对 $\tau_v$ 做 sweep（Fig. 7）发现最佳自主策略性能出现在 $\tau_v=1$（最高噪声、理论上信息量最少）而非 $\tau_v=0$（完全去噪、信息量理论上最大）。而当用"带噪 ground-truth latent"（而非模型自身预测）条件解码器测量动作重建 MSE 时（Fig. 8），最低误差出现在中间点 $\tau_v\approx0.4$。附录 E 给出两点解释：(1) 分布失配——推理时视频模型自身生成的完全去噪帧会偏离训练时使用的 ground-truth 未来帧分布，留一点噪声相当于一种训练/测试时的数据增强；(2) 表征信息量——去噪越接近 $\tau_v=0$，为最小化训练损失，视频模型各层越倾向学习"近恒等映射"，使得最终层隐藏状态对下游任务反而信息量更少。

**架构消融**（附录 C/D）：视频模型抽取层 $k=19$ 时策略性能最强，向首尾层偏移则显著下降；观测历史用 5 帧优于仅用当前 1 帧；π0.5-style VLA 基线则在 FAST 预训练骨干的第 11 层做交叉注意力效果最好，且解码器训练 2-3 epoch 后在 SIMPLER-Bridge 上收益趋于饱和。

## 四、局限性

论文在第六节明确列出三点不足：

1. **单视角视频骨干**：当前只用固定的单一 workspace 相机视角，限制了策略应对遮挡、多视角空间推理的能力；作者认为探索原生多视角的视频架构会有帮助。
2. **尚未做统一跨本体大规模预训练**：VAM 范式尚未被用来训练一个统一、跨本体的大规模通用模型，作者认为这是释放视频基座模型泛化能力的必要一步，但本文尚未做。
3. **真实世界评测范围有限**：真实机器人实验仅覆盖两个双臂灵巧手长时程任务（分拣、收纳），把该方法扩展到更广泛的操作行为种类留作未来工作。

此外，方法论层面还有几点隐含局限（笔记补充，非原文列举）：视频骨干（Cosmos-Predict2）与动作解码器分两阶段训练、非端到端联合优化；case study 的"oracle"结论建立在视频重建误差与策略成功率高度相关这一前提上，尚未在更大规模跨任务/跨具身场景下验证是否稳健；τv 的最优值被描述为"任务相关"（task-dependent），目前仍需人工 sweep 而非自动学习。

## 五、评价与展望

**优点**：mimic-video 的贡献不仅是一个新架构，更重要的是第三节那个干净的"oracle"消融实验——它把"视频预测质量"与"动作解码质量"解耦，直接证明了控制性能几乎完全由视频预测的好坏决定，这为"用视频基座模型替代 VLM 骨干"这一日益流行的方向提供了一个相对严谨的因果性证据，而不只是端到端指标提升的堆砌。方法本身也很简洁：不需要生成完整视频帧（区别于 Dreamitate、Video Language Planning 等像素空间路线），只取中间层 noisy latent，兼顾了效率与信息完整度；part-denoising 在 $\tau_v=1$ 处的性能与速度双重最优这一发现具有很好的工程可用性。与 Video Policy（同样在视频 latent 上条件化动作解码器但不能像 Flow Matching 一样高效采样边际动作分布）、FLARE（对齐 VLA 中间表征到未来视觉 embedding，仍以 VLM 为骨干）等同期工作相比，mimic-video 更彻底地把控制的"多模态长时程规划"部分甩给冻结的视频骨干，解码器只需处理简单的单模态非因果逆动力学问题，这一分工在论文的多个基准上都体现出样本效率优势。

**局限与开放问题**：τv 最优值高度依赖任务、需要手工 sweep，是否存在自适应或可学习的机制值得后续探索；视频骨干与动作解码器的两阶段训练范式虽然稳定，但也放弃了端到端联合优化可能带来的额外收益；论文未讨论当视频骨干本身对某类物理交互（如可形变物体、精细力控）预测失准时策略会如何失败，这类"视频先验偏差传导到动作"的鲁棒性问题在真实部署中值得关注；比较基线中的"π0.5-style VLA"是作者自建的、参数规模和训练数据都严格对齐的对照组，这种控制变量的比较方式提高了结论可信度，但与业界实际部署的 π0/π0.5、OpenVLA-OFT 等全量预训练模型相比仍有差距（如 LIBERO 上 OpenVLA-OFT 的 96.9% 仍明显高于 mimic-video 的 93.9%），说明在数据充分的场景下，成熟 VLA 路线依然具竞争力，VAM 的比较优势更多体现在数据稀缺、样本效率优先的场景。整体而言，这篇论文为"视频基座模型可否替代/补充 VLM 作为机器人策略骨干"这一问题提供了有说服力的实证支持，是 video-as-world-model 与 action-decoding 解耦范式的一个扎实的工程实现。

## 参考

- Black K., Brown N., et al. *pi0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.
- Kim M. J., Finn C., Liang P. *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024.
- Liang J., Tokmakov P., et al. *Video Generators are Robot Policies*, arXiv:2508.00795, 2025.
- Zheng R., Wang J., et al. *FLARE: Robot Learning with Implicit World Modeling*, arXiv:2505.15659, 2025.
- Driess D., Springenberg J. T., et al. *Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better*, arXiv:2505.23705, 2025.
