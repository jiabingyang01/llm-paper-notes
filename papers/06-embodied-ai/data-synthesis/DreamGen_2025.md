# DreamGen：通过视频世界模型解锁机器人学习的泛化能力

> **论文**：*DreamGen: Unlocking Generalization in Robot Learning through Video World Models*
>
> **作者**：Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang（共同一作）… Ming-Yu Liu, Luke Zettlemoyer, Jan Kautz, Dieter Fox, Yuke Zhu, Scott Reed, Linxi Fan（共同指导）et al.
>
> **机构**：NVIDIA、University of Washington、KAIST、UCLA、UCSD、CalTech、NTU、University of Maryland、UT Austin
>
> **发布时间**：2025 年 05 月（arXiv 2505.12705，v2 于 2025-06-17）
>
> **发表状态**：未录用（预印本）；项目页 research.nvidia.com/labs/gear/dreamgen
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.12705) | [PDF](https://arxiv.org/pdf/2505.12705)
>
> **分类标签**：`视频世界模型` `合成数据` `VLA 泛化` `伪动作标注` `neural trajectory`

---

## 一句话总结

DreamGen 提出一条极简的 4 阶段流水线：把 SOTA image-to-video 生成模型（默认 WAN 2.1）在目标机器人本体的少量遥操作数据上做 LoRA 微调，再用「初始帧 + 语言指令」滚动生成大量逼真的机器人视频，用 IDM 或 LAPA 逆推伪动作标签，得到所谓 neural trajectory（合成视频-动作对）去训练视觉运动策略——仅靠单一环境下的 pick-and-place 遥操作数据，就让 GR1 人形机器人在 22 个全新行为上实现从 0% 到 43.2%（已见环境）/ 28.5%（未见环境）的零到一突破。

## 一、问题与动机

机器人基础模型（RT-2、π0、GR00T N1、Gemini Robotics 等）严重依赖为每个新任务、新环境**人工采集遥操作数据**，成本高、劳动密集。两条常见替代路线各有硬伤:

- **仿真造数据**:自动化程度高，但受 sim2real gap、难以仿真液体/铰接物体、以及被 TAMP 或人类演示插值所限。
- **把视频世界模型当实时规划器**（RoboDreamer、UniPi、Video Language Planning 等）:让世界模型在 test-time 参与闭环，但受限于生成速度与规划精度，难以充分释放大模型的物理先验。

DreamGen 的核心洞见是:**不要把视频世界模型当实时规划器,而是当离线的合成数据生成器**。这样可以放心地使用最强、最慢的视频生成模型（无需实时性),充分利用其物理推理、自然运动与语言接地的先验,把「视频世界模型 → 合成机器人数据」变成一条可扩展的新数据轴,绕开人工采集的瓶颈。

## 二、核心方法

DreamGen 是一条与底层策略架构解耦的 4 步流水线（论文用文字描述，下列公式为对该流程的形式化重述）。

**Step 1 — 视频世界模型微调。** 在目标机器人本体的人类遥操作视频上微调视频生成模型 $G_\theta$（默认 WAN 2.1，LoRA rank 4 / alpha 4，lr 1e-4），使其学到该本体的物理约束与运动学，同时用 LoRA 尽量不遗忘互联网视频先验。对多视角数据（RoboCasa、DROID），把多个视角拼成 $2\times2$ 网格（缺的一格填黑）再微调。选型时看两个指标:instruction following（指令遵循）与 physics following（物理遵循）。

**Step 2 — 世界模型滚动生成。** 给定初始帧 $s_0$ 与语言指令 $\ell$，生成合成视频:

$$
\hat{V} = \{\hat{s}_1, \hat{s}_2, \ldots, \hat{s}_T\} = G_\theta(s_0, \ell)
$$

用大白话说:喂给微调后的世界模型一张「起始画面」和一句话（比如"给花浇水"），它就脑补出一整段机器人完成该任务的视频，且能覆盖训练里从没出现过的新动词、新环境。

**Step 3 — 伪动作标注。** 生成的视频没有动作标签，用两种方式逆推:

- **IDM（逆动力学模型）**:diffusion transformer + SigLIP-2 视觉编码器，flow-matching 目标，仅以相邻两帧为条件（不用语言/本体感知），预测长度为 $H$ 的动作块，再用滑窗逐段标注:

$$
\hat{a}_{t:t+H} = \mathrm{IDM}(s_t,\, s_{t+H})
$$

- **LAPA（潜动作）**:transformer encoder-decoder，用 VQ-VAE 目标捕捉当前帧与 1 秒后未来帧之间的**视觉 delta**，取量化前的连续 embedding 作潜动作（沿用 GR00T N1）。好处:无需目标本体的真值动作即可训练。

用大白话说:IDM 像一个"看两帧图就猜中间机器人做了什么动作"的读心器,需要真值动作训练；LAPA 则不需要真值动作,只把"画面怎么变了"编码成一个抽象动作码。视频 + 伪动作 = neural trajectory:

$$
\tau_{\text{neural}} = \langle o_t,\, i_t,\, \hat{a}_{t:t+H} \rangle
$$

**Step 4 — 在 neural trajectory 上训练策略。** 以图像观测 $o_t$ 和任务指令 $i_t$ 为条件（state 信息全置零，因为合成轨迹没有本体感知），训练策略 $\pi_\phi$ 回归伪动作:

$$
\min_\phi\; \mathbb{E}\big[\mathcal{L}\big(\pi_\phi(o_t, i_t),\, \hat{a}_{t:t+H}\big)\big]
$$

两种训练范式:①**co-training**(合成轨迹 + 真实轨迹按 1:1 采样);②**纯合成**(只用 IDM 标注的 neural trajectory)。因流水线与策略架构无关,论文在 Diffusion Policy、π0、GR00T N1 三种策略上都验证。对 GR00T N1，两类轨迹被当作**独立本体**处理（各用一套 action encoder/decoder），以适配 IDM 动作 state 全零的特性。

## 三、实验结果

三大应用场景:数据增广、行为泛化、环境泛化;并附带一个视频生成基准 DreamGen Bench。

**（1）仿真数据增广（RoboCasa，24 任务，GR00T N1）。** neural trajectory 数量与下游成功率呈稳定的 log-linear 正相关，最多可把合成数据扩到相对原始人类演示的 **333×**。

| GT 数据档位 | 伪动作 | 0 条(基线) | 240K neural traj. |
|---|---|---|---|
| High GT | IDM | 49.60 | **57.60** |
| Medium GT | IDM | 32.10 | 39.94 |
| Low GT | IDM | 17.40 | 23.32 |
| High GT | LAPA | 49.60 | 58.21 |

Table 4 显示 GR00T N1 在 300 真值轨迹 + 240K neural traj. 下平均成功率 **57.61%** vs 仅 300 真值轨迹的 49.59%。尤为关键:**纯用 neural trajectory（IDM 标注）** 训练即可达 24 任务平均 **20.55%**，说明合成轨迹质量已相当接近真值。

**（2）真机数据增广（9 任务，3 本体，每任务仅 10–13 条真实轨迹）。** co-training 后各本体 GR00T N1 平均成功率:

| 本体 | 任务数 | Low Data 基线 | + Neural Traj. |
|---|---|---|---|
| GR1 人形 | 4 | 37% | **46.4%** |
| Franka | 3 | 23% | **37%** |
| SO-100 | 2 | 21% | **45.5%** |

这些任务（锤钉、擦液体、叠毛巾、舀 M&M 豆）涉及工具操作与可形变物体，用现有仿真方法几乎无法造数据。

**（3）行为与环境泛化（GR1，仅在单一环境的 2,884 条 pick-and-place 上训练视频模型）。** 这是真正的**零到一**突破——GR00T N1 只训过 pick-and-place，对新动词/新环境几乎为 0:

| 场景 | 任务数 | GR00T N1 基线 | w/ DreamGen |
|---|---|---|---|
| 已见环境 + 新行为 | 14 | 11.2%（含部分得分） | **43.2%** |
| 新环境 + 新行为 | 13 | 0.0% | **28.5%** |

22 个新行为包括浇花、开关铰接物、用锤子/吸尘器/熨斗等。环境泛化中只需人工拍新环境的**初始帧**，不采任何该环境的物理数据，是一种 zero-shot 迁移。

**（4）DreamGen Bench（视频生成的机器人基准）。** 两指标:Instruction Following（IF，用 GPT-4o / Qwen2.5-VL 判分，与人评 Pearson 相关 >0.9）与 Physics Alignment（PA，用 VideoCon-Physics）。对比 Hunyuan、CogVideoX、WAN 2.1、Cosmos 的 zero-shot 与 fine-tuned 版本:微调后 **Cosmos-sft 最强**（RoboCasa IF-GPT 79.2 / PA 61.5），WAN2.1-sft 次之。最重要结论:**DreamGen Bench 分数与下游 RoboCasa 策略成功率正相关**（Figure 6），说明造一个更强的世界模型（更会遵循指令、更符合物理）就能带来更大的下游收益——为视频模型研究者提供了一条无需真机在环、低成本贡献机器人学习的路径。

## 四、局限性

- **算力昂贵**:生成 24 万样本的 RoboCasa 数据集需在 **1500 张 NVIDIA L40 上跑 54 小时**;如何在不牺牲视频先验的前提下降本仍是难题。
- **依赖人工提供初始帧**:环境/行为泛化都要人手拍或选初始帧，带来操作开销;论文把「用 image-to-image / inpainting 自动生成初始帧」留作未来工作。
- **任务偏简单**:目前任务只覆盖机器人全部运动学能力的一小部分，缺乏需要精细力控的复杂灵巧操作。
- **纯合成、零真值下的泛化仍开放**:文中明言「用零真值数据实现对新行为/新环境的 zero-shot 泛化仍是未解问题」;IDM 质量是主要瓶颈。
- **未与人类视频方法直接对比**:承认与「从人类演示视频学习」的工作互补，但没做正面 benchmark。
- **自动评测器会幻觉**:DreamGen Bench 用轻量开源模型判物理真实性，偶尔误判。

## 五、评价与展望

**优点。** ①范式清晰且工程简单——把"视频世界模型当离线合成数据生成器"而非实时规划器，一举回避实时性约束，充分吃满 SOTA 视频模型的物理与语言先验，这一定位比 UniPi/RoboDreamer/Video Language Planning 等 test-time 规划路线更务实。②与策略架构解耦，DP/π0/GR00T N1 三种策略上都涨点，通用性强。③log-linear 的 scaling 曲线是最有说服力的结果——它把"合成数据量"变成一条可持续扩展的坐标轴，理论上比人工遥操作可扩展得多。④行为/环境泛化的零到一（0%→28.5%）证明视频模型确实把互联网视频里的"新动词"知识迁移进了机器人策略。⑤附带的 DreamGen Bench 把「视频模型好坏」与「下游策略好坏」用正相关连起来，方法论上有价值。

**局限与开放问题。** ①整条链路的上限被 **IDM/LAPA 伪动作质量**卡住——附录 A 明确指出瓶颈主要在 neural trajectory（即视频与动作一致性），而非 IDM 本身，这意味着一旦视频出现物理不合理的运动，伪动作就会系统性带偏策略。②纯合成仅 20.55% vs 真值高档 49.60%，说明合成轨迹离真值仍有明显差距，co-training 里真实数据仍不可或缺。③初始帧仍需人工，"完全无人"尚未实现。④与同期把"视频生成 + IDM/前向动力学"联合训练、支持视频-动作 co-training 的工作（UWM、Prediction-with-Action、GR-2、Unified Video Action 等）相比，DreamGen 刻意把各组件**解耦**以榨取最强视频模型，代价是无法端到端联合优化。

**可能的改进方向。** ①用 image-to-image/inpainting 自动合成并随机化初始帧，把最后一处人工环节也自动化;②在伪动作标注中引入物理/一致性过滤（如在数字孪生里 replay IDM 动作做质检，论文已用于 GR1 但未闭环）以剔除坏轨迹;③让 IDM 联合预测 state，补上合成轨迹缺失的本体感知;④把 DreamGen Bench 的 PA 指标接入生成阶段的 reward，做拒绝采样或后训练，直接提升下游可用率。总体上，这是一篇把"视频世界模型 → 机器人数据"这条路线做扎实、并给出 scaling 证据与配套基准的代表作，短板集中在伪动作保真度与初始帧自动化两处。

## 参考

1. Ye, Jang et al. *Latent Action Pretraining from Videos (LAPA)*, ICLR 2025 — DreamGen 潜动作标注所用方法。
2. Baker et al. *Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos*, NeurIPS 2022 — IDM 逆动力学标注的思想源头。
3. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, arXiv 2503.14734 — 主要下游策略与潜动作 embedding 用法。
4. Wang et al. *WAN: Open and Advanced Large-Scale Video Generative Models*, arXiv 2503.20314 — 默认视频世界模型底座。
5. Nasiriny et al. *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*, RSS 2024 — 仿真数据增广与 scaling 实验的基准。
