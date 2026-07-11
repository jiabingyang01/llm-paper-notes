# Vidarc：面向闭环控制的具身视频扩散模型

> **论文**：*Vidarc: Embodied Video Diffusion Model for Closed-loop Control*
>
> **作者**：Yao Feng, Chendong Xiang（共同一作）, Xinyi Mao, Hengkai Tan, Zuyue Zhang, Shuhe Huang, Kaiwen Zheng, Haitian Liu, Hang Su, Jun Zhu（Hang Su、Jun Zhu 为通讯作者）
>
> **机构**：清华大学计算机科学与技术系 / 人工智能研究院（BNRist Center, THBI Lab, 清华-博世联合机器学习中心）；清华大学建筑学院
>
> **发布时间**：2025 年 12 月（arXiv 2512.17661）
>
> **发表状态**：未录用（预印本，论文标注 "Preprint. Under review."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.17661) | [PDF](https://arxiv.org/pdf/2512.17661)
>
> **分类标签**：`视频扩散世界模型` `闭环控制` `逆动力学模型` `自回归生成` `双臂操作` `KV Cache 加速`

---

## 一句话总结

Vidarc 把双臂操作用的视频扩散世界模型（Vidar）从"离线批量生成"改造成"因果自回归 + KV cache 重填（re-prefill）"的闭环控制器,并用掩码逆动力学模型（MIDM）反哺一个 embodiment-aware 扩散损失聚焦机械臂区域,在 1M 跨本体预训练数据基础上,于 RoboTwin 仿真基准（14 任务）取得 80.7% 平均成功率（对比 Vidar 71.1%、Pi0.5 52.9%）,真机部署成功率比 Vidar 高 17 个百分点、比 Pi0.5 高 15 个百分点,同时把端到端延迟比 Vidar 降低 91%。

## 一、问题与动机

机械臂操作在数据稀缺场景下极具挑战:采集大规模高质量示教数据成本高昂,难以随平台/任务/环境快速迁移。近年"预训练基础模型 + 少样本微调"路线中,视频生成模型因为能利用互联网规模视频数据、捕捉时序物理交互而备受关注(相对于必须依赖大量人类示教的 VLA 路线)。但纯视频生成式世界模型有两个关键短板一直未被很好解决:

1. **闭环控制难**:多数视频世界模型采用双向（bidirectional）扩散架构,缺乏因果性,推理延迟高、只能做开环预测或慢速串行推理,难以把实时环境反馈接入生成过程;
2. **grounding 不足**:纯视频生成模型对具身相关的动态和视觉特征缺乏针对性建模,机械臂外观/位姿的细微视觉偏差就可能导致任务失败,而背景区域的生成误差会拖累有效信息密度。

作者的前作 Vidar（Feng et al., 2025, arXiv:2507.12898）已经提出用掩码逆动力学模型（Masked Inverse Dynamics Model, MIDM）从生成的视频帧解码动作,但 Vidar 依然是双向、非因果的批量生成范式,延迟高、闭环能力弱。Vidarc 正是要解决"如何让视频扩散世界模型具备低延迟、可闭环、且专注本体相关区域"这三件事。

## 二、核心方法

### 2.1 整体流程

Vidarc 沿用"策略 = 视频生成模型 ∘ 逆动力学模型"的分解:设语言指令空间为 $\mathcal{L}$,分块观测空间为 $\mathcal{O}$,分块动作空间为 $\mathcal{A}$,策略 $\pi:\mathcal{L}\times\mathcal{O}\to\mathbb{P}(\mathcal{A})$ 被分解为视频生成模型 $G:\mathcal{L}\times\mathcal{O}\to\mathbb{P}(\mathcal{O})$ 与逆动力学模型 $I:\mathcal{O}\to\mathcal{A}$。引入环境反馈后,策略按时间步展开为:

$$
\begin{cases}
\hat{o}_{t+1} \sim G(l, \mathcal{C}(o_1,\cdots,o_t)) & \text{自回归生成} \\
a_t = I(\hat{o}_{t+1}) & \text{逆动力学解码} \\
o_{t+1} = \mathcal{T}(o_t, a_t) & \text{执行并采集}
\end{cases}
$$

**大白话**:先用视频模型"想象"下一帧观测,再用逆动力学模型从这帧想象图里"读出"应该执行的动作,执行后拿到真实的下一帧观测,循环往复——这就是把视频生成模型改造成闭环策略的基本骨架。

### 2.2 掩码逆动力学模型（MIDM,继承自 Vidar）

只有机械臂相关像素区域对动作预测有用,其余区域可能引入噪声。MIDM 用可学习的掩码预测器 $U$ 输出动作相关掩码 $m\in[0,1]$,再用动作回归器 $R$ 从掩码后的图像回归动作:

$$m = U(x), \quad \hat{a} = R(\mathrm{Round}(m) \odot x)$$

训练目标为 Huber 回归损失加掩码面积的 $L_1$ 正则（权重 $\lambda$,抑制掩码"作弊式"覆盖全图）:

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{x,a}\left[l(\hat{a}-a) + \lambda\|m\|_1\right]$$

**大白话**:让模型自己学会"该盯着画面哪块区域才能猜出机械臂在做什么",顺便把这块区域的位置信息反馈给视频生成器用。

### 2.3 因果自回归训练（Causal Training）

Vidarc 用 CausVid 方法把双向的文生视频扩散模型（Wan2.2 backbone,5B 参数）转成逐帧生成的因果模型:生成当前帧时,此前所有帧 $x_{\text{prev}}$ 都是已去噪的干净帧,可直接被注意力关注:

$$\mathcal{L}_{\text{causal}} = \mathbb{E}_{x_0,x_1,t,c}\left[\|v_\theta(x_t,t,c,x_{\text{prev}}) - (x_0-x_1)\|_2^2\right]$$

**大白话**:训练时不再"看完整段视频再统一去噪",而是模拟真实推理时"边生成边把前面已经生成好的干净帧当条件"的场景,消除训练-推理不一致。

### 2.4 Embodiment-aware 扩散损失

视频预测常在机械臂区域出现伪影（论文 Figure 3 展示了三个真实案例）,影响任务成功率。Vidarc 用 MIDM 学到的掩码 $m$ 反过来重加权扩散损失,让视频模型更关注动作相关区域:

$$\mathcal{L}_{\text{embodiment-aware}} = \mathbb{E}_{x_0,x_1,t,c}\left[\|(1+\eta\cdot U(x_1))\odot(v_\theta(x_t,t,c,x_{\text{prev}})-(x_0-x_1))\|_2^2\right]$$

其中 $\eta$ 是控制重加权强度的超参数,消融实验显示 $\eta=0$（退化为普通扩散损失）成功率 74.6%,$\eta=3$（默认）80.7%,$\eta=10$ 77.1%——在较宽范围内都保持较高成功率。

**大白话**:哪里是机械臂,哪里的预测误差就被放大惩罚,逼着视频生成器把"画质预算"优先花在动作相关的地方,而不是均匀关注全图背景。

### 2.5 KV Cache 重填闭环推理（Re-prefill）

完整推理算法（论文 Algorithm 1）:每轮用 KV cache 自回归生成 $n_c$ 帧构成一个动作 chunk,用 IDM 解码出动作序列并在环境中执行,采集到真实观测后,**不是重新计算整段历史的 KV cache**,而是弹出最后一个生成 chunk 的 KV cache,只用最新采集到的真实观测做"块级重填”（chunk prefill）,大幅降低重新预填充（prefill）的计算量。这样保证下一轮生成始终以"真实环境反馈"作为条件,而不是持续基于自己此前的想象续写,从而防止误差累积(论文 Figure 5/6 的案例研究直观展示:有重填时,第 47 帧模型的错误想象被真实观测纠正,任务成功;无重填时,想象与执行的偏差不断累积,最终任务失败)。

**大白话**:每执行完一小段动作,就拿真实拍到的画面"更新记忆",而不是让模型在自己脑补的画面基础上一直往下编——这是保证闭环、防止"越想越错"的关键。

## 三、实验结果

**硬件与数据**:目标平台为 Aloha 双臂机器人（3 相机、14 自由度、单臂载荷 1.0kg、臂展 0.6m）,动作空间为绝对关节位置目标（不依赖历史）。预训练数据汇聚 4 个公开来源共约 100 万条视频片段:Egodex（230,949 条,人类第一视角）、AgiBot（728,209 条,Genie-1 机器人）、RDT（6,083 条,Aloha）、RoboMind（Franka 9,589 条 + Aloha 7,272 条）。微调用两套领域数据:RoboTwin 仿真（50 任务each 20 条,共 1,000 条,Agilex Aloha 平台）和自采真机数据集"Vidarc"（219 任务,共 2,307 条,Aloha 平台,相机与机械臂配置与预训练数据完全不同）。

**基线**:(1) Vidar——用同样的 Wan2.2 backbone 复现,同样经过 10k 步预训练 + 14k 步下游微调;(2) Pi0.5——强 VLA 基线,2B 参数,在全部下游数据上联合微调。Vidarc 本身基于微调后的 Vidar 权重热启动,再额外做 4k 步因果化微调。IDM 为 92M 参数,单独训练 60k 步,$\lambda=3\times10^{-3}$,在 Vidar 与 Vidarc 之间共享。

**RoboTwin 仿真基准（14 任务,每任务 20 episode）**:

| Method | 平均成功率 | Handover Mic | Open Laptop | Place Can Basket | Place Cans Plasticbox |
|---|---|---|---|---|---|
| Pi0.5 | 52.9% | 20.0% | 30.0% | 35.0% | 15.0% |
| Vidar | 71.1% | 0.0% | 50.0% | 50.0% | 0.0% |
| **Vidarc** | **80.7%** | 65.0% | 55.0% | 45.0% | 85.0% |
| Vidarc w/o Embodiment-aware | 74.6% | 50.0% | 65.0% | 20.0% | 70.0% |
| Vidarc w/o Closed-loop | 66.8% | 25.0% | 40.0% | 35.0% | 50.0% |

14 任务全表（Table 6,节选若干）:Click Alarmclock 三方法均 100%（Vidar/Vidarc）；Grab Roller 95%（Vidarc）vs 100%（Vidar）vs 75%（Pi0.5）；Place A2B Left 三者均较低（10/45/35%）,是三种方法共同的薄弱任务。总体 Vidarc 在需要精细双臂协作的任务（如 Handover Mic、Place Cans Plasticbox）上相对 Vidar 提升尤为明显,体现闭环控制对高精度任务的价值。

**真机部署（三类场景:Seen / Unseen / Dynamic,Dynamic 指执行过程中人为改变目标物体位置）**:

| Method | Average | Seen | Unseen | Dynamic |
|---|---|---|---|---|
| Pi0.5 | 41.0% | 48.0% | 28.0% | 48.0% |
| Vidar | 39.0% | 72.0% | 44.0% | 0.0% |
| **Vidarc** | **56.0%** | 72.0% | 56.0% | 40.0% |

Vidar 在开环设定下 Dynamic 场景成功率为 0%——完全无法应对执行中的环境变化;Vidarc 借助闭环重填机制把 Dynamic 成功率拉到 40%,验证了 C2/C3 主张（泛化到未见场景 + 低延迟纠错）。但值得注意,Pi0.5 在 Dynamic 场景反而略高于 Vidarc（48% vs 40%,见"四、局限性"）。

**推理速度（单卡 NVIDIA A100,统一 6.4 秒执行时长）**:

| Method | Latency (s) | Prefill Cost | VAE Cost | Diffusion Cost | End-to-end Cost |
|---|---|---|---|---|---|
| Pi0.5 | 0.482 | - | - | - | 5.76 |
| Vidar | 34.3 | - | 6.25 | 26.9 | 34.3 |
| **Vidarc** | **3.03** | 0.896 | 6.45 | 10.3 | 24.2 |

Vidarc 相对 Vidar 延迟降低 91%（34.3s → 3.03s）,主要得益于因果生成 + KV cache 重填（重填相比全量 prefill 再省 6% 端到端耗时,25.8s → 24.2s）；但绝对延迟（3.03s/chunk）仍远高于原生 VLA（Pi0.5 的 0.482s）,说明视频扩散路线在实时性上依然有代差。

**超参数敏感性（$\eta$,Table 9）**:$\eta\in\{0,3,10\}$ 对应平均成功率 74.6% / 80.7% / 77.1%,说明 embodiment-aware 损失在较宽范围内有效且鲁棒,默认 $\eta=3$ 最优。

## 四、局限性

- **绝对延迟仍高于原生 VLA**:虽然比 Vidar 降低 91%,但 Vidarc 单次 chunk 生成仍需 3 秒量级,相比 Pi0.5 的 0.48 秒有数量级差距,实时性并非真正"实时"，论文自己也承认需依赖"硬件进步与量化/蒸馏等进一步优化"。
- **动态场景仍非全面领先**:真机 Dynamic 场景中 Vidarc（40%）低于 Pi0.5（48%）,说明闭环视频重填机制虽显著优于无闭环的 Vidar（0%）,但相比专门为高频闭环控制设计的 VLA,在快速动态扰动下仍有差距。
- **仍是"数据饥渴"的两阶段范式**:预训练用了约 100 万条跨本体视频（4,500 A100 GPU 小时）,微调阶段仍需数千条目标平台示教数据（RoboTwin 1,000 条 + 真机 2,307 条）,与论文动机中强调的"数据稀缺场景"存在一定张力。
- **MIDM 依赖单独训练且强绑定本体**:掩码逆动力学模型需要 60k 步单独训练,迁移到新本体/新相机配置时需要重新标定（论文摘要中提到"embodiment-specific masks"的微调步骤),不是即插即用。
- **评测覆盖有限**:仅在 Aloha 双臂平台（仿真 RoboTwin + 自建真机集)上验证,未展示向单臂、非 Aloha 构型（如 Franka、人形)的迁移效果;基线只对比 Vidar 与 Pi0.5 两个方法,未与更多闭环视频策略（如 VidMan、VPP）做直接性能对比,只在 Related Work 中定性讨论。
- Ethics Statement 中作者也指出,低延迟纠错型通用策略部署于家庭等敏感场景存在安全与隐私风险,论文未做进一步技术性缓解讨论。

## 五、评价与展望

Vidarc 的贡献可以概括为对 Vidar 的"闭环化 + 本体感知化"改造:核心创新不在于提出全新的生成范式,而在于系统性地解决了视频世界模型做闭环控制时的两个工程/建模痛点——因果化+KV cache 重填解决延迟与训练-推理 gap,embodiment-aware loss 解决"注意力被背景稀释"的问题。两个改造分别都有清晰的消融证据（Table 1 的 w/o Embodiment-aware 和 w/o Closed-loop 两行）,思路干净,可复现性较好（作者声明将开源代码与权重）。

与相关工作的关系:CausVid、Self Forcing、Diffusion Forcing、MAGI-1 等提供了因果自回归视频扩散的通用训练技巧,Vidarc 是这些技巧首次被系统迁移到"视频驱动的机器人闭环控制"场景,并叠加了动作维度的 IDM 反馈闭环,这是相对新颖的组合。与追求效率优先的 VidMan、VPP 相比,作者指出后者不在完整视觉观测空间内建模任务、能力受限;Vidarc 坚持在像素空间生成完整视频再解码动作,代价是仍比这类轻量方法慢,但换来了更强的可解释性（可以直接看到"模型在想象什么")和视频生成模型固有的强泛化先验。

开放问题与可能的改进方向:(1) 3 秒级的 chunk 延迟距离高频闭环（如 30Hz)控制仍有明显差距,单步/少步扩散蒸馏或许是下一步的自然选择;(2) embodiment-aware loss 目前依赖同结构 MIDM 提供掩码,能否用更通用的显著性/光流估计替代,从而降低新本体的标定成本,值得探索;(3) 论文的动态场景实验只是"人为移动目标物体"这一种扰动,更系统的鲁棒性评测（遮挡、光照变化、多物体干扰）尚缺;(4) 真机 Dynamic 场景中 Vidarc 仍不敌 Pi0.5,提示"视频想象 + 逆动力学解码"这条链路在快速反应场景中可能存在结构性延迟劣势,值得后续工作专门分析闭环频率与任务成功率的关系。

## 参考

1. Feng et al. *Vidar: Embodied Video Diffusion Model for Generalist Bimanual Manipulation*. arXiv:2507.12898, 2025.（Vidarc 直接基线与骨架来源)
2. Huang et al. *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion*. arXiv:2506.08009, 2025.
3. Yin et al. *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid)*. CVPR 2025.（因果训练方法来源)
4. Black et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054, 2025.（对比 VLA 基线)
5. Wang et al. *Wan: Open and Advanced Large-Scale Video Generative Models*. arXiv:2503.20314, 2025.（视频扩散 backbone)
