# WAM-RL：面向World-Action模型的强化学习——重建奖励与在线视频SFT

> **论文**：*WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT*
>
> **作者**：Zezhong Qian, Xiaowei Chi (Project Leader), Yu Qi, Haozhan Li, Zhi Yang Chen, Shanghang Zhang (Corresponding Author)
>
> **机构**：北京大学多媒体信息处理国家重点实验室（计算机学院）；东北大学；清华大学
>
> **发布时间**：2026 年 06 月（arXiv 2606.17906）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.17906) | [PDF](https://arxiv.org/pdf/2606.17906)
>
> **分类标签**：`World-Action Model` `强化学习` `在线视频SFT` `重建奖励` `KL正则化`

---

## 一句话总结
WAM-RL首次把强化学习引入World-Action（WA）范式，用"重建一致性"密集奖励在线优化actor、用带KL正则化的在线视频SFT微调world model，使二者协同进化；在Genie Envisioner-ACT架构上，LIBERO-Object成功率从68%（Base）经actor-only的πRL提升到78%，WAM-RL联合优化达到82%，RLBench Water Plants任务从19%提升到22%（而πRL反而降到18%）。

## 一、问题与动机
现有WA模型（Video Prediction Policy、Unified World Models、Genie Envisioner的GE-Act、Cosmos Policy、LingBot-VA、DreamZero等）通过联合建模未来观测与动作获得比传统VLA更强的泛化和数据效率，但训练几乎全部依赖专家轨迹的监督学习，带来两个根本限制：一是策略被训练数据支持集束缚，难以获得超出示范分布的精细操作技能；二是模型无法通过与环境交互持续改进，不能适应新场景或在线纠错。

直接引入RL并不trivial：WA模型由world model（生成未来预测）和action model/actor（把预测转成可执行动作）两个紧耦合部件构成。已有VLA-RL工作（ConRFT、CO-RFT、ARFM、RIPT-VLA、SimpleVLA-RL、πRL、TwinRL-VLA、VLA-RFT、GR-RL、PLD等）大多只优化policy侧，把视觉表征当作固定不变。但在WA模型里actor深度依赖world model的latent空间，若朴素地在线微调world model，latent分布漂移会让actor迅速失效，导致训练不稳定甚至性能倒退。

## 二、核心方法
WAM-RL基于一个关键观察：WA模型的核心能力主要来自world model（捕捉预测结构、支持隐式规划），actor更多扮演"翻译器"角色，把world model的latent预测转成可执行动作。据此设计了两条协同的优化机制。

**(1) World model：带KL正则化的在线视频SFT。** 用交互中收集的成功轨迹 $x_{1:T}$ 做标准视频建模自监督：

$$\mathcal{L}_{\text{video}} = \mathbb{E}_{x_{1:T}}\left[\ell\big(f_\theta(x_{<t}), x_t\big)\right]$$

大白话：让world model在真实成功案例上继续学"下一步会发生什么"，这样才能预见失败与恢复动作。但直接联合微调world model和actor会造成actor的latent输入分布剧烈漂移，使actor迅速失效。为此引入KL正则化，把当前特征 $z_t=f_\theta(x_{<t})$ 与冻结的预训练模型特征 $z_t^{\text{old}}$ 都近似为对角高斯（方差由EMA统计得到）：

$$\mathcal{L}_{\text{KL}} = \mathbb{E}_t\left[D_{\text{KL}}\Big(\mathcal{N}(z_t,\Sigma_\theta)\,\|\,\mathcal{N}(z_t^{\text{old}},\Sigma_{\text{old}})\Big)\right],\qquad \mathcal{L}_{\text{WM}} = \mathcal{L}_{\text{video}} + \lambda_{\text{KL}}\mathcal{L}_{\text{KL}}$$

大白话：约束新world model的latent几何形状不要偏离预训练模型太远，让actor"认得出"新的特征分布，代价是限制了world model能学到多远——论文自己承认这是稳定性与适应性之间的权衡，"改进虽一致但幅度有限"。

**(2) Actor：基于重建一致性的密集奖励RL。** 定义奖励为world model想象的未来观测与actor在真实环境中实际执行后得到的观测之间的相似度：

$$r_t = \mathrm{sim}(\hat{x}_{t+1:t+H}, x_{t+1:t+H})$$

大白话：不直接用任务成功与否的稀疏0/1信号，而是奖励actor"把world model脑内想象的画面真正演出来"——world model想象的隐式plan越被actor忠实执行，奖励越高。相似度函数试了像素级MSE、光流一致性、DINOv2特征相似度、V-JEPA2特征相似度四种。策略优化用标准策略梯度：

$$\nabla_\phi J = \mathbb{E}\left[\nabla_\phi \log\pi_\phi(a_t\mid s_t)\,A_t\right]$$

其中动作的对数似然通过把确定性flow matching ODE（$dx_t=v_\theta(x_t,t)dt$）转成Flow-SDE（$dx_t=v_\theta(x_t,t)dt+\sigma\,dW_t$）获得可处理的转移分布 $p(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma^2I)$，从而把flow-based动作模型的去噪过程视作latent空间的MDP（该技术源自πRL一类flow-RL方法）。

实现上，WAM-RL基于Genie Envisioner-ACT架构：world model是DiT视频生成器，actor消费其中间latent特征输出动作；训练用8卡NVIDIA A800，在线RL与视频微调混合训练8小时。

## 三、关键结果

主结果（Table 1，成功率）：

| 方法 | LIBERO-Object | RLBench (Water Plants) |
|---|---|---|
| Base（预训练WA模型，无RL） | 68% | 19% |
| πRL（仅优化actor的在线RL） | 78% | 18% |
| WAM-RL（本文，联合优化） | **82%** | **22%** |

在LIBERO-Object上WAM-RL比actor-only的πRL（78%）再高出4个百分点；在RLBench Water Plants上πRL反而比Base（19%）略降到18%，WAM-RL则提升到22%——论文以此支撑核心论点：仅优化actor在短程任务上有效，但难以应对长程/多步任务中累积的world model预测误差，唯有world model与actor联合进化才能在复杂任务上取得明显收益。

重建奖励消融（Table 2，RLBench Water Plants）：

| 相似度函数 | 成功率 |
|---|---|
| Base | 19% |
| πRL | 18% |
| Pixel MSE | **21%** |
| Optical Flow MSE | 19% |
| DINO MSE | 16% |
| V-JEPA2 | 17% |

一个反直觉发现：光流一致性在成功/失败轨迹间的奖励区分度（reward discriminability）最大，但下游任务成功率并非最高；像素级MSE区分度更弱，却拿到了最好的下游表现（21%）。作者解释为像素级损失与world model自身的训练目标（预测视觉观测）更对齐，且对OOD动作的像素误差惩罚更强，从而对策略起到更强的正则作用。

视频SFT消融为定性展示（Fig. 3）：无video SFT时，抓取失败后模型不预测纠正动作，轨迹持续漂移进入OOD直至任务失败；加入video SFT后，同一个open-loop action chunk内模型能预见失败并生成"重新定位夹爪、重新尝试抓取"的恢复动作，最终成功——这是论文对"world model质量决定WA模型能力上限"这一核心论点的关键定性证据，但未给出量化的长程任务基准对比。

## 四、评价与展望
**贡献与优点。** WAM-RL是较早（论文自称"首个"）把RL系统性引入World-Action范式并同时处理world model与actor联合优化的工作，填补了VLA-RL文献（ConRFT/RIPT-VLA/SimpleVLA-RL/πRL等）只优化policy侧、把视觉表征当固定项的空白。KL正则化在线视频SFT这一稳定化手段设计动机清晰，"actor-only RL在长程任务上收益有限、需要world model协同进化"这一实验洞察对该领域有一定的指导意义，也与作者组内此前的DreamZero（World Action Models are Zero-shot Policies）一脉相承，属于同一系列WA模型工作在RL阶段的自然延伸。

**局限性（论文自陈）。** 一是KL正则化虽然稳定了联合训练，但限制了world model能偏离预训练分布的幅度，在更大规模下可能制约能力上限扩展；二是重建奖励依赖预训练表征或人工设计的相似度度量，实验显示这些奖励在成功/失败轨迹间的区分度普遍有限，需要更具表达力、任务感知的奖励学习机制。

**开放问题与可能的改进方向。** (1) 奖励的自指性（self-referential）风险：reconstruction reward本质是"actor执行结果与world model自己想象结果的一致性"，若world model的想象本身有系统性偏差（如反复想象错误的抓取策略），该奖励可能强化而非纠正这类偏差，论文未讨论这一潜在的奖励攻陷（reward hacking）问题及规避手段。(2) 长程任务的量化验证不足：论文反复强调联合优化对长程任务至关重要，但主表两个基准（LIBERO-Object单任务、RLBench Water Plants单任务）均非专门的长程benchmark，支撑长程论点的证据主要是Fig. 3的单个定性案例，缺少专门的长程任务量化对比。(3) 训练规模有限：8×A800、8小时的在线RL与视频SFT，且仅两个任务、未见方差或多seed报告，泛化性和可重复性有待更大规模验证。(4) video SFT依赖"成功轨迹"做自训练，存在与self-training类似的冷启动问题——若Base模型在某任务上几乎从不成功，world model将缺乏可学习的成功样本，这一局限未被讨论。(5) 仅在仿真（LIBERO、RLBench）验证，未见真实机器人实验，真实世界中的重建奖励噪声（光照、传感器噪声）对不同相似度函数选择的影响仍是开放问题。

## 参考
- World Action Models are Zero-shot Policies（Ye et al., 2026）——本文延伸的DreamZero/WAM范式提出者
- Genie Envisioner（Liao et al., 2025）——本文所基于的GE-Act架构来源
- πRL: Online RL Fine-tuning for Flow-based VLA Models（Chen et al., 2026）——本文核心对比baseline（actor-only RL）及Flow-SDE技术来源
- Video Prediction Policy（Hu et al., 2025）——WA范式的早期代表工作
- Cosmos Policy（Kim et al., 2026）——同期WA模型代表工作，将预训练视频模型后训练为机器人策略
