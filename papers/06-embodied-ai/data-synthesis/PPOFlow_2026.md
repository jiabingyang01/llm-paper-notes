# PPOFlow：用生成式 3D 世界扩展机器人 VLA 的 Sim-to-Real 强化学习

> **论文**：*Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds*
>
> **作者**：Andrew Choi, Xinjie Wang, Zhizhong Su, Wei Xu（通讯作者）
>
> **机构**：Horizon Robotics（地平线机器人）
>
> **发布时间**：2026 年 03 月（arXiv 2603.18532，v2 2026-03-28）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.18532) | [PDF](https://arxiv.org/pdf/2603.18532)
>
> **分类标签**：`生成式仿真` `Sim-to-Real RL` `VLA` `flow matching` `零样本泛化`

---

## 一句话总结

用「语言驱动的场景设计器 + 3D 世界生成模型」自动批量造出上百个可交互的桌面操作场景，在 GPU 并行仿真里对预训练 flow-matching VLA（π₀）做稀疏奖励 RL 微调，并提出把多步 flow 策略转成单步 Gaussian 策略的 **PPOFlow** 算法；仿真成功率从 9.7% 提到 79.8%（+70.1 pt），真机成功率从 21.7% 提到 75%（+53.3 pt），并首次系统性地证明「训练场景多样性直接提升零样本泛化」。

## 一、问题与动机

训练机器人基座模型的「数据金字塔」第三层——任务/本体专属微调——最费人力。当前主流是**直接在真机上做 RL 微调**：它绕开了 sim-to-real gap，但在物理世界里扩展场景/物体多样性代价极高，于是几乎所有工作都只在**极窄的场景集合**上微调，把一个「广泛预训练、有泛化能力」的 VLA 反而**退化成只会做特定场景的过拟合策略**（论文称之为 paradoxical outcome）。

在仿真里训练能提供多样场景、自动 reset、特权信息，但**手工搭建 3D 环境同样昂贵**（一次性但可观的人力）。本文的核心问题因此是:**能否让 RL 微调覆盖尽可能宽的场景分布？如果能,场景多样性到底怎样影响零样本泛化?** 作者的答案是让生成式 3D 世界模型自动构造大规模可交互训练分布,从而突破以往「少数手工环境」的天花板。

## 二、核心方法

整套系统有两根支柱：(1) 生成式仿真**数据引擎**（把任务描述→上百个可交互 3D 场景）；(2) **PPOFlow** RL 算法（把 flow-matching VLA 高效微调成单步 Gaussian 策略）。

### 2.1 生成式仿真数据引擎

以 ManiSkill3 为 GPU 并行物理仿真器，扩展 EmbodiedGen 作为环境生成后端。流程为：

1. **语言驱动场景设计器（GPT-4o）**：给一句指令如「put the pen in the pen holder」，解析成一棵**结构化场景图**，节点带语义角色（background / context / distractors / manipulation targets / robot），边带空间关系（ON、IN 等）。这种图表示可灵活调控**组合复杂度**与 distractor 密度，从而系统性地调节训练难度。
2. **文本→3D 资产生成 + 布局合成**：EmbodiedGen 把场景图实例化为完整可交互 3D 世界。
3. **GPT-4o 驱动的 QA 闭环**：三级检查器——Semantic Appearance（前景图是否匹配目标类别与关键视觉属性，早期过滤，不合格立即重采 seed 重试）、Mesh Geometry（网格是否完整无重大几何缺陷）、Cross-modal Text-to-3D Alignment（最终 3D 资产是否与文本语义一致，捕捉 3D 生成引入的语义漂移）。不合物理/几何一致性的配置直接丢弃，避免仿真失稳。
4. **物体自动缩放**：WidowX 250S 夹爪最大开口仅 74 mm，超尺寸物体按包围盒自动缩小到可抓取，并降低网格分辨率简化接触计算。

### 2.2 MDP 与 π₀ 基座

建模为有限视界 POMDP。观测 $\mathbf{o}_t = [\mathbf{I}_t, \mathbf{e}_t, \mathbf{q}_t]$，即 RGB 图像、语言指令、末端位姿 $\mathbf{q}\in\mathbb{R}^7$（xyz、rpy、gripper）。动作为 action chunk $\mathbf{A}\in\mathbb{R}^{C\times 7}$，把整块 chunk 当作**一个决策步**，故动作空间 $\mathcal{A}\subset\mathbb{R}^{C\times 7}$。奖励是**稀疏、基于仿真状态的规则奖励**：

$$\text{success} = \text{contact}(A,B)\ \wedge\ \neg\text{contact}(A,\text{table})\ \wedge\ \neg\text{contact}(A,\text{robot})$$

用大白话说：成功 = 被抓物 $A$ 碰到目标物 $B$，同时既不碰桌子也不碰机器人（即被真正拎起来放到目标上）。

基座用 π₀（VLM backbone $E_\theta$ = SigLip 400M + Gemma 2B，action expert $v_\theta$ = 300M），在 BridgeV2 上预训练。用 rectified flow-matching 目标：

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}\left[\left\lVert v_\theta(\mathbf{A}_t^\tau, KV_\theta(\mathbf{o}_t), \tau) - (\mathbf{A}_t^1 - \boldsymbol{\epsilon})\right\rVert_2^2\right]$$

其中 $\mathbf{A}_t^\tau = \tau\mathbf{A}_t^1 + (1-\tau)\boldsymbol{\epsilon}$，$\tau\in[0,1]$ 是连续积分时间，$KV_\theta(\mathbf{o}_t)$ 是 VLM 供 action expert 交叉注意力的 key-value。用大白话说：在真实动作 $\mathbf{A}_t^1$ 和噪声 $\boldsymbol{\epsilon}$ 之间连一条直线，网络学着预测这条直线的「速度」，推理时从纯噪声出发数值积分 $K$ 步（$K=1/\Delta\tau$）就得到动作。这条预训练模仿策略记作 $\pi_{\text{pre}}$（$K=10$、$C=4$）。

### 2.3 PPOFlow：把确定性 flow 变成可求重要性比的 Gaussian 策略

flow matching 是**确定性**的，算不出 PPO 需要的重要性比。借鉴 ReinFlow，在每个积分步注入可学噪声 $\sigma_\phi$，把确定性欧拉步变成一次 Gaussian 采样：

$$\hat{\mathbf{A}} = \mathbf{A}_t^\tau + v_\theta(\mathbf{A}_t^\tau, KV_\theta(\mathbf{o}_t), \tau)\Delta\tau,\qquad \mathbf{A}_t^{\tau+\Delta\tau}\sim\mathcal{N}\big(\hat{\mathbf{A}},\ \sigma_\phi(\mathbf{A}_t^\tau, \mathbf{z}_t, \tau)\big)$$

其中 $\mathbf{z}_t = \text{sg}(E_\theta(\mathbf{o}_t))$ 是停梯度的隐状态。这样整条去噪链的联合对数概率可解析写出：

$$\log\pi(\mathbf{A}_t^0,\dots,\mathbf{A}_t^1\mid\mathbf{o}_t) = \log\mathcal{N}(\mathbf{0},\mathbf{I}) + \sum_{k=0}^{K-1}\log\pi\big(\mathbf{A}_t^{(k+1)\Delta\tau}\mid\mathbf{A}_t^{k\Delta\tau}, \mathbf{o}_t\big)$$

再加一个 value head $V_\psi$，用一个**幂次缩放**的重要性比配合标准 PPO clip 目标：

$$\hat{r}_t = \left(\frac{\pi_\theta(\mathbf{A}_t^0,\dots,\mathbf{A}_t^1\mid\mathbf{o}_t)}{\pi_{\theta,\text{old}}(\mathbf{A}_t^0,\dots,\mathbf{A}_t^1\mid\mathbf{o}_t)}\right)^{s},\qquad \mathcal{L}_{\text{PPOFlow}} = \mathbb{E}_t\left[\min\big(\hat{r}_t\hat{A}_t,\ \text{clip}(\hat{r}_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\right]$$

$s\in(0,1)$（取 0.2）用大白话说：整条链的 log-prob 是很多步累加，比值容易爆成天文数字导致几乎全被 clip、梯度全丢；开个 $s$ 次幂把比值压回稳定数值区间，是训练不立刻崩溃的关键。作者据此把 $K$ 直接设成 **1**——即把 π₀ 从多步 flow 策略直接坍缩成**单步 Gaussian 策略**（以往被认为会损害性能），实测反而更稳、推理更快（见 §3）。

### 2.4 sim-to-real 三板斧与训练配置

浅层模型的 sim-to-real 经验被证明可迁移到 VLA：(1) 高保真生成 3D 物体/场景；(2) 简单 domain randomization（物体 xy 位/yaw、环境光 RGB、相机 xyz、方向光亮度、机器人高度/关节扰动）；(3) **PD 控制 + 重力补偿**——开了重力补偿后，只要 sim 与真机控制器都能以足够低跟踪误差到达大多数目标位姿，**显式系统辨识基本可省**，动力学 gap 被压到最小。训练用 8×RTX 6000 Ada 对 VLM 做 LoRA（rank 32）、对 action expert 做全量微调，跑约 5 天；192 并行环境、batch 19200、5 Hz 控制、$\gamma=0.99$。

## 三、实验结果

任务为 pick-and-place。生成 100 个独一场景组成全集 $\mathcal{W}$；随机采三份各 50 场景的子集 $\mathcal{H}_i$，其 OOD 集 $\bar{\mathcal{H}}_i = \mathcal{W}\setminus\mathcal{H}_i$；对 $N\in\{1,3,10,25,50\}$ 各跑三个独立种子（用 $\mathcal{H}_i$ 的前 $N$ 个场景），另在全集 $\mathcal{W}$ 上跑 $N=100$。EG=EmbodiedGen 生成场景，SE=SimplerEnv 三个手工 Bridge 桌面场景。

### 3.1 仿真：场景数 N 的影响（成功率 SR %）

| 训练 | EG 分布内 SR | EG OOD SR | EG 全集 SR | SE SR |
|---|---|---|---|---|
| $N=0$（$\pi_{\text{pre}}$ 模仿基线） | – | 9.6 | 9.7 | 23.7 |
| $N=3$（SE，3 个手工场景基线） | – | 36.5 | 36.0 | **96.7** |
| $N=1$ | **94.3** | 53.2 | 51.6 | 36.1 |
| $N=3$ | 88.7 | 61.3 | 60.0 | 47.4 |
| $N=10$ | 87.0 | 72.4 | 72.1 | 54.3 |
| $N=50$ | 80.5 | 77.9 | 79.2 | 68.4 |
| $N=100$ | – | – | **79.8** | 74.3 |

关键读数：

- **多样性直接买泛化**：$N$ 越大，分布内 SR 从 94.3% 单调下滑到 80.5%（因固定 batch 下每场景样本 $\propto 1/N$），但 OOD 从 53.2% 升到 77.9%、SE 从 36.1% 升到 68.4%。EG 分布内与 OOD 的 gap 随 $N$ 从 **41.1 pt→27.4→14.6→2.6 pt** 收窄；EG 与 SE 的 gap 从 58.2 pt 降到 12.1 pt。贡献 2 的「50 场景比单场景零样本 +24.7 pt」即 OOD 53.2→77.9。
- **手工场景覆盖太窄**：只用 3 个手工 SE 场景训练，在 SE 上高达 96.7%，但到 EG 上只有 36.0%，**60.7 pt 断崖**，凸显生成场景多样性的价值。
- **RL 相对模仿基线的绝对增益**：$N=100$ 把 EG 成功率 9.7%→79.8%（**+70.1 pt**），完成时间约 10 s→8 s（**1.25×**）；且 $N=100$ 在从未训过的 SE 上拿到最好的 74.3%。

### 3.2 真机 sim-to-real（12 场景、240 次试验，WidowX 250S + 单目 C922）

| 指标 | $\pi_{\text{pre}}$ 模仿基线 | $N=100$ RL |
|---|---|---|
| 部分成功率 PSR（正确抓起） | 0.45 | **0.883** |
| 整体成功率 SR | 0.217 | **0.75** |
| 动力学失败率 DFR ↓ | 0.667 | **0.183** |
| 语义失败率 SFR ↓ | 0.183 | **0.067** |
| 完成时间 TF (s) ↓ | 11.5 | **10.2** |

真机 SR 从 21.7%→75%（**+53.3 pt**），TF 11.5→10.2 s（**1.13×**）。DFR 66.7%→18.3% 说明抓取鲁棒性大幅提升；SFR 18.3%→6.7% 说明任务 grounding 也变好。在**未参与 RL 训练**的 OOD 物体上仍迁移良好：场景 10（screwdriver，训练分布外）RL 50% vs 基线 0%；场景 11（teacup 堆叠，未见实例）RL 50% vs 20%。

### 3.3 消融：多模态到底重不重要（$K$ 的影响）

| $K$ | RL SR (%) | 反向传播 (s) ↓ | 推理延迟 reg / compile (s) ↓ |
|---|---|---|---|
| 10 | – | – | 0.267 / 0.172 |
| 4 | 77.29 | 108.80 | 0.153 / 0.107 |
| 2 | 77.23 | 87.55 | 0.120 / 0.086 |
| 1 | **79.75** | **74.74** | **0.098 / 0.073** |

单步 $K=1$（单峰 Gaussian）SR 反而最高，且相对 $K=4$ 反传快 1.17×、相对 $K=10$ 推理快 2.72×（torch.compile 下 **2.36×**）。作者据此呼应 Pan 等人观点：**多模态不是 diffusion/flow 机器人策略高性能的关键**——多模态在模仿学习（示范分布本身多模态）时有用，但在奖励引导优化下单峰策略已足够。

### 3.4 生成引擎的量化开销（Appendix C）

100 个环境共产出 **516** 个独一 3D 资产，平均每场景 **5.16** 个可交互物体。GPT-4o QA 单次通过率：Semantic Appearance 83.3%、Mesh Geometry 75.2%、Cross-modal Alignment 91.9%，平均 **1.37** 次尝试即满足全部约束，**85%** 的环境无需人工干预可直接用于端到端 RL。单张 RTX 4090 全在线顺序生成 **46.8±5.0 min/场景**（背景 25.0 min、单物体 3.9 min）；若从预建资产库复用可交互资产，则降到约 **2 min/场景**。

## 四、局限性

- **仅限 pick-and-place**：受 EmbodiedGen 当前生成能力所限，只做了抓放任务（虽然抓放占 BridgeV2 逾 70%）；铰接物体操作、工具使用、多阶段任务尚未支持。
- **生成成本仍高**：全在线生成近 47 min/场景，靠资产库复用才降到 2 min；且 QA 后仍有约 15% 场景需人工修正（尺度不匹配、初始摆放不稳）。
- **基座依赖第三方权重**：$\pi_{\text{pre}}$ 用的是第三方 open-pi-zero 在 BridgeV2 上的 checkpoint，而非 Physical Intelligence 官方 π₀；作者还报告官方 π₀.₅ 微调后跨种子反而更差，说明结论对基座选择有一定敏感性。
- **单一本体、单目视觉、稀疏奖励**：只在 WidowX + 单目 C922 上验证；90% 以上成功率是「非饱和」目标下的结果，作者明确不追求窄场景刷到 >90%。
- **N 增大时分布内 SR 下滑**未在正文根治，仅假设「同步放大 batch/mini-batch 可缓解」，未做实验证明。

## 五、评价与展望

**优点。**（1）把「生成式 3D 世界模型」当作 RL 微调的**可扩展数据分布源**，而非静态视觉资产（对比 TRELLIS/WorldGen 只出静态资产、Holodeck/RoboGen 交互性有限），并首次给出「场景多样性→零样本泛化」的清晰单调 scaling 证据（ID-OOD gap 41.1→2.6 pt），这是本文最有分量的经验发现。（2）PPOFlow 用幂次缩放重要性比稳住了大模型 flow 策略的 PPO 微调，并给出「$K=1$ 单步 Gaussian 不损性能且推理快 2.36×」这一反直觉但工程上极实用的结论，与 Diffusion Steering、FPO、consistency policy 等一路呼应。（3）sim-to-real 三板斧（生成资产 + 简单 DR + 重力补偿免 sysID）把浅层模型经验成功搬到基座级 VLA，真机 +53.3 pt 有说服力。

**缺点与开放问题。**（i）与 ReBot（real-to-sim-to-real 视频合成）、VLA-RFT（learned world model 内学习）等同期路线相比，本文走的是「纯生成场景 + 显式物理仿真」，缺一个与这两类方法的正面对照实验。（ii）GPT-4o 既做场景设计又做 QA，是否引入**生成分布的系统性偏置**（如物体类别集中在常见厨房桌面物）未被审视；生成场景的「多样性」目前只是物体/颜色/干扰项层面的组合多样，尚未触及物理属性（质量、摩擦、可变形）多样性。（iii）稀疏 contact 奖励对铰接/多阶段任务不可直接推广，这也是把该 pipeline 推向更宽任务分布的主要瓶颈。（iv）「单步 Gaussian 足够」的结论建立在单次运行、抓放任务上，是否在接触丰富的灵巧操作上仍成立值得追问。总体上，这是一篇**把生成式仿真真正当数据引擎、并用干净 scaling 曲线证明其价值**的扎实工作，指出的方向（自动扩展场景/任务分布以撬动 VLA 泛化）比其具体算法更值得关注。

## 参考

1. K. Black et al. *π₀: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.（本文 flow-matching VLA 基座）
2. X. Wang et al. *EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence.* arXiv:2506.10600, 2025.（场景生成后端）
3. T. Zhang, C. Yu, S. Su, Y. Wang. *ReinFlow: Fine-tuning Flow Matching Policy with Online RL.* NeurIPS 2025.（PPOFlow 的直接前身）
4. S. Tao et al. *ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI.* RSS 2025.（并行仿真器）
5. C. Pan et al. *Much Ado About Noising: Dispelling the Myths of Generative Robotic Control.* arXiv:2512.01809, 2025.（"多模态非关键"的呼应工作）
