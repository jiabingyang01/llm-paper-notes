# WEAVER：更好、更快、更长——一个高效的机器人操作世界模型

> **论文**：*WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation*
>
> **作者**：Arnav Kumar Jain\*、Yilin Wu\*、Jesse Farebrother、Gokul Swamy、Andrea Bajcsy（\* 共同一作）
>
> **机构**：Mila - Québec AI Institute、Université de Montréal、Carnegie Mellon University、McGill University
>
> **发布时间**：2026 年 06 月（arXiv 2606.13672，v2 于 2026-06-16）
>
> **发表状态**：未录用（预印本，标注 Preprint）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.13672) | [PDF](https://arxiv.org/pdf/2606.13672)
>
> **分类标签**：`世界模型` `机器人操作` `多视角预测` `flow-matching` `测试时规划`

---

## 一句话总结

WEAVER（World Estimation Across Views for Embodied Reasoning）把视频生成社区的 diffusion forcing / flow-matching / 预训练编码器、latent world model 的 reward-critic 头、JEPA 的 latent 预测目标、以及 Ctrl-World 的多视角与记忆架构缝合成一个 928M 的多视角机器人操作世界模型,首次让 fidelity（保真）、consistency（长时一致）、efficiency（高效）三个诉求同时达标——在真实 Franka 硬件上取得策略评估 $\rho=0.870$ 的成功率相关性、纯世界模型合成数据把 $\pi_{0.5}$ 基座策略成功率提升 38%（0.44 → 0.82）、并以比 Ctrl-World 快 $5\text{--}10\times$（延迟层面约 $20\times$)的速度解锁测试时规划。

## 一、问题与动机

世界模型(WM,即"可学习的模拟器")对机器人的三大承诺是:**策略评估**(不必真机 rollout 就能估算策略好坏)、**策略改进**(用想象轨迹合成数据回灌策略)、**测试时规划**(把 WM 放进搜索/规划循环)。要同时兑现这三点,作者论证一个机器人 WM 必须**联合满足**三个诉求:

- **(i) fidelity 保真**:生成的轨迹在物理上与真实结果相关(否则评估/改进不可信);
- **(ii) consistency 长时一致**:长 horizon 下保持时序连贯(多阶段任务、遮挡场景尤其难);
- **(iii) efficiency 高效**:生成足够快,以满足实时规划的延迟约束。

现有工作各有短板:视频生成模型保真高但太慢;JEPA 式 WM 的 latent 无法被解码成评估任意 visuomotor 策略所需的图像;Dreamer-v4 从头学编码器损害 OOD 鲁棒性;而操作领域最强的 Ctrl-World 运行速度远慢于真机,使其难以用于测试时规划和策略改进。机器人操作还额外带来多视角、从历史推断被遮挡物体、以及需要预测真实世界状态(而不只是"好看")的复杂性。**没有任何已有机器人 WM 同时满足三诉求**——这是本文要填的空。

## 二、核心方法

### 2.1 setup 与整体架构

任务由自然语言指令 $\ell$ 指定。机器人本体状态记为 $q\in\mathbb{R}^8$(7-DoF 关节 + 夹爪),场景有 $n$ 个 RGB 视角 $\mathbf{I}:=(I^1,\dots,I^n)$。时刻 $t$ 的观测 $o_t:=(\mathbf{I}_t,q_t)$。基座策略 $\pi_\theta$ 给出 $h$ 步 action chunk $\mathbf{a}_t:=a_{t:t+h}$,在 WM 或真实环境里执行。

WM 用**预训练编码器**(Stable Diffusion 3 的 VAE encoder)把观测 $o_t$ 映射为 latent $z_t\in\mathcal{Z}$。关键设计是同时以两组条件驱动生成:一是**长期稀疏记忆** $\mathbf{z}_t^{\text{mem}}:=(\dots,z_{t-2k},z_{t-k})$(每隔 $k$ 帧取一帧,捕捉长期上下文),二是 **$m$ 步短期历史** $\mathbf{z}_t^{\text{hist}}:=(z_{t-m},\dots,z_t)$(捕捉近期后果)。给定记忆、历史与 $h$ 步动作计划 $\mathbf{a}_l$,WM 预测 $h$ 个未来 latent:

$$
\hat{\mathbf{z}}_t \sim f_\phi(\cdot \mid \mathbf{z}_t^{\text{mem}}, \mathbf{z}_t^{\text{hist}}, \mathbf{a}_l), \quad \hat{\mathbf{z}}_t := \hat{z}_{t+1:t+h+1}
$$

同时训练一个 reward model $\hat{r}_t\sim R(\cdot\mid\hat{\mathbf{z}}_t,\ell)$ 评分 latent 与指令的对齐度。为让策略能被反复调用,用**预训练解码器** $\hat{o}_t\sim\mathcal{D}_\eta(\hat{\mathbf{z}}_t)$ 把 latent 解回观测(相机视图 + 本体状态),把 $\hat{o}_{t+h+1}$ 喂回策略生成下一段动作。

> 用大白话说:这就像给机器人配了一个"脑内电影放映机"。它把每一帧压缩成一小段编码(latent),记住"很久以前的几个关键镜头(稀疏记忆)"和"刚刚发生的几帧(短期历史)",然后根据你想让机器人做的一串动作,预测出接下来几帧会长什么样,还顺便打个分说"这一串动作干得好不好"。

### 2.2 高保真、时序一致生成的关键设计

- **多视角相机预测**:同时预测外部相机与腕部相机,腕部视角处理遮挡、外部视角处理全局一致性。每个视角经 SD3 VAE 编码为 $H\times W$ patch tokens,本体状态 $q_t$ 投影到同一 token 维度拼进 $z_t$。
- **本体状态预测**:显式预测未来本体配置(而不只是视觉),对接触密集的可形变物体操作(需要精确知道臂位和夹爪开合)至关重要。
- **稀疏记忆 + 短期历史**:一致性要求 WM 理解"什么在变、什么不变",在腕部视角频繁进出 FOV、物体被夹爪遮挡时尤其关键。
- **latent dynamics model**:采用高效的 2D transformer,$L$ 个 dynamics block 由 spatial attention + causal temporal attention 组成,以 latent tokens、action tokens、flow timestep embedding 为条件自回归生成 $h$ 步 chunk;每块用 RMSNorm、RoPE、QKNorm、SwiGLU。

### 2.3 训练目标:flow-matching + diffusion forcing

令 $x_t^1:=z_{t+1:t+h+1}$ 为 ground-truth 未来 $h$ 帧 latent,$x_0\sim\mathcal{N}(0,I)$ 为高斯噪声。定义插值 $x_t^\tau=\tau x_t^1+(1-\tau)x_t^0$,$\tau\in[0,1]$。训练 $f_\phi$ 去预测"速度" $x_t^1-x_t^0$,最小化:

$$
\mathcal{L}^{\text{WM}}(\phi)=\mathbb{E}_{x_t^0,x_t^1,\tau}\left[\left\|(x_t^1-x_t^0)-f_\phi(\mathbf{z}_t^{\text{mem}},\mathbf{z}_t^{\text{hist}},\mathbf{a}_t,x_t^\tau,\tau)\right\|_2^2\right]
$$

为改善长 horizon 一致性,采用 **diffusion forcing**(对未来各时间步独立采样噪声等级),并用 **SPRINT block** 激进丢弃 patch token 提升效率。

> 用大白话说:flow-matching 就是学"从一团纯噪声,沿直线走向真实的未来帧,该往哪个方向走多快"。diffusion forcing 让不同未来时刻的帧带着不同程度的噪声一起训练,这样模型在真正一步步往后生成很长一段时,不会"越滚越离谱"。

### 2.4 加速推理

延迟 = 前向次数 × 迭代去噪次数,两者都要压:

- **(a) 前向成本**:对 memory / history token 做 KV caching,跨去噪步复用(因为它们噪声等级恒定 $k=1$、不随去噪变化)——最高省 30% 时间(Table 4)。
- **(b) 去噪成本**:用 **cosine / power noise schedule**(而非 linear),在低噪声区分配更多预算以生成细节,保真更高(Table 5);
- **rectified flow 后训练(WEAVER-ReFlow)**:先用去噪过程生成高质量 latent 轨迹作为 target,再做二次蒸馏,把生成压到几次前向即可完成,达到测试时规划所需的速度。

### 2.5 从 latent 直接估值:reward + critic

- **Reward head $R$**:用 AdaPool 聚合 latent tokens + MLP,MSE 蒸馏一个 off-the-shelf reward model(RoboMeter)的分数,**无需解码成图像再喂给外部 VLM judge**——这是效率关键。
- **Critic $V$**:估计超出想象 horizon 之外的价值,支持截断 rollout。以 bootstrapped $\lambda$-return 为目标:

$$
\mathbf{v}_t^\lambda=R(z_t,\ell)+\gamma\Big((1-\lambda)V(z_{t+1},\ell)+\lambda\mathbf{v}_{t+1}^\lambda\Big)
$$

MSE 训练 $\mathcal{L}^{\text{critic}}(V)=\|V(z_t,\ell)-\mathbf{v}_t^\lambda\|_2^2$。

### 2.6 三大下游应用

- **策略评估**:把真机 rollout 的动作轨迹 open-loop 喂进 WM,沿途记录预测 reward。任务常需 40+ 次迭代评估,凸显一致性与效率的重要性。
- **策略改进**:从策略采 $h$ 步 chunk,前向模拟 $K$ 次共 $H=Kh$ 步,采一批 $B$ 个 rollout,计算 Monte-Carlo 的 $H$ 步 advantage $\hat{A}_t^b=\sum_{\ell=1}^{H}\gamma^{\ell-1}R(\hat{z}^b_{t+\ell},\ell)+\gamma^H V(\hat{z}^b_{t+H},\ell)-V(z_t,\ell)$。若最高分 rollout 的 advantage 超过小正阈值 $\hat{A}_t^{b^*}>\epsilon_{\text{adv}}$,则将其蒸馏回基座策略——**基于 advantage 的过滤**避免在"所有采样计划都比现状更差"的状态上更新策略。
- **测试时规划**:single-chunk best-of-$N$。给定当前观测与指令,采 $B$ 个候选 chunk,用 WM 想象各自结果,用 reward+critic 头估 advantage,执行最优者。$B=4$、想象 horizon $h=12$。

## 三、实验结果

**平台**:基座策略 $\pi_{0.5}$(DROID 上训练的 VLA);单臂 Franka Emika Panda;两侧 Zed 2i 外部相机 + 腕部 Zed Mini;$\pi_{0.5}$ 与 WEAVER 只用右相机 + 腕相机。WEAVER 共 928M 参数,DROID 上预训练 1M 步(batch 32,LR $1e^{-4}$,$4\times$H100 训 10 天),再在 5 个任务各 50 条(共 250 条)真机数据上微调 16k 步。任务:Stack Bowls、PnP Bag、PnP Marker、PnP Towel、Pour Beans。

### 3.1 世界模型本身(vs Ctrl-World,1.5B 参数)

WEAVER 在保真-预算的 Pareto 前沿上全面压制 Ctrl-World(Table 1,H100 上生成 10s 片段的耗时):

| 数据集 | 方法 | NFE | 外部 FID $\downarrow$ | 外部 FVD $\downarrow$ | 腕部 FID $\downarrow$ | 腕部 FVD $\downarrow$ | 时间(s) $\downarrow$ |
|---|---|---|---|---|---|---|---|
| DROID(val) | Ctrl-World | 50 | 22.44 | 55.05 | 25.32 | 91.77 | 42.33 |
| DROID(val) | WEAVER | 16 | 10.20 | 27.83 | 21.50 | 90.72 | **4.78** |
| DROID(val) | WEAVER | 50 | **9.51** | **26.54** | **16.75** | **66.89** | 14.25 |
| OOD 任务 | Ctrl-World | 50 | 31.44 | 91.48 | 33.47 | 145.86 | 42.33 |
| OOD 任务 | WEAVER | 50 | **23.48** | **87.03** | **27.37** | 145.04 | 14.25 |

- **速度**:WEAVER 在 NFE=8 时仅 2.53s 就超过 Ctrl-World NFE=50(42.33s)的质量,推理最高快 $16\times$(Fig 5、Table 3)。
- **长 horizon**:150 步(10s)rollout 下,WEAVER 的 FID 随 horizon 增长始终显著低于 Ctrl-World(Fig 3)。
- **noise schedule**:cosine / power 优于 linear / sigmoid(Table 5);KV cache 省最多 30%(Table 4);WEAVER-ReFlow 在 NFE=4 就逼近 WEAVER-FT NFE=16(Table 6)。

### 3.2 latent reward 预测精度(Table 8,OOD 任务)

| 方法 | RMSE $\downarrow$ | Spearman $\uparrow$ | Pearson $\uparrow$ | MMRV $\downarrow$ |
|---|---|---|---|---|
| Ctrl-World | 0.410 | 0.523 | 0.552 | 0.215 |
| WEAVER | 0.359 | 0.594 | 0.563 | 0.155 |
| WEAVER-FT | **0.188** | **0.870** | **0.863** | **0.035** |

### 3.3 三大下游任务(20 trials/任务)

**策略评估**:预训练 WM 倾向低估策略性能;微调后 WEAVER-FT 与真实成功率的 Pearson 相关性达 $\rho=0.870$、MMRV 降到 0.035,对 PnP Towel、Pour Beans 等困难任务的结果预测明显更准。

**策略改进**(成功率,$\pi_{0.5}$ 基座 vs 各微调数据源):

| 任务 | $\pi_{0.5}$ | Real(1k) | Syn(1k) | Mixed(2k) |
|---|---|---|---|---|
| PnP Marker | 0.30 | 0.60 | 0.50 | 0.70 |
| Pour Beans | 0.30 | 0.60 | 0.55 | 0.65 |
| PnP Towel | 0.40 | 0.75 | 0.75 | 0.90 |
| Stack Bowls | 0.60 | 0.75 | 0.75 | 0.90 |
| PnP Bag | 0.60 | 0.85 | 0.80 | 0.95 |
| **平均** | **0.44** | **0.71** | **0.67** | **0.82** |

纯合成数据(Syn)与真机数据(Real)只差约 4% 平均性能,说明合成数据质量高到可媲美昂贵真机采集;real+syn 混合再比纯 real 高 11%,平均 0.44 → 0.82 即**提升 38%,且不需任何真机交互**。Pour Beans 上 syn 数据从 1k 扩到 5k,性能持续提升并最终超过纯 real 微调。

**测试时规划**(best-of-$N=4$ steering):平均成功率 $\pi_{0.5}$ 0.44 → 0.58,即 **+14%**(最大单任务 +20%);想象计算是主要瓶颈,但 WEAVER 在 A6000 Ada GPU 上 horizon=15/batch=4 的 dynamics 预测仅 1.4547s vs Ctrl-World 29.4244s,快 $20.2\times$,使基于 WM 的在线规划变得可行。

## 四、局限性

作者在 A5 系统地列出:

1. **部分可观测**:纯视觉只能看到底层物理状态的一部分,接触力、抓取稳定性、被遮挡几何都缺失;腕部视角连续变化、杂乱场景下物体出 FOV 时尤甚——可能需要触觉/力矩/深度传感。
2. **复杂可形变与动态交互**:毛巾、袋子、颗粒(倒豆子)等高维、依赖历史的动力学难从有限机器人数据学到,小误差随时间累积导致定性错误 rollout。
3. **测试时规划 horizon 受限**:延迟仍把在线规划限制在单个 action chunk,无法对延迟后果或多阶段恢复行为做长时前瞻。
4. **数据覆盖与本体多样性**:主要在 DROID 上预训练,绑定特定本体/采集配置,倒豆子等动态在预训练数据中欠代表。
5. **noisy reward 监督**:reward/critic 头蒸馏自 off-the-shelf 的 RoboMeter,对细微失败模式的监督含噪或不完整,可能同时影响评估与改进。

## 五、评价与展望(学术视角)

**优点**:

- **"缝合"做得系统而克制**:本文最大的价值不是单一新组件,而是精准诊断出机器人操作 WM 的三诉求瓶颈,并从各社区各取其长——video generation 的 diffusion forcing[7]/flow matching[27]/预训练编码器[35]、latent WM(Dreamer-v4[16])的 reward-critic 头、JEPA[3]的 latent 预测目标、Ctrl-World[12]的多视角与记忆——最终在 Pareto 前沿上同时改善保真、一致、效率。这种"把效率当一等公民"的取向,直接解锁了此前被算力挡住的策略改进与测试时规划。
- **端到端真机闭环验证**:三大下游应用都在真实 Franka 上跑通,$\rho=0.870$ 的评估相关性与 38% 的策略改进是硬指标;尤其"纯合成数据接近真机数据、混合再涨 11%"是对"WM 合成数据能否真正改进策略"这一开放问题的有力正面证据。
- **latent 空间 reward/critic 头**避免解码图像 + 外部 VLM judge,是效率能达标的核心工程决策,也让 best-of-$N$ 规划在 GPU 上真正可跑。

**缺点与开放问题**:

- **基线单一**:WM 层面主要对比 Ctrl-World,虽是最相关的多视角操作 WM,但缺少与 video-generation 类 WM、V-JEPA2[3]、DreamGen 类方法在同一 setup 下的横向对比,难判断各设计选择的独立贡献(消融偏工程细节而非组件级 ablation)。
- **改进幅度的口径**:38% 是相对基座 $\pi_{0.5}$(0.44)的**相对**提升(0.44 → 0.82 约 +86% 相对、+38 个百分点),摘要"improvement of 38%"指绝对百分点,阅读时需留意口径;测试时规划的 +14% 亦为绝对百分点且当基座越弱增益越大。
- **可形变/颗粒物体仍是真正短板**:Pour Beans 即使微调后评估相关性仍最差,说明当前架构对 granular dynamics 的物理先验不足——这与作者自陈一致,也是与"神经-物理混合模拟器"路线竞争的空间。
- **规划 horizon 短**:single-chunk best-of-$N$ 只是浅层测试时搜索,尚未触及需要长时前瞻/多阶段恢复的规划,与 MPC / tree-search 类方法相比仍属初级。

**可能的改进方向**:引入触觉/深度等多模态以缓解部分可观测;更强的物理先验或 hybrid neural-physics 处理颗粒/可形变;更可靠的大规模 reward model(含校准不确定性)替代 RoboMeter;把 rectified-flow 蒸馏进一步压缩以支持更长 horizon 的在线规划;以及跨本体/仿真/人类视频扩大预训练覆盖以提升 OOD 鲁棒性。

## 参考

1. Guo et al., *Ctrl-World: A Controllable Generative World Model for Robot Manipulation*, ICLR 2026 — 最直接的对比对象与多视角/记忆架构来源。
2. Hafner, Yan, Lillicrap, *Training Agents Inside of Scalable World Models*(Dreamer-v4),2025 — reward/value 头与 latent WM 思路来源。
3. Assran et al., *V-JEPA 2: Self-supervised Video Models Enable Understanding, Prediction and Planning*, 2025 — latent 预测目标(而非像素重建)的来源。
4. Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-sequence Diffusion*, NeurIPS 2024 — 长时一致性的关键训练技巧。
5. Physical Intelligence et al., *$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization*, 2025 — 被评估/改进的基座策略。
