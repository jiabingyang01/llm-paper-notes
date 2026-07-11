# PersistWorld：持久化机器人世界模型——通过强化学习稳定多步 Rollout

> **论文**：*Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning*
>
> **作者**：Jai Bardhan, Patrik Drozdik, Josef Sivic, Vladimir Petrik
>
> **机构**：Czech Institute of Informatics, Robotics and Cybernetics, Czech Technical University in Prague（捷克理工大学 CIIRC）
>
> **发布时间**：2026 年 03 月（arXiv 2603.25685）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.25685) | [PDF](https://arxiv.org/pdf/2603.25685)
>
> **分类标签**：`world-model` `reinforcement-learning` `diffusion-post-training` `autoregressive-rollout` `robot-manipulation`

---

## 一句话总结

针对动作条件视频世界模型在自回归推理下"误差滚雪球"的暴露偏差问题，把 DiffusionNFT 的对比式扩散 RL 目标迁移到 Ctrl-World 的 $x_0$-prediction 参数化上，让模型在**自己生成的 rollout**（而非 ground-truth 历史）上做后训练，用多视角感知奖励（LPIPS/SSIM/PSNR）打分、组内归一化，在 DROID 上把 14 步（约 11 s）rollout 的外视角 LPIPS 降低 14%、腕视角 SSIM 提升 9.1%，paired 对比赢 98%、盲测人类偏好 80%。

## 一、问题与动机

动作条件视频扩散世界模型（world model, WM）能根据机器人动作序列生成未来多视角画面，被寄望成为可扩展的"虚拟环境"来评测和改进 VLA 策略。但要真正当模拟器用，需要生成**长时程**（数秒）的连贯视频——即自回归 rollout，每一步把上一步生成的画面重新编码后喂回历史缓冲区。

问题恰恰出在这里，即经典的**暴露偏差（exposure bias）/ closed-loop gap**：

- **训练** 用 teacher forcing：给定长度 $H+L$ 的真实片段，模型在**干净的 ground-truth 历史** $H$ 帧上学习去噪预测后 $L$ 帧。
- **测试** 时没有 ground-truth 历史，每个生成片段被编码后追加进历史缓冲区，再作为下一步的条件。模型从未见过"带瑕疵的历史"，于是任一步的微小空间/时间误差在下一步被放大，形成误差复合循环。
- 后果：几秒内被操作物体失去结构（论文 Fig 1 里青色碗溶解成一团模糊），机械臂偏离指令轨迹，整个场景失真。

作者强调这**不是数据不足的问题**：再多 teacher-forced 数据或更长训练都无法让模型对自己的不完美历史鲁棒，因为"带瑕疵的历史"按构造根本不在训练分布里。真正需要的是一个**直接在模型自身自回归输出上计算的训练信号**——奖励连贯的闭环生成、惩罚复合漂移。

RL 是天然框架，但把 RL 用到扩散模型上有两个障碍：(1) 标准策略梯度需要扩散模型给不出的似然，且穿透整条去噪链反传代价极高；(2) 近期 DiffusionNFT [42] 给出的优雅绕开方案（生成多个候选、用奖励打分、把相对质量直接编码进去噪损失、完全不反传去噪链）是为**单图生成**设计的——而自回归视频里没有固定 prompt，每步都在演化一个共享状态，需要新机制来产生可比较的候选。

## 二、核心方法

方法建立在预训练的 **Ctrl-World**（一个用 EDM $x_0$-prediction、SVD 骨干的动作条件多视角 WM）之上，每个自回归步接收三路输入：历史帧 latent、历史末端执行器（EEF）位姿、未来 EEF 位姿目标 $e_{t+1:t+L}\in\mathbb{R}^{L\times 7}$，同时生成三个相机视角（两外视角 + 一腕视角）的 $L=5$ 未来帧。

### 2.1 把对比式去噪从 velocity 迁到 $x_0$-prediction

DiffusionNFT 原本为 velocity-prediction 的 flow-matching 模型（如 SD3）推导。作者证明该框架能**精确地** 迁移到 Ctrl-World 的 $x_0$-prediction：因为从网络输出到干净数据预测的映射是仿射的，对比目标的构造与策略提升保证原样成立，**无需任何 $\sigma$-相关的修正项**。

设 $\hat{x}_{0,\theta}$ 为当前模型对干净 latent $x_0$ 的预测，$\hat{x}_0^{\text{old}}$ 为冻结的 EMA 参考模型（reference policy）的预测。对一个归一化奖励权重 $r\in[0,1]$ 和混合系数 $\beta$，构造隐式的正/负分支干净数据预测：

$$\hat{x}_0^{+} = (1-\beta)\,\hat{x}_0^{\text{old}} + \beta\,\hat{x}_{0,\theta}, \qquad \hat{x}_0^{-} = (1+\beta)\,\hat{x}_0^{\text{old}} - \beta\,\hat{x}_{0,\theta}$$

再最小化奖励加权的去噪损失：

$$\mathcal{L}(\theta) = \mathbb{E}\Big[\, r\,\|\hat{x}_0^{+} - x_0\|_2^2 + (1-r)\,\|\hat{x}_0^{-} - x_0\|_2^2 \,\Big]$$

**用大白话说**：$\hat{x}_{0,\theta} - \hat{x}_0^{\text{old}}$ 是"当前模型相对参考模型漂移的方向"。正分支 $\hat{x}_0^{+}$ 沿这个方向**外推**（把改动放大 $\beta$ 倍），负分支 $\hat{x}_0^{-}$ 则**反着走**（构造一个反事实预测）。损失用奖励 $r$ 在两者间插值：对高奖励样本（$r\approx1$）就去拟合 $\hat{x}_0^{+}$，即**强化** 模型往这个方向的漂移；对低奖励样本（$r\approx0$）就去拟合 $\hat{x}_0^{-}$，即**把模型从自己的这类输出上推开**。$\beta$ 越大信号越强但越可能训练失稳，$\beta\to0$ 退化为对参考模型的普通监督去噪。关键是：整个式子只用到干净的生成样本和参考预测，**完全不反传去噪链**，因此兼容任何黑盒采样器。

附录 A 进一步给出理论保证：在无限数据/容量下，$\mathcal{L}(\theta)$ 的唯一最小值点是把参考后验均值 $\mu^{\text{old}}$ 沿"奖励对齐改进方向"移动恰好 $2/\beta$ 步的预测器：

$$\hat{x}_{0,\theta^\star} = \mu^{\text{old}} + \frac{2}{\beta}\,\Delta_{x_0}, \qquad \Delta_{x_0} = \alpha\,(\mu^{+} - \mu^{\text{old}})$$

其中 $\Delta_{x_0}$ 由低奖励均值 $\mu^{-}$ 指向高奖励均值 $\mu^{+}$。作者还说明：velocity-prediction 与 $x_0$-prediction 之间只差一个与 $\theta$ 无关的逐噪声级仿射变换，故改进方向和该定理直接迁移，**理论保证是参数化无关的**。

### 2.2 把 group-relative 训练适配到自回归视频

DiffusionNFT 假设存在"共享条件 + 多个独立候选"的分组结构（单图里就是同一 text prompt 采 $K$ 张图）。自回归视频没有固定 prompt——每步都在改写共享历史缓冲区。作者的关键观察是：**任一 rollout 步之前的历史缓冲区状态，恰好扮演了 prompt 的角色**。冻结这个缓冲区、从它独立采 $K$ 个候选续段，就得到了共享上下文 + 独立响应的分组结构。具体 4 阶段 rollout 协议（对应算法 1）：

- **$S_1$ 生成共享前缀**：从单帧 ground-truth 观测出发（历史缓冲区用其编码 latent 复制回填），自回归生成 $P$ 步，每步把自己的输出喂回历史。前缀长度 $P\sim\text{Unif}\{0,1,\dots,9\}$——随机采样让训练同时暴露到早期几乎干净的缓冲区和后期严重漂移的缓冲区。
- **$S_2$ 分支 $K$ 个候选续段**：从冻结前缀缓冲区独立采 $K=16$ 个候选，每个是长为 $F$ 个 chunk 的短自回归序列（每 chunk 生成三视角 $L$ 帧、追加到该候选**私有** 的缓冲区副本），各自走出不同随机轨迹。
- **$S_3$ 打分与排序**：对每个候选算视觉奖励 $R_t^{(k)}$（对比生成帧与留出的 ground-truth 帧，三视角综合），在 $K$ 候选内做组归一化得到相对 advantage，消除不同 rollout 位置绝对奖励尺度的影响。
- **$S_4$ 更新模型**：用组归一化奖励权重缩放正/负去噪损失，梯度下降更新。**只有 LoRA adapter（rank $r=64$, $\alpha=64$）和 action encoder 收梯度，骨干冻结**。

变长前缀一石二鸟：既保证模型在全 rollout 深度上都保持质量（不只短时程），又让更新暴露到多样的历史缓冲区腐化程度，防止过拟合到单一误差区间。

### 2.3 多视角视觉奖励

奖励需要足够 dense、直接反映感知质量而非代理统计量、跨三视角一致。对时刻 $t$ 的候选片段 $\hat{x}^{(v)}_{t+1:t+L}$，先在片段 $L$ 帧上做时间平均，再在 wrist/ext$_1$/ext$_2$ 三视角上等权平均，最后组合成单标量奖励：

$$R_t = -w_{\text{LPIPS}}\,\overline{\text{LPIPS}}_t + w_{\text{SSIM}}\,\overline{\text{SSIM}}_t + w_{\text{PSNR}}\,\overline{\text{PSNR}}_t$$

三个指标互补：LPIPS（深度特征感知相似度）抓像素级指标看不见的感知失真、SSIM 抓局部结构、PSNR 作全局像素漂移的粗指示器（灾难性场景漂移）。组归一化用 z-score 再线性缩放到 $[0,1]$：

$$A^{(k)} = \frac{R^{(k)} - \mu_R}{\sigma_R + \epsilon}, \qquad r^{(k)} = \frac{\text{clip}(A^{(k)}, -1, 1) + 1}{2}$$

**用大白话说**：早期 rollout 步的候选普遍比晚期步分高，如果直接用绝对奖励，训练信号会被"rollout 位置"这个混杂因素主导；组内 z-score 把它变成"在同一上下文下，这个候选比同伴好还是差"的相对排名，让梯度幅度不受该位置绝对画质水平影响。

### 2.4 实现要点

- 基座 = Ctrl-World；LoRA（$r=64,\alpha=64$）加在 UNet 骨干 + 微调 action encoder，其余（UNet 层、VAE 等）全冻结。
- 8000 步、学习率 $1\times10^{-4}$、Muon 优化器、batch size 64、group size $K=16$。每次更新还**只取 10 个最有信息量的样本**（按奖励排序取 top-5 + bottom-5）。
- 奖励权重 $w_{\text{LPIPS}}=w_{\text{SSIM}}=1$，$w_{\text{PSNR}}=\tfrac{1}{32}$（把 PSNR 拉到与 $\text{SSIM}\in[0,1]$ 可比的量纲）。
- 8 张 NVIDIA H200 训练 3 天。
- 推理用 Euler sampler 采 50 步、不用 CFG。

## 三、实验结果

**数据集/评测**：DROID（Franka Emika Panda，标准三相机：两外 + 一腕），用 Ctrl-World 的留出验证集。评测 14 个连续片段（$14\times L=70$ 帧，约 11 s @ 5 Hz）的自回归 rollout，从单帧观测起步。$*$ 为原论文数字、$\dagger$ 为作者复现。

**Table 1：14 步（约 11 s）自回归 rollout 视觉质量**

| 视角 | 模型 | SSIM↑ | PSNR↑ | LPIPS↓ |
|---|---|---|---|---|
| 外视角 | WPE$^*$ | 0.77 | 20.33 | 0.131 |
| 外视角 | IRASim$^*$ | 0.77 | 21.36 | 0.117 |
| 外视角 | Ctrl-World$^*$ | 0.83 | 23.56 | 0.091 |
| 外视角 | Ctrl-World$^\dagger$ | 0.84 | 23.02 | 0.081 |
| 外视角 | **Ours** | **0.86** | **24.42** | **0.070** |
| 腕视角 | Ctrl-World$^\dagger$ | 0.62 | 17.80 | 0.310 |
| 腕视角 | **Ours** | **0.67** | **19.39** | **0.277** |

相对 Ctrl-World 基线：外视角 PSNR +1.40 dB、LPIPS −14.0%；相对 WPE / IRASim：PSNR 分别 +4.09 / +3.06 dB，LPIPS 分别 −46.6% / −40.2%。**最显著的增益在腕视角**（SSIM +9.1%、PSNR +1.59 dB）——即捕捉细粒度物体接触与手眼协调细节的视角，也是对下游策略评测最关键的视角。

**Table 2：物体/机器人区域掩码指标**（用 RoboEngine 分割前景，评测任务相关的空间/控制一致性）

| 视角 | 模型 | Obj SSIM↑ | Obj PSNR↑ | Obj LPIPS↓ | Robot SSIM↑ | Robot PSNR↑ | Robot LPIPS↓ |
|---|---|---|---|---|---|---|---|
| 外视角 | Ctrl-World$^\dagger$ | 0.88 | 22.25 | 0.025 | 0.82 | 17.62 | 0.039 |
| 外视角 | **Ours** | **0.89** | **23.60** | **0.021** | **0.86** | **19.25** | **0.033** |
| 腕视角 | Ctrl-World$^\dagger$ | 0.73 | 18.52 | 0.088 | 0.83 | 25.50 | 0.027 |
| 腕视角 | **Ours** | **0.76** | **19.87** | **0.078** | **0.86** | **27.24** | **0.023** |

掩码后增益更集中：物体掩码外视角 LPIPS 降 16.3%（vs 全帧 14.0%），腕视角 SSIM +5.4%；机器人掩码 PSNR 外/腕分别 +1.63 / +1.74 dB。说明增益确实来自更准的交互建模，而非背景保真。

**paired 对比（Fig 4）**：1-to-1 逐样本对比，本方法在约**98%** 验证样本上优于基线（$p<10^{-6}$）。

**人类盲测偏好（Fig 6）**：8 名 CV/ML/机器人领域专家（硕士以上，部分博士），2AFC 界面对比本方法 vs 基线（附 ground-truth 参考）。本方法偏好率**80%**（174 胜 vs 43 负），Elo 884.8 vs 715.2，$p=3.5\times10^{-20}$，rater 间 $\kappa\approx0.4$（中等一致），95% CI $[72\%,100\%]$。

**Table 5（长时程演化）**：随 rollout 变长两模型都自然退化，但本方法始终保持更高 PSNR/SSIM、更慢的 LPIPS 漂移，有效延长了稳定预测时程。

**消融（附录 B，均在相同梯度更新数下评测）**：

| 消融因素 | 结论 |
|---|---|
| 学习率 | $1\times10^{-4}$（ours）胜 $3\times10^{-4}$；lr 是最关键超参，激进 lr 会破坏骨干 |
| 奖励 | 三指标组合是唯一六项指标都有竞争力的配置；单指标各自会牺牲其他维度 |
| 前缀采样 | 均匀随机 $P\sim\text{Unif}\{0,\dots,9\}$ 匹配或超过定长基线与 curriculum，更简单 |
| group size / best-of-$N$ | best-of-2 还能再涨（+0.27 dB 外 PSNR），但为稳定用了 best-of-5，group size 16 |
| 预测时程 | $H=1$（ours）稳胜 $H=3$；单步后训练给出更紧、更低方差的梯度 |
| KL 正则 | 改善六项中五项，防止微调策略漂离参考太远 |
| 参考策略 EMA | EMA 升到 0.5 的 schedule 略优，早期防失稳 |

**世界模型做策略评测（附录 C，Fig 8）**：3 任务（Put Banana in Box / Put Green Block in Bowl / Rotate Marker）× 3 策略（$\pi_0$ / $\pi_0$-FAST / GROOT N1.5），每任务每策略 5 次真机 + 11 次 WM rollout，比对 WM 内进度率与真机进度率。本方法 Pearson $r=0.822$、MMRV$=0.006$（$p=0.007$），优于基线的 $r=0.796$、MMRV$=0.053$——WM 内进度与真机趋势相关更高、策略排名违背更少。

## 四、局限性

- **计算开销**：group-relative 训练每次更新要采 $K=16$ 个独立候选，相比标准监督微调开销显著增加。
- **奖励只管视觉保真**：当前奖励是纯视觉指标（LPIPS/SSIM/PSNR），不显式施加物理或几何约束；作者把 physics-informed 约束和 geometry-aware 指标列为未来工作。
- **仅在 DROID / Ctrl-World / Franka 单一设置验证**：没有跨机体、跨骨干、跨数据集的泛化证据；基座 WM 换成非 $x_0$-prediction 参数化时的可迁移性仅有理论论证。
- **未闭环进策略优化**：论文只把 WM 当被动模拟器评测，把"放进策略优化 loop 加速训练鲁棒 agent"留作未来工作——而这正是长时程稳定 WM 最有价值的用途。
- **奖励依赖 ground-truth 未来帧**：$S_3$ 打分需要留出的真实未来帧，因此后训练本身仍绑定在有配对真机演示的数据上，无法对纯想象的 rollout 无监督地改进。

## 五、评价与展望

**优点**：

- **问题定位准且重要**。暴露偏差是自回归视频 WM 落地当模拟器的核心障碍，作者把它清晰刻画为"数据无法解决、必须换训练目标"的结构性问题，这个论断是站得住的——只靠 teacher forcing 确实永远看不到腐化历史。
- **方法迁移干净、理论自洽**。把 DiffusionNFT 从 velocity/flow-matching 迁到 EDM $x_0$-prediction，并证明保证参数化无关、无需 $\sigma$-修正项，附录 A 从最优性标签 Bayes 分解一路推到"移动 $2/\beta$ 步"的闭式最优预测器，是一条完整且优雅的推导。用"冻结历史缓冲区 = prompt"把 group-relative 结构搬进自回归视频，是本文最巧的一步。
- **评测扎实**。除了全帧指标，还用 RoboEngine 掩码把增益归因到物体/机器人区域（排除靠背景保真刷分），加上 98% paired 胜率、80% 人类盲测、Elo、WM-to-real 相关性四条互相印证，说服力比只报三行 SSIM 的同类工作强不少。消融覆盖 lr / 奖励 / 前缀 / group size / 时程 / KL / EMA 七个维度，诚实地承认 best-of-2 更好却选了更稳的 best-of-5。

**缺点与开放问题**：

- **与同期工作高度撞车**。作者自己引的 WorldCompass [36] 是同期几乎同款思路（同样把 DiffusionNFT 式对比 RL 用到相机位姿条件 WM、同样有 prefix rollout 策略），本文的差异化卖点（随机前缀而非定长 schedule、多视角奖励、$x_0$ 适配、机器人操作场景）偏工程增量；RLVR-World [38] 也已证明"RL 目标在 rollout 保真上优于 MLE"，只是用在 token 模型上。这个方向的原创性窗口正在快速收窄。
- **奖励需要 ground-truth 未来帧**，本质仍是"更聪明的监督"而非真正无监督/自监督的闭环自改进——这限制了它在无配对数据的想象 rollout 上的适用性，也与"WM 当可扩展虚拟环境"的宏大叙事有张力。一个自然的改进方向是引入无参考的自一致性/物理一致性奖励（如深度/光流一致、接触先验），让后训练摆脱对配对真机帧的依赖。
- **$\beta$、reward 权重、group 采样等超参敏感**，消融显示 lr 稍高就破坏骨干，暗示这套后训练的稳定裕度不宽；缺少对失败模式（如奖励 hacking：模型学会生成"看着像但物理错"的高 LPIPS 分帧）的分析。
- **未验证下游价值**。全文最有说服力的应用（放进策略优化 loop）没做，WM-to-real 相关性实验规模也小（3 任务 3 策略）。若能证明"用 PersistWorld 做 rollout 训出的策略真机成功率更高"，价值会质变。

**展望**：本文是"扩散 RL 后训练 + 自回归世界模型稳定化"这条正在快速升温赛道上一篇执行完整、评测诚实的工作。技术上最值得延续的是两点：(1) 无参考物理/几何一致性奖励，摆脱对配对真机帧的依赖；(2) 把持久化 WM 真正闭环进 VLA 策略训练/评测，验证 rollout 保真度提升能否转化为下游策略性能。方法本身模块化、reward-agnostic，替换奖励即可扩展，这也是它相对固定目标微调的主要优势。

## 参考

1. Guo et al. **Ctrl-World: A Controllable Generative World Model for Robot Manipulation.** arXiv:2510.10125, 2025.（本文基座 WM）
2. Zheng et al. **DiffusionNFT: Online Diffusion Reinforcement with Forward Process.** arXiv:2509.16117, 2025.（本文对比式扩散 RL 目标的来源）
3. Wang et al. **WorldCompass: Reinforcement Learning for Long-Horizon World Models.** arXiv:2602.09022, 2026.（同期最相似工作，对比 RL 后训练相机位姿条件 WM）
4. Wu et al. **RLVR-World: Training World Models with Reinforcement Learning.** arXiv:2505.13934, 2025.（RL 可验证奖励改善 token 型 WM 转移保真）
5. Khazatsky et al. **DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.** arXiv:2403.12945, 2024.（评测数据集）
