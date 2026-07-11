# Dreamer 4：在可扩展世界模型内部训练智能体

> **论文**：*Training Agents Inside of Scalable World Models*
>
> **作者**：Danijar Hafner\*, Wilson Yan\*, Timothy Lillicrap（\* 共同一作）
>
> **机构**：Google DeepMind（San Francisco, USA）
>
> **发布时间**：2025 年 09 月（arXiv 2509.24527）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.24527) | [PDF](https://arxiv.org/pdf/2509.24527)
>
> **分类标签**：`world-model` `imagination-training` `offline-RL` `shortcut-forcing` `Minecraft`

---

## 一句话总结

Dreamer 4 用一个 **shortcut forcing** 目标 + 高效 block-causal transformer 训练出可在单张 GPU 上实时交互（21 FPS、9.6 秒上下文、2B 参数)的 Minecraft 世界模型,再纯粹在该模型内部"想象训练"策略,成为首个**仅靠 2.5K 小时离线数据、零环境交互**就在 Minecraft 中挖到钻石的智能体(比 OpenAI VPT 少用 100× 数据),世界模型生成质量 FVD 57 远超前作。

## 一、问题与动机

世界模型(world model)让智能体从视频里学习通用知识、在"想象"中通过强化学习或规划来选择动作。对机器人等场景,与部分训练好的策略在线交互往往不安全、不高效,因此**纯离线**地在世界模型内部优化策略极具价值。但已有工作存在两难:

- **Dreamer 3 一类**世界模型快而准,但架构容量小、只能拟合窄分布的简单仿真环境;
- **Genie 3 / Oasis / Lucid 一类**可控视频模型基于可扩展的 diffusion transformer,能生成多样场景,却**学不准物体交互与游戏机制**,且常需多张 GPU 才能实时仿真单个场景,难以支撑想象训练。

作者要的是一个**同时具备高容量、准确物理、单卡实时**三者的世界模型,并证明能在其内部训练出解决长程控制任务的智能体。测试床选 Minecraft 钻石挑战:人类平均需约 24,000 次鼠标键盘动作、20 分钟才能挖到钻石,是典型的长程稀疏奖励任务;而这里全程只用 VPT 的 2.5K 小时承包商录像做离线学习,不允许任何在线交互。

## 二、核心方法

Dreamer 4 = **causal tokenizer**(把视频压成连续表征)+ **interactive dynamics model**(在表征空间做动作条件预测),两者共用同一套高效 block-causal transformer。训练分三阶段(Algorithm 1):世界模型预训练 → 智能体微调(BC+奖励头)→ 想象训练(RL)。

### 2.1 背景:从 flow matching 到 shortcut / diffusion forcing

世界模型建立在 flow matching 之上:网络 $f_\theta$ 预测速度向量 $v = x_1 - x_0$(从纯噪声指向干净数据),信号水平 $\tau \in [0,1]$($\tau=0$ 为纯噪声,$\tau=1$ 为干净数据):

$$x_\tau = (1-\tau)x_0 + \tau x_1, \qquad \mathcal{L}(\theta) = \|f_\theta(x_\tau,\tau) - (x_1 - x_0)\|^2$$

推理时从噪声出发,以步长 $d = 1/K$ 迭代 $K$ 步。两条关键前作被融合进来:

- **Shortcut models**:除信号水平 $\tau$ 外还把**步长 $d$** 作为条件输入,用 bootstrap 损失把两个小步蒸馏成一大步,从而只需 2–4 步、少量前向就能生成高质量样本。
- **Diffusion forcing**:给序列里**每个时间步分配不同的信号水平**,使每帧既是去噪目标又是后续帧的历史上下文,推理时支持"给定干净/轻噪历史生成下一帧"的灵活模式。

*用大白话说*:shortcut 让"扩散"从几十步压到几步,diffusion forcing 让扩散能像自回归那样逐帧往下生成——把二者合起来,就得到又快又能长程 rollout 的视频世界模型。

### 2.2 Causal Tokenizer

编码器把当前帧的 patch tokens 与可学习的 latent tokens 一起处理,读出 latent 后经**低维投影 + tanh** 得到连续表征,解码器逐帧解回 patch。时间上用 causal attention 实现时序压缩,同时保持逐帧解码以支持交互推理。训练用 masked autoencoding:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{MSE}}(\theta) + 0.2\,\mathcal{L}_{\text{LPIPS}}(\theta)$$

对输入 patch 以 $p \sim U(0, 0.9)$ 随机丢弃(MAE 风格),显著提升动态模型生成视频的空间一致性。

### 2.3 Interactive Dynamics 与 shortcut forcing(核心创新)

动态模型在冻结 tokenizer 产出的表征上、以**交织序列**运行:动作 $a$、离散信号水平 $\tau$、步长 $d$、被腐蚀的表征 $\tilde z$ 一并输入,预测干净表征 $z_1$。关键设计:

**(1) x-prediction 而非 v-prediction。** 传统 shortcut 预测速度 $v$,适合整块一次性生成;但逐帧长视频 rollout 时 v-prediction 产生的高频输出会累积误差。作者改为直接预测**干净表征**(x-space),显著改善长程生成。shortcut forcing 损失(式 7,把网络输出转到 v-space 计算 bootstrap 再缩回 x-space)对最细步长 $d = d_{\min}$ 用 flow matching 项,对更大步长用两个半步平均出的 bootstrap 目标:

$$\mathcal{L}(\theta) = \begin{cases} \|\hat z_1 - z_1\|_2^2 & \text{if } d = d_{\min} \\ (1-\tau)^2\,\big\|(\hat z_1 - \tilde z)/(1-\tau) - \text{sg}(b_1 + b_2)/2\big\|_2^2 & \text{else} \end{cases}$$

**(2) ramp 损失权重。** 低信号水平($\tau$ 接近 0)的学习信号少(flow matching 退化为预测数据集均值),故用随 $\tau$ 线性上升的权重把模型容量聚焦到干净端:

$$w(\tau) = 0.9\,\tau + 0.1$$

*用大白话说*:越接近"看得清"的样本越值得学,越接近纯噪声的样本越没信息量,ramp 权重就是"越清晰越重视"。

推理时按时间自回归,每帧用 $K=4$ 步、步长 $d=1/4$;对过去输入轻微加噪到 $\tau_{\text{ctx}}=0.1$,让模型对自身生成的小瑕疵鲁棒。

### 2.4 想象训练(三阶段的后两阶)

**阶段 2:BC + 奖励模型。** 把 **agent tokens**(接收任务 embedding)插入 transformer,与图像表征、动作、register tokens 交织。关键因果约束:agent tokens 可以 attend 别人,但**没有别的模态能 attend 回 agent tokens**——否则会造成"世界模型未来预测被当前任务而非动作直接影响"的因果混淆。用 multi-token prediction(MTP, $L=8$)训练策略头与奖励头:

$$\mathcal{L}(\theta) = -\sum_{n=0}^{L}\ln p_\theta(a_{t+n}\mid h_t) - \sum_{n=0}^{L}\ln p_\theta(r_{t+n}\mid h_t)$$

**阶段 3:想象内 RL。** 冻结 transformer,只更新新加的 value 头和 policy 头(policy 头有一份冻结拷贝作行为先验)。从数据集上下文出发,用 transformer 自回归 unroll(表征采自 flow 头、动作采自 policy 头),再用奖励头、value 头标注 $r$、$v$。value 头用 TD-$\lambda$($\gamma=0.997$)学折扣回报:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T}\ln p_\theta(R_t^\lambda\mid s_t), \qquad R_t^\lambda = r_t + \gamma c_t\big((1-\lambda)v_t + \lambda R_{t+1}^\lambda\big)$$

policy 头用 **PMPO**:只看优势 $A_t = R_t^\lambda - v_t$ 的**符号**、忽略大小,对正/负优势集合分别做最大似然,天然抵消不同任务回报量纲差异,并加一个反向 KL 的行为先验约束:

$$\mathcal{L}(\theta) = \frac{1-\alpha}{|\mathcal{D}^-|}\sum_{i\in\mathcal{D}^-}\ln\pi_\theta(a_i\mid s_i) - \frac{\alpha}{|\mathcal{D}^+|}\sum_{i\in\mathcal{D}^+}\ln\pi_\theta(a_i\mid s_i) + \frac{\beta}{N}\sum_{i=1}^{N}\text{KL}\big[\pi_\theta(a_i\mid s_i)\,\|\,\pi_{\text{prior}}\big]$$

其中 $\alpha=0.5$(正负集合等权),$\beta=0.3$(弱先验)。三项损失都以 nats 度量,尺度天然可比。

*用大白话说*:先在数据里模仿动作(BC),再在世界模型的"白日梦"里试错,只记住"哪些动作带来正/负反馈"而不纠结数值大小,同时不许飘太远离原始合理行为。

### 2.5 高效 transformer

2D transformer(时间 × 空间),时间上 causal。用 pre-layer RMSNorm、RoPE、SwiGLU、QKNorm、attention logit soft capping。加速要点:①空间注意力与时间注意力**分层解耦**;②时间注意力只需**每 4 层用一次**(long context every 4 layers);③全部注意力用 GQA 减小 KV cache;④交替训练短/长 batch(长于上下文长度以防止过拟合起始帧,获得任意长度泛化)。

## 三、实验结果

设置:2B 参数(tokenizer 400M + dynamics 1.6B),256–1024 TPU-v5p,FSDP 分片;VPT 数据集 2541 小时承包商录像、360p、20 FPS。Minecraft 用 256 空间 token、192 帧上下文、256 batch 长度。所有智能体都只用同一份承包商数据、60 分钟每局、空背包随机世界、1000 局统计。

### 离线钻石挑战(agent 性能,Figure 3/4)

| 里程碑 | VPT(finetuned) | BC | VLA(Gemma 3) | Dreamer 4 |
|---|---|---|---|---|
| 木棍 sticks | 53% | 高 | 高 | ~99% |
| 石镐 stone pickaxe | ~0 | 高 | 高 | **>90%** |
| 铁镐 iron pickaxe | ~0 | 0.6% | 11% | **29%** |
| 钻石 diamond | 0 | ~0 | ~0 | **0.7%** |

- Dreamer 4 是**首个纯离线**挖到钻石的智能体;对最难里程碑,想象训练带来的相对提升最大。
- 直接用承包商动作的现代 BC 已强于 VPT(finetuned);VLA(用 Gemma 3 VLM 初始化)在铁镐上到 11%;Dreamer 4 把铁镐成功率**近乎翻三倍到 29%**。
- 消融(Figure 4)显示:用**世界模型表征做 BC(WM+BC)优于用 Gemma 3 或从头训**——说明视频预测隐式学到了对决策有用的世界理解;想象 RL 不仅提成功率,还让智能体更快到达里程碑(铁镐用时约 13 分钟,而 BC/VLA 约 29–31 分钟)。

### 世界模型生成质量(Table 1,单张 H100)

| 模型 | 参数 | 分辨率 | 上下文 | FPS | 交互任务成功 |
|---|---|---|---|---|---|
| MineWorld | 1.2B | 384×224 | 0.8s | 2 | — |
| Lucid-v1 | 1.1B | 640×360 | 1.0s | 44 | 0/16 |
| Oasis(small) | 500M | 640×360 | 1.6s | 20 | 0/16 |
| Oasis(large) | — | 360×360 | 1.6s | ~5 | 5/16 |
| **Dreamer 4** | 2B | 640×360 | **9.6s** | 21 | **14/16** |

Dreamer 4 上下文长度比前作长 6×,同时保持实时;人类玩家在其中反事实交互可完成 16 个任务中的 14 个(放置/破坏方块、打怪、乘船、进传送门等),而 Oasis 建墙几步后就"自动补全"幻觉出大结构、Lucid 交互直接被忽略。

### 设计消融级联(Table 2,FVD↓)

| 模型(逐步叠加) | 训练步 (s) | 推理 FPS↑ | FVD↓ |
|---|---|---|---|
| Diffusion Forcing Transformer(基线,K=64) | 9.8 | 0.8 | 306 |
| + 少采样步 (K=4) | 9.8 | 9.1 | 875 |
| + Shortcut model | 9.8 | 9.1 | 329 |
| + X-Prediction | 9.8 | 9.1 | 326 |
| + X-Loss | 9.8 | 9.1 | 151 |
| + Ramp weight | 9.8 | 9.1 | 102 |
| + Alternating batch lengths | 1.5 | 9.1 | 80 |
| + Long context every 4 layers | 0.6 | 18.9 | 70 |
| + GQA | 0.5 | 23.2 | 71 |
| + Time factorized long context | 0.4 | 30.1 | 91 |
| + Register tokens | 0.5 | 28.9 | 91 |
| + More spatial tokens ($N_z=128$) | 0.8 | 25.7 | 66 |
| + More spatial tokens ($N_z=256$) | 1.7 | 21.4 | **57** |

最终 FVD 57,远优于朴素 baseline 306,也优于"v-space 预测 + 损失"的完整架构(124)。Figure 8 显示 shortcut forcing 只用 **4 步**就逼近 diffusion forcing **64 步**的质量,即 **16× 更快**。

### 动作泛化(Figure 7)

- **少量动作即可 grounding**:只给 **10 小时**配对动作即达"全动作"训练的 53% PSNR / 75% SSIM;给 **100 小时**升到 85% PSNR / 100% SSIM——绝大部分知识来自无标注视频。
- **动作外推**:只在 Overworld 提供动作,模型对从未见过动作的 Nether/End 维度仍达全动作模型的 **76% PSNR / 80% SSIM**,预示未来可从多样无标注 web 视频学习模拟器。
- **数据效率**(Table 3):Dreamer 4 仅 2.5K 小时离线数据,相比 VPT(2.5K 离线 + 270K web + 194K 在线)少用 **100×** 数据。

## 四、局限性

- 世界模型**远非游戏的完整克隆**:记忆短(9.6 秒上下文)、背包物品预测不精确、有时随时间变化或不清晰;作者自称 Minecraft 仍是未来世界模型的理想 benchmark。
- 钻石成功率仅 **0.7%**,虽为"首次纯离线"但绝对成功率低,长程稀疏任务远未解决。
- 想象训练时**冻结 transformer**,只训策略/价值头;作者承认微调整个 transformer 有小幅额外收益但成本高,尚未纳入主设置。
- 仅在单一机器人数据集上做了初步定性验证(Figure 6),真实世界的量化 agent 结果缺失;主结论几乎全部来自 Minecraft 单一域。
- 无法与 Genie 3 直接对比(后者只支持相机 + 单一"interact"按钮,动作空间不匹配)。

## 五、评价与展望

**优点。** 这篇工作的核心贡献是**把"快"和"准"在世界模型里同时做到**:shortcut forcing(shortcut models × diffusion forcing 的融合,加上 x-prediction 与 ramp 权重)是干净且可复用的目标,消融表把每个改动对 FVD / FPS 的边际贡献量化得很清楚,可信度高。"纯离线挖到钻石、少用 100× 数据"是一个有分量的能力节点,且与机器人"在线交互不安全"的现实动机对得上。Table 2 的级联式消融和 Figure 7 的动作泛化实验,是全文最有说服力的部分——尤其"10–100 小时动作即可 ground 一个 embodiment,其余从无标注视频学"这一结论,对"从 web 视频规模化学世界模型"路线是强证据。

**与公开工作的关系。** 相对 Dreamer 3(Nature 2025,小容量、在线、抽象 crafting 动作),Dreamer 4 是"换骨"级重构:表征从离散 RSSM 换成连续 tokenizer + diffusion/flow 动态,动作空间下沉到低级鼠标键盘,规模到 2B。相对 DIAMOND、GameNGen 等"扩散世界模型",它的差异在于**为交互推理专门优化**(few-step + causal),并真正把模型当训练环境用而非仅做视频生成。相对 Oasis/Lucid/MineWorld,准确度(14/16 vs ≤5/16)和上下文长度(6×)是硬指标上的领先。PMPO + 想象 RL 延续了 Dreamer 谱系"model-based imagination"的思路,但用符号优势 + 反向 KL 先验做了鲁棒化。

**开放问题与可能改进。**(1)长程记忆:9.6 秒上下文对真正的长任务是硬瓶颈,把外部/循环记忆接入 transformer 值得探索;(2)背包/符号状态预测不准,说明纯像素表征对离散语义信息压缩不足,或可引入结构化辅助头;(3)想象 RL 冻结主干,策略改进被世界模型的想象保真度上限锁死——想象里的分布外动作若被世界模型"善意补全",会带来乐观偏差,如何检测并抑制 imagination 的 hallucination 是关键;(4)动作外推很吸引人,但只在同一游戏的三个维度间验证,跨具身/跨真实域的外推仍是空白;(5)作者提出的"预训练在通用互联网视频、接入语言理解、用少量在线纠错数据"是自然的下一步,也是把该 recipe 推向真实机器人的必经之路。总体上,这是一篇工程与方法都扎实、结论克制、可复现性叙述清楚的旗舰级世界模型论文,短板主要在绝对性能与单域评测上。

## 参考

1. D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap. *Mastering diverse control tasks through world models*(Dreamer 3). Nature, 2025.
2. B. Baker et al. *Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos*. NeurIPS 2022.
3. K. Frans, D. Hafner, S. Levine, P. Abbeel. *One Step Diffusion via Shortcut Models*. arXiv:2410.12557, 2024.
4. B. Chen, D. Martí Monsó, Y. Du, M. Simchowitz, R. Tedrake, V. Sitzmann. *Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion*. NeurIPS 2024.
5. E. Alonso et al. *Diffusion for World Modeling: Visual Details Matter in Atari*(DIAMOND). NeurIPS 2024.
