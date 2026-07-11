# AMPLIFY：用于从视频中学习机器人的无动作运动先验

> **论文**：*AMPLIFY: Actionless Motion Priors for Robot Learning from Videos*
>
> **作者**：Jeremy A. Collins*, Loránd Cheng*, Kunal Aneja, Albert Wilcox, Benjamin Joffe, Animesh Garg（*为共同一作）
>
> **机构**：Georgia Tech；Georgia Tech Research Institute
>
> **发布时间**：2025 年 06 月（arXiv 2506.14198）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.14198) | [PDF](https://arxiv.org/pdf/2506.14198)
>
> **分类标签**：`无动作视频先验` `latent motion tokens` `forward/inverse dynamics` `keypoint tracking` `data-efficient BC`

---

## 一句话总结

AMPLIFY 把策略学习拆成"预测运动"和"推断动作"两段：先用 FSQ 把稠密关键点轨迹压成离散的 **latent motion tokens**，在海量**无动作视频**上训一个自回归 forward dynamics（运动先验),再用少量带动作数据训一个不看目标、只做"轨迹跟随"的 inverse dynamics；轨迹预测精度 MSE 比 ATM 好 3.7×、像素精度好 2.5×,并首次在 LIBERO 上实现**零目标域动作数据**的零样本迁移（对最优 BC 基线平均 27× 提升,绝对成功率约 60.5%)。

## 一、问题与动机

- **动机来源**:训练通用机器人策略的主流做法 behavior cloning（BC)需要大量成对的(动作,观测)专家数据,而这类数据全网只有约几百小时;相比之下无动作视频约有十亿小时。视频里含有关于时序动态、行为、语义的丰富先验,却难以直接转成可执行策略。
- **已有路线的短板**:
  - 视频表征学习(reward/value 预训练、time-contrastive)只学静态编码器,不显式建模时序动态。
  - 像素空间的完整视频预测(如 UniPi/Seer)推理昂贵,被迫开环或部分去噪。
  - latent action 方法(Genie/LAPA/Moto)仍以像素重建为目标,容易抓到纹理等与策略无关的表观特征。
  - 关键点/光流方法(ATM、Track2Act)更贴近"运动"这一抽象,但:Track2Act 依赖不现实的目标图像假设且只支持单物体刚体;ATM 依赖训练期难以在推理复现的采样启发式,且两者都在**像素空间**预测轨迹,计算量大、泛化差(ATM 推理时只生成 32 条稀疏轨迹)。
- **核心命题**:用 **latent(而非像素空间)关键点运动**作为抽象,把"**什么(what)**定义任务"与"机器人**怎么(how)**执行"解耦——前者能在任意视频上学,后者能在任意交互数据上学,从而两段独立扩容。

## 二、核心方法

AMPLIFY 是三阶段解耦框架,策略写作 $\pi(o_t,q_t,g)=f_{\text{inv}}(o_t,q_t,f(o_t,g))$,其中 $o_t$ 为图像观测、$q_t$ 为本体状态、$g$ 为语言目标。

**数据分类(该框架的立论基础)**:把数据按模态分成三类——action-free videos $\langle o_t,g \rangle$(有目标无动作)、undirected interaction data $\langle o_t,q_t,a_t \rangle$(有动作但非目标导向,如探索/play)、expert demonstrations $\langle o_t,q_t,a_t,g \rangle$(全模态)。forward dynamics 用视频集 $\mathcal{V}$(前者+专家),inverse dynamics 用交互集 $\mathcal{R}$(后两者),BC 却只能吃专家演示。

**第 0 步·预处理**:用现成点跟踪器 CoTracker,在每帧初始化 $20\times20=400$ 点的**均匀网格**,向后跟踪 $T=16$ 帧。注意它对**每个窗口都重新初始化网格**(与 ATM 只初始化一次不同),保证任意帧全覆盖、无遮挡,能优雅处理动/摇镜头,代价是预处理算力 $T\times$。作者发现均匀网格比"按运动量选关键点"的启发式更鲁棒,后者会学到虚假相关。

**第一步·Motion Tokenization(图 2a)**:先由轨迹差分得单步速度 $u_t$,keypoint encoder $\mathcal{E}_\theta$(因果掩码 Transformer)编码后经 **FSQ**(Finite Scalar Quantization,VQ-VAE 的即插即用替代,隐式码本、单一重建损失,避免表征坍塌)量化成离散码 $z_t$。解码器 $\mathcal{D}_\theta$ 关键设计:**不直接回归坐标**,而是对每个点在以上一帧位置为中心的 $W\times W=15\times15$ 局部窗口上输出 $W^2=225$ 类的分类分布(即"下一步落到哪个速度格子"),用交叉熵训练:

$$\mathcal{L}_{AE}(\theta) = \mathrm{CE}\big(\mathcal{D}_\theta(h(\mathcal{E}_\theta(u_t))),\, \omega_t\big)$$

其中 $\omega_t=\Omega(u_t)$ 把真值速度映射为局部窗口内的类别,$h$ 为 FSQ 离散化函数。用大白话说:与其让模型硬回归一个连续坐标(回归 MSE 会往均值/零运动收敛、预测发糊),不如把"往哪动"当成分类题——局部窗口分类天然带"运动是局部的"归纳偏置,还能表达**多峰**的运动分布。多视角输入被拼进同一串码。

**第二步·Forward Dynamics = 无动作运动先验(图 2b)**:自回归 Transformer $f(o_t,g)$ 由当前图和任务描述预测未来运动的 token 串 $z_t$。图像用 ResNet-18 摊平成 $7\times7=49$ 个 vision token,语言用 T5 的 summary token;这些条件 token 与 SOS、运动 token 拼接,采用 block-causal 掩码(条件部分非因果、运动 token 因果)。损失只回传给预测器,tokenizer 冻结:

$$\mathcal{L}_{\text{forward}} = \mathrm{CE}\big(f(o_t,g),\, \mathrm{sg}(\mathcal{E}_\theta(u_t))\big)$$

用大白话说:给一张图和一句话,让 Transformer 像写句子一样自回归"续写"出未来 16 帧运动被压出来的那串离散码($\mathrm{sg}$ 是 stop-gradient,tokenizer 不动)。这一步**完全不需要动作标签**,是 AMPLIFY 能吃人类视频、跨本体的关键。

**第三步·Inverse Dynamics(图 2c)**:$f_{\text{inv}}(o_t,q_t,z_t)$ 把 latent 运动 token 解码成动作块 $a_{t:t+T}$。它是 Transformer decoder,一组可学 query 交叉注意到图像 token、本体状态投影、以及运动码,产出 $T$ 步动作块。**关键:它不看目标 $g$**,只是一个"通用轨迹跟随器",因此可在任意交互数据上训。动作头沿用 BAKU 的各向同性高斯先验,NLL 损失带时间折扣 $\gamma$:

$$\mathcal{L}_{\text{inv}} = -\sum_{\tau=t}^{t+T-1}\gamma^{\tau-t}\cdot\log p\big(a_\tau \mid \mu_{\tau-t},\,\sigma_{\tau-t}\big)$$

用大白话说:把预测运动翻成动作,越靠后的动作($\gamma^{\tau-t}$ 权重越小)越不信,因为远期运动预测越不准。实践中在 forward model 的**预测输出** $\hat z_t$ 上微调动作解码器(而非只用真值码),两个前置模块冻结;keypoint decoder 此阶段不用(策略只吃 latent 运动而非解码后的像素轨迹)。

**推理**:forward 逐步自回归出 $\hat z_t$,inverse 结合图像/本体解出动作块,并沿用 ACT 的 temporal ensembling(同一 $\gamma$)聚合历史预测。

三个模块参数量分别为 tokenizer 31M / forward 70M / inverse 57M,均在**单张 GPU**(RTX 6000 或 L40S)上训练。

## 三、实验结果

数据:BridgeData v2(6 万+真机 rollout,24 环境)、Something-Something v2(22 万+人类日常操作视频)、LIBERO(130 任务,取 6500 条演示当视频用)。

**(1) 轨迹预测精度(Table 2,$\Delta_{\text{AUC}}$ 为点跟踪常用面积指标,越高越好)**

| 方法 | LIBERO MSE↓ | LIBERO $\Delta_{\text{AUC}}$↑ | LIBERO Pixel Acc↑ | BridgeData v2 $\Delta_{\text{AUC}}$↑ | Sth-Sth v2 $\Delta_{\text{AUC}}$↑ |
|---|---|---|---|---|---|
| ATM | 0.022 | 0.767 | 0.250 | – | – |
| Track2Act（用目标图） | – | – | – | 0.770 | 0.700 |
| Seer（需全视频预测） | – | – | – | 0.914 | – |
| **AMPLIFY** | **0.006** | **0.913** | **0.629** | **0.968** | **0.725** |

MSE 相对 ATM 好 3.7×、像素精度好 2.5×;在 Bridge/Sth-Sth 上对用了目标图的 Track2Act 与需全视频预测的 Seer 有 4–6% 优势——注意 AMPLIFY 只用 latent 一致性损失、不做像素预测。

**(2) 标准 BC(LIBERO,Table 3,成功率)**

| 方法 | 视频预训练 | Long | 90 | Object | Spatial | Goal |
|---|---|---|---|---|---|---|
| BAKU | 否 | **0.86** | **0.90** | – | – | – |
| AMPLIFY（仅 inverse） | 否 | 0.76 | 0.83 | 0.64 | **0.83** | **0.92** |
| UniPi | 是 | 0.06 | – | 0.60 | 0.69 | 0.12 |
| ATM | 是 | 0.44 | 0.63 | 0.81 | 0.79 | 0.59 |
| AMPLIFY（完整） | 是 | 0.75 | 0.88 | **0.93** | 0.73 | **0.92** |

数据充足时 AMPLIFY 与 SOTA 的 BC(BAKU)持平,并明显优于其它视频预训练法(ATM/UniPi)。作者自陈:数据一多,标准 BC 就够用,预训练的价值主要在**低数据/新任务**上体现。

**(3) 少样本(Table 16)**:forward 用全部视频、inverse 只用 4%/10%/20% 演示。极端情况每任务仅 **2 条演示**时,AMPLIFY 平均比 ATM 提升 **1.94×**;AMPLIFY 与 ATM 都稳定超过"无视频预训练"变体,且 latent 运动优于直接喂像素轨迹。

**(4) 跨本体(人类视频→机器人,Table 4)**:forward 同时吃人+机视频,inverse 只吃机器人动作,BC 头统一换成 Diffusion Policy 头做公平比较。三个真机任务平均成功率 **0.58 vs Diffusion Policy 0.42**,随任务难度增加优势从 1.32× 扩到 1.5×。

**(5) 零样本泛化(Table 5,最亮点)**:forward 在 LIBERO 全子集观测上训,inverse 与 BC 基线**只在 LIBERO-90 动作**上训,再评四个从未见过动作的目标子集。BC 方法几乎全零(0.00–0.02),AMPLIFY 取得 Long 0.52 / Object 0.80 / Spatial 0.69 / Goal 0.41(平均约 **60.5%**),对最优 BC 基线平均 **27×**,是首个在 LIBERO 上不用任何目标域动作数据就报出非平凡成功率的工作。

**(6) 视频生成(Table 6)**:用 forward 预测的运动 token 去条件化 AVDC 视频生成,BridgeData v2 上 PSNR 15.93→16.40、SSIM 0.56→0.59(LPIPS 略升)。说明 latent 运动是可迁移到控制以外任务的通用世界模型接口。

**关键消融(Tables 11–13)**:因果注意力 0.919 > 逐帧 0.877;局部窗口分类 0.919 > MSE 回归 0.883;码本 2048、码长 16、隐维 768 最佳;forward 预测短horizon像素精度更高(4 步 0.757 vs 16 步 0.613),但 **inverse 反而 horizon 越长越好**(16 步成功率 0.75 vs 4 步 0.36),故折中取 16;ResNet-18 与 DINOv2/ViT 持平且更省算;冻结视觉编码器 = 微调;高斯/Diffusion/Flow-Matching 三种动作头几乎无差(取最简高斯)。视频缩放实验(Table 14,LIBERO-Object 仅 2 条动作)显示视频量 0→50 时成功率 0.00→0.55,验证无动作视频确实能补策略。

## 四、局限性

1. **2D 轨迹的动作歧义**:多个不同动作可能对应同一组 2D 轨迹,inverse dynamics 存在歧义;显式 3D 轨迹(SpatialTracker/DELTA 类)有望给出不依赖固定/已知相机视角的更鲁棒表征。
2. **只建模确定性环境动态**:随机环境下需额外机制把"智能体动作"从"外生噪声"里分离(dichotomy-of-control 类问题),纯状态到状态数据尤甚。
3. **inverse 未用在线探索数据**:既然能从离任务数据学,值得探索用探索策略在线采集的数据来训 inverse。
4. **backbone 规模有限**:预测骨干仍是 ResNet-18/小 Transformer,尚未扩到通用 VLM 或大视频模型,视频/语言泛化上限受限。
5. 均匀网格 + 逐窗口重初始化带来 $T\times$ 预处理开销;LIBERO-90 只跑单 seed、真机每任务 10 rollout,统计样本偏小。

## 五、评价与展望

**优点**:
- **解耦范式干净且有说服力**:forward/inverse 各吃互不相同的数据分布,把"任务理解"与"任务执行"彻底分家,是当前 video-pretraining-for-VLA 里少见能同时在轨迹精度、少样本、跨本体、零样本四条线上都拿到明显收益的工作,尤其零样本 27× 那条几乎是把 BC 的"无泛化机制"缺陷摆上台面。
- **在 latent 空间而非像素空间预测运动**这一取舍是核心:相比 ATM/Track2Act 的像素轨迹,离散码 + 自回归 Transformer 既省算又能建多峰运动,消融里局部窗口分类 vs MSE 的对比很干净地印证了"分类避免均值糊化"的直觉。
- 工程可复现性好(单卡可训、ResNet-18 足够、动作头无所谓),对算力不友好的团队友好。

**与其它公开工作的关系**:
- 与 **ATM**(any-point trajectory modeling)、**Track2Act** 同属关键点轨迹派,主要差异在 latent 化 + 无目标图假设 + 逐窗口重初始化;数据效率与零样本上明显反超。
- 与 **LAPA/Genie/Moto** 的 latent action 派互补:后者从像素重建里抽 latent action,AMPLIFY 则用显式点跟踪当监督信号,抽象更贴"运动"、更少表观干扰,但也因此吃跟踪器质量的亏。
- inverse dynamics 的"通用轨迹跟随器"思路与 model-based RL 的 forward/inverse 分解、以及 IDM 家族一脉相承。

**开放问题与可能改进方向**:
- 3D/深度感知轨迹是最直接的下一步,能同时缓解 2D 歧义和固定视角依赖;
- forward 骨干换成预训练视频扩散模型或 VLM,是否能把 Sth-Sth 这类真实人类视频的收益进一步放大;
- 随机动态下把外生噪声与可控运动解耦,是让该框架走出"确定性桌面操作"的关键;
- 运动 tokenizer 的码本是否可跨数据集/跨本体共享,决定了它能否成为真正通用的"运动词表";
- 依赖 CoTracker 作为伪真值,跟踪器在强遮挡、透明/反光物体上的失效会直接传导到策略,鲁棒性尚未系统评估。

## 参考

1. Wen et al., *Any-point Trajectory Modeling for Policy Learning (ATM)*, arXiv:2401.00025 — 最直接的像素空间轨迹预测基线。
2. Bharadhwaj et al., *Track2Act: Predicting Point Tracks from Internet Videos Enables Diverse Zero-Shot Robot Manipulation*, arXiv:2405.01527 — 用目标图预测点轨迹的对照工作。
3. Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*, arXiv:2309.15505 — AMPLIFY 运动 tokenizer 的量化基础。
4. Ye et al., *Latent Action Pretraining from Videos (LAPA)*, arXiv:2410.11758；Chen et al., *Moto: Latent Motion Token as the Bridging Language*, arXiv:2412.04445 — 从视频抽 latent action 的对照范式。
5. Haldar et al., *BAKU: An Efficient Transformer for Multi-Task Policy Learning*, arXiv:2406.07539 — inverse 动作头(各向同性高斯)与 BC 基线来源。
