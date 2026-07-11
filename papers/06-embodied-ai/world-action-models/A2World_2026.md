# A2World：从动作到世界建模的可迁移动力学先验学习

> **论文**：*Learning Transferable Dynamics Priors from Action to World Modeling*
>
> **作者**：Ze Huang, Jiahui Zhang, Hairuo Liu（三人共同一作）, Chenxi Zhang, Ran Cheng, Li Zhang（通讯作者）
>
> **机构**：复旦大学数据科学学院（School of Data Science, Fudan University）、上海创智学院（Shanghai Innovation Institute）、上海交通大学（Shanghai Jiao Tong University）、麦吉尔大学（McGill University）
>
> **发布时间**：2026 年 06 月（arXiv 2606.29501，提交于 2026-06-28）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.29501) | [PDF](https://arxiv.org/pdf/2606.29501)
>
> **分类标签**：`动作条件世界模型` `多视角扩散Transformer` `长时程自回归仿真` `视频-动作联合预测` `策略评估`

---

## 一句话总结

论文把"动作条件视频生成"本身当作一种可扩展的、可迁移的机器人动力学先验来学习:先在 2100k+ 条、覆盖 20+ 具身的机器人操作轨迹上预训练一个多视角 DiT 扩散世界模型 A2World,再用几乎不改变骨干结构的轻量适配,把同一份权重分别转化为长时程自回归仿真器 A2World-sim(真实成功率与仿真成功率 Pearson r = 0.965)和视频-动作联合预测策略 A2World-policy(LIBERO 平均成功率 98.6%,OOD 的 LIBERO-Plus Spatial 上纯动作到视频预训练变体达到 88.5%,基本追平专门为策略设计的联合预训练 88.6%)。

## 一、问题与动机

近年机器人学习方法越来越依赖视频生成模型,大致分两条路线发展:一条是把视频生成模型直接适配成视觉-语言-动作(VLA)策略,近来进一步演化为视频-动作联合预测;另一条是训练动作条件世界模型用于数据增广和策略评估,近期还开始结合 reward model 和强化学习做策略 post-training。

作者指出,当前工作普遍存在一个共同缺口:尚未充分把"机器人数据预训练"本身当作可迁移动力学先验的来源。具体表现为三类:(i)许多方法直接从通用视频生成 checkpoint 微调,并未在大规模、带真实动作标注的机器人数据上做专门预训练;(ii)另一些方法只在单一数据集上预训练,限制了模型能吸收的具身类型、相机配置和运动模式的多样性;(iii)即便是大规模的努力,也往往针对单一下游目标优化,而非显式为"仿真器中心"和"策略中心"两条流水线的可迁移复用而设计。

论文给出的核心直觉是:在操作任务中,动作提供了一种天然的因果监督信号——物体、场景、视角在不同数据集之间差异巨大,但底层的交互规律(接触、抓取、推动、释放如何引发状态变化)是共通的。用动作去预测视觉场景的演化,会迫使模型编码可控的、以动作为根据的动力学,超越单纯"表观级"的视频预测。基于这一直觉,论文提出直接在真实机器人动作标注上预训练一个动作条件视频世界模型 A2World,不依赖辅助的 latent-action 模型产生间接伪标签,从而鼓励模型在具身、相机设置、任务和运动模式上都获得更强的泛化能力。

## 二、核心方法

### 2.1 A2World:动作条件基座世界模型

基座模型建模的是给定当前帧和未来动作 chunk 条件下的未来观测分布:

$$\text{A2World} : p\left(o_{t+1:t+k} \mid o_t, a_{t+1:t+k}\right). \tag{1}$$

动作条件的注入方式是:动作 chunk $a$ 先经 MLP 编码为 $e = \mathrm{MLP}(a)$,再与扩散时间步嵌入相加、供每个 DiT block 使用:$\bar\tau(\sigma) = \tau(\sigma) + e$,随后 $\bar\tau(\sigma)$ 经 AdaLN 产生动态调制参数(scale、shift、gate)。基座模型中把通用条件特征置零($\mathbf{c}=\mathbf{0}$),模型完全依赖这种动作条件化的时间调制来感知动作。模型在 WAN2.1 tokenizer 产生的连续 latent 空间上工作,用标准 EDM 去噪得分匹配目标训练:

$$\mathcal{L}_{\text{A2World}}(\sigma) = \mathbb{E}_{\mathbf{z},\mathbf{n}}\left[\left\| \text{A2World}(\mathbf{z}+\mathbf{n}; \sigma, \mathbf{c}=\mathbf{0}, a) - \mathbf{z} \right\|_2^2 \right], \tag{2}$$

其中 $\mathbf{z}$ 是 VAE 编码后的干净 latent 视频,$\mathbf{n}\sim\mathcal{N}(0,\sigma^2\mathbf{I})$ 为噪声,$\sigma$ 为噪声水平。

**大白话说**:模型看一帧图 + 一段未来动作 chunk,通过去噪声预测出对应的未来帧序列;动作不是当作"文字条件"送进交叉注意力,而是直接注入到每个 transformer block 的归一化调制层里,逼着网络把"动作如何改变画面"刻进权重本身。

**多视角生成**:多数机器人操作场景天然带有多路相机(第一/第三视角)。论文把 $V$ 个视角在时间维度打包成一个统一序列 $\mathbf{z}_{\text{mv}}$,用可学习的视角嵌入 $\epsilon_{\text{view}}(v)\in\mathbb{R}^{d_e}$ 在 patch embedding 之前沿通道维拼接进 latent,显式提供相机身份:

$$\tilde{\mathbf{z}}_{\text{mv}}^{(v)} = \text{concat}\left(\mathbf{z}_{\text{mv}}^{(v)}, \epsilon_{\text{view}}(v)\right), \quad \tilde{\mathbf{z}}_{\text{mv}} \in \mathbb{R}^{B\times(C+d_e)\times(V\cdot T)\times H\times W}. \tag{3}$$

并在每个 DiT block 内插入跨视角注意力模块,让不同视角 token 互相 attend 以保持时空一致性:

$$\tilde{\mathbf{z}}_{\text{mv}}^{(w)} \leftarrow \tilde{\mathbf{z}}_{\text{mv}}^{(w)} + \text{CrossViewAttn}\left(\tilde{\mathbf{z}}_{\text{mv}}^{(w)}, \tilde{\mathbf{z}}_{\text{mv}}^{(u)}, u\neq w\right). \tag{4}$$

**大白话说**:与其为每台相机单独训一个模型,不如把多视角"揉"进同一个视频扩散序列,靠可学习的"视角身份牌"加跨视角注意力保证不同机位讲的是同一个故事。

**预训练数据**:论文统一各数据集的动作为 7 维双臂末端位姿 + 夹爪状态表示(单臂机器人缺失的手臂做 zero-pad),整合 AgiBot、DROID、OPEN-X、InternData-A1、InternData-M1-Triplex、RoboCoin、Galaxea 等公开数据集,预训练轨迹总量 2156k 条,覆盖 20+ 具身类型。由于不同数据集相机设置和视角约定差异很大,直接混合会造成冲突,论文采用 dataset-consistent batching——每个 mini-batch 只从单一数据集采样。

### 2.2 A2World-sim:适配为长时程自回归仿真器

为把预训练动力学先验复用到长时程仿真,论文把 A2World 微调为一个历史感知的自回归世界模型 A2World-sim:

$$\text{A2World-sim} : p\left(o_{t+1:t+k} \mid o_t, a_{t+1:t+k}, \mathcal{H}_{t-1}\right), \quad \mathcal{H}_{t-1} = o_{1:t-1}. \tag{5}$$

**姿态引导的历史采样(Algorithm 1)**:给定历史序列和已执行动作,论文并不用简单的滑动窗口,而是根据相对动作计算加权弧长(arc-length),沿弧长空间均匀采样一组紧凑帧,在固定历史预算下优先保留转折点等关键状态。

**历史注入**分两路:一是把采样后的历史 latent token 化为 $\mathbf{H}=\text{Tok}_{\text{hist}}(\mathcal{H}[\mathcal{S}])\in\mathbb{R}^{B\times N_h\times D}$,替代预训练时被置零的交叉注意力条件,由目标视频 latent token 吸收:

$$\mathbf{z} \leftarrow \mathbf{z} + \text{CrossAttn}(\mathbf{z}, \mathbf{H}); \tag{6-7}$$

二是把同一批历史 token 额外投影为 key/value 记忆,与当前 latent 的自注意力 K/V 拼接,构成全局历史记忆通路,使目标 token 能在自注意力内部直接与压缩后的历史状态交互。

**自回归生成**:给定初始观测和未来动作,模型以 chunk-wise 方式自回归滚动:每一步预测的未来帧被追加进历史缓冲区,复用为下一步的条件,从而把短时程预测器转化为长时程动作条件仿真器。为提升长时程稳定性,训练中采用 Self-forcing 风格策略——周期性地让模型以自己生成的帧(而非 ground-truth)作为条件。论文指出,由于给定初始帧和动作条件下未来轨迹已在很大程度上被底层动力学决定,不需要额外训练一个单独的 teacher 模型来做蒸馏,Self-forcing 直接暴露模型自身的 rollout 误差并训练其从中恢复。

**大白话说**:把"看一帧、猜未来"的一次性世界模型,改造成能一步步往前滚的仿真器——每一步只把"最有信息量"的历史帧(而非简单最近若干帧)喂给模型,并且训练时故意让模型看自己生成的、带误差的历史,逼它学会自纠错,否则长 rollout 会越滚越飘。

### 2.3 A2World-policy:适配为视频-动作联合预测策略

从同一份预训练权重出发,论文进一步把 A2World 转化为一个 MoE 式的视频-动作联合预测模型,把世界模型直接实例化为指令条件下的机器人策略:

$$\text{A2World-policy} : p\left(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l\right). \tag{8}$$

语言指令 $l$ 用预训练 T5 文本编码器编码,其 token 嵌入 $\mathbf{h}_l$ 作为每个 DiT block 的交叉注意力上下文。视频 latent $\mathbf{z}^v$ 和动作 latent $\mathbf{z}^a$ 各自独立加高斯噪声:

$$\mathbf{z}_{\sigma_v}^v = \mathbf{z}^v+\mathbf{n}^v,\ \mathbf{n}^v\sim\mathcal{N}(0,\sigma_v^2\mathbf{I}), \qquad \mathbf{z}_{\sigma_a}^a = \mathbf{z}^a+\mathbf{n}^a,\ \mathbf{n}^a\sim\mathcal{N}(0,\sigma_a^2\mathbf{I}). \tag{9}$$

在共享时间步的训练变体中,先采样一个统一的基础噪声水平 $\sigma_{\text{base}}$,再按模态缩放 $\sigma_v=\alpha_v\sigma_{\text{base}}$、$\sigma_a=\alpha_a\sigma_{\text{base}}$(补充材料给出 $\alpha_v=\sqrt6,\ \alpha_a=0.5$),在保留模态特异噪声尺度的同时改善视频-动作对齐。

**MoE 式视频-动作 block**:视频 token $\mathbf{z}^\ell_v$ 和动作 token $\mathbf{z}^\ell_a$ 在每个 block 内共享同一个自注意力模块(视为跨模态交互共享的"专家"),同时各自保留独立的轻量去噪分支(AdaLN + MLP)。视频 token 每层都被更新;动作 token 通过共享自注意力同时 attend 视频和动作 token,再各自过 modality-specific 的 AdaLN/MLP 分支,最终动作序列由最后一层动作 token 经线性头预测。训练目标为加权联合去噪:

$$(\hat{\mathbf{z}}^v,\hat{\mathbf{z}}^a) = \text{A2World-policy}(\mathbf{z}^v+\mathbf{n}^v,\mathbf{z}^a+\mathbf{n}^a;\sigma_v,\sigma_a,\mathbf{c}=\mathbf{h}_l), \tag{10}$$

$$\mathcal{L}_{\text{A2World-policy}}(\sigma_v,\sigma_a) = \mathbb{E}\left[\mathbf{w}(\sigma_v)\|\hat{\mathbf{z}}^v-\mathbf{z}^v\|_2^2 + \lambda_a\,\mathbf{w}(\sigma_a)\|\hat{\mathbf{z}}^a-\mathbf{z}^a\|_2^2\right]. \tag{11}$$

推理阶段采用模态级 classifier-free guidance,视频、动作可分别设置引导系数 $s_v,s_a$:

$$\hat{\mathbf{z}}_{\text{cfg}}^m = \hat{\mathbf{z}}_u^m + s_m(\hat{\mathbf{z}}_c^m-\hat{\mathbf{z}}_u^m),\quad m\in\{v,a\}, \tag{12}$$

当使用单一引导标量 $\gamma$ 时 $s_v=s_a=1+\gamma$。

**大白话说**:策略不再是"世界模型之外再外挂一个动作头",而是让动作 token 和视频 token 共用同一套 attention "专家"互相感知,却各留一条独立去噪通道——这样动作生成能直接复用世界模型学到的视觉动力学先验,无需再单独做一段纯动作预训练。

## 三、实验结果

**训练设置**:A2World 的 DiT 骨干从 Cosmos-Predict2-2B-Video2World checkpoint 初始化,加上多视角模块后总参数量 2.5B;给定初始帧和 20 步动作 chunk,预测未来 20 帧。预训练用 64×H200,单卡 batch 12、梯度累积 4、训练 2 epoch,fused Adam,lr = 1e-4,weight decay = 0.1。A2World-sim 微调用 8×H200,batch 24,历史长度 $T_h=20$。A2World-policy 从预训练 A2World 初始化,交叉注意力从 Cosmos-Predict2-2B checkpoint 初始化(因预训练时交叉注意力条件被置零),动作专属模块复制视频分支参数以稳定联合微调;用 32×H200,global batch 256,lr = 1e-4,LIBERO 和真机微调各 2 万步;OOD 评估则在 LIBERO 上再微调 2.4 万步后测试 LIBERO-Plus Spatial。

**真机平台**:自建双臂 Flexiv(Rizon 4S)机械臂 + Robotiq-2F-85 夹爪平台,仿照 Toyota Research Institute(TRI)的搭建方式,两臂对称摆放于 45°,通过 VR 遥操作采集数据;视觉为 1 台前视 Intel RealSense D435i + 2 台腕部 D405(480×640,30fps)。真机任务集含 5 项:insert RAM module、flip small box、toggle power switch、lift box high、put chain in the box,涉及接触密集、铰接物体、可形变物体等挑战场景。

**世界模型 rollout 质量(LIBERO / 真机,Table 2)**:

| 方法 | LIBERO PSNR↑ | LIBERO SSIM↑ | LIBERO tSSIM↑ | LIBERO EPE↓ | 真机 PSNR↑ | 真机 SSIM↑ | 真机 tSSIM↑ |
|---|---|---|---|---|---|---|---|
| Cosmos-Predict2 | 25.36 | .8792 | .7631 | .4009 | 24.99 | .8355 | .7009 |
| Ctrl-World | 23.60 | .8632 | .7445 | .6827 | 21.94 | .8422 | .7158 |
| Prophet | 26.12 | .8887 | .7789 | .3667 | 25.55 | .8454 | .7102 |
| A2World-sim-T-pre(文本条件对照) | 26.18 | .8892 | .7794 | .3533 | 24.64 | .8198 | .6411 |
| **A2World-sim** | **26.64** | **.8957** | **.7862** | **.3498** | **25.95** | **.8511** | .7139 |

A2World-sim 在两个数据源上都取得最好或接近最好的画质与动作忠实度(optical-flow 一致性 EPE/cos)指标,且明显优于同样用相同机器人数据预训练但把动作条件替换为 T5 文本条件的 T-pre 对照,说明动作条件优于文本条件的动力学建模能力。

**RoboNet 视频预测(Table 3,与自回归基线对比)**:

| 方法 | FVD↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| MaskViT | 211.7 | 20.4 | 67.1 | 17.0 |
| iVideoGPT | 197.9 | 23.8 | 60.8 | 14.7 |
| SAMPO | 175.3 | 25.3 | 84.7 | 12.3 |
| **A2World-sim** | **146.1** | 24.1 | 81.6 | **8.9** |

**OOD 仿真器泛化(LIBERO-Plus Spatial,Table 4)**:微调后在训练时未见的场景分布上评估,A2World-sim 相对 DreamDojo 基线在 PSNR(25.91 vs 23.79)、SSIM(.8719 vs .8571)、EPE(.1301 vs .2738)、cos(.2761 vs .2107)上均更优,仅 tSSIM 略低(.7401 vs .7778);补充材料的定性对比显示 DreamDojo 在未见背景场景下更容易漂移回训练域外观,而 A2World-sim 更好地保持了新场景与交互动力学。

**仿真器与真实世界一致性**:把策略在 A2World-sim 内闭环 rollout(约 64 次仿真 rollout)得到的成功率,与同一策略约 25 次真机 rollout 的成功率做相关性分析,结果为 Spearman $\rho=0.916$,Pearson $r=0.965$,$R^2=0.930$($N=8$,即 8 组"策略×任务"),线性拟合近似 $y\approx1.008x+0.007$,接近恒等映射,表明 A2World-sim 可以作为较为忠实的真实世界策略评估替身。

**LIBERO 策略成功率(Table 5)**:A2World-policy 作为直接策略在标准 4-suite 协议下取得 98.2 / 99.2 / 98.6 / 98.2(Spatial/Object/Goal/Long)、总体平均 **98.6%**,为该基准最优平均表现;对照的一批近期 VLA 基线(Diffusion Policy、4D-VLA、Dita、$\pi_0$、UniVLA、$\pi_{0.5}$、OpenVLA-OFT、CogVLA 及基于 Cosmos-Predict2 初始化的策略基线)平均成功率整体集中在 88.6%–98.5% 区间,说明 LIBERO 本身已接近饱和、方法间差距被压缩到百分点量级。

**OOD 策略迁移(LIBERO-Plus Spatial,Table 6)**:对比四种预训练/初始化方式微调后的平均成功率——C-init(直接从 Cosmos-Predict2 初始化)80.2%,T-pre(在同一机器人数据上做文本条件世界模型预训练)85.8%,**A-pre**(本文的动作到视频预训练)**88.5%**,P-pre(专门针对策略设计的文本-视频-动作联合预训练)88.6%。A-pre 明显优于 C-init 和 T-pre,且与专门为下游策略设计的 P-pre 几乎打平,说明动作到视频预训练本身已经捕获了策略迁移所需的大部分动力学先验,不需要额外设计策略专属的预训练目标。

**消融:预训练变体(Table 7,LIBERO 微调后总成功率)**:

| 设置 | 总成功率 |
|---|---|
| 文本条件 Cosmos 初始化(C-init) | 97.0 |
| 文本条件 A2World 预训练(T-pre) | 97.4 |
| 动作-视频 A2World 预训练(A-pre) | 98.6 |
| 策略专属联合预训练(P-pre) | **98.8** |
| 视频分支冻结、仅共享自注意力可训 | 86.2 |

冻结视频分支后成功率从 98.6% 骤降到 86.2%,说明动作生成能力高度依赖对视频分支的联合可训练性,迁移收益并非来自"冻结骨干 + 轻量适配头"式的纯迁移,而是仍需联合微调整个视频-动作骨干。

**消融:历史采样策略(Table 8,LIBERO rollout 质量)**:

| 方法 | PSNR↑ | SSIM↑ | tSSIM↑ | EPE↓ | cos↑ |
|---|---|---|---|---|---|
| 无历史 | 25.41 | .8806 | .7663 | .3969 | .5778 |
| 滑动窗口 | 25.63 | .8840 | .7699 | .3900 | .5853 |
| **姿态引导采样(本文)** | **26.64** | **.8957** | **.7862** | **.3498** | **.6045** |

姿态引导的弧长采样在同等历史 token 预算下持续优于滑动窗口和无历史两种基线,补充材料的定性结果显示滑动窗口容易在长 rollout 中导致物体位置漂移或"消失",姿态引导采样能更好保持物体持久性(object permanence)。

**视频-动作耦合**:训练过程中每个验证 checkpoint 上,视频一致性(基于 Farneb\"{a}ck 光流的动作诱导运动余弦相似度)和归一化动作质量分数呈现一致的正相关趋势,且完整联合训练达到的"帕累托前沿"高于冻结视频分支的变体,说明世界模型的视觉预测能力与动作生成能力在训练中相互促进。

**真机策略评估**:在自建 5 任务真机套件上,A2World-policy 由第三方数据采集操作员按标准化协议评分(task progress 与 final success),结果显示 A2World-policy 在全部 5 项任务上的 progress/success 均超过 $\pi_{0.5}$ 和 LingBot-VA 两个基线,在难度最高的长时程、接触密集任务(put chain in the box,涉及放入变形链条并合上盒盖两阶段)上领先幅度最大——基线常见的失败模式是只完成部分子任务(如放入链条但未合上盒盖,或完全无法放入),而 A2World-policy 更能可靠完成完整的多阶段序列。

## 四、局限性

论文正文没有专设"局限性"小节,以下局限主要从方法设计与实验协议中归纳:

1. **数据与算力门槛高**:预训练依赖 2156k 条带真实动作标注的机器人轨迹和 64×H200 GPU 规模的算力,相比依赖伪动作/latent-action 标签、可从无动作标注视频中学习的路线,数据获取和复现成本显著更高。
2. **"可迁移"仍以下游微调为前提**:A2World-sim 在 LIBERO/真机上都需要再微调(如 rollout 质量评测中提到的 30k 步微调)才能取得最佳指标,OOD 迁移(Table 4、Table 6)也建立在"先在 LIBERO 上微调、再评估 LIBERO-Plus Spatial"的协议之上,说明论文验证的是"强初始化下的高效微调",而非严格意义上的 zero-shot 世界模型迁移。
3. **消融显示迁移收益依赖联合微调整个骨干**:Table 7 中冻结视频分支、仅训练共享自注意力的变体成功率从 98.6% 降到 86.2%,说明动作能力的获得高度依赖对整个视频-动作骨干的联合可训练性,而非"冻结预训练权重 + 轻量适配头"式的低成本迁移。
4. **真机评测规模有限**:自建平台仅覆盖 5 个任务、单一双臂 Flexiv 具身,成功率/进度依赖人工定义的任务专属 milestone 打分,任务多样性和评测规模都相对有限,难以充分验证在更广谱具身和任务上的可迁移性。
5. **仿真器一致性验证样本量小**:图 8 中 Spearman/Pearson 相关性基于 $N=8$(策略×任务组合)估计,未报告置信区间,统计功效有限。
6. **动作表示的适用范围**:统一的 7 维双臂末端位姿 + 夹爪表示对全身/移动底盘等更高自由度或非机械臂具身的可迁移性未在文中验证。
7. **推理成本未量化**:长时程自回归 rollout(chunk-wise 生成 + Self-forcing 训练)相对单步策略的额外推理开销、以及与直接微调 Cosmos-Predict2 等更轻量基线相比的成本-收益权衡,论文未给出系统的 wall-clock 或算力对比。

## 五、评价与展望

**优点**:论文把此前分开发展的两条技术路线——"世界模型作为仿真器"和"视频生成模型作为 VLA 策略"——统一到同一份动作条件预训练权重之下,通过对交叉注意力/自注意力路径的极简改造分别得到评估用仿真器 A2World-sim 和可执行策略 A2World-policy,这种"一次预训练、两处复用"的设计兼具工程效率和研究价值。消融实验(Table 7、图 11)干净地论证了核心论点:纯动作到视频预训练(A-pre)在 OOD 策略迁移上(88.5%)几乎追平专门为策略设计的联合预训练(P-pre,88.6%),说明不必为每个下游目标单独设计预训练配方即可获得接近最优的迁移效果,这是比很多"世界模型仅作为辅助预训练目标"的工作更干净的对照实验。仿真器忠实度的验证方式(策略在仿真与真机上的成功率强相关,Pearson r = 0.965)也为"世界模型能否替代真机评测"这一开放问题提供了一个较为严谨的正面数据点。此外,跨数据集异构相机配置这一现实痛点,论文通过多视角交叉注意力 + dataset-consistent batching 做了针对性处理,是不少单一数据集预训练的世界模型工作未系统解决的细节。

**与其他公开工作的关系**:相关工作一节列出的一批同期"world-action model"工作(如 Cosmos Policy、DreamZero、LingBot-VA 等)大多采取"视频模型微调后直接产生动作相关预测"或"视频-动作 token 交织的自回归策略"路线;A2World 与之的关键区别在于先做纯粹的动作条件视频预训练、再分别迁移到仿真器和策略两个下游,论文认为这避免了单独的纯动作预训练阶段、只需轻量的策略专属适配。与 Ctrl-World(基于 DROID 的可控生成式世界模型)、Prophet(单视角、面向策略后训练的动作条件世界模型)、DreamDojo(基于大规模人类视频的世界模型)等公开工作相比,A2World 在多视角一致性和跨数据集异构相机处理上做了更系统的设计,并在同一批基准上都取得了持平或更优的画质与动作忠实度指标。

**开放问题与可能的改进方向**:(1)A-pre 与 P-pre 在 OOD 策略迁移上的差距仅 0.1 个百分点,这一结果虽然支持"动作到视频预训练已经足够"的论点,但也提出了一个问题——引入额外的纯世界模型预训练阶段(相对直接做 joint video-action 预训练)在等计算量条件下是否仍然合算,论文未做严格的等算力对比;(2)长时程自回归仿真依赖 Self-forcing 式训练抑制误差累积,但论文未报告 rollout 长度的可靠上限或典型失败模式,长时程(如超过论文展示的 200 帧规模)下的稳定性边界仍不清楚;(3)该范式对动作标注质量高度敏感,动作作为核心的因果监督信号意味着方法难以直接扩展到海量但缺乏精确动作标注的网络视频或人类操作视频,这是相对于"仅用视频预训练、再后验对齐动作"路线的一个先天扩展性限制,如何把这类无动作视频数据也纳入动作条件预训练框架是值得探索的方向;(4)真机验证目前局限于单一双臂平台和 5 个任务,后续工作若能在更多具身形态(如移动底盘、灵巧手)和更大规模的真机任务集上验证仿真器一致性结论,将显著增强其作为通用策略评估工具的说服力。

## 参考

- Agarwal et al. *Cosmos World Foundation Model Platform for Physical AI.* arXiv, 2025——A2World 的 DiT 骨干(Cosmos-Predict2-2B-Video2World)来源。
- Guo, Y., Shi, L.-X., Chen, J., Finn, C. *Ctrl-World: A Controllable Generative World Model for Robot Manipulation.* ICLR 2026——rollout 质量对比的核心基线之一,同为 DROID 上的可控生成式世界模型。
- Zhang, J., Huang, Z. et al. *Reinforcing Action Policies by Prophesying.* arXiv, 2025——Prophet,单视角动作条件世界模型基线,光流一致性度量的来源。
- Gao, S. et al. *DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos.* arXiv, 2026——OOD 仿真器泛化实验的对比基线。
- Liu, B. et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* NeurIPS 2023——策略评估与 OOD(LIBERO-Plus Spatial)实验的基准数据集。
