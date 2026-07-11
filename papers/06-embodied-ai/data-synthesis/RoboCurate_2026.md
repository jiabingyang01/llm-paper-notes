# RoboCurate：用动作校验的神经轨迹为机器人学习榨取多样性

> **论文**：*RoboCurate: Harnessing Diversity with Action-Verified Neural Trajectory for Robot Learning*
>
> **作者**：Seungku Kim*、Suhyeok Jang*（共同一作）、Byungjun Yoon、Dongyoung Kim、John Won、Jinwoo Shin（通讯）
>
> **机构**：KAIST；RLWRLD
>
> **发布时间**：2026 年 02 月（arXiv 2602.18742）
>
> **发表状态**：未录用（预印本，Preprint February 24, 2026）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.18742) | [PDF](https://arxiv.org/pdf/2602.18742)
>
> **分类标签**：`合成数据` `neural-trajectory` `动作校验` `视频生成` `VLA预训练`

---

## 一句话总结

RoboCurate 把"视频生成 + IDM 反标动作"得到的 neural trajectory 做成一条可用的机器人预训练数据管线：一方面用 I2I 图像编辑与 V2V 视频迁移在**保运动**的前提下扩充场景/外观/任务多样性,另一方面把"IDM 标注的动作对不对"这件事转化为"把该动作在仿真里重放、再比较重放视频与生成视频的运动一致性",用一个挂在冻结 V-JEPA2 上的轻量 attentive probe 打分过滤;在 GR-1 Tabletop（300 demos）相对纯真机基线成功率提升 +70.1%、DexMimicGen +16.1%、真实 ALLEX 人形灵巧手协同微调上 +179.9%。

## 一、问题与动机

机器人数据昂贵稀缺,近来一条思路是用视频生成模型造"neural trajectory"——从初始帧 + 任务指令生成任务执行视频,再用 inverse dynamics model（IDM）把无动作的视频反标成带动作的轨迹,作为策略训练的合成数据(代表工作 DreamGen, Jang et al. 2025)。相比仿真,它视觉上更接近真实、还能学到训练集之外的新动作。但这条管线有两个顽疾:

1. **多样性受限**。生成视频受"人工采集的初始帧/视觉上下文"约束,场景、外观、任务的变化都很窄。
2. **动作标签有噪声**。视频生成阶段可能不遵循指令、或生成物理不合理的运动;即使视频看着对,依赖学出来的 IDM 也会给出错误动作标签,污染下游策略。

现有做法多用 VLM 去判"视频是否遵循指令 / 是否物理合理",但作者指出:这类判断太粗,只看表面是否符合基本物理,**并不直接评估动作本身**(比如手臂是否真的够到了物体这种对策略至关重要的运动),因此不足以筛出真正有益的合成数据。RoboCurate 要同时解决"多样性不够"和"动作没被真正校验"这两件事。

## 二、核心方法

整体两阶段(见原文 Figure 1):**Stage 1 生成**(I2I+I2V 或 I2V+V2V 两条支路造多样化视频)、**Stage 2 过滤**(把 IDM 动作在仿真重放,与生成视频做运动一致性分类,一致的进 Curated Data 喂 VLA,不一致的丢弃)。

### 预备:flow matching、IDM 与 neural trajectory

视频扩散用 Flow Matching,在数据 $\mathbf{x}=\mathcal{E}(\mathbf{w})$ 与高斯噪声 $\boldsymbol{\epsilon}$ 间线性插值:

$$\mathbf{x}_t = (1-t)\mathbf{x} + t\boldsymbol{\epsilon},\qquad \mathcal{L}(\theta)=\mathbb{E}_{t,\mathbf{x},\boldsymbol{\epsilon}}\big[\lVert v_\theta(\mathbf{x}_t,t,\mathbf{c})-(\boldsymbol{\epsilon}-\mathbf{x})\rVert_2^2\big]$$

用大白话说:训练一个"速度场"网络,让它从任意含噪视频指回干净视频的方向,推理时沿这个方向积分就能去噪生成视频。

IDM 从当前观测 $x_t$ 与未来观测 $x_{t+H}$ 反推中间动作块:

$$a_{t:t+H-1}=\mathrm{IDM}(x_t, x_{t+H})$$

用大白话说:看"前一帧和 H 步之后的一帧",猜出这段时间机器人做了什么动作——这就是把无动作视频变成带动作数据的"伪标注器"(本文 IDM 用 DiT + flow matching,视觉编码器为 SigLIP-2)。

策略端是带 diffusion action head 的 VLA(基座 GR00T N1.5),同样 flow-matching 目标:

$$\mathcal{L}_{\mathrm{IL}}(\theta;\mathcal{D})=\mathbb{E}_{t,A_t,\tau}\big[\lVert v_\theta(A_t^\tau,\tau\mid o_t,q_t,I)-(\boldsymbol{\epsilon}-A_t)\rVert_2^2\big]$$

其中 $A_t^\tau=\tau A_t+(1-\tau)\boldsymbol{\epsilon}$。注意合成数据 $\mathcal{D}_{\mathrm{syn}}$ 的本体感受状态被**置零填充**(因为 IDM 只出动作、不出状态)。最终在 $\mathcal{D}=\mathcal{D}_{\mathrm{real}}\cup\mathcal{D}_{\mathrm{syn}}$ 上联合训练。

### 3.1 生成多样的操作场景

沿两个轴扩多样性,核心约束是**改外观不改运动**,这样 IDM 反标的动作仍然可复用:

- **视觉多样性**。(1) 对初始帧做 instruction-guided 的 I2I 编辑(FLUX.2-dev),沿四个轴变化:桌面外观、目标物体身份/外观、光照、背景;为保结构不崩,用 Canny 边缘图做条件,并先让 VLM 描述初始图再拼上原指令引导编辑。编辑后的初始帧再送 I2V 生成视频。(2) 对已生成视频做 V2V 迁移(Cosmos-Transfer2.5-2B),同样以 Canny 边缘视频为条件,只改纹理和颜色、保物体身份与形状,并额外要求"保持机器人本体颜色不变"以免改动具身外观。
- **任务多样性**。用(专有)VLM 针对初始场景生成合理的新指令,沿四个轴:行为、目标物体、放置、机器人用手(明确指定 active hand,单双手 1:1,支持双手操作),few-shot 提示保证一致性。

### 3.2 动作级过滤:把动作校验变成视频运动一致性分类

一条 neural trajectory 表示为(生成视频, IDM 动作)对 $(\mathbf{w}_{\mathrm{gen}}, a_{\mathrm{IDM}})$。RoboCurate 把 $a_{\mathrm{IDM}}$ 在仿真里重放,渲染出运动一定与 $a_{\mathrm{IDM}}$ 一致的 rollout 视频 $\mathbf{w}_{\mathrm{sim}}(a_{\mathrm{IDM}})$,于是**动作对不对 ⇔ 生成视频与仿真重放视频运动一不一致**。

关键:probe 的正负样本**只用真机演示构造,不用带噪的 neural trajectory**。对每条真机动作 $a_{\mathrm{real}}$,渲染仿真重放 $\mathbf{w}_{\mathrm{sim}}(a_{\mathrm{real}})$ 组正样本:

$$\mathcal{P}^+=\big\{(\mathbf{w}_{\mathrm{real}}^{t:t+H},\ \mathbf{w}_{\mathrm{sim}}(a_{\mathrm{real}})^{t:t+H})\big\}_t$$

负样本 $\mathcal{P}^-=\mathcal{P}^-_{\mathrm{shift}}\cup\mathcal{P}^-_{\mathrm{cross}}$ 两类:同一 episode 内**时间错位** $\mathcal{P}^-_{\mathrm{shift}}=\{(\mathbf{w}_{\mathrm{real}}^{t:t+H},\mathbf{w}_{\mathrm{sim}}(a_{\mathrm{real}})^{t':t'+H})\mid t'\neq t\}$,以及**跨 episode 配错** $\mathcal{P}^-_{\mathrm{cross}}=\{(\mathbf{w}_{\mathrm{real}}^{t:t+H},\mathbf{w}_{\mathrm{sim}}(a'_{\mathrm{real}})^{t:t+H})\mid a'_{\mathrm{real}}\neq a_{\mathrm{real}}\}$。

用大白话说:告诉 probe"真机片段和它自己动作的仿真重放"是对齐的正例,而"时间错开半拍"或"配了别人的动作"是负例——这样它学到的是**细粒度运动/机器人几何是否吻合**,而不是外观相似度。

两段视频各用冻结视频编码器 $f_\phi$（V-JEPA2）编码 $\mathbf{z}_1,\mathbf{z}_2$,喂一个**单层 cross-attention + 可学习 query token** 的 attentive probe,线性头出对齐 logit,再取 sigmoid 得对齐概率,用 BCE 训练:

$$\ell=g_\theta([\mathbf{z}_1,\mathbf{z}_2]),\quad p=\sigma(\ell),\quad \mathcal{L}(\theta;\mathcal{P})=\mathbb{E}_{((\mathbf{w}_1,\mathbf{w}_2),y)}\big[-y\log p-(1-y)\log(1-p)\big]$$

推理时把 $(\mathbf{w}_{\mathrm{gen}},\mathbf{w}_{\mathrm{sim}}(a_{\mathrm{IDM}}))$ 送 $g_\theta$,只保留 $p>c$ 的样本。

### 3.3 Best-of-N:把过滤器当生成阶段的 critic

同一个 $p$ 也能当视频生成的 critic:采样 $N$ 个候选(不同噪声种子)连同各自 IDM 动作,选 $p$ 最高的那对。好处是**数据稀缺的微调场景下不必丢样本**,直接在生成时挑出动作最可信的那条(ALLEX 协同微调用的就是这套 RoboCurate*,不做视觉增广)。

### 训练配置要点

- 预训练:数据 ActionNet(Fourier GR1-T1 人形,44 维关节空间,原 30K 遥操作、取 3K 子集);WSD 学习率,60K step,前 50K 用全部 neural trajectory、后 10K 只用 curated;real:neural 采样 1:1,real 与合成用不同 embodiment tag;batch 512。
- 生成侧:视频基座 Cosmos-Predict2-14B(在 ActionNet + GR00T-GR1-100 上微调),约 10K 桌面视频,I2I:V2V = 2:1;后处理用 Gemini 3 Pro 判指令遵循、Gemini 2.5 Flash 打 1–5 物理合理性(取 ≥3),留下正好 10K 条 pre-filtering 轨迹。
- probe:0.3B V-JEPA2-large 冻结,clip 长 H=16、时间步幅 4、输入 256×256、8 头 1 层 cross-attn、AdamW lr 1e-4。

## 三、实验结果

基座策略统一为 GR00T N1.5;所有用合成数据的方法共享同一份 10K 过滤前 neural trajectory 以公平对比。DreamGen 指"I2V + IDM、无视觉多样化、无动作过滤"的前作管线。

**Table 1 — GR-1 Tabletop**(50 trials × 24 任务,18 rearrangement + 6 articulated,成功率 %）:

| 方法 | Synth | Filter | 300 Rearr | 300 Artic | 300 Avg | 1000 Avg |
|---|---|---|---|---|---|---|
| Real | ✗ | ✗ | 16.1 | 13.3 | 15.4 | 30.3 |
| w/ DreamGen | ✓ | ✗ | 21.1 | 14.7 | 19.5 | 32.2 |
| w/ RoboCurate | ✓ | ✗ | 23.2 | 21.0 | 22.7 | 34.8 |
| **w/ RoboCurate** | ✓ | ✓ | **25.4** | **28.7** | **26.2** | **37.9** |

300 demos 下 15.4→26.2 即 **+70.1%** 相对提升;而 DreamGen 只有 +26.6%。

**Table 2 — DexMimicGen**(6 任务:3 GR-1 人形 + 3 双臂 Panda 灵巧手,100 demos/任务,3 seed 平均):

| 方法 | Synth | Filter | GR-1 人形 | 双臂 Panda | Avg |
|---|---|---|---|---|---|
| Real | ✗ | ✗ | 56.9 | 32.2 | 44.6 |
| w/ DreamGen | ✓ | ✗ | 57.1 | 35.6 | 46.4 |
| w/ RoboCurate | ✓ | ✗ | 59.3 | 39.3 | 49.3 |
| **w/ RoboCurate** | ✓ | ✓ | **62.7** | **40.9** | **51.8** |

44.6→51.8 即 **+16.1%**(DreamGen 仅 +4.0%);且注意用 GR-1 人形数据预训练的先验能跨具身迁移到双臂 Panda 灵巧手。

**Table 3 — 真实 ALLEX 人形灵巧手协同微调**(24 trials × 3 任务,seen 任务仅 48 demos,OOD 任务不采任何真机数据;RoboCurate* = 生成阶段 Best-of-N,无视觉增广):

| 方法 | ID: Place Can | OOD 新物体: Place Cup | OOD 新行为: Pour Can | Avg |
|---|---|---|---|---|
| Real | 25.0 | 16.7 | 0.0 | 13.9 |
| w/ DreamGen | 37.5 | 33.3 | 12.5 | 27.8 |
| **w/ RoboCurate\*** | **47.9** | **43.8** | **25.0** | **38.9** |

13.9→38.9 即 **+179.9%**(DreamGen +100.0%);新物体子项 16.7→43.8 为 +162.3%;最难的"新行为 pour can"从 **0.0%→25.0%**,展示了对分布外新动作的涌现。

**Table 4 — 与其它过滤策略比**(GR-1 Tabletop,1000 demos,同 10K 轨迹):动作级过滤显著优于纯"视频级物理合理性"判官。

| 方法 | Avg |
|---|---|
| Real + Neural(不过滤) | 32.1 |
| w/ DreamGenBench(VLM 判物理) | 35.4 |
| w/ VideoCon-Physics(7B VLM 判物理) | 35.2 |
| **w/ RoboCurate** | **38.3** |

**Table 5 — 多样性消融**(300 demos,固定 10K 轨迹):性能随任务多样性单调上升,视觉增广在同等任务多样性下再加一档。

| 视觉增广 | 任务多样性 | Avg |
|---|---|---|
| ✗ | 25% | 12.5 |
| ✗ | 50% | 17.3 |
| ✗ | 100% | 19.7 |
| ✓ | 100% | **23.3** |

**Table 6 — 过滤组件消融**(300 demos,编码器统一 V-JEPA2):attentive probe(用自动构造的正负对、无人工标注)取得最优 26.2;仅用冻结编码器 cosine 相似度阈值只到 23.8;而**引入人工标注反而更差**(23.5),因为对细微动作错配人工判断噪声大,自动配对反而提供了更一致的监督。

## 四、局限性

- **强依赖仿真重放**:方法要求能把 IDM 动作在仿真里重放并渲染出对应具身的 rollout 视频,需要该具身的仿真资产与 IDM;换新具身(如 ALLEX)得从零训 IDM 并采真机数据训 probe,迁移成本不低。
- **管线重、成本高、可复现性弱**:串联 Cosmos-Predict2-14B、FLUX.2-dev、Cosmos-Transfer2.5、V-JEPA2,外加专有 Gemini 3 Pro / 2.5 Flash 做指令生成与后处理,生成与筛选算力开销大,专有 VLM 也影响复现。
- **校验的是"运动一致",不是"任务达成"**:probe 判的是生成视频与仿真重放的运动是否吻合;若 IDM 出的动作物理上自洽但并未真正完成任务目标,该指标未必能识别。
- **两段视频都是代理**:生成视频与仿真重放本身各有渲染域差,运动一致性度量对这种双向 domain gap 的鲁棒性未充分分析。
- **绝对成功率仍偏低**且评测集中在桌面操作(GR-1 Tabletop 最好也仅约 26%);合成数据本体感受状态被置零填充,长时序/接触丰富任务上的收益未验证。

## 五、评价与展望

- **最有价值的贡献**是把"IDM 动作标签质量"这个 neural-trajectory 管线的核心痛点,用"仿真当动作渲染器 + 视频运动一致性分类"闭环起来:相比 DreamGenBench、VideoCon-Physics 这类只看视频物理合理性的 VLM 判官,它把校验对准了动作本身,Table 4/6 也用数据支持了"动作级 > 视频级"这一论点。用真机演示自动构造正负对、避免人工标注(且证明人工标注会掉点)的设计干净利落。
- **与公开工作的关系**:本质上 RoboCurate = DreamGen(Jang et al. 2025,I2V+IDM 的 neural trajectory)+ 视觉多样化(I2I/V2V)+ 动作过滤,两个新增件都做了对照消融,归因清晰。它属于"世界模型/视频生成即数据"的大方向(与 UniPi、RoboDreamer、GR-2、GigaBrain 等同脉),但独特点在于用仿真做 grounding 而非再堆一个判别式世界模型。probe 挂在冻结 V-JEPA2 上、单层 cross-attention 的做法也很轻。
- **开放问题与可能改进**:(1) 跨具身泛化——能否训一个具身无关的运动一致性 probe,免去每个具身重训 IDM/probe?(2) 双向 domain gap——生成视频与仿真视频风格差异是否会系统性误判某类运动,值得做鲁棒性研究;(3) Best-of-N critic 与在线 RL/DAgger 结合,或把 $p$ 作为样本加权而非硬阈值,可能进一步提升数据利用率;(4) 把"运动一致"升级为"任务达成"层面的校验(例如结合成功判别器),或许能堵住"自洽但不达标"的漏洞。总体是一篇工程完整、消融扎实、把合成机器人数据"造得更多样 + 筛得更可信"这条路走通的扎实工作,主要门槛在于对仿真与多模型基座的重依赖。

## 参考

1. Jang et al., *DreamGen: Unlocking Generalization in Robot Learning through Video World Models*, arXiv:2505.12705, 2025 —— 直接前作与最强对照基线。
2. Baker et al., *Video Pretraining (VPT): Learning to Act by Watching Unlabeled Online Videos*, NeurIPS 2022 —— IDM 反标动作的思想来源。
3. Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985, 2025 —— 过滤所用的冻结视频编码器。
4. Bjorck et al., *GR00T N1.5: An Improved Open Foundation Model for Generalist Humanoid Robots*, 2025 —— 本文基座 VLA 策略。
5. Jiang et al., *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning*, ICRA 2025 —— 主要评测基准之一。
