# RoboDream：面向可扩展机器人数据合成的组合式世界模型

> **论文**：*RoboDream: Compositional World Models for Scalable Robot Data Synthesis*
>
> **作者**：Junjie Ye, Rong Xue, Basile Van Hoorick, Pavel Tokmakov, Muhammad Zubair Irshad, Vitor Guizilini, Yue Wang et al.
>
> **机构**：USC Physical Superintelligence (PSI) Lab；Toyota Research Institute
>
> **发布时间**：2026 年 06 月（arXiv 2606.02577）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.02577) | [PDF](https://arxiv.org/pdf/2606.02577)
>
> **分类标签**：`视频世界模型` `机器人数据合成` `组合式解耦` `模仿学习`

---

## 一句话总结

RoboDream 把机器人操作分解为“动作 / 物体 / 场景”三个可重组要素,在视频扩散模型上分别用**渲染的机械臂纯运动视频**(锚定运动学)、**物体先验图**、**场景先验图**三路显式条件驱动生成,从而无需针对新场景微调即可零样本“重绘”出带新物体、新场景、新视角的逼真示范;在四个真实 Franka 操作任务上,50 条真机 + 生成数据混合(Gen-Mix)把 Diffusion Policy 平均成功率从 36.3% 提升到 62.5%,并提出“免道具遥操作”把 50 条轨迹的采集时间从约 2 小时压到 55 分钟。

## 一、问题与动机

模仿学习的主导范式依赖遥操作示范,但真机大规模采集昂贵、耗时且难以跨环境/物体做多样化。生成式(视频扩散)数据引擎是有希望的路线,但现有方法有两类硬伤:

- **纯视觉增强**(ROSIE、RoboEngine 等):只换纹理/背景,机器人轨迹固定,无法造出新的物理交互配置,行为分布被原始示范死死限制。
- **文本→整段视频 + 逆动力学抽动作**(DreamGen 等):可扩展,但机器人本体形态或运动学容易与真实偏离(embodiment hallucination),部署到真机上策略失败。

作者自己的前作 **AnchorDream** 把生成锚定到渲染的机器人运动视频上,解决了本体一致性,但**必须对每个新任务/新环境先采一批域内真数据来微调**,而且环境分布是隐式学到的,对场景和物体缺乏显式可控性。

RoboDream 的核心主张:操作任务在本质上是**组合的**——动作、物体、场景是彼此独立、可重组的要素。一个好的数据引擎应当能围绕一段有效机器人运动“inpaint”任意物体和场景,而不是记住某个数据集的固定分布。要做到这种解耦,就需要在**足够多样**的多环境数据上训练。

## 二、核心方法

RoboDream 是一个多模态视频扩散 Transformer,把“机器人运动生成”与“上下文(物体/场景)生成”解耦。它近似如下条件分布:

$$
p_\theta(o_{1:T} \mid v_{\text{rob}}, I_s, I_o, \ell, \tau)
$$

其中 $v_{\text{rob}} \in \mathbb{R}^{T\times H\times W\times 3}$ 是渲染出的机械臂纯运动视频, $I_s$ 是场景先验图(背景), $I_o$ 是物体先验图(目标物体外观), $\ell$ 是语言指令, $\tau$ 是全局轨迹状态。

用大白话说:模型不再“凭空想象整个画面”,而是拿到三张“分工明确的底稿”——一段只有机械臂在动的动画告诉它“手怎么动、相机在哪(因为运动被画在像素域,视角信息隐含其中)”,一张背景图告诉它“房间长什么样”,一张物体拼贴图告诉它“要操作的东西长什么样”,然后它负责把三者“合成”成一段逼真、物理可行的操作视频。正因为视角信息藏在 $v_{\text{rob}}$ 里,只要换个视角重新渲染运动、再配一张对应视角的场景先验,就能生成新视角的示范。

**四个关键设计:**

**(1) 多模态通道扩展(Multi-Modal Channel Extension)。** 把噪声视频 latent $z_t$、VAE 编码的运动视频 $\mathcal{E}(v_{\text{rob}})$、编码后的场景先验沿通道拼接。由于背景基本静止,把单张场景图沿时间维复制成静态视频 $I_s^T$:

$$
x_{\text{in}} = \mathrm{Concat}\big(z_t,\ \mathcal{E}(v_{\text{rob}}),\ \mathcal{E}(I_s^T)\big)
$$

直觉:给模型一个逐像素对齐的强参考——每一帧都同时“看到”机械臂该动到哪、背景该长什么样。

**(2) 多视角 Token 化(Multi-View Tokenization)。** 不把不同相机(第三人称 + 腕部)的帧横向拼成一张宽图(会带来空间歧义),而是把每个视角当成独立视频条目,各自 tokenize $x_{\text{in}}$ 后把所有 token 堆叠再送入 Transformer。直觉:让模型同时处理多路视角却保留各自的几何透视。

**(3) 物体先验经自注意力注入(Object Prior via Self-Attention)。** 物体先验 $I_o$ 是把任务相关物体**随机旋转、缩放后随机摆在一张空白画布**上,用与视频同一个 VAE 编码得 $z_{\text{obj}}=\mathcal{E}(I_o)$,然后作为额外的 K/V 拼进自注意力:

$$
\mathrm{Attention}\big(Q_{\text{vid}},\ [K_{\text{vid}}; K_{\text{obj}}],\ [V_{\text{vid}}; V_{\text{obj}}]\big)
$$

直觉:视频生成的任意位置都能“回看”物体的视觉细节,从而在场景里任意处准确 inpaint 物体;随机摆放则防止模型把物体外观和它在原图里的位置绑死,逼它学“外观”而非“位置”。

**(4) 交叉注意力条件(Cross-Attention Conditioning)。** 指令 $\ell$ 用 T5 编码;沿用 AnchorDream 的做法,全局轨迹状态 $\tau$ 用 MLP 编码;二者经 cross-attention 注入,保证高层语义与运动学一致。

**先验的自动抽取管线(无需人工标注):** 给第一帧 $o_1$ 和指令,先用 **GPT-5-nano** 识别任务相关物体名(过滤桌子、墙等背景);再用 **Grounded-SAM** 分割这些物体,裁剪后随机旋转/缩放贴到干净画布上得到 $I_o$;把分割出的物体从 $o_1$ 中抠掉,用扩散式 inpainting 模型 **OmniPaint** 补洞,得到干净背景 $I_s$。

**两种部署模式(本文的主要卖点):**

- **检索与重生(Retrieval and Rebirth)。** 对新任务,用 T5 对指令做嵌入,与已有数据集(DROID)所有轨迹指令算余弦相似度,取 top 匹配轨迹;在 **Isaac Lab** 里回放这些轨迹、从新视角渲染出 $v_{\text{rob}}$;再配上新的场景/物体先验,把旧轨迹“重生”到全新上下文中——不用采一条新示范。
- **免道具遥操作(Prop-Free Teleoperation)。** 操作员对着**空气**做操作动作(哑剧/pantomime),可以在真实空工作台或直接在仿真里进行(因为只需要有效的运动学轨迹,不需要真实接触数据);记录轨迹渲染成 $v_{\text{rob}}$,再让模型配上任意目标物体和场景先验,合成机械臂与物体交互的逼真视频。省掉了逐次物理复位、精确摆物,采集可以连续快速进行。

**训练设置:** 从 **Cosmos-Predict2 2B** 基础模型微调,在约 **40k 条**有相机标定的 DROID episode 上训练,2 节点 × 8 张 A100,约一周;处理第三人称静态相机 + 腕部相机两路观测;下游策略统一用 **Diffusion Policy**,以隔离“生成数据”本身对策略学习的影响。

## 三、实验结果

真实机器人:Franka Panda(DROID 平台),四个日常操作任务:Put Marker into Bowl、Remove Marker from Bowl、Put Cube into Cup、Wipe Table with Towel。每个策略评测 **20 次 rollout**;抓放类任务部分成功(抓起但放置失败)算半分。

**表 1:检索与重生 vs. 基线(策略成功率 %)**

| 任务 | Real-50 | Orig-100 | Orig-Mix | Gen-100 | Gen-Mix |
|---|---|---|---|---|---|
| Put Cube into Cup | 35 | 0 | 55 | 20 | **65** |
| Put Marker into Bowl | 30 | 0 | 35 | 15 | **55** |
| Remove Marker from Bowl | 20 | 0 | 20 | 5 | **35** |
| Wipe Table with Towel | 60 | 0 | 70 | 20 | **95** |
| **平均** | 36.3 | 0 | 45.0 | 15.0 | **62.5** |

其中 Real-50 = 50 条域内真机示范;Gen-100 = 从 DROID 检索并重生的 100 条;Gen-Mix = 真机 + 生成各 50% 采样;Orig-100/Orig-Mix = 直接用原始检索到的 DROID episode(不重生)。要点:**Orig-100 全 0%**——直接把 DROID 原始多样环境的数据喂给策略,视角/布局/物体实例与目标域协变量偏移过大,策略完全无法迁移;RoboDream 的“重生”把这个 gap 补上了,Gen-Mix 平均 62.5% 显著超过 Real-50(36.3%)与 Orig-Mix(45.0%)。所有实验用的物体与场景在训练时均未见过(零样本)。

**表 2:免道具遥操作 vs. 真机采集(成功率 %)**

| 任务 | Real-50 | Real w/ Gen Obs | Prop-Free |
|---|---|---|---|
| Put Cube into Cup | 35 | 25 | 30 |
| Put Marker into Bowl | 30 | 20 | 20 |
| Remove Marker from Bowl | 20 | 15 | 20 |
| Wipe Table with Towel | 60 | 60 | 60 |
| **平均** | 36.3 | 30.0 | 32.5 |

Real w/ Gen Obs(真轨迹但观测全换成 RoboDream 生成)平均 30.0%,接近 Real-50 的 36.3%,说明视觉生成保真度高;Prop-Free 平均 32.5%,与真机基线基本持平/略低。**效率**:采 50 条真机约 2 小时(要复位、摆物),采 50 条免道具轨迹仅 55 分钟,约 **2.2×** 更快;而且因抓放类任务运动模式相近,只采了**一个池子的 50 条免道具轨迹**就用来给三个抓放任务分别“上色”,进一步放大了效率收益。

**表 3:生成数据扩展性(固定 Real-50,混入不同量生成数据,成功率 %)**

| 任务 | Real-50 | Mix-100 | Mix-200 | Mix-300 | Mix-400 |
|---|---|---|---|---|---|
| Put Cube into Cup | 35 | 65 | 75 | 80 | 75 |
| Put Marker into Bowl | 30 | 55 | 70 | 70 | 70 |
| Remove Marker from Bowl | 20 | 35 | 45 | 50 | 50 |
| Wipe Table with Towel | 60 | 95 | 100 | 95 | 100 |
| **平均** | 36.3 | 62.5 | 72.5 | 73.75 | 73.75 |

加生成数据持续超越 Real-50,但收益在约 **Mix-200 处饱和**——作者归因为检索轨迹(来自 DROID)的多样性有限或生成的域间隙,最终限制了边际收益。

**组合式生成(定性,Fig. 6):** 从单条基轨迹出发,通过换输入即可得到:换 $I_o$ →新物体实例(蓝笔换红笔);换 $I_s$ →新场景(无缝融入未见背景、无明显伪影);换 $I_o$ 复用同一运动学 →新任务(marker→cube,前提是抓取 affordance 兼容);重渲染运动 + 配对应场景先验 →新视角(可从单视角数据训练鲁棒多视角策略)。均为零样本。

## 四、局限性

- **强依赖渲染运动视频的运动学正确性**:方法假设 $v_{\text{rob}}$ 忠实刻画了机器人在目标场景中该走的轨迹;若轨迹本身在目标场景不可行(如物体位置与哑剧动作不匹配),模型无法纠正——生成保真但语义/物理可能错位。
- **生成质量受训练分布覆盖限制**:作者认为可通过进一步扩大训练(包括把人类视频当作一种 embodiment、锚定到人手运动)来缓解,但本文未验证。
- **继承视频扩散骨干的固有限制**:时序长度、分辨率都受 Cosmos-Predict2 骨干限制。
- **扩展收益早饱和(Mix-200)**:检索轨迹多样性有限,单纯堆生成量收益递减;真正的行为多样性上限仍受源数据集约束。
- **实验规模偏小**:仅 4 个任务、单臂 Franka、每策略 20 次 rollout、下游只测 Diffusion Policy;免道具模式的绝对成功率(32.5%)其实略低于真机(36.3%),卖点是“效率而非提点”,统计显著性在 20 次 rollout 下较弱。

## 五、评价与展望

**优点。** (1) 把“运动学锚定”与“显式场景/物体先验”合到一个框架里,是对 AnchorDream 的关键升级——既保留本体一致性,又把此前隐式、不可控、需域内微调的环境分布变成**显式可换、零样本**的输入,概念上干净且实用。(2)“免道具遥操作”是一个真正原创的采集范式:抓住了“模仿学习真正稀缺的是运动学轨迹而非接触物理”这一洞察,把复位/摆物这一采集瓶颈直接绕开,2.2× 提速且共享轨迹池的做法很聪明。(3) 先验抽取管线全自动(GPT-5-nano + Grounded-SAM + OmniPaint),无需人工标注,工程上可复制。

**缺点与开放问题。** (1) 与仿真派数据生成(MimicGen、Real2Render2Real、DemoGen)相比,RoboDream 主打“不需要 3D 资产/数字孪生、直接在视觉域造新配置”,但代价是**没有真实物理接触信号**——生成的是像素上看起来对的交互,抓取力、碰撞、遮挡下的真实动力学并未被建模,一旦下游需要接触丰富(contact-rich)技能,这条路线的天花板存疑。(2) 行为多样性上限被源数据集(DROID)锁死:检索+重生本质是“旧轨迹换皮”,Mix-200 饱和印证了这点;要突破得靠 prop-free 生成真正的新运动模式,但那又回到“运动学是否可行”的假设风险。(3) 评测太小,免道具模式并未证明能超过真机,只证明“不掉太多且更快”,说服力有限。(4) 与纯视觉增强派(RoboEngine、RoboVIP、RoboTransfer)的边界:RoboDream 声称能造“新物理配置”,但当前实验里“新配置”主要体现在换物体/场景/视角,真正的新交互布局(如不同抓取顺序、不同物体空间关系)展示较少。

**可能的改进方向。** (a) 把接触/力反馈或可微物理作为额外条件或校验,弥补纯视觉域缺乏动力学的问题;(b) 沿作者设想引入大规模人类视频做 embodiment 迁移,突破源数据集多样性天花板;(c) 用生成数据的置信度/一致性做过滤(如多视角一致性、逆动力学回验),缓解“生成保真但运动学错位”的风险;(d) 扩大任务集、多策略(如 ACT、VLA)与更大 rollout 数,给出统计更硬的结论。总体看,这是一篇工程完整度高、洞察扎实(尤其 prop-free)的数据引擎工作,方法优雅,但受限于视觉域无物理、小规模评测,证据强度还不足以确立“可扩展数据引擎”的普适性。

## 参考

1. Ye et al., *AnchorDream: Repurposing Video Diffusion for Embodiment-Aware Robot Data Synthesis*, ICRA 2026 —— 本文直接前作,提出运动学锚定但需域内微调。
2. Khazatsky et al., *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*, RSS 2024 —— RoboDream 的训练数据与检索源。
3. NVIDIA Cosmos, *Cosmos-Predict2: General-Purpose World Foundation Models for Physical AI*, 2025 —— RoboDream 微调所用的视频扩散骨干。
4. Jang et al., *DreamGen: Unlocking Generalization in Robot Learning through Video World Models*, CoRL 2025 —— 文本→视频→逆动力学的对照范式。
5. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning*, CoRL 2023 —— 仿真侧组合式数据生成的代表,与 RoboDream 视觉域路线形成对比。
