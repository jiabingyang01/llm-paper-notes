# MCIL / LangLfP：从海量无标签 Play 数据中学会自由语言条件的模仿学习

> **论文**：*Language Conditioned Imitation Learning over Unstructured Data*
>
> **作者**：Corey Lynch*, Pierre Sermanet*（* 同等贡献）
>
> **机构**：Robotics at Google
>
> **发布时间**：2020 年 05 月（arXiv 2005.07648，v2 修订于 2021 年 07 月）
>
> **发表状态**：未录用（预印本，PDF 中未标注录用信息）
>
> 🔗 [arXiv](https://arxiv.org/abs/2005.07648) | [PDF](https://arxiv.org/pdf/2005.07648)
>
> **分类标签**：`语言条件模仿学习` `Multicontext Imitation Learning` `Play 数据 relabeling` `Latent Motor Plans` `预训练语言嵌入迁移`

---

## 一句话总结

作者提出 Multicontext Imitation Learning（MCIL）：把"目标图像条件""任务 ID 条件""自由语言条件"等异构的模仿学习数据集统一编码进同一个隐目标空间 $z$,用同一个策略网络端到端联合训练;在此框架下的具体实例 LangLfP 只需给约 7 小时、~1000 万段无标签teleoperated "play" 轨迹中随机抽取的 1 万段配上人写的事后（hindsight）自然语言指令(占总数据不到 1%),就能训练出一个仅凭自由语言就能在 3D 桌面环境里连续完成长程操作任务的视觉运动策略,在 18 任务基准上像素输入下达到 68.6% 成功率、四段长程任务（925 条指令）51%~62% 成功率,且换用预训练多语言句子编码器（MUSE）后可零样本泛化到同义词改写指令（60.2% vs 随机策略 0%）和 16 种未训练语言的指令（56.0%）。

## 一、问题与动机

在开放世界里部署模仿学习的机器人,一个关键但常被简化的问题是**任务指定（task specification）**:未经训练的用户如何告诉机器人要做什么?此前的目标条件模仿学习/强化学习大多假设条件是任务 ID（one-hot）、目标图像,或状态空间中的目标配置——这些在仿真里可行,但在开放世界中往往不现实。自由形式的自然语言才是人类最灵活直观的任务描述方式,但已有的指令跟随（instruction following）研究通常对观测空间、动作空间或语言本身做了限制性假设(如 2D 观测、简化的离散动作 primitive、受限词表/语法的合成语言),难以直接用于高维像素输入、8 自由度连续控制、长程 3D 物体操作这样复杂的机器人场景。据作者称,本文是第一个把"自然语言条件 + 高维像素输入 + 8-DOF 连续控制 + 长程 3D 操作"这四者结合在一起的指令跟随工作。

语言条件模仿学习还带来一个新的核心矛盾:要训练语言条件策略,似乎必须给每条演示配上语言标签,但大规模采集"演示 + 语言标注"的成对数据成本很高。与此同时,已有工作(Lynch et al. 2019 的 Learning Latent Plans from Play,即 LfP/LMP)证明可以从完全无标签、无任务分段的 teleoperated "play" 数据中,通过事后重标注（relabeling）学习目标图像条件策略——但这类方法的目标空间被限制为"曾经真实到达过的观测状态",无法用自然语言这种不在状态空间 $S$ 内的目标来重标注,因而无法把无标签数据用于语言条件学习。本文要解决的问题就是:**能否设计一种方法,让语言条件的视觉运动策略主要从廉价的无标签 play 数据中学习感知、控制和语言理解,只用极少量人工语言标注就完成语言 grounding?**

## 二、核心方法

整体流程(对应论文 Fig. 1)分四步:1) 采集大量无结构 teleoperated play 数据;2) 把 play 数据重标注为目标图像条件的模仿样本;3) 用 Multicontext Imitation Learning 训练一个可被"目标图像"或"语言"条件的单一策略;4) 测试时只用语言条件。

**1) Play 数据与目标图像重标注（沿用 Lynch et al. 2019 的做法）。** 人类远程操作机器人在场景中自由、无脚本地探索各种物体交互,得到一段不分段的观测-动作流 $\{(s_{0:t}, a_{0:t})^n\}$。Algorithm 2 对这条长流按窗口大小 $w \in [w_{\text{low}}, w_{\text{high}}]$ 做滑窗采样,把每个窗口 $\tau = (s_{i:i+w}, a_{i:i+w})$ 的最后一帧观测当作"事后目标" $s_g = s_w$,得到目标图像条件模仿数据集 $D_{\text{play}} = \{(\tau, s_g)\}$。训练目标是标准的目标条件模仿损失

$$\mathcal{L}_{\text{GCIL}} = \mathbb{E}_{(\tau,\, s_g)\sim D_{\text{play}}}\left[\sum_{t=1}^{|\tau|} \log \pi_\theta(a_t \mid s_t, s_g)\right]$$

**用大白话说**：把一段随意操作的长视频切成很多 1-2 秒的小片段,每段的"最后一帧长什么样"就被自动标记为"这段动作要去的目标",不需要人工干预就能把无标签数据变成海量的"给定目标图像,学怎么走过去"的监督样本。

**2) 用极少量人工语言标注对齐 play 与语言（Algorithm 3）。** 从 $D_{\text{play}}$ 中随机抽取 $K$（本文取 $K=10\text{K}$，相对 ~1000 万段窗口的总量不到 1%）段轨迹 $\tau$,让标注员观看首尾帧循环视频后回答"How do I go from start to finish?",写下一句无约束的自由语言描述作为事后指令 $l$,得到语言条件数据集 $D_{(\text{play,lang})} = \{(\tau, l)\}$。这一步把语言标注的成本从"每条演示都要标"降到"仅需给一小撮随机窗口配语言"。

**3) Multicontext Imitation Learning（MCIL,Algorithm 1）。** 核心思想是用同一个隐目标条件策略 $\pi_\theta(a_t \mid s_t, z)$ 统一表示一大类由不同"上下文类型"（goal image、task id、natural language 等）描述的策略族。给定 $K$ 个异构上下文数据集 $D=\{D^0,\dots,D^K\}$（每个 $D^k = \{(\tau^k_i, c^k_i)\}$ 是 (演示, 上下文) 对）,以及对应的 $K$ 个编码器 $\mathcal F=\{f_\theta^0,\dots,f_\theta^K\}$（分别把图像目标、任务 ID、语言等原始上下文映射到共享隐目标空间 $z=f_\theta^k(c^k)\in\mathbb R^{32}$）,每一步训练从每个数据集采样一个 batch,编码得到 $z$,累加各数据集上的模仿损失后按数据集数取平均,再对策略与所有编码器联合做一次梯度更新:

$$\mathcal{L}_{\text{MCIL}} = \frac{1}{|D|}\sum_{k=0}^{K} \mathbb{E}_{(\tau^k,\,c^k)\sim D^k}\left[\sum_{t=1}^{|\tau^k|} \log \pi_\theta\big(a_t \mid s_t,\, z = f_\theta^k(c^k)\big)\right]$$

**用大白话说**：不同任务描述方式（图像目标、one-hot 任务号、自然语言）本质上都是在说"要去哪里",只是"语言"不同;MCIL 干脆给每种描述方式配一个专门的编码器,把它们都翻译成同一种"隐目标语言" $z$,再用同一个策略网络学"看到当前状态 $s_t$ 和目标 $z$,该怎么做"。因为编码器和策略是联合端到端训练的,来源丰富的数据集（如无标签 play 图像目标）可以通过共享的隐目标空间把学到的感知与控制能力"迁移"给数据稀缺的上下文类型（如语言）——作者将其类比为一种通过共享目标空间实现的迁移学习。

**4) LangLfP：MCIL 在 play + 语言两个数据源上的具体实例化。** LangLfP 就是把 $D=\{D_{\text{play}}, D_{(\text{play,lang})}\}$ 喂给 MCIL,编码器为图像目标编码器 $g_{\text{enc}}$（感知模块输出接 2 层 2048 单元 ReLU MLP,得到 32 维隐目标）和语言编码器 $s_{\text{enc}}$（子词 tokenizer → 8 维子词向量查表 → 平均池化 → 2 层 2048 单元 ReLU MLP）。控制模块沿用 Lynch et al. 2019 的 Latent Motor Plans（LMP）：一个条件 seq2seq CVAE,用后验 $q(z^p\mid\tau)$ 把整条状态-动作序列编码为"计划"（plan）隐变量,用先验 $p(z^p\mid s_0, z^g)$ 只依据初始状态和隐目标预测计划分布,再用 teacher forcing 的 RNN 解码器 $p(a_t\mid s_t, z^p, z^g)$ 重构动作,训练目标是标准 CVAE 的 ELBO（$\beta$-VAE 形式,论文取 $\beta=0.01$）：

$$\mathcal{L}_{\text{LMP}} = \mathbb{E}_{q(z^p\mid\tau)}\Big[\sum_{t=1}^{|\tau|}\log p(a_t\mid s_t, z^p, z^g)\Big] - \beta\, D_{\mathrm{KL}}\big(q(z^p\mid\tau)\,\|\,p(z^p\mid s_0, z^g)\big)$$

**用大白话说**：因为同一句指令（如"开抽屉"）在 play 数据里可能对应无数种具体的执行轨迹(手先摸门把手还是先绕开障碍物),模型需要一个额外的隐变量 $z^p$（"计划"）来吸收这种多模态性——训练时"作弊"看到完整轨迹去编码这个计划,推理时只能靠先验从当前状态和目标去猜一个合理的计划,再据此逐步解码动作,和标准 CVAE 的训练/推理分离逻辑一致。感知模块是 3 层卷积 + spatial softmax + 全连接的小型 CNN,把 200×200×3 RGB 图像压成 64 维向量,与 8 维本体感受拼接成 72 维感知嵌入,不做任何图像增强,全程与策略、编码器一起端到端训练,不使用任何辅助的自监督表征损失。

**5) TransferLangLfP：接入预训练语言模型获得对同义词/跨语言的鲁棒性。** 把语言编码器换成冻结的多语言 Universal Sentence Encoder（MUSE,200K 词表,句子→512 维向量),该向量经 2 层 2048 单元 ReLU MLP 投影到隐目标空间,不微调 MUSE 权重。动机是:训练指令集有限,但测试时用户可能用同义词甚至外语表达同一任务(如把"drag the block from the shelf"换成"retrieve the brick from the cupboard")。因为 MUSE 的预训练语义空间已经把同义句子映射到相近向量,策略只需学会在这个空间里做正确响应,就能不靠新增机器人示教覆盖到大量语言表达变体。

## 三、实验结果

实验环境为论文沿用 Lynch et al. 2019 的仿真 3D "Playroom"桌面场景(MuJoCo 物理引擎):一台 8-DOF 机械臂(位置+欧拉角+夹爪的笛卡尔控制,30Hz 闭环),桌上有滑动门、抽屉、3 个可推按钮、一块可移动方块,旁边有垃圾桶,覆盖 18 种子任务(开/关滑门、开/关抽屉、抓取/抓起/竖抓、敲击、拉出/放入置物架、按红/绿/蓝按钮、左/右旋转、扫动/左扫/右扫)。评测分两种输入设定:**pixel**(原始图像+本体感受,感知端到端学习)和**state**(直接给真实物体状态,作为性能上界参考)。长程评测通过组合 18 个子任务的所有合法转移构造 Chain-2/3/4 基准,Chain-4 共 925 条长程指令。

**主结果(Table I,人类语言条件视觉操作)：**

| 方法 | 输入 | 训练数据来源 | 任务条件 | Multi-18 成功率(18任务) | Chain-4 成功率(925条长程指令) |
|---|---|---|---|---|---|
| LangBC | pixels | 预定义演示(100条/任务) | 语言 | 20.0% ± 3.0 | 7.1% ± 1.5 |
| Restricted LangLfP | pixels | 无标签 play(限量对齐) | 语言 | 47.1% ± 2.0 | 25.0% ± 2.0 |
| LfP | pixels | 无标签 play | 图像目标 | 66.4% ± 2.2 | 53.0% ± 5.0 |
| LangLfP(本文) | pixels | 无标签 play | 语言 | 68.6% ± 1.7 | 52.1% ± 2.0 |
| **TransferLangLfP(本文)** | pixels | 无标签 play | 语言 | **74.1% ± 1.5** | **61.8% ± 1.1** |
| LangBC | states | 预定义演示 | 语言 | 38.5% ± 6.3 | 13.9% ± 1.4 |
| Restricted LangLfP | states | 无标签 play(限量对齐) | 语言 | 88.0% ± 1.4 | 64.2% ± 1.5 |
| LangLfP(本文) | states | 无标签 play | 语言 | 88.5% ± 2.9 | 63.2% ± 0.9 |
| **TransferLangLfP(本文)** | states | 无标签 play | 语言 | **90.5% ± 0.8** | **71.8% ± 1.6** |

要点:1) LangLfP 在所有基准上都能匹配甚至(states 设定下)略优于只用图像目标条件的 LfP,说明约 1% 的语言标注量就足以把"无标签 play 学到的能力"迁移到语言条件设定,回答了论文的 Q1;2) 在训练数据总量对齐的受控对比中(Restricted LangLfP vs LangBC),利用无标签 play 数据的方法在 pixel 设定下 Multi-18 由 20.0% 提升到 47.1%,Chain-4 由 7.1% 提升到 25.0%,验证了大规模无标签演示数据能显著提升语言条件性能(Q2);3) TransferLangLfP 系统性优于 LangLfP 和 LfP,是论文报告的"预训练句子嵌入可以提升语言条件机器人控制策略收敛"的首个证据(Q3)。

**分布外(OOD)鲁棒性(Table II)：**

| 方法 | OOD-syn(约 14,609 条同义词替换指令) | OOD-16-lang(约 24 万条 16 语言指令) |
|---|---|---|
| 随机策略 | 0.0% ± 0.0 | 0.0% ± 0.0 |
| LangLfP | 37.6% ± 2.3 | 27.94% ± 3.5 |
| **TransferLangLfP** | **60.2% ± 3.2** | **56.0% ± 1.4** |

只有接入预训练语言模型的 TransferLangLfP 能较好地应对训练集之外的同义词改写与 16 种从未训练过的语言(通过 Google 翻译 API 生成),验证了 Q4:预训练语言嵌入能让策略以零新增示教的方式扩大"可理解的指令表达方式"数量,但论文强调这只是扩展了对同一组固定训练行为的"表达方式"覆盖,并不测试对新种类操作任务的泛化。

**其它关键发现：**
- **模型容量与数据类型的交互(Fig. 6)：** 在 states 输入下,从无标签 play 学习的 Restricted LangLfP 性能随模型参数量(10M→120M)近似线性增长;而只用预定义多任务演示训练的 LangBC 在约 20M 参数时性能见顶后随容量增大而下降——说明"收集大规模无标签数据 + 扩大模型容量"是比"扩大预定义任务演示规模"更具可扩展性的组合。
- **语言标注量的边际收益(Fig. 15)：** 把 (play, language) 配对数从 250 扩到 10K,states 输入模型在 5K→10K 区间已趋于饱和,但 pixel 输入模型在 10K 时仍未收敛,说明语言标注量的主要瓶颈在于视觉 grounding 而非语言理解本身。
- **语言解锁人机协作(Fig. 7)：** 语言条件让操作员可以在策略卡住时用自然语言实时插话纠偏(例如先说"move back""move the door all the way right"再重发原指令),这是图像目标或 one-hot 条件难以支持的交互模式。
- **零样本任务组合(Sec. P,Fig. 13/14)：** LangLfP 可以把训练集里从未出现过的复合任务(如"把物体放进垃圾桶"、"把物体放上置物架")拆成两条已训练过的子指令(先"pick up the object"再"put the object in the trash")顺序执行,零样本完成。
- **评测协议的公平性(Sec. R,Fig. 12)：** 论文指出若像先前工作那样把每个测试 episode 初始化到对应演示的第一帧状态("correlated" 初始化),初始位姿本身会泄露任务信息,LangBC 在这种设定下第一步表现尚可但迅速在链式任务中退化;论文改用固定的"neutral"初始位姿以强制策略完全依赖语言完成任务推理,LangLfP 在两种初始化间的性能差距远小于 LangBC。

## 四、局限性

论文在正文及附录中明确指出以下局限:

1. **无自主策略提升机制。** LangLfP 本质上仍是纯模仿学习方法,不具备强化学习式的自主策略改进能力;作者认为未来一个有前景的方向是结合 teleoperated play 的数据覆盖度、multicontext 模仿的可扩展性与强化学习的自主提升(如 Relay Policy Learning 中 LfP+RL 的组合)。
2. **任务/环境范围受限于单一仿真场景。** 全部实验都在固定物体集合的单个模拟 3D 桌面环境中完成,遵循标准模仿学习"训练/测试任务同分布"假设,论文明确留下"能否在覆盖多房间、多物体的大规模 play 语料上训练以泛化到未见过的房间或物体"作为开放问题。
3. **闭环失败模式。** 定性视频显示策略在部分任务上会反复尝试但最终超时失败;还观察到一种复合误差——机械臂在遥操作 play 中较少出现、但策略推理时会翻转进入别扭的姿态构型(可能与欧拉角旋转表示的奇异性有关),作者猜测更稳定的旋转表示或更丰富的 play 数据采集有望缓解。
4. **OOD 泛化的范围有限。** 同义词/跨语言鲁棒性实验(OOD-syn、OOD-16-lang)只测试了"用不同方式描述同一组已训练行为"的能力,并不测试对训练集中未出现过的新操作技能的泛化;且这一鲁棒性完全来自预训练语言嵌入的语义结构,论文没有对预训练语言模型做机器人数据上的联合微调,是否能进一步提升尚不清楚。
5. **无真实机器人验证与视觉增强。** 全部结果均来自 MuJoCo 仿真环境,未在真实机器人上验证;感知模块也未做任何图像光度增强,作者自己承认这可能限制了性能(以及潜在的 sim-to-real 迁移能力)。

## 五、评价与展望

**贡献与优点：** 本文最核心的贡献是把 goal relabeling(与 Hindsight Experience Replay 思路一脉相承)从"只能重标注到已访问状态"的图像目标场景,巧妙扩展成可以纳入任意异构上下文类型(语言、任务 ID 等)的通用 Multicontext Imitation Learning 框架,并证明只需给不到 1% 的无标签 play 数据配语言标注,就足以让语言条件性能追平乃至部分超过图像目标条件基线。这为"语言标注昂贵、演示数据廉价"这一现实约束提供了一条具体、简洁、纯监督学习范式内即可实现的解法,不依赖任何强化学习或复杂的奖励设计。将预训练句子嵌入(MUSE)简单接入策略语言分支即可获得对同义词与跨语言指令的鲁棒性,这一"表征迁移"思路后来在更大规模的语言条件机器人学习工作(如以 CLIP/大语言模型embedding 为语言编码器的后续 VLA 路线)中被广泛沿用,可以视为其重要先声。

**与其他公开工作的关系：** 本文的控制骨干(Latent Motor Plans)与 play 数据采集/重标注协议直接继承自同一作者团队的前作 Learning Latent Plans from Play(Lynch et al., CoRL 2019),核心增量在于语言条件与 MCIL 训练范式;其"重标注 + 目标条件"的思想谱系上承 Hindsight Experience Replay(Andrychowicz et al., 2017)的目标重标注技巧,并将其从强化学习迁移到模仿学习设定。与同期依赖标注任务演示、预训练目标检测器的指令跟随工作(论文脚注中提及的 [44])相比,本文完全端到端从像素学习感知、语言理解与控制,不依赖任何预训练视觉检测组件,这一点在当时颇具特色。

**开放问题与可能的改进方向：** 一是模型是否能扩展到多房间、多物体、真实机器人的更大规模场景,论文自己也把这列为未解问题;二是模仿学习与强化学习的结合(自主改进)在多大程度上能弥补"策略卡住却无法自我纠错"的缺陷;三是预训练语言模型目前完全冻结、不参与机器人数据的联合训练,若允许语言编码器随控制目标一起微调(即让语言表征本身也从具身交互中学习),是否能进一步提升 grounding 质量,是论文明确留白的方向,也呼应了后续大量"视觉-语言-动作"基础模型将语言/视觉编码器纳入联合预训练的发展路线;四是评测协议方面,论文提出的"neutral 初始化"揭示了以往指令跟随/目标条件评测中普遍存在的"初始状态泄露任务信息"问题,这一评测方法论上的贡献值得后续基准建设借鉴。

## 参考

- Lynch, C., Khansari, M., Xiao, T., Kumar, V., Tompson, J., Levine, S., Sermanet, P. Learning Latent Plans from Play. CoRL 2019.（arXiv:1903.01973，本文控制架构 LMP 与 play 数据重标注协议的直接前作）
- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., Zaremba, W. Hindsight Experience Replay. NeurIPS 2017.（目标重标注思想的源头之一）
- Gupta, A., Kumar, V., Lynch, C., Levine, S., Hausman, K. Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning. CoRL 2019.（论文在"未来工作"中提到的 LfP+RL 结合范例）
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018.（arXiv:1810.04805，激发本文"迁移预训练语言表征"思路的代表工作）
- Yang, Y., Cer, D., Ahmad, A., Guo, M., Law, J., Constant, N., Hernandez Abrego, G., Yuan, S., Tar, C., Sung, Y.-H. et al. Multilingual Universal Sentence Encoder for Semantic Retrieval. 2019.（arXiv:1907.04307，TransferLangLfP 所用的预训练多语言句子编码器 MUSE）
