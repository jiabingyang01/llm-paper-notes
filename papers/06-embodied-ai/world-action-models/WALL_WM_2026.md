# WALL-WM：在事件关节处雕刻世界动作建模

> **论文**：*WALL-WM: Carving World Action Modeling at the Event Joints*
>
> **作者**：X Square Robot Team（核心贡献者 Shalfun Li、Victor Yao、Charles Yang、Truth Qu、Regis Cheng、Ryan Yu、Howard Lu、Newton Von、Vincent Chen 等；项目负责人 Hao Wang；通讯作者 Qian Wang）
>
> **机构**：X Square Robot
>
> **发布时间**：2026 年 06 月（arXiv 2606.01955）
>
> **发表状态**：未录用（预印本），代码开源于 github.com/X-Square-Robot/wall-x
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.01955) | [PDF](https://arxiv.org/pdf/2606.01955)
>
> **分类标签**：`World Action Model` `事件中心预训练` `VLA` `多视角视频生成` `Latent CoT`

---

## 一句话总结

WALL-WM 把 VLA/世界动作模型预训练的原子学习单元从"定长 action chunk"替换成"语义完整的动作事件"（action-grounded semantic event），用同一套事件对齐的多视角视频-动作去噪器同时支持变长的 event 模式推理和定长 chunk 的 unified 模式推理（后者靠 Staircase Decoding 提供并行隐式 CoT），并配套事件中心数据生态与 Muon/蒸馏/FP8 系统工程；在四个真机评测套件上，事件模式相对 π0.5、DreamZero、LingBot-VA 等基线取得最强综合 Task Progress（如 Diverse Manipulation 套件 75.86 分 vs. 55.64/39.97/29.71）。

## 一、问题与动机

现有 VLA/世界动作模型普遍在当前观测和指令条件下预测**定长** action chunk，方便训练和部署，但作者指出这掩盖了一个结构性错配：language 描述的是粗粒度目标和事件、vision 按连续场景动态演化、action 在控制级时间尺度上运行——三者的语义/时间粒度天生不同，被硬塞进同一个固定窗口后，VLA 训练退化为"短时程相关性拟合"：既没有充分利用预训练视觉-语义先验，又会用 chunk 特有的动作捷径覆写这个先验，削弱组合泛化和长程泛化。

论文把 video 视为连接 language 与 action 的天然桥梁：互联网规模的视频预训练捕获了单靠具身交互难以学到的丰富视觉动态，同时在语义上与 language 一样，在事件边界处结构化（caption-to-video 的隐式关联），而在时间上又比纯文本稠密得多，足以暴露 action 执行所需的 timing、transition、contact 变化。因此把 video 基础模型"提升"（lift）为 WAM 不是一次简单的微调，而是一次**保留先验的迁移**：既要保留 video 生成模型偏好的语义不变性、视觉合理性、时间平滑性，又要提供**时间接地**（temporal grounding）能力——language 指令描述全局任务或语义事件，而观测与动作要以帧、以控制步展开。

由此论文提出三条设计原则，用来判断"对齐单元"该长什么样：

- **几何保持**（geometry preservation）：连接 language、video、action，但不把它们的原生结构压扁进同一个共享空间；
- **先验保持**（prior preservation）：与视频基础模型继承的 caption-to-video 结构兼容；
- **可执行因果性**（executable causality）：预测目标要有清晰的时间支持，但其 duration 应随任务本身变化，而非跟随外部固定时钟。

这三条原则排除了定长 action chunk 作为基本单元（它可能切在一个语义行为中间、把多个行为合并进一个目标、或需要历史上下文才能确定自己指代什么），WALL-WM 用 **action-grounded semantic event** 取代它：一段可以用语言表达（reach、grasp、lift、move、place 等）、在视频中观测、并通过动作实现的、时序连贯的可执行行为片段。事件的起止随底层可执行行为的变化而变化，而不是随外部时钟。

## 二、核心方法

### 2.1 整体建模

WALL-WM 建模

$$p_\theta(\mathbf{V}_e, \mathbf{a}_e \mid \mathbf{V}_0, \mathbf{s}, c_e)$$

其中 $\mathbf{V}_0$ 是当前多视角观测（每相机一个关键帧），$\mathbf{s}$ 是当前本体感受状态，$(\mathbf{V}_e,\mathbf{a}_e)$ 是与事件对齐、长度随事件而定的未来多视角视频和末端轨迹，$c_e$ 是描述该动作事件的 per-event caption。架构上，video tower 继承自 Wan 系列 text-to-video 模型，action tower 是随机初始化的 action DiT，两者按层耦合（layer-coupled），跨模态对齐通过逐层的 video→action cross-attention 学习，而不是把三种模态压进一个共享 embedding 空间。

### 2.2 多视角视觉世界建模（video tower）

在 Wan 单视角 DiT 基础上嫁接三个组件：

**(i) 多视角跨视角自注意力**：每个 DiT block 在原有 within-view self-attention 之外，额外跑一个 cross-view self-attention 分支，输出经零初始化投影后通过 AdaLN 门控加回主干：

$$\mathbf{h}_i^V \leftarrow \mathbf{h}_i^V + g_i\, W_{\text{view}}\,\text{CrossViewAttn}_i(\mathbf{h}_i^V), \qquad W_{\text{view}}\text{ 初始化为 } 0$$

用大白话说：跨视角分支训练一开始"什么也不做"（权重为零），训练中逐渐学会怎么融合多相机信息，预训练的单视角外观与语言对齐能力被原样保留，不会被跨视角训练冲掉。

**(ii) Camera RoPE**：给每个视角一个可学习的、无需外参标定的旋转身份编码，让同一个 DiT 能跨异构多具身相机配置工作，增删相机只需改嵌入表。

**(iii) 跨视角几何掩码**：两套互补的几何感知 mask，训练时使用、推理时去掉（保持 rollout 对标定无依赖）。sight-cone mask 基于相机视锥的 3D 相交关系判断两个 patch 是否"共视"，不共视的 token 对在注意力里加 $(1-\mathcal{M}_{\text{sc}})\cdot(-\infty)$ 的偏置，禁止跨视角路由到几何上不可能相关的位置；tube patch masking 则以概率 $p_{\text{tube}}$ 随机遮住某个视角一个 $k\times k$ 的时空管道，迫使模型只能从其它视角重建被遮内容，人为制造"必须做跨视角推理"的监督信号。

video 侧的 flow-matching 目标（Wan 风格 v-prediction）为

$$\mathcal{L}_V = w_V(t^V)\Big(\sum_{u\notin\mathcal{T}}\|\hat{C}_u^V-C_u^{V\star}\|^2 + \lambda_{\text{mask}}\sum_{u\in\mathcal{T}}\|\hat{C}_u^V-C_u^{V\star}\|^2\Big)$$

其中 $\mathcal{T}$ 是被 tube masking 遮住的 token 集合。用大白话说：被遮住、必须靠跨视角信息才能重建的那些 patch，损失权重被额外放大（$\lambda_{\text{mask}}$），逼着模型认真做跨视角推理而不是绕过去。border masking（排除画面外区域和合成黑边）始终生效，是主配方里唯一常开的正则。

### 2.3 事件中心动作动力学建模（action tower）

action tower 是与 video tower 同深度的 action DiT，每层读取配对 video block 的多视角特征做 cross-attention，再用 flow matching 去噪末端轨迹。层间耦合是单向的：

$$\tilde{\mathbf{h}}_i^V = \text{ViewConcat}\big(\mathbf{h}_{\pi(i)}^V\big) + E_\tau(\tau^V) + E_{\text{abs}}(t_{\text{abs}}), \qquad \pi(i)\text{ 把 action block }i\text{ 映射到配对的 video block}$$

用大白话说：action 每一层"偷看"对应深度 video 层算出来的多视角特征当 KV，只单向流动（video 塔不会被 action 反向影响），两个可学习的时间编码负责把 action query 和 video key/value 对齐到同一个时间坐标系。论文定义了两种时间窗口：**事件中心窗口**（预训练用，每个事件独立编号帧索引）和**观测中心窗口**（unified 模式微调/部署用，带 $M$ 帧历史 $+1$ 观测锚点 $+N$ 帧未来的滑动窗口，历史帮助消解全局指令在时间上的歧义）。

两塔各自有独立的 flow-matching 去噪 schedule，需要显式指定 action step 读哪个 video denoising step 的特征。默认用**非对称 1-to-$N_d$ 映射**：固定一个 schedule anchor $s^\star=45$（在 50 步 schedule 上选出的中噪点），video 塔每个 optimizer step 只跑一次前向、广播给全部 $N_d$ 个 action step 复用；video 塔冻结，只训练 action 塔——这是大规模训练的默认配方（相对的是 symmetric 1-to-1，仅用于小数据端到端联调）。action 目标为

$$\mathcal{L}_A = \frac{1}{K}\sum_{k=1}^K w(t_k^A)\,\big\|\hat{\mathbf{y}}_k^A - \mathbf{y}_k^{A\star}\big\|^2$$

默认用 v-prediction，可切换为 x-prediction（直接输出干净动作）应对 contact-heavy 场景，并可加一个按噪声水平指数衰减加权的 Type-II DCT 辅助损失，用于抑制帧间抖动、强调整体运动形状。

### 2.4 Staircase Latent CoT（语言引导推理）

推理分支从 Qwen3.5-9B 初始化，以 Mixture-of-Transformers（MoT）方式挂在冻结的 WAM 主干上，产生 $K_c$ 个连续隐推理状态而非逐词自回归生成文本 token：

$$\hat{y}_{1:K_c} = \mathcal{F}_{\text{stair}}(x; N_r)$$

传统 latent CoT 逐步串行生成，每步都要重跑一遍底层视觉-语言特征，长链推理延迟大。WALL-WM 在 relay depth $N_r$ 处把 Transformer 切开：只有第一个 latent 位置走完下层，产生一个所有推理位置共享的 relay 表征；上层再为每个推理位置独立、并行地计算。用大白话说：像接力赛，前半程（视觉-语言 grounding 的公共部分）只跑一次，后半程按需要的推理步数分道并行冲刺，避免每步都重复计算底层特征。监督方式是 frozen latent-to-text reconstruction：隐状态经 prefix projector 投影成 soft prefix，喂给一个冻结的小语言模型（Qwen3.5-0.8B）自回归重建对应文本 CoT trace，只训练 staircase 分支和 projector：

$$\mathcal{L}_{\text{CoT}} = -\sum_{m=1}^{M_r} \log P_\phi\big(r_m \mid \mathbf{z}_{1:K_c}, r_{<m}\big)$$

这样隐状态被鼓励编码高层推理语义，而非精确复刻 token 级解码轨迹。

### 2.5 两种推理模式

- **Event 模式**：VLM/人类/agent 每步只给出下一个事件的语言描述，WALL-WM 执行对应的变长 video-action 片段，rollout 随任务自然节奏推进，不受固定 horizon 约束。
- **Unified 模式**：保留传统定长 chunk 推理，但 chunk 不再直接条件在原始 global instruction 上，而是由 VLM + Staircase Decoding 生成事件结构化的 latent 推理表示来指导下一个局部 chunk，同时保持一条 gradient-continuous 的 VLA 路径。三种文本侧来源（原始全局指令的连续编码、每 chunk 一条的 atomic instruction、VLM-CoT latent）可以在同一个 denoiser 上互换而无需重新训练。

### 2.6 数据与工程（简述）

数据生态覆盖四象限——web 规模互联网视频（含 1.2M-clip OpenVID 切片）、egocentric 人类视频（Ego4D、EPIC-KITCHENS）、robot-free UMI 式无具身采集（自研可穿戴 rig XRZero-G0）、异构机器人遥操作/开放数据集（DROID、AgiBot World 及自采）——中心是 human-intervention 与 failure-recovery 数据。自研部署平台涵盖桌面双臂、两种移动机器人平台（QUANTA X1/X1 Pro）和轮式人形机器人 QUANTA X2。关键后处理包括：基于光流运动信号与动作差分信号做互相关的**时序同步**（修正 video-action 固定 lag）；Task/Subtask/Action/Segment 四级事件中心分层 caption（边界锚定在 atomic manipulation action 上）；VL 聚类 + action 聚类驱动的 cluster-balanced sampling 应对长尾分布；针对 contact-rich 事件、围绕 nominal contact pose 局部扰动采样的 recovery data 增强。此外还有一个长度感知的 caption-drop schedule，按事件跨度 $L_e$ 用 cosine 插值调整训练时丢弃 caption 的概率 $\rho(L_e)$（从 0.1 到 0.9），让短事件更多做无 caption 的纯视觉续写、长事件更依赖语义指导。

系统工程侧包括分布式 Muon 优化器实现 DMuon（pipeline scheduling + CuteDSL kernel 把优化器开销降为次要成本）、自研 kernel 库、multi-event sequence packing（把多个变长事件打包进定长序列训练）；部署压缩用 distribution-matching distillation（DMD，配合 joint distillation 目标同时保 action 监督——消融显示去掉这一保护会让 action MAE 退化 53%）叠加 FP8 per-block 量化，配合 CUDA Graph 把端到端推理做到 **10Hz**。

## 三、实验结果

**具身视频生成（WorldArena 协议，对比 Wan2.1-1.3B / Wan2.2-5B 基座）：**

| 指标类别 | 具体指标 | Wan2.1-1.3B | Wan2.2-5B | WALL-WM |
|---|---|---|---|---|
| Visual Quality | Image Quality | **0.577** | 0.527 | 0.503 |
| | Aesthetic Quality | 0.389 | **0.409** | 0.393 |
| Motion Quality | Dynamic Degree | 0.199 | 0.418 | **0.484** |
| | Flow Score | 0.061 | 0.109 | **0.148** |
| | Motion Smoothness | 0.619 | 0.683 | **0.771** |
| Semantic Consistency | Subject Consist. | 0.476 | 0.769 | **0.795** |
| | Background Consist. | 0.522 | 0.817 | **0.838** |
| | Semantic Alignment | 0.857 | 0.805 | **0.886** |
| Physical Plausibility | Interaction Quality | 0.219 | 0.226 | **0.434** |
| | Perspective | 0.819 | 0.807 | **0.821** |
| | Instruction Following | 0.308 | 0.298 | **0.391** |
| | Trajectory Acc. | 0.214 | 0.223 | **0.234** |

WALL-WM 在 Motion Quality、Semantic Consistency、Physical Plausibility 三大类几乎全面领先，仅在纯感知向的 Visual Quality（Image/Aesthetic Quality）上略低于 Wan 基座，说明大规模事件中心具身训练把通用视频先验转化成了更强的物理/交互先验，但轻微牺牲了一点纯粹的图像美学质量。

**3D 感知（CO3Dv2 探针）：**

| 模型 | Point Err ↓ | Depth Err ↓ | AUC@5 ↑ | AUC@30 ↑ |
|---|---|---|---|---|
| DINOv2 | 0.559 | 0.209 | 0.051 | 0.508 |
| V-JEPA | 0.439 | 0.214 | 0.076 | 0.619 |
| CogVideoX | 0.485 | 0.231 | 0.051 | 0.569 |
| Aether | 0.501 | 0.249 | 0.054 | 0.571 |
| Open-Sora2.0 | 0.391 | 0.196 | 0.096 | 0.643 |
| WAN2.1-14B | 0.284 | 0.151 | 0.200 | **0.736** |
| WALL-WM | **0.271** | **0.132** | **0.210** | 0.727 |

WALL-WM 在三项指标上最优，仅 AUC@30 略低于参数量大得多的 WAN2.1-14B。

**真机四套件 Task Progress（0–100 分制，均值）：**

| 套件 | π0.5 | LingBot-VA | DreamZero | WALL-WM-U-Scratch | WALL-WM（Event 模式） |
|---|---|---|---|---|---|
| Diverse Manipulation | 55.64 | 29.71 | 39.97 | 63.00 | **75.86** |
| Reasoning Manipulation | 56.40 | 31.60 | 32.70 | 59.50 | **71.60** |
| Dexterous Manipulation | 15.00 | 24.00 | 25.00 | 31.25 | **32.00** |
| Generalization | 24.00 | – | 28.50 | 18.50 | **53.75** |

WALL-WM-U-Scratch 是去掉事件中心预训练、直接在观测中心定长 chunk 上从头训练的消融基线（不是 §5.5.2 的 unified 推理模式，而是专门测"预训练带来多少可迁移先验"）。四套件里 event 模式全面领先，其中 Generalization（多物体共存场景下按随机顺序切换指令）提升最大（53.75 vs. 18.50），Dexterous Manipulation（高精度插拔类任务）提升最小（32.00 vs. 31.25），提示事件级预训练更多帮助的是"任务/子任务切换与语义定位"，而非最底层的接触控制精度。

**事件条件执行 + 跨视角建模消融（Table 4，去掉 View-Interaction Self-Attention 与事件条件执行的预训练基线 vs. 完整 event 模式）：** Reasoning Manipulation 均分从 32.6 升到 71.6，其中 `Press Button in Order` 从 0 分升到 64 分；Generalization 均分从 22.0 升到 53.75。说明单纯的预训练表示不足以支撑时序排序、关系匹配一类任务，需要跨视角交互和事件条件执行的组合才能兑现。

## 四、局限性

- **事件边界仍依赖密集标注**：论文在 Future Work 中明确承认，当前数据构造仍依赖大规模时序对齐和细粒度 caption 来"暴露"事件结构，而非从视觉-语言-动作信号中自监督发现事件边界，这限制了流程向更大规模互联网视频的无缝扩展。
- **KV-cache 流式方案只能部分缓解时序对齐问题**：Appendix 9.4 指出 DreamZero、LingBot-VA 等采用 KV-cache 流式 rollout 来缓解 V-A 时间对齐问题，但这类系统（包括未特别说明的部分场景）仍以**固定时长**推理——生成预定帧数后停止，而不是在指令语义的执行端点处自动终止，修复是"partial"的。
- **精细接触控制收益有限**：Dexterous Manipulation 套件上 event 模式相对 WALL-WM-U-Scratch 的提升很小（32.00 vs. 31.25），论文自己指出这类任务更受限于底层位姿精度、contact timing、窄容差对齐，而非高层事件分解——说明事件中心预训练对最细粒度力控/插拔类任务的帮助有限。
- **并非所有任务都受益于显式事件分解**：Generalization 套件里 `Cover Pot with Lid` 一项，WALL-WM-U-Scratch（32）反而优于 event 模式（26），提示某些视觉直接的运动可以仅靠任务级监督学到，显式事件分解不是普适增益。
- **评测环境对齐优势**：论文在 Discussion 中主动指出，真机评测在其内部自研具身平台上进行，WALL-WM 的大规模预训练数据本身就是从该平台采集或对齐的，这给自身系统带来了基线无法分享的"评测-训练环境一致性"优势；π0.5/DreamZero/LingBot-VA 等基线只能通过各自标准动作接口适配，未获得针对该平台的专门调优，比较并非完全同条件。
- **技术报告性质**：45 页内容以 X Square Robot 内部技术报告形式发布，未经同行评审，缺乏第三方复现；跨越"10B 以下到数十亿-数百亿"参数规模的 scaling 观察停留在定性描述（"consistent trend"），未给出完整的量化 scaling 曲线或消融。
- **部署侧仍需重度压缩**：基础（未压缩）WAM 每步推理仍是数十步去噪跑一个数十亿参数网络，10Hz 闭环控制要依赖蒸馏（DMD）+ FP8 量化 + CUDA Graph 的组合工程手段才能达到，对研究复现和小团队落地的门槛较高。

## 五、评价与展望

**优点**：论文对"粒度错配"问题的诊断清晰，并给出了架构-数据-系统三位一体的系统性解法，而非仅一个新模块或新损失；三条设计原则（几何保持/先验保持/可执行因果性）提供了一个可迁移到其他 WAM 设计的抽象框架，有一定的方法论价值。Staircase latent decoding 对 latent CoT（如 LaDiR、LaST₀ 一类工作）给出了一个效率导向的结构改进：用 relay depth 分离共享底层 grounding 与并行独立推理，理论上能降低长链隐推理的串行延迟，这一思路对其他需要 latent CoT 的具身/多模态模型有一定借鉴价值。跨视角几何掩码（sight-cone mask + tube patch masking）用几何先验去指导/强制跨视角 attention 学习"该学什么、不该学什么"，相比直接拼接多视角 token 训练更具物理可解释性，也是论文对 3D-aware 表征学习的一个具体贡献（Table 3 的 3D 探针结果支持这一设计）。四套件真机对比与两组消融（U-Scratch 去事件预训练、Table 4 去 VI-SA）相对扎实地支持了"预训练表示单独不够，需要跨视角交互 + 事件条件执行组合"这一核心论点。

**局限与开放问题**：如前节所述，评测环境与基线适配的不对等是这类内部平台自评估论文的共性局限，读者应谨慎解读跨系统对比的绝对分数差距。事件边界目前仍由密集人工/半自动 caption 标注驱动，如何做到自监督的事件发现（论文自己列为 Future Work）是决定这条路线能否规模化到互联网视频体量的关键开放问题。与同类"统一视频-动作扩散"工作（Related Work 中列出的 LaDi-WM、AdaWorld、Motus、LDA-1B、MotuBrain 等）相比，WALL-WM 的差异化卖点是"事件"这一原子单位的选择本身，而不是某个具体网络模块；这与近期 DreamZero（"world action models are zero-shot policies"）、LingBot-VA 等同样试图解决 fixed-horizon/时序错配问题的工作在动机上高度相近，但具体解法（事件 caption 驱动的变长执行 vs. KV-cache 流式定长 rollout）不同，Appendix 9.4 也做了直接对比讨论，这构成一个值得后续工作系统比较的设计空间。Dexterous Manipulation 上的边际改善提示，事件中心预训练的收益可能主要来自"何时切换动作阶段"这类中层决策，而非最底层的力控/精细对齐，未来或许需要专门的接触级建模（如论文 Fig.1 提及但未展开使用的 tactile-force 模态）来补足这一维度。

## 参考

1. **π0.5**（Physical Intelligence et al., 2025）——真机评测中的核心 VLA 基线之一。
2. **DreamZero**（Ye et al., arXiv:2602.01705, 2026）——"world action models are zero-shot policies"，KV-cache 流式 V-A 对齐的直接比较对象。
3. **LingBot-VA**（Li et al., arXiv:2601.21998, 2026）——causal world modeling for robot control，统一 video-action 去噪架构的比较对象。
4. **LaDiR**（Kang et al., arXiv:2510.04573, 2025）——latent diffusion reasoning，Staircase latent CoT 对比的 latent CoT 基线之一。
5. **Wan**（Team Wan et al., arXiv:2503.20314, 2025）——WALL-WM video tower 所继承的视频基础模型。
