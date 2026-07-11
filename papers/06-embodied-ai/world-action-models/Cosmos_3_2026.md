# Cosmos 3：面向 Physical AI 的全模态世界模型

> **论文**：*Cosmos 3: Omnimodal World Models for Physical AI*
>
> **作者**：Ming-Yu Liu（Supervision）et al.（NVIDIA Cosmos 团队，贡献者名单按组别字母序列于附录 G，无单一通讯作者标注）
>
> **机构**：NVIDIA
>
> **发布时间**：2026 年 06 月（arXiv 2606.02800）
>
> **发表状态**：未录用（预印本，NVIDIA 技术报告性质，随论文开源代码/权重/数据/评测基准）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.02800) | [PDF](https://arxiv.org/pdf/2606.02800)
>
> **分类标签**：`世界动作模型` `全模态大模型` `Mixture-of-Transformers` `具身操作策略` `视频生成`

---

## 一句话总结

Cosmos 3 用双塔 Mixture-of-Transformers（自回归推理塔 + 扩散生成塔，共享自注意力单向交互）把语言、图像、视频、音频、动作五种模态装进同一 token 空间，一个模型不改架构即可切换扮演 VLM、图文/图生视频生成器、control-conditioned 视频转换器、以及前向动力学/逆向动力学/策略三种动作生成模式；Edge/Nano/Super 三档模型分别为 4B/16B/64B 参数，Super 用 2048 张 GB200 训练 17.86T token，后训练版本在 Artificial Analysis 文生图与图生视频榜单均排名开源模型第一，机器人策略模型 Cosmos3-Nano-Policy-DROID 同时拿下 RoboArena（真机众包评测）、RoboLab-120（仿真）、MolmoSpaces（仿真）三个第三方榜单第一名。

## 一、问题与动机

当前 Physical AI 系统把"理解"（VLM 做定位/规划）、"生成"（视频生成器/世界模拟器做未来预测）、"动作预测"（VLA/WAM 做控制信号生成）当成三个独立范式来分别建模，导致一个具身智能体（例如"清理餐桌"）必须拼接三套模型：VLM 定位餐具并生成计划、VLA/WAM 生成动作序列、前向动力学模型/世界模型模拟并评估未来状态。这种碎片化架构既不优雅也浪费计算。作者认为理解与生成本质上是耦合的——理解需要推理世界的未来演化及动作的后果，生成则依赖对世界状态和智能体行为的紧凑结构化表示——因此提出把动作提升为与语言、图像、视频、音频并列的一等模态，用单一网络原生覆盖 VLM、文生图、文/图生视频、音视频联合生成、control-conditioned 视频转换、前向动力学、逆向动力学、机器人策略共 8 类模型职能。

## 二、核心方法

### 2.1 模态编码器与统一动作表征

视觉编码分理解/生成两路：理解侧用 ViT（16×16 patch，仿 Qwen3-VL 用 DeepStack 聚合多层特征，与主干联合训练）；生成侧复用 Wan2.2-TI2V-5B 的视频 VAE（时间 4× 压缩、空间 16×16 压缩 + 2×2 patch merge，生成阶段冻结）。音频用 48kHz 立体声 VAE（hop=1920 samples，25 token/秒，冻结）。

动作表征是本文的关键设计：把跨具身（自动驾驶、相机运动、第一视角人手、单/双臂机器人、人形机器人）的原生控制信号统一映射为最多三段的紧凑向量——ego pose 9D（3D 平移 + 6D 旋转，Zhou et al. 2019 的过参数化旋转表示）、effector pose 9D、grasp state（机器人为 1D 开合值，人手为 15D 五指位置）。ego/effector 位姿用相邻帧 SE(3) 相对变换编码为伪动作：

$$
\Delta \mathbf{T}_t = \mathbf{T}_{t-1}^{-1}\mathbf{T}_t
$$

grasp state 则直接编码瞬时状态而非帧间差分。动作 token 化采用 domain-aware 的输入/输出线性投影（每个具身域独立权重，共享同一 MoT 主干）：

$$
\mathbf{z} = \mathbf{W}_{\text{in}}^{(k)}\mathbf{x} + \mathbf{b}_{\text{in}}^{(k)}
$$

解码时用对称的 $\mathbf{W}_{\text{out}}^{(k)}$ 投影回动作空间，6D 旋转再经 SVD 转回 3×3 SO(3) 旋转矩阵。用大白话说：不同机器人/相机/人手的"控制指令"格式五花八门（关节角、转向量、腕部 6D 位姿……），这一步先把它们都"翻译"成同一种语言的动作向量再喂进 Transformer，只在输入输出的"翻译层"保留各具身私有参数，主干完全共享。

### 2.2 Token 排布与七种生成模式

输入序列固定分两段：自回归（AR）子序列（语言 token + ViT 视觉 token，负责理解）在前，扩散（DM）子序列（VAE 视觉/音频/动作 token，负责生成）在后。依据两段的拼接与 clean/noisy 排布差异，同一模型可无缝切出 Language、Text-to-Image、Text-to-Video(+Audio)、Image/Video-to-Video(+Audio)、Video Transfer（控制视频 → RGB 视频）、Action 六大类任务，其中 Action 又细分前向动力学（FD，给定动作生成未来视觉状态）、逆向动力学（ID，给定视觉转变推断动作）、策略（Policy，联合生成动作与其视觉后果）三种模式，仅靠"谁 clean 谁 noisy、谁在前谁在后"的排布差异实现，不需要任何架构改动。

### 2.3 Mixture-of-Transformers（MoT）主干

每层 Transformer 含两套独立参数（各自的 LayerNorm、MLP、注意力投影），分别处理 AR 和 DM token，二者只在共享自注意力算子里交互。AR 部分用因果自注意力：

$$
\mathbf{O}_{\text{AR}} = \text{Attn}_{\text{causal}}(\mathbf{Q}_{\text{AR}}, \mathbf{K}_{\text{AR}}, \mathbf{V}_{\text{AR}})
$$

DM 部分用双向全注意力，可同时看到 AR、DM 两段的 key/value：

$$
\mathbf{O}_{\text{DM}} = \text{Attn}_{\text{full}}\!\big(\mathbf{Q}_{\text{DM}},\, [\mathbf{K}_{\text{AR}};\mathbf{K}_{\text{DM}}],\, [\mathbf{V}_{\text{AR}};\mathbf{V}_{\text{DM}}]\big)
$$

且 AR token 永远不会被 DM token 更新（保持推理塔的因果封闭性）。两塔权重均从同一个预训练 VLM 初始化，继承强语言/视觉推理能力后再学习生成。用大白话说：一层楼里有两套"独立办公室"（推理组/生成组），各自按自己的规则处理文件，但生成组开会时可以随时翻阅推理组已写好的会议记录（单向可见），推理组则完全不知道生成组在做什么——这样既保证语言生成的因果性不被打乱，又让扩散生成随时能"看懂"文本/图像上下文。

### 2.4 多模态位置编码：3D MRoPE + 绝对时间调制

在 Qwen3-VL 的 3D MRoPE 基础上扩展：视频/音频/动作 token 各自沿时间轴前进（音频按 hop 步进、动作按采样步进，空间坐标固定为 0），并引入"每秒时间步数"（TPS）概念对齐不同采样率——视频 TPS = 帧率 / 4（VAE 时间压缩比），音频 TPS = 48000/1920 ≈ 25，动作 TPS 即采样频率；以 24FPS 视频为基准 $\text{TPS}_{\text{base}}=6$，位置增量换算为

$$
\delta t = \frac{\text{TPS}_{\text{base}}}{\text{TPS}}
$$

此外在 AR 与 DM 子序列之间人为插入固定时间间隔（取值 15000），避免最后一个语言 token 与首帧视觉 token 位置过近导致的生成过饱和/棋盘伪影（该问题在参数量更大的 Super 模型上尤为明显）。用大白话说：视频可能 30fps、音频恒为 25Hz、动作可能 15Hz，如果直接按 token 序号编号，不同模态的"1 秒"长度完全不一样；这里给每种模态按真实物理秒数重新换算刻度，让所有模态共享同一条时间轴。

### 2.5 训练数据与算力规模

Reasoner 两阶段训练：预训练（22.0M 样本，源自 Nemotron-Nano2 数据子集 + 补充数据，经语义去重 [K-means 聚类 + 余弦相似度阈值 0.95] 与 Gemma-4-31B AI 裁判质量过滤两级流水线）+ SFT（2.2M 样本，专注机器人/自动驾驶/智慧基建三大物理 AI 领域，视频-文本占比提升到 50%）。

Generator 用统一 Rectified Flow 匹配目标，对图像/视频/音频/动作各自独立采样噪声水平（logit-normal + mode sampling）；预训练阶段用 767M 图像 + 347.7M 视频片段（源自 78 亿原始图像、30 亿原始视频），多分辨率（256p/480p/720p）+ 可变长度序列打包（固定 7.4 万 token 预算/序列）联合训练。中训练（mid-training）阶段引入动作与 transfer 模态：动作数据规模为 840 万 episode、6.13 万小时，四大来源——第一视角人手操作（67.4%，含 170 万 episode 双手操作、头戴相机采集的专有数据）、自动驾驶（16.3%，NVIDIA Hyperion 平台采集）、机器人（8.7%，90.4K 任务 / 51.67 万 episode，覆盖 AgiBot / Franka Panda / Google Robot / WidowX-250 / UMI / UR 六种具身）、相机运动（7.5%，从预训练视频中用 ViPE + DepthAnything3 挖出的位姿轨迹）。

算力：Cosmos3-Nano 预训练用 1024 张 GB200 训练 31.05T token，Cosmos3-Super 用 2048 张 GB200 训练 17.86T token；中训练阶段 Nano 再耗 2.4T token、Super 再耗 1.9T token。三档模型：Edge（4B，从零训练 28 层 2B 稠密 Transformer）、Nano（16B，初始化自 Qwen3-VL-8B）、Super（64B，初始化自 Qwen3-VL-32B），本文开源 Nano/Super，Edge 留待后续发布。

后训练三条线均在共享架构上进行：Text-to-Image（→ Cosmos3-Super-Text2Image）、Image-to-Video（→ Cosmos3-Super-Image2Video）、机器人策略（在 DROID 数据集——76k 轨迹 / 350 小时 / 86 任务 / 564 场景——上后训练 → Cosmos3-Nano-Policy-DROID，预测 32 步未来关节位置动作、15Hz 闭环执行，推理时用 4 步扩散采样 + CFG parallelism 压缩延迟，可部署在 2 张 RTX Pro 6000 上）。

## 三、实验结果

**Reasoner（理解侧，48 个 benchmark 汇总）**

| 维度 | Cosmos3-Super | Cosmos3-Nano | Gemini 3.1 Pro（闭源） | Qwen3-VL-32B |
|---|---|---|---|---|
| General（19 项） | 73.7 | 69.6 | 77.5 | 72.8 |
| Robotics（17 项） | 57.8 | 55.1 | 58.2 | 52.6 |
| Smart Infra.（9 项） | 62.6 | 61.0 | 58.6 | 56.1 |
| Driving（3 项） | 79.3 | 76.0 | 47.2 | 40.7 |

通用能力上仍落后闭源 Gemini 3.1 Pro，但机器人/智慧基建/自动驾驶三个物理 AI 垂类上与 Gemini 基本持平甚至大幅超越（驾驶领域领先约 32 分）。

**Generator（生成侧，Table 1 核心数字）**

| 能力 | Cosmos3-Super | Cosmos3-Nano | 最强对比基线 |
|---|---|---|---|
| Text2Image（UniGenBench All） | 91.36（后训练 Text2Image 变体） | 84.61 | Gemini 3 Pro Image 90.69 |
| Text2Video（PAIBench-G Overall） | 80.0 | 79.4 | Veo-3.1（闭源）79.1 |
| Image2Video（PAIBench-G Overall） | 82.8 | 82.7 | Veo-3.1（闭源）82.6 |
| Audio（SoundBench AVQ） | 7.31 | 7.34 | Seedance-1.5-Pro（闭源）7.64 |
| FD:Robot（DROID, PSNR dB） | 26.0（MT-init 后训练） | 25.5 | Ctrl-World 23.0 |
| Policy:Robot（RoboLab success） | — | 39.7（后训练） | π₀.₅ 28.1 |

第三方榜单交叉验证：Cosmos3-Super-Text2Image 在 Artificial Analysis 文生图榜单排开源模型第一（全部含闭源第四）；Cosmos3-Super-Image2Video 在图生视频榜单（无音频）排开源第一（全部第 22）；Human World Bench（第一视角人手操作 I2V 人评）上 Cosmos3-Super 以 71.9 分成为**所有模型（含闭源）中的最高分**，超过闭源 Veo-3.1 的 67.8 分；Physics-IQ 物理一致性评测中 Cosmos3-Super 直接采样即在 I2V（43.8）与 V2V（59.7）两种条件下都取得开源/闭源最佳，配合 WMReward+Best-of-N 重排后进一步提升到 48.9/63.4。

**动作生成 / 世界-动作模型关键结果（本文核心贡献：验证统一动作中训练带来可复用先验）**

论文设计了 PT-init（仅用预训练权重，未见过动作数据）vs MT-init（经过统一动作中训练）两种后训练起点的对照实验：

| 任务 | 指标 | PT-init 后训练 | MT-init 后训练 | 专用基线 |
|---|---|---|---|---|
| 自动驾驶逆动力学 | RRE°/RTE m/ATE m（Super） | 0.284/0.018/1.32 | 0.232/0.014/0.90 | VGGT 0.596/0.768/23.46 |
| 相机运动前向动力学 | RRE°/RTE m/ATE m（Super） | 0.293/0.036/1.82 | 0.142/0.026/0.99 | Lingbot-World 0.299/0.057/2.88 |
| 第一视角前向动力学 | PSNR dB（Super） | 15.34 | 16.19 | LOME 9.36 |
| 机器人（DROID）前向动力学 | PSNR dB（Super/Nano） | 22.69/23.24 | 26.04/25.52 | Ctrl-World 22.99 |
| LIBERO-10 新具身适应 | 闭环成功率 @500 迭代 | 0.0% | 24.6% | — |
| LIBERO-10 新具身适应 | 闭环成功率 @2000 迭代 | 95.2% | 97.4% | — |

机器人策略模型 Cosmos3-Nano-Policy-DROID 在 RoboLab-120（specific 指令粒度）上平均成功率 39.7%，超过 VLA 基线 π₀.₅（28.1%）与同为 world-action-model 定位的 DreamZero（约 25.2%）；在真机众包 A/B 评测 RoboArena 上排名第一（得分 1870，第二名 Spirit v1.6 为 1785）；在仿真基准 MolmoSpaces 上排名第一（oracle 成功率 39.0%，第二名 WALL-OSS-0.5 为 38.5%）。跨域协同实验（synergy matrix）显示动作域之间大多正迁移：相机运动前向动力学 PSNR 加入自动驾驶协同训练后从 11.96 升至 12.82（+0.86）；第一视角人手数据对 AgiBot 机器人适应有持续"热身"效应（相同步数下比直接从预训练初始化多约 1.3–1.6dB PSNR）。PushT 小规模消融（Cosmos3-Edge）显示联合训练 FD/ID/Policy 三种动作模式，在同等优化预算下比单独训练 ID 误差降低 72%（1.11×10⁻³→3.09×10⁻⁴）、策略覆盖率从 74.1% 升至 77.3%，仅 FD 重建质量小幅下降（27.13→26.22 PSNR）。

## 四、局限性

（论文正文没有独立的 Limitations 小节，以下为通读全文后的客观归纳）

- Reasoner 通用能力仍落后闭源 Gemini 3.1 Pro（73.7 vs 77.5），机器人子项也略逊（57.8 vs 58.2），说明"全模态统一"目前是以让出部分纯语言/视觉通用推理上限为代价换取物理 AI 垂类能力。
- 音频生成的感知质量（PQ）落后闭源 Seedance-1.5-Pro 与 Veo-3.1（6.28–6.32 vs 6.68–7.06），Cosmos 3 的优势集中在语义对齐/音画同步而非声音本身的保真度。
- Physics-IQ 等物理一致性指标的 SOTA 成绩相当依赖 WMReward + Best-of-N 推理时重排（如 V2V 从直接分 59.7 提升到 63.4），直接单次采样质量与加入推理时计算后的质量存在明显差距。
- 机器人策略后训练目前只在 DROID 单一数据集 / 单一 Franka Panda 具身上系统验证（论文原文自称为 "pilot study"），即便在指令最明确的 specific 粒度下成功率也只有 39.7%，vague 粒度下降到约 20%，离通用可靠操作仍有距离。
- 训练高度依赖 NVIDIA 内部专有数据（Hyperion 自动驾驶日志、170 万 episode 的专有双手第一视角操作数据集等）与数千卡 GB200、数十万亿 token 级算力，尽管开源了代码/权重/部分合成数据集，完整复现训练管线的门槛仍然很高。
- 多阶段（预训练→中训练→后训练）叠加多模态损失加权（如动作损失额外 ×10）与大量人工设定的超参数（如 AR-DM 时间间隔固定取 15000），系统工程复杂度高，虽消融充分但每个设计选择的因果贡献仍难以完全解耦。

## 五、评价与展望

**优点**：把"理解-生成-动作"三件事纳入同一 token 空间和同一组权重，通过双塔 MoT + 因果/双向两种注意力模式在架构层面干净地解决了自回归推理与扩散生成的兼容问题；相比只做图文的统一多模态框架，Cosmos 3 是较早把"动作"作为一等模态纳入 Mixture-of-Transformers、并系统性验证前向动力学/逆向动力学/策略三种动作生成模式相互受益（PushT 协同消融、跨域 synergy 矩阵）的公开工作，MT-init vs PT-init 的对照实验也提供了较有说服力的证据链，证明统一动作中训练确实沉淀出可迁移的"世界-动作先验"。

**与其他公开工作的关系**：相比 Cosmos-Predict2.5 / Cosmos-Transfer2.5 等前代专用模型（分别针对预测/控制条件生成单独建模），Cosmos 3 用单一 backbone 在多数指标上反超这些专用模型（如 video transfer 任务上，单一统一模型打平或超过 Cosmos-Transfer2.5 为每种控制模态单独配备 ControlNet 分支的设计）；相比同样定位 world-action model 的 DreamZero，Cosmos3-Nano-Policy-DROID 在 RoboLab 与 RoboArena 上直接反超；相比 BAGEL、Transfusion 等通用统一多模态生成框架，Cosmos 3 的差异化在于把动作、音视频同步生成、control-conditioned 视频转换这些 Physical AI 特有任务系统纳入了同一套评测体系，而不只停留在图文理解-生成层面。

**开放问题**：（1）统一模型在通用语言/视觉推理与物理 AI 垂类能力之间的权衡边界（scaling law）尚未给出，进一步扩大模型规模能否同时补齐两端有待验证；（2）动作表征目前基于 6D 旋转 + 3D 平移的刚体伪动作，对柔性物体、双臂协同接触力等更复杂物理量尚未覆盖；（3）机器人策略验证局限于 DROID / 单臂 Franka 平台，向人形机器人、双臂协同等更高自由度具身的可扩展性有待后续工作验证；（4）依赖推理时 Best-of-N 重排才能达到部分指标的 SOTA，提示单次采样的物理一致性仍有改进空间；（5）大量使用自建/内部 benchmark（HUE、RBench、TAR、VANTAGE-Bench 等），虽已配合第三方公开榜单（Artificial Analysis、RoboArena、MolmoSpaces）交叉验证，完全独立的第三方复现仍需时间检验。

## 参考

- Zhou et al., 2019. *On the Continuity of Rotation Representations in Neural Networks.*（6D 旋转表示，Cosmos 3 动作 token 化的几何基础）
- Bai et al., 2025a/b. *Qwen3-VL.*（Reasoner 初始化底座，3D MRoPE 设计的直接来源）
- Deng et al., 2025. *BAGEL: Unified Multimodal Pretraining for Understanding and Generation.*（decoder-only 统一理解生成架构，本文明确对比的相关设计）
- Khazatsky et al., 2024. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.*（机器人策略后训练所用数据集）
- Ye et al., 2026. *DreamZero.*（同为 world-action model 的策略基线，被 Cosmos3-Nano-Policy 在 RoboLab / RoboArena 上反超）
