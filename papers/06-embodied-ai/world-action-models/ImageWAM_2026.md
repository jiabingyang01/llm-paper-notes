# ImageWAM：World Action Model 真的需要视频生成吗，还是图像编辑就够了？

> **论文**：*ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?*
>
> **作者**：Yuang Zhang*、Wenyao Zhang*†、Zekun Qi、He Zhang、Haitao Lin、Jingbo Zhang、Yao Mu、Xiaokang Yang、Wenjun Zeng、Xin Jin（通讯）等
>
> **机构**：Shanghai Jiao Tong University、Eastern Institute of Technology、Tencent Robotics X、Tsinghua University、Zhongguancun Academy
>
> **发布时间**：2026 年 06 月（arXiv 2606.19531）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.19531) | [PDF](https://arxiv.org/pdf/2606.19531)
>
> **分类标签**：`World Action Model` `图像编辑` `流匹配动作专家` `KV 缓存条件化` `RoboTwin2.0` `LIBERO-Plus`

---

## 一句话总结

用预训练图像编辑模型（OmniGen2 / Ovis-U1 / FLUX.2）替代视频生成模型作为 World Action Model 的骨干：只预测一帧任务相关的目标终帧，训练/推理时都不解码这帧图像，而是直接把编辑分支去噪过程中产生的中间层 KV 缓存喂给一个流匹配动作专家，在 RoboTwin2.0（随机化）上做到 93.56% 成功率、真实机器人四任务平均 84.5%，同时把推理 FLOPs 降到视频式 WAM 的约 1/6、延迟降到约 1/4。

## 一、问题与动机

当前一批 World Action Model（WAM）把视频生成模型当作机器人策略的骨干：先用视频扩散/自回归模型预测完整的未来视频轨迹，再用一个逆动力学模型或动作头把预测的视频翻译成动作（"先想象、后行动"范式）。这条路线的直觉是：视频预训练模型见过大量物体运动、时序连续性、物理交互，能给策略学习提供丰富的视觉动态先验。

但作者指出这一路线存在三个耦合的问题：

1. **推理开销大**：视频生成需要在多帧上生成密集的时空 token，实时机器人控制难以承受。
2. **容量错配**：完整视频预测必须建模外观细节、背景变化、相机运动等与动作弱相关的因素，这些"容量"本可以用在真正与任务相关的视觉变化上。
3. **长时误差误导动作**：生成物理一致的视频是一个很难的代理任务，尤其是精细操作场景中的微小接触/位移，视频里的几何畸变、错误生成一旦发生，会直接误导下游的动作预测（论文 Figure 5 给出了具体的失败可视化）。

由此作者提出一个直接的问题：**World Action Model 真的需要视频生成吗？** 他们认为图像编辑模型天然更贴合语言条件机器人操作的核心需求——理解"在给定指令下，当前场景里哪些地方需要改变"，而不是渲染一段逼真的未来视频。图像编辑预训练本身就是"指令 → 源图到目标图的变换"，与操作任务的本质高度对齐，同时提供了三个优势：指令到变化的强对齐、比完整视频预测更简单/更贴近动作的代理任务、以及无需解码密集多帧视频的紧凑推理路径。

## 二、核心方法

### 2.1 问题形式化

标准的语言条件操作策略在给定观测 $o_t$ 和指令 $l$ 后预测一个动作块

$$\mathbf{a}_{t:t+H} = (a_t, a_{t+1}, \dots, a_{t+H})$$

$$\pi_\theta(\mathbf{a}_{t:t+H} \mid o_t, l)$$

视频式 WAM 在动作预测前插入一个显式的未来视频预测步骤：

$$(o_t, l) \to \hat{o}_{t+1:t+H+1} \to \mathbf{a}_{t:t+H}$$

即先生成一段完整的未来视频轨迹 $\hat{o}_{t+1:t+H+1}$，再据此预测动作。

**用大白话说**：以前的 WAM 要"拍一整段未来的电影"再决定怎么动手，ImageWAM 则只要"P 一张未来会变成什么样的静态照片"（甚至连这张照片都不用真正画出来）。

ImageWAM 把这一步替换为只预测**单帧终点**：

$$(o_t, l) \to \hat{o}_{\text{edit}} \equiv \hat{o}_{t+H+1} \to \mathbf{a}_{t:t+H}$$

其中 $\hat{o}_{\text{edit}}$ 是一个源图条件化（source-conditioned）的单帧图像，概括了当前观测在任务指令下应发生的关键视觉变换，作为动作预测的紧凑"世界—动作"中间表征。

### 2.2 架构：把动作专家接到图像编辑分支上

ImageWAM 基于一个图像编辑模型变体（论文实现了 OmniGen2、Ovis-U1、FLUX.2 三种骨干），给其编辑分支接上一个动作专家（Action Expert）。关键设计是：**不用编辑分支去解码出最终的编辑图像，而是复用它去噪过程中产生的中间层 KV 缓存作为条件**。

训练时随机采样一个编辑去噪时间步 $\tau$，对每个 Transformer 层 $\ell$ 收集对应的 KV 缓存：

$$\mathcal{C}^\tau_{\text{edit}} = \{(K_\ell^\tau, V_\ell^\tau)\}_{\ell=1}^L = f^\tau_{\text{edit}}(o_t, l)$$

其中 $L$ 是 Transformer 层数。这个缓存是视觉隐向量已经与任务指令交互之后计算出来的，因此天然携带"任务条件下视觉应如何变换"的信息，而不需要先解码出最终的编辑图。

**用大白话说**：图像编辑模型内部在"画"这张目标图的过程中，其实已经在脑子里想好了"哪里要变、怎么变"，ImageWAM 把这个"脑内草稿"（KV 缓存）直接掏出来给动作专家用，省去了真的把图画完再让动作专家去看图的步骤。

动作专家通过联合自注意力（joint self-attention，MoT 结构）把语言上下文 token、视觉条件 token、视觉预测 token、动作 token 四类 token 拼接起来统一处理：动作 token 单向地关注其他三类 token，而带噪的视觉预测 token 只关注干净的上下文 token（保持编辑分支不被动作分支的噪声梯度污染）。三种骨干变体的动作专家规模：OmniGen2 版约 760M 参数，FLUX.2 4B/9B 版分别为 642M/952M 参数，Ovis-U1 版为 1.1B 参数。训练时冻结 VLM/多模态理解组件（如 Qwen2.5-VL-3B、Qwen3-4B/8B、Qwen3-1.7B），只更新扩散图像生成分支和动作专家。

### 2.3 训练目标

**图像编辑目标**：编辑分支被训练去预测任务相关的未来终帧。设 $z^*_{t+H+1} = E_{\text{vae}}(o_{t+H+1})$ 为目标观测的 VAE 潜表征，采样图像噪声 $\epsilon_z \sim \mathcal{N}(0, I)$ 和流时间 $r \in (0,1)$，构造插值潜变量

$$z_r = (1-r)\, z^*_{t+H+1} + r\, \epsilon_z$$

图像扩散分支预测对应的速度场：

$$\mathcal{L}_{\text{img}} = \mathbb{E}_{z^*, \epsilon_z, r}\left[\, \| u_\phi(z_r, r \mid o_t) - (\epsilon_z - z^*_{t+H+1}) \|_2^2 \,\right]$$

（原文该式右侧写作 $z^*_{t+K}$，与正文对目标帧 $z^*_{t+H+1}$ 的定义不一致，推测是笔误，此处按其定义还原为 $z^*_{t+H+1}$。）

**用大白话说**：这是标准的 flow-matching 训练——让模型学会一条从纯噪声"流"到目标未来帧潜变量的速度场，本质就是训练编辑模型好好完成"P 出未来那张图"这件事，只是训练完之后并不真的拿它去出图。

**动作流匹配**：动作专家用流匹配目标生成动作块。设 $\mathbf{a}^*_{t:t+H}$ 为专家动作块，$\epsilon_a \sim \mathcal{N}(0, I)$，采样动作流时间 $s \in (0,1)$，构造

$$\mathbf{a}_s = (1-s)\, \mathbf{a}^*_{t:t+H} + s\, \epsilon_a$$

在当前观测、任务指令、编辑上下文缓存 $\mathcal{C}^\tau_{\text{edit}}$ 条件下，动作专家预测速度场：

$$\mathcal{L}_{\text{act}} = \mathbb{E}_{\mathbf{a}^*, \epsilon_a, s, \tau}\left[\, \| v_\theta(\mathbf{a}_s, s \mid o_t, l, \mathcal{C}^\tau_{\text{edit}}) - (\epsilon_a - \mathbf{a}^*_{t:t+H}) \|_2^2 \,\right]$$

其中 $\tau$ 是训练时用来抽取编辑缓存的去噪时间步，随机采样 $\tau$ 使动作专家能适应去噪过程不同阶段的编辑缓存。总损失为两项之和：$\mathcal{L} = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{img}}$，联合优化图像扩散分支和动作专家。

### 2.4 高效推理

推理时既不跑完整的未来视频生成，也不解码出完整的编辑图，而是只固定一个编辑去噪时间步 $\tau^*$，跑**一次**编辑分支前向：

$$\mathcal{C}^{\tau^*}_{\text{edit}} = f^{\tau^*}_{\text{edit}}(o_t, l)$$

再让动作专家基于这个缓存去噪出动作块：

$$\hat{\mathbf{a}}_{t:t+H} \sim p_\theta(\mathbf{a}_{t:t+H} \mid o_t, l, \mathcal{C}^{\tau^*}_{\text{edit}})$$

**用大白话说**：视频式 WAM 要把多帧未来视频的 token 全部去噪并很可能还要解码出来，ImageWAM 只需算一次编辑分支的层级 KV 缓存，省掉了密集时空 token 的生成与解码，这是推理提速的主要来源。

作为对照，论文还实现了一个 **Fast-WAM 风格的视频式 WAM 变体**：训练时用未来视频 token 做协同训练，但推理时把这些视频 token 全部去掉，动作专家只用当前观测/指令产生的 KV 缓存做条件——这样得到一个与 Fast-WAM 推理接口相同（无未来 token）的视频式 WAM 基线，用于和 ImageWAM 做同等推理开销下的对比（论文表中记作 FastWAM）。

## 三、实验结果

评测覆盖 RoboTwin2.0（双臂仿真，50+ 任务）、LIBERO 与 LIBERO-Plus（分布偏移鲁棒性）、以及基于 Dobot Xtrainer 双臂平台的真实机器人四任务。ImageWAM **不使用额外的具身策略预训练（P.T.）**，只在下游 benchmark 演示数据上训练。

**RoboTwin2.0**（Clean / Rand / Avg，成功率 %）：

| 方法 | P.T. | Clean | Rand | Avg |
|---|---|---|---|---|
| π0 | ✓ | 65.92 | 58.40 | 62.16 |
| π0.5 | ✓ | 82.74 | 76.76 | 79.75 |
| ABot-M0 | ✗ | 81.20 | 80.40 | 80.80 |
| Motus | ✓ | 88.66 | 87.02 | 87.80 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| FastWAM | ✗ | 91.88 | 91.78 | 91.83 |
| **ImageWAM（本文）** | ✗ | **93.20** | **93.56** | **93.38** |

**LIBERO**（Spatial/Object/Goal/Long/Avg，四套件各 500 条示范）：ImageWAM 无 P.T. 达到 Avg 98.4%（Spatial 97.2 / Object 99.2 / Goal 98.8 / Long 98.4），与需要额外预训练的 LingBot-VA（98.5）基本持平，明显优于 Fast-WAM（97.6）和 π0.5（96.9）。

**LIBERO-Plus**（在标准 LIBERO 示范上训练，评测七种扰动维度：Camera/Robot/Language/Light/Background/Noise/Layout）：

| 方法 | P.T. | Avg |
|---|---|---|
| π0 | ✓ | 53.6 |
| WorldVLA | ✓ | 25.0 |
| OpenVLA-OFT | ✓ | 69.6 |
| FastWAM | ✗ | 51.5 |
| ImageWAM(OmniGen2) | ✗ | 71.8 |
| ImageWAM(Ovis-U1) | ✗ | 71.2 |
| **ImageWAM(FLUX.2 4B)** | ✗ | **83.1** |

FLUX.2 4B 版本在 Camera（80.8）、Noise（93.8）等维度大幅领先所有对比方法，说明编辑式条件在分布偏移下的鲁棒性优于视频式 WAM 和标准 VLA。

**真实机器人**（Dobot Xtrainer 双臂，T1 摞碗 / T2 叠毛巾 / T3 开抽屉存放记号笔 / T4 挂杯子，各任务约 100 条演示，成功率 %）：

| 方法 | T1 | T2 | T3 | T4 | Avg |
|---|---|---|---|---|---|
| π0 | 57 | 58 | 54 | 54 | 55.8 |
| π0.5 | 83 | 77 | 74 | 55 | 72.3 |
| FastWAM | 88 | 75 | 77 | 76 | 79.0 |
| **ImageWAM** | **94** | **84** | **78** | **82** | **84.5** |

（注：正文称"over 100 trials"，附录 7.1 节又称"50 trials per task"，二者表述不一致，原文未澄清，此处如实标注。）

**效率对比**（A6000 GPU，延迟/TFLOPs）：

| 方法 | 延迟 (ms) | TFLOPs | 中间表征 |
|---|---|---|---|
| FastWAM-IDM（完整视频式） | 1081 | 63.65 | 视频 |
| FastWAM（单步去噪） | 302 | 13.21 | 缓存 |
| **ImageWAM（本文）** | **263** | **9.72** | 缓存 |

对比完整视频式 WAM（FastWAM-IDM），ImageWAM 延迟降到约 1/4，FLOPs 降到约 1/6，同时任务成功率保持竞争力，与摘要中的"FLOPs 降至 1/6、延迟降至 1/4"数字一致。进一步用 `torch.compile`/静态 CUDA graph 优化后，ImageWAM 延迟可进一步压到 69 ms（相对 FastWAM 单步基线 4.38× 加速）。

**与统一理解-生成模型的对比**（Table 6，LIBERO / RoboTwin2.0-Clean-Only / RoboTwin2.0-Clean2Hard）：

| 方法 | P.T. | LIBERO | Clean-Only | Clean2Hard |
|---|---|---|---|---|
| UniVLA | ✓ | 95.5 | – | – |
| BagelVLA（带关键帧预测） | ✓ | – | 75.3 | 20.9 |
| BagelVLA（不带关键帧预测） | ✓ | – | 56.7 | 15.9 |
| ImageWAM（本文） | ✗ | **98.4** | **84.4** | 18.3 |

ImageWAM 在 LIBERO 和 RoboTwin2.0-Clean-Only 上明显领先两个"统一理解-生成"骨干模型（UniVLA、BagelVLA），但在跨域泛化最严苛的 Clean2Hard 设置下（18.3%）略低于 BagelVLA 的关键帧变体（20.9%），后者是有额外具身策略预训练的模型。

**消融**：(1) 编辑骨干替换（OmniGen2/Ovis-U1/FLUX.2 4B）在 LIBERO-Plus 上均优于 FastWAM，FLUX.2 4B 最优（83.1%），说明方法不依赖特定编辑模型，更强的编辑骨干能直接提升策略鲁棒性；(2) 换用更大的 FLUX.2 9B 骨干，LIBERO-Plus Avg 从 83.1% 提升到 85.2%，但提升并非在所有扰动维度单调（Camera/Light/Noise 未必受益）；(3) 注意力可视化（Figure 4）显示 ImageWAM 的编辑缓存注意力集中在被操作物体、目标容器和接触区域，而 FastWAM 的注意力更分散；(4) 定性分析（Figure 5）显示视频式 WAM 基线生成的未来帧在任务相关物体周围出现明显几何畸变和空间布局不一致，可能误导动作专家，而 ImageWAM 因不在推理时解码密集未来帧，避免了这类视觉伪影的累积。

## 四、局限性

- **跨域大幅偏移下的泛化仍有差距**：在 RoboTwin2.0 Clean2Hard 这一最严苛的分布迁移测试中（18.3%），ImageWAM 略逊于带策略预训练的 BagelVLA 关键帧变体（20.9%），说明在没有额外具身预训练的情况下，纯图像编辑先验对极端 domain shift 的覆盖仍不如经过大规模具身数据预训练的统一模型。
- **主文本与附录关于真实机器人试验次数的表述不一致**（正文称 100 trials，附录 7.1 节称 50 trials per task），论文未对此做澄清，可能影响对真实世界结果统计置信度的判断。
- **单帧终点表征可能丢失中间轨迹信息**：方法只预测任务终点的单帧图像，对于需要精细中间状态引导（如多阶段接触、避障轨迹形状本身就是任务约束）的长时序/高精度操作，单终帧代理是否总能提供足够信息尚未被充分讨论，论文的真实任务集合（摞碗、叠毛巾、开抽屉、挂杯）总体仍偏短时序。
- **编辑去噪时间步 $\tau^*$ 的选取**：推理时固定单一去噪时间步 $\tau^*$ 抽取缓存，论文未给出选择 $\tau^*$ 的系统性分析或消融（只说训练时随机采样 $\tau$），其对下游性能的敏感度不明确。
- **骨干规模增大带来的收益不是单调**：Table 7 显示换用更大编辑骨干（FLUX.2 9B）时，Camera / Light / Noise 三个扰动维度并未随之改善，说明"更强编辑先验 = 更强策略鲁棒性"这一假设只在部分扰动类型上成立。

## 五、评价与展望

**贡献与优点**：这篇论文提出了一个简洁但反直觉的问题——World Action Model 是否真的需要"生成完整未来视频"这一较重的中间表征——并给出了较有说服力的证据：图像编辑模型的"源图 + 指令 → 目标图"训练目标本身就与操作任务的"当前观测 + 指令 → 需要发生的变化"高度同构,而且只需复用编辑去噪过程中的中间层 KV 缓存作为条件、无需解码最终图像，就能兼顾表征质量与推理效率。这一设计相比"预测完整未来视频再解码/翻译成动作"的传统 WAM 路线（如论文引用的 Video Prediction Policy、Cosmos-policy 等系列工作）在推理成本上有数量级级别的优势（FLOPs ~1/6、延迟 ~1/4），且在多个 benchmark（RoboTwin2.0、LIBERO、LIBERO-Plus、真实机器人）上没有牺牲甚至反而提升了成功率与鲁棒性，是"生成式视觉预训练是否必须走视频路线"这一问题上一个有分量的反例。

**与其他公开工作的关系**：ImageWAM 与同期的 Fast-WAM（作者引用为 [13]，同样质疑"WAM 是否需要测试时未来想象"）在问题动机上高度相关，但切入点不同——Fast-WAM 保留视频式训练但去掉推理时的未来 token 生成，ImageWAM 则更进一步，从骨干模型的选择上直接用图像编辑替代视频生成，两者在本文中被组合/对比（论文将 Fast-WAM 风格的"训练用视频、推理去 token"策略实现为视频式 WAM 基线）。与 UniVLA、BagelVLA 等"统一理解-生成"模型的对比（Table 6）也提供了一个有意思的架构选择证据：作者认为联合优化理解与生成两个目标可能相互干扰（理解需要高层语义抽象，生成需要细粒度空间/结构细节），因此选择冻结 VLM 理解分支、只训练生成分支和动作专家，这一设计与消融结果一致地优于两个统一模型基线（在无策略预训练的情况下），为"理解-生成解耦 vs 联合训练"这一开放问题提供了一个正面案例。

**开放问题与可能的改进方向**：(1) 单终帧编辑代理在多阶段、长时序任务上的信息充分性仍待更系统的验证——论文的真实任务集合和 LIBERO/RoboTwin 任务本身时序跨度有限，是否在更复杂的多阶段任务上仍然优于视频式表征值得进一步检验；(2) 编辑去噪时间步 $\tau^*$ 的选择、以及训练时对 $\tau$ 的采样策略对最终表征质量的影响机制还不够透明，可能存在更优的调度策略；(3) Clean2Hard 泛化差距提示"图像编辑先验 + 无额外具身预训练"路线在极端分布迁移下可能仍需要引入某种形式的大规模跨场景/跨具身数据预训练来弥补，这也是与走大规模视频预训练路线的 WAM 相比一个值得跟踪的权衡点；(4) 论文只在三个静态图像编辑骨干（OmniGen2、Ovis-U1、FLUX.2）上验证了方法的可迁移性，编辑骨干的指令遵循能力/编辑精度与下游策略性能之间的定量关系（尤其是消融中观察到的"规模增大但部分扰动维度不单调改善"现象）仍缺乏更细粒度的解释。

## 参考

- [13] Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao. *Fast-WAM: Do World Action Models Need Test-Time Future Imagination?* arXiv:2603.16666, 2026.
- [3] Lin Li, Qihang Zhang, Yiming Luo, et al. *Causal World Modeling for Robot Control.* arXiv:2601.21998, 2026.（对比方法 LingBot-VA）
- [84] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, et al. *OmniGen2: Exploration to Advanced Multimodal Generation.* arXiv:2506.18871, 2025.（本文所用图像编辑骨干之一）
- [6] Yucheng Hu, Jianke Zhang, Yuanfei Luo, et al. *BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation.* arXiv:2602.09849, 2026.（统一理解-生成模型对比对象）
- [89] Tianxing Chen, Zanxin Chen, Baijun Chen, et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation.* arXiv:2506.18088, 2025.
