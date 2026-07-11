# Action Images：把 7 自由度动作变成像素——多视角视频生成即策略学习

> **论文**：*Action Images: End-to-End Policy Learning via Multiview Video Generation*
>
> **作者**：Haoyu Zhen, Zixian Gao, Qiao Sun, Yilin Zhao, Yuncong Yang, Yilun Du, Pengsheng Guo, Tsun-Hsuan Wang, Yi-Ling Qiao, Chuang Gan
>
> **机构**：UMass Amherst、NVIDIA、Harvard University、Genesis AI
>
> **发布时间**：2026 年 04 月（arXiv 2604.06168，v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.06168) | [PDF](https://arxiv.org/pdf/2604.06168) | 项目页 [ActionImages.github.io](https://ActionImages.github.io)
>
> **分类标签**：`世界动作模型` `像素级动作表征` `多视角视频生成` `零样本策略` `视频扩散骨干` `动作图像`

---

## 一句话总结

把机器人 7 自由度（DoF）动作编码为三个语义 3D 点（位置点/法向点/上方点）在多相机视角下渲染出的 RGB 高斯热力图"动作图像"，与观测视频拼进同一段 latent 序列，用统一的多视角视频扩散骨干（微调自 Wan 2.2/2.1）联合建模观测与动作，从而让视频骨干本身直接充当零样本策略，无需额外动作头；在 RLBench 零样本任务上平均成功率约 39%、真实 xArm 零样本任务上约 26%，显著超过 π0.5、MolmoAct、TesserAct、Cosmos-Policy、MV-Policy 等基线（多数任务成功率为 0）。

## 一、问题与动机

世界动作模型（World Action Models, WAMs）近年被寄予厚望：借助预训练视频生成骨干强大的物理/场景先验来做机器人策略学习。但作者指出一个核心 gap——**视频生成的泛化能力不会自动转化为策略的泛化能力**：模型可能生成非常逼真的未来帧，却仍然不知道在没见过的环境里该怎么做。

根本原因在于动作在现有方法中的表征方式，通常走两条路：
- 在世界模型上外挂**独立的动作头/动作模块**（如 DreamZero、TesserAct 等），让一个额外网络从视频特征里解码控制信号；
- 把动作转成与图像空间不对齐的**低维/隐编码 token**（如部分基于 Cosmos 的方法），视频骨干的预训练知识无法被这类动作接口直接利用。

两种做法都使得"预测世界"和"据此行动"之间只有间接联系，泛化的负担被甩给了这个额外的、专门化的控制接口——而这恰恰是迁移最容易失效的地方。本文的思路是**在表征层面解决泛化问题**：把动作也变成像素，让策略学习就是视频生成本身。

## 二、核心方法

### 2.1 动作即图像（Action as Images）

在时刻 $t$，机器人动作定义为 $\mathbf{a}_t = [\mathbf{p}_t, \boldsymbol{\theta}_t, g_t] \in \mathbb{R}^7$，其中 $\mathbf{p}_t \in \mathbb{R}^3$ 是末端执行器位置，$\boldsymbol{\theta}_t \in \mathbb{R}^3$ 是姿态，$g_t \in \mathbb{R}$ 是夹爪开合度。作者把它转成三个语义 3D 点：位置点直接是 $\mathbf{q}_t^{\text{pos}} = \mathbf{p}_t$；另外两个点通过绕末端执行器的两个规范轴旋转再延伸一小段长度 $\ell$ 得到：

$$\mathbf{q}_t^{\text{up}} = \mathbf{p}_t + \ell\, \mathbf{R}(\boldsymbol{\theta}_t)\,\mathbf{e}_x, \qquad \mathbf{q}_t^{\text{normal}} = \mathbf{p}_t + \ell\, \mathbf{R}(\boldsymbol{\theta}_t)(-\mathbf{e}_z)$$

**大白话**：与其把姿态压成一个抽象的四元数或欧拉角，不如把它变成三个"看得见"的空间路标点——一个代表夹爪在哪，另两个分别代表夹爪朝哪个方向、法平面朝哪个方向，三点共同唯一确定末端 6D 位姿。

给定相机视角 $v$ 的投影函数 $\pi_t^{(v)}(\cdot)$，把三个 3D 点投影到图像空间：

$$\mathbf{u}_t^{\text{pos},(v)} = \pi_t^{(v)}(\mathbf{q}_t^{\text{pos}}), \quad \mathbf{u}_t^{\text{normal},(v)} = \pi_t^{(v)}(\mathbf{q}_t^{\text{normal}}), \quad \mathbf{u}_t^{\text{up},(v)} = \pi_t^{(v)}(\mathbf{q}_t^{\text{up}})$$

再用 2D 高斯核 $\mathcal{G}(\mathbf{x};\mathbf{u},\sigma)$ 把三点渲染成一张三通道"动作图像" $\mathbf{A}_t^{(v)} \in \mathbb{R}^{H\times W\times 3}$：红通道编码位置点，绿通道编码法向点：

$$\mathbf{A}_t^{(v)}(:,:,1) = \mathcal{G}(\cdot;\mathbf{u}_t^{\text{pos},(v)},\sigma), \qquad \mathbf{A}_t^{(v)}(:,:,2) = \mathcal{G}(\cdot;\mathbf{u}_t^{\text{normal},(v)},\sigma)$$

蓝通道先渲染上方点的高斯响应 $\tilde{\mathbf{A}}_t^{(v)}(:,:,3)$，再在响应较弱的背景区域注入夹爪开合信号：

$$\mathbf{A}_t^{(v)}(i,j,3) = \begin{cases} \tilde{\mathbf{A}}_t^{(v)}(i,j,3), & \tilde{\mathbf{A}}_t^{(v)}(i,j,3) > 0.25 \\[4pt] 0.25 \cdot g_t, & \text{否则} \end{cases}$$

**大白话**：三个通道各管一件事——红/绿两通道分别是两个"路标点"的空间热力图，蓝通道既画第三个路标点，又顺手把背景填成一片能反映夹爪开合度的灰度值，一举两得地把 7 个数字全部塞进了一张普通 RGB 图片里，格式和机器人观测视频完全一致。

把多个时刻的动作图像沿时间堆叠即得到"动作视频" $\mathcal{A}^{(v)} = \{\mathbf{A}_1^{(v)}, \dots, \mathbf{A}_T^{(v)}\} \in \mathbb{R}^{T\times H\times W\times 3}$，它与对应视角的 RGB 观测视频 $\mathcal{O}^{(v)}$ 在时空结构上完全对齐，二者由此构成统一的"视频空间"观测-动作表征。之所以要用**多视角**而非单视角，是因为单一视角常常只能给出运动的一个模糊投影，遮挡时更难仅凭像素恢复完整 7-DoF 动作；多视角互相印证既让动作图像更容易被重建，也提升了对局部遮挡的鲁棒性。

### 2.2 动作图像解码

生成出的动作图像要能倒推回连续控制量。夹爪开合度直接由蓝通道背景区域的均值恢复：

$$\hat{g}_t = \frac{1}{0.25}\cdot\frac{1}{|\Omega_t|}\sum_{(i,j,v)\in \Omega_t} \mathbf{A}_t^{(v)}(i,j,3), \qquad \Omega_t = \{(i,j,v) \mid \mathbf{A}_t^{(v)}(i,j,3) < 0.25\}$$

其余三个语义点的 3D 位置通过一个"主视角选点 + 射线投射 + 侧视角匹配"的几何流程恢复：先在主视角热力图上做加权质心得到 2D 锚点，再从主视角相机中心沿射线在近远平面之间采样一批候选 3D 点 $\{\mathbf{x}_{t,k}\}_{k=1}^K$，把每个候选点投影到侧视角，与侧视角热力图 $\mathbf{H}_t^{(2)}$ 做匹配打分，取响应最高的候选作为重建结果：

$$\hat{\mathbf{x}}_t = \arg\max_{\mathbf{x}_{t,k}} \mathbf{H}_t^{(2)}\Big(\pi_t^{(2)}(\mathbf{x}_{t,k})\Big)$$

**大白话**：主视角告诉你"这个点大概在图像哪个位置"，但缺深度信息；把这条视线拉长成一条 3D 射线后，用侧视角来"验货"——沿着射线走的每一步，看看投到侧视角上是否也落在热力图的高响应区，最吻合的那一步就是真实深度。三个语义点都用同样的流程重建后，取 $\hat{\mathbf{e}}_x^r=\mathrm{norm}(\hat{\mathbf{q}}^{\text{up}}-\hat{\mathbf{q}}^{\text{pos}})$、$\hat{\mathbf{e}}_z^r=\mathrm{norm}(\hat{\mathbf{q}}^{\text{pos}}-\hat{\mathbf{q}}^{\text{normal}})$、$\hat{\mathbf{e}}_y^r=\hat{\mathbf{e}}_z^r\times \hat{\mathbf{e}}_x^r$ 组成旋转矩阵，即可还原姿态 $\hat{\boldsymbol{\theta}}_t$，最终得到解码动作 $\hat{\mathbf{a}}_t=[\hat{\mathbf{p}}_t,\hat{\boldsymbol{\theta}}_t,\hat{g}_t]$。作者指出，只要预测热力图本身准确，解码误差主要来自离散化（射线采样间隔决定深度精度、热力图空间分辨率决定定位精度），而非表征本身的信息损失。

### 2.3 统一世界-动作模型训练

模型基于预训练视频生成骨干 Wan 2.2（论文正文写 "Wan 2.2"，附录实现细节写 "Wan2.1-I2V-14B-480P"，推理效率表则将部署规模标注为 5B，三处表述在具体版本号/参数量上略有出入）做微调。对每个相机视角 $v$，把 RGB 观测片段 $\mathbf{V}^{(v)}_{1:T}$ 和对齐的动作帧片段 $\mathbf{A}^{(v)}_{1:T}$ 分别经 3D-VAE 编码后沿时间维拼接成统一的输入 token 序列 $\mathbf{X}_v=[\mathbf{V}^{(v)}_{1:T}, \mathbf{A}^{(v)}_{1:T}]$。

训练采用**多重掩码策略（multiple mask strategy）**在同一套 flow-matching 扩散框架下随机切换四种任务：(1) 动作-视频联合生成——同时遮住 $\mathcal{V}$、$\mathcal{A}$（仅留首帧观测），由文本+相机条件生成两者；(2) 动作条件视频生成——保留 $\mathcal{A}$ 可见、遮住 $\mathcal{V}$，让模型据给定动作合成未来观测；(3) 视频到动作标注——保留 $\mathcal{V}$ 可见、遮住 $\mathcal{A}$，让模型从输入视频推断动作；(4) 纯视频生成——对没有可用动作标注的数据只用视频 token 训练。相机条件沿用 ReCamMaster 的 Plücker embedding 注入方案，保证跨视角一致性。整体用 flow matching 目标训练（目标速度 $\mathbf{v}=\boldsymbol{\epsilon}-\mathbf{X}$），在被掩码的 latent token 上取 $L_2$ 损失：

$$\mathcal{L} = \mathbb{E}\Big[\big\| M \odot \big(\mathbf{v} - \mathbf{v}_\theta(\mathbf{X}, \mathcal{T}, \mathbf{cam})\big) \big\|_2^2\Big]$$

**大白话**：同一个骨干、同一套损失，只靠"哪部分 token 可见、哪部分要被预测"这个开关，就能在推理时切换成策略（给观测测动作）、世界模型（给动作测未来）、动作标注器（给视频反推动作）四种角色，天然获得了统一表征带来的多任务能力。

训练数据混合 DROID（8 万条轨迹、2 视角、真实机器人、有动作标注）、RLBench（18 万条轨迹、4 视角、仿真环境、动作与相机信号精确，并用 Robot-Colosseum 背景增强补充视觉多样性）、BridgeV2（3 万条轨迹、1-4 视角、真实机器人视频，缺相机标定与动作-相机对齐，用 VGGT 估计相机参数、仅用于纯视频生成）。训练用 32 张 A100、DeepSpeed ZeRO、bfloat16、per-device batch size 1，constant-with-warmup 学习率 $5\times10^{-7}$，100k 优化步；推理用 classifier-free guidance（scale 10.0）、50 步去噪、4-GPU 统一序列并行，叠加 CFG 并行/VAE 缓存/torch.compile 等系统优化后骨干可达 71 FPS。

## 三、实验结果

**零样本策略成功率（Table 2）**：RLBench 环境完全从训练集划分中移除，真实场景在 xArm 平台上物体/环境/语言指令全新。对比基线包括 MV-Policy（多视角版 Diffusion Policy）、π0.5、MolmoAct（可推理 2D 轨迹再提升为 3D 的推理式模型）、TesserAct、Cosmos-Policy（后两者用相同 Wan 2.2 骨干复现，公平对比）。

| 方法 | pick cup | reach target | close drawer | close laptop | Place Cup | Pick Unseen Toy | Pick Tissue | Close Drawer | Close Box |
|---|---|---|---|---|---|---|---|---|---|
| MV-Policy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| π0.5 | 0 | 5 | 35 | 20 | 5 | 0 | 0 | 0 | 0 |
| MolmoAct | 20 | 5 | 10 | 0 | 10 | 5 | 5 | 0 | 0 |
| TesserAct | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cosmos-Policy | 0 | 5 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Ours** | **30** | **60** | **50** | **15** | **40** | **20** | **15** | **45** | **10** |

RLBench 零样本四任务平均约 39%，真实五任务平均约 26%，全面超越所有基线；论文强调**分布偏移越强、优势越明显**，支持了"像素级动作表征更泛化"的核心论点。

**RLBench 域内评测（Table 3，9 个任务）**：不额外加动作头时，本方法平均成功率 20.6%，与 TesserAct（20.6%）持平、优于 Cosmos-Policy（20.0%）、MV-Diffusion Policy（17.8%）、π0.5（14.4%）、零样本 MolmoAct（14.4%）；额外接一个轻量 MLP 动作头（吃视频 latent + 相机参数 + 解码动作/观测做回归）后跃升到 **36.7%**，尤其在精度敏感任务（close box 55→80、open bottle 5→40）上提升显著，说明几何解码本身仍有信息损失、可被一个小型学习头进一步弥补。

**联合视频-动作生成质量（Table 4，RLBench 域内）**：

| 模型 | PSNR↑ | SSIM(%)↑ | FVD↓ | LPIPS↓ | 2D 误差↓ | 3D 误差(×10³)↓ |
|---|---|---|---|---|---|---|
| Cosmos-Predict2.5-14B（零样本） | 17.92 | 50.77 | 208.65 | 0.409 | – | – |
| Cosmos-Policy | 18.29 | 53.41 | 192.58 | 0.418 | 2.11 | 19.4 |
| TesserAct | 20.83 | 59.20 | 154.38 | 0.351 | 1.84 | 19.0 |
| TesserAct-RGB | 20.31 | 60.19 | 147.83 | 0.372 | **1.55** | 14.2 |
| **Ours** | **23.48** | **78.62** | **143.74** | **0.209** | 1.61 | **12.2** |

视频质量四项指标全面最优，3D 动作误差也最低（2D 误差略输 TesserAct-RGB）。

**动作条件视频生成（Table 5，对比 2D 轨迹条件基线 Tora）**：PSNR 31.35 vs 19.76，SSIM 67.16% vs 52.43%，LVD 115.02 vs 187.41，LPIPS 21.78% vs 39.62%，全面占优。

**视频到动作标注（Table 6，对比点跟踪基线 TAPIR/CoTracker3）**：

| 方法 | 轨迹误差↓ | Jaccard@4↑ | 平均 Jaccard↑ |
|---|---|---|---|
| TAPIR | 14.80 | 40.26 | 29.77 |
| CoTracker3 | 12.91 | 46.15 | 31.20 |
| **Ours** | **5.785** | **64.92** | **46.71** |

**推理效率（附录 Table 7）**：在单张 H100、50 步、164 帧、512×512 分辨率下单次生成耗时 49.1s；加 8-GPU 并行降到 11.8s；再加缓存并减到 16 步降到 2.3s（对应约 71 FPS 的骨干吞吐）。作为对比，TesserAct（5B，49 帧、480×640）需 137.5s，DreamZero（14B，48 帧、176×320）需 5.7s，DreamZero-Flash（14B，2×GB200，1 步）仅 0.15s 但论文指出其激进的单步去噪会显著牺牲视频质量。

**其他定性结果**：真实 xArm 零样本 rollout（未见物体/环境）中，解码出的 3D 轨迹经点云回放验证可执行，且与强视频生成基线 Veo 3.1 的定性对比显示动作预测与生成的视觉运动一致；在 FR3M 房间数据集上对未见物体/任务/环境的组合泛化测试中优于 LTX-2-Fast；附录还展示了模型可以对 π0 演示视频、甚至 Genie-3 生成的人手视频做动作标注，显示该表征具有跨数据集、跨具身的一定迁移能力。

## 四、局限性

论文明确指出的局限：当前系统只验证了强开环（single forward prediction，无在线重规划）结果，**尚未发展成闭环策略**；作者计划通过扩散蒸馏/加速技术压缩推理延迟，并将其接入闭环控制管线。

结合实验数据可以看到的其他局限：
- 几何解码（无动作头）在域内任务上比加了轻量 MLP 头低 16 个百分点（20.6% vs 36.7%），说明"零额外模块即为完整策略"的叙事在精度敏感任务上打了折扣，实用中很可能仍需要某种形式的轻量后处理/精修。
- 训练/推理成本高昂：训练用 32×A100、100k 步；即便叠加系统级优化，最快的单次生成仍需 2.3 秒（且以牺牲去噪步数为代价），离真正的高频闭环控制仍有明显差距，目前只能支撑单次预测的开环执行。
- 表征天然依赖多视角与相机标定：真实部署需要至少两路已知外参/内参的相机（或用 VGGT 现场估计），比起直接输出关节/末端位姿的策略头引入了额外的几何依赖和潜在误差源。
- 零样本真实机器人评测仅在单一 xArm 平台、5 个任务上进行，域内 RLBench 也只覆盖 9 个任务；尚未在更大规模、多本体的基准（如 RoboCasa、真实家庭环境）上验证。
- 论文未与最相关的同类工作 DreamZero（同样主打零样本策略、跨具身迁移）做成功率层面的直接对比，只在推理速度表里出现，方法层面的优劣缺少定量印证。

## 五、评价与展望

**优点**：把动作表征直接"降维"到与观测同构的像素空间，是一个简洁且工程上颇具吸引力的想法——它让预训练视频模型的先验知识可以被策略学习直接复用，而不必经过一个专门训练的、容易成为泛化瓶颈的动作头。零样本结果（尤其是强分布偏移下的真实机器人任务）对比一众强基线（π0.5、MolmoAct、TesserAct、Cosmos-Policy）优势明显，且同一套骨干、同一套掩码机制自然涌现出视频-动作联合生成、动作条件视频生成、视频到动作标注、纯视频生成四种能力，体现了统一表征的可扩展性；附录中对人手视频、Genie-3 生成视频的动作标注实验也为该表征的跨域迁移潜力提供了佐证。

**与同类公开工作的关系**：与 DreamZero、TesserAct 等依赖独立动作模块/动作头的世界动作模型相比，本文的核心差异化卖点是"动作原生于像素"；与 Cosmos-Policy 等在隐空间做动作编码的方法相比，像素级 3D 语义点+多视角三角化的设计提供了更强的空间可解释性和跨视角一致性。论文也提到了一项并发工作（Multi-view Video Diffusion Policy，arXiv 2604.03181）同样探索了视频空间下的策略表征，但据作者陈述其未对完整 7-DoF 动作做像素级编码——这类并发工作的出现说明"动作即视频"正成为一个被多个团队同时探索的方向，值得后续做更系统的横向比较。

**开放问题与可能的改进方向**：(1) 如何在不牺牲"无额外动作头"这一卖点的前提下缩小几何解码与学习头之间 16 个百分点的差距，例如探索更高分辨率热力图、可学习的射线采样或端到端可微分解码；(2) 闭环化和推理加速是论文自陈的下一步，扩散蒸馏/一致性模型等技术能否在不明显牺牲动作精度的前提下把延迟压到闭环可用的量级，是该路线能否走向真实部署的关键；(3) 目前的多视角依赖对硬件部署提出了更高要求，能否通过单视角+多视角联合训练、测试时用更少视角做退化推理，是拓展适用场景的一个自然方向；(4) 与 DreamZero 等最相关方法在成功率层面的直接、可复现对比仍然缺失，是评估该表征相对优势的重要空白。

## 参考

1. Ye, Y. et al. *World action models are zero-shot policies.* arXiv:2602.15922 (2026)（DreamZero，最相关的对比方法，采用独立动作模块）
2. Zhen, H. et al. *Tesseract: learning 4D embodied world models.* arXiv:2504.20995 (2025)（TesserAct，本文主要基线之一，与本文第一作者存在署名重叠）
3. Kim, M.J. et al. *Cosmos policy: Fine-tuning video models for visuomotor control and planning.* arXiv:2601.16163 (2026)（Cosmos-Policy，本文主要基线之一）
4. Intelligence, P. et al. *π0.5: A vision-language-action model with open-world generalization.* arXiv:2504.16054 (2025)（π0.5，VLA 类基线）
5. Li, P. et al. *Multi-view video diffusion policy: A 3D spatio-temporal-aware video action model.* arXiv:2604.03181 (2026)（并发工作，同样探索视频空间下的策略表征）
