# World2Act：基于世界模型动力学的隐动作空间后训练

> **论文**：*World2Act: Latent Action Post-Training from World Model Dynamics*
>
> **作者**：An Dinh Vuong*, Tuan Van Vo*, Abdullah Sohail, Haoran Ding, Liang Ma, Xiaodan Liang, Anqing Duan, Ivan Laptev, Ian Reid（*共同一作）
>
> **机构**：Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
>
> **发布时间**：2026 年 03 月（arXiv 2603.10422）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.10422) | [PDF](https://arxiv.org/pdf/2603.10422)
>
> **分类标签**：`世界模型` `VLA` `隐动作后训练` `对比学习` `残差策略` `GR00T-N1.6` `Cosmos-Predict2`

---

## 一句话总结

World2Act 用双向 InfoNCE 对比学习把冻结世界模型（Cosmos-Predict2）的视频隐变量与冻结 VLA（GR00T-N1.6）的动作隐变量对齐进同一个低维共享空间，再只训练一个轻量残差策略在该空间里对动作隐变量做修正，从而绕开"世界模型想象 rollout→解码像素→逆动力学伪标签"这条容易被视觉伪影污染的传统链路；在 RoboCasa/LIBERO/Bridge-SIMPLER 三个仿真基准上分别取得 **72.6%/98.1%/59.0%** 的成功率（较微调基线 +2.5/+1.1(相对DreamGen)/+1.4 个百分点，且在 LIBERO 上 DreamGen 的像素级后训练反而把基线从 97.0% 拖到 92.1%），真机三任务平均成功率从约 13% 提升到约 20%（+6.7 个百分点）。

## 一、问题与动机

- 世界模型（WM）能提供物体-机器人交互如何演化的动力学先验，是后训练 VLA 策略、弥补行为克隆泛化不足的一条有前景路线；但现有 WM 后训练方法（如 DreamGen、VLA-RFT、Ctrl-World）几乎都依赖**像素空间监督**：先用 WM 生成（想象）视频 rollout，再用逆动力学模型（IDM）从解码后的像素帧里反推伪动作标签，或从像素 rollout 里计算奖励。
- 问题在于生成视频天然存在视觉伪影（visual artifacts）：结构性幻觉（如论文附录展示的杯子上多长出一个把手）、多视角不一致（LIBERO 中杯子在主视角出现但腕部视角消失）、精细部件缺失（真机实验中门把手渲染缺失）。这些伪影一旦被解码成像素再喂给 IDM，就会直接污染动作伪标签，论文用受控实验证明：把 IDM 的输入从真实视频换成 WM 解码 rollout，在 LIBERO-Object 50 个留出场景上，动作 MSE 从 0.082 升到 0.097（+18%），MAE 从 0.167 升到 0.199（+19%）——这正是 DreamGen 在 LIBERO 上让基线成功率不升反降（97.0%→92.1%）的具体成因。
- 核心论点：既然 WM 的隐变量（VAE latent，解码前）本身携带的时序动力学信息是可靠的，只是解码到像素这一步引入了噪声，那么后训练信号应该直接在隐空间里对齐，完全跳过解码-重建-逆动力学这条易碎的链路。

## 二、核心方法

World2Act 是一个两阶段后训练框架：世界模型 $\mathcal{W}$（Cosmos-Predict2）与 VLA 骨干（GR00T-N1.6）全程冻结不改，只训练两组轻量适配器/策略。

**预备知识：指令条件视频世界模型。** 给定初始观测 $s_1$ 和语言指令 $\ell$，$\mathcal{W}$ 通过 flow matching 训练的去噪网络在结构化的 video-VAE 隐空间里直接预测未来视觉动力学：

$$
\mathbf{V} = \mathcal{W}_{\text{denoise}}(\epsilon; s_1, \ell), \qquad \mathbf{V}=\{V_t\}_{t=1}^{T},\quad V_t \in \mathbb{R}^{C\times H\times W}
$$

用大白话说：世界模型从纯噪声出发，一步步去噪出一段未来视频在 VAE 隐空间里的"压缩表示"序列，每个 $V_t$ 对应一小段（$M$ 帧）不重叠的视频 chunk。World2Act 始终只用这个解码前的隐变量 $\mathbf V$ 作为后训练目标，从不把它渲染成 RGB 像素。

**阶段一：视频-动作隐空间对齐。** 给定同步的状态-动作专家演示，世界模型编码演示视频得到隐轨迹 $\mathbf V^{\text{gt}}=\{V_t^{\text{gt}}\}$；一个 CNN 视频适配器 $\mathcal B_{\mathrm v}$ 把每个 $V_t^{\text{gt}}$ 映到 $D$ 维嵌入 $z_t^{\mathrm v}$，一个 MLP 动作适配器 $\mathcal B_{\mathrm a}$ 把对应的低层动作 chunk $\bar a_t^{\text{gt}}$（$M$ 步动作）映到同一空间的 $z_t^{\mathrm a}$。为保证隐变量仍能解回可执行动作，额外训练一个动作解码器 $\mathcal D_{\mathrm a}$，用重建损失 $\mathcal L_{\text{recon}}=\lVert \mathbf a_{\text{gt}}-\hat{\mathbf a}\rVert^2$ 约束。核心是块级（chunk-aware）双向 InfoNCE 对比损失，batch 内正样本为同一演示的时间对齐视频-动作对，负样本包含同任务不同演示（难负例，占比 0.25）与不同任务样本：

$$
\mathcal{L}_{\text{contrastive}} = -\log\frac{\exp(\mathrm{sim}(\mathbf{z}_i^{\mathrm v},\mathbf{z}_i^{\mathrm a})/\tau)}{\sum_{j=1}^{B}\exp(\mathrm{sim}(\mathbf{z}_i^{\mathrm v},\mathbf{z}_j^{\mathrm a})/\tau)} -\log\frac{\exp(\mathrm{sim}(\mathbf{z}_i^{\mathrm a},\mathbf{z}_i^{\mathrm v})/\tau)}{\sum_{j=1}^{B}\exp(\mathrm{sim}(\mathbf{z}_i^{\mathrm a},\mathbf{z}_j^{\mathrm v})/\tau)}
$$

其中 $\mathrm{sim}(\mathbf z_i,\mathbf z_j)=\frac{1}{T}\sum_{t=1}^{T}\cos(z_{i,t},z_{j,t})$ 是块级平均余弦相似度，而非把整段轨迹压成单一全局向量再比较。用大白话说：让同一段演示对应的"世界模型看到的未来视频"和"专家实际做的动作"在共享空间里越靠越近，不同演示/不同任务的则推远；用逐 chunk 平均相似度而非整段轨迹的单一向量，是为了强制模型学到细粒度的时序对应关系，而不是抄任务身份或背景这种"捷径"。总损失为 $\mathcal L=\mathcal L_{\text{recon}}+\mathcal L_{\text{contrastive}}$，训练 30K 步（batch size 16），完成后 $\mathcal B_{\mathrm v},\mathcal B_{\mathrm a},\mathcal D_{\mathrm a}$ 全部冻结。

**阶段二：世界模型引导的隐空间残差后训练。** 借鉴 Silver et al. 的残差策略学习思路，World2Act 不直接微调参数量巨大、容易灾难性遗忘的 VLA 骨干 $\pi_{\text{base}}$，而是冻结它，只训练一个轻量残差策略 $f^\theta$，在阶段一学到的共享隐空间里对动作隐变量做加性修正：

$$
\mathbf{z}_{\text{base},t}^{\mathrm a}=\mathcal{B}_{\mathrm a}(\bar{\mathbf a}_{\text{base},t}),\qquad \Delta\mathbf{z}_t^{\mathrm a}=f^\theta(s_t,\mathbf{z}_{\text{base},t}^{\mathrm a}),\qquad \mathbf{z}_{\text{final},t}^{\mathrm a}=\mathbf{z}_{\text{base},t}^{\mathrm a}+\Delta\mathbf{z}_t^{\mathrm a}
$$

用大白话说：VLA 先照常给出一个基础动作块 $\bar a_{\text{base},t}$，映射到隐空间后，一个小 Transformer（自注意力 2 层、4 头）根据当前视觉/本体观测再"打个补丁" $\Delta z_t^{\mathrm a}$，补丁与基础隐变量相加后经冻结的 $\mathcal D_{\mathrm a}$ 解码回可执行动作块 $\bar a_{\text{final},t}$，开环执行 $M$ 步后再重新查询。训练信号来自：在 $B$ 个并行仿真环境中用当前增强策略 rollout 得到动作隐轨迹 $\mathbf z_{\text{final}}^{\mathrm a}$，同时世界模型基于同一初始观测和指令想象出对应的视频隐轨迹 $\mathbf z^{\mathrm v}$，两者之间同样用式 (2) 的对比损失对齐（同一 rollout 内为正对，不同 rollout 为负对），梯度只回传给 $f^\theta$，不使用仿真器的奖励、成功率或可微状态转移信号——即完全不需要解码到像素，也不需要环境奖励，是纯粹的隐空间弱监督。

**后训练数据合成。** 用一个在自建"原子技能"数据上微调过的 Cosmos-Predict2（称为 Skill-WM）为 RoboCasa/LIBERO/Bridge-SIMPLER 各采样约 1000 个场景初始化（合计约 3000 条），后训练场景与评测场景互斥以防测试泄露。合成流程：先用 DeepSeek 把高层指令拆成有序的原子技能提示词，Skill-WM 逐段生成短视频并以上一段末帧为条件自回归拼接成长时程想象 rollout，只保留解码前的隐变量作为目标轨迹。原子技能数据本身通过夹爪开合信号自动切分演示、再用 LLM 按预定义技能 schema 对齐时间戳获得（RoboCasa 同步率 96.2%，LIBERO 86.9%），使 RoboCasa 训练片段从 67,593 条增至 114,192 条、LIBERO 从 2,007 条增至 11,782 条，长视频的时长分布也从长尾变为更集中的单峰分布，缓解了世界模型对长时程生成的不稳定性。

## 三、实验结果

**主基准（GR00T-N1.6-ft 为骨干，"+World2Act" 为本文方法）**

| 基准 | GR00T-N1.6（原始） | GR00T-N1.6-ft（微调） | +DreamGen | +World2Act（Ours） |
|---|---|---|---|---|
| RoboCasa（avg SR） | 66.2% | 70.1% | 70.5% | **72.6%** |
| LIBERO（avg SR） | — | 97.0% | 92.1%（退化） | **98.1%** |
| Bridge-SIMPLER（7 任务 avg SR） | — | 57.6% | 58.3% | **59.0%** |

World2Act 相对 GR00T-N1.6-ft 分别 +2.5pp（RoboCasa）、+1.1pp（LIBERO）、+1.4pp（Bridge-SIMPLER）；相对未微调的 GR00T-N1.6 在 RoboCasa 上 +6.4pp，相对同样具备 WM 结构的 Cosmos Policy 基线（65.7%）+6.9pp。LIBERO 上 DreamGen 的像素级伪标签反而使成功率从 97.0% 掉到 92.1%，是论文核心论点（像素监督不稳）的直接实证；World2Act 在 LIBERO 上距离最强已报告结果 Cosmos Policy+World2Act（98.6%）仅差 0.4pp，同时也超过 $\pi_{0.5}$、OpenVLA-OFT、CoWVLA、UniVLA 等基线。

**后训练方法横向对比（Table 4，RoboCasa，同以 GR00T-N1.6-ft 为起点）**

| 方法 | RoboCasa SR |
|---|---|
| GR00T-N1.6-ft（无后训练） | 70.1% |
| +World2Act w/ BC（对比目标换成监督 BC，消融） | 70.4% |
| +DreamGen | 70.5% |
| +VLA-RFT | 71.0% |
| +Ctrl-World | 69.8% |
| **+World2Act（Ours）** | **72.6%** |

**适配器与对比目标消融（Table 5，RoboCasa）**

| 变体 | 训练耗时 | SR |
|---|---|---|
| 单向 InfoNCE + chunk 相似度 | – | 72.0% |
| Marginal 对比 + chunk 相似度 | – | 70.7% |
| 双向 InfoNCE + 全局轨迹相似度 | – | 69.3% |
| LoRA（$r=16$）微调 VLA | 14.6h | 71.4% |
| LoRA（$r=32$）微调 VLA | 15.3h | 72.1% |
| **本文：残差策略 + 双向 InfoNCE + chunk 相似度** | **6.8h** | **72.6%** |

残差策略训练耗时仅为 LoRA($r=32$) 的 1/2.25（6.8h vs 15.3h），成功率反而更高；双向对比和逐 chunk 时序对齐两个设计都不可或缺，把 chunk 级相似度换成粗粒度全局轨迹相似度会使成功率降至 69.3%，说明细粒度视频-动作时序对齐是收益的关键来源。

**动作空间误差与跨任务泛化**

| 指标 | GR00T-N1.6-ft | +World2Act |
|---|---|---|
| 动作 MSE（RoboCasa Human split，3 任务均值） | 0.034 | **0.021**（↓38.2%） |

在 RoboCasa 24 任务中划分 12 seen/12 unseen（held-out 为 Pick-and-Place 系列任务），随 seen 任务数增至 12，World2Act 在 unseen 任务上比对应原始策略提升 +2.4pp（GR00T-N1.6-base）/+1.1pp（Cosmos Policy），且未经任务特定微调，说明学到的是可迁移的动力学先验而非任务记忆。

**真实机器人（Franka Research 3，3 任务：拿杯放盘、拿碗、关微波炉；每任务 20 条演示 + 100 条 WM 生成轨迹，20 次 rollout/任务）**

World2Act 相对 GR00T-N1.6-ft 平均成功率 +6.7pp（约 13%→20%），相对 +DreamGen +8.3pp（约 12%→20%），定性上即便 WM 想象序列末帧出现把手消失等伪影，真机执行仍能成功完成任务，验证了隐空间对齐对像素伪影的鲁棒性。

**推理速度（RoboCasa，RTX 4090）**：GR00T-N1.6-ft 274.1Hz（3.6ms/步）→ +World2Act 251.9Hz（4.0ms/步）；Cosmos Policy 20.8Hz（48.1ms）→ +World2Act 20.5Hz（48.8ms），额外开销极小，仍能维持 250Hz 以上闭环控制。

## 四、局限性

论文第 5 节明确列出三点：

1. **接触密集、非抓取式任务仍是难点**：这类任务的接触动力学复杂，物体运动或接触演化中的小误差会在 WM rollout 里累积放大，使动力学先验的可迁移性下降；
2. **真机成功率仍然偏低**：即便加上 World2Act，三任务平均成功率也只有约 20%，表明当前 WM 与 VLA 都尚未充分捕捉复杂真实物理动力学，存在持续的 sim-to-real 域差距；
3. **chunk 级时间对应关系是一个较强假设**：当前方法依赖想象轨迹与实际执行轨迹之间的块级时序对齐，作者认为放松这一对齐约束、支持更灵活的执行动力学是未来方向。

此外，附录 F 的失败案例分析显示 WM 想象与真实执行之间仍可能出现语义层面的落差：例如 Turn Off Stove 任务中，WM 成功想象出抓握并旋转炉灶旋钮的画面，但 VLA 在真实控制中未能牢固抓住旋钮而任务失败，作者将其归因于机器人运动学的刚性物理约束远比像素空间想象复杂，这也说明"隐空间对齐"虽规避了像素伪影问题，但并未完全解决想象到执行的语义/物理落地差距。

## 五、评价与展望

**优点。** World2Act 的核心贡献是一个诊断加一个解法：诊断出像素空间 WM 后训练（DreamGen 式"解码 rollout→IDM 伪标签"）的脆弱性根源在于视觉伪影会被 IDM 直接转译为噪声动作标签，并用受控实验（同一 IDM、只换输入视频源，MSE/MAE 各升 18%/19%）把这一因果链条钉实；解法则是把监督信号完全限制在解码前的 VAE 隐空间，用双向对比学习对齐视频与动作两种模态，再用一个轻量残差策略（而非 LoRA 全量微调）承接对齐信号。这种"冻结 WM + 冻结 VLA + 只训练两组适配器/一个小残差网络"的设计使方法在工程上很轻（6.8 小时训练、推理仅增加约 0.4ms/步），也天然避免了大模型后训练常见的灾难性遗忘问题。消融实验中"块级双向对比"相对"全局轨迹对比"（72.6% vs 69.3%）与"残差策略"相对"LoRA"（72.6%/6.8h vs 72.1%/15.3h）的双重优势，把增益来源拆解得比较干净。

**与相关工作的关系。** 论文附录 H 专门澄清了与三类最接近工作的区别：与 **Cosmos Policy**（联合学习视频-动作-价值嵌入空间的一体化架构）不同，World2Act 不训练新的联合 WM-策略架构，而是后训练一个已有 VLA，这也解释了为什么在已具备 WM 结构的 Cosmos Policy 上叠加 World2Act 时增益明显更小（RoboCasa 仅 +0.6pp、LIBERO 仅 +0.1pp）——这更像是对"隐空间对齐是否只是在补齐 Cosmos Policy 已经隐式学到的东西"这一问题的间接佐证，论文将其解读为"锦上添花式互补"，但也可以理解为该方法的边际收益天花板依赖基座 VLA 本身是否已具备类 WM 结构；与 **V-JEPA2**（通过潜空间目标驱动动作朝终态优化）不同，World2Act 不针对某个终态优化动作，而是把 WM 的逐步动力学轨迹作为分步监督目标；与 **CoWVLA**（用预训练视频 VAE 联合建模稀疏关键帧和动作 token，仍需协同训练大型 VLA 骨干）相比，World2Act 完全冻结 VLA 骨干，是更轻量的纯后训练范式。整体上，World2Act 延续了 latent action/latent video-action representation 一脉（如 UWM、Moto、AdaWorld 等）"用隐空间桥接视频与动作"的直觉，但把重点放在"后训练阶段的弱监督对齐"而非"预训练阶段的联合表征学习"，是该方向上一个偏工程实用、强调与现有 VLA 兼容性的变体。

**开放问题。** (1) 论文用 Skill-WM（在自建原子技能数据上微调的 Cosmos-Predict2）而非通用 Base-WM 取得最佳效果（Table 14：GR00T-N1.6-ft+World2Act 用 Base-WM 为 71.5%，用 Skill-WM 为 72.6%），但原子技能切分本身依赖夹爪开合启发式和 LLM 对齐，同步率在 LIBERO 上只有 86.9%，这层数据构建流程自身也存在噪声——与论文批评的"IDM 伪标签噪声"是同一类问题在训练侧的翻版，论文未讨论这层噪声上限对最终效果的敏感性；(2) 真机验证仅在单一 Franka 平台、3 个任务、20 次 rollout 上进行，样本量偏小，且成功率本身不高（约 20%），跨本体和更复杂真实任务上的可扩展性有待验证；(3) 阶段二的弱监督信号来自"想象轨迹与执行轨迹的对比对齐"，其质量隐式依赖阶段一学到的共享空间是否足够可靠，但论文没有给出对齐质量本身的失败模式分析（例如当 WM 想象的物理不合理时，对比目标是否会把残差策略带偏）；(4) 消融显示方法对 Cosmos Policy 这类已有 WM 结构的骨干增益有限，如何将隐空间对齐思路扩展到更大范围的 VLA 骨干、以及在增益饱和时能否叠加其他机制（如附录 F.3 提到的想象-执行语义落差问题）仍是待解决的方向。

## 参考

1. Jang et al., *DreamGen: Unlocking Generalization in Robot Learning through Video World Models*, CoRL 2025.（本文主要的像素空间后训练对比基线）
2. Kim et al., *Cosmos Policy: Fine-tuning Video Models for Visuomotor Control and Planning*, ICLR 2026.（WM-capable VLA 基线，附录 H 详细对比架构差异）
3. Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985, 2025.（潜空间目标驱动动作优化的相关工作）
4. GEAR Team, *GR00T N1.6: An Improved Open Foundation Model for Generalist Humanoid Robots*, NVIDIA, Dec 2025.（本文使用的冻结 VLA 骨干）
5. Agarwal et al., *Cosmos World Foundation Model Platform for Physical AI*（含 Cosmos-Predict2）, arXiv:2501.03575, 2025.（本文使用的冻结世界模型）
