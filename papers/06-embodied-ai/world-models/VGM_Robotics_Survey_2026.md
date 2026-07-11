# VGM-Robotics Survey：机器人学中的视频生成模型 —— 应用、研究挑战与未来方向

> **论文**：*Video Generation Models in Robotics: Applications, Research Challenges, Future Directions*
>
> **作者**：Zhiting Mei\*、Tenny Yin\*、Ola Shorinwa、Apurva Badithela、Zhonghe Zheng、Joseph Bruno、Madison Bland、Lihan Zha、Asher Hancock、Jaime Fernández Fisac、Philip Dames、Anirudha Majumdar（\* 共同一作）
>
> **机构**：Princeton University（普林斯顿大学，IRoM Lab）、Temple University（天普大学）
>
> **发布时间**：2026 年 01 月（arXiv 2601.07823）
>
> **发表状态**：未录用（预印本，eess.SY）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.07823) | [PDF](https://arxiv.org/pdf/2601.07823)
>
> **分类标签**：`world-models` `video-generation` `robot-manipulation` `survey` `visual-planning`

---

## 一句话总结

这是一篇系统梳理**视频生成模型作为具身世界模型（embodied world models）在机器人学中应用**的综述（40 页、292 篇引用），把散落的工作归纳为四大应用方向——模仿学习中的数据生成/动作预测、强化学习中的动态与奖励建模、可扩展的策略评测、视觉规划——并逐条剖析物理幻觉、指令跟随差、长视频、数据/算力成本等阻碍其"可信部署"的开放挑战。

## 一、问题与动机

机器人算法长期依赖**环境模型（world model）**来高效学策略，而传统两条路都有硬伤：

- **语言模型（VLM/LLM）路线**：语言这一抽象天生"表达力不足"，难以精细刻画接触/形变等物理交互过程（例如描述夹爪与一块布的细粒度接触），也难以准确建模真实世界事件之间的时空依赖。
- **物理仿真器路线**：为了计算可行性，物理引擎普遍用简化假设（原始几何体近似、简化动力学），带来严重的 sim-to-real gap，且资产制作（asset curation）昂贵、难以模拟复杂形变体。

视频生成模型（尤其是 diffusion/flow-matching 类）通过在互联网规模数据上训练，学到了**照片级、物理上更一致的时空世界模型**，天然绕开上述两个瓶颈：它能合成高保真、可由文本/图像/动作/轨迹多模态条件驱动的视频,把细粒度物理交互直接"画"出来。作者据此论证——视频模型正成为机器人领域的通用具身世界模型，值得一篇专门面向 robot manipulation 的综述（区别于此前偏向自动驾驶或泛世界模型的综述）。

## 二、核心方法（综述的知识骨架）

### 2.1 两类学习式世界模型

**Markovian 状态式世界模型**：假设未来只依赖当前状态 $s_t$ 与动作 $a_t$，训练一个动态预测器：

$$
s_{t+1} \sim p_\eta(s_{t+1} \mid s_t, a_t)
$$

通常在隐空间运作，由编码器、动态预测器、奖励预测器三件套组成：

$$
s_t \sim \mathcal{E}_\gamma(s_t \mid o_t), \quad \hat{s}_{t+1} \sim p_\eta(\hat{s}_{t+1} \mid s_t, a_t), \quad \hat{r}_{t+1} \sim p_\zeta(\hat{r}_{t+1} \mid \hat{s}_{t+1})
$$

**用大白话说**：先把观测压成一个"状态向量",再学"给定状态和动作、下一步状态会变成啥",顺带预测能拿多少奖励——这就是 Dreamer/RSSM 这一系的经典配方。

**视频世界模型**：不显式建 Markovian 状态,而是直接学**时空映射**,把整段视频的像素/patch 随时间演化。演进脉络（综述梳理得很清楚）：早期视频预测（局部空间变换）→ GAN（保真度高但 mode collapse）→ VAE/VQ-VAE（用离散码本解决自回归解码器的 latent collapse）→ 视频 Transformer → 扩散/流匹配。

### 2.2 扩散/流匹配视频模型

前向加噪的闭式边缘分布：

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t; \sqrt{\bar\alpha_t}\,\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I}\big)
$$

训练目标（预测噪声，或等价的 velocity 参数化 $\boldsymbol{v}=\sqrt{\bar\alpha_t}\boldsymbol{\epsilon}-\sqrt{1-\bar\alpha_t}\mathbf{x}_0$）：

$$
L_\epsilon = \mathbb{E}_{\mathbf{x}_0,t,\boldsymbol{\epsilon}}\big[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\big]
$$

可控性靠 **classifier-free guidance**（已成主流），推理时用引导尺度 $w$ 混合有/无条件预测：

$$
\tilde\epsilon_\theta(x_t, t, y) = (1+w)\,\epsilon_\theta(x_t, t, y) - w\,\epsilon_\theta(x_t, t)
$$

**用大白话说**：扩散就是"学会把纯噪声一步步去噪还原成真实视频";guidance 则是推理时把"带条件的预测"往外多推一点、把"不带条件的"减掉,让生成结果更贴合文本/动作提示。

**架构**：U-Net（2D 卷积升到 3D 加时序注意力，Stable Video Diffusion 的数据/预训练配方被开源界广泛沿用）与 DiT（把视频切 patch 成 token，SOTA 模型如 Wan/Cosmos 多用 DiT + 专用视频编码器做时序压缩）两大主干。**条件注入**三种机制：channel concatenation（像素对齐的深度/位姿条件）、cross-attention（文本这类语义非空间条件）、adaptive normalization（AdaLN/FiLM，做帧率/运动强度这类全局标量控制）。条件模态覆盖 T2V、I2V、以及 motion/trajectory 引导（ControlNet、直接注入 joint 位置/力矩等 embodiment-specific 低层状态）。

### 2.3 V-JEPA 类（隐空间预测，不画像素）

与"画像素"的扩散模型相对，V-JEPA 走自监督对比路线（InfoNCE / BYOL / DINO 传承），在隐空间预测被 mask 的时空特征：

$$
\min_{\theta,\phi,\Delta_y} \big\|P_\phi(\Delta_y, E_\theta(x)) - \mathrm{sg}(E_{\bar\theta}(y))\big\|_1
$$

其中 $\mathrm{sg}(\cdot)$ 为 stop-gradient，$E_{\bar\theta}$ 为 $E_\theta$ 的 EMA。**用大白话说**：不费劲还原每个像素，只在抽象特征层面"猜"看不见的部分,从而逼模型学高层语义、构建内部世界模型;V-JEPA 2 更能在训练中条件于机器人动作直接用于规划。代价是仍易 representation collapse。

### 2.4 隐式 vs 显式视频世界模型

- **隐式（implicit）**：3D 场景仅编码在视频模型内部，只能靠"生成视频"来可视化。代表：Pandora（微调 DynamiCrafter 支持长时长文本条件控制）、FreeAction、**Vid2World**（把 DynamiCrafter 扩成因果自回归、动作条件，用 diffusion forcing）。
- **显式（explicit）**：用视频模型重建出**具体 3D 表示**。代表：**Enerverse**（多视角视频 + 训 4D Gaussian Splatting 做新视角合成）、**Aether**（微调 CogVideoX 生成 depth/camera raymap 再反投影得 3D）、**Genie Envisioner**（视频生成 + action decoder，可做动作预测/评测/数据生成）。

## 三、四大应用方向（综述主体）

| 应用方向 | 核心思路 | 代表工作 |
|---|---|---|
| ① 模仿学习数据生成 + 动作预测 | 微调视频模型生成专家演示，再从视频反解动作 | Cosmos Predict / Wan（数据生成器）、DreamGen、VPP、Vidar、ARDuP |
| ② RL 动态 + 奖励建模 | 视频模型当高保真 dynamics / 稠密 reward | Dreamer 4、World-Env、VIPER、Diffusion Reward |
| ③ 可扩展策略评测 | 视频模型闭环 rollout 估真实成功率 | WorldGym、Ctrl-World、Veo World Simulator |
| ④ 视觉规划 | 生成"完成任务的图像序列"当子目标/参考轨迹 | Visual Foresight、FLIP、UniPi、NovaFlow、MindJourney |

**① 数据生成与动作预测**：动作反解分两路（Figure 5）。**End-to-end** 用 latent action model（VQ-VAE 无监督推两帧间隐动作，需少量带标数据对齐）或 inverse-dynamics model（IDM，监督式从带动作视频学，如 DreamGen 训 IDM）。**Modular** 靠位姿追踪 / 单目深度 / 光流 / CAD 模型估目标物位姿再 retarget 到机器人（AVDC、VideoAgent），可零样本部署。此外**视频模型直接当策略骨干**：统一 video-action 方法联合预测视频与动作——GR-1（自回归 Transformer）、RPT（masking）、UVA（学视频+动作联合隐表示）、UWM、DreamVLA（额外监督 depth/动态区域/语义特征）、UniVLA（加语言 token 监督）。

**② RL**：Dreamer 4 从零训动作条件视频模型当 dynamics predictor,在 Minecraft 上 RL 微调策略;World-Env 用预训动作条件视频模型 + VLM 当 reward reflector;VIPER 用视频预测似然当奖励,Diffusion Reward 用生成分布的条件熵当奖励（越接近专家轨迹熵越低）。

**③ 策略评测**：真机评测因硬件/人力成本高、需覆盖组合爆炸的环境而昂贵。视频模型可对采样初始观测 + 任务指令做闭环 rollout 估成功率。改进包括条件于策略隐动作而非物理动作、多视角一致生成、纳入历史帧缓解误差累积。**关键指标**：Pearson correlation coefficient（预测与真实成功率的线性相关）、Mean Maximum Rank Violation（MMRV，衡量策略排序不一致）。Veo World Simulator 还能靠图像编辑快速构造 OOD 场景做鲁棒性/安全评测。

**④ 视觉规划**：**Action-guided**（三步：生成动作提案 → 视频模型当 dynamics 合成轨迹 → 按目标函数评分），如 Visual Foresight 用 cross-entropy method、FLIP 用条件 VAE + value network、MindJourney 用 VLM 提相机轨迹；**Action-free**（直接把生成帧当图像子目标）,如 CLOVER 训 IDM 映射当前观测+子目标到动作、UniPi、NovaFlow（从生成视频提物体/粒子位姿当粗轨迹再用粒子动力学优化）。

## 四、评测体系（综述整理）

**指标**：帧级（PSNR、SSIM、CLIP score、Inception Score、FID、LPIPS）+ 时空级（FVD、KVD、FVMD、光流一致性、VLM 判物理一致性）。

**基准**：

| 类别 | 基准 | 侧重 |
|---|---|---|
| 综合质量 | WorldModelBench、EvalCrafter、VBench（16 项细粒度）、EWMBench、T2V-CompBench、WorldSimBench、PAI-Bench | 指令跟随、视觉/时序/美学质量、物理粘合度 |
| 物理常识 | Physics-IQ、PhyGenBench（27 条物理定律、160 prompt）、VideoPhy、VP² | 光学/热力学/引力/摩擦、物体交互、规划物理对齐 |

综述强调：**这些基准普遍显示,即便 scale 上去,视频模型仍难遵守物理定律**（美学与时序一致性随 scale 改善，但物理不然）；VP² 更指出模型/数据 scale 带来的增益很快 plateau。

**几个可复现的量化事实**（原文明确给出）：

| 事实 | 数值 |
|---|---|
| Open-Sora 2.0 训练成本 | 约 \$200k（当前最省的开源 SOTA 之一） |
| Veo 3 推理速度 | 约 12 帧/秒（单张 NVIDIA A100） |
| 长视频时长上限 | Veo 3.1 约 8 秒、Wan 2.5 约 10 秒（远不够分钟级机器人任务） |
| Dreamer 4 提速 | 时序注意力稀疏化到每第 4 帧、解耦时空注意力 |

## 五、开放挑战（综述最有价值的部分，共 10 条）

1. **物理幻觉与违反物理**：物体凭空出现/消失/穿模，违反牛顿定律、能量/动量/质量守恒（如倒水但杯中液面不变）；模型倾向"照抄最近训练样本",按 color→size→velocity→shape 顺序迁移物体属性;prompt engineering 与 scaling 都解决不了,需新架构/训练法。未来方向：Hamiltonian/Lagrangian 先验、PhysGen/WonderPlay 引入物理仿真、affordance/hotspot 条件。
2. **不确定性量化（UQ）**：视频生成不满足标准 Bayesian UQ 的 i.i.d. 假设，帧间强相关；模型无法"表达置信"。S-QUBED（语义空间 T2V UQ）、C³（隐空间稠密逐 patch 置信度）。
3. **指令跟随差**：常只部分执行/完全忽略指定动作，难生成视频内文字、难控相机运动。ViMi、Aid、InteractiveVideo、ATI 用多模态条件改善。
4. **评测缺统一框架**：现有指标只测感知质量或语义一致，机器人更看重物理一致与预测精度；常被迫用下游任务成功率相关性做代理度量。
5. **安全内容生成**：视频模型易生成犯罪/暴力/误导内容，SAFEWatch、T2VSafetyBench 等 guardrail 仍任务特定、覆盖窄。
6. **安全机器人交互**：需物理安全（避碰）+ 语义安全（别朝人扔利器）；现有安全 filter 多限于 Markovian 状态空间，扩到视频世界模型的时空隐空间仍是难题。
7. **动作估计**：latent action model 受码本大小限制、难解释、需真机数据微调；IDM 需大量带标数据、泛化差。
8. **长视频生成**：MALT（记忆向量压缩）、FramePack（缓漂移）、TTTVideo/LaCT（test-time training）、LCT（扩上下文窗）、MoC（稀疏检索）、Diffusion Forcing（独立噪声级、变长时域）、NUWA-XL（diffusion-over-diffusion）——仍难稳定生成分钟级视频。
9. **数据制作成本**：WebVideo-10M/Panda-70M 重规模轻质量（字幕不准、模糊、乱切镜头）；VidGen-1M/OpenVid-1M 用质量分预过滤。机器人场景**必须含失败演示**,否则生成视频有"乐观偏差"（把物体挪到易抓位置、高估抓取可行性、漏建障碍物），真机执行必败。
10. **训练/推理成本**：训练动辄数十万美元、限于大团队；Veo 3 这类推理慢，难满足闭环实时反馈。Dreamer 4 稀疏时序注意力、OpenSora 深压缩自编码器、shortcut/consistency 模型、量化/蒸馏是提速方向。

## 六、评价与展望（学术视角）

**优点**：

- **组织清晰、机器人导向**：区别于此前偏自动驾驶或泛世界模型的综述，本文把落点牢牢放在 robot manipulation，四大应用 + 十条挑战的划分对进入该领域的研究者极友好，Figure 2 的 taxonomy 树几乎是一张领域地图。
- **覆盖新且全**：引用直到 2025 年底的工作（Veo 3、Wan 2.5、Gemini 3、Dreamer 4、V-JEPA 2、Genie Envisioner、DreamVLA 等），对 latent action model vs IDM、隐式 vs 显式世界模型、action-guided vs action-free 规划等对立范式的两两对比很到位。
- **挑战部分务实**：不只罗列，每条都配"未来方向"与代表方法，尤其"生成数据必须含失败样本以免乐观偏差"这一点，切中数据合成用于策略训练的要害。

**局限（作为综述）**：

- **无原创方法/实验，缺横向定量对比表**：作为综述这可以理解，但读者拿不到"同一基准上各方法数字排名",难判断谁 SOTA;所引数字多为零散事实（\$200k、12 fps、8/10 秒）而非系统对比。
- **V-JEPA 类着墨偏少**：全文重心在 diffusion/flow 的"画像素"路线，对"不画像素"的隐空间预测路线（V-JEPA 2 已能规划）分析较浅，而后者恰是绕开物理幻觉与推理成本的一条潜在捷径,值得更平衡的讨论。
- **实证结论多为转述**："scale 救不了物理一致性""失败数据不可或缺"等关键论断依赖所引 benchmark，本文未做二次验证或元分析，读者需回溯原文。

**与其他公开工作的关系**：本文可视为 [62]（*Video as the new language for decision making*）在机器人操作这一垂直领域的细化与更新，也与聚焦自动驾驶的世界模型综述（[65][66][67]）、泛世界模型综述（[58]-[61][64]）形成互补。其价值不在提出新方法，而在为"视频模型 = 具身世界模型"这一正在爆发的方向立一个可信、可检索的坐标系。

**开放问题与可能改进方向**：

1. **物理一致性的内生化**：与其事后用 VLM/仿真器打补丁（本身也会幻觉），不如探索把物理定律作为归纳偏置写进训练目标/架构（Hamiltonian/Lagrangian 神经网络、affordance hotspot 条件）。
2. **长时域 + 实时的联合突破**：机器人任务分钟级，而 SOTA 仅 8-10 秒且推理慢；如何在扩上下文窗（LCT/MoC）的同时压推理成本（Diffusion Forcing + 稀疏注意力 + 蒸馏）是落地闭环控制的卡点。
3. **带校准保证的 UQ**：现有 UQ 只在训练分布内校准，OOD（恰是安全关键场景）失效；可证明保证的时空 UQ 是安全部署前提。
4. **面向机器人的评测**：亟需能定量、高效、任务相关地评"细粒度操作视频质量"的机器人中心基准，并把生成视频的物理一致性与其对应 3D 重建的运动做交叉核验。

## 参考（最相关的 5 篇）

1. Sherry Yang et al. *Video as the New Language for Real-World Decision Making*, 2024（arXiv 2402.17139）—— 本综述的思想母题。
2. Bruce et al. *Genie: Generative Interactive Environments*, ICML 2024 —— latent action + 可控视频世界模型开创性工作。
3. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025（arXiv 2506.09985）—— 隐空间预测路线的代表。
4. Jang et al. *DreamGen: Unlocking Generalization in Robot Learning through Video World Models*, 2025（arXiv 2505.12705）—— 视频生成数据 + IDN 反解动作的代表。
5. Fang et al. *WorldModelBench: Judging Video Generation Models as World Models*, 2025（arXiv 2502.20694）—— 面向世界模型能力（指令跟随/物理粘合/常识）的评测基准。
