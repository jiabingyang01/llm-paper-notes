# LaST-HD：从可规模化人手数据中学习潜在物理推理以用于机器人操作

> **论文**：*LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data for Robot Manipulation*
>
> **作者**：Jiaming Liu, Yinxi Wang, Chenyang Gu, Siyuan Qian, Xiangju Mi, Hao Chen, ... Hao Tang, Shanghang Zhang（通讯作者）et al.
>
> **机构**：北京大学 多媒体信息处理国家重点实验室 / 计算机学院；香港中文大学；Simplexity Robotics；Aether Tech
>
> **发布时间**：2026 年 06 月（arXiv 2606.23685）
>
> **发表状态**：未录用（预印本，标注为 "LaST-HD Technical Report"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.23685) | [PDF](https://arxiv.org/pdf/2606.23685)
>
> **分类标签**：`人手数据` `世界模型对齐` `潜在推理VLA` `跨本体迁移` `数据手套`

---

## 一句话总结

LaST-HD 用一个 action-conditioned 世界模型（Ctrl-World）的前向动力学潜特征作为监督目标，把**非配对**的人手与机器人示教对齐到一个共享的"潜在物理推理空间"（reasoning-before-acting VLA 的 latent CoT），从而绕开动作层/几何层的刚性 retargeting；配合自研 $<$100 g 的 OOL 数据手套与 "mixed-to-human" 训练配方，在六个真实任务上域内平均成功率 73%（π0.5 为 62%），并且仅用 20 分钟人手在线纠正数据即可让新场景成功率超过 90%。

## 一、问题与动机

- **VLA 的数据瓶颈**：VLA 泛化依赖海量真机遥操作数据，采集昂贵低效。人手示教是"直接、数据丰富、含多样物理交互先验"的替代来源。
- **人手到机器人的核心障碍**：以往人到机的迁移主要走两条路——(1) kinematic retargeting / 形态学映射，把人手位姿硬映射到机器人关节；(2) 从人类视频抽取视觉表征或物体中心轨迹先验。两者要么对数据规模敏感，要么**只做动作层/表征层对齐,忽略了人手与机器人之间"物理动力学"的对齐**。
- **本文出发点**：作者提出一个问题——能否把 VLA 模型内部的**物理推理（latent reasoning）**当作中间接口,来更好地把人手示教迁移到机器人动作学习?核心洞察是"操作任务背后的物理不变性":同一个推动动作,不管由人手还是机械臂执行,都会对物体产生相似的运动后果;因此在**前向动力学(forward-dynamics)潜空间**里,人手与机器人天然更容易对齐,而不是在外观/几何层。

## 二、核心方法

整体由三部分组成：MoT 结构的 reasoning-before-acting VLA、人手到机器人的潜在对齐（world model 作桥）、mixed-to-human 训练配方；外加低成本 OOL 数据手套硬件。

### 2.1 基础架构：MoT 的"先推理后动作"VLA

在 LaST$_0$（Janus-Pro 基座的 latent-CoT VLA）之上构建。视觉编码器用 SigLIP-Large（观测 $384\times384$），特征经 MLP 投影进 LLM 隐空间；backbone 用 DeepSeek-LLM 1.5B 的 24 层 decoder-only transformer，改造成含两个专家的 Mixture-of-Transformers（MoT）：

- **reasoning expert**：自回归地预测一串潜在状态 $\mathcal{Z} \in \mathbb{R}^{N_{lat}\times d_l}$，$N_{lat}$ 为推理 token 数；
- **action expert**：通过 flow-matching 预测动作 chunk。策略形式为 $\mathbf{a}_{t+1:t+H} \sim \pi_\theta(\cdot \mid \mathbf{I}_t, \mathbf{l})$。

两个专家之间用 **shared attention** 传递潜在推理知识——这就是注入"形态无关物理推理先验"的接口。

> 用大白话说：模型先在脑子里"想一遍这一步物理上会发生什么"（吐出一串潜在推理 token），再让动作专家照着这个"想法"去生成机械臂动作。关键是这个"想法"要能被人手和机器人共享。

### 2.2 用世界模型当对齐桥（本文最核心创新）

怎么给 reasoning expert 的这串潜在 token 提供监督信号，还得让人手和机器人共用一套？作者不用未来帧视觉，而是：

1. 在**混合的（手套采集人手 + 真机）**示教上 fine-tune 一个 action-conditioned 世界模型（Ctrl-World [26]）；两个域的数据**无需严格配对**。
2. 世界模型吃视觉观测，把连续动作 chunk 通过 cross-attention 注入到每一层来引导生成。
3. 在最后一个 denoising step，从 U-Net **最深层**抽特征——这些特征编码了"动作引发的物理后果"，是 domain-invariant 的。
4. 经 MLP aligner 投到 $d_l$ 维、flatten、再用 adaptive average pooling 压成 $N_{lat}$ 个 latent token，作为监督 reasoning expert 的 **ground-truth 潜在目标** $\mathbf{z}^{\text{GT}}$。

作者强调：只用世界模型做**潜在监督**而非直接预测动作，因为它的潜特征对高效控制不够紧凑，且动作条件化会引入信息泄漏。

> 用大白话说：世界模型是个"物理后果预测器"。作者不拿它去直接开机器人，而是把它对"这个动作会造成什么物理变化"的内部理解抽出来，当成一把"标准答案尺子"，去教 VLA 的推理专家。因为物理后果与本体无关，人手数据和机器人数据被这把同一把尺子拉到了同一个潜空间。

### 2.3 训练目标

reasoning expert 用 cosine similarity loss 对齐预测潜 token $\hat{\mathbf{z}}_t$ 与目标 $\mathbf{z}_t^{\text{GT}}$：

$$\mathcal{L}_{\text{latent}} = \sum_{t=1}^{N_{lat}} \left(1 - \frac{\hat{\mathbf{z}}_t \cdot \mathbf{z}_t^{\text{GT}}}{\|\hat{\mathbf{z}}_t\|\,\|\mathbf{z}_t^{\text{GT}}\|}\right)$$

action expert 用 flow-matching 动作损失 $\mathcal{L}_{\text{act}}$。总目标：

$$\mathcal{L}_{\text{loss}} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{latent}}$$

> 用大白话说：一边学"想得对不对"（潜在推理与世界模型给的标准答案方向一致），一边学"手动得对不对"（动作贴合示教），$\lambda$ 平衡两者。

### 2.4 OOL 数据手套（Out-of-Lab Glove）

为规模化采人手数据自研的低成本可穿戴设备：

- 每只手套 $<$100 g，6 个紧凑 IMU（9 轴）模块，追踪 20 个手部关键点 + 1 个手腕关键点，统一到 hand-centric 坐标系；
- $>$200 Hz 采样、$<$10 ms 端到端时延、**亚毫米级每关键点 RMS 位置误差**；
- 采集时同步 egocentric 视觉（头/胸挂 ZED 2i）+ 两个手腕视角（Insta360 GO 3S）+ 手腕 6-DoF tracker；
- 采集速度比标准真机遥操作快 **4–5 倍**；相比外骨骼手套记录的是"原生人手动作"而非设备特定关节角；
- 采后统一转为 hand-centric 表示：并爪 gripper 命令由指尖距离导出，灵巧手关节角由 IK 从人手关键点相对空间关系解算——**同一份人手数据可同时 retarget 到并爪与高自由度灵巧手**。

### 2.5 Mixed-to-Human 训练配方

- **Stage 1 · Mixed Human-Robot Co-training**：先在混合数据上训世界模型（预训练混合含 OOL 手套数据 + Tianji 双臂数据，且世界模型无需为每个下游任务重训）；再用世界模型离线预计算潜在目标，co-train LaST-HD VLA——action expert 学真机可执行动作，reasoning expert 被对齐潜在目标监督。
- **Stage 2 · Human-Hand Online Correction（人手在线纠正）**：部署真机 rollout，找出易失败状态，**只用 OOL 手套在失败态补采人手纠正示教**（替代繁琐的真机遥操作补数）。世界模型冻结，仅 post-train 1–2 epoch，用 balanced replay：$\mathcal{B} = \mathcal{B}_{\text{prev}} \cup \mathcal{B}_{\text{dagger}}$，$|\mathcal{B}_{\text{prev}}| = |\mathcal{B}_{\text{dagger}}|$（人手 DAgger buffer 与旧数据等量采样）。

## 三、实验结果

**设置**：六任务、三本体。双臂并爪——Galaxea R1 Lite 上 Unscrew Bottle Cap / Organize Box（$\mathbf{a}_m \in \mathbb{R}^{14}$），Tianji Marvin 上 Sort Fruits / Put Items to Bag and Zip（$\mathbb{R}^{16}$）；灵巧手——Marvin + 20 关节 WUJI 手上 Pour Water / Grasp with a Clamp（$\mathbf{a}_m=[\Delta\theta^R_{1:7},\Delta h^R_{1:20},\Delta\theta^L_{1:7},\Delta h^L_{1:20}]\in\mathbb{R}^{54}$）。三路 $384\times384$ 视角。每任务 100 条真机 + 50 条 OOL 手套示教，每任务 20 次 rollout。Baselines：LaST$_0$、π0.5、Cosmos-Policy。MoT 主干为公平只在真机轨迹上预训练（400K 轨迹 / 28M 帧，OXE+DROID+RoboMIND）。

### 域内结果（Table 1，成功率）

| 方法 | Unscrew Cap | Organize Box | Sort Fruits | Put and Zip | Pour Water | Grasp Clamp | **Avg** |
|---|---|---|---|---|---|---|---|
| π0.5 | 0.70 | **0.70** | 0.85 | 0.75 | 0.30 | 0.40 | 0.62 |
| Cosmos-Policy | 0.75 | 0.50 | 0.85 | 0.60 | 0.20 | 0.20 | 0.52 |
| LaST$_0$ | 0.80 | **0.70** | 0.75 | 0.60 | 0.40 | **0.50** | 0.63 |
| **LaST-HD** | **0.85** | **0.70** | **0.95** | **0.80** | **0.60** | 0.45 | **0.73** |
| **LaST-HD (Mix-HD)** | **0.85** | **0.70** | 0.85 | **0.80** | 0.40 | 0.45 | 0.68 |

- LaST-HD 平均 73%，明显高于最强 baseline（LaST$_0$ 63%、π0.5 62%），在多步任务 Sort Fruits / Put and Zip 上分别 95% / 80%，在高自由度灵巧手 Pour Water 上 60% 相对优势更大。
- **Mix-HD**（把一半真机数据换成 50 条人手数据）在 6 任务中 4 个持平 LaST-HD，说明手套人手数据能有效替代部分真机数据而不掉性能。

### 泛化结果（Table 2，三种未见场景平均）

| 方法 | Unseen Position | Unseen Object | Unseen Background | Global Avg |
|---|---|---|---|---|
| π0.5（零样本） | 0.12 | 0.36 | 0.42 | 0.30 |
| Cosmos-Policy（零样本） | 0.13 | 0.28 | 0.38 | 0.26 |
| LaST$_0$（零样本） | 0.15 | 0.32 | 0.43 | 0.30 |
| LaST-HD (Mix-HD)（零样本） | 0.15 | 0.35 | 0.43 | 0.31 |
| LaST$_0$ (+未见人手数据) | 0.33 | 0.49 | 0.58 | 0.46 |
| **LaST-HD (+未见人手数据)** | **0.41** | **0.58** | **0.68** | **0.56** |

- 零样本下所有方法在 Unseen Position 骤降（π0.5 仅 12%）。
- 每个未见场景补 60 条 OOL 手套人手示教后，LaST-HD 在 Unseen Object 达 58%（超 π0.5 约 22 个点），Unseen Background 达 68%，全局平均 56%（LaST$_0$ 同法为 46%）——证明"共享潜空间 + 低成本人手数据"能驱动机器人泛化到新场景。

### 人手在线纠正（Fig 3a，Sort Fruits）

- 采 60 条 OOL 手套轨迹仅需 **20 分钟**；补 20 条人手轨迹即让 Unseen Background 成功率达 **100%**，补 60 条让 Unseen Object 达 **100%**；即便最难的 Unseen Position 也从 60% 单调升到 80%。

### 关键消融（Fig 3b / Table 5，Sort Fruits 三场景平均）

| 潜在对齐目标 | LaST-HD (世界模型潜) | WM-only | SigLIP 潜 | 去掉潜在推理 |
|---|---|---|---|---|
| 成功率 | **73** | 66 | 63 | 60 |

- 去掉 latent reasoning 从 73% 掉到 60%——说明纯动作层 co-training 无法充分利用人手数据；action-conditioned 世界模型潜特征优于 SigLIP 未来帧视觉潜与无动作条件的 WM-only。

| 数据采集范式 | OOL Glove | Real-60 | Real-12 | Bare-hand | UMI | Palm-view |
|---|---|---|---|---|---|---|
| 成功率 | **0.73** | 0.75 | 0.60 | 0.63 | 0.65 | 0.67 |

- OOL 手套(73%)显著超裸手视觉追踪(63%)与 UMI(65%);同等采集时间下超真机 Real-12 达 13 个点,并与花更多时间的 Real-60(75%)持平;手腕相机放在虎口(thumb-index web space)优于掌心视角(67%)。
- 其它:世界模型 denoising steps 取 2/5/10 成功率 0.73/0.72/0.76(取 2 省算力);shared latent 长度 2/4/8/12/16 成功率 0.67/0.73/0.67/0.70/0.78(权衡时延取 4)。
- 附录 Table 4 披露已累计 **2000+ 小时** OOL 手套示教(家务 913.7h、可变形物体 986.3h 等),但本文 MoT 主干为公平仅用真机预训练,该 2000h 未进入主干预训练。

## 四、局限性

1. **潜在推理非实时**(作者自陈的主要局限):reasoning expert 的自回归潜在解码增加推理时延,latent 长度越长越慢;未来拟走 fast-slow 双系统或进一步压缩潜空间。
2. **retargeting 仍是启发式**:每引入一种新灵巧手需手工构造 retargeting 管线,mapping 质量对预训练与微调都关键;作者指出应换成可学习的 retargeting。
3. **失败模式**:灵巧手抓取时物体运输中滑落(接触力不精);流体任务(倒水)因液体动力学随机、过程预测不可靠致溢出;双臂持续接触任务(拧瓶盖)相对位姿误差累积。
4. 泛化的绝对数值仍偏低(Unseen Position 补数后也仅 41%),距实用尚远;评测为真机小样本(20 rollout/任务),方差与统计显著性未报告。

## 五、评价与展望（学术视角）

**优点**：
- **对齐层次上移是有价值的立意**：把人到机的对齐从"动作/几何/未来帧视觉"提升到"动作条件世界模型的前向动力学潜空间",用物理后果的本体不变性来桥接非配对数据,概念上比 EgoMimic/DexWild 的显式 kinematic 对齐、比 Track2Act 的物体轨迹先验更抽象也更贴合"物理"。这与 LaST$_0$、EgoScale [24]、H-RDT [23] 等"把人手当另一种 embodiment 做跨本体预训练"的路线互补。
- **系统完整**:硬件(OOL 手套)、算法(latent 对齐)、数据配方(mixed-to-human + 人手 DAgger 在线纠正)闭环,且给出"用人手数据替代真机补数"的实用价值(20 分钟达 90%+)。
- 消融较扎实地把"世界模型潜 vs SigLIP 视觉潜 vs 无潜在"拆开,支撑了核心主张。

**缺点与开放问题**：
- **世界模型是"教师"的可信度未被独立验证**:整套方法把世界模型潜特征当 ground-truth,但世界模型本身只在有限混合数据上 fine-tune,其"物理后果预测"是否可靠、误差如何传导到 VLA,论文未量化;若世界模型对人手域的建模弱于机器人域,潜在目标本身就是有偏的。
- **"物理推理"的证据偏间接**:主要靠 UMAP 人手/机器人 latent 重叠 + attention 集中于接触区来佐证,缺少反事实/因果性检验(如打乱动作条件后潜目标是否退化),难以排除"只是学到了任务语义聚类"。
- **公平性与规模**:声称 2000h 人手数据却未用于主干预训练,使"scalable human data"的规模优势在主实验中未真正兑现;baseline 是否在同等数据/算力下调优也需更细披露。
- **可复现性**:OOL 手套为自研硬件,LaST$_0$/世界模型 Ctrl-World 均为该团队近作,外部复现门槛高。

**可能改进方向**：把 retargeting 换成可学习模块并与 latent 对齐联合优化;引入 fast-slow 解耦让潜在推理实时化;为世界模型教师加不确定性估计、对低置信潜目标降权;对灵巧手接触/流体任务补 tactile 或 fluid-aware 感知。总体看,这是一篇"立意新、系统全、但核心机制的因果证据与规模承诺尚不充分"的技术报告,方向上对"人手数据 → 机器人预训练"很有启发。

## 参考

1. LaST$_0$ (Liu et al., 2026, arXiv 2601.05248) — Janus-Pro/MoT 的 latent spatio-temporal CoT VLA,本文直接基座。
2. Ctrl-World (Guo et al., 2025, arXiv 2510.10125) — 可控生成式世界模型,本文的"对齐桥"来源。
3. EgoScale (Zheng et al., 2026, arXiv 2602.16710) / H-RDT (Bi et al., AAAI 2026) — 把人手当另一 embodiment 做跨本体预/中训练的代表工作。
4. DexWild (Tao et al., 2025, arXiv 2505.07813) / EgoMimic (Kareer et al., ICRA 2025) — 依赖显式 kinematic 对齐的人到机模仿,本文的对照路线。
5. π0.5 (Intelligence et al., 2025, arXiv 2504.16054) / Cosmos-Policy (Kim et al., 2026, arXiv 2601.16163) — 本文主要 VLA 与 world-action baseline。
