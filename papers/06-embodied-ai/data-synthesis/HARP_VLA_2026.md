# HARP-VLA：面向视觉-语言-动作模型的人机对齐表征学习

> **论文**：*HARP-VLA: Human-Robot Aligned Representation Learning for Vision-Language-Action Model*
>
> **作者**：Xiang Zhu, Puzhen Yuan, Yichen Liu（三人共同一作）, Jianyu Chen（通讯）
>
> **机构**：清华大学交叉信息研究院（Institute for Interdisciplinary Information Sciences, Tsinghua University）；上海期智研究院（Shanghai Qi Zhi Institute）
>
> **发布时间**：2026 年 05 月（arXiv 2605.31234）
>
> **发表状态**：未录用（预印本，代码仓库为匿名链接，正文致谢注明"若被接收"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.31234) | [PDF](https://arxiv.org/pdf/2605.31234)
>
> **分类标签**：`人机表征对齐` `潜在动作模型` `VLA 预训练` `跨具身`

---

## 一句话总结

HARP-VLA 提出一个三阶段框架：用少量"人-机成对示范"作为跨具身桥梁、海量非成对人/机视频作动力学监督，训练一个**只改机器人分支、把人分支冻结当锚点**的视觉编码器与潜在动作模型（LAM），配合一个新的 **source-relative pair-discriminative（SRPD）对齐损失**，同时缩小人机之间的"视觉表征鸿沟"和"动作执行鸿沟"，在 CALVIN ABC→D 上把平均完成长度从 3.917 提到 4.481，真机四任务平均成功率较最强基线提升 7.1 个百分点。

## 一、问题与动机

从大规模人类操作视频里学可泛化的 VLA 策略很诱人，但受两个跨具身鸿沟制约：

1. **动作执行鸿沟（action execution gap）**：人手动作难以直接翻译成可执行的机器人指令。潜在动作模型（LAM，如 LAPA / UniVLA / UniSkill / IGOR）通过从相邻帧学习"具身无关的潜在转移码"来缓解，不再直接预测机器人指令。
2. **视觉表征鸿沟（visual representation gap）**：人和机器人的外观、形态差异，使得同一操作动态被编码到两个分离的特征流形上。关键的是，**LAM 的潜在动作本身建立在视觉观测之上**——如果人机视觉表征没对齐，抽出来的潜在动作就会变成"域相关"的，破坏人-机共训与下游策略学习。

作者的核心观察：现有工作要么只处理视觉鸿沟（稀疏标注、图像级人到机替换、视频级表征适配如 HR-Align），要么只处理动作鸿沟（LAM），**没有把"帧级视觉对齐"和"帧级操作动力学 / 潜在动作学习"显式耦合起来**。于是提出一个在有限成对监督 + 海量非成对数据下，联合统一"视觉状态"和"潜在动作"的通用框架。

## 二、核心方法

HARP 是一个三阶段框架：**Stage 1** 从成对与非成对视频联合学一个机器人适配视觉编码器 + LAM；**Stage 2** 用学到的编码器和 LAM 生成的潜在动作标签预训练 VLA 策略；**Stage 3** 用少量真机动作数据 + 轻量 real-action head 微调出可执行策略。学到的三个组件分别记作 **HARP-VE**（对齐视觉编码器）、**HARP-LAM**（潜在动作模型）、**HARP-VLA**（完整策略）。

### 2.1 数据与预处理

训练数据 $\mathcal{D}=\mathcal{D}_p\cup\mathcal{D}_u$：成对集 $\mathcal{D}_p=\{(H_i,R_i,l_i)\}$ 是同一任务、同一指令 $l_i$ 的人视频 $H_i$ 与机器人视频 $R_i$；非成对集 $\mathcal{D}_u=\{(V_j,l_j)\}$ 的每条视频来自具身 $e_{V_j}\in\{h,r\}$。对每条视频抽取辅助标注 $A_X=\{K_X,E_X\}$：$K_X$ 是 2D 物体位置轨迹（object tracks），$E_X$ 是 2D 手腕/末端执行器轨迹。对成对视频用手腕关键点作相似度、以 DTW（动态时间规整）沿最优匹配路径重采样人视频，得到严格帧级对齐的人机对。

> 用大白话说：成对数据是"同一件事，人做一遍、机器人做一遍"的对照录像，是把两个具身的语义拴在一起的锚；非成对数据只提供丰富的动力学；物体轨迹和手腕轨迹是"两种具身都共享的、比外观更不挑具身的运动线索"。

物体关键点抽取专门处理重度遮挡：无指令时先用 Qwen3-VL-8B-Instruct 生成被操作物描述，GroundingDINO 定位首帧框，再用 TAPIR 逐帧跟踪（利用其可见性分数抑制遮挡帧的不可靠预测）；人手关键点用 WiLoR 回归 MANO 参数取手腕 2D 坐标，机器人手腕用相机外参投影得到。

### 2.2 Embodiment-aware 视觉编码（冻人、改机）

给定具身 $e_X\in\{h,r\}$ 的视频 $X$，用一个"具身感知"编码器把帧编码为 patch 级视觉 token：

$$\Phi_\theta(X,e_X)=\begin{cases}F(X), & e_X=h\\ T_\theta(X), & e_X=r\end{cases}$$

其中 $F$ 是**冻结的预训练视觉编码器**（Prismatic 的 DINOv2+SigLIP 融合特征），$T_\theta$ 是在 $F$ 上加"机器人专用 adapter"的可训练编码器。视觉 token 记 $Z_X=\{z_X^t\}_{t=1}^{T_X}$，同时保留冻结教师特征 $Z_{X0}=F(X)$。

> 用大白话说：人的视频永远用原始冻结编码器编码，当作"不动的靶心"；只给机器人分支挂上轻量 adapter，把机器人特征往人的语义空间上拉。这样做的妙处（见 Stage 2 讨论）：网络级 VLM 预训练本就以人类中心数据为主、人的表征更语义鲁棒，冻住人分支就保住了 VLM 原有"视觉编码器—LLM"的对齐。

adapter 结构：在 DINOv2 与 SigLIP 每层的每个 attention 和 FFN 块后接一个 2 层 MLP，预测一个残差加回原特征；第一层高斯初始化、第二层零初始化，**初始化时输出与原特征完全一致**（不破坏预训练特征）。

### 2.3 自预测 / 交叉预测的潜在动作学习

LAM 是一个 VQ-VAE 式的逆向-正向动力学模型（结构follow UniVLA）：编码器从视觉转移推断潜在动作，解码器在"当前观测 + 量化潜在动作"条件下预测未来视觉表征。每个转移用 $N_q=4$ 个离散潜在动作 token 表示（VQ 码本大小 $K=16$，潜在维 $d_q=128$）。给定转移 $(z_X^t, z_X^{t+\Delta t})$ 和指令 $l_X$：

$$a_X^t=E_\theta(z_X^t,z_X^{t+\Delta t},l_X),\qquad q_X^t=Q_\theta(a_X^t),\qquad \hat{Y}_X^t=D_\theta(\bar{z}_X^t,q_X^t,l_X)$$

关键在于**解码目标 $Y_X^t$ 和条件 $\bar{z}_X^t$ 取谁，取决于样本是成对还是非成对**：

- 非成对视频 $V$——**自预测**：$\bar{z}_V^t=z_V^t,\ Y_V^t=z_V^{t+\Delta t}$（在自己视频内预测未来）。
- 成对视频 $(H,R)$——**交叉预测**：潜在动作从"源具身"推断，解码器却以"目标具身"的当前表征为条件、去预测目标具身的未来：

$$\bar{z}_H^t=z_R^t,\ Y_H^t=z_R^{t+\Delta t};\qquad \bar{z}_R^t=z_H^t,\ Y_R^t=z_H^{t+\Delta t}$$

> 用大白话说：交叉预测强迫"从人视频抽出的潜在动作，必须能解释机器人视频的未来动态"（反之亦然）。如果潜在动作还带着具身指纹，这个跨具身预测就做不好——于是潜在动作被逼成"跨人机共享的、纯运动"的编码，把潜在动作学习和人机对齐直接耦合到了一起。

潜在动作预测损失（未来表征预测 + 标准 VQ 码本/承诺损失）：

$$\ell_{\mathrm{lam}}(X,t)=\left\|\hat{Y}_X^t-Y_X^t\right\|_2^2+\ell_{\mathrm{vq}}(X,t)$$

**共享线索辅助损失**：在 LAM 编码器里加专门的辅助 token（1 个 keypoint token + 1 个 end-effector token），用两个轻量 head 预测物体关键点 $\hat{K}_X^\tau$ 与手腕/末端位置 $\hat{E}_X^\tau$，以带可见性掩码的 Huber 损失监督：

$$\mathcal{L}_{\mathrm{aux}}=\lambda_K\mathcal{L}_K+\lambda_E\mathcal{L}_E$$

> 用大白话说：让潜在动作去"顺带预测物体怎么动、手腕怎么走"，把它往物体运动和粗粒度动作意图上偏，而这些线索比原始外观更跨具身通用（对没有成对监督的非成对数据尤其重要）。

### 2.4 Source-relative pair-discriminative 对齐损失（本文核心创新）

对成对示范，作者要让适配后的机器人表征满足两个约束：(a) **相对于成对人视频**，比冻结机器人表征更靠近人；(b) 成对对应关系仍要能区分于非匹配对。记 $f^H=\rho(Z_H)$（冻结人特征）、$f^{R0}=\rho(Z_{R0})$（冻结机器人特征）、$f^R=\rho(Z_R)$（适配机器人特征），$\rho$ 是视频级池化，特征做 $\ell_2$ 归一化，$d(u,v)=1-\cos(u,v)$。

**Source-relative（SR）项**——要求适配后的机器人特征相对冻结版"有进步"：

$$\mathcal{L}_{\mathrm{SR}}=\mathbb{E}_{(H,R)\sim\mathcal{D}_p}\left[m_s+d(f^R,f^H)-d(f^{R0},f^H)\right]_+$$

> 用大白话说：不是硬把机器人特征拉到某个绝对距离目标（那会让所有成对特征塌到一起、丢掉区分度），而是只要求"改后的机器人比改前更接近它配对的人"——用冻结特征当自参照基线，是一种更温和的相对约束。

**Pair-discriminative（PD）项**——用 batch 内非匹配对当负样本，双向拉开：

$$\mathcal{L}_{\mathrm{PD}}=\mathbb{E}_{(H,R)\sim\mathcal{D}_p}\sum_{\alpha\in\{R\to H,\,H\to R\}}\lambda_\alpha\left[m_t+d(f^R,f^H)-\bar{d}^\alpha\right]_+$$

其中 $\bar{d}^{R\to H}=\mathbb{E}_{H'\neq H}\,d(f^R,f^{H'})$、$\bar{d}^{H\to R}=\mathbb{E}_{R'\neq R}\,d(f^{R'},f^H)$ 是两个方向上的平均负样本距离。最终对齐损失 $\mathcal{L}_{\mathrm{align}}=\mathcal{L}_{\mathrm{SR}}+\mathcal{L}_{\mathrm{PD}}$（当 batch 只有 1 对时省掉 PD 项）。

> 用大白话说：SR 保证"靠近正确的人"，PD 保证"别靠错人"——匹配对要比该机器人到其它人、该人到其它机器人的距离都更近一个 margin。二者合起来既压缩了绝对鸿沟又保住了跨具身可区分性。

### 2.5 三阶段训练目标

**Stage 1** 总目标：

$$\mathcal{L}_{\mathrm{stage1}}=\mathcal{L}_{\mathrm{lam}}+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$$

**Stage 2（VLA 潜在动作预训练）**：用 HARP-LAM 给每帧打统一潜在动作标签 $q_X^t$，把一个 VLM（Prismatic-7B / LLaMA-2 骨干，follow OpenVLA-OFT）训练成预测这些潜在动作 token 的策略，交叉熵损失

$$\mathcal{L}_{\mathrm{pretrain}}=-\mathbb{E}_{(x^t,l_X)\sim\mathcal{D}}\left[\sum_{i=1}^{N_q}\log\pi_\theta\!\left(\hat{q}_i=q_{X,i}^t\mid x^t,l_X\right)\right]$$

并且**把 Stage 1 学到的 adapter 权重复制到 VLM 的视觉编码器**——因为 Prismatic 的融合视觉编码器在网络级预训练时是冻结的，复制权重恰好保留了原有的表征对齐性质，让人机视觉对齐 + 潜在动作对齐同时生效。

**Stage 3（真机动作微调）**：因为策略只会输出潜在动作、不能直接控制，接一个 action head 把潜在动作 embedding 映射到归一化真实动作（从零训、L1 损失），同时用 LoRA 微调 VLA 骨干，得到最终 HARP-VLA。

## 三、实验结果

数据规模（Table A1）：非成对人类视频 HOI4D(8.9M)+OpenEgo(36.4M)，非成对机器人 Bridge-V2(8.6M)+自采灵巧手(3.6M)；成对人-机 RH20T(8.16M)+Human2Robot(9.9M)+自采人-灵巧手(5.7M)；真机动作数据 CALVIN(1.1M) 与自采(0.5M)。评测分表征对齐、RLBench 下游策略、CALVIN 与真机三层。

**表征对齐（Table 1，held-out 成对视频上的双向跨具身检索 Recall@1）**：H2R 用人视频查配对机器人、R2H 反之。

| 方法 | H2R R@1 | R2H R@1 | 平均 R@1 |
|---|---|---|---|
| Unadapted（未适配） | 44.09 | 43.01 | 43.55 |
| HR（HR-Align） | 45.16 | 45.16 | 45.16 |
| HARP-HR | 46.24 | 60.22 | 53.23 |
| HARP-L2 | 70.97 | 52.69 | 61.83 |
| HARP-SR | 84.95 | 64.52 | 74.74 |
| **HARP-SRPD** | **87.10** | **69.89** | **78.50** |

SRPD 把平均 R@1 从 43.55 拉到 78.50，且加了 PD 项后（相比只有 SR 的 74.74）跨具身可区分性进一步提升。Figure 5 的成对余弦距离箱线图也显示 HR-Align 类目标几乎不降低成对距离，而 HARP 变体显著压小。

**RLBench 下游策略（Table 2，18 任务、冻结视觉编码器、其余架构/数据/预算完全一致，平均成功率 %）**：

| 方法 | 平均成功率 |
|---|---|
| Unadapted | 37.56 |
| HR | 39.70 |
| HR-Style | 38.22 |
| HARP-HR | 35.11 |
| HARP-HR-Style | 40.07 |
| HARP-L2 | 40.78 |
| HARP-SR | 43.41 |
| **HARP-SRPD** | **46.59** |

只换视觉编码器，SRPD 就把成功率从 37.56 提到 46.59，说明更强的跨具身可区分性能转化为更有用的下游特征。

**真机操作（Table 3，Xarm7 + Robotera XHand 灵巧手，18 DoF，每任务 60 次试验，成功率 %）**：

| 模型 | Pick | Push | Press | Flip | 平均 |
|---|---|---|---|---|---|
| $\pi_0$ | 58.3 | 75.0 | 56.7 | 35.0 | 56.3 |
| $\pi_{0.5}$ | 71.7 | **83.3** | 68.3 | 53.3 | 69.2 |
| OpenVLA | 0.0 | 23.3 | 18.3 | 0.0 | 10.4 |
| UniVLA | 38.3 | 61.7 | 31.7 | 21.7 | 38.4 |
| OpenVLA-OFT | 51.7 | 71.7 | 76.7 | 43.3 | 60.9 |
| HARP-VLA（L2 对齐） | 70.0 | 71.7 | 81.7 | 56.7 | 70.0 |
| HARP-VLA（不冻编码器） | 76.7 | 80.0 | 78.3 | 58.3 | 73.3 |
| **HARP-VLA（完整）** | **76.7** | 81.7 | **85.0** | **61.7** | **76.3** |

完整 HARP-VLA 平均 76.3%，较最强基线 $\pi_{0.5}$（69.2%）高 7.1 个百分点。

**CALVIN ABC→D（Table 4，Task1–5 为完成前 k 个子任务的比例，Avg.Len 为平均完成子任务数，越高越好）**：

| 模型 | Task1 | Task2 | Task3 | Task4 | Task5 | Avg.Len |
|---|---|---|---|---|---|---|
| $\pi_0$ | 92.3 | 82.4 | 72.1 | 62.2 | 53.7 | 3.627 |
| $\pi_{0.5}$ | 94.4 | 86.0 | 76.4 | 69.7 | 61.0 | 3.875 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.270 |
| UniVLA | 95.4 | 85.5 | 75.4 | 66.9 | 56.5 | 3.800 |
| OpenVLA-OFT | 94.2 | 86.4 | 78.0 | 70.4 | 62.7 | 3.917 |
| HARP-VLA（L2 对齐） | 95.8 | 89.7 | 81.3 | 72.8 | 64.8 | 4.044 |
| HARP-VLA（不冻编码器） | 98.8 | 93.9 | 86.1 | 77.7 | 68.5 | 4.250 |
| **HARP-VLA（完整）** | **99.8** | **96.7** | **91.3** | **84.4** | **75.9** | **4.481** |

**两个关键消融**：（1）SRPD 对齐 vs 朴素 L2 对齐——SRPD 在 CALVIN 平均长度上 +0.437（4.044→4.481）、真机 +6.3%（70.0→76.3）；（2）Stage 2 是否冻结视觉编码器——冻结把 CALVIN 从 4.250 提到 4.481、真机从 73.3% 提到 76.3%，印证"adapter 先补齐人机视觉鸿沟后，冻住编码器能同时保住人机对齐和网络级预训练的视觉特征"。

## 四、局限性

- **强依赖成对桥梁数据**：性能取决于 bridge 数据的多样性、时序对齐（DTW）与共享线索的鲁棒性；作者也承认对"噪声辅助线索"的鲁棒性有待提升。
- **评测面窄**：仅在单一机器人平台（Xarm7+XHand）的桌面操作上验证，人类视频来源有限；未涉及长时序、双臂、多具身。
- **潜在动作离散度极低**：$N_q=4$、码本 $K=16$、$d_q=128$，潜在动作空间容量很小，能表达的动作粒度可能受限，论文未讨论码本容量对细粒度/接触丰富任务的影响。
- **未与最新 LAM 系工作充分横比**：RLBench/真机主要与 HR-Align、OpenVLA(-OFT)、$\pi_0/\pi_{0.5}$、UniVLA 比，缺 LAPA、IGOR、UniSkill 等直接同类 LAM 在同一协议下的对照。
- **匿名预印本**：代码为匿名仓库、结果未经同行评审，成对数据/真机部分含未公开的自采集数据，第三方复现完整链路有难度。

## 五、评价与展望

**优点**：（1）问题定位准——把 LAM 的"潜在动作依赖视觉表征"这一隐患点破，指出只做视觉对齐或只做 LAM 都不够，主张两者耦合，是对 UniVLA / LAPA 一脉的自然而扎实的推进。（2）"冻人分支、只改机器人分支"是一个既省事又有道理的非对称设计：把人类中心的 VLM 表征当锚，既避免了对齐时的表征塌缩，又保住了 VLM 原有的视觉—语言对齐，工程上还能直接把 adapter 权重搬进 VLA 复用，闭环很干净。（3）SR + PD 对齐损失是本文最有价值的技术点——相对参照（比冻结基线更近）而非绝对目标，配合双向 pair-discriminative 负样本，检索指标（43.55→78.50）和下游成功率的强相关，为"表征可区分性 → 策略有用性"提供了较清晰的经验证据。（4）交叉预测把对齐信号注入潜在动作学习本身，而非仅作为额外正则，机制上比"先对齐再学动作"更紧。

**与公开工作的关系**：本质是 HR-Align（视觉域适配）× UniVLA/LAPA（LAM）的耦合体，并用成对人-机数据（RH20T、Human2Robot 及自采）当桥。相比 EgoMimic/RoVi-Aug/DexUMI 用视觉变换或线索缩小域差、Im2Flow2Act/Dream2Flow 用物体流当具身无关接口，HARP 的差异在于把"帧级视觉对齐"直接写进潜在动作的训练目标，且给出了一个能落到 VLA 权重初始化的完整三阶段流程。

**开放问题与改进方向**：（1）成对桥梁数据是瓶颈——能否用生成式手段（人到机的视频/图像转换）合成"伪成对"以放大 SR/PD 的监督量，减少对真实成对采集的依赖？（2）$N_q=4$、$K=16$ 的极小码本在接触丰富、双臂、长时序任务上是否够用，值得做容量-性能的系统消融。（3）辅助线索（TAPIR 物体轨迹、WiLoR 手腕）在重度遮挡下必然带噪，可见性掩码之外能否引入不确定性加权或自蒸馏来抗噪。（4）SR 项以"冻结机器人特征"为自参照，若冻结特征本身质量差，参照基线也差——可探索动态更新参照或课程式收紧 margin。（5）跨多机器人本体、跨相机视角的泛化尚未验证，这决定该框架能否作为通用预训练接口而非单平台方案。

## 参考

1. LAPA: *Latent action pretraining from videos* (Ye et al., 2024) — LAM 从无标注视频学离散潜在动作码的开创工作，本文动作抽象的直接前身。
2. UniVLA: *Learning to act anywhere with task-centric latent actions* (Bu et al., 2025) — 强调任务相关运动、抑制无关动态，本文 LAM 结构直接沿用。
3. HR-Align: *Mitigating the human-robot domain discrepancy in visual pre-training for robotic manipulation* (Zhou et al., CVPR 2025) — 视觉域适配基线，本文对齐损失的主要对照对象。
4. OpenVLA-OFT: *Fine-tuning vision-language-action models: Optimizing speed and success* (Kim et al., 2025) — 本文 VLA 骨干与 action head 设计所依据的架构。
5. Human2Robot: *Learning robot actions from paired human-robot videos* (Xie et al., 2025) — 提供成对人-机桥梁数据，是本文跨具身对齐监督的关键数据源之一。
