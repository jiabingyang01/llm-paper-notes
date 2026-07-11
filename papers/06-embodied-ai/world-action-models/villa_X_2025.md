# villa-X：增强视觉-语言-动作模型中的隐动作建模

> **论文**：*villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models*
>
> **作者**：Xiaoyu Chen, Hangxing Wei, Pushi Zhang et al.
>
> **机构**：Microsoft Research、清华大学、武汉大学、香港科技大学、南京大学
>
> **发布时间**：2025 年 07 月（arXiv 2507.23682，v3 于 2025 年 09 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.23682) | [PDF](https://arxiv.org/pdf/2507.23682)
>
> **分类标签**：`隐动作建模` `VLA预训练` `跨具身泛化` `联合流匹配扩散` `人类视频学习`

---

## 一句话总结

villa-X 给隐动作模型（LAM）加入一个以机器人本体感知为监督目标的 proprio-FDM 分支，让隐动作从"纯视觉重建"变为"视觉 + 物理动态双重接地"，再用 ACT-latent / ACT-robot 两个专家在同一个流匹配（flow matching）联合扩散过程中显式建模"隐动作规划→机器人动作"的因果链；在 SIMPLER 上取得 Google Robot 平均成功率 77.7%、WidowX 62.5%，在 LIBERO 四套件平均 90.1%，并在真实 Realman 机械臂上零样本迁移到训练时从未见过的具身。

## 一、问题与动机

隐动作（latent action）是当前 VLA 预训练里让机器人策略"白嫖"海量动作缺失的人类视频/异构机器人视频的主流范式：先用一个 Latent Action Model（LAM）以 Inverse Dynamics Model（IDM）+ Forward Dynamics Model（FDM）从相邻两帧 $(o_t, o_{t+K})$ 中抽出一个隐动作 token $z_t$，再把它当作伪动作标签去增广机器人策略的模仿学习。LAPA、Moto、GR00T、IGOR、Go-1（AgiBot World）等一系列工作都遵循这一思路，但作者指出两个尚未系统解决的问题：

1. **隐动作学习本身不够物理接地**。现有 LAM 只靠视觉重建损失约束 $z_t$，而末端旋转、夹爪开合这类对控制至关重要但像素变化 subtle 的动作，视觉模型天然不敏感，导致学到的隐动作和真实机器人动态脱节，削弱了跨具身知识迁移的效果（论文提到近期工作 IGOR 也指出过这一局限）。
2. **隐动作如何更有效地整合进 VLA 预训练**。已有方法要么只把隐动作当预训练权重初始化后就丢弃（LAPA），要么用两阶段 teacher-forcing 的自回归隐动作规划器（Go-1 式），容易产生训练/推理不一致，整合方式都不够结构化。

villa-X 针对这两点分别给出改进：（i）在 LAM 里加一个 proprio-FDM 辅助解码器，用机器人本体状态/动作做监督，把隐动作"物理接地"；（ii）在策略（ACT）里用联合扩散显式建模隐动作专家（ACT-latent）和机器人动作专家（ACT-robot），并用注意力让机器人动作生成显式依赖隐动作规划，而不是仅靠权重初始化传递知识。

## 二、核心方法

villa-X 分两个模块、三阶段训练：（1）LAM 预训练；（2）ACT 联合预训练（隐动作 + 机器人动作建模）；（3）针对具体具身的微调。

**LAM：加入本体感知接地。** 标准 LAM 的目标只是视觉一致性：

$$z_t = \text{IDM}(o_t, o_{t+K}), \qquad \hat{o}_{t+K} = \text{FDM}(o_t, z_t)$$

这个目标能保证视觉变化的一致性，但忽略了物理动态，在有机器人状态可用时接地不足。villa-X 引入一个**本体感知前向动力学模型**（proprio-FDM），用当前本体状态 $q_t$ 和隐动作 $z_t$ 预测未来 $K$ 步的状态与动作：

$$(\hat{q}_{t+1}, \dots, \hat{q}_{t+K}, \hat{a}_{t+1}, \dots, \hat{a}_{t+K}) = \text{proprio-FDM}(q_t, z_t, c_e)$$

用大白话说：原来 LAM 只要求隐动作"重建出下一帧长什么样"就行，现在还要求它能"倒推出机器人接下来怎么走、关节怎么动"，这就逼着隐动作 token 把注意力放到末端旋转、夹爪这类视觉上不明显但对控制关键的动态上，而不是只关心画面像素的宏观变化。

其中 $c_e$ 是具身上下文向量，用来消解异构数据集（不同形态、不同控制频率）带来的歧义：

$$c_e = f(\text{dataset ID}, \text{control frequency})$$

数据集 ID 映射为可学习 embedding，控制频率用正弦特征 + MLP 编码，两者拼接后与 $q_t$ 一起输入 proprio-FDM，使模型能把"具身特有的动力学差异"与"隐动作本身的一致性"解耦。LAM 的总损失是图像重建损失 + 本体预测损失 + VQ 承诺损失的联合优化；对没有本体标签的人类视频，直接省略本体项。最终取 VQ 码本中心的连续向量作为隐动作。

架构细节（附录 A）：IDM 是 ST-Transformer（patch 14，隐藏维 768，32 头，12 层），输入 8 帧视频、产出 7 个隐动作 token；VQ 码本大小 32；FDM 是 12 层 ViT-Base；proprio-FDM 是双头 2 层 MLP。批大小 512、学习率 $1.5\times10^{-4}$，在 128 张 A100 上训练约 4 天。

**ACTor 模块（ACT）：隐动作与机器人动作的联合扩散。** ACT 把策略显式分解为两个条件分布：

$$\pi(a_{t:t+m-1}, z^K_{t:t+(n-1)K} \mid o_t, l, q_t, c_e) = \underbrace{\pi_{\text{robot}}(a_{t:t+m-1} \mid z^K_{t:t+(n-1)K}, o_t, l, q_t, c_e)}_{\text{ACT-robot}} \cdot \underbrace{\pi_{\text{latent}}(z^K_{t:t+(n-1)K} \mid o_t, l)}_{\text{ACT-latent}}$$

用大白话说：先由 ACT-latent 像"中层规划器"一样，根据当前画面和语言指令想清楚接下来一串隐动作该怎么走（不依赖具体机器人形态）；再由 ACT-robot 把这份隐动作计划、连同本体状态和具身上下文，翻译成具体的低层机器人动作序列。这比 LAPA 那种"隐动作只用来初始化权重、训练完就扔掉"的做法更结构化，也避免了 Go-1 式两阶段 teacher-forcing 带来的训练/推理不一致。

ACT 由三个专家组成、共享 block-wise causal attention mask：VLM（PaliGemma-3B，编码视觉语言高层特征）、ACT-latent（18 层 Transformer，隐藏维 1024、8 头，预测隐动作 token）、ACT-robot（同规格，预测低层动作 chunk，额外接入本体状态、具身上下文、可选腕部相机）。

实现上用流匹配联合建模隐动作和机器人动作（记二者拼接为 $x_t$，条件记为 $O_t=(o_t,l,q_t,c_e)$）：

$$L_\tau(\theta) = \mathbb{E}_{p(x_t \mid O_t), q(x_t^\tau \mid x_t)} \left\| v_\tau^\theta(x_t^\tau, O_t) - u(x_t^\tau \mid x_t) \right\|^2$$

其中 $x_t^\tau = \tau x_t + (1-\tau)\epsilon$、$\epsilon \sim \mathcal{N}(0,I)$，网络学习去噪速度场 $u(x_t^\tau \mid x_t) = \epsilon - x_t$。训练时 $\tau$ 从 Beta 分布采样，且刻意让隐动作对应的 $\tau$ 偏向更"噪"的区间，鼓励模型先粗粒度规划隐动作再精细化机器人动作；式中的显式因果分解通过 block-wise causal attention 实现。

**随机掩码策略**（借鉴 Moto 与 RDT）：训练时 50% 的 batch 完全屏蔽机器人动作到隐动作的注意力，另外 50% 的 batch 里随机屏蔽一半隐动作 token。作者强调这一设计"在实践中至关重要"——否则机器人动作分支会走捷径过度依赖隐动作，降低鲁棒性。

策略头借鉴 HPT，为每个具身分配独立的状态/动作投影层、共享其余参数；腕部相机特征经预训练 ResNet-18 提取后通过共享交叉注意力融合为 16 个 token，训练时按 50% 概率随机丢弃腕部视图。每个专家约 3 亿参数，从零训练，学习率 $5\times10^{-5}$，梯度裁剪范数 1.0，在 64 张 A100 上训练约 4 天。

**预训练数据。** 机器人侧以 OpenX 为主体，共 160 万条轨迹、2.235 亿帧，单一数据集里 AgiBot World Beta 占比最高（20%）；人类视频侧以 Ego4D 占比最大（21.46%），还混入 EPIC-KITCHENS、HoloAssist、Something-Something V2、HOI4D、HO-Cap、EgoPAT3D、EGTEA Gaze+ 等，共 360 万段视频片段。LAM 预训练只用第三人称主视角，ACT 预训练可选接入腕部视角（50% dropout）。

## 三、实验结果

**LAM 质量：probing 实验与 proprio-FDM 消融。** 在 LIBERO 数据集（未参与 LAM 训练）上冻结 LAM、训一个 3 层 MLP 从隐动作预测真实机器人动作，以八维动作空间（位置 3 + 旋转 4 + 夹爪 1）上的最大 L1 误差衡量隐动作的信息量。结果（Figure 3 直方图）显示：带 proprio-FDM 的版本（w/pp）在低误差区间样本数明显多于不带 proprio-FDM 的版本（wo/pp），高误差区间则反过来更少，说明 proprio-FDM 确实提升了隐动作对机器人动态的捕捉能力。进一步的量化消融（验证集重建损失 + 未见具身零样本 probing）：

| 指标 | w/o context | Ours（完整） | 相对提升 |
|---|---|---|---|
| Visual FDM loss ↓ | 0.068 | 0.057 | 16.2% |
| Proprio FDM loss ↓ | 0.078 | 0.070 | 10.3% |
| 未见具身 probing loss ↓ | 0.165 | 0.152 | 7.9% |
| 未见具身 xyz loss ↓ | 0.0675 | 0.0574 | 15.0% |
| 未见具身 rot loss ↓ | 0.00861 | 0.00619 | 28.1% |
| 未见具身 gripper loss ↓ | 0.928 | 0.873 | 5.9% |

**SIMPLER：小规模数据下的策略消融**（仅 10% Fractal + 10% BridgeV2 + 100% Something-Something V2 预训练）：

| 方法 | Google Robot 平均 | WidowX 平均 |
|---|---|---|
| Ours（w/pp，完整 LAM） | 58.5 | 40.8 |
| wo/pp（无 proprio-FDM） | 57.4 | 32.3 |
| wo/LAM（不用隐动作预训练） | 35.0 | 33.1 |
| LAPA-style 整合方式 | 43.8 | 1.0 |
| Go-1-style 整合方式 | 32.8 | 14.8 |

不使用隐动作预训练（wo/LAM）性能大幅落后，说明隐动作预训练本身价值巨大；在隐动作的整合方式上，villa-X 的联合扩散显著优于 LAPA 式"权重初始化后丢弃"和 Go-1 式"两阶段自回归规划器"。

**SIMPLER：全量预训练 + 微调后与已发表基线对比：**

| 方法 | Google Robot 平均 | WidowX 平均 |
|---|---|---|
| RT-1-X（预训练直评） | 49.4 | 1.1 |
| OpenVLA（预训练直评） | 32.7 | — |
| RoboVLMs | 60.8 | 37.5 |
| $\pi_0$ | 58.7 | 27.1 |
| $\pi_0$-FAST | 61.9 | 32.1 |
| OpenVLA-OFT | 63.0 | — |
| GR00T-N1.5 | 57.9 | 62.0 |
| TraceVLA | 57.3 | 27.7 |
| Magma | 62.3 | 44.8 |
| MoTo | 59.2 | — |
| LAPA | — | 57.3 |
| Ours w/o latent | 36.5 | 49.0 |
| **Ours（villa-X）** | **77.7** | **62.5** |

villa-X 在两个平台平均分均最高；GR00T-N1.5 在 WidowX 的 Carrot 子任务（54.3）略高于 villa-X（46.3），但 villa-X 在 Spoon/Eggplant/Cube 上全面领先，平均分仍居首。

**LIBERO 四套件：**

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| $\pi_0$（复现） | 88.0 | 88.5 | 87.0 | 61.0 | 81.1 |
| $\pi_0$-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| Ours w/o latent | 86.0 | 86.5 | 85.0 | 70.0 | 81.9 |
| **Ours（villa-X）** | **97.5** | **97.0** | **91.5** | **74.5** | **90.1** |

**真实机器人：Xarm + XHand 灵巧手**（5 任务，seen/unseen 双设置，成功率 %）：

| 方法 | Pick&Place | Stack Cube | Cup Upright | Pour Water | Flick Ball |
|---|---|---|---|---|---|
| GR-1 | 56/40 | 15/5 | 0/0 | 0/0 | 40/10 |
| GR00T | 44/28 | 20/0 | 20/0 | 0/0 | 30/0 |
| Ours w/o latent | 72/60 | 70/40 | 40/30 | 40/10 | 50/30 |
| **Ours** | 84/68 | 75/50 | 60/30 | 60/30 | 50/40 |

**真实机器人：Realman 机械臂 + 夹爪**（seen 任务 + 换积木色/换桌布两类分布外泛化测试）：

| 方法 | Pick in | Pick out | Push | Stack | Unstack | 换积木色 | 换桌布 |
|---|---|---|---|---|---|---|---|
| GR00T | 30 | 70 | 10 | 10 | 60 | 50 | 30 |
| Ours w/o latent | 40 | 80 | 30 | 60 | 70 | 40 | 30 |
| **Ours** | 30 | 100 | 50 | 50 | 100 | 60 | 60 |

值得注意，Pick-in 一项 villa-X（30）反而低于自身的 w/o latent 消融（40），Stack 一项也与 w/o latent 打平（50），说明隐动作带来的收益并非在所有子任务上都一致为正。

**策略架构消融**（同小规模设置）：去掉随机掩码策略（w/o mask）使 Google Robot 平均从 58.5 掉到 53.2、WidowX 从 40.8 掉到 34.0；去掉具身上下文（w/o context）则掉到 49.1 / 38.5，说明掩码策略和具身上下文对整合效果同样关键。

**零样本泛化可视化**：用一台训练时从未出现过的 Realman 机械臂，让 ACT-latent 单独生成隐动作计划，再用另外训练的世界模型把隐动作渲染成视频来验证效果（如"touch the corn"/"touch the hotdog"等指令，含符号卡片测试的开放词汇理解），定性显示 ACT-latent 能够零样本迁移到未见具身并理解训练集中罕见的符号概念。

## 四、局限性

论文自陈的局限（Conclusion 部分）：隐动作专家虽然能通过视觉 + 本体状态联合规划未来，但这种"规划能力"本身在本文中未被充分挖掘——目前只是单次采样后直接送入 ACT-robot，没有引入验证/筛选机制。作者提出未来可以引入一个带视觉语言先验的 critic，对隐动作专家的多次采样结果进行拒绝采样，过滤掉不符合语言指令的规划轨迹。

此外，通读全文可以观察到几点论文未明确讨论、但从实验数据可以看出的局限：

- proprio-FDM 的物理接地依赖机器人本体标签，对完全没有本体信息的人类视频只能退化为纯视觉 FDM 目标，人类视频部分并未真正享受到"物理接地"的改进，方法在人类视频上的适用边界还不清楚。
- 真实机器人实验（Realman 平台）显示在 Pick-in、Stack 等个别任务上，完整方法并不总是优于 w/o latent 消融，收益不一致，论文没有展开失败分析。
- proprio-FDM 和具身上下文的设计需要预先知道数据集 ID 与控制频率这类元信息，对于野外采集的新数据集/新具身，如何自动构造一致的 $c_e$ 仍是开放问题。
- 评测规模总体中等（SIMPLER 标准协议、LIBERO 各 10 任务、真实机器人多为 10~50 次试验），正文未给出置信区间；且相当一部分基线数值直接引自原论文而非本地统一复现（GR00T 除外）。

## 五、评价与展望

villa-X 的核心贡献可以概括为对隐动作范式的两处补课：一是让 LAM 不再是纯视觉自监督，而是显式引入本体感知的前向预测目标做接地（这与 IGOR、Como 等同期工作对"隐动作到底学到了什么"的关切一致，但 villa-X 给出了具体的训练目标改造方案）；二是把隐动作从"预训练权重初始化"或"独立自回归规划器"提升为策略推理时的显式条件变量，通过联合流匹配和 block-wise causal attention 实现结构化的信息传递，这一设计思路与 RDT、Moto 的掩码策略一脉相承，也吸收了它们在防止捷径学习上的经验。

横向看，villa-X 在 SIMPLER 和 LIBERO 上相对于同期公开的 $\pi_0$、$\pi_0$-FAST、GR00T-N1.5、Magma、OpenVLA-OFT 等基线取得了当时最优的平均成功率，尤其在 LIBERO-Long（长时序任务）上从 61 分区间提升到 74.5，显示隐动作规划对长程任务的组合泛化确有帮助；真实世界的双平台（夹爪 + 灵巧手）实验也提供了一定的可信度佐证。但需要指出，论文的基线对比中相当一部分数值直接引自原论文而非本地复现，数据集混合比例、计算规模（128/64 张 A100 × 4 天）与各基线并不完全对齐，横向比较的公平性有一定局限，这也是隐动作 VLA 这个方向普遍存在的评测可比性问题。

开放问题与可能的改进方向：（1）proprio-FDM 目前只用机器人本体状态/动作做监督，论文提到可以扩展到末端关键点检测、人手位姿估计等结构化线索作为接地信号，这为把方法进一步扩展到纯视频（无机器人本体）场景提供了空间；（2）隐动作专家的多样性采样与筛选（critic-based rejection sampling）仍未实现，是否能像扩散策略里的 best-of-N 一样带来显著提升值得跟进；（3）具身上下文向量目前是"数据集 ID + 控制频率"的手工编码，能否用更通用的自监督方式（如从动作统计量自动聚类）替代人工元信息，是走向真正即插即用跨具身的关键一步；（4）当前实验的隐动作码本大小仅为 32，是否需要随预训练数据规模增长而扩大、以及码本大小与下游泛化能力的关系，论文未做系统研究。

## 参考

- Ye, S. et al. *Latent Action Pretraining from Videos*（LAPA）. arXiv:2410.11758, 2024.
- Chen, Y. et al. *Moto: Latent Motion Token as the Bridging Language for Robot Manipulation*. arXiv:2412.04445, 2024.
- Chen, X. et al. *IGOR: Image-Goal Representations are the Atomic Control Units for Foundation Models in Embodied AI*. arXiv:2411.00785, 2024.
- NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734, 2025.
- Bu, Q. et al. *UniVLA: Learning to Act Anywhere with Task-Centric Latent Actions*. arXiv:2505.06111, 2025.
