# RepWAM：基于表征化视觉-动作分词器的世界动作建模

> **论文**：*RepWAM: World Action Modeling with Representation Visual-Action Tokenizers*
>
> **作者**：Junke Wang, Qihang Zhang, Shuai Yang, Yiming Luo, Yujun Shen, Zuxuan Wu, Yu-Gang Jiang, Yinghao Xu et al.
>
> **机构**：复旦大学可信具身智能研究所（Institute of Trustworthy Embodied AI, Fudan University）；Robbyant, Ant Group；香港科技大学
>
> **发布时间**：2026 年 06 月（arXiv 2606.13674）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.13674) | [PDF](https://arxiv.org/pdf/2606.13674)
>
> **分类标签**：`世界动作模型` `视觉动作分词器` `潜在动作` `表征对齐` `RoboTwin 2.0`

---

## 一句话总结

RepWAM 用一个与冻结视觉基础模型（Perception Encoder）语义对齐的视觉分词器、并在同一语义空间内以"转移"形式学习潜在动作（可迁移动作分词器），替代了以往世界动作模型（WAM）直接沿用重建导向视频分词器（如 WAN VAE）的做法；RepWAM-5B 在 RoboTwin 2.0 全部 50 个任务上取得 Easy 89.3 / Hard 88.4 的平均成功率，在真实 Franka 双臂插试管任务上达到 60% 成功率（比 π0.5 高 50 个百分点），且消融证实语义对齐分词器相比同规模 WAN2.2 VAE 可将 RoboTwin 2.0 平均成功率从 78.0/76.0 提升到 86.6/83.1。

## 二、问题与动机

WAM 把机器人操作建模为两阶段因果过程：世界模型专家预测下一时刻视觉观测，动作专家（常以逆动力学模型 IDM 实现）从该转移中推断动作，两阶段通常都在预训练视频分词器（如 WAN）产生的像素级隐空间 $z_{1:T}$ 上运行。作者指出这一设计存在两个结构性缺陷：（1）视觉侧，重建导向的分词器把隐空间容量主要花在背景纹理等低层外观上，物体身份、空间关系、交互线索等操作相关语义反而欠表征，导致语言指令难以在隐空间中被有效"接地"；（2）动作侧，视觉隐变量与电机指令处于互不相通的空间，每一步都要靠 IDM 强行桥接模态鸿沟，动作专家与世界模型专家在表征上是解耦的。论文提出的核心问题是："世界动作模型到底需要什么样的表征？"，并给出答案：视觉隐空间应语义对齐，潜在动作应被定义为该语义空间内的状态转移，而非从原始像素变化中提取的孤立编码。

## 三、核心方法

RepWAM 由表征视觉-动作分词器（RepViTok）+ 因果世界动作模型两部分组成，训练分三阶段：分词器训练 → WAM 预训练（潜在动作） → 具身适配（真实动作）。

**1）语义视觉分词器**：ViT 自编码器，首帧切成 16×16 patch，后续帧切成 4×16×16（时间×高×宽）tubelet，编码器 $E_\theta$ 采用时间因果掩码 + 帧内全空间注意力，输出视觉隐变量 $z \in \mathbb{R}^{T'L\times d_v}$；解码器 $D_\theta$ 对称，末端用转置卷积反 patch 化。重建目标为

$$\mathcal{L}_{\text{rec}} = \lambda_1\|o-\delta\|_1 + \lambda_{\text{perc}}\mathcal{L}_{\text{perc}}(o,\delta) + \lambda_{\text{gan}}\mathcal{L}_{\text{gan}}(\delta)$$

（大白话：像素级 L1 + 感知损失 + 对抗损失三件套，是标准的视频自编码器重建目标。）在此之上再加一个语义对齐损失，把隐变量投影后逼近冻结视觉基础模型（Perception Encoder）的时间平均池化特征：

$$\mathcal{L}_{\text{align}} = \left\|W_{\text{align}}z - \text{avg}(G(o))\right\|_2^2$$

（大白话：强迫分词器的隐空间"长得像"一个已经学好语义的现成视觉模型的特征空间，而不是自由发挥地只顾像素还原。）总损失 $\mathcal{L}_{\text{vis}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}}$。

**2）潜在动作分词器（LAT）**：冻结上一步的视觉分词器，训练 IDM $q_\phi$ 与 FDM $f_\psi$，把相邻两帧隐变量的转移压缩为低维潜在动作 $\ell_t \in \mathbb{R}^{d_\ell}$（$d_\ell \ll d_v$，防止内容泄漏），再由 FDM 把 $\ell_t$ 还原为一个"软传输算子" $K_t$ 与残差 $\delta_t$：

$$\ell_t = q_\phi(z_t,z_{t+1}), \quad K_t,\delta_t = f_\psi(z_t,\ell_t), \quad \hat{z}_{t+1} = K_t z_t + \delta_t$$

（大白话：$K_t$ 类似光流思想，在语义 token 空间里把视觉内容按状态变化"搬运"到新位置，$\delta_t$ 补充搬运解释不了的残余变化；这样潜在动作描述的是任务级状态转移而非具体机械臂的电机坐标，因而更容易跨具身迁移。）训练目标为前向下一隐变量预测损失加反向一致性损失（在反转的 $(z_{t+1},z_t)$ 对上再跑一次 LAT 得到 $\hat{z}_t$）：

$$\mathcal{L}_{\text{fwd}} = \sum_{t=1}^{T'-1}\|\hat{z}_{t+1}-z_{t+1}\|_2^2, \qquad \mathcal{L}_{\text{cons}} = \sum_{t=1}^{T'-1}\|\hat{z}_t-z_t\|_2^2$$

**3）因果世界动作模型**：语言指令经预训练文本编码器编成条件 token $c$；把视觉 token 与潜在动作按时间窗打包成 chunk $u_{t:t+k}=[z_{t:t+k},\ell_{t:t+k-1}]$，序列 $s$ 以语言 token 与初始视觉上下文为前缀，用块因果掩码让每个 chunk 只能看到之前的 chunk；视觉与动作 token 共享注意力权重但使用各自模态的前馈网络。训练采用联合条件流匹配：采样噪声 $\epsilon$ 与时间标量 $\alpha$，构造线性插值 $x_\alpha=(1-\alpha)\epsilon_{t:t+k}+\alpha u_{t:t+k}$，目标速度 $\dot x_\alpha = u_{t:t+k}-\epsilon_{t:t+k}$，损失为视觉分支与动作分支速度回归 MSE 之和（$\lambda_a$ 加权动作项）。（大白话：世界模型专家和动作专家共享一个因果 Transformer 骨干、用同一套流匹配范式同步去噪，只是各自专属 FFN 和损失权重不同。）预训练阶段动作 token 用潜在动作 $\ell_t$（无标注视频即可训练），随后在真实机器人示教数据上把潜在动作替换为真实连续电机指令（末端位姿+夹爪）做具身适配，实现闭环控制。RepViTok 在 Panda-70M 上训练；WAM 在 AgiBot（约 100G 视频-动作隐 token）上预训练；具身适配阶段混合 AgiBot、RoboMIND、RoboCOIN、InternData-A1（约 300G token）。模型分 1.3B / 5B 两档，均为 30 层因果扩散 Transformer 从零训练（不借助预训练视频生成骨干），动作专家共享深度、隐藏维度缩减，额外约 350M 参数；用 Muon 优化器，峰值学习率 1e-2。

## 四、关键结果

**真实机器人（Franka 双臂，3 任务，10 次 rollout，50 条示教微调 500 步）**：

| 任务 | π0.5 | Lingbot-VA | RepWAM-1.3B | RepWAM-5B |
|---|---|---|---|---|
| Pick the fruit | 10% | 50% | 60% | 60% |
| Push the drawer | 50% | 70% | 50% | **80%** |
| Insert the tube | 10% | 40% | 30% | **60%** |

RepWAM-5B 在每个任务上均为最佳或并列最佳，长视野（push drawer）和精细接触（insert tube）任务上优势最大。

**RoboTwin 2.0（50 任务，官方随机化设置，平均成功率）**：

| 模型 | 骨干预训练 | Easy | Hard |
|---|---|---|---|
| π0.5 | ✓ | 82.7 | 76.8 |
| Motus | ✓ | 88.7 | 87.0 |
| Lingbot-VA | ✓ | 92.6 | 91.6 |
| RepWAM-1.3B | ✗（从零） | 86.6 | 83.1 |
| RepWAM-5B | ✗（从零） | 89.3 | 88.4 |

作者将与 Lingbot-VA 的差距归因于后者使用了 WAN 视频生成预训练权重，而 RepWAM 完全从零训练。

**消融 1（分词器语义 vs 重建，AgiBot Eval Seen/Unseen + PickFruit 闭环）**：RepViTok 相比 WAN2.2 VAE，gFVD 降低 9.5%/13.2%（Seen 61.01 vs 67.42，Unseen 72.91 vs 83.98），OLS（开环动作准确率）从 13.68/11.21 提升到 18.82/14.15，闭环 PickFruit 成功率从 20%/10%（WAN2.2 VAE / ViTok）提升到 30%。将 WAN2.2 VAE 直接替换为 RepViTok（其余保持 1.3B WAM 设置不变）后，RoboTwin 2.0 平均成功率从 78.0/76.0 升至 86.6/83.1。

**消融 2（潜在动作训练策略）**：w/o 潜在动作（直接在真实动作上训练）PickFruit 30%；Joint Pred（单一预测头联合预测视觉与动作隐变量）gFVD 反而恶化到 94.25/98.77，PickFruit 20%；论文提出的 Two Stages（先在语义视觉+潜在动作上预训练，再适配真实动作）在所有指标上最优（gFVD 48.23/58.83，PSNR 22.86/19.93，OLS 19.87/16.98，PickFruit 50%）。

**其他发现**：RepViTok 学出的潜在动作可视化上更聚焦于物体位移、接触诱发的运动区域（相比 LAPA 更弥散）；冻结潜在动作后训练同一 IDM 解码真实动作，RepViTok 的动作损失低于 LAPA，说明其潜在动作更易映射到机器人动作空间。视频无分类器引导（CFG）消融显示 RepWAM 在 CFG scale=1.0（即不额外做 CFG 外推）时平均成功率最高（88.9%），且随 CFG 增大不再稳定提升，而 Lingbot-VA（依赖 WAN 预训练）则随 CFG 增大持续受益，说明语义对齐的隐空间本身已具备较强的指令对齐性，减弱了对 CFG 的依赖。重建质量上（ImageNet/UCF101），RepViTok 与 WAN2.2 VAE 基本持平甚至局部更优（如 UCF101 512×17 rFVD 0.16 打平 ViTok、优于 WAN2.2 VAE 的 0.68），说明语义对齐目标并未明显牺牲重建保真度。

## 五、评价与展望

**优点**：论文抓住了当前 WAM 研究中一个常被忽视但结构性的问题——视觉分词器的语义贫乏与视觉-动作模态鸿沟，并给出了干净的两阶段解法（先对齐视觉语义，再在该语义空间内定义动作为转移）。实验设计较为扎实：既有真实机器人闭环验证，也有 RoboTwin 2.0 大规模仿真基准；消融覆盖了分词器语义化收益（Table 2/3）、潜在动作训练范式选择（Table 4）、CFG 依赖性（Figure 5）、重建保真度（Table 5）多个维度，逐层拆解验证了核心假设，而不是仅报告端到端数字。转移算子 $K_t$（受光流启发的"软传输"）是一个有一定新意的设计，把潜在动作的物理直觉——"内容如何被搬运"——显式建模出来，区别于把动作当作孤立离散码本（如 LAPA、Genie）的做法。

**局限**：（1）RepWAM 完全从零训练、未借助预训练视频生成骨干，在 RoboTwin 2.0 上仍落后于使用 WAN 预训练的 Lingbot-VA（Easy 89.3 vs 92.6，Hard 88.4 vs 91.6），语义对齐是否能与大规模视频生成先验叠加而非二选一，论文未给出联合方案；（2）真实机器人评测规模有限——仅 3 个任务、每任务 10 次 rollout、每任务仅 50 条示教微调，样本量较小，数字的统计稳健性有限；（3）潜在动作的跨具身可迁移性目前只有 IDM 损失曲线和定性可视化两类间接证据（Figure 4），未在多个真实具身形态间做系统的量化迁移实验；（4）论文自陈的未来方向——把预训练扩展到大规模互联网视频尤其是第一人称人类视频——仍是待验证的开放问题，人类视频与机器人动作空间之间的对齐是否需要额外机制目前未知。

**与相关工作的关系**：RepWAM 与 Lingbot-VA（因果世界建模+自回归扩散策略执行）、DreamZero（从预训练视频生成模型初始化做联合视频-动作建模）、Fast-WAM（论证测试时是否真的需要显式想象再执行）、Motus（专家混合架构分派理解/生成/逆动力学/动作预测）共同构成了近期 WAM 路线的谱系；相较于 LAPA、Moto 等把潜在动作作为独立于视觉分词器的附加信号，RepWAM 的差异化贡献在于把视觉分词器与潜在动作分词器统一到同一个语义空间中联合设计，是对"WAM 表征该长什么样"这一问题的一次系统性回答，而非单纯的架构或数据规模扩展。

## 参考

- Li et al. *Causal world modeling for robot control* (Lingbot-VA). RSS 2026.
- Ye et al. *World action models are zero-shot policies* (DreamZero). arXiv:2602.15922, 2026.
- Yuan et al. *Fast-WAM: Do world action models need test-time future imagination?* arXiv:2603.16666, 2026.
- Ye et al. *Latent action pretraining from videos* (LAPA). ICLR 2025.
- Chen et al. *RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation.* arXiv:2506.18088, 2025.
- Bolya et al. *Perception Encoder: The best visual embeddings are not at the output of the network.* NeurIPS 2026.
