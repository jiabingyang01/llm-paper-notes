# RynnWorld-4D：面向机器人操作的 4D 具身世界模型

> **论文**：*RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation*
>
> **作者**：Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, Zhongyu Li（Haoyu Zhao 与 Xingyue Zhao 共同一作；Siteng Huang、Deli Zhao、Zhongyu Li 为通讯作者）
>
> **机构**：DAMO Academy, Alibaba Group；Hong Kong Embodied AI Lab；CUHK；Hupan Lab
>
> **发布时间**：2026 年 07 月（arXiv 2607.06559）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.06559) | [PDF](https://arxiv.org/pdf/2607.06559)
>
> **分类标签**：`4D世界模型` `RGB-D-Flow` `视频扩散模型` `机器人操作` `逆动力学策略`

---

## 一句话总结

把预训练视频扩散模型（Wan 2.2-TI2V-5B）扩成三分支结构，从**单张 RGB-D 图 + 语言指令** 在一个去噪回路里**同步生成 RGB、深度、光流（RGB-DF）** 三路视频,从而显式地刻画场景的 4D 动态;再用一个逆动力学头（RynnWorld-4D-Policy）**单次前向** 读取世界模型的内部 4D 表征直接出动作,在真实双臂灵巧手上取得 SOTA,尤其在 Hand-over(28.57% vs 基础模型 0%)、Lid Placement/Bowl Stacking(均 65.71%)等需要精确空间协调的任务上大幅领先。

## 一、问题与动机

开放世界机器人操作不仅要"看懂场景长什么样",还要"预判 3D 结构在交互下如何运动"。作者的核心论点是:**纯 2D 像素视频世界模型丢失了关键的空间关系**,导致三类问题:

1. 无法做精确的 6-DoF 位姿估计与深度感知交互;
2. 缺乏几何锚定,时间上会出现物体尺度漂移、非物理的形状扭曲,破坏策略学习的鲁棒性;
3. 预测(2D 像素变化)与机器人真正要输出的 3D 动作之间存在巨大表征鸿沟。

已有 4D 场景建模路线各有短板:基于 NeRF/3DGS 的优化方法计算重、场景特定;前馈 4D 模型多依赖多视角输入或难以扩到复杂场景;动态 SfM 只能重建、不具备从单图预测未来的生成能力。作者主张把生成式世界建模从 2D 视频推进到**几何融合的 4D 场景演化**,是具身智能的必要一步。

关键设计选择:不用显式 3D 体素/4D 高斯,而是提出一个**投影式(projective)4D 表征**——让模型预测同步的 RGB、深度、光流序列。深度把每个像素抬升到 3D 位置,深度+光流又能在针孔相机假设下反投影出逐点 3D 场景流。这样几何与运动都被显式化,同时保持 **2D 对齐格式**,得以继承大规模视频扩散先验的可扩展性。

## 二、核心方法

### 2.1 RGB-DF 表征与 3D 场景流反投影

给定第 $t$ 帧的深度图 $D_t$,像素 $\mathbf{p}_t = [u,v,1]^\top$ 反投影到相机 3D 空间:

$$\mathbf{P}_t = D_t(u,v)\cdot \mathbf{K}^{-1}\mathbf{p}_t$$

其中 $\mathbf{K}$ 是相机内参,得到 metric 点云 $\mathcal{C}_t = \{\mathbf{P}_t^i\}_{i=1}^{H\times W}$。

> 用大白话说:深度值就是"这个像素离相机多远",配上相机内参就能把一张 2D 图"撑"成一片真实尺度的 3D 点云。

再用同步生成的光流 $\mathbf{f}_{opt} = [\Delta u, \Delta v]^\top$ 把 3D 点追踪到下一帧,得到 metric 场景流 $\mathbf{f}_{3D}$:

$$\mathbf{P}_{t+1} = D_{t+1}(u+\Delta u,\, v+\Delta v)\cdot \mathbf{K}^{-1}\big(\mathbf{p}_t + [\Delta u, \Delta v, 0]^\top\big),\quad \mathbf{f}_{3D} = \mathbf{P}_{t+1}-\mathbf{P}_t$$

> 用大白话说:光流告诉你像素在图像里往哪挪,深度告诉你它挪之后离相机多远;两者一拼,就知道每个点在真实三维空间里"走了多少",这才是物理上说得通的运动,而不是画面上的幻觉。之后用深度梯度 $\|\nabla D\| > \tau$ 的边缘滤波去掉深度不连续处的伪影,再投到 BEV 可视化。

### 2.2 三分支架构 + Joint Cross-Modal Attention

以 Wan 2.2-TI2V-5B(30 层 DiT,隐藏维 $d=3072$,FFN 维 14336)为骨干,把单分支 RGB 骨干**复制**成 RGB/深度/光流三分支——每个分支单独建模自己的分布(RGB 管纹理、深度管几何、光流管位移),避免不同模态互相干扰。三分支共享文本 cross-attention 的 K/V(语言语义与模态无关)。

跨模态一致性由 **Joint Cross-Modal Attention (JA)** 模块保证:每隔 3 层插一个,分布在全部 30 层(第 0,3,6,…,27 层),共 10 个,接在各分支帧内自注意力之后。

混合前,每个分支先加一个**零初始化**的可学习模态嵌入 $\mathbf{e}^m$ 并做逐模态 LayerNorm:

$$\tilde{\mathbf{z}}_l^m = \text{LN}^m(\mathbf{z}_l^m + \mathbf{e}^m)$$

每个分支只产生**一个 query 和一个共享 K/V**(被其他分支的 query 复用),把每块参数量从 $18d^2$ 降到 $12d^2$。token 从 $[B, T\cdot S, d]$ reshape 成 $[B\cdot T, S, d]$,使跨模态注意力**只在同一时间帧内** 发生,并对 Q、K 施加 3D RoPE 注入空间位置:

$$\mathbf{A}_l^m = \text{Attn}\big(\text{RoPE}(\mathbf{Q}_l^m),\ \text{RoPE}(\mathbf{K}_l^{\text{cross}}),\ \mathbf{V}_l^{\text{cross}}\big),\quad \mathbf{K}_l^{\text{cross}}=\text{concat}(\{\mathbf{K}_l^j\}_{j\neq m})$$

即每个模态的 query 只去看**另外两个互补模态**的 K/V。输出用零初始化投影 + 可学习门 $g_l^m$(初始化为 1)的 tanh 残差接回:

$$\tilde{\mathbf{z}}_l^m = \mathbf{z}_l^m + \tanh(g_l^m)\cdot \text{OutProj}_l^m(\mathbf{A}_l^m)$$

> 用大白话说:一开始 OutProj≡0,新插的跨模态通路等于"没接上",保证不破坏原视频模型的能力(平滑热启动);但门用 tanh(1)≠0,所以梯度能流进去,让它慢慢学着该开多大。这刻意避开了 ControlNet 那种双零初始化会卡在鞍点的死锁。3D RoPE 是关键的"对齐桥梁",让 JA 在像素级做空间感知融合,而非只做全局语义平均(去掉它 δ1 从 0.610 掉到 0.450)。

### 2.3 三阶段训练与 Branch Dropout

用 flow matching 目标训练。每个模态学一个速度场 $\mathbf{v}_\theta^m$,沿 $\mathbf{z}_t^m = (1-t)\mathbf{z}_0^m + t\boldsymbol{\epsilon}^m$ 把噪声运到数据;三模态**共享同一个高斯噪声** $\boldsymbol{\epsilon}^{\text{rgb}}=\boldsymbol{\epsilon}^{\text{depth}}=\boldsymbol{\epsilon}^{\text{flow}}$ 以保持去噪轨迹时序对齐。首帧是干净的 I2V 条件帧(真实 RGB、真实深度、零光流),不参与监督:

$$\mathcal{L}_{\text{total}} = \sum_{m\in\mathcal{M}} \lambda_m\, \mathbb{E}_{\mathbf{z}_0^m,\boldsymbol{\epsilon}^m,t,c}\Big[\big\|\mathbf{v}_\theta^m(\mathbf{z}_t^m,t,c)_{[1:]} - (\boldsymbol{\epsilon}^m - \mathbf{z}_0^m)_{[1:]}\big\|_2^2\Big]$$

三阶段课程弥合模态分布差距:

- **Stage 1 模态适配**:关掉 JA,三分支独立训练,让深度/光流分支从 RGB 先验里"改造"出几何/运动能力(LR 2e-5,$\lambda_{\text{flow}}$=0.5)。
- **Stage 2 联合注意力**:冻结骨干与各分支自注意力/FFN,只训 10 个 JA 相关参数(LR 5e-5,Branch Dropout 0.2)。
- **Stage 3 全参联合 SFT**:解冻整模在全量数据上微调(LR 1e-5,Branch Dropout 0.1)。

**Branch Dropout**:以概率 $p_{\text{drop}}$ 随机把 {深度, 光流} 之一的噪声潜变量替成纯高斯噪声,逼 JA 从可见模态重建它;RGB 分支**永不 drop**(它是外观锚点)。

### 2.4 RynnWorld-4D-Policy(逆动力学头)

把冻结的 RynnWorld-4D 当作**预测式 4D 视觉编码器**:单次前向抽取三分支中间隐状态(第 15 个 block,扩散 timestep $t=500$),沿通道拼成 $F_p \in \mathbb{R}^{B\times T\times 3C\times H\times W}$。再用带可学习 query 的 **Flow Former** 压缩:

$$\mathbf{Q}_i' = \text{Spat-CrossAttn}(\mathbf{Q}_i, F_p[i]),\quad \mathbf{Q}'' = \text{FFN}(\text{Temp-SelfAttn}(\mathbf{Q}'))$$

最后用 flow matching 策略头,以 4D token $\mathbf{Q}''$、文本嵌入、本体感知 $p_0$ 为条件生成动作,推理时 ODE 求解 N=4 步,一次预测 K=10 个动作块(action chunking)。

> 用大白话说:世界模型只跑一次前向(N=1)拿到"想象出来的未来 4D",策略头再在这个轻量表征上跑 4 步 ODE 出动作——**绕开了世界模型逐步去噪的昂贵开销**,才能做到闭环实时控制。

### 2.5 数据集 Rynn4DDataset 1.0

254.4M 帧、2354.9 小时 @30fps、7 个来源:

| 大类 | 帧数占比 | 组成 |
|---|---|---|
| 人类第一视角演示 | 20.6M（8.1%） | EgoVid 15.9M、Epic-Kitchens 4.7M |
| 具身交互序列 | 233.8M（91.9%） | AgiBot 158.4M、RoboCoin 51.1M、Galaxea 14.6M、RoboMIND 7.7M、RDT-1B 2.0M |

伪标注管线:Qwen3-VL 生成 caption(1 FPS 采样、5 秒切片);DPFlow 估光流(25 FPS);Depth Anything 3(DA3NESTED-GIANT-LARGE-1.1)估深度(工作分辨率 392,深度裁剪到 $[0.0, 5.0]$ 米,按 $I=\lfloor d/d_{\max}\times 255\rfloor$ 量化成 8-bit 灰度)。

## 三、实验结果

**推理与硬件**:RTX 5090、FP8 + FlashAttention 3。总前向 ~1106 ms(≈0.9 Hz 规划频率),其中三分支 Transformer 占 **89.5%**(990 ms)是主要瓶颈;靠 K=10 动作块并行执行,有效控制频率 **≈9 Hz**,机器人以 50 Hz 从缓存回放已规划动作。真机平台为 TIANJI M6(7-DOF 臂)+ WUJI Hand(20-DOF 灵巧手),双臂共 54 DOF,RealSense D435i 第一视角。

### 3.1 4D 世界建模质量(测试集 50 段视频)

| 方法 | IQ↑ | SC↑ | SSIM↑ | PSNR↑ | LPIPS↓ | AbsRel↓ | δ1↑ | AEPE↓ |
|---|---|---|---|---|---|---|---|---|
| Wan-2.1-I2V-14B | **0.684** | 0.891 | 0.536 | 12.72 | 0.568 | N/A | N/A | N/A |
| Free4D | 0.354 | 0.787 | 0.492 | 12.40 | 0.597 | 0.804 | 0.179 | N/A |
| TesserAct | 0.608 | 0.904 | 0.693 | 16.91 | 0.335 | 0.699 | 0.279 | N/A |
| 4DNeX | 0.637 | 0.917 | 0.649 | 14.47 | 0.404 | 0.423 | 0.327 | N/A |
| **RynnWorld-4D** | 0.635 | **0.957** | **0.754** | **17.85** | **0.269** | **0.310** | **0.610** | **0.170** |

关键点:视觉质量 IQ 与顶级视频模型持平,但重建保真度(SSIM/PSNR/LPIPS)明显更好;几何上 δ1=0.610 **几乎翻倍** 于 4DNeX(0.327)与 TesserAct(0.279);且是唯一能产出显式光流的方法(AEPE 0.170)。

**消融(世界模型侧)**:

| 变体 | AbsRel↓ | δ1↑ | AEPE↓ | 说明 |
|---|---|---|---|---|
| Independent Branches | 0.737 | 0.245 | 0.247 | 去掉三分支互动,深度/光流大幅退化 |
| w/o Modality Adaptation | 0.507 | 0.479 | 0.231 | 无 Stage 1,几何精度掉 |
| w/o 4D Pre-training | 0.797 | 0.263 | **0.729** | 只用任务特定数据,AEPE 暴涨 |
| w/o RoPE in JA | 0.420 | 0.450 | 0.210 | 去 3D RoPE,几何对应被破坏 |
| shared FFN | 0.580 | 0.380 | 0.280 | 共享 FFN 引发灾难性干扰 |
| **完整模型** | **0.310** | **0.610** | **0.170** | — |

### 3.2 真机操作成功率(%,每任务 35 次连续试验)

| 方法 | Dual Picking | Block Pushing | Hand-over | Bimanual Lifting | Lid Placement | Bowl Stacking |
|---|---|---|---|---|---|---|
| DP | 77.14 | 85.71 | 17.14 | 88.57 | 57.14 | 57.14 |
| π0 | 88.57 | 94.29 | 2.86 | 91.43 | 34.29 | 51.43 |
| π0.5 | 94.29 | **100.00** | 0.00 | 94.29 | 37.14 | 42.86 |
| **RynnWorld-4D-Policy** | **94.29** | 97.14 | **28.57** | **97.14** | **65.71** | **65.71** |

亮点:在需要精确空间协调的 Lid Placement 与 Bowl Stacking 上均达 65.71%,超次优基线(DP)8.57 个百分点;最惊人的是 **Hand-over**(机器人内部左右手传递)——基础模型几乎全失败(π0.5=0%、π0=2.86%),本方法 28.57%。作者归因:基础模型预训练偏向平行夹爪、缺灵巧手先验,且 2D 策略难以推理两只高自由度末端间的相对 3D 距离与自遮挡。

**消融(策略侧)**:去掉 RynnWorld-4D 改用 ResNet-18 编码器,Dual Picking 从 94.29% 掉到 71.43%;逐模态贡献显示 RGB+Depth 提升空间精度类任务(Hand-over 28.57%、Bimanual Lifting 97.14%),RGB+Optical Flow 提升运动敏感任务,三者齐上才最优。

## 四、局限性

1. **控制频率受限**:4D 序列生成依赖扩散去噪,即使 FP8 + FA3,RTX 5090 上有效控制频率也仅 ≈9 Hz(三分支 Transformer 占 89.5% 延迟),对超高频精细控制仍是瓶颈。
2. **视角单一**:模型主要针对第一视角(egocentric)优化,如何把 4D 时空一致性扩到多视角系统或多机器人协作仍是开放问题。
3. **深度是伪标签、量化到 5 米**:训练深度来自 Depth Anything 3 单目估计并裁剪/量化到 8-bit,几何"真值"本身有上限,长距离或反光/透明物体的几何可靠性存疑(文中未评估)。
4. **真机评测规模**:每任务仅 35 次试验、6 个任务、单一硬件(TIANJI M6 + WUJI Hand),统计显著性与跨本体泛化未充分验证。
5. **场景流仅用于可视化/策略条件**,并未直接把 metric 场景流当监督或几何一致性损失回灌世界模型,反投影质量对相机内参与深度尺度较敏感。

## 五、评价与展望

**优点**:
- **表征选型精妙**。RGB-DF 把"要几何/运动显式化"与"要继承视频扩散可扩展性"这对矛盾用 2D 对齐的多模态格式化解,是本文最有价值的洞见——比 TesserAct 的 RGB-D-Normal 多了光流这一显式动力学线索,而光流恰是反投影出 3D 场景流的关键,对灵巧操作(末端与物体的细粒度轨迹决定成败)尤其重要。
- **工程细节扎实**。JA 的"1 query + 共享 KV"降参、tanh 门规避 ControlNet 鞍点、帧内 3D RoPE、Branch Dropout、三阶段课程,每一处都有对应消融支撑,而非堆砌。
- **世界模型→策略的一体化**很清爽:同一个冻结骨干既是生成器又是逆动力学的视觉编码器,单次前向绕开逐步去噪,这是它能上真机闭环的前提。
- Hand-over 上对 π0/π0.5 的碾压性优势,较有说服力地论证了"显式 4D 表征对双末端空间推理不可替代"。

**存疑/可改进**:
- **δ1 翻倍的对比稍显不公**。4DNeX/TesserAct 的深度是网络副产物,而本文深度分支与 depth 伪标签同源(都可追溯到 DA3 类估计),存在评测与训练分布同源的嫌疑;若换用带激光真值的基准(如真实 metric 深度)再比,优势幅度待观察。
- **9 Hz 的"有效频率"靠 action chunking 撑起**,规划频率其实只有 0.9 Hz,一旦物体在 1.1 秒执行窗内被大幅扰动,重规划延迟可能暴露;作者称潜变量覆盖"空间体积而非像素点"故恢复范围更宽,但缺乏对抗扰动的定量实验。
- 与 UniPi/VPP/GR 系列"用未来预测引导策略"及 TesserAct/4DNeX 的关系交代清楚,但**未与同期同样引入光流/场景流的操作策略**(如 NovaFlow 等 actionable-flow 工作)做直接对比,4D 相较 3D-flow 的增量收益尚未被单独 isolate。

**开放问题**:(1)把 metric 场景流作为几何一致性损失回灌,而非仅可视化;(2)蒸馏/一致性模型把三分支去噪压到 1-2 步以突破 9 Hz;(3)多视角/多机协作下的 4D 一致性;(4)在带真值几何的仿真-真机联合基准上做更公平的几何评测。总体上,这是把 4D 生成式世界模型真正落到双臂灵巧手闭环的一篇扎实工作,RGB-DF 表征值得后续跟进。

## 参考

1. Wang et al., *Wan: Open and Advanced Large-Scale Video Generative Models*, arXiv 2025 — 本文三分支骨干(Wan 2.2-TI2V-5B)。
2. Zhen et al., *Learning 4D Embodied World Models*(TesserAct), CVPR 2025 — 最相关的 RGB-D-Normal 4D 世界模型基线。
3. Chen et al., *4DNeX: Feed-forward 4D Generative Modeling Made Easy*, arXiv 2025 — 前馈 4D 生成基线,几何指标主要对手。
4. Lipman et al., *Flow Matching for Generative Modeling*, arXiv 2022 — 世界模型与策略头共用的 flow matching 训练目标。
5. Lin et al., *Depth Anything 3*, arXiv 2025 — Rynn4DDataset 深度伪标注来源。
