# JOPAT：点跟踪提升世界动作模型

> **论文**：*Point Tracking Improves World Action Models*
>
> **作者**：Jiarui Guan, Wenshuai Zhao†, Yue Pei, Ziliang Chen, Arno Solin, Juho Kannala（† 通讯作者）
>
> **机构**：Aalto University；ELLIS Institute Finland；University of Oulu；Sun Yat-sen University；Peng Cheng Laboratory；Beihang University
>
> **发布时间**：2026 年 05 月（arXiv 2605.23856）
>
> **发表状态**：未录用（预印本，标注 "Preprint"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.23856) | [PDF](https://arxiv.org/pdf/2605.23856)
>
> **分类标签**：`world-action-model` `2D点跟踪` `扩散Transformer` `动作无关视频预训练` `LIBERO` `占据/可见性预测`

---

## 一句话总结

JOPAT（JOint Pixel-And-Track World-Action Model）用一个统一的去噪 Diffusion Transformer 同时预测未来视觉隐向量、**带可见性的 2D 点跟踪**与机器人动作，把"点轨迹"作为像素之外显式的、对遮挡和出画鲁棒的运动接口；在 LIBERO 40 任务上取得 97.8% 平均成功率（排名第一，超过 CogVLA 的 97.4%），在 LeRobot SO-101 真机四任务上平均成功率 57.5%，比 ACT、UWM 分别高 17.5、25.0 个百分点，且在长时序、遮挡、分布外扰动场景下增益最大。

## 一、问题与动机

Vision-Language-Action（VLA）模型把预训练视觉语言模型迁移到动作生成上，语义能力强，但训练数据以图文为主，缺乏对环境动力学的理解。世界动作模型（World-Action Model, WAM）试图通过将动作生成与未来世界状态预测耦合来补上这一环：模型学习一个联合分布

$$p_\theta(a_{t:t+K-1}, s_{t:t+H} \mid \mathbf{c}_t),$$

策略可以看作是对这个更大预测分布的动作边际 $\pi_\psi(a_{t:t+K-1} \mid \mathbf{c}_t)=\int p_\theta(a_{t:t+K-1}, s_{t:t+H} \mid \mathbf{c}_t)\, ds$。这样一来策略学习就转化成了一个"接口设计"问题：预测什么样的未来世界状态 $s_{t:t+H}$ 才对动作生成真正有用？

最常见的做法是**像素隐向量式 WAM**（pixel-latent WAM）：用编码器 $E_o$ 把未来观测窗口压缩成紧凑视觉隐向量 $\mathbf{z}^o_{t+H:t+H+1}=E_o(o_{t+H:t+H+1})$，与动作 token 一起用共享 Transformer 联合去噪：

$$[\hat\epsilon^a,\hat\epsilon^o]=T_\theta^{\text{WAM}}([\tilde{\mathbf z}^a,\tilde{\mathbf z}^o,\mathbf r];\mathbf c_t,\tau_a,\tau_o).$$

**用大白话说**：这类模型能借助动作无关视频做预训练（因为只需要预测像素），但其状态接口被外观（appearance）主导——纹理、光照、背景这些与控制无关的因素和物体真实位移混在同一套隐向量里，动作 token 必须隐式地从"图像变了"中反推出"哪个部件动了、往哪动了、证据是否被遮挡"，这是一个不必要的额外负担。作者指出，几项已有研究也发现动作无关视频上的像素级预训练并不总能提升下游策略成功率。

JOPAT 的核心主张：**把运动对应关系显式化**。作者提出用带可见性的 2D 点跟踪（point tracks with visibility）作为像素之外的第二个未来状态分量——轨迹编码跨帧的持久场景对应关系，可见性标记这些对应关系何时被遮挡或离开画面，从而把"物体去哪了""证据是否还在"这两个动作相关的关键问题显式暴露给去噪过程，同时视觉隐向量继续保留物体身份、可供性（affordance）等语义信息。

## 二、核心方法

### 2.1 统一像素-轨迹-动作架构

给定当前观测窗口 $o_{t-1:t}$（2 帧 RGB），条件编码器 $E_c$（ResNet-18，ImageNet 初始化）产生全局条件特征 $\mathbf c_t=E_c(o_{t-1:t})$，通过 AdaLN 注入每个 DiT block，并附带各模态独立的扩散时间步 $\tau_a,\tau_o,\tau_p$。JOPAT 在同一序列中联合去噪动作、未来视觉隐向量与轨迹 token：

$$\mathbf Z=[\tilde{\mathbf z}^a_{t:t+K-1},\ \tilde{\mathbf z}^o_{t+H:t+H+1},\ \tilde{\mathbf z}^p_{t:t+H_p-1},\ \mathbf r],$$

DiT 风格 Transformer 对该序列做双向全注意力，预测各模态噪声：

$$\hat\epsilon^a,\hat\epsilon^o,\hat\epsilon^p=T_\theta(\mathbf Z;\mathbf c_t,\tau_a,\tau_o,\tau_p).$$

**用大白话说**：动作、未来画面、未来轨迹三种 token 拼在一起过同一个 Transformer，通过自注意力互相"看见"彼此——动作生成不再只能间接感知想象中的未来像素，还能直接读取想象中的点位移与可见性。实现里用 2 帧观测、未来偏移 $H=16$，轨迹/动作视界 $H_p=K=19$，未来视觉窗口预测 2 帧。视觉目标用冻结的 SDXL VAE 编码为 $28\times28\times4$ 隐向量；Transformer 深度 12 层、隐藏维 768、12 个注意力头、8 个可学习寄存器 token（register tokens）。

### 2.2 轨迹的构造与"轨迹当视频"编码

以当前帧 $o_t$ 为参考帧，在其上撒 $N$ 个网格查询点，用现成点跟踪器 CoTrackerV3 把它们向未来跟踪 $H_p$ 步，得到

$$\mathbf P^{(t)}\in\mathbb R^{H_p\times N\times2},\qquad \mathbf V^{(t)}\in\{0,1\}^{H_p\times N}.$$

为了紧凑编码轨迹，JOPAT 把点维度重新排回空间网格（$N=625=25\times25$），构造 $\mathbf G^p\in\mathbb R^{2\times H_p\times H_g\times W_g}$（$H_gW_g=N$），再套 3D 卷积 patchifier：

$$\mathbf z^p=E_p(\mathbf P^{(t)})=\text{Patchify}_{3D}(\mathbf G^p)\in\mathbb R^{L_p\times d}.$$

去噪后，轨迹 token 被解码为坐标与可见性预测：

$$\hat{\mathbf P}^{(t)}=D_p(\hat{\mathbf z}^p),\qquad \hat{\mathbf V}^{(t)}=D_v(\hat{\mathbf z}^p)\in\mathbb R^{H_p\times N}.$$

**用大白话说**：与其把 625 个点的轨迹当成一串独立数字序列，不如把"时间×网格行×网格列×(x,y)"这四维张量当成一段"运动视频"，用时空卷积（patch size $2\times5\times5$）打成 patch token——这样轨迹 token 与视觉 patch token 天然对齐、量级相当，能公平地和图像 token、动作 token 挤进同一个 Transformer。特别要注意：可见性只作为**输出**被预测，**不会**作为输入喂回轨迹编码器，避免训练时泄漏真值可见性给去噪过程。

### 2.3 训练目标与推理

对每个模态 $m\in\{a,o,p\}$ 独立采样扩散时间步 $\tau_m$，做标准 DDPM 前向加噪与去噪目标：

$$\mathbf x^m_{\tau_m}=\sqrt{\bar\alpha_{\tau_m}}\,\mathbf x^m_0+\sqrt{1-\bar\alpha_{\tau_m}}\,\epsilon^m,\qquad \mathcal L_m=\|\hat\epsilon^m-\epsilon^m\|_2^2.$$

轨迹坐标损失 $\mathcal L_p$ 只作用于 2D 坐标；可见性由独立的头用二元交叉熵监督：

$$\mathcal L_{\text{vis}}=\frac{1}{H_pN}\sum_{\tau=0}^{H_p-1}\sum_{i=1}^{N}\text{BCE}(\hat V_{t+\tau,i},V_{t+\tau,i}).$$

动作标注演示 $\mathcal D$ 与动作无关视频 $\mathcal V$ 分别用如下目标训练：

$$\mathcal L_{\mathcal D}=\mathcal L_a+\lambda_o\mathcal L_o+\lambda_p\mathcal L_p+\lambda_{\text{vis}}\mathcal L_{\text{vis}},\qquad \mathcal L_{\mathcal V}=\lambda_o\mathcal L_o+\lambda_p\mathcal L_p+\lambda_{\text{vis}}\mathcal L_{\text{vis}}.$$

**用大白话说**：面对没有动作标签的视频，直接把动作分支的输入/损失全部屏蔽，只让模型学"未来画面长啥样、点怎么动、什么时候看不见"；这样同一套架构既能吃海量动作无关视频学运动先验，又能在少量机器人演示上把这些先验"接地"到可执行动作。推理时用 DDIM（100 步训练、10 步采样）从高斯噪声联收敛地初始化动作/视觉/轨迹三路 token 并联合去噪，只解码执行动作分支（视觉、轨迹 token 只是中间变量，通过自注意力影响动作但不被直接执行），实际部署每次去噪出一个动作 chunk 后执行前 8 步再重规划（receding-horizon）。全部损失权重 $\lambda_a=\lambda_o=\lambda_p=\lambda_{\text{vis}}=1$。训练用 4×NVIDIA H200 做动作无关预训练约 5 天，单卡 H200 做任务微调约 1 天，部署时单张 RTX 4090 上推理约 10Hz。

## 三、实验结果

### 3.1 LIBERO 仿真基准（Q1：主性能）

40 个任务（Spatial / Object / Goal / Long 四个 suite，各 10 任务、每任务 50 条演示），在 LIBERO-90 上预训练 100K 步后按目标 suite 微调 10K 步。

| 方法 | Spatial | Object | Goal | Long | Average SR | 排名 |
|---|---|---|---|---|---|---|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 | 15 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 13 |
| π0 fine-tuned | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | 7 |
| π0.5-KI | 98.0 | 97.8 | 95.6 | 85.8 | 96.0 | 4 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | 3 |
| STAR | 95.5 | 98.3 | 95.0 | 88.5 | 94.3 | 6 |
| CogVLA | 98.6 | 98.8 | 96.6 | 95.4 | 97.4 | 2 |
| UWM | 82.3 | 92.2 | 86.8 | 77.6 | 84.7 | 9 |
| **JOPAT（本文）** | 97.2 | 98.9 | 98.4 | **96.4** | **97.8** | **1** |

JOPAT 在四个 suite 的平均成功率上排名第一，且优势最集中体现在时序最长、最考验遮挡与长程一致性的 *Long* suite（96.4% vs. CogVLA 95.4%、π0.5-KI 85.8%）。

### 3.2 真机实验（LeRobot SO-101，4 任务）

| 方法 | Cook-Soup | Insert-Peg | Push-Tomato | Pick-Grocery | 平均 |
|---|---|---|---|---|---|
| ACT | 40% | 0% | 70% | 50% | 40% |
| UWM | 10% | 0% | 80% | 40% | 32.5% |
| **JOPAT** | **60%** | **10%** | **100%** | **60%** | **57.5%** |

JOPAT 在 Push-Tomato 达到 100%（接触密集的物体位移任务，得益于轨迹提供的运动接口），Insert-Peg（毫米级插孔）对所有方法都困难，作者认为这需要额外的 3D/接触感知，而非 2D 轨迹 + RGB 隐向量可以解决。

### 3.3 关键消融

**联合 vs. 单模态**（LIBERO-Long，DROID 动作无关视频预训练相同设置下）：

| 变体 | 平均 SR |
|---|---|
| Latent-only（去掉轨迹预测） | 77.4 |
| Track-only（去掉视觉隐向量预测） | 26.2 |
| **Joint（本文）** | **96.4** |

联合建模比 Latent-only 高 19.0 点，比 Track-only 高 70.2 点；真机上对应 Latent-only 35.0% / Track-only 15.0% / Joint 57.5%（分别 +22.5 / +42.5 点）。作者解释：轨迹提供对应关系级别的运动约束防止长程漂移，视觉隐向量提供语义/可供性防止"动作正确但物体认错"，两者必须耦合才能起效。

**可见性预测消融**（真机）：去掉可见性头，平均 SR 从 57.5% 降到 47.5%，在 Cook-Soup（长时序反复交互、易被遮挡）上收益最大，Pick-Grocery 几乎不变——说明可见性头主要贡献"缺失证据建模"而非单纯扩容。

**动作无关视频预训练是否有用**（LIBERO-Long，不同演示量）：

| 方法 | 10 demos | 25 demos | 50 demos |
|---|---|---|---|
| JOPAT w/o 预训练 | 11.9% | 31.6% | 66.1% |
| JOPAT（DROID 预训练） | 64.2% | 82.7% | 96.4% |
| JOPAT（OpenVid-1M 预训练） | 48.5% | 84.6% | 95.1% |

10 条演示时 DROID 预训练把成功率从 11.9% 拉到 64.2%，连通用视频数据集 OpenVid-1M 预训练也能到 48.5%——说明收益不完全依赖机器人域视频的相似性；50 条演示时 DROID 与 OpenVid-1M 差距几乎消失（96.4% vs 95.1%），说明领域内预训练在监督稀缺时最有价值，标注充足后通用视频先验也能被对齐利用。

**分布外鲁棒性**（改造版 LIBERO-Long：物体初始化范围扩大到 5cm、每任务替换一个背景物体为未见物体、目标物前放置未见干扰物）：

| 方法 | 平均 SR |
|---|---|
| Diffusion Policy | 0.32 |
| UWM | 0.34 |
| π0.5 fine-tuned | 0.53 |
| **JOPAT** | **0.66** |

JOPAT 比 Diffusion Policy/UWM/π0.5 fine-tuned 分别高 34/32/13 点；π0.5 fine-tuned 在部分任务（Book-Candy、Soup-Cheese、Moka-Moka）峰值最好，但在 Bowl-Drawer、Mug-Mug 上表现差，导致平均值反而低于 JOPAT，说明大型预训练 VLA 在受干扰/遮挡场景下可能不稳定，而显式对应关系 + 可见性预测提供了更一致的未来状态接口。

## 四、局限性

论文在结论与附录中明确列出三点局限：

1. **网格轨迹的空间稀疏性**：25×25 网格点跟踪可能遗漏亚厘米级精细形变所需的细节，这与 Insert-Peg 等毫米级接触任务上的失败一致。
2. **依赖现成点跟踪器（CoTrackerV3）设了一个性能天花板**：监督信号本身跟不到的动力学，模型也学不到——轨迹质量是整个方法的上限。
3. **推理速度与相机假设的限制**：统一 Transformer 的推理速度（实测约 10Hz）对于 >20Hz 的高频控制仍是瓶颈；当前架构针对静态相机优化，若相机随本体自运动（ego-motion），点跟踪与轨迹接口的有效性会被打折扣，限制了向移动操作场景的直接迁移。

此外附录的任务级失败分析补充：Cook-Soup 失败多发生在锅盖处（镜面反射削弱视觉 grounding）；Push-Tomato 失败源于小幅定位/接近角误差的累积；Pick-Grocery 主要反映策略在杂乱布局下的空间泛化不足（容易过拟合到演示中常见的抓取位置）。

## 五、评价与展望

**优点**：JOPAT 的核心贡献是把"世界动作模型该预测什么样的未来状态"这一接口设计问题，从隐式的像素/隐向量重建，转向显式的、以对应关系（correspondence）为中心的点跟踪 + 可见性表示，并且是**在同一个去噪序列里与视觉隐向量、动作联合采样**，而非像 Track2Act、ATM、ReKep 等此前工作那样把跟踪当作策略输入或独立的规划表示。可见性预测这一设计尤其值得肯定：它把"证据是否仍然可观测"显式建模为一个可学习变量，而不是强迫模型把每个点都解释成一个可见坐标，这对长时序、自遮挡（如机械臂遮挡目标物）场景的鲁棒性有直接帮助，实验中的消融也验证了这一点。40 个 LIBERO 任务上的 SOTA（97.8%）以及在样本效率上远超从零训练（10 条演示提升 5 倍以上）都是有说服力的证据，且论文用三组严格的消融（联合 vs 单模态、有无可见性、有无预训练）分离验证了三个假设分量各自的贡献，实验设计是本文的一大亮点。

**局限与开放问题**：（1）方法本质上是"用现成点跟踪器蒸馏 + 联合扩散"的组合，其上限被 CoTracker 的跟踪质量硬编码，尚未探讨与更强或可微调的跟踪器联合训练是否能进一步提升；（2）2D 网格点跟踪缺乏显式深度/3D 几何信息，这正是 Insert-Peg 等接触密集、毫米级精度任务上表现受限的根源，与近期一批显式使用 3D 点云/pointmap 作为动作或状态接口的工作（如同期的 PointAction 等）相比，JOPAT 停留在图像平面 2D 轨迹层面，是否要为精细接触任务引入 3D 或触觉信号，是一个自然的后续方向；（3）真机实验规模有限（4 个任务、每任务 50 条微调演示、10 次评测 rollout），且局限于单臂静态相机的 LeRobot SO-101 平台，尚未验证在双臂、移动底盘或第一人称相机等更复杂本体上的可迁移性；（4）Table 1 中的强基线（CogVLA、π0.5-KI、OpenVLA-OFT）本身已高度饱和（>95%），JOPAT 的净增益（相对 CogVLA +0.4 点）在仿真主表上并不算大，其更有说服力的证据其实来自 *Long* suite 与真实机器人、OOD 场景下更大的相对增益，这提示点跟踪表示的价值可能更多体现在长时序/强扰动场景，而非标准短程任务上的天花板突破。总体而言，本文为"世界动作模型该预测什么"提供了一个简洁、可复现、消融充分的答案，其"轨迹作为联合生成变量而非辅助监督"的设计思路对后续将几何/对应关系表示纳入生成式策略框架具有参考价值。

## 参考

- Doersch et al. *TAP-Vid: A Benchmark for Tracking Any Point in a Video*, 2023（点跟踪评测基准）
- Karaev et al. *CoTrackerV3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos*, 2024（本文轨迹标签来源）
- Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023（主评测基准）
- Zhu et al. *Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets*, 2025（UWM，最主要的像素级 WAM 对比基线）
- Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset*, 2024（动作无关视频预训练数据源）
