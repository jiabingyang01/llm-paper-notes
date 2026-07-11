# MECo-WAM：为世界-动作模型学习 4D 几何先验以实现高效推理

> **论文**：*Learning 4D Geometric Priors for Inference-Efficient World Action Models*
>
> **作者**：Jianjun Zhang, Jian Zhu (Project Leader), Taiyi Su, Chong Ma, Zitai Huang, Yi Xu (Corresponding), Hanli Wang (Corresponding) et al.
>
> **机构**：Tongji University；AIRC, Midea Group
>
> **发布时间**：2026 年 07 月（arXiv 2607.05468）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.05468) | [PDF](https://arxiv.org/pdf/2607.05468)
>
> **分类标签**：`World Action Model` `4D几何先验` `多专家协同训练` `VGGT蒸馏` `推理效率` `RoboTwin/LIBERO`

---

## 一句话总结

MECo-WAM 在训练时额外接入一个由冻结 VGGT 编码器监督的轻量 4D 几何专家，通过"decayed 4D read-mask attention"把仅限当前帧的几何信息临时注入 video-action 通路、再逐步撤除访问权限，并配合 action-aware 的时空几何关系蒸馏损失把动作相关的 4D 先验"烧进"共享表示；部署时整个 4D 专家和 VGGT 编码器被完全移除，推理图与基座 Fast-WAM 完全一致，在 LIBERO 上达到 98.2% 平均成功率、RoboTwin 2.0 上达到 92.6%（较不做几何注入的 Fast-WAM 提升 0.79 点），而单动作块推理延迟几乎不变（198.73ms vs. 198.65ms）。

## 一、问题与动机

World Action Model（WAM）把"预测未来视觉状态"和"生成动作序列"联合训练，相比纯 VLA 策略能提供更丰富的运动/交互先验（如 DreamZero、Motus、LingBot-VA、Cosmos Policy 等）。但作者指出一个结构性缺陷：现有 video-action 协同训练方法优化的主要是**外观导向（appearance-oriented）**的视频 latent——它能生成看起来合理的未来画面，却不必然保留判断抓取是否可达、物体是否与目标对齐、接触是否会导致稳定转移所需要的**空间关系**；而这些关系随机械臂接近、接触、移动、释放物体的过程是**随时间演化**的（即"4D"而非静态 3D）。

已有的几何增强 WAM/VLA（如 SpatialVLA、BridgeVLA、GeoVLA，以及同期 WAM 方向的 X-WAM、WAM4D）大多选择把显式 4D 重建/深度预测做成部署时的输出分支，这带来两个问题：
1. 增加推理成本（多一个几何解码/预测分支）；
2. 泛化的几何监督信号是"通用"的，不区分哪些空间关系与当前机器人动作真正因果相关，容易把优化目标拉向与动作生成弱耦合的几何重建任务本身。

于是论文提出一个聚焦问题：**能否让 WAM 在训练阶段获得动作相关的时序几何能力，同时在部署阶段仍保持与基座模型完全相同、轻量的 video-action 推理图？** 答案即 MECo-WAM（Multi-Expert Co-Training World Action Model）。

## 二、核心方法

### 2.1 问题形式化与专家 token 构造

部署策略仍是标准的观测-动作映射：

$$
p_\theta(a_{1:H} \mid o_0, a_0, \ell). \tag{1}
$$

用大白话说：机械臂看到当前观测 $o_0$、当前本体状态 $a_0$、语言指令 $\ell$，输出未来 $H$ 步的动作序列——这条接口在训练和部署阶段完全不变。

训练时引入三路"专家"，共享同一套 flow-matching 加噪-去噪范式：

- **视频专家**：以 VAE 编码的当前帧 $f_0$ 为干净视觉上下文，对未来帧 latent 做加噪 $[f_1,f_2,\dots,f_h]=(1-r)\bar Y_v + r\epsilon_v$，预测 $[f_{p1},f_{p2},\dots,f_{ph}]=E_v(X_v,r;\ell)$；
- **动作专家**：以本体状态 $a_0$ 为干净锚点，同样构造噪声动作片段并预测未来动作块；
- **4D 几何专家**：token 来自**冻结**的 VGGT 编码器（而非 VAE），当前帧几何 $g_0$ 保持干净、未来几何 $[g_1,\dots,g_h]$ 加噪，预测 $[g_{p1},\dots,g_{ph}]=E_{4d}(X_{4d},r)$，仅在选定关键帧 $\mathcal K\subseteq\{1,2,h\}$ 上施加 4D 损失。

三路专家的 token 拼接为 $X=[X_v,X_a,X_{4d}]$，在每层 Transformer 内各自有独立的 QKV 投影，通过一个专家级注意力掩码 $M$ 控制信息流：

$$
Y = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d}+M\right)V. \tag{8}
$$

关键设计是**非对称的专家可见性**：当前锚点 $f_0,a_0,g_0$ 只能自注意；未来视频 token 只读视频分支（不读动作）；未来动作 token 只读干净视觉上下文和动作分支（不读带噪未来视频）——这样可以阻止"未来信息"通过非因果捷径泄漏到动作生成，逼迫模型走真实的时序推理路径而不是"抄近道"。

### 2.2 Decayed 4D Read-Mask Attention

这是把 4D 几何先验"迁移"进部署通路、同时不留下永久依赖的核心机制。论文在基础掩码之上，额外加入一条从未来视频/动作 query 到**当前帧几何 token $g_0$** 的临时读边——之所以安全，是因为 $g_0$ 只编码自当前 RGB 帧，不涉及任何未来信息泄漏。这条读边是随机且随训练步衰减的：

$$
\gamma_s \sim \mathrm{Bernoulli}\big(p_{4d}(s)\big), \qquad
p_{4d}(s)=\begin{cases} p_{\text{start}}+(p_{\text{end}}-p_{\text{start}})\dfrac{s}{S_{\text{decay}}}, & s<S_{\text{decay}} \\ p_{\text{end}}, & s\ge S_{\text{decay}} \end{cases} \tag{9,10}
$$

用大白话说：训练早期以较高概率把当前帧的几何信息"喂"给视频/动作分支，让它们习惯用几何信息来去噪；随着训练推进，这条通道被逐步"断奶"，读取概率线性衰减到 $p_{\text{end}}=0$。到部署时读概率恒为 0，4D token 根本不会被实例化，模型自然退化回原始的纯 video-action 推理图——这就是论文标题"inference-efficient"的关键机制：几何先验只在训练阶段的梯度里留下印记，不在推理阶段的计算图里留下分支。

### 2.3 Action-Aware Temporal Geometric Distillation

蒸馏目标以冻结 VGGT 为几何教师，但只监督训练时 4D 专家自己的预测，包含三部分：

**(1) 关系匹配（而非绝对坐标匹配）**。对齐后的学生几何特征 $Z^k=P_{\text{align}}(g_{pk})$ 与教师目标 $G^k_T=\bar g_k$ 都不直接比较绝对值，而是比较 token 间的成对欧氏距离关系：

$$
R^k_{4d}(i,j)=\|Z^k_i-Z^k_j\|_2, \qquad R^k_T(i,j)=\|G^k_{T,i}-G^k_{T,j}\|_2. \tag{12}
$$

用大白话说：不要求学生的几何特征和 VGGT 教师"长得一模一样"（不同编码器的绝对坐标系本就对不齐），只要求物体、夹爪、目标之间的**相对空间布局**一致——这是一种更宽松、更聚焦结构的监督信号。

**(2) Action-aware 权重，只强调动作相关区域**。用视频 token $v^k_i$ 与该时间片段对应的动作表征 $\bar a^k$ 做缩放点积相似度并 softmax：

$$
s^k_i=\frac{(W_vv^k_i)^\top(W_a\bar a^k)}{\tau\|W_vv^k_i\|_2\|W_a\bar a^k\|_2}, \qquad r^k_i=\mathrm{softmax}_i(s^k_i), \tag{13}
$$

再与均匀先验混合防止权重坍缩到单一区域：

$$
w^k_i=N\Big((1-\eta)r^k_i+\frac{\eta}{N}\Big), \qquad w^k_{ij}=\sqrt{w^k_iw^k_j}. \tag{14}
$$

据此得到帧内几何损失：

$$
\mathcal L^{\text{act}}_{\text{geo}}=\frac{\sum_{k\in\mathcal K,i,j}w^k_{ij}\left|\widehat R^k_{4d}(i,j)-\widehat R^k_T(i,j)\right|}{\sum_{k\in\mathcal K,i,j}w^k_{ij}}. \tag{15}
$$

**(3) 时序关系匹配**。物体几何关系如何随时间演化（接近、接触、搬运、释放）同样重要。对相邻关键帧对 $(k,k^+)$ 定义归一化关系变化量：

$$
\Delta(R^k,R^{k^+})=\frac{R^{k^+}-R^k}{|R^{k^+}|+|R^k|+\epsilon}, \tag{16}
$$

再用相邻帧对权重 $w^{k,k^+}_{ij}=\sqrt{w^k_{ij}w^{k^+}_{ij}}$ 加权对齐学生与教师的关系变化量差异，得到 $\mathcal L^{\text{act}}_{\text{tem}}$（式 19）。

### 2.4 训练目标

视频/动作专家沿用基座 WAM 的条件 flow-matching 目标：

$$
\mathcal L_{\text{FM}}(y)=\mathbb E_{y,\epsilon,r}\Big[\|u_\theta(y_r,r,o_0,a_0,\ell)-(\epsilon-y)\|_2^2\Big], \tag{21}
$$

4D 目标为几何损失与时序损失之和 $\mathcal L_{4d}=\alpha_{\text{geo}}\mathcal L^{\text{act}}_{\text{geo}}+\alpha_{\text{tem}}\mathcal L^{\text{act}}_{\text{tem}}$，总损失为三者加权和：

$$
\mathcal L_{\text{total}}=\lambda_{\text{video}}\mathcal L_{\text{video}}+\lambda_{\text{action}}\mathcal L_{\text{action}}+\lambda_{4d}\mathcal L_{4d}. \tag{23}
$$

### 2.5 实现细节

视频骨干为 Wan2.2-TI2V-5B；动作专家 30 层 DiT block、24 个注意力头、每头 128 维、隐藏宽度 1024（约 1B 参数）；4D 专家隐藏宽度 512（约 0.45B 参数），几何监督来自冻结的 VGGT-1B 编码器。每个训练片段含 33 个机器人步（动作视野 $H=32$）、9 个视频帧（动作:视频时序比 4:1）。连续 flow matching、1000 训练时间步、shift 5.0，AdamW（lr $1\times10^{-4}$，weight decay 0.01，cosine decay，bf16，梯度裁剪 1.0），64 张 NVIDIA H20 96GB 训练；推理用 10 步去噪、CFG=1.0，单张 RTX 5090 32GB；真机实验使用 ARX-R5 机械臂。

## 三、实验结果

**LIBERO（四套件，各 10 任务/500 条专家演示，success rate %）**：

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| LingBot-VA (P.T.) | 98.5 | 99.6 | 97.2 | **98.5** | 98.5 |
| Fast-WAM (w/o P.T.) | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **MECo-WAM (w/o P.T.)** | **98.8** | **100.0** | **98.2** | 95.8 | **98.2** |

在没有具身策略预训练（P.T.）的前提下，MECo-WAM 比 Fast-WAM 高 0.6 点、比 Motus 高 0.5 点，接近有预训练的 LingBot-VA（98.5%）；提升主要集中在几何敏感的 Spatial（+0.6）和 Goal（+1.2）套件。

**RoboTwin 2.0（双臂操作，clean/randomized，success rate %）**：

| Method | P.T. | Clean | Rand. | Average |
|---|---|---|---|---|
| π0.5 | ✓ | 82.74 | 76.76 | 79.75 |
| Motus | ✓ | 88.66 | 87.02 | 87.84 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| Fast-WAM | ✗ | 91.88 | 91.78 | 91.83 |
| **MECo-WAM** | ✗ | **93.26** | **91.98** | **92.62** |

clean 场景提升更明显（+1.38 点），随机化场景也为正（+0.20 点）；MECo-WAM 平均成功率甚至略超有预训练的最强 WAM（LingBot-VA 92.20%）。

**推理延迟-成功率权衡（RoboTwin，Figure 1，动作块推理延迟 ms / 成功率 %）**：

| Method | Latency (ms) | Success (%) |
|---|---|---|
| π0 | 136.27 | 62.2 |
| Fast-WAM | 198.65 | 91.8 |
| **MECo-WAM** | **198.73** | **92.6** |
| GigaWorld-Policy | 218.12 | 86.0 |
| FastWAM-IDM | 844.11 | 91.3 |
| Motus | 1956.83 | 87.8 |

MECo-WAM 与 Fast-WAM 的延迟几乎相同（+0.08ms），验证了"4D 先验只在训练时起作用"的设计确未增加推理开销。

**真机实验（ARX-R5，两项桌面任务，SR/PR/CR/CT 均为多次试验平均）**：

| Task | Method | SR (%) | PR (%) | CR (次) | CT (s) |
|---|---|---|---|---|---|
| Stack Cubes | Fast-WAM | 60.0 | 75.0 | 1.67 | 27.06 |
| Stack Cubes | MECo-WAM | 60.0 | 75.0 | 0.83 | 25.71 |
| Sort Cubes by Size | Fast-WAM | 60.0 | 75.0 | 1.33 | 38.49 |
| Sort Cubes by Size | MECo-WAM | **70.0** | **80.0** | 1.00 | 31.96 |

两任务平均 SR 提升 5.0 点、PR 提升 2.5 点，纠正次数减少约 39%，完成时间缩短约 12%——说明几何先验带来的收益在真机上体现为更稳的接触/对齐、更少的重试，而不仅是仿真指标的数字游戏。

**消融（RoboTwin，视频质量/动作 MSE/任务成功率）**：

| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | Action MSE(×10)↓ | Avg SR↑ |
|---|---|---|---|---|---|
| Fast-WAM | 29.55 | 0.936 | 0.038 | 0.032 | 91.83 |
| + 4D expert（无衰减读取） | 29.81 | 0.935 | 0.039 | 0.034 | 91.87 |
| + decayed read | 30.06 | 0.939 | 0.038 | 0.026 | 92.14 |
| + $\mathcal L^{\text{act}}_{\text{geo}}$ | 29.97 | 0.938 | 0.037 | 0.022 | 92.22 |
| + $\mathcal L^{\text{act}}_{\text{tem}}$ | 30.42 | 0.940 | 0.039 | 0.019 | 92.25 |
| Full w/o action-aware weight | 30.31 | 0.942 | 0.038 | 0.017 | 92.38 |
| **MECo-WAM（完整）** | **30.72** | **0.942** | **0.037** | **0.013** | **92.62** |

两个关键发现：①仅加孤立的 4D 专家（不带 decayed read）几乎没有收益（91.83→91.87，动作 MSE 反而从 0.032 升到 0.034）——说明几何监督若被"锁死"在辅助分支内、不与视频-动作通路交互，几乎白学；②去掉 action-aware 权重（均匀关系匹配）比完整模型低 0.24 点（92.38 vs 92.62），说明收益不仅来自"多了一路 4D 监督"，还来自"聚焦在动作相关区域"这一设计选择本身。

此外论文还做了表示探针（representation probing）：冻结 video-action 骨干后训练一个匹配的 DPT 式深度头，MECo-WAM 的 tokens 能重建出比 Fast-WAM 更清晰的深度结构（Figure 4），佐证训练时的 4D 目标确实把几何先验迁移进了部署时的共享表示，而非仅停留在被丢弃的辅助分支里。

## 四、局限性

论文正文没有单列"Limitations"小节，以下依据方法与实验描述归纳：

- **依赖外部冻结几何编码器 VGGT 的质量**：4D 监督信号的上限由 VGGT-1B 本身的几何估计精度决定，论文未评估 VGGT 在操作场景（近距离、遮挡、反光物体）下的误差如何传导到最终策略。
- **消融显示"仅加 4D 专家不带衰减读取"几乎无收益甚至轻微损害动作 MSE**，说明该方法对 decayed read-mask 的超参（起止概率、衰减步数 $S_{\text{decay}}$）以及关键帧选取集合 $\mathcal K$ 较敏感，论文未给出这些超参的敏感性分析或消融。
- **训练成本未纳入"inference-efficient"的宣传口径**：虽然推理时延迟几乎不变，但训练阶段仍需额外跑一个 VGGT 前向 + 4D 专家的三路协同训练，64×H20 的训练算力开销高于纯 Fast-WAM 基座，论文未报告训练时间/显存的具体增量。
- **评测规模有限**：真机实验仅两个桌面堆叠/排序任务、单一 ARX-R5 平台，任务的几何复杂度（主要考察高度对齐和尺寸排序）相对单一，未覆盖更复杂的接触丰富型操作（如插拔、柔性物体）。
- **RoboTwin/LIBERO 上的绝对提升幅度总体是"点位级"的**（RoboTwin +0.79 点，LIBERO +0.6 点），在已经接近饱和（Object 套件 100%）的指标上进一步提升的空间本身有限，论文没有讨论方法在更难、未饱和的基准上的可扩展性。

## 五、评价与展望

**优点**：MECo-WAM 抓住了一个务实且工程上可复现的问题——WAM 领域近期涌现出不少把 4D/深度信息塞进部署时输出的做法（如同期的 X-WAM 用多视角 RGB-D 未来预测、WAM4D 用 spatial register token），这些方法普遍要在推理时保留几何分支或额外解码步骤。MECo-WAM 与 Fast-WAM 一脉相承的"训练时监督、推理时删除"思路（decayed read-mask attention）提供了一个几乎零推理代价的几何先验注入方案，其消融实验（尤其是"孤立 4D 专家无效、必须搭配衰减读取通道才有效"这一发现）具有一定的方法论价值，说明几何知识向策略表示的迁移需要显式的"读取通道"而非仅靠共享参数的隐式传播。Action-aware 权重与相对关系匹配（而非绝对坐标匹配）的组合设计也回应了跨编码器（VAE 表示 vs VGGT 表示）对齐的现实困难，是一个合理的工程选择。

**与相关工作的关系**：论文将自己明确定位在"Fast-WAM 效率范式 + 几何感知 WAM"的交叉点——继承 Fast-WAM"训练用未来想象、推理不做未来想象"的判断，同时区别于 X-WAM/WAM4D 把几何做成部署时输出的路线。这个定位比较清晰，但也意味着其比较基线本质上是"Fast-WAM + 4D 蒸馏"，而并未与 X-WAM、WAM4D 在同一基准下做直接的数字对比（论文相关工作节仅作文字描述，主表中未出现这两个方法的 LIBERO/RoboTwin 成功率），使得"MECo-WAM 的训练时监督范式是否优于显式 4D 输出范式"这一更有价值的问题仍未被实验直接回答。

**开放问题与可能方向**：(1) VGGT 教师本身是通用几何模型，并非为操作场景微调，用更贴近操作场景（近距离、透明/反光物体、遮挡手部）的几何先验模型替换是否能进一步放大收益，值得探究；(2) decayed read-mask 的衰减调度目前是人工设定的线性调度，是否可以做成与任务难度/几何复杂度自适应的调度；(3) 该框架把"4D 关系蒸馏"限定在训练阶段，一个自然的后续问题是能否把同样的"训练时注入、推理时移除"范式推广到其他模态先验（如触觉、力矩)，以及是否能与显式几何输出范式（X-WAM/WAM4D）做混合，在需要精细接触推理的任务上按需切换"是否携带几何分支推理"。

## 参考

- Yuan, T.; Dong, Z.; Liu, Y.; Zhao, H. Fast-WAM: Do world action models need test-time future imagination? arXiv:2603.16666, 2026.（MECo-WAM 的直接基座与部署接口来源）
- Guo, J. et al. Unified 4D world action modeling from video priors with asynchronous denoising (X-WAM). arXiv:2604.26694, 2026.（同期把 4D 做成部署时显式输出的对比范式）
- Li, Y. et al. WAM4D: Fast 4D World Action Model via Spatial Register Tokens. arXiv:2606.14048, 2026.（同期几何-WAM 方向的另一路线）
- Wang, J.; Chen, M.; Karaev, N.; Vedaldi, A.; Rupprecht, C.; Novotny, D. VGGT: Visual Geometry Grounded Transformer. CVPR 2025.（本文使用的冻结几何教师编码器）
- Bi, H. et al. Motus: A unified latent action world model. CVPR 2026.（对比的预训练 WAM 基线之一）
