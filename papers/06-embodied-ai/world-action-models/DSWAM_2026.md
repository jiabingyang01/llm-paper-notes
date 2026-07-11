# DSWAM：面向细粒度机器人操作的双系统世界-动作基础模型

> **论文**：*DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation*
>
> **作者**：Jian Zhu, Jianjun Zhang, Taiyi Su, Tianbin Liu, et al.
>
> **机构**：AIRC, Midea Group（美的集团）；Tongji University（同济大学）
>
> **发布时间**：2026 年 07 月（arXiv 2607.04927）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.04927) | [PDF](https://arxiv.org/pdf/2607.04927)
>
> **分类标签**：`World Action Model` `双系统架构` `Flow Matching` `RoboTwin 2.0` `可变形物体操作` `TensorRT加速`

---

## 一句话总结

DSWAM 把"System 1 WAM 执行器（默认路径，靠视频协同训练学到 world-aware 表征，但推理时不生成未来视频、直接用 flow matching 出动作块）"与"System 2 视觉语言子任务规划器（仅在指令需要分解时才可选启用）"解耦为双系统架构，在与 VLA 基线 DeMaVLA 严格对齐机器人平台/数据/评测协议的真实叠衣任务上，把 WAM-only 模式的平均成功率从 92.5% 提升到 96.3%、完成时间从 2′18″ 降到 1′44″，并在 RoboTwin 2.0 的 50 项双臂操作任务上取得 92.38%（clean）/91.90%（randomized）的当前最佳平均成功率。

## 一、问题与动机

World Action Models（WAMs）近年被视为 VLA（Vision-Language-Action）策略之外的一条有前景路径：VLA 通常是"单帧观测 + 语言指令 → 动作"的直接映射，时序上下文有限；而 WAM 用视频建模学习"视觉状态如何在动作下演化"，为接触丰富、可变形物体操作（抓取、放置、折叠）提供更 dense 的物理监督，实证表现更强。

但作者指出两个尚未解决的问题：

1. **分解能力缺口**：家庭任务中的粗粒度指令（例如"把桌上的东西分类"）往往需要先被拆解为一串可执行的细粒度子任务；VLM-based VLA 天然具备语言级的语义理解和规划接口，而现有 WAM（多设计为纯执行策略）通常缺乏这种显式的语言级任务分解能力。
2. **公平对比缺口**：现有 WAM 与 VLA 之间的 real-robot 比较常常混杂了数据来源、机器人本体、任务协议等差异，无法判断性能差异究竟来自策略设计本身还是实验设置。

DSWAM 的应对思路：把"语义任务组织"和"world-aware 物理执行"解耦成两个系统——System 1 作为默认执行路径始终在线，System 2 仅在指令确实受益于分解时才被激活；同时在 DeMaVLA 提出的家庭叠衣协议下，用完全相同的机器人平台、预训练数据、后训练数据和评测标准，对 WAM 执行器和 VLA 策略做受控比较。

## 二、核心方法

### 整体接口

在时刻 $t$，机器人观测多视角 RGB 图像、语言指令/子任务与本体状态：

$$\mathbf{O}_t=[\mathbf{I}_t^1,\mathbf{I}_t^2,\mathbf{I}_t^3,\ell_t,\mathbf{q}_t]$$

策略输出一个动作块（action chunk）：

$$\mathbf{A}_t=[\mathbf{a}_t,\mathbf{a}_{t+1},\dots,\mathbf{a}_{t+H-1}]$$

用大白话说：模型每次不是只吐一个动作，而是一口气预测未来 $H$ 步的双臂连续控制指令，$\ell_t$ 既可以是用户原始指令，也可以是 System 2 给出的当前子任务。

### System 2：可选的视觉语言子任务规划器

System 2 基于 Rynnbrain4B 风格的多模态视觉语言模型实现。给定全局任务 prompt $p$，规划器观测按 1Hz 采样的最近 $T=5$ 帧短时视觉历史：

$$\mathbf{X}_t=\{x_{t-4},x_{t-3},x_{t-2},x_{t-1},x_t\},\quad x_i\in\mathbb{R}^{H\times W\times 3}$$

预测下一条子任务级指令：

$$s_t=\mathcal{P}_\phi(\mathbf{X}_t,p)$$

当 System 2 未激活时，直接令 $s_t=p$，绕过规划步骤。

**子任务边界监督**：训练数据由预训练的视觉语言标注器把机器人轨迹视频切分成带边界 $\tau_k$ 的子任务片段，帧以 1FPS 采样并组成滑窗 $\mathbf{W}_i=\{x_i,\dots,x_{i+4}\}$。监督目标是"过渡感知"（transition-aware）的：

$$y_i=\begin{cases}s_k, & \mathbf{W}_i\text{ 落在子任务 }k\text{ 内部}\\ s_{k+1}, & \mathbf{W}_i\text{ 是非终止子任务 }k\text{ 的最后一个窗口}\\ \text{done}, & \mathbf{W}_i\text{ 是终止子任务的最后一个窗口}\end{cases}$$

用大白话说：不仅要模型认出"现在在做什么子任务"，还要在快接近子任务边界时提前"吐出下一条指令"，从而实现平滑衔接。训练目标是自回归语言建模损失：

$$\mathcal{L}_{\text{plan}}(\phi)=-\mathbb{E}_{(\mathbf{X}_i,p,y_i)}\left[\log P_\phi(y_i\mid \mathbf{X}_i,p)\right]$$

**System 1/2 耦合**：部署时 System 1 每 $\Delta t=2\text{s}$ 把最近 5 帧送给规划器一次：

$$s_t=\mathcal{P}_\phi(\mathbf{X}_t^{\text{WAM}},p),\quad \Delta t=2\text{s}$$

返回的子任务在下次更新前持续作为 System 1 的语言条件。虽然 System 2 训练时用固定 1FPS 采样，作者发现它对真实机器人执行中变频率的视觉输入仍然鲁棒。

### System 1：WAM 执行器

执行器把当前多模态上下文编码为 latent world features：

$$\mathbf{z}_t=f_\theta(\mathbf{I}_t^{1:3},\ell_t,\mathbf{q}_t)$$

并从该表征直接预测动作块：

$$p_\theta(\mathbf{A}_t\mid \mathbf{O}_t)=p_\theta(\mathbf{A}_t\mid \mathbf{z}_t)$$

关键设计是 $\mathbf{z}_t$ 由单次前向产生——推理时不采样、不去噪、不解码任何未来帧。这延续了 Fast-WAM 提出的证据：world-modeling 监督可以和 test-time 的未来想象（future imagination）解耦。

动作分支用条件 flow matching 生成连续动作块（更适合双臂操作的平滑轨迹，而非离散化的语言 token）。记 $\mathbf{y}$ 可以是动作块 $\mathbf{A}_t$，也可以是未来的 latent 视觉 token $\mathbf{V}_{t+1:t+T}$，对流时间 $\tau\in(0,1)$ 和高斯噪声 $\epsilon$：

$$\mathbf{y}_\tau=(1-\tau)\mathbf{y}+\tau\epsilon$$

模型预测速度场 $\epsilon-\mathbf{y}$，损失为：

$$\mathcal{L}_{\text{FM}}(\mathbf{y})=\mathbb{E}_{\mathbf{y},\epsilon,\tau}\left[\lVert v_\theta(\mathbf{y}_\tau,\tau,\mathbf{O}_t)-(\epsilon-\mathbf{y})\rVert_2^2\right]$$

动作损失与视频协同训练损失共享同一套 flow-matching 目标：

$$\mathcal{L}_{\text{act}}=\mathcal{L}_{\text{FM}}(\mathbf{A}_t),\qquad \mathcal{L}_{\text{vid}}=\mathcal{L}_{\text{FM}}(\mathbf{V}_{t+1:t+T})$$

$$\mathcal{L}=\mathcal{L}_{\text{act}}+\lambda_{\text{vid}}\mathcal{L}_{\text{vid}}$$

用大白话说：训练时给模型"顺便"布置一个预测未来视觉 token 的任务，逼视觉 backbone 学会编码物理动态；但动作 token 被禁止用未来视觉 token 作为特权信息，保证训练输入分布和推理时（只有当前观测）保持一致。执行器 backbone 是预训练视频模型 Wan2.2-TI2V-5B 加一个动作专家（action expert）。

### 部署工程：RTC + 异步 + TensorRT

为使执行路径在真实机器人上可用，DSWAM 结合了 real-time chunking（RTC，借用 Black et al. 提出的推理时机制）与异步执行：policy worker 用最新观测持续预测新动作块，同时底层控制器继续执行当前块，避免 policy 查询阻塞控制回路。

计算上把 transformer 重的执行路径拆成两个 TensorRT engine：visual-context engine（为视频 latent、扩散时间步、语言/本体上下文、视觉注意力掩码构建逐层缓存）与 action-denoising engine（消费动作 latent、共享上下文、缓存视觉状态、联合 video-action 注意力掩码，预测一次去噪更新）。实测中 BF16 TensorRT 路径把端到端 policy 延迟从 PyTorch 的 198.2ms 降到 73.8ms（2.69× 加速），与 PyTorch 参考的动作一致性很高（最大相对误差 0.0106，余弦相似度 0.99977）。

## 三、实验结果

**RoboTwin 2.0（50 项双臂操作任务，跨全部任务的平均成功率）**

| 方法 | Clean | Randomized |
|---|---|---|
| $\pi_0$ | 65.92% | 58.40% |
| $\pi_{0.5}$ | 82.74% | 76.76% |
| DeMaVLA | 88.42% | 86.78% |
| Motus | 88.66% | 87.02% |
| Fast-WAM | 91.88% | 91.78% |
| **DSWAM** | **92.38%** | **91.90%** |

DSWAM 两种设置下均取得最佳平均成功率；相对最强 VLA 基线 DeMaVLA，clean/randomized 分别提升 3.96 / 5.12 个百分点；相对最强 WAM 基线 Fast-WAM，提升相对较小，仅 0.50 / 0.12 个百分点。

**真实叠衣（DeMaVLA 匹配协议，ALOHA 式双臂平台，shirt/skirt/pant/towel 四类衣物，每类 2 个物理实例 × 10 trials = 80 trials；此对比中 DSWAM 关闭 System 2，仅用 WAM 执行器直接对原始指令响应，聚焦纯执行能力比较）**

| 方法 | Shirt SR/耗时 | Skirt SR/耗时 | Pant SR/耗时 | Towel SR/耗时 | 平均 SR/耗时 |
|---|---|---|---|---|---|
| $\pi_0$ | 90.0% / 1′55″ | 95.0% / 1′03″ | 65.0% / 3′01″ | 55.0% / 3′44″ | 76.3% / 2′26″ |
| DeMaVLA | 95.0% / 2′15″ | 100.0% / 1′30″ | 75.0% / 3′01″ | 100.0% / 2′26″ | 92.5% / 2′18″ |
| **DSWAM** | 95.0% / 2′14″ | **100.0%** / 0′58″ | **90.0%** / 2′19″ | **100.0%** / 1′27″ | **96.3%** / 1′44″ |

相对 DeMaVLA，DSWAM 平均成功率提升 3.8 个百分点、平均耗时缩短 34 秒；提升最明显的是 pant 任务（75.0%→90.0%，耗时 3′01″→2′19″）。

**System 2 子任务监督消融**（真实桌面分拣任务，粗指令 "Sort objects on the table"，子任务级监督把其拆为"拿起玩具车放入箱子"和"拿起面包放到盘子里"两条可执行指令）

| 指令设置 | 训练步数 | SR | 每次 rollout 失误数 |
|---|---|---|---|
| 原始粗指令 | 6000 | 71.4% | 3.75 |
| 原始粗指令 | 18000 | 80.0% | 3.30 |
| 子任务级指令 | 6000 | **100.0%** | 1.00 |
| 子任务级指令 | 18000 | **100.0%** | **0.30** |

平均下来，原始粗指令达到 75.7% SR、3.53 次失误/rollout，子任务级指令达到 100.0% SR、0.65 次失误/rollout。

**TensorRT 推理效率（batch size 1，warmed end-to-end 延迟，NVIDIA RTX 5090）**

| 执行路径 | 延迟 (ms) | 加速比 | 最大相对误差 | 余弦相似度 |
|---|---|---|---|---|
| PyTorch | 198.2 | 1.00× | – | – |
| TensorRT BF16 | 73.8 | 2.69× | 0.0106 | 0.99977 |

**同步 vs 异步 TensorRT+RTC（easy 衣物实例，用于单独评估部署机制，不直接与叠衣主表可比）**

| 设置 | 任务 | SR | 平均成功耗时 |
|---|---|---|---|
| 同步 TensorRT | Shirt | 100% | 1′47″ |
| 同步 TensorRT | Pants | 70% | 1′50″ |
| 异步 TensorRT+RTC | Shirt | 100% | 1′28″ |
| 异步 TensorRT+RTC | Pants | 100% | 1′08″ |

异步 TensorRT+RTC 在保持 shirt 满成功率的同时把 pants 成功率从 70% 提升到 100%，两个任务的平均耗时都进一步缩短。

## 四、局限性

论文正文没有单独的 limitation 小节，以下局限是从方法设计和实验范围中归纳的：

1. **子任务标签来自自动标注器**：System 2 的训练监督（子任务边界与文本标签）由"预训练视觉语言标注器"对轨迹视频自动切分产生，标注质量本身未做验证或消融，可能引入系统性偏差。
2. **System 2 评测样本单薄**：分解能力仅在一个真实桌面分拣任务（2 个子任务）上验证，子任务数量、任务多样性都很有限，尚未展示在更长、更复杂多步骤任务链上的规划泛化能力，也未与叠衣这类主基准结合验证。
3. **核心 real-robot 对比的因果链不够纯粹**：Table 2 的受控比较本质上是"WAM 执行器（关闭 System 2）vs VLA 策略"整体系统对比，两者 backbone（视频扩散模型 Wan2.2-TI2V-5B vs DeMaVLA 的 VLM）完全不同，并非同一 backbone 下"是否做 video co-training"这种更干净的消融。
4. **相对最强 WAM 基线的边际收益很小**：RoboTwin 2.0 上相对 Fast-WAM 仅提升 0.50 / 0.12 个百分点，说明"去掉 test-time 未来生成"这条技术路线本身已相对成熟，DSWAM 执行器架构的增量创新有限。
5. **真实机器人试验规模较小**：每类衣物仅 20 trials，分拣消融每个设置也只在百余级别的训练步数区间内比较，统计显著性有限。
6. **未评估跨本体/跨平台泛化**：所有真实实验均在同一 ALOHA 式双臂平台上完成。
7. **加速评测覆盖不全**：Table 5 的同步/异步对比仅用了 easy 衣物实例的 shirt/pants 两类任务，未覆盖 skirt/towel 及 hard 实例。

## 五、评价与展望

**优点**：

- 双系统解耦的设计目标明确、消融直接对应 claim：把"是否做语言级任务分解"做成可选开关而非强制串联的规划器，是对"VLM-based VLA 规划强但物理执行偏弱"与"WAM 执行强但缺语言级分解接口"两条路线的一种务实折中。
- 受控对比方法论的价值：借用 DeMaVLA 协议，在同一机器人平台、预训练数据、后训练数据、任务定义与成功标准下比较 WAM 执行器与 VLA 策略，这种"matched setting"在当前 WAM/VLA 各说各话、数据与本体互不可比的领域中是相对稀缺的严谨实践，方法论上值得后续工作借鉴。
- 部署工程扎实：TensorRT 双 engine 拆分 + RTC + 异步执行不是空谈"支持实时部署"，而是给出了延迟（2.69× 加速）与动作一致性（余弦相似度 0.99977）的量化验证。

**局限与开放问题**：

- 与同期 WAM 工作相比（相关工作一节列举了 DreamZero、Cosmos Policy、Motus、LingBot-VA、X-WAM、GigaWorld-Policy、Fast-WAM 等），DSWAM 在"executor 是否需要 test-time 未来生成"这一问题上和 Fast-WAM 站在同一立场（训练时监督足够，推理不需要显式想象），但在 RoboTwin 2.0 上相对 Fast-WAM 的绝对提升很小，暗示该路线的执行器架构本身已趋近饱和，DSWAM 真正的增量贡献更多落在 System 2 的可选规划接口而非 executor 的架构创新。
- System 2 的自动伪标签依赖以及仅在小样本分拣任务上验证的事实，使"子任务级监督普遍提升执行稳定性"这一结论的外部有效性还有待更大规模、更多任务类型的检验。
- 一个自然的后续方向是把 System 2 的分解能力和 System 1 的执行能力放到更长时间跨度、更多步骤的真实家庭任务上做联合端到端评测（而不仅是 2 个子任务的分拣 demo），以验证双系统设计的收益是否随着任务复杂度上升而放大；另一个开放问题是能否在同一 backbone 下做更干净的"WAM 训练目标 vs 纯 VLA 训练目标"消融，把 executor 优势和 backbone 选择的贡献分离开。

## 参考

[1] Yuan, Dong, Liu, Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv:2603.16666, 2026.

[2] Su, Zhu, Wang, He, Huang, Zhang, Ding, Xu. DeMaVLA: A vision-language-action foundation model for generalizable deformable manipulation. arXiv:2605.31286, 2026.

[3] Black, Galliker, Levine. Real-time execution of action chunking flow policies (RTC). Advances in Neural Information Processing Systems, 2026.

[4] Chen, Chen, Chen, Cai, Liu, Liang, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv:2506.18088, 2025.

[5] Ye, Wang, Ni, Huang, Zhao, Li, et al. GigaWorld-Policy: An efficient action-centered world-action model. arXiv:2603.17240, 2026.
