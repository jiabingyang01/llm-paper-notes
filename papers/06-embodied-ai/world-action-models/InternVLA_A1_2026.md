# InternVLA-A1：统一理解、生成与动作的机器人操作框架

> **论文**：*InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation*
>
> **作者**：Junhao Cai, Yang Li, Haoxiang Ma, Jiangmiao Pang（通讯作者）, Zherui Qiu, Yang Tian, Jia Zeng（项目负责人）, Hongrui Zhu, et al.（InternVLA-A1 Team）
>
> **机构**：Shanghai AI Laboratory（上海人工智能实验室）；Humanoid Robot (Shanghai) Co., Ltd.
>
> **发布时间**：2026 年 01 月（arXiv 2601.02456，v2 于 2026 年 2 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.02456) | [PDF](https://arxiv.org/pdf/2601.02456)
>
> **分类标签**：`VLA` `Mixture-of-Transformers` `视觉前瞻生成` `Flow Matching` `统一理解生成动作`

---

## 一句话总结

InternVLA-A1 用 Mixture-of-Transformers 架构把"理解专家（MLLM）+ 生成专家（COSMOS 隐空间视觉前瞻）+ 动作专家（Flow Matching）"通过统一掩码自注意力串联成单一网络,在真实/仿真/人类视频三源数据（692M 帧）上联合预训练,在 12 个真实机器人任务上较 π0.5 静态任务 +4.4%、动态任务 +26.7%,在 RoboTwin 2.0 仿真基准上 +2.6%。

## 一、问题与动机

论文指出当前生成式策略的两条技术路线各有硬伤：

1. **纯 MLLM 路线的 VLA**（如 π0、π0.5、GR00T N1/N1.5）语义理解强,但把视觉输入映射为离散文本 token 空间不适合建模物理规律,导致模型偏向"感知到动作"的反应式映射,而非对状态演化（动量、惯性、接触）的推理——这一缺陷在传送带分拣等**动态场景**中尤为明显。
2. **视频预测式世界模型路线**（如 VPP、Genie Envisioner）通过预测未来观测来引导决策,但缺乏语义 grounding,并且对视频预测误差很敏感,容易把错误的想象结果传导给动作头。同时,主流视频生成架构（扩散或下一 token 预测）计算量过大,难以满足操作策略所需的高频实时推理——即便是工程优化后的 SANA-Sprint 在 RTX 4090 上也只能达到 0.16 秒/帧（≤6Hz）,DreamZero 在 38 倍加速后于 GB200 上也仅 7Hz。

数据侧,真实机器人数据采集成本高（π0 用了 10,000 小时仍难以覆盖长尾场景变化）,而仿真数据虽可规模化但存在 sim-to-real gap。论文的解法是**架构 + 数据双管齐下**：用统一架构耦合语义理解与动力学预测,同时联合训练仿真轨迹、真实机器人演示、人类第一视角视频三种异构数据源。

## 二、核心方法

### 2.1 三专家 Mixture-of-Transformers 架构

InternVLA-A1 由三个 decoder-only Transformer 专家串联而成（信息单向流动：理解 → 生成 → 动作）：

- **理解专家（Understanding Expert）**：直接复用 InternVL3 或 Qwen3-VL 的架构,将多视角观测 $o_t$ 与语言指令 $l$ 编码为上下文嵌入 $h_{\text{und}} = f_{\text{und}}(l, o_t)$,作为下游专家可通过掩码自注意力访问的"共享语义记忆"。
- **生成专家（Generation Expert）**：预测未来视觉隐特征,为动作专家提供"物理前瞻"。
- **动作专家（Action Expert）**：结合 $h_{\text{und}}$、$h_{\text{gen}}$ 与本体感知状态 $q_t$,用 Flow Matching 输出动作 chunk。

**大白话**：理解专家先看懂"现在是什么场景、要做什么",生成专家再脑补"接下来一瞬间世界会变成什么样",动作专家结合这两路信息决定"手该怎么动"——三者共享一套 attention,而不是三个独立模型拼接。

### 2.2 生成专家的轻量化设计

为解决视频生成模型推理慢、无法支撑高频闭环控制的问题,生成专家做了三处关键工程取舍（借鉴 Janus Pro 的解耦视觉编码思路：理解用 ViT 抓语义,生成用 VAE 保空间细节）：

1. **输入 token 化**：取 3 个视角（头部相机 + 双腕相机）、2 个时间戳（当前 $t$ 与历史 $t-15$）共 6 张 $256\times256$ 图像,用 COSMOS CI8×8 连续 VAE 编码器压缩为 $32\times32$ 隐网格（每图 1024 token）。
2. **Token 压缩**：用 $8\times8$ 卷积核对隐网格下采样至 $4\times4$（每图 16 token）,6 张图共 96 token 送入 Transformer——避免 6144 长序列拖慢训练与推理。
3. **并行解码**：Transformer 处理后沿时间轴做平均池化聚合为 48 token（3 视角 × 16),再经反卷积上采样回 $32\times32$,由 COSMOS VAE 解码器一次性并行重建 $t+15$ 时刻的未来帧——**非自回归**,单次前向即可完成,牺牲部分高频视觉细节换取实时性。

生成目标（对隐空间特征做回归,而非像素级重建）：

$$
\mathcal{L}_{\text{gen}} = \mathbb{E}_{\xi_1}\left[\left\| f_{\text{gen}}(z_{t-m}, z_t; h_{\text{und}}) - \text{sg}[z_{t+m}] \right\|^2 \right]
$$

其中 $z_t = \phi_{\text{cosmos}}(o_t)$ 是 COSMOS 隐特征、$m=15$、$\text{sg}[\cdot]$ 表示 stop-gradient（未来帧的 VAE 编码只作监督目标,不回传梯度）。**大白话**：不要求生成专家"画出"未来画面,只要求它在隐空间里预测对,这样既省算力又不受像素噪声干扰。

### 2.3 动作专家：Flow Matching

给定 $\xi_2 = (a_{t:t+k}, o_{t-m}, o_t, q_t, l)$,采样时间步 $\tau \sim \text{Beta}(1.5, 1.0)$,构造插值动作 $a_{t:t+k}^{\tau} = (1-\tau)\epsilon + \tau a_{t:t+k}$（$\epsilon \sim \mathcal{N}(0, I)$）,学习一个速度场 $v_\theta$ 把噪声"搬运"到目标动作：

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{\xi_2}\left[\left\| v_\theta(q_t, a_{t:t+k}^{\tau}; h_{\text{und}}, h_{\text{gen}}) - (a_{t:t+k} - \epsilon) \right\|^2 \right]
$$

推理时从高斯噪声出发,沿学到的速度场用 Euler 法迭代 $K$ 步（步长 $\Delta\tau = 1/K$）积分出动作：

$$
a_{t:t+k}^{\tau+\Delta\tau} = a_{t:t+k}^{\tau} + \Delta\tau \cdot v_\theta(q_t, a_{t:t+k}^{\tau}; h_{\text{und}}, h_{\text{gen}})
$$

总损失是两个目标的加权和：

$$
\mathcal{L}_{\text{total}} = \lambda \cdot \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{act}}, \quad \lambda = 0.01
$$

### 2.4 注意力机制与训练细节

三专家的 token 流用**分块因果掩码**串联：后面的 block 可以看到前面所有 block 的 token,前面的 block 不能看到后面的（信息单向：理解 → 生成 → 动作）；每个专家内部的 token 之间双向全注意力。动作专家内部进一步拆出 state token（只关注自身与更早 block）与 action token（关注 state token、更早 block 及彼此）。

**模型规模**（表 1）：2B 版以 InternVL3-1B 为理解专家（0.94B）,生成/动作专家取自 InternVL3 底层的 Qwen2.5（各 0.36B）,总 1.8B；3B 版以 Qwen3-VL-2B 为理解专家（2.13B）,生成/动作专家取自 Qwen3（各 0.44B）,总 3.2B。两者在 RTX 4090 上用 torch.compile 均能跑到约 **13Hz**（2B 版因 InternVL3 需要 $448\times448$ 高分辨率输入,抵消了参数量优势,速度与 3B 版持平）。

**两阶段训练**：预训练 700K 步（batch 512,恒定学习率 $5\times10^{-5}$）+ 后训练（针对具体任务微调）60K 步（batch 128,学习率 $5\times10^{-5}\to5\times10^{-6}$,2000 步 warmup）。

**Load-balanced Parallel Training（LPT）**：为避免朴素地把全部异构数据集实例化到每个 worker 上导致的显存溢出与 I/O 争用,LPT 用贪心负载均衡把数据集分派给 worker：

$$
\pi(i) = \operatorname*{argmin}_{k \in \{1,\dots,K\}} \sum_{j:\pi(j)=k} s_j
$$

即每次把（按体量降序排列的）下一个数据集分给当前负载最小的 worker,数据集数量小于 worker 数时允许受控复制,从而在预训练规模（692M 帧、多个体量悬殊的数据集）下获得近似均匀的每 worker 吞吐。

### 2.5 数据配方

预训练混合 692M 帧,含三类来源（表 3,括号内为采样权重）：InternData-A1（仿真,396M,0.64）、RoboTwin（仿真,17M,0.08）、AgiBot-World Beta（真实,206M,0.18）、RoboMind（真实,5M,0.02）、EgoDex（人类第一视角视频,68M,0.08）。其中 InternData-A1 是团队前作,含 63 万条轨迹、7433 小时、4 种本体、18 类技能、70 个任务、227 个场景,由自研合成流水线 Nimbus 在 8 张 RTX 4090 上每天生成 209.7 小时仿真数据；EgoDex 提供 829 小时第一视角灵巧操作视频（200+ 任务）,预训练阶段**不使用**人类动作标签,仅用于让生成专家学习更丰富的人手-物体交互动态。

## 三、实验结果

评测覆盖 3 种真实机器人本体（Agibot Genie-1、ARX Lift-2、ARX AC One）、12 个真实任务（10 静态 + 2 动态任务族）以及 RoboTwin 2.0 仿真基准,每任务 30 次 rollout。

**静态任务**（表 4,成功率 %,对比 GR00T N1.5 / π0 / π0.5）：

| 方法 | Zip Bag | Unscrew Cap | Sort Parts | Make Sandwich | Sweep Trash | Operate Oven | Sort Rubbish | Wipe Stain | Place Markpen | Place Flower | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GR00T N1.5 | 33.3 | 0.0 | 6.7 | 46.7 | 16.7 | 46.7 | 66.7 | 40.0 | 40.0 | 33.3 | 33.0 |
| π0 (3.3B) | 40.0 | 66.7 | 53.3 | 66.7 | 43.3 | 73.3 | 96.0 | 73.3 | 53.3 | 40.0 | 60.6 |
| π0.5 | 60.0 | 66.7 | 53.3 | 73.3 | 50.0 | 80.0 | 97.3 | 93.3 | 66.7 | 66.7 | 70.7 |
| InternVLA-A1 (2B) | 66.7 | 33.3 | 46.7 | 73.3 | 63.3 | 53.3 | 97.3 | 80.0 | 66.7 | 66.7 | 64.7 |
| InternVLA-A1 (3B) | 73.3 | 66.7 | 53.3 | 93.3 | 66.7 | 86.7 | 97.3 | 86.7 | 66.7 | 60.0 | **75.1** |

2B 版（1.8B 参数)已超过参数量更大的 π0（3.3B),3B 版较 π0.5 提升 +4.4pt、较 π0 提升 +14.5pt,在 Make Sandwich（长时序双臂协作)上以 93.3% 对 π0.5 的 73.3% 拉开明显差距。

**动态任务**（图 6,目标随传送带持续移动）：

| 任务 | GR00T N1.5 | π0 | π0.5 | InternVLA-A1 (2B) | InternVLA-A1 (3B) |
|---|---|---|---|---|---|
| Express Sorting | 40.0 | 36.7 | 53.3 | 70.0 | **80.0** |
| In-motion Ingredient Picking | 20.0 | 20.0 | 66.7 | 46.7 | **93.3** |
| 两任务平均 | — | — | — | — | **86.7** |

3B 版在 Ingredient Picking 上较 π0.5 提升 +26.6pt,在动态场景下优势远大于静态场景,印证了视觉前瞻对高动态、时间敏感任务的价值；而 π0/GR00T N1.5 在该任务上仅 20.0%,几乎无法应对运动中的抓取时机判断。

**RoboTwin 2.0 仿真基准**（50 个双臂任务,Easy=干净 / Hard=域随机化,27,500 条演示微调,100 次评测）：

| 设置 | π0 | π0.5 | InternVLA-A1 (3B) |
|---|---|---|---|
| Easy | 80.0 | 86.8 | **89.4** |
| Hard | 79.5 | 87.0 | **89.6** |

**消融实验**：

- **预训练的作用**：去掉预训练直接从零训练,12 任务平均成功率从 77.0% 骤降至 25.4%（Operate Oven 直接归零),证明大规模预训练是关键归纳偏置,而非仅靠架构本身。
- **预训练数据源组合**（表 5）：仅用仿真数据 RoboTwin Easy/Hard 已达 88.3/88.5,但真实任务泛化弱（Place Flower 53.3%、Sort Parts 33.3%）；加入人类视频小幅提升仿真与 Sort Parts（→40.0%）；仿真+真实+人类三源联合训练效果最好（RoboTwin 89.4/89.6,Place Flower 60.0%,Sort Parts 53.3%）,验证了异构数据联合训练策略的有效性。
- **生成专家的作用**：去掉生成专家（仅理解 + 动作两专家）,12 任务中 11 个下降,平均成功率从 77.0% 降至 57.6%,降幅最大的正是两个动态任务（Express Sorting 80.0→46.7,Ingredient Picking 93.3→40.0),仅 Sort Rubbish 一项打平（97.3% 不变),表明视觉前瞻对动态场景的贡献远大于静态场景。

**可视化**：预测的未来帧能准确刻画运动趋势（如物块被抓起、瓶子被夹住的轨迹方向),但牺牲了高频细节清晰度——论文认为对于指导动作而言,隐特征包含的动态信息比像素级视觉保真度更重要。

## 四、局限性

论文明确指出两点未解决的问题：

1. 理解专家没有与大规模多模态 VQA 数据联合训练,削弱了通用语义推理与复杂指令跟随能力（即理解专家本身的"聪明程度"未被专门强化)。
2. 为保证生成模块的实时推理效率,牺牲了未来帧预测的保真度,限制了生成细节的粒度——这是一个效率与保真度之间的权衡,论文未给出量化的"细节损失换来多少速度"的分析,只做了定性可视化展示。

此外,论文自身实验设计也有一些未讨论的边界：仿真基准 RoboTwin 2.0 上的提升幅度（+2.6pt）明显小于真实动态任务（+26.7pt）,提示生成专家带来的收益可能与真实世界的物理复杂度/传感器噪声更相关,在低方差的仿真环境中收益有限；另外论文未报告长时程误差累积（生成专家仅预测 $t+15$ 一步之后,是否需要多步 rollout 才能应对更长时间尺度的动态场景)以及生成专家对分布外物体/场景的鲁棒性。

## 五、评价与展望

**优点**：InternVLA-A1 提出了一个工程上可落地的答案,回应了"世界模型式视觉前瞻"与"VLA 语义理解"如何融合而不牺牲实时性这一具体难题——相比 VPP、Genie Envisioner、DreamZero 等依赖大规模预训练视频生成模型（扩散或自回归)做前瞻、动辄跑在个位数 Hz 的方案,本文用 COSMOS VAE 隐空间 + 大幅 token 压缩 + 非自回归并行解码把生成专家做到与理解/动作专家同频（约 13Hz),这是一个务实的效率-能力折中,而非追求生成质量的极致。三专家共享统一掩码自注意力（而非三个独立网络级联)也比"VLM + 独立扩散头"式的双系统架构（如 GR00T N1/N1.5）在信息流上更紧耦合,消融实验（生成专家去除后动态任务成功率腰斩）为这一设计选择提供了较有说服力的证据。

**与同类工作的关系**：本文与作者团队前作 F1（Lv et al., 2025,同样致力于"理解到生成到动作"的桥接,但用 next-resolution 迭代前向,牺牲了实时性)构成明确的技术演进关系；与 π0/π0.5 的差异在于是否显式建模视觉前瞻,与 GR00T N1/N1.5 的差异在于是否用统一 Transformer 而非双系统级联,与 VPP/Genie Envisioner 的差异在于生成模块是否语义 grounded、是否轻量化到可实时。定位上,本文更贴近"给 VLA 加一个轻量、隐空间、单步的前瞻分支",而非"训练一个通用视频世界模型再蒸馏出策略"。

**开放问题**：(1) 生成专家目前只做单步（$t\to t+15$)前瞻,对更长时间尺度、多阶段动态任务（例如需要预判多个物体连续运动的场景)是否需要多步或自回归式 rollout,论文未探讨,这也是效率与前瞻时长之间新的权衡点；(2) 理解专家未联合 VQA 数据训练,其语义推理能力上限尚不明确,是否会在更复杂的长指令、组合泛化任务上暴露短板有待验证；(3) 生成专家的监督信号（COSMOS 隐特征回归)本质上是"教师强制"式的确定性预测,对于本质多模态的未来（例如物体可能倒向左或右)如何避免均值化预测导致的前瞻信息模糊,论文未做讨论；(4) 消融显示仿真数据独立贡献已经很强（RoboTwin 88%+),而真实数据的边际贡献主要体现在分布外真实任务上,如何进一步降低对真实数据的依赖（例如加大仿真域随机化广度)仍是值得跟进的方向。

## 参考

- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.
- Black et al., *π0.5: A Vision-Language-Action Model with Open-World Generalization*, arXiv:2504.16054, 2025.
- Bjorck et al., *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, arXiv:2503.14734, 2025.
- Hu et al., *Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations*, ICML 2025.
- Lv et al., *F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions*, arXiv:2509.06951, 2025.
- Tian et al., *InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy*, arXiv:2511.16651, 2025.
