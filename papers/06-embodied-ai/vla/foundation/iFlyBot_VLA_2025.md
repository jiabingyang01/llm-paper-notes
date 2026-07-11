# iFlyBot-VLA：科大讯飞双臂机器人 VLA 技术报告

> **论文**：*iFlyBot-VLA Technical Report*
>
> **作者**：Yuan Zhang†, Chenyu Xue†, Wenjie Xu†, Chao Ji, Jiajia Wu, Jia Pan* et al.（†共同一作，*通讯作者）
>
> **机构**：iFlyTek Research and Development Group（科大讯飞研究院）；LindenBot
>
> **发布时间**：2025 年 11 月（arXiv 2511.01914）
>
> **发表状态**：未录用（预印本，Technical Report）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.01914) | [PDF](https://arxiv.org/pdf/2511.01914)
>
> **分类标签**：`VLA` `latent action model` `flow matching` `dual-arm manipulation` `LIBERO`

---

## 一句话总结

iFlyBot-VLA 用"隐式隐动作 token（跨本体人类/机器人视频预训练）+ 显式 FAST 离散动作 token"双层监督联合训练 VLM 与 flow-matching 动作专家，并配合两阶段梯度截断/放开策略保护 VLM 通用能力，在 LIBERO 上平均成功率达到 93.8%（超过 π0 的 86% 和 OpenVLA 的 76.5%），真实世界抓取放置、长时序包裹分拣、双臂叠衣任务上均优于同数据量微调的 π0 基线。

## 一、问题与动机

当前主流 VLA 框架多采用"VLM 感知 + 扩散/流式动作专家生成"的混合架构（如 π0），兼顾输入输出兼容性与两者优势，但存在一个核心工程难题：**如何设计训练策略，在最大程度保留 VLM 通用感知/推理能力的同时，让随机初始化的动作专家学会精细、平滑的连续控制**。作者观察到，若让随机初始化的 flow-based 动作专家端到端只用机器人轨迹数据训练，很容易破坏 VLM 骨干原有的感知能力，而这一能力恰恰是策略泛化性的关键。

另一方面，早期自回归 VLA（如 OpenVLA 把连续动作离散化为 256 个 bin）在动作精度与可扩展性上存在瓶颈；FAST 用 DCT 频域压缩 + BPE 编码提升了离散动作的学习效率；LAPA、UniVLA 等隐动作方法证明用 VQ-VAE 从无标注视频中学习离散隐动作空间，能有效桥接视觉语言感知与细粒度动作生成，但如何把隐动作表示和显式低层动态表示结合、协同监督 VLM 与动作专家，仍是开放问题。

## 二、核心方法

### 1. 整体架构

模型 $\pi_\theta$ 由语言 Transformer 骨干（Qwen2.5-VL 3B）和一个 Diffusion Transformer 动作专家组成，控制双臂机器人生成长度为 $k$ 的动作块：

$$a_t = \pi_\theta(l, o_t, s_t) \tag{1}$$

其中 $l$ 为语言指令、$o_t$ 为多视角视觉观测、$s_t$ 为机器人本体状态。VLM 每层输出的 KV cache 会传给下游动作专家，但**只有 latent action token 对应的 KV cache 被保留转发，discrete FAST action token 对应的 KV cache 被丢弃不转发**——因为离散动作 token 数量大会拖慢推理，且其特征与动作专家输出高度相似，若直接提供给专家容易导致过拟合、削弱泛化。

### 2. 隐动作模型（Stage I：Latent Action Training）

用编码器–解码器结构的 VQ-VAE 以自监督方式从大规模人类 + 机器人操作视频中学习隐动作。编码器含空间 Transformer + 时间 Transformer，输入当前帧 $o_t$ 与未来帧 $o_{t+k}$（间隔固定为 1 秒，$k$ 依各数据集帧率而定），输出隐动作 $c_t$；解码器只含空间 Transformer，输入 $c_t$ 和 $o_t$ 重建更远的未来帧 $o_{t+H}$。

量化目标：

$$c_t = \arg\min_n \|x_{enc} - c_n\|^2 \tag{2}$$

大白话：把 encoder 输出的连续特征映射到码本中距离最近的离散码字，相当于把"两帧之间的动作变化"压缩成有限个"动作词"。码本大小 $|C|=32$，每步检索 8 个离散码。

复现标准 VQ-VAE 时遇到梯度坍塌问题，作者改用 **NSVQ**（noise substitution vector quantization）替代原始 straight-through estimator：

$$c'_t = x_{enc} + \frac{\|x_{enc}-c_t\|}{\|w\|}w \tag{3}$$

其中 $w \sim \mathcal N(0,1)$。大白话：不直接把量化误差当梯度直通，而是用一个随机噪声方向、按误差范数缩放后加回 encoder 输出，用噪声近似模拟量化带来的跳变误差分布，从而避免优化器"看到"离散跳变导致梯度消失/坍塌。此外解码时对当前帧做 stop-gradient，强迫模型必须依赖隐动作 $c_t$ 才能解码未来帧，避免模型走捷径。监督 loss 为重建帧与目标帧的 MSE。

### 3. 离散动作 token 编码（Discrete Action Token Encoding）

采用 FAST Action Token 方法对未来动作窗口 $a_t$ 编码为离散 token，**仅用于监督 VLM**、隐式引导其学习动作相关语义（辅助隐式规划），训练与推理时其对应特征均不进入动作专家，以兼顾生成质量与推理效率。

### 4. VLA 架构与双层监督

VLM 被同时监督预测两套序列：latent action token（引导规划与空间落地）和 discrete FAST action token（引导动作相关语义学习）。proprioceptive 状态通过占位 token + 全连接层特征替换的方式注入。

动作专家为 Diffusion Transformer，采用 flow-matching 建模连续动作分布：

$$L^\tau(\theta) = \mathbb{E}_{p(A_t\mid l,o_t,s_t),\, q(A_t^\tau\mid A_t)} \left\| \pi_\theta(A_t^\tau, l, o_t, s_t) - \pi(A_t^\tau \mid A_t) \right\|^2 \tag{4}$$

其中带噪动作 $A_t^\tau = \tau A_t + (1-\tau)\epsilon$，$\epsilon\sim\mathcal N(0,1)$，目标去噪场 $\pi(A_t^\tau\mid A_t)=\epsilon - A_t$。大白话：训练时把真实动作和随机噪声按比例 $\tau$ 线性混合成"带噪动作"，网络学习预测一个指向真实动作方向的速度场，本质是 flow-matching/rectified flow 在动作序列上的应用。动作专家对同一动作窗口内所有 token 采用**全双向 attention mask**（非因果），使窗口内 token 可并行去噪，兼顾时序连续性与生成效率；训练时 $\tau$ 的采样权重用 Beta 分布，偏向给较小（更嘈杂）的 timestep 更高权重。

推理时从随机高斯噪声 $A_t^0\sim\mathcal N(0,1)$ 出发，做离散前向 Euler 积分：

$$A_t^{\tau+\sigma} = A_t^\tau + \sigma\, \pi_\theta(A_t^\tau, l, o_t, s_t) \tag{5}$$

实践中用 5 步积分（$\sigma=0.2$）。VLM 的 KV cache 只需计算一次，不随积分步数重复计算，从而保证高推理效率。

### 5. 三阶段训练与梯度调度

- **Stage I** 隐动作训练（同上）；
- **Stage II** 基础预训练：VLM 在混合数据（内部空间 QA + OXE 子集 + AgiBot-World 子集 + 全部自采双臂数据）上训练。对纯 QA 样本，action 输出置零、不计算 action loss，且**梯度不传入动作专家**（截断梯度流），避免随机初始化的动作专家污染已有 VLM 能力；
- **Stage III** 任务特定后训练：针对叠衣、杂乱场景操作、抓取不规则物体等复杂任务用高质量自采数据二阶段微调。此阶段**放开**动作专家到 VLM 的梯度回传（因 VLM 此时已具备强表示，不再易受破坏），并采用 multi-sample noise perturbation——对同一动作序列采样多个不同噪声版本联合去噪 + 反传，加速训练、提升动作专家稳定性。

### 6. 训练数据

隐动作模型训练数据涵盖人类操作视频（HoloAssist、Ego4D、EgoDex、HOI4D、Something-Something V2、EgoVid）与单/双臂机器人数据集（OXE、AgiBot、RoboMind、Galaxea）。iFlyBot-VLA 正式训练数据共 3,594K 条：机器人轨迹占 92.6%（OXE 子集 23%、AgiBot-World 18.4%、自采数据 51.2%）+ VQA 数据 7.4%；QA 数据内部又分 Understanding QA（19.7%）与 Perception QA（80.3%，含 2D Trajectory、3D Grounding、Space/Object Pointing、Affordance、2D Grounding 等子类）。

自采 iFLYTEK 数据集使用 **26 台双臂机器人**（两种配置）采集，含三类任务：叠衣（5 种 T 恤 + 3 种短裤，各 190 条轨迹，均 4.5 分钟，共约 110 小时）、通用抓取放置（30 类物体，各 400 条轨迹，均 27 秒，共约 90 小时）、长时序包裹分拣（2,752 条轨迹，均 61 秒，共约 47 小时）。动作/状态统一 padding 到 20 维（左臂 10 + 右臂 10）；单臂数据集在 Stage-I 训练时动作数据随机分配给左/右臂以保持维度一致。

## 三、实验结果

### LIBERO 仿真基准

四个任务套件各 10 任务、10 条示教，遵循 OpenVLA 数据预处理（剔除失败示教）。LIBERO-Long 训练 70,000 步，其余套件 50,000 步，全局 batch size 64，动作窗口大小 7；仅用第三人称图像 + 文本指令；LIBERO 样本与预训练/隐动作训练数据严格无重叠，确保泛化测试的严格性。

正文明确给出的关键数字：

| 方法 | LIBERO 平均成功率 | LIBERO-Goal |
|---|---|---|
| OpenVLA | 76.5% | — |
| π0 | 86.0% | 95%（四套件中最高） |
| **iFlyBot-VLA** | **93.8%** | 93%（唯一未反超 π0 的套件） |

iFlyBot-VLA 在 Spatial / Object / Long 三个套件上均取得 SOTA（具体数值见论文 Fig.6 柱状图，未在正文给出精确表格），仅 LIBERO-Goal 略低于 π0。

### LIBERO 消融（显式 + 隐式规划的贡献）

| 配置 | 平均成功率 | 相对完整模型 |
|---|---|---|
| w/o FAST and LAM（去掉两种动作 token 监督） | 73% | -20.8% |
| w/o FAST（只保留隐动作监督） | 87.8% | -6% |
| w/o LAM（只保留 FAST 显式监督） | 90.3% | -3.5% |
| **iFlyBot-VLA（完整）** | **93.8%** | — |

两种监督机制均正向贡献，尤其在 LIBERO-Long 长时序任务上组合效应最明显。

### 真实世界实验

**通用抓取放置**（175 小时数据，约 32,000 条轨迹，30 类物体，指令模板 "put A into B"；π0 用同一自采数据按官方流程微调作为基线）：

| 配置 | π0 基线 | iFlyBot-VLA |
|---|---|---|
| Basic | 94.79% | 96.25% |
| Light Illumination Variation | 92.71% | 96.04% |
| Unseen Objects | 81.67% | 88.21% |
| Unseen Scenes | 87.91% | 93.57% |

（每配置 20 次尝试，seen 物体 24 类 / unseen 物体 14 类）

**长时序包裹分拣**（47 小时数据，2,752 条轨迹，指令 "若标签朝上则翻转包裹并放入篮筐"）：40 次重复试验，每次 3 个包裹（2 个需翻转）。在 Allow Correction 评价标准（允许最多 2 次纠正）下，iFlyBot-VLA 比 π0 基线**成功率高 7.5 个百分点**。

**双臂叠衣任务**（110 小时数据，8 类衣物共约 1,600 条轨迹）：按拾取衣物/拖拽/摊平/纵向对折/完成分步评估，iFlyBot-VLA 在每一步完成率上均优于 π0 基线（Fig.12(b)，未给出精确数值表）；受限于与基线公平比较，主对比采用未优化的 drag-flattening 推理方式，若换用 flick-flattening 并配合优化推理代码，"摊平"单步成功率可达约 90%（仅见于项目主页演示视频，未纳入主表格对比）。每次完整执行限时 3 分钟。

## 四、局限性

作者在第 6 节明确指出：

- 面对训练中未见过的新概念/新指令仍会失败，尤其是抓取从未见过形状的物体；
- 与所有模仿学习方法一样，面对分布外（OOD）输入时可能难以维持或恢复性能；
- 未来计划扩大模型规模、扩充训练数据、引入更丰富的空间表示，并计划引入强化学习（RL）机制进一步提升泛化能力与鲁棒性，以克服模仿学习的固有局限——但报告未给出任何 RL 相关的初步实验或具体路线图。

## 五、评价与展望

**优点**：（1）双层动作监督（latent action + FAST discrete token）把"隐式高层意图"和"显式低层动力学"解耦，只让隐动作 token 的 KV 进入动作专家，兼顾了避免过拟合与控制推理时延，消融实验也验证了两者对长时序任务的显著贡献；（2）两阶段梯度截断/放开策略针对性解决了"随机初始化的动作专家污染预训练 VLM"这一常见工程痛点，是一个简单但有效的方案，与 π0 系列同类工作面临的问题相互呼应；（3）实验设计较为扎实：LIBERO 严格保证与预训练/隐动作训练数据零重叠，真实世界实验覆盖抓取放置、长时序变形物体分拣、双臂叠衣三种难度递增的任务，且均与同数据量微调的 π0 做了对照。

**局限与开放问题**：（1）作为 Technical Report，工程细节（如 codebook 学习率、NSVQ 超参、Beta 分布参数、QA 数据的具体 prompt 模板）披露有限，核心代码与权重尚未开源（仅承诺未来开源部分数据集与代码），多数结果目前难以独立复现验证；（2）隐动作学习范式（VQ-VAE encoder-decoder 重建未来帧）与 LAPA、UniVLA 思路相近，论文的主要增量在训练阶段的梯度调度与双层 token 分工，而非隐动作学习本身的范式创新，且未与 UniVLA 做直接数值对比；（3）基线选择偏少，仅对比了 LAPA、OpenVLA、π0，未与同期更新的 VLA 工作（如 π0.5、GR-3、InternVLA-M1，均出现在其参考文献中）做正面比较，"SOTA"结论的说服力有限；（4）多数关键图表（Fig.6/7/9/12）仅以柱状图呈现、正文未给出精确数值表格，部分对比只能目测估读，透明度不足；（5）叠衣任务中 "flick-flattening≈90%" 的结果未纳入与基线的公平对比表格，这类"仅在演示视频中展示的最佳结果"容易造成过度乐观的印象。

**与公开工作的关系**：该工作沿着"隐动作预训练（LAPA、UniVLA）+ flow/diffusion 动作专家（π0 系列）"两条主线做工程整合，架构思路与同期"VLM + action expert"混合范式一脉相承；其贡献更多体现在训练策略（分阶段梯度调度、双层动作 token 分工）与大规模自采双臂数据（尤其是衣物、软包裹等变形物体操作）上，而非提出全新的架构范式。

## 参考

1. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. Ye et al. *LAPA: Latent Action Pretraining from Videos*. arXiv:2410.11758, 2024.
3. Bu et al. *UniVLA: Learning to Act Anywhere with Task-centric Latent Actions*. arXiv:2505.06111, 2025.
4. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.
5. Black, Galliker & Levine. *Real-Time Execution of Action Chunking Flow Policies*（FAST Action Token 方法来源）. arXiv:2506.07339, 2025.
