# World Pilot：用世界-动作先验引导视觉-语言-动作模型

> **论文**：*World Pilot: Steering Vision-Language-Action Models with World-Action Priors*
>
> **作者**：Zefu Lin, Rongxu Cui, Junjia Xu, Lue Fan, Zhaoxiang Zhang, et al.
>
> **机构**：中国科学院自动化研究所（Institute of Automation, CASIA）；南京大学；北京航空航天大学
>
> **发布时间**：2026 年 06 月（arXiv 2606.12403）
>
> **发表状态**：未录用（预印本，正文 Acknowledgments 部分仍为模板占位文字，尚未定稿）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.12403) | [PDF](https://arxiv.org/pdf/2606.12403)
>
> **分类标签**：`VLA` `World-Action Model` `Latent Steering` `Flow Matching` `OOD Robustness`

---

## 一句话总结

World Pilot 把一个全程冻结的视频预训练 World-Action Model（WAM，实例化为 Cosmos Policy）产生的两路输出——场景演化 latent 与粗粒度动作轨迹——分别通过 Latent Steering（感知层残差交叉注意力）和 Action Steering（动作生成层单 token 前缀）注入 VLA 决策链，在 LIBERO-Plus 零样本 OOD 基准上以 84.7% 的 Total 成功率取得 SOTA（比最强基线高 2.6 点），并在四项真机任务的全部 12 个 ID/OOD 设置上都取得最高成功率，OOD 掉点普遍控制在 20 点以内（基线普遍掉 25-50 点）。

## 一、问题与动机

VLA 策略的语义 grounding 来自 VLM 主干在静态图文对上的大规模预训练，在其微调所覆盖的操作分布内表现称职。但图文对预训练无法提供"场景如何在动作作用下演化"的模型：VLM 下游的动作生成器消费的是纯语义隐状态，对自己必须产生的连续、接触密集的动力学过程没有内部建模。因此一旦视角、几何或接触容差偏离训练分布，VLA 就会变得脆弱。

视频预训练是天然的补充：动作条件下的场景演化在视频中天然存在。Cosmos Policy、mimic-video、DreamZero 等视频预训练的 World-Action Models（WAM）学到的场景动态与接触动力学表示能跨具身、跨视觉条件迁移。WAM 的输出恰好对应 VLA 缺失的那部分——一个描述可见状态将如何变化的场景演化 latent，以及一个粗粒度的动作轨迹假设（该轨迹正是 latent 所预测效果的成因）。由于两个预测来自同一个联合训练的共享编码器，二者在结构上天然对齐：语义 grounding 由 VLA 提供，场景动态由 WAM 提供，二者互补。

真正的难点不在于"要不要把两个模型放在一起"，而在于如何把 WAM 的信号转化为策略能力的实际提升：提取哪些信号、以什么形式携带、注入 VLA 的哪一层，才能让动态知识到达真正需要它的部位而不在传输中被稀释。已有工作探索过若干路线（联合生成未来图像和动作、用预测未来图/子目标图引导策略、用 latent 或隐式特征传递世界模型知识），但要么让视觉重建损失污染动作表征、要么在像素空间引入了大量与控制无关的外观细节、要么只依赖静态的未来快照而非连续时空演化。World Pilot 要回答的正是这个"信号-形式-入口"的设计问题。

## 二、核心方法

### 2.1 问题建构与整体框架

标准 VLA 用 VLM 把图像和语言编码成多模态隐状态，再由动作生成器产生动作 chunk $\mathbf{A}_t=(a_t,\dots,a_{t+K-1})$。World Pilot 额外引入一个视频预训练的 WAM，从相同输入联合预测场景演化 latent 和粗粒度动作轨迹假设：

$$(\mathbf{Z}_t^w, \widetilde{\mathbf{A}}_t^w) = W_\phi(\mathbf{O}_t, \ell, \mathbf{q}_t), \qquad \hat{\mathbf{A}}_{\theta,t} = \pi_\theta(\mathbf{O}_t, \ell, \mathbf{q}_t; \mathbf{Z}_t^w, \widetilde{\mathbf{A}}_t^w)$$

用大白话说：WAM 看一眼当前观测和指令，同时吐出"画面会怎么变"（latent）和"大概会怎么动"（粗轨迹）两个预测；VLA 在原本输入基础上，把这两个预测也当作条件，生成最终可执行的动作 chunk。

两路信号通过两条通路进入决策链（图 2）：语义主干先把当前图像和指令编码为 VLM 隐状态 $\mathbf{H}_t$；Latent Steering 在感知层用 $\mathbf{Z}_t^w$ 调制 $\mathbf{H}_t$ 得到动态增强隐状态 $\bar{\mathbf{H}}_t$；Action Steering 在动作生成层把 $\widetilde{\mathbf{A}}_t^w$ 编码成一个轨迹级前缀 token。两者都是加性的：Latent Steering 以残差方式加到 $\mathbf{H}_t$ 上、不改变 token 序列长度和结构；Action Steering 只在生成器输入序列里插入一个前缀 token、不改变去噪递归过程，因此两条通路可以互相独立地做消融（第三节实验证实）。

### 2.2 Latent Steering（感知层）

WAM 用 VAE 编码当前观测 $\mathbf{O}_t$，再用 Diffusion Transformer（DiT）去噪，得到逐视角的场景演化 latent $\mathbf{Z}_t^w$。之所以传 latent 而不是解码出来的未来图像，是因为像素内容携带了大量与动作无关的细节（纹理、光照、背景、生成伪影），会稀释 latent 直接编码的动态结构。

World Pilot 用一个动态编码器 $f_{\text{dyn}}$ 投影该 latent，再加一个时间嵌入 $\rho_{\text{fut}}$ 标记这些 token 属于"未来场景"（实验发现不加这个标记，先验的贡献会明显减弱）：

$$\mathbf{D}_t^w = f_{\text{dyn}}(\mathbf{Z}_t^w) + \rho_{\text{fut}}$$

设 $\mathbf{H}_t \in \mathbb{R}^{L\times d}$ 为 VLM 隐状态，Latent Steering 用交叉注意力把 $\mathbf{D}_t^w$ 的信息路由进来，再以残差形式加回：

$$\bar{\mathbf{H}}_t = \mathbf{H}_t + \text{CrossAttn}(\mathbf{H}_t, \mathbf{D}_t^w)$$

用大白话说：让每个 VLM token 自己去"未来场景 token 池"里挑跟自己空间位置最相关的动态线索加权融合，而不是给全体 token 打一个统一的"全局补丁"；残差形式保证不破坏原有 VLM 的 token 结构，$\bar{\mathbf{H}}_t$ 可以直接无缝接入标准的动作生成路径，不需要任何下游接口改动。

### 2.3 Action Steering（动作生成层）

WAM 产生的粗轨迹 $\widetilde{\mathbf{A}}_t^w$ 先通过重采样对齐到 VLA 的动作 horizon $K$，再用动作编码器 $f_{\text{act}}$ 压缩成单个先验 token：

$$\mathbf{s}_t^w = f_{\text{act}}\big(\text{Align}_K(\widetilde{\mathbf{A}}_t^w)\big)$$

单 token 只概括轨迹的整体运动形状，而不是把每一步都钉死到 WAM 的具体数值——这样生成器仍有自由度，去产出一个同时反映先验和动态增强隐状态的连续 chunk；逐步（per-step）对齐反而会把每个输出步骤都拴在对应的 WAM 步骤上，实验发现当 WAM 轨迹本身不够精确时，这样做鲁棒性更差。

flow-matching 动作生成器在 flow time $\tau$ 下把带噪轨迹 $\mathbf{X}_{\tau,t}$ 去噪到干净动作 chunk。World Pilot 把输入扩展为 $[\mathbf{u}_t; \mathbf{s}_t^w; \mathbf{Q}_t; \mathbf{X}_{\tau,t}]$（$\mathbf{u}_t$ 为可选状态 token，$\mathbf{Q}_t$ 为可学习的未来查询 token），$\bar{\mathbf{H}}_t$ 作为交叉注意力条件；$\mathbf{s}_t^w$ 以前缀形式加入，通过自注意力条件化去噪递归，但自身不参与被去噪的过程。

### 2.4 训练目标

沿用 ABot-M0 的 clean-action 参数化（等价于动作到速度变换下的重加权速度空间目标），监督目标始终是专家 chunk $\mathbf{A}_t^\*$，WAM 先验只通过条件路径起作用，不需要额外的先验损失。给定高斯噪声 $\epsilon$ 和采样的 flow time $\tau$，带噪轨迹为 $\mathbf{X}_{\tau,t} = \tau\mathbf{A}_t^\* + (1-\tau)\epsilon$，动作生成器预测：

$$\hat{\mathbf{A}}_{\theta,t} = g_\theta(\mathbf{X}_{\tau,t}, \tau, \mathbf{u}_t, \mathbf{s}_t^w, \mathbf{Q}_t \mid \bar{\mathbf{H}}_t)$$

训练目标为：

$$\mathcal{L}_{\text{World Pilot}} = \mathbb{E}_{\tau,\epsilon}\Big[w(\tau)\,\|\hat{\mathbf{A}}_{\theta,t} - \mathbf{A}_t^\*\|_2^2\Big], \qquad w(\tau) = \frac{1}{(1-\tau)^2}$$

用大白话说：整个训练目标其实还是标准的动作回归损失，只是每个 flow time 上按 $1/(1-\tau)^2$ 加权，等价于速度空间下的 flow-matching 损失；WAM 全程冻结，梯度只更新 VLA 侧参数（VLM 主干、Latent Steering 交叉注意力、动作编码器 $f_{\text{act}}$、flow-matching 生成器）。WAM 的前向可以离线预计算并缓存，训练时不进主循环，因此 World Pilot 是"用一个现成的世界模型来引导 VLA"，而不是联合训练一个新的世界模型；VLA 微调的梯度也从不回传进 WAM，不会扰动其预训练得到的世界先验。

实现细节：World Pilot 建在 ABot-M0 之上，VLM 主干为 Qwen3-VL，动作头为基于 DiT 的 flow-matching，WAM 用 Cosmos Policy（5 步去噪）；训练时对 WAM 条件 $\mathbf{D}_t^w$ 和 $\mathbf{s}_t^w$ 施加 dropout rate 0.3，防止策略过度依赖先验；在 8 张 RTX PRO 6000 上微调。

## 三、实验结果

评测覆盖两个仿真基准（LIBERO-Plus：基于 LIBERO 构建的 10,030 个扰动任务，覆盖 Camera / Robot / Language / Light / Background / Noise / Layout 七个扰动轴，只在 LIBERO 上训练、在扰动上零样本评测；RoboCasa：厨房场景长时程操作）和四项真机任务（叠积木、叠毛巾、水果放盘、盖子对齐），并配套四组消融。

#### 仿真主实验（LIBERO / LIBERO-Plus 各轴 / RoboCasa，Total 为 LIBERO-Plus 全部扰动任务的平均成功率，三随机种子平均）

| 方法 | LIBERO | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** | RoboCasa |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | 84.7 | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 | – |
| WorldVLA | 81.8 | 0.1 | 27.9 | 41.6 | 43.7 | 17.0 | 10.9 | 38.0 | 25.0 | – |
| UniVLA | 95.2 | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 | – |
| π0 | 94.4 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 | 42.4 |
| π0.5 | 96.9 | – | – | – | – | – | – | – | 77.4 | 41.4 |
| RIPT-VLA | 93.6 | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 | – |
| DreamVLA | 97.5 | 26.2 | 17.6 | 67.0 | 77.5 | 71.5 | 53.6 | 43.5 | 48.9 | – |
| Being-H0.7 | 99.2 | – | – | – | – | – | – | – | 82.1 | 62.1 |
| Cosmos Policy | 98.5 | 69.6 | 51.0 | 89.6 | 97.7 | 85.7 | 87.3 | 83.7 | 79.7 | 67.1 |
| ABot-M0 | 98.6 | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 | 54.0 |
| **World Pilot（本文）** | 98.5 | **82.8** | 60.6 | 87.2 | **98.6** | **96.4** | **93.6** | 80.5 | **84.7** | 65.5 |

World Pilot 在 Camera 轴取得全表最大单轴增益（82.8，比次优基线高 13.2 点），在外观相关的 Light / Background / Noise 三轴全部领先（图文预训练带来外观鲁棒性、视频预训练带来相机位姿鲁棒性，二者叠加）；在 Language / Robot / Layout 三轴落后于最强基线；在 LIBERO 本身（分布内）已接近饱和（多数方法 >98%），增益集中体现在 OOD 轴上；在 RoboCasa 上 65.5 具有竞争力但并非最优（不敌 Cosmos Policy 的 67.1）。

#### 真机四任务成功率（每个设置 20 trials，括号内为相对 ID 设置的绝对下降）

**Stack Blocks**（OOD：色块颜色、堆叠高度）

| 方法 | ID | Color | Height |
|---|---|---|---|
| π0.5 | 40 | 15 (-25) | 0 (-40) |
| ABot-M0 | 60 | 25 (-35) | 10 (-50) |
| Cosmos Policy | 65 | 30 (-35) | 15 (-50) |
| **World Pilot** | **70** | **55 (-15)** | **50 (-20)** |

**Fold Towel**（OOD：毛巾方向、新毛巾实例）

| 方法 | ID | Direction | Novel Towel |
|---|---|---|---|
| π0.5 | 55 | 25 (-30) | 10 (-45) |
| ABot-M0 | 50 | 20 (-30) | 5 (-45) |
| Cosmos Policy | 45 | 15 (-30) | 10 (-35) |
| **World Pilot** | **85** | **75 (-10)** | **70 (-15)** |

**Fruit-to-Plate**（OOD：新水果品类、新摆放布局）

| 方法 | ID | Novel Fruit | Layout |
|---|---|---|---|
| π0.5 | 35 | 10 (-25) | 5 (-30) |
| ABot-M0 | 65 | 30 (-35) | 15 (-50) |
| Cosmos Policy | 70 | 35 (-35) | 20 (-50) |
| **World Pilot** | **90** | **75 (-15)** | **70 (-20)** |

**Container-Lid Alignment**（OOD：新物体、新盖子姿态；对闭合几何容差要求最严）

| 方法 | ID | Novel Object | Lid Pose |
|---|---|---|---|
| π0.5 | 40 | 15 (-25) | 5 (-35) |
| ABot-M0 | 60 | 25 (-35) | 10 (-50) |
| Cosmos Policy | 60 | 25 (-35) | 10 (-50) |
| **World Pilot** | **80** | **70 (-10)** | **65 (-15)** |

World Pilot 在全部 12 个 ID/OOD 设置上都取得最高成功率，ID 到 OOD 的掉点始终控制在 20 个百分点以内，而基线普遍掉 25-50 点。Container-Lid Alignment 是几何容差最严的设置：在 OOD 姿态/物体变化下 World Pilot 成功 13-14/20 次，而所有基线均不超过 6/20 次。

#### 消融一：两条通路各自的贡献（LIBERO-Plus Total）

| 变体 | Success (%) |
|---|---|
| ABot-M0（baseline） | 80.5 |
| 仅 Latent Steering | 83.7 (+3.2) |
| 仅 Action Steering | 83.1 (+2.6) |
| 完整 World Pilot | 84.7 (+4.2) |

#### 消融二：WAM 未经动作后训练时先验是否仍有效（把 Cosmos Policy 换成未做动作后训练、只产生未来 latent 的 Cosmos-Predict，仅激活 Latent Steering）

| Benchmark | ABot-M0 | + Latent Steering（Cosmos-Predict） |
|---|---|---|
| LIBERO-Plus | 80.5 | 82.6 (+2.1) |
| RoboCasa | 54.0 | 62.7 (+8.7) |
| RoboTwin2.0（clean） | 81.2 | 85.3 (+4.1) |

即便 WAM 只是纯场景预测模型、从未见过动作后训练，Latent Steering 依然在三个基准上都带来正向增益；换成经过动作后训练的 Cosmos Policy 后信号进一步锐化（同配置下 Latent-Steering-only 从 82.6 提升到 83.7，即 +1.1），但先验在没有动作后训练时也已生效。

#### 消融三：未来场景表示形式（LIBERO-Plus）

| 未来信息形式 | Success (%) |
|---|---|
| 1 步 latent | 84.6 |
| 3 步 latent | 84.5 |
| 5 步 latent（默认） | 84.7 |
| 解码后的未来图像 | 83.5 |

三种 latent 深度的结果彼此相差不超过 0.2 点，说明 World Pilot 依赖的是状态转移线索和局部动态结构而非像素级真实感；换成完整解码的未来图像反而下降 1.2 点，因为像素解码引入视觉伪影、稀释了动态结构。

#### 消融四：动作先验的注入形式（LIBERO-Plus）

| 形式 | Success (%) |
|---|---|
| 单个编码 token（World Pilot 默认） | 84.7 |
| 逐步编码 token | 83.6 |
| 用 $\widetilde{\mathbf{A}}_t^w$ 做 flow 初始化 | 84.1 |
| 直接使用原始 $\widetilde{\mathbf{A}}_t^w$ | 83.0 |

单 token 形式最优；逐步 token 和原始轨迹都把生成过程钉死在带噪的逐步信号上，会把 WAM 轨迹的噪声传播并在 chunk 内累积误差；flow 初始化能挽回部分差距，但把最终输出和 WAM 的动作质量绑定得更紧，压缩了生成器用 VLA 侧线索纠正先验的空间。

## 四、局限性

论文第五节明确列出了以下局限：

- **继承 WAM 的覆盖边界**：一旦测试场景落在 WAM 视频预训练分布之外，两路先验都会退化，增益随之收窄。
- **增益并不均匀**：在 LIBERO-Plus 的 Language、Robot、Layout 三个轴上落后于最强基线；真机 OOD 成功率相对 ID 仍下降 10-20 个百分点，说明先验只是削弱、而非消除 OOD 偏移的影响；在 RoboCasa 上也未取得最优（65.5 对 Cosmos Policy 的 67.1）。
- **仅通过动作损失耦合**：WAM 与 VLA 是模块化设计，仅通过动作损失间接耦合，这保持了组件可替换性（可换更强世界模型或不同 VLA 主干），但没有做更紧的先验-策略协同适配，联合训练本可能提供这种适配空间。
- **额外推理开销**：每个决策步都要多做一次 WAM 前向（VAE 编码 + DiT 去噪），限制了在高频反应式控制场景下的适用性；论文未给出具体延迟/吞吐的量化数字。

作者提出的三个后续方向：不确定性感知的先验门控（应对 WAM 覆盖不足）、WAM-VLA 联合协同微调（closing the prior-policy loop）、先验蒸馏或自适应查询（降低每步开销）。

## 五、评价与展望

**优点**：整体设计干净、模块化，WAM 全程冻结、只用两条互补通路（latent 残差交叉注意力 + 单 token 前缀）分别向感知层和动作生成层注入动态先验和运动先验，工程上易于替换 WAM 或 VLA 主干。消融做得比较扎实：不仅验证了两条通路各自独立的贡献（消融一），还系统地对"先验以什么形式注入"做了控制变量实验——latent 优于解码图像（消融三）、单 token 优于逐步硬对齐/flow 初始化/原始轨迹（消融四）——论据方向合理，避免了把生成器过度绑定到 WAM 可能不准的中间输出上。消融二尤其有说服力：即使把 WAM 换成从未做过动作后训练的纯场景预测模型（Cosmos-Predict），Latent Steering 仍能在三个基准上全面提升，说明该框架真正消费的是"世界模型的通用场景动态先验"而不苛求 WAM 本身已经是个动作模型，这为方法对不同来源 WAM 的普适性提供了较强证据。真机结果中，Container-Lid Alignment 这类几何容差最紧的接触密集型任务上 margin 尤其突出（基线全部 ≤6/20，World Pilot 达 13-14/20），体现了视频预训练动态先验确实补上了 VLA 缺失的信息。

**局限与开放问题**：RoboCasa 上并非最优，作者自己也只claim"competitive"，说明该框架在长时程厨房任务上的增益不如在 OOD 扰动鲁棒性上明显，这可能与 RoboCasa 更依赖语言/任务规划而非底层动态先验有关，与 LIBERO-Plus 上 Language/Layout/Robot 三轴落后的现象相呼应。与近期同类"把 WAM 信号路由进 VLA"的工作（论文 Related Work 中提到的 Motus、DreamVLA、$\pi_{0.7}$、VISTA、Being-H0.7、WoG 等）相比，World Pilot 的核心贡献主要落在"latent 残差交叉注意力 + 单 token 前缀"这一具体设计选择上，概念上属于该赛道内的工程化改进而非全新范式；真正的增量价值在于系统性地做了"先验形式"的受控消融，为后续工作提供了"怎么喂先验"的经验性结论。作者自陈的三个未来方向都指向同一个核心矛盾：当前 WAM 与 VLA 解耦训练虽然工程简洁、便于替换组件，但牺牲了先验与策略协同适配的空间；如何在保持模块化的同时获得联合训练的收益，以及如何摊销每步额外 WAM 前向带来的推理开销，仍是开放问题。此外，论文 Acknowledgments 一节仍为模板占位文字，说明这是一份尚未定稿、未经同行评审的预印本，文中数字应视为作者自报告结果。

## 参考

1. M. J. Kim, Y. Gao, T.-Y. Lin, et al. *Cosmos Policy: Fine-tuning video models for visuomotor control and planning*. arXiv:2601.16163, 2026.（本文使用的 WAM 实例）
2. Y. Yang, S. Zeng, T. Lin, et al. *ABot-M0: VLA foundation model for robotic manipulation with action manifold learning*. arXiv:2602.11236, 2026.（本文的 VLA 基座）
3. S. Fei, S. Wang, J. Shi, et al. *LIBERO-Plus: In-depth robustness analysis of vision-language-action models*. arXiv:2510.13626, 2025.（主要评测基准）
4. Physical Intelligence, K. Black, N. Brown, et al. *$\pi_{0.5}$: A vision-language-action model with open-world generalization*. arXiv:2504.16054, 2025.
5. W. Zhang, H. Liu, Z. Qi, et al. *DreamVLA: A vision-language-action model dreamed with comprehensive world knowledge*. arXiv:2507.04447, 2025.
