# GigaWorld-Policy：以动作为中心的高效世界-动作模型

> **论文**：*GigaWorld-Policy: An Efficient Action-Centered World–Action Model*
>
> **作者**：GigaWorld Team（Angen Ye、Boyuan Wang、Chaojun Ni、Guan Huang 等，按字母序排列）
>
> **机构**：GigaAI
>
> **发布时间**：2026 年 03 月（arXiv 2603.17240）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.17240) | [PDF](https://arxiv.org/pdf/2603.17240)
>
> **分类标签**：`世界-动作模型` `因果注意力` `flow matching` `低延迟推理` `动作中心监督`

---

## 一句话总结

GigaWorld-Policy 用一个 5B 参数的因果扩散 Transformer（Wan 2.2 初始化）把"未来视觉动态预测"设计成训练期的稠密辅助监督、推理期可选的分支，在 RoboTwin 2.0 上以 Motus 9 分之一的推理延迟（360ms vs 3231ms）取得与 Motus 相当的仿真成功率（0.86 vs 0.89），并在 AgileX PiPER 真机四任务上以平均 0.83 的成功率超过所有对比方法（比 Motus 高 7 个百分点，比 π0.5 高约 14 个百分点），仅用 10% 训练数据即可达到 π0.5 用满量数据的成功率。

## 一、问题与动机

VLA（Vision-Language-Action）模型的核心痛点是**监督稀疏**：观测和任务条件是高维、语义丰富的，而动作监督却稀疏且低多样性，容易让模型退化为依赖浅层上下文线索、把多变的情形压缩成少数几种重复行为，而不是学习具有物理约束的动作。

已有两类补救思路：

1. **VLM-based VLA + 未来观测预测辅助损失**（如 Cen et al. 2025 WorldVLA、Ni et al. 2025、Zhang et al. 2025 DreamVLA）：在现有 VLA 框架里注入未来状态预测作为辅助监督（论文 Fig.2(a)）。但 VLM 骨干本身是为判别式推理优化的，不易保证预测图像的高保真度和物理一致性。
2. **World Model（WM）与策略学习结合，即 World-Action Model（WAM）**（如 Motus、Cosmos-Policy、Mimic-video、LingBot-VA 等）：利用视频生成模型学到的时空先验提供稠密监督。但这类方法通常需要在推理时迭代采样、rollout 出完整未来视频（Fig.2(b)(c)），带来高延迟；且视频预测中的误差会沿着"先生成视频再解码动作"的链路传播、复合放大，尤其在长时程闭环控制中会因早期小误差累积而失控。

GigaWorld-Policy 的定位（Fig.2(d)）：把未来视觉动态**只在训练期**当作稠密监督/推理信号来使用，推理期显式视频预测变成**可选**分支——默认只解码动作，从而避免视频 rollout 带来的高延迟与误差传播，同时仍能获得世界模型式的稠密监督收益。

## 二、核心方法

### 2.1 问题形式化

标准 VLA 策略建模为条件分布

$$a_{t:t+p-1} \sim q_\Theta(\cdot \mid o_t, s_t, l),$$

即仅依赖演示中的动作监督，观测空间没有显式监督信号。

GigaWorld-Policy 用同一个统一模型 $g_\Theta$ 参数化两个互补的条件分布：动作建模

$$(a_{t:t+p-1}, c_t) \sim g_\Theta(\cdot \mid o_t, s_t, l),$$

以及以预测出的动作条件信号 $c_t$ 为条件的未来视觉前馈动力学建模

$$(o_{t+\Delta}, o_{t+2\Delta}, \ldots, o_{t+K\Delta}) \sim g_\Theta(\cdot \mid o_t, s_t, l, c_t),$$

其中 $\Delta$ 是预测未来观测的时间步幅，$K=\lfloor p/\Delta \rfloor$。**大白话**：模型不是先"想清楚未来会发生什么画面"再决定动作，而是把动作预测放在主线，未来画面预测是围绕这条主线派生出来的"内部检验信号"，用来在训练时校正动作是否符合物理演化。

### 2.2 架构：因果扩散 Transformer

骨干是在 Wan 2.2（5B 参数扩散 Transformer，Wan et al. 2025）基础上适配的动作中心目标。

**多视角输入压缩**：为了在不修改骨干结构的前提下支持多视角生成，把左/前/右三路相机图像拼接成一张同分辨率的合成图：

$$o_t^{comp} = \mathrm{Compose}\big(o_t^{left}, o_t^{front}, o_t^{right}\big).$$

**共享 Transformer block**：当前观测 $o_t^{comp}$ 与预测的未来观测 $\{o_{t+k\Delta}^{comp}\}_{k=1}^K$ 用同一个预训练 VAE 编码，得到时空视觉 token $T_o$、$T_f$；本体感知状态和动作通过线性投影嵌入同一隐藏维度得到 $T_s$、$T_a$；语言指令由预训练语言编码器得到 $T_l$（以交叉注意力方式外部注入，不参与因果自注意力序列）。视觉 token 用 2D 位置编码，状态/动作 token 用 1D 时间位置编码。不同于 MoE/专家分离设计，所有 token 类型共享同一套 Q/K/V 投影矩阵。

**因果自注意力掩码**：把各模态 token 拼接为统一序列

$$T_t = [\,T_o;\ T_s;\ T_a;\ T_f\,],$$

并施加块级因果掩码（Fig.4），约束：(i) $T_s$、$T_o$ 互相可见，但看不到 $T_a$、$T_f$；(ii) $T_a$ 可见 $\{T_s,T_o\}$，但看不到 $T_f$；(iii) $T_f$ 可见 $\{T_s,T_o,T_a\}$。**大白话**：动作 token 只能"看现在"，不能偷看模型自己预测出来的未来画面（防止信息泄漏），而未来画面 token 则可以同时参考当前观测和刚生成的动作——这样保证了训练时未来画面预测是"给定动作之后会怎样"的诚实前馈预测，也保证了推理时可以合法地跳过未来画面 token 只解码动作。

### 2.3 训练目标：flow matching

对动作 token 或未来视频 token 中的任一模态 $x$，采样流时刻 $s\sim\mathcal U(0,1)$ 和噪声 $\epsilon\sim\mathcal N(0,I)$，构造插值噪声变量

$$x^{(s)} = (1-s)\epsilon + sx, \qquad \dot x^{(s)} = x-\epsilon.$$

视频侧（$z_f$ 为未来观测的 VAE 潜变量）：

$$\mathcal L_{video} = \mathbb E_{s,\epsilon}\Big[\big\| g_\Theta\big(z_f^{(s)}, s \mid T_s,T_o,T_a,T_l\big) - \dot z_f^{(s)} \big\|^2\Big].$$

动作侧：

$$\mathcal L_{action} = \mathbb E_{s,\epsilon}\Big[\big\| g_\Theta\big(a^{(s)}, s \mid T_s,T_o,T_l\big) - \dot a^{(s)} \big\|^2\Big].$$

预训练阶段只优化 $\mathcal L_{video}$（此时没有动作监督）；后训练阶段联合优化

$$\mathcal L_{all} = \lambda_{video}\mathcal L_{video} + \lambda_{action}\mathcal L_{action},$$

实现中取 $\lambda_{action}=5$、$\lambda_{video}=1$，即以动作预测为主、视频一致性为正则项。

### 2.4 三阶段课程训练

1. **视频基础初始化**：从大规模网络视频预训练的 Wan 2.2 5B 初始化，获得通用视觉动力学先验。
2. **具身数据预训练**：在真实机器人视频 + 大规模第一人称人类视频混合数据上继续做纯视频 flow-matching 预训练，共约 **10,000 小时**（Table 1：EgoDex 800h、Agibot 2,500h、EGO4D 3,500h、RoboMind 300h、RDT 25h、Open X-Embodiment 3,500h、DROID 350h、ATARA 10h、Something-Something V2 200h），目的是让表征适配机器人特有视角与操作交互模式。
3. **目标机器人后训练**：在目标机器人的图像-语言-动作三元组轨迹上后训练，联合 $\mathcal L_{video}$ 与 $\mathcal L_{action}$，专精到目标机器人的控制接口与状态分布。

### 2.5 推理：动作专用解码

推理时上下文为 $w_t=(T_l,T_s,T_o)$，只从学到的动作 flow 模型中采样动作 token：初始化 $a^{(0)}\sim\mathcal N(0,I)$，沿学到的速度场从 $s=0$ 积分到 $s=1$：

$$\frac{da^{(s)}}{ds} = g_\Theta\big(a^{(s)}, s \mid w_t\big), \qquad s\in[0,1],$$

得到 $a^{(1)}$ 后解码为连续动作块 $\hat a_{t:t+p-1}$ 执行，**完全不实例化未来视频 token**。若需要未来预测，可选启用视频分支：或将视频 token 与动作 token 一起联合去噪，或复用动作去噪时缓存的 KV Cache 再单独去噪视频 token。

**实现细节**：Wan 2.2 5B 骨干，动作块长度 $p=48$，未来观测采样步幅 $\Delta=12$（即 $K=4$ 帧）；预训练用 6,000 GPU 小时、全局 batch size 256、AdamW（$\beta_1=0.85$，$\beta_2=0.9$，原文如此），学习率从 $1\times10^{-4}$ 余弦衰减到 $1\times10^{-6}$；仿真每任务评测 100 条 episode，真机每任务 20 次试验、每次最多 5 次尝试。

## 三、实验结果

**对比基线**：VLM-based VLA —— π0.5（Intelligence et al. 2025）、GigaBrain-0（Team et al. 2025，同一实验室的姊妹工作）、X-VLA（Zheng et al. 2025）；WAM 类 —— Motus（Bi et al. 2025，MoT 架构 + UniDiffuser 式调度器）、Cosmos-Policy（Kim et al. 2026，视频模型后训练直接生成动作/未来图像/价值估计）。

**表 1：RoboTwin 2.0 仿真（50 个任务，Clean / 域随机 Rand 两种场景，Table 2/8 平均）**

| 方法 | Clean | Rand |
|---|---|---|
| π0.5 | 0.43 | 0.44 |
| X-VLA | 0.73 | 0.73 |
| Motus | **0.89** | **0.87** |
| GigaWorld-Policy（本文） | 0.86（第二） | 0.85（第二） |

即在仿真上 GigaWorld-Policy 略逊于 Motus（约 2-3 个百分点），但比 π0.5 高约 43pp（原文表述为"over 44 percentage points"）。

**表 2：A100 推理延迟与成功率（Table 3）**

| 方法 | 类型 | 延迟 (ms) | 仿真 SR | 真机 SR |
|---|---|---|---|---|
| π0.5 | VLA | 225 | 0.48 | 0.69 |
| GigaBrain-0 | VLA | 452 | – | 0.68 |
| Motus | WAM | 3231 | **0.88** | 0.76（第二） |
| Cosmos-Policy | WAM | 1413 | – | 0.58 |
| GigaWorld-Policy | WAM | **360** | 0.86（第二） | **0.83** |

相比 Motus，GigaWorld-Policy 推理延迟降至约 **1/9**（3231ms→360ms），仿真成功率相当，真机成功率反而更高（+7pp）——论文将其归因于低延迟带来更高的有效控制频率与更强的闭环误差修正能力。

**表 3：真机四任务（AgileX PiPER 6-DoF，Table 4，各 20 次试验）**

| 方法 | Clean the Desk | Scan a QR Code | Sweep up Trash | Stack Bowls | 平均 |
|---|---|---|---|---|---|
| π0.5 | 0.75 | 0.55 | 0.65 | 0.80 | 0.69 |
| GigaBrain-0 | 0.70 | 0.65 | 0.60 | 0.75 | 0.68 |
| Motus | 0.80 | **0.75** | 0.70 | 0.80 | 0.76 |
| Cosmos-Policy | 0.65 | 0.50 | 0.45 | 0.70 | 0.58 |
| GigaWorld-Policy | **0.90** | **0.75** | **0.75** | **0.90** | **0.83** |

四个真机任务全部最佳或并列最佳，平均成功率比最强 VLA 基线 π0.5 高约 14pp，比最强 WAM 基线 Motus 高约 7pp。

**数据效率（Fig.7）**：5/25/50 条演示下，GigaWorld-Policy 真机成功率分别为 69%/78%/83%，π0.5 为 46%/58%/69%；仅用 10% 训练数据即达到 π0.5 用满量数据的成功率。

**具身预训练数据量消融（Fig.8）**：真机成功率随具身预训练数据占比单调提升：0%→57%，10%→66%，20%→71%，40%→75%，80%→78%，100%→83%。

**未来帧预测密度消融（Table 5，固定动作块长度 48，调节 $\Delta$）**：

| $\Delta$ | 0（无视频预测） | 4 | 8 | 12 | 24 | 48 |
|---|---|---|---|---|---|---|
| 真机 SR | 0.60 | 0.76 | 0.78 | **0.83** | 0.80 | 0.76 |

$\Delta=0$（完全不做未来视频预测，退化为纯动作解码器）明显最差（0.60），说明前馈动力学建模确有增益；但预测过密（$\Delta=4$）或过稀（$\Delta=48$）都不如中等密度（$\Delta=12$），存在收益递减甚至轻微下降。

**因果自注意力消融（Table 6）**：与"无约束全注意力"变体相比，因果掩码版本 SR 相近（0.83 vs 0.81）但视频生成质量更高（PSNR 28.41 vs 27.87，SSIM 0.901 vs 0.892），且因果掩码是使"推理期视频预测可选"这一关键工程特性成立的前提。

**预训练配置消融（Table 7）**：从零训练 SR 0.45；仅视频预训练初始化（无具身数据阶段）SR 0.57；仅具身数据预训练（无视频模型初始化）SR 0.73；两者结合 SR 0.83（最佳），说明两阶段预训练互补而非冗余。

## 四、局限性

论文正文没有单独的 Limitations 小节，以下基于其自身实验数据与设计选择归纳：

1. **仿真成绩仍非最优**：在 RoboTwin 2.0 上 GigaWorld-Policy 的成功率（Clean 0.86 / Rand 0.85）仍低于 Motus（0.89 / 0.87），本质是用少量仿真成功率换取约 9 倍推理加速，并非在所有指标上全面超越现有 WAM。
2. **未来预测密度需要调参**：Table 5 显示 $\Delta$（预测步幅/帧数）存在一个非单调的最优点，过密或过稀都会掉点，说明这不是一个"越多越好"的自由增益，需要针对具体动作块长度和任务做调试。
3. **真机验证范围有限**：真机实验仅在单一平台（AgileX PiPER，单臂、夹爪末端）上做了 4 个任务，未展示双臂、移动操作或跨本体的真机泛化结果；预训练阶段虽然混合了多种机器人和人类视频来源，但下游真正部署验证的本体单一。
4. **多视角拼接是架构妥协**：为了不修改骨干结构，把左/前/右三视角图像拼成一张合成图（Eq.4）输入统一 VAE，这是效率导向的工程折中，论文未单独消融这种拼接相较于原生多视角/多路 token 输入在跨视角一致性上的信息损失。
5. **预训练成本仍然可观**：5B 参数骨干 + 6,000 GPU 小时的预训练规模，对资源有限的团队复现门槛较高，属于工业级实验室的典型规模。
6. **评测协议宽松**：真机实验允许每次试验最多 5 次尝试才算完成，比严格的单次尝试评测更宽松，跨论文横向比较成功率时需注意协议口径差异（不过论文中所有基线都在同一协议下测试，内部比较仍公平）。

## 五、评价与展望

GigaWorld-Policy 的核心贡献在于把"世界模型式稠密监督"和"低延迟部署"两个通常互斥的目标，用一个架构选择（因果注意力掩码 + flow matching 统一动作/视频建模）同时兼顾：训练时让视觉动力学作为正则信号稠密监督动作学习，推理时通过因果掩码保证的信息隔离性，把视频预测整体砍掉而不破坏动作解码的正确性。这与同期两阶段 WAM（Mimic-video 先生成视频再用逆动力学模型解码动作；LingBot-VA 显式滚动预测世界状态）和联合双向注意力 WAM（Motus 的 MoT 架构、VideoVLA 的多模态扩散 Transformer）形成明确对比：后两类范式在推理期都需要真正走一遍（部分）视频生成流程，因此延迟天然是动作块解码延迟的数倍到数十倍（如 Motus 的 3231ms vs 本文 360ms）。这个"训练稠密监督、推理可选丢弃"的设计思路具有较好的通用性，理论上可以移植到其他基于扩散/flow 的 WAM 架构上。

与同实验室的 GigaBrain-0（用世界模型生成数据增广 VLA 训练）相比，GigaWorld-Policy 走的是另一条路线——不生成新数据，而是把世界模型能力直接内嵌为架构级辅助监督，两者可以视为"数据增广 vs 架构耦合"两种利用世界模型的互补范式，二者原则上也可叠加（用 GigaBrain-0 式数据增广去训练一个 GigaWorld-Policy 式架构）。

开放问题与可能的改进方向：
- **未来预测密度的自适应选择**：当前 $\Delta$/$K$ 是手工网格搜索得到的固定超参数（Table 5），一个开放问题是能否让模型根据任务动力学复杂度自适应地决定预测密度，而不是全局固定。
- **多视角信息融合的原生化**：当前的拼图式多视角输入（Eq.4）是不改架构的权宜之计，若未来放开对预训练骨干结构的约束，原生的多视角/多相机 token 设计可能进一步提升跨视角一致性和空间推理能力。
- **在仿真上反超 Motus 的空间**：仿真成功率仍有约 2-3pp 的差距，值得研究是否可以在不牺牲低延迟特性的前提下，通过更长的具身预训练或更精细的课程设计来缩小甚至反超这一差距。
- **跨本体真机泛化**：论文的具身预训练数据本身覆盖多种人类/机器人视频来源，但真机验证只在单一本体单一末端执行器上完成；后续工作可以补充双臂、移动底盘或不同末端执行器下的零样本/少样本迁移评测，以检验其架构设计对本体差异的鲁棒性是否也能带来类似 Motus、π0.5 等跨本体基础模型的迁移收益。

## 参考

- Bi, Hongzhe, et al. "Motus: A unified latent action world model." arXiv:2512.13030, 2025.
- Kim, Moo Jin, et al. "Cosmos-policy: Fine-tuning video models for visuomotor control and planning." arXiv:2601.16163, 2026.
- Physical Intelligence, et al. "π0.5: A vision-language-action model with open-world generalization." arXiv:2504.16054, 2025.
- Pai, Jonas, et al. "Mimic-video: Video-action models for generalizable robot control beyond VLAs." arXiv:2512.15692, 2025.
- Shen, Yichao, et al. "VideoVLA: Video generators can be generalizable robot manipulators." arXiv:2512.06963, 2025.
