# Next Forcing：基于多块预测的因果世界建模

> **论文**：*Next Forcing: Causal World Modeling with Multi-Chunk Prediction*
>
> **作者**：Gangwei Xu、Qihang Zhang、Jiaming Zhou、Xing Zhu、Yujun Shen、Xin Yang、Yinghao Xu 等（Qihang Zhang 为 Project Lead，Xin Yang、Yinghao Xu 为通讯作者）
>
> **机构**：Robbyant；HUST（华中科技大学）；HKUST（香港科技大学）；HKUST(GZ)（香港科技大学广州）
>
> **发布时间**：2026 年 06 月（arXiv 2606.11187）
>
> **发表状态**：未录用（预印本，首页标注 "Preprint"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.11187) | [PDF](https://arxiv.org/pdf/2606.11187)
>
> **分类标签**：`World Action Model` `Multi-Chunk Prediction` `自回归视频生成` `Flow Matching` `双臂操作` `RoboTwin`

---

## 一句话总结

针对自回归 World Action Model 中 teacher-forced next-chunk denoising 存在的"外观捷径"（相邻 chunk 视觉相似导致模型走捷径而不学真实动力学）问题，作者提出 **Next Forcing**：借鉴 LLM 的 multi-token prediction，给主模型挂载三个轻量、链式的 MCP（Multi-Chunk Prediction）辅助模块，同时监督 next¹/next²/next³ 未来 chunk，在 RoboTwin 上取得 94.1/93.5（Clean/Random）新 SOTA，50fps 高帧率下 5k 步即相对 LingBot-VA 提升 93.1%（Random 设置）、达到 2.3× 训练收敛加速，推理时保留 MCP 模块还可再获 2× 加速。

## 一、问题与动机

World Action Model（WAM）范式先预测未来视觉动态、再从预测帧解码动作。当前主流训练目标是 **teacher-forced next-chunk denoising**：模型在干净历史 chunk 条件下对当前 chunk 去噪（对应论文 Eq. 3，建立在 LingBot-VA 框架之上）。这一目标虽稳定，但作者指出它本质上是**局部（local）任务**，存在"外观捷径"（appearance shortcut）：相邻 chunk 视觉高度相似，模型只需学一个近似恒等映射加小幅残差修正就能把去噪 loss 压得很低，无需真正捕捉支配场景演化的长程动力学。作者称这一现象为 **myopic supervision（短视监督）**。

该问题在**高帧率**下尤为致命：以 50fps 为例，相邻 chunk 之间的外观差异窄到捷径几乎无损，标准 teacher forcing 收敛明显更慢、最终精度更低（对应 Figure 1 的实证曲线）。

核心洞察：把局部的"预测下一个"目标换成长程的"多 chunk 预测"目标，能强迫模型学习支配时序演化的潜在动力学而非依赖外观捷径——这一思路在语言模型中已被 multi-token prediction（MTP，如 DeepSeek-V3、Gloeckle et al. 2024）验证有效。但把 MTP 从离散 token 搬到连续视频世界模型并不 trivial：(1) 预测目标是连续视频 latent 而非离散 token；(2) 生成通过多步迭代去噪而非单步采样；(3) 时序依赖跨越多个不同尺度的 horizon。Next Forcing 正是针对这三点做的适配性设计。

论文同时指出自己与另一族 "forcing" 方法（Diffusion Forcing、Self Forcing）的关系：后两者分别改变"如何调度噪声"和"模型在训练时看到什么上下文"，而 Next Forcing 改变的是**"模型被要求预测什么"**，三者是不同的正交轴，理论上可组合。

## 二、核心方法

### 2.1 预备知识：Flow Matching 与 Teacher Forcing

Flow matching 学习一个速度场，把噪声分布搬运到数据分布。给定干净样本 $\mathbf{x}_0$ 和高斯噪声 $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})$，噪声样本按线性插值构造：

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon}$$

网络 $v_\theta(\mathbf{x}_t,t,\mathbf{c})$ 学习预测速度 $\mathbf{v}^*=\boldsymbol{\epsilon}-\mathbf{x}_0$：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|v_\theta(\mathbf{x}_t,t,\mathbf{c})-(\boldsymbol{\epsilon}-\mathbf{x}_0)\|^2\right]$$

**大白话**：训练时把干净视频和纯噪声按比例 $t$ 混合，让网络学会"往回走"的方向，推理时从纯噪声沿这个方向积分回干净样本。

Next Forcing 建立在 LingBot-VA 的自回归 teacher forcing 之上：在自回归步 $i$，模型在干净历史 chunk $\mathbf{x}_0^{(1:i-1)}$ 和语言指令 $\ell$ 条件下去噪当前 chunk $\mathbf{x}_t^{(i)}$，这一条件设定天然贴合闭环部署（每步执行动作后用真实观测替换生成帧），但监督信号被限制在当前 chunk，即前述的短视问题。

### 2.2 Multi-Chunk Prediction（MCP）目标

**时间平移构造多深度目标**：给定训练视频 latent $\mathbf{x}_0\in\mathbb{R}^{C\times F\times H\times W}$（$F$ 为 chunk 数，每 chunk 含 $M$ 帧，$M$ 在 $\{1,\dots,M_{\max}\}$ 中随机采样以增强多尺度鲁棒性）。对 MCP 深度 $k\in\{1,2,3\}$，构造把视频 latent 沿时间前移 $k$ 个 chunk 的目标：

$$\mathbf{x}_0^{[k]}[i] = \mathbf{x}_0[\min(i+k,F)]$$

超出序列边界的 chunk 用最后一 chunk 复制填充。**大白话**：depth-$k$ 的 MCP 模块要预测的不是"当前"而是"未来第 $k$ 个" chunk。

**独立加噪，且噪声水平更高**：每个平移目标用各自的 timestep 独立加噪：

$$\mathbf{x}_{t_k}^{[k]} = (1-t_k)\mathbf{x}_0^{[k]} + t_k\boldsymbol{\epsilon}_k,\quad \boldsymbol{\epsilon}_k\sim\mathcal{N}(0,\mathbf{I})$$

其中 $t_k$ 的 timestep-shift 参数 $s_{\text{mcp}}$（=10）显著大于主模型的 $s_{\text{main}}$（=5）。**大白话**：故意把 MCP 的输入弄得更"糊"（更高噪声），这样 MCP 输入自身携带的信息更少，模块被迫更依赖主模型传来的表示去解噪，从而把 MCP 的监督梯度"逼"进主模型内部，而不是被轻量 MCP 模块自己就地吸收掉。（附录 C 给出 timestep-shift 的具体形式 $\tilde\sigma_i = s\sigma_i/(1+(s-1)\sigma_i)$，$s$ 越大噪声分布越往高噪声端偏移。）

**位置编码告知时间偏移**：把 chunk 偏移量编码进 RoPE：

$$\mathrm{RoPE}(\mathbf{x}_0^{[k]}[i]) = \mathrm{RoPE}(i+k)$$

使每个 MCP 模块清楚自己预测的是未来第几个 chunk。

### 2.3 多层特征融合 + 链式 MCP 模块

**大白话直觉**：主模型浅层学粗粒度结构、深层学精细细节，只用最后一层给 MCP 提供信息会丢掉浅层信息；同时希望 MCP 的监督梯度能反传进主模型的多个深度而不只是顶层。

主模型 30 层 Transformer 中，在 $\{4,12,20,30\}$ 四个中间层收集隐藏状态（同时含噪声当前 latent 与干净历史 latent 的表示），拼接后经两层 MLP 压缩：

$$\mathbf{h}_{\text{fuse}} = \mathrm{MLP}([\mathbf{h}_4;\mathbf{h}_{12};\mathbf{h}_{20};\mathbf{h}_{30}])\in\mathbb{R}^{B\times N\times d}$$

三个 MCP 模块形成**因果链**：深度 $k$ 的模块把噪声平移目标的 patch embedding 与上一深度输出拼接、投影：

$$\mathbf{z}^{[k]} = W_k\left[\mathbf{h}_{\text{prev}}^{[k-1]};\mathrm{Embed}(\mathbf{x}_{t_k}^{[k]})\right],\quad W_k\in\mathbb{R}^{d\times 2d}$$

其中 $\mathbf{h}_{\text{prev}}^{[0]}=\mathbf{h}_{\text{fuse}}$。$\mathbf{z}^{[k]}$ 经过 3 层轻量 Transformer block 预测 flow matching 速度 $\hat{\mathbf{v}}^{[k]}$，同时该输出作为 $\mathbf{h}_{\text{prev}}^{[k]}$ 供 depth-($k$+1) 使用，即 depth-2 建立在 depth-1 特征之上，depth-3 建立在 depth-2 之上。三个 MCP 模块与主模型共享同一套注意力掩码（附录 A 给出细节：噪声 token 只对因果之前的干净 token 及同 chunk 内噪声 token 做注意力；干净 token 不可关注噪声 token），因此每训练步只需构造一次 mask，训练开销较低。

### 2.4 联合视频-动作架构与总损失

沿用 LingBot-VA 的双阶段分解：先预测未来视觉状态，再用逆动力学从（预测的）未来观测解码动作：

$$\mathbf{x}_{i+1}\sim p_\theta(\cdot\mid \mathbf{x}_{\le i},\mathbf{a}_{\lt i},\ell),\qquad \mathbf{a}_i\sim g_\psi(\cdot\mid \mathbf{x}_{\le i+1},\mathbf{a}_{\lt i},\ell)$$

视频流与动作流在统一的 Mixture-of-Transformers（MoT）架构里通过跨模态注意力逐层融合；MCP 模块只作用于视频流，改进后的视频表示通过共享的跨模态注意力间接惠及动作解码。

主损失为视频与动作两条 flow matching 损失，MCP 损失对每个深度 $k$ 计算（末尾被 padding 覆盖的 chunk 排除在损失外）：

$$\mathcal{L}_k^{\text{MCP}} = \mathbb{E}_{t_k,\mathbf{x}_0^{[k]},\boldsymbol{\epsilon}_k}\left[\left\|v_\theta^{[k]}(\mathbf{x}_{t_k}^{[k]},t_k,\mathbf{c})-(\boldsymbol{\epsilon}_k-\mathbf{x}_0^{[k]})\right\|^2\right]$$

总损失（$w_1=0.5,w_2=0.2,w_3=0.1$，深度越远权重越小）：

$$\mathcal{L} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{action}} + \sum_{k=1}^{3} w_k\cdot\mathcal{L}_k^{\text{MCP}}$$

### 2.5 推理：两种可自由切换的模式

同一训练好的 checkpoint 支持两种部署模式，无需重训：

- **零开销模式（Zero-Overhead）**：丢弃全部 MCP 模块（融合 MLP、投影层、轻量 Transformer block），主模型按标准自回归流水线运行，延迟/显存与 baseline 完全一致。所有质量提升都来自训练阶段 MCP 目标反哺进主模型权重的效果，推理端零代价。
- **并行 Chunk 生成模式（Parallel Chunk Generation）**：保留 depth-1 MCP 模块，在主模型对当前 chunk 去噪的同一次前向中，depth-1 模块并行产出下一 chunk 的预测。因 MCP 的 Transformer block 比主模型轻一个数量级，几乎不增加前向开销，每个自回归步能同时推进两个 chunk，达到 **2× 推理加速**。depth-2/3 的预测在下一步会被主模型的新预测覆盖而未被使用；作者指出把该机制扩展到更高深度可换取更高加速比，但会引入误差累积（drift），留作未来工作。

## 三、实验结果

**评测基准**：RoboTwin（50 个双臂操作任务，Clean=固定初始配置 / Random=随机化物体位姿与场景布局，报告 50 任务平均成功率）；PhyWorld（纯视频生成的物理规律遵循度评测，指标 FVD 与 Abnormal Ratio）。**实现细节**：基于 LingBot-VA 框架 + Wan2.2 Transformer 主干（30 层），$s_{\text{main}}=5$（历史噪声增强概率 0.5），MCP 三深度各 3 个 Transformer block，$s_{\text{mcp}}=10$，$M_{\max}=4$；先在大规模多本体数据集预训练，再在 RoboTwin 上后训练（2,500 条 Clean 演示 + 25,000 条 Random 演示，最多 50k 步，64 GPU，多帧率对比）；消融实验用 16 GPU、仅 2,500 条 Clean 演示、25fps、20k 步。

**表 1：RoboTwin 主结果对比（50 任务平均成功率 %）**

| 方法 | Clean | Random |
|---|---|---|
| X-VLA | 72.9 | 72.8 |
| $\pi_0$ | 65.9 | 58.4 |
| $\pi_{0.5}$ | 82.7 | 76.8 |
| Motus | 88.7 | 87.0 |
| Being-H0.7 | 90.2 | 89.6 |
| Fast-WAM | 91.9 | 91.8 |
| LingBot-VA | 92.9 | 91.5 |
| **Next Forcing** | **94.1** | **93.5** |

**表 2：消融研究（RoboTwin Clean 子集，20k 步，16 GPU）**

| Baseline（LingBot-VA）设计消融 | SR(%) | MCP 模块消融（基线=Baseline+MCP 默认 85.8%） | SR(%) |
|---|---|---|---|
| Baseline 默认（$s_{\text{main}}=5$+噪声历史增强） | 75.6 | Baseline + MCP（默认配置） | 85.8 |
| $s_{\text{main}}=1$ | 65.3 | $s_{\text{mcp}}=5$（等于主模型） | 83.2 |
| $s_{\text{main}}=10$ | 78.4 | 去掉多层特征融合 | 83.6 |
| $s_{\text{main}}=20$ | 77.6 | 去掉权重初始化（主模型末几层） | 83.8 |
| $s_{\text{main}}=25$ | 77.2 | Transformer block=1 | 86.5 |
| 去掉噪声历史增强 | 69.8 | Transformer block=5 | 85.0 |

MCP 使 baseline 从 75.6% 提升到 85.8%（+10.2 点）。作者默认用 3-block（而非略高的 1-block 86.5%），因为 3-block 在并行生成模式下产生更少视觉伪影。

**训练收敛（详见附录表 5，摘录关键点）**：

| FPS | 设置 | 方法 | 5k 步 | 20k 步 | 50k 步 |
|---|---|---|---|---|---|
| 12 | Clean | LingBot-VA / Next Forcing | 74.0 / 84.9 | 90.8 / 92.3 | 92.8 / 94.1 |
| 12 | Random | LingBot-VA / Next Forcing | 73.5 / 80.6 | 88.3 / 90.5 | 91.5 / 93.5 |
| 50 | Clean | LingBot-VA / Next Forcing | 45.5 / 70.2 | 78.5 / 87.4 | 88.6 / 91.8 |
| 50 | Random | LingBot-VA / Next Forcing | 31.9 / 61.6 | 69.4 / 85.0 | 85.2 / 90.5 |

50fps/Random 下 5k 步相对提升 $(61.6-31.9)/31.9=93.1\%$（abstract 重点强调的数字），Clean 下相对提升为 $(70.2-45.5)/45.5=54.3\%$。Next Forcing 在 50fps/Random 仅 20k 步（85.0）即达到 LingBot-VA 45k 步的精度（84.5），对应约 **2.3× 训练收敛加速**。

**表 3：PhyWorld 组合泛化评测**（OOT=out-of-template，IT=in-template；去掉动作流，只评视频生成部分）

| 方法 | FVD-OOT↓ | FVD-IT↓ | Abnormal-OOT↓ | Abnormal-IT↓ |
|---|---|---|---|---|
| LingBot-VA | 5.3 | 3.5 | 12% | 3% |
| **Next Forcing** | **4.7** | **3.2** | **8%** | **2%** |

OOT 上增益更大，说明 MCP 促进的是可泛化的物理动力学理解，而非模板记忆。

**表 4：推理加速（不同帧率下的成功率对比）**

| 推理模式 | 12fps Clean/Random | 25fps Clean/Random | 50fps Clean/Random |
|---|---|---|---|
| 标准（逐 chunk 串行） | 94.1 / 93.5 | 92.6 / 91.4 | 91.8 / 90.5 |
| MCP 加速（2×，并行 chunk） | 93.5 / 90.6 | 91.0 / 89.8 | 92.2 / 91.3 |

2× 加速下精度基本持平（个别设置甚至更高），说明 depth-1 MCP 的预测质量已足够支撑并行解码。

**通用视频预训练泛化性**：在约 350 万段（5-10 秒）以人类活动为主的自建视频数据集上预训练（去掉动作流，32 GPU），两个各 1,024 样本的held-out 测试集上评 FVD。50k 步时 Test Set 1（人类活动）FVD 从 225 降到 94（**降低 58%**），Test Set 2（相机驱动场景动态）从 204 降到 97（**降低 52%**）；且 Next Forcing 仅 10k 步就已超过 LingBot-VA 训练满 50k 步的水平，说明该训练目标对一般视频世界模型同样有效，不局限于机器人操作数据。

## 四、局限性

论文结论部分明确写出的唯一 limitation：**MCP 模块引入额外的训练开销**（The main limitation is that the MCP modules introduce extra training cost）。

此外从方法与实验细节中可读出的隐含局限：

1. **并行加速仅利用 depth-1**：并行生成模式只启用了深度最浅的 MCP 模块，depth-2/3 的预测在下一步会被主模型新预测覆盖而浪费；若要外推到更高深度换取更大加速，会引入误差累积（drift），作者明确留给未来工作，未给出解决方案或量化的 drift 影响。
2. **超参数较敏感、需要针对性调试**：消融显示 $s_{\text{mcp}}$、多层融合层选择 $\{4,12,20,30\}$、MCP 权重初始化方式、损失权重 $w_k$ 等都对最终精度有明显影响（例如去掉多层融合从 85.8% 掉到 83.6%，$s_{\text{mcp}}=5$ 时掉到 83.2%），这些选择目前依赖经验消融而非理论指导。
3. **未量化训练开销的具体数字**：论文只定性描述 MCP 模块"lightweight"（比主模型轻一个数量级），但未报告增加的训练 FLOPs、显存或 wall-clock 时间百分比，读者无法评估"更快收敛"相对"每步更贵"两者的净收益。
4. **强依赖 base WAM 框架**：整套方法建立在 LingBot-VA（Wan2.2 因果 WAM 框架）之上，是训练目标层面的增量式改进；虽然 Related Work 中声称与 Diffusion Forcing/Self Forcing"正交、可组合"，但论文未做三者叠加或在其他 WAM 骨干（如 DreamZero、Fast-WAM）上的交叉验证实验，正交性论断缺少实证支持。
5. **对比基线单一**：PhyWorld 和通用视频预训练两组实验都只与 LingBot-VA 一个 baseline 对比，未与其他训练范式（Diffusion Forcing、Self Forcing 等）在同一 backbone 上做直接横向比较。

## 五、评价与展望

**优点**：思路简洁且有清晰的类比直觉——把 LLM 领域已验证有效的 multi-token prediction 迁移到连续视频 latent 上，并针对性解决了"离散 token→连续 latent"迁移中的三个关键工程问题（时间平移目标构造、独立高噪声注入以强化对主干表示的依赖、多层特征融合以让梯度深入主干各层）。更难得的是同时改善了**训练收敛/精度**与**推理速度**两个通常存在权衡的目标，且零开销模式下部署零代价，工程可用性强。系统性消融覆盖了 baseline 自身的噪声调度超参和 MCP 自身设计的各个组件，可复现性较好；额外在纯视频（非机器人）预训练上验证了训练目标的通用性（FVD 降低 50%+），把贡献的适用范围从机器人操作扩展到更广的视频世界模型训练。

**与已有工作的关系**：与 Diffusion Forcing、Self Forcing 同属 "forcing" 方法家族但作用于不同轴——后两者改变噪声调度方式或训练时的上下文构造以弥合 train-test gap，Next Forcing 改变的是监督目标本身（预测什么），三者理论上正交、可叠加，但论文未给出叠加实验。与 Gloeckle et al. 的 multi-token prediction 相比，本文的核心改造点（时间平移替代 token 位移、非对称加噪强化耦合、多层融合）是对"MTP 如何迁移到连续 diffusion/flow matching 视频世界模型"这一跨模态问题的具体工程回答，而非简单套用。

**开放问题与可能方向**：(1) MCP 深度目前只到 3，更大预测 horizon（如 5-10 chunk）对收敛增益是否边际递减、能否支撑更激进的并行解码加速比；(2) 深度 2/3 MCP 目前只用于训练监督、推理时被浪费，如何设计更充分利用它们（例如推测解码式的验证-重试机制）以在维持精度的同时把加速比推高于 2×；(3) 高帧率增益显著，但论文未展示低于 12fps 的数据点，MCP 在低帧率（外观捷径本就不明显）场景下的边际价值仍待验证；(4) 缺少与 Diffusion Forcing/Self Forcing 在同一 backbone 上的直接横向对比及正交性叠加实验，是验证该训练目标普适性的自然下一步。

## 参考

1. Li et al. *LingBot-VA: Causal world modeling for robot control.* arXiv:2601.21998, 2026. —— Next Forcing 所基于的 base WAM 框架与主要对比 baseline。
2. Chen et al. *Diffusion Forcing: Next-token prediction meets full-sequence diffusion.* NeurIPS 2024. —— 同属 "forcing" 方法家族，作用于噪声调度轴。
3. Huang et al. *Self Forcing: Bridging the train-test gap in autoregressive video diffusion.* NeurIPS 2025. —— 同属 "forcing" 方法家族，作用于上下文构造轴。
4. Gloeckle et al. *Better & Faster Large Language Models via Multi-Token Prediction.* ICML 2024. —— MCP 训练目标的语言模型灵感来源。
5. Chen et al. *RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation.* arXiv:2506.18088, 2025. —— 论文主要评测基准。
