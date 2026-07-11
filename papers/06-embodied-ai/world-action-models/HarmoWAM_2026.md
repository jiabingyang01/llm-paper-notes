# HarmoWAM：以自适应世界-动作模型统一泛化转移与精确操作

> **论文**：*HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models*
>
> **作者**：Qiuxuan Feng*、Jiale Yu*、Jiaming Liu*†、Yueru Jia* 等（*为共同一作，†项目负责人）,通讯作者 Shanghang Zhang
>
> **机构**：Peking University（State Key Laboratory of Multimedia Information Processing, School of Computer Science）、Simplexity Robotics、The Chinese University of Hong Kong
>
> **发布时间**：2026 年 05 月（arXiv 2605.10942）
>
> **发表状态**：未录用（预印本，作者自标 "Preprint"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.10942) | [PDF](https://arxiv.org/pdf/2605.10942)
>
> **分类标签**：`世界动作模型` `视频扩散` `双专家架构` `自适应门控` `真机操作`

---

## 一句话总结

论文用真机实验实证了世界动作模型（WAM）两大范式"先想象后执行"（转移准、交互差）与"联合建模"（交互准、转移差）之间的互补性缺陷，进而提出 HarmoWAM：同一个视频扩散世界模型同时驱动一个 predictive expert（精细交互）和一个 reactive expert（泛化转移），并用一个基于关键帧标签训练的轻量 MLP 门控网络实时决定二者的切换,在 6 项真机任务上 ID 平均成功率 89%（比领先 VLA/WAM 分别高 15/11 个百分点）,OOD（背景/位置/物体三类）平均 82%（比领先 VLA/WAM 分别高 33/29 个百分点,相对 ID 仅下降 7.9%）。

## 一、问题与动机

World Action Models（WAM）通过预测未来视觉观测为动作生成提供时空物理先验,现有工作可分两类范式：

- **Imagine-then-Execute**（先想象后执行）：先用视频生成模型预测未来帧,再用逆动力学模型（IDM）从预测帧对中反推动作,如 DreamGen、WoW。
- **Joint Modeling**（联合建模）：直接联合建模动作与视频的联合分布,把世界模型的隐特征作为条件或监督信号,如 VPP、UVA、UWM、Cosmos-Policy、Motus。

作者在两项代表性真机任务（*Put Flowers in Vase* 长时程双臂协同、*Stack Coke Cans* 精细堆叠）上系统对比了两种范式的代表实现（均以 Wan2.2-TI2V-5B 为骨干：Imagine-then-Execute 用 AnyPos 做 IDM,Joint Modeling 用 Action DiT 做联合条件建模,遵循 VPP 的做法）,将每次执行拆分为 **transit（转移到物体附近）** 和 **interaction（抓取/堆叠/递送/插入等精细交互）** 两阶段分别打分,在 ID 与三类 OOD（背景/位置/物体语义）下评测（Table 1）。核心发现（论文 Motivation 一节的"定音锤"实验）：

- Imagine-then-Execute 的 transit 阶段近乎完美（即使 OOD 也常年 10/10）,但 interaction 阶段成功率 ID 平均低于 75%、OOD 低于 55%,精细操作精度不足。
- Joint Modeling 的 ID 综合成功率超过 90%,但 OOD 下 transit 显著崩溃（低至 32%）,反映其探索空间被 SFT 训练数据分布约束；然而只要把机器人初始化在目标物体附近,OOD 下 interaction 成功率仍能达到 95%,说明其短板是"够不着目标"而非"操作不精确"。

这揭示了一个结构性权衡：**先想象后执行擅长可泛化的转移但缺乏操作精度,联合建模具备高精度但难以在陌生环境中可靠地探索并接近目标**。HarmoWAM 的动机正是设计一个能同时吃到两种范式优势的统一框架。

## 二、核心方法

### 2.1 整体架构

HarmoWAM 由三部分组成（Figure 2）：一个共享的视频扩散世界模型（基于 Wan2.2-TI2V-5B,并在约 190 万条机器人轨迹上做进一步预训练,数据含公开的 DROID（201,119 条）、AgiBot World Colosseo（3,017 条）、RoboMIND（1,721,985 条）以及闭源机器人数据,分辨率 256×320,预测 13 帧未来视频）,以及两个互补的动作专家：

- **Predictive expert**（1B 参数 DiT,28 个 Transformer block）：以 SigLIP 图像特征 $\mathcal{F}_t^{img}$、文本特征 $\mathcal{F}^{text}$（T5 文本编码器）为条件,通过 cross-attention 融合世界模型当前步的隐特征 $\mathcal{F}_{t:t+H}^{V} \in \mathbb{R}^{B\times 80\times 3072}$ 作为额外条件,以扩散去噪方式生成结构化动作序列,负责精细操作阶段。
- **Reactive expert**（DINOv2-base + Orientation Decoder $\mathcal{D}_{ori}$,多尺度卷积层,沿用 AnyPos 的设计）：对每个未来时刻 $s$,取世界模型预测帧 $\mathbf{V}_s$ 及其隐特征 $\mathcal{F}_s^{V}$,用 DINOv2 提取 patch 级几何特征 $\mathcal{F}_s^{patch} \in \mathbb{R}^{B\times 1369\times 768}$,并将 $\mathcal{F}_s^{V}$ 沿通道维平均池化（3072→768）后与 $\mathcal{F}_s^{patch}$ 拼接得到融合表征 $\mathcal{F}_s^{fuse} = [\mathcal{F}_s^{patch}; \mathcal{F}_s^{V}]$,再由 $\mathcal{D}_{ori}$ 解码出动作 $\hat{\mathbf{a}}_s = \mathcal{D}_{ori}(\mathcal{F}_s^{fuse})$。与传统只依赖观测视频帧的 IDM 不同,reactive expert 额外注入了世界模型的隐动态信息,负责泛化转移阶段。

两专家共享同一个世界模型给出的显式（预测视频帧）与隐式（latent 特征）时空条件,分别对应两条推理通路：predictive expert 直接吃当前步 latent 条件做扩散去噪生成动作,reactive expert 则基于未来预测帧+其 latent 做类 IDM 推理,但注入了高层时空知识而不仅是低层视觉线索。

### 2.2 Process-Adaptive Gating（过程自适应门控）

门控网络是一个轻量 MLP 分类器,复用当前观测的视觉 token $\mathcal{F}_t^{img}$ 作为输入,输出置信度 $s_t \in [0,1]$,表示当前处于交互阶段的概率。训练标签通过关键帧抽取流程（沿用 Perceiver-Actor 的做法）自动构造：依据机器人本体感知信号（夹爪状态切换、任务相关的末端高度阈值）识别关键交互事件,并将事件前后各 20 帧标注为 interaction 段（$y=1$）,其余标注为 transit 段（$y=0$）。门控网络用二元交叉熵训练：

$$
\mathcal{L}_{gate} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(s_i) + (1-y_i)\log(1-s_i)\right] \tag{1}
$$

**用大白话说**：门控网络学的是一个"现在该精雕细琢还是该大步流星"的二分类器,只要看当前这一帧图像的视觉 token,就能判断机器人是在"赶路"（转移）还是在"动手"（交互）。

推理时以阈值 0.5 切换：$s_t > 0.5$ 时路由给 predictive expert 做精确交互,$s_t \le 0.5$ 时路由给 reactive expert 扩大探索。离线在 1,637 个 held-out 测试帧对上评测,门控帧级分类准确率达 **96.95%**。

### 2.3 两阶段训练

**Stage 1（世界模型微调）**：用条件 Flow Matching 目标在真机演示数据上全参数微调世界模型。给定干净视频 latent $\mathbf{x}_1$、高斯噪声 $\mathbf{x}_0\sim\mathcal{N}(0,I)$、插值变量 $\xi\in[0,1]$,构造 $\mathbf{x}_\xi=(1-\xi)\mathbf{x}_0+\xi\mathbf{x}_1$,目标速度 $\mathbf{v}_\xi = \mathrm{d}\mathbf{x}_\xi/\mathrm{d}\xi = \mathbf{x}_1-\mathbf{x}_0$,训练损失为：

$$
\mathcal{L}_{stage1} = \mathbb{E}_{\mathbf{x}_0,\mathbf{x}_1,\mathbf{c}}\left[w(\xi)\left\|f_\theta(\mathbf{x}_\xi,\xi,\mathbf{c}) - \mathbf{v}_\xi\right\|_2^2\right] \tag{2}
$$

**用大白话说**：让世界模型学会沿着噪声到真实视频的最短"流动路径"走,权重 $w(\xi)$ 只是按插值步数调节不同阶段样本的重要性。

**Stage 2（动作专家微调,冻结世界模型）**：predictive expert 用标准扩散去噪损失（对加到动作序列上的高斯噪声做回归）,reactive expert 用 Smooth L1 损失对齐预测动作与专家演示动作：

$$
\mathcal{L}_{pred} = \mathbb{E}_{\mathbf{a}_{t+1:t+H},\epsilon\sim\mathcal{N}(0,1)}\left[\left\|\epsilon_\theta-\epsilon\right\|_2^2\right],\quad
\mathcal{L}_{react} = \mathbb{E}\left[d(\hat{\mathbf{a}}_{t+1:t+H},\mathbf{a}_{t+1:t+H})\right] \tag{3}
$$

其中 Smooth L1 距离 $d(x,\hat{x})$ 在 $|x-\hat{x}|<\beta$ 时取 $0.5(x-\hat{x})^2/\beta$,否则取 $|x-\hat{x}|-0.5\beta$（$\beta=0.1$）。整体 Stage 2 目标为加权和：

$$
\mathcal{L}_{stage2} = \mathcal{L}_{pred} + \lambda_{react}\mathcal{L}_{react} + \lambda_{gate}\mathcal{L}_{gate} \tag{4}
$$

其中 $\lambda_{react}=0.1$，$\lambda_{gate}=0.05$。全部模型在 8 张 NVIDIA H20 上训练。

### 2.4 实验平台与任务

真机平台为双 Franka Research 3（单臂/双臂两种构型）,3D 打印 UMI 夹爪,单臂配三路 Intel RealSense 相机（640×480,一路第三人称+两路腕部）,双臂配一路全局+两路腕部相机,状态维度单臂 7-DoF（位置3+欧拉角3+夹爪1）,双臂 14-DoF。演示数据经 SpaceMouse 遥操作采集,每任务 100 条轨迹。六项真机任务（四单臂+两双臂,均拆解为多个可顺序评分的子阶段）：*Pick Fruit to Plate*、*Stack Coke Cans*、*Pour Coke into Beaker*、*Write "Yes"*（单臂）,*Put Flowers in Vase*、*Put Items to Bag and Zip*（双臂,最长时程,5 个子阶段）。

## 三、实验结果

对比基线覆盖三类：VLA（π0.5、QwenVLA-OFT,后者以 Qwen3-VL-4B 为骨干并在 40 万条跨本体轨迹上预训练）、Imagine-then-Execute WAM（Wan2.2-TI2V-5B + AnyPos）、Joint Modeling WAM（VPP、Cosmos-Policy）。每个方法每任务评测 20 个独立回合（ID/OOD 均随机化桌面物体位置）。

**ID 结果（Table 2,任务级平均成功率）**：

| 方法 | Pick Fruit | Stack Cans | Pour Coke | Write "Yes" | Put Flowers | Put Items | Avg |
|---|---|---|---|---|---|---|---|
| π0.5 | 0.80 | 0.68 | 0.75 | 0.83 | 0.72 | 0.67 | 0.74 |
| VPP | 0.80 | 0.60 | 0.78 | 0.73 | — | — | 0.73 |
| Wan+AnyPos | 0.88 | 0.60 | 0.78 | 0.72 | 0.53 | 0.52 | 0.67 |
| QwenVLA-OFT | 0.78 | 0.30 | 0.73 | 0.72 | — | — | 0.63 |
| Cosmos-Policy | 0.93 | 0.65 | 0.80 | 0.83 | 0.75 | 0.72 | 0.78 |
| **HarmoWAM（Ours）** | **0.95** | **0.90** | **0.88** | **0.92** | **0.85** | **0.85** | **0.89（±3%）** |

ID 平均比领先 VLA（π0.5, 74%）高 15 个百分点,比领先 WAM（Cosmos-Policy, 78%）高 11 个百分点。

**OOD 泛化结果（Table 3,Global Avg 为三类 OOD 场景的总平均,括号为相对 ID 的相对下降幅度）**：

| 方法 | Background | Position | Objects | Global Avg |
|---|---|---|---|---|
| π0.5 | 0.60 | 0.32 | 0.54 | 0.49（33.8%↓） |
| VPP | 0.43 | 0.23 | 0.57 | 0.41（43.8%↓） |
| Wan+AnyPos | 0.53 | 0.49 | 0.58 | 0.53（20.9%↓） |
| QwenVLA-OFT | 0.46 | 0.28 | 0.50 | 0.41（34.9%↓） |
| Cosmos-Policy | 0.57 | 0.26 | 0.50 | 0.44（43.6%↓） |
| **HarmoWAM（Ours）** | **0.81** | **0.80** | **0.85** | **0.82（7.9%↓）** |

OOD 平均比领先 VLA（π0.5, 49%）高 33 个百分点,比领先 WAM（Wan+AnyPos, 53%）高 29 个百分点。三类 OOD 中"unseen position"（目标物体置于训练轨迹空间覆盖范围之外）对基线冲击最大（π0.5 降到 32%,Cosmos-Policy 降到 26%）,HarmoWAM 仍保持 80%,验证了 reactive expert 借助世界模型语义引导突破 SFT 分布局限的作用。

**消融研究（Section 4.4,在 Put Flowers in Vase 和 Pick Fruit to Plate 上做,报告文字明确给出的数字）**：

- 去掉 reactive expert（仅 predictive expert）：position OOD 成功率骤降到 **14%**,说明 predictive expert 单独难以把世界模型知识转化为可执行的探索信号。
- 去掉 predictive expert（仅 reactive expert）：position OOD 降到 **56%**,object-instance OOD 降到 **60%**,说明仅靠 reactive 推理不足以完成精细动作生成。
- 门控机制对比：用"两专家输出平均"替代自适应门控,在 position OOD 下性能下降 **46%**；用"仅在交互阶段做平均"（Keyframe-Based Averaging）将下降幅度收窄到 **31%**,但仍显著劣于自适应门控。
- 去掉 reactive expert 的世界模型隐特征条件：ID 性能降到 65%,OOD 平均降到 54%；去掉 predictive expert 的世界模型隐特征条件：ID 性能从 95% 降到 62%。
- 世界模型去噪步数消融（Table 9,Put Flowers in Vase 任务）：3 步→成功率 80%、推理频率 4 Hz（图像模糊丢细节）；5 步→**85%、4 Hz**（论文选用配置,性价比最优点）；10 步→85%、3.6 Hz；50 步→87%、3 Hz（边际收益递减,推理频率明显下降）。

**推理速度**：论文摘要/引言中报告 HarmoWAM 整体动作生成速度达 **48 Hz**（action chunk size = 12）,但这明显快于 Table 9 中"世界模型本身"的去噪频率（5 步去噪仅 3–4 Hz）,说明系统实际是"慢世界模型周期性刷新条件 + 快动作专家高频出动作"的双时钟设计,但论文正文并未展开说明世界模型与动作专家之间具体的调用频率解耦细节。

## 四、局限性

论文在 Appendix I 明确列出两条局限（并给出对应未来方向）：

1. **固定生成时域**：预训练世界模型以固定的 13 帧预测时域运作,下游任务必须保持相同的未来视频时域才能与其学到的时空动态对齐,限制了对不同时长任务需求的适应灵活性。作者提出的方向是探索自适应的未来帧生成,根据任务上下文和执行进度动态调整预测窗口。
2. **像素级生成的开销**：像素级未来帧生成提供了直观的视觉引导,但引入了不必要的生成计算开销。作者提出未来将探索 latent 级别的预测表征,以实现更高效的下游动作生成。

此外,论文在 Appendix G 的失败案例分析中报告了三类具体失败模式：**倾斜堆叠**（第三视角相机对前后位移估计不准,导致堆叠物体轻微前后偏移,松爪后倾倒/滑落）、**插入偏差**（花朵插花任务中,抓取位置的细微变化会改变茎秆有效外露长度,导致插入姿态难以对齐,双臂递送场景下相对位姿恢复本身就困难）、**拉链抓取滑脱**（夹爪材质/机械特性不足以在持续拉拽过程中保持对拉链头的稳定摩擦接触）。这些案例共同指向：HarmoWAM 改善的是"探索-精度"的宏观权衡,但接触力学/材料摩擦相关的精细失败模式仍未被现有框架覆盖。

另外从实验设计角度看,ID/OOD 评测每方法每任务仅 20 个回合（motivation 实验甚至只有 10 回合）,真机评测的统计功效有限（作者对主表复测三次估计标准差 ±3%,一定程度缓解了这一顾虑,但消融实验未见方差报告）；双臂任务的基线覆盖不全（VPP、QwenVLA-OFT 未报告双臂结果,论文说明是因为其余基线在复杂双臂任务上表现明显更差而只保留每类最强基线,这一比较可能对最终"11/29 个百分点"优势的普适性有一定影响）。

## 五、评价与展望

**优点**：本文的核心贡献并非又一个更大的世界模型,而是一次扎实的**诊断性实证研究**。作者用同一套真机任务、同一个世界模型骨干、拆解 transit/interaction 两阶段单独打分,清晰量化了 Imagine-then-Execute 与 Joint Modeling 两大范式的互补边界（Table 1 的对照实验设计本身就具有方法论参考价值,可供后续 WAM 工作复用作诊断工具）。在此基础上提出的双专家+自适应门控架构,思路上与近期"快慢双系统"VLA（如 Fast-in-Slow）、以及 Motus 等联合隐动作世界模型有相通之处,但 HarmoWAM 的差异化在于两专家共享同一个世界模型的显式（预测帧）与隐式（latent）双通道条件,而不是简单地并联两个独立策略头。真机实验规模（6 任务、四单臂两双臂、三类系统性 OOD）和门控网络的高准确率（96.95%）也提供了较有说服力的证据。

**局限与开放问题**：

1. 门控网络的监督标签来自本体感知信号（夹爪状态切换、末端高度阈值）自动抽取,这一启发式对不同任务的迁移鲁棒性未做验证。对于没有明显夹爪开合信号的任务（如连续力控作业、可变形物体操作）,这套关键帧抽取流程是否仍然有效是一个开放问题。
2. 论文未消融门控阈值 0.5 的选择,也未报告门控在决策边界附近（$s_t\approx 0.5$）时系统的稳定性或抖动（chattering）问题。高频切换两个专家是否会引入执行不连续性,值得进一步分析。
3. "48 Hz 动作生成"与"世界模型 5 步去噪仅 3–4 Hz"之间的频率差异,论文未详细说明具体的异步调用机制（世界模型是否按 action chunk 周期性刷新、还是有专门的缓存/插值策略）,这是理解系统实际可部署性的关键工程细节,留待后续版本或代码开源后厘清。
4. 世界模型预训练数据虽号称约 190 万条轨迹,但绝大部分（约 172 万条,占 89%）来自 RoboMIND 单一数据集,数据集构成的不均衡是否会让所谓"世界模型物理先验"的泛化能力被少数几种机器人本体/环境分布主导,论文未做进一步分析。
5. 固定 13 帧预测时域的局限（作者已自陈）在长时程任务（如 *Put Items to Bag and Zip* 的 5 阶段流程）中如何被规避（是否靠滑动窗口反复重新预测）也未详细说明,这与"自适应未来帧生成"的未来方向直接相关。

总体而言,HarmoWAM 提供了一个关于 WAM 范式权衡的清晰实证框架和一个工程上可行的融合方案,其真机结果在同类工作中具有较强说服力,但门控机制的可解释性边界、系统的实际推理时序细节、以及预训练数据构成的偏置分析,仍是后续工作可以深挖的方向。

## 参考

1. Chi, X. et al. *WoW: Towards a World Omniscient World Model through Embodied Interaction*. arXiv:2509.22642, 2025.（Imagine-then-Execute 范式代表工作,HarmoWAM motivation 实验的参照对象之一）
2. Hu, Y. et al. *Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations*. 2025.（VPP,Joint Modeling 基线的原型）
3. Kim, M. J. et al. *Cosmos Policy: Fine-tuning Video Models for Visuomotor Control and Planning*. arXiv:2601.16163, 2026.（论文中最强 WAM 基线）
4. Tan, H. et al. *AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation*. arXiv:2507.12768, 2025.（IDM 组件,亦是 reactive expert Orientation Decoder 设计的参照）
5. Physical Intelligence. *π0.5: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.15453, 2025.（论文中最强 VLA 基线）
