# MotionWAM：面向实时人形全身移动操作的基础世界-动作模型

> **论文**：*MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation*
>
> **作者**：Jia Zheng†, Teli Ma†, Yudong Fan, Zifan Wang, Shuo Yang*, Junwei Liang* et al.（†共同一作，*通讯/共同指导）
>
> **机构**：Mondo Robotics；HKUST(GZ)；HKUST
>
> **发布时间**：2026 年 06 月（arXiv 2606.09215）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.09215) | [PDF](https://arxiv.org/pdf/2606.09215)
>
> **分类标签**：`World Action Model` `人形全身操作` `实时闭环控制` `流匹配` `统一动作空间`

---

## 一句话总结

MotionWAM 用一次前向的"单步去噪特征"（而非多步迭代去噪）把视频世界模型的动态先验注入策略，并将人形机器人的上肢操作与下肢移动统一到同一个全身运动 latent 空间中，在 Unitree G1 的九项真实全身移动操作任务上把最强 VLA 基线（GR00T-N1.7）的平均成功率从 43.9% 提升到 76.1%，同时以 4.9 Hz 的闭环频率运行，比同参数量级的世界模型策略 Cosmos Policy（0.7 Hz）快 7 倍。

## 一、问题与动机

World Action Models（WAM）通过让策略"看见"视频生成器预测的未来来注入动态先验，在桌面臂式操作上已展现潜力，但存在两个尚未解决的问题：

1. **实时性瓶颈**：现有 WAM 大多依赖对高维视频-动作 latent 的迭代式去噪（多步 diffusion/flow 采样），计算开销大，即使在固定底座的桌面机械臂上也难以做到实时闭环控制，更遑论需要动态平衡的人形机器人。
2. **上下肢动作空间割裂**：主流人形全身移动操作系统采用分层范式——高层操作策略只输出上肢的精细关节目标，下肢则交给一个仅接收粗粒度底座指令（速度、躯干高度、朝向）的低层控制器。这种设计把双腿限制为"维持平衡"的被动角色，无法表达任务驱动的脚部行为（例如踩踏板、踢球），因为下肢从未被纳入以任务为中心的动作空间。

论文提出的核心问题是：能否把 WAM 丰富的动态先验部署到实时、统一 latent 空间下的人形全身移动操作中？由此提出 **MotionWAM**：一个从单目自我中心相机驱动人形机器人自主全身移动操作的端到端实时 WAM。

## 二、核心方法

### 2.1 问题建模：预测视频动态-再反演

MotionWAM 遵循"predict-video-dynamics-then-invert"范式，预测一个统一的**全身运动 latent**来驱动整个身体：

$$
\mathbf{o}_{t+1} \sim p_v(\cdot \mid \mathbf{o}_t, l), \quad \mathbf{m}_t \sim p_a(\cdot \mid \mathbf{o}_t, p_t, \mathcal{H}(\mathbf{o}_{t+1}^{\tau_v})), \quad \text{其中 } \mathbf{o}_{t+1}^{\tau_v} \xrightarrow{\tau_v \to 0} \mathbf{o}_{t+1}.
$$

**大白话**：$l$ 是语言目标，$\mathbf{o}_t$ 是当前自我中心观测，$p_t$ 是本体感知状态，$\mathbf{o}_{t+1}^{\tau_v}$ 是视频生成过程中流步 $\tau_v$ 时刻的"半成品未来帧"。模型不需要把未来帧真正去噪完成，只需要从这个半成品状态里用 $\mathcal{H}(\cdot)$ 抽取隐藏特征，就足以指导全身动作 $\mathbf{m}_t$ 的预测——训练时联合建模观测和动作的联合分布 $p_{va}(\mathbf{o}_{t+1}, \mathbf{m}_t \mid \mathbf{o}_t, p_t, l)$。

**统一全身运动 latent**：$\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \mathbf{k}_t)$ 建立在 SONIC（一个通用人形全身控制器）之上。SONIC 用 Finite Scalar Quantization（FSQ，2 个 token、每个 32 级）把不同运动目标压缩到一个共享 latent 中，得到 $\mathbf{k}_t \in \{-1, -15/16, \ldots, 1\}^{64}$，这是一个 64 维离散化向量，概括了移动、身高、脚部交互意图。连续部分 $\mathbf{m}_t^{\text{cont}}$ 则收纳 SONIC 未覆盖的灵巧通道（左右夹爪或灵巧手指令），直接驱动双手。

### 2.2 双 DiT 架构：Video DiT + Motion DiT

Video DiT 从 Cosmos-Predict2.5-2B 初始化——一个因果时空 VAE 加流匹配 diffusion transformer，以 Cosmos-Reason1 语言嵌入为条件——把条件帧 $\mathbf{o}_t$ 与未来帧 $\mathbf{o}_{t+1}$ 压缩进 latent $\mathbf{z}_t^0, \mathbf{z}_{t+1}^0$。关键设计是在单个 transformer block 上安装一个前向钩子，在**固定**流时间步 $\tau_f$ 截取其激活：

$$
\mathbf{h}_t^{\tau_f} = \mathcal{H}\big[v_\theta^{\text{video}}\big]\big(\mathbf{z}_{t+1}^{\tau_v}, \tau_f \mid \mathbf{z}_t^0, l\big), \quad \mathbf{z}_{t+1}^{\tau_f}\big|_{\tau_f \to 1} \sim \mathcal{N}(0, I)
$$

**大白话**：把 $\tau_f$ 固定在噪声调度的"纯噪声端"（$\tau_f \approx 1$），也就是说 Video DiT 只跑**一次**前向——给定干净的条件帧 latent 和未来帧的高斯噪声，一次前向得到的激活就已经编码了"场景即将往哪走"的信息，完全不需要真正把未来帧迭代去噪出来。作者称之为"one-shot imagination"体制，这正是让 MotionWAM 能在闭环人形控制中保持实时性的核心机制。

Motion DiT 通过交错的自注意力/交叉注意力消费 $\mathbf{h}_t^{\tau_f}$，同时嵌入本体感知状态 $p_t$ 和带噪的全身运动 latent token，输出速度场，积分后得到运动 latent $\mathbf{m}_t$。多具身预训练期间，共享的 Motion DiT 主干外围包裹按具身区分的输入/输出投影器；部署时复用同一主干，只挂载单一的 Unitree G1 投影器。

### 2.3 三阶段训练流程

Video 和文本编码器全程冻结；训练遵循"先分工、再联合"的直觉：

- **Stage 1——自我中心视频预训练**：仅用 $\mathcal{L}_{\text{video}}$ 单独训练 Video DiT，在约 2,136 小时的自我中心人类视频与人形机器人视频上训练：

$$
\mathcal{L}_{\text{video}} = \mathbb{E}_{\tau_v, \mathbf{z}_{t+1}^0, \epsilon_v}\Big[\big\| v_\theta^{\text{video}}(\mathbf{z}_{t+1}^{\tau_v}, \tau_v \mid \mathbf{z}_t^0, l) - (\epsilon_v - \mathbf{z}_{t+1}^0) \big\|_2^2\Big]
$$

  核心洞察：这一阶段的瓶颈是自我中心视觉动态而非动作多样性——用便宜的、无动作标签的视频单独训练 Video DiT，主干可以吸收规模而不被更小的带动作标注数据池卡脖子，产出一个隐藏状态编码了"可信自我中心未来"的机器人中心动态先验。

- **Stage 2——跨具身动作后训练**：挂上 Motion DiT，在跨越不同末端执行器和动作标注格式的异构 Unitree G1 人形数据上联合训练，经由具身专属的输入/输出投影器路由到共享 Motion DiT 主干：

$$
\mathcal{L}_{\text{motion}} = \mathbb{E}_{\tau_a, \mathbf{m}_t^0, \epsilon_m}\Big[\big\| v_\phi^{\text{motion}}(\mathbf{m}_t^{\tau_a}, \tau_a \mid \mathbf{h}_t^{\tau_f}, p_t, e) - (\epsilon_m - \mathbf{m}_t^0) \big\|_2^2\Big]
$$

  为防止动作信号一到来就冲刷掉动态先验，保留视频目标作为表征正则项，联合损失为

$$
\mathcal{L}_{\text{Stage 2}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}
$$

- **Stage 3——全身遥操作数据微调**：在目标任务上采集遥操作全身演示（九项真实任务各 200 条），沿用 Stage 2 的联合损失，把动作输出切换为驱动人形机器人端到端的统一全身运动 token。

**部署侧解码**：运动 latent 分解为 $\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \tilde{k}_t)$，其中 $\tilde{k}_t \in \mathbb{R}$ 是概括全身移动操作轨迹的单个标量槽位，对应 SONIC 运动 token 索引 $k_t \in \{0, \ldots, K-1\}$。作者没有额外引入一个分类头，而是让 $\tilde{k}_t$ 以连续标量的形式活在 $\mathbf{m}_t$ 内，在同一个流匹配目标下回归整个 latent，推理时通过最近邻取整恢复离散索引，再由 SONIC 把组装好的 latent 解码成关节指令：

$$
\mathbf{m}_t = (\mathbf{m}_t^{\text{cont}}, \tilde{k}_t) \xrightarrow{\text{Eq.(4)}} \hat{\mathbf{m}}_t = (\hat{\mathbf{m}}_t^{\text{cont}}, \hat{k}_t) \xrightarrow{\hat{k}_t = \text{round}(\tilde{k}_t)} (\hat{\mathbf{m}}_t^{\text{cont}}, \hat{k}_t) \xrightarrow{\text{SONIC}} \mathbf{a}_t
$$

**大白话**：与其单独训一个离散分类头去预测运动 token 的类别编号，不如让这个编号本身"伪装"成一个连续数，跟其他连续动作通道一起用流匹配统一回归，推理时四舍五入取整即可——省掉了一套额外的架构和损失。

## 三、实验结果

**平台与任务**：Unitree G1 人形机器人 + 双 ALOHA2 夹爪（7-DoF 双臂，16-DoF 上肢操作接口），头戴 Intel RealSense D435i 单目 RGB 相机；遥操作经 PICO VR 三点跟踪 + SMPL 重定向采集，部署时由 SONIC 全身控制器跟踪。策略以 WebSocket 服务运行于单张 NVIDIA RTX 4090。九项真实任务：PnP Bottle、Kick Soccer、Retrieve Item、Load Cart、Toss Garbage、Lift Basket、Stock Shelves、Wipe Board、Do Laundry，每项均要求腿部/躯干主动参与而非仅维持平衡，每个任务 20 次试验评测成功率。

**基线**：Diffusion Policy、ACT（非 VLA 视觉运动策略）；π₀.₅、GR00T-N1.7（SOTA 通用 VLA）；以及作者自建的参数量匹配消融 Qwen3DiT——把 Video DiT 替换为 Qwen3-VL 2B 视觉语言主干、其余（Motion DiT、统一动作空间、观测/本体接口）与 MotionWAM 完全一致，用以单独隔离"视频世界模型先验 vs. 静态 VLM 先验"的贡献。所有基线在同一批 Stage 3 演示上微调。

**Q1：与 SOTA VLA 的对比**（图 5）：

| 方法 | 主干类型 | 平均成功率 |
|---|---|---|
| MotionWAM（本文） | 视频世界模型（单步去噪特征） | **76.1%** |
| GR00T-N1.7 | VLM + diffusion 动作头 | 43.9%（最强基线） |
| π₀.₅ | VLM + flow-matching 动作专家 | ＜20%（整体） |
| Qwen3DiT（消融） | 静态 VLM（Qwen3-VL 2B）+ 同款 Motion DiT | 在移动密集型任务上接近 0 |
| Diffusion Policy / ACT | 非 VLA 视觉运动策略 | 在多数移动操作任务上失败 |

MotionWAM 在全部九项任务上均取得最高成功率，相对最强基线整体提升超过 32 个百分点（43.9%→76.1%）。差距最大的任务集中在需要上肢之外的全身协调能力上：Kick Soccer（+40%）、Load Cart（+40%）、Retrieve Item（+40%）、Wipe Board（+45%）、Do Laundry（+30%）——统一运动 latent 让 MotionWAM 具备了任务驱动的脚部/躯干行为，而分离式上下肢接口做不到。VLM-only 的 Qwen3DiT 在每个移动密集型任务上都接近零成功率，说明单纯的强语义先验无法迁移到闭环物理全身移动操作需求；耦合视频世界模型才是弥合这一差距的关键。

**Q2：三阶段训练框架消融**（Table 1，五项代表性任务：Lift Basket、Retrieve Item、Load Cart、Toss Garbage、Kick Soccer，Stage 3 全程启用）：

| 变体 | Stage 1 | Stage 2 | Lift Basket | Retrieve Item | Load Cart | Toss Garbage | Kick Soccer | 平均 |
|---|---|---|---|---|---|---|---|---|
| w/o Stage 2 | ✓ | – | 65 | 45 | 30 | 30 | 40 | 42.0% |
| w/o Stage 1 | – | ✓ | 70 | 75 | 60 | 35 | 55 | 59.0% |
| Full | ✓ | ✓ | 80 | 90 | 75 | 45 | 60 | **70.0%** |

去掉 Stage 1（自我中心视频预训练）导致 11 个百分点的绝对下降；去掉 Stage 2（跨具身动作后训练）导致 28 个百分点的绝对下降，是更关键的一环。作者解释：缺少 Stage 1，Video DiT 直接进入 Stage 2 时携带的是通用、非自我中心的动态先验，预测的运动 latent 明显更不准；缺少 Stage 2，Motion DiT 直接挂接并只在小规模目标任务数据集上训练，没有跨具身的动作落地基础，性能在每个任务上都塌缩。两阶段互补：Stage 1 提供自我中心视觉动态先验，Stage 2 把这个先验落地到跨具身的动作空间中。

**Q3：实时推理频率**（Table 2，单张 NVIDIA A100 上测得的闭环整段动作 chunk 发出频率）：

| 模型 | 可训练参数量 | 频率 |
|---|---|---|
| GR00T-N1.7 | 1.6B | 6.5 Hz |
| Qwen3DiT | 2.3B | 9.0 Hz |
| Cosmos Policy | 2.0B | 0.7 Hz |
| MotionWAM（本文） | 2.5B | 4.9 Hz |

与同为世界模型驱动策略、参数量相当的 Cosmos Policy 相比，MotionWAM 快 7 倍（4.9 Hz vs. 0.7 Hz），原因是 Cosmos Policy 需要在产出动作前迭代去噪出完整未来视频，而 MotionWAM 只在单次前向中读取中间去噪特征。这证实了 MotionWAM 在图 5 中的精度提升并未以牺牲闭环人形平衡所需的实时控制速率为代价。

## 四、局限性

论文第 6 节明确列出三点：

1. **具身泛化未验证**：Stage 3 微调仅在 Unitree G1 一种具身上验证，三阶段范式是否能迁移到其他人形硬件平台尚未确认。
2. **缺乏受控的新物体泛化研究**：训练集与测试集物体在视觉上高度相似，论文未报告严格分布外（out-of-distribution）物体上的成功率。
3. **单目相机视野受限的失败模式**：依赖单一自我中心头戴相机，当被操作物体离开视野或头部相机视角偏离训练分布时，策略会丢失视觉锚定，出现停滞或走向不准确的全身轨迹（附录 F 给出 Lift Basket、Kick Soccer 的具体失败案例）。

## 五、评价与展望

**优点**：MotionWAM 对现有 WAM 研究给出了两个明确且可验证的贡献。其一，"单步 one-shot imagination"这一工程选择直接回应了 WAM 领域长期被诟病的实时性瓶颈——相比同参数规模、同样基于世界模型的 Cosmos Policy，7 倍的频率提升是具体且可复现的对比（同一 A100、相近参数量）。其二，用统一全身运动 latent 替代分层上下肢解耦，是对人形全身移动操作范式的一次有意义的挑战：论文用"踢球""踩踏板"等任务驱动脚部行为的成功率差距（相对基线 +30～45 个百分点）实证了分层范式在表达任务驱动下肢行为上的结构性缺陷，而不仅仅停留在直觉论证层面。三阶段训练的消融（Stage 2 影响明显大于 Stage 1）也提供了一个值得后续 WAM 工作参考的经验：动作接地（action grounding）阶段的跨具身数据量级，可能比视频先验预训练阶段更决定下游真机表现。

**局限与开放问题**：（1）论文的评测规模不大——九个任务、每任务 20 次试验、单一具身、单一相机视角，成功率的统计置信区间未报告，跨任务的方差也不明确；(2) 消融只覆盖了五个任务子集而非全部九个，Stage 1/2 的贡献是否在 Kick Soccer 这类强动力学任务上被放大、在纯操作任务上被稀释，缺乏进一步拆解；(3) 与同期 WAM 工作（如 WorldVLA、UVA、Motus 等自回归/隐动作路线，及 DiT4DiT 这类联合视频-动作建模的直接前驱）相比，本文没有做直接的正面对比，仅以自建的 Qwen3DiT（静态 VLM 消融）和 Cosmos Policy（速度对比）间接说明优势，视频先验的具体收益边界（例如替换成更小/更大的 Video DiT backbone、替换 SONIC 为其他全身控制器）仍是开放问题；(4) SONIC 的 FSQ 离散化 + 连续标量回归再取整（Eq. 6）这一设计虽然工程上简洁，但离散索引的取整误差如何影响长时程闭环稳定性，论文未给出误差分析；(5) 附录 F 指出的单目视野丢失问题，是当前几乎所有基于单相机 WAM 的共性弱点，多视角或主动视觉（如可控头部朝向）可能是后续提升鲁棒性的方向。总体而言，MotionWAM 提供了一个把 WAM 的动态先验成功落地到高动力学、全身、实时闭环场景的具体范例，其"读中间特征而非等去噪完成"的思路对整个 WAM 领域的实时化具有一般性的借鉴意义。

## 参考

- Luo et al. *SONIC: Supersizing motion tracking for natural humanoid whole-body control.* arXiv:2511.07820, 2025.（MotionWAM 运动 latent 与低层解码所依赖的全身控制器）
- Ali et al. *World simulation with video foundation models (Cosmos-Predict2.5).* arXiv:2511.00062, 2025.（Video DiT 初始化来源）
- Ma et al. *DiT4DiT: Jointly modeling video dynamics and actions for generalizable robot control.* arXiv:2603.10448, 2026.（联合流匹配目标的直接灵感来源）
- Kim et al. *Cosmos Policy: Fine-tuning video models for visuomotor control and planning.* arXiv:2601.16163, 2026.（Table 2 中速度对比的同类世界模型策略基线）
- Bjorck et al. *GR00T N1: An open foundation model for generalist humanoid robots.* arXiv:2503.14734, 2025.（图 5 中最强 VLA 基线 GR00T-N1.7 的基础模型）
