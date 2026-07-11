# DiT4DiT：视频动力学与动作联合建模,让视频生成成为机器人策略学习的 Scaling Proxy

> **论文**：*DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control*
>
> **作者**：Teli Ma, Jia Zheng, Zifan Wang et al.（Junwei Liang、Shuo Yang 为通讯作者/共同指导）
>
> **机构**：Mondo Robotics；HKUST(GZ)（香港科技大学广州）；HKUST（香港科技大学）
>
> **发布时间**：2026 年 03 月（arXiv 2603.10448，v2 于 2026-03-22 更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.10448) | [PDF](https://arxiv.org/pdf/2603.10448)
>
> **分类标签**：`视频动作模型` `Dual Flow-Matching` `Diffusion Transformer` `VLA` `世界模型代理目标`

---

## 一句话总结

DiT4DiT 用双 DiT 架构——冻结 VAE 的 Cosmos-Predict2.5-2B 视频扩散 Transformer 加一个仿 GR00T-N1 的动作扩散 Transformer——通过"非对称三时间步"的双流匹配（dual flow-matching）联合训练,把视频去噪过程中第 18 层的中间隐藏特征直接作为动作预测的条件,在从零训练、不使用额外动作预训练数据的前提下于 LIBERO 取得 98.6%、RoboCasa-GR1 24 任务取得 50.8% 的平均成功率,真机 Unitree G1 上以约 15% 于 GR00T-N1.5 的预训练数据量取得全面领先,并证明视频生成本身可以作为比语义 grounding、FLARE 式隐特征对齐更高效的策略学习代理目标(样本效率提升超 10 倍,收敛速度最高提升 7 倍)。

## 一、问题与动机

Vision-Language-Action(VLA)模型大多建立在静态图文预训练的 VLM 骨干之上,物理动态的学习完全依赖下游有限的动作标注数据。与此同时,视频生成模型(VGM)在海量互联网视频上学到了丰富的时空结构与隐式物理先验,理应是机器人策略学习的理想基础模型骨干,但此前工作大多只是把视频模型当作辅助手段——要么用它合成额外训练数据,要么从冻结的视频特征中训练单独的逆动力学模型,整体上属于多阶段(multi-stage)、非端到端的间接使用方式,尚未回答"视频生成模型应该如何作为一个有原则性的策略学习骨干被整合进来"这一核心问题。

论文明确提出两个待回答的问题:(1)视频生成本身能否作为训练鲁棒动作策略的有效训练目标?(2)视频模型学到的时空表征应该如何被提取并与动作生成耦合?

为回答第一个问题,论文在第 3 节做了一次专门的验证实验:在 RoboCasa-GR1 24 任务基准上对比三种代理目标——Grounding(用带检测头的 VLM 做物体级语义对齐)、FLARE 式(用可学习查询与 VLM 未来帧特征做 cross-attention 对齐,不显式做去噪)、视频生成(用视频 DiT 学习物理合理的未来动态)。结果显示视频生成目标不仅最终成功率更高,收敛也显著更快,且在不同训练数据量、不同训练轮次下都表现出更有利的 scaling 趋势(数据效率提升可达 10 倍以上,收敛加速可达 7 倍),这为"视频生成是可行的、乃至更优的策略学习代理目标"提供了实证支撑,也是后续架构设计的出发点。

## 二、核心方法

**问题形式化。** 不同于常规 VLA 直接学习 $\pi_\theta(a_t \mid o_t, l)$ 的映射,DiT4DiT 遵循"预测视频动态 → 逆动力学"的范式:

$$o_{t+1} \sim p_v(\cdot \mid o_t, l), \qquad a_t \sim p_a\big(\cdot \mid o_t, \mathcal{H}(o_{t+1}^{\tau_v})\big),\ \text{其中}\ o_{t+1}^{\tau_v} \xrightarrow{\tau_v \to 0} o_{t+1}$$

即先用视频分布 $p_v$ 采样未来帧,再用逆动力学分布 $p_a$ 从"未完全去噪的中间态" $\mathcal{H}(o_{t+1}^{\tau_v})$（而非重建出的完整未来帧）推出动作,训练目标是建模两者的联合分布 $a_t, o_{t+1} \sim p_{va}(\cdot \mid o_t, l)$。大白话说:模型不是先把未来画面完整画出来再"看图做动作",而是在视频还没画完、只画到一半的中间状态时就把这半成品的内部表征直接拿来指导动作。

**Flow Matching 基础。** 两个 DiT 都用流匹配(Flow Matching)训练,插值路径 $x_\tau = (1-\tau) x_0 + \tau z,\ \tau \in [0,1]$（$\tau=0$ 为干净数据,$\tau=1$ 为纯高斯噪声),目标速度场 $v^*(x_\tau, \tau) = z - x_0$,训练损失为

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, z, \tau}\big[\lVert v_\theta(x_\tau, \tau) - (z - x_0) \rVert^2\big],$$

推理时用一阶 Euler 离散化对 ODE $dx/d\tau = v_\theta(x, \tau)$ 反向积分,从纯噪声 $x_1$ 逐步走向干净数据 $x_0$。

**Dual-DiT 架构。** Video DiT 以 Cosmos-Predict2.5-2B 初始化,由因果视频 VAE 和视频扩散 Transformer 组成,按流预测参数化,并通过 Cosmos-Reason1 的多层嵌入接受语言指令条件。关键设计是不直接使用去噪完成后的输出帧,而是把 DiT 当作特征提取器:用前向 hook 机制在**固定确定性时间步** $\tau_f$ 拦截某一层(经消融确定为第 18 层)的中间隐藏激活,得到

$$\mathbf{h}_t^{\tau_f} = \mathcal{H}\big[v_\theta^{\text{video}}\big](\mathbf{z}_{t+1}^{\tau_f}, \tau_f \mid \mathbf{z}_t^0, l), \quad \mathbf{z}_{t+1}^{\tau_v} \xrightarrow{\tau_v \to 0} \mathbf{z}_{t+1}^0$$

Action DiT 则是仿 GR00T-N1 改造的独立流匹配模型,用 AdaLN 注入扩散时间步、用 cross-attention 让带噪动作 token 融合 $\mathbf{h}_t^{\tau_f}$ 与机器人本体状态,最终输出速度场,经数值积分得到动作序列。大白话说:视频 DiT 承担"感知+预测未来"的角色,但只借用它半路生成的"直觉",而不是等它把整段视频画完;动作 DiT 是一个专门的翻译器,把这份直觉和机器人当前姿态翻译成具体该怎么动。

**非对称三时间步的双流匹配联合训练。** 联合优化视频生成与动作推理面临一个矛盾:视频生成需要在全部噪声水平上训练才能学到完整去噪轨迹,而动作条件却需要一个稳定、确定的输入信号。论文的解法是把扩散过程拆成三个相互解耦的时间步:

- $\tau_v \sim \mathcal{U}[0,1]$——视频生成用均匀采样,让模型见识全部噪声水平,学会完整的去噪轨迹;
- $\tau_f$——特征提取用固定确定性时间步(从 $\{0/T, 1/T, \dots, T/T\}$ 中采样,但训练/推理时机制保持一致),作为一个"操作点",既保留早期去噪阶段的全局结构信息,也兼顾晚期阶段的细粒度细节;
- $\tau_a = 1-\sigma,\ \sigma \sim \text{Beta}(\alpha,\beta)$——动作 DiT 用 Beta 分布采样,把更多训练算力分配到流轨迹中最关键的阶段。

联合训练目标(第 4.3 节,式 10)为

$$\mathcal{L}_t^{\text{total}} = \underbrace{\mathbb{E}_{\tau_a,\epsilon}\Big[\big\lVert v_\phi^{\text{action}}(\mathbf{a}_t^{\tau_a}, \tau_a \mid \mathbf{h}_t^{\tau_f}, s) - (\epsilon - \mathbf{a}_t^0) \big\rVert^2\Big]}_{\text{动作流匹配损失}} + \lambda \underbrace{\mathbb{E}_{\tau_v,z}\Big[\big\lVert v_\theta^{\text{video}}(\mathbf{z}_{t+1}^{\tau_v}, \tau_v \mid \mathbf{z}_t^0, l) - (z - \mathbf{z}_{t+1}^0) \big\rVert^2\Big]}_{\text{视频流匹配损失}}$$

训练时文本编码器与视觉 VAE 全程冻结,仅更新两个 DiT 的参数(Algorithm 1)。推理阶段(Algorithm 2)则完全解耦采样:视频 DiT 走完整 $N_v$ 步 Euler 反向积分得到预测未来帧;而动作 DiT 只需在固定时间步 $\tau_f$ 上做**单次**确定性前向传播提取 $\mathbf{h}_t^{\tau_f}$,再走 $N_a$ 步积分得到动作——完全绕开了视频生成多步去噪循环的计算瓶颈。

消融实验(图 8b)进一步验证了这一设计的合理性:用于特征提取的去噪步数从 1 增加到 32,成功率单调下降,单步前向反而效果最好。作者的解释是联合训练下动作损失强烈正则化了第一步的隐藏态,使其编码高层可控语义而非过度拟合到某个特定重建目标的像素细节。t-SNE 可视化(图 8c)显示联合训练相比"视频/动作独立训练"的分离式方案,silhouette score 从 0.09 提升到约 0.17(近两倍),说明联合训练诱导出了更平滑、按 Early/Middle/Late 执行阶段更清晰分离的时间演化表征。

## 三、实验结果

**LIBERO(表 1,四个 500-demo 任务套件,从零训练、不使用额外动作预训练数据)：**

| 方法 | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Dita | 97.4 | 94.8 | 93.2 | 83.6 | 92.3 |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| $\pi_{0.5}$ | **98.8** | 98.2 | 95.6 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| CogVLA | 98.6 | 98.8 | 96.6 | 95.4 | 97.4 |
| GR00T-N1.5 | 96.2 | 94.2 | 96.0 | 90.0 | 94.1 |
| Qwen3DiT(from scratch,论文自建基线) | 98.0 | 98.8 | 96.0 | 93.6 | 96.6 |
| **DiT4DiT(from scratch)** | 98.4 | **99.6** | **98.6** | **97.6** | **98.6** |

DiT4DiT 在 Object/Goal/Long/Average 四项上均为最高,唯独 Spatial 项被 $\pi_{0.5}$（98.8%）以微弱优势领先。LIBERO-Long(长时序、多阶段任务)上的优势最明显(97.6% vs 次优 95.4%),作者将其归因于视频 DiT 显式建模了时空动态转移,有助于处理长时物理状态变迁。相比参数匹配的直接基线 Qwen3DiT(96.6%),DiT4DiT 在四个套件上均一致提升。

**RoboCasa-GR1(表 2,24 项家居桌面任务,每任务 1000 条演示,从零训练)：** DiT4DiT 平均成功率 50.8%,超过 GR00T-N1.5(41.8%)9.0 个百分点、GR00T-N1.6(40.8%)10.0 个百分点,超过参数匹配基线 Qwen3DiT(36.2%)达 14.6 个百分点(绝对值)。在 24 个任务中的 16 个上取得最高成功率,增益最显著的例子包括 CanToDrawerClose(74.0% vs 最强基线 56.0%)和 FromCuttingboardToPan(76.0% vs 62.0%)。

**真机 Unitree G1(图 5,7 项家居任务,每任务 200 条真机演示训练、20 次 rollout 评测;DiT4DiT 先在 24.1 万条 GR1 仿真数据上预训练——约为 GR00T-N1.5 预训练数据量的 15%——再在真机数据上微调)：**

| 任务 | DiT4DiT | GR00T-N1.5 | Qwen3DiT(相同流程) |
|---|---|---|---|
| Insert Plate | 85 | 80 | 10 |
| Drawer Interaction | 90 | 80 | 0 |
| Arrange Flower | 75 | 25 | 0 |
| Move Spoon | 40 | 15 | 10 |
| Pick & Place | 100 | 90 | 10 |
| Box Packing | 50 | 40 | 0 |
| Stack Cup | 60 | 25 | 5 |

Qwen3DiT 基线(同架构、仅用图文预训练 VLM 初始化)在真机上近乎完全崩溃,任一任务成功率不超过 10%,在 Drawer Interaction / Arrange Flower / Box Packing 三项上为 0%,凸显纯静态图文先验难以支撑精细物理接触任务;DiT4DiT 相对预训练数据量大得多的 GR00T-N1.5 仍保持全面领先,尤其在 Arrange Flower(需要把细茎插入花瓶的高精度对齐任务,75% vs 25%)和 Stack Cup(60% vs 25%)上优势显著。

**零样本泛化：** 仿真中限定训练分布为单一物体类别(仅用 Bottle 训练三个 close 任务),测试未见类别(Can/Cup/Milk/Wine):ToDrawerClose 上 DiT4DiT 54.5% vs Qwen3DiT 32.0%(领先 22.5pp),ToCabinetClose 34.0% vs 24.5%,ToMicrowaveClose 30.5% vs 17.0%。真机上设计了类别变化(Stack Cup/Arrange Flower 换用不同材质外观的物体)、物体替换(Box Packing 用玉米替换茄子)、数量变化(Stack Cup 由 3 只变 4 只)三类 OOD 场景:Stack Cup(Category) DiT4DiT 80% vs GR00T-N1.5 40% vs Qwen3DiT 10%;Arrange Flower(Category) 70% vs 10% vs 0%;Stack Cup(Number) 50% vs 20% vs 0%,证明生成式视频表征对表面视觉/干扰物变化具有较强的物理不变性。

**部署效率(表 3)：**

| 方法 | 可训练参数量 | 真机控制频率 |
|---|---|---|
| GR00T-N1.5 | 2.7B | 13Hz |
| Qwen3DiT | 2.3B | 9Hz |
| DiT4DiT | 2.2B | 6Hz |

DiT4DiT 参数量最小,但受视频扩散骨干拖累,推理频率也最低(单张 NVIDIA A100 实测),不过因动作解码只需单次前向而非完整视频生成循环,仍可支持实时闭环控制;论文指出固定任务下 LLM 语言特征可预先提取缓存以进一步提高有效部署频率。

## 四、局限性

论文附录 A.4 自陈两点局限,加上正文实验暴露的问题,主要包括:

1. **单目视觉依赖。** 真机部署仅用单个自我中心(egocentric)摄像头,双臂任务中机械臂本身或较大物体容易遮挡视野,破坏视觉特征的时间连续性;作者提出未来可融合腕部相机、触觉反馈等辅助感知模态。
2. **预训练数据规模仍偏小。** 真机预训练语料仅为 GR00T 等当代大规模模型数据量的约 15%,虽然这凸显了方法的数据效率,但尚未验证在海量、跨具身(不同运动学结构、夹爪、相机参数)数据上进一步扩展后能否维持甚至放大当前优势——这是作者明确列出的"自然且有前景的下一步"。
3. **推理频率偏低。** 视频扩散骨干带来了明显的计算开销:6Hz 远低于 GR00T-N1.5 的 13Hz 与 Qwen3DiT 的 9Hz,尽管动作解码本身只需单次前向,这一权衡在需要更高频闭环控制的动态任务上可能构成瓶颈。
4. **关键超参数经验化。** 特征提取层(第 18 层)与提取时间步(单步)均来自任务特定的消融搜索,论文未给出在不同视频骨干规模或不同机器人形态下是否需要重新搜索的普适规律。
5. **任务场景较局限。** RoboCasa-GR1 与真机实验均为桌面 / 人形上肢操作,尚未覆盖移动操作或更开放、非结构化的环境。

## 五、评价与展望

**优点。** 论文没有停留在"用视频模型生成增强数据"或"从冻结视频特征训练独立逆动力学"这类多阶段套路,而是通过非对称三时间步的巧妙设计,让视频生成对全部噪声水平的建模需求与动作条件对稳定确定性特征的需求在同一个训练目标里共存,实现了真正的端到端联合优化。第 3 节针对 Grounding / FLARE-style / 视频生成三种代理目标的正面对比是一个扎实的消融证据,没有想当然地假设"视频生成更好",而是先用实验验证了核心假设再展开架构设计,论证链条比较完整。推理阶段"单步前向即可提取最优动作条件特征"这一发现(图 8b)具有明确的实践价值——它意味着可以只保留视频扩散模型的"表征"价值而非"生成"价值,为这类视频-动作模型(Video-Action Model, VAM)的实际部署效率提供了一条现实路径。

**与其他公开工作的关系。** 论文在第 2 节将自己与两条最相关的同期工作做了明确区分:Cosmos Policy(Kim et al., 2026)直接微调预训练视频扩散模型,把动作和未来期望值编码为视频扩散过程中的连续潜在帧一并输出;mimic-video(Pai et al., 2025)则将预训练视频骨干与独立的 flow-matching 动作解码器配对,让策略基于某个中间去噪时刻的部分去噪视频潜变量做条件。DiT4DiT 与二者的核心差异在于坚持对视频与动作做**联合训练**,而非"微调已训练好的视频模型"或依赖"固定的中间去噪状态",使动作模型能主动学习如何跨视频生成的不同阶段提取有效特征。与 FLARE(VLM-centric 隐式世界建模,用 cross-attention 让可学习查询与未来帧特征对齐,不做显式像素级去噪)相比,论文用第 3 节的实证结果说明基于视频 DiT 的路线在样本效率和收敛速度上都更优,这为该子领域今后选择代理目标提供了一份直接的实证参考。

**开放问题与可能的改进方向。** (1)视频骨干(Cosmos-Predict2.5-2B)与动作骨干均为 2B 级别的中等规模模型,论文未研究更大规模视频骨干是否会带来表征质量的进一步提升,以及这种提升能否等比例转化为下游成功率,这与当前 VAM/WAM 研究普遍关心的 scaling law 问题直接相关;(2)特征提取层与时间步的最优值高度依赖经验消融,是否存在与网络深度、任务复杂度相关的普适选择准则仍是开放问题;(3)当前方法仅在单臂/双臂桌面操作与人形上肢操作上验证,尚未拓展到移动操作(导航+操作联合)或多相机、触觉等多模态融合场景;(4)真机实验中 Qwen3DiT 基线(与 DiT4DiT 同架构但仅以图文预训练 VLM 初始化)几乎完全失效,这一强烈对比虽然有力支撑了"视频生成先验优于静态图文先验"的核心论点,但也提示该结论的稳健性可能部分依赖于所选视频骨干(Cosmos-Predict2.5)本身已在大规模物理/具身相关视频上做过专门预训练——若换用未经过物理场景强化的通用互联网视频扩散模型作为初始化,优势幅度是否会缩小,仍待进一步验证。

## 参考

- Kim et al. *Cosmos Policy: Fine-tuning Video Models for Visuomotor Control and Planning.* arXiv:2601.16163, 2026——论文在第 2 节点名的最相关同期工作,直接微调预训练视频扩散模型统一输出动作与未来期望值。
- Pai et al. *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs.* arXiv:2512.15692, 2025——另一最相关同期工作,视频骨干配独立 flow-matching 动作解码器,在部分去噪的中间视频潜变量上做条件。
- Zheng et al. *FLARE: Robot Learning with Implicit World Modeling.* arXiv:2505.15659, 2025——论文核心对比基线之一(VLM-centric 隐式世界建模,cross-attention 对齐未来帧特征)。
- Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734, 2025——Action DiT 架构的改编来源,也是真机实验的主要预训练权重基线(GR00T-N1.5)。
- Ali et al. *World Simulation with Video Foundation Models for Physical AI.* arXiv:2511.00062, 2025——Video DiT 初始化所用的 Cosmos-Predict2.5-2B 视频扩散骨干来源。
