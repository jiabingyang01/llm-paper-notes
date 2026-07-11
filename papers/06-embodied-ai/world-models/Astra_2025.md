# Astra：基于自回归去噪的通用交互世界模型

> **论文**：*Astra: General Interactive World Model with Autoregressive Denoising*
>
> **作者**：Yixuan Zhu*, Jiaqi Feng*, Wenzhao Zheng†, Yuan Gao, Xin Tao, Pengfei Wan, Jie Zhou, Jiwen Lu（*同等贡献，†项目负责人）
>
> **机构**：Tsinghua University；Kuaishou Technology
>
> **发布时间**：2025 年 12 月（arXiv 2512.08931，v3 于 2026 年 1 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.08931) | [PDF](https://arxiv.org/pdf/2512.08931)
>
> **分类标签**：`世界模型` `自回归去噪` `动作条件视频生成` `MoAE` `flow matching`

---

## 一句话总结

Astra 在冻结的 Wan-2.1（1.3B）视频扩散骨干之上做 **chunk-wise 自回归去噪**，用轻量 **ACT-Adapter** 把动作注入 latent、用 **noise-as-mask** 破坏历史帧以缓解 "视觉惯性"、用 **Mixture of Action Experts（MoAE）** 路由相机/机器人/键鼠等异构动作模态，仅训练约 366M 参数即在 6 项指标上全面领先，指令跟随 0.669、相机旋转误差从基线 2.96 降到 1.23。

## 一、问题与动机

标准 T2V/I2V 模型只能生成短的、自包含的片段，缺乏对外部动作（智能体移动、视角变化、控制信号）的**在线响应** 能力，因此本质上是"被动"的高保真渲染器，而非可交互、可探索的世界模拟器。作者指出把自回归（AR）与扩散结合来做长时预测时，存在两个核心矛盾：

1. **动作可控性 vs. 生成质量**：预训练视频 DiT 的条件机制是为文本 cross-attention 设计的，天然不擅长表达"动作引发的 latent 空间位移"这种细粒度控制。
2. **长时一致性 vs. 响应灵敏度**：作者观察到一个现象——延长历史条件能提升时序连贯，却会削弱动作响应，模型倾向于"抄"过去的平滑视觉信息而忽略当前动作指令。作者把它命名为 **visual inertia（视觉惯性）**。附录 Figure C 量化了这一权衡：当上下文长度从约 0 增加到 80，imaging quality 从约 0.40 升到 0.77，但 action-following 从约 0.68 骤降到 0.19。

此外真实交互环境涉及**异构动作模态**（相机位姿、机器人位姿、离散键鼠命令），单一模型难以统一建模。Astra 针对这三点分别提出 ACT-Adapter、noise-as-mask 和 MoAE。

## 二、核心方法

### 2.1 自回归去噪骨架（flow matching）

把视频离散成 chunk 序列 $z^{1:N}$，按条件分布逐块生成：

$$p(z^{1:N}) = \prod_{i=1}^{N} p(z^i \mid z^{<i})$$

**用大白话说**：不是一次性把整段视频吐出来，而是"看着已经生成的前几块，再猜下一块"，这样才可能在生成中途接收并响应新动作。

每一块用 flow matching 训练。对目标 chunk 采样带噪插值：

$$z_t^i = (1-t)\, z_0^i + t\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I),\ t \in [0,1]$$

训练 flow model $v_\theta$ 去估计"指向干净方向"的速度场：

$$\mathcal{L}(\theta) = \mathbb{E}_{i,t,\epsilon}\Big[\big\| v_\theta(z_t^i, t \mid z^{<i}) - v^*(z_t^i, t \mid z^{<i}) \big\|_2^2\Big]$$

**用大白话说**：给下一块加噪，让网络学会"该往哪个方向去噪才能还原真实的下一块"，$v^*$ 是真值速度。推理时从纯噪声去噪得到 $z^{i+1}$，再拼回历史继续预测——这就是 AR–去噪循环。

### 2.2 Action-Aware Flow Transformer（ACT-Adapter）

作者把"动作"理解为对视频特征（即去噪器 latent）的一次**平移/位移**，灵感来自光流。做法：

- 用一个 **action encoder** 把动作投影到与视频 latent 对齐的特征空间，然后在**每个 transformer block 里以逐元素相加** 的方式注入动作特征。
- 为最大化复用预训练知识，**冻结 flow transformer 的绝大部分参数，只微调 self-attention 层**；并在每个 self-attention block 之后插入一个**用单位矩阵初始化的单层线性 adapter**（即 ACT-Adapter），让模型从"恒等映射"出发平滑地学到动作感知的变换。
- 历史条件采用 **frame-dimension conditioning**：把前面已生成的 chunk 沿时间维拼接到待预测块之前。完整条件集为 $\mathcal{C} = \{z^{1:i-1},\, a^{1:i},\, c\}$（历史、动作、文本 prompt）。

**Action-Free Guidance（AFG）**：类比 CFG，训练时随机丢弃动作条件；推理时做引导：

$$v_{\text{guided}} = v_\theta(z_t, t, \emptyset) + s\cdot\big(v_\theta(z_t, t, a) - v_\theta(z_t, t, \emptyset)\big)$$

其中 $s$ 为引导尺度（默认 3.0），$\emptyset$ 为空动作条件。**用大白话说**：把"有动作"和"没动作"两次预测的差值放大，从而让动作效果更锐利、响应更精准。

### 2.3 噪声记忆(noise-as-mask)对抗视觉惯性

不同于 Yume（Mao et al. 2025）随机 mask 视觉 token，Astra 直接**给历史条件视频注入随机噪声，模糊掉它的信息量**。两大好处：

1. 不需要改架构、不引入额外可学习参数；
2. 破坏干净视觉上下文可以防止模型"直接抄历史帧"，逼它把动作线索整合进生成。

关键点：这个腐蚀噪声与扩散噪声**相互独立**，因此推理时仍可喂入干净历史帧。此外借鉴 Zhang & Agrawala（2025）的压缩思路——保留首帧、把中间历史压成紧凑 visual token，在不淹没动作信号的前提下延长有效历史窗口。

### 2.4 Mixture of Action Experts(MoAE)

统一异构动作模态：相机位姿 $a_{\text{cam}}$、机器人位姿 $a_{\text{rob}}$、离散键鼠命令 $a_{\text{cmd}}$，各经模态专属投影器 $\mathcal{R}_m$ 映射到共享动作空间 $\tilde{a}^i = \mathcal{R}_m(a_m^i)$，$m \in \{\text{cam}, \text{rob}, \text{cmd}\}$。路由网络算门控分数 $g^i = \mathrm{Router}(\tilde{a}^i)$ 选 top-$K$ 专家（每个专家为独立 MLP），聚合得动作嵌入：

$$e^i = \sum_{k=1}^{K} g_k^i\, E_k(\tilde{a}^i)$$

再喂入 flow transformer。为同时处理历史与当前动作，动作空间还额外加一个二值指示符标明"过去/当前"。**用大白话说**：不同来源的动作先各自翻译成同一种"通用动作语言"，再由路由器挑最相关的专家处理，既专精又只激活少量专家(每步仅 1 个专家 active)、省算力。

### 2.5 训练配置

从 Wan-2.1（1.3B，30 个 DiT-style flow transformer block）初始化；8 卡（各 80GB）、per-GPU batch=1、AdamW、LR $1e{-}5$、30 epoch，约 24 小时收敛；在 3D VAE 的 latent 空间训练，分辨率 480×832；像素空间条件帧数在 $[1,128]$ 随机采样，目标帧固定 33。**可训练参数仅约 366M**（对比 YUME 全调 ~14B、MatrixGame ~1.8B、NWM ~1B）。

## 三、实验结果

**数据（Table 1，合计约 397K 段、360 小时）**：nuScenes（相机 7 维，自动驾驶，850）、Sekai（相机 12 维，步行/无人机视角，50K）、SpatialVID（相机 7 维+键鼠，野外视频，200K）、RT-1（机器人位姿 7 维，机器人操作，9978；取自 Open X-Embodiment）、Multi-Cam Video（相机 12 维，人体运动，136K）。评测自建 **Astra-Bench**（每数据集 20 个 held-out 样本），生成 480×832、20 FPS、96 帧、50 步；指令跟随用 20 名用户人评，其余 5 项用 VBench。

**主结果（Table 2，↑ 越大越好，加粗为最优）**：

| 方法 | Instr. Follow | Subj. Cons. | Bg. Cons. | Motion Smooth | Aesthetic | Imaging |
|---|---|---|---|---|---|---|
| Wan-2.1 | 0.061 | 0.854 | 0.903 | 0.958 | 0.489 | 0.691 |
| MatrixGame | 0.268 | 0.916 | 0.928 | 0.981 | 0.441 | **0.748** |
| Yume | 0.652 | 0.936 | 0.938 | 0.985 | 0.523 | 0.741 |
| **Astra** | **0.669** | **0.939** | **0.945** | **0.989** | **0.531** | 0.747 |

Astra 在 6 项里 5 项最优，仅 imaging quality 微差于 MatrixGame（0.747 vs 0.748）；指令跟随相对 Wan-2.1（0.061）提升一个数量级。

**动作对齐(Table A，相机旋转/平移误差,↓ 越小越好)**：

| 方法 | RotErr ↓ | TransErr ↓ |
|---|---|---|
| Wan-2.1 | 2.96 | 7.37 |
| YUME | 2.20 | 5.80 |
| MatrixGame | 2.25 | 5.63 |
| NWM | 2.47 | 6.13 |
| **Astra** | **1.23** | **4.86** |

Astra 的相机旋转、平移误差均显著最低，与人评的指令跟随结论一致。（注：Table A 中 YUME/MatrixGame 的 Instruction Following 数值与 Table 2 互换，疑为排版笔误，此处只取更稳健的 RotErr/TransErr。）

**消融(Table 3，nuScenes)**：

| 变体 | Instr. Follow | Subj. Cons. | Bg. Cons. | Motion | Aesthetic | Imaging |
|---|---|---|---|---|---|---|
| w/o AFG | 0.545 | 0.841 | 0.892 | 0.957 | 0.492 | 0.703 |
| w/o noise（无噪声记忆） | 0.359 | 0.903 | 0.927 | 0.979 | 0.523 | 0.739 |
| cross-attn. adapter（替换 ACT-Adapter） | 0.642 | 0.926 | 0.903 | 0.948 | 0.512 | 0.694 |
| w/o MoAE | 0.651 | 0.930 | 0.941 | 0.975 | 0.520 | 0.727 |
| **Astra（完整）** | **0.669** | **0.939** | **0.945** | **0.989** | **0.531** | **0.747** |

**去掉噪声记忆** 指令跟随从 0.669 暴跌到 0.359，是最大掉点，直接佐证 noise-as-mask 是动作响应的关键；去掉 AFG 掉到 0.545；用 cross-attention adapter 替换 ACT-Adapter 也逊于本文方案（0.642 vs 0.669）。

**跨场景泛化(Table D，CityWalker 完全未见的 100 个 held-out 场景)**：Astra 六项全优——Instr. Follow 0.641、Subj. 0.948、Bg. 0.944、Motion 0.983、Aesthetic 0.554、Imaging 0.695，全面高于 Yume（0.619/0.933/0.927/0.972/0.511/0.628）等基线，说明强性能不是小测试集的假象。附录还展示了对室内、动漫、Minecraft 等分布外场景的动作可控生成。

## 四、局限性

- **推理效率**：基于扩散的 AR 逐块生成，每帧需多步去噪，实时部署困难，难以直接用于低延迟场景（在线控制、交互式机器人）。作者把蒸馏/师生压缩列为未来方向。
- **动作对齐评测依赖人评**：指令跟随主要靠 20 名用户主观打分，MegaSaM 之类的自动位姿估计在复杂几何/快速运动下误差大，客观化程度有限。
- **交互时长仍受限**：Table C 自报交互时域约 8–10 秒，与 YUME 同级，尚未做到无限时长。
- **动作模态需预定义投影器**：MoAE 的每个新模态都要设计专属 projector 与训练数据，扩展新动作类型并非零成本。
- **Table A 数值存在排版不一致**（指令跟随列疑与 Table 2 互换),给严格复现带来轻微困扰。

## 五、评价与展望

**优点**：（1）工程上"以小博大"——冻结 1.3B 骨干、只训 366M 的 adapter+self-attn，就把被动 T2V 改造成可交互世界模型，参数/算力开销在同类里最低；单位矩阵初始化的 ACT-Adapter 是把预训练权重"温柔接管"的干净做法。（2）**visual inertia 的命名与 noise-as-mask 的对策** 是本文最有洞见之处：把"历史太干净会压制动作"这一现象量化(Figure C),并用与扩散噪声解耦的腐蚀噪声解决——既无需改架构、又能推理时用干净历史，消融显示它是响应灵敏度的第一功臣。（3）MoAE 让单一模型统一相机/机器人/键鼠动作,是走向"通用"交互世界模型的务实一步。

**与公开工作的关系**：Astra 属于"AR+扩散"混合世界模型这一密集赛道。相较 StreamingT2V、MAGI-1（Teng et al. 2025）等纯视频续写，它强调动作条件；相较把动作做成 cross-attention 的 MatrixGame（He et al. 2025）与走 masked-token 的 WorldDreamer/iVideoGPT，本文用逐元素相加+identity adapter 更轻；相较 Yume（同样基于 Wan、走 masked-video-diffusion 与随机 mask token），Astra 的差异在 noise-as-mask（腐蚀而非丢弃）与 MoAE 多模态路由,消融也直接对比了这两点并胜出。与专注机器人的 WorldVLA、Vid2World 相比,Astra 的机器人只是五个模态之一,操作精度未做 VLA 式的闭环验证。

**开放问题与可能改进**：（1）**实时化** 是硬约束,少步/一步蒸馏、consistency/flow-map 蒸馏是自然的下一步；（2）动作对齐评测应引入更可靠的自动 3D 指标或统一 protocol,减少对主观人评的依赖,也便于与闭源的 Genie-3 类做可比;（3）**长时误差累积**：8–10 秒仍偏短,noisy memory + 首帧保留的压缩策略能否扩到分钟级值得验证;（4）机器人模态目前是"视频预测"层面,若要服务下游策略学习,需要补充操作成功率、动力学一致性等具身指标,而非只看视觉质量;（5）MoAE 的路由目前每步仅激活 1 个专家,跨模态协同(如相机+机器人同时动)的建模能力有待更细的分析。总体上,Astra 是一份工程扎实、观察敏锐、结果扎实的交互世界模型工作,noise-as-mask 与轻量 adapter 的组合尤其值得后续借鉴。

## 参考

1. Team Wan et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314, 2025.（Astra 的预训练骨干）
2. X. Mao et al. *Yume: An Interactive World Generation Model.* arXiv:2507.17744, 2025.（同基于 Wan 的 masked-video-diffusion 世界模型，主要对比对象）
3. X. Peng, C. Peng et al. *Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model.* arXiv:2508.13009, 2025.（cross-attention 动作条件对比基线）
4. L. Zhang and M. Agrawala. *Packing Input Frame Context in Next-Frame Prediction Models for Video Generation.* arXiv:2504.12626, 2025.（历史压缩/首帧保留思路来源）
5. A. Bar, G. Zhou, D. Tran, T. Darrell, Y. LeCun. *Navigation World Models.* CVPR 2025.（NWM，动作对齐对比基线）
