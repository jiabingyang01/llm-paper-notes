# Efficient-WAM：低成本未来想象的十亿参数世界-动作模型

> **论文**：*Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination*
>
> **作者**：Jiajun Li*、Tiecheng Guo*、Yifan Ye*（共同一作）、Rongyu Zhang、Xiaowei Chi（项目负责人）等，通讯作者 Shanghang Zhang
>
> **机构**：The University of Hong Kong、Peking University、Muka Robotics、Institute of Automation, Chinese Academy of Sciences、Nanjing University
>
> **发布时间**：2026 年 06 月（arXiv 2606.10040）
>
> **发表状态**：未录用（预印本），项目主页 efficientwam.github.io
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.10040) | [PDF](https://arxiv.org/pdf/2606.10040)
>
> **分类标签**：`World-Action Models` `具身操作` `视频扩散蒸馏` `推理加速` `Mixture-of-Transformers`

---

## 一句话总结

Efficient-WAM 用"结构剪枝蒸馏出的紧凑视频专家 + 低分辨率未来 latent + 非对称视频-动作去噪步数"三板斧,把 World-Action Model 的未来想象分支压到 1B 参数、每 chunk 98 ms,在 RoboTwin 2.0(86.7% clean)和真机 Astribot S1(66.25% 平均成功率)上追平甚至超过 5B/8B 的重量级 WAM 基线,相对 Motus 实现约 30 倍推理加速。

## 一、问题与动机

World-Action Model(WAM)通过把未来视觉预测(future visual prediction)与动作生成(action generation)耦合在一起,将物理动力学先验注入策略,是近年具身控制的一个有前景范式。但现有主流 WAM(如基于 WAN-2.2-5B 等大型视频生成器构建的系统)普遍相信"预测越逼真、动作越好",因而依赖大模型、密集 visual token 和多步迭代去噪来生成照片级真实的未来帧——这带来高延迟、高算力门槛,难以做到真实机器人上的实时闭环部署。

作者观察到两条已有证据挑战了这一假设:VPP 表明即便把去噪压缩到单步,动作生成依然有效;Fast-WAM 表明推理时完全跳过显式未来生成,WAM 仍能保持较好效果。由此作者提出核心假设:策略真正需要的不是逼真像素,而是保留任务相关几何(geometry)、运动趋势(motion tendencies)、接触线索(contact cues)的"未来表征"。基于此提出 **action-centric future imagination** 设计原则,并将 WAM 效率问题重新表述为一个建模问题——不是去优化视频生成质量,而是系统性地压缩产生未来表征所需的三个可控因子:模型规模、token 数量、去噪步数。

## 二、核心方法

### 2.1 问题形式化与成本分解

WAM 显式建模未来场景演化与控制动作的联合分布。给定当前观测 $o$、语言指令 $l$、机器人状态 $s$,联合预测目标为 $p(z^v, a_{1:H} \mid o,l,s)$,其中 $z^v$ 为显式的未来视觉 latent,$a_{1:H}$ 为动作 chunk。Efficient-WAM 将其分解为"未来想象过程"与"以未来为条件的动作生成过程":

$$p(z^v, a_{1:H} \mid o,l,s) = \underbrace{p_\phi(z^v \mid o,l)}_{\text{视频分支}} \cdot \underbrace{p_\theta(a_{1:H} \mid o,l,s,z^v)}_{\text{动作分支}}$$

**大白话说**：先想象未来会发生什么(视频分支),再基于这个想象决定怎么做(动作分支)。传统做法是把想象做得越逼真越好,Efficient-WAM 反其道而行,只问想象是否"够用"。

进一步把视频侧的计算成本抽象为三个可控因子的函数:

$$\mathcal{C}_{\text{video}} = \mathcal{F}_{\text{video}}(\mathcal{M}_v,\, N^v_{\text{tok}}(r_v),\, K_v)$$

其中 $\mathcal{M}_v$ 是视频模型规模,$N^v_{\text{tok}}(r_v)$ 是由未来预测分辨率 $r_v$ 决定的视觉 token 数,$K_v$ 是视频去噪步数。**大白话说**：视频分支到底烧多少算力,由"模型多大、token 多密、迭代多少步"三个旋钮共同决定,Efficient-WAM 把这三个旋钮同时往下拧。

### 2.2 紧凑架构与世界知识迁移

采用 Mixture-of-Transformers(MoT)架构,让轻量视频专家(video expert)与专职动作专家(action expert)逐层耦合交互,而非共享一套统一权重。视频专家并非随机初始化训练,而是通过结构化剪枝(structured pruning)从 WAN-2.2-5B 蒸馏而来:

- **深度方向**:采用 12 层 WAN 骨架,通过 layer slicing 直接抽取教师模型的第 [1, 2, 4, 6, 8, 11, 14, 17, 20, 23, 26, 30] 层权重作为学生初始化(而非随机初始化),连带抽取对应的 attention head、FFN 通道、embedding、调制参数(modulation)和输出头。
- **宽度方向**:压缩到 2048 隐藏维度、8192 FFN 维度、16 个注意力头,压缩后视频专家约 0.8B 参数。
- **蒸馏损失**:在标准视频 flow-matching 目标之外,叠加教师引导的蒸馏损失,包含隐藏状态对齐 $\mathcal{L}_{hid}$(学生/教师在对齐层的 256 维投影特征做余弦相似度)和时序动态对齐 $\mathcal{L}_{mot}$(逐帧空间平均后取帧间差分,对齐运动线索的余弦相似度):

$$\mathcal{L}_{hid} = \frac{1}{|\mathcal{A}_{hid}|}\sum_{l \in \mathcal{A}_{hid}} \mathbb{E}_n\Big[1-\cos\big(\tilde h^{l}_{s,n}, \tilde h^{r(l)}_{t,n}\big)\Big], \qquad \mathcal{L}_{mot} = \frac{1}{|\mathcal{A}_{mot}|}\sum_{l \in \mathcal{A}_{mot}} \mathbb{E}_f\Big[1-\cos\big(\Delta\bar h^{l}_{s,f}, \Delta\bar h^{r(l)}_{t,f}\big)\Big]$$

**大白话说**：不是让学生从零学会拍视频,而是先把教师大模型里"哪些通道、哪些层负责几何/动力学/接触"这套先验抄过来,再顺带教它模仿教师特征"怎么随时间变化",这样压缩后的模型仍然继承物理直觉而不是丢成一张白纸。

任务指令通过 cross-attention 注入,机器人状态和带噪动作被编入 action token;每个 MoT 层内,action token 会 attend 视频 token 提取未来上下文再映射回动作流。正式的动作训练阶段中,压缩后的视频专家被冻结,只优化动作专家。

### 2.3 多尺度视频-latent 布局(粗粒度未来预测)

标准 WAM 通常以统一分辨率预测未来帧,把算力浪费在与动作无关的视觉细节上。Efficient-WAM 采用多尺度视频-latent 布局:当前观测经 VAE 编码为高分辨率条件 token(如 384×320),而目标未来帧先在空间上下采样到更小的"未来视频尺寸"(如 192×160)再送入 VAE 编码,得到 token 稀疏的低分辨率未来 latent;两组 latent 分别 patchify 后拼接成统一视觉上下文,动作专家在此多尺度 token 序列上做联合视频-动作注意力——保留当前状态的高保真空间细节,同时把低分辨率未来 latent 仅作为粗粒度动态引导。

### 2.4 非对称视频-动作去噪(training-free)

常规生成式 WAM 中视频分支和动作分支共享同一套迭代去噪调度,但二者收敛速度不同:动作生成需要多步精细采样才能得到安全可执行轨迹,而未来视频只需提供粗粒度动态上下文——物体几何、接触边界等全局结构线索往往在去噪的最初几步就已出现,继续迭代生成照片级纹理是计算浪费。

Efficient-WAM 利用这一差异,在推理时给动作分支分配更大去噪预算(如 5-10 步),视频分支只用远少于此的步数刷新(如仅初始 2 步),两次视频刷新之间复用缓存的视频特征持续引导动作精化。该技巧无需重新训练(training-free)。

### 2.5 训练目标(三阶段)

统一采用条件流匹配(conditional flow matching):目标数据 $\mathbf{x}_1$(干净未来视频 latent $\mathbf{x}_1^v$ 或动作 chunk $\mathbf{x}_1^a$),$\mathbf{x}_0 \sim \mathcal{N}(0,I)$,插值路径 $\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1$,目标速度 $\mathbf{u}_t=\mathbf{x}_1-\mathbf{x}_0$:

$$\mathcal{L}_{FM} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\big[\|f(\mathbf{x}_t,t;c)-\mathbf{u}_t\|_2^2\big]$$

训练分三阶段:**Stage 1** 仅训练视频专家,$\mathcal{L}_{\text{stage-1}} = \mathcal{L}_{\text{video-FM}} + \lambda_{\text{dist}}(\mathcal{L}_{hid}+\mathcal{L}_{mot})$,蒸馏权重 $\lambda_{\text{dist}}$ 从 0.2 逐步衰减到 0.1 再到 0;**Stage 2** 接入动作专家、冻结视频分支,$\mathcal{L}_{\text{stage-2}} = \mathcal{L}_{\text{action-FM}} + \lambda_v \mathcal{L}_{\text{video-FM}}$;**Stage 3** 端到端联合精化两个专家,视频/动作分支各自独立采样去噪时间步 $t_v, t_a \in [0,1]$($\mathbf{x}^a_{t_a}=(1-t_a)\epsilon_a+t_a\mathbf{a}$,$\mathbf{x}^v_{t_v}=(1-t_v)\epsilon_v+t_v\mathbf{z}^v$),联合目标为 $\mathcal{L}_{\text{joint}} = \lambda_a\|f_a(\mathbf{x}^a_{t_a})-\mathbf{u}^a\|_2^2 + \lambda_v\|f_v(\mathbf{x}^v_{t_v})-\mathbf{u}^v\|_2^2$。训练采用 AdamW、cosine 学习率调度、bf16 混合精度,每阶段仿真约 2.5 epoch、真机约 5 epoch。

论文同时给出两个模型变体做消融隔离:**Efficient-WAM**(结构基线,保留高分辨率未来预测 + 对称去噪,只体现紧凑视频专家的收益,建立 1B 架构的能力上界)与 **Efficient-WAM-RT**(叠加低分辨率未来 latent + 非对称去噪,面向真实部署的完全优化版本)。

## 三、实验结果

### 3.1 仿真:RoboTwin 2.0(50 个双臂任务,clean/随机化两种视觉设置,各 100 trials/任务)

| 方法 | 类型 | 参数量 | Clean (%) | Random (%) |
|---|---|---|---|---|
| $\pi_0$ | VLA | 3.3B | 65.9 | 58.4 |
| StarVLA-$\alpha$ | VLA | 2B | 76.8 | 79.1 |
| $\pi_{0.5}$ | VLA | 3.3B | 82.7 | 76.8 |
| ABot-M0 | VLA | 4.2B | 86.1 | 85.1 |
| LingBot-VLA | VLA | 4B | 86.5 | 85.3 |
| UWM | WAM | 5B | 81.7 | 78.6 |
| GigaWorld-Policy | WAM | 5B | 86.4 | 85.0 |
| Motus | WAM | 8B | **88.7** | **87.0** |
| Efficient-WAM | WAM | **1B** | 86.7 | 85.7 |
| Efficient-WAM-RT | WAM | **1B** | 83.1 | 82.0 |

仅 1B 参数的 Efficient-WAM 达到 86.7% clean 成功率,超过 4B 的 LingBot-VLA 与 5B 的 GigaWorld-Policy,仅比 8B 的 Motus 低 2.0 个百分点;完全面向部署优化的 Efficient-WAM-RT 在牺牲部分精度后仍优于 $\pi_0$、StarVLA-$\alpha$ 等多个重量级基线。

### 3.2 真机:Astribot S1(4 个任务,各 100 条人类示教、20 trials 评测,单任务专用策略)

| 任务/指标 | $\pi_{0.5}$ | Motus | Efficient-WAM-RT(Ours) |
|---|---|---|---|
| pipette-tray grasping | 100.0 | 85.0 | 95.0 |
| reagent-bottle transfer | 75.0 | 80.0 | 75.0 |
| LEGO color sorting | 30.0 | 65.0 | 65.0 |
| pen uncapping | 10.0 | 25.0 | 30.0 |
| 平均成功率 | 53.75 | 63.75 | **66.25** |
| 单 chunk 延迟(ms) | 113.0 | 3215.0 | **98.0** |
| 单步延迟(ms) | 7.1 | 200.9 | **6.1** |

Efficient-WAM-RT 平均成功率(66.25%)略高于重量级 WAM 基线 Motus(63.75%),同时单 chunk 延迟仅 98 ms,相对 Motus 提速约 32 倍;完成一次成功 trial 平均约 30 秒,而 Motus 因迟滞的启停式执行需约 2 分钟。真机失败模式归纳为三类:细粒度空间对不准(fine spatial misalignment)、长时序场景覆盖不全(incomplete scene coverage)、接触碰撞失败(contact and collision failure)。

### 3.3 消融研究(RoboTwin,20 rollouts/任务)

**紧凑视频专家(Table 3a)**:随机初始化的压缩模型(Compact-random)成功率骤降至 69/68%;仅做 layer slicing 不蒸馏(Compact-sliced)恢复到 82/81%;加上教师引导蒸馏的完整 Efficient-WAM 达到 87/86%,逼近未压缩的 Full WAN(5B,86.4/85.5%),同时延迟从 2013 ms 降到 430 ms。

**未来分辨率(Table 3b)**:token 数从 240(高分辨率)降到 60(低分辨率),延迟从 430 ms 降到 377 ms,成功率仅从 87/86% 降到 83/82%。

**非对称去噪(Fig. 4 / Table 7)**:固定动作去噪步数为 10,视频去噪步数从 [10,10] 降到 [2,10] 时,延迟从 430 ms 降至 139 ms(3.1 倍加速),成功率仅从 87.1% 降到 86.3%;但极端压缩(如仅 1 步视频去噪 [1,10],或同时压缩动作步数 [1,1])会显著掉点(降至 77.2%-79.3%),说明存在一个"安全解耦"的最小步数阈值。

**系统级延迟分解(Table 5)**:Full WAN 2013 ms → 紧凑架构 430 ms → 加低分辨率未来 377 ms → 加非对称去噪 139 ms → 真机硬件优化部署(RTX 4090,消除仿真专属开销)后的 Efficient-WAM-RT 仅 **98 ms**,相对标准 WAM 实现约 30 倍加速。

## 四、局限性

论文明确列出两点(第 6 节):

1. **细粒度任务的权衡**:预测粗粒度、低分辨率未来 latent 会牺牲视觉保真度,对宏观操作(抓取、搬运、分拣)有效,但对需要极致像素级精度的微操作(如穿线插入)可能仍需更高分辨率的视觉引导。
2. **静态推理调度**:Efficient-WAM-RT 当前使用固定的非对称去噪调度(如 [2, 10]),未根据任务不确定性或物理复杂度动态调整;作者提出未来可探索动态计算分配,仅在任务动态不确定或复杂时才增加视频去噪预算。

此外,从实验设计看还可以补充的局限:真机评测仅覆盖 4 个任务、每任务 20 trials,规模有限;真机部分未做随机化视觉设置下的鲁棒性测试(仿真部分才有 clean/random 对照);蒸馏依赖特定教师模型(WAN-2.2-5B)和人工选定的层抽取方案([1,2,4,...,30]),对其他视频基座模型的可迁移性未验证。

## 五、评价与展望

**优点**：本文的核心贡献并非又一个更大更强的 WAM,而是一个清晰、可复现的"效率优先"设计原则——把 WAM 推理成本显式分解为模型规模、token 数、去噪步数三个正交因子,并逐一给出对应的压缩手段(结构化蒸馏剪枝、多尺度 latent、非对称去噪),三者可独立叠加、效果可加性验证清楚(Table 5 的逐步延迟分解是全文最有说服力的证据链)。与 Fast-WAM(推理时完全跳过显式未来生成)、VPP(单步去噪即可)等同期工作相比,Efficient-WAM 选择保留一个"退化但仍可用"的视频分支,并用消融证明这种粗粒度未来想象仍能提供 Fast-WAM 式方案之外的额外控制增益(尤其在长时序、多阶段任务上),是对"WAM 是否需要照片级真实感"这一开放问题的一次较为系统的实证回答。

**与其他公开工作的关系**：论文将 Motus(8B)、GigaWorld-Policy(5B)、UWM(5B)作为同类 WAM 基线,将 $\pi_0$/$\pi_{0.5}$/LingBot-VLA/ABot-M0 等 VLA 方法作为跨范式对照,覆盖较为全面;知识蒸馏来源选用 WAN-2.2-5B 而非从零训练视频生成器,这一"从大型开源视频基座抽取世界知识再做动作定制"的思路与 Being-H0.7、GigaWorld-Policy 等工作方向一致,说明"复用视频生成基座 + 轻量化改造"正成为 WAM 效率化的主流路线。

**开放问题与可能的改进方向**：(1)非对称去噪调度目前是手工设定的固定步数配比([2,10]),论文自己也承认这是局限——一个自适应/可学习的去噪步数分配器(依据场景不确定性或任务阶段动态调整 $[T_v, T_a]$)是自然的下一步,可能进一步扩大"精度-延迟"帕累托前沿;(2)结构化蒸馏中的层选择([1,2,4,6,8,11,14,17,20,23,26,30])依赖人工经验,是否存在更系统的层重要性评估方法(如结合梯度/激活敏感度的自动化剪枝准则)值得探索;(3)真机实验规模偏小(4 任务)且未覆盖高精度插孔类任务,论文自己指出的"细粒度任务权衡"局限尚缺乏专门实验量化这个边界具体在哪里;(4)本文的三个压缩维度(模型、token、步数)是否可以进一步与策略侧的高效化技术(如动作专家的量化、early-exit、consistency distillation,论文相关工作中也提到但视为"互补方向"未做整合)联合优化,是一个尚待验证的系统级问题。总体而言,这是一篇工程实证扎实、消融充分、对 WAM 部署瓶颈给出明确诊断和可迁移解法的论文,其"action-centric future imagination"原则具备较好的推广潜力。

## 参考

- Hu et al. *Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations*, ICML 2025.(VPP,证明单步去噪仍可支撑有效动作生成)
- Yuan et al. *Fast-WAM: Do World Action Models Need Test-Time Future Imagination?*, arXiv:2603.16666, 2026.(推理时跳过显式未来生成)
- Bi et al. *Motus: A Unified Latent Action World Model*, arXiv:2512.13030, 2025.(本文最强的 8B WAM 对照基线)
- Ye et al. *GigaWorld-Policy: An Efficient Action-Centered World-Action Model*, arXiv:2603.17240, 2026.(同类效率化 WAM 路线)
- Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models*, arXiv:2503.20314, 2025.(本文蒸馏所用教师模型 WAN-2.2-5B 的来源)
