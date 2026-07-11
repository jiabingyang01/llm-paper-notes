# FiS-VLA：把"快系统"嵌进"慢系统"——统一快速操作与慢速推理的双系统基础模型

> **论文**：*Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning*
>
> **作者**：Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, Pheng-Ann Heng
>
> **机构**：The Chinese University of Hong Kong；Peking University（State Key Laboratory of Multimedia Information Processing, School of Computer Science）；AI²Robotics；Beijing Academy of Artificial Intelligence (BAAI)
>
> **发布时间**：2025 年 06 月（arXiv 2506.01953）
>
> **发表状态**：未录用（预印本，论文页脚标注 "Preprint. Under review."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.01953) | [PDF](https://arxiv.org/pdf/2506.01953)
>
> **分类标签**：`VLA` `双系统架构（System1/System2）` `扩散策略` `动作分块` `3D点云条件化` `具身操作预训练`

---

## 一句话总结

FiS-VLA 不再像以往双系统 VLA 那样给 VLM（System 2）外挂一个独立训练的轻量策略头当 System 1,而是直接把预训练 LLM 的**最后几个 transformer block 复用为 System 1**,让快慢两个系统共享同一套参数、异步运行（频率比 1:4）,在 RLBench 仿真上以 21.9 Hz（8 步动作分块下理论可达 117.7 Hz）取得 69% 平均成功率,比 CogACT 高 8 个百分点、比 π0 高 14 个百分点。

## 一、问题与动机

机器人操作基础模型面临"泛化能力"与"执行效率"的两难:借助互联网规模预训练 VLM 的 VLA 模型(如 OpenVLA、RT-2)语义理解强,但自回归逐 token 解码动作导致控制频率很低,难以支撑闭环实时控制。受 Kahneman 双系统理论启发,近期工作(HiRT、"Towards synergistic..."双系统方法、Figure 的 Helix 等)把 VLM 作为慢速、高层的 System 2,再另接一个轻量策略模型作为快速的 System 1 执行动作。但作者指出这类方案有一个根本缺陷:System 1 是**独立注入、从零训练**的模块,只能消费 System 2 输出的特征,无法直接继承 VLM 的互联网级预训练知识,也无法参与推理过程本身——两个系统始终是"两个模型"而非"一个模型的两种模式"。

论文提出的核心问题是:"如果 VLM 是机器人的'大脑',能否让它同时承载 System 1 和 System 2 两种过程,实现推理与执行的协同?"

## 二、核心方法

### 2.1 总体架构:在 System 2 内部"长出"System 1

FiS-VLA 基于 Prismatic VLM 初始化,视觉编码器并联使用 SigLIP（$f^{\text{SigLIP}}\in\mathbb R^{N_v\times 1024}$）与 DINOv2（$f^{\text{DINO}}\in\mathbb R^{N_v\times 1152}$）并在通道维拼接,兼顾高层语义与局部空间细节;LLM 主干是 LLaMA2-7B(32 层 transformer block)。**关键设计**:保留完整 32 层 LLM 作为 System 2 做自回归推理,同时把**最后若干个 transformer block 复用/重接为 System 1** 的扩散动作生成模块——System 1 不是外挂头,而是同一模型末端被赋予了第二种"角色"，构成"慢系统里长出的快系统"(fast system within slow system)。

用大白话说:System 2 好比一位深思熟虑的大脑,把对场景和指令的理解总结成一段"中间层特征摘要";System 1 就是这颗大脑最后几层神经元,被同时训练成既能读懂这份摘要、又能直接把它翻译成手部的连续动作指令,免去了"翻译"给另一个独立小脑的信息损失。

### 2.2 异步频率与异构输入

两个系统被明确设计成不同的角色:
- **System 2(慢)**:低频运行(每 $n$ 步更新一次),输入为 2D 图像 + 语言指令,输出一个中间 block(实验中为第 30 层)的隐藏特征,作为对未来 $H$ 个时间步的高层语义"指引"(latent condition)。
- **System 1(快)**:高频运行(每步都执行),输入为最新的 2D 图像、机器人本体状态(proprioceptive state)、**3D 点云**,以及 System 2 周期性提供的隐藏特征,联合去噪生成动作。3D 点云先经由轻量 3D tokenizer(最远点采样降采样 + KNN 局部聚合 + 线性投影)转成 token,再送入与 System 2 共享的视觉编码器,从而以极小的参数代价把 3D 几何信息投影进 LLM 的语义嵌入空间。

训练时用"异步采样"(每隔 $n$ 步才刷新 System 2 的条件特征)来模拟真实推理时的频率差,迫使 System 1 在同一份高层指引下保持多步时间一致性。消融实验显示 1:4 的频率比最优(见下文表格)。

### 2.3 训练目标:扩散 + 自回归的双感知协同训练

FiS-VLA 把机器人策略学习形式化为在异质观测 $o_{t-1}$(状态、多视角图像、点云)和语言指令 $l$ 条件下最大化动作序列似然:

$$\max_{\theta}\ \mathbb E_{(a_{t:t+H},o_{t-1},l)\sim\mathcal D}\left[\log \pi_\theta(a_{t:t+H}\mid o_{t-1},l)\right]$$

**System 1(执行)** 采用扩散建模:给定动作序列 $\tilde a$,在时刻 $\tau\sim\mathcal U(1,T)$ 注入高斯噪声,前向过程为

$$\tilde a_\tau=\sqrt{\beta_\tau}\tilde a+\sqrt{1-\beta_\tau}\eta,\qquad \eta\sim\mathcal N(0,I)$$

训练目标是预测噪声:

$$\mathcal L_{\text{fast}}=\mathbb E_{\tau,\tilde a,\eta}\Big[\big\|\eta-\pi_{\theta_f}\big(\sqrt{\beta_\tau}\tilde a+\sqrt{1-\beta_\tau}\eta,\,c,\,\tau\big)\big\|^2\Big]\tag{1}$$

其中条件 $c$ 由 System 2 的低频潜特征与 System 1 的高频输入共同组成。用大白话说:这就是标准 DDPM 去噪目标——模型学会"猜出被加进动作里的噪声长什么样",从而在推理时反复去噪把随机向量变成一段可执行动作。

**System 2(推理)** 若只用扩散目标单独训练嵌入其中的 System 1,容易让整条 LLM 主干"灾难性遗忘"原本的自回归推理能力。为此论文额外保留 next-token 预测目标,监督信号可以是离散动作 token 或语言子目标(subgoal)：

$$\mathcal L_{\text{slow}}=-\sum_{i=1}^{D_t}\log P\big(\hat a_i \mid \text{context},\theta\big)\tag{2}$$

最终把两者相加,构成"双感知协同训练"(dual-aware co-training)目标:

$$\mathcal L_{\text{FiS-VLA}}=\mathcal L_{\text{fast}}+\mathcal L_{\text{slow}}\tag{3}$$

直觉上,这相当于给共享参数的末端 block 同时布置两门"课程":一门是连续动作去噪,一门是离散语言/动作续写,逼着它既保留 VLM 原有的语义理解,又学会精细的低延迟控制。消融显示去掉 $\mathcal L_{\text{slow}}$ 后 RLBench 平均成功率从 69% 掉到 62%。

### 2.4 预训练与数据

FiS-VLA 先在整合自 37 个开源数据集(Open X-Embodiment、DROID、RoboMind 等)、共 **86 万条轨迹 / 3600 万帧** 的混合语料上训练 5 个 epoch(权重最高的几项为 DROID 14.2%、Kuka 10.5%、BridgeData V2 9.3%、Fractal 6.8%、Maniskill 7.5%、BC-Z 6.3%、FMB 6.0%);预训练阶段两个系统都只用单视角图像作观测,System 2 用离散动作序列做自回归监督。随后在自采集的真实世界(AgileX、AlphaBot 双臂平台)与仿真(RLBench)数据上微调,微调阶段为 System 2 额外补充人工标注的子任务语言计划。

## 三、实验结果

**仿真(RLBench,10 项 Franka Panda 单臂任务,前视相机 RGB+点云)**,对比 ManipLLM、OpenVLA、π0、CogACT:

| 模型 | 参数规模 | 平均成功率 | 控制频率 |
|---|---|---|---|
| ManipLLM | — | 0.38 ± 0.04 | 2.2 Hz |
| OpenVLA | — | 0.40 ± 0.04 | 6.3 Hz |
| π0 | 2.6B LLM | 0.55 ± 0.03 | 13.8 Hz |
| CogACT | 7B LLM | 0.61 ± 0.03 | 9.8 Hz |
| **FiS-VLA** | 7B LLM | **0.69 ± 0.03** | **21.9 Hz** |

FiS-VLA 在 10 项任务中 8 项取得最优,例如 Close box、Close laptop lid 均为 1.00,Toilet seat down 0.95,Close fridge 0.90;最弱的是 Water plants(0.20)。动作分块从 1 步扩到 8 步时,理论控制频率在 1:4 频率比下最高可达 **117.7 Hz**,而成功率基本保持稳定(0.69→0.66,轻微波动)。

**真实世界(Table 2,双臂平台,人工判定成功率)**,对比 π0:

| 平台 | 任务数 | π0 平均成功率 | FiS-VLA 平均成功率 |
|---|---|---|---|
| AgileX(末端位姿控制) | 4 | 0.59 | **0.68** |
| AlphaBot(关节位置控制) | 4 | 0.61 | **0.74** |

其中 AgileX 上 *Place bottles at rack* 任务 FiS-VLA 0.70 对 π0 0.55;AlphaBot 上 *Fold towel and put*(可形变物体折叠)提升最明显,FiS-VLA 0.60 对 π0 0.40。

**泛化实验(Table 3,未见物体 / 复杂背景 / 光照扰动)**:在 AgileX 的 Place Bottles at Rack 与 AlphaBot 的 Pick Bowl and Place Object 两个任务上,相对原始成功率的下降幅度:

| 扰动类型 | FiS-VLA(AgileX) | π0(AgileX) | FiS-VLA(AlphaBot) | π0(AlphaBot) |
|---|---|---|---|---|
| 未见物体 | −21% | −27% | −19% | −38% |
| 复杂背景 | −29% | −36% | −25% | −38% |
| 光照变化 | 仍 > 50% 成功率 | 下降更明显 | 仍 > 50% 成功率 | 下降更明显 |

**消融要点(RLBench)**：(1)System 1 复用的共享 block 数从 1 增到 8,性能先升后趋于饱和,2 个 block 时已达最佳(0.69),说明少量共享层即可继承 VLM 预训练知识;(2)System 1 输入去掉点云或去掉点云+状态,平均成功率从 0.69 分别掉到 0.61 / 0.44 / 0.22(仅剩图像时最差),验证机器人状态与 3D 点云均有实质贡献;(3)慢快频率比从 1:1 到 1:8 扫描,**1:4 最优**(0.69),过密(1:1)或过疏(1:8)都更差。

## 四、局限性

- **静态配置,非自适应**:作者在结论中明确指出,System 1 复用的共享 block 数量、以及两系统间的协同频率是**固定超参**,依赖离线消融搜索得到(在 RLBench 10 任务上搜索),并未根据任务难度或环境复杂度动态调整,作者将其列为未来工作方向。
- **依赖单视角深度图构造点云**:3D 输入来自单视角深度图反投影,深度传感/估计噪声会直接传导到 System 1 的空间条件,论文未评估点云质量退化下的鲁棒性。
- **未探索真正的并行硬件部署**:论文承认受限于机器人硬件不支持双 GPU 部署两个系统(不同于 Helix 的分离 GPU 方案),117.7 Hz 是理论控制频率而非双卡并行下的实测上限,论文聚焦的是"最优协同频率比"这一研究问题而非工程极限吞吐。
- **评测规模有限**:仿真 10 项任务、真实世界仅 2 个平台各 4 项任务,尚未在更大规模的跨本体通用操作基准(如大规模 SimplerEnv 式评测)上验证。
- **扩散推理步数未披露**:论文未报告 System 1 实际推理时使用的去噪步数,这直接影响真实延迟,与"117.7 Hz"这一理论频率之间的关系未完全展开。

## 五、评价与展望

FiS-VLA 的核心贡献是把"双系统"从**架构层面的两个模型**收敛为**同一模型内部的两种运行模式**——这是对 HiRT、"Towards synergistic, generalized, and efficient dual-system"、GR00T N1、Helix 等此前双系统 VLA 路线的一个针对性改进:此前这些工作的 System 1 都是新引入、从零训练的轻量策略头,只能被动消费 System 2 输出的特征,天然割裂了"理解"与"执行"之间的表示空间;FiS-VLA 通过直接复用 LLM 尾部若干 block 并联合两种损失协同训练,让执行模块共享推理模块的预训练先验,这与近期 HybridVLA(在单一模型内部同步融合扩散和自回归两种生成方式)属于同一脉络下的不同解法——HybridVLA 强调的是同频率下生成机制的融合,FiS-VLA 强调的是异步频率下角色的复用,两者互补而非重复。

从结果看,FiS-VLA 用 7B 主干做到了比 2.6B 的 π0 更高的控制频率(21.9 Hz vs 13.8 Hz),说明"参数共享+异步调度"比单纯缩小模型规模更能兼顾精度与速度,这是一个值得后续工作借鉴的架构范式,而不只是又一次"堆参数换性能"。但论文的比较对象止步于 π0 与 CogACT,并未与同期更晚近的、同样采用共享/复用式双系统设计的工作(如更晚提出的 π0.5、GR00T N1 的完整数字)做直接头对头对比,其架构优势的边界仍有待更大规模基准检验。

开放问题包括:(1)共享 block 数与频率比是否存在与任务复杂度、动作维度相关的解析规律,可否用可学习门控替代离线网格搜索;(2)扩散 System 1 的多步去噪本身仍引入串行延迟,能否与一致性蒸馏/单步流匹配(如 π0 的 flow matching)结合以进一步压低延迟同时保留 FiS 的参数复用优势;(3)3D 点云条件化目前只验证了单视角深度输入,多视角融合或主动深度传感器下的表现尚不清楚;(4)动态调整 System1/2 协同频率(而非固定 1:4)对长时程、多阶段任务(如论文中已展示的 handover、pour water 等)的边际收益值得进一步量化。

## 参考

- CogACT (Li et al., 2024, arXiv:2411.19650) —— 本文最主要的同规模(7B)单模型基线
- π0 / π0.5 (Physical Intelligence, Black et al., arXiv:2410.24164 / 2025) —— flow-matching 动作生成基线,轻量 LLM 高频对照组
- HiRT (Zhang et al., 2024, arXiv:2410.05273) —— 早期分离式双系统 VLA(System1 为独立策略头)
- Helix (Figure AI, 2025) —— 双 GPU 部署的异步双系统人形机器人 VLA
- HybridVLA (Liu et al., 2025, arXiv:2503.10631) —— 同一模型内融合扩散与自回归生成的相关思路
