# VidMan：利用视频扩散模型中的隐式动力学实现高效机器人操作

> **论文**：*VidMan: Exploiting Implicit Dynamics from Video Diffusion Model for Effective Robot Manipulation*
>
> **作者**：Youpeng Wen, Junfan Lin（共同一作）, Yi Zhu, Jianhua Han, Hang Xu, Shen Zhao, Xiaodan Liang（通讯作者）et al.
>
> **机构**：中山大学深圳校区、鹏城实验室、华为诺亚方舟实验室
>
> **发布时间**：2024 年 11 月（arXiv 2411.09153）
>
> **发表状态**：NeurIPS 2024（第 38 届神经信息处理系统会议）
>
> 🔗 [arXiv](https://arxiv.org/abs/2411.09153) | [PDF](https://arxiv.org/pdf/2411.09153)
>
> **分类标签**：`视频扩散预训练` `双阶段训练` `隐式逆动力学` `扩散动作头` `Open-Sora` `CALVIN`

---

## 一句话总结

VidMan 借鉴神经科学"双过程理论"（System 1 快系统 / System 2 慢系统），把同一个视频扩散 Transformer（基于 Open-Sora 的 STDiT-XL/2）先在 Open X-Embodiment（OXE）上做未来帧预测式预训练获得动力学先验，再通过一个轻量的层级自注意力适配器（layer-wise adapter）把它单次前向直接改造成隐式逆动力学动作预测器，在 CALVIN 长时序基准上比 GR-1 相对提升 11.7%，在 OXE 小规模子数据集上精度提升超 9%。

## 一、问题与动机

现有可用的机器人轨迹数据规模远小于互联网视频数据，若不考虑视觉观测与动作之间的关系直接拟合数据，会导致数据利用效率低下。视频扩散生成模型（如 Open-Sora）已展现出对复杂物理动力学的强理解能力，作者希望借助这类模型学到的动力学先验来提升动作预测精度，但简单地把视频生成模型接一个独立的逆动力学模型头，存在两个问题：一是视频扩散的迭代去噪过程耗时，不适合高频机器人控制；二是并非所有像素都与动作预测相关，直接依赖生成的观测图像会引入不必要的偏差和时间成本。因此需要一种既能利用视频扩散先验、又能快速单次推理出动作的架构。

## 二、核心方法

VidMan 采用与双过程理论对应的两阶段训练范式，两阶段共享同一套 VDT（Video Diffusion Transformer，以 Open-Sora 的 STDiT-XL/2 为基座，12 层、16 头、隐藏维度 1152）参数。

**阶段一：Dynamics-aware Visionary Stage（类比 System 2）。** 用预训练视频自编码器（Stable Video Diffusion 的 VideoAutoencoderKL）把轨迹中的历史帧 $O_h$（$m$ 帧）和未来帧 $O_f$（$n$ 帧）编码为 $V_h, V_f$，前向扩散过程加噪得到 $V_s^k \leftarrow \varepsilon(V_s,\epsilon,k)$，$\epsilon\sim\mathcal N(0,1)$，扩散步 $k\in[1,K]$（$K=1000$）。历史帧 embedding 用零矩阵 $V^0$ 补齐形状后沿通道维与 $V_s^k$ 拼接，形成条件视觉输入 $V_c^k$，与 CLIP 文本编码的语言指令 $y$ 一起送入 VDT，训练目标是标准去噪扩散损失：

$$\mathcal{L}_v(\theta)=\mathbb{E}_{(V_c^k,y,k)}\left[\|\epsilon-\epsilon_\theta(V_c^k,y,k)\|_2^2\right].$$

用大白话说：这一步就是教模型"看历史帧+语言指令，学会想象未来会发生什么"，只用第三人称视角（因为多数机器人数据集只有第三人称相机，固定视角也能让模型专注于机械臂本身的状态转移）。

**阶段二：Dynamics-modulated Action Stage（类比 System 1）。** 直接把阶段一学到的 VDT 改造成隐式逆动力学模型，而不是另起一个独立模型：在 VDT 每一层后插入一个受 Flamingo 启发的层级自注意力适配器（自注意力 + 带 tanh 门控的 FFN），引入 $h$ 个可学习动作 token $Q_{\text{action}}$，与每层 VDT 输出特征做交叉融合，汇聚成动作 embedding。为了避免耗时的迭代去噪，这一阶段将扩散步固定为最大值 $k\leftarrow K$，即把 $V_s^K$ 直接设为纯高斯噪声（而不是真实历史帧的噪声版本），只做一次前向：

$$V_{\text{action}}=\epsilon_{(\theta,\phi_{\text{ada}})}(V_c^K,y,K,Q_{\text{action}}).$$

用大白话说：让模型"假装"自己还在做阶段一的去噪任务（喂纯噪声、报最大噪声步），但这次不真正生成图像，而是把每层里蒸馏出的动力学知识灌进几个动作 token 里，一次前向就拿到结果——省掉了迭代去噪的时间开销。消融实验（附录 Table 5）证实这个"纯噪声"技巧至关重要：纯噪声（MSE 4.8 / xyz 32.7 / angle 37.6）远优于不加噪声（11.2/6.6/0.4）和填零（12.1/2.3/6.5），因为纯噪声输入与阶段一的去噪训练分布对齐，能把预训练的"未来感知能力"蒸馏进动作预测。

最后用一个基于 Diffusion Policy 的扩散动作头 $\pi_{\phi_{\text{dec}}}$（最大扩散步 $L=100$）把动作 embedding 解码为 7 自由度末端位姿 + 夹爪状态：

$$\mathcal{L}_a(\theta,\phi_{\text{ada}},\phi_{\text{dec}})=\mathbb{E}_{(V_{\text{action}},l)}\left[\|\epsilon'-\pi_{\phi_{\text{dec}}}(\varepsilon(V_{\text{action}},\epsilon',l),l)\|_2^2\right].$$

该动作头相对 VDT 主体很小，计算和时间开销可忽略。阶段二联合使用 OXE（预测 12 步动作，额外接入腕部相机观测）和 CALVIN（预测 10 步动作，接入本体感知）数据训练。

## 三、关键结果

**CALVIN 零样本长时序（ABC→D，5 连续任务，Lang 表示只用语言标注子集训练）：**

| 方法 | 训练数据 | Avg. Len（满分 5） |
|---|---|---|
| MCIL | All | 0.67 |
| HULC | All | 0.90 |
| RT-1 | Lang | 0.90 |
| RoboFlamingo | Lang | 2.48 |
| SuSIE | All | 2.69 |
| GR-1 | Lang | 3.06 |
| 3D Diffuser Actor | Lang | 3.35 |
| **VidMan（本文）** | Lang | **3.42** |

VidMan 相对 GR-1 提升 11.7%，并小幅超过依赖深度信息的 3D Diffuser Actor（VidMan 只用 2D 视觉即可预训练大规模数据，深度图更难采集）。

**OXE 小规模子数据集离线评测（相对 Octo-base 的平均 xyz+angle 精度增益）：** Bridge +5.6%，Taco Play +2.6%，Cable Routing +9.9%，Autolab UR5 +9.0%；数据越稀缺的子集，两阶段预训练带来的收益越明显。

**关键消融（CALVIN Avg. Len）：** 两阶段解耦训练（阶段二只用动作损失）3.42，若阶段二联合训练视频生成损失和动作损失（co-train）则降至 2.70，验证了 System1/System2 解耦的必要性；预训练来源方面，无预训练 2.89 < 通用视频数据 Ego4d 预训练 3.29 < 机器人专属数据 OXE 预训练 3.42，说明具身域内视频预训练收益明显大于泛化视频；层级适配器方面，去掉适配器降至 1.54，冻结 VDT 只训适配器为 2.98，两者都放开训练达到最优 3.42（且冻结版本收敛更快）。帧采样间隔上，间隔为 3 时效果最好（CALVIN Avg. Len 3.42，间隔 1/2/4 分别为 2.24/2.82/3.03）。此外在 RLBench-100（18 任务，100 条演示）上，VidMan 平均成功率 67.4%，优于 PerAct（42.7%）、RVT（65.1%）、Act3D（65.1%）。

## 四、评价与展望

VidMan 的核心贡献是提出了一种参数共享的方式，把视频扩散预训练模型"直接"改造为单次前向的隐式逆动力学动作模型，避免了此前"先生成未来帧、再用独立逆动力学模型推动作"这种两阶段串行推理的延迟问题，这一设计相比 UniPi、AVDC 等显式依赖生成图像做规划的路线更适合高频真实机器人控制。"纯噪声输入对齐去噪分布"这一发现具有一定的普适价值，为后续把视频扩散骨干迁移为动作模型提供了一个简洁、可复现的技巧。

局限性方面，作者在附录中明确指出：（1）模型仅在 2D 视觉上运作，缺乏 3D 空间理解能力；（2）语言理解能力受限于 CLIP 文本编码器，不如现代 LLM 精细；（3）感知粒度较粗，将整张图像作为单一输入，未引入物体级的框/掩码等细粒度辅助信号。此外，方法仍强依赖第三人称固定视角的预训练设计，对视角变化和多相机融合的扩展性未做深入验证；两阶段训练流程（视频扩散预训练 100k 步 + 动作预训练 300k 步）计算成本较高（16×V100 训练数十小时），相对于直接端到端训练的 RT-1/Octo 增加了工程复杂度。与同期利用视频生成做机器人策略的工作（如 GR-1 的 GPT 式逐帧自回归预测、UniPi 的显式视频规划）相比，VidMan 的消融清楚地表明"扩散式多帧预测 + 隐式动力学蒸馏"优于"GPT 式单帧自回归"（VidMan-GPT 基线全面弱于 VidMan，见 Table 7），这为后续视频生成基座选型提供了一个直接的对照证据。开放问题包括：能否把该范式与显式 3D 表征（如深度、点云）结合以弥补 2D 局限；纯噪声蒸馏技巧能否推广到其他视频扩散骨干（如 Sora 类更大模型）以及是否存在更优的噪声调度策略；以及在更大规模、更多样化机器人本体上的可扩展性尚待验证。

## 参考

- Wu et al. *GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation*, arXiv:2312.13139.
- Team et al. *Octo: An Open-Source Generalist Robot Policy*, 2023.
- Padalkar et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, arXiv:2310.08864.
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv:2303.04137.
- Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
