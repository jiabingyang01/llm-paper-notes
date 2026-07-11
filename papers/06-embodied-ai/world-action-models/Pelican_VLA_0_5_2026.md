# Pelican-VLA 0.5：先关注后行动——BotTokens 瓶颈诱导的操作中心注意力与生成泛化

> **论文**：*Pelican-VLA 0.5: Attending Before Acting Benefits Generalization*
>
> **作者**：Zeyuan Ding, Wenhai Liu, Yang Xu, Jiayu Hu, Yinda Chen, Yi Zhang, Yong Dai, Jian Tang, Xiaozhu Ju et al.
>
> **机构**：北京人形机器人创新中心（Beijing Innovation Center of Humanoid Robotics, X-Humanoid）WFM System Group
>
> **发布时间**：2026 年 07 月（arXiv 2607.06655）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.06655) | [PDF](https://arxiv.org/pdf/2607.06655)
>
> **分类标签**：`VLA` `注意力可解释性` `瓶颈Token` `未来帧预测` `flow matching` `跨具身泛化`

---

## 一句话总结

Pelican-VLA 0.5 在 Qwen3-VL 4B 骨干上用一组 K=32 的可学习 **BotTokens** 强制感知信息必须经过一个压缩瓶颈才能到达动作解码器,发现这个纯架构约束（而非任何目标函数或标注）就能在预训练阶段自发诱导出"操作中心注意力"（zero-shot 下目标物体的注意力 IoU 从 0.054 升至 0.124,Spearman ρ=0.76,p=0.01）；RoboTwin 微调后达到 91.4%（Clean）/91.0%（Randomized）,是文中对比的 9 个开源 VLA 基线中平均成功率最高的,但作者也坦承严格 zero-shot 操作成功率仍然很低,提出"表征级泛化"和"动作级泛化"之间存在尚未弥合的 **representation-to-action gap**。

## 一、问题与动机

VLA 模型的核心目标是跨物体、场景、任务、具身的泛化,但现有模型仍严重依赖任务/环境特定的数据采集与微调。作者观察到代表性开源 VLA 模型的动作通路（action pathway）注意力图往往*弥散*地分布在机械臂本体、背景杂物和任务无关物体上,而非稳定聚焦在待操作目标和潜在接触区域——这被认为是动作解码器未被显式约束去提取任务关键视觉信息、进而依赖视觉捷径或环境相关性的一个原因。由此提出核心问题：VLA 能否在预训练阶段就自发形成操作中心的注意力表征,并以此作为跨场景/物体/具身泛化的早期基础？

## 二、核心方法

### 统一 token 序列与架构

Pelican-VLA 0.5 是构建在单一共享 Qwen3-VL 4B 骨干上的统一 Transformer,不采用双系统或 Mixture-of-Transformers（MoT）式的独立推理/执行专家设计,所有功能共享同一 token 流。输入序列由四段拼接而成：

$$\mathbf{Z} = [\text{Prefix}; \text{Middle}; \text{Bottleneck}; \text{Suffix}]$$

- **Prefix（视觉-语言）**：当前相机观测经 Qwen3-VL 视觉编码器编码后与语言指令 token 拼接,段内双向注意力,提供当前场景的语义表征。
- **Middle（Cosmos 潜变量）**：用冻结的 Cosmos-Tokenizer 将 $t{-}15$ 与 $t$ 两个时刻的多视角图像编码为连续潜特征,提供像素级、动态感知的场景视图,段内及与 Prefix 双向注意力,由未来帧预测目标监督。
- **Bottleneck（BotTokens）**：$K{=}32$ 个可学习瓶颈 token $\mathbf{S}_0 \in \mathbb{R}^{1\times K\times D}$,从 $\mathcal{N}(0, 0.02^2)$ 初始化并在 batch 间广播,训练时以概率 $p{=}0.1$ 施加 dropout。
- **Suffix（本体状态 + 噪声动作）**：本体状态投影为单个 state token,噪声动作块 $\mathbf{x}_\tau \in \mathbb{R}^{H\times d_a}$（$H{=}50$，$d_a{=}32$）与 flow time 的正弦嵌入融合后得到 $H$ 个动作 token,用于 flow-matching 去噪。

### 瓶颈机制：如何让 BotTokens 真正成为瓶颈

仅仅插入 BotTokens 并不自动构成信息瓶颈,论文用三种机制强制感知信息必须经过它：

1. **课程式瓶颈掩码**：定义上游感知 token 集合 $\mathcal{C}=\{\text{prefix}\}\cup\{\text{middle}\}$,硬约束下移除 suffix 到 $\mathcal{C}$ 的所有直接注意力边

$$A_{i\to j}=0,\quad \forall i\in \text{suffix},\ j\in\mathcal{C}$$

大白话：动作 token 不许"偷看"原始视觉/语言 token,只能通过 BotTokens 这道"闸门"获取信息。该硬约束以概率 $p_t=\min(1, t/T_{\text{warm}})$（$T_{\text{warm}}{=}10\text{k}$）随训练步数线性引入,先让模型在容易的稠密通道下学习,再逐步收紧到瓶颈通道,避免训练初期不稳定。

2. **正交正则化**：对 $\ell_2$ 归一化后的 BotTokens 输出矩阵 $\hat{\mathbf{Z}}^{\text{tokens}}_b$ 施加

$$\mathcal{L}_{\text{reg}} = \frac{1}{B}\sum_{b=1}^{B}\frac{1}{K^2}\left\|\hat{\mathbf{Z}}_b^{\text{tokens}}(\hat{\mathbf{Z}}_b^{\text{tokens}})^{\top} - \mathbf{I}_K\right\|_F^2$$

大白话：惩罚不同 BotTokens 之间的 Gram 矩阵偏离单位阵,逼着 32 个 token 学到互补而非冗余的信息,防止全部坍缩到同一个方向。

3. **BotTokens 门控生成**：对 BotTokens 输出均值池化得到 $\bar{\mathbf{z}}$,计算门控 $\mathbf{g}=\sigma(\mathbf{W}_g\bar{\mathbf{z}})$,用其调制未来帧生成特征 $\tilde{\mathbf{F}}_{\text{gen}} = \mathbf{F}_{\text{gen}} \odot \mathbf{g}$,把 BotTokens 与"预测将要发生什么"绑定,而不只是编码静态外观。

### 训练目标

总损失为四项加权和：

$$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_{\text{gen}}\mathcal{L}_{\text{gen}} + \lambda_{\text{task}}\mathcal{L}_{\text{task}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$$

（$\lambda_{\text{gen}}{=}0.01,\ \lambda_{\text{task}}{=}0.1,\ \lambda_{\text{reg}}{=}0.01$）,其中 $\mathcal{L}_{\text{action}}$ 是标准 conditional flow-matching 速度场回归损失,$\mathcal{L}_{\text{gen}}$ 是未来帧在 Cosmos 潜空间的回归损失（避免直接像素重建的开销与不稳定）,$\mathcal{L}_{\text{task}}$ 是 BotTokens 轨迹表征与语言指令表征之间的对称 InfoNCE 对比损失（同指令样本的跨对角项被 mask 掉,避免误伤真实语义匹配）。

### 缓存推理

推理时利用 BotTokens 只依赖当前视觉-语言上下文（与 flow time 无关）这一特性做三段式缓存：先算 prefix 的 KV cache,再扩展 middle 和 BotTokens 的 cache（此后只保留 K 个 BotTokens 的 KV 对作为去噪循环的视觉上下文,丢弃更大的 prefix/middle cache）,最后从高斯噪声出发用 $N{=}10$ 步 Euler 法反向积分 flow ODE 得到动作。这让重复去噪的成本只随 suffix 长度和 BotTokens 数量变化,而不随完整视觉上下文长度变化,支持高频控制。

## 三、关键结果

**预训练数据**：AgiBot World Alpha、InternData-A1、Galaxea Open-World Dataset,以及约 1000 小时自采的 Tienkung/UR 遥操作数据（格式对齐 RoboMIND）,经 LeRobot 接口统一,总量超过 6000 小时；状态/动作统一 pad 到 32 维联合位置动作空间。当前模型仅在该混合数据上训练了约 0.4 个 epoch（相当于有效看过约 2400 小时数据）。

**RoboTwin 基准（seen tasks，Table 1）**：

| 方法 | Clean | Randomized | Average |
|---|---|---|---|
| π0 | 80.0 | 79.5 | 79.8 |
| π0.5 | 86.8 | 87.0 | 86.9 |
| X-VLA | 72.9 | 72.8 | 72.9 |
| StarVLA-OFT | 88.2 | 88.3 | 88.3 |
| ABot-M0 | 86.1 | 85.1 | 85.6 |
| LingBot-VLA | 88.6 | 86.7 | 87.7 |
| Qwen-VLA | 86.1 | 87.2 | 86.7 |
| JoyAI-RA 0.1 | 90.5 | 89.3 | 89.9 |
| Hy-VLA（次优） | 90.9 | 90.1 | 90.5 |
| **Pelican-VLA 0.5** | **91.4** | **91.0** | **91.2** |

Clean 与 Randomized 之间仅差 0.4 分,作者将其解释为路由感知信息通过紧凑 BotTokens 抑制了低层视觉捷径。

**Zero-shot 泛化**：直接把预训练模型部署到完全未参与训练的 RoboTwin 2.0（新物体、新场景布局、新具身）,并用 TF-IDF 余弦相似度 + 人工复核排除与 InternData-A1 的任务泄漏。模型能做出有指向性、连贯的抓取/放置动作（如捡可乐瓶、把玩具车放上平台、开关按压）,但论文明确说明严格 zero-shot 设定下的成功率仍然很低,失败主要出现在稳定抓取、精确放置等细粒度阶段,而非目标选择/趋近阶段。

**真机评测**：在 TienKung 人形机器人上做桌面清理任务,微调自遥操作示教数据,在随机化物体类型/数量/摆放的固定测试集上达到 **80%** 成功率。

**架构归因（消融，4.1-4.2 节）**：在数据、优化器、调度、参数量完全一致的条件下，对照训练一个动作专家可稠密访问全部上游感知的 MoT 架构，两者训练动作损失相近，但 MoT 注意力仍然弥散、与目标物体弱对齐，而 Pelican-VLA 0.5 形成集中的操作中心注意力——说明效应来自架构（BotTokens 路由方式）而非数据本身。进一步的 5 级消融梯（仅动作损失 / +未来帧损失 / +语言对比损失（无瓶颈）/ +BotTokens / +BotTokens+对比损失）显示：单独加未来帧损失或对比损失（在稠密模型上）都不会改变注意力（接近随机水平）,操作中心注意力只在引入 BotTokens 瓶颈（第 4 档）时才骤然出现——把效应清晰归因到瓶颈本身,而非辅助损失。

**预训练动态定量证据（Table 2）**：机械臂本体区域的注意力 IoU 从预训练 50k 步到 500k 步基本不变（0.360→0.350，ΔIoU=-0.010，ρ=-0.43，p=0.21，不显著）；而目标物体（瓶子）的注意力 IoU 显著上升（0.054→0.124，ΔIoU=+0.070，ρ=0.76，p=0.01）。

**Zero-shot 与微调表征相似度（Table 3）**：微调前后目标注意力 IoU（0.124→0.134→0.127）与注意力相似度（1.000→0.932→0.928）均保持在高位,说明微调主要是把已经预训练好的操作中心注意力"翻译"为可执行动作,而非从零建立新表征——这正是论文强调的 representation-to-action gap 的直接证据。BotTokens 接口插入 MoT 式架构后同样能诱导集中注意力（Fig. 8）,表明该机制具有一定的跨架构可迁移性；但语言条件化的注意力偏移（指令换成"打开抽屉"后注意力随之转移，Fig. 7）主要出现在早期 checkpoint,随训练推进而减弱,作者将其归因于训练数据中语言-动作对齐存在噪声。

## 四、评价与展望

**优点**：（1）用极简的架构改动（一个可学习瓶颈 + 课程式硬掩码）把"表征是否操作中心化"这一通常靠事后可视化定性讨论的问题，转化成了可控消融、可定量追踪（IoU、Spearman ρ、表征相似度）的研究对象，方法论干净、归因链条完整（4.1 架构对照 + 4.2 五级消融相互印证）。（2）明确区分了"表征级泛化"与"动作级泛化"，并给出量化证据（Table 3 的高注意力相似度 vs. 微调后成功率大幅提升）支撑这一区分，这是一个有价值且诚实的问题框定，避免了把 zero-shot 注意力对齐直接包装成 zero-shot 操作能力。（3）RoboTwin 上以 91.2% 平均成功率超过包括 π0.5、StarVLA-OFT、Hy-VLA 在内的多个 2025-2026 开源 VLA 基线。

**局限与开放问题**：（1）核心结论建立在注意力图可视化和 IoU 指标之上，这类归因方法本身存在局限（注意力权重不完全等价于因果重要性），论文虽然做了消融但未引入更严格的因果干预分析（如遮挡/反事实测试）来验证 BotTokens 内容确实驱动了下游动作。（2）严格 zero-shot 操作成功率数值在正文中始终以"low""early glimmer"等定性描述带过，未给出具体百分比，使得"表征级泛化"到底转化了多少动作级能力难以量化评估。（3）语言条件化注意力随训练推进反而减弱，论文归因于数据中语言-动作对齐噪声，但未提供数据质量的定量分析或改进后的对照实验，这是一个尚未闭环的开放问题。（4）当前版本仅用了约 2400 小时数据（不到 1 个 epoch）、joint-position 动作表示（相比 end-effector 动作更不利于跨具身迁移），作者自己也将 representation-to-action gap 主要归因于此，并计划扩展到约 7000 小时数据——这意味着当前 91.2% 的 RoboTwin 成绩和 zero-shot 现象都应被视为一个早期检查点上的观察，其可扩展性和稳定性还有待更大规模训练验证。（5）与同类"统一未来帧预测 + 动作生成"路线（如 GR00T N1 系列、InternVLA-A1）相比，本文的差异化卖点是瓶颈架构本身而非未来帧预测（未来帧预测损失在消融中被证明并非注意力涌现的原因），这提示未来帧生成分支的必要性本身也值得进一步审视——论文中未来帧生成损失单独使用时对注意力模式没有影响，其价值可能更多体现在下游任务性能或表征质量的其他维度上，而非本文重点考察的注意力集中性上。

## 参考

1. Black et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization*, 2025.
2. Cai et al. *InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation*, 2026.
3. Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation*, 2025.
4. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025.
5. Agarwal et al. *Cosmos World Foundation Model Platform for Physical AI*, 2025.
