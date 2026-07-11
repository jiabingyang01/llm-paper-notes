# MiVLA：基于人机互模仿预训练的可泛化视觉-语言-动作模型

> **论文**：*MiVLA: Towards Generalizable Vision-Language-Action Model with Human-Robot Mutual Imitation Pre-training*
>
> **作者**：Zhenhan Yin, Xuanhan Wang, Jiahao Jiang, Kaiyuan Deng, Pengqi Chen, Shuangle Li, Chong Liu, Xing Xu, Jingkuan Song, Lianli Gao, Heng Tao Shen
>
> **机构**：Tongji University；University of Electronic Science and Technology of China
>
> **发布时间**：2025 年 12 月（arXiv 2512.15411, v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.15411) | [PDF](https://arxiv.org/pdf/2512.15411)
>
> **分类标签**：`人机互模仿` `跨具身学习` `VLA预训练` `扩散策略` `RoboTwin-2.0`

---

## 一句话总结

MiVLA 用"人机互模仿"预训练目标——机器人示教既学预测自身动作也学反推对应的人手动作,人类视频既学预测人手动作也学反推对应的机器人动作,二者通过拇指指关节与末端执行器姿态之间的运动学映射双向桥接——完全不依赖真机数据即在 RoboTwin-2.0 上取得 69%/66%(easy/hard)平均成功率,在 PiPer/ARX/LocoMan 三种真实具身上平均成功率 55%,均超过 π0、π0.5、H-RDT 等基线。

## 一、问题与动机

VLA 的可扩展预训练受限于真机数据的稀缺:真实采集成本高、覆盖场景有限。已有两类替代方案各有短板——模拟机器人数据提供动作先验但存在 Sim2Real 视觉/动力学 gap；人类视频(如 Ego4D、EgoDex)提供丰富的真实场景与日常行为知识,但人手与机械臂之间存在形态差异(action gap),难以直接迁移为机器人控制信号。论文提出的核心问题是:能否在不使用任何真机数据的前提下,把模拟数据的动作先验与人类视频的行为知识统一到同一个模型中,训练出可泛化的 VLA?

为此作者设计了人机互模仿(mutual imitation)预训练范式:给定单一具身(机器人或人)的示教,模型不仅要预测该具身自身的动作轨迹,还要"想象"并生成另一未见具身在同一场景下应有的动作,从而让模型同时习得人类行为的真实保真度(behavioral fidelity)与模拟机器人数据的操作多样性(manipulative diversity)。

## 二、核心方法

**统一动作空间。** 论文定义三类动作表示以兼容不同粒度的监督:

- 人体关节 $a_h^{\text{joint}}$,48 维:双侧手腕位姿(3D 位置 + 6D 朝向,共 18 维)+ 十指指尖位置(3D $\times$ 10,共 30 维);
- 机器人关节 $a_r^{\text{joint}}$,14 维:每臂 6 关节 + 1 夹爪,双臂共 14 维;
- 末端执行器(EEF)位姿,每个具身 14 维:3D 位置 + 4D 四元数,双臂共 14 维。

**人机动作映射(kinematic bridging)。** 以人手拇指指关节位姿和机器人末端执行器位姿作为参照点,建立双向几何变换。人到机器人(式 2):

$$
a_{r,t}^{\text{l-eef}} = a_{r,0}^{\text{l-eef}} + R^h\left(a_{h,t}^{\text{l-thumb}} - a_{h,0}^{\text{l-thumb}}\right),\quad a_{r,t}^{\text{l-joint}} = f_{IK}\left(a_{r,t}^{\text{l-eef}}\right)
$$

即机器人末端相对初始位姿的位移,由人手拇指相对初始位姿的位移经旋转矩阵 $R^h$ 变换而来,再经逆运动学(PyBullet 实现)求出关节角。右臂对称处理。机器人到人(式 3):

$$
a_{h,t}^{\text{l-thumb}} = R^m\left(a_{r,t}^{\text{l-eef}}\right),\quad a_{h,t}^{\text{l-joint}} = a_{h,t}^{\text{l-thumb}} + f_d\left(a_{h,t}^{\text{l-thumb}}\right)
$$

其中 $f_d(\cdot)$ 是基于解剖学先验、由拇指位置估计手指关节相对距离的经验函数。用大白话说:两种映射本质上都是"把末端/拇指的相对位移搬到对方坐标系里,再各自用运动学规则把其余关节补全",这样任意一条示教(无论人手还是机器人)都能自动生成"对方具身"的伪标签,不需要额外采集配对数据。

**模型架构。** 双分支扩散 Transformer:人类观测分支与机器人观测分支各自使用 DINOv2 + SigLIP 做视觉编码(224×224 输入,每帧 392 个视觉 token)、T5 做语言编码、MLP 做本体感知状态投影,再送入各自的扩散 Transformer(基于 flow-matching 的迭代去噪)预测动作 chunk。

**预训练目标。** 不再是"观测到本具身动作"的单一映射,而是从单一具身观测同时预测本具身动作与对方具身动作(式 4):

$$
P_\theta\left(A_r, \hat{A}_h \mid O_r^t\right),\quad P_\theta\left(A_h, \hat{A}_r \mid O_h^t\right)
$$

对应两个 $\ell_2$ 损失,机器人示教下的机器人到人模仿损失

$$
\ell_{r2h} = \|A_r - A_r^*\|_2 + \|\hat{A}_h - \hat{A}_h^*\|_2
$$

和人类示教下的人到机器人模仿损失

$$
\ell_{h2r} = \|A_h - A_h^*\|_2 + \|\hat{A}_r - \hat{A}_r^*\|_2
$$

(星号为式 2/3 运动学映射生成的伪标签),总预训练损失为 $\mathcal{L} = \ell_{r2h} + \ell_{h2r}$。用大白话说:机器人数据教会模型"手该怎么抓",人类视频教会模型"世界长什么样、人会怎么做日常任务",互模仿损失则强迫两条分支的动作分布在同一坐标系下对齐,从而把两种数据的优势捏合进一个模型。

预训练用 4×A100(有效 batch 128,lr 1e-4,bf16),微调用 2×A100(有效 batch 32)。据文中消融讨论,MiVLA 预训练所用混合数据规模约 900 小时,远小于 π 系列所用的万小时级真机数据。

## 三、关键结果

基线为 ACT、π0、π0.5(在多样化真实家庭场景上进一步开放世界预训练)、H-RDT(EgoDex 上预训练的双臂扩散 Transformer),均在相同数据/配置下微调以保证公平。

**RoboTwin-2.0 仿真(50 任务,评测代表性 20 任务,easy/hard 两难度,50 条/任务训练示教)：**

| 方法 | Easy 平均 SR | Hard 平均 SR |
|---|---|---|
| ACT | 9% | 8% |
| π0 | 23% | 25% |
| π0.5 | 35% | 53% |
| H-RDT | 36% | 43% |
| **MiVLA** | **69%** | **66%** |

附录给出的全部 50 任务平均:ACT 5.4/5.4%,π0 19.1/21.7%,π0.5 33.6/53.8%,H-RDT 27.4/37.2%,MiVLA **62.0/63.6%**,与代表性子集结论一致。

**真实机器人(PiPer/ARX-5/LocoMan 三种异构具身,各任务 30 条示教微调,指标含成功率 SR、完成度 Completeness、耗时 T)：**

| 方法 | SR | C |
|---|---|---|
| ACT | 0% | 0% |
| π0 | 43.0% | 56% |
| π0.5 | 54% | 64% |
| H-RDT | 27% | 34% |
| **MiVLA** | **55%** | **69%** |

细分看,π0.5 在两个单臂任务(PiPer 上"移动瓶子到垫子"、ARX 上"整理伞架")上分别取得 66%/75% 的最高 SR,优于 MiVLA 的 54%/60%；但在双臂复合具身 LocoMan(四足 + 轻量双臂)的"收拢散落物体"任务上,π0/π0.5/H-RDT 均明显弱于 MiVLA(50% vs 20%/0%)。作者将其归因于两点:一是 π 系列的大规模真机预训练数据以单臂任务为主,二是 LocoMan 这类"四足 + 双臂"的复合运动学结构相对该预训练分布是未见的(OOD)新形态。摘要中给出的"仿真提升 25%、真机提升 14%"整体表述,对应 hard 模式及三具身平均下 MiVLA 相对 π0/π0.5/H-RDT 均值的领先幅度。

**消融(Table 3,验证互模仿两个损失的贡献,数值为各设置下 RobotWin2.0/Piper/ARX/LocoMan 成功率)：**

| 设置 | RobotWin2.0 | Piper | ARX | LocoMan |
|---|---|---|---|---|
| From scratch(不做预训练) | 37% | 0% | 25% | 0% |
| 仅人类数据预训练(无跨具身) | 43% | 36% | 60% | 0% |
| 仅 $\ell_{h2r}$ | 46% | 30% | 49% | 20% |
| $\ell_{h2r}+\ell_{r2h}$(完整 MiVLA) | **66%** | **54%** | 60% | **50%** |

可见互模仿对复合具身 LocoMan 的提升最显著(0%→50%),说明跨具身模仿损失是解锁模型在陌生形态上泛化能力的关键,而非单纯堆叠人类数据规模。

**少样本适配(Table 4)与三类泛化(位置/物体/场景,Table 5)** 也一致显示:完整互模仿预训练在约 20 条示教即可达到较好适配效果,且在未见位置泛化上收益最明显,未见物体/场景泛化收益相对有限(平均泛化成功率仅 54%,弱于位置泛化)。

## 四、评价与展望

**优点。** 该工作提出了一个简洁但有效的运动学桥接方案(拇指指关节 ↔ 末端执行器 + IK/解剖学先验),把"人类视频提供场景/行为多样性、模拟机器人数据提供动作精度"这一常见共识,转化为一个可端到端优化的互模仿损失,而不依赖额外的配对人-机数据采集或复杂的视觉域适配模块。在完全不使用真机数据预训练的前提下,于三种异构具身(单臂 PiPer/ARX-5、复合双臂四足 LocoMan)上取得优于 π0、π0.5、H-RDT 的平均表现,尤其在 LocoMan 这类分布外复合具身上优势明显,验证了跨具身模仿目标对泛化的贡献。

**局限与开放问题。** 论文附录坦诚指出:1)在真机单臂任务上,基于大规模真机数据 + VLM 骨干预训练的 π0.5 仍具竞争力甚至反超 MiVLA,作者将其归因于 π0.5 继承了 VLM 的语义/常识推理能力(理解"什么"和"为什么"),而 MiVLA 基于纯扩散 Transformer 更擅长视觉运动模式的直接拟合(理解"怎么做"),缺乏显式语义泛化能力；2)三类失效模式集中在分布外场景——新物体的抓取姿态生成失败、高度杂乱/异常初始位姿下轨迹搜索失败、视觉显著的干扰背景导致任务目标误判；3)人机运动学映射本身依赖简化假设(拇指-末端对应、解剖学经验函数估计手指距离),对精细操作(如多指协调抓握)的映射保真度未做单独消融或量化误差分析,这部分留作开放问题。此外,真机评测任务数量(3 个)和具身数量(3 种)相对有限,泛化结论的稳健性还需要更大规模任务集验证。

**与相关工作的关系。** MiVLA 与利用模拟数据(RoboTwin 系列)或利用人类视频(EgoVLA、EgoDex/H-RDT 等)分别提升 VLA 的路线不同,核心差异在于把两者显式统一进同一个互模仿训练目标,而非简单联合训练或分阶段预训练+微调。这与近期"整合 VLM 语义推理能力和扩散策略生成能力"的融合思路(作者在结论中明确指出的未来方向)形成呼应,提示后续工作可能会在 MiVLA 式运动学桥接的动作监督之上,叠加 VLM 级别的语义 grounding 与规划模块,以同时获得视觉运动精度与开放世界语义泛化能力。

## 参考

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- Black et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization*. CoRL 2025.
- Bi et al. *H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation*. arXiv:2507.23523, 2025.
- Mu et al. *RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins*. CVPR 2025.
- Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Manipulation*. arXiv:2506.18088, 2025.
