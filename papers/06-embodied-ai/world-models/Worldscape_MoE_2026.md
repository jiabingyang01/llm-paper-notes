# Worldscape-MoE：面向可扩展异构动作控制的统一混合专家世界模型

> **论文**：*Worldscape-MoE: A Unified Mixture-of-Experts World Model for Scalable Heterogeneous Action Control*
>
> **作者**：Jianjie Fang, Yongyan Xu, Ziyou Wang（共同一作）, Chen Gao, Haisheng Su, Yu Shang, Wei Wu, Xinlei Chen, Yong Li（通讯）et al.
>
> **机构**：清华大学（Tsinghua University）、Manifold AI
>
> **发布时间**：2026 年 07 月（arXiv 2607.03964）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.03964) | [PDF](https://arxiv.org/pdf/2607.03964)
>
> **分类标签**：`世界模型` `混合专家` `异构动作控制` `视频扩散` `具身操作`

---

## 一句话总结

Worldscape-MoE 用一个基于 Diffusion Transformer 的**共享世界骨干 + 控制感知稀疏专家（MoE）** 架构,把相机轨迹（locomotion）、双臂机器人动作（manipulation）、手关节动作图（hand-joint）三种异构动作控制统一进同一个视频世界模型;通过"共享专家学世界公共动力学、模态专家学各自动作语义"的分工,在 iWorldBench locomotion（Avg 0.7556）、WorldArena manipulation（EWM 62.84）、EgoDex hand-motion（FVD 110.94）三个赛道上均取得第一,且异构联合训练相比单独训练与 dense 混训都有增益。

## 一、问题与动机

当前可控世界模型被"控制形式"割裂成互不相通的孤岛:交互式/导航类模型（Genie、Matrix-Game 系列、HY-World)主打相机/导航控制;具身操作模型（Ctrl-World、Cosmos-Predict）主打机器人动作条件;第一人称交互模型（Hand2World、Generated Reality）引入手关节/手势控制。它们各自有独立的架构、训练目标和数据管线。

作者指出这一分裂是**结构性** 的:数据、模型容量、已学到的世界知识都沿着"控制边界"被反复切分,即便这些控制描述的底层场景与物理规律本质上是共享的。核心挑战不是缺少某一模态的可控生成器,而是**缺少一个能把异构动作监督吸收进共享世界动力学表示的可扩展学习框架**。相机轨迹、机器人关节动作向量、手关节动作图在表示形式与注入路径上差异巨大,但它们约束的是同一个隐世界:物体持续存在、交互服从物理规律、运动轨迹时序连贯、动作引发因果一致的场景变化。因此设计原则应是"分解而非隔离":把问题拆成**共享世界动力学** 与**控制专属的动作解释**。

作者用四个研究问题组织全文:RQ1 单一世界模型能否在不牺牲各控制单独性能的前提下支持多种异构控制;RQ2 MoE 学到的是有意义的专家路由与特化,还是纯粹靠加容量;RQ3 异构训练在加入新控制/数据源时能否提升可扩展性;RQ4 统一模型能否泛化到训练分布之外并支持 loco-manipulation 组合场景。

## 二、核心方法

### 2.1 异构动作控制的世界建模形式化

给定当前观测帧 $O_t$、文本世界描述 $C_t$、动作条件 $A_t$,模型预测下一段世界观测:

$$W_{t+1} = f_\theta(O_t, C_t, A_t)$$

其中动作条件可取多种异构形式:

$$A_t \in \{A_{\mathrm{traj}},\; A_{\mathrm{hand}},\; A_{\mathrm{act}},\; A_{\mathrm{other}}\}$$

- $A_{\mathrm{traj}} \in \mathbb{R}^{T_b \times d_{\mathrm{traj}}}$:时序相机运动序列,转成稠密相机控制特征;
- $A_{\mathrm{hand}} \in \mathbb{R}^{T_b \times H \times W \times C_{\mathrm{hand}}}$:稠密手关节动作图序列;
- $A_{\mathrm{act}} \in \mathbb{R}^{T_a \times d_{\mathrm{act}}}$:低维操作序列,实现中 $T_a=17$、$d_{\mathrm{act}}=14$,即每个预测窗口一个 $17\times 14$ 的双臂动作张量。

预测出的 $W_{t+1}$ 是连续帧片段,其末帧 $O_{t+1}$ 回灌作为下一步初始观测,支持长时程自回归世界生成。**用大白话说**:同一个 $f_\theta$ 既能"你给一条相机轨迹我就漫游场景",也能"你给一串机械臂动作我就演示抓取",还能"你给一张手骨架图我就生成手物交互",区别只在动作 $A_t$ 长什么样、从哪条路注进去。

三条设计要求贯穿全文:**Control faithfulness**(不同控制通过不同因果路径影响场景,须保各自接口语义)、**Shared world learning**(跨模态复用视觉动力学、物体恒常性、接触规律、时序连贯)、**Extensibility**(加新控制不必重建整个模型)。

### 2.2 非对称控制注入

三种控制按其结构性质走不同注入路径,统一写成模态相关的条件函数:

$$\hat{\epsilon} = f_\theta\big(z_v \oplus \phi_{\mathrm{vis}}(A),\; z_c,\; e_t + \phi_{\mathrm{tmp}}(A)\big)$$

其中 $z_v$ 是视频帧与参考图的 VAE latent,$z_c$ 是文本 embedding,$e_t$ 是 timestep embedding,$\phi_{\mathrm{vis}}$ 是控制相关的视觉注入函数,$\phi_{\mathrm{tmp}}$ 是控制相关的时序调制函数。具体地:

- **Locomotion**:控制信号转成稠密相机控制特征,经一个轻量可训练 Control Adapter 映射到与视频 latent 对齐的特征空间,在 patch-token 级与视频 token 融合($\phi_{\mathrm{vis}}$);
- **Hand Motion**:动作图先由冻结 VAE 编码成 latent 控制特征,作为额外视觉条件融入条件 latent 分支($\phi_{\mathrm{vis}}$);
- **Manipulation**:低维动作序列经可训练 Action MLP 编码成紧凑动作 embedding,融入 timestep embedding 及其投影调制向量,通过时序调制路径影响动力学($\phi_{\mathrm{tmp}}$)。

**用大白话说**:空间上稠密的控制(相机特征、手动作图)走"视觉 token"通道,直接和画面对齐;而紧凑的低维动作走"时间调制"通道,像调节钟表一样调制生成的时序演化。这种非对称是刻意为之的。

### 2.3 控制感知的 MoE-FFN

把每个 DiT block 里的 FFN 替换成控制感知的 MoE-FFN。对支持 $M$ 个控制模态的模型,第 $l$ 层专家集为 $\mathcal{E}^{(l)}=\{\mathcal{E}_0^{(l)}, \mathcal{E}_1^{(l)}, \dots, \mathcal{E}_M^{(l)}\}$,其中 $\mathcal{E}_0^{(l)}$ 为**所有样本共享**,$\mathcal{E}_r^{(l)}$ 特化于第 $r$ 个模态。所有专家从预训练 FFN 或当前共享专家初始化(而非随机),保住视频生成先验。

设样本 $i$ 在第 $l$ 层的隐 token 为 $H_i^{(l)}$,可用控制的二值向量为 $\mathbf{u}_i=[u_{i,1},\dots,u_{i,M}]$,构造资格掩码:

$$\mathbf{m}_i = [1, u_{i,1}, \dots, u_{i,M}] \in \{0,1\}^{M+1}, \qquad \mathcal{A}_i = \{k \in \{0,\dots,M\} \mid m_{i,k}=1\}$$

即共享专家永远合格,模态专家仅在其对应控制信号存在时才激活。路由器从均值池化的隐状态预测样本级 logits,再对合格专家做 masked softmax:

$$\mathbf{z}_i^{(l)} = \frac{1}{N}\sum_{n=1}^{N} h_{i,n}^{(l)}, \qquad \mathbf{a}_i^{(l)} = \mathbf{W}_{\mathrm{route}}^{(l)} \mathbf{z}_i^{(l)}$$

$$\alpha_{i,k}^{(l)} = \frac{m_{i,k}\,\exp(a_{i,k}^{(l)}/\tau)}{\sum_{j=0}^{M} m_{i,j}\,\exp(a_{i,j}^{(l)}/\tau)}, \qquad k \in \{0,\dots,M\}$$

MoE-FFN 输出在激活专家集 $\mathcal{A}_i$ 上聚合:

$$\mathrm{MoE}^{(l)}\big(H_i^{(l)}, \mathbf{u}_i\big) = \sum_{k \in \mathcal{A}_i} \alpha_{i,k}^{(l)}\, \mathcal{E}_k^{(l)}\big(H_i^{(l)}\big)$$

路由器实现为**零初始化、无 bias 的线性层**,训练初始时对 $\mathcal{A}_i$ 内专家给均匀权重;**不引入任何辅助负载均衡损失**,专家特化完全由"确定性的模态资格 + 扩散目标端到端优化"诱导。**用大白话说**:不像传统 MoE 让路由器自己去猜该激活谁(还得靠负载均衡损失防塌缩),这里用规则硬性规定"来的是什么控制就只让对应专家 + 共享专家上场",这样保证每个样本都更新共享专家(积累跨控制世界知识),同时不让无关模态专家被它读不懂的控制信号污染梯度。

### 2.4 Worldscape-MoE Tuning:渐进式扩展

为在保留预训练生成先验的同时快速吸收新控制,采用两条策略:

- **Grouped Learning**:新引入或强模态专属的模块用基础学习率;从预训练继承的共享组件(共享专家 + 部分 adapter)用更小学习率。防止共享专家过拟合到最近/最频繁的控制类型,同时让新专家/adapter 快速获得控制专属行为。
- **Progressive Control Expansion**:模态增量式,分两阶段。**Stage I** 从预训练视频生成模型出发,先用相机控制 + 机械臂信号训练,建立可靠生成先验、稳定的相机轨迹控制与具身任务能力;**Stage II** 逐步引入新模态及其专家,每个新专家从**当前共享专家** 初始化(而非随机),先继承已积累的世界先验再特化到自己的控制接口。作者还发现:若从预训练的 Locomotion 专家克隆去初始化其它专家,会导致 Locomotion 专家权重更新幅度显著变小、原有 locomotion 控制部分塌缩;而从 dense 模型骨干统一克隆初始化更稳定平衡。

### 2.5 数据构建

- **相机控制数据**:RealEstate10K、DL3DV-10k(真实);按 iWorld-Bench 的 81 种相机运动模式在 Unreal Engine 采集风格化仿真;Sekai、SpatialVid 用 VIPE 重标轨迹;caption 沿用 Matrix-Game 3.0 策略。
- **机器人具身数据**:RoboTwin 2.0 仿真;基于 Seedance 2.0 + Qwen3.6-Plus 的具身数据增强管线,变换材质/颜色/被操作物体但保留动作监督与物理交互;LIBERO 用于单臂扩展。
- **第一人称手部数据**:EgoDex + Ego4D;统一用 HaMeR 做单目 3D 手部重建,投影到 2D 像素坐标后光栅化成与视频帧像素对齐的手关节动作图。

## 三、实验结果

评测覆盖三种控制:locomotion(iWorldBench,500 例评测集)、manipulation(WorldArena 协议,EWMScore = 16 个归一化视频指标算术平均,六大维度)、hand-motion(EgoDex 测试集随机 100 个手物交互样本)。

**Locomotion(iWorldBench,表 1,越高越好)**:

| 模型 | Avg | Motion Smooth. | Trajectory Acc. |
| --- | --- | --- | --- |
| **Worldscape-MoE** | **0.7556** | **0.9941** | 0.6300 |
| w/o MoE | 0.6869 | 0.9930 | 0.6100 |
| videox-fun-Wan | 0.7443 | 0.9899 | **0.7645** |
| HY-World 1.5 | 0.7322 | 0.9908 | 0.6844 |
| AC3D | 0.7262 | 0.9934 | 0.6729 |
| RealCam-I2V | 0.7063 | 0.9901 | 0.7050 |

Avg 相比最强 baseline VideoX-Fun-Wan 高 0.0113,Motion Smoothness 领先。注意其 Trajectory Accuracy(0.6300)并非最高,增益主要来自生成质量维度。

**Manipulation(WorldArena,表 2,EWM Score)**:

| 模型 | EWM Score |
| --- | --- |
| **Worldscape-MoE** | **62.84** |
| w/o MoE | 61.88 |
| CtrlWorld | 59.98 |
| Wan 2.6 | 59.80 |
| CogvideoX | 58.79 |
| Cosmos-Predict 2.5 (action) | 54.29 |
| GigaWorld-0 | 50.96 |

超过 CtrlWorld +2.86、Wan 2.6 +3.04。细粒度看(表 6/7):Physics Adherence 的 Interaction Quality 达 0.8008(全场最高),Controllability 的 Instruction Following 0.9348、Semantic Alignment 0.9039 均为最高;Trajectory Accuracy 0.4610 仅次于 CtrlWorld(0.4766);3D Accuracy 的 Perspectivity 0.9686 略逊于 w/o MoE(0.9744)。

**Hand-motion(EgoDex,表 3,越低越好除 Image Quality)**:

| 模型 | FID-VID | FVD | FID | Image Quality |
| --- | --- | --- | --- | --- |
| **Worldscape-MoE** | **3.80** | **110.94** | **5.78** | **0.7325** |
| w/o MoE | 5.39 | 128.87 | 15.34 | 0.7250 |
| Cosmos-Predict 2.5 | 15.02 | 628.96 | 51.36 | 0.6158 |
| HunyuanVideo-1.5 | 23.18 | 517.42 | 56.31 | 0.6419 |
| MimicMotion | 26.74 | 589.47 | 48.92 | 0.5324 |
| LOME | 144.58 | 1794.84 | 67.82 | 0.5281 |

相比 w/o MoE 变体,FID-VID / FVD / image-level FID 分别下降 1.59 / 17.93 / 9.56。

**MoE 路由分析(RQ2)**:对 action-only 评测样本挂前向 hook,累计各专家在所有 token 与去噪步上的路由权重。结果:**共享专家占门控加权计算的 69.48%**;专属专家占比呈明显模态依赖——locomotion 20.91%、manipulation 47.95%、hand motion 35.97%。即相机类 locomotion 大部分可由共享时空先验处理(专属修正较小),而操作与手部控制因涉及物体接触与细粒度手物交互,需要更强的模态专属计算。这支持了"共享专家建模公共世界动力学、专属专家建模接口专属动作语义"的设计意图。

**其它结论**:(RQ3 可扩展性)引入新模态后 locomotion 会出现短暂退化,随训练进行逐步恢复(图 9,500→6500 步),专家权重 L2 更新分析显示共享专家最稳定、ActionMap 早期适应最强(表 8);LIBERO 单臂扩展仅 5K 微调步即能生成贴合 GT 的动作条件 rollout,且不损害其它模态能力。(RQ4)OOD 上能在大视觉偏移下生成连贯 locomotion、把仿真 RoboTwin 机械臂泛化到真实家政场景、把单数据集手姿控制迁到未见场景,并能合成 loco-manipulation 组合行为。

## 四、局限性

作者在附录明确列出:(1)训练**计算昂贵**,需大量 GPU;(2)随着模态增多训练速度下降,限制了当前框架的实际可扩展性(一个可能的缓解是预先编码 VAE 特征);(3)未针对**实时部署** 优化,如何蒸馏成更快模型、接入现有加速/蒸馏管线仍是开放问题。此外,笔者观察到几点未被充分讨论的短板:Trajectory Accuracy 与 Action Following 等"精确遵从"指标并非最优(locomotion 的轨迹精度输给 VideoX-Fun-Wan,manipulation 的 Action Following 0.0955 远低于 CtrlWorld 之外的整体);EWMScore 直接沿用 WorldArena 已发布结果,manipulation 未在完全一致的推理配置下自测,存在协议对齐的不确定性;共享专家 69.48% 占比的分析仅在 action-only 采样样本上,未给出跨随机种子的方差。

## 五、评价与展望

**优点**。(1)问题框定清晰且有价值——把"异构动作控制世界建模"上升为一个 scaling 问题,论点"缺的不是单模态可控生成器,而是能汇聚异构监督的统一框架"很有说服力。(2)方法工程上干净:非对称注入(稠密控制走视觉 token、紧凑动作走时序调制)+ 确定性资格路由 + 从共享专家初始化的渐进扩展,三者配合较自洽;尤其**用确定性模态资格替代软路由 + 负载均衡损失** 这一点,规避了标准 MoE 训练不稳/专家塌缩的老问题,也天然保证共享专家每步都被更新,是本文相对通用 MoE(Switch/GShard/DeepSeekMoE)最实质的差异化设计。(3)三赛道全面第一 + w/o MoE 消融证明增益并非纯来自加容量,再加上路由占比呈现合理的模态依赖模式,证据链相对完整。

**不足与开放问题**。(1)"共享世界先验"目前只是通过下游指标与路由统计间接论证,缺少对共享专家究竟学到了什么物理规律的机制性探针(如反事实干预、跨模态迁移的因果分析)。(2)确定性资格路由虽稳,但也意味着**模态数固定死在结构里**,每加一个模态就得加一个专家分支与控制通路,与标题"scalable"追求的"任意异构监督即插即用"之间仍有距离——真正的可扩展或许需要让新模态共享/复用已有专家而非总是新开分支。(3)与并行工作的关系:相机控制侧对标 CameraCtrl/MotionCtrl/CamI2V/RealCam-I2V/AC3D,具身侧对标 Ctrl-World/Cosmos-Predict/GigaWorld-0/IRASim,手部侧对标 Hand2World/Generated Reality/LOME,本文的独特卖点正是"把这三条线塞进一个骨干";但这也带来公平性隐忧——各 baseline 分辨率/时长/FPS 不一(附录表 5),统一评测虽已尽量对齐官方配置,跨模型的可比性仍是这类"大一统"论文的通病。(4)可改进方向:引入更强的物理一致性约束或可微物理监督以提升 Action Following/Trajectory Accuracy;探索让共享专家做世界模型、专属专家做策略/动作头的"世界-策略"联合训练;以及作者自己点到的 DiT 压缩与实时蒸馏。总体看,这是一篇定位准、执行扎实的"异构可控世界模型统一化"工作,把 MoE 从容量工具重新诠释为"共享动力学与控制解释解耦的机制",对后续统一具身世界模型有较好的参考价值。

## 参考

1. Bruce et al. *Genie: Generative Interactive Environments.* ICML 2024. — 相机/导航控制交互世界模型代表,本文 locomotion 谱系源头之一。
2. Guo et al. *Ctrl-World: A Controllable Generative World Model for Robot Manipulation.* arXiv:2510.10125, 2025. — 机器人动作条件世界模型,本文 manipulation 主要对比与超越对象。
3. Wang et al. *Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures.* arXiv:2602.09600, 2026. — 手关节/手势条件第一人称交互生成,本文 hand-motion 谱系代表。
4. Shang et al. *WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models.* arXiv:2602.08971, 2026. — 本文 manipulation 评测(EWMScore)所依赖的统一具身世界模型基准。
5. Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models.* arXiv:2401.06066, 2024. — 共享 + 专属专家分工的 MoE 特化思想,可与本文的确定性资格路由对照。
