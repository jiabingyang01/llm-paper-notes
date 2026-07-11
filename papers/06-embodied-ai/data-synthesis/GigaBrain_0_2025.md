# GigaBrain-0：世界模型驱动的视觉-语言-动作模型

> **论文**：*GigaBrain-0: A World Model-Powered Vision-Language-Action Model*
>
> **作者**：GigaBrain Team（按字母序）Angen Ye, Boyuan Wang, Chaojun Ni, Guosheng Zhao, Guan Huang, Xiaofeng Wang, Zheng Zhu et al.
>
> **机构**：GigaAI
>
> **发布时间**：2025 年 10 月（arXiv 2510.19430）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.19430) | [PDF](https://arxiv.org/pdf/2510.19430)
>
> **分类标签**：`世界模型` `VLA` `数据合成` `Real2Real` `具身CoT` `RGB-D`

---

## 一句话总结

GigaBrain-0 用一套世界模型数据引擎（GigaWorld）批量合成 Real2Real / View / Sim2Real / 人类视频迁移 / 视频生成五类多样化机器人数据,再叠加 RGB-D 输入与具身 Chain-of-Thought 监督训练 VLA,在灵巧、长程、移动操作六个真实任务上普遍超过 π0(如台面清理 65%→90%、纸巾准备 20%→50%),并靠合成数据把外观/摆放/视角泛化成功率从 20-30% 拉到 80-90%。

## 一、问题与动机

VLA 训练高度依赖大规模真机数据,而真机采集昂贵、耗时,且多样性受限——绝大多数部署反复采样同一批狭窄场景,导致外观、物体摆放、相机视角一旦变化策略就崩。作者的核心主张是:**用世界模型充当可扩展的"数据引擎"**,以合成但物理可信的轨迹去覆盖材质、颜色、光照、视角等真机难以穷举的变化,从而降低对真机数据的依赖并提升跨任务、跨条件的泛化。

在数据引擎之外,论文还提出两个框架层面的改进以增强策略鲁棒性:(1) RGB-D 输入建模,让模型获得更丰富的 3D 几何与空间布局理解;(2) 具身 CoT 监督,让模型显式生成操作轨迹、子目标语言与离散动作等中间推理,支撑长程任务与精细动作的顺序决策。

## 二、核心方法

### 2.1 模型骨架:Mixture-of-Transformers + 流匹配动作专家

GigaBrain-0 是端到端 VLA $g_\theta$,采用 mixture-of-transformers 架构:以预训练 VLM PaliGemma2 编码多模态输入,配一个基于 Diffusion Transformer(DiT)的动作专家 $f_\theta$,用流匹配(flow matching)预测动作块(action chunk)。训练时用 **Knowledge Insulation**(知识隔离)缓解连续动作空间学习与 VLM 语义推理之间的相互干扰;同时给 VLM head 增加**离散动作 token 预测** 以加速预训练收敛。

- 用大白话说:一个"大脑"负责看懂图像和语言,一个"小脑"专门吐连续动作;两者共享注意力但用"隔离"机制防止小脑训练把大脑的语言能力带偏——所以论文不用手工给语言/动作损失调权重。

### 2.2 RGB-D 输入建模

输入张量形状为 $B \times H \times W \times 4$(RGB + depth),先归一化再用 SigLIP 提特征。为适配深度通道,把 SigLIP 首层卷积**用零初始化的核扩展一个深度通道**,从而保留原 RGB 特征提取能力的同时引入深度感知。SigLIP 全程可训练;训练中**随机丢弃深度通道(补零)**,以保证对纯 RGB 输入也兼容。若采集帧缺深度,用 MoGe 生成度量尺度深度图。

- 用大白话说:给现成的图像编码器"加一根深度输入线",初始时这根线权重为 0 不破坏原能力,再慢慢学会用深度;偶尔把深度拔掉训练,保证没深度时也能跑。

### 2.3 具身 Chain-of-Thought(Embodied CoT)

模型显式生成三类中间推理 token:

1. **操作轨迹**:末端执行器路径在图像平面上的 2D 投影,取 10 个均匀采样关键点;
2. **子目标语言**:对中间目标的自然语言描述;
3. **离散动作 token**:加速后续 DiT 连续动作块预测收敛的离散表示。

其中轨迹预测**不用自回归解码**,而是引入 10 个可学习 trajectory token 作为 VLM 辅助输入,以**双向(非因果)注意力** 与全视觉上下文交互做整体空间推理,再经一个轻量 GRU 解码器回归 2D 像素坐标。子目标语言与离散动作 token 则以标准 next-token 预测自回归生成。

### 2.4 统一训练目标

轨迹回归、子目标语言、离散动作 token、DiT 连续动作块在统一目标下联合优化:

$$
\mathcal{L} = \mathbb{E}_{\mathcal{D},\tau,\epsilon}\left[ -\sum_{j=1}^{n-1} M_{\text{CoT},j}\,\log p_\theta\!\left(x_{j+1}\mid x_{1:j}\right) + \left\lVert \epsilon - a_{\text{chunk}} - f_\theta\!\left(a^{\tau,\epsilon}_{\text{chunk}}\right) \right\rVert^2 + \lambda\left\lVert \text{GRU}(\hat{t}_{1:10}) - t_{1:10} \right\rVert^2 \right]
$$

其中 $\tau \in [0,1]$ 是流匹配时间步,$\epsilon \sim \mathcal{N}(0,I)$ 为高斯噪声,$a^{\tau,\epsilon}_{\text{chunk}} = \tau\cdot a_{\text{chunk}} + (1-\tau)\cdot\epsilon$ 是加噪动作块;$M_{\text{CoT},j}\in\{0,1\}$ 是逐 token 掩码,标记位置 $j$ 是否属于 CoT 推理流(子目标语言或离散动作);$\hat{t}_{1:10},\,t_{1:10}$ 为预测/真值 2D 轨迹关键点,$\lambda=1$。

- 用大白话说:三项加起来——第一项让模型把子目标和离散动作"说对"(自回归交叉熵),第二项是流匹配让动作专家把噪声还原成正确动作,第三项让轨迹关键点贴近真值。语言项和动作项因为有知识隔离,不必人工调损失权重,各自独立学习。

### 2.5 数据引擎:真机数据 + GigaWorld 世界模型合成

**真机数据**:公开数据集 AgiBotWorld、RoboMind、Open X-Embodiment;自采数据用 Agilex Cobot Magic 平台(199 小时)与 AgiBot G1 平台(983 小时),合计 **1182 小时**,覆盖 3100 m²、工业/商业/办公/居住/实验室五大类环境、14 个细分场景。标注上用夹爪开合状态转换自动切分子任务,再用 Qwen-VL-2.5 在结构化模板+预定义动作词表约束下生成子目标语言(防幻觉);2D 轨迹由 3D 末端坐标投影到头戴相机像面得到。采用全标注/部分标注/无标注混合,并对每个任务去重、至多保留 50 条多样轨迹。

**GigaWorld 合成数据**——多条互补管线:

- **Real2Real Transfer**:基于扩散的视频生成,以 VideoDepthAnything 深度 + Canny 边缘作为 ControlNet 空间条件,保持动作语义与布局一致,对每段真机视频文本提示生成约 10 个材质/纹理/光照/配色不同的变体。
- **View Transfer**:用深度把 RGB 重投影到新视角,DiT 视频补全模型 inpaint 遮挡区,IK 求关节角、URDF 在物理仿真渲染,可选可微物理引擎微调运动可信度,得到多视角一致的重渲染。
- **Sim2Real Transfer**:Isaac Sim 中用 EmbodiedGen 程序化资产或 ArtVIP 物体搭场景,IK 算末端轨迹,再用扩散视频生成器以仿真深度为条件贴真实外观;可全控物体位置、视角、背景、物理属性(摩擦、质量)。
- **Human Video Transfer**:把 EgoDex 第一人称人手视频转成机器人可执行序列——SAM2 抠除人手,以 3D 手腕位姿为目标末端位姿,IK+URDF 渲染成机械臂。
- **Video Generation + IDM**:单张图按文本提示生成未来操作视频,再用逆动力学模型(IDM)反推动作序列作为合成训练数据;并支持多视角一致视频生成(拼接多视角噪声图)。

效率上用 NATTEN + 步蒸馏(单步生成) + FP8 推理,相比基线扩散模型达 **50× 生成加速**;生成数据再经几何一致性、多视角一致性、文本对齐、物理可信度四维打分,决定用于预训练/微调/丢弃。

## 三、实验结果

评测在双臂 PiPER 与 AgiBot G1 两平台上进行,基线为 **π0**(官方开源代码、同配置微调)。

### 主任务成功率(读自 Fig. 10 柱状图,近似值)

| 任务 | 类型 | 平台 | π0 | GigaBrain-0 |
|---|---|---|---|---|
| Laundry Folding | 灵巧 | G1 | 50% | 60% |
| Paper Towel Preparation | 灵巧 | PiPER | 20% | 50% |
| Juice Preparation | 长程 | G1 | 90% | 90% |
| Table Bussing | 长程 | PiPER | 65% | 90% |
| Boxes Moving | 移动 | G1 | 80% | 90% |
| Laundry Baskets Moving | 移动 | PiPER | 20% | 30% |

论文文字明确:灵巧任务较 π0 分别提升约 30% 和 10%;移动操作两任务各提升约 10%;长程两任务均取得最高成功率。

### 合成数据配比 α 的泛化增益(读自 Fig. 17,近似值)

α 为训练时采样世界模型合成数据的概率。三组均以 50 条真机轨迹后训练,分别叠加 Real2Real / Sim2Real / View Transfer 合成数据:

| 采样概率 α | Laundry Folding 外观泛化 | Table Bussing 摆放泛化 | Table Bussing 视角泛化 |
|---|---|---|---|
| 0% | ~24% | ~25% | ~33% |
| 25% | ~43% | ~44% | ~57% |
| 50% | ~67% | ~71% | ~75% |
| 75% | ~75% | ~91% | ~81% |
| 90% | ~83% | ~89% | ~87% |

结论:纯真机(α=0)在新外观/新摆放/新视角下大幅退化;混入合成数据后,外观泛化 α 到 50% 已近 70%、到 75-90% 超 80%;摆放泛化 α 到 75% 超 90%;视角泛化 α 到 90% 超 80%。评测协议:外观泛化测 10 件衣物(1 白 + 9 异色异纹)各折 5 次;摆放泛化测 10 种布局各 5 次;视角泛化测 9 个相机视角各 5 次。

### 端侧变体 GigaBrain-0-Small(Table 2,NVIDIA Jetson AGX Orin,table bussing 1K 集微调)

GigaBrain-0-Small 换用轻量 VLM SmolVLM2,动作专家约 100M 参数,并做消除 CPU-GPU 拷贝、torch.autocast 混合精度、RoPE 正余弦查表缓存、torch.compile 静态图等系统级优化:

| 模型 | FLOPs (GFLOPs) | 参数量 | 推理显存 (GB) | 推理延迟 (s) | 成功率 |
|---|---|---|---|---|---|
| π0 | 4400 | 3.2B | 17.5 | 1.28 | 80% |
| GigaBrain-0-Small | 840 | 402M | 1.9 | 0.13 | 80% |

仅用约 12.5% 参数、约 1/10 延迟、约 1/9 显存,端侧成功率与 π0 持平(均 80%)。

## 四、局限性

- **世界模型细节缺席**:论文反复说"所有模型与训练细节将在即将发布的 GigaWorld 技术报告中详述",本文对 GigaWorld 各生成器的架构、训练数据、保真度量化几乎没有实验支撑,合成数据质量只有定性图示与一个未展开的四维打分。
- **基线单一**:主实验只与 π0 对比,未与同类"世界模型造数据"工作(如 DreamGen、GR00T N1.5、RoboTransfer 等)做端到端策略对照,难以判断增益来自数据引擎本身还是 RGB-D/CoT 等正交改进。
- **消融不足**:RGB-D、具身 CoT、知识隔离、离散动作 token 各自贡献多少缺乏拆解;只给了合成数据配比 α 的 sweep,未隔离五类合成管线各自的作用。
- **样本规模小、评测次数少**:泛化实验每条件仅 5 次试验,单任务微调 demo 数在 50-489 之间,成功率读自柱状图无精确表格,统计显著性存疑。
- **合成数据的偏差与幻觉**:视频生成不可避免含幻觉/伪影,虽有质检打分,但打分阈值、被丢弃比例、以及合成数据是否引入系统性动作偏差均未评估。

## 五、评价与展望

**优点**:这是一篇工程完成度很高的"世界模型即数据引擎"系统论文,把 Real2Real / View / Sim2Real / 人类视频迁移 / 视频生成五条合成管线统一进一个 VLA 训练配方,并给出了合成数据配比与泛化成功率的清晰单调关系(α 越大、外观/摆放/视角泛化越好),这条曲线本身是对"合成数据能替真机数据买到泛化"这一命题相当有说服力的经验证据。RGB-D 零初始化扩展、随机丢深度以兼容 RGB、具身 CoT 中轨迹用双向注意力+GRU 回归(避免自回归)都是务实且可复用的小设计。端侧 GigaBrain-0-Small 用 12.5% 参数达到同等成功率,对落地有实际意义。

**与其他公开工作的关系**:数据引擎思路与 DreamGen(视频世界模型解锁泛化)、GR00T N1.5、RoboTransfer(几何一致视频扩散做视觉策略迁移)、EMMA/EmbodiedDreamer(real2sim2real 生成迁移)一脉相承,可视为把这些散点整合成一个覆盖面更广的合成数据矩阵;架构上继承 π0 的流匹配动作专家 + PaliGemma2,叠加 Knowledge Insulation(Driess et al.)与 FAST 式离散动作 token 加速。相对贡献更多在"数据侧广度 + 系统集成",而非单点新方法。

**开放问题与可能改进方向**:(1) 合成数据的分布偏差与"以假乱真的错误动作"如何量化并过滤,是这类方法的根本风险,值得一个专门的诊断实验;(2) 应补充逐管线、逐组件的消融,把泛化增益归因清楚;(3) 与其他合成数据方法在同一策略骨架下的对照缺失,削弱了"世界模型优于传统采集"的强结论;(4) 论文在 future work 中提出把世界模型从被动数据引擎升级为可交互的"策略环境"(在世界模型内 rollout+奖励做 RL)乃至"策略生成器",并闭环自改进——这是自然但更难的下一步,涉及世界模型物理保真度是否足以承载 RL 信用分配的老问题。

## 参考

1. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.(主基线与动作专家范式来源)
2. Jang et al. *DreamGen: Unlocking Generalization in Robot Learning through Video World Models.* arXiv:2505.12705, 2025.(世界模型造数据的直接对标)
3. Driess et al. *Knowledge Insulating Vision-Language-Action Models.* arXiv:2505.23705, 2025.(知识隔离机制)
4. Liu et al. *RoboTransfer: Geometry-Consistent Video Diffusion for Robotic Visual Policy Transfer.* arXiv:2505.23171, 2025.(几何一致视频迁移)
5. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734, 2025.(数据多样性 VLA 对照)
