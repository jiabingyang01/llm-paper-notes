# Qwen-RobotWorld：通过语言条件视频生成统一具身世界建模

> **论文**：*Qwen-RobotWorld Technical Report: Unifying Embodied World Modeling through Language-Conditioned Video Generation*
>
> **作者**：Jie Zhang、Xiaoyue Chen（共同一作）等，Qwen Team；通讯作者 Chenxu Lv、Xiong-Hui Chen、Chenfei Wu
>
> **机构**：Qwen Team（阿里巴巴 Qwen 系列）
>
> **发布时间**：2026 年 06 月（arXiv:2606.17030，v3 于 2026-06-17）
>
> **发表状态**：未录用（预印本 / 技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.17030) | [PDF](https://arxiv.org/pdf/2606.17030)
>
> **分类标签**：`语言条件世界模型` `视频生成` `MMDiT` `具身操作预训练` `跨本体`

---

## 一句话总结

用**自然语言作为统一动作接口**，把操作、驾驶、导航、human-to-robot 迁移四类具身域塞进同一个语言条件视频世界模型：冻结 Qwen2.5-VL 做动作编码 + 20B 双流 MMDiT 做状态转移，训练在 8.6M 条跨 20+ 本体、500+ 动作类别的 EWK 语料上；结果在 EWMBench（4.60）与 DreamGen（4.952）总分双双第 1，WorldModelBench（8.99）开源第 1 且四项物理一致性满分 1.00。

## 一、问题与动机

具身智能需要智能体在物理世界里感知、推理并行动，而直接在真机上训练成本高、效率低、有安全风险。世界模型提供了可扩展的替代方案：学习环境动力学后，作为可交互的训练/评估平台。作者把世界模型形式化为**状态转移函数**：给定当前状态 $s_t$ 与动作 $a_t$，预测下一状态

$$s_{t+1} = f(s_t, a_t)$$

用大白话说：世界模型就是"给它当前画面 + 一条指令，让它脑补出接下来会发生什么"。这里状态 $s_t$ 是视觉观测（视频帧或其 latent），动作 $a_t$ 作者主张用**自然语言**表示——一句 "pick up the red cup and place it on the shelf" 已隐式编码了完整动作序列、目标状态与物理约束，且不依赖任何机器人专属接口。

当前世界模型有一个根本张力：

- **通用视频生成模型**（Sora、Veo3、Wan 等）从互联网规模数据学到丰富视觉先验，但**无法准确建模具身物理**——接触动力学、刚体结构约束、动作-后果关系；
- **领域专用具身模型**（Cosmos、各类 VLA world model）绑定单一场景（桌面操作或驾驶），依赖关节角/waypoint 等**机器人专属结构化动作表示**，难以跨本体、跨任务泛化。

更深一层的障碍是**表征异构**：操作用关节角/末端 waypoint、驾驶用转向指令与速度曲线、导航用朝向向量，各自需要独立模型或接口。作者的破题思路是把所有动作**投影到共享的自然语言空间**，从而 Franka 抓取、自动驾驶、室内导航都变成同一个"语言条件视频生成"任务，可在单模型下联合训练，而各域的物理知识彼此增强而非冲突。

## 二、核心方法

系统由三部分构成：数据（EWK 语料 + action-language mapping）、模型（双流 MMDiT + 3D RoPE + Scene2Robot）、训练（general+expert 渐进课程）。

### 2.1 Action-Language Mapping 与五层标注

把 20+ 机器人本体、500+ 动作类别统一到自然语言接口，使 $s_{t+1}=f(s_t,a_t)$ 与底层物理域无关。为产出"动作充分"的字幕，设计**五层渐进标注**框架（前三层构成结构化 chain-of-thought）：

1. **Task Goal Layer**：推断该转移的高层意图（$s_t \to s_{t+1}$ 应改变什么）；
2. **Action Detail Layer**：把 $a_t$ 分解为时空轨迹、微动作、速度、力度，并**强制显式声明视角信息**（第一人称主视/腕部视/外部视/多视拼接）；
3. **Physical Feedback Layer**：描述动作对环境的可观测后果（位移、形变、接触状态变化），把每次转移锚定到可验证的物理结果。

在此之上生成两种粒度：**Comprehensive Description**（50–100 词，完整 viewpoint-agent-action-feedback 四元组）与 **Concise Description**（15–30 词，仅保留关键要素，用于推理时简短高层指令）。训练时两种粒度各 50% 概率采样。

用大白话说：与其教模型 "关节 3 转 15 度"，不如逼标注器写清 "从主视角看，右臂抓起粉色瓶子并向左倾倒把水浇到花上，水面出现晃动"——这样一句话既是动作也是物理结果，模型才能只凭语言就预测下一帧。

### 2.2 双流 MMDiT + MLLM 动作编码

架构含三件套：MLLM 动作编码器、VAE 状态编解码器、MMDiT 转移函数。

- **MLLM 动作编码器**：冻结的 Qwen2.5-VL，对输入文本 $S$ 取末层隐状态作为动作条件

$$\mathbf{h} = \phi(S)$$

- **VAE 状态编解码器**：Wan-VAE，$\mathbf{z} = \mathcal{E}(\mathbf{x})$，把视频帧编码为 latent 并解码回像素；
- **MMDiT 转移函数**：**双流** 设计——understanding stream 接收经可训练 connector 投影的 MLLM 编码 $\mathbf{h}$，generation stream 接收来自 VAE 的带噪状态 latent，两流在**每一层通过 joint attention 双向融合**。

用大白话说：一条"读题流"专门理解语言指令，一条"画图流"专门去噪生成视频，两条流每层握手一次，让每个待生成的 token 同时盯着"画面锚点"和"语义动作说明"。用 MLLM（而非 T5/CLIP 这类轻量编码器）当动作编码器有两个红利：(1) 深层语言理解能把复杂组合指令解析成细粒度条件；(2) 其内化的世界知识隐式约束"物理上可能的转移空间"（如知道机械臂是定长刚体），配合 T2I 联合训练可避免跨帧物体变形。

骨干规格：60 个双流 block，24 个注意力头（head dim 128），hidden size 3072，patch size 2×2；参数量 MLLM 7B + VAE 127M（encoder 54M + decoder 73M）+ MMDiT 20B；上下文最长支持 48,360 个 video token。

### 2.3 非对称 3D RoPE

用 3D RoPE 独立编码时间、空间高、空间宽三轴。不均分维度，而是**非对称切分**：时间轴 16 维、高/宽各 56 维，共 128 维（`pe_axes_dim = [16, 56, 56]`）。直觉：相邻帧强相关，时间轴给少维即可；空间轴要覆盖更大的物体位置与布局多样性，故给多维。再叠加 Scalable RoPE，支持推理时泛化到不同分辨率与时长。

### 2.4 Scene2Robot：human-to-robot 的三段式条件

- **First-Frame Conditioning（TI2V 基线）**：首帧作为固定视觉条件，其 VAE latent 在 generation stream 中被赋 timestep $t=0$ 且排除出去噪 loss。
- **三段式扩展（用于 human-to-robot 迁移）**：human-to-robot 本质是视频编辑问题——既要参考场景（背景/物体布局/光照），又要参考仿真 demo 给出的目标机器人运动轨迹。作者把输入组织成三个连续段（都走同一 VAE-MMDiT 管线，无需改架构）：
  1. **Scene condition**（F 帧）：抠掉人手的原始人类演示视频，VAE 编码，提供外观/布局/物体状态；
  2. **Robot reference**（F 帧）：MuJoCo 渲染的仿真机器人执行，VAE 编码，提供目标本体的运动学轨迹与形态；
  3. **Generation**（F 帧）：待去噪 latent，生成最终逼真机器人执行视频。

段 (1)(2) 同样赋 $t=0$ 并排除出 loss，仅段 (3) 回传梯度；3D RoPE 给每段分配各自的时间索引区间，joint attention 让生成 token 同时吸收场景外观、机器人运动与语言语义。

### 2.5 训练：general + expert 渐进课程

优化目标为 flow matching（视频经 VAE 编码到 latent，噪声采自标准正态；timestep 采自 log-normal 并按序列长度自适应 shifting；TI2V 任务首帧 timestep 固定为 0）。两阶段课程：

- **预训练**：14 个高质量视频平台 200M+ 真实观测样本建立通用世界先验；引入 Ego4D、EPIC-Kitchen 等大规模第一人称手部操作数据作为"通用→具身"的桥梁；T2I / T2V / TI2V 三任务在共享骨干上联合训练，任务比例从纯 T2I 逐渐过渡到三任务联合。T2I 充当"视觉质量锚"，把物体形态知识经共享骨干迁移到视频生成，抑制形变与身份漂移。
- **SFT（具身特化）**：四阶段数据混合——单视图操作 → 多视图扩展 → 多视图拼接生成 → 复杂任务与跨域数据；具身部分里操作占 ~90% 采样权重（保证物理落地深度），多视图拼接与导航/驾驶各 ~5%（保证广度）。全程通用世界数据每个 batch 都参与（具身 70% / 通用 30%），使具身特化与通用世界建模一起进步而非此消彼长。

基础设施用 Megatron-LM，混合并行 + 对部分双流 block 做 selective activation recomputation。

### 数据规模速览

| 项目 | 规模/构成 |
|---|---|
| EWK 总量 | ~8.6M 视频-文本对，>200M 观测帧；具身 70% / 通用 30% |
| 操作 | ~5.9M 样本，20+ 机器人形态（EgoHOD/Bridge V2/RH20T/DROID/Robomind/RoboCoin/AgiBot-World/InternData-A1 等） |
| 自动驾驶 | ~200K（Waymo E2E / NVIDIA PhysicalAI-AD / Bench2Drive / Sekai），共 1,744,405 clips / 2,405 h |
| 室内导航 | 6,064 episodes（VLNVerse，Isaac Sim，134 室内场景，256×256@10FPS） |
| human-to-robot | 自动 MANO-to-robot 管线，覆盖 14 种机械臂；另 ~80K episodes 仿真-真实配对 |
| 具身内部结构 | 单视图操作 ~4.3M，多视图拼接（2–4 视）~1.6M，导航+驾驶 ~200K |

## 三、实验结果

四个基准全面评测；表内粗体为该列最优。

### EWMBench（Table 2，具身运动保真，21 样本/7 任务）

| 类型 | 模型 | SceneC | HSD | nDTW | BLEU | CLIP | Logics | Overall |
|---|---|---|---|---|---|---|---|---|
| 通用 | Sora2 | 0.8526 | 0.2807 | 0.2754 | 0.2466 | 0.9100 | 0.9474 | 3.89 |
| 具身 | LVP（次优） | 0.8795 | 0.4248 | 0.6226 | 0.2179 | 0.8995 | 0.9524 | 4.05 |
| 具身 | **Ours** | **0.9142** | **0.5660** | 0.6708 | 0.2079 | 0.8834 | **1.0000** | **4.60** |

总分 4.60 排**第 1**，超次优 LVP（4.05）达 +0.55；运动保真 HSD 0.566 领先 LVP 0.425 约 **33%**；场景一致性 SceneC 0.914 与逻辑约束满足 Logics 1.00 均为最优。

### DreamGen Bench（Table 3，GR1 机器人三子集，IF=指令遵循，PA=物理对齐）

| 模型 | GR1-Env PA/IF | GR1-Object PA/IF | GR1-Behavior PA/IF | Total |
|---|---|---|---|---|
| LVP | 0.810 / 0.772 | 0.745 / 0.829 | 0.713 / **0.889** | 4.758 |
| GigaWorld | 0.621 / 0.933 | 0.500 / 0.852 | 0.426 / 0.884 | 4.216 |
| **Ours** | **0.828** / 0.793 | **0.840** / **0.878** | **0.781** / 0.832 | **4.952** |

总分 4.952 排**第 1**；GR1-Object IF 0.878 第 1，展现强的物体级组合泛化；三子集 PA 一致领先。唯 GR1-Behavior IF 0.832 略逊 LVP（0.889）与 GigaWorld（0.884）——长程行为泛化是明确的改进方向。

### PBench（Table 4，Domain=QA 式物理理解 6 域，Quality=VBench 8 指标）

| 模型 | Mot（运动平滑） | Domain | Overall |
|---|---|---|---|
| Cosmos | 0.931 | 0.840 | 0.802 |
| LVP | 0.962 | 0.772 | 0.792 |
| **Ours** | 0.990 | **0.857** | **0.804** |

**开源模型中总分最高**（0.804）；Domain 0.857 全体第 3（领域理解为最强维度，超多数闭源）；运动平滑 0.990 为开源第 2。美学（0.455）、成像（0.649）偏低，作者归因于面向具身任务、输出分辨率更低——但对下游机器人控制已足够。

### WorldModelBench（Table 5，350 实例/7 域，物理违规 5 类）

| 类型 | 模型 | Instr(0–3) | CS Overall | Phys Overall | Total |
|---|---|---|---|---|---|
| 闭源 | Wan2.6 | 2.50 | 1.94 | 4.83 | 9.27 |
| 闭源 | Veo3 | **2.52** | 1.93 | 4.80 | 9.25 |
| 具身 | Cosmos | 2.14 | 1.94 | 4.86 | 8.94 |
| 具身 | **Ours** | 2.33 | 1.72 | **4.94** | 8.99 |

总分 8.99，**超所有开源模型**（全体第 3，仅次于闭源 Wan2.6 与 Veo3）；物理一致性在 Newton's law / 质量守恒 / 流体 / 重力四项均**满分 1.00**（penetration 0.94），指令遵循 2.33/3.0；common-sense 略低同样归因于低输出分辨率。

### 定性与泛化

- **细粒度语言 grounding**：对比对（同初始帧仅改一个关键词，目标物体/放置目的地/动作类型均正确判别）+ 复杂长程组合指令；
- **跨本体/跨任务/跨视角**：一条指令驱动 single-arm/dual-arm/humanoid/dexterous hand 四种形态；multi-view 三路同步相机几何一致；
- **RoboTwin-IF 零样本**：作者在 RoboTwin 模拟器上新构造的 Instruction Following 基准，4 个 Unitree G1 任务对比 LVP 与 Cosmos2.5-14B，尽管只混入少量开源 RoboTwin 数据，仍展现强零样本表现与稳定多视一致性；
- **human-to-robot 迁移**：跨 8 种目标本体（ARX-L5、Kinova Gen3、PIPER、xArm7、Franka Panda、Sawyer、KUKA Iiwa、Kinova Jaco）保持任务意图并适配各自运动学约束；
- **移动性生成**：自动驾驶（Bench2Drive/PhysicalAI-AD/Sekai/Waymo）与第一人称室内导航（VLNVerse）。

## 四、局限性

1. **下游价值只停在"应用方向"层面**：摘要主打三大应用（合成数据引擎、策略评估环境、动作规划器），但正文**没有任何 VLA 策略实训/闭环评估的量化结果**——世界模型质量高 ≠ 生成数据真能提升策略成功率，这一关键闭环缺席。
2. **长程行为泛化偏弱**：DreamGen GR1-Behavior IF（0.832）明显落后 LVP/GigaWorld，说明多步、抽象目标的指令遵循仍是短板。
3. **分辨率-质量权衡**：为服务具身任务采用较低输出分辨率，导致 PBench 美学/成像与 WorldModelBench common-sense 偏低；对需要细节纹理判别的下游任务可能有影响。
4. **human-to-robot 依赖仿真参考**：Scene2Robot 需要 MuJoCo 渲染的机器人参考段作为运动学来源，本质仍依赖仿真轨迹，并未真正"从零脑补"新本体的可行运动。
5. **可复现性**：技术报告未公开权重/代码/完整超参与数据配比细节，20B MMDiT + 8.6M 私有语料的规模也难被外部复现；RoboTwin-IF 为自建基准，缺少第三方交叉验证。
6. **评测口径**：EWMBench 仅 21 样本，统计代表性有限；多处"第 1/第 3"的对比对象未含最新闭源全序列，横向可比性存在解读空间。

## 五、评价与展望

**优点**。核心贡献不在架构新奇，而在"**语言作为统一动作接口**"这一表征选择被系统性地贯彻到数据（500+ 动作类别的 action-language mapping + 五层标注）、模型（MLLM 动作编码进双流 MMDiT）、训练（general+expert 联合课程）三层，并用四个公开基准的第一/开源第一给出了较有说服力的经验支撑。用冻结 MLLM 的内化世界知识隐式约束"物理可行转移空间"、配合 T2I 联合训练抑制形变，是相对优雅、可解释的设计。Scene2Robot 把 human-to-robot 迁移重构成"抠人手场景段 + 仿真机器人参考段 + 生成段"的多段条件，无需改架构即复用 TI2V 管线，工程上很讨巧。

**与其他公开工作的关系**。相对 Cosmos（NVIDIA）、GigaWorld、LVP、Vidar、WoW 等具身世界模型，本文的差异化主要是"跨域统一 + 语言接口"而非某一域做深；相对通用视频模型（Sora/Veo/Wan），它以物理一致性（WorldModelBench 四项满分）换取了部分像素级美学。与 DreamGen（用神经轨迹解锁机器人学习泛化）、TesserAct（4D 具身世界模型）等相比，本文更强调"一个骨干覆盖操作/驾驶/导航/迁移"的广度，但也因此在单一域的深度评测（尤其闭环策略）上留白。物理一致性拿满分与真实动力学正确之间仍有 gap——基准用 QA/规则判违规，未必等价于接触力学的定量正确。

**开放问题与可能改进**。(1) 最该补的是**世界模型→策略** 的端到端闭环：用其合成数据训 VLA 并报告真机/仿真成功率增益，才能兑现"合成数据引擎/策略评估环境"的承诺。(2) 长程组合指令的行为泛化（GR1-Behavior）可考虑引入显式子任务分解或规划头，而非纯靠语言 chain-of-thought。(3) human-to-robot 段目前依赖仿真参考，若能减少对 MuJoCo 轨迹的依赖、直接从语言+人手视频推断新本体运动，泛化面会更大。(4) 低分辨率是刻意取舍，可探索级联超分或"低分做物理、高分补细节"的两段式，以在下游细粒度操作中兼顾。(5) 作为技术报告，公开权重与 EWK 的数据卡将极大提升其作为社区基座的价值。

## 参考

1. Li et al. *WorldModelBench: Judging Video Generation Models as World Models*, 2025（arXiv:2502.20694）——本文物理推理主评测基准。
2. Yue et al. *EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models*, 2025（arXiv:2505.09694）——具身运动保真评测。
3. Zhou et al. *DreamGen: Unlocking Generalization in Robot Learning through Neural Trajectories*, 2025（arXiv:2505.12705）——最相关的"世界模型生成轨迹促进策略泛化"工作。
4. Agarwal et al. *Cosmos World Foundation Model Platform for Physical AI*, 2025（arXiv:2501.03575）——主要具身世界模型基线。
5. Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*（MMDiT / SD3），ICML 2024——双流 MMDiT 骨干来源。
