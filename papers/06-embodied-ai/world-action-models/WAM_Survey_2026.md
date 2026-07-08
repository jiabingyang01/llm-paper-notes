# World Action Models 综述：具身智能的下一个前沿

> **论文**：*World Action Models: The Next Frontier in Embodied AI*
>
> **作者**：Siyin Wang、Junhao Shi、Zhaoyang Fu、Xinzhe He、Feihong Liu、Chenchen Yang、Yikang Zhou、Zhaoye Fei、Jingjing Gong、Jinlan Fu、Mike Zheng Shou、Xuanjing Huang、Xipeng Qiu、Yu-Gang Jiang
>
> **机构**：复旦大学（可信具身智能研究院）、上海创智学院、新加坡国立大学；OpenMOSS
>
> **发布时间**：2026 年 5 月（arXiv 2605.12090）
>
> **发表状态**：未录用（arXiv 预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.12090) | [PDF](https://arxiv.org/pdf/2605.12090) | [项目主页 Awesome-WAM](https://openmoss.ai/Awesome-WAM/) | [代码/Repo](https://github.com/OpenMOSS/Awesome-WAM)

---

## 一句话总结

这是第一篇系统梳理 **World Action Models（WAMs，世界动作模型）** 的综述：把 WAM 定义为"联合预测未来状态与动作、目标是联合分布 $p(o', a \mid o, l)$ 而非仅动作 $p(a \mid o, l)$"的具身基础模型，将其与 VLA、世界模型、Video Action Model、Video Policy、Action World Model 逐一厘清边界；再按结构把现有方法组织成 **Cascaded（级联）** 与 **Joint（联合）** 两大范式（Joint 再细分自回归 / 扩散、单流 / 多流、显式 / 隐式未来预测），系统分析驱动 WAM 的四类数据（机器人遥操作、便携人类演示、仿真、互联网 egocentric 视频）与三维评估协议（视觉保真、物理常识、动作可行性），最后提出七大开放挑战。

---

## 一、问题与动机：为什么需要 WAM

标准 **VLA** 模型（RT-2、OpenVLA、π₀）把动作生成建模成"观测 → 动作"的条件 token 预测，在大规模视觉语言预训练之上取得了强语义泛化。但它们学的是**反应式映射** $p(a \mid o, l)$，**不显式建模世界在干预下如何演化**——缺乏对未来状态的预测性物理推理，泛化因此受限，尤其在需要预判后果的接触密集 / 长时程任务上。

给策略装上世界模型能力就成了自然方向：用环境动力学的预测模型给智能体"物理前瞻"。这条路快速积累出一大批方法（视频规划、潜在动力学条件化、统一状态-动作生成），但文献碎片化、术语混乱、缺乏统一概念框架。本综述要做的就是：**给这个新范式一个正式定义、一张设计空间地图、一份进场指南**。

---

## 二、定义与形式化：WAM 到底是什么

### 2.1 三个基本分布

综述用一个概率视角统一具身任务。智能体每步收到观测 $o$、语言指令 $l$，产生动作 $a$，下一步观测记为 $o'$。三类模型的目标函数：

$$\mathcal{L}_{\text{VLA}} = \mathbb{E}\big[-\log p(a \mid o, l)\big]$$

$$\mathcal{L}_{\text{WM}} = \mathbb{E}\big[-\log p(o' \mid o, a)\big]$$

$$\mathcal{L}_{\text{WAM}} = \mathbb{E}\big[-\log p(o', a \mid o, l)\big]$$

- **VLA**：只学反应式的观测→动作映射。
- **World Model（WM）**：正向动力学，给定状态和动作预测未来状态，充当"状态的概率传播器"。
- **WAM**：联合建模未来状态 **与** 动作，用丰富时空先验换取更深的物理理解和更强的零样本泛化。

### 2.2 WAM 必须满足的两个硬条件

1. **正向预测建模（Forward Predictive Modeling）**：必须以可量化的未来状态 $o'$ 表征预报环境的物理演化——可以是显式视觉预测（像素级视频、稠密光流），也可以是隐式物理表征（physics-grounded 潜在）。
2. **耦合动作生成（Coupled Action Generation）**：必须通过严格对齐"预期未来状态 $o'$"来推导动作 $a$——耦合可表现为联合概率输出，或级联 / 统一潜在架构里的策略条件化。

### 2.3 与相邻概念的辨析（这是综述最有价值的部分之一）

| 概念 | 与 WAM 的关系 | 关键区别 |
| --- | --- | --- |
| **Video Action Model（VAM）** | WAM 的子集 | VAM 特指"用合成视频帧对齐动作"；WAM 更宽——视频只是建模世界的一种代理，还可用单图状态转移、点云、触觉/力等模态 |
| **Video Policy** | 部分重叠但不等同 | Video Policy 由"结构血统"定义（用视频生成骨架如 DiT 抽时空表征）；WAM 骨架无关，要求**主动的预测性承诺**——合成下一状态 $o'$ 必须是模型推理与输出的显式组成，而非骨架里的隐式特征 |
| **Action World Model（AWM）** | 功能相似，层级不同 | AWM 里名词是"世界模型"（被增强的模拟器）；WAM 把系统重定位为一级的 Agent 范畴，"世界"（预测物理）与"动作"（运动控制）co-equal，是 VLA 血统的直接概念继承者 |

---

## 三、背景：从模型基 RL 到基础模型

综述追溯了世界建模与动作生成两条线的交织演化：

- **动作条件世界模型** $P(o' \mid o, a)$：显式像素预测（ACVP、CDNA、Deep Visual Foresight、SV2P）→ 自回归视频世界模型（iVideoGPT、Genie 用潜在动作模型从无标注视频学）→ 扩散视频世界模型（缓解逐步误差累积）；隐式潜在动力学（RSSM/PlaNet、Dreamer 系列、TSSM/TransDreamer）与预测表征学习（JEPA、I-JEPA、V-JEPA 2、LeWorldModel）。
- **语言条件世界模型** $P(o' \mid o, l)$：视频基础模型演化（GAN → U-Net/VDM → ViT/Sora/Latte → VAE/LDM/VideoGPT → Wan、Sora 2、Kling 3、Veo 3、Gen-4），把互联网世界知识注入世界建模。
- **具身世界模型**：SWIM、DreamDojo（连续潜在动作）、DexWM、RoboDreamer（语言组合分解）、IRASim、Ctrl-World、RoboScape（物理感知）、WoW（SOPHIA 自优化 + Flow-Mask 逆动力学闭环）、VT-WM（触觉）、PointWorld（3D point-flow）。
- **世界模型服务于 VLA 的两种角色**：
  - **学习**：为模仿学习造数据（DREMA、Ctrl-World、RoboScape）；为 RL 当代理环境（Dreamer 系列）；提供奖励信号（VIPER、Diffusion Reward、GenReward、SRPO；RoboScape-R 内生奖励）。
  - **评估**：当数据驱动模拟器做策略回滚测试（Ctrl-World、Veo Robotics、Interactive World Simulator、WorldEval、WorldGym）。

关键转折：世界模型从"策略外部的工具（模拟器 / 奖励 / 基准）"内化为"策略内部的预测核心"，才诞生了能实时推理世界动力学的 WAM。

---

## 四、架构分类：两大范式（综述的核心骨架）

### 4.1 Cascaded WAM（级联世界-动作模型）

显式因式分解目标 $p(o', a \mid o, l) = p(a \mid o', o, l)\, p(o' \mid o, l)$：**先合成表征预期未来的视觉规划，再由独立动作模型从中解码可执行命令**。两阶段解耦训练，天然归纳偏置——世界模型不必推理机器人运动学，动作模型不必解长时程场景预测；代价是两阶段间的耦合约束了每个设计决策。按中间规划载体分两类：

**(1) 显式规划（像素空间表征）**：中间是 RGB 视频帧。

- *学习式动作抽取*：UniPi（文本条件视频 + 卷积 IDM 回归动作）奠定两阶段蓝图；VLP（VLM 分层子目标 + 树搜索）、RoboEnvision（非自回归关键帧）缓解语义漂移与误差累积；ThisThat（指示代词消歧）、Say-Dream-and-Act（对抗蒸馏 + 帧率无关规划）；TesserAct（视频 + 深度 + 法向的 4D 规划）、MVISTA-4D（轨迹级潜在优化 + 残差 IDM）扩展规划表征；Vidar、Gen2Act（零样本 VideoPoet + 闭环神经策略）、Veo-Act（门控：粗导航用 Veo-3 生成、接触时交给反应式 VLA）、VAG（1D U-Net 动作解码器）、π₀.₇（第二阶段 IDM 换成预训练 VLA）。
- *几何式动作抽取*（把动作抽取从逆动力学问题变成解析几何问题）：AVDC（光流 + SE(3) 解析）、Im2Flow2Act（潜在光流）、3DFlowAction（3D 光流）、NovaFlow / Dream2Flow（零训练，预训练视频生成 + 深度/点跟踪）、Dreamitate / RIGVid（位姿跟踪 + FoundationPose）、4DGen、LVP（3D 手部 + Diffusion Forcing）。

**(2) 隐式规划（潜在表征）**：中间是压缩潜在特征序列，省掉解码回像素的开销，利于实时部署。VPP（预训练 VAE + 单步潜在预测 + 轻量策略，首次实时）、VILP（多视图潜在）、S-VAM（自蒸馏把多步 SVD 教师压成单次前向）、Video Policy（冻结视频 U-Net + 独立动作 U-Net）、ARDuP、mimic-video（flow matching + 部分去噪）、MWM（语义 mask 潜在，抗视觉扰动）、LAPA（VQ-VAE 潜在动作预训练）、villa-X（proprio-FDM + 联合扩散）、OmniVTA（视觉-触觉双流）。

### 4.2 Joint WAM（联合世界-动作模型）

直接建模联合分布 $p(o', a \mid o, l)$，世界建模与动作生成在**共享表征里联合训练**，强制模型内化环境动力学与控制信号的因果互依。按生成机制分两支：

**(1) 自回归生成**：把世界变量和动作变量序列化成统一时间序列，因果左到右预测。核心张力是缓解灾难性误差传播 + 平衡序列解码的计算瓶颈与实时低延迟。三种表征范式：

- *显式解耦表征*：GR-1（195M，GPT 式因果 Transformer，双分支解码未来 patch + 连续动作）、GR-MG（PROG token 分层）、GR-2（VQGAN 离散视觉 + CVAE 动作分块）。
- *统一离散表征*（动作和图像都量化进单一 LLM 词表）：CoT-VLA（7B，先因果自回归幻想视觉 CoT，再切全注意力预测动作）、WorldVLA（7B Chameleon，模态特定因果掩码阻断动作块误差）、RynnVLA-002（5B + 轻量连续动作头）、F1（4.2B MoT，foresight-guided 逆动力学）。
- *预测式潜在表征*：VLA-JEPA（2B，抽连续潜在动作 token 条件化自回归世界模型，用冻结 target 网络预测未来表征，结构上无泄漏）。

**(2) 扩散式（非自回归）生成**：多步生成过程捕捉复杂未来分布，突破自回归的序列瓶颈，实现高频闭环。按预测流的结构耦合分两族（见下方"综述架构地图"）：

- **单流架构**（世界与动作吸收进单个 DiT）：
  - *显式未来预测*：PAD（拼接未来图像潜在 + 动作 token，从零训 + 无动作视频协同）、VideoVLA（复用 CogVideoX-5B）、**UWM**（给世界/动作变量独立噪声调度 → 一个模型可切策略/正逆动力学/纯视频）、DreamZero（14B Wan2.1，KV-cache 观测替换，7 Hz）、Cosmos Policy（2B，潜在帧注入，单 ckpt = 策略 + 世界模型 + 价值）、GigaWorld-Policy（只注意历史/当前，推理时不生成未来视频以加速）、X-WAM（交错深度分支 + 异步噪声采样）、UD-VLA（离散扩散联合 mask-and-predict）。
  - *隐式未来预测*：FLARE（可学习未来 token 对齐冻结 teacher 的未来 embedding）、FRAPPE（RDT 上的 Mixture-of-Prefix-and-LoRA）。
- **多流架构**（世界-动作耦合上升为持久的架构分解）：
  - *交叉注意力耦合*：CoVAR（Bridge Attention）、**LDA-1B**（共享 MM-DiT 注意力 + 模态特定投影，DINO 潜在，质量分层监督）、**DUST**（MM-DiT + 独立噪声时间步 + 异步采样）、LingBot-VA（MoT + KV-cache 历史）、DexWorldModel（双态 TTT 记忆，DINOv3）、AIM（intent-causal 注意力掩码）、Motus（三模态联合注意力 + 语义专家）、MotuBrain（H-bridge + 3D RoPE）、AdaWorldPolicy（力预测第三专家）。
  - *隐状态耦合*（世界分支产出内部隐状态供动作分支条件化）：DiT4DiT（hook 算子 + 三时间步）、**Fast-WAM**（推理时丢掉未来视频分支，只保训练时耦合）、WAV（轨迹-价值分支）、Act2Goal（多尺度时序哈希 + HER 目标重标注）。
  - *共享表征 / 统一编码器*：UVA（0.5B，掩码训练，两个扩散头）、PhysGen（0.73B NOVA，因果掩码 + lookahead 多 token 预测）。

> **综述架构地图（Fig 6，扩散式 Joint WAM 的主结构）**
>
> 1. **单流**：$V$ 与 $A$ token 共走一个 DiT，未来建模隐式或显式。
> 2. **多流-交叉注意力耦合**：独立 Video DiT 与 Action DiT，靠显式 cross-attention 交互。
> 3. **多流-隐状态耦合**：Video DiT 的中间隐状态条件化 Action DiT。
> 4. **多流-共享表征**：视觉与动作先经统一编码器融合，再各自解码。

本仓库已有笔记的坐标：[WorldVLA](WorldVLA_2025.md) = 自回归统一离散；[Fast-WAM](FastWAM_2026.md) = 多流隐状态耦合；[LDA-1B](LDA_1B_2026.md) 与 [DUST](../vla/reasoning/DUST_2026.md) = 多流交叉注意力耦合；[FLARE](../vla/reasoning/FLARE_2025.md) = 单流隐式未来预测。

---

## 五、训练数据：四类范式的权衡

WAM 独特优势在**统一数据摄入**——既用高质量 $(o_t, a_t, o_{t+1})$ 三元组紧耦合内部表征，又能靠联合训练策略吃海量无配对数据（如无动作视频）。综述用"迁移难度（Y 轴）× 扩展难度（X 轴）"给数据版图定位（Fig 7）：

| 数据范式 | 特点 | 代表 |
| --- | --- | --- |
| **机器人遥操作** | 高质量、精确运动学接地、几乎零 sim2real gap；但采集贵、构型受限 | OXE（100 万+ 轨迹 / 22 机器人）、DROID、RoboMIND、AgiBot World、BridgeData、RT-1；UnifoLM-WBT（高 DoF 人形） |
| **便携人类演示（UMI 式）** | 良好保真、低成本、in-the-wild 多样；手持夹爪 + 可穿戴相机 | UMI、FastUMI（10 万+ 轨迹）、RealOmin、Hoi!、RDT2（~1 万小时） |
| **仿真** | 易扩展、完全可控、有特权信息（精确深度 / 物体位姿 / 无遮挡多视图）；有域差 | MimicGen、RoboCasa、RoboTwin 2.0、InternData-A1（63 万轨迹）、SynGrasp-1B（1000 万抓取）、TLA（触觉-语言-动作） |
| **人类 / Egocentric** | 近乎无界规模、被动世界动力学先验；但迁移难度大 | Ego4D（3670h）、HowTo100M（1.36 亿 clip）、Kinetics-700、DreamDojo-HV（43,827h）、EgoDex（829h）、Ego-Exo4D、EgoScale、EgoVerse |

数据构建的核心不再是"堆机器人数据"，而是**战略性混合**——把严格耦合的少量演示与无约束的海量观测混起来，弥合"精确低层控制 ↔ 广域开放世界泛化"的鸿沟。最新趋势是把分散数据集聚合成单一学习引擎的通才预训练混合物（Ego-Centric Human Manipulation、UniHand、EgoDex、Humanoid Everyday、PH²D）。

---

## 六、评估：解耦的三维协议

现状是**解耦评估**——世界建模能力与动作策略能力分开、按模块专属指标测，尚无联合评估协议。

### 6.1 世界建模能力（三个并行维度）

- **视觉保真**：像素级 PSNR / SSIM，感知/语义级 LPIPS / DreamSim / DINO 相似度，分布级 FVD。
- **物理常识**：物体动力学（VideoPhy、PhyGenBench、VBench-2.0、WorldModelBench、Physics-IQ）、运动与轨迹可信度（WorldScore 分解运动准确/幅度/平滑，EWMBench 用 EEF 轨迹 HSD/nDTW/DYN）。
- **动作可行性**：最独特的一维——生成视频是否保留了足够动作相关信息以支持控制推理。WorldSimBench（隐式操作评估）、Wow-wo-val!（**IDM 图灵测试**：对生成视频跑逆动力学得动作，再看真机执行成功率——发现很多"视觉上很可信"的模型在此测试下成功率坍到接近零，凸显"视觉可信 ≠ 动作可执行"）。

### 6.2 动作策略能力（40+ 基准，5 类）

综述以近 40 个 2019–2026 主流基准系统评述（Table 9），按机器人形态和场景分五类：

- **通用操作**：MetaWorld、RLBench、LIBERO（及 LIBERO-plus / pro / X）、ManiSkill 系列、RoboCasa、RoboVerse、COLOSSEUM、AGNOSTOS、GemBench、SimplerEnv、PolaRiS、RoboMME（记忆）。
- **双臂 / 人形**：RoboTwin、BiGym、HumanoidBench、HumanoidGen。
- **移动操作**：ManipulaTHOR、HomeRobot、BEHAVIOR-1K。
- **接触 / 形变操作**：SoftGym、PlasticineLab、DaXBench；TacSL、ManiFeel（触觉）。
- **真机基准**：RoboArena、RoboChallenge、Maniparena（1 万+ 真机轨迹，目前最大真机操作基准）。

---

## 七、开放挑战与机会（七条）

1. **架构耦合无系统对比**：级联 / 联合扩散 / 离散 tokenization / 隐式对齐百花齐放，但没有在同等规模-数据-协议下的受控研究。**显式像素预测是否物理接地所必需？** 一条有前景的方向是 latent-predictive（JEPA 式）——预测抽象未来状态而非重建高维像素，绕开像素预测瓶颈。
2. **多模态物理状态表征**：现有 WAM 几乎都在 RGB 里预测未来，而接触密集操作最关键的物理信息（触觉分布、接触力、材料顺应性）在像素里几乎不可见——世界模型的盲区正好落在最该建模的物理交互上。需要**模态自适应预测**：有触觉/力流时生成 physics-grounded 预测，没有时优雅退化到纯视觉。
3. **数据利用与混合设计**：各数据源的边际贡献是其规模与域差的什么函数？人类视频预训练的收益主要是语义还是动力学？需要把人类视频的可迁移知识拆成低层物理先验 / 中层因果动力学 / 高层任务逻辑三层，并发展**构型感知过滤**——选择性蒸馏通用物理律、抑制运动学不兼容的行为。
4. **长时程规划与时序抽象**：WAM 多在短时程测；长时程有分布漂移累积、误差复合、长轨迹连续生成计算/架构不可行三重挑战。三条路：模块化层级（WAM 当低层执行器 + VLM 高层规划）、内生层级 WAM（多分辨率未来预测）、扩展时序上下文。
5. **推理延迟与计算效率**："延迟税"威胁闭环可行性——DreamZero 优化到 7 Hz 仍远低于非生成 VLA 的 50 Hz。理论问题：下游控制到底需要多少预测保真度？指向**任务自适应预测保真度**与"最小充分世界模型"。
6. **评估方法论**：世界建模按像素指标测（允许物体悬浮/流体违反重力却拿高分）、动作只按任务成功测，二者脱节，恰恰漏掉 WAM 的核心前提。缺失的是**量化"想象未来 ↔ 生成动作"因果一致性的联合指标**——如 Counterfactual Consistency（动作是否随想象未来的扰动而适配）、Foresight-Conditioned Success。
7. **安全与可靠物理部署**：预测能力越强，错误的物理想象越可能诱发难以中断/恢复的长动作序列后果。但预测性也带来机会——**prediction-integrated safety**：把对想象未来的不确定性估计当作安全监控的一级输入，在执行前用物理约束核验预测。

---

## 八、个人思考

- **这篇综述最大的贡献是"立范式 + 划边界"**：WAM 之前，UniPi、GR-1、UWM、DreamVLA、FLARE、WorldVLA 各说各话，"video policy / world model / action world model / video action model"术语混用。综述用 $p(o', a \mid o, l)$ 这一个联合分布把它们收编，并用"是否有主动预测承诺、世界与动作是否 co-equal"两条判据把 WAM 从 Video Policy / AWM 里切出来——这套判据本身就很有 load-bearing 的价值。
- **Cascaded vs. Joint 的二分抓住了真正的设计轴**：级联（先想象未来、再解码动作，两阶段解耦）归纳偏置清晰但受两阶段耦合约束；联合（共享表征强制内化因果互依）表达力强但训练/推理更难。Joint 里再按"自回归 vs 扩散"和"单流 / 多流 / 显式 / 隐式未来"细分，几乎能给本仓库每一篇相关笔记精确定位（见 §4.2 末尾的坐标表）。
- **对本仓库的直接用处**：它是一张"把 [LDA-1B](LDA_1B_2026.md)、[DUST](../vla/reasoning/DUST_2026.md)、[Fast-WAM](FastWAM_2026.md)、[WorldVLA](WorldVLA_2025.md)、[FLARE](../vla/reasoning/FLARE_2025.md)、[DreamVLA](../vla/reasoning/DreamVLA_2025.md)、[SpatialVAM](SpatialVAM_2026.md) 摆到同一坐标系"的地图。之后读到任何一篇 WAM/世界模型新作，都可以先问：级联还是联合？自回归还是扩散？未来预测是显式像素还是隐式潜在？世界与动作在哪一层耦合（共享注意力 / 隐状态 / 共享编码器）？
- **三个我认为最尖锐的开放问题**：(1) 开放挑战 #1 与 #6 其实是同一枚硬币——"显式像素预测是否必需"没法回答，正因为缺"联合评估因果一致性"的指标；Fast-WAM 已经用"推理时丢掉视频分支不掉点"给出反例，暗示训练期梯度可能才是世界建模的主要红利。(2) #2 的多模态物理状态是真正的护城河问题：只要世界建模停留在 RGB，接触密集操作的天花板就摆在那。(3) #5 的延迟税决定 WAM 能不能真正上高频闭环控制——生成式前瞻与实时性之间的张力，可能才是 WAM 从"论文 SOTA"走向"真机默认范式"的最后一公里。
- **一个小遗憾**：综述自己承认现状是解耦评估，因此它给出的所有方法对比表（Table 1/2/3）基本是"结构 + 骨架 + I/O 模态 + 评测环境"的定性归类，**没有跨方法的统一成功率横评**——这不是综述的锅（本就没有公共联合基准），但也说明"WAM 到底谁更强"目前无法从文献直接回答，正呼应它自己提的开放挑战 #6。

---

## 参考

- **UniPi**（Du et al., 2023，NeurIPS，arXiv 2302.00111）：级联 WAM 的两阶段蓝图（文本条件视频 + IDM）
- **GR-1 / GR-2**（Wu et al., 2024）：自回归 Joint WAM 的显式解耦表征奠基作
- **UWM**（Zhu et al., 2025，arXiv 2504.02792）：单流扩散 Joint WAM，独立噪声调度实现模式切换，[LDA-1B](LDA_1B_2026.md) 的前身
- **WorldVLA**（Cen et al., 2025，arXiv 2506.21539）：自回归统一离散 Joint WAM，本仓库已有笔记
- **FLARE**（Zheng et al., 2025，arXiv 2505.15818）：单流隐式未来预测（对齐 teacher 未来 embedding）
- **Fast-WAM**（Yuan et al., 2026，arXiv 2603.16666）：多流隐状态耦合，质疑"测试时未来想象是否必需"，本仓库已有笔记
- **V-JEPA 2 / LeWorldModel**（Bardes et al., 2024）：latent-predictive 世界建模，综述开放挑战 #1 指向的替代路线
- **Wow, wo, val!**（2026）：提出 IDM 图灵测试，动作可行性评估的代表基准（对生成视频跑逆动力学再看真机执行成功率）
