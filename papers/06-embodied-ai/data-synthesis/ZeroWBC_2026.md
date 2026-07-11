# ZeroWBC：从人类第一视角数据中学习自然的全身人形交互

> **论文**：*ZeroWBC: Learning Natural Whole-Body Humanoid Interaction from Human Egocentric Data*
>
> **作者**：Haoran Yang, Jiacheng Bao, Yucheng Xin（共同一作）, Haoming Song, Yuyang Tian, Bin Zhao, Dong Wang（通讯）, Xuelong Li
>
> **机构**：中国科学技术大学；上海人工智能实验室；清华大学；上海交通大学；TeleAI（中国电信）
>
> **发布时间**：2026 年 06 月（arXiv 2603.09170，v3）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.09170) | [PDF](https://arxiv.org/pdf/2603.09170)
>
> **分类标签**：`人形全身控制` `人类第一视角视频` `生成-跟踪` `免遥操作数据`

---

## 一句话总结

ZeroWBC 用"胸挂相机 + 动捕服"采集的**人类第一视角视频（含同步全身动作与文本标注）**替代昂贵的人形整机遥操作，走"先生成后跟踪"两阶段路线：微调 Qwen2.5-VL-3B 从初始 egocentric 图像 + 语言指令自回归生成未来全身动作 token，解码重定向到 Unitree G1，再由一个强调交互轨迹的通用跟踪策略执行——在无任何机器人遥操作示范的前提下，实机障碍规避 96%、坐沙发 84%，并对训练中完全未见的"椅子导航/坐椅"零样本达 90.0% / 76.0%。

## 一、问题与动机

人形机器人要在真实场景做**面向物体与场景几何的全身交互**（坐、踢、搬箱、避障），但现有路线两头都不理想：

- **依赖大规模机器人遥操作数据**：整机全身遥操作既贵又难，常被迫采用上下半身**解耦控制**，动作不协调、易受 sim-to-real gap 影响；
- **仿真中任务专用 RL**：只能覆盖狭窄任务，泛化差。

作者的核心观察是**数据成本的巨大差异**：论文给出的对比是——人形整机遥操作在同一任务上约需 2 名操作员、8 小时才能采到约 100 条成功示范（VR 遥操作还常因抓取困难、掉箱、失衡而失败）；而 1 名演示者戴动捕服 + 胸挂相机，2 小时就能采约 300 条 egocentric 示范。人类第一视角视频天然提供了第一人称场景观测、稠密全身动作标签、任务级语义三者合一，且更便宜可扩展。

由此提出**免遥操作（teleoperation-free）**的静态场景全身交互控制框架 ZeroWBC。之所以用"生成全部未来动作后再一次性跟踪"的**开环**设计，是因为当前 VLM 推理时延高（后文测得端到端约 400 ms），做不了高频闭环；而多数静态交互任务在初始观测里已含足够空间上下文，开环可行。

## 二、核心方法

框架分两阶段（见原文 Fig. 2）：**阶段一多模态动作生成**，**阶段二通用交互动作跟踪**。

### 2.1 动作 tokenizer（VQ-VAE）

先把连续 SMPL 人体动作离散成 token 供自回归生成。每段动作 $\mathbf{m}$ 表示为 SMPL 姿态 $\mathbf{m}_{\text{smpl}} \in \mathbb{R}^{66}$ 与全局平移 $\mathbf{m}_{\text{trans}} \in \mathbb{R}^{3}$。ResNet 编码器把长度 $n_i$ 的片段下采样到长度 $n_t$ 的隐序列（$n_i = n_t \cdot 2^{n_{\text{down}}}$），码本大小 2048。训练损失同时含重建、commitment、速度、根旋转、根平移五项：

$$
\mathcal{L}_{\text{VQ}} = \lambda_r \lVert \mathbf{m}-\hat{\mathbf{m}}\rVert_1 + \lambda_c \lVert \text{sg}[\mathbf{z}_e]-\mathbf{z}_q\rVert_2^2 + \lambda_v \lVert \Delta\mathbf{m}-\Delta\hat{\mathbf{m}}\rVert_1 + \lambda_{rr}\lVert \mathbf{m}_{0:3}-\hat{\mathbf{m}}_{0:3}\rVert_1 + \lambda_p\lVert \mathbf{m}_{\text{trans}}-\hat{\mathbf{m}}_{\text{trans}}\rVert_1
$$

用大白话说：不仅要求重建出姿态，还专门用速度项保证时序平滑、用根旋转/根平移项保住全局朝向和轨迹——因为后面"去哪儿、朝哪坐"这类空间信息全靠根轨迹，不能被压缩掉。

### 2.2 微调 VLM 做动作生成

以 **Qwen2.5-VL-3B** 为骨干，把 VQ-VAE 码本索引变成词表里的特殊 token（如 `<motion_token_1>`、`<motion_start>`、`<motion_end>`），从而把全身动作生成变成标准的 next-token 预测。给定初始 egocentric 图像 $\mathbf{v}$ 和语言指令 $\mathbf{t}$，模型**一次生成**整段未来动作 token 序列（对应开环规划），目标是最小化负对数似然：

$$
\mathcal{L}_{\text{gen}} = -\sum_{i=1}^{T}\log P_\theta\!\left(z_i \mid \mathbf{v}, \mathbf{t}, z_{1:i-1}\right)
$$

其中 $z_{1:i-1}$ 为已生成的动作 token。用大白话说：把"看一眼场景 + 听一句指令 → 想象出接下来整套全身动作"当成机器翻译来学，图像给空间接地、语言给任务语义。训练用**两阶段微调**：先在大规模公开图文-动作数据上学通用跨模态对齐，再在自采 egocentric 数据上做领域微调以增强空间接地；且只微调 LM decoder、其余冻结，保住预训练的视觉-语言对齐能力（用 32 张 A100）。

### 2.3 通用交互动作跟踪（RL）

低层策略要跟踪生成动作。跟踪目标是一个复合向量 $G = \{\mathbf{g}_{\text{inter}}, \mathbf{m}_{\text{ref}}\} \in \mathbb{R}^{D_G}$，拆成两部分：

- **关键身体部位交互轨迹** $\mathbf{g}_{\text{inter}} = \{p_k, \dot{p}_k, w_{\text{root}}\} \in \mathbb{R}^{33}$，其中 $p_k, \dot{p}_k \in \mathbb{R}^3$ 是关键部位 $k \in \{\text{root, L-wrist, R-wrist, L-foot, R-foot}\}$（即根、左右腕、左右脚）的全局位置与线速度，$w_{\text{root}}$ 是根角速度；
- **重定向参考动作** $\mathbf{m}_{\text{ref}} = \{q_{\text{ref}}, \dot{q}_{\text{ref}}\} \in \mathbb{R}^{2n_j}$，关节角与角速度，$n_j = 23$。

**关键设计（Interaction-oriented reward）**：奖励**优先对齐全局根轨迹与关键末端部位的世界坐标轨迹**，同时**弱化**对参考动作关节空间偏差的惩罚（Table 7 中腕/脚交互位置项权重 1.0，参考关节位置项仅 0.1）。直觉是：人到机器人的形态/关节限位/接触差异使纯关节跟踪虽能复现姿态却对不准任务关键点（搬运的手、坐下的躯干），所以宁可放松关节精度，也要保住"手/脚/身体中心去到该去的世界坐标"。

三个训练技巧提升稳定性与泛化：

1. **自适应动作调度**：每段 clip 按当前跟踪难度动态采样，难度分综合 EMA 跟踪误差与成功率——$r_i = (1-w)\,\text{clip}\!\left(\tfrac{E_i}{c},0,1\right) + w\,(1-\hat{p}_i^{\text{succ}})$，再经温度化 softmax（含均匀探索项）转成采样概率，多练难的又不丢覆盖面。
2. **渐进难度暴露（curriculum）**：把动作按语义/运动学复杂度分 1–10 级（1=站立挥手，8=爬行跪姿，9–10=侧翻后空翻跳舞），从最低级起，满足 $\text{MPJPE}_{l_{\max}} < \theta_{\text{pos}}$ 且 $\text{MPJAE}_{l_{\max}} < \theta_{\text{ang}}$ 才解锁上一级，并对已解锁级设最小采样比防遗忘。
3. **未来动作时序编码 + 非对称 PPO**：critic 训练时看特权参考与未来动作状态，actor 部署时只用紧凑观测（本体感受 + 当前/短程 2 帧/长程 5 帧目标位姿），长程目标经轻量时序卷积编码，帮策略预判速度变化与接触事件。actor 观测维度 880，跟踪策略在 2 张 RTX 4090 上训约 14 天。

## 三、实验结果

数据：**Nymeria**（约 300 小时同步 egocentric 视频/全身动作/语言，切成 5–10 s 片段得约 14 万训练 + 2 万测试样本，供 VQ-VAE 与一阶段 VLM 微调）、**HumanML3D**（14,616 条文本标注动作，约 28.59 h，做文本-动作对齐并用 GMR 重定向到 G1 后预训练跟踪器）、**自采数据集**（约 5 小时，覆盖搬箱/踢球/坐沙发/避障，约 1500 条，供二阶段 VLM 与任务自适应微调）。

### 动作生成质量（Table 1，训练数据消融）

| 测试集 | 训练数据 | FID↓ | R@3↑ | MM-Dist↓ |
|---|---|---|---|---|
| Nymeria | Nymeria | 0.756 | 0.755 | 3.124 |
| Nymeria | +HumanML3D | 0.423 | 0.828 | 2.514 |
| Nymeria | +Self | **0.298** | **0.847** | **2.286** |
| Self | Nymeria | 0.784 | 0.782 | 3.456 |
| Self | +HumanML3D | 0.628 | 0.825 | 3.128 |
| Self | +Self | **0.245** | **0.892** | **2.456** |

公开大数据提供通用动作先验，自采 egocentric 数据提升领域内交互空间接地。**输入模态消融（Table 6）**上，"图+文"全面优于单模态（Nymeria FID：仅图 0.925 / 仅文 0.518 / 图+文 0.298）——语言给任务语义、egocentric 图像给场景级空间上下文，缺一不可（实机纯文/纯图输入成功率近乎为零）。相比纯文本基线 **MotionGPT**（FID 0.869 / R-Prec 0.536 / MM-Dist 3.139），ZeroWBC 达 0.756 / 0.755 / 3.124。

### 关节级跟踪（Table 3 / 11，越低越好）

| 数据 | 方法 | MPJPE↓ | MPJAE↓ | MPJVE↓ | MPIPE↓ | MPIVE↓ |
|---|---|---|---|---|---|---|
| HumanML3D | GMT | 0.5530 | 0.1046 | 0.4882 | 0.3821 | 0.3216 |
| HumanML3D | SONIC | 0.2526 | 0.0957 | 0.3906 | 0.2108 | 0.1992 |
| HumanML3D | **Ours** | 0.2571 | **0.0734** | 0.4378 | **0.1885** | **0.1753** |
| Generation | GMT | 0.6013 | 0.1375 | 0.5112 | 0.4219 | 0.3654 |
| Generation | SONIC | **0.3340** | 0.1058 | 0.4691 | 0.2671 | 0.2318 |
| Generation | **Ours** | 0.3653 | 0.1074 | 0.4680 | **0.2138** | **0.2005** |

其中 MPIPE / MPIVE 为**交互身体部位**位置/速度误差。相比更强的 SONIC，本方法在纯关节位置 MPJPE 上不总是最优（SONIC 更侧重整体关节跟踪），但在两项**交互指标 MPIPE/MPIVE 上全数据集最优**——正是"优先保交互轨迹"的设计取舍所致。消融中去掉交互目标（w/o Interact）虽能略降 MPJPE，却显著恶化 MPIPE/MPIVE，佐证核心论点。

### 实机（Unitree G1，Table 4）

| 任务 | 通用跟踪器 | 交互跟踪器 |
|---|---|---|
| 障碍规避 | 80% | **96%** |
| 踢球 | 60% | **78%** |
| 坐沙发 | 48% | **84%** |
| 搬箱 | 16% | **64%** |

每次试验随机初始化物体摆放与机器人初始位姿（防止简单重放固定轨迹），只保证目标物出现在初始 egocentric 视野内。交互跟踪器（通用跟踪器基线为 SONIC）全面胜出，搬箱任务提升最大（16%→64%）。**零样本**：训练数据中已过滤所有含椅子的样本，机器人对未见的"椅子导航/坐椅"（50 次试验、4 种椅型）达 **90.0% / 76.0%**，说明 Qwen-VL 为未见物体/任务提供了有用语义先验。

## 四、局限性

- **仅适用于大体静态场景**：开环"先生成后跟踪"，初始观测后若物体或人移动则无法重规划。
- **时延**：即便短程生成，端到端也约 **400 ms**（限制到 4 个 token 时 Qwen-VL 推理约 300 ms，加图像传输/解码/重定向），会引入相位滞后与抖动，做不了高频闭环。
- **接触密集任务弱**：搬箱常能抓起却在搬运中滑落，因手部摩擦有限且**无力/触觉反馈**。
- **本体单一**：仅在 Unitree G1 上、任务集小；迁移到其他人形需形态感知重定向与跟踪器适配。

## 五、评价与展望

**优点**：（1）动机切中要害——把"人形整机全身遥操作太贵"这一数据瓶颈用人类 egocentric 视频绕开，成本对比（2 名操作员 8 h/100 条 vs 1 人 2 h/300 条）有说服力，属数据来源层面的范式转换。（2）"交互导向奖励"的取舍很清晰：放弃部分关节保真度换取任务关键点的世界坐标对齐，并用 MPIPE/MPIVE 与 w/o Interact 消融直接证伪替代方案，实验闭环干净。（3）零样本坐椅（训练已剔椅）是较强证据，说明收益部分来自 VLM 的开放世界语义而非过拟合。

**不足与开放问题**：（1）与同期通用跟踪工作（SONIC、GMT、BeyondMimic 等）相比，本文**低层跟踪器并非全新**，SONIC 在纯关节 MPJPE 上仍更优，本文贡献更多在"生成模块 + 交互导向的目标/奖励定义 + 数据管线"，而非跟踪算法本身。（2）开环 + 400 ms 时延是硬伤，把方法框死在静态任务，与真正"动态场景交互"仍有距离；作者自列的模型蒸馏/闭环重规划是必需的下一步。（3）无力/触觉反馈导致接触密集任务（搬运）掉链子，纯视觉-运动学监督天花板明显；接入力/触觉或接触感知跟踪是自然方向。（4）自采数据仅约 5 小时、任务 4 类，泛化边界未充分探明；GPT-5 改写文本、GMR 重定向、SMPL→G1 的形态 gap 也都会累积误差。

**与公开工作的关系**：思路上承接 OmniH2O、Twist2 等人到人形迁移与 VideoMimic 的视觉模仿，但区别于依赖硬件遥操作或第三人称观测——强调"第一人称视频 + 同步全身动作"这一低成本监督源，并把 MotionGPT 式离散 token 生成与 RL 通用跟踪拼成端到端交互管线。对"如何用海量廉价人类第一视角数据规模化训练场景交互人形"是一次可复用的样板。

## 参考

1. Z. Luo et al. *SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control.* arXiv:2511.07820, 2025.（本文通用跟踪器基线）
2. T. He et al. *OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning.* arXiv:2406.08858, 2024.
3. B. Jiang et al. *MotionGPT: Human Motion as a Foreign Language.* NeurIPS, 2023.（离散动作 token 生成基线）
4. L. Ma et al. *Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild.* ECCV, 2024.（主训练数据）
5. J. P. Araujo et al. *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking (GMR).* arXiv:2510.02252, 2025.（人到 G1 重定向）
