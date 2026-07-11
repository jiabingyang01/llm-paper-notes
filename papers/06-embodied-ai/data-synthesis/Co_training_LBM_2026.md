# Co-training-LBM：面向机器人操作的大行为模型协同训练——数据模态与策略的系统性研究

> **论文**：*A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation*
>
> **作者**：Fanqi Lin, Kushal Arora, Jean Mercat, Haruki Nishimura, Paarth Shah, …, Jose Barreiros（通讯） et al.
>
> **机构**：Toyota Research Institute (TRI), Cambridge MA and Los Altos CA, USA；Tsinghua University, Beijing, China
>
> **发布时间**：2026 年 02 月（arXiv 2602.01067）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.01067) | [PDF](https://arxiv.org/pdf/2602.01067)
>
> **分类标签**：`co-training` `VLA` `数据模态消融` `large-behavior-model` `机器人操作`

---

## 一句话总结

一项迄今规模最大的机器人操作协同训练（co-training）对照实验：用约 4,000 小时机器人+人类数据与 50M 视觉-语言样本，训练并评测 89 个 VLA 策略（58,000 次仿真 + 2,835 次真机 rollout），系统回答"哪些异构数据模态、以何种阶段化策略混入才真正提升泛化"——结论是**多样化视觉-语言数据 + 跨本体机器人数据**收益显著（最终模型在仿真未见任务上 72.6% 成功率、较无 co-training 基线提升 36.4%），而**离散动作 token（FAST/VQ-VAE）与显式 CoT 条件化在其操作基准上无统计显著收益、甚至有害**。

## 一、问题与动机

大行为模型（Large Behavior Models, LBM）把模仿学习扩展到大规模多任务机器人数据上，具备了较强的灵巧操作能力，但泛化仍受限于**可获取机器人数据覆盖面不足**——机器人数据规模比训练 VLM 的互联网级文本图像语料小几个数量级（"data gap"）。业界的普遍做法是 **co-training**：把目标机器人数据与异构数据模态（标准 VL 数据、机器人轨迹语言标注、跨本体机器人数据、人类视频、离散动作 token）联合训练，以借入物理世界理解与泛化能力。

但现有工作各自只验证其中一个子集，且实验设定互不一致，导致一个基本问题悬而未决：**不同 co-training 模态、在训练的不同阶段引入、以何种策略组合，究竟如何影响策略性能？** 本文用统一架构、统一评测、严格统计检验，把这张"哪种信号有用、哪种策略最优"的经验地图系统地画出来。研究围绕五个问题展开：(1) 各模态在不同训练阶段对分布内/DS/未见任务/语言跟随的影响；(2) 有效模态叠加是否累积增益；(3) co-training 能否提升表征质量、支撑对未见长程灵巧任务的快速微调适应；(4) 有效模态如何塑造 VLM backbone；(5) 用 co-training 数据学到的 CoT 显式条件化动作生成是否有益。

## 二、核心方法

### 2.1 统一架构（VLM backbone + Action Flow Transformer）

策略 $\pi_\theta$ 输入 $n$ 帧图像序列 $I_t^{1:n}$ 与文本提示 $\ell$。backbone 由 **PaliGemma2-PT（google/paligemma2-3b-pt-224）** 初始化；动作头为 8 层 flow transformer（ActionFT），通过 adaLN MLP 注入观测条件与 flow timestep。关键设计：向 backbone 词表引入一个特殊 **observation encoding token** 追加到文本末尾，取该 token 在 VLM **最后四层**的隐状态拼成单一全局条件向量喂给 ActionFT——即用**一个紧凑 token** 而非 π0 式的全层 attention KV 做视觉语言条件，消融显示这种压缩表征反而增强了对未见任务和 DS 的泛化。动作 chunk horizon = 16。

连续动作用 **flow matching** 学习。给定动作块 $A_t$、flow 时刻 $\tau \in [0,1]$、噪声 $\epsilon \sim N(0, I)$，构造带噪动作 $A_t^\tau = \tau A_t + (1-\tau)\epsilon$，损失为

$$\mathcal{L}_{FM} = \left\|\pi_\theta^a(I_t^{1:n}, \ell, A_t^\tau, \tau) - (A_t - \epsilon)\right\|^2$$

> 用大白话说：让网络学会"从带噪动作指向干净动作"的那根箭头（flow 向量），推理时沿箭头一步步去噪就得到真实动作。

离散 token（文本 / 离散动作 token）用交叉熵：

$$\mathcal{L}_{CE} = \mathcal{H}\left(x_{1:M},\, \pi_\theta^\ell(I_t^{1:n}, \ell)\right)$$

联合优化时用带掩码的加权和：

$$\mathcal{L} = M_{FM}\,\mathcal{L}_{FM} + w \cdot M_{CE}\,\mathcal{L}_{CE}$$

其中 $M_{FM}$ 指示该样本是否需要预测连续动作，$M_{CE}$ 指定参与 CE 的 token 位置。

> 用大白话说：一个模型同时干两件事——对能给动作的样本走 flow matching，对要生成文字/离散 token 的样本走 next-token 预测，用掩码把两条损失按样本类型接通或断开。

### 2.2 三种阶段化 co-training 策略（Table 1）

| 策略 | 一阶段 | 二阶段 |
|---|---|---|
| Single-phase（单阶段） | 目标机器人连续动作 + co-training 数据 | — |
| Two-phase 1st-phase-only（两阶段·仅一阶段混入） | 仅 co-training 数据 | 仅目标机器人连续动作 |
| Two-phase full（两阶段·全程混入） | 仅 co-training 数据 | 目标机器人 + co-training 数据 |

论文的核心实验就是：把每个模态 $\times$ 每种策略跑遍，用统计检验判定"何种模态在何阶段引入最优"。

### 2.3 五大 co-training 数据模态（约 4,000 小时 + 50M VL 样本）

- **目标机器人数据 TRI-Ramen**（源自 [1]）：523 小时、403 任务、53,411 演示；含真机 TRI-Ramen-Real（478h/362 任务/46,063 演示）与仿真 TRI-Ramen-Sim（45h/41 任务/7,348 演示）；双臂 Franka Panda 遥操作采集，4 路 RGB，动作为末端位姿+夹爪宽度的相对轨迹。
- **标准视觉-语言数据**：RoboPoint（1.3M 样本 / 8.2M QA 对）+ RefSpatial（2.5M 样本 / 20M QA 对），均为空间指代类。
- **机器人轨迹稠密语言标注**：(1) 脚本标注（沿 [92] ECoT 思路，用启发式规则对 16 步 horizon 生成逐步低层动作原语)；(2) VLM 标注（用 **GPT-5** 生成上下文丰富的描述，语言多样性与物体-环境交互信息更强）。
- **跨本体机器人数据 OXE-Ramen**：Open X-Embodiment [55] 的精选子集，1,150 小时、12 种机器人配置、924 任务、466,415 演示。
- **人类视频**（两条路线）：(1) **Latent Actions**：在 Ego4D/EgoDex/Something-Something V2/Epic Kitchen/HoloAssist 上（过滤后 2,271 小时）训练 latent action model（LAM = IDM + FDM + ActionFDM，DINOv2 提视觉特征，码本 $C=32$，每段量化为 8 个 token）；(2) **VLM 生成标注**：GPT-5 对人类视频生成动作描述，9M 样本，也当作一种 VL 数据。
- **离散机器人动作 token**：(1) **FAST** [58]（现成 tokenizer，在 TRI-Ramen 上平均长度 42.1、词表 2,048)；(2) **VQ-VAE** [72]（动作块压成 8 个 token、码本 32，比 FAST 更紧凑）。

LAM 的核心方程：IDM 由相邻三帧特征推两段 latent action

$$Z_{t:t+\frac{\Delta t}{2}},\ Z_{t+\frac{\Delta t}{2}:t+\Delta t} = \text{IDM}\!\left(h_t,\, h_{t+\frac{\Delta t}{2}},\, h_{t+\Delta t}\right)$$

FDM 重建未来视觉特征、ActionFDM（仅机器人数据有真值时）重建真实动作块：

$$\hat{h}_{t+\frac{\Delta t}{2}} = \text{FDM}\!\left(h_t,\, Z_{t:t+\frac{\Delta t}{2}}\right),\qquad \hat{A}_{t:t+\frac{\Delta t}{2}} = \text{ActionFDM}\!\left(Z_{t:t+\frac{\Delta t}{2}}\right)$$

> 用大白话说：latent action 是从"这几帧之间画面怎么变"里反推出的一个离散"动作码"，用它同时预测未来画面（所有视频都有）和真实机器人动作（只有机器人数据有），从而把无动作标签的人类视频也变成可训练的动作监督信号。

### 2.4 显式 CoT 条件化（对照范式）

除把 co-training 数据当纯辅助监督外，还测试"先让 backbone 产出中间 CoT trace（脚本标注 / VLM 标注 / latent action），再以其为条件生成连续动作"。训练时以概率 $p$ 条件化、$1-p$ 直接生成；对比 50%-CoT Training-Only、50%-CoT with Inference、100%-CoT 三种设定。

## 三、实验结果

评测：仿真基准建于 Drake，13 个 seen + 8 个 unseen 任务，各 50 rollout，含 nominal 与 DS（光照/背景/相机参数/物体纹理颜色扰动）两种条件，指标为成功率；真机为双臂 Franka，含语言跟随（seen objects / 指令泛化 / unseen objects，共 49 seen + 52 unseen 物体）与三项未见长程灵巧任务（PackItemsIntoStringBag / PourIngredientsIntoSoup / StoreCleanDishes，约 13 步 / 93 秒，各 200 演示微调 / 30 rollout）。统计用 Welch t 检验 + Compact Letter Display（5% FWER）+ 贝叶斯不确定性。

### 3.1 各模态"是否有效 + 最佳阶段策略"总览

| 模态 | 是否有效 | 最佳策略 | 主要收益维度 |
|---|---|---|---|
| 标准 VL 数据（RoboPoint/RefSpatial） | 有效 | Two-phase full | DS / 未见任务 / 语言跟随（尤其 unseen objects） |
| 机器人轨迹语言标注（脚本 / VLM） | 有效，VLM 优于脚本 | Two-phase 1st-phase-only | DS / 未见任务 / 语言跟随 |
| 跨本体机器人数据（OXE-Ramen） | 有效 | Two-phase 1st-phase-only | DS / 未见任务 / 语言跟随 |
| 人类视频—VLM 生成标注 | 有效 | Two-phase full | DS / 未见任务 / unseen objects |
| 人类视频—latent action | 弱（仅低机器人数据量下有益） | Two-phase 1st-phase-only | 未见任务（疑为算力增益而非知识迁移） |
| FAST 离散 token | 无效 / 有害 | — | 降低未见任务泛化 |
| VQ-VAE 离散 token | 边际 | — | 未见任务微增，DS 略降 |

关键定性结论：**几乎所有真正有用的模态本质上都是"多样化 VL 数据"**（标准 VL、机器人轨迹的 VLM 标注、人类视频的 VLM 标注）——强化 backbone 的视觉-语言理解即转化为更好的策略泛化；跨本体机器人数据最好只放在一阶段做通用表征、二阶段专注目标本体；离散动作 token 会把 backbone 偏向精确动作映射、损害可泛化特征。所有模态对**分布内（seen）性能均无统计显著影响**。

### 3.2 有效模态叠加与关键数字（Fig 10/11）

逐步累加有效数据源（Baseline → +VL → +机器人标注 → +人类视频标注 → +跨本体 = Final Model），各维度**累积增益一致**：

| 指标（Final Model vs. Baseline） | Baseline | Final Model | 提升 |
|---|---|---|---|
| 仿真未见任务成功率 | ~36.2% | **72.6%** | **+36.4 pt** |
| 真机语言跟随平均完成率 | ~24.1% | **69.4%** | **+45.3 pt** |

微调到未见长程灵巧任务（各仅 200 演示）：

| 策略 | 平均任务完成率 |
|---|---|
| Single Task（从零单任务） | ~47.3% |
| FT Baseline（仅机器人数据预训练后微调） | ~67.4% |
| **FT Final Model（有效模态 co-training 后微调）** | **90.2%（+22.8 pt vs FT Baseline，+42.9 pt vs Single Task）** |

（注：以上 Baseline 绝对值由论文给出的 Final Model 值与"提升幅度"反推，均用经验均值。）

### 3.3 backbone 视觉-语言能力（Fig 12）与 CoT（Fig 13）

- **仅用机器人数据训练会严重侵蚀 VLM 的视觉-语言理解**（在 MMBench/MME/SeedBench/RealWorldQA/GQA/SpatialEval/LEGO 上全面退化，几乎"不会说话"）；引入有效 co-training 模态可修复这些能力，叠加后逼近甚至匹配指令微调版 PaliGemma2-Mix——印证"策略泛化与 backbone 表征质量绑定"。
- **显式 CoT 条件化在其操作基准上全面失效**：相较隐式两阶段 co-training 无提升，用 VLM 标注 / latent action 作 CoT 源时反而下降——因为生成 CoT 中的误差会直接传播、放大到动作预测。作者归因于其任务目标清晰、观测到动作映射较直接，隐式推理已足够。

## 四、局限性

1. **未按任务类型细分 VL 数据**：VQA / 图像描述 / 目标检测 / 空间推理各自对哪种策略能力有贡献，未做系统拆解，难以据此做更精准、样本高效的数据配比。
2. **人类视频只用了粗粒度表征**（latent action 与语言标注）；随手部姿态估计成熟，显式抽取细粒度灵巧动作或成更强信号。
3. **CoT 探索仅限 co-training 数据自然产生的低层动作抽象**，缺少高层规划 / 复杂推理形式（历史、反思、层次链）。
4. **仅覆盖模仿学习**；world modeling / 强化学习范式下的 co-training 尚属开放前沿。
5. 结果对 backbone（PaliGemma2）与目标数据集（TRI-Ramen，双臂 Franka）有一定绑定，跨 backbone / 跨本体的可迁移性待验证；FAST"有害"结论也被作者自己限定为"当前数据规模下"，更大规模时或翻转。

## 五、评价与展望

**优点**：(1) 这是一份罕见地"把话说全"的对照研究——统一架构、5 模态 × 3 阶段策略全网格、89 策略 / 6 万级 rollout、并配 Welch t + CLD + 贝叶斯的严格显著性检验，把此前散落在 π0 [6]、ECoT [92]、GR00T、Univla [10] 等工作里的零散经验整合成一张可操作的经验地图，工程指导价值很高。(2) 结论清晰且反直觉：**离散动作 token（FAST/VQ-VAE）与显式 CoT 在清晰目标的操作任务上不划算**，这对当前一窝蜂做"动作离散化 + 推理链"的路线是一次有价值的降温；同时给出"跨本体只放一阶段、VL 数据全程混入"等可直接照搬的配方。(3) "仅机器人数据会侵蚀 backbone、co-training 可修复"的 backbone benchmark 证据，为"策略泛化 ∝ backbone 视觉-语言能力"提供了直接支撑，并巧妙呼应 Good Regulator Theorem。

**缺点 / 开放问题**：(a) 绝对数字多以 bar/violin 图呈现，正文可直接引用的标量偏少（本笔记多处 Baseline 绝对值须由"提升幅度"反推），复现与横向对比略吃力。(b) 结论强绑定 PaliGemma2 + TRI-Ramen 单一设置，"FAST 有害""latent action 只是算力增益"等判断是否随 backbone 规模、数据规模而变，作者已自我设限但未给出规模化曲线（Fig 8 仅在机器人数据量维度上验证了 latent action 的边际递减）。(c) 与并行工作的关系值得追问：GR00T-N1 [5]、π0.5 [33]、Univla [10]、Villa-X [13] 等都主张 latent action / 离散 token 有用，本文在其设置下给出相反结论，差异究竟来自"低数据 vs 高数据"区制还是评测任务性质，是后续最该厘清的问题。可能的改进方向：把 VL 数据按任务类型做因果化配比、引入更高层的规划式 CoT、以及将该协同训练框架迁移到 world-model / RL 后训练范式下重测这些结论的稳健性。

## 参考

1. Barreiros et al. *A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*. arXiv:2507.05331, 2025.（TRI-Ramen / 基础 LBM 与仿真基准来源）
2. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.（flow matching VLA 架构参照）
3. Zawalski et al. *Robotic Control via Embodied Chain-of-Thought Reasoning (ECoT)*. arXiv:2407.08693, 2024.（脚本化机器人轨迹标注与 CoT 来源）
4. Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747, 2025.（离散动作 token 对照）
5. O'Neill et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. ICRA, 2024.（OXE-Ramen 跨本体数据来源）
