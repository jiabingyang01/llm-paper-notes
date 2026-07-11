# VISTA：面向分层操作策略的可扩展世界模型

> **论文**：*Scaling World Model for Hierarchical Manipulation Policies*
>
> **作者**：Long Qian、Yueze Wang、Jiaxi Song 等（三人共同一作；通讯作者 Xuguang Lan、Huaping Liu；Project Lead Xinghang Li）
>
> **机构**：西安交通大学、北京人工智能研究院（Beijing Academy of Artificial Intelligence）、清华大学、新加坡国立大学、中国科学院自动化研究所
>
> **发布时间**：2026 年 02 月（arXiv 2602.10983）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.10983) | [PDF](https://arxiv.org/pdf/2602.10983)
>
> **分类标签**：`世界模型` `分层VLA` `视觉子目标生成` `Goal-conditioned策略` `Flow Matching`

---

## 一句话总结

VISTA 用一个 34.1B 参数、在 EMU3.5 基础上继续训练的**生成式世界模型**把长程操作指令自回归分解成"文本子任务+视觉目标图"交替序列,再用一个 goal-image 条件化的 flow-matching VLA（GoalVLA）执行,仅用 2 小时真机数据训练,在涉及 21 个未见物体的分布外（OOD）场景中把成功率从 π0 基线的 14% 提升到 69%。

## 一、问题与动机

VLA 模型（RT-2、π0、GR00T N1 等）依赖大规模真机数据做端到端模仿学习,但在分布外场景下很脆弱,而真机数据采集成本高。作者指出根源在于 VLM 与 VLA 的数据结构错配：VLM 在成对图文数据上建模离散语言分布,VLA 却要在连续动作回归目标下学习成百上千步的长轨迹,导致 VLA 缺乏 VLM 那样的零样本泛化能力。

分层任务分解是常见的缓解思路,但存在表征取舍的两难：

- 纯语言子目标（如 Helix、G0）语义上泛化好,但缺乏具体的空间/物理约束;
- 稠密视频预测（world model 做逐帧生成）虽然细节丰富,但长时域下容易时序漂移、物理不一致。

由此提出核心问题："如何利用基础模型的泛化能力,把操作任务抽象为一种既能提升鲁棒性又能提升数据效率的中间表征?" VISTA 的答案是：**只预测稀疏的关键帧（视觉子目标）而非稠密视频**,并将其与文本子任务交织成统一的离散序列,由世界模型自回归生成,再交给低层 goal-conditioned VLA 执行。

## 二、核心方法

### 2.1 问题形式化

给定全局指令 $L\in\mathcal T$ 和当前观测 $I_t$,系统不直接把 $L$ 映射到动作,而是先分解为一串中间里程碑：第 $i$ 阶段包含文本子任务 $l_i\in\mathcal T$（做什么）和视觉目标 $g_i\in\mathcal I$（怎么做）。世界模型 $W$ 根据指令 $L$ 与历史 $\mathbf h=(l_0,g_0,\dots,l_{i-1},g_{i-1})$ 预测下一个里程碑,低层策略 $\pi_\theta$ 再根据当前观测和当前里程碑推理动作序列：

$$(s_i, g_i) = W(L, \mathbf h), \qquad \mathbf a = \pi_\theta(I_t, l_i, g_i)$$

大白话：世界模型是"导演",不断喊出下一步该做什么、长什么样;GoalVLA 是"演员",只管把当前这一步的画面演出来。机器人执行动作直到当前观测在视觉上与 $g_i$ 对齐,系统才更新历史并向 $W$ 索要下一阶段目标。

### 2.2 世界模型：统一离散序列的自回归规划器

**Tokenization。**图像用 EMU3.5 自带的 IBQ-Tokenizer 量化（词表 131,072,每 16×16 patch 一个 token,512×512 输入对应 1024 个视觉 token/图）,文本用 Qwen3 tokenizer（词表 151,854）,二者共享一个统一词表 $\mathcal V$（合计 282,926 tokens）。多视角图像按固定顺序展平,整段序列写作：

$$S=(\phi(I_0),\phi(L),\phi(l_0),\phi(g_0),\dots,\phi(l_N),\phi(g_N))=(u_1,u_2,\dots,u_K)$$

大白话：把"初始图 + 全局指令 + 一串（子任务文本, 目标图）"全部拍扁成一条 token 序列,文本和图像在同一个 Transformer 里被当作"同一种语言"来建模。

**训练目标。**标准自回归交叉熵损失,teacher forcing + 因果掩码：

$$\mathcal L=-\sum_{k=1}^{K}\log P(u_k\mid u_{<k};\theta_{\mathcal W})$$

**推理（子任务规划）。**用 beam search（宽度 $B$）迭代采样候选序列,联合概率为各 token 条件概率之积：

$$P(S)=\prod_j P(u_j\mid u_{<j})$$

采样到终止符后,取整体概率最高的序列,用逆 tokenizer $\phi^{-1}$ 在像素级重建目标图 $g_i$。大白话：不是贪心逐帧生成,而是像做机器翻译一样对整条"计划"做全局搜索,避免早期一步选错导致后面越走越偏。

世界模型在开源 EMU3.5 checkpoint 上做 2000 步继续训练,数据为交织的导航+操作图文数据。

### 2.3 GoalVLA：目标图条件化的低层策略

架构类似 π0 的 MoE 结构：PaliGemma-3B 作为 VLM 主干 + 0.3B 规模的动作专家,采用分块因果掩码——VLM block 只看自身特征,本体感知 block（与动作 block 共享权重）看自身+VLM 特征,动作 block 看全部 block,各 block 内部双向注意力。每步输入 6 张图（3 张当前观测 + 3 张目标图,经 SigLIP 编码）、子任务文本 prompt、末端 6D 位姿本体感知信号。

采用 flow matching 目标学习连续动作轨迹。定义噪声样本 $z\sim\mathcal N(0,\mathbf I)$ 与真实动作块 $a$ 之间的线性插值：

$$\mathbf x_\tau=(1-\tau)z+\tau a,\quad \tau\in[0,1]$$

预测对应速度场 $v_\tau=a-z$,以 $(l_i,I_t,g_i,\tau)$ 为条件,目标是最小化均方误差：

$$\mathcal L_{FM}=\mathbb E_{\tau,z}\Big[\big|\pi_\theta(\mathbf x_\tau,l_i,I_t,g_i,\tau)-v_\tau\big|^2\Big]$$

大白话：不直接回归动作,而是学一个"如何把随机噪声一步步拉向真实动作"的速度场,推理时做 10 步 flow matching 去噪,生成长度 30 的动作块。

**两个关键工程细节：**

1. **Subtask-Aware Action Padding**：动作块可能跨越子任务边界（包含属于下一阶段的步骤）。训练时把每条真值动作块中里程碑完成之后的部分置零,显式教会策略"目标达成就停",避免提前执行下一阶段动作。
2. **Random Goal Image Offset**：世界模型预测的目标图相对真实子任务终止点可能有几帧的提前/滞后偏差。为此在阶段边界附近定义一个时间重叠窗口,训练时对窗口内样本随机使用 $g_i$ 或 $g_{i+1}$ 作为目标条件（并相应重标注阶段/是否置零）,让策略对边界噪声更鲁棒。

推理时用闭环绝对末端位姿控制,每次只执行预测的 30 步动作块中的前 10 步,并取第 5、10 步的绝对末端位姿作为目标路点以获得更平滑的控制。

### 2.4 数据构建

**自动里程碑标注流水线（三步）：**（1）用 Qwen3 对指令动词聚类,构建约 50 个原子技能库;（2）在运动轨迹上用 Ramer–Douglas–Peucker（RDP）算法结合夹爪状态切换检测候选里程碑边界（基于物理状态变化,而非固定时间步长）;（3）用 Qwen2.5-VL 72B 合并技能相同的相邻片段并生成自然语言子任务描述。

该流水线把来自 Open X-Embodiment、AgiBot World、Mobile Aloha 的 120 万条轨迹转换为交织的子任务/目标图序列,覆盖 14 种本体、支持多视角,总计 15.2B token。同时构建了一个 15.2B token 的 Any-to-Image（X2I）协同训练数据集（整合 SEED-Data-Edit、WeatherStream、ShareGPT-4o-Image 等开源数据,并用 MASt3R 标注多图相对位姿构建 camera-view edit 数据）,用于强化世界模型的图像生成质量、指令跟随与多视角一致性。

**规模：**世界模型共 34.1B 参数（Transformer 层 31.2B + embedding 层 2.9B）,最大序列长度 16,384,用 Megatron-LM（tensor-parallel=8）在 128×H100 上训练 2 天。GoalVLA 分两阶段微调：阶段一在 AgiBot-Beta 的 20 万条轨迹上训 10 万步（batch 512,lr 5e-5）;阶段二在 737 条自采 Aloha 轨迹上训约 2 万步（10 epoch,batch 128,lr 2e-5）。

## 三、实验结果

真机任务：5 个 pick-place 任务（cola can、egg、bread/croissant、apple、milk）,仅用 2 小时数据采集（5 任务 × 约150条 = 737 条轨迹）。基线为 π0（语言指令引导）及其变体 π0-subtask（把训练数据中的原始指令替换为分解后的子任务文本）。

**Table I：5 个训练任务上的 Basic / Unseen Distractor / Unseen Target 三种域内设置（15 个场景,均值列）**

| 设置 | 方法 | 平均 Approach | 平均 Success |
|---|---|---|---|
| Basic | π0 | 1.00 | **0.96** |
| Basic | π0-subtask | 1.00 | 0.91 |
| Basic | Ours (VISTA) | 1.00 | 0.93 |
| Unseen Distractor | π0 | 1.00 | 0.73 |
| Unseen Distractor | π0-subtask | 1.00 | 0.78 |
| Unseen Distractor | Ours (VISTA) | 1.00 | **0.82** |
| Unseen Target | π0 | 0.40 | 0.04 |
| Unseen Target | π0-subtask | 0.73 | 0.31 |
| Unseen Target | Ours (VISTA) | **1.00** | **0.67** |

可见：在完全域内（Basic）设置下 π0 略优于 VISTA（0.96 vs 0.93）,但只要引入未见干扰物或未见目标物,VISTA 的优势迅速拉开——尤其 Unseen Target 设置下 π0 的 approach 成功率仅 40%（把物体错认成视觉相似的训练物体）,而 VISTA 达到 100% approach、67% 执行成功,证明目标图提供的显式空间线索是关键。

**更严苛的 OOD 泛化（论文摘要与 Fig.1 headline 数字）：**在涉及 21 个未见物体、3 种桌布×布局组合共 63 个新场景的评测中,VISTA 综合成功率达到 **69%**,而语言指令引导的 π0 基线仅 **14%**——这是论文反复强调的核心对比数字。论文还专门做了极端干扰实验（图案桌布 × box/bottle/fruit/bread 等物体类别）,发现语义丰富的背景会让所有方法性能下降,但 VISTA 的降幅明显最小。

**定性结果：**VISTA 生成的目标图能保持跨视角（头部相机、双手腕相机）的三维空间一致性,能在从未见过跨臂协作的训练数据下生成双臂组合任务的合理序列,能做未训练过的组合指令（如"把 cola 和 milk 放盘子")、空间理解指令（"把左边的香蕉放盘子"）、语义理解指令（"把馒头放在有女人的图片上"）,并展现出跨本体迁移能力（同一 prompt 可为 Aloha、AgiBot G1、WidowX 生成对应本体的目标序列）。

## 四、局限性

论文第七节及附录 G 明确指出三类局限：

1. **任务范围有限**：真机评测目前局限于 pick-and-place,虽然定性展示了叠衣服、装箱等更复杂长程任务的目标图生成,但缺乏对应的定量执行评测,液体倾倒、可变形物体折叠等仍停留在展示阶段。
2. **目标图时序/空间误差传导**：生成的目标图相对真实子任务终止点可能有轻微提前/滞后（时序误差）,GoalVLA 会过度拟合去匹配目标图中的机械臂姿态而非适应真实执行状态,导致下压不足、抓取失败;此外目标图（尤其手腕相机视角）可能存在空间偏移,导致夹爪与物体碰撞。
3. **训练分布外的目标位置泛化差**：即便目标图生成准确,若指定的目标位置显著超出训练数据的空间分布,GoalVLA 仍难以准确跟随,作者认为需要提升训练数据的空间多样性来缓解。此外,世界模型继承自 EMU3.5 的指令遵循能力在机器人数据微调后有所衰减,仍会出现幻觉。

## 五、评价与展望

**优点：**VISTA 的核心贡献在于把"用世界模型做任务分解"这件事从稠密视频预测降维成稀疏关键帧+文本的交替序列,这个设计选择本身是有说服力的权衡——相比 Helix/G0 等纯语言子目标方案,它保留了显式空间约束;相比 GR-1、Inverse Dynamics、RoboEnvision 等稠密视频预测世界模型,它避开了长时域视频生成的漂移和高昂算力开销。用统一离散词表把图像 token 和文本 token 放进同一个自回归 Transformer,并直接复用 EMU3.5 这样的通用图文生成大模型做继续训练,是"借力大规模通用生成模型的世界知识做具身泛化"这一思路的又一次扎实验证,和同期 GR00T N1、π0.5 等强调基础模型能力迁移的工作方向一致。2 小时真机数据能在 21 个未见物体上把成功率从 14% 拉到 69%,是相当有说服力的数据效率证据。

**局限与开放问题：**其一,高层世界模型与低层 GoalVLA 是分阶段训练、松耦合的两个模块,目标图的时序/空间误差会直接传导到执行层,论文自己也承认这是当前最大的失败来源,如何让低层策略对目标图误差更鲁棒（例如引入不确定性估计、目标图与当前观测的置信度融合)是一个尚未解决的方向。其二,34.1B 参数的世界模型 + beam search 解码在实际部署中的推理延迟未见报告,分层框架天然引入了"世界模型生成一次目标图,执行多步"的异步调度问题,论文没有给出端到端推理频率或延迟数据,这对真实高频闭环控制场景是需要补充的信息。其三,评测任务仍以桌面 pick-place 为主,尚未验证在双臂协作、可变形物体、力控敏感任务等更复杂操作上的定量表现,而这些恰恰是稀疏关键帧表征最可能遇到麻烦的场景（例如液体倾倒这种连续物理过程很难用几张关键帧描述)。其四,与近期同样探索"视觉子目标/goal-image 条件化策略"的工作（如 VLM-TDP、LEAP）相比,VISTA 的优势是把子目标生成也做成了可大规模预训练、可零样本泛化的生成模型而非轨迹级 latent,但也因此更依赖底层图像生成模型（EMU3.5）的通用生成质量,存在対上游生成模型能力的强依赖风险。整体而言,VISTA 提供了一个数据效率优异、思路清晰的分层具身智能范式,后续工作值得在闭环误差修正、长程复杂任务扩展和推理效率方面继续推进。

## 参考

1. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* RSS 2025.
2. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734, 2025.
3. Cui et al. *Emu3.5: Native Multimodal Models are World Learners.* arXiv:2510.26583, 2025.
4. Du et al. *Learning Universal Policies via Text-Guided Video Generation.* NeurIPS 2023.（Inverse Dynamics 世界模型路线）
5. Bu et al. *AgiBot World Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems.* arXiv:2503.06669, 2025.
