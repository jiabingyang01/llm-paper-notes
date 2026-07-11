# DexFlyWheel：面向灵巧操作的可扩展自增强数据生成框架

> **论文**：*DexFlyWheel: A Scalable and Self-improving Data Generation Framework for Dexterous Manipulation*
>
> **作者**：Kefei Zhu, Fengshuo Bai, YuanHao Xiang, Yishuai Cai, Xinglin Chen, Ruochong Li, Xingtao Wang, Hao Dong, Yaodong Yang, Xiaopeng Fan, Yuanpei Chen（et al.）
>
> **机构**：Harbin Institute of Technology；PsiBot；Peking University；PKU-Psibot Lab；Harbin Institute of Technology Suzhou Research Institute
>
> **发布时间**：2025 年 09 月（arXiv 2509.23829）
>
> **发表状态**：NeurIPS 2025（39th Conference on Neural Information Processing Systems）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.23829) | [PDF](https://arxiv.org/pdf/2509.23829)
>
> **分类标签**：`灵巧操作` `数据合成` `残差强化学习` `自增强数据飞轮`

---

## 一句话总结

DexFlyWheel 把 **模仿学习（diffusion policy 基座）+ 残差 RL（SAC）+ 策略 rollout + MimicGen 式增强** 串成一个闭环"数据飞轮":每个任务只给 **一条 VR 遥操作示范**,经三轮迭代自增强出 **2040 条覆盖 20 类物体、500+ 场景** 的灵巧操作数据,训练策略在困难测试集上平均成功率 **81.9%**,并通过数字孪生零样本迁移到真机(双臂 Lift 78.3%)。

## 一、问题与动机

灵巧操作(多指手)因高自由度和复杂接触,对数据的规模、多样性、质量要求远高于夹爪抓取,而数据采集是瓶颈:

- **人类遥操作**:需大量人力、局限于实验室场景;可穿戴动捕虽能野采但仍需人参与且存在 cross-embodiment gap。
- **纯仿真生成**:运动规划/启发式方法难以应对高维接触;LLM 驱动方法只能给高层指令,无法提供指级精细控制;纯 RL 探索困难、易产生非拟人行为、sim-to-real 差。
- **replay/编辑类方法(如 DexMimicGen)**:只对录制轨迹做空间变换,存在两大硬伤——(1) **无法探索新策略**,被局限在原示范行为范围;(2) **数据多样性不足**,物体几何/环境变化受限。当物体几何显著改变(如球→立方体),replay 无法自适应调整手指轨迹。

作者的关键观察:**操作不同物体通常只在轨迹上引起"微小改动"(minor adjustment)**。这意味着人类示范不只是"回放素材",更是能引导探索的**强行为先验**——只要在此先验上做**指级精细微调**即可泛化到新物体。由此提出 IL + 残差 RL 的组合。

## 二、核心方法

将每个操作任务建模为 MDP $\mathcal{M} = \langle S, A, \pi, \mathcal{T}, R, \gamma, \rho, G \rangle$,策略 $\pi$ 依据当前状态 $s_t$ 生成动作分布 $a_t$,目标是最大化未来物体状态序列 $(s_{t+1}, \dots, s_{t+T})$ 的似然。框架分两阶段。

### (1) Warm-up 阶段:单示范 → 初始数据集 $\mathcal{D}_1$

- **VR 遥操作采集**:用 Apple Vision Pro 在仿真中追踪人手/腕/头姿态,每个任务仅采 **一条** 种子示范 $d_{\text{seed}}$。
- **数据增强模块 $\mathcal{A}_{\text{EP}}$**:在 MimicGen 基础上扩展,支持跨环境(E)、空间(P)的多维增强并叠加仿真域随机化,即 $\mathcal{D}_1 = \mathcal{A}_{\text{EP}}(d_{\text{seed}}; \mathcal{C}_1)$。

> 用大白话说:先花极少人力录一条"标准动作",再用几何变换 + 换背景/换位置把它扩成一小批,给飞轮一个能起步的火种。

### (2) 自增强数据飞轮阶段:闭环四步(迭代 $i \in \{1,\dots,n-1\}$)

**Step 1 — Base Policy(IL)**:用 diffusion policy 在 $\mathcal{D}_i$ 上训基座 $\pi_{\text{base}}$。输入 $s_t = \{s_t^{\text{vis}}, s_t^{\text{obj}}, s_t^{\text{prop}}\}$(视觉图 + 物体 6D 位姿/速度 + 机器人本体感知),输出动作序列 $(a_t, \dots, a_{t+H})$,预测步长 $H=8$,每个动作 = 末端 6D 位姿 + 目标关节角(灵巧手 $n=10$/单臂 $n=7$)。

**Step 2 — Residual Policy(RL)**:冻结 $\pi_{\text{base}}$,用 SAC 训一个只看物体状态 $s_t^{\text{obj}}$ 和本体感知 $s_t^{\text{prop}}$ 的残差策略 $\pi_{\text{res}}$,输出修正量 $\Delta a = (\Delta a_t, \dots, \Delta a_{t+H})$,按尺度 $\alpha$ 叠加:

$$\tilde{a}_t = a_t + \alpha \cdot \Delta a_t, \qquad \pi_{\text{combined}}^{i} = \pi_{\text{base}}^{i} + \alpha \cdot \pi_{\text{res}}^{i}$$

训练时采用 policy-decorator 的渐进探索调度:

$$\pi_{\text{combined}}(s) = \begin{cases} \pi_{\text{base}}(s) + \alpha \cdot \pi_{\text{res}}(s) & \text{w.p. } \epsilon \\ \pi_{\text{base}}(s) & \text{w.p. } 1-\epsilon \end{cases}$$

其中混合系数 $\epsilon$ 在 $T$ 步内从 0 线性升到 1,控制权从基座逐渐移交给残差。

> 用大白话说:基座已经会做"八九不离十"的动作,残差策略只需在此之上学"临门一脚"的指级微调。$\alpha=0.1$ 把修正量压得很小,保证既不推翻基座技能、又能适配新物体;一开始几乎全听基座、后期才放开残差,避免早期乱探索把训练搞崩。

**Step 3 — Rollout 采集**:用冻结的 $\pi_{\text{combined}}^{i}$ 在随机化物体配置下 rollout,按任务成功过滤,收 $K$ 条高质量轨迹得 $\mathcal{D}_O^i$。这种"策略在环"采集能对物体做鲁棒泛化,而几何无关的轨迹编辑做不到。

**Step 4 — 数据增强**:再用 $\mathcal{A}_{\text{EP}}$ 对 $\mathcal{D}_O^i$ 做环境/空间增强,得 $\mathcal{D}_{i+1} = \mathcal{A}_{\text{EP}}(\mathcal{D}_O^i; \mathcal{C}_{\text{aug}}^{i+1})$,喂入下一轮训更强的基座。如此循环,**数据多样性与策略能力同步螺旋上升**,即"飞轮效应"。

**Reward 设计(人工设计,分任务)**。以抓取为例,鼓励精确指位 + 成功抬升:

$$r_{\text{grasp}} = \exp\!\left(-5 \cdot \max\!\Big(\textstyle\sum_i d_i - 0.05,\ 0\Big)\right) + 100 \cdot \max\big(0.2 - |z_{\text{target}} - z_{\text{current}}|,\ -0.01\big)$$

其中 $d_i$ 为拇指/食指/中指到物体中心距离,$z$ 为高度项。Pour/Lift/Handover 另设含 grasp 距离、抬升、倾角惩罚、双手协同(sync)等分项的复合奖励。

**"微小改动"的量化**。作者用残差范数比 **RNR** 验证观察:

$$\text{RNR} = \frac{\|a_t^{\text{res}}\|}{\|a_t^{\text{base}}\| + \epsilon}$$

RNR 越低说明适配新物体只需微调而非大改,佐证了"IL 先验 + 小残差"的合理性;并配合 JointDiff(名义与自适应轨迹的平均关节角差)与 curriculum(几何相似物体→复杂几何逐步引入)的物体课程学习。

## 三、实验结果

**设置**:仿真平台 OmniGibson,80 类物体、12 种环境;四任务——Grasp/Pour 单臂(Franka Panda + Inspire 手)、Lift/Handover 双臂(7-DoF Real-Man RM75-6F 臂 + 6-DoF PsiBot G0-R 手,头部 RealSense D435 第一视角)。迭代 $i=\{1,2,3\}$ 分别生成 20/100/500 条/任务。基座 diffusion policy 训练用 8×A100,残差 SAC 训练用单张 RTX 4090。

**飞轮效应(Table 1,四任务均值)**:数据多样性与泛化成功率随迭代同步暴涨。

| 迭代 | 物体数 O | 环境 E | 位姿 P | 场景 Configs | 轨迹 Traj | $\pi_{\text{combined}}$ 在 $T_{OEP}$ 成功率 |
|---|---|---|---|---|---|---|
| $i=1$ | 1.0 | 3.0 | 2.5 | 9.5 | 20 | 16.5% |
| $i=2$ | 6.8 | 6.8 | 5.5 | 322.0 | 100 | 43.9% |
| $i=3$ | 20.0 | 12.0 | 8.8 | 2040.0 | 500 | **81.9%** |
| $1\to3$ 提升 | 20.0× | 4.0× | 3.5× | 214.7× | 25.0× | +396.4% |

残差策略($\pi_{\text{base}} \to \pi_{\text{combined}}$)在物体测试集 $T_O(i)$ 上平均把物体泛化提升 **32.1%**。

**对比 baseline(Table 2,多因子泛化测试集 $T_{OEP}$ 成功率 %)**:

| 方法 | Grasp | Pour | Lift | Handover | 平均 |
|---|---|---|---|---|---|
| Human Demo (Default) | 6.1 | 16.7 | 13.9 | 0.8 | 9.4 |
| Human Demo (Enhanced) | 15.0 | 36.1 | 2.5 | 0 | 13.4 |
| DexMimicGen (Default) | 30.3 | 38.9 | 28.2 | 28.3 | 31.4 |
| DexMimicGen (Enhanced,10× 数据优势) | 50.3 | 44.4 | 43.7 | 42.5 | 45.2 |
| **DexFlyWheel(Ours,单示范)** | **90.0** | **85.8** | **79.4** | **72.5** | **81.9** |

只用 1 条示范就大幅超越用 20 条示范的 Human Demo(81.9% vs 13.4%),也稳超 replay 类 DexMimicGen。

**数据生成成功率 / 时间(Table 3–4)**:DexFlyWheel 生成成功率均值 **89.8%**(DexMimicGen 63.0%),在动态性强的 Handover 上优势最大(85.7% vs 14.8%)。采集 500 条成功轨迹仅需 **2.4 小时**,比 DexMimicGen(4.4h)快 1.83×,比人类遥操作(12.5h)快 5.21×;单条轨迹 15s(与 DexMimicGen 相同,远快于遥操 60s)。

**消融(Fig 4/5)**:去掉残差策略(w/o Res)成功率下降最剧烈,是泛化的核心;成功处理的物体数从 8.25(w/o Res)升至 **20**(完整版)。

**扩展迭代(Table 7)**:成功率随迭代继续升但边际递减,$i=3$ 是性价比拐点。

| 迭代 | Grasp SR (%) | Lift SR (%) |
|---|---|---|
| $i=1$ | 15.0 | 13.9 |
| $i=2$ | 58.0 (+43.0) | 44.4 (+30.5) |
| $i=3$ | 90.0 (+32.0) | 79.4 (+35.0) |
| $i=4$ | 92.5 (+2.5) | 82.1 (+2.7) |
| $i=5$ | 93.2 (+0.7) | 83.5 (+1.4) |

**真机部署(数字孪生)**:用 FoundationPose 获取真实物体位姿、RealSense D455 采图,双臂真机 Lift 成功率 **78.3%**、Handover **63.3%**(每组 20 条,3 组)。

## 四、局限性

- **奖励需人工设计**:残差 RL 依赖分任务手写 reward(论文给了 grasp/pour/lift/handover 四套),新任务需重新设计,规模化受限;作者提出未来用 LLM 驱动奖励生成。
- **无触觉反馈**:策略与仿真均缺触觉信号,富接触任务受限,依赖视觉 + 本体感知。
- **仿真基座依赖**:强依赖 OmniGibson 高保真渲染与准确物理/物体资产;sim-to-real 靠数字孪生对齐,对真机场景外的分布仍未系统评估。
- **物体课程需人工组织**:curriculum(几何相似→复杂)与增强场景配置仍含人工先验,并非全自动。
- **迭代边际递减**:$i>3$ 后收益快速衰减,飞轮上限受基座架构与奖励塑形制约。

## 五、评价与展望

**优点**:(1) 把 LLM 后训练的"self-improving 飞轮"迁到机器人数据生成,思路清晰、闭环自洽——**用策略采集替代轨迹回放**,从根上解决 replay 类方法"不能探索新策略"的痛点。(2) **IL 提供拟人行为先验 + 残差 RL 只学指级微调** 的分工非常契合"minor adjustment"观察,$\alpha=0.1$ 的小残差既保稳定又实现物体泛化,且 RNR/JointDiff 给出了量化证据而非只讲故事。(3) 数据效率突出:单示范起步、真机零样本迁移可用,工程落地价值高。

**与公开工作的关系**:相较 MimicGen/DexMimicGen(纯轨迹编辑,几何无关、Handover 类动态任务崩溃)和纯 RL 生成(探索难、非拟人、sim2real 差),本文正是取两者之长;残差 RL 叠加冻结基座的做法与 policy-decorator(在线精化大策略)、residual policy learning 一脉相承,数据飞轮框架则显式借鉴了 Arena Learning、STAR 等 LLM 自举思想。可视为 "DemoGen/Demonstration 合成 + 残差 RL 泛化" 在灵巧手场景的系统性整合。

**开放问题与可能改进**:① **奖励自动化** 是最大瓶颈,若能用 VLM/LLM 从语言目标自动生成 dense reward 或用成功判据做偏好式 RL,可解锁任务规模化;② 触觉/力信息缺失使富接触(插拔、拧、In-hand 重定向)难覆盖,引入触觉仿真是自然延伸;③ 飞轮"收敛上限"由基座架构决定,可探索更强的 VLA 基座或把残差 RL 换成 offline/ preference-based RL 以减少在线交互;④ 真机评估仅两任务、样本量小(20×3),泛化到杂乱真实场景、跨本体迁移仍需更充分验证;⑤ curriculum 与场景采样的自动化(如按当前策略弱点主动选物体/位姿)有望进一步提升飞轮效率。总体是一篇**方法完整、数字扎实、落地导向**的数据合成工作,对"以极少人类示范撬动大规模灵巧数据"给出了有说服力的范式。

## 参考

1. Jiang et al. *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning*, arXiv:2410.24185, 2024.(最直接对比的 replay/编辑 baseline)
2. Mandlekar et al. *MimicGen: A Data Generation System for Scalable Robot Learning Using Human Demonstrations*, arXiv:2310.17596, 2023.(增强模块 $\mathcal{A}_{\text{EP}}$ 的基础)
3. Yuan et al. *Policy Decorator: Model-agnostic Online Refinement for Large Policy Model*, arXiv:2412.13630, 2024.(渐进残差探索调度来源)
4. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, IJRR 2023.(base policy 架构)
5. Luo et al. *Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena*, arXiv:2407.10627, 2024.(数据飞轮思想来源)
