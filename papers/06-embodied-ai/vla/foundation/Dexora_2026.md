# Dexora：面向高自由度双臂灵巧操作的开源VLA

> **论文**：*Dexora: Open-source VLA for High-DoF Bimanual Dexterity*
>
> **作者**：Zongzheng Zhang, Jingrui Pang（共同一作）, Zhuo Yang, Kun Li, Minwen Liao, Saining Zhang, Guoxuan Chi, Jinbang Guo, Huan-ang Gao, Modi Shi, Dongyun Ge, Yao Mu, Jiayuan Gu, Rui Chen, Hao Dong, Huazhe Xu, Li Yi, Yixin Zhu, Hang Zhao, Pengwei Wang, Shanghang Zhang, Guocai Yao, Jianyu Chen, Hongyang Li, Hao Zhao（通讯作者）et al.
>
> **机构**：清华大学（Tsinghua University）、北京智源人工智能研究院（Beijing Academy of Artificial Intelligence）、香港大学（The University of Hong Kong）、上海交通大学（Shanghai Jiao Tong University）、上海科技大学（ShanghaiTech University）等
>
> **发布时间**：2026 年 05 月（arXiv 2605.18722）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.18722) | [PDF](https://arxiv.org/pdf/2605.18722)
>
> **分类标签**：`双臂双手VLA` `高自由度灵巧手` `混合遥操作` `判别器数据质量筛选` `跨具身泛化`

---

## 一句话总结

**Dexora** 是首个开源的双臂、双手、36 自由度 VLA 系统：通过"外骨骼背包捕捉手臂大动作 + Apple Vision Pro 无标记手指追踪捕捉精细指法"的混合遥操作接口,同步驱动实体机器人与 MuJoCo 数字孪生采集embodiment-matched 数据（仿真 10 万条轨迹 650 万帧 + 真实 1 万条 292 万帧）,并用一个判别器对真实示教数据做质量打分、加权训练扩散 Transformer 策略;最终在基础任务上平均成功率达 89.6%,灵巧任务上从 51.7%（此前最强开源基线 GR00T N1）提升到 66.7%,且能零样本降维迁移到单臂夹爪、双臂夹爪、单臂低自由度手等更简单具身。

## 一、问题与动机

现有 VLA 系统在"具身覆盖"上存在明显空白（Fig. 2）：要么面向双臂但只配低自由度平行夹爪（π0、π0.5、RDT-1B、GO-1、GR-3、GR00T N1）,要么面向单臂但配高自由度灵巧手（DexGraspVLA、Being-H0、Dexonomy）,尚无系统同时覆盖"双臂 + 高自由度双手"这一象限。论文用三个直观例子说明这一空白的代价：

- **活塞插入**（piston insertion）需要双臂协同,单臂做不了;
- **从密集货架取书**（book retrieval）需要手指精细操作,平行夹爪抓不稳;
- **开瓶盖**（bottle opening）需要 12 自由度手指的拇指-食指协同扭转力矩,6 自由度手做不到。

**Dexora** 的目标就是用一套端到端 VLA 同时覆盖双臂协同与高自由度灵巧操作,并验证这样训练出的策略能否反过来"降维"泛化到更简单的具身。

## 二、核心方法

### 2.1 硬件与混合遥操作

实体平台为两臂 AIRBOT（各 6 自由度）+ 一对 XHand 灵巧手（各 12 自由度、全部指关节独立驱动,拇指与食指还支持侧向外展/内收,可实现拟人化的手内重定向与扭转操作,如拧瓶盖）,整机共 **36 自由度**。

遥操作把"手臂大动作"与"手指精细动作"解耦：

- 定制的双臂外骨骼背包捕捉操作者肩-肘-腕角度,直接映射到机器人关节空间,避免视觉逆运动学重定向常见的抖动与奇异点问题,轨迹无漂移、低延迟;
- Apple Vision Pro 提供无标记 3D 手指骨架追踪,经短暂标定后重定向到 XHand,并施加关节限位与安全约束。

该接口同时驱动真实机器人与一个同构的 MuJoCo 数字孪生,四路 RGB 视角与 36 维关节状态以 20Hz 时间对齐记录,操作者可在真实/仿真之间无缝切换采集（如 "apple→plate" 任务在真实与仿真下用同一接口采集）,从而缩小 sim-to-real 差距。

### 2.2 数据集构建

**仿真数据**：用 Qwen2.5-VL 从 Objaverse 中挖掘可操作物体并自动赋予物理参数,构建 3 大基础任务族共 200 个任务;每任务采集 3-5 条遥操作种子演示,再用 DexMimicGen 的方案重定向到新场景生成 500 条/任务的轨迹,场景布局与成功判据由 Qwen 自动生成。总计约 **10 万条轨迹、650 万帧、361 小时视频**。

**真实数据**：同一具身下采集,除常见物体/基础任务外还加入难以在仿真中还原的灵巧工具使用与铰接物体场景,共 200 个任务、每任务 50 条遥操作演示,得到 **1 万条 episode、40.5 小时、292 万帧**,统一转换为 LIBERO-2.1 格式并开源。

### 2.3 判别器引导的质量感知训练

真实遥操作数据存在操作者水平、遮挡、传感噪声与延迟带来的噪声/不稳定演示,直接训练会损害策略。论文设计了三阶段"数据过滤 → 判别器训练 → 质量感知后训练"流程（Fig. 5）。

**(a) 数据过滤**：把每条 episode 的本体感知状态 $s_t \in \mathbb{R}^{36}$ 用中心差分求速度、加速度、加加速度（jerk）：

$$v_t=\frac{s_{t+1}-s_{t-1}}{2\Delta t},\quad a_t=\frac{v_{t+1}-v_{t-1}}{2\Delta t},\quad j_t=\frac{a_{t+1}-a_{t-1}}{2\Delta t}$$

再做 episode 级 RMS 聚合（$D=36$、$T$为episode长度）：

$$A_{ep}(\tau)=\sqrt{\frac{1}{(T-6)D}\sum_{t=4}^{T-3}\sum_{k=1}^{D} a_{t,k}^2},\qquad J_{ep}(\tau)=\sqrt{\frac{1}{(T-6)D}\sum_{t=4}^{T-3}\sum_{k=1}^{D} j_{t,k}^2}$$

按 $A_{ep}$、$J_{ep}$ 分别排序取最低 20%,取交集得到预筛选集合 $\mathscr{S}_{pre}$（约保留 18% 的 episode）;再对 $\mathscr{S}_{pre}$ 做开环回放,只保留"无碰撞完成任务"的作为高质量正例集 $\mathscr{S}_{high}$（约占真实数据 15%）。

**用大白话说**：先挑动作最平顺、抖动最小的一批演示（运动学层面的"稳"）,再实际回放看它们是否真能无碰撞地把任务做完（结果层面的"对"）,两者交集才算高质量数据——避免了"静止不动、加速度天然很低但毫无意义"的假阳性。

**(b) 判别器训练**：对每条 episode 采样 $K$ 个子片段,每个片段 token 化为 $\xi=(s_t,\mathbf{o}_t,\ell,\mathbf{a}_{t:t+L-1},\widehat{\log\pi}_t)$,其中 $\widehat{\log\pi}_t$ 是用一个冻结的预训练扩散策略算出的"策略兼容性"代理指标,基于去噪残差能量的负值：

$$E_t=\frac{1}{|\mathscr{S}|L}\sum_{s\in\mathscr{S}}\sum_{\tau=t}^{t+L-1}\left\| \varepsilon_\theta(\mathbf{o}_\tau,\ell,\mathbf{a}_{\tau:\tau+L-1},s_\tau)-\varepsilon \right\|^2$$

$$\widehat{\log\pi}_t=-\mathrm{zscore}(E_t)=-\frac{E_t-\mathrm{Mean}(E)}{\sqrt{\mathrm{Var}(E)+\varepsilon}}$$

**用大白话说**：如果预训练策略在某个片段上的去噪误差很小,说明这段动作"很符合模型已学到的规律"，于是给它一个更高的 $\log\pi$ 代理分,作为判别器的额外输入特征。

片段 token 经一个浅层 Transformer（结构与策略网络类似）编码后接 MLP+sigmoid,输出片段质量分 $d(C_k)\in(0,1)$。训练目标是正-无标记（PU）二分类损失,以 $\mathscr{S}_{high}$ 为正例、其余真实数据 $\mathscr{U}=\mathscr{D}_{real}\setminus\mathscr{S}_{high}$ 视为无标记（当负例处理）：

$$\mathscr{L}_D=\eta\,\mathbb{E}_{\tau\in\mathscr{S}_{high}}[-\log d(\tau)]+\mathbb{E}_{\tau\in\mathscr{U}}[-\log(1-d(\tau))]$$

其中 $\eta=0.5$,并将打分校准到 $[0.1,0.9]$ 区间以稳定训练。判别器 12 层、隐藏维 512、8 头、约 3000 万参数,在 8×A100 上以 batch size 64 训练 1 万步。

**(c) 质量感知后训练**：沿用 DWBC（Discriminator-Weighted Behavior Cloning）的映射方式,把校准后的判别器分数转换为样本权重 $w_i$,代入加权扩散损失（辅以短暂权重 warm-up）：

$$\mathscr{L}_\pi=\sum_{l=1}^{L} w_i \left\| \varepsilon_\theta(\cdot)-\varepsilon \right\|^2$$

**用大白话说**：判别器给每段真实演示打一个"像不像高质量数据"的分,分越高在扩散损失里权重越大——用同一批带噪数据训练,但让策略更多向高质量片段学习、少被低质量片段带偏。推理阶段只用策略网络,不再调用判别器。

### 2.4 策略网络

解码器式扩散 Transformer,28 层、隐藏维 1024、16 头（视觉用 SigLIP、语言用 T5 编码,交替注入 Transformer block）。输入当前本体状态 $s_t$（36 维）、多视角图像 $\mathbf{o}_t$、语言指令 $\ell$ 与加噪动作块,预测动作块：

$$\pi_\theta(s_t,\mathbf{o}_t,\ell)=\widehat{\mathbf{a}}_{t:t+L-1}$$

训练用标准 DDPM,推理用 DPMSolver++ 加速,动作块长度 $L=32$。整体流程：先在 10 万条仿真轨迹上预训练 10 万步获得基础操作能力（pick & place、组装等）,再用判别器权重在真实数据集上后训练,把基础能力升级为灵巧双手技能。

## 三、实验结果

对照基线：DP（Diffusion Policy,直接回归 36 维连续动作）、π0（flow-matching VLA,加 2 层 MLP 投影器把原生低自由度输出映射到 36 维关节指令）、GR00T N1（开源人形 VLA,VLM+DiT）。所有方法每任务 100 条演示微调 5 万步,同一控制频率、动作块长度、相机内外参;每任务 20 次 rollout 统计成功率。

**基础任务（12 个任务：5 个 Pick-and-Place、5 个 Assemble/Disassemble、2 个 Articulated Object,含双臂协同任务）：**

| 方法 | Pick-and-Place（5 任务均值范围） | Assemble/Disassemble | Articulated Object | 平均成功率 |
|---|---|---|---|---|
| DP | 25–65 | 10–60 | 20–65 | 34.2% |
| π0 | 30–75 | 20–65 | 35–60 | 50.4% |
| GR00T N1 | 60–100 | 60–90 | 80–95 | 82.1% |
| **Dexora** | 80–100 | 80–95 | 90–100 | **89.6%** |

Dexora 在 7/12 个子任务上达到 ≥90% 成功率,且在双臂任务上领先最稳定。

**灵巧操作任务（6 个任务：Use pen、Fetch book、Cut leek、Place plates、Rough dough、Twist cap）：**

| 方法 | Use pen | Fetch book | Cut leek | Place plates | Rough dough | Twist cap | 平均 |
|---|---|---|---|---|---|---|---|
| DP | 5 | 20 | 0 | 0 | 15 | 0 | 6.7% |
| π0 | 20 | 45 | 60 | 20 | 15 | 0 | 26.7% |
| GR00T N1 | 45 | 60 | 65 | 60 | 80 | 0 | 51.7% |
| **Dexora** | 65 | 80 | 80 | 70 | 80 | 25 | **66.7%** |

GR00T N1 是最强基线但用 6 自由度手,在需要拇指-食指协同的 in-hand 技能（如 Use pen）和 Twist cap（拧瓶盖,需要稳定扭转力矩且原地全部方法在此任务上普遍很低,Dexora 仅 25% 也是全表最低成功率任务）上表现受限。

**判别器消融（Table III,报告成功率与归一化关节加速度/jerk,越低越平顺）：**

| 任务 | 无判别器 S.R. | 无判别器 Acc./Jerk | 有判别器 S.R. | 有判别器 Acc./Jerk |
|---|---|---|---|---|
| Corn→plate | 85% | 0.034 / 0.043 | 95% | 0.020 / 0.032 |
| Lift basket | 55% | 0.041 / 0.052 | 80% | 0.023 / 0.036 |

**训练数据构成消融**（Sim Only / Sim+50%Real / Sim+All Real,评测 Apple→plate、Stack ring blocks、Use pen、Cut leek）：灵巧任务提升最明显,Use pen 从 0%→35%→65%,Cut leek 从 10%→60%→85%;基础任务也随真实数据增加稳步上升,但增幅相对温和。说明仿真主要负责"打底"基础能力,真实数据对灵巧能力的形成是关键。

**OOD 泛化**：在 "pick apple to plate" 任务上测试 unseen background / unseen lighting / unseen object / occlusion / clutter / height change 六种条件,成功率均维持在约 90%–100%（遮挡与杂乱场景相对最低,约 90%）,证明策略对分布外扰动具有较强鲁棒性。

**跨具身泛化**：论文假设"36 维双臂双手策略是低自由度具身的超集",通过对未用到的动作维度做 padding、缺失相机做 mask,在三种目标具身上各用 100 条演示微调：EC-1 单臂夹爪（Franka Emika Panda）、EC-2 双臂夹爪（Cobot Magic ALOHA,2×(6自由度臂+1自由度夹爪)）、EC-3 单臂单手（Unitree G1 7 自由度臂 + Inspire Hand 6 自由度）。结果显示抓取类任务在各具身上都能顺利迁移,而灵巧度要求高的任务迁移差距最大——支持"高维策略向低维具身投影,比反过来把夹爪策略提升到灵巧手容易"的论点。

## 四、局限性

- **触觉缺失导致的操作失败**：论文明确指出 Twist cap（拧瓶盖)在所有方法（含 Dexora)上成功率最低。该任务需要在防止打滑的同时产生稳定的扭转力矩,依赖精确的法向力调节、指尖摩擦与手内对齐,而当前系统缺乏触觉反馈、指尖为低摩擦刚性材质,导致打滑。
- **跨具身泛化并非零样本**：所谓的"降维迁移"实际上仍需要在每个目标具身上用 100 条演示做微调,并手工做维度 padding / 相机 mask,并非真正意义上的跨具身零样本策略,与 π0、RDT-1B 等直接多具身联合训练的路线在本质上不同。
- **判别器依赖的正例集合较小且为近似标签**：高质量正例集 $\mathscr{S}_{high}$ 仅占真实数据的约 15%,且 PU 学习把其余全部数据当作"无标记≈负例"处理,是一种近似;判别器的有效性和迁移性尚未在其训练分布之外（如新任务/新具身)被验证。
- **评测规模有限**：每任务仅 20 次 rollout,统计误差较大;真实数据总量（1 万 episode)相对于互联网规模的 VLA 预训练语料仍偏小。
- 遥操作依赖专用硬件（外骨骼背包 + Apple Vision Pro),数据采集成本与规模化速度仍受人工操作限制。

## 五、评价与展望

**贡献与优点**：Dexora 补上了"双臂 + 高自由度双手"这一此前开源 VLA 生态中缺失的象限（对照 π0/π0.5/RDT-1B/GR-3/GR00T N1 的双臂低自由度路线,以及 DexGraspVLA/Being-H0/Dexonomy 的单臂高自由度路线),并且数据、代码、模型全部开源,对社区复现和后续研究价值较高。将 DWBC（判别器加权行为克隆)这一原本用于离线强化学习次优演示过滤的思想,迁移到扩散 Transformer VLA 的后训练阶段,并用去噪残差能量构造 $\log\pi$ 代理特征增强判别器,是一个实用且轻量的工程化数据质量筛选方案,消融实验（Table III)也确认了其对成功率与动作平顺度的双重收益。

**开放问题与可能改进方向**：

1. **触觉闭环**：论文自己也将"通过触觉感知实现接触感知控制,解决拧瓶盖一类问题"列为未来方向,这与灵巧操作领域近年来触觉-视觉融合的趋势一致,是自然的后续工作。
2. **长时程与分层规划**：论文提出结合记忆、子目标分解与语言引导工具使用的分层 VLA 规划作为第二个未来方向,当前 200 个任务大多是单步或少步骤操作,尚未系统验证长时程多步骤任务链。
3. **判别器的可迁移性**：判别器是在与策略同分布的数据上训练的,其质量打分能否泛化到新任务、新场景甚至新具身尚未讨论,是一个值得深入的开放问题。
4. **"降维"式跨具身 vs. 联合训练式跨具身**：Dexora 的跨具身路线（高维策略 padding/mask 后逐具身微调)与 RDT-1B、π0 等通过大规模多具身数据联合预训练实现的跨具身泛化是两种不同哲学,两者的样本效率、可扩展性与最终性能上限值得系统性对比。
5. Twist cap 类任务的持续低成功率提示,单纯依靠更多演示数据可能难以突破物理层面的传感器限制（如指尖摩擦、力反馈缺失),需要软硬件协同改进而非纯数据/算法侧优化。

## 参考

1. K. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. J. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734, 2025.
3. S. Liu, L. Wu, B. Li, et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*. arXiv:2410.07864, 2025.
4. Z. Jiang, Y. Xie, K. Lin, et al. *DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning*. arXiv:2410.24185, 2024.
5. H. Xu, X. Zhan, H. Yin, H. Qin. *Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations*. ICML, 2022.
