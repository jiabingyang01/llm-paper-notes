# HumanX：面向敏捷且可泛化的人形交互技能——从人类视频学习

> **论文**：*HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos*
>
> **作者**：Yinhuai Wang, Qihan Zhao, Yuen Fui Lau, Runyi Yu, Hok Wai Tsui, Qifeng Chen, Jingbo Wang, Jiangmiao Pang, Ping Tan（Yinhuai Wang / Qihan Zhao / Yuen Fui Lau 共同一作）
>
> **机构**：The Hong Kong University of Science and Technology（香港科技大学）；Shanghai AI Laboratory（上海人工智能实验室）
>
> **发布时间**：2026 年 02 月（arXiv 2602.02473）
>
> **发表状态**：未录用（预印本），项目主页 https://wyhuai.github.io/human-x/
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.02473) | [PDF](https://arxiv.org/pdf/2602.02473)
>
> **分类标签**：`人类视频学习` `人形交互技能` `物理数据合成` `模仿学习` `whole-body control`

---

## 一句话总结

HumanX 是一套"从单目人类视频到真机技能"的全栈框架：**XGen** 把一段手机拍摄的人类视频编译成物理可信、可大规模增广的人形-物体交互数据，**XMimic** 用统一交互模仿奖励把这些数据蒸馏成可零样本上真机的策略；在篮球/足球/羽毛球/搬运/对抗五个领域、10 个技能上，仅用单条视频演示即达到比先前方法高约 8 倍的泛化成功率（平均 GSR 超 80%），并零样本迁移到 Unitree G1，能打出 pump-fake 转身后仰跳投、与人连续 10+ 回合传接球等复杂交互。

## 一、问题与动机

让人形机器人执行敏捷、自适应的交互任务长期受制于两个瓶颈：

- **数据瓶颈**：行为克隆(BC)需要大规模、昂贵的遥操作演示；而真实、多样的交互数据本就稀缺。
- **奖励瓶颈**：强化学习(RL)+物理仿真虽能减少对数据的依赖，却通常需要针对每个任务精心设计奖励函数，难以跨任务扩展。

人类视频是一座尚未开采的富矿——蕴含丰富的交互技能，但直接利用面临难点：单目视频中人体与物体姿态各自估计后**朴素拼接**会因遮挡与深度歧义产生物理上不可信的结果（穿模、悬空）。

作者的核心洞察：**对机器人技能获取而言,交互的"物理可信性"远比"光度/几何上的忠实重建"重要**。因此 XGen 不追求精确重建,而是把范式转为"用物理先验约束下的交互轨迹合成",从一条演示就能生成一大簇物理一致、可泛化的交互轨迹。目标是打通一条**无需任务特定奖励、可扩展、任务无关**的技能获取通路。

## 二、核心方法

HumanX 由两个协同设计的组件构成：数据侧的 XGen 与学习侧的 XMimic。

### 2.1 XGen：从单目视频合成物理可信交互数据

三阶段流水线：(1) 提取人体运动并重定向到机器人；(2) 物理驱动的物体轨迹合成 + 接触感知精修；(3) 沿多个维度增广以最大化覆盖。

**(a) 提取并重定向人形运动。** 给定 $K$ 帧单目 RGB 视频，先用 GVHMR 估计 3D 人体位姿序列，第 $i$ 帧记为

$$\mathbf{h}_i = (\mathbf{h}_i^{\text{root}}, \mathbf{h}_i^{\text{joint}}), \quad i=1,\dots,K$$

其中 $\mathbf{h}_i^{\text{root}}\in\mathbb{R}^6$ 是人体根节点的 6D 位姿，$\mathbf{h}_i^{\text{joint}}\in\mathbb{R}^{J\times3}$ 是 $J$ 个 SMPL 关节的 3D 旋转。再用 GMR（关键点对齐 + 骨架缩放 + IK 优化）重定向到目标人形机器人，得到

$$\mathbf{r}_i = (\mathbf{r}_i^{\text{root}}, \mathbf{r}_i^{\text{joint}}), \quad \mathbf{r}_i^{\text{root}}\in\mathbb{R}^6,\ \mathbf{r}_i^{\text{joint}}\in\mathbb{R}^{N\times1}$$

（$N$ 为机器人关节数）。

> 用大白话说：先把视频里的人"扒"成 SMPL 骨架动作,再把这套动作按体型差异"翻译"成 G1 机器人能做的关节动作。

**(b) 合成人形-物体交互（分接触/非接触两相）。** 以搬箱子为例,把视频按时间戳标注为三段：接触前非接触相($t<t_s$)、接触相($t_s\le t\le t_e$)、接触后非接触相($t>t_e$)。

- **接触相**：核心表征是一个**预定义 anchor 与物体之间的相对位姿 $\phi$ 的不变性**。anchor 有两类定义——(1) 双手掌心中点（适合双手稳定持握，如搬箱、投篮、上篮），(2) 某个特定身体部位（适合单点交互，如击打羽毛球、踢足球）。从关键帧用 SAM-3D 估计物体网格及其相对 anchor 的旋转 $\phi$（或手动指定，从而支持"视频里物体不可见"也能合成）。物体轨迹由 anchor 轨迹（源自机器人运动 $\{\mathbf{r}_i\}$）**保持 $\phi$ 不变地传播**得到；再在 **force-closure（力封闭）约束**下逐帧优化机器人位姿，得到精修后的 $\hat{\mathbf{r}}_t$ 与物体位姿 $\mathbf{p}_t$，保证接触期间物理可信。

  > 用大白话说：物体和"手"的相对关系锁死,机器人手怎么动,物体就跟着怎么动;再用抓握力学约束把手指姿态调到真能"抓稳"的样子。这个相对关系是跨形态可迁移的——人怎么拿,机器人就怎么拿。

- **非接触相**：相变处对身体位姿在 $k$ 帧窗口内做线性插值以保证平滑；物体轨迹交给物理仿真器（IsaacGym）生成。分两种情形：① 接触结束后($t>t_e$)以 $\mathbf{p}_{t_e}$ 和预定义初速度初始化仿真、正向记录（投篮、踢球、放置）；② 接触开始前($t<t_s$，如接球）则**反向仿真**——从 $\mathbf{p}_{t_s}$ 出发把物体在时间上倒着仿真再翻转序列，得到抛物线入手轨迹；为保证反向仿真物理可信，将物体阻尼系数取反。

**(c) 交互增广。** 沿三个维度扩数据：① **物体几何缩放**——在网格获取阶段缩放/替换物体网格，后续合成保证与新物体仍物理一致（同一动作作用于不同物体）；② **接触相轨迹增广**——对接触段物体轨迹做平移/缩放等几何变换（如"从不同高度搬起同一箱子"）；③ **非接触相轨迹增广**——对物体初速度做参数化随机化（如"从一条羽毛球击打演示生成不同抛物线""从一段投篮视频生成不同距离的投篮"）。由此单条视频可扩成大簇训练片段。

### 2.2 XMimic：统一交互模仿学习

统一模仿框架，含四点创新：统一交互奖励、灵活感知方案、泛化优先训练、多技能模式的可扩展获取。采用**两阶段 teacher-student**。

**策略形式。** 观测 $s_t$ 输入，动作输出为高斯分布

$$\pi(a_t \mid s_t) \sim \mathcal{N}(\phi_\pi(s_t), \Sigma_\pi)$$

$\phi_\pi$ 是预测均值的 MLP，$\Sigma_\pi$ 可学习协方差；动作 $a_t\in\mathbb{R}^n$（$n$ 为 DoF 数）经 PD 控制器转为关节力矩。

**Stage 1 — 特权 teacher。** 对 XGen 生成的 $n$ 种技能模式数据集 $\{\mathcal{D}_1,\dots,\mathcal{D}_n\}$，每个数据集单独训一个 teacher $\pi_{\text{tea}}$。teacher 接收**特权状态** $s_t=\{o_t, o_t^{\text{priv}}, s_t^{\text{ext}}\}$（本体感受 + 特权身体信息 + 物体状态），用 PPO 最大化累积奖励。

**Stage 2 — 蒸馏为可部署 student。** student 在合并数据集 $\mathcal{D}=\bigcup_i \mathcal{D}_i$ 上训练，两点区别：① 观测**剔除全部特权信息**，只留本体感受 + 可选物体观测；② 目标在 PPO 策略梯度之外加一项 BC 蒸馏损失：

$$\mathcal{L}_{\text{BC}} = \mathbb{E}_{(a,i)\sim\mathcal{G}}\left[\left\lVert \pi_{\text{stu}}(a\mid s) - \pi_{\text{tea}}^i(a\mid s)\right\rVert^2\right]$$

> 用大白话说：先让每个"专项教练"在开卷(能看到物体真值、身体特权量)条件下学好一门技能;再训一个"学生"在闭卷(只有本体感受)条件下同时模仿所有教练,这样最终策略不依赖特权信息,能直接上真机。

**统一交互模仿奖励。** 复合奖励

$$r_t = r_t^{\text{body}} + r_t^{\text{obj}} + r_t^{\text{rel}} + r_t^{c} + r_t^{\text{reg}}$$

- $r_t^{\text{body}}$：身体位置/旋转/关节位置/线速度/角速度/关节速度 + **AMP 对抗动作先验**（提升自然度），每个子项(除 AMP)形如 $r_t^\alpha=\gamma^\alpha\exp(-\lambda^\alpha e_t^\alpha)$；
- $r_t^{\text{obj}}$：物体位置/旋转跟踪；
- $r_t^{\text{rel}}$：身体-物体**相对**空间关系（相对位置误差 $e_t^{\text{rel\_p}}=\lVert\mathbf{u}_t-\hat{\mathbf{u}}_t\rVert_2$、相对旋转的测地角误差），保证跨平移+旋转都对齐；
- $r_t^{c}$：**接触图(Contact Graph)** 模仿。接触状态记为二值向量 $s_t^{cg}\in\{0,1\}^J$，奖励 $r_t^{cg}=\exp(-\sum_{j=1}^{J}\lambda_j^{cg}\,e_t^{cg}[j])$，惩罚接触时机/位置的偏差；
- $r_t^{\text{reg}}$：平滑性等正则，改善部署稳定性。

> 用大白话说：不光要"身体动作像",还要"物体轨迹像""手和物体的相对关系像""什么时候接触/松手也像"——四个维度一起对齐,才能真正复现交互而不是只学个空挥手。

### 2.3 感知设计与泛化机制

**从本体感受感知外力（无力/力矩传感器）。** 浮基人形动力学方程

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) + \boldsymbol{\tau}_f + \mathbf{J}_{\text{ext}}^{\top}\mathbf{F}_{\text{ext}}$$

移项得

$$\mathbf{J}_{\text{ext}}^{\top}\mathbf{F}_{\text{ext}} = \boldsymbol{\tau} - \left(\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{G} + \boldsymbol{\tau}_f\right)$$

在 G1 上 $\mathbf{q},\dot{\mathbf{q}}$ 可测、$\boldsymbol{\tau}$ 由 PD 命令力矩近似、$\ddot{\mathbf{q}}$ 用多帧速度历史隐式提供、其余项近似常数——因此策略只要观测里含这些量，就能**隐式感知外力**，据此做更好的动态交互。

> 用大白话说：人闭着眼也能靠"手感"稳稳接球投篮;机器人同理,把力矩、关节位置/速度、速度历史喂给策略,它就能从动力学方程里"算出"手上受了多大外力,不用装力传感器。

**两种部署模式。** ① **NEP（无外部感知）**：student 训练时去掉物体观测，纯靠本体感受，支持投篮/上篮/运球/pump-fake 转身跳投等——部署最简单鲁棒，但处理不了接飞来的球这类非接触交互；② **MoCap 模式**：物体位姿由动捕提供，但动捕常因遮挡丢帧，故训练时**注入仿真丢帧**，实现对真实动捕间歇性丢包的零样本适应。

**泛化三支柱**：(1) XGen 的多样离线数据覆盖广物体状态；(2) 训练中**扰动初始化(DI)**——对根旋转/根位移/关节角/物体初始位姿加随机扰动，在线扩展状态覆盖；(3) **交互终止(IT)**——加性奖励易收敛到"只学身体动作、忽略交互"的局部最优，故当参考帧处于接触态时监控物体与关键身体的相对位置误差，超阈值则以一定概率终止 episode，优先保交互成功、抑制对身体动作的过拟合。此外配合域随机化(DR：物体尺寸/质量/恢复系数、摩擦、质心偏移、感知噪声、持续随机外力)。

## 三、实验结果

**设置**：视频用 iPhone 16 拍；训练/仿真在 Isaac Gym，单张 NVIDIA RTX 4090、16,384 并行环境，每策略默认训 20,000 迭代；真机为 Unitree G1；MoCap 用 Noitom 光学系统（5×5×2.6m 空间、14 相机），策略与动捕 100 Hz、底层 PD 1000 Hz。仿真三任务：Basketball Catch-Shot（接传来的球并投中，落点距篮心 20 cm 内算成功）、Badminton Hitting（击中飞来羽毛球）、Cargo Pickup（走向并举起随机放置的货物，抬到目标高度 10 cm 内）。指标：原演示上的物体位置误差 $E_o$、关键身体位置误差 $E_h$、成功率 SR，以及**泛化范围内成功率 GSR**（测例采样自增广分布，如球初始位置 ±0.3 m 均匀扰动、货物在正前方 3 m 半径半圆内随机）。

**主结果（与先前 HOI 模仿方法对比，XMimic 为最终 +Tea-Stu 版本）：**

| 任务 | 方法 | SR ↑ | GSR ↑ |
|---|---|---|---|
| Basketball Catch-Shot | SkillMimic | 0.0% | 0.0% |
| | HDMI（最强基线） | 53.1% | 2.4% |
| | **XMimic** | 86.8% | **64.7%** |
| Badminton Hitting | SkillMimic | 68.2% | 30.9% |
| | HDMI | 90.0% | 25.3% |
| | **XMimic** | 100% | **90.6%** |
| Cargo Pickup | SkillMimic | 0.4% | 0.0% |
| | HDMI | 95.8% | 1.8% |
| | **XMimic** | 99.3% | **96.3%** |

SkillMimic / OmniRetarget 在这三个真机导向任务上几乎全崩（多为 0% GSR）；HDMI 在原演示上 SR 尚可，但 GSR 仅个位数，泛化极弱。XMimic 最终版平均 GSR 超 80%，约为 HDMI 的 8 倍。

**消融（以羽毛球 GSR 为例，最能体现各组件贡献）：** Base 41.6% → +DI(扰动初始化) 66.0% → +IT(交互终止) 67.2% → +Data Aug(XGen 增广，从 1 段扩到 50 段) 84.4% → +Tea-Stu 90.6%。可见数据增广与师生蒸馏是泛化的主要来源；篮球投篮 GSR 也从 Base 4.9% 一路升到 64.7%。注意 +Data Aug 后原演示 SR 会略降（如篮球 93.4%→77.6%），这是"学可泛化技能"的自然权衡。

**多模式技能（单 student 学一个技能内的三种不同交互模式，Table II）：**

| 技能 | w/o Tea-Stu | w/ Tea-Stu |
|---|---|---|
| Football Kicking (GSR) | 74.2% | 93.1% |
| Badminton Hitting (GSR) | 52.4% | 84.3% |

师生框架在多模式设定下增益尤其显著。

**真机（Table III）：**

| 模式 | 技能 | 成功率 |
|---|---|---|
| MoCap | Basketball Catch-Pass | 41 / 50 |
| MoCap | Cargo Pickup | 43 / 50 |
| MoCap | Football Kicking | 42 / 50 |
| MoCap | Reactive Fighting | 37 / 50 |
| NEP | Basketball Pickup | 10 / 10 |
| NEP | Jumpshot | 8 / 10 |
| NEP | Dribble | 8 / 10 |
| NEP | Pump-fake 转身后仰 | 9 / 10 |
| NEP | Layup | 7 / 10 |
| NEP | Spin Move | 9 / 10 |

真机上可与人连续完成 10+ 回合篮球接传、14+ 次连续回踢足球。并观察到**涌现的自适应行为**：搬运时被人推仍稳握并补偿；物体被拿走放到别处，机器人自主走过去重新拾起；对抗中能区分假动作与真出拳，仅对真攻击做完整防反。Sim-to-real 分析表明：训练必须含**持续随机外力**否则高动态交互时易失衡；必须**模拟动捕丢帧**否则物体信号临时丢失时会崩溃。

## 四、局限性

- **接触相强依赖 anchor 假设**：把"anchor-物体相对位姿不变"作为核心表征，对稳定持握/单点击打有效，但对手内滑移、抓握重构、多次换手、可形变或铰接物体等场景可能不成立。
- **单点估计的脆弱性**：物体网格与初始 $\phi$ 依赖 SAM-3D 单帧估计（或人工指定），关键帧估计误差会传导进整条合成轨迹；相变处仅用线性插值，快速非接触段的动力学未必精确。
- **NEP 模式能力边界明确**：纯本体感受处理不了接飞来的球等非接触交互，仍需 MoCap；而 MoCap 依赖外部动捕基础设施，限制了"任意场景即插即用"。
- **奖励虽"无任务特定项",但结构仍复杂**：body/obj/rel/contact/reg 五大类加多项子奖励、多个 $\lambda/\gamma$ 超参与接触图标注，工程量与调参成本不小；"无任务奖励"更多指不用手工设计任务目标函数，而非零配置。
- **真机 MoCap 任务 SR 约 74%–86%**，对抗(37/50)偏低；评估集中在体育/搬运类交互，尚未覆盖精细操作(如插拔、拧动)与长程多物体任务。
- **单条演示的多样性上限**：增广维度(几何缩放、平移缩放、初速度随机)是围绕单条演示的参数化扰动，无法凭空创造演示中不存在的运动风格/技能。

## 五、评价与展望

**优点**：这是一篇工程完整度很高的"全栈"工作，把"单目视频 → 物理可信数据 → 统一模仿 → 真机零样本"整条链路打通，且核心 insight（物理可信 > 精确重建）干净有说服力。其两点方法论价值突出：① **anchor 相对位姿不变性**作为交互的跨形态可迁移表征，绕开了人-机运动学差异这一老大难；② 从本体感受**隐式感知外力**的动力学推导，为"无力传感器的力感知交互"提供了简洁理论支撑，NEP 模式因此格外鲁棒（真机盲投类技能 8–10/10）。泛化提升(8×)有清晰的消融归因，Data Aug 与 Tea-Stu 是主推力。

**与相关工作的关系**：本质是把 PhysHOI / SkillMimic（仿真内交互模仿）与 HDMI（从人类视频学交互控制）向"真机 + 强泛化 + 无任务奖励"推进。相对 OmniRetarget（交互保持的数据生成）和 GMR（通用运动重定向），HumanX 的差异在于**接触相物理精修 + 非接触相(含反向)仿真 + 多维增广**三者合一地服务于"可泛化真机技能"，而非仅做数据/重定向。实验里 SkillMimic/OmniRetarget 在真机导向任务上几乎全崩、HDMI 泛化个位数，从侧面印证"泛化优先训练(DI+IT)+ 物理增广"确是这一代方法与前代拉开差距的关键。

**开放问题与可能改进**：(1) 把 anchor 从"固定相对位姿"放宽为**可学习/时变的接触表征**，以覆盖手内操作与可形变物体；(2) 用视频/多视角先验或可微渲染替代单帧 SAM-3D 的点估计，降低关键帧误差传导；(3) 探索**纯视觉(egocentric RGB)部署**以摆脱对外部 MoCap 的依赖，让 MoCap 模式技能也能"即插即用"；(4) 把这条数据合成管线扩到精细操作与长程多物体任务，检验"物理可信优先"范式在低容错场景下的上限；(5) 增广目前是围绕单演示的参数扰动，若能与生成式运动先验结合，或可跨演示合成新技能组合。总体看，HumanX 为"人类视频→人形交互技能"提供了一个强基线与清晰的数据合成范式，其"接触/非接触分相 + force-closure 精修 + 反向仿真"的数据侧设计，对更广义的具身交互数据引擎有直接参考价值。

## 参考

1. **SkillMimic**：Wang et al., *Skillmimic: Learning basketball interaction skills from demonstrations*, CVPR 2025. —— 仿真内篮球交互技能模仿，本文重要基线与接触图奖励来源。
2. **HDMI**：Weng et al., *HDMI: Learning interactive humanoid whole-body control from human videos*, arXiv:2509.16757, 2025. —— 从人类视频学交互控制，本文最强对比基线。
3. **PhysHOI**：Wang et al., *Physhoi: Physics-based imitation of dynamic human-object interaction*, arXiv:2312.04393, 2023. —— 物理驱动的动态人-物交互模仿，交互模仿范式的前驱。
4. **OmniRetarget**：Yang et al., *Omniretarget: Interaction-preserving data generation for humanoid whole-body loco-manipulation and scene interaction*, arXiv:2509.26633, 2025. —— 交互保持的数据生成，与 XGen 直接可比。
5. **GMR**：Araujo et al., *Retargeting matters: General motion retargeting for humanoid motion tracking*, arXiv:2510.02252, 2025. —— 本文所用的人体到机器人通用运动重定向方法。
