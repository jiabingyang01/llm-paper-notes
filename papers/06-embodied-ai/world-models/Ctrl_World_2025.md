# Ctrl-World：一个用于机器人操作的可控生成式世界模型

> **论文**：*Ctrl-World: A Controllable Generative World Model for Robot Manipulation*
>
> **作者**：Yanjiang Guo\*, Lucy Xiaoyang Shi\*, Jianyu Chen, Chelsea Finn（\* 同等贡献）
>
> **机构**：Stanford University；Tsinghua University
>
> **发布时间**：2025 年 10 月（arXiv 2510.10125）
>
> **发表状态**：ICLR 2026（Published as a conference paper at ICLR 2026）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.10125) | [PDF](https://arxiv.org/pdf/2510.10125)
>
> **分类标签**：`世界模型` `机器人操作` `策略评估` `多视角视频扩散` `合成数据后训练`

---

## 一句话总结

Ctrl-World 把一个预训练的被动视频扩散模型（SVD 1.5B）改造成可与通才 VLA 策略"闭环交互"的世界模型——通过 multi-view 联合预测、pose-conditioned 记忆检索与 frame-level 动作条件三项改造，实现在 DROID 上 20 秒以上的时空一致 imagination rollout；不需真机就能对策略排名(instruction-following 相关性 $y=0.87x-0.04$)，并用世界模型内部合成的成功轨迹做后训练，把 $\pi_{0.5}$ 在新指令/新物体上的成功率从 38.7% 提升到 83.4%（相对 +44.7%）。

## 一、问题与动机

通才机器人策略(VLA)如今能做很多操作技能,但**评估**与**改进**它们在陌生物体/指令上的能力仍是瓶颈:

- **评估贵**:严谨评估需要大量真机 rollout,在多任务多环境上反复重复才能得到统计显著的结论,慢、贵、难扩展。
- **改进难**:一旦发现弱点,现有方法只能靠额外采集带专家标注的纠正数据,同样昂贵。

世界模型(在 imagination 里 rollout)是一条可扩展的替代路线,但作者指出现有面向机器人的视频预测/世界模型存在三个硬伤,使其无法真正与现代 VLA 策略"在环交互":

1. **单视角**:多数模型只模拟单个第三人称视角,导致严重的部分可观测(如物体在没有物理接触时"吸附"进夹爪的幻觉),且不兼容需要"第三人称 + 腕部相机"双输入的现代 VLA。
2. **控制粒度粗**:通常只以语言指令为条件,缺乏刻画高频动作因果效应所需的细粒度动作控制。
3. **长程不一致**:长视频生成中难以维持时间一致性,误差累积、漂移。

本文目标:构造一个**多视角、可控、长程一致** 的世界模型,使策略能完全在想象空间里做多步 rollout,从而既能**评估** 又能**改进** 策略。

## 二、核心方法

### 问题形式化

现代通才策略 $\pi$ 把多视角观测和语言指令映射为动作序列。观测 $o_t = [I_t^1, ..., I_t^n, q_t]$ 含 $n$ 个相机视图与机械臂位姿 $q_t$;给定指令 $l$,策略输出 $H$ 步动作 chunk:

$$a_{t+1}, a_{t+2}, ..., a_{t+H} \sim \pi(\cdot \mid o_t, l)$$

世界模型 $W$ 需要根据这一动作块预测未来多视角观测:

$$o_{t+1}, ..., o_{t+H} \sim W(\cdot \mid o_t, A_t)$$

其中 $A_t = [a_{t+1}, ..., a_{t+H}]$。随后把预测 $o_{t+H}$ 再喂回策略得到下一个动作块 $A_{t+H} \sim \pi(\cdot \mid o_{t+H}, l)$,策略与世界模型**自回归交互**,完全在想象空间里做长程 rollout。

> **用大白话说**:策略像司机,世界模型像模拟器。司机看着模拟画面打方向盘(出动作),模拟器根据方向盘更新画面,两者你来我往,就能在"梦里"把一整段任务开完,全程不碰真车。

### 三项关键改造(从预训练 SVD 出发)

初始化自 1.5B 的 Stable Video Diffusion,唯一新初始化的模块是一个把 7 维笛卡尔空间动作投影到 1024 维隐空间的 3 层 MLP,其余参数保持预训练权重。

**(1) 多视角联合预测(Multi-View Joint Prediction)**。把 $N$ 个输入视图(2 个第三人称 + 1 个腕部相机)的 token 拼接,在每一步联合预测所有视图 $o_{t:t+H}$。这既匹配现代 VLA 的双输入格式,又显著提升一致性、大幅减少接触阶段的幻觉——腕部视角的联合预测提供了物体接触/状态变化的细粒度信息。

**(2) Pose-conditioned 记忆检索(Memory Retrieval)**。为抑制长 rollout 的误差累积与漂移,把过去帧加入上下文;为避免上下文过长,以步长 $m$ 稀疏采样 $k$ 帧历史 $[o_{t-km}, ..., o_t]$。关键在于把对应的机械臂位姿 $[q_{t-km}, ..., q_t]$ 通过 spatial transformer 内的 **frame-wise cross-attention** 嵌入到每一帧,使模型能用臂位姿识别过去的相似状态,把未来预测"重新锚定" 到相关历史帧。论文可视化显示,预测 $t=4s$ 帧时对同位姿的 $t=0s$ 帧有很强注意力。

**(3) Frame-level 动作条件(Frame-level Action Conditioning)**。预训练视频模型只以文本/图像为条件,控制精度不足。作者把策略输出的动作序列 $a_{t+1:t+H}$ 转成笛卡尔空间臂位姿 $a'_{t+1:t+H}$,与历史位姿拼接后,在 spatial transformer 里对每帧视觉 token 做逐帧 cross-attention,使每一帧对齐其对应的动作嵌入——历史帧对应 $[q_{t-km}, ..., q_t]$,未来帧对应 $a'_{t+1:t+H}$。这让生成 rollout 紧跟控制信号、反映每个动作的因果效应。

### 训练目标

以预训练 SVD 为骨干,用扩散损失微调。前向加噪 $x_{t'} = \sqrt{\alpha_{t'}} x_0 + \sqrt{1-\alpha_{t'}}\,\epsilon_{t'}$,预测目标 $x_0 = o_{t+1:t+H}$,总目标为:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t'} \big\| \hat{x}_0(x_{t'}, t', c) - x_0 \big\|^2$$

其中条件 $c = [q_{t-km}, ..., q_t,\; a'_{t+1:t+H},\; o_{t-km}, ..., o_t]$ 汇集所有模型输入。

> **用大白话说**:训练就是让模型看着"稀疏历史帧 + 历史位姿 + 未来动作",从一堆噪声里把未来该出现的画面还原出来。位姿告诉它"以前哪一帧长得像现在",动作告诉它"手接下来要怎么动"。

### 用世界模型评估与改进策略(Algorithm 1)

- **评估**:给定初始观测 $o_0$ 与指令 $l$,策略与世界模型自回归生成合成轨迹 $\tau$;由人类偏好判断该轨迹是否成功。世界模型内的排名可直接对齐真机排名。
- **改进**:VLA 策略行为高度确定(总去抓同一个物体),为扩大搜索空间,作者引入两种结构化扰动增加 rollout 多样性——(i) 用 LLM API **改写指令**(如"place glove in box"→"pick up the cloth and put it inside the box");(ii) 把策略**重置到随机初始状态**。生成合成 rollout 后按人类偏好打分,保留成功轨迹构成 $D_s$,再用监督损失微调策略:

$$\mathcal{L}_\theta = \mathbb{E}_{o_t, a_{t:t+H}\sim D_s}\big\| \pi_\theta(o_t, l) - a_{t:t+H} \big\|^2$$

### 与官方 VLA 策略对接(适配器)

官方 $\pi_0$-DROID / $\pi_0$-FAST-DROID / $\pi_{0.5}$-DROID 输入关节角、输出关节速度,而世界模型以笛卡尔位姿为条件。作者训练一个 2 层 MLP **adapter** 把当前关节角 + 预测关节速度映射为未来关节构型,再用 Franka Panda 正运动学 FK 转成笛卡尔位姿喂给世界模型,从而实现无缝的想象空间自回归交互。

### 实现要点

VAE 空间下采样 $8\times 8$;$k=7$ 帧历史(各帧独立加噪);动作条件窗口 1 秒 = 15 步,时间下采样到 5 步;每帧 $3\times 192\times 320$ 图像编码为 $24\times 40$ 隐特征,输入 token 形状 $B\times(7+5)\times(3\times 24\times 40)$;2×8 张 H100,batch size 64,lr 1e-5,训练 100k 步约 2–3 天。

## 三、实验结果

**平台与数据**:DROID(Panda 臂 + Robotiq 夹爪,1 腕部 + 2 个随机放置的第三人称相机),95,599 条轨迹、564 个场景,含约 76k 成功 + 19k 失败。分辨率 192×320,7 帧历史(帧间隔 1–2 秒),条件于 15 步(1 秒)未来动作。

### 世界模型质量(表 1)

10 秒轨迹、每步 15 步动作块、自回归 10 轮、256 clip 平均。第三人称视角:

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ |
|---|---|---|---|---|---|
| WPE-Single-View | 20.33 | 0.772 | 0.131 | 25.50 | 156.4 |
| WPE-Multiview | 21.17 | – | – | – | 147.1 |
| IRASim-Single-View | 21.36 | 0.774 | 0.117 | 26.46 | 138.1 |
| IRASim-Multiview | 20.21 | – | – | – | 165.4 |
| Ctrl-World-Single-View | 21.27 | 0.793 | 0.110 | 23.47 | 127.5 |
| **Ctrl-World(完整)** | **23.56** | **0.828** | **0.091** | 25.00 | **97.4** |

Ctrl-World 全面优于 WPE、IRASim;multi-view 联合预测进一步提升生成质量(FVD 127.5→97.4)。基线在机器人–物体交互处常产生幻觉(如"移绿毛巾/抓红碗"失败),Ctrl-World 靠联合预测腕部视角精确建模接触事件。

### 消融(表 2)

| 视角 | 变体 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ |
|---|---|---|---|---|---|---|
| 第三人称 | Ctrl-World | 23.56 | 0.828 | 0.091 | 25.00 | 97.4 |
| 第三人称 | w/o memory | 23.06 | 0.812 | 0.099 | 26.14 | 105.5 |
| 第三人称 | w/o frame-level cond | 21.20 | 0.789 | 0.109 | 27.52 | 122.7 |
| 腕部 | Ctrl-World | 19.18 | 0.665 | 0.252 | 25.78 | 127.1 |
| 腕部 | w/o memory | 18.84 | 0.655 | 0.265 | 26.23 | 133.1 |
| 腕部 | w/o frame-level cond | 15.69 | 0.571 | 0.375 | 33.51 | 179.1 |
| 腕部 | w/o joint pred | 15.94 | 0.580 | 0.345 | 26.46 | 158.1 |

去掉任一组件都掉点:去 memory 使画面变糊,去 frame-level 条件使控制精度下降(腕部视角受影响最大),去联合预测则腕部视角显著恶化。定性上模型可对相差仅几厘米的动作产生不同的精确未来(cm 级可控)。

### 策略评估(图 7)

在自建 DROID 平台上零样本评估 $\pi_0$、$\pi_0$-FAST、$\pi_{0.5}$,涵盖 Pick-Place / Fold-Towel / Drawer / Wipe-Table / Close-Laptop / Pull-tissue / Stack 七类任务。想象空间与真机结果的相关性:

- **Instruction-following**:$y = 0.87x - 0.04$
- **Success rate**:$y = 0.81x - 0.11$

即世界模型能忠实还原策略的**高层指令跟随**排名,但**低估执行成功率**(低层精细执行、碰撞/滑动/旋转等物理建模不足;且真机里策略失败后会反复重试,世界模型有时未捕捉)。

### 策略改进(图 9 + 表 4–7)

以 $\pi_{0.5}$ 为基座,每任务生成 400 条轨迹、保留 25–50 条成功轨迹,在合成集上微调 2k 步(4×H100)。四类下游任务:

| 类别 | Base $\pi_{0.5}$ | 后训练后 |
|---|---|---|
| Spatial Understanding | 0.2875 | 0.875 |
| Shape Understanding | 0.4374 | 0.9125 |
| Towel-Folding-Direction | 0.575 | 0.80 |
| Novel Objects | 0.25 | 0.75 |
| **平均** | **38.7%** | **83.4%** |

后训练把平均成功率从 38.7% 提到 83.4%(**相对 +44.7%**),完全不依赖真机数据,靠世界模型内部合成的成功轨迹即可对齐新指令、新物体。

## 四、局限性

- **低层执行低估**:在需要精确物理交互(碰撞、物体滑移、旋转)或长程推理的任务上会失败;世界模型系统性低估真机成功率($y=0.81x-0.11$),因此适合改进 instruction-following,而对"已见指令上的低层成功率"帮助有限。
- **对初始观测敏感**:性能受初始 observation 影响较大。
- **失败模式覆盖不足**:DROID 虽含失败数据,但数据分布外仍有大量失败模式未被建模;真机中"失败后重试"行为常未被捕捉。作者预期采集更多在环 rollout 数据可缩小这一差距。
- **成功判定仍靠人**:成功/失败由人类偏好标注,尚未用 VLM reward model 自动化(留作 future work)。
- **动作/视图配置绑定 DROID**:框架依赖笛卡尔位姿 + FK adapter 与固定三相机布局,迁移到异构本体/相机需重新适配。

## 五、评价与展望

**优点**:(1) 把"被动视频扩散骨干 → 可控交互式模拟器"这条改造路径讲得干净且有效,三项改造(多视角联合、pose 记忆检索、frame-level 动作条件)各有清晰消融支撑,不是堆砌;(2) 真正做到**策略在环**(policy-in-the-loop),与官方 $\pi_0/\pi_{0.5}$ 无缝对接,评估结论用相关性直线量化,而非只看画质指标;(3) 打通"评估 + 改进"闭环,+44.7% 的后训练收益是较强的实用证据,且全程不碰真机,数据成本可观地降低;(4) 基于开源 DROID + 开源策略,可复现性好。

**与其他公开工作的对比**:相较 IRASim(Zhu et al. 2024)、WPE(Quevedo et al. 2025)等单视角 action-conditioned 模拟器,Ctrl-World 的差异化在多视角(含腕部)联合预测与显式 pose 记忆,长程一致性(FVD 97.4)与可控性明显更好。与 UniSim/Genie 系列这类大规模交互世界模型相比,它更聚焦真机操作、直接服务 VLA 后训练而非通用交互;与 GR00T-dreams/Genie-Envisioner/DreamGen(neural trajectories)一类"用视频生成造训练数据"的工作相比,Ctrl-World 的独特点是把生成器嵌入策略闭环、用策略自身动作条件生成并回收成功轨迹,而非离线一次性造数据。

**开放问题与可能改进**:① 低层物理保真度是核心短板,可考虑引入显式动力学/接触先验、深度或点云条件(如 ParticleFormer 一类的粒子/3D 表示),或用真机在环 rollout 做 DAgger 式修正来缩小 sim-real 成功率 gap;② 成功判定依赖人工,接入 VLM/foundation reward model 自动打分是自然的下一步,可让 Algorithm 1 全自动闭环、支持在线迭代;③ 目前只做一轮离线后训练,迭代式 rollout-微调(策略与世界模型交替更新)是作者亦点明的方向,但需警惕世界模型偏差被策略放大的分布漂移;④ 结构对 DROID 的强绑定(笛卡尔 + FK adapter、固定相机)限制了跨本体泛化,统一的动作/相机表征是扩展点;⑤ 后训练收益主要来自 instruction-following 对齐,对"精细操作技能本身"的提升尚未验证,值得进一步分离 credit。

## 参考

1. Zhu et al. *IRASim: Learning Interactive Real-Robot Action Simulators*. arXiv:2406.14540, 2024.（最直接的 action-conditioned 模拟器基线)
2. Quevedo, Liang, Yang. *Evaluating Robot Policies in a World Model (WPE)*. arXiv:2506.00613, 2025.（世界模型做策略评估的基线)
3. Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset*. arXiv:2403.12945, 2024.（训练与评测平台)
4. Blattmann et al. *Stable Video Diffusion*. arXiv:2311.15127, 2023.（1.5B 视频扩散骨干)
5. Physical Intelligence et al. *$\pi_{0.5}$: a VLA with Open-World Generalization*. arXiv:2504.16054, 2025.（被评估/被改进的通才策略)
