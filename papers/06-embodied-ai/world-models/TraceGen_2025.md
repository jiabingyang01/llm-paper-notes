# TraceGen：在 3D 轨迹空间中做世界建模,实现从跨本体视频中学习

> **论文**：*TraceGen: World Modeling in 3D Trace-Space Enables Learning from Cross-Embodiment Videos*
>
> **作者**：Seungjae Lee\*、Yoonkyo Jung\*、Inkook Chun、Yao-Chih Lee、Zikui Cai、Furong Huang et al.(\* 共同一作)
>
> **机构**：University of Maryland, College Park；New York University
>
> **发布时间**：2025 年 11 月(arXiv 2511.21690)
>
> **发表状态**：未录用(预印本)
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.21690) | [PDF](https://arxiv.org/pdf/2511.21690)
>
> **分类标签**：`世界模型` `3D轨迹` `跨本体学习` `few-shot操作`

---

## 一句话总结

TraceGen 提出把世界模型的输出空间从"像素/语言 token"换成一个紧凑的 **3D trace-space**(场景关键点的未来 3D 轨迹),用配套数据引擎 TraceForge 把 12.3 万条人手+异构机器人视频统一转成 3D 轨迹标注(共 180 万条 observation–trace–language 三元组)做预训练;仅用 5 条同本体机器人视频做 warmup 即在真机 4 个任务上达到 **80%** 成功率,仅用 5 条未标定手机拍摄的人类演示做 human→robot 迁移也达 **67.5%**,推理比视频生成式世界模型快 **50–600×**。

## 一、问题与动机

机器人操作要泛化到新平台、新场景,靠采集大量特定机器人的专家演示既慢又贵;而人类视频虽海量易得,却因**本体(embodiment)、相机、环境**三重差异难以直接复用。作者的核心问题是:能否利用跨本体视频来突破新机器人/新任务的小样本(small-data)困境?

现有世界模型的输出空间各有硬伤(论文 Fig. 3 给出失败案例):

- **像素空间(视频生成)**:如 AVDC、NovaFlow(基于 Wan2.2 / Veo3.1)。把容量浪费在与控制无关的背景和纹理上,推理极慢,还会幻觉出错误几何/可供性(hallucinated gripper、physically infeasible motion)。
- **语言 token 空间(VLM planner)**:token 的时空分辨率不足以表达细粒度物体运动;有的把动作表示成 skill token,又受限于预定义的抽取器。
- **2D trace / flow 空间**:如 3DFlowAction、Im2Flow2Act。虽更高效,但多依赖静态相机或物体检测/bounding box,误差级联且无法捕捉机器人自身运动,物理表示不完整。

关键洞察:尽管不同本体在运动学与尺度上差异巨大,被操作物体与末端执行器的**运动本身共享同一套以场景为中心的 3D 结构**。作者把这个紧凑符号表示称为 trace-space——一串 3D 轨迹,保留运动的"在哪里、怎么动"(where & how),而抛弃外观与背景,从而天然对相机与环境不变,给复用跨本体、in-the-wild 视频提供了一条可行路径。

## 二、核心方法

整个系统分两块:数据引擎 **TraceForge**(把异构视频转成一致的 3D 轨迹标注)与世界模型 **TraceGen**(在 trace-space 里预测未来运动)。

### 2.1 表示:3D trace-space

在每个事件片段开头选一张参考帧,在其图像上撒一个均匀的 $20\times20$ 关键点网格 $K$,追踪它们未来 $L$ 步。每个 3D 轨迹点写成 $(x,y,z)$,其中 $(x,y)$ 是图像平面坐标、$z$ 是对应深度。这样 3D 轨迹与 2D 轨迹共享同一套屏幕对齐(screen alignment),可对 2D/3D 两种模态做一致监督与联合训练。

**用大白话说**:不去猜整张未来画面,而是只盯着场景里撒的一把"钉子",预测这些钉子接下来往哪儿飘。既保留三维几何(带深度 $z$),又能和纯二维标注混着用。

### 2.2 TraceForge 数据引擎(四步)

1. **事件切分 + 指令生成(Sec 3.1)**:从视频中切出任务相关片段;用 VLM 为每段生成三种互补指令——简短祈使句、分步骤分解、自然的类人请求,增强对措辞的鲁棒性。
2. **带相机位姿与深度的 3D 点追踪(Sec 3.2)**:采用 TAPIP3D 作为 3D 追踪模型、CoTracker3 作为点追踪器;为提效,把其中的 MegaSaM 换成来自 SpatialTrackerV2 的微调版 VGGT 深度+相机位姿预测器,精度相当但无需 3D 优化因而快得多。另外单跑 CoTracker3 得到纯 2D 轨迹以扩充数据,语料中约 **20%** 为 2D-only。
3. **世界坐标→相机坐标变换(Sec 3.3)**:把所有 3D 轨迹变换到参考相机系 $\mathrm{cam}_{\mathrm{ref}}$,以保持跨时间的视角一致性(point-of-view consistency),有效补偿相机运动。
4. **速度重定标(Speed retargeting,Sec 3.4)**:同一任务的人类/机器人演示时长与执行速度不同。沿 3D 路径计算累积弧长,按归一化弧长参数重参数化,再在 $L$ 个等间隔目标点上重采样,把每条轨迹在保持相对运动轮廓的前提下时序归一到固定长度 $L$,消除跨本体速度差异。

产物 **TraceForge-123K**:12.3 万段 episode(约 180 万条 observation–trace–language 三元组),来自 8 个数据源。人手视频占 36%(SSV2 35K、Epic-Kitchen 10K),机器人视频占 64%(Agibot 35K、BridgeV2 17K、Droid 13K、Libero 6K、OXE 6K、Robomimic 1K)。作者称这是比同类工作大 $15\times$ 的 image–trace–language 三元组数据。

### 2.3 TraceGen 架构(Sec 4)

一个 flow-based 世界模型,骨干改编自 CogVideoX 的 3D transformer,并采用 Prismatic-VLM 的多编码器融合策略。

**多编码器特征提取**(所有编码器全程冻结,只训融合层+解码器,沿用 Prismatic VLM 的结论——微调视觉骨干会灾难性遗忘预训练先验):

- RGB 双流:DINOv3(ViT-L/16)给几何感知特征 $\mathbf{F}_{\mathrm{dino}}$;SigLIP(Base-Patch16-384)给语义对齐特征 $\mathbf{F}_{\mathrm{siglip}}$。
- 深度编码器:深度图经带 learnable stem adapter($1\times1$ 卷积把单通道深度投到三通道)的第三个编码器,得 $\mathbf{F}_{\mathrm{depth}}$。
- 文本:冻结的 T5-base,固定 $M=128$ token,维度 $D=768$。

Prismatic 融合沿特征维拼接三条视觉流再线性投到统一维度 $D=768$:

$$\mathbf{F}_{\mathrm{vis}} = \mathrm{Linear}\big(\mathrm{Concat}(\mathbf{F}_{\mathrm{dino}}, \mathbf{F}_{\mathrm{siglip}}, \mathbf{F}_{\mathrm{depth}})\big)$$

视觉 token 与文本 token 拼成条件输入 $\mathbf{F}_{\mathrm{cond}}$,经 Adaptive LayerNorm 注入解码器。

**用大白话说**:让"擅长看几何的眼睛(DINOv3)+ 擅长看语义的眼睛(SigLIP)+ 专门看深度的眼睛"三只眼一起看这一帧,拼出统一的视觉线索,再配上文本指令,一起告诉解码器"该往哪画轨迹"。

**Flow 解码器**:输入是 $K\times L$ 网格,$K=20\times20$ 个空间关键点在 $L=32$ 个未来时刻上被追踪,每点是相机系里的 $(x,y,z)\in\mathbb{R}^3$。做 $2\times2$ 的空间 patchify(每 $2\times2$ 组关键点当一个 token),每帧得 $10\times10$ 个空间 token。

### 2.4 用 stochastic interpolant 做轨迹生成

模型不直接预测网格的绝对 3D 坐标,而是预测**速度式的帧间增量**:

$$\Delta \mathbf{T}^{t}_{\mathrm{ref}} = \mathbf{T}^{t+1}_{\mathrm{ref}} - \mathbf{T}^{t}_{\mathrm{ref}}$$

把该增量记作目标数据 $\mathbf{X}\in\mathbb{R}^{K\times L\times 3}$。采用 Stochastic Interpolant 框架(Albergo et al.),它用数据分布与噪声分布之间的插值路径统一了 diffusion 与 flow-matching。随机插值定义为

$$\mathbf{I}_\tau = \alpha_\tau \mathbf{X}^1 + \sigma_\tau \boldsymbol{\varepsilon}, \quad \tau\in[0,1]$$

其中 $\mathbf{X}^1$ 是真值轨迹增量、$\boldsymbol{\varepsilon}\sim\mathcal{N}(0,\mathbf{I})$。网络学习的速度场为

$$\mathbf{v}(\mathbf{x},\tau,\mathbf{F}_{\mathrm{cond}}) = \mathbb{E}\big[\dot{\mathbf{I}}_\tau \mid \mathbf{I}_\tau=\mathbf{x}, \mathbf{F}_{\mathrm{cond}}\big]$$

作者取**线性插值 ODE**:$\alpha_\tau=\tau,\ \sigma_\tau=1-\tau$,于是

$$\mathbf{X}^\tau = (1-\tau)\mathbf{X}^0 + \tau \mathbf{X}^1, \quad \tau\in[0,1]$$

此线性 schedule 下速度场简化为 $\dot{\mathbf{X}}^\tau=\mathbf{X}^1-\mathbf{X}^0$,与时间无关。训练目标即回归这个常数速度:

$$\mathcal{L}_{\mathrm{SI}} = \mathbb{E}_{\tau,\mathbf{X}^0,\mathbf{X}^1}\Big[\big\|v_\theta(\mathbf{X}^\tau,\tau,\mathbf{F}_{\mathrm{cond}}) - (\mathbf{X}^1-\mathbf{X}^0)\big\|^2\Big]$$

**用大白话说**:从纯噪声 $\mathbf{X}^0$ 到真实轨迹增量 $\mathbf{X}^1$ 连一条直线,让网络学"这条直线的斜率(速度)"。因为是直线,速度处处相同,学起来干净;测试时从噪声出发沿速度场积分即可生成轨迹。测试用 **100 步 ODE 积分**,Brush/Clothes 任务开 classifier-free guidance(guidance scale 2)。

### 2.5 执行

TraceGen 学到的是与本体无关(embodiment-agnostic)的场景级 3D 轨迹策略。落到具体机器人时,用一个轻量 warmup 把轨迹"翻译"进该机器人的动作空间;实际执行用 **inverse kinematics** 把预测的 3D 轨迹映射成关节指令(论文用最基本的 tracking controller,更复杂的策略留作未来工作)。抓取动作用一段外部 scripted grasping 提供(TraceGen 只建模放置等 trace 部分)。

## 三、实验结果

真机平台为 **Franka Research 3**,4 个任务:Clothes(叠衣)、Ball(网球入盒)、Brush(扫垃圾入簸箕)、Block(方块放到紫色区域)。基线含视频生成式(AVDC、NovaFlow 用 Wan2.2 / Veo3.1)与 trace 式(3DFlowAction、Im2Flow2Act)。

### 主结果:两种 warmup 设置

| 设置 | warmup 数据 | 整体成功率 | 备注 |
|---|---|---|---|
| Robot→Robot(同本体) | 5 条目标机器人视频 | **80%** | 所有 $<$10B 参数方法零样本均 0% |
| Human→Robot(无目标机器人数据) | 5 条未标定手机人类演示(每条 3–4 秒) | **67.5%** | From Scratch 变体 0% |

Human→Robot 分任务(Fig. 8):Clothes 70%、Ball 40%、Brush 80%、Block 80%;而 From Scratch 四项全 0%,说明跨本体预训练是 human→robot 迁移成立的必要条件。采集 20 条演示(4 任务)总耗时不到 **4 分钟**。

### 效率

TraceGen 仅 **0.67B** 参数。推理比 trace 生成基线快 $3.8\times$、比大型视频生成模型快 $50\times$ 以上;NovaFlow(Wan2.2)推理时间是其 $600\times$ 以上。摘要给出的整体区间为 **50–600×**。$<$10B 参数的方法(除 TraceGen)在零样本下均无法产出可执行轨迹(0%);大型视频生成模型能有非零零样本成功但推理极慢。

### 表 1:跨本体预训练的作用(warmup 分别 5/15 条机器人视频)

| warmup | 预训练 | Clothes | Ball | Brush | Block | 整体 SR |
|---|---|---|---|---|---|---|
| 5 条 | Random init | 10/10 | 0/10 | 0/10 | 0/10 | 25.0% |
| 5 条 | TraceGen | 10/10 | 6/10 | 8/10 | 8/10 | **80%** |
| 15 条 | Random init | 10/10 | 0/10 | 0/10 | 2/10 | 30.0% |
| 15 条 | TraceGen | 10/10 | 9/10 | 8/10 | 6/10 | **82.5%** |

要点:5→15 条 warmup 对预训练模型仅微涨(80→82.5%),对 scratch 也仅 25→30%,说明**性能主要来自预训练**,warmup 只是把已有运动先验对齐到任务配置。

### 表 2:预训练数据源的作用(统一 5 条 warmup,Ball/Block 两任务)

| 任务 | From scratch | 仅 SSV2 | 仅 Agibot | TraceForge-123K |
|---|---|---|---|---|
| Ball | 0/10 | 3/10 | 4/10 | 6/10 |
| Block | 0/10 | 2/10 | 5/10 | 8/10 |
| 整体 SR | 0% | 25% | 45% | **70%** |

仅人手数据(SSV2,35K)25%、仅机器人数据(Agibot,35K)45%,均低于全量混合训练的 70%——说明**本体对齐(robot-centric 数据)与异构运动覆盖(人+机器人)都重要,合起来最好**。

### 附录关键数据

- **轨迹抽取精度(Table 3)**:与前向运动学得到的真值末端轨迹相比,9 段遥操作 episode(平均位移 70.96 cm)上端点误差均值 $x/y/z=1.66/1.79/2.26$ cm(标准差 $0.82/1.82/2.69$),整体 sub-2.3 cm(厘米级),说明 TraceForge 监督信号可靠。
- **长时序 Sorting 任务(Table 4)**:4 个连续放置子任务,预训练模型逐步成功率 $0.8/0.8/0.8/0.8$,而 From Scratch 为 $1.0/0.8/0.5/0.4$——scratch 随步数累积误差、后段明显退化,预训练先验能稳定拼接技能。

## 四、局限性

作者在 Sec 7 自陈:

1. 只用了线性插值 + ODE 的 stochastic interpolant,尚未探索其他插值 schedule 或机制来对**歧义任务显式控制生成哪种轨迹模式**(multi-modality 控制)。
2. 演示数据质量参差:部分源视频含低效或纠错动作(探索性移动、完成前的失误),引入次优监督;虽做了过滤但仍残留噪声。
3. 零样本生成在**新本体/未见环境**下尚不完全可靠,偶尔产出看似合理但物理上不可行的轨迹。
4. 对细粒度操作,生成轨迹可能细节不足,不够精确执行。
5. 未来需扩到互联网规模数据+更强过滤,并测试超出"类人机械臂"的差异更大机器人上 trace-space 抽象的极限。

补充可见的隐性局限:执行依赖外部 scripted grasping 与 IK tracking 控制器,并未闭环学习抓取本身;真机评测仅 4 个桌面任务、每任务 10 次 rollout,统计样本偏小;绝对精度受 VGGT 深度估计误差上限约束。

## 五、评价与展望

**优点**:

- **表示选择漂亮**。把世界模型输出从像素/token 换成场景级 3D trace,同时规避了视频生成的算力浪费与幻觉、以及 2D flow 依赖检测器/静态相机的顽疾;$(x,y,z)$ 让 3D/2D 监督共享屏幕对齐,是打通异构数据的关键工程巧思。用 0.67B 小模型换来 50–600× 的推理提速,对闭环实时规划很有价值。
- **数据工程扎实**。TraceForge 的"相机补偿→世界到相机变换→速度重定标"三件套,直击 in-the-wild 视频的相机运动与执行速度差异这两个复用障碍;附录厘米级精度的 sanity check 增强了可信度。
- **迁移证据有力**。5 条手机人类视频→67.5% 真机成功、而 scratch 全 0,是"跨本体预训练必要性"较干净的对照实验;Table 2 的 source ablation 进一步把收益拆成"本体对齐+运动多样性",论证清晰。

**不足与存疑**:

- 与 3DFlowAction(2506.06199)同属 3D flow / trace 世界模型,主要差异在于抛弃 mask/检测、规模大 $15\times$、并用 stochastic interpolant 生成;但正因两者思路接近,真正的增量在于"数据规模+去检测器",方法学新意相对温和。
- 用 stochastic interpolant 却只取最简线性 schedule,把生成式框架的多模态优势基本闲置——歧义任务(多个合理落点)下的模式坍缩风险未被正面评估。
- 执行链路弱:IK+tracking 控制器 + 外部抓取脚本,使"成功率"部分依赖于非学习组件,难以判断瓶颈究竟在轨迹预测还是执行;和端到端 VLA(如 OpenVLA、RT-2)不完全可比。
- 评测规模偏小,缺少更长时序、多物体杂乱场景、以及非机械臂本体的验证。

**开放问题与可能改进方向**:

- 引入非线性 interpolant 或显式 latent 模式变量,让模型对歧义任务生成多条候选轨迹并由下游选择;
- 把 IK/tracking 换成学到的闭环 trace-conditioned 策略,或联合建模抓取,减少对脚本抓取的依赖;
- 在 trace 编码器上叠加隐式世界模型表示(作者也指出这条互补路线),或引入接触/力约束以缓解"物理不可行轨迹";
- 向真正差异大的本体(灵巧手、移动底盘、双臂)扩展,检验 trace-space 抽象是否仍成立。

## 参考

- Zhi et al. *3DFlowAction: Learning cross-embodiment manipulation from 3D flow world model.* arXiv 2506.06199, 2025.(最直接对比:同为 3D flow 世界模型)
- Li et al. *NovaFlow: Zero-shot manipulation via actionable flow from generated videos.* arXiv 2510.08568, 2025.(视频生成式主基线)
- Karamcheti et al. *Prismatic VLMs: Investigating the design space of visually-conditioned language models.* ICML, 2024.(多编码器融合与冻结编码器策略来源)
- Albergo, Boffi & Vanden-Eijnden. *Stochastic interpolants: A unifying framework for flows and diffusions.* arXiv 2303.08797, 2023.(生成目标的理论框架)
- Yang et al. *CogVideoX: Text-to-video diffusion models with an expert transformer.* ICLR, 2025.(解码器骨干)
