# Scaling Up & Distilling Down：语言引导的机器人技能获取

> **论文**：*Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition*
>
> **作者**：Huy Ha, Pete Florence, Shuran Song
>
> **机构**：Columbia University;Google DeepMind
>
> **发布时间**：2023 年 07 月（arXiv 2307.14535,v2 于 2023 年 10 月）
>
> **发表状态**：CoRL 2023（7th Conference on Robot Learning, Atlanta, USA）
>
> 🔗 [arXiv](https://arxiv.org/abs/2307.14535) | [PDF](https://arxiv.org/pdf/2307.14535)
>
> **分类标签**：`LLM引导数据生成` `语言条件策略` `diffusion-policy` `机器人技能蒸馏` `sim2real`

---

## 一句话总结

用 LLM 作为"次优的数据采集策略"——它把语言任务递归分解成 task tree、grounding 到一套 6DoF 采样式机器人原语（motion planner / grasp sampler）来自主探索、并自动生成 success 判据的代码片段做 verify & retry——从而无需人类示范或奖励工程就规模化产出带语言标签和成败标签的机器人数据;再用 success 过滤把这些数据蒸馏进一个多任务语言条件 diffusion policy,在 5 个域 18 个任务上蒸馏策略比其数据采集策略平均绝对成功率再提升 **33.2%**,并零微调 sim2real 迁移到真机迁移任务上取得 **76%** 成功率。

## 一、问题与动机

如何可规模化地获取鲁棒、可复用的真实世界操作技能?领域内两条主线各有短板:

- **规模化采集数据**:主流范式靠人类遥操作/标注示范(如 RT-1、BC-Z、Perceiver-Actor),数据量受人力上限约束;想自主扩展则要么靠随机/RL 探索(长时序稀疏奖励下效率极低),要么靠人类第一视角视频(缺动作标签、需跨具身迁移),要么靠经典 TAMP(需领域特定的转移函数工程)。
- **从数据中有效学习**:多任务语言条件模仿学习通常是从专家示范做行为克隆,而如何表征动作、如何设计策略架构以支持高精度多模态行为仍是开放问题。

本文的核心立场:**LLM 的 common-sense 与 zero-shot 规划能力很有用,但语言本身不是鲁棒精细操作的理想表征**。因此不把 LLM 当作最终策略(如 Code-as-Policy、SayCan 那样直接执行 LLM 生成的代码/计划),而是把 LLM 当作一个"次优的数据采集策略",只把它探索出来的**成功轨迹**蒸馏进一个观测信息(observable-information)的闭环视觉运动策略,使蒸馏策略能**超越**其 LLM 采集策略的性能。整套框架不用任何人类示范,也不用手工指定奖励。

## 二、核心方法

框架分两阶段:**Scale Up**(语言引导数据生成)与 **Distill Down**(语言条件策略蒸馏)。四条设计观察打底:RL 探索重要但对长时序稀疏奖励太低效;LLM 有用但语言不是好的动作表征;受 TAMP 启发但要免领域工程、推理时不依赖真值状态;蒸馏阶段要简单有效、避开昂贵人类数据。

### 2.1 数据采集策略 = LLM + 6DoF 探索原语

采集策略是一个能调用一套 **6DoF Exploration Primitives**(采样式机器人工具:motion planner、几何式 grasp/placement sampler、面向 articulated 物体的 motion primitives)的 LLM(实验用 GPT-3 `text-davinci-003`,temperature 0.0)。给定任务描述,它依次做三步:

**(1) Simplify——任务规划与分解(§3.1)**。递归地把任务分解为层级化 task tree。LLM 先判断任务是涉及多物体(需分解)还是单物体(基础情形);多物体则拆成子任务并递归下探,单物体则问 LLM 该与哪个 object part 交互。这样得到一棵探索计划树:

$$\mathcal{T} = \text{LLM-decompose}(\text{task desc},\ \text{sim state})$$

> 用大白话说:像人做事一样先"想清楚步骤"。"寄快递"要先开信箱、放包裹、关信箱、再升起邮筒小旗;LLM 把这层常识拆解出来,而且是递归拆——大任务拆成小任务,小任务再判断是否要继续拆,直到叶子是"抓某个物体/操作某个关节"这种可执行的原子动作。

**(2) Ground——把计划编译成机器人原语(§3.2)**。对 task tree 的每个叶节点,用其任务描述 + object part 名 + 仿真状态,解析物体的运动学结构(区分刚体/articulated),输出一串 6DoF 原语 API 调用:非 articulated 物体让 LLM 选 pick/place 的 object part,从 object part 点云上**均匀采样** grasp 与 placement 候选,再喂给在关节空间均匀采样的 RRT motion planner;articulated 物体则采样 grasp 后接一个由关节参数(关节类型、轴、原点)条件化的旋转/平移原语。执行按**先序遍历**,并用当前子任务的描述给这一段轨迹打上**稠密自动文本标签**(同一段动作会同时挂上子任务描述和根任务描述)。

> 用大白话说:计划是"文字",要变成"机器人真能做的动作"。这里的关键是动作空间的选择——用采样式(伪随机)的抓取/放置/运动规划工具,天然能生成**多样**的 6DoF 轨迹,既比平面 pick-and-place 灵活(能开抽屉这类 prismatic/revolute 关节),又不需要人类示范。因为是采样,每次结果都不同,这为后面"重试"提供了多样性来源。

**(3) Verify & Retry——把数据采集策略鲁棒化(§3.3)**。规划和 grounding 都可能失败(某个 grasp 让所有 placement 候选不可行)。LLM 为每个任务再**推断一个 success function 代码片段**(给定任务描述、仿真状态、可查询仿真的 API),输出布尔值判成败(如"升起邮筒小旗"→检查旗铰链是否抬起)。轨迹被判失败时,**不重置仿真状态**、只换随机种子重跑同一串采样式原语,直到成功或超时;树遍历中某节点只有 success 判据返回真才回溯到父任务。最终 replay buffer 里唯一的失败轨迹是超时或进入无效状态(物体掉地)的。

> 用大白话说:LLM 自己既当"操作员"又当"裁判"——写一小段判成功的代码来自动打标签,失败就换个随机种子再试(相当于换个抓法/放法),死磕到成功。这一步不仅提高了采集成功率,更重要的是**在数据里留下了"如何从失败中恢复"的行为**,这个 retry trait 会被下游策略学到。

数据生成阶段产出一个 replay buffer $\mathcal{D}=\{(\mathbf{o},\ell,\mathbf{a},s)\}$:观测 $\mathbf{o}$、语言标签 $\ell$、动作 $\mathbf{a}$、成败标签 $s$。

### 2.2 蒸馏 = success 过滤 + 语言条件 diffusion policy(§3.4）

把 Diffusion Policy(Chi et al. 2023,原为单任务行为克隆的 SOTA)**扩展到多任务语言条件闭环控制**。策略输入:CLIP 编码的任务描述 $\ell$、proprioception 历史、两路 RGB 观测(一个 wrist-mounted 视角 + 一个 global workspace 视角,各用一个 ResNet18 编码器),输出一段末端执行器控制指令(gripper poses + close command)。动作生成沿用条件 DDPM 的去噪迭代,只是把语言 $\ell$ 加入条件:

$$\mathbf{A}_t^{k-1} = \alpha\!\left(\mathbf{A}_t^{k} - \gamma\,\epsilon_\theta(\mathbf{O}_t,\ \ell,\ \mathbf{A}_t^{k},\ k)\right) + \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

训练只用 success 过滤后的成功轨迹(可视为最简单形式的 offline RL / success-filtered offline RL):

$$\min_\theta \ \mathbb{E}_{(\mathbf{o},\ell,\mathbf{a},s)\sim\mathcal{D},\ s=1}\ \big\|\epsilon - \epsilon_\theta(\mathbf{o},\ell,\mathbf{a}+\epsilon,\ k)\big\|^2$$

> 用大白话说:diffusion 天生擅长建模"多模态"分布,正好用来吸收采集策略里那些五花八门、高熵但又要精确的 6DoF 轨迹(同一句指令、不同种子会给出不同颜色的成功轨迹)。而"只学成功轨迹"这一过滤,让蒸馏策略继承鲁棒重试行为的同时,把成功率**推高到超过**采集策略——"Robustness In, Robustness Out"。

工程细节:动作 10 维(3 位置 + 旋转矩阵上两行的 6 维 + 1 个 gripper close);推理只用最新一帧视觉 + 完整 proprioception 历史;配合 DDIM,推理 5 步 / 训练 50 步(10× 更短),在 RTX 3080 上以 $\approx 35$ Hz 闭环运行。

### 2.3 多任务 benchmark

基于 MuJoCo、用 Google Scanned Objects 资产、桌面 6DoF 机械臂,构建 5 个域共 18 个任务,覆盖既有 benchmark 缺乏的能力:

| 域 | 复杂几何 | Articulation | Common-sense | Tool-use | 多任务 | 长时序 | 代表任务 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Balance | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | balance the bus on the block(直觉物理) |
| Catapult | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | 用弹射臂把方块射进指定 bin(3 任务) |
| Transport | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 把玩具搬进左/右 bin(几何泛化到新物体) |
| Mailbox | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | send the package for return(≈800 控制周期,≥4 子任务) |
| Drawer | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | 开抽屉放物再关(12 任务,≈300 周期) |

评测用**人工设计**的 success 函数(而非 LLM 生成的那个),每任务 200 episode 取平均。

## 三、实验结果

### 数据采集 & 蒸馏策略成功率(Table 2,200 trials)

Planar = Balance/Catapult/Transport,6DoF = Mailbox/Drawer。上半为数据采集策略(baseline 以 Code-as-Policy 为基),下半为蒸馏策略。

| 方法 | Balance | Catapult | Transport | Mailbox | Drawer | Average |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM-as-Policy (2D) | 28.0 | 33.3 | 21.5 | 0.0 | 0.0 | 27.6 |
| (+) 6DoF Robot Utils | 5.5 | 2.5 | 35.0 | 0.0 | 1.3 | 8.8 |
| (+) Verify & Retry | 45.0 | 2.5 | 82.0 | 3.0 | 31.8 | 33.8 |
| Distill No-Retry | 67.5 | 38.5 | 32.5 | 0.0 | 22.7 | 32.2 |
| **Distill Ours** | **79.0** | **58.3** | **80.0** | **62.0** | **55.8** | **67.0** |

关键读数:

- **6DoF 探索是刚需**:平面动作在 Mailbox/Drawer 上直接 0.0;引入 6DoF 原语后在 Transport/Drawer 这类需复杂几何抓取和 articulated 操作上才有非零成功率(代价是简单任务短期掉点,靠后续蒸馏补回)。
- **Verify & Retry 恒有增益**:在所有域都提升,Transport/Catapult/Balance/Drawer 上带来 2×/3×/8×/… 的提升;没有它,Mailbox 域 0.0% 成功,凸显无瑕执行长串 6DoF 动作之难与"失败恢复"的重要性。
- **蒸馏超越采集策略(核心卖点)**:Distill Ours 平均 67.0 vs 其采集策略 (+) Verify & Retry 的 33.8,**绝对提升 +33.2%**(即摘要所述)。Distill No-Retry 平均仅 32.2,比 Ours 低 **34.8%**——因为它的采集策略不重试,学出来的策略脆弱、不会恢复。

### 误差来源与传播(Table 3,Mailbox 域,planning / verify / execution 准确率 %)

| 子任务 | Planning | Verify | Execution |
|---|:---:|:---:|:---:|
| Open mailbox | 100 | 100 | 43.5 |
| Put package in mailbox | 100 | 100 | 28.5 |
| Raise mailbox flag | 100 | 100 | 62.0 |
| Close mailbox | 100 | 100 | 94.2 |

规划与 success 验证在 Mailbox 域全对(GPT-3),瓶颈在底层**执行**——这正是 verify & retry 反复重试所要克服的。

### LLM 规模消融(Table 4,planning / success 推断准确率 %)

| 模型 | 参数量 | Planning | Success |
|---|:---:|:---:|:---:|
| LLAMA2 | 7B | 42.0 | 10.0 |
| LLAMA2 | 13B | 62.0 | 48.3 |
| GPT-3 | 175B | 82.0 | 91.1 |

小模型(尤其 7B)难以遵循 prompt(如 drawer 域漏掉开/关抽屉);13B 比 7B 在 planning/success 上各 +20.0% / +38.3%,呈随规模上升趋势,而 175B GPT-3 明显领先——数据生成质量对 LLM 能力敏感。

### 策略学习消融(Table 5,Balance 单任务域,success %)

| 架构 | Action Rep. | Exec | Pred | Pool | Proprio | Success |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BC-Z (FeedForward) | Delta | 1 | 10 | Avg | ✗ | 0.0 |
| BC-Z (FeedForward) | Delta | 8 | 10 | Avg | ✗ | 18.5 |
| Ours (FeedForward) | Abs | 8 | 16 | Spatial | ✓ | 35.5 |
| Ours (Diffusion) | Delta | 8 | 16 | Spatial | ✓ | 69.5 |
| Ours (Diffusion) | Abs | 8 | 16 | Avg | ✓ | 76.5 |
| **Ours (Diffusion)** | **Abs** | **8** | **16** | **Spatial** | ✓ | **79.0** |

要点:diffusion 动作解码器一致优于确定性 MLP;spatial softmax 比 mean pool +5.0%;delta→absolute 动作空间在 MLP/diffusion 上分别 +6.5% / +9.5%。

### 鲁棒重试与 sim2real

- **Fig 6(Balance)**:蒸馏策略随时间单调爬升到 79.0%,其采集策略 (6DoF+Retry) 停在 45.0%、No-Retry 蒸馏停在 67.5%、纯 2D/6DoF LLM 策略只有 28.0%/5.5%——蒸馏策略继承了"每次失败后换一组多样的抓取/放置继续尝试"的行为。
- **Sim2Real**:在域随机化的合成数据上训练,零微调迁移到真实 Transport 任务(5 个新物体、10 episode),平均 **76%** 成功。

## 四、局限性

- **依赖特权仿真状态**:LLM 推断 success 判据用到了真值 contact、关节信息、物体位姿,因此数据生成阶段**只能在仿真里做**,真机部署要靠 sim2real 迁移(本文用域随机化)。
- **依赖现成 3D 资产/环境**:数据集多样性受限于已有资产,作者指出可用 3D 生成模型或程序化生成来扩展(但本文未做)。
- **只学了 root task**:虽然数据集给所有子任务都打了文本+成败标签,但策略只在根任务上评测过;从所有子任务学习、随时间积累可复用 sub-skill 以实现组合泛化被留作 future work。
- **benchmark 全为仿真、单臂桌面**:18 任务/5 域虽有针对性(长时序/工具/直觉物理),但规模与真实场景多样性仍有限;真机只验证了单个 Transport 任务。
- **采集成本**:每条成功轨迹要多次重跑采样式原语 + 多次 LLM 调用(planner 拆成多模块以省 token 并可缓存),长时序任务(Mailbox 底层执行准确率仅 28–62%)的重试开销不小。

## 五、评价与展望

**优点**。(1) 观念清晰且有说服力——"把 LLM 当次优数据采集策略而非最终策略",既借用其常识规划,又用 diffusion 蒸馏绕开语言不适合精细控制的短板,并通过 success 过滤实现"蒸馏策略反超采集策略"(+33.2%),这一"Robustness In, Robustness Out"的经验证据是本文最有价值的贡献。(2) verify & retry 用 LLM 自生成 success 代码来自动标注+驱动重试,把"失败恢复"这一稀缺行为写进了数据,而非事后靠策略架构硬学,思路优雅。(3) 6DoF 采样式原语 + 多样种子重试,天然产生高熵多模态数据,与 diffusion 的多模态建模能力形成协同,消融(Table 5)把这条因果链拆解得很清楚。

**局限与争议**。(1) 数据生成强依赖特权仿真状态与人工搭建的仿真环境/资产,离"真正自主、真机可扩展"的数据飞轮还有距离,是与后续大规模真机遥操作路线(RT-1/RT-2、Open X-Embodiment)相比的根本短板。(2) LLM 规模消融显示对 GPT-3 级别能力高度依赖,7B/13B 开源模型 success 判据准确率骤降,复现门槛和成本不低。(3) 只评测 root task,声称的"可复用技能库/组合泛化"停留在愿景。

**与其他公开工作的关系**。与 Code-as-Policy / SayCan / Inner Monologue 等把 LLM 作为最终执行者的路线正好互补——本文明确把 LLM 降格为"采集器"。与 GenSim、RoboGen、以及后续大量"LLM + 仿真自动生成任务/数据"的工作同属一条"用基础模型自动化机器人数据引擎"的主线,本文是较早把"LLM 规划 + 采样式 TAMP 风格原语 + success 自动标注 + diffusion 蒸馏"完整串起来的代表作之一。策略侧则是把单任务 Diffusion Policy(Chi et al.)扩展到多任务语言条件的早期尝试,与 Octo、多任务 diffusion policy 等一脉相承。

**开放问题与可能改进方向**。(1) 去特权化:用可从观测直接判定的 success 检测(如 VLM 判成败)替代真值状态查询,以把数据生成搬到真机或更真实的传感。(2) 资产/环境的程序化与生成式扩展(3D 生成 + 域随机化),提高数据多样性与 sim2real 覆盖。(3) 真正利用子任务标签构建可复用、可组合的 skill library,评测组合泛化。(4) 把 verify & retry 从"换随机种子重跑"升级为带记忆的定向探索(利用失败反馈引导下一次采样),提升长时序任务的采集效率。(5) 用更强/更省的开源 LLM + prompt 缓存降低数据引擎成本。

## 参考

1. C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, S. Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023.(被本文扩展为多任务语言条件版）
2. J. Liang et al. *Code as Policies: Language Model Programs for Embodied Control.* ICRA 2023.(本文数据采集 baseline 的基础,LLM 作为最终策略的对照）
3. E. Jang et al. *BC-Z: Zero-shot Task Generalization with Robotic Imitation Learning.* CoRL 2022.(策略学习 baseline;FiLM/多任务条件对照）
4. C. R. Garrett et al. *Integrated Task and Motion Planning (TAMP).* Annual Review of Control, Robotics, and Autonomous Systems, 2021.(本文以 LLM 免工程地扩展 TAMP 思路）
5. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models (DDPM).* NeurIPS 2020.(动作扩散去噪的理论基础）
