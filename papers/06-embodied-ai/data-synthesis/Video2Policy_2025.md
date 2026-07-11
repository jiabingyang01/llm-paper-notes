# V2P：Video2Policy：借助互联网视频规模化生成仿真操作任务

> **论文**：*Video2Policy: Scaling up Manipulation Tasks in Simulation through Internet Videos*
>
> **作者**：Weirui Ye, Fangchen Liu, Zheng Ding, Yang Gao, Oleh Rybkin, Pieter Abbeel
>
> **机构**：Tsinghua University；Shanghai Qi Zhi Institute；Shanghai Artificial Intelligence Laboratory；UC Berkeley；UC San Diego
>
> **发布时间**：2025 年 02 月（arXiv 2502.09886）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2502.09886) | [PDF](https://arxiv.org/pdf/2502.09886)
>
> **分类标签**：`数据合成` `视频转仿真` `LLM奖励生成` `Sim2Real` `强化学习`

---

## 一句话总结

Video2Policy(V2P)把互联网人类操作视频自动转成 IsaacGym 里可训练的仿真任务(场景+资产+奖励/成功函数代码),再用 RL 学出策略、蒸馏成通用视觉策略,在 9 个 SSv2 任务+3 个自采困难任务上平均成功率 **0.88**,显著超过 Code-as-Policy(0.34)、RoboGen(0.45)、Eureka(0.71)三个基线,并首次展示了随视频任务数增多(10→100)通用策略成功率持续提升(0.13→0.75)以及 sim2real 部署(72%仿真 / 47%真机)的可扩展性。

## 一、问题与动机

大规模生成机器人训练数据的两条已有路线各有硬伤:纯文本+LLM 的任务生成(如 Gensim、RoboGen)缺乏真实机器人知识落地,容易产生不有趣、不真实、物体分布单一的任务;基于真实场景扫描的 Real2Sim 数字孪生(如 Hsu et al. 2023、Torne et al. 2024)需要精细的真实-仿真对齐,难以规模化到多场景。作者提出第三条路:直接利用海量互联网 RGB 视频(以 Something-Something V2 为主)作为任务和物体分布的真实来源,自动重建仿真场景资产与 6D 位姿,再用 VLM 生成可执行任务代码(含奖励/成功函数),通过 RL 迭代求解,最终把学出的专家轨迹蒸馏为一个可 sim2real 部署的通用视觉策略,构成一条端到端的"数据引擎"。

## 二、核心方法

V2P 分两阶段:(1)从视频重建仿真场景;(2)任务代码生成+RL 求解并蒸馏为通用策略。

**阶段一:视频场景重建**。① 物体检测/分割:用 Grounding DINO 在首帧上以视频文本 caption/物体标签(SSv2 自带,自采视频需人工标注)做 grounding,再用 SAM-2 做整段视频的分割追踪。② 网格重建:用单图 3D 生成模型 InstantMesh,通常取首帧(严重遮挡时取末帧)重建物体网格。为估计真实尺度,先用 UniDepth 预测相机内参 $\mathbb{K}$ 和逐像素深度 $d_{i,j}$,把掩码区域反投影到相机坐标系 $\mathbf{p}(i,j)=\mathbb{K}^{-1}\cdot[x,y,1]^{\mathrm{T}}\cdot d_{i,j}$,取掩码内两点最大距离作为图像侧尺度

$$D_{\text{image}} = \max_{(i_1,j_1),(i_2,j_2)\in M} \Vert \mathbf{p}(i_1,j_1)-\mathbf{p}(i_2,j_2)\Vert,$$

再与网格顶点最大距离 $D_{\text{mesh}}$ 相除得缩放比 $\rho=D_{\text{image}}/D_{\text{mesh}}$,把重建网格缩放到真实尺寸(绝对尺度可能有噪声,但物体间相对尺度较准,因为二者在同一相机坐标系下计算)。③ 6D 位姿追踪:用 FoundationPose,以网格、相机内参、逐帧深度为输入,输出整段视频每物体的 6D 位姿,并自动生成对应 URDF。三步产出一个包含视频信息、物体尺寸/URDF/6D 位姿序列的任务 JSON 文件。

**阶段二:任务代码生成与强化学习**。任务代码固定拆成六部分:场景信息、reset 函数、success 函数、observation 函数、observation space 函数、reward 函数,均由 GPT-4o 一次性根据视频信息+提示生成(生成 8 份候选,GPT-4o 自查正确性/合理性/效率后选 1 份作为基础代码)。训练目标把稀疏成功奖励与稠密奖励加权组合:

$$\mathcal{R}_{\text{train}} = \mathcal{R} + \lambda\,\mathcal{R}_{0/1},\quad \lambda=100,$$

用 PPO 训练。仿照 Eureka 的 in-context reward reflection,但从零生成而非依赖人工先验成功函数:每轮采样 $N=8$ 个候选 reward 函数并行训练,收集训练/评估日志,挑最优者连同日志、CoT 示例反馈给 GPT-4o 生成下一轮候选(SSv2 任务迭代 5 轮,自采困难任务迭代 8 轮)。

**Sim2Real 蒸馏**。把 V2P 视为"数据引擎":收集学出策略的成功轨迹(RGB 图像观测+7 维动作)作为专家数据,再用行为克隆(Resnet18 backbone+3 层 MLP head)训练一个跨任务/跨物体的通用视觉策略;真机部署时用 SAM-2 分割掩码替代原始 RGB 作为策略输入以缩小视觉域差,并对动作噪声(0.02)、随机延迟(0.01–0.02s)、物体物理属性做域随机化。

## 三、关键结果

实验基于 IsaacGym,单臂桌面操作,horizon=300,8192 并行环境,每任务 3 个种子×10 条评估轨迹取均值方差。

**与基线对比**(表 1,9 个 SSv2 任务+3 个自采困难任务):

| 基线 | 平均成功率 |
|---|---|
| Code-as-Policy | 0.34 |
| RoboGen | 0.45 |
| Eureka | 0.71 |
| **Video2Policy** | **0.88** |

单物体任务上 RoboGen/Eureka 与 V2P 接近,但多物体任务(如 Uncover sth from sth: V2P 0.97 vs RoboGen 0.63)和自采困难任务(Throw Garlic into Bowl: V2P 0.70±0.36 vs Eureka 0.37±0.29,CoP/RoboGen 接近 0)上 V2P 优势明显扩大;Code-as-Policy 在需要动态/精确控制的抛掷类任务上几乎全部失败。

**通用策略泛化**(以 lifting 行为为例):在 SSv2 100 个 lifting 视频上生成场景并训练策略,收集专家轨迹训练 BC 通用策略,在 10 个含新形状/新类别物体的未见视频任务上评测:BC-V2P 平均成功率 **75%**,而基于状态输入的 Code-as-Policy 为 32%,同样用 BC 蒸馏但数据来自 CoP 的 BC-CoP 仅 26%(BC-V2P 在 10 个任务中 9 个显著优于 BC-CoP)。

**可扩展性**(图 6):训练任务数 $N$ 从 10 增至 100,通用策略成功率从 0.13 单调提升至 0.75(10→0.13,30→0.45,50→0.60,70→0.65,100→0.75),验证了"视频越多、性能越好"的数据引擎属性。

**Sim2Real**(表 2):用 100 个 lifting 任务场景各收集 200 条轨迹训练 BC 策略,在 10 个仿真新物体上达 72% 成功率,部署到 Franka 真机(Robotiq 夹爪+Stereolabs 相机,256×256 输入)后平均成功率 **47%**(Mouse 0.50、Cup 0.40、Bread 0.50);其中面包虽未在仿真训练集中出现且为软体,但因分割掩码观测+较大接触面积反而成功率不低,抓取湿滑的杯子成功率最低。

**消融**(表 3,均值):完整 V2P 0.87;去视觉信息(仅用 caption 生成代码)降至 0.75;去成功函数挑选(不再从 8 个候选中选优)降至 0.57;去迭代 reward 反射降至 0.48;去多 reward 采样(每轮只生成 1 个候选)降至 0.51 —— 说明迭代式 reward reflection 和多候选采样是最关键的两个组件。

## 四、评价与展望

**优点**:相比 Gensim/RoboGen 等纯文本驱动的任务生成,V2P 用真实视频锚定物体资产、尺度和任务分布,缓解了 LLM"凭空造任务"的幻觉问题;相比 Real2Sim 数字孪生类工作(单场景精细重建,难扩展),V2P 只需粗糙的单视角网格+6D 追踪即可支持上百个场景的并行生成,规模化成本低得多;把"视频→仿真专家轨迹→BC 通用策略→sim2real"整条链路串通并给出真机数字,是同类 LLM-driven RL 任务生成工作(Eureka、RoboGen)少有的完整闭环验证。

**局限**:①管线强依赖上游基础模型质量——单图网格重建(InstantMesh)、深度/内参估计(UniDepth)、位姿追踪(FoundationPose)任一环节出错都会传导为不真实的仿真物理,作者在结论中明确承认这是当前瓶颈,绝对尺度估计存在噪声;②SAM-2 分割在多数 SSv2 视频严重遮挡场景下需要人工挑选首/末帧兜底,自动化程度打了折扣;③真机验证只做了单一 lifting 行为、3 类物体、10cm 位置扰动范围,尚未验证抛掷等动态行为、更复杂多物体任务的 sim2real 效果;④72%→47% 的 sim2real 成功率跌幅(约 35 个百分点)表明视觉/物理 domain gap 仍未被分割掩码策略完全消除;⑤仅覆盖单臂桌面刚体/软体物体操作,未涉及双臂、可变形长时程任务。

**与其他工作的关系及开放问题**:V2P 与 Eureka 的核心区别是奖励/成功函数与场景任务代码一起"从零生成"而非依赖人工预设的成功函数起点,消融显示这一设计(迭代 reflection+多候选采样)贡献最大;与 RoboGen 相比,V2P 用真实视频视觉先验替代纯 LLM 想象的物体关系,在多物体/动态任务上优势明显放大。开放方向包括:用更强的单图/视频 3D 重建与位姿估计模型进一步收窄仿真物理与真实的差距;将该数据引擎扩展到双臂、可变形物体和更长时程任务;探索用生成的大规模多任务数据直接预训练可迁移的视觉运动基础策略,而不仅是单一行为(lifting)内的泛化。

## 参考

- Wang et al. RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation. 2023.
- Ma et al. Eureka: Human-Level Reward Design via Coding Large Language Models. 2023.
- Liang et al. Code as Policies: Language Model Programs for Embodied Control. ICRA 2023.
- Hsu et al. Ditto in the House: Building Articulation Models of Indoor Scenes through Interactive Perception. ICRA 2023.
- Torne et al. Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation. 2024.
