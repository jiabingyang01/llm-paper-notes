# ActionToken Survey：VLA 模型综述——基于动作分词(Action Token)的视角

> **论文**：*A Survey on Vision-Language-Action Models: An Action Tokenization Perspective*
>
> **作者**：Yifan Zhong*, Fengshuo Bai*, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, Zhiquan Qi, Yitao Liang, Yuanpei Chen†, Yaodong Yang† et al.（*equal contribution，†corresponding author）
>
> **机构**：Institute for AI, Peking University；PKU-PsiBot Joint Lab；School of Computer Science, Peking University
>
> **发布时间**：2025 年 07 月（arXiv 2507.01925）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.01925) | [PDF](https://arxiv.org/pdf/2507.01925)
>
> **分类标签**：`综述` `VLA` `动作分词` `具身智能` `Vision-Language-Action`

---

## 一句话总结

本文提出一个统一框架——视觉语言输入被一串 VLA 模块逐步处理成越来越"可执行"的**动作令牌（action token）**链条——并据此把现有 VLA 研究归纳为语言描述、代码、affordance、轨迹、目标状态、隐变量表征、原始动作、推理八大类动作令牌，逐类梳理代表工作、优劣势与数据来源，最终给出面向"分层架构 + agent 化 + 模仿学习转强化学习 + 数据规模化"的未来路线判断。

## 一、问题与动机

近年 VLA（Vision-Language-Action）模型数量爆炸式增长（论文引用超过 300 篇工作，覆盖 SayCan、PaLM-E、Code as Policies、RT 系列、OpenVLA、π0、GR00T N1 等），但架构设计五花八门：有的输出自然语言计划，有的直接生成可执行代码，有的预测关键点/边界框/分割掩码，有的预测未来目标图像，有的直接回归末端执行器位姿。作者观察到一个共性：几乎所有 VLA 都能被抽象为"视觉语言输入 → 一串 **VLA 模块**（支持端到端梯度回传的最大可微子网络，或运动规划等不可微功能单元）→ 逐步产出**动作令牌**链条 → 最终可执行动作"。**动作令牌是 LLM 中"language token"概念在具身场景下的推广**——真正区分不同 VLA 设计的核心维度，是动作令牌"长什么样"，而不是表面上的骨干网络差异。基于这一观察，本文以动作分词为主线，系统梳理 VLA 研究版图，弥补该领域此前缺乏统一分类视角的空白。

## 二、核心方法（八类动作令牌分类体系）

论文将现有 VLA 研究中的动作令牌归纳为以下八类（Table 1 汇总各类优势/局限/代表成果）：

**(1) 语言描述（Language Description）**。分为高层的 **language plan**（如"pick up the cup"，描述整个子任务）与细粒度的 **language motion**（如"move arm forward"，接近电机指令）。代表工作：SayCan 用 affordance function 对 LLM 生成的候选计划重加权以保证可行性；Inner Monologue 引入闭环反馈（成功检测、场景描述）实现多轮反思式规划；Hi Robot 与 π0.5 用高层 VLM 规划器 + 低层生成式控制策略处理自由格式指令，实现长时程家务任务；RT-H 引入 language motion 中间层，让语义差异很大但底层动作模式相似的任务（如"擦桌子"与"推滑块"）能共享数据。

**(2) 代码（Code）**。代码天然携带条件、循环等控制结构，能直接对接第三方感知/控制 API 库。Code as Policies 用 GPT-3/Codex 生成 Python 代码调用感知控制 API 并可用 NumPy 做空间推理；ProgPrompt 用有限状态机组织代码生成与执行状态断言。核心局限是 API 覆盖不到的物理属性（如"湿滑表面""易碎物体"）会让语法正确的代码在物理层面执行失败，即 symbol grounding 问题。

**(3) 可供性（Affordance）**。按精度递增分为 keypoint（$\mathbf{k}=[\mathbf{x},\mathbf{d}]$，$\mathbf{x},\mathbf{d}\in\mathbb{R}^3$ 分别表示接触位置与作用方向，用大白话说就是"在哪儿摸、往哪儿使劲"）、bounding box、segmentation mask、affordance map（$\mathbf{A}\in\mathbb{R}^{H\times W}$，逐像素打交互适宜度分数）。代表：ReKep 把操作建模为跟踪关键点上的约束优化问题；VoxPoser 用 LLM+VLM 在感知空间合成 3D value map 实现零样本轨迹生成；DexGraspVLA 用 bounding box 结合分割在杂乱场景中做灵巧抓取。

**(4) 轨迹（Trajectory）**。三种粒度：point trajectory（$\mathbf{P}\in\mathbb{R}^{T\times K\times 2}$，$T$ 步内 $K$ 个关键点的路径）、visual trajectory（把路径直接画进图像/视频，$\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$）、optical flow（$\mathbf{V}\in\mathbb{R}^{H\times W\times 2}$，逐像素运动场）。代表：RT-Trajectory 用 2D/2.5D 末端执行器轨迹条件化 RT-1 式策略，在未见任务上超过 RT-1、RT-2 及 RT-1-Goal；ATM、Im2Flow2Act 只需极少（甚至零）真实机器人动作标注，从人类视频学到轨迹再迁移到低层策略，天然具有跨任务泛化能力。

**(5) 目标状态（Goal State）**。单帧图像/RGB-D/点云或多帧视频形式的预测未来观测，作为"先想象再执行"的中间目标。SuSIE、CoT-VLA、VPP 用扩散模型生成 goal image/video 再由低层策略解码为动作；其数据可通过 hindsight relabeling 从原始轨迹自动抽取末帧，无需人工标注，天然具备规模化能力。

**(6) 隐变量表征（Latent Representation）**。在无动作标注的大规模数据（人类视频、跨 embodiment 机器人数据）上以 VQ-VAE/FSQ/NSVQ 等方式无监督构造隐动作码本，三阶段流水线为 latent construction → latent pretraining（VLM 预测隐动作）→ action fine-tuning（解码为目标本体的低层动作）。代表：Genie 从游戏视频学到可控隐动作；LAPA 把该思路搬到机器人操作，其跨 embodiment 隐动作预训练效果反而**优于**用真实动作标签预训练；UniVLA 先用 DINOv2 语义特征代替原始像素、再做任务中心/任务无关信息解耦，仅用 OpenVLA **4.45%** 的训练时间即达到可比性能。

**(7) 原始动作（Raw Action）**。VLA 模块直接输出末端执行器/关节空间可执行动作，分自回归离散化路线（RT-1、RT-2、OpenVLA）与扩散/流匹配动作分块（chunking）路线（Octo、π0、RDT、GR00T N1）。π0 用 flow matching 支持高达 **50 Hz** 控制频率，比 RT-2 的 5 Hz 高一个数量级；π0-FAST 用离散余弦变换（DCT）编码动作块，token 压缩最高达 **13.2×**；Real-Time Chunking 用流匹配 inpainting + 软掩码解决相邻动作块衔接处的不连续与"卡顿"问题。

**(8) 推理（Reasoning）**。以自然语言显式外化决策过程，作为其他动作令牌生成前的"元令牌"（meta-token）。ECoT 在 OpenVLA 上引入固定结构的 CoT（任务分解 → 抓取点 → 物体框），异步执行可将推理引入的额外延迟抵消约 40%；RAD 进一步用人类视频（借助 HaMeR 手部关键点提取）扩充推理数据来源；DriveVLM 把 CoT 拆成场景描述、场景分析、分层规划三模块用于自动驾驶。

## 三、关键结果

综述本身不做新实验，其"结果"体现为对各类动作令牌代表性成果与关键数字的横向汇总（源自正文 Table 1 及各节数字）：

| 动作令牌类型 | 代表工作 / 任务 | 关键数字 |
|---|---|---|
| 语言计划 | π0.5 整理床铺；Hi Robot 做三明治、超市采购 | π0.5 可零样本清理未见过的厨房 |
| 语言动作 | RT-H 从分发器抽纸巾 | 细粒度 language motion 实现跨任务数据共享 |
| 代码 | Instruct2Act 归位重排 | 依赖预定义 API，越界场景易失败 |
| 可供性 | ReKep 倒茶；DexGraspVLA 杂乱场景灵巧抓取 | keypoint 在遮挡下精度显著下降 |
| 轨迹 | RT-Trajectory 用鸡毛掸子擦桌子 | 超越 RT-1/RT-2/RT-1-Goal 的未见任务泛化 |
| 目标状态 | VPP 用移液管转移液体 | AVDC 生成 8 帧视频目标约耗时 10 秒；Gen2Act 推理仅 3 Hz；VPP 单步去噪后可达 7–10 Hz |
| 隐变量表征 | GO-1 叠衣服；OmniJARVIS Minecraft 挖钻石 | UniVLA 仅用 OpenVLA 4.45% 训练时长即达可比性能 |
| 原始动作 | π0 叠洗衣物、点蜡烛（Real-Time Chunking） | π0 控制频率 50 Hz（RT-2 为 5 Hz）；π0-FAST token 压缩最高 13.2× |
| 推理 | DriveVLM 自动驾驶决策 | ECoT 异步执行抵消约 40% 推理延迟 |

数据规模方面（Section 12）：RT-1 数据集含 13 万条演示、覆盖 700+ 任务；OXE 整合超百万条轨迹、22 种机器人、60 个数据集；但作者估算 **OXE 数据集的 token 总量仅约为大规模语言模型语料的 1/200,000**，凸显机器人数据仍严重稀缺。

## 四、评价与展望

**优点**：（1）首次以"动作令牌"这一统一坐标系横向比较八类差异巨大的 VLA 设计，覆盖 300+ 篇代表性工作与 10 张详尽对照表（Table 1–10），信息密度高，具备工具书式的检索价值；（2）提出的未来分层架构——顶层用语言计划/代码做长时程规划、中层用 3D affordance + 轨迹 + 目标视频做中间运动表征、底层用低层策略映射到原始动作——给出了具体、可操作的路线图，而非空泛的展望；（3）对隐变量表征持审慎态度，明确指出其在粒度、语义完备性、与人类意图对齐三方面尚未成熟，因而未将其纳入推荐的近期主线架构，体现了不盲目乐观的判断力；（4）Figure 3 的语言/视觉/多模态基础模型与 VLA、数据源演化时间线（"U 形"布局）本身即是一份可复用的领域地图。

**局限与开放问题**：（1）综述本质上是文献分类整理，不含作者自建的新实验或新基准，部分类别边界（如 language motion 与 raw action 的区分、latent representation 与 goal state 的区分）在具体工作上存在一定主观划分空间；（2）对强化学习、agent 化、安全对齐等趋势的讨论停留在方向性建议层面，例如未给出如何在真实机器人上切实降低 reset cost、提高交互效率的具体技术方案；（3）正如作者自陈，当前绝大多数 VLA 评测仍局限于简化的实验室夹爪操作场景，"远未达到通用具身智能体的要求"，但综述本身也未能提出跳出该局限的评测协议或标准化 benchmark 建议；（4）数据规模化部分对模态覆盖不足（触觉、听觉、嗅觉、味觉）、embodiment 数据割裂等问题的诊断准确，但给出的解决方向（更好的采集设备、更多仿真数据）仍偏愿景性，未提出超越 GR00T N1 式"数据金字塔"的新范式。

**与其他公开工作的关系**：相较于此前偏重"通用策略"或"世界模型驱动机器人学习"视角的 VLA 综述，本文以"动作分词"作为统一分类轴，类比 LLM 综述中以 tokenization 切入的思路，视角较新且贴合当前 VLA 架构表面多样、内核趋同的现状；其对 π0、OpenVLA、RT 系列、Code as Policies、VoxPoser 等公开工作的归类与横向对比，也为后续研究者快速定位自身工作在领域坐标系中的位置提供了便利。

**开放问题**：隐变量表征与显式表征（轨迹/affordance）之间在可解释性与可扩展性上的权衡尚无定论；作者主张推理应从基于语言 token 转向基于动作 token（action-token-based reasoning），但目前主流推理型 VLA（ECoT、RAD、DriveVLM）仍以语言 CoT 为主，这一范式转型路径能否成立仍待验证；模仿学习向强化学习的转型面临真实世界 reset 成本高、交互效率低的根本性障碍，如何在物理机器人上切实落地仍是开放课题。

## 参考

- Ichter et al. *SayCan: Do As I Can, Not As I Say*. CoRL 2022.
- Liang et al. *Code as Policies*. ICRA 2023.
- Brohan et al. *RT-2*. 2023.
- Kim et al. *OpenVLA*. CoRL 2024.
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024.
