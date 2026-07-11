# CASHER：超线性扩展的机器人学习

> **论文**：*Robot Learning with Super-Linear Scaling*
>
> **作者**：Marcel Torne, Arhan Jain, Jiayi Yuan, Vidaaranya Macha, Lars Ankile, Anthony Simeonov, Pulkit Agrawal, Abhishek Gupta（前四位为共同一作）
>
> **机构**：Massachusetts Institute of Technology；University of Washington；Stanford University
>
> **发布时间**：2024 年 12 月（arXiv 2412.01770，v3 为 2025 年 10 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2412.01770) | [PDF](https://arxiv.org/pdf/2412.01770)
>
> **分类标签**：`real-to-sim-to-real` `众包数字孪生` `RL数据飞轮` `超线性扩展` `generalist policy`

---

## 一句话总结

CASHER（Crowdsourcing and Amortizing Human Effort for Real-to-Sim-to-Real）用手机 3D 扫描众包海量真实场景的数字孪生，在仿真里用"人类示范 bootstrap 的 RL"生成操作数据,再让多环境 generalist policy 的泛化能力逐步顶替人类示范——形成数据飞轮,使性能相对人工投入呈**超线性** 扩展;真机 pick-and-place-to-sink 任务上,训练环境从 9 增到 56 时 zero-shot 成功率从约 16% 升到 62%,且新场景只需一段视频扫描即可无额外人工地微调涨点约 55%。

## 一、问题与动机

机器人学习的核心瓶颈是数据。与视觉/语言不同,机器人**没有天然可"被动采集"的海量数据**:遥操作采数据的成本随人力线性增长(每个新场景都要人在场、要专家、要物理机器人),而程序化生成的仿真场景又与真实世界的分布(自然的杂乱、布局、光照)对不上。

作者要回答:**能否找到一种"性能随人工投入超线性增长"的数据来源?** 他们的赌注是 real-to-sim-to-real:

- **内容扩展(content scaling)**:把"造场景"这件事从设计师转移给非专家 + 廉价工具——众包手机扫描建数字孪生,天然覆盖真实分布;
- **行为扩展(behavior scaling)**:随着训练跨越越来越多环境,generalist policy 会积累"非平凡泛化",可以直接替人去新环境里生成示范数据。

关键 insight:**模型的泛化能力可以持续替换人类示范**,于是"数据孕育更多数据",人工投入被摊销(amortize),越往后每个新环境需要的人力越少。

## 二、核心方法

CASHER 三大组件:(1) 用 3D 重建快速造数字孪生;(2) 多环境模型学习,靠自主数据采集 + 模型泛化摊销数据采集;(3) 用 3D 扫描 + 极少示范在新环境高效微调。整条流水线是 **real → sim → real** 四步:众包真实扫描 → 仿真采示范 → 从示范做 RL 微调 → 蒸馏成视觉 generalist policy。

### 1. Real-to-Sim 场景合成

用手机端商用软件(Polycam、ARCode)对真实厨房做照片/激光扫描,经 Gaussian Splatting、NeRF 等 photogrammetry 方法在 5 分钟内建出高保真 3D mesh,导入 Isaac Sim。再用一个 GUI 做场景 articulation 与 curation:切分 mesh、标注可动关节、放入目标物体(碗/杯/盒)和目标位点(水槽/柜子),导出 USD。数字孪生只提供**几何/外观/物理**,不含任何示范。众包海报(Fig 12)发向全球非专家用户,贡献者分布覆盖美国、印度等地。

用大白话说:让全世界的人用手机随手扫自己家厨房,免费给你造出成千上万个"真实感的仿真练功房",但里面还没有"怎么干活"的答案。

### 2. Amortized 数据采集(核心)

把全部环境切成大小为 $K$ 的批次。**第一批** $K$ 个环境里,多环境视觉策略 $\pi_G$ 随机初始化、无泛化,只能靠**人类示范 bootstrap 的 RL** 起步:在每个环境用少量人类示范引导 RL,产出该环境的最优 visuomotor 轨迹 $\mathcal{D}$,再蒸馏成单个基于感知(RGB 点云 $o_t$)的 generalist 策略 $\pi_G$。

**关键递推**:一旦 $\pi_G$ 在**下一批** $\mathcal{E}_{K+1},\dots,\mathcal{E}_{2K}$ 上展现出非平凡泛化,就直接部署 $\pi_G(a_t\mid o_t)$ 去这些新环境自主 rollout,过滤出成功轨迹 $\mathcal{T}$——**用模型替代了人类示范者**。这些轨迹同时记录了视觉观测 $o_t$、动作 $a_t$ 和低维的特权 Lagrangian 状态 $s_t$。

拿到特权状态轨迹后,用"示范 bootstrap 的 RL"(PPO + BC 损失)在特权状态空间训练一个 state-based 策略 $\pi_{s}$:

$$
\pi_s \leftarrow \max_{\theta,\phi}\; \alpha \sum_{\mathcal{E}_i}\sum_{(s_t,a_t,r_t)} \min\!\Big(\tfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t,\; \mathrm{clip}\big(\tfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},1-\epsilon,1+\epsilon\big)\hat{A}_t\Big) + \beta \sum_{\mathcal{E}_i}\sum_{s_t}\big(V_\phi(s_t)-V_t^{\mathrm{targ}}\big)^2 + \gamma \sum_{(s_i,a_i)\in\mathcal{T}}\log\pi_\theta(a_i\mid s_i)
$$

三项分别是:PPO 的 clip 策略目标(用 advantage $\hat{A}_t$)、价值函数回归、以及在模型自采轨迹 $\mathcal{T}$ 上的 BC 对数似然。

用大白话说:先让老策略去新厨房里瞎试,把"碰巧成功"的录下来当模仿对象,再用 RL 把它打磨成一个"知道物体真实位姿(特权信息)、成功率很稳"的状态策略——因为用特权状态,不用渲染,可以开超大并行(2048 个环境)、训练飞快。

对少数 $\pi_{s1}$ 成功率仍低于阈值 $r$ 的**兜底环境集合** $\mathcal{F}$,退回去请人类补高质量示范,训第二个状态策略 $\pi_{s2}$。$\pi_{s1}$ 与 $\pi_{s2}$ 分别在 $\{\mathcal{E}_K,\dots,\mathcal{E}_{2K}\}\backslash\mathcal{F}$ 和 $\mathcal{F}$ 上生成数据,并入总集 $\mathcal{D}$。

**Teacher-student 蒸馏**:真实世界拿不到特权状态,所以把状态 teacher 蒸馏成以彩色点云为输入的 visuomotor 学生 $\pi_G$:

$$
\pi_G \leftarrow \max_\theta\; \mathbb{E}_{(o_i,a_i)\sim\mathcal{D}}\big[\log \pi_{G\theta}(a_i\mid o_i)\big]
$$

蒸馏时每个场景采 1000 条轨迹,其中 500 条由两个仿真相机渲染点云、500 条**直接从 mesh 采样合成**(全可见、更平滑,利于学习)。点云编码器用 Convolutional Occupancy Networks,输出 128 维。

用大白话说:老师能"作弊"看到物体真坐标,学生只能看点云;把老师的本事抄进只用点云的学生里,学生才能上真机。

**超线性由此产生**:每处理完一批,$\pi_G$ 跨的环境更多、泛化更强,能替人干活的比例越来越高,兜底集 $\mathcal{F}$ 越缩越小——于是"每个新环境的人力"随规模持续下降。Algorithm 1 就是这个 while 循环:采 2K 数字孪生 → rollout 过滤成功 → RL 微调 $\pi_s$ → 找失败环境补人类示范 → 蒸馏更新 $\pi_G$。

### 3. 部署时的场景微调

新目标场景 $\mathcal{E}_{\mathrm{test}}$ 上,预训练 $\pi_G$ 有非平凡 zero-shot 泛化但未必最优。两种微调:

- **无监督 scanned-deployment 微调**:只给一段视频扫描建出 $\mathcal{E}_{\mathrm{test}}$ 的数字孪生,让 $\pi_G$ 在其中自主 rollout,只留成功轨迹 $\mathcal{T}_{\mathrm{test}}$,再走 Eq.1 + Eq.2 的"RL→蒸馏"流程微调 $\pi_G$。**全程零人类示范/反馈**(Algorithm 2)。
- **Few-shot 有监督微调**:每个新环境采 10 条人类示范,冻结前面的"视觉处理层",只微调最后的全连接层。

## 三、实验结果

硬件:两家机构各一台 Franka Research 3(7-DoF + 平行夹爪,装在移动桌上),各挂两台 Intel RealSense D435i 深度相机做对齐点云。任务:(a) 把碗/杯/马克杯放进水槽(obj2sink,众包 56 个场景);(b) 把盒子放进柜子(obj2cabinet,36 个场景);(c) 开柜子(articulated,10 个场景)。真机评估横跨两机构、3 个厨房、每厨房 6 物体,共 108 次 rollout/策略。动作空间为 14 个离散动作(位置 $\pm0.03$m×3 轴、姿态 $\pm0.02$rad×3 轴、夹爪开/合)。

### Zero-shot 扩展律(obj2sink)

| 训练环境数 | 3 个厨房 zero-shot 成功率 | 说明 |
|---|---|---|
| 9 | 22.2% / 19.4% / 14.6% | 起步 |
| 36 | 38.9% / 30.6% / 35.0% | 随规模上升 |
| 56 | 64.8% / 47.2% / 75.0% | 综合约 62% |

- Fig 3a:训练环境从 0 增至 56,zero-shot 成功率单调上升至 **62%**。
- Fig 3b:仿真成功率与真机成功率呈**线性正相关**($R^2 = 0.92$),说明"在仿真里扩展"能等比例转化为真机涨点。
- Fig 3c:额外 8 个未见厨房上,训练环境 9→56 使成功率从 **16% 升到 60%**。

### 与大规模真机数据模型对比(Table V,move-to-sink)

| 方法 | 成功率 |
|---|---|
| **CASHER**(点云) | **62 ± 5** |
| Imitation Learning(点云) | 10 ± 5 |
| OpenVLA(zero-shot RGB) | 0 ± 0 |
| Octo(zero-shot RGB) | 0 ± 0 |
| Octo(fine-tuned RGB,10 demos) | 0 ± 0 |

在 Open X-Embodiment(80 万+真机轨迹)上训练的 OpenVLA、Octo **zero-shot 与 few-shot 都是 0%**,凸显该类任务数据仍严重不足;CASHER 靠仿真扩展显著胜出。

### 微调与鲁棒性

| 场景 | 设置 | 结果 |
|---|---|---|
| 无监督 scanned 微调 | 36-env 基座在 2 个差场景($\le$20%) | 平均 **+55%**,零额外示范 |
| Few-shot 微调(sink) | 3 个差场景,各 10 demos | 平均 **+54%**;IL 基线仅 10% |
| Few-shot 微调(box→cabinet) | 36-env,10 demos | 比 IL(0%)高 **36%** |
| Few-shot 微调(开柜子) | 10-env,10 demos | 比 IL(0%)高 **30%** |
| 多物体外推 | 单物体训练,清 3 物体场景 | 1/2/3 物体成功率 100% / 80% / 10% |
| 鲁棒性(暗光/杂乱/人为扰动) | 真机压力测试 | 成功率均 $\ge$30%(30%/30%/50%) |

**摊销效果**(Fig 4):持续数据采集下,每个环境所需人类示范数随批次**递减**(Fig 4b);且因成功率变高、达到同样成功次数所需 rollout 变少,**所需算力也随规模下降**(Fig 4c);持续采集版本在人力与性能上都优于"纯靠人类示范跑 CASHER"的对照。

### 成本

数据集规模:56 环境 × 1000 轨迹 = 56000 条,平均 120 步,共 672 万 state-action 对。单环境建场景 + 采 10 条仿真遥操作示范约 **1 小时**;RL 微调在 Quadro RTX 6000 上约 20 小时收敛;teacher-student 蒸馏采仿真轨迹 4 小时 + 合成 2 小时 + 蒸馏进视觉策略 **5 天**。

## 四、局限性

- 作者自陈:虽然相对人类示范实现了超线性扩展,**负担被转移到算力**;尽管算力也随规模下降,但仍高于直接采真机示范的时间成本。
- 仿真保真度受限:液体、可形变物体等**尚无法准确仿真**,任务局限在刚体 pick-place 与简单 articulation(开柜)。
- 依赖特权状态的 sparse-reward RL:每个任务需**手工设计成功判据/奖励**(如"物体位点距水槽位点 $<0.25$ 且直立且夹爪张开"),向复杂长程任务扩展的奖励设计成本未讨论。
- 众包数字孪生的物理参数(摩擦、质量)大量用 GUI 默认值,real-to-sim 的物理 gap 未系统量化;真机绝对成功率(62%)对生产部署仍偏低。
- "超线性"主要以曲线趋势 + $R^2$ 展示,**缺少对 scaling 指数的定量拟合**,统计上更像"经验观察"而非严格标度律。

## 五、评价与展望

**优点**:这篇工作把"real-to-sim-to-real + 众包 + 数据飞轮"三条线拧成一个自洽闭环,最漂亮的点是**把人力从"造行为"转移到"造场景"**——扫描厨房这件事非专家、异步、可并行、可全球众包,天然规避了遥操作对专家和物理机器人的线性依赖。$R^2=0.92$ 的 sim-real 线性关系是很有说服力的证据:它把"仿真里扩环境"直接兑换成"真机涨点"。数据飞轮(模型泛化替代人类示范)在实验上被验证为**人力单调递减**,这是对"self-improving robot data"叙事少见的真机级支撑。

**与公开工作的关系**:方法上直接继承作者团队的 Reconciling-Reality-through-Simulation(RialTo,arXiv:2403.03949)的 real-to-sim-to-real 与特权状态 RL,本文的增量是"跨环境泛化 + 众包 + 摊销飞轮"。与 RoboCasa、GenSim、URDFormer 等**程序化/LLM 生成仿真场景**路线相比,CASHER 主打"扫真实场景"以保住真实分布,是对"程序化生成分布不匹配"这一痛点的直接回应;代价是场景数量受众包意愿限制(实际 obj2sink 只 22/29 靠众包)。与 OpenVLA/Octo 的大规模真机 VLA 路线形成鲜明对照——后者在这些精细 pick-place 上 0%,说明"广覆盖但每任务稀疏"的真机数据 ≠ "窄任务但密集"的仿真数据,两条路线其实互补。

**开放问题与改进方向**:(1) 奖励/成功判据仍需手工,若能接入 VLM 自动判成功或自动生成奖励,飞轮才能真正无人化扩到长程任务;(2) 仿真 gap(尤其可形变/流体/接触丰富任务)是天花板,可与 differentiable sim、system-ID(如作者引用的 ASID)或 real-world 少量交互闭环结合;(3) "超线性"值得给出显式标度指数并做跨任务复现,以支撑其作为 scaling law 的强主张;(4) 众包激励与数字孪生质量参差是可扩展性的现实障碍,场景自动 curation/自动 articulation 的鲁棒性决定了这套 pipeline 能否真正跑到"成千上万"环境。总体上,这是把"仿真数据合成"从 procedural 推向 crowdsourced-digital-twin 的一次扎实实证,叙事完整、真机验证到位,是 real-to-sim 数据引擎方向的重要参考。

## 参考

1. Torne et al. *Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation*. arXiv:2403.03949, 2024.（RialTo,本文方法基座）
2. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.（大规模真机 VLA 对照基线)
3. Nasiriany et al. *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*. RSS, 2024.(程序化仿真场景生成路线)
4. Chen et al. *URDFormer: Constructing Articulated Simulation Environments from Real-World Images*. arXiv:2405.11656, 2024.(从图像自动重建可动仿真环境)
5. Schulman et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.(Eq.1 的 RL 基础)
