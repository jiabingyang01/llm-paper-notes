# LangMotion：基于语言化动作表征弥合机器人控制中的尺度差异

> **论文**：*Bridging Scale Discrepancies in Robotic Control via Language-Based Action Representations*
>
> **作者**：Yuchi Zhang、Churui Sun、Shiqi Liang、Diyuan Liu、Chao Ji、Wei-Nan Zhang（通讯作者）、Ting Liu
>
> **机构**：哈尔滨工业大学社会计算与交互机器人研究中心；科大讯飞研究院认知智能国家重点实验室；哈尔滨工业大学苏州研究院
>
> **发布时间**：2025 年 12 月（arXiv 2512.08548）
>
> **发表状态**：PDF 版权页标注 "Copyright © 2026, Association for the Advancement of Artificial Intelligence"，应为 AAAI 2026 录用论文
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.08548) | [PDF](https://arxiv.org/pdf/2512.08548)
>
> **分类标签**：`VLA` `动作表征` `分布偏移` `预训练` `motion token` `跨数据集泛化`

---

## 一句话总结

用规则化生成的自然语言"motion"描述（如"move forward left up, tilt up, rotate clockwise, close gripper"）作为动作的中间语义表征——只编码方向类别、不编码数值幅度——来消除多机器人数据集间因采样频率、末端执行器尺度差异导致的动作分布偏移;配合自适应阈值+多尺度分层窗口的运动检测算法（人工核验准确率 86.37% vs 固定阈值基线 57.62%），在 7 个 OXE 子数据集（约 1.2 万条轨迹）上做"仅预测 motion token"的预训练,再在 LIBERO/BridgeV2 上做"先 motion 后 action"两阶段微调,3B 模型在 LIBERO 四套件均值达到 78.1%（超过同规模 OpenVLA 的 76.5%），在 SimplerEnv/BridgeV2 四任务均值达到 35.3%（超过 Octo-Small 的 26.7% 与同数据集训练的 ECoT-7B 的 20.1%）。

## 一、问题与动机

近年来 VLA（OpenVLA、Octo、π0、RDT 等）普遍借助 Open X-Embodiment（OXE，覆盖 22 种机器人本体、逾百万任务）这类跨本体数据集做大规模预训练，但不同数据集之间在采集环境、机器人硬件、采集协议上的差异，导致动作命令存在严重的数值尺度偏移（distribution shift）：同一物理动作在不同数据集里的数值幅度可能相差数倍，这直接妨碍了预训练知识向下游任务的有效迁移，使模型往往需要大量微调才能在新领域可用。另外，已有的语言条件模仿学习方法通常在每个时间步都提供动态的视觉输入，却只给一条静态的语言指令，语言模态对动作生成的引导作用被削弱，语言的潜力没有被充分利用。

已有工作从不同角度缓解embodiment不一致问题：RDT 提出"物理可解释统一动作空间"来对齐不同来源的数据；HPT 用本体专属的 tokenizer（"stems"）把不同机械臂的本体感知/视觉信息映射到共享隐空间；RT-H、ECoT、Emma-X、CoA 等则引入语言链式描述或思维链来辅助动作生成，但依赖人工干预纠错（RT-H）或固定阈值切分运动（ECoT），跨数据集鲁棒性有限。本文的思路不同：不去对齐物理单位，而是把动作转写成一种**天然对数值尺度不敏感、只强调方向语义**的自然语言描述，用语言的语义鲁棒性去"抹平"跨数据集的数值差异，且整个转写过程是纯规则化的，不需要人工标注或外部模块（如 ChatGPT 纠错）。

## 二、核心方法

方法按顺序分三部分：动作 tokenizer、motion 生成、两阶段训练。

**动作 tokenizer**：延续 RT-2/OpenVLA 的做法，把 7 维动作 $(\Delta X,\Delta Y,\Delta Z,\Delta\text{roll},\Delta\text{pitch},\Delta\text{yaw},\text{GripperState})$ 离散化为 256 个 bin，对应词表尾部新增的 256 个专用 token。归一化时排除每一维中落在 1st–99th 百分位数之外的离群值，否则归一化区间被极端值拉大，bin 会变粗，损害精度。

**motion 表征**：固定模板的自然语言描述——"move [forward/backward] [left/right] [up/down]，tilt [up/down]，rotate [clockwise/counterclockwise]，[open/close] gripper"，若所有维度均未检测到运动则标记为"stop"。这套表征只编码运动的方向类别，不编码具体位移/角度数值，因此天然对不同数据集的数值尺度不敏感。

**自适应阈值**：为压制高速运动引入的抖动误判，把固定阈值改为随近期运动速度动态调整的自适应阈值：

$$
T_i(t) = T_{base}^i + \beta \cdot \frac{1}{\tau}\sum_{t-\tau}^{t} |\hat\Delta_i(s)|
\tag{1}
$$

用大白话说：如果最近一段时间（窗口 $\tau$ 内）机械臂本来就动得快，就适当抬高判定"这是一次有效动作"的门槛，避免把高速运动中的正常抖动误判成多个独立动作片段。

**分层多尺度检测窗口**：借鉴奇异系统中快变/慢变子系统的思路，设计快（fast）/中（mid）/慢（slow）三种时间尺度的运动检测器，$p(t)$ 为 $t$ 时刻 3D 末端位置，$T$ 为预定义阈值：

$$
M_f := \|\Delta_{t_f} p\| > 2T
\tag{2}
$$

$$
M_m := \|\Delta_{t_m} p\| > T \;\wedge\; \min_{\tau\in[t-\Delta t_m,\, t]} \|\Delta_\tau p\| > 0
\tag{3}
$$

$$
M_s := \|\Delta_{t_s} p\| > T \;\wedge\; \min_{\tau\in[t-\Delta t_s,\, t]} \|\Delta_\tau p\| > \frac{T}{2\Delta t_s}
\tag{4}
$$

$$
\text{Motion}(t) := M_f(t) \vee M_m(t) \vee M_s(t)
\tag{5}
$$

用大白话说：快窗口用高门槛（2T）捕捉短促剧烈的动作；中窗口在 ECoT 式判断逻辑上做修改，额外要求窗口内始终处于运动状态（不能中途静止），针对常规连续动作；慢窗口面向响应慢、状态变化缓慢的运动，用大窗口检测，但额外约束运动必须朝同一方向"匀速"推进，防止大窗口把多段独立动作误判成一次整体慢动作。三者取或，只要任一尺度判定为运动即记为一次 motion。作者用人工标注的 5%（部分大数据集为 3%）数据核验，本方法动作标注平均准确率 86.37%，显著高于 ECoT 式固定阈值方法的 57.62%；失败案例分析显示固定阈值方法容易把执行过程中的轻微抖动误判为多个独立动作。

**两阶段训练**：架构基于 OpenVLA，图像统一到 224×224，用 SigLIP + DINOv2 通道拼接编码，LLM 主干为 Qwen2.5（0.5B/1.5B/3B 三档），词表尾部追加 256 个动作专用 token。训练目标分解为：

$$
\phi(a, m \mid o, p) = \phi_h(m \mid o, p)\,\phi_l(a \mid o, p, m)
\tag{6}
$$

用大白话说：先学"大方向"再学"精细动作"——第一阶段（motion-only 预训练）只需要模型根据观测和指令预测粗粒度的 motion token（自回归 next-token 方式），呼应课程学习（curriculum learning）思想，用在 7 个 OXE 子数据集（含 furniture-bench、jaco 等，共约 1.2 万条轨迹，特意不含 LIBERO 和 Bridge V2 以检验跨数据集泛化能力）上训练；第二阶段（下游微调）在 LIBERO 和 Bridge V2 上，模型先预测 motion token 再以其为上下文条件预测具体的 action token，因为纯 motion 描述过于粗粒度，直接执行精度不够，需要在此基础上再学习细粒度动作。训练数据格式仿照 LLaVA 式 VLM 监督微调数据构造（对话式 system/user/motion/assistant 分段，只对 motion 与 action 段计算损失）。预训练 batch size 2048、微调 batch size 512，学习率 2e-5，A100-80G 上训练。

## 三、关键结果

**RQ1（消融）**：无论是否预训练、无论模型规模（0.5B/1.5B/3B），加入 motion 表征均一致提升 LIBERO 均值成功率（例如 3B 预训练后 78.1% vs 71.2%，+6.9 个百分点；SimplerEnv 上 3B 预训练后 35.3% vs 21.2%，+14.1 个百分点）。自适应阈值+分层窗口的优化相对原始（未优化）motion 表征也有稳定增益：

| 方法（0.5B, LIBERO） | Spatial | Object | Goal | Long | 均值 |
|---|---|---|---|---|---|
| Ours（自适应阈值+分层窗口） | 86.2 | 84.6 | 76.2 | 51.1 | 74.5 |
| 原始 motion（无优化） | 86.0 | 83.2 | 74.1 | 50.5 | 73.5 |
| 无 motion | 85.1 | 82.3 | 69.0 | 49.3 | 71.4 |

**RQ2（vs 基线/SOTA，预训练后微调，成功率 %）**：

| Method | LIBERO 均值 | SimplerEnv/BridgeV2 均值 |
|---|---|---|
| Diffusion Policy | 72.4 | — |
| Octo | 75.1 | 17.5（Base）/ 26.7（Small） |
| OpenVLA | 76.5 | 4.2 |
| ECoT（7B） | — | 20.1 |
| RT1-x | — | 1.1 |
| Ours 0.5B | 74.6 | 14.1 |
| Ours 1.5B | 74.6 | 19.0 |
| Ours 3B | **78.1** | **35.3** |

3B 模型在两个基准上均取得最优均值，且相对 ECoT-7B 用更小模型规模和更少预训练数据取得更优结果。所有方法（含全部基线）在 SimplerEnv 的"stack green block on yellow block"任务上成功率均为 0%，作者认为一是该任务对精度要求过高（抓取并精确堆叠两个很小的方块），二是微调数据 Bridge V2 采自真实世界而 SimplerEnv 是基于该数据集复现的仿真环境，存在 sim-to-real 差距。1.5B 模型在 SimplerEnv 上的增益相对有限，作者归因于真实世界微调数据与仿真测试环境的差距叠加了参数规模受限的影响。

**RQ3（表征对齐）**：用 PCA + 置信椭圆可视化 LIBERO-Spatial 任务上的 action/motion token 嵌入。端到端直接预测 action token 的基线模型中，action token 特征明显偏离原始词表分布；引入 motion 表征后（无论是否预训练）这一差距显著缩小；预训练后的 action token 特征进一步聚拢，与操作性能提升的趋势一致，而从头训练的表征则更分散，说明收敛不充分。

## 四、评价与展望

优点：把"跨数据集动作尺度不一致"这一具体痛点，转化为一个纯规则驱动、零人工标注成本的语言中间表征生成问题，思路简洁且工程可复现；自适应阈值+分层窗口的运动检测设计针对性解决了固定阈值方法（如 ECoT）在多数据集混合场景下的抖动误判问题，并给出了量化的人工核验结果（86.37% vs 57.62%）支撑设计合理性；两阶段训练（motion-only 预训练 → motion+action 联合微调）以较小规模数据（约 1.2 万条轨迹）和较小模型（0.5B–3B）取得了对标或超越 OpenVLA、Octo、ECoT-7B 的结果，显示出该表征在数据/算力效率上的潜力；RQ3 的表征可视化为"语言化动作表征缩小模态差距"这一核心假设提供了直接证据。

局限与开放问题：其一，motion 表征本身刻意粗粒度（只有方向类别，无幅度），论文也承认这类粗粒度预测"更容易学习但执行精度更差"，因此终究仍需下游微调阶段学习细粒度 action token，motion 更多起"方向先验/课程学习热身"作用而非直接可执行的动作输出，其收益上限依赖于第二阶段微调质量；其二，方法涉及多个人工设定的超参数（$T_{base}$、$\beta$、$\tau$ 及三个时间尺度窗口大小),虽然"生成过程"本身不需人工标注,但这些超参数的选取/调优本身仍是一种隐性人工先验,论文未给出这些超参数的敏感性分析；其三，预训练数据规模（7 个 OXE 子数据集、约 1.2 万条轨迹）远小于 OpenVLA、Octo 等基线所用的完整 OXE 规模，跨规模比较的公平性有一定局限；其四，仅在 LIBERO 与 SimplerEnv/BridgeV2 两个仿真/半仿真基准上验证，未见真实机器人实验，且 SimplerEnv 结果本身受 sim-to-real gap 影响（如全 0 的堆叠任务）。

与相关工作的关系：本文与 RT-H、ECoT、Emma-X 等"语言辅助动作生成"工作同属一个大方向——用中间语言表征连接高层语义与低层控制,但区别在于本文的 motion 语言是完全规则化、确定性生成的粗粒度方向描述,不依赖大模型纠错或人工干预链条,定位更接近"表征归一化工具"而非"推理链"；与直接把动作转写为更丰富自然语言描述（如同时期的 language-action pretraining 方向做法，将位移/角度数值也写入语言模板）相比，本文选择完全舍弃数值、只保留方向类别，代价是精度信息的损失，收益是对跨数据集尺度偏移的最大程度免疫，这一设计取舍及其与"更细粒度语言化动作"路线相比的效果差异，是值得后续工作系统比较的开放问题。

## 参考

- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246.
- Zawalski et al. *ECoT: Robotic Control via Embodied Chain-of-Thought Reasoning*. arXiv:2407.08693.
- Belkhale et al. *RT-H: Action Hierarchies Using Language*. arXiv:2403.01823.
- Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*. arXiv:2410.07864.
- O'Neill et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. ICRA 2024.
