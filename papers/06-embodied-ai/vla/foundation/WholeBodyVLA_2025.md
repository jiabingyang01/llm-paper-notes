# WholeBodyVLA：面向全身移动操作的统一隐动作 VLA 框架

> **论文**：*WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control*
>
> **作者**：Haoran Jiang, Jin Chen, Qingwen Bu, Li Chen, Modi Shi, Yanjie Zhang, Delong Li, Chuanzhe Suo, Chuang Wang, Zhihui Peng, Hongyang Li et al.
>
> **机构**：Fudan University；OpenDriveLab & MMLab at The University of Hong Kong；AgiBot；SII
>
> **发布时间**：2025 年 12 月（arXiv 2512.11047）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.11047) | [PDF](https://arxiv.org/pdf/2512.11047)
>
> **分类标签**：`humanoid loco-manipulation` `latent action model` `whole-body control` `VLA` `RL locomotion controller`

---

## 一句话总结

WholeBodyVLA 用分离训练的操作/移动隐动作模型（LAM）从廉价的第一视角人类视频里学习移动-操作先验，再用离散指令式的 Loco-Manipulation-Oriented（LMO）强化学习策略把 VLA 输出的隐动作稳定落地为双足人形的全身控制，在 AgiBot X2 人形机器人上把三项真机任务的平均成功率从最强 VLA 基线的 56.7% 提升到 78.0%（+21.3 个百分点）。

## 一、问题与动机

- 人形机器人被视为通用具身智能体的理想载体，但完成"移动-操作（loco-manipulation）"任务的关键在于**移动为操作主动创造前提条件**（靠近、调姿、稳定），而不是把移动和操作当作两个独立阶段简单串行。
- 现有两条路线各有短板：(1) 模块化流水线（导航规划器 + 操作策略分开训练、高层调度）闭环反馈有限，误差会累积，交接处的机器人位形常常不利于后续操作；(2) 端到端全身模仿学习虽能避免模块间的交接问题，但需要大规模全身遥操作数据，采集成本（MoCap 或专业操作员遥操作）极高，导致数据稀缺——作者认为这是最根本的瓶颈。
- 现有基于强化学习的移动控制器普遍采用连续速度跟踪（velocity-tracking）目标，适合巡航但缺乏 episode 级别的可控性（精确起停、朝向精度），在移动-操作场景中经常出现踉跄、路径偏移、转弯不到位等问题；论文在附录故障统计中指出，这类失败很多时候并非来自 VLA 的高层决策错误，而是底层 RL 控制器精度/稳定性不足。
- 由此，本文一方面探索"能否让 VLA 从无动作标签的第一视角人类视频里学到移动-操作行为"以缓解数据稀缺，另一方面单独设计一个针对移动-操作（而非通用巡航）的底层 RL 控制器。

## 二、核心方法

整体流程（三阶段）：(1) 分离训练操作/移动隐动作模型（LAM）→ (2) VLA 在混合数据上联合预测两类隐动作 → (3) 轻量动作解码器把隐动作 grounding 到机器人指令，其中下肢移动指令由 LMO RL 策略转成稳定的下肢动作。

**1）分离的操作/移动 Latent Action Model。** 沿用 Genie、UniVLA 的 VQ-VAE 范式，编码器 $\mathcal{E}_i$（基于 DINOv2 特征的时空 Transformer）把连续帧 $(o_t,o_{t+k})$ 编码为连续隐向量 $z_t=\mathcal{E}_i(o_t,o_{t+k})$，量化为码本中最近的离散码 $c_t=\arg\min_{c\in\mathcal{C}_i}\|z_t-c\|_2$，解码器 $\mathcal{D}_i$ 用当前帧和量化隐动作重建未来帧 $\hat o_{t+k}=\mathcal{D}_i(o_t,c_t)$，训练目标为标准 VQ-VAE 损失：

$$\mathcal{L}_{\text{LAM}}=\mathcal{L}_{\text{mse}}+\|\text{sg}[c_t]-z_t\|_2^2+\beta\|c_t-\text{sg}[z_t]\|_2^2$$

大白话：把两帧之间"发生了什么"压缩成码本里的一个离散"词"，用能否重建出下一帧来监督这个词学得对不对，全程不需要真实动作标签。作者发现操作视频里相机几乎静止、画面变化以手臂运动为主，移动视频里相机持续运动、画面变化以整体场景相对运动为主——两种"隐动作"的注意力目标相互冲突，混合训练一个 LAM 会让表征退化并产生歧义编码。因此分别训练：操作 LAM 用 AgiBot World（大规模真机双臂操作数据集）训练；移动 LAM 用自采的第一视角"操作导向的移动"视频训练——单操作员头戴 RealSense D435i 或视场更大的 GoPro 相机，无需 MoCap 或遥操作，覆盖前进、侧移、转弯、下蹲等基本运动，且移动过程始终朝向潜在的操作目标物，共采集约 300 小时。

**2）VLA 联合预训练。** 以 Prismatic-7B 为骨干，在混合数据上联合预测操作与移动两类隐动作 token（极大似然/交叉熵）：

$$\min_\theta\big[-\log \pi_\theta(c_t^{\text{mani}}, c_t^{\text{loco}} \mid o_t,\ell)\big]$$

大白话：让同一个 VLA 在看到同一张图和语言指令时，同时预测"手该怎么动"和"腿该怎么动"这两个隐码，迫使模型学到二者如何协同，而不是各管一段。训练用 8×H100，batch 1024，共 20,000 步。

**3）轻量动作解码器 + LoRA 微调。** 解码器 $f$ 把预测出的隐动作 grounding 为机器人可执行指令：$a_t=f(\hat c_t^{\text{mani}},\hat c_t^{\text{loco}},s_t)$（$s_t$ 为机器人本体状态），输出上肢关节角和一个下肢移动指令；在 AgiBot X2 遥操作数据上用 LoRA 微调（batch 64，10,000 步，单模型覆盖全部三个任务）。

**4）Loco-Manipulation-Oriented（LMO）RL 策略。** 把下肢控制形式化为目标条件调节问题，用离散指令接口取代连续速度跟踪：$u_t=[s_x,s_y,s_\psi,h^\star]\in\{-1,0,1\}^3\times\mathbb{R}$（前后/左右/转向的三态启停标志加目标站高）。观测仅用本体感知短历史 $O_t=[u_t,\omega_t,g_t,q_t,\dot q_t,a_{t-1}]$，不依赖特权环境信息。为避免离散指令引发加速度突变，用平滑门控把三态意图转成速度参考：

$$v_k^{\text{ref}}(t)=v_k^{\text{goal}}\tanh\big[\alpha(s_k-\bar s_k(t))\big],\qquad \bar s_k(t)\leftarrow(1-\lambda)\bar s_k(t-1)+\lambda s_k$$

采用两阶段课程：Stage I 学习基础步态（随机采样目标速度幅值，关节活动范围随课程因子逐步放宽，暴露腿部于渐强扰动）；Stage II 固定巡航速度、专攻精度与稳定性，用朝向终止偏差

$$\mathcal{J}_{\text{dir}}=\big|\text{wrap}(\psi_{\text{end}}-\psi_{\text{start}})\big|$$

惩罚转向漂移，并把从 AgiBot World 中截取的真实手臂运动片段插值、时间扭曲后回放作为结构化扰动（而非随机噪声），迫使腿部学会补偿操作引起的真实惯性耦合；对静止 episode（$s_x=s_y=s_\psi=0$）加站立惩罚 $\mathcal{J}_{\text{stand}}=\|a_i^{\text{leg}}\|_2^2$ 抑制多余腿部动作。LMO 在 MuJoCo（单 H100）中训练，部署时以 50 Hz 运行在机载 NanoPi 上，VLA 主干以约 10 Hz 运行在 RTX 4090 工作站，二者经 ZeroMQ 通信实现闭环。

硬件平台为 AgiBot X2 人形机器人原型（双臂各 7 自由度 + Omnipicker 夹爪，双腿各 6 自由度，1 自由度腰部），头部装 RealSense D435i 第一视角相机；遥操作数据采集用 Meta Quest Pro VR 头显控制上肢、手柄控制移动指令，每个任务采集 50 条执行轨迹。

## 三、关键结果

三个真机任务套件（Agibot X2 实机，各拆两个子目标，每子目标 25 次试验）：Bag Packing（双臂抓纸袋→侧移下蹲放入纸箱）、Box Loading（下蹲抓箱→起身转身放上小车）、Cart Pushing（抓 50 kg 手推车把手→推行数米）。

| 方法 | Bag Packing（两子目标） | Box Loading（两子目标） | Cart Pushing（两子目标） | 平均成功率 |
|---|---|---|---|---|
| Modular Design（导航 + 操作模块化拼接，近似 oracle 上界） | 22/25, 12/25 | 9/25, 9/25 | 22/25, 22/25 | 64.0% |
| GR00T N1.5 + LMO | 20/25, 10/25 | 6/25, 4/25 | 12/25, 11/25 | 42.0% |
| OpenVLA-OFT + LMO | 19/25, 6/25 | 12/25, 12/25 | 22/25, 14/25 | 56.7% |
| **WholeBodyVLA（本文）** | **23/25, 13/25** | **19/25, 17/25** | **23/25, 22/25** | **78.0%** |

WholeBodyVLA 比最强 VLA 基线（OpenVLA-OFT + LMO）高 21.3 个百分点（论文原话："outperforming prior baseline by 21.3%"）。消融实验表明：把 LMO 换回传统连续速度跟踪 RL 会显著拉低整体成功率，作者将大部分差距（约 91.7%）归因于每个任务里包含更多移动成分的第二个子目标失败；去掉统一隐动作预训练（w/o LAM）、或只用操作型 LAM（不做移动预训练）都会明显掉点，且在需要较多移动才能完成操作的任务上掉点最大；用单一共享 LAM 同时编码操作和移动，效果也不如分离训练的两个 LAM（论文用 Relative Reconstruction Gain 指标进一步佐证：分开训练的 LAM 在各自专项上的重建增益都高于共享 LAM）。

LMO 底层控制精度对比（MuJoCo 仿真，位置/朝向误差，均值 ± 标准差，越低越好）：

| 控制器 | 前后行走 | 左右横移 | 原地转向 | CoM 摆动（站立/下蹲） |
|---|---|---|---|---|
| **LMO（本文）** | **0.21±0.01 / 0.05±0.01** | **0.55±0.01 / 0.06±0.01** | 0.05±0.01 / 0.19±0.01 | **0.03 / 0.03** |
| 速度跟踪基线（velocity-based policy） | 0.24±0.02 / 0.12±0.02 | 0.60±0.05 / 0.17±0.06 | 0.26±0.01 / 0.20±0.06 | 0.06 / 0.05 |

其余泛化实验要点：(1) 数据规模曲线显示，当人类视频预训练比例超过 50% 时，仅用 25 条遥操作轨迹微调即可达到用 25% 预训练比例、200 条遥操作轨迹才能达到的水平，说明统一隐动作学习显著降低了对遥操作数据的依赖；(2) 起始位姿变化、未见物体、未见桌面外观等 12 组真实世界泛化实验中 WholeBodyVLA 均保持领先，在未见负载/未见物体的视觉扰动测试里平均成功率约 64%，明显高于 GR00T + LMO（约 29%）和 OpenVLA-OFT + LMO（约 39%）；(3) 在地形穿越、长时序多步任务、视觉导航跟随、真空吸尘、桌面擦拭等训练分布外的扩展任务上，WholeBodyVLA 综合表现优于模块化基线以及去掉 LMO/LAM 的消融变体。

## 四、评价与展望

- 亮点：把"从无动作视频学隐动作"这一在桌面操作 VLA 中已被验证有效的思路（如 UniVLA、LAPA、IGOR）系统性地扩展到人形全身移动-操作场景，并给出了一个有说服力的工程洞察——操作视频与移动视频因相机运动模式根本不同，混合训练单个 LAM 会相互干扰，分离训练更优。同时论文清楚指出并解决了一个此前被忽视的问题：移动-操作失败往往不是高层 VLA 决策错误，而是底层 RL 速度跟踪控制器精度不够，用离散启停指令加两阶段课程的 LMO 直接对症下药，是本文除隐动作学习之外的第二个相对独立的贡献。
- 局限：三项主任务与扩展任务的成功率评测均由人工裁判打分（两名评委仲裁），每子目标仅 25 次试验，统计显著性有限；LAM 训练依赖 AgiBot World 这类特定平台的大规模操作数据以及自采的约 300 小时第一视角移动视频，跨机器人本体、跨相机配置的可迁移性未在文中验证；论文自陈仍难以处理长时序、精细灵巧的操作任务，未来工作方向是引入轻量建图/记忆以支持更长的规划视野，以及主动感知策略以应对杂乱或动态环境。
- 与同期公开工作的关系：相比 Humanoid-VLA（只做移动）、GR00T N1（更强调上肢操作）、Being-0/HEAD/R²S²（用 VLM 做高层规划器去调度模块化底层技能库，存在技能边界脆弱、依赖云端感知带来的延迟问题）、Boston Dynamics 的大行为模型演示（依赖昂贵 MoCap、工作空间受限），WholeBodyVLA 是较早在真实大空间场景下用统一端到端框架同时覆盖移动与操作、且不依赖 MoCap 或额外检测器信息的系统。开放问题包括：分离 LAM 的设计是否能进一步扩展到两种以上的"运动模态"；离散三态指令接口在更复杂地形或非平面移动场景下的表达能力是否足够；以及如何在不显著增加遥操作数据规模的前提下把该框架迁移到不同的人形本体。

## 参考

- Ye et al., 2025. LAPA: Latent Action Pretraining from Videos. ICLR.
- Bu et al., 2025b. UniVLA: Learning to Act Anywhere with Task-Centric Latent Actions. RSS.
- Bruce et al., 2024. Genie: Generative Interactive Environments. ICML.
- Bjorck et al., 2025. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
- Ding et al., 2025. Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration. arXiv:2502.14795.
