# Robot-WM Survey：面向机器人学习的世界模型综述

> **论文**：*World Model for Robot Learning: A Comprehensive Survey*
>
> **作者**：Bohan Hou, Gen Li, Jindou Jia, Tuo An, Xinying Guo, Sicong Leng, Haoran Geng, Yanjie Ze, Tatsuya Harada, Philip Torr, Oier Mees, Marc Pollefeys, Zhuang Liu, Jiajun Wu, Pieter Abbeel, Jitendra Malik, Yilun Du, Jianfei Yang（并列一作按字母序；Jianfei Yang 通讯）
>
> **机构**：Nanyang Technological University；UC Berkeley；Stanford University；The University of Tokyo；University of Oxford；Microsoft；ETH Zurich；Princeton University；Harvard University
>
> **发布时间**：2026 年 04 月（arXiv 2605.00080，v1，30 Apr 2026）
>
> **发表状态**：未录用（预印本，作者维护持续更新的 GitHub 仓库）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.00080) | [PDF](https://arxiv.org/pdf/2605.00080)
>
> **分类标签**：`世界模型` `机器人学习` `VLA` `视频生成` `综述`

---

## 一句话总结

这是一篇以**策略（policy）为中心**的机器人世界模型综述：它用一个统一的联合分布 $p(o_{t+1:t+k}, a_{t+1:t+k}\mid o_t, l)$ 把"策略模型 / 被动世界模型 / 可控世界模型 / 逆动力学模型"统一为同一分布的不同边缘或条件查询，并据此把近三年（2023.1–2026.3）文献沿三条主线组织——世界模型作为策略（6 类架构）、作为模拟器（RL 训练 + 决策评估）、作为机器人视频生成器（4 个能力层级）；同时汇总了 LIBERO/RoboTwin/CALVIN/SIMPLER 上的代表结果（如 Cosmos Policy 在 LIBERO 四套件平均 98.5、Say-Dream-ACT 98.1）与训练数据集/评测基准的属性对照。

## 一、问题与动机

纯反应式的 VLA 策略（RT-2、OpenVLA、$\pi_0$ 等）把当前观测直接映射到动作，作者指出其在**长程推理、时序信用分配、复合误差下的鲁棒性**上受限：问题不只在动作预测能力不足，更在于缺少"对世界如何随动作演化"的显式预测结构。世界模型（预测性表征）因此被重新重视。

作者刻意采取**机器人学习视角**而非最宽泛的生成式定义：世界模型的价值不在于生成"看起来合理"的未来，而在于生成**动作一致（action-conditioned）、可用于下游决策**的未来。据此提出一个可执行世界模型应具备三种核心能力：

- **foresight（前瞻）**：执行前预判未来状态或动作后果；
- **imagination-driven planning（想象驱动规划）**：用想象 rollout 比较、选择候选行为；
- **data amplification（数据放大）**：合成额外演示 / 交互轨迹来改善学习。

与先前综述（Zhang et al., 2025d）相比，本文声称三点差异：对世界模型范式更细粒度的划分、对其在策略/规划/仿真/评估/视频生成中角色更全面的分析、以及更清晰的"以机器人为中心"的世界模型定义（强调 action-conditioned consistency、长程可靠性、可部署性）。

## 二、核心方法（分类框架）

### 2.1 统一的概率视角

综述先把动作分为**低层运动指令 $a$** 与**高层语言指令 $l$**，给出通用世界模型形式：

$$
p(x_{t+1:t+H}\mid x_t,\, a_{t:t+H-1},\, l)
$$

其中 $x_t$ 可以是视觉观测、latent 状态、结构化物理状态甚至符号状态。视觉实例化即**视频生成世界模型** $p(v_{t+1:t+H}\mid o_t, a_{t:t+H-1}, l)$。

核心洞见（第 3 节）：以联合分布 $p(o_{t+1:t+k}, a_{t+1:t+k}\mid o_t, l)$ 为母体，几个看似不同的范式其实是它的不同查询：

$$
\text{Policy Model:}\quad p(a_{t+1:t+k}\mid o_t,l)=\int p(o_{t+1:t+k},a_{t+1:t+k}\mid o_t,l)\, do
$$

$$
\text{Passive WM:}\quad p(o_{t+1:t+k}\mid o_t,l)=\int p(o_{t+1:t+k},a_{t+1:t+k}\mid o_t,l)\, da
$$

$$
\text{Controllable WM:}\ p(o_{t+1:t+k}\mid o_t,a_{t+1:t+k}); \quad \text{IDM:}\ p(a_{t+1:t+k}\mid o_{t:t+k})
$$

**用大白话说**：策略、视频生成、可控世界模型、逆动力学模型不是四种不相干的东西，而是同一张"未来观测 + 未来动作"联合概率表的四种切法——把动作积掉就得到视频预测，把观测积掉就得到策略，固定动作条件就得到可控 rollout。这解释了为什么世界模型与策略能天然耦合。

### 2.2 世界模型作为策略（第 3 节，6 类架构，见 Table 1）

按"预测生成与动作生成如何交互"从解耦到端到端排序：

| 范式 | 代表工作 | 耦合方式 |
|---|---|---|
| **IDM-style（解耦：先预测后行动）** | UniPi, VidMan, Vidar, Gen2Act, VPP, Video2Act, MimicVideo, TC-IDM, LVP, Say-Dream-ACT | 世界模型冻结/轻调，另接逆动力学策略头 |
| **Single-backbone（单一生成骨干联合建模）** | UVA, UWA, VideoVLA, VideoPolicy, Cosmos Policy, DreamZero, UD-VLA, GigaWorld-Policy | 观测/动作 token 在同一去噪过程中联合建模 |
| **MoE/MoT（专家世界模型骨干）** | GE-Act, Motus, LingBot-VA, BagelVLA, Fast-WAM, LDA-1B, FRAPPE, DiT4DiT | 视频专家 + 动作专家经共享/交叉注意力反复交互 |
| **Unified VLA（统一多模态模型内化预测）** | GR-1, UP-VLA, WorldVLA, DreamVLA, UniVLA, CoWVLA, F1, InternVLA-A1, HALO, TriVLA | 未来图像/latent/结构知识作为联合训练信号 |
| **Latent-space WM（表征空间世界建模）** | FLARE, VLA-JEPA, JEPA-VLA, WoG, DIAL | 预测 latent 目标而非像素，避免生成解码开销 |

IDM-style 的形式：先构造未来（像素或 latent）

$$
\hat{\mathbf{o}}_{t+1:t+H}=\mathcal{W}(o_t,l),\qquad \hat{\mathbf{z}}_{t+1:t+H}=\mathcal{W}\!\left(\mathrm{E}_{\text{img}}(o_t),\mathrm{E}_{\text{text}}(l)\right)
$$

再让策略条件在当前观测与预测未来上 $\pi(a\mid o_t,l)=P(a\mid \mathrm{E}_{\text{img}}(o_t),\mathrm{E}_{\text{text}}(l),\Phi(\hat{\mathbf{o}}))$。

Single-backbone 把"先预测后行动"折叠为统一去噪目标（$\mathbf{x}=[z^v;z^a]$ 为视觉+动作表征拼接）：

$$
\hat{y}=f_\theta(\tilde{\mathbf{x}}_\tau,o_t,l,\tau),\qquad \mathcal{L}_{\text{unified}}=\mathbb{E}\big[\ell(\hat{y},y)\big]
$$

MoT 则保留专家分工、逐层交互：$(\mathbf{h}^v_{\ell+1},\mathbf{h}^a_{\ell+1})=\mathcal{F}^{\text{mix}}_\ell(\mathbf{h}^v_\ell,\mathbf{h}^a_\ell;o_t,l)$。

**用大白话说**：这条主线的历史趋势是"越绑越紧"——从两阶段解耦（视频模型 + 独立动作头），到共享一个扩散骨干，再到多专家深度耦合，最后到把预测能力完全内化进 latent 表征。作者反复强调：视频预训练骨干是否一定优于 VLM 骨干"仍是开放的经验问题"，当前证据只能算"有前景的归纳偏置"而非定论。

### 2.3 世界模型作为模拟器（第 4 节，见 Table 与 Fig. 5）

**4.1 用于强化学习**：把世界模型当作低成本可控虚拟环境，让 VLA 在想象 rollout 里试错。转移由 $(\hat{o}_{t+1},\hat{r}_t,\hat{d}_t)\sim p_\phi(\cdot\mid o_{\le t},a_{\le t},l)$ 生成，策略以

$$
J(\theta)=\mathbb{E}_{\hat{\tau}\sim(\pi_\theta,p_\phi)}\!\Big[\textstyle\sum_t \gamma^t \hat{r}_t\Big]
$$

或 GRPO-style 目标 $\mathcal{L}_{\text{RL}}=-\mathbb{E}\big[\min(r_t\hat{A}_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\big]$ 优化。代表：UniSim、World-Env、VLA-RFT、DiWA、World4RL、PlayWorld、RehearseVLA、WMPO、RISE、GigaBrain-0.5M*。第二层工作（World-VLA-Loop、VLAW、WoVR）进一步承认"模拟器本身不完美"，引入策略—世界模型**协同进化**：

$$
\phi^{k+1}\!\leftarrow\!\mathrm{UpdateWM}\big(\phi^k, D_{\text{real}}\cup D_{\text{policy}}(\pi_{\theta^k})\big),\quad \theta^{k+1}\!\leftarrow\!\mathrm{UpdatePolicy}\big(\theta^k, \hat{D}(\phi^{k+1})\big)
$$

**用大白话说**：物理机器人上做 RL 慢、贵、危险，于是把学到的世界模型当健身房让策略在想象里刷题；但健身房自己会出错，所以要用策略失败轨迹反过来修健身房，两者交替升级。

**4.2 用于评估**：世界模型给候选动作打分、排序、拒绝、做安全过滤，或作为 MPC 的前向动力学。代表：GPC、IRASim、World-in-World、DreamPlan（rollout 排序）；TD-MPC2、LeWorldModel（latent MPC）；Veo World Simulator（评估 Gemini Robotics）、WorldEval、WorldArena（作为策略评估代理与安全探针）。作者强调评估场景下 **action-faithfulness 尤为关键**：幻觉与长程误差会直接污染评分信号，"视觉真实并不充分"。

### 2.4 机器人视频世界模型（第 5 节，4 个能力层级，见 Table 2）

1. **Imagination-based（想象供监督/规划）**：Dreamitate、RoboDreamer、ManipDreamer、DreMa、PhysWorld、DreamGen、UniPi、VLP——用生成视频扩充监督或作视觉计划。
2. **Action-controllable（动作可控 rollout）**：IRASim、RoboEnvision、RoboMaster、Ctrl-World、EnerVerse-AC、Interactive World Simulator、EVA——重点从"像不像"转向"是否忠实跟随动作指令"。
3. **Structure-aware（结构感知，几何/交互先验）**：Mask2IV、TesserAct（扩展到 RGB+depth+normal 的 4D）、RoboVIP。
4. **Foundation video WM（基础规模可复用世界模型）**：Vid2World、Genie Envisioner、DreamDojo、WoW、UnifoLM-WMA-0、Cosmos Predict 2.5、GigaWorld-0、ABot-PhysWorld。

作者点明该领域的**核心瓶颈**：不再是生成逼真未来，而是生成"因果对齐动作、物理与运动学长程自洽、跨视角/跨本体一致、交互下稳定、可执行到足以支撑真实策略提升"的未来。

### 2.5 其他应用与评测（第 6–7 节）

- **导航**：Pathdreamer、VISTA/VISTAv2、NWM、SparseVideoNav、EgoWM——世界模型把未见空间变成可规划的预测基质。
- **自动驾驶**：MILE、OccWorld、GAIA-1、DriveDreamer、Drive-WM、UniDWM、DriveWorld-VLA、DriveVLA-W0、SteerVLA。
- **评测三分类**：开环（RBench、EWMBench、DreamGen Bench、EVA-Bench）、闭环任务效用（WorldArena、WorldEval、WorldGym、World-in-World）、物理一致性/可执行性诊断（WorldSimBench、WoW-World-Eval、WM-ABench）。反复的结论是"视觉可信度只是弱代理，action-grounded 一致性与可控性才是下游有用性的可靠指标"。

## 三、实验结果（综述汇总的代表数字）

综述本身不做实验，仅汇总各方法**原文直接报告**的成功率。以下为 Table 5（LIBERO 标准 4 套件，%）与 Table 6（RoboTwin/CALVIN/SIMPLER）节选。

**Table 5 — LIBERO（Spatial / Object / Goal / Long / Avg）**

| 范式 | 方法 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| Decoupled | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
| Single-backbone | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | **98.5** |
| MoE/MoT | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** |
| MoE/MoT | Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Unified VLA | RynnVLA-002 | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| Unified VLA | F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| Unified VLA | DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| Latent-space WM | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| Latent-space WM | JEPA-VLA | 97.2 | 98.0 | 95.6 | 94.8 | 96.4 |

**Table 6 — RoboTwin / CALVIN / SIMPLER（节选，%）**

| 方法 | RT-A | RT-B | CALVIN(C-D) | SIMPLER(S-W) |
|---|---|---|---|---|
| LingBot-VA | 92.9 | 91.6 | — | — |
| InternVLA-A1 | 89.4 | 87.0 | — | — |
| Motus | 88.7 | 87.0 | — | — |
| CoWVLA | — | — | 4.47 | 76.0 |
| WoG | — | — | — | 63.5 |

（RT-A/RT-B = RoboTwin 非随机/随机环境；CALVIN 报的是 ABCD 序列平均任务数 5.0 满分；SIMPLER S-G/S-W/S-O 分别对应 Google Robot/WidowX/其他设置。）

作者从中提炼三条观察：① 强结果**不局限于单一架构**——解耦、单骨干、MoT、统一、latent 各有强者，说明"逼真视频生成对有效具身控制并非必要"；② LIBERO 上 **Long 套件仍是主要区分器**，多数方法在 Spatial/Object 已接近饱和而在 Goal/Long 掉点更多；③ 跨基准（LIBERO↔RoboTwin↔SIMPLER）迁移弱，当前具身世界模型对本体/动作空间/评测协议差异仍敏感。

## 四、局限性

作者在第 8 节把**领域级**（而非某一方法的）开放挑战列为六点，也是这篇综述作为"路线图"的主要输出：

1. **因果条件不足（Causal Conditioning Gaps）**：许多预测目标主要由观测历史/任务意图驱动，未真正因果地绑定到待执行动作，生成的未来可能"语义合理但对候选动作不忠实"，损害闭环控制价值。
2. **效率瓶颈**：世界模型策略在训练与推理上远比 VLA 昂贵；缓解手段包括冻结骨干 + 轻量 adapter、部分去噪（MimicVideo、LingBot-VA）、latent 建模（LeWorldModel）、以及只在训练用、推理弃（Fast-WAM）。
3. **多模态感知瓶颈**：几乎只用视觉+本体，缺触觉/力反馈；异步、频率与维度差异大的信号在联合 latent 优化中易被高维视觉特征淹没。
4. **经典控制整合**：作为 MPC 前向动力学时 rollout 计算量巨大限制实时性；与 Lyapunov 稳定性/鲁棒控制等形式化保证的融合仍缺失。
5. **符号结构整合**：像素/latent rollout 长程误差累积，符号世界模型（谓词、关系、占据图）更利于长程组合推理，但需要抽象与感知 grounding，混合方案尚未成熟。
6. **评估指标缺失**：缺乏被广泛接受的功能导向评测；视觉保真既非必要也非充分，需要联合评估预测质量、动作敏感性、长程一致性与控制效用的"function-aware"框架。

作为综述本身的局限：绝大多数被引工作是 2025–2026 的预印本，很多数字来自各自原文的非统一协议（作者已在 Table 5/6 注明"按原文报告、不宜严格排名"），横向可比性有限；且覆盖偏重视觉/操作，对触觉、符号规划、真实机器人长期部署证据着墨相对较少。

## 五、评价与展望

**优点**。（1）**统一概率视角**是全文最有价值的贡献：把 policy/passive-WM/controllable-WM/IDM 收敛为同一联合分布的边缘/条件，为混乱的命名空间提供了清晰的坐标系，也解释了"为何世界模型与策略天然可耦合"。（2）**双坐标分类**（Sec 3 按架构耦合度、Sec 5 按视频世界建模能力）比单纯按发表时间罗列更有解释力，Table 1/2 的"backbone × coupling style × 推理时是否仍生成"三元对照尤其实用。（3）时效性极强，覆盖到 2026 上半年的一批工作，并配持续更新的仓库。

**缺点/风险**。（1）作为极新预印本，大量条目本身也是未经同行评审的预印本，"分类"先于"沉淀"，若干范式边界（如 Single-backbone 与 Unified VLA、MoT 与 Unified VLA）在实践中相当模糊，作者也承认"这些类别并非严格互斥"。（2）汇总表混用不同协议、部分单元格缺失，容易被读者误当作可直接排名的 leaderboard；作者虽加了免责声明，但仍有过度精确之嫌。（3）对"视频预训练骨干 vs VLM 骨干孰优"这一核心争议只给出"开放问题"的谨慎表态，缺少定量元分析支撑。

**与其他公开工作的关系**。本文明确对标并区别于 Zhang et al. (2025d) 的机器人操作世界模型综述，主打更细的架构—角色划分与机器人中心定义；理论根基上承接 Ha & Schmidhuber (2018) 的 recurrent world models 与 JEPA（Assran et al., 2023/2025）表征预测思想，并将后者延伸到 VLA 场景的 latent 世界建模一支（FLARE/VLA-JEPA/JEPA-VLA/WoG）。

**开放问题与可能改进方向**（作者视角 + 客观补充）：① 建立 function-aware、跨本体可比的标准评测（policy-ranking fidelity、可执行性诊断、IDM-based Turing Test 等），把"视觉可信"与"真正可执行"分离；② 攻克 action-faithfulness 与因果条件——让预测真正随机器人自身干预而变化，而非仅随任务意图；③ 触觉/力等 contact-rich 信号与视觉的对齐融合，以支撑接触密集任务；④ 神经 latent 与符号结构的混合世界模型，用于长程组合推理；⑤ 效率路线（部分去噪、latent 世界建模、训练用/推理弃）的系统化比较。总体判断：这是一篇**定位准确、组织清晰、时效性强的路线图式综述**，最适合作为该子领域的入门地图与选型参考，但其分类与数字应被当作"当下快照"而非稳定结论使用。

## 参考

1. Ha & Schmidhuber. *Recurrent World Models Facilitate Policy Evolution.* NeurIPS 2018.（世界模型思想源头）
2. Assran et al. *V-JEPA 2: Self-supervised Video Models Enable Understanding, Prediction and Planning.* arXiv:2506.09985, 2025.（latent 预测世界模型代表）
3. Zhang et al. *A Step Toward World Models: A Survey on Robotic Manipulation.* arXiv:2511.02097, 2025d.（本文主要对标的先前综述）
4. Cen et al. *WorldVLA: Unified Vision-Language-Action Model.* arXiv:2506.19850, 2025.（统一 VLA + 世界模型代表）
5. Zhu et al. *WMPO: World Model-based Policy Optimization for VLA.* ICLR 2026.（世界模型作为 RL 模拟器代表）
