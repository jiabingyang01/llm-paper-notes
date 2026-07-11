# ReBot：用真到仿到真的机器人视频合成扩展机器人学习

> **论文**：*ReBot: Scaling Robot Learning with Real-to-Sim-to-Real Robotic Video Synthesis*
>
> **作者**：Yu Fang, Yue Yang, Xinghao Zhu, Kaiyuan Zheng, Gedas Bertasius, Daniel Szafir, Mingyu Ding
>
> **机构**：University of North Carolina at Chapel Hill（计算机系）；Robotics and AI Institute；University of Washington（电子与计算机工程系）
>
> **发布时间**：2025 年 03 月（arXiv 2503.14526）
>
> **发表状态**：未录用（预印本，v1 [cs.CV]，IEEE 会议双栏格式）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.14526) | [PDF](https://arxiv.org/pdf/2503.14526)
>
> **分类标签**：`real2sim2real` `VLA数据扩增` `轨迹重放` `视频inpainting` `SimplerEnv`

---

## 一句话总结

ReBot 把真实机器人轨迹在仿真里**原样重放** 来替换被操作物体（real-to-sim），再把仿真里的机器人与物体**贴回被 inpaint 抹掉前景的真实背景上**（sim-to-real），从而全自动合成"物理真实、时序一致、多视角一致"的机器人视频来扩增 VLA 训练数据；在 SimplerEnv 上把 OpenVLA 域内成功率提升 21.8%、真机 Franka 上把 Octo/OpenVLA 成功率分别提升 17%/20%，且 VBench 时序一致性 93.0% 逼近真实视频的 96.1%。

## 一、问题与动机

VLA 模型（Octo、OpenVLA）直接在 Open X-Embodiment 这类真实机器人数据上训练，但真机采集昂贵（需要机器人 + 遥操作员），数据规模成为瓶颈,限制了 VLA 向新目标域的泛化——作者称之为机器人操作的"最后一公里部署难题"。

现有两条扩数据的路都有硬伤：

- **纯仿真数据**：便宜可扩，但动作空间与观测空间都存在 sim-to-real gap,策略难迁到真机。
- **生成式扩增**（如 ROSIE 用文生图 inpainting 直接在真实视频上换物体）：会产生 AI 伪影、纹理不一致，**时序不一致**（后几帧物体漂移变形）、**多视角不一致**，还常常不严格遵循指令条件。这些新的域偏差反而让 VLA 学不到稳定连续的动作。

作者的核心洞察：**与其用生成模型"凭空画"，不如把真实轨迹搬进仿真里重放。** 这样既拿到仿真的可扩展性（随便换物体），又用真实数据锚定动作空间（轨迹原封不动）和观测空间（背景来自真实 inpainting），把 sim-to-real gap 压到最小。

## 二、核心方法

给定真实数据集 $D = \{\tau_i\}_{i=1}^{M}$，每条 episode 记为 $\tau_i = \{o_t, a_t, \mathcal{L}\}_{t=1}^{T}$（$o_t$ 视频帧、$a_t$ 动作、$\mathcal{L}$ 语言指令）。目标是产出新合成 episode $\tau'_j = \{o'_t, a_t, \mathcal{L}'\}_{t=1}^{T}$，组成合成集 $D' = \{\tau'_j\}_{j=1}^{N}$ 来把 VLA 适配到目标域。

> **用大白话说**：注意合成 episode 里的动作 $a_t$ **和原始轨迹完全一样**，只有画面 $o_t \to o'_t$ 和指令 $\mathcal{L} \to \mathcal{L}'$ 变了——被操作的物体从"黄杯子"换成了"勺子"。所以动作标签是"免费白送"且物理自洽的。

整条流水线分三步，全自动、零人工干预。

### A) Real-to-Sim 轨迹重放

**场景解析与对齐**：在 Isaac Sim 里搭机器人、相机、桌子的数字孪生（digital twin），对齐到首帧 $o_1$。机器人/相机原型预先建好、只需调位姿；桌面高度这样估：对 $o_1$ 取 metric depth 建点云 → 用 GroundingDINO 以文本 prompt "table" 分割桌面 → 用四分位距（IQR）剔除离群点 → 取剩余点云的平均高度作为桌高。

**轨迹重放**：分析夹爪动作序列定出两个关键时刻——$t_{\text{start}}$（夹爪闭合抓取）与 $t_{\text{end}}$（夹爪张开放置）。先重放 $\{a_t\}_{t=1}^{t_{\text{start}}}$ 拿到 $t_{\text{start}}$ 时刻的夹爪位置，据此把新的仿真物体**放到原始真实物体所在的位置**；可选地在 $t_{\text{end}}$ 夹爪位置放一个容器。然后完整重放 $\{a_t\}_{t=1}^{T}$，录下操作新物体的仿真画面 $\{o_t^{\text{sim}}\}_{t=1}^{T}$。

> **用大白话说**：把新物体精准摆在"当年真实物体待过的地方"，机器人照着老轨迹走，就能顺理成章地抓到它——不用重新规划动作。

**重放校验**：换了物体后不一定抓得起来（取决于新物体与原物体的 affordance 兼容性）。通过监控 $t_{\text{start}}$ 到 $t_{\text{end}}$ 之间**物体与夹爪的笛卡尔距离** 自动判定是否成功操作，失败的 episode 直接丢弃。

### B) 真实背景 inpainting

目标是得到"任务无关"的干净真实背景 $\{o_t^{\text{real}}\}_{t=1}^{T}$，即把原视频里的**原始物体和真实机械臂** 都抹掉。

**分割与跟踪**：用 GroundedSAM2（= GroundingDINO + SAM2）。机器人用文本 prompt "robot" 在 $o_{t_{\text{start}}}$ 上分割（此刻机器人最显眼、效果最好）。而原始物体没有外观描述、文本 prompt 极易被干扰物误导——巧妙之处：real-to-sim 那步已经估出了物体的 3D 位置，把它用相机位姿投影到 $o_{t_{\text{start}}}$ 得到一个 **2D 点 prompt** 喂给 SAM2 分割物体。拿到 $t_{\text{start}}$ 的语义掩码 $m_{t_{\text{start}}}$ 后，用 SAM2 传播到全部帧得到 $\{m_t\}_{t=1}^{T}$。

**移除**：用 ProPainter（视频 inpainting）依据 $\{o_t, m_t\}$ 抹掉原始物体和真实机器人，得到 $\{o_t^{\text{real}}\}$。连真实机器人也一起抹掉、后面用虚拟机器人替代，是为了保证正确的遮挡关系和真实的物理交互。

### C) Sim-to-Real 视频合成

最后把仿真前景贴回真实背景：

$$
o'_t = \text{merge}\big(\text{extract}(o_t^{\text{sim}}),\; o_t^{\text{real}}\big)
$$

即从 $o_t^{\text{sim}}$ 里抠出机器人和被操作物体，叠到 $o_t^{\text{real}}$ 上；再把指令里的物体（"yellow mug"→"spoon"）和容器（"table"→"towel"）替换成重放时用的新物体，得到 $\mathcal{L}'$，组成 $\tau'_j = \{o'_t, a_t, \mathcal{L}'\}_{t=1}^{T}$。

> **用大白话说**：仿真负责"物理正确的运动前景"，真实图像负责"逼真的背景与遮挡恢复"，各取所长拼起来。因为前景在 3D 环境里渲染，**多视角一致性天然免费**——同一物体在两个相机视角里是同一个东西。

## 三、实验结果

**设置**：真实数据用 BridgeData V2（WidowX 250 6DOF）与 DROID（Franka Panda 7DoF + Robotiq 2F-85），另自采 220 条真机 episode。仿真用 Isaac Sim 4.1 + Isaac Lab，物体资产取自 Objaverse 厨房类。每任务用 100 条合成 episode 微调。4×A6000：Octo 全量微调（bs=256, lr=4e-5），OpenVLA 用 LoRA（bs=32, lr=5e-4）。基线 ROSIE 因未开源,作者用 Stable Diffusion 自行复现。

### 视频质量（VBench，越高越好）

| 维度 | ROSIE | ReBot | 真实视频 |
|---|---|---|---|
| Imaging Quality（成像质量） | 53.4% | **66.4%** | 70.1% |
| Subject Consistency（主体一致性） | 65.6% | **87.7%** | 93.2% |
| Background Consistency（背景一致性） | 83.7% | **92.2%** | 96.2% |
| Motion Smoothness（运动平滑度） | 85.2% | **99.2%** | 99.0% |

ReBot 时序质量三项均值 **93.0%**，逼近真实视频 96.1%；运动平滑度 99.2% 甚至反超真实视频 0.2%（作者归因于仿真消除了运动模糊）；成像质量仅比真实低 3.7%，比 ROSIE 高 13.0%。ROSIE 主体一致性只有 65.6%（后几帧物体漂移变形），且缺乏多视角一致性。

### SimplerEnv 域内性能（WidowX，Table I，Grasp/Success 平均）

| 模型 | Grasp 均值 | Success 均值 |
|---|---|---|
| Octo | 46.5% | 16.0% |
| Octo + ROSIE | 22.3% | 0.7% |
| **Octo + ReBot** | **54.7%** | **23.2%** |
| OpenVLA | 14.6% | 1.1% |
| OpenVLA + ROSIE | 31.3% | 0.0% |
| **OpenVLA + ReBot** | **59.4%** | **22.9%** |

域内成功率：Octo +7.2%、OpenVLA +21.8%；OpenVLA 抓取率从 14.6% 拉到 59.4%。**ROSIE 反而普遍拖累成功率**（尤其对依赖两帧观测历史的 Octo，因其时序不一致），多数任务 0.0%。

### 泛化性能（SimplerEnv，Fig 6，Success 均值，跨 physical/semantics/subject 三类）

| 模型 | 基线 | +ROSIE | +ReBot |
|---|---|---|---|
| Octo | 6.5% | 0.1% | **26.4%** |
| OpenVLA | 0.7% | 1.2% | **11.1%** |

Octo 泛化成功率 6.5%→26.4%（+19.9%，与摘要一致）；OpenVLA 0.7%→11.1%（抓取率均值 15.6%→66.8%）。注：摘要/结论把 OpenVLA 泛化增益概括为 **+9.4%**，与 Fig 6 逐格数字算出的 +10.4% 有轻微出入（论文内部小不一致，此处以图表数字为准）。

### 跨本体（Fig 7）

用 DROID（Franka）扩增数据微调 OpenVLA，再评测别的本体：WidowX 上平均成功率 OpenVLA 1.4%→ROSIE 3.1%→**ReBot 12.5%**；Google Robot "pick coke can" 三种位姿（站/横/竖）ReBot 分别达 41%/49%/9%，全面超越基线与 ROSIE。

### 真机（Franka Panda，Table II，10 trials/任务，Grasp/Success 平均）

| 模型 | Grasp 均值 | Success 均值 |
|---|---|---|
| Octo | 15% | 8% |
| Octo + ROSIE | 15% | 10% |
| **Octo + ReBot** | **35%** | **25%** |
| OpenVLA | 40% | 25% |
| OpenVLA + ROSIE | 18% | 5% |
| **OpenVLA + ReBot** | **50%** | **45%** |

（微调时另混入自采 220 条真机 episode。）真机成功率：Octo +17%、OpenVLA +20%。ROSIE 仅给 Octo 从 8%→10% 的边际提升,对 OpenVLA 甚至**掉到 5%**。对 Octo 初始 0% 的困难任务（put carrot in blue plate），ReBot 把抓取率抬到 40%、成功率抬到 20%。

## 四、局限性

- **只扩物体、不扩动作**：轨迹被原样重放，$a_t$ 不变。所以合成数据带来的是物体/外观多样性，**没有新增动作/运动多样性**,动作空间的扩展仍受限于原始真机轨迹分布。
- **只扩前景、不扩场景**：背景来自真实视频 inpainting，场景多样性被原始数据集"锁死",无法生成全新场景/光照/相机布局。
- **依赖数字孪生与精确感知**：需预建对应机器人的仿真孪生，并依赖 metric depth、相机位姿、桌面/物体分割的准确性；一旦这些环节出错，前景贴合与遮挡会失真。
- **affordance 兼容性过滤**：换的新物体若与原物体抓取可供性不兼容，重放会失败被丢弃——并非任意物体都能复用同一条轨迹。
- **任务范围窄**：仅桌面 pick-and-place，作者自己也把"超越桌面场景"列为 future work。
- **基线对比不完全公平**：ROSIE 未开源，采用作者基于 Stable Diffusion 的自复现（原版用 Imagen），复现质量可能影响对比结论。物体资产也仅限 Objaverse 厨房类。

## 五、评价与展望

**优点**：这是一个思路清晰、工程闭环、"性价比"很高的数据扩增方案。它的核心贡献不在某个新模型，而在于**用 real2sim2real 的组合拳把生成式扩增的两大顽疾（时序不一致、多视角不一致）从根上绕过去**——运动来自物理仿真所以物理真实，背景来自真实 inpainting 所以观测无 gap，多视角一致性由 3D 渲染免费获得。VBench 时序一致性逼近真实视频、且下游 VLA 提升显著且稳定，实证扎实。"物体位置估计同时充当分割点 prompt"是一处漂亮的模块复用设计。

**与其他公开工作的关系**：

- 相较 **ROSIE / GenAug / RoVI-Aug** 等生成式图像扩增，ReBot 用仿真锚定物理与时序,牺牲了"凭空生成新场景"的自由度换来了稳定性,实验中生成式基线甚至常常**负增益**，凸显时序一致性对 VLA（尤其带观测历史的 Octo）的重要性。
- 相较 **SimplerEnv / RoboTwin / RoboGSim** 等把 real2sim2real 用于"构建评测平台"的工作，ReBot 把同一策略**换到"数据扩增"用途** 上，是一个有价值的应用迁移；且强调全自动、无需人工搭孪生（区别于以往 real2sim 需大量手工建模）。
- 相较 **RoboDreamer / 视频世界模型** 类"生成新运动"的路线，ReBot 恰好互补：它不生成新动作但保证物理正确，而世界模型能生成新动作却缺物理接地。

**开放问题与可能的改进方向**：

1. **补上动作多样性**：目前只重放不扰动。可在仿真里对轨迹做物理可行的扰动/重规划、或对同一物体生成多种抓取，把"只扩物体"升级为"扩物体 + 扩动作"。
2. **补上场景多样性**：把 inpainting 背景与生成式/3DGS 场景合成结合，突破原始数据集的场景边界。
3. **扩到更复杂物体与任务**：铰接体、可变形物体、多步长程任务、非桌面场景。
4. **跨本体数据合成**：论文已初步展示 Franka→WidowX/Google Robot 的跨本体收益，若能自动适配不同相机布局与机器人形态,有望成为通用的跨本体数据放大器。
5. **更公平的基线**：与官方或更强的生成式/世界模型扩增方法在同等算力下正面对比,会更有说服力。

## 参考

1. Yu et al., *Scaling Robot Learning with Semantically Imagined Experience (ROSIE)*, arXiv:2302.11550, 2023 —— 本文主要对比的生成式扩增基线。
2. Li et al., *Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv)*, arXiv:2405.05941, 2024 —— 仿真评测平台与 real2sim2real 评测思路来源。
3. Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024 —— 主要被扩增/适配的 VLA。
4. Zhou et al., *ProPainter: Improving Propagation and Transformer for Video Inpainting*, ICCV 2023 —— 背景 inpainting 关键组件。
5. Ren et al., *Grounded SAM*, 2024 / Ravi et al., *SAM 2*, arXiv:2408.00714, 2024 —— 分割与跟踪组件（GroundedSAM2）。
