# EmbodiSwap：面向零样本机器人模仿学习的具身换装

> **论文**：*EmbodiSwap for Zero-Shot Robot Imitation Learning*
>
> **作者**：Eadom Dessalene\*, Pavan Mantripragada\*, Michael Maynord, Yiannis Aloimonos（\* 共同一作）
>
> **机构**：University of Maryland, College Park（马里兰大学计算机系）
>
> **发布时间**：2025 年 10 月（arXiv 2510.03706）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.03706) | [PDF](https://arxiv.org/pdf/2510.03706)
>
> **分类标签**：`人到机器人换装` `零样本模仿学习` `V-JEPA` `自我中心视频`

---

## 一句话总结

用一条"分割 → 3D 手部重建 → 深度估计 → 图像 inpaint → 渲染融合"的五步流水线（EmbodiSwap），把自我中心（egocentric）人类视频里的手换成机器人夹爪，得到照片级真实的"机器人合成演示"；在这些合成视频上微调预训练视频预测模型 V-JEPA 去回归未来末端位姿，真机 UR10 上 5 类动作零样本达到 82%（70/85）成功率，比用 30 条真机演示训练的 π0（17.6%）高约 54 个百分点。

## 一、问题与动机

机器人操作的核心瓶颈是数据：为每个任务、每个环境、每种本体采集真机演示（遥操作、VR、脚本控制）代价高昂，且数据在硬件与环境上强偏置、难以规模化。相比之下，互联网上的第一人称人类视频海量、多样、天然富含手-物交互，是天然的行为数据源。但人类视频里没有机器人本体，直接拿来训练存在**本体鸿沟**（embodiment gap）：手的外观、自由度、接触方式都和机械爪不同。

已有的人类视频学机器人工作大多仍需真机演示做微调（如 Phantom、Egozero、Masquerade 仍依赖 in-lab 演示），或依赖目标图像条件（goal image conditioning，如 Zeromimic，等于给了部分演示）。本文的目标是**真正的零样本**：训练监督完全来自 in-the-wild 人类视频，部署时既不看真机演示，也不做目标图像条件。为此需要解决两件事：一是把人类视频"改造"成机器人视频以弥合外观鸿沟，二是找到一个能从这类视频里学会预测未来手/末端轨迹的策略骨干。

## 二、核心方法

方法分两块：数据侧的 **EmbodiSwap** 合成流水线，与模型侧基于 **V-JEPA** 的闭环策略。

### 2.1 EmbodiSwap：把人手换成机器人

对每一帧含手的人类 RGB 图像，逐帧执行如下操作（论文 Fig 3）：

1. **Body Segmentation**：用 SAM2 生成人体二值掩码（用在 VISOR 低分辨率分割上采到的点做 prompt，得到高分辨率人体掩码）。
2. **3D Hand Extractor**：用 HaWoR 联合重建 3D 手部骨架与相机位姿（world-space 手部运动重建）。
3. **Depth Model**：用 UniDepthV2 估计逐像素度量深度。
4. **Image Inpainting**：用 OmniEraser 结合人体掩码，把人及其影响（阴影等）从场景中抹除，得到"无人场景"。
5. **Render and Blend**：把 3D 手位姿重定向为夹爪位姿，用 PyBullet 的 IK 渲染出一个 RGB-D 机器人；再把场景深度图与机器人渲染深度图**逐像素比较，取更近（深度更小）的那一侧**来融合 inpaint 后的场景与机器人渲染。这样机器人替换了人手，同时物体对机器人的遮挡也能被正确保留。

**手到夹爪的位姿重定向（Gripper Pose Re-Targeting）**：从 HaWoR 预测的 MANO 参数取 21 个手部关节点 $kp_i,\ i \in [0,20]$，同时支持二指与三指夹爪。夹爪中心取手掌中心：

$$G_c = \frac{1}{5}\left(kp_1 + kp_5 + kp_9 + kp_{13} + kp_{17}\right)$$

夹爪 Z 轴对齐手掌法向：

$$G_z = (kp_5 - kp_0) \times (kp_{17} - kp_0)$$

夹爪 X 轴由拇指第一关节指向其余四指第一关节的质心：

$$G_x = \frac{1}{4}\left(kp_5 + kp_9 + kp_{13} + kp_{17}\right) - kp_1$$

Y 轴按右手 $G_y = G_z \times G_x$、左手 $G_y = -G_z \times G_x$ 补齐，最终夹爪位姿 $T_g = [\hat{G}_x\ \hat{G}_y\ \hat{G}_z\ G_c]$，既作为 inpaint 后场景中机器人渲染的姿态，也作为模型回归的目标。

> **用大白话说**：手上有 21 个关键点。取五根手指根部的平均当"手心"作夹爪位置；用手掌上两条边叉乘算出"手掌朝哪个方向"当 Z 轴；用"拇指指向其它手指中心"这条线当 X 轴；剩下的 Y 轴靠叉乘补齐，左右手差一个正负号。这样就把一只柔软的人手压缩成了一个刚性的六自由度夹爪位姿。

**动作边界重标注**：现有 egocentric 数据集的时间动作边界是按"动作分类"标的，在时间上定义模糊、不适合学运动。作者改用 **Therblig 子动作本体**（一套互斥的、按接触划分的低层子动作），**只依据手的运动**重标注边界：动作从工具物体被 grasp 时开始，在 use 操作结束前终止（如门开合完成、可切物体被切开）。

**监督信号（Ground Truth）**：把每帧与一个**未来帧**的手位姿配对，取两者间的相对平移与旋转作为 6D 位姿标签。lookahead 帧数按动作快慢自适应——快动作（open/close）用短偏移，慢动作（pour）用长偏移。

### 2.2 基于 V-JEPA 的闭环策略

骨干是 V-JEPA（在约 2M 人类动作片段上自监督预训练的视频预测 transformer），本文只用其 encoder 与 predictor 两个子网。输入是合成机器人图像 $I_0^{*}$、对应未来帧 $I_{1:T}$ 的位置 mask token $M_{1:T}$，以及可选的本体感知 $p_0$ 与动作位置 $l_0$：

- $I_0^{*}$ 过 encoder 得到嵌入；predictor 借助 $M_{1:T}$ 预测视频 $I_{0:T}^{*}$ 的特征级表征；
- $p_0$、$l_0$ 各经全连接编码，与 predictor 输出拼接后送入注意力探针 $C$（2 层 cross-attention + 2 层 self-attention），再过一层全连接输出**单个相对位姿向量**；
- 训练用 **L1 损失**，监督目标是 3D 手部重建网络给出的 $I_0$ 到 $I_T$ 之间的相对 3D 变换。

> **用大白话说**：V-JEPA 本来是拿来"看一段视频、预测未来画面在特征空间长什么样"的。作者把它掰弯成一个策略网络——不再预测像素，而是让它输出"末端接下来该怎么平移和旋转"，用未来那一帧手的位姿差当答案去拟合。

**闭环部署**：每一步网络输出一个动作，机器人执行，环境变化，新观测再喂回网络，循环固定步数。推理时**不依赖任何目标图像条件**（区别于 Zeromimic）。为弥合渲染机器人与真机外观的差异，部署时用真机关节角渲染合成 RGB 覆盖到输入图上（论文称此步经验上影响很小）。训练用单张 RTX A5000，40 epoch、batch 32、初始 lr $1\mathrm{e}{-3}$、cosine 退火到 $1\mathrm{e}{-7}$、weight decay 0.01；实验发现喂视频序列相比喂单帧反而**轻微掉点**，color augmentation 也会掉点。

## 三、实验结果

数据来自三个 egocentric 数据集：EPIC-Kitchens（全部 5 类动作）、HOI4D（place/open/close）、Ego4D（pour/cut）。评测分两部分。

### 3.1 预训练骨干对比（Table I，离线末端位姿预测）

在 EmbodiSwap 合成数据上，冻结 13 种视觉骨干、只训注意力探针，比较预测未来末端位姿的误差（Trans 为平移误差，单位米；Rot 为旋转误差，无量纲；Composite 为 5 动作聚合）。关键行如下：

| 模型 | 监督 | 模态 | 数据类型 | Composite Trans↓ | Composite Rot↓ |
|---|---|---|---|---|---|
| ResNet-50 | supervised | image | non-robot | 0.109 | 0.354 |
| DINOv2 | self-sup | image | non-robot | 0.084 | 0.320 |
| VC-1 | self-sup | image | non-robot | 0.096 | 0.333 |
| Octo | supervised | image | robot | 0.130 | 0.361 |
| RoboFlamingo | supervised | image | robot | 0.226 | 0.383 |
| $\pi_0$ | self-sup | image | robot | 0.102 | 0.323 |
| Hiera-L | self-sup | video | non-robot | 0.094 | 0.337 |
| V-JEPA+T (ViT-L) | self-sup | video | non-robot | 0.076 | 0.274 |
| V-JEPA (ViT-H) | self-sup | video | non-robot | **0.073** | 0.286 |
| V-JEPA (ViT-L) | self-sup | video | non-robot | 0.076 | 0.275 |

要点：（1）V-JEPA 系列全面领先所有 non-VJEPA 骨干；作者称 V-JEPA (ViT-L) 相比 $\pi_0$ 在平移/旋转误差上分别低约 34% / 15%。（2）**在大规模机器人数据上训练的骨干（RoboFlamingo、$\pi_0$、Octo）反而最不济**——说明"学机器人轨迹分布"对预测 egocentric 视频里的轨迹帮助不大。（3）除 V-JEPA 外最有竞争力的是 DINOv2，作者归因于两者都做"特征级预测"式预训练。

### 3.2 真机零样本评测（Table II，UR10 + Robotiq 夹爪）

在与训练厨房场景不同背景/光照的实验室里，用 ur_rtde 控制 UR10 评测 5 类动作。open/close 用力控（6 自由度柔顺），pour/cut/place 用位控（碰撞即停）。对比三者：$\pi_0$(30)=30 条真机演示微调；$\pi_0$(ES)=$\pi_0$ 视觉骨干在 EmbodiSwap 数据上微调、无真机演示；Ours=V-JEPA(ViT-L)+EmbodiSwap。

| 方法 | Open | Close | Pour | Place | Cut | All |
|---|---|---|---|---|---|---|
| $\pi_0$ (30 真机演示) | 6/20 | 4/20 | 2/15 | 2/15 | 1/15 | 15/85（17.6%） |
| $\pi_0$ (EmbodiSwap) | 18/20 | 3/20 | 1/15 | 0/15 | 0/15 | 24/85（28.2%） |
| **Ours (V-JEPA + ES)** | **19/20** | **17/20** | **10/15** | **10/15** | **14/15** | **70/85（82.4%）** |

要点：本文方法总成功率 82.4%，比用 30 条真机演示训练的 $\pi_0$（17.6%）高约 **54 个百分点**。$\pi_0$(ES) 仅在 open 上具竞争力（依赖抓取姿态与运动方向对齐），在 close/pour/cut/place 上大幅落后，说明同样喂 EmbodiSwap 数据，V-JEPA 骨干远优于 $\pi_0$ 骨干。Table I 的离线预测误差与 Table II 的真机成功率正相关——手部轨迹预测做得越好，真机越成功。

成功判据示例：open 需转门转过 65° 或抽屉拉出 80% 行程；pour 需 80% 以上泡沫块落入容器（用泡沫代替液体）；cut 需刀切透目标物。主要失败模式：关节奇异导致停机、自遮挡丢失关键视觉线索、以及"看似合理但实际未成功"的轨迹（如倒时略微没对准容器）。

## 四、局限性

- **零样本但非语言可控**：模型是逐动作独立训练/评测的（每类动作单独训一个），没有做成语言条件的单一通用策略，跨动作泛化未验证。
- **动作空间在笛卡尔末端位姿**：直接回归末端 6D 位姿，易触发关节奇异停机；作者建议改到关节空间预测，但未实现。
- **单视角 + 自遮挡脆弱**：推理时单视角，物体/夹爪自遮挡会抹掉关键线索；作者提出用多视角缓解，同样留作未来工作。
- **依赖一长串现成感知模型**：整条合成流水线串联 SAM2、HaWoR、UniDepthV2、OmniEraser、PyBullet IK，任一环节（尤其 3D 手部重建）出错都会污染监督信号；论文未系统量化各环节误差对最终策略的传导。
- **评测规模有限**：真机每类动作 15–20 次试验、单一机器人（UR10）、单一实验室场景，且 place 为凸显难度刻意用大物体，统计置信区间未给出。
- **物体 3D 未建模**：方法主要建模手/夹爪运动，未显式建模被操作物体的 3D，涉及 in-hand 操作的任务受限。

## 五、评价与展望

**优点**：（1）问题切口干净——把"人类视频学机器人"的两大痛点（外观本体鸿沟、监督信号）分别用"深度感知的机器人换装合成"和"未来手位姿相对变换"两把钥匙解决，且真正做到部署时零真机演示、零目标图像条件，这在同类工作（Phantom/Egozero/Masquerade 仍需真机演示，Zeromimic 需目标图像）中定位清晰。（2）最有价值的实证发现是**"特征级视频预测预训练（V-JEPA/DINOv2）显著优于在机器人数据上预训练的骨干（π0/Octo/RoboFlamingo）"**，这与 VLA 社区"越多机器人数据越好"的直觉相左，13 个骨干的横向对比证据较扎实，值得后续工作重视。（3）单卡 A5000 可训、代码/数据集/checkpoint 全开源，复现门槛低。

**缺点与开放问题**：（1）"零样本"的代价是把感知负担全压给了现成模型链，方法的上限受制于 HaWoR 等手部重建的精度，鲁棒性边界不清。（2）深度融合遮挡处理仅按"取更近深度"这种硬阈值，对手-物接触的细粒度物理（接触力、软体形变）无从表达，pour/cut 这类接触丰富任务的成功可能更多靠"看起来对"的轨迹拟合而非物理理解。（3）逐动作独立建模削弱了"通用策略"的说服力，与 π0/Octo 这类语言条件通才策略的对比其实并不完全对等——后者牺牲了单任务精度换取通用性。

**可能的改进方向**：改到关节空间或加入奇异性感知的动作参数化；引入多视角/主动视角以对抗自遮挡；把物体 3D 或接触表征纳入合成与监督；将逐动作策略统一为语言条件的单一策略并测跨任务/跨本体泛化；系统评估合成流水线各环节误差的传导，或用可微渲染做端到端优化。总体上，这篇工作在"人类视频合成机器人数据 + 视频预测骨干"这条路径上给出了一个干净、可复现、结论有反直觉价值的样本点，主要待补的是规模化评测与更强的物理/接触建模。

## 参考

1. A. Bardes et al. *V-JEPA: Latent Video Prediction for Visual Representation Learning.* 2023.（本文策略骨干）
2. K. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.（真机对比基线）
3. M. Lepert, J. Fang, J. Bohg. *Phantom: Training Robots without Robots using only Human Videos.* arXiv:2503.00779, 2025.（人类视频学机器人，需真机演示）
4. J. Shi et al. *Zeromimic: Distilling Robotic Manipulation Skills from Web Videos.* arXiv:2503.23877, 2025.（依赖目标图像条件的对照路线）
5. J. Zhang et al. *HaWoR: World-space Hand Motion Reconstruction from Egocentric Videos.* CVPR 2025.（3D 手部重建来源）
