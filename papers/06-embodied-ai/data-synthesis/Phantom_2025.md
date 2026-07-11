# Phantom：无需机器人、仅用人类视频训练机器人

> **论文**：*Phantom: Training Robots Without Robots Using Only Human Videos*
>
> **作者**：Marion Lepert, Jiaying Fang, Jeannette Bohg
>
> **机构**：Stanford University
>
> **发布时间**：2025 年 03 月（arXiv 2503.00779）
>
> **发表状态**：CoRL 2025（9th Conference on Robot Learning, Seoul, Korea）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.00779) | [PDF](https://arxiv.org/pdf/2503.00779)
>
> **分类标签**：`人类视频学习` `数据编辑` `跨本体迁移` `零样本部署`

---

## 一句话总结

Phantom 把第三人称 RGBD 人类操作视频，通过"手部姿态估计抽动作 + 抠除人手并叠加虚拟机器人渲染"这套数据编辑流水线，转换成机器人的观测-动作对，从而**完全不采集任何机器人数据** 就训练出闭环 imitation learning 策略，并可零样本部署到真实 Franka / Kinova 上，室内单场景成功率最高达 92%，跨场景（OOD）仍达 72%–84%。

## 一、问题与动机

机器人操作数据集比视觉/语言大模型的训练语料小几个数量级，瓶颈在于遥操作采集慢、贵、难扩展，且不仅要"量"更要"多样性"。人类视频天然海量、多样、任务信息丰富，但用它训机器人有两大障碍：

1. **缺显式动作标签**——视频里没有末端执行器位姿/夹爪开合等 action。
2. **本体外观差异大**——人手/人臂和机械臂/夹爪长相完全不同，纯视觉策略难以跨本体泛化。

以往做法要么与机器人数据 co-train（EgoMimic、MimicPlay、Vid2Robot 等），要么用强化学习或物体中心（object-centric）轨迹，本质上仍**依赖机器人数据来弥合鸿沟**，因而在"人类数据远多于机器人数据"的目标数据体制（附录量化为机器人数据占比 ≪ 1%）下不可扩展。作者的目标是构造一个**只需人类视频、机器人数据为零**的可扩展框架。核心洞见来自机器人-机器人（robot-to-robot）跨本体迁移中的**数据编辑**技术（Mirage / Rovi-Aug / Shadow），本文把它首次成功迁移到更难的人-机器人（human-to-robot）场景。

## 二、核心方法

### 问题设定

给定人类视频数据集 $\mathcal{D}_{\text{human}} = \{\tau_h^i\}_{i=1}^{N}$，每条演示是一串第三人称 RGBD 图像 $\{I_{h,t}\}_{t=1}^{T}$，演示用**拇指-食指捏取（pinch grasp）** 完成。目标是把每帧转换为机器人观测-动作对：

$$I_{h,t} \rightarrow (I_{r,t},\, a_{r,t})$$

机器人动作定义为末端位姿加夹爪开合：

$$a_{r,t} = (\mathbf{p}_t,\, \mathbf{R}_t,\, g_t),\quad \mathbf{p}_t \in \mathbb{R}^3,\ \mathbf{R}_t \in \mathbb{R}^6,\ g_t \in [0,1]$$

其中 $\mathbf{R}_t$ 用 6D 连续旋转表示，$g_t$ 为归一化夹爪开度。假设部署相机外参已知，且训练/测试相机视角相近（但场景可以完全不同）。

> 用大白话说：把"人手怎么动"翻译成"机械臂末端该在哪、朝哪、夹多紧"，再把画面里的手换成机械臂，这样训练出来的策略看到的东西就跟真机器人一模一样。

### 2.1 从人类视频抽动作标签

- 用 **HaMeR** 对每帧估计手部姿态，输出 21 个解剖关键点 $\hat{\mathbf{X}}_t \in \mathbb{R}^{21\times3}$ 和 778 顶点手网格 $\hat{\mathbf{V}}_t \in \mathbb{R}^{778\times3}$。
- HaMeR 基于单目图像，绝对 3D 位置不准。于是引入**深度**：用 **SAM2** 分割手得到掩码 $M_t$，结合深度图 $D_t$ 抠出部分手点云 $\mathbf{P}_t$，再用 **ICP** 把 HaMeR 网格对齐到点云，求得刚体变换 $\mathbf{T}_t \in SE(3)$，使 $\mathbf{P}_t \approx \mathbf{V}_t = \mathbf{T}_t \hat{\mathbf{V}}_t$；由于网格与关键点内部一致，同一变换修正关键点：

$$\mathbf{X}_t = \mathbf{T}_t \hat{\mathbf{X}}_t$$

- 抓取时手指被遮挡，HaMeR 会给出不合理姿态。作者把拇指、食指最后两个关节**约束成单自由度** 并限制在解剖可行范围内。
- 用修正后的关键点定义动作：位置 $\mathbf{p}_t$ 取拇指尖 $\mathbf{x}_t^{\text{thumb,tip}}$ 与食指尖 $\mathbf{x}_t^{\text{index,tip}}$ 的**中点**；朝向 $\mathbf{R}_t$ 由拟合过拇指-食指所有关键点的平面法向 + 拇指主轴向量确定；夹爪开度 $g_t$ 取两指尖距离。为缓解抓取滑移，把单条轨迹里预测夹爪距离的**最低 20 百分位** 强制置为完全闭合。
- HaMeR 输出在相机坐标系，用已知外参转到机器人坐标系得到最终 $a_{r,t}$。

> 用大白话说：先用现成的手部估计器猜出手的骨架，再用深度点云"校准"到正确的空间位置；两指尖中点当作夹爪该去的地方，两指尖连线的方向当作夹爪该转的角度，两指尖间距当作夹爪开多大。

### 2.2 弥合视觉观测鸿沟（数据编辑）

沿用 Rovi-Aug 的数据编辑思路，从 robot-to-robot 改造到 human-to-robot：

- **训练阶段**：用 SAM2 分割人臂 → 用 **E2FGVI** 视频修复（inpainting）抹掉人臂还原背景 → 用 **Mujoco** 渲染目标机器人（末端置于 $(\mathbf{p}_t,\mathbf{R}_t,g_t)$），按已知外参从对应视角合成机器人图像叠加回原图 → 用深度判断遮挡关系，让被环境物体挡住的机器人部分正确被遮。得到"像真机器人在干活"的编辑图。
- **推理阶段**：真机观测里是真实机械臂，但训练图里是渲染机器人（颜色纹理略有差异）。为最小化 domain shift，测试时把**虚拟机器人臂叠加到真实机器人臂上**，保证训练/测试视觉一致。（作者选此简单方案，未采用 Rovi-Aug 训练时加颜色扰动的做法。）

## 三、实验结果

设置：Franka（OSC 控制器）与 Kinova（IK 控制器）两种机器人；策略用 **Diffusion Policy**；每任务采 250–350 条人类演示（附录 Table 7：各任务 268–313 条，Kinova 扫地任务 950 条）；每次评测 25 次 rollout。对比三种数据编辑基线：Hand Inpaint（本文 Phantom）、Hand Mask（改自 Shadow，测试时用扩散模型生成手掩码）、Red Line（改自 EgoMimic，人臂涂黑+红线）、以及不做任何编辑的 Vanilla。

### 室内同分布场景（Table 1，成功率）

| 任务 | Phantom（Hand Inpaint） | Hand Mask | Red Line | Vanilla |
|---|---|---|---|---|
| Pick/Place Book | 0.92 | 0.92 | 0.0 | 0.0 |
| Stack Cups（杯径仅差 1.5 cm） | 0.72 | 0.52 | 0.0 | 0.0 |
| Tie Rope（打 8 字帆结） | 0.64 | 0.60 | 0.0 | 0.0 |
| Rotate Box | 0.72 | 0.76 | 0.0 | 0.0 |
| Grasp Brush | 0.88 | 0.75 | 0.0 | 0.0 |
| Sweep $>0$ 块 | 0.80 | 0.75 | 0.0 | 0.0 |
| Sweep $>2$ 块 | 0.72 | 0.72 | 0.0 | 0.0 |
| Sweep $>4$ 块 | 0.40 | 0.68 | 0.0 | 0.0 |

要点：Phantom 与 Hand Mask 都能高成功率完成，Red Line 和 Vanilla **全 0**——说明单纯抹掉人臂不叠加机器人本体无法跨越视觉鸿沟。Hand Mask 因测试时要额外跑扩散模型生成掩码，rollout 平均慢 **73%**。

### 跨分布（OOD）场景——扫地任务（Table 2）

采集 950 条跨室内外场景人类演示（80% 室内），在三个未见环境评测：

| 场景 | Phantom | Hand Mask |
|---|---|---|
| Outdoor lawn | 0.72 | 0.52 |
| Indoor lounge | 0.84 | 0.76 |
| Indoor lounge + 未见表面 | 0.64 | 0.68 |

室内 lounge 最好（与 80% 训练数据室内一致）；换未见表面掉约 20%（训练仅 4 种表面）。Phantom 综合更优且 rollout 快 73%。

### 修复质量消融（Table 3，Indoor Lounge 扫地）

| 修复策略 | 成功率 |
|---|---|
| E2FGVI 高质量修复 | 0.84 |
| OpenCV 低质量修复 | 0.76 |
| Mask only（只涂黑不修复） | 0.60 |

结论：高质量修复最好，但低质量 OpenCV 仅差 8pt——训练时的天然多样性让模型对伪影不敏感；完全不修复掉 24pt。而 Mask only（60%）远高于 Red Line（0%），佐证**训练时叠加机器人本体渲染是关键**。

### 附录补充实验

- **Co-train 收益（Table 5）**：仅用 100 条 Kinova 遥操作数据，室内 0.88、换新场景**掉到 0.0**；与 950 条多样人类视频 co-train 后，新场景升到 **0.80**——人类数据的价值在于跨环境可扩展性。
- **人类 vs 机器人数据（Table 6）**：50 条时机器人 0.52 vs 人类 0.44（$p=0.778$，无显著差异）；100 条 0.88 vs 0.64（$p=0.095$）；人类数据扩到 300 条则升至 **0.84**——单条人类演示精度略低，但易采集可靠"量补质"。
- **本体无关（Fig 8）**：同一段人类视频可编辑成 Franka / Jaco / UR5e / Kinova Gen3 / Sawyer / IIWA 等多种机器人。
- **对比表（Table 4）**：Phantom 是表中唯一同时满足"无需机器人数据 + 支持可变形物体 + 闭环执行"三项的方法。

## 四、局限性

作者自陈四点：

1. 依赖现成手部姿态估计器，遮挡时 HaMeR 出错会传导为动作误差（但也意味着随估计器进步而变好）。
2. 只在"机器人能照搬人类策略"时有效；人手不会撞环境但机械臂可能撞，且人指尖与夹爪表面属性不同会导致物体运动差异。
3. 仅限捏取（parallel-jaw 平行夹爪），无法覆盖灵巧多指抓取（作者指出这是几乎所有大规模机器人数采的共性限制）。
4. 只评测准静态（quasi-static）任务，未处理人类演示与真机 rollout 之间的延迟失配。

## 五、评价与展望

**优点**：思路简洁且验证扎实——把已在 robot-to-robot 迁移中成熟的数据编辑（Rovi-Aug/Shadow/Mirage 一脉）迁到更难的人-机器人场景，并用"HaMeR + 深度 ICP + 关节约束"这套工程化组合把动作抽取做得够稳。最有说服力的是 Red Line/Vanilla 全 0 与 Mask-only 60% 的对照，干净地证明"训练时必须叠加目标机器人本体渲染"这一因果性结论；可变形物体（打绳结）与多物体扫地也真实跑通，闭环执行区别于 R+X（开环）与 ORION/DITTO（物体中心、对可变形/多物体失效）。零机器人数据 + 跨室内外部署，方向上直指人类视频数据体制的可扩展性。

**缺点与开放问题**：

- **假设偏强**：需第三人称 RGBD、相机外参已知、训练/测试视角相近、演示统一为捏取。这让"任何人拿手机随手拍"的宣称打了折扣（真正 in-the-wild 单目、任意视角尚未验证）。
- **规模未兑现**：论文自认只在"渐近数据体制"下论证，实际每任务仍是数百条同场景演示、每任务单独训练，并未真正展示"人类数据涨几个数量级"时的 scaling 曲线。
- **动作精度天花板**：完全受限于 HaMeR + 深度对齐的手位姿精度，Table 6 也显示人类数据单条精度确实低于遥操作。
- **策略头较传统**：用 Diffusion Policy 单任务训练，未与近来的大规模自回归 VLA（如 π0、RT-X 类）结合；作者在结论里也把"把编辑后的观测-动作对喂给通用 VLA"列为未来工作，这是最自然的下一步。
- 可改进方向：放宽视角/外参假设（多视角采集鲁棒化）、扩到灵巧手与非准静态任务、把数据编辑产物直接注入大规模 VLA 预训练语料以检验真实 scaling。

总体看，Phantom 是"用数据编辑弥合人-机器人本体差、零机器人数据"这条线上一篇干净有效的工作，结论清晰、消融到位，主要短板在假设强度与规模验证的缺失。

## 参考

1. Chen et al., *Rovi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning*, CoRL 2024.（本文数据编辑方案的直接来源）
2. Lepert et al., *Shadow: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer*, CoRL 2024.（Hand Mask 基线来源）
3. Pavlakos et al., *Reconstructing Hands in 3D with Transformers (HaMeR)*, CVPR 2024.（手部姿态估计骨干）
4. Kareer et al., *EgoMimic: Scaling Imitation Learning via Egocentric Video*, arXiv 2024.（Red Line 基线，需与机器人数据 co-train）
5. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, IJRR 2023.（策略头）
