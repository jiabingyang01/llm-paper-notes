# HoMMI：从人类演示中学习全身移动操作

> **论文**：*HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations*
>
> **作者**：Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Jeannette Bohg, Shuran Song et al.
>
> **机构**：Stanford University；Toyota Research Institute
>
> **发布时间**：2026 年 03 月（arXiv 2603.03243）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.03243) | [PDF](https://arxiv.org/pdf/2603.03243)
>
> **分类标签**：`人到机器人` `移动操作` `UMI 扩展` `whole-body control` `egocentric`

---

## 一句话总结

HoMMI 把手持式 UMI 采集接口扩展出**头戴 egocentric 相机**（两部 iPhone 装夹爪 + 一部装帽子，靠 ARKit 多机协同建立统一坐标系），再用「embodiment-agnostic 3D 视觉表征 + 3D look-at 头部动作表征 + gripper-centric 坐标系 + 约束感知全身控制器」四件套弥合人-机具身差异，从而**无需任何机器人 teleoperation 数据**即可直接从人类演示中学到双臂全身移动操作技能，在 Laundry / Delivery / Tablescape 三个长程任务上分别取得 **90% / 85% / 80%** 成功率，全面超过 UMI（Wrist-Only）等 baseline。

## 一、问题与动机

移动操作要求 **whole-body coordination**：协调头部（egocentric 相机）、双臂、躯干、移动底座去完成导航、双臂协作、任务进度跟踪等。现有从人演示学习的范式大多依赖机器人 teleoperation 采集，昂贵、缓慢、难以在多样真实场景铺开；而 UMI 这类手持设备虽可 in-the-wild、robot-free 地采集，但**只有腕部相机**，视野局部，欠观测（under-observe）导航、双臂协作与全局任务上下文所需的信息。

自然的补救是加一个**头戴相机**提供 egocentric 全局视野。但作者指出，把 egocentric 感知硬塞进 UMI 框架会引入**更大的人-机具身差异（embodiment gap）**：

- **视觉差异（Visual gap）**：人臂与机械臂外观不同；人机身高不同导致 egocentric 视角高低不一。
- **运动学差异（Kinematic gap）**：人与机器人身体形态、颈部自由度不同。直接回归并跟踪双手 + 头部的 6-DoF 轨迹往往产出机器人**不可行**的动作。

因此已有 egocentric 系统要么额外依赖 teleoperation 数据做 action grounding，要么退回固定底座、放弃全身协调。HoMMI 的目标是：**在保留 UMI 可扩展采集优势的前提下，为 egocentric 观测显式弥合具身差异，实现移动操作的直接人到机器人迁移**。

## 二、核心方法

系统三段式：可扩展采集接口（§IV）→ 跨具身 hand-eye 策略（§V）→ 约束感知全身控制器（§VI）。

### 2.1 数据采集接口

用三部 iPhone：两部装在夹爪、一部装在帽子（cap）上，借 Apple ARKit 的多设备协同建立**共享坐标系**；每次演示以 60 Hz 同步记录三路 RGB 视频、深度图、6-DoF 位姿与夹爪开合。夹爪沿用 UMI 的 fin-ray 设计并复刻到机器人上。附录测得 ARKit 相对 MoCap 的跟踪误差在 **5.0 mm 位置 / 0.8° 旋转**以内。

> 用大白话说：一顶帽子 + 两个手持夹爪就是整套"数据手套"，戴上边干活边录，全程不碰机器人。

### 2.2 跨具身 Hand-Eye 策略（三大设计）

主干是 Diffusion Policy（DiT backbone）。每一步以观测窗口 $O_t = o_{t-T_o+1},\dots,o_t$ 为条件，预测动作序列 $A_t = a_{t+1},\dots,a_{t+T_p}$。为跨具身迁移引入三点关键设计：

**(1) 3D 视觉表征弥合视觉差异**（灵感来自 Adapt3R）：把 egocentric 观测**升维到 3D**。对每帧头部相机图像，先得到 pointmap（iPhone 深度采集时给出，机器人端用 stereo depth estimation 估计），patchify + 最近邻下采样使每个 $16\times16$ patch 对应一个 3D 点；再用 **DINOv3 ViT** 提取 patch 特征，与该 3D 点的正弦位置编码拼接，把外观特征绑定到 3D 几何，从而对头部位姿与身高变化鲁棒。关键一步是**遮蔽手臂点**：把 pointmap 变换到左右夹爪坐标系，丢弃 $z<0$ 的点（手臂在夹爪后方），消掉演示者手臂/身体的外观。最后用 attention pooling 聚合所有 token 得到头部观测嵌入 $F_{ego}$。腕部图像 resize 到 $224\times224$，用共享 dinov3-vitb16 的 CLS token 得到 $F_{wrist}$。

> 用大白话说：不让策略去看"人手长什么样"，只让它看"三维空间里有什么"，人手/机械臂的外观差异被从源头抹掉。

**(2) 3D look-at 点头部动作表征弥合运动学差异**：不直接拷贝人头的 6-DoF 位姿，而是把机器人注视放松为一个 **3D look-at 点** $\ell_t \in \mathbb{R}^3$。训练时该点取中心相机射线与场景 pointmap 的交点。推理时头控制器把 $\ell_t$ 转成可行头姿：设当前头位 $c_t$、当前头姿 $R_t^{cur}=\langle x_t, y_t, z_t \rangle$，先求指向

$$\hat{d}_t = \frac{\ell_t - c_t}{\lVert \ell_t - c_t \rVert}$$

再把当前 x 轴投影到与 $\hat{d}_t$ 正交的平面上：

$$x'_t = x_t - (x_t^{\top}\hat{d}_t)\hat{d}_t,\qquad \hat{x}_t = \frac{x'_t}{\lVert x'_t \rVert}$$

补出第三轴 $\hat{y}_t = \hat{d}_t \times \hat{x}_t$，得目标头姿 $R_t = \langle \hat{x}_t, \hat{y}_t, \hat{d}_t \rangle$（当 $\lVert x'_t \rVert$ 趋零时用世界竖直向量替代 $x_t$ 再投影）。

> 用大白话说：策略只说"往哪儿看"（一个空间点），不管"脖子怎么摆"；具体头姿由机器人在自己的关节限制内解出来，避免复制人头姿造出机器人做不到的动作，同时保住主动感知意图。

**(3) gripper-centric 坐标系**：把所有观测与动作（proprioception、动作、头部 pointmap、look-at 点）统一变换到**左夹爪坐标系**，让策略始终在一个以操作器为中心的一致空间里推理。相比会随头动/身高漂移出分布（OOD）的 egocentric 坐标系，这个锚定在操作器上的坐标系显著降低跨具身错配。

动作 23 维：左右夹爪各 9 维位姿（3 维位置 + 6 维旋转，取旋转矩阵前两列）、3 维 look-at 点、2 维夹爪开合，即 $23 = 2\times9 + 3 + 2$。观测历史 $T_o=2$，动作时域 $T_p=32$，20 Hz。

### 2.3 约束感知全身控制器

策略只输出末端位姿与 look-at 点，需要全身控制器解算关节 + 底座动作。用 Mink 实现的**微分全身 IK**，每步解带约束的 QP：

$$\min_{\Delta q \in \mathbb{R}^{n_v}} f(\Delta q) + \lambda \lVert \Delta q \rVert_2^2$$

其中目标 $f(\Delta q) = C_{ee}(\Delta q) + C_{nominal}(\Delta q) + C_{current}(\Delta q) + C_{com}(\Delta q)$，四项分别为：$C_{ee}$ 双臂 SE(3) 末端跟踪（主任务、最高权重）、$C_{nominal}$ 偏向预设"类人"姿态、$C_{current}$ 抑制关节/底座突变、$C_{com}$ 质心保持在底座上方以稳态。约束含配置界 $G_{cfg}$、关节速度界 $G_{joint\text{-}vel}$、底座速度界 $G_{base\text{-}vel}$、碰撞避免 $G_{coll}$，以及躯干直立的等式约束 $A_{upright}\Delta q = 0$。IK 用 daqp 求解，异步跑在 **100 Hz**，桥接 10 Hz 策略环与 500 Hz 机器人控制环；look-at 点从 IK 目标中剥离、单独解算颈部朝向。

> 用大白话说：策略负责"手往哪放、往哪看"，控制器负责在"别摔倒、别自撞、别越关节限、质心压在底座上"这一堆物理约束下把它做出来。

## 三、实验结果

平台为 Rainbow Robotics RB-Y1（7-DoF 双臂、6-DoF 躯干、2-DoF 颈、holonomic 麦轮底座），颈上装一对宽角 stereo 相机，腕上装 RGB 相机，夹爪换成与采集端一致的 fin-ray 指。三个长程双臂移动操作任务各跑 20 rollout。Baseline 含 Wrist-Only（原版 UMI）、RGB-Only（UMI+Ego，直接回归 6-DoF 头姿，类似 ViA）、Head-Only（去掉腕部只用 3D 头观测）、w/o Active Neck（本方法但禁用头动）。

| 任务 | 采集演示数 | Wrist-Only | RGB-Only | Head-Only | w/o Active Neck | **Ours (HoMMI)** |
|---|---|---|---|---|---|---|
| Laundry（抓布→找筐→放筐） | 200 | 20% | 0% | 0% | 75% | **90%** |
| Delivery（搬箱→找推车→放置，6×6 m） | 166 | 15% | 45% | 5% | 55% | **85%** |
| Tablescape（抓垫两边→展开铺平） | 115 | — | — | — | 55% | **80%** |

（Tablescape 的三个 baseline 具体数值论文仅在柱状图中给出，正文未逐一列数，此处从略；其定性失败模式为 Wrist-Only 转动过晚、RGB-Only 压得过重触发 wrench 安全护栏、Head-Only 与垫子失接触。）

鲁棒性 / 泛化附加实验（以 Laundry 为主）：

| 维度 | 设置 | 结果 |
|---|---|---|
| 深度噪声敏感性 | 注入 Gaussian 噪声 std 0/2/10/20 mm | 成功率 90/90/90/**50**%（≤1 cm 基本不掉，更大才退化） |
| 未见物体泛化 | 5 seen + 若干 unseen 物体 | 90.63% ± 12.10% |
| 光照变化 | 4300 / 3370 / 1840 / 810 lux | 93.75% ± 10.83%（最暗 810 lux 降到 75%） |
| 采集者身高迁移 | 167 cm 与 182 cm 两人演示 | 均成功迁移 |

关键结论（论文 F1–F6）：Wrist-only 欠观测全局上下文与双臂协作（F1）；只用头相机也不够、缺腕部局部接触线索会抓不准（F2）；**naive 加 egocentric RGB 在具身错配下反而更差**（RGB-Only 在 Laundry 上 0%，F3）；主动头控确实能有效采集任务相关信息并维持策略可观测性（F4）；跨具身 hand-eye 策略学到聚焦于任务相关物体/接触的干净 egocentric attention（F5）；系统可从不同体型演示者迁移（F6）。

## 四、局限性

- **短观测历史**：策略仅用 $T_o=2$ 步历史，长程任务与部分可观下的失败恢复能力有限，作者点名 longer-term memory 为下一步。
- **纯视觉感知**：无力反馈/触觉，接触密集操作与安全性受限，未来可加 force/tactile 与显式柔顺控制。
- **硬件具身差**：尽管对齐了相机位置与夹爪几何，采集端（iPhone + 手持夹爪）与部署端（机器人）之间仍残留硬件层面的具身差，作者认为更好的软硬件 co-design（乃至生成式硬件设计）可进一步提升可迁移性。
- **深度依赖**：3D 表征依赖 pointmap，深度噪声大于 ~1 cm 时性能明显退化（20 mm 掉到 50%），机器人端 stereo 估计质量是隐性前提。

## 五、评价与展望

**优点**：(1) 把"UMI 手持采集"从固定底座桌面操作推进到 **whole-body 移动操作**，且真正做到"零机器人 teleoperation 数据"，这在数据可扩展性上很有价值；(2) 面对 egocentric 引入的具身差，没有回避而是**逐项拆解**——视觉差用 3D 表征 + 手臂遮蔽 + gripper-centric 坐标系解，运动学差用 look-at 点松弛解，落地差用约束感知全身控制器解，工程上自洽且消融清晰（RGB-Only 0% vs Ours 90% 的对比很说明"表征设计"而非"是否加头相机"才是关键）；(3) look-at 点抽象是本文最漂亮的一招，用"往哪看"替代"6-DoF 头姿"既保住主动感知语义又天然绕开颈部运动学不可行。

**与公开工作的关系**：本文可看作 UMI（[5]）在 egocentric + 移动 + 双臂方向上的系统化延伸；3D 视觉表征沿用 Adapt3R（[34]）思路并叠加 DINOv3（[28]）；RGB-Only baseline 对标直接回归头姿的 ViA（Vision in Action, [37]）与 ActiveUMI / Egomi 一类工作，而其结论恰恰是"直接回归头姿在跨具身下崩溃"。相较需要 co-training / fine-tune 机器人数据的 egocentric human video 方法（Humanplus、EgoMimic 等），HoMMI 主打**完全免机器人数据**。

**开放问题与可能改进**：(1) 三任务各 100–200 条演示、每任务 20 rollout，统计量偏小，泛化到更开放场景的证据仍有限；(2) look-at 单点注视对需要宽视野扫视或多目标切换的任务可能不够，可考虑 look-at 轨迹或视锥表征；(3) pointmap 对深度质量敏感，弱纹理/强反光/室外场景下 stereo 估计可能成为瓶颈，可探索深度无关的 3D 表征或时序融合；(4) 纯 vision + 短历史，接触密集与失败恢复是明确短板，引入触觉与显式记忆是自然方向；(5) 采集端-部署端硬件差仍靠人工对齐，缺乏定量刻画。

## 参考

1. Chi et al. *Universal Manipulation Interface: In-the-Wild Robot Teaching Without In-the-Wild Robots.* RSS 2024.（UMI，本文采集接口的基座）
2. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR 2025.（策略主干）
3. Wilcox et al. *Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning.* CoRL 2025.（3D 视觉表征灵感来源）
4. Xiong et al. *Vision in Action: Learning Active Perception from Human Demonstrations.* CoRL 2025.（直接回归头姿的对照路线，对应 RGB-Only baseline）
5. Siméoni et al. *DINOv3.* 2025.（视觉编码器）
