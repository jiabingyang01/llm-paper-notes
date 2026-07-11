# X-Diffusion：在跨本体人类演示上训练扩散策略

> **论文**：*X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations*
>
> **作者**：Maximus A. Pace\*, Prithwish Dan\*, Chuanruo Ning, Atiksh Bhardwaj, Audrey Du, Edward W. Duan, Wei-Chiu Ma†, Kushal Kedia† et al.（\* 共同一作，† 共同通讯/指导）
>
> **机构**：Cornell University
>
> **发布时间**：2025 年 11 月（arXiv 2511.04671）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.04671) | [PDF](https://arxiv.org/pdf/2511.04671)
>
> **分类标签**：`cross-embodiment` `human-video` `diffusion-policy` `ambient-diffusion` `imitation-learning`

---

## 一句话总结

X-Diffusion 把"人类演示"看成"机器人动作的高噪声版本"——沿前向扩散加噪时本体特有的执行差异会先被抹掉、任务级引导却保留，于是借助 Ambient Diffusion 训练一个"人-机分类器"逐条估计每个人类动作变得与机器人不可区分的**最小噪声步** $k^\star$，只在 $k \ge k^\star$ 的高噪声区间用人类动作监督扩散策略，从而在吸收海量人类视频引导的同时滤除机械臂无法执行的动作细节；5 个真实操作任务上平均成功率比朴素协同训练与人工过滤高约 **16%**。

## 一、问题与动机

人类视频是可规模化的机器人学习数据源（采集比遥操作快得多），但**人和机器人的本体（embodiment）差异巨大**，很多人类动作对机器人来说物理上不可执行。典型例子（论文 Fig. 1）：搬盘子时人可以灵巧地把手指滑到盘子底下抠起来，而平行夹爪机器人更可靠的策略是从侧面推/滑盘子。这带来一个核心两难：

- 若像 naive co-training 那样把所有人类数据（不管可不可执行）都当成机器人动作直接拟合，就会把不可行的执行策略（如侧向抓取、手指托底）灌进策略，**造成负迁移、反而不如只用机器人数据**；
- 若像一些方法那样先把人类演示用逆运动学(IK)回放到机器人上、人工过滤掉失败轨迹，则会**整条丢弃**大量含有有用任务引导的演示（作者实测约一半人类演示回放会运动学/动力学失败）。

作者的目标是：**只吸收人类演示里"该做什么、和哪些物体怎么交互"的粗粒度引导，而不迁移本体特有、不可执行的执行细节**。他们注意到生成建模里有一个高度类似的问题——如何在混有低质量数据时不损害模型——并借用最近的 Ambient Diffusion 思路：低质量数据只在扩散过程的高噪声步参与训练。

关键洞察（贯穿全文）：**把人类动作视为机器人动作的"带噪副本"**。前向扩散不断加高斯噪声，会渐进地抹去动作中本体特有的高频细节；当噪声足够大时，人类轨迹与机器人轨迹在分布上变得**不可区分**，此时残留的只是任务级结构。于是"在低噪声步精确监督、在高噪声步才引入人类数据"就能既拿到引导又不牺牲可执行性。

## 二、核心方法

### 2.1 统一状态与动作表示

沿用 Motion Tracks / Point Policy 的思路，把跨本体数据映射到共享空间：状态 $s_t = (q_t, o_t)$，动作序列 $\mathbf{A}_t = a_{t:t+S}$ 预测未来 $S$ 步，其中 $a_t = q_{t+1}$。本体状态 $q_t \in \mathbb{R}^7$ 含末端执行器 3D 位置、旋转、夹爪开合。

- 对人类演示：假设单手、起始为张开抓握、有两台标定 RGB 相机。用 **HaMeR** 检测每帧 2D 手部关键点并三角化到 3D 机器人坐标系；抓取点取拇指与食指指尖中点，朝向由拟合局部手坐标系后重定向到末端执行器得到，夹爪开合由拇指-食指间距推断。
- 为缩小视觉域差异：用 **Grounded SAM 2** 把任务相关物体分割成彩色 mask，并在每帧叠加末端执行器位姿的关键点渲染（Fig. 2），策略输入即"这张 masked 图 + 本体状态"。

**用大白话说**：先把"人手视频"翻译成"机械臂末端在动"的统一语言，再把画面抽象成"彩色物体轮廓 + 一个虚拟夹爪骨架"，让人和机器人的画面尽量长得一样，只留任务信息。

### 2.2 前向扩散与"最小不可区分步" $k^\star$

扩散策略学习的是给动作序列去噪。给定干净动作 $\mathbf{A}_t^0$，前向过程逐步加噪：

$$q(\mathbf{A}_t^{k+1} \mid \mathbf{A}_t^k) = \mathcal{N}\!\left(\sqrt{1-\beta_k}\,\mathbf{A}_t^k,\ \beta_k I\right)$$

设 $p_H^k, p_R^k$ 分别为第 $k$ 步噪声下人类、机器人动作的分布。类比 Ambient Diffusion 里的 $\epsilon$-merging，定义**最小不可区分步 (minimum indistinguishability step)**：

$$k^\star = \min\left\{k \mid D_{KL}(p_H^k \Vert p_R^k) \le \epsilon\right\}$$

**用大白话说**：$k^\star$ 是"加噪加到多脏时，人类动作和机器人动作就分不清了"的那个临界步。只要噪声超过 $k^\star$，人类动作已被抽象得像机器人动作，就可以安全地用来监督机器人策略，而不会把不可执行的动作细节传过去。

### 2.3 训练人-机噪声动作分类器

$k^\star$ 无法直接计算，作者训练一个分类器 $c_\theta(\cdot \mid k, \mathbf{A}_t^k, s_t)$，输入"扩散步 $k$、加噪动作序列、当前状态"，输出该动作源自机器人($y=1$)而非人类($y=0$)的概率，用二元交叉熵训练：

$$\mathcal{L}_{\text{class}}(\theta) = \mathbb{E}_{(k,\mathbf{A}_t^k,s_t)\sim\mathcal{D}_R}\!\left[-\log c_\theta(k,\mathbf{A}_t^k,s_t)\right] + \mathbb{E}_{(k,\mathbf{A}_t^k,s_t)\sim\mathcal{D}_H}\!\left[-\log\!\left(1-c_\theta(k,\mathbf{A}_t^k,s_t)\right)\right]$$

由于人类数据远多于机器人数据（$\lvert\mathcal{D}_H\rvert \gg \lvert\mathcal{D}_R\rvert$），两边等概率采样以免偏向"人类"标签。随后对每条人类动作序列，把最小不可区分步实例化为"分类器首次给出 $\ge 0.5$ 机器人概率"的扩散步：

$$k^\star(\mathbf{A}_t) = \min\left\{k \mid c_\theta(k,\mathbf{A}_t^k,s_t) \ge 0.5\right\}$$

**用大白话说**：训练一个"侦探"去分辨这段加噪动作到底是人做的还是机器人做的。噪声小时侦探一眼看穿是人；噪声一大到某步侦探开始蒙圈（认成机器人的概率过半），那一步就是这条演示的 $k^\star$。可行的人类动作（如顶部抓取，本就贴近机器人分布）$k^\star$ 很低，不可行的（如侧向抓取）要加很多噪声才混淆，$k^\star$ 很高（Fig. 3）。

### 2.4 选择性把人类数据注入扩散策略损失

朴素协同训练在**所有**噪声步都用人类数据监督（等价于 Eq. 1 的 co-train BC 损失），会逼策略朝不可行动作去噪。X-Diffusion 改成：机器人动作在所有噪声步监督，人类动作只在 $k \ge k^\star(\mathbf{A}_t)$ 时才进损失：

$$\mathcal{L}_{\text{X-DP}}(\theta) = \mathbb{E}_{(k,\mathbf{A}_t,s_t)\sim\mathcal{D}_R}\,\ell(\pi_\theta,\mathbf{A}_t^k) + \mathbb{E}_{(k,\mathbf{A}_t,s_t)\sim\mathcal{D}_H}\,\mathbb{1}_{\{k \ge k^\star(\mathbf{A}_t)\}}\,\ell(\pi_\theta,\mathbf{A}_t^k)$$

**用大白话说**：低噪声（细节层）只信机器人自己的 5 条示范，保证动作可执行；高噪声（大方向层）才让海量人类演示进来出主意。每条人类演示"从多脏的噪声起才有资格发言"由它的 $k^\star$ 自适应决定——越不可行的演示，越只能在很粗糙的层面上提供引导。相比"整条丢弃不可行演示"的人工过滤，它能从不可行演示里也榨出高层任务信号。

## 三、实验结果

**设置**：7-DOF Franka Emika Panda；每个任务采 **5 条机器人演示 + 100 条人类演示**；5 个真实任务 Close Drawer / Pan On Plate / Push Plate / Mug On Rack / Bottle Upright；每方法每任务 10 次真机 rollout 报平均成功率。

**对比基线**：Diffusion Policy（仅 5 条机器人数据）、Point Policy（人机数据全量协同，状态用 DIFT + Co-Tracker 关键点）、Motion Tracks（全量协同，动作统一为关键点但用原始图像观测）、DemoDiffusion（前 60% 反扩散步用人类策略、后 40% 用机器人策略）。

### 3.1 与跨本体基线对比（Fig. 4，成功率为柱状图近似读数）

| 任务 | Diffusion Policy | Point Policy | Motion Tracks | DemoDiffusion | **X-Diffusion** |
|---|---|---|---|---|---|
| Close Drawer | ~0.90 | ~0.90 | ~0.95 | ~0.65 | **~0.90** |
| Pan On Plate | ~0.40 | ~0.10 | ~0.60 | ~0.40 | **~0.85** |
| Push Plate | ~0.40 | ~0.10 | ~0.50 | ~0.50 | **~0.75** |
| Mug On Rack | ~0.50 | ~0.10 | ~0.50 | ~0.55 | **~1.00** |
| Bottle Upright | ~0.10 | ~0.05 | ~0.20 | ~0.10 | **~0.30** |

（上表为 Fig. 4 柱状图目测读数，非原文表格精确值；论文明确的总体结论是 X-Diffusion 在全部 5 任务上成功率最高，平均比朴素协同训练与人工过滤基线高 **16%**。）关键定性发现：naive 协同（Motion Tracks/DemoDiffusion）相对纯机器人几乎无提升，Point Policy 甚至因学到次优动作而**负迁移**；在 Push Plate / Pan On Plate 上多条人类演示采用侧向抓取（机器人不可行），基线会照单全收（Fig. 5：桌面碰撞、不可行抓取），X-Diffusion 则通过分类器把这些动作限制到高噪声区间。

### 3.2 数据用法系统消融（Fig. 6）

作者构造 FILTERED 数据集：把人类演示用 IK 回放到机器人、人工剔除失败轨迹（约**一半**演示因运动学/动力学失败被丢弃），对比四种用法：

| 任务 | Robot Only | Naive | Filtered | **X-Diffusion** |
|---|---|---|---|---|
| Mug On Rack | ~0.50 | ~0.50 | ~0.70 | **~1.00** |
| Pan On Plate | ~0.40 | ~0.40 | ~0.60 | **~0.85** |
| Push Plate | ~0.40 | ~0.20 | ~0.50 | **~0.75** |

（Fig. 6 柱状图近似读数。）结论：Filtered 优于 Naive，印证"在不可行人类演示上训练会损害策略"；而 X-Diffusion 在**全部**任务上又优于 Filtered——因为它不整条丢弃，而是把成功演示在各噪声步都用、不可行演示也在高噪声区间贡献粗引导，从不可行数据里也提取到信号。

### 3.3 迁移量化（Fig. 7）

- **左**：随前向扩散噪声升高，X-Diffusion 纳入训练的人类数据比例上升；不同任务差异明显——Mug On Rack、Pan On Plate 全程纳入更大比例人类数据，Bottle Upright 纳入较少。
- **右**：相对纯机器人基线的性能增益是**任务相关的正迁移**——人类演示与机器人动力学越吻合、纳入越多，增益越大（Mug On Rack 增益最大，Bottle Upright 因人类演示动力学兼容性差、纳入少而增益最小）。作者强调：X-Diffusion 的迁移**始终为正**，而基线常出现负迁移。

## 四、局限性

1. **依赖标定的多相机环境**：人手 3D 重建需要两台已标定 RGB 相机 + HaMeR，尚不能直接吃互联网上无标定的野外单目视频；作者明确把"大规模、非结构化互联网视频"列为未来工作。
2. **数据与任务规模有限**：仅 5 个桌面单臂任务、每任务 5 条机器人 + 100 条人类演示，10 次 rollout，属小样本真机验证，统计噪声大、未做双臂/长程/灵巧手。
3. **对统一动作表示的强假设**：单手、起始张开抓握、抓取点=拇指食指中点、夹爪由指距推断——本质上把人手退化成平行夹爪，丢弃了人类多指灵巧性，对需要指内操作(in-hand)的任务不适用。
4. **分类器质量决定上限**：$k^\star$ 由学习到的分类器 0.5 阈值定义，分类器过/欠敏感会直接错配人类数据的注入噪声层；论文未报告分类器本身精度或 $k^\star$ 的敏感性分析，$\epsilon$/阈值 0.5 的选择也偏经验。
5. **成功率报告以柱状图为主**：正文缺少带方差/置信区间的数值表，"16%" 为平均口径，难以逐任务复核显著性。

## 五、评价与展望

**优点**：这篇工作最漂亮的地方是给"跨本体人类数据到底怎么用"提供了一个**连续、自适应、按噪声分级**的答案，而不是二值化的"用/不用"或"过滤/保留"。它把 Ambient Diffusion 里"低质量数据只在高噪声步参与"的思想干净地映射到 imitation learning：噪声轴天然成了"从执行细节到任务意图"的抽象梯度，人类演示可行度越低就被自动推到越粗的引导层。相比 Point Policy / Motion Tracks 等"全量协同"和"IK 回放 + 人工过滤"，它在概念上更优雅、实验上一致占优且始终正迁移。分类器逐条给出 $k^\star$ 的做法也让"不可行演示也能贡献高层信号"成为可能，回收了人工过滤会浪费掉的约一半数据。

**与其他公开工作的关系**：方法根基是 Daras 等的 Ambient Diffusion 系列（NeurIPS'23 及 Ambient Diffusion Omni, NeurIPS'25）；跨本体表示直接站在 Motion Tracks（ICRA'25）与 Point Policy（CoRL'25）肩上；与 DemoDiffusion（ICRA'26）同属"在扩散/去噪过程里做人机融合"，但 DemoDiffusion 是推理期按固定 60/40 切换策略、不训练机器人去噪人类动作，X-Diffusion 则在训练期按学习到的 $k^\star$ 分级注入，思路更细。与 Phantom / Masquerade / ZeroMimic 等"渲染机器人臂覆盖人类视频"的视觉对齐路线正交——本文核心创新在**动作分布层面**而非像素层面。

**开放问题与可能改进**：(1) 去标定化——能否用单目手部估计 + 尺度/深度先验替代双标定相机，是走向互联网视频的关键瓶颈；(2) $k^\star$ 的可靠性——把 0.5 硬阈值换成软加权（按分类器概率连续加权人类损失）或对 $k^\star$ 做校准/不确定性估计，可能更稳；(3) 表示局限——统一到平行夹爪丢掉了多指信息，若目标含灵巧手可探索保留更高维手部动作再分级去噪；(4) 规模化验证——在更大人类数据集与更多任务上检验"噪声分级注入"是否仍带来单调正迁移，以及分类器在分布外人类演示上的泛化。总体上，这是一个机制清晰、实证扎实的小而美工作，主要价值在于为"人类视频预训练机器人策略"给出了一条可操作的、按可执行度分级利用数据的范式。

## 参考

1. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. IJRR 2024.（去噪动作策略基座）
2. Daras et al. *Ambient Diffusion: Learning Clean Distributions from Corrupted Data*. NeurIPS 2023；及 *Ambient Diffusion Omni: Training Good Models with Bad Data*. NeurIPS 2025.（低质量数据分噪声步注入的方法根基）
3. Ren et al. *Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning*. ICRA 2025.（统一动作表示与协同训练基线）
4. Haldar & Pinto. *Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation*. CoRL 2025.（关键点协同训练基线）
5. Park et al. *DemoDiffusion: One-Shot Human Imitation using Pre-trained Diffusion Policy*. ICRA 2026.（推理期人机去噪融合对照）
