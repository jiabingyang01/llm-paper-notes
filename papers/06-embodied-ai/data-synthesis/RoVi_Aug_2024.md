# RoVi-Aug：跨本体机器人学习的机器人与视角增广

> **论文**：*RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning*
>
> **作者**：Lawrence Yunliang Chen*, Chenfeng Xu*, Karthik Dharmarajan, Muhammad Zubair Irshad, Richard Cheng, Kurt Keutzer, Masayoshi Tomizuka, Quan Vuong, Ken Goldberg（*等贡献）
>
> **机构**：UC Berkeley、Toyota Research Institute、Physical Intelligence
>
> **发布时间**：2024 年 09 月（arXiv 2409.03403）
>
> **发表状态**：未录用（预印本，PDF 中未标注录用会议）
>
> 🔗 [arXiv](https://arxiv.org/abs/2409.03403) | [PDF](https://arxiv.org/pdf/2409.03403)
>
> **分类标签**：`跨本体学习` `数据增广` `扩散模型图像编辑` `视角鲁棒性` `模仿学习`

---

## 一句话总结

RoVi-Aug 用微调 SAM 分割 + ControlNet 图像到图像的"机器人换脸"（Robot-to-Robot, R2R）+ ZeroNVS 视角合成三段式流水线，把单一机器人的示教数据离线合成为多机器人、多相机视角的训练数据，使策略零样本迁移到未见过的机器人和相机姿态上成功率从 0% 大幅提升（如 Franka→UR5 Open Drawer 任务从 0% 提升到 90%），并使多机器人多任务协同训练策略的下游微调效率提升最高约 30%。

## 一、问题与动机

Open-X Embodiment（OXE）等跨数据集联合训练已证明可以带来正迁移，但数据集在机器人型号和相机视角分布上严重不均衡（被 Franka、xArm 等少数机器人主导），导致策略容易对训练时见过的机器人外观和相机视角过拟合，换一个机器人或稍微挪动相机就性能骤降。

已有的测试时自适应方法 Mirage（cross-painting）通过精确的机器人 URDF 和相机内外参，把未见目标机器人"渲染替换"成源机器人的样子来复用源策略，但存在三个局限：(1) 需要精确的机器人模型和相机标定矩阵；(2) 不允许对策略做微调；(3) 由于深度重投影误差，只能处理较小的相机姿态变化。

RoVi-Aug 的核心思路是把跨本体迁移从"测试时补丁"前移到"训练时数据增广"：显式合成机器人种类 × 相机视角的叉乘组合数据，让策略在训练阶段就学到对视觉外观和视角变化的不变性，从而不依赖精确标定、支持后续微调，且能处理更大的视角偏差。

## 二、核心方法

流水线分为两个正交模块：机器人增广（Ro-Aug）和视角增广（Vi-Aug），设 $D_i^S=\{o_1^S,...,o_{H_i}^S\}$ 为源机器人 $S$ 的一段图像序列，目标是生成合成目标机器人 $\mathcal{T}$（甚至新视角）下的观测序列，再用行为克隆训练策略 $\pi(a_t \mid o_t^{\mathcal{T}}, p_t)$。

**Ro-Aug（机器人增广）三步走**：

1. **机器人分割**：现成分割模型（SAM 等）在机器人图像上表现不佳，作者用 Robosuite 仿真器随机采样机器人姿态和相机姿态、批量生成"机器人+分割掩码"合成数据（4 种机器人 Franka/UR5/Sawyer/Jaco，各约 80 万张图），并做亮度增广、贴到 ImageNet 背景图上模拟真实场景，再用 LoRA 微调冻结的 SAM 得到能应对多样机器人/相机姿态的分割模型。
2. **R2R（Robot-to-Robot）生成**：把分割出的源机器人区域，通过一个基于 Stable Diffusion v1.5 微调的 ControlNet（每对机器人单独训练一个，同样用仿真配对图像训练：同一姿态下渲染两种机器人）转换成目标机器人的样子。直觉是把跨本体的视觉迁移当成一个图像到图像翻译问题，绕开 Mirage 需要的精确相机标定。
3. **机器人补绘（Robot Inpainting）**：用视频补全模型 E2FGVI 修复原图中被抠掉机器人后留下的空洞背景，再把 R2R 生成的目标机器人贴回背景。为缓解"仿真训练 R2R 模型 → 真实测试图像"的光照域差，对生成的机器人做 HSV 空间的随机亮度扰动（$\pm30$，值通道），实验证明这一步对最终策略性能影响显著。

**Vi-Aug（视角增广）**：使用 ZeroNVS——一个可从单张图零样本合成 360° 新视角、且能处理带复杂背景的多物体场景（不要求先分割出物体）的 3D 感知扩散模型。对轨迹中每张图 $o_t$ 采样一个 $SE(3)$ 扰动 $(\tilde R_t,\tilde T_t)$（平移分量按区间均匀采样，旋转用欧拉角参数化），生成 $o_t^{\tilde R,\tilde T}=f(o_t;\tilde R,\tilde T)$。实现中平移范围取 $\tilde T_{x},\tilde T_{z}\in(-0.25\text{m},0.25\text{m})$，垂直方向 $\tilde T_{y}\in(-0.1\text{m},0.1\text{m})$（更小，因为竖直方向移动过大会让 ZeroNVS 产生明显伪影），每个欧拉角在 $\pm0.1$ 弧度内采样。实验对比了"整条轨迹用同一个扰动（consistent）" vs "逐帧独立采样（inconsistent）"两种策略。

用大白话说：Ro-Aug 相当于给同一段操作视频"换主角机器人"，Vi-Aug 相当于给这段视频"换摄像机机位"；两者都是离线、数据侧的增广，训练完策略天然对机器人外观和相机视角具有一定不变性，不需要在部署时再做任何图像变换或已知相机标定。

策略骨架采用 Diffusion Policy（基于 DROID 代码库），ResNet-18 视觉编码器 + 1D-UNet，共约 80M 参数，输入 128×128 降采样图像和末端位姿，预测 16 步动作序列、执行 8 步后重新推理。

## 三、关键结果

物理实验在 Franka 与 UR5 之间做双向迁移，5 个任务（Open Drawer / Place Tiger / Stack Cup 采集自 Franka，Sweep Cloth / Transport Tiger 采集自 UR5，每任务 150 条遥操作轨迹，10 次试验评估）。

**零样本跨机器人迁移（相同相机姿态，Table 1）**：

| 方法 | Open Drawer | Place Tiger | Stack Cup | Sweep Cloth | Transport Tiger |
|---|---|---|---|---|---|
| 无增广 | 0% | 0% | 0% | 0% | 40% |
| Mirage | 60% | 90% | 50% | 100% | 70% |
| Ro-Aug | 90% | 80% | 30% | 100% | 80% |
| Ro-Aug（去掉亮度随机化） | 90% | 50% | 10% | 40% | 60% |

Ro-Aug 达到与 Mirage 相当的零样本性能，但去掉亮度增广后性能明显下滑，验证了该项设计的必要性。在需要高精度对齐的 Stack Cup 任务上 Mirage 仍略优（得益于精确 URDF+相机标定），但 Mirage 无法微调。

**小样本微调（Table 2）**：在 Ro-Aug 数据上预训练后用目标机器人 5/10 条示教微调，全面超过同等示教量下从零训练的策略，例如 Franka→UR5 Stack Cup 从 10-shot 的 50% 提升到 Ro-Aug+10-shot 的 80%；且全面超过 Table 1 中所有零样本基线。

**相机视角鲁棒性（Table 3，Place Tiger 任务）**：相机平移+旋转扰动越大（10cm/20°→25cm/35°→40cm/45°），策略在大幅视角变化下的成功率越高，但代价是原始视角下性能略降；逐帧独立采样（inconsistent）略优于整条轨迹一致采样（consistent），最终选用 25cm 扰动范围 + 逐帧独立采样作为默认配置。

**机器人+视角联合增广（RoVi-Aug，Table 4，不同机器人+不同相机角度同时变化）**：

| 方向/任务 | 视角偏移 | Mirage | Ro-Aug | RoVi-Aug |
|---|---|---|---|---|
| Franka→UR5 Open Drawer | 10cm,20° / 25cm,35° | 50% / 30% | 60% / 20% | 80% / 50% |
| Franka→UR5 Place Tiger | 10cm,20° / 25cm,35° | 30% / 20% | 30% / 10% | 70% / 30% |
| UR5→Franka Sweep Cloth | 10cm,20° / 25cm,35° | 80% / 30% | 0% / 0% | 80% / 40% |
| UR5→Franka Transport Tiger | 10cm,20° / 25cm,35° | 20% / 0% | 40% / 40% | 40% / 30% |

只做机器人增广（Ro-Aug）在相机角度也变化时明显不够，Mirage 依赖已知相机姿态在大偏移下也会失效；只有同时做机器人+视角增广的完整 RoVi-Aug 在几乎所有设置下取得最佳或接近最佳成绩。

**多机器人多任务与下游微调效率**：将 Franka 的 Place Tiger 数据、UR5 的 Transport Tiger 数据及其互相的机器人增广版本混合训练一个多机器人多任务策略，两个机器人在两个任务上都能成功执行（Table 5：Franka 上 Place Tiger 80%/Transport Tiger 60%，UR5 上 Place Tiger 70%/Transport Tiger 80%）。将 RoVi-Aug 应用到 OXE 中的 Berkeley UR5 数据集并微调 Octo 生成式策略，用目标机器人（Franka）仅 50 条示教微调，相比 Octo-Base 直接微调，成功率从 30%/20% 提升到 60%/40%（Sweep Cloth / Transport Tiger，Table 6），说明预训练阶段见过合成的目标机器人形态能加速下游微调收敛。

作者还在附录展示了一个定性实验：将 Franka 图像跨绘制为 Boston Dynamics Spot（移动机器人）的效果，视觉上较逼真，但未做真实硬件验证。

## 四、评价与展望

RoVi-Aug 把"测试时跨绘制"（Mirage）的思路系统性地前移到训练时数据增广，在不需要精确相机标定和机器人 URDF 的前提下取得与 Mirage 相当甚至更好的零样本迁移效果，并额外获得了 Mirage 不具备的两个能力：支持策略微调、支持大幅度相机视角变化下的鲁棒性。这一设计使其相比纯粹依赖测试时图像变换的方法（Mirage）更适合融入标准的模仿学习/大规模预训练流程，与 RoboAgent、GenAug 等物体/背景/任务侧的生成式数据增广工作是互补关系而非竞争关系。

局限也很明显：整条流水线由分割、R2R 生成、视频补绘、视角合成四个独立模型串联而成，误差会级联放大（如物体分割不准会导致 R2R 生成机器人形变或幻觉），论文附录也系统列出了光照差异、分割不准、时序不一致、R2R 生成伪影四类典型 artifact；R2R 模型需要对每一对机器人单独训练，扩展到更多机器人型号时组合数量爆炸，未来需要统一多机器人对的生成模型；视角合成模型 ZeroNVS 是现成模型未在机器人数据上微调，效果有提升空间（论文提出可考虑基于视频扩散模型的替代方案）；只处理相机视角变化未处理背景变化，也只验证了固定基座机械臂之间的迁移，未涉及夹爪形态差异较大（如多指手）或移动机器人的真实迁移。此外在需要高精度末端对齐的任务（如 Stack Cup）上，生成式方法目前仍不及基于精确几何模型的 Mirage，说明"生成式跨本体"与"几何式跨本体"在精度-通用性上仍有权衡，是一个开放的改进方向。

## 参考

- Chen et al. *Mirage: Cross-embodiment Zero-shot Policy Transfer with Cross-painting*, RSS 2024（本文对比的核心测试时基线）
- Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, ICRA 2024（本文使用的跨本体数据集背景）
- Sargent et al. *ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Real Image*（本文视角增广所用的 3D 感知扩散模型）
- Zhang et al. *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)*（R2R 生成模型的骨架）
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*（本文策略架构基础）
