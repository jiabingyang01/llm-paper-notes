# ManipDreamer3D：基于占据感知3D轨迹的可信机器人操作视频合成

> **论文**：*ManipDreamer3D: Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory*
>
> **作者**：Ying Li, Xiaobao Wei, Xiaowei Chi, Yuming Li, Zhongyu Zhao, Hao Wang, Ningning Ma, Ming Lu, Sirui Han, Shanghang Zhang
>
> **机构**：北京大学多媒体信息处理国家重点实验室（计算机学院）/ 香港科技大学 / NIO 蔚来汽车自动驾驶研发部 / 北京大学软件与微电子学院
>
> **发布时间**：2025 年 09 月（arXiv 2509.05314，v2 于 2025 年 11 月 13 日更新）
>
> **发表状态**：论文页脚版权声明为 "Copyright © 2026, Association for the Advancement of Artificial Intelligence"，据此应为 AAAI 2026 录用稿
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.05314) | [PDF](https://arxiv.org/pdf/2509.05314)
>
> **分类标签**：`机器人操作视频生成` `3D占据栅格` `轨迹规划` `CHOMP优化` `扩散模型条件控制`

---

## 一句话总结

先用单张第三视角图像重建 3D 占据栅格，再以 CHOMP 式梯度优化规划出无碰撞、短路径、速度符合物理规律的 3D 末端执行器轨迹，最后通过零额外参数的隐空间掩码编辑把物体与夹爪轨迹注入 SVD/CogVideoX 扩散模型来合成操作视频；DiT 版本在 FVD 上从 RoboMaster 的 147.31 降到 93.98，轨迹跟随误差降到 15.38（机器人）/16.59（物体）。

## 一、问题与动机

真实机器人操作演示数据的采集成本高、受硬件限制，是训练可泛化操作策略的关键瓶颈。扩散模型合成操作视频是缓解数据稀缺的一条路径，但其效果高度依赖控制指令的精确性与合理性。作者指出现有轨迹条件视频生成方法（This&That、RoboMaster、DragAnything 等）存在两个核心缺陷：

1. **忽视机器人动作本质上规划于 3D 空间**：现有方法主要以 2D 轨迹作为控制信号，天然存在 3D 空间歧义，生成的轨迹可能违反物理约束、缺乏避障能力和执行效率，且仍依赖人工选定物体（如 ORV、This&That、RoboMaster）。
2. **生成场景与真实世界几何/物理不一致**：即使 2D 视频感知质量尚可，物体尺寸、位置、接触状态的不准确也会导致操作交互不真实，限制了这类视频数据用于训练可泛化策略的价值。

针对以上问题，作者提出 ManipDreamer3D：给定单张第三视角观测图与文本指令，先重建场景 3D 占据表示、规划一条物理合理（避障、短路径、合理速度分布）的末端执行器轨迹，再据此驱动视频扩散模型，实现关键点、全轨迹、抓取部位（affordance）三级细粒度控制（对比 This&That 仅支持关键点控制、RoboMaster 支持关键点+全轨迹但不支持 affordance 控制）。

## 二、核心方法

ManipDreamer3D 分四个模块：3D 占据地图重建 → 3D 轨迹规划与时间重分配 → 训练数据的 3D 轨迹标注流水线 → 轨迹引导的视频合成。

**1）3D 占据地图重建。** 利用 VGGT（视觉几何 grounded transformer）从单张第三视角图像重建初始相机坐标系点云；由于遮挡区域点云不连续，用神经表面重建技术恢复连续表面并重新均匀采样得到稠密点云；最终离散化为 $64\times64\times64$ 的占据栅格 $O\in\mathbb{R}^{h\times w\times d}$，每个体素表示该处是否有物质占据，兼顾效率与精度。

**2）三阶段 A\* 初始轨迹 + CHOMP 式优化。** 先用 A\* 算法在占据栅格中分三段规划初始轨迹：接近阶段 $P_1$（末端执行器→物体）、操作阶段 $P_2$（抓取移动到目标位置）、返回阶段 $P_3$（末端执行器回到起始位置）。随后对每段轨迹用 Adam（学习率 0.1）联合优化四类损失，起止点固定不参与优化：

$$\mathcal{L}_{col}=\sum_{i=1}^{N}\big(\mathrm{SDF}(p_i)\big)$$

用大白话说：以场景静态背景构建的有符号距离场（SDF）惩罚轨迹点离障碍物太近，实现避障。

$$\mathcal{L}_{len}=\sum_{i=1}^{N-1}\lVert p_i-p_{i+1}\rVert^2$$

用大白话说：让路径尽量短，提升操作效率。

$$\mathcal{L}_{acc}=\frac12\sum_{i=1}^{N-2}\lVert p_{i+2}-2p_{i+1}+p_i\rVert^2$$

用大白话说：惩罚加速度突变，避免机械臂猛冲猛停。

$$\mathcal{L}_{cur}=\frac12\sum_{i=1}^{N-2}\left(\frac{\lVert v_i\times a_i\rVert^2}{\lVert v_i^3\rVert^2+\epsilon}\right)$$

用大白话说：惩罚路径曲率过大（急转弯），$v_i,a_i$ 分别是相邻点算出的速度、加速度近似量，$\epsilon=10^{-6}$ 防止分母为零。四项损失加权求和后联合优化，得到既短又平滑、可避障的最终路径 $P^3_{opt}$。

**3）路径感知时间重分配（Path-aware Time Reallocation）。** 上述几何优化未考虑真实机械臂运动学特性——原始轨迹点的速度分布不符合真实机器人"先加速后减速"的运动规律。作者提出后处理策略：按每段子轨迹的弧长比例重新分配采样点数目，再沿路径按预设速度剖面（默认用正弦波）插值重新定位各点，使最终轨迹的空间分布更贴合真实运动学（论文 Fig. 3 定性展示了处理前后速度分布从"锯齿状"变为平滑的正弦形）。

**4）训练数据的 3D 轨迹标注流水线。** 为训练轨迹条件视频扩散模型，需从已有操作视频中反标注 3D 轨迹：用 VGGT 逐帧重建整段视频的时序一致点云与相机参数；用针对夹爪指尖微调的 YOLO 检测末端执行器，两指检测框中心点的中点作为夹爪 3D 位置；沿用 This&That 的做法，用抓取起始时刻夹爪 2D 中心点作为物体位置的初始线索，结合从指令中解析出的物体名称，通过 Qwen-VL 做视觉定位，再用 SAM 生成精确物体掩码。

**5）轨迹引导的视频合成。** 先做 3D→2D 投影：假设抓取过程中相机—物体距离近似恒定（借助已知末端执行器 3D 轨迹估计距离变化），将物体和夹爪都建模为球体（半径分别取物体包围盒最大边长、预设固定小半径），用标准透视投影把 3D 球投影为 2D 圆（圆的尺度隐含了距离信息），得到逐帧物体/夹爪掩码。随后做隐空间编辑：用视频扩散模型的 VAE 编码首帧得到初始 latent；对物体/夹爪掩码区域做池化得到代表性 latent 向量（夹爪额外加偏置区分开合状态，沿用 RoboMaster 的做法），逐帧叠加到首帧 latent 上构造出"动态视频 latent"。与 This&That 需要 ControlNet、RoboMaster 需要额外时空卷积注入模块不同，本文直接把构造出的动态 latent 沿通道维度与噪声视频 latent 拼接，替代传统的"重复静态首帧"条件，不引入任何新参数即可实现精确空间控制。该框架分别在 UNet 架构 SVD 与 DiT 架构 CogVideoX-5B（4x 时间压缩 VAE，需额外做 2 层 2x 时域平均池化对齐）上实现，对应 ManipDreamer3D(SVD) 与 ManipDreamer3D(DiT) 两个版本。

## 三、关键结果

**训练数据**：整合 Bridge V1 与 Bridge V2 数据集，经上述标注流水线筛选得到 8.7k 条有效 episode，按 9:1 划分训练/测试集。

**视频质量与轨迹精度**（Table 2，指标：FVD↓、PSNR↑、SSIM↑，以及末端执行器/物体的轨迹误差 TrajError↓）：

| 方法 | 类型 | FVD↓ | PSNR↑ | SSIM↑ | TrajError$_{robot}$↓ | TrajError$_{obj}$↓ |
|---|---|---|---|---|---|---|
| DragAnything | SVD | 158.42 | 21.13 | 0.792 | 18.97 | 27.41 |
| This&That | SVD | 148.69 | 20.93 | 0.758 | 62.07 | 37.12 |
| **ManipDreamer3D (SVD)** | SVD | **143.33** | **22.75** | **0.807** | **17.40** | **18.77** |
| RoboMaster | DiT | 147.31 | 21.55 | 0.803 | 16.47 | 24.16 |
| **ManipDreamer3D (DiT)** | DiT | **93.98** | **23.64** | **0.847** | **15.38** | **16.59** |

DiT 版本相比同骨干的 RoboMaster，FVD 从 147.31 大幅降至 93.98（降幅约 36%），SSIM 从 0.803 升至 0.847。作者将轨迹精度优势归因于对夹爪—物体协同表征的显式建模：This&That 仅依赖两个模糊的关键手势点，DragAnything 只在整体实体层面建模缺乏末端执行器精度，RoboMaster 隐式建模末端执行器位置，均不如本文显式投影两者 3D 轨迹精确。

**VBench 感知质量指标**（Table 3，节选）：ManipDreamer3D(SVD) 在 Temporal Flickering（97.98）、Motion Smoothness（98.47）、Subject Consistency（95.39）、Background Consistency（96.57）上均为 SVD 组最优；但 Aesthetic Quality（52.46）与 Imaging Quality（69.24）低于 This&That（57.27 / 70.09），说明本文方法在时序一致性与轨迹保真度上占优，但绝对画面美学质量并非全面领先。DiT 版本在 Motion Smoothness（98.70）和一致性指标上同样优于 RoboMaster，但 Temporal Flickering（98.18 vs 98.27）略逊。

**消融/定性分析**：(1) Fig. 5 定性对比显示 ManipDreamer3D 更好保持物体原始形状，This&That 存在明显形变；(2) affordance 精细控制实验（Fig. 7）通过在同一物体（锅）上变换抓取接触点，验证模型能按指定抓取部位生成对应操作视频；(3) 轨迹优化消融（Fig. 6，"Take pumpkin out of sink"）表明使用优化后轨迹 $P^3_{opt}$ 生成的视频会主动上移避开水槽边缘碰撞，而未优化的初始轨迹 $P^3_{init}$ 会贴着障碍物产生不安全路径，验证了轨迹优化对下游视频（进而下游 VLA 训练）安全性的重要性。

## 四、评价与展望

**优点**：把机器人操作轨迹生成从纯 2D 像素空间提升到显式 3D 占据空间规划，是对 This&That/RoboMaster/DragAnything 等纯 2D 或隐式轨迹条件方法的直接改进，思路上与同期 ORV（4D 占据为中间表示）呼应但用法不同——ORV 把占据用作生成条件本身，本文把占据仅用于离线路径规划，规划结果再投影为轻量掩码注入扩散模型，工程上更简洁，且不需要额外 ControlNet/注入模块（零新增参数）。CHOMP 式多目标优化 + 速度剖面重分配的组合，是把经典运动规划工具务实迁移到视频生成条件构造中的一个干净示例，Fig. 6 的碰撞规避消融是该设计价值的直接证据。三级控制粒度（关键点/全轨迹/affordance）的对比也提供了较清晰的能力坐标系（Table 1）。

**局限与开放问题**：作者自陈的局限是当前规划主要针对刚体交互和准静态抓取，尚未处理接触力/柔顺性，也未涵盖铰接体或可形变物体，留待未来引入接触感知目标与更强生成先验。此外从实验设计看还存在几点可讨论之处：(1) 训练与测试数据均来自 Bridge V1/V2 同分布切分（8.7k episode 内部 9:1 划分），未展示跨具身、跨场景域外泛化能力，也未做"生成视频数据反哺下游 VLA 策略训练"的闭环验证——这恰是数据合成类工作最终需要回答但本文未直接给出的问题；(2) 标注流水线依赖 VGGT + YOLO + Qwen-VL + SAM 多个预训练模型级联，误差可能逐级传播，论文未报告该标注流水线自身的准确率或人工筛除比例；(3) 距离估计基于"抓取过程中相机—物体距离恒定"的强假设，物体/夹爪均简化为球体做透视投影，在非对称或细长物体上的几何近似误差未做专门分析；(4) VBench 美学/成像质量指标不敌 This&That，说明轨迹精度与画面美学之间可能存在一定权衡，论文未深入讨论原因。与本文作者此前工作 ManipDreamer（Li et al. 2025，基于动作树和视觉引导但未显式建模 3D 占据）相比，本文是该系列向 3D 空间推理的自然延伸，二者的定量对比论文中未给出，是可补充的一个角度。

## 参考

- Fu, X. et al. *RoboMaster: Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control*. arXiv:2506.01943.
- Wang, B. et al. *This&That: Language-gesture controlled video generation for robot planning*. ICRA 2025.
- Yang, X. et al. *ORV: 4D Occupancy-centric Robot Video Generation*. arXiv:2506.03079.
- Wu, W. et al. *DragAnything: Motion control for anything using entity representation*. ECCV 2024.
- Wang, J. et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025.
