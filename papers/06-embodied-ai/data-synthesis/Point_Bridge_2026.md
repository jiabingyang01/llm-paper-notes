# Point Bridge：面向跨域策略学习的 3D 点云表征

> **论文**：*Point Bridge: 3D Representations for Cross Domain Policy Learning*
>
> **作者**：Siddhant Haldar, Lars Johannsmeier, Lerrel Pinto, Abhishek Gupta, Dieter Fox, Yashraj Narang, Ajay Mandlekar
>
> **机构**：NVIDIA、New York University、University of Washington
>
> **发布时间**：2026 年 01 月（arXiv 2601.16212，v4 于 2026 年 3 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.16212) | [PDF](https://arxiv.org/pdf/2601.16212)
>
> **分类标签**：`点云表征` `sim-to-real` `VLM场景过滤` `合成数据` `多任务模仿学习`

---

## 一句话总结

Point Bridge 用 VLM（Gemini + Molmo + SAM-2）自动从图像中抠出任务相关物体、再用立体深度（Foundation Stereo）反投影为统一的 3D 点云表征,把仿真里 MimicGen 大规模合成的演示数据和真实机器人数据映射到同一个"域无关"点云空间,从而实现零样本 sim-to-real 迁移与真机小样本联合训练,零样本迁移比最强图像基线提升 39%（单任务）/44%（多任务）,联合训练后领先图像基线 61%/66%。

## 一、问题与动机

机器人基座模型的进步受限于真实世界大规模操作数据的稀缺,遥操作采集成本高、难以规模化。仿真与合成数据（如 MimicGen）提供了可扩展的替代方案,但可用性受限于 sim-to-real 的视觉域差（table 外观、背景、光照等差异）。已有的 sim+real 联合训练方法（如 Maddukuri et al. 2025 的 DexMimicGen 系列工作、Mittal et al. 2023 的光照真实感仿真器)通常要求场景在视觉/物体层面精细对齐（"digital cousin"式仿真-真实资产对应),对齐成本高、难以泛化到新任务。另一条路线是任务相关关键点表征（Haldar & Pinto 2025 等),它们把机器人和场景抽象为与原始像素外观无关的关键点集合,但现有方法大多依赖人工标注关键点、且主要解决具身差异（embodiment gap）而非视觉差异,并且局限于单任务设定。Point Bridge 的核心问题是：能否构造一种统一、自动化、域无关的点云表征,让 MimicGen 这类大规模合成仿真数据集不需要显式视觉/物体对齐就能零样本迁移到真实机器人,并自然扩展到多任务场景？

## 二、核心方法

Point Bridge 分三个阶段：(1) 用 MimicGen 把少量人类演示扩展为大规模仿真数据集；(2) 通过 VLM 引导的点提取流水线把仿真与真实观测都统一为任务相关 3D 点云；(3) 用 Transformer 策略在统一点云上学习,并支持真机数据联合训练与多任务扩展。

**数据采集与合成扩增。** 使用 MimicLabs 任务套件定义原子任务,每个任务下每对物体实例只需人类采集 5 条演示，再用 MimicGen 通过对每条演示片段施加恒定 SE(3) 变换

$$T_W^{o_i'}(T_W^{o_i})^{-1}$$

把源物体位姿映射到新场景的目标物体位姿，从而保持末端执行器与物体的相对几何关系不变，将其扩增到每个物体对 300 条、每任务共 1200 条仿真演示（4 个物体对 × 300）。

**VLM 引导的场景过滤（点提取流水线）。** 给定初始场景图像和自然语言任务描述，先用 `Gemini-2.5-flash` 做纯文本层面的任务相关物体识别（如"把碗放到盘子上"→识别出 {bowl, plate}），再用 `Molmo-7B` 做像素级定位（pointing）得到初始坐标，作为 `SAM-2` 分割的种子点得到逐物体 mask，并利用 SAM-2 自带的记忆机制在后续帧中持续跟踪 mask 以处理遮挡。随后在每个 mask 内（先向内收缩 20% 以避免边界噪声）均匀采样 2D 关键点，结合 Foundation Stereo 估计的立体深度图和相机内外参反投影为 3D 点，再用最远点采样（farthest point sampling）降采样到每物体 $M \ll N$ 个代表点（实验中每物体 128 点），最后统一变换到机器人基座坐标系，得到最终物体点集 $\mathcal{P}_i^{3D}$。仿真中则绕开 VLM，直接从物体 mesh 采样，但为消除"mesh 可采全表面点、真实相机只能看到可见面"的差异，先用与真实相同的相机内外参把 mesh 点投影到图像平面 $\tilde{x}=K[R\mid t]X_{mesh}$，再用仿真真值深度图反投影回 3D，并注入标准差 1cm 的高斯噪声模拟真实传感器噪声。

**机器人表征。** 类似 Haldar & Pinto (2025)，把末端执行器抽象为夹爪上的一组关键点：给定 $t$ 时刻机器人位姿 $T_r^t$，定义 $N$ 个相对该位姿的刚体变换 $T^i$，计算

$$(T_r^t)^i = T_r^t \cdot T^i,\quad \forall i \in \{1,...,N\}$$

用大白话说：就是把夹爪抽象成固定挂在末端执行器坐标系上的若干个"骨架点"，机器人一动，这些点跟着位姿矩阵一起刚性变换，从而得到与视觉外观无关、只依赖运动学的机器人点集。

**策略学习。** 借鉴 BAKU 的 decoder-only 多任务 Transformer 架构，把机器人点 $\mathcal{P}_r$ 与物体点 $\mathcal{P}_o$ 合并为统一点云 $\mathcal{P}$，用 PointNet 编码；多任务设定下额外输入用 6 层 MiniLM（Sentence Transformers）编码的语言指令嵌入 $\mathcal{L}$，共同送入 Transformer 主干与确定性动作头，预测末端位姿与夹爪状态：

$$\mathcal{O}^{t-H:t} = \{\mathcal{P}_r^{t-H:t}, \mathcal{P}_o^{t-H:t}, \mathcal{L}\}, \qquad \hat{\mathcal{A}}^{t+1} = \pi(\cdot \mid \mathcal{O}^{t-H:t})$$

用大白话说：策略吃进过去 $H$ 步的机器人点、物体点和（可选的）语言指令，输出下一步动作；采用动作分块（action chunking）+指数时间平均来平滑轨迹，训练目标是预测动作与真值动作的 MSE。

**推理时的多路深度传感策略。** 部署时同样先用 VLM 流水线得到 2D 物体关键点，再支持三种深度获取方式：(1) 主线方案——立体图像 + Foundation Stereo，5Hz；(2) 商用 RGB-D 相机直接读深度，15Hz 但噪声大、对反光/透明物体失效；(3) 双目 2D 关键点用 MAST3R 跨视角匹配 + CoTracker 逐帧跟踪后多视角三角化，2.5Hz。三种方案在吞吐量与精度间做权衡，供不同部署需求选择。

## 三、关键结果

实验在 Franka Research 3 + Franka Hand 上进行（Deoxys 20Hz 控制器，RoboTurk 采集仿真演示、Open Teach 采集真机演示，均降采样到 10Hz 训练；感知端 Intel RealSense RGB-D + ZED 2i 双目相机），全文共完成 **1410 次真实机器人评测**。三个仿真+真机联合任务为 bowl on plate / mug on plate / stack bowls（各配置 30 次评测，3 个物体对 × 10 次）；另有 fold towel / close drawer / put bowl in oven 三个仅用真实数据（各 20 条遥操作演示、无仿真数据）的软体/铰接物体任务。

**单任务零样本 sim-to-real（Table 1）：**

| 观测模态 | 数据配置 | Bowl on plate | Mug on plate | Stack bowls |
|---|---|---|---|---|
| 图像基线 | Real | 9/30 | 10/30 | 11/30 |
| 图像基线 | Co-Train Sim | 2/30 | 17/30 | 14/30 |
| Point Bridge | Real | 25/30 | 25/30 | 24/30 |
| Point Bridge | Zero-Shot Sim | 23/30 | 21/30 | 24/30 |
| Point Bridge | Co-Train Sim | **29/30** | **30/30** | **29/30** |

**多任务设定（Table 2）** 呈现相同趋势，且 Point Bridge 多任务 Co-Train Sim 在三个任务上均达到 **30/30**，多任务策略性能与单任务持平或更优。论文汇总：零样本迁移相比最强图像基线单任务提升 39%、多任务提升 44%；联合真机数据（45 条真机演示 + 每任务 1200 条仿真演示，80:20 比例）后相比图像基线联合训练分别领先 61%（单任务）与 66%（多任务），且加入真机数据比纯零样本再提升最多 30%。

**软体/铰接物体任务（Table 3，纯真实数据、无仿真）：** fold towel 17/20，close drawer 18/20，bowl in oven 16/20，三任务平均约 85% 成功率，验证表征在可形变/铰接物体上依然适用。

**系统设计消融（Table 4，Bowl on plate / Mug on plate / Stack bowls）：**

| 类别 | 变体 | Bowl on plate | Mug on plate | Stack bowls |
|---|---|---|---|---|
| 深度获取 | Point Tracking (MAST3R+CoTracker) | 5/30 | 7/30 | 6/30 |
| 深度获取 | RGB-D | 15/30 | 12/30 | 13/30 |
| 深度获取 | Foundation Stereo | **23/30** | **21/30** | **24/30** |
| 相机对齐 | Aligned（真实相机外参对齐仿真采样） | 23/30 | 21/30 | 24/30 |
| 相机对齐 | Ground truth（仿真全表面均匀采样，无视角约束） | 12/30 | 7/30 | 6/30 |

Foundation Stereo 在反光/透明物体上明显优于 RGB-D 与逐帧跟踪方案；训练时用真实相机外参"视角对齐"采样仿真点云，比直接用仿真真值全表面均匀采样效果好得多，说明"仿真点云需要与真实可见表面分布匹配"是迁移成功的关键设计。

## 四、评价与展望

**优点：** Point Bridge 把"域无关表征"思路从此前依赖人工标注关键点的单任务设定（如 Haldar & Pinto 2025）扩展为全自动化 VLM 流水线 + 多任务 Transformer 策略，是关键点/点云表征路线在数据规模与自动化程度上的一次有意义的推进；系统性地对比了三种推理期深度获取方案（立体深度、RGB-D、多视角三角化）与两种仿真点采样策略（对齐视角 vs. 全表面真值），消融揭示"训练时点云的可见性分布应与部署时相机视角匹配"这一容易被忽视但影响巨大的设计选择，具有较强的工程参考价值。相比要求场景级视觉/资产对齐的 sim+real 联合训练工作（如依赖 digital-cousin 资产的 DexMimicGen 系列），Point Bridge 显式声称只需"最小"视觉/物体对齐，这是其核心卖点。

**局限与开放问题：** 论文自陈四点局限：(1) 强依赖 VLM/分割/深度估计模型的鲁棒性，流水线中任一环节（Gemini 物体识别、Molmo 定位、SAM-2 跟踪、Foundation Stereo 深度）出错会直接传导到策略输入；(2) 仍需要相机位姿在仿真与真实之间大致对齐以避免视角分布漂移，论文提出的"训练时随机化仿真相机视角"仅是缓解方向、尚未在主实验中验证；(3) 点表征天然丢弃了非任务相关的场景上下文，在杂乱背景/强遮挡环境下的鲁棒性存疑，论文未给出定量结果；(4) 由于额外的深度与点云处理开销，Point Bridge 的控制频率（5–15Hz，视深度方案而定）显著低于纯图像基线，可能在需要快速反馈的动态任务上受限。此外，评测任务集中在刚体拾放（3 个仿真任务）与少量软体/铰接任务（3 个纯真实任务），任务复杂度和长时序程度有限，尚未验证该方法在大规模多任务（数十至上百任务）或需要精细力控/接触丰富的操作上的可扩展性；VLM 流水线的延迟与失败率分析被放在附录中，正文未充分讨论其对整体系统可靠性的影响。与近期同样探讨仿真到真实数据桥接的工作（如 MimicGen 本身、依赖真实感渲染的仿真器路线）相比，Point Bridge 提供了一种更轻量但对上游感知模型质量更敏感的替代方案，两条路线的优劣权衡（渲染保真度 vs. 表征抽象层级）仍是该领域一个开放的比较维度。

## 参考

- Mandlekar et al. *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*, 2023.
- Haldar & Pinto. *BAKU: An Efficient Transformer for Multi-Task Policy Learning*, 2025（正文亦引用其关键点表征思路）。
- Maddukuri et al. *Sim-and-Real Co-Training for Robot Manipulation*, 2025.
- Deitke et al. *Molmo: pointing-capable Vision-Language Model*, 2024.
- Ravi et al. *SAM 2: Segment Anything in Images and Videos*, 2024.
