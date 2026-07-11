# RoboMaster：面向机器人操作的协同轨迹控制视频生成

> **论文**：*Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control*
>
> **作者**：Xiao Fu, Xintao Wang, Xian Liu, Jianhong Bai, Runsen Xu, Pengfei Wan, Di Zhang, Dahua Lin
>
> **机构**：The Chinese University of Hong Kong；Kuaishou Technology；Zhejiang University
>
> **发布时间**：2025 年 06 月（arXiv 2506.01943，v3 修订于 2026 年 01 月）
>
> **发表状态**：ICLR 2026（Published as a conference paper）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.01943) | [PDF](https://arxiv.org/pdf/2506.01943)
>
> **分类标签**：`轨迹条件视频生成` `机器人操作数据合成` `视频扩散模型` `逆动力学动作标注` `Bridge数据集`

---

## 一句话总结

RoboMaster 把机器人操作视频生成中"机械臂轨迹"与"被操作物体轨迹"从各自独立建模改为按交互相位（交互前/交互中/交互后）分段、由当时的"主导主体"统一驱动的协同轨迹（collaborative trajectory），配合外观+形状耦合的圆形体（circular volume）物体表征，在 Bridge 数据集上把 FVD 从 Tora 的 152.28 降到 147.31、机械臂轨迹误差从 18.14 降到 16.47、用户偏好从 17.74% 提升到 45.16%，并证明其合成视频经逆动力学模型提取动作后能在 RLBench/SIMPLER 上取得比 Tora、TesserAct、OpenVLA 更高的动作规划成功率。

## 一、问题与动机

视频扩散模型正被用作生成机器人决策数据（demonstration）的低成本手段：给定初始帧和轨迹条件即可合成看似合理的操作视频，再用逆动力学模型从视频中反推可执行动作标签，从而缓解真实机器人数据采集昂贵、难以规模化的瓶颈。但已有的轨迹条件方法（如 Tora、DragAnything、MotionCtrl）都采用"分解物体（decompose objects）"的思路——机械臂和被操作物体各自拥有一条独立轨迹，在整段视频里始终并行驱动。论文指出这种"去中心化"（decentralized）建模在物体发生实际物理接触、产生遮挡重叠的交互阶段会导致特征纠缠（feature entanglement）：模型主要是在物体独立运动的样本上训练的，一旦两条轨迹在空间上重叠，生成质量明显下降（例如 Tora 复现实验中苹果在被抓取瞬间直接消失，见论文 Fig. 2）。此外，用户想要精确标注机械臂在交互阶段的起止时间和相对位置本身也很困难。

论文的核心洞察是：机械臂发起并结束动作，而在交互阶段物体的运动其实是对机械臂动作的物理响应，二者存在隐式同步关系；因此没有必要让两条轨迹在全程都独立平权地驱动生成，而应按"谁是当前阶段的主导者"来切换条件信号。

## 二、核心方法

**任务形式化。** 给定初始帧 $\mathbf{I}$、主导主体 $\mathbf{o}_d$（机械臂）与从属主体 $\mathbf{o}_s$（被操作物体）的二值掩码 $\mathbf{M}_d,\mathbf{M}_s$、文本提示 $\mathbf{c}$，以及一条协同轨迹 $\mathcal{C}=\{(x,y)_t\}_{t=1}^F$，模型学习条件分布并输出视频 $\mathbf{X}\in\mathbb{R}^{F\times3\times H\times W}$。轨迹被按时间切成三段：交互前 $\mathcal{C}_1=\{(x,y)_t\}_{t=1}^{F_1}$、交互中 $\mathcal{C}_2=\{(x,y)_t\}_{t=F_1+1}^{F_2}$、交互后 $\mathcal{C}_3=\{(x,y)_t\}_{t=F_2+1}^{F}$。

**1）外观-形状耦合的主体表征（Subject Representation）。** 初始帧经 3D VAE 编码得到 RGB 潜特征 $\mathbf{z}$，物体掩码下采样对齐后，用平均池化在掩码有效像素上提取主体嵌入：

$$\bar{\mathbf{v}}_{d,s}[i] = \frac{1}{\sum_{h,w}\mathbf{m}_{d,s}[h,w]}\sum_{h,w}\tilde{\mathbf{z}}_{d,s}[i,h,w]$$

用大白话说：把物体掩码盖在图像的隐特征上，把掩码覆盖到的所有像素特征取平均，得到一个能代表这个物体"长什么样"的向量。

再把这个向量在每个时间步的轨迹点 $(x,y)_t$ 处"摊开"成一个半径 $r\propto\sqrt{\text{掩码面积}}$ 的圆形区域（circular volume）：

$$\mathbf{v}_{d,s}[i,j,k] = \bar{\mathbf{v}}_{d,s}[i] \ \text{if}\ (j-x)^2+(k-y)^2\le r_{d,s}^2 \ \text{else}\ 0$$

用大白话说：不再用一个点或一个 bounding box 来标记物体在哪，而是画一个大小与物体真实尺寸成比例的"圆盘"贴在轨迹点上，圆盘里填的是物体的外观特征——这样模型既知道物体在哪，也知道它长什么样、有多大，比 Tora/DragAnything 用的纯点表示（point representation）信息量更丰富，能提升训练收敛速度和跨帧身份一致性。

**2）协同轨迹表征（Collaborative Trajectory）。** 与其在整段视频里同时喂两条独立轨迹，RoboMaster 把联合分布按相位分解为三个"物体感知"的子分布连乘：

$$p_\theta(\mathbf{x}_1\mid \mathbf{I},\mathbf{c},\mathbf{v}_d,\mathcal{C}_1)\cdot p_\theta(\mathbf{x}_2\mid \mathbf{I},\mathbf{c},\mathbf{v}_d,\mathbf{v}_s,\mathcal{C}_1,\mathcal{C}_2)\cdot p_\theta(\mathbf{x}_3\mid \mathbf{I},\mathbf{c},\mathbf{v}_d,\mathbf{v}_s,\mathcal{C}_1,\mathcal{C}_2,\mathcal{C}_3)$$

即交互前/后阶段只由机械臂（主导主体）的圆形体特征 $\mathbf{v}_d$ 驱动（此时物体基本静止），交互阶段改由从属物体的轨迹与圆形体特征 $\mathbf{v}_s$ 驱动——直觉是交互期间物体的运动能隐式引导机械臂的生成，且"物体独立运动→双方联动→再度独立运动"这种行为模式变化本身是有用的时序线索。为保证跨帧平滑，还引入因果表示（causal representation）：每一时刻把上一帧的潜特征图向前传播、再被当前帧的物体特征覆盖写入，契合 3D VAE 的因果结构。用户交互上，这一设计也降低了标注负担——用户只需标注交互相位的起止而不必给出机械臂全程轨迹，且物体区域可用画笔粗略涂抹（对不精确掩码具有鲁棒性）。

**3）运动注入模块（Motion Injection Module）。** 协同轨迹潜特征 $\mathbf{V}$ 先经零初始化的 2D 卷积（空间）与 1D 卷积（时间）编码为紧凑表示 $\bar{\mathbf{V}}$，再以即插即用方式叠加进 DiT block 的隐状态：$\mathbf{h}=\mathbf{h}+\text{norm}(\bar{\mathbf{V}})+\bar{\mathbf{V}}$。整体用标准扩散噪声预测损失训练（对 DiT block 与运动注入模块采用不同学习率联合优化）。

**数据构造。** 基础模型为预训练 CogVideoX-5B，训练数据来自 BridgeData V2，标注流程：(1) 用 CoTracker3 做逐像素稠密网格点跟踪；(2) 用 Grounded-SAM 做物体检测分割，解析 prompt 中的名词得到从属物体，机械臂为预定义的"black robotic gripper"；(3) 通过运动阈值 $\tau$ 检测物体启动/停止时刻，自动切分交互前/中/后三段。整条自动化+人工过滤流水线最终在 Bridge 上产出约 21k 条高质量 video-trajectory 样本（Grounded-SAM 单物体场景标注成功率约 0.67，多物体场景仅 0.14，因此约 4000 条视频需人工重新标注）。

## 三、关键结果

训练配置：CogVideoX-5B 主干，分辨率 480×640，37 帧，AdamW（DiT 学习率 2e-5，运动注入模块 1e-4），batch size 16，30,000 步，8×A800，推理 50 步 DDIM、CFG=6.0。测试集为 Bridge 上 214 个样本，覆盖 move/pick/open/close/upright/topple/pour/wipe/fold 等技能，基线（Tora、MotionCtrl、DragAnything、IRASim、TesserAct）均在同一数据集上用相同 CogVideoX-5B 骨干重新训练以保证公平对比。

**视频质量与轨迹精度（Bridge，Table 2）**

| 方法 | FVD↓ | PSNR↑ | SSIM↑ | TrajError_robot↓ | TrajError_obj↓ | 用户偏好↑(%) |
|---|---|---|---|---|---|---|
| TesserAct | 261.84 | 18.99 | 0.778 | 37.34 | 54.64 | 8.01 |
| IRASim | 159.04 | 20.88 | 0.782 | 19.25 | 34.39 | 6.45 |
| MotionCtrl | 170.79 | 19.89 | 0.761 | 21.17 | 28.52 | 9.68 |
| DragAnything | 158.42 | 21.13 | 0.792 | 18.97 | 27.41 | 12.90 |
| Tora | 152.28 | 21.24 | 0.788 | 18.14 | 26.43 | 17.74 |
| **RoboMaster** | **147.31** | **21.55** | **0.803** | **16.47** | **24.16** | **45.16** |

**下游动作规划成功率（100 episodes，Table 4，节选）**：RLBench 5 任务（pick up cup / put knife / put plate / open microwave / close box）RoboMaster 得 0.83/0.76/0.85/0.54/0.79，SIMPLER 4 任务（pick coke can / close drawer / move near / pick object）得 0.91/0.63/0.67/0.81；在全部 10 个任务中 8 项优于 Tora，且明显优于 OpenVLA 与 TesserAct，验证了"更准确的交互建模→更高质量的示教视频→更可靠的逆动力学动作标签"这一核心动机。动作标注管线：先用 300 条 video-action 样本训练 AVDC 式逆动力学模型，并额外用 Cosmos-Predict2.5-2B 的 action-conditioned 变体做验证——预测动作重建的视频与真值训练视频质量相当（PSNR 25.12 vs 25.48，FVD 127 vs 132）。

**消融（Table 5，Bridge 全测试集）**：去掉因果嵌入（w/o Causal Embedding）FVD 升至 151.62；把圆形体表征换回点表征（w/ Points，近似 Tora 做法）TrajError_obj 从 24.16 恶化到 31.41；把协同轨迹换回两条独立轨迹（w/ Separate Trajectories）FVD 升至 152.01；把加法式运动注入换成交叉注意力（w/ Cross Attention）效果最差（FVD 163.56）。此外论文还报告了对掩码稀疏度（60%~90% 有效像素下 PSNR 相对完整掩码保持 97.9%~99.8%）、轨迹扰动（±5%~20% 偏移下 PSNR 保持 97%~99%）、以及错误 prompt（10%~40% 描述被替换为语义相近或不同的错误词，PSNR 仍保持 96.5%~98.4%）的鲁棒性实验，均显示较强容忍度。VBench 通用视频质量评测（Table R12）中 RoboMaster 在 Temporal Flickering（98.27）、Motion Smoothness（98.81）、Subject Consistency（93.55）、Background Consistency（95.40）上均取得最优。

## 四、评价与展望

**优点**：把"交互相位分段 + 主导物体切换"这一物理直觉转化为一个干净的因子分解式条件生成公式，并配合"外观-形状耦合的圆形体"这一比点/框更信息丰富但仍轻量的物体表征，是对 Tora/DragAnything/MotionCtrl 一类去中心化轨迹条件方法的针对性改进，消融实验（点表征 vs 圆形体表征、独立轨迹 vs 协同轨迹）较为干净地拆解了两个设计的独立贡献。论文没有止步于视频质量指标，而是完整跑通"视频生成→逆动力学动作提取→仿真部署评测成功率"的下游闭环（RLBench + SIMPLER），并与 TesserAct、OpenVLA 等真实动作规划基线对比，这比单纯报 FVD/PSNR 更能说明生成视频作为训练数据的实际价值。对掩码稀疏度、轨迹扰动、错误 prompt 的鲁棒性测试也体现了对真实用户交互场景（标注不精确）的关注。

**局限与开放问题**：（1）方法完全在 2D 像素空间建模轨迹与交互，未引入深度/3D 信息，论文附录也承认即便加入深度线索，三种基线方法在 3D 空间下依然存在特征纠缠问题，如何在不显著增加用户标注负担（z 轴标注、深度估计噪声）的前提下扩展到 3D 交互建模仍是未解问题。（2）跨本体（embodiment）泛化能力有限：论文明确指出即使是支持多本体的 TesserAct 也只能覆盖训练集中出现过的机械臂型号，RoboMaster 同样无法泛化到训练时未见过的机械臂形态（如 xArm、MobileALOHA），需要更大规模多本体数据或测试时轻量适配（如 LoRA）。（3）自动分割依赖 Grounded-SAM，在多物体/相似物体场景下成功率仅 0.14~0.21（Table R11），意味着若要把该流水线规模化到更大更杂的真实数据上，物体检测与分割仍是显著瓶颈，目前依赖人工补标。（4）"下游主导、上游从属"的相位划分对于双臂协作、非抓取式接触（如推、扫）等更复杂交互模式是否依然成立，论文未做验证，属于该协同轨迹范式的潜在边界。整体上，该工作可视为轨迹条件视频生成从"独立控制多个物体"转向"按交互物理逻辑分相位、分主次协同控制"的一个有代表性的中间态方案，为后续用生成视频反哺模仿学习/VLA 预训练的数据合成路线提供了一个可复现的强基线。

## 参考

- Zhang et al. Tora: Trajectory-oriented Diffusion Transformer for Video Generation. CVPR 2025.
- Wu et al. DragAnything: Motion Control for Anything using Entity Representation. ECCV 2024.
- Zhu et al. IRASim: Learning Interactive Real-Robot Action Simulators. arXiv:2406.14540, 2024.
- Zhen et al. Tesseract: Learning 4D Embodied World Models. ICCV 2025.
- Walke et al. BridgeData V2: A Dataset for Robot Learning at Scale. CoRL 2023.
- Ko et al. Learning to Act from Actionless Videos through Dense Correspondences (AVDC). arXiv:2310.08576, 2023.
