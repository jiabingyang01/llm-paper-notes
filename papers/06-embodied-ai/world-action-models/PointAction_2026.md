# PointAction：以 3D 点作为机器人控制的通用动作表示

> **论文**：*PointAction: 3D Points as Universal Action Representations for Robot Control*
>
> **作者**：Mutian Tong†, Han Jiang†, Qiao Feng, Lingjie Liu, Jiatao Gu（† 共同一作）
>
> **机构**：University of Pennsylvania
>
> **发布时间**：2026 年 06 月（arXiv 2606.03943）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.03943) | [PDF](https://arxiv.org/pdf/2606.03943)
>
> **分类标签**：`Video-Action Model` `4D生成` `点云动作表示` `扩散策略` `跨本体迁移`

---

## 一句话总结

PointAction 把 Video-Action Model 的"隐式 RGB 中间表征"换成显式的**动态 3D 点图（pointmap）**，用一个统一的 4D 视频扩散模型（RGB+XYZ 联合生成）作为跨任务、跨本体可预训练的通用组件，再接一个轻量、按机器人本体单独训练的点到动作扩散解码器；在 RoboCasa365 仿真上 ID/OOD-Env/OOD-Task 三个体制均超过 GR00T N1.7、π0.5、VPP、Cosmos Policy 等基线，并在两个预训练阶段从未见过的真实机械臂（xArm7、YAM）上零样本迁移出可执行策略。

## 一、问题与动机

近年 Video-Action Models（VAMs）利用预训练视频扩散骨干来预测未来场景 rollout，作为动作生成的显式推理轨迹，分为两类范式：一类先联合建模未来观测和动作（端到端），一类先生成未来观测再用逆动力学模块解码动作（解耦式）。但作者指出这类方法有两个耦合的瓶颈：

1. **RGB-only 的中间表征具有几何歧义**：即便动作是联合或从 rollout 解码而来，中间表征通常仍以 RGB 为主，度量意义上的 3D 运动、接触相关几何、细粒度空间约束都是隐式的，迫使动作模块去学习一个从"外观变化"到"控制量"的困难映射。
2. **学习这种隐式 grounding 需要大量成对的观测-动作监督**，而这类数据采集成本高、和具体本体绑定、难以跨任务跨环境扩展。

论文的核心论点：动态 3D 点图（dynamic pointmaps）可以打破这一表征-监督瓶颈——作为中间表征，它们显式暴露任务相关 3D 几何随时间的度量运动和接触相关空间约束；作为训练信号，点级监督可以从视频经多视图重建、单目深度估计、运动线索等方式**大规模、无需机器人专属动作标签**地获得。

## 二、核心方法

### 2.1 问题形式化与分解

标准 VAM 建模为：给定 $t$ 时刻 RGB 观测 $o_t$、本体状态 $s_t$、指令 $l$，联合采样未来观测块 $\tilde o$ 与动作块 $\tilde a$：

$$(\tilde o,\tilde a)\sim \pi_\theta^{\text{VAM}}(\cdot \mid s_t,o_t,l) \tag{3.1}$$

用大白话说：策略同时"想象"接下来会看到什么画面、同时决定要怎么动，但如果"想象"只是 RGB 像素，动作模块就得自己脑补三维运动和接触关系。

PointAction 把 pointmap $u_t\in\mathbb R^{H\times W\times 4}$ 引入为共享中间变量，每个像素存 $u_t(p)=(x_p,y_p,z_p,\alpha_p)$，$\alpha_p=1$ 表示该像素落在机器人本体表面（几何通道 $(x,y,z)$ 由 4D 视频模型联合预测，掩码 $\alpha$ 由推理时的分割模型给出）。整个策略被显式分解为：

$$\pi(\tilde o,\tilde a\mid s_t,o_t,l)\approx \int \pi_\theta^{\text{4DVM}}(\tilde o,\tilde u\mid o_t,l)\cdot \pi_\psi^{\text{DEC}}(\tilde a\mid \tilde u,s_t)\,d\tilde u \tag{3.2}$$

用大白话说：先用一个"看视频学几何"的通用模型 $\pi_\theta^{\text{4DVM}}$ 猜出画面和三维点会怎么动（这个模型不依赖具体机器人状态，本体无关，可以在大规模视频上预训练、跨机器人复用），再用一个"看点云开车"的小模型 $\pi_\psi^{\text{DEC}}$ 把三维点的运动翻译成该机器人的具体控制指令（本体相关，只需少量配对动作数据微调）。

### 2.2 通用 video-to-point 预训练（$\pi_\theta^{\text{4DVM}}$）

**骨干与初始化**：从基础机器人视频模型 LVP（Large Video Planner）初始化，用 LoRA 做参数高效微调，在冻结文本编码器和 VAE 的情况下只更新约 2.09 亿参数（LoRA rank 128 / α 64 / dropout 0.0），既保留骨干原本的 RGB 生成质量，又能适配新的 RGB-XYZ 联合目标。

**空间对齐的模态融合**：不是简单把 XYZ 当作 6 通道输入（直接拼接会让视频 DiT 难以对齐新引入的几何通道和预训练的纹理表征），而是借鉴 4DNeX 的思路，用共享的冻结 VAE $\mathscr E$ 分别编码 RGB 帧 $z^o=\mathscr E(o)$ 与 XYZ 点图 $z^u=\mathscr E(u)$，再沿空间宽度维拼接：

$$\bar z^{\text{joint}}=\text{WidthConcat}(z^o,z^u)\in\mathbb R^{C\times h\times 2w} \tag{3.3}$$

用大白话说：把 RGB 和几何两路"图像"横向拼在一起送进同一个 Transformer，靠自注意力去学习每个视觉 patch 和它对应几何 patch 的局部关系，而不是引入全新的输入通道去破坏预训练权重的结构；同时引入 RGB/XYZ 两套可学习的模态嵌入向量，并把 RoPE 沿拼接后的宽度维重复，以适配翻倍的 token 长度。

**训练目标（Diffusion Forcing + Flow Matching）**：联合隐序列 $z^{\text{joint}}$ 被随机切成历史上下文 $\hat z$（长度 $m$）和未来轨迹 $\tilde z$，各自独立加噪；未来帧在噪声水平 $\tau\in[0,1]$ 下的插值目标为

$$\tilde z_\tau=(1-\tau)\tilde z+\tau\epsilon,\quad \epsilon\sim\mathcal N(0,I) \tag{A.1}$$

历史上下文以独立噪声水平 $\tau'$ 加噪，$\tau'=0$（即干净上下文）的概率设为 50%，用于鼓励对噪声/干净两种历史条件都鲁棒。整体流匹配损失：

$$\mathscr L_{\text{flow}}=\mathbb E\left[\left\|v_\theta(\tilde z_\tau,\hat z_{\tau'},l,\tau)-v\right\|_2^2\right],\quad v=(\epsilon-\tilde z) \tag{A.2}$$

用大白话说：模型学的是"从当前噪声点回到干净轨迹"所需的速度场，训练时既要学会预测未来帧，又要学会在部分历史信息带噪的情况下仍然稳健地续写。

**机器人中心点轨迹提取**：4D 视频模型输出的是场景级稠密 XYZ（没有机器人专属监督），因此在推理时用开放词表分割模型 SAM 3 对生成的 RGB 轨迹加 "robot" 提示词得到掩码 $\bar\alpha$，取 $\tilde u_{\text{robo}}=\tilde u_{xyz}\odot\bar\alpha$ 作为动作解码器的实际输入。

### 2.3 本体专属 point-to-action 解码（$\pi_\psi^{\text{DEC}}$）

灵感来自 3D Diffusion Policy：对每一预测帧，用最远点采样（FPS）把机器人中心点集下采样到 $N=512$ 个点，用 PointNet 风格的 3 层 MLP $\Phi$ 编码；一个轻量 DiT 解码器 $\epsilon_\psi$ 以逐帧点特征为 token 对齐条件，通过 AdaLN 注入扩散步数与首帧机器人状态 $s_t$，一次性并行去噪整段 49 步动作块。训练目标：

$$\mathscr L_{\text{dec}}=\mathbb E\left[\left\|\epsilon'-\epsilon_\psi\!\left(a^{(j)},\Phi(\tilde{\mathscr P}),s_t,j\right)\right\|_2^2\right] \tag{A.3}$$

解码器仅 6 层 Transformer block、隐藏维 256、4 个注意力头，训练用标准 100 步扩散加噪 + DDIM 10 步推理采样，规模远小于 4D 视频骨干，因此只需要少量本体专属配对数据（每任务约 100 条遥操作示教用于仿真，20-50 条用于真机）即可训练/微调。

### 2.4 训练数据与推理开销

预训练语料来自 BridgeData V2（WidowX 250）与 DROID（Franka Panda），过滤掉相机参数损坏或执行失败样本后约 7.5 万条高质量视频片段（DROID 5 万 + BridgeData V2 2.5 万）；DROID 原始深度噪声较大，改用 FoundationStereo 从双目对重新计算，BridgeData V2 是单目相机，用 Depth-Anything-V3 估计度量深度和伪相机内参。所有视频统一下采样到 49 帧、分辨率 832×480。推理用 UniPC 采样器 40 步、CFG scale 2.5，单次前向在 NVIDIA B200 上约 6 分钟。

## 三、实验结果

### 3.1 RoboCasa365 仿真评测（Table 1，100 rollout/cell）

评测三种体制：ID（同任务同环境）、OOD-Env（同任务新环境/纹理，测视觉鲁棒性）、OOD-Task（5 个训练未见的新任务，测语义泛化）。

| Setting | GR00T N1.7 | π0.5 | VPP | Cosmos Policy | PointAction |
|---|---|---|---|---|---|
| ID（10 已见任务） | 44.5 | 39.8 | 34.5 | 45.2 | **47.7** |
| OOD-Env（10 已见任务） | 37.6 | 35.2 | 32.2 | 42.9 | **44.1** |
| OOD-Task（5 未见任务） | 8.6 | 6.9 | 7.4 | 14.0 | **17.0** |

PointAction 在 ID/OOD-Env 上比最强基线 Cosmos Policy 分别高 +2.5/+1.2 个百分点，比 VLA 基线差距更大；GR00T N1.7 在环境迁移下退化最严重（-6.0 个百分点），PointAction 的降幅（-3.6）更温和。零样本 OOD-Task 上 PointAction（17.0%）约为两个 VLA 基线的 2-2.5 倍（GR00T N1.7 8.6%、π0.5 6.9%），高于 VAM 基线（VPP 7.4%、Cosmos Policy 14.0%）；作者强调绝对成功率仍低，说明零样本指令跟随远未解决，但相对优势提示显式空间接口比图像/隐空间条件更容易迁移到新的任务组合。

### 3.2 跨本体真机部署（Table 2）

两条预训练阶段完全未见过的机械臂：

**(a) xArm7**（50 条示教微调/任务，100 rollout/任务）

| Method | Pick&Place | Stack Cubes | Stack Cups | Avg |
|---|---|---|---|---|
| GR00T N1.7 | 30.0 | 7.0 | 7.0 | 14.7 |
| π0.5 | 42.0 | 12.0 | 14.0 | 22.7 |
| PointAction | **67.0** | **28.0** | **34.0** | **43.0** |

**(b) YAM 臂**（20 条示教微调/任务，20 rollout/任务）

| Method | Stack Cubes | Pick Pens | Insert Cups |
|---|---|---|---|
| GR00T N1.5 | 0 | 20 | 15 |
| π0 | 0 | 10 | 15 |
| PointAction | **20** | **60** | **50** |

在每个任务和每条机械臂上都全面超过 VLA 基线，其中 YAM 上多个任务基线几乎完全失败而 PointAction 取得非平凡成功率，验证了点表示作为本体无关接口的可迁移性。

### 3.3 消融实验（Table 3，ID / OOD-Env 成功率）

| 解码器输入 | ID ↑ | OOD-Env ↑ |
|---|---|---|
| RGB Only | 25.1 | 20.3 |
| RGB + XYZ（Robot Only） | 37.2 | 30.9 |
| XYZ Only – Full Scene | 27.1 | 19.4 |
| XYZ Only – Robot + Scene | 40.3 | 33.7 |
| XYZ Only – Robot Only（DA3 后处理深度） | 28.4 | 21.7 |
| **XYZ Only – Robot Only（本文）** | **47.7** | **44.1** |

两点关键发现：(1) 把 XYZ 换成纯 RGB 造成最大跌幅（47.7→25.1），说明像素输入的几何歧义是主要失败模式；在 XYZ 上再叠加 RGB 反而也会掉点（37.2 vs 47.7），说明与任务无关的视觉伪影（光照、纹理）会干扰轻量解码器。(2) 点源消融显示"只用机器人本体点、掩掉场景点"严格优于全场景点或场景+机器人双编码器方案，因为场景点不携带机器人状态信息、反而注入歧义；用 DepthAnything-V3 对生成 RGB 做后处理深度替代联合预测的 XYZ，ID 成功率跌到 28.4%，说明联合 RGB-XYZ 生成不能被级联深度估计替代。

### 3.4 4D 生成质量（Table 4，300 条 DROID+BridgeData V2 held-out 轨迹）

| Method | PSNR↑ | SSIM↑ | FVD↓ | AbsRel↓ | δ1↑ | Chamfer L1↓ |
|---|---|---|---|---|---|---|
| TesserAct | 12.225 | 0.487 | 746 | 0.403 | 0.641 | 0.389 |
| 4DNeX | 13.858 | 0.542 | 818 | 0.348 | 0.681 | 0.370 |
| LVP | 19.613 | 0.816 | 330 | - | - | - |
| Wan 2.1（14B） | 14.532 | 0.674 | 671 | - | - | - |
| PointAction（+StreamVGGT 级联几何） | - | - | - | 0.382 | 0.675 | 0.341 |
| **PointAction（联合生成）** | **19.631** | **0.821** | **320** | **0.176** | **0.890** | **0.122** |

联合直接预测 XYZ 全指标最优，且明显好于"生成 RGB 再跑 StreamVGGT 重建几何"的级联方案，说明级联式设计会累积误差，而统一 4D 生成器可以避免。附录 Table 7 进一步在 RoboCasa365 未见轨迹上对比模拟器 ground-truth 深度：PointAction 联合 RGB-XYZ（AbsRel 0.118, δ1 0.872, Chamfer-L1 0.151）优于在其生成 RGB 上跑 MegaSAM（0.187/0.775/0.327）或 DepthAnything-V3（0.198/0.755/0.361），且方差显著更低，说明几何是真正被模型学到的，而非可从 RGB 输出事后恢复。

## 四、局限性

1. **推理慢**：受限于视频扩散模型本身固有的慢推理（单次前向约 6 分钟/B200），难以做到实时控制。
2. **开环执行、无法纠错**：策略以单次前向一次性去噪出完整 49 步动作块，中途不重新规划，无法应对抓取过程中物体滑落、环境接触扰动等意外事件，物理干扰下容易产生复合误差。作者提出的未来方向是将视频骨干蒸馏为自回归、可做 KV 缓存的因果架构，以实现低延迟闭环控制。
3. **自遮挡导致几何输入退化**：当机器人在生成视频中遮挡自身部分肢体时，机器人中心 XYZ 点图会不完整，动作解码器收到的几何输入质量下降，在夹爪背后、从相机视角部分不可见的末端执行器姿态上最为明显。
4. **依赖推理时的分割模型**：机器人掩码 $\bar\alpha$ 由 SAM 3 在生成的 RGB 轨迹上事后提取，本质是一个训练与推理解耦的后处理步骤，其质量会直接传导到点源质量（这也是消融实验证明"机器人专属点"最重要的原因之一，但也意味着分割失败会成为误差来源）。
5. **绝对泛化能力仍有限**：OOD-Task 零样本成功率仅 17%，绝对水平不高，作者也坦承零样本指令跟随远未解决。

## 五、评价与展望

**优点**：论文的核心贡献在于用一个概念上干净的分解——本体无关的通用 4D 视频预训练 + 轻量本体专属点到动作解码——同时缓解了 VAM 领域两个耦合已久的问题：RGB-only 中间表征的几何歧义，以及跨本体动作监督的稀缺性。相较于同期的 4D 生成/机器人控制工作（4DNeX 侧重通用 4D 场景生成、TesserAct 走 RGB-D-Normal 再做法向积分重建的两阶段路线），PointAction 直接在单一联合扩散模型里输出空间对齐的 XYZ 点图，实验（Table 4）表明这种"一步到位"的联合生成在 4D 生成质量上明显优于先生成 RGB 再级联深度/4D 重建的两阶段方案，为"视频先验 grounding 到低层动作"提供了一个不依赖预定义 CAD 模型（不同于 4DGen 需要夹爪 CAD 模型的假设）、也不需要连续可见性假设的实现路径。消融实验设计得比较扎实，清晰拆分了"模态选择"（RGB vs XYZ）和"点源选择"（全场景 vs 机器人+场景 vs 纯机器人）两个正交因素，定量支持了"机器人中心掩码化"这一设计选择的必要性。

**局限与开放问题**：(1) 该方法本质上仍是"生成式世界模型 + 逆动力学/条件解码"这一 VAM 范式下的一个变体，其性能天花板受限于视频扩散骨干（LVP）本身捕捉物理动态的能力和分割模型（SAM 3）在具体遮挡场景下的稳健性，论文自身的失败模式分析（自遮挡、开环执行）也印证了这点。(2) OOD-Task 场景下 17% 的绝对成功率表明当前范式在真正意义上的组合泛化和长程语言指令跟随上仍有很大差距，相对基线的优势更多体现在"结构化几何接口比隐式表征更容易跨任务复用"，而非任务理解本身有质变提升。(3) 训练/推理的分离（离线可大规模学习几何、在线依赖分割模型抠出机器人点）是一种务实的工程折中，但也引入了额外的推理时误差源，未来工作可以考虑让分割/掩码信号本身也纳入端到端可微的训练循环，减少级联误差。(4) 该工作与同期趋势（如 CoT-VLA 的视觉 CoT、DreamVLA/3D-VLA 引入额外深度模态、UniVLA/WorldVLA 联合优化未来图像与动作预测）共同指向一个更大的开放问题：视频/世界模型先验到底该以何种中间表征（像素、隐空间 token、显式几何点、还是混合模态）grounding 到机器人低层控制最为高效，PointAction 提供了"显式度量点云是更优中间层"这一有力的实证支持，但尚未与深度/法向量等其他显式几何表示做直接的同设置对比，也没有涉及力/触觉等接触模态,这些都是值得后续工作系统比较的方向。(5) 论文提出的"蒸馏为自回归因果架构+KV缓存"是缓解开环执行问题的合理路径，但尚未给出实验验证，闭环、可纠错的点动力学预测与控制如何联合训练仍是一个开放课题。

## 参考

1. Chen et al. *Large video planner enables generalizable robot control.* arXiv:2512.15840, 2025.（本文的 4D 视频骨干 LVP 初始化来源）
2. Kim et al. *Cosmos-Policy: Fine-tuning video models for visuomotor control and planning.* arXiv:2606.16163, 2026.（仿真评测中最强的端到端 VAM 基线）
3. Zhen et al. *Tesseract: Learning 4D embodied world models.* arXiv:2504.20995, 2025.（预测 RGB-深度-法向并做法向积分重建的两阶段 4D 世界模型，与本文的单阶段联合 XYZ 生成路线形成对比）
4. Chen et al. *4DNeX: Feed-forward 4D generative modeling made easy.* arXiv:2508.13154, 2025.（本文空间对齐模态融合设计的直接灵感来源，也是 4D 生成质量对比基线）
5. Hu et al. *Video Prediction Policy: A generalist robot policy with predictive visual representations.* ICML, 2025.（解耦式 VAM 代表工作 VPP，仿真评测基线之一）
