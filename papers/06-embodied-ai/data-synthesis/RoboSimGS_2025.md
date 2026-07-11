# RoboSimGS：基于高斯泼溅的高保真仿真数据生成实现真实世界零样本机器人操作学习

> **论文**：*High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting*
>
> **作者**：Haoyu Zhao、Cheng Zeng、Linghao Zhuang（共同一作）等，通讯作者 Siteng Huang、Hua Zou
>
> **机构**：武汉大学、阿里巴巴达摩院（DAMO Academy, Alibaba Group）、湖畔实验室（Hupan Lab）、香港中文大学、清华大学、华中科技大学、浙江大学
>
> **发布时间**：2025 年 10 月（arXiv 2510.10637）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.10637) | [PDF](https://arxiv.org/pdf/2510.10637)
>
> **分类标签**：`Real2Sim2Real` `3D-Gaussian-Splatting` `MLLM物理估计` `关节物体重建` `零样本Sim2Real`

---

## 一句话总结

提出 **RoboSimGS**：一个 Real2Sim2Real（R2S2R）框架，用 3DGS + 网格的混合表示重建真实场景，并首次引入 MLLM（GPT-4o）自动推断物体的物理参数（密度/杨氏模量/泊松比）与关节运动学结构（铰链/滑轨），使策略仅在生成的仿真数据上训练即可零样本迁移到真实操作任务；在 8 个真实任务上验证，50 条真实数据 + 50 条 RoboSimGS 仿真数据的混合训练相比纯真实数据在多项任务上显著提升成功率（如 π0 在 Wiping 任务上从 0.00 提升到 0.86）。

## 二、问题与动机

通用机器人策略的训练受限于真实数据采集的成本与人力瓶颈（遥操作难以规模化）。仿真数据理论上可随算力指数扩展，但传统 Sim2Real 方法（domain randomization、system identification）受限于仿真器本身的视觉与物理保真度。近期兴起的 Real2Sim2Real（R2S2R）范式借助 NeRF/3DGS 从真实多视角图像重建照片级场景以缩小视觉域差，但论文指出现有 3DGS 类 R2S2R 工作（如 Robo-GS、SplatSim、RoboGSim、Re3Sim）普遍存在一个关键短板：**只追求视觉照片级真实感，场景基本是静态、非交互的**，无法模拟抽屉、铰链等关节运动或可变形物体，因而难以支撑接触密集型（contact-rich）操作任务的策略训练。同时期的通用数据生成流水线（如 DexMimicGen 使用固定仿真资产、RoboVerse 只支持刚体）也未能同时解决"视觉保真"与"物理可交互"两个维度。RoboSimGS 的目标即是在保持 3DGS 照片级视觉保真度的同时，让场景中的物体真正具备物理可交互性。

## 三、核心方法

RoboSimGS 是一个两阶段流水线：**场景重建**（3DGS 背景 + 关节化 mesh 前景）→ **Sim2Real 环境对齐** → **整体场景增强**生成多样化数据。

### 3.1 场景重建

**背景重建**：静态背景用 3DGS 建模，每个高斯由位置、不透明度、颜色（球谐系数）和协方差 $\Sigma$ 描述，渲染时按深度排序并做 alpha 混合投影到图像平面。为了让重建场景能与仿真器中的语义/几何精确对齐，每个高斯额外附加一个语言监督的语义特征向量 $\mathbf{f}_i \in \mathbb{R}^d$，用 CLIP 文本编码器把类别文本 prompt（如 "a robot arm"、"a red block"）编码为目标特征，用对比损失监督渲染出的 2D 特征图。

相机投影会诱导协方差变换：

$$\Sigma' = \mathbf{J}\mathbf{W}\Sigma\mathbf{W}^\top\mathbf{J}^\top \tag{1}$$

用大白话说：3D 空间里每个高斯"椭球"的形状，要通过相机的投影矩阵 $\mathbf{W}$（及其雅可比 $\mathbf{J}$）变换成它在 2D 图像平面上呈现的形状。

颜色和语义特征同时按 alpha 混合渲染：

$$\{\hat{\mathbf{F}}, \hat{\mathbf{C}}\} = \sum_{i \in N} \{\mathbf{f}_i, \mathbf{c}_i\} \cdot \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \tag{2}$$

用大白话说：按照高斯离相机由近到远的顺序做透明度加权叠加——离相机越近、不透明度越高的高斯，对最终像素颜色和语义特征的贡献越大，后面的高斯会被前面的部分遮挡。

**物体重建**：用 ARCode 从多视角图像自动分割出前景物体并生成静态 mesh，但静态 mesh 无法参与物理仿真。论文为此设计了两个 MLLM 驱动的自动化步骤：

- **MLLM 驱动的关节推断**：把物体 mesh 的多视角渲染图交给 GPT-4o，让其识别物体类别、提出潜在的关节结构（关节类型如 prismatic/revolute，以及待分割部件的语义标签，如 "drawer body"、"main cabinet"）；随后用 AffordDex 的开放词汇分割方法，以这些文本标签为 prompt 对 mesh 做部件切分；最后再次询问 MLLM 以确定关节轴与运动限位，生成 URDF 兼容的关节定义，完成从静态 mesh 到可动关节资产的转化。
- **MLLM 驱动的物理估计**：引入一个由 GPT-4o 驱动的 "physics expert agent"，输入 3D 资产的四个正交视图，估计密度 $\rho$（kg/m³）、杨氏模量 $E$（Pa）、泊松比 $\nu$（无量纲），用于控制物体的质量分布、刚度和受力形变响应。

### 3.2 Sim2Real 环境对齐

**世界坐标系对齐**：以机器人几何作为公共锚点，将 3DGS 世界坐标系 $\mathcal{R}_{gs}$ 与仿真器 URDF 坐标系 $\mathcal{R}_{urdf}$ 对齐。先分别从两侧生成机器人几何点云，再用 ICP 求解刚性变换：

$$\mathcal{T}_{scene} = \arg\min_{R\in SO(3), \mathbf{t}\in\mathbb{R}^3} \sum_i \|(R\mathbf{p}_i^{gs}+\mathbf{t}) - \mathbf{q}_i^{urdf}\|^2 \tag{3}$$

用大白话说：找一个旋转加平移，使 3DGS 重建出的机器人点云尽量对齐到 URDF 模型定义的机器人点云上，进而把整个场景变换到仿真器坐标系下。

**相机位姿对齐**：把相机对齐问题转化为最小化渲染图与真实参考图之间的光度误差：

$$\mathcal{L}_{cam} = \|\mathcal{R}(\mathcal{T}_{cam}) - I_{real}\| \tag{4}$$

（$\mathcal{R}(\cdot)$ 是可微渲染函数，$\|\cdot\|$ 为 L1 范数）用大白话说：从一个粗略初始位姿出发，借助 3DGS 渲染器的可微性做梯度下降，不断微调虚拟相机的位置和朝向，直到它"拍"出来的仿真画面与真实相机拍的照片在像素级上尽量重合。

### 3.3 整体场景增强（Holistic Scene Augmentation）

为避免策略在静态数字孪生上过拟合，论文设计四类同时进行的随机化：**物体级增强**（随机化可交互物体的 6-DoF 位姿与均匀缩放）、**相机视角增强**（以对齐后的部署视角为中心施加随机平移旋转扰动）、**光照增强**（对 3DGS 高斯的视觉属性做整体缩放/偏移模拟对比度亮度变化，并逐高斯独立加噪声模拟传感器噪声）、**轨迹增强**（用逆运动学求解器让末端执行器先到达一个相对目标点有随机位置偏移的中间路点，再抵达最终目标，打破轨迹确定性）。消融实验（见下）证明这四类增强需要整体组合使用，仅做物体位姿随机化远远不够。

## 四、关键结果

实验平台：LeRobot 框架 + 双 RGB 相机（640×480），仿真数据在单张 NVIDIA RTX 5060 Ti GPU（+ i5-14400F CPU）上生成，策略推理用 NVIDIA H20 GPU。评测策略为 Diffusion Policy（DP，单任务）与 π0（通用 VLA）。主指标为 35 次连续试验的成功率（20 秒内完成判定为成功）。共设计 8 项真实任务：Stack Cubes、Pick & Place、Deformable Pick & Place、Upright Bottle、Move Bottle、Drawer Close、Box Close、Wiping。

**真实数据 + 仿真数据的协同效应（部分任务，成功率）**：

| 方法 | 数据来源 | Stack Cubes | PickPlace | Deformable PickPlace | Wiping |
|---|---|---|---|---|---|
| DP | 50 Real | 0.60 | 0.71 | 0.77 | 0.91 |
| DP | 100 RoboSimGS | 0.57 | 0.86 | 0.86 | 0.91 |
| DP | 50 Real + 50 RoboSimGS | **0.69** | **0.83** | **0.86** | **0.94** |
| π0 | 50 Real | 0.40 | 0.94 | 0.97 | 0.00 |
| π0 | 100 RoboSimGS | 0.54 | 0.91 | 0.94 | 0.88 |
| π0 | 50 Real + 50 RoboSimGS | **0.54** | **0.94** | **0.97** | **0.86** |

值得注意的是 π0 用 50 条真实数据在 Wiping 任务上完全失败（0.00），而混入 50 条 RoboSimGS 数据后提升到 0.86，体现出仿真数据对稀缺任务模式的强补充作用。

**消融（DP，50 条 RoboSimGS 数据，成功率对比）**：

| 消融设置 | Deformable PickPlace | Wiping |
|---|---|---|
| 完整 RoboSimGS | 0.69 | 0.85 |
| w/o 物理估计 | 0.54 | 0.51 |
| w/o 整体场景增强 | 0.51 | 0.00 |

去除 MLLM 物理估计后，接触密集型任务（Wiping、Deformable PickPlace）性能大幅下降；仅做物体位姿随机化而不做完整场景增强会导致策略过拟合到静态相机视角/光照/确定性轨迹，性能进一步崩溃（Wiping 直接降为 0）。

**泛化到未见设置**（DP 成功率，测试光照/物体尺寸/场景杂乱/相机位姿/桌面外观变化）：50 条真实数据在这些扰动下成功率普遍在 0-0.23（几乎失效），而 50 条 RoboSimGS 数据成功率在 0.46-0.63，50 真实 + 50 RoboSimGS 混合达到 0.54-0.71，泛化能力显著优于纯真实数据。

**数据生成效率**：8 个任务的仿真数据采集成功率在 0.93-1.00 之间，单条演示生成耗时 6.2-10.3 秒，单卡（RTX 5060 Ti）日产超过 10,000 条演示，而全职人工遥操作日产约 1,000 条，吞吐提升超过 10 倍。

**数据规模效率**：在 Stack Cubes 任务上，200 条 RoboSimGS 仿真演示训练出的 DP 策略性能已接近 100 条真实演示训练的水平。

**跨域一致性**：Sim-to-Real 成功率与 Real-to-Real 高度接近，验证了零样本 Sim2Real 迁移的有效性；Real-to-Sim 表现同样良好，说明仿真环境是真实场景的高保真数字孪生，可反过来用于评估真实数据训练出的策略。

## 五、评价与展望

**优点**：RoboSimGS 的核心贡献是把"MLLM 自动推断关节结构 + 物理参数"引入 R2S2R 流水线，用一套自动化的视觉-语言推理取代了此前工作（Robo-GS、SplatSim、RoboGSim 等）中静态、非交互的资产假设，是对纯视觉照片级真实感路线的关键补充。消融实验清晰地量化了物理估计与整体场景增强各自的贡献，尤其是 Wiping 这类接触密集型任务对物理参数极为敏感（去掉物理估计后成功率近乎腰斩），为"仿真数据的物理保真度与视觉保真度同等重要"提供了直接证据。数据生成效率（单卡万级/天）与数据规模-性能对照（200 仿真 ≈ 100 真实）也给出了较为清晰的成本-效益量化。

**局限与开放问题**：论文自陈的主要局限是场景重建流程仍然耗时且复杂——每个新场景约需 10 分钟人工扫描，虽是一次性成本，但对大规模、多场景部署仍构成瓶颈，作者将更快的 3D 重建方法列为未来工作。此外，物理参数估计（$\rho, E, \nu$）完全依赖 GPT-4o 对物体外观的常识推理，论文未给出这些估计值与真实测量值的定量误差对照，对于材质外观相似但力学性质迥异的物体（论文自己也承认这是长期挑战），估计的可靠性边界并不清楚。评测规模上，8 项任务、35 次试验、单臂桌面操作的设置仍偏小，物体类别和场景多样性有限，尚未验证在更大规模、更杂乱真实环境下的可扩展性。关节推断依赖"MLLM 提议 + AffordDex 分割 + MLLM 定关节参数"三步串联的流水线，论文没有专门分析部件误分割导致的关节轴误判会如何在下游任务中累积传播。与同属 3DGS-R2S2R 路线的 RoboGSim、Re3Sim、RL-GSBridge、SplatSim 相比，RoboSimGS 的差异化卖点在于"关节化 + 可变形物体的物理交互"，但论文只在 Related Work 中做了定性区分，未提供同一任务集上的头对头定量比较，后续工作可以补齐这一环。可能的改进方向包括：加速/自动化单场景重建以降低部署门槛，将 MLLM 物理估计与真实力学测量（如力传感器标定）结合做闭环校准，以及扩展到双臂或更长时程任务以检验关节资产在更复杂交互链条下的鲁棒性。

## 参考

- Lou et al., *Robo-GS: A physics consistent spatial-temporal model for robotic arm with hybrid representation*, arXiv:2408.14873, 2024（首个引入 3DGS 混合表示做机器人数字资产的 R2S2R 工作，但缺乏物理交互性）
- Qureshi et al., *SplatSim: Zero-shot sim2real transfer of RGB manipulation policies using Gaussian Splatting*, arXiv:2409.10161, 2024（用 3DGS 实现零样本 sim2real，但局限于刚性资产）
- Li et al., *RoboGSim: A real2sim2real robotic Gaussian Splatting simulator*, arXiv:2411.11839, 2024
- Jiang et al., *DexMimicGen: Automated data generation for bimanual dexterous manipulation via imitation learning*, ICRA 2025（自动数据生成但依赖固定仿真资产）
- Chi et al., *Diffusion Policy: Visuomotor policy learning via action diffusion*, IJRR 2023；Black et al., *π0: A vision-language-action flow model for general robot control*, 2024（论文实验中使用的评测策略骨干）
