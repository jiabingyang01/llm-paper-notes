# MoWorld：一个"闪电"世界模型

> **论文**：*MoWorld: A Flash World Model*
>
> **作者**：MoWorld Team（核心贡献者 Deyi Ji, Tianrun Chen, Xin Zhang, Jiale Yang, Qi Zhu et al.）
>
> **机构**：KOKONI 3D, Moxin Technology；浙江大学（Zhejiang University）
>
> **发布时间**：2026 年 07 月（arXiv 2607.06216）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.06216) | [PDF](https://arxiv.org/pdf/2607.06216)
>
> **分类标签**：`real-time-world-model` `camera-control` `autoregressive-distillation` `NPU-deployment`

---

## 一句话总结

MoWorld 以 Wan2.2-A14B（14B MoE 视频基座）为骨干，通过"几何感知数据引擎 + 课程式跨帧预训练 + 免 ODE 初始化的自回归蒸馏（50 步压到 4 步）+ 算法-系统-硬件协同"四支柱，在华为 Ascend NPU 上实现最高 50 FPS 的实时交互式相机可控世界模拟，平均推理成本仅为现有世界模型的 30%–50%，并在 VBench-I2V 采样子集上取得 Average 85.22 的领先成绩。

## 一、问题与动机

作者提出一个判断：世界模型的下一阶段不在于继续把模型做大，而在于**做实用**——需要同时优化模型能力、计算效率、部署成本与推理帧率。对具身智能、自动驾驶这类延迟敏感的闭环应用，慢速生成会从根本上限制交互、规划与控制。

据此论文定义了一类新概念 **Flash World Model**：能够持续以 ≥30 FPS 生成的实时世界模型（图形学中 30 FPS 被视为实时渲染与流畅交互的下限）。作者指出现有主流世界模型（如 Genie 系列、Matrix-Game、Yume 等）都难以满足该标准。MoWorld 就是面向这一目标做的"数据-算法-系统-硬件"端到端协同设计，目标是在**不依赖高端 GPU**的前提下达到 50 FPS。

一个突出定位：MoWorld 声称是**第一个跑在 NPU（Neural Processing Unit）上的实时交互式世界模型**。由于 NPU 已广泛集成进边缘设备与智能系统，把世界模型搬到 NPU 上能显著降低部署成本、功耗与硬件门槛。论文实验在 Huawei Ascend CloudMatrix384 超节点上完成。

## 二、核心方法

MoWorld 由四个紧耦合支柱构成，覆盖世界模型的完整生命周期。

### 支柱一：几何感知数据引擎（Geometric Aware Data Engine）

不同于主流"松散配对的视频-文本片段"，MoWorld 主张高质量世界建模需要显式关联视觉外观、相机运动与可控信号。数据引擎分四阶段：

1. **数据采集**：开放域视频（提供视觉覆盖）+ 游戏交互数据（提供显式控制与几何）两条互补来源。
2. **几何补全与质量控制**：VLM 预过滤（画面洁净度、相机稳定性、运动合理性），再做动态遮罩、特征匹配、几何重建，并用多源几何验证剔除几何/视觉不可靠的样本。
3. **视觉-语言标注**：VLM 标注视觉风格、主体、环境背景、光照、色彩等场景级描述，并刻意把相机运动/时序标记/动作指令**排除在文本条件之外**（避免文本泄漏相机控制信号）。
4. **离线特征预计算**：视觉/几何/文本特征离线缓存，建立索引与检索，直接供大规模训练消费。

该协议沿用 VGGT-Omega 的几何中心式数据构造思路，并进一步为每个场景标注稠密 3D 点云。数据来自多达 500 名标注员的自建采集与合成流程。

### 支柱二：课程式跨帧预训练（Curriculum Cross-Frame Training）

在 Wan2.2 骨干上扩展为相机可控、长时程的生成框架。模型 $f_\theta$ 以初始帧 $x_0$、文本 $c^{\text{txt}}$、逐帧相机控制 $c_{\text{cam}}^{1:T}$ 为条件，预测未来帧。采用标准 diffusion / flow-matching 目标：

$$\mathcal{L}_{\text{pre}}(\theta) = \mathbb{E}_{x_{0:T},\, c^{\text{txt}},\, c_{\text{cam}}^{1:T},\, t}\left[\left\| f_\theta\!\left(x_0, c^{\text{txt}}, c_{\text{cam}}^{1:T}, t, z_t\right) - v_t \right\|_2^2\right]$$

**用大白话说**：给模型一张起始画面、一段文字、一条相机轨迹和一段被加了噪声的未来视频，让它预测"去噪方向"（速度场 $v_t$），学会在服从相机控制的前提下把干净的未来帧还原出来。这个双向 DiT 模型是后续蒸馏的教师。

**相机控制注入**：逐帧内参 $K_i$ 与外参 $T_{c2w}^{(i)}$ 转成每个像素的 Plücker 射线坐标（世界坐标射线方向 $d_w$ 与力矩 $m_w$ 的拼接），Camera Adapter $A_\phi$ 把它投影到视觉 token 空间，在 patch-token 层面与噪声 latent 相加：

$$C_{\text{cam}} = A_\phi(R_{\text{cam}}) \in \mathbb{R}^{N\times d}, \qquad X_{\text{cam}}^{(t)} = \text{PatchEmbed}(z_t) + C_{\text{cam}}$$

**用大白话说**：把"每个像素对应的相机射线"当作附加坐标信息，直接贴到每个图像块上，让视觉模型在建模每个时空块时就带上视角约束——几何一致性与生成过程被紧耦合。

**课程设计**：视频长度渐进增加——Short-Clip（$T\in\{125,250\}$，练外观生成、首帧一致性、基础相机控制）→ Medium-Clip（$T\in\{500,1000\}$，练时序连贯与物体定位、缓解中程漂移）→ Long-Clip（$T=2000$，练长上下文、空间记忆、重访/回看/场景再入，防止时空漂移）。大部分训练在低成本的短/中片段，长片段只在最后阶段引入，兼顾质量与训练成本。骨干保留 Wan2.2 的高/低噪声 MoE 专家（高噪声专家建模全局结构与大尺度运动，低噪声专家精修纹理边缘等细节）。

**NPU 训练系统**：高/低噪声专家分配到两个 NPU 资源池 + 离线缓存；用 FSDP 分片参数。由于注意力显存随序列长度平方增长——

$$L = \mathcal{O}(T), \qquad \text{AttnMemory} = \mathcal{O}(T^2)$$

因此短/中片段用 Ulysses SP（按注意力头切分），长片段（$T=2000$）改用 Unified Sequence Parallelism（USP，token 级切分），配合 HCCL 通信与 CANN 运行时的融合 NPU 注意力核（分块在线 softmax，不显式物化 $L\times L$ 注意力矩阵）。

### 支柱三：自回归蒸馏（Autoregressive Distillation）

把带双向窗口建模能力的预训练教师，转化为少步（few-step）自回归学生，兼顾"少步生成压缩采样"与"双向窗口→因果自回归"两个目标。三个组件：

**(1) History Context 选择**：自回归生成时历史无限增长，全喂进窗口会让显存与注意力长度爆炸，只用最近段又会丢失早期外观与重访视角。MoWorld 从已生成 latent 中选三类：最近帧（短程连续）、早期帧（稳住外观与全局布局）、相机相关帧（处理重访/环绕）。相机相关帧的检索靠**相机 latent 向量**：算当前 chunk 的平均相机 latent，与各候选历史帧的相机 latent 比距离，距离越小视角越近。History Bank 存 **latent 帧而非 KV Cache**，大幅降低常驻显存；RoPE 用全局 latent 时间 ID，历史 token 只参与生成、不作预测目标。

**(2) AR Flow Matching 预训练**：双向模型不能直接当自回归生成器用。该中间适配阶段用**真值前缀历史**（GT prefix）+ 当前 chunk 监督，把双向权重适配到因果、分块的接口。chunk 索引 $i\sim\text{Uniform}\{0,\dots,i_{\max}\}$，历史库 $\mathcal{M}_{<i}$ 由真值 latent 构成，锚帧 $X_i^{\text{anchor}}$ 取前一个 chunk 的最后一帧 latent。目标：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}\left[\left\| f_\theta^{\text{AR}}\!\left(X_i^{\text{anchor}}, c^{\text{txt}}, C_i^{\text{cam}}, \mathcal{M}_{<i}, t, Z_i^t\right) - V_i^t \right\|_2^2\right]$$

$$\mathcal{L}_{\text{AR}} = \mathcal{L}_{\text{FM}} + \lambda_{\text{anchor}}\,\mathcal{L}_{\text{anchor}}, \qquad \lambda_{\text{anchor}} = 1.0$$

其中锚点损失 $\mathcal{L}_{\text{anchor}} = \lVert \hat{X}_i^0 - X_i^{\text{anchor}} \rVert_2^2$ 约束 chunk 边界重叠 latent 的连续性。

**用大白话说**：先让学生"扶着正确答案（真值历史）学走路"——在稳定的历史上下文里练会怎么读历史、怎么在 chunk 边界对齐，避免一开始就用自己生成的脏历史递归污染，也省掉了很多蒸馏管线里昂贵的多步教师采样/ODE 初始化。

**(3) 免 ODE 初始化的 Self-Forcing 蒸馏**：沿用 DMD 式分布匹配，但把学生采样路径替换成**部署时真实使用的自回归 rollout**——学生逐 chunk 生成整段 latent 视频：

$$\hat{Z}_\theta = \text{Rollout}_\theta^{\text{AR}}(x_0, c^{\text{txt}}, C^{\text{cam}}, \epsilon;\, \mathcal{S})$$

训练在 **Fake 阶段**（冻结学生与教师，让 Fake Model 学习当前学生分布的速度场）与 **Student 阶段**（冻结教师与 Fake，教师给真实分布速度 $v_t^{\text{real}}$、Fake 给学生分布速度 $v_t^{\text{fake}}$）之间交替。分布匹配梯度与学生目标：

$$g_{\text{DMD}} = v_t^{\text{real}} - v_t^{\text{fake}}, \qquad \mathcal{L}_{\text{student}} = \mathbb{E}\left[\text{sg}(g_{\text{DMD}})\cdot\hat{Z}_\theta\right]$$

**用大白话说**：教师和 Fake Model 分别估计"真实视频该往哪走"和"学生现在往哪走"，两者之差就指明学生分布要朝教师靠拢的方向。关键是让学生在训练时就用推理时的自回归链条 rollout，于是边界误差、历史选择误差、长程漂移都在训练中被暴露和纠正，同时把生成压进 4 步。

### 支柱四：实时推理的算法-系统-硬件协同

三个层级优化，目标是资源受限（单 NPU）下低显存、以及低延迟：

- **Pipeline 级**：On-demand Module Loading——编码器只在首步跑一次产出初始条件后即释放权重，DiT 与解码器再顺序加载；多帧每步生成并缓冲以支持流式显示。
- **Parallelism 级**：Hierarchical Sequence Parallelism（头级 + token 级两层切分）；把 Encoder⊕DiT 的 NPU 与解码器 NPU 解耦；组内 ring 通信、跨组 AllToAll，经 HCCL 执行。
- **Kernel 级**：Dynamic Mixed-Precision Quantization——权重/激活 BF16 经一步 kernel warm-up 量化到 INT8（约 2× 显存缩减），INT8×INT8 累加成 INT32 再反量化回 BF16，仅作用于 DiT 块（编码器因精度敏感保持 BF16）；HBM-efficient End-to-end Attention——RMSNorm 稳定激活尺度 + NPU 融合注意力核 + 在线 softmax，减少 HBM 数据搬运。

## 三、实验结果

评测遵循近期视频生成式世界模型协议，用 VBench-I2V 官方图生视频基准的**采样子集**，八项指标：SC（主体一致性）、BC（背景一致性）、MS（运动平滑）、DD（动态程度）、AQ（美学质量）、IQ（成像质量）、I2V-S（图-视频主体一致）、I2V-B（图-视频背景一致）；Quality 为前六项加权均值（DD 权重 0.5，其余为 1），Average 为八项算术平均。

**表 1：VBench-I2V 采样子集**（加粗=最优，下划线=次优）

| Model | SC↑ | BC↑ | MS↑ | DD↑ | AQ↑ | IQ↑ | I2V-S↑ | I2V-B↑ | Quality↑ | Average↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| CameraCtrl | 95.44 | 95.36 | **98.92** | 38.00 | 54.86 | 66.27 | 95.95 | 97.48 | 78.15 | 80.29 |
| SEVA | 87.07 | 90.18 | 97.62 | **62.00** | 52.65 | 61.40 | 93.83 | 95.64 | 76.35 | 80.05 |
| Lingbot | 93.91 | 94.94 | 97.80 | 46.00 | 57.54 | **71.20** | 97.27 | 97.14 | 79.71 | 81.97 |
| **MoWorld** | **95.47** | **95.68** | 98.13 | 60.00 | **64.63** | 71.11 | **98.04** | **98.68** | **82.73** | **85.22** |

MoWorld 在 Quality（82.73）与 Average（85.22）上均领先，SC/BC/I2V-S/I2V-B/AQ 取得最优，说明相机条件训练没有显著牺牲外观一致性、视觉美学与首帧保真。

**表 2：自建相机/世界模型数据集采样子集（VBench 风格八维）**

| Model | SC↑ | BC↑ | MS↑ | DD↑ | AQ↑ | IQ↑ | I2V-S↑ | I2V-B↑ | Quality↑ | Average↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| SEVA | 90.08 | 92.23 | 98.48 | 56.00 | 53.53 | 50.09 | 96.40 | 93.18 | 74.98 | 78.75 |
| CameraCtrl | 84.31 | 93.17 | 97.86 | **100.00** | 46.26 | 54.78 | 93.27 | 95.05 | 77.52 | 83.09 |
| WorldPlay | **95.88** | 94.84 | **99.29** | 88.00 | 51.76 | 70.92 | **99.00** | **99.12** | 83.04 | 87.35 |
| Lingbot | 91.82 | 93.93 | 98.01 | 92.00 | 57.09 | 73.30 | 97.08 | 96.64 | 83.66 | 87.48 |
| **MoWorld** | 95.46 | **96.33** | 98.12 | **100.00** | 59.40 | **76.65** | 98.22 | 98.33 | **86.54** | **90.31** |

在自建交互控制数据上 MoWorld 的 Quality（86.54）与 Average（90.31）领先，DD 满分且 BC/IQ 最优，作者据此说明模型能按指定相机运动产生视角变化，同时不明显损失主体一致、背景稳定与图像条件保持。

**效率与部署主张**（论文正文陈述，未给独立基准表）：支持 14B MoE 世界模型的预训练/蒸馏/推理；蒸馏把去噪步数从 50 压到 4；最高 50 FPS 实时推理；平均推理成本降到现有世界模型的 30%–50%；端到端部署成本在实际设置下降 30%–50%。

**定性与应用**：长时程（最长 2000 帧）交互生成覆盖室内图书馆、哥特厅堂、金字塔遗迹、商场走廊、夜景住宅等，键盘图标标注前后/左右/转向等动作；下游支持视频风格迁移、视频编辑、点云重建（SfM + 单目深度）、3D Gaussian Splatting 重建，以及导航（VLA 预测动作序列驱动 MoWorld 作为环境模型，图中标注 "Coming Soon"）。

## 四、局限性

- **缺乏效率与消融的定量证据**：50 FPS、成本 30%–50%、显存约 2× 缩减等关键卖点均以正文陈述形式给出，正文页未见延迟/吞吐/显存的对照基准表，也没有四支柱、蒸馏步数（50→4）、History 选择策略的消融，工程收益难以独立核验。
- **数据与几何引擎完全私有**：500 人自建采集 + 合成、VGGT-Omega 系几何构造，均不可复现；评测又用 VBench-I2V 与自建集的**采样子集**而非完整基准，且部分基线（CameraCtrl、SEVA）本质是相机控制方法而非实时世界模型，可比性有限。
- **具身/导航能力停留在演示层**：VLA-Navigation 标注 "Coming Soon"，导航仅给定性图示，没有任何具身任务的成功率/SPL 等量化指标，"面向具身智能"更多是愿景而非实证。
- **强绑定 Huawei Ascend NPU**：训练需 CloudMatrix384 超节点，诸多优化（HCCL/CANN 融合核、INT8 warm-up 量化）依赖 NPU 软硬件栈，在通用 GPU 上的可迁移性与真实成本未评估；"无需高端 GPU"的表述对训练侧并不成立。
- **方法层新意有限**：蒸馏骨架（Self-Forcing / DMD / Causal Forcing）与骨干（Wan2.2）多为既有工作组装，核心增量在系统-硬件协同与 NPU 落地，学术方法贡献相对增量。
- **长程自回归的漂移未量化**：2000 帧下的几何/身份一致性只有定性展示，缺少随时间的误差累积曲线或长程 3D 一致性指标。

## 五、评价与展望

**优点**：这是一篇"把世界模型做实用"的系统工程论文，价值在于端到端协同——从几何感知数据、课程式长时程预训练，到免 ODE 初始化的自回归蒸馏，再到 NPU 上的 pipeline/parallelism/kernel 三级部署优化，形成了少见的完整落地闭环。几个设计有借鉴价值：History Bank 存 latent 而非 KV Cache 来控制常驻显存；用相机 latent 距离检索"相机相关历史"以支持重访/环绕；把蒸馏的学生采样路径直接替换成推理时的自回归 rollout，从而在训练中暴露并纠正边界误差与长程漂移。相机以 Plücker 射线在 patch-token 层注入的做法与 CameraCtrl / CamCo / VD3D 一脉相承但更强调几何耦合。

**与公开工作的关系**：骨干 Wan2.2；蒸馏沿 DMD（Yin et al.）、Self-Forcing（Huang et al., 2506.08009）、Causal Forcing（Zhu et al., 2602.02214）一线；序列并行用 Ulysses SP / USP。同期实时/交互世界模型密集——Matrix-Game 2.0、Genie 3、Yume-1.5、WorldPlay、Hunyuan-GameCraft-2、PAN、Hy-world 1.5 等。MoWorld 的差异化主要在三点：NPU 原生部署、最长 2000 帧的长时程、以及显式相机可控；但在纯生成质量上与 WorldPlay、Lingbot 等的差距并不悬殊（表 2 Average 90.31 vs 87+），真正的护城河是"同等质量下的成本/帧率"，而这恰恰缺少硬基准支撑。

**开放问题与改进方向**：(1) 补齐效率基准——在统一硬件上给出 FPS/延迟/显存/能耗与质量的帕累托前沿，以及四支柱和蒸馏步数的消融；(2) 把导航从 "Coming Soon" 落到具身任务的量化闭环（成功率、可交互物理一致性），检验世界模型作为环境模型的真实价值；(3) 量化长程（≥2000 帧）自回归的漂移与 3D 一致性衰减；(4) 验证在通用 GPU 上的可迁移性，避免结论被单一硬件栈绑定；(5) 相机之外引入更丰富的动作/物理条件（力、接触、物体级操作），才能更贴近具身操作而非仅相机漫游。

## 参考

1. Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models*. arXiv:2503.20314, 2025.（骨干）
2. Huang et al. *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion*. arXiv:2506.08009, 2025.
3. Zhu et al. *Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation*. arXiv:2602.02214, 2026.
4. Yin et al. *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models*. CVPR 2025.（DMD 式蒸馏）
5. He et al. *CameraCtrl: Enabling Camera Control for Text-to-Video Generation*. arXiv:2404.02101, 2024.（相机控制基线）
