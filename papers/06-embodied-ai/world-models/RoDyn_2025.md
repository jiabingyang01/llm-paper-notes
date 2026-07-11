# RoDyn：驯服面向机器人操作的交互式机器人动态 2.5D 世界模型

> **论文**：*RoDyn: Taming Interactive Robot-Dynamic 2.5D World Model for Robotic Manipulation*
>
> **作者**：Chuanrui Zhang, Zhengxian Wu, Guanxing Lu, Yansong Tang, Ziwei Wang
>
> **机构**：Nanyang Technological University（新加坡）；Tsinghua University（清华大学，北京）
>
> **发布时间**：2025 年 10 月（arXiv 2510.09036，v2 修订于 2026 年 6 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.09036) | [PDF](https://arxiv.org/pdf/2510.09036)
>
> **分类标签**：`2.5D世界模型` `机器人操作` `深度与掩码先验` `自回归token预测` `模型基强化学习`

---

## 一句话总结

RoDyn 用 RGB + 深度 + 动态掩码构成的"2.5D"离散 token 空间，配合掩码引导的自回归 Transformer，在保持 2D 视频世界模型推理速度的同时逼近 3D 世界模型的空间感知能力，真实机器人部署上把模仿学习成功率从 2D 基线 iVideoGPT 的 45% 提升到 87%（绝对提升 42 个百分点），甚至略超直接用真实演示数据训练的策略（83%）。

## 一、问题与动机

现有动作条件世界模型分裂为两条路线：（1）以 iVideoGPT 为代表的纯 2D 视频自回归模型，推理快，但只建模 RGB 像素，缺乏空间/运动学推理，在遮挡或复杂多物体交互下容易产生物理上不合理的"幻觉动力学"；（2）以同团队前作 GWM 为代表的显式 3D 世界模型（基于 3D Gaussian Splatting），空间感知强，但依赖高质量稠密 3D 重建，在单目观测下重建质量脆弱、难以扩展到多样场景，且计算/显存开销大，推理速度不足以支撑高频机器人闭环控制。论文将这一现象刻画为"3D 空间感知 vs 2D 推理效率"的传统权衡区（trade-off zone），提出 RoDyn 试图在其中实现突破：既不做代价高昂的稠密 3D 重建，也不满足于物理无感的纯 2D 像素，而是构造一个几何感知的紧凑"2.5D"（RGB + 深度 + 掩码）潜在空间。

## 二、核心方法

**2.5D 观测空间。** 每一帧观测 $\mathbf{O}_t$ 由 RGB 图像、深度图与以机器人/交互物体为中心的动态掩码三模态组成。真实机器人实验用 Intel RealSense D435i 直接采集 RGB-D；公开数据集（BAIR、RoboNet、DROID）没有深度真值，用 Video Depth Anything 估计度量深度，用 Grounding DINO + SAM 2 生成时序一致的 agent-centric 掩码作为伪标注。

**Robot-Dynamic Tokenizer（双编解码器）。** 沿用 VQGAN 式 tokenizer 思路，但设计了 context 对 $(E_c,D_c)$ 与 predictive 对 $(E_p,D_p)$ 两套编解码器：

$$
\mathbf{Z}^c_t = E_c(\mathbf{O}_t),\quad \hat{\mathbf{O}}_t = D_c(\mathbf{Z}^c_t),\quad t=1,\dots,T_0
$$

$$
\mathbf{Z}^d_t = E_p(\mathbf{O}_t \mid \mathbf{O}_{1:T_0}),\quad \hat{\mathbf{O}}_t = D_p(\mathbf{Z}^d_t \mid \mathbf{O}_{1:T_0}),\quad t=T_0+1,\dots,T
$$

大白话：开局 $T_0$ 帧的背景、桌面等静态场景变化很少，专门用一路网络编码成"底图 token"；后续帧相对开局画面大部分内容冗余，真正变化的只有机器人和被操作物体那一小块区域，所以用第二路网络、通过与历史上下文 token 的交叉注意力条件化，只负责预测这部分"动态增量"，从而大幅省算力。

三模态融合不是简单通道拼接，而是 **RGB-dominated cross-attention**：深度层与掩码层并行处理后，反过来去 query/调制核心 RGB 语义特征，再经 mask-driven fusion 算子（记为 $\otimes$）把掩码先验注入，让 token 表征capacity自适应地集中到主动的机器人-物体交互区域。Tokenizer 训练目标为：

$$
\mathcal{L}_{\text{tokenizer}} = \sum_{t=1}^{T_0}\mathcal{L}_{\text{VQGAN}}(\mathbf{O}_t; E_c,D_c) + \sum_{t=T_0+1}^{T}\mathcal{L}_{\text{VQGAN}}(\mathbf{O}_t; E_p,D_p)
$$

其中 $\mathcal{L}_{\text{VQGAN}}$ 包含 L1 重建损失、commitment loss、感知损失，可选加对抗损失。

**Mask-guided Autoregressive Transformer。** 将 token 序列化为 $\mathbf{X}_T=(\mathbf{Z}^c_1,[S_1],\dots,\mathbf{Z}^c_{T_0},[S_{T_0}],\mathbf{Z}^d_{T_0+1},[S_{T_0+1}],\dots,\mathbf{Z}^d_T)$，并在每一步转移之间插入 Embodiment Transition Token $[S_t]$ 作为"物理边界标志"：

$$
[S_t] = [S_{\text{base}}] + P_{\text{act}}(\mathbf{a}_t) + P_{\text{mask}}(\mathbf{m}_t)
$$

大白话：不是简单把动作向量拼进序列，而是把末端执行器位姿动作 $\mathbf{a}_t$ 和当前提取的物理掩码 $\mathbf{m}_t$ 都揉进这一个 token 里，相当于给模型一个明确的"接下来会发生什么样物理边界事件"的提示。Backbone 采用 LLaMA 风格因果 Transformer（RMSNorm + SwiGLU + RoPE），训练目标仅作用于动态 token（context token 作为只读前缀 prompt）：

$$
\mathcal{L}_{\text{AR}} = -\sum_{t=T_0+1}^{T}\log P(\mathbf{Z}^d_t \mid \mathbf{X}_{<t})
$$

大白话：不在"背景没变"的部分浪费训练信号，逼模型把容量都用在预测真正会变化的交互区域上。推理阶段严格自回归滚动生成，新预测 token 反馈回序列以模拟闭环操作 rollout。

**两种下游用法。** （1）模仿学习数据增强：给定初始观测和一条成功动作轨迹，RoDyn 合成多条几何一致的"想象"演示 rollout（仅合成图像+空间拓扑，动作仍按 chunk 单独预测），用来训练 3D Diffusion Policy；（2）模型基 RL：仿照 MBPO 框架，RoDyn 充当 MDP 转移模型 $\mathcal{P}$，用真实回放池 $\mathcal{D}_{\text{real}}$ 训练后生成想象回放池 $\mathcal{D}_{\text{imag}}$，DrQ-v2 actor-critic 在 $\mathcal{D}_{\text{real}}\cup\mathcal{D}_{\text{imag}}$ 上联合更新，并额外加一个辅助 reward head 预测逐步奖励。

## 三、关键结果

真实机械臂部署（GALAXEA A1 + RealSense D435i；10 个任务采集训练数据，每任务 50 条人类演示；选 5 个代表性任务做硬件评测，策略为 3D Diffusion Policy）：

| 任务 | iVideoGPT（2D基线，补深度） | RoDyn | GT（真实数据训练） |
|---|---|---|---|
| Stack Green Cup | 50% | 90% | 90% |
| Stack Two Cups | 30% | 70% | 60% |
| Stack Cubes | 45% | 90% | 85% |
| Pick and Place Banana | 55% | 95% | 95% |
| Pick and Place Bread | 45% | 90% | 85% |
| **总成功率** | **45%** | **87%** | **83%** |

RoDyn 比 2D 基线绝对提升 42 个百分点，且略超直接用真实数据训练的策略，论文解释为合成轨迹几何一致、掩码引导能"抹平"次优人类演示的噪声。真实高分辨率（256×256）视觉指标上，RoDyn 也全面领先：PSNR 38.11 / SSIM 0.941 / LPIPS 0.091，对比 iVideoGPT 的 32.14 / 0.901 / 0.130。

公开数据集动作条件生成（FVD↓/PSNR↑/SSIM↑/LPIPS↓/AbsRel↓）：BAIR 上 RoDyn 为 60.9/23.82/0.896/0.051/0.045，全面优于 MaskViT、iVideoGPT；RoboNet 上 RoDyn 的 PSNR(28.0)/SSIM(0.920)/LPIPS(0.032)/AbsRel(0.031) 均为最优，但 FVD(67.6) 略逊于 iVideoGPT(65.8)，是全文唯一一处未拿到最优的指标；DROID 上（follow BridgeV2W 协议）RoDyn 取得 PSNR 24.37/SSIM 0.877/LPIPS 0.076/FVD 122.7，全面超过 IRASim、Cosmos、EVAC-Cast、BridgeV2W 等最新基线。

Meta-World 六任务视觉 MBRL（DrQ-v2 + MBPO 框架，对比 GWM/iVideoGPT/DreamerV3）：RoDyn 平均成功率比最强基线绝对提升超过 10%，六个子任务全部拿到 SOTA，hammer 任务上相对最大提升约 40%。

BAIR 消融：Tokenizer 效率上，Naive Concat（简单通道拼接，序列长 3 倍）FVD 739.2、推理耗时 860s；RD Tokenizer 把 FVD 降到 67.6，推理仅需 10s——加速 86 倍。模块贡献上，Baseline(2D) FVD 70.2/PSNR 22.50 → +RD Tokenizer FVD 67.5/PSNR 22.67 → +Mask-Guided AR FVD 70.6/PSNR 22.84 → 完整 RoDyn FVD 67.6/PSNR 23.22，两个模块各自贡献增量，组合后综合最优。

## 四、评价与展望

**优点。** 提出"2.5D"这一介于纯 2D 视频模型与显式 3D（3DGS）世界模型之间的折中表征，用 RGB 主导交叉注意力 + 动态掩码融合把深度和掩码先验较优雅地注入 token（而非简单通道拼接），并用消融实验（86 倍推理加速）证明了这一设计的效率收益；下游同时验证了真实机器人模仿学习和仿真 MBRL 两条路径的增益，证据链较完整，42% 的真实成功率提升和"合成数据训练的策略略超真实数据训练"的结论具有一定说服力。

**局限。** 论文结论部分自陈：当前预训练数据规模和涉及的实体/动作空间多样性有限，尚未验证跨具身、跨动作空间的通用泛化能力，是未来扩展的主要方向。此外可以观察到几点局限：（1）深度和掩码先验本身来自 Video Depth Anything、Grounding DINO+SAM 2 等现成模型的伪标注，公开数据集上没有深度真值做校验，2.5D 表征精度不可避免会受限于这些上游模型的误差；（2）RoboNet 上 FVD 反而略逊于纯 2D 基线 iVideoGPT，说明分布级真实感（FVD）与逐帧保真度/深度精度（PSNR/AbsRel）并不总是同步提升，3D-aware 设计对时序连贯性的收益并非在所有指标上都是正向的；（3）真实机器人实验规模偏小（5 个部署任务、10 个训练任务、每任务 50 条演示），结论的统计显著性和跨场景鲁棒性有待更大规模验证；（4）与同团队前作 GWM（3DGS-based）的直接对比只出现在 Meta-World MBRL 一处实验，未在真实机器人模仿学习或 DROID 等场景中直接较量，"3D 显式表征 vs 2.5D token 表征"孰优孰劣的边界仍不够清晰。

**与其他公开工作的关系。** 相比 iVideoGPT（纯 2D VideoGPT 式世界模型），RoDyn 增加了显式几何与掩码信息；相比 GWM（3DGS-based 显式 3D 世界模型），RoDyn 用离散 token 化的 2.5D 表征换取推理效率，牺牲了完整多视角 3D 一致性；相比 TesserAct（RGB-DN 视频生成），RoDyn 补上了缺失的动作条件；相比同期 BridgeV2W（用 embodiment mask 把视频生成模型桥接为世界模型），RoDyn 在 DROID 上直接对比并取胜，说明"用掩码/深度做具身先验注入"是该子领域几个并行工作共同收敛的方向。

**开放问题。** 2.5D 离散 token 的空间分辨率和深度量化误差如何影响长时程 rollout 的累积漂移；RGB-dominated cross-attention 能否推广到多视角输入；辅助 reward head 在更复杂长程任务上的可靠性；以及跨 embodiment（不同机械臂/动作空间）预训练后的 zero-shot 迁移能力，这些论文均未给出实验证据，留待后续工作验证。

## 参考

- Wu et al. *iVideoGPT: Interactive VideoGPTs are Scalable World Models*, NeurIPS 2024 — 2D 自回归世界模型 baseline，RoDyn 的主要对比对象。
- Lu et al. *GWM: Towards Scalable Gaussian World Models for Robotic Manipulation*, arXiv:2508.17600 — 同团队 3DGS-based 显式 3D 世界模型。
- Zhen et al. *TesserAct: Learning 4D Embodied World Models*, arXiv:2504.20995 — RGB-DN 4D 具身世界模型，缺显式动作条件。
- Chen et al. *BridgeV2W: Bridging Video Generation Models to Embodied World Models via Embodiment Masks*, arXiv:2602.03793 — DROID 数据集上的最新对比基线，同样采用 embodiment mask 思路。
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, IJRR 2023 — RoDyn 下游模仿学习所用的策略头（3D Diffusion Policy）。
