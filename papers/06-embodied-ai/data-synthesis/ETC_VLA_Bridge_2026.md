# ETC：两座桥、一条路——用具身轨迹耦合数据把VLM桥接为可泛化VLA

> **论文**：*Two Bridges, One Pathway: From VLMs to Generalizable VLAs with Embodied Trajectory-Coupled Data*
>
> **作者**：Linqi Yin, Shiduo Zhang（并列一作、项目负责人）, Shenling Qiu（并列一作）, Chenxin Li, Zhaoyang Fu, Lei Xiao, Xiang Wang, Chenchen Yang, Zhe Xu, Pengfang Qian, Jingjing Gong, Xipeng Qiu, Xuanjing Huang, Yu-Gang Jiang（通讯作者）et al.
>
> **机构**：复旦大学可信具身智能研究所（Institute of Trustworthy Embodied AI, Fudan University，OpenMOSS 团队）、上海创新研究院（Shanghai Innovation Institute）
>
> **发布时间**：2026 年 06 月（arXiv 2606.08520）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.08520) | [PDF](https://arxiv.org/pdf/2606.08520)
>
> **分类标签**：`VLM到VLA迁移` `数据合成` `VQA式中间监督` `灾难性遗忘` `组合泛化`

---

## 一句话总结

论文提出 **ETC（Embodied Trajectory-Coupled）数据**——从机器人动作数据同源的场景/轨迹中离线派生出的 VQA 式视觉-语言监督，并设计 **Distribution Bridging → Objective Bridging → Retentive Adaptation** 三阶段、共享 next-token 预测目标的训练路径，把 VLM 的输入分布鸿沟与训练目标鸿沟分两步而非一步跨越；VLABench Track 1（in-domain）成功率从纯动作训练的 0.372 提升到 Distribution Bridging + Full ETC 的 0.648，真实 WidowX 咖啡包任务的 OOD 成功率从 41.67%（纯动作）提升到 91.67%（Gemini 自动生成的 OOD ETC），且少量 OOD ETC 即可在**不采集新动作数据** 的前提下诱导对未见视觉-语言组合的泛化（"compositional induction"）。

## 一、问题与动机

把预训练 VLM 变成可部署 VLA 策略，需要同时跨越两个**正交** 的鸿沟：

- **输入分布轴**：VLM 训练于网络规模的通用图文数据，而操作策略必须面对充满机器人特定视角、物体构型与交互几何的具身场景；
- **训练目标轴**：VLM 以 next-token 视觉语言理解为目标优化，而策略必须生成可执行的动作 token。

直接在机器人动作数据上微调 VLM，等于要求模型一步跨越两个鸿沟，学习曲线陡峭，且预训练期间习得的丰富泛化能力容易退化而非迁移。已有工作分为两类：一类在策略训练前做"具身推理增强"（如具身 VQA、空间 grounding），但其作为 VLA 初始化的价值未被系统验证，且获得的能力可能在后续纯动作训练中退化；另一类做"多模态协同训练"，但所用的通用多模态数据与具身操作行为的分布相距较远，是一种间接的桥接。论文要问：（1）具身多模态监督能否提供更强的 VLA 初始化？（2）什么样的多模态数据能最有效地桥接 VLM 与 VLA 的双重鸿沟？（3）如何设计训练配方，在扩展策略能力边界的同时尽量保留预训练知识？

## 二、核心方法

**ETC 数据的定义与三类监督**。ETC 是与动作数据来自同一具身场景和轨迹的 VQA 式视觉-语言监督，保持 VLM 原有的 next-token 预测目标，同时在输入分布上与 VLA 具身分布对齐。论文将其组织为抽象程度递减的三类：

- **Scene-Grounding ETC**（任务无关的场景感知）：空间查询探测几何关系（相对末端执行器方向、跨视角物体对应），物体描述查询识别/定位/描述操作相关属性；
- **Task-Oriented ETC**（以指令为条件）：affordance 查询预测最优抓取点或验证区域是否可稳定抓取，任务规划查询把指令分解为有序子动作、识别当前阶段或预判下一动作；
- **Action-Aligned Planning ETC**（离动作 token 最近的一环）：以图像平面关键点的形式预测给定子任务的夹爪未来轨迹。

三类 ETC 均离线、无需额外人工标注地从既有机器人轨迹中派生：几何标签通过相机标定（或按 [30] 把动作统一到相机系）确定性计算，语义与推理标签由 Gemini 生成后人工校验。

**三阶段训练路径**（共享 next-token 预测目标）：

- **Stage 1：Distribution Bridging**——只在 VLM 骨干上、纯用三类 ETC 数据训练，解冻 ViT 塔，把输入分布从通用图文流形搬运到具身流形，输出目标仍是文本 token，不引入动作；
- **Stage 2：Objective Bridging**——在 Stage 1 对齐后的表征上引入动作 token 预测，用同源场景/轨迹的大规模预训练动作数据与 ETC 联合训练（ETC batch 为动作 batch 的 25%），ETC 的作用从"建立"对齐转为"保持"对齐，防止动作学习侵蚀 Stage 1 的对齐效果；
- **Stage 3：Retentive Adaptation**——沿用 Stage 2 的协同训练机制，把动作数据换成目标部署场景数据，同时用为目标场景构造的 ETC 继续监督，实现"能力扩展而不侵蚀已有知识"。

三阶段的联合优化实现方式是：每个优化步同时取一个动作 batch 和一个 ETC batch，分别做前向/反向传播、累加梯度后统一执行一次优化器更新，从而在保持联合更新的同时解耦两个目标的梯度评估，减少相互干扰。

**Compositional Induction（组合诱导泛化）**。当 Stage 2 与 Stage 3 都联合 ETC 与动作数据训练时，对于目标部署中缺乏动作演示的视觉-语言组合（如未见目标物体、未见场景布局），只需构造覆盖这些条件的 ETC 并混入协同训练流（VLABench 实验中 OOD ETC 与 in-domain ETC 比例约为 6:94），策略即可把已习得的动作技能迁移到仅通过 ETC 见过的新条件上，而**无需为这些组合采集任何新的机器人动作演示**。

## 三、关键结果

实验以 PaliGemma-3B 为骨干（并在附录用 SigLIP+Gemma-2B、Qwen2.5-VL-3B 做消融），VLA 架构沿用 Pifast（PiFAST 动作分词），在 SimplerEnv、LIBERO、VLABench（Track 1 in-domain / Track 2 OOD 组合）三个仿真基准以及真实 WidowX 平台（Inserting Flower、Placing Coffee Bag 两个任务，基于 BridgeV2 预训练的 Pifast）上评测。

**Distribution Bridging 的效果**（纯动作训练下，对比 w/ vs w/o Stage 1）：

| 基准 | w/o Distribution Bridging | w/ Distribution Bridging | 提升 |
|---|---|---|---|
| SimplerEnv SR | 0.792 | 0.812 | +2.0 pt |
| LIBERO SR | 0.894 | 0.907 | +1.3 pt |
| VLABench T1 SR | 0.372 | 0.443 | +7.1 pt |
| VLABench T2 SR | 0.236 | 0.283 | +4.7 pt |

冻结 ViT 只做语言侧微调则 VLABench Bridge SR 从 0.812 降至 0.673，说明对齐需要联合视觉-语言适配，而非单纯语言侧调优；原始 PaliGemma（0.792）明显优于未做联合视觉-语言预训练的 SigLIP+Gemma（0.385），说明通用 VLM 预训练本身不足以实现具身对齐。

**Objective Bridging 的效果**：在 Distribution-Bridged 骨干上，ETC 协同训练相对纯动作训练在 SimplerEnv 上提升 SR +0.032，VLABench T1/T2 分别提升 +0.205/+0.116；CKA 分析显示纯动作的 Objective Bridging 表征逐渐偏离 Distribution-Bridged 表征，而 ETC 协同训练能显著抑制这种表征漂移。三类 ETC 的消融（VLABench Track1/Track2）：

| Stage 2 监督 | T1 SR | T1 PS | T1 IS | T2 SR | T2 PS | T2 IS |
|---|---|---|---|---|---|---|
| Action only | 0.443 | 0.531 | 0.537 | 0.283 | 0.351 | 0.365 |
| + Scene-grounding ETC | 0.562 | 0.696 | 0.736 | 0.407 | 0.481 | 0.473 |
| + Task-oriented ETC | 0.593 | 0.719 | 0.748 | 0.314 | 0.423 | 0.507 |
| + Action-Aligned Planning ETC | 0.621 | 0.744 | 0.771 | **0.438** | 0.499 | 0.482 |
| + Full ETC | **0.648** | **0.773** | **0.778** | 0.414 | 0.498 | **0.511** |

三类信号各有所长且互补：Action-Aligned Planning ETC 执行力最强，Task-Oriented ETC 意图理解最好（Track2 Intention Score 0.507），Scene-Grounding ETC 是不偏科的感知基础，Full ETC 在 Track1 综合最优。

**Retentive Adaptation（真实 WidowX，coffee-bag 任务，in-domain 分数）**：Stage2+Stage3 均带 ETC 得 95.83，仅 Stage3 带 ETC（无 Stage1/2 铺垫）只有 61.11，说明 Stage3 的收益依赖前序阶段已积累的可复用具身能力；纯从零训练基线为 63.88。

**Compositional Induction（组合泛化）**：VLABench Track2 混入 6% OOD ETC 后 PS/IS 分别提升 +0.8%/+13.2%（SR 基本持平）。真实机器人上更明显：coffee-bag 任务 OOD 成功率从纯动作训练的 41.67% 提升到 in-domain ETC 的 69.44%，再到 Gemini 自动生成 OOD ETC 的 **91.67%**；flower 任务中，in-domain ETC 对 OOD 无提升（27.08%→27.08%），而 OOD ETC（人工标注真实图像）把 OOD 成功率提到 **50.00%**，用 GPT-Image-2 生成的照片级目标条件图像构造 ETC（无需采集新视觉观测）也能提升到 37.50%（仍低于真实图像人工标注设置）。这表明 ETC 必须覆盖目标未见条件本身才能诱导泛化，且自动化标注/生成式构造为组合泛化提供了低成本、可扩展的路径。

## 四、评价与展望

**优点**：论文把 VLM→VLA 的鸿沟明确拆解为"输入分布"与"训练目标"两个正交维度，并证明二者需要分阶段、用同源数据桥接而非一步到位或依赖分布疏远的通用多模态数据，这一分析框架和 ETC 三类监督（场景/任务/动作对齐）的分级设计具有较好的可解释性，消融也做得较为完整（CKA 表征漂移分析、t-SNE 可视化、per-category ETC 贡献拆解）。"低成本 OOD ETC 诱导组合泛化而无需新增动作数据"这一发现有实际部署价值——用 Gemini/GPT-Image-2 等现成生成模型自动构造目标条件的 VQA 监督，比人工标注甚至更有效（coffee-bag 任务上 80.56%→91.67%），为规模化扩展操作策略的视觉-语言覆盖面提供了一条比采集新演示更便宜的路径。

**局限与开放问题**：（1）作者自陈三阶段流水线比直接 VLA 微调更计算密集，每个阶段都要在 VLM 骨干上额外训练；（2）真机验证仅限单一 WidowX 机械臂，跨本体（cross-embodiment）迁移是开放问题；（3）ETC 数据的语义/推理标签依赖 Gemini 生成后人工校验，其规模化质量控制、标注偏差如何传导到最终策略未被深入讨论；（4）compositional induction 的适用边界不清楚——flower 任务中 in-domain ETC 完全无法提升 OOD 表现，而 GPT-Image-2 生成图像构造的 ETC 效果明显弱于真实图像人工标注（37.50% vs 50.00%），说明生成式数据的域适配质量仍是瓶颈；论文未系统探讨何种视觉-语言组合可被诱导泛化、何种不能。（5）与通用多模态协同训练（如 A-OKVQA/COCO/LaTeX-OCR 混合）的对比确认了"内容需与具身场景耦合"的重要性，但论文未与其他专门的具身中间监督工作（如基于场景图、3D 表征或世界模型预测的中间目标）做直接效果对比，ETC 相对这些替代路线的相对优势仍待进一步验证。整体上，该工作为 VLA 数据配方设计提供了一个清晰、可复现的分阶段范式，其"用同源轨迹派生 VQA 监督做中间桥梁"的思路对后续具身预训练数据流水线的设计具有参考价值。

## 参考

- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, arXiv:2501.09747, 2025.
- Beyer et al. *PaliGemma: A Versatile 3B VLM for Transfer*, arXiv:2407.07726, 2024.
- Ji et al. *RoboBrain: A Unified Brain Model for Robotic Manipulation*, CVPR 2025.（ShareRobot 数据来源）
- Gao et al. *VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in VLA Models*, NeurIPS 2026.
- Zhang et al. *VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks*, ICCV 2025.
