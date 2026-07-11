# FineVLA：面向可控指令对齐的细粒度视觉-语言-动作策略框架

> **论文**：*FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies*
>
> **作者**：Xintong Hu, Xuhong Huang, Jinyu Zhang, Shuai Bai, Tao Yu et al.
>
> **机构**：XLANG Lab, The University of Hong Kong；Qwen Team, Alibaba Inc.
>
> **发布时间**：2026 年 05 月（arXiv 2605.27284）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.27284) | [PDF](https://arxiv.org/pdf/2605.27284)
>
> **分类标签**：`VLA` `细粒度指令对齐` `可控策略` `机器人视频理解` `数据构建`

---

## 一句话总结

FineVLA 用 DTW 聚类 + 人工核验把 10 个开源机器人数据集的 972,247 条轨迹精炼成 47,159 条带十维细粒度过程级标注的 FineVLA-Data（指令平均词数从 9.3 增至 96.8，密度提升 10.4 倍），并证明用细粒度指令与原始目标级指令按约 1:1 混合训练 VLA 策略，在 RoboTwin 仿真中达到 **86.8%/82.5%**（Easy/Hard，较纯目标级指令 +15.0/+11.1），在真实双臂平台上把可控评分从 49.9 提到 **62.7/100**。

## 一、问题与动机

VLA 策略正从"完成任务"走向"按人类指定方式执行任务"（即论文定义的 steerability：用哪只手臂、接近哪个方向、在哪里接触等）。但现有开源机器人数据集普遍只有粗粒度目标级指令（如"拿起杯子"），同一任务下的执行细节——主动臂选择、接近方向、接触区域、轨迹朝向、失败与恢复等——完全缺失。作者指出三个具体缺口：（i）异构数据集动作/状态表示不统一，且同任务演示高度冗余，缺少可扩展的细粒度标注基础设施；（ii）缺少评估机器人视频"过程级"理解能力的基准和可扩展标注器（现有视频语言模型/密集字幕方法多关注场景外观而非动作细节，现有具身基准也不系统评测过程级操作细节）；（iii）即便有细粒度数据，细粒度监督是否真的能提升策略学习、以及细粒度与目标级指令应如何配比训练，社区缺乏系统证据。

## 二、核心方法

**FineVLA-Tool（数据构建）**分四阶段：(1) 数据收集与格式转换——统一 BridgeData-V2、BC-Z、RT-1、Galaxea、RoboMIND-V1/V2、RoboCOIN、RH20T、RDT-1B、DROID 共 10 个数据集的 972,247 条轨迹（85,739 个任务）为 LeRobot 2.1 格式；(2) 动作-状态规范化与清洗——将不同数据集的时间参照（绝对/增量/相对首帧）和运动学表示（关节空间 vs 末端执行器空间的多种旋转编码）统一为绝对坐标 + 规范化四元数，并用动作-状态 DTW 一致性距离过滤损坏或控制约定不一致的轨迹；(3) 基于动作的聚类与代表性采样——在每个任务内对规范化动作序列计算成对 DTW 距离矩阵

$$D_{\text{DTW}}(i,j) = c(\mathbf{x}_i,\mathbf{y}_j) + \min\{D_{\text{DTW}}(i-1,j-1),\, D_{\text{DTW}}(i-1,j),\, D_{\text{DTW}}(i,j-1)\}$$

（用大白话说：把两条长度可能不同的动作轨迹按最优对齐方式逐帧比对，帧代价函数在关节空间用位置+夹爪状态的加权距离，在末端执行器空间再加入四元数测地距离项），再做层次聚类（自动按谱隙确定簇数）并从每簇按簇中心邻近度+轨迹质量挑 2–3 条代表轨迹，把 972,247 条原始轨迹压缩为 47,159 条代表性轨迹；(4) 标注流水线——先用 Qwen3.5-Plus 生成十维细粒度描述初稿（action sequence、active actor、target object、initial/final configuration、contact & approach、trajectory & orientation、object interaction、failure & recovery、body motion），再由人工核对时间顺序、物体/主动臂身份、接触区域、运动方向、状态转换、是否幻觉等六项一致性，得到人工核验后的 FineVLA-Data。

**RoboFine-Bench（评测基准）**：500 段留出视频（覆盖 32 种本体、多视角、多任务），拆解为 11,631 条人工核验原子事实，支持两条赛道——VQA 赛道（1,030 道题，按 Entity/Scene Grounding、Action/Motion Understanding、Interaction/State Reasoning 三轴分布在十个细粒度维度上）和 Caption 赛道（模型生成有序步骤级描述，由 LLM 判官对齐原子事实产出 Consistency/Coverage/Anti-Hallucination 分数；easy 设定提供原始任务指令，hard 设定仅给视觉观测）。所有 benchmark 轨迹与 SFT 训练集、策略训练数据严格互斥。

**RoboFine-VLM（可扩展标注器）**：在 FineVLA-Data 上对 Qwen3.5-397B-A17B（MoE 视觉语言模型）做全参微调，得到专用于机器人动作理解的标注模型，用于未来轨迹的规模化标注（论文强调：本文所有策略实验均使用 FineVLA-Tool 产出的人工核验标注，RoboFine-VLM 不参与策略训练监督）。

**FineVLA-Policy（可控策略训练）**：不提出新架构，而是在共享 Qwen3.5-4B 视觉语言主干上接两种现成动作解码框架——StarVLA-OFT（MLP 回归头并行预测动作块，沿用 OpenVLA-OFT 思路）与 StarVLA-GR00T（DiT flow-matching 双系统架构，对齐 GR00T N1.5）——以验证结论与解码架构无关。训练构造两套并行数据集：FG 数据集（每条代表轨迹配细粒度过程级指令）与 Raw 数据集（同源全部轨迹配原始目标级指令），二者动作标签与视觉观测完全相同、仅配对语言不同；训练时按 FG:Raw 比例（Raw-only、1:4、1:2、1:1、2:1、4:1、FG-only）控制每步采样来自哪个数据集的概率，从而隔离"细粒度语言监督"这一单一变量。

## 三、关键结果

**RoboFine-Bench（VQA / Caption，%）**：

| 模型 | VQA Overall | Caption Easy Overall | Caption Hard Overall |
|---|---|---|---|
| Qwen3-VL-Plus | 55.7 | 75.4 | 64.4 |
| Gemini-3.1-Pro | 59.6 | 80.1 | 75.9 |
| GPT-5.4 | 60.2 | 81.4 | 78.0 |
| **RoboFine-VLM（本文）** | **68.2** | **83.2** | **82.2** |

RoboFine-VLM 在 VQA 上比最强通用基线 GPT-5.4 高 8.0 个百分点，最大增益出现在 Action/Motion Understanding 维度（75.7% vs 64.6%）；在更难的 hard 字幕设定（仅靠视觉推断过程）上四项指标全部第一，且 10 名人工评分者的排名与自动分数高度相关（easy 设定 Pearson 0.937 / Spearman ρ 0.943）。

**RoboTwin 仿真成功率（%，20 episodes/任务）**：

| 配置 | RDT-OFT Easy/Hard | RDT-GR00T Easy/Hard | AlohaMix-OFT Easy/Hard |
|---|---|---|---|
| Raw-only | 61.5 / 60.0 | 55.1 / 53.4 | 71.8 / 71.4 |
| FG:Raw = 1:1 | 73.9 / 72.4 | **69.4 / 68.2** | **86.8 / 82.5** |
| FG:Raw 最优点 | 74.1/72.1（1:2） | 69.4/68.2（1:1） | 86.8/82.5（1:1） |
| FG-only | 62.9 / 62.0 | 62.1 / 61.5 | 78.3 / 76.1 |

三组（数据集、架构）组合下 FG-only 均优于 Raw-only（+1.4/+2.0 到 +7.0/+8.1），且随 FG 比例从 0% 升到 100%，成功率呈一致的倒 U 型，峰值稳定落在 FG:Raw = 1:2 到 1:1 之间；最佳设置 AlohaMix-OFT 在 1:1 时达 86.8%/82.5%，较 Raw-only 提升 +15.0/+11.1。

**真实世界 Cobot Magic 双臂评测（100 分制，Table 5）**：

| 监督方式 | Avg (ID) | Avg (All) | Pose | Color | Approach | Rotate | Arm | OOD (L→R) |
|---|---|---|---|---|---|---|---|---|
| Raw-only | 49.9 | 43.6 | 24 | 22 | 60 | 76 | 60 | 0 |
| FG:Raw = 1:1（最优） | **62.7** | **56.1** | **47** | **40** | **78** | **86** | 64 | 10 |
| FG-only | 54.4 | 47.6 | 41 | 25 | 70 | 80 | 60 | 0 |

FG:Raw = 1:1 相对 Raw-only 在每个语言敏感因子上均提升：Pose (+23)、Color (+18)、Approach (+18)、Rotate (+10)、Arm (+4)，增益最大的正是目标级语言完全不给指导的因子（姿态、颜色、接近方向）；两项通用任务 Clean Table（72→84）、Stack Block（35→40）混合监督同样不劣于甚至优于 Raw-only；OOD 主动臂-目标绑定探针分数从 0 升到 10/100，说明部分因子级泛化但组合泛化仍未解决。此外论文还发现：细粒度监督能缩小 OFT 与 GR00T 架构差距（Raw-only 下差距 6.4/6.6，FG-only 下降到 0.8/0.5），且在更大规模的 AlohaMix 上收益比 RDT 更明显（+6.5/+4.7 vs +1.4/+2.0），暗示细粒度监督是可随数据规模继续扩展的监督轴。

## 四、评价与展望

**优点**：(1) 用统一的 DTW 规范化 + 聚类去冗余方案，把跨本体、跨动作表示的十个开源数据集压缩为高信息密度的细粒度语料，是目前较少见的、以动作对齐（而非仅场景描述）为标注核心的大规模数据整理工作；(2) 通过固定动作与视觉、仅切换配对语言这一控制变量设计，在两种解码架构（OFT/GR00T）、三种（数据集、架构）组合、仿真与真实世界双重环境下反复复现"细粒度：目标级 ≈ 1:1 至 1:2 最优"的倒 U 型规律，证据链完整，比单一实验设置的结论更有说服力；(3) 真实世界的分因子控制评测（同一视觉场景仅切换一个语言控制变量）直接量化了"可控性"而非仅任务成功率，弥补了 RoboTwin 等仿真基准无法测量语言遵循度的缺口。

**局限与开放问题**：(1) OOD 主动臂-目标绑定探针显示，细粒度监督虽提升单因子 grounding（尤其在更高 FG 比例下），却不能解决新组合的组合泛化问题，这与 STEER、PartInstruct 等细粒度/部件级指令跟随工作面临的挑战本质相同；(2) RoboFine-VLM 只是降低了标注成本，并未消除人工核验环节，其标注质量在何种规模下可完全替代人工仍未知；(3) 真实世界验证局限于单一 Cobot Magic 桌面双臂平台和 12 项任务，跨本体（如移动操作、单臂）的可控性收益能否复现有待验证；(4) 论文自身也指出，细粒度指令跟随在物理部署中会引入可行性与安全性问题（如强制执行有风险的接触/旋转指令），尚未与安全检查机制结合；(5) 与 Galaxea、RoboCOIN、RoboInter 等提供子任务/层级标注的工作相比，FineVLA 的十维 schema 更聚焦"过程级、动作对齐"而非阶段/部件划分，但该 schema 是否是描述执行约束的最优或最小充分维度集合，论文未做消融或与其他 schema 的正面比较。总体看，FineVLA 更像一个扎实的数据/评测基础设施 + 系统性配比实验论文，而非架构创新；其"倒 U 型混合比例"和"细粒度监督随数据规模收益增大"两条经验规律对后续细粒度 VLA 数据构建具有较强的可操作参考价值。

## 参考

- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024（arXiv:2406.09246）——细粒度策略所依赖的通用 VLA/OFT 解码基线。
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024（arXiv:2410.24164）——GR00T 式 flow-matching 动作解码的代表工作。
- Mu et al. *RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins*, 2024（arXiv:2409.02920）——本文仿真评测所用基准。
- Smith et al. *STEER: Flexible Robotic Manipulation via Dense Language Grounding*, 2024（arXiv:2411.03409）——同属细粒度/密集语言监督路线的相关工作。
- Yin et al. *PartInstruct: Part-Level Instruction Following for Fine-Grained Robot Manipulation*, 2025（arXiv:2505.21652）——部件级指令跟随，与本文过程级 schema 形成对照。
