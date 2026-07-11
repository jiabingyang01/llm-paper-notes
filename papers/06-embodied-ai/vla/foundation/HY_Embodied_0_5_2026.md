# HY-Embodied-0.5：面向真实世界智能体的具身基础模型

> **论文**：*HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents*
>
> **作者**：Xumin Yu、Zuyan Liu、Ziyi Wang、He Zhang 等（核心贡献者）；项目负责人 Yongming Rao；项目主管 Han Hu
>
> **机构**：Tencent Robotics X × HY Vision Team（腾讯混元）
>
> **发布时间**：2026 年 04 月（arXiv 2604.07430）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.07430) | [PDF](https://arxiv.org/pdf/2604.07430)
>
> **分类标签**：`具身基础模型` `Mixture-of-Transformers` `空间推理` `VLA机器人控制` `GRPO强化学习后训练`

---

## 一句话总结

腾讯提出具身 VLM 系列 HY-Embodied-0.5（2B 激活的 MoT-2B 边缘模型 + 32B 激活的 MoE-A32B 复杂推理模型），通过模态自适应 Mixture-of-Transformers 架构、视觉隐式 token、大规模具身/空间预训练数据，以及"RL(GRPO)+RFT 迭代自进化+大模型到小模型 on-policy 蒸馏"的后训练闭环,在 22 个具身/空间/感知基准上 MoT-2B 以 58.0 的平均分在 16/22 上超过同量级模型（超 Qwen3-VL-4B 10.2 点、RoboBrain2.5-4B 8.6 点）,MoE-A32B 以 67.0 平均分超过 Gemini 3.0 Pro（63.6）;在此基础之上训练的 VLA 模型在真机三任务上媲美或超过 π0/π0.5 基线,其中 Mug Hanging 任务成功率 75% 明显领先 π0（45%）和 π0.5（50%）。

## 一、问题与动机

论文认为通用 VLM 若要支撑真实世界具身智能体,存在两大缺口：(1)**细粒度视觉感知不足**——现有 VLM 在物体级/点级定位、深度、分割等精细感知上表现欠佳,难以支撑物理层面的落地决策;(2)**面向预测-交互-规划的具身推理能力不足**——主流 VLM 以静态网络数据训练为主,缺乏面向动作导向的动态预测、交互与规划能力。作者的目标是构建一个既保留通用 VLM 的开放世界知识、又针对具身场景系统性增强空间/时间感知与推理的基础模型家族,并进一步验证其能否作为下游 VLA（Vision-Language-Action）模型的良好初始化。

## 二、核心方法

**整体架构**：在腾讯混元 Hunyuan-1.8B LLM 基础上,搭配自研原生分辨率 ViT（HY-ViT 2.0,400M 参数,由更大的内部 ViT 蒸馏而来,并训练出码本大小 2k、将每 8×8 patch 压缩为单个离散码的视觉离散表示用于监督）,构成两个变体：**MoT-2B**（2B 激活/4B 总参数,Mixture-of-Transformers,面向边缘部署）和 **MoE-A32B**（32B 激活/407B 总参数,Mixture-of-Experts,面向复杂推理）。

三项关键设计（重点体现在 MoT-2B 上）：

1. **模态自适应 Mixture-of-Transformers**：在多模态训练开始前,复制语言模型的 FFN 与 QKV 参数,分别构成 Vision-MoT 与 Language-MoT 两套非共享参数（初始权重均来自预训练 LLM）,视觉分支采用双向"局部全注意力",语言分支保持因果注意力。这样既扩容了模型容量以承载视觉建模,又避免大量视觉训练对语言能力造成的退化。此外引入"视觉下一码预测"任务,用教师 ViT 生成的离散码作为监督信号，视觉损失为交叉熵：

$$\mathcal{L}_{\text{vision}} = -\frac{1}{N_v}\sum_{i=1}^{N_v}\log p_i(z_i)$$

大白话：让视觉分支像语言模型预测下一个词一样去预测"下一个视觉离散码",逼迫视觉表征学到更细粒度、可预测的结构。

2. **视觉隐式 token（latent token）**：在每个视觉元素（图像或视频帧）末尾追加一个可学习隐式 token,并用大 ViT 的全局 CLS 特征作为监督（负余弦相似度）：

$$\mathcal{L}_{\text{global}} = -\frac{f_{\text{latent}}^\top f_{\text{teacher}}}{\|f_{\text{latent}}\|\,\|f_{\text{teacher}}\|}$$

大白话：给每张图/每帧配一个"摘要 token",强迫它对齐大模型看到的整体语义,充当视觉和语言之间的信息枢纽（论文用注意力可视化验证该 token 确实同时关注图像关键区域和对应的语言语义实体）。预训练阶段总损失为 $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{llm}}+\mathcal{L}_{\text{vision}}+\mathcal{L}_{\text{global}}$,中训练及后续微调阶段只保留 $\mathcal{L}_{\text{llm}}$。

**数据与训练流程**：预训练语料 600B+ token（389B 通用理解 + 236B 具身/感知,其中空间与机器人数据占后者 43%）;具身数据涵盖 Omni-Detection（62M）、深度估计（36M）、分割（5M）、指点与计数（11M）等基础感知,以及 grounding/affordance/trajectory/understanding/planning/reasoning 六类具身中心数据（来自 Molmo、RoboPoint、RoboAfford、ShareRobot、MolmoAct、RoboVQA 等开源集 + 用 cotracker3 从大规模操作视频中提取轨迹）;空间数据覆盖对应关系、几何、构型、度量、动力学五类（源自 ScanNet/ScanNet++/ARKitScenes）。中训练阶段以约 2500 万条高质量样本、按通用:具身:空间≈12:5:3 混合,MoT-2B 采用长/短两种推理链（`\think`/`\no_think`,follow Qwen3-VL）。

**后训练闭环**：(1) SFT 冷启动约 10 万条人机协作构建并经 LLM 校验的 CoT 数据；(2) 强化学习采用 GRPO 目标,针对具身任务输出高度异构（几何 grounding、连续回归、离散判断、轨迹、开放式推理）的特点设计四类任务感知奖励（grounding 用 IoU/点距离,回归用相对误差,轨迹用 DTW+Fréchet 距离,文本类用精确匹配或 LLM-as-judge），组内相对优势 $A_i=(r_i-\mu(\mathbf r))/\sigma(\mathbf r)$、非对称裁剪 $[0.8,1.35]$；(3) **迭代自进化**：RL 与拒绝采样微调（RFT）交替进行,每轮用最新模型多次采样、只保留"部分成功"（既非全对也非全错）的样本作为下一轮训练数据,并用更强教师模型对推理轨迹质量打分筛选（约 100 万候选筛至 30 万条）；(4) **大到小 on-policy 蒸馏（OPD）**：学生模型先用自身策略采样出响应 $y\sim\pi_s(\cdot\mid x)$,教师再对学生生成的前缀做 teacher forcing 打分,最小化逐 token KL：

$$\mathcal{L}_{\text{OPD}} = \mathbb{E}_{x,y\sim\pi_s(\cdot\mid x)}\Big[\tfrac{1}{|y|}\sum_{t=1}^{|y|}\mathrm{KL}\big(\pi_t(\cdot\mid x,y_{<t})\,\|\,\pi_s(\cdot\mid x,y_{<t})\big)\Big]$$

大白话：不是简单模仿教师生成的答案（离线蒸馏）,而是让教师在学生自己实际会走到的每一步状态上纠偏,从而把 MoE-A32B 大模型在 RL+RFT 中学到的推理能力迁移进 MoT-2B 边缘模型。

**下游 VLA**：在 MoT-2B 基座之上,follow π0/π0.5 结构搭建 Action Expert,先用 5000 小时 UMI 数据（不绑定具体本体）做动作专家预训练（32 GPU、batch 32/GPU、20 万步）,再用 300-700 条真机演示对三项任务分别做 SFT。

## 三、关键结果

**Table 1（HY-Embodied-0.5 MoT-2B，22 个基准，vs. Qwen3-VL 2B/4B、RoboBrain2.5-4B、MiMo-Embodied 7B）**：

| 能力类别 | 代表基准 | MoT-2B | 同量级最优基线 |
|---|---|---|---|
| 视觉感知 | CV-Bench / DA-2K | 89.2 / 92.3（均第一） | MiMo-Emb 88.8 / Qwen3-VL 76.5 |
| 具身理解 | ERQA / ShareRobot-Aff. | 54.5 / 26.8（均第一） | MiMo-Emb 46.8 / Qwen3-VL 25.5 |
| 空间理解 | VSIBench / Where2Place | 60.5 / 68.0（均第一） | Qwen3-VL 55.2 / 65.0 |
| 综合 | 22 基准平均 | 58.0（16/22 第一,4/22 第二） | 超 Qwen3-VL-4B +10.2、RoboBrain2.5-4B +8.6 |

**Table 2（HY-Embodied-0.5 MoE-A32B vs. Kimi K2.5、Seed 2.0、Qwen3.5-A17B、Gemini 3.0 Pro）**：22 基准平均分 **67.0**,7 项第一（32%）、6 项第二（27%）,超 Gemini 3.0 Pro（63.6）+3.4、Seed 2.0（66.2）+0.8、Qwen3.5-A17B（66.1）+0.9、Kimi K2.5（61.1）+5.9。在通用理解基准（RealWorldQA、Hallusion-Bench、BLINK、DocVQA、OCRBench、TextVQA）上 MoT-2B 与同量级通用 VLM（InternVL3.5-2B、Qwen3-VL-2B-Thinking）表现相当,未因具身专精而牺牲通用能力。

**真机 VLA 结果**（20 trials/任务/模型，Xtrainer 双臂平台）：

| 任务 | HY-Embodied-0.5 | π0 | π0.5 |
|---|---|---|---|
| Precision Plug-in Packing | 85% | 80% | 85% |
| Tableware Stacking | 80% | 60% | 85% |
| Mug Hanging | **75%** | 45% | 50% |

**效率分析**：MoT 架构相较等规模稠密（Dense-2B）基线训练收敛更快、loss 更低,推理侧因解码阶段主导总耗时,MoT 引入的额外开销可忽略,推理速度与稠密模型基本持平。

## 四、评价与展望

**优点**：(1) 系统性地把"细粒度感知预训练数据 + 模态解耦架构 + RL/RFT/蒸馏后训练闭环"整合为一套可复现的具身 VLM 训练配方,并给出 22 个基准的翔实数字,横向对比覆盖开源（Qwen3-VL/3.5、RoboBrain2.5、MiMo-Embodied）和闭源前沿模型（Gemini 3.0 Pro、Kimi K2.5、Seed 2.0）；(2) on-policy 蒸馏（OPD）把大模型 RL 后训练获得的推理能力迁移到边缘可部署的小模型,是对"先做大模型 RL 再往小模型蒸馏"这一实践路线（如 Thinking Machines 的 on-policy distillation）在具身场景的具体落地；(3) 真机实验直接对比 π0/π0.5 而非只做 VLM 基准评测,提供了从"感知基础模型"到"可执行 VLA"的完整闭环证据。

**局限与开放问题**：(1) 真机评测仅 3 个任务、每任务 20 trials,任务集中在抓取/堆叠/悬挂等短程操作,尚未展示长程、多阶段任务或跨本体泛化能力;(2) MoT-2B 在 RoboBench-Planning、RoboSpatial-Home、ShareRobot-Traj.、SAT 等 4 个基准上仍落后于 MiMo-Embodied-7B 等更大模型,说明规模差距在部分长程规划/轨迹任务上尚未被架构与数据设计完全抵消;(3) 报告中 Qwen3.5-VL 因"重复思考模式导致得分偏低"而被替换为 Qwen3-VL 作为主要基线,且部分模型统一取 thinking/non-thinking 两模式中较优结果,而 HY-Embodied 固定报 thinking 模式结果,比较协议偏保守但也说明跨模型公平比较本身存在方法论张力;(4) 视觉隐式 token、视觉下一码预测等设计目前只在 2B 级别验证,MoE-A32B 是否采用同款架构、二者训练配方差异细节披露有限;(5) 5K 小时 UMI 数据预训练到具身 SFT 之间的迁移机制、以及不同本体上的可迁移性,论文未做进一步消融。整体上,该工作代表了当前"通用 VLM 强化到具身/空间专精,再蒸馏至边缘可部署尺寸,最终对接 VLA 动作专家"这一产业界主流技术路线的一个公开范例,其后训练闭环（GRPO + RFT + OPD 交替迭代）设计具有较高的方法论参考价值。

## 参考

- Liang et al., *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models*, TMLR 2024（HY-Embodied 的 MoT 架构基础）
- Shao et al., *DeepSeekMath*（GRPO 组相对策略优化的来源）
- Bai et al., *Qwen3-VL Technical Report*, 2025（主要开源基线）
- Tan et al., *RoboBrain 2.5: Depth in Sight, Time in Mind*, 2026（具身专用 VLM 基线）
- Xiaomi Embodied Intelligence Team, *MiMo-Embodied: X-Embodied Foundation Model*, 2025（具身专用 VLM 基线）
