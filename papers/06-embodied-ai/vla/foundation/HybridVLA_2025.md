# HybridVLA：统一视觉-语言-动作模型中扩散与自回归的协同生成

> **论文**：*HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model*
>
> **作者**：Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Shanghang Zhang et al.
>
> **机构**：Peking University（北京大学多媒体信息处理国家重点实验室）、Beijing Academy of Artificial Intelligence（北京智源人工智能研究院，BAAI）、CUHK（香港中文大学）
>
> **发布时间**：2025 年 03 月（arXiv 2503.10631，最新版 v3 于 2025 年 06 月更新）
>
> **发表状态**：未录用（预印本，作者在正文首页标注 "Preprint. Under review."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.10631) | [PDF](https://arxiv.org/pdf/2503.10631)
>
> **分类标签**：`VLA` `扩散自回归混合` `统一LLM动作生成` `协同动作集成` `双臂操作` `RLBench`

---

## 一句话总结

HybridVLA 不再像 π0/CogACT 那样在 VLM 之后外挂一个独立的扩散动作头，而是把扩散去噪过程直接注入同一个 LLM 的下一 token 预测流内（扩散 token 置于自回归 token 之前，用 \<BOD\>/\<EOD\> 标记边界，二者共享梯度联合训练），推理时再用自回归 token 的置信度门控两路预测的集成；在 RLBench 10 任务模拟（平均成功率 74% vs. CogACT 60%/OpenVLA 41%）、真实 Franka 单臂（83% vs. CogACT 61%）与 AgileX 双臂（71% vs. π0 55%）上均取得当时最优表现。

## 一、问题与动机

现有 VLA 大体分两条路线，各有明显短板：一是以 RT-2、OpenVLA、ManipLLM 为代表的自回归（ARM-based）方法，把连续动作离散化为若干 bin 并塞进 LLM 词表做 next-token 预测，量化本身破坏了动作的连续性，不利于精细控制；二是以 π0、CogACT、TinyVLA、DiVLA 为代表的扩散式方法，在 VLM 之后加一个独立的扩散头，仅以 VLM 单次前向抽取的特征作为条件去噪生成动作——扩散头本身脱离了 LLM 的预训练推理机制（next-token prediction），未能真正复用 VLM 在互联网规模数据上学到的语义推理能力。作者由此提出核心问题："如何优雅地构建一个真正融合自回归与扩散两种策略优势的统一 VLA，而不是简单地把二者拼接在一起？"——这正是 HybridVLA 要回答的问题，区别于同期把扩散头当作独立"慢系统"模块的双系统（dual-system）设计思路（如 GR00T N1）。

## 二、核心方法

**骨干与状态编码**：HybridVLA 提供 7B 与 2.7B 两种规模，均基于 Prismatic VLMs 初始化。7B 版用 LLAMA-2 作 LLM，DINOv2+SigLIP 双视觉编码器；2.7B 版用 Phi-2 作 LLM，仅 CLIP 视觉编码器。机器人状态 $r_t$ 不再像 ManipLLM 那样离散化后并入语言 query，而是用一个可学习 MLP 直接映射进 LLM 词嵌入空间 $f_r\in\mathbb R^{B\times1\times4096}$——因为扩散动作 token 的去噪以所有前置 token 为条件，离散化的机器人状态会扰乱连续动作预测的条件质量。

**token 序列设计（决定训练稳定性的关键选择）**：论文在 4 种 token 排布方式间做了消融（Table 1，均在 RLBench 10 任务上分别评估纯扩散预测 Dif 和纯自回归预测 AR 的成功率）：Type1（本文采用，扩散 token 置于自回归 token 之前，用 \<BOD\>/\<EOD\> 包裹）Dif 0.66/AR 0.62；Type2（扩散 token 直接预测被 mask 的离散 token，无明确边界）Dif 0.56/AR 0.54；Type3（离散化机器人状态并入语言 query，即 ManipLLM 式）Dif 0.61/AR 0.59；Type4（自回归 token 置于扩散 token 之前）Dif 0.57/AR 0.60。选择"扩散在前"的原因是自回归训练采用 teacher forcing，训练时问题和真值答案都可见，若把自回归 token 放在扩散 token 之前，会作为条件把真值信息泄漏给扩散过程（GT leakage）；而扩散作用于噪声输入，天然不受这一泄漏影响。

**混合训练目标**：扩散部分沿用 Diffusion Policy 式的噪声 MSE 损失，自回归部分是标准的离散 token 交叉熵，二者共享同一 LLM 主干、梯度联合反传：

$$L_{dif}=\mathbb E_{a,i,c}\|\epsilon-\epsilon_\pi(a_t^i,i,c)\|^2,\qquad L_{hybrid}=L_{dif}+L_{ce}$$

用大白话说：扩散头学的是"给定条件把加在动作上的噪声猜回来"，自回归头学的是"给定条件把离散动作 token 猜对"，两个任务的损失直接相加，反向传播时一起更新同一套 LLM 参数——这样扩散的连续动作表征和自回归的语义推理能力都被压进同一个模型里，互相强化。为保证机械臂动作稳定，扩散部分未使用 classifier-free guidance。

**结构化两阶段训练**：加载预训练 VLM 权重后，先在 Open X-Embodiment、DROID、ROBOMIND 等 35 个开源数据集（共 76 万条轨迹、3300 万帧）上训练 5 个 epoch 做大规模预训练（8×NVIDIA A800，仅用单视角 2D 观测），再在下游自采集的仿真/真实数据上做任务微调（视任务用单/多视角观测）。

**推理阶段——协同动作集成（Collaborative Action Ensemble）**：扩散与自回归两支并行生成。扩散侧用 DDIM 采样，作者发现把采样步数降到仅 4 步也不掉点；同时在扩散 token 之前引入 KV cache，仅在首步前向传递条件信息和纯噪声，后续步骤只需重复前向时间步与噪声本身，从而降低推理延迟。自回归侧解码器额外输出预测 token 的平均置信度 $c^{ar}_{t+1}$。若 $c^{ar}_{t+1}>\theta$（$\theta=0.96$，经验设定），认为自回归预测足够可靠，将其与扩散预测取平均作为最终动作；否则只信任扩散预测。这一设计源于两个经验观察：（1）扩散预测在需要精细操控的任务（如 Phone on base、Close laptop lid）上更强，自回归预测在需要场景语义推理的任务（如 Water plants、Frame off hanger）上更强；（2）在 80% 以上成功完成的测试样本中，自回归 token 的平均置信度超过 0.96，说明该置信度是可靠的质量信号。此外论文还提供 HybridVLA-dif 变体：权重与完整模型相同（同样用协同训练配方训得），但推理时只走扩散分支，以换取更高吞吐。

## 三、关键结果

**RLBench 仿真（10 个 tabletop 任务，Franka Panda，多任务联合训练；NVIDIA 4090D 上测推理速度）：**

| 方法 | 平均成功率 | 推理速度 |
|---|---|---|
| ManipLLM (7B) | 0.38 | 2.2 Hz |
| OpenVLA (7B) | 0.41 | 6.3 Hz |
| π0 (2.6B) | 0.55 | 13.8 Hz |
| CogACT (7B) | 0.60 | 9.8 Hz |
| HybridVLA-dif (7B) | 0.66 | 9.4 Hz |
| HybridVLA (2.7B) | 0.58 | 12.3 Hz |
| **HybridVLA (7B)** | **0.74** | 6.1 Hz |

**真实机器人（20 rollouts/任务，人工评判成功率）：**

| 平台 | π0 (2.6B) | CogACT (7B) | HybridVLA-dif (7B) | HybridVLA (7B) |
|---|---|---|---|---|
| Franka 单臂（5 任务均值） | 0.45 | 0.61 | 0.80 | **0.83** |
| AgileX 双臂（5 任务均值，14-DOF） | 0.55 | 不支持多视角，未测 | 0.66 | **0.71** |

**消融（Table 3，RLBench 10 任务均值）**：完整模型（AR+Dif+大规模预训练 LSP+机器人状态嵌入 RSE+协同训练配方 CTR+动作集成 CAE）0.74；去掉 LSP 骤降至 0.22，是影响最大的单一因素；去掉 RSE 降至 0.68；对比"单头训练+CTR"（Ex1 纯扩散评估 0.66、Ex3 纯自回归评估 0.62）与"单头训练无 CTR"（Ex2 0.60、Ex4 0.57），即使推理时只用一路输出，协同训练配方本身也能带来约 5\textasciitilde6 个百分点的提升，支持了论文"两种范式互相强化"的核心论点。

**泛化实验（Table 4，对比未见物体/背景/高度/光照 4 类扰动，相对下降幅度）**：单臂场景以 CogACT 为基线、双臂以 π0 为基线，HybridVLA 在几乎所有扰动条件下相对下降幅度更小，例如未见物体扰动：HybridVLA 单臂 -33% vs. CogACT -43%，HybridVLA 双臂 -6% vs. π0 -8%；未见背景：HybridVLA 单臂 -11% vs. CogACT -37%。

## 四、评价与展望

**优点**：HybridVLA 是较早明确把扩散去噪嵌入同一 LLM next-token 流（而非在 VLM 输出特征上外挂独立扩散头）的工作，结构上区别于 π0/CogACT/TinyVLA/DiVLA 这一整个"VLM 特征 + 独立扩散头"家族。消融做得比较扎实：token 排布顺序既有理论论证（GT 泄漏）又有 Table 1 的实证支持；LSP/RSE/CTR/CAE 四个设计模块逐一 on/off 消融，逻辑清晰。推理时的置信度门控集成是训练无关（training-free）的轻量机制，且确有实证收益（0.74 的集成结果优于 0.66 的纯扩散和 0.62 的纯自回归单头结果）。真机验证覆盖单臂（Franka）与双臂（AgileX，14-DOF）两种平台，并系统测试了物体/背景/高度/光照四个泛化轴，覆盖面在同类论文中不算窄。

**局限与开放问题**：（1）完整集成模式在 7B 规模下推理速度仅 6.1Hz，明显慢于 π0（13.8Hz）和 CogACT（9.8Hz），论文自己也承认推理速度"受限于较慢的自回归生成"；HybridVLA-dif 用单一扩散分支换回速度（9.4Hz）但放弃了集成带来的精度增益，说明该方法在延迟与成功率之间仍需用户显式取舍，并非"免费午餐"。（2）动作集成的置信度阈值 $\theta=0.96$ 是在 RLBench 上经验标定的固定超参，论文未验证其在不同机器人本体/任务分布下是否需要重新调参，而这恰恰是决定"何时信任自回归分支"的关键旋钮，存在迁移脆弱性风险。（3）LSP 消融显示去掉大规模预训练（76 万条轨迹）后成功率从 0.74 骤降到 0.22，说明方法对大规模预训练语料的依赖非常重，"两种范式互相强化"的收益与"大规模预训练本身让任何合理架构都变强"这两个因素在此实验设计下不易完全解耦。（4）RLBench 训练数据来自运动规划器生成的轨迹而非人类真实示教，真机每个任务仅 20 次 rollout，量级偏小，细粒度的百分比数字统计噪声不可忽视。（5）论文未与同期的离散扩散 VLA（如 Discrete Diffusion VLA 一类工作）或更晚近的双系统（fast/slow）VLA 架构做直接对比，"扩散嵌入 LLM"这一设计选择相对于其他几种"融合扩散与自回归"路线的优劣仍是开放问题。

## 参考

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164 — 主要扩散/flow-matching VLA 基线。
- Li et al. *CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation*, arXiv:2411.19650 — 独立扩散头式 VLA 的代表基线，也是 HybridVLA-dif 对比的主要对象。
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246 — 主要自回归 VLA 基线。
- Li et al. *ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation*, CVPR 2024 — 离散化机器人状态并入语言 query 的对比设计（Table 1 中的 Type3）来源。
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, IJRR 2023 — 本文扩散动作损失函数设计的直接来源。
- Karamcheti et al. *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models*, arXiv:2402.07865 — HybridVLA 两种规模模型初始化所用的预训练 VLM 基座。
