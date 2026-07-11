# CARE：面向机器人控制的连续隐动作表示多任务预训练

> **论文**：*CARE: Multi-Task Pretraining for Latent Continuous Action Representation in Robot Control*
>
> **作者**：Jiaqi Shi、Xulong Zhang（二人共同一作）、Xiaoyang Qu、Jianzong Wang（通讯作者）
>
> **机构**：Ping An Technology (Shenzhen) Co., Ltd.（中国平安科技，深圳）；University of Science and Technology of China（中国科学技术大学，合肥）
>
> **发布时间**：2026 年 01 月（arXiv 2601.22467）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.22467) | [PDF](https://arxiv.org/pdf/2601.22467)
>
> **分类标签**：`隐动作模型` `VLA预训练` `多任务学习` `LIBERO` `无动作标签预训练`

---

## 一句话总结

CARE 把隐动作模型（LAM）训练直接融入 VLM 预训练，用 VLM 最后隐藏状态中隐动作占位符对应的部分作为连续隐动作表示（替代 VQ-VAE 离散码本或独立的 encoder-decoder LAM），再通过"下一帧特征预测"与"CoTracker 关键点轨迹预测"两个解码器、以不确定性加权损失联合训练；仅用 Open X-Embodiment 140k 机器人轨迹 + Something-Something v2 约 100k 人类视频（共约 240k，无动作标签）预训练，再用 RT-1 数据集 3% 子集做 LoRA 微调动作头，在 LIBERO 四套件上平均成功率达到 77.7%，超过有动作标签预训练的 OpenVLA（75.0%），并大幅超过同为无动作标签的 LAPA（64.3%）与作者复现的 CoMo（69.2%）。

## 一、问题与动机

VLA 模型受限于动作标注的高成本。已有隐动作模型路线——Genie 的离散码本 LAM、LAPA 的 VQ-VAE 隐动作、Moto 用 ViT 编码器预测高层语义嵌入、CoMo 的连续隐动作+交叉注意力——仍存在三个未解决问题：(1) 偏差传播：VQ-VAE 量化误差和码本坍塌被下游 VLA 完全继承且无法纠正；(2) 隐动作缺乏显式动作编码：训练目标只是下一帧重建，推断出的隐动作可能只是压缩了帧间差异，即便这些差异并非控制动作引起；(3) shortcut learning 风险：单一训练目标和独立 LAM 结构容易让模型退化成"未来帧预测器"而非"隐动作建模器"。CARE 针对这三点提出：不再训练一个独立的 encoder-decoder LAM 再对接 VLA，而是把 LAM 训练直接嫁接进 VLM 预训练本身，并引入关键点轨迹预测这一新的辅助任务来强化显式动作编码、抑制 shortcut learning。

## 二、核心方法

### 架构：VLM 即隐动作编码器

模型基于 Prismatic-7B VLM：视觉编码器由 SigLIP 与 DinoV2 两个特征序列 $f_{sig}, f_{dino}\in \mathbb R^{N_p\times D_v}$ 拼接后经两层 MLP 投影得到视觉特征 $f_v$；文本侧经 tokenizer + embedding 得到 prompt embedding $f_T$（prompt 形如 "What action should the robot take to \{Instruction\}?"，并在末尾附加隐动作占位符 token）；两路特征拼接输入 7B 参数的 Llama LLM 主干，取最终隐藏状态 $h_{last}=LM_\theta([f_T;f_v])$ 中对应隐动作占位符的部分作为连续隐动作表示 $z$，直接用于多任务学习。用大白话说：不再单独训练一个"隐动作提取器"再接到 VLA 上，而是让 VLM 在做"看图回答该采取什么动作"这道填空题时，直接把答案写成一段连续向量，这段向量本身就是隐动作。

### 双解码器多任务预训练

两个训练目标共享同一个 $z$：

1. 帧预测任务：与 LAPA/Genie/Moto 类似，用当前帧视觉特征 $f_v^t$ 与 $z$ 预测下一帧视觉特征 $f_v^{t+1}$。通过交叉注意力 $z_f=\mathrm{softmax}(QK^\top/\sqrt d)V$（$Q=zW_z+b_z$，$K,V=f_v^tW_f+b_f$）得到融合表示 $z_f$，再输入 FrameDecoder。
2. 关键点轨迹预测任务（本文新提出）：输入当前帧 256 个均匀分布二维点坐标 $k_t\in\mathbb R^{B\times256\times2}$（真值轨迹 $k_t,k_{t+1}$ 由预训练模型 CoTracker 提取），与 $z$ 同样经交叉注意力得到 $z_k$，输入 PointDecoder 预测下一帧点坐标 $k_{t+1}$。

两个任务损失均为 MSE：$\mathcal L_f^t=\mathrm{MSE}(f_v^{t+1},\hat f_v^{t+1})$、$\mathcal L_p^t=\mathrm{MSE}(k^{t+1},\hat k^{t+1})$。采用 Kendall et al. 的不确定性加权（UWL）合并二者：

$$\mathcal L^t=\frac{1}{\sigma_1^2}\mathcal L_f^t+\frac{1}{\sigma_1^2}\mathcal L_p^t+\log\sigma_1+\log\sigma_2$$

（原文公式两项系数均写作 $1/\sigma_1^2$；结合正文"$\sigma_1,\sigma_2$ 分别自适应学习 $\mathcal L_f$ 与 $\mathcal L_p$ 相对权重"的描述及 Kendall et al. 原始 UWL 定义，第二项应为 $1/\sigma_2^2$，此处疑似原文笔误。）用大白话说：让模型自己学一套"信任度"，如果某个任务（比如点轨迹）本身噪声大、天然难预测，就自动调低它的权重，不用手工调超参数去平衡两个 loss 谁更重要。

### 微调

预训练得到的隐动作 VLM 不能直接控制机器人，因为隐动作并非真实的末端执行器或关节动作。第二阶段在带动作标签的机器人演示数据上，用 LoRA 微调隐动作 VLM，并接一个轻量残差 MLP 动作头（接在解码器输出的隐藏表示之后），以 L1 回归损失拟合真实动作，微调数据为 RT-1 数据集 3% 均匀采样子集。作者称该设计把传统 VLA 训练流程从四阶段（独立 LAM 预训练→VLM 预训练→动作头微调→...）压缩为三阶段。

## 三、关键结果

预训练数据：Open X-Embodiment 机器人轨迹 140k 条 + Something-Something v2 人类日常活动视频约 100k 条，共约 240k。基准为 LIBERO 四套件（Spatial / Object / Goal / Long，各 10 任务、每任务 50 条人类遥操作演示），对照 OpenVLA、Octo、Diffusion Policy、MDT（含腕部相机）四个使用动作标签预训练的基线，以及 LAPA、CoMo（作者用 Prismatic-7B 和相同数据复现，非原论文的扩散策略实现）两个无动作标签基线。

**表：LIBERO 成功率（%）**

| 方法 | 用动作标签预训练 | Goal | Spatial | Object | Long | 平均 |
|---|---|---|---|---|---|---|
| OpenVLA | 是 | 76.5 | 82.6 | 88.2 | 52.8 | 75.0 |
| Octo | 是 | 82.9 | 76.1 | 84.3 | 51.6 | 73.7 |
| Diffusion Policy | 是 | 68.3 | 78.3 | 92.5 | 50.5 | 72.4 |
| MDT（含腕部相机） | 是 | 71.2 | 77.3 | 88.1 | 62.6 | 74.8 |
| LAPA | 否 | 57.1 | 74.3 | 72.2 | 53.6 | 64.3 |
| CoMo（复现） | 否 | 63.2 | 76.0 | 83.1 | 54.6 | 69.2 |
| CARE（仅人类视频） | 否 | 64.3 | 70.2 | 69.5 | 49.2 | 63.3 |
| CARE（仅 Bridge 子集） | 否 | 67.5 | 75.0 | 72.2 | 58.5 | 68.3 |
| CARE（Bridge+RT-1 子集） | 否 | 72.0 | 80.5 | 77.6 | 63.8 | 73.5 |
| **CARE（人类+Bridge+RT-1）** | 否 | **77.9** | **81.2** | **86.4** | **65.3** | **77.7** |

CARE 全量预训练数据配置在四个任务上均超过复现版 CoMo 与 LAPA，平均成功率 77.7% 甚至超过有动作标签预训练的 OpenVLA（Goal 高 1.4 个百分点、Long 高 12.5 个百分点）。数据消融显示随着预训练数据源逐步扩展（仅人类视频→仅 Bridge 子集→Bridge+RT-1 子集→人类+Bridge+RT-1 全量），成功率单调上升，作者以此论证方法遵循 scaling law。（注：正文 3.1 节说明微调阶段固定使用 RT-1 3% 子集，而表 1 消融标签中又出现 "Bridge""RT-1" 字样指代预训练数据来源，论文对两处 "RT-1" 的用法未做明确区分。）

**表：可解释性与 shortcut learning 消融（LP-MSE↓ / S-PCFC↓）**

| 方法 | Goal | Spatial | Object | Long |
|---|---|---|---|---|
| CoMo | 0.839 / 0.899 | 0.881 / **0.892** | **0.662** / 0.902 | 0.754 / 0.910 |
| LAPA | 1.241 / 0.980 | 0.924 / 0.992 | 1.136 / 0.942 | 0.741 / 0.945 |
| CARE | **0.647** / 0.833 | **0.717** / 0.945 | 0.690 / **0.860** | **0.643** / **0.820** |

线性探针 LP-MSE 上 CARE 在 Goal / Spatial / Long 三个套件全面低于 CoMo 和 LAPA（Goal 相对 CoMo 降 22.8%、相对 LAPA 降 47.8%；Spatial 降 18.6%/22.4%；Long 降 14.7%/13.2%），但 Object 套件上 LP-MSE 反而略高于 CoMo（0.690 对 0.662）。S-PCFC（越低说明 shortcut learning 越弱）上 CARE 在 Goal / Object / Long 三项最优，但 Spatial 上 CoMo（0.892）优于 CARE（0.945）。语义标签预测实验显示，仅用"首帧 + 9 个隐动作"作为输入，CARE 隐动作训出的分类器语义准确率达 84.2%，不仅高于 LAPA（64.1%）和 CoMo（71.2%），甚至略高于直接用"首帧 + 9 个真实后续帧"（80.4%），说明隐动作已压缩了足够的任务语义信息。

**多任务消融（Prismatic-7B, LIBERO 平均 SR）**：仅帧预测 67.8%（latent loss 收敛值 0.012）；仅点轨迹预测 54.3%（latent loss 0.009）；两者联合的多任务训练 77.7%（latent loss 0.046）。作者以此论证两个任务存在协同增益（"1+1>2"），且点轨迹单独使用效果最弱，与既有文献中"帧预测优于单独点跟踪"的结论一致。

## 四、评价与展望

**优点**：CARE 的核心贡献是把隐动作抽取从"独立的 encoder-decoder LAM"改造为"VLM 自身推理产物"，省去了单独训练/维护 LAM 再对接 VLA 的额外阶段，在架构上比 LAPA、Moto 这类先训练独立 LAM 再迁移到 VLA 的两阶段路线更紧凑。关键点轨迹预测这一辅助任务提供了一个轻量、可复用现成模型（CoTracker）、无需额外人工标注的信号；语义标签预测实验（84.2% 准确率，甚至超过用真实后续帧的 80.4%）为"隐动作确实编码了任务语义"提供了较有说服力的直接证据，而不只是通过下游 LIBERO 成功率间接论证。

**局限与开放问题**：论文自己在结论中承认，相较于用动作标签预训练的方法仍存在性能差距（例如 Object 套件上 CARE 的 86.4% 仍低于 Diffusion Policy 的 92.5%、OpenVLA 的 88.2%），并计划通过引入动作分块（action chunking）和多维感知视角在未来工作中缩小差距。评测局限于 LIBERO 仿真基准的四个任务套件，未见真实机器人实验，方法的 sim-to-real 有效性尚不清楚。正文中"相较于基于扩散策略的 CoMo 方法在 Goal 和 Object 任务上略逊"的表述与表 1 实际数据存在出入——表 1 中复现版 CoMo（同样基于 Prismatic-7B、非扩散策略实现）在全部四个任务上都被 CARE 超过，这句话更可能是在与 CoMo 原论文中基于 Diffusion Policy 的版本（未在本文表中列出）作比较，行文未做清晰区分。可解释性与 shortcut learning 消融也并非全面占优：Object 套件的 LP-MSE、Spatial 套件的 S-PCFC 上 CoMo 仍优于 CARE，论文没有对这两处例外做进一步分析。此外不确定性加权损失公式中两个系数均写作 $1/\sigma_1^2$，与正文文字描述及所引用 Kendall et al. 原始 UWL 定义不符，应为笔误。

**与相关工作的关系**：CARE 处在 Genie → LAPA（VQ-VAE 离散隐动作）→ Moto（语义嵌入预测）→ CoMo（连续隐动作 + Q-former 式交叉注意力）这一隐动作模型演进链条的下游，其增量贡献主要是：(1) 用 VLM 直接承担隐动作编码器角色而非独立训练 LAM；(2) 新增关键点轨迹预测任务，并用不确定性加权自动平衡多任务损失。相比 CoMo 同样追求连续隐动作表示与抗 shortcut learning，CARE 提供了 VLM-native 抽取路线这一替代方案，二者互为本文对照实验中的直接竞品。开放问题包括：VLM 直接兼任隐动作编码器是否会因为同时承担"语言理解"和"隐动作预测"两种目标，在更大规模、更多样化任务上出现容量竞争；关键点轨迹预测依赖 CoTracker 的跟踪质量，其误差是否会传导进隐动作表示，论文未作讨论；以及该框架能否扩展到双臂、灵巧手等更高自由度的动作空间仍待验证。

## 参考

- Ye et al. *Latent Action Pretraining from Videos (LAPA)*, ICLR, 2025.
- Chen et al. *Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos*, ICCV, 2025.
- Yang et al. *CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning*, arXiv:2505.17006, 2025.
- Bruce et al. *Genie: Generative Interactive Environments*, ICML, 2024.
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL, 2024.
