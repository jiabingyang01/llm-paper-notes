# VQ-VLA：通过规模化向量量化动作分词器改进 VLA 模型

> **论文**：*VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers*
>
> **作者**：Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, Tong He（通讯作者）et al.
>
> **机构**：Shanghai AI Lab；同济大学；中国科学技术大学（USTC）；浙江大学（ZJU）；南京大学（NJU）；上海交通大学（SJTU）
>
> **发布时间**：2025 年 07 月（arXiv 2507.01016）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.01016) | [PDF](https://arxiv.org/pdf/2507.01016)
>
> **分类标签**：`VLA` `动作分词` `VQ-VAE` `离散动作表示` `长程操作` `sim2real`

---

## 一句话总结

用卷积残差 VQ-VAE 替换 OpenVLA 原本的逐维分 bin 动作分词器，在比以往同类工作大 100 倍以上的真实（OpenX-Embodiment）+仿真（LIBERO/ManiSkill/RLBench）轨迹数据上训练该分词器，发现动作轨迹的 sim2real domain gap 极小、可放心用仿真数据做 scaling；LIBERO-90 仿真成功率从 73.53% 提到 80.98%（+7.45pt），真实机器人 6 任务短程平均成功率从 23% 提到 46.25%、长程两任务分别提到 50%和 30%，推理频率从 4.16Hz 提到 11.84Hz（近 3 倍）。

## 一、问题与动机

OpenVLA/RT-2 一类 VLA 沿用逐维、逐时间步的 256-bin 离散化动作表示，粒度粗、每步只预测单个动作，导致推理慢且长程任务误差累积明显。FAST（DCT 变换）、MiniVLA、VQ-BeT/behavior-generation-with-latent-actions 等工作已经在探索动作量化分词，但训练数据规模通常局限于单任务/单数据集，分词器的可扩展性和精度上限未被充分验证。本文提出并验证两个假设：(1) 分词越精确，长程动作建模的收益越明显，因此值得用更大规模、覆盖更多任务的轨迹数据训练分词器；(2) 与视觉、物理模态不同，动作轨迹在仿真与真实环境间的 domain gap 很小，这使得可以低成本地用海量仿真轨迹（远比采集真实数据便宜）来扩大动作分词器的训练数据规模，而分词器训练本身相比扩大整个 VLA 模型计算量极低（单卡 A100 约一周），是一条高性价比的 scaling 路径。

## 二、核心方法

整体分两阶段（图 1）：(1) 训练一个通用的卷积残差 VQ-VAE 动作分词器；(2) 冻结该分词器，用 LoRA 微调 OpenVLA（Prismatic-7B 骨干），让 VLM 直接预测 VQ token 序列而非分 bin token。

**残差 VQ-VAE 架构**。不同于以往 Residual VQ-VAE（如 SoundStream）用简单 MLP 做 encoder/decoder，本文受 pyramidal flow matching 中 VAE 设计启发，改用 2D 时序卷积/反卷积层，以更好地捕捉动作序列的局部关系与层级时序依赖。给定动作片段 $a_{t:t+n}\in\mathbb{R}^{n\times d}$，encoder 得到隐向量 $x=\phi_{enc}(a_{t:t+n})$，再用残差向量量化（RVQ）逐级分解：$q(x)=\sum_{i=1}^{N_q} q_i(r_i)$，其中 $r_1=x,\ r_{i+1}=r_i-q_i(r_i)$，$N_q$ 为量化层数。解码器用 2D 时序反卷积重建动作序列 $\hat a_{t:t+n}=\phi_{dec}(q(x))$。训练损失为

$$
\mathcal{L}=\|a_{t:t+n}-\hat a_{t:t+n}\|_2^2+\lambda\left(\|sg(x)-q(x)\|_2^2+\|x-sg(q(x))\|_2^2\right),
$$

其中 $sg(\cdot)$ 为 stop-gradient，$\lambda=4$。**用大白话说**：第一项让重建出的动作尽量贴近原始动作；后两项是标准 VQ-VAE 的 codebook 损失和 commitment 损失，用 stop-gradient 让"编码器输出"和"最近的 codebook 向量"互相靠拢但不互相干扰梯度。

为增强 encoder 处理结构化动作数据的能力，输入前额外加两类 embedding：正弦 **time embedding**（捕捉高低频时序模式）和可学习的 **action-type embedding**（区分动作向量中 XYZ 位置、Euler 角、gripper 状态等不同维度的语义，给出强先验）。训练策略是**渐进式**的：先在噪声较大的真实数据集 OpenX-Embodiment 上训练，再逐步混入更干净平滑的仿真数据集（LIBERO、ManiSkill、RLBench），使 VQ 模型收敛到更平滑稳定的表示；分词器训练仅以动作序列自身为输入（不接视觉/语言条件），以保持通用性，训练数据规模比以往单任务方案扩大 100 倍以上，单张 A100 约一周训完。

**分词器接入 VLA**。$N_q$ 层残差 VQ 各自的 codebook index 被分配不重叠 ID 区间：第 $i$ 层偏移 $(i-1)\times256$，即 $z_q^i\in[256(i-1),\,256i-1]$（如第一层 $[0,255]$，第二层 $[256,511]$），避免跨层语义混淆。VLM 用标准 next-token 交叉熵直接预测这些 token：

$$
\mathcal{L}_{VLM}=-\sum_{i=1}^{N_q}\log P\!\left(\hat z_q^i=z_q^i \mid o_{t-h:t}\right),
$$

替换词表中最少使用的 token（做法与 OpenVLA 一致，但按层区间保持替换的一致性）。由于一次推理预测的是被压缩过的多步动作 token（而非逐步预测单个动作），显著减少了训练/推理所需 token 数，从而加速推理并降低长程累积误差。

## 三、关键结果

**LIBERO 仿真**（Table 1，架构对比，ALL-LIBERO 训练）：

| 方法 | LIBERO-10 | LIBERO-GOAL |
|---|---|---|
| Original OpenVLA | 51.0% | 75.8% |
| MLP Residual VQ-VAE | 53.4% | 72.6% |
| Conv Residual VQ-VAE | 60.0% | 75.2% |

**域外仿真数据 scaling**（Table 2，评测 LIBERO-90，分词器训练数据完全不含 LIBERO）：

| 模型 | LIBERO-90 成功率 |
|---|---|
| baseline（OpenVLA） | 73.53% |
| VQ_M（仅 ManiSkill 训练） | 14.38% |
| VQ_{M+R}（ManiSkill+RLBench） | **80.98%**（+7.45pt） |

VQ_M 单独用 ManiSkill 训练反而大幅劣于 baseline，说明分词器训练数据不够大/不够多样时会严重损害下游性能，scaling 曲线并非单调平滑。

**真实机器人实验**（Franka Research 3 + RealSense D435，20Hz 控制，SE(3) 绝对末端位姿动作；6 任务，50 demo/任务，20 trial/任务评测）：短程 4 任务（拔纸巾、捡玩具、放玩具入篮、翻正倒扣的锅）平均成功率从 baseline 23% 提升到 VQ_{O+L+M}（OpenX+LIBERO+ManiSkill 共训）的 46.25%，其中"拔纸巾"任务 baseline 仅 5%、VQ 系模型 ≥20%，"翻正锅"任务提升约 30pt。长程 2 任务：将两个杯子依次放入篮子，baseline 15% → VQ_{O+L+M} 50%；开抽屉放玩具再关抽屉，baseline 近乎 0%（20 次中 15 次只完成开抽屉这一步）→ VQ_{O+L+M} 30%，且能 100%完成开抽屉这一步骤。全部任务平均成功率 VQ_{O+L+M} 比 baseline 高 23.25pt。

**Sim2real domain gap 分析**（Table 3，仅用 LIBERO 仿真数据训练的 VQ_L 与真实数据训练的 VQ_O、混合训练的 VQ_{O+L} 在 3 个真实任务上对比）：三者表现相近（如"翻正锅"：baseline 30.0% / VQ_O 45.0% / VQ_L 55.0% / VQ_{O+L} 45.0% / VQ_{O+L+M} 60.0%），支持动作轨迹的 sim2real domain gap 很小这一核心假设。

**推理速度**（Table 4）：VQ-VLA 11.84Hz vs. OpenVLA 4.16Hz，VQ-VAE 压缩比为 5 时推理速度近乎提升 3 倍。

**消融**：(1) Action chunking 方式对比（Table 5，LIBERO-90 / 翻正锅 / 放玩具入篮）：baseline 单步预测 74.76% / 30.0% / 20.0%；把 OpenVLA 改造成自回归输出 5 步 chunk 反而更差，66.53% / 10.0% / 0.0%，且观察到 chunk 内动作值高度雷同的"shortcut learning"现象（直接复制前一步动作）；VQ-based chunking（VQ_{O+L+M}）达到 86.61% / 60.0% / 45.0%，显著优于两者。(2) Embedding 消融（Table 6）：VQ_{O+L} 去掉 time/action-type embedding 时为 85.17% / 40.0% / 35.0%，加上后为 86.16% / 45.0% / 35.0%。

## 四、评价与展望

**优点**：把"扩大整个 VLA 模型规模"这一高成本命题转化为"扩大动作分词器训练数据规模"这一低成本命题（单卡 A100 一周即可），并给出了"动作轨迹 sim2real gap 远小于视觉/物理模态"这一有实践价值的经验结论，为利用仿真器（ManiSkill、RLBench 等可近乎无限生成轨迹）训练可迁移动作表示提供了依据。与 FAST（离散余弦变换分词）、VQ-BeT/latent-actions（小规模单任务 VQ-VAE）等同类工作相比，本文的差异化贡献主要在训练数据规模（>100×）和"冻结后即插即用、零样本迁移到不同下游任务"的定位，而非分词算法本身的新颖性（RVQ 架构沿用 SoundStream/VQ-BeT 一脉，仅将 MLP 换成时序卷积）。消融中"自回归 chunking 出现 shortcut learning、chunk 内动作趋同"这一负面发现，对后续 action chunking 设计也有一定参考价值。

**局限与开放问题**：作者自陈的方向包括扩展到更大规模仿真数据集（如 RLBench）、把推理加速与 VLM 蒸馏/量化结合、在分词器架构中引入动作频率等额外条件。此外，论文中 VQ_M 单独用 ManiSkill 训练时成功率暴跌至 14.38%（远低于 baseline），但论文只归因于"合成数据规模不足"，未深入分析失败机制，scaling 是否单调、在何种数据配比下会失效仍不清楚。真实实验任务数（6 个）和每任务样本量（50 demo / 20 trial）都较小，置信区间较宽，且全部在同一台 Franka、固定第三人称视角下完成，未验证跨本体、跨视角的可迁移性；长程任务的绝对成功率（25%~50%）距离实用仍有明显差距。论文的对比基线仅为 OpenVLA 原生 bin 分词器及自建的自回归 chunking baseline，未与 FAST 等同期动作分词方法做直接的 success rate 横向对比，证据链不算完整。此外，多层残差 VQ 的不重叠 token ID 区间设计会扩大 VLM 的有效动作词表规模，论文未讨论该设计对语言/视觉 token 容量的挤占效应，也未报告 codebook 利用率（是否存在 codebook collapse）。

## 参考

- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246 — 本文的骨干模型与主要基线
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, arXiv:2501.09747 — 同期另一条动作分词路线（离散余弦变换）
- Lee et al. *Behavior Generation with Latent Actions*（VQ-BeT）, arXiv:2403.03181 — Residual VQ-VAE 动作表示的直接前身
- Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, arXiv:2306.03310 — 仿真评测 benchmark
- Mu et al. *ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations*, arXiv:2107.14483 — 分词器训练所用大规模仿真轨迹来源之一
