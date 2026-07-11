# RoboFlamingo：视觉语言基础模型作为高效机器人模仿者

> **论文**：*Vision-Language Foundation Models as Effective Robot Imitators*
>
> **作者**：Xinghang Li, Minhuan Liu（共同一作）, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, Hang Li, Tao Kong（通讯作者）
>
> **机构**：ByteDance Research、清华大学、上海交通大学、新加坡国立大学
>
> **发布时间**：2023 年 11 月（arXiv 2311.01378）
>
> **发表状态**：未录用（预印本，PDF 全文标注为 Preprint，未标注接收会议信息）
>
> 🔗 [arXiv](https://arxiv.org/abs/2311.01378) | [PDF](https://arxiv.org/pdf/2311.01378)
>
> **分类标签**：`VLA` `视觉语言基础模型` `模仿学习` `CALVIN` `OpenFlamingo` `冻结骨干微调`

---

## 一句话总结

RoboFlamingo 把开源视觉语言模型 OpenFlamingo 拆分为"逐帧感知"（冻结绝大部分参数，只微调 Perceiver Resampler 与门控交叉注意力层）与"显式历史决策"（额外接一个 LSTM 策略头）两部分，仅用 CALVIN 中 1% 的语言标注演示数据、共 1B 可训练参数微调，在 CALVIN ABCD→D 上把 Avg Len（连续完成任务数期望）做到 4.09，显著超过此前 SOTA HULC（3.06）与 RT-1（2.45）；而把同一 backbone 整体微调（3B 可训练参数）反而使 Avg Len 崩溃到 0.50，验证了"冻结大部分 VLM + 小策略头解耦微调"这一范式的有效性。

## 一、问题与动机

语言条件机器人操作需要模型同时具备视觉理解、语言理解与决策能力，此前将预训练视觉语言模型（VLM）/大语言模型（LLM）接入机器人控制大致分三类：(1) 从零训练策略（如 HULC、MCIL），未充分利用 VLM 的视觉语言对齐能力；(2) LLM 高层规划（如 SayCan），依赖预训练底层技能策略执行，LLM 本身不理解具体视觉观测；(3) 端到端 co-finetune（如 PaLM-E、RT-2），把 VLM 整体在海量视觉语言数据 + 低层机器人动作数据上联合微调，虽有效但依赖私有模型和数十亿级参数、海量算力，普通研究者难以复现。

论文提出的核心问题：能否用一个开源、低成本的方案，把预训练 VLM 直接适配为机器人操作策略，同时保留其视觉语言对齐能力，又不需要海量算力？作者指出直接套用需解决三个挑战：VLM 输入是静态图文对，机器人需要处理视频（时序观测）；VLM 输出是文本 token，机器人需要输出连续控制信号；下游机器人演示数据量远小于文本-图像预训练数据规模，如何避免小数据微调破坏原有 VL 对齐能力。

## 二、核心方法

**总体框架。** 策略 $\pi_\theta(a \mid o, l)$ 由 Flamingo 骨干 $f_\theta$（逐步产生视觉语言联合表示 $X_t = f_\theta(o_t, l)$）与显式策略头 $p_\theta$（结合历史隐状态 $h_{t-1}$ 预测动作 $a_t = p_\theta(X_t, h_{t-1})$）两部分组成，训练目标沿用标准目标条件模仿学习：

$$\ell = \mathbb{E}_{(\tau,l)\sim\mathcal{D}}\left[\sum_{t=0}^{|\tau|} \log \pi_\theta(a_t \mid o_t, l)\right].$$

**视觉编码。** 第三人称相机图像 $I_t$ 与手爪相机图像 $G_t$ 经 ViT 编码为视觉 token 序列，再由 Perceiver Resampler 用一组可学习 query $Q_R$ 通过注意力将 token 数量从 $N$ 压缩到 $N_r$：

$$K_R = \hat{X}_t^v W_K^R,\quad V_R = \hat{X}_t^v W_V^R,\quad X_t^v = \mathrm{softmax}\!\left(\frac{Q_R K_R^T}{\sqrt{d}}\right)V_R.$$

用大白话说：这一步就是把两路相机的大量图像 patch 特征"摘要"成固定数量的几十个 token，降低后续融合的计算量。

**特征融合解码器。** 语言指令 token 通过门控交叉注意力层与视觉 token 融合，Transformer 自注意力层直接复用预训练 LLM（LLaMA/GPT-NeoX/MPT）权重并全程冻结，只有交叉注意力层和门控参数 $\alpha$ 参与微调：

$$\hat{X}_t^l = \mathrm{Tanh}(\alpha)\cdot \mathrm{MLP}\big(A(X_t^l W_Q^C, X_t^v W_K^C, X_t^v W_V^C)\big) + X_t^l,$$
$$X_t^{l+1} = \mathrm{MLP}\big(A(\hat{X}_t^l W_Q^S, \hat{X}_t^l W_K^S, \hat{X}_t^l W_V^S)\big) + \hat{X}_t^l.$$

用大白话说：门控 $\tanh(\alpha)$ 初始为 0，训练开始时新加入的视觉信息几乎不干扰原始语言模型的表示，随训练推进逐渐"放开"视觉信息注入，从而保护预训练语言/视觉语言对齐能力不被小规模机器人数据破坏。

**策略头（历史建模）。** 作者比较了 4 种策略头形式（MLP w/o hist、MLP w hist、GPT、LSTM），默认取实现最简单且效果与 GPT 相当的 LSTM：对每步联合表示做最大池化后送入 LSTM 维护隐状态，再用 MLP 分别输出末端位姿与手爪开合：

$$\tilde{X}_t = \mathrm{MaxPooling}(X_t);\quad h_t = \mathrm{LSTM}(\tilde{X}_t, h_{t-1});\quad a_t^{pose}, a_t^{gripper} = \mathrm{MLP}(h_t).$$

**训练损失。** 位姿用 MSE 回归，手爪开合状态用二元交叉熵分类（$\lambda_{gripper}$ 为权重）：

$$\ell = \sum_t \mathrm{MSE}(a_t^{pose}, \hat{a}_t^{pose}) + \lambda_{gripper}\,\mathrm{BCE}(a_t^{gripper}, \hat{a}_t^{gripper}).$$

**部分冻结微调策略。** 整个训练过程只更新 Resampler、每层的门控交叉注意力模块与策略头，其余（ViT、LLM 自注意力层）全程冻结，最终可训练参数约 1B（相对总参数 3B\textasciitilde9B），使得单台 8×A100 服务器即可完成训练（MPT-3B 每 epoch 约 13 小时，第 3 epoch 最优；MPT-9B 每 epoch 约 26 小时，第 4 epoch 最优）。这种"解耦感知与决策 + 部分冻结"的设计也带来一个额外收益：可以只在感知端做一次推理、由策略头一次性预测多步动作实现开环（open-loop）控制，从而降低测试时的计算与延迟需求。

## 三、关键结果

实验统一基于 CALVIN 基准（4 个环境 split A/B/C/D，34 类任务，1000 条最长 5 步的语言指令链，仅约 1% 数据带语言标注，约 2.4 万条轨迹）。主力模型为 backbone M-3B-IFT（MPT-1B 语言模型，经指令微调）。

**主对比（ABCD→D，全部四环境训练、测试 D）：**

| 方法 | 训练数据 | 1 | 2 | 3 | 4 | 5 | Avg Len |
|---|---|---|---|---|---|---|---|
| MCIL | ABCD(Full) | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 |
| HULC | ABCD(Full) | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06 |
| RT-1（复现） | ABCD(Lang) | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| **RoboFlamingo** | ABCD(Lang) | **0.964** | **0.896** | **0.824** | **0.740** | **0.66** | **4.09** |

**零样本视觉泛化（ABC→D，训练不含 D 环境）：** RoboFlamingo 达到 Avg Len 2.48，同样大幅领先 RT-1（0.90）与 HULC（在 Full 数据下仅 0.67）。**语言泛化（GPT-4 改写的 enriched 指令）：** 冻结融合解码器词嵌入层的变体（freeze-emb）取得 Avg Len 2.12，优于原版（1.85）与 RT-1（约 0.86），说明直接以词 token 作为输入训练对同义句更敏感，冻结嵌入层可缓解该问题。

**全模型微调 vs. 部分冻结微调（消融，关键结果）：** 把同一 backbone（MPT-3B-IFT）全部 3B 参数解冻微调，Avg Len 从 4.09 崩溃到 0.50（任务 1 成功率从 0.964 跌到 0.415，任务 5 从 0.66 跌到 0.001），证明小规模机器人数据下全量微调会严重破坏 VLM 已有能力，是本文最核心的设计验证。

**VL 预训练必要性消融：** 去掉 OpenFlamingo 预训练权重（随机初始化交叉注意力+resampler）或冻结整个 VLM 只训策略头，性能均大幅下降，表明视觉语言预训练本身、而非仅架构，是性能来源。

**策略头消融：** 忽略历史的 MLP w/o hist 最差；引入历史的 MLP w hist 有提升但仍明显弱于 GPT 与 LSTM，二者表现接近，作者出于简洁性选 LSTM。

**模型规模 × 数据量：** 用完整数据时，更小的 backbone（M-3B 系）已能与更大模型（L-9B、M-9B）竞争；但仅用 10% 语言标注数据训练时，模型规模对性能影响显著放大（如 M-9B 的 Avg Len 0.83 明显高于 M-3B-IFT 的 0.13），说明数据受限场景下更大 VLM 更"数据高效"。指令微调（IFT）版本（M-3B-IFT、G-4B-IFT）相对未 IFT 版本在下游任务上均有提升。

**与机器人专用表征模型对比：** 相同微调协议下，RoboFlamingo（ABCD→D，Avg Len 4.09）明显优于微调后的 Voltron（2.08）与冻结/微调的 R3M（Avg Len ≤0.11），说明大规模通用 VLM 预训练相对专门化的机器人视觉表征预训练在此设置下更具优势。

## 四、评价与展望

**优点。** 方法论上最有价值的贡献是清晰揭示了一个此前较少被系统验证的现象：在演示数据规模远小于原始 VL 预训训练数据规模时，全量微调大模型会灾难性破坏其已有能力，而"冻结绝大部分参数 + 小规模适配层/策略头"的部分微调策略不仅避免遗忘，反而是性能更优的路径——Table 8 的对比（4.09 vs. 0.50）是全文说服力最强的单个实验。此外论文对策略头形式（MLP/GPT/LSTM）、VL 预训练必要性、模型规模与数据量的交互，以及机器人专用表征（R3M/Voltron）与通用 VLM 的对比都做了较系统的消融，工程可复现性强（单机 8×A100 即可训练，代码与模型公开）。

**局限。** 论文本身也承认：(1) 全部实验限于 CALVIN 仿真基准，未在真实机器人上验证，作者在结论中明确将此列为未来工作；(2) 与彼时的 RT-2（更强的私有 VLA 基线）未做直接对比，理由是代码、数据、权重均不可获取，因此"State-of-the-art"的结论严格意义上只相对公开可复现基线成立；(3) 后续任务成功率随序列步数下降的趋势仍然存在（Avg Len 4.09/5，非满分），且论文承认 RoboFlamingo 在语言 token 直接训练下对同义句表达比使用冻结句子编码器的 HULC 更敏感，需要额外冻结嵌入层缓解；(4) 策略头本身容量较小（LSTM+MLP），历史建模能力有限，是否是长时程任务性能上限的瓶颈未被充分探讨。

**与其他工作的关系。** RoboFlamingo 与同期 RT-2、PaLM-E 代表了将 VLM/LLM 接入机器人控制的两条路线：后者依赖私有大规模数据与端到端 co-finetune 把动作离散为语言 token 的一部分，训练与部署成本高、闭源；RoboFlamingo 则选择"解耦感知与策略、小规模适配层微调"的低成本路线，更接近同期 HULC、MCIL 等开源模仿学习工作在数据体量上的可比性，但通过引入预训练 VLM 的视觉语言对齐先验取得了更好的样本效率与泛化性。该"冻结骨干+轻量适配层/动作头"的范式在其后大量 VLA 工作（如基于扩散/flow 动作头接在冻结或部分冻结 VLM 骨干之后的后续方法）中被广泛沿用，可视为该方向的早期系统性验证之一。开放问题包括：该结论在更大规模、更多样化的真实机器人数据上是否依然成立；冻结比例与可训练参数量之间是否存在更优的平衡点；以及历史建模模块（LSTM）是否会在更长时程、更高频控制任务中成为新的瓶颈。

## 参考

- Awadalla et al. *OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models*, arXiv:2308.01390, 2023.
- Alayrac et al. *Flamingo: A Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
- Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, arXiv:2307.15818, 2023.
- Mees et al. *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks*, RA-L, 2022.
- Driess et al. *PaLM-E: An Embodied Multimodal Language Model*, arXiv:2303.03378, 2023.
