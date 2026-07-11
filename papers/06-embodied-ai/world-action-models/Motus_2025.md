# Motus：统一潜动作世界模型（Unified Latent Action World Model）

> **论文**：*Motus: A Unified Latent Action World Model*
>
> **作者**：Hongzhe Bi, Hengkai Tan et al.（Bi 与 Tan 为共同一作兼共同项目负责人；Shenghao Xie、Zeyuan Wang、Shuhe Huang、Haitian Liu 亦为共同一作）
>
> **机构**：清华大学（计算机科学与技术系、人工智能研究院、BNRist Center、THBI Lab、Tsinghua-Bosch Joint ML Center）；北京大学；地平线机器人（Horizon Robotics）
>
> **发布时间**：2025 年 12 月（arXiv 2512.13030，v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.13030) | [PDF](https://arxiv.org/pdf/2512.13030)
>
> **分类标签**：`统一世界模型` `潜动作` `Mixture-of-Transformers` `VLA` `双臂机器人`

---

## 一句话总结

Motus 用 Mixture-of-Transformers（MoT）把预训练视频生成模型（Wan2.2-5B）、预训练视觉语言理解模型（Qwen 系列 VL-2B）和一个新训练的动作专家通过共享自注意力层（Tri-model Joint Attention）接到一起，并用光流压缩出的"潜动作"作为像素级 delta action 打通无标注视频与有标签机器人数据，从而单一权重同时支持 VLA / World Model / IDM / VGM / 视频-动作联合预测 5 种推理模式；在 RoboTwin 2.0 仿真上相对 X-VLA 提升约 15%、相对 π0.5 提升约 45%,真实世界两台双臂平台上相对 π0.5 提升 11%~48%。

## 一、问题与动机

论文把"具身智能体应具备的能力"拆成 5 个概率分布,分别对应当前领域内 5 类主流建模范式：

- VLA：$p(\boldsymbol{a}_{t+1:t+k} \mid \boldsymbol{o}_t, \ell)$
- World Model（WM）：$p(\boldsymbol{o}_{t+1:t+k} \mid \boldsymbol{o}_t, \boldsymbol{a}_{t:t+k})$
- 逆动力学模型（IDM）：$p(\boldsymbol{a}_{t+1:t+k} \mid \boldsymbol{o}_{t:t+k})$
- 视频生成模型（VGM）：$p(\boldsymbol{o}_{t+1:t+k} \mid \boldsymbol{o}_t, \ell)$
- 视频-动作联合预测：$p(\boldsymbol{o}_{t+1:t+k}, \boldsymbol{a}_{t+1:t+k} \mid \boldsymbol{o}_t, \ell)$

现状是这 5 种能力被割裂在不同模型里训练：VLA 只学"看图/听指令 → 出动作",世界模型/生成式方法只学"预测未来",而近期的 $\mathcal{F}_1$ 虽然把 VLA 和 IDM 融合,却排除了世界模型/视频生成模型,统一并不完整。论文指出两个核心挑战：

**挑战一：多模态生成能力的统一本身不平凡。** 已有的统一世界模型（UWM）提供了理论雏形,但要么从零训练,要么基于较小的基座模型,要么即使引入了先验也总是缺一半——缺 VLM 的视觉语言理解先验,或者缺 VGM 的物理交互先验,两者难以兼得。

**挑战二：如何利用大规模异构数据。** 不同本体的动作空间维度、语义差异很大,大量互联网视频、第一人称人类视频天然没有动作标签,导致动作专家难以像 VLM/VGM 一样吃到网络规模的预训练数据。

## 二、核心方法

### 2.1 MoT 架构与 Tri-model Joint Attention

Motus 由三个专家组成：视频生成专家（基于 Wan2.2-5B 初始化）、动作专家（与 Wan 同规模、从头训练的 Transformer block）、理解专家（基于 Qwen 系列 VL-2B,论文写作 Qwen3-VL-2B,骨干冻结❄️,仅在其最后一层 token 之上再堆叠若干 Transformer block）。三个专家各自保留独立的输入编码、AdaLN/LayerNorm 与 FFN,只在多头自注意力层把三路 token 拼接在一起做联合注意力,称为 **Tri-model Joint Attention**——既保留各专家的专精能力,又实现跨模态知识融合,这一设计延续了 Bagel 等统一多模态模型里 MoT 共享自注意力层的思路,并首次把它用到"理解 + 视频生成 + 动作"三路专家的具身场景。

### 2.2 UniDiffuser 式训练目标（统一 5 种模式的关键）

Motus 用整流流匹配（rectified flow）同时对视频和动作两路建模,各自采样独立的加噪时间步和噪声：

$$l_{\text{action}}^{\theta} = \mathbb{E}_{(\boldsymbol{o}_{t:t+k},\boldsymbol{a}_{t+1:t+k},\ell)\sim\mathcal{D}}\left\|v_a^{\theta} - (\epsilon_a - \boldsymbol{a}_{t+1:t+k})\right\|_2^2,\quad \tau_a\sim\mathcal{U}(0,T_\tau),\ \epsilon_a\sim\mathcal{N}(0,\boldsymbol{I})$$

$$l_{\text{obs}}^{\theta} = \mathbb{E}_{(\boldsymbol{o}_{t+1:t+k},\boldsymbol{a}_{t+1:t+k},\ell)\sim\mathcal{D}}\left\|v_o^{\theta} - (\epsilon_o - \boldsymbol{o}_{t+1:t+k})\right\|_2^2,\quad \tau_o\sim\mathcal{U}(0,T_\tau),\ \epsilon_o\sim\mathcal{N}(0,\boldsymbol{I})$$

$$l^{\theta} = l_{\text{action}}^{\theta} + l_{\text{obs}}^{\theta}$$

**大白话说**：把"预测视频"和"预测动作"都当成整流流匹配任务——模型学的是一条从纯噪声直线飞向真实数据的"速度场"。关键技巧是视频和动作各自随机采样自己的加噪程度 $\tau_o,\tau_a$（类似 UniDiffuser 的做法）：训练时两者噪声程度随机组合,于是推理时只要把某一路的时间步钉死在"全干净"（当条件）或"全噪声"（当待生成目标）,就能让同一套权重在 VLA / WM / IDM / VGM / 联合预测 5 种模式间自由切换（具体切换规则见附录 Algorithm 2-6：例如做 IDM 时令观测时间步为 0、动作时间步从 $T_\tau$ 递减到 0）,不必训练 5 个独立模型。

### 2.3 Action-Dense Video-Sparse Prediction

动作分块（action chunking）会同时预测未来一段视频帧和一段动作序列 $\boldsymbol{o}_{t+1:t+k}, \boldsymbol{a}_{t+1:t+k}$,但视频 token 数量远多于动作 token,导致 Tri-model Joint Attention 里模型偏向拟合视频、削弱动作预测能力,同时也拖慢训练/推理。解法很直接：训练和推理时都对视频帧降采样,让视频 token 数与动作 token 数基本持平,例如把视频帧率设为动作帧率的 1/6（实际配置为视频 8 帧 @5Hz、动作 chunk 48 步 @30Hz,比例正好 1:6）。

### 2.4 潜动作（Latent Action）：把无标注视频变成"动作预训练数据"

为解决挑战二,Motus 引入基于光流的潜动作表示,把视觉动态和控制信号连接起来：

1. 用 DPFlow 计算相邻帧间的稠密光流,并转成 RGB 光流图；
2. 用一个预训练的深度卷积自编码器（DC-AE,ICLR 2025）重建光流,同时把光流压缩编码成 4 个 512 维 token；
3. 一个轻量编码器把拼接后的 $4\times512$ 特征投影成 14 维向量 $z_t$（与常见机器人动作空间的维度量级相当),作为"潜动作"。

这样,即使是没有动作标注的互联网视频、第一人称人类视频,也能提供"delta action"级别的运动监督信号,让动作专家在预训练阶段就能吃到大规模无标签数据。

为了让潜空间对齐到真实可执行动作分布,训练中混入 AnyPos 风格的任务无关数据（用 Curobo 在目标机器人动作空间内随机采样得到的 image-action 对）,90% 数据做无监督光流重建、10% 有标签轨迹做弱动作监督,总损失为：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_a \|\boldsymbol{a}_{\text{real}} - \boldsymbol{a}_{\text{pred}}\|^2 + \beta \mathcal{L}_{\text{KL}}$$

**大白话说**：潜动作 VAE 主体靠"光流重建"这个自监督信号学会捕捉画面里的运动模式,只搭配一小撮真实动作标签做"锚点"对齐,再用 KL 正则防止潜空间学偏,这样就把动作预训练的数据来源从"必须有动作标签的机器人轨迹"扩展到了任意包含运动信息的视频。

### 2.5 三阶段训练与六层数据金字塔

| 阶段 | 数据 | 训练对象 |
|---|---|---|
| 预训练基座（现成） | Level 1 网络数据 | VGM 与 VLM（各自独立预训练好的模型） |
| Stage 1（视频生成) | Level 2 第一人称人类视频、Level 3 合成数据、Level 5 多机器人任务轨迹 | 仅 VGM |
| Stage 2（统一训练+潜动作) | Level 2、Level 3、Level 4 任务无关数据、Level 5 | Motus 全部三个专家,使用潜动作 |
| Stage 3（目标机器人微调) | Level 6 目标机器人任务轨迹 | Motus 全部三个专家,使用真实动作 |

六层"具身数据金字塔"（Web 数据→第一人称人类视频→合成数据/任务无关数据→多机器人轨迹→目标机器人轨迹)按数据量递减、数据质量/相关性递增排列,Level 3 与 Level 4 的顺序在实践中可互换。预训练数据具体构成（附录 Table 12)：Egodex 人类视频 230,949 条、Agibot（Genie-1 机器人)728,209 条、RDT/RoboMind（Aloha、Franka）共约 2.3 万条、RoboTwin 合成数据 27,500 条、任务无关数据 1,000 条、目标机器人自采数据 2,000 条。三阶段训练总算力约为 Stage1 8000、Stage2 10000、Stage3 400 GPU 小时（合计约 1.84 万 GPU 小时),Batch Size 均为 256,AdamW 优化器。模型总参数约 8B（VGM 5.00B + VLM 2.13B + 动作专家 641.5M + 理解专家新增块 253.5M)。

## 三、实验结果

**仿真（RoboTwin 2.0,50 个任务,Clean/Randomized 两种场景,附录 Table 14 全量平均）**

| 方法 | Clean | Randomized |
|---|---|---|
| GO-1 | 37.8% | 36.24% |
| π0.5 | 42.98% | 43.84% |
| X-VLA | 72.8% | 72.84% |
| Motus（无预训练,仅架构） | 77.56% | 77.00% |
| Motus（仅 Stage1 预训练） | 82.26% | 81.86% |
| **Motus（完整三阶段）** | **88.66%** | **87.02%** |

正文摘要口径：仿真中相对 X-VLA **+15%**、相对 π0.5 **+45%**（绝对成功率差值意义下)。

**真实世界（AC-One、Agilex-Aloha-2 两台双臂平台,每任务 100 条示教,partial success rate 指标,附录 Table 3）**

| 平台（任务数） | π0.5 | 无预训练 | Motus |
|---|---|---|---|
| AC-One（9 任务均值） | 14.79% | 25.86% | **63.22%** |
| Agilex-Aloha-2（5 任务均值） | 48.60% | 26.60% | **59.30%** |

个别子任务细节：AC-One 上"Grind Coffee Beans with Grinder"从 π0.5 的 8% 提升到 Motus 的 **92%**；"Put Bread into Oven"细分子目标显示 Motus 在"关烤箱门/按开始键"等后段子任务上明显优于无预训练版本（partial success rate 42% vs 40% vs π0.5 12%)。Agilex-Aloha-2 上"Get Water from Water Dispenser"从 π0.5 的 62%、无预训练的 8% 提升到 **96%**。摘要口径：真实场景相对基线提升 **+11%~48%**。

**分模式能力验证（附录）**

- IDM 模式：Motus 直接作为逆动力学模型使用时,动作 MSE 为 **0.014**,低于专门训练的 ResNet18+MLP（0.044)和 DINOv2+MLP（0.122)基线。
- VGM 模式：真实机器人数据上视频生成质量（Agilex-Aloha-2 / AC-One 均值）FID 11.209、FVD 61.21、SSIM 0.866、LPIPS 0.064、PSNR 25.07。
- VLA 模式 vs 联合预测模式：RoboTwin 2.0 Randomized 平均成功率,纯 VLA 模式 83.90% 低于联合视频-动作预测模式 87.02%,说明同时生成视频对动作预测有正向迁移。
- LIBERO-Long：Motus 达到 **97.6**,与 X-VLA 并列最优,高于 π0（85.2)、GR00T-N1（90.6)、UniVLA（94.0)、OpenVLA-OFT（94.5)。
- VLABench：In-Distribution 平均成功率 π0.5 0.43 → Motus 0.48；Cross-Category 平均 π0.5 0.22 → Motus 0.25。

## 四、局限性

- **训练成本高昂**：三阶段预训练累计约 1.84 万 GPU 小时（约 767 GPU-day),叠加 8B 参数规模,复现门槛较高,论文未讨论更小算力预算下的可行方案。
- **依赖冻结的现成基座先验**：理解专家的 VLM 骨干（Qwen 系列 VL-2B)在全程训练中保持冻结,只训练其上新增的 Transformer block；视频专家初始化自 Wan2.2-5B。这意味着 Motus 的场景理解上限、物理常识边界很大程度上继承自这两个现成模型,论文没有做"解冻 VLM""替换更大/更小基座"的消融,统一架构本身带来的增益与基座模型质量的贡献难以完全解耦。
- **潜动作对光流质量的依赖**：delta action 由 DPFlow 光流经 DC-AE 压缩而来,遮挡、快速运动、透明/反光物体等光流估计失效场景下,潜动作监督信号的可靠性未被单独评估。
- **真实世界评测规模有限**：仅在 2 个自建双臂平台（AC-One、Agilex-Aloha-2)、共 14 个任务、每任务 100 条示教上验证,尚未在第三方公开的真机 benchmark 上交叉验证；且部分可变形物体任务（如 Fold Towel)绝对成功率仍偏低（AC-One 14.5%、Aloha-2 39%),显示柔性物体操作仍是短板。
- **降采样比例等超参数缺乏系统消融**：Action-Dense Video-Sparse Prediction 中视频/动作帧率比 1:6 是手工设定,论文未给出该比例的敏感性分析或理论依据。
- 论文标题为"Conclusion and Limitations"的收尾部分实际主要是展望（更先进的统一架构、更通用的运动先验、从网络规模视频学习潜动作),并未给出显式的失败案例分析,局限性讨论有限。

## 五、评价与展望

**优点**：Motus 相对 UWM（Unified World Models)的核心改进,是把"从零训练/小模型统一"换成了"复用现成大规模预训练 VGM+VLM 先验再统一",直接对应论文提出的挑战一,且实验（Table 2/14 中"无预训练"vs"仅 Stage1"vs"完整三阶段"的阶梯对比)较为干净地展示了预训练先验和潜动作预训练各自带来的增量。MoT + Tri-model Joint Attention 的设计选择——专家各自保留结构特点、只共享自注意力层——与统一多模态生成模型（如 Bagel)的思路一脉相承,是该范式向"理解 + 生成 + 动作"三路扩展的一次具体实践。用光流构造像素级"delta action"作为潜动作,延续了 LAPA/AdaWorld/Moto 等潜动作模型的思路,但把这一表示同时接入 IDM、VGM、WM、VLA、联合预测五种推理模式并逐一验证（而非只验证下游策略成功率),是本文实验设计上比较扎实的地方——尤其 IDM 模式 MSE 反超专用 IDM baseline、联合预测模式优于纯 VLA 模式这两组结果,为"统一建模能带来正迁移"提供了较有说服力的证据。

**开放问题与可能的改进方向**：其一,推理时切换 VLA/WM/IDM/VGM/联合预测模式依赖人工设定 $\tau_o,\tau_a$ 的固定取值（见附录 Algorithm 2-6),本质是同一权重在不同"噪声时间步配方"下的五次调用,而非模型自主判断当前该用哪种模式或融合多种模式,让调度本身可学习是一个值得探索的方向。其二,论文未像"What do latent action models actually learn?"（Zhang et al., 2025)那样对潜动作的语义/可迁移性做专门的探针分析,潜动作到底学到了可跨本体迁移的运动基元,还是仅充当了额外的自监督正则项,仍不完全清楚。其三,与同期的 $\mathcal{F}_1$（VLA+IDM 融合,但不含 WM/VGM)和 UWM（统一但基座较小/从零训练)相比,Motus 是目前公开工作中先验最完整的统一具身模型,但也因此规模和算力开销最大,统一架构相对于"分别用大模型做视频生成+分别训练 VLA"两条独立管线的性价比,论文未给出同等算力下的直接对照实验,留待后续工作补充。其四,真实世界验证目前局限于两台自建双臂平台,能否在更广泛的开源机器人平台/公开真机数据集上复现类似增益,是外部可复现性层面的开放问题。

## 参考

- Zhu, C. et al. *Unified World Models: Coupling Video and Action Diffusion for Pretraining Large Robotic Datasets.* arXiv:2504.02792, 2025.（UWM,本文直接对标的统一世界模型前作）
- Lv, Q., Kong, W. et al. *F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions.* arXiv:2509.06951, 2025.（VLA+IDM 融合但不含 WM/VGM 的相关统一模型）
- Deng, C. et al. *Bagel: Emerging Properties in Unified Multimodal Pretraining.* arXiv:2505.14683, 2025.（MoT 共享自注意力层统一理解与生成的架构来源）
- Zheng, J. et al. *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model.* arXiv:2510.10274, 2025.（仿真实验的主要基线之一）
- Black, K. et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization.* CoRL, 2025.（仿真与真实世界实验的主要基线）
