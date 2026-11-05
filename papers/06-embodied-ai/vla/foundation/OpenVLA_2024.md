# OpenVLA：开源 7B 通用视觉-语言-动作基础模型

> **论文**：*OpenVLA: An Open-Source Vision-Language-Action Model*
>
> **作者**：Moo Jin Kim\*, Karl Pertsch\*, Siddharth Karamcheti\*, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn
>
> **机构**：Stanford University · UC Berkeley · Toyota Research Institute (TRI) · Google DeepMind · Physical Intelligence · MIT
>
> **发布时间**：2024 年 6 月（`arXiv:2406.09246`）
>
> **发表状态**：`CoRL 2024`（PMLR vol. 270, pp. 2679–2713）
>
> **链接**：[arXiv](https://arxiv.org/abs/2406.09246) | [PDF](https://arxiv.org/pdf/2406.09246) | [项目主页](https://openvla.github.io) | [代码](https://github.com/openvla/openvla)
>
> **分类标签**：`VLA Foundation Model` `Autoregressive Action Tokenization` `SigLIP + DINOv2` `Open X-Embodiment` `LoRA Fine-Tuning` `Quantized Inference`

## 一句话总结

提出首个完全开源的 7B 参数级通用视觉-语言-动作模型 **OpenVLA**：基于 **Prismatic-7B VLM**（通道拼接 **DINOv2** 细粒度空间特征与 **SigLIP** 高层语义特征 + **Llama-2 7B** 语言骨干），在精选自 Open X-Embodiment 的 **97 万条**真机轨迹上端到端放开视觉编码器训练 27 个 Epoch，通过分位数 256-bin 离散化覆盖词表最不常用 Token 进行自回归动作预测，在 29 项多机器人评测任务上以 $1/7$ 参数量超越闭源 55B 模型 **RT-2-X** 达 **+16.5%**，并系统验证了单卡 **LoRA 微调**（仅训练 1.4% 参数即可追平全量微调）与 **INT4 量化推理**（显存减半至 7.0 GB 且成功率无损）的高效适配范式。

## 核心贡献

1. **融合语义与空间表征的 7B 开源 VLA 架构**：揭示单一视觉编码器（如纯 CLIP 或纯 SigLIP）缺乏机器人精细操作所需的底层空间几何表征；通过引入 **Prismatic-7B VLM** 的双路视觉编码器（将预训练 **SigLIP** 与 **DINOv2** 输出的 256 个图像 Patch 特征沿通道维拼接后经 MLP 映射至 **Llama-2 7B** 词嵌入空间），在多物体语言接地与空间操作任务上比 LLaVA 骨干绝对成功率提升约 **10%**（LLaVA 又比 IDEFICS-1 高 **35%**）。
2. **颠覆传统 VLM 训练经验的 VLA 预训练配方**：在 64 张 A100（21,500 GPU 小时）上对 Open X-Embodiment 97 万条单臂轨迹进行大规模预训练，总结出三条关键工程律：
   - **必须放开视觉编码器微调（Unfrozen Vision Encoder）**：在传统 VLM 预训练中冻结视觉编码器通常最优，但在 VLA 训练中放开视觉编码器至关重要（Bridge 对比实验中放开视觉编码器平均成功率达 **80.0%**，而冻结视觉编码器仅为 **46.7%** 且伴随机械臂抖动）；
   - **多轮迭代直至动作 Token 准确率超过 95%**：不同于 LLM/VLM 仅训练 1–2 个 Epoch，VLA 真机性能随训练轮数持续攀升，最终完整遍历数据集 **27 个 Epoch**；
   - **$224\times 224$ 分辨率性价比最优**：$384\times 384$ 输入耗时增加 $3\times$ 却未带来真机控制收益。
3. **建立参数高效微调（LoRA）与低精度量化部署（INT4）完整工具链**：证明使用 **LoRA（$r=32$）** 仅需更新 **1.4%（97.6M）** 参数即可在单张 A100 上用 10–15 小时适配新机械臂（平均成功率 **68.2%** vs 全参数微调 **69.7%**，显存从 163.3 GB 降至 59.7 GB）；推理期采用 **INT4 量化**可将显存从 `bfloat16` 的 16.8 GB 压缩至 **7.0 GB**，且真机成功率完全无损（**71.9%** vs **71.3%**）。

## 1. 问题与动机

以 RT-2 和 RT-2-X 为代表的视觉-语言-动作模型（VLA）证明了将互联网预训练视觉-语言大模型（VLM）迁移到机器人低层动作控制的巨大潜力。然而，VLA 在机器人社区的普及受制于两大瓶颈：
1. **闭源壁垒与关键训练细节缺失**：RT-2-X（55B）等早期主流 VLA 完全闭源，仅提供受限 API；而开源社区虽有 Octo（93M）和 RT-1-X（35M），但它们均从零训练、未继承互联网规模 VLM 先验，面对未见过的干扰物、新物体与复杂语言指令时泛化能力急剧退化。
2. **下游新本体与新任务的高效微调范式空白**：现有工作主要评估“开箱即用”的零样本泛化，但实际部署往往需要针对全新机械臂（如不同控制频率的 Franka Panda）和少量演示数据（10–150 条）进行快速适配。如何在有限算力（单卡或消费级显卡）下高效微调并实时推理 7B 级 VLA 模型，缺乏系统性的实证研究。

## 2. 模型架构与预训练方法

### 2.1 三段式多模态架构（SigLIP + DINOv2 + Llama-2 7B）

给定单视角第三人称图像观测 $I_t \in \mathbb{R}^{224\times 224\times 3}$ 与自然语言任务指令 $l$，OpenVLA 预测 7 维单步末端执行器相对控制动作 $\mathbf{a}_t = [\Delta x, \Delta y, \Delta z, \Delta \phi, \Delta \theta, \Delta \psi, g]^\top \in \mathbb{R}^7$（3 维位置增量、3 维姿态增量、1 维夹爪开合）：

1. **双路融合视觉编码器（Fused Vision Encoder，600M 参数）**：输入图像 $I_t$ 同时送入预训练的 **SigLIP ViT-SO400M**（提供高层图文对齐语义特征）和 **DINOv2 ViT-L**（提供自监督细粒度空间几何特征），各自提取 $N_v = 256$ 个图像 Patch 嵌入 $\mathbf{E}_{\text{SigLIP}} \in \mathbb{R}^{256\times d_1}$ 与 $\mathbf{E}_{\text{DINO}} \in \mathbb{R}^{256\times d_2}$，沿特征通道维直接拼接：

$$
\mathbf{E}_{\text{vis}} = \big[\mathbf{E}_{\text{SigLIP}} \;\|\; \mathbf{E}_{\text{DINO}}\big] \in \mathbb{R}^{256 \times (d_1 + d_2)}
$$

2. **视觉-语言投影器（Projector）**：采用轻量级两层（或 GELU 三层）MLP $f_{\text{proj}}$ 将拼接后的视觉特征映射至语言模型的词嵌入维度 $d_{\text{LLM}}$：

$$
\mathbf{H}_{\text{vis}} = f_{\text{proj}}(\mathbf{E}_{\text{vis}}) \in \mathbb{R}^{256 \times d_{\text{LLM}}}
$$

3. **自回归语言模型骨干（Llama-2 7B）**：将视觉 Token $\mathbf{H}_{\text{vis}}$ 与语言指令 Token $\mathbf{H}_{\text{lang}}$ 沿序列维拼接后输入 Llama-2 7B 解码器，以自回归方式逐个预测 7 个离散动作 Token。

### 2.2 基于分位数的连续动作离散化与词表覆写

为使标准因果语言模型直接输出连续控制量，OpenVLA 对动作空间每一维独立执行 **分位数均匀分箱（Quantile-based 256-bin Discretization）**：
- **1%–99% 分位数截断（抗离群值设计）**：RT-2 直接使用训练集中的最小/最大值 $[\min, \max]$ 划分 256 个区间，极易受采集数据中的极端离群动作影响而拉宽区间跨度、损失有效控制精度。OpenVLA 改用训练集中第 $1\%$ 分位数 $q_{0.01}^{(d)}$ 与第 $99\%$ 分位数 $q_{0.99}^{(d)}$ 作为第 $d$ 维动作的归一化边界，将区间 $[q_{0.01}^{(d)}, q_{0.99}^{(d)}]$ 均匀划分为 $B = 256$ 个离散桶：

$$
\hat{a}_{t,d} = \mathrm{clip}\!\left(\left\lfloor 256 \cdot \frac{a_{t,d} - q_{0.01}^{(d)}}{q_{0.99}^{(d)} - q_{0.01}^{(d)}} \right\rfloor,\, 0,\, 255\right) \in \{0, 1, \dots, 255\}, \quad d \in \{1,\dots,7\}
$$

- **Llama 词表尾部 256 Token 覆写**：由于 Llama-2 分词器仅预留了 100 个特殊 Token，不足以容纳 256 个动作桶，OpenVLA 直接覆写 Llama-2 词表中**使用频率最低的最后 256 个 Token** 来表示离散动作整数 $\{0,\dots,255\}$。
- **自回归交叉熵训练目标**：模型仅在 7 个预测动作 Token 上计算下一步预测交叉熵损失：

$$
\mathcal{L}_{\text{OpenVLA}}(\theta) = -\sum_{d=1}^{7} \log p_\theta\!\left(\hat{a}_{t,d} \;\middle|\; \mathbf{H}_{\text{vis}},\, \mathbf{H}_{\text{lang}},\, \hat{a}_{t,<d}\right)
$$

### 2.3 预训练数据策展与关键清洗细节（970k 轨迹）

OpenVLA 从 Open X-Embodiment（OpenX）中筛选出具备至少一个第三人称相机、采用单臂末端执行器（End-Effector）控制的 27 个子数据集，合计 **970k 条真机演示轨迹**，沿用 Octo 的启发式采样权重平衡跨本体与场景多样性（其中 Fractal 占 12.7%、Kuka 占 12.7%、BridgeData V2 占 13.3%、BC-Z 占 7.5%、FMB 占 7.1%；初期以 10% 权重加入 DROID，后因动作 Token 拟合缓慢在最后 1/3 训练阶段移除 DROID）。

**关键数据清洗发现（过滤全零 No-Op 动作）**：作者发现原始 BridgeData V2 数据集中每条轨迹的第 1 个时间步均记录了全零动作（All-zero action）。高容量的 7B VLA 若不过滤这些全零步，在测试部署时会频繁预测全零动作而“原地僵死（Freezing）”（闭源 RT-2-X 正因此缺陷，在评测 Bridge 时不得不采取“始终采样第二高概率 Token”的妥协技巧）。OpenVLA 通过在预处理阶段剔除每条轨迹的首步全零动作及 LIBERO 中的静止帧，从根源上消除了策略僵死现象。

## 3. 实验结果与定量分析

### 3.1 多机器人平台开箱即用泛化评测（BridgeData V2 & Google Robot）

在 **WidowX 机械臂（BridgeData V2 环境，17 项任务 $\times$ 10 次 = 170 次真机测试）** 与 **Google Mobile Manipulator（12 项任务 $\times$ 5 次 = 60 次真机测试）** 上，沿着视觉（Visual）、运动（Motion）、物理（Physical）、语义（Semantic）泛化及多物体语言接地（Language Grounding）五个维度对比 RT-1-X（35M）、Octo（93M）与闭源 RT-2-X（55B）（附录表 2、表 4）：

| 评测平台与泛化类别 | 任务数 / 总回合数 | RT-1-X (35M) | Octo (93M) | RT-2-X (55B 闭源) | **OpenVLA (7B 开源)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BridgeData V2 — 视觉泛化 (Visual)** | 5 任务 / 50 回合 | 8.0% | 29.0% | 52.0% | **87.0%** |
| **BridgeData V2 — 运动泛化 (Motion)** | 2 任务 / 20 回合 | 25.0% | 7.5% | 55.0% | **60.0%** |
| **BridgeData V2 — 物理泛化 (Physical)** | 3 任务 / 30 回合 | 10.0% | 20.0% | 26.7% | **76.7%** |
| **BridgeData V2 — 语义泛化 (Semantic)** | 4 任务 / 40 回合 | 26.3% | 0.0% | **38.8%** | 36.3% |
| **BridgeData V2 — 语言接地 (Language Grounding)** | 3 任务 / 30 回合 | 30.0% | 40.0% | 85.0% | **90.0%** |
| **BridgeData V2 总平均成功率** | **17 任务 / 170 回合** | $18.5 \pm 2.7\%$ | $20.0 \pm 2.6\%$ | $50.6 \pm 3.5\%$ | **$70.6 \pm 3.2\%$（+20.0%）** |
| **Google Robot — 分布内 (In-Distribution)** | 5 任务 / 25 回合 | 32.0% | 44.0% | 72.0% | **88.0%** |
| **Google Robot — 分布外 (OOD)** | 7 任务 / 35 回合 | 34.3% | 14.3% | **82.9%** | **82.9%** |
| **Google Robot 总平均成功率** | **12 任务 / 60 回合** | $33.3 \pm 6.1\%$ | $26.7 \pm 5.8\%$ | $78.3 \pm 5.4\%$ | **$85.0 \pm 4.6\%$** |

在全部 29 项真机任务（230 次测试）上，OpenVLA 以 7B 参数量在 BridgeData V2 上超越 55B 的 RT-2-X 达 **+20.0%**（综合 29 项任务绝对提升 **+16.5%**）。消融实验（附录表 8）证实：若移除 OpenX 跨本体预训练仅在 BridgeData V2 上训练（`OpenVLA-Bridge`），成功率从 **76.3%** 暴跌至 **45.6%**（-30.7%）；若进一步移除 DINOv2 仅用 SigLIP（`OpenVLA-Bridge-SigLIP`），成功率继续跌至 **40.6%**（-5.0%）。

### 3.2 新机械臂小样本微调实验（Franka 真机与 LIBERO 仿真）

1. **Franka-Tabletop（5Hz，6 项任务）与 Franka-DROID（15Hz，Wipe Table）真机微调（10–150 条演示，附录表 5）**：

| 评测环境与设定 | Diffusion Policy (Scratch) | Diffusion Policy (Matched) | Octo (Fine-tuned) | OpenVLA (Scratch，无 OpenX) | **OpenVLA (Fine-tuned)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Franka-Tabletop 平均（99 回合）** | $48.5 \pm 4.9\%$ | $43.4 \pm 4.7\%$ | $43.4 \pm 4.4\%$ | $43.4 \pm 4.6\%$ | **$67.2 \pm 4.0\%$** |
| **Franka-DROID 平均（30 回合）** | $35.0 \pm 8.0\%$ | $26.7 \pm 7.5\%$ | $38.3 \pm 8.5\%$ | $21.7 \pm 6.6\%$ | **$58.3 \pm 7.2\%$** |

Diffusion Policy 在单一指令窄任务（如 `"Pour Corn into Pot"`）上轨迹更平滑，但在包含多物体与多语言指令的任务（如 `"Move <object> onto Plate"`、`"Knock <object> Over"`）上严重退化；OpenVLA 是唯一在所有任务上成功率均 $\ge 50\%$ 的方法，综合超越从头训练的 Diffusion Policy 达 **+20.4%**。

2. **LIBERO 仿真基准微调结果（4 大套件 $\times$ 10 任务 $\times$ 500 回合 $\times$ 3 随机种子，附录表 11）**：

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | **Average SR $\uparrow$** | 平均排名 $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diffusion Policy (Scratch)** | $78.3 \pm 1.1\%$ | **$92.5 \pm 0.7\%$** | $68.3 \pm 1.2\%$ | $50.5 \pm 1.3\%$ | $72.4 \pm 0.7\%$ | 2.5 |
| **Octo (Fine-tuned)** | $78.9 \pm 1.0\%$ | $85.7 \pm 0.9\%$ | **$84.6 \pm 0.9\%$** | $51.1 \pm 1.3\%$ | $75.1 \pm 0.6\%$ | 2.0 |
| **OpenVLA (LoRA $r=32$ Fine-tuned)** | **$84.7 \pm 0.9\%$** | $88.4 \pm 0.8\%$ | $79.2 \pm 1.0\%$ | **$53.7 \pm 1.3\%$** | **$76.5 \pm 0.6\%$** | **1.5** |

### 3.3 参数高效微调（PEFT）与量化推理分析

1. **PEFT 微调策略对比（表 1，Franka-Tabletop 代表性任务）**：
   - **Full FT（全量微调）**：成功率 **$69.7 \pm 7.2\%$**，可训练参数 $7{,}188.1\text{M}$，显存 $163.3\text{ GB}$（需 2 卡 FSDP）；
   - **Last layer only / Frozen vision**：成功率仅 $30.3\% / 47.0\%$，再次证明视觉特征适配目标场景的必要性；
   - **Sandwich（放开视觉编码器 + 词嵌入 + 最后一层）**：成功率 $62.1\pm 7.9\%$，可训练参数 $914.2\text{M}$，显存 $64.0\text{ GB}$；
   - **LoRA ($r=32$)**：成功率 **$68.2 \pm 7.5\%$**，仅训练 **$97.6\text{M}$（1.4%）** 参数，单卡显存降至 **$59.7\text{ GB}$**，算力开销降低 $8\times$。
2. **INT8 / INT4 量化推理与控制频率耦合效应（表 2 & 附录表 10）**：
   - 在非阻塞控制器（5Hz）下，`bfloat16`（显存 16.8 GB）成功率为 **$71.3\%$**；`INT4`（显存 **7.0 GB**）成功率为 **$71.9\%$**；而 `INT8`（显存 10.2 GB）成功率却跌至 $58.1\%$。
   - 作者通过切换为“阻塞式控制（Blocking Control，每步等推理完成后再执行）”揭开了原因：`INT8` 算子存在反量化开销，在 A5000 上推理频率跌至 $1.2\text{ Hz}$，破坏了 5Hz 非阻塞控制器的系统动力学匹配（在阻塞控制下 `bfloat16 / INT8 / INT4` 成功率分别为 $70.0\% / 74.4\% / 68.8\%$，完全一致）；而 `INT4` 因访存减半抵消了算子开销，在 RTX 4090 上可达 ~6Hz，完美兼顾低显存与高成功率。

## 4. 局限性与后续演进（通往 OpenVLA-OFT）

OpenVLA 奠定了开源自回归 VLA 的基石，但作者在文末明确指出了其三大固有局限（这直接催生了后续 RSS 2025 的 **OpenVLA-OFT**）：
1. **单步自回归解码速度慢（3–6 Hz）且不支持动作分块（Action Chunking）**：每次预测 $D=7$ 维动作需串行执行 7 次 LLM 解码器前向传播；若引入长度为 $K$ 的动作分块，则需串行解码 $7K$ 个 Token，延迟随 $K$ 线性暴增，无法部署于 25–50 Hz 的双臂灵巧操作（如 ALOHA）；
2. **256-bin 离散化损失连续动作精度**，且单视角输入（无腕部相机与本体状态）限制了精细遮挡操作；
3. 在长视野任务（如 `LIBERO-Long` 仅 $53.7\%$）上误差累积严重。
