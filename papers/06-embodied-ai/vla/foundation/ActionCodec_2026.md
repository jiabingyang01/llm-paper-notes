# ActionCodec：什么造就了好的动作分词器

> **论文**：*ActionCodec: What Makes for Good Action Tokenizers*
>
> **作者**：Zibin Dong, Yicheng Liu et al.（前两位为共同一作，通讯作者为 Hang Zhao、Jianye Hao）
>
> **机构**：Knowin AI（工作完成于实习期间）、清华大学、天津大学、复旦大学、上海创智学院（Shanghai Innovation Institute）
>
> **发布时间**：2026 年 02 月（arXiv 2602.15397）
>
> **发表状态**：未录用（预印本，原文标注 "Preprint. February 18, 2026."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.15397) | [PDF](https://arxiv.org/pdf/2602.15397)
>
> **分类标签**：`VLA` `动作分词器` `向量量化` `信息论` `离散动作表示`

---

## 一句话总结

论文从信息论角度把"好的动作分词器"归结为四条可操作准则——最大化相邻动作块的 token 重叠率、控制词表容量而非一味追求高保真、平衡视觉-语言对齐与残差语法、保持 token 间独立性——并据此设计出 ActionCodec：在完全不使用机器人预训练数据的情况下，让 SmolVLM2-2.2B 仅靠自回归微调即在 LIBERO 上达到 95.5%（配合 Block-wise Autoregression 架构达 97.4%）的成功率，超过多个依赖大规模机器人预训练的模型。

## 一、问题与动机

VLA 模型若保持 VLM 原生的自回归范式（把动作离散化为 token，而不是接 diffusion/regression head），往往能更好地保留预训练模型的世界知识和指令跟随能力，这在 π0.5、GR00T-N1、Knowledge Insulation 等近期工作中已成为趋势。但动作 token 化的设计长期停留在三条路线：（1）启发式方法，如 OpenVLA 式均匀分箱（binning）、把动作直接转成原始字符串；（2）半数据驱动方法，如 FAST 对动作的频域信号做 Byte-Pair Encoding；（3）数据驱动方法，主要基于 Vector Quantization（VQ）学习离散潜表示（如 VQ-VLA、MiniVLA）。

VQ 类方法理论上对先验依赖最少、最灵活，但实践中反而经常打不过分箱这类简单方案。作者认为症结在于：已有工作大多只用重建保真度（reconstruction fidelity）评价 VQ tokenizer，却忽视了 tokenizer 对 VLA 训练动态（training dynamics）的直接影响，把 VQ tokenizer 当成黑盒组件对待。论文要回答的核心问题是——从 VLA 优化的角度看，什么才是好的动作 tokenizer？

## 二、核心方法

### 2.1 把动作 token 化建模为监督信号设计问题

论文先将 VLA 的负对数似然损失做分解：

$$
\mathbb{E}[\mathcal{L}_{\text{NLL}}] = D_{KL}(P_{\text{data}} \parallel P_\theta) + H(C \mid V, L)
$$

用大白话说：VLA 的训练损失可以拆成"模型拟合数据分布的难度"和"tokenizer 本身带来的监督歧义"两部分。后一项 $H(C\mid V,L)$ 是条件熵——如果同一个视觉-语言输入被 tokenizer 映射到多种彼此冲突的 token 序列，模型就要花大量梯度去拟合这种伪噪声而不是物理规律。因此好的 tokenizer 首先要压低这一项。

进一步用互信息把条件熵拆开：

$$
H(C \mid V, L) = \underbrace{H(C \mid A)}_{\text{Artifact Entropy}} + \underbrace{I(C; A)}_{\text{Capacity}} - \underbrace{I(C; V, L)}_{\text{Perceptual Alignment}}
$$

据此得到三条设计准则。

**(a) Overlap Rate（重叠率）。** $H(C\mid A)$ 刻画 token 空间的拓扑不稳定性：如果连续动作 $A$ 受到微小扰动（如传感器噪声）就让离散编码 $C$ 发生跳变，监督信号就会不稳定。论文用相邻时间步动作块（$A_t$、$A_{t+1}$）编码出的 token 序列重叠率（overlap rate，OR）作为该项的经验代理指标——重叠率越高，供给 VLA 的监督信号越稳定。

**(b) Capacity 与词表规模。** $I(C;A)$ 受信息瓶颈上界 $H(C;A) \le n \log_2 S$ 约束（$n$ 为 token 预算、$S$ 为词表大小）。容量太小重建失真会加剧，容量太大又会把高频噪声和虚假相关也编码进去、抬升 VLA 过拟合风险；实验发现 token 预算 $n$ 对过拟合的影响明显强于词表大小 $S$。

**(c) 视觉-语言对齐 vs. 残差语法。** 用链式法则把单个 token 的总信息增益进一步拆解：

$$
I(c_k; V, L, c_{\prec k}) = \underbrace{I(c_k; V, L)}_{\text{Visual-Language Alignment}} + \underbrace{I(c_k; c_{\prec k} \mid V, L)}_{\text{Residual Grammar}}
$$

（这里 $c_{\prec k}$ 表示位置 $k$ 之前已生成的所有 token）。用大白话说：一个 token 的信息可能来自"当下的视觉/语言输入"（视觉-语言对齐），也可能来自"前面已经生成的 token"（残差语法，即 token 间的自回归依赖）。若 tokenizer 让 token 过度依赖历史 token，模型在预测时会倾向于顺着已生成序列往下编，而不是老实地看环境反馈，鲁棒性会下降。

### 2.2 基础架构与验证实验

ActionCodec 用 Perceiver 式 Transformer（仅 cross-attention + FFN）搭建 VQ-VAE 的编码器/解码器：编码器 $\mathcal{F}$ 把动作序列 $A$ 映射为连续隐变量 $Z=[z_1,\dots,z_n]$，每个 $z_k$ 在可学习 codebook $\mathcal{B}=\{e_j\}_{j=1}^S$ 中取最近邻得到 token $c_k=\arg\min_j \Vert z_k-e_j\Vert_2$，解码器 $\mathcal{G}$ 重建动作 $\hat A$，标准 VQ-VAE 目标为：

$$
\mathcal{L}_{VQ} = \Vert A-\hat A\Vert_2^2 + \Vert \text{sg}[Z]-e_c\Vert_2^2 + \Vert Z-\text{sg}[e_c]\Vert_2^2
$$

用大白话说：第一项让重建准，后两项（commitment loss + codebook loss）让连续向量与被选中的码本向量互相靠拢，稳定训练。

论文在 LIBERO-Goal 任务、SmolVLM2-256M 骨干上做受控变量实验，逐条验证上述准则：

- **重叠率实验**：合成三种 OR 水平（26%/40%/70%，通过在 VQ 目标中加入 InfoNCE 式对比损失控制潜空间聚拢程度），70% OR 的 tokenizer 仅需 500 步训练即达到 33.4% 成功率，远超朴素 VQ-VAE（8.2%）和 FAST 基线（2%）。对 `[BOS]` 隐状态做 t-SNE 可视化显示，高 OR 让 VLA 更早、更稳定地在隐空间形成清晰任务簇，缓解过拟合。
- **容量/词表实验**：扫描 $n\in\{8,16,32,64\}$、$S\in\{512,1024,2048,4096\}$，发现 $n$ 对抗过拟合能力的影响明显强于 $S$；综合重建保真度与过拟合风险，最终选定 $S=2048$、$n=16$ 作为实用配置。
- **视觉-语言对齐实验**：在 tokenizer 训练中加入 Time Contrastive Learning（TCL，拉近时间上相邻动作块的隐向量）和 CLIP 式跨模态对比损失（拉近动作隐向量与语言 embedding），发现两者都能提升 OR 与结构性；但注意力图可视化显示，CLIP 训练出的 tokenizer 使 VLA 更关注指令相关物体，而 TCL 训练出的 tokenizer 让模型偏向数据集特有的演示模式（如固定抓取点）。最终 ActionCodec 同时采用 TCL 与 CLIP 作为改进 OR 的主要手段。
- **残差语法实验**：对比 Perceiver 内部三种架构——只用 cross-attention（Independent，token 间无关联）、加双向 self-attention（SA）、加因果 self-attention（Causal）。在生成序列的特定位置注入扰动后测量重建 L1 误差，发现 Independent 架构误差最低最稳定，SA/Causal 对早期位置扰动格外敏感，会产生"时间幻觉"（过度依赖历史 token 导致的错误传播）。因此最终选择 token 间彼此独立的解耦式 tokenization，以最大化 VLA 对多模态输入的敏感度。

### 2.3 ActionCodec 的工程实现

在以上四条准则基础上，ActionCodec 加入两个实用增强：

- **Embodiment-specific soft-prompt**：Perceiver 全局参数跨 embodiment 共享，但为每个 embodiment（LIBERO、BridgeData、DROID 等）配一个可学习 soft-prompt 以捕捉不同的机械约束与控制频率，并用 Fourier embedding 编码基于控制频率与动作时长算出的时间戳。这使得控制频率各异的平台（如 Bridge-WidowX 5Hz、LIBERO-Franka 20Hz、DROID-Franka 15Hz、xArm 30Hz）能共享同一个动作 token 空间，支持跨本体动作零样本迁移与更快的新平台微调。
- **RVQ post-training**：标准 Residual Vector Quantization 虽能进一步降低重建误差，但代价是常把 OR 压到 20% 以下，严重破坏拓扑稳定性。ActionCodec 采用两阶段方案：先训练单层 VQ 模型把 OR / 感知对齐最大化，建立稳定的监督流形；然后冻结编码器与主 codebook，只额外训练残差 codebook 来消除剩余重建误差。这样能在几乎不牺牲 OR、不损害 VLA 训练效率的前提下提升重建精度，并兼容 Block-wise Autoregression（BAR）等需要多级 codebook 的解码范式。

## 三、实验结果

统一实验设置：均在 SmolVLM2 系列骨干（256M/500M/2.2B）上做无额外结构改动的全参数自回归微调，仅把词表扩展 $S$ 个专用 action token。

**(1) 与主流 tokenizer 对比（LIBERO，SmolVLM2-2.2B 骨干）。** 学习曲线上 ActionCodec 收敛速度显著领先：仅用 5K 训练步即达到 89.5% 成功率，同期次优的 FAST 基线只有 38.6%；10K 步时二者分别约为 92.7% 与 85.0%，20K 步时约为 95.5% 与 90.6%。最终不使用任何机器人预训练权重，ActionCodec（2.2B）在 LIBERO 四个任务集（Goal/Spatial/Object/Long）上平均达到 95.5% 成功率；换用 Block-wise Autoregression 解码架构后进一步提升到 97.4%，成为不依赖机器人预训练数据的 VLA 新 SOTA，甚至超过了 OpenVLA-OFT、π0、π0.5 等依赖大规模机器人预训练数据的模型。

**(2) 骨干规模消融。** 用同一 ActionCodec tokenizer，把 VLM 骨干从 2.2B 缩到 256M/500M 后，成功率依然超过"更大骨干 + 其他 tokenizer"的组合，说明动作表示的质量与 VLM 规模同样关键。

**(3) 推理效率。** 与 Binning、String 等朴素方案相比，ActionCodec 用更小的 token 预算（$n=16$）、更高的重叠率（约 72%）取得更高吞吐、更低延迟；论文明确指出 Binning/String 这类方案延迟高、吞吐低，难以满足高频闭环控制的实时性要求，而 VQ 类 baseline（VQVLA、MiniVLA、FAST）虽有改善，但吞吐和成功率仍不及 ActionCodec。

**(4) 与三种主流 VLA 解码范式的兼容性。** ActionCodec 无缝兼容 Parallel Decoding（双向注意力单次前向预测全部 token）、Knowledge Isolation（独立 300M 动作专家做 flow-matching 去噪、梯度不回传 VLM）、Block-wise Autoregression（利用 RVQ 多级 codebook 做块内并行、块间因果）三种范式，均带来正向增益；其中 BAR 变体表现最好，是不依赖机器人预训练时的 SOTA。Knowledge Isolation 变体成功率略低于原生自回归基线，但仍优于其 FAST 版本对照（π0.5 无机器人预训练版本）。

**(5) 零样本泛化（SimplerEnv-WidowX）。** 在完全域外（out-of-domain）的桌面操作任务上，ActionCodec-BAR 在该基准下取得所有对比模型（含使用了机器人预训练数据的模型）中最高的平均排名，验证其分布外鲁棒性。

**(6) 真实机器人评测。**

| 基准 | 对比设置 | 关键结果 |
|---|---|---|
| SO100-ShapeSorter（10 类形状插槽真实任务） | ActionCodec（含 co-training）vs. w/o CT vs. FAST vs. π0 | 平均 Pick 成功率 74% / 71% / 59% / 61%；平均 Place 成功率 36% / 31% / 25% / 23%；co-training 的最大增益集中在需要"失败后重新对准"的 Place 阶段（复原类长尾行为） |
| xArm-PickVeg（多任务蔬菜抓取真实任务） | ActionCodec w/ 跨本体预训练 vs. w/o PT vs. π0 vs. π0-FAST | 成功率 82.5% vs. 74.1% vs. 75.0% vs. 72.5% |

**(7) 消融研究（LIBERO 四任务集平均成功率）。**

| ID | Co-Training | Soft-Prompt | RVQ Post-training | VLM 骨干 | LIBERO Avg. |
|---|---|---|---|---|---|
| 0 | 无 | 无 | 有 | SmolVLM2 | 92.0（-3.5） |
| 1 | 有 | 无 | 有 | SmolVLM2 | 92.7（-2.8） |
| 2 | 有 | 有 | 无 | SmolVLM2 | 95.2（-0.3） |
| 3（完整版） | 有 | 有 | 有 | SmolVLM2 | 95.5 |
| 4 | 有 | 有 | 有 | Qwen2.5VL-3B | 95.1 |
| 5 | 有 | 有 | 有 | InternVL3.5-2B | 94.6 |

去掉 soft-prompt（ID1）对成功率的伤害（-2.8）明显大于去掉 RVQ post-training（ID2，-0.3），说明拓扑稳定性（OR）对 VLA 优化的重要性超过绝对重建保真度；换成 Qwen2.5VL-3B、InternVL3.5-2B 两种不同架构骨干后性能基本持平（95.1/94.6），验证 ActionCodec 对骨干架构不敏感。

## 四、局限性

- 论文在结论中承认：ActionCodec 目前只在有限的几个大规模机器人数据集（LIBERO、BridgeData、DROID 及少量真实机器人采集数据）上做预训练，尚未验证在更广泛、更狂野（in-the-wild）本体分布上的迁移能力。
- 四条设计准则的验证实验主要基于 LIBERO-Goal 单一任务集、SmolVLM2-256M 单一骨干展开，虽然后续在更大骨干和真实机器人上做了复现，但重叠率、容量、对齐、独立性四要素之间是否存在交互效应、是否在其他任务分布上同样成立，论文没有做系统的交叉验证。
- Overlap Rate 只是 Artifact Entropy $H(C\mid A)$ 的经验代理指标而非直接测量；论文附录也承认这是在"确定性 VQ 编码器数学上条件熵为零"这一理想化假设下退而求其次引入的近似。
- 残差语法实验建议放弃 token 间的自回归依赖（Independent 架构），但这与 Block-wise Autoregression 等依赖块间因果依赖的解码范式存在一定张力；论文做了兼容性验证，但未深入讨论这种设计取舍上的潜在矛盾。
- Knowledge Isolation 范式下 ActionCodec 表现略低于原生自回归基线，论文只给出了定性归因（认为 KI 更适合大规模机器人先验预训练而非小样本微调），未做更细致的分析。

## 五、评价与展望

ActionCodec 的价值不在于又提出了一种更花哨的 VQ 网络结构，而在于用信息论语言把"什么样的离散动作表示对 VLA 训练友好"讲清楚，并证明了长期被低估的 VQ 路线（相较 FAST 的频域 BPE、OpenVLA 式分箱）只要按照正确准则设计，完全可以反超——这填补了"tokenizer 只按重建误差调"与"VLA 下游表现"之间脱节的经验空白，是有意义的方法论贡献。

优点：分析框架具有一定普适性，四条准则（重叠率、容量、对齐、独立性）彼此正交、可分别验证，论文用受控变量实验逐条验证而非只看端到端指标，方法论比较扎实；覆盖面广，同时在仿真（LIBERO、SimplerEnv-WidowX）、真实机器人（SO100、xArm）、多种骨干规模与架构（SmolVLM2/Qwen2.5VL/InternVL3.5）、三种主流解码范式（PD/KI/BAR）上验证，泛化性证据较充分；RVQ post-training 的两阶段方案（先保拓扑稳定性、后补精度）是一个简洁但有效的工程技巧，直接回应了"RVQ 提高保真度但破坏 OR"这一具体矛盾。

值得关注的开放问题：其一，与同样代表离散动作路线的 FAST（频域 BPE）、VQ-VLA/MiniVLA（朴素 VQ）相比，ActionCodec 本质上是把重叠率最大化作为核心正则化目标显式引入训练，这与语音/图像领域里码本坍缩（codebook collapse）与利用率的经典 VQ-VAE 难题有相通之处，但论文没有讨论与该经典文献的联系，一个自然的后续方向是把 Artifact Entropy 的分析框架推广到更通用的离散表示学习问题上。其二，OR 作为 Artifact Entropy 的代理指标在长时序、高自由度（如双臂、灵巧手）任务上是否依然有效是一个开放问题——本文验证的任务大多是单臂 7-DoF LIBERO 或桌面操作，尚未在更高维动作空间上压力测试。其三，彻底放弃 token 间依赖（Independent 架构）虽换来了鲁棒性，但某种程度上牺牲了 VQ 表示本可以捕捉的动作内部结构（如运动学约束），这与后续仍需要块间依赖的 Block-wise Autoregression 等范式之间的关系值得进一步理论化。其四，目前的验证基本围绕 LIBERO 这一单一数据源展开设计准则的消融，若后续工作能在更多样化、更大规模的真实数据上重新验证四条准则是否依然成立，结论会更具说服力。

## 参考

- Kim, M. J. et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. CoRL 2024.
- Pertsch, K. et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747, 2025.
- Wang, Y. et al. *VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers*. ICCV 2025.
- Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- Liu, Y. et al. *Faster: Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization*. arXiv:2512.04952, 2025.
