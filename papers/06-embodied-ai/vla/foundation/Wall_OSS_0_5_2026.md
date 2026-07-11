# Wall-OSS-0.5：梯度桥接协同训练下的可直接部署VLA预训练技术报告

> **论文**：*Wall-OSS-0.5 Technical Report*
>
> **作者**：Ryan Yu, Pushi Zhang, Starrick Liu, Brae Liu, Miracle Kang, Shalfun Li et al.（X Square Robot Team）
>
> **机构**：X Square Robot
>
> **发布时间**：2026 年 06 月（arXiv 2605.30877）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.30877) | [PDF](https://arxiv.org/pdf/2605.30877)
>
> **分类标签**：`VLA预训练` `梯度桥接协同训练` `Mixture-of-Transformers` `RVQ动作分词器` `flow matching` `零样本操作评测`

---

## 一句话总结

Wall-OSS-0.5 用"梯度桥接协同训练"（离散动作token交叉熵作为梯度桥、多模态交叉熵作为泛化锚点、连续flow matching作为部署接口三者联合优化）证明VLA预训练本身即可产生可直接在真机上评测的操作行为：4B参数模型在17任务零样本套件上均分51.1%，微调后在15任务真机基准上以60.5%的平均任务进度超越π0.5（43.0%）17.5个百分点，同时嵌入式视觉-语言定位能力提升21.8个百分点而不牺牲整体的具身理解。

## 一、问题与动机

大规模VLA预训练已被广泛采纳为机器人策略的基础范式，但现有工作报告的能力几乎总是在任务特定微调之后测得的，这使一个基础性问题始终悬而未决：**VLA预训练本身究竟产生了可执行的机器人行为，还是仅仅提供了一个更好的下游策略学习初始化？**

作者将这一目标称为"面向部署的VLA预训练"（deployment-oriented VLA pretraining），并据此提出三点要求：预训练checkpoint必须在无任何任务特定微调的情况下就能执行有用的操作技能；必须保留足够的VLM衍生视觉-语言能力以维持指令对齐（instruction-grounded）；并且要为下游适应提供更高样本效率的先验。

这一问题的技术张力在于：连续flow matching是VLA的自然执行接口（直接建模未量化的机器人动作），但单独使用时它对预训练VLM主干的梯度更新较弱；而离散动作token的下一token交叉熵是与VLM训练接口原生匹配的强信号，能强烈塑造主干，但解码出的离散动作对精确控制而言又过于粗糙。冻结或截断梯度可以保留VLM先验，但代价是阻止精确的动作目标去塑造这个大型预训练主干。

## 二、核心方法

### 2.1 梯度桥接协同训练（Gradient-Bridged Co-training）

Wall-OSS-0.5 在单一训练阶段联合优化三个互补目标：离散动作token交叉熵作为**梯度桥**（gradient bridge），把强、VLM原生的动作信号灌入主干；多模态交叉熵作为**泛化锚点**（generalization anchor），维持指令跟随、视觉定位与具身场景理解；连续flow matching则训练**部署时使用** 的动作生成器。三者的复合目标为：

$$\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_{\text{act}} \cdot \mathcal{L}_{\text{act-CE}} + \lambda_{\text{mm}} \cdot \mathcal{L}_{\text{mm-CE}}$$

其中 $\lambda_{\text{act}} = \lambda_{\text{mm}} = 0.01$，动作数据与多模态数据以9:1的批次比例混合。**用大白话说**：flow matching损失在动作空间监督下天然比两个交叉熵损失小两个数量级，因此用统一的小权重把交叉熵项压到与flow损失可比的量级，防止"语言式"的token预测目标压过动作学习；训练结束后推理只走连续flow matching通路，离散通路的唯一作用是在训练阶段充当梯度桥。梯度分析显示：训练早期之后，flow matching对主干更新的贡献稳定在约5%的小而持续的份额上，主导性的主干更新来自两个交叉熵损失——这正是消融实验（见下）验证的核心动因。

### 2.2 Mixture-of-Transformers（MoT）路由骨架

模型从 Qwen2.5-VL-3B-Instruct 初始化，并扩展出 Mixture-of-Transformers 骨架，参数总量超过4B：原3B VLM保留为 *VL Expert*，新增的 *Action Expert*（连同连续动作头中的动作投影）提供额外的动作生成容量。视觉、语言、本体感知与离散动作四类token流经VL Expert，噪声连续动作token经由Action Expert路由并用flow matching训练。两路共享同一序列级注意力上下文（joint attention），使Action Expert能关注视觉与语言信息；但注意力掩码使离散与连续动作token在前向传播中互不可见，从而让两条动作通路可被独立训练与评估——这是一种"路由分解"而非"梯度隔离"：梯度仍可从flow matching端到端流向VL Expert。

### 2.3 Vision-Aligned RVQ 动作分词器

用一个学习得到的**视觉对齐残差向量量化**（Vision-Aligned RVQ）动作分词器替换基于规则的 FAST 分词器。它在delta-action空间中运行，采用Encoder–RVQ–Decoder结构：编码器通过时序交叉注意力压缩以观测为条件的动作块；RVQ码本在早期层级捕获粗略运动结构，在后续层级捕获精细残差修正；解码器以观测状态为条件重建动作序列。除重建目标外，还引入两个辅助目标共同塑造token空间：视觉-动作对齐（将动作latent拉向VLM视觉特征）与DCT域的下一帧预测（抑制高频抖动、鼓励对动作后果的预测）。**用大白话说**：这个分词器不是单纯的"动作压缩器"，而是要让离散token本身携带语义，使其成为VLM主干可读的训练接口。

### 2.4 Action-Space Supervision（动作空间监督）

Flow matching从带噪动作块出发学习一个把噪声搬运到干净动作的速度场，采用线性高斯概率路径：

$$\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t + (1-\tau)\epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

沿用 π0 的做法，将时间步采样偏向高噪声区间：

$$u \sim \text{Beta}(1.5, 1), \quad \tau = s(1-u), \quad s = 0.999$$

网络输出仍是速度预测，但损失被定义在还原出的动作上而非速度场上：

$$\hat{A} = A^{\tau} + (1-\tau) \cdot f_\theta(A^{\tau}, \tau)$$

$$\mathcal{L}_A = \mathbb{E}_{\tau,\epsilon}\left[\|\hat{A} - A\|^2\right]$$

该形式等价于速度空间中一个 $(1-\tau)^2$ 加权的损失：

$$\mathcal{L}_A = \mathbb{E}_{\tau,\epsilon}\left[(1-\tau)^2\|f_\theta(A^{\tau},\tau) - (A-\epsilon)\|^2\right]$$

**用大白话说**：机器人动作序列低维且平滑，其任务相关结构主要体现在低频的轨迹形状而非高频细节；这一加权把监督重心放到高噪声、决定轨迹全局形状的时间步上，让模型优先学对"大致往哪走"，再逐步细化。消融显示该设计在LIBERO仿真上收敛更快、峰值更高、训练更稳定。

### 2.5 动作接口与优化细节

模型遵循VLM式对话序列输入格式（系统prompt给出具身信息，观测图像token+指令+本体感知token，输出离散动作AR token与连续动作flow查询token交替排列）。动作空间为26维：每条机械臂的相对3D位置+相对6D旋转+1D夹爪状态（单臂10维，双臂20维），加上3D移动底盘速度、1D升降高度、2D云台/头部动作（共6维）。采用相对动作表示和6D旋转（而非欧拉角/四元数）以避免SO(3)的不连续性和奇异性问题。

优化上，Muon被应用于每个专家的2D参数，AdamW处理视觉embedding和LM head；Muon对动量做正交化后再更新，产生对梯度量级不变的谱归一化更新，这对VL Expert和Action Expert梯度尺度显著不同的场景至关重要。为使Muon在大规模分布式训练下实用，作者实现了DMuon：按Longest-Processing-Time-first做矩阵级负载均衡、异步post-step广播，将Muon引入的额外优化器开销从朴素实现下约2倍的前后向时间压缩到约0.02倍（约100倍降低）。预训练使用有效全局batch size 8192、bf16混合精度、梯度裁剪1.0、余弦学习率调度、峰值学习率1e-4；微调学习率5e-5，所有模块可训练。

推理侧针对高分辨率VLA构建了部署级推理栈：将去噪步骤整体捕获为单个CUDA Graph以消除CPU派发瓶颈，并将RoPE/RMSNorm等零散算子融合为整体CUDA kernel以消除中间张量的显存搬运。在单张RTX 5090上，三视角输入224×224分辨率下达约21Hz，448×448分辨率下达约15Hz（去噪步数T=10），相较PyTorch eager模式基线取得4倍端到端加速。

### 2.6 数据配方

预训练数据一次性混合三个来源：高质量自采数据（自研桌面双臂系统+移动操作平台+一个具身无关的采集设备XRZero-G0）、经统一动作schema和跨源预处理后的10个开源多具身数据集（AgiBotWorld Beta占比24.6%、RoboMIND v2.0占21.7%、Fractal占12.3%、RealOmin占10.9%、DROID占10.7%、RoboCOIN占7.3%、RoboMIND v1占3.8%、RoboChallenge占3.6%、BRIDGE v2占2.7%、Galaxea Open-World占2.4%），以及一个约9000万样本的多模态语料（7800万开源样本+1200万从动作轨迹自动构造的"具身桥接"样本）。多源数据按组（数据源×任务）做幂律采样（权重 $w_i = n_i^{p}$，$p=0.5$）以缓解长尾不均衡，并对大组设置容量上限做迭代再分配；一个训练epoch覆盖超过100万条轨迹（约60%自采、40%开源）。具身桥接数据沿物体、空间、场景、任务四个理解层次组织，把动作预训练语料转成可用于多模态监督的问答/定位/轨迹预测样本，作为"泛化锚点"贴近实际执行场景。

## 三、实验结果

### 3.1 预训练模型零样本真机评测（无任何任务特定微调）

17个任务（12个seen + 5个held-out unseen），覆盖语义理解、刚性物体操作、可变形物体操作、精细操作、长时程多步操作五个维度，按预定义分步评分标准打分（task progress，满分100，每任务10条轨迹取均值）。

| 训练步数 | 50k | 100k | 200k | 300k | 350k | 400k |
|---|---|---|---|---|---|---|
| Seen均值（12任务） | 26.1 | 31.7 | 40.1 | 40.4 | 48.1 | **50.0** |
| Unseen均值（5任务） | 24.2 | 41.0 | 38.8 | 34.8 | 47.6 | **53.6** |
| 总体均值（17任务） | 25.5 | 34.5 | 39.8 | 38.7 | 47.9 | **51.1** |

400k checkpoint下的代表性任务（部分）：

| 任务 | 类别 | Seen/Unseen | Task progress |
|---|---|---|---|
| Block Sorting | 语义理解 | Seen | 100% |
| Fruit Sorting | 语义理解 | Seen | 96% |
| Ring Stacking | 刚性操作 | Seen | 86% |
| Rope Tightening | 可变形操作 | **Unseen** | 82% |
| Cup Grasping | 刚性操作 | Seen | 64% |
| Bean Pouring | 可变形操作 | **Unseen** | 60% |
| Towel Folding | 可变形操作 | Seen | 10%（零样本能力边界之外） |
| Table Setting | 长时程 | Seen | 9% |
| Charger Plugging | 精细操作 | Seen | 9% |

值得注意的是，held-out的可变形操作任务 Rope Tightening 达到82%，说明这并非纯粹的任务模板记忆，而是具备一定的可迁移操作能力。

### 3.2 真机微调基准对比

在15个真机任务（10操作+5推理型任务）、约500条演示轨迹/任务、统一微调与评测协议下，与两个官方预训练权重初始化的基线对比：

| 模型 | 操作子集(10) | 推理子集(5) | 总体(15) |
|---|---|---|---|
| **Wall-OSS-0.5** | **61.1** | 59.3 | **60.5** |
| π0.5 | 35.0 | **58.9** | 43.0 |
| DreamZero（WAM） | 33.7 | 32.7 | 33.4 |

Wall-OSS-0.5在15个任务中10个取得最高分，操作子集领先π0.5达26.1个百分点（如Color Block Sorting 96% vs 42%，Drawer Organization 52% vs 7%，Spoon-in-Bowl 80% vs 43%），但在Pencil Case Packing（双臂精细可变形操作）上仅18.5%，是主要短板。

### 3.3 多任务微调扩展性（5→10→19任务）

同一预训练checkpoint下，随微调任务集从5扩展到10再到19，共享任务的表现不降反升：5个简单共享任务均值从73.96%升至83.75%（+9.8pp），10个共享任务（5简单+5复杂/推理）均值从59.98%升至64.78%（+4.8pp），19任务配置新增的9个跨背景/跨具身分布任务也达到65.59%。说明扩大微调任务规模能通过补齐细粒度能力（动作原语、语言表达、状态变化分布的覆盖）反哺原有任务，而非稀释表现。

### 3.4 具身多模态理解（动作训练是否侵蚀VL能力）

以Qwen2.5-VL-3B为基线对比：

| 维度 | 基准 | Qwen2.5-VL-3B | Wall-OSS-0.5 | 变化 |
|---|---|---|---|---|
| 通用VQA | RealWorld VQA | 59.2% | 44.2% | -15.0 |
| 通用VQA | ERQA | 38.3% | 32.8% | -5.5 |
| 具身场景理解 | EO-Bench | 20.8% | 24.7% | +3.9 |
| 具身定位（内部构建） | Embodied Grounding | 9.0% | 30.8% | **+21.8** |
| 放置推理 | Where2Place | 4.0% | 15.0% | +11.0 |

结果呈现出一种"专业化"效应：性能从开放域VQA向与机器人执行更相关的具身感知信号迁移。作者将此归因于1200万具身桥接样本的引入——若无这部分数据，早期实验中强动作token目标会显著压低对常见机器人观测分布的多模态得分。

### 3.5 关键消融

- **协同训练策略对比**（从零训练70k步，5个真机消融任务）：co-training均值57.0%，优于flow-only（36.6%）、stop-gradient（31.9%）和"先stop-gradient再切co-training"两阶段方案（49.6%），去掉三种信号中的任意一种会使真机表现下降7.4–25.1个百分点。
- **Action-Space Supervision vs 速度空间损失**（LIBERO仿真）：动作空间损失峰值成功率96.5%（25k步），超过速度空间损失峰值6.2pp；20k步时动作空间损失已达95.8%，而速度空间损失35k步内未突破90.3%，且在训练中段（20k步）出现明显的稳定性波动。
- **Vision-Aligned RVQ vs FAST 分词器**（相同协同训练设置，从VLM权重开始）：VQA准确率75.7%→77.5%（+1.8pp），4个真机任务的平均task progress从29.3%大幅提升到48.1%（+18.9pp）。

## 四、局限性

论文第7节明确列出四点局限：（1）梯度桥机制目前仅在3B级VLM主干上验证，扩展到更大主干可能显著改变三种训练信号之间的相对几何关系与相互作用强度；（2）当前模型只接受单帧图像输入，这很可能限制了需要时序记忆和持久状态追踪的长时程任务的零样本表现；（3）Vision-Aligned RVQ动作分词器及其训练流程目前绑定于固定的26维动作表示，限制了向灵巧手等更高自由度具身形态的直接迁移；（4）任务评测仍依赖人工设计的分步评分标准，当前真机基准尚未覆盖多机协作、长时间部署或更开放世界的交互场景。

## 五、评价与展望

**贡献与优点**：本文的核心学术贡献在于把"VLA预训练本身能否直接产生可执行行为"这一此前被默认跳过的问题，转化为可测量、可复现的实证命题，并给出了一套具体、消融充分的训练配方（梯度桥、MoT路由、语义化动作分词器、动作空间监督）来支撑这一命题。相较最接近的 π0.5（同样协同训练离散FAST通路与flow matching通路，但对主干采用stop-gradient式知识隔离），本文的关键区别在于坚持端到端梯度流动并将离散动作token明确定位为"梯度桥"而非并行输出头，这一设计选择由消融直接验证（stop-gradient显著弱于co-training）。工程层面，DMuon将分布式Muon优化器开销压缩约100倍、CUDA Graph+算子融合带来4倍推理加速，这些是具备复现价值的系统贡献，且模型、代码、评测工具均开源。

**值得商榷之处**：其一，"零样本"评测中seen/unseen任务的难度并未严格匹配，作者自己也承认这一点，使得"unseen均值反超seen均值"的解读需要谨慎；其二，通用VQA能力的明显下降（-15.0、-5.5）虽被解释为"专业化"是合理的trade-off，但缺乏进一步探究——是否存在能同时保留通用VQA与具身定位能力的混合比例或课程设计尚未给出定量分析；其三，微调基准中DreamZero作为世界-动作模型（WAM）范式代表落后较多（33.4% vs 60.5%），但论文未深入讨论这一范式差距的成因，是数据量/微调协议不利于WAM，还是WAM范式本身在小样本微调下天然弱势，值得后续工作澄清；其四，动作空间监督（Action-Space Supervision）的验证仅在LIBERO仿真中做了受控实验，未见与真机结果的直接消融对照。

**开放问题与可能方向**：论文自身指出的三个方向（更大VLM主干、时序观测与分层规划、更通用的动作表示以支持灵巧手等高自由度具身）都切中要害。此外，一个尚未被充分讨论的问题是：梯度桥机制中离散动作token的"语义容量"（由Vision-Aligned RVQ的视觉对齐目标决定）与flow matching生成质量之间的耦合强度如何随分词器容量（码本大小、RVQ层级数）变化，这可能是理解为何"训练时使用离散token、部署时只用连续通路"依然有效的关键，也是与CogACT、HPT等"认知-动作解耦"路线进一步比较的切入点。

## 参考

1. Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054, 2025. —— 最接近的基线与对比对象，stop-gradient式知识隔离协同训练的代表。
2. Bai, S. et al. *Qwen2.5-VL Technical Report*. arXiv:2502.13923, 2025. —— Wall-OSS-0.5 的3B VLM初始化主干。
3. Pertsch, K. et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747, 2025. —— 被Vision-Aligned RVQ替换的规则式动作分词器基线。
4. Liang, W. et al. *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models*. arXiv:2411.04996, 2024. —— MoT路由骨架的架构来源。
5. Lipman, Y. et al. *Flow Matching for Generative Modeling*. arXiv:2210.02747, 2022. —— 连续动作生成（部署时执行接口）所依赖的flow matching基础理论。
