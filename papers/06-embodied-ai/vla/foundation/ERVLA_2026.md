# ERVLA：重新审视具身思维链在通用机器人操作中的应用

> **论文**：*Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation*
>
> **作者**：Nan Sun, Yuan Zhang, Yongkun Yang, Wentao Zhao, Peiyan Li, Jun Guo, Wenxuan Song, Pengxiang Ding, Runze Suo, Yifei Su, Xin Xiao, Xinghang Li, Huaping Liu et al.（Yuan Zhang 为项目负责人，Yuan Zhang 与 Huaping Liu 为通讯作者）
>
> **机构**：清华大学、小米机器人（Xiaomi Robotics）、北京大学、中国科学院自动化研究所（CASIA）、香港科技大学（广州）、浙江大学、复旦大学、武汉大学、上海创智学院
>
> **发布时间**：2026 年 06 月（arXiv 2606.03784）
>
> **发表状态**：未录用（预印本，标注 Preprint）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.03784) | [PDF](https://arxiv.org/pdf/2606.03784)
>
> **分类标签**：`embodied chain-of-thought` `vision-language-action` `reasoning dropout` `choice policy` `diffusion transformer` `大规模CoT数据集`

---

## 一句话总结

作者构建了目前规模最大的具身链式思维（embodied CoT）语料（97.87 万条轨迹、2.263 亿样本、2592.5 小时，覆盖 AgiBot World / DROID / Fractal / BridgeData V2 / MolmoAct），通过系统消融证明"动作相关"CoT 信号（运动描述、末端点轨迹）比"高层语义"CoT（目标、子任务、抽象推理）更能提升动作学习，且显式 CoT 一旦被当作自回归动作前缀就无法可靠地随数据量 scale；据此提出 ERVLA：用推理 dropout、choice policy 分支和 KV 缓存的知识截断，把 CoT 内化为动作感知的 VLM 表征而非强制的测试时文字输出，在 LIBERO-Plus 上取得 86.9% 的平均成功率（超过 π0.5 的 85.5%），VLABench 上平均 SR/PS/IS 达 53.2/65.9/70.4，并在真实机器人的语义消歧与长时序任务上明显优于 π0.5、UniVLA、WorldVLA 与原始 ECoT。

## 一、问题与动机

近年 VLA（vision-language-action）模型通过把预训练 VLM 的视觉-语义先验迁移到动作生成上，为开放词汇、长时序操作提供了统一接口，但更强的感知与语义覆盖并不必然带来更好的动作生成——模型在复杂分布外场景中仍常常"先感知后卡壳"。这促使了"先推理后行动"范式的兴起：ECoT（Embodied Chain-of-Thought，Zawalski et al. 2024）等工作证明，通过下一词预测监督显式推理轨迹能提升鲁棒性与泛化性。但此后该设计空间迅速扩张（多种推理模态、多种推理-动作耦合架构、各类专门训练配方），却始终缺少一个系统性理解，作者将其归纳为三个悬而未决的问题：

1. **哪种具身 CoT 真正有用？** 已有方法从场景理解、子任务分解、空间定位、末端轨迹预测、未来帧预测等不同视角实例化推理，但这些选择与具体架构/训练目标紧耦合，难以判断究竟是哪类推理信号在真正帮助控制。
2. **推理应如何与策略交互？** 早期 ECoT 式设计把推理当作动作的自回归前缀（先生成长推理链，再离散化生成动作 token），简单但慢且脆——长推理链增加延迟，动作生成又强依赖前置推理的正确性，从而在推理阶段产生复合误差（compounding errors）。近期工作转而将推理蒸馏为隐变量 latent plan，或仅在训练时用推理数据训练、推理时不显式生成 CoT。这引出核心问题：推理应被视为一个可见表示、一个隐变量，还是一个重塑策略表征的训练信号？
3. **具身 CoT 能否可扩展（scale）？** 近期 VLA 预训练开始纳入更丰富的语义/推理监督，但公开的、大规模的具身推理增强机器人数据集稀缺——密集标注多种感知任务的代价高昂——因此 CoT 数据的 scaling 行为尚不明确。

围绕这三个问题，作者首先建立了迄今最大规模的具身 CoT 语料与统一的标注 pipeline，在同一设置下对比多种 CoT 信号的有效性，并识别出一种此前被忽视的失效模式——**CoT contamination（CoT 污染）**：大规模自动标注中不可避免地存在噪声标签（尤其是抖动的检测框、漂移的末端坐标），这些密集但不准确的监督会在语义相似帧上施加不一致的监督信号，损害动作学习；该问题可通过更好的标注 pipeline、对不稳定定位任务的稀疏监督，以及设计合理的推理 dropout 来缓解。

## 二、核心方法

### 2.1 具身 CoT 数据构建：分层格式 + 多视角一致标注

作者没有把 CoT 当作附着在轨迹上的自由格式解释，而是将具身推理拆解为四类结构化字段，分别对应控制中的不同角色（见 Figure 2）：

- **Understanding（理解）**：episode 级的高层目标（goal）；
- **Grounding（定位）**：当前帧可见物体的边界框（visible_objects），归一化到 $[0,1000]$ 坐标系；
- **Planning（规划）**：episode 级全局计划（plan）、当前子任务（subtask）与子任务级推理（reasoning）；
- **Acting（执行）**：动作导向的运动描述（movement_description，如"向后移动 3 厘米，然后合上夹爪"）、几何投影得到的夹爪像素位置（gripper）、未来末端点轨迹（point_trajectory）。

标注 pipeline（Appendix B，Figure 7）采用分阶段而非逐帧独立的策略：先用末端位姿的平移幅度 $\Delta p_t = \|\mathbf{p}_{t+1}-\mathbf{p}_t\|_2$、旋转变化 $\Delta R_t = \|\log(\mathbf{R}_t^\top \mathbf{R}_{t+1})\|_2$ 与夹爪状态变化 $\Delta g_t$ 生成候选子任务边界，再调用 Qwen3.5-397B 对整段轨迹一次性生成全局一致的 goal / plan / reasoning（避免逐帧重复生成导致的漂移）；动作导向字段（movement_description）由未来动作 chunk $\mathbf{A}_{t:t+K}$ 的位移与夹爪变化阈值化后转成短语；夹爪位置与未来点轨迹对有标定外参的静态相机采用几何投影而非检测器定位（因为检测器容易受遮挡、运动模糊影响产生较大空间抖动）；物体框用 LLMDet 生成，且仅在关键帧/固定间隔上稀疏标注，避免逐帧独立检测框在近似观测间强烈抖动带来的不一致监督。数据集在物体框、夹爪位置、未来轨迹三类空间字段上都**按相机视角（base/front/wrist）分别存储**，支持多视角一致的空间推理，并对单臂/双臂轨迹分别保留左右臂的状态与轨迹。

最终语料（Table 5）汇总 AgiBot World、DROID、Fractal、BridgeData V2、MolmoAct 五个公开数据集，规模为 **978,743 条轨迹、226.3M 样本、2592.5 小时**，是当前已知规模最大、且首批同时覆盖多视角与双臂设置的具身 CoT 数据集之一。

### 2.2 关键实证发现（先于架构设计的系统性消融）

在提出 ERVLA 之前，作者先用统一骨干（Qwen3-VL-4B）和固定的自回归 CoT+FAST 动作 token 接口，逐字段消融 CoT 信号，得到三条支撑架构设计的结论：

**发现一：动作相关字段比高层理解字段更直接有用。** 孤立的高层字段（goal、planning、subtask、reasoning）单独使用时收效甚微甚至有害（goal 单独使用使 VLABench 均分下降 1.2 点），而动作相关字段效果明显更强：movement_description 单独带来 +4.1，point trajectory 单独带来 +4.8；组合信号进一步提升，Movement+Reasoning 达 +5.2，Subtask+Movement+Point trajectory 达 +7.4。这表明有效的具身 CoT 必须把语义理解与可执行运动连接起来，而不只是复述任务意图。

**发现二：预训练只在标签可靠时才有帮助，reasoning dropout 缓解"CoT 污染"。** 在 Bridge 数据上做 CoT 预训练后，坐标类字段中不可靠的部分（gripper、bounding box，来自检测器而非仿真器 ground truth）反而成为最有害的信号（分别导致 -5.6、-6.1 的下降），因为大规模检测器生成的标签存在噪声，会在语义相似的相邻帧上施加不一致监督。引入推理 dropout 后，这两项的损害被大幅收窄至 -0.8、-1.0，同时保留了 point trajectory 等可靠动作字段的收益（从 +1.4 提升到 +3.0）。

**发现三：显式 CoT 在自回归动作建模下不能可靠 scale。** 固定使用 Bridge-only 预训练的完整 CoT 监督时收益尚可，但随着 CoT 预训练数据从 Bridge 逐步扩展到 Fractal、MolmoAct、Droid（Table 2），VLABench 上性能反而逐渐下滑——四数据集混合设置下，In-dist. 下降 3.6、Cross-category 下降 3.0、Texture 下降 3.4。作者将此归因于自回归动作解码的脆弱性：推理链的长度与质量参差不齐、定位噪声会直接污染动作前缀，CoT 的错误会连带传导到后续动作 token。这条发现是 ERVLA 放弃"CoT 作为自回归前缀"设计的直接动因。

此外，作者还发现**具身 CoT 让更强的 VLM 骨干更可靠地迁移为更强的 VLA**：在无 CoT 监督时，骨干能力（PaliGemma-2-3B 到 Qwen3-VL-8B 等九种骨干）与下游 VLA 性能相关性弱，加入显式 CoT 后二者相关性显著增强，Qwen3-VL 系列在多条赛道上领先。

### 2.3 ERVLA 架构：把 CoT 变成塑造表征的训练信号而非解码负担

基于以上发现，ERVLA 采用"混合 Transformer"架构（VLM 推理骨干 + DiT 扩散动作头），核心设计目标是让推理监督重塑 VLM 内部表征，同时让动作生成不依赖脆弱的自回归 CoT 解码。

**(1) 推理模型。** 以 Qwen3-VL-4B 为骨干，输入图像 $I$、指令 $x$、可选 CoT 文本 $c$、状态 $s$、$N$ 个动作查询 token $\{a_i\}$ 与一个打分查询 token $a_{\text{score}}$，输出隐藏状态与逐层 KV 缓存：

$$\mathbf{H}^{\text{vlm}}, \{(\mathbf{K}_\ell^{\text{vlm}}, \mathbf{V}_\ell^{\text{vlm}})\}_{\ell=1}^{L} = f_{\text{vlm}}(I, x, c, s, \{a_i\}, a_{\text{score}})$$

**用大白话说**：这一步就是把"看图 + 听指令 + （可选）读推理链"统一编码成一串隐藏状态和注意力缓存，其中专门留出几个可训练的"占位 token"（对应状态、候选动作、打分）来承接后续的动作解码。

**(2) Choice policy 分支（辅助动作查询）。** 借鉴 choice policies 的思路，模型不是只回归一个动作 chunk，而是预测 $N$ 个候选动作 chunk 及其对应打分：

$$\hat{\mathbf{a}}_t^{(n)} = g_{\text{act}}(\mathbf{H}_a)_{t,n}, \quad t=1,\ldots,T,\ n=1,\ldots,N, \qquad \hat{\mathbf{r}} = g_{\text{score}}(\mathbf{H}_s)$$

这一支路作为辅助的动作回归损失，引导 VLM 骨干不仅生成文字推理，还要理解与动作相关的语义，起到稳定训练、加速收敛、把高层推理与低层动作生成对齐的"锚点"作用；最终动作仍由下面的扩散头生成，choice 分支只提供候选级别的判别信号。

**(3) 扩散动作头（DiT + flow matching）+ 知识截断。** 最终连续动作由 DiT 通过 flow matching 生成，其条件不是池化后的 VLM 特征，而是 VLM 逐层 KV 缓存本身：

$$\hat{\mathbf{A}}^{(n)} = [\hat{\mathbf{a}}_1^{(n)}, \ldots, \hat{\mathbf{a}}_T^{(n)}] \in \mathbb{R}^{T\times D}, \qquad \hat{\mathbf{v}}_\theta = f_{\text{dit}}\big(\mathbf{z}_\tau, \tau, s \mid \{(\mathbf{K}_\ell^{\text{vlm}}, \mathbf{V}_\ell^{\text{vlm}})\}\big)$$

但 DiT 并不直接看到完整 KV 缓存，而是只允许其关注**语义前缀**部分，显式排除状态/动作查询等控制 token（"知识截断"，knowledge truncation）：

$$\text{Attn}\big(\mathbf{Q}, [\mathbf{K}_\ell^{\text{KT}}; \mathbf{K}_\ell^{\text{dit}}], [\mathbf{V}_\ell^{\text{KT}}; \mathbf{V}_\ell^{\text{dit}}]\big), \qquad \{(\mathbf{K}_\ell^{\text{KT}}, \mathbf{V}_\ell^{\text{KT}})\} = \text{SlicePrefix}\big(\{(\mathbf{K}_\ell^{\text{vlm}}, \mathbf{V}_\ell^{\text{vlm}})\}, m_{\text{cond}}\big)$$

**用大白话说**：如果让扩散头直接看到附加在末尾的状态/动作占位 token，它很容易"抄近道"（shortcut）——直接从这些人为拼接的控制 token 里读取捷径信息，而不是真正利用语义推理表征；知识截断相当于把 DiT 的视野限制在"图像+指令+CoT"这段干净的语义记忆上，逼它老老实实地从语义表征里提炼动作相关信息。

**(4) 训练目标与推理 dropout。** 与"知识隔离"（knowledge-insulated，阻断动作专家梯度回传到 VLM 骨干）不同，ERVLA 允许 flow-matching 损失端到端反传进 VLM 骨干，使推理与动作系统全程可训练。总损失为四项联合优化：

$$\mathcal{L} = \lambda_{\text{vlm}}\mathcal{L}_{\text{vlm}} + \lambda_{\text{flow}}\mathcal{L}_{\text{flow}} + \lambda_{\text{choice}}\mathcal{L}_{\text{choice}} + \lambda_{\text{score}}\mathcal{L}_{\text{score}}$$

其中 $\mathcal{L}_{\text{vlm}}$ 是对 CoT 文本的 token 级交叉熵，$\mathcal{L}_{\text{flow}}$ 是连续动作的 rectified flow 损失；choice 分支取 $N$ 个候选中与真值最接近者作为监督目标（best-of-N）：

$$\mathcal{L}_{\text{choice}} = \frac{1}{B}\sum_{b=1}^{B} \min_n d_b^{(n)}, \qquad d_b^{(n)} = \frac{1}{TD}\left\|\hat{\mathbf{A}}_b^{(n)} - \mathbf{A}_b^{*}\right\|_1$$

打分分支则回归各候选的误差、并对目标做 stop-gradient：

$$\mathcal{L}_{\text{score}} = \frac{1}{B}\sum_{b=1}^{B} \left\|\hat{\mathbf{r}}_b - \text{sg}\big([d_b^{(1)}, \ldots, d_b^{(N)}]\big)\right\|_2^2$$

训练时以概率 $p_{\text{cot}}=0.5$ 对每条样本随机切换"带 CoT"或"不带 CoT"两种模式（reasoning dropout，样本要么保留完整 CoT 文本，要么把该段替换为空的推理占位符，同时动作监督不变）。这使显式 CoT 变成一种可选的训练条件而非强制的推理时轨迹，既缓解了噪声标签造成的 CoT 污染，又鼓励推理信息被内化进骨干隐藏状态与"缓存条件化"的动作接口中，从而天然支持测试时的部分/稀疏/完全省略 CoT 推理。

**训练规模**：骨干 Qwen3-VL-4B，动作视界 30，动作/状态维度 padding 到 60 维（兼容单臂/双臂），DiT 36 层，候选数 $N=5$，reasoning dropout $p_{\text{cot}}=0.5$，启用 token packing（最大打包长度 17,600），训练 120,000 步，batch size 64，学习率 $5\times10^{-5}$，DeepSpeed + bfloat16 混合精度。预训练混合采样权重（Table 10）：AgiBot 0.518、DROID 0.180、Fractal 0.120、Bridge 0.100、MolmoAct 0.082。

## 三、实验结果

评测覆盖 LIBERO-Plus（LIBERO 的相机/状态/语言/背景/布局扰动鲁棒性扩展）、VLABench（更难的跨类别迁移、常识推理、语义指令跟随、纹理分布外基准）与真机四档难度任务。

### 3.1 仿真基准

| 基准 | ERVLA | 最强公开基线 |
|---|---|---|
| LIBERO-Plus（均分） | **86.9%** | π0.5 85.5% |
| VLABench 均分 SR / PS / IS | **53.2 / 65.9 / 70.4** | π0.5 SR 约 48.1（弱于 ERVLA） |

VLABench 分赛道 SR（ERVLA）：In-distribution 69.7、Cross Category 47.0、Commonsense 44.0、Instruction 58.0、Texture 47.4。ERVLA 同时超越了以 latent 接口著称的 UniVLA 与以视觉预测为推理-动作接口的 WorldVLA。

**架构消融（Table 3/4）**：去掉 choice policy 分支（"No Choice (End-to-End)"）或去掉知识截断（"Choice + No Knowledge Truncation"，让 DiT 直接读取完整 KV 缓存包括控制 token）都会在两个基准上造成明显下降；作者将其解读为——choice 分支把候选级别的判别信号注入 VLM，帮助塑造动作感知表征，而知识截断则防止 DiT 走"读控制 token 抄近道"的捷径，二者缺一不可。

**CoT scaling 对比（Figure 5）**：随着具身 CoT 预训练数据从约 35M 增至完整 226.3M 样本，ERVLA（choice 分支）在 LIBERO-Plus 与 VLABench 上均稳定提升；而自回归 CoT+FAST 动作 token 方案（如 ECoT、Emma-X 的设计思路）以及"隔离式"VLM+DiT（类似 ThinkAct 的推理/动作分离）都表现出更弱或饱和的 scaling 曲线——证明"具身 CoT 是否有用"不是瓶颈，"如何让它指导动作学习而不成为解码负担"才是瓶颈。

### 3.2 真实机器人实验

在配备第三人称相机与腕部相机的物理机器人平台上，围绕"物品放入抽屉"和"清理桌面杂物"两类任务，设计 Basic / Distractors / Semantic / Long-horizon 四档难度（每档 5 条指令，共 20 个任务，每任务 5 次试验，共 100 次真机 rollout/方法），对比 ECoT、WorldVLA、UniVLA、π0.5 与 ERVLA：

| 方法 | Basic (SR/PS) | Distractors (SR/PS) | Semantic (SR/PS) | Long-horizon (SR/PS) | 均值 (SR/PS) |
|---|---|---|---|---|---|
| ECoT | 60 / 68 | 18 / 30 | 10 / 25 | 6 / 18 | 24 / 35 |
| WorldVLA | 78 / 84 | 28 / 42 | 18 / 35 | 12 / 28 | 34 / 47 |
| UniVLA | 76 / 82 | 31 / 45 | 22 / 38 | 18 / 34 | 37 / 50 |
| π0.5 | **97 / 98** | **45 / 57** | 31 / 45 | 35 / 38 | 53 / 60 |
| **ERVLA** | 96 / 97 | 44 / 58 | **42 / 58** | **38 / 55** | **55 / 67** |

清洁场景（Basic）下 π0.5 与 ERVLA 表现相近（97/98 vs 96/97），说明强连续动作策略本身已能很好处理简单低层操作；但在需要消歧、忽略干扰物、维持长时序任务意图的 Semantic 与 Long-horizon 档，ERVLA 优势明显扩大：Semantic 档 SR 从 π0.5 的 31 提升到 42、PS 从 45 提升到 58；Long-horizon 档 PS 从 38 提升到 55。作者将其归因于 ERVLA 把子任务与运动级别的结构内化进了动作相关表征，而非仅依赖低层动作生成能力。ECoT 因自回归推理-动作解码的延迟与脆弱性表现最差；WorldVLA、UniVLA 受限于视觉预测/latent 推理未能直接解决细粒度语义定位问题。

## 四、局限性

作者在论文正文（Sec. 4）与专门的 Limitations 附录中坦承：

1. **推理监督质量是硬约束。** ERVLA 的推理预训练收益上限受标注"基底"质量约束：语言字段通常稳健但可能欠具体，密集定位字段（物体框、夹爪位置）更直接可执行但更易受检测器误差、标定偏差、遮挡影响。reasoning dropout 与知识截断只能缓解不完美监督带来的损害，并不能替代更好的标注质量，也没有把"不确定性"本身建模进标注。
2. **推理被处理为离线预训练信号，缺乏测试时动态刷新机制。** ERVLA 的高效推理设置在推理时不强制解码 CoT（即默认走 no-CoT 高效通路），这提升了推理效率，但也意味着长时序操作中可能需要的"记住已完成子任务""从失败尝试中恢复""在场景与指令不匹配时寻求澄清"等能力目前未被显式建模。
3. 数据 pipeline 依赖大模型（Qwen3.5-397B）与开集检测器（LLMDet）自动标注，标注管线本身的系统性偏差未被单独量化；真机评测样本量（每方法 100 次 rollout）相对有限，跨具身形态（如人形/双臂）的真机验证尚未覆盖。
4. 论文承认"结果不应被简单理解为‘更多显式推理就更好’"——反而是"何时用语言表达、何时内化、何时直接行动"仍是开放问题。

## 五、评价与展望

**贡献与优点**：这项工作的核心价值不在于又一个"推理增强 VLA"，而在于用受控消融把此前分散在 ECoT、Emma-X、CoT-VLA、ThinkAct、UniVLA 等工作中的经验碎片系统化——明确回答了"哪类 CoT 信号有用""推理该不该做自回归前缀""CoT 数据该怎么 scale"三个问题，并且给出了目前公开可比的最大规模具身 CoT 数据集与标注 pipeline，这本身对社区是有独立价值的基础设施贡献。识别"CoT contamination"这一失效模式并给出量化证据（Table 1/2 中密集坐标字段在预训练后反而变负收益）是一个有意思且此前较少被明确讨论的观察，比单纯堆叠更多推理监督更有指导意义。

**与相关工作的关系**：ERVLA 的设计哲学介于两派之间——一派是 ECoT/Emma-X 式的"推理即自回归前缀"，另一派是 ThinkAct/Fast-ThinkAct/UniVLA 式的"推理蒸馏为隐变量、与动作头解耦"。前者简单但脆（本文证明其不能 scale），后者稳健但通过阻断梯度反传弱化了动作反馈对推理表征的塑造。ERVLA 试图两者兼得：保留 choice policy 分支和知识截断维持推理-动作的端到端反馈通路，同时用 reasoning dropout 避免测试时对显式文本的强依赖。这一思路与知识隔离（knowledge-insulated VLA）路线形成直接的方法论对照，论文中的"No Choice + Knowledge Insulation"消融也正是对标此类方法，实验显示其弱于完整 ERVLA。

**开放问题与可能的改进方向**：(1) 标注可靠性目前是二元的（保留/dropout），一个更精细的方向是让模型学习按字段置信度加权监督，即论文结尾建议的"让不确定性成为标注的一部分"；(2) 该工作把 CoT 完全处理为离线预训练信号，如何在长时序、可能失败/需要澄清的真实部署场景中做**测试时自适应的稀疏推理刷新**（而非非 0 即 1 的"要不要 CoT"）仍待探索；(3) 真机实验固定在两类桌面任务族、单一机器人平台上，跨本体（尤其双臂/人形）上的 CoT 表征迁移能力尚未验证，而作者构建的数据集本身已包含 AgiBot World 的双臂数据，后续工作有条件在这一维度上补充证据；(4) choice policy 的候选数 $N=5$、知识截断的边界 $m_{\text{cond}}$ 等超参数的敏感性分析在正文中着墨不多，其对不同任务复杂度的鲁棒性仍是可深挖的方向。

## 参考

- Zawalski et al. *Robotic control via embodied chain-of-thought reasoning* (ECoT). arXiv:2407.08693, 2024.
- Sun et al. *Emma-X: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning*. arXiv:2412.11974, 2024.
- Physical Intelligence et al. *π0.5: a vision-language-action model with open-world generalization*. arXiv:2504.16054, 2025.
- Bu et al. *UniVLA: Learning to act anywhere with task-centric latent actions*. arXiv:2505.06111, 2025.
- Chen et al. *Training strategies for efficient embodied reasoning*（reasoning dropout 的直接来源）. arXiv:2505.08243, 2025.
