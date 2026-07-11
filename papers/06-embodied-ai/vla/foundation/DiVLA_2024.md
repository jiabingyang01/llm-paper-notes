# DiVLA：融合自回归推理与扩散策略的可泛化、可解释机器人基础模型

> **论文**：*Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning*
>
> **作者**：Junjie Wen, Yichen Zhu, Minjie Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Xiaoyu Liu, Chaomin Shen, Yaxin Peng, Feifei Feng
>
> **机构**：美的集团（Midea Group）、华东师范大学、上海大学
>
> **发布时间**：2024 年 12 月（arXiv 2412.03293）
>
> **发表状态**：已收录 ICML 2025（PMLR 267）
>
> 🔗 [arXiv](https://arxiv.org/abs/2412.03293) | [PDF](https://arxiv.org/pdf/2412.03293)
>
> **分类标签**：`VLA` `扩散策略` `自回归推理` `具身基础模型` `可解释性`

---

## 一句话总结

DiVLA 用同一个 VLM 骨干同时输出"推理 token"和"动作 token"，把推理 token 的末层 embedding 通过 FiLM 直接"注入"扩散动作头（而非像 ECoT 那样把推理文本递归拼回输入），在仅用约 39K 条预训练轨迹（约为 OpenVLA/Octo 所用 OXE 970K 轨迹的 1/25）的情况下，于多任务学习、工厂分拣、zero-shot 抓取 102 个未见物体、双臂收台等真机任务上全面超过 Diffusion Policy、Octo、TinyVLA、OpenVLA，且 DiVLA-2B 在 A6000 单卡上可跑到 82Hz（比同参数量的 OpenVLA-7B 快约 16 倍/在 7B 规模下快 8 倍）。

## 一、问题与动机

VLA 领域存在两条并行但互补性不足的技术路线：

- **自回归 NTP 路线**（RT-2、OpenVLA）：把连续动作离散化为 token，用 next-token-prediction 训练，能继承 LLM 的推理/语言能力，但离散化会破坏动作的连续性和精度，且逐 token 自回归解码在高频控制场景下推理效率低。
- **扩散策略路线**（Diffusion Policy、π0 用 flow matching）：把动作生成建模为去噪过程，更好地捕捉动作的多模态分布，生成速度快于 NTP，但天然不具备语言推理能力——而推理恰恰是 LLM 领域被证明能显著提升复杂任务表现的关键组件。

论文提出的核心问题：**能否把自回归模型的推理能力和扩散模型的高频、鲁棒动作生成能力结合起来？** 直接把两者拼在一起（推理输出→再喂给策略）会产生"逻辑推理"与"可执行策略"之间的隐式鸿沟，简单缝合不能充分释放推理的价值，这促使作者设计了一个直接注入的机制而非递归式的 pipeline（如 ECoT 需要把推理文本先生成、再作为下一轮输入喂回模型，带来额外的迭代开销）。

## 二、核心方法

### 2.1 整体架构

给定图像/文本/视频交织的输入序列：

1. 用共享的 **SigLIP** 视觉编码器对（可能多路的）相机视图分别编码，再经过一个 Transformer 投影为固定数量 $N$ 个视觉 embedding，多视角的视觉 token 直接拼接（concatenate），而非用独立分支处理各视角。
2. 语言-视觉骨干使用 **Qwen2-VL**（2B / 7B / 72B 三档规模，对应 DiVLA-2B/7B/72B），初始化自公开预训练权重，视觉理解能力保留。
3. VLM 最终层同时输出两路 token：**Reasoning Tokens**（自回归生成的任务分解/解释文字）和 **Action Tokens**（固定数量，代表待生成动作的表征）。
4. Action Tokens 经过一个两层 MLP + LayerNorm 的投影模块，对齐到扩散模型的输入维度（这一投影思路与 LLaVA 等视觉-语言模型中常见的桥接模块类似）。
5. **扩散动作头**采用标准 Diffusion Policy 结构（Chi et al., 2023），权重随机初始化，负责把动作 token 表征去噪为连续动作序列。动作解码器最底层接一个 MLP 输出关节空间动作；若要扩展新本体（embodiment），无需像 Octo 那样为每个本体单独复制一整套动作解码器，只需新初始化这一层 MLP 即可复用其余预训练知识，实现快速本体迁移。

### 2.2 推理注入模块（Reasoning Injection Module）

这是本文的核心创新。不同于多数自回归 VLA 需要"递归"设置——即把上一轮的推理输出转成下一轮模型的输入（如 ECoT）——DiVLA 把推理组件 tokenize 后的**末层 embedding**直接通过 **Feature-wise Linear Modulation (FiLM)** 注入到扩散策略网络的中间层，用以调制（modulate）策略网络各层的特征：

$$h' = \gamma(r) \odot h + \beta(r)$$

其中 $h$ 是策略网络某层的中间特征，$r$ 是推理 token 的末层 embedding，$\gamma(\cdot),\beta(\cdot)$ 是由 $r$ 映射得到的缩放/偏移参数。

**用大白话说**：模型先"嘴上说"一遍打算做什么（比如"抓黄色的辣椒"），这句话在网络内部被编码成一个向量后，不是被当成新的一段文字重新喂回模型走一遍推理，而是像一个"旋钮"，直接去拧一下扩散动作头里每一层特征的缩放和偏移，让接下来生成的动作"带着"这个语义指令的味道。因为不需要把推理结果重新过一遍自回归解码，所以不增加额外的推理时延，这也是该模块能在做到"既有推理又不拖慢速度"的关键。

### 2.3 训练目标

$$L = L_{diff} + \alpha L_{ntp}$$

其中 $L_{diff}$ 是扩散动作头的去噪损失，$L_{ntp}$ 是推理 token 的 next-token-prediction 损失，$\alpha$ 是权重超参数。作者观察到 $L_{ntp}$ 的量级通常比 $L_{diff}$ 小一个数量级左右，因此统一取 $\alpha=10$ 使两个损失项量级可比。

### 2.4 数据与预训练策略

- 预训练数据集：**DROID**（Khazatsky et al., 2024）和 **OXE**（O'Neill et al., 2023）。DiVLA-2B/7B 仅用 DROID 预训练；DiVLA-72B 因模型更大、需要更多数据，混合使用 OXE + DROID。
- DROID 原始数据只含动作，部分才配有观测和语言指令、且不含推理标注。作者用 **GPT-4o** 把 DROID 数据自动转写为带自然语言推理的形式，使预训练和微调阶段的数据格式保持一致（都含"推理 + 动作"）。
- 微调用 **LoRA**（沿用 π0 的设置）只调 VLM 部分，视觉编码器和 VLM 主干冻结，学习率 2e-5。

## 三、实验结果

真机实验覆盖 Franka 单臂和 AgileX 双臂（Aloha 式）两种平台，四类任务设置：多任务学习、工厂分拣、zero-shot 抓取（bin picking）、双臂收台（table bussing）。

**多任务学习（Franka，Table 1）**：DiVLA-2B 仅用 39K 条预训练轨迹（Octo/OpenVLA 用 970K 条 OXE 轨迹，约为 DiVLA 预训练数据的 25 倍）：

| 模型 | 预训练轨迹数 | 域内平均成功率 | 视觉泛化（OOD）平均成功率 |
|---|---|---|---|
| Diffusion Policy | - | 27.9% | 8.9% |
| TinyVLA | - | 45.5% | 17.8% |
| Octo | 970K | 24.3% | 28.9% |
| OpenVLA-7B | 970K | 39.4% | 26.7% |
| **DiVLA-2B** | **39K** | **83.6%** | **57.8%** |

**工厂分拣任务**（500 条训练轨迹，四类物体分拣到四个区域）：DiVLA 平均成功率 66.2%，超过次优 OpenVLA（45.3%）达 20.9 个百分点；在最难的"混合未见物体+杂乱"设置下 DiVLA 仍保持 60.0%，而 Diffusion Policy 骤降到 9.2%。

**zero-shot 抓取 102 个未见物体**：DiVLA-2B 63.7%，对比 Diffusion Policy 8.9%、Octo 19.6%、OpenVLA 28.4%、TinyVLA 23.5%。

**双臂收台（AgileX，Table 2）**：

| 场景 | Diffusion Policy | OpenVLA | DiVLA-2B |
|---|---|---|---|
| Seen | 45.8% | 0% | 72.9% |
| Mixed（含未见物体） | 31.2% | 0% | 70.8% |

**视角变化泛化**（相机位置大幅改变）：DiVLA-2B 60% vs. OpenVLA 0%、Diffusion Policy 0%。

**推理注入模块消融**（Table 8，多任务学习设置）：去掉推理注入后平均成功率从 83.6% 降到 50.3%，其中"把杯子放到盘子上"任务从 90.9% 骤降到 27.3%，证明推理信号对策略学习有实质贡献而非摆设。

**推理速度**（A6000 单卡）：DiVLA-2B 82Hz，DiVLA-7B 42Hz，OpenVLA-7B 5Hz——同为 7B 量级时 DiVLA 快约 8 倍。

**模型规模扩展**（Table 10）：

| 任务 | DiVLA-2B | DiVLA-7B | DiVLA-72B |
|---|---|---|---|
| 工厂分拣 | 66.2% | 74.9% | 82.4% |
| Zero-shot 抓取 | 63.7% | 66.7% | 75.9% |

模型越大、预训练数据越多，域内和 OOD 表现均单调提升，符合规模化规律（但 72B 同时切换到了更大的 OXE+DROID 混合数据，规模效应与数据效应存在混淆，见下文局限性）。

**新指令泛化**（Table 9）：面对训练数据中完全未出现的物体名+多步顺序指令（如"先拿西瓜，再拿蓝色纸垃圾，最后拿柠檬水"），OpenVLA 在三步顺序指令上成功率为 0/3，DiVLA-2B 为 2/3，作者将其归因于推理模块学会了把长指令分解为子任务。

**VQA/对话能力**（Table 11）：DiVLA 未经专门的视觉-语言联合训练，仍保留基本问答能力，颜色识别任务全部答对，但物体识别存在偏差（如把玩具龙识别成玩具虎，把橄榄球识别成普通球），提示模型更多依赖颜色而非形状/纹理特征。

## 四、局限性

论文正文没有独立的 Limitations 小节，仅有例行的 Impact Statement，以下局限基于实验设置和数据的批判性阅读：

- **全部为真机实验，缺乏标准仿真基准**：论文未在 LIBERO、SimplerEnv 等社区常用仿真基准上报告结果，只有自建的 Franka/AgileX 真机任务，难以和更广泛的 VLA 工作做严格的横向比较。
- **规模效应与数据效应存在混淆**：DiVLA-72B 相比 2B/7B 不仅参数量更大，预训练数据也从纯 DROID 换成了 OXE+DROID 混合，Table 10 展示的性能提升无法干净地归因于"模型更大"本身。
- **基线的预训练-微调设置可能不完全公平**：OpenVLA/Octo 使用远大于 DiVLA 的 OXE 预训练数据（970K vs 39K），但微调阶段用的是同样小规模的任务数据（500~580 条轨迹）；作者将"数据高效"作为卖点，但也意味着对比并未控制预训练数据源和规模这一变量，观察到的差距中有多少来自架构设计、多少来自数据分布差异并不完全清晰。
- **推理标注来自 GPT-4o 自动生成、无人工校验描述**：DROID 数据的推理标签是自动转写的，论文未描述任何人工质检或过滤流程，标注噪声/幻觉的影响未被评估。
- **自我纠错能力仅有单个示例验证**（Figure 6 的"蓝色玩具车→六角扳手"），未做系统性的、量化的推理鲁棒性测试（例如推理出错时策略是否会连带出错）。
- **VQA/对话能力仅做了少量定性展示**（Table 11 寥寥数题），未使用标准 VQA 基准量化评估，且已暴露出依赖颜色而非形状/结构特征的浅层视觉理解问题。
- **FiLM 注入方式缺乏消融对比**：论文只验证了"有无推理注入"，没有与其他注入机制（如 cross-attention、拼接）做对比，因此无法判断 FiLM 是否是该场景下的最优选择，还是仅仅"有比没有好"。
- **超参数 $\alpha=10$ 未做敏感性分析**：该权重被固定用于所有实验，但只给出了"约十倍量级差异"的定性依据，未展示对不同 $\alpha$ 取值的消融曲线。

## 五、评价与展望

**优点**：DiVLA 提出的推理注入范式相比 ECoT 式的"生成推理文本→重新输入→再走一遍自回归+扩散"的做法更直接、少一次推理开销，工程上是一个清晰可行的折中方案；在真机实验中同时展示了成功率、推理速度、跨本体适应、可解释性（通过推理文字追踪失败原因）等多个维度的优势，实验覆盖面（多任务、分拣、zero-shot 抓取、双臂收台、视角泛化、模型规模）在同期 VLA 论文中较为全面；"仅用 1/25 预训练数据即超过基线"这一结果对追求数据效率的实践者有较强吸引力。

**与其他公开工作的关系**：DiVLA 与 π0（Black et al., 2024）同属"扩散/flow-matching 生成连续动作"一脉，但 π0 本身不显式建模语言推理；与 ECoT（Zawalski et al., 2024）同属"给 VLA 加推理"一脉，但注入机制不同（FiLM 直接调制 vs. 递归式思维链）；与 RT-2、OpenVLA 同属"VLM 骨干做机器人策略"一脉，但动作头从离散 token 换成了扩散去噪。可以说 DiVLA 是在"推理注入位置"这一设计维度上，对已有三条技术路线做的一次系统性缝合与实证。

**开放问题与可能的改进方向**：

1. 更严谨的消融应当把"预训练数据规模/来源"作为独立控制变量，单独验证架构（推理注入）与数据效率两个卖点各自的贡献；
2. FiLM 只是条件注入的一种朴素形式，是否存在更细粒度（如逐 token 的 cross-attention 或分层门控）的注入机制能进一步提升推理到动作的信息传递效率，值得探索；
3 论文展示的自我纠错例子提示"推理错误会不会传导为动作错误"是一个值得系统评测的方向，特别是在长程、多阶段任务中推理链条更长、误差累积风险更高；
4. 目前的推理监督信号来自 GPT-4o 自动标注，如何设计更可靠、可验证的推理数据生成/筛选流程（而非纯粹依赖大模型自动转写），是提升该类"自生成推理"VLA 上限的关键瓶颈之一。

## 参考

- Brohan, A. et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818, 2023.
- Kim, M. J. et al. OpenVLA: An open-source vision-language-action model.
- Black, K. et al. π0: A vision-language-action flow model for general robot control. arXiv:2410.24164, 2024.
- Chi, C. et al. Diffusion Policy: Visuomotor policy learning via action diffusion. arXiv:2303.04137, 2023.
- Zawalski, M. et al. ECoT: Robotic control via embodied chain-of-thought reasoning. arXiv:2407.08693, 2024.
