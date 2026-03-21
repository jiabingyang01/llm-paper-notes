# ChatVLA：统一多模态理解与机器人控制的 VLA 模型

> **论文**：*ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model*
>
> **作者**：Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, Feifei Feng
>
> **机构**：Midea Group、华东师范大学、上海大学、北京人形机器人创新中心、清华大学
>
> **发布时间**：2025年2月
>
> **链接**：[arXiv](https://arxiv.org/abs/2502.14420) | [项目主页](https://chatvla.github.io/)
>
> **发表会议**：EMNLP 2025

---

## 一句话总结

ChatVLA 首次系统分析了 VLA 训练中的 spurious forgetting 和 task interference 问题，提出 Phased Alignment Training（先学控制再恢复理解）+ MoE 双专家架构（共享 attention 隔离 MLP），在仅 2B 参数下同时实现接近基座 VLM 的多模态理解能力和优于 OpenVLA 的 25 项真实机器人控制任务表现。

---

## 二、问题与动机

### 2.1 VLA 的能力割裂困境

现代 VLA 模型建立在预训练 VLM 之上，理论上继承了强大的视觉-语言理解能力。然而实际训练后，VLA 模型几乎完全丧失了对话和多模态理解能力。这一现象是矛盾的——为什么在一个"理解力"极强的模型上训练后，反而失去了理解力？

本文系统考察了三种 VLA 训练范式：

| 训练设置 | 代表方法 | 控制能力 | 理解能力 |
| --- | --- | --- | --- |
| 仅 robot data | OpenVLA, TinyVLA, $\pi_0$ | 正常 | 完全丧失（所有 benchmark = 0） |
| robot data + reasoning | ECoT, DiffusionVLA | 有所提升 | 部分恢复（模板化推理"重激活"对齐） |
| robot data + visual-text data | RT-2 | 显著下降 | 接近基座模型 |

### 2.2 两个核心问题

**Spurious Forgetting（伪遗忘）**：仅用 robot data 训练后，VLM 的多模态理解能力看似完全消失（所有 benchmark 归零），但这并非知识的真正丢失，而是视觉-文本对齐被机器人数据"覆写"。证据：仅加入模板化 reasoning 数据就能部分恢复理解能力，说明知识仍在，只是对齐链路断裂。

**Task Interference（任务干扰）**：当 robot data 和 visual-text data 联合训练时，控制任务和理解任务在共享参数空间中竞争，导致控制性能显著下降。这解释了为什么 RT-2 路线虽然保留了理解能力，但控制表现不佳。

### 2.3 ChatVLA 的核心思路

需要一个统一框架：
1. **先保障控制能力**——控制任务复杂度远高于恢复理解
2. **再恢复理解能力**——预训练 VLM 的对齐仅需少量 visual-text 数据即可"重激活"
3. **隔离任务表征**——用独立参数处理两类任务，但保留共享层促进知识迁移

---

## 三、预备知识

### 3.1 VLA 形式化定义

机器人控制：从示教数据集 $D_{\text{robot}} = \{\tau_i\}_{i=1}^{N}$ 学习策略 $\pi(a_t | v_t, t_t)$，其中 $\tau_i = \{((v_1, t_1), a_1), \ldots, ((v_T, t_T), a_T)\}$。

多模态理解：从 visual-text 数据集 $D_{v\text{-}t} = \{\phi_i\}_{i=1}^{M}$ 学习分布 $\pi(t | v)$。

统一目标：单模型 $\pi$ 同时擅长两者。

### 3.2 Dual Coding Theory

Paivio (1991) 提出人脑通过两个独立但互联的系统处理信息——一个负责物理运动技能，另一个负责语言和视觉。ChatVLA 的架构设计直接受此启发：共享 attention 层对应"互联"，独立 MLP 专家对应"独立系统"。

---

## 四、核心方法

### 4.1 Phased Alignment Training（分阶段对齐训练）

受 curriculum learning 启发的两阶段训练策略：

**Stage 1：控制优先训练**
- 训练数据：robot data + reasoning data
- 激活组件：仅 Control Expert + Action Head
- 目标：让模型首先掌握具身控制能力
- reasoning data 的作用：维持视觉-文本组件的连续对齐，防止对齐链路完全断裂

**Stage 2：联合共训练**
- 训练数据：robot data + visual-text data（比例 3:1）
- 激活组件：Control Expert + Understanding Expert + Action Head + LLM Head
- 目标：在控制能力已稳固的基础上，"重激活"冻结的多模态理解能力

关键 insight：将"恢复理解"放在第二阶段而非一开始联合训练。因为控制任务训练难度远高于恢复理解——预训练 VLM 本身就有理解能力，只需少量数据重新对齐。

### 4.2 Mixture-of-Experts 架构

对 LLM 骨架的 MLP 层施加静态 MoE，设两个专家：

给定第 $l$ 层的输入 $x^l$，先通过共享的多头自注意力：

$$x^{l'} = \text{MHA}(x^{l-1}) + x^{l-1}$$

然后进入 MoE 层，根据输入数据类型静态路由：

$$\text{MoE}(x^{l'}) = \begin{cases} f(\text{FFN}_{v\text{-}t})(x^{l'}) & m = 0 \text{（理解任务）} \\ f(\text{FFN}_{\text{robot}})(x^{l'}) & 1 \leq m \leq M_r \text{（控制任务）} \end{cases}$$

最终输出：$x^l = x^{l'} + \text{MoE}(x^{l'})$

**设计要点**：

- **路由是静态的**（非学习型 gating）：通过 system prompt 区分任务类型——"Answer based on question" 触发理解专家，"Predict robot action" 触发控制专家
- **共享 self-attention 层**：理解和控制共享注意力表征，因为场景理解（识别物体、判断位置）对两个任务都必不可少
- **隔离 MLP 层**：避免两类任务在参数空间中相互干扰
- **推理时只激活一条路径**：保持与基座模型相同的参数量和推理开销

### 4.3 为什么共享 Self-Attention？

一种替代方案是 Mixture of Attention（对注意力层也做分离）。但作者认为理解和控制任务共享有益的底层表征——机器人控制本身就需要场景理解、物体识别和空间推理，这些高维表征与多模态理解具有相似的语义概念。实验也验证了共享 attention 比分离 attention 效果更好。

### 4.4 输出头设计

- **Action Head**：处理控制任务，输出机器人动作
- **LLM Head**：处理理解/对话任务，输出自然语言 token

两个头分别接在各自专家的输出之后，通过 system prompt 在推理时选择激活哪条路径。

### 4.5 整体架构

- VLM 骨架：**Qwen2-VL-2B**
- 视觉编码器：ViT + LoRA 微调
- Action Head：沿用 DiffusionVLA 的设计
- 学习率：2e-5
- 数据比例：visual-text : robot = 1:3（Stage 2）
- Visual-text 数据来源：LLaVA-1.5 数据集（54K）

---

## 五、实验结果

### 5.1 多模态理解与 VQA

| 方法 | 参数 | MMMU | MMStar | MME | OCRBench | HallBench | TextVQA | DocVQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2-VL（基座） | 2B | 41.1 | **48.0** | **1872.0** | **809** | **41.7** | **79.7** | **88.57** |
| OpenVLA | 7B | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ECoT | 7B | 5.4 | 0 | 0 | 12 | 0.9 | 0 | 0 |
| DiVLA | 2B | 17.2 | 21.1 | 186.5 | 294 | 9.0 | 7.5 | 15.2 |
| **ChatVLA** | **2B** | **37.4** | **47.2** | **1435.2** | **729** | **39.9** | **71.2** | **83.3** |

核心发现：
- ChatVLA 在所有 VLA 方法中**遥遥领先**，MMMU 上比 ECoT 高 **6 倍**（37.4 vs 5.4），MMStar 47.2 vs ECoT 的 0
- 与基座 Qwen2-VL 相比差距很小：MMMU 37.4 vs 41.1，TextVQA 71.2 vs 79.7，说明理解能力被有效恢复
- 仅 2B 参数，远小于 OpenVLA/ECoT 的 7B

### 5.2 长时域直接指令任务

| 方法 | Task 1 Avg.Len. | Task 2 Avg.Len. | Task 3 Avg.Len. | Task 4 Avg.Len. |
| --- | --- | --- | --- | --- |
| Octo | 0.08 | 0.21 | 0.11 | 0.33 |
| OpenVLA | 0.06 | 0.29 | 0.15 | 0.42 |
| **ChatVLA** | **0.54** | **0.64** | **1.00** | **0.75** |

ChatVLA 在 Task 3（开抽屉→放玩具→关抽屉）上达到 **100% 成功率**，涉及三个不同技能的组合。

### 5.3 长时域高层规划任务

| 方法 | Task 5-8 Avg.Len. | Task 9-10 Avg.Len. | Task 11-13 Avg.Len. |
| --- | --- | --- | --- |
| Octo | 0.23 | 0.28 | 0.08 |
| OpenVLA | 0.31 | 0.33 | 0.10 |
| **ChatVLA** | **0.94** | **0.83** | **0.59** |

Task 5-8（移积木到篮子→放玩具进抽屉）的平均成功长度 0.94，OpenVLA 仅 0.31，提升 **3 倍**。

### 5.4 跨技能多任务

| 方法 | Bathroom | Kitchen | Tabletop | Avg. |
| --- | --- | --- | --- | --- |
| Octo | 4/33 | 3/22 | 11/52 | 18/107 |
| OpenVLA | 5/33 | 5/22 | 10/52 | 20/107 |
| **ChatVLA** | **16/33** | **9/22** | **30/52** | **55/107** |

在浴室（受限空间）和厨房（复杂环境）中优势尤为突出。25 项真实任务共 528 次试验。

### 5.5 消融实验

#### 数据比例消融（visual-text : robot data）

| 比例 | MMMU | MMStar | MME | OCRBench | HallBench | TextVQA |
| --- | --- | --- | --- | --- | --- | --- |
| 1:1 | 36.1 | 44.7 | 1426.9 | 691 | 36.2 | 72.6 |
| 3:1 | 35.3 | 45.3 | 1399.5 | 726 | 36.4 | 72.7 |
| **1:3** | **37.4** | **47.2** | **1435.2** | **729** | **39.9** | 71.2 |

令人意外的发现：**更少的 visual-text 数据反而效果更好**。这验证了"spurious forgetting"假说——理解能力并非真正丢失，少量数据即可重激活对齐。

#### MMMU 细粒度分析

与 Qwen2-VL 相比，ChatVLA 在 Art Theory、Lab Medicine、Pharmacy、Literature、Psychology 五个子领域差距最大。原因：LLaVA 数据集（COCO、GQA、OCR-VQA、TextVQA、VisualGenome）缺乏这些领域的专业知识。这说明性能差距主要来自数据覆盖而非方法缺陷。

---

## 六、局限性与未来方向

1. **专业领域理解受限于 co-training 数据**：LLaVA 数据缺乏医学、文学等专业领域知识。作者指出更合适的专业数据有望进一步缩小与基座 VLM 的差距

2. **控制任务评估以真实机器人为主，缺乏标准仿真基准**：未在 CALVIN、LIBERO 等标准仿真环境上评测，与其他 VLA 方法缺乏直接对比

3. **静态路由而非学习型 gating**：当前根据 system prompt 硬性切换专家，无法处理需要同时推理和操作的混合任务

4. **MoE 增加了参数量**：虽然推理时只激活一条路径，但训练时需要维护两套 MLP 参数

5. **单一动作表征**：沿用 DiffusionVLA 的动作头设计，未探索不同动作表征（离散 token、flow matching 等）对统一框架的影响

---

## 七、个人思考

### 7.1 Spurious Forgetting 的深层意义

ChatVLA 最重要的贡献可能不是方法本身，而是对 spurious forgetting 的系统验证。纯 robot data 训练后所有理解 benchmark 归零，但加入模板化 reasoning 就能部分恢复，这说明 VLM 的知识并非被"擦除"，而是对齐通路被"改写"。这一发现对整个 VLA 领域意义深远：**VLA 训练的核心挑战不是学新知识，而是保护旧对齐**。这与 [BridgeVLA](BridgeVLA_2025.md) 的"输入-输出对齐"哲学和 [SF](SF_2025.md) 的隐式空间对齐理念一脉相承。

### 7.2 MoE vs 全参数微调

ChatVLA 用静态 MoE 解决 task interference，本质上是将一个 2B 模型拆成两个"子网络"——共享 attention + 独立 MLP。这比传统的 multi-task learning 更优雅，但也更保守。一个有趣的对比是 [DreamVLA](DreamVLA_2025.md) 的 block-wise 结构化注意力——同样是隔离不同任务的表征，但 DreamVLA 在注意力层面做分离，而 ChatVLA 在 MLP 层面做分离。两者的设计决策基于不同的假设：DreamVLA 认为不同预测目标需要独立的注意力模式，ChatVLA 认为理解和控制共享注意力但需要独立的特征变换。

### 7.3 训练顺序的重要性

Phased Alignment Training 的核心 insight 是"先难后易"——控制任务训练难度高，应优先保障；理解能力恢复容易（少量数据即可）。这与大多数 curriculum learning 的"先易后难"策略相反。背后的逻辑差异在于：控制能力是需要"从头学"的，而理解能力是"已存在但需要重激活"的。这一不对称性决定了训练顺序。

### 7.4 与其他 VLA 基础模型的对比

| 维度 | ChatVLA | [π₀](pi0_2024.md) | [GR-3](GR3_2025.md) | OpenVLA |
| --- | --- | --- | --- | --- |
| 参数量 | 2B | 3B | 3B | 7B |
| 基座 VLM | Qwen2-VL-2B | PaliGemma | SigLIP+Phi-3 | Prismatic-7B |
| 动作表征 | 扩散模型 | Flow Matching | 离散 token | 离散 token |
| 多模态理解 | MMMU 37.4 | 未评测 | 未评测 | 0 |
| 核心创新 | MoE + 分阶段训练 | Flow Matching + Action Chunking | 生成式世界模型 | 大规模 robot data |
| 统一理解与控制 | 是 | 否 | 否 | 否 |

ChatVLA 是目前唯一认真对待"理解 + 控制统一"的 VLA 模型。其他方法要么不评测理解能力，要么完全丧失理解能力。

### 7.5 1:3 数据比例的启示

消融实验中最反直觉的发现：visual-text 数据越少（1:3），理解性能反而越好。可能的解释：(1) 过多的 visual-text 数据在 Stage 2 引入更强的 task interference，MoE 难以完全隔离；(2) 少量高质量数据恰好"唤醒"对齐而不破坏控制表征。这为后续工作指明了方向——**精选少量高质量 visual-text 数据可能比大规模混合训练更有效**。

---

## 参考

- [DiffusionVLA](https://arxiv.org/abs/2412.03293) — ChatVLA 的分析基座和动作头来源，统一扩散与自回归的 VLA
- [ECoT](https://arxiv.org/abs/2407.08693) — 通过 chain-of-thought 推理增强 VLA 的代表方法，ChatVLA 的主要对比对象
- [RT-2](https://arxiv.org/abs/2307.15818) — 首个探索 robot data + visual-text data 联合训练的 VLA 方法
- [MoE-LLaVA](https://arxiv.org/abs/2401.15947) — 在多模态大语言模型中引入 MoE 的先驱工作
- [OpenVLA](https://arxiv.org/abs/2406.09246) — 开源 VLA 基线，25 项真实任务的主要对比方法
- [Qwen2-VL](https://arxiv.org/abs/2409.12191) — ChatVLA 的 VLM 骨架
