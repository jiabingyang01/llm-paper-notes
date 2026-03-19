# REVERIE：反思式指令微调缓解 LVLM 幻觉

> **论文**：*Reflective Instruction Tuning: Mitigating Hallucinations in Large Vision-Language Models*
>
> **作者**：Jinrui Zhang, Teng Wang, Haigang Zhang, Ping Lu, Feng Zheng
>
> **机构**：Southern University of Science and Technology (SUSTech)、The University of Hong Kong、Shenzhen Polytechnic University、ZTE、Peng Cheng Laboratory
>
> **发布时间**：2024年7月（**ECCV 2024**）
>
> **链接**：[arXiv](https://arxiv.org/abs/2407.11422) | [项目页](https://zjr2000.github.io/projects/reverie)
>
> **分类标签**：`反思微调` `正负 Rationale` `细粒度推理监督` `数据集` `指令微调`

---

## 一句话总结

在视觉指令微调中引入**正向 rationale（解释正确答案的理由）和负向 rationale（解释错误答案的错误原因）**的反思学习，构建首个包含 115k 指令 × 254k (指令, 回答, rationale) 三元组的 REVERIE 数据集，以多轮对话格式将 rationale 学习与回答预测解耦，POPE 提升 12.7 点，MME 提升 321 点。

---

## 一、问题与动机

### 1.1 幻觉的一个被忽视的根因：缺少细粒度推理监督

现有幻觉缓解方法关注了数据噪声、视觉感知不足、模态对齐和语言偏差等因素，但忽略了一个关键贡献者：**训练时缺乏细粒度推理监督**。

传统视觉指令微调（Vanilla Instruction Tuning）只让模型预测最终回答，不提供中间推理步骤（rationale）。这会导致：

- 模型在指令和回答之间建立**表面捷径（superficial shortcuts）**，而非内化推理逻辑
- 模型无法学习到达正确答案所需的**关键视觉证据和区分性信息**
- 对复杂推理指令（如 ScienceQA）尤其容易产生错误推断

### 1.2 人类学习的启示：反思促进理解

人类在学习过程中通过反思（reflection）来提升——分析每一步推理、从错误中学习。现有 LVLM 训练缺乏这种反思机制。

### 1.3 现有数据集的不足

| 数据集 | 训练实例 | 负面回答 | 正向 Rationale | 负向 Rationale |
| --- | --- | --- | --- | --- |
| LLaVA-1.5 | 665k | 无 | 77k | 无 |
| InstructBLIP | 1.2M | 无 | 77k | 无 |
| MiniGPT-4 | 3.5k | 无 | 无 | 无 |
| LRV-Instruction | 400k | 无 | 无 | 无 |
| **REVERIE** | **254k** | **有** | **115k** | **138k** |

关键缺失：
1. **没有负面回答标注**：无法提供区分正确与错误回答的判别性监督
2. **正向 rationale 有限**：LLaVA/InstructBLIP 仅在开放式推理任务中有部分 rationale，且仅覆盖正确回答
3. **没有负向 rationale**：缺乏解释错误原因的监督信号

---

## 二、REVERIE 数据集

### 2.1 数据收集流水线

数据构建分三步：

**Step 1：指令-回答生成**

使用 Gemini-Vision-Pro 对 Visual Genome、COCO、ScienceQA 图像生成指令和回答对：
- 要求生成**需要多步推理**的指令，避免简单查询
- 每条指令同时生成一个正确回答和一个**易混淆的错误回答**

**Step 2：反思 Rationale 生成**

用两个独立 prompt 分别生成正向和负向 rationale：

- **正向 rationale**：包含视觉内容分析、核心视觉概念识别、基于视觉信息和知识的逐步推理
- **负向 rationale**：突出区分性细节，说明为什么错误答案是错误的

**Step 3：一致性过滤**

利用正负 rationale 间的一致性来过滤噪声样本。原理：正负 rationale 虽强调不同方面，但应包含**互相一致**的信息。用 Gemini-Pro 检测正负 rationale 间的矛盾（如冲突事实），过滤不一致样本。

### 2.2 数据统计

| 维度 | 数值 |
| --- | --- |
| 图像总数 | 71,558 |
| 图像来源 | Visual Genome 50,938 + COCO 15,706 + ScienceQA 4,914 |
| 指令数 | 115,280 |
| 正面回答 | 115,280 |
| 负面回答 | 138,897（每条指令平均 1.2 个） |
| 总训练实例 | 254,177 个 (指令, 回答, rationale) 三元组 |

任务类型分布：

| 任务类型 | 占比 |
| --- | --- |
| Short-Answer QA | 62% |
| Open-Ended QA | 20% |
| Multiple-Choice | 15% |
| Yes/No | 3% |

Rationale 质量：50%+ 的 rationale 超过 25 词，包含 8+ 名词，表明 rationale 提供了大量信息和知识。

---

## 三、反思式指令微调

### 3.1 整体架构

标准 LVLM 架构（视觉编码器 + 视觉-语言连接器 + LLM）。给定图像 $I$ 和指令 $X$，自回归生成回答 $A$。

### 3.2 多轮对话格式——解耦 Rationale 与回答

直接在单轮预测中同时生成 rationale 和回答会干扰推理结果（因为某些任务不需要显式 rationale）。解决方案：**将 rationale 学习与回答预测解耦为多轮对话**。

**第一轮**：输入 $I$ 和 $X$，模型预测回答 $A$

**第二轮**：引入 rationale 生成提示，模型预测 rationale

这样设计的关键好处：**回答预测仅依赖 $I$ 和 $X$ 的上下文**，不受 rationale 干扰，推理时只需第一轮即可。

### 3.3 Rationale 学习提示

**正向 rationale 提示**：

> "Explain why. Including any necessary facts or knowledge"

**负向 rationale 提示**（提供错误答案让模型解释为什么错）：

> "Explain why this answer is wrong: {incorrect answer}. Including any necessary facts or knowledge."

### 3.4 训练细节

模型在两个基线上验证：

| 模型 | 基座 | 训练数据 |
| --- | --- | --- |
| REVERIE-1.0-7b-lora | LLaVA-1.0-7b-lora | REVERIE + LLaVA-Instruct-80k |
| REVERIE-1.5-7b-lora | LLaVA-1.5-7b-lora | REVERIE + 原始 665k 指令 |

训练设置：LoRA（attention dim=128, alpha=256），batch size 128，1 epoch，3% warmup + cosine decay。

---

## 四、实验结果

### 4.1 六基准性能对比

| 方法 | ScienceQA$^I$ | POPE | MME | MMBench | MM-Vet | GQA |
| --- | --- | --- | --- | --- | --- | --- |
| LLaVA-1.0-7b-lora | 42.7 | 71.1 | 819.8 | 27.2 | 30.0 | 7.10 |
| **REVERIE-1.0-7b-lora** | **70.1** | **83.8** | **1168.1** | **55.4** | 27.8 | **36.5** |
| 提升 | +27.4 | **+12.7** | **+348.3** | **+28.2** | -2.2 | +29.4 |

| 方法 | ScienceQA$^I$ | POPE | MME | MMBench | MM-Vet | GQA |
| --- | --- | --- | --- | --- | --- | --- |
| LLaVA-1.5-7b-lora (w/ ScienceQA) | 76.3 | 86.6 | 1439.2 | 67.9 | 31.1 | 60.7 |
| **REVERIE-1.5-7b-lora** | **80.5** | 86.4 | **1474.9** | 67.3 | 30.8 | **61.8** |

关键发现：
- POPE 提升 12.7 点（1.0 基线），表明 REVERIE 提供的细粒度监督显著缓解对象幻觉
- MME 提升 348 点，表明 REVERIE 包含更广泛的视觉概念和知识
- 对比公平基线（LLaVA-1.5 + ScienceQA QA 数据），单纯加 QA 数据反而降低 MME/GQA，而加 rationale 后性能提升——说明**仅学习回答可能建立捷径**

### 4.2 正负 Rationale 的互补效果（POPE）

| Rationale 配置 | Random F1 | Popular F1 | Adversarial F1 |
| --- | --- | --- | --- |
| 仅回答 | 86.90 | 82.90 | 78.49 |
| +正向 rationale | 85.68 | 82.11 | 78.58 |
| +负向 rationale | 86.43 | 83.57 | 79.23 |
| **+正负 rationale** | **87.10** | **84.03** | **80.31** |

- 单独用正向 rationale 甚至轻微下降，单独用负向 rationale 稳定提升
- **两者结合效果最佳**，正向提供视觉细节，负向提供区分性线索

### 4.3 正负 Rationale 的能力侧重（MMBench）

| 配置 | AR | CP | FP-S | FP-C | RR | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 仅回答 | 61.69 | 60.09 | 50.51 | 42.76 | 42.61 | 51.70 |
| +正向 | 63.18 | **67.11** | 44.14 | **49.16** | 40.87 | 52.13 |
| +负向 | **64.68** | 64.43 | 50.84 | 44.83 | **43.48** | 52.30 |
| +正负 | 65.14 | 68.46 | **57.24** | 51.52 | 50.43 | **55.44** |

- **正向 rationale 增强感知**（CP、FP-C 更高）：正向 rationale 提供更多视觉细节
- **负向 rationale 增强推理**（AR、RR 更高）：负向 rationale 提供区分正误的关键线索
- 负向 rationale 帮助模型更好区分正确答案与负面回答，对多选题等有候选答案的任务收益更大

### 4.4 消融实验

**Rationale 生成提示设计**：

| 变体 | Explicit Guidance | Type-Specific | Length Control | MMB | POPE |
| --- | --- | --- | --- | --- | --- |
| (a) 无引导 | | | | 55.4 | 83.8 |
| (b) 引导+类型特定 | Yes | Yes | | 55.2 | 83.2 |
| (c) 引导+类型+长度 | Yes | Yes | Yes | 54.3 | 83.8 |
| **(d) 仅引导** | **Yes** | | | **55.4** | **83.8** |

结论：简单的 explicit guidance 效果最好，过多约束反而阻碍模型理解任务。

**一致性过滤效果**：

| 配置 | MMB | POPE | GQA | MME |
| --- | --- | --- | --- | --- |
| 未过滤 | 55.2 | 83.5 | 36.3 | 1132.3 |
| **过滤后** | **55.4** | **83.8** | **36.5** | **1168.1** |

过滤噪声数据对所有基准均有提升。

**对话上下文设计**：

| 格式 | MMB | POPE | GQA | MME |
| --- | --- | --- | --- | --- |
| 正向 rationale 在前 | 49.3 | 79.5 | 36.2 | 1144.7 |
| 负向 rationale 在前 | 48.6 | 82.4 | 36.4 | 1160.2 |
| **分离上下文** | **55.4** | **83.8** | **36.5** | **1168.1** |

将正负 rationale 放在同一对话中会导致**信息泄露**——模型倾向于利用已有 rationale 信息而非真正学习推理，分离上下文效果最佳。

### 4.5 MMHal-Bench 幻觉评估

| 指标 | LLaVA-1.0 | REVERIE-1.0 | LLaVA-1.5 | REVERIE-1.5 |
| --- | --- | --- | --- | --- |
| Avg. Score (↑) | 1.39 | 1.43 (+2.9%) | 2.23 | 2.36 (+5.8%) |
| Halluc. Rate (↓) | 0.76 | 0.73 (-3.9%) | 0.55 | **0.50 (-9.1%)** |

### 4.6 更多模型验证

| 模型 | POPE | MMBench | ScienceQA | 平均 |
| --- | --- | --- | --- | --- |
| MOE-LLaVA-1.6Bx4 | 85.9 | 63.3 | 63.9 | 71.0 |
| MOE-REVERIE | **86.7** | **64.5** | **77.1** | **76.1** |
| LLaVA-Phi3-LoRA | 85.6 | 68.2 | 73.8 | 76.2 |
| REVERIE-Phi3-LoRA | **86.3** | **69.0** | **86.7** | **80.7** |

在 MoE 架构和更强 LLM（Phi-3）上也一致提升。

---

## 五、局限性与未来方向

1. **数据依赖 Gemini**：REVERIE 的 rationale 由 Gemini-Vision-Pro 生成，质量受限于生成模型本身的能力。一致性过滤虽能部分缓解噪声，但无法完全消除
2. **基座模型受限**：实验仅在 LLaVA 系列 7B 模型上验证，对更大或更新的模型（如 InternVL、Qwen-VL-2）的效果未知
3. **推理时不使用 rationale**：rationale 仅在训练时使用，推理时只取第一轮回答。能否在推理时也利用 rationale（如 Chain-of-Thought）可能带来更大收益
4. **负面回答质量**：每条指令仅 1.2 个负面回答，且依赖 Gemini 生成"易混淆"回答的质量

---

## 六、个人思考

### 6.1 与项目内其他幻觉缓解工作的对比

| 方法 | 干预阶段 | 是否需要训练 | 核心思路 | 关键创新 |
| --- | --- | --- | --- | --- |
| **REVERIE** | 训练时 | 是（数据） | 正负 rationale 反思学习 | 细粒度推理监督 |
| CSR | 训练时 | 是（迭代 DPO） | CLIP 校准自奖励 | 模态对齐奖励 |
| SENTINEL | 训练时 | 是（C-DPO） | 句子级偏好学习 | 早期干预 |
| HALC | 解码时 | 否 | FOV 对比 + beam search | 局部视觉 grounding |
| ICD | 解码时 | 否 | 扰动指令对比解码 | 多模态对齐不确定性 |
| OPERA | 解码时 | 否 | 注意力惩罚 + 回溯 | 聚合模式检测 |
| HIME | 推理前 | 否 | 零空间投影编辑 | 层自适应模型编辑 |
| MemVR | 解码时 | 否 | FFN 视觉回溯 | 不确定性触发 |

REVERIE 和 CSR/SENTINEL 同属**训练阶段**方法，但切入角度不同：
- **CSR** 用 CLIP 分数做自奖励来对齐模态
- **SENTINEL** 用偏好学习在幻觉首发处干预
- **REVERIE** 提供显式的推理过程监督，教模型"为什么对、为什么错"

三者可**级联使用**：先用 REVERIE 的 rationale 数据做指令微调，再用 SENTINEL 的 C-DPO 做偏好对齐。

### 6.2 "捷径学习"假说的意义

REVERIE 揭示了一个重要现象：仅加入 ScienceQA 的 QA 数据反而导致 MME/GQA 下降，说明**没有 rationale 的复杂推理数据可能让模型学到更多捷径**。这与 ICD 中发现的"扰动指令放大幻觉"现象相呼应——模型并非真正理解了指令的含义，而是在走统计近路。

rationale 的作用类似于**知识蒸馏中的 soft label**：不仅告诉模型"答案是什么"，还告诉它"为什么是这个答案"，提供了更丰富的梯度信号。

### 6.3 正负 Rationale 的不对称贡献

实验中负向 rationale 单独使用时比正向 rationale 效果更好（POPE Adversarial: 79.23 vs 78.58），这与对比学习中"难负样本更有价值"的经验一致。正向 rationale 提供的视觉细节可能与图像 token 本身信息冗余，而负向 rationale 提供的**区分性信息**才是模型真正缺乏的。

### 6.4 分离上下文的信息论解释

将正负 rationale 分离为独立样本（而非同一对话中的连续轮次）效果最好。这可以从信息论角度理解：在同一上下文中，模型可能直接"抄"前一个 rationale 的信息来生成当前 rationale，导致**因果推理被信息泄露替代**。分离后，每个 rationale 必须从图像和指令中独立推导，真正实现了推理能力的训练。

---

## 参考

- **LLaVA**（Liu et al., NeurIPS 2024）：视觉指令微调——REVERIE 的基座模型和训练框架
- **ICD**（Wang et al., ACL 2024）：指令对比解码——同样关注指令-回答捷径问题，但从解码端解决
- **LRV-Instruction**（Liu et al., ICLR 2024）：鲁棒视觉指令微调——通过平衡数据集减少幻觉，但无 rationale 标注
- **Chain-of-Thought**（Wei et al., NeurIPS 2022）：思维链推理——REVERIE 的 rationale 可视为训练时的 CoT 监督
- **ScienceQA**（Lu et al., NeurIPS 2022）：科学问答多模态推理——REVERIE 包含其人工标注的 QA 对和正向 rationale
