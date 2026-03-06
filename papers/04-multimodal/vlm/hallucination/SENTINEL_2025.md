# SENTINEL：句子级早期干预缓解 MLLM 对象幻觉

> **论文**：*Mitigating Object Hallucinations via Sentence-Level Early Intervention*
>
> **作者**：Shangpin Peng, Senqiao Yang, Li Jiang, Zhuotao Tian
>
> **机构**：Harbin Institute of Technology (Shenzhen), The Chinese University of Hong Kong, CUHK (Shenzhen)
>
> **发布时间**：2025年7月
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2507.12455) | [GitHub](https://github.com/pspdada/SENTINEL)
>
> **发表会议**：ICCV 2025
>
> **分类标签**：`MLLM` `Object Hallucination` `Preference Learning` `DPO` `In-domain Data`

---

## 一句话总结

发现幻觉主要在生成早期萌发并向后传播，提出 SENTINEL 框架：通过域内自举采样 + 开放词汇检测器交叉验证构建句子级偏好数据，配合上下文感知 DPO (C-DPO) 在幻觉首次出现处实施早期干预，Object HalBench 上幻觉率降低约 92%，同时保持甚至提升通用能力。

---

## 一、问题与动机

### 1.1 MLLM 的对象幻觉及其传播特性

多模态大语言模型（MLLM）在跨模态理解上取得了显著进展，但仍然面临严重的**对象幻觉（Object Hallucination）** 问题：模型生成与图像内容不一致的虚假描述。

SENTINEL 的核心发现是幻觉具有**传播性**：

1. **幻觉随文本长度增长**：通过分析 300 张 COCO 图片的描述，发现随着生成文本变长，真实对象出现频率下降，幻觉对象出现频率上升（Fig. 2a）
2. **早期干预可遏制传播**：如果在第二句就消除幻觉对象，后续句子的幻觉显著减少、真实对象增多（Fig. 2b）

这一发现揭示了一个关键策略：**在幻觉首次出现时就进行干预，而非等到生成完成后再修正**。

### 1.2 现有方法的不足

| 类别 | 代表方法 | 问题 |
| --- | --- | --- |
| 增强解码 | VCD, OPERA, DoLa | 推理时引入额外计算开销，增加延迟 |
| 偏好学习（大模型/人工标注） | HA-DPO, RLHF-V, HSA-DPO | 依赖 GPT-4 重写或人工标注，成本高昂 |
| 偏好学习（输出重写） | POVID, HA-DPO | 重写引入**分布偏移**——训练数据的文风/表达与模型原始输出不一致 |
| 无训练编辑 | Nullu, [HIME](HIME_2026.md) | 不涉及偏好学习，直接编辑权重 |

SENTINEL 要解决的核心问题是：

> **如何在不依赖大规模外部模型或人工标注的前提下，高效构建高质量、域内一致的偏好数据，并在幻觉首次出现时精准干预？**

### 1.3 域内数据的重要性

论文通过实验证明（Tab. 2），用 GPT-4 重写偏好数据会导致性能下降：

| 数据类型 | Object HalBench Resp.↓ | Ment.↓ | MM-Vet↑ |
| --- | --- | --- | --- |
| 域内数据（原始采样） | **4.3** | **2.6** | **32.6** |
| GPT-4 重写数据 | 4.8 | 2.9 | 31.3 |

分析发现：重写数据偏离模型原始输出分布，导致训练时正负样本区分度降低、损失收敛更慢。**保持训练数据与模型输出的文风一致性至关重要。**

---

## 二、预备知识

### 2.1 Direct Preference Optimization (DPO)

DPO 直接从偏好数据中学习，无需训练奖励模型。给定输入 $\boldsymbol{x} = [\boldsymbol{v}, \boldsymbol{q}]$（图像+提示）、正样本 $\boldsymbol{y}_w$、负样本 $\boldsymbol{y}_l$，DPO 损失为：

$$\mathcal{L}_{\text{DPO}}(\boldsymbol{\theta}) = -\mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}_w, \boldsymbol{y}_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\boldsymbol{\theta}}(\boldsymbol{y}_w | \boldsymbol{x})}{\pi_{\text{ref}}(\boldsymbol{y}_w | \boldsymbol{x})} - \beta \log \frac{\pi_{\boldsymbol{\theta}}(\boldsymbol{y}_l | \boldsymbol{x})}{\pi_{\text{ref}}(\boldsymbol{y}_l | \boldsymbol{x})} \right) \right]$$

其中 $\pi_{\boldsymbol{\theta}}$ 是策略模型，$\pi_{\text{ref}}$ 是参考模型，$\beta$ 控制偏离程度。

### 2.2 开放词汇目标检测

SENTINEL 使用两个开放词汇检测器进行交叉验证：

- **GroundingDINO**：基于 grounded pre-training 的检测器，接受文本查询定位图像中的对象
- **YOLO World**：基于 YOLO 架构的实时开放词汇检测器

两者交叉验证的核心逻辑：**都确认存在→事实性对象；都确认不存在→幻觉对象；一方冲突→不确定，忽略**。

---

## 三、核心方法

SENTINEL 包含三个核心模块，流程如下：

> **输入**：图像 $v$，提示 $q$，上下文 $c$（初始为空）
>
> **输出**：偏好训练数据 $(v, q, c, \boldsymbol{y}_w^+, \boldsymbol{y}_l)$

**while** 模型未生成 $\langle\text{/s}\rangle$：

1. **域内候选采样**：在 $(v, q, c)$ 条件下采样 $n$ 个候选句子
2. **对象提取**：SceneGraphParser 解析三元组 → 提取名词实体
3. **对象存在验证**：GroundingDINO + YOLO World 交叉验证
4. **偏好数据构造**：非幻觉句子 → $\boldsymbol{y}_w^+$，幻觉句子 → $\boldsymbol{y}_l$
5. **迭代上下文自举**：将 $\boldsymbol{y}_w^+$ 追加到 $c$ 中，继续下一轮
6. 用 **C-DPO** 训练模型

### 3.1 域内候选自举（In-domain Candidate Bootstrapping）

#### 3.1.1 域内候选采样

对当前模型使用**采样解码**（而非贪心）生成 $n$ 个候选句子（到句号截止）。关键设计：正样本 $\boldsymbol{y}_w$ 和负样本 $\boldsymbol{y}_l$ 都来自**同一模型的采样分布**，确保文风和语言结构一致——这正是"域内"（in-domain）的含义。

#### 3.1.2 对象提取

使用 SceneGraphParser 将候选句子转化为三元组。例如：

> "A little black cat sits on a chair next to a table."

解析为：(cat, is, little)、(cat, is, black)、(cat, sit on, chair)、(chair, next to, table)

从中提取名词实体（cat, chair, table），并用 SpaCy + NLTK WordNet Lemmatizer 过滤非具体名词（如 feeling, attribute, event 等抽象词类）。

**优势**：仅依赖轻量 NLP 工具，无需 GPT-4 或 LLaMA-70B 等大模型辅助。

#### 3.1.3 对象存在验证

使用 GroundingDINO + YOLO World 对提取的对象在图像中做交叉检测：

| 检测结果 | 分类 | 处理方式 |
| --- | --- | --- |
| 两个检测器都确认**存在** | 事实性 (Factual) | 可用作正样本 |
| 两个检测器都确认**不存在** | 幻觉 (Hallucinated) | 用作负样本 |
| 两个检测器**结果矛盾** | 不确定 (Uncertain) | **忽略**，不参与训练 |

包含幻觉对象的句子标记为"幻觉句"，仅包含事实性对象的句子标记为"非幻觉句"。

**为什么忽略不确定对象？** 消融实验（Tab. 9）表明，将不确定对象视为事实性或幻觉都会引入噪声，降低训练效果。

### 3.2 上下文感知偏好数据生成

#### 3.2.1 偏好数据构造

偏好数据由五元组 $(v, q, c, \boldsymbol{y}_w^+, \boldsymbol{y}_l)$ 组成，其中 $c$ 是**上下文**（已生成的所有句子，不含当前句）。

正样本 $\boldsymbol{y}_w$ 被进一步细分为两类：

- **上下文相关正样本** $\boldsymbol{y}_w^+$：描述的对象在上下文 $c$ 中有提及（与上下文有强相关性）
- **上下文无关正样本** $\boldsymbol{y}_w^-$：描述的对象在上下文中没有提及

论文发现（Tab. 3），使用 $\boldsymbol{y}_w^+$ 能有效缓解幻觉且不损害通用能力；而混入 $\boldsymbol{y}_w^-$ 会导致性能下降。

| 正样本类型 | 数据量 | Object HalBench Resp.↓ | ScienceQA↑ | MM-Vet↑ |
| --- | --- | --- | --- | --- |
| $\boldsymbol{y}_w^+$ 100% | 8.6K | **4.3** | **69.2** | **32.6** |
| $\boldsymbol{y}_w^+$ 50% + $\boldsymbol{y}_w^-$ 50% | 10.0K | 4.8 | 69.0 | 32.0 |
| $\boldsymbol{y}_w^-$ 100% | 14.0K | 4.6 | 68.7 | 31.6 |

**直觉**：上下文相关正样本蕴含了更丰富的上下文信号，引导模型在生成时优先关注与已描述内容连贯的真实对象，而非"跳跃"到无关或幻觉对象。

#### 3.2.2 迭代上下文自举（Iterative Contextual Bootstrapping, ICB）

这是 SENTINEL 的关键设计之一。给定当前上下文 $c_i$，生成偏好数据对后，将非幻觉正样本 $\boldsymbol{y}_w^+$ 追加到上下文中，构造新上下文 $c_{i+1} = c_i + \boldsymbol{y}_w^+$，然后基于新上下文继续采样和构造偏好数据。

$$c_0 = \text{""（空上下文）}$$

$$\xrightarrow{\text{采样}} \text{构造偏好对 } (v, q, c_0, \boldsymbol{y}_w^+, \boldsymbol{y}_l) \xrightarrow{\text{追加到数据集}}$$

$$c_1 = c_0 + \boldsymbol{y}_w^+$$

$$\xrightarrow{\text{采样}} \text{构造偏好对 } (v, q, c_1, \boldsymbol{y}_w^+, \boldsymbol{y}_l) \xrightarrow{\text{追加到数据集}}$$

$$c_2 = c_1 + \boldsymbol{y}_w^+ \quad \rightarrow \quad \cdots \text{（直到模型生成 } \langle\text{/s}\rangle \text{）}$$

**ICB 的作用**：确保偏好数据覆盖**不同长度的上下文**，使模型在各种生成阶段都能区分幻觉与非幻觉内容，增强鲁棒性。

消融实验（Tab. 7）验证了 ICB 的有效性：

| 配置 | Object HalBench Resp.↓ | AMBER CHAIR↓ | MM-Vet↑ |
| --- | --- | --- | --- |
| w/ ICB | **4.3** | **2.9** | **32.6** |
| w/o ICB | 5.3 | 3.1 | 31.8 |

#### 3.2.3 上下文的选择策略

上下文用**非幻觉句子**填充，还是用幻觉句子或贪心解码的自然句子？Tab. 4 给出了对比：

| 上下文策略 | Object HalBench Resp.↓ | AMBER Hal↓ |
| --- | --- | --- |
| 非幻觉上下文 | **4.3** | **14.6** |
| 自然上下文（贪心解码） | 8.6 | 15.6 |
| 幻觉上下文 | 14.3 | 18.6 |

**启示**：用非幻觉句子构建上下文最优——这与"早期干预"的核心理念一致：保证上下文无幻觉，模型才能学会在干净上下文中继续生成无幻觉内容。

### 3.3 上下文感知偏好学习（C-DPO）

SENTINEL 提出 Context-aware DPO (C-DPO)，将上下文 $c$ 纳入输入但**不参与损失计算**：

$$\mathcal{L}_{\text{C-DPO}}(\boldsymbol{\theta}) = -\mathbb{E}_{(\boldsymbol{x}', \boldsymbol{y}_w^+, \boldsymbol{y}_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\boldsymbol{\theta}}(\boldsymbol{y}_w^+ | \boldsymbol{x}')}{\pi_{\text{ref}}(\boldsymbol{y}_w^+ | \boldsymbol{x}')} - \beta \log \frac{\pi_{\boldsymbol{\theta}}(\boldsymbol{y}_l | \boldsymbol{x}')}{\pi_{\text{ref}}(\boldsymbol{y}_l | \boldsymbol{x}')} \right) \right]$$

其中 $\boldsymbol{x}' = [\boldsymbol{v}, \boldsymbol{q}, \boldsymbol{c}]$。

**为什么 mask 上下文？**

上下文 $c$ 在正负样本中完全相同。由于自回归模型的特性，相同前缀的 log 概率（logps）在正负样本的前向传播中完全一致，所以上下文部分的梯度在正负样本间**互相抵消**，不影响参数更新。mask 掉只是减少不必要的计算和数值误差。

**C-DPO vs 标准 DPO 的对比**（Tab. 11）：

| 方法 | Resp.↓ | Ment.↓ | MM-Vet↑ |
| --- | --- | --- | --- |
| C-DPO (8.6K) | **4.3** | **2.6** | **32.6** |
| 标准 DPO (8.6K) | 10.1 | 5.5 | 31.7 |

标准 DPO 用完整描述作为正负样本，正负样本差异过大，导致 logps 波动剧烈、训练不稳定、收敛更慢。C-DPO 聚焦在幻觉首次出现处的**句子级差异**，梯度更稳定。

---

## 四、实验结果

### 4.1 实验设置

**基线模型**：LLaVA-v1.5-7B/13B（公平对比设置）；扩展实验：LLaVA-v1.6、Qwen2-VL-2B/7B、Qwen2.5-VL-7B

**训练设置**：
- 数据来源：Visual Genome 约 4K 图像，用 SENTINEL 流程自动构建 8.6K（7B）/ 7.0K（13B）偏好样本
- 训练：C-DPO + LoRA (rank=128, α=256)，1 epoch，lr=2e-6（7B）/ 3e-6（13B）
- 优化器：AdamW，ZeRO Stage 2

**评估基准**：
- 幻觉：Object HalBench (Resp./Ment.)、AMBER (CHAIR/Hal/Cog)、HallusionBench
- 通用能力：VQAv2、TextVQA、ScienceQA、MM-Vet

### 4.2 主实验结果（LLaVA-v1.5-7B）

| 方法 | 类别 | Resp.↓ | Ment.↓ | CHAIR↓ | VQAv2↑ | TextVQA↑ | ScienceQA↑ | MM-Vet↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | - | 52.7 | 28.0 | 8.4 | **78.5** | **58.2** | 66.8 | 31.0 |
| VCD | 解码 | 51.3 | 25.9 | 9.1 | 77.0 | 56.1 | 68.7 | 29.8 |
| OPERA | 解码 | 45.3 | 22.9 | 6.5 | 78.2 | **58.2** | 68.2 | 30.3 |
| DoLa | 解码 | 44.0 | 25.1 | 6.2 | 76.3 | 56.6 | 67.5 | 30.8 |
| HA-DPO | 偏好 | 37.0 | 20.9 | 6.7 | 77.6 | 56.7 | 69.7 | 30.6 |
| POVID | 偏好 | 33.4 | 16.6 | 5.3 | 77.2 | 56.6 | 68.8 | 31.8 |
| RLAIF-V | 偏好 | 7.8 | 4.2 | 2.8 | 75.2 | 55.1 | 68.2 | 29.9 |
| TPO | 偏好 | 5.6 | 3.2 | 3.6 | 75.9 | 55.3 | 67.1 | 25.7 |
| **SENTINEL** | **偏好** | **4.3** | **2.6** | **2.9** | 78.4 | **58.2** | **69.2** | **32.6** |

**关键观察**：

1. **幻觉率降低约 92%**：Resp. 从 52.7 降至 4.3，Ment. 从 28.0 降至 2.6，超越前 SOTA TPO（5.6/3.2）
2. **通用能力不降反升**：VQAv2（78.4 vs 78.5）和 TextVQA（58.2）几乎无损，ScienceQA（69.2 vs 66.8）和 MM-Vet（32.6 vs 31.0）反而提升——这与其他偏好学习方法（如 RLAIF-V、TPO）形成鲜明对比，后者通常以牺牲通用能力为代价
3. **六类幻觉全面改善**（AMBER 判别部分）：Existence (+6.3)、Attribute (+3.6)、State (+3.6)、Number (+4.9)、Action (+1.0)、Relation (+2.9)

### 4.3 LLaVA-v1.5-13B 结果

| 方法 | Resp.↓ | Ment.↓ | CHAIR↓ | Hal↓ | Cog↓ | MM-Vet↑ |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 46.0 | 23.0 | 6.9 | 31.9 | 3.3 | 36.0 |
| HSA-DPO | 5.3 | 3.2 | 2.1 | 13.4 | 1.2 | 33.7 |
| **SENTINEL** | **3.3** | **1.9** | **2.7** | **11.7** | **0.9** | **36.2** |

在 13B 模型上，SENTINEL 持续保持最低幻觉率且 MM-Vet 超越 baseline（36.2 vs 36.0），而 HSA-DPO 下降到 33.7。

### 4.4 跨模型泛化（Tab. 19）

| 模型 | Resp. (base→SENTINEL)↓ | Ment.↓ | MM-Vet↑ |
| --- | --- | --- | --- |
| LLaVA-v1.6-vicuna-7B | 15.3→5.0 | 10.1→3.4 | 40.9→45.4 |
| LLaVA-v1.6-vicuna-13B | 13.7→4.0 | 7.7→2.6 | 47.8→48.5 |
| Qwen2-VL-2B-Instruct | 15.3→2.3 | 8.6→1.7 | 49.4→49.8 |
| Qwen2-VL-7B-Instruct | 14.3→4.8 | 8.5→4.0 | 62.7→62.8 |
| Qwen2.5-VL-7B-Instruct | 15.0→4.7 | 9.2→2.8 | 72.0→72.2 |

**模型无关性**：SENTINEL 在 LLaVA 和 Qwen 两大系列、2B 到 13B 不同规模上均有效，且通用能力保持或提升。

### 4.5 与已有方法的互补性

SENTINEL 可以与已有偏好学习方法叠加使用。将 6K SENTINEL 数据加入 HA-DPO 的 4.4K GPT 生成数据中联合训练（Tab. 5）：

| 方法 | Resp.↓ | Ment.↓ | AMBER F1↑ | MM-Vet↑ |
| --- | --- | --- | --- | --- |
| HA-DPO 单独 | 37.0 | 20.9 | 78.0 | 30.6 |
| HA-DPO + SENTINEL (6K) | **8.0** | **4.6** | **84.2** | **33.5** |

域内数据与 GPT 重写数据形成互补，大幅降低幻觉的同时显著提升通用能力。

### 4.6 关键消融实验

#### 交叉验证的有效性

| 检测器 | Resp.↓ | Ment.↓ |
| --- | --- | --- |
| OmDet 单独 | 19.3 | 9.9 |
| GroundingDINO 单独 | 14.3 | 7.7 |
| YOLO World 单独 | 12.3 | 6.9 |
| **GroundingDINO + YOLO World** | **6.6** | **3.8** |

两个检测器交叉验证显著优于单一检测器，有效减少假阳性。

#### 数据规模效应

| 数据量 | Resp.↓ | Ment.↓ |
| --- | --- | --- |
| 1K | ~20 | ~10 |
| 2K | ~14 | ~7 |
| 4K | ~8 | ~5 |
| 8.6K | **4.3** | **2.6** |

幻觉率随数据规模近似线性下降，展现出良好的可扩展性。由于数据构建不依赖大模型或人工标注，扩展成本极低。

---

## 五、局限性与未来方向

1. **仅针对对象幻觉**：SENTINEL 的对象检测验证机制天然适用于对象存在性的判断，对属性幻觉、关系幻觉、动作幻觉等更细粒度的类型需要更通用的验证工具
2. **缺乏时空信息**：无法有效处理视频 MLLM 中需要长程推理的幻觉问题
3. **依赖检测器质量**：交叉验证虽然缓解了假阳性，但检测器的覆盖范围和精度仍然是数据质量的瓶颈。更强的检测器（如 Tab. 8 所示）直接带来更好的训练效果
4. **句子级粒度**：以句号为分隔的句子级干预可能在某些复杂句式中不够精细

---

## 六、个人思考

### 6.1 "早期干预"的深层洞察

SENTINEL 最有价值的贡献或许不是具体方法，而是**"幻觉在生成早期萌发并逐步放大"** 这一经验发现。这暗示了自回归模型中幻觉的传播机制：一旦某句话引入了虚假对象，后续生成会将其作为"已建立的上下文"，进一步编造相关细节。这与 LLM 中的"滚雪球幻觉"现象高度一致。

### 6.2 与 [HIME](HIME_2026.md) 的互补性

SENTINEL（训练时偏好学习）和 HIME（训练无关的权重编辑）分别从训练和推理两个维度缓解幻觉，理论上可以叠加：

| 维度 | SENTINEL | HIME |
| --- | --- | --- |
| 干预时机 | 训练时（偏好学习） | 离线（权重编辑） |
| 推理开销 | 零 | 零 |
| 是否需要训练 | 是（LoRA 微调） | 否 |
| 幻觉定位粒度 | 句子级（生成过程中） | 层级（decoder 层维度） |
| 数据依赖 | 自举采样 + 检测器 | 对比数据集（LURE） |

两者的结合可能进一步压低幻觉率：HIME 从模型参数层面移除幻觉子空间，SENTINEL 从训练数据层面教会模型区分幻觉与事实。

### 6.3 域内数据的普适意义

SENTINEL 证明了一个重要原则：**训练数据应该与模型输出分布保持一致**。这不仅适用于幻觉缓解，对 DPO/RLHF 的通用实践也有启示——用外部大模型重写的 preference data 虽然标注质量更高，但引入的分布偏移可能抵消质量优势。在线采样 + 自动验证可能是更好的范式。

### 6.4 潜在改进

- **多轮自迭代**：用 SENTINEL 训练后的模型重新采样构建更高质量的偏好数据，进行多轮迭代优化（类似 RLAIF-V 的做法但用域内数据）
- **细粒度验证**：引入 VLM-as-Judge（如 GPT-4V 或开源 VLM）做属性/关系级别的验证，扩展到更多幻觉类型
- **与推理时方法结合**：SENTINEL + VCD/OPERA 可能进一步压低幻觉率（正交策略叠加）

---

## 参考

- **HA-DPO** (Zhao et al., 2023)：用 GPT-4 重写构建偏好数据，SENTINEL 证明域内数据优于重写数据
- **RLAIF-V** (Yu et al., 2024)：迭代对齐 + "Feedback from Peer" 策略，是 SENTINEL 的直接对比方法
- **TPO** (He et al., 2024)：主题级自纠正，SENTINEL 的前 SOTA 对比基线
- **VCD** (Leng et al., 2024)：视觉对比解码，代表推理时干预方法
- **DPO** (Rafailov et al., 2023)：直接偏好优化，SENTINEL 的 C-DPO 是其上下文感知扩展
- **[HIME](HIME_2026.md)** (Akl et al., 2026)：层自适应权重编辑，与 SENTINEL 从不同维度缓解幻觉，可互补
