# OTTER：文本感知视觉特征提取的 VLA 模型

> **论文**：*OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction*
>
> **作者**：Huang Huang\*, Fangchen Liu\*, Letian Fu\*, Tingfan Wu, Mustafa Mukadam, Jitendra Malik, Ken Goldberg, Pieter Abbeel
>
> **机构**：UC Berkeley、Meta AI
>
> **发布时间**：2025年3月
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.03734) | [项目主页](https://ottervla.github.io/)
>
> **发表会议**：ICML 2025

---

## 一句话总结

OTTER 冻结预训练 CLIP 编码器，利用文本-视觉余弦相似度选择性提取与语言指令语义对齐的视觉 patch 特征（而非独立传递所有特征），仅训练温度参数和轻量策略网络（~12M），在真实机器人 4 种操作原语上实现 67-77% 零样本泛化成功率，显著超越 Octo（4%）、OpenVLA（0.6%）等基线。

---

## 一、问题与动机

### 1.1 现有 VLA 的困境

现有 VLA 模型的典型流程是**直接特征传递（Direct Feature Passing）**：分别编码视觉和语言特征，独立传递给策略网络。这带来两个核心问题：

1. **微调破坏预训练对齐**：RT-2、OpenVLA 在机器人数据上微调 VLM，但机器人数据的语义多样性远不如 LAION 等大规模视觉-语言数据集，微调容易过拟合并退化泛化性能
2. **策略网络负担过重**：策略需同时完成"理解哪些视觉区域与任务相关"和"规划动作"，在小规模机器人数据上难以同时学好

这在实际表现中体现为：**训练任务上看似不错，但换个物体/环境就崩溃**。

### 1.2 OTTER 的核心思路

**保留而非破坏预训练 VLM 的对齐能力**：

- **冻结 CLIP 编码器**：完整保留大规模预训练学到的视觉-语言语义理解
- **文本感知视觉特征提取**：先用语言指令"查询"视觉特征，只提取与任务相关的视觉信息
- **解耦任务规划与动作规划**：文本感知特征提取负责"关注什么"（利用预训练对齐），策略网络只需专注"怎么做"

---

## 二、预备知识

### 2.1 CLIP 视觉编码器的注意力结构

ViT-based CLIP 视觉编码器由一系列残差注意力块组成。每个块接收视觉 token $X = [x_{\text{cls}}, x_1, \ldots, x_{h \times w}]^T$：

$$q, k, v = \text{Proj}_{q,k,v}(\text{LN}(X))$$

$$X_{\text{sum}} = X + X_{\text{attn}} = X + \text{Proj}(\text{Attn}(q, k, v))$$

$$X_{\text{out}} = X_{\text{sum}} + \text{FFN}(\text{LN}(X_{\text{sum}}))$$

### 2.2 ClearCLIP 的发现

ClearCLIP (Lan et al., ECCV 2024) 发现：CLIP 最后一层自注意力块的**注意力输出** $X_{\text{attn}}$ 包含比最终输出 $X_{\text{out}}$ 更干净的语义信息。

直觉理解：由于 LayerNorm 沿通道维度操作而非 patch 维度，$\hat{f}_v = \text{LN}_{\text{post}}(f_v)w_v$ 保持线性，因此 $X_{\text{out}}$ 可以分解为三个独立分量：

$$\hat{f}_v = \text{LN}_{\text{post}}(X_{\text{res}})w_v + \text{LN}_{\text{post}}(X_{\text{attn}})w_v + \text{LN}_{\text{post}}(\text{FFN}(\text{LN}(X_{\text{sum}})))w_v$$

其中 $X_{\text{res}}$（残差连接）和 FFN 项引入噪声，降低视觉-语言对齐质量。OTTER 只使用 $X_{\text{attn}}$ 项，获得更干净的语义特征。

---

## 三、核心方法

### 3.1 文本感知视觉特征提取

这是 OTTER 的核心创新。整个过程**不含可学习参数**（除温度 $\tau$），完全利用冻结 CLIP 的预训练对齐。

**Step 1：特征提取与归一化**

从冻结 CLIP 中提取：
- **文本特征** $f_l$（$m$ 个 token）：CLIP 语言编码器的逐 token 输出
- **视觉特征** $f_v$（$n = h \times w$ 个 patch token）：最后一层注意力块的 $X_{\text{attn}}$

分别通过 CLIP 的模态投影矩阵归一化到共享潜在空间：

$$\hat{f}_l = \text{LN}_{\text{final}}(f_l) w_l, \quad \hat{f}_v = \text{LN}_{\text{post}}(f_v) w_v$$

再做 L2 归一化：$\hat{f}_l = \hat{f}_l / \|\hat{f}_l\|_2$，$\hat{f}_v = \hat{f}_v / \|\hat{f}_v\|_2$。

**Step 2：温度加权注意力融合**

$$f_{vl} = \text{softmax}(\hat{f}_l \hat{f}_v^\top / \tau)(\hat{f}_v + PE)$$

其中：
- $\hat{f}_l \hat{f}_v^\top \in \mathbb{R}^{m \times n}$：每个文本 token 与所有视觉 patch 的余弦相似度矩阵
- $\tau$：可学习温度参数（clipped 在 $[0, 100]$），**整个 CLIP 中唯一被训练的参数**
- $PE$：2D sin-cos 位置编码，告知策略网络每个 patch 的空间位置
- $f_{vl} \in \mathbb{R}^{m \times d}$：文本感知视觉特征，每行是与对应文本 token 语义相关的视觉 patch 加权平均

**直觉理解**：softmax 充当选择函数——对于指令 "pick up the green triangle" 中的 "green" token，绿色区域的 patch 获得最大权重；对于 "triangle" token，三角形所在区域被选中。$\tau$ 越小选择越集中（hard selection），$\tau$ 越大分布越均匀（soft selection）。

### 3.2 整体架构

**注意力池化压缩**：将 $f_{vl}$（$m$ 个 token）通过可学习交叉注意力池化压缩为单个 token $f'_{vl}$（$N_q = 4$ 个查询，输出拼接）。同样对文本特征 $f_l$ 做池化得到 $f'_l$。

**本体感受编码**：末端位置 $(x, y, z)$ + 6DoF 旋转向量 + 夹爪状态（共 10 维）通过 FFN 编码为 $f_e$。

**策略网络输入**：每个时间步将 $f'_l$、$f'_{vl}$、$f_e$ 沿通道维度拼接为单个 token $f_t$。

**策略网络**：4 层 8 头因果 Transformer（hidden dim 512），上下文窗口 $T = 12$ 步，每步预测未来 12 步动作。推理时使用 temporal ensembling + receding horizon control，action horizon 为 8 步。

**动作参数化**：delta 末端执行器位姿，10 维（3 平移 + 6DoF 旋转 + 1 夹爪），旋转用 SO(3) 矩阵前两行展平表示。

### 3.3 关键设计总结

| 设计选择 | 具体实现 | 动机 |
| --- | --- | --- |
| 冻结 CLIP | 仅温度 $\tau$ 可学习 | 保留预训练视觉-语言对齐 |
| $X_{\text{attn}}$ 而非 $X_{\text{out}}$ | ClearCLIP 启发 | 更干净的语义特征 |
| 文本感知提取 | 余弦相似度 + softmax 选择 | 解耦任务规划与动作规划 |
| 注意力池化 | 4 个可学习查询 | 压缩为固定长度输入 |
| 轻量策略网络 | 4 层 Transformer，~12M 参数 | 只需学动作规划 |

---

## 四、实验结果

### 4.1 真实机器人——单原语多任务（Pick & Place）

724 条人类遥操作轨迹，100 次训练任务试验 + 70 次未见任务试验：

| 方法 | Training Tasks | Unseen Tasks |
| --- | --- | --- |
| π₀-Fast-Droid | - | 61% ± 5.3% |
| Finetuned Octo | 15% ± 3.4% | 12% ± 3.6% |
| OTTER w.o. CLIP vision | 17% ± 2.9% | 11% ± 2.5% |
| Finetuned OpenVLA | 30% ± 3.9% | 9% ± 3.1% |
| DFP-OTTER | 29% ± 3.7% | 4% ± 1.6% |
| OTTER (Finetune CLIP) | 26% ± 4.0% | 15% ± 3.9% |
| OTTER w.o. $f_e$ | 40% ± 4.0% | 29% ± 4.3% |
| OTTER w.o. $f'_l$ | 57% ± 4.4% | 53% ± 4.6% |
| **OTTER** | **68% ± 4.3%** | **62% ± 4.2%** |
| **OTTER-OXE** | **72% ± 3.9%** | **73% ± 2.8%** |

关键观察：
- OTTER 训练/未见任务**性能差距极小**（68% vs 62%），零样本泛化能力突出
- 微调 CLIP 反而导致性能严重下降（26% vs 68%），验证冻结策略的正确性
- DFP-OTTER（同架构但独立传递特征）在未见任务仅 4%，说明文本感知提取是泛化关键

### 4.2 真实机器人——多原语零样本泛化

4 种原语共 1,185 条轨迹训练，150 次**完全未见任务**试验：

| 方法 | Pouring | Drawer | Poking | Pick & Place | Mean |
| --- | --- | --- | --- | --- | --- |
| π₀-Fast-Droid | 0% | 0% | 0% | 61% | 29% ± 3.5% |
| Finetuned π₀-Fast-Droid | 0% | 45% | 27% | 51% | 35% ± 3.8% |
| Finetuned Octo | 0% | 0% | 0% | 5% | 4% ± 1.2% |
| Finetuned OpenVLA | 0% | 0% | 0% | 1% | 0.6% ± 0.5% |
| **OTTER** | **63%** | **50%** | **93%** | **61%** | **67% ± 3.8%** |
| **OTTER-OXE-L** | **77%** | **75%** | **93%** | **75%** | **77% ± 3.3%** |

所有基线在 pouring/drawer/poking 上几乎完全失败（0%），而 OTTER 在所有原语上均保持 50-93% 成功率。OTTER-L（更大策略网络）和 OTTER-OXE（OXE 预训练）进一步提升性能，说明 OTTER 能沿模型规模和数据规模两个轴 scaling。

### 4.3 仿真实验（LIBERO）

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | Train Avg | Unseen |
| --- | --- | --- | --- | --- | --- |
| Finetuned Octo | 79% | 86% | 85% | 83% | 26% |
| Finetuned OpenVLA | 85% | 88% | 79% | 84% | 29% |
| DFP-OTTER | 78% | 80% | 82% | 80% | 28% |
| **OTTER** | **84%** | **89%** | **82%** | **85%** | **59%** |

训练任务性能接近，但**未见任务上 OTTER 领先超 30 个百分点**。

### 4.4 消融实验

| 消融 | LIBERO-Object | Unseen |
| --- | --- | --- |
| OTTER w.o. CLIP vision | 80% | 29% |
| OTTER w.o. $f_e$ | 79% | 48% |
| OTTER w.o. $f'_l$ | 71% | 49% |
| **OTTER (full)** | **89%** | **59%** |

- **去掉 CLIP 视觉编码器**（从头训练 ViT）：训练任务相当，未见任务暴跌到 29%——预训练 VLM 对泛化至关重要
- **去掉本体感受 $f_e$**：训练/未见任务均下降约 10%——本体提供关键空间物理信息
- **去掉文本 token $f'_l$**：物体不够逼真时，$f'_l$ 提供补充信息帮助定位正确物体

### 4.5 CLIP 规模缩放

从 ViT-B/32 → ViT-B/16 → ViT-L/14：训练任务成功率 +27.5%，未见任务成功率 +39.3%。说明 OTTER 能有效利用更强的视觉-语言编码器。

### 4.6 特征提取方式消融

| 方法 | Unseen 成功率 |
| --- | --- |
| DFP-OTTER (CLS) | 6% ± 0.8% |
| OTTER (xattn) | 2% ± 0.5% |
| **OTTER** | **62% ± 4.2%** |

使用 `[CLS]` token 或标准交叉注意力池化均完全失败——**余弦相似度 + softmax 的参数无关选择机制是泛化的关键**。

---

## 五、局限性与未来方向

1. **构型限制**：动作参数化依赖 SE(3) 变换，难以扩展到多指灵巧手等不易用 SE(3) 表示的构型
2. **长时域任务未探索**：实验聚焦短时域操作原语，未充分验证在长时域复杂任务上的表现
3. **CLIP 能力天花板**：零样本泛化能力受限于 CLIP 的零样本识别能力——CLIP 不认识的物体 OTTER 也会失败

---

## 六、个人思考

### 6.1 "冻结比微调好"的启示

OTTER 的核心洞察直觉但深刻：**在数据稀缺场景下，利用预训练模型的方式决定了泛化能力**。微调 CLIP 在真实实验中从 68% 暴跌到 26%（训练任务）和 15%（未见任务），而冻结 CLIP 反而更好。这与 VLM 幻觉缓解中的发现（如 EFUF、LessIsMore）形成有趣呼应——盲目微调可能破坏模型原有的能力。

### 6.2 与其他 VLA 基础模型的对比

- **π₀/π₀.₅**：Flow Matching + VLM 微调，数据需求大但上限高
- **BridgeVLA**：通过输入-输出对齐实现样本效率，但仍需训练 VLM 预测热力图
- **GR-3**：VL 数据协同训练增强泛化，但仍需微调
- OTTER 的策略网络仅 ~12M 参数，几乎所有"理解"都来自冻结 CLIP，这使得它在极小数据集（724 条轨迹）上就能实现强泛化。但代价是依赖 CLIP 的能力上限。

### 6.3 ClearCLIP 的跨领域价值

OTTER 从 ClearCLIP（开放词汇分割）借用了 $X_{\text{attn}}$ 特征的思想，成功迁移到机器人控制。这说明对基础模型内部特征的深入理解可以在看似无关的下游任务中产生巨大价值。DFP-OTTER CLS（6%）和 OTTER xattn（2%）的惨烈失败也表明，不是随便用 CLIP 特征就行——**如何提取**至关重要。

### 6.4 数据效率的启示

OTTER 只用 724 条 pick-and-place 轨迹就达到 68%/62% 的训练/未见成功率，而 π₀-Fast-Droid 预训练在 OXE + π 数据集上才达到 61% 未见成功率。这暗示：**在机器人数据稀缺的现实中，充分利用大规模预训练的视觉-语言对齐可能比扩大机器人数据集更高效**。

---

## 参考

- **ClearCLIP** (Lan et al., ECCV 2024)：$X_{\text{attn}}$ 比 $X_{\text{out}}$ 包含更干净的 CLIP 语义特征
- **OpenVLA** (Kim et al., 2024)：微调 Prismatic-7B VLM 的开源 VLA 基线
- **Octo** (Octo Model Team, 2024)：从头训练的 Transformer VLA 策略
- **π₀** (Black et al., 2024)：Flow Matching VLA 基础模型
- **CLIP** (Radford et al., 2021)：对比学习视觉-语言预训练模型
- **FiLM** (Perez et al., AAAI 2018)：早期视觉-语言条件化方法
