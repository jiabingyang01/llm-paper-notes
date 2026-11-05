# StarVLA：面向视觉-语言-动作模型的乐高式模块化研发框架

> **论文信息**：Jinhui Ye, Ning Gao, Yilun Chen, Weiyu Guo, Zixuan Wang, Yuxing Chen, Fangjing Wang, Senqiao Yang, Chengyao Wang, Yuqi Liu, Meng Chu, Changsheng Lu, Pengguang Chen, Shu Liu, Jiaya Jia 等 (StarVLA Community / HKUST / Von Neumann Institute)
> **发表期刊/会议**：arXiv 2026 (`arXiv:2604.05014`)
> **代码仓库**：[starVLA/starVLA](https://github.com/starVLA/starVLA)

## 1. 核心定位与研究动机

随着视觉-语言-动作（Vision-Language-Action, VLA）模型与世界动作模型（World Action Models, WAM）的快速发展，具身智能领域涌现出大量异构架构（自回归离散化、并行回归、扩散策略、流匹配等）。然而，现有研究在三个层面陷入严重的碎片化（“巴别塔困境”）：
1. **架构层碎片化**：VLM 基座与动作解码头（Action Head）高度耦合，难以在固定同一视觉-语言基座（如 `Qwen3-VL-4B`）的前提下公平对比不同动作解码范式；
2. **系统与训练层碎片化**：跨本体预训练（Cross-Embodiment Learning）与多模态协同训练（VLM + Action Co-Training）往往针对特定模型硬编码，缺乏可复用的统一流水线；
3. **评测层碎片化**：各工作在 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1、BEHAVIOR-1K 的不同子集与不同评测协议下报告结果，导致横向复现极其困难。

**StarVLA** 提出了 **“乐高式（Lego-like）”骨干-动作头解耦架构** 与统一的策略中心形式化框架，将 VLM 基座与世界模型基座（如 Cosmos）、四种主流动作解码范式、多模态协同训练以及五大主流仿真/真机基准统一在同一抽象接口之下。

## 2. 统一形式化框架与四大动作解码范式

### 2.1 策略中心的统一数学抽象

StarVLA 将所有 VLA 与 WAM 范式统一建模为从多模态历史观测与语言指令到未来动作块（Action Chunk）及可选辅助输出的映射策略：

$$
\pi(\mathbf{a}_{t:t+k}, \mathbf{y}_{\mathrm{aux}} \mid \mathbf{x}_{\le t}, \ell)
$$

其中：
- $\mathbf{x}_{\le t} = \{o_{\le t}^{\mathrm{vis}}, o_{\le t}^{\mathrm{depth}}, o_{\le t}^{\mathrm{tactile}}, \dots\}$ 表示截至时间步 $t$ 的多模态观测历史；
- $\ell$ 为自然语言任务指令；
- $\mathbf{a}_{t:t+k}$ 为预测的未来 $k$ 步动作块；
- $\mathbf{y}_{\mathrm{aux}}$ 为未来时域上的可选辅助输出（例如未来视觉帧 $o_{t+1:t+k}^{\mathrm{vis}}$ 或子目标规划文本 $\ell_{\mathrm{plan}}$）。

对应的统一训练目标分解为：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{action}} + \mathcal{L}_{\mathrm{aux}}
$$

在该框架下：
- **直接策略建模（Direct VLA）**：令 $\mathcal{L}_{\mathrm{aux}} = 0$，仅直接监督动作生成；
- **基于 VLM 的协同推理 VLA**：引入语言对齐辅助损失 $\mathcal{L}_{\mathrm{aux}}$（如子任务拆解、2D/3D 空间接地或思维链推理）；
- **基于世界模型的 VLA（WM-based VLA）**：引入未来视觉帧或潜空间状态预测作为 $\mathcal{L}_{\mathrm{aux}}$，为动作生成提供前向动力学先验。

### 2.2 四种可插拔动作解码范式（StarVLA 四大变体）

在固定同一 VLM 基座（默认采用 `Qwen3-VL-4B`）的前提下，StarVLA 实现了四种标准化的动作解码头：
1. **`StarVLA-FAST`（自回归频域离散 Token 范式）**：基于 FAST（Frequency-space Action Sequence Tokenization）将连续动作块通过离散余弦变换（DCT）与 BPE 压缩为离散动作 Token，直接由 VLM 自回归预测交叉熵损失；
2. **`StarVLA-OFT`（并行回归范式）**：遵循 OpenVLA-OFT 设计，在输入序列中追加可学习的动作查询 Token（Action Queries），启用双向注意力掩码并通过 MLP 头以 $L_1$ 回归损失单次前向并行输出整段动作块；
3. **`StarVLA-GR00T`（潜特征条件流匹配范式）**：遵循 GR00T-N1 的双系统架构，提取 VLM 隐状态作为条件，通过 Cross-Attention 注入独立的 Diffusion Transformer（DiT）动作专家，最小化流匹配速度场损失 $\mathcal{L}_{\mathrm{FM}}$；
4. **`StarVLA-π`（联合自注意力流匹配范式）**：遵循 $\pi_0 / \pi_{0.5}$ 的 Mixture-of-Transformers（MoT）架构，让 VLM Token 与动作专家 Token 在每一层通过分块因果掩码（Blockwise Causal Attention）直接共享 Key/Value 进行深层交互。

## 3. 训练范式与基准评测结果

### 3.1 四大标准基准上的单模型统一微调对比（基于 Qwen3-VL-4B）

StarVLA 提供了不依赖繁琐数据工程的极简可复现基线配方。在 **RoboTwin 2.0**（50 个双臂任务统一训练一个策略，Clean 与 Randomized 各 5,000 次评测）、**LIBERO**、**LIBERO-Plus（7-OOD）** 与 **RoboCasa-GR1** 上的核心结果如下：

| 模型变体 / 基线方法 | 参数量 | RoboTwin 2.0 Clean (%) | RoboTwin 2.0 Random (%) | LIBERO Avg. (%) | LIBERO+ 7-OOD (%) | RoboCasa-GR1 Avg. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\pi_0$** | 3.3B | 65.9 | 58.4 | 94.2 | 53.6 | 36.3 |
| **$\pi_{0.5}$** | 3.3B | 82.7 | 76.8 | 96.9 | 84.4 | 39.6 |
| **X-VLA** | 0.9B | 72.9 | 72.8 | 98.1 | - | - |
| **LingBot-VLA** | - | 88.6 | 86.7 | - | - | - |
| **`StarVLA-FAST`** | 4.0B | 72.5 | 83.2 | - | - | - |
| **`StarVLA-OFT`** | 4.0B | 88.2 | 88.3 | - | - | - |
| **`StarVLA-GR00T`** | 4.0B | 88.0 | 88.5 | **96.6** | **74.1** | **40.6** |
| **`StarVLA-π`** | 4.0B | 88.1 | **88.8** | - | - | - |

> 注：在无大规模预训练、仅使用单数据集训练的场景下，亦可观测到 StarVLA（4.0B）在 RoboTwin 2.0 上取得 Clean $46.5\%$ / Rand. $3.2\%$（平均 $24.8\%$），而在遵循官方 LeRobot 干净+随机化联合训练配方时，`StarVLA-OFT`、`StarVLA-GR00T` 与 `StarVLA-π` 均在 Clean 与 Randomized 上突破 **88%** 成功率。

### 3.2 多模态协同训练（VLM + Action Co-Training）与全基准合一训练

1. **多模态协同训练（Co-Training）**：StarVLA 支持在训练动作专家的同时混合通用视觉-语言问答与空间接地数据，防止微调过程中的灾难性遗忘（Catastrophic Forgetting），显著提升长程指令遵循与未见物体泛化能力；
2. **All-Bench-in-One 跨本体联合训练**：StarVLA 验证了将 LIBERO、SimplerEnv、RoboTwin 2.0、RoboCasa-GR1 等异构动作空间的数据打包在单一模型中联合训练的可行性，证明多基准联合训练可实现正向技能迁移。

### 3.3 大规模分布式训练系统吞吐与扩展性报告

作者基于 `StarVLA-GR00T`（`Qwen3-VL-4B` 基座，A100 80GB 集群）系统评测了单机批大小与多机扩展效率（固定单卡 Batch = 8，统计 100K 步墙钟耗时）：

| GPU 数量 | 全局批大小（Global Batch） | 100K 步总耗时 | 单步耗时（s/step） | 样本吞吐率（Samples/s） | 多机并行扩展效率 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8 GPUs（1 节点）** | 64 | 20:25:48 | $0.735\text{ s}$ | $87.0$ | $100.0\%$ |
| **16 GPUs（2 节点）** | 128 | 23:36:00 | $0.850\text{ s}$ | $150.7$ | $86.7\%$ |
| **32 GPUs（4 节点）** | 256 | 24:58:45 | $0.899\text{ s}$ | $284.7$ | $81.9\%$ |
| **64 GPUs（8 节点）** | 512 | 25:40:59 | $0.925\text{ s}$ | $553.8$ | $79.6\%$ |
| **128 GPUs（16 节点）** | 1024 | 25:35:26 | $0.921\text{ s}$ | $1111.5$ | $79.9\%$ |
| **256 GPUs（32 节点）** | 2048 | 25:51:41 | $0.931\text{ s}$ | **2200.0** | **79.1%** |

**核心系统启示**：从单节点（8 卡）扩展到 4 节点（32 卡）时，跨节点通信引入了一次性延迟（从 $0.735\text{ s/step}$ 增至 $0.899\text{ s/step}$）；但当规模继续从 64 卡扩展至 **256 卡** 时，单步耗时稳定在 $\sim 0.93\text{ s}$，并行扩展效率稳定维持在 **79%~80%**，样本吞吐率线性攀升至 **2,200 samples/s**。

## 4. 总结与核心贡献

StarVLA 通过清晰的“基座-动作头”模块化抽象消除了 VLA 与 WAM 研究中的工程壁垒，证明了在统一的现代 VLM 基座（如 `Qwen3-VL`）与标准化训练管线下，并行回归（OFT）与流匹配（GR00T / $\pi$）均可取得极高且相近的操控成功率，为具身智能社区提供了高吞吐、可复现的研发底座。
