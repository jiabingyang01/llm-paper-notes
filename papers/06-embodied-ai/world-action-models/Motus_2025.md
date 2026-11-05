# Motus：统一潜动作世界模型与三模态专家协同框架

> **论文信息**：Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, Jun Zhu (Tsinghua University / Peking University / Horizon Robotics)
> **发表期刊/会议**：arXiv 2025 (`arXiv:2512.13030`)
> **项目主页**：[motus-robotics.github.io/motus](https://motus-robotics.github.io/motus)

## 1. 核心思想与两大关键挑战

构建通用具身智能体要求系统同时具备场景理解、指令遵循、未来动力学推演与精细动作控制能力。然而现有研究普遍面临两大瓶颈：
1. **多模态生成能力碎片化（Challenge 1）**：现有模型往往孤立地建模视觉-语言-动作策略（VLA）、世界模型（WM）、逆动力学模型（IDM）或视频生成模型（VGM），缺少能够同时继承预训练 VLM 语义理解先验与预训练 VGM 物理交互先验的统一网络；
2. **跨本体异构与无标签视频利用受限（Challenge 2）**：不同机器人动作空间维度与物理量纲差异巨大，且海量互联网视频与第一视角人类演示视频缺乏显式动作标签，导致动作专家难以开展真正的大规模预训练。

为此，清华大学朱军团队提出 **Motus（8.0B）** ——一个统一的**潜动作世界模型（Unified Latent Action World Model）**：
- **三模态专家混合架构（Tri-Model Mixture-of-Transformers, MoT）**：将预训练视频生成专家（`Wan 2.2 5B`）、预训练视觉-语言理解专家（`Qwen3-VL-2B`）与同深度的动作专家（Action Expert）通过**三模态联合注意力（Tri-Model Joint Attention）** 深度耦合；
- **UniDiffuser 风格双时间步调度器**：为视频观测流与动作流分配独立的 Rectified Flow 时间步 $(\tau_o, \tau_a)$，使单一网络在推理时可无缝切换为 **5 种基础具身建模模式**（VLA、WM、IDM、VGM、视频-动作联合预测）；
- **基于光流压缩的 14 维潜动作（Optical-Flow Latent Actions）与六层具身数据金字塔**：利用 `DPFlow` 提取帧间光流并通过 `DC-AE` 变分自编码器压缩为与双臂控制维度对齐的 14 维像素级“增量动作（Delta Action）”，配合三阶段渐进式训练，在 RoboTwin 2.0 随机化场景上超越 $\pi_{0.5}$ 达 **+43% 以上**，超越 X-VLA 达 **+14% 以上**。

## 2. 统一多模态生成形式化与三专家 MoT 架构

### 2.1 五种具身生成分布与 UniDiffuser 双时间步切换

记当前视觉观测为 $\mathbf{o}_t$，未来 $k$ 步观测序列为 $\mathbf{o}_{t+1:t+k}$，动作块为 $\mathbf{a}_{t+1:t+k}$，语言指令为 $\ell$。Motus 使用 Rectified Flow 同时建模未来视频块与动作块的速度场 $v_o^\theta, v_a^\theta$，联合优化目标为：

$$
\mathcal{L}_{\text{action}}^\theta = \mathbb{E}_{\tau_a \sim \mathcal{U}(0, T_\tau),\; \epsilon_a \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left\| v_a^\theta - (\epsilon_a - \mathbf{a}_{t+1:t+k}) \right\|_2^2
$$

$$
\mathcal{L}_{\text{obs}}^\theta = \mathbb{E}_{\tau_o \sim \mathcal{U}(0, T_\tau),\; \epsilon_o \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left\| v_o^\theta - (\epsilon_o - \mathbf{o}_{t+1:t+k}) \right\|_2^2
$$

$$
\mathcal{L}^\theta = \mathcal{L}_{\text{action}}^\theta + \mathcal{L}_{\text{obs}}^\theta
$$

通过在推理时设置视频流时间步 $\tau_o$ 与动作流时间步 $\tau_a$ 的初始状态（$0$ 代表已知干净条件，$T_\tau$ 代表从纯高斯噪声开始去噪），Motus 在单一权重下统一涵盖五种推理模式：

| 模式名称 | 条件概率分布 | 视频流时间步 $\tau_o$ | 动作流时间步 $\tau_a$ | 物理含义 |
| :--- | :--- | :---: | :---: | :--- |
| **1. 视频生成模型（VGM）** | $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ | $T_\tau \to 0$（去噪生成） | 固定为 $T_\tau$（纯噪声） | 根据指令与首帧想象未来视频 |
| **2. 世界模型（WM）** | $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{a}_{t+1:t+k})$ | $T_\tau \to 0$（去噪生成） | 固定为 $0$（干净动作输入） | 预测给定动作序列的视觉后果 |
| **3. 逆动力学模型（IDM）** | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_{t:t+k})$ | 固定为 $0$（干净视频输入） | $T_\tau \to 0$（去噪生成） | 从视频帧序列反推底层控制动作 |
| **4. 视觉-语言-动作（VLA）** | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ | 固定为 $T_\tau$（纯噪声） | $T_\tau \to 0$（去噪生成） | 不生成视频，直接快速解码动作 |
| **5. 视频-动作联合预测** | $p(\mathbf{o}_{t+1:t+k}, \mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ | $T_\tau \to 0$（同步去噪） | $T_\tau \to 0$（同步去噪） | 边想象未来视频边协同生成动作 |

### 2.2 三模态联合注意力（Tri-Model Joint Attention）与动作密-视频疏策略

1. **三专家 MoT 异构集成**：
   - **理解专家（Understanding Expert）**：采用 `Qwen3-VL-2B` 提取最后一层视觉-语言语义 Token，经独立的 LayerNorm 与 FFN 块处理；
   - **视频生成专家（Generative Expert）**：采用 `Wan 2.2 5B` 作为时空先验主干；
   - **动作专家（Action Expert）**：构建与 `Wan 2.2` 层数相同的 Transformer 模块，通过 AdaLN 注入动作流时间步 $\tau_a$；
   - **三模态联合自注意力**：各专家保有各自独立的投影层与 FFN（防止跨模态梯度干扰），但在每一层的 Multi-Head Self-Attention 中将三路 Query/Key/Value 拼接后做联合自注意力计算。
2. **动作密集-视频稀疏预测（Action-Dense Video-Sparse Prediction）**：
   - 若以相同帧率预测长动作块对应的全部视频帧，视频 Token 数量将远超动作 Token，导致联合注意力被视频生成主导并拖慢推理；
   - Motus 将**视频采样帧率下采样为动作频率的 $1/6$** （例如每 6 步动作对应 1 帧关键未来图像），使视频 Token 与动作 Token 数量保持平衡，兼顾动力学引导与控制精度。

## 3. 基于光流 VAE 的潜动作学习与六层数据金字塔

### 3.1 光流潜动作变分自编码器（Latent Action VAE）

为从无标签人类/互联网视频中提取可迁移的运动监督：
1. 使用 `DPFlow` 计算相邻视频帧间的稠密光流场并转换为 RGB 光流图；
2. 采用深度卷积变分自编码器（`DC-AE`）将光流图编码为 $4 \times 512$ 维特征并重建光流；
3. 通过轻量级编码器将 $4 \times 512$ 特征进一步压缩为 **14 维连续潜动作向量 $\mathbf{z}_t \in \mathbb{R}^{14}$**（恰好对齐双臂机器人 $2 \times 7 = 14$ 自由度控制空间的尺度）；
4. **弱动作监督对齐（90% 无标签 + 10% 带标签/AnyPos 任务无关采样）**：利用 CuRobo 随机采样机械臂工作空间生成的任务无关图像-动作对（AnyPos）与少量演示轨迹，在训练 VAE 时联合优化光流重建、真实动作对齐与 KL 正则：

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \lambda_a \|a_{\text{real}} - a_{\text{pred}}\|^2 + \beta \mathcal{L}_{\text{KL}}
$$

### 3.2 六层具身数据金字塔与三阶段训练流水线

Motus 将多源具身数据组织为从底向上的**六层数据金字塔**：
- **Level 1**：海量互联网图文/视频数据（用于初始化预训练 VLM 与 VGM）；
- **Level 2**：第一视角人类操作视频（如 EgoDex）；
- **Level 3**：仿真合成多视角操作数据；
- **Level 4**：任务无关（Task-Agnostic）随机探索图像-动作对（AnyPos）；
- **Level 5**：多机器人跨本体操作轨迹（AgiBot-World、RoboMind、RDT 等）；
- **Level 6**：目标机器人特定任务专家轨迹。

对应的**三阶段训练配方**：
- **Stage 1（视觉动力学适应）**：在 Level 2 + 3 + 5 上仅微调视频生成专家（VGM），使其学会根据指令与首帧生成物理合理的机器人/人手交互视频；
- **Stage 2（基于 14 维潜动作的全模型统一预训练）**：冻结 VLM，在 Level 2 + 3 + 4 + 5 上使用光流提取的 **潜动作 $\mathbf{z}_{t+1:t+k}$** 联合预训练全部三个专家，将视觉运动先验注入动作专家；
- **Stage 3（目标机器人微调 SFT）**：在 Level 6 目标机器人真实动作数据上微调全体三个专家，完成从潜动作空间到目标机器人物理控制空间的精确映射。

## 4. 实验评估与核心结果

### 4.1 RoboTwin 2.0（50 个双臂任务联合训练，Clean + Randomized）

所有模型均从各自预训练权重出发，在 RoboTwin 2.0 的 50 个任务（2,500 条 Clean + 25,000 条强随机化演示）上统一微调 40k 步，每个任务评测 100 次试验：

| 方法 | 模型类别 | 参数量 | RoboTwin 2.0 Clean (%) | RoboTwin 2.0 Randomized (%) | 平均成功率 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\pi_{0.5}$** | VLA | 3.3B | 42.98 | 43.84 | 43.41 |
| **X-VLA** | VLA | 0.9B | 72.80 | 72.84 | 72.82 |
| **Motus (w/o Pretrain)** | WAM | 8.0B | 72.80 | 77.00 | 74.90 |
| **Motus (Stage 1 Pretrain Only)** | WAM | 8.0B | 82.86 | 81.86 | 82.36 |
| **Motus (Full Stage 2 Pretrain)** | WAM | **8.0B** | **88.66** | **87.02** | **87.84** |

- **消融洞见**：不经过预训练的 Motus 在 Randomized 场景下为 $77.00\%$；仅加入 Stage 1 视频动力学预训练即可提升至 $81.86\%$（$+4.86\%$）；进一步加入 Stage 2 **光流潜动作预训练**后跃升至 **87.02%** （Clean **88.66%**，平均 **87.84%**），相比 $\pi_{0.5}$ 取得 **+43.2%** 的绝对提升，相比 X-VLA 提升 **+14.2%**。在 LIBERO 基准上，Motus 同样取得 **97.7%** 的高平均成功率。

### 4.2 双平台真机长程灵巧操作实验（AC-One 与 AgileX-Aloha-2）

在包含叠毛巾、手冲咖啡机煮咖啡、研磨咖啡豆、饮水机接水、水壶浇花、烤箱放面包等高难度长程多步骤任务上（采用多任务单模型联合训练，汇报子目标加权的部分成功率 Partial Success Rate）：
- **AC-One 双臂平台（9 个任务平均）**：$\pi_{0.5}$ 仅为 $14.79\%$，无预训练版 Motus 为 $25.86\%$，完整预训练 **Motus** 达到 **63.22%** （提升 **+48.43 个百分点**；其中研磨咖啡豆从 $8\%$ 跃升至 **92%**，煮咖啡从 $0\%$ 跃升至 **62%**，OOD 放方块从 $28.1\%$ 跃升至 **75.0%**）；
- **AgileX-Aloha-2 双臂平台（5 个任务平均）**：$\pi_{0.5}$ 为 $48.60\%$，无预训练版 Motus 为 $26.60\%$，完整预训练 **Motus** 达到 **59.30%** （其中饮水机接水从 $62\%$ 提升至 **96%**）。
