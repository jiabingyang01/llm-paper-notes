# FlowWAM：以光流作为世界动作模型的统一视频原生动作表示

> **论文信息**：Yixiang Chen, Peiyan Li, Yuan Xu, Qisen Ma, Jiabing Yang, Kai Wang, Jianhua Yang, Dong An, He Guan, Gaoteng Liu, Jianlou Si, Jun Huang, Jing Liu, Nianfeng Liu, Yan Huang, Liang Wang (CASIA NLPR / UCAS / FiveAges / MBZUAI / Alibaba Group)
> **发表期刊/会议**：arXiv 2026 (`arXiv:2607.13017`)
> **项目主页**：[flow-wam.github.io](https://flow-wam.github.io)

## 1. 核心思想与研究动机

世界动作模型（World Action Models, WAMs）利用预训练视频生成模型（如 `Wan2.2`）中的时空动力学先验来赋能机器人控制与未来推演。然而，如何为预训练视频生成器设计一种**既契合其原生视觉分布、又能保留稠密跨帧运动线索以解码精确控制指令**的“动作表示（Action Representation）”，始终是一个悬而未决的核心难题：
- **数值型动作向量（Numerical Actions）**：物理控制精确，但与预训练视频生成器的像素/潜空间格式存在巨大模态鸿沟，且不同机器人动作空间互不兼容；
- **抽象潜动作（Learned Latent Actions，如 LAPA / Motus）**：通过帧间转移学习的低维潜向量过于抽象，丢失了精细操作必需的逐像素空间接地（Spatially Grounded）运动细节；
- **静态图像空间动作条件（如动作射线图、掩码 Masks）**：仅提供“动作发生在哪”的静态空间提示，缺乏“本体各部位如何随时间连续移动”的跨帧速度与方向信息。

**FlowWAM（5.8B）** 提出了一个简洁而本质的破局洞察：**稠密光流（Optical Flow）是连接预训练视频先验与底层机器人控制的天然统一动作表示。** 通过将逐像素位移 $(u, v)$ 经可逆 HSV 色轮编码映射为标准 RGB 光流视频，FlowWAM 构建了**双流扩散 Transformer（Dual-Stream DiT）**：
1. **视频原生同构性**：光流视频与场景 RGB 视频共享完全相同的维度格式，可直接复用冻结的预训练视频 VAE 与共享的 DiT 主干块；
2. **一模双用（Unified Policy & World-Model Modes）**：在**策略模式（Policy Mode）** 下，模型从噪声同步生成未来 RGB 与光流视频，并由轻量动作专家（780M）从中间层特征解码 14-DoF 机器人动作；在**世界模型模式（World-Model Mode）** 下，固定给定的目标光流序列，条件化渲染高保真未来 RGB 视频；
3. **无动作标签人类视频预训练**：由于光流可直接用 `RAFT` 从任意原始视频中自动提取，FlowWAM 能在无动作标注的第一视角人类操作数据集（`EgoDex`，约 800 小时）上直接预训练双流视频生成器，在 **RoboTwin 2.0**（Clean **92.94%** / Random **92.14%**）与 **WorldArena** 世界模型基准（EWMScore **63.71**，轨迹精度相对提升 **+18.4%**）上双双夺冠。

## 2. 方法详解：双流 RGB-Flow 生成与抗误差动作解码

### 2.1 可逆 HSV 光流 RGB 编码与 T 型多视角拼图

1. **可逆 HSV 光流编码**：
   给定相邻帧间的二维光流场 $\mathbf{f}_t = (u, v) \in \mathbb{R}^{H \times W \times 2}$，通过色轮映射 $\phi(\cdot)$ 将运动方向编码为色相（Hue）、运动幅值编码为饱和度（Saturation）：

$$
F_t = \phi(\mathbf{f}_t): \qquad \text{H} = \frac{\operatorname{atan2}(v, u) + \pi}{2\pi}, \qquad \text{S} = \frac{\min(\|\mathbf{f}_t\|, m)}{m}, \qquad \text{V} = 1
$$

   其中最大幅值截断常数取 $m = 25\text{ px}$（滤除 $< 0.5\text{ px}$ 的背景微噪）。该映射完全可逆（$\phi^{-1}(F_t)$ 可精确还原数值光流场），且生成的 $F_t$ 与普通 RGB 图像格式完全一致。
2. **多视角 T 型拼图（T-Shape Tiling）与机器人专属光流**：
   - 将主视角（Head Camera，$320 \times 256$）置于上方，左右腕部视角（Wrist Cameras）下采样后并排拼接于下方，组成单一的 $320 \times 384$ 图像送入冻结的 `Wan2.2 VAE`；
   - 在 RoboTwin 训练中，利用仿真器仅重放机器人本体运动（Robot-Only Replay）提取纯机械臂光流，隔离背景干扰。

### 2.2 共享权重双流 DiT 与两种工作模式

设 RGB 视频序列与光流视频序列经同一个冻结的 VAE 编码器 $\mathcal{E}$ 分别映射为潜变量 $\mathbf{z} = \mathcal{E}(\mathbf{V})$ 与 $\mathbf{z}^f = \mathcal{E}(\mathbf{F})$：
- **参数极致共享**：基于 `Wan2.2-TI2V-5B` 初始化，RGB 流与光流流**仅各自拥有独立的 Patch Embedding 层与输出投影头**（光流流由主干对应层深拷贝初始化），**中间全部 Transformer Blocks 完全共享权重**；
- **流内独立 RoPE + 跨流联合自注意力**：在每一层自注意力内部，对 RGB Token 与光流 Token 分别施加各自的旋转位置编码（RoPE），随后拼接在一起执行全序列联合自注意力（Joint Self-Attention），完毕后再拆回各自流中；
- **两种推理模式切换**：
  - **策略模式（Policy Mode）**：首帧固定为当前干净观测潜变量，后续帧的 $\mathbf{z}$ 与 $\mathbf{z}^f$ 均从高斯噪声初始化并同步去噪（25 步），同时由动作专家解码出 32 步动作块；
  - **世界模型模式（World-Model Mode）**：将整段光流潜变量 $\mathbf{z}^f$ 固定为目标运动轨迹的干净 VAE 编码，仅对 RGB 潜变量 $\mathbf{z}$ 从高斯噪声去噪，生成严格遵循指定运动轨迹的未来视频。

### 2.3 运动感知流损失重加权（Motion-Aware Reweighting）

在机器人操作场景中，运动高度集中在机械臂与被操作物体局部区域，静态背景占据绝大部分像素。为防止静态背景主导光流流匹配损失 $\mathcal{L}_{\text{flow}}$，作者根据当前帧光流潜变量 $\mathbf{z}^f$ 相对于参考首帧 $\mathbf{z}_{\text{ref}}^f$ 的通道平均偏移量，构造逐空间位置的动态聚焦权重 $w_{\text{motion}}$：

$$
w_{\text{motion}} = 1 + \alpha \cdot \frac{\langle |\mathbf{z}^f - \mathbf{z}_{\text{ref}}^f| \rangle_c}{\max \, \langle |\mathbf{z}^f - \mathbf{z}_{\text{ref}}^f| \rangle_c}
$$

其中 $\langle \cdot \rangle_c$ 表示沿潜空间通道维度取均值，运动增强系数取 $\alpha = 2.0$。双流视频总损失为：

$$
\mathcal{L}_{\text{video}} = (1 - \lambda_f)\,\mathcal{L}_{\text{RGB}} + \lambda_f\,\mathcal{L}_{\text{flow}}, \qquad \lambda_f = 0.1
$$

### 2.4 随机加噪潜特征条件化的动作专家（Stochastic Latent Conditioning）

动作专家（Action Expert，30 层 Transformer，隐藏维度 1024，参数量 $\sim 780\text{M}$）通过逐层 Cross-Attention 读取双流视频生成器的中间层隐状态，并结合当前本体状态预测未来 $N = 32$ 步的 14 维动作块 $\hat{\mathbf{a}}_0 \in \mathbb{R}^{32 \times 14}$。
- **训练-推理分布偏移问题**：训练时双流视频生成器若仅输入干净的真实 VAE 潜变量，动作专家将过度依赖完美特征；而在策略推理时，双流生成器输出的潜变量不可避免带有残余去噪误差；
- **随机潜变量加噪（Stochastic Latent Conditioning）**：在训练阶段，以 $p = 0.5$ 的概率向输入给视频生成器的干净潜变量混入随机强度 $\sigma \sim \mathcal{U}[0, 1]$ 的高斯噪声，并将噪声水平 $\sigma$ 作为辅助嵌入告知动作专家：

$$
\tilde{\mathbf{z}}^f = (1 - \sigma)\,\mathbf{z}^f + \sigma\,\bm{\epsilon}^f, \qquad \tilde{\mathbf{z}} = (1 - \sigma)\,\mathbf{z} + \sigma\,\bm{\epsilon}^r, \qquad \sigma \sim \mathcal{U}[0, 1]
$$

在其余 $1 - p = 0.5$ 的训练步中则保持纯干净潜变量输入。总体有监督联合训练损失为 $\mathcal{L} = \mathcal{L}_{\text{video}} + \lambda_a \mathcal{L}_{\text{action}}$（$\lambda_a = 1.0$）。

## 3. 实验评估与核心结果

### 3.1 RoboTwin 2.0（50 个双臂操作任务）策略模式对比

FlowWAM 采用两阶段训练：Stage 1 在无动作标签的第一视角人类灵巧操作视频数据集 **EgoDex（约 0.8k 小时）** 上仅用 $\mathcal{L}_{\text{video}}$ 预训练双流视频生成器；Stage 2 在下游机器人数据上联合训练动作专家：

| 方法 | 模型类别 | 参数量 | RoboTwin 2.0 Clean (%) | RoboTwin 2.0 Random (%) | 平均成功率 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\pi_{0.5}$** | VLA | 3.3B | 42.98 | 43.84 | 43.41 |
| **X-VLA** | VLA | 0.9B | 72.88 | 72.84 | 72.86 |
| **GigaWorld-Policy** | WAM | - | 86.36 | 85.04 | 85.70 |
| **Motus** | WAM | 8.0B | 88.66 | 87.02 | 87.84 |
| **X-WAM** | WAM | - | 89.76 | 90.68 | 90.22 |
| **Fast-WAM** | WAM | 6.0B | 91.88 | 91.78 | 91.83 |
| **FlowWAM (w/o PT，Full 配方)** | WAM | 5.8B | 82.40 | 80.80 | 81.60 |
| **FlowWAM (w/ EgoDex PT，Ours)** | WAM | **5.8B** | **92.94** | **92.14** | **92.54** |

> 注：在 Clean2Random（仅用 Clean 演示训练、直接零样本评测 Randomized 泛化）的跨本体预训练基线评测表中，`FlowWAM (Base w/o PT)` 取得 Clean $86.6\%$ / Rand. $38.1\%$（平均 $62.3\%$），而经 `Raw EgoDex PT (0.8k h)` 预训练后跃升至 Clean **89.2%** / Rand. **53.0%** （平均 **71.1%**），展现出强大的跨物种（人手 $\to$ 机械臂）运动迁移能力。

### 3.2 WorldArena 世界模型模式评测（16 项指标与 EWMScore）

在权威的具身世界模型评测基准 **WorldArena** 上，对比通用视频生成模型与动作条件机器人世界模型：

| 世界模型方法 | 动作条件类型 | JEPA Sim. $\uparrow$ | Subj. Cons. $\uparrow$ | Bg. Cons. $\uparrow$ | **Traj. Acc. (轨迹精度)** $\uparrow$ | Depth Acc. $\uparrow$ | Sem. Align. $\uparrow$ | **EWMScore (总分)** $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wan 2.6** | Text | 72.57 | 73.15 | 84.29 | 12.18 | 75.43 | 88.09 | 59.80 |
| **ABot-PhysWorld** | Text | 90.36 | 80.56 | 89.02 | 31.50 | 71.99 | 89.58 | 62.63 |
| **Cosmos-Predict 2.5** | Action | 90.65 | 67.24 | 72.72 | 27.49 | 90.35 | 89.86 | 54.29 |
| **Ctrl-World** | Action | 92.77 | **83.56** | **90.30** | 48.20 | 93.25 | 88.68 | 59.98 |
| **Vidar** | Img-Act | 61.10 | 70.37 | 78.24 | 21.26 | 79.77 | 88.46 | 51.92 |
| **GigaWorld-1** | Img-Act | 96.77 | 80.97 | 86.43 | 54.27 | 98.44 | 89.42 | 62.34 |
| **FlowWAM (Ours)** | **Optical Flow** | **97.14** | 82.46 | 89.97 | **64.26**（**+18.4% 相对提升**） | **98.97** | **89.93** | **63.71** |

### 3.3 消融实验、可解码性检验与真机部署

1. **核心组件消融（RoboTwin 策略成功率 & WorldArena EWMScore）**：
   - 策略侧：完整 FlowWAM 达 **89.8%**；若去掉随机加噪潜特征条件（w/o stochastic AE cond.）降至 $82.1\%$；去掉运动感知流重加权（w/o flow-loss reweighting）降至 $83.9\%$；将 HSV 光流图像退化为原始双通道 $(u, v)$ 张量降至 $72.3\%$；直接改用数值动作向量降至 $69.8\%$；
   - 世界模型侧（本地验证集）：完整 HSV 光流条件达 **65.23**，显著优于静态图像掩码（$57.84$）、原始 $(u,v)$ 流（$56.72$）、数值动作（$54.18$）与纯文本条件（$49.31$）。
2. **光流可解码性定量验证（Flow Decodability）**：在 RoboTwin 全部 50 个任务上，统计模型预测光流与真实机械臂光流之间的端点误差（EPE）与策略任务成功率的相关性，发现两者呈现极强的负相关（**Pearson $r = -0.81$**），直接证实策略性能提升确实源于更精准的像素级运动规划。
3. **真机单臂与双臂实验**：在真实单臂与双臂操作套件上，FlowWAM 取得 **75.7%** 的平均成功率，超越 $\pi_{0.5}$（$61.4\%$）达 **+14.3 个百分点**，超越 Motus（$57.1\%$）达 **+18.6 个百分点**，且在需要双臂高度协同的任务上优势尤为显著。
