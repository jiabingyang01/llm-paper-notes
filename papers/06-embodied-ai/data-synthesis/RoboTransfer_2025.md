# RoboTransfer：面向操作策略迁移的可控几何一致视频扩散

> **论文**：*RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Transfer*
>
> **作者**：Liu Liu, Xiaofeng Wang, Guosheng Zhao, Keyu Li, Wenkang Qin, Jiagang Zhu, Jiaxiong Qiu, Zheng Zhu, Guan Huang, Zhizhong Su（Liu Liu 与 Xiaofeng Wang 为共同一作，Zhizhong Su 为通讯作者）
>
> **机构**：Horizon Robotics（地平线）、GigaAI、CASIA（中科院自动化所）
>
> **发布时间**：2025 年 05 月（arXiv 2505.23171）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.23171) | [PDF](https://arxiv.org/pdf/2505.23171)
>
> **分类标签**：`视频扩散` `数据合成` `多视角一致性` `Sim2Real` `策略迁移`

---

## 一句话总结

RoboTransfer 把真实机器人演示分解为**几何条件**（metric depth + surface normal）与**外观条件**（背景参考图 + 物体 CLIP 特征），用一个多视角拼接的视频扩散模型合成几何一致、外观可控的操作数据;在最难的 Diff-All 设定下，Spoon Pick&Place 任务成功率从 real-only 的 13.3% 提升到 real + 物体&背景增广的 46.7%（相对 +251%），且纯合成数据训练的策略也能达到 40% 成功率、反超纯真实数据基线。

## 一、问题与动机

模仿学习（IL）是视觉运动操作的主流范式，但真实演示采集代价极高;仿真虽然便宜，却面临资产稀缺与 sim-to-real gap。用生成模型合成机器人训练数据是一条有前景的路线，但作者指出两个尚未解决的关键难题:

1. **多视角一致性**。真实机器人常依赖多相机并行观测（如双腕相机 + 头部相机），但主流视频生成模型逐视角独立生成，难以保证跨视角几何一致。逐帧图像增广方法（ROSIE）会破坏时空一致性;RoboEngine 基于图像 inpainting 有噪声与抖动、缺乏时序一致;Cosmos-Transfer1 虽是视频扩散、有时序一致，但没有多视角一致。
2. **精确可控性**。操作任务复杂且交互性强，仅靠文本 prompt（text-to-video）无法精确控制场景。Domain randomization 通常只做全局颜色扰动，无法建模局部、结构性的变化。

作者用 Table 1 把自己定位为**首个在机器人数据合成中同时保证时序一致 + 多视角一致，并对背景/物体/环境提供细粒度解耦控制的视频扩散框架**。

| 方法 | 模型类型 | 时序一致 | 多视角一致 | 背景控制 | 物体控制 | 环境控制 |
|---|---|---|---|---|---|---|
| ROSIE | 图像扩散 | ✗ | ✗ | ✗ | ✓ | ✓ |
| RoboEngine | 图像扩散 | ✗ | ✗ | ✗ | ✓ | ✗ |
| Cosmos-Transfer1 | 视频扩散 | ✓ | ✗ | ✗ | ✗ | ✓ |
| **RoboTransfer** | 视频扩散 | ✓ | ✓ | ✓ | ✓ | ✓ |

## 二、核心方法

RoboTransfer 从预训练的 Stable Video Diffusion（SVD）微调而来，采用 EDM 框架。整体分为三块:多视角一致建模、几何条件注入、外观条件注入。

### 2.1 扩散预备（EDM）

以 EDM 参数化的去噪目标为:

$$\mathcal{L}(D_\theta, \sigma) = \mathbb{E}_{\mathbf{x}_0, y_s, y_u}\left[\|\mathbf{x}_0 - D_\theta(\mathcal{E}(\mathbf{x}_0 + \mathbf{n}), \tau_u(y_u), \tau_s(y_s), \sigma)\|_2^2\right]$$

其中 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$，$y_s$ 为空间对齐条件（如 depth），$y_u$ 为非结构化条件（如 CLIP embedding），$\tau_s / \tau_u$ 是对应编码器。跨噪声等级的加权总目标为 $\mathcal{L}(D_\theta) = \mathbb{E}_\sigma[\frac{\lambda(\sigma)}{\exp(u(\sigma))}\mathcal{L}(D_\theta,\sigma) + u(\sigma)]$，权重 $\lambda(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$，且 $\ln(\sigma) \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$。

用大白话说:模型学的是从加噪视频里把干净视频"猜"回来，$\lambda(\sigma)$ 保证不同噪声强度的样本都对训练有均衡贡献，避免只学到某个噪声区间。

### 2.2 多视角一致建模——"拼宽度"而非加跨视角模块

关键设计极简:给定 $N$ 路同步视频 $\{V_1, V_2, \ldots, V_N\}$，沿**宽度维**拼成一张大图后用 VAE 一次性编码:

$$\mathbf{x}_0 = \mathcal{E}([V_1, V_2, \ldots, V_N])$$

这样一来，预训练视频扩散骨干的 self-attention 天然在拼接后的画面上做跨视角推理（in-context learning），把跨视角一致性并入全局空间一致性中。好处是**无需任何结构改动**，可直接加载预训练单视角权重，收敛快、迁移代价低——这与另加一个 cross-view attention 模块的做法（如 SyncammMaster）形成对比。

用大白话说:与其给网络装一个专门"对齐不同相机"的零件，不如把几路画面并排贴成一张宽图喂进去，让原本就会看整张图的注意力顺手把几路对齐。

### 2.3 几何条件注入——metric depth + surface normal

用 metric depth map 与 surface normal map 表征场景几何:深度给出到相机的空间距离，法向编码局部表面朝向，二者互补。由于 depth/normal 视频与 RGB 视图空间对齐，用一个由堆叠卷积构成的 VAE encoder 对几何线索联合下采样编码，再沿**通道维**与噪声 latent 拼接，从而让生成全程被一致且物理合理的 3D 线索引导。

### 2.4 外观条件注入——背景图（VAE/空间对齐）+ 物体（CLIP/cross-attention）

外观从两个互补视角控制:

- **背景外观**:用一张参考背景图 $C_b$，经 VAE 编码得到与生成 latent 空间对齐的表示，与 latent 拼接，控制全局纹理与背景风格。
- **物体外观**:一组物体参考图 $C_o$，因数量可变、空间分布不定，被当作**非结构化条件**——用 CLIP 编码为全局特征，经 cross-attention 引导生成。

一个关键点:外观参考图必须**精心挑选、不与几何先验冲突**（否则 depth/纹理错配会同时削弱几何一致性与视觉保真度）。

### 2.5 数据构造流水线——把真实演示拆成三元组

训练三元组 = 几何条件 + 外观条件 + ground-truth 图像，全自动化:

- **几何条件**:两个腕部相机 + 一个头部相机。因部分机器人只有 RGB 或 RGB-D 原始深度噪声大，用 Video Depth Anything（VAD）出一致深度，再与 RGB-D 传感器用**鲁棒最小二乘**对齐尺度;法向用 LOTUS / MoGe 估计。深度尺度对齐（Algorithm 1/2）把传感器稀疏深度当 **metric anchor**、VAD 稠密输出当 **structural prior**，通过迭代离群点过滤学习尺度 $s$ 与偏移 $b$（$\min_{s,b}\|s\mathbf{p} + b\mathbf{1} - \mathbf{s}\|^2$，有闭式解），既继承传感器的公制精度、又保留 VAD 的完整性。
- **外观条件**:采多视角 RGB 关键帧，VLM descriptor 生成场景/物体描述 → 指导 Grounding-SAM（Grounding DINO + SAM2）分割逐物体 mask;把分割出的物体 inpaint 掉得到干净的无物体背景（如空桌面）;物体区域再过 CLIP 得语义 embedding。

## 三、实验结果

### 3.1 合成质量:几何条件消融（Table 2）

评测多视角一致（Pix.Mat，相邻视图匹配像素数）、几何一致（深度 RMSE/Abs.Rel/Sq.Rel、法向 Mean/Med.Err）与 FVD。**metric depth + normal 联合条件（Metric D.P.+N.）在各视图综合最优**。以左相机为例:

| 条件 | RMSE ↓ | Abs.Rel ↓ | Mean Err ↓ | Med.Err ↓ | Pix.Mat ↑ | FVD ↓ |
|---|---|---|---|---|---|---|
| 原始传感器深度 D.S. | 0.074 | 0.124 | 4.86 | 2.88 | 142.90 | 218.51 |
| 预测深度 D.P. | 0.054 | 0.090 | 3.91 | 2.28 | 149.68 | 123.31 |
| Metric D.P. | 0.049 | 0.081 | 3.48 | 1.99 | 183.26 | 112.43 |
| **Metric D.P.+N.** | **0.047** | 0.079 | **3.31** | **1.92** | **202.03** | **107.43** |

要点:相比 D.S.，D.P. 让中间视图 RMSE / Mean Err 相对改善 27.4% / 14.2%，中间与左右视图 Pix.Mat 分别提升 4.7% / 37.6%;metric 多视图尺度对齐进一步把左/右视图 Pix.Mat 提升 22.4% / 16.8%。

### 3.2 合成质量:外观条件消融（Table 3）

**先 inpaint 背景再作条件**能保住几何结构、并提升外观一致性（RMSE/中位误差/背景相似度约 1% 相对改善）;**逐物体分开编码**（Obj Split）比整图统一编码带来约 1% 的物体 CLIP 相似度提升。二者结合（左相机）取得 BG Sim 0.805、RMSE 0.047、Pix.Mat 202.03、FVD 107.43 的最佳组合。

### 3.3 真机策略实验（Table 4）

平台 Agilex Cobot Magic（双 PIPER 臂 + 三 RealSense D435i，仅用 RGB），基线 ACT，每任务 100 条 ALOHA 遥操演示。两个长程双臂任务:Spoon Pick&Place（4 阶段）、Towel Folding（3 阶段）;两种泛化设定:Diff-Obj（换新物体）、Diff-All（物体 + 环境都变）。指标为成功率 SR 与阶段分 Score。

| 数据构成 | Spoon Diff-Obj SR | Spoon Diff-All SR | Towel Diff-Obj SR | Towel Diff-All SR |
|---|---|---|---|---|
| Real only | 33.3% | 13.3% | 16.7% | 12% |
| Domain Random Aug | 44.4% | 11.1% | 16.7% | 12% |
| Real + Obj Aug | 44.4% | 22.2% | 16.7% | 12% |
| **Real + Obj&Bg Aug** | **66.7%** | **46.7%** | **50.0%** | **28%** |

要点:同时增广物体 + 背景外观是关键;Spoon Diff-All 相对提升 251%（13.3% → 46.7%），Towel Diff-Obj 从 16.7% 跃至 50.0%。值得注意的是 domain randomization 在部分设定下甚至**低于** real-only（Spoon Diff-All 11.1% < 13.3%），佐证仅做全局颜色扰动不足以覆盖结构性域偏移。

### 3.4 合成/真实混合比例（Figure 10）

在 Spoon Pick&Place Diff-All 上扫合成数据占比 0%→100%:SR 与 Score 在 **50/50** 达峰值，SR 从 13.3% 升到 46.7%、阶段分近乎翻倍（1.6→3.0）;继续加大合成占比反而下降（合成数据在接触动力学/材质等细粒度物理保真上有欠缺）。有意思的是**纯合成（100%）仍达 40.0% SR，反超纯真实基线**——说明合成提供视觉多样性、真实提供物理落地，二者互补。

### 3.5 训练配置

扩散模型:SVD 微调，640×384，AdamW，lr 3e-5，全局 batch 8，70K 步;推理 EDM scheduler、30 步去噪 + CFG。数据:Cobot Magic 采集 ~24k 段 10Hz/30 帧片段，12 任务、每任务 1000 样本，另引入 AgiBot-World 增背景多样性。策略:先真实数据预训练 100k 步（batch 512, lr 1e-4），再合成数据微调 50k 步（lr 1e-5），8×NVIDIA H20，预训练 ~24h、微调 ~12h。

## 四、局限性

- **物理保真不足**:合成数据在接触动力学、材质等细粒度物理属性上仍有欠缺，合成占比过高会损害策略，需靠真实数据"锚定"。
- **物体一致性度量受限**:腕部相机因运动/遮挡使物体跟踪不可靠，Obj.Sim 只能在中心视图评估，多视角物体外观一致性缺乏定量证据。
- **依赖精心挑选的参考图**:外观参考图必须与几何先验不冲突，这一"人/规则筛选"环节没有量化其失败率与自动化上限。
- **任务与相机规模有限**:真机仅两个双臂桌面任务、三相机固定布局;拼宽度建模在视角数很多时的 latent 分辨率/显存代价未讨论。
- **几何依赖单目估计**:纯 RGB 场景的深度/法向来自 VAD/MoGe 估计，其误差如何传导到下游策略未做敏感性分析。

## 五、评价与展望

**优点**。(1) "拼宽度 + 复用预训练 self-attention" 实现多视角一致是本文最漂亮的工程简化——不加跨视角模块即可加载 SVD 权重，收敛快、可扩展性好，比 SyncammMaster 一类显式跨视角模块更轻量。(2) 几何/外观显式解耦、且几何用 metric depth + normal 双线索、外观用 VAE 背景 + CLIP 物体的"结构化/非结构化"分工，条件设计干净且各有消融支撑。(3) 深度尺度对齐（稀疏 metric anchor + 稠密 structural prior + 鲁棒最小二乘）是把 RGB-D 噪声深度变成可用监督的实用工程，闭式解、易复现。(4) 真机端到端验证到策略成功率（而非仅 FVD/一致性代理指标），50/50 混比与纯合成反超基线的发现有说服力。

**不足与开放问题**。(1) 与 Cosmos-Transfer1、RoboEngine 的对比主要是定性图示（Figure 9），缺少同数据集下同下游策略的**定量**头对头（如相同任务上三种合成方法各自的真机 SR），"首个多视角一致"的优势没有被量化成策略收益差。(2) 多视角一致性的收益（Pix.Mat 提升）与下游策略成功率之间缺乏因果链路分析——到底是"多视角一致"还是"外观多样性"在起主要作用，从 Table 4 看更像后者（Obj&Bg 增广贡献最大）。(3) domain randomization 基线偏弱（只做颜色扰动），对比稍显不公平。(4) 合成占比过高即退化，暴露该类"重渲染保真、轻物理"的世界数据方法的共性天花板——它擅长补视觉多样性，难补接触/形变物理。

**与公开工作的关系**。RoboTransfer 与 Cosmos-Transfer1（视频到视频、depth/seg 条件）、RoboEngine（图像 inpainting 背景增广）、ReBot（背景 inpaint 多样化）同属"用生成模型给真实演示做外观/背景增广"的路线，差异在于其唯一同时保证多视角一致 + 细粒度背景/物体解耦控制;与 TesserAct（RGB-D-normal 4D 联合生成）、RoboDreamer/UniPi（文本驱动未来帧 + 逆动力学）等"生成式世界模型"路线相比，它不预测动作/未来，而是**保结构改外观**的数据翻译器。未来方向:引入物理/接触约束或可微仿真提升合成物理保真、把物体一致性做成可训练的多视角约束、以及在更多相机与更长时程任务上验证拼宽度建模的可扩展性。

## 参考

1. Abu Alhaija et al. *Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control.* arXiv:2503.14492, 2025.（最直接对比:视频扩散、时序一致但无多视角一致）
2. Blattmann et al. *Stable Video Diffusion.* arXiv:2311.15127, 2023.（本文微调所用骨干 SVD）
3. Chen et al. *Video Depth Anything: Consistent Depth Estimation for Super-Long Videos.* arXiv:2501.12375, 2025.（几何条件的深度来源 VAD）
4. Bai et al. *SyncammMaster: Synchronizing Multi-Camera Video Generation from Diverse Viewpoints.* arXiv:2412.07760, 2024.（多视角一致的对照思路:显式跨视角模块）
5. Zhao et al. *TesserAct: Learning 4D Embodied World Models.* （RGB-depth-normal 4D 联合生成的世界模型路线对照）
