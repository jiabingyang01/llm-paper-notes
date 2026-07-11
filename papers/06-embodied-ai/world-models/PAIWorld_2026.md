# PAIWorld：面向机器人操作的 3D 一致世界基础模型

> **论文**：*PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation*
>
> **作者**：Yuhang Huang, Jiazhao Zhang, Xuan Lv, Junyan Xu, Zhiyuan Yu, Ruizhen Hu, Kai Xu（通讯作者）et al.
>
> **机构**：Institute of AI for Industries, Chinese Academy of Sciences（中国科学院工业人工智能研究所）
>
> **发布时间**：2026 年 06 月（arXiv 2606.18375，v3 于 2026-06-23）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.18375) | [PDF](https://arxiv.org/pdf/2606.18375)
>
> **分类标签**：`世界模型` `多视角3D一致性` `机器人操作` `扩散Transformer` `REPA`

---

## 一句话总结

PAIWorld 把单视角世界基础模型（WFM）改造成多视角 3D 一致的世界模型：用 **Geometry-Aware Cross-View Attention + Geometric RoPE** 打通视角间的信息通路，再用 **Latent 3D-REPA** 把冻结的 Depth Anything 3 的 3D 感知特征蒸馏进 DiT，使跨视角内容满足几何约束；在 WorldArena 上以 EWMScore 72.31 排名第 1、AgiBot-Challenge2026 上以 0.8245 排名第 2（Scene Consistency 0.9041 居首），文本条件多视角生成中在 7 个指标里 6 个最优（MEt3R 14.20 相对次优提升约 10%）。

## 一、问题与动机

机器人操作系统天生是 **多视角** 的：常同时配备第一人称（egocentric）、手到眼（eye-to-hand）与腕部（wrist）等多台相机，为策略学习提供互补的几何与语义线索。当世界模型充当这类系统的模拟器时,它必须在所有视角上同步生成未来观测,并保持严格的 **多视角 3D 一致性**:同一物体须出现在几何相容的位置、深度与纹理在每个视角每个时刻都一致。任何一致性破裂（跨视角物体漂移、深度矛盾、纹理错位）都会让想象轨迹在物理上失真,并把误差传播进下游规划与控制。

现有做法达不到这一要求。Cosmos、CogVideoX、Vista 等单视角 WFM 结构上只支持一个视角；Genie、iVideoGPT 等多视角做法通常沿序列维 **flat concatenation** 把不同视角的 token 直接拼接,把跨视角 token 当成时间 token 一样处理,让模型完全从数据里隐式发现跨视角对应关系——随着视角数与场景复杂度上升,这种隐式发现越来越不可靠。

作者把失败归结为两个层级不同的根因:

- **缺少显式的视角间通信机制（architectural level）**：flat concatenation 不区分同视角/跨视角 token，每个视角实际是孤立生成的，无法协调预测或消解跨视角冲突。
- **缺少 3D 几何先验（training-objective level）**：即便有了通信通路，模型也得不到"什么才是几何一致的 3D 结构"这一监督信号，信息交换会退化成匹配色板、复制纹理这类肤浅捷径。

核心论点:两者 **既必要又充分**,且必须同时具备——只有通路而无几何目标会坍缩成捷径,只有几何先验而无通路则约束无法跨视角传播。PAIWorld 由此提出"通路 + 目标"两大支柱。

## 二、核心方法

PAIWorld 建立在 DiT 流匹配（flow matching）视频世界模型骨干之上（实现用 Cosmos-Predict2.5，约 14B 参数），插入三个即插即用组件:两个构成通路（Cross-View Attention 与 Geo-RoPE），一个构成几何目标（Latent 3D-REPA）。

### 2.1 问题设定与流匹配骨干

系统有 $V$ 台相机,时刻 $t$ 观测到 $\{I_t^v\}_{v=1}^V$ 及各自内参 $\mathbf{K}^v$、外参 $\langle \mathbf{R}^v, \mathbf{t}^v \rangle \in \mathrm{SE}(3)$,在文本/动作条件 $c$ 与上下文帧下建模条件分布 $p_\theta(\{I_{t_0+1:T}^v\}\mid\{I_{1:t_0}^v\},\{\mathbf{K}^v,\mathbf{R}^v,\mathbf{t}^v\},c)$。骨干在预训练视频 VAE（Wan2.1 的 spatial-temporal VAE）的隐空间上训练一个速度场,沿线性插值路径把噪声搬运到数据:

$$
\mathbf{z}_s = (1-s)\mathbf{z}_0 + s\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{s,\boldsymbol{\epsilon}}\big[\|u_\theta(\mathbf{z}_s,s) - (\boldsymbol{\epsilon}-\mathbf{z}_0)\|_2^2\big]
$$

用大白话说:在噪声与干净隐编码之间连一条直线,让网络在直线上任意一点都能预测出"该往干净方向走的速度"。文本条件经 AdaLN 注入;动作条件不是当成抽象向量,而是沿用 EVAC 把动作渲染成像素空间的 **spatial action maps**（末端轨迹投影到每个相机视图）再与噪声隐编码沿通道拼接,让动作语义落到像素几何上。

### 2.2 Geometric Rotary Position Embedding（Geo-RoPE）

把每个注意力头的 query/key 沿维度切成两个等分子空间:**ray 子空间**（$d_r=d/2$,编码像素级射线方向）与 **pose 子空间**（$d_p=d/2$,编码视角级相机位姿）。

射线方向由像素反投影 + 用相机旋转逆旋转到世界系得到:

$$
\mathbf{d}^v(h,w) = \mathrm{normalize}\Big((\mathbf{R}^v)^\top (\mathbf{K}^v)^{-1}\,[\,h{+}0.5,\ w{+}0.5,\ 1\,]^\top\Big)\in\mathbb{R}^3
$$

位姿特征是一个 12 维向量,包含 Euler 角、平移、相机中心 $-(\mathbf{R}^v)^\top\mathbf{t}^v$ 与光轴 $(\mathbf{R}^v)^\top\mathbf{e}_z$:

$$
\mathbf{e}^v = \big[\ \underbrace{\text{yaw},\text{pitch},\text{roll}}_{\text{Euler}},\ \underbrace{\mathbf{t}^v}_{\text{平移}},\ \underbrace{-(\mathbf{R}^v)^\top\mathbf{t}^v}_{\text{相机中心}},\ \underbrace{(\mathbf{R}^v)^\top\mathbf{e}_z}_{\text{光轴}}\ \big]\in\mathbb{R}^{12}
$$

ray 子空间用逐像素的 $\mathbf{d}^v(h,w)$ 做 RoPE 旋转,pose 子空间用视角内共享的 $\mathbf{e}^v$ 做旋转,拼回即得 $\tilde{\mathbf{q}}=[\tilde{\mathbf{q}}_{\text{ray}};\tilde{\mathbf{q}}_{\text{pose}}]$（key 同理）。

用大白话说:让每个 token 的"位置编码"不再是抽象序号,而是它在 3D 世界里看向哪条射线（细粒度像素对应）、来自哪台相机（视角身份）。这样两个看向同一 3D 点的跨视角 token 会天然得到高内积、被注意力自动关联;把 ray 与 pose 分成两块,是防止"逐像素变化的射线信号"和"视角内恒定的位姿信号"互相干扰。

### 2.3 Geometry-Aware Cross-View Attention

标准 DiT 的时序自注意力把 $V$ 折进 batch 维,视角间零交互。PAIWorld 在选定层插入专门的 **Cross-View Attention** 子块:各视角的 query/key 先各自用本视角相机几何做 Geo-RoPE 旋转,再让某视角 query 去 attend 所有视角拼接后的 key/value:

$$
\hat{\mathbf{Z}}_t^v = \mathbf{Z}_t^v + \text{gate}\cdot\mathrm{softmax}\!\left(\frac{\tilde{\mathbf{Q}}_t^v\,[\tilde{\mathbf{K}}_t^1;\dots;\tilde{\mathbf{K}}_t^V]^\top}{\sqrt{d}}\right)[\mathbf{V}_t^1;\dots;\mathbf{V}_t^V]
$$

gate 用 AdaLN-Zero 初始化为 0,保证第 0 步严格等价于预训练单视角模型,新模块随训练渐进生效。此外周期性地把 view 与 spatial 维展平做 **spatial-concat self-attention**,给每个 token 更宽的跨视角感受野。作者强调:通路只决定信息 **怎么流**,不决定流的内容是否 3D 一致,所以还需下面的几何目标。

### 2.4 Latent 3D-REPA（3D 几何先验)

用冻结的 **Depth Anything 3（DA3)** 作为 3D 感知特征提取器（DA3 在显式几何监督下训练,能预测深度、point map、相机位姿,其中间特征内化了真实 3D 结构,而非仅 2D 外观)。不直接逐 token 回归 DA3 特征,而是蒸馏 token 间的 **关系结构**（对两个编码器的特征空间差异更鲁棒)。用 anchor sampling 把 token-token 相似度矩阵的计算从 $O(N^2)$ 降到 $O(MK)$:随机取 $K$ 个 anchor,度量每个 token 对 anchor 的余弦相似度

$$
\mathbf{S}(\mathbf{F})_{i,a} = \frac{\mathbf{f}_i^\top \mathbf{f}_a}{\|\mathbf{f}_i\|\,\|\mathbf{f}_a\|},\quad a\in\mathcal{A}
$$

在两种粒度上对齐 DiT 与 DA3 的关系矩阵:帧内（跨视角+空间）$\mathbf{S}_{\text{intra}}$ 与跨帧（时间维）$\mathbf{S}_{\text{inter}}$,均用 SmoothL1:

$$
\mathcal{L}_{\text{REPA}} = \underbrace{\mathrm{SmoothL1}(\mathbf{S}_{\text{intra}}^{\text{DiT}},\mathbf{S}_{\text{intra}}^{\text{DA3}})}_{\mathcal{L}_{\text{spatial}}} + \underbrace{\mathrm{SmoothL1}(\mathbf{S}_{\text{inter}}^{\text{DiT}},\mathbf{S}_{\text{inter}}^{\text{DA3}})}_{\mathcal{L}_{\text{temporal}}}
$$

用大白话说:不逼 DiT 的特征去逐点等于 DA3,而是逼"DiT 里哪些位置彼此相像"这套关系网络去模仿 DA3 那套几何关系网络——因为 DA3 的关系是由真实 3D 结构决定的,模仿它就等于把 3D 一致性灌进扩散过程。

### 2.5 联合机制与总目标

两支柱耦合成一个自增强回路:Cross-View Attention 开通路、Geo-RoPE 把通路偏置到几何对应 token 并给所有视角一个共同参照系;Latent 3D-REPA 保证交换的内容几何有意义、约束能跨视角一致传播。总损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda\,\mathcal{L}_{\text{REPA}},\qquad \lambda=0.5
$$

DA3 全程冻结作固定 3D 先验;Cross-View 块 AdaLN-Zero 门控初始化保留预训练权重。

**训练配置**：骨干 Cosmos-Predict2.5（约 14B）,文本 embedder 用 Cosmos-Reason1（Physical AI VLM);数据约 2.5M 多视角机器人操作视频片段,来自 5 个源——AgiBot-World 35%、RoboMIND 20%、Galaxea 15%、RoboTwin 15%、RoboCOIN 15%;训练 30,000 步,AdamW + cosine,前 3,000 步线性 warmup 到峰值 $3\times10^{-5}$,NVIDIA H200、约 30k GPU-hours。

## 三、实验结果

### 3.1 动作条件生成 · WorldArena（Table 1，7 项指标)

（下表节选代表性条目,展示各维度的最优者与 PAIWorld 的均衡领先;完整排行含更多基线。)

| 方法 | EWMScore↑ | Visual Q.↑ | Motion Q.↑ | Content Cons.↑ | Physics↑ | 3D Acc.↑ | Controll.↑ |
|---|---|---|---|---|---|---|---|
| WorldScape v0.2 | 68.32 | 62.65 | 42.34 | 65.18 | **73.29** | 96.28 | 87.59 |
| Pelican-Unify | 70.38 | 63.60 | 61.73 | 60.41 | 63.98 | **97.65** | 87.60 |
| BWM-Fast | 72.15 | 62.79 | 78.79 | 58.30 | 61.18 | 91.53 | 88.58 |
| UNIS | 72.16 | 60.85 | **81.60** | 56.44 | 61.56 | 91.16 | **90.19** |
| **PAIWorld** | **72.31** | 63.04 | 80.45 | 57.85 | 61.66 | 91.51 | 87.16 |

要点:PAIWorld 以 EWMScore 72.31 排名第 1,险胜 UNIS（72.16）与 BWM-Fast（72.15);其卖点是"全维度均衡靠前",而竞品各自只在单一维度突出(WorldScape 物理最好但 Motion 崩到 42.34;UNIS 的 Motion/Controllability 最高但 Content/3D 一般)。注意 Motion Quality 上 PAIWorld 80.45 实为 **第 2**（UNIS 81.60 更高),3D Accuracy 91.51 也非全场最高（Pelican-Unify 97.65)——正文与此一致,唯摘要"best Motion Quality among all entries"的措辞与表格/正文相矛盾(见第四节)。

### 3.2 动作条件生成 · AgiBot-Challenge2026（Table 2)

| 队伍 | EWMScore↑ | PSNR↑ | Scene Cons.↑ | nDTW↑ |
|---|---|---|---|---|
| NeoVerse-ABot | **0.829** | **0.6246** | 0.8974 | **0.9651** |
| Loop | 0.8241 | 0.6207 | 0.9024 | 0.9492 |
| Wild Path | 0.8232 | – | – | – |
| VIPL-GENUN | 0.8195 | – | – | – |
| **PAIWorld** | 0.8245 | 0.6161 | **0.9041** | 0.9531 |

PAIWorld EWMScore 0.8245 排名第 2,但在最贴近 3D 一致性的 Scene Consistency 上以 0.9041 居首(比第 1 名 NeoVerse-ABot 高 +0.67 个百分点);nDTW 0.9531 第 2,说明动作条件下生成轨迹紧贴真值,验证了 action-map 条件化。

### 3.3 文本条件多视角生成 · AgiBot-World（Table 3)

| 方法 | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ | Scene Cons.↑ | Geometric↓ | MEt3R↓ |
|---|---|---|---|---|---|---|---|
| Genie-Envisioner | 0.7445 | 0.3345 | 83.78 | 207.20 | **0.9231** | 0.5327 | 15.75 |
| Cosmos-Predict2 | 0.5870 | 0.3251 | 58.28 | 188.64 | 0.8456 | 0.4824 | 17.47 |
| Wan2.1 | 0.5715 | 0.3354 | 56.47 | 184.22 | 0.8617 | 0.4716 | 16.59 |
| **PAIWorld** | **0.7683** | **0.1844** | **45.04** | **175.78** | 0.9041 | **0.4056** | **14.20** |

PAIWorld 在 7 项里 6 项最优:SSIM 0.7683（较次优 +3.2%)、LPIPS 0.1844（较次优约 -45%)、FID 45.04（较 Wan2.1 约 -20%)。最关键的 **MEt3R**（点云跨投影度量的 3D 重建误差,越低越好)14.20,较次优 Genie-Envisioner 15.75 约提升 10%;**Geometric**（Sampson 极线距离,越低越好)0.4056 最优。唯 Scene Consistency 0.9041 略逊 Genie-Envisioner 的 0.9231——后者靠强文本 grounding,PAIWorld 以此小差换取所有几何度量的决定性领先。

### 3.4 消融（Table 4，AgiBot-World)

CVA = Cross-View Attention（含 Geo-RoPE);REPA = Latent 3D-REPA;Δ 为相对纯骨干的 MEt3R 改善量。

| CVA | REPA | SSIM↑ | LPIPS↓ | FID↓ | MEt3R↓ | ΔMEt3R |
|---|---|---|---|---|---|---|
| ✗ | ✗ | 0.6912 | 0.2783 | 53.17 | 16.84 | — |
| ✓ | ✗ | 0.7204 | 0.2361 | 50.02 | 15.91 | 0.93 |
| ✗ | ✓ | 0.7156 | 0.2447 | 49.88 | 16.12 | 0.72 |
| ✓ | ✓ | **0.7683** | **0.1844** | **45.04** | **14.20** | **2.64** |

超可加(super-additive):单独加通路改善 0.93、单独加几何目标改善 0.72,二者合计 1.65,但联合改善 2.64——这一非叠加跃升正是"通路传信息、目标使信息 3D 一致"耦合回路的经验印证;SSIM/LPIPS/FID 也在双组件时改善最大,说明几何增益不以视觉保真度为代价。

## 四、局限性

- **物理交互建模缺失**：当前只保证几何/外观一致,未显式建模接触动力学、可变形物体、流体等——真实操作中的物理接触仍是软肋(作者列为首要 future work)。
- **长时程一致性未验证**：定量实验多在约 180 帧内的 rollout,超长时程 3D 一致性(需分层/递归架构)尚未系统评估。
- **依赖冻结 3D 教师**：Latent 3D-REPA 的几何上限被 Depth Anything 3 的质量与其在机器人域的泛化所约束;若 DA3 在杂乱操作场景估计不准,蒸馏信号会带噪。
- **摘要与表格轻微矛盾**：摘要称"best Motion Quality among all entries",但 Table 1 与正文明确写 PAIWorld 为第 2（UNIS 更高),属论文自洽性小瑕疵,读者引用 WorldArena 数字时需以表格为准。
- **多视角相机监督成本**：方法依赖准确的多视角内外参(Geo-RoPE 与 point map 重建都用到),对无标定或标定漂移的真实机器人平台鲁棒性未讨论;评测仍偏合成/受控数据,缺真机闭环控制验证。
- **算力门槛**：约 14B 骨干 + 30k H200 GPU-hours,复现与迭代成本高。

## 五、评价与展望

**优点**：（1）问题诊断清晰——把多视角世界模型的失败拆成"通路（architectural)"与"目标（objective)"两个正交层级,并论证二者缺一不可,消融的超可加结果给了漂亮的经验支撑,这套"两个层级的缺陷需两个层级的补丁"叙事比单纯堆模块更有说服力。（2）Geo-RoPE 把相机几何(射线 + 位姿)以旋转位置编码方式注入注意力,是对 CameraCtrl 类 Plücker 射线条件化在多视角 attention 上的自然推广,且 ray/pose 分子空间的设计有物理直觉。（3）Latent 3D-REPA 把 REPA 从 2D 图像生成拓展到多视角视频、并改成关系蒸馏 + anchor 采样,工程上把 $O(N^2)$ 降到 $O(MK)$,是把"3D 基础模型（DA3/VGGT/DUSt3R 系)"当监督教师注入生成式世界模型的务实路线。（4）AdaLN-Zero 门控保证从预训练权重平滑起步,是把单视角 WFM 低成本改造成多视角的可迁移技巧。

**缺点/存疑**：（1）核心新意更多是"已有构件的良好组合"——RoPE 相机编码、cross-attention、REPA 对齐单看都非首创,真正的贡献在多视角一致性这一具体设定下的整合与实证。（2）与 SyncDreamer/MVDream/SV4D 等 3D-aware 多视角生成的对比只停留在定性区分(对象中心 vs 场景级、静态 vs 动态、密视角 vs 宽基线),缺少直接定量比较。（3）3D 一致性最终仍由 MEt3R/Geometric 等 2D 投影一致性度量间接刻画,没有真机策略在该世界模型上做 model-based planning/policy 的闭环性能数字来佐证"一致性→下游收益"这一因果链(论文只在文字层面主张)。（4）基线命名多为化名式队名,外部可复现性与公平性难核验。

**与公开工作的关系**：定位介于 Cosmos/Wan 这类单视角 WFM 骨干与 Genie/iVideoGPT 这类 token 拼接式多视角之间,主打"显式几何"这一被前两者忽视的维度;与同作者团队前作 LaDi-WM(latent 扩散世界模型做预测操作)一脉相承,可视为把 latent 世界建模推向多视角 3D 一致的续作。相对 EnerVerse、GR-2、IRASim 等具身视频/动作世界模型,PAIWorld 的差异化在"多相机 3D 一致"而非单流高保真。

**开放问题与可能改进**：把 point map/深度作为可微渲染约束直接进损失(而非仅关系蒸馏)、引入 3D Gaussian/NeRF 中间表征做显式几何 bottleneck、用 VGGT 类联合预测相机 + point map 的教师替代或增强 DA3、以及最关键的——建立"世界模型一致性 → 真机策略成功率"的闭环评测,都是把该方向从"生成质量竞赛"推进到"具身决策可用"的下一步。

## 参考

1. Yu et al. *REPA: Representation Alignment for Generation.* ICLR 2025.（Latent 3D-REPA 的母框架)
2. NVIDIA. *Cosmos World Foundation Model Platform for Physical AI.* arXiv:2501.03575, 2025.（DiT 世界模型骨干血统)
3. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing, 2024.（RoPE,Geo-RoPE 的基础)
4. He et al. *CameraCtrl: Enabling Camera Control for Video Diffusion Models.* ICLR 2025.（Plücker 射线相机条件化,可比技术)
5. Huang et al. *LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation.* arXiv:2505.11528, 2025.（同作者团队前作)
