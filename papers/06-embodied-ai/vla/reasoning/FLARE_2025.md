# FLARE：用隐式世界建模做机器人学习

> **论文**：*FLARE: Robot Learning with Implicit World Modeling*
>
> **作者**：Ruijie Zheng、Jing Wang、Scott Reed（共同一作），Johan Bjorck、Yu Fang、Fengyuan Hu、Joel Jang 等，Jan Kautz、Furong Huang、Yuke Zhu、Linxi Fan（共同通讯）
>
> **机构**：NVIDIA、马里兰大学帕克分校、南洋理工大学、德克萨斯大学奥斯汀分校
>
> **发布时间**：2025年5月（RSS 2025 Workshop SWOMO Oral；方法已集成进 NVIDIA GR00T N1.5）
>
> **论文链接**：[arXiv](https://arxiv.org/abs/2505.15659) | [项目主页](https://research.nvidia.com/labs/gear/flare/) | [代码（集成于 Isaac-GR00T）](https://github.com/NVIDIA/Isaac-GR00T)
>
> **分类标签**：`隐式世界模型` `未来潜在对齐` `动作感知嵌入` `Q-former` `人类视频协同训练`

---

## 一句话总结

FLARE 在流匹配 DiT 策略里加入 $M$ 个**可学习未来 token**，把 DiT 中间层对应这些 token 的激活与**未来观测的动作感知潜在嵌入**做余弦对齐（$\lambda=0.2$），从而在不重建未来帧的前提下实现**隐式世界建模**——架构改动极小，RoboCasa 24 任务 70.1%、GR1 24 任务 55.0% 超越 UWM/GR00T N1 等基线，真实 GR1 人形最高 95.1%，并解锁了用**无动作标签的人类第一视角视频**协同训练（仅 1 条机器人轨迹即达 60% 新物体成功率）。

---

## 一、问题与动机

### 1.1 显式视频世界模型的两难

近期一批工作（UWM、UVA、GR-1/GR-2、UniPi 等）尝试"边生成未来视觉帧、边预测动作"地联合学习世界模型与策略。这种显式重建路线存在两个根本困难：

| 困难 | 说明 |
| --- | --- |
| **算力与延迟** | 高保真视觉预测依赖大规模生成模型，开销大、时延高，与高频机器人控制（通常需要几十 Hz）天然冲突 |
| **目标冲突** | 像素级重建追求空间细节与纹理合成，而动作预测需要的是**紧凑、抽象、任务相关**的表示。两个目标抢占模型容量，互相稀释学习效率 |

### 1.2 FLARE 的核心思路

人类伸手去拿桌上的咖啡杯时，会无意识地预判手如何移动、会碰到什么障碍、抓住后杯子是什么手感——这种对未来状态的内部建模是高效运动控制的基础，且**几乎完全是隐式的**，并不需要在脑中渲染出逐像素的未来画面。

FLARE 据此主张：**绕过对未来帧/latent 的显式重建**，只在一个紧凑的、动作感知的潜在空间里做"未来对齐"即可。它把世界建模退化为一个**表示对齐辅助损失**，与现有 VLA 架构（π₀、GR00T N1）完全兼容，只需新增极少量 token。

---

## 二、预备知识：流匹配策略

FLARE 沿用 π₀ / GR00T N1 的流匹配（flow matching）动作生成框架。记观测 $o_t$（多视角图像 + 语言指令）、本体状态 $q_t$、专家动作块 $A_t=(a_t,\dots,a_{t+H})$，并定义视觉-语言嵌入 $\phi_t = VL(o_t)$。

给定流匹配时间步 $\tau\in[0,1]$ 与高斯噪声 $\epsilon\sim\mathcal{N}(0,I)$，构造加噪动作块：

$$A_t^\tau = \tau A_t + (1-\tau)\epsilon$$

模型 $V_\theta$ 学习逼近去噪方向（速度）$\epsilon - A_t$，最小化流匹配损失：

$$\mathcal{L}_{fm}(\theta) = \mathbb{E}_\tau\big[\,\|V_\theta(\phi_t, A_t^\tau, q_t) - (\epsilon - A_t)\|^2\,\big]$$

时间步采样自偏向小 $\tau$ 的 Beta 分布 $p(\tau)=\text{Beta}(\frac{s-\tau}{s};1.5,1),\ s=0.999$。推理时从 $A_t^0\sim\mathcal{N}(0,I)$ 出发，用前向 Euler 迭代细化：

$$A_t^{\tau+1/K} = A_t^\tau + \tfrac{1}{K}V_\theta(\phi_t, A_t^\tau, q_t)$$

全程取 $K=4$ 步。骨干 $V_\theta$ 是交替"交叉注意力 + 自注意力"的 Diffusion Transformer（DiT），交叉注意力用于 condition 在 $\phi_t$ 上。

---

## 三、核心方法

### 3.1 未来潜在表示对齐（FLARE Loss）

FLARE 把输入 DiT 的 token 序列扩展为**三个组成部分**（图 2）：

1. 当前本体状态 $q_t$，经状态编码器编码为 1 个 state token；
2. 加噪动作块 $A_t^\tau$，经动作编码器编码为动作 token；
3. **$M$ 个可学习的 future tokens**（`nn.Embedding`）。

这三股 token 拼成一条序列在 DiT 中做自注意力交互，并对当前 VL 嵌入 $\phi_t$ 做交叉注意力。关键操作：在某个**中间层 $L$** 切出对应 future tokens 的激活，经一个 MLP 投影后，与**未来观测**的冻结 VL 嵌入 $g(\phi_{t+H})$ 做余弦相似度对齐：

$$\mathcal{L}_{align}(\theta) = -\,\mathbb{E}_\tau\big[\cos\big(f_\theta(\phi_t, A_t^\tau, q_t),\ g(\phi_{t+H})\big)\big]$$

其中 $f_\theta\to\mathbb{R}^{B\times M\times D}$ 是第 $L$ 层 future token 的激活，$g\to\mathbb{R}^{B\times M\times D}$ 是未来观测 $\phi_{t+H}$ 的（冻结）编码。总损失为：

$$\mathcal{L} = \mathcal{L}_{fm} + \lambda\,\mathcal{L}_{align}$$

经验上 $\lambda=0.2$ 最优。

用大白话说：动作流匹配负责"把噪声去成动作"，未来对齐负责逼着 future token 的中间表示"长得像未来那一刻观测的潜在编码"。两者沿**两条独立的流**在 DiT 内并行，仅通过自注意力软交互——既迫使 DiT 内部去推理未来潜在状态，又不破坏其动作预测能力。

### 3.2 与 REPA 的两点关键区别

FLARE 的对齐思想借鉴了图像扩散里的 Representation Alignment（REPA），但针对"世界建模"做了两处本质改造：

- **未来 vs. 当前**：REPA 对齐的是**当前观测**的表示（加速收敛）；FLARE 对齐**未来观测** $\phi_{t+H}$——这才构成"世界模型"语义，让策略预判后果。
- **独立 token 流**：FLARE 新增专门的 future token 流，使流匹配与对齐两个目标解耦、各自成流，仅经自注意力交互；而非像 REPA 那样直接监督主干表示。

### 3.3 动作感知的未来嵌入模型

对齐目标 $g(\cdot)$ 用什么至关重要。FLARE 不用通用编码器，而是专门训了一个兼顾**紧凑性**与**动作感知**的 VL 嵌入模型（图 10）：

> 1. **骨干**：采用 `siglip2-large-patch16-256` 的视觉与文本编码器。$256\times256$ 图像 → 256 个 patch token；指令 → 32 个文本 token。
> 2. **跨模态融合**：256 + 32 = 288 个 token 拼接后过 **4 层自注意力 Transformer** 融合。
> 3. **压缩**：用 **Q-former** 让 $M=32$ 个随机初始化的可学习 query token，与 288 个融合 token 经交替自注意力 / 交叉注意力交互，压成 32 个紧凑 VL token（天然支持多相机输入）。
> 4. **注入动作感知**：在嵌入模型上挂 8 个 DiT block，用流匹配动作目标端到端训练，强制把所有任务相关信息压进这 32 个 token。

这里的 future token 数 $M$ 与 Q-former 的 query token 数都取 32，对齐时逐 token 做余弦相似度。

**预训练规模**（附录 B/C）：用约 2,000 小时（实际 Table 3 统计为 169.5M 帧、2,989.5 小时）跨本体数据 = GR00T N1 的仿真 + 真实人形数据，外加 7 个 Open X-Embodiment 数据集（DROID / RT-1 / Language Table / Bridge-v2 / MUTEX / Plex / RoboSet）。在 **256 张 H100、batch 8192、150k 步**上预训练，AdamW + cosine 调度。

### 3.4 EMA 缓解分布偏移

下游后训练时，目标嵌入模型若完全冻结会与策略编码器产生分布偏移。FLARE 改用 **EMA** 让目标缓慢跟随策略编码器更新：

$$\theta_{\text{target\_vl}} \leftarrow \rho\,\theta_{\text{target\_vl}} + (1-\rho)\,\theta_{\text{policy\_vl}}$$

经验上 $\rho=0.995$ 最优。直觉：目标缓慢演化既能适应下游视觉分布，又保持训练稳定性。

### 3.5 训练循环概览

> 1. 取一批 $(o_t,\ q_t,\ A_t,\ o_{t+H})$。
> 2. 采样噪声 $\epsilon$ 与时间步 $\tau$，构造 $A_t^\tau=\tau A_t+(1-\tau)\epsilon$，速度目标 $\epsilon - A_t$。
> 3. 把 state token、动作 token、future token 拼接，连同 $\phi_t=VL(o_t)$ 一起送入 DiT。
> 4. 动作头解码动作 token → 计算 $\mathcal{L}_{fm}$（MSE）。
> 5. 用冻结的 target 嵌入算 $g(\phi_{t+H})$，解码 future token → 计算 $\mathcal{L}_{align}=1-\cos(\cdot)$。
> 6. 反传 $\mathcal{L}=\mathcal{L}_{fm}+\lambda\mathcal{L}_{align}$，并以 EMA 更新 target 嵌入。

值得注意的是：$\mathcal{L}_{align}$ **不需要动作标签**——它只依赖"未来观测嵌入"。这正是后文人类视频协同训练能成立的关键。

---

## 四、实验结果

### 4.1 多任务基准（in-domain 嵌入，公平对比）

两个基准：单臂 **RoboCasa**（24 个 Panda 厨房原子任务）与 **GR1 人形 tabletop**（24 个 GR-1 灵巧操作任务，18 个重排 + 6 个articulated）。为公平对比，本节**不使用**跨本体预训练嵌入，而是在同样的 in-domain 数据上训嵌入 80k 步。所有方法训 80k 步（UWM 因尚未收敛额外训到 400k 步，即 5 倍预算），每 1000 步评 50 episode，取最后 5 个 checkpoint 的最高成功率。

| 方法 | RoboCasa 24 任务均值 | GR1 24 任务均值 |
| --- | --- | --- |
| Diffusion Policy | 51.7% | 40.9% |
| GR00T N1 (Scratch) | 60.6% | 45.1% |
| UWM（5× 训练预算） | 60.8% | 29.5% |
| FLARE (Policy Only) | 61.9% | 44.0% |
| **FLARE** | **70.1%** | **55.0%** |

两个关键结论：

1. **FLARE 全面领先**，且超越了拿到 5 倍训练预算的 UWM；把 Policy-Only 步数翻倍到 160k 仅得 44.1%，说明**增益不来自更多训练步**。
2. 即便只用 Policy 目标，FLARE 也能与用更大 VLM 骨干的 GR00T N1 (Scratch) 持平，佐证 **Q-former 嵌入本身质量很高**。

### 4.2 数据高效后训练（跨本体预训练嵌入）

用 3.3 节的跨本体预训练嵌入作对齐目标，在 24 RoboCasa + 4 真实 GR1 任务上做数据受限后训练（FLARE 只 warm-start VL 嵌入，Policy-Only baseline 则同时初始化 VL 嵌入与 DiT）：

| RoboCasa 后训练 | Policy Only | FLARE |
| --- | --- | --- |
| 24 × 100 轨迹 | 42.3% | **52.1%**（+10%） |
| 24 × 300 轨迹 | 60.2% | **66.4%** |
| 24 × 1000 轨迹 | 65.3% | **71.3%** |

- 数据越少增益越大（100 条/任务时 +10%）。
- 尤为关键：预训练嵌入**从未见过 RoboCasa**，但用 1000 条做对齐目标达 **71.3%**，已**追平在 RoboCasa 上专门训练的 in-domain 嵌入（70.2%）**，说明跨本体嵌入泛化性极强。
- 真实 GR1 人形：平均成功率 81.2% → **95.3%**（+14%，最高 95.1%）。定性上 FLARE 学会绕/悬停过水瓶、可乐罐再抓取，而 Policy-Only 常把物体撞倒——这正是"未来潜在推理"带来的好处。

### 4.3 利用无动作标签的人类第一视角视频

最具扩展性的能力。选 5 个训练集外、几何独特（需新抓取策略，如大卷蓝胶带需顶部抓取）的新物体：每个物体采 150 条人类 GoPro 头戴视频 + 仅 10 条机器人遥操作，混合 GR-1 预训练数据共同训练。

- **真实机器人数据**（有动作）→ 同时用 $\mathcal{L}_{fm}$ + $\mathcal{L}_{align}$；
- **人类视频**（无动作）→ **仅用 $\mathcal{L}_{align}$** 学习潜在动力学。

| 后训练数据 | FLARE | FLARE + 人类视频 |
| --- | --- | --- |
| 1 条机器人轨迹/物体 | 37.5% | **60.0%** |
| 10 条机器人轨迹/物体 | 42.5% | **80.0%** |

仅 1 条遥操作即达 60%；10 条 + 人类视频协同达 80%，约为纯动作标签 baseline 的两倍（抓住但未放入篮子给 0.5 部分分）。

### 4.4 消融实验

**目标嵌入模型**（Table 2，GR1 仿真）：

| 配置 | 成功率 |
| --- | --- |
| No FLARE loss | 43.9% |
| SigLIP2（256 token/图） | 49.6% |
| SigLIP2（2×2 平均池化，64 token/图） | 50.9% |
| **动作感知嵌入** | **55.0%** |

即便用通用 SigLIP2 也有约 7% 的稳定增益，说明框架**对 teacher 编码器鲁棒**；动作感知嵌入最优。

**FLARE loss 应用层 $L$**（图 8，共 8 层）：主实验取**第 6 层**。太浅（第 4 层）显著掉点——对齐目标必须与动作去噪过程"对齐"，太早施加会与动作预测冲突。

**系数 $\lambda$**（图 8）：在 0.1–0.5 范围内都稳健（均超 43.9% 的 Policy-Only），$\lambda=0.2$ 附近最优。

**EMA 系数 $\rho$**（图 9，24×300 RoboCasa）：

| $\rho$ | Baseline | 0.99 | 0.995 | 0.999 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| 成功率 | 60.1% | 63.5% | **66.4%** | 65.7% | 64.6% |

所有 EMA 变体都超 baseline；$\rho=0.995$ 最佳；$\rho=1.0$（无 EMA）仍超 baseline；$\rho=0.99$ 最差（更新太频繁不稳定）。

> 多任务实验训练配置：32 张 H100、batch 1024、80k 步，其余超参与预训练一致。

---

## 五、局限性与未来方向

论文指出的主要局限：

1. **任务局限**：主要验证在模仿学习 + pick-and-place；更精细的灵巧操作、以及把 RL 引入训练范式仍待探索。
2. **仍需少量演示**：泛化到新物体依然依赖少量专家演示，在难以采集数据的场景可能受限。
3. **受控视频采集**：人类视频是头戴 GoPro 的受控环境数据，扩展到自然场景的大规模 egocentric 数据是重要方向。
4. **聚焦协同训练而非规划**：FLARE 面向策略与世界模型**协同训练**，而非 DINO-WM 式的零样本规划——把规划作为下游扩展是 future work。

---

## 六、个人思考

### 6.1 "隐式 vs. 显式"世界模型的路线之争

FLARE 与项目中 [DreamVLA](DreamVLA_2025.md)、[FutureVLA](FutureVLA_2026.md) 同属"预测未来 → 辅助决策"范式，但在**预测什么、怎么预测**上各走一路：

| | FLARE | DreamVLA | FutureVLA |
| --- | --- | --- | --- |
| **预测目标** | 未来观测的动作感知 VL 嵌入（单一紧凑潜在） | 三类结构化世界知识（动态/深度/语义） | 联合视觉-运动潜在嵌入（JVPM） |
| **是否重建** | **完全不重建**（只对齐 cos 相似度） | 各类知识分别解码 | 3D-VAE 编码 17 帧 |
| **解耦方式** | 独立 future token 流 + 中间层切片 | block-wise 结构化注意力 | 双流解耦监督 + 门控交叉注意力 |
| **teacher** | 自训的动作感知 Q-former 嵌入 | CoTracker/Depth Anything/DINOv2 | 3D-VAE |
| **推理开销** | 零额外开销（future token 仅训练用） | 需推 dream 查询 | 推理零开销（潜在对齐迁移先验） |

FLARE 是其中**最"轻"**的一支：不定义任何具体语义（动态/深度/语义），不做任何重建，纯靠"对齐未来嵌入"。它和 FutureVLA 的"潜在对齐迁移先验、推理零开销"思想高度同源，差别在 FLARE 把 teacher 做成了**动作感知**的专用嵌入而非通用 3D-VAE。

### 6.2 "动作感知 teacher" 是性能的真正来源之一

消融里 No-FLARE（43.9%）→ 通用 SigLIP2（49.6%）→ 动作感知嵌入（55.0%）的阶梯说明：**对齐目标的"任务相关性"比对齐机制本身更值钱**。这与 [DreamVLA](DreamVLA_2025.md) 发现"动态区域（二值掩码）比光流（连续场）更好"、以及 [SF](../perception/SF_2025.md) 用 VGGT 3D 表征做隐式对齐的思路一致——**信息瓶颈 + 任务对齐**往往胜过"信息更全"。可以追问：如果把 FLARE 的 teacher 换成 VGGT/DINOv3 这类几何/语义更强的编码器，是否能进一步涨点？

### 6.3 无动作视频协同训练的杠杆

$\mathcal{L}_{align}$ 不需要动作标签，这一点让 FLARE 天然吃得下人类视频，是它相对 [WMPO](../rl/WMPO_2025.md) 等"必须有动作/奖励"的世界模型路线的结构性优势。1 条机器人轨迹 → 60%、加人类视频 → 80% 的结果很有说服力。这与 [UniVLA](../foundation/UniVLA_2025.md)、[EC-Flow](../../imitation-learning/EC_Flow_2025.md) 等"从无动作视频学策略"的主线呼应，但 FLARE 不需要潜在动作量化或光流提取工具，工程更简洁。

### 6.4 与 GR00T 谱系的关系

FLARE 出自 GR00T N1 同一团队（Linxi Fan、Yuke Zhu 等），并已作为 "FLARE Integration" 落进 **GR00T N1.5**。它本质是给 GR00T N1 的流匹配 DiT 头加一个对齐辅助损失——这解释了为什么实验里 GR00T N1 (Scratch) 是核心 baseline。把它看作 [π₀](../foundation/pi0_2024.md) / GR00T N1 流匹配范式上的一个**即插即用世界建模插件**，比看作独立架构更准确，这也是我把它归入 `reasoning/`（世界模型）而非 `foundation/` 的理由。

---

## 参考

- GR00T N1（NVIDIA, 2025）：FLARE 的架构底座与核心 baseline，FLARE 已集成进其 N1.5 版本
- π₀（Black et al., 2024）：流匹配 VLA 基础，FLARE 沿用其 Beta 时间步采样与 $K=4$ 去噪
- REPA（Yu et al., ICLR 2025）：表示对齐加速图像扩散，FLARE 将其从"对齐当前"改为"对齐未来"
- UWM（Zhu et al., 2025）：联合去噪视频 VAE latent + 动作的显式世界模型，FLARE 的主要对比基线
- DINO-WM（Zhou et al., 2024）：用 DINO 特征做潜在动力学 + 零样本规划，与 FLARE 的"协同训练"定位互补
- SigLIP-2（Tschannen et al., 2025）：动作感知嵌入的视觉/文本骨干（siglip2-large-patch16-256）
- BLIP-2 / Q-former（Li et al., 2023）：嵌入模型用以把 288 融合 token 压缩为 32 query token
