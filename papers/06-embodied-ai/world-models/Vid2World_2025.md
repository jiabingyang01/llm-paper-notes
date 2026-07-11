# Vid2World：将视频扩散模型改造为交互式世界模型

> **论文**：*Vid2World: Crafting Video Diffusion Models to Interactive World Models*
>
> **作者**：Siqiao Huang、Jialong Wu（共同一作）、Qixing Zhou、Shangchen Miao、Mingsheng Long
>
> **机构**：Tsinghua University；Chongqing University
>
> **发布时间**：2025 年 05 月（arXiv 2505.14357）
>
> **发表状态**：ICLR 2026（Published as a conference paper at ICLR 2026）
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.14357) | [PDF](https://arxiv.org/pdf/2505.14357)
>
> **分类标签**：`世界模型` `视频扩散` `因果化` `动作条件` `机器人操作`

---

## 一句话总结

Vid2World 系统性地把一个在互联网规模视频上预训练的**被动、全序列、非因果**视频扩散模型（DynamiCrafter，约 1.1B 参数）改造成**自回归、可交互、逐帧动作条件**的世界模型,核心是两招——**video diffusion causalization**（架构上做因果化的 Extrapolative Weight Transfer + 训练用 Diffusion Forcing）与 **causal action guidance**（逐帧注入动作 + 动作维度的 classifier-free guidance);在机器人操作(RT-1)、3D 游戏(CS:GO)、开放世界导航(RECON)三域全面领先,其中 CS:GO 相对最佳基线 FID 提升 79.9%、FVD 提升 71.1%。

## 一、问题与动机

世界模型(world model)能从历史观测与动作预测未来状态,是序列决策的关键组件。但现有世界模型有两大痛点:

1. **数据饥渴**:传统世界模型只依赖 in-domain 的带动作标注数据,或至多扩展到 cross-domain 的带动作数据。这类数据采集昂贵、量少,导致预测低保真、缺乏物理真实感,难以泛化到复杂环境。
2. **未被利用的最大数据源**:互联网规模的**无动作标注视频**(action-free video)才是"世界模型数据金字塔"最底层、最丰富、最易采集、蕴含最强世界先验的部分。视频扩散模型(如 Sora、Veo)已经证明能生成高保真、多样的真实动力学视频。

作者的主张:与其在海量视频上从零训练一个世界模型,不如**把预训练视频扩散模型的物理先验与生成能力直接迁移**过来。这是一条更省成本、更平滑的路径("从非因果生成迁移到因果生成,本身可能比直接学因果模型更容易")。但要把被动视频扩散模型改造成交互世界模型,横亘着两道结构性障碍(见原文 Figure 2):

- **无法因果生成**:标准视频扩散用双向时序注意力,未来帧会影响过去帧,违背自回归 rollout 中"未来严格依赖过去"的因果律。
- **缺乏动作条件**:视频扩散通常只吃粗粒度的 video-level 条件(如文本 prompt),没有细粒度、逐帧的动作条件机制,也就无法做反事实推理(不同动作导致不同未来)。

## 二、核心方法

Vid2World 把改造拆成两个层次:先 **causalization**(4.1 节)把非因果架构变因果并适配 Diffusion Forcing 训练目标,再 **causal action guidance**(4.2 节)注入并强化动作信号。

### 2.1 预备:视频扩散与逐帧噪声

视频扩散把样本表示为帧序列 $(\mathbf{x}_t^{k_t})_{1 \le t \le T}$,其中 $t$ 是帧下标、$k_t$ 是该帧的噪声水平。训练用标准的噪声预测损失:

$$\mathcal{L}(\theta) = \mathbb{E}_{k,\boldsymbol{\epsilon},\mathbf{x}^0}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}^k, k)\|^2\right]$$

**用大白话说**:给干净帧加已知噪声,让网络猜出加了什么噪声,猜准了就学会了去噪。传统做法所有帧共用同一噪声水平;Diffusion Forcing(Chen et al. 2024)改为**每帧独立采样噪声水平** $k_t \sim \mathcal{U}(0,K)$。把历史帧设 $k_t=0$(干净)、未来帧设 $k_t=K$(纯噪声)、当前帧沿 $\{K,\dots,0\}$ 逐步去噪,模型就获得了自回归生成的能力。

### 2.2 Video Diffusion Causalization(架构因果化)

**时序注意力层**:直接加因果 mask 即可。注意力本质是 query-key 点积,天然适应变长序列,限制感受野不影响已学到的 token 间关系,**无需改参数**。

**时序卷积层(难点)**:卷积核对称地聚合过去与未来帧特征,简单裁剪会浪费预训练权重。作者比较三种权重迁移策略(原文 Figure 3):

- **Shift Weight Transfer(SWT)**:把整个核向过去平移 $m$ 步。保留全部权重,但引入**时序错位**——第 $i$ 位现在聚合的是 $i-m$ 时刻的信息,无法保证产生相似表征。附录 A.1 给出反例:即使输入是完美线性(smooth)的 $f(t)=\alpha t$,SWT 的近似误差随 $\alpha \to \infty$ 无界增长,**不受平滑常数 $L$ 控制**,可能灾难性失败。
- **Masked Weight Transfer**:只保留过去/当前权重,其余置零(相当于初始化时施加硬因果 mask)。保证因果,但丢弃了 future-facing 权重里潜在有用的信息。
- **Extrapolative Weight Transfer(EWT,本文首选)**:核心思想是**沿时间维做局部线性外推来保留原卷积的输出表征**。假设未来帧特征可由过去 $p$ 帧线性外推:

$$\mathbf{z}_{t+k} \approx \sum_{j=0}^{p-1} \gamma_{k,j}\, \mathbf{z}_{t-j} + \boldsymbol{\beta}_k$$

其中系数由过去 $p$ 帧的 OLS 线性回归确定,且 $\sum_j \gamma_{k,j}=1$ 使截距 $\boldsymbol{\beta}_k$ 自动消失。目标是让新的因果计算尽量复现原非因果卷积 $\sum_{i=-m}^{m} w_i \mathbf{z}_{t+i} + \mathbf{b}$,于是把原本作用在未来帧上的权重 $\{w_i\}_{i>0}$,**按线性外推关系重新分配回过去帧**:

$$w_j' = \mathbf{1}_{[j \ge -m]}\cdot w_j + \mathbf{1}_{[-p+1 \le j \le 0]}\cdot \sum_{i=1}^{m}\gamma_{i,-j} w_i$$

**用大白话说**:因果化后不能再看未来帧了,但未来帧上原来那份"权重预算"不能白扔。EWT 假设"未来大致是过去的线性延续",于是把这份预算折算成"过去几帧的线性组合"再加回去,让改造后卷积的输出和原来尽量一致。DynamiCrafter 的时序卷积核大小为 3,故实际取 $m=1,p=2$,此时外推退化为最直观的形式 $\mathbf{z}_{t+k} \approx (k+1)\mathbf{z}_t - k\mathbf{z}_{t-1}$(即用最近两帧的斜率外推)。

附录 A.3 给出 EWT 的误差上界(Proposition 1),对二阶可微、$L$-smooth 的输入:

$$\|\mathbf{y}^{\text{orig}} - \mathbf{y}^{\text{EWT}}\|_2 \le \frac{L}{2}\sum_{i=1}^{m}|w_i|\left(i^2 + \frac{6p^2}{p+1} + \frac{(p-1)(p-2)}{6}\right)$$

**用大白话说**:与 SWT 的无界误差相反,EWT 的误差被输入平滑度 $L$ 干净地控制住了,这就是它更稳健的理论依据。

### 2.3 Causal Action Guidance(因果动作引导)

**因果动作注入**:预测第 $t$ 帧观测 $o_t$ 时,把上一时刻动作 $a_{t-1}$ 经一个轻量 MLP 编码后,加到时序位置 $t$ 的隐表征上——实现**逐帧、时序对齐**的细粒度动作条件(区别于 prior work 把整段动作压成一个 video-level embedding)。

**动作维度的 classifier-free guidance**:在自回归动作条件设定下引入 CFG。训练时以概率 $p$ 对每帧动作做 **action dropout**:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\sum_{t=0}^{T}\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta([\mathbf{x}_\tau^{k_\tau}]_{\le t}, [\tilde{\mathbf{a}}_\tau]_{<t}, [k_\tau]_{\le t})\|^2\right],\quad \tilde{a}_t = \begin{cases}\varnothing, & \text{w.p. } p\\ a_t, & \text{otherwise}\end{cases}$$

采样时按 $\boldsymbol{\epsilon}_{\text{guided}} = (1+\lambda)\boldsymbol{\epsilon}_{\text{cond}} - \lambda\,\boldsymbol{\epsilon}_{\text{ucond}}$ 引导,其中 $\lambda$ 为引导尺度、无条件分支把最近动作 mask 成 $\varnothing$。

**用大白话说**:让模型既学"给了动作的预测"又学"不给动作的预测",推理时用两者之差把生成往"更贴合当前动作"的方向推;调 $\lambda$ 就能在测试时灵活控制对动作变化的响应强度。

**Theorem 4.1(动作引导 = 概率转向)**:令 $\mathcal{H}_t := ([\mathbf{x}_\tau]_{\tau<t}, [\mathbf{a}_\tau]_{\tau<t-1})$ 为不含当前动作的历史。上述 score 组合等价于从如下"锐化后验"采样($\omega \propto (1+\lambda)$ 为常数):

$$\tilde{p}(\mathbf{x}_t \mid \mathbf{a}_{t-1}, \mathcal{H}_t) \propto p(\mathbf{x}_t \mid \mathcal{H}_t)\cdot p(\mathbf{a}_{t-1} \mid \mathbf{x}_t, \mathcal{H}_t)^{\omega}$$

**用大白话说**:引导项相当于一个隐式分类器,把生成拉向"与用户当前动作最匹配"的区域,同时保留视频先验的高保真生成。附录 A.5 进一步指出:通过 $\text{do}(a_t)$ 式地注入并放大动作似然,该机制实质在执行 interventional causality(干预式因果),让模型生成"该动作导致的反事实未来"而非仅依观测相关性的通用未来。

## 三、实验结果

基座:DynamiCrafter(约 1.1B 可训练参数,320×512 分辨率的 latent video diffusion,基于 Stable Diffusion VAE + 3D U-Net)。机器人域后训练 100k 步(约 4×A100 上 7 天)。指标:FVD、FID、SSIM、LPIPS、PSNR、DreamSim。† 表示非自回归(NAR)预测、* 表示自回归(AR)预测、‡ 表示单步预测。

**机器人操作(RT-1 数据集)** —— Vid2World-NAR 与其他 NAR 迁移法比,几乎全面最优;Vid2World(AR,更难)在 FVD/FID 上仍领先:

| 模型 | FVD ↓ | FID ↓ | SSIM ↑ | LPIPS ↓ | PSNR ↑ | DreamSim ↓ |
|---|---|---|---|---|---|---|
| Pre-trained Base† | 237.6 | 5.432 | 0.712 | 0.228 | 20.6 | - |
| ControlNet† | 27.1 | 3.248 | 0.836 | 0.148 | 24.5 | - |
| Action-Conditioned† | 24.2 | 2.965 | 0.852 | **0.134** | **25.6** | - |
| AVID† | 39.3 | 3.436 | 0.842 | 0.142 | 25.3 | - |
| **Vid2World-NAR†** | **18.7** | 5.871 | **0.856** | 0.140 | **25.8** | **0.048** |
| **Vid2World*(AR)** | **18.5** | 5.806 | 0.842 | 0.152 | 24.6 | 0.054 |

(注:FID 一列 Action-Conditioned 等 NAR 基线更低,但作者指出这些是 non-autoregressive、任务更简单;Vid2World 是逐帧 AR。)

**3D 游戏模拟(CS:GO,对比 SOTA 自回归世界模型 DIAMOND)** —— 全指标大幅领先:

| 模型 | FVD ↓ | FID ↓ | SSIM ↑ | LPIPS ↓ | PSNR ↑ | DreamSim ↓ |
|---|---|---|---|---|---|---|
| DIAMOND-Fast* | 577.1 | 115.6 | 0.449 | 0.547 | 18.2 | 0.2817 |
| DIAMOND-HQ* | 368.5 | 87.2 | 0.447 | 0.510 | 18.3 | 0.2416 |
| **Vid2World*** | **106.6** | **17.5** | **0.481** | **0.404** | **18.7** | **0.135** |

相对最佳基线,**FID 提升 79.9%、FVD 提升 71.1%**。

**开放世界导航(RECON,对比 Navigation World Model NWM)** —— Vid2World 是自回归(受误差累积),NWM 是单步预测(‡,更容易),但 Vid2World 在单步设定下 6 项指标中有 4 项超过 NWM+Ego4D,并在 SSIM/PSNR 上超过原始 NWM:

| 模型 | FVD ↓ | FID ↓ | SSIM ↑ | LPIPS ↓ | PSNR ↑ | DreamSim ↓ |
|---|---|---|---|---|---|---|
| NWM (1B)‡ | **31.2** | **34.1** | 0.389 | **0.295** | 15.343 | **0.091** |
| NWM+Ego4D (1B)‡ | 41.0 | 34.9 | 0.361 | 0.368 | 14.072 | 0.138 |
| **Vid2World*** | 59.4 | 42.9 | **0.481** | 0.324 | **16.10** | 0.108 |

值得注意:Vid2World 用历史长度 4 + 预测 16 帧 = 总上下文 20,超过训练时的 16,展现了强时序泛化。

**Real2Sim 策略评估(RT-1 关抽屉任务,Figure 5)** —— 用世界模型 rollout 三个训练阶段的策略并由人工判成功,与真实成功率高度一致:

| 策略 | 真实成功率 | Vid2World 估计 |
|---|---|---|
| RT-1 (Begin) | 0.000 | 0.260 |
| RT-1 (15%) | 0.889 | 0.820 |
| RT-1 (Converged) | 0.926 | 0.880 |

**消融(30k 步,Table 2)** —— 验证权重迁移(WT)与动作引导(AG)的贡献:

| WT 策略 | AG | FVD ↓ | FID ↓ | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|---|---|---|---|---|---|---|
| Shift | | 29.9 | 7.85 | 0.799 | 0.185 | 21.5 |
| Masked | | 29.4 | 7.07 | 0.824 | 0.169 | 22.9 |
| Extrapolative | | 28.6 | 7.52 | 0.832 | 0.162 | 23.4 |
| Masked | ✓ | 25.8 | 6.84 | **0.840** | **0.159** | 23.9 |
| **Extrapolative** | ✓ | **22.4** | **6.16** | 0.839 | **0.159** | 23.9 |

结论:Masked/Extrapolative 均优于 Shift(印证 SWT 的错位问题);Extrapolative 略优于 Masked;开启 AG 一致提升。引导尺度 $\lambda$ 存在最优值(Figure 8),过大会因 over-sharpening 反而变差。

## 四、局限性

- **推理慢**:基座参数大 + 扩散迭代去噪,相比 teacher-forcing 训练的模型推理速度慢,限制了在需实时交互(如 RL 训练循环)场景的落地;作者未在世界模型内做 RL 策略训练。
- **训练耗时**:即便用相对轻量的 1.1B 基座,机器人域仍需 100k 步/约 4×A100×7 天;作者明言"训练过程仍较耗时",期待更少步数达到相当性能。
- **基座规模受限**:受算力约束只用了 1.1B 模型,未验证在 Genie-3/V-JEPA-2/Cosmos 等工业级基座上的表现,作者猜测更大基座会更好但未证实。
- **EWT 的线性外推假设**:核大小为 3 时只能用 $m=1,p=2$ 的一阶外推;高阶外推可能更好但设计更复杂,留作未来工作。误差界依赖输入二阶可微 $L$-smooth,对高频/突变内容假设可能偏理想。
- **导航域 AR 指标**:在 FVD/FID 上并未超过单步 baseline,自回归误差累积在长 rollout 下仍是隐忧(Figure 6 显示优势主要体现在与 DIAMOND 的对比,对 NWM 是"持平/可比")。

## 五、评价与展望

**优点**。(1) 问题定位精准:把"如何复用互联网视频先验做世界模型"这一开放问题,归结为 causalization + action conditioning 两个可操作的子问题,并给出清晰的 recipe。(2) EWT 是本文最有分量的技术贡献——它不是简单的 causal mask,而是从"保留原卷积输出表征"出发的原则性权重再分配,且配了反例(A.1)和误差界(A.3)的理论支撑,把工程 trick 上升为可分析的方法。(3) Theorem 4.1 把 action-CFG 解释为对后验 $p(a_{t-1}\mid x_t)$ 的锐化,与 interventional causality 挂钩,理论叙事完整。(4) 三域(操作/游戏/导航)+ Real2Sim 的评测覆盖面广,数字扎实。

**与公开工作的关系**。与 UniSim(Yang et al. 2024,学习交互式真实世界模拟器)、AVID(Rigter et al. 2024,用 adapter 调制冻结视频模型)、iVideoGPT(Wu et al. 2024)、GameFactory / DIAMOND(游戏世界模型)、NWM(导航世界模型)相比,Vid2World 的差异在于**同时解决时序因果性与逐帧动作条件**,并直接在架构与权重层面迁移(而非仅加 adapter 或从零训)。其 Diffusion Forcing 训练目标沿用 Chen et al. 2024;与 AVID 的"冻结主干 + adapter"路线形成对照——本文选择改造主干本身,消融显示这带来了更强性能但也牺牲了轻量性。

**开放问题与可能改进**。(1) **加速**:结合一步/少步扩散(consistency models、mean flows、self-forcing)或 KV-cache 式缓存,是让 Vid2World 真正可用于闭环控制/RL 的关键瓶颈。(2) **更大基座 + scaling law**:1.1B → 工业级基座能否线性受益,以及 EWT 在更大核/更深网络上的稳健性,值得系统研究。(3) **更强的干预式因果**:当前 action guidance 仍是似然锐化,能否引入显式的因果结构或反事实一致性约束,提升多分支预测的可信度。(4) **长程 rollout 的误差累积**:导航域 AR 指标未全面超越单步 baseline,提示需要更强的抗漂移机制(如 rollout 训练、误差反馈)。(5) EWT 的高阶外推($p>2$)与非平滑内容下的行为是自然的后续方向。

总体看,这是一篇"把预训练视频扩散当作世界模型基座"这一范式下,方法完备、理论工整、实证扎实的代表作,主要短板集中在推理效率与长程稳定性。

## 参考

1. Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-sequence Diffusion*. NeurIPS 2024.（逐帧独立噪声水平的训练目标,Vid2World 因果训练的基础）
2. Xing et al. *DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors*. ECCV 2024.（Vid2World 的预训练基座）
3. Rigter et al. *AVID: Adapting Video Diffusion Models to World Models*. RLC 2024.（adapter 式迁移的对照工作,本文主要 baseline）
4. Alonso et al. *Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND)*. NeurIPS 2024.（自回归扩散世界模型,CS:GO 域主要 baseline）
5. Bar et al. *Navigation World Models (NWM)*. CVPR 2025.（导航世界模型,RECON 域主要 baseline）
