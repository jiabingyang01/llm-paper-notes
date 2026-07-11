# PERSIST：超越像素历史——具备持久 3D 状态的世界模型

> **论文**：*Beyond Pixel Histories: World Models with Persistent 3D State*
>
> **作者**：Samuel Garcin\*、Thomas Walker\*（共同一作）、Steven McDonagh、Tim Pearce、Hakan Bilen、Tianyu He、Kaixin Wang、Jiang Bian
>
> **机构**：University of Edinburgh；Microsoft Research
>
> **发布时间**：2026 年 06 月（arXiv 2603.03482）
>
> **发表状态**：ICML 2026（Proceedings of the 43rd International Conference on Machine Learning, PMLR 306）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.03482) | [PDF](https://arxiv.org/pdf/2603.03482)
>
> **分类标签**：`世界模型` `持久3D状态` `自回归视频扩散` `可微渲染`

---

## 一句话总结

PERSIST 把交互式世界模型的"记忆载体"从像素历史换成了一个以智能体为中心、随时间自回归演化的**潜 3D 场景（world-frame）**，通过"3D 世界演化 + 相机预测 + 可微投影 + 学习式着色"三段式分解,使长时程生成的空间一致性与稳定性大幅提升——在 Luanti(类 Minecraft)环境上 600 帧 rollout 的 FVD 从 Oasis 的 875 降到 148(带真值 3D 初始化的 PERSIST+$w_0$ 更降到 104),且 FID 几乎不随时间退化。

## 一、问题与动机

现有交互式世界模型主流走**自回归视频扩散(AR video diffusion)**路线:模型对过去若干帧观测与动作做因果注意力,逐帧去噪生成下一帧(Oasis、GameNGen、WorldMem 等)。这条路线天然支持实时动作条件、可无限展开,但存在两个结构性缺陷:

1. **3D 一致性只能隐式学习**。模型没有显式 3D 表征,几何一致性全靠数据里"悟"出来,长时程容易漂移、穿模、地形反复变化。
2. **空间记忆受限于有限上下文窗口**。高维像素观测编码代价高,硬件只能塞进"几秒钟"的过去帧。一旦生成超过这个时域,历史就丢了。折中方案是维护一个像素观测的**记忆库(memory bank)**并检索关键帧(WorldMem 等),但像素观测是**局部的、冗余的、视角相关的**,记忆库越大检索越难,检索代价随 episode 长度增长。

作者的核心主张:**放弃像素历史,用"主动关键帧生成(active key frame generation)"替代"关键帧检索"**。灵感来自传统 3D 模拟器/游戏引擎——它们的一致性来自"从一个持久且动态演化的 3D 状态中渲染像素",而非缓存像素。于是把"记忆"放进一个显式演化的 3D 潜状态里,信息检索代价与 episode 长度无关。

## 二、核心方法

### 2.1 形式化:用代理隐状态近似环境

把交互环境定义为元组

$$
\mathcal{E} = \langle \mathbb{S}, \mathbb{O}, \mathbb{A}, \Omega, p \rangle
$$

其中 $\mathbb{S}$ 为潜状态空间、$\mathbb{O}$ 为观测空间、$\mathbb{A}$ 为动作空间、$\Omega:\mathbb{S}\to\mathbb{O}$ 为部分投影(观测函数)、$p(s'\mid a,s)$ 为转移概率。真实隐状态 $s$ 既不可测又难以从中还原观测,作者用一个**代理隐状态**替代:

$$
\bar{s} = \langle \boldsymbol{w}, \boldsymbol{c} \rangle
$$

$\boldsymbol{w}$ 是以智能体为中心、固定大小空间区域的 **world-frame(3D 体素场景)**,$\boldsymbol{c}$ 是编码智能体视角的**相机状态**。

**用大白话说**：与其去猜测游戏程序内部那团没法直接渲染的隐藏内存,不如显式维护"我周围这片 3D 世界长啥样(w)"和"我此刻从哪个角度看它(c)"——前者当空间记忆,后者当查询这块记忆的"钥匙"。

### 2.2 底座:Rectified flow matching + Diffusion forcing

生成模块建立在 rectified flow 上。噪声过程线性插值 $x^\tau=(1-\tau)x^0+\tau x^1$($x^1\sim\mathcal{N}(0,\mathbb{I})$),网络预测指向干净数据的速度场 $v=x^0-x^1$,条件流匹配损失:

$$
\mathcal{L}(\theta) = \left\| \mathcal{V}_\theta(x^\tau,\tau) - (x^0 - x^1) \right\|^2, \quad \tau \sim p(\tau)
$$

推理时从纯噪声按步长 $d^k$ 迭代去噪:$x^{\tau-d^k}=x^\tau+\mathcal{V}_\theta(x^\tau,\tau)\,d^k$。自回归生成采用 **diffusion forcing**:训练时每帧独立采样噪声等级,推理时当前帧从随机噪声去噪、过去上下文帧只加一个小的固定噪声 $\tau_\text{ctx}$。

**用大白话说**：flow matching 负责"从噪声一步步画出这一帧";diffusion forcing 让"已经画好的过去帧"带一点噪声再喂回去,好让模型对自己上一步的瑕疵不那么敏感。

### 2.3 三段式分解

PERSIST 把世界模拟拆成三个耦合模块,各自单独训练、推理时拼装、无需联合微调。

**(a) World-frame 模型 $\mathcal{W}_\theta$——3D 世界怎么演化。**每步采样新的 world-frame:

$$
\bar{\boldsymbol{w}}_t \sim \mathcal{W}_\theta\!\left(\bar{\boldsymbol{w}}_t \mid \bar{W}^{t-1}_{t-K},\, A^t_{t-K},\, C^{t-1}_{t-K-1},\, \bar{O}^{t-1}_{t-K-1}\right)
$$

$K$ 为时域上下文窗口。骨干是**因果 DiT**:空间模块改造成处理三维,把 RoPE 空间位置编码换成"每个体素 token 质心 XYZ 坐标"的位置嵌入;动作与相机经 MLP 联合嵌入,通过 AdaLN 注入;像素补丁再拼接由相机算出的 Plücker 嵌入(携带 3D 投影信息)经 cross-attention 注入;最后用 3D-VAE 解码回体素。关键点:$\mathcal{W}_\theta$ 支持在 $W=\varnothing$ 条件下工作,因此能仅凭初始条件 $\langle \boldsymbol{o}_0,\boldsymbol{c}_0\rangle$ 生成初始 world-frame $\boldsymbol{w}_0$,推理时不依赖真值 3D。

**(b) 相机模型 $\mathcal{C}_\theta$——视角作为记忆查找键。**相机是 10 维向量 $\boldsymbol{c}=\langle \text{pos},\text{rot},\text{fov}\rangle$(位置 $\in\mathbb{R}^3$、6D 连续旋转 $\in\mathbb{R}^6$、视场角 $\in\mathbb{R}$)。用 1D 因果 transformer(RoPE 时域编码)预测残量 $\bar{\boldsymbol{c}}=\langle \text{pos},\Delta\text{pitch},\Delta\text{yaw},\Delta\text{fov}\rangle$,MSE 训练。相机在此充当**空间查找键**:给定 $\boldsymbol{c}_t$ 就能从 $\boldsymbol{w}_t$ 里投影出重建当前观测所需的那部分 3D 信息,而无需知道真实观测函数。

**(c) World-to-pixel 生成 $\mathcal{P}_\theta$——可微投影 + 学习式延迟着色。**先用投影算子建立世界到屏幕的对应:

$$
\mathcal{R}(\boldsymbol{c}, \boldsymbol{w}) = (\tilde{\boldsymbol{w}}_{2D}, \boldsymbol{d})
$$

$\tilde{\boldsymbol{w}}_{2D}\in\mathbb{R}^{h\times w\times l\times m}$ 是每像素、按深度排序的 $l$ 层体素特征($m$ 通道),$\boldsymbol{d}$ 是线性深度;二者通道拼接为 $\boldsymbol{w}_{2D}\in\mathbb{R}^{h\times w\times z}$,$z=l\times(m+1)$。实现上用 GPU 原生三角光栅化(把体素特征贴到静态体素网格 mesh 的面上)+ **depth-peeling** 得到深度有序特征栈。像素帧再由

$$
\bar{\boldsymbol{o}}_t \sim \mathcal{P}_\theta\!\left(\bar{\boldsymbol{o}}_t \mid {W_{2D}}^t_{t-K},\, A^t_{t-K},\, \bar{O}^{t-1}_{t-K-1}\right)
$$

生成。$\mathcal{P}_\theta$ 是一个**学习式延迟着色器(deferred shader)**:补上 3D 潜表征里没有的纹理、光照、粒子效果、屏幕叠加等,骨干为因果 DiT。关键设计:给 $\boldsymbol{w}_{2D}$ 分配的潜通道数**多于** $\bar{\boldsymbol{o}}$,以偏置模型把 3D 潜帧当作主要信息源。

**用大白话说**：$\mathcal{W}_\theta$ 先把"世界这一步长成什么样"以 3D 体素形式画好;$\mathcal{C}_\theta$ 决定"相机站哪";投影算子像游戏引擎的 G-buffer 一样把 3D 场景拍平成带深度的屏幕特征;$\mathcal{P}_\theta$ 再当"神经着色器"给它上色补细节。像素只是渲染产物,3D 才是记忆本体。

### 2.4 缓解 exposure bias

去噪器训练时条件的是真值编码,推理时却要条件在自己(以及其他模块)的预测上,产生训练—推理分布错配。除 diffusion forcing 外,作者额外在训练 $\mathcal{W}_\theta$ 时给 $\bar{O}$ 加 10% 随机噪声、训练 $\mathcal{P}_\theta$ 时给 $\bar{W}$ 加 10% 噪声,以增强对彼此不完美预测的鲁棒性。

## 三、实验结果

**环境与数据**:开源体素引擎 **Luanti**(类 Minecraft)+ Craftium 平台,采集约 4000 万次环境交互、约 10 万条轨迹、460 小时 24Hz 游戏。3D 观测为以智能体为中心的 $48^3$ 体素栅格。动作编码为 23 维多热向量(按键 + 离散化鼠标)。2D/3D-VAE 分别把像素/世界帧压成 $10\times10$ 像素、$4^3$ 体素的潜补丁。训练用 8×A100 / 8×H100,3D-XL 与像素去噪器各约 10 天。推理默认每帧 20 步去噪。基线 Oasis(滑窗最近 K 帧)、WorldMem(相机检索关键帧)共用同一 DiT 骨干。评测用从留出世界收集的 168 条轨迹。

**Table 1｜FVD($\downarrow$),不同 episode 长度**

| 方法 | 200 帧 | 400 帧 | 600 帧 |
| --- | --- | --- | --- |
| Oasis | 409 | 687 | 875 |
| WorldMem | 358 | – | – |
| No-3D-Upscale(消融) | 216 | 231 | 247 |
| Camera-GT(消融) | 161 | 152 | 152 |
| PERSIST-S(小 3D 去噪器) | 159 | 170 | 179 |
| **PERSIST(基础配置)** | **129** | **141** | **148** |
| PERSIST+$w_0$(真值 3D 初始化) | 80 | 93 | 104 |

要点:纯像素历史基线(Oasis/WorldMem)FVD 随时程急剧恶化;PERSIST 各配置在长时程上保持稳定,凸显条件在 3D 状态上的价值。消融 No-3D-Upscale(跳过 3D-VAE 上采样、直接投影低分辨率潜帧)惩罚最大,说明高分辨率 3D 潜帧对屏幕对齐很关键;而 PERSIST-S 仅略差,说明底层 3D 表征对空间压缩(3D-VAE + 3D-S 去噪器组合最高达 512×)高度鲁棒。Camera-GT(用真值相机)FVD 与基础版几乎相同,但会引入 FVD 捕捉不到的物理不一致(智能体穿墙、悬浮)。

**Table 2｜人类主观评分(1–5,28 名参与者、800+ 视频)**

| 方法 | 视觉保真 VF↑ | 3D 一致 3D↑ | 时序一致 Temp↑ | 总体 Overall↑ |
| --- | --- | --- | --- | --- |
| Oasis | 2.1±0.1 | 1.9±0.1 | 1.8±0.1 | 1.9±0.1 |
| WorldMem | 1.7±0.09 | 1.7±0.09 | 1.5±0.08 | 1.5±0.07 |
| PERSIST-S | **2.8±0.1** | **2.7±0.1** | **2.5±0.1** | **2.6±0.09** |
| PERSIST | **2.8±0.09** | 2.5±0.09 | **2.5±0.09** | **2.6±0.08** |
| PERSIST+$w_0$ | 3.2±0.1 | 2.8±0.1 | 2.8±0.1 | 3.0±0.1 |

所有指标上 PERSIST 各配置一致优于基线;PERSIST+$w_0$ 最优。有趣的是 PERSIST-S 虽 FVD 有惩罚,但人类几乎感知不到质量下降,说明 FVD 惩罚不转化为可感知退化。真值视频在所有指标上仍被一致偏好,揭示长时程世界建模与真实轨迹之间尚存明显差距。

**FID 随时间(Figure 6)**:Oasis 到第 600 帧 FID 冲到约 350,PERSIST 几乎不退化。PERSIST 起始 FID 略高,因为它为保证像素/世界帧对齐会重新生成初始观测;PERSIST+$w_0$(初始帧完美对齐)证明这仅是对齐伪影。作者还展示了一段 **2000 步(83 秒)**的 rollout:3D 表征局部会漂移(个别方块闪现/消失),但全局保持一致并对生成起净稳定作用,$\mathcal{P}_\theta$ 能借 $\boldsymbol{w}_{2D}$ 的接地信息从纹理漂移/穿模中恢复。

**推理效率**:PERSIST-S 每帧 2672 ms(A100,20 步),快于 Oasis(3007 ms)与 WorldMem(6387 ms),得益于 KV 缓存。组件耗时:world-frame 1538 ms、pixel 1111 ms、相机 22 ms、投影 0.63 ms——被两个迭代去噪器主导。把去噪步降到 2($\mathcal{W}_\theta$)/4($\mathcal{P}_\theta$)可得 3× 加速(886 ms,1.13 FPS),FVD 从 159/170/179 升到 207/230/244,无需微调或步蒸馏。

**涌现能力**:①**单图 3D 生成**——$\mathcal{W}_\theta$ 从单张 RGB 生成多样但连贯的初始 world-frame,对未见区域做外推;②**中途 3D 编辑**——任意时刻取出 $\boldsymbol{w}_t$ 手动编辑(改地形/生物群系/放置树木)再续跑;③**离屏持久动态**——3D 状态即使不被观测也会演化(洞穴灌水、水流淌到玩家身上),并支持与视野外物体的碰撞建模。

## 四、局限性

- **依赖真值 3D 监督**:三个模块训练都需 3D 标注(体素场景 + 相机),因此目前只适用于能提供 3D 标注的模拟器/数据集,难以直接迁到真实无标注视频。作者提议用 2D-to-3D 基础模型(VGGT、SAM 3D 等)合成 3D 标注作为预处理来松绑。
- **Exposure bias 仍在**:各模块独立训练、推理时相互条件在预测上,分布错配随时间累积;人类评测中真值仍被一致偏好,长 rollout 后期伪影更频繁。作者建议在生成 rollout 上做端到端后训练(类似 self-forcing)。
- **空间记忆有界**:只跟踪以智能体为中心的固定 3D 区域,智能体走远后远处信息被丢弃,无法在任意大环境中长期保持空间记忆。作者设想用 3D memory bank 按需加载空间块(相比像素记忆库天然去重、空间有序)。
- **非实时**:最快配置仅 1.13 FPS。
- 评测局限于单一(类 Minecraft)体素域,尚未验证到真实机器人/连续控制场景。

## 五、评价与展望

**优点。**这篇工作最有价值的贡献是把"游戏引擎的持久 3D 状态 + 延迟着色"这套经典图形学思想,系统性地嫁接到自回归扩散世界模型上,并给出一个干净的三段式可微分解(演化 / 查询 / 渲染)。相较 WorldMem 用"像素关键帧检索"续接记忆、Oasis 用"滑窗"硬扛,PERSIST 把记忆检索代价与 episode 长度解耦,这在原理上更可扩展;长时程 FID/FVD 几乎不退化的结果也确实支撑了"显式 3D 状态带来长时程稳定"的核心论点。把相机作为"对 3D 记忆的查询键"、把投影当作可微 G-buffer 的抽象也很优雅,并顺带解锁了单图 3D 生成、中途 3D 编辑、离屏动态等靠纯像素模型难以做到的能力。

**缺点与开放问题。**(1)最硬的瓶颈是**训练必须有真值 3D 与相机**,这几乎把方法锁死在可插桩的合成模拟器里;作者提出的"2D-to-3D 基础模型生成伪标注"是合理但未验证的路线,伪标注噪声会如何反噬 $\mathcal{W}_\theta$ 的一致性优势尚不清楚。(2)以智能体为中心的**有界体素窗口**意味着"持久"其实是局部持久,真正的开放世界持久记忆还需要 3D memory bank,这与 Genie 3 一类追求的无界一致性仍有距离。(3)三模块独立训练 + 推理拼装虽降低了训练复杂度,却把 exposure bias 显式化到跨模块条件里,长 rollout 的漂移不可避免;端到端 / self-forcing 后训练是自然的下一步。(4)体素 + $48^3$ 栅格对 Minecraft 式方块世界友好,但迁到连续几何、真实纹理、机器人操作场景时,3D 表征的选择(体素 vs. 高斯/隐式场)会成为关键设计决策。

**与公开工作的关系。**PERSIST 处在"交互式世界模型/神经游戏引擎"(GameNGen、Oasis、WorldMem、Matrix-Game、GameFactory、Genie 系列)与"显式 3D 生成"(Xiang 等的 structured 3D latents、Voyager、Marble)两条线的交叉点:前者强于交互与动作条件但缺 3D,后者有 3D 却多是静态场景。PERSIST 的独特定位是让 3D 表征**随时间演化并支持交互**,可视作把神经延迟着色(Thies 等)接进 AR 扩散骨干。对希望用"数据中学到的模拟器"来安全训练具身智能体的方向,这类持久 3D 世界模型是有吸引力的候选,但当前的真值 3D 依赖与非实时速度是落地前必须跨过的两道门槛。

## 参考

1. Decart et al. *Oasis: A Universe in a Transformer.* 2024.(纯像素滑窗 AR 世界模型基线)
2. Xiao et al. *WorldMem: Long-term Consistent World Simulation with Memory.* arXiv 2504.12369, 2025.(像素关键帧检索记忆基线)
3. Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-sequence Diffusion.* NeurIPS 2024.(AR 扩散稳定性,本文缓解 exposure bias 的基础)
4. Xiang et al. *Structured 3D Latents for Scalable and Versatile 3D Generation.* CVPR 2025.(本文 3D-VAE 架构来源)
5. Thies et al. *Deferred Neural Rendering: Image Synthesis using Neural Textures.* ACM TOG 2019.(学习式延迟着色思想来源)
