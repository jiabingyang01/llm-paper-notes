# Interactive Video WM Survey：面向交互式视频世界模型——前沿、挑战、基准与未来趋势

> **论文**：*Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends*
>
> **作者**：Jiuming Liu, Chaojun Ni, Mengmeng Liu, Chensheng Peng, Fangjinhua Wang, Sitian Shen, Marc Pollefeys, Masayoshi Tomizuka, Ayush Tewari, Per Ola Kristensson
>
> **机构**：University of Cambridge（剑桥大学）；Peking University（北京大学）；University of Twente（特文特大学）；University of California, Berkeley（加州大学伯克利分校）；ETH Zurich（苏黎世联邦理工，Marc Pollefeys 同时隶属 Microsoft）；University of Oxford（牛津大学）
>
> **发布时间**：2026 年 05 月（arXiv 2606.01164，v1 标注 31 May 2026）
>
> **发表状态**：未录用（预印本，采用 IEEE 期刊 LaTeX 模板排版）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.01164) | [PDF](https://arxiv.org/pdf/2606.01164)
>
> **分类标签**：`世界模型` `交互式视频生成` `综述` `具身智能` `长时程一致性`

---

## 一句话总结

这是一篇系统梳理**交互式视频世界模型（Interactive Video World Model, IWM）** 的综述：它把"从被动视频生成到闭环、可交互、动作条件化的世界模拟"这一转变拆解为**动作可控性、长时程交互与记忆、实时动作响应性**三大技术瓶颈，并横向对比了开放世界探索、游戏引擎、自动驾驶、具身 AI 四类应用的基准与代表方法，指出 2026 年前 5 个月相关 arXiv 论文数已激增至约 66 篇。

## 一、问题与动机

随着 AIGC 与多模态大模型的发展，世界模型（world model）通过建模真实环境的动力学，能对历史观测和动作做出反事实（counterfactual）的未来预测，服务于自动驾驶、具身 AI、游戏引擎等下游任务。作者指出，**交互性（interactivity）** 是世界模型的一项根本属性：人类正是通过持续与环境交互来提升认知与推理能力，机器智能亦然。

早期世界模型交互性有限。近年工作把**被动的视频生成**转化为**用户渐进动作驱动的主动介入**，形成了动作条件化（action-conditioned）的视频/3D 生成范式。作者以 Google 的 Genie 系列、Meta 的 WorldGen、NVIDIA 的 Lyra 2.0 等工业级里程碑为例，说明该方向已在学界与工业界同时爆发，但仍面临**长时程一致性、实时响应、有效且可泛化的动作注入、高效记忆检索**等挑战。文献中此前缺乏对这些进展、未决挑战与常用基准的系统综述——本文即填补此空白。

作者用 Fig. 2 对比了两种范式：传统视频扩散模型只做 **one-shot 指令**（一次性文本/图像条件），而 IWM 强调**frame-level / region-level 的多轮（multi-round）指令**，用户在每一轮基于最新输出继续下达动作，形成 human-in-the-loop 的闭环。

## 二、核心方法（综述的组织框架与技术脉络）

### 2.1 形式化定义

世界模型常用**部分可观测马尔可夫决策过程（POMDP）** 元组表示：

$$\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R} \rangle$$

其中 $\mathcal{S}$ 是世界状态，$\mathcal{A}$ 是人/智能体执行的动作集合，$\mathcal{O}$ 是视觉观测集合，转移函数 $\mathcal{T}$ 描述内部演化 $p(s_{t+1}\mid s_t, a_t)$，$\mathcal{R}$ 是从生成视频中提取的奖励。**用大白话说**：世界模型就是"给定我现在看到的画面和我要做的动作，预测下一帧世界会变成什么样"的一个可反复推演的仿真器。

交互式视频世界模型强调**闭环递归生成**，下一观测服从：

$$\mathbf{o}_{t+1} \sim p_\phi\left(\mathbf{o}_{t+1} \mid \mathcal{H}_t, \mathbf{a}_t, \mathbf{c}_t\right)$$

历史被定义为：

$$\mathcal{H}_t = \{\mathbf{o}_{1:t},\ \mathbf{a}_{1:t-1},\ \mathbf{c}_{1:t-1}\}$$

$p_\phi$ 是参数为 $\phi$ 的交互式生成世界模型，$\mathcal{H}_t$ 是交互历史，$\mathbf{a}_t$ 是当前用户/智能体动作，$\mathbf{c}_t$ 是额外条件（文本、图像或编辑控制）。**用大白话说**：与"一句话生成整段视频"不同，交互式世界模型每生成一帧都要重新读入"过去看到/做过的一切 + 你此刻的新动作"，因此它天然是流式、可打断、可续接的。

### 2.2 三条演进主线（Section 3）

作者用三个"从…到…"刻画趋势：
1. **应用场景：从专才到通才**——早期方法只服务单一领域（如游戏引擎），近期借助 Wan 2.2 等大规模视频先验做跨域泛化（UniSim 混合仿真/真机/人类/全景数据，AdaWorld 从无标注视频自监督抽取潜动作，Astra 用动作专家混合路由异构模态）。
2. **世界状态：从静态单智能体到动态自演化、多智能体**——早期只生成静态可探索环境（WonderWorld）；WorldCanvas/MotionStream 引入 drag-and-drop 对象级动态；LiveWorld 建立全局自演化状态，让视野外（out-of-sight）区域也持续演化；VerseCrafter 用 per-object 4D 高斯建模多智能体；Solaris/MultiWorld/Combo 支持多玩家、以逐智能体自我中心观测做多体协作。
3. **交互模态：从单感官到多感官**——除视觉外融合文本、音频（SonoWorld、Pixelverse-R1 音频指令）、物理（force/gravity 作为条件）、自动驾驶专属信号（ego 速度、道路语义、HD 地图、多相机同步）。

### 2.3 瓶颈一：用户动作可控性（Section 4）

- **One-shot 控制**（4.1）：ControlNet 式一次性文本/视觉指令；或"相机可控 3D 重建"路线（ViewCrafter、MotionCtrl、Uni3C）——但相机轨迹一旦设定即结束交互，泛化受限于视图重建而非世界状态演化。
- **多轮细粒度控制**（4.2）：采用**逐帧动作条件化**。Dreamer/MuZero 用 RNN 提供步级交互但可扩展性差；近期改用自回归（AR）把动作编码成因果 token 序列，兼得可扩展性与步级交互（iVideoGPT 自回归 transformer、UniSim 自回归扩散）；WorldCanvas/NeoVerse/VerseCrafter 进一步做到 region-/object-level 控制。
- **多样的动作注入方式**（4.3）——这是本综述最有价值的分类之一。相机位姿/轨迹注入归纳为四类：

| 注入方式 | 机制 | 代表工作 |
|---|---|---|
| Concatenation | 动作升维后与视觉 token 拼接 | Genie、iVideoGPT、AdaWorld |
| Scaling & Shifting | 动作生成 scale/shift 调制潜特征 | Lingbot-World、GameGen-X |
| Camera-controlled Rendering | 重述为相机可控渲染/仿真 | Vmem、LiveWorld、SWM |
| Matrix Transformation | 球面旋转矩阵（全景） | IaaW、GenEx |

文本指令注入则普遍用 T5 编码后经 cross-attention 注入视频 token；连续鼠标运动做 concatenation、离散键盘输入做 cross-attention（Matrix-Game 系列）。**用大白话说**：动作要"喂"进扩散/自回归网络有很多姿势——拼接、缩放平移、当成相机去渲染、当成旋转矩阵——不同姿势决定了控制的粒度和物理一致性。

### 2.4 瓶颈二：长时程交互与记忆（Section 5）

核心难题是自回归范式的**误差累积（compounding errors）与长时程漂移**。作者按时间线梳理四条解法：
- **历史帧作条件**（5.1）：最新/多帧历史帧拼入扩散去噪，去噪目标从 Genie 的 $\epsilon_\theta(o_t\mid o_{t-1}, a_{t-1})$ 扩展到多帧 $\epsilon_\theta(o_l\mid o_{\le t-1}, a_{\le t-1})$；UniSim 用重叠 chunk 拼接最后四帧。缺点：固定窗口限制时间跨度。
- **记忆构建**（5.2）：
  - *视频潜 token 记忆*：WorldMem token 级记忆库 + 状态感知 cross-attention；VRAG 检索历史三元组（位置感知状态+动作+帧）；RELIC 用 KV cache 滑窗 + 压缩长时程空间记忆缓解线性显存增长；HY-WorldPlay 混合短期时序 + 非邻长期空间记忆并做 Temporal Reframing 修正 RoPE 外推；WorldCam 位姿锚定记忆。
  - *显式 3D 几何记忆*：Vmem Surfel-indexed 视图记忆按重叠最大原则检索；DeepVerse 显式 4D 表征；Spmem 分静态点图 + episodic 稀疏历史帧；MosaicMem 混合显式/隐式。
- **显式 3D 重建**（5.3）：Wonder 系列（WonderJourney LLM 引导、WonderWorld FLAGS 提速、WorldGen 模块化 3D + 导航 mesh、VDAWorld 用 VLM + Critic Prompt 纠错）从图/文参考"从任意处到任意处"生成可探索 3D 场景。
- **噪声增强与 Forcing 训练**（5.4）：Oasis 动态加噪；Diffusion Forcing 对历史帧加变量高斯噪声；Self-Forcing（Matrix-Game 2.0）用自生成 rollout 而非 GT 减小曝光偏差；Geometry Forcing 对齐几何基座 3D 特征；Context Forcing（HY-WorldPlay）用记忆增强自 rollout 缓解"长上下文记忆学生 vs 短上下文无记忆教师"的失配；LIVE 用 cycle-consistency。**用大白话说**：模型训练时总看"完美历史"、推理时却只能看自己生成的"带瑕疵历史"，这个 train-inference 鸿沟正是漂移之源，各种 Forcing 就是让训练时也吃自己的粗糙输出、提前适应。

### 2.5 瓶颈三：实时动作响应性（Section 6）

- **一致性 vs 动作跟随的冲突**（6.1）：历史帧越多长时一致性越好，但会削弱对新动作的响应。Astra 向条件帧注入随机噪声模糊其影响、逼模型即时听从动作；HY-WorldPlay 用 Context Forcing 同时兼顾长时一致与实时响应。
- **高效 rollout**（6.2）：*模型蒸馏*（GameNGen 蒸馏到 50 FPS、HY-GameCraft 蒸成 8 步、MotionStream 自 forcing 分布匹配、Matrix-Game 3.0 多段蒸馏至 40 FPS、Fast-WAM 质疑显式未来想象是否必要）；*缓存加速*（Yume 层级缓存复用残差、HY-World 1.0 缓存 + 多 GPU 并行）；以及并行解码、关键帧重建、少步采样量化、模型剪枝等。

## 三、实验结果（基准与代表方法横向对比）

综述按四类场景整理数据集（Table 2）与代表方法数字。**注意：以下数字均转引自原文表格，反映的是各原论文自报结果。**

**（1）开放世界探索（Table 3，↑ 越高越好；CC=相机可控性，3DC=3D 一致性，SQ=主观质量，CLS/QA/CLA=CLIP/Q-Align/CLIP-aesthetic）**

| 方法 | CC | 3DC | SQ | CLS | QA | CLA |
|---|---|---|---|---|---|---|
| Text2Room | **94.01** | 88.71 | 36.69 | 34.58 | 2.359 | 4.912 |
| LucidDreamer | 88.93 | **90.37** | 58.99 | 31.35 | 2.439 | 5.576 |
| WonderJourney | 84.60 | 80.60 | **66.56** | 28.13 | 3.121 | 5.682 |
| WonderWorld | 92.98 | 86.87 | 49.81 | 32.28 | 3.437 | 6.123 |
| WonderFree | – | – | – | **35.00** | **3.912** | **6.493** |

趋势：早期方法（SceneScape/Text2Room/LucidDreamer）交互与主观质量低；WonderJourney/WonderWorld 显式引入交互与多视图一致性；WonderFree 在 CLIP 类、Q-Align、美学分上最优。

**（2）具身机器人操作（Table 4，WorldArena；AF=动作跟随，DA=深度精度，IF=指令跟随，IQ=图像质量）**

| 方法 | AF | AQ | BC | DA | IQ | IF |
|---|---|---|---|---|---|---|
| GigaWorld-1 | 0.28 | 41.17 | 86.43 | **98.44** | 51.18 | 82.14 |
| Wan2.6 | **9.92** | 44.40 | 84.29 | 33.63 | 47.92 | 89.96 |
| Veo3.1 | 8.82 | 48.79 | 91.67 | 74.27 | 65.57 | **97.14** |
| Ctrl-World | 3.90 | 37.05 | 90.30 | 93.25 | 42.44 | 67.68 |
| Genie Envisioner | 1.07 | 26.39 | 87.54 | 86.83 | 19.91 | 20.36 |

趋势：通用大视频模型（Wan2.6 动作跟随最强、Veo3.1 指令跟随最强）在部分感知/指令指标上领先，但 Genie Envisioner 等专用具身世界模型在综合动作/指令跟随上仍落后于强视频基线——说明"具身专用"与"通用大模型"各有短板。

**（3）机器人策略学习（Table 5，成功率↑；RoboTwin 2.0 与 Libero）**

| 方法 | RoboTwin 2.0 Avg | Libero Avg |
|---|---|---|
| π0 | 62.2 | 94.2 |
| π0.5 | 79.8 | 96.9 |
| Motus | 87.8 | – |
| LingBot-VA | **92.2** | **98.5** |
| Fast-WAM | 92.6 | 97.6 |

趋势：把交互式世界模型直接当策略（world-action model），如 LingBot-VA、Fast-WAM，在 RoboTwin 2.0 上超过强 VLA 基线 π0.5（79.8）；Fast-WAM 甚至质疑"是否需要显式未来想象"仍取得 92.6 均值。

**（4）游戏引擎与自动驾驶**：Table 6 显示实时性演进——Oasis（500M，20 FPS，640×360，无限时长）、Matrix-Game 2.0（5B，25 FPS，480×832，带记忆）、Yan（1080p、60 FPS）。Table 7（DrivingGen）中 Vista 取得较低 FVD 392.8 / FTD 27.13，GEM、DrivingDojo 等以视频真实感、时序一致性、轨迹对齐为评测重心。此外 Fig. 1 统计显示论文数量从 2024 年前的约 5 篇激增到 2026 年前 5 个月的约 66 篇 arXiv。

## 四、局限性

1. **作为综述本身**：主要面向 IWM 这一细分方向，对底层扩散/自回归生成理论、可扩展性定律、以及 3D/4D 表征本身（3DGS、mesh 生成）着墨相对有限；表格数字取自各原论文自报、评测协议不完全统一，横向可比性需谨慎。
2. **它所综述的领域仍存在的开放问题**（作者在 Section 8 明确列出五条）：
   - **大规模动作标注数据稀缺**：成对"观测-动作"数据昂贵、领域特定、难扩展；从无标注视频学到的动作常与场景外观/相机运动/数据集偏置纠缠，弱接地。
   - **反事实推理与真实对齐不足**：当前模型多依赖统计相关而非显式因果机制，可能生成"看似合理却因果错误"的未来，在安全攸关域尤其危险。
   - **极长时程一致性**：分钟级已可，扩到小时级仍难；长时会出现时序漂移、结构不一致、物体消失。
   - **物理感知缺失**：多数为数据驱动，未显式建模牛顿定律、刚体/柔体差异、力条件视频。
   - **可访问的直接操作界面**：需依 Shneiderman 的 direct manipulation 原则，把用户可理解的视觉概念映射到世界模型内部表征。

## 五、评价与展望（纯学术视角）

**优点**：（1）三大瓶颈（可控性、长时程记忆、实时响应）的划分清晰且互相正交，作为领域的"技术骨架"很有组织力，尤其把散落的 Forcing 家族（Teacher/Diffusion/Self/Geometry/Context Forcing + LIVE）统一到"缓解 train-inference 失配"这一主线下，是同类综述少见的深度综合。（2）动作注入四分类（拼接/缩放平移/相机渲染/矩阵变换）与记忆三分类（隐式 token / 显式 3D / 混合）提供了可操作的设计词表。（3）罕见地覆盖了开放世界、游戏、驾驶、具身四域并给出统一表格，对刚入门者极友好；配套 Awesome 仓库便于追踪。

**不足与开放问题**：（1）综述停留在"方法归类 + 自报数字"，缺少作者自行统一复现的公平对照，四域评测指标各异（如具身用 AF/IF、驾驶用 FVD/FTD、开放世界用 CLIP/Q-Align），跨域"孰优孰劣"难以定论。（2）对生成主干本身（DiT vs 自回归 vs latent action model）的可扩展性权衡讨论偏定性。（3）"物理感知"与"因果反事实"两条未来方向虽点明，却主要指向外接物理仿真器或 VLM 纠错等工程手段，尚缺理论层面的解法。

**与其他公开工作的关系**：本文可视为对 Genie、UniSim、iVideoGPT、Diffusion/Self-Forcing、Wonder 系列、Matrix-Game 系列等一批 2024–2026 代表作的整合式梳理。相较于偏重"视频生成质量"的综述，本文更强调**闭环交互性**这一区分维度；相较于纯具身 VLA 综述，它把"世界模型即策略/数据引擎"（GigaWorld-0、LingBot-VA、Fast-WAM）纳入统一框架，桥接了生成模型社区与机器人策略学习社区。

**可能的改进方向**：（1）建立跨四域统一的交互性评测协议（现有 WorldScore、OmniWorldBench、MIND、iWorldBench 已在尝试，但尚未收敛）；（2）把 latent action 表征与运动学约束、物体中心先验、几何一致性结合，做既可扩展又语义可解释的动作码；（3）分层记忆（短期运动细节 + 长期场景依赖）+ 显式几何索引，攻小时级一致性。

## 参考

1. Bruce et al., *Genie: Generative Interactive Environments*（Latent Action Model，动作 token 拼接的里程碑）
2. Yang et al., *Learning Interactive Real-World Simulators (UniSim)*（重叠 chunk、混合多源数据的通用仿真器）
3. Wu et al., *iVideoGPT: Interactive VideoGPTs are Scalable World Models*（自回归 transformer + slot token 帧界）
4. Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*（缓解曝光偏差的训练范式）
5. *Matrix-Game 2.0 / Self-Forcing*（自生成 rollout 减小误差累积，实时长时程游戏世界）
