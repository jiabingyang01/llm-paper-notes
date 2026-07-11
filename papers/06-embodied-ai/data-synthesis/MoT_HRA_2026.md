# MoT-HRA：从大规模人类演示中学习人类意图先验以服务机器人操作

> **论文**：*Learning Human-Intention Priors from Large-Scale Human Demonstrations for Robotic Manipulation*
>
> **作者**：Yifan Xie, YuAn Wang, Guangyu Chen, Jinkun Liu, Yu Sun, Wenbo Ding（Yifan Xie 与 YuAn Wang 共同一作,Yu Sun 为 project lead,Wenbo Ding 为通讯作者）
>
> **机构**：Tsinghua University；ByteDance
>
> **发布时间**：2026 年 04 月（arXiv 2604.24681,v2 于 2026 年 5 月 21 日）
>
> **发表状态**：未录用（预印本,PDF 首页标注 "Preprint"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.24681) | [PDF](https://arxiv.org/pdf/2604.24681)
>
> **分类标签**：`人类视频预训练` `VLA` `MANO 手部意图` `Mixture-of-Transformers` `知识隔离`

---

## 一句话总结

先用一条 hand-centric 过滤 + 3D 重建 + 时序切分 + 语言对齐流水线,把海量无标注人类视频重建成 **HA-2.2M**（220 万条动作-语言片段）数据集;再用 **MoT-HRA**（Mixture-of-Transformers 分层 VLA）把操作分解为「embodiment 无关 3D 轨迹 → MANO 手部意图 → 机器人动作」三个专家,并用**只读 KV 的知识隔离** 防止机器人动作损失反噬人类先验,最终在 SimplerEnv-WidowX 上以 **66.1%** 平均成功率显著超过 SpatialVLA（42.7%）与 ThinkACT（43.8%）。

## 一、问题与动机

VLA 的规模化受限于机器人示教数据昂贵、绑定硬件、只覆盖日常操作的窄切片。人类视频记录了跨家庭、厨房、工坊、视角、任务风格的海量物体交互,是比纯机器人数据更宽广的操作先验来源。但原始人类视频并非机器人示教:它把**场景理解、手部运动、相机运动、任务进度、具身相关约束** 纠缠在一起,很多片段只有可见的手却无有目的的操作,且极少提供时序对齐的动作标签或机器人可执行控制。

近期 human-centric 方法尝试抽取手部运动、把视频与动作标签对齐、或在视频与动作间引入 latent bridge。作者的核心观察是:**人类演示最有价值的用法,不是把它当作机器人动作的"含噪替代品",而是当作关于操作意图的结构化证据**——手的轨迹提供 embodiment 无关的空间脚手架,而关节化的手部运动捕捉时序协调与面向接触的运动偏好;机器人控制应被学成对这一先验的具身特定实现。据此,人到机器人的迁移应是分层的:先在 3D 空间中 ground 交互,再在结构化手部运动空间中建模人类意图,最后把意图感知表征适配到机器人动作块。这样既保留人类行为中可复用的部分,又把最终策略留给目标机器人自己的运动学与动作约定。

## 二、核心方法

方法分两块:数据侧 HA-2.2M 的重建流水线,与模型侧 MoT-HRA 的分层三专家架构。

### 2.1 HA-2.2M 数据流水线（Fig. 1）

数据源混合了 egocentric 与第三人称视频（Ego4D、EPIC-KITCHENS、Something-Something-V2、Ego-Exo4D、HowTo100M 等）。流水线三级级联:

- **Coarse Filter（粗过滤,面向高召回）**：先用 Gemini 标注可能含手部动作的片段,再用轻量 V-JEPA 分类器在这些标签上训练并铺到全量数据。两阶段分别去除明显无关内容、抑制多模态标注带来的"可见手但无操作"假阳性,避免对每个片段都昂贵地调用多模态大模型。
- **Perspective & Background Reconstruction（透视与背景重建）**：逐帧用 ViTPose 定位手、细化框以抗模糊遮挡,用 HaMeR 估计绝对尺度的 MANO 手部姿态;把 crop 空间的预测映射回原图并做时序平滑;并行用 Depth Anything 3 预测相对单目深度作背景布局,通过在手部像素上拟合把深度对齐到手的绝对尺度,从而把场景几何放进与手一致的 3D 坐标系。
- **Fine Filter（细过滤 + 语言对齐）**：用在 38K 人工标注片段上训练的 V-JEPA 时序切分模型预测动作边界,使每个 clip 对应一个操作原语（而非任意固定窗口);由于高召回边界预测会过切复杂活动,再用 Gemini 合并意图连续的相邻 clip 并生成简洁动作描述,含左右手分角色时的 hand-specific 标签（如 left_hand / right_hand)。

最终得到 220 万条时序一致的动作-语言片段,作为学习潜在人类动作先验的预训练单元。

### 2.2 MoT-HRA 架构（Fig. 2）

**问题分解。** 不把观察直接映射到动作,而是把 chunk 策略在给定图像 $o$、语言指令 $w$、chunk horizon $H$ 下分解为三层:embodiment 无关的空间计划 $\tau_{1:H}$、潜在人类意图 $h^{\text{int}}_{1:H}$、机器人动作 $a_{1:H}$:

$$p_\theta(a_{1:H}\mid o,w) \approx \int p_{\theta_a}(a_{1:H}\mid o,w,\tau_{1:H},h^{\text{int}}_{1:H})\, q_{\theta_m}(h^{\text{int}}_{1:H}\mid o,w,\tau_{1:H})\, p_{\theta_\tau}(\tau_{1:H}\mid o,w)\, d\tau\, dh.$$

用大白话说:先猜"交互该发生在空间哪里"（3D 轨迹),再在此基础上猜"人的手会怎么动"（意图),最后把这个意图桥接到"这台机器人该怎么执行"（动作)。MANO 手部姿态序列 $m_{1:H}$ 只用来监督意图分支,**不当作机器人指令**;动作流最终条件在潜在意图 $h^{\text{int}}_{1:H}$ 而非解码出的手部姿态上。

**统一架构。** MoT-HRA 是一个 Mixture-of-Transformers:一条 cross-task 共享注意力 trunk + 三个任务专家。完整 token 序列
$$X = \left[x^{\text{img}}_{1:N_v};\ x^{\text{txt}}_{1:N_t};\ z^{3d}_{1:H};\ z^{m}_{1:H};\ z^{a}_{1:H}\right].$$
共享 trunk 构建多模态上下文,后续专家只更新自己 span 的 hidden,并通过**只读 KV 缓存** 读取更早的隐藏状态——这就是 **knowledge insulation（知识隔离)**:上游表征对下游可见,但不会被下游损失改写。具体地,视觉-语言/轨迹专家不受 MANO 与动作去噪损失污染,而机器人动作专家可以利用意图线索却不会把它们坍缩成单一纠缠 latent。

**Vision-Language 专家(3D 轨迹)。** 由预训练多模态 decoder（PaliGemma 2 类)初始化,把视觉-语言上下文转成粗空间 waypoint。为长度 $H$ 的 chunk 追加 $H$ 个可学 3D query token,每个自回归预测离散化 waypoint $\tau_h=(b_h^x,b_h^y,b_h^z)$,每坐标量化到 $B$ 个 bin,坐标级交叉熵训练:
$$\mathcal{L}_{3d} = -\sum_{h=1}^{H}\sum_{c\in\{x,y,z\}}\log p_{\theta_\tau}(b_h^c \mid o,w,\tau_{<h}).$$
用大白话说:强迫模型先把"该在哪儿动手"定位成一串离散 3D 锚点,做几何 grounding 但不绑定任何具身,给下游专家一个紧凑的空间脚手架。

**Intention 专家(MANO 手部意图)。** 条件在共享上下文与预测 3D 轨迹上,用 conditional flow matching 把高斯噪声去噪成 MANO 风格手部序列。每个手部状态
$$m_h = [w_h, j_h],\qquad w_h \in \mathbb{R}^{7},\quad j_h \in \mathbb{R}^{60},$$
其中 $w_h$ 存手腕平移与四元数朝向,$j_h$ 堆叠 15 个手指关节的四元数。四元数目标做符号规范化,解码后重归一化。设 $\epsilon^m\sim\mathcal{N}(0,I)$、$t\sim\mathcal{U}(0,1)$、$x_t^m=(1-t)\epsilon^m+tm$,训练目标把**手腕损失与手指损失分离**（分别按 $\tfrac{1}{7H}$ 与 $\tfrac{1}{60H}$ 归一)以免 7 维手腕信号被 60 维关节项淹没:
$$\mathcal{L}_{\text{mano}} = \mathbb{E}\left[\frac{1}{7H}\sum_{h=1}^{H}\left\|v_h^w - (w_h - \epsilon_h^w)\right\|_2^2 + \frac{1}{60H}\sum_{h=1}^{H}\left\|v_h^j - (j_h - \epsilon_h^j)\right\|_2^2\right].$$
用大白话说:用显式手部运动学（而非无约束 latent code)把海量人类视频变成一个可解释、与物理交互对齐的意图空间。MANO token 在自身 span 内用因果注意力,防止有效早期状态去 attend 未来/padding。

**Fine 专家(机器人动作)。** 再用一个 conditional flow matching head 把意图感知表征映射到机器人动作块 $a_{1:H}\in\mathbb{R}^{H\times d_a}$,条件变量 $c^a$ 含图文特征、3D 轨迹状态、以及**潜在意图状态 $h^{\text{int}}_{1:H}$**——即策略是在给 MANO 表征去噪时产生的潜在状态上条件,而非把解码出的手部姿态当模仿目标:
$$\mathcal{L}_{\text{act}} = \mathbb{E}\left[\left\|v_{\theta_a}(x_t^a,t,c^a) - (a - \epsilon^a)\right\|_2^2\right].$$
动作 token 对所有前置模态做注意力,并在动作块内用双向注意力做联合细化(Fig. 3 注意力 mask:图文双向、3D 与 MANO 在各自 span 内因果、机器人动作对全前缀 attend + 块内双向)。

**联合训练(部分监督多任务)。**
$$\mathcal{L} = \lambda_{3d}\mathcal{L}_{3d} + \lambda_m \mathbb{1}_m \mathcal{L}_{\text{mano}} + \lambda_a \mathbb{1}_a \mathcal{L}_{\text{act}},$$
$\mathbb{1}_m,\mathbb{1}_a$ 指示该样本是否有 MANO / 机器人动作监督。人类演示主要监督轨迹与意图专家,机器人操作数据监督轨迹与 fine 专家;在机器人样本上意图专家仍通过只读接口提供 latent 条件但**不接收 MANO 损失,下游动作梯度也不回传进意图缓存**。于是人类视频塑造意图流形,机器人数据特化具身执行,推理时无需 MANO 标注即可直接产生可执行动作。

## 三、实验结果

训练:仅用 HA-2.2M + AgiBot-World 真机数据集,64 张 NVIDIA Hopper GPU（PyTorch FSDP2),global batch 2048,20,000 步,AdamW,base lr $2.5\times10^{-5}$,cosine decay + 1000 步 warmup,chunk horizon $H=15$,MANO 生成用 CFG scale 6.0,输入 $224\times224$。

**手部运动生成（Table 1,ADE/DTW 单位米、Rot/Joint-Rot 单位度,越低越好；egocentric 用 held-out Ego4D,验证泛化而非记忆)。**

| Method | Ego ADE | Ego DTW | Ego Rot | Ego Joint-Rot | TP ADE | TP DTW | TP Rot | TP Joint-Rot |
|---|---|---|---|---|---|---|---|---|
| Being-H0 | 0.185 | 0.174 | 38.27 | 44.03 | 0.245 | 0.233 | 44.91 | 49.18 |
| VITRA | 0.154 | 0.146 | 33.26 | 41.81 | 0.211 | 0.201 | 42.59 | 41.72 |
| **Ours** | **0.136** | **0.127** | **28.95** | **34.16** | **0.184** | **0.176** | **38.47** | **40.12** |

在两个数据集上均最优,手腕对齐与关节姿态质量都提升;第三人称视角(更大外观 gap)上仍领先,说明该意图先验不局限于训练视觉风格。指出细粒度手指旋转(Joint-Rot)是强域迁移下最难的一项,提升幅度相对较小。

**SimplerEnv-WidowX 操作成功率(Table 2,%;考察分布漂移下的视觉 grounded 控制)。**

| Method | Spoon | Carrot | Stack | Eggplant | Average |
|---|---|---|---|---|---|
| RoboVLMs | 45.8 | 20.8 | 4.2 | 79.2 | 37.5 |
| OpenVLA-OFT | 34.2 | 30.0 | 30.0 | 72.5 | 41.7 |
| $\pi_0$ | 29.1 | 0.0 | 16.6 | 62.5 | 27.1 |
| $\pi_0$-FAST | 29.1 | 21.9 | 10.8 | 66.6 | 32.1 |
| SpatialVLA | 16.7 | 25.0 | 29.2 | **100.0** | 42.7 |
| ThinkACT | 58.3 | 37.5 | 8.7 | 70.8 | 43.8 |
| **Ours** | **78.1** | **62.5** | **40.6** | 83.3 | **66.1** |

平均 66.1% 大幅领先,增益主要来自 Spoon 与 Carrot（需精确空间 grounding 与稳定放置);SpatialVLA 仅在 Eggplant 单场景刷满 100.0 但整体过拟合,MoT-HRA 在该任务仍有竞争力(83.3)。

**真机长程操作(Fig. 5,20 次试验,对比 $\pi_0$ 与 GigaBrain-0;每具身用 150 条任务轨迹后训练,评测含物体位置/类别/颜色的 OOD 变化)。** 平行夹爪上,MoT-HRA 在 Clean/Pouring 约 80% / 65%,均高于 $\pi_0$（约 75% / 55%);灵巧手上 Clean 约 65%、Pouring 约 50%,同样在两具身上取得更可靠的任务完成率。

**消融(Table 3;第一行相当于 $\pi_0$ 式直接 VLA)。**

| 3D Traj | Intention Expert | Knowledge Insulation | ADE | DTW | Rot | Joint-Rot | SimplerEnv Avg |
|---|---|---|---|---|---|---|---|
| — | — | — | 0.205 | 0.196 | 40.16 | 44.90 | 48.4 |
| ✓ | — | — | 0.182 | 0.173 | 36.64 | 42.53 | 52.1 |
| ✓ | ✓ | — | 0.140 | 0.133 | 30.45 | 35.97 | 62.7 |
| ✓ | ✓ | ✓ | **0.136** | **0.127** | **28.95** | **34.16** | **66.1** |

加 3D 轨迹分支即提升手部指标与 SimplerEnv 均值(48.4→52.1);引入意图专家带来最大跃升(→62.7),说明显式人类意图建模对连贯手腕轨迹与合理手姿至关重要;知识隔离进一步把均值提到 66.1 并同时降低运动生成误差。从 $\pi_0$ 式 baseline 到完整模型的**单调** 提升支持了"增益来自分层结构而非单纯堆容量"的论断。

## 四、局限性

- 受自动化人类演示的质量与覆盖限制:含噪、手-物接触模糊、动作-语言对齐不完美都会削弱学到的意图先验。
- 评测覆盖代表性的手部运动与操作任务,但**未涉及高动态交互、多物体长程规划、以及差异极大的具身**。
- 未来方向:改进数据校验、扩展具身覆盖、加入 failure detection 以支持更可靠的开放世界部署。

## 五、评价与展望

**优点。** (1) 把"人类视频到机器人"这一迁移问题清晰地解耦为「空间 grounding → 人类意图 → 具身动作」三层,并给了每层各自合适的表征形式(离散 3D bin 做自回归、MANO 做 flow matching、动作块做 flow matching),分工干净、消融单调,论证链条比多数 human-video VLA 更可信。(2) 用 MANO 显式手部运动学作为意图空间,比 latent-action / latent-bridge 方法(如 IGOR、Moto、UniVLA)更可解释,也天然对齐物理接触。(3) 借用 π 系的 knowledge insulation（只读 KV)缓解大规模异构训练中"机器人损失反噬人类先验"的负迁移,这是本文相对 Being-H0、VITRA 等同期工作的关键差异化设计,且消融证明其独立贡献(62.7→66.1 且降低手部误差)。

**可讨论之处。** (1) SimplerEnv-WidowX 上 $\pi_0$ 仅 27.1% 偏低,而 $\pi_0$-style 的本文消融首行却有 48.4,配置差异(数据、微调)可能放大了对比优势,跨论文数字需谨慎横比。(2) 真机结果仅以柱状图给出且每具身仅 20 次试验、150 条后训练轨迹,统计功效有限,难以断言相对 GigaBrain-0 的稳定优势。(3) 数据流水线重度依赖 Gemini + HaMeR + Depth Anything 3 + ViTPose 的级联,任一环节的系统性偏差(尤其单目绝对尺度深度对齐)会静默污染 220 万条监督信号,而论文未量化重建误差对下游先验的传播。(4) MANO 只建模手,未建模被操作物体的位姿/接触,面向接触的"意图"仍是间接的。

**与公开工作的关系与开放问题。** 本文与 Being-H0、VITRA、EgoVLA、Vidbot 同属"从人类视频抽取 3D 手部/轨迹作先验"的显式 3D 路线,区别在于把手部运动隔离为**中间意图流形** 而非直接动作标签。一个自然的开放问题是:意图专家产出的潜在 $h^{\text{int}}$ 是否真的编码了可迁移的运动协调,还是主要充当了空间轨迹的精修?可通过 probing / 跨具身零样本迁移进一步验证。另一方向是把物体级 affordance / 接触先验与手部意图联合建模,以及把当前离散 3D waypoint 升级为连续 SE(3) 轨迹以支持更高动态的操作。

## 参考

1. H. Luo et al. *Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos*. arXiv:2507.15597, 2025.（本文手部生成与操作主对照基线)
2. Q. Li et al. *Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos (VITRA)*. arXiv:2510.21571, 2025.（另一主对照基线)
3. K. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.（flow-matching VLA 基座,本文 fine 专家与首行消融的原型)
4. D. Driess et al. *Knowledge Insulation for Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better*. arXiv:2505.23705, 2025.（本文只读 KV 知识隔离机制来源)
5. G. Pavlakos et al. *Reconstructing Hands in 3D with Transformers (HaMeR)*. CVPR, 2024.（数据流水线的绝对尺度 MANO 手部重建工具)
