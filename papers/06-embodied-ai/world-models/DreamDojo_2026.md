# DreamDojo：从大规模人类视频中预训练通用机器人世界模型

> **论文**：*DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos*
>
> **作者**：Shenyuan Gao†、William Liang†、Kaiyuan Zheng\*、Ayaan Malik\*、Seonghyeon Ye、Sihyun Yu、Wei-Cheng Tseng、Yuzhu Dong、Kaichun Mo、Chen-Hsuan Lin、Qianli Ma、Seungjun Nah、Loic Magne、Jiannan Xiang、Yuqi Xie、Ruijie Zheng、Dantong Niu、You Liang Tan、K.R. Zentner、George Kurian、Suneel Indupuru、Pooya Jannaty、Jinwei Gu、Jun Zhang、Jitendra Malik、Pieter Abbeel、Ming-Yu Liu、Yuke Zhu‡、Joel Jang‡、Linxi “Jim” Fan‡（† 共同一作，\* 核心贡献者，‡ 项目负责人）
>
> **机构**：NVIDIA、香港科技大学（HKUST）、加州大学伯克利分校（UC Berkeley）、华盛顿大学（UW）、斯坦福大学、KAIST、多伦多大学（UofT）、加州大学圣地亚哥分校（UCSD）、得克萨斯大学奥斯汀分校（UT Austin）
>
> **发布时间**：2026 年 2 月（arXiv 2602.06949）
>
> **发表状态**：ICML 2026（已录用）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.06949) | [PDF](https://arxiv.org/pdf/2602.06949) | [项目主页](https://dreamdojo-world.github.io/) | [代码](https://github.com/NVIDIA/DreamDojo) | [HuggingFace](https://huggingface.co/nvidia/DreamDojo)
>
> **分类标签**：`视频世界模型` `第一视角人类视频` `连续潜动作 (Latent Action)` `Self Forcing 自回归蒸馏` `实时交互仿真` `策略评估` `测试时模型规划`

---

## 一句话总结

针对机器人动作条件视频世界模型长期受困于“真机遥操作数据场景狭窄、专家轨迹缺乏意图随机性导致反事实动作失控、以及双向扩散多步去噪无法实时自回归交互”的三大核心瓶颈，NVIDIA GEAR 联合多校团队提出首个面向开放世界灵巧操作的通用基础世界模型 **DreamDojo**（提供 **2B** 与 **14B** 两档规模）：首先构建迄今最大规模的世界模型预训练第一视角人类视频混合语料（总计 **44,711 小时 / 117.9 万条轨迹**，其中自建众包 **DreamDojo-HV** 达 43,827 小时，时长、技能数与场景数分别达此前最大数据集的 **$15\times$、$96\times$ 与 $2,000\times$**）；其次训练 700M 时空 Transformer VAE 提取 32 维**连续潜动作**（Continuous Latent Actions）作为跨具身统一代理动作标签，配合**潜帧重设基准相对动作**、**4 帧对齐分块因果注入**与**时序速度差一致性损失**完成大规模物理与控制预训练，仅需少量目标机器人数据重置动作首层微调即可精确迁移至 GR-1、Unitree G1、AgiBot 等多构型；最后引入两阶段 **Self Forcing 长视域分布匹配自回归蒸馏**，将双向 35 步扩散教师压缩为 4 步因果滑动窗口学生模型，在单卡上实现 **$640 \times 480$ @ 10.81 FPS**（提速近 $4\times$）且连续交互超 **1 分钟（600 帧）** 无退化，成功打通**高保真离线策略评估（Pearson $r = 0.995$）**、**测试时模型规划（成功率较最优单检查点 +17 pp、较均匀采样翻倍）** 与 **RTX 5090 本地实时 VR 虚拟遥操作**三大闭环应用。

---

## 一、问题与动机

### 1.1 机器人动作条件世界模型的三重现实壁垒

动作条件交互式世界模型（Interactive World Model）旨在拟合状态转移分布 $s_{t+1} \sim p(\cdot \mid s_t, a_t)$，一旦在开放世界中具备可靠的物理与动作因果预测能力，便可直接充当无限重置的策略评估器、强化学习沙盒与在线规划大脑。然而，相较于离散按键空间的游戏世界模型（如 Genie 系列）或场景结构单一的自动驾驶世界模型（如 GAIA、Vista），高维富接触机器人操作面临三重独特壁垒：

1. **真机数据覆盖度极窄，分布外（OOD）物理崩溃**：现有机器人世界模型几乎完全依赖实验室遥操作数据集（如 BridgeData V2 仅 130 小时 / 24 场景，DROID 仅 350 小时 / 564 场景，AgiBot-World 仅 2.9k 小时 / 106 场景）。狭窄的动词、物体与背景分布导致模型一旦遇到训练未见物体或环境切换便出现严重的刚体穿模、变形消失等物理失真。
2. **专家演示缺乏行为随机性，反事实动作（Counterfactual Actions）失灵**：现有机器人数据集绝大部分由成功的专家轨迹组成，动作与场景高度共线。在缺乏失败探索与多样意图覆盖的情况下，视频扩散模型极易退化为“仅凭首帧视觉线索猜专家套路”而无视输入的真实控制指令——当输入“伸手拍打玩具”或“故意抓偏悬停”等反事实动作时，模型依然固执地渲染出抓取成功的幻觉画面。
3. **无动作人类视频的因果缺失与双向扩散的推理高延迟**：海量第一视角人类日常视频虽然天然具备跨具身一致的物理交互规律与极高的行为随机性，但既无精细关节动作标签，用 HaMeR 等单目手部姿态估计器后处理又难以覆盖手臂/全身移动（Loco-manipulation）且在重度遮挡与头动下严重失准，低层人手骨骼特征更与异构机械臂存在形态鸿沟；此外，主流视频扩散模型依赖固定窗口双向注意力与 35–50 步迭代去噪，帧率不足 3 FPS 且无法维持跨窗口历史记忆，难以支撑实时闭环交互。

---

## 二、预备知识：Cosmos-Predict2.5 视频潜空间流匹配基座

DreamDojo 建立在预训练视频基础模型 **Cosmos-Predict2.5**（Ali et al., 2025）之上：

- **WAN2.2 时空压缩潜空间**：原始像素视频帧经冻结的 **WAN2.2 视频 Tokenizer**（Wan et al., 2025）编码为连续潜空间张量 $\mathbf{x}$，其**时间维压缩比为 4**——即除首帧条件外，每个视频潜帧 $x_i$ 严格对应像素空间中连续 4 帧 $f_{i:i+4}$，$x_{i+1}$ 对应 $f_{i+4:i+8}$，依此类推。
- **DiT 条件调制与标准流匹配目标**：文本嵌入通过交叉注意力注入每个 DiT Block（DreamDojo 预训练期统一固定为空提示词以专注动作-物理因果），连续时间步 $t$ 经正余弦位置编码与轻量 MLP 投影后驱动自适应层归一化（**AdaLN**）的动态缩放、平移与门控调制（scale, shift, gate）。设干净视频潜张量为 $\mathbf{x}$、高斯噪声为 $\epsilon \sim \mathcal{N}(0, I)$、加噪潜变量为 $\mathbf{x}_t$、条件集合为 $\mathbf{c}$，去噪网络 $u(\cdot; \theta)$ 回归目标速度场 $\mathbf{v}_t = \epsilon - \mathbf{x}$：

$$
\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{\mathbf{x}, \epsilon, \mathbf{c}, t}\Bigl\lVert u(\mathbf{x}_t, t, \mathbf{c}; \theta) - \mathbf{v}_t \Bigr\rVert^2
$$

---

## 三、核心方法

DreamDojo 的完整构建流程由**三源人类视频语料构建**与“**人类视频潜动作预训练 $\to$ 目标机器人动作空间后训练 $\to$ Self Forcing 自回归实时蒸馏**”三阶段管线组成。

### 3.1 44,711 小时三源第一视角人类视频混合语料（DreamDojo-HV Mixture）

为同时兼顾受控动作验证、精细五指桌面交互与开放世界全域场景覆盖，论文整合了三个互补的第一视角人类数据源（Table 1 & Figure 2）：

| 数据集名称 | 数据类型 | 代表使用工作 | 总时长（Hours） | 轨迹数（Traj.） | 技能数（Skills） | 场景数（Scenes） | 核心定位与采集形态 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **BridgeData V2** | 机器人 | IRASim, UniSim, WorldGym | 130 | 60.1k | 13 | 24 | 传统桌面单臂遥操作基准 |
| **DROID** | 机器人 | Ctrl-World, UWM | 350 | 76k | 86 | 564 | 跨实验室多场景单臂遥操作 |
| **AgiBot-World** | 机器人 | EnerVerse-AC | 2.9k | 1,000k | 87 | 106 | 此前最大规模双臂机器人世界模型语料 |
| **Nymeria** | 人类 | PEVA, EgoTwin, EgoControl | 300 | 1.2k | — | 50 | 全身日常移动动捕视频 |
| **In-lab（本文子集 1）** | 人类 | DreamDojo | **55** | 13.9k | 35 | 1 | Manus 数据手套 + Vive 追踪器，含 GR-1 未见物体/动词，可精确重定向 |
| **[EgoDex](../data-synthesis/EgoDex_2025.md)（本文子集 2）** | 人类 | DreamDojo | **829** | 30k | 194 | 5 | Apple Vision Pro 高精度 3D 手指骨骼追踪，极大丰富日常家用物体交互 |
| **DreamDojo-HV（本文主力）** | 人类 | DreamDojo | **43,827** | 1,135k | $6,015^\dagger$ | 1,135k | **大规模众包全域日常移动与操作视频**，覆盖 17 大行业与生活门类 |
| **Our Mixture（总混合集）** | **人类** | **DreamDojo** | **44,711** | **1,179k** | **$\ge 6,015^\dagger$** | **$\ge 1,135\text{k}$（$>9,869$ 独立物理场所）** | **时长 $15\times$、技能 $96\times$、场景 $2,000\times$，覆盖 $43,237$ 类唯一物体** |

其中主力众包数据集 **DreamDojo-HV（43,827 小时）** 横跨 17 大真实环境门类（Figure 2a）：家务操作（Home Task，18,483 h / **42.2%**）、零售与消费品（6,161 h / **14.1%**）、汽车与交通维修（3,895 h / **8.9%**）、餐饮备餐（3,693 h / **8.4%**）、服饰配饰（2,360 h / **5.4%**）、通用维修服务（1,956 h / **4.5%**）、行政办公（1,217 h / 2.8%）、生产工厂（1,192 h / 2.7%）、酒店服务（1,084 h / 2.5%）、印刷设计（1,026 h / 2.3%）以及教育、体育、食品加工、工业制造、创意工坊、健康美业与能源设施，绝大多数视频为包含多子任务转换的长时程工具使用与全身移动操作（Loco-manipulation）。预训练阶段按 **`In-lab : EgoDex : DreamDojo-HV = 1 : 2 : 10`** 的采样权重混合抽取。

### 3.2 阶段一：信息瓶颈连续潜动作模型（LAM）与世界模型预训练

#### （1）自监督时空 Transformer VAE 提取 32 维连续潜动作
为在完全无动作标签的 44.7k 小时视频中挖掘跨具身通用的“帧间因果控制信号”，论文构建了一个 **700M 参数的时空 Transformer 变分自编码器**（24 层编码器 + 24 层前向动力学解码器，输入缩放至 $320 \times 240$ 并随机按 `{1, 2, 3, 4}` 倍做时间下采样以覆盖快慢不同速率运动，在三个人类数据集与内部 Unitree G1 / Fourier GR-1 / AgiBot / YAM 机器人视频混合集上从头训练 400k 步）：
- **编码器 $q_\varphi(\hat{a} \mid f_{t:t+1})$** 接收相邻两帧 $f_t, f_{t+1}$，提取全局时空特征并压缩投影至低维连续高斯潜向量 $\hat{a}_t \in \mathbb{R}^{32}$；
- **解码器 $p_\theta(f_{t+1} \mid \hat{a}, f_t)$** 仅接收**前一帧 $f_t$ 与紧凑潜动作 $\hat{a}_t$** 重建下一帧 $f_{t+1}$，由后一帧重构损失与超弱 KL 正则（$\beta = 10^{-6}$，兼顾表征容量与下游微调可迁移性）联合施加**信息瓶颈**（Information Bottleneck）：

$$
\mathcal{L}_{\text{pred}}^{\theta, \varphi}(f_{t+1}) = \mathbb{E}_{q_\varphi(\hat{a} \mid f_{t:t+1})} \log p_\theta(f_{t+1} \mid \hat{a}, f_t) - \beta\, D_{\mathrm{KL}}\Bigl(q_\varphi(\hat{a} \mid f_{t:t+1}) \,\Big\|\, p(\hat{a})\Bigr)
$$

**用大白话说**：解码器已经拿到了完整的上一帧背景画面 $f_t$，而 $\hat{a}_t$ 只有区区 32 个浮点数且受 KL 瓶颈约束——模型根本塞不下冗余背景像素，为了准确还原下一帧 $f_{t+1}$，它被逼着只把“两帧之间手或机械臂往哪儿动了”提炼进 $\hat{a}_t$。跨数据集最近邻检索（Figure 3 右）直观证实：拥有相似连续潜动作向量 $\hat{a}_t$ 的人类帧对与不同构型机器人帧对，尽管背景与身体外观天差地别，正在执行的物理动作方向与交互语义却高度一致！

#### （2）精细动作可控性的两项核心架构改造（Sec. 3.3.1）
直接把完整绝对动作序列作为全局条件喂给视频 DiT 会引发严重的分布过拟合与“非因果偷看未来噪声”。DreamDojo 对动作注入通路实施了两项关键改造：
1. **逐潜帧重设基准的相对动作变换（Latent-Frame Rebaselined Relative Actions）**：不再输入机器人绝对关节位姿，而是以**每个视频潜帧起始时刻（即每 4 个时间步）的位姿为局部基准**计算相对动作增量。相对动作集中在跨轨迹高度共享的紧凑邻域内，大幅降低了连续复合动作的建模复杂度。
2. **匹配 Tokenizer 压缩比的分块因果对齐注入（Chunked Causal Action Injection）**：严格对齐 WAN2.2 的 $4\times$ 时间压缩结构，将连续 4 步动作 $a_{t:t+4}$（预训练期为 4 步连续潜动作 $\hat{a}_{t:t+4}$）拼接为一个动作块（Chunk），经轻量 MLP 投影至时间步嵌入同维后，**一对一专属注入对应的单个视频潜帧 $x_i$ 的 DiT AdaLN 调制项中**（与时间步嵌入相加），彻底杜绝未来动作对当前潜帧预测的非因果干扰。受 ControlNet 启发，**动作 MLP 的最后一层权重初始化为全零**，避免训练初期随机动作梯度破坏 Cosmos-Predict2.5 原有的预训练物理先验。

#### （3）时序速度差一致性损失（Temporal Consistency Loss）
标准流匹配损失 $\mathcal{L}_{\text{flow}}$ 对每个潜帧独立计算误差，忽略了帧与帧之间的动态跃迁关联。设长度为 $K$ 的视频潜序列预测速度场为 $\mathbf{z}_t = u(\mathbf{x}_t, t, \mathbf{c}; \theta) = [z_1, z_2, \dots, z_K]$、真值速度场为 $\mathbf{v}_t = [v_1, v_2, \dots, v_K]$，DreamDojo 显式监督相邻潜帧的一阶时间差分匹配：

$$
\mathcal{L}_{\text{temporal}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{K-1} \Bigl\lVert (z_{i+1} - z_i) - (v_{i+1} - v_i) \Bigr\rVert^2\right]
$$

$$
\mathcal{L}_{\text{final}}(\theta) = \mathcal{L}_{\text{flow}}(\theta) + \lambda\,\mathcal{L}_{\text{temporal}}(\theta), \qquad \lambda = 0.1
$$

该项直接强迫模型对齐帧间物体运动增量与动作响应斜率，显著加快动作跟随收敛并消除被操作物体的残影与撕裂伪影。预训练在 **256 × NVIDIA H100** 上以 $640 \times 480$ 分辨率、13 帧序列长度、全局 Batch Size 1024 训练 **140k 步**（AdamW，学习率 $1.6 \times 10^{-4}$，权重衰减 $0.1$，全程维护 EMA），同步产出 **DreamDojo-2B** 与 **DreamDojo-14B** 两款基座。

### 3.3 阶段二：目标机器人构型轻量后训练（Post-Training）

迁移至任意下游目标机器人（如 Fourier GR-1、Unitree G1、AgiBot、YAM）时，将目标机器人视频按约 **10 Hz** 采样切分为 13 帧片段（第 1 帧作为条件帧，后 12 步转换为潜帧重设基准的相对真实机器人动作）。**仅重置动作 MLP 的第一层权重**以适配新机器人的物理动作自由度维度，随后在 **128 × H100** 上以 Batch Size 512 对全模型参数端到端微调 **50k 步**。得益于第一阶段连续潜动作与 44.7k 小时物理交互预训练，即使目标真机数据仅采集自极有限的单一实验室桌面，微调后也能零样本泛化至全新场景与未见物体交互。

### 3.4 阶段三：Self Forcing 长视域自回归实时蒸馏（Distillation）

预训练/后训练教师模型 $G_{\text{teacher}}$ 采用双向注意力且需 **35 步去噪**（无需 CFG），单次只能以单帧为条件整块生成 12 帧（仅 2.72 FPS）且跨块重置条件帧会丢失遮挡前历史。DreamDojo 基于 **Self Forcing**（Huang et al., NeurIPS 2025）将其蒸馏为同参数结构但采用 **12 帧滑动窗口因果注意力（Causal Attention）+ 仅 4 步去噪**的自回归学生模型 $G_{\text{student}}$（在 **64 × H100** 上执行）：

1. **阶段 3A：ODE 轨迹回归预热（Warmup，10k 条教师 ODE 轨迹，10k 步，Batch Size 256）**：在 Teacher Forcing 上下文（历史潜帧均取自教师干净输出）下直接回归教师 ODE 解 $x_0$：

$$
\mathcal{L}_{\text{warmup}}(G_{\text{teacher}}, G_{\text{student}}) = \mathbb{E}_{x, t}\Bigl\lVert G_{\text{student}}(x_t, t) - x_0 \Bigr\rVert^2
$$

2. **阶段 3B：随机超长视域自滚出分布匹配蒸馏（Self-Forcing DMD，3k 步，Batch Size 64）**：彻底切换为**学生自生成历史潜帧作为自身滑动窗口上下文**，消除训练-推理分布偏移。为抑制超过教师原始窗口 $N=13$ 后的长时程累积漂移，创新性地令学生在训练期**随机自回归连续生成 $N' \in [13, 49]$ 帧（$N' > N$）**，并截取最后 $N=13$ 帧窗口通过真实得分网络 $s_{\text{real}}$（教师模型）与在线伪造得分网络 $s_{\text{fake}}$ 计算**分布匹配蒸馏**（DMD）KL 散度梯度反传：

$$
\mathcal{L}_{\text{distill}} = D_{\mathrm{KL}}\bigl(p_{\text{teacher}} \,\|\, p_{\text{student}}\bigr), \qquad \nabla \mathcal{L}_{\text{distill}} = -\mathbb{E}_{z, t}\left[\bigl(s_{\text{real}}(x_t, t) - s_{\text{fake}}(x_t, t)\bigr)\frac{\mathrm{d} G_{\text{student}}}{\mathrm{d}\theta}\right]
$$

下表汇总 DreamDojo 全流程四大训练模块的模型规模、分辨率、算力与优化配置（Sec. 4.1）：

| 训练阶段与模块 | 模型架构与参数量 | 输入分辨率 / 序列长度 | 算力配置与 Batch Size | 训练步数与学习率（AdamW） |
| :--- | :--- | :---: | :---: | :--- |
| **连续潜动作模型（LAM）** | 24+24 层时空 Transformer VAE（**700M**，$d_{\hat{a}}=32$） | $320 \times 240$ / 相邻帧对（$1\text{–}4\times$ 随机降采样） | 全局 BS $= 256$ | $400\text{k}$ 步，lr $= 2.5 \times 10^{-5}$，wd $= 0.01$，$\beta = 10^{-6}$ |
| **人类视频世界模型预训练** | Cosmos-Predict2.5 DiT（**2B / 14B**，WAN2.2 $4\times$ 压缩） | $640 \times 480$ / 13 帧（1:2:10 采样） | **256 × H100**，BS $= 1024$ | $140\text{k}$ 步，lr $= 1.6 \times 10^{-4}$，wd $= 0.1$，EMA + $\lambda=0.1$ |
| **目标机器人后训练** | 重置动作 MLP 首层 + 全参数微调（2B / 14B） | $640 \times 480$ / 13 帧（~10 Hz，1 帧 + 12 步相对动作） | **128 × H100**，BS $= 512$ | $50\text{k}$ 步（消融 $25\text{k–}30\text{k}$），35 步去噪（关闭 CFG） |
| **自回归实时蒸馏** | 12 帧滑窗因果注意力 + **4 步去噪**学生（2B） | $640 \times 480$ / Warmup 13 帧 $\to$ DMD $13\text{–}49$ 帧 | **64 × H100**，BS $= 256 \to 64$ | Warmup $10\text{k}$ 步（$10\text{k}$ ODE 轨迹）+ DMD $3\text{k}$ 步 |

---

## 四、实验结果

### 4.1 六大分布外（OOD）与反事实评测基准设计

以 **Fourier GR-1 人形机器人**为主力受控测试平台，专门构造 6 个在 GR-1 训练集上完全未见的严苛评测集（Figure 4）：
- **三大跨源物理镜像基准**：复刻三个人类数据集中 GR-1 未见的新物体与新动作，采集真机真实执行视频真值——**`In-lab Eval`**、**`EgoDex Eval`**、**`DreamDojo-HV Eval`**（各 100 条 49 帧三轮自回归预测样本，评测 PSNR $\uparrow$ / SSIM $\uparrow$ / LPIPS $\downarrow$）；
- **反事实动作可控性基准（`Counterfactual Eval`）**：专门采集常规机器人数据集绝不出现的非任务/失败动作（如伸手拍打玩具、伸向物体却故意抓空收回）；
- **两大开放环境背景编辑基准**：调用 **Gemini 2.5 Flash Image** 将 `EgoDex Eval` 与 `DreamDojo-HV Eval` 的实验室背景重绘为原始人类数据集中的多样家庭/商超/工坊真实复杂环境，构建无真值视频的 **`EgoDex-novel Eval`** 与 **`DreamDojo-HV-novel Eval`**（各 25 条，共 50 个样本对），由 **12 名志愿者**在盲测双视频加真值对照 Web UI（Figure 7）上从**物理正确性**（物体永续性、形状一致性、接触因果）与**动作跟随度**两维进行成对胜率评估。

### 4.2 核心消融一：连续潜动作 vs. 无动作与理想硬件真值上界（Table 2）

在相同步数（50k 步人类数据预训练 + 25k 步 GR-1 后训练）下对比四种预训练动作条件（灰色斜体代表需要额外穿戴动捕手套或 Vision Pro 硬件才能获取的理想真值上界）：

| 预训练动作条件方式 | `In-lab Eval` PSNR $\uparrow$ / SSIM $\uparrow$ / LPIPS $\downarrow$ | `EgoDex Eval` PSNR $\uparrow$ / SSIM $\uparrow$ / LPIPS $\downarrow$ | 可扩展性与因果迁移结论 |
| :--- | :---: | :---: | :--- |
| **w/o pretrain（直接微调 Cosmos-Predict2.5）** | 20.576 / 0.774 / 0.222 | 19.952 / 0.787 / 0.219 | 无人类交互物理先验基线 |
| **action-free（被动无动作视频预测预训练）** | 20.797 / 0.773 / 0.222 | 19.924 / 0.783 / 0.222 | **增益极微弱甚至在 EgoDex 倒退**，缺失动作-结果因果解耦 |
| **latent action（本文自监督 32 维连续潜动作）** | **20.913** / **0.776** / **0.219** | **20.344** / **0.790** / **0.214** | **纯像素自监督逼近硬件动捕上界，零外设无限扩展至 44k 小时** |
| *Oracle: retargeted action / Vision Pro MANO* | *20.960 / 0.773 / 0.219* | *20.474 / 0.795 / 0.211* | *依赖 Manus 手套或 Vision Pro 原生多相机 SLAM 的理想上界* |

后训练全程 PSNR 收敛曲线（Figure 13）进一步证实：连续潜动作预训练显著抬高了下游真机微调的**渐近性能天花板（Upper Bound）**，而被动无动作视频预训练很快与不预训练基线重叠停滞。

### 4.3 核心消融二：人类视频规模扩展与 2B / 14B 开放场景人类偏好（Table 3 & Table 4）

逐步叠加人类预训练数据源（50k 步均匀采样预训练 + 25k 步 GR-1 后训练）及全量 140k 步预训练的 **DreamDojo-2B / 14B** 完整表现如下：

| 模型 / 预训练数据混合 | `In-lab Eval` PSNR / SSIM / LPIPS | `EgoDex Eval` PSNR / SSIM / LPIPS | `DreamDojo-HV Eval` PSNR / SSIM / LPIPS | `Counterfactual Eval` PSNR / SSIM / LPIPS |
| :--- | :---: | :---: | :---: | :---: |
| **Cosmos-Predict2.5（无人类预训练）** | 20.576 / 0.774 / 0.222 | 19.952 / 0.787 / 0.219 | 18.274 / 0.754 / 0.236 | 20.472 / **0.802** / 0.190 |
| **+ In-lab (55 h)** | 20.913 / 0.776 / 0.219 | 20.267 / 0.785 / 0.218 | 18.621 / 0.754 / 0.233 | 20.755 / 0.796 / 0.187 |
| **+ In-lab + EgoDex (884 h)** | 20.972 / 0.778 / 0.216 | 20.334 / **0.791** / 0.215 | 18.706 / **0.762** / 0.230 | 20.797 / 0.796 / 0.188 |
| **+ In-lab + EgoDex + HV (44,711 h，50k 步)** | 21.016 / 0.781 / 0.215 | 20.414 / 0.790 / 0.216 | 18.724 / 0.759 / 0.232 | 20.852 / 0.799 / 0.188 |
| **DreamDojo-2B（全量 140k 步 1:2:10）** | 21.114 / 0.774 / 0.222 | 20.411 / 0.775 / 0.226 | 18.813 / 0.747 / 0.238 | 20.907 / 0.787 / 0.192 |
| **DreamDojo-14B（全量 140k 步 1:2:10）** | **21.413** / **0.788** / **0.208** | **20.525** / 0.787 / **0.213** | **18.924** / 0.751 / **0.228** | **21.087** / 0.793 / **0.185** |

在更具挑战性的全新复杂背景基准（`EgoDex-novel Eval` + `DreamDojo-HV-novel Eval` 共 50 组样本）上，12 名评估员的盲测成对偏好胜率（Table 4）展现出压倒性优势：

| 人类盲测评估维度 | DreamDojo-2B 胜过 Cosmos-Predict2.5 | DreamDojo-14B 胜过 Cosmos-Predict2.5 | DreamDojo-14B 胜过 DreamDojo-2B |
| :--- | :---: | :---: | :---: |
| **Physics Correctness（物理交互正确性）** | **62.50%** | **73.50%** | **72.50%** |
| **Action Following（动作精确跟随度）** | **63.45%** | **72.55%** | **65.53%** |

### 4.4 核心消融三：架构损失改造与实时自回归蒸馏（Table 5, Table 6 & Table 7）

#### （1）相对动作、分块因果注入与时序一致性损失逐项消融（Table 5，GR-1 数据 30k 步微调）

| 潜帧相对动作 | 4 步分块因果注入 | 时序速度差损失 $\mathcal{L}_{\text{temporal}}$ | `GR-1 Val`（专家轨迹）PSNR / SSIM / LPIPS | `Counterfactual Eval`（反事实动作）PSNR / SSIM / LPIPS |
| :---: | :---: | :---: | :---: | :---: |
| 否 | 否 | 否 | 16.199 / 0.557 / 0.315 | 19.448 / 0.768 / 0.211 |
| 是 | 否 | 否 | 16.522 / 0.576 / 0.304 | 19.482 / 0.772 / 0.212 |
| 是 | 是 | 否 | 17.626 / 0.620 / 0.267 | 20.783 / 0.790 / 0.193 |
| **是** | **是** | **是** | **17.630** / **0.622** / **0.266** | **20.980** / **0.796** / **0.189** |

尤为亮眼的是：**4 步分块因果注入（Chunked Injection）单枪匹马将 `GR-1 Val` 与 `Counterfactual Eval` 的 PSNR 暴拉 $+1.10\text{ dB}$ 与 $+1.30\text{ dB}$**，证明消除未来动作泄漏是治愈世界模型“无视反事实控制指令”的关键药方；叠加 $\mathcal{L}_{\text{temporal}}$ 后反事实 PSNR 进一步推升至 **20.980**。

#### （2）1 分钟（600 帧）超长视域蒸馏评测与预训练增益保持（Table 6 & Table 7）
在 **600 帧（1 分钟连续多阶段操作）`GR-1 Long Eval`** 压力测试（单张 H100 测速）与四大 OOD 基准蒸馏对比中：

| 对比组 | 模型变体 | `GR-1 Long Eval` PSNR / SSIM / LPIPS | 单卡帧率 FPS $\uparrow$ | 单步预测长 / 上下文长 | 四大 OOD 基准蒸馏后 PSNR（In-lab / EgoDex / HV / Counterfactual，Table 7） |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **教师 vs. 学生（Table 6）** | **Teacher（双向 35 步）** | **14.086** / **0.442** / **0.412** | 2.72 | 12 帧 / 1 帧 | — |
| | **Student（因果 4 步）** | 13.146 / 0.379 / 0.485 | **10.81（$3.97\times$）** | **4 帧** / **12 帧** | — |
| **蒸馏教师来源（Table 7）** | **Student（w/o 人类预训练）** | — | 10.81 | 4 帧 / 12 帧 | 20.304 / 19.119 / 17.869 / 19.782 |
| | **Student（w/ 人类预训练）** | — | **10.81** | **4 帧 / 12 帧** | **20.733** / **19.313** / **18.195** / **19.891** |

定性对比（Figure 11）证实，当被操作物体被机械臂短暂遮挡或相机视角发生偏移时，仅靠单帧条件的教师模型会出现物体凭空消失或几何塌缩，而拥有 **12 帧滑动上下文**的因果蒸馏学生模型能稳定记忆并恢复被遮挡物体，且 44.7k 小时人类视频预训练赋予的 OOD 泛化优势在 4 步极速学生模型中得到完整继承。

### 4.5 三大具身下游闭环应用实证（Section 4.7 & Figure 5–6）

1. **零硬件离线策略评估（Policy Evaluation，AgiBot 长时程水果装袋）**：
   在 20 个不同水果组合（梨、芒果、香蕉、杨桃，装满 5 个为 100% 成功）与摆位的桌面上，对单视角无本体状态版 **GR00T N1.5** 训练过程中的 6 个检查点（A–F），分别执行 **80 秒真机闭环 Rollout** 与相同初始帧下的 **DreamDojo-2B 80 秒完整虚拟闭环仿真**。两者平均成功率展现出极高的线性拟合度：**Pearson 相关系数达 $r = 0.995$，平均最大排序违背度 $\mathrm{MMRV} = 0.003$**（完美复现真机策略优劣排序）。
2. **测试时模型前瞻规划（Test-Time Model-Based Planning，10 个 AgiBot 水果装袋场景）**：
   集成 5 个策略检查点并行提出候选动作序列，由**蒸馏版实时 DreamDojo-2B** 批量推演未来视频片段，并交由基于冻结 **DINOv2** + 全局注意力的 4 帧滑窗外部**价值模型**（Appendix D.6，预测距当前子任务完成边界的归一化剩余步数，选取从起点至波谷平均值最小的候选动作下发真机执行）。两组检查点池的真机闭环成功率对比（由 Figure 5b 精确读取）如下：

| 候选策略池设置 | 5 个单策略检查点各自真机成功率（Model A–E） | 均匀随机抽样（Uniform Sampling） | **+ DreamDojo 前瞻规划（w/ DreamDojo）** | 核心增益幅度 |
| :--- | :---: | :---: | :---: | :--- |
| **高方差训练检查点组** | 5% / 16% / 35% / 47% / **53%** | 37% | **70%** | **较最强单检查点 +17 pp，较均匀采样提升 $1.89\times$（近翻倍）** |
| **已收敛后期检查点组** | 35% / 39% / 41% / 48% / **63%** | 55% | **68%** | **较最强收敛检查点再提 +5 pp，较均匀采样 +13 pp** |

3. **本地消费级显卡实时虚拟 VR 遥操作（Live Teleoperation，Figure 6）**：将蒸馏版 DreamDojo-2B 部署在配备单张 **NVIDIA RTX 5090** 的本地台式机上，接入 **PICO VR 手柄**实时采集操作员上半身动作指令，成功以实时帧率流畅遥操作虚拟 Unitree G1 机器人与桌面物体进行闭环物理交互。

---

## 五、局限性与未来方向

1. **极端高频罕见动作与精细失败模态的乐观偏差（Optimism Bias）**：面对拍打（slapping）、快速挥手等罕见高速动作时画面仍会出现残影；且在 AgiBot 策略评估中，DreamDojo 仿真的**绝对成功率系统性高于物理真机**（例如 Figure 5a 中最强检查点 F 在真机成功率约 45%，而在 DreamDojo 中约 81%），说明模型倾向于生成顺滑抓起的乐观结果，对毫米级打滑脱手等细微接触失败模拟不足，未来需引入真实策略失败 Rollout 数据（如 1X World Model、[WMPO](../vla/rl/WMPO_2025.md)）校准失败边界。
2. **尚不支持多视角同步仿真**：当前仅模拟单一第一视角相机，而 [$\pi_0$](../vla/foundation/pi0_2024.md)、GR00T N1 等主流真机策略普遍依赖“主视角 + 双腕相机”多视角联合输入。
3. **全参数后训练的灾难性遗忘风险**：目标机器人后训练阶段全量更新所有权重可能冲刷掉部分人类视频预训练长尾知识，引入 LoRA 或参数融合（Parameter Merging）是重要改进方向。

---

## 六、个人思考

1. **人类视频赋能具身智能的两大终极路线交汇——“隐式表征塑形（[EgoWAM](../world-action-models/EgoWAM_2026.md)）” vs. “显式像素交互仿真器（DreamDojo）”**：把 DreamDojo 与 [EgoWAM](../world-action-models/EgoWAM_2026.md) 放在一起看，恰好构成了利用海量人类视频的两极范式。[EgoWAM](../world-action-models/EgoWAM_2026.md) 坚决抛弃像素重建，只在训练期用 9M 参数的 3D Motion Flow / DINO 头塑造策略主干、推理期零开销摘除；而 **DreamDojo** 则正面强攻最难的**像素级动作条件生成式仿真器**，靠 **44.7k 小时规模暴力美学 + 32 维连续潜动作瓶颈 + 4 帧分块因果注入 + Self Forcing 10.81 FPS 蒸馏**，真的把像素视频世界模型推到了能做 80 秒长时程离线策略排序（$r=0.995$）和真机测试时规划（53% $\to$ 70%）的实用拐点！两者的共识同样深刻：**被动无动作视频预测（Action-free Video Prediction）对具身控制几乎无效**（DreamDojo Table 2 中 `action-free` 在 EgoDex 上甚至负增长），必须通过连续潜动作或三维稳像流显式建立“施动者动作 $\to$ 环境状态变迁”的因果瓶颈。
2. **为什么“4 步分块因果注入”能一举治好视频世界模型的“反事实失聪症”？**：Table 5 中 `+chunked` 带来的 $+1.30\text{ dB}$ 跃升非常值得所有做 WAM 和世界模型的研究者警醒。当把整条 12 步动作序列作为全局 Cross-Attention 条件暴露给所有视频潜帧时，双向 DiT 会偷看未来动作来反推当前帧（破坏物理马尔可夫因果律），甚至学会走“首帧视觉外观 $\to$ 专家默认轨迹终点”的捷径而屏蔽局部动作扰动。将每 4 步相对动作严格锁死在对应的单个 WAN2.2 视频潜帧 AdaLN 上，相当于在潜空间强制施加了**逐块因果结构归纳偏置**，这是让模型听懂反事实动作与支撑后续因果注意力自回归蒸馏的结构前提。
3. **离线策略评估中“排序极准（$r=0.995$）但绝对值虚高（81% vs. 45%）”的双刃剑效应**：这一现象与本仓库 RL 板块中的 [VLA-RFT](../vla/rl/VLA_RFT_2025.md)、[World-VLA-Loop](../vla/rl/World_VLA_Loop_2026.md) 高度呼应——生成式世界模型做**检查点相对好坏筛选**（Checkpoint Selection）与**多候选相对优劣打分**（Best-of-$N$ Planning）已经极其可靠，因为相对排序对全局乐观偏置免疫；但如果直接拿它在纯虚拟环境里做长周期强化学习策略优化（Model-Based RL），策略很快会学会钻这种“只要靠近水果就能被世界模型脑补吸进袋子”的物理漏洞（Reward Hacking）。打通“44k 小时人类预训练基座 + 真机闭环失败数据迭代回流微调（Real-to-Sim Failure Calibration）”将是填平这最后一道鸿沟的必由之路。
4. **跨数据集知识拼图的完美闭环**：DreamDojo 直接吸纳了同批次精读的 [EgoDex](../data-synthesis/EgoDex_2025.md)（829 小时高灵巧桌面数据）作为精细手物交互支柱，而 [EgoVerse](../data-synthesis/EgoVerse_2026.md) 持续增长的 1,362+ 小时全球多场景带精确 3D 手部追踪流又为下一代开源通用世界模型提供了天然的高质量开放语料。从 **数据基座（[EgoDex](../data-synthesis/EgoDex_2025.md) / [EgoVerse](../data-synthesis/EgoVerse_2026.md)）** $\to$ **通用实时世界仿真器（DreamDojo / [BridgeV2W](BridgeV2W_2025.md) / [MIND-V](MINDV_2026.md)）** $\to$ **世界动作模型与闭环强化演进（[EgoWAM](../world-action-models/EgoWAM_2026.md) / [Fast-WAM](../world-action-models/FastWAM_2026.md) / [WMPO](../vla/rl/WMPO_2025.md)）**，具身智能的完整技术栈拼图已清晰成型。

---

## 参考

- **Cosmos-Predict2.5**（Ali et al., 2025）/ **Wan2.2**（Wan et al., 2025）：DreamDojo 所继承的视频潜空间流匹配基础模型与 $4\times$ 时序压缩连续视频 Tokenizer
- **AdaWorld**（Gao et al., ICML 2025）/ **Genie**（Bruce et al., ICML 2024）/ **LAPA**（Ye et al., ICLR 2025）：本文 700M 连续潜动作时空 Transformer VAE 的核心自监督潜动作先驱工作
- **Self Forcing**（Huang et al., NeurIPS 2025）/ **DMD**（Yin et al., CVPR/NeurIPS 2024）：本文阶段三将双向多步扩散蒸馏为 10.81 FPS 因果自回归流式世界模型的核心算法基础
- **[EgoDex](../data-synthesis/EgoDex_2025.md)**（Hoque et al., ICLR 2026）：本文预训练三大人类数据源之一及 `EgoDex Eval` / `EgoDex-novel Eval` 基准与 MANO 上界实验的数据来源
- **[EgoVerse](../data-synthesis/EgoVerse_2026.md)**（Punamiya et al., RSS 2026）/ **[EgoWAM](../world-action-models/EgoWAM_2026.md)**（Li et al., CoRL 2026）：利用大规模第一视角人类操作数据实现跨具身迁移的数据生态与非像素世界动作模型对照路线
- **[BridgeV2W](BridgeV2W_2025.md)**（2025）/ **[MIND-V](MINDV_2026.md)**（2026）：本仓库世界模型板块中分别通过运动学像素对齐掩码与分层语义桥接 + GRPO 物理对齐提升机器人视频世界模型可控性的代表作
- **WorldEval**（Li et al., 2025）/ **SIMPLER**（Li et al., 2024）：本文离线策略评估实验所遵循的 Pearson 相关系数与 MMRV 排序一致性评测协议来源
