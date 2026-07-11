# LingBot-VLA 2.0：从基础模型到实际应用——VLA模型的工程化改进

> **论文**：*From Foundation to Application: Improving VLA Models in Practice*
>
> **作者**：Wei Wu、Fangjing Wang（共同一作）等，Kecheng Zheng（项目负责人）
>
> **机构**：Ant Group（Robbyant 团队）；egocentric 人类视频数据由 Ant Digital Technologies 旗下 Phecda Laboratory 与 Genrobot.ai Co., Ltd. 提供
>
> **发布时间**：2026 年 07 月（arXiv 2607.06403）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.06403) | [PDF](https://arxiv.org/pdf/2607.06403)
>
> **分类标签**：`VLA基础模型` `Mixture-of-Experts` `跨具身预训练` `长时程移动操作` `视频/深度蒸馏`

---

## 一句话总结

LingBot-VLA 2.0 通过重构预训练数据管线（约6万小时、20种具身形态的机器人数据+人类第一视角数据）、把动作空间扩展到头/腰/移动底盘/灵巧手的全身自由度、并引入"预测未来观测"作为代理任务（深度+因果视频双教师蒸馏），在 GM-100 双臂基准和两个长时程移动操作任务上相较前代 LingBot-VLA-1.0 与 π_0.5 取得一致提升。

## 一、问题与动机

VLA 基础模型近年来快速发展，但实验室基准与真实部署之间仍存在明显落差：(1) 泛化不仅要跨任务，还要跨异构具身配置与数据来源；(2) 真实平台往往比标准双臂设置具有更多自由度，包括头部运动、腰部、移动底盘和灵巧手；(3) 真实执行需要对未来场景演化和动作后果做预判，而不只是对当前观测做反应。这些问题共同限制了现有 VLA 基础模型的实用性。

作者提出应同时从数据、具身/动作空间、预测能力三个维度改进 VLA 系统，而不是只做单点式提升。核心假设是：实用型 VLA 系统不仅要在模型和数据规模上做 scale，还要更好地对齐真实机器人的需求——更广的具身支持、更丰富的可控动作空间、更强的动态场景预测理解。LingBot-VLA 2.0 即是沿着这一思路对前代 LingBot-VLA（技术报告见 arXiv:2601.18692，同一作者团队）的系统性改进。

## 二、核心方法

方法分三大板块：预训练数据与统一动作表示、MoE 动作专家架构、面向预测性动力学建模的双查询蒸馏。

### 2.1 预训练数据管线（约 60,000 小时）

**机器人数据**：从 20 种具身形态原始采集约 90,000 小时数据（涵盖单臂、双臂、半人形、全人形+灵巧手/夹爪平台），经过重新设计的过滤管线后保留 50,000 小时高质量数据。过滤依据包括：对动作与状态信号计算三阶有限差分（jerk）及速度/加速度的 Z-score，超过预设阈值（按具身分别设定）的 episode 被丢弃；若某 episode 中状态/动作信号"几乎静止"的时长占比超过 95%，同样被丢弃；通过 URDF 把机器人重投影到图像平面、人工核查视频与状态是否一致，剔除错位样本；此外人工剔除模糊、严重遮挡、丢帧或多视角错位的视频。

**第一视角（egocentric）数据**：从约 20,000 小时候选人类视频池中，经统一的 VLM 预筛选（剔除第三人称、场景漫游、无手-物交互、无可操作物体、以及非操作者手部主导的片段）、egocentric SLAM 位姿重建与手部姿态估计（恢复相机坐标系下的 MANO 参数）、有效帧比例≥20%阈值过滤、剔除不稳定 SLAM 轨迹与违反人体生理约束的手部运动后，最终保留 10,000 小时高质量数据。训练时统一以世界坐标系存储手部轨迹，在采样某一帧作为当前观测时再转换到该帧相机坐标系：

$$\mathbf{p}_\tau^{C_t} = \mathbf{T}_{C_t \leftarrow W}\,\mathbf{p}_\tau^{W}$$

用大白话说：轨迹统一存一份"世界坐标系"的版本，训练时取哪一帧就实时把未来轨迹转换到那一帧相机看到的坐标系下，这样既统一了开源数据和自采数据的动作表示，又把"手部自身运动"和"摄像头本身运动（头动、走动）"解耦开。

**统一动作表示**：设计 55 维规范向量同时表示状态和动作——14 维手臂关节位置、14 维末端位姿（每臂 XYZ+四元数共 7 维）、2 维夹爪、12 维手部关节、4 维腰部、2 维头部、3 维移动信号，其余 4 维预留；不具备某自由度的具身在对应维度做 padding。20 种具身形态的自由度配置汇总于论文 Table 1，累计训练数据 60,000 小时。

**数据标注**：用 Qwen3.6-27B 构建全自动流水线，把每段操作视频切分为若干子任务并生成对应语言指令；每个子任务从 15 个基元动作（move、pour、push、pull、rotate、open、close、detach、fold、unfold、wipe、stir、cut、press、attach）加 3 个辅助标签（transit 空手移动、idle 静止、other 词表外）构成的闭合词表中取一个类别，并配一个主要交互物体。统计上 move 占子任务频次的 46.21%（是频次最高的基元动作），辅助标签 transit 占 42.48%。

### 2.2 MoE 动作专家：token 级 loss-free 负载均衡

为提升跨具身预训练的模型容量，作者在 action expert 的每个 transformer block 中用稀疏 MoE FFN 替换稠密 FFN，采用细粒度专家切分 + 共享专家隔离（1 个共享专家保留通用先验 + N_r 个路由专家提供专精能力，每 token 激活 top-K 个路由专家）。

MoE 层输出：

$$m_\ell(u_{\ell,t}) = E_\ell^{(s)}(u_{\ell,t}) + \lambda \sum_{j \in \mathcal{R}(u_{\ell,t})} g_{\ell,j}(u_{\ell,t})\, E_{\ell,j}^{(r)}(u_{\ell,t})$$

用大白话说：每个 token 的输出 = 共享专家的固定贡献 + 被选中的若干路由专家的加权贡献之和（λ 是路由输出的整体缩放系数）。专家统一用 SwiGLU MLP 实现：$E(u) = W_{down}(\mathrm{SiLU}(W_{gate}u) \odot W_{up}u)$。

路由部分先用 FP32 精度的线性层计算 token 与专家的相似度 logits $z_{\ell,j}(u_{\ell,t}) = u_{\ell,t}^\top e_{\ell,j}$，再用 sigmoid（而非 softmax，避免专家间强竞争）转成亲和度 $s_{\ell,j}(u_{\ell,t}) = \mathrm{Sigmoid}(z_{\ell,j}(u_{\ell,t}))$。借鉴 DeepSeek-V3 的 auxiliary-loss-free 负载均衡思路，每个专家维护一个路由修正偏置 $b_{\ell,j}$：实际混合权重仍由未加偏置的原始亲和度归一化得到，

$$g_{\ell,j}(u_{\ell,t}) = \frac{s_{\ell,j}(u_{\ell,t})}{\sum_{k\in\mathcal{R}(u_{\ell,t})} s_{\ell,k}(u_{\ell,t})}, \qquad j\in\mathcal{R}(u_{\ell,t})$$

而被选中的专家集合则由加了偏置后的分数决定：

$$\mathcal{R}(u_{\ell,t}) = \mathrm{TopK}_j\big(s_{\ell,j}(u_{\ell,t}) + b_{\ell,j},\ K\big)$$

偏置按累计负载与均值负载偏差的符号做迭代更新：

$$b_{\ell,j} \leftarrow b_{\ell,j} - \gamma \cdot \mathrm{sign}\Big(n_{\ell,j} - \frac{1}{N_r}\sum_{k=1}^{N_r} n_{\ell,k}\Big)$$

用大白话说：选谁来算（路由）看的是"偏置修正后"的分数以保证各专家分到的 token 数量大致均衡，但真正加权融合输出时用的还是"没被偏置污染"的原始置信度；这样负载均衡的调节动作不会污染动作学习这一主目标，也就不需要在主 loss 里额外加一项负载均衡损失。

Scaling 实验：在严格匹配 active 参数量的前提下（Dense 0.6B vs MoE 总参数 1.6B / 激活参数 0.6B），MoE 在预训练 loss 和 GM-100 验证动作误差两个指标上，训练全程（10k–50k 步）都持续优于同等激活参数量的 Dense 模型，说明收益来自稀疏激活对模型容量的更有效分配，而非单纯的总参数量增加。

### 2.3 双查询蒸馏：把"预测未来"变成代理任务

为增强几何感知和因果时序推理，作者引入 dual-query distillation：在 VLM 的视觉/文本 token 序列后追加两个可学习查询 $[\mathbf{Q}_t, \mathbf{Q}_{t+T}]$，$\mathbf{Q}_t$ 对齐当前观测，$\mathbf{Q}_{t+T}$ 对齐未来 T 步（动作 chunk 长度）后的观测，分别由两个互补 teacher 监督蒸馏。

深度 teacher（LingBot-Depth，提供显式几何监督）：

$$\mathcal{L}_{depth} = \mathbb{E}\Big[\|\mathrm{Proj}_{depth}(\mathbf{Q}_t)-\mathbf{D}_t\|_1 + \|\mathrm{Proj}_{depth}(\mathbf{Q}_{t+T})-\mathbf{D}_{t+T}\|_1\Big]$$

因果视频 teacher（DINO-Video，提供时序落地的视觉表征）：

$$\mathcal{L}_{video} = \mathbb{E}\Big[\|\mathrm{Proj}_{video}(\mathbf{Q}_t)-\mathbf{Z}_t\|_F^2 + \|\mathrm{Proj}_{video}(\mathbf{Q}_{t+T})-\mathbf{Z}_{t+T}\|_F^2\Big]$$

用大白话说：给模型内部专门开两个"占位查询"，一个负责去猜"现在这个场景长什么样"，另一个负责去猜"T 步之后场景会演变成什么样"，分别拿深度图特征和视频特征当 ground truth 做蒸馏，逼着 VLA 在推理时天然具备对未来几何结构和场景演化的预判能力，而不需要显式生成未来帧或深度图，推理开销可控。

DINO-Video 是作者自研的"机器人场景感知"视频表征模型：在 DINOv3 基础上加入 block-wise 因果时序注意力和 3D-RoPE 位置编码，使每个时刻的特征只依赖当前及历史观测；用约 5M 段跨互联网/第一视角/机器人来源的视频、结合视频版 DINO 与 iBOT 自蒸馏目标训练而成。在 LARYBench 分类与回归评测（Table 3）上，DINO-Video 在四项子指标中的三项取得最优。

## 三、实验结果

### 3.1 DINO-Video 表征质量（LARYBench，Table 3）

| 模型 | 参数量(M) | Composite Human↑ | Composite Robot↑ | RoboCOIN↓ | AgiBotWorld-Beta↓ |
|---|---|---|---|---|---|
| V-JEPA 2 | 303.89 | 80.35 | 70.43 | 0.32 | 0.33 |
| DINOv3 | 303.13 | 76.19 | 69.06 | 0.22 | 0.24 |
| DINO-Video（本文） | 303.13 | 80.21 | 71.97 | 0.20 | 0.19 |

DINO-Video 在 Composite Robot、RoboCOIN、AgiBotWorld-Beta 三项上最优，仅 Composite Human 略低于 V-JEPA 2。

### 3.2 GM-100 双臂通用操作（generalist mixed-training，9 任务，Table 5，单位 Prog./Succ. %）

Agilex Cobot Magic 平台（整体均值）：

| 模型 | Prog. | Succ. |
|---|---|---|
| GR00T N1.7 | 36.3 | 17.8 |
| π_0.5 | 59.1 | 32.2 |
| LingBot-VLA-1.0 | 58.2 | 30.0 |
| LingBot-VLA-2.0 | 66.2 | 34.4 |

Galaxea R1Pro 平台（整体均值）：

| 模型 | Prog. | Succ. |
|---|---|---|
| GR00T N1.7 | 16.4 | 5.6 |
| π_0.5 | 27.4 | 8.9 |
| LingBot-VLA-1.0 | 32.7 | 15.6 |
| LingBot-VLA-2.0 | 34.6 | 15.6 |

LingBot-VLA-2.0 相较 1.0 在 Agilex 平台提升 +8.0/+4.4pt，相较 π_0.5 提升 +7.1/+2.2pt；Galaxea 平台上相较 π_0.5 提升 +7.2/+6.7pt。单任务层面提升更显著，如 Agilex "Retrieve keychain" 从 1.0 的 67.5/60.0 跃升到 2.0 的 100.0/100.0，"Pick out toy bone" 从 77.5/70.0 升到 95.0/90.0；Galaxea "Pick out toy bone" 从 62.5/40.0 升到 87.5/70.0。作者将其归因于更强的 VLM backbone 带来更好的视觉 grounding，以及未来信息条件化的动作预测。

### 3.3 长时程移动操作（Table 6，每设置 15 次独立试验，ID=分布内，OOD=初始位姿扰动±10cm+部分场景换成未见物体类别）

| 具身 | 任务 | 设置 | LingBot-VLA-2.0 (Prog./Succ.%) | π_0.5 (Prog./Succ.%) |
|---|---|---|---|---|
| Astribot S1 | 冰箱物品分拣 | ID | 77.1 / 60.0 | 65.3 / 46.7 |
| Astribot S1 | 冰箱物品分拣 | OOD | 37.0 / 13.3 | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | 灶台清洁 | ID | 84.3 / 66.7 | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | 灶台清洁 | OOD | 67.5 / 40.0 | 62.5 / 33.3 |

LingBot-VLA-2.0 在 ID 与 OOD 设置下均一致优于 π_0.5。冰箱分拣任务从 ID 到 OOD 的性能衰减比灶台清洁更大，原因是该任务的 OOD 设置同时改变了机器人初始位姿和被操作物体类别，需要更强的物体级泛化与长时程恢复能力；灶台任务的 OOD 只扰动位姿，保留了物体与场景结构。

### 3.4 消融实验（4 个 GM-100 真机任务：Barcode Scan / Scoop Rice / Squeeze Ketchup / Take Bowl from Microwave，均值成功率 %）

| 消融维度 | 选项 A | 成功率 | 选项 B | 成功率 |
|---|---|---|---|---|
| 动作目标 | 绝对关节角 absQpos | 33.7 | 相对关节角 relQpos | 55.0 |
| 动作空间 | EEF | 56.0 | Joint | 55.0 |
| 归一化 | MinMax | 47.5 | MeanStd | 55.0（Q01–Q99 为 47.4） |
| 损失函数 | L1 | 46.4 | L2 | 55.0 |

要点：relQpos 把预测目标从"全局关节构型回归"转为"局部运动量回归"，标准差压缩到 absQpos 的 31%–37%（池化标准差从约 0.80 降到约 0.28），显著降低方差、提升成功率；EEF 与 Joint 动作空间平均分接近，但任务级偏好差异很大（如 Barcode Scan 上 Joint 58.7 远超 EEF 24.0，Squeeze Ketchup 上 EEF 81.7 远超 Joint 41.7），单纯的动作分布对齐度只能部分解释这一现象，还与任务本身的物理结构（可达性、姿态约束）有关；MeanStd 归一化保留了 relQpos 的长尾分布特性（约 10% 样本落在 \|x\|>1.5），优于压缩过度的 MinMax（标准差仅 0.15）；损失函数上 L2 更适合 relQpos 以小幅高密度修正为主的分布，L1 只在接触密集、重尾运动的 Squeeze Ketchup 任务上更稳健。

## 四、局限性

- 论文自身在结果分析中指出：GM-100 双臂任务上模型经常"做到最后一步（精确放置/释放/收尾）时失败"，progress score 与 success rate 之间存在明显 gap，说明长尾的精细收尾动作仍是主要瓶颈。
- 不同具身平台间性能差距明显（Agilex 66.2/34.4 vs Galaxea 34.6/15.6），作者将其归因于运动学、相机视角、动作空间对齐等具身特异性因素，但论文未给出针对性的解决方案。
- OOD 评测仅覆盖位姿扰动（±10cm）与物体类别替换两类扰动，未覆盖光照变化、遮挡、干扰物、语言复杂度等更广泛的分布偏移场景。
- 关于动作空间/归一化/损失函数的消融实验只在 4 个真机任务上开展，样本量和任务多样性有限，"任务物理结构决定最优动作空间"这一结论目前仍偏观察性，缺乏更细粒度的因果解释。
- 论文未披露模型总参数量、具体训练算力、MoE 专家数 N_r 与激活数 K 等关键超参数的完整取值，可复现性依赖后续开源代码与权重。
- 作为技术报告性质的工作，尚缺乏第三方或跨机构复现验证；长时程移动操作评测仅覆盖两种具身、两个任务场景，统计规模偏小（每组 15 次试验）。
- 预测性蒸馏（dual-query distillation）模块未单独做端到端消融，其对最终 GM-100/长时程任务提升的独立贡献大小无法从论文中确认。

## 五、评价与展望

本文延续了 LingBot-VLA 系列（前作 arXiv:2601.18692，同一作者团队）的系统性工程报告风格，核心贡献不在于单点算法创新，而在于把数据管线、全身动作空间统一表示、MoE 可扩展架构、预测性蒸馏目标四者作为一个整体协同优化。这与 GR00T N1、π_0.5、Being-H 系列等公开工作代表的"基础模型规模化"路线是互补而非对立的：GM-100 与长时程移动操作实验表明，在同等评测协议下这种"工程集成"路线能带来一致但非戏剧性的提升（多数指标提升幅度在 5–10 个百分点量级）。

优点：(1) 55 维统一动作表示 + padding 机制，是目前公开报道中处理 20 种异构具身（含头/腰/移动底盘/灵巧手）较为系统的方案之一；(2) auxiliary-loss-free MoE 路由借鉴 DeepSeek-V3 思路，并在 action expert 上做了受控的 active-parameter 对齐实验，提供了比"仅比较总参数量"更有说服力的稀疏架构收益证据；(3) dual-query distillation 把"预测未来"转化为对预训练视觉 teacher（深度 + 因果视频）的蒸馏目标，不需要显式生成未来帧或深度图，推理开销可控，思路上与近期"latent action model"和"world model 作辅助监督"一脉相承，但用双 teacher 分别提供几何先验与语义-时序先验的设计较为直接清晰。

局限与开放问题：(1) 论文没有对预测性蒸馏模块做独立消融，无法确认其在最终指标提升中的边际贡献，该收益目前是与数据规模、动作空间扩展等改进耦合在一起报告的；(2) DINO-Video 作为自研视频 teacher 仅在 LARYBench 四项子指标中的三项最优，并未全面超过 V-JEPA 2/DINOv3，其对下游 VLA 性能的边际价值有待更细粒度消融验证；(3) GM-100（The Great March 100）是作者关联的自建基准，尚未见广泛第三方使用，横向可比性有限；(4) 不同具身间的性能差距未被彻底解释，是否存在更好的具身条件化路由（而非 token 级无差别路由）是可能的改进方向；(5) 长时程移动操作评测规模（2 具身 × 1 任务 × 15 trial）偏小，统计显著性有限。整体而言，这是一篇偏工程系统集成、数据/架构/目标三线并进的 VLA 实践报告，对理解"从基础模型走向实际部署"过程中哪些工程杠杆（数据质量过滤阈值、动作表示的相对化与归一化、MoE 负载均衡策略）真正带来收益具有参考价值。

## 参考

[4] Bjorck et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv:2503.14734, 2025.

[5] Black et al. π_0.5: A vision-language-action model with open-world generalization. CoRL, 2025.

[19] Liu et al. DeepSeek-V3 technical report. arXiv:2412.19437, 2024.

[26] Siméoni et al. DINOv3. arXiv, 2025.

[38] Wu, Lu, Wang et al. A pragmatic VLA foundation model (LingBot-VLA 1.0 前作). arXiv:2601.18692, 2026.
