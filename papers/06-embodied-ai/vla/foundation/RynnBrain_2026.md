# RynnBrain：开源具身基础模型

> **论文**：*RynnBrain: Open Embodied Foundation Models*
>
> **作者**：Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu 等（核心贡献者按字母序排列）; 高级顾问包括 Jun Cen, Siteng Huang, Wenqiao Zhang（浙江大学）, Chengju Liu（同济大学）, Jianfei Yang / Shijian Lu（南洋理工大学）, Deli Zhao
>
> **机构**：DAMO Academy, Alibaba Group
>
> **发布时间**：2026 年 02 月（arXiv 2602.14979）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.14979) | [PDF](https://arxiv.org/pdf/2602.14979)
>
> **分类标签**：`具身基础模型` `时空定位` `Chain-of-Point推理` `视觉语言导航` `VLA` `GRPO强化学习`

---

## 一句话总结

RynnBrain 是基于 Qwen3-VL 构建的开源具身"大脑"基础模型家族（2B/8B/30B-A3B MoE 三种规模,近 20M 训练样本）,把物体、区域、affordance、轨迹全部离散化为 $[0,1000]$ 坐标 token 与文本统一自回归输出,并引入 Chain-of-Point（CoP）交织推理与 GRPO 强化学习让推理步骤锚定在具体空间证据上；在此基础上派生出 RynnBrain-Nav（导航）、RynnBrain-Plan（操作规划）、RynnBrain-VLA（动作执行）、RynnBrain-CoP（空间推理）四个后训练变体,在 28 个基准上大幅超过 RoboBrain 2.0、MiMo-Embodied、Pelican-VL 等开源具身大脑模型,8B 版 CoP 推理模型在 affordance/area/trajectory 三项平均得分 73.8,超过 RoboBrain2.0-32B 达 16.1%。

## 一、问题与动机

具身智能领域目前缺少一个统一的、物理接地的基础模型,能够在真实世界的时空动态中同时完成感知、推理与规划。作者指出现有 VLM（如 GPT-4o、Gemini）虽通用能力强,但并未内在地建立在物理动态之上,难以做到时空一致性、物理推理与可执行规划；反过来,以动作为中心训练的具身模型虽然贴近执行,却往往牺牲了从大规模多模态预训练继承来的高层语义泛化能力。

论文进一步把已有"具身大脑"模型（如 RoboBrain 2.0、Robix 等）的局限总结为三点:

1. **认知能力狭窄**——训练往往局限在有限任务类别或感知模态内,复杂环境下鲁棒性不足；
2. **空间推理停留在静态图像**——缺乏支持全局场景感知和移动操作所需的连贯时空表示；
3. **高层推理/规划在纯文本空间中进行**——容易产生幻觉,和物理约束脱节。

RynnBrain 试图用一个统一框架同时解决这三点,强调四项核心能力：全面的第一视角（egocentric）理解、多样化的时空定位（物体/区域/affordance/轨迹）、物理接地的推理（Chain-of-Point）、以及物理感知的规划（把 affordance/区域/物体坐标直接写入规划输出,供下游策略模型使用）。

## 二、核心方法

### 2.1 架构与统一输出空间

RynnBrain 沿用 Qwen3-VL 的解码器-only 视觉-语言架构（视觉编码器 + 投影层 + LLM backbone,分别从 Qwen3-VL-2B/8B/30B-A3B-Instruct 初始化）,并引入 DeepStack 与 Interleaved MRoPE 增强多模态融合。核心设计是**物理接地的输出空间**：不同于把空间量当作自由文本生成的传统 VLM,RynnBrain 把所有空间实体——包括边界框 $\mathcal{B}$、点 $\mathcal{P}$、轨迹路点 $\mathcal{T}$——统一归一化到 $[0,1000]$ 区间并编码为整数 token,把连续空间预测转化为分类问题,使模型能用与语言生成完全相同的自回归机制产生精确、物理有意义的空间输出。

训练损失是标准的下一 token 预测目标：

$$\mathcal{L} = -\sum_{i=1}^{L} \log P\left(y_i \mid y_{<i}, \mathbf{V}, \Theta\right).$$

用大白话说：模型不区分"这是在说话"还是"这是在指坐标",统一当作序列里的下一个 token 去预测,坐标只是词表里多出来的一批"数字词"。

### 2.2 训练基础设施：解决长尾序列长度导致的负载不均

具身多模态训练数据序列长度呈现高方差长尾分布（短的定位任务 vs 长视频描述/推理）,朴素地把样本均分给各 DP worker 会造成严重的 straggler 效应。论文设计了一个在线负载均衡流水线：先根据图像尺寸和文本 token 数预估每条样本的序列长度,再用贪心近似算法——按长度降序排序、每次把样本分给当前总长度最小的 buffer——在数据预取阶段把全体样本重新分配到各 DP worker,使每个 worker 的累计序列长度尽量接近。

为避免传统按 token 数归一化损失（式 1）需要跨 DP 组做 all-gather 同步带来的通信开销：

$$\mathcal{L} = \frac{1}{\sum_{i=1}^{n}\sum_{j=1}^{b_i} s_{ij}} \sum_{i=1}^{n}\sum_{j=1}^{b_i}\sum_{k=1}^{s_{ij}} l_{ijk},$$

RynnBrain 改用按样本归一化的损失（式 2）：

$$\mathcal{L} = \frac{1}{b}\sum_{i=1}^{n}\sum_{j=1}^{b_i}\frac{1}{s_{ij}}\sum_{k=1}^{s_{ij}} l_{ijk},$$

其中 $b$ 是全局 batch size（各 worker 已知的常量）,不再需要额外通信。用大白话说：与其为了精确的全局归一化去"开会对齐"每个 worker 算出的 token 总数,不如换一种数学等价但各算各的归一化方式,省掉的通信开销让整体训练吞吐提升了约一倍。工程上,2B/8B 用 ZeRO-1 + 逐 block 梯度检查点,30B-A3B（MoE）用 ZeRO-2 + 专家并行（world size 2）,并基于 NVIDIA CUTLASS 实现分组线性算子、用 DeepEP 做跨 GPU token 分发。

### 2.3 预训练数据（约 19.89M 样本）

预训练数据分四大类（Table 2 汇总,单位 M 样本）：

| 类别 | 子任务 | 样本量 |
|---|---|---|
| General MLLM | 通用图文/视频理解 | 4.80 |
| Cognition | 物体理解 1.10、空间理解 2.50、计数 0.30、OCR 1.00、第一视角任务理解 2.77 | 7.67 |
| Localization | 物体定位 1.20、区域定位 3.37、affordance 定位 1.13、轨迹预测 0.56、抓取姿态 1.00 | 7.26 |
| Planning | 操作规划 | 0.16 |
| **合计** | | **19.89** |

数据构建强调"人机协同数据飞轮"：先用预训练基础模型（Qwen2.5-VL、Grounding DINO 1.5、SAM2、MASt3R-SLAM 等）做初始标注/候选生成,再由人工标注员在关键决策点校验,以在有限标注预算下把语料扩展到千万级。例如空间理解数据通过 MASt3R-SLAM 重建 3D 点云、RANSAC 检测地面实现重力对齐坐标系,再基于几何关系模板化生成距离/方位 QA；抓取姿态数据基于 Grasp-Anything 的定向矩形标注转换为 4 角点表示。

### 2.4 Chain-of-Point（CoP）物理接地推理

第 4 节提出 CoP 推理范式：在文本推理链中显式交织空间定位。冷启动 SFT 阶段的数据构建流程是——先用 Qwen3-VL-235B 生成逐步文本推理链,并用方括号标记候选实体（如 `[white flower-patterned wallpaper]`）；再用一个内部模型把每个实体分类为 "area" 或 "object"；最后人工标注员为每个实体选取最清晰的帧并做精确标注（area 标点集,object 标 bbox）,再把定位结果按 `<object/area> <frame n>: (coordinates)` 格式插回推理文本中,形成"推理步骤—具体视觉证据"交织的 CoT 数据集。SFT 用全参数微调,LM/投影层峰值学习率 $1\times10^{-5}$,视觉编码器 $2\times10^{-6}$,2FPS 采样最多 2048 帧,最大上下文 16384 token。

随后用 GRPO（Group Relative Policy Optimization）做强化学习对齐,目标函数为：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta\, \mathbb{D}_{KL}(\pi_\theta(o_i\mid q)\|\pi_{\text{ref}}(o_i\mid q))\right)\right],$$

其中重要性采样比 $\rho_i = \pi_\theta(o_i\mid q)/\pi_{\theta_{old}}(o_i\mid q)$,组内优势通过组内奖励归一化得到：

$$A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\}) + \epsilon}.$$

用大白话说：GRPO 不需要单独训练一个价值网络（critic）来估计好坏,而是同一个 prompt 采样 $G$ 个输出后互相比较,谁比组内平均好就是"正优势",省去了 PPO 里 critic 带来的显存和不稳定性。

奖励函数按任务类型设计成规则化、可复现的几何度量：

- **轨迹奖励**：先对预测点序列 $\mathcal{P}$ 与真值序列 $\mathcal{G}$ 按弧长重采样为等点数,再计算离散 Fréchet 距离（DFD, 式 6 递归定义耦合距离 $c(i,j)$）,奖励做指数衰减：$r_{\text{traj}} = \exp(-\lambda_{\text{traj}} \cdot D_F)$。直觉：不是简单比较端点误差,而是比较两条曲线"最坏对齐点"的距离,更贴合轨迹形状是否吻合。
- **Affordance 奖励**：用双向平均欧氏距离（Chamfer 距离变体）同时惩罚无效预测（精度）和覆盖不全（召回）：$D_{\text{bidir}}(\mathcal{P},\mathcal{G}) = \frac{1}{2}\left(\frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}}\min_{g\in\mathcal{G}}\|p-g\|_2 + \frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}\min_{p\in\mathcal{P}}\|p-g\|_2\right)$,奖励 $r_{\text{aff}} = \exp(-\lambda_{\text{aff}} \cdot D_{\text{bidir}})$。
- **区域奖励**：把区域识别看作多边形内点检验问题,$r_{\text{area}} = \frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}} \mathbb{I}(p \in S_{\mathcal{G}})$,即预测点落在真值多边形内的比例。

RL 训练数据经"难度感知过滤"筛选：用 SFT 模型给候选样本打分,只保留中等难度（40\textendash 80 分区间）样本,并额外补充一批 SFT 模型选错关键帧的失败案例,最终得到 3 万条高质量强化学习样本,把探索约束在物理合理的区间内以减少幻觉。

### 2.5 定位任务的评测指标

对定位（Grounding）任务,论文用 Acc@0.5——要求模型先选中含有效真值的关键帧,再要求预测框与真值 IoU 超过 0.5：

$$\text{Acc@0.5} = \mathbb{I}\left(\mathcal{G}_t \neq \emptyset \wedge \text{IoU}(\mathcal{B}, \mathcal{G}_t) > 0.5\right).$$

对 Pointing 任务（area/trajectory/affordance）,若模型选错关键帧则直接记 0 分,体现了论文对"先定位时间、再定位空间"两阶段联合评价的强调。

### 2.6 四个后训练变体

- **RynnBrain-Nav（VLN）**：离散动作空间 $\{\uparrow,\leftarrow,\rightarrow,\text{STOP}\}$（前进 30cm、转向 15°）,采用多轮对话格式（借鉴 StreamVLN）把轨迹表示为交织的图文序列 $\{o_0,a_0,o_1,a_1,\dots,o_n,a_n\}$。训练数据用 Habitat 仿真在 R2R、R2R-EnvDrop、RxR（60 个 MP3D 场景,45 万条）+ ScaleVLN 子集（30 万条）采集,并用多轮 DAgger 迭代补充在线轨迹。
- **RynnBrain-Plan（操作规划）**：用多轮对话格式的历史交互作为显式记忆缓冲区,定位标注只施加在每轮对话的最后一帧,使当前决策同时依赖即时观测和累积记忆；论文强调该方案数据效率极高,几百条样本即可获得鲁棒的长时程规划与泛化能力。
- **RynnBrain-VLA（动作执行）**：基于 RynnBrain-2B,采用 flow matching 框架在每步预测一个动作 chunk（follow $\pi_0$ 范式）,VLM backbone 作为单流 Diffusion Transformer,把条件（指令、观测、状态）与带噪动作打包进同一序列,动作被放在序列末尾以支持推理时的 KV cache 复用。在 Franka Emika 机械臂上用遥操作采集的 6 类拾放任务（3 种物体）微调 6 万步。
- **RynnBrain-CoP（复杂空间推理）**：即 2.4 节所述交织推理模型,专注 affordance/area/trajectory 三类物理接地预测任务。

## 三、实验结果

### 3.1 具身认知能力（Table 3,8B 及以下规模；Table 4,≥30B 规模）

| 基准 | RynnBrain-8B | 对比最优（≤8B 组开源） | RynnBrain-30B-A3B | 对比（≥30B 组，含闭源） |
|---|---|---|---|---|
| VSI-Bench | **71.0** | Qwen3-VL-8B 60.3 | **74.5** | Gemini 3 Pro 49.2 |
| RoboSpatial | **73.1** | MiMo-Embodied-7B 61.8（超 11.3%） | 70.0 | Gemini 3 Pro 56.0 |
| EgoTaskQA | 72.5 | RynnBrain-2B 73.9 | **78.9** | 超前方法 10.5%（文中口径） |
| Open-X VQA | **74.0** | Qwen3-VL-8B 59.8 | **83.4** | 超前方法 6.6% |
| RynnBrain-Object（自建） | **71.2** | Qwen3-VL-8B 41.8 | 73.3 | 超前方法 20.2% |
| RynnBrain-Spatial（自建） | **59.9** | Qwen3-VL-8B 35.0 | **59.3** | 超前方法 25.1% |

### 3.2 空间定位能力（Embodied Location, 五类：物体/区域/affordance/轨迹/抓取姿态）

RynnBrain-8B 在除 ShareRobot-Trajectory 外的所有定位基准上领先（该项 2B 模型反而最优,以 5.8% 优势超过基座 Qwen3-VL）；RefSpatial-Bench 上 8B 达 59.2,超过最接近的竞争者 7.7%。抓取姿态定位：8B 在 Cornell-Grasp 达 26.6、VMRD-Grasp 达 14.1,显著超过其他 8B 规模模型。30B-A3B 在 Cornell-Grasp (33.6)、VMRD-Grasp (14.5)、RynnBrain-Grounding (83.9)、RynnBrain-Affordance (90.5) 上均取得最佳,并在 RefSpatial-Bench、RynnBrain-Area、RynnBrain-Trajectory 上逼近远大于自己的 Gemini 3 Pro。

### 3.3 物理接地推理（RynnBrain-CoP-8B, Table 5）

| 模型 | Affordance | Area | Trajectory | 平均 |
|---|---|---|---|---|
| InternVL3.5-8B | 63.1 | 9.2 | 47.8 | 40.0 |
| MiMo-Embodied-7B | 85.3 | 47.1 | 64.9 | 65.8 |
| RoboBrain2.0-32B | 73.2 | 39.5 | 60.5 | 57.7 |
| Qwen3-VL-30B-A3B-Thinking | 62.2 | 33.0 | 54.8 | 50.0 |
| GPT-5.2 | 83.3 | 35.8 | 70.5 | 63.2 |
| Gemini-3-Pro | 83.9 | 50.7 | 60.6 | 65.1 |
| **RynnBrain-CoP-8B** | **90.3** | **59.6** | **71.2** | **73.8** |

RynnBrain-CoP-8B 是唯一在 affordance 上突破 90 分的模型,在 area（历来最难的任务,多数基线低于 40 分）上接近 Qwen3-VL-30B-A3B 的两倍,以 8B 参数超过参数量大四倍的 RoboBrain2.0-32B 达 16.1%。

### 3.4 视觉语言导航（Table 6, R2R-CE / RxR-CE Val-Unseen）

| 模型 | R2R NE↓ | R2R OS↑ | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ |
|---|---|---|---|---|---|---|
| StreamVLN | 4.98 | 64.2 | 56.9 | **51.9** | 52.9 | 46.0 |
| **RynnBrain-Nav-8B** | **4.92** | **71.6** | **58.6** | 49.6 | **56.1** | **49.6** |

论文承认：OS（71.6%）远高于 SR（58.6%）,说明模型擅长粗粒度导航但在终止停靠这一精确动作上仍有欠缺。多轮 DAgger 迭代下 SR 从 50.6%（初始 SFT）提升到 56.4%（第一轮）、58.5%（第二轮）,第三轮收益已边际递减。规模消融显示：2B 版 RynnBrain-Nav 比同规模 Qwen3-VL 微调基线 SR 高 7.2%、SPL 高 7.6%,但 30B-A3B（MoE, 3B 激活）在导航任务上未能超过 8B 稠密模型,论文将其归因于 MoE 的稀疏激活机制未被 VLN 任务充分利用。

### 3.5 操作规划与 VLA（Table 7/8, Figure 6）

三阶段评估体系：（1）RynnBrain-Plan 做高层规划 + 人工 UMI 操作员做可靠底层执行；（2）RynnBrain-VLA 独立评估精细抓取；（3）RynnBrain-Plan + RynnBrain-VLA 端到端集成部署于 Franka 机械臂。

在 Object Classification / Desk Organization / Distribute Tableware（域内）与 Table Bussing（域外 OOD）四项长时程任务、Easy/Medium/Hard 三档难度下,RynnBrain-Plan-30B-A3B 以 Task Progress 指标全面超过 Qwen3-VL-30B 与 Gemini-3-Pro：例如 Desk Organization-Hard,Qwen3-VL 与 Gemini-3-Pro 接近 0% 完成度,RynnBrain-30B 保持 75% 以上；OOD 的 Table Bussing-Hard 上 Qwen3-VL 完全失败（低于 10%）、Gemini-3-Pro 约 60%,RynnBrain-30B 接近 100%。多轮对话数据的消融（Table 7）显示：单轮对话训练的 RynnBrain-Plan-ST 在 Hard 难度下几乎完全失效（多为 0）,而多轮对话版本（MT）大幅提升,证明历史记忆对长时程规划的必要性。

VLA 评估（Table 8,Pickup Success/Recognition Success/Success Rate）：RynnBrain-VLA 总体 PSR=0.8、RSR=0.97、SR=0.77,显著超过 $\pi_{0.5}$-Finetuned（PSR=0.67, RSR=0.57, SR=0.47）与同架构对照 Qwen3-VL-Finetuned（PSR=0.60, RSR=1.00, SR=0.60）。论文将 $\pi_{0.5}$ 的低 RSR 归因于其在细粒度图文对齐上的能力瓶颈,而 RynnBrain-VLA 得益于大规模 embodied pointing 预训练带来的更强场景理解与定位能力。

## 四、局限性

1. **导航的"粗定位强、精停靠弱"问题**：论文自己指出 R2R-CE 上 OS（71.6%）远高于 SR（58.6%）,表明模型能大致找到目标区域但终止动作精度不足,该问题尚未被专门解决。
2. **MoE 架构在 VLN 任务上未表现出规模优势**：30B-A3B（3B 激活）未能超越 8B 稠密模型,论文将其归因于稀疏激活机制与 VLN 任务的不匹配,但未给出根本性分析或改进方案。
3. **Area 预测仍是最难的子任务**：即便是表现最好的 RynnBrain-CoP-8B 也仅达到 59.6 分（满分 100）,远低于其在 affordance（90.3）上的表现,说明"开放区域"级别的空间推理仍有较大提升空间。
4. **真机操作评测范围有限**：RynnBrain-VLA 仅在单台 Franka Emika 机械臂、3 种物体、6 类拾放任务上微调评测,缺乏双臂、柔性物体、多样化具身形态或大规模真机部署验证；RynnBrain-Plan 与 RynnBrain-VLA 训练所用的"in-house"数据集规模很小且未公开细节,复现性存疑。
5. **规模趋势不完全一致**：ShareRobot-Trajectory 基准上 2B 模型反超 8B 模型,论文未展开解释这一反直觉现象。
6. **部分核心结论依赖自建基准**：RynnBrain-Object/Spatial/Grounding/Affordance/Trajectory/RynnBrain-Bench 等均为作者自行构建并标注,20.2%、25.1% 等亮点提升数字主要来自这些自建评测集,独立第三方基准上的验证相对有限。

## 五、评价与展望

RynnBrain 的核心贡献并非单一算法创新,而是把"统一物理接地输出空间 + 时空记忆 + 交织式空间推理"这一设计理念,在工业级规模（近 20M 样本、三档模型规模、Apache 2.0 全开源）下系统实现并跑通了从认知、定位、推理到规划/执行的完整闭环。相比 RoboBrain 2.0、Robix 等同期"具身大脑"工作,RynnBrain 的差异化优势在于：（1）把 affordance/area/trajectory 等定位原语纳入统一坐标 token 空间,使规划输出天然可被下游 VLA 消费,而非停留在纯文本指令；（2）Chain-of-Point 把推理链和空间证据显式绑定,在一定程度上缓解了纯文本 CoT 常见的幻觉问题,这与近期"visual grounding for reasoning"一类工作（如 grounded CoT、tool-augmented VLM 推理）的方向一致,但 RynnBrain 把这种交织机制系统性地铺到了预训练+SFT+RL 三阶段。

与 π0/π0.5 等专注动作生成的 VLA 路线相比,RynnBrain 明确走的是"先做强大脑、再挂接口"的分层路线：RynnBrain-VLA 只是四个下游变体之一,而且是在小规模数据上快速适配得到的,论文的证据（表 8 中 RSR 接近 1.0）支持其论点——强大的场景理解和定位预训练可以大幅降低下游动作模型对目标识别能力的要求,但这也意味着 RynnBrain-VLA 本身在动作生成的多样性、精细操作（如变形物体、柔顺控制）上还未经充分检验,其展示的任务仍局限于简单拾放。

开放问题包括：（1）Chain-of-Point 与 GRPO 的组合目前只覆盖了 affordance/area/trajectory 三类奖励可显式定义的任务,对于更长时程、奖励难以几何化定义的复杂操作任务（如多步骤装配）能否扩展尚不清楚；（2）MoE 在导航任务上不增反降的现象提示"具身大脑"类模型的架构选择可能需要针对任务类型做专门设计,而非直接套用通用语言模型的 MoE scaling 经验；（3）论文展示的规划-执行分层系统（RynnBrain-Plan + RynnBrain-VLA）仍然是显式的两阶段架构,是否可以进一步压缩为端到端统一自回归+扩散联合训练、同时保留物理接地推理的可解释性,是值得跟进的方向。整体而言,RynnBrain 提供了一套相当完整且可复现（开源代码、权重、评测集）的具身基础模型基线,为后续"具身大脑"研究提供了一个较高的起点。

## 参考

1. BAAI RoboBrain Team et al. *RoboBrain 2.0 Technical Report*. arXiv:2507.02029, 2025.
2. Fang, Huang et al. *Robix: A Unified Model for Robot Interaction, Reasoning and Planning*. arXiv:2509.01106, 2025.
3. Black, Brown, Driess, Esmail et al. *$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization*. 9th Annual Conference on Robot Learning (CoRL), 2025.
4. Bai, Cai, Chen et al. *Qwen3-VL Technical Report*. arXiv:2511.21631, 2025.
5. Hao, Zhou, Huang et al. *MiMo-Embodied: X-Embodied Foundation Model Technical Report*. arXiv:2511.16518, 2025.
