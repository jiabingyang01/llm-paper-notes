# Sim-and-Real Co-Training：面向视觉机器人操作的仿真-真实协同训练简单配方

> **论文**：*Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation*
>
> **作者**：Abhiram Maddukuri, Zhenyu Jiang, Lawrence Yunliang Chen, Soroush Nasiriany（共同一作）+ Ajay Mandlekar, Linxi Fan, Yuke Zhu（项目负责人）et al.
>
> **机构**：UT Austin、NVIDIA、UC Berkeley、New York University
>
> **发布时间**：2025 年 03 月（arXiv 2503.24361，v2 于 2025-04-02）
>
> **发表状态**：未录用（预印本，项目页 co-training.github.io）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.24361) | [PDF](https://arxiv.org/pdf/2503.24361)
>
> **分类标签**：`Sim-and-Real Co-Training` `合成数据` `Diffusion Policy` `Digital Cousin` `MimicGen`

---

## 一句话总结

把小规模真机演示与用 MimicGen/DexMimicGen 自动合成的大规模仿真数据按一个 co-training 比例 $\alpha$ 混进同一个 batch 里联合训练视觉操作策略（不做任何 sim-to-real 迁移微调），就能在 Panda 机械臂与 GR-1 人形两个平台、六个任务上把真机成功率平均从 45.3% 提到 83.2%（相对提升约 38%），并系统总结出"仿真数据要多出真机几个数量级、co-training 比例要调、相机视角要大致对齐"这三条可操作配方。

## 一、问题与动机

- **真机数据太贵**：训练通用操作策略依赖大规模高质量真机演示，采集耗时耗力、扩展性差；单纯堆真机数据能否训出通用策略仍是未知。
- **纯仿真 sim-to-real 太难**：传统路线要么靠 domain randomization 精心调随机化范围，要么靠 system identification / digital twin 精确对齐物理与视觉，都需要大量人工，且难以泛化到多样任务。
- **一个更省事的替代方案**：直接把仿真数据与真机数据"混起来"协同训练（co-training）。近期若干工作（RoboCasa、Nasiriany 等）已初步显示这样能超过纯真机，且**不要求 sim 与 real 完美对齐**。但社区缺乏对该策略的系统理解——仿真数据要与真机"像"到什么程度？哪些数据组成因子必须对齐？混合比例怎么定？
- **本文目标**：给出一个简单可复现的配方，通过大量对照实验回答"要收获仿真数据的红利，到底需要做什么"。作者刻意跨两个差异极大的机器人本体（7-DoF 机械臂 + 灵巧手人形）、覆盖 pick-and-place、铰接物体、非抓取式（倒球）等多类任务来验证配方的普适性。

## 二、核心方法

**总体流程（三步）**：① 选定一个真机目标任务，采集数十条遥操作演示；② 针对该任务构造仿真环境并用自动数据生成工具把演示放大到成百上千倍；③ 将真机数据与仿真数据按比例混采、联合行为克隆训练一个策略，训完**直接部署到真机**，不再做迁移微调。

### 1. Co-training 目标函数

在真机演示集 $\mathcal{D}_{\text{real}} = \{\xi_i\}_{i=1}^{N}$ 与仿真演示集 $\mathcal{D}_{\text{sim}} = \{\xi_i\}_{i=1}^{M}$（通常 $M \gg N$）上，最小化加权行为克隆损失：

$$\mathcal{L}_{\text{total}}(\theta;\mathcal{D}_{\text{real}},\mathcal{D}_{\text{sim}}) = \alpha \cdot \mathcal{L}(\theta;\mathcal{D}_{\text{sim}}) + (1-\alpha)\cdot \mathcal{L}(\theta;\mathcal{D}_{\text{real}})$$

其中单集损失是标准负对数似然 $\mathcal{L}(\theta;\mathcal{D}) = \frac{1}{|\mathcal{D}|}\sum_{(o_i,a_i)\in\mathcal{D}} -\log \pi_\theta(a_i \mid o_i)$，$\alpha \in [0,1]$ 是 **co-training ratio**。

**用大白话说**：就是把"学仿真"和"学真机"两个损失按 $\alpha$ 和 $1-\alpha$ 加权求和。$\alpha$ 越大，策略越"听仿真的话"。

**工程等价实现**：不真的算两个损失，而是把 $\alpha$ 解释为"每个 batch 里一条样本来自仿真的概率"——即 $P[(o_i,a_i)\in\mathcal{D}_{\text{sim}}]=\alpha$、$P[(o_i,a_i)\in\mathcal{D}_{\text{real}}]=1-\alpha$，通过先按数据集大小归一化权重、再对 sim 样本乘 $\alpha$、对 real 样本乘 $1-\alpha$ 来实现重加权采样。**大白话**：装满一个训练 batch 时，每抓一条数据,有 $\alpha$ 的概率去仿真桶里抓、$1-\alpha$ 的概率去真机桶里抓。

### 2. 两种仿真数据（对齐程度不同）

作者把数据集拆成一组 **data composition factors**（任务组成、场景组成、物体组成、初始状态分布、相机参数、动力学参数），并据此定义两档仿真数据：

- **Task-aware digital cousin（DC，任务感知数字表亲）**：本文对 Dai 等人 [25] "digital cousin" 概念给了更精确定义——一个 DC 数据集须保留真机任务的四要素：① 相同机器人与动作空间；② 相同任务目标（相同成功判据、相同语言指令）；③ 相同物体类别（具体实例几何/纹理可不同）；④ 相同环境 fixture 类别。用 MimicGen/DexMimicGen 从数十条源演示放大：Panda 每任务生成 10k 条、GR-1 每任务生成 1k 条。关键还对 DC 的仿真相机做了后处理，重渲染以**近似对齐**真机相机位姿。
- **Task-agnostic prior simulation（Prior，任务无关先验仿真）**：在目标任务出现之前就存在的、开箱即用的大规模多任务仿真数据集。Panda 用 RoboCasa（去掉抽屉/灶台旋钮任务后 60k 条 / 20 任务），GR-1 用作者在 RoboCasa 里搭的 10 任务人形先验集（10k 条）。这些数据与真机在物体类别、纹理、干扰物、物理参数、机器人基座位置上都有明显差异，**语义上没有一个任务与真机任务相同**。

**用大白话说**：DC 是"照着真机任务量身定制、只是渲染和实例还不完美"的仿；Prior 是"网上现成的一大堆别的仿真任务"。前者对齐度高、后者对齐度低但量更大、更省事。

### 3. 数据生成与策略

- **数据放大**：MimicGen（机械臂）/ DexMimicGen（双臂/灵巧手人形）把源演示按物体为中心切段、做线性变换后重新拼接成新轨迹，从数十条人类演示放大几个数量级。
- **策略网络**：均用 Diffusion Policy（Chi 等 [37]）。Panda 用 transformer + ResNet 视觉编码器，输入 3 路 $128\times170$ 图像 + 本体感知，输出 7-DoF delta 末端控制 + 夹爪；GR-1 用 UMI 的 Transformer 视觉编码器 + UNet 扩散骨干，输入第一人称 RGB + 关节位置，输出双臂+灵巧手关节目标。GR-1 因仿真数据远多于真机，用 CLIP + FiLM 做语言条件，默认真机 $\alpha$ 权重 0.10、仿真 0.90。

## 三、实验结果

**主表（Table I，六任务真机成功率）**：C2SPnP=CounterToSinkPnP，C2CPnP=CounterToCabPnP。

| 数据组成 | C2SPnP | C2CPnP | CloseDoor | CupPnP | MilkPnP | Pouring | 平均 |
|---|---|---|---|---|---|---|---|
| Real（纯真机） | 44% | 38% | 10% | 65% | 50% | 65% | 45.3% |
| Real + DC | 67% | 72% | 100% | 95% | 70% | 85% | 81.1% |
| Real + Prior | 58% | 53% | 100% | 80% | 80% | 70% | 76.8% |
| **Real + DC + Prior** | **72%** | 72% | 100% | 85% | 80% | 90% | **83.2%** |

要点：① 加 DC 平均相对纯真机 +35.8%；② 令人意外的是，**完全不针对真机定制的 Prior 数据也能 +31.5%**；③ DC+Prior 全上最好，平均 +37.9%（摘要口径约 +38%）；④ CloseDoor 上纯真机仅 10%，任何一档协同训练直接到 100%，差距悬殊。

**泛化到未见物体 / 未见位置（Table II）**：

| 数据组成 | 未见物体·Panda | 未见物体·GR-1 | 未见位置·Panda | 未见位置·GR-1 |
|---|---|---|---|---|
| Real | 33% | 10% | 11% | 43% |
| Real + DC | 50% | 80% | 28% | 100% |

仿真里的物体多样性与初始位置随机化，能显著提升真机对新物体/新摆位的鲁棒性（多处约翻倍）。

**数据充裕时仍有效（Fig 4，MultiTaskPnP，固定 4000 条 DC，变真机条数）**：

| 真机演示数 | 40 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| 协同训练 | 0.38 | 0.68 | 0.71 | 0.78 | 0.89 |
| 纯真机 | 0.17 | 0.27 | 0.31 | 0.31 | 0.39 |

即便真机演示加到 400 条，协同训练依旧稳定领先——sim-and-real co-training 在数据充裕场景仍有增益。

**Co-training 比例的影响（Fig 5，CupPnP，20 真机 + 1000 DC）**：

| $\alpha$（采样仿真概率） | 10% | 30% | 90% | 99% | 99.5% | 99.9% |
|---|---|---|---|---|---|---|
| 成功率 | 0.60 | 0.75 | 0.95 | **0.95** | 0.80 | 0.60 |

**1:1（50%）并非最优，最佳约 99%**——即仿真样本要占绝大多数；但比例过高（99.9%）会因真机信号被淹没而崩到 0.6。

**其余关键消融**：
- **仿真量要足**：Panda C2SPnP 的 DC 从 10k 减到 500，成功率 67%→53%；GR-1 CupPnP 的 DC 从 1k 减到 100，95%→75%。
- **相机对齐关键**：用未对齐（默认）相机的 DC，Panda C2SPnP 67%→56%、GR-1 CupPnP 95%→70%；但对齐无需完美（人形真机是鱼眼、仿真未建模畸变仍有效）。
- **视觉真实感（Vid2Vid，Table VI）**：微调 CogVideo-X 把 DC 渲染变逼真，仅在低数据区显著（Sim 100 / Real 1000：40%→53%），真机数据充足时 sim 逼真度作用很小（Sim 1000：95%→95%）。
- **双臂长程仍难（FAQ，BimanualPnP）**：50 真机=15%，加 1000 DC=50%，100 真机=30%；而任务无关的单臂 prior 数据会让双臂策略退化成单臂行为、近乎 0——**Prior 有用的前提是行为模式与真机一致**。
- **差距不只是数量（Appendix L）**：CloseDoor 把真机演示翻倍到 100 条，纯真机也只到 80%、达不到协同训练的 100%。

## 四、局限性

- 任务主要集中在 pick-and-place（及少量铰接门、倒球），未覆盖高精度插入、长程任务；作者明言留待未来。
- 即便协同训练，策略也未做到完美（如 CloseDoor 纯真机翻倍仍 80%，双臂任务上限偏低）。
- **可形变物体与液体难以精确仿真**，从根本上限制了此类任务上仿真数据的可用性；作者建议未来用视频生成模型 / 世界模型产出协同训练数据来弥补这一渲染/物理鸿沟。
- 结论多为经验规律（"多出几个数量级""99% 比例"），缺乏对最优 $\alpha$ 与对齐程度的理论刻画；不同本体/任务上的最优超参需重新搜。
- 未做动力学对齐消融的正向结论（GR-1 上调物理参数对开环 rollout 无差别、称对本文任务无必要），但这仅在其测试任务成立，未必普适。

## 五、评价与展望

**优点**：
- **系统性强、可操作**：不像多数论文只报一个 SOTA 数字，本文把"仿真数据组成"拆成六个因子逐一对照，最终凝练成三条工程师能直接照做的配方（仿真量级、混合比例、相机对齐），实用价值高。
- **跨本体验证**：同一配方在 Franka 机械臂与 Fourier GR-1 灵巧手人形上都成立，且 Prior 开箱即用数据也有效，说明结论不是过拟合到单一平台。
- **"完美对齐不必要"这一负向结论有价值**：既否定了昂贵的 digital twin / system identification 路线，又给出"近似对齐 DC + 大量 Prior"这条低成本路径。

**缺点与开放问题**：
- **本质是数据配比工程**，方法层面创新有限——loss 就是加权 BC，数据靠现成的 MimicGen/DexMimicGen/RoboCasa。真正的贡献在实证与配方，而非新算法。
- **"为什么 Prior 有效"缺机理解释**：任务语义完全不同的仿真数据竟能 +31.5%，作者归因于视觉/动作多样性带来的正则化，但未做表征层面的分析（如是否主要帮了视觉编码器）。
- **最优 $\alpha$ 高度依赖数据量比**：99% 这个数字与"1000 仿真 : 20 真机"强绑定，实际部署每换一个数据规模都要重扫比例，缺一个自适应/理论指导。
- 与其它公开工作的关系：与 RoboCasa [7]、Open X-Embodiment、"Re-mix" [58]（数据混合比例优化）、"What matters in learning from…" [62] 等数据组成研究互补——后者更偏大规模真机混合，本文专注 sim↔real 两桶混合并给出相机对齐这一独有维度。

**可能的改进方向**：① 用自适应 / 课程式 $\alpha$ 调度替代人工扫比例；② 把 Vid2Vid 的思路推进到直接用视频/世界模型生成协同训练数据，绕开图形渲染器对可形变物体和液体的短板（作者自己也点到）；③ 扩展到高精度插入与长程双臂任务，检验配方在接触丰富场景是否仍成立；④ 与 VLA 大模型结合，看 co-training 是否能替代/补充其预训练语料。

## 参考

1. Mandlekar et al. *MimicGen: A data generation system for scalable robot learning using human demonstrations.* CoRL 2023.（本文放大机械臂仿真数据的核心工具 [9]）
2. Jiang et al. *DexMimicGen: Automated data generation for bimanual dexterous manipulation.* ICRA 2025.（人形/双臂数据放大 [10]）
3. Nasiriany et al. *RoboCasa: Large-scale simulation of everyday tasks for generalist robots.* RSS 2024.（Prior 任务无关仿真数据来源 [7]）
4. Dai et al. *Automated creation of digital cousins for robust policy learning.* arXiv 2410.07408, 2024.（"digital cousin" 概念出处 [25]）
5. Chi et al. *Diffusion Policy: Visuomotor policy learning via action diffusion.* RSS 2023.（本文所用策略骨干 [37]）
