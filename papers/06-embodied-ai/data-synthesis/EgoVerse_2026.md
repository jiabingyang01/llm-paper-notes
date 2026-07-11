# EgoVerse：面向机器人学习的全球众源第一人称人类数据集

> **论文**：*EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World*
>
> **作者**：Ryan Punamiya\*, Simar Kareer\*（共同一作 / 项目负责人），Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Davide Liconti … Marc Pollefeys, Robert Katzschmann, Xiaolong Wang, Shuran Song, Judy Hoffman, Danfei Xu et al.
>
> **机构**：Georgia Institute of Technology、Stanford University、University of California San Diego、ETH Zürich、MIT CSAIL、Meta Reality Labs Research、Mecka AI、Scale AI
>
> **发布时间**：2026 年 4 月（arXiv 2604.07607，v2 于 2026 年 7 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.07607) | [PDF](https://arxiv.org/pdf/2604.07607)
>
> **分类标签**：`人类演示数据集` `human-to-robot transfer` `跨本体协同训练` `flow matching`

---

## 一句话总结

EgoVerse 把学术实验室（Project Aria 眼镜）、工业伙伴（定制头戴相机）与大众（手机头戴）三类来源的第一人称人类操作演示,通过云端 EgoDB 系统汇聚成一个可持续生长的"活体数据集"（1,362 小时 / 79,692 episodes / 1,965 任务 / 240 场景 / 2,087 名演示者）,并首次在跨实验室、跨三种机器人本体的标准化协议下系统验证了**人类数据协同训练可使机器人策略在域内/域外泛化上最高提升约 30%**,同时给出一条关键规律:**只有存在少量"域对齐"人类数据作为锚点,多样人类数据的正向 scaling 才会出现**。

## 一、问题与动机

机器人学习越来越依赖大规模、高多样性数据,但真机遥操作数据采集昂贵、需专用硬件与专家操作,难以规模化。第一人称（egocentric）人类演示是一条有希望的替代路径:人类每天在多样环境中自然完成大量操作行为,数据量远超机器人自采。但现有人类数据集存在三大缺陷:

- **规模受限、难扩展**:多为一次性、静态发布,面向特定研究,无法持续增长;
- **缺乏机器人相关标注**:如精确 3D 手部姿态、相机标定,难以转成可执行的机器人轨迹;
- **碎片化**:分散在各机构,格式不统一,无法系统研究"人到机器人迁移"。

作者提出两个尚未解决的核心问题:(1) 有效的 human-to-robot transfer 到底依赖什么?(embodiment gap、scaling 行为);(2) 如何构建一个能持续接入新贡献者、可复现研究的人类数据生态。EgoVerse 同时给出一个**数据集 + 平台 + 大规模研究**的三位一体答案。

## 二、核心方法

### 2.1 数据集构成:A（学术受控）+ I（工业开放）双分支

- **EgoVerse-A**（Academic）:多个学术实验室在**受控、标准化协议**下"镜像"采集,用于可复现研究与系统性消融。定义 6 个 **Flagship Tasks**（object-in-container、cup-on-saucer、bag-grocery、fold-clothes、scoop-granular、sort-utensils）,沿 **task / scenario / demonstrator** 三个轴组织多样性:每任务 8–12 个场景、每场景 1–10 个 dataset units、约 40cm×60cm 工作区、每任务采样至多 30 个物体、每实验室 1–8 名演示者。轻量的 per-episode 标注(任务描述、场景 ID、主要物体、演示者元数据)。

- **EgoVerse-I**（Industry）:工业伙伴 in-the-wild 采集,强调**规模、任务多样性与稠密标注**,是当前**最大的带动作标注第一人称人类数据集**(约 1,400 小时 / 近 2,000 任务 / 240 场景 / 2,087 名演示者)。提供 1–2 秒粒度的稠密语言标注、active-hand 指示、静态/移动操作标志等,适合训练语言条件策略(如 VLA)。任务覆盖 Logistics / Cooking / Cleaning / Laundry / Hardware / Crafts / Gardening 等 1,500+ 开放类别。

统一标注:每帧估计双手 **21 关键点 3D 手姿**,并用 visual–inertial SLAM 恢复 **6-DoF 头姿**;学术端用 Aria 的 MPS 服务,工业端用厂商 SLAM + 基于模型的手姿估计。

### 2.2 EgoDB:可持续生长的数据管理系统

一个云端(S3 支撑)系统,支持异构人类/机器人数据持续接入并转成统一的、可直接训练的格式。上传时标注 operator/lab/task/embodiment/scene/objects,原始文件按 UTC 时间戳哈希入 S3;一个 Postgres SQL 表(schema 见 Table V)登记 episode 元数据,支持按 task/embodiment/scene/lab 结构化查询;夜间 Ray 处理守护进程(3 个集群分别负责 Aria MPS、训练格式转换、机器人数据转换)做标准化预处理/校验/索引;`EgoVerseDataset` 提供 PyTorch 接口,按 filter 从 S3 同步到本地缓存构建训练集。这套设计实现了"living dataset"——数据集随贡献者持续演化,而非静态快照。此外还提供**基于手机的采集管线**(iPhone 头戴、1080p/30FPS 超广角、云端恢复 6-DoF 头姿 + 21 关键点手姿),把采集门槛降到人人可及。

### 2.3 人到机器人迁移研究:表示、对齐与架构

**人类动作表示**。手部轨迹以移动相机为参考不稳定,故构造**以相机为中心的稳定参考系**:把未来手部位置投影到第 $t$ 帧的设备坐标系下,得到动作

$$a^H_{t:t+k} = \left[\left(T^{\text{device}}_t\right)^{-1} \cdot T^{\text{device}}_{t+i} \cdot p^H_{t+i}\right]_{i=1}^{k}$$

用大白话说:把"接下来手要去哪儿"全部换算成"相对我现在头（相机）的位置",这样即便脑袋在动,动作标签也稳定、可跨人跨机器人对齐。

**人机数据对齐**。采用对离群点鲁棒的 **quantile normalization**:把特征分布的第 1 与第 99 百分位映射到 $[-1,1]$:

$$\hat{x} = 2 \cdot \left(\frac{x - q_{0.01}}{q_{0.99} - q_{0.01}}\right) - 1$$

用大白话说:不用最大最小值(容易被极端值带偏),而用"掐头去尾 1%"的分位数来做归一化,让人和不同机器人的本体感/动作落到同一量纲。训练时还加随机裁剪与色彩抖动以适配不同相机。

**策略架构**（跨本体 encoder–decoder,见原文 Fig.7）。模态专属浅层 stem:图像用 ResNet-18(取 global-pool 前的 $7\times7\times512$),本体感用 MLP;每个 stem 交叉注意出 $L{=}16$ 个 query token。前置 $M{=}64$ 个可学习 context token,送入跨本体 Transformer 编码器 $f_\phi$($N_{\text{enc}}{=}16$ 块、$d{=}256$)。动作解码器 $\pi_\theta$ 是 **flow matching** 多块 Transformer($N_{\text{dec}}{=}6$),初始化 $T$ 个 learnable token,时间步 $\tau\sim\text{Beta}(1.5,1.0)$。

**训练目标**:BC 协同训练损失,在聚合的人类集 $D_H$ 与机器人集 $D_R$ 上做标准行为克隆:

$$\mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) = \mathbb{E}_{(o,a)\sim D_H \cup D_R}\left[\mathcal{L}_{\text{BC}}\!\left(\pi_\theta(f_\phi(o)), a\right)\right]$$

实际按本体分别算 conditional flow matching(CFM)损失并相加:

$$\mathcal{L}_{\text{BC-cotrain}} = \mathcal{L}^{\text{robot}}_{\text{CFM}} + \mathcal{L}^{\text{human}}_{\text{CFM}}$$

$$\mathcal{L}^e_{\text{CFM}} = \mathbb{E}_{\tau, a_0, a_1, s}\left[\left\|\pi_\theta(x_\tau, \tau, f_\phi(s)) - (a_0 - a_1)\right\|^2\right],\quad x_\tau = \tau a_0 + (1-\tau)\,a_1$$

用大白话说:让网络学会"从纯噪声 $a_0$ 一路流向真实动作 $a_1$ 的速度场";人类与机器人两路各算一份 CFM 损失合在一起,就是"一边看人类怎么做、一边看机器人怎么做"联合训练。训练 150k 步、batch 32–64、LR $1\times10^{-4}$、人:机 = 1:1。

**三种机器人本体**(跨实验室共享协议):Robot A(两台 6-DoF ARX5 平行夹爪,竖直安装,Aria 头相机 + 腕部 RealSense D405);Robot B(两台 ARX5 侧装在 3D 打印"类人肩"结构上,头 Aria + 腕 Logitech);Robot C(Unitree G1 + 7-DoF 臂 + 6-DoF Inspire 灵巧手,ZED 2 立体相机)。

### 2.4 受控多样性研究

为隔离"场景多样性"与"演示者多样性"的独立作用,单实验室用固定 **16 演示者 × 16 场景** 池采集 cup-on-saucer 与 fold-clothes,用离线 **Avg-MSE** 作为泛化代理指标:

$$\text{Avg-MSE}(\hat{a}_{1:T}, a_{1:T}) = \frac{1}{T}\sum_{t=1}^{T}\frac{1}{D}\left\|\hat{a}_t - a_t\right\|_2^2$$

设三种缩放:单场景演示者缩放(1→16,固定 2h)、多场景交互(4–12 演示者 × 1–8 场景,固定 8h)、场景多样性缩放(1→16 场景,固定演示者池)。

## 三、实验结果

### 3.1 数据集规模(Table II)

| 组成部分 | 占比 | 时长 (h) | Episodes | 任务数 |
|---|---|---|---|---|
| EgoVerse-A（全体伙伴） | 5.5% | 75 | 2,385 | 6 |
| EgoVerse-I partner A | 76.1% | 1,035 | 72,993 | 1,898 |
| EgoVerse-I partner B | 18.4% | 250 | 3,128 | 45 |
| **合计** | — | **1,362** | **79,692** | **1,965** |

EgoVerse-I 任务类别分布(Table III):Logistics 15.4%/209h、Cooking 13.7%/186h、Cleaning 11.6%/158h、Laundry 10.9%/148h、Hardware 6.8%/92h、Crafts 4.0%/54h、Gardening 3.2%/44h。

### 3.2 协同训练是否提升机器人表现(Fig.9–10)

在 object-in-container(单臂)、cup-on-saucer(精细双手)、bag-grocery(长程双手)三个 Flagship 任务、每任务 20 域内 + 20 域外 rollout 下:

| 数据配置 | 关键结论 |
|---|---|
| Robot-only vs 协同训练（EV 8h + ID 2h） | 域内(ID)与域外(OOD)**均一致提升,最高约 30% 相对增益**,且在 3 台机器人、多数任务上稳健——作者称这是首次在标准化跨实验室、多机器人下验证 |
| 仅 EV(8h)、无 domain-aligned 数据 | **不足以** 带来显著提升(无论 ID 还是 OOD) |
| ID(2h) 对齐数据 + 多样 EV 数据 | 对齐数据"锚定"学习,使多样数据可正向 scaling:仅 2h 对齐数据即可促成从 2h 多样 EV 的迁移,并随 EV 增至 8h 持续增益 |
| Robot B, bag-grocery | **反而下降**:该本体受限,机器人开袋策略被迫偏离人类"单手撑袋、另一手放物"的策略 → 人机行为分布不一致,削弱跨本体对齐 |

一句话:人类数据能帮机器人,但**帮多少取决于人机数据在任务语义与场景上的对齐程度**;非对齐的多样数据单独不 work。

### 3.3 多样性如何影响泛化(离线 Avg-MSE,Fig.11 / 17 / 18)

| 缩放维度 | 结论 |
|---|---|
| 单场景 · 演示者 1→16（固定 2h） | Avg-MSE 单调下降(fold-clothes),演示者多样性提升对**未见演示者**的泛化;cup-on-saucer 低样本时略非单调,规模够大后改善 |
| 多场景 · 演示者 4→12（固定 8h） | 两任务均一致、明显改善 |
| 场景 1→16 | Avg-MSE 一致下降,**场景多样性是最可靠的泛化驱动**,低数据预算下尤为显著(数据密度饱和后,扩场景比堆密度更有效) |
| 4h 预算联合缩放 | 场景多样性一致降低 MSE;演示者收益随任务而异(fold-clothes 8 演示者 > 4,cup-on-saucer 反之) |

结论:**演示者多样性 → 对陌生人类形态的鲁棒性;场景多样性 → 对新环境泛化的主导因素**,两者互补但相对重要性随任务而变。

## 四、局限性

- **研究范式单一**:仅覆盖"人机协同训练"一种;pre-train + fine-tune 等其它算法路径未探索,而这恰是异构人类数据最有前景的用法之一。
- **多样性结论依赖离线代理指标**:场景/演示者多样性的核心结论(Sec. IV-F)全部基于离线 Avg-MSE,并**未做真机 rollout 验证**,该指标能否直接对应下游机器人成功率仍需实验支撑。
- **对齐要求偏强**:核心规律指出"没有域对齐锚点数据,多样数据不带来正向 scaling",意味着仍需为每个下游任务采一小批与机器人语义/场景匹配的人类数据,离"纯 in-the-wild 数据零成本迁移"仍有距离。
- **本体受限反噬**:Robot B 上 bag-grocery 的退化说明,当机器人硬件无法复现人类策略时,协同训练可能有害,方法未提供自动检测/缓解机制。

## 五、评价与展望

**优点**:(1) 把"数据集"升级为"活体数据生态 + 云平台(EgoDB)",从工程上解决了社区人类数据碎片化、难增长的痛点,并用手机采集把门槛降到大众级,这是超越单纯 releasing dataset 的贡献;(2) 研究"reproducible by design",在三种截然不同的机器人本体、多实验室共享协议下做对照,结论的系统鲁棒性显著强于以往单本体、单实验室的迁移研究(如 Egomimic、EgoBridge);(3) "域对齐数据作为锚点决定多样数据能否正向 scaling"是一个可证伪、可操作的规律,对如何配比人机数据有直接指导。

**与公开工作的关系**:数据侧继承 Ego4D / Ego-Exo4D / Epic-Kitchens 的第一人称传统,但补齐了机器人可执行标注(手姿 + 头姿 + 相机标定);方法侧的以相机为中心 SE(3) 手部动作表示、quantile 归一化对齐承自作者团队前作 Egomimic / EgoBridge / EMMA;跨本体 encoder–decoder + 浅层 modality stem 借鉴 HPT(异构预训练 Transformer),flow matching 解码器与 π0 一脉相承;规模化协同训练的动机则呼应 Open X-Embodiment / DROID 的多本体 scaling。相比之下 EgoVerse 的差异化在于"人类数据作为一等公民 + 平台化持续增长 + 跨本体对照研究"三者合一。

**开放问题与可能改进**:(1) 把多样性研究从离线 Avg-MSE 推进到真机成功率,验证 Avg-MSE 的预测效度;(2) 系统研究 pre-train→fine-tune 范式,量化异构人类数据在预训练阶段的收益;(3) 针对"本体无法复现人类策略"的场景,引入策略级对齐检测或 embodiment-aware 加权,避免负迁移;(4) 探索弱对齐/无对齐条件下如何用少量对齐监督"接地",降低每任务采对齐数据的成本;(5) 结合稠密语言标注(EgoVerse-I)训练语言条件 VLA,评估语言接地对跨本体迁移的增益。

## 参考

1. Kareer et al. *Egomimic: Scaling imitation learning via egocentric video.* arXiv:2410.24221, 2024.（作者团队前作,相机中心手姿表示的源头）
2. Punamiya et al. *EgoBridge: Domain adaptation for generalizable imitation from egocentric human data.* NeurIPS 2025.（人机域适配,直接前驱）
3. O'Neill et al. *Open X-Embodiment: Robotic learning datasets and RT-X models.* ICRA 2024.（多本体大规模 scaling 的对照参照）
4. Wang et al. *Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers (HPT).* arXiv:2409.20537, 2024.（跨本体 encoder–decoder 架构基础）
5. Black et al. *π0: A vision-language-action flow model for general robot control.* arXiv:2504.16054, 2025.（flow matching 动作解码器的方法学参照）
