# MotoVLA：超越动作标注数据的通用机器人操作

> **论文**：*Generalist Robot Manipulation beyond Action Labeled Data*
>
> **作者**：Alexander Spiridonov, Jan-Nico Zaech, Nikolay Nikolov, Luc Van Gool, Danda Pani Paudel
>
> **机构**：INSAIT, Sofia University "St. Kliment Ohridski"（保加利亚）；ETH Zurich（瑞士）
>
> **发布时间**：2025 年 09 月（arXiv 2509.19958）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.19958) | [PDF](https://arxiv.org/pdf/2509.19958)
>
> **分类标签**：`VLA` `无动作视频学习` `动态点云` `跨本体迁移`

---

## 一句话总结

MotoVLA 提出用**手/夹爪处的稠密动态 3D 点云** 作为跨本体、无关动作标签的中间表示：第一阶段在 137k 条无动作标注的人类+机器人视频上自监督预测未来点云序列以学习细粒度运动先验,第二阶段仅用 38k 条带动作标签数据把点云表示对齐到机器人动作,从而实现**从无动作标注视频(甚至人类演示)直接学新任务** 的 out-of-action 泛化;在 SIMPLER 域内任务上平均成功率 68.2%,超过 π0 基线 11.4 个百分点。

## 一、问题与动机

通用机器人操作(generalist manipulation)当前主要靠在大规模遥操作(teleoperation)机器人演示上训练 VLA 模型。但这一路线有两个瓶颈:

1. **数据瓶颈**:高质量、带动作标签的机器人演示极其昂贵,而模型一旦任务超出训练分布性能就急剧下降。
2. **人类视频难以直接利用**:互联网规模的人类操作视频包含丰富时空信息、任务和环境多样,却没有动作标签、存在人-机本体域差(human-to-robot domain gap),且含大量与控制无关的冗余/干扰特征。

已有工作要么停留在小规模专用策略(specialist policy),要么只做隐式视觉表征、或预测目标状态图/光流后再接一个专门的逆动力学模型(inverse-dynamics model)执行,要么把人手 retarget 到机器人夹爪但域差很大。作者指出:据其所知,尚无端到端通用 VLA 能直接从大规模无标注演示中学到**细粒度运动先验**。MotoVLA 的核心动机是找到一种既能编码运动的空间与时间关系、又"embodiment-agnostic(本体无关)"、还天然与 3D 机器人动作对应的表示——即**动态点云**——来打通这条链路。

## 二、核心方法

### 2.1 两类数据与两个映射目标

**无标注视频(阶段一)**:数据集 $\mathcal{T}_o = \langle (\tau_o^{(i)}, l^{(i)}) \rangle_{i=1}^{N_o}$,其中 $l^{(i)}$ 是语言描述,$\tau_o^{(i)} = \langle \mathbf{I}_t^{(i)} \rangle_{t=1}^{T}$ 是相机 RGB 图像序列。作者从中提取每帧的**动态点云** $\mathbf{p}_t \in \mathbb{R}_{n\times 3}$($n$ 个一致的点,代表相机坐标系下的手或夹爪)。VLA 学习的映射是"给定当前图像、语言和过去点云,预测未来点云序列":

$$
\mathbf{f}_\theta^{points}(\mathbf{I}_t^{(i)}, l^{(i)}, \mathbf{p}_{t-h:t}^{(i)}) \rightarrow \mathbf{p}_{t:t+c}^{(i)}
$$

记 $\mathbf{P}_t = \mathbf{p}_{t:t+c}$。

**带动作标签演示(阶段二)**:较小数据集 $\mathcal{T}_a$,含 RGB 图像 $\mathbf{I}_t$ 与机器人本体感知 $\mathbf{q}_t$。第二阶段把 VLA 微调为直接预测机器人动作:

$$
\mathbf{f}_\theta^{act}(\mathbf{I}_t^{(i)}, l^{(i)}, \mathbf{q}_{t-h:t}^{(i)}) \rightarrow \mathbf{q}_{t:t+c}^{(i)}
$$

记 $\mathbf{A}_t = \mathbf{q}_{t:t+c}$。

**用大白话说**:第一阶段让模型学"手/夹爪接下来会怎么在 3D 空间里动"(不需要知道具体关节指令);第二阶段只是做一次"手眼标定"式的轻量对齐,把已经学好的点云动态换算成本机器人能执行的末端指令。

### 2.2 动态点云的提取(关键工程)

从无标注视频里"造"出监督信号的流水线:

1. 采样帧索引 $t \sim \mathcal{N}(T/2, 0.4\cdot T)$;
2. 用 **Grounding DINO** 检测手/夹爪得到 bounding box,送入 **SAM 2** 得到分割掩码;
3. 在掩码内均匀采样像素点,用 **BootsTAPIR** 前向+后向跟踪(2D 轨迹);
4. 用仿射不变单目深度估计器 **MoGE** 把 2D 轨迹连同预测深度与相机内参提升(lift)到 3D,得到手/夹爪点云序列。

**用大白话说**:没有真值动作,那就用现成的检测+跟踪+单目深度模型,把视频里手/夹爪的运动"抠"成一串会动的 3D 点,这串点既描述了运动、又与末端执行器动作强相关,所以第二阶段对齐才特别容易。

### 2.3 架构:Mixture-of-Transformers

- **VLM 主干**:Paligemma(3B,SigLIP 视觉编码器 + Gemma 语言模型),负责视觉-语言语义推理;前向一次返回 KV cache $\mathbf{h}_t$ 供 Predictor 复用,加速推理。
- **Predictor**:一个更小的 Transformer(约 300M,Gemma 架构,特征维降到 1024、注意力内 MLP 维 4096),通过 self-attention 关注 VLM 隐藏特征,并用 **flow matching(条件流匹配)** 做预测。阶段一它是 **3D Dynamics Predictor**,阶段二被初始化并调成 **Action Predictor**(两者结构完全相同,权重从阶段一继承,只有线性编解码器随机初始化)。
- 注意力模式:Paligemma token 内部双向注意但看不到 Predictor token;Predictor token 可注意所有其他 token(唯一例外:当前点云/本体感知 token 不注意噪声 token)。

### 2.4 两个损失(均为流匹配)

点云预测损失:

$$
\mathcal{L}_{points} = \mathbb{E}_{p(\mathbf{P}_t\mid\mathbf{o}_t),\epsilon,\tau}\|\mathbf{v}_\theta^{points}(\mathbf{P}_t^\tau, \mathbf{o}_t) - (\mathbf{P}_t - (1-\sigma_{min})\epsilon)\|^2
$$

其中 $\mathbf{P}_t^\tau = (1-(1-\sigma_{min})\tau)\epsilon + \tau\mathbf{P}_t$,$\epsilon\sim\mathcal{N}(0,\mathbf{I})$,条件 $\mathbf{o}_t = [\mathbf{h}_t, \mathbf{p}_{t-1}]$;时间采样沿用 π0 的 $\tau = (1-\sigma_{min})(1-z)$,$z\sim\text{Beta}(1.5,1)$。

动作预测损失结构完全对称,只是把点云换成动作 $\mathbf{A}_t$、条件换成 $\mathbf{o}_t = [\mathbf{h}_t, \mathbf{q}_{t-1}]$:

$$
\mathcal{L}_{action} = \mathbb{E}_{p(\mathbf{A}_t\mid\mathbf{o}_t),\epsilon,\tau}\|\mathbf{v}_\theta^{act}(\mathbf{A}_t^\tau, \mathbf{o}_t) - (\mathbf{A}_t - (1-\sigma_{min})\epsilon)\|^2
$$

**用大白话说**:两阶段用的是同一套"从噪声流向目标"的生成式回归公式,只是回归对象从"未来点云"换成"未来动作",所以阶段二天然继承阶段一学到的运动结构。

### 2.5 训练数据与配置

- **阶段一(动态点云训练)**:RH20T(人类)+ BridgeData V2(机器人)+ RT-1(机器人),混合权重均为 1.0;机器人数据的动作标签在此阶段**被丢弃**,只用其视觉。共约 137k 条 episode。冻结视觉编码器以保留开放词汇视觉能力。
- **阶段二(动作对齐)**:仅用 BridgeData V2 的动作(约 38k 条);此阶段视觉编码器可训练(训练时长短、影响小)。
- 硬件:TPUv5e-256,阶段一/二分别训练 15 小时 / 4 小时。
- 点云序列长度 4、每帧采 200 个点、预测点云位置的变化量;末端执行器控制(delta 笛卡尔位置 + delta 欧拉角 + 夹爪开合);推理用 Euler 积分 $\Delta t = 0.1$。

## 三、实验结果

评测机器人:WidowX 250S;域内在 SIMPLER 仿真、域外在真机。基线含 OpenVLA(OXE 全量训练)、LAPA(OXE,潜动作预训练)、π0(B)(在 BridgeData V2 从头训,验证点云预训练增益的对照)、ATM(B)(层级式 2D 轨迹方法,改造到零样本设定)。**MotoVLA (R)** = 只用机器人数据(无 RH20T 人类数据)的消融版;**MotoVLA (R+H)** = 完整版。

### 3.1 域内(SIMPLER,4 任务 × 24 配置)

| 策略 | Put carrot on plate | Put eggplant in basket | Put spoon on towel | Put gray on yellow block | 平均 |
| --- | --- | --- | --- | --- | --- |
| ATM (B) | 16.6 | 43.8 | 18.8 | 4.1 | 20.8 |
| LAPA (OXE) | 37.5 | 50.0 | 70.8 | 58.3 | 54.1 |
| π0 (B) | 39.6 | 83.3 | 72.9 | 31.2 | 56.8 |
| MotoVLA (R) | **75.0** | **100.0** | **75.0** | 12.5 | 65.6 |
| MotoVLA (R+H) | 54.1 | 97.9 | 72.9 | 47.9 | **68.2** |

要点:完整版平均 **68.2%**,超 LAPA 14.1 个百分点(尽管 LAPA 在整个 Open X-Embodiment 上预训练),超 π0(B)11.4 个百分点——说明"跨域跨本体点云运动先验"能提升即便带动作监督的下游任务。R+H 相比 R 的提升主要来自堆叠任务(Put gray on yellow block:12.5→47.9),因为该动作在人类演示数据中有充分覆盖。层级式 ATM(B)在通用设定下明显最差。

### 3.2 域外(真机 WidowX 250S,8 任务 × 10 配置 × 3 次)

- 完整版 MotoVLA (R+H) 取得**最佳平均成功率**(约 47.9%);比 OpenVLA 高约 9.8 个百分点,而 OpenVLA 用了更多机器人数据——凸显任务专属人类演示的价值(比等量机器人数据更易采集)。
- 最大增益来自 **Push Button / Cube on Scale / Cable in Basket / Clamp in Cup**,这些任务恰好出现在阶段一的人类演示中,说明方法能把无标注数据里的技能直接迁移到未见物体与任务上(跨本体)。
- 对**完全域外**(阶段一、二均未见)的任务(如 Push USB Stick in Pot),反而是 MotoVLA (R) 最好、π0(B)紧随其后——这类任务仍然困难。

### 3.3 消融(SIMPLER 域内平均)

| 变体 | 平均成功率 |
| --- | --- |
| MotoVLA Separation(两个独立模型) | 46.9 |
| MotoVLA Co-Training(点云+动作并行预测) | 47.9 |
| MotoVLA Grid(均匀采点而非采夹爪) | 54.7 |
| MotoVLA 32 Points(32 点而非 200 点) | 60.9 |
| MotoVLA (R+H) 2D(用 2D 轨迹替代 3D 点云) | 64.2 |
| **MotoVLA (R+H)** | **68.2** |

结论:①3D 点云优于 2D 轨迹(SIMPLER 高 4.0%、真机高 12.5%),因为 3D 与末端动作域差更小;②点数与采样位置很关键(200 点、且采在夹爪上远好于 32 点或均匀网格);③把点云与动作放同一 Predictor 里"共训练"或干脆"完全分离"都会掉点——阶段式初始化(先学点云、再对齐动作)才是最优。

## 四、局限性

1. **人类演示分布偏窄**:RH20T 是静态第三人称相机、静态背景,已经很接近机器人域;而真正互联网规模的人类数据(EPIC-Kitchens、Ego4D)是**头戴式自我中心(egocentric)非静态相机**,要提取静态相机系下的点云序列还需额外估计相机外参(如用 MonST3R),这一步尚未验证。
2. **跟踪失败**:人手快速运动时 TAP(BootsTAPIR)会跟丢查询点。
3. **深度估计瓶颈**:单目深度估计器在多样背景下的几何一致性仍具挑战,直接限制点云质量。
4. 方法的进一步提升高度依赖自我中心视频理解与快速运动记录等"本文范围之外"的上游能力改进。

## 五、评价与展望

**优点**:
- **表示选择精准**:动态点云同时满足"本体无关、可解释、编码时空运动、且与 3D 末端动作天然对应"四点,这使第二阶段对齐几乎退化为一次隐式手眼标定,数据效率极高(38k 标注即可)。这比 LAPA 的潜动作(latent action,不可解释、需较多数据)和 ATM 的 2D 轨迹(域差更大)都更有针对性,消融数据(2D vs 3D 掉 12.5%)直接佐证了这一点。
- **端到端且规模化**:相比同期只预测目标状态图(如 3D-VLA、SuSIE)或需外挂逆动力学模型的层级方法,MotoVLA 把运动先验直接注入一个端到端通用 VLA,并首次把这类"无动作数据学习"扩展到通用(而非专用)策略规模。
- **对照实验干净**:R vs R+H 隔离了人类数据的贡献,π0(B)从头训作为同架构对照,较有说服力地把增益归因到点云预训练本身而非模型容量。

**缺点与开放问题**:
- **管线依赖多个现成模型**(Grounding DINO + SAM 2 + BootsTAPIR + MoGE),任一环节(尤其单目深度、快速运动跟踪)出错都会污染监督信号,鲁棒性和可复现性存疑;真正的互联网 egocentric 数据尚未跑通,当前"人类数据"其实与机器人域很接近,泛化声明打了折扣。
- **完全域外任务仍弱**:阶段一、二都没见过的任务上完整版并不占优,说明点云先验主要在"技能已被无标注数据覆盖"时起作用,对真正的零样本组合泛化帮助有限。
- **仅在 WidowX 单一本体、末端控制下验证**,跨本体(不同自由度、灵巧手)对齐是否仍只需轻量微调尚待检验;点云只表征手/夹爪几何,未显式建模接触力/抓取物,精细接触任务可能受限。
- **可能的改进方向**:引入 MonST3R 式动态外参估计以吃进 Ego4D 级数据;把被跟踪物体点云一并纳入以显式建模物体-夹爪相对运动;用更强的时空一致点跟踪/深度模型或多视图约束提升点云质量;探索点云表示与力/触觉模态的联合预训练。

## 参考

1. Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.(流匹配动作专家与时间采样的直接来源)
2. Ye, Jang et al., *LAPA: Latent Action Pretraining from Videos*, ICLR 2025.(无标注视频学潜动作的主要对照)
3. Wen et al., *ATM: Any-point Trajectory Modeling for Policy Learning*, arXiv:2401.00025, 2024.(2D 轨迹层级式基线)
4. Zhen et al., *3D-VLA: A 3D Vision-Language-Action Generative World Model*, ICML 2024.(引入非动作视频但只预测目标状态的代表工作)
5. Wang et al., *MoGE: Unlocking Accurate Monocular Geometry Estimation for Open-domain Images*, arXiv:2410.19115, 2024.(点云 lift 到 3D 所用单目几何估计器)
