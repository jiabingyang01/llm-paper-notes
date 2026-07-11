# METIS：面向一体化灵巧操作视觉-语言-动作模型的多源自视角训练

> **论文**：*METIS: Multi-Source Egocentric Training for Integrated Dexterous Vision-Language-Action Model*
>
> **作者**：Yankai Fu, Ning Chen, Junkai Zhao, Shaozhe Shan, Guocai Yao, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang（Yankai Fu 与 Ning Chen 共同一作,Junkai Zhao 为 project leader,Shanghang Zhang 为通讯作者）
>
> **机构**：北京大学计算机学院多媒体信息处理国家重点实验室；北京智源人工智能研究院（BAAI）
>
> **发布时间**：2025 年 11 月（arXiv 2511.17366）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.17366) | [PDF](https://arxiv.org/pdf/2511.17366)
>
> **分类标签**：`灵巧操作` `VLA` `多源自视角数据集` `motion-aware dynamics` `人类数据预训练`

---

## 一句话总结

把 8 个异构自视角数据源(人手动捕、VR、遥操作机器人、自采增强)统一到同一动作空间构成 **EgoAtlas**(343K 轨迹 / 89.72M 图像-动作对),再提出 **motion-aware dynamics**——用 VQ-VAE 离散视觉动态 + RQ-VAE 离散手部运动动态(合计 44 个 token)作为 VLA 的紧凑监督信号,让 7B VLM 同时学会 reasoning 与 acting;在 Unitree G1 + Inspire 灵巧手的 6 个真实任务上取得最高平均成功率(Pick-and-Place 85%、长程 Open Drawer+Bread 75% SR / 82.5% PSR),仅用 10% 数据微调仍达 50% 成功率。

## 一、问题与动机

灵巧操作(dexterous manipulation)的核心瓶颈是**带动作标注的大规模数据极度稀缺**:遥操作采集困难且昂贵,导致现有 VLA 研究大多停留在简单夹爪任务,灵巧手操作基本未被充分探索。

与机器人数据的稀缺相反,**人类数据海量且语义丰富**,天然蕴含操作行为的先验。但作者指出已有"从人类数据学操作"的路线存在两大缺陷:

1. **场景同质化 / 上下文偏置**:人类视频往往局限于特定家庭或工作场景(如桌面、厨房),场景覆盖不均、上下文偏置强。
2. **人-机之间存在巨大的视觉与动作空间 gap**:纯人类视频含大量与操作无关的冗余内容,且难以直接迁移到机器人本体。

作者的思路是:(a) 用**多源**自视角数据消除单一数据集的场景偏置;(b) 用**统一动作空间**跨本体对齐人手与机器人手,弥合 embodiment gap;(c) 从人手动作中显式提取**运动先验**,把视觉变化与运动信息一起建模,而非直接在原始像素/连续动作上学习。

## 二、核心方法

METIS 由三部分构成:统一动作空间的数据集 EgoAtlas、motion-aware dynamics 离散表示、以及集成 reasoning+acting 的 VLA 主干。

### 2.1 EgoAtlas 数据集与统一动作空间

**四大类数据源**(共 8 个数据集):(1) 基于视觉的动捕数据集(多相机光学系统,精确 3D 手部标注,但局限小桌面);(2) 基于 VR 的数据集(设备端 SLAM + 标定相机);(3) 遥操作机器人数据;(4) 自采增强动作数据集——作者用**可穿戴手套-Tracker 系统**采集 10K 高保真人手轨迹。

自采系统硬件:Manus Quantum Metagloves 记录每只手 25 个关键点的 3D 位置,每只手套上 VIVE Tracker 提供 6-DoF 腕部位姿,头戴相机采第一人称视角,头戴另一 VIVE Tracker 做外参标定,系统 20Hz 运行,还额外提供俯视第三人称视角。每条轨迹配语言指令并做子任务级(subtask-level)细粒度分割。

**统一动作空间**:观测为 $o_t = \{I_t, S_t\}$,轨迹 $\tau = \{(o_t, a_t)\}_{t=1}^T$。动作由两部分组成——18D 腕部位姿 $P_t^w$(每只手 3D 位置 + 6D 旋转向量,遵循 Zhou et al. 的连续旋转表示,统一到自视角相机坐标系)与 30D 指尖位姿 $P_t^f$(每只手指尖 3D 位置,标定到腕部坐标系)。灵巧手关节角通过 FK 映射到指尖位置;推理时反过来用 IK 把预测的指尖目标还原为关节角。人手与机器人手的腕部坐标系通过标定对齐,保证跨本体一致性。

EgoAtlas 关键统计(Tab. 1,总计 343K 轨迹 / 89.72M 帧,其中 EgoDex 独占约 77.9M 帧):

| 数据集 | 轨迹数 | 帧数 | Pose | Subtask | Human | Robot | In-the-wild |
|---|---|---|---|---|---|---|---|
| ARCTIC | 296 | 214.5K | ✓ | ✗ | 100% | 0% | ✗ |
| H2O | 109 | 65.3K | ✓ | ✓ | 100% | 0% | ✗ |
| HoloAssist | 100 | 777.3K | ✓ | ✓ | 100% | 0% | ✗ |
| Oakink | 134 | 146K | ✓ | ✗ | 100% | 0% | ✗ |
| EgoDex | 314.8K | 77.9M | ✓ | ✗ | 100% | 0% | ✓ |
| PH2D | 1.8K | 416.5K | ✓ | ✗ | 66.1% | 33.9% | ✓ |
| ActionNet | 15.7K | 7.4M | ✗ | ✗ | 0% | 100% | ✓ |
| Ours(自采) | 10K | 2.8M | ✓ | ✓ | 100% | 0% | ✓ |

预训练用加权采样平衡分布(Tab. 6):EgoDex 40.3%、自采增强数据 25.4%、ActionNet 13.4%、HoloAssist 10.6%、PH2D 5.4%、ARCTIC 2.8%、OAKINK 1.9%、H2O 0.8%。

### 2.2 Motion-Aware Dynamics(核心贡献)

动机:标准 VLA 用 tokenizer 把连续动作离散化后自回归生成,但随着 action chunk 变长、系统自由度(DoF)升高,token 序列爆炸导致生成慢,且难以捕捉灵巧操作里细微手指运动与接触交互的细节。作者提出把动态信息拆成**视觉动态 $D_{vis}$** 与**运动动态 $D_{mot}$** 两条互补支路,分别离散化。

**(a) 视觉动态离散化(VQ-VAE)。** 用一个 Inverse Dynamics Model(IDM)风格的编码器和 Forward Dynamics Model(FDM)风格的解码器:

$$\mathcal{I}(D_{vis} \mid I_t, I_{t+k}, P_{t,t+1,\ldots,t+k}), \qquad \mathcal{F}(I_{t+k} \mid I_t, D_{vis})$$

编码器(含 spatial + temporal transformer)融合视觉观测与连续运动,抽取"与运动相关的视觉变化";解码器(仅 spatial transformer)据 $I_t$ 和 $D_{vis}$ 预测未来帧。量化用 VQ-VAE:

$$D'_{vis} = \mathbf{VQ}(\hat{D}_{vis}), \qquad I_{t+k} = \mathrm{Dec}_V(I_t, D'_{vis})$$

关键:不重建原始像素(像素含大量与操作无关的冗余),而是用预训练 DINOv2 抽取高层语义特征做重建目标。最终每段选 $V=4$ 个 token,码本大小 $|C_v|=16$。

**用大白话说**:与其让模型去背下"下一帧长啥样"的每个像素,不如让它只记住"从这一帧到下一帧,手做了什么导致画面语义变了"——把这个"视觉上的动作后果"压缩成 4 个离散码字,既省又抓重点。

**(b) 运动动态量化(RQ-VAE)。** 另设一套码本专注手部运动本身。先用 PoseNet(多尺度时序卷积 + 轨迹自注意力)编码 3D 手部运动,再用两层残差量化(RQ-VAE)离散化,残差结构可防码本坍塌并从粗到细分层捕捉运动模式;训练时用 TCN 解码器重建原运动序列做监督:

$$D_{mot} = \mathbf{RQ}(P_{t,t+1,\ldots,t+k}), \qquad M_{t,t+1,\ldots,t+k} = \mathrm{TCN}(D_{mot})$$

每段选 $R=40$ 个 token,共享码本大小 $|C_m|=512$。所有码本统一特征维度 $d=128$。

**用大白话说**:手部运动比画面变化更精细、更"抖",单层量化容易漏细节或直接坍缩成几个码字;残差量化像"先画大轮廓再逐层补细节",第一层记大致挥手方向,第二层补指尖微调,合起来 40 个码字就能相当忠实地复原一段灵巧手运动。

### 2.3 METIS 模型:统一 reasoning 与 acting

- **主干**:基于 VLM,参数由 Prismatic-7B 初始化。视觉端为 SigLIP + DINOv2 混合编码器($f^{SigLIP}\in\mathbb{R}^{N_v\times 1024}$、$f^{DINO}\in\mathbb{R}^{N_v\times 1152}$ 沿通道拼接),LLM 主干为 7B LLaMA-2(decoder-only,32 层)。
- **词表扩展**:把 LLaMA-2 词表扩出 $|C_1|+|C_2|$ 个特殊 token,分别对应视觉动态与运动动态码本。每段自视角序列被离散成 motion-aware action token $D_{vis}$ 与 $D_{mot}$,每个 dynamics 特征映射到最近码字→唯一特殊 token。自回归目标:

$$\mathcal{L}_{ar} = \mathbb{E}_{o_t,l,a_{d,<i}}\left[-\sum_{i=1}^N \log \pi_\phi(\hat{a}_{d,i} \mid o_t, l, a_{d,<i})\right]$$

其中 $N$ 为 dynamics token 总长 = 44(4 视觉 + 40 运动)。

**用大白话说**:不像 RT-2/OpenVLA 把动作切成一堆均匀 bin(维度一高就爆且不稳),METIS 让 VLM 只需自回归吐出 44 个"动作语义 token",既保住语言模型原有先验,又把灵巧操作的细粒度动态注了进去。

- **Action Decoder**:把 dynamics token 翻译成可执行低层动作。输入 = dynamics token + 视觉 embedding + 当前本体感知;视觉与动态特征经多头注意力池化聚合,本体感知经两层 MLP 投影到隐空间,融合后线性投影预测未来 1 秒的动作序列(30 步 @ 30Hz)。总损失:

$$\mathcal{L} = \mathcal{L}_{ar} + \lambda \mathcal{L}_{action}$$

**用大白话说**:VLM 负责"想清楚要做什么动作语义",Action Decoder 负责"把语义翻译成机器人当下这一秒具体怎么动关节",两者用一个加权和一起训。

- **Chain-of-Thought Reasoning for Action**:借鉴 CoT,把高层指令拆成短子任务,每个子任务配细粒度手部动作描述(如 "use right hand to pour the juice from the tall blue mug to the bowl")。引入两个特殊 token:reasoning 起始 $[BOA]$、dynamics 起始 $[BOD]$。**仅在子任务切换时**才进入 reasoning 模式(大幅降低推理延迟);预测到 $[BOD]$ 时 VLM 直接输出 motion-aware dynamics 交给 decoder 执行。这种自适应切换在提升 reasoning 与 control 相互理解的同时减少延迟。

## 三、实验结果

**硬件与设置**:Unitree G1 人形机器人 + 一对 Inspire 6-DoF 灵巧手,头部 Intel RealSense D435 采自视角 RGB。机器人示范经手套-Tracker 遥操作采集(DexCap 风格),腕部位姿经 IK 转关节角,指尖轨迹经 IK 重定向到灵巧手关节空间。6 个任务:3 个短程(Pick and Place / Close Laptop / Open Drawer)+ 3 个长程(Grasp Two Drinks into Basket / Put Cola into Basket / Open Drawer and Put Bread),每任务 100 条高质量示范、默认 20 次试验评测。指标:成功率 SR 与进度成功率 PSR(长程任务子任务平均完成比)。基线:ACT、OpenVLA-OFT、π0.5、Gr00t N1.5。

**主结果(Tab. 2,SR / PSR,%)**:

| 方法 | Pick&Place SR | Close Laptop SR | Open Drawer SR | Grasp 2 Drinks SR/PSR | Put Cola SR/PSR | Open Drawer+Bread SR/PSR |
|---|---|---|---|---|---|---|
| ACT | 35 | 65 | **95** | 25 / 40 | 50 / 53.3 | 5 / 5 |
| OpenVLA-OFT | 50 | 80 | 10 | 40 / 57.5 | 55 / 56.7 | 0 / 1 |
| π0.5 | 60 | 85 | 70 | 65 / 72.5 | **75** / 76.7 | 60 / 65 |
| GR00T N1.5 | 70 | 80 | 80 | 65 / 70 | 70 / 73.3 | 70 / 72.5 |
| **METIS** | **85** | **95** | 90 | **75 / 85** | 70 / **76.7** | **75 / 82.5** |

METIS 取得最高平均成功率,且在所有长程任务上 PSR 最高。个别项被超越:Open Drawer 上 ACT 95% 略高于 METIS 90%;Put Cola SR 上 π0.5 75% 高于 METIS 70%(但 PSR 打平 76.7%)。作者分析:专用模型 ACT 短程强、长程崩(缺集成 reasoning);π0.5 因未在大规模灵巧数据上预训练,精细灵巧操作偏弱;GR00T N1.5 借大规模预训练有竞争力,但缺显式 reasoning 与反馈修正,长程受限。

**样本效率(Fig. 6)**:仅用 10% 数据微调,METIS 在 Pick and Place 上仍达 50% 成功率,验证多源自视角预训练 + 统一动作空间提供了可快速迁移的视觉运动先验。

**指令跟随(Fig. 5)**:桌面放三种不同颜色水果(apple/orange/lemon),给不同语言指令,METIS 能识别目标水果并执行对应抓取。

**OOD 泛化(Tab. 3,以 Open Drawer and Put Bread 为例,%)**:

| 方法 | 未见背景 | 未见光照 | 未见物体 | 杂乱场景 |
|---|---|---|---|---|
| π0.5 | 50 | **70** | 65 | 55 |
| GR00T N1.5 | 65 | 65 | 65 | 60 |
| **METIS** | **70** | 65 | **70** | **70** |

**跨本体泛化(Fig. 7)**:迁到 Sharpa Beta 本体的一对 22-DoF SharpaWave 灵巧手,Grasp Apple into Basket 达 85%、Tool Use 达 70%。因 METIS 预测指尖轨迹而非直接关节角,天然不受手部运动学差异影响。

**消融**:

| 预训练消融(Tab. 4) | Pick&Place | Open Drawer+Bread |
|---|---|---|
| METIS-NoPretrain | 60 | 35 |
| METIS-HumanPretrain(仅开源人类数据) | 70 | 60 |
| METIS-FullPretrain(EgoAtlas 全量) | **85** | **75** |

| motion-aware dynamics 消融(Tab. 5) | Pick&Place | Open Drawer+Bread |
|---|---|---|
| w/o(仅监督连续动作) | 30 | 0 |
| w/ | **85** | **75** |

去掉 motion-aware dynamics 后长程任务直接归零,证明离散动态表示对时序一致性与细粒度动作预测至关重要;无预训练版本虽在部分任务有一定成功率,但预训练阶段 loss 波动大、部署时关节抖动明显。

**训练开销**:预训练全参优化(视觉编码器 + LLM + action decoder),global batch 768、FSDP、AdamW lr 2e-5,24×H100,60k 步约 72 小时。后训练 8-GPU、LoRA(rank 32,加在 LLM 与视觉编码器)+ action decoder 全参微调,LLM/编码器 bf16、action decoder fp32。

## 四、局限性

- **仅依赖自视角观测**:难以感知完整物体几何与精细交互细节;作者建议后续加入腕部相机或外部相机弥补。
- **预训练排除了大规模第三人称数据**:目前只用自视角源,未利用互联网上海量第三人称/多视角操作数据,扩展到更广的多视角数据集是明确的 future work。
- **评测规模有限**:每任务仅 20 次试验,6 个任务、单一 Unitree G1 主平台,统计置信区间偏窄;跨本体仅测 2 个任务。
- **motion-aware dynamics 的超参(V=4/R=40、码本 16/512)缺乏系统消融**,码本尺寸对不同任务复杂度的敏感性未探讨。

## 五、评价与展望

**优点**:(1) **统一动作空间的工程完整度高**——把腕部 6D 旋转 + 指尖 3D 位置作为跨本体公共接口,配合 FK/IK 与坐标系标定,让人手动捕、VR、遥操作、机器人数据真正落到同一动作语义上,这是本文能"多源混训"的地基,也是它能零成本跨到 22-DoF 手的原因。(2) **motion-aware dynamics 的设计有巧思**:视觉动态(VQ-VAE,DINOv2 语义重建而非像素)与运动动态(RQ-VAE 残差量化)分工明确,44 个 token 的紧凑表示既缓解了长序列自回归的效率问题,又比"均匀动作 bin"更贴合灵巧操作的高频细节;消融里去掉它长程归零,说明这不是锦上添花而是承重结构。(3) **reasoning/acting 的自适应切换**(仅子任务切换时 reasoning)在长程任务的 PSR 优势上得到印证。

**与公开工作的关系**:本文处在"从人类数据预训练灵巧 VLA"这条正在快速拥挤的赛道上。与 EgoVLA、Being-H0 等"从大规模人类视频学 VLA"相比,METIS 强调**多源混合 + 统一动作空间**去打散单一数据集的场景偏置;与 LAPA/UniVLA 的 latent action 路线相比,METIS 的 dynamics 是**显式配对了运动信息(IDM/FDM)且分视觉/运动双支路**,而非纯从像素学隐动作;与 GR00T N1.5、π0.5 这类通用人形 VLA 相比,它把"灵巧手指尖轨迹"作为一等公民,牺牲了部分夹爪场景通用性换取灵巧操作精度。数据集层面 EgoAtlas 与 EgoDex(其最大来源,占 40% 权重)高度绑定,某种程度上 METIS 是"以 EgoDex 为骨、多源为肉"的组合。

**开放问题与改进方向**:(1) 论文的对比缺少"同数据、同本体下把 motion-aware dynamics 换成 latent action / flow-matching 连续动作头"的对照,难以判断增益究竟来自离散动态表示本身还是来自多源数据;(2) 44 token、V/R 与码本尺寸缺消融,残差量化层数与任务复杂度的关系值得系统扫;(3) reasoning 由人工标注子任务驱动,标注成本与自动化是可扩展性瓶颈,能否用 VLM 自动生成子任务链值得探索;(4) 纯自视角的几何盲区可用腕部相机或引入深度/点云缓解;(5) 评测规模(20 试验/任务)偏小,长程任务的稳健性结论需要更大样本支撑。总体上,这是一篇"数据统一 + 离散动态表示"两手都硬、工程扎实的灵巧 VLA 工作,主要遗憾在于消融未能干净地把"数据贡献"和"表示贡献"解耦。

## 参考

1. Hoque et al. *EgoDex: Learning Dexterous Manipulation from Large-scale Egocentric Video*, arXiv:2505.11709, 2025.(EgoAtlas 最大数据来源)
2. Wang et al. *DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation*, arXiv:2403.07788, 2024.(自采手套-Tracker 系统与遥操作范式来源)
3. Yang et al. *EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos*, arXiv:2507.12440, 2025.(同类"从人类自视角学 VLA")
4. Luo et al. *Being-H0: Vision-Language-Action Pretraining from Large-scale Human Videos*, arXiv:2507.15597, 2025.(大规模人类视频 VLA 预训练对照)
5. NVIDIA. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025.(主要人形 VLA 基线)
6. Ye et al. *LAPA: Latent Action Pretraining from Videos*, arXiv:2410.11758, 2024.(latent action 离散表示路线对照)
