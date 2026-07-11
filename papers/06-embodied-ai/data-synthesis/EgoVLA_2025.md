# EgoVLA：从第一视角人类视频中学习视觉-语言-动作模型

> **论文**：*EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos*
>
> **作者**：Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang et al.
>
> **机构**：UC San Diego、UIUC、MIT、NVIDIA
>
> **发布时间**：2025 年 07 月（arXiv 2507.12440，v3 于 2025-07-18）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.12440) | [PDF](https://arxiv.org/pdf/2507.12440)
>
> **分类标签**：`egocentric-human-video` `VLA` `humanoid-manipulation` `unified-action-space` `MANO`

---

## 一句话总结

把"人手"当成一种特殊的机器人本体：先在约 50 万对第一视角人类视频（手腕位姿 + MANO 手部参数）上预训练一个 Human VLA，再用 MANO 手指尖统一动作空间把人手动作 IK/retarget 到双臂人形机器人，最后仅用每个任务 100 条机器人示教做 post-training，在自建的 Ego Humanoid Manipulation Benchmark 上短程任务 Seen 成功率达 77.78%、长程 45.93%，显著超过无人类预训练版本（64.55% / 26.67%）与逐任务专家 ACT（24.87% / 2.22%）。

## 一、问题与动机

模仿学习依赖真机数据采集,而真机数据需要机器人硬件 + 专家操作员(joint mapping / exoskeleton / VR 遥操作),这从根本上限制了数据规模与任务、场景多样性。作者的核心提问是:能否直接从人类视频学操作?如果把人类视为一种特殊机器人,全世界有"80 亿台机器人"持续在各种真实环境中操作,这是天然的海量、多样化数据源。

关键观察:**人类动作空间与机器人动作空间的差异其实没那么大,可以用少量几何变换来近似**。因此不必像 OpenVLA / π0 那样在机器人数据上从零训练 VLA,而是先在人类数据上训练一个 Human Egocentric VLA(预测未来的人手手腕位姿与手部关节角),再通过 IK(手腕→末端执行器)和 retargeting(手关节→机器人手关节)转换为机器人动作。这样得到的 Human VLA 本质上"已经是"一个机器人策略,只是输入是人手图像、且动作输出存在误差——用少量机器人示教微调即可修正,从而避免对大规模真机数据的依赖。

## 二、核心方法

### 1. 模型结构

以 **NVILA-2B** 作为 VLM 骨干,输入为:

- **视觉观测**:当前帧 + 前 5 帧共 6 帧 RGB,以 0.2 秒间隔采样、覆盖 1 秒历史,每帧分辨率 $384 \times 384$;
- **语言指令**:描述即时目标行为(聚焦技能执行而非高层规划);
- **本体感知(proprioception)**:手腕平移/旋转 + 手部位姿参数,经 MLP 编码;
- **action query tokens**:取词表最后 $H=30$ 个 word ID 作为查询 token。

VLM 输出的隐特征送入 **action head**(一个 300M 的 transformer,6 层 encoder,hidden size 1536),预测未来 1 秒、双手各 30 步(30 Hz)的动作序列 $A_t = [a_t, a_{t+1}, \dots, a_{t+H}]$。

每步动作包含:**手腕位姿**(相机系下 3D 平移 + rot6D 旋转)与 **手关节角**(用 MANO 手模型的前 15 个 PCA 主成分表示)。

训练目标为 L2 回归损失的加权和:

$$\mathcal{L} = \lambda_{\text{wrist trans}}\,\mathcal{L}_{\text{wrist trans}} + \lambda_{\text{wrist rot}}\,\mathcal{L}_{\text{wrist rot}} + \lambda_{\text{joint}}\,\mathcal{L}_{\text{joint}}$$

其中 $\mathcal{L}_{\text{wrist trans}} = \lVert \mathbf{T}_{\text{pred}} - \mathbf{T}_{\text{gt}} \rVert_2^2$、$\mathcal{L}_{\text{wrist rot}} = \lVert \mathbf{R}_{\text{pred}} - \mathbf{R}_{\text{gt}} \rVert_2^2$(旋转先由 rot6D 转成旋转矩阵再算 L2)、$\mathcal{L}_{\text{joint}} = \lVert \boldsymbol{\Theta}_{\text{pred}} - \boldsymbol{\Theta}_{\text{gt}} \rVert_2^2$。权重取 $\lambda_{\text{wrist trans}}=20.0,\ \lambda_{\text{wrist rot}}=5.0,\ \lambda_{\text{joint}}=5.0$。

> 用大白话说:模型看 1 秒的第一视角历史帧 + 一句指令,直接"脑补"接下来 1 秒里双手的手腕怎么动、手指怎么弯,输出是一整段 30 步的动作轨迹(action chunking),而不是单步动作。

### 2. 统一动作空间(核心创新)

人手和机器人手形态差异大,直接迁移困难。作者用 **MANO 手指尖 3D 位置** 作为人机共享的动作空间:

- **训练时把机器人数据转成人类表示**:给定观测到的机器人指尖位置 $\mathbf{J}_{\text{obs}} \in \mathbb{R}^{5\times 3}$(5 指),优化一组 MANO 参数 $\boldsymbol{\Theta} \in \mathbb{R}^{15}$,使 MANO 正运动学算出的指尖 $\mathbf{J}_{\text{pred}}(\boldsymbol{\Theta})$ 与之对齐:

$$\mathcal{L}(\boldsymbol{\Theta}) = \frac{1}{5}\sum_{i=1}^{5}\text{SmoothL1}\big(\mathbf{J}_{\text{pred}}(\boldsymbol{\Theta})_i,\ \mathbf{J}_{\text{obs},i}\big)$$

  末端执行器位姿则用 3D 变换对齐人机坐标系。这样机器人示教就被改写成"人手表示",可直接拿去微调 EgoVLA,无需改动网络结构或重新初始化。

- **部署时把人手预测转成机器人指令**:预测出的手腕位姿先经 3D 变换得到机器人末端位姿,再用 IK 求臂关节角;手部则用 MANO 模型从预测参数算出 3D 手部关键点,再由一个轻量四层 MLP(hidden $[64,128,64]$)把指尖位置映射为机器人手各自由度的驱动指令。该 retargeting MLP 用"机器人示教被转成人手表示"的配对数据训练,平均指尖位置误差仅 $5\times 10^{-5}$ m,回放原始示教验证任务成功率基本不变。

> 用大白话说:不管是人的手还是机器人的三指/多指手,都先归约到"五个指尖在空间里的位置"这一层。训练时把机器人手"伪装"成人手,推理时再把人手预测"翻译"回机器人手。IK 负责手臂、MLP 负责手指。

### 3. 数据集与训练

**Ego-Centric Human Manipulation Dataset**:混合四个第一视角人类数据源(相对比例见原文 Fig.3):

| 数据源 | 规模/内容 | 占比 | 特点 |
| --- | --- | --- | --- |
| HOI4D | 4,000 段单手操作(取放、重定向、铰接物体) | 39% | 任务多样 |
| HOT3D | 833 分钟、与 33 个刚体交互 | 25% | 精确 3D 手/相机位姿标注 |
| HoloAssist | 166 小时复杂任务(换电池、组装家具、装机) | 23% | 双手交互丰富但手标注较噪;均匀采样 1/10 以平衡 |
| TACO | 2,317 段运动、151 个 tool-action-object 三元组 | 13% | 工具使用 |

数据处理:用世界系相机位姿把未来手腕位置投影回当前相机帧以保证监督一致;RGB 按 3 FPS 采样。合计约 **500,000** 对图像-动作。HOT3D 无语言标注,用占位指令;HoloAssist 非 MANO 的手模型 retarget 到 MANO 统一表示;统一采用 MANO 平均手形。

训练分两阶段(32 张 A100):人类视频预训练 **20** epoch(LR 1e-4,cosine);机器人示教 post-training **115** epoch(LR 2e-5,100 epoch 后降到 2e-6,constant),全模型(含视觉编码器)一起微调。

### 4. Ego Humanoid Manipulation Benchmark

基于 **NVIDIA Isaac Lab** 自建的仿真评测台:Unitree H1 人形机器人 + 两只 Inspire 灵巧手,共 **12** 个任务——短程原子任务(Push-Box、Flip-Mug、Pour-Balls、Close-Drawer、Open-Drawer、Open-Laptop、Stack-Can)与长程多阶段任务(Sort-Cans、Insert-Cans、Unload-Cans、Insert-And-Unload-Cans、Stack-Can-Into-Drawer)。动作空间 36 维(臂 IK + 手直接驱动),每只手 12 DoF(6 主动 + 6 联动),控制频率 30 Hz。5 种房间纹理 × 5 种桌面纹理 = 25 种视觉背景组合。示教用 OpenTelevision + Meta Quest 3 采集,每任务 100 条成功示教(100–500 帧)。评测:Seen 每任务 27 次 rollout(3 个背景 × 9);Unseen 每任务 66 次(22 个未见背景 × 3)。指标为成功率 SR 与进度率 PSR(长程任务中已完成子任务占比)。

## 三、实验结果

**人手运动建模**:训练后 EgoVLA 对人手手腕平移的平均未来预测误差约 **8 cm**,投影到 2D 图像平面归一化误差约 **0.13**,与 HOI-forecast 的 SOTA 相当;改动语言指令(如把 "Put it in the drawer" 改成 "Take it out of the drawer")能相应改变预测轨迹,说明模型学到了语义意图而非仅拟合运动。

**短程任务平均(Table 1)**:

| 方法 | Seen SR | Seen PSR | Unseen SR | Unseen PSR |
| --- | --- | --- | --- | --- |
| ACT(逐任务专家) | 24.87 | 59.79 | 24.89 | 54.22 |
| EgoVLA-NoPretrain | 64.55 | 71.87 | 57.58 | 62.63 |
| EgoVLA (50% 机器人数据) | 48.15 | 61.73 | — | — |
| **EgoVLA** | **77.78** | **84.92** | **69.11** | **76.26** |

**长程任务平均(Table 2)**:

| 方法 | Seen SR | Seen PSR | Unseen SR | Unseen PSR |
| --- | --- | --- | --- | --- |
| ACT(逐任务专家) | 2.22 | 26.47 | 0.61 | 23.51 |
| EgoVLA-NoPretrain | 26.67 | 54.93 | 26.30 | 36.20 |
| EgoVLA (50% 机器人数据) | 7.41 | — | — | — |
| **EgoVLA** | **45.93** | **80.78** | **28.79** | **69.11** |

关键结论:

- **必须有真机 post-training**:仅在人类视频预训练、不做机器人微调时,零样本部署在人形机器人上所有任务成功率为 **0%**(外观、感知、运动学都有失配),即便是预训练见过的 Pouring Balls 也失败。
- **人类预训练提升同域性能**:相比 NoPretrain,增益在需要精细操作的任务(Stack-Cans、Sort-Cans、Insert-And-Unload-Cans、Flip-Mug)上尤为明显;长程任务上 EgoVLA 比 NoPretrain 高约 20 个百分点。作者归因于人类预训练学到了与本体无关的通用"手"操作先验。
- **人类预训练增强跨域泛化**:短程任务从 Seen 到 Unseen,EgoVLA 平均成功率下降幅度较小(77.78→69.11),而 NoPretrain 退化更明显;长程 Unseen 仍有约 28.79% 成功率,且 PSR 掉幅小于 SR,说明失败主要发生在任务后期而非早期子目标。
- **机器人数据规模消融**:只用 50% 机器人示教(EgoVLA 50%),长程平均成功率从 45.93% 骤降到 **7.41%**——预训练虽能提升同域与泛化,但仍需适量任务相关真机示教。
- **预训练数据混合消融(Fig.7)**:四源全用(HOI4D+HOT3D+HoloAssist+TACO)在 Unseen 短程任务上 SR/进度最佳,数据多样性越高泛化越好;即便 HoloAssist 标注噪声大、HOT3D 缺语言、TACO 视觉多样性有限,仍有正迁移。

## 四、局限性

- **依赖带手/腕位姿标注的人类数据**,可获得性受限(不过 Quest 3 / Vision Pro / Aria 等 AR/VR 设备普及会缓解)。
- **无法零样本部署**,必须在适量真机数据上微调;统一动作空间只是缩小了差距,没有消除。
- 评测**全在仿真**完成(未做真机 sim-to-real):作者以"仿真评测与真机高度相关"的相关工作为依据,并称方法本身与模态无关、可用任意高质量遥操作示教;但论文并未提供真机部署结果。
- 手腕平移预测误差约 8 cm 仍偏大;retargeting/IK 在人机形态差异大时误差会传导。
- 长程任务 Unseen 成功率仅约 28.79%,离实用尚远。

## 五、评价与展望

**优点**:(1)"人手即机器人本体"的思路清晰,MANO 指尖统一动作空间在工程上很干净——把机器人手"伪装"成人手来对齐、用 IK + 轻量 MLP 拆分臂/手,避免了对网络结构的侵入式改动;(2)预训练/微调两阶段的消融做得扎实,尤其"零样本 0%""50% 数据长程 45.93→7.41"这两个数字明确界定了人类视频预训练的作用边界——它提供先验而非替代真机数据,这一诚实结论比很多夸大"零样本人到机"的工作更有参考价值;(3)配套开源了一个 12 任务、含长程与视觉泛化拆分的 Isaac Lab 人形双臂 benchmark,填补了灵巧操作可复现评测的空白。

**不足与开放问题**:(1)缺真机验证是最大短板,统一动作空间在真实相机噪声/接触力下能否成立仍存疑;(2)与 EgoMimic、"Humanoid Policy ∼ Human Policy" 等同期"第一视角人类视频→机器人"工作缺乏定量对比,baseline 仅有 NoPretrain 消融与逐任务 ACT,未与 OpenVLA / π0 这类主流机器人 VLA 正面比较;(3)动作头采用 L2 回归而非扩散/流匹配(对比 π0),对多峰动作分布的建模能力可能受限;(4)语言指令被刻意限制在"即时行为"层面,回避了长程规划,长程能力主要靠 action chunking 硬撑,这也解释了长程 Unseen 仍偏低。

**可能的改进方向**:引入更 embodiment-agnostic 的预训练目标以逼近真正的零样本迁移;把回归头换成流/扩散策略头以提升灵巧多峰动作建模;在统一动作空间中显式建模接触与力,而非仅指尖位置;补充真机 sim-to-real 与跨机器人本体的验证。

## 参考

1. Liu et al. *NVILA: Efficient Frontier Visual Language Models*, 2024（arXiv 2412.04468）—— EgoVLA 的 VLM 骨干。
2. Romero, Tzionas, Black. *Embodied Hands: Modeling and Capturing Hands and Bodies Together*(MANO), SIGGRAPH Asia 2017 —— 统一动作空间所用手模型。
3. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, CoRL 2024 —— 直接在机器人数据上训练 VLA 的对照范式。
4. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024 —— 流匹配动作头的代表性 VLA。
5. Kareer et al. *EgoMimic: Scaling Imitation Learning via Egocentric Video*, 2024（arXiv 2410.24221）—— 同类第一视角人类视频驱动的机器人模仿学习工作。
