# Qwen-VLA：跨任务、跨环境、跨机器人本体的统一视觉-语言-动作建模

> **论文**：*Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments*
>
> **作者**：Qiuyue Wang*、Mingsheng Li*、Jian Guan* 等（* 同等贡献）、Shuai Bai†（通讯作者）等 Qwen Team 全体作者
>
> **机构**：Qwen Team
>
> **发布时间**：2026 年 06 月（arXiv 2605.30280）
>
> **发表状态**：未录用（预印本，Qwen 官方技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.30280) | [PDF](https://arxiv.org/pdf/2605.30280)
>
> **分类标签**：`VLA基础模型` `跨具身泛化` `flow matching动作专家` `统一动作-轨迹表示` `操作+导航联合训练` `强化学习后训练`

---

## 一句话总结

Qwen-VLA 在 Qwen3.5-4B 多模态骨干上外挂一个约 1.15B 参数的 DiT flow-matching 动作专家,用"具身感知 prompt"+统一的动作-轨迹张量接口,把机器人操作、视觉语言导航、人类第一视角示范、轨迹预测四类异质任务纳入同一个生成式模型,配合 T2A→CPT→SFT→RL 四阶段渐进训练;最终模型 Qwen-VLA-Instruct 在 LIBERO(97.9%)、RoboTwin-Easy/Hard(86.1%/87.2%)、真实 ALOHA 分布外任务均值(76.9%)、DOMINO 动态操作零样本(SR 26.6%/MS 39.5)等多个基准上追平或超过为单一任务专门训练的 specialist 策略。

## 一、问题与动机

具身智能研究长期碎片化:操作模型多针对桌面/灵巧手场景单独训练,导航模型围绕 waypoint 或离散动作预测单独设计;不同任务在观测格式、控制频率、预测 horizon、动作维度、评估协议上差异巨大,难以像通用视觉语言预训练那样规模化扩展。作者的核心洞察是:尽管表层形态不同,操作、导航、轨迹预测其实共享同一计算结构——都是"给定视觉观测、语言指令与具身约束,预测未来动作或轨迹"的条件生成问题。基于这一观察,论文提出一个联合预训练框架,把操作轨迹、人类第一视角示范、合成仿真数据、视觉语言导航数据统一吸收进单个 VLA 模型,验证跨具身、跨任务的联合建模是否既不牺牲单任务性能,又能带来更强的分布外鲁棒性。

## 二、核心方法

### 2.1 统一问题形式化

在时间步 $t$,模型接收视觉上下文 $o_t$、语言指令 $x$、具身描述 $e$、可选任务标识 $z$,预测未来 $H$ 步目标序列:

$$p_\theta(y_{t:t+H-1} \mid o_t, x, e, z)$$

用大白话说:不管是移动机械臂末端、移动导航 waypoint,还是移动人手在 MANO 姿态空间中的关节,都被建模成同一种"看图+读指令+知道自己是谁 → 吐出一段未来轨迹"的条件生成问题。

### 2.2 架构:VLM 骨干 + DiT 动作专家

骨干为 Qwen3.5(原生多模态早期融合,视觉 token 与文本 token 交织进同一 token 流;混合注意力设计,多数层用门控线性注意力、定期插入分组查询 softmax 注意力以保留长序列全局推理能力)。动作专家是单流 DiT 风格 flow-matching 头:将 VLM 隐状态与带噪动作 chunk 拼接成一个序列后做联合 self-attention,用 AdaLN 注入时间步条件,多段 RoPE 与骨干对齐,推理时只需几步 Euler 积分即可低延迟出动作。动作专家约 1.15B 参数(16 个 DiT block,每块 70.8M,合计 1.13B;其余为动作投影 MLP 4.9M、隐状态到 DiT 通道线性层 3.9M、时间步嵌入 2.8M、输出 AdaLN 调制 4.7M)。

### 2.3 具身感知 prompt 条件化

每条训练样本前缀一段具身文本 prompt:"The robot is {robot_tag} with {single arm/dual arms}[, waist][, and mobile base]. The control frequency is {FPS} Hz. Please predict the next {chunk_size} control actions to execute the following task: {ori_instruction}."。robot_tag 与可选修饰项(腰部、移动底盘)按具体本体设定,FPS 与 chunk_size 反映该数据集原生控制频率与预测 horizon。

用大白话说:模型不需要为每种机器人单独设计输出头或子网络,只靠一句自然语言描述"你是谁、几只臂、多快频率、预测多少步",就把控制惯例注入同一套共享参数,新增具身只需换一句 prompt。

### 2.4 统一动作-轨迹表示

论文并不强行把所有本体的物理动作语义压进同一空间,而是统一张量接口与掩码机制:每个样本贡献目标张量 $\mathbf{Y}\in\mathbb{R}^{H\times K}$($H$ 为固定预测 horizon,$K$ 为跨所有控制模式共享的通道维度上限)。给定控制模式使用 $c\le K$ 个通道,任务相关值放在 $\mathbf{Y}$ 的前 $c$ 维,其余维度零填充;二值掩码 $M\in\{0,1\}^{H\times K}$ 记录 $M_{h,k}=1$ 当且仅当通道 $k\lt c$ 且时间步 $h$ 落在该任务 chunk 长度 $H_{\text{task}}\le H$ 内。该方案无需具身专属输出头,单一 DiT 参数集处理所有控制模式,padding 位置被掩码完全排除在梯度之外。

覆盖两族连续控制信号:操作信号(末端位姿增量 $(\Delta x,\Delta y,\Delta z)$、欧拉角/四元数末端旋转、绝对关节角、夹爪开合、灵巧手关节角)与导航轨迹信号(地面坐标系下每 waypoint 的相对位移与朝向变化 $(\Delta x,\Delta y,\Delta\theta)$)。两族信号物理语义不同,但都是"horizon 上的实值向量序列",因此动作专家一视同仁处理。

### 2.5 训练目标

Flow-matching 动作损失采用"每通道-每步"两级平均,避免梯度被 padding 稀释。给定干净目标 $\mathbf{Y}_0$、噪声 $\mathbf{Y}_1\sim\mathcal{N}(0,\mathbf{I})$、线性插值 $\mathbf{Y}_\tau=(1-\tau)\mathbf{Y}_0+\tau\mathbf{Y}_1$,先对每个有效通道 $k\lt c$ 算时间步平均误差:

$$\ell_k=\frac{\sum_{h=1}^{H} M_{h,k}\left\|\left(v_\theta(\mathbf{Y}_\tau,\tau\mid o_{1:t},x,e,z)-(\mathbf{Y}_1-\mathbf{Y}_0)\right)_{h,k}\right\|_2^2}{\sum_{h=1}^{H} M_{h,k}}$$

再对所有有效通道均匀平均:

$$\mathcal{L}_{\text{act}}=\mathbb{E}_{\tau,\mathbf{Y}_0,\mathbf{Y}_1}\left[\frac{1}{c}\sum_{k=0}^{c-1}\ell_k\right]$$

用大白话说:不管某个本体用了多少个动作通道,每个通道对总梯度的贡献都被强制均等,padding 维度彻底不参与训练。

视觉语言损失保留标准下一 token 预测,以维持并强化骨干的多模态能力:

$$\mathcal{L}_{\text{vl}}=-\sum_i \log p_\theta(w_i \mid w_{\lt i}, o_{1:t})$$

总损失为加权和 $\mathcal{L}=\lambda_{\text{act}}\mathcal{L}_{\text{act}}+\lambda_{\text{vl}}\mathcal{L}_{\text{vl}}$,同一 mini-batch 内按固定比例混合操作、导航、VL 样本联合更新骨干与动作专家。

### 2.6 四阶段渐进训练配方

论文把动作学习类比为"压缩-解压"问题:一句语言指令加具身 prompt 只有几十个 token,却要解压成成百上千维、覆盖数百步的高频动作轨迹,两者之间存在巨大维度落差;同时 VLM 骨干已充分预训练而 DiT 头是随机初始化,直接联合训练会浪费算力且不稳定。为此设计四阶段配方(Figure 2):

- **Stage I:文本到动作 DiT 预训练(T2A)**——冻结 VLM,只训练 DiT,故意不给图像,仅用文本+具身 prompt 重建动作分布。让解码器先学会"语言索引动作先验":语言决定动作空间的哪个区域,具身 prompt 决定该任务意图如何映射为平台特定的运动程序,flow-matching 动力学则在此之上学习生成过程,为后续视觉接地提供热启动。
- **Stage II:持续预训练(CPT)**——解冻骨干与 DiT,在真实+仿真混合语料(见 3.2 节)上联合训练,让 T2A 建立的动作先验在真实视觉观测中落地,产出通用具身基础模型 **Qwen-VLA-Base**。
- **Stage III:多任务监督微调(SFT)**——从 CPT checkpoint 分两条并行支线:一支多任务 SFT(VQA、空间 grounding、操作、导航,具身均衡+任务均衡采样,操作/导航动作 loss 权重 1.0、VL loss 权重 0.1);另一支在真实 ALOHA 遥操作数据上微调,检验 CPT 学到的跨域先验能否迁移到实体硬件。
- **Stage IV:强化学习(RL)**——从多任务 SFT checkpoint 出发,用 PPO+GAE(RLinf 框架,$\gamma=0.99,\lambda=0.95,\epsilon=0.2$)在单一仿真环境 SimplerEnv 上以稀疏二值成功奖励做闭环优化,产出最终模型 **Qwen-VLA-Instruct**。关键工程细节:flow-matching 策略本身没有显式概率密度,论文将确定性概率流 ODE 转换为等价 SDE,在每步 Euler 去噪中注入受控噪声,使每次状态转移成为可解析计算 log-概率的高斯分布,PPO 更新时只需重新评估速度场、无需数值 ODE 积分;每个 rollout 默认随机选一个去噪步做 log-概率估计,奖励与优势在 action-chunk 粒度($H=16$)统一分配给 GAE;价值头直接挂在 VLM 隐状态均值池化之上,对骨干做 stop-gradient,以约 20× 于 actor 的学习率单独训练。

### 2.7 预训练数据构成(Table 1,合计五大数据族)

机器人操作轨迹占 74.2%(公开数据集含 RobotSet、Galaxea、AgiBot World、RoboCOIN、RoboMIND V1/V2、RDT-1B、DROID、BridgeData V2、RH20T、RT-1、BC-Z,合计逾 10,000 小时;另加 InternData-A1、GR00T-X-Embodiment-Sim 仿真轨迹;再加超 1,000 小时 in-house 真机遥操作数据,约占总混合的 20%);人类第一视角轨迹 6.0%(Ego4D、经 VITRA 流水线处理的 EPIC-KITCHENS、EgoDex 829 小时/194 任务、EgoVerse 1,300+ 小时/1,965 任务/240 场景、Xperience-10M;每只手用 SE(3) 6 维腕部动作+10 个 PCA eigengrasp 系数,共 32 维/步);导航轨迹 7.5%(指令跟随 4.3%+目标搜索 2.3%+目标跟踪 1.0%,2FPS 采样,假设移动机器人 3 自由度);自研合成仿真轨迹 3.7%(基于 IsaacLab+cuRobo 的 ROBOINF 流水线:20 个桌面场景×10 种物体摆位=200 个基础场景,450 个短/长 horizon 任务,每任务 300 条带域随机化的轨迹,视觉条件数据含 359,848 条完整轨迹及子任务分段;另有纯语言-动作数据,6 类操作模板×6 种单臂本体≈7.2M 条轨迹、逾 14,000 小时,专供 T2A 阶段使用);以及视觉语言辅助数据合计 8.5%(细粒度具身动作字幕 0.2%即约 48,000 条视频-字幕对,由 Qwen3.6-plus 两阶段标注+人工复核生成;自动驾驶 VQA 2.4%;2D 空间 grounding 2.5%;通用视觉语言数据 3.4%)。

## 三、实验结果

### 3.1 仿真操作主结果(Table 4)

在 LIBERO、RoboCasa-GR1(双臂人形厨房)、Simpler-WidowX、RoboTwin-2.0(Easy/Hard)四个基准上,单一泛化模型对比各基准专门微调的 specialist 策略:

| 方法 | 类型 | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---|---|---|---|---|
| π0 | specialist | 94.4 | – | – | 65.9 | 58.4 |
| StarVLA-OFT | specialist | 96.6 | 48.8 | 64.6 | 50.4 | – |
| GR00T N1.6 | specialist | 97.2 | 49.9 | 63.2 | 47.6 | – |
| π0.5 | specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | specialist | 98.6 | 58.3 | – | **86.0** | **85.0** |
| Being-H0.5 | specialist | 97.6 | 53.3 | – | – | – |
| Qwen-VLA-Base | generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| **Qwen-VLA-Instruct** | generalist | **97.9** | **56.7** | **73.7** | 86.1 | 87.2 |

SFT+RL 相对 Base 的提升:LIBERO +7.1pp、RoboCasa-GR1 +16.3pp、Simpler-WidowX +9.4pp、RoboTwin-Easy +21.8pp、RoboTwin-Hard +20.8pp。

### 3.2 真实 ALOHA 双臂平台

对比"从零训练"(Qwen-VLA-aloha w/o pretrain)与"从 Qwen-VLA-Base 微调"(Qwen-VLA-aloha w/ pretrain)两个同架构变体:

| 模型 | Pick&Place | Table Cleaning | Bowl Stacking | Bowl Pick&Place | Towel Folding | Fine-grained | 均值 |
|---|---|---|---|---|---|---|---|
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| π0.5 | 73.1 | 84.6 | 88.5 | 69.2 | **80.8** | 33.3 | 71.6 |
| Qwen-VLA-aloha (w/o pretrain) | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| **Qwen-VLA-aloha (w/ pretrain)** | **96.2** | **92.3** | **98.7** | **87.2** | 65.4 | **61.5** | **83.6** |

OOD 泛化(颜色/实例/位置/背景/指令五类变化):

| 模型 | Color | Instance | Position | Background | Instruction | 均值 |
|---|---|---|---|---|---|---|
| GR00T N1.6 | 46.2 | 38.5 | 3.8 | 19.2 | 19.2 | 25.4 |
| π0.5 | 57.7 | 61.5 | 19.2 | 26.9 | 42.3 | 41.5 |
| Qwen-VLA-aloha (w/o pretrain) | 42.3 | 30.8 | 34.6 | 30.8 | 42.3 | 36.2 |
| **Qwen-VLA-aloha (w/ pretrain)** | **88.5** | **76.9** | **53.8** | **80.8** | **84.6** | **76.9** |

预训练版比 π0.5 高 35.4pp,比同架构从零训练版本高 40.7pp;背景与指令泛化提升最显著(80.8%、84.6%),说明大规模具身预训练带来的不只是同分布性能,更是对视觉/语言分布偏移的鲁棒性。

### 3.3 视觉语言导航(VLN-CE,Table 7)

| 方法 | R2R NE↓ | R2R OS↑ | R2R SR↑ | R2R SPL↑ | RxR NE↓ | RxR SR↑ | RxR SPL↑ | RxR nDTW↑ |
|---|---|---|---|---|---|---|---|---|
| NaVid | 5.7 | 49.2 | 41.9 | 36.5 | 5.7 | 45.7 | 38.2 | – |
| Uni-NaVid | 5.6 | 53.3 | 47.0 | 42.7 | 6.2 | 48.7 | 40.9 | – |
| NaVILA | 5.2 | 62.5 | 54.0 | 49.0 | 6.8 | 49.3 | 44.0 | 58.8 |
| StreamVLN | **5.0** | 64.2 | 56.9 | **51.9** | 6.2 | 52.9 | 46.0 | **61.9** |
| Qwen-VLA-Base | 5.2 | 61.7 | 53.8 | 49.4 | 6.4 | 55.1 | 45.8 | 56.2 |
| **Qwen-VLA-Instruct** | 5.1 | **69.0** | **57.5** | 51.2 | **5.8** | **59.6** | **47.8** | 57.1 |

Qwen-VLA-Instruct 在 R2R Val-Unseen 上 Oracle Success(OS)和 Success Rate(SR)均为最优,分别超 StreamVLN 4.8pt 和 0.6pt;RxR Val-Unseen 上 SR、SPL 均领先,但 nDTW 略逊于 StreamVLN(57.1 vs 61.9)。操作与导航联合训练下,导航性能仍保持在同量级最优水平。

### 3.4 分布外操作(静态 & 动态)

SimplerEnv-OOD(6 个未见任务类型,仅在简单 pick-and-place 上做过微调):

| 方法 | MoveAway | MoveRight | PlaceNear | PlaceRight | PutFront | StackYellow | 均值 |
|---|---|---|---|---|---|---|---|
| π0.5 | 26.1 | 0.0 | 0.0 | 32.1 | **13.0** | 4.2 | 12.6 |
| Qwen-VLA-Base | 31.3 | 31.6 | 16.7 | 47.1 | 6.3 | 18.8 | 25.3 |
| **Qwen-VLA-Instruct** | **43.8** | **33.3** | **39.6** | **47.9** | 4.2 | **22.9** | **32.0** |

DOMINO 动态操作基准(35 个 suite,零样本设定,仅用当前帧观测、无动态操作数据微调):

| 方法 | 设定 | SR(%) | MS |
|---|---|---|---|
| PUMA | DOMINO 专门微调 | 17.2 | 35.0 |
| StarVLA-OFT | 专门微调 | 10.9 | 30.5 |
| OpenVLA-OFT | zero-shot | 6.7 | 20.0 |
| π0.5 | zero-shot | 7.5 | 20.4 |
| LingBot-VA | zero-shot | 24.1 | 36.1 |
| Qwen-VLA-Base | zero-shot | 21.1 | 37.4 |
| **Qwen-VLA-Instruct** | zero-shot | **26.6** | **39.5** |

Qwen-VLA-Instruct 不仅是零样本类别里的最优,MS 指标还超过专门在 DOMINO 上微调、并使用了额外时序运动输入的 PUMA(+4.5),SR 超 PUMA 9.4pp,论文将其归因于统一动作-轨迹预训练带来的可迁移空间-运动学先验。

### 3.5 关键消融

- **T2A 设计(Fig 6)**:T2A 语料中约 20% 语言-only 合成数据+80% 真实数据(视觉丢弃)效果最好,SFT 成功率达 71.1%,比不做 T2A 的基线(60.9%)高 10.2pp;全序列预测始终优于 chunk 预测(10% 合成数据下 +4.9pp);T2A 阶段引入图像反而有害(−2.9pp),确认应彻底剥离视觉、逼迫解码器建立语言-动作先验;flow-matching 时间步分布上,T2A 用 Sigmoid-Normal、CPT/SFT 用 Beta 的组合最优(71.1%),两阶段都用 Beta 或都用 Sigmoid-Normal 均会掉点;T2A 训练步数在 2,000 步时达峰值,40,000 步会因对 T2A 语料过拟合而降至 60.4%。
- **VL 数据是否助益动作学习(Fig 7a)**:在需要细粒度物体识别与组合指令解析的 RoboCasa-GR1、RoboTwin-2.0 上,混入 VL 数据分别带来 +4.9pp(51.1%→56.0%)和 +4.6pp(81.8%→86.4%)提升;在较简单的 LIBERO、Simpler-WidowX 上两者持平,无负迁移。
- **预训练 DiT vs 随机初始化 DiT(Fig 7b)**:预训练 DiT 收敛更快、峰值更高。
- **异质动作空间投影设计(Table 10)**:Multi-MLP、Concatenation、Zero-Padding 三种方案性能差距均小于 1.2pp,均不逊于单具身独立训练;Zero-Padding 参数量最省,被定为默认方案。
- **RL 后训练的累积效果(Table 11)**:CPT→+SFT→+RL 逐阶段提升,RL 收益最大的是训练所在环境 SimplerEnv(70.8%→73.7%,+2.9pp);未参与 RL rollout 的基准(RoboCasa、RoboTwin、LIBERO、DOMINO)性能保持或小幅提升(如 DOMINO SR 25.7%→26.6%、MS 39.1%→39.5%),未观察到灾难性遗忘。
- **状态条件化(Table 12,RoboTwin-2.0)**:无本体状态输入 88.7/87.4,状态编入 VLM prompt 89.3/88.7,状态直接输入 DiT 89.4/88.3——三者差距 ≤1.3pp,收益边际;论文最终决定不引入显式状态输入,仅靠具身 prompt 作为唯一平台特定接口。

## 四、局限性

论文在第 7 节明确列出三点局限:第一,具身动作数据规模与多样性仍远小于视觉语言预训练数据,限制了模型对长尾物体、长尾环境与接触密集型交互的鲁棒性;第二,视觉语言理解、导航、动作生成三者联合训练存在优化权衡——以动作为导向的训练会轻微削弱部分纯视觉语言与导航评测表现,说明目标权重、数据课程、模块化专精之间仍需更好的平衡;第三,当前评测仍以短 horizon、benchmark 驱动为主,长时程、易失败的真实世界部署仍是未解难题。

此外从方法设计看还存在几点隐性局限:RL 阶段仅在单一仿真环境(SimplerEnv)、稀疏二值奖励下进行,虽然论文验证了收益能迁移到未参与训练的基准,但奖励信号本身的稀疏性与单一环境来源,使得该 RL 配方在更复杂长时程任务或真实世界闭环场景下的可扩展性尚未被验证;具身感知 prompt 依赖手写文本模板描述平台与控制惯例,对于训练时完全未见过的新机器人形态,其零样本迁移能力仅在同一 ALOHA/仿真平台内的物体-背景-指令层面做了验证,并未在全新硬件本体上系统评测。

## 五、评价与展望

**优点**:论文将操作、导航、人类第一视角示范、轨迹预测统一进同一动作-轨迹张量空间,并用轻量的文本 prompt(而非为每个具身单独设计输出头或适配器)承载平台差异,这一设计在工程上具备较好的可扩展性——加入新机器人本体理论上只需扩充 prompt 词表与数据,无需改动模型结构。T2A→CPT→SFT→RL 的分阶段配方针对性地解决了"预训练 VLM 骨干"与"随机初始化动作专家"两个模块优化状态不对称的冷启动问题,消融实验(T2A 数据配比、预测模式、时间步分布、训练步数)论证充分,是本文除主结果外最具参考价值的部分。实验覆盖面广(4 个操作基准+2 个真实平台评测+2 个导航基准+2 个 OOD 基准),且以"单一泛化模型 vs. 多个专用 specialist"的方式呈现,论证了联合训练在大多数场景下不牺牲、甚至提升单任务性能。

**与其他公开工作的关系**:相较 π0/π0.5(Physical Intelligence)与 GR00T N1(NVIDIA)等同样采用 VLM+flow-matching/diffusion 动作头架构的通用 VLA 路线,Qwen-VLA 的差异化在于更彻底的跨任务统一(把导航与人类第一视角数据也纳入同一动作-轨迹空间)以及显式的语言先验预训练阶段(T2A);相较 RDT-1B 等专注操作的双臂基础模型,Qwen-VLA 覆盖的本体与任务谱系更宽;相较 StreamVLN 等专用导航 VLA,Qwen-VLA 在导航指标上整体持平或领先,说明"操作-导航联合训练不必然牺牲导航专精"。论文自陈其路线与"以视觉预测为中心的世界模型"方法不同——不显式建模未来视觉状态,而是把多模态理解直接接到可执行动作上,这一设计取舍(动作可执行性 vs. 更丰富的前向动力学先验)是该子领域一个值得持续讨论的开放问题。

**开放问题与可能的改进方向**:(1)RL 阶段目前局限于单一仿真器与稀疏奖励,后续可探索多仿真环境混合 rollout、更稠密或学习式奖励模型,以及真实世界在线 RL 的可行性;(2)具身感知 prompt 本质上仍是人工设计的离散模板,能否用学习到的具身嵌入或少样本适配替代手写模板、以支持真正未见过的机器人形态零样本上线,是一个开放问题;(3)论文承认长时程、失败恢复场景仍未被现有 benchmark 充分覆盖,结合情景记忆(episodic memory)与世界建模的后续工作方向与近期多篇具身基础模型论文(如 Gemini Robotics、Being-H0.5)的展望方向趋同,可能是下一步的必争之地;(4)状态条件化消融显示本体感受信息收益有限,这一负结果对后续"是否需要显式 proprioception 输入"的架构设计具有参考价值,值得在更多任务/本体上复验。

## 参考

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.
- Black et al. *π0.5: A Vision-Language-Action Flow Model for General Robot Control*, Physical Intelligence Technical Report, 2025.
- NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, arXiv:2503.14734, 2025.
- Liu et al. *RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation*, ICLR, 2025.
- Wei et al. *StreamVLN: Streaming Vision-and-Language Navigation via Slowfast Context Modeling*, arXiv:2507.05240, 2025.
