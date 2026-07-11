# MV-WAM：流形感知的价值增强世界动作模型

> **论文**：*MV-WAM: Manifold-Aware World Action Model with Value Augmentation*
>
> **作者**：Jintao Chen, Peidong Jia, Qingpo Wuwu, Jiaming Liu, Mengfei Du, Chun-Kai Fan, Xiaowei Chi, Hao Chen, Chengyu Bai, Zezhong Qian, Hao Wang, Jiajun Cao, Weishi Mi, Xiaozhu Ju, Jian Tang, Shanghang Zhang（通讯作者）et al.
>
> **机构**：北京大学计算机学院多媒体信息处理国家重点实验室；北京人形机器人创新中心（Beijing Innovation Center of Humanoid Robotics）
>
> **发布时间**：2026 年 06 月（arXiv 2606.21088）
>
> **发表状态**：未录用（预印本，作者标注 "Preprint"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.21088) | [PDF](https://arxiv.org/pdf/2606.21088)
>
> **分类标签**：`world-action-model` `流形感知优化` `Mixture-of-Transformers` `分布外泛化` `价值引导回滚`

---

## 一句话总结

MV-WAM 指出统一视频-动作世界动作模型（WAM）的域外（OOD）泛化增益之所以跟不上域内增益,根源在于视觉流形（高曲率,$\kappa_v=3.8\pm0.6$）与动作流形（低曲率,$\kappa_a=1.3\pm0.2$）的几何失配；据此提出流形感知的双专家 MoTs 架构（视频用速度场预测、动作用直接 $x_0$ 回归)+跨模态因果掩码+蒙特卡洛价值回滚三件套,在 RoboTwin 2.0 的 Random 设定上取得 **55.7%** 平均成功率(领先最强基线 HALO 29.3 个百分点),真机四任务平均 **77.5%**。

## 一、问题与动机

Vision-Language-Action（VLA）模型受限于示范数据的视觉多样性,难以泛化到未见纹理、光照、背景。World Action Models（WAM）试图用海量无动作标注的视频数据补齐这一缺口,但作者观察到一个反常现象：现有 WAM(无论是 decoupled、fully unified 还是基于 Mixture-of-Transformers 的设计)在域内性能上持续爬升,域外增益却相对滞后,有时差距甚至在扩大(附录 D.1 给出实证)。

论文将这一现象归因于**流形失配**(manifold mismatch)：视觉观测是高维、稠密结构化的,受空间与光度规律支配；机器人动作是低维、时序结构化、精确性要求高的控制信号。两者天然存在异质表征流形,若用同一目标联合优化,会压制动作学习并放大分布偏移下的脆弱性——视觉 token 体量大,天然主导共享目标,这种失配在域内被强跨模态相关性掩盖,一旦分布偏移就会通过共享参数传导并损害动作预测。核心问题因此被表述为:**为什么更深的视频-动作耦合没有转化为更强的动作泛化?**

为验证这一假设,作者做了两组诊断实验(附录 D)：

- **t-SNE 表征分析**：Video Expert 在 clean/random 条件下的特征仍然高度混杂(视觉表征对扰动鲁棒),而 Action-Value Expert 的特征在同样扰动下分裂成两个明显分离的簇(动作表征对分布偏移敏感)——说明共享隐空间联合优化"必要但不充分"。
- **流形曲率估计**：用测地距离/欧氏距离比值在原始数据空间(与模型、训练过程无关)估计曲率,50 个任务上视觉曲率 $\kappa_v=3.8\pm0.6$ 显著高于动作曲率 $\kappa_a=1.3\pm0.2$(Mann-Whitney U 检验 $p \lt 0.001$),证实两模态几何差异是内禀的而非训练产物。

## 二、核心方法

### 2.1 理论动机:曲率决定泛化界

作者引用流形上 Lipschitz 函数的泛化误差界(用于给策略网络 $\pi_\theta$ 的分析),对定义在截面曲率为 $\kappa$ 的紧致黎曼流形上的 $L$-Lipschitz 函数,泛化误差承受一个曲率相关的惩罚项

$$\psi(\kappa,L) = \sqrt{|\kappa|}/L,$$

且该界只有在优化目标与目标流形的几何**匹配** 时才会收紧。现有统一 WAM 用单一速度场目标(校准到高曲率视觉流形 $\mathcal{M}_v$)同时约束两个分支,由于联合优化,整体 Lipschitz 正则性被更复杂的视觉几何主导,动作分支实际承受的曲率从其原生的 $\kappa_a$ 被"抬升"到 $\kappa_v$。因为 $\psi$ 随 $|\kappa|$ 单调递增,这一结构性失配使动作侧的泛化惩罚 $\psi(\kappa_v,L) \gg \psi(\kappa_a,L)$,分布偏移下直接表现为动作鲁棒性的退化。

**大白话**:视觉像连绵起伏的山地,动作更像平坦的操场;用同一把"爬山尺"去度量操场上的偏差,尺子的刻度天然会被山地的复杂度带偏,导致对平坦流形(动作)的泛化承诺变得远比实际情况更松、更不可靠。

### 2.2 架构:Manifold-Aware MoTs 双专家

MV-WAM 建立在 Mixture-of-Transformers(MoTs)骨架上,每层 DiT block 内并列两个模态专属专家、共享全局注意力空间但参数独立:

- **Video Expert**：从 WoW-1.3B(预训练视频生成模型)初始化,30 层 DiT。原始视频先经时空 VAE 压缩为紧凑 latent,与掩码 token 沿通道维拼接,patch 化为视觉 token 送入骨架做视频扩散建模。
- **Action-Value Expert**：同样 30 层 DiT,但隐维度更低(768,仅在共享自注意力内投影到 1536 维视觉 token 空间以省参数)。多视角视觉状态 $s_t$ 由 SigLIP2 编码作为视觉条件；语言 embedding 注入奇数层,SigLIP2 视觉 embedding 注入偶数层。动作 token 与状态 token 共享一套编码器-解码器做动作生成,价值 token 用独立的编码器-解码器单独做价值估计。
- 两专家的任务指令均由 umT5 文本编码器统一编码,通过交叉注意力注入。
- **RoPE 时间对齐**：Action-Value Expert 的 RoPE 缩放因子设为 Video Expert 的 1/4,建立动作 token 与视频 latent 之间统一的时间网格。

除 umT5、SigLIP2 编码器外,整体参数量约 **1.9B**。

### 2.3 跨模态因果掩码

为规范信息流向、保留模态专属先验,MV-WAM 引入结构化因果注意力掩码:

- 视频 token 只在视觉流内部做自注意力(维持高保真视频生成,不被动作信号污染)；
- 动作/状态 token 可同时关注视觉上下文与其他动作 token(动作生成始终以预测的视觉未来为条件)；
- 价值 token 联合关注视觉、动作、状态三者(保证任务进度评估拥有完整跨模态上下文)。

**大白话**:视频专家"闭门造车"专心想象未来画面,动作专家"看图说话"根据这幅想象图来决定怎么动,价值专家则是"监工",同时看图纸和实际动作来打分,三者分工明确、互不越界。

### 2.4 流形感知的双预测目标

不同于单一统一扩散目标,MV-WAM 为两分支分别定制目标,均用 flow matching 作为生成骨架。给定干净数据 $z_0$/$a_0$、噪声先验 $z_1$/$a_1$、扩散时间 $\tau\in[0,1]$,插值 $z_\tau=(1-\tau)z_0+\tau z_1$。

Video Expert(高曲率视觉流形,标准速度场预测)：

$$\mathcal{L}_v = \mathbb{E}_{\tau,z_0,z_1}\left\|\pi_\theta^v(z_\tau,\tau,c) - (z_1-z_0)\right\|_2^2$$

Action-Value Expert(低曲率动作流形,直接回归干净端点 $a_0$,等价于 $x_0$-prediction)：

$$\mathcal{L}_a = \mathbb{E}_{\tau,a_0,a_1}\left\|\pi_\theta^a(a_\tau,\tau,c) - a_0\right\|_2^2$$

整体目标为 $\mathcal{L}=\lambda_v\mathcal{L}_v+\lambda_a\mathcal{L}_a$,两权重经验设为 $\lambda_v=\lambda_a=1$。附录 F 进一步说明:若把动作分支也套用速度空间的 $x_0$-loss,等价于一个被 $1/t^2$ 重新加权的端点回归损失,当 $t\to0$ 时权重发散,会让低维动作分支的梯度压过视频损失、破坏联合训练平衡;因此改用未加权的 $x_0$-prediction 直接监督动作端点。

**大白话**:视频分支按老办法预测"该往哪个方向变"(速度),动作分支则直接告诉网络"目标动作长什么样"(端点),各自用最适合自己几何形状的方式学习,谁也不用迁就谁。

### 2.5 Progress-Valued Regulation(进度价值调节)

流形感知目标只保证训练时的几何对齐,推理时仍需要显式信号判断任务进度、检测视觉-动作错位。为此引入价值 token,用蒙特卡洛(MC)回报做监督:

$$\hat{c}(s_t,a_t) = \sum_{i=0}^{H-t} \gamma^i r(s_{t+i}, a_{t+i}), \quad \gamma=0.99$$

推理时维护一个滚动缓冲区记录最近的状态-动作-价值三元组,并跟踪历史峰值 $c_{\text{peak}}=\max_{i<t}\hat{c}_i$。若当前预测值显著低于峰值,即 $\hat{c}_t \lt c_{\text{peak}}-\delta$($\delta$ 为任务相关阈值),系统判定当前动作块可能出错,触发 **rollback-and-resample**：把执行状态回退到 $c_{\text{peak}}$ 对应的缓存检查点,丢弃有问题的动作序列,从该高进度状态重新初始化扩散采样、生成新的动作块。这一机制利用扩散采样本身的随机性实现纠错,无需重跑整条轨迹或额外查询环境。

**大白话**:价值 token 像一个"进度条读数器";一旦发现进度条突然倒退,就把执行状态"读档"回上一次进度最高的存档点,再"重掷骰子"生成一段新动作,防止误差在长时程任务中累积滚雪球。

### 2.6 训练配置

两阶段训练:Stage 1 只预训练 Video Expert,20k 步,batch size 1024,lr $1\times10^{-4}$,输入分辨率 240×320;Stage 2 联合微调完整模型(预训练 Video Expert + 从头训练的 30 层 Action-Value Expert),15k 步,batch size 512,lr $1\times10^{-4}$,动作块长度 32。学习率前 20% 步数保持恒定后线性衰减至零。单张 A800 上推理速度 **17.9 actions/s**。预训练语料约 7,500 小时,来自 EgoDex(800h)、Agibot(2500h)、RoboMind(1000h)、RoboCoin(1000h)、Open X-Embodiment(2000h)及少量内部私有数据(200h)。动作空间沿用 RDT-1B 的统一动作设计,$a_t\in\mathbb{R}^{128}$(双臂场景为 14-DoF 相对关节偏移 + 2-DoF 独立夹爪状态标准化而成);真机实验中改用 16 维动作 $[\Delta\theta_{1:7}^R, g^R, \Delta\theta_{1:7}^L, g^L]$。

## 三、实验结果

### 3.1 RoboTwin 2.0 仿真基准(50 任务,Aloha AgileX 双臂)

VLA 基线单阶段微调(50 条 Clean 示范/任务);WAM 基线与 MV-WAM 采用两阶段协议(沿用 BagelVLA):Video Expert 先在 50 Clean + 500 Random 混合示范上预训练,再仅用 Clean 配对示范微调整套模型。每任务 Clean/Random 各评测 100 次 rollout。

| 方法 | 类型 | Clean 平均 SR | Random 平均 SR | Clean→Random 相对下降 |
|---|---|---|---|---|
| DP | VLA | 28.0% | 0.6% | — |
| RDT | VLA | 34.5% | 13.7% | — |
| $\pi_0$ | VLA | 46.4% | 16.3% | — |
| UP-VLA | WAM | 52.9% | 15.2% | — |
| Fast-WAM | WAM | 71.9% | 6.3% | 91.2% |
| BagelVLA | WAM | 75.3% | 20.5% | 72.8% |
| HALO | WAM | 80.5% | 26.4% | 67.2% |
| **MV-WAM(本文)** | WAM | **84.0%** | **55.7%** | **33.7%** |

MV-WAM 在 Clean/Random 两种设定下均取得最优,较最强基线 HALO 分别领先 3.5pp(Clean)与 **29.3pp**(Random),较 Fast-WAM 领先 12.1pp/49.4pp。更关键的是 Clean→Random 的相对性能下降(33.7%)显著小于三个 WAM 基线(67.2%-91.2%),说明几何匹配的预测目标带来的鲁棒性提升不是靠域内分数堆出来的。附录中列出的逐任务结果显示,MV-WAM 在 *Blocks Ranking RGB* 等任务上 Random SR 达 88%,而多数基线在同任务上的 Random SR 接近 0。

### 3.2 零样本/少样本泛化(10 个未见任务)

在 40 个任务(共 22,000 条示范)上训练后,直接在 10 个未见任务上评测:

| 设定 | Clean SR | Random SR |
|---|---|---|
| 0-shot | 55.6% | 54.0% |
| 1-shot | 57.3% | 56.3% |
| 5-shot | 76.2% | 75.7% |
| 10-shot | 77.1% | 74.6% |

零样本即达到 55.6%/54.0%,10-shot 微调后逼近域内水平,显示 MV-WAM 学到的是可迁移的操作先验而非任务记忆。

### 3.3 真机实验(TienKung 双臂人形机器人,4 项日常任务)

每任务 100 条人类遥操示范,10 次试验评测,对比 $\pi_0$、RDT(均用相同示范集微调)：

| 方法 | Pick Backbag & Coffee | Drop Cloth | Pick Cloth | Fold Cloth | 平均 SR |
|---|---|---|---|---|---|
| $\pi_0$ | 50% | 60% | 50% | 10% | 42.5% |
| RDT | 30% | 40% | 60% | 0% | 32.5% |
| **MV-WAM** | **90%** | **100%** | **100%** | **20%** | **77.5%** |

MV-WAM 在结构化任务(Drop Cloth、Pick Cloth)上达到满分,在最难的精细可变形操作任务 Fold Cloth 上也是唯一超过 10% 的方法(20% vs $\pi_0$ 10%、RDT 0%)。论文强调这一优势是在参数量显著小于同类 WAM 竞品的情况下取得的,支持"流形感知设计而非模型规模"是泛化的关键驱动力这一论断。

### 3.4 消融研究

- **预测目标(流形感知 vs 统一目标)**：动作分支用速度预测(flow-pred)或噪声预测(noise-pred)替代 $x_0$-prediction 后,Random SR 分别崩溃到约 6% 与约 9%(相对最优配置 55.7% 大幅下降),验证了模态专属目标的必要性。
- **去噪步数**：仅 3 步已恢复 5 步时 96% 以上的 Clean 性能,同时降低 40% 去噪开销(Clean/Random:1 步 66.5%/36.4%,2 步 78.7%/50.6%,3 步 81.2%/54.2%,5 步 84.0%/55.7%);默认取 5 步。
- **价值 token 与回滚阈值 $\delta$**：加入价值 token 后相对无价值基线(83.4%/54.5%)有一致提升;$\delta\in\{0.10,0.15,0.20,0.30\}$ 区间内性能波动很小(84.0%/55.7% @ $\delta{=}0.20$),说明回滚机制对超参不敏感,默认取 $\delta=0.20$。

## 四、局限性

论文自陈两点局限:(1)尚未在更大模型规模上验证,提出的机制能否随规模扩展仍是开放问题;(2)蒙特卡洛价值估计在稀疏奖励下可能噪声较大,可能限制长时程任务中回滚机制的可靠性。此外从失败案例分析(附录 G)看,还存在两类未解决的错误模式:一是 Video Expert 的语义/空间理解局限导致预测视频"动作方向对但操作臂选错"(如 *Adjust Bottle* 任务错用右手而非左手);二是接触密集场景下动作精度不足,预测的交互方向正确但末端定位精度不够(如 *Move Can Pot* 任务)。这两类失败提示当前架构在细粒度接触控制与具身相关的空间推理上仍有改进空间。

## 五、评价与展望

**优点**：本文最有价值的贡献是把"WAM 的域外泛化增益为何跟不上域内"这一此前被普遍观察到但缺少系统解释的现象,转化为一个可测量、可证伪的几何假设(视觉/动作流形曲率差异),并用测地-欧氏距离比值这种与模型无关的方式做了独立验证,而非仅靠消融实验反推。理论上借用 Riemannian 流形 Lipschitz 泛化界解释统一目标为何会在动作分支上产生更松的泛化保证,把"MoTs 该不该用相同扩散参数化"这一此前工程经验性的设计选择,提升为有理论支撑的原则(视频用速度场、动作用直接端点回归)。价值引导回滚机制则把 world model 的"未来想象"能力进一步用作在线自我纠错信号,是对纯离线训练价值信号(如 Cosmos Policy 的 MC 回报思路)的一次在线闭环化尝试。

**局限与开放问题**：其一,流形曲率的度量本身依赖测地距离的 k 近邻图近似与 PCA 降维,是否严格反映"内禀几何"而非数据采样密度的伪影,论文未做更细致的敏感性分析(比如近邻数、任务多样性对 $\kappa$ 估计的影响)。其二,理论部分借用的曲率-Lipschitz 泛化界([34] Sarkar 2025)本身是一般性结果,论文将其套用到"策略网络"这一具体场景时的假设(如 $L$-Lipschitz 常数是否真的与曲率解耦)并未给出更严格的验证,更多是启发式论证而非严格证明,这与论文标题中"formal analysis"的表述之间存在一定张力。其三,与同期 WAM 工作(HALO、BagelVLA、Fast-WAM)相比,MV-WAM 的两阶段训练协议、动作空间设计(沿用 RDT-1B)、评测基准(RoboTwin 2.0)高度一致,这使得比较相对公平,但也意味着改进主要来自"预测目标解耦+因果掩码+价值回滚"这一组合拳,三者对最终提升的边际贡献尚未完全解耦——消融虽证明了预测目标是大头,但因果掩码本身的独立贡献未见单独消融。其四,价值回滚依赖任务相关阈值 $\delta$ 与 MC 回报的奖励函数设计,在真实世界稀疏、噪声奖励下的鲁棒性(论文亦自陈)仍待验证,回滚触发的"cached checkpoint"机制在真实物理系统上如何低成本实现状态回退也未详细展开。总体而言,MV-WAM 提供了一个从几何视角重新审视多模态统一建模的有益范式,其"模态匹配的生成式目标+价值驱动测试时纠错"思路,可能为后续 WAM 设计(尤其是如何在不牺牲动作精度的前提下利用视频先验)提供可复用的设计原则,但流形几何刻画本身的精细化与理论保证的严格化仍是值得深挖的方向。

## 参考

- Liang et al. *Mixture-of-Transformers: A sparse and scalable architecture for multi-modal foundation models.* arXiv:2411.04996, 2024.（本文所用 MoTs 骨架的直接来源）
- Chi et al. *WoW: Towards a world omniscient world model through embodied interaction.* arXiv:2509.22642, 2025.（Video Expert 初始化所用的预训练视频生成模型 WoW-1.3B）
- Liu et al. *RDT-1B: A diffusion foundation model for bimanual manipulation.* arXiv:2410.07864, 2024.（本文统一动作空间设计的来源）
- Zhang et al. *UP-VLA: A unified understanding and prediction model for embodied agent.* arXiv:2501.18867, 2025.（对比的 WAM 基线之一）
- Kim et al. *Cosmos Policy: Fine-tuning video models for visuomotor control and planning.* arXiv:2601.16163, 2026.（本文蒙特卡洛价值估计范式的启发来源）
