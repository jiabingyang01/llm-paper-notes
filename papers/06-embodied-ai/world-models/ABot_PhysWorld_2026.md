# ABot-PhysWorld：面向机器人操作、带物理对齐的交互式世界基础模型

> **论文**：*ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment*
>
> **作者**：Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Xinyuan Chang, Feng Xiong, Mu Xu et al.
>
> **机构**：AMAP CV Lab, Alibaba Group（高德地图视觉实验室,阿里巴巴集团）
>
> **发布时间**：2026 年 03 月（arXiv 2603.23376,v2 于 2026-03-27,正文日期 2026-03-20）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.23376) | [PDF](https://arxiv.org/pdf/2603.23376)
>
> **分类标签**：`world-model` `robotic-manipulation` `physics-alignment` `diffusion-DPO` `action-conditioned-generation`

---

## 一句话总结

在 14B 视频扩散 Transformer（Wan2.1-I2V-14B）上,通过"物理感知数据筛选 + 解耦 VLM 判别器驱动的 Diffusion-DPO 偏好对齐 + 并行 context block 动作注入"三件套,把视频世界模型从"画得像"推进到"物理上说得通且可动作控制",在 PBench 机器人子集上 Domain Score 达 0.9306、总分 0.8491,超过 Veo 3.1 与 Sora v2 Pro,并配套发布首个训练无关的零样本具身评测集 EZSbench。

## 一、问题与动机

视频世界模型可作为 VLA 策略的模拟器、可解释的轨迹预览器,或直接充当 World Action Model 预测"动作条件下的动态"。但当前 SOTA 视频生成模型(Veo 3.1、Sora v2 Pro)在操作序列中频繁违反基础物理:物体穿模(penetration)、无接触抓取(contactless grasping)、反重力运动、不自然形变。作者把根因归为两点:

1. **训练数据缺乏具身交互信号**:在通用视觉数据上训练,学不到摩擦、碰撞响应、质量分布等细粒度物理动态。
2. **训练目标是"一视同仁"的最大似然**:标准 MLE 对所有预测误差同等对待,无法区分"物理上合法"与"物理上违规"的转移。

此外,现有评测集大多与训练同分布,偏重视觉质量或 in-distribution 精度,缺乏对物理一致性和零样本泛化的严格考察,且常用单一 VLM 既出题又判分,存在自评偏差。

本文的目标:构建一个"视觉真实、物理合理、动作可控"的具身世界基础模型,并给出一个训练无关、跨具身、可解耦评估物理与动作的基准。

## 二、核心方法

整体三大支柱:**数据策管流水线** → **两阶段训练(SFT + 物理偏好对齐 DPO)** → **动作条件生成(A2V)**;外加评测集 **EZSbench**。

### 1. 数据策管流水线(约 300 万真实操作片段)

底料整合五个公开数据集:AgiBot、RoboCoin、RoboMind、Galaxea、OXE。作者指出 Cosmos-Curate / VideoX-Fun 等通用流水线不适配具身数据(依赖场景切分检测器、偏重美学而非物理因果),故自建三阶段:

- **具身专用过滤**:视频级质量门(丢弃异常分辨率/移动相机,序列限 80–500 帧,超长按 task index 切分);基于 Farnebäck 稠密光流(2 FPS 灰度帧)的运动过滤,剔除近零运动或非物理振荡;基于 CLIP(8 等距帧、768D 特征)的时序连贯性检测,剔除黑屏/切换/拼接错误;**视觉-动作对齐核验**——把标定后的动作(关节角、末端位姿、夹爪状态)投影到视频帧上,用 Qwen3-VL 核验时空对齐,过滤传感器标定/同步误差导致的错配。
- **分层分布均衡**(四级动态采样):① 数据集内多样性保全(OXE 内部小子集整体保留);② 跨机器人再平衡(欠表达的机器人本体上采样);③ 任务感知配额——头部任务(高数据量)封顶在原始规模的 8–15% 抑制过拟合,中量任务均匀采样 40–50%,长尾稀有任务全保留;④ 宏数据集尺度调控(大集如 AgiBot/OXE 均匀下采样、微集如 RoboMind 保证最低覆盖,三轮补充策略填补残缺配额)。
- **物理感知视频字幕**:两阶段流水线,Qwen3-VL 32B 做感知模块抽取结构化物理属性(机器人形态、被操作物属性、空间布局、接触事件、状态转移),Qwen3 32B FP8 做写作模块生成四阶段字幕(Scene Setup / Action Detail / State Transition / Camera Summary)。强调"为动作而标注",覆盖宏观任务意图、中观动词-名词技能分段、微观笛卡尔轨迹与夹爪状态、场景级物理关系(contact/support/containment)与因果(重力致落、表面形变、接触力),并用少样本正负例 + 动态词表 + "只描述可见事实"基线抑制空间关系幻觉。

### 2. 骨干与两阶段训练

**骨干**:基于 Wan2.1-I2V-14B(14B DiT),在策管后的具身数据上全量微调。Stage 1 即 SFT,给定首帧观测 + 文本指令预测未来帧。

**Stage 2 物理偏好对齐**——SFT 只教模型复现训练分布,无法区分物理正确与否,故引入偏好对齐:

**(a) 解耦 VLM 判别器**:为同一 prompt $x$ 与首帧生成 $N$ 个候选视频。为避免"同一模型既出题又判分"的自评幻觉,把评估拆成两个角色:

- **提议者(proposer)= Qwen3-VL 32B Thinking**:看首帧 + 指令,动态生成任务专属的物理检查清单。清单分层:Tier 1 是致命违规(穿模、反重力),采用"单票否决(single-vote veto)";Tier 2 是微观物理保真与接触动态,用于区分合规样本。并刻意构造正负问题的均衡混合,防止判分模型谄媚性地一律预测"无违规"。
- **打分者(scorer)= Gemini 3 Pro**:用显式 CoT(全局扫描、标注可疑帧、回溯确认)对 $N$ 个候选逐条核对清单。为在 $\mathcal{O}(N)$ 复杂度内高效选出最优 $y_w$ 与最差 $y_l$,采用两阶段锦标赛采样:淘汰赛先选最优,败者组再选最差,避免全排列比较,得到边界清晰的 DPO 三元组 $\langle x, y_w, y_l \rangle$。

**(b) Diffusion-DPO 训练**:给定三元组 $(c, v_w, v_l)$($c$ 为条件,$v_w$ 合规、$v_l$ 违规),在潜空间对视频扩散模型做偏好微调。对潜变量 $z$ 加噪 $\epsilon\sim\mathcal{N}(0,I)$、时步 $t\sim\mathcal{U}(0,T)$ 得 $z_t$,单步去噪 MSE 记 $L(\theta,z)=\lVert\epsilon_\theta(z_t,t,c)-\epsilon\rVert_2^2$。设策略模型 $\pi_\theta$ 与参考模型 $\pi_{ref}$(即 SFT 基线)的去噪误差为 $L_\theta$、$L_{ref}$,物理偏好对齐损失为:

$$\mathcal{L}_{DPO} = -\mathbb{E}_{z,\epsilon,t}\Big[\log\sigma\Big(-\frac{\beta}{2}\big[(L_\theta(z_w)-L_\theta(z_l)) - (L_{ref}(z_w)-L_{ref}(z_l))\big]\Big)\Big]$$

其中 $\beta$ 控制与参考分布的偏离(实现取 $\beta=5000$)。**用大白话说**:让模型对"合规视频"的预测误差比参考模型降得更多、对"违规视频"的误差升得更高——即在每个时步都主动把概率质量从违规转移到合规。

**显存工程**:14B DiT 若同时维护 $\pi_\theta$ 与 $\pi_{ref}$ 两套完整计算图会 OOM。解法是冻结 DiT 主干,只在自注意力(Q/K/V/输出)与 FFN 层注入 rank=64 的 LoRA;计算 $L_{ref}$ 时临时关闭 LoRA 权重即可复用同一套权重充当参考模型,零额外显存。

### 3. 动作条件生成(A2V)

世界模型要支持"给定当前观测 + 未来动作序列 → 生成忠实跟随该轨迹的物理合理视频"。低维机器人指令(如末端位姿)直接注入高维视觉管线会产生语义鸿沟,故:

- **动作图构造(Action Map)**:输入是 7D 动作向量 $\boldsymbol{a}\in\mathbb{R}^7$(3D 位置、3D 朝向、夹爪开合度),双臂扩展到 14D。用相机内外参把 3D 位置投影到 2D 中心 $(u,v)$;朝向编码为旋转矩阵的三条主轴,投影到像平面画成彩色箭头,箭头长度编码深度;夹爪状态映射为 $(u,v)$ 处的圆形 mask,不透明度线性表示开合度;双臂用红/蓝通道区分,构成多通道动作图。
- **动作注入**:既有方法要么用 AdaLN 注入 MLP 编码动作(阻碍跨本体泛化),要么把动作图直接拼进带噪潜变量(致预训练物理先验灾难性遗忘)。本文借鉴 VACE,从主 DiT 克隆一组**并行 context block** 处理动作图,其输出经零初始化卷积层残差加回对应主 DiT block:

$$\mathbf{x}_i = \text{DiT}_i(\mathbf{x}_{i-1}) + \alpha \cdot W_{\text{zero}}^{(i)}\,\mathbf{h}_i$$

其中 $\mathbf{h}_i$ 是第 $i$ 个 context block 输出,$W_{\text{zero}}$ 为零初始化卷积,$\alpha$ 是控制尺度;context block 每隔 5 个 DiT block 选择性复制一个。**用大白话说**:零初始化保证训练伊始动作分支不注入任何信号、主干权重不被扰动,从而在保留预训练物理先验的同时逐步学到动作可控性。实现中在第 0,5,10,15,20,25,30,35 层复制成可训练 context 分支,主干冻结。

### 4. EZSbench:训练无关的零样本具身评测集

首个训练无关、完全 OOD 的具身视频生成基准:把多样机器人形态、环境、任务组合成训练中未出现的新组合。初始观测池双分支构造:① 用文生图 Nano Banana 变化机器人形态/场景/任务/视角生成合成初始观测;② 用大 VLM 对真实机械臂图做背景可控编辑(保前景交互)。配套物理启发的稠密描述合成(视觉锚定 → 运动学合规的动作模拟 → 叙事合成)。评估用**解耦双模型协议**:Qwen3-VL-32B-Thinking 依初始态与指令动态生成物理检查清单(强制 30–50% 为否定问题防随机蒙对),Qwen2.5-VL-72B-Instruct 作答;物理分 $S_v=\frac{1}{\lvert Q_v\rvert}\sum_{q\in Q_v}\mathbb{I}(\text{VQA}(v,q)=\text{GT}(q))$。

## 三、实验结果

**实现**:128 张 Nvidia H20。三阶段——TI2V 基础训练(Wan2.1-I2V-14B-480P,裁剪 480×832、81 帧,6000 步,全局 batch 128,lr 1e-5);DPO(LoRA rank/scale=64,AdamW lr=1e-6,10 步 warmup,$\beta=5000$,BF16、梯度检查点,500 步/epoch × 100 epoch);A2V(VACE 框架,复制第 0/5/10/15/20/25/30/35 层为 context 分支,batch 16,lr 5e-5,20000 步)。

**PBench 机器人子集**(174 段复杂操作视频,取自 BridgeData V2、AgiBot、OXE;MLLM-as-Judge 用 Qwen2.5-VL-72B-Instruct,886 题覆盖空间/几何接触、时序/因果、物理属性/状态三维):

| 模型 | Quality Score | Domain Score | Avg. |
|---|---|---|---|
| Wan 2.5 | 0.7548 | 0.8644 | 0.8096 |
| GigaWorld-0 | 0.7591 | 0.8583 | 0.8087 |
| Veo 3.1 | **0.7740** | 0.8350 | 0.8045 |
| Wan2.1 14B(SFT前底模) | 0.7672 | 0.8391 | 0.8032 |
| Sora v2 Pro | 0.7679 | 0.7626 | 0.7652 |
| UnifoLM-WMA-0 | 0.7593 | 0.6693 | 0.7143 |
| 本文 Our Model(仅 SFT) | 0.7678 | 0.8785 | 0.8232 |
| **本文 Our Model + DPO** | 0.7676 | **0.9306** | **0.8491** |

关键读数:DPO 版把 Domain Score(物理域一致性)从底模 0.8391、SFT 版 0.8785 推到 **0.9306** 并拿下总分 SOTA 0.8491;而 Quality Score 几乎不变(0.7678→0.7676),说明物理约束的加强**没有牺牲感知质量**。Veo 3.1 / Sora v2 Pro 靠成像与美学拿高 Quality,却在 Domain Score 上明显落后(0.8350 / 0.7626)。SFT 底模的时空稳定性也很强(I2VB 0.9777、MS 0.9916)。

**EZSbench(完全 OOD 零样本)**:

| 模型 | Quality Score | Domain Score | Avg. |
|---|---|---|---|
| WoW-wan 14B | 0.7609 | 0.7951 | 0.7780 |
| GigaWorld-0 | 0.7272 | 0.7826 | 0.7549 |
| Cosmos-Predict 2.5 | 0.7089 | 0.7698 | 0.7394 |
| UnifoLM-WMA-0 | 0.7355 | 0.5232 | 0.6294 |
| **本文 Our Model** | **0.7694** | **0.8366** | **0.8030** |

在训练分布之外仍取得 Quality/Domain/总分三项 SOTA(0.7694 / 0.8366 / 0.8030),表明物理保真提升可跨分布泛化。

**动作条件生成**(从 action-to-video 数据集均匀采 200 个实例;PSNR 测像素、SSIM 测局部纹理、nDTW 用微调 YOLO 定位夹爪算轨迹一致性):

| 模型 | PSNR | SSIM | Traj. Consis. |
|---|---|---|---|
| Enerverse-AC | 20.42 | 0.7542 | 0.8157 |
| Gen-Sim | 18.05 | 0.7413 | 0.6195 |
| **Ours** | **21.09** | **0.8126** | **0.8522** |

三项全面领先,尤其轨迹一致性 0.8522 显著优于基线。定性上(Fig. 5/12/14):Sora v2 Pro、Veo 3.1 在密接触时出现夹爪/物体畸变;GigaWorld-0、Cosmos 出现抓取穿模;WoW 出现无接触抓取与几何畸变;UnifoLM、Wan 2.5 误认目标(如把 spatula 当成 rag);本文能正确识别目标、保持时空连贯、避免形变与穿模,并在"红刀→红盒、黑勺→黑盒"这类长程组合任务中正确绑定物体属性。

## 四、局限性

- 作者在结论中明确:模型**依赖固定视角数据**,尚不支持多视角生成;**缺乏闭环评估**——所有评测都是开环视频生成质量,未在真实机器人上做 rollout / 策略执行验证,故"物理合理"更多是感知层面的判定而非动力学闭环验证。
- 物理判别与 EZSbench 评分都**重度依赖专有大模型**(Gemini 3 Pro 作打分者、Qwen3-VL 系列作提议/作答),评估结论受这些外部模型能力与偏差影响,复现成本高;"物理正确性"本质上是 VLM 的判断而非硬约束的物理仿真。
- DPO 偏好数据的正确性完全取决于 proposer/scorer 的清单质量;单票否决机制虽利于抑制致命违规,但可能对边界样本过于严苛,且 Tier 1/Tier 2 的加权与阈值细节披露有限。
- $\beta=5000$ 这一异常大的取值(远大于常见 DPO 设置)提示扩散 DPO 的偏好信号较弱、需强放大,泛化稳健性有待更多消融支撑;文中未给出 DPO 相关消融(如 $N$、$\beta$、锦标赛 vs 全排列)的定量对比。
- 训练/评测规模巨大(128×H20、14B),对社区复现门槛高;EZSbench 承诺开源但初始观测依赖 Nano Banana 等闭源文生图,长期可复现性存疑。

## 五、评价与展望

**优点**。(1)问题切中要害:把"视频世界模型物理不合理"这一公认痛点,转化为可操作的偏好优化问题,思路清晰、工程完整,从数据、训练到评测形成闭环叙事。(2)解耦"出题/判分"两个 VLM 角色 + 均衡正负问题的设计,是对"VLM 自评幻觉/谄媚"这一实际难题的务实应对,比单模型打分更可信。(3)动作图 + 并行 context block + 零初始化残差的注入方式,把低维动作"渲染"成空间结构化图像再融合,既保留预训练物理先验又实现跨本体动作控制,是相对优雅的工程选择,PSNR/SSIM/轨迹一致性三项数字也支撑其有效性。(4)冻结主干 + LoRA 复用充当参考模型省显存,是 14B 级 Diffusion-DPO 落地的关键实用 trick。

**与其他公开工作的关系**。本文属于快速升温的"物理一致视频生成 + 具身世界模型"赛道:与 Cosmos、GigaWorld-0、WoW、UnifoLM-WMA-0、Enerverse-AC、Genie-Envisioner 等同期竞品直接对标;方法上继承 Wan 骨干、Diffusion-DPO(Wallace 等)、VACE 动作/结构注入、LoRA。其偏好对齐的物理动机与并发工作 PhyGDPO(2512.24551)、RDPO(2506.18655)、PhysCorr(2511.03997)高度同源——都在用 DPO 类目标注入物理先验,差异在本文强调"解耦双 VLM 判别 + 锦标赛选样"与专门的具身数据策管;贡献的独特性更多在系统整合与 EZSbench 评测,而非单点算法创新。EZSbench 补齐了 PBench 偏 in-distribution 的空白,其"训练无关 + 解耦评物理/动作"的定位若真开源,对该领域标准化评测有实际价值。

**开放问题与可能改进方向**。① 闭环化:把开环视频质量升级为"世界模型 rollout 驱动策略/规划"的闭环评测(如作为 WAM 预测 action-conditioned 动态供 MPC 使用),才能真正检验"物理合理"是否等价于"可用于规划"。② 硬物理约束:当前物理正确性是 VLM 判定,后续可引入可微物理/接触仿真或几何一致性约束(穿模、质量守恒)作为更硬的监督,减少对闭源打分模型的依赖。③ 多视角与 3D 一致性:结论已点名固定视角局限,引入多视角/3D 表征有望进一步压低几何畸变与穿模。④ 偏好信号研究:$\beta=5000$ 的必要性、锦标赛选样相对全排列/成对比较的收益、以及 proposer 清单质量对最终物理分的敏感性,都值得系统消融。⑤ 长程与可靠性:长时序生成的误差累积、以及物理评分的自动化可信度(与人类物理判断的一致性),仍是把此类模型用作 VLA 模拟器前必须回答的问题。

## 参考

1. Wallace et al. *Diffusion Model Alignment Using Direct Preference Optimization*(Diffusion-DPO,本文偏好对齐目标的直接来源)。
2. Team Wan / Alibaba. *Wan: Open and Advanced Large-Scale Video Generative Models*(arXiv 2503.20314,骨干 Wan2.1-I2V-14B)。
3. Jiang et al. *VACE: All-in-One Video Creation and Editing*(ICCV 2025,并行 context block / 结构注入范式)。
4. Cai et al. *PhyGDPO: Physics-Aware Groupwise DPO for Physically Consistent Text-to-Video Generation*(arXiv 2512.24551,并发的物理 DPO 工作)。
5. Zhou et al. *PAI-Bench: A Comprehensive Benchmark for Physical AI*(arXiv 2512.01989,主实验所用 PBench 及其机器人子集)。
