# VITRA：用真实生活人类活动视频做可扩展的机器人操作 VLA 预训练

> **论文**：*Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos*
>
> **作者**：Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Sicheng Xu, Yizhong Zhang, Dong Chen, Jiaolong Yang, Baining Guo et al.
>
> **机构**：Tsinghua University；Microsoft Research Asia
>
> **发布时间**：2025 年 10 月（arXiv 2510.21571）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.21571) | [PDF](https://arxiv.org/pdf/2510.21571)
>
> **分类标签**：`人类视频预训练` `灵巧手 VLA` `数据合成` `egocentric 视频`

---

## 一句话总结

把网上海量、无脚本、无标注的第一人称人类手部活动视频,通过一条全自动流水线(单目 3D 手/相机重建 + 手腕速度极小值切分 + GPT 轨迹叠加式打标)转成与机器人 V-L-A 数据完全对齐的 **1M episode / 26M 帧** 灵巧手数据集,预训练出的 VLA 在全新环境零样本手部动作预测上把手-物距离从 20.0cm 压到 **8.8/6.2cm**(远超 Being-H0 的 19.1cm),仅用 1.2K 条真机数据微调后,在 seen 任务达 **71.0%**、unseen 任务达 **64.6%** 平均成功率,大幅超过 π0、VPP、OXE 预训练等基线。

## 一、问题与动机

VLA 模型的预训练目前严重受限于机器人动作数据的规模与多样性:主流数据(OXE、DROID、AgiBot 等)靠实验室遥操作采集,成本高,物体/场景/技能覆盖远落后于互联网级语言与视觉数据;灵巧手动作数据更是稀缺,作者称"据我们所知没有可用于预训练的大规模灵巧手动作数据集"。

与此同时,网络上存在海量真实生活人类视频,富含日常操作与多样物理交互,但它们是 **unstructured** 的:无脚本、无分段、长度不一、任务粒度混乱、缺语言指令与 3D 动作标签。此前用人类视频做机器人学习的工作(学表征、affordance、point track,或学 latent action)都未把无标注的大规模真实视频转成**显式 3D 动作**的 VLA 预训练数据。核心问题因此被提为:**能否把这些无结构的真实人类视频转换成与现有机器人 V-L-A 训练数据格式完全对齐的数据?** 这需要解决两类对齐:

1. **任务对齐(Task alignment)**:把长视频切成原子级、短时程的动作片段,粒度匹配机器人数据(如"拿起海绵""用布擦炉子")。
2. **标签对齐(Label alignment)**:从单目、未标定、常在移动的相机视频里恢复度量空间(metric-space)的 3D 手部运动作为稠密动作标签,并配上精确的语言指令。

作者把人手视为灵巧机器人的末端执行器,给出肯定回答。

## 二、核心方法

方法分两块:(A) 把人类视频转 VLA 数据的三阶段流水线;(B) 灵巧手 VLA 模型架构与训练。

### A. 全自动人类活动分析流水线(三阶段)

**阶段一:3D 运动打标(3D Motion Labeling)。** 先用背景光流判断相机是静止还是移动;移动相机用 DroidCalib、静止相机用 MoGe-2 + DeepCalib 估计内参,并对大畸变视频做去畸变以符合针孔模型。随后用 **HaWoR** 逐帧重建相机坐标系下的 3D 手(基于 MANO 参数模型,含手腕 6D 位姿与关节角);用改造版 **MegaSaM**(把其深度估计模块换成 MoGe-2 提供的度量深度先验)追踪移动相机位姿。二者结合得到世界坐标系下的 3D 手序列,再做样条平滑并剔除离群点。长视频切成有重叠的 20 秒片段处理后再合并。世界坐标 3D 手序列可投影回任意帧的相机坐标,等效"模拟静止相机",贴合多数机器人数据。

**阶段二:原子动作切分(Atomic Action Segmentation)。** 关键洞察:人手在动作切换处常出现速度变化,极小值往往对应动作切换。于是**检测世界空间中手腕的速度极小值作为切割点**(在以每点为中心的固定窗口内取局部速度极小值)。左右手独立切分,忽略另一只手。此法高效、无需任何额外模型推理或预标注文本。

> 用大白话说:人做一个"抓起来"的动作,手会先减速接近物体、抓住时几乎停顿,这个"顿一下"的瞬间速度最低,就是一个动作的天然节拍点——按这些"顿点"下刀,就能把连绵不断的手部运动切成一个个原子动作,不用任何语义模型。

**阶段三:指令打标(Instruction Labeling)。** 对每个片段均匀采 8 帧,并把手掌从当前帧到片段末尾的世界空间轨迹**投影叠加**到这些帧上,再喂给 **GPT-4.1**,让它结合帧内容与叠加轨迹用祈使句描述该手动作;无语义意义的片段标为 "N/A"。作者实证:先切成原子片段再打标、以及叠加轨迹这两点都显著提升打标准确率(固定 1 秒切分或不叠轨迹都会掉点)。

**数据集构建。** 在 Ego4D、Epic-Kitchen、EgoExo4D、SSV2 上跑该流水线(**明确不使用**这些数据集原有的人工动作标注,因其粒度不符或缺精确起止时间),得到 **1M episode / 26M 帧**,来源占比:Ego4D 77%、Epic-Kitchen 12%、EgoExo4D 6%、SSV2 5%,涵盖烹饪、清洁、施工、维修、手工、绘画等真实活动。

### B. 灵巧手 VLA 模型

模型 $\pi$ 定义为:

$$\pi:(l, o_t, s_t)\rightarrow(a_t, a_{t+1}, \dots, a_{t+N})$$

即基于语言指令 $l$、当前视觉观测 $o_t$、本体状态 $s_t$ 预测未来 $N$ 步末端执行器动作块。

**架构。** VLM backbone 用 **PaliGemma-2**(SigLIP 视觉编码器 + Gemma-2,3B 参数,输入 $224^2$);额外注入相机 FoV 作为一个 token,帮助模型理解原图长宽比与内参;沿用 CogACT 的做法追加可学习 **cognition token**,其输出特征 $f^c$ 作为动作专家的条件。动作专家用 **Diffusion Transformer(DiT-Base)**,输入是 $f^c$、手部状态 $s_t$ 与带噪动作块的拼接,并通过 **AdaLN** 再次注入 $f^c$。训练用 MSE 去噪损失:

$$\mathcal{L}_{\mathrm{MSE}}=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,1),\, i}\lVert\hat{\epsilon}^{\,i}-\epsilon\rVert_2$$

> 用大白话说:VLM 看图和指令,把"要做什么"浓缩成一个 cognition 特征;扩散动作专家拿着这个特征当"指挥棒",从纯噪声里一步步反推出未来一串具体的手部动作。视觉编码器冻结,VLM + cognition token + 动作专家端到端训练。

**手部动作空间。** 在当前观测的相机坐标下预测:

$$a_t=[\Delta t^l, \Delta r^l, \theta_h^l, \Delta t^r, \Delta r^r, \theta_h^r]\in\mathbb{R}^{102}$$

其中 $\Delta t\in\mathbb{R}^3$、$\Delta r\in\mathbb{R}^3$ 是相邻帧手腕的相对平移与相对旋转(由旋转矩阵转 Euler 角),$\theta_h\in\mathbb{R}^{15\times3}$ 是 MANO 手模型 15 个关节在局部坐标的 Euler 角,上标 $l/r$ 表左右手。每只手 $3+3+45=51$ 维,双手共 102 维。

**统一单/双手预测。** 语言指令统一格式为 "Left hand: $\langle$左手动作$\rangle$. Right hand: $\langle$右手动作$\rangle$",当前帧不落在某手动作块内则该手写 None;为每只手配 0/1 的 action mask,mask=0 时对应带噪动作置零并排除出损失,从而统一处理单手/双手样本。

**因果动作去噪(Causal Action Denoising)。** 真实人手动作快,很多片段仅约 1 秒(约 30 帧),而预测块长 $N=16$ 会越过片段末尾;直接零填充会污染前面的预测。故动作去噪改用**因果注意力**,每个动作 step 只关注在它之前的动作,零填充位置也从损失中剔除(其 mask 置 0)。消融显示改回双向注意力会掉点,说明因果注意力更契合该预训练数据特性。

**训练与微调。** 预训练:先 warm up 动作专家 + cognition token 映射层 + FoV 的 MLP 共 5K 步,再联合微调 VLM 与动作专家 80K 步,LR 分别为 1e-4(动作专家)/1e-5(VLM),batch 512,8×H100 跑 2 天。伴随 **trajectory-aware augmentation**:随机裁剪 + 变 FoV/长宽比/裁剪中心的透视变形(保证从当前帧到末尾的投影手轨迹仍落在裁后图内)、随机翻转、文本无颜色线索时做随机颜色抖动。微调到真机:20K 步、batch 256、LR 1e-5、8×H100 约 8 小时;机器人为 Realman 机械臂 + 12-DoF XHand 灵巧手 + RealSense 头部相机;把每个机器人关节映射到拓扑上最近的人手关节、未映射维度零填充并入 mask,并用**未来执行指令**(而非从记录状态反推的动作标签)监督关节角。

## 三、实验结果

### 数据多样性(与已有 VLA 数据对比)

用 DINOv2 特征与 OpenImages 的最大余弦相似度衡量场景覆盖度(R@0.5,越高越广):

| 数据集 | 与 OpenImages 的相似度 R@0.5 ↑ |
| --- | --- |
| VITRA(本文) | 0.263 |
| VITRA(无增广) | 0.210 |
| AgiBot World beta | 0.045 |
| EgoDex | 0.038 |
| OXE(约 400K 子集) | 0.024 |
| DROID | 0.020 |

即便只采 10K episode,其多样性也已超过其它数据集全集;语言指令的 h-index / i100-index(名词/动词/形容词)亦全面领先。

### 零样本人手动作预测(全新环境,Table 1)

抓取任务(47 个未见环境、Azure Kinect、396 物体)用手-物最小距离 $d_{\mathrm{hand\text{-}obj}}$(cm,越低越好)衡量;通用动作(117 个未见环境、手机拍摄)用 23 名参与者对 top-3 动作打分(3/2/1 分)的用户评分衡量。

| 方法 | Grasp $d_{\mathrm{hand\text{-}obj}}$ 均值/中位 (cm) ↓ | General action 用户评分 ↑ |
| --- | --- | --- |
| 初始位置(参考) | 20.0 / 20.0 | – |
| Being-H0 (8B, 并行工作) | 19.1 / 18.4 | 0.15 |
| Lab data (EgoDex) | 17.6 / 18.3 | – |
| Human annotation(用原标注) | 14.1 / 14.1 | 0.96 |
| No augmentation | 11.6 / 10.7 | 1.43 |
| Bidirectional attention | 9.3 / 7.2 | 1.69 |
| **Ours** | **8.8 / 6.2** | **1.91** |

episode 构建策略消融(Table 2,350K 子集):固定 1 秒切分 10.5/8.8、不叠轨迹 11.7/10.7、**本文 9.9/8.1**,证明速度极小值切分 + 轨迹叠加两者都有效。

### 真机灵巧操作(1.2K 条微调,四任务)

**Seen(Table 3,成功率 %):**

| 方法 | Pick&place | Functional grasp | Pour | Sweep | Average |
| --- | --- | --- | --- | --- | --- |
| VPP | 57.5 | 29.2 | 12.5 | 0.0 | 24.8 |
| π0 | 37.5 | 25.0 | **75.0** | 50.0 | 46.9 |
| No VLA pretrain | 32.5 | 33.3 | 12.5 | 50.0 | 32.1 |
| Latent action pretrain (LAPA) | 42.5 | 41.7 | 37.5 | **62.5** | 46.0 |
| OXE pretrain | 40.0 | 37.5 | 62.5 | 25.0 | 41.3 |
| **Ours** | **80.0** | **66.7** | **75.0** | **62.5** | **71.0** |

**Unseen(Table 4,成功率 %):**

| 方法 | Unseen Obj\&BG Pick&place | Functional grasp | Pour | Unseen Category Pick&place | Average |
| --- | --- | --- | --- | --- | --- |
| VPP | 12.5 | 0.0 | 0.0 | 8.3 | 5.2 |
| π0 | 0.0 | 6.2 | 25.0 | 33.3 | 16.1 |
| No VLA pretrain | 31.2 | 0.0 | 0.0 | 12.5 | 10.9 |
| Latent action pretrain | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OXE pretrain | 12.5 | 6.3 | 0.0 | 12.5 | 7.8 |
| **Ours** | **68.8** | **68.8** | **50.0** | **70.8** | **64.6** |

### 缩放行为与关键观察

- **数据缩放**:抓取任务 $d_{\mathrm{hand\text{-}obj}}$ 随数据量在 log 尺度上近似线性下降;真机成功率随预训练数据规模在 seen/unseen 上持续提升。
- **多样性 > 规模**:EgoDex 虽有更多 episode、更大帧数(**130M vs 2.6M**),但其预训练模型表现还不如本文用 **仅 10%** 数据训练的模型,且在未见场景几乎全失效,归因于数据多样性不足。
- **手部预测可作代理指标**:预训练手部预测精度与微调后真机成功率呈明显正相关(Fig.10c),故该零样本手部预测 benchmark 可作为下游真机性能的快速代理,便于原型迭代。
- **显式 3D 动作优于 latent action**:把预测目标换成 LAPA 的 latent action 后,seen 尚可但 unseen 全面归零,作者认为 latent action 难以从任务无关背景中解耦任务相关运动。

## 四、局限性

1. **数据源仍受限**:当前主要来自已有第一人称数据集,尚未纳入 HowTo100M 等更大更杂的网络视频;流水线可扩展但未验证在完全 in-the-wild 网络视频上的鲁棒性。
2. **重建噪声**:单目 3D 重建(手/相机/深度)与 VLM 打标本身有误差,预训练数据存在不准确样本,作者承认需更强重建与额外过滤机制。
3. **技能停留在原子级短时程**:只学到短时程原子操作,长时程规划/推理未涉及(留作把数据组织成更高层任务结构的未来工作)。
4. **主要单手 + 单视角 + 无触觉**:真机实验以单手为主(仅做了简单"交接"验证双手可行性),缺多视角与触觉;人手到 XHand 的动作空间映射靠最近关节 + 微调对齐,而非直接位姿迁移,存在体现差异(embodiment gap)。
5. **评测规模有限**:真机任务四类、trials 数偏小(如 Pour/Sweep 仅 8 trials),统计置信度受限。

## 五、评价与展望

**优点。** (1) 真正回答了"无标注、无脚本的真实人类视频 → 显式 3D 动作 VLA 数据"这一空白,整条流水线全自动、零人工标注,可扩展性是其最大卖点;(2) 速度极小值切分是极简且无模型依赖的原子切分方案,配合"轨迹投影叠加 + GPT 打标"解决了长视频语义分段与指令生成两大难题,工程上很实用;(3) 系统性地用**数据多样性**(DINOv2-OpenImages 相似度、词频 h-index)量化说明"为什么人类视频比实验室机器人/受控人手数据更值得预训练",并以 130M vs 2.6M 的对照有力论证"多样性 > 规模";(4) 因果注意力 + action mask 对"短片段、单双手混合"这类真实数据特性的处理干净利落。

**与其它公开工作的关系。** 与并行的 Being-H0、H-RDT、EgoDex 等同样用 egocentric 3D 手动作做 VLA 预训练的工作相比,本文的差异化在于**坚持用无脚本的真实生活视频**(而非受控实验室 / VR-AR 头显 / RGBD 采集的脚本化数据),因而任务、物体、场景覆盖显著更广,零样本能力更强;在动作表征路线上与 LAPA/IGOR/UniVLA 等 latent-action 路线针锋相对,并用实验证明显式 3D 动作在未见场景上的优势;在架构上则是 CogACT(cognition token)+ π0 式扩散动作专家的组合,创新集中在**数据**而非模型。

**开放问题与可能的改进方向。** (1) 单目重建噪声是精度天花板,可引入更强的度量深度/手位姿模型与不确定性感知的样本过滤或加权;(2) 人手 51 维 MANO 到不同灵巧手/夹爪的通用重定向仍是瓶颈,可探索可微 retargeting 或体现无关的动作表征;(3) 目前把 GPT 打标当作可靠监督,但 N/A 判定与祈使句歧义会引入标签噪声,可加一致性校验或多次采样投票;(4) 把原子片段进一步组织成层级化长时程任务结构,是通向长程规划的自然延伸;(5) 该工作已成为"用真实人类视频规模化生产 VLA 数据"的一个有力范式,后续若开源数据与模型(作者承诺开源),对整个社区做人类视频驱动的预训练有较强推动。

## 参考

1. Luo et al. *Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos.* arXiv 2507.15597, 2025.(最直接的并行对比工作)
2. Hoque et al. *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.* arXiv 2505.11709, 2025.(实验室采集人手数据基线)
3. Li et al. *CogACT: A Foundational VLA Model Synergizing Cognition and Action.* arXiv 2411.19650, 2024.(cognition token 架构来源)
4. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv 2410.24164, 2024.(扩散动作专家基线)
5. Bu et al. *UniVLA / Learning to Act Anywhere with Task-Centric Latent Actions.* arXiv 2505.06111 / 2502.14420, 2025.(latent action 路线代表)
