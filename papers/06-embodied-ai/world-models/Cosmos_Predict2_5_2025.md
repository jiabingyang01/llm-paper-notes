# Cosmos-Predict2.5：面向 Physical AI 的视频世界基础模型

> **论文**：*World Simulation with Video Foundation Models for Physical AI*
>
> **作者**：NVIDIA Cosmos 团队（集体署名,完整贡献者名单见原文 Sec. A）
>
> **机构**：NVIDIA
>
> **发布时间**：2025 年 11 月（arXiv 2511.00062,v2 于 2026 年 2 月更新）
>
> **发表状态**：未录用（预印本 / 技术报告,以 NVIDIA Open Model License 开源）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.00062) | [PDF](https://arxiv.org/pdf/2511.00062)
>
> **分类标签**：`视频世界模型` `Physical AI` `flow matching` `机器人数据增强` `action-conditioned`

---

## 一句话总结

Cosmos-Predict2.5 用 flow matching 把 Text2World / Image2World / Video2World 三种生成模式统一进单一 DiT,以 Physical-AI 专用 VLM（Cosmos-Reason1）替换 T5 文本编码器,在 200M 精选视频上做渐进式预训练,再经"分域 SFT + 模型融合 + GRPO 强化学习"后训练,发布 2B / 14B 两个规模,并配套一个 controlnet 式的 Cosmos-Transfer2.5（比前代 Transfer1 小 3.5×)。在真机双臂 pick-and-place 任务上,用 Transfer2.5 做视觉数据增强,把 Diffusion Policy 在 10 个测试场景的成功率从 base 的 1/30、标准图像增强的 5/30 提升到 **24/30**。

## 一、问题与动机

Physical AI（带传感器与执行器的具身体)在真实世界直接训练慢、贵、危险,尤其早期阶段的失误可能损坏本体或环境。一个能"根据 agent 动作生成高保真、多样化视觉环境"的世界模拟器,可作为真实世界的安全代理,让 agent 在纯软件里习得感知与控制技能后再部署。

本文是前作 Cosmos-Predict1（EDM 扩散世界模型)的迭代,提出三点核心改进:

1. **数据管线更严**:处理 3500 万小时原始视频（前代 2000 万小时),经七阶段过滤,存活率从 30% 收紧到仅 4%,产出约 2 亿高质量训练片段。
2. **架构统一**:把 Text2World、Image2World、Video2World 合并进一个模型。
3. **训练配方升级**:引入 model merging、一种基于 GRPO 的强化学习后训练,并用 decoder-only 的 Cosmos-Reason1 替换 Cosmos-Predict1 里的 T5 文本编码器,以获得更丰富的文本表征和更细粒度的世界生成控制。

论文进一步把 Predict2.5 扩展为 controlnet 家族 Cosmos-Transfer2.5,用于 Sim2Real / Real2Real 的世界翻译,并展示了在机器人策略学习、VLA 合成数据、自动驾驶多视角仿真、动作条件世界模型等下游任务上的落地。

## 二、核心方法

### 2.1 Flow Matching 训练目标

模型不再用 EDM,而是用 flow matching（FM)。给定数据样本 $x$（图像或视频)、噪声 $\epsilon \sim \mathcal{N}(0,I)$、时间步 $t \in [0,1]$,插值隐变量为

$$
x_t = (1-t)x + t\epsilon
$$

对应的 ground-truth 速度为

$$
v_t = \epsilon - x
$$

训练目标是让网络 $u(\cdot;\theta)$ 预测速度,最小化 MSE:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x,\epsilon,c,t}\,\lVert u(x_t,t,c;\theta) - v_t \rVert^2
$$

其中 $c$ 是条件信息（文本嵌入、参考帧等)。**用大白话说**:FM 直接让网络学"从纯噪声指向干净数据的那根箭头（速度场)",训练目标比 EDM 的"预测标准化高斯"更直接,优化更平滑、采样质量更好；FM 与 EDM 在前向/后向扩散过程上其实数学等价,区别只在网络参数化方式。

### 2.2 偏置到高噪声:shifted logit-normal

高分辨率内容相邻像素高度相关,若注入噪声太小,模型学不会"打散"这种相关性。作者用 shifted logit-normal 分布:先从 logit-normal 采 $t$,再做单调变换

$$
t_s = \frac{\beta t}{1 + (\beta - 1)t}
$$

$\beta$ 是 shift 超参,$\beta = 1$ 时不做偏移。预训练中随分辨率递增地增大 $\beta$（256p 时 $\beta=1$,到 720p 时 $\beta=5$)。**用大白话说**:分辨率越高越要多喂"噪声很重"的样本,逼模型学会在信号被严重破坏时也能重建结构。此外还发现生成视频有"帧间突兀跳变"的伪影,归因于最高噪声区训练样本太少,于是**改调度器:强制 5% 的训练样本抽自噪声分布最高的 2%**,显著减少了跳变伪影。

### 2.3 网络架构

- **骨干**:沿用 Predict1 的 DiT（latent diffusion),关键改动是**去掉绝对位置编码、只保留相对位置编码 + 3D RoPE**,以便后训练时泛化到更高分辨率、更长序列。2B = 32 层 / 2048 维 / 8192 FFN;14B = 36 层 / 5120 维 / 20480 FFN;均用 AdaLN-LoRA（256 维)、GELU。
- **视觉 tokenizer**:WAN2.1 VAE,时间/高/宽压缩率 $4 \times 8 \times 8$,再叠 $1 \times 2 \times 2$ patchify。一次生成 93 帧 = 24 个 latent 帧,16 fps,约 5.8 秒。
- **文本编码器**:Cosmos-Reason1（Physical-AI 专用 VLM)。不同于只取单层 transformer 输出,而是**拼接多个 block 的激活并投影到 1024 维**,更好捕捉局部+全局语义,经 cross-attention 注入去噪过程。
- **三种模式统一**:Text2World（纯文本)、Image2World（文本+参考图)、Video2World（文本+视频)。Image/Video2World 用 **frame-replacement 策略**——生成序列的前若干帧始终替换为条件帧,既灵活(条件帧数可调)又强化时序一致性。

### 2.4 训练流程（预训练 + 后训练)

**渐进式预训练**（Tab. 4):Text2Image 256p → 加入 Image2World/Video2World（随机取 1 或 5 条件帧,补全剩余 92/88 帧)→ 分辨率 256p→480p→720p → 最后加入 Text2World（0/1/2 条件帧,概率 0.5/0.25/0.25)。用 mask token 二值标记哪些是条件输入,损失只作用在待生成帧上。AdamW,lr 3e-5（2B)/1.3e-5（14B),warmup 2000。

**后训练**分三步:

1. **分域 SFT**:用 InternVideo2 嵌入训练多头分类器,把数据分成 5 域——object permanence（10.4M)、high motion（1.0M)、complex scenes（1.6M)、driving（3.1M)、robotic manipulation（730K),外加 4K（388K)。**每域单独 fine-tune 一个模型**(30k 步,batch 256),再加一个 4K cooldown 阶段(lr 线性退火到 0)。SFT 模型在各自领域对预训练基线的胜率均显著提升(如 robotic manipulation 胜率 72.6%)。
2. **模型融合**:对比 model soup、TIES、DARE-Linear、DARE-TIES,发现简单网格搜索超参往往优于按单模型胜率的启发式选择,最终选 **model soup**。Fig. 4 的雷达图显示融合模型在各专域取得最优的同时,general 域仍保持 52.5%。
3. **强化学习**:把"条件视作 state、整条去噪轨迹视作 action",用 VLM 奖励模型 **VideoAlign**（评 text alignment / motion quality / visual quality),GRPO 归一化组内优势。每条件生成 8 个输出、各 20 步去噪;因显存限制把轨迹概率分解为各步条件概率之和,每两步累积梯度、共 10 步一次更新;训练 256 步、batch 32,并用 diffusion loss 做正则以缓解 reward hacking。

**时间步蒸馏**:用 rCM（连续时间一致性蒸馏 + 分布匹配蒸馏的混合前向-后向框架),配 JVP 支持的 fused flash attention,把推理压到 **4 步**且质量接近 teacher。

**基础设施**:FSDP2 逐参数分片 + TorchTitan 异步 checkpoint;Ulysses 式 context parallelism;selective activation checkpointing;RL 用弹性奖励服务(Redis + CUDA IPC 零拷贝)。4096 张 H100、720p/93 帧下,2B 的 MFU 36.49%（CP=2),14B 33.08%（CP=8)。

### 2.5 Cosmos-Transfer2.5（controlnet)

在 Predict2.5-2B 上加控制分支,条件支持 edge / blur / depth / segmentation 四种模态。与 Transfer1-7B 在开头顺序插 4 个 control block 不同,**Transfer2.5 把 4 个 control block 更均匀地分布(每 7 个主干 block 后插一个)**,让条件信息更渐进地融入。深度用 Video Depth Anything 生成 1000 万视频、分割用 SAMv2 生成 300 万、edge/blur curate 1400 万,每个控制分支独立训练 10 万步。

## 三、实验结果

### 3.1 PAI-Bench 通用生成质量

Domain Score（7 域 VQA)与 Quality Score（8 项改编自 VBench)取均值为 Overall。

**Text2World（Tab. 10)**

| 模型 | Domain | Quality | Overall |
|---|---|---|---|
| Cosmos-Predict2.5-2B post | 0.804 | **0.732** | 0.768 |
| Cosmos-Predict2.5-14B post | 0.803 | **0.732** | 0.768 |
| Wan2.1-14B | 0.794 | 0.727 | 0.761 |
| Wan2.2-27B-A14B | **0.810** | 0.728 | **0.769** |

**Image2World（Tab. 11)**

| 模型 | Domain | Quality | Overall |
|---|---|---|---|
| Cosmos-Predict2.5-2B post | 0.840 | 0.779 | **0.810** |
| Cosmos-Predict2.5-14B post | 0.838 | **0.781** | **0.810** |
| Wan2.2-27B-A14B | **0.841** | 0.772 | 0.806 |

自动指标上,2B 后训练模型在 T2W 与远大于它的 Wan2.2-27B-A14B 相当,在 I2W 上是最优。

**人类偏好**:2B 后训练在 PAI-Bench I2W/T2W 上比 Wan2.2-5B 更受偏好（30.0% vs 26.2%),与 Wan2.1-14B 相当（33.0% vs 34.8%),而参数量分别小 60.0% / 85.7%;14B 明显优于 Wan2.1-14B（48.6% vs 31.8%),与 Wan2.2-27B-A14B 持平（38.1% vs 35.9%)。

**RL 与蒸馏消融**:Tab. 6 中 RL 后 reward 显著上升(T2W 预训练 sum 1.08→1.69,merged 1.23→1.74;I2W merged 0.24→0.45);人类投票也偏好 RL 版(Pretrained+RL 41.1% win)。蒸馏后 4 步模型质量几乎无损(T2W Overall 0.768→0.764,I2W 0.810→0.816)。

### 3.2 Cosmos-Transfer2.5 controlnet（Tab. 12)

在 PAIBench-Transfer（600 视频)上,Transfer2.5-2B 全面超过 Transfer1-7B（小 3.5×)。以 Blur 控制为例,Overall Quality **9.75 vs 6.56**,Blur SSIM 0.90 vs 0.89。长视频误差累积用 RNDS 度量:

$$
\mathrm{RNDS}[i] = \left(\frac{\mathrm{DOVER}[i]}{\mathrm{DOVER}_{\mathrm{GT}}[i]}\right) \Big/ \left(\frac{\mathrm{DOVER}[1]}{\mathrm{DOVER}_{\mathrm{GT}}[1]}\right)
$$

Fig. 10 显示 Transfer2.5 在 edge/blur/depth/seg 四种控制下的 RNDS 随 chunk 衰减都远小于 Transfer1,长视频幻觉与误差累积更少。

### 3.3 真机机器人策略学习(最贴近具身操作)

平台:半人形双臂(两只 7-DoF Kinova Gen3 + Robotiq 2F-140 夹爪),头部 RealSense D455 第一视角相机。任务:双臂 pick-and-place(把 apple 放进 bowl),收集 100 条遥操作演示,训一个 UNet-based Diffusion Policy。

数据增强策略:用 Transfer2.5-2B 生成视觉增强视频——**全局 edge 控制 + 仅对机器人像素做 blur 控制**（Grounding DINO + SAMv2 定位),CFG=3;先让 VLM 给场景 caption,再用带 `[TABLE]/[COLOR_APPLE]/[COLOR_BOWL]` 等占位符的模板由 LLM 生成外观变体,每条演示配 5 个合成变体,**动作与关节状态保持不变、只增强图像**。

在 10 个测试场景（base + 9 个新场景,含替换水果为山竹、换橙色碗、铺桌布、加聚光灯、加干扰物、开抽屉、以及三种组合的分布外场景)各试 3 次(Tab. 13):

| 策略 | 成功率(总 /30) |
|---|---|
| Base（仅 100 条演示） | 1/30 |
| Baseline（标准图像增强:亮度/对比度/噪声/模糊等） | 5/30 |
| Proposed（Transfer2.5 增强观测） | **24/30** |

标准图像增强只能做低层像素扰动,无法做"换物体颜色、换背景、换光照"这类语义编辑,而世界模型可通过文本提示可控地合成这些结构化分布外变化。

### 3.4 其余具身相关下游

- **自动驾驶多视角**（Tab. 14/15):Predict2.5-2B/auto/multiview 与 Transfer2.5-2B/auto/multiview(以 world scenario map 为控制),7 相机 720p。FVD/FID 相比 Transfer1-7B-Sample-AV 提升最高 2.3×（FVD StyleGAN 23.06 vs 63.69);3D cuboid / lane 检测指标提升最高约 60%。
- **相机可控多视角**（Tab. 17):Transfer2.5-2B/robot/multiview,用 Plücker raymap 编码目标相机,在 Agibot（采 145,820 episodes / 3 视角)、MultiCamVideo、SynCamVideo 上训练。跨视角一致性 Sampson 误差 **19.73 vs 单视角 26.61**,相机轨迹精度相当。
- **VLA 合成数据**（Tab. 18):后训练 Predict2.5-14B/robot/gr00tdream-gr1,在 DreamGen benchmark（GR1 人形)上指令跟随最高——Object GPT4o 评分 **91.8**、Qwen 69.4,超过 Hunyuan、CogVideoX、WAN2.1;流程是生成机器人执行未见指令的视频后,用 latent action model 或 IDM 抽取伪动作,拼成 vision+language+action 的 VLA 训练数据。
- **动作条件世界模型**（Tab. 19/20):Predict2.5-2B/robot/action-cond,加一个 action embedder MLP 把 7 维夹爪动作 $\langle \Delta x, \Delta y, \Delta z, \Delta\theta_r, \Delta\theta_p, \Delta\theta_y, \text{GripperWidth} \rangle$ 映射后**加到 DiT 的 timestep embedding 上**,自回归逐块预测。Bridge 数据(约 20,000 episodes)上 PSNR **24.95**、SSIM 0.85、Latent L2 0.28、FVD 146,全面优于 Predict1-7B baseline（21.14 / 0.82 / 0.32 / 190)。消融显示注入方式 time-embedding > cross-attention > channel-concat。

## 四、局限性

1. **单次生成仅约 5.8 秒**（93 帧),长时程仍靠自回归拼接,虽然 Transfer2.5 显著减轻误差累积,但闭环长视频仍是开放问题。
2. **分域 SFT 是"各域单独训 + 事后融合",不是真正的联合多任务训练**;模型融合虽 general 域退化小,但对新增域的可扩展性未验证。
3. **动作条件世界模型只在 Bridge 单一小数据集(约 2 万 episodes)、单臂厨房场景验证**,7 维夹爪动作空间较窄,尚未扩展到高自由度双臂/人形闭环控制。
4. **VLA 合成数据依赖 IDM / latent action model 抽伪动作**,伪动作质量对下游策略的影响缺乏定量剖析,存在误差传播风险。
5. **物理合理性主要靠数据侧筛选**(明确排除游戏、合成图案、动画、卡通)而非显式物理约束或可微仿真,对未见物理现象的外推能力存疑;PAI-Bench 的 physics 分项也未单独报告数值。
6. **技术报告性质,部分关键细节外包给引用**(RL 完整技术、rCM 蒸馏分别引 Ye et al. 2025、Zheng et al. 2025),不完全 self-contained,复现门槛高。
7. 与 Wan 系列的优势主要体现在**人类偏好**上,自动指标(Domain/Quality)提升幅度有限,自动评测与主观偏好的一致性存疑。

## 五、评价与展望

**优点**:工程完整度极高——覆盖数据管线、flow matching、VLM 文本编码、渐进预训练、分域 SFT、模型融合、GRPO 强化学习、4 步蒸馏、大规模 H100 基础设施,并全部开源(2B/14B + 多个领域特化 checkpoint)。把 GRPO 式 RL 用到视频世界模型后训练、以及"高噪声区强制重采样修帧间跳变"是有价值的实践经验。真机数据增强 24/30 的结果对具身社区很有说服力,展示了世界模型作为"可文本控制的语义级数据增强器"相较传统图像增强的代际优势。

**缺点与开放问题**:(1)与更大的 Wan2.2-27B-A14B 在自动指标上只是持平而非明显超越,"小而强"更多靠人类偏好背书;(2)action-conditioned 世界模型仍很初步,离真正可交互、可闭环 rollout 的具身世界模型(如支持长时程 policy evaluation)有距离;(3)多视角一致性虽有改善,但 Sampson 误差绝对值仍偏高,占用作可靠"补全遮挡视角"的证据不足;(4)物理层面缺乏显式建模。

**与其他公开工作的关系**:相比 Wan、CogVideoX、Hunyuan、LTX 等通用视频生成模型,本文明确面向 Physical AI 并用领域数据 + 专用 VLM 文本编码器强化物理可控性;相比 DreamGen（Jang et al. 2025)提出的"生成视频 + 抽伪动作"范式,本文将其纳入自身闭环并在 GR1 人形上刷到最优;相比抽象隐空间世界模型(Dreamer/JEPA 一脉),本文坚持像素空间高保真视频预测路线,牺牲一定推理效率换取"可直接当合成数据生成器"的通用性。

**可能的改进方向**:引入显式物理先验或可微仿真监督以提升 physics 分项;把 action-conditioning 从 timestep-embedding 注入扩展到更丰富的跨模态交互并在多本体大规模数据上验证;探索真正的长时程闭环 rollout 与 policy-in-the-loop 评测;以及对伪动作质量做定量归因,建立"世界模型合成数据 → 策略性能"的可解释链路。

## 参考

1. NVIDIA. *Cosmos World Foundation Model Platform for Physical AI (Cosmos-Predict1 / Cosmos-Transfer1)*, 2025 — 本文直接前作与基线。
2. Lipman et al. *Flow Matching for Generative Modeling*, ICLR 2023 — 训练目标的理论基础。
3. Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)*, 2024 — shifted logit-normal 时间步分布来源。
4. Jang et al. *DreamGen*, 2025 — VLA 合成数据范式与本文 GR1 评测所用 benchmark。
5. Guo et al. *DeepSeek-R1 (GRPO)*, 2025 — RL 后训练所采用的策略优化算法。
