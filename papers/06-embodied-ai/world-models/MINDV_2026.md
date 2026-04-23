# MIND-V：认知分层视频世界模型 + GRPO 物理对齐

> **论文**：*MIND-V: Hierarchical World Model for Long-Horizon Robotic Manipulation with RL-based Physical Alignment*
>
> **作者**：Ruicheng Zhang, Mingyang Zhang, Jun Zhou, Zhangrui Guo, Zunnan Xu, Xiaofan Liu, Zhizhou Zhong, Puxin Yan, Haocheng Luo†, Xiu Li†
>
> **机构**：清华大学、X Square Robot、中山大学、HKUST、中国地质大学、中南大学
>
> **发布时间**：2025 年 12 月（arXiv v1）/ 2026 年 3 月（v2）,arXiv 2512.06628
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.06628)

---

## 一句话总结

把人脑"皮层 → 小脑"的分层运动控制迁移到视频世界模型:**SRH**(Gemini-2.5-Pro + Affordance-R1)做任务规划、**BSB**(掩码 + 三段式轨迹 + 相位转移点)做域无关中间表征、**MVG**(CogVideoX-5B + 时空引导张量注入)做像素级渲染;再用 **GRPO + PFC 奖励**(V-JEPA2 作"物理裁判"在隐空间比对预测与真值)做物理对齐,辅以推理期 **Staged Visual Future Rollouts**(Propose-Verify-Refine)抑制长时程误差累积。最终长时程任务上 PFC +9.0%、Task Success Rate +76.7%、User Preference +172.2%,并作为训练场把 OpenVLA-OFT 在 MimicGen 上成功率从 33.4% 提到 43.5%。

---

## 一、问题与动机

### 1.1 具身 AI 的数据瓶颈

VLA 训练所需的"多样 + 长时程 + 高质量"机器人操作数据极度稀缺,靠人标注成本天文。视频世界模型(VWM)本是自然出路——可无限合成轨迹。但要合成**忠实遵循指令的高质量长时程视频**,面临三个并列挑战:

1. **长时程一致性**(Long-Horizon Coherence):多个子任务间需保持因果连贯,单步错即全盘崩
2. **语义→像素生成**(Semantic-to-Pixel Generation):抽象指令(如"收拾桌面")要落到具体的时空交互上,对指令跟随精度要求极高
3. **物理合理性**(Physical Plausibility):需满足碰撞动力学、物体永续性、接触力学等基本物理律

### 1.2 现有方法的两极分化

- **(a) 视频基础模型微调**(CogVideoX、HunyuanVideo、WoW、Wan 等):指令跟随能力差,长时程视觉塌陷
- **(b) 轨迹控制生成模型**(IRASim、Cosmos、RoboMaster、MotionCtrl、DragAnything):控制精确但**严重依赖手工标注**(轨迹/掩码/锚点),无自主性,做不了大规模数据生成

两种范式都无法同时满足"自主 + 可控 + 长时程 + 物理合理"。

### 1.3 核心灵感:人脑分层运动控制

从认知科学的"**皮层 → 脊髓 → 肌肉**"分层假设出发:

- 大脑皮层:意图理解与抽象规划(相当于 SRH)
- 专门神经通路:把抽象意图翻译为具体控制(相当于 BSB)
- 小脑/运动系统:精细肌肉控制(相当于 MVG)

MIND-V 把这种"认知 → 执行"管线搬到视频世界模型中。

---

## 二、核心方法

### 2.1 整体流水线

从自然语言指令 $L$ 和初始图像 $I_0$ 开始,自顶向下:

> 1. **SRH** 把 $L$ 分解为原子子任务序列 $\{\text{SubTask}_i = (\text{ActionType}_i, \text{Object}_i, \text{Destination}_i)\}$
> 2. 对每个子任务:SRH 用 Affordance-R1 定位 $M_\text{obj}$(物体掩码)和 $P_\text{obj}$(可抓取点),VLM 规划平滑轨迹并离散化到视频帧
> 3. 把这些结构化输出打包成 **BSB**(Behavioral Semantic Bridge)
> 4. **MVG** 以 BSB 为条件,在 CogVideoX-5B 的 DiT 上做带引导注入的扩散去噪,生成该子任务的视频
> 5. **Staged Visual Future Rollouts**:每个子任务转移处采样 $K$ 份未来,由 VLM 打分,挑选最佳候选继续,必要时触发 SRH 重规划

### 2.2 Semantic Reasoning Hub (SRH)

**双模块协同**:

- **Gemini-2.5-Pro**:长时程规划、语义推理
- **Affordance-R1**:物理常识锚定(分割掩码 + 功能交互点)

这个组合把"纯 VLM 规划"升级为**physics-aware 决策引擎**:VLM 知道要抓哪个物体,Affordance-R1 知道"要抓在哪儿"。

轨迹用平滑曲线函数生成后,离散化成与视频帧对齐的点序列。

### 2.3 Behavioral Semantic Bridge (BSB)

**关键设计:域无关中间表征**。BSB 包含三元素:

| 元素 | 内容 | 作用 |
|---|---|---|
| **Object Representation** | 被操作物体掩码 $M_\text{obj}$ + 机械臂掩码 $M_\text{rob}$ | VAE 编码后按时空位置注入,保持物体身份一致 |
| **Decomposed Collaborative Trajectory** | 三段式 $(T_\text{pre}, T_\text{interact}, T_\text{post})$ 轨迹 | 每阶段有明确 active agent,显式建模"接近→交互→撤离"的物理过程 |
| **Phase Transition Points** | 帧索引三元组 $(F_\text{pre}, F_\text{interact}, F_\text{post})$ | 给每个阶段分配恰当帧数,避免"瞬移式"动作 |

**域无关**的含义:BSB 只描述"任务逻辑",不含具体外观。换场景、换物体,只要能生成对应掩码和轨迹,MVG 就能渲染出对应视频——这是 OOD 泛化的关键。

### 2.4 Motor Video Generator (MVG)

MVG 是 BSB 到视频的可学习"物理引擎",基于 CogVideoX-5B:

- **VAE**:输入 $[B,3,T,H,W]$ → 潜空间 $[B,16,T/4,H/8,W/8]$
- **DiT 主干**:30 个 block,隐维 1920,patch size $2\times 2\times 2$,3D 正弦位置编码

**BSB 注入机制**:BSB 被光栅化为时空引导张量 $[B,128,T/4,H/8,W/8]$ → 3×3 空间卷积 → 1D 时间卷积 → 得到 1920 维特征 $G$,与潜视频同分辨率。在**偶数号 DiT block 中**通过加性融合注入:

$$h_\text{new} = h + \text{norm}(G) \cdot G \tag{1}$$

其中 $\text{norm}(\cdot)$ 是 Group Normalization。奇偶交替注入策略既保证了动力学约束的贯穿,又保留了 DiT 原生的视频生成先验。

### 2.5 Test-Time Optimization: Staged Visual Future Rollouts

**问题**:长时程预测固有的误差累积——前一段小偏差,后面滚成完全失败。

**方案**:每个子任务转移点做 **"Propose-Verify-Refine"** 循环:

> 1. **Propose**:SRH 采样 $K$ 个语义可行的 BSB 候选(不同轨迹、不同分解)
> 2. **Verify**:MVG 并行 rollout 出 $K$ 段短视频 $\{V_1, \ldots, V_K\}$;VLM 从 planner 转为 **Critic**,按任务完成度 $C_1 \in \{0,1\}$、物理合理性 $C_2 \in [0,1]$、视觉质量 $C_3 \in [0,1]$ 打分,$C = C_1 + C_2 + C_3$
> 3. **Refine**:若最高分超过阈值则采用;否则 VLM 给出结构化失败反馈(如"末端位置偏移"、"抓取不稳"),触发 SRH 重规划

这等于把"System 2 慢思考"嵌入生成过程,把全局长时程规划**分解为一串局部最优决策**。主实验 $K=3$。

### 2.6 两阶段训练

**Stage 1: SFT**

$$\mathcal{L}_\text{SFT}(\theta) = \mathbb{E}_{(x_0, \text{BSB}) \sim \mathcal{D}, \epsilon \sim \mathcal{N}(0,I), t}\left[\|\epsilon - \epsilon_\theta(x_t, t, \text{BSB})\|^2\right] \tag{2}$$

在 Bridge V2 数据集上 fine-tune,30,000 步,lr $2\times 10^{-5}$。注意只用**子任务短视频**训练,层次架构让它自然泛化到任意长任务序列。

**Stage 2: GRPO Post-Training with Composite Reward**

$$R(x_0) = w_p \cdot R_\text{physics}(x_0) + w_a \cdot R_\text{aesthetic}(x_0), \quad w_p=0.2,\ w_a=1.0 \tag{3}$$

这两项分别负责**物理对齐**和**视觉美观**,下文展开。

---

## 三、Physical Foresight Coherence (PFC) 奖励

这是本文最核心的原创设计:用另一个预训练世界模型(**V-JEPA2**)当"物理裁判"。

### 3.1 V-JEPA2 作为物理先验

V-JEPA2 已经在大规模真实视频上自监督训练,并在机器人数据上 finetune,因此其编码器 $E_v$ 和预测器 $P_v$ 内化了世界动力学。给定"上下文"片段,V-JEPA2 能预测接下来 latent 该长什么样。

### 3.2 滑动窗口一致性打分

每个生成视频被切成带步长 5 的滑动窗口,每个窗口 15 帧为 context + 5 帧为 target。窗口 $i$ 的物理一致性分数:

$$\text{PFC}: s_i = \text{sim}_\cos\left(P_v(E_v(x_\text{context}^{(i)})),\ E_v(x_\text{target}^{(i)})\right) \tag{4}$$

$s_i$ 高 ⟺ 生成视频的动力学与 V-JEPA2 的物理理解一致。

### 3.3 Softmax 聚焦最差窗口

直接取平均会被大量"正常窗口"淹没最关键的违反。用温度 $\tau$ 控制的 softmax 重新加权,突出**错误最严重**的窗口:

$$R_\text{physics}(x_0) = \sum_{i=1}^{N_w} \frac{\exp((1-s_i)/\tau)}{\sum_{j=1}^{N_w}\exp((1-s_j)/\tau)} \cdot s_i \tag{5}$$

$\tau$ 越小越 focus 在 single worst-offending 窗口。这把奖励从"温和评估"变成"对动态因果链中最危险环节的定点打击"。

### 3.4 Aesthetic Reward

用 Gemini-2.5 对视频的清晰度、伪影、真实感打 1–5 整数分,作为 $R_\text{aesthetic}$。

### 3.5 GRPO 优化

对每个 prompt 采样 $G$ 个视频 $\{x_0^i\}$,组内归一化得到优势:

$$\hat{A}^i = \frac{R(x_0^i) - \text{mean}(\{R(x_0^j)\})}{\text{std}(\{R(x_0^j)\})} \tag{6}$$

PPO-style clipped 目标 + KL 正则向 SFT 初始化 $\pi_\text{ref}$ 靠拢:

$$\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\left(\min(r_i\hat{A}^i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}^i) - \beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})\right)\right] \tag{7}$$

其中 $r_i = \pi_\theta(x_0^i)/\pi_\text{ref}(x_0^i)$。1,500 次迭代,lr $5\times 10^{-5}$,KL 防止 reward hacking。

---

## 四、实验

### 4.1 Setup

- **SRH**:Gemini-2.5-Pro + Affordance-R1
- **MVG**:CogVideoX-5B 初始化
- **数据**:Bridge V2,遵循 RoboMaster 的数据处理协议
- **分辨率 / 长度**:480×640,每子任务 37 帧;长时程任务 2–4 子任务 ≈ 111 帧
- **VRAM**:推理约 50GB,**峰值恒定 70GB**(autoregressive 复用显存,子任务数线性扩展时间、常数显存)
- **硬件**:4 × H200

### 4.2 主结果

#### 短时程任务(V-Bench 纯视觉指标,表 1)

MIND-V 在 Aesthetic、Temporal Flicker、Motion Smoothness、Subject Consistency、Bg. Consistency 上都拿到 best 或并列 best,即使对比轨迹输入的 IRASim/RoboMaster/Tora 等"特权方法"。

#### 长时程任务(表 2)

| 方法 | PFC ↑ | Task Success Rate ↑ | User Pref. (%) ↑ |
|---|---|---|---|
| Robodreamer | 0.418 | 0.275 | 7 |
| WoW-1-DiT-7B | 0.423 | 0.322 | 11 |
| WoW-1-Wan-14B | 0.420 | 0.347 | 14 |
| CogVideoX-5B | 0.406 | 0.081 | 0 |
| HunyuanVideo | 0.411 | 0.098 | 1 |
| Dreamdojo | 0.424 | 0.333 | 18 |
| **MIND-V** | **0.462** | **0.613** | **49** |

相对 second-best:**PFC +9.0%、Task Success +76.7%、User Pref. +172.2%**。

### 4.3 作为 VLA 训练场(表 3)

在 MimicGen 3 个任务(Coffee、StackThree、Square)上,每任务 128 demo + 300 expert:

| 方法 | Coffee | StackThree | Square | Mean |
|---|---|---|---|---|
| Base policy(OpenVLA-OFT) | 32.6 | 30.2 | 20.2 | 27.7 |
| IL(继续训练) | 37.4 | 36.7 | 26.1 | 33.4 |
| **MIND-V + IL** | **51.7** | **48.3** | **30.4** | **43.5** |

损失函数:

$$\mathcal{L} = \lambda_\text{pix}\|\hat{o} - o\|_1 + \lambda_\text{perc} d_\text{LPIPS}(\hat{o}, o) + \lambda_\text{IL}\mathcal{L}_\text{pose} \tag{8}$$

用 MIND-V 合成的成功 rollout 作为 visual goal,和 VLA 实际执行的帧做 L1 + LPIPS 对齐,再加 pose loss。

**作者给出的深层洞察**:像素空间世界模型胜过 latent-space 世界模型,因为它和 VLA 预训练数据**同处一个像素空间**,VLA 可以直接复用视觉先验;latent 空间的 imagination 与 VLA 的感知表征脱节,难以直接利用。

### 4.4 消融(表 4,长时程)

| 变体 | Aesthetic ↑ | Imaging ↑ | PFC ↑ | subtask Avg. ↑ |
|---|---|---|---|---|
| (a) w/o GRPO | 0.491 | 0.675 | 0.429 | 0.582 |
| (b) Replace Affordance w/ YOLO+SAM2 | 0.498 | 0.680 | 0.445 | **0.455** ↓ |
| (c) w/o Staged Rollouts | 0.482 | 0.671 | 0.438 | **0.327** ↓↓ |
| (d) Replace Gemini w/ Qwen3-VL | 0.500 | 0.679 | 0.452 | 0.567 |
| **MIND-V (Full)** | **0.504** | **0.684** | **0.462** | **0.613** |

核心结论:

- **Staged Rollouts 最关键**(去掉后 Success 从 0.613 → 0.327,直接腰斩);说明"System 2"思考对误差累积的压制作用是决定性的
- **Affordance 模块次之**(去掉后 Success 0.613 → 0.455);精确 grounding 对 BSB 的 functional correctness 至关重要
- **GRPO 主要影响物理合理性**(PFC 0.462 → 0.429),对 success rate 也有贡献
- **VLM 替换仅微降**,说明框架对具体 VLM 不强依赖

### 4.5 Rollout 数量 $K$(表 7)

| $K$ | Time (s) | Peak VRAM (GB) | Success (%) | PFC |
|---|---|---|---|---|
| 1 | 144.5 | 31.8 | 35.2 | 0.405 |
| 2 | 167.1 | 50.5 | 51.7 | 0.428 |
| **3** | **181.6** | **70.1** | **61.3** | **0.445** |
| 4 | 199.7 | 94.1 | 62.1 | 0.447 |
| 5 | 223.4 | 122.0 | 62.5 | 0.450 |

$K=3$ 是性价比拐点,$K=5$ 只比 $K=3$ 好 1.2pp 但显存几乎翻倍。

---

## 五、局限性与未来方向

1. **依赖商业 API**:SRH 默认用 Gemini-2.5-Pro(开源替代 Qwen3-VL 有掉点),不利于完全离线部署
2. **PFC 依赖 V-JEPA2 质量**:V-JEPA2 本身的 OOD 理解是否足够覆盖训练集之外的物理模式,未给出系统验证;若 V-JEPA2 自己"物理不懂",PFC 就成 noise
3. **下游 VLA 实验规模小**:只在 MimicGen 3 个任务 ×128 demo 验证,真机未做
4. **合成视频直接做 VLA 监督**:L1 + LPIPS 对生成视频的视觉细节高度敏感,如果生成帧有局部瑕疵(比如贴图错位),会误导策略学习。本文未讨论这个风险
5. **训练 cost 较高**:CogVideoX-5B + 30K SFT + 1.5K GRPO iterations + 4×H200,对学术组不友好

---

## 六、个人思考

### 6.1 和 ViVa 的深层对照

ViVa(2026.04,GigaAI)和 MIND-V(2026.03,清华)分别代表"视频生成模型在机器人 RL 中的两种用法":

| 维度 | ViVa | MIND-V |
|---|---|---|
| 视频模型角色 | **value function**(value head) | **world model**(data generator) |
| 底座 | Wan2.2 视频 DiT | CogVideoX-5B 视频 DiT |
| 核心产出 | 标量 value + 未来 proprio | 长视频序列 |
| 下游用途 | 替换 RECAP 的 VLM value 做 advantage 估计 | 合成训练数据给 VLA + 视觉对齐损失 |
| 推理速度 | 0.18 s/帧(1 步 DDIM) | ~60s/子任务(多步去噪 + Propose-Verify-Refine) |
| RL 身份 | 被训练的 value model | 用 GRPO **训练自己** |

两者互补:**ViVa 把视频模型压到决策 loop 里当 critic**,**MIND-V 把视频模型放到数据侧当生产车间**。一个模型做好一件事——这才是合理的专业化。

### 6.2 PFC 奖励 = "Critic 分离"

传统 reward model 要么端到端学(成本高,数据难),要么靠规则/VLM 打分(噪声大)。PFC 提供了第三条路:**冻结一个强 world model 作为"物理评判模块"**,把奖励信号从文本/评分空间搬到 latent space 的预测一致性。这个思路可泛化:

- **PFC for VLA 奖励**:用 V-JEPA2 判别 VLA 的 rollout 是否符合物理常识,替代 rule-based 或 VLM-as-judge
- **PFC for Text-to-Video**:评判生成视频的物理合理性,免人工 preference 标注

本文引用的 Yuan et al. 2025(arXiv:2510.21840)已在做类似的"V-JEPA2 reward signal"工作,但 MIND-V 把它专门改造成了 softmax-focused 的形式,是工程上更精致的变体。

### 6.3 BSB 的"域无关中间表征"思想

BSB 的关键价值不是"多准确",而是**解耦任务逻辑和视觉外观**。同一个 BSB(掩码 + 轨迹 + 相位点)换个场景也能渲染——OOD 场景的 peach picking、dumpling lifting、cabinet opening 都能迁移,正是这种解耦带来的。

这和 Nvidia Cosmos 的"结构化控制语言"、Google RoboCat 的"task embeddings"属于同一思想谱系:**在"感知像素"和"动作轨迹"之间显式塞一层几何/拓扑中间层**。后续工作可以考虑给 BSB 加入力学 primitives(如接触、推挤、倒水)。

### 6.4 Staged Rollouts 的价值判断

消融显示 Staged Rollouts 去掉后 success 从 0.613 崩到 0.327,这是**单个组件贡献最大的**。本质上它是在**推理时花额外 compute 换准确率**(test-time scaling),且天然给 world model 带来"自我反思"能力。

但这也暴露一个问题:离开 Staged Rollouts 后,生成模型本身并没有 long-horizon 连贯性——真正的 long-horizon 能力其实住在**"SRH + VLM Critic + 多候选"循环**里,而不是 MVG 自己。这意味着 VWM 本身仍然是短视的,长时程靠 agent-level 搜索续命。

### 6.5 与 WMPO / WoVR / VLA-RFT 的对比

同样用视频世界模型辅助 VLA 训练的路线,WMPO / WoVR / VLA-RFT 是**在世界模型里跑 RL**(用 WM 作模拟器),MIND-V 是**用世界模型合成 demo 做 IL**。本质差别:

- WMPO 路线:policy 闭环交互 WM → 需要 WM 反应快、可微或可查询奖励
- MIND-V 路线:WM 一次性产出 target video → 只需 WM 产物质量高

两者并不矛盾:**MIND-V 产出的高质量 demo 完全可以作为 WMPO 路线的 bootstrapping 数据**。

---

## 参考

- **V-JEPA2**(Assran et al. 2025):本文作 PFC 物理裁判的核心组件
- **CogVideoX-5B**(Yang et al. 2024):MVG 的底座
- **RoboMaster**(Fu et al. 2025):数据处理协议 + 轨迹控制生成的前序代表
- **Dreamdojo**(Gao et al. 2026):long-horizon VWM baseline,最强对手之一
- **WoW**(Chi et al. 2025):14B 视频 world model,long-horizon baseline
- **OpenVLA-OFT**(Kim et al. 2025):下游 VLA 实验的 base policy
- **Affordance-R1**(Wang et al. 2025):SRH 的视觉锚定组件
- **Flow-GRPO**(Liu et al. 2025):GRPO 训练 flow matching 的方法学参考
- **ViVa(2026)**:同期视频 DiT 做 value model 的工作,与 MIND-V 形成"同 backbone、不同用途"对照
