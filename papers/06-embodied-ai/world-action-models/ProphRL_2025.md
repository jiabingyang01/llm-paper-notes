# ProphRL：用世界模型"预言"未来强化 VLA 动作策略

> **论文**：*Reinforcing Action Policies by Prophesying*
>
> **作者**：Jiahui Zhang、Ze Huang（共同一作）、Chun Gu、Zipei Ma、Li Zhang（通讯）
>
> **机构**：复旦大学数据科学学院（School of Data Science, Fudan University）、上海创新研究院（Shanghai Innovation Institute）、Logos Robotics
>
> **发布时间**：2025 年 11 月（arXiv 2511.20633）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.20633) | [PDF](https://arxiv.org/pdf/2511.20633)
>
> **分类标签**：`世界模型` `VLA强化学习` `Flow-GRPO` `动作条件视频生成` `机器人操作`

---

## 一句话总结

论文提出 **ProphRL**：先在 3100 万+条异构机器人轨迹上预训练一个动作条件视频世界模型 **Prophet**（可少样本适配新场景），再用为 flow-matching 动作头量身定制的 **FA-GRPO**（环境级动作 PPO 比率聚合）与 **FlowScale**（按噪声尺度重加权 flow 内部步梯度）在该世界模型中做闭环 RL 后训练 VLA 策略，在公开基准（SimplerEnv/BRIDGE、LIBERO）上取得 5%~17% 的成功率提升，在真实 UR30e 机器人上取得 24%~30% 的提升。

## 一、问题与动机

VLA（Vision-Language-Action）策略目前主要靠模仿学习训练，存在目标错配问题：似然目标不直接优化长程任务成功率，策略在分布漂移下容易变脆、误差沿时间步累积。用 RL 直接优化任务奖励可以缓解这一问题，但

- 真实机器人交互成本高、并行度低、常需人工介入；
- 经典物理仿真器工程量大，且 RGB 策略在仿真到真实之间存在明显视觉域差距；
- 离线 RL 缺乏与当前策略的闭环数据，长程信用分配弱。

数据驱动的世界模型提供了折中方案：以与 VLA 相同的视觉接口，在"想象"中让策略反复练习。但现有工作大多局限于单场景世界模型，即便与 VLA 结合也主要当作数据增广器而非可泛化的仿真器；少数把世界模型当仿真器做 RL 后训练的工作，聚焦于"替换一个已有仿真器"，没有解决"如何获得一个能在真实预算下少样本适配新具身/新任务/新场景、且始终在真实世界中可用的通用世界模型"这一核心问题。ProphRL 试图同时解决数据效率（世界模型替代昂贵真实交互）与优化稳定性（RL 算法适配 flow 动作头）两个问题。

## 二、核心方法

ProphRL 由三部分组成：动作条件世界模型 **Prophet**、面向 flow 动作头的策略梯度算法 **FA-GRPO**、梯度重加权技术 **FlowScale**，外加一个离线奖励模型（RM）提供轨迹级奖励。

### 2.1 Prophet：历史感知、双重动作条件的视频世界模型

Prophet 建立在潜空间视频扩散管线之上，从 Cosmos-Predict2-2B-Video2World 初始化，用 Wan2.1 视频自编码器（4×8×8 压缩）编码/解码，DiT 去噪器学习标准的潜空间加噪-去噪目标：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{\mathbf{z}_0,\epsilon\sim\mathcal{N}(0,I),t}\left[\|\epsilon-\epsilon_\theta(\mathbf{z}_t,t,f)\|_2^2\right]$$

用大白话说：这就是普通的扩散模型训练——给干净视频潜变量加噪声,让网络学会在给定条件 $f$（这里是首帧+动作）下把噪声"还原"回真实视频。

**动作定义**。每个末端执行器在每个时间步的动作是 7 维向量

$$c_{t,n} = [\Delta p_{t,n}^\top, \Delta e_{t,n}^\top, g_{t,n}]^\top \in \mathbb{R}^7$$

分别是平移增量、欧拉角旋转增量、夹爪开合度，其中末端位姿以相对上一时刻局部坐标系的小刚体运动表示（$\xi_{t,n}=\xi_{t-1,n}\circ\Delta\xi_{t,n}$），跨数据集统一将末端数 $N$ 固定为 2（对齐 AgiBot 双臂），不足的末端补零，从而在异构数据集上共享同一套动作参数化。

**双重动作条件**：
1. **标量动作流**：把整段动作 chunk 展平后经 MLP 映射为一个全局向量 $f_{sa}=\phi(c_{1:T})$，直接加到 DiT 的时间步嵌入上；
2. **动作帧流**（可选、更几何化）：利用相机内外参把每个末端的位置与三条局部坐标轴投影到图像平面，渲染成黑底彩色圆盘+方向线段的 2D "动作帧"视频，圆盘半径随深度单调衰减（更近更大）、颜色编码夹爪开合度；再用同一个视频自编码器编码，经 $1\times1\times1$ 卷积升维、$1\times3\times3$ 深度可分离卷积和逐点卷积做轻量 3D 投影，沿空间维平均池化并加正弦位置编码，最后经 MLP 得到 $f_{af}$，与标量条件一起叠加到时间步嵌入：$\tilde t=\bar t+f_{sa}+f_{af}$。

**历史感知机制**：仿照 FramePack，把历史帧的低分辨率潜变量经多尺度 3D 平均池化压缩成一个 memory 矩阵，映射出额外的 K/V 向量，在每个 DiT block 中以拼接记忆的注意力方式提供长程时序上下文，兼顾几何/接触的稳定预测和计算开销。

**长程闭环 rollout**：以分段自回归方式生成——首帧初始化历史缓冲区，每次给定一个动作 chunk，Prophet 生成一小段 clip，末帧作为下一段起点，clip 压入历史缓冲区，如此滚动到任意长度（算法见论文 Algorithm 1）。

**光流引导的评测协议**：作者指出 PSNR/SSIM 等帧级指标只衡量全局视觉保真度，对"动作是否被正确执行"不敏感（例如抽屉半开与全关在像素上差异很小）。为此提出用 Farnebäck 光流计算生成 rollout 与真实视频逐帧的运动场，用端点误差衡量幅值一致性：

$$\text{EPE}_t = \frac{1}{HW}\sum_x \|\mathbf{u}_t(x)-\hat{\mathbf{u}}_t(x)\|_2$$

用方向余弦相似度衡量方向一致性（只在运动显著区域 $V_t$ 统计）：

$$\cos_t = \frac{1}{|V_t|}\sum_{x\in V_t}\frac{\langle \mathbf{u}_t(x),\hat{\mathbf{u}}_t(x)\rangle}{\|\mathbf{u}_t(x)\|_2\|\hat{\mathbf{u}}_t(x)\|_2+\varepsilon}$$

用大白话说：PSNR/SSIM 只看"画面像不像"，而光流指标专门看"手臂/物体动得对不对、有没有正确建立接触"，能把视觉上都很清晰但动作执行错误的 rollout 区分出来。

### 2.2 FA-GRPO：面向环境级动作聚合的 Flow-GRPO

flow 动作头把单个 chunk 的对数似然按内部去噪步 $k=1,\dots,K$ 分解：

$$\log \pi_\theta(a_{s,c,d}\mid o_s) = \sum_{k=1}^K \log \pi_\theta^{(k)}(a_{s,c,d}\mid o_s)$$

原始 Flow-GRPO 把每个内部 flow 步都当作一个原子动作，按 $(s,c,k)$ 逐步构造 PPO 比率再对 $k$ 求和。FA-GRPO 改为先把 $K$ 个内部步的对数似然求和聚合成一个"环境级动作"对数概率 $\ell_{s,c,d}$，再在 $(s,c,d)$ 层面（$s$ 为外层环境步，$c$ 为 chunk 索引，$d$ 为动作维度）构造比率：

$$r_{s,c,d}=\exp(\ell_{s,c,d}-\ell_{s,c,d}^{\text{old}})=\dfrac{\pi_\theta(a_{s,c,d}\mid o_s)}{\pi_{\text{old}}(a_{s,c,d}\mid o_s)}$$

再用一个动作级 advantage $\hat A_{s,c}$（对每对 $(s,c)$ 唯一，广播到所有 $d,k$）做 clip 目标 + KL 正则：

$$\mathcal{L}_{\text{FA-GRPO}}(\theta)=-\mathbb{E}\left[\sum_{s,c,d}M_{s,c}\,f_{\text{clip}}(r_{s,c,d},\hat A_{s,c})\right]+\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$$

其中 $f_{\text{clip}}(r,A)=\min\{rA,\ \text{clip}(r,1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})A\}$，$M_{s,c}$ 是处理变长 episode 提前终止的掩码。用大白话说：不再把 flow 去噪的每一小步都当成一次独立"动作"来算 PPO 比率，而是先把整条 flow 轨迹的似然汇总成"这个环境动作发生的概率"，让 RL 的更新粒度和环境交互的真实粒度对齐，避免内部步之间不必要的比率震荡。

### 2.3 FlowScale：按噪声尺度重加权 flow 内部步梯度

作者观察到基于 SDE 的 flow 头在不同内部步 $k$ 上梯度幅值高度不均匀——低噪声（大 $k$，refinement 阶段）步会天然主导梯度更新。论文先做了一个高斯近似下的理论推导：把第 $k$ 步似然近似为均值 $\mu_{s,k}$、方差 $\sigma_{s,k}^2 I$ 的各向同性高斯，则对均值的得分 $\nabla_{\mu_{s,k}}\log\pi_\theta^{(k)}=(a-\mu_{s,k})/\sigma_{s,k}^2$，其期望平方范数满足

$$\mathbb{E}\big[\|\nabla_{\mu_{s,k}}\log \pi_\theta^{(k)}\|^2\big]\propto \sigma_{s,k}^{-2}$$

即噪声越小的步，得分范数天然越大、越容易主导梯度。据此，一个方差平衡的权重选择应满足 $w_{s,k}^\star\propto\sigma_{s,k}$（噪声越大的步反而应被放大权重）。

具体实现上，用 flow/扩散噪声表提供逐步噪声尺度 $\sigma_{s,k}$，按 normalize–mix–clip 规则构造停梯度权重：

$$\tilde w_{s,k}=(\sigma_{s,k}^2+\varepsilon)^p,\qquad w_{s,k}=\text{clip}\!\left((1-\alpha)\dfrac{\tilde w_{s,k}}{\frac1K\sum_{j=1}^K\tilde w_{s,j}}+\alpha,\ w_{\min},w_{\max}\right)$$

实现中取 $p=0.5$，使 $\tilde w_{s,k}\propto\sigma_{s,k}$ 恰好符合理论建议的方差平衡解；归一化保证 $\frac1K\sum_k w_{s,k}=1$（不改变梯度整体尺度），$\alpha$ 混合均匀基线防止权重塌缩到单一步，clip 限制单步权重范围。最终 loss 与 FA-GRPO 形式相同，只是把 advantage 换成加权版本：

$$\mathcal{L}_{\text{FlowScale}}(\theta)=-\mathbb{E}\left[\sum_{s,c,d}M_{s,c}\,f_{\text{clip}}(r_{s,c,d},w_{s,k}\hat A_{s,c})\right]+\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$$

用大白话说：flow 去噪链条上"最后精修的几步"天然梯度更大、更容易抢占学习信号，FlowScale 相当于给早期（噪声大、方向性强）的粗粒度步"补贴"梯度权重，让整条去噪链上的学习贡献更均衡，从而稳定长程 RL 更新。作者强调这只是一个启发式的对角预条件，理论推导为近似的启发（忽略了 clip、KL 及步间相关性）。

### 2.4 奖励模型与优势计算

RM 在轨迹级给出标量分数 $R_i=f_{\text{RM}}(\tau^{(i)},\text{text}_i)$：LIBERO 用微调过的 Qwen2.5-VL-7B 二分类器（对仿真器 rollout 训练，再用"仿真器真值标签监督世界模型渲染视频"的域桥接策略训练第二个 RM 以适配世界模型视觉域）；BRIDGE 与真机则直接用 Qwen2.5-VL-72B 做零样本、思维链式的成功/失败判别（5 次采样多数投票）。随后按 GRPO 方式做组内归一化并广播为逐 chunk 优势：

$$\hat A_{s,c}^{(i)} = \tilde R_i M_{s,c}^{(i)},\qquad \tilde R_i=\frac{R_i-\mu_{\mathcal G}}{\sigma_{\mathcal G}+\varepsilon_R}$$

## 三、实验结果

**预训练规模**：Prophet 在 AgiBot、DROID、LIBERO 及精选的 Open-X 子集（Austin Sailor、DLR Wheelchair、BC-Z、CMU Stretch、Furniture Bench、NYU Franka Play、RT-1 等）上混合预训练，共 3100 万+条采样轨迹；模型 2.058B 参数，DiT 通道维 1024，64×H200 训 2 个 epoch；微调用 LoRA（rank16）+ 8×H200。

**世界模型保真度**（Table 1，预训练 Prophet 在各数据集留出轨迹上）：

| 数据集 | PSNR↑ | SSIM↑ | EPE(均值)↓ | cos(均值)↑ |
|---|---|---|---|---|
| AgiBot | 27.05 | .8916 | .2959 | .2144 |
| DROID | 25.23 | .8813 | .2574 | .1532 |
| Open-X | 27.25 | .8810 | .4521 | .0843 |
| LIBERO | 26.29 | .9075 | .1660 | .4164 |

**BRIDGE 微调对比**（Table 2，与 LTX-Video、Genie-envisioner、Cosmos-Predict2 对比；全数据微调 30k 步）：Prophet 在同等 PSNR/SSIM 水平下动作一致性显著更优，尤其在少样本（同任务/跨物体迁移）场景差距更大。全数据设置下 Prophet PSNR 25.47 / EPE(均值) 0.9136 / cos(均值) .2234，均优于最强基线 Cosmos-Predict2（24.58 / 1.0243 / .2051）。

**组件消融**（Table 3，自建 UR30e 数据）：历史感知 + 预训练 + 动作帧条件逐项叠加，PSNR 从 24.28 提升到 26.12，EPE(均值) 从 .5396 降到 .4345。

**SimplerEnv-WidowX 单任务 RL**（Table 5，四个 WidowX 任务总体成功率，均值±标准差）：

| VLA 骨干 | SFT | +FA-GRPO | +FA-GRPO&FlowScale |
|---|---|---|---|
| VLA-Adapter-0.5B | 23.3±2.2 | 38.2±2.4 (+14.9) | 41.0±2.4 (+17.7) |
| Pi0.5-3B | 38.9±2.6 | 46.9±3.0 (+8.0) | 51.0±1.2 (+12.1) |
| OpenVLA-OFT-7B | 25.0±1.8 | 29.2±1.8 (+4.2) | 30.9±0.6 (+5.9) |

即论文摘要所称"公开基准 5%~17% 提升"的来源；FlowScale 相对 FA-GRPO 单独使用总能进一步提升。多任务联合 RL（Table 6）与将世界模型换成 Cosmos-Predict2（Table 7）的消融均表明：Prophet 作为 RL 后端优于 Cosmos-Predict2，且在仅 400 条样本的少样本微调场景下优势保持得更好。少样本 RL 数据量消融（Table 8）显示仅用 10 张图像起步也能把 VLA-Adapter-0.5B 总体成功率从 23.3 拉到 34.7（+11.4）。

**真机 UR30e 结果**（Table 9，四任务：GraspBottle / PickBowl / PulloutTissue / PlaceCube）：

| VLA 骨干 | SFT 总体 | +FA-GRPO&FlowScale 总体 |
|---|---|---|
| VLA-Adapter-0.5B | 35.8±3.1 | 60.4±0.7 (+24.6) |
| Pi0.5-3B | 52.1±3.8 | 82.1±0.7 (+30.0) |
| OpenVLA-OFT-7B | 35.4±0.7 | 62.9±0.7 (+27.5) |

对应摘要中"真机 24%~30% 提升"，且仅需 100 步 RL（相对 SFT 的 50k 步）。定性上，RL 后策略在 PlaceBowl 任务上学会了演示数据中几乎不存在的"右侧抓取"新模式（SFT 策略只能以极低概率偶发生成、RL 把这一弱模式放大为稳定行为）；在软体接触敏感的 PulloutTissue 任务上，RL 显著收紧了抓取接近位姿的分布，提升了对形变接触的鲁棒性。

**LIBERO：仿真器内 RL vs. 世界模型内 RL**（Table 10）：

| 设置 | 总体成功率 | 所需 RL 更新数 |
|---|---|---|
| VLA-Adapter（SFT） | 79.9±2.2 | — |
| 仿真器 +FA-GRPO | 87.8±3.2 (+7.9) | 259–409 |
| 仿真器 +FA-GRPO&FlowScale | 90.1±3.5 (+10.2) | 119–179 |
| Prophet 世界模型 +FA-GRPO | 82.3±0.7 (+2.9) | 100（固定预算） |
| Prophet 世界模型 +FA-GRPO&FlowScale | 84.5±1.1 (+5.1) | 100（固定预算） |

世界模型内 RL 的增益小于有仿真器可用时的 RL（长程 rollout 的几何/接触漂移叠加 RM 偏差，使信用分配更难），但在没有高保真仿真器（如软体接触、真机场景）时仍能提供有效、更新预算更小的训练信号。论文还专门做了 RM 质量诊断实验（Fig. 14），结论是 FA-GRPO 有效的关键在于 RM 保持**高召回**（不漏判真正成功的轨迹）且精度相对稳定，而单纯压低假阳性率（FPR）既非必要也非充分条件。

## 四、局限性

- **计算开销大**：RL 阶段策略需要与 2B 参数的 Prophet 交互生成闭环 rollout，这主导了训练成本、限制了可负担的迭代次数；作者在结论中明确指出需要架构简化、蒸馏为更小 student、跨 rollout 特征缓存或专用推理 kernel 才能扩展到更长时程、更大任务集和更丰富的策略探索。
- **世界模型内 RL 增益明显小于"有仿真器可用"的上限**（LIBERO 上 +5.1 vs +10.2），说明长程 rollout 的几何/接触漂移与 RM 噪声会削弱信用分配质量，Prophet 定位是"仿真器不可得/不可负担时的替代方案"而非全面替代高保真仿真器。
- **动作帧条件依赖相机内外参**：在 LIBERO 的在线伺服控制 RL 设置下无法获取真实相机参数，因此该设置下 Prophet 关闭了动作帧条件，只能退化为标量动作流，损失了部分几何精度。
- **奖励模型本身是噪声来源且需要针对场景专门构造**：LIBERO 用"仿真器真值标签监督世界模型渲染视频"的半合成域桥接策略训练第二个 RM，作者承认由此产生的奖励噪声较大、训练稳定性因任务而异；真机 RM 提示词作者自陈"并非最优"。
- Open-X 子集因缺乏可靠相机参数，只能用标量动作流训练，无法享受动作帧条件带来的额外精度增益，预训练数据的条件粒度并不统一。

## 五、评价与展望

**优点**：ProphRL 把"用世界模型做 VLA 的 RL 后端"这一研究线索推进到了更工程完整、更贴近真实约束的程度——不再局限于单一场景/单一仿真器替代，而是通过大规模异构数据预训练 + 少样本适配，第一次系统性地展示了一个通用动作条件世界模型能否作为**跨具身、跨场景**的可复用 rollout 生成器。光流引导评测协议是一个简单但实用的补充度量，切中了"PSNR/SSIM 无法区分动作是否正确执行"这一长期被忽视的评测缺口，具有较强的推广价值。FA-GRPO 与 FlowScale 对 flow-matching 动作头做 RL 时的梯度异方差问题给出了干净的理论直觉（得分范数 $\propto\sigma^{-2}$）和轻量实现，且在三种不同规模/结构的 VLA（0.5B/3B/7B）上都验证了一致收益，说明该组件具有一定的骨干无关性。

**与其他公开工作的关系**：相比 DreamGen、Genie-envisioner、Cosmos-Predict2、Enerverse-AC 等把视频生成模型用作动作条件世界模型的工作，Prophet 的差异化在于历史感知记忆（FramePack 式）+ 双重（标量/几何投影）动作条件的组合设计，以及针对跨数据集动作参数化不统一问题的显式对齐（固定末端数、局部 delta pose、gripper 归一化）；相比 Flow-GRPO 及其后续变体（如 ReinFlow）"保持逐内部步 PPO 比率"的做法，FA-GRPO 选择在环境动作粒度聚合比率，这与近期一些工作对 chunk-level/action-level advantage 广播的探索方向一致，但论文并未与 VLA-RL、SimpleVLA-RL、WMPO 等同期"世界模型/RL 后训练 VLA"工作做直接数值对比，其相对优势更多体现在自身消融（Table 7：Prophet vs Cosmos-Predict2 作为 RL 后端）而非跨论文比较。

**开放问题与可能的改进方向**：(1) 世界模型内 RL 与仿真器内 RL 之间约 5 个百分点的性能差距如何进一步收窄，是否可以通过更好的不确定性建模、集成多个 rollout 或显式建模模型误差来改善长程信用分配；(2) RM 质量诊断实验（Fig. 14）表明 RL 效果对 RM 的召回率高度敏感，如何在缺乏真值标签的真实世界场景中低成本地维持高召回 RM，仍是一个开放且实践上关键的问题；(3) 论文的计算开销分析提示"世界模型蒸馏/加速"是把该范式规模化到更长时程、更大动作空间任务的必要方向，也是评估这类方法能否走向大规模落地的关键指标。

## 参考

- Jie Liu, Gongye Liu, Jiajun Liang, et al. Flow-GRPO: Training flow matching models via online RL. NeurIPS, 2025.（FA-GRPO 的直接基线与改造对象）
- Niket Agarwal, Arslan Ali, Maciej Bala, et al. Cosmos World Foundation Model Platform for Physical AI. arXiv, 2025.（Prophet 的初始化基座 Cosmos-Predict2）
- Physical Intelligence, Kevin Black, Noah Brown, et al. π0.5: A vision-language-action model with open-world generalization. CoRL, 2025.（论文所用三个 VLA 骨干之一）
- Yihao Wang, Pengxiang Ding, Lingxiao Li, et al. VLA-Adapter: An effective paradigm for tiny-scale vision-language-action model. arXiv, 2025.（论文所用最小规模 VLA 骨干）
- Yue Liao, Pengfei Zhou, Siyuan Huang, et al. Genie Envisioner: A unified world foundation platform for robotic manipulation. arXiv, 2025.（世界模型基线之一，同为动作条件视频生成路线）
