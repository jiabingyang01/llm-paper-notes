# SpatialVAM：空间感知多视图视频扩散作为数据高效的机器人策略

> **论文**：*SpatialVAM: Spatial-Aware Multi-View Video Diffusion as a Data-Efficient Robot Policy*（arXiv 公开版标题为 *Multi-View Video Diffusion Policy: A 3D Spatio-Temporal-Aware Video Action Model*，简称 **MV-VDP**）
>
> **作者**：Peiyan Li, Yixiang Chen, Yuan Xu, Jiabing Yang, Xiangnan Wu, Jun Guo, Nan Sun, Long Qian, Xinghang Li, Xin Xiao, Jing Liu, Nianfeng Liu, Tao Kong, Yan Huang, Liang Wang, Tieniu Tan
>
> **机构**：中科院自动化所（NLPR/MAIS）+ 字节跳动 Seed + 清华大学 + 西安交通大学 + 武汉大学 + 南京大学
>
> **发布时间**：2026 年 4 月（arXiv 2604.03181）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.03181) | [项目主页](https://anonymous1219-create.github.io/Anonymous-Web/)
>
> **发表状态**：NeurIPS 2026 在投（截至 2026.05 未公开录用结果）
>
> **分类标签**：`Video Action Model` `3D World-Action Model` `多视图热力图` `Wan2.2 视频基座` `Diffusion Policy` `Meta-World` `RoboCasa` `LoRA 微调`

---

## 一句话总结

把 BridgeVLA 的"点云 → 三视图正交投影 + 多视图热力图"管线移植到 **Wan2.2-5B 视频基座**之上，让 VAM 在同一套视频扩散表征里联合预测未来 RGB 视频和未来端执行器热力图视频，并把热力图反投影成连续 3D 轨迹；首次把 3D 结构先验注入视频基础模型，仅用 10 条演示在 Meta-World、RoboCasa、真实 Franka 上分别比最强基线高 22%、10%、16%，单步去噪即可达到最佳水平。

---

## 一、问题与动机

### 1.1 VLA 和 VAM 各缺一条腿

机器人操作需要同时理解 3D 空间结构和环境的时间演化，但两大主流范式都缺一条腿：

| 范式 | 时序建模 | 3D 结构建模 | 典型方法 |
| --- | --- | --- | --- |
| VLA（图像-文本预训练 VLM 微调） | ✗（图文对静态预训练） | ✗（2D 视觉） | π₀.₅、SpatialVLA、BridgeVLA |
| VAM（视频基座微调） | ✓ | ✗（2D 像素空间） | DreamZero、Cosmos Policy、UVA |
| 3D 策略 | ✗（点云无时间） | ✓（点云/体素） | DP3、3D Diffuser Actor |
| **本文 SpatialVAM** | **✓ 视频基座** | **✓ 多视图热力图** | — |

VLA 缺少"未来想象"能力（认知科学指出：动作选择本质是预测环境如何响应，而非反应式地处理瞬时观测）；VAM 在 2D 像素空间预测，造成"观测是 2D 视频、动作在 3D 物理空间"的**观测-动作错位**，要靠大量数据弥补这条鸿沟。

### 1.2 核心命题

> 能否设计一个**3D World-Action Model**，既继承视频基座的动力学先验，又把 3D 结构隐式编码进去？

SpatialVAM 给出的答案是 —— **把 3D 表示成视频本身**：点云投三视图 RGB、末端位姿投三视图热力图，二者格式完全一致，都可被 Wan2.2 的 VAE 编码，由同一个 DiT 联合去噪。这样既不破坏视频基座的预训练分布，又让 3D 结构成为"看得见的视频"。

---

## 二、预备知识

### 2.1 BridgeVLA 的"输入输出 2D 对齐"

BridgeVLA（同组前作）已经证明：把点云正交投影成三视图、把末端位置预测变成多视图 2D 热力图分类，可以用 VLM 的预训练知识做高样本效率 3D 操作（3 条轨迹即 95.4% on RLBench）。SpatialVAM 直接复用这套投影管线（Sec. 3.1 & 附录 B），但把骨架从 PaliGemma 换成 Wan2.2，把"单帧关键位姿"扩展成"连续视频"。详见 [[BridgeVLA_2025]]。

### 2.2 Wan2.2 视频扩散基座

Wan2.2 是一个 5B 参数的视频生成 DiT，包含 3D VAE（空间 8×、时间 4× 压缩，C=16）+ DiT + 3D VAE 解码器，原本在单视图视频-文本对上预训练。SpatialVAM 通过**在每个 DiT block 中插入 view-attention**把它扩展为多视图视频扩散器。

### 2.3 Heatmap-based 平移预测

末端执行器的 3D 位置不直接回归，而是表示为三视图上以投影像素为中心的截断高斯热力图：

$$
H_i^t(\mathbf{x}) = \begin{cases} p_i^t(\mathbf{x}), & p_i^t(\mathbf{x}) \ge \tau \\ 0, & \text{otherwise} \end{cases}, \quad p_i^t(\mathbf{x}) = \exp\!\left(-\frac{\|\mathbf{x}-\hat{\mathbf{x}}_i^t\|^2}{2\sigma^2}\right)
$$

其中 $i\in\{1,2,3\}$ 是视图索引，$\hat{\mathbf{x}}_i^t$ 是时刻 $t$ 末端投到视图 $i$ 上的像素坐标，$\sigma$ 控制扩散程度，$\tau$ 为概率阈值。预测时取三视图峰值反投影即得 3D 坐标。

---

## 三、核心方法

### 3.1 整体流水线

> 1. **Projection（Sec. 3.1）**：1 m³ 工作空间裁剪后点云 → 三视图正交投影 RGB；当前末端位姿 → 三视图截断高斯热力图。
> 2. **Multi-View Video Diffusion（Sec. 3.2）**：RGB 与热力图共享同一个 VAE 编码 → 沿视图维拼接 → Wan2.2-DiT（加 view-attention）联合去噪未来 RGB 视频和未来热力图视频。
> 3. **Action Decoding（Sec. 3.3）**：预测的热力图视频 → 三视图峰值反投影 → 连续 3D 位置轨迹；预测的视频潜表征 → 轻量 Rotation & Gripper Predictor → 旋转/夹爪。

两个可训练模块共计 5.17B 参数：5B 视频扩散 Transformer（LoRA 微调）+ 170M Rotation & Gripper Predictor。

### 3.2 多视图视频扩散 Transformer

每个 DiT block 增加一个 **view-attention** 模块，让 token 在视图间显式交互：

> 1. **VAE 编码**：每个视图的 RGB 序列与热力图序列分别送入预训练 VAE，得到形状 $(B,V,T_l,H_l,W_l,C)$ 的潜变量（$T_l=1+T/4$，$H_l=H/8$，$W_l=W/8$，$C=16$）。
> 2. **沿视图维拼接 + patch + flatten**：得到 token 序列 $(B,V,T\cdot H\cdot W,C)$（这里 $T,H,W$ 已是 patchify 后的潜空间维度）。
> 3. **三段注意力**（每个 DiT block 内串联）：
>     1. **Self-Attention** — reshape 为 $(B,V,THW,C)$，在时空维度上做自注意力（继承 Wan2.2 预训练权重）。
>     2. **View-Attention** — reshape 为 $(B,T,V\cdot HW,C)$，让同一时刻不同视图的 token 互相看见。
>     3. **Cross-Attention** — 与 T5 文本 token 交互。
> 4. **VAE 解码**：去噪后的潜变量分两支解码，分别得到未来多视图 RGB 视频和未来多视图热力图视频。

为什么用 view-attention 而不是 channel concat（Tab. 3 Model #3）？后者会产生通道适配 bottleneck，掉点 8%（89.1→81.1）。视图维拼接保留每个视图的完整 token 流。

### 3.3 动作解码：热力图反投影 + 旋转/夹爪头

**位置**：对每个时刻 $t$ 取三视图热力图峰值，反投影到工作空间离散网格上，取三视图概率均值最大点作为 3D 位置（细节同 BridgeVLA / RVT-2，附录 B）。

**旋转与夹爪**（Fig. 2c）：用一个轻量 170M 模块处理去噪后潜变量：

> 1. **Time upsample**：因 VAE 在时间维 4× 压缩，先把潜变量沿时间维上采样。
> 2. **双路特征**：Global Feature Extractor（卷积，整张潜变量）+ Local Feature Extractor（卷积，热力图峰值附近局部潜变量），fusion 后沿视图维聚合。
> 3. **Cross-attention**：以第一帧潜变量作为 conditioning，预测潜变量经四层 Transformer encoder 后分流到两个 MLP head：
>     1. **Rotation head**：欧拉角离散为 72 bin × 3 维（roll/pitch/yaw），交叉熵分类。
>     2. **Gripper head**：二值开/关。
> 4. 训练目标：

$$
\mathcal{L}_{pred} = \mathcal{L}_{rol} + \mathcal{L}_{pit} + \mathcal{L}_{yaw} + \mathcal{L}_{gri}
$$

预测的是相对于 conditioning 帧的**变化量**，比绝对值更稳定。

### 3.4 训练目标：双扩散损失

视频与热力图潜变量分别加噪，DiT 同时预测两路噪声（MSE）：

$$
\mathcal{L}_{diff} = \lambda \mathcal{L}_{vid} + (1-\lambda)\mathcal{L}_{heat}
$$

默认 $\lambda=0.5$。训练前对点云和末端位姿做 SE(3) 增强，再过投影管线，保证视点与轨迹的等变性。

DiT 用 **LoRA**（rank=32，作用于 q/k/v/o/ffn.0/ffn.2）微调，全量微调（Model #2）仅多 -1.7% 但显存与算力代价显著更高，故采用 LoRA。Rotation & Gripper Predictor 用 ground-truth 潜变量 + 少量噪声训练以增强对噪声的鲁棒性。

### 3.5 推理

当前帧点云与末端 → 三视图 RGB + 热力图 → VAE 编码得到 conditioning latent → DiT 多步去噪（默认 5 步即可，A100 单卡 5 Hz）→ 两支解码：热力图 → 3D 位置；潜变量 → 旋转/夹爪 → 拼成 24 帧 action chunk 下发给控制器，单 chunk 端到端 4.6 s。

---

## 四、实验结果

### 4.1 Meta-World（5 demo/task，25 trials）

每任务仅 5 条演示，7 个任务共 35 条 → 极端低数据：

| 方法 | D-Open | D-Close | Btn | Btn-Top | Fct-Cls | Fct-Open | Handle | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniPi | 0 | 9 | 3 | 0 | 1 | 3 | 4 | 11.4 |
| BC-Scratch | 6 | 9 | 9 | 3 | 5 | 5 | 9 | 26.2 |
| BC-R3M | 1 | 15 | 9 | 1 | 6 | 17 | 13 | 35.4 |
| DP | 12 | 12 | 10 | 5 | 6 | 15 | 6 | 37.7 |
| AVDC | 18 | 23 | 15 | 6 | 14 | 6 | 21 | 58.9 |
| DreamZero (Wan2.2-5B) | 0 | 11 | 23 | 3 | 20 | 25 | 25 | 61.1 |
| Track2Act | 22 | 19 | 14 | 10 | 12 | 22 | 19 | 67.4 |
| **SpatialVAM** | **25** | **25** | **25** | **24** | 8 | 24 | **25** | **89.1** |

三条结论：(i) 视频预测是低数据制胜关键；(ii) 同样用 Wan2.2 的 DreamZero 仅 61.1%，说明**3D 多视图预测**才是真正涨点因素；(iii) 唯一短板 Fct-Cls 仅 8/25，作者归因为夹具的视觉模糊。

### 4.2 RoboCasa（10 demo/task，50 rollouts，5 个厨房任务）

| 方法 | Avg. Succ. (%) |
| --- | --- |
| 3D Diffuser Actor | 8 |
| Cosmos Policy | 32.4 |
| **SpatialVAM** | **42** |

Cosmos Policy 是把动作/价值也复读成视频做扩散的 VAM，3D Diffuser Actor 是利用 3D 输入但无视频基座；SpatialVAM 比二者分别高 10/34，进一步坐实"3D + 视频基座"的组合价值。

### 4.3 真实世界 Franka Research 3（10 demo/task，10 trials，3 + 4 任务）

3 个基础任务（Put Lion / Push-T / Scoop Tortilla）+ 4 个泛化任务（背景 Put-B、高度 Put-H、关灯 Push-L、新类别 Scoop-C）：

| 方法 | Put Lion | Push-T | Scoop Tort. | Put-B | Put-H | Push-L | Scoop-C | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DP3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| π₀.₅ | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1.4 |
| UVA | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 5.7 |
| BridgeVLA | 9 | 0 | 4 | **8** | **7** | 0 | 1 | 41.4 |
| **SpatialVAM** | **10** | **4** | **7** | 5 | 6 | **3** | **5** | **57.1** |

关键洞察：

- **DP3** 在 10 演示下严重过拟合（loss 从 1e-1 跌到 1e-8 但策略只会冲向货架，不会接近物体）；
- **π₀.₅** 大规模 VLA 预训练在分布外完全失效；
- **UVA** 行为趋势对、精度不够，常常提前闭爪或过推；
- **BridgeVLA** 在 Push-T 和 Scoop Tortilla 上崩溃（只能预测单个 key pose，无法处理连续运动/接触富集任务）；
- **SpatialVAM** 在 Put-B / Put-H 上略低于 BridgeVLA，作者归因为 BridgeVLA 的 PaliGemma 在外观变化上更强；推论：放大视频预训练后这一差距可以补上。

### 4.4 鲁棒性 & 消融

**去噪步数 $N$**（Fig. 5 左）：1/2/3/4/5/10/25/40/50 步对应 85.7/87.4/88.0/88.6/89.1/88.0/88.6/91.4/87.4。**单步就有 85.7%**，远低于普通视频扩散的 50 步需求。作者解释：热力图分布简单，且动作只依赖峰值，对整体质量不敏感。默认推荐 5 步以平衡时延（5 Hz @ A100）。

**RGB 副预测视图数**（Fig. 5 右）：0/1/2/3 个额外 RGB 视图对应 61.1/72.0/81.7/89.1，**单调上升**。说明 RGB 协同预测对 spatio-temporal 表征学习关键，且预测视图越多越好。

**关键超参数**（Tab. 5）：$\lambda \in \{0.1, 0.5, 0.9\}$ 对应 92/89.1/89.1（±3.3%）；$\sigma \in \{1.5, 2.5, 3.5\}$ 对应 89.1/89.7/86.9（±2.5%）。**对超参不敏感**。

**架构消融**（Tab. 3）：

| # | View Concat | Init W | LoRA | Trans. Decode | Avg (%) |
| --- | --- | --- | --- | --- | --- |
| 1（默认） | ✓ | ✓ | ✓ | Heatmap | **89.1** |
| 2 | ✓ | ✓ | full FT | Heatmap | 87.4 |
| 3 | channel concat | ✓ | ✓ | Heatmap | 81.1 |
| 4 | ✓ | ✓ | ✓ | TF-Regress | 76.6 |
| 5 | ✓ | scratch | ✓ | Heatmap | **4.6** |

最戏剧的是 #5：**去掉 Wan2.2 预训练权重几乎完全失败**（甚至无法拟合训练集）—— 视频基础模型的预训练知识是 SpatialVAM 数据高效的根本。

### 4.5 可解释性 & 安全部署

视频预测让动作"可视化预验"。4 人评估者各 35 轮共 140 次部署，看着预测视频判定为不安全（潜在碰撞）则重启 —— 碰撞次数从 **6/140 降到 0/140**（Fig. 6）。这是 VAM 相对 VLA 的一个不可替代的"可解释性副产品"。

---

## 五、局限性

1. **推理慢**：5B DiT + 24 帧 action chunk 端到端 4.6 s（>30 GB 显存），不适合高频灵巧任务；作者准备整合 TurboDiffusion（100–200×）和 Real-Time Chunking 实现实时部署。
2. **固定三视图**：当前用固定正交投影，自适应视图选择（VERM 路线）留作未来工作。
3. **热力图分辨率**：256×256 → 每像素约 4 mm，Scoop Tortilla 偶尔会"刮到饼上面"而不是"伸到下面"，作者建议提升分辨率。
4. **外观泛化**：Put-B（背景纹理变化）/Put-H（高度变化）上略不及 BridgeVLA，原因是 Wan2.2 视频预训练对图文外观变化的覆盖不如 PaliGemma —— 需要更大规模的视频 + 文本协同预训练才能闭合。

---

## 六、个人思考

### 6.1 与 BridgeVLA 的"代际更迭"

SpatialVAM 是 BridgeVLA 同组团队（Peiyan Li、Yixiang Chen 等）的**直接升级路径**：

| 维度 | BridgeVLA（前作） | SpatialVAM（本文） |
| --- | --- | --- |
| 骨架 | PaliGemma（图文 VLM） | Wan2.2-5B（视频扩散 DiT） |
| 输出粒度 | 单个 key pose 热力图 | 24 帧连续热力图视频 + RGB 视频 |
| 动作空间 | 离散 keyframe + motion planner | 连续 action chunk |
| 处理连续运动 | ✗（Push-T 全 0/10） | ✓（Push-T 4/10） |
| 处理接触富集 | 弱（Scoop 4/10） | 强（Scoop 7/10） |
| 可解释性 | 单帧热力图 | 完整视频可视化 |

核心范式没变 —— **3D 信息通过多视图投影 + 热力图编码**，依然是 NLPR-Bytedance 团队的"输入输出 2D 对齐"哲学；变化的是**骨架的时间建模能力**：从图文 VLM 到视频 DiT，让"key pose"变成"连续视频"。这也正面回答了 BridgeVLA 留下的最大局限（连续动作和接触富集任务）。

### 6.2 与 Fast-WAM 的呼应与对比

两篇 2026 年同期工作都基于 Wan2.2-5B，但回答不同问题（详见 [[FastWAM_2026]]）：

| 维度 | Fast-WAM | SpatialVAM |
| --- | --- | --- |
| 研究问题 | 测试时未来想象是否必要？ | 如何把 3D 注入 VAM？ |
| 测试时是否生成视频 | **不生成**（单次前向） | **生成**（5 步去噪） |
| 3D 输入 | ✗（单视图） | ✓（点云三视图） |
| 测试时 latency | 190 ms | 4600 ms |
| 主结论 | 训练时视频协同 >> 测试时未来想象 | 3D 多视图 + 视频基座联合训练 |

Fast-WAM 的"训练时视频目标 → 隐式动力学先验"在 SpatialVAM 上隐含成立（Tab. 3 #5：去 Wan2.2 预训练失败、Fig. 5 右：加 RGB 视图协同单调涨点都是侧证），但 SpatialVAM 反过来证明 —— **当预测目标本身就是动作（热力图）时，测试时显式生成"动作视频"是必要的**。两篇放在一起读，能更精细地拆分"视频协同训练 vs 测试时视频生成"的边界条件。

### 6.3 单步去噪的反直觉

Fig. 5 左最让我意外：**1 步去噪即 85.7%**。视频生成领域 CogVideoX/Wan 都要 ~50 步，而 SpatialVAM 因为：

1. **热力图分布简单**（单峰高斯，没有高频细节）；
2. **下游只用峰值位置**，整体视觉质量无关；
3. **训练监督密集**（每个像素都有 MSE 目标）。

这暗示了 VAM-as-policy 与 VAM-as-generator 的本质不同：作为策略时，潜表征只要"语义对"就够，不需要"看起来好看"。这也是为什么作者愿意为可解释性的 RGB 视频付出额外 5 步去噪的 token —— **真正的负载是热力图**，RGB 只是副产品。

### 6.4 与同组 VLA 工作的连贯叙事

NLPR-Bytedance Seed 团队在 2024–2026 的工作线条非常清晰：

> EC-Flow (2025, 具身光流, 无动作标注) → BridgeVLA (2025, 2D 多视图热力图 + VLM) → AnchorVLA4D (2026, 首帧锚帧 + 4D 编码器) → SpatialVAM (2026, 3D 多视图热力图 + 视频基座)

可以归纳出一个组内的设计哲学：**用低维 2D 表征做"3D 信息载体"，但越来越重视时间维度**。SpatialVAM 是这条线上最完整的形态 —— 同时占据"3D 输入"和"时间预测"两个轴的最优点。

### 6.5 可借鉴的方法论

1. **预训练表征对齐胜过模块复杂度**：把热力图先 colorize 成 3 通道再过 VAE 编码，看起来 "丑"，但保留了 VAE 的全部预训练知识。这种"宁愿改自己的数据格式，也不改预训练模型"的工程哲学在 BridgeVLA、SpatialVAM 一脉相承。
2. **LoRA 在视频扩散上"够用"**：Tab. 3 #1 vs #2 仅差 1.7%。考虑到 5B DiT 全参微调的显存代价（H200 也不轻），LoRA 是 VAM 微调的合理默认。
3. **多视图协同预测的边际收益单调**：0→3 视图 +28 点（Fig. 5 右）。这与多视图 3D 重建中"视图数越多越好"的直觉一致，但首次在 video diffusion policy 上量化。对 4 视图、6 视图是否继续涨值得后续验证。

---

## 参考

- **BridgeVLA**（Li et al., 2025, NeurIPS 2025）：同组前作，2D 热力图 + VLM 范式的直接前身 —— 详见 [[BridgeVLA_2025]]
- **Fast-WAM**（Yuan et al., 2026, arXiv 2603.16666）：同样基于 Wan2.2-5B 的 VAM 工作，回答互补的问题 —— 详见 [[FastWAM_2026]]
- **Wan2.2**（Team Wan, 2025, arXiv 2503.20314）：5B 视频扩散基座，本文骨架
- **DreamZero**（Ye et al., 2026, arXiv 2602.15922）：同基座 (Wan) 的零样本 VAM，Meta-World 主对比
- **Cosmos Policy**（Kim et al., 2026, arXiv 2601.16163）：英伟达把动作复读成视频的 VAM，RoboCasa 主对比
- **UVA**（Li et al., 2025, arXiv 2503.00200）：单视图 VAM，真实世界对比
- **DP3**（Ze et al., 2024, arXiv 2403.03954）：3D 视觉运动策略代表，真实世界对比
- **π₀.₅**（Physical Intelligence, 2025, arXiv 2504.16054）：通用 VLA 基线，真实世界对比
- **RVT-2 / 多视图投影**（Goyal et al., 2024）：正交投影 + 热力图反投影管线源头
- **Track2Act / AVDC**（Bharadhwaj et al., 2024 / Ko et al., 2023）：基于光流/点轨迹的 VAM 基线
- **SyncamMaster**（Bai et al., 2024, arXiv 2412.07760）：多视图视频生成的 view-attention 设计源头
