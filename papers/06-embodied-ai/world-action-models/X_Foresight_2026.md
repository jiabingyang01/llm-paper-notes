# X-Foresight：面向预测式世界建模的视觉-动作联合因果预测网络

> **论文**：*X-Foresight: A Joint Vision-Action Causal Forecasting Network via Predictive World Modeling*
>
> **作者**：Baolu Li, Jingyu Qian, Rui Guo, Yilun Chen et al.（核心贡献者按字母序排列；Project Lead: Zhuangzhuang Ding, Pengkun Zheng；Advisors: Yu Zhang, Xianming Liu）
>
> **机构**：PWM Team, XPeng Inc.（小鹏汽车）
>
> **发布时间**：2026 年 05 月（arXiv 2605.24892，页脚显示 v4 更新于 2026 年 07 月）
>
> **发表状态**：未录用（预印本，标注为 XPeng 内部 Technical Report）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.24892) | [PDF](https://arxiv.org/pdf/2605.24892)
>
> **分类标签**：`世界模型` `自动驾驶VLA` `视觉-动作联合预测` `分块自回归` `扩散渲染器`

---

## 一句话总结

X-Foresight 把预测式世界模型直接嵌入自动驾驶 VLA 架构,用"分块自回归"（chunk-wise autoregressive）取代逐帧预测,在块内保留稠密帧捕捉瞬时动力学、在块间做稀疏长跨度预测捕捉长时因果,配合课程学习和安全关键片段的重要性采样,在千卡生产级设置下把碰撞率从 0.228% 降到 0.191%（相对降 16.2%）、综合驾驶质量指标 Total CCES 从 3.8296 降到 3.6535（相对降 4.6%）。

## 一、问题与动机

论文的出发点是"物理世界知识主要蕴藏在视频里",要让 VLA 模型具备安全、可泛化的规划能力,需要通过预测式世界建模从视频数据中提取这类知识,即通过预测未来视频内化物理动力学和长时因果性。但朴素的逐帧预测（next-frame prediction）面临两个结构性困难：

1. **视频 token 低熵、冗余**：与语义上离散、稀疏、高熵的文本 token 不同,相邻视频帧高度相似,直接做下一帧预测极易退化为"平凡外推"（trivial extrapolation）,学不到有意义的动力学。
2. **时间维度的两难**：瞬时动力学（instantaneous dynamics）依赖稠密的短程视觉证据,而世界演化的长时因果性（world transitions）需要跨越可变长的长时间窗口,稠密逐帧预测在计算上无法高效覆盖这种长跨度。

现有 VLA（如 RT-2、PaLM-E、OpenVLA,以及工业界的 XPeng VLA 2.0）本质上是"反应式"的,缺乏在行动前模拟未来状态的前瞻能力,难以提前规避危险、应对复杂场景,也难以利用长时环境因果性改善控制。X-Foresight 试图把预测式世界模型直接整合进 VLA 架构,联合学习世界建模与实时动作控制。

## 二、核心方法

X-Foresight 由两大组件构成：负责联合世界预测与动作规划的 **Large Drive Model（LDM）**,以及负责把 LDM 预测的相机 token 解码为高保真多视角图像的 **Vision Renderer**。

**1. 分块自回归（Chunk-Wise Prediction）**。LDM 的多模态提示格式为

$$[\text{SYSTEM PROMPT}] \mid [l_0, O_0, A_0, Q_0] \mid [l_1, O_1, A_1, Q_1] \mid \dots \mid [l_i, O_i, A_i, Q_i]$$

每个时间块（约 1 秒）包含文本 token（指定预测窗口）、观测 token $O_i$（ViT 编码的多相机视频 token）、动作/状态 token $A_i$、以及触发未来预测的查询 token $Q_i$（预测结果直接原位写入,不额外占用视觉查询 token 预算）。大白话说：与其一帧一帧地"看图猜下一帧"（信号太弱、太容易偷懒外推）,不如把时间切成一秒一秒的块,块内保留稠密帧学瞬时动力学,块与块之间跳着预测学长时演化,一次自回归步预测一整段未来而不是一帧。

**2. 课程学习扩展前瞻（CLEF, Curriculum Learning with Extended Foresight）**。训练从相邻块间隔 1 秒的短时序开始,再逐步把块间时间步幅（stride）从 1 秒扩大到 3 秒,从而在不增加计算预算的前提下扩展预测视野。在拉长的步幅设置下,观测预测目标跨越更大时间间隔,但动作预测始终锚定在紧邻的下一控制步（非对称设计）,因为闭环控制需要时间上稠密、无跳跃的轨迹输出。

**3. 时间重要性采样（TIS, Temporal Importance Sampling）**。均匀采样会把大部分训练预算浪费在近似匀速巡航的平淡片段上,而欠采样安全关键的稀有事件。TIS 基于自车轨迹的纵向/横向加速度 $a_x, a_y$,在近未来、中时程、近历史三个时间窗口 $W_1^k, W_2^k, W_3^k$ 内取加权加速度幅值的最大值再求和,得到候选步 $k$ 的重要性分数：

$$w_k = \sum_{W \in \{W_1^k, W_2^k, W_3^k\}} \max_{t \in W}\left(\lambda_x |a_x(t)| + \lambda_y |a_y(t)|\right)$$

再用温度缩放的分布 $p_k = w_k^{1/\tau} / \sum_j w_j^{1/\tau}$ 采样候选步。大白话说：刹车、急转、突然变道这些"危险时刻"前后会被优先采样,让模型把更多注意力放在真正决定安全的关键片段上,而不是无脑巡航的大多数时间。

**4. 半因果块稀疏注意力（Semi-Causal Block Sparse Attention）**。为降低长序列训练开销,系统提示作为全局 sink token 对所有块可见；块内 token 全双向自注意力；跨块则采用时间因果设计——每个 token 只与更早块中位置对应的 token 及随时间距离收缩的空间邻域交互；查询 token $Q_i$ 可完全访问此前所有 prompt 侧 token,但被屏蔽对更早查询 token $Q_{1:i-1}$ 的注意力（防止预测结果直接依赖更早的预测结果而非真实观测）。再按 query-key 时间步差的奇偶性把注意力头分成两组互补的稀疏模式,使被访问的 block 数量随序列长度近似线性增长而非二次增长。自研的 Block Sparse Attention 相比 FlashAttention-2 把单步训练耗时从 24.50s 降到 15.40s,提速 1.59×。

**5. 训练目标**。相机 token 预测用 L2 损失 $L_{\text{cam}}=\frac{1}{HV}\sum_i\sum_v\|\hat{o}_i^v - g(I_i^v)\|_2$（$g(\cdot)$ 为冻结的 ViT 编码器）,动作预测用 L1 损失 $L_{\text{act}}=\frac{1}{H}\sum_i\|\hat{a}_i-a_i\|_1$,另加 BEV 辅助损失 $L_{\text{bev}}$,总损失 $L_{\text{total}} = L_{\text{act}} + \alpha L_{\text{cam}} + \beta L_{\text{bev}}$。

**6. Vision Renderer**。基于 X-World 的 DiT 视频生成骨干（3D causal VAE 取自 WAN2.2）,用 rectified-flow 目标训练：$y_t=(1-t)y_0+ty_1$,$y_0\sim p_{\text{data}}(y\mid c)$,$y_1\sim\mathcal{N}(0,I)$,最小化 $\mathcal{L}_{\text{velocity}}(\theta)=\mathbb{E}\left[\|v_\theta(y_t,t,c)-(y_1-y_0)\|_2^2\right]$。关键设计决策是渲染器**只**接受 LDM 的相机 token 作为条件,不接受动作 token——因为相机 token 已隐式编码了自车位姿与场景动态,若同时暴露动作 token,渲染器可能走"低熵捷径"（直接根据动作生成画面而忽视相机 token）,使 LDM 的想象与渲染器的生成彼此脱节。为缓解闭环回滚中的漂移,还引入 latent sink（跨自回归步锚定稳定参考上下文）和 latent augmentation（训练时对当前步 latent 施加扰动,模拟推理时的分布）。

**7. 三阶段训练管线**：Stage I 单独训练 LDM（教师强制,联合预测动作/相机 token/BEV,配合分块稀疏注意力+课程学习+重要性采样）；Stage II 在真值动作条件下单独训练渲染器（从 X-World 权重初始化,把频率从 WAN 原生 12Hz 适配到 LDM 的 4Hz）；Stage III 冻结 LDM,渲染器的条件源由真值动作切换为 LDM 预测的相机 token,只用 rectified-flow 目标微调渲染器,弥合训练期教师强制与推理期自回归回滚之间的差距。推理时,LDM 与渲染器以 4 Hz 严格交替：LDM 单次前向输出下一动作和 1 秒的多视角相机潜在 token,渲染器据此渲染出下一步观测,回填进滚动上下文,如此迭代形成任意长的闭环轨迹。

## 三、关键结果

所有实验基于约 28 万小时、3400 万段片段、7 相机环视的自建数据集（tokenize 后 13.8T token）,城市道路占比 86.8%、高速 13.2%。评价指标含 ADE/FDE（米）、CCES 驾驶质量套件（Compliance/Comfort/Efficiency/Safety,均以 H=1 参照行归一化为比率）及碰撞率。

**训练时长跨度 H 的效应**（128 GPU,单步前向评估,表 1）：

| H（1s块数） | ADE Lat/Long | FDE Lat/Long | 碰撞率 | Safety | Total CCES |
|---|---|---|---|---|---|
| 1（单块） | 0.1923 / 1.2409 | 0.4881 / 3.1935 | 0.263% | 1.0000 | 4.0000 |
| 6 | 0.1864 / 1.2196 | 0.4691 / 3.1178 | 0.262% | 0.9927 | 3.9396 |
| 21 | 0.1810 / 1.2110 | 0.4571 / 3.0988 | 0.245% | 0.9481 | 3.9524 |

ADE/FDE 与碰撞率随 H 增大单调改善（降 2–7%）,Safety/Compliance 也单调提升,但 Comfort/Efficiency 在 H=21 时出现轻微回退,使 Total 从 H=6 的 3.9396 略反弹至 3.9524,这一现象直接引出了下一组消融。

**CL / CLEF / TIS 消融**（均从 H=6 checkpoint 继续训练,表 2）：

| 配置 | 碰撞率 | Safety | Total CCES |
|---|---|---|---|
| 继续 H=6 | 0.270% | 0.9726 | 3.9523 |
| +H=21, 基础课程学习(CL) | 0.238% | 0.9310 | 3.8745 |
| +H=21, CLEF | 0.230% | 0.9387 | 3.8734 |
| +H=21, CLEF+TIS | 0.216% | 0.9264 | **3.8447** |

TIS 在 CLEF 基础上把碰撞率进一步降低 6.1%（相对）,取得四者中最低的 Total CCES 和最优 Safety 比率。

**生产级对比**（1024 GPU,X-Foresight = H=21+CLEF+TIS vs. 反应式 VLA baseline,表 3）：

| 指标 | Baseline | X-Foresight | 相对变化 |
|---|---|---|---|
| ADE Lat/Long | 0.1675 / 1.1387 | 0.1567 / 1.0982 | −6.4% / −3.6% |
| FDE Lat/Long | 0.4153 / 2.9117 | 0.3789 / 2.7924 | −8.8% / −4.1% |
| 碰撞率 | 0.228% | 0.191% | −16.2% |
| Safety / Compliance | 0.9441 / 0.9483 | 0.8583 / 0.8708 | −9.1% / −8.2% |
| Total CCES | 3.8296 | 3.6535 | −4.6% |

Comfort、Efficiency 也分别改善 1.0%、0.4%。定性上（图 7）,X-Foresight 在"多出口环岛按导航指令选择较远出口""夜间红灯预判即将转绿而不提前刹停"两类需要跨空间/跨时间前瞻的场景中,轨迹与真值高度吻合,而 baseline 均出现明显偏离。

**渲染质量**（表 5,FID/FVD,7 相机 4Hz 平均）：Vision Renderer 在 1 秒/6 秒预测视野下分别达到 FID 1.51/2.84、FVD 11.28/29.52,显著优于仅反映相机 token 自身重建质量的 Camera Latent Decoder（FID 10.97/11.82、FVD 135.56/158.39）；6 秒相比 1 秒仅退化 1.33 FID / 18.24 FVD,表明自回归回滚中的漂移可控。

## 四、评价与展望

**优点**：X-Foresight 把"分块自回归"作为核心杠杆,同时缓解了视频 token 低熵导致的平凡外推和瞬时动力学-长时因果的两难,思路清晰且有充分的消融支撑（表 1、2 逐层拆解 H、CL、CLEF、TIS 各自贡献,而非只报总分）。渲染器与规划/预测头解耦（相机 token 而非动作 token 作为渲染条件)的设计,以及用 Camera Latent Decoder 做"一致性审计"的验证方式,是一个值得借鉴的工程实践,能够独立检验预测的潜在表示是否真的承载了场景语义,而非渲染器凭生成先验"脑补"。千卡工业级数据规模（28 万小时、13.8T token）也让实验结果具有一定的说服力。

**局限与开放问题**：
1. 所有评测均在自建的 in-house 数据集上进行,没有公开基准（如 nuScenes、Waymo）上的数字,论文本身也未与 Sora、Genie、X-World 等其他公开世界模型做直接的量化对比,可比性和可复现性有限。
2. 表 1–3 的主表评测均限定为"单步前向"（single forward step at inference）,以隔离单步预测质量,闭环长时 rollout 的量化评估仅停留在图 7 的两个定性案例,缺乏系统的闭环误差累积分析。
3. 论文聚焦于自动驾驶端到端 VLA,而非机械臂/双足等操作类具身智能场景;其分块自回归、课程学习拉长视野、安全关键片段重要性采样等思路对操作类 VLA 的世界模型预训练具有较强的可迁移性,但迁移时"安全关键"信号的定义（driving 用加速度,manipulation 需替换为接触力/抓取失败等信号）需要重新设计。
4. Baseline 具体身份论文未直接点名,仅在引言提及 XPeng VLA 2.0 作为工业界反应式 VLA 代表,读者需要自行推断表 3 中 baseline 的确切构成,透明度略有欠缺。
5. 作者在结论中自陈的三个未来方向——闭环 rollout 中引入额外监督以提升长尾表现、融合 3D 几何监督增强物理世界知识、以及从粗到细的世界重建训练策略——均属于开放问题,尚无实验验证。

**与相关工作的关系**：Vision Renderer 直接构建于同团队的 X-World（arXiv 2603.19979）DiT 骨干和 WAN2.2 的 3D causal VAE 之上,可视为把 X-World 这一"生成式仿真器"改造为"闭环预测的像素前端"；漂移抑制的 latent sink/augmentation 思路借鉴自 Helios（实时长视频生成）,而非采用 CausVid/Self-Forcing 的蒸馏路线,体现了工程侧对训练成本与效果的权衡取舍。

## 参考

- X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving（arXiv 2603.19979）——Vision Renderer 的骨干来源
- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control（CoRL 2023）
- OpenVLA: An Open-Source Vision-Language-Action Model（arXiv 2406.09246）
- Helios: Real Real-Time Long Video Generation Model（arXiv 2603.04379）——漂移抑制设计的参考来源
- Wan: Open and Advanced Large-Scale Video Generative Models（arXiv 2503.20314）——3D causal VAE 来源
