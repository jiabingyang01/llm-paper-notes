# Green-VLA：面向通才机器人的分阶段视觉-语言-动作模型

> **论文**：*Green-VLA: Staged Vision-Language-Action Model for Generalist Robots*
>
> **作者**：I. Apanasevich、M. Artemyev、R. Babakyan et al.（Manipulation Team, Sber Robotics Center；项目负责人 A. Postnikov）
>
> **机构**：Sber Robotics Center
>
> **发布时间**：2026 年 02 月（arXiv 2602.00919）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.00919) | [PDF](https://arxiv.org/pdf/2602.00919)
>
> **分类标签**：`VLA` `分阶段训练课程` `统一动作空间` `多具身泛化` `RL对齐` `人形机器人部署`

---

## 一句话总结

Green-VLA 用 L0→L1→R0→R1→R2 五阶段课程（基座 VLM→网络多模态预训练→多具身机器人预训练→具身特定微调→RL 对齐）配合 64 维**统一语义动作空间**、光流驱动的时序对齐与联合预测引导模块（JPM）,仅用约 3,000 小时机器人数据（远少于 π0 的 1 万余小时）就把同一份策略从单臂夹爪一路桥接到 Sber 人形机器人 Green 的 32 自由度全上肢控制,在 SimplerEnv WidowX 上做到 80.5% 成功率（对比 DB-MemVLA 73.2%、X-VLA 65.6%）,并让电商货架拣选任务的分布外（OOD）成功率靠 JPM 引导从 10.7% 提升到 72.8%。

## 一、问题与动机

论文指出单纯堆数据规模并不能解决 VLA 真实部署的三个瓶颈：(1) 机器人数据集在观测、动作空间、采样率上天然异构；(2) 数据质量参差不齐（抖动、模糊、场景单一）；(3) 主流训练范式行为克隆（BC，$\mathcal{L}_{BC}=\mathbb{E}_{(s,a)\sim\mathcal{D}}[\|\pi_\theta(s)-a\|^2]$）很快饱和,无法对齐长程任务级目标。同时,像 EO-1、WALL-OSS 这类显式推理型 VLA 虽然提升了长程规划,但自回归推理环路带来的高延迟难以满足实时控制。Green-VLA 的定位是：面向 Sber 自研人形机器人 Green（32 自由度头/躯干/双臂/灵巧手全上肢控制）的真实部署,同时保持跨具身（单臂机械臂、移动双臂、人形）零样本泛化能力。

## 二、核心方法

**1. 五阶段课程**：L0（预训练基座 VLM，无机器人动作）→ L1（2400 万条非机器人网络多模态样本，学习物理常识与空间指向）→ R0（1.84 亿条机器人域样本、跨具身通才预训练，>3,000 小时）→ R1（面向目标具身的高质量数据微调）→ R2（RL 对齐,弥合 BC 饱和后的长程/纠错缺口）。每一阶段对应一个明确瓶颈：L1 补语义、R0 学 affordance 先验、R1 高效适配具身、R2 注入奖励对齐。

**2. 数据与 DataQA 质控**：R0 混合 10 个开源数据集（AgiBotWorld 双指 774h、DROID 512h、Galaxea 477h、ActionNet 143h、AgiBotWorld 灵巧手 82h、Fractal 350h、RoboMind 33h、RDT 60h、Bridge 105h、BiPlay 9.7h）加 2 个自采数据集（Green Humanoid 原始 48h、ALOHA any_pick 11.2h）。Green Humanoid 通过左右镜像（双臂/手腕相机翻转互换、关节轨迹取反对应分量）与可逆技能的时间反转增广,扩充到 167 有效小时。`DataQA` 用抖动分数 $S_{tremble}=\dfrac{|\dot s_{smooth}-\dot s|}{|\dot s_{smooth}|+|\dot s|}$（速度轨迹与其高斯平滑版本之差）、拉普拉斯清晰度 $S_{sharp}=\text{median}(\text{MaxPool}(\text{stdblock}(\nabla^2 I)))$、DINOv3 视觉多样性 $D_{vis}=\mathbb{E}_d[\text{std}_t(\mathbb{E}_s[f_{t,s,d}])]$、状态多样性 $D_{state}=\sqrt{\text{tr}(\text{Cov}(s))}$ 四个指标联合打分过滤低质片段并确定采样权重。大白话：抖动分数衡量动作是不是"手抖",清晰度分数衡量画面是不是糊的,多样性分数衡量这批数据是不是"见过的场景太单一"。

**3. 统一动作空间**：naive 做法是把各具身动作零填充进一个公共高维向量 $\tilde a_t^e=P_e(a_t^e)$ 再统一回归,论文证明这会引入"伪惩罚项"——损失 $\mathcal{L}(\theta)=\mathbb{E}[\|m_e\odot(\pi_\theta-\tilde a^e)\|^2 + \|(1-m_e)\odot(\pi_\theta-\tilde a^e)\|^2]$ 的第二项纯粹是填充产物,会因为跨具身共享维度语义不一致而互相打架。Green-VLA 改用固定语义布局的统一空间 $\mathcal{A}_u\subset\mathbb{R}^{64}$（单臂末端位姿、双臂关节、人形关节各占固定槽位）,配合映射 $\Phi_e:\mathcal{A}_e\to\mathcal{A}_u$ 与二值掩码 $m_e\in\{0,1\}^{64}$,训练目标改为掩码 BC 损失：

$$\mathcal{L}_{uni}(\theta)=\mathbb{E}\Big[\big\|m_e\odot(\pi_\theta(x_t^e,c_e)-\Phi_e(a_t^e))\big\|_2^2\Big]$$

即对未使用槽位（$1-m_e$）完全不施加梯度。大白话：每个具身只对自己"该管"的动作维度算 loss,不再强迫模型在不相关的维度上也去拟合别的机器人的动作。推理时用逆映射 $\hat a_t^e=\Phi_e^{-1}(m_e\odot\hat u_t)$ 还原到具体控制器。对于目标人形机器人,论文还做**显式 retargeting**：不是简单 padding/复制,而是把源机器人（如单臂夹爪）动作对齐到人形对应肢体的最近可行关节配置,让异构机器人数据都变成"类人形经验"。

**4. 光流驱动的时序对齐 + 速度调制**：不同数据源采集帧率/执行速度差异巨大（如 Bridge、Fractal 低频高速 vs. 人形数据高频慢速）,论文用手腕相机的平均光流幅值作为运动速度代理,对轨迹做单调三次样条重采样,把各数据源对齐到统一运动尺度。在此基础上引入速度条件调制,让动作专家的隐状态按标量 $v$ 做 RMS 风格调制：$\hat h_t=\gamma(v)\,\text{RMSNorm}(h_t)+\beta(v)$,$v>1$ 时插值更密（精细局部控制）,$v<1$ 时更稀疏（长程高效执行）,推理时 $v$ 作为可调超参数在同一模型内切换"缩放级别"。

**5. 目标平衡采样课程**：为防止高动量优化器下多具身混合训练早期就"塌缩"到少数大数据集、稀有具身被冲刷掉,采样权重按温度调度 $W_i^{(t)}=w_i^{\alpha_t}/\sum_j w_j^{\alpha_t}$,$\alpha_0=0$（均匀混合）逐渐升到 $\alpha_T=1$（目标偏置分布）,让模型先学共享结构再逐步专精。

**6. OOD 检测与 JPM 引导**：用高斯混合模型 $p_{train}(s)=\sum_k\phi_k\mathcal{N}(s\mid\mu_k,\Sigma_k)$ 拟合训练状态分布,推理时若预测轨迹进入低密度状态（$p_{train}(s)<\tau_{ood}$）,沿密度梯度纠偏 $s+\alpha\nabla p_{train}(s)$（$\alpha=0.2$）拉回训练流形。针对电商货架这类"新品未见过"场景,联合预测模块（JPM）先用指向式 VLM 预测 2D 目标点,再用深度和相机位姿反投影到 3D：$[p^\star;1]^\top=T_c^w[d(u,v)K^{-1}[u,v,1]^\top;1]$,解 IK 得到关节目标 $q^\star$,最后用 $\Pi$GDM 伪逆引导扩散/流匹配技术把生成的速度场偏向目标点,属于训练时无需重训的推理期引导。

**7. RL 对齐（R2）**：两条互补、都不直接改动基础策略权重的"保守"路线。(a) 轨迹级优化：用 IQL 训练价值-Q 评论家（含渐近分位数回归的 expectile loss）,再对已生成轨迹按 $a\leftarrow a+\eta\,\nabla_aQ(s,a)/\|\nabla_aQ(s,a)\|$ 沿 Q 梯度纠偏 N 步,并要求在真实/仿真环境中**回放验证**（操作员把环境重置回原轨迹起点、执行优化后轨迹并保存结果）才能把改进后的轨迹并入 R1 数据集重新微调,避免把 Q 函数误差直接带入训练。(b) 源分布优化：训练一个独立的 actor $\pi_{\theta_{noise}}(\epsilon\mid s)$ 来生成喂给流匹配基座模型的初始噪声（替代各向同性高斯 $p_0$）,基座权重全程冻结,行为始终贴近训练分布,比常规在线 RL 更可控。

## 三、关键结果

**ALOHA 桌面清理（Cobot Magic, R0 阶段）**：

| 策略 | Tape | Screwdrivers | Pliers | First-item SR | 平均耗时 |
|---|---|---|---|---|---|
| π0 | 46.3 | 29.7 | 31.8 | 35.6 | 2m59s |
| GR00T N1 | 38.9 | 35.4 | 29.5 | 33.2 | >5m |
| WALL-OSS | 27.4 | 14.2 | 27.3 | 12.1 | >5m |
| AgiBot GO-1 | 57.8 | 48.6 | 33.2 | 38.4 | 3m57s |
| **Green-VLA(R0)** | **83.1** | **52.1** | **63.7** | **69.5** | **1m35s** |

**SimplerEnv WidowX（Qwen3-VL-4B 骨干，Success 均值）**：Green-VLA(R1) 72.9% → Green-VLA(R2) **80.5%**,同期 DB-MemVLA 73.2%、X-VLA 65.6%、GR00T-N1.6 58.3%、EO-1 72.7%,R2 相对 R1 提升约 8 个百分点,论文称在此设置下 R2 相对 R1 绝对提升约 24%（Simpler BRIDGE 综合口径）。

**SimplerEnv Google Robot（Qwen3-VL-4B, R1）**：整体平均 71.8%（Visual Matching 77.0% / Variant Aggregation 66.7%）,高于 EO-1（69.8%）、X-VLA（63.8%）、GR00T-N1.6（60.5%）等同期公开基线,PaliGemma-3B 版本 R0/R1/R2 分别为 45.1/48.1/57.3,验证阶段递进有效性。

**电商货架拣选（JPM 引导消融）**：Top-1 成功率在 ID-Coarse/ID-SKU/OOD 三种设定下,不加引导为 82.3/38.7/10.7,加 JPM+引导后为 85.4/89.1/72.8——OOD 场景提升约 6.8 倍,是论文中最突出的单项增益。R1→R2 在难抓取品类上同样明显：Cookies 30.4→82.7、Deodorant 62.8→88.3、Pet food 25.3→68.9（Shampoo 13.8→22.4 提升较有限）。

**Green 人形机器人真机评估**：Pick 98、Place in basket 100、Pick from basket 77、Hand-over to user 99、Give item 84、Clean full table 87,平均 90（策略频率 12Hz、控制频率 50Hz）。CALVIN ABC→D 上 R1 与 π0 微调后表现相当,R2 在平均链长（ACL）指标上取得最大增益,体现 RL 对齐对长程一致性与错误恢复的作用。

## 四、评价与展望

**优点**：整套课程按 R0→R1→R2 逐阶段做了独立验证,论证链条完整,不是简单堆叠模块；统一动作空间用"掩码 BC 损失避免伪惩罚"的论证方式把一个此前多被当作工程细节的问题（零填充跨具身动作空间）形式化并给出干净解法；光流驱动的时序对齐 + 速度调制把"数据源快慢不一"这一普遍痛点转成了一个可学习、可调的连续超参数,而不是逐数据集手工调采样率；两条 RL 对齐路径都刻意不直接改动基座流匹配权重（Q 梯度轨迹级修正需环境回放验证、噪声分布优化冻结基座）,这是在流匹配模型上做 RL（PPO/GRPO 类方法估计对数似然困难）时的务实取舍。在仅约 3,000 小时机器人数据（对比 π0 声称的逾 1 万小时）下于 SimplerEnv WidowX 取得同期公开 SOTA（80.5%）,支持了"质量对齐 + 统一动作空间"优于单纯数据规模堆叠的核心论点。

**与已有公开工作的关系**：架构上延续 π0/π0.5 的流匹配动作专家路线,L1 阶段的网络语义先验借鉴 EO-1,高效推理设计参考 WALL-OSS 的 MoE/低延迟思路,基线对比覆盖 GR00T N1/N1.6、AgiBot GO-1、X-VLA、DB-MemVLA、Flower 等同期公开 VLA。JPM+引导模块本质是把 RoboPoint 一类指向式 affordance 预测与 $\Pi$GDM（Song et al., 2023）扩散后验引导组合起来解决电商场景的开放集目标定位,是已知技术的一次针对性工程组合,而非引导理论本身的创新。

**开放问题与可能的改进方向**：(1) 从非拟人机械臂/夹爪到五指灵巧手的 retargeting 依赖"最近可行关节配置"这一启发式 IK 映射,论文未报告映射保真度的量化指标,跨具身灵巧操作迁移质量目前仍缺乏独立验证；(2) R2 轨迹级 Q 梯度优化要求人工在环回放验证后才能入库,本质上仍是离线批处理而非真正闭环在线 RL,可扩展性受限；(3) 高层任务规划器（GigaVision/GigaChat）在训练中完全冻结、与底层动作专家分离,系统仍是经典的双时间尺度（系统一/系统二）架构,论文自己也把"耦合快速推理与实时控制"列为未来工作,尚未给出解决路径；(4) 论文强调"实时部署"但仅报告了人形机器人的策略/控制频率（12Hz/50Hz）,未提供与基线在参数量、FLOPs、端到端延迟上的系统性对比表,效率优势目前主要是定性论断；(5) 评测任务集中在拾取-放置/货架拣选/桌面清理,长程全身移动操作或高精度装配等更苛刻的长程任务尚未纳入基准,与论文反复强调的"长程执行"动机之间还有验证空白。

## 参考

1. Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. Black, K. et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization*. 2025.
3. Qu, D. et al. *EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control*. 2025.
4. Zhai, A. et al. *Igniting VLMs Toward the Embodied Space* (WALL-OSS). 2025.
5. Bjorck, J. et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. 2025.
