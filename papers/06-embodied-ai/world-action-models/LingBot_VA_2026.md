# LingBot-VA：面向机器人控制的因果世界建模

> **论文**：*Causal World Modeling for Robot Control*
>
> **作者**：Lin Li*, Qihang Zhang*†, Yiming Luo*, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, Yinghao Xu‡ et al.（*同等贡献，†项目负责人，‡通讯作者）
>
> **机构**：蚂蚁集团（Ant Group）— Robbyant 项目组
>
> **发布时间**：2026 年 01 月（arXiv 2601.21998，v2 修订于 2026 年 3 月）
>
> **发表状态**：未录用（预印本），代码 / 模型权重已在 GitHub 与 HuggingFace 开源
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.21998) | [PDF](https://arxiv.org/pdf/2601.21998)
>
> **分类标签**：`世界模型` `视频-动作联合预测` `自回归扩散` `VLA` `Mixture-of-Transformers` `闭环控制`

---

## 一句话总结

LingBot-VA 用一个统一的自回归扩散序列，把"视频世界模型的未来帧预测"和"逆动力学的动作解码"用因果注意力交织在一起（双流 Mixture-of-Transformers + KV-cache 长时记忆 + FDM-grounded 异步推理防止漂移），在 RoboTwin 2.0 上取得 92.93%/91.55%（Easy/Hard）、LIBERO 平均 98.5% 的成绩，均超过 π0.5、X-VLA、Motus 等强基线。

## 一、问题与动机

主流 VLA 策略是一个从观测直接映射到动作的前馈网络 $a_t \sim \pi_\theta(\cdot \mid o_t)$，用一份监督信号同时学习视觉理解、物理动力学、运动控制，作者称之为"表征纠缠"（representation entanglement）——这个瓶颈导致样本效率低、泛化差，本质上是模式匹配而非对物理规律的原理性理解。

近期把世界模型引入机器人策略的工作（交互式神经模拟器如 UniSim、分块式视频-动作扩散如 UVA、离线视频生成做子目标合成如 Gen2Act/Act2Goal）虽然概念上有吸引力，但共同存在三个缺陷：

1. **反应性缺口**：分块/开环生成整段轨迹后才执行，难以引入实时反馈应对扰动；
2. **长时记忆有限**：逐块生成时若不持久缓存历史，长时程会出现前后不一致和漂移；
3. **违反因果性**：段内双向注意力让未来 token 影响过去预测，这与"现在只依赖过去"的物理因果律相悖。

这三点促使作者提出一个**自回归** 的视频-动作世界模型公式，从架构上强制因果一致性。

## 二、核心方法

**两阶段问题分解**。不同于直接学 $\pi_\theta(a_t\mid o_t)$，作者把控制拆成显式的两步：

$$
\text{(1) 视觉动力学预测：}\ o_{t+1}\sim p_\theta(\cdot\mid o_{\le t}),\qquad
\text{(2) 逆动力学：}\ a_t\sim g_\psi(\cdot\mid o_t, o_{t+1})
$$

用大白话说：先"想象"接下来会发生什么（想象未来一帧或几帧画面），再问"要让画面变成这样，我该做什么动作"——世界建模阶段可以吃海量无动作标注的视频数据学物理先验，逆动力学阶段只需少量机器人示教数据来落地到可执行动作。

**统一自回归序列**。把这一分解扩展为分块（chunk）形式，并显式条件化在动作历史上：

$$
z_{t+1:t+K}\sim p_\theta(\cdot\mid z_{\le t}, a_{<t}),\qquad
a_{t:t+K-1}\sim g_\psi(\cdot\mid \hat z_{t+1:t+K}, z_{\le t}, a_{<t})
$$

其中 $z_t=E(o_t)$ 是因果视频 VAE（继承自 Wan2.2）编码的潜在视觉 token，逆动力学不是只看当前/下一状态，而是额外接收动作历史（编码了末端执行器当前位姿轨迹）与观测历史（编码了"物体是否已被抓取"等多步上下文），从而更准确地回答"要达成这个预测的视觉目标该出什么动作"。

**双流 Mixture-of-Transformers 架构**。视频流从 Wan2.2-5B（$d_v=3072$，30 层）初始化，动作流同深度但更窄（$d_a=768$，约 4 倍缩小，新增约 3.5 亿参数，总模型约 5.3B），两流各自独立做 QKV 投影、再通过跨模态注意力融合，保留各模态特有的表征空间又允许相互影响。视频按时间因子 $\tau=4$ 稀疏化下采样，每帧配 $\tau$ 个连续动作交织成序列 $[z_t, a_{t,1},\dots,a_{t,\tau}, z_{t+1},\dots]$，即预测 $K$ 个视频帧等价于生成 $\tau K$ 个动作，用低频视频生成支撑高频动作输出。

**动作网络初始化** 是稳定训练的关键点：从零随机初始化会因动作 token 输出分布与视频分布差异巨大而扰乱联合注意力，导致训练发散（梯度范数高、收敛慢）。作者的方案是用缩放系数

$$
\alpha=\sqrt{d_v/d_a}
$$

将预训练视频权重插值后赋给动作流，以保持输出方差匹配，实验（Fig. 7）显示该策略比随机初始化和直接复用权重都更稳、收敛更快。

**Noisy History Augmentation（噪声历史增强）**。训练时以概率 0.5 对视频历史 $z_{\le t}$ 按流匹配插值方式注入噪声（$s_{\rm aug}\sim {\rm Uniform}[0.5,1]$），让逆动力学模型学会从部分去噪的视觉表征中提取动作相关信息，而不依赖像素级完美重建。用大白话说：动作解码不需要等视频"画面画清楚"才能猜出该做什么动作，半成品的画面草稿就足够。推理时因此只需把视频去噪到 $s=0.5$（而非 $s=1$），去噪步数直接减半，同时动作预测精度基本不掉。

训练目标是视觉动力学损失与逆动力学损失之和：$\mathcal{L}=\mathcal{L}_{\rm dyn}+\lambda\mathcal{L}_{\rm inv}$，二者均为条件流匹配的速度场回归损失，通过教师强制（teacher forcing）+ 因果注意力掩码在单次前向中联合优化，类比 LLM 的下一 token 预测训练。

**KV-cache 与 FDM-grounded 异步推理**。自回归结构天然支持 KV-cache 复用历史，避免重复计算。为解决推理延迟，作者进一步设计异步流水线：机器人执行当前动作块的同时，模型并行预测下一块。但朴素异步实现会因视频生成模型倾向"顺畅续写"预测画面而忽视最新真实观测，导致开环退化和轨迹漂移；解决办法是引入 **Forward Dynamics Model（FDM）grounding**——用最新真实反馈 $z_{t-1}$ 和已执行动作重新"想象"出 $z_t$ 并写入缓存，强制后续预测锚定在真实反馈上再继续推演，对应新增损失 $\mathcal{L}_{\rm fdm}$。消融显示（Table 3）朴素异步在 RoboTwin Easy 上只有 74.3%，FDM-grounded 异步恢复到 90.4%（同步基线 92.9%），且异步推理比同步快 2 倍。

**数据与规模**：预训练语料聚合 AgiBot World、RoboMind、InternData-A1、OXE（OpenVLA 子集）、UMI Data、RoboCOIN 六个公开数据集，约 1.6 万小时机器人操作数据；统一 30 维双臂动作表示（每臂 7 维末端位姿 + 7 维关节角 + 1 维夹爪）；预训练 1.4T token，chunk size 训练时从 $[1,4]$ 随机采样，部署时固定 $K=4$。

## 三、关键结果

**RoboTwin 2.0（50 任务，Table 1）**：

| 方法 | Easy 平均 | Hard 平均 |
|---|---|---|
| X-VLA | 72.9 | 72.8 |
| π0 | 65.9 | 58.4 |
| π0.5 | 82.7 | 76.8 |
| Motus | 88.7 | 87.0 |
| **LingBot-VA（Ours）** | **92.93**（+4.2） | **91.55**（+4.6） |

按任务步数（horizon）细分，horizon = 3（最长程）的提升最明显：+8.2（Easy）/+9.1（Hard），验证自回归长时记忆对长程任务更有效。

**LIBERO（Table 2）**：Spatial 98.5±0.3、Object 99.6±0.3、Goal 97.2±0.2、Long 98.5±0.5，平均 **98.5**，超过 X-VLA（98.1）、UniVLA（95.4）、π0（94.1）、OpenVLA-OFT（97.1）等在内的所有对比方法。

**真实机器人（6 任务，50 demo post-train）**：长程任务 Make Breakfast / Unpack Delivery、精细任务 Insert Tubes / Pick Screws、可变形物体 Fold Clothes / Fold Pants，在成功率和进度分两个指标上均一致超过 π0.5 基线（结论部分给出"在高难度任务上相比 π0.5 提升超过 20%"）。

**样本效率**（Fig. 8）：仅用 10 条示教，在真实 Make Breakfast 任务上进度分比 π0.5 高 15.6 个百分点，在 RoboTwin Easy 上高 10.3 个百分点。

**长时记忆专项测试**（Fig. 9）：设计 Wipe Plate（须精确擦拭 6 次，需要计数记忆）和 Search Box（先开右箱扑空后须记得去开左箱，无记忆则退化为 50% 随机猜）两个任务，LingBot-VA 均达到 100% 成功率，π0.5 仅 47%/50%，直接验证了因果自回归 + KV-cache 带来的持久记忆能力。

**预训练消融**（Table 3）：把预训练骨干换成未做视频-动作联合预训练的原始 WAN（Wan2.2）、同样流程微调后仅 80.6%（Easy），远低于 LingBot-VA 完整预训练的 92.9%，证明视频-动作联合预训练本身（而非架构或微调数据）是性能的主要来源。

## 四、评价与展望

**优点**：把"想象未来画面"和"从画面推动作"显式分解为因果自回归序列中的两个角色，是对现有 chunk 式视频-动作扩散模型（如 UVA、UWM）在因果性上的一个干净的架构修正；Noisy History Augmentation 与 FDM-grounded 异步推理是两个务实的工程贡献，直接针对"视频生成太慢难以做闭环控制"这一世界模型落地机器人的核心痛点给出可复现的解法；RoboTwin 2.0 长程任务上的增益规律（horizon 越长优势越大）与专门设计的记忆任务（Wipe Plate / Search Box）共同构成了较有说服力的因果假设验证，而非只靠榜单数字。

**局限与开放问题**：论文自身在结论中承认两点未来方向——视频压缩效率仍是计算开销的主要来源（尽管有噪声增强和异步加速，完整方案仍需并行运行视频流和动作流两个 transformer）；当前只用视觉观测，未引入触觉、力觉、听觉等模态，对强接触动力学任务（如插拔、装配）可能仍有信息缺口。此外，从论文细节看：（1）部署时 chunk size 固定为 $K=4$，是预测视野与控制频率之间的一个人工权衡，未见对该超参更细粒度的敏感性分析；（2）真实世界评测仅在同一批次的自建任务和数据上进行，样本量有限（20 trial/任务），跨平台/跨具身的零样本能力主要靠"post-training 用 50 条示教即可适配新平台"这一定性说法支撑，缺少更大规模的跨本体泛化基准；（3）与同期同样做"视频-动作统一自回归/扩散"的 Motus、UVA、UWM 等工作相比，LingBot-VA 的差异化主要在因果注意力 + KV-cache 记忆与异步 grounding 机制上，但论文未直接对比这些方法在同等训练数据规模下的表现（Motus 结果为引用自身论文数字），公平性上留有一定空间。

## 参考

- π0.5: A generalist robot policy with flow matching and world models（Physical Intelligence et al.）
- Motus: A unified latent action world model（Bi et al.）
- Unified World Models: Coupling video and action diffusion for pretraining on large robotic datasets（Zhu et al.，UWM）
- RoboTwin 2.0: A scalable data generator and benchmark for robust bimanual manipulation（Chen et al.）
- Mixture-of-Transformers: A sparse and scalable architecture for multi-modal foundation models（Liang et al.）
