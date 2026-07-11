# DIM-WAM：基于多类型历史事件记忆的世界-动作建模

> **论文**：*DIM-WAM: World-Action Modeling with Diverse Historical Event Memory*
>
> **作者**：Kai Wang, Zhaopeng Gu, Yixiang Chen, Yuan Xu, Qisen Ma, Peng Su, Zhaowen Li, Yan Huang, Liang Wang（通讯作者：Zhaowen Li、Yan Huang）
>
> **机构**：中国科学院自动化研究所模式识别国家重点实验室（NLPR, CASIA）；引望智能技术（Yinwang Intelligent Technology）
>
> **发布时间**：2026 年 06 月（arXiv 2606.27677）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.27677) | [PDF](https://arxiv.org/pdf/2606.27677)
>
> **分类标签**：`世界-动作模型` `记忆增强` `长时程操作` `多库记忆` `任务进度监督`

---

## 一句话总结

在世界-动作模型（World-Action Model, WAM）LingBot-VA 之上外挂一个由 8 个独立记忆库、每库 12 槽构成的"多类型历史事件记忆"（相似度+时间衰减合并写入、闭环真实观测更新、任务进度辅助监督），把 RMBench 长时程双臂操作任务的平均成功率从 28.4% 提到 69.8%（超过显式记忆基线 Mem-0 的 42.0%），真机 Franka Panda 四任务全任务成功率从 52.5% 提到 80.0%。

## 一、问题与动机

现有 WAM（GR-2、LingBot-VA、DreamZero 等）通过联合预测未来视觉状态和动作，把视频生成模型学到的时空与物理先验引入机器人控制，在短时程闭环操作上表现良好，但主要依赖短期历史和短视野未来预测，难以处理长时程任务——这类任务的正确执行依赖更早的观测和任务进度，一旦关键早期状态滑出局部窗口就会遗忘。

作者指出长时程操作至少需要四类互补的时间信息：短期历史（刚发生的事）、跨阶段历史（哪些子任务已完成）、短期未来动态（即将发生的视觉演化和动作后果）、全局任务进度（当前状态相对整个任务的目标相对位置）。简单地拉长上下文窗口会带来信息量和计算量的同时增长，且注意力竞争会稀释早期但关键的状态；更麻烦的是，如果把 WAM 自身生成的预测误差写入长期上下文，模型的内部状态会逐渐偏离真实环境（误差累积）。

论文将自己定位为对已有记忆增强 VLA 方法（SAM2Act+、MemoryVLA、MEM、ReMem-VLA 等，主要面向动作预测的显式检索式记忆）的互补：DIM-WAM 的记忆同时参与未来视觉状态生成和动作预测，且强调记忆只能被真实观测更新，从而把"预测假设"和"事实记忆"解耦，抑制生成误差向后续时段传播。

## 二、核心方法

**整体框架**：以 LingBot-VA 为基座 WAM（联合视频-动作隐空间建模，用真实观测做闭环校正），在其局部上下文（滑动窗口 + KV cache）之外增加一个外部固定容量的长期记忆模块，并用任务进度监督塑造记忆表征。每个决策片段 $i$ 相关的时间信息被显式分解为

$$\mathcal{I}_i = \left(\mathcal{H}_i^{\text{long}}, \mathcal{H}_i^{\text{short}}, \mathcal{F}_i^{\text{short}}, \mathcal{G}_i^{\text{prog}}\right)$$

即跨阶段历史事件、短期历史、短期未来演化、全局任务进度四部分。模型预测为

$$(\hat{\mathbf{z}}_{i:i+H}, \hat{\mathbf{a}}_{i:i+H}) = f_\theta(\mathcal{C}_i, c; \mathcal{M}_i)$$

其中局部上下文 $\mathcal{C}_i$ 来自滑窗+KV cache，长期记忆 $\mathcal{M}_i$ 作为额外条件参与视频/动作去噪。**直觉**：局部窗口管"刚才和马上"，外部记忆管"很久以前但仍重要"，两者联合条件化避免只靠拉长窗口。

**多库记忆结构**：不用单一记忆压缩所有历史（不同事件类型如初始物体位置、已完成子任务、最近状态变化的语义功能和保留时限不同，塞进同一容量会互相挤占槽位），而是构造 $K$ 个并行库

$$\mathcal{M}_i = \{\mathcal{M}_i^1,\dots,\mathcal{M}_i^K\},\qquad \mathcal{M}_i^k=\{\mathbf{m}_{i,k,1},\dots,\mathbf{m}_{i,k,N}\}$$

各库容量固定为 $N$ 槽，总容量恒为 $KN$ 个 token，不随轨迹增长。各库不预设语义角色，功能分工完全由联合训练（视频预测、动作生成、进度预测、跨库多样性损失）中自发涌现。

**写入（压缩）**：每个交互片段结束后，观测被压缩成一个紧凑视觉事件 token $\bar{\mathbf{u}}_i=g_\phi(\mathbf{X}_i)$，同一压缩信息喂给所有库（库身份不参与压缩阶段，只在读取时注入）。各库独立地基于相似度和时间衰减决定合并优先级：

$$\rho_k(u,v)=\frac{1+\cos(\mathbf{m}_{k,u},\mathbf{m}_{k,v})}{2}\exp\!\left(-\frac{|t_u-t_v|}{\tau}\right)$$

**直觉**：两个历史事件表征越相似、时间上越接近，就越"冗余"，越应该被合并腾出槽位。被合并的槽位做质量加权平均 $\mathbf{m}_{uv}=(\alpha_u\mathbf{m}_u+\alpha_v\mathbf{m}_v)/(\alpha_u+\alpha_v)$，累积质量 $\alpha_{uv}=\alpha_u+\alpha_v$ 记录该 token 已经压缩了多少历史证据。

每库槽位结构固定：槽 1 为初始状态锚点，槽 $N$ 为最新保留事件（不参与合并，先被保护），槽 $2,\ldots,N-1$ 为压缩后的中段历史。写入新候选时用自适应阈值判断是否值得写入：把候选与当前"最新事件"的冗余度 $\rho^{\text{new}}_{i,k}=\rho_k(N,\bar{\mathbf{u}}_i)$ 和中段历史里最冗余相邻对的 $\rho^{\text{hist}}_{i,k}$ 比较，

$$\rho^{\text{new}}_{i,k}\ge\rho^{\text{hist}}_{i,k}\ \Rightarrow\ \text{丢弃该候选（当前最新事件已足够代表新状态）}$$

否则合并中段最冗余的一对腾位，把新候选写入槽 $N$。**直觉**：只有当新观测比"继续压缩旧历史"更有信息量时才写入，避免频繁局部变化把稀疏但关键的早期事件挤掉。

**读取（跨库协同）**：每个库中每个有效槽位加上库身份嵌入 $\mathbf{b}_k$，按时间戳把所有库的 token 拉平排序并加 RoPE 时间编码：

$$\mathbf{R}_i=\mathrm{RoPE}_t\big(\mathrm{ChronoFlatten}(\{\mathbf{m}_{i,k,n}+\mathbf{b}_k\}_{k,n})\big)$$

$\mathbf{R}_i$ 同时作为视频去噪分支和动作去噪分支的共享记忆条件，注意力掩码保持因果结构（记忆条件视频/动作预测，但预测不回写记忆）。**关键闭环设计**：记忆 token 在推理时只是当前片段的临时注意力条件，预测结束后清空临时 K/V，只有在收到真实环境反馈后才更新持久记忆——这样短期预测利用世界模型的"预见"，但生成误差不会污染长期历史。

**任务进度辅助监督**：从聚合后的长期记忆读出序列预测离散任务进度分布

$$\mathbf{p}_i=\mathrm{Softmax}\big(h_{\text{prog}}(\mathrm{Pool}(\mathbf{R}_i))\big),\qquad \mathcal{L}_{\text{prog}}=\mathrm{CE}(\mathbf{p}_i,\,y_i=\mathrm{bin}(r_i))$$

$r_i$ 是该片段在完整任务中的完成比例。进度头不独立做规划、也不显式预测未来观测/动作，只是给记忆一个"全局目标相对时间位置"的正则信号。此外加入跨库多样性损失 $\mathcal{L}_{\text{div}}$（对不同库归一化平均表征的内积平方做惩罚），防止多库退化成同一份拷贝。总损失为视频损失、动作损失、多样性损失、进度损失的加权和。

## 三、关键结果

**基准与协议**：RMBench（基于 RoboTwin 2.0 的双臂操作基准）九个非马尔可夫任务，分为 5 个 $M(1)$ 任务（只需保留一个/少数关键早期观测）和 4 个 $M(n)$ 任务（需累积多次探索/交互结果）。作者先做了一个评测协议审计（Table I，以 put\_back\_block 为例）：加入腕部相机视角或用更长窗口/更大 stride 会把初始位置信息重新带入局部上下文，使短窗策略的成功率从接近理论随机水平（25%）跳升到 88%~100%，说明相机组合、窗口长度、下采样步长会造成信息泄漏、虚假拔高"长时程能力"。为避免泄漏，主实验统一用 head+front 视角、30 帧窗口、stride 1（Table II 中带 † 标记）。

RMBench 主结果（成功率 %）：

| 方法 | $M(1)$ 均值 | $M(n)$ 均值 | 总均值 |
|---|---|---|---|
| Diffusion Policy | 6.4 | 5.0 | 5.8 |
| ACT | 6.8 | 4.8 | 5.9 |
| $\pi_{0.5}$ | 14.4 | 5.5 | 10.4 |
| X-VLA | 11.8 | 7.3 | 9.8 |
| Mem-0（RMBench 官方显式记忆基线） | 52.8 | 28.5 | 42.0 |
| LingBot-VA†（本文基座 WAM） | 22.8 | 35.5 | 28.4 |
| **DIM-WAM†（本文）** | **80.6** | **56.3** | **69.8** |

九个任务中 DIM-WAM 在 8 个上取得最优，唯一例外是 Cover Blocks（Mem-0 68.0% vs. DIM-WAM 56.0%）。

真机实验（Franka Panda，4 个长时程任务：Find Blue Block、Line Swap、Triangle Swap、Press Twice，每任务 15–25 条示教、10 次独立试验，SSR=阶段成功率、SR=全任务成功率）：

| 方法 | 平均 SSR | 平均 SR |
|---|---|---|
| $\pi_{0.5}$ | 1.3 | 0.0 |
| Fast-WAM | ~0 | ~0 |
| LingBot-VA | 70.7 | 52.5 |
| **DIM-WAM** | **91.5** | **80.0** |

增益最集中在需要跨阶段保持目标身份/空间状态的任务：Find Blue Block 全任务成功率 10.0%→70.0%，Line Swap 60.0%→90.0%；而 Press Twice（示教覆盖了约 85% 最大长度、局部信息已足够）两方法都到 100%，说明长期记忆不会损害已经够用的局部控制能力。

**消融**（Swap T / Swap Blocks 两个代表性任务，Table IV）：记忆库数与容量分解比单纯扩容更重要——$1\times32$ 单库 75.5% → $4\times8$ 90.0% → 默认 $8\times12$ 96.5%；去掉进度预测头后从 96.5% 降到 90.5%，Swap Blocks 上降幅更大。跨库行为分析（对一次成功的 put\_back\_block 轨迹做 PCA 和时间线可视化）显示不同库倾向保留不同阶段/位置的事件（有的偏保留早期事件或阶段边界，有的偏保留中后段状态变化），表征空间上呈现可区分但有重叠的分布，证实增益来自互补的事件选择行为而非简单容量堆叠。

## 四、评价与展望

**优点**：（1）把"记忆"同时接入视频去噪和动作去噪两条分支，并坚持记忆只由真实观测（而非模型自身生成的未来状态）更新，这一闭环设计明确切断了 WAM 常见的生成误差累积到长期上下文的风险路径，是相对于纯粹拉长自回归上下文或单一记忆库方案的一个清晰改进点。（2）多库+相似度-时间衰减合并+质量加权的压缩机制把长视频生成领域（StreamingT2V、MALT、MemFlow、VideoMemory 等）的历史压缩思路系统地搬到了机器人 WAM 场景，并用消融证明"多库分解"本身（而不仅是容量）带来增益（75.5%→96.5%）。（3）评测协议审计（Table I）是方法之外的一个有价值的方法论贡献，揭示相机视角/窗口/下采样选择可能造成的信息泄漏，对该子领域普遍存在的"长时程能力"虚高问题具有警示价值。（4）同时提供仿真（RMBench）和真机（Franka Panda）验证，且真机上确实观察到与仿真一致的定性收益模式。

**局限与开放问题**：（1）记忆容量与库数（8×12=96 token）是固定超参，论文只在两个任务上做了容量-库数消融，未系统研究更长、更多阶段任务下压缩策略的可扩展性边界。（2）库的功能分工是训练中自发涌现、不可解释也不可预先指定，换任务域是否需要重新调库数 $K$ 尚不清楚。（3）唯一落败的 Cover Blocks 任务（显式记忆基线 Mem-0 反而更优）未被深入分析，提示压缩式多库记忆与显式检索式记忆（Mem-0、SAM2Act+ 一类）可能存在互补的适用边界，这是一个开放问题。（4）任务进度监督需要片段级完成比例标签 $r_i$（依赖任务阶段划分标注），并非完全自监督，增加了数据标注成本。（5）论文与 SAM2Act+、MemoryVLA、MEM、ReMem-VLA、EventVLA 等同期显式记忆 VLA 方法只在相关工作中做了文字层面的比较和定位，未给出直接的量化对照，实际优劣仍待第三方基准验证。（6）真机验证限于单一具身（Franka Panda）、单一第三人称视角、4 个任务、每任务 10 次试验，跨具身、跨相机配置的泛化性未测试。

## 参考

- LingBot-VA（本文基座 WAM）：*Causal world modeling for robot control*, arXiv:2601.21998
- RMBench：*Memory-dependent robotic manipulation benchmark with insights into policy design*, arXiv:2603.01229
- Mem-0（RMBench 官方显式记忆基线）
- ReMem-VLA：*Empowering vision-language-action model with memory via dual-level recurrent queries*, arXiv:2603.12942
- MemoryVLA：*Perceptual-cognitive memory in vision-language-action models for robotic manipulation*, arXiv:2508.19236
- GigaWorld-Policy：*An efficient action-centered world-action model*, arXiv:2603.17240
