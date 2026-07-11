# LoHoVLA：面向长时程具身任务的统一视觉-语言-动作模型

> **论文**：*LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks*
>
> **作者**：Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, Zhijie Deng†（通讯作者）
>
> **机构**：复旦大学（Fudan University）、上海科技大学（ShanghaiTech University）、上海交通大学（Shanghai Jiao Tong University）
>
> **发布时间**：2025 年 06 月（arXiv 2506.00411，首版提交于 2025-05-31）
>
> **发表状态**：未录用（预印本，论文标注 "Preprint. Under review."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.00411) | [PDF](https://arxiv.org/pdf/2506.00411)
>
> **分类标签**：`长时程规划` `统一VLA` `分层闭环控制` `子任务生成` `PaliGemma`

---

## 一句话总结

LoHoVLA 用单个 PaliGemma-3B 骨干共享同一个语言头，同时以下一 token 预测的方式生成子任务文本和离散动作 token，把长时程任务的高层规划与低层控制统一进一个模型，并配合"失败计数超阈值 $K$ 才重新规划子任务、否则只重新预测动作"的分层闭环控制机制来抑制误差累积；在自建的、基于 Ravens 模拟器的 LoHoSet（20 个长时程任务 + 3 个 pick-and-place 原语，每任务 1000 条专家示教）上，LoHoVLA 在几乎全部 seen/unseen 任务上取得最高平均分与成功率，例如在需要颜色识别+计数+空间推理+逻辑判断的 put-even-blocks-in-same-color-zone 任务上取得 85.1 分/81.0% 成功率，显著超过分层基线 LoHoRavens（约 8-37 分）和 vanilla VLA（普遍个位数或零成功率）。

## 一、问题与动机

真实世界的具身任务大多是长时程（long-horizon）的：高层目标无法通过单步动作完成，智能体必须先做任务分解、再执行多步动作，并能在失败或环境变化时调整策略。作者指出现有两条技术路线各有短板：

- **标准 VLA 模型**（如 RT-2、OpenVLA）直接把视觉观测与语言目标映射到底层动作，缺乏显式的多步推理和结构化任务分解能力，在长时程任务上规划能力弱；
- **分层架构**（如 LoHoRavens 一类工作）引入外部高层规划器（通常基于 LLM）把目标拆成子任务序列，再由独立的低层控制器（如 CLIPort）执行。这类模块化设计灵活，但规划器与控制器之间协调不畅、存在建模冗余，容易导致次优性能和泛化能力受限。

论文因此提出一个统一范式的问题：能否用一个共享的视觉-语言骨干，在同一套参数、同一个 token 空间里既做高层子任务推理又做低层动作预测，从而避免分层架构的协调开销，同时具备比 vanilla VLA 更强的显式规划与推理能力？

## 二、核心方法

**问题形式化**：策略 $\pi_\theta$ 根据视觉观测 $\mathbf o_t$ 和高层语言目标 $g$（如 "Clean the desk"）输出机器人动作 $\mathbf a_t$。目标 $g$ 隐含一串子任务序列 $[\hat g_1,\hat g_2,\dots,\hat g_N]$（如 "把笔放回笔筒"→"合上笔记本电脑"→"把书放回书架"），假设每个子任务可在单个时间步内完成。三种建模方式的对比：

$$
\text{Vanilla VLA:}\quad \pi_\theta(\mathbf o_t,g)\to \mathbf a_t
$$

用大白话说：vanilla VLA 只学"看到什么、要做什么"直接映射到动作，子任务规划是隐式的，模型内部到底有没有做规划、做得对不对完全不可见、不可控。

$$
\text{分层架构:}\quad \pi_\theta^{\text{planner}}(\mathbf o_t,g)\to \hat g_t,\quad \pi_\theta^{\text{controller}}(\mathbf o_t,\hat g_t)\to \mathbf a_t
$$

用大白话说：分层架构把"想清楚下一步做什么"和"具体怎么做"拆成两个独立模块，规划器先显式吐出子任务文字，控制器再据此生成动作，可解释性更强，但两个模块参数不共享、训练目标不一致，容易出现规划器说的和控制器做的对不上的协调问题。

$$
\pi_\theta(\mathbf a_t,\hat g_t \mid \mathbf o_t,g)=\pi_\theta(\mathbf a_t \mid \mathbf o_t,g,\hat g_t)\cdot \pi_\theta(\hat g_t \mid \mathbf o_t,g)
$$

用大白话说：LoHoVLA 把上面两步串成一个联合分布——同一个模型先自回归地"说出"下一个子任务 $\hat g_t$，再把这个子任务文本本身当作条件，接着"说出"当前动作 $\mathbf a_t$，高层规划和低层控制共用同一套参数和同一个 token 序列，规划结果天然地成为动作预测的上下文。

**模型架构**：骨干选用 PaliGemma-3b-mix-224（SigLIP 图像编码器 + Gemma-2B 解码器语言模型 + 线性投影层对齐视觉特征到语言 token 空间）。沿用 RT-2/OpenVLA 一类做法，将归一化后的动作值离散化为 1024 个均匀分箱，作为普通文本 token 附加到词表中，用同一个语言模型头联合生成子任务文本 token 和动作 token；推理时再由专门的"动作反 tokenizer"把离散 token 解码、反归一化回连续动作。训练损失为文本损失与动作损失之和（均为交叉熵）：

$$
\mathcal L=\mathcal L_{\text{text}}+\mathcal L_{\text{action}}
$$

**分层闭环控制机制**：作者把长时程任务的执行失败归为三类——(1) 子任务规划错误、(2) 规划正确但动作预测错误、(3) 规划和预测都正确但受到外部扰动。为避免"错误就重新规划"造成不必要的高层推理开销，LoHoVLA 采用阈值化策略：维护当前子任务的失败计数 $k$，只有当 $k$ 超过预设阈值 $K$（实验中取 $K=2$）时才触发子任务重新规划，否则仅重新预测动作。测试时的闭环控制流程（Algorithm 1）：初始化 $t=0,k=0,r=0,done=false$；循环中若 $t=0$ 或上一步获得正奖励 $r>0$ 或 $k>K$，则重新采样子任务 $\hat g_t\sim\pi_\theta(\hat g_t\mid \mathbf o_t,g)$ 并清零 $k$；随后采样动作 $\mathbf a_t\sim\pi_\theta(\mathbf a_t\mid \mathbf o_t,g,\hat g_t)$ 并执行，获取新观测、奖励和 done 标志；若本步奖励为 0（子任务未完成）则 $k\gets k+1$。用大白话说：只要还在为同一个子任务反复"手滑"，就不折腾高层重新想计划，先多让底层控制器多试几次；只有连续失败次数攒够了才怀疑是"想错了方向"，才回头重新规划。

**LoHoSet 数据集构建**：基于 Ravens 机器人模拟器（UR5e 机械臂 + 吸盘夹爪），场景含 blocks（大小两种）、bowls、zones 三类物体、11 种颜色；模拟器每秒以一定概率让夹爪掉落已抓取的方块以模拟真实不确定性，观测为 RGB + 深度的俯视正交重建图像。子任务标注由人工设计的规则基于模拟器提供的完整场景信息（目标物体位置）自动生成，动作直接表示为 $\mathbf a=(\mathcal T_{\text{pick}},\mathcal T_{\text{place}})$；对无依赖约束的子任务随机排序，有依赖约束的按预定义逻辑生成顺序，再用子任务模板套出自然语言描述。数据集包含 20 个长时程任务（其中 10 个沿用 LoHoRavens 基准以便与基线对比、另外 10 个为论文新增以增强泛化）加 3 个 pick-and-place 原语任务，每个长时程任务 1000 条专家示教。

**两阶段训练**：第一阶段仅优化文本损失，在 14 个长时程任务（4 个 LoHoRavens seen 任务 + 10 个新增任务）上全参数微调 PaliGemma，用 8 张 NVIDIA 4090（24GB）、per-device batch size 2、学习率 5e-5，训练 3 epoch（DeepSpeed 分布式）；第二阶段引入 3 个 pick-and-place 原语（每原语 10,000 条示教），同时优化文本损失和动作损失以强化低层控制，用 LoRA（rank 16，作用于所有线性层）、学习率 1e-5，训练 1 epoch。对照的标准 VLA 基线在相同数据、相同硬件与 LoRA 配置下训练 5 epoch，但不使用子任务标签。

## 三、实验结果

**主结果**（Table 2，LoHoRavens 基准 seen 任务 A-E / unseen 任务 F-K，指标为"平均得分/成功率"，得分按完成的 pick-and-place 步骤比例计算）：

| 任务 | Vanilla VLA | LoHoRavens (显式反馈) | LoHoRavens (隐式反馈) | LoHoVLA |
|---|---|---|---|---|
| A pick-and-place-primitive | **79.0**/79.0 | 67.3/– | 67.3/– | 77.5/77.5 |
| B put-block-into-matching-bowl | 14.9/0.0 | 31.4/– | 37.0/– | **97.8**/91.5 |
| C stack-smaller-over-bigger-same-color | 26.8/0.5 | 18.0/– | 22.1/– | **34.9**/22.5 |
| D stack-block-in-absolute-area | **32.3**/3.0 | 30.4/– | 33.2/– | 35.8/11.5 |
| E put-even-blocks-in-same-color-zone | 22.1/3.5 | 9.6/– | 8.2/– | **85.1**/81.0 |
| F put-block-into-mismatching-bowl（unseen） | 52.1/9.0 | 28.5/– | 21.1/– | **86.1**/41.0 |
| G stack-blocks-of-same-size（unseen） | 6.8/0.0 | 21.9/– | 14.7/– | **40.1**/25.0 |
| H stack-blocks-alternate-color（unseen） | 7.3/0.0 | 13.2/– | 5.2/– | **16.7**/7.5 |
| I stack-smaller-over-bigger-in-zone（unseen） | 43.1/1.5 | 12.8/– | 11.7/– | **77.2**/52.0 |
| J move-blocks-between-positions（unseen） | 38.6/10.5 | 27.4/– | 27.2/– | **43.6**/22.0 |
| K stack-blocks-of-same-size（unseen） | 58.2/33.0 | 4.0/– | 6.8/– | **73.8**/54.5 |

LoHoVLA 在 11 项任务中的 9 项取得最高平均分/成功率（A、D 略逊于 vanilla VLA，但差距很小），尤其在需要综合颜色识别、计数、空间推理与逻辑判断的 E 任务上以 85.1/81.0 大幅领先所有基线；在全部 unseen 任务上都稳定优于两个基线，体现出较强的跨任务泛化能力。Vanilla VLA 因缺乏子任务监督普遍表现最差（多个任务 0% 成功率），定性分析显示其倾向于过拟合训练数据中的高频模式——例如在 put-block-into-matching-bowl 任务中常常无视目标条件、把方块放入错误的碗中。

**分层闭环控制策略对比**（Table 3，三种策略均基于同一个微调好的 LoHoVLA 模型，数值为"平均得分/成功率/平均高层规划次数"）：策略 (a) 失败后只重新预测动作、从不重新规划子任务，表现最差（例如 B 任务仅 89.5/74.0，规划次数 5.7）；策略 (b) 每次失败都重新规划+重新预测动作，与策略 (c)（阈值化分层控制，$K=2$）在得分/成功率上总体相当（如 B 任务 (b) 96.4/88.5 vs (c) 97.8/91.5），但 (c) 所需的高层规划调用次数普遍更少（如 B 任务 6.4 次 vs 6.2 次，K 任务 7.2 次 vs 6.9 次）。原因是许多失败源于低层动作预测误差或外部扰动而非规划错误，此时重新规划子任务并无必要；对推理密集型任务（如 E），策略 (b) 因能及时修正子任务分配略占优；对需要精细运动控制的任务（如 D），策略 (c) 允许底层多次重试更有利。

**消融实验**：（1）训练集扩展——去掉论文新增的 10 个额外任务后，模型在 unseen 任务上的子任务规划成功率大幅下降（如 F 任务 put-block-into-mismatching-bowl 从扩展后的 100.0% 跌至 0%），原因是模型把该任务场景误认成训练中见过的相似任务 put-block-into-matching-bowl，忽略语言目标、机械地把方块放进颜色匹配的碗中，说明扩展数据集能有效缓解对 seen 任务的过拟合；（2）两阶段训练——在相同 5 epoch 总预算下，两阶段训练（前 3 epoch 只训文本损失，后续再引入动作标签和原语任务）比一阶段训练（从头同时训文本+动作损失）取得更高的子任务规划成功率（约 85.4% vs 80.9%）和任务完成成功率（约 44.2% vs 40.0%），表明过早引入动作监督会干扰高层规划能力的有效优化。

## 四、局限性

论文在结论中明确指出两点局限：（1）机器人动作被离散化为 1024 个均匀分箱来表示，这种离散结构本质上限制了动作精度，难以支持需要连续高精度控制的接触密集型操作；（2）模型假设每个子任务可在单一时间步内完成，这一假设在实时应用中未必成立，真实世界的子任务往往需要多步、变长的执行过程。此外，从实验设置看还存在未明说的局限：全部实验都在 Ravens 模拟器的 tabletop pick-and-place 场景中完成，未在真实机器人或更复杂的操作技能（如接触丰富的装配、可变形物体操作）上验证；对比基线局限于同一基准家族内的 LoHoRavens 和自建 vanilla VLA，未与更广泛的分层规划方法（如基于 LLM 的开放世界任务规划器）或近期的扩散式/自回归统一 VLA（如 DiVLA、$\pi_0$ 系列）做直接对比；骨干模型规模较小（Gemma-2B 语言模型），扩大骨干规模或替换为更强的 VLM 是否能进一步放大统一架构相对分层架构的优势尚未验证。

## 五、评价与展望

LoHoVLA 的核心贡献是一个简洁但有效的观察：把"高层子任务规划"形式化为与"低层动作预测"共享同一个 token 空间、同一套参数的联合分布 $\pi_\theta(\mathbf a_t,\hat g_t\mid \mathbf o_t,g)$，而不是像 LoHoRavens 一类工作那样用独立的 LLM 规划器 + 独立的动作控制器拼接。这一设计延续了 RT-2/OpenVLA 把动作离散化为语言 token、复用 VLM 预训练能力的思路，但把它显式扩展到了"文本子任务 token + 动作 token"的联合自回归生成，为长时程任务提供了一条不依赖外部规划模块的统一路径。分层闭环控制机制（失败计数阈值化决定是否重新规划）是一个工程上务实的补充，用极低的额外开销（只需维护一个失败计数器）在实验中证明能减少不必要的高层推理调用，同时保持与"每次失败都重新规划"相当的任务成功率。

与其他公开工作相比，LoHoVLA 的比较对象聚焦于同一系列基准（LoHoRavens）内部，实验规模也局限于一个模拟器环境，这使得其"统一架构优于分层架构"的结论目前只在这一相对受限的场景下得到验证——LoHoRavens 使用的 Planner（LLaMA 2-13B）与 Actor（CLIPort）本身并非专为该基准联合训练和调优的强基线，二者性能差距在多大程度上归因于"统一 vs 分层"这一架构选择、又在多大程度上归因于骨干模型和数据规模的差异，论文未做严格的控制变量实验。此外，子任务标注来自 Ravens 模拟器提供的特权信息（物体真值位置）并用规则生成，这类"仿真中天然可得、真实世界中昂贵"的监督信号是否能规模化到真实机器人数据仍是开放问题，也是当前"统一 VLA 做长时程规划"这一方向普遍面临的数据瓶颈——即如何低成本地为真实世界长时程示教获取细粒度子任务标注。动作离散化到 1024 分箱的精度上限、以及"子任务在单步内完成"这一简化假设，都指向了将该框架推广到更精细、更长时程真实任务时需要解决的关键问题：如何在保持统一架构简洁性的同时，引入变长子任务执行和更高精度的动作表征（如结合动作分块或连续动作头）。

## 参考

- Zhang et al. LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation. arXiv:2310.12020, 2023.
- Zeng et al. Transporter Networks: Rearranging the Visual World for Robotic Manipulation. CoRL 2021.（Ravens 模拟器）
- Beyer et al. PaliGemma: A Versatile 3B VLM for Transfer. arXiv:2407.07726, 2024.
- Brohan et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818, 2023.
- Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246, 2024.
- Shridhar et al. CLIPort: What and Where Pathways for Robotic Manipulation. CoRL 2022.
