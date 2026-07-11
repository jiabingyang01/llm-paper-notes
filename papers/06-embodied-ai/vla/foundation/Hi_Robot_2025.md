# Hi Robot：基于分层视觉-语言-动作模型的开放式指令跟随

> **论文**：*Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models*
>
> **作者**：Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, Sergey Levine, Chelsea Finn et al.
>
> **机构**：Physical Intelligence、Stanford University、University of California, Berkeley
>
> **发布时间**：2025 年 02 月（arXiv 2502.19417，v2 于 2025 年 07 月更新）
>
> **发表状态**：Proceedings of the 42nd International Conference on Machine Learning（ICML 2025，PMLR 267）
>
> 🔗 [arXiv](https://arxiv.org/abs/2502.19417) | [PDF](https://arxiv.org/pdf/2502.19417)
>
> **分类标签**：`分层VLA` `开放式指令跟随` `情境纠正` `合成数据生成` `System1-System2`

---

## 一句话总结

Hi Robot 用两个几乎同构的 VLM（均以 PaliGemma-3B 初始化）搭出"System 2 想、System 1 做"的两层结构——高层 VLM 把复杂开放式指令 / 实时用户打断翻译成低层能听懂的原子指令，低层沿用 π0 的 flow-matching VLA 执行；训练数据的关键补丁是用一个数据生成 VLM 把人工示教里的原子技能标签反推出"合理的用户提问 + 机器人应答"，合成出规模化的复杂指令语料。在三平台（单臂 UR5e、双臂 ARX、移动双臂 ARX）三任务域（清桌、做三明治、超市购物）上，Hi Robot 的指令准确率平均比 GPT-4o 高层方案高出 40% 以上，且消融显示合成数据和分层结构本身都是不可或缺的组成部分。

## 一、问题与动机

现实中的机器人指令很少是"pick up the coke can"这种原子命令，而更像"帮我做个素三明治，如果有火腿或烤牛肉，另外单独给我朋友做一份"——这既要求解析复杂语言、把复合任务分解为已有技能的组合，还要在执行过程中动态吸收纠正和反馈（"那个不是垃圾""放低一点""我对泡菜过敏"）。作者把这种区分类比为 Kahneman 的 System 1（自动化、触发已学技能）与 System 2（深思熟虑、需要对任务与反馈做推理）。

已有工作大致分两类，都不满足要求：

- **端到端语言条件模仿学习**（RT-1、RT-2、OpenVLA、π0 等）只在"简单原子指令"上训练，遇到长复合指令或中途反馈时缺乏鲁棒性；
- **LLM/VLM 做任务分解 + 预定义技能库**（SayCan、Code as Policies 等）能处理更复杂的指令组合，但技能集合固定、动作不够灵巧，且大多数缺乏对实时人类语言交互（尤其是情境化纠正）的支持。

Hi Robot 的核心诉求：让同一套语言接口既能做高层推理，又能驱动低层灵巧动作，并且能在任务执行过程中随时被用户打断、澄清、纠正。

## 二、核心方法

### 2.1 问题形式化

策略以观测 $\mathbf{o}_t$ 为输入产生动作块 $\mathbf{A}_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$（$H$ 步动作 chunk，遵循 action-chunking 思路）。观测包含多路相机图像、机器人关节 / 夹爪状态 $\mathbf{q}_t$、以及语言提示 $\ell_t$：$\mathbf{o}_t = [\mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \ell_t, \mathbf{q}_t]$，策略即 $p(\mathbf{A}_t \mid \mathbf{o}_t)$。标准 VLM 用自回归方式表示语言后缀相对于图像-语言前缀的分布：

$$p(\ell' \mid \mathbf{I}, \ell) = \prod_{t} p(\mathbf{x}_{t+1} \mid \mathbf{x}_1, \ldots, \mathbf{x}_t, \mathbf{I})$$

用大白话说：这就是"看图+读提示，逐词预测答案"的标准 VLM 建模，Hi Robot 高层和低层两个模型都基于这一套自回归/流匹配骨架，只是后缀的含义不同（前者是语言指令，后者是连续动作）。标准 VLA 是把动作离散化后塞进这个后缀里微调得到的。

### 2.2 分层推理

Hi Robot 把整体策略拆成高层和低层两个 VLM：

$$p^{\text{hi}}(\hat{\ell}_t \mid \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \ell_t), \qquad p^{\text{lo}}(\mathbf{A}_t \mid \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \hat{\ell}_t, \mathbf{q}_t)$$

高层策略读取图像和开放式任务提示 $\ell_t$（及历史用户交互上下文），输出一条低层能听懂的原子语言指令 $\hat{\ell}_t$（如"grasp the cup"），必要时附带一句要说给用户听的话 $u_t$（经 TTS 播放，随后从 $\hat{\ell}_t$ 中剔除，不进入低层）。低层策略与标准 VLA 输入形式相同，只是把 $\ell_t$ 换成了 $\hat{\ell}_t$；对简单熟悉任务，可直接令 $\hat{\ell}_t = \ell_t$ 退化为传统单层策略。

低层以更高频率运行（每步都要出动作 chunk），高层则以较低频率运行——具体策略是"每隔 1 秒重算一次 $\hat{\ell}_t$，或一旦收到新的用户交互立即重算"，这个简单规则在实现中被证明已经够用。用户可以随时用文字或语音（本地 Whisper large-v2 转写）打断任务，打断会立即触发高层重新推理；当用户说"leave it alone"表示放行后，机器人可以切回之前的指令继续执行。

### 2.3 数据采集与合成数据生成（关键创新点）

训练数据分三部分构建（对应论文 Figure 3）：

1. $\mathcal{D}_{\text{demo}}$：遥操作采集的完整任务轨迹，仅有粗粒度的整体目标标注（如"make a sandwich"）；
2. $\mathcal{D}_{\text{labeled}}$：把 $\mathcal{D}_{\text{demo}}$ 切分成 1–3 秒的短技能片段并打原子技能标签 $\hat{\ell}_t$（如"pick up one piece of lettuce"），另外用启发式方法额外抽取"move the right arm to the left"这类基础纠偏动作，得到 $(\hat{\ell}_t, \mathbf{I}_t^1,\ldots,\mathbf{I}_t^n)$ 元组；
3. $\mathcal{D}_{\text{syn}}$（合成交互数据）：直接用人工标注的原子技能训练高层策略，覆盖不了开放式复杂提示和用户打断这种长尾语言现象。因此论文用一个独立的大型数据生成 VLM $p^{\text{gen}}$，给定观测和技能标签 $\hat{\ell}_t$，反推"可能导致这个技能标签的用户提示或用户打断" $\ell_t$，以及机器人应给出的口头回应 $u_t$（例如从"pick up the lettuce"反推出用户提示"Can you add some lettuce for me?"）。生成过程按场景类型（negative task 禁止做什么、situated correction 情境纠正、specific constraint 特定约束，如饮食限制）和回应类型（简单确认、澄清、报错处理）做结构化分类，并对同一 episode 内先前的技能标签序列 $\hat{\ell}_0,\ldots,\hat{\ell}_{t-1}$ 做条件生成，保证多步任务里生成的提示前后逻辑一致。

高层策略在 $\mathcal{D}_{\text{syn}} \cup \mathcal{D}_{\text{labeled}}$ 上用标准下一词预测交叉熵训练；低层策略在 $\mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{demo}}$ 上用 flow-matching 目标训练（沿用 π0 的做法）。

用大白话说：人工标注只给出了"做什么动作"，但没有给出"用户当时可能怎么说、机器人该怎么回话"；论文的思路是让一个更强的 VLM"倒推剧本"——看着技能标签和画面去脑补一段合理的人机对话，从而把稀缺的复杂语言监督规模化。

### 2.4 模型与实现

高层、低层都以 PaliGemma-3B（Beyer et al., 2024）为初始化骨架；低层即 π0 VLA（Black et al., 2024），在 PaliGemma-3B 基础上加一个用 flow-matching 输出连续动作的 "action expert"；高层则微调用于预测离散语言指令（含机器人口头应答）。两个模型架构几乎一致，唯一区别是低层多了 flow-matching 头。语音方面：本地 Whisper large-v2 做语音转文字，Cartesia API 做文字转语音。

## 三、实验结果

评测覆盖三个机器人平台、三个任务域，每个 (任务, 方法) 组合 20 次试验，由不知晓所测方法的人类评估员打分。

**平台**：UR5e 单臂（6-DoF + 平行夹爪，7 维状态/动作）；Bimanual ARX 双臂（两条 6-DoF ARX 臂，14 维状态/动作）；Mobile ARX 移动双臂（基于 Mobile ALOHA，非完整移动底盘 +2 维，14 维状态、16 维动作）。

**任务域**：Table Bussing（清桌子，需区分垃圾/餐具，甚至用一个物体处理另一个物体，如倾倒盘中垃圾）；Sandwich Making（做三明治，涉及可变形食材的精细抓取）；Grocery Shopping（超市购物，双臂移动平台，从货架取物放入购物篮）。

**对比方法**：

- Expert human high-level（oracle）：人类专家实时给出低层指令，衡量低层策略的能力上限；
- GPT-4o high-level：用 GPT-4o 替代高层 VLM（结构与 Hi Robot 相同），提示词经过工程化以对齐低层可执行技能列表；
- Flat VLA：不含高层、不含合成数据的 π0 低层策略直接吃复杂提示（代表现有 SOTA 端到端指令跟随基线）；
- Flat VLA w/ synthetic data：单层策略但训练时加入合成数据；
- Hi Robot w/o synthetic data：完整分层结构，但高层只用人工标注数据训练（对应一个更强的 VLM 版 YAY Robot）。

**指标**：Instruction Accuracy（IA，高层预测指令是否与用户意图/当前观测一致）与 Task Progress（TP，任务完成度，按正确摆放物体比例衡量）。

| 对比维度 | 结论（来自 Figure 5/7/8） |
|---|---|
| Hi Robot vs. GPT-4o 高层 | Hi Robot 三任务域平均 IA 比 GPT-4o 高 40% 以上；GPT-4o 常在物理交互开始后丢失上下文，给出不合理指令（如"pick up bermuda triangle"）或把所有物体都标成"plate/spoon" |
| Hi Robot vs. Flat VLA | Hi Robot 全面超过无高层的 flat 策略，能正确执行"只清理垃圾不动餐具""不要番茄"等需要情境判断的部分指令 |
| Hi Robot vs. Expert human high-level（oracle） | 人类高层指令下低层策略几乎完美执行，说明失败更多来自"高层推理"而非"低层执行"，Hi Robot 的高层 VLM 部分弥合了这一差距 |
| 消融 A：有/无合成数据（Figure 7） | 平均 gap：IA 差 46%，TP 差 34%（原文标注为 IA 46% / TP 39%，注：图中两个数字分别标于 IA、TP 柱上）；无合成数据时模型忽略用户澄清与约束（如过敏、局部指令），加入合成数据后能顺畅处理组合式语言 |
| 消融 B：分层 vs. flat 策略（Figure 8） | 同样使用合成数据训练，分层结构相对 flat 策略平均 gap：IA 差 29%，TP 差 34%；flat 策略在处理"只收黄色的东西"这类中途澄清/部分指令时经常退化为清空所有物品 |

**推理效率**（RTX 4090 消费级 GPU 单/双卡实测，附录 B.3）：低层每步推理约 73ms（板载）/ 86ms（含 WiFi 传输），可支撑约 10Hz 控制频率，结合 action chunking 可等效控制到 50Hz；高层单次解码在 RTX 4090 上 47ms（prefill）+13.2ms（decode），H100 上 17.3ms+5.7ms。高层策略训练仅需 8×H100 约 2 小时。

## 四、局限性

论文在第 6 节与附录 C.4 明确列出的局限：

- **缺乏长上下文记忆**：高层策略目前不具备跨长时程的记忆能力，难以处理需要长上下文推理的指令；
- **高层训练依赖提示工程**：用于生成合成训练数据的 prompt 需要人工设计才能诱导出所需的对话/纠正行为；
- **高低层解耦、互不感知**：训练过程完全解耦，两层模型除了通过训练样本外并不知道彼此的能力边界（如高层不知道低层是否真的完成了某个子指令）；
- **低层失败模式**：（1）暂时性忽略指令，例如用户说"我对乳糖不耐受"但机器人仍因训练数据里对近处物体的偏好而抓取奶酪；（2）误差累积与分布外（OOD）恢复能力弱，如物体掉落后难以恢复；
- **高层运行频率固定**（1 秒或新交互触发），并非真正自适应/异步的多层级处理；
- 两个 VLM 架构几乎相同（仅低层多 flow-matching 头），论文自陈这种"物理分离"并非必要设计，只是当前的工程实现选择。

## 五、评价与展望

**优点**：Hi Robot 的核心贡献不在于提出全新的低层 VLA（沿用现成的 π0），而在于（1）给"高层推理 + 低层执行"的分层范式一个干净、可复现的 VLM-VLM 实例化，两层用几乎相同的骨架和训练配方，工程上简洁；（2）提出了用生成式 VLM 反推"技能标签 → 合理用户交互"的合成数据方案，直接解决了复杂语言监督稀缺的数据瓶颈——这一思路具有较强的通用性，可能被后续工作用于其他"动作标注丰富但语言标注稀缺"的场景；（3）消融实验设计干净，同时隔离了"分层结构"和"合成数据"两个变量各自的贡献，说明二者缺一不可,而不是把功劳简单归于更大的模型或更多数据。

**局限与开放问题**：（1）对比基线中的 GPT-4o 只是"开箱即用+提示工程"，并未针对该机器人具身做微调，因此"Hi Robot 领先 GPT-4o 40%"更多说明了**领域内微调 VLM** 对**通用闭源 VLM 直接调用**的优势，而非严格意义上分层设计本身的优势——若换成同等规模的微调过 GPT-4o/大型开源 VLM 做高层，差距可能缩小；（2）高低层解耦训练、异步频率固定，是论文自己承认的简化设计，后续若想做更细粒度的双层协同（如高层感知低层执行进度、失败时主动重规划）仍是开放问题，这也是该方向后续工作的核心生长点；（3）三个任务域均由 Physical Intelligence 自建的真实机器人平台采集，尚未在公开 benchmark（如 RoboTwin、LIBERO 等仿真/标准基准）上验证,跨实验室复现难度较大；（4）与同期"情境纠正"相关工作横向看，YAY Robot（Shi et al., 2024）只支持单一提示+纠正、RACER（Dai et al., 2024）依赖物理仿真器构造恢复行为，Hi Robot 用纯真实数据 + 合成语言监督覆盖了更开放的提示空间，是该子方向上覆盖面较广的一版工作，但仍未解决长时程记忆和真正的失败自恢复问题。整体而言，Hi Robot 更像是把"VLM 做高层任务分解 + VLA 做低层执行"这一已有思路，通过合成数据这一杠杆做扎实、做规模化的工程验证，而非方法论上的根本突破。

## 参考

- Black, K. et al. π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164, 2024.（低层策略的基座 VLA）
- Beyer, L. et al. PaliGemma: A Versatile 3B VLM for Transfer. arXiv:2407.07726, 2024.（高低层共同的 VLM 骨架）
- Brohan, A. et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan). CoRL 2023.（对比的"VLM/LLM 高层 + 固定技能库"范式代表）
- Shi, L. X. et al. Yell at Your Robot: Improving On-the-Fly from Language Corrections (YAY Robot). arXiv:2403.12910, 2024.（同一作者团队的前作，仅支持单一提示与纠正）
- Dai, Y. et al. RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning. arXiv:2409.14674, 2024.（依赖仿真器构造纠正数据的对比工作）
