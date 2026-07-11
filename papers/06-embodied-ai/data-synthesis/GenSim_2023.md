# GenSim：用大语言模型生成机器人仿真任务

> **论文**：*GenSim: Generating Robotic Simulation Tasks via Large Language Models*
>
> **作者**：Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, Xiaolong Wang（et al.）
>
> **机构**：MIT CSAIL；UC San Diego；Shanghai Jiao Tong University；Tsinghua University；University of Washington；CMU
>
> **发布时间**：2023 年 10 月（arXiv 2310.01361，v2 2024 年 1 月）
>
> **发表状态**：ICLR 2024（Published as a conference paper at ICLR 2024）
>
> 🔗 [arXiv](https://arxiv.org/abs/2310.01361) | [PDF](https://arxiv.org/pdf/2310.01361)
>
> **分类标签**：`仿真任务生成` `LLM代码合成` `任务级泛化` `sim-to-real`

---

## 一句话总结

GenSim 把 LLM 的语言推理与代码生成能力当成"任务工厂",通过 goal-directed 与 exploratory 两种 prompt 链、外部 task library（RAG + 反思 + 微调）和分级 pass-rate 校验,把 Ravens/CLIPort 的 10 个人工任务自举扩展到 100+ 仿真操作任务及其专家演示;在这些任务上联合训练多任务策略,可把域内泛化提升 50%+、对未见任务实现约 40% 的零样本迁移,并在真机上把 sim-to-real 长程任务成功率相对基线提升约 25%(70 任务预训练平均 62.5%)。

## 一、问题与动机

训练通用机器人策略需要海量交互数据,真机采集昂贵,因此转向仿真。但现有仿真数据生成方法主要在 **scene-level**（物体实例、姿态、域随机化）上做多样性,而在 **task-level**（任务本身的种类)上多样性很差——因为设计并验证一个新任务需要大量人力：要指定资产关系、任务进程,还要保证可完成性与可迁移性。结果是典型人工构建的仿真 benchmark 只有几十到上百个任务,策略难以展现任务级泛化。

作者提出的核心问题：**能否让 LLM 自动创造出丰富、可完成、又多样的仿真任务与专家演示,并把 LLM 的推理/代码能力"蒸馏"进低层视觉运动策略?** 关键观察是：在 Ravens 这类 benchmark 里,一个"任务"本质上就是一段初始化资产、定义空间与语言目标的 reset 代码;而写代码正是当代 LLM 的强项。于是把任务生成问题转化为受约束、可执行校验的程序合成问题。

## 二、核心方法

GenSim 框架由三部分组成：(1) Task Creator（prompt 机制,产出任务描述 + 代码实现);(2) Task Library（缓存高质量任务代码的外部记忆);(3) LLM-Supervised Multitask Policy（用生成的演示训练策略）。

**任务的形式化定义。** 本文里一个任务不是某个具体场景或轨迹,而是由代码（加语言模板)定义。可写作

$$
\tau = \langle d,\, c \rangle \;\xrightarrow{\text{仿真引擎}}\; \{(o_t,\, a_t,\, \ell)\}
$$

其中 $d$ 是自然语言任务描述、$c$ 是实现任务的 Python 代码（reset 函数初始化资产/姿态并调用 `add_goal` 设定空间与语言目标),执行后自动产出观测-动作-语言三元组的专家演示。用大白话说：LLM 只要写对一段"搭场景 + 设目标"的代码,仿真器就能自动跑出成千上万条带语言标注的示教数据。

**两种生成模式。** Task Creator 是一条 prompt 链,依据是否已知目标任务分两种模式：

- **Goal-directed（top-down)**：给定一个具体任务名,LLM 补全其描述与代码实现,面向"把某个指定任务写对"的编程能力。
- **Exploratory（bottom-up)**：不给目标,让 LLM 从已有任务自举,迭代提出与库中任务"足够不同"的新任务,面向创造力,目标是建立任务无关的基础策略。

两种模式都先生成任务描述(名称、资产、摘要),再用 few-shot 从 task library 检索参考任务代码作为示例来写实现。

**分级校验指标（关键设计)。** 生成的代码用一串递增的 pass-rate 逐级过滤,这既是评测指标也是反馈信号：

$$
P_{\text{syntax}} \;\ge\; P_{\text{runtime}} \;\ge\; P_{\text{task}}
$$

- $P_{\text{syntax}}$：syntax-correct,能否编译、格式是否正确;
- $P_{\text{runtime}}$：runtime-verified,资产 URDF 是否存在、代码推理是否跑得通（抓幻觉资产、断言错误);
- $P_{\text{task}}$：task-completed,pick-place 演示是否真的完成(环境成功率 $>5\%$)。

用大白话说：这三个指标是层层嵌套的漏斗,后一层通过必然要求前一层通过,失败在前一层就不用看后面。除代码执行反馈外,GenSim 还叠加了自反思（self-reflection)、策略训练成功率、以及人工检查等多种反馈形式。

**Task Library（外部记忆 + RAG + 反思)。** 库从人工 benchmark 的任务初始化,一方面在描述阶段提供过去任务描述列表、在代码阶段提供过去代码列表,作为**检索增强生成（RAG)**的 few-shot 示例;另一方面在新任务成功产生演示后,让 LLM **反思**该任务是否与库中已有任务重复,再集成决定是否入库。这个库既能作为 exploratory 模式的自举数据,又能反过来当作**微调数据集**训练更经济的 task creator（把强模型能力蒸馏进小模型)。

**策略训练（蒸馏为视觉运动策略)。** 沿用 CLIPort/Transporter 的双流网络：输入自顶向下的 RGB-D 重建,输出 affordance map,再转成 pick-and-place。可粗略写作

$$
\mathcal{T}_{\text{pick}} = \arg\max_{(u,v)} \; Q_{\text{pick}}\big((u,v) \mid o_t,\, \ell\big)
$$

place 位置由以 pick 邻域为模板与全图做语言条件互相关得到。用大白话说：把 LLM 生成的一大批任务变成海量带语言指令的示教,再用一个语言条件的抓放网络去拟合,LLM 的编码/推理能力就通过演示数据"流"进了低层策略。

## 三、实验结果

评测在 Ravens benchmark（0–100 分含部分成功的打分)上进行;仿真机器人为 UR5e + 吸盘,真机为 XArm-7 + 吸盘 + 俯视相机。库以 10 个人工任务初始化,GPT-4 扩展到 100+ 任务(共生成约 120 个)。

**任务生成质量（exploratory 模式,Figure 6 左)。** prompt 链 + task library 的效果显著优于单 prompt 与 zero-shot：

| 指标 | ours(链+库) | single-prompt | zero-shot |
| --- | --- | --- | --- |
| syntax-correct | 0.99 | 0.98 | 0.99 |
| runtime-verified | 0.86 | 0.68 | 0.60 |
| task-completed | 0.48 | 0.20 | 0.12 |

**Goal-directed 与模型对比（Figure 6 右)。** 闭源 GPT-4 在任务编码上仍领先其它基座模型;而在 GPT-4 生成的 100 个任务上**微调**可大幅提升较弱模型：闭源 GPT-3.5 与开源 Code-LLaMA-Instruct-13B 微调后 runtime-verified / task-completed 明显上升(如微调后 runtime-verified 可达 0.87–0.90),开源模型经微调可逼近 SOTA 表现但仍偶有高层目标与实现不一致。

**策略泛化（Figure 7)。**

| 设置 | 观察 |
| --- | --- |
| 少样本联合训练(单 CLIPort 任务 + N 个 GPT 任务) | 成功率随任务数从 16 升到 59,原任务性能提升 50%+(尤其 5 demos 低数据区) |
| 零样本迁移(在更多 GPT 任务上预训练) | 5/10/20/50/70 任务 → 成功率 10/16/28/34/39,约 40% 零样本迁移到未见任务 |
| 不同任务来源 | CLIPort 任务 28,GPT 任务 32,Code-LLaMA 任务 34——开源模型生成的任务同样有助益 |

**Sim-to-real（Table 1,XArm-7,12 个任务各 10 次)。** 少量真机数据 + 数据增强后微调仿真预训练模型：

| 方法 | 平均成功率 |
| --- | --- |
| No Adaptation | 0% |
| No Pretrain | 27.5% |
| CLIPort 预训练 | 39.2% |
| GenSim(50 任务)预训练 | 46.7% |
| GenSim(70 任务)预训练 | **62.5%** |

即 70 任务预训练相对只用 CLIPort 任务的基线提升 20%+、相对 50 任务提升 15%,长程任务(如 build-wheel)鲁棒性更好。

**其它数据点。** 仿真内策略训练：GPT-4 生成任务单任务成功率 75.8% / 多任务 74.1%,与人工 CLIPort 任务(76.6% / 76.1%)相当,说明生成任务本身可训练性不逊于人工。人工验证平均每任务约 10 秒、通过率 50%+。

## 四、局限性

- **代码质量仍不完美**：生成代码仍存在基本语法错误、资产幻觉,以及对物理/几何细节缺乏 grounding。
- **评测指标不完备**：pass-rate 无法捕捉"语言描述与实际实现不匹配"这类语义错误,因此策略训练前仍需一定程度的人工过滤。
- **任务分布偏斜**：自举导致 LLM 偏向库中多数类(如 color-coordinated、pick-place),涉及绳索/堆叠的复杂任务偏少;还存在重复任务、噪声描述、不可达动作序列等失败模式(附录 A.5 列举六类)。
- **任务空间受限**：只探索了自顶向下、可由抓放两个末端位姿参数化的桌面任务;更高自由度、接触丰富的灵巧操作及自动奖励设计尚未覆盖。
- **强依赖闭源强模型**：高质量生成主要靠 GPT-4,开源模型需微调且仍有差距,规模化成本与可复现性受制约。

## 五、评价与展望

**优点。** (1) 视角新颖且落点务实：把"任务级多样性"这一被 scene-level 数据增强忽略的维度,转化为可执行、可校验的程序合成问题,契合 LLM 的强项;(2) 分级 pass-rate + task library(RAG + 反思 + 微调)构成一个自洽的"生成-校验-记忆-蒸馏"闭环,是本文最可迁移的工程设计;(3) 实证链条完整,从任务生成质量、仿真训练、任务级泛化一直打通到真机 sim-to-real,并给出正向 scaling 趋势(任务越多泛化越强),这对"用合成任务扩数据来做基础策略"的路线是有力支撑;(4) 开源代码/任务库/权重,复现与二次开发门槛低。

**缺点与开放问题。** (1) 强绑定 Ravens/CLIPort 的自顶向下抓放动作原语,"任务"实为一段 reset 代码,与真正需要接触动力学、灵巧手、长程闭环的操作仍有本质差距,泛化结论能否外推到更一般的 VLA 设定存疑;(2) 语义正确性缺乏自动校验,人工过滤仍不可省,规模化的可信度受限;(3) 生成多样性受库自举偏置影响,长尾任务稀缺,缺少显式的 novelty/coverage 驱动。

**与其它公开工作的关系。** 思路上与 exploratory 开放式生成(POET、OMNI、Voyager)一脉相承,但落在机器人仿真任务的程序合成;与 RoboGen、Gen2Sim、以及后续用生成式世界模型/视频扩散直接合成像素级演示的数据增强路线形成互补——GenSim 走"生成任务代码 → 仿真器渲演示"的符号化路径,可完全 grounding 于物理引擎、易于清洗与检索,但受限于仿真资产与动作原语;像素级生成路径视觉更真实但物理一致性与可控性更难保证。一个自然的改进方向是引入更强的语义一致性校验器(如 VLM critic)取代人工过滤,并把动作空间从抓放拓展到 6-DoF/接触丰富原语,同时用显式的覆盖度/新颖度目标缓解自举偏置。

## 参考

1. Zeng et al., *Transporter Networks: Rearranging the Visual World for Robotic Manipulation*, CoRL 2021 —— Ravens benchmark 与抓放策略基础。
2. Shridhar et al., *CLIPort: What and Where Pathways for Robotic Manipulation*, CoRL 2022 —— 语言条件双流 affordance 策略,GenSim 的策略架构与初始任务来源。
3. Rozière et al., *Code Llama: Open Foundation Models for Code*, 2023 —— 开源代码模型,GenSim 微调的开源基座。
4. Wang et al., *Voyager: An Open-Ended Embodied Agent with LLMs*, 2023 —— LLM 驱动的开放式技能自举,exploratory 生成的相关工作。
5. Wang et al., *Self-Instruct: Aligning LM with Self-Generated Instructions*, 2022 —— 用模型自生成数据做指令微调,与本文任务库自举/蒸馏思路相关。
