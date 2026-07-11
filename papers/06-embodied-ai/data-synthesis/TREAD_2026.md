# TREAD：通过重标注视觉-动作机器人数据实现任务鲁棒性

> **论文**：*Task Robustness via Re-Labelling Vision-Action Robot Data*
>
> **作者**：Artur Kuramshin, Özgür Aslan, Cyrus Neary, Glen Berseth
>
> **机构**：Mila — Quebec AI Institute；Université de Montréal；The University of British Columbia
>
> **发布时间**：2026 年 06 月（arXiv 2606.10918）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.10918) | [PDF](https://arxiv.org/pdf/2606.10918)
>
> **分类标签**：`VLM重标注` `轨迹分解` `语言增广` `VLA数据增广`

---

## 一句话总结

TREAD 用一个现成的大型 VLM（Gemini 2.5 Pro）对已有机器人数据集做"零采集"再标注：先把长程演示切成子任务段、再为每段生成视觉接地的多样化指令，从而同时补齐数据集在**轨迹粒度**和**语言表述**两个维度上的多样性；在 LIBERO 上微调 Octo 与 π0-FAST 后,新任务/新指令的成功率有明显提升（如 Octo 的 Motion Generalization 单目标任务 7%→22%,Language Generalization 单目标任务 82%→91%）。

## 一、问题与动机

近年机器人操作策略靠"数据规模+多样性"取得了很强的泛化能力,但普遍存在一个短板：**指令跟随（instruction following）不可靠**。作者把根因归结为现有机器人数据集在两个模态上的多样性不足：

- **语言多样性稀释**：BridgeV2、DROID、RDT-1B 等新数据集虽然任务更复杂、场景更多,但单条指令往往对应一整段越来越长的多子目标轨迹。于是"每条指令覆盖的图像-动作帧数"被拉大,语言标注相对图像-动作数据被"稀释",策略容易对指令措辞过度敏感。
- **轨迹粒度单一**：长程演示内部其实包含很多有意义的子目标(抓、抬、移、放……),但整段只挂一条高层指令,子技能层面的 language-action 配对没有被显式暴露给策略。

人工重看几千段演示视频、雇标注员做不到与现代机器人学习系统同步扩展。于是核心问题是：**如何以可扩展的方式,在不采集新数据的前提下,给已有数据集补上 language-action 多样性?** 作者的观察是:互联网规模预训练的 VLM 天然具备(a) 零样本生成场景接地的语言标签、(b) 对视频做时序推理并切分子任务两种能力,恰好能填这个缺口。

## 二、核心方法

**问题形式化。** 给定离线数据集 $\mathcal{D} = [(\tau_n, \ell_n)]_{n=1}^{N}$,其中每条轨迹 $\tau_n = [(o_t^n, a_t^n)]_{t=1}^{T}$ 是观测-动作序列,$\ell_n$ 是自然语言指令。策略通过行为克隆学习:

$$\hat{\pi}^{*} = \arg\min_{\pi} \sum_{(\tau,\ell)\in\mathcal{D}} \sum_{(o_t,a_t)\in\tau} \mathcal{L}_{BC}\big(\pi(\cdot \mid o_t, \ell),\, a_t\big)$$

> **用大白话说**：就是最标准的模仿学习——给定当前画面 $o_t$ 和指令 $\ell$,让策略预测的动作分布尽量贴近人类演示的动作 $a_t$。TREAD 完全不碰这个训练目标,只在数据侧 $\mathcal{D}$ 上做文章。

TREAD 通过对同一个 VLM $\mathcal{G}$ 做**迭代式链式查询**(上一步输出喂给下一步),分三个阶段扩充数据:

**阶段 1：子任务分解(Subtask Decomposition)。** 给 VLM 输入高层指令 $\ell_n$ 和轨迹首帧 $o_1^n$(用来接地场景),让它规划出一串子任务动作标签 $[\tilde\ell_n^1, \tilde\ell_n^2, \ldots, \tilde\ell_n^{z_n}]$。这里 $z_n$ 由 VLM 自行判断——例如"把碗放进顶层抽屉",抽屉是关着还是开着会决定该拆成 2 步还是 3 步。

> **用大白话说**：先让 VLM 当"规划器",看一眼开局画面+任务描述,写出完成任务需要哪几个动作步骤。

**阶段 2：动作分割(Motion Segmentation)。** 把整段轨迹视频 $v=[o_1,\ldots,o_T]$ 连同上一步生成的子任务列表一起喂给 VLM,让它标出每个子任务在视频中的**起止秒数**,从而把整段轨迹切成一组有序的带标注子轨迹 $[(\tilde\tau^1,\tilde\ell^1),\ldots,(\tilde\tau^z,\tilde\ell^z)]$,其中 $\tilde\tau \subset \tau$。作者发现:直接让 VLM 一次性(one-shot)切视频效果很差;**先给出子任务列表再让它对齐时间轴**,分割质量明显更好。分割时还用了 3 个来自 LIBERO-90 的样例做 few-shot 提示,并要求 VLM 先在图上点出不超过 8 个物体(以 $[y,x]$ 归一化坐标输出)再判定动作边界。由此得到分解后的数据集:

$$\tilde{\mathcal{D}}_A = [(\tilde\tau_1^1,\tilde\ell_1^1),\ldots,(\tilde\tau_1^{z_1},\tilde\ell_1^{z_1}),\ldots,(\tilde\tau_n^1,\tilde\ell_n^1),\ldots,(\tilde\tau_n^{z_n},\tilde\ell_n^{z_n})]$$

> **用大白话说**：拿着"步骤清单"去视频里对表,标出每一步从第几秒到第几秒,把一整段长演示切成若干"短技能片段+短指令"。关键工程 trick:别让模型盲切,给它清单它才切得准。

**阶段 3：语言多样性(Grounded Textual Diversity)。** 对每个子轨迹 $(\tilde\tau^i,\tilde\ell^i)$,把它的指令 $\tilde\ell^i$ 和该片段首帧一起给 VLM,要求生成 $k$ 条保持语义不变的改写。因为上下文里带了图像,VLM 能把**物体属性**(颜色/形状/材质)和**空间关系**(相对位置/朝向)写进指令,而不只是同义词替换。例如"pick up the blue coffee mug"可被改写为"grasp the small coffee mug"(属性)或"retrieve the coffee mug next to the laptop"(空间关系)。得到语言增强后的数据集:

$$\tilde{\mathcal{D}}_H = [(\tilde\tau_1^1,\hat\ell_1^1),\ldots,(\tilde\tau_1^{z_1},\hat\ell_1^{z_1}),\ldots,(\tilde\tau_n^1,\hat\ell_n^1),\ldots,(\tilde\tau_n^{z_n},\hat\ell_n^{z_n})]$$

> **用大白话说**：给每个短片段的指令换 $k$ 种说法,而且是"看着画面"换——让模型学会同一个动作可以有很多种自然语言表达,降低对措辞的过拟合。

**整体流程(Algorithm 1)。** `DECOMPOSE` 产出 $\tilde{\mathcal{D}}_A$,`DIVERSIFY` 在其上产出 $\tilde{\mathcal{D}}_H$;然后分别训练两个策略:$\pi_A$ 在 $\mathcal{D}$ 与 $\tilde{\mathcal{D}}_A$ 的混合上训练,$\pi_H$ 在 $\mathcal{D}$ 与 $\tilde{\mathcal{D}}_H$ 的混合上训练。原始整段数据始终保留在混合里——增广是"加料"而非"替换"。

**混合比例。** 原始整段 $\mathcal{D}$ 与增广子轨迹 $\tilde{\mathcal{D}}_A/\tilde{\mathcal{D}}_H$ 的配比,试了 1:2、1:1.5、1:1.1 三档,最终用 Re-Mix 自动选比例,每种数据组合各取其最优配比(Appendix II 显示结果对配比相当敏感)。

**训练设置。** 微调两个通用策略:Octo-Small 1.5(基于 diffusion、在 Open-X 上预训练)与 π0-FAST(基于 PaliGemma 骨干、在 π0 跨本体数据混合上训练、动作用 FAST tokenization 离散化)。Octo 训 50k 步、batch 256、峰值 lr $3\times10^{-4}$、cosine 衰减,取 30k 步 checkpoint;π0-FAST 全量微调 30k 步、batch 32、峰值 lr $2.5\times10^{-5}$,取 15k 步 checkpoint。

## 三、实验结果

**数据规模。** 在 LIBERO-100(100 个任务,每任务 50 条人工遥操作演示)上,因算力受限只标了**每任务 5 条演示、去掉所有 STUDY_SCENE,共 570 条轨迹**;实践中把 82 个"场景-指令"任务分解为 **146 个子任务**。评测用了**完整** LIBERO-100(比近期工作常用的更小子集更难,减少过拟合风险)。

**两类自建评测集**(均为 30 次 rollout 取平均成功率,并区分单目标 / 双目标,双目标另报"1 of 2 部分成功"与"2 of 2 完全成功"):

- **Motion Generalization (MG)**:7 个新"指令-场景"组合——把已有指令配到训练里没出现过的兼容场景(如"open the top drawer of the cabinet"配 KITCHEN_SCENE4),考察子任务分解是否帮助技能迁移到新环境;同时也考语言跟随(模型不能凭场景默认执行旧任务)。
- **Language Generalization (LG)**:14 个把 LIBERO-100 指令做措辞改写的任务——如围绕连词重排子句、或在上下文冗余时删掉物体修饰词,考察语言增广是否提升对措辞变化的鲁棒性。

**主结果(Table I,成功率 % ± 标准误;粗体=最优,下划线原文=次优)**:

| 测试集 | 指标 | π0-FAST 原始 | π0-FAST w/o div. | π0-FAST TREAD | Octo 原始 | Octo w/o div. | Octo TREAD |
|---|---|---|---|---|---|---|---|
| Language Gen. | Single Goal SR | 47±20 | **77±7** | 67±14 | 82±2 | 76±17 | **91±6** |
| Language Gen. | 1 of 2 SR | 63±11 | 62±10 | **67±9** | 76±6 | 70±6 | **77±4** |
| Language Gen. | 2 of 2 SR | 36±8 | 35±7 | **39±10** | 30±7 | 28±5 | **31±5** |
| Motion Gen. | Single Goal SR | 28±18 | 31±20 | **34±16** | 7±4 | **27±10** | 22±15 |
| Motion Gen. | 1 of 2 SR | 73±3 | **82±11** | 82±9 | 13±4 | **50±10** | 43±3 |
| Motion Gen. | 2 of 2 SR | **7±3** | 0±0 | 0±0 | 0±0 | 0±0 | 0±0 |
| LIBERO-10 | 1 of 2 SR | **83±9** | 72±12 | 74±8 | **76±5** | 69±6 | 72±4 |
| LIBERO-10 | 2 of 2 SR | 57±10 | 43±12 | **57±10** | 40±7 | **41±4** | 38±4 |
| 平均 | SR | 49±10 | 50±11 | **53±10** | 41±12 | 45±10 | **47±11** |

**关键读数与结论：**

- **轨迹分解带来 Motion 泛化增益**:MG 单目标,TREAD 相对原始微调在 Octo 上 7%→22%(+15pt)、π0-FAST 上 28%→34%(+6pt);MG 双目标"首目标完成率(1 of 2)"提升更明显——Octo 13%→43%(+30pt)、π0-FAST 73%→82%(+9pt)。作者归因于多步任务里训练/测试子轨迹重叠机会更多。
- **对无 VLM 预训练的策略增益更大**:Octo(无内建多模态理解)几乎处处收益大于 π0-FAST,说明轨迹分解为"没有内建多模态推理"的模型提供了尤其有价值的归纳偏置。
- **语言增广提升措辞鲁棒性**:LG 单目标,Octo 82%→91%、π0-FAST 47%→67%(注:论文正文把两者的提升幅度 9% 与 20% 与模型名对调了,以表格数字为准)。语言增益主要体现在**短的单目标任务**,多步长程任务上不明显——作者认为是子轨迹指令偏短、与多目标长指令风格失配,且长程更易累积误差。
- **不牺牲同分布性能**:在同分布长程 LIBERO-10 上,TREAD 的双目标(2 of 2)成功率与原始微调持平(π0-FAST 57% 对 57%,Octo 38% 对 40%)。
- **平均成功率**:TREAD 均为最高(π0-FAST 53% 对 49%,Octo 47% 对 41%)。

## 四、局限性

- **依赖闭源大 VLM**:方法用 Gemini 2.5 Pro,作者自述这限制了可复现性;是否能换开源 VLM(甚至在机器人数据上微调开源 VLM 以增强具身理解)是明确的后续方向。
- **数据/评测规模小**:因算力受限只标了 570 条轨迹、评测集仅 7(MG)+14(LG)个自建任务,标准误极大(多处 ±15~±20),部分结论统计置信度有限。
- **双目标"完全成功"几乎全崩**:MG 的 2 of 2 一栏除 π0-FAST 原始的 7% 外全部为 0,即分解增广并未真正解决多步任务的端到端完成。
- **增广并非单调有益**:π0-FAST 的 LG 单目标上,只做分解(w/o div.,77%)反而优于加了语言多样性的完整 TREAD(67%);LIBERO-10 的 1 of 2 上增广也拉低了成绩,说明语言增广有时会引入分布干扰,且结果对混合比例敏感(需 Re-Mix 挑比例)。
- **正文数字与表格有对不上之处**:LG 单目标提升幅度的模型归属在正文里疑似写反,反映校对不够严谨。
- **缺与相关方法的实证对比**:与 DIAL / NILS / SPRINT 只做了概念区分,没有在同一 benchmark 上跑对照实验。

## 五、评价与展望

**优点。** (1) **定位清晰、即插即用**:纯数据侧增广,不改策略架构、不改训练目标,理论上可叠加到任意 VLA 上,工程价值明确。(2) **"分解"这一方向选得对**:与 SPRINT 把短技能拼成长序列相反,TREAD 把长演示拆成短技能——这正切中"长轨迹稀释语言标注"这一当下大数据集的真实痛点,把隐含的子技能 language-action 配对显式暴露出来。(3) **相比 NILS 更简洁**:NILS 靠多个大模型+启发式规则做分割,TREAD 只用单个 VLM 迭代查询、无手工启发式,复现门槛更低;相比 DIAL 依赖预定标签集+机器人数据微调,TREAD 走零样本路线更通用。(4) **视觉接地的改写**是亮点:把首帧塞进上下文,让改写带上颜色/材质/空间关系而非纯同义词,这类"grounded paraphrase"比无约束语言增广更可能保持任务语义一致。(5) 一个务实的工程发现——**"先给子任务清单再让 VLM 对齐时间轴"远好于一次性切视频**——对做视频时序标注的同类工作有直接参考价值。

**不足与开放问题。** 最核心的软肋是**证据强度**:样本量小、方差大、双目标完全成功全为 0,且增广偶尔反而掉点,使"TREAD 稳定有效"的结论说服力受限;若能扩到 LIBERO 全量或真实机器人数据、并补上与 DIAL/NILS/SPRINT 的同台对照,结论会硬得多。其次是**VLM 时序分割的可靠性无量化评估**——起止秒数标得准不准、切错段会不会污染训练数据,论文只给了定性可视化(Figure 3),没有分割精度指标。第三,**闭源 VLM 依赖**既是复现障碍也是成本问题,能否用开源视频 VLM(Qwen2.5-VL、VideoLLaMA 3 等,论文也点到)达到同等分割质量是关键的可落地性问题。开放方向还包括:把该框架应用到更大数据集/更大策略上验证 scaling;研究语言增广"何时有益何时有害"的边界(本文已观察到不单调,但未给机制解释);以及探索比"取一条改写"更充分的多样性利用方式(如把 $k$ 条改写全部用上而非随机取一条)。

**与公开工作的关系。** TREAD 处在"用基础模型自动重标注机器人数据"这一活跃方向的交叉点上:与 DIAL(Xiao et al., RSS 2023)、NILS(Blank et al., CoRL 2024)同属 VLM 重标注,与 SPRINT(Zhang et al., ICRA 2024)在"分解 vs. 组合"上互补,评测底座沿用 LIBERO(Liu et al., NeurIPS 2023)+ Octo / π0-FAST。整体是一篇思路清爽、落点明确但实验规模偏小的工作,方法论上的启发(视觉接地改写 + 清单引导时序分割)大于其当前数字所能支撑的结论。

## 参考

1. Ted Xiao et al. *Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models* (DIAL). RSS 2023. — 最相近的语言重标注工作,依赖预定标签集与机器人数据微调。
2. Nils Blank et al. *Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models* (NILS). CoRL 2024. — 多模型+启发式做分割/重标注,TREAD 的简化对照。
3. Jesse Zhang et al. *SPRINT: Scalable Policy Pre-training via Language Instruction Relabeling*. ICRA 2024. — 与本文相反,把短技能组合成长序列。
4. Octo Model Team. *Octo: An Open-Source Generalist Robot Policy*. RSS 2024. — 被增广数据微调的策略之一。
5. Karl Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models* (π0-FAST). arXiv 2501.09747, 2025. — 另一被微调的策略,带 FAST 动作离散化。
