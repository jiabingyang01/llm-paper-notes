# ROSIE：用语义想象的经验扩展机器人学习

> **论文**：*Scaling Robot Learning with Semantically Imagined Experience*
>
> **作者**：Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Dee M, Jodilyn Peralta, Brian Ichter, Karol Hausman, Fei Xia
>
> **机构**：Robotics at Google；Google Research
>
> **发布时间**：2023 年 02 月（arXiv 2302.11550）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2302.11550) | [PDF](https://arxiv.org/pdf/2302.11550)
>
> **分类标签**：`生成式数据增广` `扩散模型 inpainting` `机器人操作/VLA` `real-to-real 增广`

---

## 一句话总结

ROSIE 把现成的文本到图像扩散模型（Imagen Editor）当作机器人数据增广引擎：用开放词表分割自动圈出可编辑区域、用 LLM 自动生成增广文本，再在真机轨迹的**每一帧**上做语义一致的 inpainting，从而不采一条新真机数据就凭空造出全新物体、容器、背景与干扰物；在 RT-1 上微调后，pick up 全新可变形物体的成功率相对基线提升 150% 以上，place 到未见容器提升至少 75%，把物体放进从未采过数据的水槽从 0% 提到 60%。

## 一、问题与动机

机器人操作策略要泛化，核心瓶颈是**真机数据的多样性**——不仅要覆盖大量运动技能，还要覆盖大量物体和视觉域。作者引用的对比很直观：RT-1 用 17 个月、13 台机器人才采到 13 万条示范；MT-Opt 用 7 台机器人、16 个月采了 80 万条自主 episode。要在物体/场景维度继续扩数据，要么靠工程量巨大的脚本策略，要么靠费力的人类遥操，成本都极高。

传统计算机视觉里的数据增广（裁剪、翻转、加噪、调色）只能做低层像素扰动，无法给机器人带来"语义上全新"的经验（新技能、新物体、新环境）。作者的核心洞察是：internet 级预训练的文本到图像扩散模型（DALL-E 2 / Imagen / Stable Diffusion）恰好提供了三种传统增广给不了的能力——(1) 通过自然语言接口对任务做**语义级**增广；(2) 天然具备海量物体/背景的照片级零样本生成；(3) 借助 inpainting 只改图像的局部区域、保留其余内容。于是可以在真机数据上"就地"替换物体、加入新干扰物、改换背景，把生成模型的世界知识"蒸馏"进机器人经验——作者称之为一次"免费午餐"（free lunch）。

与两个并发工作的定位差异（作者明确对比）：CACTI 用 inpainting 加干扰物，但需要**人工提供 mask 和语义标签**；GenAug 用**深度引导**的扩散模型生成新物体/新任务，需要人工指定 mask 和物体网格。ROSIE 则**不需要深度**、**自动**用文本引导选择 inpainting 区域，且同时能生成新干扰物与新任务。

## 二、核心方法

给定一条数据集 $\mathcal{D} := \{e_j\}_{j=1}^N$，每条 episode 是状态-动作-语言指令三元组序列

$$e = \{(o_i, a_i, o_{i+1}, \ell)\}_{i=1}^T$$

其中 $o$ 为图像观测、$a$ 为动作、$\ell$ 为标识目标任务的语言指令。策略 $\pi(\cdot \mid o_i, \ell)$ 通过 behavioral cloning（最小化动作的负对数似然）学习：

$$\min_{\pi}\; \mathbb{E}_{(o_i, a_i, \ell)\sim\mathcal{D}}\big[-\log \pi(a_i \mid o_i, \ell)\big]$$

> 用大白话说：策略就是"看到这张图、听到这句话，模仿人给的动作"。ROSIE 不碰动作 $a_i$，只在观测 $o_i$ 和指令 $\ell$ 上动手脚，把同一段动作"贴"到一个看起来全新的场景里。

ROSIE 是一条自动化流水线，分四步：

**1）开放词表分割定位增广区域（Sec 4.1）。** 用 OWL-ViT 开放词表检测器 + 一个额外的实例分割 mask 头（冻结 OWL-ViT 主干，在 Open-Images-V5 上微调 mask 头，风格类似 Mask-RCNN）。关键设计是"passthrough 物体"概念：先检测出**目标区域**的 mask，再减去**不能改动**的物体（机器人手臂、夹爪、正在被操作的目标物）的 mask，得到干净的可 inpainting 区域：

$$m = m_{\text{region}} \setminus \big(m_{\text{arm}} \cup m_{\text{gripper}} \cup m_{\text{obj}} \big)$$

> 用大白话说：想在抽屉里加个干扰物，就得先框住抽屉、再把伸进抽屉的机械臂和手里的可乐罐"抠掉",免得把机器人自己也画花了。检测阈值也分开设：抓取/放置任务里手中物体和容器用 0.07、passthrough 物体用 0.05；水槽/新背景任务用 0.04 检测桌面、0.03 检测 passthrough。

**2）增广文本提议（Sec 4.2）。** 两种方式：(a) 手工 prompt——人手写要替换成什么物体（如 `Robot picking up a blue and white stripe cloth`），能保证生成的是分布外物体；(b) LLM prompt——用 GPT-3 一次性（1-shot）提议一大批带细节视觉描述的物体，并同时生成三段 prompt：`ViT region prompt`（要分割的区域）、`passthrough object prompt`（要保留的物体）、`inpainting prompt`（要画进去的东西）。实验主要用 LLM 提议；作者指出 LLM 有噪声但通常不损害控制性能，而且**必须**用 few-shot——zero-shot 时 LLM 会幻觉出没法用的 prompt（附录 C 给了失败例）。

**3）文本引导 inpainting（Sec 4.3）。** 用 Imagen Editor（在 Imagen 之上微调的文本引导 inpainting 模型，级联扩散架构：64×64 base + 256×256 超分，都换了新的卷积图像编码器条件输入）。对轨迹**每一帧**迭代地喂入 $(o_i, m, \ell_{\text{aug}})$，得到增广观测 $\tilde{o}_i$，既按文本插入新物体/干扰物、又保持 mask 外内容与原图一致。作者强调 ROSIE 对 inpainting 模型是**模型无关**的。若增广创造的是**新任务**，还需相应改写指令 $\ell \to \tilde{\ell}$（如把"pick green rice chip bag"改成"pick blue microfiber cloth"）。最终得到增广 episode

$$\tilde{e} = \big\{(\tilde{o}_i,\, a_i,\, \tilde{o}_{i+1},\, \tilde{\ell})\big\}_{i=1}^T$$

> 用大白话说：整段视频从头到尾把绿薯片袋换成蓝色超细纤维布，动作一帧不改，指令改成"抓蓝布",于是白捡一段"抓新物体"的示范。

**4）下游策略训练（Sec 4.4）。** 用 RT-1 架构（FiLM 条件化的 EfficientNet + TokenLearner + Transformer 输出动作）。在预训练 RT-1（35M 参数，315k 步，lr $1\times10^{-4}$）之上，用原始 13 万 episode 与 ROSIE 生成 episode **按 1:1 混合**微调 85k 步，并用更小的 lr $1\times10^{-6}$ 保证稳定。

**成本参考：** 训练策略 16 TPU / 1 天；OWL-ViT 生成 1k episode 的 mask 用 1 TPU / 1 小时；Imagen Editor 生成 1k episode 用 4 TPU / 2 小时（64×64 base 与 256×256 超分各 2 小时）。

## 三、实验结果

数据集为 RT-1 的多任务真机数据（约 13 万条示范、744 条语言指令，采自实验室办公室与厨房）。两个基线：**NoAug**（预训练 RT-1，不做增广）；**InstructionAug**（只改写指令、图像不变，类似 Xiao et al. [71]）。主结果 Table 1 覆盖 RQ1（学新技能，蓝色）与 RQ2（增强鲁棒性，橙色），各任务族分别评测 50/20/16/10/80/40/27 条 episode，共 243 条。数字为成功率：

| 任务族 | NoAug | InstructionAug | ROSIE |
|---|---|---|---|
| Move object near novel object | 0.86 | 0.78 | **0.94** |
| Pick up novel object | 0.25 | 0.30 | **0.75** |
| Place object into novel container | 0.13 | 0.25 | **0.44** |
| Place object into sink | 0.0 | — | **0.6** |
| Pick up object in new backgrounds | 0.33 | — | **0.71** |
| Place object into cluttered drawer | 0.38 | — | **0.55** |
| Pick up object (with OOD distractors) | 0.33 | — | **0.37** |

**RQ1（学全新技能）。** ROSIE 在四类难度递增的任务上都胜出：move 到新容器胜过两个基线；place 到未见容器较基线提升**至少 75%**；抓取由 ROSIE 生成的未见可变形物体（未知超细纤维布，黑/蓝色）较基线提升**至少 150%**；最极端的是把可乐/百事罐放进一个**从未采过真机数据**的金属水槽——ROSIE 用 Imagen Editor 把打开的抽屉换成水槽，训练后整体成功率 **60%**（可乐 0.8 / 百事 0.4），而 RT-1 完全定位不到水槽、成功率 **0%**。作者据此说明 InstructionAug（只改文本、不改视觉）虽把指令拉回分布内，却认不出新物体的视觉，故不足。

**RQ2（增强鲁棒性）。** (a) 未见背景：用 GPT-3 生成大量桌布并替换桌面、或在桌上插入水槽，ROSIE+RT-1 在 8 个设置中的 7 个显著超过 NoAug，剩下 1 个持平，整体 **115%** 提升。(b) 新干扰物：对"pick coke can"加入训练中未见的多个可乐罐干扰、对抽屉放置任务在抽屉里加入未见干扰物，ROSIE 均改善（RT-1 会因抽屉里已有干扰物而误判任务已完成、提前输出终止动作，ROSIE 缓解了这一分布偏差）。

**RQ3（自举高层具身推理——成功检测）。** 用 ROSIE 增广 22764 条"放物入抽屉"episode，微调 CLIP-based 成功检测器（跟随 [71]）。造两套增广：(A) 抽屉里加生成的干扰薯片袋；(B) 抽屉里加生成的汽水罐。用 F1（阈值 0.5）在分布内集与 OOD 集（抽屉里含训练未见杂物）上评测（Table 2）：

| 数据集 | No Aug | ROSIE Aug (A) | ROSIE Aug (A)+(B) |
|---|---|---|---|
| Overall | 0.43 | 0.56 | **0.62** |
| In-Distribution | 0.66 | **0.67** | 0.66 |
| OOD | 0.19 | 0.45 | **0.57** |

随增广量增大，OOD 上的成功检测 F1 从 0.19 一路升到 0.57，而分布内几乎不变——说明 ROSIE 作为通用语义一致增广，能同时服务策略学习与具身推理，且不伤原分布性能。

## 四、局限性

作者在 Sec 7 明确列出：

1. **只增广外观、不生成新运动。** ROSIE 改的是物体/场景的外观，动作序列原封不动，因此无法凭空造出"新的运动技能"。作者建议混入仿真数据作为多样运动的来源。
2. **逐帧增广导致时序一致性损失。** 对每帧独立 inpainting 会破坏视频时序一致性；作者称至少对 RT-1 这种架构没观察到性能下降，但换成需要长时序建模的架构可能吃亏。文本到视频扩散模型能保时序一致却可能损失照片真实感与物理真实感，"照片真实 vs 时序一致"的权衡是开放问题。
3. **扩散模型计算重、无法在线增广。** 生成开销大，限制了 on-the-fly 增广；作者建议改用 mask transformer 架构（如 Muse，宣称快 10×）。
4. **生成失败案例（附录 C）。** LLM zero-shot 会幻觉出无用 prompt（故必须 few-shot）；当手中物体 mask 变得不规则时，Imagen Editor 会退化，无法把绿薯片袋完整替换成蓝布或黄鸭；有时会把 woven basket / glass mason jar 画成普通碗状容器。作者的辩护是：这类不完美生成通常只造成"指令-图像轻微错位",不一定伤害策略，甚至可能额外免费带来增广收益。

其他隐含局限（评述）：所有真机实验都绑定 RT-1 单一架构与谷歌自采数据，缺少跨机器人/跨架构验证；单任务族评测样本量偏小（部分 10~27 条 rollout），成功率的置信区间较宽；未与并发的 GenAug / CACTI 做同数据的直接数字对比，只做了定性定位区分。

## 五、评价与展望

**优点。** ROSIE 把"数据增广"从像素扰动升级为**语义级、可用自然语言编程**的真机数据合成，且整条流水线（分割定位 + LLM 提议 prompt + inpainting）几乎全自动，摆脱了 GenAug/CACTI 对人工 mask、物体网格和深度的依赖，这是相对并发工作最实在的工程价值。passthrough-mask 机制（把机械臂/在手物体从可编辑区抠掉）是让"逐帧 inpainting 不破坏机器人本体一致性"的关键小设计，简单但有效。"从未采过水槽数据、却学会往水槽放物"这个 0%→60% 的结果，最有力地证明了扩散先验能把 internet 知识注入真机策略。RQ3 把同一套增广推广到成功检测，说明方法不局限于底层控制。

**缺点与开放问题。** (1) 只改外观不改运动是根本天花板——真正的技能泛化仍受限于原始轨迹的运动分布，这也是后续"世界模型 / 视频生成造运动数据"路线试图突破的地方。(2) 逐帧独立生成的时序一致性问题被作者一笔带过（"RT-1 不掉点"），但对依赖时序建模或需要精细接触动力学的策略，帧间抖动是否真无害仍缺乏定量证据。(3) 生成成本高、离线批量生成，难以做在线/闭环增广；Muse 类快速生成或一致性模型是自然的加速方向。(4) 增广质量对 mask 精度和 LLM prompt 质量高度敏感，失败模式（不规则 mask、hallucinated prompt）目前靠阈值和 few-shot 兜底，缺乏自动质量过滤或验证闭环。

**与其他公开工作的关系。** ROSIE 与 GenAug（深度引导、需网格/mask）、CACTI（需人工 mask/标签）构成同期"生成式真机增广"三件套，定位是"最自动、不需深度、能同时造任务与干扰物"。它站在 RT-1（策略）、Imagen Editor（inpainting）、OWL-ViT（开放词表分割）、GPT-3（prompt 提议）四个大模型肩上，本质是一次"基础模型组合式的数据引擎"示范。未来值得探索的方向包括：把外观增广与运动生成（仿真或视频扩散）结合以突破运动多样性天花板、引入自动化的生成质量校验以过滤有害样本、以及用更快的生成骨干实现在线增广闭环。

## 参考

1. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale.* arXiv:2212.06817, 2022.（被增广的策略架构与数据来源）
2. Wang et al. *Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting.* arXiv:2212.06909, 2022.（ROSIE 使用的 inpainting 引擎）
3. Chen et al. *GenAug: Retargeting Behaviors to Unseen Situations via Generative Augmentation.* arXiv:2302.06671, 2023.（并发的深度引导生成式增广）
4. Mandi et al. *CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning.* arXiv:2212.05711, 2022.（并发的 inpainting 加干扰物增广）
5. Xiao et al. *Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models.* arXiv:2211.11736, 2022.（InstructionAug 基线与成功检测流程来源）
