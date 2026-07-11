# LatBot：面向 VLA 模型的通用潜在动作蒸馏

> **论文**：*LatBot: Distilling Universal Latent Actions for Vision-Language-Action Models*
>
> **作者**：Zuolei Li, Xingyu Gao, Xiaofan Wang, Jianlong Fu
>
> **机构**：中国科学院微电子研究所（Institute of Microelectronics, CAS）、中国科学院大学、Microsoft Research
>
> **发布时间**：2025 年 11 月（arXiv 2511.23034）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.23034) | [PDF](https://arxiv.org/pdf/2511.23034)
>
> **分类标签**：`潜在动作学习` `VLA知识蒸馏` `人类-机器人跨具身` `视觉-语言-动作模型` `few-shot操作迁移`

---

## 一句话总结

LatBot 用任务指令引导的 VLM 从人类/机器人操作视频中学出**解耦为运动 token 与场景 token 的潜在动作**,再通过"潜在动作对齐损失 + 推理保持损失"把这份物理先验蒸馏进 VLA 模型,在 SIMPLER（Google Robot 78.0%/70.1%、WidowX 87.5%）、LIBERO（98.0%）和 Franka 真机 5 任务（10-shot 下平均 63.3%,大幅超过 π0 的 12.7% 与 π0.5 的 20.7%）上取得当时最优表现。

## 一、问题与动机

潜在动作学习（Latent Action Model, LAM）是近年 VLA 预训练的重要方向：从无动作标注的大规模视频（含人类手部操作）中学习跨具身的运动语义,以扩充训练数据来源。论文指出现有 LAM 方法（Genie、UniVLA 等）存在三个缺陷：

1. **无任务指令引导**：纯视觉重建目标（如 Genie）无法捕捉任务相关的变化,不同任务下相似的视觉动态会被混淆。
2. **多帧利用不充分**：难以准确捕捉运动动力学（如 UniVLA）。
3. **缺乏物理感知**：潜在动作只关注视觉表观变化,与真实可执行动作之间存在语义鸿沟,限制了下游迁移能力。

更本质的问题是,现有方法把"机器人自身运动"与"背景/环境变化"这两类完全不同性质的视觉动态**纠缠**在同一个潜在动作表示里,引入了任务无关的噪声（背景晃动、光照变化等）,削弱了潜在动作与真实物理动作之间的对应关系。

## 二、核心方法

LatBot 框架包含两个关键组件：**解耦潜在动作表示（Decoupled Latent Action Representation, DLA）**和**统一潜在动作解码（Unified Latent Action Decoding, UAD）**,二者联合训练；随后再通过知识蒸馏把学到的潜在动作知识迁移进 VLA 模型。

**1）解耦潜在动作表示。** 用一个预训练 VLM 作为潜在动作编码器,在词表中新增两个可学习 token `[CP_SCE]`、`[CP_MOT]`,以任务指令 $\ell$ 和多帧视觉观测为条件,提取：

$$\{Z_{\text{sce}}, Z_{\text{mot}}\} = f_{\text{vlm}}(V_{t:t+k}, \ell)$$

其中 $Z_{\text{mot}}$（运动 token）编码机器人末端执行器/双手的平移、旋转和夹爪等**主动运动**,$Z_{\text{sce}}$（场景 token）编码物体位姿、背景动态等**被动场景变化**。这样把机器人自身运动从环境噪声中显式分离出来。

用大白话说：与其让模型囫囵吞枣地学"画面前后哪里变了",不如让它先分清"是我（机械臂）动了"还是"是背景/物体自己变了",只有前者才是需要精确建模的物理动作。

**2）统一潜在动作解码。** 用解耦出的 $Z_{\text{sce}}, Z_{\text{mot}}$ 作为条件,联合指导未来帧重建 $V_{t+k}$ 与帧间动作序列生成 $A_{t:t+k}$。解码器初始化自预训练图像生成模型 SANA,在每一层实现场景与运动表示的双向交互（scene-motion bidirectional fusion）：视觉重建约束促使潜在动作捕捉可观测的场景变化,动作生成目标提供物理层面的监督,二者相互强化,使潜在动作既包含视觉动态又贴近真实物理先验。默认设置下 LAM 在 16 帧序列上运行,场景/运动各表示为 64 个 token。

**3）知识蒸馏进 VLA。** LAM 只能重建未来帧和生成帧间动作,与真正可执行的机器人动作仍有差距,因此设计两类损失把 LAM（教师）的知识迁移进 VLA 的 VLM 主干（学生）：

学生 VLM 只看首帧 $V_1$ 和指令 $\ell$（不看未来帧,因为下游推理时未来帧不可得）生成自己的动作表示：

$$\hat{Z}_a = f_{\text{vlm}}(\ell, V_1)$$

**潜在动作对齐损失（Latent Action Alignment Loss）**：

$$\mathcal{L}_a = \|\hat{Z}_a - Z_a\|_2^2 + \mathrm{KL}\big(p(\hat{Z}_a) \,\|\, p(Z_a)\big)$$

用大白话说：让学生仅凭"第一帧+指令"就能预测出教师（看过整段未来帧才推出）的潜在动作,MSE 项拉近特征、KL 项对齐分布,相当于逼学生学会"脑补"未来的运动趋势。

**推理保持损失（Reasoning Preservation Loss）**,防止蒸馏破坏 VLM 原有的语言推理能力,用下一 token 预测目标生成子任务描述：

$$\mathcal{L}_r = -\sum_i \log p(w_{i+1} \mid w_{\le i}, \ell, V_1)$$

总损失为 $\mathcal{L} = \mathcal{L}_a + \lambda_r \cdot \mathcal{L}_r$,默认 $\lambda_r = 0.5$。蒸馏之后,潜在表示仍非可直接执行的动作,还需接一个动作专家（action expert,采用 $\pi_{0.5}$ 风格的 flow-matching 头）做微调,损失拆分为末端执行器 MSE 项 $\mathcal{L}_{ee}$ 与夹爪二元交叉熵项 $\mathcal{L}_{gripper}$。

**训练规模**：LAM 预训练数据混合 OXE、AgiBoT、DROID（机器人）与 EgoDex（人类手部,提供双手骨架+指尖 3D 位置+6D 腕部朝向标注）,共约 100 万条视频片段,并设计了统一 44 维动作空间（双臂各 7 维 xyz+欧拉角+夹爪,加 10 个指尖各 3 维）来对齐人手和机器人动作表征。LAM 编码器可用 PaliGemma 或 InternVL3.5 初始化（论文默认 InternVL3.5-2B）；预训练在 16×A100(40GB) 上跑 14 天,蒸馏阶段再跑 7 天,蒸馏阶段使用的学生/下游 VLA 主干为 $\pi_{0.5}$。

## 三、关键结果

**SIMPLER（Google Robot,Table 1）**

| 设置 | 方法 | Pick Coke Can | Move Near | Open/Close Drawer | Open Drawer+Apple | Avg |
|---|---|---|---|---|---|---|
| Visual Matching | π0 | 87.3% | 35.0% | 72.6% | 16.0% | 52.7% |
| Visual Matching | MemoryVLA | 90.7% | 88.0% | 84.7% | 47.2% | 77.2% |
| Visual Matching | **LatBot** | **96.7%** | **91.7%** | **90.4%** | 33.3% | **78.0%** |
| Variant Aggregation | RT-2-X | 82.3% | 79.2% | 35.3% | 20.6% | 54.4% |
| Variant Aggregation | MemoryVLA | 80.5% | 78.8% | 53.2% | **58.3%** | 67.7% |
| Variant Aggregation | **LatBot** | **95.7%** | 78.3% | **73.0%** | 33.3% | **70.1%** |

文中强调 Variant Aggregation 设置下相对 π0 提升 24.1pp、相对闭源 RT-2-X 提升 15.7pp。

**SIMPLER（WidowX,Visual Matching,Table 2）**：LatBot 平均 **87.5%**（四任务分别 95.8%/87.5%/83.3%/83.3%）,相对基线 $\pi_{0.5}$（55.2%）提升 32.3pp,相对 UniVLA（47.9%）提升 39.6pp,相对 villa-X（40.8%）提升 46.7pp。

**LIBERO（四套件,Table 3）**：LatBot 平均 **98.0%**（Goal 98.6% / Object 98.8% / Spatial 99.0% / Long 95.4%）,优于 $\pi_{0.5}$（96.9%）、MemoryVLA（96.5%）、UniVLA（95.2%）,长时序任务上相对 $\pi_{0.5}$ 提升 3.0pp。

**Franka 真机 5 任务（Table 4）**：Franka Research 3（7 自由度+并联夹爪）,任务为 Pick up the cup（颜色判别）、Put the building block into slot、Close the oven、Dip the brush in the sauce、Put the pot in the oven,每任务 100 条人类遥操示范,分别用 10/50/全量示范微调、每设置 10 次试验评估：

| 方法 | 10-shot | 50-shot | Full |
|---|---|---|---|
| π0 | ~8%（多任务为 0%） | ~8% | 20.7%（Avg，全量列） |
| π0.5 | ~16%（多任务为 0%） | ~16% | 20.7%（Avg，全量列，见下） |
| **LatBot** | **48%** | **74%** | **60%**（Avg） |

（表中 Avg 列：π0 = 12.7%,$\pi_{0.5}$ = 20.7%,LatBot = **63.3%**,取所有任务×所有 shot 的整体平均。）文中特别指出：10 条示范即可让 LatBot 在全部 5 个任务上都取得非零成功率,而两条基线在 10-shot 下多数任务完全失败（0%）。同时观察到一个有趣的非单调现象：部分任务上 50-shot 成功率反而高于用全量数据微调,论文将其归因于全量数据中含有更多冗余/次优动作模式,使潜在动作学到了任务无关的变化,轻微降低了动作精度。

**消融（Table 5,SIMPLER WidowX 四任务）**：以 UniVLA 式基线（无 UAD、无 DLA）51.0% 为起点,仅加解耦表示（DLA）到 59.4%,仅加统一解码器（UAD）到 61.5%,两者叠加（完整模型）跃升到 **87.5%**——论文将其解释为 DLA 提供结构化的、与操作相关的潜在动作,UAD 把物理先验注入潜在动作学习过程,二者协同产生远超各自单独增益之和的效果。

**附加分析**：通过对最后一个文本 token 相对图像 patch 的注意力图做空间熵分析（式 (7)）,论文发现蒸馏后 VLM 的视觉 grounding 明显更聚焦于任务相关区域,即便存在干扰物（Task2）也能更准确锁定真实目标。

## 四、评价与展望

**优点**：把"运动 vs. 场景"解耦为独立 token 并通过双向融合解码器联合监督重建与动作生成,是对 UniVLA/Moto-GPT/villa-X 一脉 LAM 方法的自然且有效的改进——用指令引导 + 显式解耦直接回应了这些前作里"缺任务信号""视觉动态与物理动作纠缠"的具体缺陷,消融实验（Table 5）也确认了两个组件叠加的显著协同增益。真机 few-shot 结果（10 条示范即可让 5/5 任务跑通）是本文最具说服力的证据,说明蒸馏出的潜在动作确实承载了可迁移的物理先验而非单纯视觉相似性。

**与其他公开工作的关系**：整体思路与 UniVLA、villa-X、LAPA、Moto-GPT 同属"从大规模无动作标注视频学潜在动作再迁移到 VLA"的谱系,LatBot 的差异化贡献在于（i）指令条件化的潜在动作提取,（ii）motion/scene 显式解耦,（iii）统一解码器同时监督视觉重建与动作生成,（iv）面向 VLA 的专门蒸馏损失（对齐 + 推理保持）而非直接把 LAM 输出当动作 token 使用。相比直接借助大规模标注动作数据的 π0/π0.5/RT-2-X,本文延续了 LAM 系方法"扩展训练数据来源、降低对动作标注依赖"的初衷,并证明该路线在 few-shot 真机场景下具备明显优势。

**局限与开放问题**：（1）论文未设独立的 Limitations 小节,自陈的未来方向——扩展到更大更多样的操作视频数据集、扩大 VLA 模型规模、探索更复杂长时序与多具身任务——从侧面反映当前实验的数据规模（约 100 万视频片段）和任务复杂度（LIBERO 短时序、真机仅 5 个中等难度任务）仍有限,长时序（LIBERO-Long 95.4% 是四套件中最低）和多具身泛化未被充分验证。（2）50-shot 优于全量微调的现象说明潜在动作对训练数据质量/冗余度敏感,数据筛选或去冗余可能是提升鲁棒性的方向。（3）蒸馏依赖冻结的教师 LAM,教师本身的场景/运动解耦质量上限决定了学生能学到什么,尚未讨论教师 LAM 出错（如误判被动物体运动为主动运动)时的误差传播问题。（4）真机评测任务数（5 个）和试验次数（每设置 10 次）规模较小,统计置信度有限,更大规模、跨机器人本体的真机验证会是更有力的后续证据。

## 参考

- Bu et al. *UniVLA: Learning to Act Anywhere with Task-centric Latent Actions*. arXiv:2505.06111, 2025.
- Chen et al. *villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models*. arXiv:2507.23682, 2025.
- Chen et al. *Moto: Latent Motion Token as the Bridging Language for Robot Manipulation*. arXiv:2412.04445, 2024.
- Ye et al. *Latent Action Pretraining from Videos (LAPA)*. arXiv:2410.11758, 2024.
- Physical Intelligence. *$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054, 2025.
