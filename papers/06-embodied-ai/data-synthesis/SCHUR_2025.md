# SCHUR：揭示数据与模型规模化对人形机器人高层控制的影响

> **论文**：*Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots*
>
> **作者**：Yuxi Wei, Zirui Wang, Kangning Yin, Yue Hu, Jingbo Wang, Siheng Chen（Jingbo Wang、Siheng Chen 为通讯作者）
>
> **机构**：上海交通大学 · 上海人工智能实验室 · 浙江大学 · 密歇根大学
>
> **发布时间**：2025 年 11 月（arXiv 2511.09241，v2 修订于 2025 年 12 月）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.09241) | [PDF](https://arxiv.org/pdf/2511.09241)
>
> **分类标签**：`人形机器人` `人体动作数据挖掘` `text-to-motion` `数据规模化`

---

## 一句话总结

从海量人体视频/动作数据出发,经"视频动捕 → 重定向(给机器人绑 17 个虚拟关键点)→ 全身 tracker 过滤"的自动流水线,构建 260+ 小时、约 17 万条带语义标注的人形机器人动作数据集 Humanoid-Union;并用 FSQ 量化 + LLaMA 自回归的 SCHUR 框架做 text-to-robot-motion,在数据与模型双规模化下把 MPJPE 重建误差压到 0.0326(论文称较既有方法约 37%)、FID 压到 12.6(较"生成人体动作再重定向"基线约 25%),并在 Unitree G1 上完成实物部署。

## 一、问题与动机

- 机器人真机采数据昂贵;而人形机器人与人类形态相近,人体动作视频/动作数据近乎免费且海量,是天然的规模化数据源,其语义标签还能自动提取,支持模态对齐与高层控制学习。本文的子主题正是"人形动作视频挖掘"。
- 已有做法有三类不足:
  1. 直接把人体动作当先验(如 Humanoid-VLA、部分 RL 工作),但人体动作分布并不等于机器人动作分布;重定向虽能转换,却改变分布、复杂化流程,削弱模态对齐;
  2. 部分人体动作低层执行器根本执行不了,拉低数据质量、限制执行效果;
  3. 数据规模太小(常用 HumanML3D 不足 30 小时),缺乏对应的模型规模化研究,没吃透人体动作对人形机器人的潜力。
- 也有工作([15] Learning from massive human videos)尝试直接生成机器人动作,但:表征只用 root+DoF、无欧氏空间关键点正则;不过滤不可执行/噪声片段;规模化探索浅、无模型 scaling 研究;实物仅演示上肢、表达力有限。
- 目标:自动从人体视频/动作构建"大规模、高质量、可执行"的机器人动作数据集,并配一套有效的可扩展学习策略,系统性研究数据与模型规模化对人形机器人高层控制的影响。

## 二、核心方法

分两大块:Humanoid-Union 数据流水线 + SCHUR 生成框架。

### 2.1 Humanoid-Union 数据流水线(三步)

1. **Video MoCap 与描述**：检测跟踪视频中的人 → 姿态估计与优化(遵循 Motion-X [10] 的标准流程),输出 SMPL 表示 $\langle \theta_{human}, r^{pos}_{human} \rangle$;再用 VLM(GPT-4V)自动标注文本描述;并整合已有公开人体动作数据(Motion-X、AnimationGPT [8]、ScaMo [11] 等)与部分自采数据。
2. **重定向(4 步)**：
   i) 给机器人人工绑定 17 个虚拟关键点(定义与身体部件的固定拓扑关系,模仿人体骨架并保持对称性);
   ii) 用固定 T-pose 重新标定各部件尺度;
   iii) 逆运动学 IK 求解机器人各 DoF 关节位置 $j_{robot}$;
   iv) 正运动学 FK 计算虚拟关键点位置 $k^{pos}$ 与朝向 $k^{ori}$;随后类 PHC [12] 做高度校正 + 平滑后处理。
   注:这些虚拟关键点并非与机器人物理部件一一对应,只满足特定相对关系,既恢复"类人风格",又提供 3D 欧氏空间的冗余表征。
3. **全身 tracker 过滤与后处理**：在近乎完整的 AMASS 上训练一个通用 whole-body tracker(训练范式类 ExBody2 [6],把含虚拟关键点的重定向结果作为观测)。用它去"执行"每条动作,过滤明显不可行者(约 10% 被过滤),并向表征注入物理/运动学先验;同一 tracker 后续直接用于实物部署。

规模:约 260 小时机器人动作、约 170,000 序列;目标本体为 Unitree G1(29-DoF,其中 6 个固定、23 个实际参与控制)。

机器人动作最终表征为:

$$ e = \mathrm{concat}(r^{pos}_{robot},\, r^{ori}_{robot},\, j_{robot},\, k^{pos},\, k^{ori}) $$

其中 root 位置 $r^{pos}_{robot}\in\mathbb{R}^3$、root 朝向 $r^{ori}_{robot}\in\mathbb{R}^3$(roll-pitch-yaw)、DoF $j_{robot}\in\mathbb{R}^{d}$($d=29$)、虚拟关键点位置 $k^{pos}\in\mathbb{R}^{n\times3}$、朝向 $k^{ori}\in\mathbb{R}^{n\times3}$($n=17$)。

用大白话说:光给 root + 关节角(机器人"够用"的最小表征)网络其实学不好;额外塞进 17 个虚拟关键点的 3D 坐标和朝向这种"冗余"信息,相当于把同一个动作再用"像人一样的骨架点"描述一遍,给 tokenizer 更多可抓的结构线索,重建质量明显变好。

### 2.2 SCHUR:两阶段(离散 token 化 + 自回归生成)

**Stage 1 — Motion Tokenizer(FSQ-VAE)**。encoder/decoder 用卷积残差块。传统 VQ-VAE 靠 argmin 找最近码字,码本一大就容易 codebook collapse(只有少数码被反复更新)。SCHUR 改用 Finite Scalar Quantization(FSQ):对 latent 每一维做有界 + 取整,不用 argmin:

$$ \hat z = Q(z) = \mathrm{round}(f(z)) $$

$f$ 取 sigmoid 有界函数;码本大小 $|C|=\prod_{i=1}^{d}L_i$(每维量化成 $L$ 个整数)。round 不可导,用 straight-through(stop-gradient)。损失只需重建项,无需额外正则:

$$ \mathcal{L} = \lVert e - \mathrm{Dec}\big(f(z)+\mathrm{sg}(\mathrm{round}(f(z))-f(z))\big)\rVert_2^2 $$

用大白话说:VQ 像"查一本会越查越偏心的字典",码本一大就废;FSQ 干脆不用字典,把每个数字压进固定的小格子里四舍五入,天生不塌、能一路把码本做大,码本利用率始终在 99% 以上。

**Stage 2 — 自回归文本生成(LLaMA 结构,decoder-only)**。文本经预训练 T5-XL 得到 word-level tokens 作为前缀;把 Stage-1 的机器人动作 token 自回归生成出来。采用 prefix-bidirectional attention(文本部分双向,动作部分因果),块内加 RMSNorm 稳训练。损失:

$$ \mathcal{L} = -\sum_{t=1}^{n}\log p(\hat m_t \mid m_{<t},\, T) $$

$T$ 为文本 token 序列,$m$ 为动作 token。

用大白话说:把"文字 → 机器人动作"当成"续写"来做——先让模型充分读懂整句文本(双向注意力),再像写句子一样一个 token 一个 token 地把动作接出来。

## 三、实验结果

实验设置:Unitree G1 29-DoF(6 固定 / 23 控制);tracker 在 AMASS 上用 IsaacGym 训练(数据处理类 PHC),MuJoCo 评测;数据集按 80% / 15% / 5% 划分为 train/test/val,结果在 test 集报告。

**Tokenizer 消融(最大码本,重建误差,越低越好)**

| 指标 | Human Motion | 20% Data | Naive Repre | Ours |
|---|---|---|---|---|
| MPJPE | 0.0389 | 0.0483 | 0.0406 | **0.0326** |
| MPKPE | 0.0401 | 0.0497 | 0.0454 | **0.0372** |
| L1 Loss | 0.0413 | 0.0512 | 0.0463 | **0.0380** |

其中 "Human Motion" = 直接训人体动作再重定向;"20% Data" = 仅 20% 数据;"Naive Repre" = 只用 root+DoF、无虚拟关键点(FK 后计指标)。三项对照分别验证了"直接生成机器人动作 vs 人体动作后重定向""数据规模""冗余关键点表征"的价值。

**文本生成对比与消融(最大模型配置)**

| 指标 | Human Motion | 20% Data | MDM | Ours |
|---|---|---|---|---|
| FID(↓) | 16.9 | 23.5 | 20.8 | **12.6** |
| R@1(↑) | 0.688 | 0.653 | 0.667 | **0.719** |
| R@2(↑) | 0.717 | 0.689 | 0.692 | **0.754** |
| R@3(↑) | 0.755 | 0.707 | 0.713 | **0.812** |

FID 相对 "Human Motion"(生成人体动作再重定向)基线 16.9 降到 12.6,约 25% 的对齐改善;也优于扩散基线 MDM。

**不同高层控制下的 tracker 执行性能**

| 指标 | Dataset(全量) | Human Motion | W. raw data | Ours |
|---|---|---|---|---|
| Success Rate(↑) | 0.911 | 0.774 | 0.762 | **0.907** |
| MPJPE(↓) | 0.0874 | 0.1183 | 0.1095 | **0.0893** |
| MPKPE(↓) | 0.0867 | 0.1112 | 0.1028 | **0.0878** |

关键信号:SCHUR 生成动作的 tracker 成功率 0.907,几乎追平"数据全量上限"0.911,而"生成人体动作再重定向"(0.774)和"用未经 tracker 过滤的原始机器人数据训练"(0.762)都显著更低——说明分布匹配 + 数据过滤对下游可执行性至关重要。

**规模化曲线**

- Tokenizer(Fig 5):码本从 $2^8$ 扫到 $2^{16}$。FSQ 全程优于 VQ;VQ 在大码本处 collapse(重建不再提升甚至回升,如 MPJPE 从 ~0.0493 回到 ~0.0528),FSQ 则持续下降(MPJPE ~0.0598 → 0.0326)。码本利用率:FSQ 恒 >99%,VQ 随码本增大跌到 90% 以下。
- Generation(Fig 6):模型从 44M 扫到 3B,码本 256 vs 65536。模型越大所有指标越好(FID 在大码本 3B 处降到 12.6);大码本 + 大模型最佳,但当模型很小、容量不足时,配大码本反而学不动、表现更差。

## 四、局限性

- 仅在单一本体(Unitree G1 29-DoF)上完成重定向与部署,跨机器人/跨形态泛化未验证。
- "高层控制"边界只到 text-to-motion,尚无视觉闭环/感知;走向 VLA 仅作为叙述方向,未落地。
- 依赖 VLM 自动字幕质量与视频动捕(SMPL)精度,误差会传入数据集;约 10% 动作被 tracker 过滤,过滤准则较粗、缺乏定量刻画。
- 作者自陈:尝试"无低层 tracker、直接由文本输出机器人状态与动作"时,大模型实时部署困难、缺平衡先验使 locomotion 难以稳定控制,留作 future work。
- FID 绝对值 12.6 仍偏高(text-to-motion 领域 FID 常见个位数),且评测器在自建数据上训练,跨工作可比性有限。

## 五、评价与展望

优点:
- 少见地把 data scaling 与 model scaling 同时做出清晰曲线,系统性回答"规模化对人形高层控制的影响",实证价值高。
- 关键取舍务实:直接生成"机器人动作"而非"人体动作后重定向",Table II/III 表明分布匹配对下游 tracker 成功率(0.907 vs 0.762/0.774)极为关键;虚拟关键点冗余表征、FSQ 抗 collapse 均有干净的 ablation 支撑。
- 用同一个 whole-body tracker 兼做"数据过滤器"与"部署执行器",让训练数据质量与真机可执行性天然一致,是很好的闭环设计。

不足与开放问题:
- 缺乏与同期人形高层控制工作(Humanoid-VLA [2]、LangWBC [21]、RL-from-physical-feedback [27] 等)的定量对比,基线主要是自身消融(human-motion / 20% data / MDM / VQ),说服力偏内部。
- 260 小时数据由 Motion-X / AnimationGPT / ScaMo 等已有源 + 自采拼合,真正"新挖掘"的增量占比不透明;语义标签由 VLM 生成,粒度与多样性未系统评估。
- FSQ 用于动作 tokenization 并非首创(源自图像量化 [16]),贡献更多在于"把它用对,并证明其在机器人动作 + 大码本下的 scaling 优势"。
- 可能的改进:多本体/多形态重定向以研究跨机器人 scaling;把视觉观测纳入前缀走向真正的 VLA;利用 tracker 执行反馈做闭环数据再生(失败样本回灌、难例重标);引入更强的开放集语义评测与泛化测试。

## 参考

- Motion-X (Lin et al., NeurIPS 2023) [10] — 视频 → SMPL 全身动作数据源,本文动捕流程基础。
- FSQ: VQ-VAE Made Simple (Mentzer et al., 2023) [16] — 有限标量量化,tokenizer 抗 collapse 的核心。
- ExBody2 (Ji et al., 2024) [6] — whole-body tracker 训练范式来源。
- Learning from Massive Human Videos for Universal Humanoid Pose Control (Mao et al., 2024) [15] — 直接生成机器人动作的前作(只用 root+DoF),本文主要对照对象。
- PHC (Luo et al., ICCV 2023) [12] — 高度校正与 tracker 数据处理参考。
