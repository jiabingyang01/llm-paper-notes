# SimHum：面向数据高效与可泛化机器人操作的仿真-人类协同训练

> **论文**：*Sim-and-Human Co-training for Data-Efficient and Generalizable Robotic Manipulation*
>
> **作者**：Kaipeng Fang, Weiqing Liang, Yuyang Li, Ji Zhang, Pengpeng Zeng, Lianli Gao, Heng Tao Shen, Jingkuan Song
>
> **机构**：University of Electronic Science and Technology of China（电子科技大学）、Southwest Jiaotong University（西南交通大学）、Tongji University（同济大学）、Shanghai Innovation Institute（上海创智学院）
>
> **发布时间**：2026 年 01 月（arXiv:2601.19406）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.19406) | [PDF](https://arxiv.org/pdf/2601.19406)
>
> **分类标签**：`sim-and-human co-training` `data-efficient manipulation` `diffusion policy` `zero-shot generalization`

---

## 一句话总结

仿真数据提供 kinematically aligned 的机器人动作但视觉不真实，人类演示提供真实视觉但存在人-机具身鸿沟——SimHum 用一套模块化 DiT diffusion policy 在两者上联合预训练，用 domain-specific vision adaptor 与 modular action encoder/decoder 分别"解耦并提取"仿真的动作先验和人类的视觉先验，微调阶段只需 80 条真机数据即可重组出策略，在 OOD 场景下平均 SR 达 62.5%，比 Real-only 基线高**7.1** 倍、比单源基线高**+35.0** 个百分点。

## 一、问题与动机

机器人模仿学习的核心瓶颈是真机数据采集昂贵。两类可扩展的替代数据源各有软肋：

- **仿真数据**（如 RoboTwin 2.0 生成）：机器人用与真机相同的 URDF，动作空间天然对齐（$\mathcal{A}_{sim} = \mathcal{A}_{real}$），但渲染真实感不足，存在**sim-to-real 视觉鸿沟**。
- **人类演示数据**：用第一视角 RGB 采集，视觉分布接近真实世界，但人手与机械夹爪存在运动学差异，导致**human-to-robot 具身鸿沟**（$\mathcal{A}_{hum} \neq \mathcal{A}_{real}$），人类动作无法直接在机器人上执行。

已有工作要么做**explicit alignment**（optimal transport、retargeting、visual editing，依赖复杂手工调参/辅助损失），要么做**unified co-training**（把异构数据当作通用数据增强直接混合，忽视各源固有的领域差异）。作者的关键观察是：这两个数据源恰好是**互补** 的——仿真擅长给动作、人类擅长给视觉。由此提出核心问题：能否仅靠在这两个源上"简单地协同训练"，就获得数据高效且可泛化的真实世界操作能力？

## 二、核心方法

### 问题形式化

把操作建模为条件序列生成。真机数据集 $\mathcal{D}_{real} = \{(o_t, s_t, \mathbf{a}_{t:t+H})\}$，其中视觉观测 $o_t \in \mathbb{R}^{H\times W\times 3}$、本体状态 $s_t \in \mathbb{R}^{d_s}$、动作序列 $\mathbf{a}_{t:t+H} \in \mathbb{R}^{H\times d_a}$。两个鸿沟被写成分布不等式：

$$P_{sim}(o_t \mid s_t^{env}) \neq P_{real}(o_t \mid s_t^{env}), \qquad \mathcal{A}_{hum} \neq \mathcal{A}_{real}$$

用大白话说：仿真"长得不像"真实（视觉分布对不上），人类"动作方式不一样"（动作空间对不上）。目标不是强行对齐分布，而是分别把仿真里可靠的**运动学先验** 和人类里可靠的**视觉语义先验** 抽出来，组装成能泛化到真实域的策略。

### 数据采集管线：保证先验可迁移

- **仿真侧**：用与真机完全一致的 URDF，保证运动学一致，动作先验可直接迁移。
- **人类侧**：用与机器人**相同型号相机、相同视角** 拍摄，保证视觉先验对齐真机部署。
- 三个源（仿真/人类/真机）采集**同一组任务**，确保任务级操作语义共享。

### 模块化策略架构（预训练阶段）

骨干是 DiT diffusion policy。图像经**共享 vision encoder**（ResNet-18，ImageNet-1K 预训练，各相机视角独立编码不共享权重）提取初始 visual token，本体状态经浅层 MLP 投影。两组关键模块负责"解耦"：

- **Domain-specific Vision Adaptors**：仿真 token 走 simulation adaptor，人类 token 走 real-world adaptor（均为两层 MLP + GELU）。让 backbone 处理一致的视觉表征，把领域特异视觉信息隔离在各自 adaptor 内。
- **Modular Action Encoder / Decoder**：人类分支用 human encoder 把人手姿态投到 latent token、human decoder 再译回人手空间；机器人分支用 robot-specific encoder/decoder 处理机器人本体状态与控制动作。这样 backbone 学到**具身无关的操作语义**（共享 latent 空间），而把具身特异的运动学细节隔离在 encoder/decoder 里。动作表示统一采用**相对轨迹**：机器人/仿真为 16 维（14 维末端位姿[位置+四元数]+二值夹爪），人类为 44 维（腕部位姿作末端代理 + 指尖 3D 坐标），30Hz 降采到 10Hz。

### 两阶段训练范式

**阶段一 · 仿真-人类预训练**：在两数据集上联合训练，用标准 diffusion 噪声预测目标

$$\mathcal{L}(\theta; \mathcal{D}) = \mathbb{E}_{(o,a)\sim\mathcal{D},\, \epsilon\sim\mathcal{N}(0,I),\, t}\left[\left\|\epsilon - \epsilon_\theta(z_t, t, o)\right\|^2\right]$$

总损失是两源的加权组合，$\alpha \in [0,1]$ 为 co-training ratio：

$$\mathcal{L}_{total} = (1-\alpha)\cdot\mathcal{L}(\theta; \mathcal{D}_{sim}) + \alpha\cdot\mathcal{L}(\theta; \mathcal{D}_{hum})$$

用大白话说：$\alpha$ 就是每个 mini-batch 里"抽多少人类数据"的比例——batch 大小 $B$ 时采 $\alpha B$ 条人类、$(1-\alpha)B$ 条仿真，用数据配比来实现加权。实验发现**等权 $\alpha=0.5$ 最优**。

**阶段二 · 真机微调**：把策略"选择性重组"——保留来自人类流的**real-world vision adaptor**（存住真实视觉先验），配上**机器人 encoder/decoder**（保证机器人运动学对齐），**丢弃** human encoder/decoder 与 simulation adaptor，然后只在少量真机数据上微调。用大白话说：微调时把"看得真"的那半（人类视觉通路）和"动得对"的那半（机器人动作通路）拼在一起，两个鸿沟就都被绕过了。

DDPM 用 squared cosine beta schedule，训练 100 步、推理加速到 8 步；预训练 200k 步、微调 60k 步，AdamW，均在单张 RTX 4090 上完成。

## 三、实验结果

**任务与数据**：4 个双臂任务（Stack Bowls Two / Click Bell / Grab Roller / Put Bread Cabinet），采用里程碑式打分（最高 2–3 分）。仿真每任务 500 条（RoboTwin 2.0 + 默认 domain randomization），人类每任务 500 条（跨 12 个场景），真机严格限定**80 条**（50 base + 30 complex，所有方法用同一批微调保证公平）。真机平台 COBOT Magic（leader-follower 遥操），人类采集用 Meta Quest 3 采手部姿态 + 脚踏板控制起止，成本低于 \$500。指标为 Success Rate（SR，全对才算成功）与 Progress Rate（PR，里程碑完成度）。

**主结果（Table I，平均值 ± 标准误）**：

| 方法 | ID SR | ID PR | OOD SR | OOD PR |
|---|---|---|---|---|
| Real only | 40.0±3.9 | 59.4±3.9 | 8.8±3.2 | 31.7±5.1 |
| HumReal（仅人类预训练） | 42.5±3.9 | 61.7±3.8 | 22.5±4.7 | 43.3±5.4 |
| SimReal（仅仿真预训练） | 47.5±3.9 | 69.2±3.6 | 27.5±5.0 | 48.5±5.5 |
| **SimHum（Ours）** | **67.5±3.7** | **82.1±3.2** | **62.5±5.4** | **77.3±4.5** |

关键读数：Real-only 从 ID 到 OOD 崩塌（SR -31.2、PR -27.7 个百分点），说明它靠的是 spurious visual correlation；单源预训练只有有限提升（ID 上 SR +2.5~7.5、OOD 上 +13.7~18.7 个百分点），各自被视觉/具身单一鸿沟卡住；**SimHum 在 OOD 上比单源基线高 +35.0 个百分点、比 Real-only 高 7.1 倍 SR**。

**消融一 · 解耦人类与仿真的作用（Stack Bowls Two）**：

| 消融 | 现象 | 幅度 |
|---|---|---|
| 人类数据 Leave-One-Factor-Out：去掉 $\mathcal{F}_{bg}$（背景多样性） | 目标 OOD 场景 SR 下降最猛 | **-60%** |
| 去掉 $\mathcal{F}_{dist}$（干扰物） | SR 下降 | -40% |
| 去掉 $\mathcal{F}_{obj}$（目标物体） | SR 下降 | -30% |
| 去掉 $\mathcal{F}_{light}$（光照） | SR 下降 | -25% |
| 仿真数据对位置泛化的贡献（3×3 训练 → 4×4 评测） | SimHum vs HumReal：见过位置 +23% PR，未见外围位置**+36.7% PR** | HumReal 在未见区 PR 衰减 -36.9% |

结论：**人类数据主管视觉泛化**（背景多样性最关键），**仿真数据主管空间/位置泛化**（其运动学多样性提供 IK-可解、可行的动作轨迹，让策略能应对新位置）。

**消融二 · 数据效率与可扩展性**：

- 固定采集时间预算（2h/4h/8h）：SimHum 全面超 Real-only，最高**+45%**；8h 时超 SimReal **+40%**、超 HumReal **+50%**。
- 真机数据规模缩放：SimHum 仅**8 条** 演示即可媲美 Real-only 用**160 条** 的表现。
- 预训练数据缩放：人类或仿真任一从 0 增到 500 都持续涨，到 500 仍在上升趋势（未见饱和）。
- 采集速度：仿真/人类平均比真机遥操**快约 5 倍**。

**消融三 · 架构与配比（Table II，Stack Bowls Two OOD）**：

| 变体 | SR | PR |
|---|---|---|
| SHC w/o real-world adaptor | 40.0±15.5 | 68.3±14.7 |
| SHC w/o relative action（改用绝对动作） | 45.0±15.7 | 58.3±15.6 |
| **SHC（Ours）** | **75.0±13.7** | **83.3±11.8** |

去掉 real-world adaptor：SR -35（约 -20% 相对）；改绝对动作：SR -30、PR -25 个百分点，说明相对动作对统一异构坐标系至关重要。co-training ratio $\alpha$：$\alpha=50\%$ 最优；$\alpha$ 从 50% 升到 90%（人类占比过高）ID/OOD 同时下降（被人类特异运动模式带偏）；从 50% 降到 10%（仿真占比过高）OOD 退化比 ID 更严重（印证人类数据贡献泛化所需的关键视觉先验）。

## 四、局限性

作者自陈四点：(1) 评测仅限基础操作任务，未覆盖 dexterous manipulation 和极端长程任务；(2) 鲁棒性虽提升，但 OOD 仍存在性能差距，可靠性待深挖；(3) 当前为单任务策略，缺乏 multi-tasking，需与 VLA 结合以支持开放世界指令跟随与跨任务迁移；(4) 数据仍是精心 curated 的，向 in-the-wild 多样数据扩展是未来方向。

补充观察：(a) 人类侧用固定视角、与机器人同款相机采集，本质上是**受控** 的第一视角，并非真正野外视频，因此"人类数据"的可扩展性优势打了折扣；(b) 真机仅 80 条、单臂平台单一、每场景 10–20 trial，绝对样本量偏小，标准误较大（OOD ±5 以上，消融甚至 ±15），部分结论的统计强度有限；(c) 全部实验在单机单卡完成，规模较小，是否能扩到大规模 VLA 尚未验证。

## 五、评价与展望

**优点**：(1) 问题切入点漂亮——把"仿真给动作、人类给视觉"的互补性讲清楚，并用架构（vision adaptor 隔离视觉域、action encoder/decoder 隔离具身）与训练（两阶段"预训练解耦-微调重组"）双管齐下地落实"disentangle-and-extract"，比单纯 pooling 混合更有原则性；(2) 数据效率证据扎实（8 条 ≈ 160 条、5× 采集速度、固定时间预算下 +45%），对真机数据稀缺场景很有说服力；(3) OOD 上 7.1× 的提升与 Leave-One-Factor-Out 消融，较有力地支撑了"两源各司其职"的核心主张。

**与公开工作的关系**：本质是 Maddukuri 等 *Sim-and-Real Co-training*（[37]，配比 $\alpha$ 的实现直接沿用）思路在"仿真+人类"三源上的推广；数据源分别接 RoboTwin 2.0（[19]，仿真生成）与 EgoMimic/Humanoid-Policy 一脉的人类第一视角学习（[27,29]）；骨干沿用 Dasari 等的 DiT diffusion policy 配方（[80]）。相较 explicit alignment（retargeting、optimal transport）避开了手工对齐，相较 naive unified co-training 则强调了领域解耦——定位清晰但组件多为已有模块的巧妙组合，单点创新性中等。

**开放问题与改进方向**：(1) modular encoder/decoder 与 domain adaptor 增加了参数与设计复杂度，能否用更轻量的共享结构（如统一 tokenizer + 领域 prompt）达到同等解耦效果值得探究；(2) 微调时"丢弃仿真 adaptor 与 human 通路"是硬切换，是否可用 soft gating 或蒸馏保留更多仿真动作先验；(3) 人类侧受限于同款相机固定视角，若换成真正野外视频，视觉先验的领域偏移会更大，adaptor 能否消化尚存疑；(4) 单任务限制明显，与语言条件 VLA 结合、以及把两源互补性扩展到 dexterous/长程任务，是最自然的下一步。总体是一篇工程完整、动机清晰、证据充分但规模偏小的实证论文。

## 参考

1. Maddukuri et al., *Sim-and-real co-training: A simple recipe for vision-based robotic manipulation*, 2025 —— 本文 co-training 配比与"仿真+真机"协同思路的直接前身。
2. Chen et al., *RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation*, arXiv:2506.18088, 2025 —— 仿真数据生成来源。
3. Kareer et al., *EgoMimic: Scaling imitation learning via egocentric video*, ICRA 2025 —— 从人类第一视角视频学习操作，人类先验一脉。
4. Qiu et al., *Humanoid Policy ~ Human Policy*, CoRL 2025 —— 人-机跨具身策略迁移的相关工作。
5. Dasari et al., *The ingredients for robotic diffusion transformers*, arXiv:2410.10088, 2024 —— DiT diffusion policy 骨干配方。
