# ST-VLA：面向通用机器人操作的4D时空感知理解框架

> **论文**：*ST-VLA: Enabling 4D-Aware Spatiotemporal Understanding for General Robot Manipulation*
>
> **作者**：You Wu, Zixuan Chen, Cunxu Ou, Wenxuan Wang, Wenbo Huang, Lin Cao, Yangtao Chen, Weichao Qiu, Xingyue Quan, Jieqi Shi, Jing Huo, Yang Gao et al.
>
> **机构**：南京大学（计算机科学系 / 智能科学与技术学院 / 电子科学与工程学院）、东南大学（仪器科学与工程学院）、华为诺亚方舟实验室（Noah's Ark Lab）
>
> **发布时间**：2026 年 03 月（arXiv 2603.13788）
>
> **发表状态**：未录用（预印本，v1 提交于 2026-03-14）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.13788) | [PDF](https://arxiv.org/pdf/2603.13788)
>
> **分类标签**：`分层VLA` `4D时空表征` `3D轨迹引导` `空间平滑mask` `RLBench` `零样本泛化`

---

## 一句话总结

ST-VLA 用"显式 3D 轨迹 + 跨模态平滑空间 mask"构成的统一 3D-4D 中间表征取代传统分层 VLA 中易产生几何歧义、时间不连续的 2D 中间表征(点/框/mask),并配套构建 430 万样本规模的人类操作数据集 ST-Human 来训练一个 4B 参数的高层时空推理模型 ST-VLM,在 RLBench 未见场景上把零样本成功率提升 44.6%,真实世界零样本泛化与抗干扰物成功率分别提升 30.3% 和 40.8%。

## 一、问题与动机

分层 VLA(高层 VLM 做语义推理 + 低层策略做连续控制)是当前机器人操作的主流范式,但现有工作普遍用 2D 中间表征(2D waypoint、bounding box、分割 mask)来连接两层。论文指出这存在一个根本性的**表征失配(representation mismatch)**:高层语义空间根植于静态、投影式的 2D 视觉域(如 RT-Trajectory 用 2D 路径草图、HAMSTER 用 VLM 预测的 2D 路径引导 3D 策略、RoboPoint/ARRO/PEEK 用分割 mask 做物体级 grounding),而低层控制运行在连续的 3D 物理空间中。这种失配导致三类问题:

1. **几何歧义**:一条 2D 轨迹可能对应无穷多条 3D 路径,深度信息被丢弃;
2. **时间不连续**:大多数 2D 信号按帧生成,缺乏跨帧的时序一致性,难以支持动态重规划;
3. **执行不稳定**:硬分割 mask 的离散边界会在低层策略的潜空间中引入跳变,造成动作抖动(jitter)和幻觉。

作者认为,一个有效的中间表征应当同时具备 3D 几何精度、时间连续性和跨模态一致性——这正是论文要构造的"统一 3D-4D 表征"要解决的核心问题。

## 二、核心方法

### 2.1 分层架构与问题形式化

在时刻 $t$,智能体接收单视角 RGB-D 观测 $\mathbf{o}_t \in \mathcal{O}$ 和机器人状态 $\mathbf{s}_t \in \mathcal{S}$,原始观测被反投影为机器人基座坐标系下的 3D 点云 $\mathbf{P}_t$。高层 VLM $\pi_{hi}$ 与低层执行策略 $\pi_{lo}$ 的交互形式化为

$$(\mathcal{Z}, l') = \pi_{hi}(\mathbf{o}_t, l), \qquad \mathbf{a}_t = \pi_{lo}(\tilde{\mathbf{o}}_t, \mathbf{s}_t, l')$$

其中 $\mathcal{Z} = \{\tau, \mathcal{M}\}$ 由一条 3D 轨迹 $\tau$ 与对应的空间 mask $\mathcal{M}$ 组成,$l'$ 是长程任务中被更新的子指令(短程任务中 $l'=l$),$\tilde{\mathbf{o}}_t = \psi(\mathbf{o}_t, \mathcal{Z})$ 是把时空引导 $\mathcal{Z}$ 融合进观测流的增强函数。

**用大白话说**:高层模型不再只吐出一个 2D 点或一个框,而是同时吐出"一条 3D 运动轨迹"和"一张告诉低层策略该看哪里、别看哪里的平滑注意力地图";低层策略吃的是被这两样东西"预处理过"的观测,而不是原始杂乱的场景。

### 2.2 ST-Human:大规模人类操作 3D-4D 数据集

为了让 VLM 学会产出这种表征,作者构建 **ST-Human**:用固定 RGB-D 传感器采集约 6 万段视频、30 万条episode,覆盖 14 类单臂桌面操作任务,通过半自动标注流水线生成约 430 万条监督样本。标注初始/终止 2D 接触点和物体语义后,流水线自动展开为三类任务:

- **2D 任务**:将标注的 3D 抓取点反投影为像素坐标做语义对齐,并用 SAM2 做视频跟踪生成 2D 关键点轨迹(指代表达 grounding、视觉轨迹 grounding);
- **3D 任务**:融合 2D 光流与深度图,将 waypoint 提升到工作空间坐标,并用一个"关系模块"把空间标签转成物体间相对位置/朝向的场景图式表示(空间推理、3D 轨迹深度估计);
- **4D 任务**:在同一视频内跨 episode 建立依赖关系,让 $\pi_{hi}$ 从部分观测预测剩余轨迹,训练其做任务进度判断与跨动作规划(长程规划)。

### 2.3 ST-VLM:统一时空 VLM 微调

ST-VLM 基于 4B 参数的 Qwen3-VL 基座,采用两阶段 SFT:先在公开多模态数据集上做通用语义预训练,再联合 ST-Human 与 RoboPoint、FSD、SAT 等专用公开数据集做领域内微调,使模型同时具备 2D 轨迹 grounding、深度感知的 3D 感知与长程 4D 推理能力。推理时,VLM 每隔 $H$ 步刷新一次引导(两阶段协议):先由当前 RGB 观测和指令预测投影到图像平面的 2D 轨迹 $\tau_{2D}$;再结合 RGB-D 观测、锚定的起始深度 $d_{start}$,以及沿轨迹点预测的**相对深度偏移量**(而非绝对表面深度,这样描述的是末端执行器的意图深度而非静态表面高度),把 $\tau_{2D}$ 提升为 3D 轨迹 $\tau_{3D} = \{\mathbf{p}_1,\dots,\mathbf{p}_K\}$。

### 2.4 时空引导的构造:安全操作管道与跨模态平滑 mask

3D 轨迹被进一步膨胀为一个"空间管道"(spatial tube):

$$\mathcal{T} = \bigcup_{k=1}^{K} \mathcal{B}(\mathbf{p}_k, r)$$

其中 $\mathcal{B}(\mathbf{p}, r)$ 表示以 $\mathbf{p}$ 为球心、半径 $r$ 的三维球。**用大白话说**:把预测出的每个轨迹点想象成一颗珠子,每颗珠子外面套一个半径 $r$ 的球,所有球的并集就是末端执行器"该走的那根安全管道"。

为过滤任务无关的视觉干扰,ST-VLA 引入一个选择性 mask 机制:先用 SAM2 生成场景内密集实例分割 $\{M_i\}$ 并反投影为 3D 占据 $\mathcal{V}_i$,若某物体占据与空间管道相交,即 $\mathcal{V}_i \cap \mathcal{T} \neq \emptyset$,则判定为任务相关物体、予以保留;其余任务无关区域在 RGB 与深度通道上都通过插值做**跨模态平滑**inpaint(受 RGB 颜色梯度和邻域深度值约束,保持局部特征连续)。与传统的硬分割二值截断不同,平滑边界过渡能保持特征连续性和流形稳定性,即便低层策略是在原始、未增强的观测上训练的("冻结策略"配置下依然有效,只靠推理期的空间 mask 就能提升性能)。最终增强观测 $\tilde{\mathbf{o}}_t$ 是把 $\tau_{3D}$ 的 2D 投影(用红-蓝渐变编码深度)叠加到被 inpaint 过的 RGB 图上,连同更新后的子指令 $l'$ 一起送入低层策略 $\pi_{lo}$。

### 2.5 低层 3D-aware 策略特化

论文选用两种代表性的 3D-aware 低层 backbone:**3D Diffuser Actor(3DDA)**(基于去噪的动作分布建模)和 **3D FlowMatch Actor(3DFA)**(基于 flow matching 回归轨迹片段),二者都把反投影点云 $\mathbf{P}_t$ 映射为可执行的 $SE(3)$ 关键位姿,并用运动规划器转化为连续关节控制。训练时对二者应用同样的增强函数 $\psi$,把真值轨迹渲染进 RGB 图并施加平滑空间 mask,使低层策略的注意力与 VLM 提供的时空先验对齐。

## 三、实验结果

### 3.1 ST-VLM 高层能力:9 个 2D/3D/4D 基准

在 RoboRefit、Where2Place、CVBench、SAT、CRPE 等公开基准以及自建的 ST-Human-Pointing/-Spatial/-Depth/-Planning 上,与 6 个 SOTA 基线(Embodied-R1-3B、PEEK-3B、Robobrain2.0-3B、Qwen3VL-4B 基座,以及闭源的 GPT-5.2、Gemini-3-flash)对比:

| 方法 | RoboRefit | Where2Place | ST-Human-Pointing | CVBench | SAT | CRPE | ST-Human-Spatial | ST-Human-Depth | ST-Human-Planning |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.2 | 16.65% | 43.00% | 20.50% | 79.62% | 66.00% | 78.89% | 68.00% | 4.00% | 88.00% |
| Gemini-3-flash | 57.75% | **87.00%** | 75.50% | 83.94% | 64.67% | **80.21%** | 84.00% | 6.00% | 84.00% |
| Robobrain2.0-3B | 50.61% | 61.00% | 65.50% | 81.22% | 69.33% | 73.03% | 48.00% | 8.89% | 88.00% |
| Qwen3VL-4B(基座) | 55.49% | 52.67% | 64.55% | 79.21% | 68.67% | 77.89% | 62.00% | 9.33% | 88.00% |
| **ST-VLM(本文)** | **88.15%** | 73.00% | **96.50%** | **84.52%** | **75.33%** | 73.67% | **98.00%** | **46.67%** | 92.00% |

ST-VLM 在 7/9 个基准上取得最优,尤其在深度估计(ST-Human-Depth 46.67% vs 最好基线 9.33%)和空间推理(ST-Human-Spatial 98.00% vs 闭源模型最高 84.00%)上领先明显,验证了具身数据专门微调对精细空间 grounding 的必要性;而在需要广博世界知识的通用基准(如 Where2Place、CRPE)上,闭源大模型仍占优。

### 3.2 RLBench 仿真操作(多任务设置,Seen / Unseen 成功率)

在 Close Jar、Light Bulb In、Put Groceries In Cupboard 三个代表性任务上,对比朴素模仿学习基线与接入 ST-VLA 引导后的效果(此处报告论文表 2 中数值可自洽复核的"多任务(Multi-task)"设置,三随机种子均值):

| 低层策略 | 配置 | Seen 平均成功率 | Unseen 平均成功率 |
|---|---|---|---|
| 3DDA | Baseline | 65.0% | 16.0% |
| 3DDA | ST-VLA(FT) | 75.4% | 67.9% |
| 3DFA | Baseline | 62.6% | 16.7% |
| 3DFA | ST-VLA(FT) | 69.5% | 69.7% |

Unseen(分布外颜色/物体形状)场景下,接入 ST-VLA 后两种低层策略的成功率均提升超过 4 倍;论文摘要与正文进一步汇总称,微调(FT)版本相较基线整体把零样本成功率提升了 **44.6%**。此外,即便不重新训练低层策略、只在推理期叠加平滑 mask 的"冻结(Frozen)"配置,也能获得明显增益,说明该表征本身对策略的潜空间稳定性有直接贡献。

### 3.3 长程 Push-Buttons 任务(仅单按钮 episode 训练,组合式长程测试)

| 类别 | 成功率 |
|---|---|
| Seen 单步任务 | 97.4% ± 4.4 |
| 长程 2 步(Unseen 组合) | 93.3% ± 11.5 |
| 长程 3 步(Unseen 组合) | 100.0% ± 0.0 |
| 总体 | 97.3% ± 2.3 |

低层策略训练时只见过单按钮 episode,靠高层 VLM 做任务分解与子目标序列规划,即可在推理期串联执行 2-3 个按钮的长程序列,验证了"把时序推理卸载给高层"的分层设计有效性。

### 3.4 真实世界实验(Franka Emika Panda)

在物体摆放与堆叠任务上,沿三个鲁棒性维度(零样本泛化到未见物体类别/几何形状、干扰物鲁棒性、三阶段长程链式执行)评测,每个设置 10 次试验:

| 方法 | 零样本泛化成功率 | 抗干扰物成功率 |
|---|---|---|
| OpenVLA | 46% | 34% |
| RVT2 | 37% | 19% |
| 3DDA | 51% | 31% |
| 3DFA | 47% | 27% |
| ST-VLA(3DDA) | 77% | 66% |
| ST-VLA(3DFA) | 74% | 71% |

论文汇总称 ST-VLA 相较基线 3D 策略平均提升零样本泛化成功率 **30.3%**、抗干扰物成功率 **40.8%**;在三阶段长程链式摆放任务中,即便低层策略只用短程单动作演示训练,整条序列成功率仍保持在 70% 以上。

## 四、局限性

论文在结论部分明确指出两点局限:(1)在极端杂乱场景下性能会下降,因为此时 SAM2 在物体边界模糊时的实例分割会出现歧义,导致任务相关性判定不准;(2)当前接口主要针对单视角执行做了优化,尚未验证多视角场景。作者将后续工作方向定为整合更鲁棒的多视角感知,以及把该分层引导框架适配到更广泛的低层 backbone 上,以建立一个真正通用、多功能的开放世界操作框架。

## 五、评价与展望

**优点**:ST-VLA 抓住了分层 VLA 领域一个被广泛忽视但很本质的问题——2D 中间表征与 3D 物理执行空间之间的失配——并给出了一个概念清晰、工程上可落地的解法:显式 3D 轨迹 + 跨模态平滑 mask。相比 HAMSTER、RT-Trajectory 等停留在图像平面的路径式引导,以及 RoboPoint/ARRO/PEEK 等依赖硬分割 mask 的物体级 grounding,ST-VLA 同时把深度信息和时间连续性纳入中间表征,并且平滑边界设计缓解了硬 mask 常见的潜空间跳变问题。"冻结策略"实验(不重训低层策略、仅推理期注入空间引导即可涨点)是一个有说服力的消融,说明增益很大程度上来自表征本身的质量,而非单纯的联合训练带来的过拟合式适配,这提升了方法向其他低层 backbone 迁移的可信度。

**值得商榷之处**:第一,ST-Human 是人类操作视频数据集,论文没有详细讨论从人手到机器人夹爪的具身鸿沟(embodiment gap)如何在高层轨迹/mask 迁移中被处理,这类似于同期不少"人类视频驱动的机器人预训练"工作面临的共性问题,值得后续消融。第二,深度提升依赖锚定起始深度 $d_{start}$ 加相对偏移量的策略,对首帧深度误差较敏感,论文未报告深度传感器噪声下的鲁棒性分析。第三,实验中的仿真任务集中在 RLBench 的少数几个任务(Close Jar / Light Bulb In / Put Groceries),真实机器人实验平台单一(仅 Franka Panda、单一夹爪构型),尚不清楚该方法在双臂或非夹爪末端执行器上的可迁移性。第四,论文中 Table 1 与正文对闭源基线的命名存在不一致(正文提到 Gemini-3-pro-preview,结果表标注为 Gemini-3-flash),细节上略显疏漏。

**开放问题**:与 4D 时空理解相关的其他公开工作(如利用世界模型做长程规划、或用隐式 3D 场景表示如 NeRF/3DGS 做操作引导的路线)相比,ST-VLA 选择了一条更"轻量"的显式几何路线(轨迹 + mask),换取了推理效率和与现有 3D-aware 策略(3DDA/3DFA)的即插即用兼容性,但也放弃了对场景遮挡关系、可形变物体等更复杂几何结构的显式建模。如何把该表征扩展到可形变/柔性物体操作、如何在多视角或移动底盘场景下保持同样的"平滑一致性"优势,是自然的后续方向。

## 参考

1. Ke, T.-W., Gkanatsios, N., Fragkiadaki, K. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations. arXiv:2402.10885, 2024.
2. Gkanatsios, N., Xu, J., Bronars, M., Mousavian, A., Ke, T.-W., Fragkiadaki, K. 3D FlowMatch Actor: Unified 3D Policy for Single- and Dual-Arm Manipulation. arXiv:2508.11002, 2025.
3. Li, Y., Deng, Y., Zhang, J., Jang, J., Memmel, M., Yu, R., Garrett, C. R., Ramos, F., Fox, D., Li, A., et al. HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation. arXiv:2502.05485, 2025.
4. Yuan, W., et al. RoboPoint / FSD 系列开放词汇空间定位数据集与模型, 2024/2025.
5. Kim, M. J., Pertsch, K., Karamcheti, S., et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246, 2024.
