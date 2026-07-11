# LeRobot：面向端到端机器人学习的开源库

> **论文**：*LeRobot: An Open-Source Library for End-to-End Robot Learning*
>
> **作者**：Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, Mustafa Shukor, Jess Moss, Alexander Soare, Dana Aubakirova, Quentin Lhoest, Quentin Gallouédec, Thomas Wolf
>
> **机构**：Hugging Face；University of Oxford（Francesco Capuano，工作完成于 Hugging Face 期间）
>
> **发布时间**：2026 年 02 月（arXiv 2602.22818）
>
> **发表状态**：ICLR 2026
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.22818) | [PDF](https://arxiv.org/pdf/2602.22818)
>
> **分类标签**：`开源工具库` `标准化数据集格式` `低成本硬件遥操作` `异步解耦推理` `模仿学习基线`

---

## 一句话总结

LeRobot 是 Hugging Face 主导的端到端机器人学习开源库,垂直整合了从底层电机中间件、标准化多模态数据集格式（LeRobotDataset）、到异步解耦推理引擎与多种 SOTA 策略实现（ACT/Diffusion Policy/VQ-BET/π0/SmolVLA，以及 RL 的 HIL-SERL/TD-MPC）的全栈；截至 2025 年 9 月,社区已通过该库开源共享 16K+ 数据集、来自 2.2K+ 贡献者的超过 250 万条轨迹（规模超过此前中心化收集的 Open-X 数据集 150 万条与 RT-1 数据集 13 万条），其异步推理方案在真机 SmolVLA 评测中把成功率基本持平的同时,将任务完成时间从 137.5 秒压到 97.0 秒,单位时间吞吐量提升逾一倍。

## 一、问题与动机

机器人学习正从依赖运动学/规划/控制模块化流水线的**显式模型**（explicit models,存在复合误差、对非结构化场景欠建模、扩展性差等问题）转向直接从交互数据学习的**隐式模型**（implicit models）。但作者指出该转型被生态系统碎片化严重拖慢,具体有三类实际困难：

1. **中间件碎片化**：不同机器人平台的高低层控制接口各自为政,团队被迫为每个平台重复开发适配层。
2. **数据集与格式碎片化**：数据常以 TensorFlow Datasets、ROS bag、自定义 JSON 等各异格式发布,缺乏统一、模态丰富的 schema,难以聚合成更大规模的混合数据集。
3. **学习框架实现差异**：算法、数据处理、评估管线的细微实现差异会造成结果方差极大（引用 Henderson et al. 2018 的 RL 复现性研究结论）,叠加机器人硬件本身的变异性,进一步阻碍复现。

LeRobot 的目标是提供一个可及（accessibility）、可扩展（scalability）、可复现（reproducibility）的统一端到端 stack,把研究者从系统集成的琐碎工作中解放出来。

## 二、核心方法

LeRobot 由五个纵向打通的组件构成：

**(1) 统一机器人集成**：一套一致的 Python 中间件 API,原生支持 SO-100/SO-101（单臂与双臂）、Koch-v1.1、ALOHA/ALOHA-2、HopeJR 人形手臂、Stretch-3、LeKiwi 移动操作平台、Reachy-2 人形共 8 套硬件（2025 年内从最初的 3 套平台扩展而来）,通过共享中间件对接 FeeTech、Dynamixel 等低成本执行器 SDK,支持 leader-follower 遥操作范式,并预留了向新机型扩展的可组合接口。

**(2) 标准化数据集 LeRobotDataset**：自包含的多模态 schema,涵盖高频本体感受读数、多路摄像头画面、遥操作状态信号,以及任务文本描述、embodiment 细节、帧率等元数据。底层用 `.parquet` 表格文件 + 压缩 `.mp4` 视频文件存储,配合 `StreamingLeRobotDataset`（基于 `IterableDataset` 接口 + `torchcodec` 按需解码视频）实现远程流式读取,无需下载整个数据集即可训练,内存占用与数据集规模无关；论文报告在稳态（初始化后）下流式与本地预加载性能相当。

**(3) 高效可复用算法实现**：纯 PyTorch 实现的单任务模仿学习基线（ACT、Diffusion Policy、VQ-BET）、多任务/语言条件的通用策略（π0、SmolVLA）,以及强化学习基线（HIL-SERL、TD-MPC）。库设计强调易用性——从零训练一个模型不到 100 行代码,部署推理不到 40 行代码。

**(4) 优化推理引擎（异步解耦推理）**：把动作预测（inference）与动作执行（control）在物理和逻辑两个层面解耦。物理解耦允许推理运行在算力更强的远程机器上,而控制回路保持在机载低延迟环路；逻辑解耦采用异步生产者-消费者机制：推理进程以 look-ahead horizon $H$ 并行预测动作序列（action chunk）$a_{t:t+H-1}$,控制端以固定频率消费动作,重叠的多个预测块通过用户可自定义的聚合函数 $f\big(\pi(o_0), \pi(o_k)\big)$ 合并,保证动作队列非空、避免机器人因等待推理而空闲。大白话说：与其让机械臂算完一步动作再执行、执行完再等下一步,不如让"大脑"在后台提前多算几步动作存起来,"手脚"按固定节奏边执行边取用,两者并行而不互相阻塞。

**(5) 仿真支持**：因接触丰富的操作任务在仿真中仍具挑战,LeRobot 将真实数据作为训练主力,仿真环境（原生集成 LIBERO 与 Meta-World）主要用于系统性评估算法而非训练闭环。

## 三、关键结果

低成本硬件相对工业臂的成本优势（公开 BOM,附录 A）：

| 机器人 | 类型 | 成本（欧元） |
|---|---|---|
| SO-100/101 | 单/双臂机械臂 | 约 225（双臂 550） |
| Koch-v1.1 | 单/双臂机械臂 | 约 670（双臂 1346） |
| ALOHA | 双臂机械臂 | 约 21,000 |
| HopeJR-Arm | 人形手臂 | 约 500 |
| LeKiwi | 移动操作平台 | 约 230 |

社区数据下载量 Top 平台（截至 2025 年 9 月,按下载量降序）：

| 机器人 | 下载次数 | 数据集数 | Episode 数 |
|---|---|---|---|
| Panda | 1,878,395 | 588 | 926,776 |
| xArm | 1,107,329 | 74 | 450,329 |
| WidowX | 832,177 | 100 | 214,117 |
| KUKA | 662,550 | 3 | 419,784 |
| SO-101 | 319,586 | 3,965 | 58,299 |
| SO-100 | 278,697 | 5,161 | 78,510 |
| Koch-v1.1 | 43,561 | 849 | 20,959 |

策略模型的内存/延迟基准（四平台测得,此处摘录 CPU 峰值内存与 GPU 推理延迟）：

| 模型 | 参数量 | CPU 峰值内存 | RTX 4090 延迟 | A100 延迟 |
|---|---|---|---|---|
| ACT | 52M | 817.4MB | 5.0ms | 13.8ms |
| Diffusion Policy | 263M | 1.22GB | 369.8ms | 613.9ms |
| π0 | 3.5B | 4.13GB | 209.4ms | 569.0ms |
| SmolVLA | 450M | 1.69GB | 99.2ms | 278.8ms |

真机异步 vs 同步推理对比（SO-100 臂,pick-place/stacking/sorting 三任务,10 episode，每 episode 60s，SmolVLA 策略）：

| 指标 | Sync（同步） | Async（异步解耦） |
|---|---|---|
| 平均成功率（3 任务） | 78.3% | 73.3% |
| 单 episode 平均耗时 | 13.75s | 9.70s |
| 60s 内完成方块数（均值 ± 标准差） | 1.8 ± 0.45 | 3.8 ± 1.3 |

此外,LeRobot 上的轨迹数量在约一年内（2024.09—2025.09）从接近 0 增长到超过 250 万条,超越此前中心化数据收集的 Open-X（150 万条）与 RT-1（13 万条）数据集规模；ACT 因体量小、推理快、少量演示即可获得可用策略,长期是社区上传模型中占比最高的架构（不同月份份额在 59.6%–91.4% 之间）。

## 四、评价与展望

**优点**：LeRobot 是一篇扎实的系统/基础设施论文,其价值不在算法创新而在纵向整合——把中间件、数据格式、推理引擎、算法基线用统一的 PyTorch 原生接口串起来,显著降低了机器人学习的工程门槛。用公开 BOM 的低成本硬件（SO-100/101 等仅为 Franka Panda 等工业臂成本的一小部分）驱动去中心化数据采集,是与 Open X-Embodiment、DROID 等依赖少数机构中心化采集范式的一个有意义的补充路径,类似开源社区的众包模式。异步解耦推理是论文中少数带真机量化验证的工程贡献,在成功率基本持平前提下将吞吐量提升超过一倍,对实际部署（尤其在算力受限的机载设备上跑大模型策略,如 π0 在 CPU/MPS 上甚至超出 5 秒硬超时）有直接价值。

**局限与开放问题**：（1）论文自陈硬件覆盖仍"far from exhaustive",算法覆盖也非详尽,后续需要持续追加平台与算法实现；（2）库中复现的策略（ACT、Diffusion Policy、π0、SmolVLA 等）均为已有工作的重新实现,论文未提出新的学习算法或理论,其贡献边界是系统而非方法；（3）社区自愿贡献的数据质量参差,例如"Other"类别中有 2370 个数据集的机器人平台标签缺失（标记为 unknown）,反映出去中心化采集在元数据规范性上的代价,论文未给出针对性的数据质量把关机制；（4）仿真环境（LIBERO、Meta-World）目前仅用于评估而非训练闭环,真实数据与仿真数据的混合训练路径未被探索；（5）底层推理性能优化（量化、图编译等）被论文明确列为当前被忽略、留给未来工作的部分。相较于 GR00T N1、SmolVLA 等聚焦模型能力的工作,LeRobot 更适合被理解为这些工作赖以复现和部署的底层基础设施,其长期价值将取决于社区贡献数据/模型的持续增长速度与质量治理能力。

## 参考

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT / ALOHA）, 2023
- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024
- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025
- Open X-Embodiment Collaboration, *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, 2025
- Khazatsky et al., *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*, 2025
