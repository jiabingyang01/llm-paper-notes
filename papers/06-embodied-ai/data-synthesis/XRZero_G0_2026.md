# XRZero-G0：以接口、质量与配比推进灵巧机器人操作

> **论文**：*XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios*
>
> **作者**：James Wang, Primo Pu, Zephyr Fung, Alex Wang, Roy Gan (Project Lead), Hao Wang (Correspondence) et al.
>
> **机构**：X SQUARE ROBOT
>
> **发布时间**：2026 年 04 月（arXiv 2604.13001）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.13001) | [PDF](https://arxiv.org/pdf/2604.13001)
>
> **分类标签**：`UMI式数据采集` `robot-free数据` `数据配比律` `跨本体迁移` `VLA预训练`

---

## 一句话总结

XRZero-G0 用"VR 头显 + 双异构夹爪 + 背包算力"的可穿戴装置替代传统 UMI 手持相机做 robot-free 数采,配一条"采集-质检-训练-评估"闭环流水线把数据有效率做到 85%;并给出关键经验律——500 条便宜 robot-free 数据 + 仅 50 条真机数据(10:1),就能逼近 500 条纯真机的策略成功率,而 robot-free 单条采集成本仅约真机的 1/20。

## 一、问题与动机

灵巧操作基础模型的瓶颈始终是**高质量、动作对齐** 的示范数据。传统主从/VR 遥操作精度高但受机器人硬件约束、成本高、吞吐低;UMI 式手持 robot-free 采集虽可规模化,却存在三大短板,作者归纳为三个维度:

- **Interfaces(接口)**:现有手持系统依赖单目视觉 SLAM 估计位姿,在纹理缺失/动态/视觉退化场景下极易漂移,且长时序采集操作者易疲劳。
- **Quality(质量)**:多数流水线是**开环** 的,缺乏自动化质检,人体运动学产生的细微异常(运动模糊、越关节限、奇异位型、自碰撞)不被察觉,污染策略收敛。
- **Ratios(配比)**:社区缺乏"便宜 robot-free 数据 × 昂贵真机数据"如何最优混合的系统性经验指导,导致非本体数据利用率受限。

作者主张这三者必须**软硬件协同设计** 一起解决,由此提出 XRZero-G0。

## 二、核心方法

整体是一个硬件-软件协同框架,含四块:可穿戴接口、数据质量闭环、跨域配比策略、通用策略兼容。

### 2.1 Interfaces:抗漂移的可穿戴硬件

- **VR 头显 + 多视角**:采用商用 PICO 4 头显,利用其 inside-out 追踪提供毫米级 6-DoF 位姿(免视觉 SLAM,规避漂移),头顶挂一枚可调 RGB 相机给出俯视 egocentric 主视角;另配双腕相机,构成 **3 视角** 视觉流以对抗复杂任务中的严重遮挡。
- **异构双夹爪**:刚性挂接在左右 VR 手柄上,工程化出两种物理夹爪——**H 型** 按压驱动夹爪(面向快速、宏观抓取)与 **G 型** 手指驱动夹爪(面向灵巧、精细操作);采集时把两夹爪间距**显式标定** 到目标双臂机器人的基线臂距,以缩小人-机形态差。
- **背包算力 + 边端解析**:背包内含 PC 与电池;边端算力单元做严格的时空对齐,把自然语言指令、高频 6-DoF 手柄轨迹与 30Hz 多视角视频流打包同步后回传中心服务器,从而**彻底解放操作者的物理工作空间**。
- Table 1 中其定位精度 $\le 4\text{mm}$、视角数 $\ge 3$、原生支持视觉/触觉/听觉(V/T/A)三模态,均优于列出的其他 UMI 类系统。

### 2.2 Quality:自动化四级质检流水线

用大白话说:原始人体数据要连过四道质检才算"能用",最终有效率约 85%:

1. **视觉清洗与运动过滤**:自动图像质量评估丢弃严重模糊帧;对位置方差低于阈值的静止帧下采样,防止模型学到被动/不动的行为。
2. **运动学重定向与 IK 校验**:借时空对齐参数与目标本体 URDF,把人体 6-DoF 轨迹映射到末端执行器空间,IK 求解器滤除违反关节限位、遇奇异位型或自碰撞的无效片段。
3. **物理回放验证**:每类任务随机抽取一部分过滤后轨迹,在目标双臂机器人上**开环回放**;成功完成即作为该轨迹可执行性的最终判据。
4. **语义标注**:把长连续轨迹切成离散子任务块,标注被操作物体与关键帧,兼顾大规模预训练与任务微调。

有效数据率经此级联达到约

$$\eta \le 85\%$$

用大白话说:四级质检像四道漏斗层层筛,能通过全部关卡的示范约占八成半。

### 2.3 Ratios:数据混合与 scaling 策略

两阶段互补:

- **预训练阶段——构建可泛化隐空间**:海量 robot-free 数据作为"语义与空间泛化引擎",习得视觉-语义对齐、物体 affordance、拓扑轨迹规划等**与具体硬件无关** 的表征。
- **微调阶段——按比例注入真机做运动学锚定**:robot-free 数据缺乏本体特定的低层物理先验(电机延迟、摩擦、关节限位),故引入**高度受限比例** 的真机数据作 kinematic anchor,把预训练隐空间拉向目标硬件的物理约束。

形式化地,把训练集写成两域混合:

$$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{free}} \cup \mathcal{D}_{\text{real}}, \qquad r = \frac{\lvert \mathcal{D}_{\text{free}} \rvert}{\lvert \mathcal{D}_{\text{real}} \rvert}$$

用大白话说:训练数据由"便宜的 robot-free"加"少量真机"拼成,$r$ 就是两者条数之比(实验用 1:1 与 10:1)。

单条采集成本关系:

$$c_{\text{free}} \approx \tfrac{1}{20}\, c_{\text{real}}$$

用大白话说:采一条 robot-free 轨迹的成本只有真机遥操作的约 1/20。

核心经验律(作者称 **Few-Shot Physical Anchoring**):

$$P(500\ \text{free} + 50\ \text{real}) \approx P(500\ \text{real})$$

用大白话说:500 条便宜数据 + 50 条真机数据,策略成功率约等于 500 条纯真机;而后者贵得多,即"少量真机锚定 + 海量便宜数据"能替换掉约 90% 的昂贵真机数据。

### 2.4 通用策略兼容与数据集

输出的是**与算法无关** 的多模态标准化数据(同步多视角图像 + 语言指令 + IK 校验过的 6-DoF 动作),同时兼容 VLA 端到端连续控制与世界-动作模型(WAM/预测式)两大范式。据此构建 **G0-Dataset**:2,000+ 小时、3,000 个不同操作任务、长尾分布,峰值采集吞吐达 **93.2 episodes/hour**。

## 三、实验结果

四个研究问题:RQ1 采集效率、RQ2 跨本体保真、RQ3 纯 robot-free 数据 scaling、RQ4 配比律。评测基座策略为 Wall-OSS(带 Uni-CoT 的具身基础模型)、$\pi_0$(flow-matching)、$\pi_{0.5}$;两个结构迥异的双臂本体 CX001(高灵巧多关节)与 EX001(重载大工作空间)。

### RQ1 采集效率(Fig 5,单位:秒/任务)

| 任务难度 | Master-Slave | VR Teleop | XRZero-G0 | 相对 Master-Slave 加速 |
| --- | --- | --- | --- | --- |
| 简单 | 35 | 20 | 15 | 2.33× |
| 中等 | 75 | 65 | 40 | 1.88× |
| 困难 | 120 | 90 | 70 | 1.71× |

增益归因于把人体机动性与机器人运动学解耦,降低认知负荷并保留第一人称本体感反馈。

### RQ2 跨本体保真

系统在标定世界坐标系下精确捕获 6-DoF 位姿,经 IK 可**一比一(1:1)** 映射到 CX001 与 EX001 末端,真机验证支持精确空间回放,证明 robot-free 数据在功能上等价于常规真机遥操作数据。

### RQ3 纯 robot-free 数据 scaling(Fig 6)

基础抓取任务(仅用纯 robot-free 数据,300→500 episodes),成功率(%):

| 任务 | 模型 | 300 demos | 500 demos |
| --- | --- | --- | --- |
| Grasp Grape | $\pi_0$ | 37.5 | 50.0 |
| Grasp Grape | Wall-OSS | 50.0 | 62.5 |
| Grasp Eggplant | $\pi_0$ | 50.0 | 62.5 |
| Grasp Eggplant | Wall-OSS | 62.5 | 75.0 |
| Grasp Banana | $\pi_0$ | 50.0 | 62.5 |
| Grasp Banana | Wall-OSS | 62.5 | 75.0 |

复杂长时序双臂 **Flower Arrangement** 任务:纯 robot-free 数据扩到 2,000 episodes 时收敛稳定,Wall-OSS 在训练高度 $H=0.4\text{m}$ 达 **70%**,在未见高度 $H=0.45\text{m}$ 仍保持 **60%**——说明足量无约束 robot-free 数据能赋予策略真正的 3D 空间鲁棒性,克服固定基座遥操作数据的空间过拟合。

### RQ4 配比律(Fig 7,Wall-OSS 成功率 %)

基线为 500 条纯真机遥操作(500 Teleop);对比 1:1 增强(500 真机 + 500 robot-free,总量 1000)与 10:1 成本替换(500 robot-free + 50 真机,总量 550,与基线量级可比):

| 任务 | 500 Teleop | 1:1 (500+500) | 10:1 (500free+50real) |
| --- | --- | --- | --- |
| Picking Bananas | 75.0 | 100 | 75.0 |
| Picking Grapes | 87.5 | 100 | 75.0 |
| Folding Towel | 87.5 | 100 | 87.5 |
| Adding Sausage to Rice Cooker | 50.0 | 62.5 | 37.5 |
| Inserting Flower into Vase | 50.0 | 75.0 | 50.0 |

两条结论:
- **1:1 增强天花板效应**:即便真机数据已充足,再等量掺入 robot-free 数据仍作为"认知放大器"显著提点,如 Inserting Flower into Vase 从 50%(纯遥操作)升到 75%(1:1)。
- **10:1 零退化成本替换**:真机数据锐减 90%(50 vs 500)而总量相当时,策略仍**匹配或逼近** 500 条纯真机基线,如 Folding Towel 与 Picking Bananas 的 10:1 成绩与纯真机基线持平(87.5% / 75.0%)。据 $c_{\text{free}} \approx c_{\text{real}}/20$,作者据此宣称约 **20×** 的采集成本节省。图 8 给出 EX001/CX001 上多任务连续物理 rollout,验证零样本跨本体执行。

## 四、局限性

- **样本量偏薄**:RQ3/RQ4 成功率几乎全为 12.5% 的整数倍,暗示每条件仅约 8 次试验;柱状图未给置信区间(仅 scaling 曲线有阴影方差带),统计效力有限。
- **质检缺分级消融**:85% 有效率是四级流水线的合并结果,论文未给出各级(去模糊 / IK / 回放 / 标注)各自贡献多少,难以判断关键环节。
- **成本口径不一致**:"20×" 实为**单条** 成本比(1/20);而 10:1 训练态的真实节省(约 75 vs 500 真机等效单位,≈6.7×)与该 headline 数字并不一致,存在口径混用。
- **模态声明或超前于证据**:Table 1 标注原生支持 V/T/A 三模态,但正文实验未见触觉/听觉的定量结果,触觉能力更多是硬件潜力而非已验证。
- **跨本体范围有限**:"零样本"依赖 IK 重定向到已标定夹爪间距,仅在 CX001/EX001 两个同类双臂上展示,对 DoF/形态差异更大的本体未验证;任务多为短时序桌面操作。

## 五、评价与展望

**优点**:一是把 UMI 式 robot-free 采集从"手持单目 SLAM"升级为"VR inside-out 追踪 + 3 视角 + 异构双爪 + 背包算力"的系统化协同设计,抗漂移与吞吐(93.2 eps/h)都有工程价值;二是提出并量化了 **Few-Shot Physical Anchoring**——"海量便宜 + 少量真机锚定"这一配比结论对降本极具实操意义,且在 $\pi_0$/$\pi_{0.5}$/Wall-OSS 三个基座上交叉验证;三是给出 2,000 小时/3,000 任务的大规模数据集与闭环质检范式。

**与公开工作的关系**:方法上直接延续 UMI(Chi et al.)与 FastUMI 的手持 robot-free 路线,而"用 VR 头显 inside-out 追踪替代视觉 SLAM"这一取向与 ActiveUMI、exUMI 一致;异构夹爪缩小形态差的思路与 DexUMI(以人手本身作接口)相通;"少量真机 + 海量异构/网络数据混训"的配比发现与 $\pi_{0.5}$、RT-X 系跨本体协同训练的结论互相呼应;RDT2 也在探究 UMI 数据向零样本跨本体的 scaling。相比之下本文的增量在于把"接口-质量-配比"三件事系统化并给出可操作的混合比经验律,而非新的模型架构。

**开放问题与改进方向**:(1) 最优配比 $r$ 是否随任务时序长度、接触丰富度而变——当前只在两档比例上验证,缺 scaling law 指数;(2) 物理锚定律在**接触/力控** 任务(插入、擦拭、力敏操作)上是否仍成立值得存疑,因这类任务的真机低层先验更难被少量样本锚定;(3) 质检各级的定量消融与更严格的统计报告(更多试验次数 + 置信区间)将显著增强说服力;(4) human-to-robot 的形态/动力学 gap 尚未被单独量化,IK 回放成功率作为唯一保真判据偏粗。

## 参考

1. Chi et al. *Universal Manipulation Interface: In-the-Wild Robot Teaching Without In-the-Wild Robots.* arXiv:2402.10329, 2024.
2. Liu et al. *FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface.* arXiv:2409.19499, 2024.
3. Xu et al. *DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation.* arXiv:2505.21864, 2025.
4. Physical Intelligence et al. *$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization.* arXiv:2504.16054, 2025.
5. Liu et al. *RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization.* arXiv:2602.03310, 2026.
