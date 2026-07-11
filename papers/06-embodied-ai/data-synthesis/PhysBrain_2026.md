# PhysBrain 1.0：人类第一人称视频驱动的物理常识数据引擎与具身基座模型技术报告

> **论文**：*PhysBrain 1.0 Technical Report*
>
> **作者**：PhysBrain Team（Xiaopeng Lin, Shijie Lian, Bin Yu, Changti Wu, Hang Yuan, Zhaolong Shen 等；Project Lead: Kai Chen, Cong Huang, Yukun Shi）
>
> **机构**：DeepCybo；北京中关村学院（Zhongguancun Academy）；中关村人工智能研究院（Zhongguancun Institute of Artificial Intelligence，通讯作者 Kai Chen @ zgci.ac.cn）
>
> **发布时间**：2026 年 05 月（arXiv 2605.15298）
>
> **发表状态**：未录用（预印本 / Technical Report）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.15298) | [PDF](https://arxiv.org/pdf/2605.15298)
>
> **分类标签**：`数据引擎` `人类第一人称视频` `具身预训练` `VLA` `物理常识` `深度感知 QA`

---

## 一句话总结

PhysBrain 1.0 提出一条"先理解、后行动"的具身训练路线：用一个类编译器的数据引擎把大规模人类第一人称（egocentric）交互视频编译成结构化"场景元信息 + 物理落地 QA"来预训练一个更强的物理基座 VLM，再通过双通路（frozen 通用 + 可训具身）与语言敏感的动作对齐把物理先验迁移到 VLA 控制；在 ERQA/PhysBench/MME 等 7 个 VLM 基准全面提升的同时，SimplerEnv-WidowX 达到 80.2%、GoogleRobot 91.33%、RoboCasa-GR1 64.5%、LIBERO 98.8% 均为 SOTA，真机 Franka 蔬菜抓取相对 π0.5 平均提升 +16.2pp。

## 一、问题与动机

当前 VLA 主流范式围绕单一训练逻辑组织：采集机器人轨迹、拟合动作策略、靠扩大机器人交互数据规模来提升系统。作者认为这条路线有两处根本局限：

1. **数据来源受限且昂贵**：机器人遥操作轨迹依赖具体平台，采集预算高、场景多样性有限、覆盖窄。
2. **拟合轨迹 ≠ 学到物理规律**：仅仅拟合动作并不保证模型掌握了支撑鲁棒行动的物理规律（视角变化、场景布局变化、任务组合变化下仍能正确动作）。

PhysBrain 1.0 主张把具身智能训练从 **action imitation（动作模仿）** 转向 **physical commonsense acquisition（物理常识获取）**：先建一个物理理解更强的通用多模态模型，再把它适配到具身控制。这一转向要求换数据源——转向大规模人类第一人称视频，因为相比机器人数据集，egocentric 人类视频更易获取、覆盖更广、天然以"与物理世界的交互"为中心，反复暴露接触、可达性、物体状态变化、工具使用、空间约束和多步任务结构。

报告聚焦两个连锁问题：**(a) 人类第一人称视频能否被系统性地转化为可扩展的物理监督？(b) 由此学到的先验能否有效迁移到下游具身控制？** 难点在于：原始人类视频本身不是具身监督，它不提供模型能直接用于物理推理与动作理解的显式信号；同时 VLM→VLA 的模仿主导后训练已知会侵蚀原有视觉-语言能力、导致灾难性遗忘。

## 二、核心方法

整体分两大部分：**数据引擎**（把视频编译成物理 QA）与**架构**（把物理先验迁移到 VLA 且保能力）。

### 2.1 数据引擎：像编译器而非字幕生成器

**设计原则**：监督必须 **physically explicit（物理显式）**，且必须把 **scene meta-information（场景元信息）** 与 **model supervision（模型监督）** 分离。作者反对朴素做法（给视频片段配 caption 让模型模仿）——通用 caption 太弱，偏向总结外观或高层事件，丢失动作生成所需的物体几何、接触进程、相对距离、可达性、子动作顺序等物理结构。

数据引擎把原始视频先解析成显式物理记录，再增强、检查，最后渲染成 QA；每一阶段有受约束的输入-输出接口，错误在传播进最终训练集前即可被检测。

**分阶段构建数据源**（staged construction，形成"物理常识注入课程"而非扁平视频描述集合）：
- 第一阶段：Ego4D、BuildAI（Egocentric-10K）、EgoDex，切片后经视觉质量分与相机运动分（相机运动由 VGGT 导出的相机参数估计成 motion score）过滤，保留质量足够、抖动有界的片段。
- 第二阶段：扩展到 EPIC（epic-kitchens）、SEA-Small，更强调物理推理——不仅识别动作，还组织成物体、物理属性、空间关系、深度线索、状态变化、动作相关动态。
- 后续阶段：用元信息记录生成跨能力族的 free-form VQA；并混入 FineVision 等通用多模态数据作为"保留（retention）"辅助数据，而非重新标注。

### 2.2 结构化场景元信息（第一层，不直接作监督）

对每个视频段均匀采样帧，用受约束 prompt 只输出 JSON，schema 有三个顶层字段：

- `scene_elements`：静态/缓变要素——主操作物体、邻近物体、视觉细节、环境；**显式记录材质线索、几何、物理状态**（折叠/散落/透明/刚性/填充），因为物理可行性常依赖这些属性（可抓握刚性把手、可变形布料、松散小零件即使占据相似图像区域也对应不同具身解释）。
- `spatial_dynamics`：`initial_layout`（初始布局）+ `spatial_change`（空间变化），把静态识别升级为物理情境化的变化建模（手从上方接近、缩小距离直至接触、从堆中分离一件、相对支撑面重新定向等）。
- `action_execution`：`instruction_brief`（紧凑任务意图）+ `execution_detailed`（强调轨迹、速度剖面、接触物理的命令式细节，显式把观测运动链接到可执行控制描述）。

为提高质量与多样性，元信息由**多模型池**标注与交叉核验：GPT-5、Gemini 3.1 Pro、Gemini 3 Pro、Qwen3-VL-235B-A22B、Qwen3.5-397B-A17B。多标注者降低单一模型的风格/遗漏/推理偏差塌缩风险。

### 2.3 深度感知空间增强（第二层）

仅有元信息在需要 3D 关系或深度敏感规划时仍不足。对带物体 grounding 元数据的片段，管线用 **Depth Anything v3**（DA3NESTED-GIANT-LARGE 系列深度模型）计算逐点深度，将场景物体与其中心点关联、缩放到深度图坐标系，记录紧凑 `depth_info` 字典。服务两类 QA：
- **relative depth**：判断某物体相对另一物体更近/更远/更后/更低/更可达——帮助 VLM 区分"语义共现"与"物理排布"。
- **absolute depth / 度量距离**：以米/厘米学真实距离与尺度——因为部分机器人示范数据用末端执行器位置/位姿/位移表示，度量深度监督给了模型理解绝对位置与连续空间位移的更好基础。

### 2.4 QA 生成（第三层，真正的 VLM 训练样本）

用**完整多模型池**（GPT-5、GPT-5 mini、Gemini 3.1 Pro、Gemini 3 Pro、Qwen3-VL-30B-A3B、Qwen3-VL-235B-A22B、Qwen3.5-35B-A3B、Qwen3.5-397B-A17B）把结构化元信息实例化成多种监督形式。不同标注模型措辞不同、强调不同物理线索、暴露不同推理路径，避免训练出的 VLM 继承单一生成器的窄监督风格。

QA 空间按 **capability family（能力族）** 组织（Table 1 共 21 类，节选）：

| QA 族 | 主要目标 | 训练角色 |
|---|---|---|
| Spatial relations | 左右/上下/前后关系 | 空间智能 |
| Distance and depth | 相对深度与绝对度量距离 | 空间落地 |
| Size estimation | 真实长宽高与物体尺度 | 度量理解 |
| Grounding and coordinates | bbox/点/空腔坐标 | 视觉落地 |
| Viewpoint reasoning | 跨视角一致性、物体朝向 | 第一人称推理 |
| Next-step prediction | 当前观测+目标下的动作选择 | 具身决策 |
| Route planning / Long-horizon planning | 导航方向、多步任务分解 | 长程控制 |
| Affordance and safety | 可操作性、触碰安全、即时危险 | 物理常识 |
| Object state change | 操作后物理结果 | 动力学建模 |
| 通用类（OCR/Chart/Counting/Sci-knowledge/Visual logic 等） | — | 保留通用多模态能力 |

**具身推理格式**（2.6）：当任务涉及物理交互/规划/可行性时，QA 答案遵循

$$[\text{Perception - Environment}] \rightarrow [\text{Perception - Object}] \rightarrow [\text{Spatial Planning}] \rightarrow [\text{Action Execution}]$$

即先识别环境、再刻画被操作物体及其物理状态、再推理空间布局与意图变化、最后才描述具体执行。

> 用大白话说：不是逼模型写更长答案，而是强制它"先看环境→再看物体→再规划空间→最后落到动作"这个**内在思考顺序**，让思维链和具身动作对齐；这与通用指令微调不同——后者可能答对却绕过了对控制迁移真正重要的中间物理因素。

### 2.5 质量控制

检查放在**标注阶段之间的接口**而非仅最终清洗。元信息抽取阶段约束必须可解析为 JSON、含预期字段、通过可见 scene_elements/spatial_dynamics/action_execution 表达物理内容；解析失败/无可用图/超生成上限/抽取错误的记录被赋 failure 状态而非静默流入 QA 生成。深度处理另设检查：验证深度文件存在、采样图存在、深度数组可加载、物体中心映射在深度图内并采样有界；缺失/损坏时写入 sentinel 深度值与非成功 `depth_status`（`npz_missing` / `image_missing` / `npz_corrupted`），下游 QA 生成据此跳过深度相关问题但仍用其他有效场景信息。

### 2.6 架构：物理基座 VLM + 保能力 VLA 适配

**基座 VLM**：从 Qwen3-VL 出发，用数据引擎生成的 QA 训练 **PhysBrain 4B / 8B**。答案格式遵循 perception-state-planning-execution 组织，并混合 OCR/chart/visual logic/domain knowledge 等广义多模态 QA 保留通用性。

**双通路 VLA 适配**（3.3）：解决"学低层动作 vs 不覆盖通用多模态表征"的张力。通用通路（general pathway）从物理基座 VLM 初始化并**冻结**，作为稳定语义参考处理视觉观测与语言指令；具身通路（embodied pathway）同族初始化但**可训**，接收动作预测所需任务上下文。两通路经**非对称逐层融合**通信：具身通路 query 来自自身 $\mathbf{H}_E^l$，其 key-value 上下文拼接自身与来自通用通路的 stop-gradient 特征：

$$K_{\text{joint}}^l = [\text{sg}(K_G^l);\, K_E^l]$$

$$V_{\text{joint}}^l = [\text{sg}(V_G^l);\, V_E^l]$$

$$\mathbf{H}_E^{l+1} = \text{Attn}(Q_E^l,\, K_{\text{joint}}^l,\, V_{\text{joint}}^l) + \text{FFN}_E(\mathbf{H}_E^l)$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient。

> 用大白话说：具身通路可以"看"冻结通用通路里保存好的语义信息（当参考书），但动作学习的梯度只更新可训控制通路和动作解码器，绝不回流去改坏那本参考书——用两套参数分工，避免单套参数既要保通用能力又要专精控制而顾此失彼。

**动作条件语言对齐**（3.4）：只保能力不保证跟随指令，尤其在机器人数据稀少、指令高度可从场景预测时，纯模仿会学到"视觉驱动捷径"而弱用指令。用 action query 对比"纯视觉动作上下文"与"语言条件动作上下文"：

- Prior 分支：$\text{Input}_{\text{prior}} = [v,\, \mathcal{A},\, \ell]$，因果 action query 只能 attend 视觉、不能 attend 指令；
- Posterior 分支：$\text{Input}_{\text{post}} = [v,\, \ell,\, \mathcal{A}]$，action query 可 attend 视觉与语言。

二者支持一个 log-likelihood-ratio 风格目标：prior 估计"语言能从视觉与 action-query 信息中被解释多少"，posterior 给出语言条件动作表征；该对比目标鼓励动作表征保留与指令相关的信息，而非只靠视觉-动作相关性。与动作预测损失联合优化，并用 stop-gradient 防止为增大比值而退化基座语言模型。

> 用大白话说：故意做一个"看不到指令"的分支和一个"看得到指令"的分支，让模型必须证明"指令确实改变了动作"，从而在数据高效设定下逼策略真正听懂指令，而不是靠场景猜。

**统一动作生成**（3.5）：用 flow-matching 训练连续动作解码器（Diffusion Transformer, DiT）。设 $\mathbf{a}_1$ 为真值动作轨迹，$\mathbf{a}_0 \sim \mathcal{N}(0,I)$ 为高斯噪声，$\mathbf{a}_t = (1-t)\mathbf{a}_0 + t\mathbf{a}_1$，给定具身通路 query 状态条件 $\mathbf{C}$，解码器预测速度场：

$$\mathcal{L}_{\text{FM}}(\psi;\, \mathbf{C}) = \mathbb{E}_{t,\mathbf{a}_0,\mathbf{a}_1}\left[\left\|v_\psi(\mathbf{a}_t, t, \mathbf{C}) - (\mathbf{a}_1 - \mathbf{a}_0)\right\|_2^2\right]$$

动作用末端执行器坐标系（EEF）表示，含平移与旋转分量——与数据引擎里度量深度 QA 的动机一致（理解绝对距离与位移有助预测连续位姿变化）。推理时用 posterior 分支条件化动作解码器生成连续控制命令。

**机器人适配协议与数据效率**（3.6）：SimplerEnv-WidowX 用 Bridge 数据、GoogleRobot 用 fractal 数据、LIBERO 用官方 spatial/object/goal/long-horizon、RoboCasa-GR1 用 PhysicalAI-Robotics-GR00T-X-Embodiment-Sim。PhysBrain 不消除机器人数据需求而是**改变其角色**：人类视频供物理与空间先验的大部分，机器人轨迹只教"这些先验如何映射到具体具身、动作参数化和基准分布"，预期收益是数据效率——若模型已懂物体状态/可达性/空间布局/度量距离/指令条件任务结构，则需要更少机器人示范即可完成适配。

## 三、实验结果

### 3.1 VLM 基准（Table/Fig 4，对比 Qwen3-VL-4B/8B、RoboBrain2.5-8B、VST-7B-RL、MiMo-VL-7B-RL）

PhysBrain 8B 在 ERQA、PhysBench、MME、MMMU、OCRBench、TextVQA 上取最佳，RealWorldQA 由 PhysBrain 4B 取最佳；7 个基准相对 Qwen3-VL 全部提升，说明物理落地 QA 同时提升了物理推理与通用多模态能力而非以一换一。

| 基准 | Qwen3-VL-8B | PhysBrain 8B | 备注 |
|---|---|---|---|
| ERQA | 43.0 | **45.5** | 具身推理 |
| PhysBench | 48.5 | **50.2** | 物理理解 |
| MME | 2373.3 | **2431.1** | 通用多模态 |
| MMMU | 53.2 | **55.2** | 知识推理 |
| OCRBench | — | 85.7 | 通用感知 |
| TextVQA | — | 83.3 | 真实视觉 |
| RealWorldQA | 70.5（4B） | 72.7（4B，SOTA） | PhysBrain 4B 亦全面超 Qwen3-VL-4B |

### 3.2 VLA 仿真（Table 2–5）

| 基准 | 最强先前方法 | PhysBrain 1.0 | 提升 |
|---|---|---|---|
| SimplerEnv-WidowX（4 held-out 任务，OOD） | Xiaomi-Robotics-0 79.2 | **80.2** | +1.0pp（π0.5 / Isaac-GR00T-N1.6-Bridge 均 57.1，领先 +23.1pp） |
| SimplerEnv-GoogleRobot | Xiaomi-Robotics-0 89.03 | **91.33** | +2.30pp（Pick Coke Can 100.0 / Move Near 94.8） |
| RoboCasa-GR1（24 桌面双臂灵巧任务） | VP-VLA 53.8 | **64.5** | +10.7pp（较 QwenGR00T+Qwen3VL 高 15.7pp） |
| LIBERO（4 suites 平均） | Xiaomi-Robotics-0 98.7 | **98.8** | +0.1pp（L-Spatial 99.6 / L-Object 99.6 / L-Goal 99.4 / L-Long 96.4） |

WidowX 上训练用 BridgeV2 而评测在 SimplerEnv 仿真任务，属 OOD 泛化测试——PnP Put Eggplant in Yellow Basket 达 100.0、Put Spoon on Towel 与 Put Carrot on Plate 并列最佳，说明物理先验改善了 OOD 泛化。四项 VLA 评测 PhysBrain 1.0 均取最佳平均分，最大增益出现在与训练分布差异大或具身/任务更难的 RoboCasa-GR1 与两个 SimplerEnv；LIBERO 已接近饱和故边际较小但不牺牲标准单臂模仿性能。

### 3.3 真机 Franka（第 5 节，Fig 6，对比 π0.5）

Franka Research 3 + Robotiq 2F-85 夹爪，两台 Intel RealSense D435i（外部视角 + 腕部视角），LeRobot 3.0 格式采集 9 类蔬菜共 450 条示范（每类 50，SpaceMouse 遥操 6-DoF）。两模型在**同一份 Franka 数据**上后训练、同一 50 试次协议评测，隔离"物理先验"的贡献。

| 设定 | π0.5 | PhysBrain 1.0 | 平均增益 |
|---|---|---|---|
| 单物体抓取（9 类平均） | 212/450 = 47.1% | 285/450 = **63.3%** | **+16.2pp** |
| 长程语义任务（2 类平均） | 31/100 = 31.0% | 45/100 = **45.0%** | **+14.0pp** |

每类均超 π0.5，增益在可变形/视觉模糊物体（大白菜、生菜）与光滑物体（茄子）上尤为明显。结论：即使最终策略用相同真机示范训练，人类衍生物理先验仍能改善下游机器人适配。

## 四、局限性

作者在 Discussion（第 6 节）自陈：

1. **数据引擎依赖上游感知与标注质量**：分阶段管线让错误可检测，但无法完全消除语义错误、遗漏物体、模糊接触、错误物理解释。
2. **深度监督继承深度估计与物体 grounding 误差**：管线能检测缺失/损坏深度记录，但有效深度图在透明、反光、严重遮挡物体上仍可能含局部不准。
3. **人类第一人称先验 ≠ 机器人具身约束**：人手、机器夹爪、移动底盘、仿真机械臂在形态、可达工作空间、力限、传感上不同，仍需机器人适配把通用物理先验映射为可执行策略。
4. **基准覆盖有限**：SimplerEnv/LIBERO/RoboCasa 未穷尽真实世界自主性、可变形物体交互、安全关键执行、严重分布偏移下的闭环恢复。未来工作方向：更强自动验证、更好的深度与 grounding 不确定性处理、更系统的人类视频监督消融、更广的真机评测。

## 五、评价与展望

**优点。**
- **路线定位清晰且可证伪**：把"具身能力来源"从机器人轨迹解耦为"物理常识（人类视频）+ 具身映射（少量机器人数据）"，并用真机 controlled comparison（同数据、同协议、只换先验）证明先验的独立贡献 +16.2pp，这比很多"换了骨干又换了数据"的对比更有说服力。
- **数据引擎工程化程度高**："类编译器"的分阶段接口 + 结构化元信息与 QA 监督分离 + 多模型池交叉核验 + 深度 sentinel 状态位，是一套可复制、可扩展、噪声可定位的合成监督流水线，把"人类视频→物理 QA"从口号落成 schema。
- **保能力设计有理论抓手**：双通路非对称 stop-gradient 融合 + prior/posterior 语言对比目标，直面 VLM→VLA 的灾难性遗忘与"视觉捷径忽略指令"两大失败模式，机制与 π0.5、"Actions as language"等公开工作的动机呼应且更结构化。

**缺点与开放问题。**
- **消融不足**：报告更像系统级技术报告，缺少对"元信息 vs 深度增强 vs 具身推理格式 vs 双通路 vs 语言对齐目标"各组件的逐项消融，难以判断 +16.2pp 中各设计的边际贡献；depth-aware QA 是否真的驱动了度量任务提升尚无隔离证据。
- **合成监督的天花板与循环风险**：QA 由 GPT-5/Gemini/Qwen 大模型池生成并交叉核验，物理"真值"实为强 VLM 的判断，可能把这些教师模型的物理误解系统性写进训练集；缺少针对深度/接触/可达性的人工或物理仿真校验基线。
- **数据规模与配比不透明**：报告未给出人类视频片段总量、QA 总条数、各能力族占比、retention 数据比例等关键统计，"数据效率"论断（更少机器人数据）缺少数据量-性能曲线支撑。
- **与近邻工作的边界**：思路与近期"人类视频→VLA 先验""egocentric 预训练"路线（如 EgoDex、以及各类以人手视频做 affordance/depth 预训练的工作）高度相关，报告未在实验上直接对比这些同类数据引擎，只对比了机器人 VLA 基线。

**展望。** 该工作强化了一个正在成形的社区共识：具身智能的瓶颈可能不在动作模仿本身，而在于行动所依赖的物理世界理解规模；把易得的人类第一人称视频编译成结构化物理监督，是绕开机器人数据稀缺的高性价比路径。可能的改进方向包括：引入物理仿真/可微渲染做深度与接触的自动校验以脱离纯 VLM 教师；把"具身推理格式"从固定四段扩展为可学习的思维结构；给出数据量-下游成功率的缩放律；以及在可变形、安全关键、闭环恢复等未覆盖场景上做更严格的真机评测。

## 参考

1. Bai et al. *Qwen3-VL Technical Report*, arXiv:2511.21631, 2025.（PhysBrain 的基座 VLM）
2. Physical Intelligence et al. *π0.5: a vision-language-action model with open-world generalization*, arXiv:2504.16054, 2025.（真机主对比基线）
3. Chow et al. *PhysBench: Benchmarking and Enhancing VLMs for Physical World Understanding*, arXiv:2501.16411, 2025.（物理理解评测）
4. Grauman et al. *Ego4D: Around the World in 3,000 Hours of Egocentric Video*, CVPR 2022.（核心人类第一人称数据源）
5. Hancock et al. *Actions as Language: Fine-tuning VLMs into VLAs without Catastrophic Forgetting*, arXiv:2509.22195, 2025.（保能力适配的相关动机工作）
