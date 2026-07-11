# EgoZero：从智能眼镜中学习机器人操作

> **论文**：*EgoZero: Robot Learning from Smart Glasses*
>
> **作者**：Vincent Liu, Ademi Adeniji, Haotian Zhan（三人共同一作）, Siddhant Haldar, Raunaq Bhirangi, Pieter Abbeel, Lerrel Pinto
>
> **机构**：New York University；UC Berkeley
>
> **发布时间**：2025 年 05 月（arXiv 2505.20290，v2 为 2025-06-03）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.20290) | [PDF](https://arxiv.org/pdf/2505.20290)
>
> **分类标签**：`egocentric-human-data` `zero-robot-data` `morphology-agnostic-3D-points`

---

## 一句话总结

EgoZero 只用一副 Project Aria 智能眼镜采集**野外第一视角人类演示**、**零机器人数据**,通过把状态与动作统一压缩成"以自我为中心的 3D 点集"这一 morphology-agnostic 表征,训练闭环 Transformer 策略并 zero-shot 迁移到 Franka Panda 夹爪机器人,在 7 个操作任务上取得约 70%(74/105)的成功率,每个任务仅需 100 条演示 / 20 分钟采集。

## 一、问题与动机

机器人策略的核心瓶颈不在于"缺少物理劳动力",而在于**如何有效捕获并表征人类行为供机器人学习**。人类每天在自然环境里执行大量灵巧任务,是一座取之不尽的真实数据金矿,但此前把人类演示当监督信号的工作都存在可扩展性障碍:需要额外可穿戴设备、机器人数据、多相机标定、在线微调,或退化为低精度的 affordance 策略。另一类基于视觉的大规模预训练(如在多机器人数据集上训练的通用表征)虽在其训练分布内 morphology 鲁棒,却**尚未证明能纯粹从人类数据 zero-shot 迁移**。

本文提出一个雄心问题:**机器人能否仅从第一视角野外人类数据 zero-shot 学到操作技能?** 关键挑战有二:(1) 智能眼镜(单 RGB 鱼眼 + 双 SLAM 相机,视场重叠很小)既没有深度传感器,也无法可靠双目三角化,无法直接得到物体 3D 状态;(2) 眼镜只给出手掌 6DoF 位姿,不含末端执行器信息,而单目手部估计模型(HaMeR)在相机系下的绝对定位不准。EgoZero 要在没有多相机标定、没有深度、没有机器人数据的前提下,从原始视觉+里程计输入中恢复出精确的状态-动作表征。

## 二、核心方法

EgoZero 的关键思想承接 Point Policy / P3-PO 一脉:**把状态空间 $\tilde{\mathcal{S}}$ 和动作空间 $\tilde{\mathcal{A}}$ 都定义为第一视角下的 3D 点集**,从而同时统一人类与机器人分布、提升样本效率与可解释性,并对新场景、新形态泛化。每个时刻记为 $\langle I_t, H_t, T_t \rangle$:$I_t$ 是 1408×1408 RGB(投影函数 $\mathcal{P}$ 已知),$H_t, T_t \in SE(3)$ 分别是相机系下手位姿、世界系下相机位姿(均由 Aria 的 MPS 服务在线给出)。

### 2.1 统一动作空间(3D 末端点 + 夹爪开合)

眼镜只提供手掌位姿 $H_t$。作者用 **HaMeR** 得到 21 关键点手模型 $h_t \in \mathbb{R}^{21\times 3}$,但因 HaMeR 的绝对定位不准、局部形变较可靠,于是把"HaMeR 的局部手形变"与"Aria 的第一视角手位姿"融合:先由 HaMeR 关键点构造手掌位姿 $\hat{H}_t$(平移取 ThumbCMC/IndexMCP/MiddleMCP 三点质心,旋转由 Wrist–MiddleMCP 与 IndexMCP–MiddleMCP 两向量张成),再用 Aria 的 $H_t$ 纠正,最后投影回首帧,得到单链齐次变换:

$$\tilde{h}_t = T_0^{-1}\, T_t\, H_t^{-1}\, \hat{H}_t\, h_t$$

**用大白话说**:HaMeR 知道"手指怎么弯"但不知道"手在空间哪儿",Aria 知道"手在哪儿"但不知道手指细节,把两者按坐标系依次串起来,就能把每一帧的手都换算到统一的首帧世界坐标下。抓取检测则对拇指-食指欧氏距离做阈值;最终动作 = 拇指+食指坐标 + 夹爪开合的拼接向量。

### 2.2 统一状态空间(相机轨迹三角化出物体 3D 点)

因为眼镜没深度、双目又不可靠,作者转而利用 **Aria 精确的 SLAM 外参 + CoTracker3** 沿演示轨迹做三角化,三条假设保证物体状态在整段演示里静止:(1) 抓取前物体静止,(2) 相机有足够运动,(3) 环境非随机。做法:用 Grounding DINO + DIFT 把人工标注的 2D 点映到首帧,再用 CoTracker3 跟踪得到 $\langle T_i, u_i \rangle$ 轨迹,求解首帧 3D 点 $\mathbf{q}^*$ 使各帧像素重投影误差最小。由于 CoTracker3 常有"粘滞"导致点被推得偏远,作者加一个软深度惩罚偏好更近的解:

$$\mathbf{q}^* = \arg\min_{\mathbf{q}} \sum_{i\in\mathcal{I}} \left\| u_i - \mathcal{P}\!\left(T_0^{-1} T_i \mathbf{q}\right) \right\|_\rho + \lambda\, \mathbf{q}_z$$

其中 $\|\cdot\|_\rho$ 为 Huber 损失,$\mathbf{q}_z$ 是世界系深度坐标,$\lambda$ 为深度惩罚权重。附录 B 给出完整三步流程:先用**对极几何 + RANSAC**(基础矩阵 $F_{ij}=K^{-T}[\mathbf{t}_{ij}]_\times R_{ij}K^{-1}$)筛掉几何不一致的视角,再 RANSAC 三角化得候选,最后带深度偏置做非线性最小二乘精修。**用大白话说**:同一物体点在人移动脑袋时会从很多角度被看到,把这些视线在 3D 里"求交点"就还原出它的三维位置——用相机自己的运动当"多目",绕开了没有深度传感器的困境;实测重投影误差收敛到 2–4 像素。

### 2.3 策略学习与推理

在 $\mathcal{D}=\{(\tilde{s}^{(i)},\tilde{a}^{(i)})\}$ 上用行为克隆训练闭环 Transformer 策略 $\pi_\theta:\tilde{\mathcal{S}}\mapsto\tilde{\mathcal{A}}$,把预测建模为正态分布均值,最小化负对数似然:

$$\theta = \arg\min_\theta\ \mathbb{E}_{(\tilde{s},\tilde{a})\sim\mathcal{D}}\left[\frac{\|\pi_\theta(\tilde{s})-\tilde{a}\|^2}{2\sigma^2}\right],\quad \sigma=0.1$$

配合 history buffer 与时间聚合的 action chunking;关键是**3D 数据增强**:对点集注入随机噪声、施加随机 3D 变换(旋转 $R\sim\mathcal{U}(-\pi/6,+\pi/6)$、平移 $t\sim\mathcal{U}(-0.5,+0.5)$ m),使策略在更大的 3D 体积上学到可插值、可泛化的解——这是 zero-shot 野外迁移的必要条件。此外去掉位移 <1cm 的静止点、按 MAD 距离丢弃 DIFT 失败样本、长任务按 2 倍下采样。**推理时**用一部 iPhone 代表静止的第一视角(能可靠 unproject 出深度),开局把机器人初始化在工作区中点上方 30cm,用 Aruco 标定一次 iPhone-to-robot 变换,夹爪预测在 0 处二值化为 $\{-1,+1\}$,经逆运动学 $\tilde{\mathcal{A}}\mapsto\mathcal{A}$ 驱动 Franka。

## 三、实验结果

**设置**:Franka Panda 夹爪;每任务采集 100 条人类演示(约 20 分钟)、变换环境与物体位姿;**推理环境内零采集数据**;每任务评测 15 次。7 个任务:开烤箱门、把面包放盘、扫板、擦白板、水果分拣入碗、叠毛巾、书插书架。

主结果与消融(成功次数 / 15):

| 方法 | 开烤箱 | 放面包 | 扫板 | 擦白板 | 分拣水果 | 叠毛巾 | 插书 |
|---|---|---|---|---|---|---|---|
| From vision [Baku] | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| From affordances [Bahl'22] | 12 | 0 | 0 | 0 | 7 | 10 | 5 |
| EgoZero − 无 3D 增强 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| EgoZero − 单目深度替三角化 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **EgoZero(完整)** | **13** | **11** | **9** | **11** | **11** | **10** | **9** |

EgoZero 总计 74/105 ≈ **70%**,是(据作者所知)首个把野外人类数据无机器人数据地转成闭环策略的方法。关键对比与消融:

- **From vision(Baku 变体,直接从图像预测)**:因人-机视觉分布差异巨大、Aria 鱼眼进一步非均匀扭曲 2D–3D 对应,全线 0/15,说明纯像素策略无法 zero-shot 跨越 human→robot gap。
- **From affordances(开环:预训练抓取模型 + 首末抓取间线性轨迹)**:在需要复杂非线性运动的任务(放面包、擦白板)上 0/15,只在简单任务上部分成功——证明**闭环策略**对精确复杂运动不可或缺。
- **去 3D 增强 → 全 0**:没有增强,策略只学到稀疏 3D→3D 映射,无法插值到新位姿,面对新第一视角即 OOD。
- **单目深度替代三角化 → 全 0**:即便用 Depth Pro 并以场景中多个 Aruco 标定,深度误差仍 >5cm(附录 D 显示 monocular 深度存在时空非均匀 warp),所有策略彻底失败——凸显"相机轨迹三角化"这一设计的必要性。

**泛化性**:(1) 物体位姿——3D 对应编码位姿,策略泛化到远超训练体积的位形;(2) 物体语义——借 Grounding DINO 语言提示(如 "toaster oven")泛化到全新实例;(3) 相机——因基于 3D 点而 camera-agnostic,训练用 Aria、推理用 iPhone;(4) human-scale——2–3 个环境、不同桌高、多演示者、站/坐/走动,均落入同一表征空间。

## 四、局限性

- **3D 表征上限受输入点精度限制**:推理最大误差来自 DIFT 对应模型;策略在 3D 点上学习虽简单,却没有信息去纠正 3D 测量误差,精度天花板由点输入决定。
- **三角化的前提约束**:依赖 Structure-from-Motion,相机运动受限时不鲁棒;要求物体静止,因而**无法跟踪运动物体、无法处理随机(stochastic)环境**。
- **手部模型误差**:HaMeR + Aria 手位姿融合仍带 1–2cm 动作标签误差,阻碍高精度任务。
- **形态受限**:仅验证单臂夹爪,尚未覆盖灵巧手 / 双臂;仅 7 个准结构化桌面任务、每任务 15 次评测,统计规模偏小。

## 五、评价与展望

**优点**:(1) 把"morphology-agnostic 3D 点状态-动作空间"从多相机标定的受限设定推广到**纯第一视角野外设定**,并给出一整套可落地的数据处理管线(HaMeR×MPS 手位姿融合、相机轨迹三角化 + 对极/RANSAC/深度偏置精修),工程完整度高;(2) 用相机自运动替代深度传感器、用一部消费级 iPhone 完成推理,把硬件门槛压到极低,采集成本仅 20 分钟/任务;(3) zero-robot-data 且 camera-agnostic 的 zero-shot 迁移在概念上颇有说服力,消融也干净地证明了 3D 增强与三角化两处设计的因果必要性。

**局限与对照**:与同门 Point Policy / P3-PO(仍需多相机标定或部分机器人数据)相比,EgoZero 的贡献在于"去标定、去机器人数据";与 Phantom(用人类视频图像编辑造机器人数据)、Zeromimic / Bahl 等 affordance 蒸馏路线相比,它走的是"统一 3D 点 + 闭环 BC"而非"图像域对齐"或"开环轨迹",实验也显示后两者在复杂运动上失效。但其代价是把感知难点全推给了三角化的静态假设与单目对应模型:一旦物体在演示中移动、或场景随机,就无解。相较 π0 / OpenVLA 等大规模多形态 VLA,EgoZero 数据高效但任务复杂度、语言条件与长程能力都有限,更像"表征-管线级 proof-of-concept"而非通用策略。

**开放问题与可改进方向**:(1) 用廉价立体或 lidar 移除三角化的静止假设,支持动态物体与随机环境下的闭环学习;(2) 在更大数据规模下,或可用 grounded segmentation(如 SAM2)从密集无序几何直接学位姿,摆脱 DIFT 对应的误差天花板;(3) 把点表征扩展到灵巧手 / 双臂,验证在接触丰富、高精度任务上的可行性;(4) 引入语言条件与长程任务,与 VLA 范式融合;(5) 系统化研究"人类演示分布多样性"如何经 3D 增强正则化为更大可泛化体积,把成功率的统计置信度做扎实。整体而言,EgoZero 为"以人为中心、可扩展、廉价"的机器人数据来源提供了一个有价值的样板。

## 参考

1. Haldar & Pinto. *Point Policy: Unifying observations and actions with key points for robot manipulation.* arXiv 2502.20391, 2025.（本文 3D 点状态-动作空间的直接前身)
2. Levy, Haldar, Pinto, Shrivastava. *P3-PO: Prescriptive point priors for visuo-spatial generalization of robot policies.* arXiv 2412.06784, 2024.
3. Bahl, Gupta, Pathak. *Human-to-robot imitation in the wild.* arXiv 2207.09450, 2022.（affordance 基线)
4. Pavlakos et al. *Reconstructing hands in 3D with transformers (HaMeR).* arXiv 2312.05251, 2023.
5. Kareer et al. *EgoMimic: Scaling imitation learning via egocentric video.* arXiv 2410.24221, 2024.（同用 Aria、但需机器人数据)
