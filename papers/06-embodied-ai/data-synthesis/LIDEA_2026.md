# LIDEA：基于隐式特征蒸馏与显式几何对齐的人到机器人模仿学习

> **论文**：*LIDEA: Human-to-Robot Imitation Learning via Implicit Feature Distillation and Explicit Geometry Alignment*
>
> **作者**：Yifu Xu, Bokai Lin（共同一作）, Xinyu Zhan, Hongjie Fang, Yong-Lu Li, Cewu Lu, Lixin Yang（通讯）
>
> **机构**：Shanghai Jiao Tong University；Shanghai Innovation Institute；Noematrix Ltd.
>
> **发布时间**：2026 年 04 月（arXiv 2604.10677）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.10677) | [PDF](https://arxiv.org/pdf/2604.10677)
>
> **分类标签**：`人到机器人迁移` `跨具身迁移` `特征蒸馏` `3D点云对齐` `数据合成`

---

## 一句话总结

LIDEA 从二维语义与三维几何两条互补路径弥合人手与机器人夹爪的具身鸿沟：用一个"伪机器人"中间域做两阶段传递式特征蒸馏（human ≈ pseudo-robot ≈ real-robot），再用"过滤具身几何 + 填充统一虚拟夹爪"的显式点云对齐消除深度感知偏差；在四个真机任务上验证人类视频可替代最多约 80% 的机器人示教，并把人类视频里的 OOD 鲁棒性迁移到机器人（Fold Towel 干扰物场景末阶段成功率 18% → 27%）。

## 一、问题与动机

机器人操作策略的性能高度依赖数据规模,但真机示教采集昂贵、耗人力,构成"数据稀缺瓶颈";而人类操作视频是几乎无限的现实交互数据源。核心难点是**跨具身鸿沟(embodiment gap)**——人手与机器人夹爪在视觉外观、3D 几何、动作语义三个层面都存在本质差异。

作者把已有方法归为三类并指出各自软肋:

- **视觉编辑(visual editing)**:像素级 inpainting 把人类视频"改绘"成机器人视频。代价高、易引入视觉伪影,且多在 2D 操作,破坏 depth-aware 策略所需的 3D 几何一致性。
- **统一表征学习(unified representation)**:通过 co-training 或统一 tokenization 直接在人类数据上训练;但人手与机器人的运动学差异大,预测人手参数等辅助目标无法干净地映射到机器人末端执行器,迁移效率低、需大量微调。
- **物体中心(object-centric)**:提取 affordance 或重定向物体轨迹绕开具身;但依赖显式状态估计,对类别级、可形变、铰接物体易失败。

LIDEA 的定位很明确:**不提出新的 visuomotor 架构,而是专注提高"人类数据利用率"**。为隔离该因素,全程固定采用带 DINOv3 特征注入的扩散式 3D 策略 RISE-2 作为基座。

## 二、核心方法

整体分三块:2D 特征蒸馏、3D 几何对齐、混合数据策略学习。

### 2.1 隐式 2D 特征蒸馏:传递式(transitive)双阶段蒸馏

直接对齐人手与真机观测不可行——缺乏严格配对数据(要求同物体操作时的精确几何对应)。LIDEA 引入一个中间的**伪机器人(pseudo-robot)域**作为"传递桥",把对齐拆成两段等价对应:human–伪机器人、伪机器人–real-robot。

**Stage 1(Human → Pseudo-Robot)**:配对 $\langle I_H, I_P \rangle$,$I_H$ 为人类演示帧,$I_P$ 为其离线合成的伪机器人对应帧。编码器 $E_H$、$E_P$ 均由 DINOv3 权重初始化;蒸馏时冻结 $E_H$ 作教师,训练 $E_P$ 作学生。关键改动是把 DINO 标准的随机局部裁剪换成 **RoInt(Region-of-Interaction)裁剪**——把局部视图约束在"手/夹爪–物体接触区域"周围,避免跨具身下随机裁剪过度关注背景相关性。教师始终看全图,学生偶尔看 RoInt 裁剪视图,以捕获与具身无关的交互语义。

**Stage 2(Pseudo-Robot → Real-Robot)**:配对 $\langle I_{P'}, I_R \rangle$,$I_R$ 为真机图像,$I_{P'}$ 为其伪机器人对应帧。用 Stage 1 得到的 $E_P$ 作教师、$E_R$ 作学生,两者都以 Stage 1 权重初始化。由于 $I_{P'}$ 与 $I_R$ 物理拓扑完全相同,只剩 sim-to-real 的光度差异,因此弃用 RoInt、改回标准 DINO 全局+局部多裁剪。

两阶段后得到共享隐空间 $E_H \approx E_P \approx E_R$。

**用大白话说**:人手和机器人长得差太远、又没有严格配对数据,没法直接把两者的特征"对齐"。于是先造一个"长得像机器人、但动作和人一模一样"的替身当中转站——人和替身天然配对(同一帧改绘),替身和真机器人又只差个渲染质感,两小步接力就把人的特征平滑地"过继"给了机器人编码器。

**蒸馏目标**(基于 DINOv3 自监督):

$$
\mathcal{L}_{total} = \mathcal{L}_{DINO} + \mathcal{L}_{iBOT} + \lambda \mathcal{L}_{KoLeo}, \quad \lambda = 0.1
$$

其中 $\mathcal{L}_{DINO}$ 是作用于全局 CLS token 的自蒸馏损失(教师–学生语义一致),$\mathcal{L}_{iBOT}$ 是掩码 patch 预测(重建教师 patch 特征,提供 dense 局部监督),$\mathcal{L}_{KoLeo}$ 是促进 batch 内特征均匀分布的微分熵正则。作者刻意**去掉 Gram loss**(原用于稳定 dense 表征),理由是其短程跨域蒸馏不会出现相应退化。学生处理局部裁剪时只用 $\mathcal{L}_{DINO}$,关闭另两项。

**HPP-5M 数据集(为 Stage 1 供数)**:从 DexYCB、TACO、OakInk、OakInk2 四个带参数化 3D 手姿标注的手-物交互数据集出发,视觉编辑流水线为:由指尖 3D 位置拟合夹爪腕部位姿与开合度、IK 求解机械臂构型 → 生成式 inpainting(ProPainter)抹去原图人手 → path-traced 渲染器把伪机器人渲染进场景。共得 **约 5M 严格配对帧 / 18K 视频序列**,覆盖单手/双手、单物体/物-物操作、allocentric/egocentric 视角。额外并入 **23k 帧目标域自由运动人类数据(占 HPP-5M 0.4%)**,在机器人平台上采集的任务无关"人类把玩"视频,经视觉编辑转为伪机器人观测后加入 Stage 1 训练,提供部署所需的场景先验。

**Pseudo-to-Real 配对(为 Stage 2 供数)**:只需弥合"同一机器人"渲染图与真图的光度差,故仅需少量自由移动的真机录制轨迹——按相同运动学合成伪机器人网格,渲染到原生真机图 $I_R$ 上得到几何完全一致的 $\langle I_{P'}, I_R \rangle$。

### 2.2 显式 3D 几何对齐:filter-and-fill

只对齐 2D 特征还不够——depth-aware 3D 策略还需几何一致的观测。LIDEA 用"过滤具身几何 + 填充统一夹爪"构造标准化(canonical)观测空间。

**具身几何过滤**:
- 人类侧(无 URDF):用基础分割模型 **GroundedSAM2** 从 RGB 提取手臂 mask,在深度反投影时剔除所有人体相关点,得到干净的无人背景点云 $P_H^{bg}$。
- 机器人侧:用本体感知——由实时关节角 + URDF 前向运动学确定各连杆空间构型,计算机器人占据体积(occupancy volume),剔除落入预设 margin 内的点,得无臂背景点云 $P_R^{bg}$。

**统一虚拟夹爪填充**:定义按开合度 $s$ 参数化的通用夹爪点云模板 $P^g(s)$。机器人侧的 TCP 位姿 $T_R$、开合度 $s_R$ 直接取自本体感知;人类侧用可泛化多视角手姿估计器 **POEM** 得 3D 手关节,再把指尖以最小二乘对齐到虚拟夹爪指尖,反解等效 TCP 位姿 $T_H$ 与开合度 $s_H$。变换到相机坐标系后填回背景,得混合 3D 观测:

$$
P_H^{hyb} = P_H^{bg} \cup \left( T_H \cdot P^g(s_H) \right), \quad P_R^{hyb} = P_R^{bg} \cup \left( T_R \cdot P^g(s_R) \right)
$$

**用大白话说**:先把点云里"是谁在操作"的部分(人手 or 机械臂)整块抠掉,只留下与谁无关的场景;再往里塞一个统一样式的"悬浮夹爪"当末端执行器代理。这样 3D 策略无论看人还是看机器人,眼里都是"一模一样的任务场景 + 同一个夹爪",几何层面彻底解耦了具身。

### 2.3 策略学习与部署

基座为 RISE-2(3D 扩散策略)。混合数据训练:人类样本走冻结 $E_H$(即 DINOv3)取 dense 2D 特征、$P_H^{hyb}$ 走 sparse 3D 编码器;机器人样本走冻结 $E_R$、$P_R^{hyb}$ 走共享 sparse 3D 编码器。因 $E_H \approx E_R$ 且 $P_H^{hyb}/P_R^{hyb}$ 共享标准几何,融合后的 2D dense 特征与 3D sparse token 送入 RISE-2 的 transformer 迭代去噪动作轨迹。**部署时只用机器人实时传感流,无需任何生成式视觉编辑**——原始图过 $E_R$、点云过滤后填虚拟夹爪即可。

## 三、实验结果

**硬件**:Flexiv Rizon 4 机械臂 + Robotiq 2F-85 夹爪 + 全局 Intel RealSense D415 RGB-D 相机,真机示教由触觉遥操作采集。
**四个真机任务**:Close Laptop(铰接)、Stack(6 DoF 抓放)、Fold Towel(可形变)、Prepare Bread(长程)。
**对比基线**:Pseudo-Robot baseline,代表主流视觉编辑方法(如 Masquerade / Phantom),把人类图编辑成伪机器人图后训练同一 3D 策略。命名约定:R = 机器人示教数,H = 人类示教数,P = 伪机器人。

**数据效率(Fig. 5,成功率 %)**——用少量真机 + 大量人类替代大量真机:

| 任务/阶段 | 少量真机 | 大量真机 | 视觉编辑基线 | LIDEA(混合) |
|---|---|---|---|---|
| Close Laptop I | R5=53 | R54=100 | — | R5+H54=100 |
| Close Laptop II | R5=40 | R54=93 | — | R5+H54=93 |
| Stack I(抓) | R20=50 | R72=93 | R20+P72=71 | R20+H72=86 |
| Stack II(放) | R20=50 | R72=93 | R20+P72=64 | R20+H72=80 |
| Prepare Bread I | R8=27 | R48=100 | R8+P48=33 | R8+H48=80 |
| Prepare Bread II | R8=0 | R48=73 | R8+P48=0 | R8+H48=53 |
| Prepare Bread III | R8=0 | R48=67 | R8+P48=0 | R8+H48=46 |

要点:① Close Laptop 用 5 条真机 + 54 条人类即追平 54 条真机(100/93);② 长程 Prepare Bread 上视觉编辑基线因伪机器人渲染累积的伪影与深度不一致而**崩到 33/0/0**,LIDEA 达 **80/53/46**,凸显显式几何对齐在多阶段执行中抑制几何混淆因子的价值;③ 作者据此主张人类数据可替代最多约 80% 的昂贵真机示教。

**OOD 泛化(Table I,Fold Towel;训练用标准蓝毛巾真机数据,测试换成新颖粉毛巾并放一条折叠蓝毛巾作强干扰物)**:

| 方法(OOD) | I | II | III | IV |
|---|---|---|---|---|
| 40 Robot | 36 | 27 | 18 | 18 |
| 40 Robot + 40 Human(Ours) | 63 | 54 | 27 | 27 |

纯真机因蓝毛巾干扰把注意力从功能目标引开,首角抓取仅 36% 并向下游级联失败;加入 OOD 人类演示后首角升至 63%、末阶段 18% → 27%,说明 LIDEA 能从多样人类视频学到外观鲁棒的交互线索。

**消融 —— 2D 特征蒸馏(Table II,Stack,20 Robot + 72 Human)**:

| 方法 | I(抓) | II(放) |
|---|---|---|
| Ours | 86 | 80 |
| w/ DINOv3(不蒸馏) | 20 | 20 |
| w/ 仅 Stage-1 蒸馏 | 67 | 60 |
| w/o 互联网预训练数据 | 73 | 67 |
| w/o 自由运动数据 | 33 | 33 |

不做跨具身蒸馏时近乎完全失败(20%),证明通用视觉表征把人手与机械臂编码为不同语义实体;仅 Stage-1 会遗留 Pseudo-to-Real 间隙(降 19/20 个点);去掉目标域自由运动数据直接塌到 33%——互联网数据缺乏部署所需的场景先验。

**消融 —— 3D 几何对齐(Table III,Stack)**:

| 方法 | I(抓) | II(放) |
|---|---|---|
| Ours | 86 | 80 |
| w/o Filter Human | 66 | 60 |
| w/o Filter Robot | 53 | 53 |
| w/o Filter Both | 40 | 40 |

两侧都不过滤时降 46/40 个点,甚至比纯机器人基线还差(负迁移);非对称过滤造成致命的训练-部署错配——只留机器人点云会让策略部署时直面未编辑的大块机械臂点云(降 33/27)。

**特征分析(Fig. 6)**:序列级余弦相似度上,对齐后的跨域曲线(green)紧贴机器人自身序列内曲线(blue,上界),而未对齐 DINO 的跨域曲线(purple)存在明显人-机间隙;PCA 显示对齐编码器把机器人末端执行器"同化"进类人语义空间;自注意力热图显示对齐后的机器人编码器聚焦于 Region-of-Interaction。

## 四、局限性

- **只映射到平行夹爪**:显式 3D 对齐把高自由度、多关节的人手统一映射为平行夹爪,丢失了人手丰富的几何构型;作者自陈人手几何本应更契合多指灵巧手。
- **依赖较重的合成流水线**:HPP-5M 需 IK 求解 + 生成式 inpainting + path-traced 渲染,伪机器人质量、抹除残留与光照一致性都可能影响 Stage 1 蒸馏上界;论文未系统量化合成噪声对最终策略的敏感度。
- **评测规模有限**:仅 4 个真机任务、单一 Flexiv+Robotiq 平台、单相机全局视角;跨机器人本体、跨相机布置的泛化未验证。人手侧依赖 POEM 手姿估计,遮挡严重或双手交叠时的 TCP 反解精度存疑。
- **"替代 80%"口径较宽**:该结论按不同任务的最佳配比综合得出,各任务的替代比例与绝对成功率并不一致(如 Stack 混合 86/80 仍低于全真机 93/93),需谨慎解读为上限而非普适。

## 五、评价与展望(纯学术视角)

**优点**。① 问题拆解干净:把跨具身鸿沟明确分解为"2D 语义"与"3D 几何"两条正交轴,分别用蒸馏和点云 filter-and-fill 解决,概念清晰、消融支撑充分(两张消融表都显示单去任一环节即大幅退化)。② "伪机器人传递桥"是本文最巧的一笔——用一个物理上可控合成的中间域,把"缺严格配对"的难题转化为两段各自有配对的子问题,思路可迁移到其他跨具身/跨传感器对齐场景。③ 显式几何对齐直击深度策略痛点:相比纯视觉编辑派(Phantom / Masquerade / H2R)只在像素域改绘、把渲染伪影一并喂给 depth-aware 策略,LIDEA 用"抠掉具身 + 填统一夹爪"从源头消除了 3D 负迁移,长程任务上的巨大差距(80/53/46 vs 33/0/0)很有说服力。④ 部署零额外开销(推理期不需生成式编辑),工程友好。

**与其他公开工作的关系**。本文与视觉编辑派(Phantom、Masquerade、H2R、AR2-D2)是直接竞争,并把它们收编为 Pseudo-Robot 基线做对照;与统一表征/tokenization 派(Being-H0、UniVLA、LAPA、Moto、EgoVLA、Humanoid Policy)是路线之争——后者直接在人类数据上 co-train 或统一动作 token,LIDEA 则主张先在特征与几何空间做严格对齐再喂固定策略。相较 object-centric 派(ViViDex、Vidbot、ZeroMimic),LIDEA 不依赖显式物体状态估计,对可形变/铰接物更稳。基座选择固定为 RISE-2/DINOv3 的做法值得称道——把变量锁定在"数据利用"而非"架构",使结论更可信。

**开放问题与可能改进**。① 灵巧手:平行夹爪抹平了人手最有价值的多指几何,把 filter-and-fill 从"填统一夹爪"扩展为"填参数化灵巧手模板"是最自然的下一步(作者也已列为 future work)。② 从 imitation 走向 VLA:aligned encoder $E_R$ 本质是一个"人-机语义对齐"的视觉骨干,用它去预训练 VLA 或 video-action 模型、把 HPP-5M 规模的对齐监督注入大模型,潜力可能超过当前小策略验证。③ 合成质量的闭环:可引入判别器或一致性正则,量化并抑制 inpainting/渲染噪声对 Stage 1 上界的侵蚀。④ 泛化验证:亟需跨本体、跨相机、更多任务的评测来支撑"通用人到机器人迁移范式"的更强主张。总体上,这是一篇动机清晰、方法优雅、实验扎实的跨具身迁移工作,其"合成中间域做传递蒸馏 + 几何解耦"的组合对从人类视频构建机器人预训练数据具有较强的方法论参考价值。

## 参考

1. Fang et al., *AirExo-2 / RISE-2: Scaling up generalizable robotic imitation learning with low-cost exoskeletons*, CoRL 2025.(本文固定基座 3D 扩散策略)
2. Luo et al., *Being-H0: Vision-language-action pretraining from large-scale human videos*, arXiv:2507.15597, 2025.(统一表征派代表)
3. Lepert et al., *Phantom: Training robots without robots using only human videos*, CoRL 2025.(视觉编辑派,本文主要对照基线)
4. Siméoni et al., *DINOv3*, arXiv:2508.10104, 2025.(蒸馏所依赖的视觉基础模型与自监督目标)
5. Yang et al., *Multi-view hand reconstruction with a point-embedded transformer (POEM)*, TPAMI 2025.(人类侧 3D 手姿估计,用于反解虚拟夹爪 TCP)
