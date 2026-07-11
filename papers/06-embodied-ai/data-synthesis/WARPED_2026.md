# WARPED：面向从第一人称人类演示学习机器人策略的腕部对齐渲染

> **论文**：*WARPED: Wrist-Aligned Rendering for Robot Policy Learning from Egocentric Human Demonstrations*
>
> **作者**：Harry Freeman, Chung Hee Kim, George Kantor
>
> **机构**：Carnegie Mellon University Robotics Institute（卡内基梅隆大学机器人研究所）
>
> **发布时间**：2026 年 04 月（arXiv 2604.10809）
>
> **发表状态**：未录用（预印本，cs.RO）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.10809) | [PDF](https://arxiv.org/pdf/2604.10809)
>
> **分类标签**：`人到机器人` `第一人称视频` `Gaussian Splatting` `腕部视角合成` `模仿学习`

---

## 一句话总结

只用一台戴在头上的单目 RGB 相机（GoPro）拍第一人称人类演示，WARPED 通过"手-物联合优化"跟踪被操作物体的 6D 位姿、把人手轨迹 retarget 成机器人末端执行器（gripper）轨迹，再用 3D Gaussian Splatting 渲染出逼真的腕部相机视角，从而合成"腕部视角观测 + 动作"数据集训练 diffusion policy；在 5 个桌面操作任务上成功率与遥操作相当（Rotate Box 甚至 20/20 反超遥操 16/20），而数据采集时间只需遥操的 1/5～1/8。

## 一、问题与动机

模仿学习依赖大量高质量演示数据,而主流采集方式各有痛点:

- **遥操作(teleoperation)**:慢、费力、需要专用硬件(VR 头显、SpaceMouse、定制夹爪),难以快速扩展到新任务/新环境。论文测得单任务遥操采集时间在 15～32 分钟量级。
- **从人类视频学习**:虽然人手操作快、自然、采集成本低,但已有方法往往依赖**多视角相机、深度传感器、定制硬件或专门训练的生成模型**才能把人类观测转成机器人可用的观测/动作,门槛依旧很高。
- **视角错配**:大多数从人类演示学习的方法要求部署时的相机视角与采集时相似,因而多用于第三人称或固定机位策略;而**腕部相机(wrist-camera / eye-in-hand)**视角能捕捉更细粒度的交互细节、更受欢迎,却难以从人类第一人称视频直接得到。

WARPED 的目标:**只用一台单目 RGB 相机**(戴头上的 GoPro),无需多视角、深度、定制硬件或训练生成模型,就把第一人称人类演示视频转成**腕部相机视角对齐**的机器人观测-动作数据集,直接训练视觉运动策略。核心假设:物体刚性、桌面操作、场景准静态(除被操作物体外无显著变化)。

## 二、核心方法

整条流水线分五个阶段(见原文 Fig. 2):数据采集 → 交互式场景初始化 → 手-物优化 → retarget 与渲染 → 策略训练与部署。

### 1. 数据采集与场景重建

用户先拍一段不含被操作物体的工作空间短视频(通常不到 1 分钟),用 **SfM + LightGlue** 特征匹配估计相机位姿并重建稀疏几何,进而初始化整个场景的 **3D Gaussian Splat** 表示(类似 UMI 的前置扫描思路)。随后戴 GoPro Hero 9(按针孔相机建模)做人类演示,30 Hz 记录。

### 2. 交互式场景初始化

静态场景 splat 里没有手和物体的几何,需要额外初始化:

- **深度/定位**:用 Hierarchical Localization + LightGlue 把演示帧定位进静态场景估计相机位姿;用 **SpatialTrackerV2** 提取时序一致的单目深度图。由于 SfM 重建有全局尺度歧义,估计一个场景级尺度对齐把 splat 缩放到与预测深度一致。
- **手部初始化**:用 **HAMER** 逐帧估计手形与手姿(MANO 参数),再做序列级优化,让手姿与单目深度时序平滑一致。
- **物体初始化**:给一句文本描述,用 **Grounding DINO** 在首帧检测物体,**SAM2** 生成并传播分割掩码,**SAM3D** 重建物体初始网格(mesh)。作者不直接用 SAM3D 的高斯表示,而是把 mesh 多视角渲染后**自建物体 Gaussian Splat**(在演示相机轨迹下渲染保真度更高),最后用 **MegaPose** 得到初始 6D 位姿。

### 3. 手-物联合优化(方法核心)

不用深度传感器/多视角/物体扫描,而是**利用手与物在接触时提供互补几何约束**(尤其手遮挡物体时)来跟踪物体 6D 位姿。分两阶段(原文 Fig. 3)。

**(a) 物体位姿估计**:用可微渲染器 $\mathcal{R}$ 逐帧渲染物体的图像/掩码/深度,与 SAM2 掩码、单目深度做监督。关键的**遮挡感知掩码损失**:

$$\mathcal{L}_{\mathcal{M}_{obj}} = \left\| (\mathcal{M}_t^{obj} - \hat{\mathcal{M}}_t^{obj}) \odot (1 - \hat{\mathcal{M}}_t^{hand}) \right\|$$

> 用大白话说:渲染出的物体掩码要贴合观测掩码,但被手挡住的像素(手掩码 $\hat{\mathcal{M}}^{hand}$ 为 1)用 $(1-\hat{\mathcal{M}}^{hand})$ 抹掉不计,免得手挡住时逼物体去乱对齐。

由于相似掩码可能对应多种位姿(手遮挡时更甚),再加**深度一致性损失** $\mathcal{L}_{\mathcal{D}_{obj}}$;针对单目深度的稀疏与噪声,还加一个 **DINOv2 特征相似度损失**(ViT-S 骨干、取第九层特征、渲染图与掩码原图算余弦相似度)。物体位姿逐帧优化:

$$\min_{\mathbf{R}_t^{obj}, \mathbf{t}_t^{obj}} \lambda_{\mathcal{M}_{obj}} \mathcal{L}_{\mathcal{M}_{obj}} + \lambda_{\mathcal{D}_{obj}} \mathcal{L}_{\mathcal{D}_{obj}} + \lambda_{\text{DINO}} \mathcal{L}_{\text{DINO}}$$

> 用大白话说:三路信号——轮廓(掩码)、远近(深度)、语义纹理(DINOv2 特征)——共同把物体的旋转平移拟合到位,单靠掩码容易被遮挡骗到。

**(b) 手-物联合精修**:独立估计的手/物位姿不够准,于是把所有帧的物体旋转/平移/全局尺度 $\Theta^{obj}$ 与 MANO 手参数 $\Theta^{hand}$ 一起联合优化,施加交互约束:

- **接触损失**:接触区间 $t_s \le t \le t_e$ 内,鼓励常接触指尖顶点 $V^{tip}$ 贴近物体顶点

$$\mathcal{L}_{\text{contact}}(t) = \sum_{\mathbf{v}^\tau \in V^{tip}} \min_{\mathbf{v}^o \in V^{obj}} \|\mathbf{v}_t^\tau - \mathbf{v}_t^o\|_2 , \quad t_s \le t \le t_e$$

> 用大白话说:抓握时手指该贴着物体,就把指尖到最近物体点的距离压到最小。

- **碰撞损失**:用物体的截断符号距离场(TSDF)$\Phi^{obj}$ 惩罚穿模的手顶点

$$\mathcal{L}_{\text{col}} = \sum_{\mathbf{v}^h \in V^{hand}} \Phi^{obj}(\mathbf{v}^h)$$

> 用大白话说:手不该插进物体内部,凡是落在物体体积里的手顶点都罚。

- 另有**稳定抓握损失** $\mathcal{L}_{\text{sg}}$(接触时指尖到物体顶点距离保持恒定)、双向遮挡感知掩码损失、深度损失。总目标:

$$\min_{\Theta} \lambda_{\mathcal{M}}\mathcal{L}_{\mathcal{M}} + \lambda_{\mathcal{D}}\mathcal{L}_{\mathcal{D}} + \lambda_c \mathcal{L}_{\text{contact}} + \lambda_{\text{col}}\mathcal{L}_{\text{col}} + \lambda_{\text{sg}}\mathcal{L}_{\text{sg}} + \mathcal{L}_{\text{aux}}$$

### 4. Retarget 与渲染

从完整手-物轨迹里抽稀疏均匀关键帧:

- **接触前**($t < t_s$):gripper 张开,末端执行器位姿由拇指与食指关节映射得到(原文 Fig. 4b);沿用 Pan et al. 的接触前轨迹优化(funnel 损失贴近原轨迹 + 碰撞损失防夹爪撞物 + 平滑损失)。
- **接触起始帧** $t_s$:用拇指/食指指尖各 50 个接触点,精修末端执行器位置与夹爪开度,得到物理上合理的抓握(原文 Fig. 4c-f)。
- **接触中**($t_s \le t \le t_e$):夹爪与物体相对固定,末端执行器**刚性跟随物体运动**:

$$\mathbf{T}_t^{ee} = \mathbf{T}_{t_s}^{ee} \left(\mathbf{T}_{t_s}^{obj}\right)^{-1} \mathbf{T}_t^{obj}$$

> 用大白话说:一旦抓稳,夹爪就像焊在物体上,物体怎么动夹爪就怎么动,保持既定抓握关系。

关键帧插值成连续轨迹后,把**场景 splat + 物体 splat + 末端执行器**三者的高斯组合起来,用 Nerfstudio 的 **3DGUT** 渲染出**鱼眼腕部相机视角**图像(原文 Fig. 5 展示了渲染与真实腕部视角的对比)。

### 5. 策略训练与部署

训练 **diffusion policy**,以腕部视角图像 + 机器人本体感知为条件,输出相对位姿与夹爪动作 chunk。为缩小渲染图与真实机器人图之间的 sim-to-real 差距,训练时给输入图加高斯噪声。数据增强(原文 Fig. 6):物体网格重贴图(retexture)、随机平移物体、随机化初始夹爪位姿、随机缩放场景、扰动腕部相机内外参。每任务 30 条演示,增强 10 倍。部署在 UFactory xArm7 + xArm G1 夹爪上,夹爪上方装 GoPro Hero 9 + Max Lens Mod 1.0 取腕部视角。

## 三、实验结果

**实验设置**:5 个桌面任务(Rotate Box 旋盒 90°、Pour Mug 倒杯、Bottle from Rack 从架取瓶、Wipe Brush 刷子擦盘、Can on Plate 罐放盘);每任务 30 条人类演示(30 Hz,增强 10 倍);diffusion policy 在 4×H100 上训练;每任务 20 次 rollout(10 Hz),成功率为指标。

**主结果(Table I,成功率 x/20)**:

| 方法 | Rotate Box | Pour Mug | Bottle from Rack | Wipe Brush | Can on Plate |
| --- | --- | --- | --- | --- | --- |
| Teleoperation(遥操) | 16/20 | 19/20 | 16/20 | **15/20** | 19/20 |
| Alter (Heng et al.) | 7/20 | 3/20 | 0/20 | 0/20 | 8/20 |
| WARPED(无增强) | 0/20 | 17/20 | 0/20 | 0/20 | 8/20 |
| WARPED(带背景干扰物) | 18/20 | 15/20 | **17/20** | 9/20 | 17/20 |
| **WARPED** | **20/20** | 18/20 | **17/20** | 11/20 | 17/20 |
| Teleoperation + WARPED | **20/20** | **20/20** | **17/20** | 11/20 | **20/20** |

- WARPED 在 Pour Mug、Bottle from Rack、Can on Plate 上与遥操**相当**,在 **Rotate Box 上反超**(20/20 vs 16/20)——作者归因于精细旋转控制对遥操很难,而人手自然运动更顺滑。唯一明显落后的是 **Wipe Brush**(11/20 vs 15/20),因刷子小且平贴桌面,位姿估计与 retarget 更难。
- **消融**:去掉数据增强(WARPED 无增强)会在 3/5 任务上完全失败(0/20),Can on Plate 也掉一半以上——增强是弥合 sim-to-real 的关键。
- **基线 Alter**(把人手视频 inpaint 后叠加夹爪掩码,不做精确手-物几何)全面很差,说明"不精确的几何直接渲染腕部图"不足以训练策略。

**新物体泛化(Table II,x/10)**:

| 方法 | 新物体 | Rotate Box | Bottle from Rack | Wipe Brush | Can on Plate |
| --- | --- | --- | --- | --- | --- |
| Teleoperation | Obj 1 | 8/10 | 4/10 | **7/10** | 9/10 |
| Teleoperation | Obj 2 | 2/10 | 2/10 | 4/10 | 9/10 |
| **WARPED** | Obj 1 | **10/10** | **8/10** | **7/10** | **10/10** |
| **WARPED** | Obj 2 | **8/10** | **5/10** | 2/10 | 9/10 |

WARPED 在除 Wipe Brush 外的所有任务、新物体上都优于遥操,Rotate Box、Bottle from Rack 上差距最明显——作者认为得益于增强带来的几何多样性。

**采集效率(Table III,MM:SS)**:

| 方法 | Rotate Box | Pour Mug | Bottle from Rack | Wipe Brush | Can on Plate |
| --- | --- | --- | --- | --- | --- |
| Teleoperation | 22:51 | 24:59 | 31:49 | 30:16 | 15:20 |
| **WARPED** | **3:37** | **3:18** | **3:27** | **5:19** | **3:41** |

WARPED 采集(含场景扫描)比遥操**快约 5～8 倍**。

**与 UMI 对比(Table IV,x/10)**:

| 方法 | 物体 | Can on Plate | Rotate Box |
| --- | --- | --- | --- |
| UMI | Train | 9/10 | 2/10 |
| UMI | Novel | 10/10 | 0/10 |
| **WARPED** | Train | 8/10 | **10/10** |
| **WARPED** | Novel | 9/10 | **10/10** |

Can on Plate 上 UMI 略好(圆柱铝罐外观一致、演示更代表性);但 **Rotate Box 上 UMI 惨败**(训练物 2/10、新物 0/10),因为旋转靠末端执行工具做很难且对打滑敏感,人手演示的旋转更平滑,加上 UMI 框架不支持 WARPED 那类增强,只能用原始 30 条。

**其他发现**:

- **手-物优化 vs FoundationPose 位姿跟踪**:用 WARPED 手-物优化,Rotate Box、Can on Plate 均 17/20;换成 FoundationPose 分别掉到 11/20、2/20(单目深度噪声、相机内参不准、mesh 错配,且物体小又被手挡时跟踪失败)。交互约束显著提升遮挡下的跟踪精度。
- **背景干扰物**:Can on Plate、Bottle from Rack 几乎不受影响,Rotate Box/Wipe Brush 小幅下降,Pour Mug 掉得最多(主要是抓握失败)。
- **分布外场景**:Can on Plate 收 50 条演示、跨 20 个多样桌面设置,在 4 个未见场景上 16/20。
- **协同训练**:15 条遥操 + 15 条 WARPED 混训,在 4/5 任务上追平或超过纯遥操(Wipe Brush 例外,因运动更复杂、WARPED 轨迹与遥操执行差异大导致不一致)。

## 四、局限性

作者明确列出:

1. **仅支持刚性物体运动**,无法处理铰接体(articulated)与可变形物体(可考虑蒸馏 DINOv2 特征或可变形高斯来扩展)。
2. **假设准静态场景**——只有被操作物体在动;要处理移动场景需引入动态 Gaussian Splatting。
3. **物体被完全遮挡时流水线失败**(这也是当前 SOTA 物体跟踪的共性问题)。
4. Wipe Brush 这类小而平贴桌面的物体,位姿估计与 retarget 精度不足,策略明显落后遥操。
5. 依赖多个视觉基础模型(HAMER/SAM2/SAM3D/MegaPose/Grounding DINO/DINOv2/SpatialTrackerV2/MegaPose)串联,任一环节误差会传播;整条离线优化流水线计算成本与鲁棒性论文未量化。

## 五、评价与展望

**优点**:

- **门槛低是最大卖点**:只需一台戴头上的单目 RGB 相机 + 一段静态扫描,不要多视角/深度/定制夹爪/训练生成模型,采集比遥操快 5～8 倍,对新任务扩展友好。相比 UMI 需要定制手持夹爪、相比 Phantom/Masquerade 需要图像编辑或 inpaint,WARPED 走的是"显式几何重建 + 物理渲染"路线。
- **手-物联合优化很扎实**:把接触/碰撞/稳定抓握等交互约束当作互补几何信号来对抗单目歧义与手遮挡,消融显示它对遮挡下位姿跟踪的收益远超 FoundationPose,这是本文相对同类工作(RwoR、Phantom、Masquerade、Real2Render2Real 等)的方法学贡献。
- **腕部视角合成 + 强增强**是能落地的组合:合成 splat 让"重贴图/平移/缩放/扰动内外参"等增强变得廉价,直接支撑了新物体泛化与分布外场景的表现。

**缺点与开放问题**:

- **Rotate Box 反超遥操** 更多反映"遥操本身在精细旋转上很弱",而非 WARPED 通用性强;在 Wipe Brush 这类需精确接触位姿的任务上,显式几何路线的短板暴露无遗。
- **准静态 + 单刚体**假设把适用范围限制在简单桌面 pick-place/pour/rotate;真实长程、多物体、铰接/柔性任务是主要开放问题。
- **误差传播与鲁棒性未量化**:整条流水线串了 7～8 个基础模型,论文只给了终端策略成功率,没有报告位姿跟踪误差分布、失败率、单条演示处理耗时,难以判断规模化时的稳定性。
- **与生成式路线的取舍**:相对 RwoR/Phantom 等"直接学一个人→机图像转换"的生成式方法,WARPED 的显式几何更可解释、可注入物理约束,但也更脆(依赖 mesh 重建与位姿跟踪);两条路线的边界(何时该用显式几何、何时该用生成模型补全)是值得进一步研究的方向。

**可能改进方向**:引入动态 Gaussian Splatting 支持移动/多物体场景;用可变形高斯或蒸馏特征扩展到铰接/柔性体;把接触前 retarget 与抓握精修做得更鲁棒以攻克 Wipe Brush 类任务;把整条离线优化与在线策略闭环,做主动纠错以缓解跟踪失败与遮挡问题。

## 参考

1. Chi et al., *Universal Manipulation Interface (UMI)*, RSS 2024 —— 本文对标的手持采集范式与前置场景扫描思路。
2. Chi et al., *Diffusion Policy*, RSS 2023 —— 本文策略骨干。
3. Pavlakos et al., *HAMER: Reconstructing Hands in 3D with Transformers*, CVPR 2024 —— 手姿初始化。
4. Heng et al., *RwoR: Generating Robot Demonstrations from Human Hand Collection*, 2025 —— 无机器人的人→机演示生成,相关基线(Alter setup)。
5. Wu et al., *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting*, CVPR 2025 —— 腕部鱼眼视角渲染实现。
