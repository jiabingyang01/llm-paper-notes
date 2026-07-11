# ImMimic：通过映射与插值实现从人类视频的跨域模仿

> **论文**：*ImMimic: Cross-Domain Imitation from Human Videos via Mapping and Interpolation*
>
> **作者**：Yangcen Liu\*, Woo Chul Shin\*, Yunhai Han, Zhenyang Chen, Harish Ravichandar, Danfei Xu（\* 共同一作）
>
> **机构**：Georgia Institute of Technology（佐治亚理工学院 College of Computing）
>
> **发布时间**：2025 年 09 月（arXiv 2509.10952）
>
> **发表状态**：9th Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.10952) | [PDF](https://arxiv.org/pdf/2509.10952)
>
> **分类标签**：`从人类视频学习` `跨具身模仿` `域适应` `MixUp插值` `Diffusion Policy`

---

## 一句话总结

ImMimic 把海量人类操作视频当作**域适应监督** 而非普通预训练数据:先用手部重定向把人手轨迹映射到机器人动作空间,再用 DTW 在人类与机器人轨迹之间建立配对,最后沿配对做 MixUp 插值构造一条从人域到机器域的连续中间域,与少量机器人遥操作数据一起共训 Diffusion Policy;在 4 种末端执行器 × 4 个真机任务上,仅用**5 条机器人示范 + 100 条人类视频** 即可把多数任务的成功率推到接近满分,且轨迹更平滑。

## 一、问题与动机

从人类视频学机器人操作有规模与成本优势,但存在三重域鸿沟:视觉外观(人手 vs 各式夹爪/灵巧手)、形态结构(自由度/尺寸差异)、物理约束(运动学、装配方式)。作者把它归结为一个**域适应问题**:机器人(目标域)要模仿人类演示者(源域)的行为。

现有做法各有短板:

- **视觉预处理类**:把图像里的具身 mask 掉以消除视觉差异,或把动作空间限制为纯 3D 平移来回避具身动作鸿沟——都丢弃了信息。
- **潜空间对齐类**(如 XSkill、Flow 系):用无监督目标对齐两域视觉潜空间,但**只学了视觉,忽略了人手轨迹本身携带的动作信息**,动作解码器仍只从机器人数据学。
- **共训类**(如 EgoMimic):同时在人/机数据上训练,但依赖重预处理或简化动作空间,核心域偏移基本没被正面处理。

作者的三点核心洞察:(1) 重定向后的人手轨迹可以直接作为人类演示的**动作标签**,而不仅是视觉上下文;(2) 通过插值构造**中间域** 才能实现鲁棒的平滑适应;(3) 要做插值,先得在人/机数据之间建立**有效的配对映射**。

## 二、核心方法

整体是"重定向 → 共训 → 映射引导的 MixUp"三段式。策略骨干为 Diffusion Policy(ResNet18 编码图像 + 扩散去噪预测未来动作序列)。数据采集时人/机共用同一台 RealSense D435 固定视角,尽量压低视觉差异。

### 2.1 手部姿态重定向(Sec 3.1)

对每帧人类视频:MediaPipe 定位裁剪人手 → FrankMocap 输出 SMPL-X 的 21 个手关节局部 3D 坐标 → 将关节投影到深度图并求解 PnP 得到相机系下的手腕 6D 位姿。随后按 AnyTeleop 的思路,把人手关键点 $\mathbf{p}_t^i$ 优化映射为机器人关节角 $\mathbf{q}_t$:

$$\min_{\mathbf{q}_t} \sum_{i=1}^N \left\| \alpha\, \mathbf{p}_t^i - f_i(\mathbf{q}_t) \right\|^2 + \beta \left\| \mathbf{q}_t - \mathbf{q}_{t-1} \right\|^2, \quad \mathbf{q}_l \le \mathbf{q}_t \le \mathbf{q}_u$$

其中 $f_i$ 是机器人正运动学,$\alpha,\beta$ 分别平衡尺度对齐与时间平滑,关节角受上下限约束。

**用大白话说**:第一项让机器人手指尖尽量落在人手关键点缩放后的位置上(形状对齐),第二项不让相邻帧关节角跳变太猛(轨迹别抖),约束保证解在机器人物理可达范围内。这样人手视频就变成了一条"机器人能执行"的动作轨迹 $\mathbf{a}_t^{h\to r}$。

### 2.2 共训(Sec 3.2)

机器人分支:用长度 $\tau$ 的历史构造观测条件 $\mathbf{z}_t^r = [\,\mathbf{z}_{t-\tau:t}^{a,r} \,\|\, \mathbf{z}_{t-\tau:t}^{w,r} \,\|\, \mathbf{r}_{t-\tau:t}\,]$,即 agent-view 图像特征、wrist-view 图像特征(两个独立 ResNet18)与本体感知拼接。扩散策略 $\mathcal{P}_\phi$ 从含噪动作重构未来 $k$ 步动作,$\ell_2$ 损失:

$$\mathcal{L}_{\text{robot}}(\phi) = \sum_{i=1}^k \left\| \mathbf{a}_{t+i}^r - \hat{\mathbf{a}}_{t+i}^r \right\|^2$$

人类分支:因人类视频没有 wrist-view,条件里该项**零填充**,而把重定向动作补进条件——$\mathbf{z}_t^h = [\,\mathbf{z}_{t-\tau:t}^{a,h} \,\|\, \mathbf{0} \,\|\, \mathbf{a}_{t-\tau:t}^{h\to r}\,]$;重定向动作 $\mathbf{a}^{h\to r}$ 同时充当"未来动作标签"与"本体感知",算同样的 $\ell_2$ 损失 $\mathcal{L}_{\text{human}}$。总损失是两者等比例相加:$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{robot}} + \mathcal{L}_{\text{human}}$,每个 batch 中人/机数据各占一半。

**用大白话说**:同一个扩散策略,机器人数据教它"看真实机器人图像+本体感知怎么动",人类数据教它"看人手图像+重定向动作怎么动";人类那侧缺 wrist 视角就补零,但把重定向出来的动作塞进去当监督,让人手轨迹里的动作知识也参与训练动作解码器。

### 2.3 映射引导的 MixUp(Sec 3.3)——本文关键

只把两域数据混着训,潜空间里两域仍是分离的两团(见 t-SNE 可视化)。作者借 DLOW 的思路,在源域和目标域之间**人为造出一串中间域**,让两域落在一条平滑流形上。

**映射(Mapping)**:用 DTW 在人类演示 $\mathcal{D}^h$ 与机器人演示 $\mathcal{D}^r$ 之间建配对,$\mathcal{M}_{h\to r}(t)$ 表示人类时刻 $t$ 跨多条机器人演示所配到的机器人时刻集合。DTW 保证时间一致、避免不合理监督。两种映射距离:

- **动作映射(ImMimic-A)**:

$$d_{\text{act}} = \left\| \mathbf{t}^{h\to r} - \mathbf{t}^r \right\|_1 + \lambda_1 \left\| \mathbf{p}^{h\to r} - \mathbf{p}^r \right\|_1 + \lambda_2\, d_{\text{rot}}\!\left(\mathbf{o}^{h\to r}, \mathbf{o}^r\right)$$

即平移、手部位姿、朝向角距离的加权和($\mathbf{t}$ 平移、$\mathbf{p}$ 手姿、$\mathbf{o}$ 朝向、$d_{\text{rot}}$ 角距离)。

- **视觉映射(ImMimic-V)**:$d_{\text{vis}} = \|\mathbf{f}^{h\to r} - \mathbf{f}^r\|_2$,用预训练编码器视觉特征算距离。

**MixUp 插值**:对每个人类时刻 $t$,从 $\mathcal{M}_{h\to r}(t)$ 采一个机器人时刻 $t'$,在条件与预测动作上都线性混合:

$$\mathbf{z}_t^{\text{mix}} = \alpha \cdot \mathbf{z}_t^h + (1-\alpha)\cdot \mathbf{z}_{t'}^r, \qquad \mathbf{a}_{t:t+k}^{\text{mix}} = \alpha \cdot \mathbf{a}_{t:t+k}^{h\to r} + (1-\alpha)\cdot \mathbf{a}_{t':t'+k}^r$$

关键工程点:仿 DLOW 采用**渐进插值**——训练中让 $\alpha$ 沿线性时间表逐渐减小,从而把人类样本一步步"拉向"机器域,得到平滑的域流(消融显示线性调度优于从 $\beta$ 分布采 $\alpha$)。

**用大白话说**:先给每帧人手动作找到"最像它的那帧机器人动作"(用动作而不是画面来判断相似),然后把两者按比例掺在一起造出"半人半机"的中间样本一起训;训练早期掺得像人、后期越来越像机器人,策略就顺着这条渐变的路把人域知识平滑迁到机器域,而不是硬跳。

**推理**:按遥操作时长算出的上采样率 $\gamma$ 恢复机器人执行速度,配合衰减权重的 temporal ensembling 保证运动连续。

## 三、实验结果

硬件:Franka Emika Panda 机械臂 + 4 种末端——Robotiq 2F-85(二指夹爪)、Fin Ray/FR(二指软夹爪)、Allegro Hand(四指)、Ability Hand(五指)。任务:Pick and Place、Push(基础物体操作);Hammer、Flip(工具操作)。默认**5 条机器人示范 + 100 条人类视频**,每任务 10 次 rollout 计成功率。基线:Robot-only、两阶段 Fine-Tuning、Vanilla Co-Training、Random Mapping、ImMimic-V、ImMimic-A。指标:成功率 SR、平滑度 SPARC、动作距离 AD。

### 主表:成功率(Table 1,SR,4 具身 × 4 任务)

| 任务 | 方法 | Robotiq | FR | Allegro | Ability |
|---|---|---|---|---|---|
| Pick&Place | Robot Only | 0.40 | 1.00 | 0.00 | 0.80 |
| Pick&Place | Co-Training | 0.40 | 1.00 | 1.00 | 0.80 |
| Pick&Place | **ImMimic-A** | **1.00** | 1.00 | **1.00** | **1.00** |
| Push | Robot Only | 0.00 | 0.60 | 1.00 | 1.00 |
| Push | Co-Training | 0.20 | 0.60 | 1.00 | 1.00 |
| Push | **ImMimic-A** | **0.40** | **0.70** | 1.00 | 1.00 |
| Hammer | Robot Only | 0.20 | 0.90 | 0.00 | 0.00 |
| Hammer | Co-Training | 0.40 | 0.80 | 0.00 | 0.00 |
| Hammer | **ImMimic-A** | **0.50** | **1.00** | **0.20** | 0.00 |
| Flip | Robot Only | 0.60 | 0.60 | 0.00 | 0.60 |
| Flip | Co-Training | 0.60 | 0.80 | 0.00 | 0.90 |
| Flip | **ImMimic-A** | **1.00** | 0.80 | **0.20** | **1.00** |

ImMimic-A 在几乎所有具身/任务上追平或超过基线;Hammer-Ability(0.0)与 Flip/Hammer-Allegro(≤0.2)是硬件受限的失败点。

### 基线对比与消融

| 对比项(Robotiq/Ability,Pick&Place/Flip) | 关键数字 |
|---|---|
| Robot Only(Table 2) | Robotiq 0.40/0.60,Ability 0.80/0.60 |
| Fine-Tuning | Robotiq 0.80/0.70,Ability 0.50/0.40 |
| Random Mapping | Robotiq 0.40/0.50,Ability 0.80/0.50 |
| ImMimic-V(视觉映射) | Robotiq 1.00/0.50,Ability 0.90/0.40 |
| **ImMimic-A(动作映射)** | 全部**1.00** |
| STRAP(SOTA 检索基线,Table E.1) | Robotiq 0.50/0.60,Ability 0.90/0.90 |
| $\alpha$ 调度:$\beta$ 分布 vs 线性(Table E.2) | $\beta$ 分布 0.90~1.00;线性全**1.00** |

要点:动作映射 > 视觉映射 > 随机映射;渐进线性 $\alpha$ 优于 $\beta$ 分布采样;ImMimic-A 优于强检索基线 STRAP(其用 DINOv2 视觉特征做检索,未显式处理域鸿沟)。

### 平滑度、动作距离与样本效率

- **SPARC 平滑度(Table 3,越高越平滑)**:ImMimic-A 在 Robotiq(-9.44 vs Robot-Only -12.77)、Ability(-10.84)、Allegro(-13.89,Robot-Only 直接 N/A)上优于对照;仅 FR 上略逊于 Co-Training。
- **反直觉发现**:更像人手的具身不一定更好迁移。两个灵巧手的平均动作距离 AD 反而更大(Allegro 0.078、Ability 0.075),比两个夹爪(Robotiq 0.066、FR 0.065)还高——因装配方式、臂运动学也影响重定向。**AD 越小,人类视频带来的收益越大**。
- **鲁棒性(Fig 6 / Table C.1)**:在视觉/动作扰动下,动作映射 mIoU(0.70 基线、扰动后 0.67/0.63)显著比视觉映射(0.52、0.41/0.46)稳。
- **样本效率**:固定 100 条人类视频时,ImMimic-A 仅需 5 条机器人示范即达 1.0 SR,而 Robot-Only 用到 20 条仍不及;加 50 条人类视频可把 Robotiq Pick&Place 从 0.4 拉到 1.0。人类视频的类内多样性(intra-dataset AD 0.012)明显高于机器人数据(0.005)。

训练:300 epoch,A40 GPU,batch 128;部署 RTX 4090,推理与控制 30Hz。

## 四、局限性

1. **大域鸿沟下仍掉点**:当具身与人手的动作距离差异或视觉外观差异很大时,ImMimic 性能仍会退化(如 Hammer-Ability 归零),作者认为需更好的表征学习进一步对齐特征。
2. **跨具身增益不一致**:提升幅度随机器人结构设计而异,说明策略性能受硬件结构强影响——同一方法在不同末端上收益差别很大。
3. **依赖重定向质量**:整条流水线建立在 MediaPipe→FrankMocap→PnP→优化重定向之上,人手位姿估计误差会直接传导为动作标签噪声;论文未系统分析重定向失败对下游的影响。
4. **规模仍小**:验证局限于单臂、桌面级、每具身每任务 5 条机器人 + 100 条人类演示的受控采集(且人/机同视角同场景),距真正"互联网海量野生视频"仍有距离(附录 GMS-SDTW 只是对长视频检索的初步扩展)。

## 五、评价与展望(学术视角)

**优点**:

- **视角独到**:把人类视频重新定位为"域适应监督"而非辅助预训练数据,并明确提出"用动作相似度(而非视觉相似度)建配对更可靠"这一经验结论——附录用扰动下 mIoU 与长视频检索两组实验佐证,较有说服力,也修正了"潜空间视觉对齐"一派只学视觉的盲点。
- **方法组合干净**:DTW 配对 + MixUp 渐进插值 + Diffusion Policy 共训,三件都是成熟组件,工程可复现性高;渐进 $\alpha$ 调度把 DLOW 的域流思想落到了具身共训上,是有价值的迁移。
- **跨具身广度**:一套框架覆盖二指/软夹爪/四指/五指四种差异极大的末端,并给出"更像人手 ≠ 更好迁移"的反直觉证据,对末端执行器设计(加腕自由度、增大拇指长度、可调抓握间隙)也有启发。

**不足与开放问题**:

- **真机规模偏小**:每设置 10 次 rollout,成功率以 0.1 为粒度,统计噪声不可忽视;5+100 条的数据量更接近 few-shot 演示,难说明在大规模野生视频下结论是否保持。
- **与检索/共训路线的边界**:相比 EgoMimic(纯共训)、STRAP(检索)、XSkill/Flow(潜空间对齐),ImMimic 的增量主要在"映射+插值"这一步;但插值造出的"半人半机"样本在物理上是否可行、是否引入偏置,论文只做了定性 t-SNE,缺少对插值样本合理性的定量分析。
- **改进方向**:(a) 把动作映射距离学习化(而非手工加权 $\lambda_1,\lambda_2$),或用可学习的域判别器替代固定线性 $\alpha$ 调度;(b) 将重定向不确定性显式建模进 MixUp 权重,对低置信人手帧降权;(c) 扩展到双臂、移动操作与真正未剪辑的互联网视频(附录 GMS-SDTW 已是雏形);(d) 与更强的跨具身表征(如流/点云等具身无关表示)结合以攻克大域鸿沟下的掉点问题。

总体是一篇工程扎实、洞察清晰的 CoRL 实证工作:核心贡献不在单个新模块,而在"重定向动作即标签 + 映射引导渐进插值"这一组合范式,以及"动作映射优于视觉映射""更像人手不等于更好迁移"两条可复用的经验结论。

## 参考

1. Kareer et al. *EgoMimic: Scaling Imitation Learning via Egocentric Video*, 2024（最直接对标的人/机共训工作）。
2. Qin et al. *AnyTeleop: A General Vision-based Dexterous Robot Arm-Hand Teleoperation System*, 2023（重定向优化目标来源)。
3. Gong et al. *DLOW: Domain Flow for Adaptation and Generalization*, CVPR 2019(渐进插值/域流思想来源)。
4. Zhang et al. *mixup: Beyond Empirical Risk Minimization*, 2018(MixUp 插值基础)。
5. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, IJRR 2024(策略骨干)。
6. Memmel et al. *STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning*, 2024(检索式基线对照)。
