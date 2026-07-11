# Human-as-Humanoid：用人类对齐本体从第一/第三人称人类视频实现人形零样本学习

> **论文**：*Human-as-Humanoid: Enabling Zero-Shot Humanoid Learning from Ego-Exo Human Videos with Human-Aligned Embodiments*
>
> **作者**：Xiaopeng Lin, Ruoqi Yang, Shijie Lian, Zhaolong Shen, Bin Yu et al.（通讯作者 Kai Chen）
>
> **机构**：The Hong Kong University of Science and Technology (Guangzhou)、DeepCybo、ZGCA、ZGCI、Harbin Institute of Technology、Huazhong University of Science and Technology、Beihang University
>
> **发布时间**：2026 年 07 月（arXiv 2606.32009）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.32009) | [PDF](https://arxiv.org/pdf/2606.32009)
>
> **分类标签**：`human-to-humanoid` `ego-exo video` `VLA pretraining` `dexterous manipulation` `retargeting`

---

## 一句话总结

针对高自由度人形 VLA 缺可执行动作监督的痛点,该工作先设计一个与成年男性尺度对齐的 60-DoF 上身人形 PrimeU,再用近实时(约 20 FPS)的 ego-exo 人类视频到动作流水线(mesh 重建 + 分部位 staged IK)把纯相机人类演示转成控制器对齐的 60-DoF 动作块,配合 FK 感知的 flow-matching 策略 PhysDex,实现相对遥操作 **4.8–7.2 倍** 的原始数据吞吐,并在 7 个真机任务上零目标任务机器人演示即超过 GR00T N1.7。

## 一、问题与动机

VLA 策略把视觉-语言观测映射为机器人动作,其性能不仅取决于视觉-语言表征,更取决于观测-动作监督的规模与质量。对高自由度灵巧人形而言,动作标签维度高、且强绑定机器人执行接口(关节顺序、URDF 约定、关节限位、控制器),因此获取可部署监督尤其困难:

- **遥操作** 给的是控制器对齐、天然可执行的监督,但吞吐低——高 DoF 人形轨迹采集慢、劳动密集、受安全约束、难以跨场景多样化;
- **人类第一人称视频** 采集快、行为丰富(手形变化、接触时机、双手协调、重抓),且第一人称视角接近人形策略的头视观测,但原始视频不含目标机器人动作标签,且人和人形在体尺、关节结构、手形态、自由度、传感视角、可达工作空间上都有差异。

作者把"人类演示 → 机器人可学监督"拆成四个耦合需求:(i) **本体对齐**(减小体尺/工作空间/手形态/视角带来的 retarget 误差);(ii) **观测-动作兼容**(第一人称做部署对齐的策略输入,第三人称做遮挡鲁棒的运动恢复);(iii) **动作接口对齐**(转出的标签要兼容关节顺序/URDF/限位/控制器,而不只是任务空间意图);(iv) **关节-任务一致**(可执行关节指令要保住接触操作关键的腕/指尖几何)。核心数据效率问题因此是:如何在不丢失可执行动作监督的前提下,降低对机器人端遥操作的依赖。

## 二、核心方法

整个系统 **从机器人本体出发**,而非把人到机器人迁移当作纯粹的事后 retarget 问题。三块拼图:PrimeU(人类对齐本体)、Human-as-Humanoid(近实时视频到动作流水线)、PhysDex(FK 感知 VLA)。

### 2.1 PrimeU:人类对齐的 60-DoF 上身本体

上身按成年男性操作尺度设计,统一动作空间为 60-DoF:两条 7-DoF 手臂(14)、两只 20-DoF Wuji 灵巧手(40)、3-DoF 脖子、3-DoF 腰。头视 + 腕视用 Intel RealSense D435,视角结构对齐部署时策略输入。统一机器人关节向量为:

$$q = \left[\, q^{L}_{arm},\, q^{L}_{hand},\, q_{neck},\, q_{waist},\, q^{R}_{arm},\, q^{R}_{hand} \,\right] \in \mathbb{R}^{60}$$

**用大白话说**：把左右臂、左右灵巧手、脖子、腰拼成一个 60 维关节向量,retarget、训练、部署三处共用同一套 URDF 与关节顺序,于是腕位置、指尖位置、关节限位在全流程里含义一致——本体不是拿来近似人体解剖,而是让肩宽、臂展、手长贴近人的操作尺度(见 Table 1),把 retarget 需要外推的量压到最小。

### 2.2 Human-as-Humanoid:近实时动作标签构造

四阶段:① 在第三人称视频里跟踪演示者并在短时窗内传播 mask;② mesh 感知的人体重建恢复上身位姿和手部运动(mesh 只作运动估计中间量);③ 骨架在 root-relative 坐标系平滑并映射到选定运动学约定,对齐坐标轴、插补躯干/脖子关节、为双手建 palm frame;④ staged IK 把骨架 retarget 成 PrimeU 的 60-DoF 控制器对齐标签。输出不是人体位姿标注,而是机器人训练元组 $\langle o_t, \ell, q_t, q^{*}_{t+1:t+H} \rangle$(第一人称观测、语言指令、当前机器人状态、未来动作块)。

**分部位 staged IK** 是关键——整体 60-DoF 一锅 IK 会把手指/腕/臂/脖/腰全耦合、病态。对每个身体部位 $b$ 定义目标向量 $y^{*}_{b}$ 和运动学映射 $f_b$,求带正则的 IK:

$$q^{*}_{b} = \arg\min_{q_b \in \mathcal{Q}_b} \left\| W_b\!\left( f_b(q_b;\bar{q}) - y^{*}_{b}\right)\right\|_2^2 + \lambda_b \left\| q_b - q^{0}_{b}\right\|_2^2$$

顺序为:手 → 臂和腕 → 脖和腰 → 守护与平滑。手先解(指尖点 + 指段方向,先做每指相似变换对齐再 Levenberg–Marquardt);臂用阻尼 Jacobian IK,腕朝向从手 frame 抽取并在臂解后精修;脖/腰用语义头/躯干 frame 的 SO(3) 残差 $e^{ori}(q_b)=\mathrm{Log}(R_b(q_b)^\top R^{*}_b)$。

**用大白话说**：像拼装一样一段段解 IK,每段只动自己那几个关节,用 FK 预测的末端几何去贴人体目标,再用正则项把解拉住别偏离上一帧种子,避免一锅解的数值灾难;先解手保证接触几何,再解臂腕保证一致,最后脖腰扩工作空间。

局部腕-掌精修用 **守护规则**,只在确实降低该部位任务空间残差时才采纳:

$$q^{new} = \begin{cases} \tilde{q}, & E_b(\tilde{q}) < E_b(q^{base}) - \epsilon \\ q^{base}, & \text{otherwise} \end{cases}$$

**用大白话说**：局部微调"帮倒忙"就不要,只有把误差真降下来才替换,防止求解器把本来不错的标签改坏。得到的是机器人中心动作语料而非人体运动语料。

### 2.3 PhysDex:从人类衍生高 DoF 动作学 VLA

策略预测未来相对关节增量,FK 与位姿损失时解码回绝对关节:$\hat{q}_{t+h}=q_t+\hat{A}_h$、$q^{*}_{t+h}=q_t+A^{*}_h$($h=1,\dots,H$,实现中 chunk 含 40 个未来状态)。动作全程在关节空间表示(避免部署时 IK、保住多指手零空间结构、让脖腰与臂手同属一套动作约定)。

- **动作骨干**:第一人称图与指令由 PhysBrain(一个在大规模第一人称人类数据上训练、面向操作的 VLM)编码为视觉-语言 token;本体感受和扩散噪声直接进动作模型不经 VLM。动作骨干是条件 flow-matching DiT,在线性路径 $z_\tau=(1-\tau)z_0+\tau A^{*}$ 上训练,目标速度 $A^{*}-z_0$:

$$\mathcal{L}_{fm} = \mathbb{E}_{A^{*},z_0,\tau}\!\left[\,\left\| v_\theta(z_\tau,\tau,q_t,h_\phi) - (A^{*}-z_0)\right\|_2^2\,\right]$$

**用大白话说**：在噪声块和目标动作块之间学一条"直线"速度场,推理时只需 4 步积分就采样出 60-DoF 动作块,直接送控制器。

- **DS-HKC(Dual-Space Hierarchical Kinematic Constraint)**:解决"策略必须输出可执行关节,但操作成败常由腕运动、指尖落点、接触几何决定"的空间错配。用 PrimeU URDF 诱导的可微 FK 把关节映到任务空间:

$$\mathcal{F}_{U}(q) = \left(W(q),\, R(q),\, P(q)\right)$$

($W$ 两腕位置、$R$ 两腕朝向、$P$ 每手五指尖位置)。分两级层级监督:腕级约束近端手位姿 $\mathcal{L}_{wrist}$、指尖级约束远端接触几何 $\mathcal{L}_{tip}$,再加关节限位可行性 $\mathcal{L}_{lim}$:

$$\mathcal{L}_{dshkc} = \lambda_{wrist}\mathcal{L}_{wrist} + \lambda_{tip}\mathcal{L}_{tip} + \lambda_{lim}\mathcal{L}_{lim}$$

这些任务空间量不需额外标注,直接对预测/目标关节轨迹套同一 FK 得到。其为何有效可从梯度看——任务空间平方误差对关节的梯度是:

$$\nabla_q \|f_{U}(q)-f_{U}(q^{*})\|_2^2 = 2\, J(q)^\top\!\left(f_{U}(q)-f_{U}(q^{*})\right)$$

**用大白话说**：任务空间误差经 Jacobian 把梯度按 $J(q)^\top J(q)$ 重新加权分配到各关节,那些"一动就让腕/指尖大幅偏移"的关节获得更强纠正梯度,正好补上纯关节回归对接触几何不敏感的短板;因此 DS-HKC 与关节空间回归互补,而非只是多加个惩罚项。

- **总目标**(含 flow matching、绝对位姿、关节增量、平滑、DS-HKC,后者带 warm-up 系数 $\alpha(s)$):

$$\mathcal{L} = \lambda_{fm}\mathcal{L}_{fm} + \lambda_{abs}\mathcal{L}_{abs} + \lambda_{\Delta}\mathcal{L}_{\Delta} + \lambda_{sm}\mathcal{L}_{sm} + \alpha(s)\,\mathcal{L}_{dshkc}$$

预训练语料为 **1,500 小时** 自采 ego-exo 人类演示,全部经 staged-IK 转成 60-DoF 机器人动作标签——即预训练监督不只是视觉模仿,而是目标关节顺序/URDF/限位/控制器接口下的高 DoF 机器人动作监督。

## 三、实验结果

围绕四问:本体是否把人机形态差压到可稳定转换?纯相机 ego-exo 运动恢复是否够稳(相对可穿戴动捕)?人类衍生 60-DoF 动作块是否兼容真机 PrimeU 轨迹?FK 感知训练是否改善任务空间几何同时保住可执行关节动作与目标任务部署?

### 本体尺度对齐(Table 1,数值来自 URDF 运动学树与视觉网格)

| 维度 | Human (cm) | PrimeU (cm) | 比值 |
|---|---|---|---|
| Shoulder breadth 肩宽 | 41.5 | 40.4 | 0.97 |
| Shoulder-to-head height 肩到头高 | 31.5 | 37.1 | 1.18 |
| Shoulder-to-middle-fingertip reach 肩到中指尖臂展 | 78.6 | 80.3 | 1.02 |
| Hand length 手长 | 19.3 | 19.3 | 1.00 |

与操作最相关的肩宽、臂展、手长都接近 1.0;肩到头高偏大(1.18)只影响头相机安装高度,对操作可达性影响小。

### 运动恢复吞吐与稳定性

流水线约 20 FPS(接近常见 15 Hz 采集率),在 15 Hz 采集设定下相对动捕遥操作取得 **4.8–7.2 倍** 原始演示吞吐增益。投影诊断显示:可穿戴惯性动捕在投影视图有明显定位漂移(近距双手操作尤甚,常需操作员补偿或反复标定,压低采集吞吐);ego-exo 恢复的骨架与可见人体/手投影对齐更紧,即用第三人称提供稳定几何证据做运动恢复的同时,不必用第三人称替换部署对齐的第一人称策略观测。

### 动作接口兼容诊断(Table 3,单 60-DoF tokenizer,100 个真机评估窗口;Lower 越低越好)

| 诊断设置 | 训练数据 | 评估 | EE 误差 (mm) mean/p95 | Norm MAE mean/p95 |
|---|---|---|---|---|
| Cross-domain | Human only | Robot | 5.34 / 12.67 | 0.0080 / 0.0097 |
| In-domain baseline | Robot only | Robot | 4.09 / 6.84 | 0.0099 / 0.0117 |
| Mixed-domain | Robot + human | Robot | 4.86 / 9.11 | 0.0096 / 0.0114 |

关键读法:仅用人类衍生动作训练的 tokenizer,去重建从未见过的真机轨迹(严格跨域),平均归一化 MAE 仅 0.0080、末端误差 5.34 mm——说明人到 PrimeU 转换出的动作占据一个贴近真机演示的动作流形。作者强调归一化 MAE 只作动作空间兼容诊断、不作跨设定严格排名(因其还依赖动作方差尺度与各分布归一化),故以末端误差为主指标:三种设定末端误差都在毫米级。

### FK 感知训练(Figure 7)

同训练预算下,加 FK 监督不只是加辅助惩罚,而是改善高 DoF 动作学习的优化几何:纯关节 flow 目标把 60 维大体当独立回归目标,FK 监督经机器人运动链把关节耦合、并度量诱导的腕/指尖是否符合任务空间几何。损失曲线上,PhysBrain 初始化的 FK 感知模型 PhysDex 达到最低 loss,低于 PhysDex-wo-FK 与 GR00T N1.7。

### 真机部署(Figure 9,stage-final 复合分 = 有序阶段完成 + 最终成功,每任务 10 次 rollout)

| 任务 | 适配范式 | PhysDex | GR00T N1.7 |
|---|---|---|---|
| Ring placement 套环 | 仅人类演示 | **62.5%** | 51.1% |
| Magic-cube packing 装魔方入袋 | 仅人类演示 | **77.5%** | 65.8% |
| Water pouring 倒水 | 仅人类演示 | **38.0%** | 32.1% |
| Cup stacking 叠杯 | 仅人类演示 | **44.0%** | 38.8% |
| Temperature-gun measurement 测温枪 | 少量真机 | **38.3%** | 32.8% |
| Light-bulb loosening 拧灯泡 | 少量真机 | **58.0%** | 53.7% |
| Bottle-cap loosening 拧瓶盖 | 少量真机 | **44.0%** | 39.4% |

PhysDex 在全部 7 个任务复合分均高于 GR00T N1.7。提升在 **仅用目标任务人类演示、零目标任务机器人演示** 的范式(套环/装魔方/叠杯/倒水)更显著;在 **少量真机数据** 做中训练锚定 + 任务后训练的范式(测温枪/拧灯泡/拧瓶盖,均为强接触力任务)提升较小——这与人类衍生标签"提供本体特定、控制器对齐的高 DoF 动作先验、但缺接触力"的定位一致。作者明确说明每任务仅 10 次 rollout,结论作为受控初步对比而非最终统计断言。

## 四、局限性

作者在讨论中列出:

1. **位姿估计质量上界了 retarget 质量**——系统性的位姿估计失败会变成动作标签偏置;
2. **IK 质量上界了策略质量**——人类衍生动作继承 retarget 用的机器人模型、关节限位与标定;
3. **绑定特定 URDF/关节约定**——迁到新本体需重新 retarget 并调整动作维度;
4. **人类衍生动作只捕捉运动学、不直接捕捉接触力**,故接触密集设定仍需机器人数据做锚定/评估/最终适配,细粒度灵巧操作(拧盖、拧灯泡、按钮依赖接触力/摩擦/局部滑移/指尖落点/手-物形态)尤甚——retarget 手指位姿的小误差就可能改变任务结果,人手与机器手的形态差单靠骨架对齐消不掉。

因此"zero-shot"仅指 **目标任务部署不需目标任务机器人演示**,并非消除所有机器人特定建模假设。

## 五、评价与展望(纯学术视角)

**优点**:(1) 把"人到人形迁移"前置成本体设计问题(PrimeU 对齐人体操作尺度),从源头压 retarget 外推,这一"从本体出发"的取向比纯事后 retarget 更系统,Table 1 让对齐可量化核验,是值得借鉴的实验规范;(2) 分部位 staged IK + 守护规则是工程上稳健的高 DoF retarget 方案,回避了 60-DoF 一锅 IK 的病态;(3) DS-HKC 用可微 FK 在关节/任务双空间监督,Eq 11 的 Jacobian 加权解释清楚地论证了"为何任务空间约束能补关节回归短板",这是全文最扎实的方法论点;(4) 跨域 tokenizer 重建(仅人类数据重建真机轨迹达毫米级)是一个巧妙且可核验的"动作接口兼容性"诊断,比单看策略成功率更能隔离"转换链本身是否可执行"。

**不足/开放问题**:(1) 真机评估每任务仅 10 rollout,且仅比 GR00T N1.7 单一基线,缺 EgoVLA/Being-H0/VITRA 等同类人类视频到动作方法的直接对比,复合分优势的统计强度有限;(2) 1,500 小时预训练语料未公开采集/多样性细节,可复现性存疑;(3) 接触力缺失是本质短板,强接触任务提升明显收窄,说明纯运动学监督的天花板;(4) 依赖同步 ego-exo 双视角采集,规模化仍受第三人称视角布设约束——作者也把"仅第一人称遮挡/模糊下的骨架恢复"列为未来工作。

**与公开工作的关系**:相较 EgoVLA(Yang et al., 2025,从第一人称人类视频学 VLA + 3D 手部标注)、Being-H0(Luo et al., 2025)、VITRA(Li et al., 2025,把真实活动视频转机器人对齐动作段)等偏语义/手轨迹/预训练信号的方向,本文的差异点是把输出定位在 **人形上身(臂+灵巧手+脖+腰)控制器对齐的 60-DoF 关节动作**,并在近视频采集速率运行、支持无目标任务机器人演示部署;相较 Omnih2o(He et al., 2024)这类人到人形整身遥操作,本文走的是离线视频到动作的数据合成而非在线遥操作。改进方向:引入接触力/触觉先验(如用手-物交互生成配可微接触模型)、扩大跨本体 retarget 泛化、以及在更大 rollout 规模下与多基线做统计充分的对比。

## 参考

1. Bjorck et al., *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, arXiv:2503.14734, 2025.(真机部署主对比基线)
2. Yang et al., *EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos*, arXiv:2507.12440, 2025.(同类人类视频到 VLA)
3. Lin et al., *PhysBrain: Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence*, arXiv:2512.16793, 2025.(PhysDex 采用的 VLM 骨干)
4. Zheng et al., *EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data*, arXiv:2602.16710, 2026.(下游中训练/后训练配方来源)
5. He et al., *Omnih2o: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning*, arXiv:2406.08858, 2024.(人到人形迁移的相关线路)
