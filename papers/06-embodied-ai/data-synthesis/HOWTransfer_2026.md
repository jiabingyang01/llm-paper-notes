# HOWTransfer：基于开放世界接触定位的以手为中心的人到机器人视频演示轨迹迁移

> **论文**：*Hand-centric Human-to-Robot Trajectory Transfer from Video Demonstrations via Open-World Contact Localization*
>
> **作者**：Yitian Shi\*, Di Wen\*, Zhengqi Han, Zicheng Guo, Yu Hu, Edgar Welte, Kunyu Peng, Rainer Stiefelhagen, Rania Rayyes（\* 表示共同一作）
>
> **机构**：Karlsruhe Institute of Technology (KIT)，德国卡尔斯鲁厄
>
> **发布时间**：2026 年 06 月（arXiv 2606.10743）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.10743) | [PDF](https://arxiv.org/pdf/2606.10743)
>
> **分类标签**：`人到机器人迁移` `视频演示学习` `接触定位` `平行夹爪抓取` `轨迹合成`

---

## 一句话总结

HOWTransfer 用一条低成本双目人手演示视频,通过"以手为中心"的三段式流水线——重建时序一致的 3D 手部轨迹、开放世界地免语义地定位任务相关接触区间、再把人手抓取意图重定向为分类学感知(taxonomy-aware)的平行夹爪(PJ)轨迹并增广出多条可执行变体——在真机复现(replay)上取得 86% 成功率(比模板抓取基线高 23 个百分点),且在盲测偏好研究中以 80.40% 非平局胜率被人类评审偏好于遥操作演示。

## 一、问题与动机

从人手视频学习操作技能,是替代昂贵遥操作/动觉示教、可规模化采集机器人数据的诱人途径。但把视频里富含接触的 Hand-Object Interaction (HOI) 线索迁移到机器人时,存在三重困难:

- **形态鸿沟(cross-embodiment)**:人手与 PJ 夹爪自由度差异巨大。已有方法多用稀疏手部线索(指尖、thumb-index 几何、物体中心 affordance 区域)做重定向,这会"坍缩"多样的人手抓取类型,并掩盖依赖整只手接触的抓取意图(grasp intent)。
- **迁移时机(when to transfer)**:人手视频含大量冗余(漫长的接近动作、停顿、反复的松-合),而轨迹生成只需要那几个真正可迁移的关键接触阶段。这些接触区间是 PJ 抓取初始化与轨迹传播的时间锚点。
- **已有时序定位不稳**:如 EgoLoc 这类方法针对第一人称(egocentric)的接触-分离时刻设计,在非第一人称、重复、含多接触阶段的长程演示上会失稳。

作者把"人手视频演示迁移"形式化为一个**以手为中心的轨迹蒸馏问题**:从单条人手演示中抽取多条显式、可执行且保留关键 HOI 模式的机器人轨迹,供 replay、增广并针对下游物理约束校验。关键是既要恢复手**怎么动(how)**,还要判断有意义的接触**何时发生(when)**、以及**哪种 PJ 抓取(which)**能实现所演示的人手抓取意图。

## 二、核心方法

HOWTransfer(H and-Object O pen-World Transfer)由三个阶段串联,全流程免物体类别名、免语言 prompt、免任务标注接触边界。

### 阶段一:手部轨迹重建(Hand Trajectory Reconstruction)

给定双目视频 $\mathcal{V} = (I_t^1, I_t^2)_{t=1}^T$,每一视角用冻结的 WiLoR 预测腕部位姿 $M_t^n = (\omega_t^n, q_t^n)$ 与 MANO 手参数 $(\theta_t^n, \beta_t^n)$;双目几何提供度量级腕部定位。由于单帧 WiLoR 对遮挡/噪声敏感、严格双目三角化在某视角丢检时失败,作者用时序插值补全缺帧,并用 $SE(3)$ 上的 Iterative Extended Kalman Filter + Rauch-Tung-Striebel 平滑器(IEKF-RTS)精修腕部轨迹。多视角腕部朝向用四元数旋转平均融合:

$$
\bar{c}_t = \frac{c_t^1 + c_t^2}{\lVert c_t^1 + c_t^2 \rVert_2}
$$

**用大白话说**:先用现成的手部重建大模型逐帧估计手的位姿,再用双目算出真实尺度,最后用卡尔曼平滑把抖动和丢帧"抹平",得到一条干净连续的 3D 腕部+手形轨迹。

### 阶段二:开放世界接触定位(Open-World Contact Localization)

目标:从逐帧腕位 $M_t$ 与 MANO 参数 $H_t$ 估计任务相关接触区间 $\boldsymbol{C} = \{[s_k, e_k]\}_{k=1}^K$($s_k$ 为接触起始帧,$e_k$ 为释放帧)。不同于依赖物体描述/VLM 查询/任务专用接触分类器的方法,它靠交互本身来发现被操作物体。

- **免类别物体胶囊(Category-Free Object Capsule)**:先从手部流算出手中心时序线索——手闭合度 $\kappa_t$、可见性 $\nu_t$、手物邻近度 $\alpha_t$,用它们框定"物体可被可靠发现"的手-活跃时间窗;窗内用 SAM3 在两视角生成类无关掩膜提案,再按几何一致性、手接近、物体侧运动、掩膜质量、演示者(手臂)重叠剔除等做**跨视角晚绑定(cross-view late binding)**,选出唯一被操作物体轨迹。该胶囊用"交互接地"的视觉+运动证据表示物体,而非语义类别标签。
- **稀疏几何辅助**:可选地在稀疏手-活跃帧上用 DA3(Depth Anything 3)取 3D 物体状态证据,用于掩膜校验、物体运动估计与阶段精修。
- **段级证据融合与训练-free 接触门**:综合可见手线索、手物运动耦合 $\mu_t$、几何支持 $\delta_t$ 及负向"断路"线索 $\xi_t$(捕获释放、手物运动解耦、演示者重叠、物体观测不一致),定义逐帧接触证据:

$$
\chi_t = \big(1 - B(\xi_t)\big) \max\!\big(F_{\text{hand}}(\kappa_t, \nu_t, \alpha_t),\, F_{\text{motion}}(\mu_t, \alpha_t),\, F_{\text{geo}}(\delta_t, \alpha_t)\big)
$$

其中 $B(\xi_t)$ 在断路证据下抑制不可靠支持;所有门参数跨所有视频保持恒定,无需接触监督。最后用固定滞回(hysteresis)解码器从 $\chi_t$ 解出候选接触区间,再经段级一致性规则(在局部手物证据支持且不与断路证据冲突时才做 split/merge/短区间补充)得到最终 $\boldsymbol{C}$。

**用大白话说**:不告诉系统"物体是什么",而是看"手在哪儿、有没有闭合、离哪块东西近、那块东西是否跟着手一起动、深度上是不是一个紧凑实体";三路正向证据取最大、一路负向证据做否决,谁也不能单独说了算(手闭合但物体没动、或物体动了但手没耦合,都不算接触)。这样就在没有标签的情况下判断出"真正抓住东西"的时间段。

### 阶段三:跨形态轨迹重定向(Cross-Embodiment Trajectory Retargeting)

把人手演示转成 PJ 末端执行器轨迹,核心是**将抓取初始化与轨迹传播解耦**。

- **抓取重定向(Grasp Retargeting)**:每个接触段的起始帧 $s_k$ 作为最具信息量的重定向关键帧(即"何时抓取"确立的瞬间),把该帧的局部 RGB 观测 $\mathcal{I}_{s_k}$ 与重建 MANO 手状态 $\mathcal{H}_{s_k}$ 融合成交互描述子,调用 HOGraspFlow 在 $SE(3)$ 流形上用流匹配(flow matching)生成分类学感知的多模态 PJ 抓取分布:

$$
g^0 \sim p_\phi\big(g \mid \mathcal{I}_{s_k}, \mathcal{H}_{s_k}, \gamma_{s_k}\big), \qquad g^0 \in SE(3)
$$

其中 $\gamma_{s_k}$ 是推断出的抓取分类学先验。为提升开放世界鲁棒性,HOGraspFlow 在扩充的 HOI 语料(HOGraspNet + OakInk + HO3D)上训练,生成抓取再用 DBSCAN 在 $SE(3)$ 距离下聚类出代表性候选。

- **轨迹传播(Trajectory Propagation)**:接触确立后假设"手-物刚性耦合",腕相对抓取变换在同一段内保持不变。设 $T_w(s_k)$ 为段起始腕位,则

$$
g_k^t = T_w(t)\, T_w(s_k)^{-1}\, g_k^0, \qquad t \in [s_k, e_k]
$$

对每段传播并拼接得到完整末端轨迹 $\mathcal{G}_k$,既保留人手视频的任务相关交互模式,又适配目标形态。

- **轨迹精修与增广(LTE)**:抓取传播可能因手位估计误差引入起始对齐偏移。作者用 Laplacian Trajectory Editing (LTE) 做两件事:(i) **接触感知精修**——从 HOGraspFlow 抓取条件接触图与 DA3 首帧点云估计平移修正 $\bar\delta_k$,只编辑首个控制位姿、固定段终点、其余轨迹平滑变形;(ii) **碰撞感知增广**——扰动中间控制点、在固定起止约束下重解 LTE,生成保形变体,并用局部净空点云(0.05 m 半径内障碍点数 $\le N_{\max}=30$)拒绝易碰撞编辑,每条基轨迹最多保留 5 条增广。于是**一条演示 → 多条可执行 PJ 轨迹变体**(论文示例:1 条演示的 3 个抓取候选共生成 15 条轨迹)。

**用大白话说**:抓的瞬间用一个专门的抓取生成模型把"人怎么抓"翻译成"夹爪怎么抓"(还能给出多种合理抓法);抓稳之后就假设手和物体绑在一起,把夹爪位姿跟着手腕一路带过去;最后用一种"拉着首尾不动、只揉中间"的曲线编辑技术,既把抓取点对齐到真实接触面,又能揉出好几条不撞东西的备选轨迹用于扩充数据。

## 三、实验结果

评测基准:110 条人手演示视频,覆盖 11 个操作任务(每任务 10 条),含日常与工业风格;每条视频人工标注接触/分离时间戳作为真值。硬件为双 Intel RealSense D435i 标定相机 + UR10e + Robotiq 2F-85 夹爪。

### 1)时序接触定位(Table 1,整体)

| 方法 | SR(3)↑ | SR(5)↑ | SR(10)↑ | MAE↓ | MoF↑ | IoU↑ | Precision↑ | F1↑ |
|---|---|---|---|---|---|---|---|---|
| Threshold(阈值/thumb-index 闭合) | 0.364 | 0.423 | 0.508 | 30.195 | 0.784 | 0.465 | 0.508 | 0.584 |
| EgoLoc | 0.075 | 0.127 | 0.207 | 27.264 | 0.456 | 0.382 | 0.653 | 0.495 |
| **Ours (w/o DA3)** | **0.495** | 0.579 | 0.687 | 11.805 | 0.790 | 0.766 | **0.963** | 0.851 |
| **Ours** | 0.491 | **0.581** | **0.736** | **11.787** | **0.872** | **0.816** | 0.932 | **0.891** |

MAE 从基线约 27~30 帧降到约 11.8 帧;加入 DA3 几何证据后 MoF/IoU/F1 进一步提升(0.872 / 0.816 / 0.891),不加 DA3 的版本 Precision 最高(0.963)。EgoLoc 因其第一人称时间戳假设不适配本文非第一人称、多阶段设置而表现最差。

### 2)真机 replay 成功率(Fig 4 左)

用预采集人手演示为每任务生成 10 条机器人 episode 复现。**HOWTransfer 整体 replay 成功率 86%,比模板抓取匹配基线(用相同接触段但换成固定 thumb-index 抓取模板)高 23 个百分点**。在需要任务特定抓取选择与接触对齐的任务上差距尤为明显:Watering(浇水)92% vs 30%,Angle Grinder Pickup(角磨机拆卸)78% vs 0%。长程多接触任务(Pot Cooking、Breakfast)因误差累积表现下降,但仍全面优于模板基线。

### 3)盲测偏好研究(Fig 4 右)

将 HOWTransfer 与遥操作(Teleop,经 Meta Quest 3 采集)轨迹做盲测成对偏好比较,评分区间 $[-100, 100]$(正值偏好 HOWTransfer)。**平均偏好分 19.21,归一化 59.61/100,非平局胜率 80.40%**。最受偏好任务:Watering(65.05)、Erase Whiteboard/rub(63.86)、Upright(63.38);Pick-Place 最接近中性(50.48)。

### 4)下游模仿学习(Table 5)

把迁移出的轨迹当训练数据,在每任务 50 条演示上训 ACT/DP/DP3,每任务 20 次试验评测(8 个任务合计 160 次):

| 策略 | 总成功 |
|---|---|
| Diffusion Policy (DP) | 105/160 |
| 3D Diffusion Policy (DP3) | 111/160 |
| ACT | 107/160 |

说明蒸馏自人手视频的轨迹不仅能直接 replay,也能为下游模仿学习提供有效监督。

## 四、局限性

- **仅限 PJ 末端执行器轨迹重定向**:排除了灵巧手的手内操作(in-hand manipulation)、手指步态(finger-gaiting)、手内连续重定向。刚性耦合假设也限制了抓取后物体在夹爪内相对滑动的场景。
- **碰撞感知增广靠启发式**:用局部净空点数阈值而非完整物理仿真或闭环重规划,面对复杂动力学的鲁棒性留待未来工作。
- **重度依赖多个外部大模型**:WiLoR、MANO、SAM3、DA3、HOGraspFlow 任一失效都会向下游传导误差;失败分析指出接触丰富的擦拭类(轨迹偏低/抓取过深撞白板)与受约束工具类(刀/铲/盖朝向不准致碰撞)是主要瓶颈。
- **评测规模有限**:110 视频 / 11 任务 / 单一 UR10e+2F-85 平台,跨机器人本体、跨相机配置的泛化未验证;真值接触时间戳仍需人工标注。

## 五、评价与展望

**优点**:
- **免语义、训练-free 的接触定位是最有辨识度的贡献**。把"何时接触"建成一个由多路手/物/几何证据一致性投票、加负向断路否决的确定性门控,跨全部视频共享一套参数,回避了 VLM 查询与任务专用分类器,工程上很务实,MAE 相对 EgoLoc/阈值基线近乎腰斩。
- **"抓取初始化 ⊥ 轨迹传播"的解耦**干净利落:只在接触起始帧调一次生成式抓取模型、其余帧靠刚性耦合传播,既省算力又保住了演示的交互时序结构。
- **一条演示扩多条可执行变体 + LTE 保形增广**直接服务于数据规模化,且用真机 replay 成功率与下游 IL 成功率两条线做了闭环验证,而非只报中间指标。
- 盲测偏好中人手视频迁移轨迹被偏好于遥操作,是对"人手视频作为遥操作之外可扩展轨迹源"这一主张的有力支撑。

**缺点与开放问题**:
- 方法本质是**多个现成基础模型的精巧编排**,单点算法创新有限;鲁棒性上限被最弱一环(手部重建/分割/深度)锁死。
- 刚性耦合假设对"抓取后需重新定向物体、或物体在夹爪内滑移"的任务是硬约束,这也是限于 PJ、无法灵巧操作的根因。
- 接触门的众多阈值虽宣称跨视频恒定,但对新任务分布/新相机基线的迁移性未做敏感性分析。

**与其他公开工作的关系**:相较 YOTO(从单条双目演示抽双手轨迹再经物体点云变换 rollout)、R+X(检索日常人手片段做 in-context 执行)、WARPED(腕对齐渲染重建第一人称演示),本文更强调把每条演示显式转成**接触感知 + 分类学感知**的 PJ 轨迹,而非把接触阶段吸收进策略或物体中心表示;时序侧对标 EgoLoc 但明确针对非第一人称多阶段场景;抓取侧直接构建在同组 HOGraspFlow 之上。可能的改进方向:引入闭环/物理增广替代净空启发式、把接触门参数做可学习或自标定、把刚性耦合放松为可估计的手内相对运动以覆盖灵巧操作、以及在更多本体上验证跨形态泛化。

## 参考

1. Shi, Guo, Wolf, Welte, Rayyes. *HOGraspFlow: Taxonomy-aware hand-object retargeting for multi-modal SE(3) grasp generation.* arXiv:2509.16871, 2026.（本文抓取重定向的直接依赖)
2. Ma, Zhang, Zheng, Xie, Zhou, Wang. *EgoLoc: A generalizable solution for temporal interaction localization in egocentric videos.* arXiv:2508.12349, 2025.（时序定位主要对比基线)
3. Potamias, Zhang, Deng, Zafeiriou. *WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild.* CVPR 2025.（手部重建骨干)
4. Carion et al. *SAM 3: Segment anything with concepts.* arXiv:2511.16719, 2025.（免类别物体胶囊的分割来源)
5. Nierhoff, Hirche, Nakamura. *Spatial adaption of robot trajectories based on Laplacian trajectory editing.* Autonomous Robots, 2016.(LTE 精修/增广基础)
