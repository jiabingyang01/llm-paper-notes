# EgoDemoGen：面向机器人操作视角泛化的第一视角演示生成

> **论文**：*EgoDemoGen: Egocentric Demonstration Generation for Viewpoint Generalization in Robotic Manipulation*
>
> **作者**：Yuan Xu, Jiabing Yang, Xiaofeng Wang, Yixiang Chen, Zheng Zhu, Liang Wang et al.
>
> **机构**：中国科学院大学（UCAS）、中国科学院自动化研究所（CASIA）、GigaAI、清华大学、X-Humanoid
>
> **发布时间**：2025 年 09 月（arXiv 2509.22578，v2 于 2026 年 03 月更新）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2509.22578) | [PDF](https://arxiv.org/pdf/2509.22578)
>
> **分类标签**：`数据合成` `第一视角视角泛化` `视频扩散生成` `动作轨迹迁移` `模仿学习`

---

## 一句话总结

EgoDemoGen 针对"第一视角相机随机器人底座一起移动、单纯换相机位置会导致动作轨迹失效"这一具身特有难题,用 EgoTrajTransfer（几何变换+逆运动学过滤的动作轨迹迁移)配合 EgoViewTransfer（自监督双重投影训练的视频扩散模型)生成动作—观测成对一致的新视角演示,在 RoboTwin 2.0 仿真上标准/新视角成功率分别提升 +24.6% / +16.9%,真机 Mobile ALOHA 上提升 +16.0% / +23.0%,视频生成质量（FVD）从最优基线的 483.1 降到 133.5。

## 一、问题与动机

模仿学习策略对第一视角（头戴式相机)的视角变化非常敏感：机器人底座定位误差、平台重构或场地布局变化都会带来视角漂移,而按每个可能视角采集演示的成本过高。第一视角变化与常见的第三人称新视角问题有本质区别——第三人称视角变化只移动相机,动作轨迹不变;而第一视角相机通常与机器人底座固连,底座运动会**同时**改变机器人基座坐标系和相机观测,原始动作轨迹在新的基座坐标系下不再有效,必须联合完成"动作轨迹迁移"和"新视角观测合成"两件事,且二者要严格对齐。

现有三类工作都只解决了半个问题:(1) 几何重投影类方法（如 TrajectoryCrafter、Phantom)只做视觉重投影,不改动作,导致视觉—动作错位;(2) 纯视觉合成方法（如 VISTA)做零样本新视角合成但保持动作不变,只适配第三人称场景;(3) 运动重定向方法（Rocoda、ROVI-Aug 等)迁移动作到新物体位姿或新本体,但不处理第一视角相机随底座运动这一情形。论文提出**没有任何已有方法能生成在第一视角变化下视觉—动作一致的成对演示**,这正是 EgoDemoGen 要填补的空白。

## 二、核心方法

给定源演示 $(V, Q, v_{\text{src}})$（$V$ 为 RGB 视频、$Q=\{q_t\}$ 为关节轨迹)和目标新视角 $v_{\text{nov}}$（由底座平面运动 $v=(\Delta x,\Delta y,\Delta\theta)$ 参数化,诱导出坐标变换 $\Delta T \in SE(3)$),目标是生成新演示 $(\tilde V,\tilde Q,v_{\text{nov}})$:

$$
\tilde{Q}=\mathcal{T}(Q;\Delta T), \qquad \tilde{V}=\mathcal{G}(V,\tilde{Q};\Delta T),
$$

用大白话说：先用轨迹迁移算子 $\mathcal{T}$ 把动作搬到新基座坐标系下,再用条件视频生成器 $\mathcal{G}$ 依据这条新轨迹合成与之严格对齐的新视角观测,两步缺一不可。

**EgoTrajTransfer（动作轨迹迁移)**。三步走（Algorithm 1)：

1. **轨迹分段**：按每只手臂的夹爪状态独立切分为"运动段"（开爪、自由空间移动)和"技能段"（闭爪、接触密集操作),双臂分段边界在时间窗 $w_{\text{sync}}$ 内同步对齐,消除传感器噪声干扰。
2. **分段迁移**：运动段做"缩放+插值"——保留末端执行器路径结构,先算出新起止位姿 $T_{\text{end}}^{\text{new}}=\Delta T \cdot T_{\text{end}}^{\text{old}}$,再按

$$
p^{\text{new}}(t) = R_{\text{align}}(p^{\text{old}}(t) - p^{\text{start}}^{\text{old}})\,s + p^{\text{start}}^{\text{new}},
$$

其中 $s=\|p^{\text{new}}_{\text{end}}-p^{\text{new}}_{\text{start}}\| / \|p^{\text{old}}_{\text{end}}-p^{\text{old}}_{\text{start}}\|$ 为路径缩放比,姿态用 Slerp 插值;技能段则直接做刚体变换 $T_e^{\text{new}}(t)=\Delta T \cdot T_e^{\text{old}}(t)$,以保持物体相对运动不被破坏。用大白话说：自由移动阶段允许轨迹形状按比例"拉伸对齐"到新起止点,而接触操作阶段必须原封不动地整体搬移,否则抓取/操作的相对位姿关系会被破坏。
3. **逆运动学求解与过滤**：用 CuRobo 对每只手臂分别批量求解 IK,以原关节角作为种子保持臂型;失败帧用最近成功帧插值补齐;若 IK 成功率低于阈值 $\tau_{\text{IK}}$ 或最大关节角跳变超过 $\tau_{\text{jump}}$,则判该视角**不可行**并整条丢弃,从源头保证只生成运动学可行的新演示。

**EgoViewTransfer（新视角视频生成)**。以 CogVideoX-5B-I2V 为底座微调的统一 DiT 扩散模型,输入通道扩至 48 以支持双视频条件,三阶段流程：

1. **场景视频准备**：先把原视频逐帧重投影到新视角得到带空洞/伪影的视频,再用原轨迹 $Q$+URDF+新视角相机参数渲染机器人 mask 并抠除机器人区域,最后用经典 INPAINT_TELEA 方法补全空洞,得到干净的场景视频 $V_S^{\text{nov}}$。
2. **机器人运动渲染**：用迁移后的轨迹 $\tilde Q$+URDF+新视角相机参数渲染出纯机器人运动视频 $V_R^{\text{nov}}$,作为显式的动作条件。
3. **条件视频生成**：

$$
\tilde{V}=\text{EgoViewTransfer}(V_S^{\text{nov}}, V_R^{\text{nov}};\theta),
$$

场景视频编码新视角变换、机器人视频编码迁移后的轨迹,两路条件解耦,使模型能在推理时任意组合新视角与新轨迹,不必绑定训练时见过的视角—轨迹配对。

**自监督双重投影训练**。训练需要成对数据 $(V_S^{\text{nov}}, V_R^{\text{nov}}, V^{\text{target}})$,但真实采集只有单一视角演示。解法是"双重投影"：从源视角 $v_{\text{src}}$ 采样一个随机训练视角 $v_{\text{train}}$,把源视频重投影过去再投影回来,人为制造出与真实新视角合成同类的伪影和空洞,作为场景条件 $V_S^{\text{train}}$;机器人视频 $V_R^{\text{train}}$ 用原轨迹 $Q$ 在 $v_{\text{src}}$ 下渲染;监督目标就是源视频 $V$ 本身。标准扩散去噪目标：

$$
\mathcal{L}=\mathbb{E}_{t,\epsilon}\Big[\big\|\epsilon-\epsilon_\theta(z_t, V_S^{\text{train}}, V_R^{\text{train}}, t)\big\|^2\Big].
$$

用大白话说：让模型学会"修补重投影伪影+按给定机器人运动视频合成一致画面"这件事本身不需要多视角标注,只要在同一视角里"来回投影一次"制造出逼真的破损样本,模型学到的修复—合成能力就能在真正的新视角推理时直接复用。

## 三、关键结果

**仿真（RoboTwin 2.0,双臂 ARX-X5+头戴相机,7 任务,ACT 策略,100 trials/项)**——Table 1 平均值：

| 方法 | 标准视角成功率 | 新视角成功率 |
|---|---|---|
| Standard Viewpoint（仅标准视角数据训练) | 29.0 | 11.0 |
| Direct Reprojection | 45.0 | 19.1 |
| TrajectoryCrafter | 39.4 | 20.3 |
| Phantom | 53.6 | 21.4 |
| VISTA | 53.6 | 20.4 |
| **EgoDemoGen（本文)** | **60.0** | **27.9** |

对应绝对提升 +24.6%（标准视角) / +16.9%（新视角),EgoDemoGen 在 7 个任务中的 6 个取得最优新视角成功率。

**真机（Mobile ALOHA 双臂,5 任务,π0 策略,标准+4 个固定新视角各 20 trials)**——Table 2 平均值：Standard-View 训练基线 标准 60.0 / 新视角 37.0;Direct Reprojection 64.0 / 47.0;**EgoDemoGen 76.0 / 60.0**,绝对提升 +16.0%（标准) / +23.0%（新视角)。

**视频生成质量（Table 3,仿真新视角,对比真值视频)**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|---|---|---|---|---|
| Direct Reprojection | 10.10 | 0.711 | 0.335 | 621.5 |
| TrajectoryCrafter | 12.46 | 0.728 | 0.264 | 483.1 |
| VISTA | 12.67 | 0.687 | 0.271 | 1451.3 |
| Phantom | 16.97 | 0.786 | 0.174 | 1131.0 |
| **EgoViewTransfer（本文)** | **26.03** | **0.886** | **0.081** | **133.5** |

真机双重投影设定下同样领先（PSNR 26.93 vs 10.35,FVD 148.6 vs 968.7)。

**轨迹迁移有效性验证（Table 5,开环 replay 成功率,50 trials/task)**：EgoTrajTransfer 迁移后轨迹平均 replay 成功率 99.3%,而直接沿用源动作轨迹仅 22.7%（Adjust/Lift/Place 三任务分别为 52/8/8),证明第一视角变化下原始关节轨迹确实不可直接复用,轨迹迁移是构造有效成对演示的必要步骤。

**消融（Table 4,3 个仿真任务平均)**：完整版标准/新视角成功率 68.0/33.7,FVD 154.2;去掉双重投影降到 58.0/28.0,FVD 升至 229.6;去掉 mask+inpaint 降到 66.0/30.0,FVD 205.8;**去掉轨迹迁移（直接用源动作)降幅最大**,61.3/19.0,证明三个组件都必要且轨迹迁移贡献最大。

**其它分析**：数据配比在生成:真实≈0.4–0.5 时新视角性能最佳,全部替换为生成数据（比例=1.0)反而回落;固定真实数据、单纯增加生成数据在仿真和真机上均持续提升且边际收益递减；训练时使用的视角生成范围越宽,测试时对更大视角偏移的泛化越好（平均成功率从仅标准视角训练的 7.8% 提升到最宽范围训练的 20.5%);在相机与底座运动解耦（外加独立相机扰动)的更难设定下,EgoDemoGen 仍将平均成功率从 7.8% 提升到 23.3%,证明两路解耦条件设计具备超出"底座—相机刚性绑定"假设的泛化性。

## 四、评价与展望

**优点**：(1) 准确刻画了第一视角与第三人称新视角生成的本质区别（底座—相机联动 vs. 纯换镜头),这是一个此前文献普遍忽视、但对移动/双臂操作平台切实存在的问题;(2) EgoTrajTransfer 的"运动段缩放插值+技能段刚体变换+IK 可行性过滤"设计有明确物理动机,并通过 Table 5 的开环 replay 实验直接验证了轨迹迁移本身的必要性,而不仅是依赖下游策略成功率这一间接证据;(3) EgoViewTransfer 的双重投影自监督训练巧妙地把"缺乏多视角配对数据"这一常见瓶颈转化为单视角数据内部可解决的问题,场景/机器人双视频解耦条件设计也带来了推理时任意视角—轨迹组合的灵活性;(4) 消融、数据配比、视角泛化范围、底座—相机解耦四组分析实验较为完整,覆盖了方法的主要设计选择和边界条件。

**局限与开放问题**：论文正文未设独立 Limitations 小节,但从方法与结论可归纳出几点：(1) 底座运动被限定为平面运动 $(\Delta x,\Delta y,\Delta\theta)$,虽然 §4.6 做了相机—底座解耦的扩展实验,但仍未覆盖高度/俯仰变化等更一般的第一视角扰动;(2) IK 可行性过滤会直接丢弃不可行视角,极端视角下的数据利用率与生成视角覆盖范围之间的权衡未被量化;(3) 该框架是离线两阶段流水线（先训 EgoViewTransfer 再批量生成),尚不支持闭环生成,论文结论部分明确将"与世界模型结合做闭环演示生成"列为未来方向;(4) 目前局限于同本体（sim 中的 ARX-X5、真机 Mobile ALOHA)内的视角增广,未涉及跨本体迁移,这也是作者自陈的未来工作方向之一;(5) 基线对比中 TrajectoryCrafter、Phantom 仅在仿真评测,真机部分只与 Direct Reprojection 比较,视频生成类基线在真机上的表现缺失。

**与相关工作的关系**：与 VISTA（CoRL 2025)等第三人称新视角合成方法相比,EgoDemoGen 的核心区别在于显式处理了动作坐标系变化,因此能提供"配对"而非"仅视觉"的新视角监督——消融和主表都显示 VISTA 虽然标准视角分数尚可,但新视角泛化明显弱于 EgoDemoGen,印证了论文"配对动作—观测监督比单纯视觉新视角合成更有效"的核心论点;与 TrajectoryCrafter、Phantom 等几何重投影/补洞类方法相比,EgoDemoGen 用扩散模型学习修复而非依赖显式 3D 重建,在存在遮挡和大视角偏移时更鲁棒（Table 3 的 FVD/PSNR 差距印证);与 Rocoda、ROVI-Aug 等运动重定向类工作相比,二者关注跨物体位姿或跨本体的动作迁移,EgoDemoGen 关注的是同一本体、不同第一视角相机安装位置下的动作—视角联合合成,是互补而非竞争关系。整体看,该工作为"移动/双臂平台第一视角鲁棒性"这一实际部署痛点提供了一条系统化且有实验支撑的数据合成路径。

## 参考

- Tian et al. *VISTA: View-invariant Policy Learning via Zero-shot Novel View Synthesis*. CoRL 2025.
- Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation*. arXiv:2506.18088, 2025.
- Yu, Hu, Xing, Shan. *TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models*. arXiv:2503.05638, 2025.
- Lepert, Fang, Bohg. *Phantom: Training Robots without Robots using only Human Videos*. CoRL 2025.
- Sundaralingam et al. *CuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*. arXiv:2310.17274, 2023.
