# Bridging Action：以平移作为桥梁动作,把操作技能从人迁移到机器人

> **论文**：*Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots*
>
> **作者**：Sijin Chen, Kaixuan Jiang, Haixin Shi, Yanhui Wang, Weiheng Zhong, Haosheng Li, Bo Jiang, Yuxiao Liu, Xihui Liu（Sijin Chen 与 Kaixuan Jiang 共同一作,Haixin Shi 为 Project Lead,Xihui Liu 与 Haixin Shi 为通讯作者）
>
> **机构**：HKU-MMLab（香港大学）、ByteDance Seed
>
> **发布时间**：2026 年 06 月（arXiv:2606.28133v1，标注 June 29, 2026）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.28133) | [PDF](https://arxiv.org/pdf/2606.28133)
>
> **分类标签**：`human-to-robot transfer` `bridging action` `VLA` `flow-matching` `cross-embodiment co-training`

---

## 一句话总结

不要把人手当成"另一台 6DoF 机器人"去学它带噪的手腕旋转,而是只学**头相机坐标系下的相对手腕平移**这一个人与机器人共享的三维桥梁动作;再用一个 π0 式、带**交错动作 token + 注意力掩码**的 VLA 与三阶段训练,把人类操作技能迁移到双臂平行夹爪机器人。在 15 个真实任务上,纯 pick-and-place 数据几乎无法完成下游任务(整体成功率近 0),加入人类共训后整体成功率达 38.33%,且桥梁动作明显优于 6DoF 人类动作(progress 44.58% vs 34.67%),大规模人类预训练还能把 10 条轨迹的少样本后训练成功率从 35.83% 提升到 55.00%。

## 一、问题与动机

人类第一视角(ego-centric)操作数据便宜、丰富、多样,被视为扩展机器人学习最有前景的数据源之一。但把人类技能迁移到机器人一直很难,主流做法(如 GR-3、EgoMimic、EgoBridge、EgoScale 等)把人手当作**又一个 6DoF embodiment**:用手部姿态估计器抽取相对手腕的完整 6DoF 位姿,再重定向到机器人。作者指出这条路线存在两个根本问题:

1. **旋转不可靠**:人手手腕旋转来自 hand-pose estimator,预测误差大、噪声高;
2. **接触模式失配**:人手手指的接触方式与平行夹爪根本不同,手指多出的自由度使手腕旋转对"操作行为语义"的表达变弱。

因此作者主张:**从人类数据里学习"含旋转"的机器人操作信号是高度困难且次优的**。经验上直接把抽取的人类 6DoF 手腕动作回放到机器人,常导致扭曲、跑偏的行为(见 Fig.7/Fig.8 定性对比)。

动机随之而来:人和机器人都作用于"它们所观察到的东西",所以应该寻找一个在**共享观察视角下物理有意义、对旋转噪声鲁棒、且天然与 embodiment 无关**的动作空间——作者的答案是**头相机坐标系下的相对手腕平移**。

## 二、核心方法

方法由两部分组成:桥梁动作表征(4.1)与带交错动作序列的 VLA(4.2),再配三阶段训练(4.3)。

### 2.1 运动桥梁动作表征

**桥梁信号 $\mathbf{a}^{\text{3D-wrist}}$(共享)。** 设 $\mathbf{W}^{t}_{w}\in\mathbb{SE}(3)$ 为 $t$ 时刻世界系下的手腕位姿,$\mathbf{T}^{t}_{w\leftarrow c}\in\mathbb{SE}(3)$ 为该时刻头相机位姿(相机系记为 $c_t$)。先把手腕位姿投影到当前头相机系:$\mathbf{W}^{t+i}_{c_t} = (\mathbf{T}^{t}_{w\leftarrow c})^{-1}\,\mathbf{W}^{t+i}_{w}$,再在未来 $k$ 步窗口内取平移分量之差:

$$
\mathbf{a}^{\text{3D-wrist}}_{t+i} = \Delta\mathbf{W}^{\text{3D}} = \mathbf{t}\big(\mathbf{W}^{t+i}_{c_t}\big) - \mathbf{t}\big(\mathbf{W}^{t}_{c_t}\big),\quad i=1,\dots,k
$$

其中 $\mathbf{t}(\cdot)$ 抽取 $3\times1$ 平移分量。双臂拼接后 $\mathbf{a}^{\text{3D-wrist}}_t\in\mathbb{R}^{k\times6}$,人和机器人都用同一定义。

**用大白话说**:站在头上的相机往下看,只记录"手腕在未来若干帧相对现在往哪个方向平移了多少",完全不管手腕转了多少度。相机在哪、绝对姿态如何都不影响,因为一切都被换算到"此刻我看到的画面坐标系"里——这正是人和机器人唯一能对齐的东西。

**机器人末端动作 $\mathbf{a}^{\text{6D-eef}}$(仅机器人可监督)。** 相对初始末端位姿的相对运动:

$$
\mathbf{a}^{\text{6D-eef}}_{t+i} = \Delta\mathbf{W}^{\text{6D}} = (\mathbf{W}^{t}_{w})^{-1}\,\mathbf{W}^{t+i}_{w},\quad i=1,\dots,k
$$

它是两个 $\mathbb{SE}(3)$ 元素之间的相对位姿,对绝对相机姿态不变;进一步转成笛卡尔坐标 + 欧拉角,双臂拼接得 $\mathbf{a}^{\text{6D-eef}}_t\in\mathbb{R}^{k\times12}$,这才是真正下发给机器人的可执行动作。

**用大白话说**:$\mathbf{a}^{\text{3D-wrist}}$ 是"人和机器人都能懂的通用语"(只有平移),$\mathbf{a}^{\text{6D-eef}}$ 是"机器人自己的可执行方言"(带旋转)。学习时用通用语搭桥,执行时翻译成方言。

**夹爪动作 $\mathbf{a}^{\text{gripper}}$。** 每个夹爪的二值信号 $a^{\text{gripper}}_i\in\{0,1\}$(1 关 0 开),$\mathbf{a}^{\text{gripper}}_t\in\mathbb{R}^{k\times2}$;in-lab 人类数据可用手部闭合标注为夹爪信号。

**统一动作空间与分源监督。** $\mathbf{a}_t = (\mathbf{a}^{\text{3D-wrist}}_t,\ \mathbf{a}^{\text{6D-eef}}_t,\ \mathbf{a}^{\text{gripper}}_t)$。不同数据源只监督自己可靠可得的分量(Table 1):

| 数据源 | $\mathbf{a}^{\text{3D-wrist}}$ | $\mathbf{a}^{\text{6D-eef}}$ | $\mathbf{a}^{\text{gripper}}$ |
|---|:---:|:---:|:---:|
| In-the-wild 人类数据（EgoDex + 外采） | ✓ | – | – |
| In-lab 人类数据 | ✓ | – | ✓ |
| 机器人 tele-operation | ✓ | ✓ | ✓ |

### 2.2 带交错动作序列的 VLA

**架构。** π0 式端到端 VLA $\pi_\theta(l, o_t)$,给定语言指令 $l$ 与头+双腕三路相机观测 $o_t$,生成动作 chunk $\mathbf{a}_t = a_{t:t+k}$。视觉、语言、动作 token 放在同一序列共享自注意力层,但用**两套参数**平衡不同训练目标;$(o_t,l)$ 经预训练 VLM(Qwen2.5-VL)处理,其 vision-language KV-cache 作为上下文条件,Action Transformer 通过 flow matching 生成动作。人类数据缺失的腕部视图用空白图补齐。

**交错动作 token。** 按 $\mathbf{a}^{\text{3D-wrist}}\!\to\!\mathbf{a}^{\text{6D-eef}}\!\to\!\mathbf{a}^{\text{gripper}}$ 的顺序把动作 token 交错排列,借助注意力掩码 + position id 处理变长/缺失分量:某数据源缺哪个分量,就在注意力层把对应 token 掩掉、并跳过其 loss。排序背后有两个先验:①共享桥梁信号应被 6DoF 动作 token"注意到",以显式完成人到机器人的知识迁移;②夹爪信号通常在末端到位后才触发,故放最后。

**Flow-matching 目标。** 给 $\tau\in(0,1)$、$\epsilon\sim\mathcal{N}(0,\mathbf{I})$,对加噪 chunk $\mathbf{a}^\tau_t=\tau\epsilon+(1-\tau)\mathbf{a}_t$ 预测速度 $\hat{v}$ 逼近真值 $v^*=\epsilon-\mathbf{a}_t$:

$$
\mathcal{L}_{\text{FM}} = \big\|\hat{v}(\mathbf{a}^\tau_t, o_t, l, \tau) - v^*\big\|_2^2
$$

推理时只对 $\mathbf{a}^{\text{6D-eef}}$ 与 $\mathbf{a}^{\text{gripper}}$ 从 $\tau=0$ 到 1 以 $\Delta\tau=0.2$ 用 Euler 法积分得到动作。为防在动作数据上过拟合,额外用 vision-language 数据的 NTP 目标 $\mathcal{L}_{\text{NTP}}$ 共训,总损失是二者按 batch 的加权和。

### 2.3 三阶段训练

- **Stage I：人类动作预训练。** 约 600 小时人类动作(EgoDex 精选 ~70h + 外采自由家庭操作 ~500h + in-lab ~45h,后两者用 PICO 4 Ultra Enterprise 采集)。此阶段接触模式与旋转都无法标准化,故**只监督桥梁信号** $\mathcal{L}^{\text{3D-wrist}}_{\text{FM}}$。
- **Stage II：人机共训。** 只用泛化 pick-and-place 机器人数据(~72h,100 类物体,固定模板 "`put {object} into {container}`")+ 每任务 ~3h 的 15 类 in-lab 人类动作。机器人数据上三个 loss 全开。**关键技巧**:在机器人数据上**随机加入或用 $\mathbf{a}^{\text{3D-wrist}}$ 替换 $\mathbf{a}^{\text{6D-eef}}$ 作为预测目标**,把桥梁表征显式"绑定"到可执行动作——5.5 节证明这一步对迁移至关重要。
- **Stage III：少样本机器人后训练。** 每任务采 100 条遥操作轨迹,但此阶段**只用 10 条/任务**微调以研究数据效率。

**实现。** Mixture-of-Transformer(MoT)架构,约 4B 参数。Stage I 从预训练 VLM 初始化,global batch 1024 训 400k 步;Stage II global batch 256 训 120k 步;Stage III global batch 256 训 25k 步;并把 VLM 的 KV-cache 重复 4× 以增大 Action Transformer 有效 batch、加速收敛。硬件为 ByteMini 双臂移动平台(两条 7-DoF 臂、平行夹爪、头+双腕三 RGB-D 相机),rollout 时底盘固定。

## 三、实验结果

**设置。** 15 个真实操作任务(microwave / drawer / mug-cup / other 四组)。每任务 2 个含干扰物的评测场景,每场景 4 次共 8 trials,同时汇报**成功率 Succ. 与平均进度 Progress**(Fig.4 给出每任务分档打分标准)。

**主结果(Fig.5,整体 Progress)。** 纯机器人 pick-and-place 数据(w/o human)几乎无法完成下游任务,加入人类共训、大规模预训练、少样本后训练逐级提升:

| 训练配置 | 整体 Progress |
|---|:---:|
| w/o human（仅 Stage II 机器人 pick-and-place） | 0.21 |
| Co-train（Stage II，人+机共训） | 0.45 |
| Pretrain + Co-train（Stage I+II） | 0.60 |
| + Few-shot Post-train（Stage I+II+III） | 0.72 |

结论 1:桥梁动作把技能迁移**扩展到 pick-and-place 之外**(纯 pick-and-place 成功率近 0);结论 2:即便只用桥梁信号做大规模人类预训练,也能**显著提升**(蓝 vs 橙),说明表征可扩展。

**Q2:桥梁动作 vs 6DoF 人类动作(Table 2,从零共训)。** 只把人类动作换成 3DoF 平移,整体全面更优:

| 人类动作表征 | 整体 Prog(%) | 整体 Succ(%) |
|---|:---:|:---:|
| $\mathbf{a}^{\text{6D-eef}}$（6DoF） | 34.67 | 12.50 |
| $\mathbf{a}^{\text{3D-wrist}}$（平移,本文） | **44.58** | **22.50** |

分组看,微波炉组差距尤为明显(Prog 25.00→38.02、Succ 4.17→25.00),定性上 6DoF 会产生扭曲、偏离门把手的手腕姿态。

**Q3:人类预训练提升少样本后训练数据效率(Table 3,10 条/任务)。**

| 模型 | 整体 Prog(%) | 整体 Succ(%) |
|---|:---:|:---:|
| Stage III（无预训练直接后训练） | 53.79 | 35.83 |
| Stage I + III（先人类预训练） | **71.21** | **55.00** |

即便预训练只见过**不可执行**的平移动作,仍把成功率从 35.83% 抬到 55.00%。

**Q4:机器人数据上必须监督桥梁动作(Table 4,消融)。** 去掉在机器人数据上随机加/替 $\mathbf{a}^{\text{3D-wrist}}$ 的绑定策略,整体成功率从 38.33% 崩到 12.50%:

| 机器人动作监督 | 整体 Prog(%) | 整体 Succ(%) |
|---|:---:|:---:|
| w/o $\mathbf{a}^{\text{3D-wrist}}$ | 39.67 | 12.50 |
| w/ $\mathbf{a}^{\text{3D-wrist}}$ | **59.75** | **38.33** |

**Q5:预训练目标与可执行动作空间对齐(Fig.9)。** 从人类预训练初始化(蓝)相比从零(红),在共训阶段的 $\mathbf{a}^{\text{6D-eef}}$ 与 $\mathbf{a}^{\text{gripper}}$ 上都收敛到**更低的 loss**——说明优化平移桥梁信号与优化可执行 6D 动作共享相近的目标景观,解释了"仅平移预训练为何能迁移到完整动作空间"。

**Q6:桥梁目标的上界(Table 5)。** 把 100 条/任务的 in-lab 机器人示范当作"人类数据"(转成平移动作、无观察 gap 且动作噪声极小)去用同一目标训练,得到上界:

| 模型 | 整体 Prog(%) | 整体 Succ(%) |
|---|:---:|:---:|
| Default（Ours） | 59.75 | 38.33 |
| Upper Bound | **73.54** | **55.83** |

说明随着视觉 gap 与动作噪声减小,桥梁表征的迁移会越发高效,指向更广的多 embodiment 学习。

## 四、局限性

- **缺旋转,难做精细对齐**:只用手腕平移作桥梁,天然丢弃旋转监督,在需要精细旋转配置的接触密集任务上受限。失败案例(Fig.12)集中在 "insert the straw into the cup" 与 "open the drawer"——策略有明确任务意图,却在"稳固抓取吸管"或"转腕建立有效拉拽接触"这类关键步失败,这与丢弃旋转的设计选择一致。
- **难抓薄物**:共训后机器人抓取薄物体表现不佳,作者归因于观察/embodiment gap 与人类动作中不可避免的噪声。
- **评测规模有限**:15 个任务、每任务 8 trials(2 场景×4 次),真实机器人评测样本量偏小,数字方差可能较大;上界实验也只在少数 in-lab 任务上验证。
- **依赖标定与头相机**:桥梁动作定义在头相机坐标系,依赖头相机位姿与手腕位姿投影,对标定与视角一致性有隐性要求。

## 五、评价与展望

**核心贡献与优点。** 本文最有价值的洞见是把"人机迁移"的动作对齐问题**降维**:与其硬啃带噪、语义失配的 6DoF 手腕位姿,不如只保留在共享观察视角下最鲁棒、最 embodiment-agnostic 的三维平移。这一"少即是多"的取舍在实证上很有说服力——桥梁动作对 6DoF 的稳定优势(Table 2)、以及去掉桥梁监督后的成功率断崖(38.33%→12.50%,Table 4)共同支撑了核心主张。工程上,用交错动作 token + 注意力掩码统一处理"分源缺失分量"是干净的做法,避免了拼接补零或多投影头的常见 hack;在机器人数据上随机用桥梁动作替换可执行动作作为预测目标,是把"共享中间表征"真正绑进策略的关键设计。

**与其他公开工作的关系。** 主流人机迁移工作(GR-3、EgoMimic、EgoBridge、EgoScale、Emma 等)大多沿用"人手=6DoF embodiment + 重定向"的路线,或通过拼接/补维/多投影头统一动作空间;本文反其道而行,主动**放弃旋转**。这与 Moto、LAPA 等"学习潜在动作/latent action"作为桥梁语言的路线不同——后者学的是隐式表征,本文用的是显式、物理可解释的平移,可解释性与可控性更好,但表达上界也被平移天然锁死。方法骨架(π0 式 VLA + flow matching + MoT)沿用了 π0/π0.5 一系,创新集中在**动作表征与训练策略**而非模型结构。

**开放问题与可能改进方向。**
1. **有限、可靠的旋转注入**:作者在失败分析里已指明方向——纯平移在开抽屉、插吸管等任务上受限。如何从人类数据里筛出**高置信度的旋转子集**(如接触阶段、低速阶段)做选择性监督,或用不确定性加权,是最自然的下一步。
2. **平移与旋转的解耦学习**:可考虑对旋转单独设一个"低带宽/离散化"的桥梁,与平移桥梁并行,让旋转在噪声可控范围内也参与迁移。
3. **抓取薄物与接触建模**:薄物失败暗示纯运动学桥梁未编码接触/力信息,引入接触事件或触觉/深度线索可能缓解。
4. **可扩展性验证**:上界实验(Table 5)显示随噪声/gap 减小性能上升,若能在更大 embodiment 谱系(不同臂、不同夹爪甚至灵巧手)上验证桥梁动作的可迁移性,将大幅增强"通用桥梁"的说服力。
5. **头相机依赖的放松**:桥梁定义绑在头相机系,若能推广到多视角一致或无标定设定,采集门槛会进一步降低。

总体而言,这是一篇"表征取舍驱动"的扎实实证工作:观点清晰、消融到位、可复现性说明充分(数据规模、训练步数、batch 均有交代),缺点是评测规模偏小、旋转缺失导致的能力天花板明显。它给人机迁移社区提供了一个有力的反例——**不是所有人类动作分量都值得学,选对共享子空间比学全更重要**。

## 参考

1. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.（本文 VLA 骨架与 flow matching 的直接基座）
2. GR-3 Technical Report. arXiv:2507.15493, 2025.（ByteMini 机器人平台与 6DoF 人类动作重定向的代表性基线路线）
3. Hoque et al. *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.* arXiv:2505.11709, 2025.（Stage I 人类预训练的主要数据源之一）
4. Ye et al. *Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos.* ICCV, 2025.（"桥梁语言"思路的隐式表征对照）
5. Lipman et al. *Flow Matching for Generative Modeling.* arXiv:2210.02747, 2022.（动作生成所用 flow matching 的理论来源）
