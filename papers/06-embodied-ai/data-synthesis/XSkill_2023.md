# XSkill：跨本体技能发现

> **论文**：*XSkill: Cross Embodiment Skill Discovery*
>
> **作者**：Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, Shuran Song
>
> **机构**：Columbia University（哥伦比亚大学计算机系）；J.P. Morgan AI Research；Carnegie Mellon University（卡内基梅隆大学计算机学院，emeritus）
>
> **发布时间**：2023 年 07 月（arXiv 2307.09955）
>
> **发表状态**：7th Conference on Robot Learning (CoRL) 2023
>
> 🔗 [arXiv](https://arxiv.org/abs/2307.09955) | [PDF](https://arxiv.org/pdf/2307.09955)
>
> **分类标签**：`跨本体技能迁移` `自监督表征` `人类视频学习` `扩散策略` `一次示范模仿`

---

## 一句话总结

XSkill 用 SwAV 式自监督在**无标注、无分割**的人类视频和机器人遥操作视频上联合学习一组共享的 skill prototypes（借"单本体批内 Sinkhorn 聚类"强制原型跨本体共享），再用 skill-conditioned diffusion policy 把技能落到机器人动作、用 Skill Alignment Transformer (SAT) 把人类示范视频对齐到机器人当前进度,从而只看一段人类示范视频就能一次性完成未见的长程组合任务;仿真跨本体 ×1.5 速差未见任务成功率 70.2%,真实厨房 4 子任务未见组合 60%。

## 一、问题与动机

从人类示范视频学习机器人操作,核心难点是**具身鸿沟 (embodiment gap)**:人类手臂与机器人形态差异大、动作参数不可观测、执行速度也不同(人往往比机器人快)。作者把"从人类示范做模仿学习"拆成三个必备能力:

- **Discover(发现)**:把一段示范分解成一组可复用的子技能;
- **Transfer(迁移)**:把每个观察到的技能映射到机器人自身本体的动作;
- **Compose(组合)**:把学到的技能重新排列组合以完成新任务。

已有工作要么直接行为克隆(不可组合、长程任务差),要么只在单一本体上做技能发现(BUDS 等,仅用机器人数据),要么用人类视频构造奖励再上 RL(部署昂贵)。XSkill 把问题正式定义为**跨本体技能发现 (Cross-Embodiment Skill Discovery)**:纯从无标注人机视频里学到一个跨本体共享的技能表征空间,推理时无需任何任务标签或跨本体对应关系,即可一次示范泛化。

## 二、核心方法

框架三阶段,用三份数据:人类示范集 $\mathcal{D}^h$、机器人遥操作集 $\mathcal{D}^r$(含本体感知 proprioception 与动作)、以及推理时单段人类提示视频 $\tau^h_{\text{prompt}}$。两份训练数据都**未分割、未对齐**。

### 1. Discover：学习共享 skill prototypes

先用 temporal skill encoder $f_{\text{temporal}}$(3 层 CNN 视觉骨干 + 8 层 transformer encoder,附一个可学习 representation token)把每个视频片段编码成技能表征 $z_{ij} = f_{\text{temporal}}(v_{ij})$。为抵消不同本体的执行速度差,从每段视频均匀采 $M$ 帧、用长度 $L$ 的滑窗切片。

关键是引入 $K$ 个**可学习 skill prototypes** $\{c_k\}_{k=1}^{K}$(实现为无偏置的归一化线性层 $f_{\text{prototype}}$),作为连续嵌入空间里的离散锚点。采用 SwAV 式自监督:把同一片段做两次增强,各自投影到原型上用 Softmax 得到预测分布 $p_{ij}$,目标分布 $q_{ij}$ 由另一增强版本经 Sinkhorn-Knopp 在线聚类得到,最小化交叉熵:

$$
\mathcal{L}_{\text{prototype}} = -\sum_{i=1}^{B}\sum_{j=0}^{M}\sum_{k=1}^{K} q_{ij}^{(k)} \log p_{ij}^{(k)}
$$

> 用大白话说:让"同一段动作的两个不同视角/裁剪"落到同一批原型上,靠原型这组共用坐标轴,把不同本体做同一件事的表征拉到一起。

**跨本体对齐的两个诀窍**:(1)Sinkhorn 聚类只在**同一本体的批内**做,故意无视本体差异;(2)Sinkhorn 的熵正则强制"每个原型在每批里都被用到",避免了"不同本体各占嵌入空间不同区域、映射到互不重叠原型"的退化,从而逼迫算法按"技能效果"而非"本体外观"来分组。Sinkhorn 求解的是带熵正则的最优传输:

$$
\max_{Q\in\mathcal{Q}} \operatorname{Tr}\!\big(Q^{\top} C^{\top} Z\big) + \varepsilon H(Q), \qquad Q^{*} = \operatorname{Diag}(u)\,\exp\!\Big(\tfrac{C^{\top}Z}{\varepsilon}\Big)\operatorname{Diag}(v)
$$

> 用大白话说:把这批表征"软分配"到各原型上,同时约束分配得足够均匀(每个原型都被占用),用迭代归一化(论文里只跑 3 次迭代)得到双随机矩阵作为聚类目标。

此外加一项**time contrastive (TCN) 损失**,让时间上相近的片段技能分布相似、相远的相异(InfoNCE 形式):

$$
\mathcal{L}_{\text{tcn}} = -\sum_{i=1}^{B} \log \frac{\exp\!\big(S(p_{ix}, p_{iy})/\tau_{\text{tcn}}\big)}{\exp\!\big(S(p_{ix},p_{iy})/\tau_{\text{tcn}}\big) + \exp\!\big(S(p_{ix},p_{iz})/\tau_{\text{tcn}}\big)}
$$

其中 $p_{ix},p_{iy},p_{iz}$ 分别是锚点、正窗口内、负窗口外片段的原型概率,$S$ 用点积。总发现损失 $\mathcal{L}_{\text{discovery}} = \mathcal{L}_{\text{prototype}} + \mathcal{L}_{\text{tcn}}$。

> 用大白话说:光靠聚类还不够"有时间感",TCN 逼着表征沿时间平滑变化,才能把一段长视频切成"关技能—开技能—转移"这样有序的技能流。

### 2. Transfer：skill-conditioned diffusion policy

在机器人遥操作集 $\mathcal{D}^r$ 上训练一个技能条件扩散策略 $P(a_t\mid s_t, z_t)$(基于 DDPM 的 Diffusion Policy),输入机器人本体感知与视觉观测 $s_t$ 及技能表征 $z_t$,输出长度 $L$ 的动作序列 $a_t=\{a_t,\dots,a_{t+L}\}$;$z_t$ 用训练好的 $f_{\text{temporal}}$ 在观测窗口 $v_t=\{o_t,\dots,o_{t+L}\}$ 上算出。

> 用大白话说:发现阶段学会"认技能",迁移阶段学会"给定一个技能编码,机器人手臂具体怎么动"。用扩散策略是因为它能表达多模态动作分布、少量数据也稳定。

### 3. Compose：从一段人类提示视频完成未见任务

推理时把人类提示视频映射进技能空间,得到技能序列 $\tilde z = \{z_t\}_{t=0}^{T_{\text{prompt}}}$,即一份"任务执行计划"。但直接顺序照搬会脆弱(速度不匹配、失败无法重试)。故引入 **Skill Alignment Transformer (SAT)** $\phi(z_t\mid o_t, \tilde z)$:把 $\tilde z$ 里每个技能当 token、机器人当前观测经 ResNet18 state encoder 编成 state token,transformer 让 state token 关注每个技能 token,判断"哪些技能已完成、下一步该执行哪个技能",再把对齐后的 $z_t$ 喂给扩散策略。SAT 用 MSE 监督预测的 $\hat z_t$ 逼近真实 $z_t$。

> 用大白话说:人比机器人做得快,机械照抄人类视频的时间轴会错位;SAT 相当于一个"看当前状态决定进度"的对齐器,灯已经开了就跳过开灯技能,失败了(比如被人恶意关灯)能自动重规划回到目标。

## 三、实验结果

**环境**:Franka Kitchen 仿真(7 子任务、580 条机器人示范;另造一个视觉迥异的"球形 agent"并下采样其示范来模拟本体+速度差);自建 Realworld Kitchen(UR5 + WSG50 夹爪,4 子任务:开烤箱/抓布/关抽屉/开灯,175 条人类 + 175 条遥操作示范)。评价指标 = 完成子任务数 / 总子任务数,且要求按提示视频的**顺序**完成。仿真每方法测 32 种初始条件,真机每任务 10 次。

**表 1:仿真结果 (%)**(Cross Embodiment 下 ×1/×1.3/×1.5 为提示视频相对机器人的加速倍率)

| 方法 | Same ×1 | Cross ×1 | Cross ×1.3 | Cross ×1.5 | Avg |
|---|---|---|---|---|---|
| GCD Policy | 91.4 | 0.00 | 0.00 | 0.00 | 22.8 |
| GCD Policy w. TCN | 2.50 | 3.55 | 2.00 | 1.25 | 2.32 |
| XSkill w. NN-compose | 93.7 | 61.2 | 23.4 | 15.2 | 48.4 |
| XSkill w.o proto. loss | 80.1 | 56.3 | 12.5 | 3.75 | 38.2 |
| **XSkill** | **95.8** | **89.4** | **83.7** | **70.2** | **84.8** |

要点:goal-conditioned diffusion (GCD) 在同本体下也能到 91.4,但跨本体直接归零;XSkill 跨本体同速 89.4、相比同本体只掉约 5%,且在 ×1.5 大速差下仍 70.2。SAT 相对最近邻组合 (NN-compose) 在跨本体提示下领先 >50%(如 ×1.5:70.2 vs 15.2)。

**表 2:真机结果 (%)**(3 子任务分 Seen/Unseen、Same/Cross 本体;4 子任务仅 Cross Unseen)

| 方法 | 3子·Same·Seen | 3子·Same·Unseen | 3子·Cross·Seen | 3子·Cross·Unseen | 4子·Cross·Unseen | Avg |
|---|---|---|---|---|---|---|
| GCD policy | 68.3 | 53.3 | 0.00 | 0.00 | 0.00 | 24.3 |
| GCD w. TCN | 25.0 | 22.2 | 26.7 | 23.3 | 15.6 | 22.6 |
| **XSkill** | **86.7** | **80.0** | **81.7** | **76.7** | **60.0** | **77.0** |

**关键消融**:

- **去原型损失** (XSkill w.o proto. loss):同本体尚有 80.1,但速差一大(×1.5)骤降到 3.75,说明 skill prototypes 是学"形态不变、速度不敏感"表征的关键。
- **去 TCN 损失**(表 A3):同本体从 95.8 崩到 3.75,证明时间对比对捕捉技能时序至关重要。
- **原型数 $K$**(表 A2):$K=32$ 偏小会限制表征容量(跨本体 ×1.5 仅 48.7),$K=128/256/512$ 表现相近(70.2/76.7/71.8);因为投影前的连续表征 $z$ 才是喂给策略/SAT 的,$K$ 的具体取值不太敏感,仿真用 128、真机用 32。
- **未见转移泛化**(表 A1):训练时移除 25%/50% 的转移组合,$K=512$ 在 Level 2(移除 50%)跨本体同速仍达 84.4,更大 $K$ 更利于插值泛化。

## 四、局限性

- **需预设原型数 $K$**:虽不敏感,但不同数据集仍需调参(7 子任务仿真宜大 $K$,4 子任务真机 $K=32$ 即可)。
- **依赖机器人遥操作数据的多样性**:真机里"抓布后关抽屉"这类转移在遥操作数据中不存在,机器人就做不出;Drawer 子任务因缺乏多模态数据而在未见转移上失败。这是真机表现受限的主因。
- **相机/场景单一**:当前 benchmark 的人机视频都来自同一实验室相机布置,尚未验证真正 in-the-wild(如 YouTube)、多相机多环境下的可迁移性。
- **必须严格按提示视频顺序**:执行未演示子任务即判 episode 结束,不支持自由重排。
- 仍需机器人端遥操作数据训练扩散策略,并非纯从人类视频零机器人数据学策略。

## 五、评价与展望

**优点**:(1)把"跨本体技能发现"清晰形式化,并给出一条不进 RL 环、纯自监督的可扩展路线——人类示范便宜、非专家也能采,数据成本显著低于纯机器人示范;(2)最漂亮的设计是"单本体批内 Sinkhorn + 强制原型全用"这一招,用聚类的均匀性约束把跨本体对齐问题转化成"共享原型"这一隐式对齐,无需任何配对标签或视频翻译;(3)SAT 用状态判进度来吸收速度差与失败,实证相对最近邻对齐大幅领先,是把"离线示范计划"落到"闭环执行"的实用组件;(4)同时发布仿真+真机跨本体 benchmark 与代码,利于复现。

**与相关工作的关系**:相比 XIRL(cross-embodiment 逆强化学习)、Concept2Robot、AVID 等需构造奖励再上 RL 的路线,XSkill 无 RL、聚焦一次模仿,部署更省;相比 MimicPlay(学跨本体计划隐空间以减少机器人示范采集),XSkill 更强调学一个可复用的技能表征以降低推理期对机器人示范的依赖;相比 BUDS 等单本体技能发现,它把 SwAV/Sinkhorn(DeepCluster、SwAV 谱系)迁到跨本体设定;策略侧直接站在 Diffusion Policy 肩上。可看作"自监督视觉表征聚类 + 分层模仿 + 扩散策略"的一次成功缝合。

**开放问题与可能改进**:(1)$K$ 与原型作为离散锚点带来粒度上限,未来可探索连续/层次化的技能码本或可变原型数;(2)对齐仍限于近似同视角,真正 in-the-wild 人类视频(大视角、背景、手型差异)下 $f_{\text{temporal}}$ 的鲁棒性存疑,可结合更强的通用视觉表征(如 R3M/DINO 类)或显式手-物交互先验;(3)真机瓶颈在机器人侧转移覆盖不足,提示"技能发现"与"策略数据采集"应联合设计,或引入生成式/世界模型手段补齐缺失转移;(4)严格顺序约束限制了组合自由度,可引入基于目标/语言的高层规划器替代对提示视频顺序的硬依赖;(5)技能表征是否可跨任务域、跨机器人平台复用(而非同一厨房),仍待更大规模多环境数据验证。

## 参考

1. M. Caron et al. *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)*. NeurIPS 2020.（原型聚类自监督的直接来源）
2. C. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS 2023 / arXiv:2303.04137.（技能条件策略骨干）
3. K. Zakka et al. *XIRL: Cross-embodiment Inverse Reinforcement Learning*. CoRL 2021.（跨本体奖励/表征的对照路线）
4. C. Wang et al. *MimicPlay: Long-Horizon Imitation Learning by Watching Human Play*. arXiv:2302.12422, 2023.（跨本体计划隐空间,最相关同期工作）
5. P. Sermanet et al. *Time-Contrastive Networks (TCN)*. 2018.（TCN 时间对比损失来源）
