# ACE-Ego-0：统一第一人称人类视频与机器人数据的 VLA 预训练

> **论文**：*ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining*
>
> **作者**：Hao Li, Ganlong Zhao, Yufei Liu, Haotian Hou, Guoquan Ye, Tongyan Fang, Chunxiao Liu, Siyuan Huang, Jianbo Liu, Xiaogang Wang, Hongsheng Li et al.
>
> **机构**：ACE Robotics；CUHK MMLab；香港中文大学(深圳)；上海交通大学；清华大学
>
> **发布时间**：2026 年 06 月（arXiv 2606.17200）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.17200) | [PDF](https://arxiv.org/pdf/2606.17200)
>
> **分类标签**：`VLA 预训练` `人机数据统一` `第一人称视频伪动作` `flow-matching`

---

## 一句话总结

ACE-Ego-0 把第一人称人类视频、多本体机器人示范与仿真数据统一到"以头戴相机为坐标原点"的规范动作空间里，通过 morphology token(结构对齐)+ 时间对齐动作分块(时序对齐)+ 可靠性感知辅助损失(监督质量对齐)做联合预训练；在 1.48K 小时人类伪动作 + 4.53K 小时机器人数据上训练后，RoboCasa GR1 TableTop 达 72.8%、RoboTwin 2.0 Easy/Hard 达 91.12%/90.62%，真机 ARX 双臂六任务平均 78.3%(π0.5 为 71.7%,GR00T-N1.7 仅 35.6%)。

## 一、问题与动机

VLA 模型的能力强依赖于大规模、多样的具身数据，但真机遥操作示范采集昂贵、行为多样性受限。第一人称人类视频是互补且廉价的监督来源，但把人类视频与机器人数据放到一起联合训练面临两类根本障碍：

- **表示异质性(representation heterogeneity)**：不同来源在(1)动作坐标系(机器人示范多在世界系,MANO 手部重建在局部系)、(2)运动学结构(不同关节链、关节限位、物理尺寸)、(3)控制频率(10-30 Hz 不等)三个维度都不一致。
- **监督质量失配(supervision-quality mismatch)**：机器人是传感器记录的高保真动作,人类视频只能通过视觉管线反推出**伪动作(pseudo-action)**,天然带有跟踪抖动、遮挡、估计偏差。作者指出,现有工作要么绕过动作级训练(只学视觉表征/奖励),要么把噪声伪动作直接塞进和干净机器人数据相同的 BC/diffusion 目标里——后者等于逼策略去模仿重建管线的伪影和失败。

论文主张:**表示对齐** 和**监督质量** 这两条轴,当前混合源 VLA 预训练框架都没有彻底解决,ACE-Ego-0 要同时解决。

## 二、核心方法

框架分两大块:**统一动作表示**(Sec. 3.1)把异质数据在空间/结构/时序上对齐,**可靠性感知训练目标**(Sec. 3.2)把噪声人类伪动作当"辅助监督"而非"平等监督"。骨干为 Qwen3-VL-4B-Instruct + 约 600M 的 flow-matching DiT 动作专家,头+腕双相机 256×256 输入,推理 4 步 flow-matching 解码。

### 1. 统一动作表示

**(a) 规范相机空间(空间对齐)。** 把机器人和人手的动作全部表达在头戴相机坐标系里。机器人末端位姿由标定外参投影:

$$p_{\mathrm{cam}} = R_{\mathrm{cam}\leftarrow s}\, p_s + t_{\mathrm{cam}\leftarrow s}, \qquad R_{\mathrm{cam},ee} = R_{\mathrm{cam}\leftarrow s}\, R_{s,ee}$$

用大白话说:所有平台都换算到"头相机看到的坐标系",策略就不用再各自学"世界系→相机系"的变换;换个机器人只要在推理时替换一个相机外参即可。姿态用连续 6D 表示(取旋转矩阵前两列)避免四元数/欧拉角的不连续。人手没有物理末端,于是把腕关节定为末端原点(HaMeR 逐帧预测里最稳),用手掌平面+腕到指向量构造稳定的手心朝向系,再用归一化的**拇指到掌心距离** 当夹爪开合代理,线性缩放到机器人夹爪行程。最终每只手 16 维、训练时转 6D 得双臂 22 维统一动作向量 $a=\langle a_{\mathrm{left}}, a_{\mathrm{right}}\rangle$,每臂 = 3 位置 + 6D 旋转 + 1 夹爪 + 1 活动标志。

**(b) 跨本体 morphology 条件(结构对齐)。** 给动作专家喂一个 morphology token:

$$h_{\mathrm{morph}} = \begin{cases} P_{\mathrm{morph}}(E_{\mathrm{urdf}}(\mathcal{G}_r)), & \text{机器人来源 } r \\ P_{\mathrm{surr}}(e_d), & \text{人类来源 } d \end{cases}$$

用大白话说:给解码端一张"身体说明书"——机器人用它的 URDF 运动学图(每关节 29 维描述,经消息传递 GNN 池化成 body + manipulation-chain 两个摘要)编码;人手没有 URDF,就用一个**可学习的替身嵌入** $e_d$(按数据源共享,端到端反传学到)。关键设计是这个 morphology token **只注入动作解码端、不进 VLM 主干**,让语言视觉主干保持"不认身体"的本体无关性。

**(c) 时间对齐动作分块(时序对齐)。** 不按固定步数、而按固定物理时长 $T^\star$ 定义动作块,给定控制频率 $f_d$:

$$H_d = \mathrm{round}(f_d\, T^\star)$$

用大白话说:各数据集频率不同,若都预测 N 步,物理时长就对不齐;改成"都规划未来 $T^\star$ 秒"(默认 2 s),按各自频率折算步数,保证大家规划的是同一段未来时间窗。为控制变长块带来的 padding 开销,用归一化 episode 相位 $\phi=\mathrm{clip}((t+\tfrac12 H_d)/L_e,0,1)$ 与复合键 $k=(c_{\mathrm{task}}, b_\phi, b_H)$ 做结构化分桶采样,同任务同 horizon 聚到一个 batch。

### 2. 可靠性感知训练目标

核心是给人手伪动作的每个通道 $j$、每个时刻 $t$ 打一个时空可靠度权重:

$$W_{t,j} = \rho_j \cdot w_{t,j} = \rho_j \cdot w_{\mathrm{data}}(d,h(j)) \cdot w_{\mathrm{step}}(t,h(j))$$

用大白话说:这个权重是三项相乘——静态通道先验 $\rho_j$(位置通道 $\rho=1.0$,腕旋转/夹爪 $\rho=0.001$,因为姿态估计器在这些维度噪声大)、数据集级质量先验 $w_{\mathrm{data}}$(按各源存活帧比例和抖动中位数估,范围 [0.25,1.0])、以及步级平滑因子 $w_{\mathrm{step}}$(对局部大跳变/大 jerk 软性压权)。

机器人干净数据走**标准 flow-matching 主损失**:

$$\mathcal{L}_{\mathrm{action}} = \mathbb{E}_{s,\epsilon}\sum_{t,j} M_{t,j}\,\lVert \hat v_\theta(a_s,s)_{t,j} - (a-\epsilon)_{t,j}\rVert^2$$

用大白话说:让网络在相机系下预测从噪声到真实 delta 动作的速度场,这是撑起主要控制能力的锚。

人手数据走**可靠性加权的鲁棒辅助损失**(Huber):

$$\mathcal{L}_{\mathrm{haux}} = \mathbb{E}_{s,\epsilon}\,\frac{1}{Z}\sum_{t,j} M_{t,j}\,W_{t,j}\,\mathrm{Huber}_\beta\big(\hat v_\theta(a_s,s)_{t,j} - (\tilde a-\epsilon)_{t,j}\big)$$

用大白话说:人手数据只当"安全补充",用鲁棒 Huber + 可靠度加权,监督几乎集中在可信的位置通道上(6 维腕部 xyz),噪声大的旋转/夹爪被 $\rho=0.001$ 压到近乎不监督;目标 $\tilde a$ 还先做 3 帧时间平滑去手部重建抖动。总目标:

$$\mathcal{L} = \mathcal{L}_{\mathrm{action}} + \lambda_{\mathrm{haux}}\,\mathcal{L}_{\mathrm{haux}}, \qquad \lambda_{\mathrm{haux}}=0.1$$

### 3. 五阶段第一人称视频→动作数据管线

在 6 个人类视频数据集上跑五阶段管线,产出 1,478 小时伪动作标注视频:(1) 数据整理(4-30 s 切片);(2) 视频筛选(ego-interaction 过滤 + 强人脸检测剔除观察者视角 + caption 需含操作动词与可操作物体名词);(3) 3D 手部重建(SAM3 跟踪 + HaMeR 重建 MANO + 借 VIPE 相机位姿做两阶段全局轨迹优化,含二阶差分平滑项 $\mathcal{L}_{\mathrm{smooth}}$);(4) 动作参数化(手心系→22 维);(5) 质量控制(Static/Spike/Completeness/Bimanual 四道过滤)。

## 三、实验结果

**预训练数据池(约 6.0K+ 小时)**：

| 类型 | 主要来源 | 小时数 | 监督 |
|---|---|---|---|
| 人类视频 | Ego4D / EgoExo4D / EPIC-KITCHENS-100 / HOI4D / EgoDex / Xperience-10M | 1,478.9 | 伪动作 |
| 机器人 | AgiBot Alpha/Beta(1937.8)、Galaxea R1Lite(488.1)、AgiBot DigitalWorld(仿真,225.3)、RoboCasa Tabletop(仿真,83.6)、Galbot 自采(1800+) | 4,534.8+ | 真机动作 |
| 合计 | 约 176 万 episode / 约 6.04 亿帧 | 6,013.7+ | 混合 |

**RoboCasa GR1 TableTop(24 任务,每任务 50 次)平均成功率**：

| 方法 | 平均 (%) |
|---|---|
| GR00T-N1.6 | 47.6 |
| Qwen3PI | 43.9 |
| FLARE | 55.0 |
| ABot-M0 | 58.3 |
| JoyAI-RA | 63.2 |
| DIAL | 70.2 |
| **ACE-Ego-0** | **72.8** |

**RoboTwin 2.0(50 任务,每任务 100 次)平均成功率**：

| 方法 | Easy | Hard |
|---|---|---|
| π0 | 65.92 | 58.40 |
| π0.5 | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| LingBot-VLA | 88.56 | 86.68 |
| ABot-M0 | 86.06 | 85.08 |
| JoyAI-RA | 90.48 | 89.28 |
| Hy-VLA | 90.9 | 90.1 |
| **ACE-Ego-0** | **91.12** | **90.62** |

**真机 ARX 双臂(6 任务,每任务 30 次)**：ACE-Ego-0 平均 **78.3%**,π0.5 71.7%(+6.6),GR00T-N1.7 仅 35.6%。接触密集的 Scoop Coffee 需双臂紧协同,ACE-Ego-0 86.7% vs π0.5 70.0% vs GR00T-N1.7 36.7%(领先后者 50 个点)。Category Sorting 稳定 90.0%(GR00T-N1.7 仅 30.0%)。

**消融(RoboCasa,190K 步)**：

| 配置 | 成功率 (%) | Δ |
|---|---|---|
| 完整模型 | 72.8 | — |
| 去时间对齐分块 | 71.7 | −1.1 |
| 去 URDF/morphology 条件 | 70.9 | −1.9 |
| 去可靠性感知人手损失 | 69.2 | −3.6 |

数据源消融:从 Qwen 直接起(无具身预训练)65.4% → 加机器人数据 68.3%(+2.9)→ 再加人类视频 72.8%(+4.5,单项最大增益)。数据稀缺微调实验(Sweep Cubes):仅 34 条机器人示范时成功率 10%;混入 419 条任务匹配人类视频(约 11.75 万帧)升到 40%(**4× 提升**)——因为 34 条机器人示范只覆盖 0.062 m² 末端工作空间,人类视频覆盖 0.296 m²(宽 4.8×)。

## 四、局限性

- 评测仅限桌面操作,尚未验证移动操作、全身人形控制、可变形物体等更大空间约定和更长时序的场景。
- 预训练池不含灵巧手数据与力/力矩传感,接触密集任务的表征仍有限。
- 伪动作管线在旋转和精细手指运动上保真度不足,导致可靠性感知目标目前几乎只能监督位置通道(腕 xyz),旋转/夹爪基本被压掉——即人类监督实际只贡献了"位置覆盖度",姿态维度的迁移价值尚未打开。
- 最长时序、含精细合盖阶段的 Pack Shoes 上所有方法都明显掉点,说明长时程轨迹漂移的累积仍是现有预训练 VLA 架构的共性难题。

## 五、评价与展望

**优点。** (1) 把"人机数据统一"这个老问题拆成空间/结构/时序/监督质量四条正交轴并各给一个干净的机制,概念清晰、可消融,消融也确实每项都掉点(尤其可靠性损失 −3.6 最关键),说明"平等对待噪声伪动作"确实是主要毒源。(2) 规范相机空间 + morphology token 只进解码端、不污染 VLM 主干的设计,让骨干保持本体无关、换本体只换外参/注册 URDF,工程上很讨巧,与 π0/π0.5、GR00T-N1、OpenVLA 一系"共享动作空间/embodiment tokenizer"路线互补。(3) 把人手监督限制在高可信通道 + Huber + 逐源/逐步动态加权,是对 EgoMimic/EgoVLA/H2R 等"直接把手部轨迹当动作代理"做法的一个务实修正,数据稀缺微调 4× 的结果对"人类视频补覆盖度"这一价值给了直接证据。

**缺点与开放问题。** (1) RoboTwin 2.0 上相对最强基线 JoyAI-RA 仅领先 0.64/1.34 个点,Hy-VLA 也非常接近,统计显著性存疑,论文未报方差/多种子;真正拉开差距的是真机(尤其对 GR00T-N1.7)与数据稀缺微调,而这些设置里基线是否充分调优不易核验。(2) 人类监督"只在位置维度起效",本质上和"用人类视频做视觉/覆盖度先验"相差不远,论文标榜的旋转/夹爪级动作监督尚未兑现——这也是最有价值、最难的部分。(3) morphology token 靠反传学人类替身嵌入,泛化到未见本体/未见人群时的行为未评测。(4) 数据管线依赖 HaMeR/VIPE/SAM3 一长串外部模型,伪动作质量的上限被这些管线锁死,与 DIAL 那类"用潜在世界模型解耦意图与低层动作、绕开显式重建"的路线值得系统对比。可能的改进方向:引入 force/触觉或潜在动作以补足姿态维度监督、把可靠度权重做成可学习而非手工先验、以及在长时程任务上引入分层/记忆机制缓解轨迹漂移。

## 参考

1. Black et al. *π0.5: a VLA model with open-world generalization.* CoRL 2025.(flow-matching VLA、delta 动作块与本文主损失同源)
2. NVIDIA et al. *GR00T N1: An open foundation model for generalist humanoid robots.* arXiv 2503.14734, 2025.(跨本体基础模型、RoboCasa 基准来源)
3. Hoque et al. *EgoDex: Learning dexterous manipulation from large-scale egocentric video.* arXiv 2505.11709, 2025.(本文最大单一人类视频源,776.8 h)
4. Pavlakos et al. *Reconstructing hands in 3D with transformers (HaMeR).* CVPR 2024.(手部 MANO 重建,伪动作管线第 3 阶段核心)
5. Chen et al. *DIAL: Decoupling intent and action via latent world modeling for end-to-end VLA.* arXiv 2603.29844, 2026.(将人类视频引入 VLA 的替代路线,RoboCasa 强基线)
