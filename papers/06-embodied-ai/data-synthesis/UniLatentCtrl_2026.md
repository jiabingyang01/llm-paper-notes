# UniLatentCtrl：面向跨具身人形机器人控制的统一隐空间学习

> **论文**：*Learning a Unified Latent Space for Cross-Embodiment Robot Control*
>
> **作者**：Yashuai Yan, Dongheui Lee
>
> **机构**：Autonomous Systems Lab, Technische Universität Wien（TU Wien，维也纳）；Dongheui Lee 同时任职于 Institute of Robotics and Mechatronics（DLR，德国航空航天中心）
>
> **发布时间**：2026 年 01 月（arXiv 2601.15419）
>
> **发表状态**：未录用（预印本；论文排版显示投稿至 *IEEE Robotics & Automation Magazine* 具身智能特刊）
>
> 🔗 [arXiv](https://arxiv.org/abs/2601.15419) | [PDF](https://arxiv.org/pdf/2601.15419)
>
> **分类标签**：`跨本体控制` `motion retargeting` `隐空间学习` `对比学习` `目标条件策略` `人形机器人`

---

## 一句话总结

用按身体部位解耦的隐空间（左臂/右臂/躯干/左腿/右腿五个子空间）加对比学习，把人类与多种人形/机械臂机器人的动作对齐进同一共享隐空间，再仅用人类动作数据训练一个目标条件 c-VAE 控制策略，训练好的策略无需任何微调即可直接部署到 7 种机器人（TIAGo++、H1、NAO、JVRC、ATLAS、G1、Kinova Gen3），新增机器人只需再训练一个轻量的机器人专属 embedding 层（约 15 分钟），目标到达误差多数达到亚厘米级（如 NAO 仅 0.13 cm），控制频率约 100 Hz。

## 一、问题与动机

跨具身模仿学习的长期目标是让控制策略无需重训练或专用微调就能跨不同机器人形态迁移。已有工作大多学习任务相关、领域不变的表征以桥接人类和单个机器人之间的差异（如作者前作 ImitationNet，IEEE-RAS Humanoids 2023），或需要人工采集的成对示范数据来学习共享表征，数据瓶颈严重、难以扩展到新平台。本文要解决的是更广的问题：学习一个跨越人类 + 多种机器人（单臂、双臂、腿式人形机器人）的**统一**隐空间，并直接在该隐空间内训练一个目标条件控制策略，使同一个策略能够控制隐空间所编码的任意机器人，同时支持以最小代价接入新机器人。

## 二、核心方法

方法分两阶段：（1）学习统一隐空间；（2）在隐空间内训练目标条件控制策略。

**架构**：人类编码器 $E_h$、跨具身编码器 $E_X$、跨具身解码器 $D_X$ 三者在人类和所有机器人之间共享；每个机器人额外拥有自己的可学习 embedding 层 $E_r$（把该机器人原始关节表示投影到统一的 1024 维跨具身特征空间）及其逆映射 $D_r$（把共享表示投回该机器人的关节角）。所有编码器/解码器均为 8 层 MLP（每层 256 个神经元，ELU 激活，输出层 Tanh）。

**解耦隐空间**：不同于此前用单一整体隐空间表示全身动作的做法（这类做法在自由度差异很大的机器人间容易产生歧义映射），本文把隐空间拆成 LA（左臂）、RA（右臂）、TK（躯干）、LL（左腿）、RL（右腿）五个各 16 维、取值范围 $[-1,1]$ 的子空间，分别建模，从而更好地处理局部/非对称肢体结构（如 TIAGo 只有手臂、NAO/H1 躯干活动度有限）。

**分部位相似度度量**：旋转相似度基于四元数点积：

$$D_R(\mathbf{x}_A,\mathbf{x}_B)=\sum_j\left(1-\langle q_A^j,q_B^j\rangle^2\right)$$

用大白话说：两个姿态对应关节的四元数越接近（旋转越相似），该项越接近 0。手臂末端位置相似度（先按肩部坐标系归一化、再除以臂长以消除尺度差异）：

$$D_{ee}(\mathbf{x}_A,\mathbf{x}_B)=\|p_A^{ee}-p_B^{ee}\|_2$$

对左右臂子空间用旋转+末端位置的加权和作为相似度：

$$S_k(\mathbf{x}_A,\mathbf{x}_B)=D_R(\mathbf{x}_A,\mathbf{x}_B)+\omega D_{ee}(\mathbf{x}_A,\mathbf{x}_B),\quad k\in\{LA,RA\}$$

其余部位（躯干、双腿）仅用旋转相似度 $S_k=D_R$，因为这些部位关注姿态模仿而非末端精度。

**训练目标**：对每个子空间用三元组对比损失（triplet loss，margin $\alpha=0.05$）拉近语义相似动作、推开不相似动作：

$$\mathcal{L}_{contrastive}=\sum_{\mathcal S}\sum_{(z_i^o,z_j^+,z_k^-)}\max\!\left(\|z_i^o-z_j^+\|_2-\|z_i^o-z_k^-\|_2+\alpha,\,0\right)$$

机器人侧有成对数据（原始姿态→重建姿态）可以直接用重建损失 $\mathcal{L}_{rec}=\|\mathbf{x}_A-\hat{\mathbf{x}}_A\|_2$ 约束；但人类动作没有配对机器人数据，因此借鉴 ImitationNet 的循环损失思路，用**隐一致性损失**约束"解码再编码"后仍落回原隐表示附近：

$$\mathcal{L}_{ltc}=\|E_h(\mathbf{x}_H)-E_X(D_X(E_h(\mathbf{x}_H)))\|_2$$

再加一个时序损失，让人类手部速度与重定向后机器人末端执行器速度保持一致（$\mathcal{L}_{temporal}=\|v_H^{hand}-v_A^{ee}\|_2$），保证动作平滑连贯。总损失为加权和：

$$\mathcal{L}_{total}=\lambda_c\mathcal{L}_{contrastive}+\lambda_{rec}\mathcal{L}_{rec}+\lambda_{ltc}\mathcal{L}_{ltc}+\lambda_{temp}\mathcal{L}_{temporal}$$

（实验中 $\lambda_c=10,\lambda_{rec}=5,\lambda_{ltc}=1,\lambda_{temp}=0.1$）。

**阶段二：隐空间目标条件控制**。用条件变分自编码器（c-VAE）在隐空间内预测隐位移 $d_t=z_{t+1}-z_t$，条件为当前隐状态 $z_t$ 与指向目标的平均末端速度意图信号 $\bar v_{ee}$（由人类手部三帧位置估计）。训练目标为重建损失加 KL 正则：$\mathcal{L}_{cvae}=\mathcal{L}_{reconstruction}+\lambda_{KL}\mathcal{L}_{KL}$（$\lambda_{KL}=10^{-4}$）。推理时给定目标末端位置和时间视界，以自回归方式迭代更新隐状态并解码为机器人关节指令，全程约 100 Hz。

**新增机器人的可扩展性**：核心网络 $E_h,E_X,D_X$ 先在四个主力机器人（TIAGo++、H1、NAO、JVRC）上端到端联合训练；接入新机器人（ATLAS、G1、Kinova Gen3）时冻结核心网络，仅训练该机器人的 $E_r,D_r$ embedding 层，训练时间从数小时降到约 15 分钟。

**数据构造**：训练数据来自 HumanML3D（29,224 条人类动作序列，超 400 万帧姿态），机器人侧不采集任何真实动作数据，而是在各自关节空间中均匀随机采样并用 PyTorch-Kinematics 在 GPU 上批量做正运动学（FK），每步生成超 $10^5$ 个新机器人姿态、用完即弃，训练全程共"看过"数十亿个合成机器人姿态，无需存储任何机器人动作数据集。

## 三、关键结果

**人到机器人重定向**（RS 越低越好，虽命名 Similarity 实为角度误差量，单位度；NDS/NVS 为归一化距离/速度误差）：

| 目标机器人 | RS ImitationNet | RS Ours(解耦) | NDS ImitationNet | NDS Ours(解耦) | NVS ImitationNet | NVS Ours(解耦) |
|---|---|---|---|---|---|---|
| TIAGo++ | 0.7183 | 3.8293 | 0.1325 | 0.0401 | 0.3762 | 0.1071 |
| H1 | 0.6483 | 1.0947 | 0.1081 | 0.0263 | 0.2881 | 0.0962 |
| NAO | 0.6685 | 2.7097 | 0.1596 | 0.0566 | 0.3682 | 0.1448 |
| JVRC | 0.5792 | 1.4631 | 0.1006 | 0.0288 | 0.2383 | 0.0862 |

ImitationNet 为每个机器人单独训练专属模型，旋转精度（RS）明显更优；但本文用**一个统一模型**覆盖所有机器人，在末端位置/速度精度（NDS/NVS）上全面反超 ImitationNet。消融显示解耦隐空间相比整体耦合隐空间大幅改善旋转精度（如 H1 从耦合 2.1268 降到解耦 1.0947），同时保持 NDS/NVS 优势。机器人到机器人重定向（如 JVRC→H1、TIAGo++→JVRC，无 ImitationNet 基线可比）同样显示解耦设计全面优于耦合设计。

**新增机器人**（仅训练 embedding 层）：Human→G1 RS 1.2124 / NDS 0.0396；Human→ATLAS RS 1.4008 / NDS 0.0336；Human→Kinova RS 3.4249 / NDS 0.0491；跨机器人如 ATLAS→JVRC、TIAGo++→ATLAS 等精度与端到端训练的四个主力机器人相当，验证了轻量扩展方案的有效性。

**隐空间目标条件控制**（1000 次随机初始/目标位姿实验，末端到达目标距离，单位 cm）：

| 机器人 | TIAGo | H1 | NAO | JVRC | ATLAS |
|---|---|---|---|---|---|
| c-VAE 到目标距离 | 1.14 | 0.44 | 0.13 | 0.45 | 0.56 |

除 TIAGo 略超 1 cm 外均为亚厘米级精度，控制频率约 100 Hz。真实硬件上在 TIAGo-SEA（双臂）和 Kinova Gen3 上做了 RGB 相机实时遥操作（拾取放置、双臂物体传递）、机器人到机器人重定向（Kinova→TIAGo-SEA 手臂）以及隐空间直接控制完成抓取放置任务的验证。

## 四、评价与展望

**优点**：解耦式分部位隐空间设计针对性解决了"整体隐空间在自由度差异大的机器人间产生歧义映射"这一具体痛点，思路简洁且消融验证充分；完全用随机采样 + GPU 端 FK 合成机器人姿态、无需采集或存储任何机器人真实动作数据，是一种轻量但有效的数据构造方式，契合"scalable data engine"的思路；新机器人仅需训练 embedding 层（约 15 分钟）即可获得与端到端训练相当的效果，可扩展性上有实用价值；同时给出了仿真定量评测 + 真实双臂/单臂硬件遥操作的双重验证。

**局限**（含作者自述）：（1）作者在文中明确指出，训练数据 HumanML3D 基于 SMPL 人体模型，SMPL 不单独建模手部而是把手视为前臂的一部分，因此本文完全未处理手部动作重定向，限制了对精细操作/灵巧遥操作等场景的适用性，作者将其列为未来工作方向（拟引入或采集手部数据集）。（2）方法本质上是纯运动学层面的姿态对齐，训练策略所用的机器人姿态由随机关节采样 + FK 生成，不含物理动力学、碰撞检测或抓取语义，策略学到的是"到达目标位置"而非"完成任务"，真实抓取放置演示仍依赖遥操作或额外的任务策略衔接。（3）旋转相似度指标（RS）显示统一隐空间相对专属模型（ImitationNet）在关节朝向精度上仍有明显差距（如 TIAGo++ 上 3.8 度 vs 0.7 度），解耦只是部分缓解而非完全弥合该 gap。（4）相似度公式中的权重 $\omega$、损失权重 $\lambda_c$ 等超参数需要手工网格搜索（论文附消融表），对新机器人形态的迁移性未知。

**与相关工作的关系**：方法直接建立在作者前作 ImitationNet（IEEE-RAS Humanoids 2023，人到单一机器人的对比学习隐空间）之上，核心推进是把"单一机器人"泛化为"任意数量机器人共享同一隐空间"，并新增了目标条件控制策略这一下游应用层；与 Xirl、Xskill 等跨具身表征学习工作相比，本文更强调运动学层面的分部位结构化设计和面向人形机器人全身/局部躯体的精细对齐，而非任务级别的技能发现。

**开放问题**：如何在该隐空间框架中引入手部/末端执行器的精细动作与接触信息；如何将纯运动学重定向与物理仿真中的动力学可行性、碰撞约束结合，使隐空间轨迹天然满足机器人本体约束；解耦子空间之间目前独立训练，肢体间协调性（如全身平衡、步态与上肢动作耦合）如何在该框架下保证，也是值得后续探讨的方向。

## 参考

- Yan et al., "ImitationNet: Unsupervised human-to-robot motion retargeting via shared latent space," IEEE-RAS Humanoids, 2023.（本文直接基础）
- Guo et al., "Generating diverse and natural 3D human motions from text," CVPR 2022.（HumanML3D 数据集来源）
- Zakka et al., "XIRL: Cross-embodiment inverse reinforcement learning," CoRL 2022.
- Xu et al., "XSkill: Cross embodiment skill discovery," CoRL 2023.
- Choi et al., "Self-supervised motion retargeting with safety guarantee," ICRA 2021.
