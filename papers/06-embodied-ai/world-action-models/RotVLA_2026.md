# RotVLA：面向 VLA 模型的旋转型隐动作表示

> **论文**：*RotVLA: Rotational Latent Action for Vision-Language-Action Model*
>
> **作者**：Qiwei Li、Xicheng Gong、Xinghang Li、Peiyan Li、Quanyun Zhou、Hangjun Ye、Jiahuan Zhou（通讯）、Yadong Mu（通讯）
>
> **机构**：Wangxuan Institute of Computer Technology, Peking University；Xiaomi Robotics；CASIA
>
> **发布时间**：2026 年 05 月（arXiv 2605.13403）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.13403) | [PDF](https://arxiv.org/pdf/2605.13403)
>
> **分类标签**：`隐动作模型` `SO(n) 旋转表示` `flow matching` `VLA 预训练` `跨具身学习` `RoboTwin2.0`

---

## 一句话总结

把隐动作（latent action）从离散 VQ token 换成 SO(n) 旋转矩阵，并用三帧 triplet 学习框架强制"两步隐动作的矩阵乘法组合"要能重建第三帧，从而在不引入动作标注的前提下防止隐动作退化为纯外观编码；仅 1.7B 参数即在 LIBERO 上做到 98.2%、RoboTwin2.0 上做到 89.6%/88.5%（clean/randomized），超过参数量更大的 GR00T-N1.6（3B）、UniVLA*（9B）等基线,真实机器人三任务上也全面优于 π0.5。

## 一、问题与动机

VLA 预训练面临的核心难题是如何统一使用跨具身、跨数据源（含无动作标注的人类视频）的异构数据。隐动作模型（Latent Action Model, LAM）是主流方案之一：用编码器从连续两帧 $I_t, I_{t+1}$ 中推断出一个隐动作 $z_{t\to t+1}$，再用解码器（前向动力学模型）以 $I_t$ 和 $z_{t\to t+1}$ 重建 $I_{t+1}$，从而无需真实动作标注即可学到一个跨embodiment 共享的动作空间。

但作者指出主流 LAM（沿用 VQ-VAE 离散量化）存在三个耦合缺陷：

1. **退化风险**：encode-decode 范式容易退化为"直接编码目标帧"的平凡解，即隐动作只学到了外观差异而非真实的运动动力学。
2. **表征容量受限**：离散量化把连续动作强行压进有限码本，与动作本身连续的物理本质相矛盾。
3. **缺乏物理结构**：离散 token 没有尺度、没有可组合性,两个隐动作之间无法定义"先后叠加"这种在真实动作序列里天然存在的复合关系。

RotVLA 的核心洞察是：如果把隐动作表示为 SO(n) 旋转群的元素,由于 SO(n) 对矩阵乘法封闭，两个隐动作的组合可以直接定义为矩阵乘法——这天然对应真实动作的时序可组合性，同时旋转矩阵本身是连续流形上的对象,避免了离散量化的容量瓶颈。

## 二、核心方法

RotVLA 分三阶段构建：Stage I 学习连续旋转隐动作空间；Stage II 用隐动作监督预训练 VLM + flow-matching 动作专家；Stage III 微调为联合去噪隐动作与机器人动作的统一动作头。

### 2.1 Stage I：连续旋转隐动作

标准 LAM 用逆动力学编码器 $\mathcal{E}$ 和前向动力学解码器 $\mathcal{D}$，对间隔为 $k$ 的连续两帧 $I_t, I_{t+1}$ 提取隐动作：

$$z_{t\to t+1} = VQ\big(\mathcal{E}(I_t, I_{t+1})\big)$$

再由解码器重建下一帧 $\hat{I}_{t+1} = \mathcal{D}(I_t, z_{t\to t+1})$,用重建误差 $\mathcal{L}_{\text{LAM}} = \lVert \hat{I}_{t+1} - I_{t+1} \rVert_2$ 监督。

**连续化**：RotVLA 首先用 SoftVQ（对码本做 softmax 软分类分布聚合，而非硬量化）替换 VQ-VAE，在保留码本结构的同时保持隐空间连续。

**旋转化**：进一步约束隐动作落在 SO(n) 上。直接强制预测矩阵满足正交约束会过度限制表达能力，因此模型先预测一个无约束矩阵 $M$，再通过 SVD 做正交投影得到最近的旋转矩阵：

$$\mathrm{Proj}(M) = U\,\mathrm{diag}\big(1,1,\dots,\det(UV^\top)\big)\,V^\top$$

其中 $M = U\Sigma V^\top$ 是 SVD 分解。这一投影保证结果同时满足正交性和行列式为 1（即真正的旋转而非镜像反射），同时训练时保留了预测的自由度。

**用大白话说**：先让网络随便预测一个矩阵，再把它"掰正"成一个合法的旋转矩阵——这样既不会因为强行加正交约束而束缚网络的表达能力，又能保证最终得到的隐动作确实活在 SO(n) 这个有物理意义的空间里。

综合起来，隐动作提取算子定义为：

$$\mathcal{F}(I_t, I_{t+1}) = \mathrm{Proj}\Big(VQ_{\text{soft}}\big(\mathcal{E}(I_t, I_{t+1})\big)\Big)$$

**triplet 学习框架**：为进一步防止退化，RotVLA 采样三个连续帧 $I_t, I_{t+1}, I_{t+2}$，分别提取两段隐动作

$$z_{t\to t+1} = \mathcal{F}(I_t, I_{t+1}), \qquad z_{t+1\to t+2} = \mathcal{F}(I_{t+1}, I_{t+2})$$

单步重建损失照常：$\hat{I}_{t+1} = \mathcal{D}(I_t, z_{t\to t+1})$，$\hat{I}_{t+2} = \mathcal{D}(I_{t+1}, z_{t+1\to t+2})$，$\mathcal{L}_{\text{single}} = \lVert \hat{I}_{t+1}-I_{t+1}\rVert_2^2 + \lVert \hat{I}_{t+2}-I_{t+2}\rVert_2^2$。

由于隐空间存在规范（gauge）不确定性——恒等变换未必对应 SO(n) 的单位元——作者用"相同帧对"的批均值隐动作投影到 SO(n) 上，显式锚定单位元 $z_{\mathcal{I}} = \mathrm{Proj}\big(\mathbb{E}[\mathcal{F}(I_t,I_t)]\big)$。因为 SO(n) 对矩阵乘法封闭，两步隐动作可直接组合：

$$z^{\text{comp}}_{t\to t+2} = z_{t+1\to t+2}\cdot z_{\mathcal{I}}^{-1}\cdot z_{t\to t+1}$$

再用该组合隐动作直接从第一帧生成两步预测 $\hat{I}^{\text{comp}}_{t+2} = \mathcal{D}(I_t, z^{\text{comp}}_{t\to t+2})$，并对其施加重建约束 $\mathcal{L}_{\text{comp}} = \lVert \hat{I}^{\text{comp}}_{t+2}-I_{t+2}\rVert_2^2$。总训练目标为

$$\mathcal{L}_{\text{triplet}} = \mathcal{L}_{\text{single}} + \mathcal{L}_{\text{comp}} + \mathcal{L}_{\text{soft}}$$

其中 $\mathcal{L}_{\text{soft}}$ 是 SoftVQ 码本学习的 KL 损失。

**用大白话说**：只做"编码-解码-重建下一帧"很容易让隐动作偷懒——直接把目标帧的样子编码进去就能重建成功，根本不用理解真实的运动是什么。triplet 框架多加了一条约束：把 $t\to t+1$ 和 $t+1\to t+2$ 两段隐动作用矩阵乘法"串"起来，这个串起来的复合隐动作必须也能从第一帧直接跳两步、正确重建出第三帧。如果隐动作只是在"抄"目标帧的外观而不是真的学到了运动本身，这个组合约束就会失败——因为组合出来的隐动作会被套用到一张它没见过运动上下文的中间帧（$I_{t+1}$ 已经变了，但组合动作是从 $I_t$ 出发算的），只有真正编码了运动语义的隐动作才能让这条链路自洽。

### 2.2 Stage II：RotVLA 预训练

将预训练 VLM（InternVL3.5-1B）与基于 DiT 的 flow-matching 动作专家结合。给定连续帧对 $(I_t, I_{t+1})$ 和语言指令，VLM 输出视觉语言特征 $h$；同时用 Stage I 冻结的 LAM 提取对应的隐动作 token $z_{t\to t+1}\in\mathbb{R}^{n\times n}$（$n=16$，作为一个 $n$ 维动作 chunk）。动作专家用 flow matching 目标预测速度场：

$$\mathcal{L}^{\text{FM}}_{\text{LA}} = \mathbb{E}_{\tau, z_{t\to t+1}, z_0}\Big[\lVert v_\theta(z_\tau,\tau,h) - (z_{t\to t+1}-z_0)\rVert_2^2\Big]$$

其中 $z_\tau=\tau z_{t\to t+1}+(1-\tau)z_0$，$z_0\sim\mathcal{N}(0,I)$，$\tau\in[0,1]$。

### 2.3 Stage III：RotVLA 微调

下游微调阶段引入统一 flow-matching 动作头，联合预测：(1) 隐动作 $z\in SO(n)$，作为高层"规划器"；(2) $d$ 维机器人动作 chunk（视界 $N$）$a\in\mathbb{R}^{N\times d}$，作为底层"控制器"。联合损失：

$$\mathcal{L}^{\text{FM}}_{\text{LA-RA}} = \mathbb{E}_{\tau,x,x_0}\big[\lVert v_\theta(x_\tau,\tau,h)-(x-x_0)\rVert_2^2\big]$$

其中 $x=(a, z_{t\to t+1})$。**结构化注意力**是这一阶段的关键设计：隐动作 token 只关注视觉语言 token（不能看真实动作 token），而机器人动作 token 可同时关注隐动作 token 和视觉语言 token。这一非对称结构强制"规划"与"控制"分离，同时保留跨模态交互，使得微调时仍能保留预训练学到的隐动作语义,并让底层控制以隐动作规划为条件。

**用大白话说**：隐动作在这里被当成一个"先想清楚要往哪个方向动"的中间规划信号，机器人动作则是"照着这个规划落实到具体关节/末端位移"的执行信号。通过注意力遮罩强制隐动作看不到执行结果，可以避免规划信号被执行细节"抄近道"污染。

### 2.4 模型规模与数据

RotVLA 总参数约 1.7B：视觉编码器 304M（冻结 DINOv2）+ 语言模型 752M（InternVL3.5-1B）+ 动作头 305M（24 层 DiT）+ 隐动作模型 290M（Stage I 训练后冻结）。预训练数据聚合 Open X-Embodiment、AGiBot-beta、RoboMIND、RoboCOIN、Ego4D，总计超过 1700 小时,涵盖单臂/双臂/灵巧手机器人与人类第一视角视频（按数据集占比：OXE 31.43%、RoboMIND 29.06%、RoboCOIN 24.36%、AGiBOT 10.78%、Ego4D 4.37%；按具身类型：单臂 47.44%、双臂 31.01%、灵巧手 17.18%、人类视频 4.37%）。训练在 8×NVIDIA H200 上耗时 50 小时。

## 三、实验结果

**LIBERO 与 RoboTwin2.0**（Table 1，成功率 %）：

| 方法 | 参数量 | LIBERO Spatial | Object | Goal | Long | Avg | RoboTwin2.0 Clean | Rand |
|---|---|---|---|---|---|---|---|---|
| GO-1† | 2B | 96.2 | 97.8 | 96.0 | 89.2 | 94.8 | 37.8 | 36.2 |
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | – | – |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | – | – |
| $\pi_0$ | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | – | – |
| $\pi_{0.5}$ | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | – | – |
| GR00T-N1.6† | 3B | 97.7 | 98.7 | 97.5 | 94.4 | 97.0 | – | – |
| UniVLA* | 9B | 95.4 | 98.8 | 93.6 | 94.0 | 95.4 | – | – |
| X-VLA | 0.9B | 98.2 | 98.6 | 97.6 | **97.8** | 98.1 | 72.8 | 72.8 |
| StarVLA | 4B | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 | 88.2 | 88.3 |
| Motus† | 8B | – | – | – | **97.8** | – | 88.7 | 87 |
| LingBot-VLA | 4B | – | – | – | – | – | 88.6 | 86.7 |
| **RotVLA** | **1.7B** | **98.2** | 99.6 | **98.4** | 96.4 | **98.2** | **89.6** | **88.5** |

RotVLA 以最小参数量（1.7B）取得 LIBERO 均值最高（98.2%）和 RoboTwin2.0 clean/rand 最高（89.6%/88.5%）,超过参数量数倍于自己的 UniVLA*（9B）、Motus†（8B）等基线。

**真实世界实验**（ARX R5 双臂平台，与 $\pi_{0.5}$ 对比,成功率 %）：

| 任务 | $\pi_{0.5}$ Clean | $\pi_{0.5}$ Rand | RotVLA Clean | RotVLA Rand |
|---|---|---|---|---|
| Pick and Place | 93.3 | 73.3 | 93.3 | 90.0 |
| Put and Close（开抽屉） | 86.7 | 66.7 | 96.7 | 90.0 |
| Stack Three Cups（双臂叠杯） | 56.7 | 33.3 | 66.7 | 60.0 |

单臂任务上 RotVLA 均达到 90% 以上成功率；双臂叠杯这一需要长时序双臂协同规划的任务上，RotVLA 失败率显著低于基线。在背景变化和干扰物引入的域偏移设置下，RotVLA 相对基线的优势进一步拉大，作者将此归因于连续旋转隐动作预训练捕捉了高层运动语义而非背景外观。推理效率上，RotVLA 在 NVIDIA H20 上单步延迟 79ms，略高于 $\pi_{0.5}$ 的 61ms，仍属实时可用范围。

**隐动作退化检验**（Table 2）：对比"仅重建"基线（UniVLA 式）与 triplet 训练,分别测量重建误差 $\mathrm{MSE}=\mathrm{MSE}(\hat I_{t+1}, I_{t+1})$ 与"想象误差" $\mathrm{MSE}'=\mathrm{MSE}(\hat I'_{t+2}, I_{t+1})$（即把 $t\to t+1$ 学到的隐动作重新套用到 $I_{t+1}$ 上,想象出的 $\hat I'_{t+2}$ 与真实 $I_{t+1}$ 应有明显差异,若差异过小则说明隐动作没有编码真实运动）：

| LAM 训练方式 | MSE↓ | MSE′ | $\Delta=$ MSE′−MSE ↑ |
|---|---|---|---|
| 仅重建（UniVLA 式） | 0.0029 | 0.0066 | 0.0037 |
| Triplet（RotVLA） | 0.0030 | 0.0078 | **0.0048** |

两者单步重建误差相近,但 triplet 训练的想象误差显著更大（$\Delta$ 提升约 30%）,说明其隐动作确实编码了可外推的运动动力学，而非单纯记忆目标帧外观；Figure 5 的定性结果显示 triplet-LAM 能正确外推出"手臂继续抬起"的想象帧，仅重建基线则生成几乎与 $\hat I_{t+1}$ 相同的图像。

**LARY 基准上的隐动作表征质量**（Table 3，回归 MSE 越低越好、分类准确率越高越好）：

| 模型 | CALVIN↓ | VLABench↓ | RoboCOIN↓ | AgiBot↓ | Avg↓ | 分类 Robot↑ | Human↑ | Avg↑ |
|---|---|---|---|---|---|---|---|---|
| LAPA | 0.96 | 0.95 | 0.96 | 1.00 | 0.97 | 23.64 | 14.61 | 19.13 |
| UniVLA | 0.82 | 0.74 | 0.94 | 0.97 | 0.87 | 18.56 | 19.08 | 18.82 |
| villa-X | 0.86 | 0.72 | 0.94 | 0.97 | 0.87 | 29.90 | 17.80 | 23.85 |
| LAPA-DINOv3 | 0.50 | 0.25 | 0.82 | 0.84 | 0.63 | 27.04 | 64.19 | 45.62 |
| RotVLA*（排除重叠数据） | 0.38 | 0.11 | 0.47 | 0.10 | 0.27 | 61.26 | 58.13 | 59.70 |
| **RotVLA** | **0.30** | **0.08** | **0.35** | **0.07** | **0.20** | **67.62** | **74.33** | **70.98** |

RotVLA 在回归误差和分类准确率上全面超越现有隐动作方法，即使排除与 LARY 基准重叠的训练数据（RotVLA*）仍显著优于所有基线，说明其代表的是更本质的低层控制动力学与高层语义结构，而非对基准数据的记忆。

**消融**（Table 4，LIBERO 四套件）：

| 配置 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| w/o Pretrain | 94.8 | 97.8 | 94.2 | 89.8 | 94.2 |
| w/o Planner（微调时去掉隐动作监督） | 96.0 | 99.6 | 97.2 | 93.2 | 96.5 |
| $n=8$ | 97.2 | 99.4 | 98.2 | 94.4 | 97.3 |
| $n=32$ | 98.2 | 99.6 | 97.4 | 94.0 | 97.3 |
| Discrete（离散量化基线） | 95.4 | 99.4 | 96.6 | 89.6 | 95.3 |
| Cont.（仅连续化，无 SO(n)） | 96.6 | 99.4 | 97.2 | 93.6 | 96.7 |
| Cont. SO(n)（无 triplet 组合监督） | 97.0 | 99.2 | 97.6 | 94.2 | 97.0 |
| **RotVLA**（$n=16$ + 全部组件） | 98.2 | 99.6 | 98.4 | 96.4 | **98.2** |

拆解显示：离散→连续化带来 +1.4%；单纯加 SO(n) 约束但无组合监督增益边际（结构约束本身不够）；再加入 triplet 组合监督额外 +1.2%，说明性能提升主要来自"连续化 + 组合一致性监督"的组合，而非单一设计。隐动作维度 $n=16$ 为最优，过小限制表达力,过大增加优化难度和计算成本。去掉微调阶段的隐动作规划监督（w/o Planner）导致长时序任务（LIBERO-Long）性能明显下降（96.4→93.2）,验证了隐动作作为高层规划信号的作用。

**数据规模消融**（Figure 9）：仅用 40% 机器人数据预训练相比不预训练,LIBERO 提升约 3%、RoboTwin2.0 提升超 7%；扩展到全部机器人数据集（OXE+AGiBOT+RoboMIND+RoboCOIN）及加入 Ego4D 人类视频后进一步提升,人类视频对 RoboTwin2.0 的增益尤为明显（Clean 86.71→88.84，Rand 85.82→87.54），显示出较强的数据可扩展性。

## 四、局限性

论文附录 A 明确列出两点局限：

- **参数规模仍偏保守**：RotVLA 以 1.7B 总参数取得 SOTA,但隐动作模型本身仅 290M,规模相对较小；作者认为同时扩大 LAM 与 VLA 规模可能带来进一步提升,但论文未做相应的扩展实验验证这一猜想。
- **人类视频利用仍浅层**：Ego4D 目前只作为补充训练数据的一小部分（4.37%）参与联合预训练,论文未探索仅用人类视频进行隐动作预训练、或更大规模引入人类视频的效果,这是作者明确标注的未来方向。

此外，从实验设计角度看：真实世界评测仅在单一 ARX R5 平台、三个任务（其中仅一个双臂任务）上进行,任务难度和长时序程度相对有限,是否能扩展到更长时序、更精细接触的操作任务尚未验证；旋转隐动作的可组合性验证也仅停留在两步组合（triplet），未测试更长链条组合是否仍保持一致性。

## 五、评价与展望

**贡献与优点**：RotVLA 抓住了当前隐动作模型的一个真实痛点——离散量化与连续物理动作之间的表征失配——并给出了一个几何上自洽的解法：把隐动作嵌入 SO(n) 群，利用群的封闭性天然定义动作组合，再用三帧 triplet 目标把"组合一致性"转化为可监督的训练信号。这比单纯换一种连续化手段（如仅替换 VQ-VAE 为连续 VAE）更进一步,因为它显式利用了动作序列本身具有的代数结构（时序可叠加性），而不只是让隐空间连续。Table 2 的 MSE/MSE′ 差异分析和 Table 3 在 LARY 基准上的线性探测结果，从两个独立角度（生成式外推能力、判别式表征质量）验证了 triplet 组合监督确实缓解了退化问题，证据链条比较完整。以 1.7B 参数超过 GR00T-N1.6、UniVLA*（9B）等更大模型，也说明隐动作表征质量本身可能比单纯堆参数量更能决定跨具身泛化能力。

**与其他公开工作的关系**：RotVLA 延续的是 UniVLA、LAPA、GO-1 等一系列基于 encode-decode 范式的隐动作模型脉络,其直接对照基线包括 UniVLA（离散 latent action，本文 Table 3 中作为基线之一）、villa-X（增强隐动作建模）、Motus（统一隐动作世界模型）以及同期的 CoMo（从互联网视频学习连续隐动作）等——但这些工作大多止步于"离散换连续",较少显式建模隐动作之间的代数组合关系。RotVLA 与 X-VLA（软提示 Transformer，本文最强的同量级基线，0.9B 参数在 LIBERO 上达 98.1%）的对比也值得关注：X-VLA 不依赖隐动作规划而是通过软提示机制实现跨具身适配,取得了与 RotVLA 相近的 LIBERO 分数但 RoboTwin2.0 分数明显更低（72.8% vs 89.6%），提示隐动作规划信号在双臂协同这类更复杂操作场景中可能提供了软提示机制未能捕捉的结构化先验。此外,论文使用的 LARY 基准（作者附录引用为同期工作,arXiv 2604.11689）本身是一个新提出的、用于评测隐动作表征质量的线性探测基准,RotVLA 是较早在该基准上系统对比多种隐动作方法的工作之一,为后续隐动作表征评测提供了一个可复现的参照点。

**开放问题与可能的改进方向**：(1) SO(n) 组合律仅在两步 triplet 上验证,更长链条（$k>2$ 步）组合是否仍保持一致性、以及是否可以设计更长时序的组合监督目标以进一步强化时序动力学建模，是自然的后续方向；(2) 论文将隐动作维度固定为一个统一的 $n\times n$ 旋转矩阵，未探索是否可以像真实机器人关节空间那样引入分块 SO(3) 乘积结构（例如对应不同关节自由度的多个小旋转块），这可能进一步提升物理可解释性和跨具身泛化的精细度；(3) 附录中作者自陈的"LAM 与 VLA 联合扩大规模"和"更深度利用人类视频"两个方向都尚未验证,鉴于 Ego4D 仅占 4.37% 数据量却已带来可观的 RoboTwin2.0 提升（Figure 9），大幅提高人类视频占比是否能进一步缩小人类演示与机器人演示之间的动力学建模差距是一个值得关注的开放问题；(4) 79ms 单步延迟相较 $\pi_{0.5}$ 的 61ms 仍有约 30% 的额外开销,这部分来自双重去噪（隐动作+机器人动作联合 flow matching），如何在保留规划-控制解耦收益的同时进一步压缩推理延迟，是走向实时部署前需要解决的工程问题。

## 参考

- [7] Qingwen Bu, Yanting Yang, Jisong Cai, et al. *UniVLA: Learning to Act Anywhere with Task-Centric Latent Actions.* arXiv:2505.06111, 2025.（本文遵循的 LAM 编码器架构基础，也是 Table 1/3 的对照基线）
- [6] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, et al. *LAPA: Latent Action Pretraining from Videos.* arXiv:2410.11758, 2024.（LARY 基准上的主要对照基线之一）
- [11] Hao Chen, Ze Wang, Xiang Li, et al. *SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer.* CVPR 2025.（本文用于替换 VQ-VAE 的连续量化机制）
- [26] Jinliang Zheng, Jianxiong Li, Zhihao Wang, et al. *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model.* arXiv:2510.10274, 2025.（Table 1 中最强同量级基线之一）
- [51] Dujun Nie, Fengjiao Chen, Qi Lv, et al. *LARY: A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment.* arXiv:2604.11689, 2026.（本文用于评测隐动作表征质量的基准）
