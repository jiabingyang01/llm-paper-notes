# LaWAM：面向高效动力学感知机器人策略的潜在世界动作模型

> **论文**：*LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies*
>
> **作者**：Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, Chao Yu 等（通讯作者：Yuanbo Xu, Chao Yu）
>
> **机构**：Tsinghua University、Jilin University、Nankai University、Peking University、Harbin Institute of Technology、Zhongguancun Academy、Striding.AI、Infigence AI
>
> **发布时间**：2026 年 06 月（arXiv 2606.15768）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.15768) | [PDF](https://arxiv.org/pdf/2606.15768)
>
> **分类标签**：`World Action Model` `潜在动作模型` `VLA` `跨具身泛化` `非迭代未来预测` `flow matching`

---

## 一句话总结

LaWAM 把"世界动作模型（WAM）必须生成像素级未来图像/视频"这一惯性假设推翻：它把 latent action model（LAM）的前向解码器直接重用为一个只在冻结 DINOv3 特征空间里做单次非迭代前向的**潜在世界模型（LaWM，230M 参数)**,用它预测的潜在视觉子目标去条件化动作专家,在 LIBERO 上取得 98.6% 的平均成功率、RoboTwin 上取得 91.22% 的综合成功率(均为对比方法中最优或并列最优),同时把世界建模参数量比同类像素空间 WAM 常用的 5B WAN 骨干减少约 95%,推理延迟降到 187ms/动作块,相比最强像素空间基线(LingBot-VA,4482ms)最高快 24 倍。

## 二、问题与动机

VLA 模型凭借大规模视觉-语言预训练获得了很强的语义 grounding 能力,但通常直接从当前观测预测动作,缺乏对"动作会如何改变场景"的显式前瞻。World-Action Models(WAMs)试图通过让策略条件在预测未来(图像/视频/状态)之上来补上这一环,但现有 WAM 存在三个问题：(1)多数方法预测未来图像或视频,把大量建模容量花在像素级合成而非动作相关的紧凑动态上；(2)迭代式未来生成带来显著推理延迟——论文在统一评测设置下测得 LingBot-VA 单次策略推理需要 4482ms,而代表性 VLA π0.5 只需 220ms；(3)对操作任务而言,有效的未来预测应该暴露"下一个动作块所需的场景状态变化",而不只是生成视觉上合理、任务一致的画面。

论文进一步指出,latent action models(LAM,如 LAPA、CoMo 等)已经证明可以从无标注视频里学到嵌入-无关的紧凑潜在动作表示,但已有工作大多只把 LAM 的**编码器**(推断潜在动作)当作跨具身动作表示来用,训练完就丢弃**解码器**。LaWAM 的出发点(与并行工作 Garrido et al. 的观察一致)是：LAM 的解码器本身就已经隐式实现了一个"潜在动作条件化的世界模型"——把它保留下来并系统性地重新定位为策略可用的动态学接口,就能以远低于像素空间生成的代价获得动力学感知能力。

## 三、核心方法

### 问题形式化

标准 VLA 直接建模 $p(a_{1:T} \mid o, l)$。WAM 把它按水平观测 $o_T$ 分解为未来预测与逆动力学模型(IDM)两部分:

$$\underbrace{p(a_{1:T}, o_T \mid o, l)}_{\text{Joint}} = \underbrace{p(o_T \mid o, l)}_{\text{Future Prediction}}\ \underbrace{p(a_{1:T} \mid o, o_T)}_{\text{IDM}}. $$

大白话：先想清楚"未来会变成什么样"(未来预测),再想"要达到这个未来需要做什么动作"(IDM)。像素空间 WAM 在第一项上要生成完整密集图像/视频,而实际动作块控制往往只需要一个紧凑的场景变化描述。

### 阶段一：学习 LaWM(潜在世界模型)

LaWAM 把未来表示为**冻结视觉编码器特征空间**里的量,而非像素。记 $f_\psi$ 为冻结编码器,$u=f_\psi(o)$,$u_T=f_\psi(o_T)$。先在特征对上学习一个潜在动作模型:

$$z \sim q_\phi(z \mid u, u_T), \qquad \tilde{u}_T = \text{LaWM}_\omega(u, z).$$

$q_\phi$ 是潜在动作后验(相当于一个 IDM),从观测到的转移 $(u,u_T)$ 推断潜在动作 $z$；解码器则用当前特征和 $z$ 预测水平特征。训练完成后,LaWAM 保留这个**解码器**作为 LaWM,用作后续预测子目标的核心模块;编码器只在训练阶段用来产生教师潜在动作。

阶段一损失为:

$$\mathcal{L}_{\text{LAM}} = \mathcal{L}_{\text{wm}} + \mathcal{L}_{\text{aux}} + \beta D_{\text{KL}}\big(q_\phi(z\mid u_T) \,\|\, \mathcal{N}(0,I)\big),$$

其中 $\mathcal{L}_{\text{wm}}=\|\tilde{u}_T-u_T\|_2^2$ 训练 LaWM 匹配水平特征,$\mathcal{L}_{\text{aux}}=\|g(s,z)-s_T\|_2^2$ 是一个轻量末端状态预测头(训练后即丢弃),鼓励 $z$ 编码具身运动而非仅视觉外观,KL 项正则化潜在动作空间使其后续可被策略先验建模。大白话：先教会一个"给定当前状态+一个抽象动作向量,预测未来会变成什么特征"的小模型,附带一个辅助任务防止 $z$ 只学到"画面看起来变了"而没学到"末端执行器真的动了"。

### 阶段二：LaWAM(潜在世界动作模型)

部署时未来特征 $u_T$ 不可得,阶段一的 IDM 编码器无法使用。因此训练一个策略先验 $p_\theta(\hat z \mid o, l)$ 从当前观测和指令直接预测潜在动作,再送入冻结的 LaWM 解码器得到潜在视觉子目标 $\hat u_T=\text{LaWM}_\omega(u,\hat z)$。整个动作生成过程分解为:

$$\underbrace{p(a_{1:T},\hat u_T,\hat z \mid o,l)}_{\text{LaWAM}} = \underbrace{p_\theta(\hat z\mid o,l)}_{\text{Policy Prior}}\ \underbrace{p_\omega(\hat u_T\mid u,\hat z)}_{\text{LaWM}}\ \underbrace{p_\eta(a_{1:T}\mid o,l,u,\hat u_T)}_{\text{Action Expert}}. $$

阶段二总损失:

$$\mathcal{L}_{\text{LaWAM}} = \lambda_{\text{distill}}\mathcal{L}_{\text{distill}} + \lambda_{\text{wm}}\mathcal{L}_{\text{wm}} + \mathcal{L}_{\text{act}},$$

其中 $\mathcal{L}_{\text{distill}}=\mathbb{E}[\|\hat z-z\|_2^2]$ 把策略先验蒸馏向阶段一学到的真实潜在动作,$\mathcal{L}_{\text{wm}}=\|\hat u_T-u_T\|_2^2$ 监督策略驱动的子目标,$\mathcal{L}_{\text{act}}$ 是标准的条件 flow-matching 速度场回归损失,用于生成动作块 $a_{1:T}$。训练中还使用 Knowledge Insulation(KI)技术,阻止动作专家的梯度反向"污染"已经预训练好的 LaWM 动力学参数。

### 架构细节

- **LaWM**：基于蒸馏版 DINOv3 ViT-B/16 特征,编码器与解码器都是 24 层 Transformer,采用类 V-JEPA2 的时空联合 token 设计。解码器通过 **自适应层归一化(AdaLN)** 而非加性 token 注入来注入潜在动作 $z$——论文发现加性注入在跨具身设定下会因潜在动作范数波动引起全局 token 偏移和损失突刺,AdaLN 更稳定。
- **LaWAM 策略骨干**：沿用 Qwen-GR00T 架构,取 Qwen3-VL 前 16 层作为 VLM 骨干,动作专家由 4 个 Alternate-DiT block(共 16 层)组成,隐藏维度 1024。动作专家通过 Alternate-DiT 交替关注完整 VLM 隐状态流与由 $(u,\hat u_T)$ 构成的动力学流,而非像常规 VLA 那样只交替视觉/语言流。
- **物理时间对齐**：跨数据集/具身的控制频率不同,同一 token 下标不代表相同的物理时间。设分支 $b$ 原生控制频率为 $h_b$,固定物理间隔 $\tau$,离散动作horizon 为

$$H_b = \text{round}(\tau h_b),$$

再对每个动作 token 加正弦物理时间编码

$$\phi(t_{b,i}) = \text{Concat}\big[\sin(t_{b,i}\omega_k),\cos(t_{b,i}\omega_k)\big]_{k=0}^{K-1}, \quad t_{b,i}=i/h_b.$$

大白话：不同数据集帧率不同,第 5 个动作 token 在 20Hz 数据里代表 0.25 秒后,在 5Hz 数据里却代表 1 秒后——直接按 token 下标编码位置会让模型混淆"多久之后"。物理时间编码把 token 下标换算成真实秒数再编码,让所有分支在同一个物理时间坐标系里对齐。论文用受控的 LIBERO 5/10/20Hz 混频实验(Fig.7)验证了该编码能把混频训练性能拉回到接近单一 20Hz 训练的上界。
- **动作与观测约定**：所有动作标签统一转换为末端执行器(EEF)表示；训练/评测均不提供本体状态输入(避免过拟合到轨迹特定的状态痕迹);仅用 RGB 观测,分辨率统一到 256×256。
- **参数量与推理**：LaWM 仅 230M 参数,LaWAM 总参数 2.3B,比常见像素空间 WAM 所用的 5B WAN 视频生成骨干减少约 95% 的世界建模参数;推理时用 10 步 Euler 法反向积分 flow ODE,在 A100 上测得 187ms/动作块延迟。

## 四、实验结果

**预训练数据**：约 3000 小时机器人视频 + 1500 小时第一人称人类视频(来自 EgoDex、Ego4D、Lego、AgiBot World Colosseo、RoboMIND、Robocoin、Open X-Embodiment、DROID 等公开数据集);人类视频仅通过 LaWM 动力学先验发挥作用,不参与策略整合训练(缺少任务级语言标注)。LaWM 阶段一:16×H100,100k 步,AdamW,lr 3e-4,batch 1024,KL 权重 β=1e-5。策略整合阶段二:64×H100,200k 步,batch 1024。

**LIBERO(Table 1,四个套件共 40 个任务,2000 次试验,每任务 50 trial)**：

| 类别 | 方法 | 模型规模 | 延迟(ms) | Long | Goal | Object | Spatial | Average |
|---|---|---|---|---|---|---|---|---|
| VLA | OpenVLA-OFT | 7B | — | 94.5 | 97.9 | 98.4 | 97.6 | 97.1 |
| VLA | π0.5 | 3.5B | 220 | 92.4 | 98.0 | 98.2 | 98.8 | 97.0 |
| VLA | GR00T-N1.6 | 3.3B | 259 | 94.4 | 98.5 | 97.7 | 97.7 | 96.9 |
| 潜在动作 | VLA-JEPA | 3B | — | 95.8 | 97.2 | 99.6 | 96.2 | 97.2 |
| 像素WAM | Cosmos-Policy | 2.1B | 1413 | 97.6 | 98.2 | 100.0 | 98.1 | 98.5 |
| 像素WAM | LingBot-VA | 5.5B | 4482 | 98.5 | 97.2 | 99.6 | 98.8 | 98.5 |
| 像素WAM | Fast-WAM | 6B | 486 | 95.2 | 97.0 | 100.0 | 98.2 | 97.6 |
| **本文** | **LaWAM** | **2.3B** | **187** | 97.0 | 98.4 | 99.6 | **99.4** | **98.6** |

LaWAM 以最低延迟拿到最高平均成功率,且不需要 5B/6B 级视频生成骨干。

**RoboTwin 2.0(Table 2,50 项双臂任务,每设置 100 trial,Fast-WAM/LingBot-VA 在 H100 上重新评测,其余数字取自原论文)**：

| 方法 | Clean | Randomized |
|---|---|---|
| Fast-WAM | 91.98 | 90.52 |
| GigaWorld-Policy | 86.36 | 85.04 |
| LingBot-VA | 91.50 | 90.92 |
| π0.5 | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| **LaWAM** | **92.64** | 89.80 |

LaWAM 取得最佳 Clean 场景平均成功率,Randomized 场景下与最强像素空间 WAM 差距很小,综合(Clean+Rand 均值)91.22% 与摘要一致。

**真机评测(Table 3,Franka Emika Panda 单臂 + Quanta X1 双臂,30 trial/任务)**：

| 方法 | Pick-and-Place | Open Drawer | Fold Towel | Average |
|---|---|---|---|---|
| π0.5 | 86.7 | 80.0 | 83.3 | 83.3 |
| GR00T-N1.6 | 83.3 | 76.7 | 46.7 | 68.9 |
| Fast-WAM | 56.7 | 63.3 | 70.0 | 63.3 |
| LingBot-VA | 76.7 | 83.3 | 0.0 | 53.3 |
| **LaWAM** | **93.3** | 86.7 | **90.0** | **90.0** |

在长时序、动态形变的叠毛巾任务上优势尤其明显:高延迟基线(如 LingBot-VA,单次推理 4482ms)在生成下一步动作期间毛巾持续运动,导致动作与陈旧视觉状态失配、成功率跌到 0。

**消融(Fig.6, LIBERO)**：按 LaWAM → w/o pretrain → w/o distill → w/o KI & distill → w/o WM(完全去掉子目标接口)逐级削弱,性能单调下降;去掉整个 LaWM 接口造成最大降幅(尤以 LIBERO-Long 最明显),说明潜在子目标条件化是主要增益来源；去掉潜在动作蒸馏同样显著掉点,说明策略先验需要来自 LAM 后验的直接监督才能可靠驱动 LaWM。

**混频编码实验(Fig.7, 受控 LIBERO 5/10/20Hz 联合训练)**：不加物理时间编码时联合训练相对纯 20Hz 上界明显掉点,加入编码后基本恢复到上界附近,验证了物理时间对齐的必要性。

**动力学保真度(Fig.10, 500 条 LIBERO 轨迹开环 rollout)**：rollout 特征与真实未来特征的余弦相似度全程维持高位且随步数缓慢下降,同时明显偏离初始观测特征,说明 LaWM 学到的是非平凡的动力学演化而非简单复制当前观测。跨环境/跨具身开环 rollout(Fig.5/14/15)显示同一潜在动作在不同视觉场景/具身下产生连贯但场景特定的潜在变化,支持"潜在动作编码具身无关的转移,LaWM 负责在当前具身/场景中落地"这一解释。

## 五、局限性

论文第 5 节明确指出两点局限：(1)LaWM 目前在相机视角相对稳定的操作场景中最有效;当观测中相机自身运动占主导(如第一人称视频中的剧烈晃动或大幅视角变化)时,LaWM 可能学不到连贯的潜在动作空间,这限制了当前方法向观测强烈受自身运动影响的人形/移动机器人平台的推广。(2)数据覆盖不足:精细形变物体动力学(如叠毛巾任务中的微妙布料形变)在当前训练数据混合中较为稀少,使 LaWM 难以可靠建模——真机实验也印证了这一点,作者承认 LaWAM 在形变物体操作上仍有局限,当前 LaWM 特征分辨率尚不足以精确捕捉细粒度布料形变。作者将扩大数据规模与模型容量列为未来工作方向。

## 六、评价与展望

**优点**：(1)方法论干净、目标明确——把"世界模型是否需要生成像素"这个近期 WAM 领域反复被质疑的问题(参见并行工作 Fast-WAM《Do World Action Models need test-time future imagination?》),推向一个更彻底的答案:直接抛弃像素/视频重建,只在冻结视觉编码器的特征空间里做单次非迭代前向。相比 Fast-WAM 等"减少生成步数"的效率化路线,LaWAM 是"完全不生成像素"的路线,在 LIBERO/RoboTwin 双基准上仍取得最优或近最优精度,同时把世界建模参数压缩约 95%、延迟降到 187ms,latency-accuracy 权衡图(Fig.1)上的位置颇具说服力。(2)对 latent action model 解码器角色的重新定位有明确问题意识:多数在先前 LAM 工作(LAPA、CoMo、UniVLA 等)把解码器视为训练阶段的辅助监督信号、训练完即弃,LaWAM 与并行工作 Garrido et al. 一致地指出解码器本身已隐式实现潜在动作条件化世界模型,并把这一观察系统化为可部署的策略接口,是一个自然但此前未被充分利用的设计空间。(3)物理时间编码解决混频多源数据对齐这一实际工程问题,并用受控消融(固定任务分布/具身/相机,只改变控制频率映射)干净地分离出该机制的贡献,实验设计严谨。(4)消融链条完整(阶段一/阶段二/接口三个层面均有对照),开环 rollout 的定量(特征余弦相似度)与定性(跨具身/跨场景)证据相互印证,支撑了"LaWM 学到了具身无关动力学"这一核心论点。

**局限与开放问题**：(1)基线数字来源不完全统一——部分基线(Fast-WAM、LingBot-VA)在作者自己的 H100 环境下重新评测,另一部分(GigaWorld-Policy 等)直接取自原论文报告数字,这种混合口径虽是该领域常见做法,但严格意义上弱化了跨方法比较的可控性,一个完全统一环境下的重新评测会更有说服力。(2)论文未讨论 LIBERO/RoboTwin 测试任务与约 3000 小时异构预训练数据(AgiBot World、RoboMIND 等)之间是否存在场景/物体级别的重叠或泄漏,在当前"大规模异构预训练+下游基准微调"范式下,这类数据去重/泄漏排查(其他同期工作,如 Pelican-VLA 0.5,采用了 TF-IDF+人工复核做泄漏排除)本可以进一步加固结论的可信度。(3)阶段一 KL 正则权重 $\beta=10^{-5}$ 非常小,论文未给出潜在动作空间维度/信息量的定量分析,也未消融连续潜在动作(本文选择)与离散潜在动作(如 LAPA 使用的 VQ 编码)之间的优劣,这一设计选择的必要性缺少直接证据支撑。(4)作者自陈的相机运动敏感性限制了方法向人形/移动机器人的直接推广,而这恰是当前具身智能基础模型的重要目标场景之一;论文的真机验证仍停留在固定基座机械臂(Franka、Quanta X1),尚未提供任何移动/人形平台上的实证,该限制目前仍是纯理论/前瞻性讨论。(5)与同期同类"未来表示压缩"路线(如 F1、Cosmos-Policy 通过更小或更快的生成模型压缩未来预测代价)相比,LaWAM 的差异化在于把"未来"完全限定在冻结视觉编码器的语义特征空间而非任何形式的像素/视频空间,这一选择的代价——即预测特征是否会丢失动作规划所需的精细几何/接触信息——论文主要靠下游任务成功率间接验证,尚缺少专门针对"特征空间未来预测保真度上限"的独立分析,是一个值得后续工作深入的方向。

## 参考

1. Black et al. *π0.5: A Vision-Language-Action Model with Open-World Generalization*, 2025.
2. Bjorck et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025.
3. Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation*, 2025.
4. Yuan et al. *Fast-WAM: Do World Action Models Need Test-Time Future Imagination?*, 2026.
5. Garrido et al. *Learning Latent Action World Models in the Wild*, 2026.
