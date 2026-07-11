# VLAFlow：面向 VLA 训练范式受控比较的统一训练框架（Co-training 与未来隐变量对齐）

> **论文**：*VLAFlow: A Unified Training Framework for Vision-Language-Action Models via Co-training and Future Latent Alignment*
>
> **作者**：Guoyang Xia, Fengfa Li（共同一作）, Hongjin Ji, Lei Ren（项目负责人）, Fangxiang Feng（通讯作者）, Kun Zhan, Yan Xie
>
> **机构**：Li Auto Inc.；北京邮电大学人工智能学院；香港中文大学（深圳）
>
> **发布时间**：2026 年 07 月（arXiv 2607.01586）
>
> **发表状态**：未录用（预印本，文中自称 "report"）
>
> 🔗 [arXiv](https://arxiv.org/abs/2607.01586) | [PDF](https://arxiv.org/pdf/2607.01586)
>
> **分类标签**：`VLA训练范式` `flow matching` `VLM co-training` `未来隐变量对齐` `V-JEPA2` `负迁移`

---

## 一句话总结

在统一的 π0 式架构（Qwen3-VL-4B backbone + DiT flow-matching action expert）与约 5,000 小时异构机器人语料（OXEMix）之上,受控比较四种 VLA 预训练目标——纯动作建模（**MindPI**）、语言监督 co-training（**MindLPI**）、未来隐变量对齐（**MindWPI**）及二者结合（**MindLWPI**）——发现全参数纯动作预训练在跨 embodiment 迁移时会出现明显**负迁移**,而语言监督与未来隐变量监督分别从"高层动作意图"和"状态转移"两个互补角度提供中间约束,MindLWPI 综合表现最稳定（LIBERO 99.1、LIBERO-Plus 74.8、WidowX 75.5,同时 RT-1 上接近最优）。

## 一、问题与动机

VLA 预训练领域缺乏"控制变量"的公平比较：现有模型（如 LingBot-VLA、ABot-M0 等）架构、数据、动作空间、评测协议各不相同,难以判断性能差异究竟来自数据规模还是训练目标本身。作者指出已有证据表明 VLA 预训练并非简单地"数据越多越好"——当预训练数据分布与下游任务在 embodiment、动作空间、采样频率或任务语义上差异较大时,模型可能遭受负迁移。

论文将现有 VLA 训练范式归纳为三类并可组合扩展为第四类：
1. **纯动作建模**（π0 代表）：预训练与微调都只用动作 chunk 预测作为目标；
2. **VLM co-training**（π0.5、LAP、JoyAI-RA 0.1 代表）：额外让模型生成子目标 / 动作描述 / 离散动作 token,引入语言空间的高层动作意图监督；
3. **未来隐变量特征对齐**（Being-H0.7、LDA-1B 代表）：模型在预测动作的同时对齐 / 预测未来帧的隐变量表示,获得特征级的"未来想象"能力；
4. 本文提出的**组合范式**：语言动作描述与未来隐变量监督同时引入,让语言空间与视觉隐空间共同约束动作表示学习。

核心研究问题：**在近似相同的数据分布和模型架构下,不同的 VLA 训练目标会导致怎样不同的下游迁移行为？**

## 二、核心方法

### 2.1 受控比较协议与共享架构

VLAFlow 本身不是单一模型,而是一套统一评估框架,固定四个变量以排除混淆因素：
- **相同 VLM backbone + action expert**：VLM 为 Qwen3-VL-4B-Instruct（36 层, $d_{\text{vlm}}=2048$, GQA 8 个 KV head）,action expert 为 36 层 DiT（隐藏维度 1280）,通过逐层 KV-cache 共享（而非交叉注意力）复用 VLM 多模态上下文,而不是各范式独立设计架构；
- **相同 14 维动作空间**：每臂 7 维（末端平移增量、旋转增量、夹爪状态）,双臂拼接为 14 维,单臂数据零填充 + 有效性掩码接入统一空间；
- **相同数据与训练预算**：四种范式共享 OXEMix 语料（约 5,017.9 小时,153.6 万条轨迹）,由 DROID、OpenX-Embodiment（OXE）、OpenX-Augmented（OXE-AugE）、RoboCOIN 构成,其中 OXE-AugE 占时长 69.99%、OXE 占 27.20%、RoboCOIN 仅占 2.81%；除各自需要的辅助监督字段外,采样策略、训练步数、优化器、学习率完全一致；
- **相同下游评测协议**：所有预训练 checkpoint 用相同微调流程在 LIBERO、LIBERO-Plus、SimplerEnv 上评测,并训练不预训练的基线以量化正/负迁移。

Flow-matching 动作建模是四种范式共享的动作生成机制。设真实动作 chunk 为 $\mathbf{a}$、噪声 $\epsilon\sim\mathcal{N}(0,I)$、连续时间 $t\in[0,1]$,加噪动作为

$$
\mathbf{x}_t = (1-t)\epsilon + t\mathbf{a}. \tag{1}
$$

action expert 预测速度场 $\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c})$,以 $\mathbf{a}-\epsilon$ 为回归目标：

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{t,\epsilon}\big[\|\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c}) - (\mathbf{a}-\epsilon)\|^2\big]. \tag{2}
$$

**用大白话说**：训练时把真实动作和高斯噪声按比例 $t$ 混合,让网络学会"从噪声指向真实动作"的速度方向;推理时从纯噪声出发,沿着这个学到的速度场做几步欧拉积分,就能采样出连续动作 chunk。这是目前主流 VLA（π0 系列）通用的动作生成范式,四种范式的差异不在这里,而在于给这个动作损失额外加上什么样的中间监督信号。

### 2.2 四种训练范式

| 范式 | 辅助监督 | 预训练损失 | 微调损失 | 核心作用 |
|---|---|---|---|---|
| MindPI | 无 | $\mathcal{L}_{\text{act}}$ | $\mathcal{L}_{\text{act}}$ | 动作单一迁移基线 |
| MindLPI | 语言 | $\mathcal{L}_{\text{act}}, \mathcal{L}_{\text{lang}}$ | $\mathcal{L}_{\text{act}}$ | 用语言监督注入高层动作意图 |
| MindWPI | 未来隐变量 | $\mathcal{L}_{\text{act}}, \mathcal{L}_{\text{lat}}$ | $\mathcal{L}_{\text{act}}, \mathcal{L}_{\text{lat}}$ | 用未来状态预测做正则 |
| MindLWPI | 语言 + 未来隐变量 | $\mathcal{L}_{\text{act}}, \mathcal{L}_{\text{lat}}, \mathcal{L}_{\text{lang}}$ | $\mathcal{L}_{\text{act}}, \mathcal{L}_{\text{lat}}$ | 结合意图约束与状态转移约束 |

**MindLPI**（语言监督 co-training）将动作 chunk 转成 LAP 风格的动作描述文本（离散整数序列或"向前移动 12 cm、向上移动 8 cm、闭合夹爪"式自然语言模板）,作为自回归训练目标：

$$
\mathcal{L}_{\text{MindLPI}} = \mathcal{L}_{\text{act}} + \lambda_{\text{lang}}\mathcal{L}_{\text{lang}}, \quad \lambda_{\text{lang}}=0.1. \tag{4}
$$

消融显示,**去掉动作损失回传 VLM 的梯度截断反而会显著损害效果**（LIBERO-Plus 从 72.3 掉到 45.8）,说明动作损失回传到 VLM 提供了有用的"动作-结果对齐"信号,并非单纯噪声,因此主实验采用不截断梯度的设置。微调阶段只保留 $\mathcal{L}_{\text{act}}$,不再生成语言描述,避免拖慢闭环控制频率。

**MindWPI**（未来隐变量特征对齐）用冻结的 V-JEPA 2 作为隐变量特征提取器,从当前帧和未来帧分别提取 $\mathbf{z}_{\text{cur}}, \mathbf{z}_{\text{fut}}$；当前隐变量作为额外上下文输入 action expert,而 action expert 同时通过一个隐变量解码器预测未来隐变量 $\hat{\mathbf{z}}_{\text{fut}}$：

$$
\mathcal{L}_{\text{lat}} = \|\hat{\mathbf{z}}_{\text{fut}} - \mathbf{z}_{\text{fut}}\|^2, \qquad \mathcal{L}_{\text{MindWPI}} = \mathcal{L}_{\text{act}} + \lambda_{\text{lat}}\mathcal{L}_{\text{lat}}. \tag{5-6}
$$

**用大白话说**：与直接重建未来像素相比,在隐空间对齐未来特征能避开纹理、光照等与控制无关的视觉细节,同时保留动力学、接触、任务进度等对预测"动作会造成什么状态变化"更有用的信息。为防止隐变量预测走捷径直接"偷看"动作 token,论文设计了结构化注意力掩码：隐变量 token 只能看到当前观测和语言上下文、不能看到带噪动作 token；而动作 token 可以看到隐变量 token,把它当作预测性的视觉上下文。推理时只需当前帧隐变量作为条件,不改变部署接口。

**MindLWPI**（语言 + 未来隐变量组合）将三个损失联合优化：

$$
\mathcal{L}_{\text{MindLWPI}} = \mathcal{L}_{\text{act}} + \lambda_{\text{lat}}\mathcal{L}_{\text{lat}} + \lambda_{\text{lang}}\mathcal{L}_{\text{lang}}. \tag{7}
$$

预训练阶段动作损失与隐变量损失比例固定为 1:1,微调阶段隐变量损失权重降为 $\lambda_{\text{lat}}^{\text{ft}}/\lambda_{\text{act}}^{\text{ft}}=0.1:1$（消融显示预训练用强约束 1:1、微调用弱约束 0.1:1 的组合最优,WidowX 均值达 71.9,优于其他比例组合）。为控制推理开销,V-JEPA 2 的 256 个隐变量 token 通过 AvgPool-k4 压缩为 64 个 token,消融显示该压缩方式在性能与开销之间的权衡优于 AvgPool-k16、MLP-k4、MLP-k16。

论文进一步提出**"meta-action space"** 解释框架：MindPI 直接拟合异构、embodiment 相关的碎片化原始动作标签,容易受 embodiment、采样频率、动作定义差异影响；MindLPI 通过语言空间提供高层动作意图约束,MindWPI 通过未来视觉隐空间提供状态转移约束,二者从"做什么"和"动作会改变什么"两个互补维度平滑异构动作监督,MindLWPI 将两者结合形成更平滑、更可迁移的隐式"meta-action"表示。

## 三、关键结果

评测基准：LIBERO（4 个套件,each 500 次 rollout）、LIBERO-Plus（7 类零样本扰动：相机视角/机器人初始状态/语言指令/光照/背景/噪声/物体布局,仅在标准 LIBERO 上训练,不做扰动数据微调）、SimplerEnv（WidowX 4 任务均值,RT-1 的 Visual Matching / Visual Augmentation 两种设置）。

**受控比较主表（Table 2,success rate %）：**

| 方法 | 预训练 | 辅助监督 | LIBERO Avg | LIBERO-Plus Total | WidowX Avg | RT-1 VM | RT-1 VA |
|---|---|---|---|---|---|---|---|
| MindPI w/o PT | 否 | - | 97.0 | 59.9 | 59.6 | 75.7 | 60.4 |
| MindWPI w/o PT | 否 | future latent | 97.4 | 66.1 | 71.9 | 75.2 | 51.6 |
| MindPI (Frozen VLM) | 是 | - | 97.2 | **74.9** | 54.4 | 72.7 | 66.0 |
| MindPI (Full PT) | 是 | - | 97.5 | 68.8 | 65.9 | 68.2 | 55.5 |
| MindLPI | 是 | language | 97.2 | 72.3 | 65.6 | 74.6 | 59.2 |
| MindWPI | 是 | future latent | 98.5 | 72.6 | 74.5 | **86.7** | **71.1** |
| MindLWPI | 是 | language + future latent | **99.1** | 74.8 | **75.5** | 84.4 | 69.8 |

关键现象：**MindPI (Full PT) 相对无预训练基线在 RT-1 VM/VA 上明显退化**（VM 从 75.7 掉到 68.2,VA 从 60.4 掉到 55.5）,是全参数纯动作预训练在异构数据上负迁移的直接证据；冻结 VLM 只训练 action expert（MindPI Frozen VLM）能部分保留语言-视觉泛化能力,LIBERO-Plus Total 达到全表最高的 74.9,但未充分利用机器人数据学习状态变化。MindWPI 在 RT-1 VM/VA 上最强,MindLWPI 在 LIBERO、LIBERO-Plus、WidowX 上最强且 RT-1 上接近最优,验证语言监督与未来隐变量监督的互补性。

**与公开基线对比**（Table 3/4/5,VLAFlow 内部变体仅作受控对比参考,非严格 SOTA 声明）：LIBERO 上 MindLWPI 达到 99.1（对比 π0.5 96.9、OpenVLA-OFT 97.1、π0 94.4）,在 L-Long 套件上提升尤其明显（98.2 vs π0.5 的 92.4）。LIBERO-Plus 零样本鲁棒性上,MindLWPI Total 74.8、MindPI (Frozen VLM) 74.9,均显著高于 OpenVLA-OFT（69.6）、π0（53.6）、RIPT-VLA（64.8）。SimplerEnv 上 MindWPI 取得 RT-1 VM 86.7 / VA 71.1,MindLWPI 取得 WidowX 75.5,均优于同规模或更大规模的公开基线（SpatialVLA、FPC-VLA、MemoryVLA、DD-VLA 等,3B-7B 不等）。

**关键消融**：(1) 预训练数据构成对 MindPI 影响巨大——RoboCOIN 子集单独预训练反而使 WidowX 均值从 63.0（无预训练）掉到 40.4,而 OXE 原始子集能提到 65.1,说明负迁移根源不在于"是否预训练"而在于异构数据源的对齐难度；(2) LoRA 高效微调中,仅在 action expert 侧加 LoRA 即使 r=512（103.8M 参数）也只能到 78.6,远低于全参数微调水平（约 97.8）,而 VLM 侧 LoRA（r=256, 94.4M 参数）即可逼近全参数微调（97.1）,双侧 LoRA（146.3M）可达到 97.8 与全参数微调持平甚至略超,说明下游适配同样需要更新 VLM 的视觉-语言表征,而不只是 action expert。

## 四、评价与展望

**优点**：这是一篇定位清晰的"受控比较"技术报告,严格固定 backbone、动作空间、数据规模与微调协议,专门隔离"训练目标"这一变量,这在 VLA 文献中较为少见（多数新模型论文同时改动架构、数据规模、评测协议,难以归因）。负迁移现象的量化（MindPI Full PT 在 RT-1 上明显劣于无预训练基线）具有实际警示价值,提醒社区"预训练数据越多越好"的直觉在异构 embodiment 场景下并不总成立。梯度截断消融、隐变量压缩消融、LoRA 注入位置消融等一系列附加实验也提供了不少可复用的工程经验。

**局限与开放问题**：其一,论文本质是工程性的"控制变量对照报告"而非提出新架构或新理论,MindLPI 的语言监督形式局限于 LAP 风格模板/离散整数动作描述,而非自由格式或来自 VLM 预训练知识的丰富语言监督,这与 π0.5 等利用网络多模态数据的路线相比信息密度较低；其二,所有下游评测均在仿真基准（LIBERO / LIBERO-Plus / SimplerEnv）完成,论文本身在结论中明确将"在真实机器人平台上验证"列为未来工作,当前结论的 sim-to-real 外推性有待验证；其三,预训练规模（约 5,000 小时,"medium-scale"）相对当前部分工业级 VLA（如文中提及的 LingBot-VLA、ABot-M0）仍偏小,未来在更大规模上四种范式的相对排序是否稳定尚不确定；其四,预训练阶段的 loss 比例（如 MindLWPI 的 act:lat = 1:1）本身没有做充分的 pretraining-stage 消融（论文承认这点,列为未来工作）,当前推荐设置更多来自有限的下游微调比例扫描外推。与同期使用 V-JEPA 系列做隐变量世界模型对齐的 JEPA-VLA、VLA-JEPA 等工作相比,本文的贡献重点不在于隐变量对齐机制本身的创新,而在于把它和语言 co-training 放入同一受控框架中做系统对比,这一"对比研究"定位使其更适合作为方法选择的参考文献,而非新 SOTA 模型的发布。

## 参考

1. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.
2. Black et al. *π0.5: a vision-language-action model with open-world generalization.* 2025.
3. Assran et al. *V-JEPA 2: Self-supervised video models enable understanding, prediction and planning.* arXiv:2506.09985, 2025.
4. Luo et al. *Being-H0.7: A latent world-action model from egocentric videos.* arXiv:2605.00078, 2026.
5. Kim et al. *OpenVLA: An open-source vision-language-action model.* arXiv:2406.09246, 2024.
