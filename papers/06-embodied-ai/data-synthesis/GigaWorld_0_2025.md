# GigaWorld-0：世界模型作为数据引擎赋能具身智能

> **论文**：*GigaWorld-0: World Models as Data Engine to Empower Embodied AI*
>
> **作者**：GigaWorld Team（Angen Ye, Boyuan Wang, Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu et al.，按字母序署名）
>
> **机构**：GigaAI
>
> **发布时间**：2025 年 11 月（arXiv 2511.19861，v2）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2511.19861) | [PDF](https://arxiv.org/pdf/2511.19861)
>
> **分类标签**：`世界模型` `数据引擎` `VLA 预训练` `视频生成` `3DGS`

---

## 一句话总结

GigaWorld-0 把世界模型定位为 VLA 训练的**数据引擎**,由视频分支 **GigaWorld-0-Video**(可控外观/视角/跨具身的 IT2V 生成)与三维分支 **GigaWorld-0-3D**(3DGS 重建 + 可微系统辨识 + 可执行动作规划)组成;其 2B 激活参数的基础视频模型在 PBench Robot Set 上以 82.07 的综合分超过 14B 的 Cosmos-Predict2(79.88),并用生成数据训练出的 GigaBrain-0 在真实机器人上完成叠衣、备餐等任务而**训练全程无需真机交互**。

## 一、问题与动机

具身操作(尤其 VLA 学习)长期受制于真机数据采集的高成本、低多样性:真实数据在纹理、颜色、光照、视角上的覆盖有限,导致策略部署到复杂真实环境时鲁棒性差;传统仿真渲染虽能海量生成,却存在明显 sim2real 外观差距。作者主张用世界模型充当"高保真物理世界代理",可控、可扩展、低成本地合成具身交互数据,从而缓解数据瓶颈。

核心挑战在于:高质量具身数据需要**同时**满足纹理真实(视频擅长)与几何一致/物理合理(三维仿真擅长),二者此前难以统一。GigaWorld-0 的思路是让两条分支协同——视频分支负责逼真外观与细粒度可控生成,三维分支负责几何一致性与物理属性,联合产出"纹理丰富 + 几何一致 + 物理落地 + 指令对齐"的具身交互数据。

## 二、核心方法

整体由 8 个子模块构成(下表按原文 Table 1):

| 子模块 | 功能 |
| --- | --- |
| GigaWorld-0-Video-Dreamer | 具身场景的 image-text-to-video 基础模型 |
| GigaWorld-0-Video-AppearanceTransfer | 文本引导的外观迁移,改纹理/材质/光照 |
| GigaWorld-0-Video-ViewTransfer | 按用户指定相机外参渲染新视角视频 |
| GigaWorld-0-Video-MimicTransfer | 把第一人称人手演示翻译成机械臂轨迹视频 |
| GigaWorld-0-3D-FG | 生成前景可操作物体的 3D 资产 |
| GigaWorld-0-3D-BG | 用 3DGS 重建背景环境 |
| GigaWorld-0-3D-Phys | 建模物体物理属性并做可微系统辨识 |
| GigaWorld-0-3D-Act | 合成可执行、物理一致的机械臂动作 |

### 2.1 视频基础模型 GigaWorld-0-Video-Dreamer

采用 flow-matching 建模生成过程:

$$\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c})$$

其中 $\mathbf{z}_t$ 是 $t$ 时刻的隐变量,$\mathbf{c}$ 是文本+图像条件,$\mathbf{v}_\theta$ 是学习的速度场。

用大白话说:不去一步步"加噪-去噪",而是学一条从噪声直连到真实视频隐变量的"平滑流",训练/采样都更省。

输入表示上用 3D-VAE 压缩原始视频(时空压缩比 $4\times8\times8$,得到 16 通道视频隐变量),再做 $1\times2\times2$ patch 化,配 3D 旋转位置编码(3D-RoPE)、T5 文本编码器,骨干是带 sparse attention 的 DiT。FFN 内嵌 Mixture-of-Experts:

$$\mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_r} g_{i,t}\,\mathrm{FFN}_i(\mathbf{u}_t)$$

门控 $g_{i,t}$ 仅在专家得分进入 Top-$K_r$ 时取 $s_{i,t}=\mathrm{softmax}(\mathbf{u}_t^\top \mathbf{e}_i)$,否则为 0。与 DeepSeek-V2 不同,这里**不设共享专家**,配置 $N_r=4$ 路由专家、激活 $K_r=2$;并沿用 DeepSeek-V3 的负载均衡损失(平衡因子 $\alpha=0.01$)。

用大白话说:MoE 让不同专家自动分工处理视频里语义不同的区域(手臂 / 桌面 / 物体),既提容量又不必每个 token 都跑全部参数;去掉共享专家避免冗余,负载均衡损失防止"专家旱涝不均"。

**作为数据引擎的关键——IDM 回标动作。** Dreamer 在同一初始帧下由不同文本 prompt 生成多样未来视频后,再用一个逆动力学模型 GigaWorld-0-IDM 从生成视频反推关节轨迹:

$$\boldsymbol{\theta}_{1:T} = f_{\mathrm{IDM}}(\mathbf{V})$$

$\boldsymbol{\theta}_t$ 是所有 $D$ 关节的旋转角。与先前 IDM(AnyPos)不同,GigaWorld-0-IDM 用 **masked training**:先用 SAM2 分割出机械臂区域,只把分割后的臂区喂给 IDM,以削弱杂乱背景对预测的干扰。Fig 3 显示其能准确恢复 12 个臂关节 + 2 个夹爪自由度(共 14 维)。由此得到 $(\mathbf{V}, \boldsymbol{\theta}_{1:T})$ 成对数据,无需真机交互即可为 VLA 提供动作监督。

### 2.2 三种后训练可控分支

- **AppearanceTransfer(外观泛化)**:在 Dreamer 上加轻量控制分支。因 MoE 架构下复制 ControlNet 会成倍膨胀参数,改用"控制隐变量拼接":把深度/法线(VideoDepthAnything + LOTUS 抽取)经 3D-VAE 编码后与噪声隐变量沿通道拼接,再经通道压缩 MLP。文本 prompt 独立操控纹理/材质/光照,支持 real2real 与 sim2real。
- **ViewTransfer(视角泛化)**:从单视角机器人视频合成新视角,同时变换动作以保持任务一致。世界坐标系下末端位姿不变的约束给出

  $$\mathbf{K}_t = \left(\mathbf{T}^{\text{base}\to\mathcal{W}_B}\right)^{-1}\cdot \mathbf{T}^{\text{base}\to\mathcal{W}_A}\cdot \mathbf{T}_t^{\text{ee}\to\text{base}}$$

  即机器人基座从 $\mathcal{W}_A$ 迁到 $\mathcal{W}_B$ 后新的末端-相对-基座位姿 $\mathbf{K}_t$。因缺配对多视角真实视频,用 **double-reprojection**:MoGe 估深度→把 $\mathbf{V}_A$ warp 到目标视 $\mathcal{W}_B$→再投回原视得到自监督对(video condition-1 管背景几何,video condition-2 在物理仿真器 SAPIEN 里渲染变换后的臂动作管手臂几何)。
- **MimicTransfer(跨具身)**:把第一人称人手演示翻译为机械臂视频。训练时因缺"人手-机械臂"配对,仅用机械臂视频构造:condition-1 = 抠掉臂的场景,condition-2 = 用原臂轨迹驱动的仿真臂;重建原始未抠图视频。推理时抠掉人手作 condition-1,由标注的人手末端位姿经 IK 解出关节角、在仿真器渲染臂作 condition-2,合成模仿人类动作的机械臂视频。做法承接 MimicDreamer。

**多视角与加速**:多视角图沿宽度拼成全景输入,微调后即可生成时空一致的多视角视频而不改架构;推理端用去噪步蒸馏(压到单步)+ FP8,相比标准扩散实现超 $50\times$ 加速。生成后有质量评估流水线(几何一致、多视角一致、文视对齐、物理合理)算综合分,决定该样本用于预训练/微调/丢弃。

### 2.3 三维分支 GigaWorld-0-3D

以 3DGS 为核心表示。**3D-FG** 从单图/文本生成前景资产:Aesthetic-Checker 做纹理质控、GPT-4o 驱动的 ImageSegChecker 查分割、Trellis 做 image-to-3D(同时出 mesh 与 3DGS)、MeshGeoChecker 从四正交视角查几何完整性,过关资产导出 URDF。**3D-BG** 用 3DGRUT(每个高斯配 7 个代表点以支持非针孔/卷帘相机)重建背景,借鉴 ReconDreamer 训练视图修复模型幻化中间视,二阶段稠密 3DGS,再经泊松重建转水密网格。**3D-Phys** 用基于 PINN 的可微物理三阶段辨识机械臂参数(摩擦/刚度/阻尼):①真实轨迹配随机物理参数生成仿真 rollout;②训代理模型 $\mathcal{M}_{f,p,d}$ 逼近仿真动力学;③固定代理、梯度下降精修参数到 $(f^*,p^*,d^*)$。被操作物体用基于 Qwen3-VL 的多模态物理专家估质量/摩擦系数;可形变物体把弹簧-质点系统绑到高斯粒子(承 PhysTwin 精神)。**3D-Act** 两档动作生成:简单场景用少量遥操/规则演示 + MimicGen 扩展;复杂场景用遥操做冷启 + 快速在线 RL(RLPD)自举策略,收敛后批量产轨迹。

### 2.4 训练与 GigaTrain

数据 = 公开集(AgiBotWorld、RoboMind)+ 自采数据(Agilex Cobot Magic 与 AgiBot G1 平台,累计 3,100 m²,覆盖工业/商业/办公/居家/实验室五大类 14 种场景)。视频以 $480\times768$、61 帧序列训练。训练框架 GigaTrain 支持 DeepSpeed ZeRO(0–3)/FSDP2、FP16/BF16/FP8 混合精度、梯度累积/检查点/EMA,sparse attention 用 NATTEN(优于 SageAttention)。

## 三、实验结果

### 训练效率(Table 2,8×H20,batch 32)

| 配置 | 框架 | 时间(s/step) | 显存(MB) |
| --- | --- | --- | --- |
| 无 FP8/无稀疏 | FSDP-2 | 33.19 | 89355 |
| 仅 FP8 | FSDP-2 | 29.53 | 71857 |
| FP8+稀疏 | FSDP-2 | **25.38** | 73131 |
| FP8+稀疏+激活重算+MoE | FSDP-2 | 33.38 | 73997 |

FSDP-2 显存最省;FP8 一致地同时降显存与耗时;加 sparse attention 再把每步从 29.53 压到 25.38 s。上 4 专家 MoE 后需对 FFN 做激活重算才能在受限显存下稳定收敛(Zero0 直接 OOM)。

### PBench Robot Set(Table 3,越高越好)

| 模型 | 激活参数 | Domain Score | Overall Score |
| --- | --- | --- | --- |
| Cosmos-Predict2 | 14B | 84.0 | 79.88 |
| Wan2.2 | 14B | 83.2 | 78.85 |
| Wan2.2 | 5B | 80.1 | 77.15 |
| Cosmos-Predict2.5 | 2B | 84.7 | 79.95 |
| **GigaWorld-0-Video-Dreamer** | **2B** | **88.2** | **82.07** |

以最小激活参数(2B)拿到最高 Domain(88.2)与 Overall(82.07),细分项如 i2v-bg 97.6、i2v-s 97.6、sub-con 12.6、o-con 91.9 多数领先。

### DreamGen Bench(Table 4,在 GR1 机器人数据上微调后评测)

| 方法 | 参数 | GR1-Env (Qwen-IF/GPT-IF/PA) | GR1-Object | GR1-Behavior |
| --- | --- | --- | --- | --- |
| Cosmos-Predict2 | 14B | 0.966 / 0.552 / 0.586 | 0.840 / 0.760 / 0.471 | 0.894 / 0.638 / 0.458 |
| Wan2.2 | 14B | 0.900 / 0.760 / 0.549 | 0.700 / 0.780 / 0.531 | 0.870 / 0.570 / 0.477 |
| Cosmos-Predict2.5 | 2B | 0.930 / 0.480 / 0.534 | 0.920 / 0.240 / 0.503 | 0.830 / 0.320 / 0.471 |
| **GigaWorld-0-Dreamer** | **2B** | 0.966 / 0.586 / 0.529 | 0.920 / 0.540 / 0.481 | 0.894 / 0.638 / 0.446 |

尽管预训练语料几乎不含 GR1 数据,2B 激活的 GigaWorld 在三个场景的指令跟随保真度(Qwen-IF)上一致优于同量级 Cosmos-Predict2.5-2B;不过物理对齐分 PA 在部分场景略低于 14B 基线,说明其优势主要在可控性/指令对齐而非物理逼真度绝对值。

### 下游任务(5.3 节)

用 GigaWorld-0 合成数据训练 GigaBrain-0,报告其在真实机器人上完成:灵巧操作(叠衣 Laundry Folding、备纸巾 Paper Towel Preparation)、长时程移动操作(备果汁 Juice Preparation、清桌 Table Bussing)、动态移动作业(搬箱 Boxes Moving、搬洗衣篮 Laundry Baskets Moving),全程无需大量真机演示。**注意**:本文仅给出定性成功截图,具体成功率/消融的定量数字被指向 GigaBrain-0 论文(arXiv 2510.19430),本文并未列出。

## 四、局限性

1. **下游定量证据外置**:数据引擎最关键的"训出的策略好多少"缺乏本文自证的成功率数字,只有定性图,读者需另查 GigaBrain-0 论文,削弱了对"数据引擎有效性"的直接说服力。
2. **物理逼真度非最强项**:DreamGen 的 PA 分显示其物理对齐并不占优,生成视频仍可能有幻觉/伪影——虽有质量过滤流水线兜底,但过滤阈值与漏检率未量化。
3. **系统庞杂、耦合度高**:8 个子模块 + IDM + 多个外部依赖(SAM2、MoGe、SAPIEN、Trellis、Qwen3-VL、GPT-4o、MimicGen、RLPD),复现与稳定性成本高,单点误差(如 IDM 回标错、深度估计偏)会沿数据链传导污染训练集,文中未做误差传播分析。
4. **IDM 回标动作的真值缺失**:生成视频没有真实关节真值,用 IDM 反推的动作本身带模型误差,"生成视频 + 预测动作"作为监督存在自洽性风险(用模型标模型),文中仅在少量未见任务上定性验证。
5. **动作/物理分支偏工程管线**:3D-Act、3D-Phys 更多是拼装既有方法(MimicGen/RLPD/PINN 系统辨识/PhysTwin),原创方法贡献相对薄弱。

## 五、评价与展望

**优点**:①把"逼真外观(视频)"与"几何/物理一致(3D)"两条互补路线统一进一个数据引擎,方法论上比纯视频世界模型(Cosmos、DreamGen、Genie Envisioner)或纯 3D/仿真管线都更完整,直击具身数据"既要真实又要物理落地"的痛点;②工程性极强——FP8 + 稀疏注意力 + 单步蒸馏把成本压到可落地量级(>50× 加速、8×H20 可训),对社区复现友好;③以 2B 激活参数在 PBench 超过 14B 模型,验证了 MoE + 具身语料的数据效率;④四个可控分支(外观/视角/跨具身/多视角)覆盖了 VLA 泛化最缺的几个维度,ViewTransfer 的动作同步变换、MimicTransfer 的无配对训练构造较巧。

**与公开工作的关系**:视频侧延续 Cosmos/Wan 系列的 DiT + flow-matching 路线,但用 MoE + 具身数据做差异化;外观/视角迁移与同组 EMMA、RoboTransfer、EgoDemoGen、EmbodiedDreamer 一脉相承(大量自引);3D 侧站在 3DGS/3DGRUT/Trellis/ReconDreamer 肩上;跨具身承接 MimicDreamer。整体是"集大成的系统论文"而非单点算法突破,定位类似把一整套 real2sim2real 数据流水线产品化。

**开放问题与可能改进**:(a) 闭环自提升——论文展望把数据引擎升级为可交互的 policy environment 做 model-based RL,乃至"policy co-designer",但当前仍是开环单向产数据,真机 rollout 反哺世界模型的闭环尚未实现;(b) 数据质量的因果度量——需要把"生成数据的哪个维度(纹理/视角/物理)真正带来下游收益"拆开做受控消融,而非笼统汇报;(c) IDM 回标可引入不确定性估计/物理约束正则,或与 3D-Act 的仿真真值互校,降低"模型标模型"风险;(d) 物理逼真度短板可结合 3D-Phys 的可微辨识把物理先验反注入视频生成,而非两分支松耦合。总体看,这是一篇工程完整度高、对"世界模型即数据引擎"范式有较强示范意义的工作,学术新意主要体现在系统集成与数据效率而非单一方法创新。

## 参考

1. Jang et al., *DreamGen: Unlocking Generalization in Robot Learning through Video World Models*, arXiv 2505.12705, 2025.(DreamGen Bench,主要对比基准)
2. Agarwal et al., *Cosmos World Foundation Model Platform for Physical AI*, arXiv 2501.03575, 2025.(Cosmos 系列基线)
3. GigaAI et al., *GigaBrain-0: A World Model-Powered Vision-Language-Action Model*, arXiv 2510.19430, 2025.(用本文数据训练的下游 VLA)
4. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM ToG, 2023.(3D 分支核心表示)
5. Liu et al., *DeepSeek-V3 Technical Report*, arXiv 2412.19437, 2024.(MoE 负载均衡损失来源)
