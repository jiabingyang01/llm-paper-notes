# Olaf-World：为视频世界模型确立潜动作的语义方向

> **论文**：*Olaf-World: Orienting Latent Actions for Video World Modeling*
>
> **作者**：Yuxin Jiang, Yuchao Gu, Ivor W. Tsang, Mike Zheng Shou et al.
>
> **机构**：Show Lab, National University of Singapore；CFAR & IHPC, Agency for Science, Technology and Research (A*STAR), Singapore
>
> **发布时间**：2026 年 02 月（arXiv 2602.10104）
>
> **发表状态**：ICML 2026（Proceedings of the 43rd International Conference on Machine Learning, PMLR 306）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.10104) | [PDF](https://arxiv.org/pdf/2602.10104)
>
> **分类标签**：`潜动作学习` `视频世界模型` `表征对齐` `跨场景迁移` `动作可控生成`

---

## 一句话总结

论文形式化刻画了无标签潜动作学习中的"跨场景不可辨识"问题(同一动作在不同视频里可能被编码成完全不同的潜方向),提出 **Seq∆-REPA**：用冻结自监督视频编码器(V-JEPA2)给出的语义效应方向,作为跨场景共享坐标系来锚定潜动作。在此之上构建的 **Olaf-World** 潜动作世界模型,在 MIND 数据集的跨视角(1st-person/3rd-person)动作迁移实验中,把跨域探针 Macro-F1 从 AdaWorld 基线的 0.48 提升到 0.63(1st→3rd)、从 0.48 提升到 0.83(3rd→3rd);仅用 1 段标注视频(约 1 分钟标注成本)微调后,动作跟随误差 RPE-rot 就从 AdaWorld 的 0.642 降到 0.468。

## 一、问题与动机

动作可控世界模型通常依赖游戏引擎导出的逐帧对齐动作标签,成本高,还绑定特定控制接口。潜动作学习(latent action learning)试图直接从无标签视频里,用逆动力学编码器推断动作 $z_i$,再用前向模型预测下一帧,从而摆脱标签依赖。但作者指出这类方法存在两个失效模式。第一是 **shortcut learning(上下文泄漏)**：表达力强的解码器可以通过编码上下文相关的视觉线索来降低重建损失,而这些线索并非真正可迁移的控制量。第二是 **跨场景不可辨识**：标准的逐步重建目标只在单个片段内部起作用,不同视频片段完全可以各自使用不同的隐坐标系,重建损失却保持不变。结果是,同一语义动作(例如"前进")在不同环境里会映射到潜空间中不同的方向,破坏了跨场景迁移和下游可控性的基础。

作者在附录给出了一个形式化命题(Proposition A.1)来支撑这个动机：对任意一族按场景索引的双射 $\{G_c\}_c$,若用 $G_c$ 重参数化编码器与解码器,局部预测损失(逐步重建目标)完全不变。这说明局部预测目标本身无法唯一确定一个跨场景共享的潜坐标系,必须引入额外约束才能获得可迁移的动作表示。

## 二、核心方法

**基础 β-VAE 潜动作模型**。给定片段 $x_{0:K}$,因果逆动力学编码器输出 $q_\phi(z_i \mid x_{0:i+1})$,前向解码器 $p_\theta(x_{i+1}\mid x_i,z_i)$ 预测下一帧,标准目标为

$$\mathcal{L}^{VAE}_{\theta,\phi}=\frac{1}{K}\sum_{i=0}^{K-1}\Big(-\mathbb{E}_{q_\phi}[\log p_\theta(x_{i+1}\mid x_i,z_i)]+\beta\,\mathrm{KL}(q_\phi(z_i\mid x_{0:i+1})\,\|\,p(z_i))\Big)$$

用大白话说：这是"看当前帧加下一帧推出一个动作变量,再用这个变量和当前帧重建下一帧"的自编码目标,KL 项把后验拉向标准正态先验。这个目标单独使用时,不能阻止上面两种失效模式。

**Seq∆-REPA(核心创新)**。关键洞察是动作本身不可观测,但动作引起的语义效应在视频里是可观测的,因此可以充当跨场景共享的参照系。用一个冻结的自监督视频编码器 $f$(如 V-JEPA2 ViT-Giant)提取时空特征并做空间池化,得到逐帧描述子 $s_i$,再定义整个片段的效应方向为时序特征差的均值:

$$\tau_*=\frac{1}{K}\sum_{i=0}^{K-1}(s_{i+1}-s_i)\in\mathbb{R}^D$$

用大白话说：无论画面外观如何,"发生了什么样的语义变化"这件事本身是可以跨场景比较的。用时序差分而不是绝对特征,天然抑制了静态外观信息,凸显了动态变化。

在潜动作一侧,把逆模型推出的动作序列 $z_{0:K-1}$ 聚合,再经一个可训练 MLP 投影头 $h_\psi$ 映射到同一特征维度:$\bar z=\frac{1}{K}\sum_i z_i$,$u=h_\psi(\bar z)\in\mathbb{R}^D$。随后用余弦相似度把整合后的控制方向 $u$ 与效应方向 $\tau_*$ 对齐:

$$\mathcal{L}^{Seq\Delta\text{-REPA}}_\psi=1-\langle \mathrm{norm}(u),\mathrm{norm}(\tau_*)\rangle$$

用大白话说：这个损失强迫"模型认为自己执行了什么动作"和"视频里客观发生的语义变化方向"对齐。它是一个 control-to-effect 约束,和常见表征对齐工作里的 feature-to-feature 对齐(比如让生成器内部状态匹配预训练特征)不是同一类操作。序列级、方向级的相似度约束,配合 $\ell_2$ 归一化,让对齐目标对特征尺度不敏感,消融实验证明这两点都是必要的设计。总损失为

$$\mathcal{L}_{LAM}=\mathcal{L}^{VAE}_{\theta,\phi}+\lambda\mathcal{L}^{Seq\Delta\text{-REPA}}_\psi$$

其中 $\lambda=0.02$,参考编码器全程冻结。

**Olaf-World 流水线**。第一阶段用上述目标在无标签视频(MiraData 的 3D Rendering 与 City Walking 子集)上预训练潜动作模型(LAM)。第二阶段冻结 LAM,为每帧视频推出潜动作 $z_t$,线性投影后叠加到扩散时间步嵌入,经 AdaLN-Zero 调制每个 DiT block(骨干为 SkyReels-V2-1.3B I2V,分辨率 540p,片段长度 T=97 帧;由于 3D 视频 VAE 的时间压缩率 r=4,每连续 r 步潜动作被分组为一个潜时间条件向量),用标准 flow-matching 目标训练动作条件视频生成模型。第三阶段是特定场景适配:给定目标环境的少量真实动作标签,学习一个轻量动作适配器 $A_\eta$(离散动作可实现为嵌入表,用同类潜动作的均值做原型初始化),只微调该适配器和主干上的小 rank LoRA,即可用极少标注数据把预训练好的通用潜动作空间对齐到具体控制接口。

## 三、关键结果

实验主要在 MiraData(用于预训练)和 MIND(一个基于 Unreal Engine 5、带逐帧动作标签的开放数据集,含视角不同的 1ST-P/3RD-P 两个子集,共享 8 类动作:WSAD 导航加视角控制)上进行,对照基线为同架构、同数据、同训练与适配预算下的 SOTA 潜动作世界模型 AdaWorld。

跨域线性探针(Macro-F1,越高越好,评估潜动作是否线性可解码且跨域一致):

| 方法 | 1st→1st | 1st→3rd | 3rd→3rd | 3rd→1st |
|---|---|---|---|---|
| AdaWorld | 0.6004 | 0.4820 | 0.4827 | 0.4999 |
| Olaf-World | **0.8138** | **0.6250** | **0.8256** | **0.5904** |

数据高效适配(以 1ST-P 域为例,RPE-rot 越低越好,反映动作跟随的忠实度):

| #标注视频 | AdaWorld RPE-rot | Olaf-World RPE-rot |
|---|---|---|
| 0(零样本迁移) | 1.0844 | 0.8773 |
| 1(约 1 分钟标注) | 0.6420 | 0.4680 |
| 50(约 2 小时标注) | 0.3834 | 0.3785 |

在两个域、所有适配预算下,Olaf-World 的 RPE-trans/RPE-rot 均低于 AdaWorld 和"直接用真值动作条件生成"的 DirectAct 基线,且视觉质量(VBench 图像质量、时序一致性)基本持平。在未见视觉场景(风格与物体分布外)下适配后重测,Olaf-World 的 RPE-trans/RPE-rot(0.0478/1.2221)仍优于 DirectAct(0.0547/1.2343)和 AdaWorld(0.0482/1.7063),说明潜动作预训练带来的鲁棒性并非源于对适配集外观的过拟合。

消融实验显示:去掉时序差分(对齐静态特征而非效应方向)会让跨域 F1 明显下降(3rd→1st 从 0.5904 降到 0.4823);去掉 $\ell_2$ 归一化(退化为尺度敏感的 MSE 对齐)同样有损跨域一致性。参考编码器选择上,具备时序建模能力的视频编码器(V-JEPA2、VideoMAEv2)明显优于纯图像编码器(DINOv3)和无对齐基线(NONE,即等价于 AdaWorld 的设置),说明效应方向的质量依赖于参考编码器对动态信息的敏感度,而非某个特定骨干网络。

## 四、评价与展望

论文对"为什么潜动作学不出可迁移的控制接口"给出了少见的形式化刻画,把动机从直觉提升到定理层面,论证链条清晰。方法本身简洁:只是在已有的逆动力学序列上增加一个由冻结视频编码器提供目标的余弦对齐损失,不需要额外标签,不改变主干结构,工程可移植性强。受控对比设计(与 AdaWorld 严格同架构、同预算)也让改进可以较可信地归因于目标函数本身,而非训练资源的差异。

局限也比较明显。验证场景集中在游戏渲染与城市漫游类视频(MiraData、基于 UE5 的 MIND),动作语义以相机运动和刚体导航为主,尚未在接触丰富的机器人操作场景中得到验证;论文自己在附录局限部分也指出,下一步需要引入物理一致性约束,并扩展到多物体接触的复杂操作。另外,Seq∆-REPA 的效应信号是单一整合方向,无法区分相机自运动、可控智能体、其他智能体行为、环境驱动事件等多种并存的动作来源,论文展示的失败案例之一正是"新角色进入"这类事件驱动动作在跨场景迁移时被误解释为背景或镜头漂移。当前潜动作还是逐帧的原子控制,尚未形成层级化的技能表示,长时程 rollout 的组合能力有限。

从与相关工作的关系看,本文与 AdaWorld、"Learning to act without actions"(LAPA)、Genie 等潜动作世界模型一脉相承,共同目标都是从无标签视频中蒸馏出可迁移的控制接口;此前工作主要依靠隐式的潜空间约束来缓解 shortcut learning,比如 VQ 离散化或强调运动的重建目标,本文则首次把跨场景一致性显式表述成一个可优化的序列级对齐目标。这个思路与 REPA 系列表征对齐工作(如 Representation Alignment for Generation、VideoREPA)在技术手段上接近,但对齐的对象从"生成器内部状态匹配预训练特征"换成了"控制信号匹配语义效应方向",是一次有意义的问题重构。

开放问题方面,论文在结论与附录中列出了效应目标与对齐形式的进一步探索、层级化潜动作(技能)表示、物理规则一致性约束、多实体分解控制等方向。其中最值得关注的一点,是效应对齐的潜动作能否作为跨具身(human→robot)的可迁移技能表示,通过具身特定的动作到技能适配器实现桥接。这一构想若能落地,会为"从人类视频中低成本习得可迁移操作控制信号"提供一条与显式动作标注、光流或关键点追踪方案并行的新路径,但论文目前尚未在真实机器人操作数据上给出任何实证。

## 参考

- Gao, S. et al. AdaWorld: Learning adaptable world models with latent actions. ICML, 2025.
- Schmidt, D. and Jiang, M. Learning to act without actions. ICLR, 2024.
- Bruce, J. et al. Genie: Generative interactive environments. ICML, 2024.
- Assran, M. et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv:2506.09985, 2025.
- Yu, S. et al. Representation alignment for generation: Training diffusion transformers is easier than you think. ICLR, 2025.
