# CompressionGap：压缩缺口——离散动作 token 化为何限制视觉-语言-动作模型的扩展能力

> **论文**：*The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling*
>
> **作者**：Takuya Shiba
>
> **机构**：Shibattic Inc.
>
> **发布时间**：2026 年 04 月（arXiv 2604.03191）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.03191) | [PDF](https://arxiv.org/pdf/2604.03191)
>
> **分类标签**：`VLA扩展性` `离散动作tokenization` `信息瓶颈理论` `Diffusion Policy` `OAT` `视觉编码器scaling`

---

## 一句话总结

用信息论的数据处理不等式论证：视觉-动作管线的扩展行为由管线中**最紧的信息瓶颈**所在位置决定——连续动作表示（Diffusion Policy）的瓶颈在视觉编码器本身,编码器越强策略越好；离散动作表示（OAT）的瓶颈在固定容量的 codebook（约 80 bits/action chunk）,编码器质量提升会被量化阶段截断、无法传导到下游。LIBERO-10 上把编码器从 ResNet-18 升到 SigLIP,Diffusion Policy 提升 +21.2\~+26.0 个百分点,OAT 只提升 +3.6\~+10.4 个百分点；把 codebook 容量从 1000 扩到 1920 时,OAT 的 Δenc 从 +3.6pp 跃升到 +15.2pp,为"codebook 是绑定约束"提供了因果证据。

## 一、问题与动机

VLA 扩展的一条常见默认假设是：升级视觉编码器（如从 ResNet-18 换成语义更丰富的 SigLIP）应当像在纯视觉-语言建模中一样,直接带来下游操作性能的提升——[Tong et al., 2026] 的大规模多模态预训练研究已经证实更好的视觉表征能一致地转化为更强的理解与生成能力。

作者质疑这一假设是否对机器人操作策略同样成立,并指出关键变量是**动作如何被表示**：如果动作被离散 token 化（如 OAT、FAST、VQ-BeT、逐维度 binning）,视觉编码器的改进能否穿过量化阶段传导到执行端,此前没有被系统研究过。论文通过对比两种代表性动作表示家族——离散 token 化（OAT）与连续去噪（Diffusion Policy）——在相同编码器升级下的敏感度,检验这一假设在何种条件下成立、在何种条件下失效。

## 二、核心方法

**信息通路建模。** 把观测到动作的流水线抽象为马尔可夫链 $O \to Z \to A$（$O$ 为原始观测,$Z$ 为学到的表征,$A$ 为执行动作）。由数据处理不等式：

$$I(O;A) \le \min\big(I(O;Z),\, I(Z;A)\big).$$

用大白话说：把这条流水线看成一根水管,不管上游怎么疏浚,整条管道的流量都由**最窄的那一段**决定；只有当被升级的那一段恰好是最窄处时,升级才会让下游流量变大。

**连续通路（Diffusion Policy）：** $O \xrightarrow{f_{\text{enc}}} Z \xrightarrow{\epsilon_\theta} A$,全程在连续空间操作,$I(Z;A)$ 不受硬性组合上界约束,只受去噪网络容量和 $Z$ 本身质量限制,因此绑定瓶颈通常是 $I(O;Z)$,即视觉编码器。

**离散通路（OAT 及其他离散 tokenizer）：** 在 $Z$ 与 $A$ 之间多插入一个量化阶段 $Q$：$O \xrightarrow{f_{\text{enc}}} Z \xrightarrow{Q} T \xrightarrow{\mathcal{T}^{-1}} A$。对该扩展链应用数据处理不等式：

$$I(O;A) \le I(Z;T) \le \log_2 |\mathcal{V}|^{H_l} = H_l \log_2 |\mathcal{V}|.$$

OAT [Liu et al., 2026] 用一个带可学习寄存器 token 的 Transformer 编码器把动作 chunk 聚合成 $H_l$ 个隐向量,再用 Finite Scalar Quantization（FSQ,levels $[8,5,5,5]$,codebook 大小 $|\mathcal{V}|=1000$）离散化,自回归策略再逐 token 生成 $T_{1:H_l}$。取 $H_l=8$、$|\mathcal{V}|=1000$ 时,该硬上界约为每个动作 chunk **80 bits**——只要现有编码器已经能把这 80 bits 填满,再丰富的上游表征都会在量化阶段被丢弃,绑定瓶颈是 $I(Z;T)$ 而非 $I(O;Z)$。

**实验设计。** 构造 3 个二元变量的 $2\times2\times2=8$ 条件全因子实验：动作表示（OAT / Diffusion Policy,共享同一 Transformer 策略骨干与 OAT 官方代码库,DP 用 100 步训练/10 步推理的 DDIM）、视觉编码器（ResNet-18,64 维,含空间 softmax 池化 / SigLIP,1152 维,预提取离线缓存特征）、模型规模（M：4 层 Transformer,嵌入维 256,4 头；L：6 层,嵌入维 384,6 头）。另设两组补充因果实验：(1) **编码器质量梯度**——固定 M 规模,在 ResNet-18、SigLIP、SigLIP 2、DINOv2 ViT-L/14 四种编码器上比较两种动作表示；(2) **codebook 容量扫描**——固定 M 规模,在 $|\mathcal{V}|=1000/1920/4375$（FSQ levels 分别为 $[8,5,5,5]$、$[8,8,6,5]$、$[7,5,5,5,5]$,对应约 80/87/97 bits）下重新训练 tokenizer 与策略。全部在 LIBERO-10（10 任务,每任务 50 条示教,Franka Panda 7 维动作空间,预测 32 步动作 chunk、执行前 16 步后重推理）上评测,单卡 A100 训练 300 epoch,每 50 epoch 用 500 次 rollout（每任务 50 次）评估,报告峰值成功率。

## 三、关键结果

**表 1：因子实验（LIBERO-10,峰值成功率 %）——升级编码器的收益 Δenc：**

| 动作表示 | 模型规模 | ResNet-18 | SigLIP | Δenc |
|---|---|---|---|---|
| Diffusion Policy | M | 36.4 | 57.6 | **+21.2** |
| Diffusion Policy | L | 44.0 | 70.0 | **+26.0** |
| OAT | M | 53.8 | 57.4 | +3.6 |
| OAT | L | 48.0 | 58.4 | +10.4 |

**表 2：编码器质量梯度（M 规模,LIBERO-10）：**

| 编码器 | 维度 | Diffusion Policy | OAT |
|---|---|---|---|
| ResNet-18 | 64 | 36.4 | 53.8 |
| SigLIP | 1152 | 57.6 | 57.4 |
| SigLIP 2 | 1152 | 62.8 | 44.2 |
| DINOv2 ViT-L/14 | 1024 | 63.8 | 51.0 |

DP 的成功率随编码器质量**单调**提升；OAT 则在 64 维到 1152/1024 维的量级跃升下**无系统性规律地波动**。两者的相对优劣发生反转：最弱编码器（ResNet-18）下 OAT 领先 DP 达 17.4pp,最强编码器（DINOv2）下 DP 反超 OAT 达 12.8pp——接近 30pp 的逆转,说明离散/连续表示的相对优势并非固定,而取决于上游编码器质量。同时 DP 在最强编码器下也只在 60 出头见顶,提示 codebook 瓶颈消除后编码器本身（其为语义理解/自监督目标而非操作任务设计）会成为下一个绑定约束。

**表 3：codebook 容量扫描（M 规模,LIBERO-10,DP 参照值：ResNet 36.4% / SigLIP 57.6%）：**

| \|V\| | FSQ levels | bits/chunk | ResNet-18 | SigLIP | Δenc |
|---|---|---|---|---|---|
| 1000 | [8,5,5,5] | ~80 | 53.8 | 57.4 | +3.6 |
| 1920 | [8,8,6,5] | ~87 | 42.6 | 57.8 | **+15.2** |
| 4375 | [7,5,5,5,5] | ~97 | 54.6 | 58.6 | +4.0 |

|V|=1920 时出现显著不对称：SigLIP 维持在 57.8%,而 ResNet-18 骤降到 42.6%（逼近 DP 的 ResNet 基线 36.4%）,codebook 部分放松后各编码器的真实质量差异开始显现,为"codebook 是介导变量"提供了因果证据；|V|=4375 时该模式未继续单调,作者将其归因于 OAT 论文本身指出的可建模性-容量权衡。

## 四、评价与展望

**优点：** 用数据处理不等式给出一个可证伪的预测（$\Delta_{\text{enc}}>0$ vs $\approx0$）,再用三条独立证据线（因子实验、连续编码器质量梯度、codebook 容量因果扫描）交叉验证,证据链完整；复用 OAT 官方代码库与共享的策略骨干架构,把动作表示以外的混淆变量（架构、代码实现、训练协议）尽量控制住;发现的"编码器质量低时离散占优、质量高时连续占优、且存在交叉点"的现象具有实际指导意义——对于持续投入升级视觉基座的 VLA 工作（如采用 FAST/RT-2 风格离散化的路线）,该结果提示固定容量的动作 codebook 可能让编码器端的投入部分失效。论文与 [Tong et al., 2026] 关于"VAE 表征饱和、语义编码器（RAE/SigLIP 2）持续随算力提升"的发现相呼应,把这一结构性不对称从多模态预训练规模化延伸到了视觉-动作接口,为"任何有损压缩阶段都可能成为扩展瓶颈"提供了一个可迁移的分析框架。

**局限与开放问题：** 实验仅在单一基准（LIBERO-10,10 任务、每任务 50 条示教）、仿真环境、两个模型规模下完成,论文自陈缺少真实机器人验证；成功率只取训练过程中的峰值单次结果,未做多随机种子的方差评估,codebook 4375 时的非单调"回升"是否为噪声难以排除；离散侧只穷尽验证了 OAT 一种 tokenizer,FAST、逐维度 binning、VQ-BeT 是否表现出同样的压缩缺口仅是推测、未经实证；80/87/97 bits 的容量扫描通过改变 FSQ 各维度 level 组合实现,并非严格的单一容量轴控制,长度层级的具体分配是否引入额外混淆也值得进一步厘清。DP 本身在最强编码器下仅在 60% 区间见顶,提出的"下一步开发面向操作任务(接触几何、物体可供性、空间动力学)而非纯语义理解的专用视觉编码器"方向,以及自适应/可扩容 codebook、离散-连续混合解码等,都还停留在建议层面,尚待后续工作验证是否能在保留离散 token 与预训练 LLM 统一前缀解码优势的同时缩小该缺口。

## 参考

- Liu et al., "OAT: Ordered Action Tokenization", arXiv:2602.04215, 2026
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", IJRR 44(10-11):1684–1704, 2025
- Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", arXiv:2410.24164, 2024
- Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models", arXiv:2501.09747, 2025
- Tong et al., "Beyond Language Modeling: An Exploration of Multimodal Pretraining", arXiv:2603.03276, 2026
