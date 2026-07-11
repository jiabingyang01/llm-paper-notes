# OmniSAT：紧凑动作Token，更快自回归

> **论文**：*OmniSAT: Compact Action Token, Faster Auto Regression*
>
> **作者**：Huaihai Lyu, Chaofan Chen, Senwei Xie, Pengwei Wang, Xiansheng Chen, Shanghang Zhang, Changsheng Xu
>
> **机构**：中国科学院自动化研究所、中国科学院大学、中国科学院计算技术研究所、北京智源人工智能研究院（BAAI）、北京大学、鹏城实验室
>
> **发布时间**：2025 年 10 月（arXiv 2510.09667）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.09667) | [PDF](https://arxiv.org/pdf/2510.09667)
>
> **分类标签**：`动作token化` `自回归VLA` `残差向量量化` `跨具身学习` `B样条编码`

---

## 一句话总结

OmniSAT 用两阶段"一致性编码（B 样条拟合定长控制点）+ 分部位残差向量量化（position/rotation/gripper 各自 codebook，coarse-to-fine）"把变长连续动作轨迹压成更短、更保真的离散 token：在 DROID 上把训练序列缩短 6.8 倍、重建 MAE 降到 9.4e-4，同时压缩比达到 6.8 倍（优于 FAST 的 3.7 倍和 BEAST 的 4.6 倍），据此在 LIBERO 上取得 93.4% 平均成功率排名第一，在 SimplerEnv-WidowX 上以 55.2% 总体成功率超过 RT-1-X/Octo/RoboVLMs/BEAST/SpatialVLA 等基线，真机三任务平均成功率 61.3%（混入人类第一视角视频 EgoDex 微调后提升至 68.0%）。

## 一、问题与动机

自回归（AR）VLA 相比扩散类方法优化更高效、更容易接入异构数据源，但当训练目标从单步动作换成长时程 action chunk 后，token 序列显著变长，拖慢 AR 训练。已有的压缩方案各有短板：FAST 基于字节对编码（BPE）压缩 DCT 系数，依赖训练集与目标轨迹之间的 token 共现频率统计，域外泛化弱，且变长输出会使 batching/decoding 复杂化；BEAST 用 B 样条控制点表示轨迹，但在高压缩比下重建质量差。论文认为一个好的动作 tokenizer 应同时满足：（i）高保真嵌入动作轨迹，保留精细执行细节；（ii）提供足够的压缩，使生成模型能高效捕捉视觉-语言上下文与执行动作之间跨越长时程的对应关系。

## 二、核心方法

OmniSAT 分两步走：先"对齐"再"量化"。

**Consistency Encoding（一致性编码）**：先对每个数据集的每个 DoF 做鲁棒逐维归一化（用该维度的第 1、99 百分位映射到 [-1, 1]），获得跨具身数值范围一致的轨迹；再用 B 样条控制点把来自具身 $e$、长度为 $T_e$ 的变长轨迹 $\bar a_e$ 编码为定长（$T_c$）对齐表示，通过对基矩阵 $\Phi$ 做岭回归求解：

$$
c = \arg\min_c \|\Phi c - \bar a_e\|_F^2 + \lambda \|c\|_F^2
$$

用大白话说：先把每个关节/维度的数值域统一拉到 [-1, 1]，再用固定数量的"控制点"（类似贝塞尔曲线的手柄）去逼近整条轨迹曲线，不论原始轨迹有多少帧，压缩后都变成同样数量的控制点，方便跨具身批处理。

**Quantization Compression（量化压缩）**：把上一步得到的控制点特征 $z$ 按物理语义拆成 position/rotation/gripper 三组，每组独立做残差向量量化（Residual VQ-VAE）。第 $l$ 层从该层专属 codebook $\mathcal C^l$ 中选出最近的 codeword，并把残差传给下一层继续细化：

$$
q_l = \arg\min_{i\in[1,K]} \|r_{l-1}-\mathcal C_i^l\|_F^2, \qquad r_l = r_{l-1}-\mathcal C_{q_l}^l
$$

$L$ 层之后累加得到重建 $\hat s=\sum_{l=1}^L \mathcal C_{q_l}^l$，对应离散 token 列表 $\bar q=[q_1,\dots,q_L]$。用大白话说：像"由粗到细"逐层雕刻——第一层给出大致形状，后面每层只需修正上一层没描准的"残差"，层数越多重建越精，天然实现可控的压缩粒度；三个部位分组各用一套小 codebook（$K^{pos}=256, K^{rot}=256, K^{grip}=64$），既贴合动作的物理结构，又减小单个 codebook 的规模。

**训练目标**：重建损失同时约束特征级和轨迹级——$\mathcal L_{recon}=\|z-\hat z\|_F^2+\gamma\|a-\mathcal B(\hat z)\|_F^2$，其中 $\mathcal B(\cdot)$ 是把控制点解码回原始轨迹的 B 样条解码器；commitment 损失沿用标准 VQ-VAE 双向约束防止 codebook 坍塌；此外引入 quantizer-layer dropout——训练时每层残差以概率 $p=0.1$ 独立被随机跳过（推理时全部启用），迫使每一层单独具备一定的表达能力，避免模型过度依赖某几层导致测试集泛化差。三项损失以 EMA 加权组合成总目标。

**跨具身操作学习**：把预训练好的视觉 tokenizer（沿用 Emu3 所用的视觉 tokenizer）产生的逐帧视觉 token，与 OmniSAT 产生的逐帧动作 token 拼接成统一的视觉-动作交织序列；对人类第一视角视频数据集 EgoDex（约 30 万条 episode，200 个任务）和机器人示教数据（DROID 等）按具身混合权重 $\alpha^{(e)}$ 加权，统一做下一 token 预测训练，让 AR 骨干（论文采用 Emu3 配置，8.5B 参数）在同一动作-模式 token 空间里联合学习人类与机器人的操作模式。

## 三、关键结果

**DROID 上的压缩质量对比**（Table 1，76k 条示教轨迹，9:1 切分）：

| 方法 | MAE↓ | 压缩比 R↑ |
|---|---|---|
| FAST | $<10^{-5}$ | 3.7× |
| BEAST | 8.0e-2 | 4.6× |
| OmniSAT (L=10) | 8.5e-4 | 4.9× |
| OmniSAT (L=8, 默认) | 9.4e-4 | 6.8× |
| OmniSAT (L=6) | 1.3e-3 | 8.1× |

**LIBERO**（4 套件 × 10 任务，每任务 50 条示教，Table 2）：

| 方法 | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| BEAST | 92.9 | 97.5 | 93.1 | **86.4** | 92.5 |
| OmniSAT | 94.1 | **98.7** | **94.6** | 86.0 | **93.4**（第一） |

**SimplerEnv-WidowX**（Success 列，Table 3）：

| 方法 | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Overall |
|---|---|---|---|---|---|
| RT-1-X | 0.0 | 4.2 | 0.0 | 0.0 | 1.1 |
| Octo-Base | 12.5 | 8.3 | 0.0 | 43.1 | 16.0 |
| BEAST | 41.7 | 25.0 | 20.8 | 75.0 | 37.5 |
| SpatialVLA | 16.7 | 25.0 | 29.2 | **100** | 42.7 |
| OmniSAT | **58.3** | **37.5** | 29.2 | 95.8 | **55.2**（第一） |

**真机三任务**（自建 AgileX 双臂平台：PlaceObj 指令定位取放、ZipSeal 拉链密封袋、TubeRack 试管插架，Figure 5）：

| 方法 | PlaceObj | ZipSeal | TubeRack | Average |
|---|---|---|---|---|
| Pi-FAST | 38% | 18% | 38% | 31.3% |
| BEAST | 63% | 45% | 23% | 43.7% |
| OmniSAT | 73% | 63% | 48% | 61.3% |
| OmniSAT-M（混入 EgoDex 人类视频，4:1 机器人:人类） | 80% | 66% | 58% | 68.0%（+6.7pt） |

**消融**（Table 4/5）：去掉 commitment 损失导致 codebook 坍塌，Object 由 98.7 暴跌至 38.7、Long 由 86.0 跌至 26.4；去掉 Consistency Encoding 主要伤害长时程任务（Long 86.0→83.2）；去掉残差量化（仅保留一致性编码而不做 RVQ 压缩）主要伤害 Object/Goal（98.7/94.6→97.5/93.1），说明两阶段分别贡献长时程稳定性与精细执行两种不同能力。骨干从 Florence-2-Large（0.77B）换成 Emu3-Base（8.5B）后真机平均成功率由 51.7% 升至 59.3%；在 AR 损失中加入视觉 token 预测监督使 PlaceObj 从 60%→73%、TubeRack 从 33%→48%，但接触密集的 ZipSeal 因更依赖外观线索而非控制精度略降（66%→57%）。

## 四、评价与展望

OmniSAT 的核心贡献在于把"变长轨迹→定长表征"（B 样条一致性编码）和"定长表征→高保真离散码"（分部位残差向量量化）解耦为两个正交子问题分别求解，思路清晰。相比 FAST 依赖训练/目标域 token 共现统计、变长输出难以批处理，以及 BEAST 只用 B 样条控制点而无残差量化、高压缩比下重建质量差，OmniSAT 在 DROID 上同时取得更高压缩比（6.8×/8.1× vs 3.7×/4.6×）与更低重建 MAE，说明"先对齐再残差量化"确实推出了更好的压缩-保真度帕累托前沿。跨具身部分把人类第一视角视频（EgoDex）和机器人数据纳入同一动作-token 空间联合训练，是将 VQ-BeT 一类动作离散化思路与大规模异构预训练结合的自然延伸，真机上 +6.7pt 的提升也初步验证了该思路的可行性。

局限与开放问题：（1）压缩-保真度权衡的系统评估只在 DROID 单一数据集上完成，未报告跨数据集/跨具身场景下的 codebook 利用率、坍塌率等更细粒度诊断；（2）跨具身学习的收益主要体现在真机三任务，论文并未报告混入人类视频后在 LIBERO/SimplerEnv 等纯仿真基准上的效果，人类视频对视觉-语言泛化能力的贡献边界尚不清楚；（3）控制点数 $T_c$、残差层数 $L$、各分组 codebook 大小等超参是特定任务下网格搜索得到的默认值，论文未讨论这些超参随 action horizon、自由度数变化的可迁移规律，迁移到更高自由度具身（如人形机器人）是否仍然适用有待验证；（4）论文的比较对象局限于 AR 阵营内部（FAST、BEAST），未在同一评测协议下与主流扩散类 VLA 做直接对比，AR 与扩散在高精度接触式操作上的优劣仍是开放问题。

## 参考

- Pertsch et al. FAST: Efficient Action Tokenization for Vision-Language-Action Models. 2025.
- Zhou et al. BEAST: Behavior Generation via B-spline Action Tokenization. 2025.
- Lee et al. VQ-BeT: Behavior Generation with Latent Actions. 2024.
- Khazatsky et al. DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset. 2024.
- Wang et al. Emu3: Next-Token Prediction is All You Need. 2024.
- Hoque et al. EgoDex. 2025.
