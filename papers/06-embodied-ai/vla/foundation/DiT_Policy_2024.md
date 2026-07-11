# DiT Policy：扩散 Transformer 策略

> **论文**：*Diffusion Transformer Policy*
>
> **作者**：Zhi Hou, Tianyi Zhang, Yuwen Xiong, Hengjun Pu, Chengyang Zhao, Ronglei Tong, Yu Qiao, Jifeng Dai, Yuntao Chen
>
> **机构**：Shanghai AI Lab；浙江大学；香港中文大学 MMLab；北京大学；商汤科技研究院；清华大学；中国科学院香港创新研究院人工智能与机器人中心（HKISI, CAS）
>
> **发布时间**：2024 年 10 月（arXiv 2410.15959）
>
> **发表状态**：未录用（预印本，PDF 中未标注接收会议信息）
>
> 🔗 [arXiv](https://arxiv.org/abs/2410.15959) | [PDF](https://arxiv.org/pdf/2410.15959)
>
> **分类标签**：`扩散策略` `Diffusion Transformer` `VLA 预训练` `动作分块` `Open X-Embodiment` `跨具身泛化`

---

## 一句话总结

用一个从零训练、334M 参数（221M 可训练）的因果 Transformer 直接作为扩散去噪网络对动作 chunk 进行 in-context 条件去噪（图像 patch token、语言 token、时间步 token 与带噪动作 token 拼接成同一序列输入），取代以往"小 MLP/UNet 扩散头挂在冻结 Transformer 骨干后面"的做法，在 CALVIN ABC→D 仅用单目标 RGB 观测下把平均连续完成任务数从约 2.4 做到 3.61（相对自实现的 diffusion head 基线提升 0.45，相对无预训练版本提升 1.23），并在 SimplerEnv、LIBERO、Maniskill2 与真实 Franka 臂上全面超过 OpenVLA-7B 与 Octo。

## 一、问题与动机

当前的通用机器人策略（Robot Transformer 系列、OpenVLA、Octo 等）大多采用"大骨干 + 小动作头"范式：RT-1/RT-2、OpenVLA 把每个动作维度离散化成 256 个 bin 用交叉熵学习，这种离散化策略在真实执行中会引入内在偏差；Octo 则用一个以 Transformer 输出 embedding 为条件的小型 MLP 扩散头分别去噪单个连续动作 token。作者指出两个问题：

1. **离散动作头**限制了对连续动作空间的精细建模，离散化误差会传导到执行精度上。
2. **小 MLP/浅交叉注意力扩散头**先把历史图像观测和指令"融合压缩成一个 embedding"再去噪，这种早期信息压缩会限制动作预测对历史观测细节（例如动作 delta、位置微小偏移）的感知能力，而扩散去噪学习往往依赖对历史观测的直接、细粒度关注。

同时，3D Diffuser Actor、3D Diffusion Policy 等工作虽用扩散建模动作但依赖点云输入，难以直接复用大规模 2D 跨具身数据集（Open X-Embodiment）。因此作者提出：能否设计一个**大 Transformer 本身就是去噪网络**、能直接在图像 patch/语言 token 上做 in-context 条件、同时保留 Transformer 可扩展性的扩散策略架构，用于在大规模跨具身数据（Open X-Embodiment）上预训练。

## 二、核心方法

### 2.1 整体架构

DiT Policy 由四个模块组成（对应论文 Figure 3）：

- **语言编码**：冻结的 CLIP 文本编码器把指令编码为文本 token（训练中始终冻结，是模型里唯一不参与训练的部分）。
- **图像编码**：图像先 resize 到 224×224，输入 DINOv2（base 版 ViT）得到 patch 特征；与常见做法不同，DINOv2 参数**联合训练**（不冻结），因为其在网络图像上预训练，与机器人图像分布存在差异。
- **Q-Former 压缩**：为降低计算量，用一个从零训练、深度为 4 的 Q-Former（结合 FiLM 条件注入语言信息）把每帧图像的 patch 特征压缩为固定 32 个 query token。
- **动作预处理**：末端执行器动作表示为 7 维向量（3 维平移 + 3 维旋转向量 + 1 维夹爪开合），为了和图像/语言 token 对齐维度，用零填充连续动作向量；去噪噪声只加在这 7 维有效动作分量上。

**核心设计（区别于 Octo 等）**：不是把带噪动作单独喂给一个小动作头，而是把「图像 patch 特征、语言 token、时间步 embedding、带噪动作 chunk token」**直接拼接成同一个序列**，送入一个从零训练的因果 Transformer（Llama2-2 类型结构，12 层自注意力，hidden size 768），由这个大 Transformer 本身承担去噪任务——即整个因果 Transformer 就是去噪网络 $\epsilon_\theta$，以 in-context 条件的方式对动作 chunk 去噪，而非以小网络对单个动作 embedding 分别去噪。

### 2.2 训练目标（DDPM）

去噪网络记为 $\epsilon_\theta(\boldsymbol{x}^t, c_{obs}, c_{instru}, t)$，其中 $c_{obs}$ 为图像观测条件、$c_{instru}$ 为语言指令条件、$t \in \{1,\dots,T\}$ 为扩散步索引。训练时从高斯分布采样噪声 $\boldsymbol{x}^t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，加到真实动作 $\hat{\boldsymbol{a}}$ 上构成带噪动作 token，网络预测所加噪声 $\hat{\boldsymbol{x}}$，用 MSE loss 优化。

用大白话说：这一步和标准 Diffusion Policy 完全一样——网络学的不是动作本身，而是"这坨带噪动作里混进去的噪声长什么样"，训练目标就是让预测噪声尽量贴近真实加的高斯噪声。

推理阶段执行 $T$ 步去噪：

$$
\boldsymbol{x}^{t-1} = \alpha\big(\boldsymbol{x}^t - \gamma \epsilon_\theta(\boldsymbol{x}^t, c_{obs}, c_{instru}, t)\big) + \mathcal{N}(\boldsymbol{0}, \sigma^2\boldsymbol{I})
$$

其中 $\alpha, \gamma, \sigma$ 由噪声调度器给出（沿用 DDPM 惯例）。

用大白话说：从纯高斯噪声出发，反复调用同一个大 Transformer 猜"这里面的噪声成分是什么"，一步步把噪声减掉、再加一点点新的随机扰动，直到收敛成一段干净的动作序列。这一步是标准 DDPM 采样公式，但特殊之处在于——每一步调用的都是同一个承担了图文理解任务的大因果 Transformer，而不是一个独立于骨干的小去噪头。

### 2.3 预训练设置

在 Open X-Embodiment（OXE）数据集上预训练，主要跟随 OpenVLA/Octo 的数据集选择和权重配比，共选取 15 个子数据集（Fractal、Droid、Kuka、BridgeV2 权重最高，分别约 16.15%/13.69%/17.47%/21.86%，详见论文 Table 12），对动作做归一化并过滤离群动作。预训练用 DDPM 目标、$T=1000$ 步；零样本评测时用 DDIM 以 $T=100$ 步加速推理。根据 Maniskill2 上的预实验，选用 2 帧历史观测图像、预测 32 步动作 chunk。用 AdamW 训练 100,000 步，因果 Transformer 与 Q-Former 学习率 1e-4、DINOv2 学习率 1e-5，batch size 8902。模型总参数量 334M，其中可训练参数 221M（CLIP 文本编码器保持冻结）——相比 OpenVLA-7B 体量小一个数量级。

## 三、实验结果

在 SimplerEnv（Real-to-Sim，Google Robot）、CALVIN（ABC→D 长时序语言条件任务）、LIBERO（微调泛化）、Maniskill2（新视角泛化，从零训练）与真实 Franka 臂（预训练零样本/10-shot 微调/域内微调）五个 benchmark 上做了系统评测。

**SimplerEnv（Real-to-Sim，成功率 %，match / variant）**

| 方法 | coke_can | move_near | drawer |
|---|---|---|---|
| RT-1-X | 56.7 / 49.0 | 31.7 / 32.3 | 59.7 / 29.4 |
| Octo-Base | 17.0 / 0.6 | 4.2 / 3.1 | 22.7 / 1.1 |
| OpenVLA-7B | 16.3 / 54.5 | 46.2 / 47.7 | 35.6 / 17.7 |
| **DiT Policy（ours）** | **72.7 / 60.0** | **56.7 / 57.5** | 46.3 / 37.5 |

**CALVIN ABC→D（连续完成 1–5 个子任务成功率 %，Avg.Len 满分 5）**

| 方法 | 1 | 2 | 3 | 4 | 5 | Avg.Len |
|---|---|---|---|---|---|---|
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| 3D Diffuser Actor | 92.2 | 78.7 | 63.9 | 51.2 | 41.2 | 3.27 |
| diffusion head（自实现基线，无预训练） | 75.5 | 44.8 | 25.0 | 15.0 | 7.5 | 1.68 |
| diffusion head（自实现基线，预训练） | 94.3 | 77.5 | 62.0 | 48.3 | 34.0 | 3.16 |
| DiT Policy w/o pretrain | 89.5 | 63.3 | 39.8 | 27.3 | 18.5 | 2.38 |
| **DiT Policy（ours）** | **94.5** | **82.5** | **72.8** | **61.3** | **50.0** | **3.61** |

预训练使 DiT Policy 的 Avg.Len 提升 1.23（2.38→3.61）；在同样加载了预训练权重的前提下，DiT Policy 比自实现的 diffusion-head 基线高 0.45（3.16→3.61）。

**LIBERO（微调，成功率 %）**

| 方法 | SPATIAL | OBJECT | GOAL | LONG | Average |
|---|---|---|---|---|---|
| Diffusion Policy (from scratch) | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo fine-tuned | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA fine-tuned | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| **DiT Policy fine-tuned（ours）** | 84.2 | **96.3** | **85.4** | **63.8** | **82.4** |

在长时序任务 LIBERO-LONG 上领先最明显（63.8 vs OpenVLA 53.7），平均成功率比 OpenVLA 高约 6 个百分点。

**Maniskill2（5 任务，从零训练，成功率 %）**

| 方法 | All | PickCube | StackCube | SingleYCB | ClutterYCB | SingleEGAD |
|---|---|---|---|---|---|---|
| Disc ActionHead（RT-1 式离散化） | 30.2 | 41.0 | 33.0 | 22.0 | 1.0 | 54.0 |
| Diff ActionHead（Octo 式小 MLP 扩散头） | 58.6 | 86.0 | 76.0 | 37.0 | 24.0 | 70.0 |
| **DiT Policy（ours）** | **65.8** | 79.0 | 80.0 | 62.0 | 36.0 | 72.0 |

在更复杂、需要区分同类多个候选物体的 PickClutterYCB 与 PickSingleYCB 上优势最大。

**真实 Franka 臂（10-shot 微调，成功率 %，节选）**：PickBlock 上 DiT 14.8% vs Octo/OpenVLA 均 7.4%；Pick&Insert 两步任务 Step-1/Step-2 分别为 90%/10%（OpenVLA 80%/0%，Octo 20%/0%）；域内少样本微调（Table 6）总体成功率 DiT 46.9% vs 离散动作头 19.3%、Octo 式扩散头 34.8%，在未见过的新物体（tiny block / oval block）上 DiT 仍有 37.0%/7.4% 成功率，而两个基线均为 0%。

**消融要点**：(1) 动作 chunk 长度越长整体性能越好，复杂任务（PickClutterYCB）随 chunk 变长提升明显，简单任务（PickCube）在长度 >4 后趋于饱和；(2) 历史观测帧数并非越多越好——增到 3 帧反而因图像 token 增多导致收敛变难，在长 chunk（32）场景下 2 帧观测最优；(3) 执行步数（execution horizon）越短、重新规划越频繁，预测质量略优（1 步 61.6% > 16 步 58.0%）；(4) 收敛速度上 DiT Policy 明显快于同参数量的 diffusion-head 基线（Figure 7）；(5) 换用 Unet1D/Transformer/更深 MLP 等多种动作头设计（Table 11，均不预训练），DiT Policy 的 in-context 整体去噪范式仍全面领先（Avg.Len 2.38 vs 至多 1.80）。

## 四、局限性

论文在结论部分明确指出：DiT Policy 需要在推理阶段执行多步去噪（DDPM/DDIM 采样），这会拖慢真实部署时的推理速度；作者认为可以通过引入少步去噪的微调策略来加速推理，但本文未对此展开实验，留作未来工作。此外，从实验设计看还存在以下未被论文充分讨论的局限：真实机器人实验规模较小（Franka 单臂、单一第三人称相机、9 宫格摆放），跨本体（如双臂、移动底盘）泛化能力未验证；历史观测长度的消融显示模型对观测 token 数量较敏感，提示其对更长历史/更多相机视角的可扩展性仍需验证。

## 五、评价与展望

**优点**：DiT Policy 的核心贡献是把"扩散头"这一独立于骨干的附属模块，直接并入大 Transformer 主干本身，用 in-context 拼接条件的方式统一图像、语言、时间步、动作四类 token，这一设计思路上更接近 Peebles & Xie 的 DiT（Diffusion Transformer，用于图像生成）向机器人动作序列建模的自然迁移，同时保留了 Transformer 在跨具身大数据集上的可扩展性。在仅 334M 参数（不到 OpenVLA-7B 的 1/20）的规模下，取得了对 OpenVLA、Octo 全面持平或超越的效果，尤其在 CALVIN 长时序任务和 Maniskill2 复杂抓取任务上优势明显，说明"用大模型直接建模连续动作分布"比"离散化+分类"或"小容量条件去噪头"更适合刻画精细的连续动作序列。收敛速度和消融实验（Table 10/11）也从工程角度验证了该设计相对于同参数量 MLP/Unet 扩散头的有效性。

**与其他公开工作的关系**：相较 Octo（小 MLP 扩散头，条件于单一压缩 embedding），DiT Policy 把条件信息从"压缩后的单一 embedding"改为"未压缩的图像 patch/文本 token 序列"，使去噪过程可以直接关注历史观测细节；相较 OpenVLA/RT-1/RT-2 的离散化动作头，DiT Policy 保留了连续动作空间，避免离散化带来的执行偏差；相较 3D Diffuser Actor / 3D Diffusion Policy 等依赖点云的扩散策略，DiT Policy 沿用 2D 图像输入，因而能直接复用 Open X-Embodiment 这类大规模 2D 跨具身数据。

**开放问题与可能的改进方向**：(1) 推理速度是明确留下的开放问题，少步/一致性蒸馏（如同期工作 One-step Diffusion Policy）是自然的后续方向；(2) 论文尚未探索动作空间之外的其他模态输出（如力/触觉），以及是否可以把该 in-context 扩散范式扩展到双臂或移动操作等更复杂的动作空间；(3) Q-Former 将每帧图像压缩到固定 32 个 token 虽然降低了计算量，但也是一个人为设定的信息瓶颈，其压缩率对下游任务性能的影响未做系统消融；(4) 真实机器人实验样本量和任务多样性有限，该架构在更大规模真实部署、更长 horizon 任务下的鲁棒性仍待更充分验证。

## 参考

1. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246, 2024.
2. Octo Model Team et al. *Octo: An Open-Source Generalist Robot Policy.* arXiv:2405.12213, 2024.
3. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale.* arXiv:2212.06817, 2022.
4. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023 / arXiv:2303.04137.
5. Peebles & Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023.
