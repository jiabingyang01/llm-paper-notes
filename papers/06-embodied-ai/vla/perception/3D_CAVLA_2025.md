# 3D-CAVLA：深度与 3D 上下文增强 VLA 零样本泛化

> **论文**：*3D-CAVLA: Leveraging Depth and 3D Context to Generalize Vision–Language Action Models for Unseen Tasks*
>
> **作者**：Vineet Bhat, Yu-Hsiang Lan, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami
>
> **机构**：New York University (NYU)
>
> **发布时间**：2025年05月
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.05800) | [项目主页](https://3d-cavla.github.io)
>
> **会议**：CVPR 2025 Workshop
>
> **分类标签**：`VLA` `3D 深度感知` `Chain-of-Thought` `ROI 检测` `零样本泛化` `LIBERO` `OpenVLA-OFT`

---

## 一句话总结

在 OpenVLA-OFT 基础上集成 **Chain-of-Thought 叙事指令**、**PointNet 启发的轻量深度编码器**（~1M 参数）和**任务感知 ROI 检测**三个模块，将 VLA 输入从 2D 提升到 3D，LIBERO 四套任务 in-distribution 平均成功率达 **98.1%**，自行设计的 LIBERO-Unseen 10 个零样本新任务上比 OpenVLA-OFT **绝对提升 8.8%**。

---

## 一、问题与动机

### 1.1 VLA 对 unseen 任务泛化差

当前 VLA 在 in-distribution 任务上已接近饱和（OpenVLA-OFT 达 97.1%），但跨任务泛化极差。论文实验发现：将 LIBERO-Object 微调的模型直接用于 LIBERO-Goal，**所有基线方法成功率归零**。这说明 VLA 严重过拟合训练任务。

### 1.2 缺乏中间推理和空间感知

现有 VLA 的两大缺陷：
1. **直接输入-输出映射**：语言指令直接映射为关节动作，缺少中间推理步骤。对 unseen 任务无法分解为已学子技能
2. **缺乏深度感知**：绝大多数 VLA 仅使用 RGB 图像，无法精准理解物体的 3D 形状、大小和空间位置关系，影响精确操作

### 1.3 解决思路：三管齐下

| 模块 | 解决的问题 | 核心思路 |
| --- | --- | --- |
| **CoT 叙事指令** | 缺乏中间推理 | GPT-4 将指令分解为可执行子步骤 |
| **深度编码器** | 缺乏 3D 空间感知 | 点云编码器提取深度特征 |
| **任务感知 ROI** | 视觉特征冗余 | 实体检测 + 跟踪生成运动区域掩码 |

---

## 二、预备知识：OpenVLA-OFT

3D-CAVLA 基于 OpenVLA-OFT 构建。OpenVLA-OFT 的关键特性：

- **视觉编码器**：SigLIP + DinoV2 双编码器，分别处理末端执行器相机和第三人称相机图像
- **LLM 骨干**：LLaMA 2 7B，LoRA 高效微调
- **FiLM 层**：可选的 Feature-wise Linear Modulation，增强视觉-语言特征提取
- **本体感受**：8 维关节 + 夹爪状态，通过 MLP 编码
- **三大改进**：并行解码（非自回归）、动作分块（联合预测 $K$ 步）、连续输出 + $\ell_1$ 损失

所有模态的 embedding 投影到 LLaMA 2 的输入维度后拼接输入 LLM。

---

## 三、核心方法

### 3.1 Chain-of-Thought 叙事指令

**动机**：一个学会 "Grab the ball and place it in the basket" 的机器人，面对 "Move the orange into the basket" 时，如果指令被分解为子步骤——"Locate the orange, grab it from the center, move over basket, drop inside basket"——则除了定位目标物体外，其余子步骤都可复用已学技能。

**实现**：使用 GPT-4 离线将每条训练指令转换为 CoT 步骤（few-shot prompting）：

> **Prompt 格式**：
> 1. 系统角色：机器人操作员
> 2. Few-shot 示例："Put both pots on the stove" → "Grasp first pot, place on stove leaving some space, grasp second pot, place on stove next to first pot"
> 3. 约束：仅使用给定物体和平行夹爪可执行的标准动作

**关键点**：CoT 步骤在训练前离线计算，不增加推理延迟。

### 3.2 深度编码器

**深度图 → 点云**：给定深度图 $D \in \mathbb{R}^{B \times H \times W}$ 和相机内参 $(f_x, f_y, c_x, c_y)$，通过反投影恢复每个像素的 3D 坐标：

$$Z_{b,h,w} = D_{b,h,w}$$

$$X_{b,h,w} = \frac{U_{h,w} - c_x}{f_x} \cdot Z_{b,h,w}$$

$$Y_{b,h,w} = \frac{V_{h,w} - c_y}{f_y} \cdot Z_{b,h,w}$$

堆叠 $(X, Y, Z)$ 得到点云 $P \in \mathbb{R}^{B \times H \times W \times 3}$。

**PointNet 启发的编码器架构**：

> 1. **空间变换网络（STN）**：MLP 层将点云变换为空间不变表征
> 2. **残差批矩阵乘**：学习几何变换
> 3. **特征提取**：3 个 Conv2D + BatchNorm + ReLU 块
> 4. **投影层**：线性映射到 LLaMA 2 的输入维度（4096）

输出深度 embedding $\in \mathbb{R}^{B \times 4096}$，与视觉、语言、本体感受 token 拼接输入 LLM。

**设计选择**：
- 每个相机视角独立编码器（参数不共享），总参数量约 **1M**
- 端到端训练，非冻结

### 3.3 任务感知 ROI 检测

**动机**：VLA 视觉 embedding 包含所有图像 patch 的表征，但大部分 patch 对当前任务无关。对 unseen 任务，大量 OOD 物体的存在会严重干扰策略。

**流水线**：

> 1. **实体识别**：对任务指令做命名实体识别（GLiNER），提取目标物体和目标位置
> 2. **目标检测**：用 Molmo 对首帧 RGB 图像生成实体边界框
> 3. **运动追踪**：用 SAMURAI 在 GT 演示视频中追踪实体边界框的运动范围
> 4. **区域掩码**：生成二值掩码，标记任务相关的运动区域，用于池化视觉特征

**防过拟合策略**：训练时仅 **25%** 的概率使用 ROI pooling，其余 75% 使用完整视觉特征。原因是 ROI 掩码会排除背景上下文和障碍物信息，对 in-distribution 任务有轻微负面影响（见消融实验），但对 OOD 泛化有帮助。

---

## 四、实验结果

### 4.1 LIBERO In-Distribution 结果

每个任务 50 条演示训练、50 次试验评估。

**单相机设置（第三人称 + 语言指令）**：

| 方法 | Spatial | Object | Goal | Long | Average |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| Diffusion Transformers | 84.2 | **96.3** | **85.4** | 63.8 | 82.4 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| **3D-CAVLA** | **86.1** | 94.7 | 82.9 | **66.8** | **82.6** |

**双相机设置（第三人称 + 腕部 + 关节状态 + 语言指令）**：

| 方法 | Spatial | Object | Goal | Long | Average |
| --- | --- | --- | --- | --- | --- |
| Multimodal DiT | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **3D-CAVLA** | **98.2** | **99.8** | **98.2** | **96.1** | **98.1** |

**关键发现**：Long-horizon 任务提升最大（94.5 → 96.1），CoT 指令帮助策略逐步聚焦子任务。

### 4.2 消融实验

| 方法 | Spatial | Object | Goal | Long |
| --- | --- | --- | --- | --- |
| **3D-CAVLA** | **98.2** | **99.8** | **98.2** | **96.1** |
| w/o CoT | 97.8 | 99.4 | 97.9 | 94.8 |
| w/o Depth | 97.6 | 99.0 | 98.0 | 95.2 |
| w/ TA-ROI | 98.0 | 99.4 | 97.4 | 94.2 |

关键结论：
- **深度编码器**贡献最大：移除后全面下降，Object 降 0.8%、Long 降 0.9%
- **CoT 指令**在 Long-horizon 收益明显：移除后 Long 降 1.3%
- **TA-ROI 在 seen 任务上有轻微负面影响**：Goal 降 0.8%、Long 降 1.9%。原因是 ROI 掩码排除了任务所需的背景上下文（如抽屉本体被遮蔽导致策略无法定位）

### 4.3 Zero-Shot 泛化（LIBERO-Unseen）

在 LIBERO-90 上微调后测试 10 个全新任务（每任务 50 次试验）。任务设计原则：指令新颖，但物体和子技能在训练数据中出现过。

| 任务 | OpenVLA-OFT | 3D-CAVLA |
| --- | --- | --- |
| Place the white and yellow mug on the plate | 32 | **60** |
| Put the ketchup on top of the cabinet | 74 | **82** |
| Pick up the chocolate pudding and put in top drawer | **58** | 52 |
| Stack right bowl on left bowl + put pudding in tray | 0 | 0 |
| Put the chocolate pudding on the plate | 78 | **80** |
| Place cream cheese and soup inside the basket | 66 | **74** |
| Grab white bowl and keep it on the stove | **12** | 10 |
| Grab pudding, place on bowl, then both on tray | 6 | **24** |
| Turn on the stove and put the bowl on it | 14 | **38** |
| Place the mug inside right compartment of caddy | 24 | **32** |
| **Average** | **36.4** | **45.2 (+8.8)** |

**关键观察**：
- **两者都远低于 seen 任务表现**：最高 98.1% vs. 最高 82%，揭示 VLA 严重过拟合
- 3D-CAVLA 在 8/10 任务上优于或持平 OpenVLA-OFT
- 多步组合任务最难（stacking 两者均 0%），完全新颖的运动无法零样本迁移
- 单相机设置下两者均全部失败，说明双相机 + 本体感受对泛化至关重要

---

## 五、局限性与未来方向

1. **任务设计偏温和**：unseen 任务中所有物体和子技能都出现在训练数据中，仅组合新颖。面对完全新物体/新运动，两种方法均失败
2. **仿真局限**：LIBERO 任务相对简单且容易饱和，缺乏真实世界验证
3. **ROI 模块局限**：依赖 NER + Molmo + SAMURAI 三级流水线，误差可能累积；ROI 掩码对 seen 任务有负面影响，需手动调节使用概率（25%）
4. **深度编码器简单**：~1M 参数的 PointNet 变体可能无法捕捉复杂几何。论文建议探索 3D mesh 等更精细的深度表征
5. **CoT 依赖 GPT-4**：子步骤质量受限于 LLM 分解能力，且增加离线处理成本

**作者未来方向**：
- 增加 VLM 引导的闭环反馈模块，推理时实时提供环境线索纠正错误动作
- 在真实世界进行大规模实验和 zero-shot 部署验证

---

## 六、个人思考

### 6.1 显式 3D vs. 隐式 3D 的路线之争

3D-CAVLA 是**显式 3D 输入**方案的代表——直接将深度图/点云编码为 token 送入 LLM。有趣的是，同期的 SF（Spatial Forcing）走了另一条路——**隐式 3D 对齐**，训练时用 VGGT 监督 VLA 中间层表征，推理时完全无开销。两者在 LIBERO 上性能接近：

| 方法 | 3D 方案 | Spatial | Object | Goal | Long | Average | 推理开销 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3D-CAVLA | 显式输入 | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 | +深度编码 |
| SF | 隐式对齐 | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 | 零 |

SF 在 Average 上还略高 0.4%，且**推理无额外开销**。这暗示**VLA 的空间理解瓶颈可能不在输入信号缺失，而在表征空间的 3D 结构缺失**。

### 6.2 CoT 分解的可组合性假设

3D-CAVLA 的 CoT 模块基于一个隐含假设：**unseen 任务可以被分解为 seen 子技能的新组合**。这在温和的 unseen 设置下成立（所有物体出现过），但面对真正新颖的任务时会失效。这一点与 π₀.₅ 的分层推理（高层子任务预测 + 低层控制）思路接近，但 π₀.₅ 在训练时就学习子任务分解，而 3D-CAVLA 依赖外部 LLM 离线分解，泛化能力更受限。

### 6.3 ROI 模块的双刃剑

ROI 检测对 seen/unseen 任务的效果截然相反是一个值得关注的发现：
- **Seen 任务**：ROI 遮蔽背景上下文，反而丢失必要信息（如抽屉本体位置）
- **Unseen 任务**：ROI 过滤 OOD 干扰物，帮助策略聚焦操作区域

这与 VLM 领域的注意力干预方法（如 VisFlow）有异曲同工之处——选择性屏蔽不相关区域可以提升模型对关键区域的注意力，但过度屏蔽会损害全局理解。

### 6.4 Zero-Shot 评估的意义

论文最大的贡献可能不是方法本身（三个模块都相对 incremental），而是**系统性地暴露了 VLA 的零样本泛化缺陷**：
- 所有基线在跨 suite 评估时成功率归零
- 即使在温和的 unseen 设置下，最优方法也只有 45.2%
- LIBERO-Unseen 基准的公开发布有助于社区量化泛化进展

这与 VLA RL 后训练领域的动机高度一致——SFT 训练的 VLA 倾向于记忆演示而非学习可迁移技能，需要 RL 或其他机制来突破泛化瓶颈。

---

## 参考

- **OpenVLA-OFT**（Kim et al., 2025）：3D-CAVLA 的基础架构，并行解码 + 动作分块 + 连续输出的高效 VLA
- **OpenVLA**（Kim et al., 2024）：首个开源 7B VLA，使用 Open-X Embodiment 数据预训练
- **π₀**（Black et al., 2024）：Flow Matching VLA 基础模型，LIBERO 基准对比
- **PointNet**（Qi et al., CVPR 2017）：3D-CAVLA 深度编码器的设计灵感来源，点集上的深度学习开创性工作
- **Molmo**（Deitke et al., 2024）：ROI 检测流水线中的目标检测器
- **SAMURAI**（Yang et al., 2024）：ROI 检测流水线中的零样本视觉跟踪器
- **GLiNER**（Zaratiana et al., NAACL 2024）：ROI 流水线中的命名实体识别模型
- **SF**（Li et al., ICLR 2026）：隐式 3D 对齐的 VLA 方案，推理零开销达到类似性能——3D-CAVLA 的重要对比参照
