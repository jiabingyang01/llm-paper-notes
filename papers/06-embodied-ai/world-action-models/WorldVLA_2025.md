# WorldVLA：面向自回归动作世界模型

> **论文**：*WorldVLA: Towards Autoregressive Action World Model*
>
> **作者**：Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, Hao Chen
>
> **机构**：阿里巴巴达摩院、湖畔实验室、浙江大学
>
> **发布时间**：2025年6月
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.21539) | [代码](https://github.com/alibaba-damo-academy/WorldVLA)
>
> **分类标签**：`Action World Model` `自回归统一模型` `VQ-GAN` `Action Attention Mask` `LIBERO` `Chameleon`

---

## 一句话总结

在 Chameleon 统一理解-生成模型基础上，将 VLA 动作模型与世界模型统一到单个自回归框架中——动作模型从图像+指令生成动作、世界模型从图像+动作预测下一帧——两者共享权重、混合训练实现**双向增强**（动作模型 +4.4%、世界模型 FVD -6.2%），并提出 Action Attention Mask 解决自回归 Action Chunking 中的误差累积问题（+4%~23% 成功率）。

---

## 一、问题与动机

### 1.1 VLA 与世界模型的互补缺陷

| 模型类型 | 图像理解 | 图像生成 | 动作理解 | 动作生成 |
| --- | --- | --- | --- | --- |
| Action Model（如 OpenVLA） | ✓ | ✗ | ✗ | ✓ |
| World Model（如 iVideoGPT） | ✓ | ✓ | ✓ | ✗ |
| **Action World Model（WorldVLA）** | **✓** | **✓** | **✓** | **✓** |

- **VLA 模型**：动作仅作为输出，从未作为输入被模型理解，缺乏对动作语义的深层建模
- **世界模型**：能理解动作并预测未来状态，但无法直接输出动作，需要额外的策略模型

### 1.2 自回归 Action Chunking 的误差累积

预训练 MLLM 在图像和文本上有强泛化能力，但在动作领域泛化较弱（动作未参与预训练）。标准因果注意力下，后续动作依赖前序动作，错误会逐步传播放大——chunk 越长，性能衰退越严重（LIBERO-Spatial 从 ~80% 降到 ~20%）。

### 1.3 核心思路

将动作模型和世界模型统一到一个自回归 LLM 中：
- 世界模型学习环境物理动力学 → 帮助动作模型做出更好的决策
- 动作模型增强视觉理解 → 帮助世界模型生成更准确的未来帧
- 用特殊的 Attention Mask 让 Action Chunking 中每个动作独立依赖视觉输入，阻断误差传播

---

## 二、核心方法

### 2.1 架构设计

WorldVLA 基于 **Chameleon**（Meta 的混合模态早期融合基础模型）初始化，采用三种 tokenizer 将不同模态统一到离散 token 空间：

| Tokenizer | 类型 | 细节 |
| --- | --- | --- |
| Image Tokenizer | VQ-GAN + 感知损失 | 压缩比 16，codebook 8192；256×256→256 tokens，512×512→1024 tokens |
| Text Tokenizer | BPE | 词表 65,536（含 8192 图像 token + 256 动作 token） |
| Action Tokenizer | 均匀分箱 | 每维离散为 256 bin，7 tokens/步（3 位移 + 3 旋转 + 1 夹爪） |

所有模态共享同一词表，在单一 LLM 内以自回归方式统一训练。

### 2.2 统一模型形式化

动作模型 $M_\psi^{\text{policy}}$ 和世界模型 $M_\psi^{\text{world}}$ 共享参数 $\psi$：

$$M_\psi : \begin{cases} a_t = M_\psi^{\text{policy}}(a_t \mid o_{t-h:t},\, l) \\ o_t = M_\psi^{\text{world}}(o_t \mid o_{t-h:t-1},\, a_{t-h:t-1}) \end{cases}$$

- **动作模型**：输入历史图像 $\{o_{t-h}, \ldots, o_t\}$ + 语言指令 $l$，输出动作 $a_t$
- **世界模型**：输入历史图像 + 历史动作，预测下一帧 $o_t$

### 2.3 训练数据格式

**动作模型数据**（文本提示："What action should the robot take to \<task\>?"）：

$$\underbrace{[\text{BOS}]\{text\}[\text{BOI}]\{image\}[\text{EOI}]}_{\times M}[\text{EOS}]\overbrace{[\text{BOA}]\{action\}[\text{EOA}]}^{\mathcal{L}_{action}, \times K}[\text{EOS}]$$

输入 $M$ 张图像，输出 $K$ 个动作，仅对动作 token 计算损失。

**世界模型数据**（文本提示："Generate the next frame based on the current image and the action."）：

$$[\text{BOS}]\{text\}[\text{BOI}]\{image\}[\text{EOI}][\text{BOA}]\{action\}[\text{EOA}][\text{EOS}]\overbrace{[\text{BOI}]\{image\}[\text{EOI}]}^{\mathcal{L}_{world}}[\text{EOS}]$$

重复 $N$ 次逐帧预测，仅对生成的图像 token 计算损失。

### 2.4 Action Attention Mask

这是本文的关键技术贡献。标准因果 Attention Mask 下，$\text{Action}_1$ 能看到 $\text{Action}_0$，导致误差在动作序列中累积传播。

**提出的修改**：在动作生成阶段，每个动作 token **只能看到文本和图像 token，不能看到其他动作 token**。这使得每个动作独立地基于视觉输入生成，等价于并行解码，阻断了误差传播链。

具体地：
- **Text token**：标准因果 mask（只看前面的 text）
- **Image token**：可看所有 text + 前面的 image
- **Action token**：可看所有 text + 所有 image，**但不能看其他 action**

世界模型部分仍使用标准因果 mask（因为下一帧的预测确实依赖当前动作）。

### 2.5 训练目标

$$\mathcal{L} = \mathcal{L}_{action} + \alpha \mathcal{L}_{world}$$

其中 $\mathcal{L}_{action}$ 和 $\mathcal{L}_{world}$ 均为交叉熵损失。由于图像 token 远多于动作 token（256~1024 vs 7），用 $\alpha = 0.04$ 平衡两者贡献。

---

## 三、实验结果

### 3.1 LIBERO 基准

| 方法 | 类型 | 预训练 | Spatial | Object | Goal | Long | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 连续 | ✗ | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 连续 | ✓ | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| DiT Policy | 连续 | ✓ | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 |
| OpenVLA-OFT | 连续 | ✓ | 96.9 | 98.1 | 95.5 | 91.1 | 95.4 |
| OpenVLA | 离散 | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| **WorldVLA (256)** | **离散** | **✗** | **85.6** | **89.0** | **82.6** | **59.0** | **79.1** |
| **WorldVLA (512)** | **离散** | **✗** | **87.6** | **96.2** | **83.4** | **60.0** | **81.8** |

WorldVLA 在**无预训练**条件下超越有预训练的 OpenVLA（81.8% vs 76.5%，**+5.3%**）。与连续动作模型相比仍有差距（离散化的信息损失）。512 分辨率优于 256（Chameleon 的 VQ-GAN 原生优化于 512）。

### 3.2 世界模型帮助动作模型

| 配置 | Goal | Object | Spatial | Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| Action Model only | 67.3 | 82.9 | 77.8 | 23.0 | 62.8 |
| + World Model（联合训练） | 73.1 | 88.0 | 80.2 | 27.3 | **67.2 (+4.4)** |

世界模型的加入迫使模型学习环境物理动力学，提升了动作决策质量。可视化显示：纯动作模型直接移向目标而跳过抓取步骤，而 Action World Model 会反复尝试抓取直到成功。

### 3.3 动作模型帮助世界模型

| 配置 | 10 帧 FVD↓ | 10 帧 PSNR↑ | 50 帧 FVD↓ | 50 帧 PSNR↑ |
| --- | --- | --- | --- | --- |
| World Model only | **250.0** | 29.62 | 718.6 | 23.98 |
| Action World Model | 255.1 | **29.77** | **674.1** | **24.30** |

长序列（50 帧）上优势更明显：FVD 从 718.6 降至 674.1（**-6.2%**）。动作模型增强了视觉理解能力，帮助世界模型维持长程一致性。

### 3.4 Action Attention Mask 消融

| 配置 | Goal | Object | Spatial | Long | 平均 |
| --- | --- | --- | --- | --- | --- |
| Action Chunking + 标准 Mask | 79.6 | 82.9 | 36.7 | 16.9 | 54.0 |
| Action Chunking + **Our Mask** | 84.4 | 90.9 | 81.8 | 49.3 | **76.6 (+22.6)** |
| + World Model + **Our Mask** | **85.1** | **90.9** | **84.0** | **52.4** | **78.1** |

标准因果 mask 下 Action Chunking 导致严重性能退化（Spatial 从 77.8% 暴跌至 36.7%），而提出的 mask 策略将 Spatial 恢复至 81.8%，整体平均提升 **22.6%**。

### 3.5 World Model vs Video Prediction Model

世界模型（以动作为条件）在所有 4 个任务上都提升了动作模型性能，而 Video Prediction Model（以任务指令为条件、无动作输入）在 1 个任务上反而降低了性能。原因：无动作输入时，同一初始帧可对应多种合理未来，引入训练噪声。

### 3.6 其他消融

**历史帧数**：2 帧 > 1 帧（+15.6%），4 帧与 2 帧持平但推理更慢，默认 2 帧。

**世界模型预训练**：先用世界模型数据预训练再微调动作模型，平均成功率从 62.8% 提升至 66.8%（+4.0%），验证了世界模型学到的物理先验可迁移至动作生成。

---

## 四、局限性与未来方向

1. **离散 tokenizer 的信息损失**：VQ-GAN 的离散表示在语义理解能力上弱于 CLIP 等连续视觉编码器，限制了模型的感知精度。论文指出需要设计同时支持理解和生成的统一 tokenizer
2. **无大规模预训练**：WorldVLA 未在大规模机器人数据上预训练（对比 OpenVLA-OFT 使用 OXE 预训练），限制了泛化能力。数据和模型规模的 scaling 是明确的改进方向
3. **仅在仿真评测**：所有实验在 LIBERO 基准上完成，未验证真实世界迁移能力
4. **图像生成质量有限**：VQ-GAN 的重建质量不如扩散模型，可能限制世界模型对下游任务的实用价值
5. **辅助动作头的潜力**：论文提到引入额外的连续动作头（类似 $\pi_0$ 的 Action Expert）可能进一步提升抓取性能，但未实验

---

## 五、个人思考

### 5.1 自回归统一 vs 扩散统一：两种路线

WorldVLA 选择了「全离散自回归」路线（Chameleon 风格），而 UVA（Li et al., 2025）选择了「扩散头」路线。两者都追求 Action + World Model 统一，但设计哲学截然不同：

| 维度 | WorldVLA（自回归） | UVA（扩散） |
| --- | --- | --- |
| 动作表示 | 离散 token（256 bin） | 连续向量 |
| 图像生成 | 自回归离散 token | 扩散去噪 |
| 信息损失 | 离散化有损 | 无量化损失 |
| 推理效率 | 逐 token 解码，可并行动作 | 多步去噪 |
| 架构统一度 | 完全统一（单 LLM） | 需要额外扩散头 |

自回归路线的优势在于架构简洁（单一 next-token prediction），但离散化的信息损失是硬伤——论文自己也承认 VQ-GAN 的语义理解能力弱于 CLIP。

### 5.2 Action Attention Mask 的深层洞察

这个设计的核心观察是：**动作空间的泛化能力远弱于图文空间**。MLLM 在图文上经过大规模预训练，能处理 out-of-distribution 的 token 组合，但对从未见过的动作 token 序列缺乏容错能力。因此，自回归生成动作时的误差累积比文本生成严重得多。

这个洞察对所有自回归 VLA（如 OpenVLA、RT-2）都有启发：在动作维度上，「并行解码」可能比「自回归解码」更可靠。这也解释了为什么 $\pi_0$ 等连续动作模型采用 Flow Matching 一次性生成 action chunk，而非逐步自回归。

### 5.3 与 BridgeV2W 的对比视角

BridgeV2W 和 WorldVLA 都是具身世界模型，但走了截然不同的路线：

- **BridgeV2W**：复用预训练视频生成模型（CogVideoX），通过 ControlNet 注入动作条件，专注于高质量视频生成用于策略评估和规划
- **WorldVLA**：从统一理解-生成模型（Chameleon）出发，将世界模型作为动作模型的辅助训练信号，专注于提升动作生成质量

BridgeV2W 将世界模型视为「模拟器」（用于评估和规划），WorldVLA 将世界模型视为「辅助训练任务」（通过多任务学习提升动作质量）。两种定位各有价值，但 WorldVLA 的世界模型生成质量（VQ-GAN 离散重建）可能不足以支撑 BridgeV2W 式的高保真策略评估。

### 5.4 双向增强的非对称性

实验数据揭示了一个有趣的非对称性：世界模型对动作模型的帮助（+4.4% SR）比动作模型对世界模型的帮助（短序列几乎无提升，长序列 FVD -6.2%）更显著。这可能因为：

- 世界模型提供了丰富的物理动力学先验，这是动作决策的关键缺失信息
- 动作模型提供的视觉理解增强相对于世界模型已有的视觉能力是增量式的

### 5.5 与 GigaBrain 等世界模型 RL 工作的关系

GigaBrain-0.5M*、RISE、SC-VLA 等工作也使用世界模型辅助 VLA 训练，但方式不同：它们用世界模型做 imagination rollout 生成虚拟经验，然后用 RL 优化策略。WorldVLA 的方式更直接——世界模型和动作模型共享权重、联合训练，不需要显式的 rollout 和 RL 流程。这种简洁性是自回归统一架构的优势，但也意味着无法像 RL 方法那样通过 reward shaping 精确控制优化方向。

---

## 参考

- **Chameleon**（Meta, 2024）：混合模态早期融合基础模型——WorldVLA 的初始化骨架
- **OpenVLA**（Kim et al., 2024）：开源 VLA，离散动作自回归生成——WorldVLA 的直接对比基线
- **iVideoGPT**（Wu et al., 2025）：交互式离散视频世界模型——离散自回归世界模型的代表
- **UVA**（Li et al., 2025）：统一视频动作模型，扩散头路线——与 WorldVLA 走不同技术路线的同类工作
- **$\pi_0$**（Black et al., 2024）：Flow Matching VLA，连续动作并行生成——Action Chunking 的另一种解决方案
- **OpenVLA-OFT**（Kim et al., 2025）：VLA 微调优化，连续动作头——LIBERO 上的 SOTA
- **GR-1 / GR-2**（Wu et al., 2023; Cheang et al., 2024）：视频生成预训练辅助动作学习——Video Prediction Model 路线的代表
- **LIBERO**（Liu et al., 2023）：终身机器人学习基准，包含 4 个子任务集——WorldVLA 的主要评测平台
