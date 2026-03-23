# CronusVLA：高效鲁棒的多帧视觉-语言-动作建模

> **论文**：*Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling*
>
> **作者**：Hao Li*, Shuai Yang*, Yilun Chen†, Xinyi Chen, Xiaoda Yang, Yang Tian, Hanqing Wang, Tai Wang, Dahua Lin, Feng Zhao†, Jiangmiao Pang†
>
> **机构**：USTC、Shanghai AI Laboratory、Zhejiang University、CUHK
>
> **发布时间**：2026年
>
> 🔗 [项目主页](https://lihaohn.github.io/CronusVLA.github.io)
>
> **发表会议**：AAAI 2026

---

## 一句话总结

CronusVLA 提出两阶段框架将单帧 VLA 扩展到多帧范式：（1）单帧预训练建立具身视觉-语言基础；（2）多帧后训练将离散 token 预测转换为**可学习特征**，通过 **Feature Chunking** 聚合历史帧信息并用 DiT 跨帧解码器生成动作块，配合**多帧正则化**解耦骨干与时序建模。SimplerEnv 70.9%，LIBERO +26.8% 超越 OpenVLA，并提出 **SimplerEnv-OR** 观测鲁棒性基准（24 类干扰 × 120 严重度等级），鲁棒性评分全面领先。

---

## 一、问题与动机

### 1.1 单帧 VLA 的局限

当前 VLA（OpenVLA、RT-2、SpatialVLA 等）继承 VLM 的单帧图像范式，仅使用当前时刻观测 $I_t$ 预测动作：

- **丧失运动线索**：连续观测中的运动信息有助于判断当前执行阶段、消解状态歧义
- **缺乏观测鲁棒性**：单帧损坏时无法从历史一致观测中可靠推断动作
- **长时域状态混淆**：无法利用历史轨迹判断任务进度

### 1.2 直接多帧输入的两大挑战

将多帧图像直接喂入 VLM 骨干面临：

1. **计算开销平方增长**：VLM 中自注意力复杂度随 token 数量二次增长，大规模预训练代价过高
2. **冗余视觉 token 拖慢推理**：严重降低实时部署可行性（朴素多帧方案推理速度降低 40%+）

### 1.3 现有多帧方案的不足

- **RoboVLMs**：采用 LSTM 记忆建模，但从零训练具身能力，忽视了高效适配已有单帧预训练模型的潜力
- **TraceVLA**：在当前帧上绘制历史轨迹作为视觉提示，但依赖精确的历史信息，干扰时鲁棒性差
- **Dita**：多帧输入小型骨干，但无法利用大规模 VLM 预训练先验

### 1.4 CronusVLA 的核心思路

两阶段方案：先单帧预训练保留 VLM 视觉感知，再多帧后训练引入时序建模——**在特征层面而非图像层面聚合多帧信息**，避免 token 数量爆炸。

---

## 二、核心方法

### 2.1 第一阶段：单帧预训练

在 OXE 大规模数据集上，使用标准自回归离散 token 预测训练基础 VLA：

$$a_t = \text{VLA}(I_t, l)$$

- 视觉编码器：DINOv2 + SigLIP
- 动作 tokenizer：连续动作映射到 256 个 bin
- **目的**：将视觉编码器的感知能力迁移到具身场景，建立视觉-语言基础
- **优势**：单帧预训练更好地保留 VLM 的单帧视觉感知，且在大规模数据上训练成本更低

### 2.2 第二阶段：多帧后训练

#### 从离散 token 到 Feature Chunking

核心转变：不再生成离散动作 token，而是在骨干隐藏层引入**可学习特征** $f_t \in \mathbb{R}^d$：

$$f_t = \text{VL}(I_t, l)$$

构建 **Feature Chunking** 聚合历史 $M$ 帧的特征：

$$F_t^M = \{f_{t-M+1}, \ldots, f_{t-1}, f_t\} = f_{t-M+1:t}$$

**训练时**：将 $M$ 帧输入在 batch 维度重组，VLM 骨干独立处理 $B \times M$ 个单帧输入（无多帧 attention 开销）。

**推理时**：使用 **FIFO 队列**缓存历史特征，每步仅需对当前帧做一次前向计算，历史特征直接从队列读取。

#### 跨帧解码器（Cross-frame Decoder）

基于 DiT 的解码器从 Feature Chunking 解码动作块：

$$a_{t:t+K-1} = \text{Decoder}(F_t^M)$$

**Feature Modulator**：平衡当前帧和历史帧的贡献。将当前特征 $f_t$ 通过通道分裂（DIV）扩展到与历史帧数匹配，再通过 MLP 调制：

$$Z_f = \text{MD}(F_t^M) = \text{MLP}(f_{t-M+1:t-1}, \tilde{f}_t)$$

$$\tilde{f}_t = \text{DIV}(f_t), \quad f_t \in \mathbb{R}^d, \; \tilde{f}_t \in \mathbb{R}^{(M-1) \times d}$$

调制后的特征 $Z_f$ 通过**交叉注意力**与噪声动作交互（$Z_f$ 作为 key/value，噪声动作作为 query），迭代去噪生成最终动作。

#### 多帧正则化（Multi-frame Regularization）

关键设计：**解耦 VLM 骨干与多帧建模**，将时序建模限制在解码器内部。

历史帧的可学习特征使用**停止梯度**（stop-gradient）：

$$\hat{f}_{t-M+1:t-1} = \{\text{sg}(\text{VL}(I_{t-k}, l))\}, \; k = 1, \ldots, M-1$$

扩散损失：

$$\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I), i} \left[ \| \hat{\epsilon}^i - \epsilon_\theta(t, \hat{f}_{t-M+1:t-1}, f_t) \|^2 \right]$$

**两个优势**：
1. 历史帧不需梯度计算 → 降低计算和内存开销
2. 骨干始终以单帧方式更新 → 保留预训练感知能力 + 加速收敛

### 2.3 模型配置

| 变体 | LLM 骨干 | 历史帧数 | 推理速度 |
| --- | --- | --- | --- |
| CronusVLA 7B | Llama 2 7B | 6 | 8.7 Hz |
| CronusVLA 0.5B | Qwen2.5 0.5B | 3 | 11.1 Hz |

后训练数据：Bridge-v2 + Fractal，约 148k episodes、5M 多帧片段。

---

## 三、SimplerEnv-OR 基准

### 3.1 设计动机

现有基准（SimplerEnv、LIBERO）评估任务/场景多样性，但忽视**观测干扰**对 VLA 的影响——这对真实世界部署至关重要。

### 3.2 干扰维度

**空间维度**（不同位置/类型的视觉干扰）：
- **Global**：模糊、抖动、全遮挡
- **Local**：过曝、局部遮挡
- **Discrete**：噪声、脉冲

**时间维度**（不同干扰频率）：
- **Constant（1:0）**：每帧都有干扰
- **Cyclic（1:1）**：交替干扰
- **Sparse（1:3, 1:5）**：稀疏干扰

共 **24 类干扰 × 120+ 严重度等级**，超过 2,300 次试验。

### 3.3 鲁棒性评分

$$\text{R-Score}^i = 100 \times \frac{\text{SR}^i}{\text{SR}}$$

其中 $\text{SR}$ 为原始任务成功率，$\text{SR}^i$ 为干扰设置 $i$ 下的成功率。

---

## 四、实验结果

### 4.1 SimplerEnv 主实验

| 方法 | 参数 | GR-VM | GR-VA | WR-VM | **总平均** |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 7B | 35.1 | 35.9 | 3.1 | 24.7 |
| CogACT | 7B | 74.8 | 61.3 | 55.2 | 63.8 |
| TraceVLA | 7B | 45.8 | 49.8 | 27.7 | 41.1 |
| SpatialVLA | 3B | 56.0 | 51.8 | 45.8 | 51.2 |
| Magma | 8B | 48.8 | 57.5 | 44.8 | 50.4 |
| GR00T-N1.5 | 2B | 35.2 | 44.5 | 61.9 | 47.2 |
| **CronusVLA 0.5B** | **0.5B** | 70.5 | 57.8 | 39.6 | 56.0 |
| **CronusVLA 7B** | **7B** | **78.6** | **73.8** | **60.4** | **70.9** |

**核心发现**：
- CronusVLA 7B 全面 SOTA：GR-VM 78.6（超 CogACT +3.8），GR-VA 73.8（超 CogACT +12.5）
- 长时域任务 Put in Drawer（需先开抽屉再放物体）：VM 64.8 / VA 65.1，大多数方法在此任务接近 0
- **0.5B 小模型**超越多数 2B-7B 模型（总平均 56.0），说明参数量并非越大越好，有效建模更重要

### 4.2 LIBERO

| 方法 | Spatial | Object | Goal | Long | **平均** |
| --- | --- | --- | --- | --- | --- |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ + KI | 98.0 | 97.8 | 95.6 | 85.8 | 94.3 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| **CronusVLA 7B** | **97.3** | **99.6** | **96.9** | **94.0** | **97.0** |

- LIBERO 总平均 **97.0%** SOTA，Long 达 94.0%（+40.3% over OpenVLA）
- 仅额外使用手腕视角输入即超越所有方法（包括用机器人状态的 π₀、π₀.₅）

### 4.3 SimplerEnv-OR 鲁棒性测试

**时间维度**：

| 方法 | Constant R-Score | Cyclic R-Score | Sparse R-Score | 原始 SR |
| --- | --- | --- | --- | --- |
| π₀ | 43.5 | 36.8 | 34.9 | 20.9 |
| CogACT | 53.3 | 66.1 | 80.2 | 55.2 |
| **CronusVLA** | **61.2** | **86.7** | **96.2** | **60.4** |

**空间维度**：

| 方法 | Global R-Score | Local R-Score | Discrete R-Score | **总平均 R-Score** |
| --- | --- | --- | --- | --- |
| CogACT | 60.2 | 80.5 | 87.4 | 72.1 |
| RoboVLMs | 54.7 | 83.3 | 76.8 | 67.4 |
| **CronusVLA** | **85.4** | **96.6** | 80.2 | **86.9** |

**核心发现**：
- CronusVLA 在 Sparse（1:3）干扰下几乎免疫（R-Score 96.2）
- 单帧模型（π₀、SpatialVLA、CogACT）在高频干扰下产生分布外动作导致失败
- RoboVLMs 和 TraceVLA 虽然是多帧模型，但严重依赖精确历史信息，干扰时倾向于不动或重复探测
- SpatialVLA 在 SimplerEnv 上优于 RoboVLMs，但在 OR 基准上反而更差——揭示标准基准可能掩盖鲁棒性缺陷

### 4.4 消融实验

#### 后训练策略

| 配置 | 总平均 SR | 推理速度 |
| --- | --- | --- |
| 基线（单帧后训练） | 31.0 | 5.18 Hz |
| + 多帧直接输入 | 32.4（+1.4） | 3.09 Hz（-40%） |
| + 多帧 + 解码器 | 48.2（+17.2） | **8.73 Hz**（+68%） |
| + 多帧 + 解码器 + VL 骨干训练 | 67.2（+36.2） | 8.73 Hz |
| + 多帧 + 解码器 + VL + 正则化（Ours） | **70.9**（+39.9） | 8.73 Hz |

- 朴素多帧方案仅 +1.4% 性能但速度 -40%
- Feature Chunking + 解码器方案性能 +17.2% 且速度反而提升（消除自回归解码 + 缓存历史特征）
- 多帧正则化额外贡献 +3.7%，且显著加速收敛

#### 帧数影响

- CronusVLA 7B 最优帧数为 7（总 1+6 历史帧）
- CronusVLA 0.5B 最优帧数为 4
- **更多帧并非更好**：过多时序输入可能导致性能退化
- CronusVLA 推理速度随帧数增加几乎不变，而朴素基线显著退化

---

## 五、局限性与未来方向

1. **双阶段训练的额外开销**：单帧预训练 + 多帧后训练的两阶段流程比端到端训练更复杂，后训练数据选择和超参调优增加工程负担
2. **帧数需要手动调优**：7B 最优 7 帧、0.5B 最优 4 帧——最优帧数依赖模型容量和任务特性，缺乏自适应机制
3. **仅支持第三人称单视角**：当前框架假设单一固定相机视角，多视角（如手腕+第三人称）的多帧建模尚待探索
4. **SimplerEnv-OR 仅覆盖 WidowX**：鲁棒性基准尚未扩展到 Google Robot 或真实世界设置

---

## 六、个人思考

### 6.1 "特征层面聚合"的核心洞察

CronusVLA 最重要的设计决策是在**特征层面而非图像层面**聚合多帧信息。直观理解：VLM 骨干将每帧图像压缩为一个可学习特征 $f_t$，多帧信息在这个压缩表示上聚合——避免了多帧图像 token 的二次注意力开销。消融实验中，直接多帧输入仅 +1.4% 且速度 -40%，而 Feature Chunking 方案 +17.2% 且速度 +68%，差距惊人。

### 6.2 多帧正则化的解耦哲学

"骨干始终以单帧方式更新，时序建模限制在解码器内"——这种解耦类似于计算机视觉中冻结图像编码器 + 训练时序模块的范式（如 VideoBERT）。历史帧特征使用 stop-gradient 确保骨干的单帧感知不被多帧噪声干扰，同时解码器自由学习跨帧动态。

### 6.3 与 MemoryVLA 的对比

项目中的 [MemoryVLA](MemoryVLA_2025.md) 同样关注 VLA 的时序建模，采用感知-认知双流记忆库。与 CronusVLA 的关键区别：
- **MemoryVLA**：跨注意力检索 + 门控融合 + 合并压缩，显式建模长时域记忆
- **CronusVLA**：Feature Chunking + FIFO 队列 + DiT 解码器，轻量级时序聚合

CronusVLA 的方案更工程友好（FIFO 队列机制简单高效），但 MemoryVLA 的记忆库可能在超长时域任务中更有优势。

### 6.4 SimplerEnv-OR 的重要贡献

这个鲁棒性基准填补了 VLA 评估的重要空白。最有趣的发现是：SpatialVLA 在标准 SimplerEnv 上优于 RoboVLMs，但在 OR 基准上反而更差——说明**标准基准可能系统性高估了某些模型的实际部署能力**。多帧模型在观测干扰下的天然优势（从历史一致帧推断动作）在 OR 基准中得到了量化验证。

### 6.5 0.5B 模型的启示

CronusVLA 0.5B 以极小的参数量（0.5B vs 7B）超越多数大模型，总平均 56.0% 优于 SpatialVLA（51.2%）、Magma（50.4%）、GR00T-N1.5（47.2%）。这强化了一个重要观点：**有效的建模设计比单纯堆叠参数更有价值**，特别是在实时部署场景中。

---

## 参考

- **OpenVLA**（Kim et al., 2025）：基础骨干和主要基线
- **CogACT**（Li et al., 2024）：SimplerEnv 先前 SOTA
- **TraceVLA**（Zheng et al., 2025）：视觉轨迹提示多帧建模
- **RoboVLMs**（Li et al., 2024b）：LSTM 记忆式多帧 VLA
- **SpatialVLA**（Qu et al., 2025）：空间自适应动作网格 VLA
- **MemoryVLA**（Xie et al., 2025）：双流记忆库时序建模
- **DiT**（Peebles and Xie, 2023）：扩散 Transformer，跨帧解码器骨干
- **π₀/π₀.₅**（Black et al., 2024/2025）：Flow Matching VLA，LIBERO 基线
