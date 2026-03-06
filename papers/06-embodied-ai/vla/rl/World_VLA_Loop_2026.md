# World-VLA-Loop：视频世界模型与 VLA 策略的闭环联合学习

> **论文**：*World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy*
>
> **作者**：Xiaokang Liu*, Zechen Bai*, Hai Ci, Kevin Yuchen Ma, Mike Zheng Shou
>
> **机构**：Show Lab, National University of Singapore
>
> **时间**：2026 年 2 月
>
> **链接**：[arXiv](https://arxiv.org/abs/2602.06508) | [项目主页](https://showlab.github.io/World-VLA-Loop/)

---

## 一句话总结

World-VLA-Loop 提出世界模型与 VLA 策略的**闭环联合优化**范式：基于 Cosmos-Predict 2 构建带奖励预测头的 state-aware 视频世界模型，用 SANS（Success and Near-Success）数据集提升动作跟随精度，在虚拟环境中通过 GRPO 做 RL 后训练 VLA，然后将策略的失败 rollout 反馈回来迭代改进世界模型——两轮迭代后真实世界成功率从 13.3% 提升至 50.0%。

---

## 一、问题与动机

### 1.1 世界模型做 VLA RL 的瓶颈

RL 后训练已被证明能显著提升 VLA 性能，但真实世界 RL 交互成本极高。用视频世界模型作为虚拟环境是一条有前景的路线（参见 [WMPO](./WMPO_2025)、[WoVR](./WoVR_2026)、[RISE](./RISE_2026)），但现有视频世界模型存在两个关键缺陷：

1. **动作跟随精度差**：模型倾向于依赖视觉先验而非真正响应动作条件。如 Cosmos-Predict 2 在接收到错误动作时，仍然"幻觉"出成功结果——夹爪朝物体"吸"过去而非如实反映抓取失败
2. **奖励信号不可靠**：依赖外部 VLM 或启发式代理奖励，精度不足以支撑稳定 RL

### 1.2 核心洞察：世界模型和策略可以互相改进

World-VLA-Loop 的关键观察是：VLA 策略在训练过程中行为分布不断变化，世界模型需要覆盖这些变化的状态才能保持可靠。反过来，策略产生的失败 rollout 正好是世界模型最需要学习的数据。这构成了一个天然的闭环：

- **世界模型 → 策略**：提供虚拟环境做 RL 后训练
- **策略 → 世界模型**：提供新的失败/成功轨迹增强训练数据

---

## 二、预备知识

### 2.1 动作条件视频世界模型

给定初始 $h$ 帧观测 $x_0, \dots, x_{h-1}$ 和未来 $T$ 步动作 $a_1, \dots, a_T \in \mathbb{R}^6 \cup \{0, 1\}$（6-DoF 末端执行器位姿 + 夹爪开合），视频世界模型预测未来帧 $x_h, \dots, x_{h+T-1}$。

Cosmos-Predict 2 使用 Diffusion Transformer（DiT）骨架自回归预测视频块。动作通过 MLP 映射为潜空间嵌入，加到扩散时间步嵌入上注入 DiT。

### 2.2 GRPO 策略优化

World-VLA-Loop 使用 GRPO（Group Relative Policy Optimization）做策略更新。对一组 rollout $\{\tau^{(i)}\}_{i=1}^G$，计算组相对优势：

$$R(\tau^{(i)}) = \sum_t \gamma^{t-1} r_t^{(i)}, \quad \hat{A}^{(i)} = \frac{R(\tau^{(i)}) - \text{mean}(\{R\})}{\text{std}(\{R\})}$$

使用 clipped objective 更新策略，避免大幅度更新。

---

## 三、核心方法

World-VLA-Loop 包含四个阶段：(1) 构建 SANS 数据集；(2) 训练 state-aware 视频世界模型；(3) 在世界模型中做 VLA RL 后训练；(4) 部署策略、收集新数据、迭代优化。

### 3.1 SANS 数据集：成功 + 近成功轨迹

现有机器人数据集几乎只包含成功轨迹（因为主要服务于模仿学习），这导致世界模型无法准确模拟失败情况。World-VLA-Loop 引入 **SANS（Success and Near-Success）** 数据集，核心是**近成功轨迹**——几乎完成目标但因末端执行器位置的微小偏差而失败的轨迹。

近成功数据的两个关键作用：

1. **迫使模型学习细粒度差异**：近成功与成功轨迹在视觉上极其相似，模型必须关注空间动力学的精细差异才能区分
2. **覆盖真实失败模式**：策略在部署中最常出现的就是这种"差一点"的失败

数据收集方式：

| 环境 | 成功轨迹 | 近成功轨迹 |
| --- | --- | --- |
| ManiSkill（23 任务） | 控制策略 + GT 物体位姿 | 扰动位姿生成失败 |
| LIBERO | 演示数据 | OpenVLA-OFT 的失败 rollout |
| 真实世界 | 人工遥操作 | 遥操作 + 策略失败 rollout |

ManiSkill 上共收集约 35k video-action pairs 用于世界模型预训练。LIBERO 和真实世界每个任务仅需约 50 条成功 + 50 条近成功轨迹做微调。

### 3.2 State-aware 视频世界模型

基于 Cosmos-Predict 2 构建，核心创新是**联合预测视频帧和奖励信号**。

#### 3.2.1 奖励预测头

在 DiT 生成扩散潜码 $z_t$ 后，增加一个轻量 MLP 奖励头 $\phi$：

$$\hat{r}_t = \phi(z_t)$$

与 flow matching 损失联合训练：

$$\mathcal{L} = \mathcal{L}_{flow} + \lambda \sum_{t=1}^{T} \|\hat{r}_t - r_t\|^2$$

其中 $\lambda$ 根据 EDM 框架按采样噪声等级调制，确保奖励头在去噪早期（高方差潜码）仍然鲁棒。

**联合训练的双重优势**：

1. **奖励更准确**：基于生成的视觉潜码预测奖励，与视觉结果内在对齐，比外部 VLM 判断或启发式代理更可靠
2. **反向提升视频质量**：奖励监督迫使生成器更好地区分成功/失败动作的视觉结果，改善动作跟随精度

#### 3.2.2 训练流程

两阶段训练：

1. **预训练**：在 ManiSkill SANS 数据集（35k 轨迹）上从 Cosmos-Predict 2 检查点迁移学习，获得基础的机器人动作条件化能力
2. **微调**：对新的下游任务，只需少量数据（< 100 条成功 + 近成功轨迹）即可适配

### 3.3 世界模型中的 VLA RL 后训练

使用 OpenVLA-OFT 作为基础 VLA，采用 SimpleVLA-RL 的训练框架，但将物理仿真器替换为学习的世界模型。

**闭环交互**：世界模型根据策略输出的动作自回归生成下一步观测，反馈给策略做后续预测。奖励由世界模型的奖励头提供，通过阈值化（$\hat{r} > 0.9$）转为二值成功信号，供 GRPO 计算组相对优势。

统一 action chunk 大小为 24，即每次预测 24 帧视频。

### 3.4 迭代联合优化：闭环的核心

这是 World-VLA-Loop 最关键的设计——世界模型和策略的**协同进化循环**：

> 1. 用初始 SANS 数据训练世界模型 $\text{WM}_0$
> 2. 在 $\text{WM}_0$ 中做 RL，获得改进后的策略 $\pi_1$
> 3. 部署 $\pi_1$ 到真实环境，收集新的成功和失败 rollout
> 4. 用新数据增强 SANS 数据集，训练 $\text{WM}_1$
> 5. 在 $\text{WM}_1$ 中从 SFT 基线重新做 RL，获得 $\pi_2$
> 6. 重复...

**为什么有效**：

- 第一轮世界模型可能在某些边界情况上不准确（如策略用杯子背面抓取时，世界模型仍预测成功）
- 策略在这些不准确的地方会发生 reward hacking
- 将这些失败轨迹反馈给世界模型后，第二轮世界模型能正确建模这些边界情况
- 在改进后的世界模型中训练的策略不再能 hack，被迫学习真正有效的行为

---

## 四、实验结果

### 4.1 世界模型质量

**视频生成质量**（Table 1）：

| 场景 | SSIM ↑ | PSNR ↑ | LPIPS ↓ | MSE ↓ |
| --- | --- | --- | --- | --- |
| LIBERO | 0.90 | 26.57 | 0.031 | 0.0024 |
| Real-World | 0.91 | 29.61 | 0.059 | 0.0019 |
| 平均 | 0.91 | 28.09 | 0.045 | 0.0022 |

**结果对齐度**（Table 2，每任务评估 20 个样本）：

| 指标 | LIBERO-Object | LIBERO-Goal | LIBERO-Spatial | Real-World |
| --- | --- | --- | --- | --- |
| Visual Alignment | 85%/95% | 90%/75% | 85%/95% | 90% |
| Reward Alignment | 75%/90% | 85%/75% | 90%/95% | 95% |

平均 Visual Alignment 87.9%，Reward Alignment 86.4%，两者高度一致，说明奖励头的判断与人类视觉判断对齐。

### 4.2 VLA RL 后训练效果

**LIBERO + Real-World 成功率**（Table 3）：

| 模型 | Object T1 | Object T2 | Goal T1 | Goal T2 | Spatial T1 | Spatial T2 | Real-World |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SFT Base | 73.9% | 73.9% | 91.9% | 86.1% | 83.9% | 87.9% | 13.3% |
| RL Post-Training | 97.9% | 91.9% | 100% | 96.2% | 93.9% | 94.0% | 36.7% |
| **提升** | **+24.0%** | **+18.0%** | **+8.1%** | **+10.1%** | **+10.0%** | **+6.1%** | **+23.4%** |

LIBERO 平均提升 +12.7%，真实世界 +23.4%。

**迭代优化效果**（真实世界，Figure 1b）：

| 阶段 | 成功率 |
| --- | --- |
| SFT 基线 | 13.3% |
| 第 1 轮 RL | 36.7%（+23.4%） |
| 第 2 轮 RL（迭代） | 50.0%（+13.3%） |

两轮迭代累计提升 +36.7%，验证了闭环联合优化的有效性。

### 4.3 消融实验

**世界模型设计消融**（Table 4，LIBERO-Object）：

| 消融变体 | Task 1 | Task 2 |
| --- | --- | --- |
| w/o 近成功数据 | 60% | 65% |
| w/o 奖励预测头 | 60% | 70% |
| 用 Qwen3-VL 替代奖励头 | 50% | 55% |
| **完整 World-VLA-Loop（Visual）** | **85%** | **95%** |
| **完整 World-VLA-Loop（Reward）** | **75%** | **90%** |

**关键发现**：

1. **奖励预测头至关重要**：去掉后 Visual Alignment 下降约 30%，说明联合训练不仅提供奖励，还显著提升视频生成的成功/失败区分能力
2. **近成功数据不可或缺**：去掉后对齐度同样大幅下降，验证了 SANS 数据集的核心价值
3. **外部 VLM 奖励不可靠**：Qwen3-VL 作为 success/failure 判断器仅 50-55% 对齐度，存在严重幻觉——世界模型生成的帧对 VLM 而言是 OOD 输入

### 4.4 效率分析

- 世界模型单次 24 帧生成耗时约 7 秒（NVIDIA H100）
- SimpleVLA-RL 约 50 步优化收敛
- 单个任务 RL 训练约 30 小时完成

---

## 五、局限性与未来方向

1. **长时域任务受限**：当前自回归视频模型在超过 200 帧（~20 秒）后出现严重质量漂移，LIBERO-100 的长时域任务未能评估
2. **稀疏奖励**：仅使用最终状态的二值成功信号，未探索分步子目标的中间奖励
3. **迭代效率**：每轮迭代需要真实世界部署收集新数据，仍有人工成本
4. **基线策略较弱**：真实世界 SFT 基线仅 13.3%，说明 OpenVLA-OFT 在该场景的初始能力有限

---

## 六、个人思考

### 6.1 与 WoVR 的互补关系

World-VLA-Loop 和 [WoVR](./WoVR_2026) 是同期工作，都用视频世界模型做 VLA RL，但关注点不同：

| 维度 | World-VLA-Loop | WoVR |
| --- | --- | --- |
| **核心创新** | 闭环迭代联合优化 | 三级幻觉控制 |
| **世界模型骨架** | Cosmos-Predict 2 | Wan2.2-TI2V-5B |
| **奖励设计** | 内嵌奖励预测头 | 外部二值分类器 |
| **数据策略** | SANS 近成功数据 | 噪声上下文增强 |
| **幻觉应对** | 迭代数据增强 | KIR + PACE |
| **长程能力** | 受限（~200 帧） | 较强（~512 帧） |

World-VLA-Loop 的"闭环"思想更优雅——让数据驱动世界模型改进，而非纯架构层面的防御。但 WoVR 的 KIR 和首帧锚定等技术对长程稳定性更有效。两者的结合（SANS 数据 + 幻觉控制 + 闭环迭代）可能是更完整的方案。

### 6.2 奖励预测头 vs 外部奖励模型

论文最有说服力的消融是：去掉奖励头后视频生成质量也下降 ~30%。这说明**奖励监督是一种有效的生成正则化**——迫使模型真正理解动作的因果后果，而非仅仅生成视觉上合理的序列。这一洞察对所有视频世界模型的训练都有启发：即使不需要奖励信号，加入结果标签的辅助损失也可能提升生成精度。

### 6.3 SANS 数据的通用价值

近成功数据的概念不限于世界模型训练。对于 VLA 的模仿学习、偏好学习（如 [GRAPE](./GRAPE_2025)）、甚至 reward model 训练，decision boundary 附近的数据（"差一点"的失败）都极其宝贵。当前社区的数据收集范式过于偏重成功演示，SANS 的思路值得推广。

### 6.4 迭代闭环 vs 一次性训练

与 [WoVR](./WoVR_2026) 的 PACE（也是策略-模型协同进化）相比，World-VLA-Loop 的迭代更"重"——每轮需要真实世界部署收集数据。PACE 是在同一轮 RL 中在线更新世界模型，更轻量但可能更新幅度有限。真正理想的方案可能是：PACE 式的在线微调 + 周期性的真实世界数据增强。

### 6.5 与 GigaBrain 的对比视角

[GigaBrain](./GigaBrain_2026) 也用视频世界模型 + 迭代训练，但它的 RAMP 框架将世界模型用于**优势估计**（条件化 VLA），而 World-VLA-Loop 将世界模型用作**完整的交互式仿真器**。前者更轻量（不需要闭环 rollout），后者更灵活（支持完整的 on-policy 探索）。

---

## 参考

- [WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://arxiv.org/abs/2602.13977)：同期工作，幻觉感知的世界模型 RL，关注三级可靠性保障
- [RISE: Self-Improving Robot Policy with Compositional World Model](https://arxiv.org/abs/2602.11075)：组合式世界模型（动力学 + 价值）在想象空间做 RL
- [WMPO: World Model-Based Policy Optimization for VLA](https://arxiv.org/abs/2511.09515)：直接用世界模型做 imagination rollout + PPO
- [GigaBrain: World-Model-Conditioned Policy Optimization](https://arxiv.org/abs/2602.14842)：世界模型预测未来状态 + 价值条件化 VLA
- [SimpleVLA-RL: Scaling VLA Training via RL](https://arxiv.org/abs/2509.09674)：World-VLA-Loop 的 RL 训练框架基础
- [OpenVLA-OFT: Fine-Tuning VLA Models](https://arxiv.org/abs/2502.19645)：基础 VLA 策略
- [VLA-RFT: Vision-Language-Action Reinforcement Fine-Tuning](https://arxiv.org/abs/2510.00406)：用视频世界模型 + verified reward + GRPO 微调 VLA
