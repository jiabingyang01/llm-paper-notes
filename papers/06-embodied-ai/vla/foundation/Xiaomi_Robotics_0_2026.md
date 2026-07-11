# Xiaomi-Robotics-0：面向实时执行的开源视觉-语言-动作模型

> **论文**：*Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution*
>
> **作者**：Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li 等（Xiaomi Robotics 团队，作者按字母序排列，共 22 人）
>
> **机构**：Xiaomi Robotics（小米机器人）
>
> **发布时间**：2026 年 02 月（arXiv 2602.12684，v2 于 2026 年 3 月更新）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.12684) | [PDF](https://arxiv.org/pdf/2602.12684)
>
> **分类标签**：`VLA` `实时执行` `异步动作分块` `Mixture-of-Transformers` `双臂操作`

---

## 一句话总结

Xiaomi-Robotics-0 用 Qwen3-VL-4B + 16 层 DiT 的 Mixture-of-Transformers 架构（4.7B 参数）做 flow-matching 动作生成，通过 RoPE 位置偏移 + Λ 形注意力掩码修复了 Training RTC 异步执行中"直接复制已提交动作前缀而忽视视觉语言信号"的捷径问题，在 LIBERO（98.7%）、CALVIN（ABCD→D 4.80 / ABC→D 4.75）、SimplerEnv（WidowX 79.2% / Google Robot 视觉匹配 85.5%）三大仿真基准上全面 SOTA，并在消费级 RTX 4090 上实现真机双臂 Lego 拆分与毛巾折叠任务的平滑实时执行，吞吐超过 π_0.5 与训练时 RTC 基线。

## 一、问题与动机

VLA 模型基于预训练 VLM 将观测与语言指令映射为动作，但参数量可达数十亿，推理延迟高，难以平滑地连续输出动作 chunk；处理不当会导致连续推理步之间的动作跳变，把机器人带入分布外状态。现有异步执行方案中，RTC（Black & Galliker, 2025）用免训练的 inpainting 算法"冻结"已提交动作再生成后续动作；Training RTC（Black et al., 2025）在训练阶段就把已提交动作前缀作为条件输入。但作者发现：把动作生成建立在"前缀动作"条件之上会让策略学习走捷径——直接模仿前缀而非关注视觉/语言信号，导致反应性下降，在毛巾折叠这类需要根据形变实时纠错的长时序可变形物体任务上会陷入重复失败循环。本文的目标是设计一套预训练—后训练—部署的完整方案，既通过联合训练保留预训练 VLM 的视觉语义知识（避免灾难性遗忘），又能让 VLA 在真实双臂机器人上用消费级 GPU 实现快速、平滑、可复现的实时闭环执行，并将权重与推理代码开源。

## 二、核心方法

**架构**：Mixture-of-Transformers（MoT）。预训练 VLM（Qwen3-VL-4B-Instruct）处理观测图像 $\mathbf{o}_t$ 与语言指令 $l$；16 层 Diffusion Transformer（DiT）以 VLM 最后 16 层的 KV cache 为条件，通过 flow matching 生成 $T$ 步动作 chunk $\mathbf{a}_{t:t+T}$，并编码机器人本体状态 $\mathbf{s}_t$。总参数量 4.7B。

**预训练分两步**。第一步让 VLM 具备动作生成能力：采用 Choice Policies 范式，让 VLM 同时预测 $N$ 个动作 chunk 候选及其得分，以 winner-takes-all 方式（只更新与真值 L1 距离最小的候选）监督；同时以 1:6 的比例混合视觉-语言数据（next-token-prediction 目标）与机器人轨迹数据，避免遗忘预训练 VLM 的视觉语义知识。第二步冻结 VLM，从头训练 DiT，在约 200M 时间步的全部机器人轨迹数据上以 flow-matching loss 训练：

$$L(\theta) = \|\mathbf{v}_\theta(\mathbf{o}_t, l, \mathbf{s}_t, \tilde{\mathbf{a}}^\tau_{t:t+T}, \tau) - \mathbf{u}(\tilde{\mathbf{a}}^\tau_{t:t+T}, \mathbf{a}_{t:t+T}, \tau)\|_2^2$$

其中 $\tau\in[0,0.999]$，$\tilde{\mathbf{a}}^\tau_{t:t+T} = \tau\mathbf{a}_{t:t+T} + (1-\tau)\epsilon$，$\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$。用大白话说：让网络学会把一团随机噪声，沿着"噪声到真实未来动作序列"的直线路径逐步去噪，还原出真实的动作 chunk。

**数据**：机器人轨迹数据来自公开数据集（DROID、MolmoAct）及自采数据（双臂机器人 Lego 拆分 338 小时、毛巾折叠 400 小时遥操作），总计约 200M 时间步；视觉-语言数据超过 80M 样本，来自通用 VL 数据集与机器人轨迹衍生数据，后者通过 Grounded SAM + Grounding DINO 1.5 + LLMDet 交叉验证获得像素级 grounding 标注，并用预训练 VLM 对根轨迹重新标注 embodied QA、任务规划、点轨迹预测等数据。

**后训练的关键修复**：沿用 Training RTC 思路，把已提交的 $\Delta t_c$ 步动作作为 clean 前缀 prepend 到带噪动作 token 前。为避免后续预测直接"抄"前缀，提出两个简单修复：(1) 给带噪动作 token 的 RoPE 位置编码加 +10 偏移，与 clean 前缀 token 区分；(2) 把 DiT 因果注意力掩码换成 Λ 形注意力掩码——带噪动作 token 只能看到 VLM 的 KV cache（视觉语言信息）、sink token、状态 token，以及前 $w$ 个时间步的动作 token，看不到更远的前缀 token，从而迫使模型依赖视觉语言信号而非单纯复制前缀。训练时还根据在线预测动作与真值的 L1 误差动态重加权 flow-matching loss，$\Delta t_c$ 在训练中从 $\{0,\dots,6\}$ 采样。

**部署**：同步执行——执行完当前 chunk 的前 $T_e$ 步后开始推理下一 chunk，期间机器人空闲；异步执行——在推理下一 chunk 期间继续执行当前 chunk 剩余步骤，把第 $T_e$ 到 $T_e+\Delta t_c-1$ 步的已提交动作作为前缀条件送入下一次推理，只要 $\Delta t_c \geq$ 推理延迟 $\Delta t_{\inf}$，新 chunk 生成完成时执行队列中始终有可用动作，实现连续无缝切换。推理从标准高斯噪声初始化动作 chunk，做 5 步 flow-matching 积分；在消费级 NVIDIA RTX 4090 上推理延迟 $t_{\inf}=80$ ms；所有传感器输入按时间戳重采样对齐到统一 30Hz 时钟。

## 三、关键结果

**LIBERO**（平均成功率 %）：

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π_0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| EO-1 | 99.7 | 99.8 | 99.2 | 94.8 | 98.2 |
| **Xiaomi-Robotics-0** | 98.8 | 100.0 | 98.8 | 97.2 | **98.7** |

**CALVIN**（5 任务链平均完成长度）：ABCD→D 达 **4.80**（最佳基线 FLOWER 4.67），ABC→D（零样本环境泛化）达 **4.75**（最佳基线 FLOWER 4.53）。

**SimplerEnv**（overall 成功率）：WidowX **79.2%**（最佳基线 EO-1 72.7%）；Google Robot 视觉匹配 **85.5%**（EO-1 76.5%）；Google Robot 视觉聚合泛化 **74.7%**（EO-1 63.0%）。

**真机双臂实验**（RTX 4090 部署）：Lego 拆分（LA-5/10/20 + MA 多组装设置）各方法成功率接近（高 80%–90% 区间），同步方法成功率略高于异步方法，但完整异步的 Xiaomi-Robotics-0 吞吐量最高，超过 Training RTC 变体和 π_0.5。毛巾折叠任务中 π_0.5、Sync 变体、Training RTC 变体吞吐均约 1 pcs/min，而完整方法达 **1.2 pcs/min**；论文指出 Training RTC 变体常因误抓多层毛巾而陷入"反复甩动作"的循环（前缀捷径导致策略不再响应视觉信号纠错），Λ 掩码方法能有效避免此类重复失败。

**VL 能力保留**（Table 3，10 个通用 VL 基准 + ERQA embodied reasoning）：Xiaomi-Robotics-0 在 ERQA 达 40.8，POPE 88.5，MMBench 84.4，均明显优于 π_0/π_0.5（在几乎所有 VL 基准上得分为 0）和 MolmoAct（ERQA 33.5）；ERQA 上甚至略超基座 Qwen3-VL-4B-Instruct（40.0），归因于机器人轨迹衍生的 VL 数据强化了机器人中心视角的视觉感知。但在 ChartQA（59.2 vs 基座 76.8）、MMMU（46.2 vs 51.7）、SciQA（79.4 vs 92.7）等通用图表/科学推理任务上仍落后基座较多。去掉 VL 联合训练的消融（w/o VL data）在所有 VL 基准上得分为 0，验证了联合训练对防止灾难性遗忘的必要性。

## 四、评价与展望

**优点**：给出了覆盖预训练—后训练—部署三阶段的完整工程方案，并公开权重与推理代码，可复现性强；对 Training RTC"前缀捷径"问题的修复（RoPE 偏移 + Λ 形掩码 + 误差加权 loss）针对性强，用真机毛巾折叠的重复循环失败案例直观证明了该捷径问题的真实存在及修复的有效性，是本文最有信息量的贡献；三大仿真基准全面 SOTA，且相比 π_0/π_0.5 在保留通用 VL 能力方面有明显优势，说明其联合训练配方具有实用价值。

**局限与开放问题**：真机验证仅覆盖 2 类任务（Lego 拆分、毛巾折叠）与单一双臂平台，尚未展示跨本体、跨场景的真机泛化能力，这与 π_0.5、GR00T-N1 等强调开放世界泛化的工作形成对比；异步执行在高精度接触密集任务（Lego 拆分）中成功率仍略低于同步执行，说明"前缀条件 + Λ 掩码"缓解但未完全消除响应性与流畅性之间的权衡；通用 VL 能力（尤其图表理解、科学问答）相较基座 VLM 仍有明显退化，VL:轨迹 = 1:6 的混合比例是否最优尚待更系统的消融验证。与同期 RTC 系工作（RTC、Training RTC）相比，本文提出的 attention-sink + 局部窗口式掩码在大语言模型长上下文外推领域已有先例（如 StreamingLLM、LM-Infinite），更多是把该机制迁移到动作 chunk 场景做工程整合而非全新范式；后续可探索该掩码设计与扩散/流匹配步数、chunk 长度的联合超参搜索，以及在更多样化本体和任务上的系统验证。

## 参考

1. Black et al. π_0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164, 2024.
2. Physical Intelligence (Black et al.). π_0.5: A Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054, 2025.
3. Black & Galliker. Real-Time Execution of Action Chunking Flow Policies. arXiv:2506.07339, 2025.
4. Black, Ren, Equi & Levine. Training-time Action Conditioning for Efficient Real-Time Chunking. arXiv:2512.05964, 2025.
5. Qu et al. EO-1: Interleaved Vision-Text-Action Pretraining for General Robot Control. arXiv:2508.21112, 2025.
