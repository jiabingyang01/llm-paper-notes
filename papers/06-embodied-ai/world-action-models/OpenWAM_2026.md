# OpenWAM：面向系统化世界-动作模型预训练的开放模块化研究框架

> **论文信息**：Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, Donghao Zhou, Qize Yu, Yu-Kai Wang, Yuhan Rui, Shenzhe Yao, Zhen Yuan, Zhenhao Shen, Kefei Zhu, Zijie Zhu, Ning Gao, Xiaowei Chi, Guanqi He, Shanghang Zhang, Hao Dong, Lin Shao, Hang Zhao (NUS / Tsinghua / PKU / HKU / ZJU / CUHK / SJTU)
> **发表期刊/会议**：arXiv 2026 (`arXiv:2609.07398`)
> **项目主页**：[openwam-official.github.io](https://openwam-official.github.io/) · **代码仓库**：[OpenWAM-Official/OpenWAM](https://github.com/OpenWAM-Official/OpenWAM)

## 1. 核心定位与研究范式

世界-动作模型（World-Action Models, WAMs）通过从视频生成基座继承物理动力学先验，并将其转化为可执行控制信号，成为继 VLA 之后的另一大主流范式。然而现有 WAM 系统均为高度耦合的“单体黑盒（Monolithic Systems）”——视频主干、视觉编码器、动作专家架构、训练/推理信息流向以及数据配比彼此缠绕，导致社区无法回答 **“究竟哪些设计真正决定了 WAM 的性能？”**

本文构建了开放研究栈 **`OpenWAM`**，由三部分组成：
1. **`OpenWAM-Infra`**：将 WAM 设计空间解耦为可组合的标准模块，提供统一的训练、推理部署与跨基准评测底座；
2. **`OpenWAM-Study`**：围绕 **“继承什么先验（What to Inherit）”、“世界与动作如何交互（How They Interact）”、“协同效应如何随数据扩展（How Synergy Scales）”** 三大核心科学问题开展严格受控实验，提炼出三条黄金设计法则；
3. **`OpenWAM-α`（6.0B）**：遵循受控实验提炼的最佳配方，在约 **6,400 小时（5.185 亿帧，涵盖 21 种机器人本体 + 第一视角人类视频）** 异构数据上完成一阶段协同预训练，在 **8 大仿真基准**与 **3 类真机平台（单臂、双臂、五指灵巧手）** 上全面刷新或追平 SOTA。

## 2. OpenWAM-Study：三大核心科学问题与受控实验结论

### 2.1 问题一：WAM 应当继承什么样的上游世界先验？（What to Inherit）

作者固定下游动作学习模块，系统对比了不同规模的视频生成骨干与视觉表征空间（重建型 VAE vs. 表征型编码器如 SigLIP 2 / DINOv2 / WebSSL）：
- **Finding 1（上游先验继承法则）**：WAM 最有效地通过**容量充足的视频生成骨干（如 Wan2.2-5B）** 与**紧凑且信息密集的潜空间（Compact, Information-Rich Latent Space）** 继承上游物理先验。除了像素重建型 VAE（如 SD-VAE / Wan-VAE）外，经过维度压缩的语义表征编码器（Representation Auto-Encoders, RAE）同样具备极强的动力学传递能力。

### 2.2 问题二：世界先验与动作学习应当如何交互？（How World and Action Interact）

作者在 RoboTwin 2.0-Full 上开展了架构容量、训练期注意力掩码与推理期去噪调度的三维受控消融：

1. **架构容量解耦（Single- vs. Dual- vs. Tri-System）**：
   - 单系统（共享主干直接输出动作）存在模态竞争；
   - **双系统联合自注意力（Dual-System Joint Self-Attention，视频分支 + 独立动作专家在每层共享自注意力）** 达到 **91.85%** （显著优于截断梯度的 Cross-Attention 与 IDM 架构）；
   - 加入 VLM 理解分支的三系统（Tri-System）进一步达 **92.60%**。综合复杂度与纯粹性，`OpenWAM-α` 采用**双系统联合自注意力架构**。
2. **训练期跨流信息流向（Training-Time Information Flow）**：
   通过注意力掩码严格控制视频流（World）与动作流（Action）之间的可见性：

| 训练期注意力掩码策略 | 视频流能否看动作 | 动作流能否看视频 | RoboTwin 2.0 Clean (%) | RoboTwin 2.0 Random (%) | 平均成功率 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Isolated（完全隔离）** | $\times$ | $\times$ | 88.08 | 86.74 | 87.41 |
| **Video Sees Action（仅视频看动作）** | $\checkmark$ | $\times$ | 87.92 | 87.34 | 87.63 |
| **Action Sees Video（单向：动作看视频）** | $\times$ | $\checkmark$ | **92.98** | **91.80** | **92.39** |
| **Mutual（双向互见）** | $\checkmark$ | $\checkmark$ | 92.50 | **91.80** | 92.15 |

   - **核心发现**：**世界流向动作流的信息通道（Action Sees Video）是产生协同效应的必要条件** （缺失该通道会导致成功率下跌近 $5\%$）；而在大规模预训练尺度下，**双向互见（Mutual Visibility）** 表现最稳健。
3. **推理期去噪调度（Inference-Time Denoising Schedule）**：
   - **Finding 2**：推理时采用**同步联合去噪（Synchronized Joint Denoising，即视频潜变量与动作块沿对角线同步去噪）** 最优，无需先完全生成视频再解码动作。

### 2.3 问题三：世界-动作协同如何随预训练扩展？（How Synergy Scales）

- **Finding 3（预训练功能定位与一阶段混训法则）**：
  1. 具身预训练的核心收益在于**大幅扩展分布外（OOD）泛化与抗干扰能力** （如 RoboTwin Clean2Random 从头训练仅 $\sim 40\%$，预训练后跃升至 $69.0\%$）；
  2. **机器人轨迹数据**负责提供精确的物理动作接地（Action Grounding），**第一视角人类视频（Egocentric Human Video）** 负责拓宽场景与物体交互覆盖面；
  3. **一阶段混合协同训练（One-Stage Co-Training，将人类视频与多机器人轨迹按比例混合一步预训练）** 显著优于“先视频预训练、再机器人预训练”的两阶段流水线。

## 3. OpenWAM-α 预训练数据配方与全基准实验

### 3.1 6,369 小时（5.185 亿帧）精选预训练数据配方

作者对原始 **14,348 小时（13.34 亿帧）** 数据进行严格清洗与整集子采样，构建了 `OpenWAM-α`（6.0B）的 **6,369 小时（5.185 亿帧）** 预训练混合集：

| 数据源 | 数据类型 | 本体数量 | 帧率 (FPS) | 原始规模 (小时) | 清洗采样后规模 (帧 / 小时) | 每 Epoch 采样占比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Egocentric Human Video (自建)** | 野外人类操作视频 | 1 (人手) | 30 | 6,897 h | 155.7M / **1,442 h** | **30.1%** |
| **InternData-A1** | 高保真仿真轨迹 | 4 | 30 | 2,904 h | 155.5M / **1,440 h** | **30.0%** |
| **AgiBotWorld-Beta** | 真机双臂/移动操作 | 1 | 15 | 2,306 h | 96.9M / **1,794 h** | **18.6%** |
| **RoboCOIN** | 真机多本体双臂/灵巧手 | 15 | 30 | 956 h | 74.1M / **686 h** | **14.3%** |
| **DROID** | 真机单臂野外操作 | 1 | 10 | 1,285 h | 36.3M / **1,007 h** | **7.0%** |
| **总计（Total）** | **人类 + 真机 + 仿真** | **21 机器人 + 人类** | - | **14,348 h** | **518.5M / 6,369 h** | **100.0%** |

### 3.2 八大仿真基准全面评测结果

`OpenWAM-α`（6.0B）在横跨单臂、双臂、人形机器人的 8 大仿真基准上与顶尖 VLA 及 WAM 模型进行了系统对比：

| 评测基准 | $\pi_0$ (3.3B) | $\pi_{0.5}$ (3.3B) | X-VLA (0.9B) | StarVLA (4.0B) | ABot-M0 (4.2B) | Fast-WAM (6.0B) | Motus (8.0B) | **`OpenWAM-α` (6.0B)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RoboTwin 2.0-Clean2Random (Clean / Rand. / Avg)** | 46.4 / 16.3 / 31.4 | 70.7 / 46.0 / 58.4 | 68.0 / 20.9 / 44.5 | 46.5 / 3.2 / 24.9 | 70.7 / 36.0 / 53.4 | 77.8 / 1.9 / 39.9 | - | **89.4 / 48.7 / 69.0** |
| **RoboTwin 2.0-Full (Clean / Rand. / Avg)** | - | 82.7 / 76.8 / 79.8 | 72.8 / 72.8 / 72.8 | 88.2 / 88.3 / 88.3 | 86.1 / 85.1 / 85.6 | 91.9 / 91.8 / 91.9 | 88.7 / 87.0 / 87.8 | **93.74 / 93.46 / 93.60** |
| **LIBERO (4 Suites Avg.)** | 94.4 | 96.9 | 98.1 | 96.6 | 98.6 | 97.6 | 97.7 | **99.3** |
| **LIBERO-Plus (Zero-Shot 7-OOD Avg.)** | 53.6 | **84.4** | - | 74.1 | 80.5 | 51.5 | - | 69.2 |
| **RoboCasa-GR1 / Tabletop Avg.** | 36.3 | 39.6 | - | 40.6 | - | - | - | **69.7** |
| **VLABench (Overall SR / PS)** | 29.4 / 44.1 | 48.1 / 62.3 | - | - | - | - | - | **58.9 / 67.2** |
| **RoboCasa365 (Overall SR / Score)** | - | 6.91 / 11.41 | 6.52 / 10.13 | 3.24 / 6.40 | - | 2.03 / 3.48 | - | **11.92 / 17.18** （WAM 第 1） |

### 3.3 真机实验（单臂 Franka、双臂 RoboDojo、五指灵巧手）

1. **Franka-Research-3 单臂 6 任务（120 次独立真机试验）**：`OpenWAM-α` 取得 **82.5%（99/120）** 的平均成功率（其中抽屉放辣椒与抽屉放积木均达 **100%**），超越 `LingBot-VA`（$77.5\%$）与 `π₀.₅`（$55.0\%$）；
2. **Wuji + Tianji 五指灵巧手真机 4 任务（ID + OOD）**：在搭玩具塔、捡羽毛球、拧瓶盖、收衣服等高自由度灵巧手任务上，`OpenWAM-α` 在分布内（ID）与分布外（OOD）设置下全面大幅超越 `π₀.₅`（例如捡羽毛球 ID 成功率达 **60%** vs. `π₀.₅` 的 $20\%$）。

## 4. 核心洞见：WAM 与 VLA 的范式边界（Takeaways）

作者在论文末尾总结了一个极具启发性的科学判断（Takeaway 3）：
- **视频潜空间监督赋予 WAM 极强的分布内（In-Distribution）运动拟合与物理轨迹精度** （如 LIBERO 达 $99.3\%$、RoboTwin-Full 达 $93.6\%$）；
- 但由于像素级重建潜空间对剧烈相机视角偏移与高频像素噪声敏感，**以高层语义特征为核心的 VLA 在极端视觉扰动（如 LIBERO-Plus 的 Camera 与 Noise 子集）上天然更具优势**；
- 这一互补性指明了下一代具身基础模型的演进方向：将语义/几何不变的表征编码器（RAE）与世界-动作协同生成相融合。
