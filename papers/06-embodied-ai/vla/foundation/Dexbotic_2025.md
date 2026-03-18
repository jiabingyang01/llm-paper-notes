# Dexbotic：开源 VLA 模型工具箱——统一框架、更强预训练与实验驱动开发

> **论文**：*Dexbotic: Open-Source Vision-Language-Action Toolbox*
>
> **作者**：Bin Xie, Erjin Zhou, Fan Jia, Hao Shi 等（Dexmal, StepFun）
>
> **发布时间**：2025年10月
>
> 🔗 [arXiv](https://arxiv.org/abs/2510.23511) | [项目主页](https://dexbotic.com) | [代码](https://github.com/Dexmal/dexbotic)
>
> **分类标签**：`VLA Toolbox` `统一框架` `预训练基础模型` `实验驱动开发`

---

## 一句话总结

Dexbotic 是一个基于 PyTorch 的开源 VLA 工具箱，通过统一模块化架构（VLM + Action Expert）、更强的预训练基础模型（DexboticVLM → Dexbotic-Base → Dexbotic-CogACT）和以实验为中心的 Exp 脚本开发范式，让用户在**单一环境**中复现、对比和开发多种主流 VLA 策略（$\pi_0$、CogACT、OpenVLA-OFT、MemoryVLA 等），并在 SimplerEnv、CALVIN、LIBERO、ManiSkill2、RoboTwin2.0 等基准上实现大幅性能提升（最高 +46.2%）。

---

## 一、问题与动机

### 1.1 VLA 研究的碎片化困境

当前 VLA 研究分散在不同机构，每家使用不同的深度学习框架和模型架构。这导致：

1. **环境配置繁琐**：对比不同策略需要分别搭建多套实验环境和数据格式
2. **公平比较困难**：难以确保每种被比较的策略都达到其最佳性能
3. **VLM 骨架过时**：OpenVLA、CogACT、OFT 等主流方法都构建于 Llama 2 之上，其表征能力远逊于 Qwen3 等最新 LLM

### 1.2 类比 AI 1.0 时代的工具箱发展

回顾目标检测领域，mmdetection 将检测器统一拆解为 backbone、neck、head，极大推动了算法对比和迭代。Dexbotic 试图在 VLA 领域做同样的事——将所有 VLA 策略统一拆解为 **VLM + Action Expert** 两部分，提供统一的模块化开发框架。

---

## 二、核心方法

### 2.1 统一模块化 VLA 框架

Dexbotic 将所有 VLA 策略统一表示为两大组件：

| 组件 | 构成 | 功能 |
| --- | --- | --- |
| **VLM**（视觉-语言模型） | Vision Encoder + Projector + LLM | 处理观测和任务指令，生成多模态 token |
| **Action Expert**（动作专家） | Diffusion Transformer / MLP / MoE 等 | 接收 VLM 表征，输出连续/离散动作 |

VLM 部分的多模态 token 既可以直接解码为离散动作（如 RT-2、OpenVLA），也可以作为 Action Expert 的输入生成连续动作块（如 CogACT 的 Diffusion Transformer、$\pi_0$ 的 Flow Matching）。

当前支持的 VLA 策略包括：

- **$\pi_0$**：Flow Matching Action Expert，基于 PaliGemma
- **OpenVLA-OFT**：OpenVLA 的改进版，探索微调策略
- **CogACT**：提取 cognition token + Diffusion Transformer
- **MemoryVLA**：引入感知-认知记忆，提升长时域任务
- **MUVLA**：基于地图理解的导航策略

### 2.2 DexboticVLM：更强的预训练基础模型

Dexbotic 不满足于使用过时的 VLM 骨架，而是从头预训练自己的 VLM：

**架构选择**：

| 组件 | 选型 |
| --- | --- |
| Vision Encoder | CLIP |
| Projector | 两层 MLP |
| LLM | Qwen2.5 |

**训练流程**（类似 LLaVA pipeline）：

> 1. **Stage 1**：冻结 Vision Encoder 和 LLM，仅训练 Projector，完成跨模态对齐
> 2. **Stage 2**：全参数更新（Vision Encoder + Projector + LLM）

训练数据包括 LLaVA 数据集和 Cambrian 数据集。

### 2.3 三级预训练模型体系

基于 DexboticVLM，Dexbotic 提供三级递进的预训练模型：

**第一级：Dexbotic-Base（离散预训练模型）**

在 DexboticVLM 基础上，用单臂机器人数据进一步预训练离散 VLA 模型。数据来源包括：
- Open-X Embodiment 子集
- 仿真数据（RLBench、LIBERO、ManiSkill2）
- 真实机器人数据（UR5 等）

动作离散化方式：每个自由度独立量化为 256 bins，VLM 预测 $N$ 个离散 token（$N$ = DoF）。

Dexbotic-Base 可作为所有 VLM-based 策略的初始化，同时支持离散和连续动作学习——对于连续动作建模，只需在 VLM 上追加 Action Expert，VLM 部分从 Dexbotic-Base 加载，Action Expert 随机初始化。

**第二级：Dexbotic-CogACT（单臂连续预训练模型）**

在 Dexbotic-Base 基础上，加入 DiT（Diffusion Transformer）头，对整个 CogACT 模型做连续表征预训练。训练数据覆盖 8 种单臂机器人（UR5、Franka、Unitree Z1、Realman GEN72、Realman 75-6F、UMI、ARX5、WidowX），共 52 个操作任务。这些机器人具有不同的 DoF，对训练基础设施提出了挑战。

**第三级：Hybrid-arm 连续预训练模型（双臂扩展）**

原始 CogACT 不支持多视角和双臂设置。Dexbotic 的扩展策略：

- 将噪声 token 数量从 7 扩展到 16，覆盖 6-DoF 和 7-DoF 臂
- 前半部分 token 表示左臂动作，后半部分表示右臂动作
- 对于单臂数据：仅用单臂动作监督前半部分 token，后半部分 loss 被忽略
- 多视角支持：共享 Vision Encoder 处理多视角图像，视觉 token 拼接后输入 LLM
- 额外引入 Robomind、AgiBot World 数据集以及私有双臂 ALOHA 数据

### 2.4 Dexdata 统一数据格式

Dexbotic 定义了 Dexdata 格式来统一存储多机器人数据集，相比 LeRobot 和 RLDS 格式可节省存储空间：

```
video/
  episode1.mp4          # 视频文件（mp4 格式）
  episode2.mp4
jsonl/
  index_cache.json      # 全局元数据索引（自动生成）
  episode1.jsonl        # 每个 episode 一个 jsonl 文件
  episode2.jsonl
```

每行 jsonl 包含一帧的完整信息：多视角图像引用（type + url + frame_idx）、机器人状态向量、文本指令。视频以 mp4 存储，通过 frame_idx 索引到具体帧，避免了逐帧存图的存储开销。

### 2.5 以实验为中心的开发框架

Dexbotic 采用"分层配置 + 工厂注册 + 入口分发"的实验框架：

- **base_exp**：基础配置脚本，包含 optimizer、trainer、action、data、model、inference 的默认设置
- **策略 Exp**（如 CogACT_Exp）：继承 base_exp 并仅覆盖需要修改的字段
- **自定义 Exp**：用户继承并修改配置和模型类，即可开发全新策略

运行方式极简：`python xxx_exp.py -task train`（或 `inference`）。这种设计遵循**开闭原则**——对扩展开放、对修改封闭。

### 2.6 训练流水线与推理服务

**训练流水线**：

> 1. 文本指令 → Text Tokenizer → Text Encoder → text tokens
> 2. 观测图像 → Vision Encoder → image tokens → MLP Projector → 对齐到文本空间
> 3. image tokens + text tokens → LLM → 离散 token / 多模态表征
> 4. 对于离散策略：直接解码为动作；对于连续策略：送入 Action Expert 生成 action chunk
> 5. 生成的动作序列由 ground-truth 动作通过对应 loss 监督

**推理服务**：基于 Flask 的 Web API 架构。DexClient 发送请求 → Web API 接收并处理图像/文本 → VLA 模型推理 → 返回连续动作序列 → DexClient 执行动作。

### 2.7 DOS-Twins：Real2Sim 评估

Dexbotic 提出 DOS-Twins（Dexbotic Open Source-Twins），基于 Isaac Sim + 3D Gaussian Splatting 构建 Real2Sim2Real 仿真环境，在三个维度保证与真实世界的一致性：

| 维度 | 方法 | 目标 |
| --- | --- | --- |
| **视觉一致性** | 3DGS 光真实渲染 + 精确相机标定 | 仿真视角与真实相机完全匹配 |
| **运动一致性** | 低层控制器标定 | 仿真运动学/动力学对齐真实硬件 |
| **交互一致性** | 高精度 3D 扫描夹爪和物体（毫米级误差） | 抓取交互物理行为对齐 |

用户可在真实世界训练策略，然后在 DOS-Twins 中进行一致的评估。

---

## 三、实验结果

### 3.1 SimplerEnv-Bridge（WidowX）

| 方法 | Spoon on Towel | Carrot on Plate | Stack Cube | Eggplant in Basket | **Avg. Suc** |
| --- | --- | --- | --- | --- | --- |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| **DB-CogACT** | 87.5 | 65.3 | 29.2 | 95.8 | **69.5 (+18.2)** |
| OFT | 12.5 | 4.2 | 4.2 | 100.0 | 30.2 |
| **DB-OFT** | 91.7 | 76.4 | 43.1 | 94.4 | **76.4 (+46.2)** |
| MemVLA | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 |
| **DB-MemVLA** | 100.0 | 66.7 | 70.8 | 100.0 | **84.4 (+12.5)** |

DB-OFT 的提升最为惊人——从 30.2% 飙升到 76.4%，绝对提升 46.2%。说明 OFT 原始版本受限于过时的 Llama 2 骨架，换用更强的预训练模型后潜力巨大。

### 3.2 CALVIN（ABC→D 长时域泛化）

| 方法 | 1 | 2 | 3 | 4 | 5 | **Avg. Len** |
| --- | --- | --- | --- | --- | --- | --- |
| CogACT | 0.838 | 0.729 | 0.640 | 0.559 | 0.480 | 3.25 |
| **DB-CogACT** | 0.935 | 0.867 | 0.803 | 0.760 | 0.698 | **4.06 (+0.81)** |
| OFT | 0.891 | 0.794 | 0.674 | 0.598 | 0.515 | 3.47 |
| **DB-OFT** | 0.928 | 0.807 | 0.692 | 0.602 | 0.511 | **3.54 (+0.07)** |

DB-CogACT 在 CALVIN 长时域任务上平均完成任务数从 3.25 提升到 4.06，提升 0.81。在需要连续完成 5 个指令的最难设定下，从 48.0% 提升到 69.8%。

### 3.3 RoboTwin 2.0（双臂任务）

| 方法 | Adjust Bottle | Grab Roller | Place Empty Cup | Place Phone Stand | **Avg. Suc** |
| --- | --- | --- | --- | --- | --- |
| CogACT | 87 | 72 | 11 | 5 | 43.75 |
| **DB-CogACT** | 99 | 89 | 28 | 18 | **58.5 (+14.75)** |

说明 Dexbotic 的双臂连续预训练模型能有效迁移到双臂具身场景。

### 3.4 ManiSkill2（3D 感知与空间推理）

| 方法 | PickCube | StackCube | PickSingleYCB | PickSingleEGAD | PickClutterYCB | **Avg. Suc** |
| --- | --- | --- | --- | --- | --- | --- |
| CogACT | 55 | 70 | 30 | 25 | 20 | 40 |
| **DB-CogACT** | 90 | 65 | 65 | 40 | 30 | **58 (+18.0)** |
| OFT | 40 | 45 | 5 | 5 | 0 | 21 |
| **DB-OFT** | 90 | 75 | 55 | 65 | 30 | **63 (+42.0)** |

DB-OFT 在 ManiSkill2 上从 21% 提升到 63%，绝对提升 42%——原始 OFT 在多数任务上几乎不工作，换用 Dexbotic 预训练模型后全面激活。

### 3.5 LIBERO（近饱和基准）

| 方法 | Spatial | Object | Goal | Long | **Avg. Suc** |
| --- | --- | --- | --- | --- | --- |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 |
| **DB-CogACT** | 93.8 | 97.8 | 96.2 | 91.8 | **94.9 (+1.3)** |
| MemVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| **DB-MemVLA** | 97.2 | 99.2 | 98.4 | 93.2 | **97.0 (+0.3)** |

在性能已经接近饱和的 LIBERO 上仍有小幅提升，说明更强的预训练模型在天花板附近仍有边际收益。

### 3.6 真实世界任务

在 UR5e、ALOHA、ARX5、Franka 等机器人上验证，每个任务收集 500-1000 条遥操作演示：

| 任务 | 机器人 | 成功率 |
| --- | --- | --- |
| 摆放盘子 | UR5e | 100% |
| 搜索绿色箱子 | ARX5 | 80% |
| 码放碗 | ALOHA | 90% |
| 戴帽子 | Franka | 70% |
| 按按钮 | Franka | 60% |
| 碎纸 | UR5e | 40% |
| 倒薯条 | ALOHA | 20% |

精细操作任务（碎纸、倒薯条）仍然是现有 VLA 策略的挑战。

---

## 四、局限性与未来方向

1. **预训练数据组成不透明**：论文未分析不同数据源对最终性能的贡献，哪些数据在帮忙、哪些可能在拖后腿，尚不清楚
2. **策略覆盖有限**：当前版本支持的 VLA 策略以操作为主，$\pi_{0.5}$ 和 NaVid、NaVILA 等导航策略尚未集成
3. **缺乏系统性消融**：对 DexboticVLM 的各设计选择（Qwen2.5 vs 其他 LLM、CLIP vs SigLIP、训练数据量）缺少详细消融
4. **精细操作瓶颈**：真实世界实验中，碎纸（40%）和倒薯条（20%）等精细任务成功率仍低，说明更强的预训练模型不能完全解决精细操作问题
5. **DOS-Twins 评估规模有限**：Real2Sim 评估仅展示了少数任务的定性对比，缺少系统性的 sim-real gap 量化分析

---

## 五、个人思考

### 5.1 "工具箱论文"的独特价值

Dexbotic 本质上是 VLA 领域的"mmdetection"——它的核心贡献不是提出新算法，而是**统一框架 + 更强预训练 + 标准化实验流程**。这类工具箱论文的价值在于降低了入门门槛和公平比较的成本。

### 5.2 预训练骨架升级的"低垂果实"

实验结果中最令人印象深刻的是 **DB-OFT 在 SimplerEnv 上 +46.2%、ManiSkill2 上 +42.0%** 的提升——这几乎完全来自将 Llama 2 替换为 Qwen2.5 作为 LLM 骨架。这说明当前很多 VLA 方法被过时的 VLM 骨架严重拖累，升级基础模型是一个性价比极高的改进方向。

### 5.3 与项目中已有论文的联系

- **与 $\pi_0$ 的关系**：Dexbotic 将 $\pi_0$ 作为支持的策略之一，但用 Qwen2.5 替代了 PaliGemma 作为 LLM 骨架。这提供了一个有趣的对比维度——$\pi_0$ 的 Action Expert 设计 + 更强的 LLM 骨架会带来多大收益？
- **与 CogACT 的关系**：DB-CogACT 是论文中最详细展示的策略，其双臂扩展（噪声 token 7→16）提供了将单臂策略推广到双臂的一种简洁方案
- **对 VLA RL 后训练的启示**：项目中大量 RL 后训练工作（如 RISE、WMPO、SimpleVLA-RL）的基线 VLA 模型普遍使用过时骨架。如果基线模型本身就更强（如用 Dexbotic 的预训练模型），RL 后训练的边际收益可能会缩小——这是一个值得关注的问题

### 5.4 Dexdata 格式的权衡

Dexdata 用 mp4 视频 + jsonl 元数据的方式存储，相比 LeRobot 的逐帧存储确实节省空间。但 mp4 压缩是有损的，对于需要精确像素信息的任务（如精细操作），压缩损失是否会影响训练质量？论文未讨论这一点。

---

## 参考

- **mmdetection**：AI 1.0 时代的目标检测统一工具箱，Dexbotic 在设计理念上的直接借鉴
- **$\pi_0$**：Dexbotic 支持的核心 VLA 策略之一，Flow Matching + Action Expert 架构的开创者
- **CogACT**：Dexbotic 详细展示的策略，基于 Diffusion Transformer 的 Action Expert
- **OpenVLA / OpenVLA-OFT**：Dexbotic 支持的策略，受益于骨架升级最显著的方法
- **MemoryVLA**：引入感知-认知记忆的 VLA，在 Dexbotic 框架下进一步提升了长时域任务性能
