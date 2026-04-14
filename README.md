# 🧠 LLM Paper Notes

[![Website](https://img.shields.io/badge/Website-llm--paper--notes-blue)](https://llm-paper-notes.jiabingyang.cn/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

> **大语言模型及相关领域**的论文精读笔记。每篇包含问题动机、前置知识、方法拆解、公式推导、实验分析与个人思考。

👉 **在线阅读**：[llm-paper-notes.jiabingyang.cn](https://llm-paper-notes.jiabingyang.cn/)

---

## 🗺️ 分类体系

| | 分类 | 覆盖方向 |
| :---: | --- | --- |
| 🏗️ | Foundation Models | GPT、LLaMA、Mamba、Scaling Laws、MoE 预训练 |
| 🛡️ | Alignment & Safety | RLHF、DPO、RLAIF、Constitutional AI |
| 💡 | Reasoning | CoT、ToT、o1/o3、数学推理、Test-time Compute |
| 🖼️ | Multimodal | GPT-4V、LLaVA、视频理解、语音模型 |
| 🤖 | Agents | ReAct、Toolformer、WebAgent、SWE-Agent |
| 🦾 | Embodied AI | VLA、世界模型、机器人 RL、模仿学习 |
| ⚡ | Efficiency | GPTQ、AWQ、LoRA、Speculative Decoding |
| 🔍 | RAG & Knowledge | Dense Retrieval、RAPTOR、GraphRAG |
| 📊 | Evaluation | MMLU、HumanEval、Arena、LLM-as-Judge |

> 一篇论文可以出现在多个分类的索引中，但笔记 `.md` 只存一份，放在最核心的分类下。

---

## 📚 已收录论文

<details>
<summary>🏗️ Foundation Models</summary>

> 暂无笔记

</details>

<details open>
<summary>🛡️ Alignment & Safety</summary>

- [R³L (2026)](papers/02-alignment-and-safety/R3L_2026.md) — 反思-重试 + 关键点信用分配 + 正向放大，GRPO 改进框架提升 5%–52%

</details>

<details>
<summary>💡 Reasoning</summary>

> 暂无笔记

</details>

<details open>
<summary>🖼️ Multimodal</summary>

<blockquote>
<details open>
<summary>VLM</summary>

<blockquote>
<details open>
<summary>幻觉缓解</summary>

- [ACPO (2026)](papers/04-multimodal/vlm/hallucination/ACPO_2026.md) — 非对称标定系数仅压制 rejected 梯度 + 长度自适应优势目标 τ，逆转视觉锚点崩塌，InternVL3-14B POPE 89.22、MM-IFEval 0.570
- [AGLA (2025)](papers/04-multimodal/vlm/hallucination/AGLA_2025.md) — GradCAM + Image-Prompt Matching 生成增强图像，全局-局部 logit 融合 training-free 缓解幻觉，POPE +5.5%
- [AVISC (2025)](papers/04-multimodal/vlm/hallucination/AVISC_2025.md) — 盲 token 注意力校准 + 对比解码，training-free 缓解幻觉，InstructBLIP POPE +6%、AMBER 85.95
- [CIPHER (2026)](papers/04-multimodal/vlm/hallucination/CIPHER_2026.md) — 扩散反事实图像构建 OHC-25K + SVD 提取视觉幻觉子空间 + 推理时投影抑制，零额外推理开销，CHAIR$_S$ 13.05%
- [CSR (2024)](papers/04-multimodal/vlm/hallucination/CSR_2024.md) — CLIP 校准自奖励 + 句子级 beam search + 迭代 DPO，三轮迭代 CHAIR$_S$ 降低 57%
- [DLC (2025)](papers/04-multimodal/vlm/hallucination/DLC_2025.md) — CLIP 动态探针 + 相对视觉优势 + 自适应 Logits 调制，Training-Free 高效缓解 LVLM 幻觉
- [EFUF (2024)](papers/04-multimodal/vlm/hallucination/EFUF_2024.md) — CLIP 筛选 + 细粒度梯度上升遗忘幻觉子句，无需配对数据仅 3 GPU 小时，CHAIR$_S$ 平均降低 ~15%
- [FLB (2026)](papers/04-multimodal/vlm/hallucination/FLB_2026.md) — 首 token logit 指数递增叠加 + "The" 效应双机制对抗长程视觉衰减，单次前向零开销，AMBER CHAIR 11.5→6.1
- [FarSight (2025)](papers/04-multimodal/vlm/hallucination/FarSight_2025.md) — 因果掩码上三角注意力寄存器 + 渐减遮蔽率位置编码，training-free 缓解图像/视频 MLLM 幻觉，CHAIR$_S$ -6.4 pp
- [HALC (2024)](papers/04-multimodal/vlm/hallucination/HALC_2024.md) — 自适应 FOV 对比解码 + 视觉匹配 beam search，无训练即插即用缓解三种对象幻觉
- [HIME (2026)](papers/04-multimodal/vlm/hallucination/HIME_2026.md) — HIS 层自适应加权投影编辑，无训练/无开销降低 61.8% 对象幻觉
- [IBD (2024)](papers/04-multimodal/vlm/hallucination/IBD_2024.md) — 注意力图像偏置 + 内容词/功能词动态调节对比解码，仅 74K 参数全面超越 VCD/OPERA/Woodpecker
- [HIO (2024)](papers/04-multimodal/vlm/hallucination/HIO_2024.md) — 反转 BT 模型训练 Evil LVLM 精准放大幻觉 + 对比解码消除，CHAIR$_I$ 降至 2.24
- [ICD (2024)](papers/04-multimodal/vlm/hallucination/ICD_2024.md) — 指令对比解码：扰动指令放大幻觉后对比剥离，training-free 模型无关，POPE +10.5%/+6.0%
- [LessIsMore (2024)](papers/04-multimodal/vlm/hallucination/LessIsMore_2024.md) — EOS 决策视角：修改 MLE 保护模型内在终止倾向 + 数据过滤移除有害样本，CHAIR$_S$ -26.4%
- [LogicCheckGPT (2024)](papers/04-multimodal/vlm/hallucination/LogicCheckGPT_2024.md) — 逻辑闭环探测（对象→属性→对象），黑盒后处理缓解幻觉，POPE 准确率提升超 30%
- [LPOI (2025)](papers/04-multimodal/vlm/hallucination/LPOI_2025.md) — 首次列表级偏好优化：对象遮蔽渐进插值生成有序图像序列 + Plackett-Luce 排序损失，Object HalBench CHAIR$_S$ 24.3（mDPO 30.7）
- [LURE (2024)](papers/04-multimodal/vlm/hallucination/LURE_2024.md) — 共现/不确定性/位置三因素统计分析 + GPT-3.5 构造幻觉数据训练修正器，后处理兼容任意 LVLM，CHAIR$_S$ 平均降低 50%+
- [mDPO (2024)](papers/04-multimodal/vlm/hallucination/mDPO_2024.md) — 发现多模态 DPO 忽略图像条件，图像对比偏好优化 + 锚定奖励正则化缓解幻觉
- [MemVR (2025)](papers/04-multimodal/vlm/hallucination/MemVR_2025.md) — FFN key-value memory 视觉回溯 + 不确定性动态触发，1.04× 延迟缓解幻觉且提升通用能力
- [MMHalSnowball (2024)](papers/04-multimodal/vlm/hallucination/MMHalSnowball_2024.md) — 多模态幻觉雪球效应评估框架 + 残差视觉解码自适应缓解多轮对话幻觉累积
- [OPERA (2024)](papers/04-multimodal/vlm/hallucination/OPERA_2024.md) — 注意力柱状聚合模式检测 + 过度信任惩罚 + 回溯重分配，无训练 Beam Search 解码缓解幻觉
- [REVERIE (2024)](papers/04-multimodal/vlm/hallucination/REVERIE_2024.md) — 正负 rationale 反思式指令微调，254k 三元组数据集提供细粒度推理监督，POPE +12.7、MME +348
- [RFI (2026)](papers/04-multimodal/vlm/hallucination/RFI_2026.md) — Rectified Flow 动态预测输入特定干预向量 + SVD 去噪，仅 1.09x 延迟，POPE 平均 F1 +7.59%
- [SENTINEL (2025)](papers/04-multimodal/vlm/hallucination/SENTINEL_2025.md) — 域内自举 + 句子级 C-DPO 早期干预，幻觉率降低 92% 且通用能力不降反升
- [SIMA (2024)](papers/04-multimodal/vlm/hallucination/SIMA_2024.md) — 三视觉指标引导的上下文自评估 + 自生成偏好数据 DPO，无需外部模型，14 基准平均 +7.5%
- [STIC (2024)](papers/04-multimodal/vlm/hallucination/STIC_2024.md) — 两阶段自训练：good/bad prompt + 图像腐蚀构造偏好数据 DPO + 描述注入式 SFT，仅用无标签图像，7 基准平均 +4.0%
- [TAF (2026)](papers/04-multimodal/vlm/hallucination/TAF_2026.md) — Token 非对称过滤：隔离 phantom token 的 T2V 干扰 + 强调 anchor token 的视觉证据，training-free 全面 SOTA
- [VACoDe (2024)](papers/04-multimodal/vlm/hallucination/VACoDe_2024.md) — Softmax L2 距离自适应选择对比增强 + 对比解码，Training-Free 跨模型一致提升
- [VCD (2026)](papers/04-multimodal/vlm/hallucination/VCD_2026.md) — 高斯噪声扰动放大幻觉 + 原始/噪声分布对比解码 + 自适应可信度约束，Training-Free 无需外部工具，POPE +7.4 F1、MME +18%
- [VGA (2024)](papers/04-multimodal/vlm/hallucination/VGA_2024.md) — Referent Method + FAC 两阶段微调，63.8k GUI VQA 数据集，GUI 理解超越 GPT-4V/GPT-4o
- [VisFlow (2025)](papers/04-multimodal/vlm/hallucination/VisFlow_2025.md) — 双层注意力干预（Token 级增强显著视觉 token + Head 级抑制系统/文本头），无训练缓解幻觉

</details>
</blockquote>

<blockquote>
<details open>
<summary>Token 压缩</summary>

- [DART (2025)](papers/04-multimodal/vlm/efficiency/DART_2025.md) — 基于 token 重复度而非重要性剪枝视觉 token，兼容 FlashAttention，88.9% 剪枝率保持 93.7% 性能，1.99× 加速
- [Elastic Cache (2024)](papers/04-multimodal/vlm/efficiency/ElasticCache_2024.md) — Importance-driven cache merging + 固定截断点淘汰，training-free KV Cache 压缩实现 78% 加速
- [Token Pruning Survey (2025)](papers/04-multimodal/vlm/efficiency/TokenPruningSurvey_2025.md) — 系统性分析 MLLM 视觉 token 剪枝五大问题：位置偏差、语言引导条件、重要性 vs. 冗余性、FLOPs 评估偏差、训练感知压缩优势
- [VisionZip (2024)](papers/04-multimodal/vlm/efficiency/VisionZip_2024.md) — CLS 注意力选 dominant token + key 相似度合并 contextual token，LLM 前完成压缩，64/576 token 达 94% 性能，8× prefilling 加速

</details>
</blockquote>

<blockquote>
<details open>
<summary>视频生成</summary>

- [WorldForge (2025)](papers/04-multimodal/video-generation/WorldForge_2025.md) — Training-free 推理时引导框架，通过 IRR/FLF/DSG 三模块实现视频扩散模型的精确 3D/4D 轨迹控制生成

</details>
</blockquote>

</details>
</blockquote>

</details>

<details>
<summary>🤖 Agents</summary>

> 暂无笔记

</details>

<details open>
<summary>🦾 Embodied AI</summary>

<blockquote>
<details open>
<summary>VLA</summary>

<blockquote>
<details open>
<summary>基础模型</summary>

- [3D-CAVLA (2025)](papers/06-embodied-ai/vla/foundation/3D_CAVLA_2025.md) — CoT 指令分解 + PointNet 深度编码器 + 任务感知 ROI，LIBERO 98.1%，零样本 +8.8%
- [3D-MIX (2026)](papers/06-embodied-ai/vla/foundation/3D_Mix_2026.md) — 9 种 VGGT 融合策略系统对比 + 语义条件化门控即插即用模块，GR00T/π-style 双架构 SIMPLER 平均 +7.0%
- [AimBot (2025)](papers/06-embodied-ai/vla/foundation/AimBot_2025.md) — 瞄准线+准星视觉线索编码 EE 空间状态到像素空间，<1 ms 模型无关，π₀ 真实世界 27→43/50
- [AnchorVLA4D (2026)](papers/06-embodied-ai/vla/foundation/AnchorVLA4D_2026.md) — 首帧锚帧 + 冻结 Any4D 空间编码器，缓解遮挡遗忘和空间失定向，SimplerEnv +13.6%，真实世界 80%
- [BridgeVLA (2025)](papers/06-embodied-ai/vla/foundation/BridgeVLA_2025.md) — 3D 正交投影 + 2D 热力图输入-输出对齐，RLBench 88.2%、3 条轨迹 95.4%
- [ChatVLA (2025)](papers/06-embodied-ai/vla/foundation/ChatVLA_2025.md) — Phased Alignment Training + MoE 双专家，2B 参数统一多模态理解与机器人控制，MMMU 37.4、25 项真实任务超越 OpenVLA
- [CoWVLA (2026)](papers/06-embodied-ai/vla/foundation/CoWVLA_2026.md) — Video VAE 结构-运动解耦 + 潜在运动链推理 + 终端帧预测，统一世界模型与潜在动作，LIBERO 95.6%
- [CronusVLA (2026)](papers/06-embodied-ai/vla/foundation/CronusVLA_2026.md) — Feature Chunking 多帧特征聚合 + DiT 跨帧解码器，SimplerEnv 70.9%、LIBERO 97.0%、SimplerEnv-OR R-Score 86.9
- [DAM-VLA (2026)](papers/06-embodied-ai/vla/foundation/DAM_VLA_2026.md) — 动作路由 + 手臂/夹爪双扩散模型 + 双尺度加权协调
- [DeepVision-VLA (2026)](papers/06-embodied-ai/vla/foundation/DeepVisionVLA_2026.md) — VL-MoT 将 DINOv3 视觉专家与 LLM 深层共享 QKV，AGVP 动作引导视觉剪枝，RLBench 83%、真实世界 91.7%
- [DreamVLA (2025)](papers/06-embodied-ai/vla/foundation/DreamVLA_2025.md) — 预测三类综合世界知识（动态区域/深度/语义）+ 结构化注意力解耦，CALVIN 4.44 SOTA
- [Dexbotic (2025)](papers/06-embodied-ai/vla/foundation/Dexbotic_2025.md) — 开源 VLA 工具箱：统一框架 + Qwen2.5 预训练模型 + 实验驱动开发，最高 +46.2%
- [FAST (2025)](papers/06-embodied-ai/vla/foundation/FAST_2025.md) — DCT + BPE 频域压缩动作 tokenization，解决自回归 VLA 高频任务瓶颈，匹配扩散 π₀ 训练 5× 加速
- [FocusVLA (2026)](papers/06-embodied-ai/vla/foundation/FocusVLA_2026.md) — Cascaded Attention 消除结构性捷径 + Focus Attention 聚焦任务相关视觉，0.5B 参数 LIBERO 98.7% 超越 7B 模型
- [FutureVLA (2026)](papers/06-embodied-ai/vla/foundation/FutureVLA_2026.md) — 双流解耦 + 门控交叉注意力联合视觉运动预测，潜在对齐迁移时序先验，SimplerEnv 80.1%、真实机器人超 π₀ 达 26.7%
- [GR-3 (2025)](papers/06-embodied-ai/vla/foundation/GR3_2025.md) — 4B VLA：VL 协同训练 + VR 人类轨迹 10-shot 适配 + Task Status 指令跟随，全面超越 π₀
- [MoH (2025)](papers/06-embodied-ai/vla/foundation/MoH_2025.md) — 多 Horizon 动作块并行融合 + 轻量门控 + 动态推理，π₀.₅+MoH LIBERO 99%
- [MemoryVLA (2025)](papers/06-embodied-ai/vla/foundation/MemoryVLA_2025.md) — 感知-认知双流记忆库建模长时域依赖，SimplerEnv-Bridge +14.6、LIBERO 96.5%、真实世界时序 +26
- [MMaDA-VLA (2026)](papers/06-embodied-ai/vla/foundation/MMaDA_VLA_2026.md) — 原生离散扩散统一多模态 VLA，并行去噪生成目标观测+动作块，LIBERO 98.0%、CALVIN 4.78 全面 SOTA
- [OptimusVLA (2026)](papers/06-embodied-ai/vla/foundation/OptimusVLA_2026.md) — 双记忆增强 VLA：GPM 任务级先验检索 + LCM 时序一致性，LIBERO 98.6%、2.9× 加速
- [OTTER (2025)](papers/06-embodied-ai/vla/foundation/OTTER_2025.md) — 冻结 CLIP + 文本感知视觉特征提取实现零样本泛化，4 种原语未见任务 77%，Octo/OpenVLA 几乎为 0%
- [ProgressVLA (2026)](papers/06-embodied-ai/vla/foundation/ProgressVLA_2026.md) — 进度估计器 + 世界模型 + classifier guidance 引导扩散策略，CALVIN 3.73、LIBERO 84.5%、真实世界 76%（Octo 23%）
- [π₀ (2024)](papers/06-embodied-ai/vla/foundation/pi0_2024.md) — Flow Matching VLA 基础模型
- [π₀.₅ (2025)](papers/06-embodied-ai/vla/foundation/pi05_2025.md) — 异构协同训练 + 分层推理
- [SF (2025)](papers/06-embodied-ai/vla/foundation/SF_2025.md) — 隐式空间表征对齐（VGGT），推理零开销，LIBERO 98.5%，训练 3.8× 加速
- [SpatialVLA (2025)](papers/06-embodied-ai/vla/foundation/SpatialVLA_2025.md) — Ego3D 位置编码 + 自适应高斯动作网格（3 token/step），1.1M 数据预训练 3.5B 零样本超 55B RT-2-X，20 Hz
- [SPR (2026)](papers/06-embodied-ai/vla/foundation/SPR_2026.md) — See-Plan-Rewind 空间子目标进度感知 + 自主回退恢复，LIBERO 91.8%，LIBERO-Plus OOD 退化仅 18.8%
- [TCoT (2026)](papers/06-embodied-ai/vla/foundation/TCoT_2026.md) — 全局/局部轨迹思维链 + GLSR 失败恢复，多任务促进跨任务共享，LIBERO 83.3%，真实世界 +28%
- [TGM-VLA (2026)](papers/06-embodied-ai/vla/foundation/TGM_VLA_2026.md) — 关键帧采样优化 + 颜色反转 + 任务引导 Mixup，RLBench 90.5%
- [UniVLA (2025)](papers/06-embodied-ai/vla/foundation/UniVLA_2025.md) — 任务中心潜在动作两阶段解耦 + 跨具身无标注视频预训练，1/20 算力超 OpenVLA，LIBERO 95.2%、真实世界 81.7%
- [VP-VLA (2026)](papers/06-embodied-ai/vla/foundation/VP_VLA_2026.md) — 双系统解耦 + VLM 事件驱动任务分解 + SAM3 视觉提示空间锚点 + 接地辅助损失，RoboCasa +5%、SimplerEnv +8.3%

</details>
</blockquote>

<blockquote>
<details open>
<summary>高效推理</summary>

- [BitVLA (2025)](papers/06-embodied-ai/vla/efficient/BitVLA_2025.md) — 首个全参数三值化 VLA，蒸馏感知训练量化 ViT 至 1.58-bit，LIBERO 94.8%、显存仅 1.4GB
- [EfficientVLA (2025)](papers/06-embodied-ai/vla/efficient/EfficientVLA_2025.md) — 结构化 training-free 加速：层剪枝 + Token 选择 + 扩散步缓存，1.93× 加速
- [HeiSD (2026)](papers/06-embodied-ai/vla/efficient/HeiSD_2026.md) — 混合推测解码（Drafter + Retrieval SD）+ 运动学融合指标自动切换，LIBERO 2.45×、真实世界 2.41× 加速
- [LAC (2026)](papers/06-embodied-ai/vla/efficient/LAC_2026.md) — 可学习自适应 Token 缓存加速 VLA
- [PD-VLA (2025)](papers/06-embodied-ai/vla/efficient/PD_VLA_2025.md) — Jacobi 并行解码加速 Action Chunking VLA，不改模型不训练，2.52× 频率提升
- [SD-VLA (2026)](papers/06-embodied-ai/vla/efficient/SD_VLA_2026.md) — 静态-动态解耦实现长时程高效 VLA
- [RLRC (2025)](papers/06-embodied-ai/vla/efficient/RLRC_2025.md) — 结构化剪枝 + SFT/RL 恢复 + 量化，8× 显存压缩
- [RTC (2025)](papers/06-embodied-ai/vla/efficient/RTC_2025.md) — 异步动作块执行建模为修复问题：冻结前缀 + 引导修复 + 软掩码，Training-Free 实时 VLA，π₀.₅ 快 20%
- [VLA-Cache (2025)](papers/06-embodied-ai/vla/efficient/VLA_Cache_2025.md) — 训练无关跨帧 Token 缓存加速 VLA
- [VLA-Pruner (2025)](papers/06-embodied-ai/vla/efficient/VLA_Pruner_2025.md) — 双层 Token 剪枝（语义级 + 动作级注意力时序平滑）+ mRMR 选择，50% 剪枝率反超原模型

</details>
</blockquote>

<blockquote>
<details open>
<summary>推理增强</summary>

- [UAOR (2026)](papers/06-embodied-ai/vla/inference/UAOR_2026.md) — Action Entropy 检测不确定性 + 观测重注入 FFN，无训练即插即用增强 VLA

</details>
</blockquote>

<blockquote>
<details open>
<summary>RL 后训练</summary>

- [ARM (2026)](papers/06-embodied-ai/vla/rl/ARM_2026.md) — Tri-state 相对优势标注 + MIMO Transformer 奖励模型 + 长度自适应 AW-BC，长程叠毛巾 99.4%
- [ConRFT (2025)](papers/06-embodied-ai/vla/rl/ConRFT_2025.md) — 一致性策略统一离线-在线 RL 微调 VLA，真实世界 96.3% 成功率
- [DiffRL Data (2025)](papers/06-embodied-ai/vla/rl/DiffRL_Data_2025.md) — 扩散 RL 生成高质量低方差轨迹，纯合成数据训练 VLA 超越人类演示
- [FPO++ (2026)](papers/06-embodied-ai/vla/rl/FPO_2026.md) — CFM 损失差值近似似然比 + 非对称信任域，flow 策略 on-policy RL
- [GigaBrain-0.5M* (2026)](papers/06-embodied-ai/vla/rl/GigaBrain_2026.md) — 世界模型预测未来状态+价值条件化 VLA，RAMP 比 RECAP 提升 30%
- [GRAPE (2025)](papers/06-embodied-ai/vla/rl/GRAPE_2025.md) — 轨迹级偏好优化 + VLM 代价函数，plug-and-play 提升 VLA 泛化
- [GR-RL (2025)](papers/06-embodied-ai/vla/rl/GR_RL_2025.md) — 多阶段流水线特化通才 VLA 为精密操作专家
- [LRM (2026)](papers/06-embodied-ai/vla/rl/LRM_2026.md) — 三维度帧级在线奖励引擎（时序对比/进度/完成），零样本驱动 PPO 超越 RoboReward 和 ROBOMETER
- [MoRE (2025)](papers/06-embodied-ai/vla/rl/MoRE_2025.md) — Mixture of LoRA Experts + 离线 Q-learning，四足多任务 VLA 成功率提升 36%
- [π₀.₆* (2025)](papers/06-embodied-ai/vla/rl/pi06star_2025.md) — RECAP 优势条件化离线 RL 训练 VLA
- [π-StepNFT (2026)](papers/06-embodied-ai/vla/rl/pi_StepNFT_2026.md) — 无 Critic 无似然在线 RL：SDE 探索 + 逐步监督 + 对比排序，ManiSkill OOD 超 PPO 11.1%
- [πRL (2025)](papers/06-embodied-ai/vla/rl/piRL_2025.md) — Flow-Noise/Flow-SDE 两条路线解决 flow VLA 的 log-likelihood 难题，PPO 微调 π₀/π₀.₅，LIBERO 97.6%/98.3%
- [PLD (2026)](papers/06-embodied-ai/vla/rl/PLD_2026.md) — 残差 RL 专家探索 + 基础策略探针混合蒸馏实现 VLA 自改进
- [PTR (2026)](papers/06-embodied-ai/vla/rl/PTR_2026.md) — 无奖励保守离线后训练：post-action identification 评分 + 保守权重，跨构型 Generalist +13.8 pp
- [ReWiND (2025)](papers/06-embodied-ai/vla/rl/ReWiND_2025.md) — 语言条件化奖励模型 + Video Rewind，无需新演示语言引导 RL 学新任务
- [RISE (2026)](papers/06-embodied-ai/vla/rl/RISE_2026.md) — 组合式世界模型 + 想象空间 RL
- [Robo-Dopamine (2025)](papers/06-embodied-ai/vla/rl/RoboDopamine_2025.md) — 35M 多视角 GRM + Hop-based 进度归一化 + 策略不变奖励塑形，One-shot 适配 150 次交互达 95%
- [ROBOMETER (2026)](papers/06-embodied-ai/vla/rl/ROBOMETER_2026.md) — 帧级进度 + 轨迹偏好双目标训练通用机器人奖励模型
- [RoboReward (2026)](papers/06-embodied-ai/vla/rl/RoboReward_2026.md) — 反事实重标注 + 时序裁剪合成负样本，微调 Qwen3-VL 为 episode 级通用奖励模型，22 个 VLM 排名第一
- [RL-Co (2026)](papers/06-embodied-ai/vla/rl/RL_Co_2026.md) — RL-based sim-real co-training，仿真 RL + 真实数据 SFT 正则
- [RLinf (2025)](papers/06-embodied-ai/vla/rl/RLinf_2025.md) — M2Flow 大规模 RL 训练系统
- [RLinf-USER (2026)](papers/06-embodied-ai/vla/rl/RLinf_USER_2026.md) — 真实世界在线策略学习统一系统
- [RLinf-VLA (2025)](papers/06-embodied-ai/vla/rl/RLinf_VLA_2025.md) — 统一高效的 VLA+RL 训练框架
- [RL-VLA Survey (2025)](papers/06-embodied-ai/vla/rl/RL_VLA_Survey_2025.md) — 综述：RL 后训练 VLA 的架构、训练范式、部署与评测全景图
- [RLVLA (2025)](papers/06-embodied-ai/vla/rl/RLVLA_2025.md) — 系统性实证：RL 在语义和执行维度显著提升 VLA 泛化
- [RPD (2025)](papers/06-embodied-ai/vla/rl/RPD_2025.md) — PPO + MSE 蒸馏将 VLA 通才知识提炼为紧凑 RL 专家
- [SAC Flow (2026)](papers/06-embodied-ai/vla/rl/SAC_Flow_2026.md) — Flow Policy 序列建模 + off-policy RL
- [SC-VLA (2026)](papers/06-embodied-ai/vla/rl/SC_VLA_2026.md) — 稀疏世界想象 + 残差 RL 在线修正，内生奖励自改进
- [SimpleVLA-RL (2025)](papers/06-embodied-ai/vla/rl/SimpleVLA_RL_2025.md) — 二元结果奖励 + GRPO 探索增强，LIBERO 99.1%，1 条演示 RL 超越全量 SFT
- [SRPO (2025)](papers/06-embodied-ai/vla/rl/SRPO_2025.md) — 自参照策略优化：世界模型隐表征 progress-wise 奖励
- [TACO (2025)](papers/06-embodied-ai/vla/rl/TACO_2025.md) — 反探索 test-time scaling：轻量伪计数器选择 in-support 动作
- [TGRPO (2025)](papers/06-embodied-ai/vla/rl/TGRPO_2025.md) — 双层组相对策略优化：LLM 稠密奖励 + 步级/轨迹级优势融合
- [TOPReward (2026)](papers/06-embodied-ai/vla/rl/TOPReward_2026.md) — Token logits 零样本奖励：True 概率作进度信号，Qwen3-VL 0.947 VOC，ManiRewardBench
- [TwinRL (2026)](papers/06-embodied-ai/vla/rl/TwinRL_2026.md) — 数字孪生驱动的真实世界机器人 RL
- [VLAC (2025)](papers/06-embodied-ai/vla/rl/VLAC_2025.md) — 统一 Actor-Critic + pairwise progress 稠密奖励，真实世界 RL 自改进
- [VLA-RFT (2025)](papers/06-embodied-ai/vla/rl/VLA_RFT_2025.md) — 视频世界模型 + Verified Reward + GRPO，400 步超越 SFT
- [VLA-RL (2025)](papers/06-embodied-ai/vla/rl/VLA_RL_2025.md) — 在线 PPO 微调自回归 VLA
- [WMPO (2025)](papers/06-embodied-ai/vla/rl/WMPO_2025.md) — 隐空间世界模型 imagination RL 后训练 VLA
- [World-VLA-Loop (2026)](papers/06-embodied-ai/vla/rl/World_VLA_Loop_2026.md) — 视频世界模型与 VLA 策略闭环联合优化，SANS 近成功数据 + 迭代 RL
- [WoVR (2026)](papers/06-embodied-ai/vla/rl/WoVR_2026.md) — 幻觉感知世界模型 RL

</details>
</blockquote>

</details>
</blockquote>

<blockquote>
<details open>
<summary>World Models</summary>

- [BridgeV2W (2025)](papers/06-embodied-ai/world-models/BridgeV2W_2025.md) — Embodiment Mask + ControlNet 像素空间动作注入，跨构型统一世界模型
- [Fast-WAM (2026)](papers/06-embodied-ai/world-models/FastWAM_2026.md) — 训练时视频协同目标是 WAM 性能主因、测试时未来想象非必要，跳过视频生成 190 ms 推理，RoboTwin 91.8%、LIBERO 97.6%
- [Kinema4D (2026)](papers/06-embodied-ai/world-models/Kinema4D_2026.md) — 运动学 4D pointmap 控制 + DiT 联合生成 RGB+Pointmap，20 万条 4D 数据训练，首次零样本真实世界迁移
- [WorldVLA (2025)](papers/06-embodied-ai/world-models/WorldVLA_2025.md) — 自回归统一 VLA + 世界模型，Action Attention Mask 解决 Chunking 误差累积

</details>
</blockquote>

<blockquote>
<details open>
<summary>Imitation Learning</summary>

- [EC-Flow (2025)](papers/06-embodied-ai/imitation-learning/EC_Flow_2025.md) — 具身中心光流 + 目标图像对齐 + URDF 运动学，5 条无动作标注视频学操作，遮挡 +62%、柔性 +45%、非位移 +80%

</details>
</blockquote>

</details>

<details open>
<summary>⚡ Efficiency</summary>

> 暂无笔记

</details>

<details>
<summary>🔍 RAG & Knowledge</summary>

> 暂无笔记

</details>

<details>
<summary>📊 Evaluation</summary>

- [MME (2024)](papers/09-evaluation-and-benchmarks/MME_2024.md) — 首个 MLLM 综合评测基准：14 子任务覆盖感知与认知，Yes/No 人工指令对实现精确量化，30 模型评测揭示四大共性问题

</details>

<details open>
<summary>🎯 Reinforcement Learning</summary>

- [DiffusionNFT (2025)](papers/10-reinforcement-learning/DiffusionNFT_2025.md) — 前向过程扩散 RL：正/负对比隐式策略改进 + flow matching 目标，无需似然/CFG，效率比 FlowGRPO 高 3-25 倍
- [FLAC (2026)](papers/10-reinforcement-learning/FLAC_2026.md) — GSB 框架下的无似然 MaxEnt RL：动能正则化 flow/diffusion 策略，NFE=2 达到 DIME（NFE=16）水平

</details>

---

## 🚀 本地部署

### 环境要求

- [Git](https://git-scm.com/downloads)
- [Node.js](https://nodejs.org/) >= 18（推荐 LTS 版本，npm 随 Node.js 一起安装）

如果尚未安装 Node.js，根据你的操作系统选择对应方式：

```bash
# macOS（使用 Homebrew）
brew install node

# Ubuntu / Debian
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Windows
# 前往 https://nodejs.org 下载 LTS 安装包，双击安装即可
```

安装完成后验证：

```bash
node -v   # 应输出 v18.x.x 或更高
npm -v    # 应输出 9.x.x 或更高
```

### 安装与启动

```bash
# 1. 克隆仓库
git clone git@github.com:jiabingyang01/llm-paper-notes.git
cd llm-paper-notes

# 2. 安装依赖
npm install

# 3. 启动本地开发服务器（支持热更新）
npm run docs:dev
```

启动后终端会输出本地地址（默认 `http://localhost:5173`），浏览器打开即可预览。编辑任何 `.md` 文件后页面会自动刷新。

### 构建与预览

```bash
# 构建生产版本（输出到 .vitepress/dist）
npm run docs:build

# 本地预览构建产物
npm run docs:preview
```

### 部署到线上

本站使用 GitHub Pages 自动部署。推送到 `main` 分支后，GitHub Actions 会自动构建并发布到 [llm-paper-notes.jiabingyang.cn](https://llm-paper-notes.jiabingyang.cn/)。

如需手动部署到vercel，将 `.vitepress/dist` 目录部署为静态站点即可。

---

## 📝 如何添加新笔记

```bash
# 1. 复制模板
cp templates/paper_template.md papers/<分类>/论文名_年份.md

# 2. 按模板结构写笔记（公式用 LaTeX：$...$ 行内，$$...$$ 行间）

# 3. 提交
git add .
git commit -m "add: 论文名 年份 论文解读"
git push
```

**命名规范**：`论文简称_年份.md`，如 `RISE_2026.md`、`DPO_2023.md`

详细模板见 → [templates/paper_template.md](templates/paper_template.md)

---

## 📄 License

本仓库笔记内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 协议。欢迎转载，请注明出处。
