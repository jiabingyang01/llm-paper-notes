# 04 Multimodal

多模态模型：VLM（GPT-4V、LLaVA）、视频理解、语音模型、多模态生成、统一架构等。

---

## VLM — 幻觉缓解

| 论文 | 关键词 | 时间 |
| --- | --- | --- |
| [AGLA](vlm/hallucination/AGLA_2025.md) | GradCAM、全局-局部注意力组装、自适应遮蔽、Training-Free | 2024.06 |
| [CSR](vlm/hallucination/CSR_2024.md) | 校准自奖励、CLIP Score、迭代 DPO、模态对齐、Self-Rewarding | 2024 |
| [DLC](vlm/hallucination/DLC_2025.md) | 动态 Logits 校准、CLIP 探针、相对视觉优势、自适应引导、Training-Free | 2025 |
| [EFUF](vlm/hallucination/EFUF_2024.md) | 细粒度遗忘、CLIP 数据筛选、梯度上升、三重损失 | 2024.02 |
| [FarSight](vlm/hallucination/FarSight_2025.md) | 注意力寄存器、因果掩码优化、位置感知编码、Training-Free、Image+Video | 2025 |
| [HALC](vlm/hallucination/HALC_2024.md) | FOV 对比解码、JSD 双向对比、视觉匹配 Beam Search、Plug-and-Play | 2024 |
| [HIME](vlm/hallucination/HIME_2026.md) | HIS、层自适应模型编辑、零空间投影、Training-Free | 2026.02 |
| [HIO](vlm/hallucination/HIO_2024.md) | 反转 BT 模型、Evil LVLM 对比解码、多幻觉诱导、Logit 约束 | 2024.05 |
| [ICD](vlm/hallucination/ICD_2024.md) | 指令对比解码、多模态对齐不确定性、自适应截断、Training-Free | 2024.03 |
| [LessIsMore](vlm/hallucination/LessIsMore_2024.md) | EOS 决策、Selective EOS Supervision、数据过滤、训练目标修改 | 2024.02 |
| [LogicCheckGPT](vlm/hallucination/LogicCheckGPT_2024.md) | 逻辑闭环、逻辑一致性、Training-Free、Plug-and-Play | 2024.02 |
| [mDPO](vlm/hallucination/mDPO_2024.md) | 条件偏好优化、图像对比偏好、奖励锚定、无条件偏好问题 | 2024.06 |
| [MemVR](vlm/hallucination/MemVR_2025.md) | FFN Key-Value Memory、视觉回溯、不确定性触发、Training-Free、Plug-and-Play | 2025.05 |
| [MMHalSnowball](vlm/hallucination/MMHalSnowball_2024.md) | 幻觉雪球效应、残差视觉解码、自适应分布混合、Training-Free | 2024.07 |
| [OPERA](vlm/hallucination/OPERA_2024.md) | 注意力聚合模式、Over-Trust Penalty、Beam Search 回溯、Training-Free | 2024 |
| [REVERIE](vlm/hallucination/REVERIE_2024.md) | 反思微调、正负 Rationale、细粒度推理监督、REVERIE 数据集 | 2024.07 |
| [SENTINEL](vlm/hallucination/SENTINEL_2025.md) | 句子级早期干预、域内偏好学习、C-DPO、交叉验证 | 2025.07 |
| [SIMA](vlm/hallucination/SIMA_2024.md) | 自生成响应、上下文自评估、三视觉指标、DPO、Self-Improvement | 2024.05 |
| [STIC](vlm/hallucination/STIC_2024.md) | 自训练、图像理解、描述注入微调、正则化 DPO、Self-Training | 2024.05 |
| [VACoDe](vlm/hallucination/VACoDe_2024.md) | 视觉增强选择、Softmax Distance、对比解码、Training-Free、Plug-and-Play | 2024 |
| [VGA](vlm/hallucination/VGA_2024.md) | GUI 理解、Referent Method、两阶段微调、Image-Centric | 2024.06 |
| [VisFlow](vlm/hallucination/VisFlow_2025.md) | 双层注意力干预、Visual Sink/Salient Token、Head 分类抑制、Training-Free | 2025.06 |

## VLM — Token 压缩

| 论文 | 关键词 | 时间 |
| --- | --- | --- |
| [Token Pruning Survey](vlm/efficiency/TokenPruningSurvey_2025.md) | Token Pruning、位置偏差、重要性 vs. 冗余性、训练感知压缩、评估方法论 | 2025.02 |
