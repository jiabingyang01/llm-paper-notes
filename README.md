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

### 🦾 Embodied AI — VLA 基础模型

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [π₀](papers/06-embodied-ai/vla/foundation/pi0_2024.md) | 用 Flow Matching 替代自回归生成动作，构建首个能完成高频灵巧操作的通用 VLA 基础模型 | Flow Matching VLA、Action Expert、跨构型预训练 | 2024.10 |
| [π₀.₅](papers/06-embodied-ai/vla/foundation/pi05_2025.md) | 通过异构多源数据协同训练和分层推理，首次实现端到端 VLA 在全新家庭环境中执行长时域灵巧操作 | 异构协同训练、分层推理、开放世界泛化 | 2025.04 |

### 🦾 Embodied AI — VLA / RL 后训练

| 论文 | 一句话概括 | 关键词 | 时间 |
| --- | --- | --- | --- |
| [RISE](papers/06-embodied-ai/vla/rl/RISE_2026.md) | 用组合式世界模型在想象空间做 RL，让 VLA 不靠真实交互就能自我改进 | 世界模型、Imagination RL、VLA 自改进 | 2026.02 |
| [SAC Flow](papers/06-embodied-ai/vla/rl/SAC_Flow_2026.md) | 把 Flow Policy 重新理解为序列模型，用 GRU/Transformer 重参数化解决 RL 梯度不稳定问题 | Flow Policy、序列建模、SAC、off-policy RL | 2026.01 |
| [VLA-RL](papers/06-embodied-ai/vla/rl/VLA_RL_2025.md) | 将机器人操作建模为多模态多轮对话，用 PPO 在线 RL 微调自回归 VLA，配合 Robotic PRM 解决稀疏奖励 | 在线 PPO、Robotic PRM、自回归 VLA + RL | 2025.05 |

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

# 4. 本地部署
npm run docs:dev
```

**命名规范**：`论文简称_年份.md`，如 `RISE_2026.md`、`DPO_2023.md`

详细模板见 → [templates/paper_template.md](templates/paper_template.md)

---

## 📄 License

本仓库笔记内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 协议。欢迎转载，请注明出处。
