# LLM Paper Notes — 项目工作指南

## 项目概述

VitePress 构建的论文精读笔记站，聚焦大语言模型及相关领域。
- 在线地址：https://llm-paper-notes.jiabingyang.cn/
- 仓库根目录：`/DATA/disk0/yjb/projects/personal/llm-paper-notes/`

## 目录结构

```
papers/
├── index.md                          # 论文索引页（全部论文的表格）
├── 01-foundation-models/
├── 02-alignment-and-safety/
├── 03-reasoning/
├── 04-multimodal/
├── 05-agents/
├── 06-embodied-ai/
│   └── vla/
│       ├── foundation/               # VLA 基础模型（π₀、π₀.₅ 等）
│       ├── efficient/                # VLA 高效推理（LAC、VLA-Cache 等）
│       └── rl/                       # VLA RL 后训练（RISE、VLA-RL 等）
├── 07-efficiency/
├── 08-rag-and-knowledge/
└── 09-evaluation-and-benchmarks/
templates/
└── paper_template.md                 # 笔记模板
.vitepress/
└── config.mts                        # VitePress 配置（含侧边栏）
README.md                             # 仓库首页
index.md                              # 网站首页
```

## 添加新论文笔记的完整流程

### 第 1 步：确定分类和文件名

- 文件命名：`论文简称_年份.md`，如 `RISE_2026.md`、`pi06star_2025.md`
- 放入最核心的分类子目录下

### 第 2 步：撰写笔记

参考 `templates/paper_template.md` 的结构，以及已有笔记的风格（如 `papers/06-embodied-ai/vla/rl/RISE_2026.md`）。

笔记应包含：
1. 标题 + 元信息（作者、机构、时间、链接）
2. 一句话总结
3. 问题与动机（要解决什么问题、现有方法为何不足）
4. 预备知识（必要的前置概念和公式）
5. 核心方法（详细拆解，含关键公式 LaTeX 推导和直觉解释）
6. 实验结果（主要表格、关键数字、消融实验）
7. 局限性与未来方向
8. 个人思考（与其他论文的联系、洞察）
9. 参考（最相关的几篇论文）

公式用 LaTeX：`$...$` 行内，`$$...$$` 行间。

### 第 3 步：更新 3 个索引文件

每添加一篇新笔记，需要同步更新以下 3 个文件：

#### (A) `.vitepress/config.mts` — 侧边栏

在对应分类的 `items` 数组中添加条目：

```typescript
{ text: '论文简称 (年份)', link: '/papers/分类路径/文件名（无.md）' },
```

例如在 RL 后训练分类下：
```typescript
{ text: 'π₀.₆* (2025)', link: '/papers/06-embodied-ai/vla/rl/pi06star_2025' },
```

侧边栏结构位于 `themeConfig.sidebar['/papers/']` 中。

#### (B) `papers/index.md` — 论文索引表

在对应分类的表格末尾添加一行：

```markdown
| [论文简称](/papers/分类路径/文件名) | 一句话概括 | 关键词1、关键词2、关键词3 | YYYY.MM |
```

#### (C) `README.md` — 仓库首页论文列表

在对应分类的 `<details>` 块中添加：

```markdown
- [论文简称 (年份)](papers/分类路径/文件名.md) — 一句话简述
```

注意 README.md 中的链接是相对路径且**带 .md 后缀**，而 config.mts 和 index.md 中**不带 .md**。

### 第 4 步（可选）：更新首页 `index.md`

如果新论文需要作为首页"开始阅读"的推荐论文，修改 `index.md` 中 `hero.actions` 的第一个链接。

## 分类体系

| 编号 | 分类 | 覆盖方向 |
| --- | --- | --- |
| 01 | Foundation Models | 架构设计、预训练方法、Scaling Laws |
| 02 | Alignment & Safety | RLHF、DPO、Constitutional AI |
| 03 | Reasoning | Chain-of-Thought、数学推理、代码生成 |
| 04 | Multimodal | VLM、图像/视频理解、语音 |
| 05 | Agents | Tool Use、Web Agent、多 Agent 协作 |
| 06 | Embodied AI | VLA、世界模型、机器人 RL、模仿学习 |
| 07 | Efficiency | 量化、蒸馏、剪枝、推测解码 |
| 08 | RAG & Knowledge | RAG 架构、向量检索、知识图谱 |
| 09 | Evaluation | Benchmark 设计、评估方法论 |

一篇论文可以在多个分类的索引中出现，但 .md 文件只存一份，放在最核心的分类下。

## 06 Embodied AI 的子分类结构

当前 Embodied AI 下设：

```
VLA/
├── foundation/    # VLA 基础模型
├── efficient/     # VLA 高效推理
└── rl/            # VLA RL 后训练
World Models/      # 世界模型（暂无笔记）
```

如果新论文不属于已有子分类，可以新建目录并更新 config.mts 中的侧边栏结构。

## 写作风格约定

- 语言：中文为主，术语保留英文（如 flow matching、advantage conditioning）
- 公式：完整推导，附带直觉解释（"用大白话说"）
- 表格：使用 Markdown 表格呈现实验数据
- 对比：在"个人思考"中主动与项目中已有的其他论文建立联系
- 不使用 emoji（除非用户要求）
