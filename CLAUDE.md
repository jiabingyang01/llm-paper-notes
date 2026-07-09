# LLM Paper Notes — 项目工作指南

## 项目概述

VitePress 构建的论文精读笔记站，聚焦大语言模型及相关领域。
- 在线地址：https://llm-paper-notes.jiabingyang.cn/
- 仓库根目录：`/data/yjb/projects/personal/llm-paper-notes/`
- 部署方式：push 到 `main` 后由 GitHub Actions 自动构建并部署到 GitHub Pages（`.github/workflows/deploy.yml`），无需手动部署

## 目录结构

```
papers/
├── index.md                          # 论文索引页（全部论文的表格）
├── 01-foundation-models/
├── 02-alignment-and-safety/
├── 03-reasoning/
├── 04-multimodal/
│   ├── vlm/                          # VLM（含 efficiency/、hallucination/ 子方向）
│   └── video-generation/             # 视频生成
├── 05-agents/
├── 06-embodied-ai/
│   ├── vla/
│   │   ├── foundation/               # VLA 基础模型 / 训练范式
│   │   ├── perception/               # VLA 感知增强
│   │   ├── reasoning/                # VLA 推理与规划
│   │   ├── efficient/                # VLA 高效推理
│   │   └── rl/                       # VLA RL 后训练
│   ├── world-models/                 # 纯世界模型 / 模拟器（BridgeV2W、Kinema4D、MIND-V）
│   ├── world-action-models/          # 世界动作模型 WAM（LDA-1B、FastWAM、WorldVLA、SpatialVAM、WAM 综述）
│   ├── imitation-learning/           # 模仿学习（EC-Flow 等）
│   └── data-synthesis/               # 数据合成：人类演示 / 世界模型 → 机器人可训练数据（MimicDreamer、Wh0）
├── 07-efficiency/
├── 08-rag-and-knowledge/
├── 09-evaluation-and-benchmarks/
└── 10-reinforcement-learning/        # RL 基础方法（DiffusionNFT、FLAC 等）
templates/
└── paper_template.md                 # 笔记模板（新笔记的起点，与现行笔记风格保持同步）
.vitepress/
└── config.mts                        # VitePress 配置（含侧边栏）
README.md                             # 仓库首页
index.md                              # 网站首页
```

## 添加新论文笔记的完整流程

### 第 1 步：确定分类和文件名

- 文件命名：`论文简称_年份.md`，如 `RISE_2026.md`、`pi06star_2025.md`；简称中的空格、连字符、点号一律替换为下划线（如 `3D_CAVLA_2025.md`）；年份取 arXiv 首次发布年份
- 放入最核心的分类子目录下，如果你觉得需要创建新的分类子目录，请自行创建；如果笔记已经存在，不用创建，告诉我

### 第 2 步：撰写笔记

结构以 `templates/paper_template.md` 为准，风格参考近期笔记（如 `papers/06-embodied-ai/world-action-models/FastWAM_2026.md`）。

**撰写前必须阅读 PDF 原文**：绝对不要根据论文标题 + 领域先验去"合理化"地编造方法骨架、实验设置、baseline、数字、作者/机构。曾经出现过整篇 WMPO 笔记的作者、机构、方法（latent+PPO vs 实际 pixel+GRPO）、实验（LIBERO/SimplerEnv vs 实际 Mimicgen）全部是幻觉的事故，还会传染到其他笔记的对比描述。如果用户没有附 PDF，必须通过 arXiv abs/pdf 页面或项目主页取得原文要点后再写，宁可推迟也不要凭想象写。

章节骨架（与模板一致）：

1. 标题：`# 论文简称：中文翻译标题`
2. 元信息 blockquote：论文英文原题、作者、机构、发布时间、发表状态、链接行、分类标签
3. 一句话总结
4. 一、问题与动机（要解决什么问题、现有方法为何不足）
5. 二、预备知识（必要的前置概念和公式；不需要时整节省略，后续编号顺延）
6. 三、核心方法（详细拆解，含关键公式 LaTeX 推导和直觉解释）
7. 四、实验结果（主要表格、关键数字、消融实验）
8. 五、局限性与未来方向
9. 六、个人思考（与其他论文的联系、洞察）
10. 参考（最相关的几篇论文，格式：`- **论文名**（作者 et al., 年份，arXiv ID）：一句话关系`）

**发表状态**：如果用户没有显式提供，需联网检索论文是否已被期刊或会议录用（OpenReview、会议 accepted paper list、DBLP、按标题搜索均可）。已录用则注明 venue（如 `ICLR 2026`、`CoRL 2025 Oral`），查不到录用信息则写 `未录用`。

公式用 LaTeX：`$...$` 行内，`$$...$$` 行间。

### 第 3 步：更新 4 个索引文件

每添加一篇新笔记，需要同步更新以下 4 个文件：

#### (A) `.vitepress/config.mts` — 侧边栏

在对应分类的 `items` 数组中 **按方法名首字母 A-Z 顺序** 插入条目：

```typescript
{ text: '论文简称 (年份)', link: '/papers/分类路径/文件名（无.md）' },
```

例如在 RL 后训练分类下：
```typescript
{ text: 'π₀.₆* (2025)', link: '/papers/06-embodied-ai/vla/rl/pi06star_2025' },
```

侧边栏结构位于 `themeConfig.sidebar['/papers/']` 中。

#### (B) `papers/index.md` — 论文索引表

在对应分类的表格中 **按方法名首字母 A-Z 顺序** 插入一行：

```markdown
| [论文简称](/papers/分类路径/文件名) | 一句话概括 | 关键词1、关键词2、关键词3 | YYYY.MM |
```

#### (C) `README.md` — 仓库首页论文列表

在对应分类的 `<details>` 块中 **按方法名首字母 A-Z 顺序** 插入：

```markdown
- [论文简称 (年份)](papers/分类路径/文件名.md) — 一句话简述
```

注意 README.md 中的链接是相对路径且 **带 .md 后缀**，而 config.mts 和 index.md 中 **不带 .md**。

#### (D) `papers/<分类>/README.md` — 分类子索引页

每个分类目录下有各自的 `README.md`（如 `papers/06-embodied-ai/README.md`），作为该分类的独立索引页面。在对应子分类的表格中 **按方法名首字母 A-Z 顺序** 插入一行：

```markdown
| [论文简称](vla/rl/文件名.md) | 关键词1、关键词2、关键词3 | 年份 |
```

注意此处链接是 **相对于当前分类目录** 的路径，且 **带 .md 后缀**。

### 第 4 步（可选）：更新首页 `index.md`

如果新论文需要作为首页"开始阅读"的推荐论文，修改 `index.md` 中 `hero.actions` 的第一个链接。

### 第 5 步：构建验证

写完笔记和索引后，本地构建一次以捕获渲染错误（**push 前务必做**——GitHub Pages 与 Vercel 都跑同一个 vitepress build，构建失败两边都会红，站点停在上一次成功的旧版本）：

```bash
export PATH="/data/yjb/miniconda3/envs/ageshub/bin:$PATH"   # 本机 node 在 ageshub conda 环境（v20），系统 PATH 里没有
npm install        # 仅首次（当前机器默认没有 node_modules）
npm run docs:build
```

- build 能捕获 Vue 模板解析错误（裸 `<`、`{{`，见下方渲染陷阱）和 markdown 编译问题
- **build 不检查死链**（`config.mts` 设置了 `ignoreDeadLinks: true`）：4 个索引文件中新增链接的路径、大小写、是否带 `.md` 后缀必须人工核对
- 本地预览：`npm run docs:dev`
- **本文件（CLAUDE.md）、README.md、templates/ 已在 `config.mts` 的 `srcExclude` 中排除**，不会被当作页面编译。这是必需的：本文件"渲染陷阱"一节会写 `{{`、`<` 作反面示例，若被 Vue 当页面编译会报 `Interpolation end sign was not found` 并使整个构建失败。新增此类 meta / 说明文件时记得一并加进 `srcExclude`。

## VitePress / 数学渲染陷阱（写笔记必读）

以下每一条都来自实际踩坑，违反会导致页面渲染错乱或构建失败：

1. **元信息 blockquote 必须有空 `>` 行**：每条 `>` 行之间插入一个空的 `>` 行，否则 VitePress 将所有内容渲染为同一段落。正确格式：`> **论文**：...\n>\n> **作者**：...`
2. **禁止在代码块（\`\`\`）中书写算法伪代码或流程描述**：伪代码中的数学符号（下标、上标、希腊字母、矩阵等）必须使用 LaTeX 渲染，应使用 blockquote + 有序列表 + 行内 LaTeX 的形式呈现，确保所有变量和公式在页面上正确显示
3. **表格单元格内的行内公式禁用裸 `|`**：`|` 会被 Markdown 表格语法当作分列符，把公式拦腰截断。条件概率、条件期望在表格内一律写 `\mid`（如 `$\pi(a \mid s)$`）；不要用 `\|`（会渲染成双竖线范数符号）
4. **正文避免裸尖括号**：`<s, a>` 这类写法会被 Vue 当作组件标签，导致构建报错或内容被吞。数学元组用 `$\langle s, a \rangle$`，占位符用反引号包裹
5. **正文避免裸 `{{`**：是 Vue 的插值语法。需要连续两个大括号时放进行内公式或代码 span 中
6. **`$$` 公式块与上下文之间留空行**：紧贴列表或表格可能导致解析纠缠

## 分类体系

| 编号 | 分类 | 覆盖方向 |
| --- | --- | --- |
| 01 | Foundation Models | 架构设计、预训练方法、Scaling Laws |
| 02 | Alignment & Safety | RLHF、DPO、Constitutional AI |
| 03 | Reasoning | Chain-of-Thought、数学推理、代码生成 |
| 04 | Multimodal | VLM、图像/视频理解、视频生成、语音 |
| 05 | Agents | Tool Use、Web Agent、多 Agent 协作 |
| 06 | Embodied AI | VLA、世界模型、机器人 RL、模仿学习 |
| 07 | Efficiency | 量化、蒸馏、剪枝、推测解码 |
| 08 | RAG & Knowledge | RAG 架构、向量检索、知识图谱 |
| 09 | Evaluation | Benchmark 设计、评估方法论 |
| 10 | Reinforcement Learning | RL 基础方法、连续控制、生成式策略、策略优化 |

一篇论文可以在多个分类的索引中出现，但 .md 文件只存一份，放在最核心的分类下。交叉出现时，在每个相关分类的索引（config.mts 侧边栏、papers/index.md、README.md、分类 README.md）中都插入条目，链接一律指向唯一的存放路径。

## 06 Embodied AI 的子分类结构

当前 Embodied AI 下设：

```
vla/
├── foundation/    # 真·基础模型 / 训练范式：π₀、π₀.₅、GR-3、SpatialVLA、ChatVLA、UniVLA、OTTER、Dexbotic、FAST、MMaDA-VLA
├── perception/    # 感知 / 空间 / 视觉表征增强：3D 编码、VGGT 对齐、视觉提示、视觉信号利用（含观测重注入等 training-free 视觉增强）
├── reasoning/     # 推理 / 规划 / 记忆 / 世界模型辅助决策：CoT、进度估计、子目标、未来帧预测、多 horizon 决策等（含 DreamVLA、FLARE、DUST 等"世界模型增强 VLA"）
├── efficient/     # 高效推理：Token 剪枝/缓存、量化、并行/推测解码、训练加速
└── rl/            # RL 后训练
world-models/       # 纯世界模型 / 模拟器：BridgeV2W、Kinema4D、MIND-V（预测未来、作数据生成/RL 训练场/评估用，不联合产出策略）
world-action-models/ # 世界动作模型 WAM：LDA-1B、FastWAM、WorldVLA、SpatialVAM、WAM 综述（联合建模状态与动作、直接产出策略；VAM 视为 WAM 子集）
imitation-learning/  # 模仿学习：EC-Flow 等
data-synthesis/      # 数据合成：MimicDreamer、Wh0（把人类演示或生成式世界模型转成机器人可训练监督数据，复用现成骨架后训练，核心贡献是"怎么造数据"）
```

**data-synthesis 落位判据**：论文核心贡献是**生成/对齐 VLA 训练数据**（human-to-robot 迁移、生成式世界模型造数据、数据增强 pipeline），策略骨架直接复用现成模型（π₀ / VITRA 等）后训练，而非提出新策略架构或新训练范式。注意与相邻类的区别：
- **用世界模型 ≠ 做世界模型**：MimicDreamer 的 H2R Aligner、Wh0 用的 Wan-I2V 都是视频扩散/生成模型，但只作数据生成的一个环节，不以"世界模型本身"为一级产出（那才归 `world-models/`）。
- 若论文是**完整基础模型**、数据合成只是其中一个组件（如 Qwen-RobotManip 的 H2R 管线），仍归 `vla/foundation/`；只有当数据 pipeline + 训练配方本身就是全部贡献时才落 `data-synthesis/`。
- 与 `imitation-learning/` 的区别：后者核心是"从演示学策略的算法"（如无动作光流 IL），前者核心是"造出这些演示数据"。

**三分类边界（世界建模相关论文如何落位，按"世界建模是不是一级输出"判定）**：

| 落位 | 判据 | 例子 |
| --- | --- | --- |
| `vla/reasoning/` | 核心产出是 **VLA 策略**，世界模型只是辅助模块（提供预测特征 / 规划信号 / 对齐目标） | DreamVLA、FLARE、DUST（自称"世界模型增强 VLA"） |
| `world-action-models/` | **联合建模未来状态与动作** $p(o',a\mid o,l)$，世界预测与动作生成 co-equal、直接产出策略；含 VAM（Video Action Model，WAM 子集） | LDA-1B、WorldVLA、FastWAM、SpatialVAM |
| `world-models/` | 核心产出是**世界模型/模拟器本身**（视频/4D/动作条件生成），策略只是下游应用之一或干脆不产出策略 | BridgeV2W、Kinema4D、MIND-V（当 RL 训练场用） |

要点：**WAM ≠ World Model**。World Model 只预测未来（$p(o'\mid o,a)$）；World Action Model 联合建模状态+动作、把世界与动作放平级。`vla/reasoning/` 与 `world-action-models/` 都会用到世界建模，区别在于世界建模是"辅助策略的手段"还是"与动作平级的一级输出"——判不准时看论文自我定位与评测形式（只报策略成功率、世界建模是内部正则 → reasoning；把联合状态-动作预测当核心贡献 → WAM）。术语参见 `world-action-models/WAM_Survey_2026.md`（WAM 综述，含 Cascaded/Joint 完整分类）。

### 子分类判别准则（避免 foundation/ 沦为兜底箱）

放 `foundation/` 的硬条件——满足任一即可：
1. **新的训练范式**（动作 tokenization、离散扩散、潜在动作预训练等改变 VLA 范式定义）；
2. **大规模预训练 / 开源 foundation 模型**（π₀、GR-3、SpatialVLA、UniVLA 等）；
3. **首次提出某条主线**（如 OTTER 冻结 CLIP + 文本感知特征提取的范式）。

仅"在已有 base 之上做架构改进"的论文 **不应** 进 `foundation/`，而应按主要贡献落到：
- 改视觉/3D/空间表征 → `perception/`
- 改时序记忆/规划/世界模型/多 horizon → `reasoning/`
- 主打 token 剪枝/缓存/量化/解码加速 → `efficient/`

**Training-free / 测试时增强不单独立类**：method property（是否需训练）不是分类轴，应按 *作用对象* 归类。例如 UAOR 是 training-free，但核心叙事是"观测信号在深层被遗忘 → 重注入感知"，归 `perception/` 而非单独的 inference/ 类（避免出现只有一篇论文的光棍分类）。当某类 training-free 方法累计到独立成族（如 3+ 篇）再考虑独立子目录。

判别歧义时按"论文核心叙事"优先：作者讲故事的主轴是什么，就归到对应分类。例如 FocusVLA 虽是策略架构改造，但核心是 token 剪枝 + 训练加速，归 `efficient/`；DAM-VLA 虽改动作头，但核心是 VLM 推理驱动的动作路由 + 双扩散协调，归 `reasoning/`。

如果新论文不属于已有子分类，可以新建目录并更新 config.mts 中的侧边栏结构。

## 写作风格约定

- 语言：中文为主，术语保留英文（如 flow matching、advantage conditioning）
- 公式：完整推导，附带直觉解释（"用大白话说"）
- 表格：使用 Markdown 表格呈现实验数据
- 对比：在"个人思考"中主动与项目中已有的其他论文建立联系
- 纯文字笔记：不插论文截图或图片，图示信息用文字、表格和公式复现（现有全部笔记均如此）
- 不使用 emoji（除非用户要求）；唯一既定例外：元信息中的链接行固定以 `> 🔗 ` 开头
- 中文粗体排版：`**` 闭合后若紧跟中文或英文字符（非标点），加一个空格；`**` 开头前不加空格

## 自我迭代规则

当用户指出笔记中的格式或内容问题并要求纠正时，修复完成后 **必须** 同步在本文件（CLAUDE.md）的相应章节中补充或更新对应的规范条目，确保同类问题不再重复出现。自己在构建或渲染中踩到新坑时同样回写（尤其是"渲染陷阱"一节）。
