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

<details>
<summary>🛡️ Alignment & Safety</summary>

> 暂无笔记

</details>

<details>
<summary>💡 Reasoning</summary>

> 暂无笔记

</details>

<details>
<summary>🖼️ Multimodal</summary>

> 暂无笔记

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

- [π₀ (2024)](papers/06-embodied-ai/vla/foundation/pi0_2024.md) — Flow Matching VLA 基础模型
- [π₀.₅ (2025)](papers/06-embodied-ai/vla/foundation/pi05_2025.md) — 异构协同训练 + 分层推理

</details>
</blockquote>

<blockquote>
<details open>
<summary>高效推理</summary>

- [LAC (2026)](papers/06-embodied-ai/vla/efficient/LAC_2026.md) — 可学习自适应 Token 缓存加速 VLA
- [VLA-Cache (2025)](papers/06-embodied-ai/vla/efficient/VLA_Cache_2025.md) — 训练无关跨帧 Token 缓存加速 VLA
- [SD-VLA (2026)](papers/06-embodied-ai/vla/efficient/SD_VLA_2026.md) — 静态-动态解耦实现长时程高效 VLA

</details>
</blockquote>

<blockquote>
<details open>
<summary>RL 后训练</summary>

- [RISE (2026)](papers/06-embodied-ai/vla/rl/RISE_2026.md) — 组合式世界模型 + 想象空间 RL
- [RLinf (2025)](papers/06-embodied-ai/vla/rl/RLinf_2025.md) — M2Flow 大规模 RL 训练系统
- [RLinf-USER (2026)](papers/06-embodied-ai/vla/rl/RLinf_USER_2026.md) — 真实世界在线策略学习统一系统
- [RLinf-VLA (2025)](papers/06-embodied-ai/vla/rl/RLinf_VLA_2025.md) — 统一高效的 VLA+RL 训练框架
- [SAC Flow (2026)](papers/06-embodied-ai/vla/rl/SAC_Flow_2026.md) — Flow Policy 序列建模 + off-policy RL
- [VLA-RL (2025)](papers/06-embodied-ai/vla/rl/VLA_RL_2025.md) — 在线 PPO 微调自回归 VLA
- [WoVR (2026)](papers/06-embodied-ai/vla/rl/WoVR_2026.md) — 幻觉感知世界模型 RL

</details>
</blockquote>

</details>
</blockquote>

<blockquote>
<details>
<summary>World Models</summary>

> 暂无笔记

</details>
</blockquote>

</details>

<details>
<summary>⚡ Efficiency</summary>

> 暂无笔记

</details>

<details>
<summary>🔍 RAG & Knowledge</summary>

> 暂无笔记

</details>

<details>
<summary>📊 Evaluation</summary>

> 暂无笔记

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
