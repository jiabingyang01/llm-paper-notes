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

- [DLC (2025)](papers/04-multimodal/vlm/hallucination/DLC_2025.md) — CLIP 动态探针 + 相对视觉优势 + 自适应 Logits 调制，Training-Free 高效缓解 LVLM 幻觉
- [HALC (2024)](papers/04-multimodal/vlm/hallucination/HALC_2024.md) — 自适应 FOV 对比解码 + 视觉匹配 beam search，无训练即插即用缓解三种对象幻觉
- [HIME (2026)](papers/04-multimodal/vlm/hallucination/HIME_2026.md) — HIS 层自适应加权投影编辑，无训练/无开销降低 61.8% 对象幻觉
- [MemVR (2025)](papers/04-multimodal/vlm/hallucination/MemVR_2025.md) — FFN key-value memory 视觉回溯 + 不确定性动态触发，1.04× 延迟缓解幻觉且提升通用能力
- [SENTINEL (2025)](papers/04-multimodal/vlm/hallucination/SENTINEL_2025.md) — 域内自举 + 句子级 C-DPO 早期干预，幻觉率降低 92% 且通用能力不降反升
- [VisFlow (2025)](papers/04-multimodal/vlm/hallucination/VisFlow_2025.md) — 双层注意力干预（Token 级增强显著视觉 token + Head 级抑制系统/文本头），无训练缓解幻觉

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

- [DAM-VLA (2026)](papers/06-embodied-ai/vla/foundation/DAM_VLA_2026.md) — 动作路由 + 手臂/夹爪双扩散模型 + 双尺度加权协调
- [Dexbotic (2025)](papers/06-embodied-ai/vla/foundation/Dexbotic_2025.md) — 开源 VLA 工具箱：统一框架 + Qwen2.5 预训练模型 + 实验驱动开发，最高 +46.2%
- [MoH (2025)](papers/06-embodied-ai/vla/foundation/MoH_2025.md) — 多 Horizon 动作块并行融合 + 轻量门控 + 动态推理，π₀.₅+MoH LIBERO 99%
- [OptimusVLA (2026)](papers/06-embodied-ai/vla/foundation/OptimusVLA_2026.md) — 双记忆增强 VLA：GPM 任务级先验检索 + LCM 时序一致性，LIBERO 98.6%、2.9× 加速
- [π₀ (2024)](papers/06-embodied-ai/vla/foundation/pi0_2024.md) — Flow Matching VLA 基础模型
- [π₀.₅ (2025)](papers/06-embodied-ai/vla/foundation/pi05_2025.md) — 异构协同训练 + 分层推理
- [SF (2025)](papers/06-embodied-ai/vla/foundation/SF_2025.md) — 隐式空间表征对齐（VGGT），推理零开销，LIBERO 98.5%，训练 3.8× 加速
- [TGM-VLA (2026)](papers/06-embodied-ai/vla/foundation/TGM_VLA_2026.md) — 关键帧采样优化 + 颜色反转 + 任务引导 Mixup，RLBench 90.5%

</details>
</blockquote>

<blockquote>
<details open>
<summary>高效推理</summary>

- [LAC (2026)](papers/06-embodied-ai/vla/efficient/LAC_2026.md) — 可学习自适应 Token 缓存加速 VLA
- [SD-VLA (2026)](papers/06-embodied-ai/vla/efficient/SD_VLA_2026.md) — 静态-动态解耦实现长时程高效 VLA
- [RLRC (2025)](papers/06-embodied-ai/vla/efficient/RLRC_2025.md) — 结构化剪枝 + SFT/RL 恢复 + 量化，8× 显存压缩
- [VLA-Cache (2025)](papers/06-embodied-ai/vla/efficient/VLA_Cache_2025.md) — 训练无关跨帧 Token 缓存加速 VLA

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
- [PLD (2026)](papers/06-embodied-ai/vla/rl/PLD_2026.md) — 残差 RL 专家探索 + 基础策略探针混合蒸馏实现 VLA 自改进
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
