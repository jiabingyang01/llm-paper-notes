# Being-H0：从大规模人类视频中进行视觉-语言-动作预训练

> **论文**：*Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos*
>
> **作者**：Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, Zongqing Lu
>
> **机构**：Peking University、Renmin University of China、BeingBeyond
>
> **发布时间**：2025 年 07 月（arXiv 2507.15597）
>
> **发表状态**：未录用（预印本，v1）
>
> 🔗 [arXiv](https://arxiv.org/abs/2507.15597) | [PDF](https://arxiv.org/pdf/2507.15597)
>
> **分类标签**：`VLA 预训练` `人类视频` `灵巧手` `MANO 动作 tokenization`

---

## 一句话总结

把"人手"当成理想的通用操作器（foundation manipulator），先在 15 万条 / 265 万帧规模的人类视频（UniHand）上用离散 MANO 动作 token 做自回归 VLA 预训练（physical instruction tuning），再用轻量 MLP 动作头把预训练模型迁移到真实灵巧手；关键机制是把连续手部运动用 Grouped Residual Quantization 做"分部件（腕/指分离）"的毫米级离散化（重建误差可低至 0.129–0.523 cm），最终在 7 个真机灵巧操作任务上超过 GR00T N1.5 与无预训练的 InternVL3（如 Pour-Cup 成功率 1.00、Close-Lid 0.60、Unfold-Clothes 0.75）。

## 一、问题与动机

现有 VLA 面临的核心瓶颈是**数据结构性错配**：

- **数据来源受限**。主流 VLA 依赖遥操作演示（Open X-Embodiment、AgiBot 等），规模比互联网级 LMM 训练数据小几个数量级，且多聚焦简单夹爪、缺乏细粒度手指协调；合成数据则受 sim-to-real gap 拖累。作者称这是 embodied intelligence 长期陷在的 "data swamp"。
- **预训练-下游同构性缺失**。在 LLM/LMM 中，预训练（文本推理、视觉-文本理解）与下游任务天然同构，因此 visual instruction tuning 能带来巨大增益；但在 VLA 中，2D 视觉/文本输入与需要本体感受（proprioception）的 3D 动作空间之间存在异构鸿沟，导致以往的隐式对齐方法（对比学习、masked autoencoder、latent action，如 GR00T 的隐式潜动作）迁移收益不明确。

作者提出的核心问题（原文加框）：

> Can we pretrain a dexterous VLA from large-scale human videos, analogous to GPT-3, to explicitly imitate human actions and adapt to robot hands via post-training?

动机是：人手是灵巧操作的"黄金标准"，覆盖自然场景中海量任务；学习人手运动能弥合预训练-下游异构性。但要把它扩展到大规模视频，需克服四大挑战：(1) 数据异构（相机系统/坐标系/录制条件各异）；(2) 精确量化（手部精细运动需毫米级离散化）；(3) 跨模态推理（视觉-语言-精细手指运动的联合建模）；(4) 机器人控制迁移（人手与机器人手形态差异）。

## 二、核心方法

整体范式称为 **Physical Instruction Tuning**，把标准的 visual instruction tuning 扩展到物理域，包含三段：**预训练**（在人类视频上学手部运动生成）+ **物理空间对齐**（统一异构相机到一致坐标、注入 3D 先验）+ **后训练**（迁移到真实机器人）。骨干为 InternVL3（InternViT-300M 视觉编码器 + 2 层 MLP 投影 + InternLM3/Qwen2.5 语言骨干），提供 1B/8B/14B 三种规模。

### 1. 预训练目标：把手部运动当"外语"做下一 token 预测

数据 $\mathcal{D}=\{(\mathbf{v}_i,\mathbf{t}_i,\mathbf{m}_i)\}$，其中 $\mathbf{v}$ 是视觉输入，$\mathbf{t}$ 是语言指令，$\mathbf{m}=\{\theta,\mathbf{r}_{rot},\tau,\beta\}$ 是 MANO 参数化的手部运动（关节角 $\theta$、腕部旋转 $\mathbf{r}_{rot}$、平移 $\tau$、形状 $\beta$）。每个样本视作一对 instruction-following $\{\mathcal{X}_Q,\mathcal{X}_A\}$：

$$
\theta^{*}=\arg\min_{\theta}\sum_{i=1}^{N}\mathcal{L}(\Theta)=-\sum_{j=1}^{L}\log P_{\Theta}(y_j\mid \mathcal{X}_Q,\hat{y}_{1:j-1})
$$

**用大白话说**：把连续的手部运动切成一串离散 token（`<MOT>...</MOT>` 包裹），和文本、视觉 token 混进同一个序列里，模型就像预测下一个词一样预测下一个"动作 token"。这样运动生成、运动描述（motion captioning）、视觉到运动生成都统一成同一种自回归任务。

### 2. Part-Level 动作 tokenization：分部件的 Grouped Residual Quantization

连续 MANO 序列 $\mathcal{M}\in\mathbb{R}^{T\times D}$ 经 1D-Conv 编码器降到 $z\in\mathbb{R}^{\lceil T/\alpha\rceil\times d}$（$\alpha$ 为时间下采样率，取 4），再离散化后解码：

$$
\mathcal{M}\xrightarrow{\text{Encoder}}z\xrightarrow{\text{VQ}}\{m_1,\dots,m_n\}\xrightarrow{\text{Decoder}}\hat{\mathcal{M}}
$$

量化用 **GRQ-VAE**：先把通道维 $d$ 沿轴切成 $n$ 组，每组 $z^{(g)}$ 用 $L$ 层残差量化器（RQ）逐层逼近：

$$
r_0=z_i^{(g)},\quad q_l=\arg\min_{c\in\mathcal{C}^{(g)}}\lVert r_{l-1}-c\rVert_2,\quad r_l=r_{l-1}-q_l,\qquad \hat{z}_i^{(g)}=\sum_{l=1}^{L}q_l
$$

由于腕部参数 $\mathbf{r}_{rot},\tau$ 分布跨越更大 3D 空间、重建误差更大，额外加腕部专项损失，总损失为：

$$
\mathcal{L}_{\text{wrist}}=\lVert\mathbf{w}-\hat{\mathbf{w}}\rVert_2^2,\ \mathbf{w}=[\mathbf{r}_{rot},\tau];\qquad
\mathcal{L}=\mathcal{L}_{\text{recon}}+\lambda_1\mathcal{L}_{\text{commit}}+\lambda_2\mathcal{L}_{\text{wrist}}
$$

**Part-Level 关键设计**：把 $m=\{\theta,\mathbf{r}_{rot},\tau,\beta\}$ 拆成**腕部运动** $\{\mathbf{r}_{rot},\tau\}$（负责全局定位）与**手指运动** $\{\theta,\beta\}$（负责精细操作）两套独立 tokenizer。这样既提升重建质量，又给 LMM 骨干提供更清晰的 token 语义（用 part-level 时 $\mathcal{L}_{\text{wrist}}$ 省略）。实现上：8 层 RQ、组数 $n=2$、每部件码本 $K_w=K_f=4096$、码维 $d=512$，每 1 秒运动被离散成 $2\times n\times L\times\lceil T/\alpha\rceil=128$ 个 token。

**用大白话说**：手腕像"搬运工"负责把手送到大致位置，手指像"绣花匠"负责微操作，两者动态范围和精度需求完全不同，硬塞进一个码本会互相拖累。残差量化则像"逐层修正的画笔"，先画大轮廓再一层层补细节，从而在离散化的同时保住毫米级精度。

**动作特征选型**：对比了 MANO-D51（axis-angle）/D99（6D 旋转）/D109（+形状 $\beta$）/D114（D51+关节位置）/D162（D99+关节位置）五种。结论是 6D 旋转利于手指、axis-angle 利于手腕；加入辅助关节位置 $j$ 有益、加入形状 $\beta$ 反而有害。最终选 **MANO-D162**（6D 旋转 + 关节位置，形状固定用每序列首帧）。

### 3. 统一跨模态注意力

视觉/文本/运动三模态拼成统一 token 序列 $\mathbf{S}=\{s_i\}$，视觉 token 替换 `<IMG_CONTEXT>` 占位符，运动 token 组成 `<MOT>...</MOT>` 块（每块 128 token）。拼接后的隐状态 $\mathbf{H}_{v,t,m}=[\mathbf{H}_v;\mathbf{H}_t;\mathbf{H}_m]$ 共享投影做注意力：

$$
\mathbf{Q}_{v,t,m}=\mathbf{W}_Q\mathbf{H}_{v,t,m},\quad \mathbf{K}_{v,t,m}=\mathbf{W}_K\mathbf{H}_{v,t,m},\quad \mathbf{V}_{v,t,m}=\mathbf{W}_V\mathbf{H}_{v,t,m}
$$

**用大白话说**：不给运动单独开一条支路，而是让它和文字、图像在同一注意力里"平起平坐"互相看，这样才能学到"看到什么场景→用什么策略→落到哪根手指怎么动"的跨模态依赖。

### 4. 训练细节：双层 mask + 三种解码模式

运动码 $\mathcal{V}_{\text{motion}}$ 只占整个词表 $\mathcal{V}$ 一小块，因此：

- **词表级 logit mask**（概率 $\mathcal{P}=50\%$）：在运动标签上把非运动 logit 置 $-\infty$，让梯度聚焦运动嵌入空间。

$$
\bar{z}_i=\begin{cases}z_i & i\in\mathcal{V}_{\text{motion}}\\ -\infty & \text{otherwise}\end{cases}\quad(\text{以概率 }\mathcal{P})
$$

- **token 级 loss mask**：过滤 per-token 交叉熵损失，只保留落在 $[Q_{\text{low}},Q_{\text{high}}]$（取 $[15\%,95\%]$ 分位）内的中等难度 token，避免静止姿态（太易）与突发抖动（太难）主导训练。

$$
\bar{L}=\{\ell_i\in L\mid Q_{\text{low}}\le\ell_i\le Q_{\text{high}}\},\qquad \mathcal{L}_{\text{motion}}=\frac{1}{|\bar{L}|}\sum_{\ell_i\in L}\ell_i
$$

**三种解码模式**：Free-format（无约束采样，可能生成无法解码的运动块）、Block-formatted（强制只在运动词表内采样，用于定量评测）、Soft-formatted（把预测与 GT 的 MANO 参数取均值后再量化，评估局部生成质量、缓解一对多歧义）。

### 5. 物理空间对齐（弥合 3D 缺失）

骨干由 2D 视觉-文本预训练初始化、缺 3D 先验，且多源相机内参不一致。作者引入统一**弱透视相机空间**：给定源/目标内参 $K,K'$，算缩放与平移把像素 $(u,v)$ 映到 $(u',v')$：

$$
s_x=\tfrac{f'_x}{f_x},\ s_y=\tfrac{f'_y}{f_y},\ \Delta x=c'_x-s_x c_x,\ \Delta y=c'_y-s_y c_y;\qquad u'=s_x u+\Delta x,\ v'=s_y v+\Delta y
$$

并提出 **view-invariant 运动分布均衡**，在不改视角/位置、保持弱透视一致的前提下扩增小样本源：深度缩放 $\tau_c^{z'}=\lambda_s\cdot\tau_c^z$（配套图像按 $1/\lambda_s$ 重缩放）、绕光轴的面内旋转 $\tau'_c=R_z(\varphi)\tau_c,\ R'_c=R_z(\varphi)R_c$（图像同步旋转）。

**用大白话说**：不同数据集拍摄参数五花八门，直接混训模型学不到一致的"2D 看起来多大 = 3D 有多远"。先把所有画面归一到同一套虚拟相机，再用"拉近拉远手 + 转一转画面"造增强，既扩大视角覆盖又不破坏几何一致性——注意作者特意指出**普通的随机裁剪/翻转会破坏弱透视一致性，因此禁用**。

### 6. 后训练：非自回归 MLP 动作头

把预训练 VLA 当编码器：本体感受 $\mathbf{p}_t$ 经投影 $f_p$ 与视觉-文本上下文拼成 `ctx`，一组可学习 action query $\{\mathbf{q}_1,\dots,\mathbf{q}_{N_a}\}$ 在预训练编码器中 attend，再经回归头 $f_r$ 输出可执行灵巧位姿（action chunk），用 L1 模仿学习：

$$
\mathbf{a}_i=f_r\big(\Theta(\mathbf{q}_i,\texttt{ctx}\oplus f_p(\mathbf{p}_t))\big),\qquad \mathcal{L}(\Theta_a)=\frac{1}{N_a}\sum_{i=1}^{N_a}\lVert\mathbf{a}_i-\mathbf{a}_i^{*}\rVert_1
$$

**用大白话说**：人手 MANO 和真机灵巧手运动学不一样，不能直接搬运动 token；作者选了最朴素的做法——冻用预训练大脑做特征，再挂一个小 MLP 把"人手先验"翻译成机器人可执行动作，并明说更复杂的迁移（如离散动作 token / diffusion policy）留待未来。

### 7. UniHand 数据集与数据构造流水线

UniHand 聚合 11 个来源、三类数据：**动作捕捉**（如 OAKINK2，多视图高精度）、**VR 录制**（如 EgoDex，含多达 194 类家务任务，Apple Vision Pro + SLAM 追踪）、**伪标注**（如 Taste-Rob，约 10 万条 egocentric 视频、HaMeR 逐帧估计位姿）。统计（Table 1）：166.5M 指令样本、444.1K 序列、130M 帧、1155 小时。生成 1.65 亿条 motion-instruction 对，实际采样 **UniHand-2.5M** 做预训练。

数据构造：(a) **手位姿标准化**——统一到 MANO；有 mocap/SLAM 标签直接提取，仅有关节位置则梯度拟合，完全无标签用 HaMeR 估计并做左右手一致性纠错+时序插值+关节角约束平滑；(b) **任务描述标注**——按 ≤10 秒切 chunk，2 FPS 采帧，用 `Gemini-2.5-Flash-Lite` 生成 chunk 级（整体活动）+ per-second 级（接触状态/物体属性/手部/轨迹）双层标注，用 `Gemini-2.5-Pro`（约 20 个/类模板）扩指令变体；(c) 三类指令任务——Instructional motion generation（1.6M）、Motion translation（0.4M）、Contextual motion prediction（0.5M）。

## 三、实验结果

**实现**：448×448 图像，MANO-D162@15 FPS，128 运动 token/秒；AdamW，lr $1\times10^{-5}$，batch 128，32× A800-80G，视觉 adapter 与 LLM 骨干联合微调。真机：7-DoF Franka Research 3 臂 + 6-DoF Inspire 手 + RealSense L515；遥操作用 Gello 外骨骼 + D435i，每任务采 50–100 条轨迹，每任务 20 次随机试验。

### 手部运动生成 / 翻译（Table 2）

| 模型 | Valid Rate ↑ | T2M R@3 (head) ↑ | T2M R@3 (tail) ↑ |
| --- | --- | --- | --- |
| ground truth | - | 33.5 | 42.7 |
| Being-H0-1B | 64.8 | 12.5 | 14.3 |
| Being-H0-8B | 99.8 | 18.4 | 19.7 |
| Being-H0-14B | 100.0 | 19.0 | 22.1 |

模型越大，生成合法运动块的比例（1B 仅 64.8% → 8B 99.8% → 14B 100%）与运动-语言双向对齐（T2M R@3）都显著提升。

### 视觉-grounded 手部运动生成（Table 3，单位 cm / %；head=EgoDex 主分布，tail=TACO+HOI4D+H2O+OakInk2 长尾）

| 模型 | MPJPE↓ (h/t) | MWTE↓ (h/t) | PA-MPJPE↓ (h/t) | M2T R@3↑ (h/t) | FID↓ (h/t) |
| --- | --- | --- | --- | --- | --- |
| GR00T N1.5 | 9.82 / 15.35 | 8.51 / 11.20 | 1.33 / 1.41 | 13.1 / 14.8 | 11.7 / 14.4 |
| Being-H0-1B | 9.71 / 17.21 | 8.25 / 12.04 | 1.50 / 1.55 | 12.1 / 15.3 | 12.2 / 13.1 |
| Being-H0-8B | 7.20 / 9.02 | 5.69 / 8.11 | 1.09 / 1.32 | 15.9 / 18.7 | 11.5 / 13.4 |
| Being-H0-14B | **6.87 / 8.11** | **5.19 / 7.41** | **1.03 / 1.20** | **17.2 / 20.5** | **10.3 / 11.8** |

Being-H0-14B 在所有指标上均优于 GR00T N1.5 基线，且**在 tail（长尾）分裂上优势更明显**，说明扩规模主要改善对多样运动分布的泛化。

### tokenizer 重建消融（Table 5，单位 cm）

| 特征 | Part-Level MPJPE↓ | Part-Level PA-MPJPE↓ | 4-Groups MPJPE↓ | 16-Layers MPJPE↓ |
| --- | --- | --- | --- | --- |
| MANO-D51 | 0.556 | 0.209 | 1.165 | 1.466 |
| MANO-D114 (+关节位置) | **0.523** | 0.167 | 0.810 | 0.996 |
| MANO-D162 (+关节位置) | 0.573 | **0.129** | 0.704 | 1.054 |

Part-Level 分部件量化显著优于整手统一量化（4-Groups / 16-Layers），达到毫米级重建。虽然 MANO-D162 的 MPJPE 略高于 D114，但其在下游运动生成任务上一致更好（Table 6），故选为默认。

### 真机灵巧操作成功率（Table 7，%）

| 任务 | GR00T N1.5 | InternVL3（无预训练） | Being-H0 |
| --- | --- | --- | --- |
| Pick-Place-Toy (Seen) | 0.75 | 0.55 | 0.75 |
| Pick-Place-Toy (Unseen) | 0.40 | 0.55 | **0.65** |
| Pick-Place-Toy (Clutter) | 0.50 | 0.50 | **0.60** |
| Close-Toolbox | 0.80 | 0.50 | **0.85** |
| Close-Lid | 0.50 | 0.25 | **0.60** |
| Pour-Cup | 0.90 | 0.55 | **1.00** |
| Unfold-Clothes | 0.60 | 0.45 | **0.75** |

Being-H0 在全部任务上取得最高成功率。与它同架构同规模但没做 physical instruction tuning 的 InternVL3 明显更弱，直接验证"人类视频预训练"的增益。GR00T N1.5 在 seen 场景相当，但面对 unseen 物体与 clutter 泛化明显下滑。

### 其它关键结论

- **数据缩放**（Figure 9）：0.5M→2.5M，MPJPE/MWTE/M2T R@3/FID 持续改善；最大规模下 PA-MPJPE 略降但语义对齐（M2T R@3）继续升，反映模型随数据增多更重"功能/语义合理性"而非逐指精确复刻。
- **数据配方消融**（Table 6）：去掉 view-invariant 均衡在 tail 上大幅退化（易过拟合主导相机配置）；去掉 translation 监督主要伤 PA-MPJPE/M2T/FID；去掉 contextual 预测则全指标下滑。
- **数据效率**（Figure 13）：Being-H0 在 25%/50%/100% 演示数据下均稳定超基线；Close-Lid 任务基线在 25% 数据时完全失败（0%），而 Being-H0 已有 15% 成功率。

## 四、局限性

- **控制迁移过于朴素**。作者明确承认后训练仅用固定 action query + MLP 回归头，未真正把人手运动 token 迁移给机器人，也未用离散动作 token / diffusion policy 等更强策略；人-机形态差异靠 L1 模仿"硬拟合"。
- **不建模交互物体**。本版本不涉及物体 6D 位姿 / affordance / 接触点，纯建模手部运动，复杂工具使用与多物体交互留待未来。
- **缺物理属性**。力、摩擦等物理量在纯 RGB 视频中天然缺失，弱透视投影也带来深度歧义；作者在"进一步讨论"里把 RGB-D 深度、触觉、音频等多传感器融合列为未来方向。
- **伪标注噪声**。大量长尾数据靠 HaMeR 伪标注，且显式排除了遮挡严重 / 出画 / 动态视角样本（如 Ego4D、EPIC-Kitchen），说明当前流水线对"最野"的野外视频还吃不动。
- **真机评测规模有限**。仅 7 类任务、每任务 20 次试验、单一 Franka+Inspire 平台，统计置信区间偏窄；未与 π0、OpenVLA 等更多 VLA 直接同台。

## 五、评价与展望

**优点**：(1) 把"人手=通用操作器"的直觉落成一套可扩展、可复现的显式动作建模流水线，与 GR00T N1.5 的**隐式 latent action** 形成清晰对照，并在多项指标上证明显式 tokenization 泛化更好；(2) Part-Level GRQ 是本文最扎实的技术贡献，把"运动当外语"从口号做到毫米级重建，腕/指分离的设计有明确物理直觉且被消融支持；(3) UniHand 的数据工程（MANO 标准化 + 弱透视统一 + view-invariant 均衡 + LMM 分层指令标注）本身就是可复用的资产，1155 小时 / 165M 指令对是当前 egocentric 手部运动数据的较大规模整合；(4) 完整给出了规模、数据、配方三条 scaling 曲线，实证性强。

**与其它公开工作的关系**：动机与 GR00T N1.5、π0、OpenVLA 一脉相承（弥合数据瓶颈），但路线相反——不做隐式表征而做显式运动 token；动作离散化借鉴 human-body motion generation 里的 RQ/H2VQ 系列，把身体运动 tokenizer 迁移到手部并做 part-level 化；数据侧站在 EgoDex、Taste-Rob、OAKINK2、HaMeR、Dyn-HaMR 等 3D 手建模工作的肩膀上；与 MEgoHand 等同样把视觉输入引入手部运动生成的工作互为呼应。相对 FAST（离散余弦变换动作 token）而言，本文的量化是学习式码本 + 分部件，语义更结构化。

**开放问题与可能改进方向**：(1) 后训练是明显短板——把预训练学到的运动 token 真正作为机器人动作先验（而非只当特征），或引入 flow/diffusion 动作头，很可能进一步释放预训练价值；(2) 引入物体/接触/affordance 的联合建模，解决"手到了但抓不稳"的失败模式；(3) 弱透视投影导致的深度歧义可用 RGB-D 或单目深度先验缓解，物理合理性可借 RL / 物理仿真做轨迹精修（作者已在 related work 中点到 foot-sliding 类问题）；(4) 打通被排除的强遮挡 / 动态视角野外视频（Ego4D 级），才能真正逼近"GPT-3 式"的 web-scale 预训练愿景；(5) 跨本体（不同灵巧手 / 夹爪）迁移与真机评测规模仍需扩大以支撑更强结论。

## 参考

1. Bjorck et al., *GR00T N1.5* / NVIDIA GR00T（隐式 latent action 的大规模 egocentric VLA，本文主要对照基线）。
2. Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246（夹爪 VLA 代表，本文论证其缺灵巧性）。
3. Chen et al., *InternVL3*（骨干架构：InternViT-300M + 动态高分辨率 patch + InternLM3/Qwen2.5）。
4. Romero et al., *Embodied Hands (MANO)*（手部参数化模型，本文运动表征基础）。
5. Pavlakos et al., *HaMeR: Reconstructing Hands in 3D with Transformers*（伪标注位姿估计器，UniHand 无标签数据来源的关键工具）。
