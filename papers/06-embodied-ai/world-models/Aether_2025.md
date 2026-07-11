# Aether：几何感知的统一世界建模

> **论文**：*Aether: Geometric-Aware Unified World Modeling*
>
> **作者**：Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Tong He (通讯) et al.
>
> **机构**：USTC、Shanghai AI Lab、SII、SJTU、ZJU、FDU
>
> **发布时间**：2025 年 03 月（arXiv 2503.18945，v3 修订于 2025 年 07 月）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2503.18945) | [PDF](https://arxiv.org/pdf/2503.18945)
>
> **分类标签**：`世界模型` `4D 重建` `视频扩散` `相机位姿动作` `合成到真实`

---

## 一句话总结

Aether 把预训练视频扩散模型（CogVideoX-5b-I2V）后训练成一个几何感知的统一世界模型，用同一套扩散框架同时做 **4D 动态重建**、**动作条件视频预测** 和 **目标条件视觉规划** 三件事；关键设计是把「相机位姿轨迹」编码成 raymap 视频当作全局动作表征、把「深度」编码成 scale-invariant 视差视频与 RGB 一起联合去噪，且**全程只用合成数据训练**，却能零样本泛化到真实世界——在 KITTI 深度估计上刷到 Abs Rel 0.056 / δ<1.25 97.8，超过此前 SOTA 的前馈重建模型 CUT3R。

## 一、问题与动机

世界模型要为智能体提供三种能力：**感知**当前 4D 状态、在给定动作下**预测**未来、以及为达成目标而**规划**动作序列。以往工作把这三件事割裂：几何重建（DUSt3R/MonST3R/CUT3R 一系）追求把视频恢复成点云与相机轨迹但不会"想象"未来；视频生成模型（Sora/CogVideoX/Genie 2 一系）能预测未来帧却缺乏显式的 3D 几何、无法直接输出可用于导航的相机运动。

作者的核心主张是：**重建（几何）与生成（预测/规划）应当在一个模型里联合优化，二者能互相受益**。为此需要解决两个瓶颈：

1. **动作表征怎么统一**。真实动作模态五花八门（键盘、人体/机器人关节、光流……）。Aether 选定**相机位姿轨迹**作为全局动作表征，理由是它对 ego-centric 任务天然对齐：导航里相机轨迹就是行进路径，机械臂操作里手持/腕部相机的 6D 运动就是末端执行器的运动。

2. **4D 训练数据稀缺**。真实世界带精确深度+相机位姿的动态视频极难获取。Aether 干脆全部用**合成 RGB-D 视频**（取自 Cyberpunk2077、Horizon5 等 3A 游戏，沿用 DA-V 与 TheMatrix 的采集思路），再配一套**全自动相机位姿标注流水线**补齐 4D 标签，从约 12.5M 原始帧筛出约 8.9M 高质量标注帧用于训练。

## 二、核心方法

### 2.1 总体框架：一个扩散模型，条件组合切换任务

基座是 CogVideoX-5b-I2V（DiT 结构的视频扩散模型）。Aether 的目标 latent $z_0$ 由三种模态在通道维拼接而成：彩色视频 latent $z_{c0}$、深度视频 latent $z_{d0}$、动作(raymap) latent $z_{a0}$；条件输入包括彩色视频条件 $c_c$ 与动作条件 $c_a$。训练用标准扩散去噪目标：

$$
\mathcal{L}_\theta = \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I),\, t\sim\mathcal{U}(1,\mathcal{T})}\left[\,\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\,\right]
$$

其中 $z_0 = z_{c0}\otimes z_{d0}\otimes z_{a0}$，$c = c_c\otimes c_a$，$\otimes$ 表示通道维拼接。

**用大白话说**：模型就是学着把加了噪声的「彩色+深度+相机轨迹」三合一视频还原干净。训练时对不同任务用不同的「遮挡组合」——把某些通道置零当作未知、其余当作已知条件，于是同一个网络在推理时靠切换条件就能扮演不同角色：

| 任务 | 彩色条件 $c_c$ | 动作条件 $c_a$ |
| :-- | :-- | :-- |
| 4D 重建 | 输入整段视频 latent | 全部置零 |
| 视频预测（无动作） | 只给首帧、其余置零 | 全部置零（action-free） |
| 视频预测（动作条件） | 只给首帧 | 给完整目标相机轨迹 |
| 目标条件规划 | 首/尾帧给观测与目标图、中间置零 | 全部置零 |

训练时条件按概率随机遮挡（观测+目标 30%、仅观测 40%、全彩色 latent 即纯重建 28%、全部遮 2%），让一个模型泛化到所有任务。

### 2.2 深度视频编码：scale-invariant 视差

深度要塞进为 RGB 预训练的 3D VAE，必须先变成"看起来像图像"且尺度无关的表示。做法是先把深度 clip 到 $[d_{\min}, d_{\max}]$，取平方根后求倒数得到视差，再线性映射到 $[-1,1]$，最后单通道复制成三通道喂给 VAE：

$$
x_{\text{disp}} = \frac{1}{\sqrt{\text{clip}(x_d, d_{\min}, d_{\max})}},\qquad
\hat{x}_{\text{disp}} = \frac{x_{\text{disp}}}{\max(x_{\text{disp}})}\times 2 - 1,\qquad
z_d = \mathcal{E}(\hat{x}_{\text{disp}}\otimes \mathbf{1}_3)
$$

**用大白话说**：直接用绝对深度会因为场景尺度差异巨大而没法学；改成「视差＝1/深度」并按整段最大值归一化，就得到一个尺度无关、近处大远处小、动态范围友好的"伪灰度图"，这样冻结的视频 VAE 几乎不用改就能编码它。

### 2.3 相机轨迹编码：raymap 视频

相机参数（内参 $K$、外参 $E$）被转成 **raymap 视频**——每个像素对应一条射线，6 通道 = 3 通道射线方向 $r_d$ + 3 通道射线原点 $r_o$，从而与视频扩散的时空结构对齐。平移分量因数值范围大，先按最大视差归一化再过 signed-log 压缩：

$$
t' = \frac{t}{\max(x_{\text{disp}})}\cdot s_{\text{ray}},\qquad
t_{\log} = \text{sign}(t')\cdot\log(1+|t'|)
$$

射线方向 $r_d$ 在齐次坐标下按内参算出、**不做单位归一化**（保留 z 轴为单位长度以便反解）；raymap 空间上按 8 倍双线性下采样、时间上每连续 4 帧沿通道拼接，与 VAE latent 尺寸对齐。反解时先从 $r_o$ 恢复平移，再由射线方向反推内外参（附录给出闭式 Algorithm 1）。

**用大白话说**：相机运动本来是几个矩阵，DiT 不好直接吃；把它"画成"一张每像素一条光线的图片视频，相机位姿就变成和 RGB/深度同构的一段视频，网络就能像处理画面一样处理动作。signed-log 是为了让远近平移都落进网络舒服的数值区间。

### 2.4 两阶段训练

- **模型初始化**：加载 CogVideo-5b-I2V 权重，新增的深度/raymap 输入输出投影层通道**零初始化**（保证起步不破坏预训练先验）；不使用文本 prompt，喂空文本 embedding。可变帧数 $T\in\{17,25,33,41\}$、FPS 从 $\{8,10,12,15,24\}$ 采样，RoPE 系数随之插值。
- **阶段一**：latent 空间标准 MSE 扩散损失。
- **阶段二**（约为阶段一 1/4 步数）：把生成 latent 解码回像素空间加三项损失——彩色用 MS-SSIM、深度用 scale-shift-invariant (SSI) 损失、并由深度与 raymap 投影出的 pointmap 加 pointmap 损失：

$$
P = D\cdot R_d + R_o,\qquad
\mathcal{L}_{\text{pointmap}} = \frac{1}{N}\sum_{i=1}^{N} w_i\,\|\hat{P}_i - P_i\|_p
$$

其中权重 $w_i$ 与深度成反比，pointmap 损失只回传梯度到 raymap latent、并对视差侧停梯度。**用大白话说**：光在 latent 里对不齐还不够，第二阶段把结果解码成真图再从"感知质量、深度尺度不变、3D 点云一致"三个角度纠正，其中 pointmap 把深度和相机轨迹绑在同一个 3D 空间里互相校准。

- **工程规模**：FSDP(Zero-2)+DDP，单卡 batch 4、有效 batch 320，**80 张 A100-80GB 训练约两周**，AdamW + OneCycle。

### 2.5 自动 4D 数据标注流水线（4 阶段）

给定合成 RGB-D 视频，全自动标注相机内外参：(1) **动态掩膜**——Grounded SAM 2 按语义类别分割潜在动态物体（比光流分割更稳）；(2) **重建友好切片**——用 SIFT 关键点数、动态区域占比、RAFT 光流幅度与前后向一致性把长视频切成干净短片；(3) **粗相机估计**——DroidCalib 借静态区域深度给出初值；(4) **精化**——CoTracker3 长程跟踪 + SIFT/SuperPoint 特征点，Ceres Solver 做 bundle adjustment（Cauchy 损失），并用高质量稠密深度做前后向重投影在 3D 空间最小化误差。

## 三、实验结果

评测均为**零样本合成到真实**（重建任务仅去噪 4 步）。

**零样本视频深度估计（Table 1，Abs Rel↓ / δ<1.25↑）**：

| 方法 | Sintel Abs Rel | Sintel δ<1.25 | BONN Abs Rel | KITTI Abs Rel | KITTI δ<1.25 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| MonST3R-GA | 0.378 | 55.8 | **0.067** | 0.168 | 74.4 |
| CUT3R | 0.421 | 47.9 | 0.078 | 0.118 | 88.1 |
| **Aether**（重建类） | **0.324** | 50.2 | 0.273 | **0.056** | **97.8** |
| **Aether**（扩散类对齐 scale&shift） | **0.314** | **60.4** | 0.308 | **0.054** | **97.7** |

Sintel 上 Abs Rel 超过 MonST3R-GA；KITTI 上 Abs Rel 0.056 / δ<1.25 97.8 刷新前馈重建纪录、显著优于 CUT3R。BONN（室内、老数据集、含运动模糊）相对偏弱。

**零样本相机位姿估计（Table 2，前馈类，ATE/RPE trans/RPE rot ↓）**：

| 方法 | Sintel ATE | Sintel RPE trans | TUM-dyn RPE trans | ScanNet ATE |
| :-- | :-- | :-- | :-- | :-- |
| DUSt3R | 0.290 | 0.132 | 0.106 | 0.246 |
| CUT3R | 0.213 | 0.066 | 0.015 | **0.099** |
| **Aether** | **0.189** | **0.054** | **0.012** | 0.176 |

前馈方法中 Aether 在 Sintel ATE/RPE trans、TUM-dynamics RPE trans 上最优；ScanNet（室内）相对落后。

**视频预测（VBench 加权平均，in-domain / out-domain / overall）**：

| 设置 | 模型 | 加权平均 | 动态度 Dynamic Degree |
| :-- | :-- | :-- | :-- |
| 无动作（Tab 3） | CogVideoX | 79.01 / 77.52 / 78.51 | — |
| 无动作（Tab 3） | **Aether** | **80.34 / 79.42 / 80.04** | — |
| 动作条件（Tab 4） | CogVideoX | 79.56 / 80.70 / 79.92 | 83.87 / 93.02 / 86.76 |
| 动作条件（Tab 4） | **Aether** | **80.33 / 81.55 / 80.71** | **100.00 / 83.72 / 94.85** |

CogVideoX 倾向生成静止高清画面，Aether 能真正**跟随相机动作**产生高动态度场景，out-domain 提升更明显。

**消融与规划（去掉深度重建目标 vs 完整模型）**：

| 任务 | 指标 | Aether-no-depth | **Aether** |
| :-- | :-- | :-- | :-- |
| 动作条件导航（Tab 5，overall） | PSNR↑ / LPIPS↓ | 18.97 / 0.3074 | **19.70 / 0.2659** |
| 无动作视觉路径规划（Tab 6，overall VBench） | 加权平均 | 79.59 | **80.67** |

去掉 4D 重建目标后规划质量明显下降，直接支撑"重建目标能反哺生成/规划"的核心论点。**效率（Table 7，A100）**：Aether 在 480×640 下 6.14 FPS，而 DUSt3R-GA/MASt3R-GA/MonST3R-GA 在更小的 144×512 下仅 0.31–0.76 FPS。

## 四、局限性

1. **相机位姿精度仍是短板**——作者归因于 raymap 表征与视频扩散先验之间的不兼容，且去噪步数少导致轨迹带噪（推理时需再套 Kalman 滤波平滑），RPE rot 等指标不及优化式方法。
2. **室内明显弱于室外**——训练数据以室外 3A 游戏场景为主，BONN/ScanNet 等室内数据集上重建与位姿都掉点。
3. **高动态场景下无语言 prompt 易失败**——为统一多任务丢掉了文本条件，代价是复杂动态场景的预测稳定性。
4. **纯合成训练的固有天花板**——从未见过真实数据，域间隙、合成标注瑕疵与运动模糊仍限制上限。

## 五、评价与展望

**优点**。(1) 概念统一优雅：用"条件遮挡组合"把重建/预测/规划三类看似异质的任务收进一个扩散目标，是"世界模型三要素（感知-预测-规划）"少见的可运行统一实现。(2) 两个表征选择很关键——scale-invariant 视差让冻结的视频 VAE 免改造即可编码深度，raymap 让相机动作与画面同构，二者是把"几何"塞进"生成先验"的巧妙工程。(3) 用 pointmap 损失把深度与相机轨迹绑到同一 3D 空间做互校准，配合消融证据，较有说服力地证明了"重建反哺生成"。(4) 全合成训练 + 零样本迁移，且前馈推理比优化式重建快一个数量级，实用性突出。

**局限与开放问题**。(1) 动作表征只有相机位姿，对真正的**接触式操作/灵巧手**类动作（关节角、力/接触）无能为力——这决定了它更适合导航/ego-motion 而非精细操作，作者也把"探索新动作表征"列为首要 future work。(2) 位姿精度受限说明 raymap 这一"把外参画成图"的做法与扩散先验尚未完全调和，是否有比 raymap 更适配 DiT 的相机编码值得研究。(3) 纯合成数据的路线上限如何、真实数据 co-training 收益多大，仍待验证。

**与公开工作的关系**。重建侧对标 DUSt3R/MASt3R/MonST3R/CUT3R 这条"前馈点图"路线，Aether 的差异是把重建当作生成模型的一个子任务而非独立目标；生成侧站在 CogVideoX 之上，并与 Cat3D/Cat4D、Genie 2、Motion Prompting 等可控/4D 生成工作同处一个快速演进的空间。可能的改进方向包括：引入真实数据 co-training、保留语言 prompt 的可控性、以及设计更细粒度且与操作任务对齐的动作空间。

## 参考

1. Yang Zhuoyi et al. *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer.* arXiv 2408.06072, 2024.（基座视频扩散模型）
2. Wang Qianqian et al. *CUT3R: Continuous 3D Perception Model with Persistent State.* arXiv 2501.12387, 2025.（前馈重建主要对比基线）
3. Zhang Junyi et al. *MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion.* arXiv 2410.03825, 2024.（动态场景几何估计基线）
4. Chen Junyi et al. *Where Am I and What Will I See: An Auto-Regressive Model for Spatial Localization and View Prediction.* 2024.（raymap 空间表征来源）
5. Yang Honghui et al. *Depth Any Video with Scalable Synthetic Data (DA-V).* arXiv 2410.10815, 2024.（合成 RGB-D 数据采集思路来源）
