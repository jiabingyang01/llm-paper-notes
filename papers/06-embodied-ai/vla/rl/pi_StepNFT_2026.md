# π-StepNFT：更宽的探索空间需要更细粒度的监督——Flow-based VLA 的在线 RL 框架

> **论文**：*π-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs*
>
> **作者**：Siting Wang, Xiaofeng Wang, Zheng Zhu, Minnan Pei, Xinyu Cui, Cheng Deng, Jian Zhao, Guan Huang, Haifeng Zhang, Jun Wang
>
> **机构**：GigaAI、中国科学院自动化研究所、中国科学院大学、清华大学、中关村学院、爱丁堡大学、伦敦大学学院
>
> **发布时间**：2026年3月
>
> **链接**：[arXiv](https://arxiv.org/abs/2603.02083) | [项目主页](https://wangst0181.github.io/pi-StepNFT/)

---

## 一句话总结

提出 π-StepNFT，一个**无 Critic、无似然**的在线 RL 框架：通过 SDE 采样拓宽探索空间，将监督目标从终端 $x_0$ 下移到逐步转移 $x_t \to x_{t^-}$，并用 logistic 对比排序损失替代 weighted-MSE 消除隐式惩罚，在 LIBERO few-shot 上比 SFT 提升 32.9%，ManiSkill OOD 场景比 PPO 高 11.1%。

---

## 一、问题与动机

### 1.1 Flow-based VLA 的 RL 瓶颈

当前最先进的 VLA 模型（如 π₀、π₀.₅）普遍采用 **flow matching** 作为动作生成范式。Flow matching 通过学习一个时间依赖的向量场 $v_\theta(x, t, c)$，将高斯噪声 $x_1 \sim \mathcal{N}(0, I)$ 通过 ODE 积分映射为动作 $x_0$。

然而，用 RL 微调 flow-based VLA 面临一个**根本性瓶颈**：

- **似然不可计算**：标准策略梯度需要 $\log \pi_\theta(a|s)$，但 flow policy 的对数似然需要沿整个生成轨迹积分 Jacobian 迹，计算代价极高且数值不稳定
- **ODE 确定性导致探索不足**：确定性 ODE rollout 的探索空间完全受限于初始噪声分布，策略很快坍缩到一条窄线（narrow manifold），缺乏自我改进的能力

### 1.2 现有方案的不足

面对上述瓶颈，现有方法各有局限：

| 方法 | 策略 | 局限 |
| --- | --- | --- |
| GR-RL | 隐空间价值蒸馏绕过似然 | 需要额外的 value network，容易过拟合多模态特征 |
| π₀.₆* | 偏好反馈 + 离线 RL | 不做在线探索，受限于离线数据分布 |
| πRL | ODE → SDE 变换近似似然 | 仍需 PPO + Critic，Critic 在视觉多样场景易过拟合 |
| Diffusion-NFT | 前向过程上的无似然优化 | 专为图像生成设计，直接迁移到具身控制效果不佳 |

### 1.3 核心洞察：Wider Space Needs Finer Steps

论文的核心观察可以用 Figure 1 中的三栏对比来理解：

**左栏（ODE）**：确定性 ODE 采样下，中间状态 $x_t$ 沿一条窄轨迹行进。在终端 $x_0$ 上做"点对点"监督是合理的，但探索范围太窄。

**中栏（Naive SDE）**：引入 SDE 注入噪声，探索空间变宽了，但仍然用终端 $x_0$ 做监督。问题是：噪声沿 denoising 路径累积放大，最终的 $x_0$ 方差极大，终端监督信号变得粗糙且不稳定。

**右栏（π-StepNFT）**：保留 SDE 的宽探索空间，但将监督目标**下移到每一步转移** $x_t \to x_{t^-}$，提供精确的局部梯度。同时用方差归一化消除不同时间步的尺度差异。

**用大白话说**：如果你在一个大迷宫里探索（SDE 扩大了探索范围），光告诉你终点在哪里（终端监督）是不够的——你需要在每个岔路口都有路标（逐步监督）才不会迷路。迷宫越大，路标就需要越密集。

---

## 二、预备知识

### 2.1 Flow Matching 基础

VLA 策略生成连续动作 $x_0 \in \mathbb{R}^d$，条件为上下文 $c$（包含视觉观测和语言指令）。Flow matching 学习向量场 $v_\theta(x, t, c)$，训练目标是：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t, c) - u_t\|^2\right]$$

其中 $x_t = tx_1 + (1-t)x_0$，$u_t = x_1 - x_0$。

**ODE 采样**（确定性）：从 $t=1$ 到 $t=0$ 积分 $dx = v_\theta(x,t,c)dt$，Euler 离散化为：

$$x_{t^-} = x_t - v_\theta(x_t, t, c)\delta_t \tag{1}$$

**SDE 采样**（随机性）：注入噪声保持边际分布，Euler-Maruyama 离散化为：

$$x_{t^-} = x_t + \left[v_\theta(x_t,t) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta(x_t,t))\right](-\delta_t) + \sigma_t\sqrt{\delta_t}\epsilon \tag{2}$$

其中 $\epsilon \sim \mathcal{N}(0, I)$。

### 2.2 SDE 采样的仿射结构

SDE 的每一步转移诱导出一个**高斯转移密度**：

$$q_{\theta,t}(x_{t^-} | x_t, c) = \mathcal{N}\left(\mu_{\theta,t}(x_t, c),\ \Sigma_t\right)$$

关键性质：转移均值是向量场输出的**仿射函数**：

$$\mu_{\theta,t}(x_t, c) = U_t(x_t, t) + B_t(t) \cdot v_\theta(x_t, t, c) \tag{3}$$

其中 $U_t = 1 - \frac{\sigma_t^2 \delta_t}{2t}$，$B_t = -\delta_t - (1-t)\frac{\sigma_t^2 \delta_t}{2t}$ 是由噪声调度预确定的系数。

这个仿射关系至关重要——它意味着我们可以**直接从转移目标高效地将梯度传回策略参数**，无需通过 ODE solver 做反向传播。

### 2.3 RL 微调与似然鸿沟

标准策略梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_i \nabla_\theta \log \pi_\theta(a_i|s_i) \cdot \Psi_i\right]$$

但 flow policy 的 $\log \pi_\theta(a_i|s_i)$ 需要沿生成轨迹积分变量变换公式中的 Jacobian 迹，计算代价高且数值不稳定。这就是 **likelihood gap**——标准 RL 算法无法直接用于 flow-based VLA。

---

## 三、核心方法

π-StepNFT 是一个**两阶段交替**的在线 RL 框架（见 Algorithm 1）：Phase 1 收集数据，Phase 2 优化策略。

### 3.1 数据收集：SDE Rollout + 逐步记录

对每个任务，用 rollout 策略 $\pi_{\theta^{\text{old}}}$ 在环境中执行 $H$ 步。在每个环境步 $i$：

1. 运行 $K$ 步 Flow-SDE solver，生成 denoising 链 $\{x_{t_j}\}_{j=0}^{K}$
2. 均匀采样一个 solver 步 $j \sim \mathcal{U}\{0, \ldots, K-1\}$，记录转移 $(x_t, x_{t^-}) = (x_{t_j}, x_{t_{j+1}})$
3. 记录 rollout 向量场 $v^{\text{old}}_t = \pi_{\theta^{\text{old}}}(c, s_i, x_t, t)$
4. 执行最终动作 $x_{t_K}$，收集环境反馈

Episode 结束后获得终端信号 $r \in \{0, 1\}$（成功/失败），所有 $(x_t, x_{t^-}, v^{\text{old}}_t, t, s, c, r)$ 存入缓冲区 $\mathcal{D}$。

**为什么只采样一个 solver 步？** 效率考虑。具身控制中 $K$ 通常很小（论文用 $K=4$），随机采样保证所有 denoising 阶段都能被覆盖。消融实验（Appendix D）表明随机选择优于固定某一步。

### 3.2 镜像分支构造（Mirror Errors）

这是 π-StepNFT 的基础构件，继承自 Diffusion-NFT 的思想。

给定当前策略预测 $v_\theta = \pi_\theta(c, s, x_t, t)$ 和 rollout 策略预测 $v^{\text{old}}$，定义更新方向 $\Delta v_\theta = v_\theta - v^{\text{old}}$，然后构造两个**关于 $v^{\text{old}}$ 对称的镜像分支**：

$$v^+_\theta = (1-\beta)v^{\text{old}} + \beta v_\theta = v^{\text{old}} + \beta \Delta v_\theta \tag{4}$$

$$v^-_\theta = (1+\beta)v^{\text{old}} - \beta v_\theta = v^{\text{old}} - \beta \Delta v_\theta \tag{5}$$

其中 $\beta > 0$ 是信任域超参数。对称性保证 $v^+_\theta - v^{\text{old}} = v^{\text{old}} - v^-_\theta = \beta \Delta v_\theta$。

**直觉**：$v^+$ 代表"沿更新方向走一步"的假设，$v^-$ 代表"反方向走一步"的假设。通过比较这两个假设对观测到的转移的解释能力，我们可以判断更新方向是否正确。

由于仿射结构（Eq. 3），两个镜像速度诱导两个高斯均值 $\mu^\pm_{\theta,t}$，共享协方差 $\Sigma_t$。然后计算**方差归一化的步误差**：

$$E^+_{\theta,t} = \|x_{t^-} - \mu^+_{\theta,t}\|^2_{\Sigma_t^{-1}}, \quad E^-_{\theta,t} = \|x_{t^-} - \mu^-_{\theta,t}\|^2_{\Sigma_t^{-1}} \tag{6}$$

$E^+$ 衡量正向分支对观测转移的拟合度，$E^-$ 衡量反向分支的拟合度。用 $\Sigma_t$ 归一化可以**稳定不同时间步的梯度尺度**。

### 3.3 逐步对比排序目标（Step-wise Contrastive Objective）

给定采样的 solver 转移 $(x_t \to x_{t^-})$ 及 episode 标签 $y = 2r - 1 \in \{-1, +1\}$，π-StepNFT 的损失为：

$$\ell_t(\theta) = \text{softplus}\left(\frac{1}{2}y \cdot (E^+_{\theta,t} - E^-_{\theta,t})\right) \tag{7}$$

**含义**：

- 成功 episode（$y = +1$）：希望 $E^+ < E^-$，即正向分支比反向分支更好地解释观测转移 → 更新方向正确
- 失败 episode（$y = -1$）：希望 $E^+ > E^-$，即正向分支对观测转移解释更差 → 应该反转更新方向

**与似然比的关系**（Lemma 4.2）：由于两个分支共享协方差 $\Sigma_t$，误差差等于对数似然比：

$$\log \frac{q^+_{\theta,t}(x_{t^-} | x_t, c)}{q^-_{\theta,t}(x_{t^-} | x_t, c)} = -\frac{1}{2}(E^+_{\theta,t} - E^-_{\theta,t}) \tag{8}$$

所以**最小化 $\ell_t$ 实际上是在调整构造的转移似然比**，让它与 episode 标签一致。

### 3.4 理论保证：梯度方向与 Oracle 对齐

**Theorem 4.4** 是论文最核心的理论结果，包含三个层次：

**(a) 误差差的闭式表达**：

$$E^+_{\theta,t} - E^-_{\theta,t} = -4\langle\Sigma_t^{-1} e_t,\ d_t\rangle \tag{9}$$

其中 $e_t = x_{t^-} - \mu^{\text{old}}_t$（rollout 残差），$d_t = \mu^+_{\theta,t} - \mu^{\text{old}}_t = \beta B_t \Delta v_\theta$（均值位移）。

用大白话说：误差差完全由残差 $e_t$ 与位移 $d_t$ 的**内积**决定。如果策略更新的方向与残差对齐，排序信号就越强。

**(b) 梯度形式**：

$$-\nabla_\theta \ell_t(\theta) \propto \sigma(z_t) \cdot y \cdot \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top B_t \Sigma_t^{-1} e_t \tag{10}$$

梯度方向由残差 $e_t$ 经 $\Sigma_t^{-1}$ 归一化后、再通过仿射系数 $B_t$ 和策略 Jacobian 传回参数空间。

**(c) 小步对齐**：在二元成功信号下且更新较小时，条件期望梯度方向与 **oracle 均值差** $\Delta \mu^\star_t$ 对齐：

$$\mathbb{E}[-\nabla_\theta \ell_t(\theta) | x_t, c] \parallel \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top B_t \Sigma_t^{-1} \Delta \mu^\star_t(x_t, c) \tag{11}$$

$\Delta \mu^\star_t$ 是 oracle 后验分割（将 rollout 后验分解为成功条件和失败条件两个分支后的均值差），代表理想的局部改进方向。论文证明我们的可计算代理梯度在期望意义下指向这个理想方向。

### 3.5 与 Diffusion-NFT（Weighted-MSE）的对比

Diffusion-NFT 使用 reward-weighted MSE 目标：

$$\ell^{\text{wMSE}}_t(\theta) = r \cdot E^+_{\theta,t} + (1-r) \cdot E^-_{\theta,t}$$

**Theorem 4.5** 揭示了这个目标的分解：

$$\ell^{\text{wMSE}}_t(\theta) = \text{const} - 2y\langle\Sigma_t^{-1} e_t, d_t\rangle + \|d_t\|^2_{\Sigma_t^{-1}} \tag{12}$$

对比 π-StepNFT 的核心项（Eq. 9）：$E^+ - E^- = -4\langle\Sigma_t^{-1} e_t, d_t\rangle$，可以看到 wMSE 多了一个 $\|d_t\|^2_{\Sigma_t^{-1}}$ 项，这就是**隐式分离惩罚（implicit separation penalty）**。

这个惩罚项的问题：
- 它**与标签 $y$ 无关**，无条件地抑制分支分离
- 即使数据强烈指示应该做一个大的修正步（$e_t$ 和 $d_t$ 高度对齐），惩罚项也会压制更新幅度
- 在二元奖励下，wMSE 退化为只拟合一个分支（$r=1$ 时只拟合 $E^+$），无法同时利用正负信号

**π-StepNFT 的 Push-Pull 动力学**：对比排序损失同时"拉近"正分支、"推远"负分支，产生更强的分离梯度和更快的收敛。这是与 wMSE 的本质区别。

### 3.6 完整训练流程

> 1. 初始化 $\theta \leftarrow \theta^{\text{old}}$，清空缓冲区 $\mathcal{D}$
> 2. **数据收集**：用 $\pi_{\theta^{\text{old}}}$ 做 SDE rollout，对每个环境步随机采样一个 solver 转移，记录 $(x_t, x_{t^-}, v^{\text{old}}_t, t, s, c)$ 和 episode 终端奖励 $r$
> 3. **优化**：对缓冲区中的 mini-batch 计算：
>    - 当前策略预测 $v_{\theta,t} \leftarrow \pi_\theta(c, s, x_t, t)$
>    - 更新方向 $\Delta v_\theta \leftarrow v_{\theta,t} - v^{\text{old}}_t$
>    - 镜像分支 $v^\pm_\theta \leftarrow v^{\text{old}}_t \pm \beta \Delta v_\theta$
>    - 均值和方差 $\mu^\pm_{\theta,t}, \Sigma_t$
>    - 步误差 $E^\pm_{\theta,t}$
>    - 总损失 $\mathcal{L}_{\text{total}} = \text{softplus}\left(\frac{1}{2}y \cdot \Delta E_\theta\right) + \lambda_{\text{TR}}\|\Delta v_\theta\|^2$
>    - 梯度下降更新 $\theta$
> 4. **EMA 更新** rollout 策略：$\theta^{\text{old}} \leftarrow \alpha_m \theta^{\text{old}} + (1-\alpha_m)\theta$
> 5. 清空缓冲区，回到步骤 2

**关键设计细节**：
- 冻结 VLM backbone，只微调 action expert（约 300M 参数）
- 信任域正则 $\lambda_{\text{TR}}\|\Delta v_\theta\|^2$ 防止偏离 rollout 策略太远
- EMA 衰减率从 0.1 动态增长到 0.995，平衡早期加速与后期稳定

---

## 四、实验结果

### 4.1 实验设置

**模型**：π₀ 和 π₀.₅（PaliGemma-3B backbone + ~300M flow-matching action expert）

**基准**：
- **LIBERO**：4 个任务套件（Spatial、Object、Goal、Long），每套件 10 个子任务 × 50 个初始状态 = 500 episodes。Few-shot SFT 初始化（π₀ 用 58-208 条轨迹，π₀.₅ 用 40 条）
- **ManiSkill**：PutOnPlateInScene，4352 个组合任务（16 物体 × 17 容器 × 16 场景），测试 IND 和 OOD 泛化

**硬件**：主实验 8× H100 80GB，消融实验 8× RTX 4090 48GB

### 4.2 LIBERO：Few-shot SFT 后的潜力释放

| 模型 | Spatial | Object | Goal | Long | Avg. | Δ Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| **π₀ SFT** | 65.3 | 64.4 | 49.8 | 51.2 | 57.6 | — |
| πRL (PPO) | 98.4 | 99.4 | 96.2 | 90.2 | 96.0 | +38.4 |
| πRL (GRPO) | 97.8 | 97.8 | 83.2 | 81.4 | 90.0 | +32.4 |
| **π-StepNFT** | 93.5 | 98.0 | 83.7 | 86.7 | **90.5** | **+32.9** |
| **π₀.₅ SFT** | 84.6 | 95.4 | 84.6 | 43.9 | 77.1 | — |
| πRL (PPO) | 99.6 | 100 | 98.8 | 93.0 | 97.9 | +20.8 |
| πRL (GRPO) | 97.4 | 99.8 | 91.2 | 77.6 | 91.5 | +14.4 |
| **π-StepNFT** | 97.8 | 100 | 98.2 | 79.8 | **94.0** | **+16.9** |

**关键观察**：
- π-StepNFT 在无 Critic、无似然的条件下，达到了与 PPO 可比的性能
- 在短时域任务（Object）上与 PPO 打平，说明逐步监督在局部修正能力上非常有效
- 比同为无 Critic 的 GRPO 在 Long 任务上显著更好（π₀: 86.7% vs 81.4%），说明逐步监督提供了比 GRPO 更精细的信用分配
- PPO 在 Long 任务上仍有优势，因为 Critic 提供了时间信用分配

### 4.3 ManiSkill：无 Critic 的 OOD 泛化优势

| 模型 | IND | Vision (OOD) | Semantic (OOD) | Execution (OOD) | OOD Avg. |
| --- | --- | --- | --- | --- | --- |
| π₀ Full SFT | 38.4 | 32.6 | 8.4 | 13.2 | 18.1 |
| πRL (PPO) | 78.8 | 61.1 | 25.4 | 31.5 | 39.3 |
| **π-StepNFT** | **79.2** | **69.1** | **49.1** | 33.1 | **50.4** |
| π₀.₅ Full SFT | 40.1 | 40.2 | 16.6 | 22.4 | 26.4 |
| πRL (PPO) | 90.9 | 68.0 | 34.5 | 45.4 | 49.3 |
| **π-StepNFT** | 85.4 | **76.9** | **56.6** | 45.1 | **59.5** |

**核心发现**：
- IND 性能与 PPO 相当，但 **OOD 全面领先**
- π₀ 上 OOD 平均 50.4% vs PPO 的 39.3%（**+11.1%**）
- Semantic OOD（未见物体/指令）上差距最大：49.1% vs 25.4%，**几乎翻倍**
- **原因分析**：PPO 的 Critic 从视觉-语言嵌入估计价值，容易过拟合到训练分布中的视觉纹理和特定语言表述。π-StepNFT 完全依赖**真实环境反馈**（二元成功信号），避免了 Critic 引入的多模态过拟合

### 4.4 消融实验

#### 4.4.1 SDE vs ODE 探索

| 采样策略 | 效果 |
| --- | --- |
| 确定性 ODE | 早期就平台期，受限于窄 manifold |
| SDE 无均值修正 | 探索更宽但学习信号未对齐 |
| SDE + 均值修正 | 显著提升，噪声感知的学习信号是关键 |

**结论**：有效探索不仅需要遍历更宽的空间，还需要学习信号能数学上将噪声转移对齐回策略的向量场。

#### 4.4.2 逐步监督 vs 终端监督

| 监督目标 | 稳定性 | 收敛速度 |
| --- | --- | --- |
| $x_0$（终端，$\sigma_0=0.9$） | 不稳定，需保守同步 | 慢 |
| $x_0$（终端，$\sigma_0=0.1$） | 略好但仍不稳定 | 慢 |
| $x_{t^-}$（逐步，$\sigma_0=0.1$） | **稳定**，激进更新下也不崩溃 | **快** |

**结论**：精确的局部监督是抵消 SDE 探索引入的分布偏移的关键。终端监督的梯度太粗糙，无法维持 manifold 上的有效学习。

#### 4.4.3 对比排序 vs wMSE

- wMSE 在二元奖励下退化为单分支拟合，无法利用正负信号
- 对比排序同时利用 Positive 和 Negative 分支，产生 push-pull 动力学
- 单独用 Positive 或 Negative 分支都有部分提升，结合两者效果最好

#### 4.4.4 无需 Critic

- 二元轨迹级奖励 vs 归一化 GRPO 优势 vs 归一化 GAE 优势
- 二元信号产生更平滑的训练动态，因为利用的是**准确的环境 ground-truth**
- 有界的成功概率 $r \in [0,1]$ 绕过了无界优势分数需要的复杂归一化和裁剪

#### 4.4.5 超参数敏感性

- **噪声水平 $\sigma$**：0.2 最优。太大阻碍收敛（搜索空间过大），太小限制探索
- **信任域 $\beta$**：$[1.0, 2.0]$ 最优。太大违反局部线性假设，太小梯度不稳
- **EMA 衰减 $\alpha$**：动态策略（0.1 → 0.995）最优。常数或过高/过低衰减都不好

---

## 五、局限性与未来方向

1. **长时域任务的信用分配**：π-StepNFT 使用 episode 级别的二元奖励，在 Long-horizon 任务上不如 PPO 的 Critic 提供的时间信用分配。论文指出可以无缝替换为离线学习的逐步成功概率预测器
2. **探索效率**：SDE 注入均匀噪声，未考虑任务结构，可能在高维动作空间中探索效率较低
3. **真实世界验证**：实验全部在仿真中完成，真实机器人上的效果有待验证
4. **与 Critic 方法的结合**：论文定位为 Critic-free 方案，但在 IND 场景下 PPO 仍有优势，两者的互补可能是有价值的方向

---

## 六、个人思考

### 6.1 与项目中已有论文的联系

**与 Diffusion-NFT 的关系**：π-StepNFT 本质上是将 Diffusion-NFT 的镜像构造从图像生成迁移到具身控制，但做了三个关键改进：ODE→SDE、终端→逐步、wMSE→排序。论文的贡献更多是**发现并解决领域迁移中的关键差距**，而非提出全新框架。

**与 πRL 的关系**：πRL 同样将 ODE 转为 SDE 来近似似然，但仍依赖 PPO + Critic。π-StepNFT 走了一条更极端的路——完全绕过似然和 Critic。ManiSkill OOD 结果表明，在需要泛化的场景下，去掉 Critic 反而是优势。

**与 GRPO 系列（SimpleVLA-RL、TGRPO）的对比**：这些方法也是无 Critic 的，但它们在 episode 级别用组相对优势做信用分配，而 π-StepNFT 的逐步监督提供了更细粒度的信号。Long 任务上的优势验证了这一点。

**与 FPO++ 的对比**：FPO++ 用 CFM 损失差值来近似似然比，属于"近似似然"路线；π-StepNFT 用镜像构造完全绕过似然，属于"无似然"路线。两者代表了 flow policy RL 的两条技术路线。

### 6.2 方法论洞察

论文最有价值的 insight 是**"探索宽度与监督粒度必须匹配"**这一原则。这不仅适用于 flow-based VLA，可能是一个更普遍的 RL 设计原则——任何扩大探索范围的策略（如更高的温度、更多的噪声注入）都需要配套更精细的监督信号。

### 6.3 实用性评估

π-StepNFT 的实用优势明显：
- 单次前向传播/优化步（vs PPO 需要 Critic 前向+反向）
- 无需训练和维护 value network
- 在 OOD 场景下泛化更好

但也有明显局限：
- 二元奖励在长时域任务上的信用分配不如 Critic
- IND 性能被 PPO 压制
- 需要仔细调节 $\sigma$、$\beta$、$\alpha$ 三个超参数

---

## 参考

- **Diffusion-NFT**（Zheng et al., 2025）：提出镜像构造和无似然优化的原始框架，π-StepNFT 的直接基础
- **πRL**（Chen et al., 2025）：Flow-SDE 采样 + PPO 的在线 RL 方案，π-StepNFT 使用其 SFT checkpoint 初始化
- **π₀ / π₀.₅**（Black et al., 2026/2025）：实验使用的基础 VLA 模型
- **Diffusion-DPO**（Wallace et al., 2024）：扩散模型的偏好优化，对比排序思想的来源之一
- **RL4VLA**（Liu et al., 2026）：ManiSkill 基准设置的来源
- **RLinf**（Yu et al., 2025）：π-StepNFT 的实现基于此 RL 训练框架
