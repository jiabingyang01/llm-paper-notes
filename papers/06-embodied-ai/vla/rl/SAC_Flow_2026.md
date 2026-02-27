# SAC Flow：基于速度重参数化序列建模的 Flow 策略高效强化学习——原理详解

> 论文：*SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling*
> 机构：清华大学、Carnegie Mellon University、理想汽车、上海 AI Lab
> 发布时间：2026年1月
> 🔗 [arXiv](https://arxiv.org/abs/2509.25756) | [PDF](https://arxiv.org/pdf/2509.25756) | [代码](https://github.com/Elessar123/SAC-FLOW)

---

## 一、论文要解决什么问题？

### 1.1 Flow-based 策略的兴起

在连续控制和机器人操作领域，**高斯策略**（Gaussian Policy）是主流选择——动作分布用一个均值 + 方差的高斯来表示。但高斯是单峰分布，无法捕捉**多模态的动作分布**。比如机器人绕过障碍物可以走左边也可以走右边，两种路径都合理，但高斯只能给出一个"折中"的平均路径（可能直接撞上去）。

**Diffusion Policy（扩散策略）** 通过多步去噪来表示任意分布，解决了多模态问题，但训练和推理都很慢。**Flow-based Policy（流策略）** 是更轻量的替代方案——基于 Flow Matching 训练，通过确定性 ODE 积分从噪声映射到动作，训练更简单，推理更快，质量接近甚至超过扩散策略。

### 1.2 Flow 策略 + Off-policy RL 的核心矛盾

Flow 策略最初在**模仿学习**中大获成功，但纯模仿学习受限于数据覆盖和质量，无法超越示教者性能。自然的下一步是用**强化学习（RL）** 来训练 Flow 策略。

- **On-policy 方法**（如 PPO 适配 Flow）效果好，但**样本效率极低**
- **Off-policy 方法**（如 SAC、TD3）样本效率高得多，但遇到了一个致命问题

问题出在 Flow 策略的**动作采样过程**。Flow 策略通过 $K$ 步 Euler 积分从高斯噪声生成动作：

$$A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot v_\theta(t_i, A_{t_i}, s)$$

Off-policy RL（如 SAC）需要将 Q 函数的梯度**反向传播**穿过这 $K$ 步积分。这就像让梯度穿过一个 $K$ 层的递归网络——**梯度爆炸和消失**问题不可避免。

### 1.3 已有的妥协方案

为了绕开梯度不稳定问题，先前工作要么：

1. **代理目标（Surrogate Objectives）**：不对完整 Flow rollout 求导，而是设计替代损失函数（如 FlowRL 的 Wasserstein 约束）——但脱离了原始 SAC 目标，削弱了表达能力
2. **策略蒸馏（Policy Distillation）**：将 Flow 策略蒸馏为简单的单步 Actor 来做 RL 更新（如 FQL）——但丢失了 Flow 策略的多模态优势

两种方案本质上都是**回避问题**而非解决问题。

### 1.4 SAC Flow 的核心洞察

SAC Flow 提出了一个全新视角：**Flow 策略本质上是一个序列模型**。

将 Flow rollout 的 $K$ 步 Euler 更新写出来：

$$A_{t_{i+1}} = A_{t_i} + f_\theta(t_i, A_{t_i}, s)$$

这不就是一个**残差 RNN** 吗？中间动作 $A_{t_i}$ 是隐状态，$(t_i, s)$ 是输入，$f_\theta$ 是 RNN cell。

既然问题等价于 RNN 的梯度不稳定，那解决方案也很明确——借鉴 **GRU 和 Transformer** 等现代序列模型的稳定梯度设计，**重参数化速度网络**。

---

## 二、预备知识

### 2.1 强化学习基本设定

标准 MDP：$\langle \mathcal{S}, \mathcal{A}, p, r, \rho \rangle$，连续状态空间和动作空间。目标是学习最优策略：

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[\sum_{h=0}^{\infty} \gamma^h r_h\right]$$

### 2.2 Soft Actor-Critic（SAC）

SAC 在标准目标上加入**熵正则化**，鼓励策略保持随机性以促进探索：

$$\hat{J}(\pi) = \mathbb{E}_\pi \left[\sum_{h=0}^{\infty} \gamma^h (r_h + \alpha \mathcal{H})\right]$$

其中 $\mathcal{H}(\pi(\cdot|s_h)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 是策略熵，$\alpha$ 是温度系数。

**Critic 更新**（TD 损失）：

$$L(\psi) = \left[Q_\psi(s_h, a_h) - (r_h + \gamma Q_{\bar{\psi}}(s_{h+1}, a_{h+1}) - \alpha \log \pi_\theta(a_{h+1}|s_{h+1}))\right]^2$$

**Actor 更新**（策略梯度）：

$$L(\theta) = \alpha \log \pi_\theta(a_h^\theta | s_h) - Q_\psi(s_h, a_h^\theta)$$

关键：$a_h^\theta$ 是**重参数化**的动作采样，梯度需要从 $Q$ 穿过动作传回策略参数 $\theta$。

### 2.3 Flow-based Policy

Flow 策略通过一个时间索引的映射 $\varrho: [0,1] \times \mathcal{A} \times \mathcal{S} \to \mathcal{A}$，将简单基分布 $p_0(\cdot|s) = \mathcal{N}(0, I_d)$ 传输到目标策略分布 $p_1(\cdot|s)$。

采用 **Rectified Flow**，使用直线路径 $A_t = (1-t)A_0 + tA_1$，对应 Flow Matching 训练目标：

$$\hat{\theta} = \arg\min_\theta \mathbb{E}_{A_0, A_1, t} \left[\|A_1 - A_0 - v_\theta(t, (1-t)A_0 + tA_1, s)\|_2^2\right]$$

推理时通过 $K$ 步 Euler 积分生成动作：

$$A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot v_\theta(t_i, A_{t_i}, s), \quad 0 = t_0 < \cdots < t_K = 1$$

最终 $a = A_1 \sim \pi_\theta(\cdot|s)$ 就是策略输出的动作。

---

## 三、方法论详解：从 Flow Rollout 到序列模型

### 3.1 关键洞察：Flow Rollout ≡ 残差 RNN

将中间动作 $A_{t_i}$ 视为隐状态，$(t_i, s)$ 视为输入，则 Euler 积分步：

$$A_{t_{i+1}} = A_{t_i} + f_\theta(t_i, A_{t_i}, s), \quad f_\theta(\cdot) = \Delta t_i \cdot v_\theta(\cdot)$$

这就是一个**残差 RNN**。因此，用 off-policy 损失（如 SAC）训练 Flow 策略时，梯度需要反向传播穿过 $K$ 层递归更新——与 vanilla RNN 一样，会出现**梯度爆炸**。

实验验证（Fig. 6a）：随着反向传播步数 $k$ 从 3 到 0，naive SAC Flow 的平均梯度范数从 2.32 飙升到 27.17（>10 倍放大）。而 SAC Flow-G 和 Flow-T 的梯度范数在整个反向传播路径上最大变化仅 0.29。

### 3.2 Flow-G：GRU 风格的门控速度

借鉴 GRU 的门控机制，为速度网络引入**更新门** $g_i$：

$$g_i = \text{Sigmoid}(z_\theta(t_i, A_{t_i}, s))$$

$$A_{t_{i+1}} = A_{t_i} + \Delta t_i \left(g_i \odot (\hat{v}_\theta(t_i, A_{t_i}, s) - A_{t_i})\right)$$

其中 $\hat{v}_\theta$ 是候选网络，$\odot$ 是逐元素乘法。

**直觉理解**：

- $g_i \approx 0$：门关闭，保持当前中间动作不变（"什么都不做"）
- $g_i \approx 1$：门打开，用候选网络的输出重写当前动作（"大幅更新"）

门控机制自适应地控制每步更新幅度，防止梯度在反向传播时被无限放大。等价的速度形式为：

$$v_\theta = g_i \odot (\hat{v}_\theta - A_{t_i})$$

这是标准 Flow rollout $A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot v_\theta$ 的直接替换，不改变算法框架。

**实现细节**：
- 门网络和候选网络各使用一个 MLP（隐藏维度 128，Swish 激活）
- 门的偏置初始化为 5.0（初始时 $g_i \approx 1$，接近无门控的原始行为）
- 候选输出用 $50 \cdot \tanh(\cdot)$ 限幅

### 3.3 Flow-T：Transformer Decoder 风格的解码速度

用 Transformer Decoder 架构参数化速度函数，通过**交叉注意力**条件在环境状态 $s$ 上：

**第一步：嵌入**

$$\Phi_{A_i} = E_A(\phi_t(t_i), A_{t_i}), \quad \Phi_S = E_S(\phi_s(s))$$

其中 $E_A, E_S$ 是线性投影，$\phi_t, \phi_s$ 是位置/特征编码器。

**第二步：$L$ 层 Pre-Norm Decoder Block**

每层执行对角 self-attention（不混合时间步之间的信息）+ state-conditioned cross-attention + FFN：

$$Y_i^{(l)} = \Phi_{A_i}^{(l-1)} + \text{Cross}_l(\text{LN}(\Phi_{A_i}^{(l-1)}), \text{context} = \text{LN}(\Phi_S))$$

$$\Phi_{A_i}^{(l)} = Y_i^{(l)} + \text{FFN}_l(\text{LN}(Y_i^{(l)}))$$

**第三步：投影到速度空间**

$$v_\theta(t_i, A_{t_i}, s) = W_o(\text{LN}(\Phi_{A_i}^{(L)}))$$

然后代入标准 Euler 步：

$$A_{t_{i+1}} = A_{t_i} + \Delta t_i \cdot W_o(\text{LN}(\Phi_{A_i}^{(L)}))$$

**关键设计**：self-attention 使用对角 mask，每个 action-time token 只处理自己，不跨时间步混合信息。上下文整合完全依赖对共享状态嵌入 $\Phi_S$ 的 cross-attention。这保持了 Flow 的 Markov 性质，同时利用 Transformer 的 pre-norm 残差结构和注意力机制提供了稳定的梯度流。

**实现细节**：$d=64$, $n_H=4$ heads, $n_L=2$ layers, 观测编码器 32 → SiLU → 64。

### 3.4 为什么这两种设计能稳定梯度？

三种速度网络在序列模型视角下的对应关系：

| Flow 速度参数化 | 等价序列模型 | 梯度稳定性 |
| --- | --- | --- |
| MLP（标准） | 残差 RNN | ❌ 梯度爆炸 |
| **Flow-G（门控）** | **GRU** | ✅ 门控调节更新幅度 |
| **Flow-T（解码器）** | **Transformer Decoder** | ✅ Pre-norm 残差 + 注意力 |

这三种参数化都是 $v_\theta$ 的**即插即用替换**，不改变 Flow rollout 框架和外围算法。

---

## 四、训练 Flow 策略的 SAC 算法

梯度稳定性解决后，下一个关键问题是：SAC 的熵正则化需要计算 $\log \pi_\theta(a|s)$，但 $K$ 步确定性 rollout 的密度是**不可解析的**。

### 4.1 噪声增强 Rollout

核心思路：将确定性 rollout 变为**随机 rollout**，同时保持最终动作的边际分布不变。

在每步 Euler 更新中注入高斯噪声并加入补偿漂移：

$$A_{t_{i+1}} = A_{t_i} + b_\theta(t_i, A_{t_i}, s)\Delta t_i + \sigma_\theta \sqrt{\Delta t_i}\varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, I_d)$$

其中补偿漂移 $b_\theta$ 适当放大原始速度以抵消扩散效应：

$$b_\theta(t_i, A_{t_i}, s) = \frac{1 - t_i + \frac{t_i \sigma_\theta^2}{2}}{1 - t_i} v_\theta(t_i, A_{t_i}, s) - \frac{t_i \sigma_\theta^2}{2(1-t_i)t_i} A_{t_i}$$

直觉：第一项放大速度以对抗扩散带来的偏移，第二项向直线路径收缩以保证终端分布不变。

### 4.2 可解析的联合路径密度

经过噪声增强后，每步转移变成高斯条件分布：

$$\eta_\theta(A_{t_{i+1}} | A_{t_i}, s; \Delta t_i) = \mathcal{N}(A_{t_i} + b_\theta \Delta t_i, \sigma_\theta^2 \Delta t_i I_d)$$

联合路径密度分解为：

$$p_c(\mathcal{A}|s) = \zeta(A_{t_0}) \prod_{i=0}^{K-1} \eta_\theta(A_{t_{i+1}} | A_{t_i}, s; \Delta t_i) \cdot \|\det \mathcal{J}(a)\|^{-1}$$

其中 $\zeta$ 是标准高斯基分布，$\mathcal{J}(a)$ 是 tanh 压缩的 Jacobian。每一项都是闭式高斯密度——完美解决了似然计算的难题。

### 4.3 From-Scratch 训练

**Actor 损失**：

$$L_{\text{actor}}(\theta) = \alpha \log p_c(\mathcal{A}^\theta | s_h) - Q_\psi(s_h, a_h^\theta)$$

**Critic 损失**：

$$L_{\text{critic}}(\psi) = \left[Q_\psi(s_h, a_h) - (r_h + \gamma Q_{\bar{\psi}}(s_{h+1}, a_{h+1}) - \alpha \log p_c(\mathcal{A}_{h+1} | s_{h+1}))\right]^2$$

其中 $a_h^\theta = \tanh(A_{t_K}^\theta)$，$\mathcal{A}^\theta$ 是完整的中间动作路径。

### 4.4 Offline-to-Online 训练

对于稀疏奖励任务，在 Actor 损失中加入**近邻正则化**：

$$L_{\text{actor}}^o(\theta) = \alpha \log p_c(\mathcal{A}^\theta | s_h) - Q_\psi(s_h, a_h^\theta) + \beta \|a_h^\theta - a_h\|_2^2$$

训练流程：先用 Flow Matching 目标在离线数据上预训练，然后切换到带近邻正则的在线 SAC 训练。$\beta$ 在离线阶段设大值以保持保守，在线阶段逐步减小以释放探索。

---

## 五、实验设置

### 5.1 评估环境

三大基准，覆盖从运动控制到机器人操作：

| 基准 | 任务 | 奖励类型 | 用途 |
| --- | --- | --- | --- |
| **MuJoCo** | Hopper, Walker2D, HalfCheetah, Ant, Humanoid, HumanoidStandup | 密集 | From-scratch 训练 |
| **OGBench** | Cube-Double/Triple/Quadruple（UR-5 机械臂多积木放置） | 稀疏 | Offline-to-Online |
| **Robomimic** | Lift, Can, Square（机器人抓放、操作） | 稀疏 | Offline-to-Online |

### 5.2 基线方法

**From-scratch 基线**：

| 方法 | 类型 | 核心思路 |
| --- | --- | --- |
| **QSM** | Diffusion + RL | 用 Q 函数梯度匹配扩散策略的 score function |
| **DIME** | Diffusion + RL | 最大熵 RL，通过 KL 散度最小化训练扩散策略 |
| **FlowRL** | Flow + RL | Wasserstein-2 约束下直接最大化 Q 值（当前 SOTA） |
| **SAC** | 高斯策略 | 经典 off-policy RL |
| **PPO** | 高斯策略 | 经典 on-policy RL |

**Offline-to-Online 基线**：

| 方法 | 类型 | 核心思路 |
| --- | --- | --- |
| **ReinFlow** | Flow + On-policy | 注入噪声网络实现似然计算，用 PPO 微调 |
| **FQL** | Flow + Off-policy | 将 Flow 蒸馏为单步策略，用 SAC 更新 |
| **QC-FQL** | Flow + Off-policy | FQL 扩展至动作分块（action chunking） |

### 5.3 关键超参数

- 采样步数 $K = 4$（所有 Flow 方法统一）
- 噪声标准差 $\sigma_\theta = 0.10$（固定）
- 优化器：Adam（$b_1 = 0.5$ for Flow）
- Batch size：512（from-scratch）/ 256（offline-to-online）
- 所有实验跑 5 个随机种子，报告 95% 置信区间

---

## 六、实验结果

### 6.1 From-Scratch 训练（MuJoCo）

SAC Flow-T 和 SAC Flow-G 在 6 个 MuJoCo 任务中的 5 个上达到了**最优或可比**的性能：

**核心发现**：

1. **HumanoidStandup 上 130% 提升**：SAC Flow 在最具挑战性的任务上远超所有基线（含 FlowRL SOTA），验证了 Flow 策略的多模态表达力在复杂任务上的价值
2. **稳定收敛**：在 Hopper、Walker2D、HalfCheetah 等简单任务上不仅性能好，收敛曲线也很平滑
3. **样本效率远超 On-policy**：PPO 需要的样本量远高于所有 off-policy 方法
4. **Humanoid 是唯一例外**：DIME 和 FlowRL 在该任务上更好，可能因为 Humanoid 的 17 维动作空间对 Flow rollout 的 4 步采样提出了更高挑战

**稀疏奖励任务的瓶颈**：所有 from-scratch 方法在 Robomimic-Can 和 OGBench-Cube 上均表现不佳（接近 0 成功率），说明**纯 from-scratch 无法应对大探索空间 + 稀疏奖励**，offline-to-online 训练不可或缺。

### 6.2 Offline-to-Online 训练

在稀疏奖励的机器人操作任务上，SAC Flow 展现了显著优势：

**OGBench**（UR-5 机械臂多积木放置）：

- **Cube-Double**：SAC Flow-T 达到 ~80% 成功率，显著优于 FQL (~40%) 和 QC-FQL (~50%)
- **Cube-Triple**：SAC Flow-T 达到 ~70% 成功率，比基线最高高出 **60%**
- **Cube-Quadruple**：SAC Flow-T 仍保持 ~50% 成功率，其他方法几乎为 0

**Robomimic**（Lift / Can / Square）：

- 在严格正则化（$\beta$ 较大）的 Robomimic 上，SAC Flow-G 和 Flow-T 与 QC-FQL **持平**
- 这是因为大 $\beta$ 值限制了 Flow 模型的学习能力，使其退化为类似单步策略的行为
- 但相比 on-policy 基线 ReinFlow，SAC Flow 在 1M 在线步数内**全面超越**，体现了 off-policy 方法的数据效率优势

### 6.3 消融实验

#### 6.3.1 速度网络参数化的效果

| 方法 | 采样步 0 梯度范数 | 采样步 3 梯度范数 | Ant-v4 性能 |
| --- | --- | --- | --- |
| Naive SAC Flow（MLP） | 27.17 | 2.32 | 训练崩溃 |
| **SAC Flow-T** | 7.42 | 2.44 | ✅ 稳定收敛 |
| **SAC Flow-G** | 7.37 | 2.53 | ✅ 稳定收敛 |

梯度范数从步骤 3（最接近损失）到步骤 0（最远）：
- Naive：放大 **12 倍**（2.32 → 27.17）
- Flow-T/G：变化 **< 0.29**

直接后果：Naive SAC Flow 在 Ant-v4 上训练崩溃（负 reward），在 Cube-Double 上成功率始终为 0。

#### 6.3.2 Flow 采样步数的鲁棒性

| 采样步数 $K$ | Flow-T (Ant-v4) | Flow-G (Ant-v4) |
| --- | --- | --- |
| 4 | ✅ 稳定 | ✅ 稳定 |
| 7 | ✅ 稳定 | ✅ 稳定 |
| 10 | ✅ 稳定 | ✅ 稳定 |

更多采样步数意味着更深的反向传播，对梯度稳定性要求更高。结果表明 Flow-T 和 Flow-G **对步数不敏感**，尤其 Flow-T 在 $K=10$ 时仍保持优异性能。

---

## 七、用类比总结 SAC Flow 的核心原理

想象你在学画画，需要从一团随机笔触（噪声）一步步精炼出一幅好画（动作）。

**标准 Flow 策略**（MLP 速度）：就像让一个没有记忆管理的画家做 4 步精炼。老师（Q 函数）只看最终画作给反馈，但这个反馈从第 4 步往回传时，每经过一步就被放大——到第 1 步时信号已经失真得面目全非。画家完全不知道第 1 步该怎么改。

**SAC Flow-G**（GRU 门控）：给画家每步加了一个"保守旋钮"——如果当前画得还行，就小幅修改（门接近 0）；如果差距大，才大幅重画（门接近 1）。反馈在回传时被旋钮自动衰减，不会失控。

**SAC Flow-T**（Transformer Decoder）：每步精炼时，画家都重新审视参考照片（状态 $s$，通过 cross-attention），然后基于当前画面做有针对性的调整。Transformer 的 pre-norm 残差结构天然保证了信息传递的稳定性。

**噪声增强 rollout 的意义**：SAC 要求画家报告"画出这幅画的概率有多大"。确定性的 4 步精炼无法计算概率。噪声增强就像让画家每步轻微抖手——抖动使得每步都变成了高斯分布，概率就可以算了，而且最终画面的分布与不抖手时完全一致。

---

## 八、与 RISE 的关系与区别

SAC Flow 和 RISE 都是用 RL 训练机器人策略，但思路截然不同：

| 维度 | RISE | SAC Flow |
| --- | --- | --- |
| **核心问题** | 真实世界 RL 成本太高 | Flow 策略的 off-policy RL 梯度不稳定 |
| **解决方案** | 在世界模型的想象空间中做 RL | 重参数化速度网络为 GRU/Transformer |
| **策略架构** | VLA（$\pi_0$ 系列） | Flow-based Policy |
| **RL 算法** | 优势条件化（概率推断框架） | SAC（最大熵 RL） |
| **是否需要世界模型** | ✅ 必须（动力学 + 价值） | ❌ 不需要 |
| **训练环境** | 想象空间（零真实交互） | 仿真环境（MuJoCo、OGBench） |
| **应用场景** | 真实机器人 | 仿真连续控制 + 仿真机器人操作 |

两者可以互补：SAC Flow 提供了稳定训练 Flow 策略的 RL 算法，RISE 提供了在想象空间中做 RL 的框架。未来可能的方向是用 SAC Flow 的稳定训练算法在 RISE 的想象空间中训练 Flow 策略。

---

## 九、局限性与未来方向

### 9.1 Humanoid 任务上的不足

在 17 维动作空间的 Humanoid 上，SAC Flow 不如 DIME 和 FlowRL。这暗示在**极高维动作空间**中，4 步 Flow 采样可能不足以捕获复杂的多模态结构，或者序列模型的参数化需要进一步优化。

### 9.2 Robomimic 上的正则化瓶颈

在严格正则化条件下（大 $\beta$），Flow 策略的表达优势被正则化约束压制，退化为类似单步策略的行为。如何设计**更优雅的正则化策略**以在保守与探索之间取得更好平衡是开放问题。

### 9.3 真实机器人验证

当前实验全部在仿真环境中完成。论文指出未来将在真实机器人上验证 SAC Flow，并探索 sim-to-real 的鲁棒性。

### 9.4 更轻量的序列参数化

Flow-T 的 Transformer Decoder 虽然有效，但引入了额外的计算开销。探索更轻量的序列模型参数化（如线性注意力、状态空间模型）是有价值的方向。
