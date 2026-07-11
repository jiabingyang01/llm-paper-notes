# CLWM：因果隐式世界模型驱动的具身任务自动化学习

> **论文**：*DexWorldModel: Causal Latent World Modeling towards Automated Learning of Embodied Tasks*
>
> **作者**：Yueci Deng、Guiliang Liu、Kui Jia
>
> **机构**：DexForce AI（跨维智能）
>
> **发布时间**：2026 年 04 月（arXiv 2604.16484）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2604.16484) | [PDF](https://arxiv.org/pdf/2604.16484)
>
> **分类标签**：`世界动作模型` `DINOv3隐空间预测` `Test-Time-Training记忆` `Speculative异步推理` `Sim-to-Real`

---

## 一句话总结

CLWM 用冻结的 DINOv3 语义特征替代像素/VAE 隐空间作为世界模型的生成目标以解耦交互语义与视觉噪声,配合 Dual-State TTT 记忆把 KV Cache 替换为 $\mathcal{O}(1)$ 恒定内存、Speculative Asynchronous Inference 把推理阻塞延迟降低约 50%,并用 EmbodiChain 在线仿真数据流训练;在 RoboTwin 上平均成功率达到 94.00%(优于 $\pi_{0.5}$、X-VLA、Motus、LingBot-VA 等基线),并且完全不使用真实数据训练即可 zero-shot 迁移到真实双臂平台(4 项任务 95/90/80/65%),显著超过用 50 条真实演示微调的 $\pi_0$、GR00T N1.5 和 Sim2Real-VLA。

## 一、问题与动机

论文指出标准 VLA(如 RT-2、$\pi_0$、OpenVLA)是纯反应式的前馈策略,把高维视觉理解、物理动力学和低维运控纠缠在同一个表征空间里,限制了显式因果推理能力。近期兴起的 World Action Models(WAM,如 UWM、LingBot-VA、GigaBrain 等)把控制问题重新表述为"先预测未来观测(forward visual dynamics),再从预测状态解码动作(inverse dynamics)"的两阶段因果自回归生成过程,理论上更贴近物理因果性,但现有 WAM 普遍在像素空间或 VAE 隐空间直接建模未来状态,存在三个具体工程瓶颈:

1. **表征冗余**：像素级重建把大量模型容量浪费在光照变化、杂乱背景等任务无关的高频纹理上,不仅计算昂贵,也削弱了 sim-to-real / real-to-real 迁移时的域泛化能力。
2. **$\mathcal{O}(T)$ 内存爆炸**：标准因果 Transformer 的 KV Cache 随自回归步数线性增长,长时程操作(数百到数千步)会导致内存耗尽和推理延迟不断攀升。
3. **串行推理延迟**：传统"感知—计算—执行"流水线中,模型必须等待物理动作执行完毕、拿到新观测后才能启动下一步昂贵的扩散/flow matching 采样,造成机器人频繁空闲等待,难以支持高频闭环控制。

此外,论文认为鲁棒策略的训练还受限于**物理经验生成速率**这一根本瓶颈——真实机器人数据采集成本高、覆盖度有限,单纯扩大静态数据集不足以支撑模型规模的增长("Efficiency Law of Embodied Intelligence",概念沿用自作者团队另一篇工作 Liu et al. 2025)。CLWM + EmbodiChain 就是针对上述四点(表征冗余、内存墙、推理延迟、数据吞吐)分别给出的架构与训练范式方案。

## 二、核心方法

### 2.1 问题形式化

机器人操作被建模为 POMDP。标准 WAM 将因果生成过程拆成两阶段:

前向视觉动力学预测未来观测

$$\hat o_{t+1} \sim p_\theta(\cdot \mid o_{\le t}, a_{<t}, l)$$

逆动力学从预测状态解码动作

$$a_t \sim g_\psi(\cdot \mid o_{\le t}, a_{<t}, \hat o_{t+1}, l)$$

两者都用 Conditional Flow Matching(CFM)实现:从高斯噪声 $\epsilon$ 出发,沿最优传输路径 $x^{(s)}=(1-s)\epsilon+s\cdot x$ 学习恒定目标速度场 $\dot x^{(s)}=x-\epsilon$,训练目标为

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{s,\epsilon,x,c}\big[\|v_\phi(x^{(s)}, s \mid c) - \dot x^{(s)}\|^2\big]$$

**大白话说**：与其让网络直接一步生成图像/动作,不如学一个"速度场",告诉从纯噪声到真实数据这条直线路径上每一点该往哪个方向走多快,推理时用 ODE 数值积分把噪声逐步"擦"成目标。

### 2.2 CLWM：以 DINOv3 语义特征为生成目标

CLWM 用冻结的 DINOv3 提取每帧的语义特征作为生成目标,替代像素/VAE 隐码:

$$f_t = \Phi_{\text{DINO}}(o_t) \in \mathbb{R}^{C\times H' \times W'}, \quad H'=H/P,\ W'=W/P,\ P=16$$

架构采用参数高效的 Mixture-of-Transformers(MoT):Latent Video Model($\phi_{\text{vid}}$)与 Action Model($\phi_{\text{act}}$)共享由 Wan2.2-5B 初始化的核心 Transformer 主干,只在输入/输出投影层保持各自独立:

$$\phi_{\text{vid}} = \phi_{\text{vid}}^{\text{out}}\circ\phi_{\text{share}}\circ\phi_{\text{vid}}^{\text{in}}, \quad \phi_{\text{act}} = \phi_{\text{act}}^{\text{out}}\circ\phi_{\text{share}}\circ\phi_{\text{act}}^{\text{in}}$$

**Stage 1（隐式视频 flow matching）**：在历史记忆 $h_{\le t}$ 和语言指令 $l$ 条件下,对未来 DINOv3 特征 $f_{t+1}$ 做 flow matching 回归:

$$\mathcal{L}_{\text{video}} = \mathbb{E}\big[\|v_{\phi_{\text{vid}}}(f_{t+1}^{(s)}, s \mid h_{\le t}, l) - \dot f_t^{(s)}\|^2\big]$$

**Stage 2（动作 flow matching）**：Action Model 以块大小 $\tau=16$ 解码动作块,条件是历史 $h_t$、语言 $l$、以及 Stage 1 预测出的未来语义 $\hat f_{t+1}$。为了增强对不完美视觉历史的鲁棒性,训练时以 $p=0.5$ 的概率向历史特征注入不同尺度的高斯噪声(history augmentation):

$$\tilde f_{\le t} = \begin{cases} (1-s_{\text{aug}})\epsilon + s_{\text{aug}}\cdot f_{\le t}, & p=0.5,\ s_{\text{aug}}\in[0.5,1] \\ f_{\le t}, & 1-p=0.5 \end{cases}$$

**大白话说**：与其让模型学着重建每一根睫毛、每一处光斑,不如只让它学 DINOv3 这种天然对光照/背景鲁棒的语义特征怎么随时间演化——省下的容量全部用来学"物体和交互怎么变",这是 sim-to-real 泛化能力的关键来源。而训练时故意在历史里掺"脏"特征,是逼模型学会即使输入的历史观测本身带噪声(比如上一步预测不完全准),也能解出正确动作,这对后面的 Speculative Inference 至关重要。

### 2.3 Dual-State TTT 记忆：用隐式权重取代 KV Cache

CLWM 把标准因果注意力的 KV Cache 替换为 Test-Time Training(TTT)层。核心组件是 TTT-MLP:输入 token $z_t$ 经低秩投影 $\theta_K,\theta_V,\theta_Q$(类比标准自注意力的 K/V/Q),用自监督重建损失在线更新隐藏权重 $\mathcal{W}$:

$$\ell_{\text{self}}(\mathcal{W}; z_t) = \|f(\theta_K z_t; \mathcal{W}) - \theta_V z_t\|^2$$

内层模型 $f_{TTT_{mlp}}(x;\mathcal{W})=x+\text{LN}(\text{MLP}(x;\mathcal{W}))$(GELU、4 倍扩张比的两层 MLP),权重更新后用查询投影取输出隐状态 $l_t=f_{TTT_{mlp}}(\theta_Q z_t;\mathcal{W}_t)$,并用可学习门控向量 $\alpha$(初始化为 0.1)防止微调初期性能骤降:

$$f_{TTT}(z_t;\mathcal{W}_t) = \tanh(\alpha)\otimes f_{TTT_{mlp}}(\theta_Q z_t;\mathcal{W}_t) + z_t$$

**大白话说**：KV Cache 是把每一步的历史 token 原样存起来,序列越长越占内存;TTT 则是把整段历史"压缩"进一个小型 MLP 的权重里,来一个新观测就用它做一次梯度下降更新权重,而不是往缓存里再塞一条——内存占用永远是常数,不随时间步数增长。

为适配 flow matching 的两阶段级联生成,论文进一步设计 **Dual-State** 记忆:一个持久的 **Long-Term TTT Memory**(仅在收到真实历史观测/动作时更新)和一个每步临时 fork 出来的 **Working TTT Memory**:

$$\mathcal{W}_t^{\text{long}} = \mathcal{W}_{t-1}^{\text{long}} - \eta\nabla_{\mathcal{W}}\ell_{\text{self}}(\mathcal{W}_{t-1}^{\text{long}}; h_t) \qquad \mathcal{W}_t^{\text{work}} \leftarrow \mathcal{W}_t^{\text{long}}$$

Working Memory 在 ODE 积分的所有中间流时间 $s\in(0,1)$ 内保持冻结(保证数值稳定),仅在 Stage 1 预测出 $\hat f_{t+1}$ 后于 $s=0$ 处做一次瞬时更新,吸收该预测以指导 Stage 2 动作解码:

$$\mathcal{W}_t^{\text{work}\prime} \leftarrow \mathcal{W}_t^{\text{work}} - \eta\nabla_{\mathcal{W}}\ell_{\text{self}}(\mathcal{W}_t^{\text{work}}; \hat f_{t+1})$$

**大白话说**：真实历史(Long-Term)和"模型自己脑补的未来"(Working)必须分开存放——如果把预测值也混进真正的历史记忆里,一步预测错就会污染后面所有决策(因果混淆);Working Memory 就像一张"草稿纸",每步现 fork 现用,更新完就扔,真正的历史账本(Long-Term)永远只记录已发生的事实。

### 2.4 Speculative Asynchronous Inference（SAI）

为打破"等物理执行完→等真实观测→再启动扩散采样"的串行瓶颈,SAI 把去噪过程拆成两段并与物理执行重叠:

- **Phase 1（后台投机预去噪）**：在上一动作块 $a_{t-1}$ 物理执行期间,真实观测 $o_t$ 还未到达,SAI 把上一步已经预测出的语义状态 $\hat f_t$ 当作"代理观测",用它更新 Working TTT Memory,并让 ODE 求解器从 $s=0$ 积分到中间阈值 $s_{\text{mid}}$——这部分计算完全隐藏在机器人物理运动的时间窗口里。
- **Phase 2（瞬时校准）**：物理执行结束、真实观测 $o_t$ 到位后,Long-Term Memory 立即用真实特征校准,替换掉投机上下文,ODE 求解只需从 $s_{\text{mid}}$ 继续积分到 $s=1$,机器人实际等待的只是这一小段精细去噪。

论文强调该策略之所以数值稳定,是因为训练阶段的 history augmentation(2.9 式)已经让模型学会在带噪声/不完美历史条件下仍给出正确方向的速度场,保证了 Phase 1 的"投机"轨迹本身足够可靠,Phase 2 校准才能有效收敛。实测显示 SAI 相比严格串行的自回归基线 LingBot-VA,单动作块推理阻塞延迟降低约 50%。

### 2.5 EmbodiChain：在线数据流训练范式

为解决"物理经验生成速率"这一根本瓶颈,论文提出 EmbodiChain 框架,包含四个环节:

1. **生成式仿真资产**：用生成模型合成原始 3D 网格,再做多目标优化(几何/尺度/坐标系/质量分布/摩擦系数/碰撞体),导出物理仿真可用的 USD 资产;并自动生成场景布局,梯度优化消除物体穿插。
2. **域扩展数据生成**：包括 reachability-aware 采样(在可达工作空间内最大化末端执行器接近方向、接触几何等任务空间特征的多样性,避免遥操作/传统规划器带来的轨迹同质化)、闭环错误恢复(把失败场景的纠正轨迹重新标注注入数据集)、参数化视觉增强(在光照温度、表面 BRDF、传感器漂移等维度做时序一致的随机化)、物理约束生成(保持质量/摩擦等物理参数的一致性)。
3. **Online Data Streaming（ODS）**：仿真与训练之间用无锁环形缓冲区异步读写、CPU/GPU 显存内零拷贝交换,学习者直接消费流式数据,不做磁盘序列化,兼具在线 RL 的"新鲜数据"特性和离线 RL 的显存缓冲稳定性。
4. **Efficiency Law**：论文提出用"经验吞吐量" $\mathcal{E}$(每训练迭代摄入的独特状态-动作对数量)来刻画数据生成速率对策略能力 $\mathcal{I}$ 的影响,主张在给定算力 $C$、参数量 $P$ 下,只有 $\mathcal{E}$ 超过某个阈值 $\tau(C,P)$ 时智能才能有效提升,因此训练目标应转向最大化"动态经验密度"而非扩大静态数据集规模。

## 三、实验结果

**训练与数据配置**：预训练语料聚合 RoboMind、AgiBot World Beta、InternData-A1(公开数据集),动作空间按 LingBot-VA 方式统一为双臂 30 维连续动作($(7\text{末端位姿}+7\text{关节位置}+1\text{夹爪})\times 2$ 臂)。预训练用 AdamW、lr $1\times10^{-4}$、batch 128,约 20 epoch,在 64×NVIDIA H100 上训练约 20 天连续算力;RoboTwin 微调阶段用 25,000 条 EmbodiChain 合成轨迹、40k 迭代、lr $1\times10^{-5}$。后训练阶段完全不使用任何人工采集的真实/下游演示,仅依赖 EmbodiChain 合成数据。

**RoboTwin 仿真基准**(45 项任务,节选如下,完整平均值见原文 Table 1):

| 任务 | $\pi_{0.5}$ | X-VLA | Motus | LingBot-VA | CLWM(Ours) |
|---|---|---|---|---|---|
| Grab Roller | 100% | 100% | 100% | 100% | 100% |
| Stack Blocks Three | 76% | 10% | 95% | 98% | **100%** |
| Handover Mic | 97% | 0% | 63% | 96% | **97%** |
| Pick Diverse Bottles | 71% | 36% | **91%** | 82% | 85% |
| Place Container Plate | 95% | 95% | **99%** | 97% | 91% |
| Open Laptop | 96% | **100%** | 91% | 94% | 93% |
| Hanging Mug | 17% | 27% | 38% | 28% | **40%**（全局最低） |
| Turn Switch | 54% | 61% | **78%** | 44% | 65% |
| **平均（45 任务）** | 76.76% | 72.84% | 87.02% | 91.55% | **94.00%** |

CLWM 在绝大多数任务上取得最优或并列最优成绩,总体平均成功率 94.00% 为四个基线中最高;但并非全面碾压——如 Place Container Plate、Turn Switch、Pick Diverse Bottles、Open Laptop 等任务上 Motus 或 X-VLA 反而更优,Hanging Mug 是所有方法里最难的任务(CLWM 也只有 40%)。

**效率分析**：在 2000 步长时程操作 episode 中,标准 KV-Cache 基线显存随步数线性增长($\mathcal{O}(T)$),而 Dual-State TTT 全程保持平坦的常数显存占用;SAI 相比严格串行自回归基线(LingBot-VA)将端到端阻塞延迟降低约 50%。（论文未报告 SAI 对最终 success rate 是否有影响,仅报告延迟数字。）

**EmbodiChain 消融**(在 Hanging Mug / Turn Switch / Stack Bowls 三个代表性任务上做,ID = 分布内、OOD = 未见物体/纹理/光照/布局):

| 数据生成配置 | ID 成功率 | OOD 成功率 |
|---|---|---|
| 仅空间随机化(Baseline) | 64% | 25% |
| + 视觉增强 | 75% | 42% |
| + 物理约束生成 | 81% | 56% |
| + Reachability-aware 采样（完整版） | **95%** | **82%** |

| 训练配置（5000 迭代, batch 64） | Hanging Mug | Turn Switch | Stack Bowls |
|---|---|---|---|
| 静态基线（1500 条演示） | 62% | 85% | 88% |
| ODS，reuse 上限 213 次（≈匹配静态基线采样次数） | 60% | 84% | 85% |
| ODS，reuse 上限 50 次 | 92% | 92% | 96% |
| ODS，reuse 上限 10 次 | **96%** | **98%** | **98%** |

数据表明当 ODS 的轨迹重放次数被人为拉高到与静态基线相当(213 次)时,性能退化到与静态训练基本持平(甚至更低),说明收益并非来自"多训练几步",而确实来自持续注入新鲜轨迹(重放次数越低、经验吞吐越高,性能提升越明显),这是论文对 Efficiency Law 的核心实证支撑。

**真实机器人部署**（Agilex CobotMagic 双臂平台,4 项任务,CLWM 完全在仿真中训练、zero-shot 部署,基线 $\pi_0$/GR00T N1.5 用 50 条真实演示微调,Sim2Real-VLA 与 CLWM 同样走纯仿真路线）：

| 方法 | 双臂倒水 | 桌面整理 | 物品交接摆放 | 平底锅开合摆放 |
|---|---|---|---|---|
| $\pi_0$ | 25% | 20% | 20% | 5% |
| GR00T N1.5 | 35% | 20% | 15% | 5% |
| Sim2Real-VLA | 80% | 80% | 40% | 35% |
| **CLWM（Ours）** | **95%** | **90%** | **80%** | **65%** |

CLWM 在全部四项真实任务上大幅领先,即便对比的是同样走纯 sim-to-real 路线的 Sim2Real-VLA,也有明显优势;相比用了 50 条真实演示微调的 $\pi_0$/GR00T N1.5 优势更为悬殊。论文未报告每个任务的具体 trial 数与统计方差。

## 四、局限性

论文正文没有专设 Limitations 小节,以下基于方法设计与实验报告的分析性归纳:

1. **隐空间预测保真度未独立验证**：DINOv3 特征分辨率被 patch size $P=16$ 大幅下采样,论文没有报告纯视频/隐特征预测质量本身的指标(如特征空间预测误差),只通过下游动作成功率间接验证,难以判断语义特征是否会丢失精细接触/形变等操作所需的低层视觉线索。
2. **Dual-State TTT 与全注意力的直接对比缺失**：论文只对比了显存占用曲线($\mathcal{O}(1)$ vs $\mathcal{O}(T)$),没有在等量长时程任务上对比 TTT 记忆与完整 KV Cache 注意力的成功率差异,TTT 隐式压缩历史是否会牺牲长程精确检索能力尚不明确。
3. **SAI 对精度的影响未量化**：效率分析只报告了延迟降低约 50%,没有给出开启/关闭 SAI 前后 success rate 的对比消融,无法判断"投机预去噪"是否在某些任务上以精度换速度。
4. **真实机器人评测规模有限**：仅在单一双臂平台(Agilex CobotMagic)、4 个任务上验证,Table 4 只给出成功率百分比,未标注具体 trial 数与标准差,统计显著性存疑。
5. **训练成本高昂**：64×H100 连续训练约 20 天,加上 RoboTwin 微调 40k 迭代,复现门槛较高。
6. **EmbodiChain 细节分散**：资产生成、场景合成、ODS 缓冲区的具体实现更多依赖引用作者团队另一份技术报告和 GitHub 仓库(EmbodiChain Developers, 2025),本文对该部分的描述总体上偏概括,独立可复现性有待检验。
7. **并非全面最优**：在 45 项 RoboTwin 任务中,CLWM 在 Place Container Plate、Turn Switch、Pick Diverse Bottles、Open Laptop、Press Stapler 等若干任务上仍不及 Motus 或 X-VLA,说明该架构的优势并非在所有任务类型上均一体现。

## 五、评价与展望

**优点**：CLWM 把 WAM 的生成目标从像素/VAE 隐空间迁移到自监督语义特征(DINOv3)空间,是对"像素级世界模型"路线(如通用视频生成骨干直接用作 WAM)的一次有效解耦,呼应了近期隐空间世界模型(如 DINO-WM、V-JEPA 系列思路)在操作策略上的落地趋势。将 Test-Time Training 系统性引入自回归 WAM、替代 KV Cache,是较早把 TTT(Sun et al. 2024)与视频流 TTT(Wang et al. 2025)结合进具身 WAM 架构、并针对长时程操作内存墙给出工程化方案的工作之一。Speculative Asynchronous Inference 本质上是"预测性解码"(speculative decoding)思想在 flow matching 动作生成中的类比迁移,把生成式世界模型固有的"预测未来"能力转化为掩藏推理延迟的手段,是有实际部署价值的工程贡献。

**不足与开放问题**：论文反复强调的"Efficiency Law"更多是经验性主张而非严格刻画的 scaling law——Fig. 4 只是示意图,没有实测的 loss-吞吐量拟合曲线,该概念本身沿用自作者团队另一篇同期工作(Liu et al. 2025),独立验证力度有限。真实机器人对比中所有基线都只用 50 条真实演示微调,而 CLWM 完全 zero-shot,这个设定凸显了 EmbodiChain 数据生成的价值,但没有给出"若 CLWM 也用相同的 50 条真实数据继续微调"这一自然的上限对照实验,难以判断纯 sim-to-real 路线相对"sim 预训练 + 少量真实微调"路线的边际收益。CLWM 与同样走纯 sim2real 路线的 Sim2Real-VLA 相比优势明显(95/90/80/65% vs 80/80/40/35%),但这一差距具体来自架构创新(DINOv3 语义目标 + TTT 记忆)还是 EmbodiChain 数据生成质量,论文未做解耦实验。

**与其他公开工作的关系**：CLWM 与 LingBot-VA(Li et al., 2026)、UWM(Zhu et al., 2025)、GigaBrain(2026)等同属"生成式 World Action Model"路线,共享"视频与动作联合以 flow matching 生成"的基本范式,动作空间统一方式直接沿用 LingBot-VA。CLWM 的差异化贡献集中在两个具体工程瓶颈上——用 DINOv3 语义特征解决表征冗余,用 Dual-State TTT 解决内存/延迟——方向具体、目标明确,但截至本文,尚未见到 CLWM 本身的开源代码链接(仅 EmbodiChain 工具链在 GitHub 公开),独立复现和第三方验证还有待观察。

**潜在改进方向**：(1)对隐空间预测保真度做独立定量评估(如特征空间预测误差或探针评测),厘清语义特征作为生成目标的信息损失边界;(2)针对 2000 步以上的真正长时程任务做 Dual-State TTT 记忆能力的成功率评测,而不仅是显存曲线;(3)补充 SAI 开启/关闭对最终动作精度的消融;(4)扩大真实机器人评测的平台数、任务数与 trial 数并报告方差,以及补充"sim 预训练+少量真实微调"这一常见对照组。

## 参考

1. Li et al. Causal world modeling for robot control (LingBot-VA). arXiv:2601.21998, 2026.
2. Ye et al. World action models are zero-shot policies. arXiv:2602.15922, 2026.
3. Team Wan et al. Wan: Open and advanced large-scale video generative models. arXiv:2503.20314, 2025.
4. Sun et al. Learning to (learn at test time): RNNs with expressive hidden states. arXiv:2407.04620, 2024.
5. Black et al. $\pi_0$: A vision-language-action flow model for general robot control. arXiv:2410.24164, 2024.
