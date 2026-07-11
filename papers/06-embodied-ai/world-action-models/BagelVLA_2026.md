# BagelVLA：通过交错式视觉-语言-动作生成增强长程操作

> **论文**：*BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation*
>
> **作者**：Yucheng Hu, Jianke Zhang, Yuanfei Luo, Yanjiang Guo, Xiaoyu Chen, Xinshu Sun, Kun Feng, Qingzhou Lu, Sheng Chen, Yangang Zhang, Wei Li, Jianyu Chen et al.
>
> **机构**：Tsinghua University、ByteDance Seed
>
> **发布时间**：2026 年 02 月（arXiv 2602.09849）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.09849) | [PDF](https://arxiv.org/pdf/2602.09849)
>
> **分类标签**：`VLA` `交错规划` `视觉预测` `Flow Matching` `长程操作`

---

## 一句话总结

BagelVLA 用 Mixture-of-Transformers 把"文本子任务规划 → 关键帧视觉预测 → 动作生成"三件事显式串成因果链,并提出 Residual Flow Guidance（RFG，用当前观测而非纯高斯噪声去初始化关键帧噪声）把双路 Flow Matching 的推理延迟从 6.04s 压到 1.23s 且性能不降反升,在 Calvin ABC-D 上平均完成长度达 4.405（对比 π0 的 3.648）,在 RoboTwin2.0 上 Clean/Randomized 成功率 75.26%/20.87%,在真机长程规划任务上成功率达 73.3%/63.3%,大幅领先 π0、VPP 等基线。

## 一、问题与动机

现有 VLA 模型通常只在"语言规划"和"视觉预测"两条辅助能力之间二选一：RT-2、OpenVLA 等直接把语言指令映射到动作,擅长语义泛化但缺乏对物理动态的建模；VPP、Cosmos Policy 等用视频/图像生成模型预测未来视觉状态作为条件,擅长动态建模但因缺乏专用 VLM backbone 而在需要复杂推理的任务上指令跟随能力弱。论文认为,对于像"按红→黄→蓝→绿顺序摞积木"这类长程任务,一个把全局指令 $L$ 直接映射到动作的黑盒策略 $p_\theta(a_t\mid v_t, L)$ 是不够的,模型需要显式推理任务的因果链：先搞清楚当前该做哪个子步骤,再预见做完这一步后世界会变成什么样,最后才生成与这两者对齐的动作。

同时,近期统一理解-生成模型（如 Bagel、Chameleon、Show-o）已经展现出用单一 Transformer 联合处理文本/图像理解与生成的涌现能力,但这类通用模型并非为具身连续控制设计,直接用于机器人时其多模态推理能力往往被稀释成"只保留子集能力",没有真正做到分步骤的多模态思维链（CoT）。BagelVLA 的目标就是把这种统一多模态推理能力显式注入到长程操作场景,同时解决"融合视觉生成会带来高推理延迟"这个实际部署问题。

## 二、核心方法

### 2.1 交错规划（Interleaved Planning）的因果分解

给定全局指令 $L$ 和当前观测 $v_t$,BagelVLA 将当前子任务 $l_t$、未来关键帧 $v_{t+k}$、动作 $a_t$ 的联合分布按操作任务的逻辑依赖关系做因式分解：

$$
\mathcal{J} = -(\mathcal{L}_l + \mathcal{L}_v + \mathcal{L}_a) = \max_\theta \mathbb{E}_{\mathcal{D}} \log\; p_\theta(l_t\mid v_t, L)\cdot p_\theta(v_{t+k}\mid v_t, L, l_t)\cdot p_\theta(a_t\mid v_t, L, l_t, v_{t+k})
$$

**用大白话说**：把"看指令做动作"拆成三步顺序推理——先想清楚现在该干哪个子任务（语言规划）,再在脑子里"想象"做完这步之后画面会变成什么样（视觉预见）,最后才照着这个子任务和想象出的画面去生成具体动作（动作生成）。三步的损失（分别是文本交叉熵 $\mathcal{L}_l$、关键帧 Flow Matching 损失 $\mathcal{L}_v$、动作 Flow Matching 损失 $\mathcal{L}_a$）联合优化,而不是像传统 VLA 那样一步到位。

### 2.2 Mixture-of-Transformers 架构

模型由三个通过自注意力连接的独立 Transformer "专家"组成（如 Fig.2）：

- **理解专家（Understanding Expert）**：7B,架构沿用 Qwen2.5-LLM-7B,输入用 SigLIP2 编码为 ViT 特征,自回归输出文本子任务 $l_t$,损失为交叉熵 $\mathcal{L}_l = -\log p_\theta(l_t\mid v_t, L)$。
- **生成专家（Generation Expert）**：7B,同样基于 Qwen2.5-LLM-7B 架构但输出图像,输入用 FLUX 的 VAE 编码,通过 Flow Matching 迭代去噪生成关键帧,损失 $\mathcal{L}_v = -\log p_\theta(v_{t+k}\mid v_t, L, l_t)$。理解、生成专家均从统一多模态预训练模型 Bagel 初始化。
- **动作专家（Action Expert）**：结构与 Qwen2.5 LLM 相同但把 MLP 中间层维度压缩到原来的 1/5,只有 2B 参数,负责处理本体感知（proprioception）和动作模态,用 Flow Matching 建模动作 chunk：$\mathcal{L}_a = -\log p_\theta(a_t\mid v_t, L, l_t, v_{t+k})$。动作去噪时可以关注 VAE/ViT 特征、全局指令、生成的子任务,以及生成专家关键帧去噪过程中的中间隐状态。

三个专家通过共享的多模态自注意力（Multi-modal Self Attention）交互,但各自维护独立的 FFN 和 KV,是一种非对称的双 Flow-Matching 耦合结构。

### 2.3 双 Flow-Matching 的三种耦合方案

论文的关键设计问题是：动作专家该如何"看"生成专家正在去噪的关键帧？给出三种方案（Fig.3）：

1. **Complete Denoise（完整去噪）**：生成专家先完整跑完 $N_1$ 步去噪得到关键帧,再拼接进上下文让动作专家跑 $N_2$ 步去噪。等价于"世界模型（WM）+ 逆动力学模型（IDM）"的串联,理论上信息最完整,但总延迟 $N_1+N_2$ 步,且测试时若图像去噪中间态落入分布外（OOD）,会连带拖累动作质量。
2. **Joint Denoise（联合去噪）**：关键帧和动作同步去噪 $N$ 步,动作专家在每一步都能看到当前噪声水平下的关键帧,把延迟降到 $N$ 步。
3. **Single-step Denoise（单步去噪）**：动作生成只依赖关键帧去噪第一步（$\tau=0$，即纯噪声输入）算出的 KV-cache,相当于动作只借用生成专家"想开始生成画面"那一刻的中间特征,而不必等图像真正生成完,把延迟降到 1 步。

三种方案的损失形式统一为 $\mathcal{L}_{a} = \mathbb{E}\big[\Vert \mathbf{v}_{a,\theta}(\cdot)-(a_t^1-a_t^0)\Vert_2^2\big]$,区别只在于生成专家条件项里关键帧的去噪时间步 $\tau$ 取值不同。

### 2.4 Residual Flow Guidance（RFG）——本文核心创新

单步去噪方案虽然快,但因为初始噪声图像 $v_{t+k}^{\tau=0}$ 是纯高斯噪声,动作专家提取到的"预测视觉特征"信息量很弱。论文提出 RFG,把关键帧去噪的噪声初始化从纯高斯改成以当前观测 $v_t$ 为均值的高斯：

$$
\text{Naive Single-step Denoise}:\ v_{t+k}^{\tau=0}\sim\mathcal{N}(0, I) \qquad\qquad \text{RFG}:\ v_{t+k}^{\tau=0}\sim\mathcal{N}(v_t, I)
$$

**用大白话说**：与其让生成专家从一张纯雪花噪声图开始"凭空想象"未来会发生什么,不如直接把当前这一帧画面当作起点、只叠加噪声去建模"从现在到未来会发生哪些变化"（即残差 residual）。这样模型只需要花精力建模抓取物体、移动机械臂等动态变化区域,而不必浪费容量去重建桌面、背景这些几乎不变的静态像素,因此仅需极少去噪步数（如 10 步）就能得到高质量、语义正确的关键帧预测,同时动作专家拿到的 KV-cache 也携带了更强的"当前帧先验"，动作学习收敛更快、精度更高。

### 2.5 数据引擎与两阶段训练

数据分四类：自采集机器人数据（Agibot、Open-galaxea、RoboTwin 等平台）、开源机器人数据、通用多模态数据（VQA/图文）、第一人称人类视频。对缺少细粒度标注的公开数据集,用 Seed-1.5-VL-thinking 自动生成子任务文本 $l_t$ 及起止关键帧边界。

- **阶段 1（预训练）**：只微调理解专家 + 生成专家,学习子任务规划与关键帧预测,同时用通用 VQA 数据（论文正文报告约 298 万条 QA 对）联合训练以保留基础模型的语言能力；机器人/人手数据规模在正文与附录表述略有出入（正文报告人手数据 31 万条、开源机器人数据约 44 万条、自采真机数据 7.5 万条,附录 Table 6 给出的细分数字略有不同,但量级一致）。
- **阶段 2（微调）**：解冻全模型,同时在 Calvin（ABC split）、RoboTwin（50 任务 ×50 episode = 2500 episode）、ALOHA 基础任务（3000 episode）、ALOHA 长程任务（1500 episode）上联合学三个规划任务的动作标签。

**推理策略**：每个去噪步只激活一个专家（文本/关键帧用 7B,动作用 2B）,单步去噪方案进一步提升执行频率——把当前帧、指令上下文和一张纯噪声图拼接算出理解 + 生成专家的 KV,再用它条件化动作生成。单 RTX 5090 上单次推理 1.2 秒/chunk（chunk size 48,约合 40Hz 实时执行频率）。此外引入异步执行（训练时随机用前一帧替换当前帧,推理时降低理解/生成专家 KV 上下文的更新频率,只更新本体感知输入来产生新动作 chunk）,可将执行频率进一步提升到 72Hz。

## 三、实验结果

### 仿真基准：Calvin ABC→D 与 RoboTwin2.0

| Model | Calvin ABC-D（Avg. Len） | RoboTwin Clean（%） | RoboTwin Randomized（%） |
|---|---|---|---|
| π0 | 3.648 | 46.42 | 16.34 |
| RDT | – | 34.50 | 13.72 |
| UP-VLA | 4.078 | 52.92 | 15.16 |
| VPP | 4.329 | – | – |
| w/o Textual-planning（消融） | – | 54.00 | 19.20 |
| w/o Keyframe-forecasting（消融） | 3.345 | 56.72 | 15.92 |
| **BagelVLA** | **4.405** | **75.26** | **20.87** |

Calvin 详细分解（连续完成 1~5 个任务的成功率,Table 8）：BagelVLA 在完成 1/2/3/4/5 个任务上的成功率分别为 99.3%/95.4%/89.3%/82.4%/74.1%,均高于 VPP（96.5%/90.9%/86.6%/82.0%/76.9%）与 UP-VLA。

### 真机基础任务（9 类技能,每类跑 20 次,Table 2）

| Model | Pick\&Place Seen | Pick\&Place Unseen | Stack Cubes | Sweep Rubbish | 平均成功率 |
|---|---|---|---|---|---|
| π0 | 95 | 55 | 65 | 55 | 65.0 |
| VPP | 85 | 45 | 50 | 45 | 59.5 |
| **BagelVLA** | 95 | **85** | **80** | **80** | **75.5** |

在未见物体（OOD）的 Pick\&Place Unseen 上 BagelVLA 领先 π0 30 个百分点（85 vs 55）,论文将此归因于从统一理解 / 生成专家继承的语义特征在 VLA 微调后仍被较好保留。

### 真机长程规划任务（Table 3，成功率 / 规划准确率,Easy/Middle/Hard 三档难度）

| Model | 摞积木-成功率 | 摞积木-规划准确率 | 算式摆放-成功率 | 算式摆放-规划准确率 |
|---|---|---|---|---|
| π0 | 40.0 | 55 | 23.3 | 30 |
| VPP | 25.0 | 45 | 23.3 | 40 |
| w/o Keyframe-forecasting | 53.3 | 60 | 50.0 | 75 |
| w/o Textual-planning | 43.3 | 70 | 33.3 | 50 |
| **BagelVLA** | **73.3** | **95** | **63.3** | **85** |

"算式摆放"任务要求模型先用 VLM 的算术推理能力算出结果（如 21+3=?）,再把对应数字积木摆到正确位置,是验证交错规划能否保留基础模型推理能力的直接证据。论文特别指出,规划准确率（约 90%）显著高于任务成功率,说明多模态规划本身基本正确,差距主要来自精细运动控制的执行误差,而非语义/规划层面的错误。

### 关键消融

**双 Flow-Matching 耦合方案对比**（Calvin 单视角,10k 步训练,A800 测延迟,Table 4）：

| 方案 | 延迟 | ABC-D（Avg. Len） |
|---|---|---|
| Complete Denoise | 6.04s | 2.480 |
| Joint Denoise | 2.90s | 2.038 |
| Single-step Denoise | 1.23s | 3.345 |
| **RFG** | **1.23s** | **3.600** |

RFG 在与朴素单步去噪相同延迟（1.23s）下把 ABC-D 完成长度从 3.345 提到 3.600,同时论文归因于其在颜色/场景域偏移下更不易进入 OOD 中间态。

**预训练与 RFG 的真机消融**（Figure 6,Pick\&Place OOD / Pour Fries / Sweep Rubbish / Stack Cubes 四任务）：去掉预训练（w/o pretrain）在 OOD 任务上相对下降约 50%；去掉 RFG（w/o RFG,退化为朴素单步去噪）在多个任务上下降 20%~25%,证实语言/视觉预训练与 RFG 机制均对最终性能有实质贡献。

## 四、局限性

1. **执行精度与规划正确性存在明显 gap**：论文自己在长程任务实验中承认,规划准确率接近 90%,但任务成功率仅 63%~73%,差距来自"模型和数据集在精细运动控制上的局限",说明交错规划主要解决的是语义/时序决策问题,并未从根本上提升底层动作执行的精度。
2. **仍需在延迟与生成质量间权衡**：Complete Denoise 效果理论上限最高但延迟高达 6s,RFG 是在"牺牲完整视觉生成保真度"换取速度的折中方案,论文没有给出在更大动作 chunk 或更高精度任务下 RFG 是否仍然够用的边界分析。
3. **关键帧而非视频级预测**：视觉预见只建模单个未来关键帧 $v_{t+k}$,而非连续视频轨迹,相较 VPP、Cosmos Policy 等基于视频扩散的方法,对中间过程动态的建模粒度更粗。
4. **依赖闭源标注模型**：公开数据集的子任务切分和关键帧标注依赖 Seed-1.5-VL-thinking（字节内部闭源模型）自动生成,数据引擎的可复现性受限于对该外部标注模型的可用性。
5. **模型体量偏大**：理解 + 生成 + 动作三专家合计约 16B 参数,即便有单步去噪和异步执行优化,仍需 RTX 5090 级别硬件才能达到 40~72Hz,对边缘/低算力部署仍有较高门槛。
6. **正文与附录数据规模表述不完全一致**（如 Stage-1 通用 VQA 对数、开源机器人 episode 数在正文 Sec.3.5 与附录 C.1 存在出入）,论文未对此做说明,略微影响数据规模数字的可信度。

## 五、评价与展望

BagelVLA 的核心贡献可以概括为两点：一是把"语言规划 → 视觉预见 → 动作生成"的因果结构显式建模为一个联合似然的因式分解,而不是像多数 VLA 工作那样把辅助视觉预测目标当作正则项松散加入；二是 RFG 这个工程上简洁但有效的设计——通过把去噪起点从纯噪声换成当前观测,同时解决了"融合视觉生成的 VLA 普遍存在高延迟"和"单步去噪信息量不足"两个问题,消融实验（Table 4、Figure 6）证明其收益是真实的而非仅仅换了个更快的路径。

与同类工作相比：相较 VPP、Cosmos Policy 等纯视觉预测驱动的策略,BagelVLA 通过继承 Bagel 的统一理解-生成骨干获得了更强的语言推理能力（体现在算式摆放任务上对算术推理的保留）；相较 UP-VLA、F1、VILLA-X 等同样引入生成专家的 MoT 类 VLA,论文强调自己是"完整实现分步骤多模态思维链"而非只保留原模型能力的子集,并在实验上验证了 textual-planning 和 keyframe-forecasting 两个组件分别带来的独立增益（Sec.4.3.4）。这种"分解-消融"式的实验设计是本文比较扎实的地方,但也应注意到所有基线（π0、RDT、UP-VLA、VPP）都是在与 BagelVLA 相同的下游动作数据上重新微调评测的,尚缺少与近期同样基于统一多模态骨干的 F1、DreamVLA、UniCode 等工作的直接横向对比,这些方法在方法论上高度相关,若能在同一 benchmark 下对比会让"交错规划"相对于"隐式生成专家"的增量收益更具说服力。

开放问题包括：(1) 关键帧级预见是否在更长时域、更多子步骤的任务上仍然足够,还是需要引入多关键帧或短视频预见；(2) RFG 依赖当前帧到未来关键帧变化幅度较小的假设,在大范围场景变化或相机大幅移动的任务中是否仍然有效尚未验证；(3) 论文的真机评测任务集中在桌面抓取/摞放类操作,场景复杂度、物体多样性相对有限,更大规模、更强干扰下的鲁棒性有待后续工作检验；(4) 规划准确率与执行成功率之间的 gap 提示,未来把交错规划与更精细的低层控制模块（如力反馈、闭环视觉伺服）结合可能是提升长程任务成功率的下一步方向。

## 参考

1. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.
2. Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*, arXiv:2410.07864, 2024.
3. Zhang et al. *UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent*, arXiv:2501.18867, 2025.
4. Hu et al. *VPP: Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations*, arXiv:2412.14803, 2024.
5. Deng et al. *Emerging Properties in Unified Multimodal Pretraining (Bagel)*, arXiv:2505.14683, 2025.
6. Chen et al. *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation*, arXiv:2506.18088, 2025.
