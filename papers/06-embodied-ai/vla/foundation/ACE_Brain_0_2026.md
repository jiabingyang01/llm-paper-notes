# ACE-Brain-0：以空间智能作为通用具身的共享脚手架

> **论文**：*ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments*
>
> **作者**：Ziyang Gong, Zehang Luo, Anke Tang, Zhe Liu, Shi Fu, Zhi Hou（项目负责人）, Xue Yang, Dacheng Tao, Xiaogang Wang（通讯作者）et al.
>
> **机构**：ACE Robotics、上海交通大学、南洋理工大学、香港中文大学、香港大学、中国科学技术大学、复旦大学、厦门大学、华东师范大学、武汉大学、中山大学
>
> **发布时间**：2026 年 03 月（arXiv 2603.03198）
>
> **发表状态**：未录用（预印本，技术报告）
>
> 🔗 [arXiv](https://arxiv.org/abs/2603.03198) | [PDF](https://arxiv.org/pdf/2603.03198)
>
> **分类标签**：`具身基础大脑` `空间智能` `跨具身学习` `数据无关模型融合` `GRPO强化学习` `自动驾驶+无人机+机器人操作统一MLLM`

---

## 一句话总结

ACE-Brain-0 主张**空间智能是跨具身（自动驾驶、无人机低空感知、机器人操作）迁移的通用脚手架**,提出 Scaffold-Specialize-Reconcile（SSR）三阶段范式——先训空间专家打地基,再各领域独立微调避免梯度冲突,最后用数据无关的任务向量融合（WUDI）把专家合并回一个模型,外加 Embodied SFT 与 GRPO 强化学习收尾——训出的 8B 统一 MLLM 在 24 个空间/驾驶/低空/具身基准上多项达到或超过 SOTA（SAT 92.0%、MindCube 82.1%、NuPlanQA 91.7%、AircopBench 70.3%）,消融证实以空间专家为初始化可让 AD/UAV/Embodied 专家分别再提升 +25.6/+16.5/+5.4 个点,且 SSR 训练同时避免了联合训练的梯度干扰与序贯训练的灾难性遗忘。

## 一、问题与动机

通用具身智能要求模型在自动驾驶、机器人操作、无人机等形态迥异的具身体上都保持鲁棒泛化,但现有两条路线各有硬伤:联合训练在异构领域数据上混合优化,长尾分布与梯度干扰会稀释各领域专精度;序贯的领域专属微调虽能在目标域上做深,却不可避免地对此前学到的能力产生灾难性遗忘。作者认为症结不在数据多样性或模型容量,而在于**缺少一种有原则的机制来组织、整合并保存跨具身的物理知识**。

论文的关键洞察是:尽管自动驾驶、机器人交互、低空感知在形态与动作空间上差异巨大,它们都依赖对 3D 空间关系的理解——物体布局、几何关系、动作的空间后果预测——这一共性使空间建模成为一个天然的、与具体形态无关的基础,可以作为跨异构物理域的迁移桥梁。进一步地,空间基础天然锚定了一条"由粗到细"的认知进阶链:自动驾驶与低空感知偏向空间感知型规划（类 VLN 的轨迹规划/行为决策）,而具身交互需要更细粒度的执行（类 VLA 的底层运动控制与精确物体操作）。

## 二、核心方法

**任务形式化。** 将领域集合记为 $\mathcal{M}=\{m_{\text{general}}, m_{\text{embodied}}, m_{\text{spatial}}, m_{\text{driving}}, m_{\text{aerial}}\}$,每个领域诱导一个任务分布 $\mathcal{D}_{m_k}$,采样三元组 $(o,c,y)$（多模态观测、任务条件、目标输出）。所有任务统一建模为条件自回归形式 $p_\theta(y\mid o,c)$,由单一共享 MLLM 参数 $\theta$ 承载。

**架构。** 沿用 Qwen3-VL 式设计:视觉编码器 + MLP Projector 将单视图图像/多视图图像/视频统一映射为视觉 token,并按 General/Spatial/Driving/Aerial/Embodied 五类概念上组织;文本指令经 Tokenizer 转为文本 token;二者拼接后送入 ACE-Brain LLM Decoder 自回归生成。训练目标为标准从左到右自回归损失,仅在文本 token 上计算,并采用平方平均（square averaging）的 token 权重以平衡不同长度样本的梯度贡献。

**Scaffold-Specialize-Reconcile（SSR）五阶段训练（Table 1）：**

- **Stage 1 空间脚手架训练**：先用通用数据（Cambrian-737K 等）对 Qwen3-VL 做指令微调得到 $\theta_{\text{base}}$,再用大规模空间数据训出空间专家 $\theta_{\text{spatial}}$,作为后续所有专家训练的共同起点。
- **Stage 2 领域专家隔离微调**：从 $\theta_{\text{spatial}}$ 出发,分别独立训练 $\theta_{\text{ad}}$（自动驾驶感知/规划/控制）与 $\theta_{\text{uav}}$（低空感知/导航）,隔离训练避免不同领域冲突梯度相互干扰。
- **Stage 3 跨具身融合（Reconcile）**：以数据无关方式合并专家。定义任务向量 $\tau_m := \theta_m-\theta_{\text{base}}$,融合参数通过求解

$$\theta^*_{\text{merge},l} = \theta_l + \arg\min_{\tau_{\text{merge},l}} \sum_{i=1}^K \mathbb{E}_{x_{i,l}\sim\mathcal{D}_{m_i,l}} \left\| (\tau_{i,l}-\tau_{\text{merge},l}) x_{i,l} \right\|_2^2$$

用大白话说:把每个专家相对基座模型"走过的方向"（任务向量）看作一条条独立的更新路径,融合就是找一个折中方向,使它在所有专家各自的数据分布上产生的"行为差异"最小,而不需要重新拿数据训练。论文采用 WUDI（源自 Shen et al. 提出的加权集成 MoE 融合框架）为主方法,并与朴素平均（ModelSoups）、TSVM（SVD 任务奇异向量融合）对比;优化用 Adam、lr=1e-5、weight decay=0、迭代 1,000 步,基于 FusionBench 框架实现。
- **Stage 4 具身数据 SFT**：融合模型 $\theta_{\text{merge}}$ 在大规模具身与第一人称多模态数据上继续微调,得到 $\theta_{\text{embodied}}$,强化任务规划与动作预测能力同时保留空间与跨域能力。
- **Stage 5 GRPO 强化学习**：以 spatial/ad/uav/embodied 混合语料（约 100k）对 $\theta_{\text{embodied}}$ 做 Group Relative Policy Optimization,组内相对奖励归一化得到优势 $\hat{A}_{i,t}=(r_i-\text{mean}(\mathbf{r}))/\text{std}(\mathbf{r})$,不含参考策略的 KL 惩罚项（实验发现裁剪代理目标本身已足够稳定）。

**理论支撑（附录）。** 论文给出两条形式化论证:(1) 单步干扰界（Theorem 1）证明联合更新下某一形态的风险变化会被其余形态梯度的负内积项拖累,量化了梯度干扰的来源,为隔离训练提供理论依据；(2) 脚手架到形态的迁移界（Theorem 2）$R_m(\theta_{\text{spatial}}) \le R_{\text{sp}}(\theta_{\text{spatial}}) + C_m\delta_m + 2L_g\varepsilon_g + \varepsilon_m$,把目标域风险分解为脚手架自身风险、几何分布偏移项 $\delta_m$、几何可恢复性误差 $\varepsilon_g$ 与形态残差项,论证空间脚手架作为"通用桥梁"的有效性边界。

## 三、关键结果

训练语料约 1.58B token,覆盖通用多模态指令（Cambrian-737K 等）、空间智能（VSI-590K、SAT、VICA-322K、GPT4Scene、Scene-30K、VLM-3R、MindCube、SpaceR-151K 等）、自动驾驶（MAPLM、DriveAction、NuScenes-QA、NuPlanQA、LingoQA）、低空（HRVQA、AirSpatial-VQA、AirCopBench、AVI-Math、CapERA）、具身与第一人称（MuEP、OWMM-VLM、Eb-Alfred、Eb-Habitat、RoboVQA、EgoPlan、EgoCOT）数据,UAV 数据在语料中占比相对最小（长尾）。ACE-Brain-0-8B 在 24 个基准上的代表性结果：

| 领域 | 基准 | ACE-Brain-0-8B | 对比基线 |
|---|---|---|---|
| 空间智能 | SAT | 92.0% | Gemini2.5-Pro 79.3% / MiMo-Embodied-7B 78.7% |
| 空间智能 | MindCube | 82.1% | Gemini2.5-Pro 57.6% / GPT-4o 46.1% |
| 空间智能 | VSI | 63.3% | Gemini2.5-Pro 47.8% / Vlaser-8B 60.3%（最强具身脑） |
| 空间智能 | Multi3DRef | 59.6% | VeBrain-7B 67.8%（仍领先） |
| 自动驾驶 | MME-RealWorld | 71.2% | Gemini2.5-Pro 67.0% |
| 自动驾驶 | MAPLM | 77.8% | MiMo-Embodied-7B 74.5% |
| 自动驾驶 | NuPlanQA | 91.7% | Pelican-VL-7B 83.4% |
| 低空 | UrbanVideo-Bench | 56.9% | Qwen-VL-Max 45.5% |
| 低空 | AircopBench | 70.3% | GPT-4o 51.8% |
| 低空 | Airspatial-VQA（MAE↓） | 258.0 | GPT-4o 192.4（更优） |
| 具身 | RoboVQA | 64.6% | Qwen2.5-VL-7B-Inst 57.2% |
| 具身 | EmbSpatial-Bench | 77.3% | Gemini2.5-Pro 78.7%（接近但未超） |
| 具身 | EgoPlan-Bench2 | 55.3%（同类最佳） | Qwen3-VL-8B-Inst 53.5% |

**消融（Table 6-8）关键结论：** 以空间专家 $\theta_{\text{spatial}}$ 为初始化训练领域专家,相对直接从基座模型（Qwen3-VL-8B-Instruct,AD/UAV/Embodied 基线均分别为 47.0/37.8/52.7）训练带来大幅额外增益：AD +25.6pp（72.6）、UAV +16.5pp（54.3）、Embodied +5.4pp（58.1）；而直接从基座模型训练的 Embodied 专家反而比基座**下降 1.9pp**,作者将其归因于具身操作所需的细粒度动作理解难以从通用域直接迁移,须先经过空间脚手架中转。融合策略上 WUDI 全面领先朴素平均与 TSVM,在三领域均超过单一最强专家（如 Spatial 76.7% 超过单专家 72.5%,体现"超加性"组合效应）。训练范式对比中,联合训练（Joint）与序贯训练（Sequential）均在部分领域出现明显掉点（Sequential 在 Spatial/AD 上分别 -4.9/-2.5pp）,唯有 SSR（+ GRPO 后）在四个领域同步取得正向或近零变化（Spatial +6.6、AD -0.5、UAV -0.2、Embodied +1.9）。

## 四、评价与展望

**贡献与优点。** 论文把"空间智能作为跨具身共享结构先验"这一直觉，落实为一套可执行、可复现的三阶段训练配方，并额外提供了形式化的梯度干扰上界与脚手架迁移界作为理论支撑，这在同类具身基础大脑技术报告中并不常见。数据无关模型融合（任务向量优化 + WUDI）被系统性地引入物理智能这一较少被探索的领域，24 个基准的横向对比（覆盖 RoboBrain2.0/2.5、VeBrain、Pelican-VL、MiMo-Embodied、Vlaser 等同期具身脑）也提供了较全面的参照系。

**局限与开放问题。** 其一，ACE-Brain-0 目前统一的仍是判别式/生成式问答与规划层输出（Q\&A、轨迹描述、动作选择题等），并未真正扩展到闭环连续控制的 VLA 策略——这一点作者在结论中明确列为未来方向之一，意味着"空间脚手架能否提升低层连续动作策略"仍待验证。其二，脚手架带来的增益并不均匀：具身操作域的提升（+5.4pp）明显弱于自动驾驶（+25.6pp）与无人机（+16.5pp），论文自陈是因为精细动作理解与粗粒度空间规划之间存在迁移鸿沟，这也是该框架当前最大的短板所在。其三，SSR 流程本身较重——需要先训练多个 8B 规模的领域专家再做融合、SFT、RL 共五个阶段，论文未给出与端到端联合训练的算力/时间成本对比，实际可扩展性（例如扩展到更多具身形态，如足式机器人、水下航行器，正如结论所展望）有待检验。其四，个别基准上 ACE-Brain-0 并非最优（如 Multi3DRef 落后 VeBrain-7B、Airspatial-VQA 的 MAE 落后 GPT-4o、EmbSpatial-Bench 略低于 Gemini2.5-Pro），说明其"通用但非处处最强"的定位仍需权衡。

**与已有工作的关系。** 该框架延续了 RoboBrain 系列、VeBrain、MiMo-Embodied、Pelican-VL 等"具身基础脑"路线朝多任务、多领域统一迈进的趋势，区别在于用显式的隔离训练 + 数据无关融合替代了单纯的联合或序贯微调，其融合算法直接承接自持续学习/多任务模型合并文献（AdaMerging、WUDI 等），为该文献向物理智能场景的迁移提供了一个具体实例。后续工作值得关注的方向包括：将空间脚手架的收益进一步延伸至连续动作/VLA 层，以及探索 SSR 范式在专家数量、领域数量持续增长下的可扩展性与终身学习特性。

## 参考

- RoboBrain 2.0 / RoboBrain 2.5（BAAI RoboBrain Team）——同类具身基础脑技术报告，Table 2-5 中的主要对比基线。
- VeBrain：Visual Embodied Brain——本文在 Multi3DRef 上唯一未被超越的具身脑基线。
- MiMo-Embodied：X-Embodied Foundation Model——AD/UAV/Embodied 多域基线之一。
- Shen et al. 提出的 WUDI 加权集成 MoE 模型融合方法——本文 Stage 3 采用的核心数据无关融合算法。
- AdaMerging——首个为多任务模型融合引入自适应系数学习的工作，为本文 Reconcile 阶段的理论脉络起点。
