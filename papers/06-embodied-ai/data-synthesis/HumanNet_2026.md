# HumanNet：将以人为中心的视频学习扩展到百万小时

> **论文**：*HumanNet: Scaling Human-centric Video Learning to One Million Hours*
>
> **作者**：Yufan Deng、Daquan Zhou（通讯作者 Daquan Zhou）
>
> **机构**：DAGroup、SimpleSilicon Innovation Team、Peking University
>
> **发布时间**：2026 年 05 月（arXiv 2605.06747）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.06747) | [PDF](https://arxiv.org/pdf/2605.06747)
>
> **分类标签**：`人类视频语料` `egocentric/exocentric 双视角` `VLA 预训练` `数据 curation pipeline` `human-to-robot transfer`

---

## 一句话总结

作者构建了一个号称百万小时（Figure 1 头条统计写 967K 小时、150K+ 物体、720K+ 任务）、同时覆盖第一人称（egocentric）与第三人称（exocentric）的**以人为中心视频语料 HumanNet**,并把"人类中心过滤、时序切分、视角多样性、标注富化"当作一等公民的 curation pipeline;在固定架构与固定下游数据的**受控 VLA 后训练 ablation**下,用其 1000 小时第一人称视频初始化的 Qwen VLM,在验证 loss 上追平乃至在若干任务组略超用 100 小时真机（Magic CoBot）数据初始化的模型,并显著逼近 20000 小时真机初始化的 LingBot 基线,借此论证"大规模第一人称人类视频是可扩展、低成本的机器人数据替代品"。

## 一、问题与动机

具身学习长期受数据瓶颈制约:与语言/视觉-语言基座靠海量网络文本、图像、多模态数据持续 scaling 不同,物理交互模型通常只在小几个数量级、任务狭窄、且绑定特定机器人平台/控制接口/传感栈的数据上训练。作者认为这种规模错配是通用具身智能最明显的瓶颈之一。

人类活动与教学视频是天然的替代来源:人在家庭、工作场所、商店、厨房、仓库、公共与户外场景中自然完成操作、工具使用、移动、导航、多步流程等丰富行为;第一人称视频保留了动作执行的视角,暴露接触动力学、手-物关系与时序意图;第三人称视频补充全身运动、姿态、交互上下文与场景级动态。已有 Ego4D、EPIC-KITCHENS、Ego-Exo4D、HOI4D 等资源,以及 EgoScale、EgoVerse、EgoMimic、Being-H0 系列等把人类轨迹接入机器人学习的工作,但作者指出现有语料在**时长、跨采集碎片化、或为窄下游任务优化**上仍受限。

据此论文主张一个 data-centric 的答案:**激进地扩大以人为中心视频规模,同时把 curation、视角多样性、标注 taxonomy 当作核心科学贡献而非记账工作**,把非结构化互联网视频转成可直接用于表示学习、动作理解、运动生成与 human-to-robot transfer 的预训练基础设施。

## 二、核心方法

### 2.1 什么样的人类视频适合具身学习(四原则)

作者把 human-centric video 定义为"人类活动作为片段组织信号"的视频,须包含物理上有意义的行为(操作物体、用工具、穿行任务相关空间、装配/拆解、操作器具或界面、搬运、多人协调、执行有可见状态变化的多步流程),并刻意排除大量被动/弱 grounding、人体运动只是附带、活动时序不连贯的视频。数据集围绕四条原则设计:

- **Scale**:规模要大到能支撑活动、环境、身体运动、交互风格上的长尾覆盖,而非在窄任务族上饱和。
- **Viewpoint diversity**:第一人称与第三人称都保留并显式索引,让模型学到互补的以执行者为中心 / 以观察者为中心线索。
- **Physical relevance**:保留对具身学习有用的信息——手-物邻近、全身运动、状态变化、动作顺序、流程结构、场景上下文。
- **Pretraining readiness**:组织成能对接现代大规模训练流水线的形态——分块、元数据索引、质量过滤、caption、运动标注、可选文本/结构标签对齐。

### 2.2 三阶段 Data Pipeline

Figure 3 把端到端构建拆成三段,刻意让"源获取 / 片段级清洗 / 监督生成"解耦,各阶段可独立审计、扩展、重跑:

1. **Data collection**:关键词发现(种子关键词 → 关键词扩展 → 基于关键词的爬取与清洗 → 已有源整合 → 频道爬取)驱动内容检索,候选来自视频平台搜索、通用搜索引擎、开源数据集,以及**真实环境下的自采集**(受控第一/第三人称录制,补充公共平台难以可靠获取的欠表示活动)。频道级与源级过滤剔除跑题、低质、被动观察类内容,并去除明显重复源。
2. **Data processing**:去重与归一化(统一帧率/分辨率/容器)→ 内容过滤(保留有意义人类动作与可观测运动)→ 质量过滤(丢弃严重运动模糊、重遮挡、静止取景等)→ 按视觉变化做 scene split → 视频切片(固定粒度片段),把异构录制替换成边界清晰、适合标注的片段群。
3. **Annotation**:3D 手部/身体位姿检测恢复细粒度运动结构;对满足稳定性与视差要求的第一人称片段用**单目 SLAM**估计相机轨迹;**retargeting 模块**把恢复的人类运动对齐到统一 humanoid 骨架;LLM 辅助生成 video caption、motion 描述与活动分类(三级层次标签 L1/L2/L3)。

**Robot-ready 门控准则**——只有 retarget 误差足够小且有效帧覆盖足够高的片段才被标为"机器人可用子集":

$$\text{robot\_ready}(c)=\mathbb{1}\!\left[\,e_{\text{retarget}}(c)<15\,\text{mm}\ \wedge\ \rho_{\text{valid}}(c)>60\%\,\right]$$

> 用大白话说:把人的动作"贴"到机器人骨架上时,如果贴合误差小于 15 毫米、且一段视频里有超过 60% 的帧都贴得住,这段才算是能拿去训机器人的"干净料";其余的仍留作表示学习/运动建模用。

### 2.3 受控验证的形式化

第 3.5 节的核心不是提出新算法,而是一个**控制变量的初始化对比**:固定策略架构 $\pi_\theta$(统一采用 LingBot-VLA)与下游数据 $\mathcal{D}_{\text{ds}}$(100 任务 × 20 episode = 34 小时机器人交互),只改预训练来源 $S$:

$$\theta^\star(S)=\arg\min_\theta \ \mathcal{L}_{\text{ds}}\big(\theta;\ \theta_0=\text{Pretrain}(S)\big),\quad S\in\{\text{Qwen},\ \text{CoBot-100h},\ \text{Ego-1000h},\ \text{LingBot-20000h}\}$$

然后在五个 held-out 任务组上比较验证损失 $\mathcal{L}_{\text{val}}(\theta^\star(S))$。四种配置里,LingBot 直接用其预训练 VLM + action expert;其余三种用对应微调后的 VLM 搭配**重新初始化的 action expert**。

> 用大白话说:让四个"不同出身"的模型走完全相同的下游微调流程,谁在没见过的任务上验证损失更低,就说明它的"出身"(预训练来源)更值钱。唯一变的就是预训练那一步喂的是通用网络数据、100 小时真机、1000 小时人类第一人称视频、还是 20000 小时真机。

## 三、实验结果

### 3.1 与代表性语料的规模对比(Table 1 摘录)

| 数据集 | 规模 | 视角 | 活动范围 | 具身可用性 |
|---|---|---|---|---|
| EPIC-KITCHENS-100 | ~100h | 第一人称 | 厨房动作 | Limited |
| Ego4D | ~3,670h | 第一人称 | 日常活动 | Indirect |
| EgoScale | 20,854h | 第一人称 | 灵巧操作 | Direct |
| EgoVerse | 1,362h / 80k episodes | 第一人称 | 人类演示 | Direct |
| Ego-Exo4D | 1,286h | 第一+第三 | 技能活动 | Indirect |
| Human2Robot (H&R) | 2,600 episodes | 第三人称 | 人类演示学机器人动作 | Direct |
| **HumanNet(本文)** | **1,000,000h** | **第一+第三** | **细粒度人类活动** | **Direct** |

头条统计(Figure 1):967K 小时、150K+ 物体、720K+ 任务。注:标题/摘要/Table 1 写"百万小时 / 1,000,000h",而 Figure 1 面板给出 967K 小时,两处数字不完全一致。

### 3.2 语料结构统计(Figure 5)

pose-score 分布在质量过滤后集中于高置信端(适合密集位姿/手/运动监督);motion-score(P99 ≤ 4.18)与 motion-length(P99 ≤ 48.88)均为重尾,以短、聚焦的交互单元为主,同时保留少量更长、更剧烈的片段。按 Level-1 类别的平均运动长度与得分:

| Level-1 类别 | pose_score | motion | mean motion_len(相对) |
|---|---|---|---|
| Sports & Athletics | 0.715 | 3.07 | 最长档 |
| Daily Activities | 0.524 | 4.72 | 中等 |
| Fitness & Outdoor Activities | 0.604 | 2.41 | 较长 |
| Locomotion | 0.616 | 1.89 | 较短 |
| Social Interactions & Leisure | 0.541 | 0.81 | 短 |
| Game Character Actions | 0.509 | 1.43 | 最短档 |

### 3.3 第一人称数据的下游验证(Figure 6,五个 held-out 任务组的验证 loss)

四种初始化在 In-Domain / OOD-Average / OOD-Short-Horizon / OOD-Long-Horizon / OOD-Mobile-Manipulation 上的**验证损失**趋势(loss 越低越好):

| 初始化配置 | 预训练来源 | 是否见过真机 | 验证 loss 定性排序 |
|---|---|---|---|
| lingbot | 20000h 真机(Qwen backbone) | 是 | 最低(最好),快速收敛并早早 plateau |
| ego1000h | 1000h 第一人称人类视频(HumanNet) | 否 | 追平乃至若干任务组略优于 cobot100h,显著逼近 lingbot |
| cobot100h | 100h 真机(Magic CoBot) | 是 | 与 ego1000h 相当 |
| qwen | 通用网络视觉-语言 | 否 | 最高(最差) |

两点结论:(1)第一人称预训练一致地缩小了"通用网络级 VL 初始化"与"机器人专用初始化"之间的差距;(2)尽管 ego1000h 在预训练中**从未见过真实机器人**,它仍追平并在数个任务组略超 cobot100h,支持"第一人称人类视频是比遥操真机数据更可扩展、更低成本的替代"这一中心论点。需强调:该验证**只报告验证损失,未报告任务成功率**,且 ego1000h 并未完全追上 20000h 真机的 LingBot,只是"substantially closes the gap"。

## 四、局限性

作者自陈五点,补充笔者观察:

1. **人类行为不是机器人行为**:即便百万小时,人手/身/工具/移动/控制空间与机器人的 embodiment gap 依然存在,数据价值在于表示学习与可迁移先验,而非一对一替代真机数据。
2. **规模引入噪声**:开放世界视频不可避免地带来模糊标签、不一致任务边界、缺失元数据、视角不均衡与质量参差;caption、位姿估计、运动标注在扩大覆盖的同时引入自身误差,故需透明报告标注置信度与子集质量。
3. **覆盖仍不均衡**:即使很大,仍可能偏向特定地理、社会经济、职业、视角、体型、家庭routine 或公共活动;百万小时可能制造"普适性错觉"而掩盖盲区。
4. **隐私与安全**:第一人称可能拍到旁人、私人室内、文档、屏幕、专有工作流,第三人称可能含可识别个人;公开发布须含 license 审查、脱敏、受限内容过滤与访问控制。
5. **dual-use**:同一数据既可加速辅助系统与机器人操作,也可能强化监控类感知或让模型继承源材料的社会/地理偏见。

笔者补充的方法论局限:(a)论文明显偏"预印本 / 愿景公告"形态——Figure 4 caption 写"the final figure will show…",多处图称"illustrative",头条统计(967K)与标题(百万)不一致,且未给出 ego/exo、自采/爬取、各源占比等实际组成拆分;(b)下游验证只有**单一架构(LingBot-VLA)、单一机器人平台、单一 1000h 子集、且仅以验证 loss 为代理指标**,缺任务成功率、缺与其它人类视频预训练方法(如 EgoScale/Being-H0)的直接对照;(c)仅 2 位作者署名对"百万小时全量执行 + 全套 3D 位姿/SLAM/retarget/LLM 标注"的可复现性构成疑问,数据集在 v1 阶段尚未见可下载证据。

## 五、评价与展望

**优点**:选题切中具身学习最实的痛点——数据规模错配;把 curation、视角多样性、标注 taxonomy 上升为一等设计原则,而非单纯堆时长,方向正确。相比 EgoScale(20k 小时纯第一人称)、EgoVerse(1362 小时)、Ego-Exo4D(1286 小时),本文在"规模 + ego/exo 双视角 + robot-ready 门控子集"上给出了更激进的定位;robot-ready 的量化门控(retarget<15mm、有效帧>60%)是一个务实、可操作的过滤准则,值得同类人类视频→机器人流水线借鉴。受控初始化 ablation 的设计(固定架构与下游数据、只换预训练源)在概念上是干净的因果对照。

**不足与开放问题**:整体成熟度偏低,更像"规模宣言 + 初步信号"而非已交付的数据集论文。最关键的方法论软肋是**用验证 loss 而非策略成功率**衡量下游价值——loss 更低并不必然等于操作成功率更高,尤其对长程/移动操作;"1000h 人类视频 ≈ 100h 真机"这一亮眼结论因此证据偏弱。其次,与紧邻工作(Being-H0/H0.5/H0.7 的 latent world-action 建模、GR00T N1 的异构混训、EgoMimic 的 ego 轨迹-真机对齐)缺少正面比较,难以判断 HumanNet 的边际增益究竟来自"更大规模"还是"更好 curation"。数据侧的组成透明度、真实可下载性、以及爬取来源的 license/隐私合规,都是发布前必须补齐的硬约束。

**可能的改进方向**:(1)把评测从验证 loss 升级为多平台真机/仿真的任务成功率,并做数据规模-性能的 scaling curve(如 100h→1000h→10000h 人类视频),验证是否真有可预测收益;(2)在同一下游协议下与 EgoScale/Being-H0 等做 apples-to-apples 对照,拆分"规模 vs. 标注质量 vs. 视角多样性"的贡献;(3)公开 ego/exo、source、场景、robot-ready 子集的真实占比与质量分布;(4)把 exocentric→retarget→机器人这条"第三人称桥"也纳入下游验证(目前验证只用了 egocentric 子集),否则双视角卖点在实验上未被兑现。

## 参考

1. Luo et al. *Being-H0: Vision-language-action pretraining from large-scale human videos* (arXiv 2507.15597, 2025) — 从大规模人类视频做 VLA 预训练,本文思想上的近邻。
2. Zheng et al. *EgoScale: Scaling dexterous manipulation with diverse egocentric human data* (arXiv 2602.16710, 2026) — 20854h 第一人称灵巧操作数据,规模上最直接的对照。
3. Punamiya et al. *EgoVerse: An egocentric human dataset for robot learning from around the world* (arXiv 2604.07607, 2026) — 共享式持续增长的第一人称机器人学习生态。
4. NVIDIA et al. *GR00T N1: An open foundation model for generalist humanoid robots* (arXiv 2503.14734, 2025) — 异构机器人日志与人类视频混训的开源 VLA。
5. Grauman et al. *Ego-Exo4D: Understanding skilled human activity from first- and third-person perspectives* (arXiv 2311.18259, 2024) — 第一+第三人称技能活动数据集,双视角思路来源。
