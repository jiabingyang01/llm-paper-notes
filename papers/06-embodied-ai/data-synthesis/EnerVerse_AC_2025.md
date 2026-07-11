# EVAC：EnerVerse-AC——用动作条件世界模型构想具身环境

> **论文**：*EnerVerse-AC: Envisioning Embodied Environments with Action Condition*
>
> **作者**：Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, Guanghui Ren et al.
>
> **机构**：AgiBot；上海交通大学（SJTU）；香港中文大学 MMLab（MMLab-CUHK）
>
> **发布时间**：2025 年 5 月（arXiv 2505.09723）
>
> **发表状态**：未录用（预印本），代码/模型/数据集开源于项目主页
>
> 🔗 [arXiv](https://arxiv.org/abs/2505.09723) | [PDF](https://arxiv.org/pdf/2505.09723)
>
> **分类标签**：`动作条件视频生成` `世界模型` `数据引擎` `策略评测器` `多视角生成`

---

## 一句话总结

EVAC 在 UNet 视频扩散基座（沿用 EnerVerse 架构）上引入多层级动作条件注入（末端位姿投影图 + Delta Action Attention + ray map 多视角编码），把人类采集的轨迹增广成更多样的合成轨迹（20 条专家示范 + 30% 合成数据使抓瓶任务成功率从 0.28 升到 0.36），同时作为无需实体机器人/仿真资产的策略评测器，其四任务成功率与真实机器人评测高度一致。

## 一、问题与动机

机器人模仿学习已从静态任务迈向动态交互任务,但测试和评估依然昂贵——要么依赖真实机器人(成本高、耗时)、要么依赖复杂 3D 仿真资产(制作劳动密集、sim-to-real 差距明显)。近期用视频生成模型充当"世界模拟器"成为一个有前景的方向,但已有的具身视频世界模型大多只根据语言指令生成视频、或仅预测动作而非依据动作生成未来观测,并不是真正意义上能对智能体动作做出响应的世界模拟器。EVAC 的目标是构建一个动作条件化的世界模型:给定初始观测和一段预测的动作序列,生成对应的、可控且高保真的未来视觉观测,使其既能作为策略学习的数据引擎,又能作为策略评测器替代真实机器人部署。

## 二、核心方法

**基座与问题形式化**。EVAC 基于 UNet 的视频扩散模型(沿用 DynamiCrafter/EnerVerse 一脉),记视频集合 $O \in \mathbb{R}^{V\times(H+K)\times 3\times h\times w}$（$V$ 视角数、$H$ 历史帧、$K$ 待预测帧),编码到潜空间 $z\in\mathbb{R}^{V\times H\times C\times h\times w}$ 后用条件扩散建模

$$z_t = p_\theta(z_{t-1}, c, t)$$

用大白话说:模型看过去若干帧图像,拿到一段动作序列作为条件 $c$,一步步去噪生成未来帧。条件信号来自机器人末端位姿轨迹 $A\in\mathbb{R}^{(H+K)\times d}$,单臂 $d=7$(位置 $x,y,z$+滚转/俯仰/偏航+夹爪开合度),双臂 $d=14$。

**多层级动作条件注入(3.1 节)**。① *Spatial-Aware Pose Injection*:把每个时间戳的 6D 末端位姿用标定相机参数投影到 2D 像素坐标,借鉴视觉提示(visual prompting)技巧,用沿三个方向轴的单位向量可视化 roll/pitch/yaw 姿态,用单位圆的深浅色表示夹爪开合度(浅色开、深色合),渲染在黑色背景上得到"动作图"(action map),再经 CLIP 视觉编码器提特征、与 RGB 观测特征沿通道维拼接送入 UNet。② *Delta Action Attention Module*:额外计算相邻帧间的末端位姿增量(delta motion),用线性投影器编码为固定数量的潜表示 token,再通过 cross-attention 与参考图特征融合后注入 UNet——这一模块显式引入了动作的一阶时间导数(近似速度/加速度信息),使模型能区分具有不同加速度特征的相似动作(见消融)。

**多视角条件注入(3.2 节)**。为覆盖头部相机与随手臂运动的腕部相机,EVAC 引入空间 cross-attention 模块实现跨视角信息交互,并用 ray map 编码相机运动:对每路相机计算其光线原点与方向 $r=(o_r, d_r)$ 相对不同时刻位姿的取值,由于腕部相机随手臂移动,其 ray map 天然隐式编码了末端位姿的运动信息,与动作图拼接后提供更丰富的轨迹上下文——这一设计解决了固定视角投影圆圈无法反映腕部相机自身运动的问题。

**两大应用(3.3 节)**。① *数据引擎*:人类先采集 $M$ 条示范轨迹,通过夹爪开合度变化识别接触起止时间戳,把每条轨迹切分为 fetch/grasp/home 三段;对 fetch 段的早期动作做空间增广并插值生成新的动作轨迹,固定接触时刻动作 $a_{t_b}$,将初始观测 $O_{t_b}$ 与反转后的增广动作序列喂给 EVAC 生成视频帧,再反转回正确时序,从而把 $M$ 条轨迹扩充为更多样的合成轨迹集合用于策略训练。② *评测器*:给定初始观测和指令,策略模型生成动作 chunk,连同当前观测一起送入 EVAC 生成下一步观测,如此迭代直到策略输出的动作幅度低于阈值;多名人工评估员(或未来可用 Video-MLLM)据生成视频判断任务成功与否,从而免去搭建复杂仿真资产或占用真实机器人的成本。

## 三、关键结果

**数据来源与规模**。训练数据主要来自 AgiBot World 数据集(210+ 任务、100 万+ 轨迹);为覆盖失败场景(对评测/数据增广至关重要),团队与 AgiBot-Data 团队合作获取原始数据中挖掘出的大量失败轨迹,并搭建自动化流水线在遥操作和真机推理中采集额外失败案例。

**实现与训练成本**。基座为 UNet 视频扩散模型,冻结 CLIP 视觉编码器与 VAE 编码器,微调 UNet/Resampler/线性层;batch size 16,单视角版本约 32×A100 训练 2 天,多视角版本约 32×A100 训练 8 天;memory 设为 4 帧(取自上一 chunk 生成结果)、chunk size 16 时质量与算力开销最佳;视频分辨率 320×512;策略模型统一采用官方单视角 GO-1。

**生成稳定性**。单视角设置下生成视频可在长达 30 个连续 chunk 内保持清晰、可靠;多视角设置受限于腕部相机常拍到人员走动等背景噪声,稳定推理仅能维持约 10 个 chunk。

**作为评测器**:选取 4 个抓取类任务(Take a Bottle / Take a Toast / Take a Bacon / Take a Lettuce),每个任务分别在真实机器人和 EVAC 模拟环境下评测 40 次,由三名独立评估员判定成功/失败。四个任务的绝对成功率差异很大(从约 25%~30% 到接近满分),但真实评测与 EVAC 评测的**相对趋势高度一致**;在同一策略于不同训练步数(约 4K/8K/13.5K step)下的对比中,EVAC 与真实评测捕捉到了相同的性能提升梯度,证明其能可靠反映训练过程中的性能波动。

**作为数据引擎**(Table 1,抓取一瓶被纸箱紧密包裹的水的任务):

| 训练数据 | 成功率 SR |
|---|---|
| Baseline(20 条专家示范) | 0.28 |
| Augmented(+30% EVAC 合成轨迹) | 0.36 |

**失败数据的作用**(定性,图 8):不引入失败轨迹训练的模型在机械臂"空抓"(未实际接触瓶子)场景下会过拟合成功案例,产生"幻觉"式地生成瓶子已被抓起的画面;引入失败数据后模型能正确识别抓取失败、生成符合物理的"空抓"结果。

**消融(Delta Action Attention)**:在需要区分"向上抛"与"向上晃"等高低加速度轨迹的复杂操作任务中,去掉 Delta Action 模块会导致生成视频出现物体闪烁/突然消失等时序不一致现象;加入该模块后运动一致性显著改善。此外附录给出了 LIBERO(417 条轨迹微调后)定性验证结果,以及多相机视角(head / head_left / head_right 鱼眼 / left_hand / right_hand)一致生成的示例,但未报告 LIBERO 上的定量成功率指标。

## 四、评价与展望

EVAC 的核心贡献在于把"动作条件"以多层级方式(空间位姿投影图 + 一阶差分注意力 + ray map)显式注入视频扩散模型,相比只做语言/图像条件的具身视频生成工作(如 RoboDreamer、This\&That 等)更贴近"世界模拟器"应作出的动作响应式生成;同时把这一世界模型同时用作数据引擎和策略评测器,是一个务实的双重落地路径,直接对接了 AgiBot World 这类大规模真机数据集与 GO-1 策略模型的实际迭代闭环。

局限与开放问题也比较明确:(1)论文自陈的核心限制是夹爪开合度用单位圆深浅色编码,难以泛化到灵巧手等更复杂末端执行器,迁移到新硬件形态需要额外的预处理与改造;(2)腕部相机背景噪声(如人员走动)显著增加生成难度,导致多视角推理的可持续 chunk 数(10)远低于单视角(30),这也是多视角具身视频生成的一个普遍瓶颈;(3)数据引擎实验只在单一任务(抓瓶)、单一增广比例(30%)上验证了 0.28→0.36 的提升,尚缺少跨任务、跨增广比例的系统性scaling 曲线,合成数据在多大程度上可替代/超越真实数据仍待更大规模验证;(4)评测器与真实机器人的一致性目前仅体现在"相对趋势"层面而非绝对数值吻合,论文也未给出量化的相关系数(如 Pearson/Spearman);(5)论文明确指出与 actor-critic/强化学习方法的结合尚属未探索的未来方向。整体上,EVAC 代表了"视频生成世界模型 = 数据引擎 + 廉价评测环境"这一路线的一个扎实但仍偏早期的实例,其失败数据增广(避免生成幻觉式"成功")的发现对后续具身世界模型训练具有较通用的参考价值。

## 参考

- Huang et al., *EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation*, arXiv:2501.01895 — EVAC 直接沿用的动作条件世界模型基座架构与稀疏 memory 机制。
- Bu et al., *Agibot World Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems*, arXiv:2503.06669 — EVAC 的主要训练数据来源(AgiBot World)及策略模型 GO-1 的出处。
- Xing et al., *DynamiCrafter: Animating Open-Domain Images with Video Diffusion Priors*, arXiv:2310.12190 — EVAC 采用的 UNet 视频扩散基座。
- Zhou et al., *RoboDreamer: Learning Compositional World Models for Robot Imagination*, arXiv:2404.12377 — 对比对象:偏语言条件的具身视频世界模型。
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, NeurIPS 2023 — EVAC 附录中用于定性泛化验证的仿真基准。
