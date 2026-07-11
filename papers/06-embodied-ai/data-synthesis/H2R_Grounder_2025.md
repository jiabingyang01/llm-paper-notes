# H2R-Grounder：无需配对数据的人类交互视频到物理合理机器人视频翻译范式

> **论文**：*H2R-Grounder: A Paired-Data-Free Paradigm for Translating Human Interaction Videos into Physically Grounded Robot Videos*
>
> **作者**：Hai Ci, Xiaokang Liu, Pei Yang, Yiren Song, Mike Zheng Shou\*(通讯作者)
>
> **机构**：Show Lab, National University of Singapore(新加坡国立大学)
>
> **发布时间**：2025 年 12 月（arXiv 2512.09406v1，10 Dec 2025）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.09406) | [PDF](https://arxiv.org/pdf/2512.09406)
>
> **分类标签**：`人到机器人视频` `视频扩散生成` `paired-data-free` `in-context learning`

---

## 一句话总结

把机器人视频里的机械臂"抠掉"得到干净背景、再叠加一个标记 gripper 二维位置与朝向的圆点+箭头,组成跨本体共享表征 **H2Rep**,用它以 in-context LoRA 方式微调 Wan 2.2 视频扩散模型学会"往场景里插回机械臂";测试时对人类视频做同样处理即可零配对地生成物理合理的机器人操作视频——在 DexYCB OOD 测试上人类偏好率在运动一致性/背景一致性/画质/物理合理性四项分别达 54.5%/56.8%/61.4%/63.6%,全面领先 Kling、Runway Aleph、RoboMaster。

## 一、问题与动机

机器人操作数据的采集始终是瓶颈:真机演示慢、贵、受实验室限制。相比之下互联网上的人类操作视频海量且行为多样,若能让机器人直接从人类视频学习将极大加速数据积累。但横亘其间的是 **视觉本体差异(visual embodiment gap)**:人的手臂/手在外观与运动上都与机械臂/夹爪差异巨大。

已有"机器人化(robotize)"路线(Phantom、Masquerade、H2R 等)把人手替换成渲染出来的机械臂,但存在两大硬伤:

- **物理不一致**:直接叠加的机械臂无视光照、深度与场景几何,常出现机械臂"悬浮"在物体上方、遮挡关系错误(论文 Fig.2 直接展示 Phantom 渲染臂悬浮、H2R 公开数据集机械臂漂移且相机错位)。
- **依赖标定**:需要精确的相机内外参与手-相机-机器人标定,在真实野外(in-the-wild)视频里根本拿不到。

本文提出 **H2R-Grounder**,核心洞见是:不需要任何配对的人-机演示,只用 **非配对的机器人视频** + 一个人机通用的抽象条件信号,就能训练一个"往场景里插机械臂"的生成器。关键在于——训练时模型看的是真实机器人视频,因此它天然观察到了正确的接触、遮挡与物理交互,而从不需要见过人。

## 二、核心方法

整体三阶段(论文 Fig.3):(1) 从机器人数据集构造训练数据;(2) in-context 微调视频生成器;(3) 把野外人类视频迁移为机器人视频。

**符号约定。**$V_r$、$V_h$ 分别为机器人视频与人类视频;$H_r$、$H_h$ 为对应的 H2Rep;$\mathcal{S}$ 为文本提示视频分割(Grounded-SAM2);$\mathcal{I}$ 为视频目标移除(inpainting);$\Pi$ 为 6-DoF 到 2D 的位姿投影;$\mathcal{R}$ 为把位姿渲染成图形叠层(红点表位置、蓝箭头表朝向);$\mathrm{Blend}(A,B;\alpha)=(1-\alpha)A+\alpha B$ 为 alpha 混合($\alpha=0.4$);$\mathrm{Enc},\mathrm{Dec}$ 为视频 VAE 编解码器,$e(\cdot)$ 为文本嵌入。

### 2.1 共享抽象 H2Rep

作者观察到人类交互视频与机器人视频都能自然分解为两部分:(i) 携带动作语义的 **位姿轨迹**(人手或机器人夹爪),(ii) 保留场景布局与被操作物体物理状态的 **背景视频**。只要把人手位姿和夹爪位姿对齐,"位姿序列 + 背景"就成为两域共享的信息载体,记作 **H2Rep**。

**从机器人视频构造 H2Rep。** 先分割机械臂掩码:

$$\mathbf{M}_r = \mathcal{S}(\mathbf{V}_r, \text{"robotic arm"})$$

把末端执行器(EEF)6-DoF 轨迹 $\mathbf{T}_\text{EEF}(t)=\langle \mathbf{p}(t), \mathbf{R}(t)\rangle$ 用相机内外参 $\langle \mathbf{K},\mathbf{R}_c,\mathbf{t}_c\rangle$ 投影到图像平面:

$$\mathbf{P}_r(t) = \Pi\big(\mathbf{K},\mathbf{R}_c,\mathbf{t}_c;\mathbf{p}(t),\mathbf{R}(t)\big)$$

再用视频 inpainting 移除机械臂得到干净背景:

$$\mathbf{V}_r^{\mathcal{I}} = \mathcal{I}(\mathbf{V}_r,\mathbf{M}_r)$$

(经验上 Minimax-Remover 比 E2FGVI 更可靠地保留背景、移除机械臂,论文 Fig.4 佐证。)最后把渲染出的位姿叠层与干净背景混合:

$$\mathbf{H}_r = \mathrm{Blend}\big(\mathbf{V}_r^{\mathcal{I}}, \mathcal{R}(\mathbf{P}_r);\alpha\big),\quad \alpha=0.4$$

得到训练对 $\mathcal{D}_r=\{(\mathbf{H}_r^{(i)},\mathbf{V}_r^{(i)})\}_{i=1}^N$,其中 $\mathbf{H}_r$ 携带夹爪运动与场景演化,$\mathbf{V}_r$ 是物理合理的重建目标。

> 用大白话说:H2Rep 就是"把机械臂 P 掉、只留干净桌面场景,再在夹爪该出现的地方画一个红点(表位置)加一个蓝箭头(表朝向)"。这张图既不含人也不含机械臂本体,所以人类视频和机器人视频都能被转成同一种长相,本体差异被抹平了。

### 2.2 物理合理机器人视频的 in-context 学习

训练一个条件视频生成器 $G_\theta$(Wan 2.2 骨干),在固定文本提示 $c_\text{text}=$"A robotic arm is interacting with objects." 下,以 $\mathbf{H}_r$ 为条件合成 $\mathbf{V}_r$。采用 in-context learning:把 $\mathbf{H}_r$ 与 $\mathbf{V}_r$ 用同一 VAE 编码、经自注意力融合;只训练 Q/K/V 投影上的 LoRA 适配器,骨干其余权重全部冻结:

$$\mathbf{z}_H=\mathrm{Enc}(\mathbf{H}_r),\quad \mathbf{z}_V=\mathrm{Enc}(\mathbf{V}_r),\quad \mathbf{c}=[\mathbf{z}_H;e(c_\text{text})]$$

用 flow-matching 目标训练:令 $\mathbf{z}_0=\mathbf{z}_V$,采样 $\mathbf{z}_1\sim\mathcal{N}(\mathbf{0},\mathbf{I})$,线性插值 $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$,目标速度 $\mathbf{v}_t=\mathbf{z}_1-\mathbf{z}_0$,学习条件速度场:

$$\mathcal{L}=\mathbb{E}_{t\sim\mathcal{U}(0,1),(\mathbf{H}_r,\mathbf{V}_r)\sim\mathcal{D}_r,\mathbf{z}_1\sim\mathcal{N}}\Big[\big\|u_\theta(\mathbf{z}_t,t,\mathbf{c})-\mathbf{v}_t\big\|_2^2\Big]$$

推理时 $\widehat{\mathbf{V}}_r=G_\theta(\mathbf{H}_r,\mathbf{z}_1,t,c_\text{text})$。

> 用大白话说:不改动预训练大模型的原有权重,只在注意力的 Q/K/V 上挂几个轻量 LoRA,让模型学"看到红点箭头 + 干净背景,就把一条符合物理接触/遮挡的机械臂长回去"。因为监督信号 $\mathbf{V}_r$ 来自真机视频,生成结果被鼓励物理合理。冻结骨干让模型保留强 OOD 泛化能力,才能迁移到没见过的人类视频。

### 2.3 人类视频 → 机器人视频

对任意第三人称 HOI 视频 $\mathbf{V}_h$,构造其 H2Rep 再喂给已训好的生成器:

- **人物分割 + 手部位姿**:先用 Grounded-SAM 2.1 得掩码 $\mathbf{M}_h=\mathcal{S}(\mathbf{V}_h,\text{"person"})$;用 ViT-Pose 估计人体位姿并定位手部框 $B_h$,再用 HaMeR 精估手部位姿 $P_\text{hand}$;取食指指尖与拇指指尖中点作为手的位置、拇指方向作为朝向,构成代理位姿 $\mathbf{P}_h=\mathcal{D}(\mathbf{V}_h)$(即用作 gripper 投影位姿的 2D 代理)。
- **人物移除(背景视频)**:$\mathbf{V}_h^{\mathcal{I}}=\mathcal{I}(\mathbf{V}_h,\mathbf{M}_h)$(仍用 Minimax-Remover)。
- **组合人类 H2Rep**:$\mathbf{H}_h=\mathrm{Blend}(\mathbf{V}_h^{\mathcal{I}},\mathcal{R}(\mathbf{P}_h);\alpha),\ \alpha=0.4$。
- **H2R 翻译**:直接把训好的机器人生成器作用于人类 H2Rep,$\widehat{\mathbf{V}}_r=G_\theta(\mathbf{H}_h,\mathbf{z}_1,t,c_\text{text})$。

关键点是:人手抓握由"食指-拇指中点+拇指方向"约化为与机器人夹爪投影位姿同格式的 2D 代理,配合冻结骨干+轻量 LoRA 的强 OOD 泛化,使得只在 Droid 室内数据上微调也能推广到野外人类视频。

**为何用 α-混合而非双视频流(补充材料)。** 另一种自然设计是把位姿与背景当两路独立视频流(位姿画在纯白/黑画布上)。但在 in-context 框架下双流会让输入 token 翻倍,计算与显存按平方级放大(约 4×)。α-混合以受控透明度叠加,既最小化对背景内容的干扰又大幅省算力,且与人类参考帧和最终机器人帧都保持像素对齐,便于生成器学习。

## 三、实验结果

**设置。** 训练用 Droid 数据集(约 76K 条第三人称 Franka 臂操作视频),留 50 条验证。OOD 评测用 DexYCB(第三人称、20 类物体),取 subject 01、相机 932122062010 俯视视角下 100 条视频作测试集;不使用其真值人手掩码/物体位姿,而用本文自动标注管线模拟真实测试条件。另收集互联网视频作定性对比。视频统一 1280×720、10 fps,帧数满足 $n \bmod 4 = 1$,每段采样至多 49 帧。骨干为 Wan 2.2 TI2V-5B,in-context 微调 200 步、mini-batch 4、8 张 H200、梯度累积 2;VACE 对照训 2 个完整 epoch。基线含渲染类(Phantom/Masquerade,因需标定被排除)、动画类(RoboMaster,半手工构造输入)、商用编辑(Kling、Runway Aleph),以及作为替代条件方案的 VACE(1.3B/14B,配 Qwen2.5-VL 生成详细字幕)。

**DexYCB 人类偏好率(Table 1,首选率 %)。** 22 名 CS 背景参与者对三方法排序,允许并列。

| 方法 | 运动一致性 | 背景一致性 | 视觉质量 | 物理合理性 |
|---|---|---|---|---|
| RoboMaster | 2.3 | 2.3 | 2.3 | 18.2 |
| Runway Aleph | 22.7 | 15.9 | 9.1 | 6.8 |
| Kling | 9.1 | 34.1 | 40.9 | 9.1 |
| **Ours** | **54.5** | **56.8** | **61.4** | **63.6** |

H2R-Grounder 四项首选率全面第一,尤以视觉质量(61.4%)与物理合理性(63.6%)领先。Kling 凭商用编辑管线在背景稳定与画质上第二,但运动一致性与物理合理性仅约 9%,常出现结构失真或不合理交互。RoboMaster 因预定义、非自适应运动表现最弱。

**DexYCB VLM 打分(Table 2,Gemini,1–5 分)。**

| 方法 | 运动一致性 | 背景一致性 | 视觉质量 | 物理合理性 |
|---|---|---|---|---|
| RoboMaster | 2.6 | 4.5 | 3.5 | 2.8 |
| Runway Aleph | **3.7** | 4.5 | 3.6 | 3.9 |
| Kling | 3.5 | **4.9** | **4.1** | 3.6 |
| **Ours** | **3.7** | **4.9** | 4.0 | **4.4** |

VLM 结论与人类偏好一致:本文在运动一致性(3.7)、背景一致性(4.9)、物理合理性(4.4)取得最高或持平分数,仅视觉质量(4.0)略逊 Kling(4.1,可能受益于其打磨过的渲染风格)。

**Droid 消融(Table 3,SSIM↑ / LPIPS↓)。**

| 配置 | SSIM ↑ | LPIPS ↓ |
|---|---|---|
| **H2R-Grounder 5B(本文)** | **0.82** | **0.22** |
| w/o pose indicator | 0.80 | 0.23 |
| w/o LoRA | 0.80 | 0.26 |
| w/ 14B backbone | 0.79 | 0.23 |
| w/ VACE (1.3B) | 0.68 | 0.30 |
| w/ VACE (14B) | 0.71 | 0.27 |

要点:(1) 去掉位姿指示会导致明显运动漂移,生成机械臂常偏离预期轨迹——位姿线索对运动控制不可或缺;(2) 不做 LoRA 微调,模型倾向过拟合、根本长不出机械臂;(3) 用 VACE(ControlNet 式条件)替换 in-context 生成器,SSIM 更低、LPIPS 更高,说明该条件方式对维持运动-背景一致性效果更差;(4) 换 14B 骨干无明显质量提升,却大幅拖慢推理并把序列长度从 49 帧压到 17 帧。综合精度与效率,最终采用 5B + in-context 方案。

**推理效率(补充材料)。** 5B in-context 模型约 13 秒/帧,在单张 H200 上生成一段 49 帧、704×1280 视频约需 648 秒,峰值显存 63 GB。

## 四、局限性

- **单手到单臂**:目前仅支持单手→单臂翻译;双手双臂场景作者称需要相应双臂机器人数据,留作未来工作。
- **本体绑定 Franka**:因训练全在含 Franka 臂的数据集上进行,当前只能产出 Franka 风格输出;适配其他机器人本体需为每种机型微调或训练轻量 LoRA。
- **依赖机器人真实轨迹作监督**:训练需从机器人视频拿到 6-DoF EEF 轨迹与相机参数做位姿投影(仅在训练侧,推理侧不需标定),因此对机器人数据集的元数据质量有隐性要求。
- **评测规模有限**:DexYCB 定量评测仅用单主体单视角 100 条视频,VLM 评审用单一 Gemini,人类研究 22 人;缺少下游策略学习(policy learning)的闭环验证——生成视频最终能否提升真机操作成功率未被直接度量。

## 五、评价与展望

**优点。**(1) **范式层面的巧思**:把"跨本体翻译"重构成"往抠干净的背景里按 2D 位姿提示插机械臂",既回避了配对人-机数据的采集,又通过"监督信号来自真机视频"天然获得正确的接触/遮挡/物理,这比 Phantom/Masquerade/H2R 那类"渲染叠加"更根本地解决了物理不一致问题。(2) **去标定**:用 2D 位姿点+箭头替代严格 3D 对齐,使方法能推广到无标定的野外视频,这是相对 Phantom 一系的实质进步。(3) **工程务实**:in-context + LoRA 冻结骨干既保留 Wan 2.2 的强先验与 OOD 泛化,又把可训参数压到极小,消融显示其显著优于 ControlNet 式的 VACE。

**缺点与开放问题。**(1) **缺下游闭环**:全文停留在"视频看起来物理合理"的感知级评测,没有证明生成视频用于模仿学习/表征学习能真正提升机器人策略——而这正是 Phantom、H2R、GigaWorld 等同类工作宣称的终极价值,也是本文动机的落脚点。(2) **2D 位姿代理的信息损失**:把手的姿态压成"两指中点+拇指方向"的 2D 点箭头,丢弃了深度、开合度、腕部自由度等信息,在需要精细抓握或强透视场景下可能约束不足;补充材料也承认 α-混合是为省算力对"更解耦的双流表征"做的折中。(3) **本体与相机分布窄**:仅 Franka + Droid 单风格,跨机型/跨视角的泛化仍待验证。

**与公开工作的关系。** 与 RoboMaster(单图动画、需手工物体掩码与轨迹)相比,本文全自动且强调运动-背景一致性;与 Phantom/Masquerade/H2R(渲染叠加、需标定)相比,本文走全生成路线、免标定且物理更合理;与 MimicDreamer(在机器人渲染上条件化生成、仍需标定)相比,本文条件信号更抽象。整体处在"生成式世界模型/视频作为数据引擎"(如 GigaWorld、DreamGen、Cosmos 一类)潮流中,定位是"从无标注人类视频扩展机器人学习"的数据合成器。可能的改进方向:接入下游策略学习做闭环增益验证;把 2D 位姿代理升级为带深度/开合度的 2.5D 提示;多本体/多相机联合训练;以及把 α-混合折中换成更解耦但算力可控的条件注入。

## 参考

1. Marion Lepert et al. *Phantom: Training robots without robots using only human videos*. 2025.(渲染叠加式 H2R,主要对比对象)
2. Guangrun Li et al. *H2r: A human-to-robot data augmentation for robot pre-training from videos*. 2025.(egocentric 渲染叠加,公开数据集被本文用于展示漂移伪影)
3. Xiao Fu et al. *RoboMaster: Learning video generation for robotic manipulation with collaborative trajectory control*. arXiv:2506.01943, 2025.(动画类主要基线)
4. Team Wan et al. *Wan: Open and advanced large-scale video generative models*. 2025.(TI2V-5B 生成骨干)
5. Zeyinzi Jiang et al. *VACE: All-in-one video creation and editing*. 2025.(ControlNet 式条件的对照方案)
