# Embody4D：面向具身 4D 世界建模的通用数据引擎

> **论文**：*Embody4D: A Generalist Data Engine for Embodied 4D World Modeling*
>
> **作者**：Peiyan Tu, Hanxin Zhu（共同一作）, Jingwen Sun, Shaojie Ren, Cong Wang, Yuyan Xu, Jiayi Luo, Xiaoqian Cheng, Zhibo Chen（通讯作者）et al.
>
> **机构**：浙江大学、北京中关村学院、中国科学技术大学、中国科学院自动化研究所、上海交通大学、北京航空航天大学
>
> **发布时间**：2026 年 06 月（arXiv 2605.01799）
>
> **发表状态**：未录用（预印本；正文注明匿名项目主页，疑似处于双盲评审中）
>
> 🔗 [arXiv](https://arxiv.org/abs/2605.01799) | [PDF](https://arxiv.org/pdf/2605.01799)
>
> **分类标签**：`数据引擎` `4D 世界模型` `新视角合成` `warp-then-inpaint` `视频扩散` `跨具身泛化`

---

## 一句话总结

Embody4D 用"MuJoCo 前景机械臂 + 真实背景"的组合式合成数据训练一个 warp-then-inpaint 范式的视频扩散模型,把单目机器人视频转成任意目标视角的时序一致视频("4D 具身数据"),核心创新是隐空间置信度专家路由(copy/repair/inpaint)与交互感知注意力两个模块,在 VBench+MEt3R 上取得 SOTA,并在真机 π0.5 策略上把成功率从 32%(单视角基线)提升到 74%。

## 一、问题与动机

具身智能体需要鲁棒的 3D 时空表征来支撑空间推理、操作理解与决策,但现实中采集的机器人数据几乎都来自固定或稀疏视角,只能提供局部、视角相关的观测,限制了多视角感知与跨视角泛化能力。在真实世界中大规模布设多相机同步系统成本高、难以规模化部署,而仿真中的多视角数据又与真实机器人交互(外观、动力学、接触行为)存在域差。

因此论文提出一种"数据引擎"范式:不直接采集多视角视频,而是训练一个专用的 video-to-video 4D 世界模型,把广泛存在的单目/稀疏视角机器人视频,转换为在目标相机视角下时序一致的稠密多视角视频("4D 具身数据"),从而弥补观测缺口,为下游机器人规划与策略学习提供稠密时空监督。

与依赖动作条件生成新交互轨迹的方法(如 Human2Robot、EnerVerse-AC)不同,Embody4D 走"新视角合成"路线:保留原始动作标注不变,只丰富每条轨迹的多视角观测,因而不受人到机器人动作迁移误差、或生成动力学与输入动作难以严格对齐等问题困扰。

同时,4D 生成模型本身受限于高质量多视角动态训练数据的稀缺,这一问题在具身场景下更为突出(缺少大规模、跨具身、带交互标注的多视角机器人视频),这构成论文要解决的第二个核心问题。

## 二、核心方法

Embody4D 建立在预训练的 TrajectoryCrafter(warp-then-inpaint 范式的相机轨迹重定向模型)之上微调,包含三个关键组件。

### 1. 组合式 4D 具身数据合成(Compositional 4D Embodied Data Synthesis)

为缓解具身 4D 数据稀缺,论文从 MuJoCo Menagerie 中选取 30 种跨形态机械臂(人形、单臂、双臂、小型夹爪),在 MuJoCo 中控制其绕默认姿态做随机运动,产生多样的前景机器人动态；背景则从 DL3DV 采样图像对,先按帧间相机平移最小化筛选候选对,再用 GPT-4o 过滤保证场景中心在多视角下均可见。

前景-背景合成时配置两台虚拟 MuJoCo 相机,其外参与选定的 DL3DV 图像对对齐。第一视角下机器人以随机旋转和缩放方式合成进参考视角；第二视角引入 **3D 锚点追踪** 策略以保持跨视角几何一致性:用 VGGT 重建源视频的逐像素深度与相机位姿,从源视频前 49 帧中随机采样子集 $\mathcal{S}$($|\mathcal{S}|=5$),把机器人中心 $\mathbf{u}_k$ 提升到 VGGT 世界坐标系后再变换到目标相机坐标系:

$$\mathbf{P}^c_{k\to t'} = \mathbf{R}_{t'}\mathbf{R}_k^{-1}\left(z_k\mathbf{K}^{-1}\bar{\mathbf{u}}_k - \mathbf{T}_k\right) + \mathbf{T}_{t'},$$

对采样帧取平均得到鲁棒 3D 锚点 $\bar{\mathbf{P}}^c_{t'} = \frac{1}{|\mathcal{S}|}\sum_{k\in\mathcal{S}} \mathbf{P}^c_{k\to t'}$,再投影回目标图像平面得到目标视角机器人中心。用大白话说:与其直接把机器人硬贴到第二个虚拟视角(容易漂移出画面或和背景透视矛盾),不如先用 VGGT 把源视频的深度/位姿都算出来,多帧取平均去噪后再重新投影,这样合成出的第二视角机器人位置就和第一视角在几何上自洽。

最终训练数据分两阶段:第一阶段用 23K 条组合合成的 4D 样本学习跨形态机器人动态与背景多样性；第二阶段用从 AGIBOT、RH20T、RoboSet、BC-Z、InternData-A1 五个数据集均匀采样的 24K 条真实单目具身数据(通过 warp-then-inpaint 构造伪 4D 监督)学习真实机器人交互操作。

### 2. 隐空间置信度感知专家调制(Latent Confidence-Aware Expert Modulation)

warp-then-inpaint 范式先用重建几何把源视角观测投影(warp)到目标相机,再用生成模型精修。但投影得到的先验在空间上是异质的:对齐良好的区域应保留,几何不一致区域需要修正,遮挡/高不确定区域需要补全。论文没有用粗糙的二值有效性掩码来划分,而是学习一张连续的隐空间置信度图,把目标视角生成分解为三种置信度条件行为:**copy(复制)、repair(修复)、inpaint(补全)**。

监督信号来自 VAE 隐空间中 warp 后视频与真实目标视频之间的隐残差,作为"理想修正量"的 oracle:

$$\mathcal{L}_\Delta = \left\|f_\phi\left(\mathcal{E}(\mathbf{x}_w), \text{Resize}(\mathbf{M}_s)\right) - \left(\mathcal{E}(\mathbf{x}_t) - \mathcal{E}(\mathbf{x}_w)\right)\right\|_2^2,$$

其中轻量 3D U-Net $f_\phi$ 只用 warp 后的隐向量与几何掩码作为输入(不依赖真实目标视频),预测该残差幅值,推理时用它生成连续的路由线索。用大白话说:与其直接判断"这块像素能不能信",不如训练一个小模型去预测"如果信了这块像素,还差多远才能对上真值",差得越多说明这块越不可靠,该交给 repair 甚至 inpaint 专家。

该路由线索被 token 化为逐 token 专家权重 $\mathbf{A}^{\text{tok}}$,作为 FFN 分支上的路由自适应专家残差叠加到 Transformer 块:

$$\mathbf{h}_n^{l+1} = \mathbf{h}_n^l + \mathbf{g}_n^l\left[\mathbf{u}_n^l + s\sum_{r\in\{\text{copy,repair,inpaint}\}} \mathbf{A}_{n,r}^{\text{tok}} E_r^l(\mathbf{u}_n^l)\right],$$

保留原始 FFN 通路作为稳定生成骨干,同时用置信度感知的专家残差自适应分配建模能力。附录给出的路由阈值为 $\tau_v=0.65,\ \tau_c=0.50,\ \tau_i=0.60,\ \tau_{iv}=0.50,\ \tau_f=0.80$,并采用 top-2 专家路由(辅助专家权重 $\alpha=0.20$)加局部平滑以避免路由碎片化。

### 3. 具身交互感知注意力(Embodied Interaction-Aware Attention)

为增强复杂交互(接触、抓取)的生成保真度,论文将交叉注意力拆分为全局路径与交互引导路径:

$$\mathbf{O}_{\text{global}} = \text{Softmax}\!\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V}, \qquad \mathbf{O}_{\text{guided}} = \text{Softmax}\!\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}} + \mathbf{B}\right)\mathbf{V},$$

其中当查询 token $i$ 与键 token $j$ 均落在机器人-物体交互区域内时 $\mathbf{B}_{i,j}=\lambda$,否则为 0(交互掩码由渲染直接获得,真实数据用 MemFlow 或 SAM3 提取)。两路径以课程系数 $\alpha$ 融合:

$$\mathbf{O} = (1-\alpha)\mathbf{O}_{\text{global}} + \alpha\,\mathbf{O}_{\text{guided}},$$

训练中 $\alpha$ 逐步增大,使模型从全局结构建模逐渐过渡到细粒度交互建模。为弥补合成数据中物体操作多样性不足,论文还用"前向-后向视角循环"(源视频 warp 到新视角再 warp 回原视角)构造伪 4D 配对数据补充交互监督。

**实现细节**:骨干为微调的 TrajectoryCrafter,分辨率 384×672,视频长度 49 帧,batch size 2,8×A100 训练;VGGT 作为几何重建基础模型估计相机参数与深度图。

## 三、关键结果

**视频生成质量**(VBench + MEt3R,对比 ReCamMaster、Ex-4D、Reangle-A-Video、TrajectoryCrafter):

| 方法 | Subject↑ | Background↑ | Temporal↑ | Motion↑ | Imaging↑ | MEt3R(3D 一致性)↓ |
|---|---|---|---|---|---|---|
| ReCamMaster | 0.8981 | 0.8976 | 0.9717 | 0.9841 | 0.5914 | 0.2454 |
| Ex-4D | 0.8088 | 0.8906 | 0.9213 | 0.9942 | 0.5732 | 0.2713 |
| Reangle-A-Video | 0.9152 | 0.9224 | 0.9711 | 0.9879 | 0.6437 | 0.2288 |
| TrajectoryCrafter | 0.9202 | 0.9388 | 0.9714 | 0.9911 | 0.6257 | 0.2040 |
| **Embody4D(Ours)** | **0.9351** | **0.9491** | **0.9734** | **0.9937** | **0.6566** | **0.1681** |

全部指标最优,尤其 MEt3R 3D 一致性相对 TrajectoryCrafter 提升约 17.6%。

**消融**(PSNR/SSIM/LPIPS/MEt3R,基线为仅用伪 4D 数据微调的 TrajectoryCrafter):

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | MEt3R↓ |
|---|---|---|---|---|
| Baseline | 19.03 | 0.6564 | 0.3303 | 0.1923 |
| + 组合数据(data) | 23.13 | 0.7899 | 0.2136 | 0.1757 |
| + data + 交互感知注意力(IA) | 23.49 | 0.7907 | 0.2112 | 0.1702 |
| + data + IA + 置信度专家(Expert,完整版) | **23.64** | **0.8064** | **0.1846** | **0.1681** |

完整模型相对基线 PSNR 提升 24.23%、SSIM 提升 22.85%、LPIPS 降低 44.11%、MEt3R 降低 12.58%,验证三个组件均有实质贡献且相互增益。

**真机下游策略实验**(Franka Research 3 + π0.5-Droid,5 项抓放任务、每任务 10 次试验,T1-T3 为已见任务、T4-T5 为分布外/OOD 任务,对比单视角基线、ReCamMaster、TrajectoryCrafter 增强训练):

| 方法 | T1 | T2 | T3 | T4(OOD) | T5(OOD) | 总成功率 |
|---|---|---|---|---|---|---|
| Single-view | 5/10 | 5/10 | 4/10 | 1/10 | 1/10 | 32% |
| ReCamMaster | 4/10 | 5/10 | 2/10 | 2/10 | 2/10 | 30% |
| TrajCrafter | 6/10 | 6/10 | 4/10 | 0/10 | 7/10 | 46% |
| **Embody4D** | **8/10** | **8/10** | **9/10** | **6/10** | **6/10** | **74%** |

**仿真基准**:在已趋于饱和的原始 LIBERO 上三种训练设置(原始数据 / Embody4D 增强 / 真实多视角 oracle)总成绩相近(96.9% / 95.8% / 96.1%)。在更具挑战性的 LIBERO-Plus(7 类扰动)上,Embody4D 增强训练在相机扰动上把成功率从 64.8% 提升到 86.8%(+22.0 个百分点),噪声扰动上从 78.8% 提升到 86.1%,且在相机、噪声、背景三类扰动上略优于"真实多视角 oracle"数据(作者归因于合成数据自带的背景变化与相机抖动起到了隐式数据增强作用);但语言、光照、布局等非视角类扰动上相对原始基线仍有下降,因为该方法本身只针对视角变化做增强。

## 四、评价与展望

**优点**:Embody4D 把 4D/新视角生成技术系统性地引入具身数据增强场景,三个组件设计针对性强——组合式合成数据用可控仿真解决数据稀缺与跨形态泛化,隐空间置信度路由用一个轻量残差预测网络把粗粒度二值 valid mask 升级为连续、可学习的软路由信号,交互感知注意力则直接针对操作类任务中最容易被生成模型"幻觉"掉的接触区域做偏置。三者的消融(Table 2)清晰显示逐项收益,真机实验(32%→74%)和 LIBERO-Plus 结果都给出了有说服力的下游任务证据,而不只是停留在生成质量指标上,这是许多同类"生成式数据引擎"论文常缺失的一环。

**与相关工作的关系**:方法论上直接建立在 warp-then-inpaint 范式(TrajectoryCrafter、ReCamMaster 一脉)之上做微调,而非从头训练生成模型,这既是效率优势也意味着上限受制于骨干模型本身的生成能力与时长限制。相较 ReCamMaster 依赖相机外参直接驱动生成(在大视角变化下几何畸变明显)、Ex-4D/Reangle-A-Video 依赖 LoRA 微调(补洞不完整、画质下降),论文声称的优势主要来自更充分利用了源视频几何先验(VGGT 深度/位姿)与更精细的区域路由。与 EnerVerse-AC、Human2Robot 等"动作条件生成新交互"路线相比,Embody4D 选择保留原始动作、只做视角扩展,规避了动作迁移误差,但代价是无法像动作条件方法那样合成全新的交互轨迹,本质仍是观测层面的增强而非数据量的凭空扩张。

**局限性**(原文明确指出):现有视频生成模型的时间窗口固定(49 或 81 帧),长输入视频须切分为多个 clip 分别生成,拼接时会在 clip 边界引入轻微抖动和画质下降,论文附录图 17 展示了失败案例,但未给出系统性的解决方案(如跨 clip 一致性约束)。

**开放问题与可能的改进方向**:(1)论文的置信度路由阈值($\tau_v,\tau_c,\tau_i,\tau_{iv},\tau_f$)是手工标定得到的,是否可以让路由函数完全可学习、去除这组超参数是一个自然的后续方向；(2)合成数据仅用 MuJoCo Menagerie 的运动学随机运动,物体交互多样性有限(论文自己也承认需要靠前向-后向视角循环构造伪配对来补充交互监督),更丰富的具身操作仿真(如带物理接触反馈的双臂协作)可能进一步提升复杂交互场景下的生成保真度；(3)LIBERO-Plus 上语言/光照/布局扰动的性能相对基线有所下降,说明该数据引擎目前只解决了"视角"这一个泛化轴,与其他数据增强手段(语言复述、光照随机化等)组合是否能取得互补收益值得探索；(4)真机实验样本量偏小(每任务仅 10 次试验),结论的统计显著性有限,更大规模、跨具身平台的验证会让"数据引擎"这一定位更有说服力。

## 参考

- Yu et al. *TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models*. arXiv:2503.05638, 2025.（本文微调的骨干模型）
- Bai et al. *ReCamMaster: Camera-Controlled Generative Rendering from a Single Video*. arXiv:2503.11647, 2025.（对比基线，游戏引擎合成相机条件数据）
- Wang et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025.（几何重建基础模型,用于估计相机参数/深度）
- Hu et al. *Ex-4D: Extreme Viewpoint 4D Video Synthesis via Depth Watertight Mesh*. arXiv:2506.05554, 2025.（对比基线）
- Jeong et al. *Reangle-A-Video: 4D Video Generation as Video-to-Video Translation*. arXiv:2503.09151, 2025.（对比基线）
