# DeepVerse：4D自回归视频生成作为世界模型

> **论文**：*DeepVerse: 4D Autoregressive Video Generation as a World Model*
>
> **作者**：Junyi Chen, Haoyi Zhu et al.（通讯作者 Tong He）
>
> **机构**：上海交通大学（SJTU）、上海人工智能实验室（Shanghai AI Lab）、中国科学技术大学（USTC）、清华大学（THU）、浙江大学（ZJU）、复旦大学（FDU）、南洋理工大学（NTU）
>
> **发布时间**：2025 年 06 月（arXiv 2506.01103）
>
> **发表状态**：未录用（预印本，首页标注 "Preprint."）
>
> 🔗 [arXiv](https://arxiv.org/abs/2506.01103) | [PDF](https://arxiv.org/pdf/2506.01103)
>
> **分类标签**：`世界模型` `4D表征` `自回归视频生成` `深度感知记忆` `游戏环境仿真`

---

## 一句话总结

DeepVerse 把视频世界模型的状态从纯 RGB 观测 $v_t$ 扩展为"视觉 + 几何"的复合 4D 状态 $\hat{s}_t=(v_t,g_t)$（$g_t$ 含深度与相机位姿），并配合几何感知的记忆检索机制,在 MM-DiT 自回归扩散框架下将长时程生成的 FVD/VBench 指标相较去掉深度模态的基线全面提升（例如 120 帧时 subject consistency 从 0.76812 提升到 0.81652,imaging quality 从 0.37975 提升到 0.44639）。

## 一、问题与动机

交互式世界模型的在线自回归方法（如 GameNGen、Oasis、DIAMOND）普遍存在误差累积与"遗忘"问题：视频本质是动态 3D/4D 物理世界的 2D 投影,若模型只在像素/视觉空间里做自回归预测,而不显式建模底层几何结构,长时程预测就会不可避免地漂移和失真。已有工作多从"如何更高效地压缩/保留历史帧"入手（如 FramePack 把历史帧压成定长表示）,但这类纯视觉策略绕不开一个根本问题：单目图像天生存在尺度歧义（scale ambiguity）,缺乏显式 3D 先验会导致新视角生成时几何不一致、误差逐帧传播。

DeepVerse 的出发点是：将 POMDP 中难以直接获得的隐状态 $s_t$ 用一个更informative的复合代理状态来近似,即同时预测视觉观测和几何观测（深度、相机位姿）,并让当前预测显式条件于历史时刻的几何估计,从架构层面抑制漂移与遗忘。

## 二、核心方法

**复合 4D 状态表示。** 定义

$$\hat{s}_t=(v_t,g_t) \tag{1}$$

其中 $v_t$ 是视觉观测,$g_t$ 编码与相机视角 $c_t$、深度 $d_t$ 相关的局部 3D 几何信息。大白话说：不再只让模型"看图猜下一帧",而是同时告诉它"这一帧对应的深度图和相机在哪、朝哪",把 2D 投影的歧义显式补全成 3D/4D 信息。深度采用视差平方根参数化 $e_t=\sqrt{1/d_t}$,相机视角用 raymap（每像素的射线原点 + 方向,共 6 通道）参数化,三者沿通道维拼接后统一送入预训练 VAE 编码,以复用图像 latent 空间的自回归生成能力。

**带选择性记忆的自回归预测。** 整体建模为

$$f_\theta = P\big(\hat{s}_{t+1:t+k}\mid a_t,\hat{s}_t,\hat{s}_{t-m:t-1},\psi(\hat{s}_{0:t-m-1})\big) \tag{2}$$

即预测未来 $k$ 步状态时,条件于当前动作 $a_t$、当前状态、最近 $m$ 步历史,再加上从更早历史中通过选择机制 $\psi$ 挑出的少量"显著相关"旧状态。这个设计避免了把全部历史都塞进有限上下文窗口。

**几何感知记忆检索。** 借鉴 GigaGS 的空间邻近状态选择策略,记忆检索定义为

$$\psi(\hat{s}_t\mid\{\hat{s}_{t-1},\dots,\hat{s}_0\})=\underset{j\in S}{\arg\min}\ \angle(R_t,R_j),\quad S=\underset{i\in\{t-1,\dots,0\}}{\arg\min}^{(k)}(T_t-T_i)^2 \tag{3}$$

即先按平移距离 $T$ 找出 $k$ 个空间上最近的历史观测,再从中选旋转角度 $R$ 最接近当前视角的一个作为"空间条件"注入当前预测。大白话说：模型在生成新画面前会先"回头看看"自己之前在同一片空间、同一朝向拍到过什么,以此对齐几何、防止走远了忘记原来的场景长什么样。

**通用控制（General Control）。** 与以往方法（如 GameNGen）额外采集控制器信号作为独立模态不同,DeepVerse 刻意不引入新模态,而是把控制信号统一转成文本描述（如相机的前后左右移动、顺/逆时针旋转都能由相机位姿变化算法式地转成文字）。这样既能最大程度复用预训练视频生成模型的文本控制能力,又便于对新控制器做轻量微调。

**数据构造。** 采集约 1000 万帧游戏录屏,用 ReShade 去除 UI 元素;沿用同组前作 Aether 的相机自动标注管线获取精确内外参、深度图；按"chunk 内累计旋转角低于阈值 $\delta_{rot}$、位移超过阈值 $\delta_{move}$"做质量过滤；用 Qwen-VL 做第一/三人称视角变化描述,CLIP+T5 生成文本 embedding（沿用 SD3 的方式）。

**架构与训练目标。** 骨干为 MM-DiT,以 flow matching 训练。论文对比了两种历史信息注入方式：Model 1（通道维拼接,仿 GameNGen,用 SD3-medium 初始化）与 Model 2（token 维拼接,仿 Pyramid-Flow,用 Pyramid-Flow 初始化,3D VAE 做 8 倍时间压缩）。两者均为 2B 参数,在 A100 上用 FSDP+ZeRO-2 训练。最终模型（附录）采用 token-wise 方案：24 层、hidden dim 1536、24 个注意力头、raymap 用关键帧策略压缩通道数（80→38 通道）,分类器无关引导（CFG）对文本条件和空间条件分别用引导系数 4 和 5,训练集约 150 万段视频片段,2 个 epoch 共耗时约 23,000 A100 GPU 小时。

## 三、关键结果

评测统一采用 VBench 六项指标（subject/background consistency、aesthetic/imaging quality、motion smoothness、dynamic degree）与 FVD,在 32/60/64/96/120/128 帧等不同长度下对比。

**架构对比（Model 1 vs Model 2）：** token-wise 拼接（Model 2）在几乎所有 VBench 指标上优于 channel-wise 拼接（Model 1）,尽管计算量更高（平均 1280.9 GFLOPs vs 1049.4 GFLOPs）。作者指出 channel-wise 方案在 DOOM 这类专用游戏场景（GameNGen/DIAMOND 的设定）中具竞争力,但把多帧历史压进单一 token 会在更长、更通用的场景下加剧误差累积,因此最终架构选用 token-wise。

**深度模态消融（Table 1,核心结果）：**

| 配置 | 帧数 | subject consistency | background consistency | aesthetic quality | imaging quality | motion smoothness | dynamic degree |
|---|---|---|---|---|---|---|---|
| w/ depth（本文） | 60 | 0.86939 | 0.92617 | 0.53415 | 0.48844 | 0.99032 | 1.00000 |
| w/ depth（本文） | 120 | 0.81652 | 0.91087 | 0.50028 | 0.44639 | 0.99147 | 1.00000 |
| w/o depth | 60 | 0.83602 | 0.91899 | 0.49106 | 0.43774 | 0.98975 | 1.00000 |
| w/o depth | 120 | 0.76812 | 0.89650 | 0.44095 | 0.37975 | 0.99062 | 1.00000 |

引入深度模态在 60/120 帧下于 subject consistency、background consistency、aesthetic/imaging quality 上均一致优于去掉深度的基线,且优势随帧数增加（漂移更严重时）而扩大；配套 FVD 曲线（32→128 帧）显示 w/ depth 全程更低,验证几何约束能有效缓解自回归漂移。

**空间记忆检索的定性效果：** 论文可视化对比"有/无空间条件"在"离开场景再返回"（go away and come back）场景下的生成结果,显示引入几何感知记忆检索能让模型在长时程生成中正确"记起"此前访问过的空间区域,而无空间条件的基线会出现场景漂移/失配。

**仿真质量：** 分别以游戏截图、真实世界照片、AI 生成图像（Dreamina 文生图模型产出）作为单张起始帧,DeepVerse 均能生成保持视角-物体动态一致、遵循文本/控制条件的连续 4D 未来序列,论文强调其区别于"先重建再重渲染"范式,而是同时预测视角动态与环境交互。

## 四、评价与展望

**优点与定位。** DeepVerse 的核心贡献是把显式几何（深度 + 相机位姿）作为一等状态变量纳入自回归视频世界模型,并给出了一套完整的工程方案：几何编码方式（raymap+视差平方根深度）、关键帧式的通道压缩、几何感知记忆检索、以及"控制信号统一转文本"的通用控制范式。相较于 GameNGen、Oasis、DIAMOND、GameFactory 等纯视觉自回归交互式视频生成工作,以及 Genie 用隐动作模型（LAM）做通用控制的路线,DeepVerse 提供了一个更强调"3D/4D 一致性优先于纯像素保真度"的替代思路,和同团队前作 Aether（几何感知统一世界建模）一脉相承,是其在自回归视频框架下的延伸。与同样做长期记忆的 WorldMem（用 3D 位姿做历史检索）相比,DeepVerse 的记忆检索规则（式 3,先按平移距离筛近邻再按旋转角选最相关）更简单显式,且是首个把该思路整合进 4D 自回归视频生成框架的工作（作者在 Related Works 中明确声称"is the first to incorporate 4D representations into auto-regressive world models"）。

**局限性（原文明确列出）：** 训练数据完全来自游戏渲染的合成数据（ReShade 处理后的约 1000 万帧游戏录屏 + Aether 管线标注),尚未验证在真实世界场景中的泛化能力,这是论文自陈的主要局限,也是后续工作的直接改进方向。

**开放问题与可能的改进方向：** (1) 深度/位姿标签目前依赖游戏引擎或合成管线的"精确"标注,如何在真实视频（缺乏 ground-truth 深度）上获得同等质量的几何监督（例如结合单目深度估计或 SfM/SLAM 伪标签）是 sim-to-real 泛化的关键；(2) 控制信号统一转文本虽然提升了通用性,但相较于结构化的隐动作/连续控制信号,文本条件在细粒度、高频控制（如机器人操作的关节级动作）上的精度和响应延迟尚未验证,论文实验也未涉及机器人操作任务,仅在游戏/通用场景视频上评测；(3) 几何感知记忆检索目前只按位姿邻近度选取单一历史帧,尚未探讨检索多帧融合或与语义信息联合检索的效果；(4) 23,000 A100 GPU 小时、2B 参数的训练成本相对较高,论文未给出更小规模/更高效训练配置下的性能曲线,可比性和可复现性上留有空间。

## 参考

- Aether Team et al. *Aether: Geometric-aware unified world modeling*, arXiv:2503.18945, 2025（同团队前作,DeepVerse 的相机标注管线与几何范式直接沿用）
- Valevski et al. *Diffusion models are real-time game engines*（GameNGen), 2024（channel-wise 历史拼接对比基线）
- Jin et al. *Pyramidal flow matching for efficient video generative modeling*, arXiv:2410.05954, 2024（token-wise 拼接架构与初始化来源）
- Xiao et al. *WorldMem: Long-term consistent world simulation with memory*, 2025（同样用 3D 位姿做历史记忆检索的对比工作）
- Bruce et al. *Genie: Generative interactive environments*, ICML 2024（隐动作模型通用控制路线的对照）
