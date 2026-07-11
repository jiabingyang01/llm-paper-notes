# MVP-LAM：基于跨视角重建学习动作中心化的潜在动作

> **论文**：*MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction*
>
> **作者**：Jung Min Lee, Dohyeok Lee, Seokhun Ju, Taehyun Cho, Jin Woo Koo, Li Zhao, Sangwoo Hong, Jungwoo Lee
>
> **机构**：Seoul National University、Microsoft Research Asia、Konkuk University、HodooAI Labs
>
> **发布时间**：2026 年 02 月（arXiv 2602.03668）
>
> **发表状态**：ICML 2026（Proceedings of the 43rd International Conference on Machine Learning, PMLR 306）
>
> 🔗 [arXiv](https://arxiv.org/abs/2602.03668) | [PDF](https://arxiv.org/pdf/2602.03668)
>
> **分类标签**：`Latent Action Model` `多视角学习` `跨视角重建` `互信息` `VLA预训练`

---

## 一句话总结

MVP-LAM 用**跨视角重建**目标训练 latent action model：在同步多视角视频上,把从视角 A 编码出的离散潜在动作 token,拿去解码视角 B 的下一帧特征,迫使潜在动作丢弃视角相关线索、只保留跨视角共享的动作信息;在 Bridge V2 上潜在动作与真值动作的互信息比 UniVLA 提升约 62%(KSG 估计 1.10 vs 0.68),用作 VLA 预训练伪标签后 SIMPLER 平均成功率从 39.6%（UniVLA）提升到 60.4%,LIBERO(Bridge V2 预训练)平均成功率 94.1% vs UniVLA 92.5%,且在 LIBERO-Long 上以远少于 OXE 规模的机器人轨迹(≤60k vs ≥970k)反超了 state-of-the-art VLA π0（90.8 vs 85.2）。

## 一、问题与动机

机器人真实演示数据的采集成本高、速度慢,而海量人类操作视频是低成本、可规模化的替代信息源,但视频缺少低层动作标签,无法直接做模仿学习。近期方法学习 latent action model(LAM)——用一个 VQ-VAE 式的编解码器,把相邻帧转移编码成离散的"伪动作"token,再用它作为伪标签预训练 VLA(如 LAPA、Moto、UniVLA)。这类伪标签要真正有用,必须满足论文定义的**动作中心化(action-centric)**：潜在动作 $Z_t$ 应尽可能高地保留关于真值动作 $A_t$ 的互信息 $\mathcal I(Z_t;A_t)$,即便训练时根本看不到 $A_t$。

作者指出一个关键障碍——**外生噪声(exogenous noise)**：帧间视觉变化不仅由动作引起,还会被与动作无关的因素干扰。他们聚焦其中最突出的一种：**视角变化**。人类视频常来自第一人称、视角差异极大的多个来源,相机运动与动作驱动的转移相互纠缠;在有限容量的离散瓶颈(VQ 量化)下,LAM 会倾向于把编码容量"浪费"在视角相关的表观变化上,挤占本该用于编码真实动作的容量,导致学到的潜在动作和真值动作互信息偏低、不可控。已有的去噪思路(引入少量动作监督如 LAOM、物体中心分解、借助 VLM 过滤干扰)都引入了额外依赖(动作标签、可靠的物体分割、预训练 VLM 质量),且大多只在合成干扰的受控 benchmark 上验证。MVP-LAM 想在**不使用任何动作标注、不依赖额外模型**的前提下,单纯利用"同一时刻多个视角同步拍摄"这一结构信息,让潜在动作自动丢掉视角相关成分。

## 二、核心方法

**信息论动机。** 记视觉观测 $O_t=f(I_t)$、相机位姿(视角)$V_t$、状态 $S_t$、动作 $A_t$。目标是最大化 $\max_{Z_t}\mathcal I(Z_t;A_t)$。在简化假设下,论文给出一个下界：

$$
\mathcal I(Z_t;A_t) \ge \underbrace{\mathcal H(Z_t)}_{\text{总容量}} - \underbrace{\mathcal I(Z_t;V_t,V_{t+1}\mid S_t,S_{t+1})}_{\text{花在视角上的容量}} - C
$$

用大白话说：潜在动作的"表达容量"是固定的(离散码本大小 $\mathcal H(Z_t)$ 封顶),这块容量如果被视角信息占用得越多,能留给真实动作的就越少。要提高动作中心性,就要主动压低 $Z_t$ 对视角 $(V_t,V_{t+1})$ 的依赖。

**训练结构（图 2）。** 给定时间同步的双视角图像对 $(I_t^{v_1},I_t^{v_2})$,先用冻结的 DINOv2 提取特征 $o_t^v=f(I_t^v)$。对每个视角,时空 Transformer 编码器产生连续潜在 $e_t^v=E_\theta(o_t^v,o_{t+1}^v)$,再向量量化为离散 token $z_t^v=\mathrm{Quantize}(e_t^v)$,空间 Transformer 解码器 $D_\theta$ 用当前帧和潜在动作重建下一帧特征。核心创新是把两条 loss 结合：

- **自视角重建(self-viewpoint)**：$\mathcal L_{\text{self}}=\frac12\sum_{v}\lVert o^v_{t+1}-D_\theta(o^v_t,z^v_t)\rVert_2^2$——标准 LAM 目标,同视角内预测下一帧。
- **跨视角重建(cross-viewpoint)**：$\mathcal L_{\text{cross}}=\frac12\sum_{v\ne\tilde v}\lVert o^{\tilde v}_{t+1}-D_\theta(o^{\tilde v}_t,z^v_t)\rVert_2^2$——把视角 $v$ 编出的潜在动作 token **换到**另一个视角 $\tilde v$ 去解码它的下一帧(图 2 中的 "Swap" 操作)。

总损失为 $\mathcal L_{\text{MVP-LAM}}=\mathcal L_{\text{self}}+\mathcal L_{\text{cross}}+\mathcal L_{\text{quant}}+\mathcal L_{\text{commit}}$,其中量化/承诺损失沿用标准 VQ-VAE 形式 $\mathcal L_{\text{quant}}=\lVert \mathrm{sg}[e_t]-z_t\rVert_2^2$、$\mathcal L_{\text{commit}}=\beta\lVert e_t-\mathrm{sg}[z_t]\rVert_2^2$。

用大白话说：如果潜在动作里混入了"视角 A 特有"的信息(比如相机相对物体的角度),那么把它塞给视角 B 的解码器去猜视角 B 下一帧,结果一定很差,因为视角 B 根本不认识那些视角 A 专属的线索;唯一能让跨视角重建也做对的信息,是两个视角共享的东西——也就是真实发生的物理动作。解码器本身不输入视角信息,进一步杜绝了"作弊"（编码视角相关因子来降低重建误差）的空间。

**训练数据与规模。** LAM 训练集混合了 Open X-Embodiment(按 OpenVLA 训练配比)机器人轨迹与 EgoExo4D 多视角人类操作视频,共 312k 条轨迹,训练 160k 步。下游把离散潜在动作用作伪标签：先用交叉熵损失预训练 Prismatic-7B VLM 去预测 MVP-LAM 的潜在动作 token(图 6 左),再用 LoRA 把该 VLM 微调为 VLA,以 UniVLA 式的多头注意力机制把离散 token 解码为连续机器人动作(图 6 右)。

## 三、关键结果

**互信息与线性探针（Bridge V2）。** 三种互信息估计器(KSG、Barber-Agakov、MINE)均显示 MVP-LAM 潜在动作与真值动作互信息最高：

| 估计器 | MVP-LAM | UniVLA | LAPA | Moto |
|---|---|---|---|---|
| KSG | **1.10** | 0.68 | 0.50 | 0.37 |
| BA | **1.42** | 0.74 | 1.04 | 0.87 |
| MINE | **1.96** | 1.02 | 1.31 | 1.07 |

线性探针(冻结 LAM、只训练一层线性层从 $Z_t$ 预测动作,报告 NMSE,越低越好)在 Bridge V2 上 MVP-LAM 为 0.780 vs UniVLA 0.899；OOD 泛化到 LIBERO 各子集时,MVP-LAM 在 Spatial(0.547 vs 0.596)、Object(0.468 vs 0.545)、Long(0.617 vs 0.676)上均更低,仅 Goal 子集略逊(0.709 vs 0.708)。

**SIMPLER benchmark（用 MVP-LAM 潜在动作预训练 VLA 后微调）。**

| 方法 | StackG2Y | Carrot2Plate | Spoon2Towel | Eggplant2Bask | AVG 成功率 |
|---|---|---|---|---|---|
| MVP-LAM | 33.3 | 66.7 | 66.7 | 75.0 | **60.4** |
| UniVLA | 16.7 | 20.8 | 54.2 | 66.7 | 39.6 |
| LAPA† | 54.2 | 45.8 | 70.8 | 58.3 | 57.3 |
| OpenVLA† | 41.6 | 50.0 | 37.5 | 16.7 | 36.4 |
| π0 | 37.5 | 33.3 | 29.2 | 45.8 | 36.5 |

MVP-LAM 平均成功率全场最高(60.4%),四个任务上都比同架构、同数据源的直接对照组 UniVLA 高。

**LIBERO benchmark（Bridge V2 预训练组，与 OXE 大规模预训练组对比）。**

| 方法 | 预训练数据 | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|---|
| π0*（OXE，≥970k 轨迹） | — | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA（Bridge V2，≤60k 轨迹） | — | 95.2 | 95.4 | 91.9 | 87.5 | 92.5 |
| MVP-LAM（Bridge V2，≤60k 轨迹） | — | 96.0 | 94.6 | 94.8 | **90.8** | 94.1 |

MVP-LAM 用不到 π0 所需数据量的十六分之一(≤60k vs ≥970k 轨迹),且未使用 LIBERO 数据做 LAM/VLM 训练,平均成功率(94.1%)已逼近 state-of-the-art 的 π0(94.2%),并在最难的长时程套件 LIBERO-Long 上以 90.8% 反超 π0 的 85.2%,同时全面优于同数据规模的 UniVLA。

**消融（Table 3, Bridge V2 上 NMSE↓ / MI(KSG)↑）：** 仅机器人数据 + $\mathcal L_{\text{cross}}$：0.91 / 0.50；机器人+人类数据但**不用** $\mathcal L_{\text{cross}}$(即两条重建都是 self-viewpoint)：0.96 / 0.27(反而更差)；机器人+人类+$\mathcal L_{\text{cross}}$(完整 MVP-LAM)：**0.73 / 1.10**。说明单纯加入多视角人类视频而不改变训练目标是不够的、甚至可能有害，动作中心性主要来自跨视角重建这一目标本身,而非多视角数据本身。

**鲁棒性与可扩展性。** 对视角扰动(用新视角合成模型构造的 3.7k 扰动转移)、多相机同步误差($\ell\in\{0,2,4\}$ 帧的错位)MVP-LAM 的 MI/NMSE 几乎不变,显示其不需要严格的帧级多视角对齐即可工作；沿视角数、数据配比、模型规模三个轴放大时,动作中心性(MI↑、NMSE↓)持续改善,其中增加视角数带来的增益最大。

## 四、评价与展望

**优点：** (1)方法极简——只是把标准 LAM 重建目标换成跨视角版本,不引入动作标注、物体分割、VLM 过滤等额外依赖,理论(互信息下界)与实证(MI 估计、线性探针、消融)三线一致地支持"跨视角重建目标本身是动作中心性的来源"这一核心论点，而非仅仅因为用了多视角数据(Table 3 消融清楚地把两者解耦)。(2)下游收益扎实：同架构、同数据下相对 UniVLA 在 SIMPLER 上有大幅提升(39.6%→60.4%)，在 LIBERO 上用远少于 OXE 规模的数据逼近甚至反超 π0，说明潜在动作的信息质量比单纯堆数据规模更关键。(3)对同步误差和视角扰动的鲁棒性验证,为在"野生"多视角人类视频(难以做到帧级精确同步)上使用该方法提供了可行性依据。

**局限：** (1)方法本质依赖时间同步的多视角采集,虽然人类视频的多视角采集比大规模机器人遥操作演示更易获得,但仍比单目视频要求更高的采集基础设施；论文也承认这是未来需要放松（弱同步/伪配对多视角数据）的方向。(2)全部实验限于仿真(SIMPLER、LIBERO),未做真实机器人验证，作者在局限中明确指出这一点。(3)"动作中心性高"只是必要而非充分条件——论文附录讨论了潜在动作表示的"最小性(minimality)"问题：一个同时编码动作和视角的表示也可能表现出高互信息,该文用线性探针和 MI 估计间接佐证但未给出更严格的最小性保证。

**与其他工作的关系：** MVP-LAM 直接沿用 UniVLA 的 LAM 架构与动作解码 pipeline 作为对照基线,是对其目标函数的针对性改造，因此论文能够做"控制变量"式的干净比较（同架构、同下游 VLA 微调流程，仅改变 LAM 训练目标）。与 LAOM(引入少量动作监督去噪)、object-centric latent action(依赖物体分解)、VLM-guided latent action(依赖预训练 VLM 质量)等去噪外生噪声的路线相比，MVP-LAM 不新增外部依赖，代价是仅针对"视角变化"这一种噪声源建模，对光照、背景动态物体等其他外生噪声未做专门处理。开放问题包括：能否将跨视角重建思想推广到更弱的多视角配对信号(如非严格同步、单目视频通过合成新视角构造伪配对)、以及跨视角目标与显式动作最小性约束(如信息瓶颈正则)结合能否进一步压榨潜在动作的动作纯度。

## 参考

- Bu, Q. et al. UniVLA: Learning to act anywhere with task-centric latent actions. arXiv:2505.06111, 2025.（本文的主要对照基线与 LAM 架构基础）
- Ye, S. et al. LAPA: Latent action pretraining from videos. arXiv, 2024.（离散潜在动作预训练 VLA 的代表工作）
- Chen, Y. et al. Moto: Latent motion token as the bridging language for robot manipulation. arXiv:2412.04445, 2024.（大 VQ 码本的隐动作 token 方法）
- Nikulin, A. et al. Latent action learning requires supervision in the presence of distractors (LAOM). ICML, 2025.（外生噪声/引入动作监督去噪的相关工作）
- Kim, M. et al. OpenVLA: An open-source vision-language-action model. arXiv:2406.09246, 2024.（下游 VLA 对照基线之一）
