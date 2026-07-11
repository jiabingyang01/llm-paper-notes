# Real2Edit2Real：通过 3D 控制界面生成机器人演示数据

> **论文**：*Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface*
>
> **作者**：Yujie Zhao、Hongwei Fan（共同一作）、Di Chen、Shengcong Chen、Liliang Chen、Xiaoqi Li、Guanghui Ren、Hao Dong（通讯作者）
>
> **机构**：北京大学计算机学院 CFCS；PKU-AgiBot Lab；AgiBot（智元机器人）
>
> **发布时间**：2025 年 12 月（arXiv 2512.19402）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2512.19402) | [PDF](https://arxiv.org/pdf/2512.19402)
>
> **分类标签**：`数据合成` `3D编辑生成演示` `多视角视频生成` `VLA预训练`

---

## 一句话总结

Real2Edit2Real 用"真机重建 → 3D 点云编辑 → 深度控制的多视角视频生成"三段式流水线,把 1–5 条真机演示扩增成 200 条空间多样的多视角操作视频,不依赖仿真器与数字资产、纯 RGB 输入且兼容主流 VLA;在四个真机任务上用生成数据训练 Go-1 / π0.5,仅需 5 条源演示即可达到 78.8% / 81.3% 成功率,超过用 50 条真机演示训练的 61.3%,数据效率提升 10–50 倍。

## 一、问题与动机

VLA 与 Diffusion Policy 的鲁棒性高度依赖大规模、多样化的高质量演示,而覆盖物体空间随机摆放(spatial generalization)所需的真机采集成本极高、重复劳动繁重。已有的"少样本扩增"路线各有短板:

- **MimicGen 系列**(MimicGen / SkillMimicGen / DexMimicGen / DemoGen)在图形引擎里做物体中心的轨迹切分与插值,存在 Sim2Real 视觉/物理 gap,且需要被操作物体的数字资产,无法直接增广已有真机演示。
- **DemoGen** 把 MimicGen 式生成与点云编辑结合、增强了 3D Diffusion Policy 的空间泛化,但只作用于点云,**无法用于 RGB 图像与 2D 策略**——而后者才是当前 VLA 部署的主流。
- **Real2Render2Real / RoboSplat** 用 3DGS 做真实到仿真,视觉保真度提升,但仍有渲染域差、且稠密扫描限制了可扩展性。

核心痛点:**目前没有方法能在保留视觉真实性与交互保真度的前提下,快速把真机 2D 多视角操作视频扩增出新轨迹**。作者的关键洞察是——**深度天然编码了机器人运动与物体交互,是连接 3D 编辑能力与 2D 观测的自然界面**。于是提出用深度作为主控制信号,把"3D 可编辑性"注入"2D 视觉数据"。

## 二、核心方法

设一条源演示为 $\mathbf{O} = (\mathbf{I}, \mathbf{q}, \mathbf{a})$,含多视角视频 $\mathbf{I}=\{(I_h^t, I_l^t, I_r^t)\}$(头部 + 左右腕三路 RGB)、关节角 $\mathbf{q}$、动作 $\mathbf{a}$;$\mathcal{K}=\langle \mathcal{K}_{robot}, \mathcal{K}_{cam} \rangle$ 为 URDF 与相机内外参。目标是把它扩增成一大批新配置演示:

$$
\mathcal{O} = \{\mathbf{O}_i\}_{i=1}^N = \text{Real2Edit2Real}(\mathbf{O}, \mathcal{K})
$$

整套框架分三个模块,依次对应"重建—编辑—生成"。

### 模块一:Metric-scale 几何重建(Metric-VGGT)

先把源演示重建成度量尺度、多视角一致的深度与相机位姿:

$$
\mathbf{D}, \mathbf{T} = \mathcal{R}(\mathbf{I})
$$

前馈重建模型(VGGT)直接用会遇到域差:相机位姿错、度量尺度与真机不匹配。作者提出**混合训练**——真机数据只有可靠的度量尺度(但深度传感噪声大),仿真数据只有精确相机位姿与干净深度(但物体/场景尺度分布偏),二者互补。三项损失如下:

$$
\mathcal{L}_{\text{camera}} = \sum_{v\in\{h,l,r\}} \mathcal{L}_1(\hat{T}_v^{\text{sim}}, T_v^{\text{sim}})
$$

$$
\mathcal{L}_{\text{depth}} = \sum_{v\in\{h,l,r\}}\Big(\mathcal{L}_{\text{conf}}\big(\mathcal{M}(\hat{D}_v^{\text{real}}), \mathcal{M}(D_v^{\text{real}})\big) + \mathcal{L}_{\text{conf}}(\hat{D}_v^{\text{sim}}, D_v^{\text{sim}})\Big)
$$

$$
\mathcal{L} = \lambda\,\mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pointmap}}, \quad \lambda = 10
$$

**用大白话说**:相机位姿只信仿真(它有完美 ground truth),所以 $\mathcal{L}_{\text{camera}}$ 和点图损失 $\mathcal{L}_{\text{pointmap}}$ 只在仿真上算;深度则真机和仿真都学,但真机深度先用阈值掩码 $\mathcal{M}$ 把无效噪声区滤掉再算带不确定度的 conf loss。$\lambda=10$ 是为了把相机损失量级抬到与其他损失可比、稳定优化。这样训出的 Metric-VGGT 兼得"仿真的精确位姿 + 真机的正确尺度",给出干净点云,是后续编辑的可靠底座。

### 模块二:深度可靠的 3D 空间编辑

在重建点云上做物体重摆与轨迹合成,输出**运动学一致的深度序列**作为控制信号:

$$
\{(\mathbf{D}_i, \mathbf{T}_i, \mathbf{a}_i)\}_{i=1}^N = \mathcal{E}(\mathbf{D}, \mathbf{T}, \mathcal{K}, \mathbf{q}, \mathbf{a})
$$

关键设计(受 DemoGen 启发但作了 RGB 化改造):

- **轨迹分段**:把演示拆成 **motion segment**(机器人在空中自由移动,用 CuRobo 运动规划重新生成到新位姿的路径)与 **skill segment**(机器人与物体接触,对物体施加刚体变换 $\mathbf{T}$ 时,把同一变换施加到机器人相应段的点云上),从而保持"机器人—物体"相对关系不变、交互真实。
- **机器人位姿矫正(RPC)**:此前编辑把整臂当刚体搬走会破坏运动学。RPC 只变换末端执行器,其余臂段用 IK 重新对齐以保持运动学有效;并用 URDF + 源关节角把原机器人链分割出去,再按合成动作重渲染臂的深度,得到无刚体拖影的可靠深度图。
- **背景补全**:物体/机器人移动后投影深度会留空洞,先用图像编辑模型(SeedEdit)inpaint 背景、再用 RANSAC 平面对齐(Algorithm 2)把 inpaint 引入的尺度不一致修正回度量尺度。

### 模块三:3D 控制的多视角视频生成

把编辑好的深度/动作/位姿序列送入视频扩散模型 $\mathcal{G}$(基于 GE-Sim,底座 Cosmos-Predict-2B),从首帧生成写实、多视角一致、物理合理的操作视频:

$$
\{\mathbf{I}_i\}_{i=1}^N = \{\mathcal{G}(\mathbf{D}_i, \mathbf{T}_i, \mathbf{a}_i, \mathcal{C}(\mathbf{D}_i))\}_{i=1}^N
$$

其中 $\mathcal{C}(\cdot)$ 是从深度算出的 Canny 边缘。三个关键设计:

- **Dual-attention**:intra-view attention 在单视角内做自注意力(捕捉视角内空间细节)、cross-view attention 跨所有视角同时自注意力(利用多视角对应关系),兼顾多视角一致性并显著降低相比全局注意力的计算成本。
- **深度控制界面**:把深度图与图像 latent 拼接后一起喂给 transformer,让生成条件化于 3D 结构线索;辅以 Canny 边缘、动作、ray map,进一步锐化物体边界、增强运动 grounding 与多视角一致性。
- **平滑物体重定位(SOR)**:首帧里如何把物体挪到目标位是难点,作者把物体重定位转成一个平滑变换,操作开始前对平移与旋转做插值,合成物体移动的过渡轨迹,把"图像编辑"变成"视频生成"、与演示生成统一到一个流水线。
- **条件 dropout**:训练时深度与 Canny 各以 0.5 概率独立丢弃、以 0.1 概率联合丢弃,避免强度类条件主导、提升对编辑后不完美控制信号的鲁棒性。

## 三、实验结果

硬件为 Agibot Genie G1 人形机器人(头 + 左腕 + 右腕三路 RGB,50cm×40cm 桌面工作区)。四个真机任务:Mug to Basket(单臂放杯入篮)、Pour Water(左臂拎壶倒水)、Lift Box(双臂抬箱)、Scan Barcode(左手持物 + 右手扫码,双臂)。两个 VLA 策略:**Go-1**(仅微调 action expert、冻结 backbone,输出 6D 末端位姿)、**π0.5**(因具身不匹配全参微调,输出 7-DoF 关节角)。生成数据训练时,从采集数据里随机采若干条源演示、对物体做随机重摆(40cm×40cm 平移 + 30°–60° 旋转),每条源合成出 200 条,**只用生成数据训练**。

### 主结果:生成数据 vs 真机数据(四任务平均成功率)

| 训练数据 | Go-1 | π0.5 |
|---|---|---|
| Real 10 | 36.3% | 32.5% |
| Real 20 | 48.8% | 45.0% |
| Real 50 | 61.3% | 61.3% |
| Real 1 + Gen 200 | 65.0% | 57.5% |
| Real 2 + Gen 200 | 70.0% | 70.0% |
| **Real 5 + Gen 200** | **78.8%** | **81.3%** |

真机数据在演示 $\le 20$ 条时平均成功率跌破 50%,空间泛化能力弱;仅用 **1 条源演示**生成的 200 条即可匹配 50 条真机(Go-1 65.0% vs 61.3%);用 **5 条源演示**生成的数据反超 50 条真机 17.5 / 20.0 个百分点,数据效率提升 **10–50 倍**。

### 生成数据规模化(单条源演示,Table 9,四任务平均)

| 生成条数 | Go-1 | π0.5 |
|---|---|---|
| Gen 50 | 50.0% | 28.8% |
| Gen 100 | 57.5% | 41.3% |
| Gen 200 | 65.0% | 57.5% |
| Gen 300 | 73.8% | 62.5% |
| Gen 400 | 77.5% | 75.0% |

仅从 1 条源演示,生成超过 300 条时两个策略均已超过 50 条真机的水平,呈现清晰的规模化增益。Diffusion Policy 上也验证了同样趋势(Mug to Basket:Real 50 得 11/20,而 Real 5 + Gen 200 得 17/20)。

### 扩展能力

| 任务/设置 | 训练数据 | 结果 |
|---|---|---|
| 高度泛化(Go-1,新平台高度) | Tabletop Real 20 | 桌面 5/5,平台 0/5,总 50% |
| 高度泛化(Go-1) | Tabletop Real 1 + Gen 40(含平台) | 桌面 4/5,平台 4/5,总 **80%** |
| 纹理泛化(Go-1,5 种桌面色) | Real 50 | 总 50% |
| 纹理泛化(Go-1) | Real 1 + Gen 200(纯原纹理) | 总 52% |
| 纹理泛化(Go-1) | Real 1 + Gen 200*(含多纹理) | 总 **68%** |

即由于首帧可编辑,框架能通过换背景纹理、编辑物体高度来提升策略对未见高度/纹理的鲁棒性,展示了作为统一数据引擎的灵活性。

### 视频生成质量与效率

生成视频质量显著优于底座 GE-Sim 的条件 I2V(FVD 663.4→**352.9**、LPIPS 0.2038→**0.1252**、SSIM 0.7491→**0.8647**、PSNR 20.41→**22.95**)。8×H100 并行下,一段 20 秒 30 FPS 视频平均生成约 48.6 秒;时间分析显示 GPU 数越多,生成吞吐越快、可在短时间内超越人工遥操的成功率曲线。

### 消融

- **几何重建**(Fig 5):Metric-VGGT 给出最干净点云与最准相机位姿,原始 VGGT 与真机数据均含大量杂点/位姿误差。
- **机器人位姿矫正 RPC**(Fig 6):去掉后深度图错误 → 生成模糊、不一致;加上后运动学一致、机器人动作真实。
- **平滑物体重定位 SOR**(Fig 7):去掉后物体摆放明显错位、演示不可用;加上后精确落位。
- **控制条件**(Fig 17–20):去掉 depth 或 Canny 任一,均出现物体模糊与错误交互,严重降质。

## 四、局限性

1. **生成耗时是瓶颈**:视频扩散在 GPU 数少时算力开销大(作者提到 KV cache、模型蒸馏为后续加速方向)。
2. **物体泛化受限**:对**关节体(articulated)与可形变(deformable)物体**生成质量差(Fig 16 失败案例),表现为运动模糊/结构不一致,根因是训练分布缺此类物体。
3. **依赖多视角标定与 URDF**:整套流水线建立在已知相机内外参、URDF、可分段的"motion/skill"结构之上,迁移到未标定或长时序复杂任务的成本未评估。
4. **评测范围有限**:仅 4 个真机任务、单一 G1 人形本体;跨本体、跨场景的泛化未验证。生成数据的物理保真依赖深度控制,接触力/柔性交互等未建模。

## 五、评价与展望

**优点**:(1)真正打通了"3D 可编辑 ↔ 2D RGB 视频"这条链路——相比只能作用于点云的 DemoGen,它直接产出多视角 RGB,能训主流 2D VLA;相比 Real2Render2Real / RoboSplat 依赖稠密 3DGS 扫描,它只需前馈重建 + 视频扩散,更可扩展、无需数字资产。(2)以深度为主控信号 + Canny/ray map/action 多条件,是当前视频生成机器人数据方向里较扎实的可控性设计,消融证明每个条件都有贡献。(3)Metric-VGGT 的"真机管尺度、仿真管位姿"混合训练是个干净、可复用的工程贡献。(4)数字诚实:主表、规模化表、DP 复现、视频质量表相互印证,10–50× 的说法有据。

**缺点与开放问题**:(1)方法在 idea 层面是"DemoGen 的点云编辑 + GE-Sim 的可控视频生成"的组合,单个模块新意有限,创新集中在系统集成与 RGB 化改造。(2)重度依赖 CuRobo 运动规划与手工"motion/skill"分段,对接触密集、长程、双臂协同的复杂任务能否自动分段存疑。(3)生成数据训练出的策略性能上限被视频扩散模型的物理保真度锁死,关节体/形变体已暴露短板;而这恰是真实操作最需要泛化的品类。(4)与同期 R2RGen、RoboTransfer、MVAug、Egodemogen 等"一到多演示生成"工作高度同质化(Table 4 自比),差异化主要在"无仿真 + 纯 RGB + 支持新纹理/新轨迹"这几个勾选项的组合完备性上。

**可能改进方向**:引入接触/力条件或物理先验以提升交互保真;把 SOR 从刚体插值扩展到关节/形变物体的参数化编辑;用蒸馏/一致性模型压缩生成时延使之逼近在线数据引擎;以及验证生成数据在跨本体、真实分布漂移下的迁移收益。总体上,这是一篇工程完成度高、数字扎实、对"用世界模型/视频生成造具身操作数据"路线有实证价值的系统性工作。

## 参考

1. DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning(RSS 2025)——点云编辑式演示生成,本文轨迹分段思路来源。
2. VGGT: Visual Geometry Grounded Transformer(CVPR 2025)——前馈几何重建底座,本文微调为 Metric-VGGT。
3. Genie Envisioner / GE-Sim: A Unified World Foundation Platform for Robotic Manipulation(arXiv 2508.05635)——本文视频生成模块底座。
4. Real2Render2Real: Scaling Robot Data without Dynamics Simulation or Robot Hardware(CoRL 2025)——3DGS 真实到仿真的对照路线。
5. AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems(arXiv 2503.06669)——真机与 DigitalWorld 训练数据来源、Go-1 策略。
