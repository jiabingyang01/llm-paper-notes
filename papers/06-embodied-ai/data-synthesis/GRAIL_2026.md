# GRAIL：从三维资产与视频先验生成人形全身操控数据

> **论文**：*GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors*
>
> **作者**：Tianyi Xie, Haotian Zhang, Jinhyung Park, Zi Wang（共一）, Bowen Wen, Jiefeng Li, Xueting Li, Qingwei Ben, Haoyang Weng, Yufei Ye, David Minor, Tingwu Wang, Chenfanfu Jiang, Sanja Fidler, Jan Kautz, Linxi Fan, Yuke Zhu, Zhengyi Luo, Umar Iqbal, Ye Yuan（后三位为 Project Leads）
>
> **机构**：NVIDIA、UCLA
>
> **发布时间**：2026 年 06 月（arXiv 2606.05160）
>
> **发表状态**：未录用（预印本）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.05160) | [PDF](https://arxiv.org/pdf/2606.05160)
>
> **分类标签**：`人形机器人` `数据合成` `视频生成先验` `全身操控` `4D人物交互重建` `sim-to-real`

---

## 一句话总结

GRAIL 是一条"直到真机部署前完全保持虚拟"的人形 loco-manipulation 数据生成流水线：先用已知的 3D 场景 / 相机 / 物体几何驱动视频基础模型合成人-物交互参考视频,再利用这些已知量做交互感知的 4D HOI（human-object interaction）重建,产出超过 20,000 条可直接 retarget 到人形机器人的轨迹,仅用生成数据训练的 egocentric 视觉策略在 Unitree G1 上实现 84% 物体抓取成功率与 90% 爬楼梯成功率。

## 一、问题与动机

训练人形 loco-manipulation 策略需要跨多样物体、全身动作和场景几何的 robot-compatible 演示数据,但两条主流数据来源都难以规模化：teleoperation 和 motion capture 每采一批新物体 / 新地形都要重新布置物理场景、依赖专业演员和机器人操作;而从 in-the-wild 视频重建 4D robot-ready 轨迹虽然视觉覆盖面广,却要从模糊的单目观测中同时反推相机参数、尺度、几何、人体形状、接触和世界坐标运动,深度歧义与人形-机器人形态失配问题突出。

GRAIL 的核心 insight 是反转问题设定：与其把相机、尺度、几何、人体形态全部当作待重建的未知量去求解一个欠约束的逆问题,不如先把这些量在 3D 资产层面**固定为已知**,再用视频生成先验（VFM）合成多样交互,最后在这些"特权"已知量的条件下做重建。这样将病态的重建问题转化为条件更好的恢复问题,显著降低深度歧义和形态失配。

## 二、核心方法

流水线分三阶段：(1) 机器人中心的人体视频生成;(2) 交互感知的 4D HOI 重建;(3) 任务通用的 loco-manipulation tracker 训练,并进一步蒸馏为 egocentric 视觉策略部署到真机。

**阶段一：Robot-Centric Human Video Generation。** 用 Infigen 构建候选环境,用刚体仿真将物体安放至稳定、无碰撞的初始位姿,并把一个已 prefit 到目标机器人体型的人体资产放置在旁边,用 Blender 以已知相机内参 $C_K$、外参 $C_E$ 渲染首帧。一个 VLM（GPT-4）根据渲染帧生成交互文本 prompt,视频基础模型（Kling）在静态相机设定下合成参考交互视频 $\{I_t\}_{t=1}^T$,同时保留相机参数供后续重建复用。

**阶段二：Interaction-Aware HOI Reconstruction。** 先独立做初始估计：人体用 GENMO 逐帧估计 SMPL-X 姿态参数（体型固定为 prefit 形态,不重新估计）,手部用 WiLoR 逐帧估计 MANO 参数（遮挡处线性插值 + Savitzky-Golay 滤波去抖,再通过手腕 IK 对齐进 SMPL-X）;物体用在自建合成数据上微调过的 FoundationPose（深度通道置零以适配 RGB-only 设置）做 6-DoF 跟踪,并用 SAM2 分割掩码校验、剔除几何不一致的片段。

随后对残差运动参数 $\{\Delta\Theta_t^{\mathcal H}\}$、$\{\Delta\Theta_t^{\mathcal O}\}$（6D 旋转参数化）做全局联合优化,目标函数为

$$L=\lambda_{kp}L_{kp}+\lambda_{proj}L_{proj}+\lambda_{depth}L_{depth}+\lambda_{cont}L_{cont}+\lambda_{reg}L_{reg}$$

用大白话说：五项损失分别管"图像上像不像""物体投影对不对齐""深度尺度准不准""接触部位贴不贴合""运动是否平滑防抖"。其中：

- $L_{kp}$ 是 2D 关键点重投影误差,让优化后的人体姿态投影回图像后与检测到的身体 / 手部关键点对齐;
- $L_{proj}$ 保持 FoundationPose 给出的 image-aligned 物体位姿不被优化破坏;
- $L_{depth}=\frac{1}{T}\sum_t \mathcal{CD}(V_t^{\mathcal H,vis},\mathbf P_t^{\mathcal H})+\mathcal{CD}(V_t^{\mathcal O,vis},\mathbf P_t^{\mathcal O})$,先用 MoGe-2 估计深度并对齐到渲染场景的 ground-truth 背景深度获得 metric-scale 深度,再用 SAM2 分割人 / 物区域反投影成点云,用双向 Chamfer 距离把 mesh 顶点拉向点云——大白话说：靠场景已知深度把单目深度的尺度"钉死",这是解决深度歧义的关键一步;
- $L_{cont}$ 只在检测到接触的帧上、且只对投影落入接触区域的物体顶点施加**仅深度方向**的 Chamfer 惩罚（因为图像空间对齐已由 $L_{kp}$、$L_{proj}$ 保证,接触损失只需再管深度方向贴不贴合）;
- $L_{reg}=L_{foot}+L_{vel}+L_{smooth}$ 分别抑制脚部滑动、约束骨盆速度匹配 GENMO 的全局估计、惩罚一二阶时间差分以保证时序平滑。

**阶段三：Task-General Loco-Manipulation Tracking。** 用 GMR 把重建的 SMPL-X 动作 retarget 到 Unitree G1,在预训练全身控制器 SONIC 基础上训练两种互补、按任务族共享（而非逐序列 / 逐物体单独拟合）的 tracker：

- **Object-aware adaptor** $\pi_\phi$：冻结 SONIC 的 encoder / quantizer / decoder,只训练一个 adaptor 策略,输入本体感知 $s_t$ 与物体参考 $o_t$（机器人体坐标系下的物体位姿、手-物变换、指尖接触力、basis point set 形状编码、delta 观测）,输出 64 维 latent 残差 $\Delta z_t$ 和映射到 7 个手指 DoF 的二值抓握 primitive：$(\Delta z_t, a_t^{hand})=\pi_\phi(s_t,o_t)$，$a_t^{body}=\mathcal G(z_t+\lambda\Delta z_t)$，$\lambda=0.1$。
- **Scene-aware tracker**：用局部 height map（2D 卷积编码器 $\epsilon_h$）微调整个 controller,并训练一个并行的 kinematic decoder 提供辅助 MSE loss 稳定训练,用于上下台阶、坐椅子等地形交互。

两者共享 motion-tracking reward $R_t^{motion}=\sum_i w_i\exp(-\|\tilde x_{i,t}-x_{i,t}\|^2/\sigma_i^2)$（大白话：模拟状态越贴近参考轨迹,高斯核奖励越高）;object-aware tracking 额外加物体跟踪 reward 和接触门控的抓握 reward（接触时长 + 抓握姿态余弦 + 指尖-接触中心距离三项）。训练用 PPO,Isaac Lab,64 张 NVIDIA L40,每 GPU 1024 环境,30,000 iterations。

**Sim-to-Real。** 将 object-aware 和 scene-aware tracking policy 分别蒸馏为两个 egocentric 视觉策略（用于 pick-up 和 stair-climbing）,输入头部相机 RGB,输出 SONIC controller 的 latent token,配合 domain randomization 迁移;真机侧用 RTX 5090 桌面机 + Luxonis OAK-D W 相机,10 Hz 推理,部署到 Unitree G1。

## 三、关键结果

**HOI 生成质量**（Table 1,20 个来自 ComAsset 的共享评测物体,对比 training-based CHOIS / HOIDiff 和 training-free DAViD,用 InterMimic 做物理可执行性验证）：

| 方法 | Contact ↓ | Pen. ↓ | Inter. Score ↑ | Human Smo. ↓ | Obj Smo. ↓ | SR ↑ | Body Dev. ↓ | Obj Dev. ↓ |
|---|---|---|---|---|---|---|---|---|
| HOIDiff | 0.012 | 2.07% | 1.79 | 0.0043 | 0.0118 | 15.8% | 0.2120 | 0.3352 |
| CHOIS | 0.034 | 3.74% | 2.47 | 0.0055 | 0.0062 | 10.5% | 0.2564 | 0.3642 |
| DAViD | 0.246 | 1.46% | 2.74 | 0.0024 | 0.0605 | 24.0% | 0.4723 | 0.5826 |
| **GRAIL** | **0.008** | **0.90%** | **3.58** | 0.0033 | **0.0022** | **88.9%** | **0.0913** | **0.0851** |

GRAIL 的物理可执行 SR（88.9%）大幅超过次优基线 DAViD（24.0%）。

**任务通用 tracker**（Table 2,对比 HDMI 和 ResMimic,124 动作、43 物体基准）：GRAIL SR 81.4% 对比 HDMI 48.5%、ResMimic 49.2%;ObjPos 误差 0.135 对比 0.283 / 0.393。Ablation 显示：去掉 SONIC 预训练从头训练 SR 降到 45.0%;去掉 adaptor $\pi_\phi$（纯 SONIC body imitation）MPJPE-L 反而最优（37.1）但 SR 最低（39.7%）,说明单纯体态跟踪精确不代表能完成操控任务;把相对物体观测换成绝对观测 SR 降到 57.9%。

**数据规模**：1,000 个物体资产（来自 Robocasa、ComAsset、OMOMO、Hunyuan3D）,1,000 个程序化生成地形配置,共生成超过 20,000 条序列,覆盖 pick-up、whole-body manipulation、sitting、terrain traversal 四大类。

**真机部署**（Table 3,10 trials / 物体）：Seen objects（Cube/Apple/Tea Box/Carrot/Wet Wipes）平均成功率 84%（100/60/100/70/90%）;Unseen objects（Spray Can/Lint Roller/Peach/Flashlight/Medicine Bottle）平均 80%（100/50/90/80/80%）。爬楼梯真机成功率 90%。

## 四、评价与展望

**优点：** GRAIL 的方法论转折点在于把"从任意视频重建 4D 交互"这一病态逆问题,转化为"先固定 3D 场景 / 相机 / 尺度 / 形态,再用生成视频合成动作 + 在已知量条件下做优化"的良态问题,这是对 DAViD、ZeroHSI 一类 VFM-as-prior 方法的工程化改造,在 Table 1 的四个维度上全面超过 training-free 和 training-based 基线。用 adaptor + height-map 两种轻量修改复用同一个预训练全身 controller（SONIC）而非逐序列 / 逐物体单独拟合,是很实际的规模化选择;ablation 揭示的"体态跟踪精度与操控成功率之间的权衡"（去掉 adaptor 后 MPJPE-L 更低但 SR 更低）对后续 HOI tracking 的评价指标设计有参考价值。论文提供了完整的 sim-to-real 闭环验证（真实 Unitree G1 上 84%/90% 的数字,并验证了对 unseen 物体 80% 的泛化）,比许多只报告仿真内指标的数据生成工作更具说服力。

**局限：** 作者在 Limitations 中明确指出：流水线依赖 3D 物体资产、可仿真的场景搭建,以及 VFM 必须"听话"地生成符合 prompt 的交互视频;在严重遮挡、快速运动或物体外观不一致时重建质量下降,failure-filtering 会丢弃"non-trivial fraction"的序列,但论文未报告具体丢弃比例,数据产出效率不透明。手部抓握被简化为映射到预定义抓握配置的二值 open/close primitive,而非连续精细的手指控制,限制了对插拔、旋钮等精细操控任务的覆盖。Task-general tracker 在 motion family 发生较大变化时仍需重新训练或微调,并非真正 zero-shot 的通用 controller。真机验证只覆盖 pick-up 和 stair-climbing 两类任务（且环境道具相对固定）,whole-body manipulation、sitting、slope 等虽在仿真中展示但未做 sim-to-real 验证,真实世界泛化的证据链尚不完整。此外流水线串联了 VFM（Kling）与多个感知模块（GENMO/WiLoR/FoundationPose/MoGe-2/SAM2）,累计误差和对特定模型版本的依赖是复现与迁移到其他 VFM 时的潜在风险。

**与其他工作的关系与开放问题：** 相较 CHOIS、HOIDiff 这类直接从语言 / affordance 学习 HOI 生成的方法,GRAIL 选择了"具身条件生成 + 已知 3D 量重建"的路线;相较同样用 VFM 做 4D HOI 恢复先验的 training-free 方法 DAViD,GRAIL 的差异化贡献在于把相机 / 尺度 / 几何 / 形态全部前置为已知输入而非重建目标,这一"特权设置"框架具有一定可推广性,理论上可迁移到其他需要"video prior + 已知 3D 上下文"的 4D 重建任务。开放问题包括：如何量化并提升 VFM 生成交互视频的"良率"以降低数据成本;如何在不牺牲 task-general tracker 可扩展性的前提下把二值抓握 primitive 升级为连续力控的灵巧手操作;完全虚拟的生成流水线在真实世界物理属性（摩擦、材质、质量分布）上是否存在系统性 sim-to-real gap,论文尚未与 teleop / mocap 数据训练的策略做同基准对比。

## 参考

- Kim et al., 2025, *DAViD*（training-free VFM-based 4D HOI 重建基线）
- Li et al., 2024, *CHOIS*（training-based HOI 生成基线）
- Peng et al., 2025, *HOIDiff*（HOI 生成基线）
- Weng et al., 2025, *HDMI*（loco-manipulation tracking 基线）
- Zhao et al., 2025, *ResMimic*（loco-manipulation tracking 基线）
- Luo et al., 2025, *SONIC*（预训练全身控制器,GRAIL tracker 的骨干）
