# Cosmos Policy Sim2Real：从合成先验实现世界-动作模型的高效 Sim-to-Real 迁移

> **论文**：*Efficient Sim-to-Real Transfer of World-Action Models from Synthetic Priors*
>
> **作者**：Zixing Wang、Kausik Sivakumar、Jinghuan Shang、Yafei Hu、Zhaoming Xie、Ran Gong、Xiaohan Zhang、Karl Schmeckpeper（后三位为共同指导 Equal Advising）
>
> **机构**：Purdue University；Robotics and AI Institute
>
> **发布时间**：2026 年 06 月（arXiv 2606.31101）
>
> **发表状态**：未录用（预印本；作者在正文明确注明本文是即将发布的完整版"官方世界-动作模型零样本 sim-to-real 迁移工作"的早期阶段结果）
>
> 🔗 [arXiv](https://arxiv.org/abs/2606.31101) | [PDF](https://arxiv.org/pdf/2606.31101)
>
> **分类标签**：`world-action-model` `sim-to-real` `video-diffusion-policy` `domain-randomization`

---

## 一句话总结

将视频扩散世界-动作模型 Cosmos Policy 与 Isaac Gym 域随机化仿真、AnyTask 自动化演示生成流水线相结合，仅用约 800 条/任务的纯合成演示、零真实演示，在 Franka Research 3 上对四项操作任务零样本部署，平均成功率 35%，据作者所知是首个成功跨越 sim-to-real 鸿沟的世界-动作模型。

## 一、问题与动机

真实机器人示教数据采集成本高昂，仿真提供了可并行、可扩展的替代方案，但视觉与物理层面的 sim-to-real gap 常导致迁移失败。世界-动作模型（world-action models：联合预测未来视觉观测与机器人动作的统一生成模型）此前已在 sim-to-sim 或 real-to-real 场景中展示出潜力，但是否能够跨越仿真到真实这一更难的鸿沟尚不明确。这一空白很关键：一旦被验证可行，就意味着 sim-to-real 学习"用廉价、可扩展、自动生成的数据取代真实遥操作数据"这一核心优势，可以被扩展到世界-动作模型这一新的策略族上。

作者特别声明本文的定位：目标不是提供一个受控消融来证明"联合视频-动作预测对 sim-to-real 是必要的"，而是回答一个更简单的存在性问题——完全用合成演示训练的世界-动作模型，能否零样本完成真实世界操作？

## 二、核心方法

**底座模型**：复用 Cosmos Policy（Kim et al., ICLR 2026）——将预训练视频扩散模型 Cosmos-Predict2 通过单阶段后训练（single-stage post-training）改造为机器人策略，动作、未来观测、价值估计均被编码为统一扩散过程中的潜在帧（latent frames）。本文没有改动 Cosmos Policy 的模型结构，贡献在于验证其 sim-to-real 迁移能力。

**仿真环境与域随机化**：基于 GPU 加速仿真（Isaac Gym），覆盖四类随机化：(1) 纹理——物体表面、桌面材质、背景取自程序化生成库与真实图片库；(2) 相机位姿——腕部相机与第三人称相机的平移/旋转扰动；(3) 光照——灯光数量、位置、强度、色温的变化；(4) 物体摆放——在可达工作空间内均匀采样。目标是让随机化训练分布尽可能覆盖真实测试场景的外观分布，而不需要手工复刻真实测试场景。

**演示生成**：借助 AnyTask 框架（Gong et al., arXiv:2512.17853）通过基础模型引导的运动规划（ViPR）自动生成专家演示，全程无人工遥操作参与。覆盖四个任务——lift banana、lift brick、open drawer、put strawberry into bowl——每任务约 800 条演示（总计约 3,200 条），每条包含 RGB 观测与末端执行器动作轨迹的配对数据。

**训练配置**：Cosmos Policy 在 40 块 H100 GPU 上训练 32 个 epoch，耗时约 72 小时。

**定性验证**：论文用"put strawberry into bowl"任务的 approach / grasp / lift 三个阶段做前向预测检查——模型预测的未来帧（Pred Cam）与真实同步相机画面（Live Cam）在物体位置、机械臂姿态、任务阶段上高度对齐，说明尽管训练数据全部来自仿真，模型仍学到了对真实场景动力学有用的表征。

## 三、关键结果

论文在 Franka Research 3 上做零样本部署（不采集任何真实演示、不做真实数据微调），仅将标准 RGB 相机流（腕部+第三人称）接入策略推理。

**表 1：真实机器人成功率（每任务 10 次试验）**

| 方法 | Banana | Brick | Drawer | Strawberry | 平均 |
|---|---|---|---|---|---|
| DP w/ 10 条真实演示 | 0/10 | 0/10 | 2/10 | 0/10 | 5% |
| DP w/ 50 条真实演示 | 4/10 | 3/10 | 3/10 | 0/10 | 25% |
| **Ours（800 条仿真演示，0 条真实）** | **5/10** | **5/10** | 2/10 | **2/10** | **35%** |

- 完全用合成数据训练、零真实演示的世界-动作模型平均成功率 35%，高于用 50 条真实演示训练的 Diffusion Policy（DP）基线（25%），远高于仅用 10 条真实演示的 DP 基线（5%）。
- 作者明确提醒：DP 基线并非架构匹配的受控消融——它们用了真实数据，回答的是不同的问题；这一对比只是提供"sim-to-real 方法试图规避的真实数据成本"的一个实用参照，不能直接得出"世界-动作模型架构上优于 Diffusion Policy"的结论。
- Drawer 开门任务上该方法没有体现优势（2/10，与 10-demo DP 持平，弱于 50-demo DP 的 3/10），说明迁移增益并非在所有任务上一致。
- **OOD 泛化**：额外测试抓取训练任务集之外的新物体（瓶子），策略仍能完成 approach-grasp-lift 全流程，作者认为这提示模型习得了可泛化的视觉运动基元，而非单纯记忆训练物体。
- 作者将迁移成功归因于三个因素的组合——预训练视频先验、广泛域随机化、联合动作-视频预测目标——但明确说明本文未做受控消融拆解各因素的独立贡献，留待未来工作。

## 四、评价与展望

**优点**：这是已知首次把世界-动作模型（而非仅是 Diffusion Policy 一类的行为克隆式策略）成功从仿真零样本迁移到真实机器人操作的报告，把 sim-to-real 学习的核心优势——用廉价可扩展的合成数据替代真实演示——扩展到了这一新的策略族上。数据效率突出：约 800 条/任务合成演示全部由 AnyTask + ViPR 运动规划自动生成，无需任何人工遥操作，对真实数据采集成本敏感的应用场景有直接参考价值。定性可视化（预测帧与真实相机帧的阶段对齐）为"模型学到了跨越仿真-真实鸿沟的场景动力学表征"提供了直观证据，OOD 物体抓取的初步结果也支持"泛化视觉运动基元"这一假设。

**局限与开放问题**：论文本身是一篇 3 页的"早期结果"报告，作者原话说明这只是即将发布的完整版工作的一部分，实验规模有限——仅 4 个任务、单一机械臂（Franka Research 3）、每任务仅 10 次试验，统计功效较弱，结论稳健性有待完整版验证。最大的方法论空白是完全没有做消融：预训练视频先验、域随机化范围、联合视频-动作预测目标三者各自对迁移成功贡献多少，论文自己承认未拆解，这直接限制了对结果可解释性和可复现性的判断。DP 基线的非架构匹配对比意味着 35% 对 25% 的差距不能归因于架构优势，只能读作成本参照，这也说明本文尚未真正回答"联合视频生成式预测本身是否是 sim-to-real 成功的必要条件"这一更有价值的问题——而这恰恰是作者在开篇就主动排除、留给未来工作的核心问题。Drawer 任务上无优势提示该方法的收益可能与任务的接触/关节复杂度相关，但论文未展开失败模式分析。此外，40×H100×72 小时的后训练成本不低，标题强调"Efficient"，但论文没有给出与传统真实数据采集/训练路径的直接总成本对比，"高效"更多体现在免除真实机器人示教这一维度而非端到端计算效率。

**与相关工作的关系**：方法直接构建在 Cosmos Policy（Kim et al., ICLR 2026）与 AnyTask（Gong et al., 2025）之上，是二者在 sim-to-real 场景下的一次组合验证，而非提出新架构或新训练目标，其贡献本质是一次存在性证明（existence proof）而非机制性洞察。这与 Du et al.（2024）等世界模型/视频生成策略工作在 sim-to-sim、real-to-real 场景下已有的正向结果相互补充，本文把验证场景明确扩展到了 sim-to-real 这一此前空白的方向。开放问题包括：该方法能否扩展到更多任务和更多本体（embodiment）、能否量化域随机化各因子的边际贡献、能否像常见 sim-to-real 实践那样结合少量真实数据做微调进一步提升成功率（本文完全未探索），以及作者预告的完整版工作是否会提供更严谨的消融实验和更大规模评测。

## 参考

1. Kim et al. Cosmos Policy: Fine-tuning video models for visuomotor control and planning. ICLR, 2026.
2. Gong et al. AnyTask: an automated task and data generation framework for advancing sim-to-real policy learning. arXiv:2512.17853, 2025.
3. Makoviychuk et al. Isaac Gym: High performance GPU-based physics simulation for robot learning. arXiv:2108.10470, 2021.
4. Tobin et al. Domain randomization for transferring deep neural networks from simulation to the real world. IROS, 2017.
5. Du et al. Learning universal policies via text-guided video generation. NeurIPS, 2024.
