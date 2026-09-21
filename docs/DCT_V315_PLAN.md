# v3.15：先减法，再验证核心模块

日期：2026-09-20。状态：实验基线计划，不替换 v3.10 主方法。

实施更新：独立模型、配置、训练接入、专项测试及三种子高维合成检查已完成。实际结果和边界见 DCT_V315_BASELINE.md；真实五折未执行。

## 判断

v3.14 的 NLL、rank、per-slot NLL、diversity、重构并非天然互补：每槽复制患者结局与槽专门化存在拉扯，在线 encoder 重构可学均值，额外训练分支和超参数使归因困难。v3.15 不再继续叠加损失。

## 设计与取舍

- 唯一训练目标：患者级离散生存 NLL；辅助返回值严格为零。参考配置 alpha_surv=0，使用未额外下调删失项的似然；与旧版对照时必须匹配 alpha。
- WSI/各 pathway 小型投影 → 单次 attention 槽池化（默认各 8 槽），不使用 GRU 反复更新或二次软聚合。
- 槽平均内容用于基础 hazard 预测；在交互支路中去掉患者共有内容并按槽方差标准化。
- 只解一个余弦成本的 balanced OT，不设置阶段、三种几何、evidence gate、event transformer、风险锚点或 decoder。
- 核心 RTI（Residual Transport Interaction）：对比实际耦合 T 与独立耦合 ab^T，只读超出独立配对的双线性交互。

\[
v_d=\sum_{ij}(T_{ij}-a_i b_j)R^w_{id}R^o_{jd},\quad
\ell=f_{global}(\mu_w,\mu_o,availability)+Wv.
\]

交互头无 bias，因而独立配对时增量严格为零；可以输出逐槽对、逐时间区间的 logit 加和分解。它不是因果解释，也不证明对齐具有真实生物学语义。

## 实施顺序

1. 写独立 model.py/losses.py，不继承/导入 v3.14 或任何旧模型。
2. 注册方法、接通标准训练器 NLL 和配置；保留旧版工作区改动。
3. 测试 NLL、边际、反传、缺失输入、无 label 输入依赖、排列性质、checkpoint 与 GPU。
4. 检查关键代数性质：独立耦合归零、无槽差异归零、加和分解一致、改变 T 确实影响预测。
5. 做同维度参数量与合成训练检查，报告剩余退化风险；不把结构测试称为 C-index 收益。

## 先验边界

Slot Attention 与病理/组学 OT 都有先例。此次简化是本项目中的结构候选，不声称这些积木或中心化交互公式首次提出。
参考：[Slot Attention, NeurIPS 2020](https://proceedings.nips.cc/paper_files/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)、[MOTCat 官方实现, ICCV 2023](https://github.com/Innse/MOTCat)。
单次 pooling 和去除冲突监督不构成不塌缩定理；必须监控槽差异幅度、attention overlap、交互幅度与验证集表现。
