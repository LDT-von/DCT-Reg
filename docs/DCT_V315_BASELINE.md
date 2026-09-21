# v3.15：NLL-only Residual Transport Interaction 基线

日期：2026-09-20。代码已实现、结构验证通过；真实癌种结果未建立。

## 1. 为什么不继续给 v3.14 加损失

问题不是“五项一定比一项差”，而是每个额外目标是否有不可替代的作用和独立证据。

| v3.14 部件/目标 | v3.15 取舍 | 理由与代价 |
|---|---|---|
| 患者 NLL | 保留唯一训练目标 | 直接拟合离散 hazard 和删失似然 |
| IPCW-rank 与跨批缓存 | 移除 | 排序目标不是似然的数学重复，但引入额外权重、KM 参考和缓存；先测单目标基线，可能损失 C-index 优化能力 |
| per-slot NLL / 每槽 hazard head | 移除 | 不再要求每个槽都独立解释同一患者全部结局；代价是不提供独立“每槽生存曲线” |
| diversity loss | 移除 | 不再靠另一项惩罚抵消逐槽监督；不意味着从此保证不塌缩 |
| masked/full reconstruction、decoder、mask token、warmup | 移除 | 不让在线表征重构和均值重建捷径干扰简洁基线；也失去可能有益的辅助正则化 |
| 10 次 GRU 槽更新、二次聚合 | 单次 attention pooling | 保留 token→slot 路由，不用迭代相互平均 |
| 4 阶段×3 几何 OT | 1 个余弦 OT | 少一个归因层次和多组求解；不再声称阶段特异几何 |
| evidence gate、event Transformer、风险锚点 | 移除 | 用直接交互读出替代复杂事件链路；不保留旧版干预审计合同 |

所以 v3.15 是一条**独立简洁基线**，不是与 v3.14 仅损失不同的严格消融。v3.10 仍是 catalog 的 primary method。

## 2. 整个结构只围绕一个 RTI 模块

1. WSI patch / 每个 pathway：一层投影、LayerNorm、GELU。
2. 每种模态单次槽池化，默认各 8 槽；query 初始正交但可学习，不把 query 向量直接注入槽值。
3. 槽均值构成基础患者表示；一个小 MLP 预测基础 hazard logits。
4. RTI 对槽差异求一个 OT 耦合，读取相对于独立配对的局部双线性交互，并加到基础 logits。

没有 GRU、重构 decoder、event Transformer、逐槽 hazard head、风险锚点或辅助生存标签分支。
forward 不使用 y、c、event_time；标签只在训练器中的唯一 NLL 出现。
OT epsilon 固定，不再有随 epoch 变化的隐式推理状态。

### 2.1 单次槽池化

query 与患者内中心化的 token key 计算归一化点积，乘 sqrt(D)/0.5 后在 token 维 softmax。
槽值是实际 token 值的加权平均。路由 key 去中心化，值不去中心化：共有患者信息不会在这里丢掉。
这不是容量平衡的 token routing，也不提供每个槽覆盖不同组织的硬保证。

### 2.2 为什么不能“搬运完直接平均”

若 balanced OT 的行列边际为 a、b，搬运后的槽为
\(\widetilde W_j=\sum_i T_{ij}W_i/b_j\)，则

\[
\sum_j b_j\widetilde W_j=\sum_i a_iW_i.
\]

也就是说，直接线性平均会把耦合结构消掉：模型看起来有 OT，最终读出的均值却与 T 无关。
v3.15 用双线性槽对交互读出，避免这个代数上的失效方式。这是设计动机，不是声称旧版的非线性 reader 也必然满足这条退化等式。

### 2.3 RTI 的完整公式

每种模态在患者内部的槽维去均值，并逐特征标准化：

\[
\mu_m=K_m^{-1}\sum_i S^m_i,\quad
R^m_{id}=\frac{S^m_{id}-\mu_{md}}
 {\sqrt{K_m^{-1}\sum_j(S^m_{jd}-\mu_{md})^2+10^{-4}}}.
\]

余弦成本与 balanced entropic OT：

\[
C_{ij}=1-\cos(R^w_i,R^o_j),\quad
T=\operatorname{Sinkhorn}(C; a_i=1/K_w,b_j=1/K_o,\epsilon=0.2).
\]

只读取独立配对基线以上的交互：

\[
v_d=\sum_{ij}(T_{ij}-a_i b_j)R^w_{id}R^o_{jd},\quad
\ell=f_{global}(\mu_w,\mu_o,availability)+Wv.
\]

W 无 bias。T 是非负耦合，T−abᵀ 是有正有负的交互对照权重，**不是另一个运输概率矩阵**。
由于 R 在槽维中心化，独立项在精确运算中本来就为零；差分写法明确零基线，不是额外损失或新的 OT 理论。
这个向量相当于耦合下的逐特征互相关统计，不构造 D×D 全协方差矩阵。

基础 MLP 本身可以学习全局跨模态关系；RTI 提供的是额外的**局部槽对关系**，不能声称模型的全部跨模态作用只来自 RTI。

### 2.4 可验证的解释边界

每个槽对、每个时间区间都有贡献：

\[
\Delta\ell_{ijc}=(T_{ij}-a_i b_j)\sum_d W_{cd}R^w_{id}R^o_{jd}.
\]

加和严格恢复 RTI logits。解释的是 logits 增量，不是经过 sigmoid / 累计生存乘积后的风险分数可加和分解。
独立耦合时贡献为零；任一模态全部槽内容相同时交互为零；缺失任一模态时显式关闭交互。
这些是代数/实现性质，不是临床因果解释或真实生物学对齐证明。

## 3. 只有一个训练损失

\[
L=\frac1B\sum_{b=1}^{B}L_{NLL}(\ell_b,y_b,c_b).
\]

参考 YAML 设置 alpha_surv=0，删失项不再额外乘 0.5。主损失用 logits/logsigmoid 稳定计算，auxiliary 返回严格为零；没有任何辅助损失系数需要配平。
仍保留普通 AdamW weight decay、dropout 和梯度裁剪，它们不是新添的多任务目标。
共享 CLI 的 alpha 默认仍为 0.5；请使用本配置或显式设置 `--alpha_surv 0`。若做跨版本公平比较，必须匹配 alpha、slots、encoder 容量、seed、split 和训练预算。

## 4. 实现隔离与接口

- 独立目录：survot_rank/research/methods/legacy/experimental/dct_v315_residual_transport。
- 模型直接继承 nn.Module，不导入任何旧版模型，甚至不依赖共享 encoder 实现。
- 只有 model.py、losses.py、__init__.py；仅使用 PyTorch 和标准库。
- 稳定 NLL 公式来自本仓库既有审计，不把通用累计生存似然声称为新损失。
- 注册名：dct_v315_residual_transport；别名 dct_v315；status=candidate。
- 标准训练器的模型创建、损失选择、两批次训练、评估适配、checkpoint 已测试。
- checkpoint 校验模式、OT 参数、alpha、dropout 等 recipe；不同研究设置不静默混载。
- 缺失模态仍需形状正确的占位 tensor 和显式 availability 标记；可忽略不可用位置的 NaN，但不能据此宣称缺失模态泛化已验证。
- 旧 DCT 的风险锚点/代价干预专用审计脚本不适用于本模型，应使用新的加和交互解释接口。

## 5. 当前验证证据

### 5.1 结构与反传

v3.15 专项测试包含 NLL 手算、极端 logits、OT 边际、耦合及成本梯度、每个参数的 NLL 梯度、独立配对归零、真实 OT 增量非零、排列不变性、共有偏置不制造交互、常量 token、单患者/单 pathway/单槽、NaN/缺失模态、checkpoint 和实际训练入口。
最终 v3.15 专项 **33 passed**；与 v3.10/v3.13/v3.14/hierarchical slots 合并回归 **101 passed**。
doctor、编译和 diff 检查通过。全仓既有的 4 个历史测试导入问题未在本轮改动，不能把相关回归称为全仓全部通过。
GPU 测试覆盖 1024→256 维、4096 patch 的 FP32 与 FP16 autocast。

合成诊断：CPU OT、CPU independent、GPU OT，各 3 个种子×30 步，总 270 步；均为有限损失/梯度。
三份原始记录附有源码 SHA-256，已与当前模块比对一致：

- audit_results/v315_cpu.json
- audit_results/v315_independent.json
- audit_results/v315_gpu.json

### 5.2 规模与槽行为

在相同的 1024 输入、256 隐层、8+8 槽、12 个 pathway（每个 16 维）、4 个时间区间下：

| 模型 | 参数量 |
|---|---:|
| v3.14（10 次槽迭代、2 层 event encoder） | 6,257,938 |
| v3.15 | 460,296 |

减少约 92.6%。这是整个结构删减的结果，不能全算在 RTI 一个公式上；真实 pathway 数量/宽度不同会改变绝对参数量。

GPU 合成批次 B=8、4096 patch、3 种子、30 步后的槽内容与路由：

| seed | WSI 槽 cosine | WSI 槽标准差 | WSI attention overlap | 组学槽 cosine | 组学槽标准差 |
|---|---:|---:|---:|---:|---:|
| 3 | 0.6541 | 0.2205 | 0.0038 | 0.4748 | 0.3183 |
| 17 | 0.8108 | 0.1560 | 0.0088 | 0.4843 | 0.3119 |
| 29 | 0.8173 | 0.1624 | 0.0123 | 0.4976 | 0.3069 |

OT 边际最大误差约 1.49e−8，RTI 的 logit RMS 非零；independent 对照中交互及增量 logits 全为零。
GPU 本地诊断记录的峰值 allocated memory 约 568 MiB，单步中位时间约 0.088–0.117 秒；这是合成缩小 pathway 设置、单卡环境，不是完整真实配置的资源承诺。

**不能据此声称优于 v3.14 或达到顶级性能。** 合成小批数据被反复训练，训练 NLL 很快接近零，反而提醒需要严格防范真实小样本过拟合。
此前 v3.14 报告常用 16+16 槽、不同 encoder 和 alpha；其数值不能直接作为 v3.15 的公平性能比较。
多样性未由损失或定理保证：query 仍可靠近、encoder 可丢失差异、全局支路可能忽略 RTI、或 RTI 只学到边际分布相似性。真实数据仍需检查。

## 6. 最小真实实验，不再扩成大量损失组合

先只做两个主臂，使用同样 splits / seeds / 参数初始化、训练预算和全局支路：

1. RTI-OT：当前基线。
2. Independent：把 T 设成 abᵀ，交互严格归零；这就是同参数结构的全局-only 控制。

然后检查保留训练模型、只在评估时切换耦合是否改变预测。这能验证依赖性，但不能代替重新训练的消融。
仅在真实验证集稳定显示 NLL-only 排序不足时，再比较一个 NLL+rank 版本；现在不把 rank、diversity、reconstruction 默认加回。
要证明 RTI 而不是轻量 encoder 有效，除同骨干 independent 控制外，还应报告容量/优化预算匹配的旧方法对照。
保存每折预测与曲线，内部验证调参，独立外层测试只做最后评估，不用训练 NLL 或最佳验证分数冒充外测性能。

## 7. 运行入口

在 v313-implementation 仓库目录运行，先替换真实服务器数据路径：

```powershell
python -m survot_rank.cli train --config configs/dct_v315_blca_uni.yaml
python -m survot_rank.cli train --config configs/dct_v315_blca_uni.yaml --set dct_v315_transport_mode=independent --set specific_simple=dct_v315_independent --set results_dir=/data1/DCT-Reg/results/dct_v315_independent/blca
python -m pytest tests/test_dct_v315_residual_transport.py -q
python -m scripts.audit_v315 --output audit_results/v315_cpu.json --steps 30
python -m scripts.audit_v315 --output audit_results/v315_gpu.json --steps 30 --device cuda --encoding-dim 1024 --dim 256 --patches 4096 --batch-size 8
```

对照必须使用不同 results_dir，避免覆盖已有运行。本轮未启动真实五折，未提交或推送。

## 8. “顶级模块”与先验边界

目标是简单、机制有效、可反驳且实验能支撑的模块，不是给普通操作换名字。
本实现是提出并落地了一个项目内候选；尚未证明广泛原创性、跨癌种优势或统计显著提升。
已有 Slot Attention 与病理/组学 OT 先例，必须在写论文时区分已有积木和本项目的具体简化及验证：

- [Object-Centric Learning with Slot Attention, NeurIPS 2020](https://proceedings.nips.cc/paper_files/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
- [MOTCat, ICCV 2023 官方实现](https://github.com/Innse/MOTCat)

中心化双线性交互/独立耦合对照也不是本报告声称首次提出的数学概念；本次先验检索不构成穷尽性的 novelty 审查。
