# DCT v3.14 全面排查与修复报告

日期：2026-09-20。工作目录：`E:\DCT-v3.11-main\v313-implementation`。

## 1. 结论与适用范围

发现了真实的前向信息通道、反向数值稳定性、槽路由及 checkpoint 问题，已在 v3.14 内修复。
当前是**可继续做实验的候选实现，不是已经证明最优或不会塌缩的最终方法**。

这轮改动不只是几个损失的开关：槽路由、阶段编码、稳定 NLL 和保存重载也改变了。
后续和 v3.10 / v3.11 / v3.13 的性能比较，需要区分结构变化与损失变化。
旧版本模型没有修改，也没有提交或推送。仓库顶层的独立 v3.11 重写版本不在本次修改范围。

## 2. 发现的问题及处理

| 位置 | 问题或设计风险 | 本次处理 |
|---|---|---|
| 版本依赖 | 上一版仍继承 v3.10，不利于单独抽离 | 本目录自有 backbone/model/losses，无旧版本模型导入或继承；只复用通用 SNN_Block、WSI_Mlp |
| 槽路由 | 反复槽更新后再软语义聚合，会进一步平均槽内容；合成复测出现 cosine 接近 1 | 每次迭代保持不同原型 query，直接汇聚患者 token；去掉第二次混合式聚合 |
| 高维注意力 | 单位向量点积随维度增加缩小，固定温度使 D=256 路由接近平均 | 注意力分数使用维度缩放 `sqrt(D)/0.5`；补 4096 patch、1024→256 维 GPU 检查 |
| diversity | 同一患者 per-slot NLL 要求共同结局，而原 hazard 方差目标又要求风险分离；近退化时梯度很弱 | 改为归一化内容的槽对距离 hinge，不强制 hazard 不同；近重合与精确重合分别讨论 |
| per-slot NLL | 删失似然之外再给删失样本乘 IPCW，会使尾部样本被额外放大，并改变原目标 | 去掉这层额外权重；主/槽 NLL 使用一致的 alpha，IPCW 留在风险排序中 |
| NLL 数值 | sigmoid 后截断再取 log，在极端错误预测时可能截断纠错梯度 | 主损失和槽损失都改为 logits + logsigmoid 的累计生存似然；主损失求和后只除一次 B |
| mask 信息通道 | 完整组学产生的 factual gate 即使 detach，仍可携带隐藏 token 信息 | 辅助分支不再读取 factual gate，重新计算 masked OT，阶段均匀平均；同时检查数值不变性和零梯度 |
| reconstruction target | 目标含训练态 AlphaDropout 噪声 | 目标分支临时关闭 dropout、停止梯度，随后恢复训练状态；目标仍是当前 encoder，不是 EMA teacher |
| 阶段编码 | 所有特征方向相同的阶段常数偏置可被 LayerNorm 抵消 | 使用不同特征方向的正余弦阶段编码，使阶段区分和 event gate 有实际梯度 |
| OT 数值 | 距离为零的开方反传、极小成本均值归一化和半精度容易不稳定 | 使用 vector_norm 的零点次梯度；归一化尺度下限 0.01；成本 FP32；非有限成本显式报错 |
| 缺失模态 | 仅关闭辅助项不能阻止“缺失”的原始值进入主干；NaN×0 也不是安全清除 | 编码前按 availability 选择有效值，编码后再次清零；无效值不进入主干/损失，全部模态缺失报错 |
| 删失参考 | 整数时间可能截断权重；并列事件/删失时间的 reverse-KM 分母不一致 | 时间/删失值转浮点；reverse-KM 先扣除同刻观测事件；排序使用 G(T-)；对照本机 sksurv 测试 |
| 保存重载 | 可变长度参考 buffer 不能直接装入空 buffer；退火 epoch 未保存导致推理 OT 不一致 | 重建参考 buffer 形状；保存 transport_epoch 并同步 args.cur_epoch；严格校验损失 recipe |
| 无效配置 | 部分历史开关保留在参数中但实际被禁用，记录容易与执行不符 | 同步冻结参数，公开 v3.14 三个辅助系数、mask ratio 和重建模式，并写入 checkpoint |

## 3. 当前执行目标，不以 YAML 注释代替代码

默认：

\[
L=\frac{L_{NLL,\alpha}^{sum}}{B}
 +0.10 L_{IPCW-rank}
 +0.05 L_{slot-NLL,\alpha}
 +0.02 L_{content-distance}
 +0.10r(e)L_{recon},\quad
r(e)=\operatorname{clip}((e-2)/5,0,1).
\]

epoch 使用训练器从 0 开始的编号。编号 0–2 不执行辅助重构，编号 7 达到完整系数。
direction、dose、reconfiguration、ETAR、listwise、MGPTR、POC、POM、Cox 不参与总目标；日志中的 OT distance 是诊断量，不是额外训练损失。

### 3.1 主 NLL 与 per-slot NLL

令 y 为离散时间区间，c=0 为观测事件、c=1 为删失，h 为 sigmoid(logits)：

\[
\ell=- (1-c)\left[\sum_{t<y}\log(1-h_t)+\log h_y\right]
      -(1-\alpha)c\sum_{t\le y}\log(1-h_t).
\]

主损失对患者求和，训练器除 B 一次。槽损失在可用患者及槽上取均值，再平均可用模态。
同一模态所有槽共享一个 Linear hazard head；WSI 与组学分别有自己的 head。

参考配置显式保留历史训练器的 `alpha_surv=0.5`，也就是删失项权重为 0.5。
**这不是严格的未加权 NLL**；alpha=0 才是普通删失似然。选择 0.5 是保持比较协议，不是已经证明 0.5 最优。

IPCW 排序的可比较对要求 i 观测到事件且 T_i<T_j，使用截断于 10 的 1/G(T_i-)² 权重。
参考删失分布仅由训练折拟合；64 条历史风险缓存停止梯度，并在 epoch / 参考重拟合 / checkpoint 重载时重置。

### 3.2 anti-collapse 不再优化 hazard 方差

对每个患者、每种模态，先在特征维中心化并归一化槽内容，得单位向量 u：

\[
L_{div}=\operatorname{mean}_{b,k\ne j}
\left[\max\left(0,1-\frac{\|u_{bk}-u_{bj}\|_2}{\sqrt{0.2}}\right)\right]^2.
\]

距离达到 sqrt(0.2) 对应余弦相似度不高于 0.9，就不再排斥。它不要求生物学程序正交，也不要求同一患者的各槽预测不同结局。
按患者内部槽对归一化，不能把患者间距离混入 anti-collapse。
距离范数对很小但非零的槽差异有更有效的梯度；**完全相等时仍可能零梯度**。
所以同时保留原型路由打破对称，而不是只提高一个正则系数。
不同原型只影响 routing，不直接加到输出值：所有输入 token 完全相同时，不会凭空制造“不同内容”。

### 3.3 重构目标与四种模式

单个重构分支使用：0.5×余弦距离 + 0.5×Smooth-L1；预测和 detached target 都做特征归一化。
self 从组学槽解码；cross 从 WSI 经重新求解的 OT 搬运到组学槽位置后解码。两者各占一半。

| 模式 | 输入/计分位置 | 总预算 |
|---|---|---|
| masked，默认 | 先遮 token，再聚槽和求 OT；只计遮挡位置 | 0.10r(e) |
| full | 完整 token，所有 pathway 位置计分 | 0.10r(e) |
| hybrid | masked 与 full 各取一半 | 仍是 0.10r(e)，不是 0.20r(e) |
| off | 不执行辅助重构 | 0 |

**不建议现在把 v3.13 重构再额外叠加一份。** 先在同一 v3.14 骨干上比较上述模式，才可区分是否需要完整位置约束。
这里的 full 是思想对应的局部实现，不是旧 v3.13 的完整模型/训练配方，因此不能把这项消融命名为“复现 v3.13 全模型”。

mask ratio 目标为 20%，整数化后至少遮一个、P>1 时至少留一个。例如诊断的 12 个 pathway 实际遮 2 个，即 16.7%。
生物学 pathway 可能共享基因，所以这里只排除了被遮 token 的直接输入通道，不宣称原始基因信息完全独立。

## 4. 前向与反向检查

事实分支：WSI/pathway encoder → 原型槽路由 → 分阶段三种几何成本及 evidence 边际 → Sinkhorn → 事件读取器 → event gate → hazard logits。

辅助分支使用独立的 masked 组学视图，但事实生存分支始终读取全部可用模态。
梯度检查分别覆盖：主 NLL、rank、slot NLL、diversity、MTR 到各 encoder、槽模块、原型、stage_pair_cost、decoder、event gate 的路径。

- MTR 应更新 decoder、可见组学路由、WSI 路由、mask token 和 OT 成本；测试确认非零梯度。
- MTR 不应通过完整组学 factual gate 传播信息或梯度；测试确认 gate 不影响输出、梯度为 None。
- 被遮 token 在辅助输入路径的梯度为零；作为 detached target 也不反传。
- rank 没有可比较对时为零是预期行为，不是断梯度 bug。
- 重构 warmup / off 时跳过其计算，不把零权重乘在可能不稳定的多余图上。
- GPU FP32 与 FP16 autocast 都检查有限 loss/gradient、参数更新；不等于支持手工把整模型权重转为 FP16。

最终小尺寸 seed=3 初始化批次的 WSI encoder 梯度范数：主 NLL 0.5440，已加权 slot-NLL 0.02459，diversity 0.00775，MTR 0.01332。
WSI 槽模块中已加权 diversity / 主 NLL 梯度范数约 0.110，说明该约束在这批数据上已经不是近乎无效的梯度。
这只是局部尺度检查，不代表整个训练过程的梯度比例，更不能据此确定最优系数。

## 5. 结构诊断结果与不能据此得出的结论

所有数据都是合成的，同一小批数据重复训练；不是 TCGA 五折，也没有独立验证集。
最后四种模式使用独立于模型初始化的同一数据随机种子、同样参数初始化和 alpha=0.5。
完整原始记录及源码 SHA-256 位于 audit_results/v314_*.json。

### 5.1 旧实现可复现的退化

v314_before.json：16 个槽、10 次迭代、3 个种子、各 30 步后，WSI cosine 为 0.999636–0.999973，组学为 0.999569–0.999991。
旧 diversity 梯度远小于主 NLL，例如 seed=3 的 WSI 槽模块梯度范数约 0.000057 对 3.384。
这些旧记录采用 alpha=0.15，且数据生成消耗了初始化之后的 RNG；与最终版不是严格配对性能比较，不能比较其训练 NLL 后宣称提高预测性能。

### 5.2 最终小尺寸与高维结果

最终小尺寸：B=16，WSI 输入/隐层 32 维，48 个 patch，12 个 pathway，每种模态 16 槽、10 次迭代，种子 3/17/29 各 30 步。

| 种子 | WSI cosine | 组学 cosine | WSI 有效秩 | 组学有效秩 |
|---|---:|---:|---:|---:|
| 3 | 0.9134 | 0.3948 | 7.29 | 8.68 |
| 17 | 0.9251 | 0.4288 | 7.57 | 8.72 |
| 29 | 0.8754 | 0.3456 | 8.15 | 8.65 |

高维 GPU：B=8，4096 patch，WSI 输入 1024 维、隐层 256 维；其他合成设置同上。

| 种子 | WSI cosine | 组学 cosine | WSI 有效秩 | 组学有效秩 |
|---|---:|---:|---:|---:|
| 3 | 0.9509 | 0.4622 | 9.65 | 9.10 |
| 17 | 0.9876 | 0.5982 | 4.09 | 9.48 |
| 29 | 0.9820 | 0.4979 | 5.51 | 9.44 |

这里的 cosine 是原始槽内容余弦，不等同于训练损失中的特征中心化余弦。有效秩由患者内部去均值后的槽矩阵奇异值熵计算，尺度无关，必须连同槽间标准差一起看。
测试匹配了 patch/特征/槽维度，但没有匹配真实 pathway 数量及每个 pathway 的基因宽度，OT 迭代和 event 层数也是缩减诊断设置；不是完整生产显存/性能基准。

**仍有剩余风险：** 高维 WSI 有的种子内容高度相近且有效秩下降；中间 B=2、10 步检查的 WSI cosine 仍接近 0.9999。
这些证据不支持“塌缩已彻底解决”。修复缓解了已观察到的退化，真实训练还必须监控内容、分配和预测三个层面。
相同患者的槽 hazard 接近不自动等于内容塌缩；反过来，仅有效秩高也不能排除差异幅度极小。

### 5.3 重构比较的边界

masked/full/hybrid/off 有同骨干、同 seed 的 30 步控制检查。其作用是核实损失预算、梯度与数值，而不是选出泛化最佳模式。
即使它们在合成训练 NLL 上很接近，也不能证明重构不重要或 masked 优于 full。
参考系数 0.05/0.02/0.10 均为可配置起点，本轮没有凭合成训练分数宣布最优权重。

## 6. 自动检查与仓库边界

相关回归目标：v3.10、v3.13、v3.14、hierarchical slots 四个测试文件。
最终结果：**68 passed**，包括 44 项 v3.14 测试；40 条提示均为 Transformer nested-tensor 优化未启用的警告，不是测试失败。
最终源码的 CPU 三种子 90 步、GPU 三种子 90 步、full/hybrid/off 控制各 30 步，共 270 个合成优化步骤，损失和梯度均有限。
五份最终诊断 JSON 的全部模块 SHA-256 均已逐一与当前文件比对一致；历史中间文件的 hash 不同是预期现象。
覆盖 NLL 手算/极端 logits、mask 不变性/梯度、可用模态、非有限值、重复槽、整数 metadata、reverse-KM、配置覆盖、版本隔离、warmup、四种重建模式、checkpoint 和真实 train_one_epoch/batch adapter。
project doctor 通过。GPU 确认是本机 CUDA PyTorch 环境，不是仅做静态编译。

整个 tests 目录仍有 4 个历史 collection error，涉及已移动/缺失的 risk_ordering、v32、v330 模型或脚本引用。
这些不是 v3.14 测试失败，本轮未删除或改写历史测试来制造“全仓绿灯”。只宣称相关回归通过。

## 7. 下一步应如何定权重和决定是否加入完整重构

在真实患者分组、固定 splits、训练折拟合 bins / censor reference 的前提下，按次序做小规模对照：

1. 固定新骨干，先比较 per-slot NLL 系数 0、0.01、0.05；同时看内容槽差异和验证风险排序，不能只看训练 NLL。
2. 固定上一步，比较 diversity 0、0.01、0.02、0.05；监控其 encoder 梯度相对主任务的比例，不能仅比较各项 loss 的数值。
3. 固定前两步，比较 off / masked / full / hybrid，再比较总重构系数 0.05、0.10；不能同时把结构与所有系数换掉后归因于一个损失。
4. 原始 cosine、槽间标准差、有效秩、assignment overlap/覆盖、mask target 患者间方差与 hazard 方差一并保存；多个种子复核。
5. 只用内部验证选择系数，外部测试不参与调参；保存每折预测、最优 checkpoint、曲线、配置、split hash 与代码版本。

在线 detached target 仍可能随 encoder 漂移；decoder 也可能学会 pathway 均值。重构不是自动的 anti-collapse 保证。
若真实实验暴露该问题，再比较固定/EMA teacher 或更严格重构目标，而不是无证据加入另一组损失。
缺失模态测试只验证忽略无效输入和数值可执行性，不建立缺失模态泛化结论。
评估中的 low/high cost 干预是模型内部诊断，不是临床因果干预；direction=0 时也不保证风险响应方向有序。

## 8. 复查入口

```powershell
python -m pytest tests/test_dct_v310_directional_regularized_transport.py tests/test_dct_v313_transport_reconstruction.py tests/test_dct_v314_masked_transport_reconstruction.py tests/test_hierarchical_prognostic_slots.py -q
python -m survot_rank.cli doctor
python -m scripts.audit_v314 --output audit_results/v314_final_masked.json --steps 30 --device cpu
python -m scripts.audit_v314 --output audit_results/v314_final_gpu_shape.json --steps 30 --seeds 3 17 29 --device cuda --projection-dim 256 --encoding-dim 1024 --patches 4096 --batch-size 8
```

配置：configs/dct_v314_blca_uni.yaml。需在真实训练前替换 Linux 数据路径；本机没有这些特征数据，未启动真实训练。
