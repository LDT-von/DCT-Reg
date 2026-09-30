# DCT v3.13：初稿所需实验与执行顺序

> **2026-09-30 更新**：本文件保留初步实验建议及针对 `93d8314` 的临时补跑命令。当前确认的实施范围、两套协议、SlotSPE 两种配方和验收标准，以 [V313_IMPLEMENTATION_PLAN.md](V313_IMPLEMENTATION_PLAN.md) 为准。后续新配方单支重建已在设计中修正，不应继续套用本文件的 `scale=2` 临时补偿。

更新：2026-09-29。审阅代码：`93d83143a67aa3921abf5f266cffdda1ddf7057b`。本清单根据当前可执行代码及用户正在运行的实验制定；未启动训练、推理或数据处理。服务器任务状态与原始结果尚未在线读取。

## 1. 当前 70 个任务实际回答什么

`scripts/run_ablation_loss_components.py` 安排 BLCA、KIRC 两癌种，7 组、各 5 折，UNI2-h、2048 patches、batch 8、30 epochs、seed 3。

| 实验 | 实际目标（重建 ramp 满后） | 能回答的问题 |
|---|---|---|
| Exp0 | NLL | 同一 DCT 主干仅用主生存损失的表现 |
| Exp1 | Exp0 + 0.10 IPCW rank | 在 NLL 基础上加排序的效果 |
| Exp2 | Exp1 + 0.05 slot NLL | 在前述目标上加逐槽监督的效果 |
| Exp3 | Exp2 + 0.10 diversity | 在前述目标上加多样性约束的效果 |
| Exp4 | Exp3 + **0.025 self reconstruction** | 当前实现是低权重 self-only |
| Exp5 | Exp3 + **0.025 cross reconstruction** | 当前实现是低权重 cross-only |
| Exp6 | Exp3 + **0.05 self + 0.05 cross** | 完整 v3.13 |

Exp0 仍有完整 DCT 架构，不能称为原版 SlotSPE 或“无 OT”。Exp0→Exp3 是按固定顺序逐步加项，只能说明对应背景下的增量；不能把每一步增益当作该项在完整模型中的独立贡献。

### 先处理 Exp4 / Exp5 的权重混杂

当前 v3.13 初始化在禁用一条重建支路时，先把总重建权重由 0.10 降为 0.05；`reconstruction_losses` 中仍乘保留支路的 0.5，最终为 **0.025**。完整模型每条支路则为 **0.05**。这不是推测，是两个系数相乘的静态代码结果。

保留现有两组作为低权重记录。若要比较完整模型与“只删去一条支路”，在 **93d8314** 上对这两组补 `--set dct_v313_lambda_reconstruction_scale=2.0`，保留支路即可回到 0.05。不要给 Exp6 加这个补偿；不要在以后修改了模型权重逻辑后继续沿用这个补偿。

对应源码：v3.13 `model.py:156–181`、`:391–394`、`:474–483`。本轮只生成计划，未修改模型或运行中的调度器。

## 2. 优先级与新增工作量

“新增任务数”按两癌种各 5 折或五癌种各 5 折计算；是训练次数，不是 GPU 时间估计。已完成任务只有在输入、划分、模型、实际目标及评估协议一致时才能复用。

| 优先级 | 实验 | 最小范围 | 目的 / 工作量 |
|---|---|---|---|
| P0 | 明确选模型与最终评估的分离 | 所有正式主表方法 | 先确定协议，再扩大训练；见第 3 节 |
| P0 | 单支重建权重匹配 | BLCA + KIRC | 补 Exp4、Exp5；共 20 次，可将尚未开始的对应任务替换为正确配置 |
| P0 | 同协议主表：v3.13 与原版 SlotSPE | 预先确定 5 个癌种 | 每方法 25 次；优先 BLCA/KIRC，其他队列补齐；已有合格记录复用 |
| P0 | 普通跨模态重建 vs OT 条件重建 | BLCA + KIRC | 新增 1 臂、10 次；最直接检验 v3.13 的新增贡献 |
| P0 | 学习 OT vs 独立耦合 | BLCA + KIRC | 新增 1 臂、10 次；检验主预测对匹配结构的依赖 |
| P1 | 阶段共享计划、单几何、去原型坐标 | BLCA + KIRC | 3 臂、30 次；分别支撑阶段、多几何、原型三个模块主张 |
| P1 | WSI-only、omics-only、简单 late fusion | 先 BLCA + KIRC，正式主表尽量扩展 | 对应输入可用性下验证多模态价值；每新增臂在两癌种上为 10 次 |
| P1 | 代表性公开方法 | 与主表相同队列 | 建议 MCAT、MOTCat；按相同输入/划分/终点和选模预算复现，不能直接抄其论文分数 |
| P1 | 完整模型及关键对照的随机种子重复 | BLCA + KIRC、3 个预先指定 seed | 优先 full、普通重建、独立耦合；复用符合协议的已有 seed，其他版本不必全部重复 |
| P1 | IBS、IPCW C-index、校准、KM、效率 | 已保存的独立评估预测与模型 | 通常不必重训；保存所需预测、时间分箱、训练删失参考 |
| P2 | 外部队列/中心划分、特征骨干复现 | 数据可用时 | 加强泛化证据；缺少外部队列时明确研究范围 |

建议主表沿用既定的 BLCA、KIRC、HNSC、LUSC、SKCM 五队列，先核对 UNI2-h 与组学配对覆盖，不根据结果好坏增删队列。若加入 UCEC，提前确定为第六队列并对全部比较方法一致执行。

**近期最有价值的新任务**：完成当前损失实验并补齐单支权重；原版 SlotSPE 同协议复现；普通重建对照；独立耦合对照。先得到这四类证据，再扩展大规模超参扫描。

## 3. 正式评估协议

当前 `survot_rank/training/train_runner.py:592` 只取 train/val；`:657` 每 epoch 在 val 上评估，`:681–686` 按同一 val 的 C-index 挑最高 epoch 并保存其指标。当前 best-val 五折均值不是独立测试均值。

### 建议的主表协议

1. 在患者层面固定外层五折；同患者全部切片进入同一折。
2. 每个外层训练集内再留一个验证子集，选择 epoch 与超参数；外层测试折不用于模型选择。
3. 分箱、删失分布、组学标准化、风险锚点与任何特征筛选只使用允许的训练部分。
4. 所有比较方法使用相同外层患者、特征版本、终点和随访规则；各方法可以保留合理的原生超参数，但选模预算和选模数据来源一致。
5. 固定方法与配置后报告外层测试预测。既往已被反复查看的队列仍属于内部验证研究；不能据此宣称全新外部验证。

70 个任务可以作为开发消融保留；若要形成正式独立评估证据，相关关键实验需按上述协议运行。当前调度脚本未实现这层内验证/外测试，不能仅把 `val` 名称改成 `test`。

如果为了对照既有公开实现保留 best-validation 结果，应单列“相同开发评估口径”表，写清选模方式；固定 epoch 也必须在观察评估结果前确定，事后改报第 30 轮不能自动消除开发选择影响。

原则依据：[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) 要求性能评估数据与训练、调参及模型选择数据区分。

## 4. 怎样证明运输感知重建带来收益

### 最关键的四组

| 臂 | 保持不变 | 唯一改变 | 结论边界 |
|---|---|---|---|
| A：无重建（Exp3） | DCT 主干及 NLL/rank/slot/diversity | 关闭 self/cross | A vs D 衡量完整重建正则的增量 |
| B：仅 self（权重匹配 Exp4） | 同上 | self 0.05，cross 0 | B vs D 衡量添加 cross 的增量 |
| C：普通 cross + self | 主预测继续用事实 OT；相同查询解码器、目标、权重和 ramp | cross 解码直接读未运输的 WSI slots | C vs D 检验“使用运输结果”是否优于一般额外重建 |
| D：完整 v3.13（Exp6） | 默认配置 | self 0.05 + OT cross 0.05 | 论文主方法 |

C 不能通过直接换成整个 SlotSPE 或 v3.15 实现；那样会同时改变主干、读取器和容量。应在 v3.13 内仅替换 cross 解码器收到的 memory。当前仓库没有本次确认过的 C 臂独立开关，需要新增实现。

同时报告 held-out C-index、IBS、重建误差与成对差值。重建误差更低不能单独证明预后更好；自编码表征尺度也会变化，优先参考归一化误差并检验特征方差。

可选机制臂：仅对 cross 支路使用 `stop_gradient(T)`，预测支路仍保留可微 OT。与 D 比较检验重建反向约束运输的作用；该臂也需新增实现。

### OT 与其他结构的单项对照

- **独立耦合**：把学习计划替换为当前相同边际的 `a bᵀ`，重新训练，保留读取器、损失和其余网络；不要在非均匀边际下直接用 `1/(K_w K_o)` 当作同约束替代。
- **阶段共享**：各阶段使用同一成本/计划，保留阶段读取器的数量与输出形状，使差别尽量集中在阶段条件匹配；仅关闭一个代价项不必然等价于“无分阶段”，因为阶段边际仍可能变化。
- **单几何**：只保留余弦匹配，说明如何保持读取器维度或匹配参数预算，不能只改标签。
- **无原型坐标**：直接用局部 slots 进入后续 OT，保持 slot 数、维度与其余设置相同。

这些是实验规格，不是已确认可运行的命令行开关。推理时替换计划可作为低成本依赖诊断；其分布变化导致的性能下降不能替代重新训练的架构消融。

## 5. 少量重复训练之外，优先补哪些图表

- **主表**：每癌种各折 C-index、均值及标准差，同时提供 IPCW C-index 与 IBS。各指标在同一个选定模型上计算，不能分别选各自最佳 epoch。
- **不确定性**：保存逐患者成对预测，报告关键差值和置信区间。按患者配对、保留折结构进行重采样；将其解释为给定拟合模型的条件不确定性。需要训练不确定性时使用种子/重复划分。不要把 5 折当作大量独立样本，也不要混合癌种风险分数后直接计算一个 C-index。
- **校准/KM**：时间点在训练随访支持内预先选择；KM 分组阈值由训练/内验证确定。分层显著不能替代性能和校准检验。
- **机制图**：阶段运输图＋对应 patch 位置＋通路；需要坐标/切片资源。预先确定病例选择规则，同时展示成功与失败例。
- **槽诊断**：WSI/omics 分开统计 slot 两两距离、hazard 方差、分配重叠或有效槽数。较好的 C-index 不能证明 slots 已形成不同语义。
- **效率表**：可训练参数数目、推理耗时、训练耗时、峰值显存，统一硬件和 batch。区分参数量与 checkpoint 文件大小。

训练器已经计算 IPCW C-index/IBS/iAUC，且保存患者 risk/time/censor/logits。先检查这些指标是否有限、评估时间范围是否合法及是否来自独立评估，再决定是否补推理。[指标定义与适用条件](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)。

缺失组学不宜作为当前主打实验：v3.13 cross 重建使用由两种模态共同计算的事实 OT，重建头也未作为标准推理填补流程。重建通路嵌入不等同于从 WSI 恢复原始基因表达。

## 6. 93d8314 可直接使用的单支权重补跑命令

以下为 Linux 服务器命令，沿用当前开发协议，共 20 次训练。选择空闲 GPU 后由用户运行；新目录与当前 70 个任务分开，不覆盖历史结果。GPU 0 是示例，修改 `DCT_GPU` 即可。该命令不是正式内验证/外测试训练器。

```bash
cd /data1/DCT-Reg
set -euo pipefail
DCT_PY=/home/ubuntu/.conda/envs/trisurv/bin/python
DCT_GPU=0
export PYTHONPATH=/data1/DCT-Reg

for cancer in blca kirc; do
  for arm in self_only cross_only; do
    if [ "$arm" = self_only ]; then
      disable_self=false
      disable_cross=true
    else
      disable_self=true
      disable_cross=false
    fi
    log_dir="/data1/DCT-Reg/logs/v313_recon_matched_93d8314/$cancer/$arm"
    mkdir -p "$log_dir"
    for fold in 0 1 2 3 4; do
      CUDA_VISIBLE_DEVICES="$DCT_GPU" "$DCT_PY" -m survot_rank.cli train \
        --config configs/dct_v313_blca_uni2h.yaml \
        --set "gpu=$DCT_GPU" --set "study=$cancer" \
        --set "k_start=$fold" --set "k_end=$((fold + 1))" \
        --set "specific_simple=v313_recon_matched_93d8314_${cancer}_${arm}" \
        --set "results_dir=/data1/DCT-Reg/results/v313_recon_matched_93d8314/$cancer/$arm" \
        --set dct_lambda_ipcw_rank=0.10 \
        --set dct_v311_lambda_slot_nll=0.05 \
        --set dct_v311_lambda_slot_diversity=0.10 \
        --set "dct_v313_disable_self_reconstruction=$disable_self" \
        --set "dct_v313_disable_cross_reconstruction=$disable_cross" \
        --set dct_v313_lambda_reconstruction_scale=2.0 \
        2>&1 | tee "$log_dir/fold${fold}.log"
    done
  done
done
```

后续权重敏感性实验在此版本应使用 `dct_v313_lambda_reconstruction_scale`；直接改某些 `lambda_reconstruction` YAML 字段不等于当前类常量实际改变。每个实验留存模型实际权重、禁用标记、完整命令、提交号、划分文件及各轮曲线。

现有 `scripts/run_slotspe_blca_paper.sh` 使用 UNI 1024 维、4096 patches、batch 32 和 `5fold`，与本轮 UNI2-h 不同，不能直接作为匹配基线。现有六癌调度器仍采用 best-val 训练器，不能代替第 3 节的新评估协议。

## 7. v3.10 初稿迁移

原稿位于 `paper/DCT_唯一初稿.md` 与同名 `.docx`。新的 Markdown 工作稿为 `paper/DCT_v313_初稿.md`。

| 旧稿内容 | v3.13 处理 |
|---|---|
| 摘要的 0.703 vs 0.697、方向损失 +0.01 | 删除 v3.13 名下的这些旧结论，等待同协议新结果 |
| 共享原型、阶段多几何、运输事件读取 | 保留可核对结构，准确说明两模态有各自跨患者共享的字典 |
| §3.6 direction loss、§3.8 三项冻结目标 | 改为逐槽监督、多样性与运输条件重建；采用 v3.13 实际目标 |
| 以方向一致性审计作为中心贡献 | 调整为补充分析；没有新实测就不保留 v3.10 的审计数字 |
| 结构图中的低/高风险训练端点 | 主图改画 self/cross 重建训练支路，注明推理不执行该支路 |
| 实验与结论 | 围绕全模型、同协议基线、普通重建、学习 OT 等问题重写 |

可以立即写完方法与实验设计。摘要结果句、结果表和最终结论随新实验更新；负向或无增益结果也按实填写。

公开基线来源：[MCAT 官方实现](https://github.com/mahmoodlab/MCAT)、[MOTCat 官方实现](https://github.com/Innse/MOTCat)。跨癌种任务数量为本项目建议，不是这些来源规定的发表门槛。
