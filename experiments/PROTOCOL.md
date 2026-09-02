# DCT-Reg 正式证明协议

## P0：冻结与数据门

在任何正式测试前冻结：Git commit、配置、患者级 splits、终点定义、风险分数、checkpoint 选择规则、统计方法和本文件中的通过线。所有锚点、删失参考和时间分箱只能由训练折估计。

采用嵌套评估：内层验证用于 checkpoint 选择，外层 fold 只评估一次。不得把“用来选最佳 epoch 的验证集”改名为独立测试集。

## E1：匹配的 2×2 目标消融

BLCA 五个相同 folds：`nll_only`、`ipcw_only`、`direction_only`、`full`。除两项权重外，数据、初始化、训练预算和模型选择完全一致。报告每折数值、配对差、均值、标准差和配对置信区间。

```powershell
python scripts/run_dct_v310_experiments.py plan
```

## E2：最终预测与跨癌泛化

冻结 full 方法，在 BLCA、UCEC、KIRC、HNSC、SKCM、LUSC 上各五折。报告 Harrell C-index、IPCW C-index、IBS、time-dependent AUC 和校准；同时至少在 UCEC、LUSC 运行匹配的 IPCW-only 比较。

```powershell
python scripts/run_dct_v310_final_cross_cancer.py plan
```

## E3：机制必要性与零假设

至少在 BLCA、UCEC、LUSC 的 folds 1、2、4 运行：

- `fixed_coupling`：不重新 Sinkhorn；
- `noisy_batch_mean_anchors`：非预后噪声锚点；
- `permuted_reference`：训练参考时间置乱；
- `stage_jitter`：阶段边界扰动；
- 持出 checkpoint 上的 shuffled/uniform feasible plans，保持边际后再解码。

```powershell
python scripts/run_dct_v310_experiments.py plan --cancers blca,ucec,lusc --folds 1,2,4 --variants fixed_coupling,noisy_batch_mean_anchors,permuted_reference,stage_jitter
```

## E4：连续干预与患者级审计

冻结 checkpoint，在外层持出患者上对 `alpha={0,.25,.5,.75,1}` 的低/高风险方向分别重求解 OT。导出 factual/low/high 风险、患者级 delta、计划 TV、边际误差、DCR、DMR 和置信区间。按癌种、fold、事件/删失状态分层展示，不只汇报宏平均。

## E5：强基线、统计与外部性

在相同特征和 splits 下比较非 OT 融合及代表性生存/OT 基线；报告参数量、推理时间和显存。多癌种比较进行多重校正。若无独立机构数据，只能宣称内部跨队列泛化，不能宣称临床外部泛化。

## 结果包

每折目录必须包含：

```text
predictions.csv
audit_cases.parquet (or pkl)
audit_metrics.json
training_curve.csv
checkpoint.pt
resolved_config.yaml
split_manifest.json
environment.json
run_manifest.json
```

`run_manifest.json` 至少记录 Git commit、命令、开始/结束时间、随机种子、数据版本和各文件 SHA-256。

