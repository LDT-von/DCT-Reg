# 当前状态（2026-08-29）

## 首轮正式运行已完成

P1（BLCA 五折 2×2 消融）、P2（folds 1,2,4 机制对照）、P3（plan necessity 审计）、P4（锚点特异性）、P5（双向剂量）的 30-epoch 固定预算正式运行全部完成，证据包落盘于 `results/dct_v3.10_experiments/robust/`。逐折表格与审计指标见 `RESULTS_20260829.md`。

关键结果速览：训练 C-index 正常（blca full 0.638）；但审计侧 P3 计划替换后风险变化趋近 0、P4 方向一致率低于机会、P5 剂量单调率远低于通过线。**在排查审计解码路径是否真实消费被替换计划之前，不得将审计结果写为正式结论。**

## 已完成（实现层面）

- DCT-Reg 冻结目标、事实 Sinkhorn、低/高风险 cost intervention、re-Sinkhorn 和共享风险读取已实现。
- 评估模式可导出 factual/low/high 真实计划、三侧边际误差和真实 Plan TV。
- uniform/shuffled 可行计划控制已实现，保持原边际后使用同一个风险读取器解码。
- 双向方向一致率、累计病例 offset、固定 coupling 当前 batch 重放和正式证据 manifest 已实现。
- 固定预算训练期间不查看外层 fold，结束后只评估一次。
- 单元测试与项目 doctor 已通过；这些仍只属于 P0 结构证据。

## 已实现但尚未正式运行

- P6：六癌种 × 五折锁定 full 运行（用户已确认不重跑，旧 50ep 队列作为对照基准保留）。
- P7：独立机构/外部队列。

## 尚未实现或仍缺失

- full、pair-only、plan-only 三读取器结构消融。
- 正式结果聚合中的患者级 bootstrap CI、多重比较校正和跨癌异质性表。
- 独立机构/外部队列 P7。

## 当前结论

截至本记录，训练侧已产出首轮正式 C-index（见 `RESULTS_20260829.md`）。审计侧 P3/P4/P5 结果异常，在核实审计解码路径正确性前，审计结论保持未定。`MATRIX.csv` 的 `execution_status` 仅在完整工件审核后更新。

