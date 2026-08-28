# 当前状态（2026-08-28）

## 已完成

- DCT-Reg 冻结目标、事实 Sinkhorn、低/高风险 cost intervention、re-Sinkhorn 和共享风险读取已实现。
- 评估模式可导出 factual/low/high 真实计划、三侧边际误差和真实 Plan TV。
- uniform/shuffled 可行计划控制已实现，保持原边际后使用同一个风险读取器解码。
- 双向方向一致率、累计病例 offset、固定 coupling 当前 batch 重放和正式证据 manifest 已实现。
- 固定预算训练期间不查看外层 fold，结束后只评估一次。
- 单元测试与项目 doctor 已通过；这些仍只属于 P0 结构证据。

## 已实现但尚未正式运行

- P1：BLCA 五折 2×2 目标消融。
- P2：BLCA/UCEC/LUSC 预注册 folds 的 full 与 fixed-coupling 对照。
- P3：持出 checkpoint 的 factual/uniform/shuffled plan necessity 审计。
- P4：noisy/permuted/stage-jitter 训练控制和 anchor-swap 持出审计。
- P5：high/low 两侧 alpha sweep 与患者级 DMR。
- P6：六癌种 × 五折锁定 full 运行。

## 尚未实现或仍缺失

- P4 的跨 fold frozen-anchor 敏感性分析。
- full、pair-only、plan-only 三读取器结构消融。
- 正式结果聚合中的患者级 bootstrap CI、多重比较校正和跨癌异质性表。
- 独立机构/外部队列 P7。

## 当前结论

截至本记录，正式结果仍为零。代码就绪不得写成 P1-P7 已完成；只有 `MATRIX.csv` 的 `execution_status` 在完整工件审核后才能从 `pending` 更新。

