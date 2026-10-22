# v3.13 保留 / 拟删结果目录逐条核对表（63 项）

- 清单文件：`docs/cleanup/V313_RETENTION_DELETION_MANIFEST_20261022.json`（revision 3.1）
- 状态：`PROPOSED_ONLY_NOT_EXECUTED` — **未执行任何清理**，本表不构成删除授权
- 本轮核对范围：仅 `results/` 大文件 63 项 / 20.739 GiB
- 机器判定：含全仓唯一产物 21 项；被活跃脚本引用 3 项；无任何引用 2 项

**全仓唯一** = 该预测/权重在 `results/` + `实验档案/` + `历史备份/` 范围内仅此一份，删除后无法从仓内任何位置恢复。

## 逐条表

| # | 路径 | GiB | 方法 | 队列 | 折 | 种子 | 预测pkl | 全仓唯一 | 权重 | 全仓唯一 | 标记 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `dct_v3.10` | 3.812 | dct_v310_directional_regularized_t | blca,brca,hnsc,kirc, | 0 | 3 | 62 | 0 | 32 | 25 | 唯一产物 快照 |
| 2 | `dct_v311_blca_uni_fixed_pancancer` | 3.411 | dct_v311_slot_interpretable | brca,coadread,hnsc,l | 0,1,2,3,4 | 3 | 60 | 4 | 30 | 26 | 唯一产物 |
| 3 | `dct_v3.10_uni2h_extend` | 2.962 | dct_v310_directional_regularized_t | coadread,luad,stad,u | - | 3 | 48 | 0 | 25 | 12 | 唯一产物 |
| 4 | `dct_v3.2` | 2.522 | dct_v32_transport_guided_slot_reag | blca | - | 3 | 40 | 0 | 20 | 0 | 快照 |
| 5 | `dct_v311_blca_uni_fixed` | 1.254 | dct_v311_slot_interpretable | blca | 0,1,2,3,4 | 3 | 23 | 1 | 11 | 0 | 唯一产物 快照 |
| 6 | `dct_v316_slot_mi_blca_uni2h` | 0.709 | dct_v316_slot_mi_decomposition | blca | - | 3 | 10 | 0 | 5 | 5 | 唯一产物 |
| 7 | `dct_v314_blca_uni2h` | 0.701 | dct_v314_masked_transport_reconstr | blca | - | 3 | 12 | 0 | 6 | 4 | 活跃脚本 唯一产物 |
| 8 | `dct_v312_skcm_5fold` | 0.572 | dct_v312_slot_interpretable_impute | blca,skcm | - | 3 | 10 | 4 | 5 | 5 | 唯一产物 |
| 9 | `dct_v312_blca_uni_fold0_aligned` | 0.572 | dct_v312_slot_interpretable_impute | blca | 0 | 3 | 10 | 0 | 5 | 4 | 唯一产物 |
| 10 | `dct_v3.10_uni_blca_p2048_30ep` | 0.569 | dct_v310_directional_regularized_t | blca | 0,1 | 3 | 10 | 0 | 5 | 0 | — |
| 11 | `dct_v3.10_uni_all10` | 0.569 | dct_v310_directional_regularized_t | blca,brca | - | 3 | 10 | 0 | 5 | 0 | — |
| 12 | `dct_v311_blca_uni_v2` | 0.569 | dct_v311_slot_interpretable | blca | - | 3 | 10 | 0 | 5 | 0 | — |
| 13 | `dct_v3.2_smoke` | 0.504 | dct_v32_transport_guided_slot_reag | blca | - | 3 | 8 | 0 | 4 | 0 | — |
| 14 | `transport_candidates` | 0.239 | dct_risk_ordering_transport | blca | 0 | 3 | 2 | 0 | 2 | 0 | — |
| 15 | `transport_candidates_smoke` | 0.238 | dct_risk_ordering_transport | blca | - | 3 | 4 | 0 | 2 | 0 | — |
| 16 | `dct_v311_blca_uni2h_fixed_v2` | 0.238 | dct_v311_slot_interpretable | blca | - | 3 | 4 | 0 | 2 | 0 | — |
| 17 | `dct_v311_blca_uni2h` | 0.238 | dct_v311_slot_interpretable | blca | - | 3 | 4 | 0 | 2 | 0 | 快照 |
| 18 | `dct_v312_blca_uni_smoke` | 0.229 | dct_v312_slot_interpretable_impute | blca | - | 3 | 4 | 0 | 2 | 1 | 唯一产物 |
| 19 | `dct_v311_blca_uni_v2_partial_backup_0742` | 0.228 | dct_v311_slot_interpretable | blca | - | 3 | 4 | 0 | 2 | 0 | — |
| 20 | `dct_v311_blca_uni` | 0.228 | dct_v311_slot_interpretable | blca | - | 3 | 4 | 0 | 2 | 0 | 快照 |
| 21 | `dct_v3.10_experiments_smoke` | 0.119 | dct_transport_intervention_consist | blca | - | 3 | 2 | 0 | 1 | 0 | — |
| 22 | `dct_v311_blca_uni_v3_smoke` | 0.114 | dct_v311_slot_interpretable | blca | - | 3 | 2 | 0 | 1 | 0 | — |
| 23 | `dct_v315_blca_uni2h` | 0.075 | dct_v315_residual_transport | blca | - | 3 | 12 | 0 | 6 | 4 | 活跃脚本 唯一产物 |
| 24 | `dct_v315_rti_rank_blca_uni2h` | 0.063 | dct_v315_residual_transport | blca | - | 3 | 10 | 0 | 5 | 5 | 活跃脚本 唯一产物 |
| 25 | `kaplan_meier_analysis` | 0.001 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 26 | `e4_audits` | 0.001 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 27 | `e4_v311_fixed_audit` | 0.001 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 28 | `v310_proof_b_anchor_geometry_trajectory` | 0.001 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 29 | `v310_proof_a_cross_cancer_audit` | 0.000 | - | hnsc,lusc,skcm | - | - | 0 | 0 | 0 | 0 | — |
| 30 | `v310_proof1_gain_trajectory` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 31 | `dct_v312_stad_5fold` | 0.000 | dct_v312_slot_interpretable_impute | blca,stad | - | 3 | 0 | 0 | 0 | 0 | — |
| 32 | `audit_hnsc` | 0.000 | - | hnsc | - | - | 10 | 5 | 0 | 0 | 唯一产物 |
| 33 | `dct_v312_ucec_5fold` | 0.000 | dct_v312_slot_interpretable_impute | blca,ucec | - | 3 | 0 | 0 | 0 | 0 | — |
| 34 | `v310_proof5_epoch_curve_stats` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 35 | `transport_fix_minimal` | 0.000 | - | blca | - | - | 0 | 0 | 1 | 1 | 唯一产物 快照 |
| 36 | `audit_skcm` | 0.000 | - | skcm | - | - | 10 | 5 | 0 | 0 | 唯一产物 |
| 37 | `dct_v3.10_experiments` | 0.000 | dct_transport_intervention_consist | blca | - | 3 | 0 | 0 | 0 | 0 | 快照 |
| 38 | `v310_proof_d_per_slot_hazard` | 0.000 | - | - | - | - | 1 | 1 | 0 | 0 | 唯一产物 |
| 39 | `v310_proof_f_v311_audit` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 40 | `audit_lusc` | 0.000 | - | lusc | - | - | 7 | 5 | 0 | 0 | 唯一产物 |
| 41 | `audit_blca` | 0.000 | - | blca | - | - | 5 | 5 | 0 | 0 | 唯一产物 快照 |
| 42 | `audit_kirc` | 0.000 | - | kirc | - | - | 4 | 2 | 0 | 0 | 唯一产物 |
| 43 | `audit_best_epochs_blca` | 0.000 | - | - | - | - | 5 | 5 | 0 | 0 | 唯一产物 快照 |
| 44 | `audit_blca_redesign` | 0.000 | - | blca | - | - | 5 | 5 | 0 | 0 | 唯一产物 |
| 45 | `ideal_anchors` | 0.000 | - | - | - | - | 5 | 5 | 0 | 0 | 唯一产物 快照 |
| 46 | `e049_blca_dryrun` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 47 | `v310_proof_c_audit` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 48 | `STAT_TESTS_20261005` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 49 | `proof_D_visualization` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 50 | `audit_comparison` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 51 | `v311_fixed_proof` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 52 | `small_test` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 53 | `v310_proof2_gradient_flow` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 54 | `v310_proof_synthesis` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 55 | `v310_proof3_anchor_distance` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 56 | `v310_proof2bis_per_slot_grad` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | — |
| 57 | `dct_v311_blca_uni_fixed_v2` | 0.000 | dct_v311_slot_interpretable | blca | - | 3 | 0 | 0 | 0 | 0 | — |
| 58 | `statistical_analysis` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 59 | `dct_v3.11` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |
| 60 | `dct_v312_cptac_luad_5fold` | 0.000 | dct_v312_slot_interpretable_impute | blca | - | 3 | 0 | 0 | 0 | 0 | — |
| 61 | `dct_v312_cptac_lusc_5fold` | 0.000 | dct_v312_slot_interpretable_impute | blca | - | 3 | 0 | 0 | 0 | 0 | — |
| 62 | `dct_v312_gbmlgg_5fold` | 0.000 | dct_v312_slot_interpretable_impute | blca | - | 3 | 0 | 0 | 0 | 0 | — |
| 63 | `anchor_diagnostics` | 0.000 | - | - | - | - | 0 | 0 | 0 | 0 | 快照 |

## 逐条删除理由

| # | 路径 | 体积 | 删除理由 |
|---|---|---|---|
| 1 | `dct_v3.10` | 3.812 GiB | v3.10 方法产物；v3.13 依赖该方法代码，但结果本身已被 v3.13 消融与六队列结果取代【注意】；含全仓唯一预测/权重 25+ 份；历史快照引用 36 处 |
| 2 | `dct_v311_blca_uni_fixed_pancancer` | 3.411 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代【注意】；含全仓唯一预测/权重 30+ 份 |
| 3 | `dct_v3.10_uni2h_extend` | 2.962 GiB | v3.10 方法产物；v3.13 依赖该方法代码，但结果本身已被 v3.13 消融与六队列结果取代【注意】；含全仓唯一预测/权重 12+ 份 |
| 4 | `dct_v3.2` | 2.522 GiB | v3.2 TGSR 方法产物；非 v3.13 主线，成果已归档于 历史备份 E020/E021【注意】；历史快照引用 5 处 |
| 5 | `dct_v311_blca_uni_fixed` | 1.254 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代【注意】；含全仓唯一预测/权重 1+ 份；历史快照引用 6 处 |
| 6 | `dct_v316_slot_mi_blca_uni2h` | 0.709 GiB | v3.16 Slot-MI 方法产物；非 v3.13 主线，代码未证明严格 MI/PID 分解【注意】；含全仓唯一预测/权重 5+ 份 |
| 7 | `dct_v314_blca_uni2h` | 0.701 GiB | v3.14 方法产物；非 v3.13 主线，BLCA UNI2-h 五折均值低于 v3.13【注意】；被活跃脚本 scripts/run_blca_uni2h_5fold.py 引用；含全仓唯一预测/权重 4+ 份 |
| 8 | `dct_v312_skcm_5fold` | 0.572 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案【注意】；含全仓唯一预测/权重 9+ 份 |
| 9 | `dct_v312_blca_uni_fold0_aligned` | 0.572 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案【注意】；含全仓唯一预测/权重 4+ 份 |
| 10 | `dct_v3.10_uni_blca_p2048_30ep` | 0.569 GiB | v3.10 方法产物；v3.13 依赖该方法代码，但结果本身已被 v3.13 消融与六队列结果取代 |
| 11 | `dct_v3.10_uni_all10` | 0.569 GiB | v3.10 方法产物；v3.13 依赖该方法代码，但结果本身已被 v3.13 消融与六队列结果取代 |
| 12 | `dct_v311_blca_uni_v2` | 0.569 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代 |
| 13 | `dct_v3.2_smoke` | 0.504 GiB | v3.2 TGSR 方法产物；非 v3.13 主线，成果已归档于 历史备份 E020/E021 |
| 14 | `transport_candidates` | 0.239 GiB | 风险排序方法产物；非 v3.13 主线 |
| 15 | `transport_candidates_smoke` | 0.238 GiB | 风险排序方法产物；非 v3.13 主线 |
| 16 | `dct_v311_blca_uni2h_fixed_v2` | 0.238 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代 |
| 17 | `dct_v311_blca_uni2h` | 0.238 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代【注意】；历史快照引用 1 处 |
| 18 | `dct_v312_blca_uni_smoke` | 0.229 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案【注意】；含全仓唯一预测/权重 1+ 份 |
| 19 | `dct_v311_blca_uni_v2_partial_backup_0742` | 0.228 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代 |
| 20 | `dct_v311_blca_uni` | 0.228 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代【注意】；历史快照引用 6 处 |
| 21 | `dct_v3.10_experiments_smoke` | 0.119 GiB | 干预一致性实验产物；v3.10 实验分支的历史运行结果 |
| 22 | `dct_v311_blca_uni_v3_smoke` | 0.114 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代 |
| 23 | `dct_v315_blca_uni2h` | 0.075 GiB | v3.15 RTI 方法产物；非 v3.13 主线，缺同模型无交互对照【注意】；被活跃脚本 scripts/run_blca_uni2h_5fold.py 引用；含全仓唯一预测/权重 4+ 份 |
| 24 | `dct_v315_rti_rank_blca_uni2h` | 0.063 GiB | v3.15 RTI 方法产物；非 v3.13 主线，缺同模型无交互对照【注意】；被活跃脚本 scripts/run_blca_uni2h_5fold.py 引用；含全仓唯一预测/权重 5+ 份 |
| 25 | `kaplan_meier_analysis` | 0.001 GiB | 统计分析/图表中间产物；统计结果已固化在历史报告与 E0xx 档案【注意】；历史快照引用 5 处 |
| 26 | `e4_audits` | 0.001 GiB | 统计分析/图表中间产物；统计结果已固化在历史报告与 E0xx 档案【注意】；历史快照引用 8 处 |
| 27 | `e4_v311_fixed_audit` | 0.001 GiB | 统计分析/图表中间产物；统计结果已固化在历史报告与 E0xx 档案【注意】；历史快照引用 3 处 |
| 28 | `v310_proof_b_anchor_geometry_trajectory` | 0.001 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 29 | `v310_proof_a_cross_cancer_audit` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 30 | `v310_proof1_gain_trajectory` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 31 | `dct_v312_stad_5fold` | 0.000 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案 |
| 32 | `audit_hnsc` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份 |
| 33 | `dct_v312_ucec_5fold` | 0.000 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案 |
| 34 | `v310_proof5_epoch_curve_stats` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 35 | `transport_fix_minimal` | 0.000 GiB | 冒烟测试产物；无正式实验价值，重复运行即可再生【注意】；含全仓唯一预测/权重 1+ 份；历史快照引用 1 处 |
| 36 | `audit_skcm` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份 |
| 37 | `dct_v3.10_experiments` | 0.000 GiB | 干预一致性实验产物；v3.10 实验分支的历史运行结果【注意】；历史快照引用 26 处 |
| 38 | `v310_proof_d_per_slot_hazard` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 1+ 份 |
| 39 | `v310_proof_f_v311_audit` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 40 | `audit_lusc` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份 |
| 41 | `audit_blca` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份；历史快照引用 1 处 |
| 42 | `audit_kirc` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 2+ 份 |
| 43 | `audit_best_epochs_blca` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份；历史快照引用 8 处 |
| 44 | `audit_blca_redesign` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份 |
| 45 | `ideal_anchors` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；含全仓唯一预测/权重 5+ 份；历史快照引用 6 处 |
| 46 | `e049_blca_dryrun` | 0.000 GiB | 冒烟测试产物；无正式实验价值，重复运行即可再生 |
| 47 | `v310_proof_c_audit` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 48 | `STAT_TESTS_20261005` | 0.000 GiB | 统计分析/图表中间产物；统计结果已固化在历史报告与 E0xx 档案 |
| 49 | `proof_D_visualization` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；历史快照引用 3 处 |
| 50 | `audit_comparison` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；历史快照引用 2 处 |
| 51 | `v311_fixed_proof` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；历史快照引用 3 处 |
| 52 | `small_test` | 0.000 GiB | 冒烟测试产物；无正式实验价值，重复运行即可再生【注意】；历史快照引用 3 处 |
| 53 | `v310_proof2_gradient_flow` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 54 | `v310_proof_synthesis` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 55 | `v310_proof3_anchor_distance` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 56 | `v310_proof2bis_per_slot_grad` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中 |
| 57 | `dct_v311_blca_uni_fixed_v2` | 0.000 GiB | v3.11 方法产物；v3.13 继承该模型代码，但结果已被 v3.13 消融与六队列结果取代 |
| 58 | `statistical_analysis` | 0.000 GiB | 统计分析/图表中间产物；统计结果已固化在历史报告与 E0xx 档案【注意】；历史快照引用 3 处 |
| 59 | `dct_v3.11` | 0.000 GiB | 空目录或仅含占位文件；无实质产物【注意】；历史快照引用 1 处 |
| 60 | `dct_v312_cptac_luad_5fold` | 0.000 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案 |
| 61 | `dct_v312_cptac_lusc_5fold` | 0.000 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案 |
| 62 | `dct_v312_gbmlgg_5fold` | 0.000 GiB | v3.12 方法产物；非 v3.13 主线，v3.13 未继承其重建方案 |
| 63 | `anchor_diagnostics` | 0.000 GiB | 审计/证明实验中间产物；结论已固化在 历史备份/实验档案 的 E0xx 报告中【注意】；历史快照引用 2 处 |

## 字段说明

- **method**：目录内出现过的 SurvOTRank_<method> 标识
- **arm**：目录层级中的实验臂标识（native/learned/independent/control/masked_plain/gain_calibrated/direction_only/ipcw_only/full 等）
- **cohort**：出现的癌种目录名
- **fold**：目录名中解析出的 fold 编号
- **seed**：目录名中解析出的 seed
- **pred_pkl**：*.pkl 预测文件数
- **pred_pkl_repo_unique**：其中在 results/ + 实验档案/ + 历史备份/ 全仓范围内仅此一份的预测文件数
- **ckpt**：*.pth/*.pt/*.ckpt 权重文件数
- **ckpt_repo_unique**：其中全仓仅此一份的权重文件数
- **has_repo_unique_artifact**：是否存在全仓唯一的预测或权重（删除后不可从仓内任何位置恢复）
- **active_script_refs**：被仓库内非归档 scripts/ 读取的路径
- **snapshot_refs_count**：被 历史备份/ 或 实验档案/ 快照引用的次数
- **doc_refs_count**：被其他文档引用的次数
- **uniqueness_method**：以 (文件大小, 文件名) 为键在全仓三处根目录统计出现次数，count==1 判为全仓唯一；未做内容哈希比对，为同尺寸同名的近似判定。

## 需单独决定的三类

### A. 被活跃脚本引用（3 项 / 0.839 GiB）

`scripts/run_blca_uni2h_5fold.py` 直接读取下列结果目录。该脚本本身不在任何删除集合内，删除后其 v3.14/v3.15/v3.15r 三个 arm 失效。

| 路径 | GiB |
|---|---|
| `dct_v314_blca_uni2h` | 0.701 |
| `dct_v315_blca_uni2h` | 0.075 |
| `dct_v315_rti_rank_blca_uni2h` | 0.063 |

### B. 含全仓唯一预测或权重（21 项）

删除后无法从仓内任何位置恢复。是否已固化在实验档案/论文表格中，需逐条确认。

| 路径 | GiB | 唯一预测 | 唯一权重 |
|---|---|---|---|
| `dct_v3.10` | 3.812 | 0 | 25 |
| `dct_v311_blca_uni_fixed_pancancer` | 3.411 | 4 | 26 |
| `dct_v3.10_uni2h_extend` | 2.962 | 0 | 12 |
| `dct_v311_blca_uni_fixed` | 1.254 | 1 | 0 |
| `dct_v316_slot_mi_blca_uni2h` | 0.709 | 0 | 5 |
| `dct_v314_blca_uni2h` | 0.701 | 0 | 4 |
| `dct_v312_skcm_5fold` | 0.572 | 4 | 5 |
| `dct_v312_blca_uni_fold0_aligned` | 0.572 | 0 | 4 |
| `dct_v312_blca_uni_smoke` | 0.229 | 0 | 1 |
| `dct_v315_blca_uni2h` | 0.075 | 0 | 4 |
| `dct_v315_rti_rank_blca_uni2h` | 0.063 | 0 | 5 |
| `audit_hnsc` | 0.000 | 5 | 0 |
| `transport_fix_minimal` | 0.000 | 0 | 1 |
| `audit_skcm` | 0.000 | 5 | 0 |
| `v310_proof_d_per_slot_hazard` | 0.000 | 1 | 0 |
| `audit_lusc` | 0.000 | 5 | 0 |
| `audit_blca` | 0.000 | 5 | 0 |
| `audit_kirc` | 0.000 | 2 | 0 |
| `audit_best_epochs_blca` | 0.000 | 5 | 0 |
| `audit_blca_redesign` | 0.000 | 5 | 0 |
| `ideal_anchors` | 0.000 | 5 | 0 |

### C. 无任何引用且无唯一产物（2 项 / 0.228 GiB）

| 路径 | GiB |
|---|---|
| `dct_v311_blca_uni_v2_partial_backup_0742` | 0.228 |
| `STAT_TESTS_20261005` | 0.000 |

