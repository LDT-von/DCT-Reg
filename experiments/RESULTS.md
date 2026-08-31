# 结果状态

## 2026-08-29 首轮正式证明运行（30-epoch 预算）

P1–P5 全部正式运行完成，工件在 `results/dct_v3.10_experiments/robust/`，逐折表格见 `transport_dependency_proof/RESULTS_20260829.md`。训练 C-index 正常（blca full 0.638）；审计侧 P3（计划替换 → 风险几乎不变）、P4（方向一致率低于机会）、P5（剂量单调率远低于通过线）结果异常，**排查审计解码路径前不写入正式结论**。

P1 消融摘要（blca 五折均值 C-index）：nll_only 0.629 / ipcw_only 0.591 / direction_only 0.570 / full 0.638。

## 2026-08-28 建立证明包

`transport_dependency_proof/` 中 P0 代码门已实现，正式运行此前未完成，P7 尚未实现。该状态不会升级任何性能或机制主张。

只有当一次运行同时满足冻结协议、外层持出评估和 manifest 完整性，才可把 `REGISTRY.csv` 的状态从 `pending` 改为 `complete`，并在这里加入逐折表格和证据路径。结构测试、smoke run、旧 v3.8.2 分数或只有最佳验证 C-index 的目录都不得填入正式结果表。

## 2026-08-26 新增远端报告审查

SurvOT-Rank commit `271466d` 新增了一份 direction-loss 两折验证汇总。该报告已保存到 `candidate_evidence/`，但因缺少原始工件，且其配置声明与 v3.10 冻结类的强制 `direction=0.05` 行为冲突，状态为 `quarantined`。其中 BLCA/LUSC 的正差值只能作为重跑优先级线索，不能写入论文正式结果或称为统计显著。
