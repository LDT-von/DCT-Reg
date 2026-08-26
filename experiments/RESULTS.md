# 结果状态

当前没有可被登记为 DCT v3.10 正式结果的完整证据包。

只有当一次运行同时满足冻结协议、外层持出评估和 manifest 完整性，才可把 `REGISTRY.csv` 的状态从 `pending` 改为 `complete`，并在这里加入逐折表格和证据路径。结构测试、smoke run、旧 v3.8.2 分数或只有最佳验证 C-index 的目录都不得填入正式结果表。

## 2026-08-26 新增远端报告审查

SurvOT-Rank commit `271466d` 新增了一份 direction-loss 两折验证汇总。该报告已保存到 `candidate_evidence/`，但因缺少原始工件，且其配置声明与 v3.10 冻结类的强制 `direction=0.05` 行为冲突，状态为 `quarantined`。其中 BLCA/LUSC 的正差值只能作为重跑优先级线索，不能写入论文正式结果或称为统计显著。
