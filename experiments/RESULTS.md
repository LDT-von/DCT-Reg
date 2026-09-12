# 结果状态

当前没有可被登记为 DCT v3.10 正式结果的完整证据包。

只有当一次运行同时满足冻结协议、外层持出评估和 manifest 完整性，才可把 `REGISTRY.csv` 的状态从 `pending` 改为 `complete`，并在这里加入逐折表格和证据路径。结构测试、smoke run、旧 v3.8.2 分数或只有最佳验证 C-index 的目录都不得填入正式结果表。

## 2026-08-26 新增远端报告审查

SurvOT-Rank commit `271466d` 新增了一份 direction-loss 两折验证汇总。该报告已保存到 `candidate_evidence/`，但因缺少原始工件，且其配置声明与 v3.10 冻结类的强制 `direction=0.05` 行为冲突，状态为 `quarantined`。其中 BLCA/LUSC 的正差值只能作为重跑优先级线索，不能写入论文正式结果或称为统计显著。

## 2026-09-10 KIRC 补全至 5 折 + 全表复现校验

本节记录运行事实与复现证据，**不改变**上文"无完整证据包"的结论：外层持出评估与 manifest 完整性仍未完成，`REGISTRY.csv` 中 `E2_final_30fold` 维持 `pending`。

### 1. 复现校验（与既有产物逐位比对）

以修复后的当前仓库代码（commit `e928359`）经官方入口 `scripts/run_dct_v310_final_cross_cancer.py` 重跑，与 `results/dct_v3.10/robust/final_50ep_old/` 下的既有产物逐位比对（best-epoch 口径）：

| 癌种 | 比对范围 | 结果 |
|---|---|---|
| BLCA | 5 折 × 50 epochs | 逐位一致 |
| HNSC | 5 折 × 50 epochs | 逐位一致 |
| LUSC | 5 折 × 50 epochs | 逐位一致 |
| SKCM | 5 折 × 50 epochs | 逐位一致 |
| KIRC | fold0 / fold1 | 逐位一致 |

同日修复的官方入口缺陷（详见 commit `e928359`）：`--dct_v382_lambda_mgptr` / `--dct_v382_adaptive_aux_weights` 被误删导致所有配置与脚本无法启动、`--fixed_anchors_path` 未定义导致 `dct_v310_fixed_anchors` 消融不可运行、`max_epochs` 被改为 30 导致 cosine 调度偏离复现配方。

### 2. KIRC 5 折结果

| fold | best epoch | val C-index |
|---|---|---|
| 0 | 12 | 0.8497483824586628 |
| 1 | 18 | 0.8660589060308556 |
| 2 | 13 | 0.8038147138964578 |
| 3 | 16 | 0.8335854765506808 |
| 4 | 22 | 0.7818717759764185 |
| **mean ± std** | — | **0.8270158509826151 ± 0.0341341778323786** |

### 3. 全表口径变更

| 项 | 变更前 | 变更后 |
|---|---|---|
| KIRC | 2 折，0.8579 | 5 折，0.8270 |
| OVERALL | 22 折，0.6814 | 25 折，0.6964 |

**⚠️ KIRC 旧值 0.8579 系仅取 fold0/fold1 的乐观估计，已作废**，后续引用一律用 0.8270 ± 0.0341。

当前 5 癌种覆盖（best-epoch 口径，来源 `results/dct_main_table_cindex.json`）：

| 癌种 | 折数 | C-index |
|---|---|---|
| BLCA | 5 | 0.7208 ± 0.0162 |
| HNSC | 5 | 0.6471 ± 0.0713 |
| KIRC | 5 | 0.8270 ± 0.0341 |
| LUSC | 5 | 0.6313 ± 0.0545 |
| SKCM | 5 | 0.6556 ± 0.0473 |
| **合计** | **25** | **0.6964** |

### 4. 待办与遗留观察

- **UCEC 5 折缺失**：`E2_final_30fold` 目标为 6 癌种 × 5 折 = 30 折，现为 25 折。
- **KIRC fold2 的 IBS 异常**：该折 val C-index 正常（0.8038），但 best-epoch `val_IBS = 0.6893`，远高于其余四折（0.107~0.174）。已如实记录，未作解释，需在引用 IBS 前排查。
- **队列中断后必须清理 `.split_*.dct_reg.lock`**：该锁为 pidfile 式且不做存活检测，残留会导致对应折被静默跳过（本次 KIRC fold3 曾因此被跳过一轮）。
