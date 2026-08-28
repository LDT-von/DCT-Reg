# DCT-Reg 工作记录索引

本目录保存开发与实验过程中的工作记录。它不是正式论文证据区；只有满足
`experiments/PROTOCOL.md` 的运行，才可以登记到
`experiments/REGISTRY.csv` 和 `experiments/RESULTS.md`。

运输依赖证明的细化状态统一见 `experiments/transport_dependency_proof/`。

## 文件分工

| 文件 | 记录内容 | 不应写入 |
|---|---|---|
| `ISSUES.md` | bug、风险、复现问题、修复验证 | 未经核验的性能结论 |
| `EXPERIMENTS.md` | 待跑、运行中、失败及候选实验 | 冒充正式结果的旧分数 |
| `IDEAS.md` | idea、创新边界、威胁和后续假设 | 已冻结方法的静默改动 |
| `DECISIONS.md` | 冻结决定、口径变化、证据升降级理由 | 临时猜测 |

## 追加规则

每条记录至少写明日期、状态、证据路径和下一步。建议状态统一使用：
`open`、`in_progress`、`blocked`、`verified`、`rejected`。

正式实验结果仍只写入 `experiments/RESULTS.md`；本目录可以先记录运行过程，
但不得把 smoke test、最佳验证分数或缺少 manifest 的结果写成论文结论。
