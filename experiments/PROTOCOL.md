# DCT-Reg 正式证明总协议

本文件只定义全局证据门。运输计划是否真正承担事实预测与干预响应的详细预注册协议，统一维护在：

- `experiments/transport_dependency_proof/PROTOCOL.md`
- `experiments/transport_dependency_proof/MATRIX.csv`
- `experiments/transport_dependency_proof/STATUS.md`

统一入口：

```powershell
python experiments/transport_dependency_proof/plan.py status
python experiments/transport_dependency_proof/plan.py plan
```

## 全局冻结门

正式运行前冻结 clean Git commit、解析后配置、患者级 splits 及哈希、终点、风险定义、固定 50 epoch 预算、统计方法和通过线。锚点、删失参考和时间分箱只能由当前训练折估计。

当前协议采用固定预算锁定交叉验证：训练期间不查看 held-out fold，epoch 50 结束后只评估一次。它不能被称为独立机构测试；外部临床泛化只能由独立队列支持。

## 证据层级

1. P0 单元测试和结构检查只证明代码链存在、可微、有限和确定。
2. P1-P5 的持出实验用于证明目标贡献、事实 plan 必要性、re-Sinkhorn 必要性、锚点特异性和双向剂量结构。
3. P6 六癌种锁定运行用于内部跨队列复现。
4. P7 独立机构数据完成前，不得宣称外部临床泛化。

## 每折最低工件

```text
predictions.csv
training_curve.csv
checkpoint.pt
resolved_config.yaml
split_manifest.json
environment.json
run_manifest.json
proof_transport_dependency/factual/audit_cases.pkl
proof_transport_dependency/factual/audit_metrics.json
proof_transport_dependency/uniform_plan/audit_metrics.json
proof_transport_dependency/shuffled_plan/audit_metrics.json
proof_transport_dependency/anchor_swap/audit_metrics.json
proof_transport_dependency/dose_both_directions/dose_metrics.json
```

`run_manifest.json` 必须递归记录所有工件 SHA-256。任一 fold 失败时训练进程必须返回非零，并在 `run_status.json` 中列出失败折。

