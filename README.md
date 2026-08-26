# DCT-Reg v3.10

这是 DCT v3.10 的独立、瘦身、论文导向仓库。它只保留冻结方法、必要运行时、正式实验协议、证据登记和唯一论文草稿。

## 它解决什么问题

常规 C-index 和漂亮的 OT 热图不能证明病理—通路运输结构真的参与了预后计算。DCT-Reg 把模型内部的运输代价沿训练折估计的低风险/高风险锚点进行干预，在相同边际下重新求解 Sinkhorn，再读取风险变化。它要检验的是：

> 病理—通路 OT 结构是否承载预后语义；改变运输代价并重新求解 OT 后，是否产生方向正确、依赖重运输、可跨折与跨癌种复现的风险响应。

冻结训练目标为：

\[
L = L_{NLL} + 0.10 L_{IPCW-rank} + 0.05 L_{direction}.
\]

这不是治疗反事实或因果效应模型。“干预”只指冻结模型内部的运输代价干预。

## 当前结论

- **代码结构已经实现**：事实 OT、低/高风险代价干预、重新 Sinkhorn、共享风险读取和方向损失均在同一前向链中。
- **结构验证不能等同于科学结论**：单元测试只能证明公式和控制项按预期执行。
- **v3.10 正式性能与机制证明尚未完成**：`experiments/REGISTRY.csv` 中的正式任务当前均标为 `pending`。
- **旧 v3.8.2 结果不是 v3.10 结果**：不得改名、合并或作为 v3.10 已完成证据。

## 一分钟入口

```powershell
python -m survot_rank.cli doctor
python scripts/run_dct_v310_final_cross_cancer.py plan
python scripts/run_dct_v310_experiments.py plan
python -m pytest -q
```

训练前必须提供 UNI2-h 特征、临床 CSV 和冻结的 `5fold_uni2h` splits：

```powershell
$env:UNI2H_ROOT = "D:/path/to/TCGA-UNI2-h-features"
$env:DCT_DATA_CSV_ROOT = "D:/path/to/dataset_csv"
python scripts/run_dct_v310_final_cross_cancer.py doctor
```

当前源库未包含 `5fold_uni2h`，本库不会用另一套 split 冒充。正式运行前须将其补齐并在 manifest 中记录哈希。

## 仓库地图

- `survot_rank/research/methods/`：DCT-Reg 及其最小继承链。
- `configs/`：冻结 v3.10 配置。
- `scripts/`：最终 30-fold 队列、消融/对照队列、持出集机制审计。
- `experiments/PROTOCOL.md`：什么实验才能证明核心主张。
- `experiments/REGISTRY.csv`：每个证据包的完成状态。
- `docs/CLAIMS_AND_EVIDENCE.md`：主张、判据和结论边界。
- `docs/MIGRATION_MANIFEST.md`：从原库迁移了什么、刻意没迁移什么。
- `paper/`：`DCT_唯一初稿.md` 是唯一可编辑主稿，DOCX 是交付快照；结果占位符必须由正式证据包填写。

## 正式结果包最低要求

每折必须归档：患者级预测、风险干预轨迹、耦合计划诊断、训练曲线、最终 checkpoint、解析后配置、split 哈希、Git commit、环境信息和 manifest。仅有最佳验证 C-index 不构成独立测试证据。
