# 实验与证据缺口记录

## 2026-08-28 运输依赖证明包

正式证明实验尚未运行完成。集中入口为 `experiments/transport_dependency_proof/`，机器可读状态见其中 `MATRIX.csv`。

| 实验 | 已完成的实现 | 仍缺少 | 执行状态 |
|---|---|---|---|
| P0 代码门 | 冻结目标、确定性 eval、关键梯度、真实计划导出、Plan TV、证据递归哈希测试 | 正式环境重跑日志归档 | completed（结构证据） |
| P1 BLCA 2×2 | 4 variants × 5 folds 队列 | 正式运行、逐折配对差、bootstrap CI | pending |
| P2 re-Sinkhorn | full/fixed-coupling 队列和计划导出 | 预注册 folds 的风险响应衰减与 CI | pending |
| P3 factual plan | uniform/shuffled 可行计划、相同读取器解码 | held-out 风险/排序/性能配对分析 | pending |
| P4 anchor specificity | noisy/permuted/jitter 队列、anchor-swap 审计 | 跨 fold frozen anchor；正式运行 | pending |
| P5 双向剂量 | high/low 两侧 alpha sweep 与 DMR | held-out 患者级曲线和 CI | pending |
| P6 六癌复现 | 30-fold 固定预算 outer-once 队列和证据包 | 正式运行、异质性、多重校正 | pending |
| P7 外部性 | 协议 | 独立机构数据、锁定推理和审计 | not_implemented |

## 执行顺序

1. 在 clean commit 上运行 doctor、单元测试和 BLCA fold-0 smoke。
2. 完成 smoke → checkpoint → factual/uniform/shuffled/anchor-swap/dose 的闭环。
3. 跑 P1 BLCA 五折 2×2；若 full 未达到预测非劣门，不进入大规模 Final。
4. 跑 P2-P5 的预注册机制门；如果 factual plan 替换后风险与排序几乎不变，弱化核心主张。
5. 机制门通过后运行 P6 六癌 30 folds。
6. P7 完成前只写内部跨队列泛化。

## 下一次运行记录模板

### YYYY-MM-DD / experiment_id

- 状态：planned / running / failed / candidate / complete
- Git commit：
- 数据与 split hash：
- 命令：
- 运行目录：
- 完整性检查：
- 主要观察（候选，不自动进入论文）：
- 失败与异常：
- 是否可升级到 `experiments/REGISTRY.csv`：否 / 是（理由）

