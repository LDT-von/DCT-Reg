# DCT-Reg 运输依赖证明包

这里是“事实预测是否真正依赖运输计划”的唯一证明实验入口。代码可运行、单元测试通过和正式持出结果是三个不同层级，状态不得互相替代。

## 使用入口

```powershell
python experiments/transport_dependency_proof/plan.py status
python experiments/transport_dependency_proof/plan.py plan
```

`plan` 先列出训练队列；只有在 `results/` 中发现真实的 full/final `checkpoint.pt` 后，才生成 factual、uniform-plan、shuffled-plan、anchor-swap 和双向 dose 审计命令。这样不会用占位 checkpoint 制造“可运行”的假象。

## 文件职责

- `PROTOCOL.md`：冻结问题、对照、统计量和通过线；正式结果出现后不得改判据。
- `MATRIX.csv`：机器可读的实现状态和执行状态。
- `STATUS.md`：当前已完成/未完成事项的人读台账。
- `plan.py`：统一计划与 checkpoint 审计命令生成器。

## 证据边界

当前可证明计算链存在、可微且评估确定；不能据此声称 factual prediction 已被证明依赖 OT。只有 P1-P6 的正式工件通过预注册判据后，才能升级对应主张。

