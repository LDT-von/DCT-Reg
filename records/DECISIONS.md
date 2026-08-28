# 决策与冻结记录

## 已确认决策

### 2026-08-26 / D-001 记录体系

- 状态：verified
- 决定：使用 4 个主题日志加 1 个索引，共 5 个 `.md` 文件。
- 理由：bug、实验、idea 和冻结决定的证据等级不同，混写容易把候选观察误当正式结果。
- 影响：正式结果仍由 `experiments/REGISTRY.csv` 和 `experiments/RESULTS.md` 管理。

### 2026-08-26 / D-002 正式证据边界

- 状态：verified
- 决定：单元测试、smoke、legacy 两折最佳验证和缺 manifest 的目录都不能升级为 v3.10
  正式结果。
- 证据：`README.md`、`experiments/PROTOCOL.md`、`experiments/RESULTS.md`。

### 2026-08-26 / D-003 方法身份

- 状态：verified
- 决定：论文主方法保持
  `NLL + 0.10 IPCW-rank + 0.05 direction`；dose、reconfiguration、MGPTR、ETAR、
  listwise 和 adaptive auxiliary weighting 不属于冻结 v3.10 目标。

### 2026-08-26 / D-004 因果表述

- 状态：verified
- 决定：只称“模型内部 prognostic ground-cost intervention / directional risk response”，
  不称治疗反事实或患者级因果效应。

### 2026-08-28 / D-005 运输依赖证明包

- 状态：verified
- 决定：将功能必要性证明集中到 `experiments/transport_dependency_proof/`；实现状态与正式执行状态分列记录。
- 依据：计算图依赖不能替代 factual-plan necessity、fixed-coupling、anchor specificity 和双向 dose 的持出证据。
- 影响范围：实验、结果登记、论文主张。

### 2026-08-28 / D-006 正式评估身份

- 状态：verified
- 决定：采用预注册固定 50 epoch 的 locked cross-validation；训练期间不查看 held-out fold，结束后只评估一次。
- 边界：该协议不称独立机构测试；P7 完成前不宣称外部临床泛化。

### 2026-08-28 / D-007 正式证据必须对应 clean commit

- 状态：verified
- 决定：正式配置启用 `formal_require_clean_git`，脏工作区启动证据包时直接失败。
- 理由：仅记录 `git_dirty=true` 不能复现未提交源码。

## 新决策模板

### YYYY-MM-DD / D-XXX 标题

- 状态：proposed / verified / superseded
- 决定：
- 依据：代码、配置、实验路径或会议结论
- 替代方案：
- 影响范围：代码 / 配置 / 实验 / 论文
- 若替代旧决定：被替代编号
