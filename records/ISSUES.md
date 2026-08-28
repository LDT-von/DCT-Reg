# 问题与 Bug 记录

## 2026-08-28 证明实验准备审计

### 已解决

| ID | 原问题 | 状态 | 当前证据 |
|---|---|---|---|
| I-001 | audit 返回值解包错误 | resolved | 审计按四项返回值解包并有回归覆盖 |
| I-002 | 每 epoch 查看 held-out fold | resolved_for_locked_protocol | `outer_eval_only` 固定 50 epoch 后只评估一次；协议不再称独立机构测试 |
| I-003 | 折异常被吞掉并返回成功 | resolved | 累计失败折、写 `run_status.json`、进程返回非零 |
| I-004 | 用 cost distance 冒充 Plan TV | resolved | factual/low/high 真实计划导出并直接计算 TV |
| I-005 | 正式证据包未实现 | resolved | config/split/environment/run manifest、预测、checkpoint、曲线和递归 SHA-256 |
| I-006 | 只记录 low 边际误差 | resolved | factual/low/high 分别记录最大边际误差 |
| I-007 | DCR 与训练命题口径混合 | resolved | 同时报告 outcome-stratified 与全病例双向 DCR |
| I-008 | 病例 ID offset 依赖固定 batch | resolved | 使用累计 `case_offset` |
| I-009 | 主稿引用不存在的旧台账 | resolved | 改为当前 claims、registry 和 proof package |
| I-011 | 脏工作区可生成不可复现正式证据 | resolved | final 配置启用 `formal_require_clean_git` |

### 仍开放

#### I-010 CodeGraph 尚未初始化

- 严重程度：P3（维护效率）
- 状态：open
- 影响：不影响训练，但结构追踪效率较低。
- 下一步：经项目负责人明确同意后初始化。

#### I-012 正式统计汇总器尚未实现

- 严重程度：P1（证据完成度）
- 状态：open
- 影响：运行可产出患者级工件，但 paired bootstrap CI、多重比较校正和跨癌异质性仍需统一汇总器。
- 下一步：在首个完整 BLCA proof smoke 工件确定字段后实现，避免对占位 schema 编程。

#### I-013 factual reader 仍含 pair-context 旁路

- 严重程度：P1（科学主张）
- 状态：open_experimental
- 影响：计算图使用 plan 不等于事实预测功能上依赖 plan。
- 下一步：P3 uniform/shuffled 控制；后续增加 full/pair-only/plan-only 读取器消融。

## 本次验证门

- `python -m survot_rank.cli doctor`
- `python experiments/transport_dependency_proof/plan.py status`
- `python experiments/transport_dependency_proof/plan.py plan`
- `python -m pytest -q`

