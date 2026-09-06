# v3.2 TGSR 后续实验计划

**更新时间**: 2026-09-06 06:20 UTC
**状态**: ✅ 代码已实现，待服务器执行

---

## ⚠️ 重要前提

**本实验计划严格区分两个独立问题：**

| 问题 | 验证方法 |
|------|---------|
| **预测有效性**：TGSR是否能改善生存预测？ | C-index, IBS, iAUC |
| **方向解释正确性**：低/高风险锚点是否可靠引导风险方向？ | DCR, DMR, Plan TV |

> C-index提升**不能**证明方向假设正确。两者需要分别验证。

---

## ✅ 已实现：TGSR 优化实验框架

**提交**: `58a2852`

**新增文件**：
- `survot_rank/research/methods/dct_v32_tgsr_objective_study/model.py` - 优化模型
- `scripts/run_dct_v32_tgsr_optimization.py` - 实验启动脚本
- `configs/dct_v32_tgsr_objective_study.yaml` - 实验配置
- `tests/test_dct_v32_tgsr_optimization.py` - 回归测试

**验证状态**：
- ✅ 全仓库 55 项测试全部通过
- ✅ 框架自检通过
- ✅ tgsr_nll 与原 TGSR 事实预测路径逐值一致
- ✅ 可学习反馈强度能够获得有限梯度
- ⏳ 待服务器执行生成真实性能分数

---

## 📋 实验队列

**启动命令**：
```bash
python3 scripts/run_dct_v32_tgsr_optimization.py run --variants all
```

**共 30 个任务**（6组 × 5折）

| 变体 | 回答的问题 | 损失函数 |
|------|-----------|---------|
| `tgsr_nll` | 重现 TGSR 基础结果 | NLL only |
| `tgsr_ipcw` | 排序损失能否提高 TGSR | NLL + IPCW 排序 |
| `tgsr_direction` | 方向损失本身是否有用 | NLL + 方向损失 |
| `tgsr_full` | TGSR 与 v3.10 完整目标能否互补 | NLL + IPCW + 方向 |
| `tgsr_full_learned` | 自适应反馈强度能否进一步提高 | 可学习反馈强度 |
| `v310_full_reference` | 相同协议下的 v3.10 公平对照 | v3.10 Full |

### 关键特性

1. **可学习的 OT 反馈强度**
   - 限制在 [0, 1]
   - 初始值 0.25
   - 记录训练过程中的反馈强度变化

2. **固定强度对照**
   - 与可学习版本严格对照
   - 确保增量来自自适应机制

3. **指纹隔离**
   - 使用代码、配置、种子生成指纹
   - 避免错误复用旧结果

---

## 🔬 实验结果解读

### 判断标准

| 条件 | 结论 |
|------|------|
| `tgsr_direction > tgsr_nll` **且** 方向审计通过 | 支持 v3.10 的方向约束思路 |
| `tgsr_full > tgsr_ipcw` | 方向项在排序损失之外仍有贡献 |
| `tgsr_full > v310_full_reference` | TGSR 能够进一步增强 v3.10 |
| C-index 提高 **但** 方向审计失败 | 只能证明预测结构有效，**不能**证明风险方向解释正确 |

### 论文路线选择

```
路线A：方向审计通过
  → 贡献："具有方向响应约束的运输模型"
  → 强因果/解释性主张，需明确适用条件

路线B：方向审计失败，但 TGSR 增益稳定
  → 贡献："运输引导的槽重聚合改善生存预测"
  → 聚焦预测性能，收缩方向解释主张
```

---

## 📁 数据位置（待生成）

```
results/dct_v3.2_tgsr_optimization/
├── tgsr_nll/blca/seed3_*/blca/.../epoch_curve_fold{0-4}.csv
├── tgsr_ipcw/blca/seed3_*/blca/.../epoch_curve_fold{0-4}.csv
├── tgsr_direction/blca/seed3_*/blca/.../epoch_curve_fold{0-4}.csv
├── tgsr_full/blca/seed3_*/blca/.../epoch_curve_fold{0-4}.csv
├── tgsr_full_learned/blca/seed3_*/blca/.../epoch_curve_fold{0-4}.csv
└── v310_full_reference/blca/seed3_*/blca/.../epoch_curve_fold{0-4}.csv
```

---

## 执行清单

- [x] **Step 0**: 实现实验框架代码
- [x] **Step 0.1**: 回归测试全部通过
- [x] **Step 0.2**: 与原 TGSR 预测路径验证一致
- [ ] **Step 1**: 查清 IBS/iAUC 异常原因（可与 Step 2 并行）
- [ ] **Step 2**: 在服务器运行 `run_dct_v32_tgsr_optimization.py run --variants all`
- [ ] **Step 3**: 分析六组实验结果
- [ ] **Step 4**: 方向审计（如 tgsr_direction 预测提升）
- [ ] **Step 5**: 根据结果确定论文路线

---

## ⚠️ 优先级 1：查清指标异常（待完成）

**问题**：OT反馈的 C-index 提高了，但 IBS 变差、iAUC 降低（尤其 Fold 1, 2）

**需要检查**：
1. 生存曲线质量（预测分布是否合理）
2. 评估实现（IBS/iAUC 计算代码）
3. 时间范围设置（time_bins 是否一致）
4. 预测分布变化对指标敏感度的影响

**行动**：先确定原因，才能判断优化是否真的改善模型。
