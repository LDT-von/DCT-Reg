## 任务完成总结

### ✅ 已修复的审计字段错误

**文件**: `scripts/e4_continuous_intervention_audit_v2.py`

修复的变量名不一致问题：
- `original_embeddings` → `original_embedding` (单个样本)
- `low_risk_embeddings` → `low_risk_embedding`
- `high_risk_embeddings` → `high_risk_embedding`
- `patient_ids[i]` → `patient_id` (batch_size=1)
- `true_times[i]` → `true_time`
- `true_events[i]` → `true_event`
- 移除了不必要的 `for i in enumerate(patient_ids)` 循环

### 📝 创建的实验脚本

#### 1. 小规模生死门测试
- **脚本**: `scripts/run_small_gate_test.py`
- **用途**: 在单个 fold 上快速验证核心功能
- **测试内容**:
  - 读取器消融（full / pair_context_only / plan_only）
  - 方向一致性审计（3 个 α 值）
- **时间**: ~15 分钟

#### 2. 完整读取器消融实验
- **脚本**: `scripts/run_reader_ablation_experiments.sh`
- **用途**: 测试 OT 计划是否被 pair-context 旁路压制
- **测试内容**: 5 fold × 3 种模式
- **关键判断**: 如果 `pair_context_only ≈ full`，说明 OT 计划被压制

#### 3. 完整方向一致性审计
- **脚本**: `scripts/run_e4_direction_audit.sh`
- **用途**: 验证方向正则化约束是否生效
- **测试内容**: 5 fold × 11 个 α 值
- **关键判断**: 期望单调性率 > 80%

#### 4. 主执行流程
- **脚本**: `run_all_experiments.sh`
- **用途**: 按正确顺序运行所有实验
- **流程**: 生死门 → 读取器消融 → 方向审计

### 🚀 如何运行

#### 方案 1: 快速测试（推荐）
```bash
cd /data1/DCT-Reg
bash quick_test.sh
```

#### 方案 2: 完整流程
```bash
cd /data1/DCT-Reg
bash run_all_experiments.sh
```

### 🎯 生死门通过标准

**必须同时满足**:
1. **读取器消融**: `|full - pair_context_only| > 0.01`
   - 证明 OT 计划有实际贡献
2. **方向审计**: 单调性率 > 60%
   - 证明方向约束生效

**如果通过** → 可以继续大规模实验  
**如果未通过** → 需要修复核心问题

### 📊 输出位置

- 小规模测试: `results/small_test/`
- 读取器消融: `results/reader_ablation/`
- 方向审计: `results/e4_audit_direction_only/`

### ⚠️ 重要提醒

1. **必须先通过生死门**: 小规模测试不通过，不要运行大规模实验
2. **检查点路径**: 确认 `results/backups/direction_only_frozen_bug_20260903_173509/` 存在
3. **GPU 可用**: 默认使用 `cuda:0`

### 📚 文档

- 详细计划: `EXPERIMENT_PLAN.md`
- 实验状态: `EXPERIMENT_STATUS.md`
