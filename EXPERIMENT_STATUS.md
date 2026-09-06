## 实验状态摘要

**日期**: 2026-09-05

### 已完成
1. ✓ 修复 `e4_continuous_intervention_audit_v2.py` 中的字段错误
   - 变量名不一致问题
   - 批次处理逻辑错误
2. ✓ 创建读取器消融实验脚本
3. ✓ 创建方向一致性审计脚本
4. ✓ 创建小规模生死门测试
5. ✓ 创建主执行脚本

### 待运行实验

#### 优先级 1: 生死门测试（必须先过）
```bash
python scripts/run_small_gate_test.py
```
- 预计时间: 10-15 分钟
- 判断标准:
  - 读取器: |full - pair_context| > 0.01
  - 方向: 单调性 > 60%

#### 优先级 2: 完整消融（如果通过生死门）
```bash
bash scripts/run_reader_ablation_experiments.sh
```
- 预计时间: 30-40 分钟
- 5 fold × 3 模式

#### 优先级 3: 完整审计（如果通过生死门）
```bash
bash scripts/run_e4_direction_audit.sh
```
- 预计时间: 1-2 小时
- 5 fold × 11 alpha 值

### 关键问题待验证

1. **OT 计划是否被旁路压制？**
   - 通过读取器消融实验回答
   - 如果 `pair_context_only ≈ full`，说明被压制

2. **方向约束是否生效？**
   - 通过方向审计实验回答
   - 期望单调性率 > 80%

### 下一步行动

**立即运行**:
```bash
cd /data1/DCT-Reg
python scripts/run_small_gate_test.py
```

**根据结果**:
- 通过 → 运行 `bash run_all_experiments.sh`
- 未通过 → 诊断并修复问题

### 预期时间线

- 生死门测试: 15 分钟
- 完整实验（如果通过）: 2-3 小时
- 总计: 2.5-3.5 小时
