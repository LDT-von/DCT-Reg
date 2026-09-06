# DCT-Reg 实验执行计划

## 概述
修复审计字段错误并设置读取器消融实验，确认 pair-context 旁路是否压制了 OT 计划。

## 已修复的问题

### 1. 审计脚本字段错误
- **文件**: `scripts/e4_continuous_intervention_audit_v2.py`
- **问题**: 变量名不一致
  - 使用了 `original_embeddings` 但只定义了 `original_embedding`
  - 使用了 `patient_ids`, `true_times`, `true_events` 但实际是单个值
  - 使用了数组索引 `[i]` 但 batch_size=1
- **修复**: 统一变量名，去除不必要的循环和索引

### 2. 创建的实验脚本

#### a. 小规模生死门测试
- **文件**: `scripts/run_small_gate_test.py`
- **功能**: 
  - 在单个 fold 上测试读取器消融
  - 在单个 fold 上测试方向一致性
  - 快速验证核心功能
- **通过标准**:
  - 读取器消融: `|full - pair_context_only| > 0.01`
  - 方向审计: 单调性率 > 60%

#### b. 完整读取器消融实验
- **文件**: `scripts/run_reader_ablation_experiments.sh`
- **功能**: 在 5 fold 上运行读取器消融
- **测试模式**:
  - `full`: 完整模型（pair-context + plan encoding）
  - `pair_context_only`: 仅使用 slot 对交互，移除 plan 编码
  - `plan_only`: 仅使用 plan 编码，移除 pair-context
- **关键判断**:
  - 如果 `pair_context_only ≈ full`: OT 计划被旁路压制
  - 如果 `plan_only << full`: 证明 pair-context 是主要贡献

#### c. 完整方向一致性审计
- **文件**: `scripts/run_e4_direction_audit.sh`
- **功能**: 测试方向正则化约束是否生效
- **测试方法**:
  - 对每个患者，在 α ∈ [0, 1] 插值到低风险/高风险锚点
  - 检查风险预测是否单调变化
- **通过标准**:
  - 低风险方向: 风险应单调递减
  - 高风险方向: 风险应单调递增
  - 期望单调性率 > 80%

#### d. 主执行脚本
- **文件**: `run_all_experiments.sh`
- **流程**:
  1. 运行小规模生死门测试
  2. 如果通过，运行完整读取器消融
  3. 如果通过，运行完整方向审计
  4. 汇总结果

## 使用方法

### 快速测试（推荐先运行）
```bash
cd /data1/DCT-Reg
python scripts/run_small_gate_test.py
```

### 完整实验流程
```bash
cd /data1/DCT-Reg
bash run_all_experiments.sh
```

### 单独运行各实验
```bash
# 仅读取器消融
bash scripts/run_reader_ablation_experiments.sh

# 仅方向审计
bash scripts/run_e4_direction_audit.sh
```

## 预期结果

### 理想情况（通过生死门）
- 读取器消融: `full - pair_context_only > 0.02`
- 方向审计: 单调性率 > 80%
- **结论**: 可以继续大规模实验

### 失败情况 1: OT 计划被压制
- 读取器消融: `full ≈ pair_context_only` (差异 < 0.01)
- **问题**: 融合模块过度依赖 slot 对，OT 计划贡献被忽略
- **修复**: 
  - 减小 pair-context 特征维度
  - 增加 plan encoding 的表达能力
  - 添加 plan 正则化损失

### 失败情况 2: 方向约束失效
- 方向审计: 单调性率 < 60%
- **问题**: 方向正则化强度不足
- **修复**:
  - 增加 `dct_v38_lambda_direction`
  - 延长 warmup epochs
  - 检查锚点是否正确初始化

## 输出位置

- **小规模测试**: `results/small_test/`
- **读取器消融**: `results/reader_ablation/`
- **方向审计**: `results/e4_audit_direction_only/`

## 关键指标

### 读取器消融
- C-index (full, pair_context_only, plan_only)
- 相对差异百分比

### 方向审计
- 单调递减率（低风险方向）
- 单调递增率（高风险方向）
- 平均风险变化量 (α=1.0)

## 注意事项

1. **必须先通过生死门**: 小规模测试必须通过才能继续大规模实验
2. **GPU 使用**: 默认使用 `cuda:0`，可在脚本中修改
3. **检查点路径**: 确保检查点路径正确
4. **数据路径**: 确保 CSV 和 WSI 特征路径正确

## 故障排除

### 如果遇到导入错误
```bash
export PYTHONPATH=/data1/DCT-Reg:$PYTHONPATH
```

### 如果遇到 CUDA 内存不足
- 减小 batch size（在审计脚本中已设为 1）
- 使用不同的 GPU

### 如果检查点加载失败
- 检查检查点路径
- 检查配置文件路径
- 确认 fold 编号正确 (0-4)
