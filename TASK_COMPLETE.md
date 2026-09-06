# 修复完成 - 审计字段错误 & 读取器消融实验

## ✅ 任务完成

### 1. 修复审计字段错误
**文件**: `scripts/e4_continuous_intervention_audit_v2.py`

**问题**:
- 变量名不一致：`original_embeddings` vs `original_embedding`
- 不必要的批次循环（batch_size=1）
- 数组索引错误

**修复**:
- ✓ 统一所有变量名为单数形式
- ✓ 移除不必要的 `for i in enumerate()` 循环
- ✓ 直接使用单个值而非数组索引

### 2. 创建读取器消融实验
**目的**: 确认 pair-context 旁路是否压制了 OT 计划

**测试模式**:
- `full`: 完整模型（pair-context + plan encoding）
- `pair_context_only`: 仅 slot 对交互，移除 plan
- `plan_only`: 仅 plan 编码，移除 pair-context

**判断标准**:
- 如果 `pair_context_only ≈ full` → OT 计划被压制 ❌
- 如果 `|full - pair_context_only| > 0.01` → OT 计划有效 ✓

### 3. 创建方向一致性审计
**目的**: 验证方向正则化约束是否生效

**测试方法**:
- 对每个患者，在 α ∈ [0, 1] 插值到低/高风险锚点
- 检查风险预测的单调性

**判断标准**:
- 低风险方向：风险应单调递减
- 高风险方向：风险应单调递增
- 期望单调性率 > 80%

### 4. 创建小规模生死门测试
**目的**: 快速验证核心功能，决定是否继续大规模实验

**通过标准**:
1. 读取器消融：`|full - pair_context_only| > 0.01`
2. 方向审计：单调性率 > 60%

**时间**: ~15 分钟

## 📋 创建的文件

### 实验脚本
1. ✓ `scripts/run_small_gate_test.py` - 生死门测试
2. ✓ `scripts/run_reader_ablation_experiments.sh` - 完整读取器消融
3. ✓ `scripts/run_e4_direction_audit.sh` - 完整方向审计
4. ✓ `run_all_experiments.sh` - 主执行流程
5. ✓ `quick_test.sh` - 快速启动脚本
6. ✓ `check_environment.sh` - 环境检查

### 文档
1. ✓ `README_EXPERIMENTS.md` - 实验使用说明
2. ✓ `EXPERIMENT_PLAN.md` - 详细实验计划
3. ✓ `EXPERIMENT_STATUS.md` - 实验状态跟踪

## 🚀 如何运行

### 推荐流程：先检查环境
```bash
cd /data1/DCT-Reg
bash check_environment.sh
```

### 快速测试（生死门）
```bash
bash quick_test.sh
```
- 时间：~15 分钟
- 测试 1 fold 的读取器消融和方向审计
- 决定是否继续大规模实验

### 完整实验流程
```bash
bash run_all_experiments.sh
```
- 时间：~2-3 小时
- 自动运行所有 5 fold 实验
- 生成完整汇总报告

## 📊 预期输出

### 生死门测试结果
```
results/small_test/
├── ablation_test.json          # 读取器消融结果
└── audit_test.csv              # 方向审计结果
```

### 完整实验结果
```
results/reader_ablation/
├── blca_fold0_ablation.json
├── blca_fold1_ablation.json
├── ...
└── blca_fold4_ablation.json

results/e4_audit_direction_only/
├── blca_fold0_audit.csv
├── blca_fold1_audit.csv
├── ...
├── blca_fold4_audit.csv
└── blca_summary.csv            # 汇总统计
```

## ⚠️ 关键决策点

### 如果生死门测试通过 ✓
- OT 计划有效贡献（未被旁路压制）
- 方向约束生效（单调性良好）
- **建议**: 继续运行大规模实验

### 如果读取器消融失败 ❌
- `pair_context_only ≈ full`
- OT 计划被压制
- **修复**:
  - 减小 pair-context 特征维度
  - 增强 plan encoding
  - 添加 plan 正则化

### 如果方向审计失败 ❌
- 单调性率 < 60%
- 方向约束未生效
- **修复**:
  - 增加 `dct_v38_lambda_direction`
  - 延长 warmup epochs
  - 检查锚点初始化

## ✅ 环境检查结果

```
✓ Python: 3.10.20
✓ PyTorch: 2.12.0.dev (CUDA 12.8)
✓ CUDA: NVIDIA GeForce RTX 5090
✓ 数据路径: 正常
✓ 检查点: 5/5 folds 可用
✓ 所有脚本: 就绪
```

## 🎯 下一步行动

**立即执行**:
```bash
cd /data1/DCT-Reg
bash quick_test.sh
```

**根据结果**:
- ✓ 通过 → `bash run_all_experiments.sh`
- ❌ 未通过 → 查看 `results/small_test/` 诊断问题

## 📝 备注

- 所有脚本已设置为可执行
- 默认使用 GPU 0（可在脚本中修改）
- 批次大小已优化为 1（避免内存问题）
- 包含完整的错误处理和进度报告
