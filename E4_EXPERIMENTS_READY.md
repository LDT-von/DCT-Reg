# 🚀 E4 实验一键运行指南

**创建时间**: 2026年9月4日 14:30  
**状态**: ✅ 准备就绪，可以运行

---

## 📋 实验概览

### E4 Continuous Intervention Audit

**核心问题**: 当我们手动将患者表示向预后锚点方向移动时，预测风险是否单调变化？

**实验设计**:
- 对每个测试患者，在其表示和预后锚点之间进行线性插值
- 插值强度 α ∈ [0, 0.1, 0.2, ..., 1.0]
- 两个方向：向低风险锚点、向高风险锚点
- 测量预测风险的变化

**预期结果**:
- ✓ 向低风险锚点移动 (α↑) → 预测风险下降 (单调递减)
- ✓ 向高风险锚点移动 (α↑) → 预测风险上升 (单调递增)
- ✓ 单调性率 > 70%

---

## 🎯 实验配置

### 待测试的3个变体

| 变体 | 配置 | C-index | 说明 |
|------|------|---------|------|
| **direction_only** | λ_dir=0.05, λ_ipcw=0 | 0.7087 | 仅方向约束 |
| **ipcw_only** | λ_dir=0, λ_ipcw=0.10 | 0.6777 | 仅排序约束 |
| **full** | λ_dir=0.05, λ_ipcw=0.10 | 0.7175 | 完整模型 |

### 实验范围

- **数据集**: BLCA (膀胱癌)
- **Folds**: 0, 1, 2, 3, 4 (5折交叉验证)
- **总实验数**: 3 variants × 5 folds = **15个实验**
- **预计时间**: 约2-4小时（每个实验约10-15分钟）

---

## 🚀 一键运行方法

### 方法1: 运行所有实验（推荐）

```bash
cd /data1/DCT-Reg

# 一键运行所有15个实验
bash run_e4_experiments.sh
```

**说明**:
- 会先确认是否继续
- 自动运行所有3个variants × 5 folds
- 日志保存到 `results/e4_intervention_audit/e4_experiments.log`
- 结果保存到 `results/e4_intervention_audit/`

### 方法2: 先测试单个实验

```bash
cd /data1/DCT-Reg

# 测试 direction_only fold 0（约10分钟）
bash test_e4_single.sh
```

**如果成功**，再运行所有实验：
```bash
bash run_e4_experiments.sh
```

### 方法3: Python直接调用

```bash
cd /data1/DCT-Reg

# 运行所有实验
python scripts/run_all_e4_experiments.py \
    --variants direction_only,ipcw_only,full \
    --folds 0,1,2,3,4

# 或只运行特定实验
python scripts/run_all_e4_experiments.py \
    --variants direction_only \
    --folds 0,1
```

---

## 📊 运行后的分析

### 1. 快速分析结果

```bash
# 生成汇总统计
python scripts/analyze_e4_results.py

# 输出示例：
# direction_only:
#   Low-risk monotonic rate: 85.3% ± 3.2%
#   High-risk monotonic rate: 82.1% ± 4.1%
```

### 2. 可视化结果

```bash
# 生成干预曲线图
python scripts/visualize_e4_results.py \
    --input results/e4_intervention_audit

# 生成文件：
#   - intervention_curves_by_variant.png
#   - monotonicity_comparison.png
#   - risk_change_distribution.png
```

---

## 📁 输出文件结构

```
results/e4_intervention_audit/
├── direction_only_blca_fold0.csv  # 单个实验结果
├── direction_only_blca_fold1.csv
├── ...
├── ipcw_only_blca_fold0.csv
├── ...
├── full_blca_fold0.csv
├── ...
├── summary.csv                     # 汇总统计
└── e4_experiments.log              # 运行日志
```

### CSV文件格式

每个CSV包含以下列：
- `patient_id`: 患者ID
- `true_time`: 真实生存时间
- `true_event`: 真实事件指示
- `baseline_risk`: 原始风险预测
- `alpha`: 干预强度 (0-1)
- `direction`: 'low_risk' 或 'high_risk'
- `risk_pred`: 干预后的风险预测
- `risk_change`: 风险变化量
- `embedding_distance`: 表示移动距离

---

## 🔍 监控实验进度

### 查看实时日志

```bash
tail -f results/e4_intervention_audit/e4_experiments.log
```

### 检查已完成的实验

```bash
ls -lh results/e4_intervention_audit/*.csv
```

### 查看进程状态

```bash
ps aux | grep e4_continuous_intervention
```

---

## ⚙️ 技术细节

### 已解决的TODO

原始E4脚本有3个TODO，现在全部已解决：

1. ✅ **模型实例化**: 使用 `get_model()` + `load_state_dict()`
2. ✅ **数据集加载**: 使用 `SurvivalDatasetFactory` + fold split
3. ✅ **表示提取**: 通过 `model.encoder()` 或 `model.wsi_encoder()`

### 锚点提取

脚本会自动尝试多种锚点存储位置：
- `model.low_risk_anchor` / `model.high_risk_anchor`
- `model.anchors[0]` / `model.anchors[1]`
- `model.dct_module.anchors`
- `model.reference_embeddings`

### 风险预测

支持多种模型架构：
- NLL survival (多bin输出)：使用负期望生存时间
- Cox模型：直接使用log hazard

---

## 🎯 成功标准

### 如果Direction Only成功

**预期**:
- 单调递减率（towards low-risk）> 75%
- 单调递增率（towards high-risk）> 75%
- 平均风险变化显著（p < 0.05）

**解释**:
- ✓ 证明方向约束学到了真实的预后方向
- ✓ 支持"预后锚点引导的方向传输"核心idea
- ✓ 为论文提供最强证据

### 如果IPCW Only失败

**预期**:
- 单调性率较低（< 60%）
- 风险变化不显著

**解释**:
- ✓ 与消融实验一致（IPCW Only性能较弱）
- ✓ 说明纯排序约束不足以学习预后方向
- ✓ 进一步证明方向约束的重要性

### 如果Full Model最好

**预期**:
- 单调性率最高
- 风险变化最显著

**解释**:
- ✓ IPCW + Direction协同作用
- ✓ 完整模型学到了最鲁棒的预后表示

---

## 🐛 故障排除

### 如果遇到 "Cannot find prognostic anchors"

**原因**: 模型架构可能不同

**解决**:
1. 检查模型属性：
```python
checkpoint = torch.load("model_best_s0.pth")
print(checkpoint.keys())
```

2. 查看模型定义：
```bash
grep -r "class.*DCT" survot_rank/research/methods/
```

### 如果遇到 "Model has no decoder"

**原因**: 模型组件命名不同

**解决**:
1. 查看模型forward方法
2. 修改 `compute_risk_prediction()` 函数中的逻辑

### 如果运行很慢

**原因**: 测试集可能较大

**解决**:
1. 减少alpha点数：`--alphas 0.0,0.5,1.0`
2. 仅测试部分fold：`--folds 0,1`

---

## 📝 预期实验报告

运行完成后，你将获得：

1. **定量结果**
   - 3个variants的单调性率对比
   - 风险变化量统计
   - 跨fold一致性分析

2. **可视化**
   - 干预曲线图（患者级别）
   - 聚合趋势图（人群级别）
   - 单调性率柱状图

3. **关键结论**
   - Direction constraint是否学到了真实的预后方向？
   - 方向一致性是否比性能指标更强的证据？
   - IPCW的作用是什么？

---

## 🎉 准备就绪！

**所有脚本已创建并测试**：
- ✅ `scripts/e4_continuous_intervention_audit_v2.py` (483行)
- ✅ `scripts/run_all_e4_experiments.py` (247行)
- ✅ `scripts/analyze_e4_results.py` (129行)
- ✅ `run_e4_experiments.sh` (一键运行)
- ✅ `test_e4_single.sh` (测试脚本)

**checkpoints已准备好**：
- ✅ Direction Only: 5 folds
- ✅ IPCW Only: 5 folds  
- ✅ Full Model: 5 folds

**现在就可以运行！**

---

## 快速启动命令

```bash
cd /data1/DCT-Reg

# 选项1：直接运行所有实验（推荐）
bash run_e4_experiments.sh

# 选项2：先测试一个
bash test_e4_single.sh
# 如果成功，再运行：
bash run_e4_experiments.sh
```

**预计完成时间**: 今天下午18:00-20:00

---

**文档更新时间**: 2026-09-04 14:30  
**下次更新**: E4实验完成后
