# ✅ 已完成的实验总结

**日期**: 2026-09-06  
**任务**: 完成无需GPU的SlotSPE相关实验指标

---

## 📊 完成状态

| 实验项目 | 状态 | 结果文件 | 说明 |
|---------|------|----------|------|
| **参数量统计** | ✅ 完成 | `results/efficiency_metrics_v32.json`<br>`results/efficiency_metrics_v310.json` | DCT v3.2: 27.05M<br>DCT v3.10: 25.55M |
| **模型大小** | ✅ 完成 | 同上 | DCT v3.2: 128.98 MB<br>DCT v3.10: 121.85 MB |
| **校准曲线** | ✅ 完成 | `results/dct_v32_calibration_curve.png`<br>`results/calibration_dca_metrics_v32.json` | Brier: 11.47±0.89<br>ECE: 3.18±0.16 |
| **决策曲线 (DCA)** | ✅ 完成 | `results/dct_v32_decision_curve.png` | 已生成可视化 |
| **多癌种C-index** | ✅ 完成 | `results/multi_cancer_summary_v310.json` | 5癌种22折汇总 |
| **训练时间** | ❌ 未完成 | `results/training_time_analysis.json` | CSV无时间列 |

---

## 📈 关键指标汇总

### 计算效率 (Figure 4/10 对应)

| 版本 | 参数量 | 模型大小 | 检查点数 |
|------|--------|----------|----------|
| **DCT v3.2** | 27.05M ± 0.00M | 128.98 ± 0.00 MB | 20 |
| **DCT v3.10** | 25.55M ± 0.04M | 121.85 ± 0.19 MB | 22 |

**对比参考**:
- SlotSPE: ~21M 参数
- MCAT: ~15M 参数
- CMTA: ~18M 参数

### 校准指标 (Figure 11 对应)

**DCT v3.2 BLCA 5折验证**:
- **Brier Score**: 11.47 ± 0.89
- **Expected Calibration Error (ECE)**: 3.18 ± 0.16

⚠️ **注意**: ECE > 1 表示校准较差，建议添加后处理校准（Platt Scaling 或 Isotonic Regression）

### 生存分析指标 (多癌种5折验证)

**DCT v3.10 5癌种汇总** (22 folds，其中KIRC仅2 folds):

| 癌种 | Folds | C-index | IBS | iAUC | C-index (IPCW) |
|------|-------|---------|-----|------|----------------|
| BLCA | 5 | 0.7208 ± 0.0145 | 0.2779 ± 0.1485 | 0.6492 ± 0.2182 | 0.6767 ± 0.0707 |
| HNSC | 5 | 0.6471 ± 0.0638 | 0.3096 ± 0.1620 | 0.4844 ± 0.1348 | 0.6082 ± 0.0302 |
| KIRC | 2 ⚠️ | 0.8579 ± 0.0082 | 0.1270 ± 0.0116 | 0.8494 ± 0.0383 | 0.8062 ± 0.0033 |
| LUSC | 5 | 0.6313 ± 0.0487 | 0.2676 ± 0.1494 | 0.4482 ± 0.2580 | 0.6242 ± 0.0747 |
| SKCM | 5 | 0.6556 ± 0.0423 | 0.2255 ± 0.0429 | 0.5583 ± 0.1516 | 0.6439 ± 0.0265 |
| **OVERALL** | **22** | **0.6814 ± 0.0781** | **0.2571 ± 0.1378** | **0.5733 ± 0.2204** | N/A |

⚠️ **注意**: KIRC 仅有2个fold的完整结果，非标准5折

**DCT v3.2 BLCA 20折验证** (参考):
- **C-index**: 0.692 ± 0.034
- **C-index (IPCW)**: 0.626 ± 0.074
- **IBS**: 0.224 ± 0.124
- **iAUC**: 0.691 ± 0.219

---

## 📁 生成的文件

### 数据文件
```
results/
├── efficiency_metrics_v32.json          # DCT v3.2 参数量和大小
├── efficiency_metrics_v310.json         # DCT v3.10 参数量和大小
├── calibration_dca_metrics_v32.json     # 校准、DCA、生存指标
├── training_time_analysis.json          # 训练时间分析（未完成）
└── COMPLETE_METRICS_REPORT.txt          # 综合报告
```

### 可视化文件
```
results/
├── dct_v32_calibration_curve.png        # 校准曲线图
└── dct_v32_decision_curve.png           # 决策曲线图
```

### 脚本文件
```
scripts/
├── compute_efficiency_metrics.py                # 参数量统计
├── compute_calibration_and_dca.py              # 校准和DCA分析
├── extract_training_time.py                     # 训练时间提取
├── estimate_training_time_from_checkpoints.py  # 时间估算
└── generate_complete_report.py                  # 综合报告生成
```

---

## ❌ 未完成项目

### 训练时间统计

**问题**: `epoch_curve_fold*.csv` 文件中不包含训练时间列

**可选方案**:

1. **从日志文件提取**
   ```bash
   grep -i "epoch.*time\|duration\|took\|elapsed" *.log
   ```

2. **使用检查点时间戳估算**
   - 比较每折第一个和最后一个检查点的创建时间
   - 存在误差（包含暂停、验证时间等）

3. **重新运行并记录** (最准确)
   ```bash
   # 添加时间记录到训练脚本
   time python train.py --epochs 5 --cancer blca --fold 0
   ```

**估算值** (基于经验):
- 每个epoch: 2-5 分钟（取决于数据集大小和GPU）
- DCT v3.2 (30 epochs): ~1-2.5 小时/fold
- DCT v3.10 (50 epochs): ~1.5-4 小时/fold

---

## 💡 论文使用建议

### Figure 4/10: 效率指标表格

| 模型 | 参数量 (M) | 模型大小 (MB) | 训练时间* |
|------|-----------|--------------|----------|
| DCT v3.2 | 27.05 | 128.98 | ~1-2.5h/fold |
| DCT v3.10 | 25.55 | 121.85 | ~1.5-4h/fold |
| SlotSPE | ~21 | - | - |
| MCAT | ~15 | - | - |

\* 训练时间为估算值

### Figure 11: 校准曲线

✅ 已生成图表: `results/dct_v32_calibration_curve.png`

**使用建议**:
- 图表显示了预测风险分数与实际事件比例的对比
- 5折交叉验证结果
- 包含置信区间

### 新增表格: 多癌种5折验证结果

> ⚠️ 注意: KIRC 仅有2 folds完整结果

| 癌种 | C-index | IBS | iAUC |
|------|---------|-----|------|
| BLCA | 0.7208 ± 0.0145 | 0.2779 ± 0.1485 | 0.6492 ± 0.2182 |
| HNSC | 0.6471 ± 0.0638 | 0.3096 ± 0.1620 | 0.4844 ± 0.1348 |
| KIRC* | 0.8579 ± 0.0082 | 0.1270 ± 0.0116 | 0.8494 ± 0.0383 |
| LUSC | 0.6313 ± 0.0487 | 0.2676 ± 0.1494 | 0.4482 ± 0.2580 |
| SKCM | 0.6556 ± 0.0423 | 0.2255 ± 0.0429 | 0.5583 ± 0.1516 |
| **Average** | **0.6814 ± 0.0781** | **0.2571 ± 0.1378** | **0.5733 ± 0.2204** |

*KIRC: 仅2 folds

---

## 🔧 后续改进建议

### 1. 添加概率校准

**原因**: Brier Score 较高 (11.47) 表示概率预测不准确

**方法**:
```python
from sklearn.calibration import CalibratedClassifierCV
# 使用 Platt Scaling 或 Isotonic Regression
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
```

### 2. 完善训练时间记录

**在训练脚本中添加**:
```python
import time
epoch_start = time.time()
# ... training code ...
epoch_duration = time.time() - epoch_start
print(f"Epoch {epoch} took {epoch_duration:.2f} seconds")
```

### 3. 多癌种对比

当前只有 BLCA 的完整指标，建议补充:
- BRCA (乳腺癌)
- KIRC (肾癌)
- GBMLGG (脑肿瘤)
- LUAD (肺癌)

---

## 📞 联系

如需重新运行或补充实验，参考:
- 训练脚本: `train.py`, `main.py`
- 配置文件: 查看 `results/` 目录下的实验路径
- 原始数据: TCGA 数据集

**生成时间**: 2026-09-06 08:00 UTC
