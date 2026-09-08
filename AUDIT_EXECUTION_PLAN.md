# 🎯 审计执行计划：使用 Best Epoch Checkpoints

**生成时间**: 2026-09-08 02:16 UTC  
**目标**: 用论文中 C-index = 0.72 的模型重新运行审计协议

---

## 📋 执行摘要

### 问题
当前审计报告的 BLCA C-index = 0.6376，但论文中报告的是 0.7208。

### 根本原因
- 论文实验：使用每个 fold 的 **best epoch checkpoint**（early stopping）
- 审计实验：使用**固定 epoch 29** checkpoint

### 解决方案
使用 `final_50ep_old` 目录中的 best epoch checkpoint 重新运行审计。

---

## 📁 Checkpoint 位置

### 论文实验 Checkpoints (C-index = 0.72)

```bash
/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca/blca/
SurvOTRank_dct_v310_directional_regularized_transport/
0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/

model_best_s0.pth  # Fold 0: C-index 0.6950 (epoch 5)
model_best_s1.pth  # Fold 1: C-index 0.7219 (epoch 11)
model_best_s2.pth  # Fold 2: C-index 0.7381 (epoch 10)
model_best_s3.pth  # Fold 3: C-index 0.7296 (epoch 21)
model_best_s4.pth  # Fold 4: C-index 0.7197 (epoch 16)
```

平均 C-index: **0.7208 ± 0.0145**

---

## 🚀 执行步骤

### 步骤 1: 准备环境

```bash
cd /data1/DCT-Reg
conda activate survot  # 或你的环境名
```

### 步骤 2: 运行审计

**选项 A: 使用 Python 协调脚本**（推荐）

```bash
python scripts/run_audit_on_best_models.py \
  --cancer blca \
  --checkpoints_dir /data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep \
  --config /data1/DCT-Reg/configs/dct_v310_directional_regularized_transport.yaml \
  --output_dir /data1/DCT-Reg/results/audit_best_epochs_blca \
  --folds 0 1 2 3 4
```

**选项 B: 手动运行单个 fold**

```bash
python scripts/audit_dct_reg.py audit \
  --config configs/dct_v310_directional_regularized_transport.yaml \
  --checkpoint /data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s0.pth \
  --fold 0 \
  --epoch 5 \
  --output-dir results/audit_best_epochs_blca/fold_0 \
  --gpu 0
```

### 步骤 3: 提取审计结果

```bash
python scripts/extract_mechanism_audit.py \
  --base_dir /data1/DCT-Reg/results/audit_best_epochs_blca \
  --cancer blca
```

---

## 📊 预期结果

### 预测性能（outer C-index）

| Fold | Best Epoch | C-index (预期) |
|------|-----------|---------------|
| 0 | 5 | 0.6950 |
| 1 | 11 | 0.7219 |
| 2 | 10 | 0.7381 |
| 3 | 21 | 0.7296 |
| 4 | 16 | 0.7197 |
| **平均** | - | **0.7208 ± 0.0145** |

### 审计指标（预期）

基于现有 epoch 29 的审计结果推测：

| 指标 | Epoch 29 | Best Epoch (预期) |
|------|----------|------------------|
| **DCR** (方向一致率) | ~46% | ~45-50% |
| **DMR** (剂量单调率) | ~20% | ~18-25% |
| **Plan TV** | ~0.11 | ~0.10-0.12 |

**关键预测**：即使 C-index 提升到 0.72，审计指标也不会显著改善。

---

## ⚠️  重要注意事项

1. **审计协议不依赖于训练 epoch**：审计是在冻结模型上运行的，只看模型学到的传输机制。

2. **预测性能 ≠ 机制有效性**：
   - C-index 0.72 说明模型预测能力好
   - DCR ~46% 说明传输机制的方向一致性差

3. **论文结论需要修正**：
   ```
   "DCT 在 BLCA 上达到 C-index 0.7208 ± 0.0145，
   表现良好。然而机制审计显示传输方向一致性仅 46%，
   表明预测性能可能主要来自其他机制（如 NLL 和 IPCW rank）
   而非传输正则化。"
   ```

---

## 📝 执行检查清单

- [ ] 验证 checkpoint 文件存在
- [ ] 确认环境配置正确（torch, survot_rank）
- [ ] 运行 fold 0 测试
- [ ] 运行全部 5 个 folds
- [ ] 提取并汇总审计指标
- [ ] 更新论文文档
- [ ] 生成对比表格（0.64 vs 0.72）

---

**最后更新**: 2026-09-08 02:16 UTC  
**状态**: ✅ 准备就绪 - 可以开始执行
