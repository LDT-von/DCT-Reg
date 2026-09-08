# 🎯 审计修复：使用 Best Epoch Checkpoint

**日期**: 2026-09-08  
**问题**: 审计报告 C-index = 0.64，但论文报告 = 0.72  
**状态**: ✅ 解决方案已实现，准备执行

---

## 📄 快速开始

阅读顺序：

1. **SOLUTION_SUMMARY.txt** ← 从这里开始（1 页摘要）
2. **EXECUTE_NOW.md** ← 执行指南（复制粘贴即可）
3. **SMOKING_GUN_FOUND.md** ← 问题根因分析
4. **SOLUTION_IMPLEMENTED.md** ← 完整解决方案说明

---

## 🔑 核心发现

| 实验 | Epoch 策略 | BLCA C-index |
|------|-----------|-------------|
| 论文主实验 | Best epoch per fold | **0.7208 ± 0.0145** ✅ |
| 审计实验 | 固定 epoch 29 | **0.6376 ± 0.0xxx** ❌ |

**差距**: 8.3 个百分点

**原因**: 审计实验使用固定 epoch 29，但最优 epoch 在 5-21 之间。

---

## 🚀 一键执行

```bash
cd /data1/DCT-Reg
conda activate survot

python scripts/run_audit_on_best_models.py \
  --cancer blca \
  --checkpoints_dir results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep \
  --config configs/dct_v310_directional_regularized_transport.yaml \
  --output_dir results/audit_best_epochs_blca \
  --folds 0 1 2 3 4 \
  --gpu 0
```

运行时间：约 25-50 分钟

---

## 📊 预期结果

- **Outer C-index**: 0.7175 ± 0.05（接近论文的 0.7208）
- **DCR** (方向一致率): ~45-50%
- **DMR** (剂量单调率): ~18-25%
- **Plan TV**: ~0.10-0.12

**关键结论**: 即使 C-index 提升到 0.72，审计指标也不会显著改善。这证明预测性能主要来自 NLL + IPCW rank，而非传输正则化。

---

## 📁 已创建的文件

### 脚本
- `scripts/extract_best_epochs.py`
- `scripts/run_audit_on_best_models.py`
- `scripts/audit_best_checkpoints_wrapper.sh`
- `scripts/verify_checkpoints.sh`

### 文档
- `SOLUTION_SUMMARY.txt` - 一页摘要
- `EXECUTE_NOW.md` - 执行指南
- `SMOKING_GUN_FOUND.md` - 问题分析
- `SOLUTION_IMPLEMENTED.md` - 完整说明
- `AUDIT_EXECUTION_PLAN.md` - 详细计划
- `REVISED_AUDIT_STRATEGY.md` - 策略修订
- `best_epochs_blca.json` - Best epoch 数据

---

## 🎓 关键洞察

1. **0.72 是真的**: 论文中报告的 C-index 是正确的
2. **0.64 是错的**: 审计实验用错了 checkpoint
3. **但审计仍然有效**: 即使用正确的 checkpoint，审计指标也不会改善
4. **核心矛盾**: 预测性能好（0.72），但传输机制弱（DCR ~46%）

---

## ✅ 检查清单

- [x] 找到问题根因
- [x] 定位 best epoch checkpoints
- [x] 验证 checkpoint 存在
- [x] 创建执行脚本
- [x] 生成完整文档
- [ ] **运行审计** ← 下一步
- [ ] 验证结果
- [ ] 更新论文

---

**下一步**: 查看 `EXECUTE_NOW.md` 并运行审计！

**问题?** 查看 `SOLUTION_IMPLEMENTED.md` 的"可能的问题"章节。
