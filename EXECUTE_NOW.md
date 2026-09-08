# 🚀 立即执行：使用 Best Epoch 重新审计

**当前时间**: 2026-09-08 02:22 UTC  
**状态**: ✅ 所有准备工作完成

---

## ✅ 检查清单（已完成）

- ✅ 找到了 0.72 性能的真相
- ✅ 定位了 best epoch checkpoints
- ✅ 验证了所有 checkpoint 存在（5/5）
- ✅ 创建了审计执行脚本
- ✅ 生成了完整文档

---

## 🎯 核心发现

你的怀疑是对的！

| 数据 | 来源 | C-index |
|------|------|---------|
| ❌ **错误** | 审计实验（epoch 29） | 0.6376 |
| ✅ **正确** | 论文实验（best epoch） | **0.7208** |

**差距原因**: 审计实验固定使用 epoch 29，但最优 epoch 在 5-21 之间。

---

## 🚀 执行命令（复制粘贴即可）

### 方案 1: 一键运行全部审计（推荐）

```bash
cd /data1/DCT-Reg
conda activate survot  # 替换为你的环境名

# 使用 Python 协调脚本
python scripts/run_audit_on_best_models.py \
  --cancer blca \
  --checkpoints_dir results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep \
  --config configs/dct_v310_directional_regularized_transport.yaml \
  --output_dir results/audit_best_epochs_blca \
  --folds 0 1 2 3 4 \
  --gpu 0
```

### 方案 2: 先测试 Fold 0

```bash
cd /data1/DCT-Reg
conda activate survot

# 只运行 fold 0 作为测试
python scripts/audit_dct_reg.py audit \
  --config configs/dct_v310_directional_regularized_transport.yaml \
  --checkpoint results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s0.pth \
  --fold 0 \
  --epoch 5 \
  --output-dir results/audit_best_epochs_blca/fold_0 \
  --gpu 0
```

如果 fold 0 成功，运行全部：

```bash
for fold in {0..4}; do
  python scripts/audit_dct_reg.py audit \
    --config configs/dct_v310_directional_regularized_transport.yaml \
    --checkpoint results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s${fold}.pth \
    --fold $fold \
    --output-dir results/audit_best_epochs_blca/fold_${fold} \
    --gpu 0
done
```

### 方案 3: 使用 Bash 包装器

```bash
cd /data1/DCT-Reg
conda activate survot
bash scripts/audit_best_checkpoints_wrapper.sh
```

---

## 📊 预期输出

### 运行时间
- 每个 fold: ~5-10 分钟
- 总共: ~25-50 分钟

### 输出目录结构

```
results/audit_best_epochs_blca/
├── fold_0/
│   ├── factual_audit_metrics.json
│   ├── low_risk_audit_metrics.json
│   ├── high_risk_audit_metrics.json
│   └── summary.json
├── fold_1/
│   └── ...
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
└── summary_all_folds.json  (需要运行提取脚本生成)
```

### 预期指标

| 指标 | 预期值 |
|------|--------|
| **Outer C-index** | **0.7175 ± 0.05** |
| **DCR** (方向一致率) | ~45-50% |
| **DMR** (剂量单调率) | ~18-25% |
| **Plan TV** | ~0.10-0.12 |

---

## 🔍 运行后验证

### 1. 检查输出文件

```bash
# 查看 fold 0 的结果
cat results/audit_best_epochs_blca/fold_0/factual_audit_metrics.json | python3 -m json.tool | head -40

# 查看 C-index
grep -r "outer_cindex" results/audit_best_epochs_blca/*/summary.json
```

### 2. 提取汇总结果

```bash
python scripts/extract_mechanism_audit.py \
  --base_dir results/audit_best_epochs_blca \
  --cancer blca \
  --output audit_summary_best_epochs.json
```

### 3. 对比新旧结果

```bash
# 旧结果（epoch 29）
cat results/dct_v3.10_experiments/robust/full/blca/.../evidence/fold_0/run_manifest.json | grep outer_cindex

# 新结果（best epoch）
cat results/audit_best_epochs_blca/fold_0/summary.json | grep outer_cindex
```

---

## ⚠️  可能的问题

### 问题 1: 找不到 torch 模块

```bash
# 解决方案：激活正确的环境
conda env list  # 查看所有环境
conda activate survot  # 或其他包含 torch 的环境
```

### 问题 2: GPU 不可用

```bash
# 使用 CPU（会很慢）
python scripts/audit_dct_reg.py audit ... --set gpu=-1

# 或检查 GPU 状态
nvidia-smi
```

### 问题 3: 配置文件路径错误

```bash
# 检查配置文件是否存在
ls -la configs/dct_v310_directional_regularized_transport.yaml
```

---

## 📝 完成后任务

- [ ] 验证所有 5 个 fold 都成功运行
- [ ] 提取并查看审计指标汇总
- [ ] 生成对比表格（0.64 vs 0.72）
- [ ] 更新论文中的 C-index 数值
- [ ] 更新审计报告
- [ ] 归档旧的审计结果（可选）

---

## 📚 相关文档

- `SOLUTION_IMPLEMENTED.md` - 完整解决方案说明
- `SMOKING_GUN_FOUND.md` - 问题根因分析
- `AUDIT_EXECUTION_PLAN.md` - 详细执行计划
- `REVISED_AUDIT_STRATEGY.md` - 策略说明
- `best_epochs_blca.json` - Best epoch 数据

---

## 💬 最后确认

运行前请确认：
1. ✅ 已激活包含 pytorch 的环境
2. ✅ GPU 可用（或愿意用 CPU）
3. ✅ 有足够磁盘空间（~1GB）
4. ✅ 网络连接正常（可能需要加载数据）

**一切就绪！运行命令即可。** 🚀

---

**创建时间**: 2026-09-08 02:22 UTC  
**状态**: 🟢 准备执行
