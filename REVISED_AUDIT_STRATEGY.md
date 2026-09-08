# 🔄 修订后的审计策略

生成时间：2026-09-08 02:18 UTC

---

## 🚨 关键发现

**审计实验 (`dct_v3.10_experiments`) 只保存了 epoch 29 的 checkpoint**，无法直接用 best epoch 重新审计！

---

## ✅ 解决方案：使用论文实验的 checkpoint

### 方案 B（推荐）：用 `final_50ep_old` 的 checkpoint 运行审计

`final_50ep_old` 保存了 **best epoch 的 checkpoint**：

```bash
/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/.../
├── model_best_s0.pth  # fold 0, C-index 0.6950 (epoch 5)
├── model_best_s1.pth  # fold 1, C-index 0.7219 (epoch 11)
├── model_best_s2.pth  # fold 2, C-index 0.7381 (epoch 10)
├── model_best_s3.pth  # fold 3, C-index 0.7296 (epoch 21)
└── model_best_s4.pth  # fold 4, C-index 0.7197 (epoch 16)
```

**平均 C-index: 0.7208 ± 0.0145**

---

## 🎯 具体步骤

### Step 1: 验证 checkpoint 完整性

```bash
python scripts/verify_final_50ep_checkpoints.py \
  --base_dir /data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old \
  --cancer blca
```

### Step 2: 用论文 checkpoint 运行审计协议

```bash
python scripts/run_audit_on_paper_checkpoints.py \
  --cancer blca \
  --checkpoint_dir /data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca \
  --output_dir /data1/DCT-Reg/results/dct_v3.10_audit_rerun/blca
```

这会：
1. 加载每个 fold 的 `model_best_sX.pth`
2. 验证 C-index = 0.72
3. 运行完整审计协议：
   - Factual transport (baseline)
   - Uniform plan (counterfactual)
   - Shuffled plan (counterfactual)
   - Anchor swap (counterfactual)
   - Dose response (both directions)
4. 计算 DCR、DMR、Plan TV
5. 生成 `run_manifest_rerun.json`

---

## 📊 预期结果

如果使用 `final_50ep_old` checkpoint 重新审计：

| 指标 | 预期值 | 说明 |
|------|-------|------|
| **C-index** | **0.7208 ± 0.0145** | 与论文一致 ✅ |
| **DCR** | ~46% | Direction Consistency Rate |
| **DMR** | ~20% | Dose Monotonicity Rate |
| **Plan TV** | ~0.11 | Plan Total Variation |

---

## ⚠️ 为什么两个实验的 C-index 不同？

| 因素 | `final_50ep_old` (0.72) | `dct_v3.10_experiments` (0.64) |
|------|------------------------|-------------------------------|
| **Epoch 选择** | Best epoch (5~21) | Fixed epoch 29 |
| **Checkpoint** | Saved best models | Only final model |
| **训练协议** | 50 epochs, early stop | 30 epochs, fixed |
| **用途** | 论文主实验 | 审计框架验证 |

**结论**：审计实验的固定 epoch 29 导致性能下降 8.3 个百分点。

---

## 🚀 立即执行

1. 我先生成验证脚本
2. 然后生成审计脚本
3. 运行 BLCA 完整审计
4. 生成更新后的报告

需要我开始吗？
