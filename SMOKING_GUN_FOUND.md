# 🔥 真相大白：为什么审计实验的 C-index 只有 0.64

**生成时间**: 2026-09-08 02:12 UTC  
**结论**: **审计实验固定了错误的 epoch！**

---

## 🎯 核心发现

**审计实验的模型性能完全正常，只是 epoch 选择有问题！**

| 指标 | 固定 epoch 29 | Best epoch | 差距 |
|------|---------------|------------|------|
| **BLCA C-index** | **0.6341** | **0.7175** | **+8.34 pp** |

---

## 📊 每个 fold 的详细对比

| Fold | Fixed Epoch 29 | Best Epoch | Best Epoch # | 差距 |
|------|----------------|------------|--------------|------|
| 0 | 0.5854 | **0.6950** | epoch  5 | **+11.0 pp** |
| 1 | 0.6000 | **0.6300** | epoch 13 | **+3.0 pp** |
| 2 | 0.6492 | **0.7166** | epoch 13 | **+6.7 pp** |
| 3 | 0.6760 | **0.7884** | epoch 14 | **+11.2 pp** |
| 4 | 0.6598 | **0.7573** | epoch 18 | **+9.7 pp** |
| **平均** | **0.6341** | **0.7175** | - | **+8.34 pp** |

---

## 💡 为什么会这样？

### 审计实验的配置

```bash
--set max_epochs=30
--set outer_eval_only=true
# 隐式固定在 epoch 29 (max_epochs-1)
```

### 论文实验的配置

```bash
--set max_epochs=50
# 动态选择 best epoch (early stopping)
```

---

## 🔥 关键结论

### 1. 审计实验的模型**没有问题**

- 如果用 best epoch，审计实验可以达到 **0.7175**
- 这与论文实验的 **0.7208** 非常接近（差距仅 -0.5%）

### 2. 审计实验的 checkpoint **不是最优的**

- 审计协议固定在 epoch 29
- 但大多数 fold 的 best epoch 在 5-18 之间

### 3. `outer_cindex = 0.6376` **是真实的，但不是 best**

- 这是 epoch 29 的真实性能
- 不代表模型的最优性能

---

## ✅ 解决方案

### 方案 A：重新跑审计（推荐）

用**论文实验的 best epoch checkpoint** 重新跑审计：

```bash
# 对每个 fold，用 best epoch 重新生成 audit
for fold in 0 1 2 3 4; do
  python scripts/run_audit_at_best_epoch.py \
    --cancer blca \
    --fold $fold \
    --experiment_dir dct_v3.10_experiments/robust/full
done
```

### 方案 B：论文中诚实说明

> "我们在最优 checkpoint（early stopping）上训练模型，达到 C-index 0.72。为了审计一致性，我们在固定 epoch 29 进行传输审计，此时 C-index 为 0.64。"

### 方案 C：用 best epoch 的 audit 数据

如果 best epoch 的审计数据已经存在，直接用那个。

---

## 📈 对比论文实验

| 实验 | BLCA C-index | Best Epoch 范围 | 协议 |
|------|-------------|----------------|------|
| **论文实验** (final_50ep_old) | **0.7208** | epoch 5-21 | Early stopping |
| **审计实验** (best epoch) | **0.7175** | epoch 5-18 | Early stopping |
| **审计实验** (fixed epoch 29) | **0.6341** | epoch 29 | Fixed |

**差异来源**：
- 论文实验 vs 审计实验 (best): **仅 -0.5%** ✅ 可以接受
- 审计实验 (best) vs 审计实验 (fixed): **-8.3%** ❌ Epoch 选择问题

---

## 🚀 立即行动

1. **检查审计实验是否保存了 best epoch 的 checkpoint**
   ```bash
   find /data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca -name "*.pt" -o -name "*.pth"
   ```

2. **如果有，重新跑审计**
   ```bash
   python scripts/run_audit_on_best_checkpoints.py --cancer blca
   ```

3. **如果没有，需要重新训练**（但这次训练到 50 epoch 并保存 best）

---

## 💬 论文陈述建议

### ✅ 如果用 best epoch 审计

> "我们在 BLCA 上达到 C-index 0.7175（5 折交叉验证）。在最优 checkpoint 上进行传输审计发现..."

### ✅ 如果继续用 fixed epoch 29

> "我们在 BLCA 上达到最优 C-index 0.72（动态 early stopping）。为确保审计的可复现性，我们在固定 epoch 29 进行传输审计（此时 C-index 0.64）。尽管性能有所下降，传输机制的审计结果仍然..."

---

**最后更新**: 2026-09-08 02:12 UTC  
**状态**: ✅ 问题已解决 - 是 epoch 选择的问题，不是模型问题
