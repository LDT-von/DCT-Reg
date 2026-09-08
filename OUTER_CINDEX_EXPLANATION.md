# outer_cindex 的定义与真相

生成时间：2026-09-08 02:10 UTC

---

## ✅ 问题解决了！

`outer_cindex` **是正常的验证集 C-index**，**不是**审计干预后的 C-index。

---

## 证据

从 `run_manifest.json` 可以看到：

```json
{
  "outer_cindex": 0.6194,  // ← 这是正常验证 C-index
  "outer_cindex_ipcw": 0.6492,
  "fixed_epoch": 29,       // ← 固定在第 29 个 epoch
  "mechanism_audit/factual": {
    "direction_consistency": { ... },  // ← 这些才是审计指标
    "reconfiguration": { ... }
  }
}
```

**关键命令行参数**：
```bash
--set outer_eval_only=true
--set max_epochs=30
--set fixed_epoch=29
```

---

## 🔍 为什么两个实验的 C-index 差这么多？

| 实验 | outer_cindex | Best Epoch | 协议 |
|------|-------------|------------|------|
| **论文实验** | **0.7208** | **动态选择** (epoch 5~21) | **Early stopping** |
| **审计实验** | **0.6376** | **固定 epoch 29** | **固定 checkpoint** |

---

## 💡 核心差异：Early Stopping vs Fixed Epoch

### 论文实验（final_50ep_old）

```python
# 每个 fold 动态选择 best epoch
fold 0: epoch  5 → C-index 0.6950
fold 1: epoch 11 → C-index 0.7219
fold 2: epoch 10 → C-index 0.7381
fold 3: epoch 21 → C-index 0.7296
fold 4: epoch 16 → C-index 0.7197

平均：0.7208 ± 0.0145
```

### 审计实验（dct_v3.10_experiments）

```python
# 所有 fold 固定 epoch 29
fold 0: epoch 29 → C-index 0.6194
fold 1: epoch 29 → C-index 0.5940
fold 2: epoch 29 → C-index 0.6305
fold 3: epoch 29 → C-index 0.6339
fold 4: epoch 29 → C-index 0.7103

平均：0.6376 ± 0.0xxx
```

---

## 🎯 根本原因

**审计实验固定了 epoch 29，但这不是 best epoch！**

让我验证一下审计实验的 epoch curve：
