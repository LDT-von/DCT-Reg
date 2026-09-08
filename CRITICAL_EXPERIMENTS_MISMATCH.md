# 🚨 关键发现：两个实验的 C-index 不一致

生成时间：2026-09-08 02:06 UTC

---

## 问题

你问了一个**非常关键的问题**：

> "为何审计结果不是 0.72？我们已经证明了 DCT 的有效性，C-index = 0.72，为何现在是 0.64？"

---

## 真相

**这是两个完全不同的实验！**

| 实验 | 路径 | BLCA C-index | Fold 数据 |
|------|------|-------------|----------|
| **论文主实验** | `/results/dct_v3.10/robust/final_50ep_old/` | **0.7208 ± 0.0145** | fold 0: 0.6950<br>fold 1: 0.7219<br>fold 2: 0.7381<br>fold 3: 0.7296<br>fold 4: 0.7197 |
| **审计实验** | `/results/dct_v3.10_experiments/robust/full/` | **0.6376 ± 0.0xxx** | fold 0: 0.6194<br>fold 1: 0.5940<br>fold 2: 0.6305<br>fold 3: 0.6339<br>fold 4: 0.7103 |

**差距：0.7208 vs 0.6376 = -8.3 个百分点**

---

## 关键差异

### 1. 实验配置不同

```bash
# 论文主实验
final_50ep_old/blca/.../sp_dct_v310_dct_reg_blca_50ep

# 审计实验  
full/blca/.../sp_dct_v310_full_blca_50ep
```

### 2. 可能的原因

| 因素 | 论文实验 | 审计实验 | 影响 |
|------|----------|----------|------|
| **数据划分** | 某个随机种子 | 可能不同种子？ | ⚠️ 高 |
| **Checkpoint** | 训练了完整 50 epoch | 可能冻结了模型？ | ⚠️ 高 |
| **验证协议** | 正常验证 | 审计干预协议 | ⚠️ 高 |
| **Epoch 选择** | Best epoch per fold | 固定 epoch？ | ⚠️ 中 |

---

## 审计实验的可能问题

### 假设 1：模型冻结导致性能下降

如果审计实验**在训练早期就冻结了模型**，那么性能下降是正常的。

**验证方法**：
```bash
# 检查审计实验的 epoch_curve
find /data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca -name "epoch_curve*.csv"
```

### 假设 2：审计干预破坏了预测

如果审计实验在**推理时进行了传输干预**，可能影响了 C-index。

**但这不合理**：C-index 应该是**冻结前**的性能，不应该受干预影响。

### 假设 3：不同的数据划分

如果两个实验用了**不同的 fold 划分**，那么 C-index 差异可能很大。

**验证方法**：
```bash
# 比较两个实验的样本数
```

---

## 🔍 需要立即验证的事项

1. **审计实验的 outer_cindex 到底是什么？**
   - 是训练完成后的正常验证 C-index？
   - 还是干预后的 C-index？

2. **两个实验用的是同一个 checkpoint 吗？**
   - 如果不是，为什么性能差这么多？

3. **审计实验真的跑完了吗？**
   - 会不会只训练了几个 epoch 就停了？

---

## 💣 对论文的致命影响

### 如果 0.6376 是真的

那么你的论文**无法声称**：

❌ "DCT 在 BLCA 上达到 0.72 的良好性能"  
❌ "DCT 超越多模态 SOTA"  
❌ "预测性能与审计结果一致"

### 如果 0.7208 是真的

那么**审计实验的数据有问题**，需要重新跑。

---

## 🚀 下一步行动

### 方案 A：验证审计实验是否跑对了

```bash
# 1. 检查 outer_cindex 的定义
grep -r "outer_cindex" /data1/DCT-Reg/survot_rank --include="*.py" | head -20

# 2. 检查审计实验的训练日志
find /data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca -name "*.log" -o -name "epoch_curve*.csv"

# 3. 对比两个实验的配置
```

### 方案 B：用论文实验的 checkpoint 重新跑审计

```bash
# 1. 找到 final_50ep_old 的 checkpoint
find /data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca -name "*.pt" -o -name "*.pth"

# 2. 用这些 checkpoint 跑审计
python scripts/run_audit_on_existing_checkpoints.py --cancer blca --checkpoint_dir final_50ep_old
```

### 方案 C：诚实报告差异

在论文中写：

> "We trained DCT v3.10 on BLCA and achieved C-index 0.7208 ± 0.0145. However, when we applied the DCT-Audit framework with frozen checkpoints, the reported outer C-index was 0.6376 ± 0.0xxx. This discrepancy suggests..."

---

## ⚠️ 我的建议

**立即验证 `outer_cindex` 的定义！**

这是**最关键**的问题：

1. 如果 `outer_cindex` 是**审计干预后的 C-index** → 0.6376 是预期的（干预会影响预测）
2. 如果 `outer_cindex` 是**正常验证 C-index** → 0.6376 说明**审计实验的模型训练有问题**

**查看代码**：
```bash
grep -A 10 "outer_cindex" /data1/DCT-Reg/survot_rank/research/methods/dct_v310_*/model.py
```

---

**最后更新**: 2026-09-08 02:06 UTC  
**状态**: 🔴 紧急 - 需要立即澄清
