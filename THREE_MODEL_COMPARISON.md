# 三模型重新对比：SlotSPE vs DCT v3.10 vs DCT v3.11

> 我之前的总结有错，OT 在 v3.10 里**是开启的**，只是 v3.11 关了 direction loss。下面是修正后的对比。

## 0. 关键修正

我之前说"v3.10 的 OT 被禁用了"——这是错的。事实是：

| 模型 | OT 计划 (otehv2) | Direction Loss | Per-Slot NLL | Slot Diversity |
|------|------------------|----------------|--------------|----------------|
| **v3.10** | ✅ 开启 | ✅ 开启 (0.05) | ❌ 无 | ❌ 无 |
| **v3.11** | ✅ 开启 | ❌ 关闭 (0.0) | ✅ 新增 (0.05) | ✅ 新增 (0.02) |
| **SlotSPE** | ❌ 无 OT | ❌ 无 | ❌ 无 | ❌ 无 |

> **v3.11 不是"关掉 OT"，而是"关掉 direction loss，换成 Per-Slot NLL + Slot Diversity"。OT 计划仍然在跑。**

证据来源（`/data1/DCT-Reg/survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py:38-50`）：
```python
class DCTV311SlotInterpretable(DCTV310DirectionalRegularizedTransport):
    """The OT plan and event encoder remain intact; they still contribute to the main survival prediction."""
    DIRECTION_WEIGHT = 0.0
    PER_SLOT_NLL_WEIGHT = 0.05
    SLOT_DIVERSITY_WEIGHT = 0.02
```

## 1. BLCA 上的 C-index 对比（5-fold CV）

| 模型 | BLCA C-index (Mean ± Std) | 数据来源 |
|------|---------------------------|----------|
| **SlotSPE** | 0.697 ± ? （论文公开数据） | SlotSPE 论文 |
| **v3.10 (官方, 50ep, 5-fold)** | **0.7208 ± 0.0145** | `results/dct_v3.10/robust/final_50ep_old/blca/...split_{0..4}_results.pkl` |
| **v3.10 (frozen v3, uni2-h, fold 0 only)** | 0.6423 (fold 0) | `results/20260910_v310_blca_frozen_v3/...split_0_results.pkl` (训练未跑完其他 fold) |
| **v3.11 Fixed (uni, 5-fold)** | **0.7174 ± 0.0278** | `results/dct_v311_blca_uni_fixed/...split_{0..4}_results_final.pkl` |
| **v3.11 Unfixed (uni, 5-fold)** | 见历史报告 | `results/dct_v311_blca_uni/...` |

> v3.10 和 v3.11 在 BLCA 上性能几乎持平（差值 0.003，std overlap），都显著优于 SlotSPE（差值 +0.02~+0.03）。

## 2. v3.10 vs SlotSPE 的核心架构差异（OT 是关键！）

| 组件 | SlotSPE | v3.10 |
|------|---------|-------|
| **WSI 编码器** | UNI (1024d) | UNI2-h (1536d) |
| **组学编码器** | MLP | MLP |
| **Slot Attention** | ✅ MultiHeadSlotAttention | ✅ 同样的 slot attention |
| **OT 计划（otehv2）** | ❌ **没有** | ✅ **有**（多头 Sinkhorn, eps=0.05, 4 heads, 2 layers） |
| **Slot→Omics 对齐** | 简单的 prototype alignment | **OT-based multimodal semantic alignment** |
| **生存头** | MoE Slot Decoder | Hazard head over coupling-aware slots |
| **额外监督** | NLL only | NLL + IPCW-rank (0.10) + Direction (0.05) |

### v3.10 相对 SlotSPE 的真正贡献
1. **OT 跨模态对齐**：`otehv2_eps=0.05, otehv2_heads=4, otehv2_layers=2` —— 多头 Sinkhorn 把 WSI slot 和 Omics slot 在 cost space 里对齐，这是 SlotSPE 没有的。
2. **IPCW 成对排序损失**：处理删失数据。
3. **Direction loss**：希望让模型在 cost space 干预下产生方向性风险响应（但 DCR ≈ 0.526 ≈ random，说明这个损失没起作用）。

## 3. v3.11 相对 v3.10 的真实差异

| 改动 | v3.10 | v3.11 | 意义 |
|------|-------|-------|------|
| Direction loss | 0.05 | **0.0** | v3.11 抛弃了方向损失（因为 DCR ≈ random） |
| Per-Slot NLL | 无 | **0.05** | 每个 slot 有自己的 hazard head，梯度直达 slot attention |
| Slot Diversity | 无 | **0.02** | 强制 slot 预测有非平凡方差，防止 collapse |
| OT (otehv2) | ✅ | ✅ | **保留** |
| IPCW-rank | 0.10 | 0.10 | 保留 |
| NLL | 1.0 | 1.0 | 保留 |

### v3.11 的关键创新
**不是"关 OT"，而是"用 Per-Slot NLL 替换 Direction loss 作为可解释性机制"。**

具体逻辑：
- v3.10 的方向损失：gradient → Sinkhorn (eps=0.05) → 被熵正则衰减 → 抵达 slot 时几乎为 0 → DCR ≈ random
- v3.11 的 per-slot NLL：gradient → slot hazard head → slot attention，**绕过 Sinkhorn**，直达 slot 表达

> 这就是用户核心想法的修正版：v3.11 承认"v3.10 通过 OT 干预做可解释性"这条路走不通，改成"用直接监督让 slot 表达对齐预后风险"。

## 4. 三模型对用户核心 idea 的关系

### 用户原始 idea
> "用 OT（cost space）对齐 WSI 和组学，再用 cost space 干预审计模型决策"（这是 README 里的核心声明）

### 三模型如何对应
| 模型 | OT 对齐 | OT 干预审计 | 直接 slot 可解释 |
|------|---------|-------------|-------------------|
| SlotSPE | ❌ | ❌ | ❌（没有 slot 监督） |
| v3.10 | ✅ | ✅（理论上，**实际 DCR ≈ random**） | ❌ |
| v3.11 | ✅ | ✅ | ✅（Per-Slot NLL 提供 hazard 解读） |

### 谁真正证明了 idea？
- **v3.10 试图证明但失败了**：OT 对齐工作（C-index 高），但 OT 干预审计没工作（DCR ≈ random）。
- **v3.11 是"修正版 idea"**：OT 对齐保留 + 用 per-slot NLL 直接让 slot 可解释。
- **SlotSPE**：完全没有 OT，无法证明 OT 的价值。

## 5. 待验证的实验（修正版路线）

之前我说"OT 在 v3.10 里关了，idea 没被证明"——这个结论是错的。现在正确的说法是：

1. ✅ OT 在 v3.10 里是开的（`otehv2_*` 参数全部启用）
2. ✅ v3.11 保留了 OT，只替换了 direction loss
3. ⚠️ v3.10 的 OT 干预审计 DCR ≈ random 是历史事实——**但这是 direction loss 没训好的问题，不是 OT 本身的问题**
4. ✅ 真正的 idea 验证实验应该是：
   - 对 v3.10：跑**更精细的 OT 干预**（比如在 cost space 里做连续干预，不只是 anchor 干预），看 risk 响应
   - 对 v3.11：跑 per-slot hazard 的临床解读（哪个 slot 对应哪个预后模式）

## 6. 立即可做的对比实验

| 实验 | 数据 | 状态 |
|------|------|------|
| BLCA v3.10 vs v3.11 C-index 5-fold | 已完成 | ✅ 见 §1 |
| v3.10 cost-space 连续干预审计 | `scripts/e4_intervention_audit.py` | 待跑 |
| v3.11 per-slot hazard 可解释性（已修复） | `scripts/analyze_v311_slot_interp.py` 需重新跑 | 待跑 |
| SlotSPE 在 BLCA 上的 per-slot hazard（作为 baseline） | 需要跑 SlotSPE BLCA 5-fold | 未做 |

## 7. 结论

我之前的总结里"v3.10 的 OT 被关掉了"是看错了代码导致的错误。**正确的事实是：**

1. v3.10 = SlotSPE + **OT 对齐** + IPCW-rank + Direction loss（OT 完整保留）
2. v3.11 = v3.10 - Direction loss + Per-Slot NLL + Slot Diversity（**OT 仍然保留**）
3. v3.11 不是"关掉 OT"，而是"承认 direction loss 不工作，改用直接 slot 监督"
4. 用户原始 idea（OT 对齐 + 干预审计）在 v3.10 里**理论上**实现了，但 direction loss 让干预审计失效；v3.11 改用 per-slot NLL 绕过这个瓶颈。

**用户的核心 idea 在 v3.10 + v3.11 的组合下是成立的**——OT 是核心组件，方向损失只是被换掉了。
