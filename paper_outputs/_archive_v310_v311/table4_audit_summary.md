# Table 4: Mechanism Audit Summary

## Table 4a: BLCA 5-Fold Mechanism Audit Detail

| Fold | High (correct/total) | Low (correct/total) | DCR | Chance gap | Status |
|------|----------------------|---------------------|-----|------------|--------|
| 0 | 8/11 | 16/46 | 0.421 | -0.079 | ❌ ≤50% |
| 1 | 0/11 | 28/38 | 0.571 | +0.071 | ✅ >50% |
| 2 | 3/11 | 16/41 | 0.365 | -0.135 | ❌ ≤50% |
| 3 | 2/9 | 20/28 | 0.595 | +0.095 | ✅ >50% |
| 4 | 3/10 | 31/40 | 0.680 | +0.180 | ✅ >50% |

| **Mean** | - | - | **0.526 ± 0.116** | - | ✅ |

**注**: DCR = Direction Consistency Rate; 随机基线 = 50%

**注**: High/Low 计数基于审计测试样本
(高风险=事件发生早, 低风险=事件发生晚或删失)

---

## Table 4b: Cross-Cancer Mechanism Audit (1-2 folds each)

| Cancer | Folds tested | Mean C-index | DCR | Status |
|--------|-------------|-------------|-----|--------|
| BLCA (Bladder) | 5 | 0.7175 | 0.526 ± 0.116 | ✅ Complete |

---

## Table 4c: Null Hypothesis Control (BLCA Fold 1)

| Condition | DCR | Interpretation |
|-----------|-----|----------------|
| Full (Learned Coupling) | 0.370 | Ground truth coupling |
| Uniform Coupling | ~0.370 | Transport plan replaced with uniform |
| Shuffled Reference | ~0.370 | Reference anchors shuffled |
| Anchor Swap | ~0.413 | Low/high anchors swapped |

**注**: 如果 uniform/shuffled 的 DCR 与 full 接近 → 机制贡献有限
**注**: Anchor swap DCR 略高说明 anchor 顺序本身有一定区分度