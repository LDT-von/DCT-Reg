# E4 Audit Experiments - Final Results Report

**Date**: September 5, 2026  
**Cancer Type**: BLCA (Bladder Cancer)  
**Experiments Completed**: 10/10 (2 variants × 5 folds each)

---

## Executive Summary

✅ **All E4 directional consistency audits completed successfully**

Two intervention variants were tested across 5-fold cross-validation:
- **Direction Only**: Uses only the direction of transport interventions
- **IPCW Only**: Uses only inverse probability of censoring weighting

---

## Results Overview

### Direction Only Variant

| Fold | Samples | Anchor Distance | Mean Risk | Std Risk |
|------|---------|----------------|-----------|----------|
| 0    | 76      | 4.9543         | -1.9884   | 1.0510   |
| 1    | 76      | 1.7710         | -3.2797   | 0.4567   |
| 2    | 76      | 1.0995         | -3.1890   | 0.3199   |
| 3    | 76      | 2.1461         | -3.1331   | 0.0825   |
| 4    | 76      | 3.3855         | -3.2449   | 0.3529   |

**Aggregate Statistics:**
- Anchor Distance: 2.67 ± 1.52 (range: 1.10 - 4.95)
- Mean Risk Score: -2.97 ± 0.55
- Risk Score Variability: 0.45 ± 0.36

### IPCW Only Variant

| Fold | Samples | Anchor Distance | Mean Risk | Std Risk |
|------|---------|----------------|-----------|----------|
| 0    | 76      | 3.8320         | -2.7657   | 0.9631   |
| 1    | 76      | 1.3599         | -2.9361   | 0.5559   |
| 2    | 76      | 1.9429         | -2.3165   | 0.8867   |
| 3    | 76      | 3.6369         | -3.1695   | 0.9217   |
| 4    | 76      | 1.6286         | -3.0809   | 0.4847   |

**Aggregate Statistics:**
- Anchor Distance: 2.48 ± 1.17 (range: 1.36 - 3.83)
- Mean Risk Score: -2.85 ± 0.34
- Risk Score Variability: 0.76 ± 0.22

---

## Comparative Analysis

| Metric | Direction Only | IPCW Only | Winner |
|--------|---------------|-----------|--------|
| **Anchor Distance** | 2.67 ± 1.52 | 2.48 ± 1.17 | IPCW (7.7% smaller) |
| **Risk Consistency** | 0.45 ± 0.36 | 0.76 ± 0.22 | **Direction Only** (40.6% more consistent) |
| **Cross-Fold Stability** | 1.52 std | 1.17 std | IPCW (23.0% more stable) |

---

## Key Findings

### 1. **Direction Only Shows Superior Risk Consistency**
- Average risk score std: **0.4526** vs 0.7624 (IPCW)
- **40.6% more consistent** predictions across interventions
- Particularly stable in folds 3 (0.08 std) and 2 (0.32 std)

### 2. **IPCW Only Has Slightly Tighter Anchors**
- Mean anchor distance: **2.48** vs 2.67 (Direction)
- 7.7% smaller separation from training centroid
- More compact embedding space

### 3. **Direction Only Shows Higher Cross-Fold Variability**
- Anchor distance variance: **1.52** vs 1.17 (IPCW)
- 30.7% more variability between folds
- Fold 0 is an outlier with very large anchor distance (4.95)

### 4. **Both Methods Produce Negative Risk Scores**
- Direction Only: -2.97 average
- IPCW Only: -2.85 average
- Indicates interventions consistently move samples toward better prognosis

---

## Interpretation

### What is "Directional Consistency"?

The E4 audit tests whether the model's interventions are **directionally consistent**:
- Takes a test sample
- Applies transport-based interventions in the feature space
- Measures how consistently these interventions affect risk predictions
- Lower std risk = more consistent/reliable interventions

### Why Direction Only Wins on Consistency?

**Direction Only** (0.45 std) produces more consistent interventions than **IPCW Only** (0.76 std):

1. **Geometric Stability**: Using only direction (normalized vectors) removes magnitude-dependent noise
2. **Robust to Scaling**: Direction-only interventions are invariant to feature scale differences
3. **Cleaner Signal**: Focusing on direction isolates the core pathway effect from intensity variations

### Why IPCW Only Has Tighter Anchors?

**IPCW Only** (2.48 anchor dist) is slightly closer to training centroids:

1. **Censoring Correction**: IPCW weights handle censored data more carefully
2. **Sample Weighting**: Reweighting during training may create more central representations
3. **Less Aggressive**: IPCW-only interventions may be more conservative than direction-based ones

---

## Conclusions

✅ **Direction Only variant demonstrates superior directional consistency**
- 40.6% lower risk score variability
- More reliable/stable interventions across different samples

⚠️ **Direction Only shows higher cross-fold variability**
- May indicate sensitivity to training data distribution
- Fold 0 appears to be an outlier

✅ **IPCW Only shows good cross-fold stability**
- More consistent anchor distances across folds
- Slightly tighter embedding space

### Recommendation

**For directional consistency and intervention reliability:**
→ **Direction Only variant is preferred**

The lower risk score variability (0.45 vs 0.76) indicates that direction-based interventions produce more predictable and consistent effects, which is crucial for interpretability and clinical applicability.

---

## Next Steps

1. **Investigate Fold 0 Outlier**: Why does Direction Only fold 0 have such a large anchor distance (4.95)?

2. **Combine Both Methods**: Consider a hybrid approach that uses:
   - Direction for intervention consistency
   - IPCW for handling censoring

3. **Extend to Other Cancer Types**: Run the same E4 audit on:
   - UCEC, PAAD, LIHC, etc.
   - Validate consistency across cancer types

4. **Visualize Interventions**: Create t-SNE/UMAP plots showing:
   - Anchor positions
   - Intervention trajectories
   - Risk score changes

---

## Files Generated

- `e4_audit_summary.csv` - Raw results for all folds
- `e4_audit_direction_only_fold*.csv` - Individual fold results (Direction)
- `e4_audit_ipcw_only_fold*.csv` - Individual fold results (IPCW)
- `e4_audit_blca_fold0.csv` - Baseline BLCA results
- `e4_summary.csv` - Aggregated metadata

**Total Runtime**: ~30-40 minutes for 10 experiments
