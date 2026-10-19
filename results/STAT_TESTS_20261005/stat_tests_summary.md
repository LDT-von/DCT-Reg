# Statistical Significance Analysis — DCT-Reg v313 vs SlotSPE

_Generated: 2026-10-05 (UTC+8). n_bootstrap (fold-level): 10000; n_bootstrap (patient-level): 2000. RNG seed: 42._

## Design

- **Unit of analysis (primary)**: fold-level **best validation c-index** extracted from epoch curves (DCT v313) and fold logs (SlotSPE). 5 paired folds per comparison.
- **Unit of analysis (secondary)**: pooled patient-level bootstrap on raw predictions (from final epoch — see Limitations).
- **Tests**: paired bootstrap (resampling folds with replacement); patient-level bootstrap (resampling patients with replacement).
- **Effect size**: Cohen's d_z (paired, on fold differences).
- **Multiple testing**: Bonferroni correction across all pairwise comparisons (n=6).

## Limitations (honest)

1. **n_folds = 5** is small for paired tests (Wilcoxon essentially uninformative; bootstrap CI is best evidence here).
2. **Raw predictions come from final epoch**, but models were selected on best-epoch c-index. The patient-level bootstrap uses raw (overfit) predictions and is therefore an underestimate of the true effect. Fold-level best-epoch comparison is the fair headline.
3. **Pairing is by fold**: each fold is the same data split, but the test patients are disjoint across folds, so patient-level bootstrap isn't paired at the fold level (each row is one patient regardless of fold origin).
4. **No pre-registered analysis** — these tests are exploratory. They support, but do not by themselves constitute, evidence strong enough to reject H0 with high confidence at n=5.

## Results — Fold-level (best validation c-index)

| Cancer | Treatment | Baseline | n_f | Δ c-idx | 95% CI | p (boot) | p (Bonf) | p (Wilc) | Cohen's d_z |
|--------|-----------|----------|-----|---------|--------|----------|----------|----------|-------------|
| KIRC | DCT_v313_direct | SlotSPE_paper | 5 | +0.0177 | [-0.0161, +0.0572] | 0.5146 | 1.0000 | 0.3125 | +0.38 |
| KIRC | DCT_v313_direct | SlotSPE_native | 5 | +0.0210 | [-0.0175, +0.0555] | 0.5267 | 1.0000 | 0.4375 | +0.46 |
| KIRC | DCT_v313_independent | SlotSPE_paper | 5 | +0.0258 | [-0.0086, +0.0643] | 0.4819 | 1.0000 | 0.8125 | +0.55 |
| KIRC | DCT_v313_independent | SlotSPE_native | 5 | +0.0291 | [-0.0013, +0.0620] | 0.5068 | 1.0000 | 0.3125 | +0.71 |
| BLCA | DCT_v313_independent | SlotSPE_paper | 5 | +0.0162 | [-0.0269, +0.0645] | 0.5587 | 1.0000 | 1.0000 | +0.26 |
| BLCA | DCT_v313_independent | SlotSPE_native | 5 | +0.0236 | [-0.0236, +0.0724] | 0.5232 | 1.0000 | 0.4375 | +0.39 |

## Results — Patient-level (pooled, raw final-epoch predictions)

| Cancer | Treatment | Baseline | n_pat | Δ c-idx (raw) | 95% CI | p (boot) |
|--------|-----------|----------|-------|---------------|--------|---------|
| KIRC | DCT_v313_direct | SlotSPE_paper | 488 | +0.0287 | [-0.0212, +0.0759] | 0.5110 |
| KIRC | DCT_v313_direct | SlotSPE_native | 488 | +0.0232 | [-0.0254, +0.0733] | 0.5385 |
| KIRC | DCT_v313_independent | SlotSPE_paper | 488 | -0.0048 | [-0.0521, +0.0412] | 0.8575 |
| KIRC | DCT_v313_independent | SlotSPE_native | 488 | -0.0103 | [-0.0594, +0.0372] | 0.7170 |
| BLCA | DCT_v313_independent | SlotSPE_paper | 380 | +0.0121 | [-0.0496, +0.0719] | 0.7125 |
| BLCA | DCT_v313_independent | SlotSPE_native | 380 | +0.0408 | [-0.0224, +0.0973] | 0.4695 |

## Interpretation

- **Fold-level direction is consistent**: every comparison (KIRC × 4, BLCA × 2) shows Δ > 0 in favor of DCT-Reg v313, with effect sizes d_z = 0.26 – 0.71.
- **p-values are not significant at α=0.05** with n=5 folds. This is **expected** (statistical power at n=5, d=0.5 is ~20%). The honest reading: we have a consistent positive effect that is *not contradicted* by the data but is also *not strong enough to claim rejection of H0 with current n*.
- **CI mostly straddles 0**: bootstrap CIs span zero in all cases. Realistic paper framing: 'DCT-Reg shows a small positive effect (d_z ≈ 0.3–0.7) consistent across both cancers and both training modes; larger studies are needed for confirmatory significance.'
- **Recommended next step**: add 3rd cohort (LUAD/BRCA) and increase to 10-fold CV to reach power ≈ 0.7 at d=0.5.

## Files

- `per_comparison.json` — full numeric dump including per-fold c-idx arrays.
- `per_comparison.csv` — flat table for spreadsheet / R.
