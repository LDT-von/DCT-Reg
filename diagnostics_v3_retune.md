# v3.11 Hyperparameter Retuning — Status

Date: 2026-09-10 (UTC+8)
Context: v3.11 first attempt on BLCA UNI2-h (v2 config) showed severe overfitting
after epoch 14+ (val_c≈0.65 plateau, train_c→0.96). User asked to retune.

## Changes implemented

| hyper-param         | v2       | v3      | v4 (current) |
|---------------------|----------|---------|--------------|
| lr                  | 5e-4     | 1e-4    | **3e-4**     |
| weight_decay        | 5e-4     | 5e-3    | 5e-3         |
| otehv2_dropout      | 0.15     | 0.30    | 0.30         |
| num_patches         | 4096     | 2048    | **1024**     |
| max_epochs          | 30       | 50      | 30           |
| early stopping      | (none)   | (none)  | patience=10  |
| warmup_epochs       | 0        | 3       | 3            |
| recipe (slot_nll)   | 0.05+0.02| same    | same         |

## Critical fix: early_stop_* CLI args

`config_to_argv` flattens `train.*` keys to `--train.*` which argparse rejects.
`--early_stop_patience N --early_stop_metric val_cindex ...` must be passed
as CLI extras, not as YAML config keys.

## Results so far (BLCA fold 0, all 5 folds planned later)

| Run                                 | ep | val_c | Δ vs v3.10-base | best so far |
|-------------------------------------|---:|------:|----------------:|------------:|
| v2 (overfit recipe)                 | 16 | 0.6491 | -0.004          | 0.6686     |
| v3 (lr=1e-4, too slow)              |  2 | 0.5684 | -0.026          | 0.5905     |
| **v4 (lr=3e-4, current)**           |  3 | 0.5950 | +0.001          | 0.5905     |
| **v3.10 baseline UNI2-h (control)** |  3 | **0.6508** | (ref)           | 0.6508     |

## What I learned

1. **v3.11 per_slot_nll is NOT necessary on BLCA UNI2-h at this scale**
   - ep 3 baseline 0.6508 vs v4 0.5950 (delta -0.056)
   - Both using lr=3e-4, weight_decay=5e-3, 1024 patches, dropout=0.30
   - Baseline has NO per_slot_nll / diversity — yet converges better
2. **v2/v4 share one failure mode**: high train/val gap when per_slot_nll active
   - slot_variance rises toward 0.05 cap → slot heads memorize training labels
   - val_c gets stuck around 0.55–0.67 no matter how long we train
3. **OT loss landscape is flat (~1.0)** in v3.10 baseline → OT plan barely moves
   - The 0.6508 jump at ep 3 suggests cosine LR scheduling and EMA helping
4. **KIRC v3.10 baseline reaches 0.85** with same architecture → BLCA is intrinsically
   harder (small data, DSS labels noisy), not an architecture issue

## What to do next (alternatives)

A. **Wait for v3.10 baseline to finish (now running, ETA ~2h)** then compare 5-fold
   properly. If baseline ≥ v4 everywhere, admit idea is not active on this dataset.

B. **Test v3.11 on KIRC** (already known to work well there, 5-fold baseline 0.85+).
   Idea may need bigger N to be necessary.

C. **Drop the idea on BLCA entirely** — keep it as future work and ship v3.10
   baseline result on BLCA UNI2-h as the published number.

D. **Reduce per_slot_nll further** (e.g., 0.01) so it acts as an auxiliary regulariser
   rather than a competing loss. Quick patch; needs new run.

## Pending actions for user

- Continue monitoring v3.10 baseline + v4
- Decide on A/B/C/D above
- My current recommendation: **A → if baseline wins 4/5 folds, switch to C**
