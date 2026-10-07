# LLM session ledger

See `LLM_LEDGER.md` for the 2026-10-06 incident (LLM read a v3.10 file as
v3.13 main result and produced 8 unsupported "red flags").

## Hard rules for the LLM working in this repo

1. **Never trust a result file by its name alone.** Before quoting any
   number, confirm: file timestamp vs the version's first training date,
   schema version, source script, and the n_splits/n_seeds/n_epochs
   protocol. Cite all 4 in the first message that quotes a number.

2. **Each version has exactly ONE authoritative main result file.** For
   v3.13:
   - Main table: `results/multi_cancer_summary_v313.json`
     (uni2h 10-cancer, only 4 cancers completed: BRCA/COADREAD/LUAD/STAD;
     n=20 overall, mean=0.6842; vs v3.10 overall=0.6964)
   - Ablation (BLCA/KIRC): `results/dct_v313_ablation_Exp0..Exp6/`
     Exp6 = "Full" = the authoritative c-index for the v3.13 main claim
     (BLCA 0.7238 ± 0.0467, KIRC 0.8224 ± 0.0163)
   - The 5-cancer 5-fold (BLCA/HNSC/LUSC/SKCM/KIRC) was never produced
     for v3.13 in uni2h. If a file claims it, it is from v3.10 or v3.11.
   - The file `results/dct_main_table_cindex.json` (Sep 11, schema
     `fold_cindices/mean/std/n_folds`) is a v3.10 main table — do not
     quote it as v3.13 even though the filename has "main_table".

3. **Never fabricate "red flags" from a number alone.** A per-fold gap
   (e.g. fold-4 = 0.16 above the rest) is an observation. "Data leak /
   overfit / bimodal" is a hypothesis — state it as a hypothesis, and
   name what would need to be checked before claiming it as a finding.

4. **When multiple numbers disagree, treat that as a single finding, not
   4 separate findings.** State the pattern once, name the context for
   each number, do not list "red flags" that imply they are independent
   results.

5. **If about to produce an N-item red-flag list, stop and re-check the
   input data first.** Probability the list is wrong goes up sharply
   with list length. Items 1-3 may be real; items 4+ are usually filler
   from overconfidence.

---

## 2026-10-07 — Transport plan replacement sweep (Figure 3, v3.13 Full)

### What was produced

`scripts/plot_fig3_transport_sweep.py` was run for BLCA and KIRC across
all 5 folds with `alphas=(0.0, 0.25, 0.5, 0.75, 1.0)`. Per-fold outputs:

- `paper/figures/fig3_sweep_blca_fold{0..4}.json`
- `paper/figures/fig3_sweep_kirc_fold{0..4}.json`

Then `--plot` was invoked to produce the aggregate figure:

- `paper/figures/fig3_transport_sweep.png`
- `paper/figures/fig3_transport_sweep.pdf`

### Provenance (the 4 things rule #1 requires)

- **Source script**: `scripts/plot_fig3_transport_sweep.py`. Each fold
  loads `results/dct_v313_ablation_Exp6_full/<cancer>/<cancer>/SurvOTRank_dct_v313_transport_reconstruction/<seed>/model_best_s{fold}.pth`,
  calls `survot_rank.evidence.v313.replay(model, payload, alphas)`,
  reuses the trained `model._transport_wsi_to_omic` pathway, and
  recomputes C-index from the recovered sweep risk.
- **Split / seed**: `which_splits=5fold_uni2h`, `seed=3` (matches the
  Exp6 training config used in the v3.13 loss ablation).
- **Per-cancer n (validation patients per fold)**: BLCA = 76, KIRC = 97
  or 98 depending on fold. `n_samples` and `n_train` are stored inside
  each JSON.
- **Replay alignment**: `replay()` already enforces `alpha=0`
  `allclose` against the factual logits before any alpha is swept, so
  `alpha=0` re-produces the saved best-epoch prediction by construction.
  `survot_rank/evidence/v313.py` exports `replay` for exactly this audit
  use case (runbook §6.1).

### Per-fold C-index (fact = α=0)

| Cancer | f0  | f1  | f2      | f3  | f4  |
|--------|-----|-----|---------|-----|-----|
| BLCA   | 0.6576 | 0.6953 | 0.7465 | 0.7744 | 0.7453 |
| KIRC   | 0.8253 | 0.8485 | 0.8202→0.8208 | 0.8094 | 0.8084 |

KIRC fold 2 is the only non-zero α-shift in the whole 10-cell × 5-α grid
(Δ = +0.0007 between α=0 and α≥0.25; cindex jumps one notch and then
flat across α=0.25..1). The remaining 49 (cancer, fold, α) cells are
constant in C-index, while the underlying risk scalars shift very
slightly (`std` of `sweep_risk` differs at the 5th–6th decimal).

### What this means, stated as one finding, not N findings

Per rule #4, treat this as one observation: with the Full v3.13
checkpoint replayed under the factual-marginal plan replacement
`T_α = (1−α)T + α·col_normalize(T·1·(Tᵀ1)ᵀ / ΣT)` at all 4 stages ×
3 geometries simultaneously, the per-fold C-index on BLCA and KIRC is
essentially insensitive to α. This is the same pattern already
recorded in the older `paper/figures/v313_main_plots/fig3_sweep_summary.json`
(which aggregated BLCA-Exp6 over 3 folds and KIRC-Exp6 over 4 folds
with the same constant-in-α result); the new run simply fills the
remaining folds and adds the corresponding KIRC fold-2 micro-shift.

### What this does NOT support (do not claim)

- It is **not** evidence that the transport plan is "passenger-only"
  in the sense of rule #3 of this ledger. A constant C-index under
  fact-marginal replacement can arise from any of:
  (a) the risk head reading is dominated by terms that α does not move;
  (b) the factual plan's column-normalised marginal product is already
  very close to the factual plan itself on these patients, so α is a
  near-identity move;
  (c) the hazard head's stage-to-stone binding is so saturated that
  logit differences at this magnitude do not change rank order.
  All three are hypotheses; none are tested here.
- It is **not** a paired audit of the v3.13 Direct/Independent control
  arms. Those are still retracted (`DCT_v313_初稿` §4.4). Sweep is a
  same-model intervention, not a control-arm comparison.
- It is **not** a re-evaluation of the Exp0 vs Full loss ablation
  numbers in Table 2 of the manuscript draft; the only thing that
  changed in this run is the figure-3 sweep deliverable.

### Companion artefacts on disk

- `paper/figures/fig3_transport_sweep.{png,pdf}` — 2-panel plot
  (BLCA, KIRC), per-fold dashed lines + mean±std across folds,
  y-limits fixed to [0.60, 0.90].
- Per-fold JSONs carry `cindex` and `risk_stats.{mean,std,min,max,
  unique_n}` for each α. These are the raw numbers that produced the
  table above; cite them, not the plot, when quoting in the manuscript.

### Caveats specific to this run

1. Standalone BLCA fold-0 and the consolidated KIRC fold-1 runs each
   hit `RuntimeError: unable to mmap ... Cannot allocate memory (12)`
   on first attempt (terminals 729494, 729495). Re-running KIRC fold 1
   in isolation (`PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 ... --fold 1`,
   terminal 729496) succeeded and produced the JSON above. Failure
   was transient (parallel-run memory pressure), not a code bug.
2. The replay still uses the trained Full model's risk head without
   retraining, so this is a within-checkpoint intervention only.
3. `risk_stats` show the absolute risk scalars do move by ~1e-4 per
   α, which is too small to flip rank order in a 76- or 98-patient
   validation set. C-index is therefore expected to be flat here even
   if the model genuinely uses the plan.
