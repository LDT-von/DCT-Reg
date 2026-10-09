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


## 2026-10-07 — Manuscript integration verification of commit 24f3f10

This note corrects the preceding sweep description without changing its source
results. The committed source has `independent_plan = (T1)(T^T1)^T / sum(T)`
and mixes raw plans; there is no column normalization before interpolation.
Any normalization in the trained downstream model remains downstream.

There are 50 cells including ten factual baselines. Four cells change, all
KIRC fold 2 at nonzero alpha; 46 are equal to their fold baseline. Considering
only the 40 intervention cells, 36 are unchanged. The single finding is that
this within-checkpoint, factual-marginal replacement has little C-index effect.

The standalone driver calls `replay`, not `export_run`. Its alpha-zero check
compares two computations inside that replay; it does not load saved best
patient predictions. Deterministic settings newly added to `export_run` are
not executed through this driver. Four-decimal factual C-index agreement is
observed, but patient-level saved-best alignment is not established here.

The JSON has no schema version, source timestamp, patient IDs, checkpoint
hash, n_train, per-patient risk differences, or marginal residuals. Use the
2026-10-07 Git commit timestamp for this delivery, not checkout mtime.
Source: `scripts/plot_fig3_transport_sweep.py`, Exp6 checkpoint family,
`5fold_uni2h`, seed 3, maximum training budget 30 epochs; five folds per
cancer, BLCA n=76 each, KIRC n=98/98/98/97/97. Those are validation patients.
Risk summary statistics shift slightly; a changed distribution's moments do
not measure paired patient-level changes.

The original plot uses population SD (`ddof=0`). The manuscript Figure 4
uses the same ten JSONs with sample SD (`ddof=1`) for consistency with its
tables. Source Figure-3 files are preserved. The manuscript uses audited
20-run v2 control records, separate from the retracted historical Stage A;
the preceding blanket statement that Direct/Independent remain retracted
does not describe this later source. No new control audit or model run was
performed during manuscript editing. The ten-cancer table remains explicitly
author-supplied and lacks complete raw provenance in this checkout; the
partial main JSON does not validate it as a complete ten-cancer result.


## 2026-10-08 Existing figure integration

Inserted Figures 2, 3, 5, 6 and 7 into the v3.13 manuscript and added raw
cohort means as Supplementary Figure S3. Figure 6 contains only the verified
BLCA pathway-slot and transport panels; WSI, Top-5 tissue patches and the
KIRC case remain incomplete.

Cohort groups are within-fold validation-risk midrank quartiles, not
training-risk thresholds. KM groups retain per-fold training-risk medians.
Counts come from actual JSON/CSV, correcting inconsistent historical report
prose. The difference heatmaps subtract the unweighted four-group mean in
absolute attention units; no row z-score or significance claim is added.

The Full/control plot uses exact factual sweep scores and audited control
scores. Difference rounding is applied after subtraction, correcting the
previous rounded-means arithmetic. Original scores, exports and images are
unchanged. Plot sources, immutable numeric snapshots and output hashes are
recorded in figures/v313_manuscript_integrated/manifest.json.

Remote progress report 6ae840a records completed server Full experiments,
but their actual results were not present in this checkout and were not
used as manuscript evidence. This document operation starts no model job.

Validation completed: Word rebuilt with 12 embedded image panels, 9 tables and
15 native equation objects. LibreOffice was unavailable; Microsoft Word COM
exported the QA PDF. All 23 pages were rasterized and inspected, including a
second review of the two pages changed by the final wording correction. No
image clipping, overlaps or orphaned captions were found. All 14 source hashes,
19 generated asset hashes and 70 Appendix A fold values passed checks. Figure 6
remains explicitly partial; no training or inference was executed.

## 2026-10-08 Ten-cancer KM scope and experiment ordering correction

The author clarified that Figure 5 must cover all ten cancer cohorts. The prior
2-cohort insertion was incomplete: BLCA/KIRC panels remain real, but the eight
other cancer panels have no patient-level source materials in the local checkout.
Figure 5 is now explicitly partial (2/10), with KIRC indexed as panel e under the
paper's canonical ten-cancer order. No missing cohort curves or p-values are filled.

Reviewed the author's local SlotSPE PDF, particularly pp. 7, 8, 24, 25 and 32.
Table 1 precedes ablations; Figure 6 covers ten cohorts and Table 7 reports log-rank
and RMST statistics. This establishes a gap in coverage and experimental evidence;
its published scores and validation-median grouping are not imported as matched
DCT measurements or substituted for this project's training-median protocol.

Moved general implementation settings to 4.1; full ten-cohort results to 4.2;
external baselines to 4.3; merged loss configurations, results and branch-weight
interpretation into 4.4. Renumbered tables and corresponding body/appendix references.
Only document edits, source reading, plot-code checks and document rendering are
performed locally. The server task is provided as a prompt for the author to dispatch.


Final local verification: the three focused KM coverage checks passed. The bundled
Python has no pytest, matplotlib or lifelines; the full 13-test pytest target and
real ten-cohort plot rendering remain server work, and are not reported as passed.
Word COM produced 23 pages; every rasterized page was visually reviewed. The short
loss-results table now stays together, and the case pathway panel fits with its
formula and caption. The document retains 9 tables, 12 image panels and 15 native
equations. All 14 integrated-figure source hashes, 19 output hashes, 70 recorded
loss folds and unchanged 50 Full folds passed structural/value checks. Figure 5
remains explicitly 2/10 and Figure 6 tissue panels remain partial. No extraction,
training or inference was run locally.


## 2026-10-08 Ten-cancer model comparison table

The author requested the supplied paper-table style: a column for every cancer,
red/bold best results, underlined second results, and a highlighted DCT row.
Implemented native Word, vector PDF/SVG, PNG and HTML outputs from saved fold
records. The table input includes the reference-model roster and the unchanged
DCT author-supplied ten-cohort folds; baseline score cells remain null. No scores,
patient counts or ranks are copied from the reference screenshot. Default output
requires complete verified model/cohort sources and a shared audited protocol.
An explicit draft preview leaves incomplete columns unranked. Ties at displayed
precision share a dense rank; Overall uses unrounded cohort means.


Six functional unittest checks passed: displayed-value ties, DCT not forced into
the top two, unrounded macro mean, missing-cohort no-output behavior, protocol
mismatch and changed source hashes. Native Word red/underline formatting was
also checked using internal fixtures, which are not published model measurements.
The delivered draft preserves the real 50 DCT fold values and their macro mean.
Its 20-row/13-column native Word table renders to one page; the final page and
vector-derived PNG were visually inspected after correcting a default title
border and a wrapped modality header. All draft columns remain unranked because
baseline data are missing. No training, inference or feature processing was run.


## 2026-10-08 Complete published-value reference comparison

Corrected the previous scope error: the author's requested paper-table reference
can be filled directly from the supplied SlotSPE PDF, without server-reproduced
baseline records. Extracted all 180 cohort means and 18 published Overall values
from Table 1 (PDF page 7); cross-checked all 180 means against appendix Table 4
(page 22), retaining all 180 standard deviations. Added the unchanged 50 author
DCT v3.13 folds; no baseline fold arrays were invented. The complete 19-model,
ten-cancer table is integrated into manuscript section 4.3 before the ablations.

Ranks are computed across all rows using three-decimal displayed values, with
ties sharing dense ranks. DCT ranks first in KIRC, LUAD and BLCA; second in LUSC,
HNSC and Overall. Published baseline Overall values are preserved; DCT Overall is
the equal mean of ten unrounded cohort means (0.702956). This is explicitly a
reported-reference comparison: the published models use UNI and DCT uses UNI2-h,
and patient splits are not aligned. Matched reproduction remains independently
pending; no model training, inference or feature processing was run.


Final verification: all nine comparison-table tests passed, including preservation
of all published Overall values, absence of fabricated baseline folds and isolation
from matched ranking. All 209 score-cell styles match computed ranks. Source PDF,
input and output SHA256 values were checked, and all 50 DCT folds are unchanged.
Native editable comparison Word is one landscape page; the complete vector-derived
PNG and final Word page were inspected. The manuscript's 23 final pages were all
visually reviewed; it retains 8 native tables, 13 image panels and 15 native math
nodes. The loss-recipe table stays together after the new comparison image.
Remote progress commit d2d4877 was preserved before this scoped delivery.

## 2026-10-08 OT and UNI2-h comparison focus

Fetched origin and confirmed d2d4877 is already preserved in current main history.
Section 6 records five diagnostic types on BLCA/KIRC, not a model comparison.
Its server figures_summary directory is absent locally; the explanatory guide
is grounded in the pushed summary script, not an asserted visual audit of server images.
Corrected the interpretation of Direct: prediction OT is retained.

Author changed baseline focus away from SlotSPE to OT or UNI2-h survival methods.
The active manuscript reference table now contains MOTCat reproduced in TTA,
TTA, OTSurv, and STEPH plus five UNI2-h MIL baselines from its official supplement.
Forty-nine published means align by exact cancer names. Full original cohort
names/means/dispersion and source SHA256 are retained, alongside unchanged 50 DCT
fold values. CRC, KIPAN, LUNG and STES are not relabelled as DCT cohorts.
LUSC is unranked because no external separate-cohort result is available.
Different cohort-set Overall is not ranked. Red and underline denote descriptive
ranks of available DSS report values, not matched independent test performance.
ProtoPathway OS results remain separate; ME-Mamba endpoint is not assumed.
MCSP-OTMR publisher abstract already describes OT reconstruction; the novelty
claim is limited to the concrete transport reuse and stage/event/pathway design,
with full-text comparison still required. Old SlotSPE artifacts remain historical.

No model training, inference or feature processing was started. Fourteen focused
provenance/ranking tests pass. All 100 score-cell styles were checked against
computed ranks; all output hashes and the source input hash match. Standalone
Word has one page, visually inspected. All 23 manuscript pages were inspected,
retaining 8 native tables, 13 image panels and 15 native math nodes.


## 2026-10-08 Expanded comparison table and verified years

Author requested reuse of the other-method results from the original SlotSPE
table, with every method dated and all SlotSPE model rows excluded. Restored
all 15 non-SlotSPE baseline rows, preserving 150 cohort means, 150 dispersions
and 15 published ten-cohort Overall values exactly. Retained six unique extra
methods from the OT/UNI2-h sources: ILRA, R2T-MIL, Patch-GCN, OTSurv, TTA and
STEPH. MOTCat, ABMIL and TransMIL consistently use the complete ten-cohort
source, avoiding duplicate methods and cellwise selection from different reports.
The active table has 21 baselines plus DCT, 183 aligned published cohort means,
and unchanged 50 real DCT fold values. Every cancer now has comparators. Overall
is populated only for the same complete ten cohorts; subset-source means stay NR.

Every row has a verified year/source. MLP 1998 is the cited textbook year and
SNNTrans 2021 is a component-reference year, both explicitly marked. TTA 2025
is its first preprint year, while the values are from its 2026 v2. DCT 2026 is
the current work year, not a publication claim. Primary year references and
source hashes are recorded in the input and literature guide. The R2T-MIL name
is retained from STEPH S2 and its original RRT-MIL paper title is explained.

All 18 focused data/provenance/ranking tests passed. All 242 score-cell texts,
red first-place styles and second-place underlines match computed ranks.
Input/output hashes were checked; the manuscript embeds the current comparison
image. The native editable comparison Word fits one landscape page and was
visually checked after fixing the COADREAD header wrap. The vector preview and
all 23 manuscript pages were visually reviewed; the manuscript retains 8 native
tables, 13 embedded panels and 15 native math nodes. No model training, inference
or feature processing was run. Cross-paper ranks remain descriptive public-value
references with explicit input/split/checkpoint differences, not matched results.

## 2026-10-08 v3.13 additional diagnostic bug audit

Audited the delivered server-summary code without starting real patient
export, inference or training. Confirmed the historical progress confused
reconstruction/native and shuffled mean_error with C-index. Those CSV rows
were never C-index; the previous nearly-unchanged shuffled-ranking inference
is withdrawn until pairing JSON/raw arrays are verified.

Fixed double LayerNorm of the arithmetic training-token mean used for
centered retrieval. Two new regression tests failed on old code and pass
with the fix. Retrieval v2 preserves and exports the actual arithmetic
mean; old Top-1/MRR requires reconstruction-only re-export from the same
checkpoints and settings. The fix leaves trained model/objectives, latent
reconstruction distance, pairing risks and patch interventions unchanged.

Replaced the unvalidated hard-coded summary script with an explicit CLI:
checked array hashes, native/shuffled C-index, patient/fold identities,
pathway order, actual repeat counts, fold-specific chance levels and budget
record fractions. Added metric/direction to CSV, with distinct pairing and
budget rows. Legacy retrieval is blocked from corrected plots; the other
four classes can be redrawn from checked legacy arrays. An optional Full
paper-fold consistency check stops mismatched inputs before output.

All 38 focused tests passed (additional evidence, summary, core evidence).
Five synthetic summary PNGs were visually checked; these QA images are
not scientific results and are not delivered as patient figures. Raw
server additional arrays and real ten diagnostic images remain absent
locally. A server prompt preserves old exports, redraws eight unaffected
plots and re-exports only the two corrected retrieval plots. No manuscript
scores, baseline table, model code, weights or training jobs were changed.

## 2026-10-08 reference-style training table, pathway composite and cost panels

User supplied a ten-cohort ablation table, performance/memory/time scatterplots,
and patient slot-pathway composite. Added a no-inference table CLI using the
70 recorded loss folds, 20 audited mechanism-control folds and unchanged 50
Full folds. BLCA/KIRC are complete across nine rows; the other eight cohorts
remain explicitly NR outside Full. Historical single-branch 0.025 weights
are labelled, not renamed as matched-weight removals. Red/underline reflect
available recorded means; one-entry columns have no competitive ranking.

Added a single-patient full pathway matrix alongside all slots' Top/Bottom-3.
Raw is default; optional tied midrank percentiles retain all raw values and
ranges. Uniform weights stay 0.5, rather than manufactured percentile contrast.
Long pathway names determine panel height. No actual case NPZ is local, so
new real composite awaits read-only server plotting, not guessed matrix values.

Expanded profile provenance without changing model/training/prediction paths:
CUDA/precision/input fingerprints and timing settings, plus benchmark patient.
Cost plotting recomputes C-index from raw best predictions, requires complete
matched folds and patient/outcome/train/split and WSI-input agreement, checks
source hashes and identical hardware/measurement settings. Baseline native
load/forward adapters remain in their own implementations; no claim that DCT
loader supports arbitrary methods. Measured-input template has no fake scores.

All 55 focused tests passed. Two real-record table PDFs/PNGs were visually
checked; synthetic composite and cost QA are explicitly synthetic and excluded
from final scientific artifacts. No real patient inference/training started.
Server instructions reuse existing outputs/checkpoints only; missing ablation
or matched baselines are reported rather than automatically trained. Existing
manuscript and external comparison table remain unchanged.

推送整合：保留远端 7bfec1f、acb7cbc；十癌种 KM 文件和 10 个来源 JSON LF 哈希通过只读核对，V313_MANUSCRIPT_STATUS 的 Figure 5 交付覆盖更新为 10/10，患者/checkpoint/split 审计仍列为待补。新消融表仍保留 a1a4584 来源快照，不混入后续正文变更。

### 2026-10-08：b3147b7 新增真实图复核与图注修正

- 拉取 f9b951c/bdf4c16/b3147b7，核对组合病例 8×329 raw 一致性、成本来源输入与三个图产物哈希、逐折均值。
- 正文修正 17 个图片相对路径；图 6c/6e 数量级、色标和配色按实际图/代码修正；图 8 明确延迟是五折中位数的均值及其折间波动，保留全部实测数值。
- 当前工作树缺少服务器 20 个预测/profile 源文件，未独立重做源审计；原服务器 pass 与本机文件审查边界分开记录。
- 十队列九行表缺失记录数纠正为 320；Full 已有结果不重训，先服务器只读 inventory，strict matched 0.05 单列。
- 更新状态记录；无模型代码或图数据改动，真实训练/推理次数为 0。复核见 paper/V313_NEW_FIGURES_REVIEW_20261008.md。

### 2026-10-09：E045 v313 模块作用核验与可视化

- 来源：当前 main 9e81a04 及保留的 E007–E010 原始记录；区分继承框架与 v3.13 运输感知通路 token 重建。
- 两张真实历史记录再分析图保留逐折负值与近零固定模型响应；不新增显著性或机制成功声明。
- 新 replay 入口要求实际五折、患者/结局/最佳预测一致、原始哈希、通路一致与同边际残差；绘制排序、风险/IQR、重建误差和槽诊断。7项针对关键失败条件的校验通过。
- 代码与图表分别归档到 E045 的专用目录；备份后续写 Word 主档，下一编号 E046。真实训练/患者推理为0；服务器新导出和修正版检索结果仍待执行。
- 完整 Word 页与图 PDF 已渲染检查；最终核验及备份哈希见 实验档案/维护资料/E045_续写核验.json。
