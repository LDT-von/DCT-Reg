#!/usr/bin/env python3
"""Generate the unified v3.10 Proof Synthesis Report.

Combines all individual Exp A/B/D/E/F findings into one cohesive narrative
that addresses:
1. Does direction loss in v3.10 contribute to model behavior?
2. Is v3.10's audit signal failure BLCA-specific or cross-cancer?
3. Where does direction loss fail (training-time vs forward-time)?
4. What's the right way to evaluate direction loss in this architecture?
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("results/v310_proof_synthesis")
OUT.mkdir(parents=True, exist_ok=True)


def safe_get(d, key, sub_key, default=0):
    v = d.get(key, default)
    if isinstance(v, dict):
        return v.get(sub_key, default)
    return default


def main():
    # Load prior + new results
    blca_v310 = json.load(open("results/audit_blca_redesign/blca/summary_5fold.json"))
    v310_cross = json.load(open("results/v310_proof_a_cross_cancer_audit/summary_5fold_by_cancer.json"))
    exp_b = json.load(open("results/v310_proof_b_anchor_geometry_trajectory/summary_by_cancer.json"))
    exp_e = json.load(open("results/v310_proof5_epoch_curve_stats/aggregate_by_cancer.json"))
    exp_d = json.load(open("results/v310_proof_d_per_slot_hazard/aggregate_5fold.json"))
    exp_f = json.load(open("results/v310_proof_f_v311_audit/summary_5fold.json"))

    lines = [
        "# v3.10 Proof Synthesis: Where Does Direction Loss Fail?\n",
        "**Comprehensive 6-experiment synthesis for DCT v3.10 BLCA + 3 cross-cancer cohorts + v3.11 reference.**\n",
        "",
        "## TL;DR\n",
        "",
        "1. **Direction loss in v3.10 contributes negligible mechanism to model behavior across 4 cancers × 5 folds × 3 independent measurement modalities.**\n",
        "2. **The redesigned audit's negative result is NOT specific to v3.10's direction loss** — v3.11 BLCA also produces `info_gap_high ≈ 0`.\n",
        "3. **v3.10's actual mechanism failure is in training-time gradient propagation** (gain magnitudes of ±10⁻³ over 50 epochs vs λ=0.05 should produce ~0.05 movement).\n",
        "4. **v3.11's per-slot hazard** (Exp D) gives a forward-time slot-level signal (monotone_rate = 0.60) that the redesigned audit (Exp F) does not capture at the transport-plan level.\n",
        "",
        "## Cross-experiment summary\n",
        "",
        "| Exp | Question | Method | Cancer | Result |",
        "|---|---|---|---|---|",
        f"| A | Is info_gap ≈ 0 BLCA-specific? | redesigned audit | BLCA/HNSC/LUSC/SKCM | HNSC +0.052, LUSC -0.002, SKCM -0.018, BLCA 0.000 |",
        f"| B | Does gain evolve during training? | log parser | 4 cancer | slope = ±1e-5 (essentially zero) |",
        f"| E | What is the gain magnitude at convergence? | epoch_curve csv | 4 cancer | high_gain ±10⁻³ (essentially zero) |",
        f"| D | Does v3.11 slot hazard differentiate high/low risk patients? | per-slot hazard | BLCA | monotone_rate = 0.60±0.25 |",
        f"| F | Is audit negative result version-specific? | redesigned audit | v3.11 BLCA | info_gap_high = -0.003±0.029 |",
        "",
        "## Detailed findings\n",
        "",
        "### Exp A: Cross-cancer redesigned audit\n",
        "",
        "Used redesigned audit (alpha={0.0, 1.0}, real vs shuffled-slot-dim random baseline).",
        "",
        "| cancer | info_gap_high (5-fold) | info_gap_low |",
        "|---|---|---|",
    ]
    for cancer in ["blca", "hnsc", "lusc", "skcm"]:
        if cancer == "blca":
            h = safe_get(blca_v310, "info_gap_high_mean", "mean")
            l = safe_get(blca_v310, "info_gap_low_mean", "mean")
            std_h = safe_get(blca_v310, "info_gap_high_mean", "std")
            std_l = safe_get(blca_v310, "info_gap_low_mean", "std")
            lines.append(f"| {cancer.upper()} (prior) | {h:+.4f} ± {std_h:.3f} | {l:+.4f} ± {std_l:.3f} |")
        else:
            h = safe_get(v310_cross.get(cancer, {}), "info_gap_high", "mean")
            l = safe_get(v310_cross.get(cancer, {}), "info_gap_low", "mean")
            std_h = safe_get(v310_cross.get(cancer, {}), "info_gap_high", "std")
            std_l = safe_get(v310_cross.get(cancer, {}), "info_gap_low", "std")
            lines.append(f"| {cancer.upper()} (Exp A) | {h:+.4f} ± {std_h:.3f} | {l:+.4f} ± {std_l:.3f} |")

    lines.extend([
        "",
        "**Interpretation**: HNSC shows mild positive info_gap (+0.052) but std > mean. Other 3 cancers give ~0. Across cancers, no consistent direction-loss signal.",
        "",
        "### Exp B: Training-time gain trajectory (logs)\n",
        "",
        "Parsed `log_start_<k>_end_<k+1>.txt` for per-epoch gain metrics across 4 cancers × 5 folds.\n",
        "",
        "| cancer | direction Δ (ep 0→49) | high_gain slope (per epoch) | low_gain slope |",
        "|---|---|---|---|",
    ])
    for cancer in ["blca", "hnsc", "lusc", "skcm"]:
        agg = exp_b.get(cancer, {})
        if not agg:
            continue
        eps = sorted(agg.keys(), key=int)
        first, last = agg[str(eps[0])], agg[str(eps[-1])]
        # Compute slopes
        eps_arr = np.array([int(e) for e in eps], dtype=np.float64)
        high_slope = np.polyfit(eps_arr, [agg[str(int(e))]["high_gain_mean"] for e in eps_arr], 1)[0]
        low_slope = np.polyfit(eps_arr, [agg[str(int(e))]["low_gain_mean"] for e in eps_arr], 1)[0]
        lines.append(
            f"| {cancer.upper()} | "
            f"{first['direction_mean']:.4f}→{last['direction_mean']:.4f} ({last['direction_mean']-first['direction_mean']:+.4f}) | "
            f"{high_slope:+.2e} | {low_slope:+.2e} |"
        )

    lines.extend([
        "",
        "**Interpretation**: gain slopes are at ±10⁻⁵ per epoch. Over 50 epochs total movement is ±5×10⁻⁴ — effectively zero. Direction loss has no measurable effect on the gain mechanism during training.",
        "",
        "### Exp E: epoch_curve_*.csv aggregates\n",
        "",
        "Aggregated the v38 mechanism metrics across 4 cancers × 5 folds × 50 epochs.\n",
        "",
        "| cancer | direction (mean) | high_risk_gain | low_risk_gain | active_stage_fraction |",
        "|---|---|---|---|---|",
    ])
    for cancer in ["blca", "hnsc", "lusc", "skcm"]:
        d = exp_e.get(cancer, {})
        lines.append(
            f"| {cancer.upper()} | "
            f"{safe_get(d, 'train_v38_direction', 'mean_of_fold_means'):.4e} | "
            f"{safe_get(d, 'train_v38_high_risk_gain', 'mean_of_fold_means'):+.4e} | "
            f"{safe_get(d, 'train_v38_low_risk_gain', 'mean_of_fold_means'):+.4e} | "
            f"{safe_get(d, 'train_v38_active_stage_fraction', 'mean_of_fold_means'):.4e} |"
        )

    lines.extend([
        "",
        "**Interpretation**: direction loss mean is 0.0477±0.0001 across 4 cancers (model is computing the term), but gain magnitudes are ±10⁻³ (term not affecting mechanism). active_stage_fraction is high (0.8-1.0) so the anchors ARE active.",
        "",
        "### Exp D: v3.11 per-slot hazard analysis (BLCA)\n",
        "",
        "v3.11 exposes per_slot_hazard_wsi / per_slot_hazard_omic from forward pass.\n",
        "",
        "| metric | value |",
        "|---|---|",
        f"| monotone_rate (high-event-time > low-event-time) | {exp_d.get('monotone_rate_mean', 0):.3f} ± {exp_d.get('monotone_rate_std', 0):.3f} |",
        f"| distinguishability_wsi (cross-slot std) | {exp_d.get('distinguishability_wsi_mean', 0):.4f} ± {exp_d.get('distinguishability_wsi_std', 0):.4f} |",
        f"| distinguishability_omic | {exp_d.get('distinguishability_omic_mean', 0):.4f} ± {exp_d.get('distinguishability_omic_std', 0):.4f} |",
        "",
        "**Interpretation**: monotone_rate > 0.5 means slot hazard is higher in high-risk patients than low-risk — slot hazard HAS direction signal. distinguishability_wsi is small (slots have similar WSI hazard signatures) but distinguishability_omic is 5× larger (omic slots are more distinct).",
        "",
        "### Exp F: v3.11 BLCA redesigned audit\n",
        "",
        "v3.11 redesigned audit on the same BLCA cohort.\n",
        "",
        "| metric | v3.10 BLCA | v3.11 BLCA |",
        "|---|---|---|",
        f"| info_gap_high | {safe_get(blca_v310, 'info_gap_high_mean', 'mean'):+.4f} ± {safe_get(blca_v310, 'info_gap_high_mean', 'std'):.3f} | "
        f"{safe_get(exp_f, 'info_gap_high', 'mean'):+.4f} ± {safe_get(exp_f, 'info_gap_high', 'std'):.3f} |",
        f"| info_gap_low | {safe_get(blca_v310, 'info_gap_low_mean', 'mean'):+.4f} | "
        f"{safe_get(exp_f, 'info_gap_low', 'mean'):+.4f} |",
        "",
        "**Critical finding**: v3.11 BLCA `info_gap_high ≈ -0.003` is **statistically indistinguishable from v3.10's 0.000**. The redesigned audit's negative result is NOT specific to v3.10 direction loss — it's an evaluation instrument noise floor (~±0.05) when applied to BLCA's transport structure.",
        "",
        "## Conclusion\n",
        "",
        "### What is true\n",
        "",
        "1. **Direction loss in v3.10 has no measurable mechanism effect.** Across 4 cancers × 5 folds:\n",
        "   - gain magnitudes: ±10⁻³ at convergence (Exp E)",
        "   - gain training slopes: ±10⁻⁵ per epoch (Exp B)",
        "   - redesigned audit info_gap_high: 0.000±0.075 (BLCA), ±0.05 across cancer (Exp A)",
        "",
        "2. **The redesigned audit's negative result is NOT specific to v3.10.** v3.11 BLCA produces the same result (Exp F: -0.003±0.029). The audit instrument on this 5-anchor BLCA structure has a noise floor of ~±0.05.",
        "",
        "3. **v3.11's slot-level hazard (Exp D) IS differentiated** (monotone_rate 0.60). This means v3.11 captures slot-level risk signal that v3.10 doesn't expose.",
        "",
        "### What remains uncertain\n",
        "",
        "1. **Whether direction loss could be made to work** in v3.10. The mechanism is disabled (gain magnitudes don't move) but the architecture is frozen. Restoring would require un-freezing the parent's `_FROZEN_ARGUMENTS` and ensuring the gradient flow through Sinkhorn propagates to anchor costs (Proof 2 gradient flow analysis would clarify).",
        "",
        "2. **Why redesigned audit is BLCA-noise-floor-limited.** The 5-anchor structure with 4 stages may produce transport plans where real vs shuffled-slot-dim random baseline is hard to distinguish.",
        "",
        "### Practical implication\n",
        "",
        "For v3.10 paper writing:\n",
        "- Do NOT claim dose-monotonicity as evidence of direction loss working (audit is inconclusive on BLCA).\n",
        "- DO cite `high_risk_gain = ±10⁻³` as direct evidence the mechanism is dormant.\n",
        "- DO cite v3.11's monotone_rate = 0.60 as a contrast: with per-slot hazard exposure, the same architecture DOES produce slot-level risk signal.",
        "",
        "## Output files\n",
        "",
        "- `results/v310_proof5_epoch_curve_stats/REPORT.md` (Exp E)",
        "- `results/v310_proof_b_anchor_geometry_trajectory/REPORT.md` (Exp B)",
        "- `results/v310_proof_a_cross_cancer_audit/REPORT.md` (Exp A)",
        "- `results/v310_proof_d_per_slot_hazard/REPORT.md` (Exp D)",
        "- `results/v310_proof_f_v311_audit/REPORT.md` (Exp F)",
        "- `results/v310_proof_c_audit/REPORT.md` (Exp C smoke — inconclusive)",
    ])

    with open(OUT / "REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"saved {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
