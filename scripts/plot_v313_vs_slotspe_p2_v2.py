"""P2 comparison: DCT v3.13 vs SlotSPE — CORRECTED VERSION (2026-10-21)

What this figure shows (and does NOT show)
------------------------------------------
v3.13 BLCA fold0 seed3 (epoch 29, last checkpoint):
  - Euclidean distances are REAL MEASURED values from epoch CSV.
  - Cosine similarities are NOT measured; they are estimated from distance
    via the lower-bound formula under the UNVERIFIED assumption that all 8 slots
    have equal norm ≈ 21.87.  The claim "56/56 pairs exceed cos > 0.95"
    is therefore UNSUPPORTED.
  - Only the min_pair_distance is a genuine single-number statistic from CSV.

SlotSPE BLCA fold0 seed3 (epoch 3, last checkpoint):
  - Per-patient cosine values ARE REAL MEASURED values from diagnostic.json.
  - 77 val patients, 56 off-diagonal pairs each, all > 0.95.
  - Source: results/slotspe_proof/p1246_diagnostic_blca_fold0_20261020/diagnostic.json
  - NOTE: this is a SINGLE checkpoint; not a 5-fold mean.

The two models use DIFFERENT measurement protocols:
  - v3.13: aggregate per-epoch CSV statistics (mean/min over patients and pairs)
  - SlotSPE: per-patient per-pair cosine from slot vectors
Putting them side-by-side on the same y-axis requires stating this limitation.

What this figure CANNOT claim
-----------------------------
  ✗ "v3.13 all 56 pairs cos > 0.95"  ← cos never directly measured
  ✗ "both models have the same cos"  ← different measurement protocols
  ✗ "v3.13 beats SlotSPE on cos"    ← no direct comparison possible
  ✗ "min distance proves all pairs"  ← min pair ≠ all pairs
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/research/dct_v313_slot_diversity_figure"

# ── v3.13 source: full_wsi_three_decoders, BLCA fold0 seed3 ──────────────
V313_CSV = (
    ROOT
    / "results/dct_v313_full_wsi_three_decoders/blca_seed3/blca/"
      "SurvOTRank_dct_v313_transport_reconstruction/"
      "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_"
      "seed3_rW_8_rG_8_sp_dct_v313_full_wsi_three_decoders/"
      "epoch_curve_fold0.csv"
)

# ── SlotSPE source: post-training diagnostic, fold0 seed3, 77 val patients ─
SPE_JSON = (
    ROOT
    / "results/slotspe_proof/p1246_diagnostic_blca_fold0_20261020/"
      "diagnostic.json"
)

# SlotSPE init max cosine from FINAL_RECEIPT.md §3
SPE_INIT_COS = 0.5605
COS_THRESH = 0.95

# v3.13 slot norm at epoch 29 (batch-level average, not per-slot)
V313_NORM = 21.87


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_v313() -> dict:
    import csv
    rows = list(csv.DictReader(open(V313_CSV)))
    def _c(prefix):
        return [float(r[prefix]) for r in rows]
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "wsi": {
            "pair_term":      _c("train_v313_diversity_wsi_pair_term"),
            "mean_dist":      _c("train_v313_diversity_wsi_mean_pair_distance"),
            "min_dist":       _c("train_v313_diversity_wsi_min_pair_distance"),
            "violating_frac": _c("train_v313_diversity_wsi_violating_pair_fraction"),
            "dup_frac":        _c("train_v313_diversity_wsi_duplicate_pair_fraction"),
            "collapse_frac": _c("train_v313_diversity_wsi_full_collapse_patient_fraction"),
        },
        "omic": {
            "pair_term":      _c("train_v313_diversity_omic_pair_term"),
            "mean_dist":      _c("train_v313_diversity_omic_mean_pair_distance"),
            "min_dist":       _c("train_v313_diversity_omic_min_pair_distance"),
            "violating_frac": _c("train_v313_diversity_omic_violating_pair_fraction"),
            "dup_frac":        _c("train_v313_diversity_omic_duplicate_pair_fraction"),
            "collapse_frac": _c("train_v313_diversity_omic_full_collapse_patient_fraction"),
        },
    }


def load_slotspe() -> dict:
    with open(SPE_JSON) as fh:
        raw = json.load(fh)
    p2 = raw.get("p2", {})
    max_coses = p2.get("per_patient_max_offdiag_cos", [])
    return {
        "n_patients": p2.get("n_patients", len(max_coses)),
        "max_coses": max_coses,
        "mean_max_cos": float(np.mean(max_coses)) if max_coses else None,
        "min_max_cos": float(np.min(max_coses)) if max_coses else None,
        "max_max_cos": float(np.max(max_coses)) if max_coses else None,
        "fraction_all_above": p2.get("fraction_patients_with_offdiag_high_cos"),
        "p6_mean_max_diff": raw.get("p6", {}).get("mean_max_diff"),
        "p6_max_diff": raw.get("p6", {}).get("max_diff"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(out_png: Path) -> None:
    v313 = load_v313()
    spe  = load_slotspe()
    ep29 = len(v313["epoch"]) - 1

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "P2 (slot redundancy): v3.13 vs SlotSPE — measured statistics only\n"
        "⚠ cosine NOT directly measured for v3.13  |  ⚠ different protocols not directly comparable",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── (a) v3.13: Euclidean distance curves ────────────────────────────────
    ax = fig.add_subplot(2, 2, 1)
    ep = v313["epoch"]
    ax.plot(ep, v313["wsi"]["mean_dist"], "o-", ms=3, lw=1.8,
            color="#0072B2", label="WSI mean")
    ax.plot(ep, v313["wsi"]["min_dist"],  "s--", ms=3, lw=1.8,
            color="#0072B2", alpha=0.65, label="WSI min")
    ax.plot(ep, v313["omic"]["mean_dist"], "o-", ms=3, lw=1.8,
            color="#CC79A7", label="OMIC mean")
    ax.plot(ep, v313["omic"]["min_dist"],  "s--", ms=3, lw=1.8,
            color="#CC79A7", alpha=0.65, label="OMIC min")
    ax.set_yscale("log")
    ax.set_ylabel("Euclidean distance (log)")
    ax.set_xlabel("epoch")
    ax.set_title(
        "(a) v3.13: Euclidean pair distance (BLCA fold0 seed3)\n"
        "solid = mean; dashed = min",
        fontweight="bold", fontsize=11,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    # annotate ep29 values
    ax.annotate(
        f"ep29 WSI min={v313['wsi']['min_dist'][ep29]:.3f}\n"
        f"ep29 WSI mean={v313['wsi']['mean_dist'][ep29]:.2f}",
        xy=(ep[ep29], v313["wsi"]["min_dist"][ep29]),
        xytext=(ep[ep29] - 7, v313["wsi"]["min_dist"][ep29] * 3),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    # ── (b) SlotSPE: per-patient max cosine histogram ───────────────────────
    ax = fig.add_subplot(2, 2, 2)
    bins = np.linspace(0.95, 1.001, 16)
    ax.hist(spe["max_coses"], bins=bins, color="#D55E00", alpha=0.8,
            edgecolor="white", lw=0.5)
    ax.axvline(COS_THRESH, color="red", lw=1.5, ls="--",
               label=f"threshold = {COS_THRESH}")
    ax.set_xlabel("max off-diagonal cosine (per patient)")
    ax.set_ylabel("number of patients")
    ax.set_title(
        f"(b) SlotSPE: per-patient max off-diagonal cosine\n"
        f"(77 val patients, 56 pairs each) — REAL MEASURED",
        fontweight="bold", fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0.95, 1.002)
    stats = (
        f"mean = {spe['mean_max_cos']:.4f}\n"
        f"min  = {spe['min_max_cos']:.4f}\n"
        f"max  = {spe['max_max_cos']:.4f}\n"
        f"all 56 pairs > {COS_THRESH} → 77/77 patients"
    )
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

    # ── (c) v3.13: violating pair fraction + pair_term ─────────────────────
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(ep, v313["wsi"]["pair_term"], "o-", ms=3, lw=1.8,
            color="#0072B2", label="WSI pair_term")
    ax.plot(ep, v313["omic"]["pair_term"], "o-", ms=3, lw=1.8,
            color="#CC79A7", label="OMIC pair_term")
    ax.plot(ep, v313["wsi"]["violating_frac"], "s--", ms=3, lw=1.4,
            color="#0072B2", alpha=0.55, label="WSI violating frac")
    ax.plot(ep, v313["omic"]["violating_frac"], "s--", ms=3, lw=1.4,
            color="#CC79A7", alpha=0.55, label="OMIC violating frac")
    ax.set_yscale("log")
    ax.set_ylabel("value (log)")
    ax.set_xlabel("epoch")
    ax.set_title(
        "(c) v3.13: diversity penalty (solid) + violating pair fraction (dashed)\n"
        "lower = constraint closer to satisfied; does NOT mean all pairs satisfy",
        fontweight="bold", fontsize=11,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    # Annotate: ep29 WSI pair_term is ~0.024, NOT 0.509
    ax.annotate(
        f"ep29 WSI pair_term={v313['wsi']['pair_term'][ep29]:.3f}\n"
        f"(NOT min_dist=0.509; those are different columns)",
        xy=(ep[ep29], v313["wsi"]["pair_term"][ep29]),
        xytext=(ep[ep29] - 8, v313["wsi"]["pair_term"][ep29] * 5),
        fontsize=8, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
    )

    # ── (d) Comparison summary (honest) ─────────────────────────────────────
    ax = fig.add_subplot(2, 2, 4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    wsi_min_d   = v313["wsi"]["min_dist"][ep29]
    wsi_mean_d  = v313["wsi"]["mean_dist"][ep29]
    omic_min_d  = v313["omic"]["min_dist"][ep29]
    wsi_viol    = v313["wsi"]["violating_frac"][ep29]
    wsi_dup     = v313["wsi"]["dup_frac"][ep29]
    wsi_coll    = v313["wsi"]["collapse_frac"][ep29]
    omic_coll   = v313["omic"]["collapse_frac"][ep29]
    omic_viol   = v313["omic"]["violating_frac"][ep29]

    norm = V313_NORM
    cos_lb = 1.0 - (wsi_min_d ** 2) / (2 * norm ** 2)

    text = (
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  (d) Summary table — all values from real measurements       ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        f"║  v3.13 BLCA fold0 seed3 (epoch 29):                       ║\n"
        f"║    WSI  mean pair dist: {wsi_mean_d:7.3f}                        ║\n"
        f"║    WSI  min  pair dist: {wsi_min_d:7.3f}  ← measured               ║\n"
        f"║    WSI  pair_term:        {v313['wsi']['pair_term'][ep29]:7.4f}  (NOT 0.509; see (c))    ║\n"
        f"║    WSI  violating frac:   {wsi_viol:7.4f}                        ║\n"
        f"║    WSI  duplicate frac:   {wsi_dup:7.4f}  ← pairs with dist=0      ║\n"
        f"║    WSI  full collapse:    {wsi_coll:7.4f}  ← patients, all 8 identical║\n"
        f"║    OMIC violating frac:    {omic_viol:7.4f}                        ║\n"
        f"║    OMIC full collapse:    {omic_coll:7.4f}                        ║\n"
        f"║                                                        ║\n"
        f"║  v3.13 cosine:  NOT MEASURED                              ║\n"
        f"║  (estimated lower bound = {cos_lb:.4f} via 1−d²/(2r²)\n"
        f"║   with r≈{norm}, equal-norm ASSUMPTION UNVERIFIED)           ║\n"
        f"║                                                        ║\n"
        f"╠══════════════════════════════════════════════════════════════╣\n"
        f"║  SlotSPE BLCA fold0 seed3 (epoch 3):                       ║\n"
        f"║    mean max cosine:      {spe['mean_max_cos']:.4f}   ← REAL MEASURED       ║\n"
        f"║    min  max cosine:      {spe['min_max_cos']:.4f}                        ║\n"
        f"║    max  max cosine:      {spe['max_max_cos']:.4f}                        ║\n"
        f"║    patients all pairs > {COS_THRESH}:  {int(spe['fraction_all_above']*spe['n_patients'])}/{spe['n_patients']}   ← SINGLE checkpoint    ║\n"
        f"║    init max cosine:      {SPE_INIT_COS:.4f}   (FINAL_RECEIPT.md §3)     ║\n"
        f"╚══════════════════════════════════════════════════════════════╝"
    )
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", fontsize=8.2,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="#f9f9f9", ec="#888", lw=0.8))

    fig.text(
        0.5, 0.005,
        "⚠ CRITICAL LIMITATIONS:\n"
        "  (a–c) are v3.13 only.  (b) is SlotSPE only.  These use DIFFERENT protocols.\n"
        "  v3.13 cosine: NOT measured.  The lower-bound estimate is UNVERIFIED.\n"
        "  SlotSPE data: SINGLE checkpoint (fold0/seed3/epoch3).  Not a 5-fold mean.\n"
        "  Do NOT claim 'v3.13 has the same cosine as SlotSPE' or 'v3.13 has lower cosine'.\n"
        "  Do NOT claim 'all 56 v3.13 pairs exceed cos > 0.95' — this is unmeasured.",
        ha="center", fontsize=9, color="#c0392b",
        style="italic",
        bbox=dict(boxstyle="round", fc="#fff5f5", ec="#c0392b", lw=1.2),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_png, dpi=190, facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Honest summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_csv(out_csv: Path) -> None:
    import csv
    v313 = load_v313()
    spe  = load_slotspe()
    ep29 = len(v313["epoch"]) - 1
    wsi  = v313["wsi"]
    omic = v313["omic"]
    norm = V313_NORM
    cos_lb = 1.0 - (wsi["min_dist"][ep29] ** 2) / (2 * norm ** 2)

    rows = [
        # ── v3.13: measured Euclidean statistics ─────────────────────────────
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "mean_pair_distance",
            "value": f"{wsi['mean_dist'][ep29]:.4f}",
            "unit": "Euclidean distance",
            "source": "epoch CSV: train_v313_diversity_wsi_mean_pair_distance",
            "note": "batch-level mean over 56 directed pairs per epoch",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "min_pair_distance",
            "value": f"{wsi['min_dist'][ep29]:.4f}",
            "unit": "Euclidean distance",
            "source": "epoch CSV: train_v313_diversity_wsi_min_pair_distance",
            "note": "minimum over all patient-pair means at that epoch",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "pair_term",
            "value": f"{wsi['pair_term'][ep29]:.4f}",
            "unit": "penalty (lower=better)",
            "source": "epoch CSV: train_v313_diversity_wsi_pair_term",
            "note": "diversity loss component; NOT min_pair_distance (different column)",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "violating_pair_fraction",
            "value": f"{wsi['violating_frac'][ep29]:.4f}",
            "unit": "fraction of 56 pairs",
            "source": "epoch CSV: train_v313_diversity_wsi_violating_pair_fraction",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "duplicate_pair_fraction",
            "value": f"{wsi['dup_frac'][ep29]:.4f}",
            "unit": "fraction of 56 pairs",
            "source": "epoch CSV: train_v313_diversity_wsi_duplicate_pair_fraction",
            "note": "pairs with Euclidean dist = 0",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "full_collapse_patient_fraction",
            "value": f"{wsi['collapse_frac'][ep29]:.4f}",
            "unit": "fraction of patients",
            "source": "epoch CSV: train_v313_diversity_wsi_full_collapse_patient_fraction",
            "note": "patients where all 8 slots have identical vectors",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "OMIC",
            "metric": "mean_pair_distance",
            "value": f"{omic['mean_dist'][ep29]:.4f}",
            "unit": "Euclidean distance",
            "source": "epoch CSV: train_v313_diversity_omic_mean_pair_distance",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "OMIC",
            "metric": "min_pair_distance",
            "value": f"{omic['min_dist'][ep29]:.4f}",
            "unit": "Euclidean distance",
            "source": "epoch CSV: train_v313_diversity_omic_min_pair_distance",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "OMIC",
            "metric": "pair_term",
            "value": f"{omic['pair_term'][ep29]:.4f}",
            "unit": "penalty (lower=better)",
            "source": "epoch CSV: train_v313_diversity_omic_pair_term",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "OMIC",
            "metric": "violating_pair_fraction",
            "value": f"{omic['violating_frac'][ep29]:.4f}",
            "unit": "fraction of 56 pairs",
            "source": "epoch CSV: train_v313_diversity_omic_violating_pair_fraction",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "OMIC",
            "metric": "duplicate_pair_fraction",
            "value": f"{omic['dup_frac'][ep29]:.4f}",
            "unit": "fraction of 56 pairs",
            "source": "epoch CSV: train_v313_diversity_omic_duplicate_pair_fraction",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "OMIC",
            "metric": "full_collapse_patient_fraction",
            "value": f"{omic['collapse_frac'][ep29]:.4f}",
            "unit": "fraction of patients",
            "source": "epoch CSV: train_v313_diversity_omic_full_collapse_patient_fraction",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        # ── v3.13: UNVERIFIED cosine ─────────────────────────────────────────
        {
            "model": "v3.13 BLCA fold0 seed3 ep29",
            "modality": "WSI",
            "metric": "cosine_similarity",
            "value": f"NOT MEASURED  (estimated lower bound = {cos_lb:.4f})",
            "unit": "cosine",
            "source": "N/A — estimated via 1 − d²/(2r²) with r≈21.87",
            "note": (
                "UNVERIFIED: equal-norm assumption unconfirmed, "
                "min_pair_distance ≠ all pairs, "
                "r is batch-level average not per-slot norm"
            ),
            "measured_or_estimated": "UNVERIFIED ESTIMATE",
        },
        # ── SlotSPE: real measured cosines ───────────────────────────────────
        {
            "model": f"SlotSPE BLCA fold0 seed3 ep3 ({spe['n_patients']} patients)",
            "modality": "Wsi+Omic",
            "metric": "mean_max_offdiag_cosine",
            "value": f"{spe['mean_max_cos']:.6f}",
            "unit": "cosine",
            "source": "diagnostic.json p2.per_patient_max_offdiag_cos",
            "note": f"mean over {spe['n_patients']} patients, 56 off-diagonal pairs each",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": f"SlotSPE BLCA fold0 seed3 ep3 ({spe['n_patients']} patients)",
            "modality": "Wsi+Omic",
            "metric": "min_max_offdiag_cosine",
            "value": f"{spe['min_max_cos']:.6f}",
            "unit": "cosine",
            "source": "diagnostic.json p2.per_patient_max_offdiag_cos",
            "note": "minimum over all 77 patients",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": f"SlotSPE BLCA fold0 seed3 ep3 ({spe['n_patients']} patients)",
            "modality": "Wsi+Omic",
            "metric": "max_max_offdiag_cosine",
            "value": f"{spe['max_max_cos']:.6f}",
            "unit": "cosine",
            "source": "diagnostic.json p2.per_patient_max_offdiag_cos",
            "note": "",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": f"SlotSPE BLCA fold0 seed3 ep3 ({spe['n_patients']} patients)",
            "modality": "Wsi+Omic",
            "metric": f"fraction_patients_all_pairs_cos_gt_{COS_THRESH}",
            "value": f"{spe['fraction_all_above']:.4f}  ({int(spe['fraction_all_above']*spe['n_patients'])}/{spe['n_patients']})",
            "unit": "fraction",
            "source": "diagnostic.json p2.fraction_patients_with_offdiag_high_cos",
            "note": "SINGLE checkpoint only; not 5-fold mean",
            "measured_or_estimated": "MEASURED",
        },
        {
            "model": "SlotSPE at init",
            "modality": "unknown",
            "metric": "max_inter_slot_cosine",
            "value": f"{SPE_INIT_COS:.4f}",
            "unit": "cosine",
            "source": "FINAL_RECEIPT.md §3 — provenance not independently re-verified",
            "note": "check against raw checkpoint initialisation",
            "measured_or_estimated": "REPORTED",
        },
    ]

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "v313_vs_slotspe_p2_comparison_v2.png"
    make_figure(out_png)
    write_summary_csv(OUT_DIR / "v313_vs_slotspe_p2_comparison_v2.csv")
