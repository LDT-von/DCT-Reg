#!/usr/bin/env python3
"""Proof-of-Idea B (FIXED): slot_variance constraint, WSI vs Omics SEPARATED.

修复说明：
  原版 Proof B 检查的是 merged variance（WSI + Omics 合并），掩盖了
  单一模态槽坍缩的问题。本版分别检查 WSI 和 Omics 槽的 variance。

数据来源：results/dct_v311_blca_uni_fixed（修复后的 v3.11 版本）

判定：
  - WSI slots: variance ∈ [0.005, 0.050]
  - Omics slots: variance ∈ [0.005, 0.050]
  - 5/5 folds 同时通过两个模态的约束 → 修复后通过
"""

from __future__ import annotations
import csv, json, pickle, statistics as stats
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data1/DCT-Reg")

# Support both uni_fixed (旧版合并 variance) and uni_fixed_v2 (新版 per-modality)
DEFAULT_RESULTS_ROOT = REPO / "results/dct_v311_blca_uni_fixed_v2"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT,
                    help="Results root containing per_slot_export/per_slot_hazard.pkl")
args, _ = parser.parse_known_args()
EXP = args.results_root
PER_SLOT_PKL = EXP / "per_slot_export/per_slot_hazard.pkl"
OUT_JSON = REPO / "results" / "proof_experiment_B.json"

V_MIN = 0.005
V_MAX = 0.050


def compute_per_sample_variance(hazard: np.ndarray) -> np.ndarray:
    """For each sample, compute variance across slots averaged over time bins.

    hazard: [N, K, C]
    Returns: [N] per-sample variance
    """
    mean_pred = hazard.mean(axis=1, keepdims=True)   # [N, 1, C]
    var = ((hazard - mean_pred) ** 2).mean(axis=(1, 2))  # [N]
    return var


def main():
    print("="*72)
    print(f" PROOF B (FIXED) — slot_variance constraint, WSI vs Omics SEPARATED")
    print(f" Source: {EXP.name}  (per_slot_hazard.pkl)")
    print("="*72)

    if not PER_SLOT_PKL.exists():
        print(f"ERROR: per_slot_hazard.pkl not found at {PER_SLOT_PKL}")
        return 1

    with open(PER_SLOT_PKL, "rb") as f:
        per_fold = pickle.load(f)

    all_folds = []
    in_band_wsi, in_band_omic = 0, 0
    in_band_both = 0

    for d in per_fold:
        fold = d["fold"]
        hazard_wsi = d["hazard_wsi"]    # [N, K_w, C]
        hazard_omic = d["hazard_omic"]  # [N, K_o, C]

        var_wsi = compute_per_sample_variance(hazard_wsi)
        var_omic = compute_per_sample_variance(hazard_omic)

        # Per-fold summary
        wsi_mean = float(np.mean(var_wsi))
        wsi_in_pct = float(((var_wsi >= V_MIN) & (var_wsi <= V_MAX)).mean() * 100)
        wsi_collapsed = float((var_wsi < 0.001).mean() * 100)

        omic_mean = float(np.mean(var_omic))
        omic_in_pct = float(((var_omic >= V_MIN) & (var_omic <= V_MAX)).mean() * 100)
        omic_collapsed = float((var_omic < 0.001).mean() * 100)

        wsi_pass = V_MIN <= wsi_mean <= V_MAX
        omic_pass = V_MIN <= omic_mean <= V_MAX
        both_pass = wsi_pass and omic_pass

        if wsi_pass:
            in_band_wsi += 1
        if omic_pass:
            in_band_omic += 1
        if both_pass:
            in_band_both += 1

        rec = {
            "fold": fold,
            "wsi": {
                "mean_var": round(wsi_mean, 5),
                "in_band_pct": round(wsi_in_pct, 1),
                "collapsed_pct": round(wsi_collapsed, 1),
                "min_var": round(float(var_wsi.min()), 5),
                "max_var": round(float(var_wsi.max()), 5),
                "std_var": round(float(var_wsi.std()), 5),
                "n_samples": int(len(var_wsi)),
                "in_band_pass": wsi_pass,
            },
            "omic": {
                "mean_var": round(omic_mean, 5),
                "in_band_pct": round(omic_in_pct, 1),
                "collapsed_pct": round(omic_collapsed, 1),
                "min_var": round(float(var_omic.min()), 5),
                "max_var": round(float(var_omic.max()), 5),
                "std_var": round(float(var_omic.std()), 5),
                "n_samples": int(len(var_omic)),
                "in_band_pass": omic_pass,
            },
            "both_pass": both_pass,
        }
        all_folds.append(rec)

        print(f"\n  Fold {fold}:")
        print(f"    WSI:  mean_var={wsi_mean:.5f}  in_band={wsi_in_pct:5.1f}%  collapsed<0.001={wsi_collapsed:5.1f}%  {'✅' if wsi_pass else '❌'}")
        print(f"    Omic: mean_var={omic_mean:.5f}  in_band={omic_in_pct:5.1f}%  collapsed<0.001={omic_collapsed:5.1f}%  {'✅' if omic_pass else '❌'}")
        print(f"    BOTH: {'✅' if both_pass else '❌'}")

    # Cross-fold summary
    wsi_means = [f["wsi"]["mean_var"] for f in all_folds]
    omic_means = [f["omic"]["mean_var"] for f in all_folds]
    wsi_in_pcts = [f["wsi"]["in_band_pct"] for f in all_folds]
    omic_in_pcts = [f["omic"]["in_band_pct"] for f in all_folds]

    print()
    print("="*72)
    print(" CROSS-FOLD SUMMARY")
    print("="*72)
    print(f"  Modality  | mean_var | in_band% (avg) | collapsed% (avg) | in_band (folds)")
    print(f"  ----------+----------+---------------+------------------+-----------------")
    print(f"  WSI       | {np.mean(wsi_means):.5f} | {np.mean(wsi_in_pcts):>5.1f}%      | {np.mean([f['wsi']['collapsed_pct'] for f in all_folds]):>5.1f}%          | {in_band_wsi}/5")
    print(f"  Omic      | {np.mean(omic_means):.5f} | {np.mean(omic_in_pcts):>5.1f}%      | {np.mean([f['omic']['collapsed_pct'] for f in all_folds]):>5.1f}%          | {in_band_omic}/5")
    print(f"  Both WSI+O| -        | -             | -                | {in_band_both}/5")
    print()
    if in_band_both == len(all_folds):
        verdict = "✅ PASS"
    elif in_band_both >= 3:
        verdict = "⚠️ PARTIAL"
    else:
        verdict = "❌ FAIL"
    print(f"  VERDICT: {verdict} (folds where BOTH WSI and Omics pass: {in_band_both}/{len(all_folds)})")

    # Save JSON
    OUT_JSON.write_text(json.dumps({
        "constraint": {"v_min": V_MIN, "v_max": V_MAX},
        "n_folds": len(all_folds),
        "n_folds_wsi_in_band": in_band_wsi,
        "n_folds_omic_in_band": in_band_omic,
        "n_folds_both_in_band": in_band_both,
        "verdict": verdict,
        "folds": all_folds,
        "results_root": str(EXP),
        "note": "Separated WSI vs Omics slot variance (修复后版本)",
    }, indent=2))
    print(f"\n[written] {OUT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
