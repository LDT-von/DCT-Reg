#!/usr/bin/env python3
"""Proof-of-Idea B: slot_variance constraint actually works.

v3.11 FROZEN_ARGUMENTS says dct_v311_variance_min=0.005, max=0.050.
We verify per-fold:
  (1) Final variance is inside [0.005, 0.050] for ≥4/5 folds
  (2) Variance is NOT collapsing to a constant (i.e., slots differentiate)
  (3) Variance correlates with val_c improvement over training
"""

from __future__ import annotations
import csv, json
from pathlib import Path
import statistics as stats

REPO = Path("/data1/DCT-Reg")
EXP = REPO / "results/dct_v311_blca_uni_fixed"
OUT_JSON = REPO / "results" / "proof_experiment_B.json"

# Frozen constraint
V_MIN = 0.005
V_MAX = 0.050


def find_curves(root: Path):
    return {int(p.stem.replace("epoch_curve_fold", "")): p
            for p in root.rglob("epoch_curve_fold*.csv")}


def main():
    curves = find_curves(EXP)
    print("="*72)
    print(f" PROOF B — slot_variance constraint (target [{V_MIN}, {V_MAX}])")
    print("="*72)
    print(f"Source: {EXP}   ({len(curves)} folds)\n")

    in_range = 0
    all_folds = []
    for fold in sorted(curves):
        rows = list(csv.DictReader(open(curves[fold])))
        v  = [float(r["train_v311_slot_variance"]) for r in rows if r.get("train_v311_slot_variance")]
        ps = [float(r["train_v311_per_slot_nll"]) for r in rows if r.get("train_v311_per_slot_nll")]
        vc = [float(r["val_cindex"]) for r in rows if r.get("val_cindex")]
        div = [float(r["train_v311_slot_diversity"]) for r in rows if r.get("train_v311_slot_diversity")]
        if not v:
            continue
        v_final = v[-1]
        v_range_total = (min(v), max(v))
        v_std = stats.stdev(v) if len(v) > 1 else 0
        in_band = V_MIN <= v_final <= V_MAX
        if in_band:
            in_range += 1
        # Pearson correlation between variance and val_c
        if len(v) >= 5 and len(vc) >= 5:
            n = min(len(v), len(vc))
            mv = sum(v[:n]) / n; mvc = sum(vc[:n]) / n
            num = sum((v[i] - mv) * (vc[i] - mvc) for i in range(n))
            den_v = (sum((x - mv)**2 for x in v[:n])) ** 0.5
            den_vc = (sum((x - mvc)**2 for x in vc[:n])) ** 0.5
            corr = num / (den_v * den_vc) if den_v and den_vc else 0
        else:
            corr = 0
        rec = {
            "fold": fold,
            "var_final": round(v_final, 5),
            "var_min": round(v_range_total[0], 5),
            "var_max": round(v_range_total[1], 5),
            "var_std": round(v_std, 5),
            "in_band": in_band,
            "var_valc_corr": round(corr, 3),
            "psnll_final": round(ps[-1], 4) if ps else None,
            "div_final": round(div[-1], 6) if div else None,
            "val_c_final": round(vc[-1], 4) if vc else None,
            "val_c_best": round(max(vc), 4) if vc else None,
        }
        all_folds.append(rec)
        print(f"  fold{fold}: var_final={rec['var_final']}  in_band={in_band}  "
              f"range=[{rec['var_min']:.4f}, {rec['var_max']:.4f}]  "
              f"std={rec['var_std']:.4f}  corr(var,val_c)={rec['var_valc_corr']:+.3f}")

    print()
    print(f"  folds in_band: {in_range}/{len(all_folds)}")
    mean_corr = stats.mean([f["var_valc_corr"] for f in all_folds])
    print(f"  mean corr(variance, val_cindex): {mean_corr:+.3f}")

    OUT_JSON.write_text(json.dumps({
        "constraint": {"v_min": V_MIN, "v_max": V_MAX},
        "n_folds": len(all_folds),
        "n_folds_in_band": in_range,
        "mean_corr_var_valc": round(mean_corr, 3),
        "folds": all_folds,
    }, indent=2))
    print(f"\n[written] {OUT_JSON}")


if __name__ == "__main__":
    main()
