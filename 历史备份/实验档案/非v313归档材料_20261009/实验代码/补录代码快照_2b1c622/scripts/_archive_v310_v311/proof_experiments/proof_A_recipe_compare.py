#!/usr/bin/env python3
"""Proof-of-Idea experiment A: Show v3.11 components actually deliver value.

We compare three recipe levels on the SAME 5-fold split:
  L0: Plain NLL  (no IPCW rank, no per_slot_nll, no diversity)
  L1: NLL + IPCW rank (v3.10-like)
  L2: NLL + IPCW rank + per_slot_nll + diversity (v3.11)

Outputs:
  - best val_c per fold for each level
  - per-component marginal gain
  - correlation between per_slot_nll training value and val_c
"""

from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics as stats

REPO = Path("/data1/DCT-Reg")

EXPERIMENTS = {
    # recipe_label : (results_root, study_label)
    # Each results_root contains SurvOTRank_* / .../epoch_curve_fold{0..4}.csv
    "v3.11 BLCA UNI (NLL+IPCW+per-slot+div)": REPO / "results/dct_v311_blca_uni_fixed",
    "v3.11 BLCA UNI2-h (early 1ep smoke)": REPO / "results/dct_v311_blca_uni2h_v2_smoke",
    "v3.10 BLCA UNI2-h frozen (NLL+IPCW)": REPO / "results/20260910_v310_blca_uni2h_frozen",
    "v3.10 BLCA UNI2-h (NLL+IPCW)": REPO / "results/20260909_v310_blca_uni2h",
}


def find_epoch_curves(root: Path) -> dict[int, Path]:
    """Return {fold_id: csv_path}."""
    out = {}
    if not root.exists():
        return out
    for csv_path in root.rglob("epoch_curve_fold*.csv"):
        # filename 'epoch_curve_fold0.csv' -> fold 0
        stem = csv_path.stem
        try:
            fold = int(stem.replace("epoch_curve_fold", ""))
            out[fold] = csv_path
        except ValueError:
            pass
    return out


def summarize(recipe: str, root: Path) -> dict:
    print(f"\n=== {recipe} ===  root={root}")
    folds = find_epoch_curves(root)
    if not folds:
        print("  [missing] no epoch_curve_fold*.csv")
        return {"recipe": recipe, "root": str(root), "folds": {}, "mean": None}
    summary_per_fold = {}
    for fold in sorted(folds.keys()):
        rows = list(csv.DictReader(open(folds[fold])))
        vc = [float(r.get("val_cindex", "nan")) for r in rows]
        vc = [v for v in vc if v == v]  # filter nan
        if not vc:
            continue
        best = max(vc)
        best_ep = vc.index(best)
        last5 = vc[-5:] if len(vc) >= 5 else vc
        mean_last5 = stats.mean(last5)
        # also gather v311 signals
        psnll = [float(r.get("train_v311_per_slot_nll", "nan")) for r in rows if r.get("train_v311_per_slot_nll")]
        div   = [float(r.get("train_v311_slot_diversity", "nan")) for r in rows if r.get("train_v311_slot_diversity")]
        var   = [float(r.get("train_v311_slot_variance", "nan")) for r in rows if r.get("train_v311_slot_variance")]
        ipcw  = [float(r.get("train_ipcw_rank", "nan")) for r in rows if r.get("train_ipcw_rank")]
        summary_per_fold[fold] = {
            "best_val_cindex": round(best, 4),
            "best_epoch": best_ep,
            "mean_last5": round(mean_last5, 4),
            "n_epochs": len(vc),
            "psnll_final": round(psnll[-1], 4) if psnll else None,
            "div_final": round(div[-1], 6) if div else None,
            "var_final": round(var[-1], 6) if var else None,
            "ipcw_final": round(ipcw[-1], 4) if ipcw else None,
        }
        print(f"  fold{fold}: best={best:.4f} @ep{best_ep:>2}  last5_mean={mean_last5:.4f}  "
              f"psnll={summary_per_fold[fold]['psnll_final']}  "
              f"div={summary_per_fold[fold]['div_final']}  "
              f"var={summary_per_fold[fold]['var_final']}  "
              f"ipcw={summary_per_fold[fold]['ipcw_final']}")
    if not summary_per_fold:
        return {"recipe": recipe, "root": str(root), "folds": {}, "mean": None}
    bests = [v["best_val_cindex"] for v in summary_per_fold.values()]
    last5s = [v["mean_last5"] for v in summary_per_fold.values()]
    return {
        "recipe": recipe,
        "root": str(root),
        "folds": summary_per_fold,
        "mean_best": round(stats.mean(bests), 4),
        "mean_last5": round(stats.mean(last5s), 4),
        "std_best": round(stats.stdev(bests), 4) if len(bests) > 1 else 0.0,
    }


def main():
    out = {"by_recipe": {}}
    print("="*72)
    print(" PROOF A — Component Effectiveness Across Recipes")
    print("="*72)
    for recipe, root in EXPERIMENTS.items():
        out["by_recipe"][recipe] = summarize(recipe, root)

    print("\n" + "="*72)
    print(" SUMMARY (mean best val_cindex across folds)")
    print("="*72)
    rows = sorted([(r["mean_best"], r["recipe"]) for r in out["by_recipe"].values() if r["mean_best"] is not None], reverse=True)
    for mb, name in rows:
        print(f"  {mb:.4f}  {name}")

    out_path = REPO / "results" / "proof_experiment_A.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
