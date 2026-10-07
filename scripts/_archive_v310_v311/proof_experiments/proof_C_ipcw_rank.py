#!/usr/bin/env python3
"""Proof-of-Idea C: IPCW rank loss actually contributes.

Strategy:
  (1) For v3.11 (with IPCW rank), show train_ipcw_rank decreases over epochs
      (loss is being minimized).
  (2) For v3.11 vs v3.10, show val_cindex gap correlates with how much
      IPCW rank dropped. If IPCW loss matters, lower IPCW -> better val_c.
  (3) Show that folds with stronger IPCW improvement also have higher
      final val_cindex (cross-fold correlation).
"""

from __future__ import annotations
import csv, json
from pathlib import Path
import statistics as stats

REPO = Path("/data1/DCT-Reg")

V311 = REPO / "results/dct_v311_blca_uni_fixed"
V310_FROZEN = REPO / "results/20260910_v310_blca_uni2h_frozen"
V310 = REPO / "results/20260909_v310_blca_uni2h"
OUT_JSON = REPO / "results" / "proof_experiment_C.json"


def find_curves(root: Path):
    return {int(p.stem.replace("epoch_curve_fold", "")): p
            for p in root.rglob("epoch_curve_fold*.csv")}


def per_fold_summary(root: Path):
    out = []
    for fold, p in find_curves(root).items():
        rows = list(csv.DictReader(open(p)))
        ipcw = [float(r["train_ipcw_rank"]) for r in rows if r.get("train_ipcw_rank")]
        vc   = [float(r["val_cindex"]) for r in rows if r.get("val_cindex")]
        if not ipcw:
            continue
        # first vs last 5 epochs
        first5 = ipcw[:5] if len(ipcw) >= 5 else ipcw
        last5  = ipcw[-5:] if len(ipcw) >= 5 else ipcw
        drop = stats.mean(first5) - stats.mean(last5)
        rec = {
            "fold": fold,
            "ipcw_first5": round(stats.mean(first5), 4),
            "ipcw_last5": round(stats.mean(last5), 4),
            "ipcw_drop": round(drop, 4),
            "val_c_best": round(max(vc), 4) if vc else None,
            "val_c_last": round(vc[-1], 4) if vc else None,
            "n_epochs": len(ipcw),
        }
        out.append(rec)
    return out


def main():
    print("="*72)
    print(" PROOF C — IPCW rank loss actually contributes")
    print("="*72)

    print(f"\n--- v3.11 BLCA UNI (with IPCW rank) ---")
    v311 = per_fold_summary(V311)
    for r in v311:
        print(f"  fold{r['fold']}: ipcw {r['ipcw_first5']:.4f} -> {r['ipcw_last5']:.4f}  "
              f"drop={r['ipcw_drop']:+.4f}  val_c_best={r['val_c_best']}  val_c_last={r['val_c_last']}")
    drops = [r["ipcw_drop"] for r in v311]
    bests = [r["val_c_best"] for r in v311]
    if len(drops) > 1 and drops:
        # corr between IPCW drop and val_c_best
        md = stats.mean(drops); mb = stats.mean(bests)
        num = sum((drops[i] - md) * (bests[i] - mb) for i in range(len(drops)))
        den = (sum((d - md)**2 for d in drops) ** 0.5) * (sum((b - mb)**2 for b in bests) ** 0.5)
        corr = num / den if den else 0
        print(f"  >>> mean IPCW drop = {stats.mean(drops):+.4f}")
        print(f"  >>> corr(IPCW_drop, val_c_best) = {corr:+.3f}")

    print(f"\n--- v3.10 BLCA UNI2-h frozen (no per-slot/diversity, but has IPCW) ---")
    v310fz = per_fold_summary(V310_FROZEN)
    for r in v310fz:
        print(f"  fold{r['fold']}: ipcw {r['ipcw_first5']:.4f} -> {r['ipcw_last5']:.4f}  "
              f"drop={r['ipcw_drop']:+.4f}  val_c_best={r['val_c_best']}")

    print(f"\n--- v3.10 BLCA UNI2-h (with IPCW) ---")
    v310p = per_fold_summary(V310)
    for r in v310p:
        print(f"  fold{r['fold']}: ipcw {r['ipcw_first5']:.4f} -> {r['ipcw_last5']:.4f}  "
              f"drop={r['ipcw_drop']:+.4f}  val_c_best={r['val_c_best']}")

    # Verdict: v3.11's IPCW should drop more (it has more learning signal)
    v311_mean_drop = stats.mean(drops) if drops else 0
    print(f"\n  >>> v3.11 mean IPCW drop: {v311_mean_drop:+.4f}")
    print(f"  >>> v3.10 frozen mean val_c_best: {stats.mean([r['val_c_best'] for r in v310fz]):.4f}")
    print(f"  >>> v3.11 mean val_c_best:       {stats.mean(bests):.4f}")

    OUT_JSON.write_text(json.dumps({
        "v311": v311,
        "v310_frozen": v310fz,
        "v310": v310p,
        "v311_mean_ipcw_drop": round(v311_mean_drop, 4),
        "v311_mean_val_c_best": round(stats.mean(bests), 4) if bests else None,
    }, indent=2))
    print(f"\n[written] {OUT_JSON}")


if __name__ == "__main__":
    main()
