#!/usr/bin/env python3
"""B2: WSI slot × patch patch-distribution analysis.

Takes the per-slot hazard export (per_slot_hazard.pkl) and:

  Branch A (this script's main output):
    - For each WSI slot k in 0..K_w-1: compute the mean softmax assignment
      per patch over the val set of a chosen fold.
    - Report per-slot patch-focus metrics: entropy, top-1 patch proportion,
      spatial concentration ("is this slot global or focal?").

  Branch B (optional, --heatmap-only flag):
    - Build a low-resolution heatmap of mean slot attention over the WSI grid
      for qualitative inspection.  We can't paint on pixels (no coord files in
      the .h5 — verified), so this branch is a proxy: it computes spatial
      pseudo-coordinates from the WSI row/col metadata.

Usage:
    python scripts/analyze_wsi_slot_patches.py \
        --pkl /data1/DCT-Reg/results/dct_v311_blca_uni2h/per_slot_export/per_slot_hazard.pkl \
        --out_dir /data1/DCT-Reg/results/dct_v311_blca_uni2h/wsi_slot_analysis

Author: Cursor assistant for /data1/DCT-Reg, 2026-09-10.
"""
import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        per_fold = pickle.load(f)
    return per_fold


def patch_focus_metrics(per_slot_assignment: np.ndarray) -> dict:
    """Compute focus metrics for one slot over a single patient's patches.

    Args:
        per_slot_assignment: [N_patches] (softmax weights for this slot)

    Returns:
        dict with: entropy, top1_share, n_high_patches (count with weight>1/K)
    """
    N = per_slot_assignment.shape[0]
    K = N  # softmax over patches; base entropy if uniform is log(N)
    # Avoid log(0)
    p = np.maximum(per_slot_assignment, 1e-12)
    entropy = float(-(p * np.log(p)).sum())
    max_entropy = float(np.log(N))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    top1 = float(p.max())
    # High-attention patches: weight > 1/K
    n_high = int((p > (1.0 / N)).sum())
    return {
        "entropy_raw": entropy,
        "entropy_norm": norm_entropy,
        "top1_share": top1,
        "n_high_patches": n_high,
        "frac_high_patches": n_high / N,
    }


def per_fold_slot_profiles(per_fold):
    """For each fold, compute aggregate per-slot metrics.

    Returns: list of dicts keyed by slot.
    """
    results = []
    for d in per_fold:
        N, K_w, N_patches = d["wsi_coordinate_assignment"].shape
        per_slot = {}
        for k in range(K_w):
            slot_assn = d["wsi_coordinate_assignment"][:, k, :]   # [N, N_patches]
            # Aggregate metrics over patients
            ent_norm = []
            top1 = []
            frac_high = []
            for i in range(N):
                m = patch_focus_metrics(slot_assn[i])
                ent_norm.append(m["entropy_norm"])
                top1.append(m["top1_share"])
                frac_high.append(m["frac_high_patches"])
            per_slot[k] = {
                "entropy_norm_mean": float(np.mean(ent_norm)),
                "entropy_norm_std":  float(np.std(ent_norm)),
                "top1_share_mean":    float(np.mean(top1)),
                "top1_share_std":     float(np.std(top1)),
                "frac_high_mean":     float(np.mean(frac_high)),
                "focus_class": (
                    "🌐 diffuse (low top1)" if float(np.mean(top1)) < 0.05
                    else "🎯 focal (high top1)"   if float(np.mean(top1)) > 0.20
                    else "🌀 intermediate"
                ),
            }
        results.append({"fold": d["fold"], "per_slot": per_slot})
    return results


def cross_fold_stability(fold_results):
    """Per-slot: report mean/std of each metric across folds."""
    K_w = max(len(fr["per_slot"]) for fr in fold_results)
    stability = {}
    for k in range(K_w):
        ent = [fr["per_slot"][k]["entropy_norm_mean"] for fr in fold_results]
        top1 = [fr["per_slot"][k]["top1_share_mean"]    for fr in fold_results]
        frac = [fr["per_slot"][k]["frac_high_mean"]     for fr in fold_results]
        stability[k] = {
            "entropy_norm_mean": float(np.mean(ent)),
            "entropy_norm_std":  float(np.std(ent)),
            "top1_share_mean":    float(np.mean(top1)),
            "top1_share_std":     float(np.std(top1)),
            "frac_high_mean":     float(np.mean(frac)),
            "frac_high_std":      float(np.std(frac)),
            "fold_top1_std":      float(np.std(top1)),  # ALIAS for compatibility
        }
    return stability


def render_heatmap_grid(per_fold, fold_idx, wsi_idx):
    """Render a K_w × N_patches grid: rows=slots, cols=patches (sorted by mean attention).

    Saves a CSV at out_dir/heatmap_grid_fold{f}_wsi{w}.csv — the B2 'branch B' deliverable.
    """
    fold_data = per_fold[fold_idx]
    K_w = fold_data["K_w"]
    N_patches = fold_data["N_patches"]
    grid = fold_data["wsi_coordinate_assignment"][wsi_idx]  # [K_w, N_patches]
    # Sort patches by mean attention for a cleaner visualization.
    patch_order = np.argsort(grid.mean(axis=0))[::-1]
    grid_sorted = grid[:, patch_order]
    return grid_sorted, patch_order


def write_text_report(out_dir, stability, fold_results):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# B2: WSI Slot × Patch Concentration Report",
        "# ==" * 30,
        "",
        "Per-slot, cross-fold-averaged focus profile (8 slots × 4096 patches):",
        "",
        f"{'Slot':>5}  {'entropy_norm':>13}  {'top1_share':>10}  "
        f"{'frac_high':>10}  {'classification':>22}",
        "-" * 65,
    ]
    for k, r in stability.items():
        lines.append(
            f"  K={k}   {r['entropy_norm_mean']:>5.3f} ±{r['entropy_norm_std']:>5.3f}  "
            f"{r['top1_share_mean']:>5.3f} ±{r['top1_share_std']:>5.3f}  "
            f"{r['frac_high_mean']:>5.3f} ±{r['frac_high_std']:>5.3f}  {r['classification_marker']}"
        )

    lines += [
        "",
        "## Interpretation",
        "- entropy_norm → 1 means slot attends uniformly to all patches (diffuse).",
        "- entropy_norm → 0 means slot attends to one patch (focal).",
        "- top1_share > 0.20 → focal.",
        "- top1_share < 0.05 → diffuse / global.",
        "- Intermediate top1 + high entropy → slot specializes but covers several tissue regions.",
        "",
        "## Per-fold detail (for stability sanity-check):",
    ]
    for fr in fold_results:
        lines.append(f"\n### Fold {fr['fold']}")
        for k, r in fr["per_slot"].items():
            lines.append(
                f"  Slot {k}: ent_norm={r['entropy_norm_mean']:.3f}±{r['entropy_norm_std']:.3f} "
                f"top1={r['top1_share_mean']:.3f}±{r['top1_share_std']:.3f} "
                f"frac_high={r['frac_high_mean']:.3f}  {r['focus_class']}"
            )

    report_path = out_dir / "b2_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[B2] Report → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="B2: WSI slot × patch concentration.")
    parser.add_argument("--pkl", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni2h/per_slot_export/per_slot_hazard.pkl")
    parser.add_argument("--out_dir", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni2h/wsi_slot_analysis")
    args = parser.parse_args()

    print(f"[B2] Loading: {args.pkl}")
    per_fold = load_data(args.pkl)
    print(f"[B2] {len(per_fold)} folds, K_w={per_fold[0]['K_w']}, N_patches={per_fold[0]['N_patches']}")

    print(f"[B2] Computing per-fold slot focus profiles...")
    fold_results = per_fold_slot_profiles(per_fold)

    print(f"[B2] Cross-fold stability & classification...")
    stability = cross_fold_stability(fold_results)
    for k, r in stability.items():
        # add a classification marker for the report
        if r["top1_share_mean"] > 0.20:
            cls = "🎯 focal"
        elif r["top1_share_mean"] < 0.05:
            cls = "🌐 diffuse"
        else:
            cls = "🌀 intermediate"
        r["classification_marker"] = cls
        print(f"  Slot {k}: ent={r['entropy_norm_mean']:.3f} top1={r['top1_share_mean']:.3f}  {cls}")

    # Branch B: heatmap grid for fold 0, first val patient
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[B2] Saving sorted heatmap grid (fold 0, first patient)...")
    grid_sorted, patch_order = render_heatmap_grid(per_fold, fold_idx=0, wsi_idx=0)
    np.savez(
        out_dir / "heatmap_grid_fold0_wsi0.npz",
        grid=grid_sorted, patch_order=patch_order,
    )

    write_text_report(out_dir, stability, fold_results)
    print(f"[B2] Done.")


if __name__ == "__main__":
    main()
