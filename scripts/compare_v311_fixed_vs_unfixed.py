#!/usr/bin/env python3
"""v3.11 Fixed vs Unfixed Per-Slot Hazard Comparison.

Reads:
  - Fixed:  results/dct_v311_blca_uni_fixed/per_slot_export/per_slot_hazard.pkl
  - Unfixed: results/dct_v311_blca_uni/per_slot_export/per_slot_hazard.pkl

Runs A1/A2/A3 analysis for both and compares head-to-head.
No torch needed - only numpy, pandas, scipy, sksurv.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sksurv.metrics import concordance_index_censored

REPO = Path("/data1/DCT-Reg")
PKL_FIXED = REPO / "results/dct_v311_blca_uni_fixed/per_slot_export/per_slot_hazard.pkl"
PKL_UNFIXED = REPO / "results/dct_v311_blca_uni/per_slot_export/per_slot_hazard.pkl"
OUT_DIR = REPO / "results/v311_fixed_proof"


# ────────────────────────────────────────────────────────────────────
# Core analysis (same as analyze_v311_slot_interp.py)
# ────────────────────────────────────────────────────────────────────

def compute_cindex_spearman(
    hazard_per_slot: np.ndarray,
    times: np.ndarray,
    censors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    N, K, C = hazard_per_slot.shape
    cindex_per_slot = np.zeros(K)
    spearman_per_slot = np.zeros(K)
    for k in range(K):
        slot_risk = hazard_per_slot[:, k, :].sum(axis=1)
        event_observed = censors == 0
        try:
            cindex, _, _, _, _ = concordance_index_censored(
                event_observed, times, slot_risk
            )
            cindex_per_slot[k] = cindex
        except Exception:
            cindex_per_slot[k] = np.nan
        hazard_over_time = hazard_per_slot[:, k, :].mean(axis=0)
        time_bins = np.arange(C)
        spearman, _ = stats.spearmanr(time_bins, hazard_over_time)
        spearman_per_slot[k] = spearman if not np.isnan(spearman) else 0.0
    return cindex_per_slot, spearman_per_slot


def assess_monotonicity(hazard_per_slot: np.ndarray) -> Dict[int, str]:
    N, K, C = hazard_per_slot.shape
    classification = {}
    for k in range(K):
        hazard_over_time = hazard_per_slot[:, k, :].mean(axis=0)
        if hazard_over_time[-1] > hazard_over_time[0] * 1.1:
            classification[k] = "good"
        elif abs(hazard_over_time[-1] - hazard_over_time[0]) < 0.05:
            classification[k] = "bad"
        else:
            classification[k] = "mixed"
    return classification


def compute_slot_variance(hazard_per_slot: np.ndarray) -> np.ndarray:
    """Slot assignment variance: mean std of slot attn across time bins."""
    # Using std of hazard values across time bins as a proxy
    return hazard_per_slot.std(axis=2).mean(axis=0)


def analyze_pkl(pkl_path: Path) -> Dict:
    with open(pkl_path, "rb") as f:
        per_fold = pickle.load(f)

    results = {
        "source": str(pkl_path),
        "folds": [],
        "summary": {},
    }

    all_wsi_cindex, all_wsi_spearman = [], []
    all_omic_cindex, all_omic_spearman = [], []
    all_wsi_good, all_omic_good = 0, 0
    total_wsi_slots, total_omic_slots = 0, 0

    for d in per_fold:
        fold = d["fold"]
        wsi_cidx, wsi_spear = compute_cindex_spearman(
            d["hazard_wsi"], d["times"], d["censors"]
        )
        omic_cidx, omic_spear = compute_cindex_spearman(
            d["hazard_omic"], d["times"], d["censors"]
        )
        wsi_class = assess_monotonicity(d["hazard_wsi"])
        omic_class = assess_monotonicity(d["hazard_omic"])

        fold_data = {
            "fold": fold,
            "K_w": d["K_w"],
            "K_o": d["K_o"],
            "N": len(d["case_ids"]),
            "wsi_cindex": wsi_cidx.tolist(),
            "wsi_spearman": wsi_spear.tolist(),
            "omic_cindex": omic_cidx.tolist(),
            "omic_spearman": omic_spear.tolist(),
            "wsi_good_rate": sum(1 for v in wsi_class.values() if v == "good") / d["K_w"],
            "omic_good_rate": sum(1 for v in omic_class.values() if v == "good") / d["K_o"],
        }
        results["folds"].append(fold_data)

        all_wsi_cindex.extend(wsi_cidx.tolist())
        all_wsi_spearman.extend(wsi_spear.tolist())
        all_omic_cindex.extend(omic_cidx.tolist())
        all_omic_spearman.extend(omic_spear.tolist())
        all_wsi_good += sum(1 for v in wsi_class.values() if v == "good")
        all_omic_good += sum(1 for v in omic_class.values() if v == "good")
        total_wsi_slots += d["K_w"]
        total_omic_slots += d["K_o"]

    results["summary"] = {
        "n_folds": len(per_fold),
        "wsi_mean_cindex": float(np.mean(all_wsi_cindex)),
        "wsi_std_cindex": float(np.std(all_wsi_cindex)),
        "wsi_mean_spearman": float(np.mean(all_wsi_spearman)),
        "wsi_std_spearman": float(np.std(all_wsi_spearman)),
        "wsi_good_pct": 100 * all_wsi_good / total_wsi_slots,
        "omic_mean_cindex": float(np.mean(all_omic_cindex)),
        "omic_std_cindex": float(np.std(all_omic_cindex)),
        "omic_mean_spearman": float(np.mean(all_omic_spearman)),
        "omic_std_spearman": float(np.std(all_omic_spearman)),
        "omic_good_pct": 100 * all_omic_good / total_omic_slots,
    }
    return results


def make_comparison_report(fixed: Dict, unfixed: Dict) -> str:
    fs = fixed["summary"]
    us = unfixed["summary"]

    def delta(fixed_val, unfixed_val):
        d = fixed_val - unfixed_val
        if d > 0:
            return f"+{d:.4f} ✅"
        elif d < 0:
            return f"{d:.4f} ⚠️"
        else:
            return "0.0000 ≈"

    md = ["# v3.11 Fixed vs Unfixed — Per-Slot Hazard 对比报告\n\n"]
    md.append(f"- **Fixed** (diversity 约束正常): `{PKL_FIXED}`\n")
    md.append(f"- **Unfixed** (训练日志显示 slot collapse): `{PKL_UNFIXED}`\n\n")

    md.append("---\n\n## A1: Slot Hazard vs Time-Bin 单调性 (Hazard 应随时间增加)\n\n")
    md.append("**方法**：每个 slot 的 mean hazard over time 与 time-bin 索引计算 Spearman ρ；")
    md.append("hazard[last_bin] > hazard[first_bin] × 1.1 判定为「good」。\n\n")
    md.append("| 指标 | Fixed | Unfixed | 差值 | 解读 |\n")
    md.append("|------|-------|---------|------|------|\n")
    md.append(f"| WSI Good Rate | {fs['wsi_good_pct']:.1f}% | {us['wsi_good_pct']:.1f}% | {delta(fs['wsi_good_pct'], us['wsi_good_pct'])} | |\n")
    md.append(f"| WSI Mean Spearman ρ (time) | {fs['wsi_mean_spearman']:.4f} | {us['wsi_mean_spearman']:.4f} | {delta(fs['wsi_mean_spearman'], us['wsi_mean_spearman'])} | |\n")
    md.append(f"| WSI Mean C-index | {fs['wsi_mean_cindex']:.4f} | {us['wsi_mean_cindex']:.4f} | {delta(fs['wsi_mean_cindex'], us['wsi_mean_cindex'])} | |\n")
    md.append(f"| Omics Good Rate | {fs['omic_good_pct']:.1f}% | {us['omic_good_pct']:.1f}% | {delta(fs['omic_good_pct'], us['omic_good_pct'])} | |\n")
    md.append(f"| Omics Mean Spearman ρ (time) | {fs['omic_mean_spearman']:.4f} | {us['omic_mean_spearman']:.4f} | {delta(fs['omic_mean_spearman'], us['omic_mean_spearman'])} | |\n")
    md.append(f"| Omics Mean C-index | {fs['omic_mean_cindex']:.4f} | {us['omic_mean_cindex']:.4f} | {delta(fs['omic_mean_cindex'], us['omic_mean_cindex'])} | |\n")

    md.append("\n**结论**：")
    if fs["wsi_good_pct"] == 100 and us["wsi_good_pct"] == 100:
        md.append("两个版本的 Hazard-Time 单调性都是 100%（Slot 都能预测随时间增加的风险）。")
    elif fs["wsi_good_pct"] > us["wsi_good_pct"]:
        md.append(f"Fixed 的 WSI Good Rate ({fs['wsi_good_pct']:.0f}%) 高于 Unfixed ({us['wsi_good_pct']:.0f}%)。")
    else:
        md.append("两个版本 WSI Good Rate 相近。")

    md.append("\n\n---\n\n## Per-Fold 详细对比\n\n")
    md.append("| Fold | 版本 | WSI Good% | WSI ρ | WSI C-idx | Omics Good% | Omics ρ | Omics C-idx |\n")
    md.append("|------|------|-----------|-------|-----------|-------------|---------|-------------|\n")
    for fd, ud in zip(fixed["folds"], unfixed["folds"]):
        md.append(f"| {fd['fold']} | **Fixed** | {fd['wsi_good_rate']*100:.0f}% | "
                  f"{np.mean(fd['wsi_spearman']):.4f} | {np.mean(fd['wsi_cindex']):.4f} | "
                  f"{fd['omic_good_rate']*100:.0f}% | {np.mean(fd['omic_spearman']):.4f} | "
                  f"{np.mean(fd['omic_cindex']):.4f} |\n")
        md.append(f"| {ud['fold']} | Unfixed | {ud['wsi_good_rate']*100:.0f}% | "
                  f"{np.mean(ud['wsi_spearman']):.4f} | {np.mean(ud['wsi_cindex']):.4f} | "
                  f"{ud['omic_good_rate']*100:.0f}% | {np.mean(ud['omic_spearman']):.4f} | "
                  f"{np.mean(ud['omic_cindex']):.4f} |\n")

    md.append("\n---\n\n## 最终结论\n\n")
    md.append("| 对比维度 | Fixed | Unfixed | 差值 | 解读 |\n")
    md.append("|----------|-------|---------|------|------|\n")
    md.append(f"| WSI 单调率 | {fs['wsi_good_pct']:.0f}% | {us['wsi_good_pct']:.0f}% | {fs['wsi_good_pct']-us['wsi_good_pct']:+.0f}%pt | |\n")
    md.append(f"| WSI Spearman ρ | {fs['wsi_mean_spearman']:.4f} | {us['wsi_mean_spearman']:.4f} | {fs['wsi_mean_spearman']-us['wsi_mean_spearman']:+.4f} | |\n")
    md.append(f"| WSI C-index | {fs['wsi_mean_cindex']:.4f} | {us['wsi_mean_cindex']:.4f} | {fs['wsi_mean_cindex']-us['wsi_mean_cindex']:+.4f} | |\n")
    md.append(f"| Omics 单调率 | {fs['omic_good_pct']:.0f}% | {us['omic_good_pct']:.0f}% | {fs['omic_good_pct']-us['omic_good_pct']:+.0f}%pt | |\n")
    md.append(f"| Omics Spearman ρ | {fs['omic_mean_spearman']:.4f} | {us['omic_mean_spearman']:.4f} | {fs['omic_mean_spearman']-us['omic_mean_spearman']:+.4f} | |\n")
    md.append(f"| Omics C-index | {fs['omic_mean_cindex']:.4f} | {us['omic_mean_cindex']:.4f} | {fs['omic_mean_cindex']-us['omic_mean_cindex']:+.4f} | |\n")

    return "".join(md)


def main():
    print("Loading Fixed pkl...")
    fixed = analyze_pkl(PKL_FIXED)
    print(f"  Fixed: {len(fixed['folds'])} folds loaded")

    print("Loading Unfixed pkl...")
    unfixed = analyze_pkl(PKL_UNFIXED)
    print(f"  Unfixed: {len(unfixed['folds'])} folds loaded")

    report = make_comparison_report(fixed, unfixed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "v311_fixed_vs_unfixed_comparison.md"
    out_path.write_text(report)
    print(f"\n✅ Report saved to {out_path}")
    print("\n" + "="*70)
    print(report)


if __name__ == "__main__":
    main()
