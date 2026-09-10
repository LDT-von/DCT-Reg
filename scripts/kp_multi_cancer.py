#!/usr/bin/env python3
"""KM curves for all cancers with available predictions.csv."""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

REPO_ROOT = Path("/data1/DCT-Reg")
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "paper_outputs" / "km_curves_multi_cancer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["blca", "hnsc", "skcm", "lusc", "kirc", "brca", "coadread", "luad", "stad", "ucec"]


def load_cancer(cancer: str) -> Optional[pd.DataFrame]:
    """Load predictions from split_*_results_final.pkl (case → {risk, censor, time})."""
    base = RESULTS / "dct_v3.10" / "robust" / "final_50ep_old" / cancer / cancer / \
           "SurvOTRank_dct_v310_directional_regularized_transport"
    if not base.exists():
        return None
    matches = list(base.glob(f"*dct_reg_{cancer}*"))
    if not matches:
        return None
    exp = matches[0]
    parts = []
    for fold in range(5):
        p = exp / f"split_{fold}_results_final.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            data = pickle.load(f)
        rows = []
        for case_id, info in data.items():
            rows.append({
                "case_id": case_id,
                "risk": float(info["risk"].mean() if hasattr(info["risk"], "__len__") else info["risk"]),
                "censor": float(info["censor"]),
                "time": float(info["time"]),
                "fold": fold,
            })
        parts.append(pd.DataFrame(rows))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def km_one_cancer(df: pd.DataFrame, cancer: str, out_dir: Path) -> Dict:
    """Compute KM curves + log-rank for one cancer."""
    # Stratify into 3 groups
    df = df.copy()
    df["risk_group"] = pd.qcut(df["risk"], q=3, labels=["Low Risk", "Medium Risk", "High Risk"])
    df["event"] = 1 - df["censor"]

    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3-group KM
    ax = axes[0]
    kmf = KaplanMeierFitter()
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    for grp, color in zip(["Low Risk", "Medium Risk", "High Risk"], colors):
        mask = df["risk_group"] == grp
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"{grp} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, lw=2.5, alpha=0.85)
    lr = multivariate_logrank_test(df["time"], df["risk_group"], df["event"])
    p = float(lr.p_value)
    ax.text(0.97, 0.97, f"Log-rank p = {p:.4g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{cancer.upper()} — 3-Group Stratification", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    res["log_rank_p_3group"] = p
    res["n"] = int(len(df))
    res["events"] = int(df["event"].sum())

    # 2-group KM
    df["risk_group2"] = pd.qcut(df["risk"], q=2, labels=["Low Risk", "High Risk"])
    ax = axes[1]
    for grp, color in zip(["Low Risk", "High Risk"], ["#22c55e", "#ef4444"]):
        mask = df["risk_group2"] == grp
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"{grp} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, lw=2.5, alpha=0.85)
    lr2 = multivariate_logrank_test(df["time"], df["risk_group2"], df["event"])
    p2 = float(lr2.p_value)
    ax.text(0.97, 0.97, f"Log-rank p = {p2:.4g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{cancer.upper()} — 2-Group Stratification", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    res["log_rank_p_2group"] = p2

    plt.tight_layout()
    out = out_dir / f"km_curves_{cancer}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    res["plot"] = str(out)
    return res


def multi_cancer_grid(per_cancer: Dict[str, Dict], out_path: Path):
    cancers = list(per_cancer.keys())
    n = len(cancers)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for idx, cancer in enumerate(cancers):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        df = load_cancer(cancer)
        if df is None:
            ax.axis("off")
            ax.set_title(f"{cancer.upper()} (no data)")
            continue
        df = df.copy()
        df["risk_group"] = pd.qcut(df["risk"], q=3, labels=["Low", "Med", "High"])
        df["event"] = 1 - df["censor"]
        kmf = KaplanMeierFitter()
        for grp, color in zip(["Low", "Med", "High"], ["#22c55e", "#f59e0b", "#ef4444"]):
            mask = df["risk_group"] == grp
            kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=grp)
            kmf.plot_survival_function(ax=ax, color=color, lw=2)
        lr = multivariate_logrank_test(df["time"], df["risk_group"], df["event"])
        p = float(lr.p_value)
        ax.text(0.97, 0.97, f"p = {p:.3g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_title(f"{cancer.upper()} (n={len(df)})", fontweight="bold")
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    # Hide unused
    used = len(cancers)
    total = rows * cols
    for j in range(used, total):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle("Kaplan-Meier Risk Stratification across Cancers (3-group)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved grid: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", default="all")
    args = parser.parse_args()

    cancers = [args.cancer] if args.cancer != "all" else CANCERS
    per_cancer = {}

    for cancer in cancers:
        df = load_cancer(cancer)
        if df is None:
            print(f"  ⚠ {cancer.upper()}: no predictions")
            continue
        print(f"\n=== {cancer.upper()} ===")
        print(f"  Loaded {len(df)} samples (events: {(1-df['censor']).sum()})")
        res = km_one_cancer(df, cancer, OUT_DIR)
        per_cancer[cancer] = res
        print(f"  3-group log-rank p = {res['log_rank_p_3group']:.4g}")
        print(f"  2-group log-rank p = {res['log_rank_p_2group']:.4g}")
        print(f"  ✓ plot saved")

    if not per_cancer:
        print("No data.")
        return

    # Summary table
    summary_df = pd.DataFrame([
        {
            "Cancer": c.upper(),
            "N": res["n"],
            "Events": res["events"],
            "log_rank_p_3group": res["log_rank_p_3group"],
            "log_rank_p_2group": res["log_rank_p_2group"],
            "Significant (3g)": "✓" if res["log_rank_p_3group"] < 0.05 else "✗",
            "Significant (2g)": "✓" if res["log_rank_p_2group"] < 0.05 else "✗",
        }
        for c, res in per_cancer.items()
    ])
    summary_path = OUT_DIR / "km_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved: {summary_path}")

    # Grid
    grid_path = REPO_ROOT / "paper_outputs" / "km_curves_all_cancers_grid.png"
    multi_cancer_grid(per_cancer, grid_path)
    print(f"\n✓ All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()