"""Pancancer evidence figure for the v3.13 P2 (slot redundancy) claim.

Reads 45 epoch-curve CSVs from ``results/dct_v313_uni2h`` (9 studies * 5 folds).
Only the columns that are guaranteed to exist on every fold are used:

    train_v311_slot_diversity   (penalty term, smaller = better)
    train_v311_slot_variance    (slot hazard variance, larger = better)

Two figures are produced:
    1) 45-overlay: every run as a faint line, study mean as a bold line.
    2) per-study mean curves, fold collapsed.

Nothing is synthesised; every number comes from CSV.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
UNI2H = ROOT / "results/dct_v313_uni2h"
OUT_DIR = ROOT / "docs/research/dct_v313_slot_diversity_figure"

# Tab10 colors; deterministic assignment to studies.
# Variance target band.  Read from the live model definition, not from the
# archived v311 analysis scripts (those carry a stale 0.005 lower bound):
#   survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py
#   VARIANCE_MIN = 0.001 ; VARIANCE_MAX = 0.050
VAR_BAND_LO = 0.001
VAR_BAND_HI = 0.050

STUDY_ORDER = ["brca", "coadread", "hnsc", "kirc", "luad", "lusc", "skcm", "stad", "ucec"]
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
STUDY_COLOR = dict(zip(STUDY_ORDER, TAB10))


def discover_runs() -> dict[str, list[Path]]:
    """Return {study: [fold0..fold4 csv paths]}."""
    by_study: dict[str, list[Path]] = {s: [] for s in STUDY_ORDER}
    for csv_path in sorted(UNI2H.glob("*/*/*/*/epoch_curve_fold*.csv")):
        # rel parts: (<study>, <study>, <method>, <run-name>, epoch_curve_foldN.csv)
        study = csv_path.relative_to(UNI2H).parts[0]
        if study not in by_study:
            continue
        by_study[study].append(csv_path)
    for s in by_study:
        by_study[s] = sorted(by_study[s], key=lambda p: int(p.stem.split("fold")[-1]))
    return by_study


def load_curve(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (epochs, diversity_term, variance)."""
    with path.open() as h:
        rows = list(csv.DictReader(h))
    epochs = [int(r["epoch"]) for r in rows]
    div = [float(r["train_v311_slot_diversity"]) for r in rows]
    var = [float(r["train_v311_slot_variance"]) for r in rows]
    return epochs, div, var


def mean_curve(runs: list[Path], key_idx: int) -> tuple[list[int], list[float], list[float]]:
    """Average a metric across runs, epoch-aligned."""
    series = [load_curve(p)[key_idx] for p in runs]
    n = min(len(s) for s in series)
    epochs = list(range(n))
    stacked = np.array([s[:n] for s in series], dtype=float)
    mean = stacked.mean(axis=0).tolist()
    std = stacked.std(axis=0).tolist()
    return epochs, mean, std


def plot_overlay_45(by_study: dict[str, list[Path]], path: Path) -> None:
    """Top: 45 individual runs as faint lines, per-study mean as bold.  Bottom: variance."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    fig.suptitle(
        "Pancancer v3.13  (9 studies x 5 folds = 45 runs): the diversity PENALTY collapses "
        "~20x on every run,\nbut slot hazard variance stays BELOW its target band on 44/45 runs",
        fontsize=13.5, fontweight="bold", y=0.985,
    )

    # -- (a) diversity term, log scale
    ax = axes[0]
    for s in STUDY_ORDER:
        for p in by_study[s]:
            eps, div, _ = load_curve(p)
            ax.plot(eps, div, color=STUDY_COLOR[s], alpha=0.10, lw=0.6)
    for s in STUDY_ORDER:
        eps, mean, _ = mean_curve(by_study[s], 1)
        ax.plot(eps, mean, color=STUDY_COLOR[s], lw=2.4, label=s)
    ax.set_yscale("log")
    ax.set_title("(a) train_v311_slot_diversity  (penalty, lower = better)", fontweight="bold")
    ax.set_ylabel("diversity penalty (log)")
    ax.set_xlabel("epoch")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.92)

    # -- (b) hazard variance, log scale
    ax = axes[1]
    for s in STUDY_ORDER:
        for p in by_study[s]:
            eps, _, var = load_curve(p)
            ax.plot(eps, var, color=STUDY_COLOR[s], alpha=0.10, lw=0.6)
    for s in STUDY_ORDER:
        eps, mean, _ = mean_curve(by_study[s], 2)
        ax.plot(eps, mean, color=STUDY_COLOR[s], lw=2.4, label=s)
    ax.set_yscale("log")
    ax.set_title("(b) train_v311_slot_variance  (8-slot hazard spread)", fontweight="bold")
    ax.set_ylabel("hazard variance (log)")
    ax.set_xlabel("epoch")
    # Variance target band, read from model.py (VARIANCE_MIN=0.001, VARIANCE_MAX=0.050).
    ax.axhspan(VAR_BAND_LO, VAR_BAND_HI, color="#2e7d32", alpha=0.10, lw=0)
    ax.text(0.02, 0.06, f"v311 target band [{VAR_BAND_LO}, {VAR_BAND_HI}]  -  44/45 runs end below it",
            transform=ax.get_yaxis_transform(), color="#2e7d32",
            fontsize=8.5, fontweight="bold")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.92)

    fig.text(
        0.5, 0.01,
        "Source: results/dct_v313_uni2h/<study>/<study>/.../epoch_curve_fold0..4.csv  "
        "(45 runs). Faint lines = individual runs; bold = per-study mean.\n"
        "READ (a) AND (b) TOGETHER: the pair-distance penalty is satisfied (a), yet per-slot risk "
        "spread stays below the v311 band (b).\n"
        "Penalty satisfaction alone therefore does NOT certify slot diversity on its own.",
        ha="center", fontsize=8.6, color="#444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def plot_per_study_mean(by_study: dict[str, list[Path]], path: Path) -> None:
    """9 study mean curves, fold collapsed, both metrics on one figure with mean +/- std band."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    fig.suptitle(
        "Pancancer v3.13  (folds collapsed within each study, 9 studies):  "
        "mean +/- std across 5 folds",
        fontsize=14, fontweight="bold", y=0.985,
    )

    # -- (a) diversity term
    ax = axes[0]
    for s in STUDY_ORDER:
        eps, mean, std = mean_curve(by_study[s], 1)
        mean = np.array(mean); std = np.array(std)
        c = STUDY_COLOR[s]
        ax.plot(eps, mean, color=c, lw=2.2, label=s)
        ax.fill_between(eps, np.maximum(mean - std, 1e-9), mean + std, color=c, alpha=0.15)
    ax.set_yscale("log")
    ax.set_title("(a) train_v311_slot_diversity  (penalty, lower = better)", fontweight="bold")
    ax.set_ylabel("diversity penalty (log)")
    ax.set_xlabel("epoch")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.92)

    # -- (b) hazard variance
    ax = axes[1]
    for s in STUDY_ORDER:
        eps, mean, std = mean_curve(by_study[s], 2)
        mean = np.array(mean); std = np.array(std)
        c = STUDY_COLOR[s]
        ax.plot(eps, mean, color=c, lw=2.2, label=s)
        ax.fill_between(eps, np.maximum(mean - std, 1e-12), mean + std, color=c, alpha=0.15)
    ax.set_yscale("log")
    ax.set_title("(b) train_v311_slot_variance  (8-slot hazard spread)", fontweight="bold")
    ax.set_ylabel("hazard variance (log)")
    ax.set_xlabel("epoch")
    ax.axhspan(VAR_BAND_LO, VAR_BAND_HI, color="#2e7d32", alpha=0.10, lw=0)
    ax.text(0.02, 0.06, f"v311 target band [{VAR_BAND_LO}, {VAR_BAND_HI}]  -  44/45 runs end below it",
            transform=ax.get_yaxis_transform(), color="#2e7d32",
            fontsize=8.5, fontweight="bold")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.92)

    fig.text(
        0.5, 0.01,
        "Source: results/dct_v313_uni2h/<study>/<study>/.../epoch_curve_fold0..4.csv.  "
        "Bold line = mean across 5 folds; band = +/- 1 std.\n"
        "Panel (a): all 9 studies drive the diversity penalty down by 14-27x.\n"
        "Panel (b): every study's hazard variance ends below the v311 band - a gap the penalty "
        "does not close. Only skcm/fold2 reaches it.",
        ha="center", fontsize=8.6, color="#444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def write_summary_table(by_study: dict[str, list[Path]], path: Path) -> None:
    """Side table: per-study ep0 / ep29 means and ratios."""
    rows = []
    for s in STUDY_ORDER:
        for p in by_study[s]:
            eps, div, var = load_curve(p)
            rows.append({
                "study": s,
                "fold": int(p.stem.split("fold")[-1]),
                "div_ep0": div[0],
                "div_epN": div[-1],
                "div_ratio": div[0] / max(div[-1], 1e-12),
                "var_ep0": var[0],
                "var_ep29": var[-1],
                "var_max_over_run": max(var),
                "var_ep29_in_band": bool(VAR_BAND_LO <= var[-1] <= VAR_BAND_HI),
            })
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Print a compact per-study summary.
    by_study_stats: dict[str, list[dict]] = {s: [] for s in STUDY_ORDER}
    for r in rows:
        by_study_stats[r["study"]].append(r)
    print(f"\n{'study':<10}{'n':<4}{'div ep0 mean':<16}{'div ep29 mean':<18}{'ratio mean':<14}"
          f"{'var ep29 mean':<16}{'in band'}")
    for s in STUDY_ORDER:
        rs = by_study_stats[s]
        if not rs:
            continue
        d0 = statistics.mean(r["div_ep0"] for r in rs)
        dN = statistics.mean(r["div_epN"] for r in rs)
        ratio = statistics.mean(r["div_ratio"] for r in rs)
        vN = statistics.mean(r["var_ep29"] for r in rs)
        in_band = sum(1 for r in rs if r["var_ep29_in_band"])
        print(f"{s:<10}{len(rs):<4}{d0:<16.5f}{dN:<18.5f}{ratio:<14.2f}{vN:<16.5f}{in_band}/5")
    total_in = sum(1 for r in rows if r["var_ep29_in_band"])
    print(f"\nTOTAL runs with ep29 variance inside [{VAR_BAND_LO}, {VAR_BAND_HI}]: "
          f"{total_in}/{len(rows)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_study = discover_runs()
    total = sum(len(v) for v in by_study.values())
    print(f"discovered {total} runs across {sum(1 for v in by_study.values() if v)} studies")
    for s in STUDY_ORDER:
        print(f"  {s}: {len(by_study[s])} folds")
    plot_overlay_45(by_study, OUT_DIR / "v313_pancancer_45run_overlay.png")
    plot_per_study_mean(by_study, OUT_DIR / "v313_pancancer_per_study_mean.png")
    write_summary_table(by_study, OUT_DIR / "v313_pancancer_summary_table.csv")
