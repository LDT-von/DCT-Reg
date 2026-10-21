"""DCT v3.13: ten-cancer slot diversity training diagnostics (5-fold mean per study).

Sources
-------
Nine-cancer set:  results/dct_v313_uni2h/<study>/...  (brca,coadread,hnsc,kirc,
  luad,lusc,skcm,stad,ucec) — all 5 folds × 30 epochs.

BLCA (tentative): results/dct_v313_blca_uni2h/blca/...  (5 folds, 30 epochs,
  same column names, same rW_8_rG_8_sp configuration).
  ⚠ NOTE: This is NOT the same experiment batch as the nine-cancer set.
  It uses the SAME model recipe (dct_v313_transport_reconstruction, seed3,
  uni2-h, rW_8_rG_8_sp) but is a separate BLCA run.
  Included as the closest available match; cross-experiment comparisons are
  indicative only.

Metrics
-------
Left panel:  train_v311_slot_diversity  (diversity penalty, lower = constraint
  closer to satisfied).  Label: "diversity penalty (log)"; NOT "slot distance".
Right panel: train_v311_slot_variance   (slot hazard variance, source: CSV
  column; reported for all studies without per-channel breakdown).
  Label: "hazard variance (log)".

Variance target band
--------------------
VAR_BAND_LO = 0.001, VAR_BAND_HI = 0.050.
⚠ SOURCE: these thresholds are from v311's dct_v311 model definition
  (survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py).
  They are applied here to v313 as-is WITHOUT validation that they are
  appropriate for v313's slot_variance distribution.  Treat the band as a
  reference line of unknown relevance, not a calibrated target.

Usage
-----
    python scripts/plot_v313_tencancer_mean_curves.py
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

# Output
OUT_DIR = ROOT / "docs/research/dct_v313_slot_diversity_figure"

# Nine-cancer data source
UNI2H = ROOT / "results/dct_v313_uni2h"

# Tentative BLCA source (different experiment batch)
BLCA_DIR = ROOT / "results/dct_v313_blca_uni2h"

# 10 studies
STUDY_ORDER = [
    "blca",    # tentative — different batch, same recipe & columns
    "brca", "coadread", "hnsc", "kirc",
    "luad", "lusc", "skcm", "stad", "ucec",
]

# 11 colours (BLCA gets bold black; others from a colour-blind-safe palette)
_COLORS = [
    "#000000",   # 0  blca  — bold / black
    "#0072B2",   # 1  brca
    "#E69F00",   # 2  coadread
    "#56B4E9",   # 3  hnsc
    "#009E73",   # 4  kirc
    "#F0E442",   # 5  luad  — high-luminosity yellow
    "#CC79A7",   # 6  lusc
    "#D55E00",   # 7  skcm
    "#7F7F7F",   # 8  stad
    "#BCBD22",   # 9  ucec
]
STUDY_COLOR = dict(zip(STUDY_ORDER, _COLORS))

# Variance target band — BORROWED from v311 without v313 validation
VAR_BAND_LO = 0.001
VAR_BAND_HI = 0.050

# --- Data loading helpers ---------------------------------------------------

def discover_nine_cancer() -> dict[str, list[Path]]:
    """Discover 9-cancer runs from results/dct_v313_uni2h."""
    by_study = {s: [] for s in STUDY_ORDER if s != "blca"}
    for csv_path in sorted(UNI2H.glob("*/*/*/*/epoch_curve_fold*.csv")):
        study = csv_path.relative_to(UNI2H).parts[0]
        if study not in by_study:
            continue
        by_study[study].append(csv_path)
    for s in by_study:
        by_study[s] = sorted(by_study[s], key=lambda p: int(p.stem.split("fold")[-1]))
    return by_study


def discover_blca() -> list[Path]:
    """Discover BLCA runs from results/dct_v313_blca_uni2h.

    Layout: blca/blca/<method>/<run-name>/epoch_curve_fold*.csv
    Two runs present: e_1 (smoke) and e_30 (full 30-epoch).
    Only the e_30 run is used here.
    """
    raw = sorted(
        BLCA_DIR.glob("blca/blca/*/*/epoch_curve_fold*.csv"),
        key=lambda p: int(p.stem.split("fold")[-1]),
    )
    # Filter to the 30-epoch run only (smoke e_1 run excluded)
    paths = [p for p in raw if "_e_30_" in str(p)]
    if not paths and raw:
        print(f"  [BLCA] WARNING: e_30 run not found; using all {len(raw)} files")
        paths = raw
    return paths


def load_curve(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Return (epochs, diversity_term, variance)."""
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    epochs   = [int(r["epoch"]) for r in rows]
    div      = [float(r["train_v311_slot_diversity"]) for r in rows]
    var      = [float(r["train_v311_slot_variance"])   for r in rows]
    return epochs, div, var


def mean_across_folds(
    paths: list[Path], series_idx: int
) -> tuple[list[int], list[float], list[int]]:
    """Average a series (by index) across fold paths; return (epochs, mean, n_folds)."""
    loaded = [load_curve(p)[series_idx] for p in paths]
    n = min(len(s) for s in loaded)
    stacked = np.array([s[:n] for s in loaded], dtype=float)
    return (list(range(n)), stacked.mean(axis=0).tolist(), len(paths))


def audit_folds(paths: list[Path], study: str) -> dict:
    """Return audit dict for a study's fold list."""
    folds_found = []
    for p in paths:
        fold = int(p.stem.split("fold")[-1])
        try:
            epochs, div, var = load_curve(p)
        except Exception as exc:
            folds_found.append({"fold": fold, "status": f"ERROR: {exc}"})
            continue
        n_epochs = len(epochs)
        has_nan = any(not np.isfinite(v) for v in div + var)
        folds_found.append({
            "fold": fold,
            "status": "ok",
            "n_epochs": n_epochs,
            "has_nan": has_nan,
            "div_ep0": div[0],
            "div_epN": div[-1],
            "var_ep0": var[0],
            "var_epN": var[-1],
        })
    fold_nums = [f["fold"] for f in folds_found if "fold" in f]
    unique = len(set(fold_nums))
    n_ok   = sum(1 for f in folds_found if f.get("status") == "ok")
    return {
        "study": study,
        "n_folds_total": len(paths),
        "n_folds_ok": n_ok,
        "n_epochs_ok": all(f.get("n_epochs") == 30 for f in folds_found if f.get("status") == "ok"),
        "unique_folds": unique,
        "has_nan": any(f.get("has_nan", False) for f in folds_found),
        "fold_details": folds_found,
    }


# --- Plotting --------------------------------------------------------------

def plot_ten_cancer_mean(
    nine: dict[str, list[Path]],
    blca: list[Path],
    out_png: Path,
) -> None:
    # Compute per-study mean curves
    study_means: dict[str, tuple[list[int], list[float], int]] = {}

    # BLCA (special handling: different batch, use bold black)
    study_means["blca"] = mean_across_folds(blca, 1)  # diversity
    # store variance separately
    _, blca_var, blca_n = mean_across_folds(blca, 2)

    for s in STUDY_ORDER:
        if s == "blca":
            continue
        study_means[s] = mean_across_folds(nine[s], 1)

    # Build variance dict
    study_var: dict[str, list[float]] = {"blca": blca_var}
    for s in STUDY_ORDER:
        if s == "blca":
            continue
        _, v, _ = mean_across_folds(nine[s], 2)
        study_var[s] = v

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Shared legend handles built during plotting
    handles_left  = []
    handles_right = []

    for i, s in enumerate(STUDY_ORDER):
        epochs, mean_div, n_folds = study_means[s]
        mean_var  = study_var[s]
        c = STUDY_COLOR[s]
        lw = 3.0 if s == "blca" else 2.0
        ls = "-"  # solid for all; BLCA is thicker + black

        # Left panel: diversity penalty
        ax = axes[0]
        ln, = ax.plot(epochs, mean_div, color=c, lw=lw, ls=ls,
                      label=f"{s.upper()}  (n={n_folds})")
        handles_left.append(ln)

        # Right panel: hazard variance
        ax = axes[1]
        ln2, = ax.plot(epochs, mean_var, color=c, lw=lw, ls=ls,
                       label=f"{s.upper()}  (n={n_folds})")
        handles_right.append(ln2)

    # --- Panel (a): diversity penalty
    ax = axes[0]
    ax.set_yscale("log")
    ax.set_title(
        "(a) Diversity penalty  (log scale)\n"
        "lower = constraint closer to satisfied",
        fontweight="bold", fontsize=12,
    )
    ax.set_ylabel("diversity penalty (log)")
    ax.set_xlabel("epoch")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10)

    # --- Panel (b): hazard variance
    ax = axes[1]
    ax.set_yscale("log")
    ax.set_title(
        "(b) Slot hazard variance  (log scale)\n"
        "⚠ Band from v311 model.py; not validated for v313",
        fontweight="bold", fontsize=12,
    )
    ax.set_ylabel("hazard variance (log)")
    ax.set_xlabel("epoch")
    # Variance target band (unvalidated for v313)
    ax.axhspan(VAR_BAND_LO, VAR_BAND_HI,
               color="#2e7d32", alpha=0.10, lw=0, zorder=0)
    ax.axhline(VAR_BAND_LO, color="#2e7d32", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(VAR_BAND_HI, color="#2e7d32", lw=0.8, ls="--", alpha=0.7)
    ax.text(
        0.98, 0.97,
        f"v311 target band\n[{VAR_BAND_LO}, {VAR_BAND_HI}]\n"
        "(source: v311 model.py — unvalidated for v313)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#2e7d32",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0fff0", ec="#2e7d32", lw=0.8),
    )
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=10)

    # Shared legend below both panels
    # Merge handles (left and right are same lines); only show one copy
    fig.legend(
        handles=handles_left,
        loc="lower center", ncol=5,
        fontsize=9,
        framealpha=0.92,
        borderaxespad=0.5,
        title="BLCA = tentative (different batch from 9-cancer set)",
        title_fontsize=8.5,
    )

    fig.suptitle(
        "DCT v3.13: ten-cancer slot diversity training diagnostics  "
        "(5-fold mean per study, seed 3, 30 epochs)\n"
        "BLCA from dct_v313_blca_uni2h; 9-cancer from dct_v313_uni2h",
        fontsize=13, fontweight="bold", y=0.99,
    )

    fig.text(
        0.5, 0.01,
        "Panel (a): lower diversity penalty → pair-separation constraint closer to satisfied.  "
        "DO NOT read as 'slots become more diverse'; it means the penalty is small.\n"
        "Panel (b): slot hazard variance.  "
        "Band [0.001, 0.050] borrowed from v311 model.py — not calibrated for v313.\n"
        "⚠ BLCA (dct_v313_blca_uni2h) is a separate experiment batch from the 9-cancer set "
        "(dct_v313_uni2h).  Same recipe & columns; treat cross-group comparisons as indicative.\n"
        "Source: results/dct_v313_blca_uni2h/  +  results/dct_v313_uni2h/",
        ha="center", fontsize=8.2, color="#555",
        style="italic",
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_png, dpi=190, facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def write_mean_csv(
    nine: dict[str, list[Path]],
    blca: list[Path],
    out_csv: Path,
) -> None:
    """Write study-level mean curve CSV with fold-count audit column."""
    header = [
        "study", "source_batch", "n_folds_in_avg",
        "epoch",
        "diversity_penalty_mean",
        "hazard_variance_mean",
        "hazard_variance_in_v311_band",
    ]
    rows: list[dict] = []

    # Gather max epoch across all runs for alignment
    all_paths = list(blca) + [p for ss in nine.values() for p in ss]
    max_epoch = 0
    for p in all_paths:
        try:
            _, d, v = load_curve(p)
            max_epoch = max(max_epoch, len(d))
        except Exception:
            pass
    epochs_range = list(range(max_epoch))

    # BLCA
    for ep in epochs_range:
        sub_div, sub_var = [], []
        for p in blca:
            try:
                _, d, v = load_curve(p)
                if ep < len(d):
                    sub_div.append(d[ep])
                    sub_var.append(v[ep])
            except Exception:
                pass
        if sub_div:
            rows.append({
                "study": "BLCA",
                "source_batch": "dct_v313_blca_uni2h",
                "n_folds_in_avg": len(blca),
                "epoch": ep,
                "diversity_penalty_mean": statistics.mean(sub_div),
                "hazard_variance_mean": statistics.mean(sub_var),
                "hazard_variance_in_v311_band": (
                    VAR_BAND_LO <= statistics.mean(sub_var) <= VAR_BAND_HI
                ),
            })

    # nine cancer
    for s in STUDY_ORDER:
        if s == "blca":
            continue
        paths = nine[s]
        for ep in epochs_range:
            sub_div, sub_var = [], []
            for p in paths:
                try:
                    _, d, v = load_curve(p)
                    if ep < len(d):
                        sub_div.append(d[ep])
                        sub_var.append(v[ep])
                except Exception:
                    pass
            if sub_div:
                rows.append({
                    "study": s.upper(),
                    "source_batch": "dct_v313_uni2h",
                    "n_folds_in_avg": len(paths),
                    "epoch": ep,
                    "diversity_penalty_mean": statistics.mean(sub_div),
                    "hazard_variance_mean": statistics.mean(sub_var),
                    "hazard_variance_in_v311_band": (
                        VAR_BAND_LO <= statistics.mean(sub_var) <= VAR_BAND_HI
                    ),
                })

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")


def write_audit_report(
    nine: dict[str, list[Path]],
    blca: list[Path],
) -> None:
    """Print fold-level audit to stdout."""
    print("\n=== AUDIT REPORT ===")

    # BLCA
    a = audit_folds(blca, "BLCA (tentative, dct_v313_blca_uni2h)")
    print(f"\n{a['study']}:")
    print(f"  source_batch: {a['study']} (separate from nine-cancer set)")
    print(f"  n_folds_total={a['n_folds_total']}, n_folds_ok={a['n_folds_ok']}, "
          f"unique_folds={a['unique_folds']}, n_epochs_ok={a['n_epochs_ok']}, "
          f"has_nan={a['has_nan']}")
    for f in a["fold_details"]:
        status = f.get("status", "?")
        if status == "ok":
            print(f"    fold {f['fold']}: OK, epochs={f['n_epochs']}, "
                  f"div={f['div_ep0']:.4f}→{f['div_epN']:.4f}, "
                  f"var={f['var_ep0']:.2e}→{f['var_epN']:.2e}, nan={f['has_nan']}")
        else:
            print(f"    fold {f['fold']}: {status}")

    # Nine cancer
    for s in STUDY_ORDER:
        if s == "blca":
            continue
        a = audit_folds(nine[s], s.upper())
        print(f"\n{a['study']}:")
        print(f"  n_folds_total={a['n_folds_total']}, n_folds_ok={a['n_folds_ok']}, "
              f"unique_folds={a['unique_folds']}, n_epochs_ok={a['n_epochs_ok']}, "
              f"has_nan={a['has_nan']}")
        for f in a["fold_details"]:
            status = f.get("status", "?")
            if status == "ok":
                print(f"    fold {f['fold']}: OK, epochs={f['n_epochs']}, "
                      f"div={f['div_ep0']:.4f}→{f['div_epN']:.4f}, "
                      f"var={f['var_ep0']:.2e}→{f['var_epN']:.2e}, nan={f['has_nan']}")
            else:
                print(f"    fold {f['fold']}: {status}")


# --- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering data...")
    nine = discover_nine_cancer()
    blca = discover_blca()

    print(f"BLCA (tentative): {len(blca)} folds")
    for s in STUDY_ORDER:
        if s == "blca":
            continue
        print(f"  {s.upper()}: {len(nine[s])} folds")

    write_audit_report(nine, blca)

    plot_ten_cancer_mean(
        nine, blca,
        OUT_DIR / "v313_tencancer_mean_curves.png",
    )
    write_mean_csv(
        nine, blca,
        OUT_DIR / "v313_tencancer_mean_curves.csv",
    )
