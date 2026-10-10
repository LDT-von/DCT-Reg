"""Plot E046 BLCA 5x3 paired C-index summary.

Inputs (read-only):
  /data1/results/v313_e046_blca_f0_smoke_20261009            (fold 0, all 3 arms)
  /data1/results/v313_e046_blca_remaining_folds_1to4_20261010 (folds 1-4, full+fixed)
  /data1/results/v313_e046_blca_wsi_additive_folds_1to4_20261009 (folds 1-4, additive)

Outputs (in /home/ubuntu/.cursor/projects/data1-DCT-Reg/terminals/):
  e046_blca_paired_bar.png
  e046_blca_paired_pairs.png
  e046_blca_train_curves.png
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDS = [0, 1, 2, 3, 4]
ARMS = ["full", "wsi_additive", "wsi_fixed_total"]
ARM_LABEL = {
    "full": "Full (λ_WSI=0)",
    "wsi_additive": "WSI Additive (λ=1.0)",
    "wsi_fixed_total": "WSI Fixed-Total (≈0.0333)",
}
ARM_COLOR = {
    "full": "#1f77b4",
    "wsi_additive": "#d62728",
    "wsi_fixed_total": "#2ca02c",
}

F0_ROOT = Path("/data1/results/v313_e046_blca_f0_smoke_20261009")
ADD_ROOT = Path("/data1/results/v313_e046_blca_wsi_additive_folds_1to4_20261009")
REM_ROOT = Path("/data1/results/v313_e046_blca_remaining_folds_1to4_20261010")


def best_cindex_from_log(log: Path) -> float:
    for line in reversed(log.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "best cindex" in line:
            return float(line.split("best cindex=")[1].split()[0])
    raise ValueError(f"no best cindex in {log}")


def log_for(arm: str, fold: int) -> Path:
    if fold == 0:
        return F0_ROOT / "logs" / f"{arm}_blca_f0_s3.log"
    if arm == "wsi_additive":
        return ADD_ROOT / "logs" / f"{arm}_blca_f{fold}_s3.log"
    return REM_ROOT / "logs" / f"{arm}_blca_f{fold}_s3.log"


def epoch_curve_csv(arm: str, fold: int) -> Path:
    if fold == 0:
        root = F0_ROOT / "outer_test" / arm / "blca" / f"fold0" / "seed3" / "blca" / "SurvOTRank_dct_v313_transport_reconstruction"
    elif arm == "wsi_additive":
        root = ADD_ROOT / "outer_test" / arm / "blca" / f"fold{fold}" / "seed3" / "blca" / "SurvOTRank_dct_v313_transport_reconstruction"
    else:
        root = REM_ROOT / "outer_test" / arm / "blca" / f"fold{fold}" / "seed3" / "blca" / "SurvOTRank_dct_v313_transport_reconstruction"
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(subdirs) == 1, f"expected one run dir under {root}, got {subdirs}"
    return subdirs[0] / f"epoch_curve_fold{fold}.csv"


# ---------- collect ----------
best = {arm: [] for arm in ARMS}
for arm in ARMS:
    for fold in FOLDS:
        best[arm].append(best_cindex_from_log(log_for(arm, fold)))

print("best C-index (5 folds x 3 arms):")
for arm in ARMS:
    print(f"  {arm:18s}  {best[arm]}  mean={statistics.mean(best[arm]):.4f}  std={statistics.pstdev(best[arm]):.4f}")

curves = {arm: [] for arm in ARMS}  # list of (epochs, cindex) per fold
for arm in ARMS:
    for fold in FOLDS:
        csv_path = epoch_curve_csv(arm, fold)
        epochs, cidxs = [], []
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # try a few common column names
                e = row.get("epoch") or row.get("Epoch") or row.get("ep")
                c = row.get("val_cindex") or row.get("cindex") or row.get("val cindex")
                if e is None or c is None:
                    continue
                try:
                    epochs.append(int(e)); cidxs.append(float(c))
                except ValueError:
                    pass
        curves[arm].append((np.array(epochs), np.array(cidxs)))

OUT = Path("/home/ubuntu/.cursor/projects/data1-DCT-Reg/terminals")

# ---------- figure 1: paired bar ----------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(FOLDS))
w = 0.27
for i, arm in enumerate(ARMS):
    ax.bar(x + (i - 1) * w, best[arm], w, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
for i, f in enumerate(FOLDS):
    for j, arm in enumerate(ARMS):
        ax.text(x[i] + (j - 1) * w, best[arm][i] + 0.003, f"{best[arm][i]:.3f}",
                ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"fold {f}" for f in FOLDS])
ax.set_ylabel("best val C-index (outer_test)")
ax.set_ylim(0.55, 0.80)
ax.set_title("E046 BLCA 3-arm ablation · outer_test · seed=3")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "e046_blca_paired_bar.png", dpi=140)
plt.close(fig)

# ---------- figure 2: paired ΔC-index (per fold, paired lines) ----------
deltas = {
    "add vs full": np.array(best["wsi_additive"]) - np.array(best["full"]),
    "add vs fixed": np.array(best["wsi_additive"]) - np.array(best["wsi_fixed_total"]),
    "fixed vs full": np.array(best["wsi_fixed_total"]) - np.array(best["full"]),
}
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for label, vals in deltas.items():
    ax.plot(FOLDS, vals, marker="o", label=f"{label} (mean={vals.mean():+.4f})")
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xticks(FOLDS)
ax.set_xlabel("fold")
ax.set_ylabel("Δ best C-index (paired)")
ax.set_title("E046 BLCA paired ΔC-index per fold (best)")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "e046_blca_paired_pairs.png", dpi=140)
plt.close(fig)

# ---------- figure 3: training curves (5x3) ----------
fig, axes = plt.subplots(len(FOLDS), 3, figsize=(11, 11), sharex=True, sharey=True)
for r, fold in enumerate(FOLDS):
    for c, arm in enumerate(ARMS):
        ax = axes[r, c]
        epochs, cidxs = curves[arm][r]
        ax.plot(epochs, cidxs, color=ARM_COLOR[arm])
        best_i = int(np.argmax(cidxs))
        ax.scatter([epochs[best_i]], [cidxs[best_i]], color="k", zorder=3, s=18,
                   label=f"best={cidxs[best_i]:.3f}@{epochs[best_i]}")
        if r == 0:
            ax.set_title(ARM_LABEL[arm], fontsize=9)
        if c == 0:
            ax.set_ylabel(f"fold {fold}\nval C-index", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.45, 0.85)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=7)
fig.suptitle("E046 BLCA validation C-index curves (5 folds × 3 arms)", y=1.00, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "e046_blca_train_curves.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- save summary table ----------
with (OUT / "e046_blca_paired_summary.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["fold", "full", "wsi_additive", "wsi_fixed_total",
                "delta_add_minus_full", "delta_add_minus_fixed", "delta_fixed_minus_full"])
    for i, f in enumerate(FOLDS):
        w.writerow([f, f"{best['full'][i]:.4f}", f"{best['wsi_additive'][i]:.4f}",
                    f"{best['wsi_fixed_total'][i]:.4f}",
                    f"{best['wsi_additive'][i] - best['full'][i]:+.4f}",
                    f"{best['wsi_additive'][i] - best['wsi_fixed_total'][i]:+.4f}",
                    f"{best['wsi_fixed_total'][i] - best['full'][i]:+.4f}"])
    mean_full = statistics.mean(best["full"]); mean_add = statistics.mean(best["wsi_additive"]); mean_fix = statistics.mean(best["wsi_fixed_total"])
    w.writerow(["mean", f"{mean_full:.4f}", f"{mean_add:.4f}", f"{mean_fix:.4f}",
                f"{mean_add - mean_full:+.4f}", f"{mean_add - mean_fix:+.4f}", f"{mean_fix - mean_full:+.4f}"])
    w.writerow(["std", f"{statistics.pstdev(best['full']):.4f}",
                f"{statistics.pstdev(best['wsi_additive']):.4f}",
                f"{statistics.pstdev(best['wsi_fixed_total']):.4f}", "", "", ""])

print("wrote:")
for p in ["e046_blca_paired_bar.png", "e046_blca_paired_pairs.png", "e046_blca_train_curves.png", "e046_blca_paired_summary.csv"]:
    print(" ", OUT / p)
