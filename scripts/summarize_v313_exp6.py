#!/usr/bin/env python3
"""Aggregate per-fold v3.13 diagnostic figures into per-cancer summary figures.

Inputs:
  results/v313_additional_20261008/blca_full/exp6_blca_f{f}_s3/additional.json + .npz
  results/v313_additional_20261008/kirc_full/exp6_kirc_f{f}_s3/additional.json + .npz
Outputs (one figure per cancer × class):
  - {cancer}_exp6_summary_reconstruction.{pdf,png} : mean latent error across 5 folds, 6 controls
  - {cancer}_exp6_summary_pathway_advantage.{pdf,png} : pathway heatmap mean across 5 folds
  - {cancer}_exp6_summary_pairing.{pdf,png}         : shuffled repeats + native cindex pool
  - {cancer}_exp6_summary_patch_deletion.{pdf,png}  : top/bottom/random deletion pool
  - {cancer}_exp6_summary_patch_budget.{pdf,png}    : retention curve pool
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/DCT-Reg/results/v313_additional_20261008")
OUT = ROOT / "figures_summary"
OUT.mkdir(parents=True, exist_ok=True)

RECON_MODES = ("native", "product", "direct", "shuffled", "train_mean", "self")


def load_run(arm_dir, run_id):
    run_dir = ROOT / arm_dir / run_id
    record = json.loads((run_dir / "additional.json").read_text(encoding="utf-8"))
    arrays = {k: np.asarray(v) for k, v in np.load(str(run_dir / "additional.npz")).items()}
    return record, arrays


def _mean_or_list(x):
    if isinstance(x, list):
        return float(np.mean(x))
    return float(x)


def _per_fold_cindex(records, attr):
    arr = []
    for r in records:
        rows = r["patches"][attr]
        per_row = []
        for row in rows:
            v = row[attr.split("_")[0] + "_cindex"] if attr != "budget" else row["cindex"]
            per_row.append(np.mean(v))
        arr.append(per_row)
    return np.array(arr)  # (n_fold, n_fractions)


def _per_fold_abs(records, attr):
    arr = []
    for r in records:
        rows = r["patches"][attr]
        per_row = []
        for row in rows:
            if attr == "deletion":
                top = row["top_abs_delta"]; bot = row["bottom_abs_delta"]; rnd = row["random_abs_delta"]
                per_row.append([top, bot, rnd])
            else:
                per_row.append([row["abs_delta"]])
        arr.append(per_row)
    return np.array(arr)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220})

    for cancer, arm_dir in [("blca", "blca_full"), ("kirc", "kirc_full")]:
        records, arrays_list = [], []
        for fold in range(5):
            rid = f"exp6_{cancer}_f{fold}_s3"
            record, arrays = load_run(arm_dir, rid)
            records.append(record)
            arrays_list.append(arrays)

        # --- summary_reconstruction: 6 controls mean error & top1 ---
        errors_per_mode = {}
        top1_per_mode = {}
        for k in RECON_MODES:
            if k == "shuffled":
                e = np.array([np.mean(np.mean(arr[f"reconstruction_shuffled_all_error"], axis=0)) for arr in arrays_list])
            else:
                e = np.array([np.mean(arr[f"reconstruction_{k}_error"]) for arr in arrays_list])
            errors_per_mode[k] = e
            top1_per_mode[k] = np.array([records[i]["reconstruction"][k]["top1"] for i in range(5)])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"{cancer.upper()} / exp6 / 5-fold pooled — latent reconstruction controls", fontsize=10)
        means = [errors_per_mode[k].mean() for k in RECON_MODES]
        axes[0].bar(RECON_MODES, means,
                    yerr=[errors_per_mode[k].std() for k in RECON_MODES],
                    color="#4382b4", capsize=3)
        axes[0].set_ylabel("Normalized latent reconstruction error (mean ± std across 5 folds) ↓")
        axes[0].tick_params(axis="x", rotation=35)
        top1_means = [top1_per_mode[k].mean() for k in RECON_MODES]
        top1_stds = [top1_per_mode[k].std() for k in RECON_MODES]
        axes[1].bar(RECON_MODES, top1_means, yerr=top1_stds, color="#4382b4", capsize=3)
        n_pat = len(arrays_list[0]["case_ids"])
        axes[1].axhline(1 / n_pat, color="#b54a48", ls="--", label=f"1/N chance (1/{n_pat})")
        axes[1].set_ylabel("Centered patient retrieval Top-1 (mean ± std) ↑")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].legend(fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, .93))
        fig.text(.5, .01, "Conditional on selected checkpoint; random repeats are not independent patients.",
                 ha="center", fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{cancer}_exp6_summary_reconstruction.{ext}", bbox_inches="tight")
        plt.close(fig)

        # --- summary_pathway_advantage: native - others, mean across 5 folds ---
        native_mean = np.stack([arr["reconstruction_native_error"].mean(0) for arr in arrays_list])  # (5, n_pathways)
        diffs = []
        for k in ("product", "direct", "train_mean"):
            d = np.stack([arr[f"reconstruction_{k}_error"].mean(0) for arr in arrays_list]) - native_mean
            diffs.append(d.mean(0))  # (n_pathways,)
        shuffled = np.stack([arr["reconstruction_shuffled_all_error"].mean((0, 1)) for arr in arrays_list])
        d = shuffled - native_mean
        diffs.append(d.mean(0))
        diffs = np.stack(diffs)  # (4, n_pathways)
        n_pathways = diffs.shape[1]
        fig, ax = plt.subplots(figsize=(12, 3.5))
        scale = max(float(np.abs(diffs).max()), 1e-8)
        heat = ax.imshow(diffs, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set_yticks(range(4), ["Product − Native", "Direct − Native",
                                  "Train mean − Native", "Shuffled − Native"])
        ax.set_xlabel(f"All {n_pathways} pathway indices (mean across 5 folds)")
        fig.colorbar(heat, ax=ax, label="Control error − native error (positive favors native)")
        fig.suptitle(f"{cancer.upper()} / exp6 / 5-fold pooled — pathway reconstruction advantage", fontsize=10)
        fig.tight_layout(rect=(0, .02, 1, .95))
        fig.text(.5, .005, "Conditional on selected checkpoint; random repeats are not independent patients.",
                 ha="center", fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{cancer}_exp6_summary_pathway_advantage.{ext}", bbox_inches="tight")
        plt.close(fig)

        # --- summary_pairing ---
        native_cindex = np.array([r["pairing"]["native_cindex"] for r in records])
        shuffled_cindex = np.concatenate([r["pairing"]["shuffled_cindex"] for r in records])
        per_patient = np.concatenate([np.abs(arr["pairing_shuffled_risk"] - arr["risk"][:, None]).mean(1)
                                      for arr in arrays_list])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{cancer.upper()} / exp6 / 5-fold pooled — shuffled pairing control", fontsize=10)
        axes[0].scatter(np.zeros(len(shuffled_cindex)), shuffled_cindex, color="#b54a48",
                        label=f"shuffled repeats (16 × 5 = {len(shuffled_cindex)})", alpha=.6)
        axes[0].scatter(np.array([-0.2] * 5), native_cindex, color="#4382b4", marker="s", s=60,
                        label="native per fold (n=5)")
        axes[0].axhline(native_cindex.mean(), color="#4382b4", ls="-",
                        label=f"native mean = {native_cindex.mean():.3f}")
        axes[0].axhline(shuffled_cindex.mean(), color="#b54a48", ls="--",
                        label=f"shuffled mean = {shuffled_cindex.mean():.3f}")
        axes[0].set_xticks([0], ["Mismatch WSI, keep recipient omics"])
        axes[0].set_ylabel("C-index")
        axes[0].legend(fontsize=7, loc="lower left")
        axes[1].hist(per_patient, bins=20, color="#4382b4",
                     label=f"pooled patients (n={len(per_patient)})")
        axes[1].axvline(per_patient.mean(), color="#b54a48", ls="--",
                        label=f"mean |Δ risk| = {per_patient.mean():.2f}")
        axes[1].set_xlabel("Per-patient mean |Δ risk| over shuffles")
        axes[1].set_ylabel("Patients")
        axes[1].legend(fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, .93))
        fig.text(.5, .01, "Conditional on selected checkpoint; random repeats are not independent patients.",
                 ha="center", fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{cancer}_exp6_summary_pairing.{ext}", bbox_inches="tight")
        plt.close(fig)

        # --- summary_patch_deletion ---
        fractions = np.array([d["removed_fraction"] for d in records[0]["patches"]["deletion"]])
        unpadded = np.array([r["patches"]["unpadded_cindex"] for r in records])
        top_c = np.array([[np.mean(d["top_cindex"]) - unpadded[i] for d in r["patches"]["deletion"]]
                          for i, r in enumerate(records)])  # (5, n_frac)
        bot_c = np.array([[np.mean(d["bottom_cindex"]) - unpadded[i] for d in r["patches"]["deletion"]]
                          for i, r in enumerate(records)])
        rnd_c = np.array([[np.mean(d["random_cindex"]) - unpadded[i] for d in r["patches"]["deletion"]]
                          for i, r in enumerate(records)])
        top_a = np.array([[d["top_abs_delta"] for d in r["patches"]["deletion"]] for r in records])
        bot_a = np.array([[d["bottom_abs_delta"] for d in r["patches"]["deletion"]] for r in records])
        rnd_a = np.array([[d["random_abs_delta"] for d in r["patches"]["deletion"]] for r in records])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{cancer.upper()} / exp6 / 5-fold pooled — patch deletion", fontsize=10)
        for label, arr_c, arr_a, color in (
            ("top", top_c, top_a, "#b54a48"),
            ("bottom", bot_c, bot_a, "#4382b4"),
            ("random", rnd_c, rnd_a, "#666666"),
        ):
            axes[0].plot(fractions, arr_c.mean(0), "o-", color=color, label=label)
            axes[0].fill_between(fractions, arr_c.min(0), arr_c.max(0), color=color, alpha=.15)
            axes[1].plot(fractions, arr_a.mean(0), "o-", color=color, label=label)
        axes[0].set_ylabel("Δ C-index from unpadded reference (mean + fold range)")
        axes[1].set_ylabel("Mean |Δ risk|")
        for ax in axes:
            ax.set_xlabel("Deleted fraction of verified real patches")
            ax.legend(fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, .93))
        fig.text(.5, .01, "Conditional on selected checkpoint; random repeats are not independent patients.",
                 ha="center", fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{cancer}_exp6_summary_patch_deletion.{ext}", bbox_inches="tight")
        plt.close(fig)

        # --- summary_patch_budget ---
        retained = np.array([1.0 - d["removed_fraction"] for d in records[0]["patches"]["deletion"]])
        bud_c = np.array([[np.mean(b["cindex"]) for b in r["patches"]["budget"]] for r in records])
        bud_a = np.array([[b["abs_delta"] for b in r["patches"]["budget"]] for r in records])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"{cancer.upper()} / exp6 / 5-fold pooled — patch budget", fontsize=10)
        axes[0].plot(retained, bud_c.mean(0), "o-", color="#4382b4", label="mean over 5 folds")
        axes[0].fill_between(retained, bud_c.min(0), bud_c.max(0), color="#4382b4", alpha=.2, label="fold range")
        axes[0].axhline(unpadded.mean(), color="#b54a48", ls="--",
                        label=f"unpadded cindex = {unpadded.mean():.3f}")
        axes[0].set_ylabel("C-index (mean + fold range)")
        axes[0].set_xlabel("Retained fraction of verified real patches")
        axes[0].legend(fontsize=7)
        axes[1].plot(retained, bud_a.mean(0), "o-", color="#4382b4")
        axes[1].set_xlabel("Retained fraction of verified real patches")
        axes[1].set_ylabel("Mean |Δ risk|")
        fig.tight_layout(rect=(0, .04, 1, .93))
        fig.text(.5, .01, "Conditional on selected checkpoint; random repeats are not independent patients.",
                 ha="center", fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{cancer}_exp6_summary_patch_budget.{ext}", bbox_inches="tight")
        plt.close(fig)

    n_files = 5 * 2 * 2  # 5 classes × 2 cancers × 2 formats
    print(f"[summary] wrote {n_files} files to {OUT}")


if __name__ == "__main__":
    main()