#!/usr/bin/env python3
"""
Proof D — Slot-Specific Survival Head 可解释性可视化
=====================================================
画 4 面板证明图，证明"8 个 slot 学到了 8 种不同的预后模式"。

Panel A: Hazard vs Time per slot (wsi + omic 双行)
Panel B: Top-K Hazard KM risk stratification
Panel C: 8 slot hazard 分布对比 (violin + heatmap)
Panel D: WSI Slot → Patch Attention 热图 (每 slot top patch 分布)
"""

from __future__ import annotations
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
REPO = Path("/data1/DCT-Reg")
DATA_DIR = REPO / "results/dct_v311_blca_uni_fixed/per_slot_export"
OUT_DIR = REPO / "results/proof_D_visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PKL_PATH = DATA_DIR / "per_slot_hazard.pkl"
CSV_PATH = DATA_DIR / "per_slot_hazard.csv"
STYLE = "seaborn-v0_8-paper"

# palette: 8 slot colors
SLOT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4", "#f032e6",
]

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_pkl():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)

def load_csv():
    df = pd.read_csv(CSV_PATH)
    return df

def compute_monotonicity(hazard_3d):
    """hazard_3d shape (N, K, C); return (K,) Spearman ρ vs time_bin."""
    N, K, C = hazard_3d.shape
    time_bins = np.arange(C)
    rhos = []
    for k in range(K):
        mean_h = hazard_3d[:, k, :].mean(axis=0)   # (C,)
        rho, p = stats.spearmanr(time_bins, mean_h)
        rhos.append(rho)
    return np.array(rhos)

def km_stratify(times, censors, risk_scores, n_groups=2):
    """Split patients into risk groups by median risk_score."""
    median = np.median(risk_scores)
    group = (risk_scores > median).astype(int)   # 0=low, 1=high
    return group

# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Hazard vs Time per slot
# ─────────────────────────────────────────────────────────────────────────────
def plot_panel_A(fig, outer, modality, hazard_3d, fold_data, title_suffix=""):
    """
    Draw per-slot hazard curves over time bins.
    outer: matplotlib GridSpec outer layout cell
    """
    ax = fig.add_subplot(outer)
    N, K, C = hazard_3d.shape
    time_bins = np.arange(C)
    mean_h = hazard_3d.mean(axis=0)   # (K, C)

    # per-slot std shading
    std_h = hazard_3d.std(axis=0)     # (K, C)

    for k in range(K):
        col = SLOT_COLORS[k]
        mean_curve = mean_h[k]
        std_curve  = std_h[k]
        ax.plot(time_bins, mean_curve, color=col, lw=2,
                label=f"Slot {k}", zorder=10-k)
        ax.fill_between(time_bins,
                         np.maximum(mean_curve - std_h[k], 0),
                         mean_curve + std_h[k],
                         color=col, alpha=0.12)

    mod_label = "WSI" if modality == "wsi" else "Omics"
    ax.set_xlabel("Time Bin", fontsize=10)
    ax.set_ylabel("Hazard", fontsize=10)
    ax.set_title(f"{mod_label} Hazard Curves{title_suffix}", fontsize=11, fontweight="bold")
    ax.set_xticks(time_bins)
    ax.set_xticklabels([f"Bin {b}" for b in time_bins])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Compute & annotate monotonicity
    rhos = compute_monotonicity(hazard_3d)
    rho_text = " | ".join([f"ρ{k}={rhos[k]:+.2f}" for k in range(K)])
    ax.text(0.02, 0.97, rho_text, transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left",
            color="dimgray", style="italic")

    return rhos

# ─────────────────────────────────────────────────────────────────────────────
# Panel B — KM Risk Stratification
# ─────────────────────────────────────────────────────────────────────────────
def plot_panel_B(fig, outer, times, censors, risk_scores, title_suffix=""):
    """
    Kaplan-Meier curves split by median risk.
    Shows whether the model's risk scores actually separate survival.
    """
    ax = fig.add_subplot(outer)
    groups = km_stratify(times, censors, risk_scores)

    t_low  = times[groups == 0]
    e_low  = censors[groups == 0]
    t_high = times[groups == 1]
    e_high = censors[groups == 1]

    kmf_low  = KaplanMeierFitter().fit(t_low,  e_low,  label="Low Risk")
    kmf_high = KaplanMeierFitter().fit(t_high, e_high, label="High Risk")

    kmf_low.plot_survival_function(ax=ax, color="#3cb44b", lw=2, ci_show=True)
    kmf_high.plot_survival_function(ax=ax, color="#e6194b", lw=2, ci_show=True)

    # log-rank
    result = logrank_test(t_low, t_high, e_low, e_high)
    pval_str = f"p = {result.p_value:.2e}" if result.p_value < 0.001 else f"p = {result.p_value:.3f}"
    ax.text(0.05, 0.05, f"Log-rank {pval_str}", transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Survival Time (months)", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.set_title(f"KM Risk Stratification{title_suffix}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    return result.p_value

# ─────────────────────────────────────────────────────────────────────────────
# Panel C — Slot diversity heatmap + distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_panel_C(fig, outer, hazard_wsi, hazard_omic):
    """
    Two sub-panels:
      C1 (left): heatmap of per-slot mean hazard at each time bin
      C2 (right): violin plot of hazard distribution per slot (last time bin)
    """
    ax = fig.add_subplot(outer)
    N, K, C = hazard_wsi.shape

    # Aggregate: mean over patients, show both modalities side by side
    mean_w = hazard_wsi.mean(axis=0)   # (K, C)
    mean_o = hazard_omic.mean(axis=0)

    # Build matrix [K, 2*C] for heatmap
    mat = np.hstack([mean_w, mean_o])  # (K, 2C)
    row_labels = [f"Slot {k}" for k in range(K)]
    col_labels = ([f"W·B{b}" for b in range(C)] +
                  [f"O·B{b}" for b in range(C)])

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(K))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("C: Mean Hazard Heatmap\n(W=WSI, O=Omics; B=bins)", fontsize=10, fontweight="bold")

    # Annotate values
    for i in range(K):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=5.5, color="black" if mat[i,j] < mat.mean() else "white")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean Hazard")

    # Bottom strip: slot variance (measure of differentiation)
    var_per_slot = hazard_wsi[:, :, -1].std(axis=0)   # (K,)
    for k, v in enumerate(var_per_slot):
        ax.text(mat.shape[1] + 0.3, k, f"σ={v:.3f}", va="center", fontsize=7,
                color=SLOT_COLORS[k])

    return mat, var_per_slot

# ─────────────────────────────────────────────────────────────────────────────
# Panel D — Slot → Patch/Omics Attention (top-feature bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_panel_D(fig, outer, wsi_assign, omic_assign, n_top=10):
    """
    For each slot, show the distribution of top-assigned patches/omics features.
    Instead of raw heatmaps (4096 patches too dense), show:
      D1: bar chart of mean slot attention per slot (bar height = mean attention)
      D2: per-slot "spread" = how concentrated the attention is (entropy)
    """
    ax = fig.add_subplot(outer)

    N, K, N_patches = wsi_assign.shape
    _, _, N_omics   = omic_assign.shape

    # Mean attention per slot across all patients
    mean_wsi_attn = wsi_assign.mean(axis=0)   # (K, N_patches)
    mean_omic_attn = omic_assign.mean(axis=0) # (K, N_omics)

    # Per-slot mean attention (normalized)
    slot_mean_w = mean_wsi_attn.mean(axis=1)   # (K,)
    slot_mean_o = mean_omic_attn.mean(axis=1)

    x = np.arange(K)
    w = 0.35
    ax.bar(x - w/2, slot_mean_w, w, color=[SLOT_COLORS[k] for k in range(K)],
           label="WSI Patches", alpha=0.85)
    ax.bar(x + w/2, slot_mean_o, w, color=[SLOT_COLORS[k] for k in range(K)],
           label="Omics Features", alpha=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in range(K)])
    ax.set_xlabel("Slot", fontsize=10)
    ax.set_ylabel("Mean Attention", fontsize=10)
    ax.set_title(f"D: Slot Attention (WSI & Omics)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Annotate entropy per slot (higher = more diffuse = less focused)
    def entropy(arr):
        arr = arr / (arr.sum() + 1e-9)
        return -np.sum(arr * np.log(arr + 1e-9))

    entropies_w = [entropy(mean_wsi_attn[k]) for k in range(K)]
    entropies_o = [entropy(mean_omic_attn[k]) for k in range(K)]
    for k in range(K):
        ax.text(k, max(slot_mean_w[k], slot_mean_o[k]) + 0.001,
                f"Hw={entropies_w[k]:.1f}", ha="center", fontsize=6, color="dimgray")

    return slot_mean_w, slot_mean_o, entropies_w, entropies_o

# ─────────────────────────────────────────────────────────────────────────────
# Main figure — 2×2 grid, per fold
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(fold_data: dict, fold_idx: int) -> plt.Figure:
    hazard_wsi = fold_data["hazard_wsi"]   # (N, 8, 4)
    hazard_omic = fold_data["hazard_omic"]
    wsi_assign  = fold_data["wsi_coordinate_assignment"]  # (N, 8, 4096)
    omic_assign = fold_data["omic_coordinate_assignment"] # (N, 8, 329)
    times   = fold_data["times"]
    censors = fold_data["censors"]
    risks   = fold_data["risks"]
    N = len(times)

    # ── Main figure — 3 rows × 2 cols ─────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Proof D — Slot-Specific Survival Head (BLCA Fold {fold_idx}, N={N})\n"
        f"'8 slots learn 8 distinct prognostic patterns'",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.2, 1.0, 0.85],
                           hspace=0.45, wspace=0.32)

    # ── Row 0: Panel A (full width) ────────────────────────────────────────────
    axA = fig.add_subplot(gs[0, :])
    for mod_idx, (mod, haz) in enumerate([("wsi", hazard_wsi), ("omic", hazard_omic)]):
        offset = mod_idx * (haz.shape[2] + 2)
        for k in range(haz.shape[1]):
            col = SLOT_COLORS[k]
            mean_c = haz[:, k, :].mean(axis=0)
            std_c  = haz[:, k, :].std(axis=0)
            x_base = offset + np.arange(haz.shape[2])
            alpha = 1.0 if mod == "wsi" else 0.75
            lw = 2.0 if mod == "wsi" else 1.5
            lbl = f"S{k}_W" if mod == "wsi" else f"S{k}_O"
            axA.plot(x_base, mean_c, color=col, lw=lw, alpha=alpha, label=lbl)
            axA.fill_between(x_base,
                             np.maximum(mean_c - std_c, 0),
                             mean_c + std_c,
                             color=col, alpha=0.08)
        if mod_idx == 0:
            axA.axvline(x=offset + haz.shape[2] + 0.9, color="gray",
                         linestyle="--", lw=0.8, alpha=0.5)

    # Monotonicity annotation
    rhos_w = compute_monotonicity(hazard_wsi)
    rhos_o = compute_monotonicity(hazard_omic)
    rho_txt = " | ".join([f"ρ{k}={rhos_w[k]:+.2f}" for k in range(8)])
    axA.text(0.01, 0.97, "WSI " + rho_txt, transform=axA.transAxes,
             fontsize=6.5, va="top", ha="left", color="crimson", style="italic")
    rho_txt2 = " | ".join([f"ρ{k}={rhos_o[k]:+.2f}" for k in range(8)])
    axA.text(0.01, 0.90, "Omic " + rho_txt2, transform=axA.transAxes,
             fontsize=6.5, va="top", ha="left", color="navy", style="italic")

    axA.set_xticks([])
    axA.set_xlabel("Time Bins  ← WSI Slots  |  Omics Slots →", fontsize=10)
    axA.set_ylabel("Mean Hazard (± std)", fontsize=10)
    axA.set_title("A: Hazard vs Time — 8 WSI Slots (left) & 8 Omics Slots (right)",
                  fontsize=11, fontweight="bold")
    axA.legend(ncol=8, loc="upper right", fontsize=7.5, framealpha=0.85)
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="y", alpha=0.3)

    # ── Row 1, Col 0: Panel B (KM) ─────────────────────────────────────────────
    axB = fig.add_subplot(gs[1, 0])
    groups = km_stratify(times, censors, risks)
    t_low  = times[groups == 0];  e_low  = censors[groups == 0]
    t_high = times[groups == 1];  e_high = censors[groups == 1]
    kmf_low  = KaplanMeierFitter().fit(t_low,  e_low,  label="Low Risk")
    kmf_high = KaplanMeierFitter().fit(t_high, e_high, label="High Risk")
    kmf_low.plot_survival_function(ax=axB, color="#3cb44b", lw=2, ci_show=True, ci_alpha=0.2)
    kmf_high.plot_survival_function(ax=axB, color="#e6194b", lw=2, ci_show=True, ci_alpha=0.2)
    result = logrank_test(t_low, t_high, e_low, e_high)
    pval = result.p_value
    pval_str = f"p = {pval:.2e}" if pval < 0.001 else f"p = {pval:.3f}"
    axB.text(0.05, 0.08, f"Log-rank {pval_str}", transform=axB.transAxes,
             fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axB.set_xlabel("Survival Time (months)", fontsize=10)
    axB.set_ylabel("Survival Probability", fontsize=10)
    axB.set_title("B: KM Risk Stratification (median split)", fontsize=11, fontweight="bold")
    axB.legend(loc="lower left", fontsize=9)
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(axis="y", alpha=0.3)

    # ── Row 1, Col 1: Panel C (heatmap) ───────────────────────────────────────
    axC = fig.add_subplot(gs[1, 1])
    mean_w = hazard_wsi.mean(axis=0)
    mean_o = hazard_omic.mean(axis=0)
    mat = np.hstack([mean_w, mean_o])
    row_labels = [f"Slot {k}" for k in range(8)]
    col_labels = ([f"W·B{b}" for b in range(4)] + [f"O·B{b}" for b in range(4)])
    im = axC.imshow(mat, aspect="auto", cmap="YlOrRd")
    axC.set_xticks(range(mat.shape[1]))
    axC.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7.5)
    axC.set_yticks(range(8))
    axC.set_yticklabels(row_labels, fontsize=8.5)
    axC.set_title("C: Mean Hazard Heatmap (W=WSI, O=Omics)", fontsize=10, fontweight="bold")
    for i in range(8):
        for j in range(mat.shape[1]):
            axC.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                     fontsize=6, color="black" if mat[i,j] < 0.05 else "white")
    plt.colorbar(im, ax=axC, shrink=0.7, label="Mean Hazard")
    # Slot variance annotation
    var_per_slot = hazard_wsi[:, :, -1].std(axis=0)
    for k, v in enumerate(var_per_slot):
        axC.text(mat.shape[1] + 0.2, k, f"σ={v:.3f}", va="center", fontsize=6.5,
                 color=SLOT_COLORS[k])

    # ── Row 2, Col 0: Panel D (slot attention bar) ────────────────────────────
    axD = fig.add_subplot(gs[2, 0])
    mean_wsi_attn = wsi_assign.mean(axis=0)
    mean_omic_attn = omic_assign.mean(axis=0)
    slot_mean_w = mean_wsi_attn.mean(axis=1)
    slot_mean_o = mean_omic_attn.mean(axis=1)
    x = np.arange(8)
    bw = 0.35
    axD.bar(x - bw/2, slot_mean_w, bw, color=[SLOT_COLORS[k] for k in range(8)],
            label="WSI Patches", alpha=0.85)
    axD.bar(x + bw/2, slot_mean_o, bw, color=[SLOT_COLORS[k] for k in range(8)],
            label="Omics Features", alpha=0.5, hatch="//")
    axD.set_xticks(x)
    axD.set_xticklabels([f"S{k}" for k in range(8)])
    axD.set_xlabel("Slot", fontsize=9)
    axD.set_ylabel("Mean Attention", fontsize=9)
    axD.set_title("D: Slot → WSI/Omics Attention", fontsize=10, fontweight="bold")
    axD.legend(fontsize=8)
    axD.spines[["top", "right"]].set_visible(False)
    axD.grid(axis="y", alpha=0.3)

    # ── Row 2, Col 1: Panel E (slot separation summary) ───────────────────────
    axE = fig.add_subplot(gs[2, 1])
    axE.axis("off")

    # Summary metrics
    rhos_w = compute_monotonicity(hazard_wsi)
    rhos_o = compute_monotonicity(hazard_omic)
    var_wsi = hazard_wsi[:, :, -1].std(axis=0)   # (8,)
    var_omic = hazard_omic[:, :, -1].std(axis=0)

    lines = [
        "─" * 50,
        "  SUMMARY METRICS",
        "─" * 50,
        f"  Patients: {N}",
        f"  Monotonicity (Spearman ρ vs time):",
        f"    WSI  : {'  '.join([f'S{k}={rhos_w[k]:+.2f}' for k in range(8)])}",
        f"    Omics: {'  '.join([f'S{k}={rhos_o[k]:+.2f}' for k in range(8)])}",
        f"    WSI mean ρ:  {rhos_w.mean():+.3f}",
        f"    Omics mean ρ: {rhos_o.mean():+.3f}",
        "",
        f"  Slot variance at last bin (WSI):",
        f"    {'  '.join([f'S{k}={var_wsi[k]:.3f}' for k in range(8)])}",
        f"    σ mean={var_wsi.mean():.3f}  σ std={var_wsi.std():.3f}",
        "",
        f"  Risk Stratification:",
        f"    Log-rank p = {pval:.2e}",
        f"    Low/High risk N: {groups.sum()}/{len(groups)-groups.sum()}",
        "─" * 50,
    ]
    text = "\n".join(lines)
    axE.text(0.05, 0.95, text, transform=axE.transAxes,
             fontsize=7.5, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    return fig, {
        "fold": fold_idx, "n": N,
        "rhos_wsi": rhos_w.tolist(), "rhos_omic": rhos_o.tolist(),
        "var_wsi": var_wsi.tolist(), "var_omic": var_omic.tolist(),
        "km_pval": pval, "slot_mean_attn_wsi": slot_mean_w.tolist(),
        "slot_mean_attn_omic": slot_mean_o.tolist(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Combined figure — all 5 folds side by side (Panel B only for compactness)
# ─────────────────────────────────────────────────────────────────────────────
def make_combined_km_figure(all_data: list) -> plt.Figure:
    """One row per fold: 3 columns (A+B+C) × 5 folds."""
    n_folds = len(all_data)
    fig, axes = plt.subplots(n_folds, 3, figsize=(18, 3.2 * n_folds))
    fig.suptitle("Proof D — All 5 Folds: A=Hazard Curve | B=KM | C=Heatmap",
                 fontsize=13, fontweight="bold", y=0.99)
    if n_folds == 1:
        axes = axes.reshape(1, -1)

    all_metrics = []
    for fi, fd in enumerate(all_data):
        hazard_wsi  = fd["hazard_wsi"]
        hazard_omic = fd["hazard_omic"]
        times   = fd["times"]
        censors = fd["censors"]
        risks   = fd["risks"]

        # ── Col A: Hazard curves ───────────────────────────────────────────────
        axA = axes[fi, 0]
        for mod_idx, (mod, haz) in enumerate([("wsi", hazard_wsi), ("omic", hazard_omic)]):
            offset = mod_idx * (haz.shape[2] + 1.5)
            for k in range(haz.shape[1]):
                mc = haz[:, k, :].mean(axis=0)
                xb = offset + np.arange(haz.shape[2])
                axA.plot(xb, mc, color=SLOT_COLORS[k],
                         lw=1.5, alpha=0.85)
            if mod_idx == 0:
                axA.axvline(x=offset + haz.shape[2] + 0.7, color="gray",
                             linestyle="--", lw=0.7, alpha=0.4)
        if fi == 0:
            axA.set_title("A: Hazard vs Time", fontsize=9, fontweight="bold")
        axA.set_xticks([])
        if fi == n_folds - 1:
            axA.set_xlabel("Time Bins", fontsize=8)
        axA.set_ylabel(f"F{fi}\nHazard", fontsize=7.5)
        axA.spines[["top", "right"]].set_visible(False)

        # ── Col B: KM ─────────────────────────────────────────────────────────
        axB = axes[fi, 1]
        groups = km_stratify(times, censors, risks)
        kmf_low  = KaplanMeierFitter().fit(
            times[groups==0], censors[groups==0], label="Low")
        kmf_high = KaplanMeierFitter().fit(
            times[groups==1], censors[groups==1], label="High")
        kmf_low.plot_survival_function(ax=axB, color="#3cb44b", lw=1.5, ci_show=False)
        kmf_high.plot_survival_function(ax=axB, color="#e6194b", lw=1.5, ci_show=False)
        res = logrank_test(times[groups==0], times[groups==1],
                          censors[groups==0], censors[groups==1])
        pval = res.p_value
        if fi == 0:
            axB.set_title("B: KM Stratification", fontsize=9, fontweight="bold")
        axB.set_xlabel("Time (mo)", fontsize=7.5)
        axB.set_ylabel("Surv.", fontsize=7.5)
        axB.text(0.05, 0.12, f"p={pval:.2e}", transform=axB.transAxes,
                 fontsize=7, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        axB.spines[["top", "right"]].set_visible(False)
        if fi == 0:
            axB.legend(fontsize=7, loc="lower left")

        # ── Col C: Heatmap ────────────────────────────────────────────────────
        axC = axes[fi, 2]
        mat = np.hstack([hazard_wsi.mean(axis=0), hazard_omic.mean(axis=0)])
        axC.imshow(mat, aspect="auto", cmap="YlOrRd")
        axC.set_xticks(range(mat.shape[1]))
        axC.set_xticklabels(
            [f"W{b}" for b in range(4)] + [f"O{b}" for b in range(4)],
            fontsize=6)
        axC.set_yticks(range(8))
        axC.set_yticklabels([f"S{k}" for k in range(8)], fontsize=7)
        if fi == 0:
            axC.set_title("C: Hazard Heatmap", fontsize=9, fontweight="bold")
        axC.set_xlabel("W/B | O/B", fontsize=7.5)

        # Metrics dict for this fold
        rhos_w = compute_monotonicity(hazard_wsi)
        rhos_o = compute_monotonicity(hazard_omic)
        all_metrics.append({
            "fold": fi, "n": len(times),
            "mean_rho_wsi": round(float(rhos_w.mean()), 3),
            "mean_rho_omic": round(float(rhos_o.mean()), 3),
            "km_pval": round(float(pval), 4),
        })

    return fig, all_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Proof D — Slot-Specific Survival Head Visualization")
    print("=" * 70)

    all_data = load_pkl()
    out = {"folds": [], "summary": {}}

    # ── Individual fold figures ────────────────────────────────────────────────
    for fi, fd in enumerate(all_data):
        print(f"\n--- Fold {fi} ---")
        fig, metrics = make_figure(fd, fi)
        fig_path = OUT_DIR / f"proof_D_fold{fi}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fig_path}")
        out["folds"].append(metrics)

    # ── Combined figure ─────────────────────────────────────────────────────────
    print("\n--- Combined 5-fold figure ---")
    fig_comb, all_metrics = make_combined_km_figure(all_data)
    comb_path = OUT_DIR / "proof_D_all_folds_combined.png"
    fig_comb.savefig(comb_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_comb)
    print(f"  Saved: {comb_path}")
    out["summary"] = {"combined_km": all_metrics}

    # ── Aggregate summary ───────────────────────────────────────────────────────
    agg = {
        "mean_rho_wsi":  np.mean([m["mean_rho_wsi"]  for m in all_metrics]),
        "mean_rho_omic": np.mean([m["mean_rho_omic"] for m in all_metrics]),
        "all_km_pvals":  [m["km_pval"] for m in all_metrics],
        "folds_significant": sum(1 for p in [m["km_pval"] for m in all_metrics] if p < 0.05),
    }
    out["summary"]["aggregate"] = agg
    print(f"\n  Mean ρ WSI:  {agg['mean_rho_wsi']:+.3f}")
    print(f"  Mean ρ Omics: {agg['mean_rho_omic']:+.3f}")
    print(f"  KM p<0.05: {agg['folds_significant']}/5 folds")

    out_path = OUT_DIR / "proof_D_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {out_path}")

    print("\n✅ Proof D complete!")
    print(f"   Individual: {OUT_DIR}/proof_D_fold{{0..4}}.png")
    print(f"   Combined:   {OUT_DIR}/proof_D_all_folds_combined.png")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
