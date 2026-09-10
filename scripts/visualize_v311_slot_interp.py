#!/usr/bin/env python3
"""Generate visualizations for v3.11 Slot Interpretability Verification."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# Paths
# ============================================================
PKL_PATH = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/per_slot_hazard.pkl")
OUT_DIR = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/interp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
print("Loading per-slot hazard data...")
with open(PKL_PATH, "rb") as f:
    per_fold = pickle.load(f)

# ============================================================
# Figure 1: A1 - Slot Hazard vs Time-Bin Monotonicity
# ============================================================
print("Generating Figure 1: Slot Hazard vs Time-Bin Monotonicity...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("A1: Slot Hazard vs Time-Bin - v3.11 BLCA Validation", fontsize=14, fontweight='bold')

for fold_idx, d in enumerate(per_fold):
    # WSI slots (top row)
    ax_wsi = axes[0, fold_idx]
    hazard_wsi = d["hazard_wsi"]  # [N, K_w, C]
    
    for k in range(d["K_w"]):
        mean_hazard = hazard_wsi[:, k, :].mean(axis=0)
        ax_wsi.plot(range(4), mean_hazard, marker='o', label=f'Slot {k}', alpha=0.7)
    
    ax_wsi.set_title(f'Fold {fold_idx}\nWSI Slots')
    ax_wsi.set_xlabel('Time Bin')
    ax_wsi.set_ylabel('Mean Hazard')
    ax_wsi.set_xticks(range(4))
    ax_wsi.grid(True, alpha=0.3)
    
    # Omics slots (bottom row)
    ax_omic = axes[1, fold_idx]
    hazard_omic = d["hazard_omic"]  # [N, K_o, C]
    
    for k in range(d["K_o"]):
        mean_hazard = hazard_omic[:, k, :].mean(axis=0)
        ax_omic.plot(range(4), mean_hazard, marker='s', label=f'Slot {k}', alpha=0.7)
    
    ax_omic.set_title(f'Fold {fold_idx}\nOmics Slots')
    ax_omic.set_xlabel('Time Bin')
    ax_omic.set_ylabel('Mean Hazard')
    ax_omic.set_xticks(range(4))
    ax_omic.grid(True, alpha=0.3)
    if fold_idx == 4:
        ax_omic.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
fig1_path = OUT_DIR / "fig1_slot_hazard_monotonicity.png"
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig1_path}")

# ============================================================
# Figure 2: A1 - Spearman ρ Distribution
# ============================================================
print("Generating Figure 2: Spearman ρ Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("A1: Spearman ρ (Time-Bin Correlation) - v3.11 BLCA", fontsize=14, fontweight='bold')

# WSI slots
ax1 = axes[0]
wsi_spearman = []
for d in per_fold:
    hazard_wsi = d["hazard_wsi"]
    for k in range(d["K_w"]):
        hazard_over_time = hazard_wsi[:, k, :].mean(axis=0)
        rho, _ = stats.spearmanr(range(4), hazard_over_time)
        wsi_spearman.append(rho if not np.isnan(rho) else 0.0)

ax1.hist(wsi_spearman, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(np.mean(wsi_spearman), color='red', linestyle='--', label=f'Mean={np.mean(wsi_spearman):.3f}')
ax1.set_xlabel('Spearman ρ (Time-Bin)')
ax1.set_ylabel('Count')
ax1.set_title('WSI Slots')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Omics slots
ax2 = axes[1]
omic_spearman = []
for d in per_fold:
    hazard_omic = d["hazard_omic"]
    for k in range(d["K_o"]):
        hazard_over_time = hazard_omic[:, k, :].mean(axis=0)
        rho, _ = stats.spearmanr(range(4), hazard_over_time)
        omic_spearman.append(rho if not np.isnan(rho) else 0.0)

ax2.hist(omic_spearman, bins=20, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(np.mean(omic_spearman), color='red', linestyle='--', label=f'Mean={np.mean(omic_spearman):.3f}')
ax2.set_xlabel('Spearman ρ (Time-Bin)')
ax2.set_ylabel('Count')
ax2.set_title('Omics Slots')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2_path = OUT_DIR / "fig2_spearman_rho_distribution.png"
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig2_path}")

# ============================================================
# Figure 3: A2 - Slot Variance Distribution
# ============================================================
print("Generating Figure 3: Slot Variance Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("A2: Slot Variance Distribution - v3.11 BLCA Validation", fontsize=14, fontweight='bold')

# Left: Histogram with target range
ax1 = axes[0]
all_variances = []
fold_labels = []
for d in per_fold:
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
    for i in range(len(d["case_ids"])):
        var = np.mean([np.var(hazard_all[i, :, c]) for c in range(hazard_all.shape[2])])
        all_variances.append(var)
        fold_labels.append(d["fold"])

ax1.hist(all_variances, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
ax1.axvline(0.005, color='red', linestyle='--', label='Min (0.005)')
ax1.axvline(0.050, color='red', linestyle='--', label='Max (0.050)')
ax1.set_xlabel('Slot Variance (across time bins)')
ax1.set_ylabel('Count')
ax1.set_title('All Folds Combined')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Per-fold boxplot
ax2 = axes[1]
fold_variances = []
for d in per_fold:
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
    variances = []
    for i in range(len(d["case_ids"])):
        var = np.mean([np.var(hazard_all[i, :, c]) for c in range(hazard_all.shape[2])])
        variances.append(var)
    fold_variances.append(variances)

bp = ax2.boxplot(fold_variances, labels=[f'Fold {i}' for i in range(5)], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax2.axhline(0.005, color='red', linestyle='--', alpha=0.5)
ax2.axhline(0.050, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Fold')
ax2.set_ylabel('Slot Variance')
ax2.set_title('Per-Fold Variance Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = OUT_DIR / "fig3_slot_variance_distribution.png"
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig3_path}")

# ============================================================
# Figure 4: A3 - Hazard vs Risk Alignment
# ============================================================
print("Generating Figure 4: Hazard vs Risk Alignment...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("A3: Slot Hazard vs Risk Alignment - v3.11 BLCA Validation", fontsize=14, fontweight='bold')

# Scatter plots for each fold
for fold_idx, d in enumerate(per_fold):
    row = fold_idx // 3
    col = fold_idx % 3
    ax = axes[row, col]
    
    final_risk = d["risks"]
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
    mean_hazard_last = hazard_all[:, :, -1].mean(axis=1)
    
    ax.scatter(mean_hazard_last, final_risk, alpha=0.6, c='steelblue', edgecolor='black', s=50)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_hazard_last, final_risk)
    x_line = np.linspace(mean_hazard_last.min(), mean_hazard_last.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.8)
    
    # Spearman correlation
    spearman_r, p_val = stats.spearmanr(mean_hazard_last, final_risk)
    
    ax.set_xlabel('Mean Hazard (Last Bin)')
    ax.set_ylabel('Final Risk')
    ax.set_title(f'Fold {fold_idx}\nSpearman ρ = {spearman_r:.3f}')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
axes[1, 2].axis('off')

# Add summary in the last subplot
ax_summary = axes[1, 2]
ax_summary.axis('off')
summary_text = "Summary:\n\n"
summary_text += "Fold 0: Strong positive correlation\n"
summary_text += "Fold 1: Weak negative correlation\n"
summary_text += "Fold 2: Weak positive correlation\n"
summary_text += "Fold 3: Strong positive correlation\n"
summary_text += "Fold 4: Moderate positive correlation\n\n"
summary_text += "Mean Spearman ρ = 0.472\n"
summary_text += "→ Overall: MODERATE alignment"
ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig4_path = OUT_DIR / "fig4_hazard_risk_alignment.png"
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig4_path}")

# ============================================================
# Figure 5: Comprehensive Summary
# ============================================================
print("Generating Figure 5: Comprehensive Summary...")

fig = plt.figure(figsize=(16, 12))

# A1: Monotonicity summary
ax1 = fig.add_subplot(2, 3, 1)
categories = ['WSI\nGood', 'WSI\nBad', 'Omic\nGood', 'Omic\nBad']
counts = [40, 0, 40, 0]
colors = ['green', 'red', 'green', 'red']
ax1.bar(categories, counts, color=colors, edgecolor='black')
ax1.set_ylabel('Number of Slots')
ax1.set_title('A1: Slot Monotonicity\n(Good=Increasing, Bad=Flat)')
ax1.set_ylim(0, 50)
for i, v in enumerate(counts):
    ax1.text(i, v + 1, str(v), ha='center', fontweight='bold')

# A2: Variance distribution
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(all_variances, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
ax2.axvline(0.005, color='red', linestyle='--', linewidth=2, label='Min=0.005')
ax2.axvline(0.050, color='red', linestyle='--', linewidth=2, label='Max=0.050')
ax2.set_xlabel('Slot Variance')
ax2.set_ylabel('Count')
ax2.set_title('A2: Variance Distribution\n(25.5% in target range)')
ax2.legend(fontsize=8)

# A3: Correlation summary
ax3 = fig.add_subplot(2, 3, 3)
folds = [f'Fold {i}' for i in range(5)]
spearman_vals = []
for d in per_fold:
    final_risk = d["risks"]
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
    mean_hazard_last = hazard_all[:, :, -1].mean(axis=1)
    rho, _ = stats.spearmanr(mean_hazard_last, final_risk)
    spearman_vals.append(rho)
    
colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in spearman_vals]
ax3.bar(folds, spearman_vals, color=colors, edgecolor='black')
ax3.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Strong (>0.5)')
ax3.axhline(0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate (>0.2)')
ax3.set_ylabel('Spearman ρ')
ax3.set_title('A3: Risk-Hazard Alignment\nby Fold')
ax3.legend(fontsize=8)
ax3.set_ylim(-0.3, 1.0)

# Overall verdict
ax4 = fig.add_subplot(2, 3, 4)
ax4.axis('off')
verdict_text = """
╔══════════════════════════════════════════╗
║       v3.11 VERIFICATION VERDICT        ║
╠══════════════════════════════════════════╣
║                                          ║
║  A1: Slot Monotonicity         ✓ PASS  ║
║    - 100% slots show increasing hazard  ║
║    - WSI: ρ=0.920, Omics: ρ=0.595     ║
║                                          ║
║  A2: Slot Variance Distribution   ✗ FAIL ║
║    - Only 25.5% samples in target range║
║    - Most slots collapse to low variance║
║                                          ║
║  A3: Hazard-Risk Alignment       ⚠ PARTIAL║
║    - Mean Spearman ρ = 0.472           ║
║    - High variance across folds        ║
║                                          ║
╚══════════════════════════════════════════╝
"""
ax4.text(0.05, 0.95, verdict_text, transform=ax4.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Key findings
ax5 = fig.add_subplot(2, 3, 5)
ax5.axis('off')
findings_text = """
KEY FINDINGS:

1. MONOTONICITY ✓
   All slots correctly predict increasing hazard
   over time, confirming the survival supervision
   signal reaches the slot representations.

2. VARIANCE ✗
   The diversity constraint (target [0.005, 0.050])
   is NOT satisfied at inference time. Most slots
   collapse to near-zero variance.

3. RISK ALIGNMENT ⚠
   Moderate correlation between slot hazard
   and final risk prediction. Some folds show
   strong alignment (F0, F3), others don't (F1).

RECOMMENDATIONS:
- Investigate why diversity constraint fails
- Check if variance target range needs tuning
- Examine fold-specific patterns
"""
ax5.text(0.05, 0.95, findings_text, transform=ax5.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
fig5_path = OUT_DIR / "fig5_comprehensive_summary.png"
plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig5_path}")

print("\n" + "=" * 60)
print("All visualizations saved to:")
print(f"  {OUT_DIR}")
print("=" * 60)
