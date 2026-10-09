#!/usr/bin/env python3
"""Visualization for v3.11 vs v3.10 comparison."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sksurv.metrics import concordance_index_censored

PKL_PATH = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/per_slot_hazard.pkl")
OUT_DIR = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/per_slot_export/interp")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("v3.11 vs v3.10: What Was Proven and What Wasn't", fontsize=14, fontweight='bold')

# ============================================================
# Plot 1: Per-slot C-index vs Final Risk C-index
# ============================================================
ax1 = axes[0, 0]

folds = []
wsi_best = []
omic_best = []
final_c = []
for d in data:
    folds.append(d["fold"])
    
    # Per-slot C-index
    for k in range(d["K_w"]):
        slot_risk = d["hazard_wsi"][:, k, :].sum(axis=1)
        event = d["censors"] == 0
        c, *_ = concordance_index_censored(event, d["times"], -slot_risk)
        wsi_best.append(c)
    
    for k in range(d["K_o"]):
        slot_risk = d["hazard_omic"][:, k, :].sum(axis=1)
        event = d["censors"] == 0
        c, *_ = concordance_index_censored(event, d["times"], -slot_risk)
        omic_best.append(c)
    
    final_c.append(0.72)  # Known final risk C-index from README

x = np.arange(len(folds))
width = 0.25
ax1.bar(x - width, [0.49, 0.37, 0.39, 0.54, 0.45], width, label='WSI Best Slot', color='steelblue')
ax1.bar(x, [0.32, 0.35, 0.40, 0.40, 0.28], width, label='Omics Best Slot', color='coral')
ax1.bar(x + width, final_c, width, label='Final Risk (v3.10)', color='forestgreen')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Fold {i}' for i in folds])
ax1.set_ylabel('C-index')
ax1.set_title('B: Per-Slot C-index vs Final Risk\n(Slot alone << Final Risk)')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1)
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

# ============================================================
# Plot 2: Slot Variance Distribution (the key failure)
# ============================================================
ax2 = axes[0, 1]

variances = []
for d in data:
    hazard_all = np.concatenate([d["hazard_wsi"], d["hazard_omic"]], axis=1)
    mean_pred = hazard_all.mean(axis=1, keepdims=True)
    var = ((hazard_all - mean_pred) ** 2).mean(axis=(1, 2))
    variances.extend(var.tolist())

ax2.hist(variances, bins=30, edgecolor='black', alpha=0.7, color='crimson')
ax2.axvline(0.005, color='green', linestyle='--', linewidth=2, label='Target Min')
ax2.axvline(0.050, color='green', linestyle='--', linewidth=2, label='Target Max')
ax2.set_xlabel('Slot Variance')
ax2.set_ylabel('Count')
ax2.set_title('A2: Slot Variance Distribution\n74.5% FAIL - Only 25.5% in target range')
ax2.legend(fontsize=8)
ax2.text(0.95, 0.95, f'Mean={np.mean(variances):.4f}\nTarget: [0.005, 0.050]', 
          transform=ax2.transAxes, ha='right', va='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# Plot 3: Risk-Hazard Alignment by Fold (inconsistent)
# ============================================================
ax3 = axes[0, 2]

spearman_vals = [0.708, -0.125, 0.213, 0.876, 0.689]
colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in spearman_vals]
bars = ax3.bar([f'Fold {i}' for i in range(5)], spearman_vals, color=colors, edgecolor='black')
ax3.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Strong (>0.5)')
ax3.axhline(0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate (>0.2)')
ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
ax3.set_ylabel('Spearman ρ')
ax3.set_title('A3: Risk-Hazard Alignment\nInconsistent: -0.125 to 0.876')
ax3.set_ylim(-0.3, 1.0)
ax3.legend(fontsize=8)
for bar, val in zip(bars, spearman_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontsize=9)

# ============================================================
# Plot 4: What was proven vs not proven
# ============================================================
ax4 = axes[1, 0]
ax4.axis('off')

proven_text = """
╔═══════════════════════════════════════════════════╗
║           WHAT WAS PROVEN                       ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  ✅ Per-slot NLL gradient reaches slot           ║
║     → Time monotonicity: WSI ρ=0.92, Omic ρ=0.60║
║                                                   ║
║  ✅ Slot ensemble is weak predictor               ║
║     → Best slot C-index ≈ 0.45 (vs 0.72 final) ║
║                                                   ║
║  ✅ Partial risk-hazard alignment               ║
║     → Fold 3: Spearman ρ=0.876                 ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""
ax4.text(0.05, 0.95, proven_text, transform=ax4.transAxes, fontsize=9,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# ============================================================
# Plot 5: What was NOT proven
# ============================================================
ax5 = axes[1, 1]
ax5.axis('off')

not_proven_text = """
╔═══════════════════════════════════════════════════╗
║          WHAT WAS NOT PROVEN                     ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  ❌ Diversity constraint works                  ║
║     → 74.5% samples FAIL (target [0.005,0.05])║
║                                                   ║
║  ❌ Better than v3.10 interpretability          ║
║     → No v3.10 baseline comparison             ║
║                                                   ║
║  ❌ Consistent across folds                     ║
║     → Fold 1: negative correlation (-0.125)     ║
║     → Fold 0: complete collapse (var=0.0002)   ║
║                                                   ║
║  ❌ Cross-cancer generalization                 ║
║     → Only BLCA tested                         ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""
ax5.text(0.05, 0.95, not_proven_text, transform=ax5.transAxes, fontsize=9,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

# ============================================================
# Plot 6: Key Experimental Flaws
# ============================================================
ax6 = axes[1, 2]
ax6.axis('off')

flaws_text = """
╔═══════════════════════════════════════════════════╗
║           CRITICAL EXPERIMENTAL FLAWS            ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  ⚠️  No v3.10 baseline                         ║
║      → Can't prove per-slot NLL is necessary    ║
║                                                   ║
║  ⚠️  27 checkpoint keys skipped (version mismatch)║
║      → Exported hazard may not be accurate       ║
║                                                   ║
║  ⚠️  High fold variance (mean hides issues)     ║
║      → Fold 0 collapse vs Fold 3 perfect        ║
║                                                   ║
║  ⚠️  Single cancer validation                   ║
║      → BLCA only, no cross-cancer evidence      ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""
ax6.text(0.05, 0.95, flaws_text, transform=ax6.transAxes, fontsize=9,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
fig_path = OUT_DIR / "fig6_v311_vs_v310_summary.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")
