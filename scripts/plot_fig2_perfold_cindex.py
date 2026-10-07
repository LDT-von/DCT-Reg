#!/usr/bin/env python3
"""
Figure 2: Per-fold C-index comparison
  Panel A: BLCA — Full / Direct / Independent, 5-fold scatter + connecting lines
  Panel B: KIRC — Full / Direct / Independent, 5-fold scatter + connecting lines
  Panel C: BLCA — Δ = Full−Direct, Full−Independent per fold
  Panel D: KIRC — Δ = Full−Direct, Full−Independent per fold

Data source: results/v313_evidence_v2/controls.json + EXP6 epoch_curves
"""
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

def load_exp6_folds(cancer):
    """EXP6 Full per-fold best cindex."""
    base = Path(f'results/dct_v313_ablation_Exp6_full/{cancer}/{cancer}/SurvOTRank_dct_v313_transport_reconstruction')
    sub = next(base.iterdir())
    out = {}
    for f in range(5):
        curves = list(csv.DictReader(open(sub / f'epoch_curve_fold{f}.csv')))
        best = max(curves, key=lambda r: float(r['val_cindex']))
        out[f] = float(best['val_cindex'])
    return out

def load_ctrl(cancer, arm):
    """direct / independent from controls.json."""
    raw = json.load(open('results/v313_evidence_v2/controls.json'))['runs']
    out = {}
    for r in raw:
        if r['arm'] == arm and r['cancer'] == cancer:
            curves = list(csv.DictReader(open(r['curve'])))
            best = max(curves, key=lambda row: float(row['val_cindex']))
            out[r['fold']] = float(best['val_cindex'])
    return out

blca_full = load_exp6_folds('blca')
blca_dir  = load_ctrl('blca', 'direct')
blca_ind  = load_ctrl('blca', 'independent')
kirc_full = load_exp6_folds('kirc')
kirc_dir  = load_ctrl('kirc', 'direct')
kirc_ind  = load_ctrl('kirc', 'independent')

folds = list(range(5))

# ── Colour / style ──────────────────────────────────────────────────────────
COLORS = {
    'Full':        '#1f77b4',   # blue
    'Direct':      '#ff7f0e',   # orange
    'Independent': '#2ca02c',   # green
}
MARKERS = {'Full': 'o', 'Direct': 's', 'Independent': '^'}
ALPHA   = 0.9
LW_CONN = 0.8
LW_ZERO = 0.6

# ── Figure layout ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9, 7.5),
                         gridspec_kw={'hspace': 0.35, 'wspace': 0.30})
fig.text(0.02, 0.98, 'Figure 2', fontsize=9, va='top', ha='left',
         color='gray')

def plot_panel(ax, full, direct, independent, cancer, show_legend=True):
    """Panel A/B: scatter + connecting lines for the three arms."""
    arms = [('Full', full), ('Direct', direct), ('Independent', independent)]
    x = np.arange(5)        # fold index 0-4
    x_jitter = {'Full': x - 0.12, 'Direct': x, 'Independent': x + 0.12}

    # Connecting lines per arm
    for label, vals in arms:
        ax.plot(x_jitter[label], [vals[f] for f in folds],
                color=COLORS[label], alpha=ALPHA,
                linewidth=LW_CONN, zorder=1)

    # Scatter per arm
    for label, vals in arms:
        ax.scatter(x_jitter[label], [vals[f] for f in folds],
                   color=COLORS[label], marker=MARKERS[label],
                   s=48, alpha=ALPHA, zorder=2, edgecolors='white',
                   linewidths=0.4)

    # Summary: mean bar
    for i, (label, vals) in enumerate(arms):
        mean_v = np.mean([vals[f] for f in folds])
        bar_x = i / 3 - 0.17  # centered below fold labels
        ax.bar(i / 3, mean_v, width=0.18, bottom=0,
               color=COLORS[label], alpha=0.25, zorder=0)
        ax.text(i / 3, mean_v + 0.004, f'{mean_v:.3f}',
                ha='center', va='bottom', fontsize=6.5,
                color=COLORS[label], fontweight='bold')

    ax.set_title(f'{cancer}', fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds], fontsize=8)
    ax.set_ylabel('C-index', fontsize=9)
    ax.set_ylim(0.58, 0.90)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', linewidth=0.4, alpha=0.4)

    if show_legend:
        patches = [mpatches.Patch(color=COLORS[l], label=l) for l, _ in arms]
        ax.legend(handles=patches, loc='lower right', fontsize=8,
                  framealpha=0.85, edgecolor='gray')

def plot_delta_panel(ax, full, direct, independent, cancer):
    """Panel C/D: per-fold Δ = Full−Direct, Full−Independent."""
    x = np.arange(5)

    d_dir = [full[f] - direct[f] for f in folds]
    d_ind = [full[f] - independent[f] for f in folds]

    w = 0.35
    b1 = ax.bar(x - w/2, d_dir, width=w,
                color=COLORS['Full'], alpha=0.80, label='Full − Direct',
                edgecolor='white', linewidth=0.4)
    b2 = ax.bar(x + w/2, d_ind, width=w,
                color=COLORS['Independent'], alpha=0.80,
                label='Full − Independent',
                edgecolor='white', linewidth=0.4)

    # Annotate bars
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:+.3f}', ha='center', va='bottom', fontsize=6.5,
                color=COLORS['Full'])
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:+.3f}', ha='center', va='bottom', fontsize=6.5,
                color=COLORS['Independent'])

    # Zero line
    ax.axhline(0, color='gray', linewidth=LW_ZERO, linestyle='--', zorder=0)

    # Mean Δ annotation
    mean_dd = np.mean(d_dir)
    mean_di = np.mean(d_ind)
    ax.axhline(mean_dd, color=COLORS['Full'], linewidth=0.6,
               linestyle=':', alpha=0.7,
               xmin=0.05, xmax=0.95)
    ax.axhline(mean_di, color=COLORS['Independent'], linewidth=0.6,
               linestyle=':', alpha=0.7,
               xmin=0.05, xmax=0.95)
    ax.text(4.55, mean_dd, f'Δ̄={mean_dd:+.3f}',
            va='center', fontsize=7, color=COLORS['Full'])
    ax.text(4.55, mean_di, f'Δ̄={mean_di:+.3f}',
            va='center', fontsize=7, color=COLORS['Independent'])

    ax.set_title(f'{cancer} — improvement', fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds], fontsize=8)
    ax.set_ylabel('Δ C-index (Full − arm)', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_ylim(-0.10, 0.12)
    ax.grid(axis='y', linewidth=0.4, alpha=0.4)

    # Count positives
    n_dd = sum(1 for v in d_dir if v > 0)
    n_di = sum(1 for v in d_ind if v > 0)
    ax.text(0.02, 0.97,
            f'Full>Direct: {n_dd}/5 folds\nFull>Indep: {n_di}/5 folds',
            transform=ax.transAxes, va='top', fontsize=7.5,
            color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.8))

    ax.legend(loc='lower left', fontsize=8, framealpha=0.85, edgecolor='gray')

# ── Draw ─────────────────────────────────────────────────────────────────────
plot_panel(axes[0, 0], blca_full, blca_dir, blca_ind, 'BLCA')
plot_panel(axes[0, 1], kirc_full, kirc_dir, kirc_ind, 'KIRC')
plot_delta_panel(axes[1, 0], blca_full, blca_dir, blca_ind, 'BLCA')
plot_delta_panel(axes[1, 1], kirc_full, kirc_dir, kirc_ind, 'KIRC')

# Row labels
fig.text(0.01, 0.72, 'A  Per-fold C-index', va='center', ha='left',
         fontsize=9, fontweight='bold')
fig.text(0.01, 0.26, 'B  Full−arm Δ per fold', va='center', ha='left',
         fontsize=9, fontweight='bold')

plt.suptitle('Per-Fold C-Index: Full (v3.13) vs Control Arms',
             fontsize=11, y=1.01)

path = OUT / 'fig2_perfold_cindex.png'
plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {path}')

# Also save PDF
path_pdf = OUT / 'fig2_perfold_cindex.pdf'
fig2, _ = plt.subplots(2, 2, figsize=(9, 7.5),
                        gridspec_kw={'hspace': 0.35, 'wspace': 0.30})
fig2.text(0.02, 0.98, 'Figure 2', fontsize=9, va='top', ha='left', color='gray')
plot_panel(fig2.axes[0], blca_full, blca_dir, blca_ind, 'BLCA')
plot_panel(fig2.axes[1], kirc_full, kirc_dir, kirc_ind, 'KIRC')
plot_delta_panel(fig2.axes[2], blca_full, blca_dir, blca_ind, 'BLCA')
plot_delta_panel(fig2.axes[3], kirc_full, kirc_dir, kirc_ind, 'KIRC')
fig2.text(0.01, 0.72, 'A  Per-fold C-index', va='center', ha='left',
           fontsize=9, fontweight='bold')
fig2.text(0.01, 0.26, 'B  Full−arm Δ per fold', va='center', ha='left',
           fontsize=9, fontweight='bold')
plt.suptitle('Per-Fold C-Index: Full (v3.13) vs Control Arms',
             fontsize=11, y=1.01)
plt.savefig(path_pdf, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {path_pdf}')
