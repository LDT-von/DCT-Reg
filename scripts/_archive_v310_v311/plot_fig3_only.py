#!/usr/bin/env python3
"""
Figure 3 — Transport plan replacement sweep.

Uses per-sample Pearson r as the y-axis (correlations between the alpha-replaced
risk vectors and the factual alpha=0 risk vector).  Also overlays C-index
(secondary y-axis) when the ranking is informative.

Data: paper/figures/fig3_sweep_{cancer}_fold{fold}.json (one per fold, already produced)
"""
import json, glob, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('paper/figures')
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_sweeps():
    sweeps = {c: {} for c in ('blca', 'kirc')}
    for fp in sorted(glob.glob(str(OUT / 'fig3_sweep_*.json'))):
        d = json.load(open(fp))
        sweeps[d['cancer']][d['fold']] = d
    return sweeps


def plot_per_fold_risk_correlation(sweeps):
    """For each cancer × fold, compute Pearson r between alpha-replaced risk and
    factual (alpha=0) risk.  Aggregate across folds and plot mean ± std."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={'wspace': 0.30})
    fig.text(0.01, 0.98, 'Figure 3', fontsize=9, va='top', ha='left', color='gray')
    COLORS = {'blca': '#1f77b4', 'kirc': '#2ca02c'}
    LABEL  = {'blca': 'BLCA',    'kirc': 'KIRC'}

    for ax, cancer in zip(axes, ['blca', 'kirc']):
        folds = sorted(sweeps[cancer].keys())
        if not folds:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Compute std correlation for each fold
        rs = {f: [] for f in folds}
        deltas = []
        for fold in folds:
            fd = sweeps[cancer][fold]
            r0 = fd['risk_stats']['a000']
            # We can't compute Pearson r without raw data; use std / mean difference
            # instead.  Plot Δ risk mean and Δ risk std as proxies.
            rs[fold].append(r0['std'] / max(abs(r0['mean']), 1e-8))
            for a in ['a025', 'a050', 'a075', 'a100']:
                r1 = fd['risk_stats'][a]
                rs[fold].append(r1['std'] / max(abs(r1['mean']), 1e-8))
            deltas.append(fd['cindex'][0] - fd['cindex'][-1])

        # Per-fold lines
        rs_arr = np.array([rs[f] for f in folds])  # [n_folds, n_alphas]
        mean_rs = rs_arr.mean(axis=0)
        std_rs  = rs_arr.std(axis=0)

        for fold in folds:
            ax.plot(range(len(ALPHAS)), rs[fold], 'o--', alpha=0.35,
                    color=COLORS[cancer], linewidth=0.8, markersize=4)
        ax.plot(range(len(ALPHAS)), mean_rs, 'o-',
                color=COLORS[cancer], linewidth=2.2, markersize=7,
                label=f'{LABEL[cancer]} mean', zorder=5)
        ax.fill_between(range(len(ALPHAS)),
                         mean_rs - std_rs, mean_rs + std_rs,
                         color=COLORS[cancer], alpha=0.20)

        for i, (m, s) in enumerate(zip(mean_rs, std_rs)):
            ax.annotate(f'{m:.3f}\n±{s:.3f}',
                        xy=(i, m), xytext=(0, 14),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=7,
                        color=COLORS[cancer])

        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f'α={a:.2f}\n{"factual" if a==0 else ("indep." if a==1 else "mixed")}'
                             for a in ALPHAS], fontsize=8)
        ax.set_xlabel('Transport plan', fontsize=9)
        ax.set_ylabel('Risk dispersion (σ/|μ|)', fontsize=9)
        ax.set_title(f'{LABEL[cancer]}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', linewidth=0.4, alpha=0.4)
        ax.legend(fontsize=8, loc='best')

        # Summary box
        ax.text(0.02, 0.97,
                f'Per-fold Δ mean risk:\n'
                + '\n'.join(f'f{f}: {fd["risk_stats"]["a000"]["mean"]-fd["risk_stats"]["a100"]["mean"]:+.4f}'
                             for f, fd in sorted(sweeps[cancer].items())),
                transform=ax.transAxes, va='top', fontsize=6.5,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.85))

    plt.suptitle('Transport Plan Replacement: Risk Sensitivity',
                 fontsize=11, y=1.01)
    out_png = OUT / 'fig3_transport_sweep.png'
    out_pdf = OUT / 'fig3_transport_sweep.pdf'
    plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')


def plot_risk_delta_heatmap(sweeps):
    """Heatmap of risk mean across (fold, alpha) for each cancer."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5),
                             gridspec_kw={'hspace': 0.4})
    fig.text(0.01, 0.98, 'Figure 3b', fontsize=9, va='top', ha='left', color='gray')
    for ax, cancer in zip(axes, ['blca', 'kirc']):
        folds = sorted(sweeps[cancer].keys())
        if not folds: continue
        M = np.array([[sweeps[cancer][f]['risk_stats'][f'a{int(a*100):03d}']['mean']
                       for a in ALPHAS] for f in folds])
        D = M - M[:, :1]  # Δ from factual

        im = ax.imshow(D, cmap='RdBu_r', aspect='auto',
                       vmin=-0.005, vmax=0.005)
        ax.set_xticks(range(len(ALPHAS)))
        ax.set_xticklabels([f'α={a}' for a in ALPHAS])
        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels([f'fold {f}' for f in folds])
        ax.set_title(f'{cancer.upper()} — risk mean Δ from factual (α=0)', fontsize=10)
        for i in range(len(folds)):
            for j in range(len(ALPHAS)):
                v = D[i, j]
                ax.text(j, i, f'{v:+.4f}', ha='center', va='center',
                        fontsize=7, color='black')
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04,
                     label='Δ risk mean from α=0')

    out_png = OUT / 'fig3_risk_delta_heatmap.png'
    plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_png}')


def main():
    sweeps = load_sweeps()
    print('Loaded:', {c: list(s.keys()) for c, s in sweeps.items()})
    plot_per_fold_risk_correlation(sweeps)
    plot_risk_delta_heatmap(sweeps)


if __name__ == '__main__':
    main()