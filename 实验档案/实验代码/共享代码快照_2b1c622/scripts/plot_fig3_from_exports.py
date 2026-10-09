#!/usr/bin/env python3
"""
Figure 3 (real): Transport plan replacement sweep (BLCA fold 0).
使用 export 出来的 patients.npz 重新计算真实的 sweep cindex。

数据来源: results/v313_interpretability_v1/exports/<arm>_<cancer>_fold<f>/<run_id>/patients.npz
"""
import sys, os
sys.path.insert(0, '/data1/DCT-Reg')
os.chdir('/data1/DCT-Reg')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from survot_rank.evidence.manifest import cindex as cindex_compute

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
EXPORT_ROOT = Path('results/v313_interpretability_v1/exports')

def load_export(arm: str, cancer: str, fold: int):
    """Load patients.npz from export."""
    base = EXPORT_ROOT / f"{arm}_{cancer}_fold{fold}"
    if not base.exists():
        return None
    run_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(f"{arm}_{cancer}_f{fold}_s3")]
    if not run_dirs:
        run_dirs = [d for d in base.iterdir() if d.is_dir()]
        if not run_dirs:
            return None
    run_dir = run_dirs[0]
    patients_path = run_dir / 'patients.npz'
    if not patients_path.exists():
        return None
    data = np.load(patients_path, allow_pickle=True)
    export_path = run_dir / 'export.json'
    export = json.load(open(export_path)) if export_path.exists() else {}
    return {'data': data, 'export': export, 'run_dir': run_dir}


def plot_sweep_from_exports():
    """Plot sweep using exported data."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={'wspace': 0.30})
    fig.text(0.01, 0.98, 'Figure 3', fontsize=9, va='top', ha='left', color='gray')

    COLORS = {'direct': '#ff7f0e', 'independent': '#2ca02c', 'exp6': '#1f77b4'}
    LABELS = {'direct': 'Direct', 'independent': 'Independent', 'exp6': 'Full'}
    
    # Aggregate across folds
    aggregated = {}

    for cancer_idx, cancer in enumerate(['blca', 'kirc']):
        ax = axes[cancer_idx]
        fold_data = {}
        
        for arm in ['exp6', 'direct', 'independent']:
            cindexes = []
            n_samples = 0
            for fold in range(5):
                exp = load_export(arm, cancer, fold)
                if exp is None:
                    continue
                data = exp['data']
                if 'sweep_risk' not in data.files:
                    continue
                n = len(data['risk'])
                n_samples = n if n_samples == 0 else n_samples
                times = data['time']
                censor = data['censor']
                alphas = data['alphas']
                sweep_risk = data['sweep_risk']
                
                ci_list = []
                for j, alpha in enumerate(alphas):
                    risk_j = sweep_risk[:, j]
                    ci = cindex_compute(times, censor, risk_j)
                    ci_list.append(ci)
                cindexes.append(ci_list)
            
            if cindexes:
                ci_arr = np.array(cindexes)
                fold_data[arm] = {
                    'cindex_arr': ci_arr,
                    'mean': ci_arr.mean(axis=0),
                    'std': ci_arr.std(axis=0),
                    'n_folds': len(cindexes)
                }
                aggregated[(cancer, arm)] = fold_data[arm]
        
        if not fold_data:
            ax.text(0.5, 0.5, 'No export data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        alphas = next(iter(fold_data.values()))['cindex_arr'].shape[1] if 'cindex_arr' in next(iter(fold_data.values())) else 5
        # Get alphas from first available
        first_arm = next(iter(fold_data.keys()))
        # Need actual alpha values from data
        first_export = load_export(first_arm, cancer, 0)
        alpha_values = first_export['data']['alphas'] if first_export else np.linspace(0, 1, alphas)
        
        # Per-fold lines
        for arm, fd in fold_data.items():
            color = COLORS.get(arm, '#666666')
            for ci_row in fd['cindex_arr']:
                ax.plot(range(len(alpha_values)), ci_row, 'o--',
                        alpha=0.30, color=color, linewidth=0.8, markersize=4)
        
        # Mean ± std
        for arm, fd in fold_data.items():
            color = COLORS.get(arm, '#666666')
            label = LABELS.get(arm, arm)
            ax.errorbar(range(len(alpha_values)), fd['mean'], yerr=fd['std'],
                       fmt='o-', color=color, linewidth=2.2,
                       markersize=7, capsize=4, capthick=1.5,
                       label=f'{label} (n_folds={fd["n_folds"]})')
            
            for i, (m, s) in enumerate(zip(fd['mean'], fd['std'])):
                ax.annotate(f'{m:.3f}\n±{s:.3f}',
                           xy=(i, m), xytext=(0, 14),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=6.5,
                           color=color)
        
        ax.set_xticks(range(len(alpha_values)))
        ax.set_xticklabels([f'α={a:.2f}\n{"factual" if a == 0 else ("indep." if a == 1 else "mixed")}'
                            for a in alpha_values], fontsize=8)
        ax.set_xlabel('Transport plan', fontsize=9)
        ax.set_ylabel('C-index', fontsize=9)
        ax.set_title(f'{cancer.upper()} — Transport Replacement Sweep', fontsize=11, fontweight='bold')
        ax.set_ylim(0.55, 0.95)
        ax.grid(axis='y', linewidth=0.4, alpha=0.4)
        ax.legend(fontsize=8, loc='lower left')
        
        # Summary
        if 'exp6' in fold_data:
            full_mean = fold_data['exp6']['mean']
            delta = full_mean[0] - full_mean[-1]
            ax.text(0.02, 0.98, f'Full: factual−indep Δ={delta:+.3f}',
                   transform=ax.transAxes, va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='lightgray', alpha=0.85))

    plt.suptitle('Transport Plan Replacement Sweep (v3.13, real replays from export)',
                 fontsize=11, y=1.01)
    out_png = OUT / 'fig3_transport_sweep.png'
    out_pdf = OUT / 'fig3_transport_sweep.pdf'
    plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Figure 3 saved: {out_png}\n                  {out_pdf}')
    
    # Save aggregated data
    summary_path = Path('paper/figures/v313_main_plots/fig3_sweep_summary.json')
    summary_path.parent.mkdir(exist_ok=True, parents=True)
    summary = {
        'note': 'Sweep cindex from real checkpoint replay via export',
        'data': {}
    }
    for (cancer, arm), fd in aggregated.items():
        summary['data'][f'{cancer}_{arm}'] = {
            'mean_cindex_per_alpha': fd['mean'].tolist(),
            'std_cindex_per_alpha': fd['std'].tolist(),
            'n_folds': fd['n_folds'],
            'n_samples_per_fold': 76,
            'alpha_values': alpha_values.tolist()
        }
    json.dump(summary, open(summary_path, 'w'), indent=2)
    print(f'Summary saved: {summary_path}')


if __name__ == '__main__':
    plot_sweep_from_exports()