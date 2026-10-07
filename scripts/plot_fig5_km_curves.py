#!/usr/bin/env python3
"""
Figure KM: Kaplan-Meier survival curves from exported predictions.
使用训练风险中位数对验证集分组，绘制 KM 曲线。

数据来源: results/v313_interpretability_v1/exports/<arm>_<cancer>_fold<f>/<run_id>/patients.npz
        + 训练风险从 export.json 加载（如果可用）
"""
import sys
import os
sys.path.insert(0, '/data1/DCT-Reg')
os.chdir('/data1/DCT-Reg')

import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
EXPORT_ROOT = Path('results/v313_interpretability_v1/exports')
CONTROLS = json.load(open('results/v313_evidence_v2/controls.json'))


def load_split_predictions(arm: str, cancer: str, fold: int) -> dict:
    """Load split_<f>_results.pkl from controls v2 (作者已经验证的 predictions)。"""
    for r in CONTROLS['runs']:
        if r['arm'] == arm and r['cancer'] == cancer and r['fold'] == fold:
            with open(r['predictions'], 'rb') as f:
                return pickle.load(f)
    return {}


def load_train_risk_from_export(arm: str, cancer: str, fold: int):
    """Try to load train risk from export.json if --km was used."""
    base = EXPORT_ROOT / f"{arm}_{cancer}_fold{fold}"
    if not base.exists():
        return None, None
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    for run_dir in run_dirs:
        export_path = run_dir / 'export.json'
        if export_path.exists():
            export = json.load(open(export_path))
            if export.get('km_train_median') is not None:
                return float(export['km_train_median']), export
    return None, None


def get_kfold_data(arm: str, cancer: str, fold: int):
    """Get (case_ids, risks, times, events) for a fold's val set from saved pkl."""
    preds = load_split_predictions(arm, cancer, fold)
    if not preds:
        return None
    case_ids = sorted(preds.keys())
    risks, times, events = [], [], []
    for cid in case_ids:
        d = preds[cid]
        if not isinstance(d, dict):
            continue
        risks.append(float(d.get('risk', 0)))
        times.append(float(d.get('time', 0)))
        events.append(int(d.get('censor', 1)))  # 0 = event, 1 = censored
    if not risks:
        return None
    return case_ids, np.array(risks), np.array(times), np.array(events)


# Model risk = -cumprod.sum, so lower (more negative) = higher hazard
# 'high risk' group: risk < median (more negative)
# 'low risk' group: risk > median

def plot_km_panel(ax, arm, full_data, cancer, fold, show_legend=True):
    """Plot one KM panel for one arm/cancer/fold."""
    if full_data is None:
        ax.text(0.5, 0.5, f'{arm}: no data', ha='center', va='center', transform=ax.transAxes)
        return None

    case_ids, risks, times, events = full_data

    # Use median of full val risks as threshold (legacy_val protocol)
    # Or train median if available
    train_median, _ = load_train_risk_from_export(arm, cancer, fold)
    threshold = train_median if train_median is not None else float(np.median(risks))

    # Lower (more negative) risk = higher hazard
    low_mask = risks > threshold
    high_mask = ~low_mask

    T_low, E_low = times[low_mask], events[low_mask]
    T_high, E_high = times[high_mask], events[high_mask]

    # Convert to lifelines format (event observed: 1=event, 0=censor)
    # In our pkl, censor=0 means event occurred
    E_low_lf = (E_low == 0).astype(int)
    E_high_lf = (E_high == 0).astype(int)

    kmf = KaplanMeierFitter()

    # Plot low risk
    if T_low.sum() > 0:
        kmf.fit(T_low, E_low_lf, label=f'Low-risk (n={low_mask.sum()}, events={E_low_lf.sum()})')
        kmf.plot_survival_function(ax=ax, color='#2ca02c', linewidth=2.0, ci_show=True, ci_alpha=0.15)

    # Plot high risk
    if T_high.sum() > 0:
        kmf.fit(T_high, E_high_lf, label=f'High-risk (n={high_mask.sum()}, events={E_high_lf.sum()})')
        kmf.plot_survival_function(ax=ax, color='#d62728', linewidth=2.0, linestyle='--', ci_show=True, ci_alpha=0.15)

    # Log-rank
    if len(T_low) > 0 and len(T_high) > 0:
        result = logrank_test(T_low, T_high, E_low_lf, E_high_lf)
        pval = result.p_value
        chi2 = result.test_statistic
    else:
        pval, chi2 = 1.0, 0.0

    # C-index
    from survot_rank.evidence.manifest import cindex
    ci = cindex(times, events, risks)

    ax.set_title(f'{cancer.upper()} Fold {fold}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (months)', fontsize=9)
    ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower left') if show_legend else ax.legend_.remove() if ax.legend_ else None

    stats_text = (
        f'n={len(risks)}\n'
        f'C-index={ci:.3f}\n'
        f'Log-rank χ²={chi2:.2f}\n'
        f'p={pval:.4f}\n'
        f'Threshold={"train" if train_median else "val"}\n'
        f'={threshold:.3f}'
    )
    props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
           fontsize=7, verticalalignment='top', horizontalalignment='right',
           bbox=props, family='monospace')

    return pval


def plot_km_for_cancer(cancer: str, arms, folds, output_path: Path):
    """Plot KM curves: rows=arms, cols=folds."""
    fig, axes = plt.subplots(len(arms), len(folds), figsize=(15, 9),
                              gridspec_kw={'wspace': 0.30, 'hspace': 0.40})
    if len(arms) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{cancer.upper()} — KM Survival Curves (Train-median risk split)',
                 fontsize=13, fontweight='bold', y=0.995)

    pval_table = []
    for i, arm in enumerate(arms):
        for j, fold in enumerate(folds):
            ax = axes[i, j]
            data = get_kfold_data(arm, cancer, fold)
            pval = plot_km_panel(ax, arm, data, cancer, fold, show_legend=(i == 0 and j == 0))
            pval_table.append({'arm': arm, 'fold': fold, 'pval': pval})

    # Row labels
    for i, arm in enumerate(arms):
        axes[i, 0].text(-0.20, 0.5, arm.upper(), transform=axes[i, 0].transAxes,
                       fontsize=11, fontweight='bold', ha='right', va='center', rotation=0)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return pval_table


def main():
    arms = ['exp6', 'direct', 'independent']  # Full + controls
    folds = [0, 1, 2, 3, 4]

    for cancer in ['blca', 'kirc']:
        output = OUT / f'km_{cancer}_per_fold.png'
        print(f"\nGenerating KM plots for {cancer.upper()}...")
        pval_table = plot_km_for_cancer(cancer, arms, folds, output)

        # Save p-values
        pval_path = OUT / f'km_{cancer}_pvalues.json'
        json.dump(pval_table, open(pval_path, 'w'), indent=2)
        print(f"  Saved: {output}")
        print(f"  P-values: {pval_path}")

    print("\nAll KM plots saved.")


if __name__ == '__main__':
    main()