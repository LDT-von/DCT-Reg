#!/usr/bin/env python3
"""
Figure D: Cohort pathway attention heatmap.
按训练风险分位数分组，对每组取组学槽 attention 平均通路值。

数据: results/v313_interpretability_v1/exports/<arm>_<cancer>_fold<f>/patients.npz
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
from matplotlib.colors import LinearSegmentedColormap

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
EXPORT_ROOT = Path('results/v313_interpretability_v1/exports')
CONTROLS = json.load(open('results/v313_evidence_v2/controls.json'))

# KEGG pathway categories for color coding
PATHWAY_CATEGORIES = {}
CAT_COLOR = {
    'proliferation': '#e41a1c',
    'immune': '#377eb8',
    'metabolism': '#4daf4a',
    'angiogenesis': '#984ea3',
    'differentiation': '#ff7f00',
    'viral': '#a65628',
    'oncogenesis': '#f781bf',
    'cancer_specific': '#999999',
    'signaling': '#1f77b4',
}


def load_pathway_attention(arm, cancer, fold):
    """Load slots_omic + pathway_names + risk from export."""
    base = EXPORT_ROOT / f"{arm}_{cancer}_fold{fold}"
    if not base.exists():
        return None
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    for rd in run_dirs:
        p = rd / 'patients.npz'
        if not p.exists():
            continue
        data = np.load(p, allow_pickle=True)
        return {
            'slots_omic': data['slots_omic'],  # [N, K_o, D]
            'risk': data['risk'],
            'pathway_names': list(data['pathway_names']),
            'self_pathway_error': data['self_pathway_error'],  # [N, P]
            'case_ids': data['case_ids'],
            'censor': data['censor'],
            'time': data['time'],
            'cross_pathway_error': data['cross_pathway_error'][:, 0, :] if 'cross_pathway_error' in data.files else None,
        }
    return None


def derive_pathway_attention_from_slots(slots_omic: np.ndarray, pathway_names: list) -> np.ndarray:
    """From slots_omic [N, K_o, D], compute [N, P] pathway attention.

    Strategy: each patient has K_o=8 omics slots. The pathway attention is the
    magnitude of pathway representation vs pathway baseline. Use slot diversity
    as proxy: variance across slots = path activation.
    Actually use: |slots_omic| averaged across slots → per-patient pathway weights.
    """
    N, K, D = slots_omic.shape
    P = len(pathway_names)

    # Each patient has K slots; project slots through pathway_names
    # We don't have pathway-membership mapping in slots. Use the cross_pathway_error
    # at alpha=0 (factorial): high error = high pathway involvement
    return None


def main():
    arms = ['exp6', 'direct', 'independent']
    cancers = ['blca', 'kirc']

    # Load self_pathway_error (per-patient per-pathway error magnitude)
    # High error = high pathway contribution = high attention

    fig, axes = plt.subplots(len(cancers), len(arms),
                            figsize=(18, 10),
                            gridspec_kw={'wspace': 0.30, 'hspace': 0.40})
    fig.suptitle('Cohort Pathway Attention Heatmap — v3.13 (BLCA/KIRC × Full/Direct/Independent)',
                 fontsize=13, fontweight='bold', y=0.995)

    all_stats = {}

    for ci, cancer in enumerate(cancers):
        for ai, arm in enumerate(arms):
            ax = axes[ci, ai]
            all_pathway_error = []  # list of arrays [P] per patient
            all_risks = []
            all_groups = []

            for fold in range(5):
                data = load_pathway_attention(arm, cancer, fold)
                if data is None:
                    continue

                # Use self_pathway_error as proxy for pathway activation
                # [N, P] per patient
                err = data['self_pathway_error']  # [N, P]
                risks = data['risk']

                # Group by quartile
                q25, q50, q75 = np.percentile(risks, [25, 50, 75])
                # Risk convention: more negative = higher hazard
                # So Q1 (highest risk) = risks <= q25
                groups = np.digitize(risks, [q25, q50, q75])
                # groups: 0=q1, 1=q2, 2=q3, 3=q4
                # But we want low risk = high values
                # So invert: Q4 (lowest risk) = highest risk group label

                # Rescale error to [0,1] per-pathway (normalize across cohort)
                err_norm = (err - err.min(axis=0)) / (err.max(axis=0) - err.min(axis=0) + 1e-8)

                all_pathway_error.append(err_norm)
                all_risks.append(risks)
                all_groups.append(groups)

            if not all_pathway_error:
                ax.text(0.5, 0.5, f'{arm}: no data', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            all_pathway_error = np.vstack(all_pathway_error)
            all_risks = np.concatenate(all_risks)
            all_groups = np.concatenate(all_groups)

            # Aggregate per group: mean error per pathway
            P = all_pathway_error.shape[1]
            n_groups = 4
            mean_attn = np.zeros((n_groups, P))
            counts = np.zeros(n_groups)
            for g in range(n_groups):
                mask = all_groups == g
                if mask.sum() > 0:
                    mean_attn[g] = all_pathway_error[mask].mean(axis=0)
                    counts[g] = mask.sum()

            # Normalize per-row for visualization
            row_max = mean_attn.max(axis=1, keepdims=True)
            mean_attn_norm = mean_attn / (row_max + 1e-8)

            # Plot heatmap
            im = ax.imshow(mean_attn_norm, aspect='auto', cmap='YlOrRd',
                          vmin=0, vmax=1)

            # Pick top pathways by variance
            pathway_var = mean_attn.var(axis=0)
            top_p_idx = np.argsort(pathway_var)[-30:][::-1]

            # Annotate
            for i, p_idx in enumerate(top_p_idx):
                ax.text(i, 3.5, '*', ha='center', va='center', fontsize=6, color='gray')

            ax.set_yticks(range(n_groups))
            ax.set_yticklabels([f'Q{i+1} (n={int(counts[i])})' for i in range(n_groups)],
                              fontsize=8)
            ax.set_xlabel(f'Pathway (P={P})', fontsize=9)
            ax.set_ylabel('Risk quartile', fontsize=9)
            ax.set_title(f'{cancer.upper()} {arm}', fontsize=10, fontweight='bold')

            # Stats
            stats_text = (
                f'Total patients: {int(counts.sum())}\n'
                f'Q1 (high-risk) n={int(counts[0])}\n'
                f'Q4 (low-risk) n={int(counts[3])}'
            )
            ax.text(1.02, 0.5, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='center',
                   family='monospace')

            plt.colorbar(im, ax=ax, label='Normalized pathway error', shrink=0.6)

            all_stats[f'{cancer}/{arm}'] = {
                'mean_attention_per_group_per_pathway': mean_attn.tolist(),
                'group_sizes': counts.tolist(),
                'n_pathways': int(P),
                'n_patients': int(counts.sum()),
            }

    plt.savefig(OUT / 'cohort_pathway_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT / 'cohort_pathway_heatmap.png'}")

    json.dump(all_stats, open(OUT / 'cohort_pathway_stats.json', 'w'), indent=2)
    print(f"Stats saved")


if __name__ == '__main__':
    main()