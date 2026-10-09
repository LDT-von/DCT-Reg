#!/usr/bin/env python3
"""
Figure D (revised): Cohort pathway attention heatmap.

- X axis: pathways (real names, top ~40 by cross-group variance)
- Y axis: patients (sorted low → high risk)
- Color: real attention_omic [N, K_o, P] → per-patient per-pathway
  attention = attention_omic.sum(axis=1) / K_o  (average across omic slots)
- Groups: split patients into quartiles; show mean attention per group

Risk convention: more negative = higher risk (closer to 0 = more severe).

Data: results/v313_interpretability_v1/exports/<arm>_<cancer>_fold<f>/<run>/patients.npz
      (attention_omic was missing from exports; after fix it will be [N, K_o=8, P=329])
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

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
EXPORT_ROOT = Path('results/v313_interpretability_v1/exports')

# ── Pathway category color bands (KEGG Hallmark) ────────────────────────────────
# Curated from MSigDB Hallmark. Add/remove as needed.
CAT_COLOR = {
    'proliferation':   '#e41a1c',
    'immune':          '#377eb8',
    'metabolism':      '#4daf4a',
    'angiogenesis':    '#984ea3',
    'differentiation':'#ff7f00',
    'viral':           '#a65628',
    'oncogenesis':     '#f781bf',
    'signaling':       '#1f77b4',
    'dna_repair':      '#8c564b',
    'stress':          '#17becf',
    'unknown':         '#999999',
}

# Simple keyword-based category assignment
def pathway_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ['cell cycle', 'e2f', 'g2m', 'mitotic', 'mitosis']):
        return 'proliferation'
    if any(k in n for k in ['immune', 'il2', 'il6', 'tnf', 'interferon', 'inflammatory', 'nk']):
        return 'immune'
    if any(k in n for k in ['glycol', 'oxidative', 'phosphorylation', 'fatty acid',
                             'cholesterol', 'heme', 'metabolism', 'tca']):
        return 'metabolism'
    if any(k in n for k in ['hypoxia', 'uv response', 'apoptosis', 'p53', 'dna repair',
                             'p53 pathway']):
        return 'stress'
    if any(k in n for k in ['notch', 'wnt', 'hedgehog', 'nfkb', 'pi3k', 'akt', 'mtor',
                             'kras', 'tgfb', 'mapk', 'egfr', 'her2', 'signaling']):
        return 'signaling'
    if any(k in n for k in ['emt', 'epithelial', 'mesenchymal', 'angiogenesis', 'angiogenic']):
        return 'oncogenesis'
    return 'unknown'


def find_run_dir(base: Path) -> Path:
    """Find the actual run subdirectory inside an export base dir."""
    for d in sorted(base.iterdir()):
        if d.is_dir():
            return d
    return base


def load_patients(base: Path):
    """Load patients.npz from a (nested) export directory.

    Returns dict with keys: risk, attention_omic, hazard_omic, pathway_names, case_ids, censor, time.
    """
    run_dir = find_run_dir(base)
    patients_path = run_dir / 'patients.npz'
    if not patients_path.exists():
        return None
    data = np.load(patients_path, allow_pickle=True)
    result = {
        'risk':            data['risk'],
        'pathway_names':  list(data['pathway_names']),
        'case_ids':       data['case_ids'],
        'censor':         data['censor'],
        'time':           data['time'],
    }
    # attention_omic may be absent in old exports; skip gracefully
    if 'attention_omic' in data.files:
        result['attention_omic'] = data['attention_omic']  # [N, K_o, P]
    else:
        result['attention_omic'] = None
    # hazard_omic [N, K_o, B] used to weight the per-slot per-pathway attention
    if 'hazard_omic' in data.files:
        result['hazard_omic'] = data['hazard_omic']
    else:
        result['hazard_omic'] = None
    return result


def build_patient_pathway_attention(attn_omic: np.ndarray,
                                    hazard_omic: np.ndarray = None) -> np.ndarray:
    """Aggregate omic slot attention into per-patient per-pathway [N, P].

    The raw `attention_omic` is post-softmax per slot, so its per-slot values
    sum to 1.0 across pathways and look almost uniform (each ≈ 1/P).  To make
    "attention" interpretable as "how much does the model attend to this
    pathway for this patient", we weight it by per-slot hazard magnitude
    (sum across time bins), so that a slot with high hazard and high
    attention on pathway P contributes strongly to patient×pathway weight.

    attention_omic: [N, K_o, P]
    hazard_omic:    [N, K_o, B] (or None to fall back to mean attention)
    Returns: [N, P]
    """
    if attn_omic is None:
        return None
    if hazard_omic is None:
        return attn_omic.mean(axis=1)  # [N, P]
    # hazard_total [N, K_o] = sum of per-bin hazard per slot
    hazard_total = hazard_omic.sum(axis=-1)  # [N, K_o]
    # weighted attention: each slot contributes (attn × hazard_total) to pathways
    weighted = attn_omic * hazard_total[:, :, None]  # [N, K_o, P]
    return weighted.sum(axis=1)  # [N, P]


def cross_group_variance(patient_attn: np.ndarray, groups: np.ndarray,
                          n_groups: int) -> np.ndarray:
    """Compute variance of mean attention across risk groups for each pathway.

    patient_attn: [N, P]
    groups: [N] int 0..n_groups-1
    Returns: [P] variance across group means.
    """
    P = patient_attn.shape[1]
    group_means = np.zeros((n_groups, P))
    for g in range(n_groups):
        mask = groups == g
        if mask.sum() > 0:
            group_means[g] = patient_attn[mask].mean(axis=0)
    return group_means.var(axis=0)


def main():
    arms    = ['exp6', 'direct', 'independent']
    cancers = ['blca', 'kirc']
    n_quartiles = 4

    # Collect all folds per (arm, cancer)
    merged = {}
    for cancer in cancers:
        for arm in arms:
            key = f'{cancer}/{arm}'
            all_attn, all_risk, all_names = [], [], None
            for fold in range(5):
                # Try several naming conventions to find the export base dir
                candidates = [
                    EXPORT_ROOT / f'{arm}_{cancer}_fold{fold}',         # exp6_blca_fold0
                    EXPORT_ROOT / f'{arm}_{cancer}_f{fold}_s3',        # exp6_blca_f0_s3 (manifest id)
                    EXPORT_ROOT / f'{cancer}_fold{fold}_{arm}',         # blca_fold0_direct
                ]
                base = None
                for c in candidates:
                    if c.exists() and (c / 'patients.npz').exists():
                        base = c
                        break
                if base is None:
                    continue
                d = load_patients(base)
                if d is None or d['attention_omic'] is None:
                    print(f'[skip] {key} fold{fold}: attention_omic not in export ({base})')
                    continue
                if all_names is None:
                    all_names = d['pathway_names']
                all_attn.append(build_patient_pathway_attention(
                    d['attention_omic'], d.get('hazard_omic')))
                all_risk.append(d['risk'])

            if not all_attn:
                merged[key] = None
                continue

            merged_attn = np.vstack(all_attn)
            merged_risk = np.concatenate(all_risk)
            merged[key] = dict(attn=merged_attn, risk=merged_risk, names=all_names)

    # ── Figure ─────────────────────────────────────────────────────────────────
    n_cancers = len(cancers)
    n_arms    = len(arms)
    fig, axes = plt.subplots(n_cancers, n_arms,
                             figsize=(6 * n_arms + 2, 4 * n_cancers + 1),
                             gridspec_kw={'wspace': 0.45, 'hspace': 0.40})

    if n_cancers == 1 and n_arms > 1:
        axes = axes.reshape(1, -1)
    if n_cancers > 1 and n_arms == 1:
        axes = axes.reshape(-1, 1)

    stats_all = {}

    for ci, cancer in enumerate(cancers):
        for ai, arm in enumerate(arms):
            ax = axes[ci, ai]
            key = f'{cancer}/{arm}'
            d = merged.get(key)
            if d is None:
                ax.text(0.5, 0.5, f'{arm}\nno data\n(re-export needed)', ha='center',
                        va='center', transform=ax.transAxes, fontsize=10,
                        color='gray', style='italic')
                ax.set_title(f'{cancer.upper()} / {arm}', fontweight='bold')
                ax.axis('off')
                continue

            attn    = d['attn']      # [N, P]
            risk    = d['risk']      # [N]
            names   = d['names']     # [P]
            N, P    = attn.shape

            # ── Sort patients low → high risk (more neg = higher risk) ────────
            sort_idx = np.argsort(risk)  # ascending: most neg (high risk) first
            attn_sorted = attn[sort_idx]
            risk_sorted = risk[sort_idx]

            # ── Quartile groups ────────────────────────────────────────────────
            q_bounds = [np.percentile(risk, q) for q in [25, 50, 75]]
            groups = np.digitize(risk_sorted, q_bounds)  # 0=high-risk Q1, 3=low-risk Q4

            # ── Normalize per-pathway to [0,1] across the cohort ─────────────
            attn_min = attn.min(axis=0, keepdims=True)
            attn_max = attn.max(axis=0, keepdims=True)
            attn_norm = (attn_sorted - attn_min) / (attn_max - attn_min + 1e-8)

            # ── Select top pathways by cross-group variance ───────────────────
            # (pathways that differ most between risk groups)
            cgv = cross_group_variance(attn, np.digitize(risk, q_bounds), n_quartiles)
            top_p = min(40, P)  # show up to 40
            top_idx = np.argsort(cgv)[-top_p:][::-1]  # descending variance

            # Subset heatmap matrix
            mat = attn_norm[:, top_idx]          # [N, top_p]
            top_names = [names[i] for i in top_idx]

            # ── Plot heatmap ──────────────────────────────────────────────────
            im = ax.imshow(mat, aspect='auto', cmap='RdYlBu_r',
                           vmin=0, vmax=1, interpolation='nearest')

            # Risk group separators
            group_starts = []
            for g in range(1, n_quartiles):
                pos = np.searchsorted(groups, g, side='left')
                ax.axhline(pos - 0.5, color='white', linewidth=1.5, alpha=0.8)
                group_starts.append(pos)
            # Show 4 quartile boundary ticks (start of each quartile)
            # but skip the very last one (N) which can collide with axhline edges
            group_labels = [f'Q{g+1}' for g in range(n_quartiles)]
            # n_quartiles ticks at group starts (Q1=0, Q2=..., Q4=group_starts[-1])
            ax.set_yticks([0] + group_starts)
            ax.set_yticklabels(group_labels, fontsize=8)

            # X tick labels: show every Nth pathway name (truncated)
            step = max(1, top_p // 12)
            ax.set_xticks(range(0, top_p, step))
            ax.set_xticklabels([top_names[i][:18] for i in range(0, top_p, step)],
                               rotation=45, ha='right', fontsize=7)

            ax.set_xlabel('Pathway (sorted by cross-group variance)', fontsize=8)
            ax.set_ylabel('Risk group\n(low → high)', fontsize=8)
            ax.set_title(f'{cancer.upper()} / {arm}\nN={N}, top {top_p} pathways',
                         fontsize=10, fontweight='bold')

            # Color bar
            cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('Norm. attention', fontsize=8)

            # Stats
            stats_all[key] = {
                'n_patients': int(N),
                'n_pathways_shown': int(top_p),
                'pathway_names': top_names,
                'cross_group_variance': cgv[top_idx].tolist(),
                'group_sizes': [int((groups == g).sum()) for g in range(n_quartiles)],
            }

    fig.suptitle('Cohort Pathway Attention Heatmap — hazard-weighted attention_omic\n'
                 'Color: per-pathway weight (Σ_slot attn × hazard); Y: patients sorted low→high risk; '
                 'separators = quartile boundaries',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.savefig(OUT / 'cohort_pathway_heatmap_v2.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {OUT / "cohort_pathway_heatmap_v2.png"}')

    json.dump(stats_all, open(OUT / 'cohort_pathway_heatmap_stats.json', 'w'), indent=2)
    print('Stats saved')


if __name__ == '__main__':
    main()
