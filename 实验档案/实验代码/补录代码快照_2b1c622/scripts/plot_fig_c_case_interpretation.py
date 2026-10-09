#!/usr/bin/env python3
"""
Figure C (revised): Case interpretation figure.

Shows for a single patient:
  A. Survival probability per time bin
  B. WSI slot attention (sum across patches)
  C. Omic slot attention × hazard
  D. Transport sweep risk vs alpha
  E. Mean OT matrix (WSI slots → Omic slots)
  F. OT per stage (mean over geometry)
  G. Top-20 pathways by real attention_omic attention (with real names)
  H. Per-slot hazard by bin

Data: results/v313_interpretability_v1/exports/<arm>_<cancer>_fold<f>/<run>/
      case_*.npz + patients.npz
"""
import sys, os
sys.path.insert(0, '/data1/DCT-Reg')
os.chdir('/data1/DCT-Reg')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
EXPORT_ROOT = Path('results/v313_interpretability_v1/exports')


def find_run_dir(base: Path) -> Path:
    """Find the actual run subdirectory inside an export base dir."""
    for d in sorted(base.iterdir()):
        if d.is_dir():
            return d
    return base


def load_case_and_patients(case_path: Path, base: Path):
    """Load case npz and corresponding patients.npz (for pathway_names)."""
    case_data = np.load(case_path, allow_pickle=True)
    run_dir = find_run_dir(base)
    pat_data = np.load(run_dir / 'patients.npz', allow_pickle=True)
    return case_data, pat_data


def project_omic_attention(plans: np.ndarray, stage_gate: np.ndarray,
                           attention_wsi: np.ndarray, attention_omic: np.ndarray):
    """Project omic attention through OT plans onto WSI attention.

    Args:
        plans: [1, S, G, Kw, Ko]  — S stages, G geometries
        stage_gate: [1, S]
        attention_wsi: [1, Kw, N_patch]
        attention_omic: [1, Ko, P]  — attention from omic slots to pathways
    Returns:
        pathway_attn: [P] — OT-projected pathway attention (per pathway)
    """
    plans = plans[0]          # [S, G, Kw, Ko]
    stage_gate = stage_gate[0]  # [S]

    # Mean over geometries
    Cs_mean = plans.mean(axis=1)  # [S, Kw, Ko]

    # Column-mass normalize: each omic slot receives unit mass
    col_mass = Cs_mean.sum(axis=1, keepdims=True)  # [S, 1, Ko]
    Bs = Cs_mean / (col_mass + 1e-8)               # [S, Kw, Ko]

    # Weight WSI attention by OT coupling, then sum over slots
    # WSI_attn [Kw, N]; Bs [S, Kw, Ko]
    # Project: H[n, k_o] = sum_kw Bs[s, kw, ko] * attn_wsi[kw, n]
    H = np.einsum('skw,kn->snk', Bs, attention_wsi[0])  # [S, N, Ko]
    # Stage-gate weighted sum
    q = stage_gate / (stage_gate.sum() + 1e-8)   # [S]
    H_avg = np.einsum('s,snk->nk', q, H)          # [N, Ko]
    # Final: attention to pathways via omic slots
    pathway_attn = H_avg @ attention_omic[0]       # [N, P] → mean over patches → [P]
    return pathway_attn.mean(axis=0)


def plot_case(case_path: Path, export_base: Path, output_dir: Path, case_label: str):
    """Plot case interpretation for one patient."""
    case_data, pat_data = load_case_and_patients(case_path, export_base)

    risk     = float(case_data['risk'][0])
    survival = case_data['survival'][0]     # [4]
    plans    = case_data['plans'][0]        # [S, G, Kw, Ko]
    attn_wsi = case_data['attention_wsi'][0]  # [Kw, N_patch]
    attn_omic = case_data['attention_omic'][0] # [Ko, P]
    stage_gate = case_data['stage_gate'][0] # [S]
    hazard_wsi = case_data['hazard_wsi'][0] # [Kw, B]
    hazard_omic = case_data['hazard_omic'][0] # [Ko, B]
    sweep_risk = case_data['sweep_risk'][0]  # [5]

    Kw, N_patch = attn_wsi.shape      # e.g. [8, 2048]
    Ko, P = attn_omic.shape          # e.g. [8, 329]

    pathway_names = list(pat_data['pathway_names'])

    # OT matrix: mean over stages and geometries
    ot_matrix = plans.mean(axis=(0, 1))  # [Kw, Ko]

    # OT-projected pathway attention
    pathway_attn_proj = project_omic_attention(
        case_data['plans'], case_data['stage_gate'],
        case_data['attention_wsi'], case_data['attention_omic']
    )

    # Direct pathway attention (from attention_omic alone, no OT)
    pathway_attn_direct = attn_omic.sum(axis=0)  # [P] — sum over omic slots

    # Top pathways (combined: 70% OT projection, 30% direct)
    alpha = 0.7
    pathway_attn_combined = alpha * pathway_attn_proj + (1 - alpha) * pathway_attn_direct
    top_idx = np.argsort(pathway_attn_combined)[::-1][:20]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.35)
    fig.suptitle(f'Case Interpretation: {case_label}  |  risk={risk:.4f}  '
                 f'(more negative = higher risk)',
                 fontsize=12, fontweight='bold', y=0.99)

    # Panel A: Survival probability per bin
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(len(survival)), survival, color='#1f77b4', alpha=0.85, width=0.7)
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Survival P')
    ax.set_title('A. Survival by bin', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: WSI slot attention (sum across patches)
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(np.arange(Kw), attn_wsi.sum(axis=1), color='#ff7f0e', alpha=0.85, width=0.6)
    ax.set_xlabel('WSI slot')
    ax.set_ylabel('Total WSI attention')
    ax.set_title(f'B. WSI slot attention\n(Kw={Kw}, patches={N_patch})', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(Kw))

    # Panel C: Omic slot attention × hazard
    ax = fig.add_subplot(gs[0, 2])
    omic_attn = attn_omic.sum(axis=1)           # [Ko]
    omic_haz  = hazard_omic.sum(axis=1)         # [Ko]
    x = np.arange(Ko)
    w = 0.35
    ax.bar(x - w, omic_attn / (omic_attn.max() + 1e-8), width=w, color='#2ca02c',
           alpha=0.85, label='Attention')
    ax.bar(x + w, omic_haz  / (omic_haz.max()  + 1e-8), width=w, color='#d62728',
           alpha=0.85, label='Hazard')
    ax.set_xlabel('Omic slot')
    ax.set_ylabel('Normalized')
    ax.set_title(f'C. Omic slot attn × hazard\n(Ko={Ko}, P={P} pathways)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(Ko))

    # Panel D: Transport sweep risk
    ax = fig.add_subplot(gs[0, 3])
    alphas = np.linspace(0, 1, len(sweep_risk))
    ax.plot(alphas, sweep_risk, 'o-', color='#9467bd', linewidth=2.5, markersize=8)
    for i, (a, r) in enumerate(zip(alphas, sweep_risk)):
        ax.annotate(f'{r:.3f}', (a, r), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='α=0 (factual)')
    ax.set_xlabel('α (transport plan mix)')
    ax.set_ylabel('Risk score')
    ax.set_title('D. Transport sweep risk', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel E: Mean OT matrix
    ax = fig.add_subplot(gs[1, :2])
    vmax = ot_matrix.max()
    im = ax.imshow(ot_matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=vmax if vmax > 0 else 1)
    ax.set_xlabel('Omic slot (Ko)')
    ax.set_ylabel('WSI slot (Kw)')
    ax.set_title('E. Mean OT matrix (avg over stages × geometries)', fontweight='bold')
    ax.set_xticks(range(Ko))
    ax.set_yticks(range(Kw))
    plt.colorbar(im, ax=ax, shrink=0.75, label='Transport mass')
    for i in range(Kw):
        for j in range(Ko):
            ax.text(j, i, f'{ot_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color='gray' if ot_matrix[i, j] < vmax * 0.5 else 'black')

    # Panel F: OT per stage (mean over geometries)
    ax = fig.add_subplot(gs[1, 2:])
    S = plans.shape[0]
    ot_per_stage = plans.mean(axis=1)               # [S, Kw, Ko]
    mat = ot_per_stage.reshape(S, -1)               # [S, Kw*Ko]
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel(f'WSI slot × Omic slot (Kw×Ko={Kw*Ko})')
    ax.set_ylabel('Stage')
    ax.set_title(f'F. OT per stage (mean over G={plans.shape[1]} geometries)',
                 fontweight='bold')
    n_stage_ticks = S
    ax.set_yticks(range(n_stage_ticks))
    ax.set_yticklabels([f'Stage {s}' for s in range(S)], fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.75, label='Transport mass')

    # Panel G: Top-20 pathways (with real names)
    ax = fig.add_subplot(gs[2, :])
    top_vals    = pathway_attn_combined[top_idx]
    top_names   = [pathway_names[i] for i in top_idx]
    top_vals_n  = top_vals / (top_vals.max() + 1e-8)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_vals_n)))
    bars = ax.barh(range(len(top_vals_n)), top_vals_n, color=colors, alpha=0.9,
                   height=0.75)
    ax.set_yticks(range(len(top_vals_n)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel('Combined pathway attention (OT-proj × direct, normalized)')
    ax.set_title(f'G. Top-20 pathways by combined attention', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.1)
    for bar, val in zip(bars, top_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=7)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = output_dir / f'case_{case_label}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')
    return output_path


def main():
    arms    = ['exp6', 'direct', 'independent']
    cancers = ['blca', 'kirc']
    cases_per_combo = 2  # number of case_*.npz to plot per (arm, cancer, fold)

    all_outputs = []
    for cancer in cancers:
        for arm in arms:
            for fold in range(5):
                base = EXPORT_ROOT / f'{arm}_{cancer}_fold{fold}'
                if not base.exists():
                    continue
                run_dir = find_run_dir(base)
                patients_path = run_dir / 'patients.npz'
                if not patients_path.exists():
                    continue

                # Load patients for case_id lookup
                try:
                    pat_data = np.load(patients_path, allow_pickle=True)
                    case_ids = pat_data['case_ids']
                    pathway_names = list(pat_data['pathway_names'])
                except Exception as e:
                    print(f'[skip] {arm}/{cancer} fold{fold}: {e}')
                    continue

                # Find case_*.npz files
                case_files = sorted(run_dir.glob('case_*.npz'))
                if not case_files:
                    continue

                out_dir = OUT / 'case_interpretation_v2' / f'{cancer}_{arm}_fold{fold}'
                out_dir.mkdir(parents=True, exist_ok=True)

                for case_path in case_files[:cases_per_combo]:
                    case_idx = int(case_path.stem.split('_')[-1])
                    case_id = str(case_ids[case_idx]) if case_idx < len(case_ids) else f'{cancer}_f{fold}_{case_idx}'
                    try:
                        plot_case(case_path, base, out_dir, case_id)
                        all_outputs.append(str(out_dir / f'case_{case_id}.png'))
                    except Exception as e:
                        print(f'[ERROR] {case_path}: {e}')

    print(f'\nGenerated {len(all_outputs)} case interpretation figures.')
    for o in all_outputs:
        print(f'  {o}')


if __name__ == '__main__':
    main()
