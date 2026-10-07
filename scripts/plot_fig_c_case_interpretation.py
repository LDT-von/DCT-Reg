#!/usr/bin/env python3
"""
Figure C: Case interpretation figure.
展示单个患者的 attention、OT matrix、pathway 参与、hazard。
数据: patients.npz + case_*.npz
"""
import sys, os
sys.path.insert(0, '/data1/DCT-Reg')
os.chdir('/data1/DCT-Reg')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram

OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
EXPORT_ROOT = Path('results/v313_interpretability_v1/exports')


def project_omic_attention(plans, stage_gate, attention_wsi, attention_omic):
    """Project omic attention through OT plans onto WSI attention.

    Args:
        plans: [S, G, Kw, Ko]
        stage_gate: [S]
        attention_wsi: [Kw, N] (Wsi_slot x patch_count)
        attention_omic: [Ko, P]
    Returns:
        H: [P, N] — projected per-pathway onto WSI patches
        Cs_mean: [Kw, Ko] — mean transport plan across stage/geometry
    """
    S, G, _, _ = plans.shape
    Kw, N = attention_wsi.shape

    # Average transport plans across geometries (per stage)
    Cs_per_stage = plans.mean(axis=1)  # [S, Kw, Ko]
    # Column-mass normalize (mass → each omic slot receives)
    col_mass = Cs_per_stage.sum(axis=1, keepdims=True)  # [S, 1, Ko]
    Bs = Cs_per_stage / (col_mass + 1e-8)  # [S, Kw, Ko]

    # Multiply WSI attention by Bs for each stage
    # We want Hs[s, n, k_o] = sum_{k_w} Bs[s, k_w, k_o] * attention_wsi[k_w, n]
    Hs = np.einsum('skw,kn->snk', Bs, attention_wsi)  # [S, N, Ko]

    # Stage gate weighted sum
    q = stage_gate / (stage_gate.sum() + 1e-8)  # [S]
    H = np.einsum('s,snk->nk', q, Hs)  # [N, Ko]

    # Pathway activation: H @ attention_omic  → [N, P]
    # Actually H [N, Ko] × attention_omic [Ko, P] = [N, P]
    final_attn = H @ attention_omic  # [N, P]

    return final_attn, Cs_per_stage


def plot_case(case_npz_path: Path, patients_npz_path: Path, output_dir: Path, case_label: str):
    """Plot case interpretation for one patient."""
    case_data = np.load(case_npz_path, allow_pickle=True)
    pat_data = np.load(patients_npz_path, allow_pickle=True)

    risk = float(case_data['risk'][0])
    survival = case_data['survival'][0]  # [4]
    plans = case_data['plans'][0]  # [S, G, Kw, Ko] = [4, 3, 8, 8]
    attention_wsi = case_data['attention_wsi'][0]  # [Kw, N] = [8, 2048]
    attention_omic = case_data['attention_omic'][0]  # [Ko, P] = [8, 329]
    stage_gate = case_data['stage_gate'][0]  # [S] = [4]
    hazard_wsi = case_data['hazard_wsi'][0]  # [Kw, B] = [8, 4]
    hazard_omic = case_data['hazard_omic'][0]  # [Ko, B] = [8, 4]
    pathway_error = case_data['self_pathway_error'][0]  # [P] = [329]
    sweep_risk = case_data['sweep_risk'][0]  # [5]

    pathway_names = list(pat_data['pathway_names'])
    time = float(pat_data['time'][int(case_npz_path.stem.split('_')[-1])])

    # Compute projection
    H, Cs_per_stage = project_omic_attention(plans, stage_gate, attention_wsi, attention_omic)
    # H: [N, P], Cs_per_stage: [S, Kw, Ko]

    # Aggregate pathway attention per omic slot
    # Use attention_omic magnitude × hazard_omic
    hazard_total = hazard_omic.sum(axis=1)  # [Ko]
    omic_slot_activation = attention_omic.sum(axis=1) * hazard_total  # [Ko]

    # Aggregate pathway activation (from projection)
    pathway_projection = H.mean(axis=0)  # [P]
    top_pathways_idx = np.argsort(pathway_projection)[::-1][:10]

    # OT matrix visualization: average across stages
    ot_matrix = plans.mean(axis=(0, 1))  # [Kw, Ko]

    # ── Figure layout ───────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.30)
    fig.suptitle(f'Case Interpretation: {case_label} (risk={risk:.3f}, time={time:.1f}m)',
                 fontsize=12, fontweight='bold', y=0.995)

    # Panel A: Survival probability per bin
    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.arange(len(survival))
    ax_a.bar(bins, survival, color='#1f77b4', alpha=0.8)
    ax_a.set_xlabel('Discrete time bin')
    ax_a.set_ylabel('Survival P')
    ax_a.set_title(f'A. Survival by bin', fontweight='bold')
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(axis='y', alpha=0.3)

    # Panel B: WSI attention per slot (sum across patches)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.bar(range(8), attention_wsi.sum(axis=1), color='#ff7f0e', alpha=0.8)
    ax_b.set_xlabel('WSI slot')
    ax_b.set_ylabel('Total WSI attention')
    ax_b.set_title(f'B. WSI slot attention', fontweight='bold')
    ax_b.grid(axis='y', alpha=0.3)

    # Panel C: Omic slot attention × hazard
    ax_c = fig.add_subplot(gs[0, 2])
    omic_attn = attention_omic.sum(axis=1)
    omic_haz = hazard_omic.sum(axis=1)
    x_pos = np.arange(8)
    ax_c.bar(x_pos - 0.2, omic_attn / (omic_attn.max() + 1e-8), width=0.4,
             color='#2ca02c', alpha=0.8, label='Attention')
    ax_c.bar(x_pos + 0.2, omic_haz / (omic_haz.max() + 1e-8), width=0.4,
             color='#d62728', alpha=0.8, label='Hazard')
    ax_c.set_xlabel('Omic slot')
    ax_c.set_ylabel('Normalized')
    ax_c.set_title(f'C. Omic slot attn × hazard', fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.grid(axis='y', alpha=0.3)

    # Panel D: Sweep risk per alpha
    ax_d = fig.add_subplot(gs[0, 3])
    alphas = np.linspace(0, 1, 5)
    ax_d.plot(alphas, sweep_risk, 'o-', linewidth=2.0, markersize=8, color='#1f77b4')
    for i, a in enumerate(alphas):
        ax_d.annotate(f'{sweep_risk[i]:.3f}', (a, sweep_risk[i]),
                      xytext=(0, 8), textcoords='offset points',
                      ha='center', fontsize=7)
    ax_d.set_xlabel('α (transport plan mix)')
    ax_d.set_ylabel('Risk score')
    ax_d.set_title(f'D. Transport sweep risk', fontweight='bold')
    ax_d.grid(alpha=0.3)

    # Panel E: Mean OT matrix
    ax_e = fig.add_subplot(gs[1, :2])
    im_e = ax_e.imshow(ot_matrix, cmap='YlOrRd', aspect='auto')
    ax_e.set_xlabel('Omic slot (Ko)')
    ax_e.set_ylabel('WSI slot (Kw)')
    ax_e.set_title(f'E. Mean OT matrix (avg over S×G)', fontweight='bold')
    plt.colorbar(im_e, ax=ax_e, shrink=0.7)
    # Annotate cells
    for i in range(8):
        for j in range(8):
            ax_e.text(j, i, f'{ot_matrix[i,j]:.2f}', ha='center', va='center',
                     fontsize=6, color='gray')

    # Panel F: Stage × Geometry OT heatmap (per stage)
    ax_f = fig.add_subplot(gs[1, 2:])
    # Show OT matrix across all stages as a strip
    ot_per_stage = plans.mean(axis=1)  # [S, Kw, Ko] = [4, 8, 8]
    # Visualize as horizontal strip: each row is a stage
    im_f = ax_f.imshow(ot_per_stage.reshape(-1, 8), cmap='YlOrRd', aspect='auto')
    ax_f.set_xlabel('Omic slot (Ko)')
    ax_f.set_ylabel('Stage × WSI slot')
    ax_f.set_title(f'F. OT per stage (mean over G)', fontweight='bold')
    n_stages = ot_per_stage.shape[0]
    ax_f.set_yticks([4 * s + 4 for s in range(n_stages)])
    ax_f.set_yticklabels([f'Stage {s}' for s in range(n_stages)])
    plt.colorbar(im_f, ax=ax_f, shrink=0.7)

    # Panel G: Top pathways by projected attention
    ax_g = fig.add_subplot(gs[2, :2])
    top_names = [pathway_names[i][:30] for i in top_pathways_idx]
    top_vals = pathway_projection[top_pathways_idx]
    top_vals_norm = top_vals / (top_vals.max() + 1e-8)
    bars = ax_g.barh(range(10), top_vals_norm, color='#9467bd', alpha=0.85)
    ax_g.set_yticks(range(10))
    ax_g.set_yticklabels(top_names, fontsize=8)
    ax_g.set_xlabel('Projected pathway attention (normalized)')
    ax_g.set_title(f'G. Top-10 pathways (via OT-projected attention)', fontweight='bold')
    ax_g.invert_yaxis()
    ax_g.grid(axis='x', alpha=0.3)

    # Panel H: Hazard across bins per modality
    ax_h = fig.add_subplot(gs[2, 2:])
    bins_range = np.arange(4)
    for s in range(8):
        ax_h.plot(bins_range, hazard_wsi[s], alpha=0.4, color='#ff7f0e', linewidth=1.0)
        ax_h.plot(bins_range, hazard_omic[s], alpha=0.4, color='#2ca02c', linewidth=1.0)
    ax_h.plot(bins_range, hazard_wsi.mean(axis=0), 'o-', color='#ff7f0e',
              linewidth=2.5, label='WSI (mean)', markersize=8)
    ax_h.plot(bins_range, hazard_omic.mean(axis=0), 's-', color='#2ca02c',
              linewidth=2.5, label='Omic (mean)', markersize=8)
    ax_h.set_xlabel('Discrete time bin')
    ax_h.set_ylabel('Hazard')
    ax_h.set_title(f'H. Per-slot hazard by bin', fontweight='bold')
    ax_h.legend(fontsize=8, loc='upper right')
    ax_h.grid(alpha=0.3)
    ax_h.set_xticks(bins_range)

    # Save
    output_path = output_dir / f'case_{case_label}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

    return output_path


def main():
    cancers = ['blca', 'kirc']

    for cancer in cancers:
        for fold in range(5):
            exp_dir = EXPORT_ROOT / f"exp6_{cancer}_fold{fold}"
            if not exp_dir.exists():
                continue
            run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
            if not run_dirs:
                continue
            run_dir = run_dirs[0]
            patients_path = run_dir / 'patients.npz'
            if not patients_path.exists():
                continue

            for case_idx in range(3):
                case_path = run_dir / f'case_{case_idx:03d}.npz'
                if not case_path.exists():
                    continue

                # Get case label
                try:
                    pat_data = np.load(patients_path, allow_pickle=True)
                    case_id = pat_data['case_ids'][case_idx]
                except Exception:
                    case_id = f'{cancer}_f{fold}_{case_idx}'

                out_dir = OUT / 'case_interpretation' / f'{cancer}_fold{fold}'
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_case(case_path, patients_path, out_dir, case_id)

    print("\nAll case interpretation figures generated.")


if __name__ == '__main__':
    main()