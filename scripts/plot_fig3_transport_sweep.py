#!/usr/bin/env python3
"""
Figure 3: Transport plan replacement sweep (BLCA/KIRC × 5 folds).

Usage:
    python scripts/plot_fig3_transport_sweep.py --cancer blca --fold 0
    python scripts/plot_fig3_transport_sweep.py --cancer kirc --fold 0
    # Run all folds:
    for fold in 0 1 2 3 4; do
        python scripts/plot_fig3_transport_sweep.py --cancer blca --fold $fold &
        python scripts/plot_fig3_transport_sweep.py --cancer kirc --fold $fold &
    done
    # Then plot:
    python scripts/plot_fig3_transport_sweep.py --plot
"""
import sys, os, json
sys.path.insert(0, '/data1/DCT-Reg')
os.chdir('/data1/DCT-Reg')

import numpy as np
import torch
from pathlib import Path
from sksurv.metrics import concordance_index_censored

# ── Import from the correct paths ────────────────────────────────────────────
# Path setup FIRST: dataset/loss live in survot_rank/research/legacy/slotspe_runtime/
from survot_rank.training.paths import ensure_compat_runtime_in_path
ensure_compat_runtime_in_path()
sys.path.insert(0, '/data1/DCT-Reg')
from dataset.dataset_survival import SurvivalDatasetFactory, SurvivalDataset
from survot_rank.training.model_factory import get_model
from survot_rank.evidence.v313 import replay
# DCTArgs doesn't exist; use argparse.Namespace as the duck-typed args holder
from argparse import Namespace as DCTArgs  # noqa

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('paper/figures'); OUT.mkdir(exist_ok=True)
RESULTS_BASE = Path('results/dct_v313_ablation_Exp6_full')


# ── Dataset factory builder ───────────────────────────────────────────────────

def make_factory(cancer, fold):
    """Build SurvivalDatasetFactory matching the EXP6 training config."""
    return SurvivalDatasetFactory(
        study=cancer,
        data_path='/data1/dataset_csv',
        rna_format='Pathways',
        label_col='survival_months_dss',
        signature='combine',
        n_bins=4,
        num_patches=2048,
        binning_mode='global_qcut',
        which_splits='5fold_uni2h',
    )


def get_test_data(factory, fold):
    """Return x_wsi [N,P,D], omics_per_pathway [329 tensors of [N, genes_i]], event, time.

    The collated batch is [img, omics_per_sample, label, time, c] where
    omics_per_sample[j] is a list of 329 pathway tensors for sample j.
    We re-organise to omics_per_pathway[k] = tensor of shape [N, genes_in_k].
    """
    from dataset.dataset_survival import _collate_pathways

    wsi_path = Path('/data1/TCGA-UNI2-h-features')
    test_data = SurvivalDataset(factory, wsi_path, 'val', fold, encoding_dim=1536)
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=8, shuffle=False,
        collate_fn=_collate_pathways, num_workers=0,
    )

    all_wsi, all_cens, all_t = [], [], []
    per_pathway = []  # will be list of [N, genes_i] tensors
    n_pathways = None
    for batch in loader:
        if batch is None: continue
        all_wsi.append(batch[0])
        all_cens.append(batch[4].numpy())  # censorship (1=event censored)
        all_t.append(batch[3].numpy())     # event_time (raw months)
        # batch[1] is list-of-list: [sample][pathway]
        # First batch determines pathway count
        if n_pathways is None:
            n_pathways = len(batch[1][0])
            per_pathway = [[] for _ in range(n_pathways)]
        bsz = batch[0].size(0)
        for k in range(n_pathways):
            per_pathway[k].append(torch.stack([batch[1][j][k] for j in range(bsz)]))

    x_wsi = torch.cat(all_wsi)
    cens = np.concatenate(all_cens)
    t_train = np.concatenate(all_t)
    omics_per_pathway = [torch.cat(tensors) for tensors in per_pathway]
    return x_wsi, omics_per_pathway, cens, t_train


# ── Model builder ─────────────────────────────────────────────────────────────

def build_args(cancer):
    """Build DCTArgs matching EXP6 config."""
    a = DCTArgs()
    a.survot_method = 'dct_v313_transport_reconstruction'
    a.study = cancer
    a.data_root_dir = '/data1/TCGA-UNI2-h-features'
    a.data_path = '/data1/dataset_csv'
    a.rna_format = 'Pathways'
    a.label_col = 'survival_months_dss'
    a.signature = 'combine'
    a.n_classes = 4
    a.num_patches = 2048
    a.encoding_dim = 1536
    a.max_epochs = 30
    a.batch_size = 8
    a.seed = 3
    a.lr = 0.0005
    a.opt = 'adamW'
    a.reg = 0.0005
    a.scheduler = 'cosine'
    a.num_workers = 4
    a.bag_loss = 'nll_surv'
    a.alpha_surv = 0.15
    a.fit_bins_on_train = True
    a.binning_mode = 'global_qcut'
    a.event_stratified_batches = True
    a.slot_num_wsi = 8
    a.slot_num_omics = 8
    a.slot_iters = 3
    a.temperature = 0.01
    a.topk_ratio = 0.25
    a.top_k_method = 'parallel_topk_st'
    a.wsi_encoder = 'uni2-h'
    a.wsi_projection_dim = 256
    a.otehv2_eps = 0.05
    a.otehv2_iter = 50
    a.otehv2_heads = 4
    a.otehv2_layers = 2
    a.otehv2_dropout = 0.15
    a.dct_num_stages = 4
    a.dct_lambda_ipcw_rank = 0.10
    a.dct_ipcw_rank_margin = 0.02
    a.dct_ipcw_rank_temperature = 0.50
    a.dct_ipcw_max_weight = 10.0
    a.dct_ipcw_rank_memory_size = 64
    a.dct_anchor_momentum = 0.90
    a.dct_evidence_mass_floor = 0.05
    a.dct_evidence_marginal_strength = 1.0
    a.dct_coupling_projection_iters = 1000
    a.dct_coupling_projection_tol = 0.0001
    a.dct_coordinate_temperature = 0.30
    a.dct_mix_ratio = 1.0
    a.spt_prog_cost = 0.20
    a.rg_eps_start = 0.10
    a.rg_eps_anneal = 12
    a.which_splits = '5fold_uni2h'
    a.on_missing_wsi = 'error'
    a._dct_user_overrides = set()
    a.cur_epoch = 0
    return a


def load_model(ckpt_path, args, factory):
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    args.omic_sizes = factory.omic_sizes
    args.omic_names = factory.omic_names
    args.pathway_names = getattr(factory, 'pathway_names', None)
    model = get_model(method=args.survot_method, args=args,
                      omic_input_dim=None,
                      omic_names=args.omic_names,
                      pathway_names=args.pathway_names)
    # The buffers dct_stage_edges/dct_censor_times/dct_censor_survival are empty
    # until configure_train_reference() runs.  Allocate them at the right size
    # *before* load_state_dict so it doesn't trip on size mismatch.
    for name in ('dct_stage_edges', 'dct_censor_times', 'dct_censor_survival'):
        key = f'{name}' if hasattr(model, name) else None
    # Easier: stub via _load_from_state_dict by manually pre-allocating buffers.
    # Get expected shapes from the state dict:
    expected_stage_edges = state.get('dct_stage_edges', None)
    expected_censor_times = state.get('dct_censor_times', None)
    expected_censor_survival = state.get('dct_censor_survival', None)
    if expected_stage_edges is not None:
        # Resize the buffer in-place
        model.dct_stage_edges = torch.empty_like(expected_stage_edges)
        model.dct_censor_times = torch.empty_like(expected_censor_times)
        model.dct_censor_survival = torch.empty_like(expected_censor_survival)
    model.load_state_dict(state, strict=False)
    # The backbone's _load_from_state_dict should have done this, but
    # configure_train_reference also works.
    if not model.has_train_reference:
        pass  # will be configured below in run_fold
    model.to(DEVICE)
    model.eval()
    return model


def get_train_labels(factory, fold):
    """Get event time + censorship for the inner train set of a fold.

    These are needed to configure stage anchors (eval-time only).
    """
    from dataset.dataset_survival import _collate_pathways
    wsi_path = Path('/data1/TCGA-UNI2-h-features')
    train_data = SurvivalDataset(factory, wsi_path, 'train', fold, encoding_dim=1536)
    loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=False,
        collate_fn=_collate_pathways, num_workers=4,
    )
    times, cens = [], []
    for batch in loader:
        if batch is None: continue
        times.append(batch[3])
        cens.append(batch[4])
    return torch.cat(times).numpy(), torch.cat(cens).numpy()


# ── C-index ──────────────────────────────────────────────────────────────────

def cindex_from_risk(risk, censorship, event_time):
    """Match _calculate_metrics convention:
       event = (1 - censorship).astype(bool), time = event_time.
    ``risk`` here is a higher = worse-survival scalar (cumulative hazard
    product); concordance_index_censored wants higher = shorter time, so
    we pass risk directly (no negation).
    """
    y = np.array([(bool(1 - c), float(t)) for c, t in zip(censorship, event_time)],
                 dtype=[('e', bool), ('t', float)])
    c, _, _, _, _ = concordance_index_censored(y['e'], y['t'], risk)
    return float(c)


# ── Sweep ─────────────────────────────────────────────────────────────────────

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_fold(cancer, fold):
    """Run alpha sweep on one fold. Returns list of cindex per alpha."""
    print(f"\n  Cancer={cancer} fold={fold}")

    # Find checkpoint
    base = RESULTS_BASE / cancer / cancer / 'SurvOTRank_dct_v313_transport_reconstruction'
    sub = next(base.iterdir())
    ckpt = sub / f'model_best_s{fold}.pth'
    print(f"  ckpt: {ckpt.name}")

    # Factory + data
    factory = make_factory(cancer, fold)
    x_wsi, omics_per_pathway, cens, t_train = get_test_data(factory, fold)
    n = x_wsi.size(0)
    n_pathways = len(omics_per_pathway)
    print(f"  samples: {n}, pathways: {n_pathways}")

    # Model
    args = build_args(cancer)
    model = load_model(str(ckpt), args, factory)
    print(f"  model loaded")

    # Configure stage anchors from inner train (eval-mode requires this)
    train_t, train_c = get_train_labels(factory, fold)
    # c in dataset = 1 means censored, 0 means event; configure_train_reference wants c==0 event
    # (We pass through; verify)
    model.configure_train_reference(train_t, train_c)
    print(f"  stage anchors configured (n_train={len(train_t)})")

    # Process in batches
    BATCH = 8
    sweep_risk_chunks = []
    n_alphas = len(ALPHAS)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        x_wsi_b = x_wsi[start:end].to(DEVICE)
        payload = {'x_wsi': x_wsi_b}
        for k in range(n_pathways):
            payload[f'x_omic{k+1}'] = omics_per_pathway[k][start:end].to(DEVICE)
        result = replay(model, payload, alphas=ALPHAS)
        sweep_risk_chunks.append(result['sweep_risk'])

    sweep_risk = np.concatenate(sweep_risk_chunks, axis=0)
    assert sweep_risk.shape == (n, n_alphas), f"got {sweep_risk.shape}"

    cindices = []
    for i, alpha in enumerate(ALPHAS):
        # censorship from batch[4], event_time from batch[3]
        ci = cindex_from_risk(sweep_risk[:, i], cens, t_train)
        cindices.append(ci)
        print(f"    α={alpha:.2f}  C-index={ci:.4f}")

    # Save
    out = OUT / f'fig3_sweep_{cancer}_fold{fold}.json'
    with open(out, 'w') as f:
        json.dump({'cancer': cancer, 'fold': fold, 'n_samples': int(n),
                   'alphas': list(ALPHAS), 'cindex': cindices,
                   'risk_stats': {f'a{int(a*100):03d}': {'mean': float(sweep_risk[:,i].mean()),
                                                          'std':  float(sweep_risk[:,i].std()),
                                                          'min':  float(sweep_risk[:,i].min()),
                                                          'max':  float(sweep_risk[:,i].max()),
                                                          'unique_n': int(len(np.unique(sweep_risk[:,i])))}
                                  for i, a in enumerate(ALPHAS)}}, f)
    print(f"  saved: {out}")
    return cindices


# ── Plot ──────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_all():
    """Aggregate all folds and plot."""
    # Load all sweep results
    sweeps = {}
    for cancer in ['blca', 'kirc']:
        sweeps[cancer] = {}
        for fold in range(5):
            fpath = OUT / f'fig3_sweep_{cancer}_fold{fold}.json'
            if fpath.exists():
                with open(fpath) as f:
                    sweeps[cancer][fold] = json.load(f)

    alphas = sweeps['blca'][0]['alphas']
    n_alphas = len(alphas)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={'wspace': 0.30})
    fig.text(0.01, 0.98, 'Figure 3', fontsize=9, va='top', ha='left', color='gray')

    COLORS = {'blca': '#1f77b4', 'kirc': '#2ca02c'}
    LABEL  = {'blca': 'BLCA',    'kirc': 'KIRC'}

    for ax, cancer in zip(axes, ['blca', 'kirc']):
        fold_data = sweeps[cancer]
        if not fold_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Per-fold lines + mean
        for fold, fd in fold_data.items():
            ax.plot(range(n_alphas), fd['cindex'], 'o--', alpha=0.35,
                    color=COLORS[cancer], linewidth=0.8, markersize=4)

        # Mean ± std across folds
        ci_arr = np.array([fd['cindex'] for fd in fold_data.values()])  # [n_folds, n_alphas]
        mean_c = ci_arr.mean(axis=0)
        std_c  = ci_arr.std(axis=0)

        ax.errorbar(range(n_alphas), mean_c, yerr=std_c,
                    fmt='o-', color=COLORS[cancer], linewidth=2.2,
                    markersize=7, capsize=4, capthick=1.5,
                    label=f'{LABEL[cancer]} mean±std')

        # Annotate
        for i, (m, s) in enumerate(zip(mean_c, std_c)):
            ax.annotate(f'{m:.3f}\n±{s:.3f}',
                        xy=(i, m), xytext=(0, 14),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=7,
                        color=COLORS[cancer])

        ax.set_xticks(range(n_alphas))
        ax.set_xticklabels([f'α={a:.2f}\n{"factual" if a==0 else ("indep." if a==1 else "mixed")}' for a in alphas],
                            fontsize=8)
        ax.set_xlabel('Transport plan', fontsize=9)
        ax.set_ylabel('C-index', fontsize=9)
        ax.set_title(f'{LABEL[cancer]}', fontsize=11, fontweight='bold')
        ax.set_ylim(0.60, 0.90)
        ax.grid(axis='y', linewidth=0.4, alpha=0.4)
        ax.legend(fontsize=8, loc='lower left')

        # Summary box
        delta = mean_c[0] - mean_c[-1]
        ax.text(0.02, 0.98,
                f'Factual − Indep: Δ={delta:+.3f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.85))

    plt.suptitle('Transport Plan Replacement Sweep (v3.13 Full, BLCA/KIRC)',
                 fontsize=11, y=1.01)
    out_png = OUT / 'fig3_transport_sweep.png'
    out_pdf = OUT / 'fig3_transport_sweep.pdf'
    plt.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nFigure 3 saved: {out_png}\n                  {out_pdf}')
    print('\nPer-fold summary:')
    for cancer in ['blca', 'kirc']:
        for fold, fd in sweeps[cancer].items():
            delta = fd['cindex'][0] - fd['cindex'][-1]
            print(f'  {cancer.upper()} f{fold}: factual={fd["cindex"][0]:.4f}  indep={fd["cindex"][-1]:.4f}  Δ={delta:+.4f}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', default=None)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    if args.plot:
        plot_all()
    elif args.cancer and args.fold is not None:
        run_fold(args.cancer, args.fold)
    else:
        parser.print_help()
