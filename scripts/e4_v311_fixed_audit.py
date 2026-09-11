#!/usr/bin/env python3
"""E4: Continuous Intervention Audit for v3.11 Fixed — Optimized Version.

Strategy:
- Run full forward ONCE per sample → caches slots_wsi, slots_omic, factual_plans
- Then for each alpha, swap stage_embedding and re-run ONLY event_encoder + event_hazard + event_gate + _risk
- This makes each re-run ~10x faster (no Sinkhorn, no slot attention)

Usage:
    python scripts/e4_v311_fixed_audit.py --folds 0 --output results/e4_v311_fixed_audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO))

from survot_rank.training.paths import ensure_compat_runtime_in_path
ensure_compat_runtime_in_path()

from survot_rank.config import load_config, flatten_config
from survot_rank.training.model_factory import get_model
from dataset.dataset_survival import SurvivalDatasetFactory, _collate_pathways, SurvivalDataset
from torch.utils.data import DataLoader
from argparse import Namespace

STUDY = "blca"
CONFIG_PATH = REPO / "configs/dct_v311_blca_uni.yaml"
CKPT_ROOT = REPO / "results/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable"
SPLIT_DIR = Path("/data1/dataset_csv/splits/5fold/blca")
DATA_ROOT = Path("/data/CPathPatchFeature")


# ---------------- Model loading ----------------
def find_ckpt(fold: int) -> Path:
    cands = sorted(CKPT_ROOT.glob(f"*_dct_v311_blca_uni_fold{fold}/model_best_s{fold}.pth"))
    if not cands:
        raise FileNotFoundError(f"No ckpt for fold {fold}")
    return cands[0]


def load_model(fold: int):
    config = flatten_config(load_config(str(CONFIG_PATH)))
    config['study'] = STUDY
    config['k_start'] = fold
    config['k_end'] = fold + 1
    defaults = {
        'wsi_projection_dim': 256, 'k': 5,
        'method': 'dct_v311_slot_interpretable',
        'use_pathway_data': config.get('rna_format') == 'Pathways',
        'num_genes': None, 'wsi_encoder': 'uni',
        'encoding_dim': 256, 'num_patches': 4096,
        'on_missing_wsi': 'error', 'batch_size': 1,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)
    args = Namespace(**config)

    factory = SurvivalDatasetFactory(
        study=args.study, data_path=args.data_path, rna_format=args.rna_format,
        signature=args.signature, n_bins=args.n_classes, label_col=args.label_col,
        num_genes=getattr(args, 'num_genes', None),
        num_patches=getattr(args, 'num_patches', 2048),
        clinical_feature_cols=None,
        binning_mode=getattr(args, 'binning_mode', 'global_qcut'),
        which_splits=getattr(args, 'which_splits', '5fold'),
    )
    omics_input_dim = sum(factory.omic_sizes)
    args.omic_sizes = factory.omic_sizes
    args.omic_names = factory.omic_names
    args.pathway_names = getattr(factory, 'pathway_names', None)

    model = get_model(
        method=args.survot_method, args=args,
        omic_input_dim=omics_input_dim,
        omic_names=args.omic_names, pathway_names=args.pathway_names,
    )
    ckpt = torch.load(str(find_ckpt(fold)), map_location="cpu", weights_only=False)
    msd = model.state_dict()
    # Direct filter: key in model state and shape matches
    filtered = {}
    for k, v in ckpt.items():
        if k in msd and hasattr(v, 'shape') and msd[k].shape == v.shape:
            filtered[k] = v
    n_skipped = len(ckpt) - len(filtered)
    if n_skipped:
        print(f"  Skipped {n_skipped}/{len(ckpt)} ckpt keys (shape mismatch or not in model)")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, factory, args


def build_loader(factory, args, fold):
    wsi_path = DATA_ROOT / STUDY / args.wsi_encoder / 'pt_files'
    test_ds = SurvivalDataset(factory, str(wsi_path), 'val', fold, args.encoding_dim, on_missing_wsi='error')
    collate = _collate_pathways if args.use_pathway_data else None
    return DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)


# ---------------- Audit core ----------------
def forward_once(model, batch):
    """Run one full forward pass to fill caches and return key intermediate values."""
    if len(batch) == 6:
        wsi, genes, label, event_time, censorship, _ = batch
    else:
        wsi, genes, label, event_time, censorship = batch

    kwargs = {'x_wsi': wsi, 'event_time': event_time, 'c': censorship, 'cur_epoch': 0}
    if isinstance(genes, list):
        flat = []
        if len(genes) > 0 and isinstance(genes[0], list):
            for s in genes:
                flat.extend(s)
        else:
            flat = list(genes)
        for idx, g in enumerate(flat, start=1):
            if g.dim() == 1:
                g = g.unsqueeze(0)
            kwargs[f'x_omic{idx}'] = g
    else:
        kwargs['x_omics'] = genes

    with torch.no_grad():
        out = model(**kwargs)
        if isinstance(out, dict):
            logits = out.get('logits') or out.get('hazard_logits')
        else:
            logits = out

    return {
        'logits': logits,
        'event_time': event_time,
        'censorship': censorship,
        'batch_size': wsi.shape[0],
    }


@torch.no_grad()
def rerisk_fast(model, batch, stage_emb_override):
    """Re-risk with overridden stage_embedding.

    Tries cached fast path first (re-runs only event part).
    Falls back to full forward if cache is unavailable.
    """
    # Fast path: cache available
    plans = getattr(model, '_factual_plan_cache', None)
    slots_wsi = getattr(model, '_last_slots_wsi', None)
    slots_omic = getattr(model, '_last_slots_omic', None)
    if plans is not None and slots_wsi is not None:
        original = model.stage_embedding
        model.stage_embedding = stage_emb_override.clone()
        try:
            logits, _ = model._encode_logits_from_plans(slots_wsi, slots_omic, plans)
            risk = model._risk(logits)
        finally:
            model.stage_embedding = original
        return risk

    # Slow fallback: full forward
    if len(batch) == 6:
        wsi, genes, label, event_time, censorship, _ = batch
    else:
        wsi, genes, label, event_time, censorship = batch

    kwargs = {'x_wsi': wsi, 'event_time': event_time, 'c': censorship, 'cur_epoch': 0}
    if isinstance(genes, list):
        flat = []
        if len(genes) > 0 and isinstance(genes[0], list):
            for s in genes:
                flat.extend(s)
        else:
            flat = list(genes)
        for idx, g in enumerate(flat, start=1):
            if g.dim() == 1:
                g = g.unsqueeze(0)
            kwargs[f'x_omic{idx}'] = g
    else:
        kwargs['x_omics'] = genes

    original = model.stage_embedding
    model.stage_embedding = stage_emb_override.clone()
    try:
        out = model(**kwargs)
        if isinstance(out, dict):
            logits = out.get('logits') or out.get('hazard_logits')
        else:
            logits = out
        risk = model._risk(logits)
    finally:
        model.stage_embedding = original
    return risk


def compute_monotone_rate(df, direction):
    sub = df[df['direction'] == direction]
    n_pat = sub['patient_id'].nunique()
    if n_pat == 0:
        return 0.0
    n_mono = 0
    for pid, grp in sub.groupby('patient_id'):
        risks = grp.sort_values('alpha')['risk_pred'].values
        if direction == 'low_risk':
            is_mono = all(risks[i] >= risks[i+1] for i in range(len(risks)-1))
        else:
            is_mono = all(risks[i] <= risks[i+1] for i in range(len(risks)-1))
        if is_mono:
            n_mono += 1
    return n_mono / n_pat


@torch.no_grad()
def run_fold(fold, alphas, output_dir):
    print(f"\n{'='*50}\nFOLD {fold}\n{'='*50}")
    model, factory, args = load_model(fold)
    loader = build_loader(factory, args, fold)

    # Get anchors from stage_embedding
    low_anchor = model.stage_embedding[0].clone()       # [dim]
    high_anchor = model.stage_embedding[-1].clone()    # [dim]
    orig_emb = model.stage_embedding.clone()             # [4, dim]
    anchor_dist = torch.norm(high_anchor - low_anchor).item()
    print(f"  Anchor dist: {anchor_dist:.4f}  stage_emb: {orig_emb.shape}")

    # Enable dct_fixed_coupling so caches get populated on each forward
    prev_fixed_coupling = model.dct_fixed_coupling
    model.dct_fixed_coupling = True

    rows = []
    n_ok, n_fail = 0, 0
    for batch_idx, batch in enumerate(loader):
        # 1. Full forward → fills caches
        info = forward_once(model, batch)
        bs = info['batch_size']

        # 2. Baseline risk (alpha=0)
        baseline = rerisk_fast(model, batch, orig_emb)
        baseline = baseline.cpu()

        # 3. Loop alphas
        for alpha in alphas:
            for direction, target in [('low_risk', low_anchor), ('high_risk', high_anchor)]:
                override = (1 - alpha) * orig_emb + alpha * target.unsqueeze(0)
                risk = rerisk_fast(model, batch, override)
                risk = risk.cpu()

                for i in range(bs):
                    if len(batch) >= 5:
                        bt = batch[3] if len(batch) == 5 else batch[3]
                        bc = batch[4] if len(batch) == 5 else batch[4]
                    else:
                        bt, bc = batch[3], batch[4]
                    rows.append({
                        'patient_id': f'p{batch_idx}_{i}',
                        'fold': fold,
                        'true_time': float(bt[i].item()),
                        'true_event': int((1 - bc[i]).item()),
                        'baseline_risk': float(baseline[i].item()),
                        'alpha': float(alpha),
                        'direction': direction,
                        'risk_pred': float(risk[i].item()),
                        'risk_change': float(risk[i].item() - baseline[i].item()),
                    })
        n_ok += bs

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx+1}/{len(loader)} samples...")

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"e4_v311_fixed_fold{fold}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows to {csv_path.name}")

    mr_low = compute_monotone_rate(df, 'low_risk')
    mr_high = compute_monotone_rate(df, 'high_risk')
    mr_mean = (mr_low + mr_high) / 2
    print(f"  Monotone rate: low={mr_low:.4f}  high={mr_high:.4f}  mean={mr_mean:.4f}")

    summary = {
        'fold': fold, 'n_samples': n_ok, 'n_failed': n_fail,
        'anchor_distance': anchor_dist,
        'mr_low': mr_low, 'mr_high': mr_high, 'mr_mean': mr_mean,
    }
    with open(output_dir / f"e4_v311_fixed_fold{fold}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Restore original dct_fixed_coupling setting
    model.dct_fixed_coupling = prev_fixed_coupling
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--output", default="results/e4_v311_fixed_audit")
    args_parsed = parser.parse_args()

    folds = [int(x) for x in args_parsed.folds.split(",")]
    alphas = [float(x) for x in args_parsed.alphas.split(",")]
    output_dir = REPO / args_parsed.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E4 v3.11 Fixed Intervention Audit")
    print(f"Folds: {folds}  |  Alphas: {len(alphas)}  |  Output: {output_dir}")
    print("=" * 60)

    summaries = []
    for fold in folds:
        s = run_fold(fold, alphas, output_dir)
        summaries.append(s)

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(output_dir / "e4_v311_fixed_summary.csv", index=False)

    mean_mr = float(df_summary['mr_mean'].mean())
    print(f"\n{'='*60}")
    print(f"5-FOLD SUMMARY (v3.11 Fixed)")
    print(df_summary[['fold','n_samples','mr_low','mr_high','mr_mean']].to_string(index=False))
    print(f"\nMean monotone_rate (v3.11 fixed): {mean_mr:.4f}")
    print(f"Baseline (v3.10):                  0.2258")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
