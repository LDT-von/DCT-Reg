#!/usr/bin/env python3
"""E4: Continuous Intervention Audit for DCT-Reg - Production Version.

This experiment directly tests the core claim of DCT-Reg:
"Directionally consistent risk response to prognostic ground-cost interventions"

For each test patient, we manually interpolate their embedding towards the
low-risk and high-risk anchors at different intervention strengths α ∈ [0, 1],
then measure how the predicted risk changes.

Expected outcome:
- If α↑ (moving towards low-risk anchor) → risk↓ (monotonic decrease)
- This would directly prove the direction constraint is working

Usage:
    python scripts/e4_continuous_intervention_audit_v2.py \
        --checkpoint_dir results/dct_v3.10_experiments/robust/direction_only/blca/.../
        --config configs/dct_v310_directional_regularized_transport.yaml \
        --variant direction_only \
        --study blca \
        --fold 0 \
        --output results/e4_intervention_audit/direction_only_blca_fold0.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SurvOT-Rank components
from survot_rank.training.paths import ensure_slotspe_in_path
ensure_slotspe_in_path()

from dataset.dataset_survival import SurvivalDatasetFactory, _collate_pathways
from survot_rank.training.model_factory import get_model
from survot_rank.config import load_config


def interpolate_towards_anchor(
    embedding: torch.Tensor,
    anchor: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """Interpolate embedding towards an anchor.
    
    Args:
        embedding: Original embedding (batch_size, dim)
        anchor: Target anchor (dim,)
        alpha: Interpolation strength [0, 1]
               0 = original, 1 = fully at anchor
    
    Returns:
        Interpolated embedding (batch_size, dim)
    """
    return (1 - alpha) * embedding + alpha * anchor.unsqueeze(0)


def compute_risk_prediction(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute risk prediction from embedding.
    
    Args:
        model: Trained DCT-Reg model
        embedding: Patient embedding (batch_size, dim)
        device: Compute device
    
    Returns:
        Risk scores (batch_size,) - higher = worse prognosis
    """
    embedding = embedding.to(device)
    
    with torch.no_grad():
        # Forward through decoder to get survival prediction
        if hasattr(model, 'decoder'):
            logits = model.decoder(embedding)
        elif hasattr(model, 'risk_predictor'):
            logits = model.risk_predictor(embedding)
        else:
            # DCT models typically have a classifier
            if hasattr(model, 'classifier'):
                logits = model.classifier(embedding)
            else:
                raise AttributeError(
                    "Model has no decoder, risk_predictor, or classifier. "
                    f"Available attributes: {dir(model)}"
                )
        
        # Convert to risk score (higher = worse prognosis)
        if logits.dim() == 2:  # Multi-bin output (NLL survival)
            # Use negative expected survival time as risk
            bin_centers = torch.arange(logits.size(1), device=device).float()
            probs = torch.softmax(logits, dim=1)
            expected_time = torch.sum(probs * bin_centers, dim=1)
            risk = -expected_time  # Negate so higher risk = worse prognosis
        else:  # Single output (Cox, etc.)
            risk = logits.squeeze()
    
    return risk


def extract_prognostic_anchors(
    model: torch.nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract learned prognostic anchors from DCT-Reg model.
    
    Returns:
        low_risk_anchor: Embedding for low-risk prototype (dim,)
        high_risk_anchor: Embedding for high-risk prototype (dim,)
    """
    # Check various possible anchor storage locations in DCT models
    if hasattr(model, 'low_risk_anchor') and hasattr(model, 'high_risk_anchor'):
        low_risk_anchor = model.low_risk_anchor
        high_risk_anchor = model.high_risk_anchor
    elif hasattr(model, 'anchors'):
        # Assume anchors is [low, high] or [high, low]
        # Need to check which is which based on learned risk
        anchors = model.anchors
        if anchors.size(0) >= 2:
            # Use first two anchors
            low_risk_anchor = anchors[0]
            high_risk_anchor = anchors[1]
        else:
            raise ValueError(f"Expected at least 2 anchors, got {anchors.size(0)}")
    elif hasattr(model, 'shared_wsi_prototypes'):
        # DCT models use shared_wsi_prototypes as anchors
        prototypes = model.shared_wsi_prototypes
        print(f"  Found shared_wsi_prototypes with shape {prototypes.shape}")
        # Typically has shape [num_stages, dim] where stages are ordered by risk
        # First stage = low risk, last stage = high risk
        if prototypes.size(0) >= 2:
            low_risk_anchor = prototypes[0]  # First stage (lowest risk)
            high_risk_anchor = prototypes[-1]  # Last stage (highest risk)
        else:
            raise ValueError(f"Expected at least 2 prototypes, got {prototypes.size(0)}")
    elif hasattr(model, 'dct_module'):
        # Nested DCT module
        dct_mod = model.dct_module
        if hasattr(dct_mod, 'anchors'):
            anchors = dct_mod.anchors
            low_risk_anchor = anchors[0]
            high_risk_anchor = anchors[1]
        elif hasattr(dct_mod, 'low_risk_anchor'):
            low_risk_anchor = dct_mod.low_risk_anchor
            high_risk_anchor = dct_mod.high_risk_anchor
        elif hasattr(dct_mod, 'shared_wsi_prototypes'):
            prototypes = dct_mod.shared_wsi_prototypes
            low_risk_anchor = prototypes[0]
            high_risk_anchor = prototypes[-1]
        else:
            raise AttributeError(
                f"dct_module exists but has no anchors. Available: {dir(dct_mod)}"
            )
    elif hasattr(model, 'reference_embeddings'):
        # Alternative naming
        ref_emb = model.reference_embeddings
        low_risk_anchor = ref_emb[0]
        high_risk_anchor = ref_emb[1]
    else:
        raise AttributeError(
            "Cannot find prognostic anchors in model. "
            f"Tried: low_risk_anchor, anchors, shared_wsi_prototypes, dct_module.anchors, reference_embeddings. "
            f"Model attributes: {[a for a in dir(model) if not a.startswith('_')]}"
        )
    
    return low_risk_anchor.to(device), high_risk_anchor.to(device)


def load_model_and_data(
    checkpoint_path: str,
    config_path: str,
    study: str,
    fold: int,
    device: torch.device
):
    """Load trained model and test dataset.
    
    Returns:
        model: Loaded model
        test_loader: Test data loader
        dataset_factory: Dataset factory for metadata
    """
    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Flatten nested config
    from survot_rank.config import flatten_config
    config = flatten_config(config)
    
    # Override study and fold
    config['study'] = study
    config['k_start'] = fold
    config['k_end'] = fold + 1
    
    # Add missing default parameters
    defaults = {
        'wsi_projection_dim': 256,
        'k': 5,
        'method': 'SurvOTRank',
        'use_pathway_data': config.get('rna_format') == 'Pathways',
        'gpu': '0',
        'num_genes': None,
    }
    for key, val in defaults.items():
        if key not in config:
            config[key] = val
    
    # Convert to argparse.Namespace (as expected by model_factory)
    from argparse import Namespace
    args = Namespace(**config)
    
    # Create dataset factory
    print(f"Creating dataset factory for {study} fold {fold}")
    dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_genes=getattr(args, 'num_genes', None),
        num_patches=getattr(args, 'num_patches', 2048),
        clinical_feature_cols=None,
        binning_mode=getattr(args, 'binning_mode', 'global_qcut'),
        which_splits=getattr(args, 'which_splits', '5fold'),
    )
    
    # Get omic dimensions
    omics_input_dim = sum(dataset_factory.omic_sizes)
    args.omic_sizes = dataset_factory.omic_sizes
    args.omic_names = dataset_factory.omic_names
    args.pathway_names = getattr(dataset_factory, 'pathway_names', None)
    
    # Create model
    print(f"Creating model: {args.survot_method}")
    model = get_model(
        method=args.survot_method,
        args=args,
        omic_input_dim=omics_input_dim,
        omic_names=args.omic_names,
        pathway_names=args.pathway_names,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle size mismatches for buffers that are initialized during training
    model_state = model.state_dict()
    filtered_checkpoint = {}
    
    for k, v in checkpoint.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_checkpoint[k] = v
            else:
                # Skip buffers with shape mismatch (they are fit on train data)
                print(f"  Skipping {k}: checkpoint shape {v.shape} != model shape {model_state[k].shape}")
        else:
            print(f"  Skipping {k}: not in model")
    
    model.load_state_dict(filtered_checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Load test split
    split_path = os.path.join(
        dataset_factory.data_path,
        "splits", dataset_factory.which_splits, dataset_factory.study,
        f"fold_{fold}.csv",
    )
    split_df = pd.read_csv(split_path)
    
    # Fit bins if needed
    if getattr(args, 'fit_bins_on_train', False):
        dataset_factory.fit_label_bins(split_df["train"].dropna().tolist())
    
    # Create test dataset (use 'val' split as test)
    wsi_path = os.path.join(
        args.data_root_dir,
        dataset_factory.study,
        getattr(args, 'wsi_encoder', 'uni2-h'),
        'pt_files',
    )
    
    from dataset.dataset_survival import SurvivalDataset
    on_missing_wsi = getattr(args, 'on_missing_wsi', 'error')
    test_dataset = SurvivalDataset(
        dataset_factory, wsi_path, 'val', fold, args.encoding_dim,
        on_missing_wsi=on_missing_wsi,
    )
    
    print(f"Loading test set: {len(test_dataset)} patients")
    
    # Create data loader
    if args.use_pathway_data:
        collate_fn = _collate_pathways
    else:
        collate_fn = None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(args, 'batch_size', 1),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return model, test_loader, dataset_factory, args


def run_intervention_audit(
    checkpoint_path: str,
    config_path: str,
    study: str,
    fold: int,
    alphas: List[float],
    device_str: str = 'cuda:0',
) -> pd.DataFrame:
    """Run continuous intervention audit on test set.
    
    Args:
        checkpoint_path: Path to trained model checkpoint (.pth file)
        config_path: Path to config file
        study: Cancer type (blca, ucec, etc.)
        fold: Fold number (0-4)
        alphas: List of intervention strengths to test
        device_str: Compute device
    
    Returns:
        DataFrame with intervention results
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    model, test_loader, dataset_factory, args = load_model_and_data(
        checkpoint_path, config_path, study, fold, device
    )
    
    # Extract prognostic anchors
    print("\nExtracting prognostic anchors...")
    low_risk_anchor, high_risk_anchor = extract_prognostic_anchors(model, device)
    
    print(f"  Low-risk anchor shape: {low_risk_anchor.shape}")
    print(f"  High-risk anchor shape: {high_risk_anchor.shape}")
    anchor_distance = torch.norm(high_risk_anchor - low_risk_anchor).item()
    print(f"  Anchor distance: {anchor_distance:.4f}")
    
    # Run intervention audit
    results = []
    
    print(f"\nRunning intervention audit with {len(alphas)} alpha values...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing test patients")):
        # Unpack batch: (wsi, genes, label, event_time, censorship)
        # or (wsi, genes, label, event_time, censorship, clinical) if clinical modality is used
        if len(batch) == 6:
            wsi, genes, label, event_time, censorship, _ = batch
        else:
            wsi, genes, label, event_time, censorship = batch
        
        # Get patient IDs from label_df based on batch indices
        # Since we're using batch_size=1, batch_idx is the index
        patient_id = test_loader.dataset.label_df.loc[batch_idx, 'case id']
        
        wsi = wsi.to(device)
        if isinstance(genes, list):
            genes = [g.to(device) for g in genes]
        else:
            genes = genes.to(device)
        label = label.to(device)
        event_time = event_time.to(device)
        censorship = censorship.to(device)
        
        true_time = event_time.item()
        true_event = 1 - censorship.item()  # 1=event, 0=censored
        
        # Get original embeddings from model
        with torch.no_grad():
            # Use model's slot attention to get pooled embeddings
            if hasattr(model, 'slot_attention_wsi'):
                wsi_slots = model.slot_attention_wsi(wsi)  # [1, num_slots, dim]
                original_embedding = wsi_slots.mean(dim=1)  # [1, dim]
            else:
                # Fallback: use wsi_mlp
                original_embedding = model.wsi_mlp(wsi.mean(dim=1))  # [1, dim]
        
        # Get baseline risk (alpha=0)
        baseline_risk = compute_risk_prediction(model, original_embedding, device)
        
        # Test interventions at each alpha
        for alpha in alphas:
            # Intervene towards low-risk anchor
            low_risk_embeddings = interpolate_towards_anchor(
                original_embeddings, low_risk_anchor, alpha
            )
            low_risk_preds = compute_risk_prediction(model, low_risk_embeddings, device)
            low_risk_distance = torch.norm(
                low_risk_embeddings - original_embeddings, dim=1
            ).cpu().numpy()
            
            # Intervene towards high-risk anchor
            high_risk_embeddings = interpolate_towards_anchor(
                original_embeddings, high_risk_anchor, alpha
            )
            high_risk_preds = compute_risk_prediction(model, high_risk_embeddings, device)
            high_risk_distance = torch.norm(
                high_risk_embeddings - original_embeddings, dim=1
            ).cpu().numpy()
            
            # Record results
            for i, pid in enumerate(patient_ids):
                # Low-risk intervention
                results.append({
                    'patient_id': pid,
                    'true_time': float(true_times[i]),
                    'true_event': int(true_events[i]),
                    'baseline_risk': float(baseline_risk[i].cpu().item()),
                    'alpha': float(alpha),
                    'direction': 'low_risk',
                    'risk_pred': float(low_risk_preds[i].cpu().item()),
                    'risk_change': float(low_risk_preds[i].cpu().item() - baseline_risk[i].cpu().item()),
                    'embedding_distance': float(low_risk_distance[i])
                })
                
                # High-risk intervention
                results.append({
                    'patient_id': pid,
                    'true_time': float(true_times[i]),
                    'true_event': int(true_events[i]),
                    'baseline_risk': float(baseline_risk[i].cpu().item()),
                    'alpha': float(alpha),
                    'direction': 'high_risk',
                    'risk_pred': float(high_risk_preds[i].cpu().item()),
                    'risk_change': float(high_risk_preds[i].cpu().item() - baseline_risk[i].cpu().item()),
                    'embedding_distance': float(high_risk_distance[i])
                })
    
    df = pd.DataFrame(results)
    print(f"\nCollected {len(df)} intervention measurements")
    print(f"  {len(df['patient_id'].unique())} unique patients")
    print(f"  {len(alphas)} alpha values × 2 directions")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="E4 Continuous Intervention Audit for DCT-Reg"
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--study', required=True,
        help='Cancer study (blca, ucec, etc.)'
    )
    parser.add_argument(
        '--fold', type=int, required=True,
        help='Fold number (0-4)'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output CSV path'
    )
    parser.add_argument(
        '--alphas', default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
        help='Comma-separated list of alpha values'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Compute device'
    )
    
    args = parser.parse_args()
    
    # Parse alphas
    alphas = [float(a) for a in args.alphas.split(',')]
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run audit
    print("=" * 70)
    print("E4 Continuous Intervention Audit")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Study: {args.study}")
    print(f"Fold: {args.fold}")
    print(f"Alphas: {alphas}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    df = run_intervention_audit(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        study=args.study,
        fold=args.fold,
        alphas=alphas,
        device_str=args.device
    )
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    for direction in ['low_risk', 'high_risk']:
        df_dir = df[df['direction'] == direction]
        print(f"\n{direction.upper()}:")
        print(f"  Mean risk change @ α=1.0: {df_dir[df_dir['alpha']==1.0]['risk_change'].mean():.4f}")
        print(f"  Std risk change @ α=1.0: {df_dir[df_dir['alpha']==1.0]['risk_change'].std():.4f}")
        
        # Check monotonicity
        n_patients = len(df_dir['patient_id'].unique())
        n_monotonic = 0
        for pid in df_dir['patient_id'].unique():
            patient_data = df_dir[df_dir['patient_id'] == pid].sort_values('alpha')
            risks = patient_data['risk_pred'].values
            if direction == 'low_risk':
                # Should decrease
                is_monotonic = all(risks[i] >= risks[i+1] for i in range(len(risks)-1))
            else:
                # Should increase
                is_monotonic = all(risks[i] <= risks[i+1] for i in range(len(risks)-1))
            if is_monotonic:
                n_monotonic += 1
        
        monotonic_rate = n_monotonic / n_patients * 100
        print(f"  Monotonic curves: {n_monotonic}/{n_patients} ({monotonic_rate:.1f}%)")


if __name__ == '__main__':
    main()
