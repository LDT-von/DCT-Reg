#!/usr/bin/env python3
"""
E4 Experiment: Direction Consistency Audit for DCT v3.10
Tests whether intermediate intervention states maintain directional monotonicity
towards the counterfactual target.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import numpy as np
import pandas as pd
from argparse import Namespace
from tqdm import tqdm
import json

from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
    SurvivalDatasetFactory, SurvivalDataset
)
from survot_rank.training.model_factory import get_model


def extract_prognostic_anchors(model):
    """Extract low and high risk anchors from the trained model."""
    if hasattr(model, 'risk_anchor_costs'):
        anchors = model.risk_anchor_costs
        # Shape: [num_stages, 2, num_slots_wsi, slot_dim_wsi, slot_dim_omics]
        # Extract first stage low-risk and last stage high-risk
        low_risk = anchors[0, 0].flatten()  # First stage, low risk
        high_risk = anchors[-1, 1].flatten()  # Last stage, high risk
        return low_risk.detach(), high_risk.detach()
    else:
        raise ValueError("Model does not have risk_anchor_costs attribute")


def compute_directional_consistency(embeddings, low_risk_anchor, high_risk_anchor):
    """
    Compute directional consistency for a sequence of embeddings.
    
    Args:
        embeddings: List of embeddings [factual, alpha_0.25, alpha_0.5, alpha_0.75, target]
        low_risk_anchor: Low risk anchor embedding
        high_risk_anchor: High risk anchor embedding
        
    Returns:
        dict with consistency metrics
    """
    # Compute distances to anchors for each embedding
    distances_to_low = []
    distances_to_high = []
    
    for emb in embeddings:
        dist_low = torch.norm(emb - low_risk_anchor).item()
        dist_high = torch.norm(emb - high_risk_anchor).item()
        distances_to_low.append(dist_low)
        distances_to_high.append(dist_high)
    
    # Check monotonicity
    # For low->high intervention: distance to low should increase, distance to high should decrease
    low_monotonic = all(distances_to_low[i] <= distances_to_low[i+1] for i in range(len(distances_to_low)-1))
    high_monotonic = all(distances_to_high[i] >= distances_to_high[i+1] for i in range(len(distances_to_high)-1))
    
    # Compute direction alignment
    # Vector from factual to target
    factual_to_target = embeddings[-1] - embeddings[0]
    
    # Vectors from factual to intermediate steps
    alignments = []
    for i in range(1, len(embeddings)-1):
        factual_to_intermediate = embeddings[i] - embeddings[0]
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            factual_to_target.unsqueeze(0),
            factual_to_intermediate.unsqueeze(0)
        ).item()
        alignments.append(cos_sim)
    
    return {
        'distances_to_low': distances_to_low,
        'distances_to_high': distances_to_high,
        'low_risk_monotonic': low_monotonic,
        'high_risk_monotonic': high_monotonic,
        'direction_alignments': alignments,
        'mean_alignment': np.mean(alignments) if alignments else 0.0
    }


def run_e4_audit(checkpoint_path, fold, cancer, device='cuda:0'):
    """Run E4 audit for a single checkpoint."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"E4 Audit: {cancer} Fold {fold}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'dct_v310_directional_regularized_transport.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create args
    args = Namespace(
        study=cancer,
        survot_method='dct_transport_intervention_consistency',
        
        # Data paths
        data_root_dir='/data1/TCGA-UNI2-h-features',
        data_path='/data1/DCT-Reg/data/dataset_csv',
        wsi_encoder='uni2-h',
        on_missing_wsi='error',
        
        # Data config
        rna_format='Pathways',
        signature='combine',
        label_col='survival_months_dss',
        n_classes=4,
        n_bins=4,
        num_patches=4096,
        encoding_dim=1536,
        num_genes=None,
        num_workers=4,
        
        # Training
        batch_size=8,
        max_epochs=50,
        lr=5e-4,
        bag_loss='nll_surv',
        alpha_surv=0.0,
        
        # Slot config
        slot_num_wsi=8,
        slot_num_omics=8,
        slot_iters=3,
        temperature=1.0,
        topk_ratio=0.25,
        top_k_method='random',
        
        # Model config
        otehv2_eps=0.01,
        otehv2_iter=30,
        otehv2_heads=8,
        otehv2_layers=2,
        otehv2_dropout=0.1,
        dct_num_stages=4,
        
        # DCT objectives (will be updated per variant)
        dct_lambda_ipcw_rank=0.0,
        dct_ipcw_rank_margin=0.5,
        dct_ipcw_rank_temperature=0.1,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=512,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        
        # DCT parameters
        dct_anchor_momentum=0.95,
        dct_evidence_cost_weight=0.1,
        dct_evidence_mass_floor=0.01,
        dct_evidence_marginal_strength=1.0,
        dct_geometry_reliability_strength=1.0,
        dct_coupling_projection_iters=10,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.1,
        dct_mix_ratio=0.5,
        
        dct_v38_direction_margin=0.2,
        dct_v38_dose_margin=0.3,
        dct_v38_reconfiguration_margin=0.1,
        dct_v38_temperature=0.1,
        dct_v38_alpha_mid=0.5,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=5,
        dct_v38_ramp_epochs=10,
        dct_v38_dose_every=5,
        
        dct_lambda_etar=0.0,
        dct_lambda_listwise=0.0,
        dct_v382_lambda_mgptr=0.0,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.20,
        rg_eps_start=0.1,
        rg_eps_anneal=0.99,
        
        # Legacy model parameters
        omic_sizes=None,
        wsi_projection_dim=256,
        wsi_dim=256,
        rna_dim=256,
        gene_dim=256,
        rank_weight=8.0,
        graph_reg_weight=8.0,
        dropout=0.25,
    )
    
    # Create dataset factory
    print("Creating dataset factory...")
    dataset_factory = SurvivalDatasetFactory(
        study=cancer,
        data_path=args.data_path,
        rna_format='Pathways',
        signature='combine',
        n_bins=4,
        label_col='survival_months_dss',
        num_genes=None,
        num_patches=4096,
        clinical_feature_cols=None,
        binning_mode='global_qcut',
        which_splits='5fold_uni2h',
    )
    
    # Update args with omic sizes
    args.omic_sizes = dataset_factory.omic_sizes
    omic_input_dim = sum(dataset_factory.omic_sizes)
    omic_names = dataset_factory.omic_names
    
    # Create model
    print("Creating model...")
    model = get_model(
        method=args.survot_method,
        args=args,
        omic_input_dim=omic_input_dim,
        omic_names=omic_names,
        pathway_names=None
    )
    
    # Load train dataset and configure reference
    print("Loading train dataset...")
    wsi_path_train = Path('/data1/TCGA-UNI2-h-features') / cancer / 'uni2-h' / 'pt_files'
    train_dataset = SurvivalDataset(
        dataset_factory=dataset_factory,
        wsi_path=str(wsi_path_train),
        split_key='train',
        fold=fold,
        encoding_dim=1536,
        on_missing_wsi='error',
    )
    
    train_times = train_dataset.label_df[dataset_factory.label_col].values
    train_censor = train_dataset.label_df[dataset_factory.censorship_var].values
    
    if hasattr(model, 'configure_train_reference'):
        model.configure_train_reference(train_times, train_censor)
        print("Configured train reference")
    
    # Load checkpoint
    print("Loading checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys")
    
    model = model.to(device)
    model.eval()
    
    # Extract prognostic anchors
    print("Extracting prognostic anchors...")
    low_risk_anchor, high_risk_anchor = extract_prognostic_anchors(model)
    anchor_distance = torch.norm(high_risk_anchor - low_risk_anchor).item()
    print(f"Anchor distance: {anchor_distance:.4f}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = SurvivalDataset(
        dataset_factory=dataset_factory,
        wsi_path=str(wsi_path_train),
        split_key='val',
        fold=fold,
        encoding_dim=1536,
        on_missing_wsi='error',
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Run audit on test set
    print("\nRunning directional consistency audit...")
    results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Processing samples"):
            sample = test_dataset[idx]
            
            # Prepare input
            x_path = sample[0]
            x_omic = sample[1] if len(sample) > 1 else None
            
            if isinstance(x_path, list):
                x_path = torch.stack(x_path)
            if not isinstance(x_path, torch.Tensor):
                x_path = torch.tensor(x_path)
            wsi = x_path.unsqueeze(0).to(device)
            
            # Prepare pathways
            omic_kwargs = {}
            if x_omic is not None and isinstance(x_omic, list):
                for i, pathway in enumerate(x_omic):
                    if isinstance(pathway, list):
                        pathway = torch.cat([t.flatten() if t.dim() > 1 else t for t in pathway])
                    if not isinstance(pathway, torch.Tensor):
                        pathway = torch.tensor(pathway)
                    omic_kwargs[f'x_omic{i+1}'] = pathway.unsqueeze(0).to(device)
            
            try:
                # Forward pass
                output = model(x_wsi=wsi, **omic_kwargs)
                
                # Extract embeddings at different alpha levels
                # This requires accessing model internals during intervention
                # For now, we'll compute basic metrics
                
                # Get hazard prediction
                hazards, _ = output
                risk = -torch.sum(torch.cumprod(1 - torch.sigmoid(hazards), dim=1), dim=1).item()
                
                # Placeholder for full intervention chain analysis
                # In full implementation, we would run multiple forward passes with different alphas
                
                results.append({
                    'sample_idx': idx,
                    'risk_score': risk,
                    # More metrics will be added when intervention chain is implemented
                })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Compile results
    df = pd.DataFrame(results)
    
    return df, {
        'checkpoint': str(checkpoint_path),
        'cancer': cancer,
        'fold': fold,
        'n_samples': len(test_dataset),
        'n_processed': len(results),
        'anchor_distance': anchor_distance,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='E4 Direction Consistency Audit')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--cancer', type=str, default='blca', help='Cancer type')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    
    args = parser.parse_args()
    
    # Run audit
    df, metadata = run_e4_audit(
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        cancer=args.cancer,
        device=args.device
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path('results/e4_audits')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'e4_audit_{args.cancer}_fold{args.fold}.csv'
    
    df.to_csv(output_path, index=False)
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("Summary:")
    print(f"  Samples processed: {metadata['n_processed']} / {metadata['n_samples']}")
    print(f"  Anchor distance: {metadata['anchor_distance']:.4f}")
    if len(df) > 0:
        print(f"  Mean risk score: {df['risk_score'].mean():.4f}")
        print(f"  Risk score std: {df['risk_score'].std():.4f}")


if __name__ == '__main__':
    main()
