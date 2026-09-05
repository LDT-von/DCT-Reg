#!/usr/bin/env python3
"""E4: Continuous Intervention Audit for DCT-Reg (Adapted Version).

This script tests the core claim: directionally consistent risk response 
to prognostic interventions.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
    SurvivalDatasetFactory, SurvivalDataset
)
from survot_rank.training.model_factory import get_model
from survot_rank.training.extended_args import process_args_extended


def load_model_and_data(
    checkpoint_path: str,
    study: str,
    fold: int,
    device: torch.device,
    data_csv_root: str = "/data1/DCT-Reg/data/dataset_csv",
    data_root: str = "/data1/TCGA-UNI2-h-features",
    wsi_encoder: str = "uni2-h",
    encoding_dim: int = 1536
):
    """Load trained model and test dataset."""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint or use defaults
    if 'args' in checkpoint:
        args = checkpoint['args']
    else:
        # Load config from YAML
        import yaml
        config_path = Path(__file__).parent.parent / 'configs' / 'dct_v310_directional_regularized_transport.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create args from config
        from argparse import Namespace
        args = Namespace(
            # Study
            study=study,
            survot_method='dct_v310_directional_regularized_transport',
            
            # Data paths
            data_root_dir=data_root,
            data_path=data_csv_root,
            wsi_encoder=wsi_encoder,
            on_missing_wsi='error',
            
            # Data config
            rna_format=config['data']['rna_format'],
            signature=config['data']['signature'],
            label_col=config['data']['label_col'],
            n_classes=config['data']['n_classes'],
            n_bins=config['data']['n_classes'],
            num_patches=config['data']['num_patches'],
            encoding_dim=encoding_dim,
            num_genes=None,
            num_workers=4,
            
            # Training (needed for model init)
            batch_size=config['train']['batch_size'],
            max_epochs=config['train']['max_epochs'],
            lr=config['train']['lr'],
            bag_loss=config['train']['bag_loss'],
            alpha_surv=config['train']['alpha_surv'],
            
            # Slot config
            slot_num_wsi=config['slot']['slot_num_wsi'],
            slot_num_omics=config['slot']['slot_num_omics'],
            slot_iters=config['slot']['slot_iters'],
            temperature=config['slot']['temperature'],
            topk_ratio=config['slot']['topk_ratio'],
            top_k_method=config['slot']['top_k_method'],
            
            # Model config
            otehv2_eps=config['model']['otehv2_eps'],
            otehv2_iter=config['model']['otehv2_iter'],
            otehv2_heads=config['model']['otehv2_heads'],
            otehv2_layers=config['model']['otehv2_layers'],
            otehv2_dropout=config['model']['otehv2_dropout'],
            dct_num_stages=config['model']['dct_num_stages'],
            
            # DCT objectives
            dct_lambda_ipcw_rank=config['model']['dct_lambda_ipcw_rank'],
            dct_ipcw_rank_margin=config['model']['dct_ipcw_rank_margin'],
            dct_ipcw_rank_temperature=config['model']['dct_ipcw_rank_temperature'],
            dct_ipcw_max_weight=config['model']['dct_ipcw_max_weight'],
            dct_ipcw_rank_memory_size=config['model']['dct_ipcw_rank_memory_size'],
            dct_v38_lambda_direction=config['model']['dct_v38_lambda_direction'],
            dct_v38_lambda_dose=config['model']['dct_v38_lambda_dose'],
            dct_v38_lambda_reconfiguration=config['model']['dct_v38_lambda_reconfiguration'],
            
            # Additional DCT parameters
            dct_anchor_momentum=config['model']['dct_anchor_momentum'],
            dct_evidence_cost_weight=config['model']['dct_evidence_cost_weight'],
            dct_evidence_mass_floor=config['model']['dct_evidence_mass_floor'],
            dct_evidence_marginal_strength=config['model']['dct_evidence_marginal_strength'],
            dct_geometry_reliability_strength=config['model']['dct_geometry_reliability_strength'],
            dct_coupling_projection_iters=config['model']['dct_coupling_projection_iters'],
            dct_coupling_projection_tol=config['model']['dct_coupling_projection_tol'],
            dct_coordinate_temperature=config['model']['dct_coordinate_temperature'],
            dct_mix_ratio=config['model']['dct_mix_ratio'],
            
            dct_v38_direction_margin=config['model']['dct_v38_direction_margin'],
            dct_v38_dose_margin=config['model']['dct_v38_dose_margin'],
            dct_v38_reconfiguration_margin=config['model']['dct_v38_reconfiguration_margin'],
            dct_v38_temperature=config['model']['dct_v38_temperature'],
            dct_v38_alpha_mid=config['model']['dct_v38_alpha_mid'],
            dct_v38_alpha_full=config['model']['dct_v38_alpha_full'],
            dct_v38_warmup_epochs=config['model']['dct_v38_warmup_epochs'],
            dct_v38_ramp_epochs=config['model']['dct_v38_ramp_epochs'],
            dct_v38_dose_every=config['model']['dct_v38_dose_every'],
            
            dct_lambda_etar=config['model']['dct_lambda_etar'],
            dct_lambda_listwise=config['model']['dct_lambda_listwise'],
            dct_v382_lambda_mgptr=config['model']['dct_v382_lambda_mgptr'],
            fet_lambda_sparse=config['model']['fet_lambda_sparse'],
            fet_lambda_faith=config['model']['fet_lambda_faith'],
            spt_prog_cost=config['model']['spt_prog_cost'],
            rg_eps_start=config['model']['rg_eps_start'],
            rg_eps_anneal=config['model']['rg_eps_anneal'],
            
            # Legacy model parameters (will be set after dataset factory)
            omic_sizes=None,
            wsi_projection_dim=256,
            wsi_dim=256,
            rna_dim=256,
            gene_dim=256,
            rank_weight=8.0,
            graph_reg_weight=8.0,
            dropout=0.25,
        )
    
    # Create dataset factory with correct paths
    dataset_factory = SurvivalDatasetFactory(
        study=study,
        data_path=getattr(args, 'data_path', data_csv_root),
        rna_format=getattr(args, 'rna_format', 'Pathways'),
        signature=getattr(args, 'signature', 'combine'),
        n_bins=getattr(args, 'n_classes', 4),
        label_col=getattr(args, 'label_col', 'survival_months_dss'),
        num_genes=getattr(args, 'num_genes', None),
        num_patches=getattr(args, 'num_patches', 4096),
        clinical_feature_cols=None,
        binning_mode='global_qcut',
        which_splits='5fold_uni2h',
    )
    
    # Update args with omic sizes
    args.omic_sizes = dataset_factory.omic_sizes
    
    # Load test dataset
    wsi_path = Path(data_root) / study / wsi_encoder / "pt_files"
    test_dataset = SurvivalDataset(
        dataset_factory=dataset_factory,
        wsi_path=str(wsi_path),
        split_key='val',
        fold=fold,
        encoding_dim=encoding_dim,
        on_missing_wsi=getattr(args, 'on_missing_wsi', 'error'),
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get omic dimensions from dataset factory
    omic_input_dim = sum(dataset_factory.omic_sizes)
    omic_names = dataset_factory.omic_names
    pathway_names = None
    
    print(f"Omic input dim: {omic_input_dim}")
    print(f"Omic names: {omic_names}")
    
    # Create model
    model = get_model(
        method=getattr(args, 'survot_method', 'dct_v310_directional_regularized_transport'),
        args=args,
        omic_input_dim=omic_input_dim,
        omic_names=omic_names,
        pathway_names=pathway_names
    )
    
    # Load model weights (filter out mismatched buffers)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Filter out buffers with size mismatch
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f"Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
        else:
            filtered_state[k] = v  # Keep it anyway, strict=False will handle it
    
    model.load_state_dict(filtered_state, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.__class__.__name__}")
    
    return model, test_dataset, dataset_factory


def extract_prognostic_anchors(
    model: torch.nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract learned prognostic anchors from DCT-Reg model."""
    
    # Check for risk_anchor_costs (stored in DCT-Reg models)
    if hasattr(model, 'risk_anchor_costs'):
        # risk_anchor_costs has shape [2, dim] where [0] is low-risk, [1] is high-risk
        anchors = model.risk_anchor_costs
        if anchors.shape[0] >= 2:
            low = anchors[0].to(device)
            high = anchors[1].to(device)
            print(f"Extracted anchors from risk_anchor_costs: shape={anchors.shape}")
            return low, high
    
    # Try different possible locations
    if hasattr(model, 'low_risk_anchor') and hasattr(model, 'high_risk_anchor'):
        low = model.low_risk_anchor.to(device)
        high = model.high_risk_anchor.to(device)
    elif hasattr(model, 'anchors'):
        anchors = model.anchors
        low = anchors[0].to(device)
        high = anchors[-1].to(device)
    elif hasattr(model, 'dct_module'):
        dct = model.dct_module
        if hasattr(dct, 'low_risk_anchor'):
            low = dct.low_risk_anchor.to(device)
            high = dct.high_risk_anchor.to(device)
        elif hasattr(dct, 'anchors'):
            low = dct.anchors[0].to(device)
            high = dct.anchors[-1].to(device)
        else:
            raise AttributeError("Cannot find anchors in dct_module")
    # Try event_reference (from DCT v3.8+)
    elif hasattr(model, 'event_reference'):
        ref = model.event_reference  # Shape: (n_bins, dim)
        low = ref[0].to(device)   # First bin = best prognosis
        high = ref[-1].to(device)  # Last bin = worst prognosis
    else:
        raise AttributeError(
            f"Cannot find prognostic anchors in model {model.__class__.__name__}. "
            f"Available attributes: {dir(model)}"
        )
    
    return low, high


def get_embedding(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """Extract embedding from model encoder."""
    
    # Get WSI features
    if 'wsi' in batch:
        wsi = batch['wsi'].to(device)
    elif 'x_path' in batch:
        wsi = batch['x_path'].to(device)
    else:
        raise KeyError(f"Cannot find WSI features in batch keys: {batch.keys()}")
    
    # Forward through encoder
    with torch.no_grad():
        if hasattr(model, 'encoder'):
            embedding = model.encoder(wsi)
        elif hasattr(model, 'wsi_encoder'):
            embedding = model.wsi_encoder(wsi)
        else:
            # Try forward with wsi only
            embedding = model(wsi_input=wsi, return_embedding=True)
    
    return embedding


def compute_risk(model: torch.nn.Module, embedding: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute risk score from embedding."""
    
    embedding = embedding.to(device)
    
    with torch.no_grad():
        # Try different decoder types
        if hasattr(model, 'decoder'):
            logits = model.decoder(embedding)
        elif hasattr(model, 'hazard_predictor'):
            logits = model.hazard_predictor(embedding)
        elif hasattr(model, 'risk_predictor'):
            logits = model.risk_predictor(embedding)
        else:
            # Try full forward with embedding
            logits = model.forward_from_embedding(embedding)
        
        # Convert to risk score
        if logits.dim() == 2:  # Multi-bin discrete-time
            # Use negative expected survival (higher = worse prognosis)
            probs = torch.softmax(logits, dim=1)
            time_bins = torch.arange(logits.size(1), device=device, dtype=torch.float32)
            risk = -torch.sum(probs * time_bins, dim=1)
        else:  # Single output
            risk = logits.squeeze(-1)
    
    return risk


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
    
    Returns:
        Interpolated embedding (batch_size, dim)
    """
    return (1 - alpha) * embedding + alpha * anchor.unsqueeze(0)


def run_intervention_audit(
    checkpoint_path: str,
    study: str,
    fold: int,
    alphas: List[float],
    device: str = 'cuda:0',
    batch_size: int = 16,
    data_csv_root: str = "/data1/DCT-Reg/data/dataset_csv",
    data_root: str = "/data1/TCGA-UNI2-h-features"
) -> pd.DataFrame:
    """Run continuous intervention audit."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model and data
    model, test_dataset, dataset_factory = load_model_and_data(
        checkpoint_path=checkpoint_path,
        study=study,
        fold=fold,
        device=device,
        data_csv_root=data_csv_root,
        data_root=data_root
    )
    
    # Extract anchors
    low_risk_anchor, high_risk_anchor = extract_prognostic_anchors(model, device)
    print(f"\nExtracted prognostic anchors:")
    print(f"  Low-risk anchor shape: {low_risk_anchor.shape}")
    print(f"  High-risk anchor shape: {high_risk_anchor.shape}")
    print(f"  Anchor distance: {torch.norm(high_risk_anchor - low_risk_anchor).item():.4f}\n")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run interventions
    results = []
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Auditing test patients")):
        # Get batch size
        batch_size = len(batch) if isinstance(batch, (list, tuple)) else batch['wsi'].shape[0] if 'wsi' in batch else batch['omics'].shape[0]
        
        # Get patient info
        if isinstance(batch, dict):
            if 'case_id' in batch:
                patient_ids = batch['case_id']
            elif 'patient_id' in batch:
                patient_ids = batch['patient_id']
            else:
                patient_ids = [f"patient_{batch_idx}_{i}" for i in range(batch_size)]
        else:
            patient_ids = [f"patient_{batch_idx}_{i}" for i in range(batch_size)]
        
        # Get survival info
        if isinstance(batch, dict):
            times = batch['survival_time'].cpu().numpy() if 'survival_time' in batch else batch['label'][:, 0].cpu().numpy()
            events = batch['event'].cpu().numpy() if 'event' in batch else batch['label'][:, 1].cpu().numpy()
        else:
            times = batch[2].cpu().numpy() if len(batch) > 2 else None
            events = batch[3].cpu().numpy() if len(batch) > 3 else None
        
        # Get original embeddings
        original_embeddings = get_embedding(model, batch, device)
        
        # Get original risk
        original_risk = compute_risk(model, original_embeddings, device)
        
        # Test each alpha
        for alpha in alphas:
            # Intervention towards low-risk anchor
            low_risk_emb = interpolate_towards_anchor(
                original_embeddings, low_risk_anchor, alpha
            )
            low_risk_pred = compute_risk(model, low_risk_emb, device)
            low_risk_dist = torch.norm(low_risk_emb - original_embeddings, dim=1)
            
            # Intervention towards high-risk anchor
            high_risk_emb = interpolate_towards_anchor(
                original_embeddings, high_risk_anchor, alpha
            )
            high_risk_pred = compute_risk(model, high_risk_emb, device)
            high_risk_dist = torch.norm(high_risk_emb - original_embeddings, dim=1)
            
            # Record results
            for i in range(len(patient_ids)):
                pid = patient_ids[i] if isinstance(patient_ids[i], str) else str(patient_ids[i])
                
                # Low-risk direction
                results.append({
                    'patient_id': pid,
                    'true_time': float(times[i]),
                    'true_event': int(events[i]),
                    'original_risk': float(original_risk[i].cpu().item()),
                    'alpha': float(alpha),
                    'direction': 'low_risk',
                    'risk_pred': float(low_risk_pred[i].cpu().item()),
                    'embedding_distance': float(low_risk_dist[i].cpu().item())
                })
                
                # High-risk direction
                results.append({
                    'patient_id': pid,
                    'true_time': float(times[i]),
                    'true_event': int(events[i]),
                    'original_risk': float(original_risk[i].cpu().item()),
                    'alpha': float(alpha),
                    'direction': 'high_risk',
                    'risk_pred': float(high_risk_pred[i].cpu().item()),
                    'embedding_distance': float(high_risk_dist[i].cpu().item())
                })
    
    return pd.DataFrame(results)


def analyze_direction_consistency(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze direction consistency from intervention results."""
    
    metrics = {}
    patients = df['patient_id'].unique()
    
    # Check monotonicity for each patient
    monotonic_low = 0
    monotonic_high = 0
    
    risk_changes_low = []
    risk_changes_high = []
    
    for pid in patients:
        patient_df = df[df['patient_id'] == pid].sort_values('alpha')
        
        # Low-risk direction: risk should decrease as α increases
        low_df = patient_df[patient_df['direction'] == 'low_risk']
        if len(low_df) > 1:
            risks = low_df['risk_pred'].values
            # Check if monotonically decreasing
            is_monotonic = all(risks[i] >= risks[i+1] for i in range(len(risks)-1))
            if is_monotonic:
                monotonic_low += 1
            # Record change from α=0 to α=1
            if len(risks) >= 2:
                risk_changes_low.append(risks[-1] - risks[0])
        
        # High-risk direction: risk should increase as α increases  
        high_df = patient_df[patient_df['direction'] == 'high_risk']
        if len(high_df) > 1:
            risks = high_df['risk_pred'].values
            # Check if monotonically increasing
            is_monotonic = all(risks[i] <= risks[i+1] for i in range(len(risks)-1))
            if is_monotonic:
                monotonic_high += 1
            # Record change from α=0 to α=1
            if len(risks) >= 2:
                risk_changes_high.append(risks[-1] - risks[0])
    
    metrics['n_patients'] = len(patients)
    metrics['monotonic_decrease_rate'] = monotonic_low / len(patients)
    metrics['monotonic_increase_rate'] = monotonic_high / len(patients)
    
    if risk_changes_low:
        metrics['mean_risk_change_low'] = float(np.mean(risk_changes_low))
        metrics['std_risk_change_low'] = float(np.std(risk_changes_low))
    
    if risk_changes_high:
        metrics['mean_risk_change_high'] = float(np.mean(risk_changes_high))
        metrics['std_risk_change_high'] = float(np.std(risk_changes_high))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="E4: Continuous Intervention Audit (Adapted)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--study', type=str, required=True,
                        choices=['blca', 'ucec', 'lusc', 'kirc', 'hnsc', 'skcm'],
                        help='Cancer type')
    parser.add_argument('--fold', type=int, required=True,
                        choices=[0, 1, 2, 3, 4],
                        help='Fold number')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file')
    parser.add_argument('--alphas', type=str, default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated intervention strengths')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Compute device')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--data-csv-root', type=str, default='/data1/DCT-Reg/data/dataset_csv',
                        help='Data CSV root directory')
    parser.add_argument('--data-root', type=str, default='/data1/TCGA-UNI2-h-features',
                        help='WSI features root directory')
    
    args = parser.parse_args()
    
    # Parse alphas
    alphas = [float(a) for a in args.alphas.split(',')]
    
    print("="*80)
    print("E4: Continuous Intervention Audit for DCT-Reg")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Study: {args.study}, Fold: {args.fold}")
    print(f"Intervention strengths α: {alphas}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Run audit
    results_df = run_intervention_audit(
        checkpoint_path=args.checkpoint,
        study=args.study,
        fold=args.fold,
        alphas=alphas,
        device=args.device,
        batch_size=args.batch_size,
        data_csv_root=args.data_csv_root,
        data_root=args.data_root
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    print(f"   Total interventions: {len(results_df)}")
    
    # Analyze direction consistency
    metrics = analyze_direction_consistency(results_df)
    
    print("\n" + "="*80)
    print("Direction Consistency Analysis")
    print("="*80)
    print(f"Number of test patients: {metrics['n_patients']}")
    print(f"\nMonotonicity rates:")
    print(f"  Towards low-risk:  {metrics['monotonic_decrease_rate']:6.2%} (risk should ↓)")
    print(f"  Towards high-risk: {metrics['monotonic_increase_rate']:6.2%} (risk should ↑)")
    
    if 'mean_risk_change_low' in metrics:
        print(f"\nRisk change at α=1.0:")
        print(f"  Towards low-risk:  {metrics['mean_risk_change_low']:+.4f} ± {metrics['std_risk_change_low']:.4f}")
        print(f"  Towards high-risk: {metrics['mean_risk_change_high']:+.4f} ± {metrics['std_risk_change_high']:.4f}")
    
    print("="*80)
    
    # Save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    import json
    with open(summary_path, 'w') as f:
        summary = {
            'study': args.study,
            'fold': args.fold,
            'checkpoint': args.checkpoint,
            'alphas': alphas,
            'metrics': metrics
        }
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to {summary_path}\n")


if __name__ == '__main__':
    main()
