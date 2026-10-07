#!/usr/bin/env python3
"""E4: Continuous Intervention Audit for DCT-Reg.

This experiment directly tests the core claim of DCT-Reg:
"Directionally consistent risk response to prognostic ground-cost interventions"

For each test patient, we manually interpolate their embedding towards the
low-risk and high-risk anchors at different intervention strengths α ∈ [0, 1],
then measure how the predicted risk changes.

Expected outcome:
- If α↑ (moving towards low-risk anchor) → risk↓ (monotonic decrease)
- This would directly prove the direction constraint is working

Usage:
    python scripts/e4_continuous_intervention_audit.py \
        --checkpoint results/.../s_0_checkpoint.pt \
        --config configs/dct_v310_directional_regularized_transport.yaml \
        --study blca \
        --fold 0 \
        --output results/e4_intervention_audit/blca_fold0.csv
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from survot_rank.datasets.survival_dataset import SurvivalDataset
from survot_rank.utils.config import load_config


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
        model: Trained model
        embedding: Patient embedding (batch_size, dim)
        device: Compute device
    
    Returns:
        Risk scores (batch_size,)
    """
    embedding = embedding.to(device)
    
    with torch.no_grad():
        # Forward through decoder to get survival prediction
        # This depends on your model architecture
        # Typically: embedding -> decoder -> hazard/risk
        if hasattr(model, 'decoder'):
            logits = model.decoder(embedding)
        elif hasattr(model, 'risk_predictor'):
            logits = model.risk_predictor(embedding)
        else:
            raise AttributeError("Model has no decoder or risk_predictor")
        
        # Convert to risk score (higher = worse prognosis)
        # This depends on your survival model type
        if logits.dim() == 2:  # Multi-bin output
            # Use expected time or median survival
            risk = -torch.sum(torch.softmax(logits, dim=1) * 
                            torch.arange(logits.size(1), device=device).float(), 
                            dim=1)
        else:  # Single output (Cox, etc.)
            risk = logits.squeeze()
    
    return risk


def extract_prognostic_anchors(
    model: torch.nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract learned prognostic anchors from model.
    
    Returns:
        low_risk_anchor: Embedding for low-risk prototype (dim,)
        high_risk_anchor: Embedding for high-risk prototype (dim,)
    """
    # This depends on your model architecture
    # DCT-Reg should have stored anchors somewhere
    
    if hasattr(model, 'low_risk_anchor'):
        low_risk_anchor = model.low_risk_anchor
        high_risk_anchor = model.high_risk_anchor
    elif hasattr(model, 'anchors'):
        # Assume anchors is [low, high]
        low_risk_anchor = model.anchors[0]
        high_risk_anchor = model.anchors[1]
    elif hasattr(model, 'dct_module') and hasattr(model.dct_module, 'anchors'):
        anchors = model.dct_module.anchors
        low_risk_anchor = anchors[0]
        high_risk_anchor = anchors[1]
    else:
        raise AttributeError(
            "Cannot find prognostic anchors in model. "
            "Expected attributes: low_risk_anchor/high_risk_anchor, anchors, or dct_module.anchors"
        )
    
    return low_risk_anchor.to(device), high_risk_anchor.to(device)


def run_intervention_audit(
    checkpoint_path: str,
    config_path: str,
    study: str,
    fold: int,
    alphas: List[float],
    device: str = 'cuda:0',
    batch_size: int = 8
) -> pd.DataFrame:
    """Run continuous intervention audit on test set.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to config file
        study: Cancer type (blca, ucec, etc.)
        fold: Fold number (0-4)
        alphas: List of intervention strengths to test
        device: Compute device
        batch_size: Batch size for inference
    
    Returns:
        DataFrame with columns:
            - patient_id: Patient identifier
            - true_time: True survival time
            - true_event: True event indicator
            - alpha: Intervention strength
            - direction: 'low_risk' or 'high_risk'
            - risk_pred: Predicted risk score
            - embedding_distance: Distance moved from original
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = load_config(config_path)
    config['study'] = study
    config['k_start'] = fold
    config['k_end'] = fold + 1
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # TODO: Instantiate model from config
    # This is project-specific, you need to adapt this
    # model = create_model_from_config(config)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to(device)
    # model.eval()
    
    print("ERROR: Model instantiation not implemented yet!")
    print("You need to implement model creation from config in this script.")
    sys.exit(1)
    
    # Extract prognostic anchors
    low_risk_anchor, high_risk_anchor = extract_prognostic_anchors(model, device)
    print(f"Extracted anchors:")
    print(f"  Low-risk anchor: {low_risk_anchor.shape}")
    print(f"  High-risk anchor: {high_risk_anchor.shape}")
    print(f"  Anchor distance: {torch.norm(high_risk_anchor - low_risk_anchor).item():.4f}")
    
    # Load test dataset
    # TODO: Load test split
    # test_dataset = SurvivalDataset(...)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("ERROR: Dataset loading not implemented yet!")
    sys.exit(1)
    
    # Run intervention audit
    results = []
    
    for batch in tqdm(test_loader, desc="Processing test patients"):
        # Extract patient data
        patient_ids = batch['patient_id']
        true_times = batch['survival_time'].cpu().numpy()
        true_events = batch['event'].cpu().numpy()
        
        # Get original embeddings
        with torch.no_grad():
            # TODO: Extract embeddings from encoder
            # original_embeddings = model.encoder(batch['features'].to(device))
            pass
        
        # Test interventions at each alpha
        for alpha in alphas:
            # Intervene towards low-risk anchor
            low_risk_embeddings = interpolate_towards_anchor(
                original_embeddings, low_risk_anchor, alpha
            )
            low_risk_preds = compute_risk_prediction(model, low_risk_embeddings, device)
            low_risk_distance = torch.norm(low_risk_embeddings - original_embeddings, dim=1)
            
            # Intervene towards high-risk anchor
            high_risk_embeddings = interpolate_towards_anchor(
                original_embeddings, high_risk_anchor, alpha
            )
            high_risk_preds = compute_risk_prediction(model, high_risk_embeddings, device)
            high_risk_distance = torch.norm(high_risk_embeddings - original_embeddings, dim=1)
            
            # Record results
            for i, pid in enumerate(patient_ids):
                # Low-risk intervention
                results.append({
                    'patient_id': pid,
                    'true_time': true_times[i],
                    'true_event': true_events[i],
                    'alpha': alpha,
                    'direction': 'low_risk',
                    'risk_pred': low_risk_preds[i].cpu().item(),
                    'embedding_distance': low_risk_distance[i].cpu().item()
                })
                
                # High-risk intervention
                results.append({
                    'patient_id': pid,
                    'true_time': true_times[i],
                    'true_event': true_events[i],
                    'alpha': alpha,
                    'direction': 'high_risk',
                    'risk_pred': high_risk_preds[i].cpu().item(),
                    'embedding_distance': high_risk_distance[i].cpu().item()
                })
    
    return pd.DataFrame(results)


def analyze_direction_consistency(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze direction consistency from intervention results.
    
    Args:
        df: Results DataFrame from run_intervention_audit
    
    Returns:
        Dictionary with metrics:
            - monotonic_decrease_rate: % of patients with monotonic risk decrease towards low-risk
            - monotonic_increase_rate: % of patients with monotonic risk increase towards high-risk
            - mean_risk_change_low: Mean risk change at α=1.0 (low-risk direction)
            - mean_risk_change_high: Mean risk change at α=1.0 (high-risk direction)
    """
    metrics = {}
    
    # Get unique patients
    patients = df['patient_id'].unique()
    
    # Check monotonicity for each patient
    monotonic_low = 0
    monotonic_high = 0
    
    for pid in patients:
        patient_df = df[df['patient_id'] == pid].sort_values('alpha')
        
        # Low-risk direction: risk should decrease as α increases
        low_risk_df = patient_df[patient_df['direction'] == 'low_risk']
        risks = low_risk_df['risk_pred'].values
        if len(risks) > 1 and all(risks[i] >= risks[i+1] for i in range(len(risks)-1)):
            monotonic_low += 1
        
        # High-risk direction: risk should increase as α increases
        high_risk_df = patient_df[patient_df['direction'] == 'high_risk']
        risks = high_risk_df['risk_pred'].values
        if len(risks) > 1 and all(risks[i] <= risks[i+1] for i in range(len(risks)-1)):
            monotonic_high += 1
    
    metrics['monotonic_decrease_rate'] = monotonic_low / len(patients)
    metrics['monotonic_increase_rate'] = monotonic_high / len(patients)
    
    # Mean risk change at full intervention (α=1.0)
    alpha_1_df = df[df['alpha'] == 1.0]
    alpha_0_df = df[df['alpha'] == 0.0]
    
    for pid in patients:
        low_risk_0 = alpha_0_df[(alpha_0_df['patient_id'] == pid) & 
                                 (alpha_0_df['direction'] == 'low_risk')]['risk_pred'].values
        low_risk_1 = alpha_1_df[(alpha_1_df['patient_id'] == pid) & 
                                 (alpha_1_df['direction'] == 'low_risk')]['risk_pred'].values
        
        if len(low_risk_0) > 0 and len(low_risk_1) > 0:
            metrics.setdefault('risk_changes_low', []).append(low_risk_1[0] - low_risk_0[0])
        
        high_risk_0 = alpha_0_df[(alpha_0_df['patient_id'] == pid) & 
                                  (alpha_0_df['direction'] == 'high_risk')]['risk_pred'].values
        high_risk_1 = alpha_1_df[(alpha_1_df['patient_id'] == pid) & 
                                  (alpha_1_df['direction'] == 'high_risk')]['risk_pred'].values
        
        if len(high_risk_0) > 0 and len(high_risk_1) > 0:
            metrics.setdefault('risk_changes_high', []).append(high_risk_1[0] - high_risk_0[0])
    
    metrics['mean_risk_change_low'] = np.mean(metrics['risk_changes_low'])
    metrics['mean_risk_change_high'] = np.mean(metrics['risk_changes_high'])
    metrics['std_risk_change_low'] = np.std(metrics['risk_changes_low'])
    metrics['std_risk_change_high'] = np.std(metrics['risk_changes_high'])
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="E4: Continuous Intervention Audit")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--study', type=str, required=True,
                        choices=['blca', 'ucec', 'lusc', 'kirc', 'hnsc', 'skcm'],
                        help='Cancer type')
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help='Fold number')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--alphas', type=str, default='0,0.25,0.5,0.75,1.0',
                        help='Comma-separated intervention strengths')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Compute device')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Parse alphas
    alphas = [float(a) for a in args.alphas.split(',')]
    
    print("="*80)
    print("E4: Continuous Intervention Audit for DCT-Reg")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Study: {args.study}, Fold: {args.fold}")
    print(f"Intervention strengths: {alphas}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Run audit
    results_df = run_intervention_audit(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        study=args.study,
        fold=args.fold,
        alphas=alphas,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    
    # Analyze direction consistency
    metrics = analyze_direction_consistency(results_df)
    print("\n" + "="*80)
    print("Direction Consistency Analysis")
    print("="*80)
    print(f"Monotonic decrease rate (towards low-risk): {metrics['monotonic_decrease_rate']:.2%}")
    print(f"Monotonic increase rate (towards high-risk): {metrics['monotonic_increase_rate']:.2%}")
    print(f"\nMean risk change at α=1.0:")
    print(f"  Towards low-risk:  {metrics['mean_risk_change_low']:.4f} ± {metrics['std_risk_change_low']:.4f}")
    print(f"  Towards high-risk: {metrics['mean_risk_change_high']:.4f} ± {metrics['std_risk_change_high']:.4f}")
    print("="*80)
    
    # Save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("E4: Continuous Intervention Audit Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Study: {args.study}, Fold: {args.fold}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Alphas: {alphas}\n\n")
        f.write("Direction Consistency Metrics:\n")
        for key, value in metrics.items():
            if not key.endswith('_changes'):
                f.write(f"  {key}: {value}\n")
    
    print(f"✅ Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
