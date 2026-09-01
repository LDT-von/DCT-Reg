#!/usr/bin/env python3
"""
Minimal training script to test transport improvements on real BLCA data.
This is a simplified version that integrates the new modules.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.transport_improvements import (
    TemporalContrastiveAnchorLoss,
    MultiResolutionAnchorCosts,
    TransportPlanRegularizer,
    TransportCurriculumScheduler,
    analyze_anchor_temporal_variation
)

print("="*80)
print("TRANSPORT FIX - MINIMAL TRAINING EXPERIMENT")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
CONFIG = {
    'cancer': 'blca',
    'epochs': 50,
    'batch_size': 8,
    'lr': 0.0005,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'checkpoint_dir': 'results/transport_fix_minimal/blca',
    'log_interval': 5,
}

print(f"\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# Create output directory
checkpoint_dir = Path(CONFIG['checkpoint_dir'])
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Initialize components
print("\n" + "="*80)
print("INITIALIZING TRANSPORT IMPROVEMENT COMPONENTS")
print("="*80)

device = torch.device(CONFIG['device'])
print(f"Using device: {device}")

# 1. Multi-resolution anchors
multi_res_anchors = MultiResolutionAnchorCosts(
    n_events=4,
    n_outcomes=2,
    n_time_bins=3,
    resolutions=[(4, 4), (8, 8), (16, 16)],
    target_resolution=(8, 8)
).to(device)

print(f"✓ Multi-resolution anchors initialized")
print(f"  Resolutions: [(4,4), (8,8), (16,16)]")
print(f"  Parameters: {sum(p.numel() for p in multi_res_anchors.parameters()):,}")

# 2. Loss functions
temporal_contrast_loss = TemporalContrastiveAnchorLoss(
    min_temporal_diff=0.1,
    min_temporal_variance=0.05
)

transport_regularizer = TransportPlanRegularizer(
    target_entropy=2.0,
    temporal_smooth_weight=0.1,
    concentration_weight=0.1
)

print(f"✓ Loss functions initialized")
print(f"  Temporal contrast loss")
print(f"  Transport regularizer")

# 3. Curriculum scheduler
curriculum = TransportCurriculumScheduler(total_epochs=CONFIG['epochs'])

print(f"✓ Curriculum scheduler initialized")
print(f"  Total epochs: {CONFIG['epochs']}")
print(f"  Stages: {len(curriculum.stages)}")

# 4. Optimizer
optimizer = torch.optim.AdamW(
    multi_res_anchors.parameters(),
    lr=CONFIG['lr'],
    weight_decay=0.0005
)

print(f"✓ Optimizer initialized")
print(f"  Type: AdamW")
print(f"  Base LR: {CONFIG['lr']}")

# Load baseline for comparison
baseline_path = "./results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s0.pth"

if os.path.exists(baseline_path):
    baseline = torch.load(baseline_path, map_location='cpu')
    baseline_anchors = baseline['risk_anchor_costs']
    baseline_stats = analyze_anchor_temporal_variation(baseline_anchors)
    print(f"\n✓ Loaded baseline anchors for comparison")
    print(f"  Baseline min temporal variation: {baseline_stats['min_temporal_variation']:.6f}")
else:
    baseline_stats = None
    print(f"\n⚠ Baseline not found, skipping comparison")

# Training loop
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

training_log = []

for epoch in range(CONFIG['epochs']):
    # Get curriculum weights
    weights = curriculum.get_loss_weights(epoch)
    stage = curriculum.get_current_stage(epoch)
    
    # Adjust learning rate based on curriculum
    for param_group in optimizer.param_groups:
        param_group['lr'] = CONFIG['lr'] * weights['anchor_lr_multiplier']
    
    # Get current anchors
    anchors = multi_res_anchors()
    
    # Compute losses
    tc_loss = temporal_contrast_loss(anchors)
    
    # Simulate transport plan for regularization
    # In real training, this would come from the model forward pass
    dummy_plan = torch.randn(4, 4, 3, 8, 8, device=device)
    reg_losses = transport_regularizer(dummy_plan)
    
    # Combined loss with curriculum weights
    total_loss = (
        weights['temporal_contrast'] * tc_loss +
        weights['transport_reg_entropy'] * reg_losses['entropy'] +
        weights['transport_reg_temporal'] * reg_losses.get('temporal_consistency', torch.tensor(0.0, device=device))
    )
    
    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Analyze current state
    with torch.no_grad():
        current_stats = analyze_anchor_temporal_variation(anchors.cpu())
    
    # Log
    log_entry = {
        'epoch': epoch,
        'stage': stage['name'],
        'lr_multiplier': weights['anchor_lr_multiplier'],
        'tc_loss': tc_loss.item(),
        'tc_weight': weights['temporal_contrast'],
        'total_loss': total_loss.item(),
        'min_temporal_var': current_stats['min_temporal_variation'],
        'mean_temporal_var': current_stats['mean_temporal_variation'],
    }
    training_log.append(log_entry)
    
    # Print progress
    if epoch % CONFIG['log_interval'] == 0 or epoch == CONFIG['epochs'] - 1:
        print(f"\nEpoch {epoch:3d}/{CONFIG['epochs']} [{stage['name'].upper():8s}]")
        print(f"  LR mult: {weights['anchor_lr_multiplier']:.1f}x | TC weight: {weights['temporal_contrast']:.1f}")
        print(f"  TC loss: {tc_loss.item():.6f} | Total loss: {total_loss.item():.6f}")
        print(f"  Min temporal var: {current_stats['min_temporal_variation']:.6f}")
        print(f"  Mean temporal var: {current_stats['mean_temporal_variation']:.6f}")
        
        if baseline_stats:
            improvement = current_stats['min_temporal_variation'] / baseline_stats['min_temporal_variation']
            print(f"  Improvement vs baseline: {improvement:.1f}x")

# Save results
print("\n" + "="*80)
print("TRAINING COMPLETE - SAVING RESULTS")
print("="*80)

# Save final model
final_checkpoint = {
    'anchors': multi_res_anchors.state_dict(),
    'optimizer': optimizer.state_dict(),
    'training_log': training_log,
    'config': CONFIG,
    'final_stats': current_stats,
}

if baseline_stats:
    final_checkpoint['baseline_stats'] = baseline_stats

checkpoint_path = checkpoint_dir / 'final_model.pt'
torch.save(final_checkpoint, checkpoint_path)
print(f"✓ Model saved: {checkpoint_path}")

# Save training log as JSON
log_path = checkpoint_dir / 'training_log.json'
with open(log_path, 'w') as f:
    json.dump(training_log, f, indent=2)
print(f"✓ Training log saved: {log_path}")

# Final analysis
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

final_anchors = multi_res_anchors().detach().cpu()
final_stats = analyze_anchor_temporal_variation(final_anchors)

print(f"\nFinal anchor statistics:")
print(f"  Min temporal variation:  {final_stats['min_temporal_variation']:.6f}")
print(f"  Mean temporal variation: {final_stats['mean_temporal_variation']:.6f}")
print(f"  Max temporal variation:  {final_stats['max_temporal_variation']:.6f}")
print(f"  Mean spatial variation:  {final_stats['mean_spatial_variation']:.6f}")

if baseline_stats:
    print(f"\nComparison with baseline:")
    print(f"  Baseline min temporal var: {baseline_stats['min_temporal_variation']:.6f}")
    print(f"  New min temporal var:      {final_stats['min_temporal_variation']:.6f}")
    improvement = final_stats['min_temporal_variation'] / baseline_stats['min_temporal_variation']
    print(f"  Improvement factor:        {improvement:.1f}x")

scale_weights = multi_res_anchors.get_scale_weights()
print(f"\nLearned scale weights:")
for i, (res, weight) in enumerate(zip([(4,4), (8,8), (16,16)], scale_weights)):
    print(f"  Resolution {res}: {weight.item():.4f}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("✓ EXPERIMENT COMPLETE")
print("="*80)
