#!/bin/bash
# E4 Audit for Full Model - Proof of Core Idea

set -e

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python3
SCRIPT=/data1/DCT-Reg/scripts/e4_audit_working.py
VARIANT_DIR=/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep
CANCER=blca
DEVICE=cuda:0

echo "========================================="
echo "E4 Audit - Full Model (Direction + IPCW)"
echo "========================================="
echo ""
echo "目标: 证明Full Model有最好的方向一致性"
echo "预期: Full Model应该比Direction Only和IPCW Only都好"
echo ""

# Create output directory
mkdir -p results/e4_audits

# Track success
total=5
success=0
failed=0

# Run E4 audit for all 5 folds
for fold in 0 1 2 3 4; do
    checkpoint="$VARIANT_DIR/model_best_s${fold}.pth"
    
    echo ">>> Running E4 audit: Full Model Fold ${fold}"
    echo "    Checkpoint: ${checkpoint}"
    
    if [ ! -f "$checkpoint" ]; then
        echo "    ERROR: Checkpoint not found!"
        failed=$((failed + 1))
        continue
    fi
    
    if $PYTHON $SCRIPT \
        --checkpoint "$checkpoint" \
        --fold $fold \
        --cancer $CANCER \
        --device $DEVICE \
        --output "results/e4_audits/e4_audit_full_model_fold${fold}.csv"; then
        echo "    ✓ Completed"
        success=$((success + 1))
    else
        echo "    ✗ Failed"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# Print summary
echo "========================================="
echo "E4 Audit Summary - Full Model"
echo "========================================="
echo "Total folds:    $total"
echo "Successful:     $success"
echo "Failed:         $failed"
echo ""

if [ $success -gt 0 ]; then
    echo "Results saved in: results/e4_audits/"
    echo ""
    echo "Aggregating results..."
    
    $PYTHON <<EOF
import pandas as pd
import json
from pathlib import Path
import numpy as np

results_dir = Path('results/e4_audits')

# Collect Full Model results
full_model_results = []
for fold in [0, 1, 2, 3, 4]:
    csv_path = results_dir / f'e4_audit_full_model_fold{fold}.csv'
    json_path = results_dir / f'e4_audit_full_model_fold{fold}.json'
    
    if csv_path.exists() and json_path.exists():
        df = pd.read_csv(csv_path)
        with open(json_path) as f:
            metadata = json.load(f)
        
        full_model_results.append({
            'fold': fold,
            'n_samples': metadata['n_processed'],
            'anchor_distance': metadata['anchor_distance'],
            'mean_risk': df['risk_score'].mean(),
            'std_risk': df['risk_score'].std(),
        })

if full_model_results:
    fm_df = pd.DataFrame(full_model_results)
    
    print("\n=== Full Model E4 Results ===\n")
    print(fm_df.to_string(index=False))
    print()
    print("Aggregate Statistics:")
    print(f"  Anchor Distance: {fm_df['anchor_distance'].mean():.4f} ± {fm_df['anchor_distance'].std():.4f}")
    print(f"  Mean Risk Score: {fm_df['mean_risk'].mean():.4f} ± {fm_df['mean_risk'].std():.4f}")
    print(f"  Risk Consistency: {fm_df['std_risk'].mean():.4f} ± {fm_df['std_risk'].std():.4f}")
    print()
    
    # Compare with Direction Only and IPCW Only if available
    print("\n=== Comparison with Other Variants ===\n")
    
    # Load Direction Only results
    direction_results = []
    for fold in [0, 1, 2, 3, 4]:
        json_path = results_dir / f'e4_audit_direction_only_fold{fold}.json'
        csv_path = results_dir / f'e4_audit_direction_only_fold{fold}.csv'
        if csv_path.exists() and json_path.exists():
            df = pd.read_csv(csv_path)
            with open(json_path) as f:
                metadata = json.load(f)
            direction_results.append({
                'anchor_distance': metadata['anchor_distance'],
                'std_risk': df['risk_score'].std(),
            })
    
    # Load IPCW Only results
    ipcw_results = []
    for fold in [0, 1, 2, 3, 4]:
        json_path = results_dir / f'e4_audit_ipcw_only_fold{fold}.json'
        csv_path = results_dir / f'e4_audit_ipcw_only_fold{fold}.csv'
        if csv_path.exists() and json_path.exists():
            df = pd.read_csv(csv_path)
            with open(json_path) as f:
                metadata = json.load(f)
            ipcw_results.append({
                'anchor_distance': metadata['anchor_distance'],
                'std_risk': df['risk_score'].std(),
            })
    
    # Create comparison table
    comparison = []
    
    if full_model_results:
        comparison.append({
            'Variant': 'Full Model',
            'Risk Consistency': fm_df['std_risk'].mean(),
            'Anchor Distance': fm_df['anchor_distance'].mean(),
            'Winner': ''
        })
    
    if direction_results:
        dir_df = pd.DataFrame(direction_results)
        comparison.append({
            'Variant': 'Direction Only',
            'Risk Consistency': dir_df['std_risk'].mean(),
            'Anchor Distance': dir_df['anchor_distance'].mean(),
            'Winner': ''
        })
    
    if ipcw_results:
        ipcw_df = pd.DataFrame(ipcw_results)
        comparison.append({
            'Variant': 'IPCW Only',
            'Risk Consistency': ipcw_df['std_risk'].mean(),
            'Anchor Distance': ipcw_df['anchor_distance'].mean(),
            'Winner': ''
        })
    
    if len(comparison) > 1:
        comp_df = pd.DataFrame(comparison)
        
        # Mark winner (lowest Risk Consistency = best)
        best_idx = comp_df['Risk Consistency'].idxmin()
        comp_df.loc[best_idx, 'Winner'] = '🏆'
        
        print(comp_df.to_string(index=False))
        print()
        
        # Calculate improvements
        if full_model_results and direction_results:
            improvement_dir = ((dir_df['std_risk'].mean() - fm_df['std_risk'].mean()) / dir_df['std_risk'].mean()) * 100
            print(f"Full Model vs Direction Only: {improvement_dir:+.1f}% consistency improvement")
        
        if full_model_results and ipcw_results:
            improvement_ipcw = ((ipcw_df['std_risk'].mean() - fm_df['std_risk'].mean()) / ipcw_df['std_risk'].mean()) * 100
            print(f"Full Model vs IPCW Only: {improvement_ipcw:+.1f}% consistency improvement")
        
        print()
        print("** Lower Risk Consistency (std) = Better directional consistency **")
    
    # Save results
    fm_df.to_csv(results_dir / 'e4_audit_full_model_summary.csv', index=False)
    print(f"\nFull Model summary saved to: {results_dir / 'e4_audit_full_model_summary.csv'}")
else:
    print("No Full Model results found!")
EOF
    
else
    echo "All E4 audits failed. Please check the errors above."
    exit 1
fi

echo ""
echo "========================================="
echo "E4 Full Model Audit Completed!"
echo "========================================="
