#!/bin/bash
# Batch E4 Audit Runner for DCT v3.10 Ablation Experiments

set -e

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python3
SCRIPT=/data1/DCT-Reg/scripts/e4_audit_working.py
BASE_DIR=/data1/DCT-Reg/results/dct_v3.10_experiments/robust
CANCER=blca
DEVICE=cuda:0

echo "========================================="
echo "E4 Direction Consistency Audit - Batch Run"
echo "========================================="
echo ""

# Function to run E4 audit for a single checkpoint
run_e4_audit() {
    local variant=$1
    local fold=$2
    local checkpoint=$3
    
    echo ""
    echo ">>> Running E4 audit: ${variant} Fold ${fold}"
    echo "    Checkpoint: ${checkpoint}"
    
    if [ ! -f "$checkpoint" ]; then
        echo "    ERROR: Checkpoint not found!"
        return 1
    fi
    
    $PYTHON $SCRIPT \
        --checkpoint "$checkpoint" \
        --fold $fold \
        --cancer $CANCER \
        --device $DEVICE \
        --output "results/e4_audits/e4_audit_${variant}_fold${fold}.csv"
    
    echo "    ✓ Completed"
}

# Create output directory
mkdir -p results/e4_audits

echo "Starting batch E4 audits..."
echo ""

# Track success/failure
total=0
success=0
failed=0

# Direction Only variant
VARIANT="direction_only"
VARIANT_DIR="$BASE_DIR/$VARIANT/$CANCER/$CANCER/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep"

echo "=== Direction Only Variant ==="
for fold in 0 1 2 3 4; do
    checkpoint="$VARIANT_DIR/model_best_s${fold}.pth"
    total=$((total + 1))
    
    if run_e4_audit "$VARIANT" "$fold" "$checkpoint"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
done

# IPCW Only variant
VARIANT="ipcw_only"
VARIANT_DIR="$BASE_DIR/$VARIANT/$CANCER/$CANCER/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_ipcw_only_blca_50ep"

echo ""
echo "=== IPCW Only Variant ==="
for fold in 0 1 2 3 4; do
    checkpoint="$VARIANT_DIR/model_best_s${fold}.pth"
    total=$((total + 1))
    
    if run_e4_audit "$VARIANT" "$fold" "$checkpoint"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
done

# Full Model variant
VARIANT="full_model"
VARIANT_DIR="$BASE_DIR/$VARIANT/$CANCER/$CANCER/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep"

echo ""
echo "=== Full Model Variant ==="
for fold in 0 1 2 3 4; do
    checkpoint="$VARIANT_DIR/model_best_s${fold}.pth"
    total=$((total + 1))
    
    if run_e4_audit "$VARIANT" "$fold" "$checkpoint"; then
        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi
done

# Print summary
echo ""
echo "========================================="
echo "Batch E4 Audit Summary"
echo "========================================="
echo "Total runs:     $total"
echo "Successful:     $success"
echo "Failed:         $failed"
echo ""
echo "Results saved in: results/e4_audits/"
echo "========================================="

# Aggregate results
echo ""
echo "Aggregating results..."
$PYTHON <<EOF
import pandas as pd
import json
from pathlib import Path

results_dir = Path('results/e4_audits')

# Collect all results
all_results = []

variants = ['direction_only', 'ipcw_only', 'full_model']
folds = [0, 1, 2, 3, 4]

for variant in variants:
    for fold in folds:
        csv_path = results_dir / f'e4_audit_{variant}_fold{fold}.csv'
        json_path = results_dir / f'e4_audit_{variant}_fold{fold}.json'
        
        if csv_path.exists() and json_path.exists():
            df = pd.read_csv(csv_path)
            with open(json_path) as f:
                metadata = json.load(f)
            
            all_results.append({
                'variant': variant,
                'fold': fold,
                'n_samples': metadata['n_processed'],
                'anchor_distance': metadata['anchor_distance'],
                'mean_risk': df['risk_score'].mean(),
                'std_risk': df['risk_score'].std(),
            })

# Create summary dataframe
summary_df = pd.DataFrame(all_results)

# Group by variant
print("\n=== E4 Audit Summary by Variant ===\n")
for variant in variants:
    variant_data = summary_df[summary_df['variant'] == variant]
    if len(variant_data) > 0:
        print(f"{variant.upper()}:")
        print(f"  Anchor distance: {variant_data['anchor_distance'].mean():.4f} ± {variant_data['anchor_distance'].std():.4f}")
        print(f"  Mean risk score: {variant_data['mean_risk'].mean():.4f} ± {variant_data['mean_risk'].std():.4f}")
        print(f"  Folds completed: {len(variant_data)}/5")
        print()

# Save summary
summary_df.to_csv(results_dir / 'e4_audit_summary.csv', index=False)
print("Summary saved to: results/e4_audits/e4_audit_summary.csv")
EOF

echo ""
echo "All E4 audits completed!"
