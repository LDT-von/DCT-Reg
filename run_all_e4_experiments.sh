#!/bin/bash
#
# 一键运行所有E4 Continuous Intervention Audit实验
# 
# 实验配置：
# - 3个变体：Direction Only, IPCW Only, Full Model
# - 每个变体5个folds (0-4)
# - BLCA数据集
# - α值：0, 0.1, 0.2, ..., 1.0 (11个点)
#

set -e  # Exit on error

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python3
SCRIPT=/data1/DCT-Reg/scripts/e4_audit_adapted.py
OUTPUT_DIR=/data1/DCT-Reg/results/e4_intervention_audit
DATA_ROOT=/data1/DCT-Reg/data
STUDY=blca
DEVICE=cuda:0
BATCH_SIZE=16

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "E4 Continuous Intervention Audit"
echo "=========================================="
echo "Study: $STUDY"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 变体配置
declare -A VARIANTS=(
    ["direction_only"]="/data1/DCT-Reg/results/dct_v3.10_experiments/robust/direction_only/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep"
    ["ipcw_only"]="/data1/DCT-Reg/results/dct_v3.10_experiments/robust/ipcw_only/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_ipcw_only_blca_50ep"
    ["full"]="/data1/DCT-Reg/results/dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep"
)

# 跟踪成功和失败
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_EXPERIMENTS=0

# 运行所有实验
for variant in direction_only ipcw_only full; do
    base_path="${VARIANTS[$variant]}"
    
    echo "=========================================="
    echo "Variant: $variant"
    echo "Base path: $base_path"
    echo "=========================================="
    
    for fold in 0 1 2 3 4; do
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        
        # 查找checkpoint文件
        checkpoint="${base_path}/model_best_s${fold}.pth"
        
        if [ ! -f "$checkpoint" ]; then
            echo "⚠️  Checkpoint not found: $checkpoint"
            echo "   Trying alternative naming..."
            
            # 尝试其他可能的命名
            alt_checkpoint="${base_path}/s_${fold}_checkpoint.pt"
            if [ -f "$alt_checkpoint" ]; then
                checkpoint="$alt_checkpoint"
            else
                echo "❌ SKIP: $variant fold $fold (checkpoint not found)"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            fi
        fi
        
        # 输出文件
        output_csv="${OUTPUT_DIR}/${variant}_${STUDY}_fold${fold}.csv"
        
        echo ""
        echo "Running: $variant, fold $fold"
        echo "  Checkpoint: $checkpoint"
        echo "  Output: $output_csv"
        
        # 运行E4审计
        if $PYTHON "$SCRIPT" \
            --checkpoint "$checkpoint" \
            --study "$STUDY" \
            --fold "$fold" \
            --output "$output_csv" \
            --alphas "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
            --device "$DEVICE" \
            --batch-size "$BATCH_SIZE" \
            --data-root "$DATA_ROOT"; then
            
            echo "✅ SUCCESS: $variant fold $fold"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ FAILED: $variant fold $fold"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "E4 Experiments Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# 如果所有实验都成功，运行汇总分析
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "Generating aggregated analysis..."
    
    # 创建简单的汇总脚本
    $PYTHON - <<'PYEOF'
import json
import pandas as pd
from pathlib import Path

output_dir = Path("/data1/DCT-Reg/results/e4_intervention_audit")

# 收集所有summary文件
summaries = []
for summary_file in output_dir.glob("*_summary.json"):
    with open(summary_file) as f:
        data = json.load(f)
        variant = summary_file.stem.split("_")[0]
        data['variant'] = variant
        summaries.append(data)

if summaries:
    # 创建汇总表
    summary_rows = []
    for s in summaries:
        metrics = s['metrics']
        summary_rows.append({
            'variant': s['variant'],
            'fold': s['fold'],
            'n_patients': metrics.get('n_patients', 0),
            'monotonic_decrease_rate': metrics.get('monotonic_decrease_rate', 0),
            'monotonic_increase_rate': metrics.get('monotonic_increase_rate', 0),
            'mean_risk_change_low': metrics.get('mean_risk_change_low', 0),
            'mean_risk_change_high': metrics.get('mean_risk_change_high', 0),
        })
    
    df = pd.DataFrame(summary_rows)
    
    # 保存汇总
    summary_path = output_dir / "e4_all_variants_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"✅ Aggregated summary saved to {summary_path}")
    
    # 打印统计
    print("\n" + "="*80)
    print("E4 Aggregate Results by Variant")
    print("="*80)
    
    for variant in df['variant'].unique():
        variant_df = df[df['variant'] == variant]
        print(f"\n{variant.upper()}:")
        print(f"  Monotonic decrease rate (towards low-risk): {variant_df['monotonic_decrease_rate'].mean():.2%} ± {variant_df['monotonic_decrease_rate'].std():.2%}")
        print(f"  Monotonic increase rate (towards high-risk): {variant_df['monotonic_increase_rate'].mean():.2%} ± {variant_df['monotonic_increase_rate'].std():.2%}")
        print(f"  Mean risk change (low-risk):  {variant_df['mean_risk_change_low'].mean():+.4f} ± {variant_df['mean_risk_change_low'].std():.4f}")
        print(f"  Mean risk change (high-risk): {variant_df['mean_risk_change_high'].mean():+.4f} ± {variant_df['mean_risk_change_high'].std():.4f}")
    
    print("="*80)
else:
    print("⚠️  No summary files found")
PYEOF
    
fi

echo ""
echo "🎉 All done! Check results in: $OUTPUT_DIR"
