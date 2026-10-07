#!/bin/bash
# 审计所有固定 anchor 的 5 folds 实验

cd /data1/DCT-Reg

# 定义路径
RESULT_DIR="results_fixed_anchors/blca/SurvOTRank_dct_v310_fixed_anchors/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_fixed_anchors_blca_proof"
AUDIT_SCRIPT="scripts/rerun_audit_with_best_epoch.py"

# 审计所有 5 个 folds
for fold in {0..4}; do
    echo "=========================================="
    echo "审计 Fold $fold - 固定 Anchor 版本"
    echo "=========================================="
    
    # 找到 epoch curve 文件
    EPOCH_CURVE="${RESULT_DIR}/epoch_curve_fold${fold}.csv"
    
    if [ ! -f "$EPOCH_CURVE" ]; then
        echo "❌ 找不到 epoch curve 文件: $EPOCH_CURVE"
        continue
    fi
    
    # 提取 best epoch
    BEST_EPOCH=$(tail -n +2 "$EPOCH_CURVE" | awk -F, '{print $2, $1}' | sort -nr | head -1 | awk '{print $2}')
    echo "✅ Best epoch for fold $fold: $BEST_EPOCH"
    
    # 审计
    python $AUDIT_SCRIPT \
        --result_dir "$RESULT_DIR" \
        --fold $fold \
        --best_epoch $BEST_EPOCH \
        --study blca \
        --output_dir "results_fixed_anchors/audits" \
        2>&1 | tee "logs/audit_fixed_fold${fold}.log"
    
    echo ""
done

echo "=========================================="
echo "✅ 所有 5 folds 审计完成！"
echo "=========================================="

# 汇总结果
python scripts/summarize_fixed_anchor_results.py
