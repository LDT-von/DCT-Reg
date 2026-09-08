#!/bin/bash
# 验证 best epoch checkpoint 是否都存在

BASE_DIR="/data1/DCT-Reg/results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep"

echo "============================================"
echo "验证 BLCA Best Epoch Checkpoints"
echo "============================================"
echo ""

all_exist=true

for fold in {0..4}; do
    ckpt="${BASE_DIR}/model_best_s${fold}.pth"
    if [ -f "$ckpt" ]; then
        size=$(ls -lh "$ckpt" | awk '{print $5}')
        echo "✓ Fold $fold: $ckpt ($size)"
    else
        echo "✗ Fold $fold: MISSING - $ckpt"
        all_exist=false
    fi
done

echo ""
echo "============================================"
if [ "$all_exist" = true ]; then
    echo "✅ 所有 checkpoint 都存在！"
    echo ""
    echo "可以运行审计了："
    echo "  cd /data1/DCT-Reg"
    echo "  conda activate survot"
    echo "  bash scripts/audit_best_checkpoints_wrapper.sh"
else
    echo "❌ 有些 checkpoint 缺失"
fi
echo "============================================"
