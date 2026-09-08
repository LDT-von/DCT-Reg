#!/bin/bash
# 对 final_50ep_old 的 best checkpoints 运行审计
# 这些 checkpoint 对应 BLCA C-index 0.7208 的模型

set -e

REPO_ROOT="/data1/DCT-Reg"
CONFIG="${REPO_ROOT}/configs/dct_v310_directional_regularized_transport.yaml"
CHECKPOINT_DIR="${REPO_ROOT}/results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep"
OUTPUT_BASE="${REPO_ROOT}/results/audit_best_checkpoints_blca"

mkdir -p "${OUTPUT_BASE}"

echo "================================================================================从 final_50ep_old 审计 BLCA best checkpoints
================================================================================

Checkpoint dir: ${CHECKPOINT_DIR}
Output dir:     ${OUTPUT_BASE}
"

# 检查 checkpoint 是否存在
for fold in {0..4}; do
    ckpt="${CHECKPOINT_DIR}/model_best_s${fold}.pth"
    if [ ! -f "$ckpt" ]; then
        echo "⚠️  Warning: checkpoint not found: $ckpt"
    fi
done

echo ""
echo "Running audits for 5 folds..."
echo ""

# 为每个 fold 运行审计
for fold in {0..4}; do
    echo "----------------------------------------"
    echo "Fold ${fold}"
    echo "----------------------------------------"
    
    ckpt="${CHECKPOINT_DIR}/model_best_s${fold}.pth"
    output_dir="${OUTPUT_BASE}/fold_${fold}"
    
    if [ ! -f "$ckpt" ]; then
        echo "✗ Skipping fold ${fold}: checkpoint not found"
        continue
    fi
    
    mkdir -p "${output_dir}"
    
    # 运行审计（假设已激活 Python 环境）
    python "${REPO_ROOT}/scripts/audit_dct_reg.py" audit \
        --config "${CONFIG}" \
        --checkpoint "${ckpt}" \
        --fold ${fold} \
        --output-dir "${output_dir}" \
        --gpu 0 \
        --set study=blca \
        --set data_root_dir=/data1/TCGA-UNI2-h-features \
        --set which_splits=5fold_uni2h
    
    if [ $? -eq 0 ]; then
        echo "✓ Fold ${fold} completed"
    else
        echo "✗ Fold ${fold} failed"
    fi
    
    echo ""
done

echo "================================================================================
汇总
================================================================================

Output directory: ${OUTPUT_BASE}

To analyze results:
  python scripts/extract_mechanism_audit.py ${OUTPUT_BASE}/fold_0/audit_metrics.json

"
