#!/usr/bin/env bash
# Sequential 5-fold, pure log-based: skip folds with "best cindex", track by log file.
set -e

LOG_BASE="/data1/DCT-Reg/logs/v312_5fold_aligned"
RESULT_DIR="/data1/DCT-Reg/results/dct_v312_blca_uni_fold0_aligned/blca"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/dct_v312_blca_uni.yaml"

export PYTHONPATH=/data1/DCT-Reg
export DCT_REG_CACHE=/data1/DCT-Reg/.cache
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "============================================"
echo "BLCA v3.12 UNI — Sequential (log-based)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for fold in 0 1 2 3 4; do
    LOG="${LOG_BASE}/fold${fold}/fold${fold}.log"

    # Skip if already done
    if [[ -f "${LOG}" ]] && grep -q "best cindex" "${LOG}" 2>/dev/null; then
        BEST=$(grep "best cindex" "${LOG}" | head -1)
        echo ">>> FOLD $fold SKIPPED (already done): $BEST"
        continue
    fi

    mkdir -p "${LOG_BASE}/fold${fold}"
    echo "" > "${LOG}"

    echo ""
    echo ">>> FOLD $fold / 4  [$(date '+%H:%M:%S')]"
    echo "    Log: ${LOG}"

    "${PYTHON}" -m survot_rank.cli train \
        --config "${CONFIG}" \
        --set max_epochs=30 \
        --set k_start="${fold}" --set k_end=$((fold + 1)) \
        --set gpu=0 \
        --set specific_simple=dct_v312_blca_uni_fold0_aligned \
        --set results_dir="${RESULT_DIR}" \
        --set batch_size=32 \
        --set slot_num_wsi=16 --set slot_num_omics=16 --set slot_iters=10 \
        > "${LOG}" 2>&1

    EXIT=$?
    echo ""
    echo ">>> FOLD $fold done (exit=${EXIT}) [$(date '+%H:%M:%S')]"
    grep "best cindex" "${LOG}" 2>/dev/null | head -3
    echo ""
done

echo ""
echo "============================================"
echo "ALL 5 FOLDS DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "=== FINAL SUMMARY ==="
for fold in 0 1 2 3 4; do
    L="${LOG_BASE}/fold${fold}/fold${fold}.log"
    if grep -q "best cindex" "${L}" 2>/dev/null; then
        echo "Fold $fold: $(grep 'best cindex' "${L}" | head -1)"
    else
        echo "Fold $fold: FAILED"
    fi
done
echo ""
VALUES=$(for fold in 0 1 2 3 4; do
    grep -E "best cindex" "${LOG_BASE}/fold${fold}/fold${fold}.log" 2>/dev/null | head -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/'
done)
echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "Mean ± Std: %.4f ± %.4f (n=%d)\n", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'
