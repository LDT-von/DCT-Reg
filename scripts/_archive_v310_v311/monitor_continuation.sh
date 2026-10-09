#!/usr/bin/env bash
# Continuation: monitor fold1, then run folds 2-4 sequentially
LOG_BASE="/data1/DCT-Reg/logs/v312_5fold_aligned"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/dct_v312_blca_uni.yaml"
RESULT_DIR="/data1/DCT-Reg/results/dct_v312_blca_uni_fold0_aligned/blca"
FOLD1_PID=4059933

echo "Monitoring fold1 PID $FOLD1_PID..."

while kill -0 $FOLD1_PID 2>/dev/null; do
    sleep 60
    LAST=$(grep "val cindex" "${LOG_BASE}/fold1/fold1.log" 2>/dev/null | tail -1)
    if [[ -n "$LAST" ]]; then
        EP=$(echo "$LAST" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
        CI=$(echo "$LAST" | sed 's/.*cindex=\([0-9.]*\).*/\1/')
        echo "[$(date '+%H:%M:%S')] fold1 epoch $EP  cindex=$CI"
    fi
done

wait $FOLD1_PID
EXIT=$?
echo ""
echo "=== FOLD 1 DONE (exit=$EXIT) ==="
grep "best cindex" "${LOG_BASE}/fold1/fold1.log" 2>/dev/null | head -3
echo ""

echo "Starting folds 2-3-4..."
for fold in 2 3 4; do
    LOG="${LOG_BASE}/fold${fold}/fold${fold}.log"

    echo ""
    echo ">>> FOLD $fold / 4  [$(date '+%H:%M:%S')]"

    export PYTHONPATH=/data1/DCT-Reg
    export DCT_REG_CACHE=/data1/DCT-Reg/.cache
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    "${PYTHON}" -m survot_rank.cli train \
        --config "${CONFIG}" \
        --set max_epochs=30 \
        --set k_start="${fold}" --set k_end=$((fold + 1)) \
        --set gpu=0 \
        --set specific_simple=dct_v312_blca_uni_fold0_aligned \
        --set results_dir="${RESULT_DIR}" \
        --set batch_size=32 \
        --set slot_num_wsi=16 --set slot_num_omics=16 --set slot_iters=10 \
        > "${LOG}" 2>&1 &

    TRAIN_PID=$!
    echo "    PID: ${TRAIN_PID}"

    while kill -0 ${TRAIN_PID} 2>/dev/null; do
        sleep 30
        LAST=$(grep "val cindex" "${LOG}" 2>/dev/null | tail -1)
        if [[ -n "$LAST" ]]; then
            EP=$(echo "$LAST" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
            CI=$(echo "$LAST" | sed 's/.*cindex=\([0-9.]*\).*/\1/')
            echo "    [$(date '+%H:%M:%S')] epoch ${EP}  cindex=${CI}"
        fi
    done

    wait ${TRAIN_PID}
    EXIT_CODE=$?
    echo ""
    echo ">>> FOLD $fold done (exit=${EXIT_CODE})"
    grep "best cindex" "${LOG}" 2>/dev/null | head -3
    echo ""
done

echo "============================================"
echo "ALL DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
for fold in 0 1 2 3 4; do
    L="${LOG_BASE}/fold${fold}/fold${fold}.log"
    BEST=$(grep -E "best cindex" "${L}" 2>/dev/null | head -1 || echo "Fold $fold: FAILED")
    echo "Fold $fold: ${BEST}"
done
echo ""
VALUES=$(for fold in 0 1 2 3 4; do
    grep -E "best cindex" "${LOG_BASE}/fold${fold}/fold${fold}.log" 2>/dev/null | head -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/'
done)
echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "Mean ± Std: %.4f ± %.4f\n", sum/n, sqrt(sumsq/n-(sum/n)^2)}'
