#!/usr/bin/env bash
# Sequential 5-fold run on GPU 0, one fold at a time.
# Resume from first incomplete fold.

LOG_BASE="/data1/DCT-Reg/logs/v313_5fold_aligned"
RESULT_DIR="/data1/DCT-Reg/results/dct_v313_blca_uni_5fold_aligned/blca"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/dct_v313_blca_uni.yaml"

mkdir -p "${LOG_BASE}"

echo "============================================"
echo "BLCA v3.13 UNI — 5-Fold Sequential (GPU 0)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Determine starting fold (first fold without a best cindex line).
START_FOLD=""
for fold in 0 1 2 3 4; do
    LOG="${LOG_BASE}/fold${fold}/fold${fold}.log"
    if [[ ! -f "${LOG}" ]] || ! grep -q "best cindex" "${LOG}" 2>/dev/null; then
        START_FOLD=$fold
        break
    fi
done
if [[ -z "${START_FOLD}" ]]; then
    echo "All five folds already contain a best-cindex record; nothing to resume."
    exit 0
fi
echo ">>> Resuming from fold ${START_FOLD}"
echo ""

for fold in $(seq $START_FOLD 4); do
    LOG_DIR="${LOG_BASE}/fold${fold}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/fold${fold}.log"

    echo ""
    echo ">>> FOLD $fold / 4  [$(date '+%H:%M:%S')]"
    echo "    Log: ${LOG_FILE}"

    export PYTHONPATH=/data1/DCT-Reg
    export DCT_REG_CACHE=/data1/DCT-Reg/.cache
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    "${PYTHON}" -m survot_rank.cli train \
        --config "${CONFIG}" \
        --set max_epochs=30 \
        --set k_start="${fold}" \
        --set k_end=$((fold + 1)) \
        --set gpu=0 \
        --set specific_simple=dct_v313_blca_uni_5fold_aligned \
        --set results_dir="${RESULT_DIR}" \
        --set batch_size=32 \
        --set slot_num_wsi=16 \
        --set slot_num_omics=16 \
        --set slot_iters=10 \
        > "${LOG_FILE}" 2>&1 &

    TRAIN_PID=$!
    echo "    Training PID: ${TRAIN_PID}  (GPU 0)"

    while kill -0 ${TRAIN_PID} 2>/dev/null; do
        sleep 30
        LAST_VAL=$(grep "val cindex" "${LOG_FILE}" 2>/dev/null | tail -1)
        if [[ -n "${LAST_VAL}" ]]; then
            EP=$(echo "${LAST_VAL}" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
            CI=$(echo "${LAST_VAL}" | sed 's/.*cindex=\([0-9.]*\).*/\1/')
            echo "    [$(date '+%H:%M:%S')] epoch ${EP}  cindex=${CI}"
        fi
    done

    wait ${TRAIN_PID}
    EXIT_CODE=$?
    echo ""
    echo ">>> FOLD $fold finished (exit=${EXIT_CODE}) [$(date '+%H:%M:%S')]"
    grep -E "best cindex|best c-index" "${LOG_FILE}" 2>/dev/null | head -3
    echo "    Last val:"
    grep "val cindex" "${LOG_FILE}" 2>/dev/null | tail -1
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
    BEST=$(grep -E "best cindex|best c-index" "${L}" 2>/dev/null | head -1 || echo "Fold $fold: FAILED")
    echo "Fold $fold: ${BEST}"
done
echo ""
VALUES=$(for fold in 0 1 2 3 4; do
    grep -E "best cindex|best c-index" "/data1/DCT-Reg/logs/v313_5fold_aligned/fold${fold}/fold${fold}.log" 2>/dev/null | head -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/'
done)
echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "Mean ± Std: %.4f ± %.4f (n=%d)\n", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'
