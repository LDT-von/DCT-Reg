#!/usr/bin/env bash
# Rerun ONLY the failed folds of double_w: fold2, fold3, fold4.
# fold0/fold1 already finished successfully.
# GPU split: fold2(GPU0), fold3(GPU1), fold4(GPU0) -- sequential.

set -e

LOG_ROOT="/data1/DCT-Reg/logs"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/dct_v313_blca_uni.yaml"
TAG="double_w"
LOG_BASE="${LOG_ROOT}/v313_abl_${TAG}"
RESULT_DIR="/data1/DCT-Reg/results/dct_v313_ablation_${TAG}/blca"
EXTRA_ARGS=(--set dct_v313_lambda_reconstruction_scale=2.0)

mkdir -p "${LOG_BASE}"

run_fold() {
    local fold="$1"
    local LOG_DIR="${LOG_BASE}/fold${fold}"
    mkdir -p "${LOG_DIR}"
    local LOG_FILE="${LOG_DIR}/fold${fold}.log"

    # Skip if already completed
    if grep -q "best cindex" "${LOG_FILE}" 2>/dev/null; then
        echo "[fold ${fold}] already has best cindex, skipping"
        grep "best cindex" "${LOG_FILE}" | tail -1
        return 0
    fi

    local GPU_ID=$(( fold % 2 ))

    echo ""
    echo ">>> [${TAG}] FOLD ${fold} / 4  [$(date '+%H:%M:%S')]"
    echo "    Log: ${LOG_FILE}"
    echo "    GPU: ${GPU_ID}"

    export PYTHONPATH=/data1/DCT-Reg
    export DCT_REG_CACHE=/data1/DCT-Reg/.cache
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    "${PYTHON}" -m survot_rank.cli train \
        --config "${CONFIG}" \
        --set max_epochs=30 \
        --set k_start="${fold}" \
        --set k_end=$((fold + 1)) \
        --set gpu=0 \
        --set specific_simple="dct_v313_ablation_${TAG}" \
        --set results_dir="${RESULT_DIR}" \
        "${EXTRA_ARGS[@]}" \
        > "${LOG_FILE}" 2>&1 &

    local TRAIN_PID=$!
    echo "    Training PID: ${TRAIN_PID}  (GPU ${GPU_ID})"

    while kill -0 ${TRAIN_PID} 2>/dev/null; do
        sleep 60
        local LAST_VAL
        LAST_VAL=$(grep "val cindex" "${LOG_FILE}" 2>/dev/null | tail -1)
        if [[ -n "${LAST_VAL}" ]]; then
            local EP CI
            EP=$(echo "${LAST_VAL}" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
            CI=$(echo "${LAST_VAL}" | sed 's/.*cindex=\([0-9.]*\).*/\1/')
            echo "    [$(date '+%H:%M:%S')] fold${fold} epoch ${EP}  cindex=${CI}"
        fi
    done

    wait ${TRAIN_PID}
    local EXIT_CODE=$?
    echo ""
    echo ">>> [${TAG}] FOLD ${fold} finished (exit=${EXIT_CODE}) [$(date '+%H:%M:%S')]"
    grep "best cindex" "${LOG_FILE}" 2>/dev/null | head -1
    echo "    Last val:"
    grep "val cindex" "${LOG_FILE}" 2>/dev/null | tail -1
    echo ""
}

# Run sequentially: fold2(GPU0), fold3(GPU1), fold4(GPU0)
run_fold 2
run_fold 3
run_fold 4

echo ""
echo "=== [${TAG}] FINAL SUMMARY ==="
for fold in 0 1 2 3 4; do
    L="${LOG_BASE}/fold${fold}/fold${fold}.log"
    BEST=$(grep "best cindex" "${L}" 2>/dev/null | head -1 || echo "Fold $fold: FAILED")
    echo "Fold $fold: ${BEST}"
done
VALUES=$(for fold in 0 1 2 3 4; do
    grep "best cindex" "${LOG_BASE}/fold${fold}/fold${fold}.log" 2>/dev/null | head -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/'
done)
echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "[%s] Mean ± Std: %.4f ± %.4f (n=%d)\n", "'"${TAG}"'", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'

echo ""
echo "============================================"
echo "RERUN DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
