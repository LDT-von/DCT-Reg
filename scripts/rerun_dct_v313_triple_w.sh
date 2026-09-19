#!/usr/bin/env bash
# Ablation: continue scaling up reconstruction weight.
# Compares to double_w (scale=2.0, mean=0.7132).
# triple_w: scale=3.0 -> effective_weight = 0.10 * 3.0 = 0.30
# GPU split: triple_w runs on GPU 0 sequentially across all 5 folds.

set -e

LOG_ROOT="/data1/DCT-Reg/logs"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/dct_v313_blca_uni.yaml"
TAG="triple_w"
LOG_BASE="${LOG_ROOT}/v313_abl_${TAG}"
RESULT_DIR="/data1/DCT-Reg/results/dct_v313_ablation_${TAG}/blca"
EXTRA_ARGS=(--set dct_v313_lambda_reconstruction_scale=3.0)

mkdir -p "${LOG_BASE}"

run_fold() {
    local fold="$1"
    local gpu="$2"
    local LOG_DIR="${LOG_BASE}/fold${fold}"
    mkdir -p "${LOG_DIR}"
    local LOG_FILE="${LOG_DIR}/fold${fold}.log"

    if grep -q "best cindex" "${LOG_FILE}" 2>/dev/null; then
        echo "[fold ${fold}] already done"
        grep "best cindex" "${LOG_FILE}" | tail -1
        return 0
    fi

    echo ""
    echo ">>> [${TAG}] FOLD ${fold} / 4  [$(date '+%H:%M:%S')]  GPU=${gpu}"
    echo "    Log: ${LOG_FILE}"

    export PYTHONPATH=/data1/DCT-Reg
    export DCT_REG_CACHE=/data1/DCT-Reg/.cache
    export CUDA_VISIBLE_DEVICES=${gpu}
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    "${PYTHON}" -m survot_rank.cli train \
        --config "${CONFIG}" \
        --set max_epochs=30 \
        --set k_start="${fold}" \
        --set k_end=$((fold + 1)) \
        --set gpu="${gpu}" \
        --set specific_simple="dct_v313_ablation_${TAG}" \
        --set results_dir="${RESULT_DIR}" \
        "${EXTRA_ARGS[@]}" \
        > "${LOG_FILE}" 2>&1 &

    local TRAIN_PID=$!
    echo "    PID: ${TRAIN_PID}"

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
    echo ""
}

for f in 0 1 2 3 4; do
    run_fold "$f" 0
done

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
echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "[%s] Mean +- Std: %.4f +- %.4f (n=%d)\n", "'"${TAG}"'", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'

echo ""
echo "============================================"
echo "TRIPLE_W DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
