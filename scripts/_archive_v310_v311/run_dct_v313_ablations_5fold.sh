#!/usr/bin/env bash
# Ablation matrix: 3 variants × 5 folds on GPU 0, sequential.
#
# Variants (each one changes v3.13 reconstruction recipe):
#   A) no_cross: --dct_v313_disable_cross_reconstruction
#      (only omics self-recon survives; cross WSI→Omics transport removed)
#   B) no_self:  --dct_v313_disable_self_reconstruction
#      (only cross transport-aware recon survives; omics self-recon removed)
#   C) double_w: --dct_v313_lambda_reconstruction_scale=2.0
#      (both recons retained, but weight doubled to 0.20)
#
# v3.13 full (both recons, weight 0.10) is already in
#   /data1/DCT-Reg/logs/v313_5fold_p4096_aligned/
#   Mean=0.7005 ± 0.0333  (n=5)
#
# Each variant is a separate 5-fold sweep; logs/results live under
#   /data1/DCT-Reg/logs/v313_abl_<tag>/
#   /data1/DCT-Reg/results/dct_v313_ablation_<tag>/blca
#
# Total wall-time: ~7.5 h (3 variants × 2.5 h/variant, GPU 0 only).

LOG_ROOT="/data1/DCT-Reg/logs"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
CONFIG="configs/dct_v313_blca_uni.yaml"

# --set accepts flags with explicit =value; boolean flags use "true"/"false".
DCT_V313_DISABLE_CROSS=(--set dct_v313_disable_cross_reconstruction=true)
DCT_V313_DISABLE_SELF=(--set dct_v313_disable_self_reconstruction=true)

run_variant() {
    local TAG="$1"
    local NAME="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    local LOG_BASE="${LOG_ROOT}/v313_abl_${TAG}"
    local RESULT_DIR="/data1/DCT-Reg/results/dct_v313_ablation_${TAG}/blca"

    mkdir -p "${LOG_BASE}"
    mkdir -p "${RESULT_DIR}"

    echo ""
    echo "============================================"
    echo "ABLATION: ${NAME}  (tag=${TAG})"
    echo "Log:   ${LOG_BASE}"
    echo "Res:   ${RESULT_DIR}"
    echo "Args:  ${EXTRA_ARGS[*]:-(none)}"
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    # Determine starting fold (resume support).
    local START_FOLD=""
    for fold in 0 1 2 3 4; do
        local LOG="${LOG_BASE}/fold${fold}/fold${fold}.log"
        if [[ ! -f "${LOG}" ]] || ! grep -q "best cindex" "${LOG}" 2>/dev/null; then
            START_FOLD=$fold
            break
        fi
    done
    if [[ -z "${START_FOLD}" ]]; then
        echo "All 5 folds already contain best-cindex; skipping variant."
        return 0
    fi
    echo ">>> Resuming variant from fold ${START_FOLD}"

    for fold in $(seq $START_FOLD 4); do
        local LOG_DIR="${LOG_BASE}/fold${fold}"
        mkdir -p "${LOG_DIR}"
        local LOG_FILE="${LOG_DIR}/fold${fold}.log"

        echo ""
        echo ">>> [${TAG}] FOLD $fold / 4  [$(date '+%H:%M:%S')]"
        echo "    Log: ${LOG_FILE}"

        # GPU assignment: even folds (0,2,4) → GPU 0; odd folds (1,3) → GPU 1
        local GPU_ID=$(( fold % 2 ))
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
                local EP
                EP=$(echo "${LAST_VAL}" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
                local CI
                CI=$(echo "${LAST_VAL}" | sed 's/.*cindex=\([0-9.]*\).*/\1/')
                echo "    [$(date '+%H:%M:%S')] epoch ${EP}  cindex=${CI}"
            fi
        done

        wait ${TRAIN_PID}
        local EXIT_CODE=$?
        echo ""
        echo ">>> [${TAG}] FOLD $fold finished (exit=${EXIT_CODE}) [$(date '+%H:%M:%S')]"
        grep -E "best cindex|best c-index" "${LOG_FILE}" 2>/dev/null | head -3
        echo "    Last val:"
        grep "val cindex" "${LOG_FILE}" 2>/dev/null | tail -1
        echo ""
    done

    # Per-variant summary
    echo ""
    echo "=== [${TAG}] SUMMARY ==="
    for fold in 0 1 2 3 4; do
        local L="${LOG_BASE}/fold${fold}/fold${fold}.log"
        local BEST
        BEST=$(grep -E "best cindex|best c-index" "${L}" 2>/dev/null | head -1 || echo "Fold $fold: FAILED")
        echo "Fold $fold: ${BEST}"
    done
    local VALUES
    VALUES=$(for fold in 0 1 2 3 4; do
        grep -E "best cindex|best c-index" "${LOG_BASE}/fold${fold}/fold${fold}.log" 2>/dev/null | head -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/'
    done)
    echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "[%s] Mean ± Std: %.4f ± %.4f (n=%d)\n", "'"${TAG}"'", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'
}

# A: kill cross-recon, keep self-recon
run_variant "no_cross" "v3.13 minus cross-recon" "--set" "dct_v313_disable_cross_reconstruction=true"
# B: kill self-recon, keep cross-recon
run_variant "no_self"  "v3.13 minus self-recon"  "--set" "dct_v313_disable_self_reconstruction=true"
# C: double weight
run_variant "double_w" "v3.13 reconstruction weight ×2" \
    "--set" "dct_v313_lambda_reconstruction_scale=2.0"

echo ""
echo "============================================"
echo "ALL ABLATIONS DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
