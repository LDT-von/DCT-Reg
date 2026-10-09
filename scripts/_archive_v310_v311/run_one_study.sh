#!/usr/bin/env bash
# Run ONE fold for any dataset, blocking
# Usage: bash run_one_study.sh <study> <fold>
# Example: bash run_one_study.sh skcm 0

STUDY=${1:-skcm}
FOLD=${2:-0}

LOG_DIR="/data1/DCT-Reg/logs/v312_6datasets/${STUDY}"
RES_DIR="/data1/DCT-Reg/results/dct_v312_${STUDY}_5fold/blca"
LOG="${LOG_DIR}/fold${FOLD}.log"

mkdir -p "$LOG_DIR"

echo "Starting ${STUDY} fold ${FOLD} at $(date)"
echo "Log: ${LOG}"

export PYTHONPATH=/data1/DCT-Reg
export DCT_REG_CACHE=/data1/DCT-Reg/.cache
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

/home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
    --config configs/dct_v312_blca_uni.yaml \
    --set max_epochs=30 \
    --set k_start="${FOLD}" --set k_end=$((FOLD + 1)) \
    --set gpu=0 \
    --set specific_simple="dct_v312_${STUDY}_5fold" \
    --set results_dir="$RES_DIR" \
    --set batch_size=32 \
    --set slot_num_wsi=16 --set slot_num_omics=16 --set slot_iters=10 \
    --set study="$STUDY" \
    > "${LOG}" 2>&1

RC=$?
echo "${STUDY} fold ${FOLD} done (exit=$RC) at $(date)"
grep "best cindex" "${LOG}" 2>/dev/null | head -3
