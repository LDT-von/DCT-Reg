#!/usr/bin/env bash
# Run ONE fold, blocking (no background processes), log to foldN/foldN.log
FOLD=${1:-1}
LOG="/data1/DCT-Reg/logs/v312_5fold_aligned/fold${FOLD}/fold${FOLD}.log"

echo "Starting Fold ${FOLD} at $(date)"
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
    --set specific_simple=dct_v312_blca_uni_fold0_aligned \
    --set results_dir=/data1/DCT-Reg/results/dct_v312_blca_uni_fold0_aligned/blca \
    --set batch_size=32 \
    --set slot_num_wsi=16 --set slot_num_omics=16 --set slot_iters=10 \
    > "${LOG}" 2>&1

echo "Fold ${FOLD} done (exit=$?) at $(date)"
grep "best cindex" "${LOG}" 2>/dev/null | head -3
