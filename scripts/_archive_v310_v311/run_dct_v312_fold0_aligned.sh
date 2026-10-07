#!/usr/bin/env bash
# 30-epoch fold 0 run for the SlotSPE-aligned v3.12 config.
set -e
cd /data1/DCT-Reg

export PYTHONPATH=/data1/DCT-Reg
export DCT_REG_CACHE=/data1/DCT-Reg/.cache
mkdir -p "${DCT_REG_CACHE}"

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python

LOG_DIR=/data1/DCT-Reg/logs/v312_fold0_aligned
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fold0.log"

RESULT_DIR=/data1/DCT-Reg/results/dct_v312_blca_uni_fold0_aligned
mkdir -p "${RESULT_DIR}"

echo "==========================================="
echo "v3.12 FOLD 0: 30 epochs, aligned params (slot=16/16, iters=10, b=32)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log:   ${LOG_FILE}"
echo "Out:   ${RESULT_DIR}"
echo "==========================================="

$PYTHON -m survot_rank.cli train \
    --config configs/dct_v312_blca_uni.yaml \
    --set max_epochs=30 \
    --set k_start=0 \
    --set k_end=1 \
    --set gpu=0 \
    --set specific_simple=dct_v312_blca_uni_fold0_aligned \
    --set results_dir="${RESULT_DIR}/blca" \
    --set batch_size=32 \
    --set slot_num_wsi=16 \
    --set slot_num_omics=16 \
    --set slot_iters=10 \
    2>&1 | tee "${LOG_FILE}"

echo "==========================================="
echo "v3.12 FOLD 0 DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="
