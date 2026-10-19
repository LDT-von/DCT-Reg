#!/usr/bin/env bash
# E051 Arm1: v2.1 baseline, BLCA 5-fold seed3 legacy_val, GPU 1.
# 对照基准，同 E050 配置（lr=0.0005, reg=0.0005）。
set -uo pipefail

PY=/home/ubuntu/.conda/envs/dct/bin/python3.10
RESULTS=/data1/DCT-Reg/实验档案/结果/E051_v21_20261015/control
DRIVER_LOG=/tmp/e051_v21_driver.log
CONFIG=configs/dct_v21_blca_uni2h.yaml

if [[ ! -x "$PY" ]]; then
    echo "FATAL: dct env not at $PY" >&2; exit 2
fi
mkdir -p "$RESULTS/logs"

echo "============================================================" >> "$DRIVER_LOG"
echo "E051 v2.1 arm start $(date -Iseconds)" >> "$DRIVER_LOG"
echo "============================================================" >> "$DRIVER_LOG"

EXIT_SUMMARY=0
for FOLD in 0 1 2 3 4; do
    FOLD_LOG="$RESULTS/logs/fold${FOLD}.log"
    echo "[$(date +%H:%M:%S)] === E051 v2.1 fold ${FOLD} start ===" >> "$DRIVER_LOG"
    export PYTHONPATH="/data1/DCT-Reg"
    export CUDA_VISIBLE_DEVICES=1
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    "$PY" -m survot_rank.cli train \
        --config "$CONFIG" \
        --set k_start="$FOLD" --set k_end=$((FOLD + 1)) \
        --set train.results_dir="$RESULTS" \
        --set train.gpu=1 \
        --set train.evaluation_protocol=legacy_val \
        --set split.split_dir="/data1/dataset_csv/splits/5fold_uni2h/blca" \
        --set specific_simple="E051_dct_v21_blca_legacy_val_fold${FOLD}" \
        > "$FOLD_LOG" 2>&1
    RC=$?
    echo "[$(date +%H:%M:%S)] === E051 v2.1 fold ${FOLD} rc=${RC} ===" >> "$DRIVER_LOG"
    if [[ $RC -ne 0 ]]; then
        echo "  fold ${FOLD} FAILED rc=${RC}, see $FOLD_LOG" >> "$DRIVER_LOG"
        EXIT_SUMMARY=$((EXIT_SUMMARY + 1))
    else
        echo "  fold ${FOLD} OK" >> "$DRIVER_LOG"
    fi
done

echo "[$(date +%H:%M:%S)] === E051 v2.1 driver done, failed=${EXIT_SUMMARY} ===" >> "$DRIVER_LOG"
exit $EXIT_SUMMARY
