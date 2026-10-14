#!/usr/bin/env bash
# E050 driver: Independent DCT v2.1, BLCA 5-fold seed3 legacy_val, GPU 1.
# Logs to /tmp/e050_v21_driver.log; each fold logs to its own results_dir.
# Run from /data1/DCT-Reg-e050 (worktree @ 76f8f25), never from /data1/DCT-Reg main.

set -uo pipefail

WORKTREE=/data1/DCT-Reg-e050
CONFIG="/data1/DCT-Reg/实验档案/结果/E050_v21_20261014/dct_v21_blca_legacy_val.yaml"
RESULTS=/data1/DCT-Reg/实验档案/结果/E050_v21_20261014
DRIVER_LOG=/tmp/e050_v21_driver.log

# Python env: must use dct env, never default python3 (RTX 5090 sm_120 trap).
PY=/home/ubuntu/.conda/envs/dct/bin/python3.10
if [[ ! -x "$PY" ]]; then
    echo "FATAL: dct env not found at $PY" >&2
    exit 2
fi

mkdir -p "$RESULTS/logs" "$(dirname "$DRIVER_LOG")"

{
    echo "============================================================"
    echo "E050 v2.1 driver start $(date -Iseconds)"
    echo "worktree=$WORKTREE  config=$CONFIG"
    echo "============================================================"
} > "$DRIVER_LOG"

cd "$WORKTREE" || { echo "FATAL: cannot cd to $WORKTREE" >&2; exit 3; }

EXIT_SUMMARY=0
for FOLD in 0 1 2 3 4; do
    FOLD_LOG="$RESULTS/logs/fold${FOLD}.log"
    echo "[$(date +%H:%M:%S)] === E050 fold ${FOLD} start ===" >> "$DRIVER_LOG"
    export PYTHONPATH="$WORKTREE"
    export DCT_REG_CACHE="$WORKTREE/.cache"
    export CUDA_VISIBLE_DEVICES=1
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    "$PY" -m survot_rank.cli train \
        --config "$CONFIG" \
        --set k_start="$FOLD" --set k_end=$((FOLD + 1)) \
        --set specific_simple="E050_dct_v21_blca_legacy_val_fold${FOLD}" \
        > "$FOLD_LOG" 2>&1
    RC=$?
    echo "[$(date +%H:%M:%S)] === E050 fold ${FOLD} rc=${RC} ===" >> "$DRIVER_LOG"
    if [[ $RC -ne 0 ]]; then
        echo "  fold ${FOLD} FAILED rc=${RC}, see $FOLD_LOG" >> "$DRIVER_LOG"
        EXIT_SUMMARY=$((EXIT_SUMMARY + 1))
        # continue to next fold; one failure does not abort pilot
    else
        echo "  fold ${FOLD} OK" >> "$DRIVER_LOG"
    fi
done

echo "[$(date +%H:%M:%S)] === E050 driver done, failed=${EXIT_SUMMARY} ===" >> "$DRIVER_LOG"
exit $EXIT_SUMMARY
