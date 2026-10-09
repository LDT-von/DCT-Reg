#!/bin/bash
# Run UNI v2 5-fold BLCA on GPU 1 in background
set -e
cd /data1/DCT-Reg

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python
LOGDIR=logs
mkdir -p $LOGDIR

NAME=dct_v311_blca_uni_v2_5fold
echo "============================================================"
echo "[$NAME] 5-fold BLCA UNI v2 (per-modality diversity)"
echo "      started: $(date '+%H:%M:%S')"
echo "============================================================"
$PYTHON -m survot_rank.cli train --config configs/dct_v311_blca_uni_v2_5fold.yaml 2>&1 \
    | tee $LOGDIR/${NAME}.log \
    | grep -E "Fold|Epoch.*val|Cindex|FINAL|Best" \
    | tee $LOGDIR/${NAME}_summary.log
echo "[$NAME] DONE: $(date '+%H:%M:%S')"
