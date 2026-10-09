#!/bin/bash
# 一键按顺序运行4个实验，使用支持 RTX 5090 (sm_120) 的 trisurv 环境
set -e
cd /data1/DCT-Reg

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python
LOGDIR=logs
mkdir -p $LOGDIR

echo "============================================================"
echo "[1/4] UNI2-h merged diversity"
echo "      started: $(date '+%H:%M:%S')"
echo "============================================================"
$PYTHON -m survot_rank.cli train --config configs/dct_v311_blca_uni2h.yaml 2>&1 | tee $LOGDIR/dct_v311_blca_uni2h.log | grep -E "Fold|Epoch.*val|Cindex|FINAL|Best" | tee $LOGDIR/dct_v311_blca_uni2h_summary.log
echo "[1/4] DONE: $(date '+%H:%M:%S')"

echo "============================================================"
echo "[2/4] UNI2-h per-modality diversity (fixed v2)"
echo "      started: $(date '+%H:%M:%S')"
echo "============================================================"
$PYTHON -m survot_rank.cli train --config configs/dct_v311_blca_uni2h_v2.yaml 2>&1 | tee $LOGDIR/dct_v311_blca_uni2h_v2.log | grep -E "Fold|Epoch.*val|Cindex|FINAL|Best" | tee $LOGDIR/dct_v311_blca_uni2h_v2_summary.log
echo "[2/4] DONE: $(date '+%H:%M:%S')"

echo "============================================================"
echo "[3/4] UNI merged diversity"
echo "      started: $(date '+%H:%M:%S')"
echo "============================================================"
$PYTHON -m survot_rank.cli train --config configs/dct_v311_blca_uni.yaml 2>&1 | tee $LOGDIR/dct_v311_blca_uni.log | grep -E "Fold|Epoch.*val|Cindex|FINAL|Best" | tee $LOGDIR/dct_v311_blca_uni_summary.log
echo "[3/4] DONE: $(date '+%H:%M:%S')"

echo "============================================================"
echo "[4/4] UNI per-modality diversity (fixed v2)"
echo "      started: $(date '+%H:%M:%S')"
echo "============================================================"
$PYTHON -m survot_rank.cli train --config configs/dct_v311_blca_uni_v2.yaml 2>&1 | tee $LOGDIR/dct_v311_blca_uni_v2.log | grep -E "Fold|Epoch.*val|Cindex|FINAL|Best" | tee $LOGDIR/dct_v311_blca_uni_v2_summary.log
echo "[4/4] DONE: $(date '+%H:%M:%S')"

echo "ALL EXPERIMENTS COMPLETE: $(date '+%H:%M:%S')"
