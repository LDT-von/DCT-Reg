#!/bin/bash
# ============================================================================
# Launch DCT v3.10 UNI all-10 queue (10 cancers)
# 顺序: blca → brca → coadread → hnsc → kirc → luad → lusc → skcm → stad → ucec, 每癌种 5 fold
# 输出: results/dct_v3.10_uni_all10/final/<cancer>/
# ============================================================================

set -euo pipefail
cd /data1/DCT-Reg

export PYTHON_BIN=/home/ubuntu/.conda/envs/trisurv/bin/python
export PYTHONUNBUFFERED=1

NAME="v310_uni_all10"
LOG_DIR=/data1/DCT-Reg/logs
mkdir -p $LOG_DIR

START_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${NAME}_${START_TS}.log"

echo "============================================================"
echo "[$NAME] 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[$NAME] log: $LOG_FILE"
echo "[$NAME] python: $PYTHON_BIN"
echo "[$NAME] GPU: ${GPU:-0}"
echo "[$NAME] 队列: 10 癌种顺序串行 × 5 fold × 30 epoch"
echo "============================================================"

$PYTHON_BIN scripts/run_v310_uni_all10.py run \
    --data-root /data/CPathPatchFeature \
    --data-csv-root /data1/dataset_csv \
    --gpu "${GPU:-0}" \
    --python "$PYTHON_BIN" 2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo "[$NAME] 退出码: ${PIPESTATUS[0]}"
echo "[$NAME] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
