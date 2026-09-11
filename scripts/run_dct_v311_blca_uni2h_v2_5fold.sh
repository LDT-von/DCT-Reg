#!/usr/bin/env bash
# Launch v3.11 BLCA on UNI2-h 5-fold sequentially (single GPU=1).
# Logs to logs/dct_v311_blca_uni2h_v2_5fold.log
set -euo pipefail
cd /data1/DCT-Reg

CONFIG=configs/dct_v311_blca_uni2h.yaml
LOG=/data1/DCT-Reg/logs/dct_v311_blca_uni2h_v2_5fold.log
mkdir -p /data1/DCT-Reg/logs

# Reserve GPU=1 (this process). Smoke test first to catch yaml errors fast.
exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] launch v3.11 BLCA uni2-h 5-fold (k=0..4)"

python -m survot_rank.cli train \
  --config "$CONFIG" \
  --set=k_start=0 --set=k_end=1 \
  --gpu 1

for K in 1 2 3 4; do
  NEXT=$((K+1))
  echo "[$(date '+%F %T')] ---- fold $K done, launching fold $NEXT ----"
  python -m survot_rank.cli train \
    --config "$CONFIG" \
    --set=k_start=$K --set=k_end=$NEXT \
    --gpu 1
done

echo "[$(date '+%F %T')] all folds done"
