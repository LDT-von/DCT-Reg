#!/bin/bash
# Run double_w ablation on GPU 0 for 9 remaining cancer types
# Each cancer: 5 fold × 30 epoch ≈ 35 min/cancer × 9 ≈ 5.2 hours total

set -euo pipefail

GPU="${1:-0}"
PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="/data1/DCT-Reg/logs/rerun"
RESULT_BASE="/data1/DCT-Reg/results/dct_v313_ablation_double_w"

CANCERS=(brca coadread hnsc kirc luad lusc skcm stad ucec)

echo "============================================"
echo "double_w on GPU $GPU for ${#CANCERS[@]} cancers"
echo "Start: $(date)"
echo "============================================"

for cancer in "${CANCERS[@]}"; do
  CONFIG="/data1/DCT-Reg/configs/dct_v313_${cancer}_uni.yaml"
  if [[ ! -f "$CONFIG" ]]; then
    echo "[$cancer] ⚠️  config not found: $CONFIG — skip"
    continue
  fi

  echo ""
  echo ">>> [$cancer] 5 fold  (GPU $GPU)  [$(date +%H:%M:%S)]"

  for fold in 0 1 2 3 4; do
    LOG_DIR="/data1/DCT-Reg/logs/v313_abl_double_w_${cancer}/fold${fold}"
    LOG_FILE="${LOG_DIR}/fold${fold}.log"
    mkdir -p "$LOG_DIR"

    # Wait if another fold is running on the same GPU
    while true; do
      BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
      if [[ "$BUSY" -eq 0 ]]; then
        break
      fi
      sleep 30
    done

    echo "    fold $fold / 5  [$(date +%H:%M:%S)]  GPU=$GPU"
    echo "    Log: $LOG_FILE"

    PYTHONPATH=/data1/DCT-Reg CUDA_VISIBLE_DEVICES=$GPU \
      "$PYTHON" -m survot_rank.cli train \
        --config "$CONFIG" \
        --set k_start="$fold" \
        --set k_end=$((fold + 1)) \
        --set max_epochs=30 \
        --set gpu="$GPU" \
        > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo "    PID: ${TRAIN_PID}"

    while kill -0 ${TRAIN_PID} 2>/dev/null; do
      sleep 60
      LAST_VAL=$(grep "val cindex" "$LOG_FILE" 2>/dev/null | tail -1)
      if [[ -n "${LAST_VAL}" ]]; then
        EP=$(echo "${LAST_VAL}" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
        CI=$(echo "${LAST_VAL}" | sed 's/.*val cindex=\([0-9.]*\).*/\1/')
        echo "    [$(date +%H:%M:%S)] fold $fold epoch $EP  cindex=$CI"
      fi
    done

    echo "    fold $fold DONE  [$(date +%H:%M:%S)]"
    sleep 5
  done

  echo ">>> [$cancer] ALL 5 FOLDS DONE"

  # Print summary
  VALUES=""
  for fold in 0 1 2 3 4; do
    LOG_FILE="/data1/DCT-Reg/logs/v313_abl_double_w_${cancer}/fold${fold}/fold${fold}.log"
    VAL=$(grep "best cindex" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/')
    VALUES="${VALUES} ${VAL}"
  done
  echo "[$cancer] Summary: $VALUES"
  echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "[$cancer] Mean +- Std: %.4f +- %.4f (n=%d)\n", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'

done

echo ""
echo "============================================"
echo "ALL CANCERS DONE: $(date)"
echo "============================================"
