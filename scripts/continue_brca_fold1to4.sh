#!/bin/bash
# 续跑 BRCA fold1-4（fold0 已完成，best cindex=0.7961 @epoch16）
# 单卡 GPU 0 串行跑 fold1, fold2, fold3, fold4

set -euo pipefail

PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
LOG_BASE="/data1/DCT-Reg/logs/v313_abl_double_w_brca"
RESULT_DIR="/data1/DCT-Reg/results/dct_v313_brca_uni/blca"
CONFIG="/data1/DCT-Reg/configs/dct_v313_brca_uni.yaml"
GPU="${1:-0}"

echo "============================================"
echo "续跑 BRCA fold1-4 on GPU $GPU"
echo "fold0 best cindex=0.7961 @epoch16 (DONE)"
echo "Start: $(date)"
echo "============================================"

for fold in 1 2 3 4; do
  LOG_DIR="${LOG_BASE}/fold${fold}"
  LOG_FILE="${LOG_DIR}/fold${fold}.log"
  mkdir -p "${LOG_DIR}"

  # 如果已跑完则跳过
  if grep -q "best cindex" "${LOG_FILE}" 2>/dev/null; then
    echo "[fold $fold] 已完成，跳过："
    grep "best cindex" "${LOG_FILE}" | tail -1
    continue
  fi

  echo ""
  echo ">>> [fold $fold] 启动 [$(date '+%H:%M:%S')] GPU=$GPU"
  echo "    log: $LOG_FILE"

  PYTHONPATH=/data1/DCT-Reg CUDA_VISIBLE_DEVICES=$GPU \
    "$PYTHON" -m survot_rank.cli train \
      --config "$CONFIG" \
      --set k_start="$fold" \
      --set k_end=$((fold + 1)) \
      --set max_epochs=30 \
      --set gpu=0 \
      > "$LOG_FILE" 2>&1 &

  TRAIN_PID=$!
  echo "    PID: $TRAIN_PID"

  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    LAST=$(grep "val cindex" "$LOG_FILE" 2>/dev/null | tail -1)
    if [ -n "$LAST" ]; then
      EP=$(echo "$LAST" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
      CI=$(echo "$LAST" | sed 's/.*val cindex=\([0-9.]*\).*/\1/')
      echo "    [$(date '+%H:%M:%S')] fold$fold epoch $EP cindex=$CI"
    fi
  done

  wait $TRAIN_PID || true
  EXIT=$?
  echo "    fold$fold 完成 (exit=$EXIT) [$(date '+%H:%M:%S')]"
  grep "best cindex" "$LOG_FILE" 2>/dev/null | tail -1
  sleep 5
done

echo ""
echo "============================================"
echo "BRCA fold1-4 全部完成: $(date)"
echo "============================================"

VALUES=""
for fold in 0 1 2 3 4; do
  L="${LOG_BASE}/fold${fold}/fold${fold}.log"
  V=$(grep "best cindex" "$L" 2>/dev/null | tail -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/')
  VALUES="$VALUES $V"
done
echo "BRCA fold0-4 best cindex:$VALUES"
echo "$VALUES" | awk '{sum+=$1; sumsq+=$1*$1; n++} END{printf "BRCA Mean ± Std: %.4f ± %.4f (n=%d)\n", sum/n, sqrt(sumsq/n-(sum/n)^2), n}'
