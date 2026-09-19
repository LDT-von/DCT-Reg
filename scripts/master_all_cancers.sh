#!/bin/bash
# ============================================
# Master scheduler: GPU 0 串行跑所有 cancer × 5 fold
# ============================================
# 顺序（v3.13 double_w ablation）：
#   brca:       fold 1 2 3 4   (fold 0 已完成 best cindex=0.7961)
#   coadread:   fold 0 1 2 3 4
#   hnsc:       fold 0 1 2 3 4
#   kirc:       fold 0 1 2 3 4
#   luad:       fold 0 1 2 3 4
#   lusc:       fold 0 1 2 3 4
#   skcm:       fold 0 1 2 3 4
#   stad:       fold 0 1 2 3 4
#   ucec:       fold 0 1 2 3 4
# GPU 1 正在跑你自己的 v311 blca 6-cancer，我们不动它。
# ============================================

set -uo pipefail

PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
LOG_BASE="/data1/DCT-Reg/logs"
RESULT_BASE="/data1/DCT-Reg/results/dct_v313_ablation_double_w"
GPU=0

CANCERS=(brca coadread hnsc kirc luad lusc skcm stad ucec)

# 已完成的 brca fold 列表（手工跳过）
BRCA_FOLDS_DONE=(0)

# ----- helpers -----
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

wait_for_gpu_free() {
  local timeout=7200  # 最多等 2 小时
  local waited=0
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | awk -v g=$GPU '$1{print}' | grep -q .; do
    # GPU $GPU 上还有进程在跑 → 等
    sleep 30
    waited=$((waited + 30))
    if [ $waited -ge $timeout ]; then
      log "WARN: 等 GPU $GPU 空闲超过 ${timeout}s，继续"
      return 0
    fi
  done
  return 0
}

run_one() {
  local cancer="$1"
  local fold="$2"
  local LOG_DIR="${LOG_BASE}/v313_abl_double_w_${cancer}/fold${fold}"
  local LOG_FILE="${LOG_DIR}/fold${fold}.log"
  mkdir -p "${LOG_DIR}"

  # Skip if already has best cindex
  if grep -q "best cindex" "${LOG_FILE}" 2>/dev/null; then
    log "[${cancer} fold${fold}] 已完成 → $(grep "best cindex" "${LOG_FILE}" | tail -1)"
    return 0
  fi

  local CONFIG="/data1/DCT-Reg/configs/dct_v313_${cancer}_uni.yaml"
  if [ ! -f "$CONFIG" ]; then
    log "[${cancer} fold${fold}] config 不存在: $CONFIG → 跳过"
    return 0
  fi

  local SPECIFIC="dct_v313_brca_uni"
  local RESULT_DIR="/data1/DCT-Reg/results/dct_v313_${cancer}_uni/blca"

  log "[${cancer} fold${fold}] 启动 GPU=${GPU}"
  log "    config:  $CONFIG"
  log "    log:     $LOG_FILE"
  log "    result:  $RESULT_DIR"

  wait_for_gpu_free

  PYTHONPATH=/data1/DCT-Reg CUDA_VISIBLE_DEVICES=$GPU \
    "$PYTHON" -m survot_rank.cli train \
      --config "$CONFIG" \
      --set k_start="$fold" \
      --set k_end=$((fold + 1)) \
      --set max_epochs=30 \
      --set gpu=0 \
      --set specific_simple="$SPECIFIC" \
      --set results_dir="$RESULT_DIR" \
      > "$LOG_FILE" 2>&1 &

  local TRAIN_PID=$!
  log "    PID=$TRAIN_PID"

  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 90
    local LAST
    LAST=$(grep "val cindex" "$LOG_FILE" 2>/dev/null | tail -1)
    if [ -n "$LAST" ]; then
      local EP CI
      EP=$(echo "$LAST" | sed 's/.*\[Epoch \([0-9]*\)\].*/\1/')
      CI=$(echo "$LAST" | sed 's/.*val cindex=\([0-9.]*\).*/\1/')
      log "    [${cancer} fold${fold}] epoch ${EP}/30 cindex=${CI}"
    fi
  done

  wait $TRAIN_PID || true
  local EXIT=$?
  if [ -n "$(grep "best cindex" "$LOG_FILE" 2>/dev/null)" ]; then
    log "    [${cancer} fold${fold}] ✅ 完成 (exit=$EXIT): $(grep "best cindex" "$LOG_FILE" | tail -1)"
  else
    log "    [${cancer} fold${fold}] ❌ 失败 (exit=$EXIT)；日志末尾："
    tail -10 "$LOG_FILE" | sed 's/^/        /'
  fi
  sleep 5
}

# ----- main -----
log "============================================"
log "MASTER: 9 cancer × 5 fold (GPU 0)"
log "Start: $(date)"
log "GPU 1 被你自己的 v311 blca 跑（不动它）"
log "============================================"

# brca 特殊处理：fold 0 跳过，跑 fold 1-4
for fold in 0 1 2 3 4; do
  if [[ " ${BRCA_FOLDS_DONE[*]} " == *" $fold "* ]]; then
    log "[brca fold${fold}] 已完成 → 跳过"
    continue
  fi
  run_one brca $fold
done

# 其他 8 cancer × 5 fold
for cancer in coadread hnsc kirc luad lusc skcm stad ucec; do
  for fold in 0 1 2 3 4; do
    run_one $cancer $fold
  done
done

log "============================================"
log "MASTER: 全部完成 $(date)"
log "============================================"

# 汇总
log "==== 汇总（best cindex） ===="
for cancer in "${CANCERS[@]}"; do
  VALUES=""
  for fold in 0 1 2 3 4; do
    L="${LOG_BASE}/v313_abl_double_w_${cancer}/fold${fold}/fold${fold}.log"
    V=$(grep "best cindex" "$L" 2>/dev/null | tail -1 | sed 's/.*cindex=\([0-9.]*\).*/\1/')
    VALUES="$VALUES ${V:-NA}"
  done
  log "[$cancer]$VALUES"
  echo "$VALUES" | awk -v c="$cancer" '{sum+=($1=="NA"?0:$1); n+=($1=="NA"?0:1); sumsq+=($1=="NA"?0:$1*$1)} END{if(n>0) printf "    %s Mean ± Std: %.4f ± %.4f (n=%d)\n", c, sum/n, sqrt(sumsq/n-(sum/n)^2), n; else printf "    %s 全失败\n", c}'
done

log "DONE"
