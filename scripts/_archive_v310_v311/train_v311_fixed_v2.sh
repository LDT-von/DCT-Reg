#!/bin/bash
# 顺序训练 5 folds (修复版 v2: per-modality diversity)
# 写日志到 logs/v311_blca_uni_fixed_v2/
# 关键：强制只用 GPU 0（CUDA_VISIBLE_DEVICES=0）

set -e
cd /data1/DCT-Reg

export PYTHONPATH=/data1/DCT-Reg

# ⚠️ 强制只用 GPU 0（nvidia-smi 编号 0），避免误用 GPU 1
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python

# Smoke test first (1 epoch, fold 0 only)
if [ "${SMOKE_ONLY:-0}" = "1" ]; then
    echo "==========================================="
    echo "SMOKE TEST: 1 epoch fold 0 (GPU 0 only)"
    echo "==========================================="
    LOG_FILE="/data1/DCT-Reg/logs/v311_blca_uni_fixed_v2/smoke.log"
    $PYTHON -m survot_rank.cli train \
        --config configs/dct_v311_blca_uni_v2.yaml \
        --set max_epochs=1 \
        --set k_start=0 \
        --set k_end=1 \
        --set gpu=0 \
        2>&1 | tee "$LOG_FILE"
    exit 0
fi

# Full 5-fold training (skip already-done fold if checkpoint exists)
SKIP_DONE=${SKIP_DONE:-0}
for fold in 0 1 2 3 4; do
    echo ""
    echo "==========================================="
    echo "Training Fold ${fold} (per-modality diversity, GPU 0 only)"
    echo "==========================================="

    LOG_FILE="/data1/DCT-Reg/logs/v311_blca_uni_fixed_v2/fold${fold}.log"
    MASTER_LOG="/data1/DCT-Reg/logs/v311_blca_uni_fixed_v2/master.log"

    # Check if already done (model_best exists + best cindex in log)
    if [ "$SKIP_DONE" = "1" ] && [ -f "$LOG_FILE" ] && grep -q "best cindex" "$LOG_FILE"; then
        echo "  Fold ${fold} already completed, skipping"
        continue
    fi

    echo "[$(date)] Starting fold ${fold}" >> "$MASTER_LOG"

    $PYTHON -m survot_rank.cli train \
        --config configs/dct_v311_blca_uni_v2.yaml \
        --set k_start=$fold \
        --set k_end=$((fold + 1)) \
        --set gpu=0 \
        2>&1 | tee "$LOG_FILE"

    echo "[$(date)] Fold ${fold} done" >> "$MASTER_LOG"

    BEST=$(grep "best cindex" "$LOG_FILE" | tail -1)
    if [ -n "$BEST" ]; then
        echo "  $BEST"
    fi
done

echo ""
echo "==========================================="
echo "ALL 5 FOLDS COMPLETED"
echo "==========================================="
