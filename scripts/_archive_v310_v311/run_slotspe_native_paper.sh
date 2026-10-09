#!/usr/bin/env bash
# SlotSPE "native" recipe (no top-k filtering) — topk_ratio=1.0 disables slot selection
# SlotSPE native = uses slot attention but with full attention (no top-k), and longer slot iterations
# - 30 epochs, batch=32
# - topk_ratio=1.0 (disable top-k)
# - slot_iters=10 (default)
# - Use BLCA on GPU 0 (KIRC companion on GPU 1)
set -euo pipefail
cd /data1/DCT-Reg/third_party/SlotSPE

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python
export PYTHONPATH=/data1/DCT-Reg/third_party/SlotSPE

STUDY="$1"   # blca or kirc
GPU="$2"     # 0 or 1
shift 2
LOG_DIR="/data1/DCT-Reg/logs/slotspe_${STUDY}_native"
mkdir -p "$LOG_DIR"
RESULTS_DIR="/data1/DCT-Reg/results/slotspe_${STUDY}_native"
mkdir -p "$RESULTS_DIR"
MASTER_LOG="$LOG_DIR/master.log"

if [ "$STUDY" = "blca" ]; then
    DATA_ROOT="/data/CPathPatchFeature/blca/uni/pt_files"
elif [ "$STUDY" = "kirc" ]; then
    DATA_ROOT="/data/CPathPatchFeature/kirc/uni/pt_files"
else
    echo "Unknown study: $STUDY"; exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date '+%F %T')] launch SlotSPE $STUDY native (no topk) on GPU $GPU" >> "$MASTER_LOG"

for fold in 0 1 2 3 4; do
    echo "==== $STUDY fold $fold ===="
    LOG_FILE="$LOG_DIR/fold${fold}.log"
    echo "[$(date)] Starting fold $fold" >> "$MASTER_LOG"
    $PYTHON survival.py \
        --study "$STUDY" \
        --data_root_dir "$DATA_ROOT" \
        --data_path /data1/dataset_csv \
        --results_dir "$RESULTS_DIR" \
        --specific_simple "slotspe_${STUDY}_native_iter10_seed3" \
        --n_classes 4 \
        --num_patches 4096 \
        --encoding_dim 1024 \
        --wsi_projection_dim 256 \
        --max_epochs 30 \
        --batch_size 32 \
        --lr 5e-4 \
        --reg 1e-3 \
        --seed 3 \
        --opt adam \
        --rna_format Pathways \
        --label_col survival_months_dss \
        --bag_loss nll_surv \
        --signature combine \
        --slot_num_wsi 8 \
        --slot_num_omics 8 \
        --slot_iters 10 \
        --temperature 0.01 \
        --topk_ratio 1.0 \
        --top_k_method parallel_topk_st \
        --gpu 0 \
        --k_start $fold --k_end $((fold+1)) \
        --which_splits 5fold \
        --scheduler cosine 2>&1 | tee "$LOG_FILE"
    echo "[$(date)] fold $fold done" >> "$MASTER_LOG"
done
echo "[$(date '+%F %T')] $STUDY native all folds done" >> "$MASTER_LOG"
