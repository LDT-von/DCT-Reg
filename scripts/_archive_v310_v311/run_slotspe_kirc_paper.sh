#!/usr/bin/env bash
# SlotSPE KIRC 5-fold paper-reproduction
# - 30 epochs per fold, batch=32 (paper), slot_iters=10, lr=5e-4, opt=adam
# - slot_num_wsi=8, slot_num_omics=8 (consistent with paper)
# - uni features (SlotSPE original; not uni2-h which we don't have for SlotSPE anyway)
# - omic NOT missing (有 WSI + Omics 完整配对)
# - CUDA 1 (KIRC companion to BLCA on GPU 0)
set -euo pipefail
cd /data1/DCT-Reg/third_party/SlotSPE

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python
export PYTHONPATH=/data1/DCT-Reg/third_party/SlotSPE
export CUDA_VISIBLE_DEVICES=1

LOG_DIR=/data1/DCT-Reg/logs/slotspe_kirc_paper
mkdir -p "$LOG_DIR"

RESULTS_DIR=/data1/DCT-Reg/results/slotspe_kirc_paper
mkdir -p "$RESULTS_DIR"

MASTER_LOG="$LOG_DIR/master.log"
echo "[$(date '+%F %T')] launch SlotSPE KIRC 5-fold (paper params)" >> "$MASTER_LOG"

# Full 5-fold run
for fold in 0 1 2 3 4; do
    echo "==== fold $fold ===="
    LOG_FILE="$LOG_DIR/fold${fold}.log"
    echo "[$(date)] Starting fold $fold" >> "$MASTER_LOG"

    $PYTHON survival.py \
        --study kirc \
        --data_root_dir /data/CPathPatchFeature/kirc/uni/pt_files \
        --data_path /data1/dataset_csv \
        --results_dir "$RESULTS_DIR" \
        --specific_simple "slotspe_kirc_paper_iter10_temp0.01_topk0.25_seed3" \
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
        --topk_ratio 0.25 \
        --top_k_method parallel_topk_st \
        --gpu 0 \
        --k_start $fold --k_end $((fold+1)) \
        --which_splits 5fold \
        --scheduler cosine 2>&1 | tee "$LOG_FILE"

    echo "[$(date)] fold $fold done" >> "$MASTER_LOG"
done

echo "[$(date '+%F %T')] all 5 folds done" >> "$MASTER_LOG"
echo ""
echo "=== FINAL SUMMARY ==="
echo "Results: $RESULTS_DIR"
echo "Logs:    $LOG_DIR"