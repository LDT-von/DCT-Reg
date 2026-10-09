#!/bin/bash
# 启动 v3.13 UNI BLCA p2048 30ep 5-fold (GPU 0)
# 配置: DCT v3.13 transport-aware pathway reconstruction

set -e

cd /data1/DCT-Reg

mkdir -p logs
LOG="logs/v313_uni_blca_p2048_30ep_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "启动: DCT v3.13 / UNI / BLCA / p2048 / 30ep"
echo "GPU: 0"
echo "Log: $LOG"
echo "k_start=0, k_end=5 (5-fold)"
echo "============================================"

PYTHONPATH=/data1/DCT-Reg CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
  --config configs/dct_v313_blca_uni.yaml \
  --set k_start=0 \
  --set k_end=5 \
  --set num_patches=2048 \
  --set batch_size=8 \
  --set gpu=0 \
  --set num_workers=4 \
  --set specific_simple=dct_v313_uni_p2048_blca_5fold_30ep \
  --set results_dir=/data1/DCT-Reg/results/dct_v3.13_uni_blca_p2048_30ep/blca \
  2>&1 | tee "$LOG"
