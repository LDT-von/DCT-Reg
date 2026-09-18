#!/bin/bash
# 启动 v3.10 UNI BLCA p2048 30ep fold 2-4 (GPU 1)
# 配置: DCT v3.10 directional regularized transport (与已跑 fold 0-1 完全一致)

set -e

cd /data1/DCT-Reg

mkdir -p logs
LOG="logs/v310_uni_blca_p2048_30ep_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "启动: DCT v3.10 / UNI / BLCA / p2048 / 30ep"
echo "GPU: 1"
echo "Log: $LOG"
echo "k_start=2, k_end=5 (fold 2,3,4)"
echo "============================================"

PYTHONPATH=/data1/DCT-Reg CUDA_VISIBLE_DEVICES=1 \
/home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli train \
  --config configs/dct_v310_directional_regularized_transport.yaml \
  --set survot_method=dct_v310_directional_regularized_transport \
  --set bag_loss=nll_surv \
  --set max_epochs=30 \
  --set dct_lambda_ipcw_rank=0.1 \
  --set dct_v38_lambda_direction=0.05 \
  --set dct_v38_lambda_dose=0.0 \
  --set dct_v38_lambda_reconfiguration=0.0 \
  --set dct_v38_warmup_epochs=0 \
  --set dct_v38_ramp_epochs=0 \
  --set dct_lambda_etar=0.0 \
  --set dct_lambda_listwise=0.0 \
  --set dct_v382_lambda_mgptr=0.0 \
  --set dct_v382_adaptive_aux_weights=false \
  --set fit_bins_on_train=true \
  --set binning_mode=global_qcut \
  --set dct_slot_init_mode=deterministic \
  --set event_stratified_batches=true \
  --set event_sampling_fraction=0.0 \
  --set dct_ipcw_rank_memory_size=64 \
  --set dct_mix_ratio=1.0 \
  --set num_patches=2048 \
  --set batch_size=8 \
  --set which_splits=5fold \
  --set on_missing_wsi=error \
  --set wsi_encoder=uni \
  --set encoding_dim=1024 \
  --set study=blca \
  --set data_root_dir=/data/CPathPatchFeature \
  --set data_path=/data1/dataset_csv \
  --set k_start=2 \
  --set k_end=5 \
  --set gpu=1 \
  --set num_workers=4 \
  --set results_dir=results/dct_v3.10_uni_blca_p2048_30ep/final/blca \
  --set specific_simple=dct_v310_uni_p2048_blca_5fold_30ep \
  2>&1 | tee "$LOG"
