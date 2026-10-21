#!/bin/bash
# 串行跑 SlotSPE transport_only BLCA 5-fold x 3-arm
# native -> bypass -> independent
set -euo pipefail

PY=/home/ubuntu/.conda/envs/dct/bin/python
BASE=/data1/DCT-Reg
SCRIPT=$BASE/scripts/run_transport_only.py
OUT=$BASE/results/slotspe_transport_only_blca_5fold
LOG_DIR=$BASE/results/slotspe_transport_only_blca_5fold/_runner_logs
mkdir -p "$LOG_DIR"

# 通用训练参数
ARGS=(
  --host slotspe
  --source-root $BASE/third_party/SlotSPE
  --data_path $BASE/data/dataset_csv
  --data_root_dir /data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files
  --study blca
  --which_splits 5fold_uni2h
  --k 5 --k_start 0 --k_end 5
  --gpu 0
  --max_epochs 30
  --label_col survival_months_dss
  --rna_format Pathways
  --signature combine
  --encoding_dim 1536
  --wsi_projection_dim 256
  --slot_num_wsi 8 --slot_num_omics 8
  --slot_iters 10
  --temperature 0.01
  --topk_ratio 0.25
  --top_k_method parallel_topk_st
  --seed 3
  --n_classes 4
  --lr 5e-4 --reg 1e-3
  --opt adam
  --batch_size 32
  --bag_loss nll_surv
  --alpha_surv 0.5
  --scheduler cosine
  --eta_min 1e-6
  --lambda_recon_loss 0.01
)

run_arm() {
  local arm="$1"
  local results_dir="$OUT/$arm"
  local log="$LOG_DIR/${arm}.log"
  mkdir -p "$results_dir"
  echo "[$(date '+%F %T')] ARM=$arm start" | tee -a "$LOG_DIR/runner.log"
  $PY -u "$SCRIPT" \
    --arm "$arm" \
    --record "$results_dir/record.json" \
    "${ARGS[@]}" \
    --results_dir "$results_dir" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] ARM=$arm done" | tee -a "$LOG_DIR/runner.log"
}

run_arm native
run_arm bypass
run_arm independent

echo "[$(date '+%F %T')] ALL ARMS DONE" | tee -a "$LOG_DIR/runner.log"
for arm in native bypass independent; do
  if [ -f "$OUT/$arm/blca/SlotSPE"/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_/summary.csv ]; then
    echo "=== $arm summary ==="
    grep mean "$OUT/$arm/blca/SlotSPE"/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_/summary.csv || true
  fi
done