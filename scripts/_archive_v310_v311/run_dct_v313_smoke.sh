#!/usr/bin/env bash
# Smoke test for DCT v3.13 transport-aware pathway reconstruction.
#
# Goal: verify the v3.13 model loads, runs forward + backward, and gets a
# reasonable val_cindex on BLCA fold 0 in ~30 minutes (5 epochs).
#
# GPU: 0 (CUDA_VISIBLE_DEVICES forced)
# Config: configs/dct_v313_blca_uni.yaml
# Output: logs/v313_smoke/smoke.log + results/dct_v313_blca_uni_smoke/

set -e
cd /data1/DCT-Reg

export PYTHONPATH=/data1/DCT-Reg
export DCT_REG_CACHE=/data1/DCT-Reg/.cache
mkdir -p "${DCT_REG_CACHE}"

# ⚠️ 强制只用 GPU 0（nvidia-smi 编号 0），避免误用 GPU 1
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PYTHON=/home/ubuntu/.conda/envs/trisurv/bin/python

LOG_DIR=/data1/DCT-Reg/logs/v313_smoke
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/smoke.log"

RESULT_DIR=/data1/DCT-Reg/results/dct_v313_blca_uni_smoke
mkdir -p "${RESULT_DIR}"

echo "==========================================="
echo "v3.13 SMOKE TEST: 5 epochs, fold 0, GPU 0"
echo "Config: configs/dct_v313_blca_uni.yaml"
echo "Log: ${LOG_FILE}"
echo "Result: ${RESULT_DIR}"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="

# Use --set max_epochs=5 to keep this short.
# Smoke test only fold 0 (k_start=0, k_end=1).
# Use --set specific_simple=... so results land in a clean subdir.
$PYTHON -m survot_rank.cli train \
    --config configs/dct_v313_blca_uni.yaml \
    --set max_epochs=5 \
    --set k_start=0 \
    --set k_end=1 \
    --set gpu=0 \
    --set specific_simple=dct_v313_blca_uni_smoke \
    --set results_dir="${RESULT_DIR}/blca" \
    --set batch_size=32 \
    --set slot_num_wsi=16 \
    --set slot_num_omics=16 \
    --set slot_iters=10 \
    2>&1 | tee "${LOG_FILE}"

echo "==========================================="
echo "v3.13 SMOKE TEST DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="

# Quick health check: did val_cindex get computed?
if grep -q "best cindex" "${LOG_FILE}"; then
    echo "[OK] smoke run produced a val cindex line:"
    grep "best cindex" "${LOG_FILE}" | tail -3
else
    echo "[FAIL] no val cindex found in log — smoke test failed"
    exit 1
fi

# Look for any new v3.13 diagnostics in the log
if grep -q "v313_reconstruction_total" "${LOG_FILE}"; then
    echo "[OK] v3.13 recon diagnostics present in training output:"
    grep -E "v313_reconstruction_self|v313_reconstruction_cross|v313_reconstruction_total" "${LOG_FILE}" | tail -3
else
    echo "[WARN] v3.13 recon diagnostics missing — model may not be v3.13"
fi
