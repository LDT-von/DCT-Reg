#!/usr/bin/env bash
# Lightweight environment wiring for /data1/DCT-Reg.
# This file only sets env vars and PYTHONPATH. It does NOT install torch,
# create 5fold_uni2h splits, or modify SurvOT-Rank. Run it with `source`.
#
# Usage:
#   source scripts/dct_reg_env.sh
#   python -m survot_rank.cli doctor
#   python scripts/run_dct_v310_final_cross_cancer.py plan

# Repo root
DCT_REG_ROOT="/data1/DCT-Reg"
export DCT_REG_ROOT

# Shared datasets that already live on this machine.
# DCT-Reg's CLI expects UNI2-h WSI features and clinical CSV roots.
# UNI2-h features live at /data1/TCGA-UNI2-h-features (10 cancers).
# Clinical CSVs and 5fold_uni2h splits live under SurvOT-Rank's packaged
# legacy runtime; the independent /data1/dataset_csv mirror is missing the
# 5fold_uni2h directory and has an extra index column on 5fold/blca/*.csv.
export UNI2H_ROOT="/data1/TCGA-UNI2-h-features"
export DCT_DATA_CSV_ROOT="/data1/SurvOT-Rank/survot_rank/research/legacy/slotspe_runtime/dataset_csv"

# Make the DCT-Reg package importable as `survot_rank` without installing.
export PYTHONPATH="${DCT_REG_ROOT}:${PYTHONPATH:-}"

# Cache site under DCT-Reg so it never touches SurvOT-Rank caches.
export DCT_REG_CACHE="${DCT_REG_ROOT}/.cache"
mkdir -p "${DCT_REG_CACHE}"

echo "DCT_REG_ROOT=${DCT_REG_ROOT}"
echo "UNI2H_ROOT=${UNI2H_ROOT}"
echo "DCT_DATA_CSV_ROOT=${DCT_DATA_CSV_ROOT}"
echo "DCT_REG_CACHE=${DCT_REG_CACHE}"
echo "PYTHONPATH starts with: ${PYTHONPATH%%:*}"