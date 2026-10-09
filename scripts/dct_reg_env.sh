#!/usr/bin/env bash
# DCT-Reg verified environment — source this and you are ready to run.
#
# Last verified: 2026-10-08 on this exact machine
#   GPU:       2× NVIDIA GeForce RTX 5090 (sm_120), driver 570.86.10, CUDA 12.8
#   nvcc:      12.8.61 at /usr/local/cuda-12.8
#   conda env: /home/ubuntu/.conda/envs/trisurv  (NOT /home/condabins/trisurb!)
#   python:    3.10.20
#   torch:     2.10.0+cu128   (arch list: sm_70,75,80,86,90,100,120)
#   torchvision: 0.25.0+cu128
#   cudnn:     9.10.2
#   numpy:     2.2.5  (conda) — if pip later installs 1.26.4 it overrides
#
# Usage (every new window):
#   source /data1/DCT-Reg/scripts/dct_reg_env.sh
#   python -c "import torch; print(torch.cuda.get_device_name(0))"
#
# If torch sees 0 GPUs or "kernel image not available", run:
#   rm -rf ~/.cache/torch_extensions
#   export TORCH_CUDA_ARCH_LIST="12.0"
#   then re-source.
#
# DO NOT trust requirements.txt for torch — it says 2.1+cu118 which is wrong.
# The line below is the only one that matches what v3.13/14/15/16 actually use.

# ---- 1. Conda env ----
export DCT_REG_ENV_NAME="trisurv"
export DCT_REG_PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"

# ---- 2. CUDA toolchain ----
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ---- 3. PyTorch JIT target (RTX 5090 = sm_120) ----
export TORCH_CUDA_ARCH_LIST="12.0"

# ---- 4. Repo wiring ----
DCT_REG_ROOT="/data1/DCT-Reg"
export DCT_REG_ROOT
export UNI2H_ROOT="/data1/TCGA-UNI2-h-features"
export DCT_DATA_CSV_ROOT="/data1/SurvOT-Rank/survot_rank/research/legacy/slotspe_runtime/dataset_csv"
export PYTHONPATH="${DCT_REG_ROOT}:${PYTHONPATH:-}"
export DCT_REG_CACHE="${DCT_REG_ROOT}/.cache"
mkdir -p "${DCT_REG_CACHE}"

# ---- 5. Verification (one-line) ----
verify_dct_reg_env() {
  "${DCT_REG_PYTHON}" - <<'PY'
import sys, torch
print(f"python     = {sys.version.split()[0]}")
print(f"torch      = {torch.__version__} (cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()})")
print(f"arch list  = {torch.cuda.get_arch_list() if torch.cuda.is_available() else 'NO CUDA'}")
if torch.cuda.is_available():
    print(f"device     = {torch.cuda.get_device_name(0)} (sm {torch.cuda.get_device_capability(0)})")
    x = torch.randn(8, 8, device='cuda')
    print(f"kernel ok  = {(x @ x).shape}")
PY
}

echo "DCT-Reg env loaded."
echo "  env    = ${DCT_REG_ENV_NAME}  (${DCT_REG_PYTHON})"
echo "  root   = ${DCT_REG_ROOT}"
echo "  cuda   = ${CUDA_HOME}"
echo ""
echo "Run: verify_dct_reg_env   to check GPU/PyTorch in one shot."
