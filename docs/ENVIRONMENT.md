# DCT-Reg environment — single source of truth

> **TL;DR** — every new window:
> ```bash
> source /data1/DCT-Reg/scripts/dct_reg_env.sh
> verify_dct_reg_env
> ```

This file replaces the "hunt for the right torch/CUDA" cycle.
Last verified **2026-10-08** on the actual machine by reading
`/home/ubuntu/.conda/envs/trisurv/conda-meta/history` and querying
the live interpreter.

---

## Hardware

| Item | Value |
|------|-------|
| GPUs | 2× NVIDIA GeForce RTX 5090 (Bus-Id `00:10.0`, `00:11.0`) |
| Compute capability | sm_120 (12.0) |
| Driver | 570.86.10 |
| CUDA (driver) | 12.8 |
| nvcc | 12.8.61 at `/usr/local/cuda-12.8/bin/nvcc` |

## Conda env

| Field | Value | Notes |
|-------|-------|-------|
| **Name** | `trisurv` | NOT `trisurb`. NOT in `/home/condabins/`. |
| **Path** | `/home/ubuntu/.conda/envs/trisurv` | |
| **Python** | 3.10.20 | created with `conda create -n trisurv python=3.10` on 2026-04-07 |
| **Pip** | 26.0.1 | |

### Created by (from `conda-meta/history`)

```bash
conda create -n trisurv python=3.10 -y
conda install -n trisurv pytorch torchvision torchaudio pytorch-cuda=12.4 \
    -c pytorch -c nvidia -y
# later upgraded via pip to torch 2.10.0+cu128
```

## PyTorch (currently installed)

| Field | Value |
|-------|-------|
| **torch** | **2.10.0+cu128** |
| **torchvision** | 0.25.0+cu128 |
| **cudnn** | 9.10.2 (91002) |
| **CUDA runtime** | 12.8 |
| **arch list** | sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, **sm_120** |

### Why this version (and not `requirements.txt`)

`requirements.txt` line 4 still says:

```
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**This is a 2024-era comment, NOT what v3.13/14/15/16 use.** The actual
training/inference runs (scripts/*.sh, scripts/*.py) all point at
`trisurv` env where torch is 2.10.0+cu128. Installing 2.1+cu118 will
break:

- v3.13: `dct_v313_transport_reconstruction` (uses OTehv2 with
  `torch.nn.functional.scaled_dot_product_attention` and Sinkhorn)
- v3.14: same module path with masking
- v3.15: uses `dct_v315_ot` (newer CUDA fused kernels)
- v3.16: 4-channel InfoNCE plus slot MI

### Reinstall command (only if you must)

```bash
/home/ubuntu/.conda/envs/trisurv/bin/pip install --force-reinstall \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

## Other deps (from `conda-meta/`)

| Package | Version | Channel |
|---------|---------|---------|
| numpy | 2.2.5 (py310) | defaults |
| numpy-base | 2.2.5 | defaults |
| pillow | 12.2.0 | defaults |
| pyyaml | 6.0.3 | defaults |
| requests | 2.34.2 | defaults |
| sympy | 1.14.0 | defaults |
| networkx | 3.4.2 | defaults |
| pytorch-cuda | 12.4.127 | pytorch |
| cuda-runtime | 12.4.1 | nvidia |

> Note: numpy was **2.2.5** from conda. If `pip` later overwrites with
> 1.26.4, that is a pip artifact, not the original env state.

`requirements.txt` deps (scikit-learn, scikit-survival, einops, h5py,
matplotlib, pandas, tqdm) are pure-Python and `pip install -r
requirements.txt` works after torch is set.

---

## Common pitfalls (and the fix)

### `CUDA kernel image not available for the device`

**Almost never an SM issue** — sm_120 is in the arch list. Real causes:

1. **JIT cache pollution**
   ```bash
   rm -rf ~/.cache/torch_extensions
   unset TORCH_EXTENSIONS_DIR
   ```
2. **Wrong env vars** — make sure `CUDA_HOME` and `LD_LIBRARY_PATH`
   point at 12.8:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.8
   export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
   export TORCH_CUDA_ARCH_LIST="12.0"
   ```
3. **Stuck process on the GPU** —
   `nvidia-smi`, then `CUDA_VISIBLE_DEVICES=1 python ...` to switch
   cards.
4. **Wrong environment** — you accidentally activated
   `/home/condabins/trisurb` (which does not exist on this box).
   `echo "$CONDA_PREFIX"` should print `/home/ubuntu/.conda/envs/trisurv`.

### `RuntimeError: No CUDA GPUs are available`

Check, in order:

1. `nvidia-smi` — does it list the cards?
2. `CUDA_VISIBLE_DEVICES=0,1 python -c "import torch; print(torch.cuda.device_count())"`
3. Driver too old? `nvidia-smi` top line shows `Driver Version: ...`.

### numpy / pandas / scipy version drift

`pip install -r requirements.txt` is fine, but **do not** `pip install
torch ...` from there — that reinstalls the wrong torch.

---

## Files that pin this (so you can audit)

| What | Where |
|------|-------|
| Env activation (env vars + verifier) | `scripts/dct_reg_env.sh` |
| Env creation history | `/home/ubuntu/.conda/envs/trisurv/conda-meta/history` |
| Per-package metadata | `/home/ubuntu/.conda/envs/trisurv/conda-meta/*.json` |
| Hard-coded python in training scripts | `scripts/run_*.sh`, `scripts/run_*.py` |
| Config-driven training entry | `survot_rank/cli.py` |

## Quick verify (one shot)

```bash
source /data1/DCT-Reg/scripts/dct_reg_env.sh
verify_dct_reg_env
```

Expected output:

```
python     = 3.10.20
torch      = 2.10.0+cu128 (cuda 12.8, cudnn 91002)
arch list  = ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
device     = NVIDIA GeForce RTX 5090 (sm (12, 0))
kernel ok  = torch.Size([8, 8])
```

If `device` or `kernel ok` is missing, you are not on the right env —
fix that before debugging anything else.
