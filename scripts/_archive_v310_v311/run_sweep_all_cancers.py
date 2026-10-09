#!/usr/bin/env python3
"""Run alpha sweep for all cancers × folds in parallel-friendly fashion.

Designed to be safe to run while another training job uses the GPU:
- Each sweep uses minimal GPU memory (~1-2GB)
- Sequential per-cancer to limit contention
- Skips already-completed sweeps
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/data1/DCT-Reg")
CANCERS = ["blca", "hnsc", "skcm", "lusc", "kirc", "brca", "coadread", "luad", "stad", "ucec"]
RESULT_BASE = REPO_ROOT / "results" / "dct_v3.10" / "robust" / "final_50ep_old"
DEFAULT_ALPHAS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

CKPT_PATHS = {}
for cancer in CANCERS:
    base = RESULT_BASE / cancer / cancer / "SurvOTRank_dct_v310_directional_regularized_transport"
    if not base.exists():
        continue
    matches = list(base.glob(f"*dct_reg_{cancer}*"))
    if not matches:
        continue
    exp = matches[0]
    for fold in range(5):
        ckpt = exp / f"model_best_s{fold}.pth"
        if ckpt.exists():
            CKPT_PATHS.setdefault(cancer, {})[fold] = str(ckpt)


def has_valid_sweep(cancer: str, fold: int) -> bool:
    p = REPO_ROOT / "results" / f"audit_{cancer}" / cancer / f"fold_{fold}" / f"sweep_fold{fold}.pkl"
    if not p.exists():
        return False
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
        return "risks_low" in d and "risks_high" in d
    except Exception:
        return False


def run_sweep(cancer: str, fold: int, gpu: int = 0) -> bool:
    ckpt = CKPT_PATHS.get(cancer, {}).get(fold)
    if not ckpt:
        return False
    out_dir = REPO_ROOT / "results" / f"audit_{cancer}" / cancer / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / f"sweep_fold{fold}.pkl"
    if has_valid_sweep(cancer, fold):
        return True
    cfg = REPO_ROOT / "configs" / "dct_v310_directional_regularized_transport.yaml"
    cmd = [
        "python3", str(REPO_ROOT / "scripts" / "audit_dct_reg.py"), "sweep",
        "--config", str(cfg),
        "--checkpoint", ckpt,
        "--fold", str(fold),
        "--alphas", DEFAULT_ALPHAS,
        "--output-dir", str(out_dir),
        "--gpu", str(gpu),
        "--set", f"study={cancer}",
    ]
    t0 = time.time()
    print(f"  [sweep] {cancer.upper()} fold {fold} ...", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=REPO_ROOT)
        elapsed = time.time() - t0
        if r.returncode == 0 and sweep_path.exists():
            print(f"  [sweep] ✓ {cancer.upper()} fold {fold} done in {elapsed:.1f}s", flush=True)
            return True
        else:
            print(f"  [sweep] ✗ {cancer.upper()} fold {fold} failed (rc={r.returncode}, t={elapsed:.1f}s)", flush=True)
            if r.stderr:
                print(f"     {r.stderr[-300:]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(f"  [sweep] ✗ {cancer.upper()} fold {fold} timeout", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    cancers = [args.cancer] if args.cancer != "all" else CANCERS

    total = 0
    done = 0
    for cancer in cancers:
        if cancer not in CKPT_PATHS:
            print(f"[skip] {cancer.upper()}: no ckpts")
            continue
        for fold in folds:
            if fold not in CKPT_PATHS[cancer]:
                continue
            total += 1
            if run_sweep(cancer, fold, gpu=args.gpu):
                done += 1
    print(f"\n=== {done}/{total} sweeps done ===")


if __name__ == "__main__":
    main()