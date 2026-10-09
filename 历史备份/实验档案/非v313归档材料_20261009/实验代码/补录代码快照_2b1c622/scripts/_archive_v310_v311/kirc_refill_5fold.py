#!/usr/bin/env python3
"""
Fill in missing KIRC folds 2,3,4 then re-sweep all 5 folds.
Runs AFTER the current batch2 sweep (KIRC folds 0,1 already done).
Waits for BRCA training to finish first (GPU must be free), then
uses --force to override any existing ckpts for consistency.
"""

import subprocess
import sys
import time
import os
import shutil
from pathlib import Path

REPO_ROOT = Path("/data1/DCT-Reg")
RESULT_NEW = REPO_ROOT / "results" / "dct_v3.10" / "robust" / "final"
RESULT_OLD = REPO_ROOT / "results" / "dct_v3.10" / "robust" / "final_50ep_old"
KIRC_EXP = "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_kirc_50ep"
SCRIPT_LOG = "/tmp/kirc_refill.log"

DATA_ROOT = "/data1/TCGA-UNI2-h-features"
DATA_CSV  = "/data1/SurvOT-Rank/survot_rank/research/legacy/slotspe_runtime/dataset_csv"

os.environ["PYTHONUNBUFFERED"] = "1"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(SCRIPT_LOG, "a") as f:
        f.write(line + "\n")

def run(cmd, **kwargs):
    log(f"RUN: {' '.join(str(x) for x in cmd[:4])} ...")
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        log(f"FAIL (rc={result.returncode}): {' '.join(cmd[:4])}")
        return False
    return True

def main():
    log("=" * 60)
    log("KIRC fold 2/3/4 refill + re-sweep script STARTING")
    log("=" * 60)

    # ------------------------------------------------------------------
    # 0. Wait for BRCA training to finish (scheduler lock must be free)
    # ------------------------------------------------------------------
    log("Waiting for BRCA training to finish (GPU must be free)...")
    BRCA_COMPLETE = REPO_ROOT / "results" / "dct_v3.10" / "robust" / "final" / "brca" / "brca" / "SurvOTRank_dct_v310_directional_regularized_transport"
    WAIT_COUNT = 0
    while True:
        # Check if BRCA has completed all 5 folds
        all_done = True
        for fold in range(5):
            ckpt = BRCA_COMPLETE / "*" / f"model_best_s{fold}.pth"
            matches = list(BRCA_COMPLETE.glob(f"*/model_best_s{fold}.pth"))
            if not matches:
                all_done = False
                break
        if all_done:
            log("BRCA training complete! Proceeding.")
            break
        WAIT_COUNT += 1
        if WAIT_COUNT % 10 == 0:
            log(f"  Still waiting... ({WAIT_COUNT} checks, ~{WAIT_COUNT*10}s elapsed)")
        time.sleep(10)

    # ------------------------------------------------------------------
    # 0.5 Clean any stale lock
    # ------------------------------------------------------------------
    lock_path = REPO_ROOT / "results" / ".locks" / ".run_gpu_0.lock"
    if lock_path.exists():
        log(f"Removing stale lock: {lock_path}")
        lock_path.unlink()

    # ------------------------------------------------------------------
    # 1. Train folds 2, 3, 4 using 30-epoch config (use trisurv python)
    # ------------------------------------------------------------------
    missing_folds = []
    for fold in [2, 3, 4]:
        ckpt_new = RESULT_NEW / "kirc" / "kirc" / "SurvOTRank_dct_v310_directional_regularized_transport" / KIRC_EXP / f"model_best_s{fold}.pth"
        if ckpt_new.exists():
            log(f"  fold {fold}: already exists (new) - SKIP")
            continue
        missing_folds.append(fold)

    if missing_folds:
        log(f"Training KIRC folds: {missing_folds}")
        folds_arg = ",".join(str(f) for f in missing_folds)
        cmd = [
            "python3",
            "scripts/run_dct_v310_final_cross_cancer.py",
            "run",
            "--cancers", "kirc",
            "--folds", folds_arg,
            "--data-root", DATA_ROOT,
            "--data-csv-root", DATA_CSV,
            "--gpu", "0",
            "--num-workers", "4",
            "--python", "/home/ubuntu/.conda/envs/trisurv/bin/python3",
            "--force",
        ]
        ok = run(cmd, cwd=str(REPO_ROOT))
        if not ok:
            log("Training failed! Exiting.")
            sys.exit(1)
    else:
        log("All KIRC folds 2/3/4 already trained.")

    # ------------------------------------------------------------------
    # 2. Symlink new ckpts into final_50ep_old so sweep finds them
    # ------------------------------------------------------------------
    old_base = RESULT_OLD / "kirc" / "kirc" / "SurvOTRank_dct_v310_directional_regularized_transport" / KIRC_EXP
    for fold in range(5):
        ckpt_new = RESULT_NEW / "kirc" / "kirc" / "SurvOTRank_dct_v310_directional_regularized_transport" / KIRC_EXP / f"model_best_s{fold}.pth"
        ckpt_old = old_base / f"model_best_s{fold}.pth"
        if ckpt_new.exists():
            if not ckpt_old.exists():
                log(f"  symlink: fold {fold} -> old location")
                ckpt_old.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(ckpt_new, ckpt_old)
                    log(f"  symlink done: {ckpt_old}")
                except FileExistsError:
                    log(f"  symlink already exists: {ckpt_old}")
            else:
                log(f"  fold {fold}: old ckpt already exists, no symlink needed")

    # ------------------------------------------------------------------
    # 3. Re-sweep KIRC all 5 folds
    # ------------------------------------------------------------------
    log("Running KIRC sweep for all 5 folds...")
    cmd = [
        "python3",
        "scripts/run_sweep_all_cancers.py",
        "--cancer", "kirc",
        "--gpu", "0",
    ]
    ok = run(cmd, cwd=str(REPO_ROOT))
    if not ok:
        log("Sweep failed! Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Touch completion flag so sequential_train.sh continues
    # ------------------------------------------------------------------
    log("Touching kirc_complete.flag")
    Path("/tmp/kirc_complete.flag").touch()

    log("=" * 60)
    log("KIRC fold 2/3/4 refill + re-sweep COMPLETE")
    log("=" * 60)

if __name__ == "__main__":
    main()
