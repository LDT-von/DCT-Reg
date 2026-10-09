#!/usr/bin/env python3
"""Ablation study: stepwise loss function components on KIRC + BLCA (uni2h).

Two scheduler variants:
  --gpu 0    only GPU 0
  --gpu 1    only GPU 1 (legacy default)
  --gpu 0 1  both GPUs in parallel (recommended: ~2× throughput)

If --gpu is omitted, the scheduler prints a plan and exits without running.

Exp0  NLL only
Exp1  NLL + ipcw_rank
Exp2  NLL + ipcw_rank + per_slot_nll
Exp3  NLL + ipcw_rank + per_slot_nll + slot_diversity
Exp4  Exp3 + reconstruction_self
Exp5  Exp3 + reconstruction_cross
Exp6  Exp3 + reconstruction_self + reconstruction_cross  (v3.13 full)

Usage:
    # Plan only:
    python3 scripts/run_ablation_loss_components.py
    # Run on both GPUs in parallel (max 4 concurrent jobs: 2/GPU):
    python3 scripts/run_ablation_loss_components.py --gpu 0 1 --run --jobs-per-gpu 2
    # Run on GPU 1 only (legacy behaviour):
    python3 scripts/run_ablation_loss_components.py --gpu 1 --run
"""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/ubuntu/.conda/envs/trisurv/bin/python"

CANCERS = ["kirc", "blca"]

BASE_RESULT_DIR = "/data1/DCT-Reg/results"
BASE_LOG_DIR = "/data1/DCT-Reg/logs"
PYTHONPATH = str(REPO_ROOT)

# 7 ablation experiments
EXPERIMENTS = {
    "Exp0_nll_only": {
        "description": "NLL only (baseline)",
        "sets": {
            "dct_lambda_ipcw_rank": 0.0,
            "dct_v311_lambda_slot_nll": 0.0,
            "dct_v311_lambda_slot_diversity": 0.0,
            "dct_v313_disable_self_reconstruction": "true",
            "dct_v313_disable_cross_reconstruction": "true",
        },
    },
    "Exp1_nll_ipcw": {
        "description": "NLL + ipcw_rank",
        "sets": {
            "dct_lambda_ipcw_rank": 0.10,
            "dct_v311_lambda_slot_nll": 0.0,
            "dct_v311_lambda_slot_diversity": 0.0,
            "dct_v313_disable_self_reconstruction": "true",
            "dct_v313_disable_cross_reconstruction": "true",
        },
    },
    "Exp2_nll_ipcw_slotnll": {
        "description": "NLL + ipcw_rank + per_slot_nll",
        "sets": {
            "dct_lambda_ipcw_rank": 0.10,
            "dct_v311_lambda_slot_nll": 0.05,
            "dct_v311_lambda_slot_diversity": 0.0,
            "dct_v313_disable_self_reconstruction": "true",
            "dct_v313_disable_cross_reconstruction": "true",
        },
    },
    "Exp3_nll_ipcw_slotnll_div": {
        "description": "NLL + ipcw_rank + per_slot_nll + slot_diversity",
        "sets": {
            "dct_lambda_ipcw_rank": 0.10,
            "dct_v311_lambda_slot_nll": 0.05,
            "dct_v311_lambda_slot_diversity": 0.10,
            "dct_v313_disable_self_reconstruction": "true",
            "dct_v313_disable_cross_reconstruction": "true",
        },
    },
    "Exp4_Exp3_recon_self": {
        "description": "Exp3 + reconstruction_self",
        "sets": {
            "dct_lambda_ipcw_rank": 0.10,
            "dct_v311_lambda_slot_nll": 0.05,
            "dct_v311_lambda_slot_diversity": 0.10,
            "dct_v313_disable_self_reconstruction": "false",
            "dct_v313_disable_cross_reconstruction": "true",
        },
    },
    "Exp5_Exp3_recon_cross": {
        "description": "Exp3 + reconstruction_cross",
        "sets": {
            "dct_lambda_ipcw_rank": 0.10,
            "dct_v311_lambda_slot_nll": 0.05,
            "dct_v311_lambda_slot_diversity": 0.10,
            "dct_v313_disable_self_reconstruction": "true",
            "dct_v313_disable_cross_reconstruction": "false",
        },
    },
    "Exp6_full": {
        "description": "Exp3 + reconstruction_self + reconstruction_cross (v3.13 full)",
        "sets": {
            "dct_lambda_ipcw_rank": 0.10,
            "dct_v311_lambda_slot_nll": 0.05,
            "dct_v311_lambda_slot_diversity": 0.10,
            "dct_v313_disable_self_reconstruction": "false",
            "dct_v313_disable_cross_reconstruction": "false",
        },
    },
}

# Shared base config for uni2h
BASE_CONFIG = "configs/dct_v313_blca_uni2h.yaml"


@dataclass
class Job:
    cancer: str
    exp: str
    fold: int
    gpu: int  # assigned later by partition_jobs
    command: tuple[str, ...]
    result_dir: Path
    log_file: Path


def build_jobs(cancers: list[str], experiments: dict) -> list[Job]:
    jobs: list[Job] = []
    for cancer in cancers:
        for exp_name, exp_cfg in experiments.items():
            result_dir = Path(BASE_RESULT_DIR) / f"dct_v313_ablation_{exp_name}" / cancer
            log_dir = Path(BASE_LOG_DIR) / f"ablation_{exp_name}" / cancer
            log_dir.mkdir(parents=True, exist_ok=True)

            for fold in range(5):
                log_file = log_dir / f"fold{fold}.log"

                # Build --set overrides (no `gpu=` here; that is added per-GPU later)
                set_overrides = []
                for key, val in exp_cfg["sets"].items():
                    set_overrides.extend(["--set", f"{key}={val}"])

                cmd = (
                    str(PYTHON),
                    "-m", "survot_rank.cli", "train",
                    "--config", str(REPO_ROOT / BASE_CONFIG),
                    "--set", f"k_start={fold}",
                    "--set", f"k_end={fold + 1}",
                    "--set", f"results_dir={result_dir.as_posix()}",
                    "--set", f"data_path=/data1/dataset_csv",
                    "--set", f"study={cancer}",
                ) + tuple(set_overrides)

                jobs.append(Job(
                    cancer=cancer,
                    exp=exp_name,
                    fold=fold,
                    gpu=-1,
                    command=cmd,
                    result_dir=result_dir,
                    log_file=log_file,
                ))
    return jobs


def partition_jobs(jobs: list[Job], gpus: list[int]) -> list[Job]:
    """Round-robin assign jobs to GPUs."""
    for i, j in enumerate(jobs):
        j.gpu = gpus[i % len(gpus)]
    return jobs


def _completion(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    text = log_file.read_text(errors="ignore")
    return "best cindex" in text


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    current = None
    for j in jobs:
        key = (j.cancer, j.exp)
        if key != current:
            current = key
            exp_cfg = EXPERIMENTS[j.exp]
            print(f"\n{'─' * 70}")
            print(f"  {j.cancer.upper()} | {j.exp} | {exp_cfg['description']}")
            print(f"{'─' * 70}")
        state = "DONE" if (not force and _completion(j.log_file)) else "RUN "
        print(f"  Fold {j.fold} (GPU{j.gpu}): [{state}]  {j.log_file.name}")
        if run_mode and state == "RUN ":
            print(f"    {shlex.join(j.command)}")


def run_queue(jobs: list[Job], *, force: bool, gpus: list[int], jobs_per_gpu: int) -> int:
    """Run jobs in parallel across `gpus`, up to `jobs_per_gpu` concurrent jobs per GPU."""
    by_gpu: dict[int, list[Job]] = {g: [] for g in gpus}
    for j in jobs:
        by_gpu[j.gpu].append(j)

    lock = threading.Lock()
    completed = 0
    total = len(jobs)
    errors: list[tuple[Job, int]] = []

    def gpu_worker(gpu_id: int, my_jobs: list[Job]) -> None:
        nonlocal completed, errors
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["PYTHONPATH"] = PYTHONPATH
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        running: list[tuple[Job, "subprocess.Popen"]] = []
        running_pids: set[int] = set()

        def reap_dead() -> None:
            nonlocal completed
            dead = []
            for j, proc in running:
                if proc.poll() is not None:
                    dead.append((j, proc))
            for j, proc in dead:
                running_pids.discard(proc.pid)
                with lock:
                    completed += 1
                    if proc.returncode != 0:
                        errors.append((j, proc.returncode))
                        print(f"[GPU{gpu_id}] [ERROR] {j.cancer}/{j.exp} fold{j.fold}: code {proc.returncode}")
                    else:
                        print(f"[GPU{gpu_id}] [OK] {j.cancer}/{j.exp} fold{j.fold} [{completed}/{total}]")
            for entry in dead:
                try:
                    running.remove(entry)
                except ValueError:
                    pass

        for job in my_jobs:
            while len(running_pids) >= jobs_per_gpu:
                reap_dead()
                if len(running_pids) >= jobs_per_gpu:
                    time.sleep(5)

            if not force and _completion(job.log_file):
                with lock:
                    completed += 1
                    print(f"[GPU{gpu_id}] skip {job.cancer}/{job.exp} fold{job.fold}: already done [{completed}/{total}]")
                continue

            job.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Inject `--set gpu=<gpu_id>` after `--config <path>` so each job
            # targets its assigned GPU even when launched from inside the
            # same scheduler process.
            cmd = list(job.command)
            try:
                cfg_idx = cmd.index("--config")
            except ValueError:
                cfg_idx = -1
            insert_at = cfg_idx + 2 if cfg_idx >= 0 else 0
            cmd[insert_at:insert_at] = ["--set", f"gpu={gpu_id}"]

            header = f"=== {job.cancer.upper()} {job.exp} | fold {job.fold} | GPU {gpu_id} ===\n"
            cmd_str = " ".join(shlex.quote(a) for a in cmd)
            with open(job.log_file, "wb") as lf:
                lf.write(header.encode())
                lf.write((cmd_str + "\n").encode())

            with lock:
                print(f"\n[GPU{gpu_id}] [{completed + len(running_pids) + 1}/{total}] {job.cancer.upper()} {job.exp} fold{job.fold} (running: {len(running_pids)})")
                print(f"        log: {job.log_file}")

            with open(job.log_file, "ab") as lf_out:
                proc = subprocess.Popen(cmd, env=env, stdout=lf_out, stderr=subprocess.STDOUT)
            running.append((job, proc))
            running_pids.add(proc.pid)

        while running:
            reap_dead()
            if running:
                time.sleep(5)

    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=gpu_worker, args=(gpu_id, by_gpu[gpu_id]), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, nargs="+", default=None,
                   help="GPU ids to use, e.g. --gpu 0 1 (default: print plan only)")
    p.add_argument("--jobs-per-gpu", type=int, default=2,
                   help="Maximum concurrent training jobs per GPU (default: 2)")
    p.add_argument("--run", action="store_true", help="Execute the jobs.")
    p.add_argument("--force", action="store_true", help="Overwrite existing folds.")
    p.add_argument("--cancers", nargs="+", default=CANCERS, choices=CANCERS)
    p.add_argument("--experiments", nargs="+",
                   default=list(EXPERIMENTS.keys()),
                   choices=list(EXPERIMENTS.keys()))
    return p


if __name__ == "__main__":
    import os
    os.chdir(REPO_ROOT)

    args = build_parser().parse_args()

    filtered_exps = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
    jobs = build_jobs(args.cancers, filtered_exps)

    gpus = args.gpu if args.gpu else [1]  # legacy default: GPU 1
    partition_jobs(jobs, gpus)

    print(f"Total jobs: {len(jobs)}  ({len(args.cancers)} cancers × {len(filtered_exps)} exps × 5 folds)")
    print(f"GPUs: {gpus}   jobs/GPU: {args.jobs_per_gpu}   max concurrent: {len(gpus) * args.jobs_per_gpu}")
    print_plan(jobs, force=args.force, run_mode=args.run)

    if args.run:
        print(f"\n{'=' * 70}")
        print(f"  Executing {len(jobs)} jobs on GPUs {gpus}")
        print(f"{'=' * 70}")
        sys.exit(run_queue(jobs, force=args.force, gpus=gpus, jobs_per_gpu=args.jobs_per_gpu))
    else:
        print("\n(plan only — pass --run to actually train)")
        sys.exit(0)
