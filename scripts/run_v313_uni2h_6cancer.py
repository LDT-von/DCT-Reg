#!/usr/bin/env python3
"""Run v3.13 (transport-aware reconstruction) on the same 6 cancers as the
v3.10 6-cancer sweep, using the UNI2-h encoder and 5fold_uni2h splits.

Cancers: blca, skcm, hnsc, lusc, kirc, ucec   (BLCA already ran, so we
keep it here for completeness; use --cancers to filter.)

Each cancer: 5 folds × 30 epochs, batch=8, 8+8 slots, identical to the
fair-comparison config that scored 0.7238 mean cindex on BLCA.

Usage:
    python3 scripts/run_v313_uni2h_6cancer.py --plan
    python3 scripts/run_v313_uni2h_6cancer.py --run --gpus 0 1
    python3 scripts/run_v313_uni2h_6cancer.py --run --gpus 0 1 --cancers skcm hnsc lusc kirc ucec
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/ubuntu/.conda/envs/trisurv/bin/python"

V313_CONFIG = "configs/dct_v313_uni2h.yaml"  # cancer-agnostic
V313_RESULTS_BASE = "/data1/DCT-Reg/results/dct_v313_uni2h"
LOG_BASE = "/data1/DCT-Reg/logs/v313_uni2h_6cancer"
DATASET_CSV_ROOT = "/data1/dataset_csv"

# Same 6 cancers as v3.10 final cross-cancer sweep
DEFAULT_CANCERS = ("blca", "skcm", "hnsc", "lusc", "kirc", "ucec")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    log_file: Path


def build_jobs(cancers: list[str], gpu_placeholder: str = "_GPU_") -> list[Job]:
    jobs: list[Job] = []
    for cancer in cancers:
        result_dir = Path(V313_RESULTS_BASE) / cancer
        log_dir = Path(LOG_BASE) / cancer
        log_dir.mkdir(parents=True, exist_ok=True)
        specific_simple = f"dct_v313_{cancer}_uni2h"
        for fold in DEFAULT_FOLDS:
            log_file = log_dir / f"fold{fold}.log"
            cmd = (
                str(PYTHON),
                "-m", "survot_rank.cli", "train",
                "--config", str(REPO_ROOT / V313_CONFIG),
                "--set", f"gpu={gpu_placeholder}",
                "--set", f"k_start={fold}",
                "--set", f"k_end={fold + 1}",
                "--set", f"results_dir={result_dir.as_posix()}",
                "--set", f"data_path={DATASET_CSV_ROOT}",
                "--set", f"study={cancer}",
                "--set", f"specific_simple={specific_simple}",
            )
            jobs.append(Job(cancer, fold, cmd, result_dir, log_file))
    return jobs


def _completion(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    return "best cindex" in log_file.read_text()


def print_plan(jobs: list[Job], *, run_mode: bool) -> None:
    current = None
    for j in jobs:
        if j.cancer != current:
            current = j.cancer
            print(f"\n=== {j.cancer} ===")
        state = "DONE" if _completion(j.log_file) else "RUN "
        print(f"  Fold {j.fold}: [{state}]  {j.log_file}")
        if run_mode and state == "RUN ":
            print(f"    {shlex.join(j.command)}")


def run_queue(jobs: list[Job], *, gpus: list[int], force: bool = False) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(REPO_ROOT)

    lock = threading.Lock()
    completed = 0
    total = len(jobs)
    errors: list[tuple[Job, int]] = []

    def worker(gpu: int, my_jobs: list[Job]) -> None:
        nonlocal completed, errors
        gpu_env = dict(env)
        gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        gpu_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        for job in my_jobs:
            if not force and _completion(job.log_file):
                with lock:
                    completed += 1
                    print(f"[GPU{gpu}] skip {job.cancer} fold{job.fold}: already done [{completed}/{total}]")
                continue
            actual_cmd = tuple(a.replace("_GPU_", str(gpu)) for a in job.command)
            job.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(job.log_file, "wb") as lf:
                lf.write(f"=== DCT v3.13 transport-aware | {job.cancer} | fold {job.fold} | GPU {gpu} ===\n".encode())
                lf.write((" ".join(shlex.quote(a) for a in actual_cmd) + "\n").encode())
            with lock:
                print(f"\n[GPU{gpu}] [{completed + 1}/{total}] {job.cancer} fold{job.fold}")
                print(f"        log: {job.log_file}")
            with open(job.log_file, "ab") as lf:
                result = subprocess.run(actual_cmd, env=gpu_env, stdout=lf, stderr=subprocess.STDOUT)
            with lock:
                completed += 1
                if result.returncode != 0:
                    errors.append((job, result.returncode))
                    print(f"[GPU{gpu}] [ERROR] {job.cancer} fold{job.fold}: code {result.returncode}")
                else:
                    print(f"[GPU{gpu}] [OK] {job.cancer} fold{job.fold} [{completed}/{total}]")

    gpu_jobs: dict[int, list[Job]] = {g: [] for g in gpus}
    for i, job in enumerate(jobs):
        gpu_jobs[gpus[i % len(gpus)]].append(job)

    threads = [threading.Thread(target=worker, args=(g, gpu_jobs[g])) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print(f"\n[ERRORS] {len(errors)} job(s) failed:")
        for job, code in errors:
            print(f"  {job.cancer} fold{job.fold}: exit {code}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", action="store_true", help="(default) print job queue without running")
    p.add_argument("--run", action="store_true", help="Execute the queue")
    p.add_argument("--force", action="store_true", help="Re-run folds that already completed")
    p.add_argument("--cancers", nargs="+", default=list(DEFAULT_CANCERS),
                   help=f"Cancers to run. Default: {list(DEFAULT_CANCERS)}")
    p.add_argument("--gpus", nargs="+", type=int, default=[0],
                   help="GPU indices. Default: [0]. For round-robin: --gpus 0 1")
    return p


def main() -> int:
    args = build_parser().parse_args()
    jobs = build_jobs(args.cancers)
    if not jobs:
        print("No jobs to run.")
        return 0
    print(f"Total jobs: {len(jobs)} across {len(args.cancers)} cancer(s), "
          f"{len(set(j.fold for j in jobs))} fold(s) each, GPUs={args.gpus}")
    print_plan(jobs, run_mode=args.run)
    if args.run:
        return run_queue(jobs, gpus=args.gpus, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
