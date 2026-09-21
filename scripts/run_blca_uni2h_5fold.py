#!/usr/bin/env python3
"""Run BLCA 5-fold with v3.13 / v3.14 / v3.15 using the v3.10 robust settings
(uni2h encoder, 5fold_uni2h splits, 2048 patches, 50 epochs, 8+8 slots, batch=8).

This is the FAIR comparison suite: every version gets identical data/split/encoder
hyperparameters.  Only the model architecture and loss terms differ.

Usage:
    python3 scripts/run_blca_uni2h_5fold.py           # plan
    python3 scripts/run_blca_uni2h_5fold.py --run      # execute
    python3 scripts/run_blca_uni2h_5fold.py --run --force   # overwrite

Versions:
    v3.13: dct_v313_transport_reconstruction  (transport-aware cross-recon)
    v3.14: dct_v314_masked_transport_reconstruction  (masked MTR)
    v3.15: dct_v315_residual_transport  (NLL-only + RTI)

Reference benchmark:
    v3.10 robust 50ep: 0.7208 ± 0.0145 (BLCA, uni2h, 5fold_uni2h)
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/ubuntu/.conda/envs/trisurv/bin/python"

VERSIONS = {
    "v313": {
        "config": "configs/dct_v313_blca_uni2h.yaml",
        "results": "/data1/DCT-Reg/results/dct_v313_blca_uni2h/blca",
        "log_base": "/data1/DCT-Reg/logs/v313_blca_uni2h_5fold",
        "method": "DCT v3.13 transport-aware reconstruction",
    },
    "v314": {
        "config": "configs/dct_v314_blca_uni2h.yaml",
        "results": "/data1/DCT-Reg/results/dct_v314_blca_uni2h/blca",
        "log_base": "/data1/DCT-Reg/logs/v314_blca_uni2h_5fold",
        "method": "DCT v3.14 masked transport reconstruction",
    },
    "v315": {
        "config": "configs/dct_v315_blca_uni2h.yaml",
        "results": "/data1/DCT-Reg/results/dct_v315_blca_uni2h/blca",
        "log_base": "/data1/DCT-Reg/logs/v315_blca_uni2h_5fold",
        "method": "DCT v3.15 NLL-only RTI baseline",
    },
}

DATASET_CSV_ROOT = "/data1/dataset_csv"


@dataclass(frozen=True)
class Job:
    version: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    log_file: Path
    config: Path


def build_jobs(versions: list[str], gpu: int = 0) -> list[Job]:
    jobs: list[Job] = []
    for ver in versions:
        cfg = VERSIONS[ver]
        cfg_path = REPO_ROOT / cfg["config"]
        result_dir = Path(cfg["results"])
        log_base = Path(cfg["log_base"])
        log_base.mkdir(parents=True, exist_ok=True)

        for fold in range(5):
            log_file = log_base / f"fold{fold}.log"
            cmd = (
                str(PYTHON),
                "-m", "survot_rank.cli", "train",
                "--config", str(cfg_path),
                "--set", f"gpu={'_GPU_'}",
                "--set", f"k_start={fold}",
                "--set", f"k_end={fold + 1}",
                "--set", f"results_dir={result_dir.as_posix()}",
                "--set", f"data_path={DATASET_CSV_ROOT}",
            )
            jobs.append(Job(ver, fold, cmd, result_dir, log_file, cfg_path))
    return jobs


def _completion(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    return "best cindex" in log_file.read_text()


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    current_ver = None
    for j in jobs:
        if j.version != current_ver:
            current_ver = j.version
            info = VERSIONS[j.version]
            print(f"\n{'=' * 60}")
            print(f"  {info['method']}")
            print(f"{'=' * 60}")
        state = "DONE" if (not force and _completion(j.log_file)) else "RUN "
        print(f"  Fold {j.fold}: [{state}]  {j.log_file.name}")
        if run_mode and state == "RUN ":
            print(f"    {shlex.join(j.command)}")


def run_queue(jobs: list[Job], *, force: bool, gpus: list[int]) -> int:
    import threading

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(REPO_ROOT)

    lock = threading.Lock()
    completed = 0
    total = len(jobs)
    errors = []

    def worker(gpu: int, my_jobs: list[Job]) -> None:
        nonlocal completed, errors
        gpu_env = dict(env)
        gpu_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        gpu_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        for job in my_jobs:
            if not force and _completion(job.log_file):
                with lock:
                    completed += 1
                    print(f"[GPU{gpu}] skip {job.version} fold{job.fold}: already done [{completed}/{total}]")
                continue
            # Substitute _GPU_ placeholder with actual GPU index
            actual_cmd = tuple(a.replace("_GPU_", str(gpu)) for a in job.command)
            job.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(job.log_file, "wb") as lf:
                lf.write(f"=== {VERSIONS[job.version]['method']} | fold {job.fold} | GPU {gpu} ===\n".encode())
                lf.write((" ".join(shlex.quote(a) for a in actual_cmd) + "\n").encode())
            with lock:
                print(f"\n[GPU{gpu}] [{completed + 1}/{total}] {job.version} fold{job.fold}")
                print(f"        log: {job.log_file}")
            with open(job.log_file, "ab") as lf:
                result = subprocess.run(actual_cmd, env=gpu_env, stdout=lf, stderr=subprocess.STDOUT)
            with lock:
                completed += 1
                if result.returncode != 0:
                    errors.append((job, result.returncode))
                    print(f"[GPU{gpu}] [ERROR] {job.version} fold{job.fold}: code {result.returncode}")
                else:
                    print(f"[GPU{gpu}] [OK] {job.version} fold{job.fold} [{completed}/{total}]")

    # Split jobs round-robin across GPUs
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
            print(f"  {job.version} fold{job.fold}: exit {code}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", action="store_true",
        help="Execute the jobs instead of just printing the plan."
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if a fold log already has a best-cindex record."
    )
    p.add_argument(
        "--versions", nargs="+", default=list(VERSIONS.keys()),
        choices=list(VERSIONS.keys()),
        help="Which versions to run. Default: all three."
    )
    p.add_argument(
        "--gpus", nargs="+", type=int, default=[0],
        help="GPU indices to use. Default: [0]. With 2 GPUs: --gpus 0 1."
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)

    jobs = build_jobs(args.versions, gpu=args.gpus[0])
    print_plan(jobs, force=args.force, run_mode=args.run)

    if args.run:
        print(f"\n{'=' * 60}")
        print(f"  Starting {len(jobs)} jobs on GPU(s) {args.gpus}")
        print(f"{'=' * 60}")
        return run_queue(jobs, force=args.force, gpus=args.gpus)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
