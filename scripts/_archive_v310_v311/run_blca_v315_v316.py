#!/usr/bin/env python3
"""Run BLCA 5-fold for v3.15 (RTI-directed IPCW rank) and v3.16 (Slot MI).
GPU 0 only. Identical to run_blca_uni2h_5fold.py but simpler.

Usage:
    python3 scripts/run_blca_v315_v316_blca.py --run
"""
from __future__ import annotations

import argparse, os, shlex, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/ubuntu/.conda/envs/trisurv/bin/python"

VERSIONS = {
    "v315r": {
        "config": "configs/dct_v315_rti_rank_blca_uni2h.yaml",
        "results": "/data1/DCT-Reg/results/dct_v315_rti_rank_blca_uni2h/blca",
        "log_base": "/data1/DCT-Reg/logs/v315_rti_rank_blca_uni2h_5fold",
        "method": "DCT v3.15 RTI-directed IPCW rank",
        "epochs": 30,
    },
    "v316": {
        "config": "configs/dct_v316_slot_mi_decomposition.yaml",
        "results": "/data1/DCT-Reg/results/dct_v316_slot_mi_blca_uni2h/blca",
        "log_base": "/data1/DCT-Reg/logs/v316_slot_mi_blca_uni2h_5fold",
        "method": "DCT v3.16 Slot MI decomposition",
        "epochs": 30,
    },
}

DATASET_CSV_ROOT = "/data1/dataset_csv"


@dataclass(frozen=True)
class Job:
    version: str
    fold: int
    command: tuple[str, ...]
    log_file: Path


def build_jobs(versions: list[str]) -> list[Job]:
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
                str(PYTHON), "-m", "survot_rank.cli", "train",
                "--config", str(cfg_path),
                "--set", f"gpu=GPU_0",
                "--set", f"k_start={fold}",
                "--set", f"k_end={fold + 1}",
                "--set", f"results_dir={result_dir.as_posix()}",
                "--set", f"data_path={DATASET_CSV_ROOT}",
            )
            jobs.append(Job(ver, fold, cmd, log_file))
    return jobs


def _done(log_file: Path) -> bool:
    return log_file.exists() and "best cindex=" in log_file.read_text()


def run_queue(jobs: list[Job], *, force: bool, gpu: int) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    done, errors = 0, []
    for job in jobs:
        if not force and _done(job.log_file):
            print(f"[GPU{gpu}] skip {job.version} fold{job.fold}: done")
            done += 1
            continue

        actual_cmd = tuple(a.replace("GPU_0", str(gpu)) for a in job.command)
        job.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(job.log_file, "wb") as lf:
            lf.write(f"=== {VERSIONS[job.version]['method']} | fold {job.fold} | GPU {gpu} ===\n".encode())
            lf.write((" ".join(shlex.quote(a) for a in actual_cmd) + "\n").encode())

        print(f"\n[GPU{gpu}] [{done+1}/{len(jobs)}] {job.version} fold{job.fold}")
        print(f"        log: {job.log_file}")

        with open(job.log_file, "ab") as lf:
            result = subprocess.run(actual_cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

        done += 1
        if result.returncode != 0:
            errors.append((job, result.returncode))
            print(f"[GPU{gpu}] [ERROR] {job.version} fold{job.fold}: code {result.returncode}")
        else:
            cidx = None
            for line in job.log_file.read_text().splitlines():
                if "best cindex=" in line:
                    cidx = line.strip()
                    break
            print(f"[GPU{gpu}] [OK] {job.version} fold{job.fold} [{done}/{len(jobs)}]"
                  + (f" → {cidx}" if cidx else ""))

    if errors:
        print(f"\n[ERRORS] {len(errors)} job(s) failed:")
        for job, code in errors:
            print(f"  {job.version} fold{job.fold}: exit {code}")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Execute jobs")
    p.add_argument("--force", action="store_true", help="Overwrite existing logs")
    p.add_argument("--versions", nargs="+", default=["v315r", "v316"],
                   choices=["v315r", "v316"])
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    os.chdir(REPO_ROOT)

    jobs = build_jobs(args.versions)

    print("=" * 62)
    print("  BLCA 5-fold: v3.15 RTI-rank  vs  v3.16 Slot MI")
    print("  GPU:", args.gpu)
    print("=" * 62)
    for j in jobs:
        state = "DONE" if (not args.force and _done(j.log_file)) else "RUN "
        print(f"  {j.version:<6} Fold {j.fold}: [{state}]  {j.log_file.name}")
    print()

    if args.run:
        return run_queue(jobs, force=args.force, gpu=args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
