#!/usr/bin/env python3
"""Run DCT v3.10 DCT-Reg with UNI encoder on BLCA fold0+1 only,
30 epochs, num_patches=2048 (down from 4096).

This is a focused experiment to test whether num_patches=2048 (matching
UNI2-h) helps close the gap with SlotSPE on BLCA.

Usage:
    python scripts/run_v310_uni_blca_p2048_30ep.py plan
    python scripts/run_v310_uni_blca_p2048_30ep.py doctor
    python scripts/run_v310_uni_blca_p2048_30ep.py run [--force]
    python scripts/run_v310_uni_blca_p2048_30ep.py smoke
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts import _queue_runtime as runtime
except (ModuleNotFoundError, ImportError):
    import _queue_runtime as runtime


REPO_ROOT = Path(__file__).resolve().parent.parent

# Only BLCA, only folds 0 and 1, 30 epochs, num_patches=2048
DEFAULT_CANCERS = ("blca",)
DEFAULT_FOLDS = (0, 1)
RESULT_ROOT = Path("results/dct_v3.10_uni_blca_p2048_30ep/final")
SMOKE_ROOT = Path("results/dct_v3.10_uni_blca_p2048_30ep/smoke")

FROZEN_OVERRIDES: dict[str, object] = {
    "survot_method": "dct_v310_directional_regularized_transport",
    "bag_loss": "nll_surv",
    "max_epochs": 30,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_v38_lambda_direction": 0.05,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    "dct_v38_warmup_epochs": 0,
    "dct_v38_ramp_epochs": 0,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_v382_lambda_mgptr": 0.0,
    "dct_v382_adaptive_aux_weights": False,
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_slot_init_mode": "deterministic",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "dct_ipcw_rank_memory_size": 64,
    "dct_mix_ratio": 1.0,
    "num_patches": 2048,                            # *** changed from 4096
    "batch_size": 8,
    "which_splits": "5fold",
    "on_missing_wsi": "error",
    "wsi_encoder": "uni",
    "encoding_dim": 1024,
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path


def parse_cancers(value: str) -> list[str]:
    return runtime.parse_cancers(value)


def build_job(args: argparse.Namespace, cancer: str, fold: int, *, smoke: bool) -> Job:
    config = Path("configs/dct_v310_directional_regularized_transport.yaml")
    result_dir = (SMOKE_ROOT if smoke else RESULT_ROOT) / cancer
    values = dict(FROZEN_OVERRIDES)
    values.update(
        {
            "study": cancer,
            "data_root_dir": args.data_root,
            "data_path": args.data_csv_root,
            "k_start": fold,
            "k_end": fold + 1,
            "gpu": args.gpu,
            "num_workers": args.num_workers,
            "results_dir": result_dir.as_posix(),
            "specific_simple": f"dct_v310_uni_p2048_{cancer}_fold{fold}_30ep",
        }
    )
    if smoke:
        values.update(
            {
                "max_epochs": 2,
                "max_smoke_batches": 2,
                "specific_simple": f"dct_v310_uni_p2048_smoke_{cancer}",
            }
        )
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        config.as_posix(),
        *runtime._override_args(values),
    )
    return Job(cancer, fold, command, result_dir, config)


def build_jobs(args: argparse.Namespace, *, smoke: bool = False) -> list[Job]:
    folds = args.folds[:1] if smoke else args.folds
    return [
        build_job(args, cancer, fold, smoke=smoke)
        for cancer in args.cancers
        for fold in folds
    ]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], *, force: bool = False, run_mode: bool = False) -> None:
    print("FINAL METHOD: DCT v3.10 Directionally Regularized Transport (DCT-Reg)")
    print("ENCODER:     UNI (1024-dim, num_patches=2048)")
    print(f"OBJECTIVE:   NLL + 0.10*IPCW-rank + 0.05*direction   (max_epochs=30)")
    print(f"Queue:       {len(jobs)} jobs; cancers={len({job.cancer for job in jobs})}")
    print(f"Result root: {RESULT_ROOT}")
    current_cancer = None
    for index, job in enumerate(jobs, start=1):
        if job.cancer != current_cancer:
            current_cancer = job.cancer
            print(f"\n[{job.cancer.upper()}]")
        completion = _completion(job) if run_mode and not force else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} fold{job.fold}")
        print("    " + shlex.join(job.command))
        if completion:
            print(f"    completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if runtime.doctor(args):
        return 2
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if not runtime.verify_child_cuda(args.python_bin, environment):
        return 1

    scheduler_lock = None
    try:
        scheduler_lock = runtime.acquire_run_lock(
            runtime.scheduler_lock_path(args.gpu, smoke),
            label=f"DCT v3.10 UNI p2048 30ep queue on GPU {args.gpu}",
        )
    except runtime.ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3

    try:
        for index, job in enumerate(jobs, start=1):
            completion = _completion(job)
            if completion and not args.force and not smoke:
                print(
                    f"[{index:02d}/{len(jobs):02d}] [skip] "
                    f"{job.cancer.upper()} fold{job.fold}: {completion}"
                )
                continue
            task_lock = None
            try:
                task_lock = runtime.acquire_run_lock(
                    runtime.task_lock_path(job),
                    label=f"DCT v3.10 UNI p2048 {job.cancer.upper()} fold{job.fold}",
                )
            except runtime.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"DCT v3.10 UNI p2048 {job.cancer.upper()} fold{job.fold}"
                )
                print(shlex.join(job.command))
                completed = subprocess.run(job.command, check=False, env=environment)
                if completed.returncode != 0:
                    print(
                        f"[ERROR] job failed with code {completed.returncode}; "
                        "queue stopped"
                    )
                    return completed.returncode
            finally:
                runtime.release_run_lock(task_lock)
        return 0
    finally:
        runtime.release_run_lock(scheduler_lock)


def build_parser() -> argparse.ArgumentParser:
    parser = runtime.build_parser()
    parser.description = __doc__
    parser.set_defaults(cancers=list(DEFAULT_CANCERS), folds=list(DEFAULT_FOLDS))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    # Inject encoder params so runtime.doctor() checks the right paths
    args.wsi_encoder = "uni"
    args.which_splits = "5fold"
    if args.mode == "doctor":
        return runtime.doctor(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    print_plan(jobs, force=args.force, run_mode=args.mode == "run")
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
