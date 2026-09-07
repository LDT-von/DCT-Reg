#!/usr/bin/env python3
"""Matched five-arm experiments for DCT v3.30 closed-loop transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts import _queue_runtime as runtime
except (ModuleNotFoundError, ImportError):
    import _queue_runtime as runtime


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path("configs/dct_v330_closed_loop_prognostic_transport.yaml")
METHOD = "dct_v330_closed_loop_prognostic_transport"
VARIANTS = (
    "baseline",
    "self_update",
    "ot_feedback",
    "confidence_gate",
    "prognostic_rank",
)


@dataclass(frozen=True)
class Job:
    variant: str
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path = CONFIG


def parse_variants(value: str) -> list[str]:
    selected = list(VARIANTS) if value.strip().lower() == "all" else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    if not selected or len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("select at least one unique variant")
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variants {unknown}; choose from {list(VARIANTS)}"
        )
    return selected


def build_parser():
    parser = runtime.build_parser()
    parser.description = __doc__
    parser.set_defaults(cancers=["blca"], folds=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", type=parse_variants, default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--rounds", type=int, choices=range(1, 9), default=1)
    return parser


def _source_hash() -> str:
    digest = hashlib.sha256()
    for root in (REPO_ROOT / "survot_rank", REPO_ROOT / "scripts"):
        for source in sorted(root.rglob("*.py")):
            digest.update(source.relative_to(REPO_ROOT).as_posix().encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def build_jobs(args, *, smoke=False):
    if args.max_epochs < 1:
        raise ValueError("max-epochs must be positive")
    if len(set(args.cancers)) != len(args.cancers) or len(set(args.folds)) != len(args.folds):
        raise ValueError("duplicate cancers/folds would repeat an experiment")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    source_hash = _source_hash()
    jobs = []
    for variant in args.variants:
        for cancer in args.cancers:
            for fold in (args.folds[:1] if smoke else args.folds):
                values = dict(
                    study=cancer,
                    data_root_dir=args.data_root,
                    data_path=args.data_csv_root,
                    k_start=fold,
                    k_end=fold + 1,
                    gpu=args.gpu,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    max_epochs=2 if smoke else args.max_epochs,
                    survot_method=METHOD,
                    dct_v330_variant=variant,
                    dct_v330_rounds=args.rounds,
                    bag_loss="nll_surv",
                )
                if smoke:
                    values["max_smoke_batches"] = 2
                identity = {
                    key: value for key, value in values.items()
                    if key not in {"gpu", "num_workers"}
                }
                identity.update(git_revision=revision, source_hash=source_hash)
                fingerprint = hashlib.sha256(
                    (REPO_ROOT / CONFIG).read_bytes()
                    + json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()[:12]
                root = (
                    Path("results/dct_v3.30_smoke")
                    if smoke else Path("results/dct_v3.30")
                )
                result_dir = root / variant / cancer / f"seed{args.seed}_{fingerprint}"
                values.update(
                    results_dir=result_dir.as_posix(),
                    specific_simple=f"v330_{variant}_{cancer}_seed{args.seed}_{fingerprint}",
                )
                command = (
                    args.python_bin, "-m", "survot_rank.cli", "train", "--config",
                    CONFIG.as_posix(), *runtime._override_args(values),
                )
                jobs.append(Job(variant, cancer, fold, command, result_dir))
    return jobs


def run_queue(args, jobs, *, smoke=False):
    if runtime.doctor(args):
        return 2
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONUNBUFFERED="1",
    )
    if not runtime.verify_child_cuda(args.python_bin, environment):
        return 1
    scheduler_lock = None
    try:
        scheduler_lock = runtime.acquire_run_lock(
            runtime.scheduler_lock_path(args.gpu, smoke),
            label="DCT v3.30 closed-loop transport queue",
        )
        for job in jobs:
            completed = list(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
            if completed and not args.force:
                print(f"[skip-complete] {job.variant} {job.cancer} fold{job.fold}")
                continue
            task_lock = None
            try:
                task_lock = runtime.acquire_run_lock(
                    runtime.task_lock_path(job),
                    label=f"DCT v3.30 {job.variant} {job.cancer} fold{job.fold}",
                )
                process = subprocess.run(
                    job.command, cwd=REPO_ROOT, env=environment, check=False
                )
                if process.returncode:
                    return process.returncode
            finally:
                runtime.release_run_lock(task_lock)
    except runtime.ActiveRunError as error:
        print(f"[already-running] {error}")
        return 3
    finally:
        runtime.release_run_lock(scheduler_lock)
    return 0


def main():
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    if args.mode == "doctor":
        return runtime.doctor(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    print(
        f"DCT v3.30 CLOSED-LOOP STUDY | {len(jobs)} jobs | "
        "matched protocol; candidate evidence only"
    )
    for job in jobs:
        print(f"[{job.variant} {job.cancer} fold{job.fold}] {shlex.join(job.command)}")
    return 0 if args.mode == "plan" else run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
