#!/usr/bin/env python3
"""Candidate v3.2 TGSR: matched A/B/C/D structure experiments, NLL only."""

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
CONFIG = Path("configs/dct_v32_transport_guided_slot_reaggregation.yaml")
METHOD = "dct_v32_transport_guided_slot_reaggregation"
VARIANTS = {"baseline": "none", "self_update": "self", "attention_feedback": "attention", "ot_feedback": "ot"}


@dataclass(frozen=True)
class Job:
    variant: str
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    config: Path = CONFIG


def parse_variants(value):
    variants = list(VARIANTS) if value == "all" else value.split(",")
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError(f"select unique variants from {list(VARIANTS)}")
    return variants


def build_parser():
    parser = runtime.build_parser()
    parser.description = __doc__
    parser.set_defaults(cancers=["blca"], folds=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", type=parse_variants, default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--rounds", type=int, choices=range(1, 9), default=1)
    return parser


def build_jobs(args, *, smoke=False):
    if args.max_epochs < 1:
        raise ValueError("max-epochs must be positive")
    if len(set(args.cancers)) != len(args.cancers) or len(set(args.folds)) != len(args.folds):
        raise ValueError("duplicate cancers/folds would repeat an experiment")
    revision = subprocess.run(("git", "rev-parse", "HEAD"), cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    # A live source fingerprint also distinguishes uncommitted code changes.
    source_hash = hashlib.sha256()
    for source in sorted((REPO_ROOT / "survot_rank").rglob("*.py")):
        source_hash.update(source.relative_to(REPO_ROOT).as_posix().encode())
        source_hash.update(source.read_bytes())
    jobs = []
    for variant in args.variants:
        for cancer in args.cancers:
            for fold in (args.folds[:1] if smoke else args.folds):
                values = dict(survot_method=METHOD, study=cancer,
                              data_root_dir=args.data_root, data_path=args.data_csv_root,
                              k_start=fold, k_end=fold + 1, gpu=args.gpu,
                              num_workers=args.num_workers, seed=args.seed,
                              max_epochs=2 if smoke else args.max_epochs,
                              dct_v32_feedback=VARIANTS[variant], dct_v32_rounds=args.rounds,
                              bag_loss="nll_surv", dct_lambda_ipcw_rank=0.0,
                              dct_v38_lambda_direction=0.0)
                if smoke:
                    values["max_smoke_batches"] = 2
                # Avoid reusing completions from another seed/config/budget.
                identity = {k: v for k, v in values.items() if k not in {"gpu", "num_workers"}}
                identity["git_revision"] = revision
                identity["source_hash"] = source_hash.hexdigest()
                digest = hashlib.sha256((REPO_ROOT / CONFIG).read_bytes()
                                        + json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
                root = "results/dct_v3.2_smoke" if smoke else "results/dct_v3.2"
                result_dir = Path(root) / variant / cancer / f"seed{args.seed}_{digest}"
                values.update(results_dir=result_dir.as_posix(),
                              specific_simple=f"v32_{variant}_{cancer}_seed{args.seed}_{digest}")
                command = (args.python_bin, "-m", "survot_rank.cli", "train", "--config",
                           CONFIG.as_posix(), *runtime._override_args(values))
                jobs.append(Job(variant, cancer, fold, command, result_dir))
    return jobs


def run_queue(args, jobs, *, smoke=False):
    if runtime.doctor(args):
        return 2
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       PYTHONUNBUFFERED="1")
    if not runtime.verify_child_cuda(args.python_bin, environment):
        return 1
    scheduler_lock = None
    try:
        scheduler_lock = runtime.acquire_run_lock(runtime.scheduler_lock_path(args.gpu, smoke),
                                                   label="DCT v3.2 TGSR candidate queue")
        for job in jobs:
            completed = list(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
            if completed and not args.force:
                print(f"[skip-complete] {job.variant} {job.cancer} fold{job.fold}")
                continue
            task_lock = None
            try:
                task_lock = runtime.acquire_run_lock(runtime.task_lock_path(job),
                                                     label=f"v3.2 {job.variant} {job.cancer} fold{job.fold}")
                process = subprocess.run(job.command, cwd=REPO_ROOT, env=environment, check=False)
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
    print(f"CANDIDATE v3.2 TGSR | NLL only | {len(jobs)} jobs | NOT final test evidence")
    for job in jobs:
        print(f"[{job.variant} {job.cancer} fold{job.fold}] {shlex.join(job.command)}")
    return 0 if args.mode == "plan" else run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
