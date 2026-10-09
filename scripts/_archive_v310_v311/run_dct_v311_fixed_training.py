#!/usr/bin/env python3
"""Run the frozen DCT v3.11 per-slot interpretable method on UNI features (BLCA).

Training recipe (matches v3.11 FROZEN_ARGUMENTS):
  - Objective: NLL + 0.10*IPCW-rank + 0.05*per_slot_nll + 0.02*diversity
  - Diversity: per-sample variance in [0.005, 0.050]
  - Direction loss DISABLED (dct_v38_lambda_direction=0.0)

Outputs to results/dct_v311_blca_uni_fixed/blca/
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
RESULT_ROOT = REPO_ROOT / "results" / "dct_v311_blca_uni_fixed"

# UNI defaults
UNI_DATA_ROOT = "/data/CPathPatchFeature"
UNI_ENCODING_DIM = 1024
UNI_ENCODER = "uni"
UNI_WHICH_SPLITS = "5fold"

# Default: BLCA, 5-fold
DEFAULT_CANCERS = ("blca",)
DEFAULT_FOLDS = (0, 1, 2, 3, 4)

# v3.11 frozen recipe
FROZEN_V311_OVERRIDES: dict[str, object] = {
    "survot_method": "dct_v311_slot_interpretable",
    "bag_loss": "nll_surv",
    "max_epochs": 30,
    # v3.10 frozen (inherited)
    "dct_lambda_ipcw_rank": 0.10,
    "dct_v38_lambda_direction": 0.0,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_v382_lambda_mgptr": 0.0,
    "dct_v38_warmup_epochs": 0,
    "dct_v38_ramp_epochs": 0,
    # v3.11 new frozen
    "dct_v311_lambda_slot_nll": 0.05,
    "dct_v311_lambda_slot_diversity": 0.02,
    "dct_v311_variance_min": 0.005,
    "dct_v311_variance_max": 0.050,
    # Training setup
    "fit_bins_on_train": True,
    "binning_mode": "global_qcut",
    "dct_slot_init_mode": "deterministic",
    "event_stratified_batches": True,
    "event_sampling_fraction": 0.0,
    "dct_ipcw_rank_memory_size": 64,
    "dct_mix_ratio": 1.0,
    "num_patches": 4096,
    "batch_size": 8,
    "which_splits": UNI_WHICH_SPLITS,
    "on_missing_wsi": "error",
    "wsi_encoder": UNI_ENCODER,
    "encoding_dim": UNI_ENCODING_DIM,
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path


def _override_args(values: dict[str, object]) -> list[str]:
    """Convert dict of overrides to --set key=value CLI flags."""
    args = []
    for key, value in values.items():
        if isinstance(value, bool):
            args.append(f"--set={key}={str(value).lower()}")
        elif value is None:
            continue
        elif isinstance(value, (list, tuple)):
            for v in value:
                args.append(f"--set={key}={v}")
        else:
            args.append(f"--set={key}={value}")
    return args


def build_job(args: argparse.Namespace, cancer: str, fold: int) -> Job:
    """Build a single training command using --set key=value format."""
    # Use v3.10 config as base (overrides handle the rest)
    config = REPO_ROOT / "configs" / "dct_v310_directional_regularized_transport.yaml"
    result_dir = RESULT_ROOT / cancer
    
    values = dict(FROZEN_V311_OVERRIDES)
    values.update({
        "study": cancer,
        "data_root_dir": args.data_root,
        "data_path": args.data_csv_root,
        "k_start": fold,
        "k_end": fold + 1,
        "gpu": args.gpu,
        "num_workers": args.num_workers,
        "results_dir": str(result_dir),
        "specific_simple": f"dct_v311_blca_uni_fold{fold}_fixed",
    })
    
    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        str(config),
        "--",
        *_override_args(values),
    )
    return Job(cancer, fold, command, result_dir)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    return [build_job(args, cancer, fold) for cancer in args.cancers for fold in args.folds]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], *, run_mode: bool = False) -> None:
    print("=" * 60)
    print("DCT v3.11 Per-Slot Interpretable (BLCA, UNI features)")
    print("OBJECTIVE: NLL + 0.10*IPCW-rank + 0.05*per_slot_nll + 0.02*diversity")
    print(f"ENCODER: uni (1024-dim) | splits: 5fold")
    print(f"GPU: {jobs[0].command[0]} | Queue: {len(jobs)} folds")
    print("=" * 60)
    for index, job in enumerate(jobs, start=1):
        completion = _completion(job) if run_mode else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} {job.cancer.upper()} fold{job.fold}")
        if not completion:
            print("    " + shlex.join(job.command[:6]) + " ...")  # truncated


def run_queue(args: argparse.Namespace, jobs: list[Job]) -> int:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    
    log_dir = REPO_ROOT / "logs" / "v311_blca_uni_fixed"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    for index, job in enumerate(jobs, start=1):
        completion = _completion(job)
        if completion and not args.force:
            print(f"[{index:02d}/{len(jobs):02d}] [skip] {job.cancer.upper()} fold{job.fold}: {completion}")
            continue
        
        log_file = log_dir / f"fold{job.fold}.log"
        print(f"\n[{index:02d}/{len(jobs):02d}] Starting {job.cancer.upper()} fold{job.fold} → {log_file}")
        with open(log_file, "w") as f:
            result = subprocess.run(job.command, check=False, env=environment, stdout=f, stderr=subprocess.STDOUT)
        
        if result.returncode != 0:
            print(f"[ERROR] Job failed with code {result.returncode}, see {log_file}")
            return result.returncode
        
        print(f"[{index:02d}/{len(jobs):02d}] fold{job.fold} completed")
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--cancers", type=lambda x: x.split(","), default=list(DEFAULT_CANCERS))
    parser.add_argument("--folds", type=lambda x: [int(f) for f in x.split(",")],
                        default=list(DEFAULT_FOLDS))
    parser.add_argument("--data-root", default=UNI_DATA_ROOT)
    parser.add_argument("--data-csv-root", default="/data1/dataset_csv")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--num-workers", default="4")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    
    jobs = build_jobs(args)
    
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    
    print_plan(jobs, run_mode=True)
    return run_queue(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
