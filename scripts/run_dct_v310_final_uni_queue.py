#!/usr/bin/env python3
"""Run the frozen DCT v3.10 / DCT-Reg paper method on UNI features.

Same frozen recipe as ``run_dct_v310_final_cross_cancer.py`` but with the
UNI feature root (``/data/CPathPatchFeature``), the 1024-dim ``uni``
encoder, and the ``5fold`` splits.  Use this queue for the cancers whose
UNI2-h coverage is incomplete: brca, coadread, luad, stad (UNI covers
100% of the clinical cohort for those, while UNI2-h is short).

The objective, bag loss, slot init, binning mode, batching strategy,
epochs (30), and lambda weights are identical to the UNI2-h queue.  The
only differences are:

    wsi_encoder   = uni
    encoding_dim  = 1024
    data_root_dir = /data/CPathPatchFeature
    which_splits  = 5fold

Output goes to ``results/dct_v3.10/robust/final_uni/<cancer>``.
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
RESULT_ROOT = Path("results/dct_v3.10/robust/final_uni")
SMOKE_ROOT = Path("results/dct_v3.10_smoke/final_uni")

# UNI defaults
UNI_DATA_ROOT = os.environ.get("UNI_ROOT", "/data/CPathPatchFeature")
UNI_ENCODING_DIM = 1024
UNI_ENCODER = "uni"
UNI_WHICH_SPLITS = "5fold"

# Default queue: the four cancers that lack UNI2-h coverage.
DEFAULT_CANCERS = ("brca", "coadread", "luad", "stad")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)

# Identical to the UNI2-h frozen recipe except for the encoder/dim/data_root/splits.
FROZEN_FINAL_OVERRIDES: dict[str, object] = {
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
    "num_patches": 2048,
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
    config: Path


def build_job(args: argparse.Namespace, cancer: str, fold: int, *, smoke: bool) -> Job:
    """Build a single training command.  No I/O, no GPU touched."""
    config = Path("configs/dct_v310_directional_regularized_transport.yaml")
    result_dir = (SMOKE_ROOT if smoke else RESULT_ROOT) / cancer
    values = dict(FROZEN_FINAL_OVERRIDES)
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
            "specific_simple": f"dct_v310_dct_reg_uni_{cancer}_50ep",
        }
    )
    if smoke:
        values.update(
            {
                "max_epochs": 2,
                "max_smoke_batches": 2,
                "specific_simple": f"dct_v310_dct_reg_smoke_uni_{cancer}",
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


def doctor_uni(args: argparse.Namespace) -> int:
    """UNI-specific doctor: features live at <root>/<cancer>/uni/pt_files/*.pt,
    splits live at <csv_root>/splits/5fold/<cancer>/fold_*.csv, clinical at
    <csv_root>/clinical/all/<cancer>.csv.
    """
    failed = False
    data_root = Path(args.data_root)
    csv_root = Path(args.data_csv_root)
    for cancer in args.cancers:
        feature_dir = data_root / cancer / "uni" / "pt_files"
        feature_ok = feature_dir.is_dir() and any(feature_dir.glob("*.pt"))
        split_dir = csv_root / "splits" / UNI_WHICH_SPLITS / cancer
        split_ok = all((split_dir / f"fold_{fold}.csv").is_file() for fold in range(5))
        clinical_path = csv_root / "clinical" / "all" / f"{cancer}.csv"
        clinical_ok = clinical_path.is_file()
        for label, ok, path in (
            ("features", feature_ok, feature_dir),
            ("clinical", clinical_ok, clinical_path),
            ("splits",   split_ok,   split_dir),
        ):
            print(f"{'OK' if ok else 'MISSING':8s} {cancer.upper()} {label}: {path}")
            failed = failed or not ok
    if failed:
        print("[BLOCKED] Provide UNI features, clinical CSVs, and frozen 5fold splits.")
    return int(failed)


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job], *, run_mode: bool = False) -> None:
    print("FINAL METHOD: DCT v3.10 Directionally Regularized Transport (DCT-Reg) — UNI")
    print("OBJECTIVE: NLL + 0.10*IPCW-rank + 0.05*direction")
    print(f"ENCODER: uni (1024-dim) | splits: 5fold | data_root: /data/CPathPatchFeature")
    print(f"Queue: {len(jobs)} jobs; cancers={len({job.cancer for job in jobs})}")
    current_cancer = None
    for index, job in enumerate(jobs, start=1):
        if job.cancer != current_cancer:
            current_cancer = job.cancer
            print(f"\n[{job.cancer.upper()}]")
        completion = _completion(job) if run_mode else None
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} fold{job.fold}")
        print("    " + shlex.join(job.command))
        if completion:
            print(f"    completed: {completion}")


def run_queue(args: argparse.Namespace, jobs: list[Job], *, smoke: bool) -> int:
    if doctor_uni(args):
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
            label=f"DCT v3.10 final UNI queue on GPU {args.gpu}",
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
                    label=f"DCT v3.10 UNI {job.cancer.upper()} fold{job.fold}",
                )
            except runtime.ActiveRunError as error:
                print(f"[skip-running] {error}")
                continue
            try:
                print(
                    f"\n[{index:02d}/{len(jobs):02d}] "
                    f"DCT v3.10 DCT-Reg UNI {job.cancer.upper()} fold{job.fold}"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "doctor", "smoke", "run"),
                        nargs="?", default="plan")
    parser.add_argument("--cancers", type=runtime.parse_cancers,
                        default=list(DEFAULT_CANCERS))
    parser.add_argument("--folds", type=runtime.parse_folds,
                        default=list(DEFAULT_FOLDS))
    parser.add_argument("--data-root", default=UNI_DATA_ROOT)
    parser.add_argument("--data-csv-root", default=runtime.DEFAULT_DATA_CSV_ROOT)
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers",
                        default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--python", dest="python_bin",
                        default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    if args.mode == "doctor":
        return doctor_uni(args)
    smoke = args.mode == "smoke"
    jobs = build_jobs(args, smoke=smoke)
    if args.mode == "plan":
        print_plan(jobs)
        return 0
    print_plan(jobs, run_mode=args.mode == "run")
    return run_queue(args, jobs, smoke=smoke)


if __name__ == "__main__":
    raise SystemExit(main())
