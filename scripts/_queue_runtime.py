#!/usr/bin/env python3
"""Shared, method-neutral helpers for DCT-Reg experiment queues."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_CANCERS = (
    "blca", "brca", "coadread", "hnsc", "kirc",
    "luad", "lusc", "skcm", "stad", "ucec",
)
DEFAULT_DATA_ROOT = os.environ.get("UNI2H_ROOT", "/data1/TCGA-UNI2-h-features")
DEFAULT_DATA_CSV_ROOT = os.environ.get("DCT_DATA_CSV_ROOT", "data/dataset_csv")
WHICH_SPLITS = "5fold_uni2h"


class ActiveRunError(RuntimeError):
    """Raised when a live queue lock already owns the requested task."""


@dataclass
class RunLock:
    path: Path


def parse_cancers(value: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SUPPORTED_CANCERS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown cancer(s): {', '.join(unknown)}")
    if not selected:
        raise argparse.ArgumentTypeError("at least one cancer is required")
    return selected


def parse_folds(value: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from error
    if not folds or any(fold < 0 or fold > 4 for fold in folds):
        raise argparse.ArgumentTypeError("folds must be selected from 0,1,2,3,4")
    return folds


def _override_args(values: dict[str, object]) -> tuple[str, ...]:
    args: list[str] = []
    for key, value in values.items():
        encoded = str(value).lower() if isinstance(value, bool) else str(value)
        args.extend(("--set", f"{key}={encoded}"))
    return tuple(args)


def verify_child_cuda(python_bin: str, environment: dict[str, str]) -> bool:
    probe = (
        "import json, torch; "
        "print(json.dumps({'available': torch.cuda.is_available(), "
        "'count': torch.cuda.device_count()}))"
    )
    completed = subprocess.run(
        (python_bin, "-c", probe), capture_output=True, text=True,
        check=False, env=environment,
    )
    if completed.returncode != 0:
        print(f"[CUDA BLOCKED] {completed.stderr.strip()}")
        return False
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        print(f"[CUDA BLOCKED] unexpected probe output: {completed.stdout!r}")
        return False
    if not payload.get("available") or int(payload.get("count", 0)) < 1:
        print(f"[CUDA BLOCKED] child process reports {payload}")
        return False
    return True


def _safe_gpu_name(gpu: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in gpu)


def scheduler_lock_path(gpu: str, smoke: bool) -> Path:
    kind = "smoke" if smoke else "run"
    return Path("results/.locks") / f".{kind}_gpu_{_safe_gpu_name(gpu)}.lock"


def task_lock_path(job) -> Path:
    return job.result_dir / f".split_{job.fold}.dct_reg.lock"


def acquire_run_lock(path: Path, *, label: str) -> RunLock:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "label": label})
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ActiveRunError(f"lock exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return RunLock(path)


def release_run_lock(lock: RunLock | None) -> None:
    if lock is not None:
        lock.path.unlink(missing_ok=True)


def doctor(args: argparse.Namespace) -> int:
    failed = False
    data_root = Path(args.data_root)
    csv_root = Path(args.data_csv_root)
    for cancer in args.cancers:
        feature_dir = data_root / cancer / "uni2-h" / "pt_files"
        # UNI2-h features on this machine ship as HDF5 (*.h5), not torch (*.pt).
        feature_ok = feature_dir.is_dir() and (
            any(feature_dir.glob("*.pt")) or any(feature_dir.glob("*.h5"))
        )
        split_dir = csv_root / "splits" / WHICH_SPLITS / cancer
        split_ok = all((split_dir / f"fold_{fold}.csv").is_file() for fold in range(5))
        clinical_path = csv_root / "clinical" / "all" / f"{cancer}.csv"
        clinical_ok = clinical_path.is_file()
        for label, ok, path in (
            ("features", feature_ok, feature_dir),
            ("clinical", clinical_ok, clinical_path),
            ("splits", split_ok, split_dir),
        ):
            print(f"{'OK' if ok else 'MISSING':8s} {cancer.upper()} {label}: {path}")
            failed = failed or not ok
    if failed:
        print("[BLOCKED] Provide UNI2-h features, clinical CSVs, and frozen 5fold_uni2h splits.")
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "doctor", "smoke", "run"), nargs="?", default="plan")
    parser.add_argument("--cancers", type=parse_cancers)
    parser.add_argument("--folds", type=parse_folds)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-csv-root", default=DEFAULT_DATA_CSV_ROOT)
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--num-workers", default=os.environ.get("NUM_WORKERS", "4"))
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--force", action="store_true")
    return parser
