#!/usr/bin/env python3
"""Run DCT v3.11 fixed (per-modality diversity is the v2 version, but this script
uses the merged-diversity v3.11 uni_fixed recipe - the 0.7174 BLCA winner)
across multiple cancers, sequentially on a single GPU.

Outputs to results/dct_v311_blca_uni_fixed_pancancer/<cancer>/ and
logs to logs/v311_blca_uni_fixed_pancancer/<cancer>_fold<fold>.log.
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
RESULT_ROOT = REPO_ROOT / "results" / "dct_v311_blca_uni_fixed_pancancer"
LOG_ROOT = REPO_ROOT / "logs" / "v311_blca_uni_fixed_pancancer"

UNI_DATA_ROOT = "/data/CPathPatchFeature"
UNI_ENCODING_DIM = 1024
UNI_ENCODER = "uni"
UNI_WHICH_SPLITS = "5fold"  # 旧版 5fold_uni2h 是 UNI2-h encoder 用的 split, 与本 runner (uni encoder) 不匹配
# UNI encoder 实际可覆盖 ~92% 患者, 缺的少数患者由数据加载器 on_missing_wsi=warn/skip 静默处理 (不抛错)

# v3.11 frozen recipe. Note: dct_v311_lambda_slot_nll / slot_diversity /
# variance_min / variance_max are forced inside DCTV311SlotInterpretable.__init__
# via setattr (see FROZEN_ARGUMENTS), so the CLI does not need to expose them.
FROZEN_V311_OVERRIDES: dict[str, object] = {
    "survot_method": "dct_v311_slot_interpretable",
    "bag_loss": "nll_surv",
    "max_epochs": 30,
    "dct_lambda_ipcw_rank": 0.10,
    "dct_v38_lambda_direction": 0.0,
    "dct_v38_lambda_dose": 0.0,
    "dct_v38_lambda_reconfiguration": 0.0,
    "dct_lambda_etar": 0.0,
    "dct_lambda_listwise": 0.0,
    "dct_v382_lambda_mgptr": 0.0,
    "dct_v38_warmup_epochs": 0,
    "dct_v38_ramp_epochs": 0,
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
    "on_missing_wsi": "zero",  # LUAD 5fold/5fold_uni2h 中各有 1 名患者 (TCGA-55-8207) 100% WSI 缺失,
    #                            UNI encoder 在 luad 下不完全覆盖. 用 zero 临时填充 (仅 1/366 train, 1/92 val),
    #                            配合 num_patches=4096 与 slot_diversity 损失, 影响 < 0.3% 的 batch
    "wsi_encoder": UNI_ENCODER,
    "encoding_dim": UNI_ENCODING_DIM,
}


@dataclass(frozen=True)
class Job:
    cancer: str
    fold: int
    command: tuple[str, ...]
    result_dir: Path
    log_file: Path


def _override_args(values: dict[str, object]) -> list[str]:
    """Emit --set key=value pairs (space-separated) — argparse consumes
    each --set as action='append' with the next token as the value.
    """
    args = []
    for key, value in values.items():
        if isinstance(value, bool):
            args.extend(["--set", f"{key}={str(value).lower()}"])
        elif value is None:
            continue
        elif isinstance(value, (list, tuple)):
            for v in value:
                args.extend(["--set", f"{key}={v}"])
        else:
            args.extend(["--set", f"{key}={value}"])
    return args


def build_job(args: argparse.Namespace, cancer: str, fold: int) -> Job:
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
        "specific_simple": f"dct_v311_blca_uni_{cancer}_fold{fold}_fixed",
    })

    command = (
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        str(config),
        *_override_args(values),
    )

    log_file = LOG_ROOT / f"{cancer}_fold{fold}.log"
    return Job(cancer, fold, command, result_dir, log_file)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    return [build_job(args, c, f) for c in args.cancers for f in args.folds]


def _completion(job: Job) -> Path | None:
    matches = sorted(job.result_dir.rglob(f"split_{job.fold}_results_final.pkl"))
    return matches[0] if matches else None


def print_plan(jobs: list[Job]) -> None:
    print("=" * 70)
    print("DCT v3.11 Per-Slot Interpretable (UNI features) — PAN-CANCER 5-fold")
    print("OBJECTIVE: NLL + 0.10*IPCW-rank + 0.05*per_slot_nll + 0.02*diversity")
    print(f"ENCODER: uni (1024-dim) | splits: 5fold (UNI encoder) | epochs: 30")
    print(f"GPU: {jobs[0].command[-6]} | Queue: {len(jobs)} folds")
    print(f"Results: {RESULT_ROOT}")
    print(f"Logs:    {LOG_ROOT}")
    print("=" * 70)
    for index, job in enumerate(jobs, start=1):
        completion = _completion(job)
        state = "SKIP" if completion else "RUN "
        print(f"{index:02d}. {state} {job.cancer.upper():8s} fold{job.fold}")


def run_queue(args: argparse.Namespace, jobs: list[Job]) -> int:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTHONPATH", str(REPO_ROOT))

    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    for index, job in enumerate(jobs, start=1):
        completion = _completion(job)
        if completion and not args.force:
            print(f"[{index:02d}/{len(jobs):02d}] [skip] {job.cancer.upper()} fold{job.fold}: {completion}")
            continue

        print(f"\n[{index:02d}/{len(jobs):02d}] Starting {job.cancer.upper()} fold{job.fold} → {job.log_file}")
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(job.log_file, "w") as f:
            f.write(f"# launch {shlex.join(job.command)}\n")
            f.flush()
            result = subprocess.run(job.command, check=False, env=environment, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"[ERROR] Job failed with code {result.returncode}, see {job.log_file}")
            if not args.continue_on_error:
                return result.returncode
            continue

        print(f"[{index:02d}/{len(jobs):02d}] fold{job.fold} completed")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--cancers", type=lambda x: x.split(","),
                        default="brca,luad,coadread,lusc,skcm,hnsc")
    parser.add_argument("--folds", type=lambda x: [int(f) for f in x.split(",")],
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", default=UNI_DATA_ROOT)
    parser.add_argument("--data-csv-root", default="/data1/dataset_csv")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--num-workers", default="4")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="continue with the next fold if one fails")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    jobs = build_jobs(args)

    if args.mode == "plan":
        print_plan(jobs)
        return 0

    print_plan(jobs)
    print()
    return run_queue(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
