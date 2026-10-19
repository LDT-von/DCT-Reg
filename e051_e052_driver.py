#!/usr/bin/env python3
"""E051/E052 unified driver: runs the 3+3 full 5-fold 30-epoch arms.

Usage (dry-run, then execute):
    python3 e051_e052_driver.py --dry-run
    python3 e051_e052_driver.py --execute

Output dirs (already created by quick test, will NOT be overwritten):
    实验档案/结果/E051_v22_20261014/{control,capacity,regularized}
    实验档案/结果/E052_v23_20261014/{control,masked_plain,gain_calibrated}

Each arm runs 5 folds sequentially on GPU 1 to avoid OOM.
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CWD = Path.cwd()
DCT_PYTHON = "/home/ubuntu/.conda/envs/dct/bin/python"


# ----------------------------------------------------------------------
# Experiment definitions
# ----------------------------------------------------------------------
class Arm:
    def __init__(self, name: str, config: Path, results_dir: Path,
                 folds: range, method: str = "dct_v22"):
        self.name = name
        self.config = config
        self.results_dir = results_dir
        self.folds = folds
        self.method = method


E051_ARMS = [
    Arm("control",      ROOT / "configs/dct_v22_blca_legacy_val_control.yaml",
        ROOT / "实验档案/结果/E051_v22_20261014/control",
        range(5), "dct_v22"),
    Arm("capacity",     ROOT / "configs/dct_v22_blca_legacy_val_capacity.yaml",
        ROOT / "实验档案/结果/E051_v22_20261014/capacity",
        range(5), "dct_v22"),
    Arm("regularized",  ROOT / "configs/dct_v22_blca_legacy_val_regularized.yaml",
        ROOT / "实验档案/结果/E051_v22_20261014/regularized",
        range(5), "dct_v22"),
]

E052_ARMS = [
    Arm("control",          ROOT / "configs/dct_v23_blca_legacy_val_control.yaml",
        ROOT / "实验档案/结果/E052_v23_20261014/control",
        range(5), "dct_v23"),
    Arm("masked_plain",     ROOT / "configs/dct_v23_blca_legacy_val_masked_plain.yaml",
        ROOT / "实验档案/结果/E052_v23_20261014/masked_plain",
        range(5), "dct_v23"),
    Arm("gain_calibrated",  ROOT / "configs/dct_v23_blca_legacy_val_gain_calibrated.yaml",
        ROOT / "实验档案/结果/E052_v23_20261014/gain_calibrated",
        range(5), "dct_v23"),
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def sha256(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


def make_task(arm: Arm, fold: int):
    """Build one training task dict."""
    cmd = [
        DCT_PYTHON, "-m", "survot_rank.cli", "train",
        "--config", str(arm.config),
        "--set", f"k_start={fold}",
        "--set", f"k_end={fold + 1}",
        "--set", f"results_dir={arm.results_dir}",
    ]
    return {
        "arm": arm.name,
        "fold": fold,
        "method": arm.method,
        "config": str(arm.config),
        "config_sha256": sha256(arm.config),
        "cmd": cmd,
    }


def build_plan(arms, label):
    plan = {
        "label": label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tasks": [],
    }
    for arm in arms:
        for fold in arm.folds:
            plan["tasks"].append(make_task(arm, fold))
    return plan


def validate_plan(plan):
    """Check source hasn't changed since planning."""
    current_rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_rev != plan["source_revision"]:
        raise ValueError(f"Source changed: was {plan['source_revision']}, now {current_rev}")
    for task in plan["tasks"]:
        cfg = Path(task["config"])
        if sha256(cfg) != task["config_sha256"]:
            raise ValueError(f"Config changed: {cfg}")


def execute_plan(plan, gpu: int = 1, dry_run: bool = False):
    """Run tasks sequentially, one fold at a time, on the given physical GPU."""
    validate_plan(plan)
    n = len(plan["tasks"])
    print(f"\n{'='*60}")
    print(f"Executing plan: {plan['label']} ({n} tasks, GPU {gpu})")
    print(f"Source commit: {plan['source_revision']}")
    print(f"{'='*60}\n")

    succeeded = []
    failed = []
    for i, task in enumerate(plan["tasks"], 1):
        arm = task["arm"]
        fold = task["fold"]
        method = task["method"]
        tag = f"[{i}/{n}] {plan['label']}/{arm}/fold{fold}"

        # Build per-task log path
        log_name = f"{plan['label']}_{arm}_fold{fold}.log"
        log_path = Path("/tmp") / log_name

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(ROOT)

        print(f"{tag} start  ({time.strftime('%H:%M:%S')})", flush=True)
        t0 = time.time()

        if dry_run:
            print(f"  DRY RUN: {' '.join(task['cmd'])}")
            rc = 0
        else:
            try:
                with log_path.open("w", encoding="utf-8") as stream:
                    rc = subprocess.run(
                        task["cmd"],
                        cwd=str(ROOT),
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    ).returncode
            except Exception as exc:
                rc = -1
                with log_path.open("w") as f:
                    f.write(f"LAUNCH ERROR: {exc}")

        elapsed = time.time() - t0
        if rc == 0:
            succeeded.append(task)
            # Grab best val cindex from log
            cindex = "?"
            try:
                content = log_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    if "best cindex=" in line:
                        cindex = line.split("best cindex=")[-1].strip()
                        break
            except Exception:
                pass
            print(f"  {tag} done  cindex={cindex}  elapsed={elapsed:.0f}s", flush=True)
        else:
            failed.append({**task, "rc": rc, "log": str(log_path)})
            print(f"  {tag} FAILED rc={rc}  elapsed={elapsed:.0f}s  log={log_path}", flush=True)

    print(f"\n{'='*60}")
    print(f"Plan complete: {plan['label']}")
    print(f"  Succeeded: {len(succeeded)}/{n}")
    print(f"  Failed:     {len(failed)}")
    if failed:
        print(f"  Failed tasks:")
        for t in failed:
            print(f"    {t['arm']}/fold{t['fold']}  rc={t['rc']}  log={t['log']}")
    print(f"{'='*60}\n")
    return succeeded, failed


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Actually run training (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without running (default if --execute absent)")
    parser.add_argument("--gpu", type=int, default=1,
                        help="Physical GPU index (default 1)")
    args = parser.parse_args()

    # Build plans
    e051_plan = build_plan(E051_ARMS, "E051_v22")
    e052_plan = build_plan(E052_ARMS, "E052_v23")

    # Print plans
    for plan in [e051_plan, e052_plan]:
        print(f"\n--- {plan['label']} plan ({len(plan['tasks'])} tasks) ---")
        for task in plan["tasks"]:
            print(f"  {task['arm']:20s} fold{task['fold']}  {task['config']}")

    if args.dry_run or not args.execute:
        print("\n(Dry-run complete. Use --execute to launch.)\n")
        return

    # Run E051 first, then E052
    # They share GPU 1; sequential execution avoids OOM
    print("\n" + "="*60)
    print("PHASE 1: E051 (v2.2 three arms)")
    print("="*60 + "\n")
    s1, f1 = execute_plan(e051_plan, gpu=args.gpu)

    print("\n" + "="*60)
    print("PHASE 2: E052 (v2.3 three arms)")
    print("="*60 + "\n")
    s2, f2 = execute_plan(e052_plan, gpu=args.gpu)

    # Summary
    total = len(s1) + len(s2)
    failed_total = len(f1) + len(f2)
    print("\n" + "="*60)
    print(f"ALL DONE: {total - failed_total}/{total} tasks succeeded")
    print(f"  E051: {len(s1) - len(f1)}/{len(s1)}")
    print(f"  E052: {len(s2) - len(f2)}/{len(s2)}")
    if failed_total:
        print(f"  {failed_total} tasks FAILED — check logs above")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
