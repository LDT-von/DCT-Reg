#!/usr/bin/env python3
"""Schedule and launch E049 transport-plugin experiment batches.

Each task is launched as an independent subprocess.  Only plan creation is
performed locally; actual training happens on the cluster or remote server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import os
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schedule_core import PlanError, build_tasks, comma_values, integer_values, safe_name, selected_arms


def _run(args, unknown):
    # Resolve arms
    arms = selected_arms(args.group, args.arm)
    if not arms:
        raise PlanError("At least one arm must be selected")

    # Resolve baselines
    baseline_values = comma_values(args.baseline, ["mcat", "survpath"])
    unknown_baselines = [b for b in baseline_values if b not in ("mcat", "survpath")]
    if unknown_baselines:
        raise PlanError(f"Unknown baselines: {', '.join(unknown_baselines)}")
    baselines = baseline_values

    # Resolve folds, seeds, gpus
    folds = integer_values(args.folds, range(5), name="folds")
    seeds = integer_values(args.seeds, [3], name="seeds")
    gpus = integer_values(args.gpus, [0], name="gpus")

    # Validate batch_dir
    batch_dir = Path(args.batch_dir).resolve()
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise PlanError(f"batch_dir already exists and is not empty: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the same data path the worker will receive; never assume repo/data.
    ref = Path(args.config) if args.config else ROOT / "configs" / "dct_v313_full_wsi_three_decoders.yaml"
    if not ref.is_file():
        raise PlanError(f"Reference config does not exist: {ref}")
    from survot_rank.config import load_config
    config_data = load_config(ref)
    data_path = Path(args.data_path or config_data["data"]["data_path"])
    split_set = safe_name(args.split_set or "5fold_uni2h_outer_test_seed3", label="split-set")
    study = safe_name(args.study, label="study")
    python_exe = Path(args.python or sys.executable).resolve()
    worker_path = (Path(__file__).resolve().parent / "worker.py").resolve()

    tasks = build_tasks(
        arms=arms,
        baselines=baselines,
        folds=folds,
        seeds=seeds,
        gpus=gpus,
        batch_dir=batch_dir,
        data_path=data_path,
        split_set=split_set,
        protocol=args.protocol,
        python_executable=str(python_exe),
        worker_path=worker_path,
        require_splits=not args.skip_splits,
        study=study,
    )

    # Write plan
    plan_path = batch_dir / "plan.json"
    if plan_path.exists():
        raise PlanError(f"Plan already exists: {plan_path}")
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump({"schema": "e049_batch_plan_v1", "tasks": tasks, "arguments": vars(args)}, f,
                  indent=2, default=str)

    print(f"Plan: {len(tasks)} tasks written to {batch_dir}")
    print(f"  Baselines: {baselines}")
    print(f"  Study: {study}")
    print(f"  Arms: {arms}")
    print(f"  Folds: {folds}, Seeds: {seeds}")
    if args.dry_run:
        print("[dry-run] No tasks launched.")
        return 0

    # Launch one task at a time so a single GPU is never oversubscribed.
    launched = 0
    for task in tasks:
        if args.execute and task["split_error"]:
            raise PlanError(task["split_error"])
        task_dir = Path(task["output_dir"])
        task_dir.mkdir(parents=True, exist_ok=True)

        # Write per-task config.yaml stub
        config_stub = task_dir / "config.yaml"
        if not config_stub.exists():
            # Copy from the reference config
            shutil.copy2(ref, config_stub)

        cmd = task["command"]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = task["gpu_visible"]
        log = task_dir / "train.log"

        if args.execute:
            with log.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd, stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(ROOT), env=env,
                )
            if proc.returncode != 0:
                raise RuntimeError(f"Task failed ({proc.returncode}): {task['task_id']}; see {log}")
            print(f"Completed {task['task_id']} -> {log}")
            launched += 1
        else:
            print(f"[queued] {task['task_id']}")

    print(f"\n{launched} tasks launched." if args.execute else f"\n{len(tasks)} tasks queued (use --execute).")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-dir", type=Path, required=True,
                        help="output directory for the experiment batch")
    parser.add_argument("--group", default="paired",
                        choices=["paired", "all", "coupling", "reconstruction", "shared"],
                        help="arm group to run (default: paired native/full plugin)")
    parser.add_argument("--arm", action="append",
                        help="explicit arm(s), overrides --group; comma-separated or repeat")
    parser.add_argument("--baseline", action="append",
                        help="baseline(s): mcat and/or survpath; repeat or comma-separated")
    parser.add_argument("--study", default="blca", help="TCGA cancer cohort (default: blca)")
    parser.add_argument("--folds", action="append",
                        help="fold indices; repeat or comma-separated")
    parser.add_argument("--seeds", action="append",
                        help="random seeds; repeat or comma-separated")
    parser.add_argument("--gpus", action="append",
                        help="GPU indices for task distribution")
    parser.add_argument("--protocol", default="outer_test",
                        choices=["legacy_val", "outer_test"],
                        help="validation protocol")
    parser.add_argument("--data-path", type=str,
                        help="root path to patient data (default: data/)")
    parser.add_argument("--split-set", type=str,
                        help="name of the split set directory (default: 5fold_uni2h_outer_test_seed3)")
    parser.add_argument("--config", type=str,
                        help="reference config.yaml to copy into each task dir")
    parser.add_argument("--python", type=str,
                        help="python executable (default: current)")
    parser.add_argument("--execute", action="store_true",
                        help="actually launch tasks (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show plan without launching")
    parser.add_argument("--skip-splits", action="store_true",
                        help="skip split-file existence checks")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"Unknown arguments: {' '.join(unknown)}")
    if args.execute and args.skip_splits:
        parser.error("--skip-splits is only permitted for a dry run")
    try:
        return _run(args, unknown)
    except PlanError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
