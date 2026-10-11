"""Plan E048 ablations; only --execute launches training.

The default prints a plan and creates no files. Execute validates all split CSVs
first, claims a new batch directory exclusively, records the plan, and invokes
one worker at a time. It never resumes or overwrites an existing batch.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from schedule_core import (
    ARM_GROUPS, PROTOCOLS, PlanError, build_tasks, check_split_unchanged,
    comma_values, integer_values, safe_name, selected_arms,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=HERE / "configs" / "base.yaml")
    result.add_argument("--group", choices=tuple(ARM_GROUPS), default="first")
    result.add_argument("--arm", action="append", help="Override group; repeat or comma-separate.")
    result.add_argument("--cancer", action="append", help="Repeat or comma-separate; default blca,kirc.")
    result.add_argument("--fold", action="append", help="Fold indices, default 0,1,2,3,4.")
    result.add_argument("--seed", action="append", help="Random seeds, default 3.")
    result.add_argument("--gpu", action="append", help="Physical GPU indices, default 0; execution is serial.")
    result.add_argument("--batch-name", help="Unique directory name; default UTC timestamp.")
    result.add_argument("--results-root", type=Path, help="Override train.results_dir from base YAML.")
    result.add_argument("--data-root", help="Override data.data_root_dir (UNI2-h feature root).")
    result.add_argument("--data-path", help="Override data.data_path (dataset CSV root).")
    result.add_argument("--split-set", help="Split-set name, or absolute root containing cancer directories.")
    result.add_argument("--protocol", choices=PROTOCOLS, default="legacy_val")
    result.add_argument("--max-epochs", type=int, help="Explicit override for user-run debugging; default base recipe (30).")
    result.add_argument("--plan-out", type=Path, help="Optionally save plan JSON, refusing an existing file.")
    result.add_argument("--execute", action="store_true", help="Explicitly launch the validated task grid.")
    return result


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise PlanError("Base config must be a YAML mapping.")
    for section in ("data", "split", "train", "slot", "model", "split_setup"):
        if not isinstance(config.get(section), dict):
            raise PlanError(f"Base config requires a mapping section: {section}.")
    for section, key in (("data", "data_path"), ("data", "data_root_dir"),
                         ("train", "results_dir"), ("split_setup", "which_splits")):
        if not isinstance(config[section].get(key), str) or not config[section][key].strip():
            raise PlanError(f"Base config requires a nonempty string: {section}.{key}.")
    if config["model"].get("dct_v313_recon_weighting") != "per_branch":
        raise PlanError("E048 requires dct_v313_recon_weighting=per_branch for fair removal ablations.")
    return config


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def task_config(base: dict[str, Any], task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["name"] = f"e048_{task['arm']}_{task['study']}_fold{task['fold']}_seed{task['seed']}"
    config["data"]["study"] = task["study"]
    config["data"]["data_path"] = args.data_path or config["data"]["data_path"]
    config["data"]["data_root_dir"] = args.data_root or config["data"]["data_root_dir"]
    config["split"].update(k_start=task["fold"], k_end=task["fold"] + 1)
    config["split_setup"]["which_splits"] = args.split_set or config["split_setup"]["which_splits"]
    config["train"].update(
        seed=task["seed"], gpu=0, results_dir=task["output_dir"],
        specific_simple=config["name"], evaluation_protocol=task["protocol"],
    )
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    return config


def display_plan(plan: dict[str, Any]) -> None:
    print(f"E048: {len(plan['tasks'])} tasks; protocol={plan['protocol']}; mode={plan['mode']}")
    print(f"Batch: {plan['batch_dir']}")
    if plan["protocol"] == "legacy_val":
        print("Endpoint: selected validation scores; do not label these as held-out test results.")
    else:
        print("Endpoint: validation-selected checkpoint evaluated on the explicit outer test partition.")
    print("Workers run serially; GPU assignment sets CUDA_VISIBLE_DEVICES, with logical --gpu 0.")
    for task in plan["tasks"]:
        status = task["split_sha256"][:12] if task["split_sha256"] else "MISSING/INVALID"
        print(f"{task['task_id']}  GPU={task['gpu_visible']}  split_sha256={status}")
        print("  " + (subprocess.list2cmdline(task["command"]) if os.name == "nt" else shlex.join(task["command"])))
    invalid = {task["split_path"]: task["split_error"] for task in plan["tasks"] if task["split_error"]}
    for path, error in invalid.items():
        print(f"Split not ready: {path}: {error}")
    if plan["mode"] == "plan":
        print("Plan only: no worker was launched. Task configs are materialized only with --execute.")


def check_completion(task_dir: Path) -> None:
    """A zero subprocess status alone does not establish completed training."""
    marker = task_dir / "completion.json"
    if not marker.is_file():
        raise PlanError(f"Worker returned 0 without completion.json: {task_dir}")
    try:
        with marker.open("r", encoding="utf-8") as stream:
            completion = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"Invalid completion.json in {task_dir}: {exc}") from exc
    if not isinstance(completion, dict):
        raise PlanError(f"completion.json must be a mapping: {task_dir}")
    if completion.get("status") != "complete" or completion.get("training_complete") is not True:
        raise PlanError(f"Worker did not confirm completed training in {task_dir}.")


def execute_plan(plan: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> int:
    batch_dir = Path(plan["batch_dir"])
    batch_dir.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive claim prevents overwrite even if another process raced planning.
    batch_dir.mkdir(exist_ok=False)
    write_json_new(batch_dir / "plan.json", plan)
    log_dir = batch_dir / "logs"
    log_dir.mkdir(exist_ok=False)
    for task in plan["tasks"]:
        check_split_unchanged(task)
        task_dir = Path(task["output_dir"])
        task_dir.mkdir(parents=True, exist_ok=False)
        with Path(task["config_path"]).open("x", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(task_config(base, task, args), stream, allow_unicode=True, sort_keys=False)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = task["gpu_visible"]
        started = datetime.now(timezone.utc).isoformat()
        print(f"Starting {task['task_id']} on GPU {task['gpu_visible']}", flush=True)
        log_path = log_dir / (task["task_id"].replace("/", "__") + ".log")
        with log_path.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(
                task["command"], cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        write_json_new(task_dir / "process.json", {
            "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
        })
        if completed.returncode:
            print(f"FAILED: {task['task_id']}; see {log_path}", file=sys.stderr)
            return completed.returncode if completed.returncode > 0 else 1
        try:
            check_completion(task_dir)
        except PlanError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    write_json_new(batch_dir / "completed.json", {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "tasks_completed": len(plan["tasks"]),
    })
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        base = read_config(args.config)
        arms = selected_arms(args.group, args.arm)
        studies = comma_values(args.cancer, ("blca", "kirc"))
        studies = list(dict.fromkeys(study.lower() for study in studies))
        folds = integer_values(args.fold, range(5), name="fold", maximum=4)
        seeds = integer_values(args.seed, (3,), name="seed")
        gpus = integer_values(args.gpu, (0,), name="gpu")
        if args.max_epochs is not None and args.max_epochs < 1:
            raise PlanError("--max-epochs must be >= 1.")
        name = args.batch_name or "e048_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        safe_name(name, label="batch-name")
        results_root = (args.results_root or Path(base["train"]["results_dir"])).expanduser().resolve()
        batch_dir = results_root / name
        if args.execute and batch_dir.exists():
            raise PlanError(f"Batch directory already exists; choose a fresh --batch-name: {batch_dir}")
        if args.plan_out and args.plan_out.exists():
            raise PlanError(f"Plan output already exists; refusing overwrite: {args.plan_out}")
        if args.execute and args.plan_out and args.plan_out.resolve().is_relative_to(batch_dir):
            raise PlanError("With --execute, --plan-out must be outside the new batch directory; batch/plan.json is saved automatically.")
        tasks = build_tasks(
            arms=arms, studies=studies, folds=folds, seeds=seeds, gpus=gpus,
            batch_dir=batch_dir, data_path=Path(args.data_path or base["data"]["data_path"]),
            split_set=args.split_set or base["split_setup"]["which_splits"], protocol=args.protocol,
            python_executable=sys.executable, worker_path=HERE / "worker.py", require_splits=args.execute,
        )
        if args.execute and not (HERE / "worker.py").is_file():
            raise PlanError(f"Worker is missing: {HERE / 'worker.py'}")
        plan = {
            "schema_version": 1, "mode": "execute" if args.execute else "plan",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "batch_dir": str(batch_dir),
            "protocol": args.protocol, "base_config": str(args.config.resolve()),
            "selected_arms": arms, "studies": studies, "folds": folds, "seeds": seeds,
            "tasks": tasks,
        }
        display_plan(plan)
        if args.plan_out:
            write_json_new(args.plan_out, plan)
        return execute_plan(plan, base, args) if args.execute else 0
    except (PlanError, OSError, yaml.YAMLError) as exc:
        print(f"E048 ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
