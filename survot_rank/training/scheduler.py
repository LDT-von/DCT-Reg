"""Unified experiment scheduler for DCT v3.13 §3.2.

Implements the v3.13 plan §4.1–§4.2 unified CLI entry:

* ``--protocol legacy_val|outer_test``
* ``--arm <name>`` (repeatable; default: all 11 v3.13 arms)
* ``--cancer <study>`` (repeatable; default: ``blca,kirc``)
* ``--fold <int>`` (repeatable; default: 0..4)
* ``--seed <int>`` (repeatable; default: 3)
* ``--gpu 0 --gpu 1`` (repeatable; comma-separated indices also supported)
* ``--jobs-per-gpu N`` (concurrent subprocesses per GPU; default 1)
* ``--results-root PATH`` (optional root for a separate result batch)
* ``--execute`` (actually run; without this flag the scheduler prints
  the task list and exits — dry-run is the safe default)

The scheduler emits a deterministic, fingerprint-stable task plan.  Every
task is a ``(arm, protocol, study, fold, seed)`` tuple together with its
target results path, the resolved YAML config path, and the SHA-256
fingerprint of the parent split CSV (so leakage between the inner_train
and outer_test cohorts can be checked downstream).

Each GPU has its own thread pool of child-process launchers, capped by
``--jobs-per-gpu``.  Each subprocess invokes the existing ``train``
subcommand with the resolved config, per-arm overrides, and the task's
single-fold range and output path.

The scheduler does NOT touch the results directory on disk — subprocesses
do that — so a dry-run is genuinely a no-op on disk beyond writing a
plan JSON under ``results/v313_paper_v1/schedule``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import ExitStack
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Experiment matrix definition.
# ---------------------------------------------------------------------------

# The 11 experiment arms defined in paper/V313_IMPLEMENTATION_PLAN.md §2.5.
# Each arm describes:
#   - config_yaml: a stable config filename pattern under configs/.
#   - base_method: the survot_method key passed via --set.
#   - extra_set:   additional --set overrides to apply on top of the YAML.
#   - description: human-readable description for the dry-run output.
#
# The default BLCA/KIRC matrix expands to 11 arms × 2 cancers × 5 folds × 1
# seed = 110 tasks per protocol; both protocols combined = 220 tasks.

ARMS: dict[str, dict] = {
    # ---- 2.1 loss ablation (7 arms) ----
    "exp0": {
        "display": "Exp0 — NLL only",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_lambda_ipcw_rank=0.0", "dct_v311_lambda_slot_nll=0.0",
                      "dct_v311_lambda_slot_diversity=0.0",
                      "dct_v313_disable_self_reconstruction=true",
                      "dct_v313_disable_cross_reconstruction=true",
                      "dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    "exp1": {
        "display": "Exp1 — +0.10 IPCW rank",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v311_lambda_slot_nll=0.0",
                      "dct_v311_lambda_slot_diversity=0.0",
                      "dct_v313_disable_self_reconstruction=true",
                      "dct_v313_disable_cross_reconstruction=true",
                      "dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    "exp2": {
        "display": "Exp2 — +0.05 per-slot NLL",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v311_lambda_slot_diversity=0.0",
                      "dct_v313_disable_self_reconstruction=true",
                      "dct_v313_disable_cross_reconstruction=true",
                      "dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    "exp3": {
        "display": "Exp3 — +0.10 diversity (full pre-recon recipe)",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v313_disable_self_reconstruction=true",
                      "dct_v313_disable_cross_reconstruction=true",
                      "dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    "exp4": {
        "display": "Exp4 — +0.05 self recon only",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v313_disable_cross_reconstruction=true",
                      "dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    "exp5": {
        "display": "Exp5 — +0.05 cross recon only",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v313_disable_self_reconstruction=true",
                      "dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    "exp6": {
        "display": "Exp6 — full v3.13 (0.05 self + 0.05 cross)",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v313_recon_weighting=per_branch"],
        "config_template": "{arm}_{study}_uni2h.yaml",
    },
    # ---- 2.2 SlotSPE recipes (2 arms) ----
    "slotspe_native": {
        "display": "SlotSPE native recipe",
        "base_method": "slotspe_reference",
        "extra_set": ["slotspe_recipe=native"],
        "config_template": "slotspe_native_{study}_uni2h.yaml",
    },
    "slotspe_matched": {
        "display": "SlotSPE matched recipe",
        "base_method": "slotspe_reference",
        "extra_set": ["slotspe_recipe=matched"],
        "config_template": "slotspe_matched_{study}_uni2h.yaml",
    },
    # ---- 2.3 direct cross (1 arm) ----
    "direct": {
        "display": "direct cross reconstruction (no OT)",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v313_cross_mode=direct", "dct_v313_recon_weighting=per_branch"],
        "config_template": "exp6_{study}_uni2h.yaml",
    },
    # ---- 2.4 independent coupling (1 arm) ----
    "independent": {
        "display": "independent coupling (T = a bᵀ)",
        "base_method": "dct_v313_transport_reconstruction",
        "extra_set": ["dct_v313_plan_mode=independent", "dct_v313_recon_weighting=per_branch"],
        "config_template": "exp6_{study}_uni2h.yaml",
    },
}

DEFAULT_CANCERS: tuple[str, ...] = ("blca", "kirc")
DEFAULT_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_SEEDS: tuple[int, ...] = (3,)


# ---------------------------------------------------------------------------
# Task plan dataclass.
# ---------------------------------------------------------------------------


@dataclass
class ScheduledTask:
    """One row of the v3.13 experiment matrix.

    Stable across dry-runs (the only fields that can change are
    ``gpu_id`` / ``jobs_per_gpu``-derived GPU assignment when the user
    changes ``--gpu`` and ``--jobs-per-gpu``).
    """

    arm: str
    protocol: str
    cancer: str
    fold: int
    seed: int
    config_yaml: str
    config_yaml_path: str
    extra_set: list[str] = field(default_factory=list)
    results_dir: str = ""
    split_fingerprint: str = ""
    gpu_id: str = ""

    def task_id(self) -> str:
        return f"{self.arm}/{self.protocol}/{self.cancer}/fold{self.fold}/seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self) | {"task_id": self.task_id()}


# ---------------------------------------------------------------------------
# Resolution helpers.
# ---------------------------------------------------------------------------


def _split_fingerprint(data_path: str, which_splits: str, study: str,
                       fold: int) -> str:
    """Stable SHA-256 over the parent split CSV (used for paper audit)."""
    path = os.path.join(data_path, "splits", which_splits, study, f"fold_{fold}.csv")
    if not os.path.exists(path):
        return "missing"
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _resolve_config_yaml(arm: str, study: str, project_root: str) -> tuple[str, str]:
    """Return (config_template, full path) for an arm + study pair.

    The full path may legitimately not exist yet for arms that the
    operator has not yet authored (e.g. ``exp0_kirc_uni2h.yaml``).  The
    scheduler still emits a task for it so the dry-run shows the full
    matrix — the task's ``config_yaml_path`` field surfaces the missing
    config so the operator knows what to author.
    """
    arm_spec = ARMS[arm]
    template = arm_spec["config_template"].format(arm=arm, study=study)
    full = os.path.join(project_root, "configs", template)
    return template, full


def _results_dir_for(arm: str, protocol: str, study: str, fold: int,
                     seed: int, project_root: str, results_root: str | None = None) -> str:
    """Stable, collision-free per-task results directory.

    Includes the protocol so the legacy_val and outer_test runs never
    share a directory even when the arm/cancer/fold/seed are equal.
    """
    root = results_root or os.path.join(project_root, "results", "v313_paper_v1")
    if not os.path.isabs(root):
        root = os.path.join(project_root, root)
    return os.path.join(
        os.path.abspath(root),
        protocol,
        arm,
        study,
        f"fold{fold}",
        f"seed{seed}",
    )


def build_task_plan(
    *,
    protocols: Iterable[str],
    arms: Iterable[str],
    cancers: Iterable[str],
    folds: Iterable[int],
    seeds: Iterable[int],
    project_root: str,
    data_path: str,
    outer_split_root: str,
    results_root: str | None = None,
) -> list[ScheduledTask]:
    """Build the deterministic task list.  Order: protocol → arm →
    cancer → fold → seed.  This ordering is stable across dry-runs so
    the task IDs and GPU assignments are reproducible."""
    tasks: list[ScheduledTask] = []
    for protocol in protocols:
        # outer_test uses the *_outer_test_seed3 split directory; legacy_val
        # uses the base ``which_splits`` directory in the YAML.
        for arm in arms:
            for cancer in cancers:
                for fold in folds:
                    for seed in seeds:
                        template, full = _resolve_config_yaml(arm, cancer, project_root)
                        arm_spec = ARMS[arm]
                        # Outer-test protocol also appends --set flags
                        # for evaluation_protocol + inner_val_fraction +
                        # split_seed so the train_runner routes correctly.
                        extra = list(arm_spec["extra_set"])
                        if protocol == "outer_test":
                            which = "5fold_uni2h_outer_test_seed3"
                            extra += [
                                "evaluation_protocol=outer_test",
                                "inner_val_fraction=0.20",
                                "split_seed=3",
                                f"which_splits={which}",
                            ]
                        # Always set the seed (stable across arm/fold).
                        extra += [f"seed={seed}"]
                        results_dir = _results_dir_for(
                            arm=arm, protocol=protocol, study=cancer,
                            fold=fold, seed=seed, project_root=project_root,
                            results_root=results_root,
                        )
                        # These values MUST reach the child CLI, rather than
                        # merely appearing as labels in the plan. The base YAML
                        # otherwise runs every fold into its shared base directory.
                        extra += [
                            f"survot_method={arm_spec['base_method']}",
                            f"study={cancer}",
                            f"k_start={fold}",
                            f"k_end={fold + 1}",
                            f"results_dir={results_dir}",
                            f"evaluation_protocol={protocol}",
                        ]
                        which_splits = (
                            outer_split_root if protocol == "outer_test" else "5fold_uni2h"
                        )
                        # _split_fingerprint needs which_splits; we use
                        # the actual split dir.
                        fingerprint = _split_fingerprint(
                            data_path=data_path,
                            which_splits=which_splits,
                            study=cancer,
                            fold=fold,
                        )
                        tasks.append(
                            ScheduledTask(
                                arm=arm,
                                protocol=protocol,
                                cancer=cancer,
                                fold=fold,
                                seed=seed,
                                config_yaml=template,
                                config_yaml_path=full,
                                extra_set=extra,
                                results_dir=results_dir,
                                split_fingerprint=fingerprint,
                            )
                        )
    return tasks


def assign_gpus(
    tasks: list[ScheduledTask],
    gpu_ids: list[str],
    jobs_per_gpu: int,
) -> list[ScheduledTask]:
    """Round-robin GPU assignment with ``jobs_per_gpu`` parallelism.

    Tasks get a stable ``gpu_id`` field that survives re-runs.  The
    scheduler only fills in ``gpu_id`` — actual subprocess launching
    happens in :func:`launch_tasks`.
    """
    if not gpu_ids:
        return tasks
    # Round-robin index over total slots = len(gpu_ids) * jobs_per_gpu.
    total_slots = len(gpu_ids) * jobs_per_gpu
    for i, task in enumerate(tasks):
        slot = i % total_slots
        gpu_index = slot // jobs_per_gpu
        within_gpu = slot % jobs_per_gpu
        task.gpu_id = f"{gpu_ids[gpu_index]}/{within_gpu}"
    return tasks


# ---------------------------------------------------------------------------
# Subprocess launcher.
# ---------------------------------------------------------------------------


def _build_launch_command(task: ScheduledTask, project_root: str) -> list[str]:
    """Build the exact argv for ``dct-reg train`` for one task."""
    cmd = [
        sys.executable, "-m", "survot_rank.cli", "train",
        "--config", task.config_yaml_path,
    ]
    for kv in task.extra_set:
        cmd += ["--set", kv]
    return cmd


def _launch_one_task(task: ScheduledTask, project_root: str) -> dict:
    """Run one task as a subprocess.  Returns a small status dict."""
    gpu_id = task.gpu_id.split("/")[0] if task.gpu_id else "0"
    env = os.environ.copy()
    # GPU allocation MUST happen before torch is imported in the
    # subprocess — set CUDA_VISIBLE_DEVICES here so the subprocess
    # only sees the assigned GPU.
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONPATH"] = project_root
    cmd = _build_launch_command(task, project_root=project_root)
    print(
        f"[launch] task_id={task.task_id()} gpu_id={gpu_id} cmd={' '.join(cmd[4:])[:200]}",
        flush=True,
    )
    try:
        completed = subprocess.run(cmd, env=env, cwd=project_root, capture_output=True, text=True)
        return {
            "task_id": task.task_id(),
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-400:] if completed.stderr else "",
        }
    except Exception as exc:
        return {"task_id": task.task_id(), "returncode": -1, "error": repr(exc)}


def launch_tasks(
    tasks: list[ScheduledTask],
    project_root: str,
    *,
    gpus: list[str],
    jobs_per_gpu: int,
    execute: bool,
) -> dict:
    """Either dispatch the task subprocesses (execute=True) or print the
    dry-run plan (execute=False).

    Returns a summary dict with the task count and per-task status.
    """
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be positive")
    gpus = list(dict.fromkeys(gpus)) or ["0"]
    tasks = assign_gpus(tasks, gpus, jobs_per_gpu)
    summary = {
        "total": len(tasks),
        "executed": 0,
        "failed": 0,
        "results": [],
    }
    if not execute:
        # Dry-run: just print the plan and exit.
        for task in tasks:
            exists = os.path.exists(task.config_yaml_path)
            print(
                f"[dry-run] {task.task_id()} | gpu={task.gpu_id} | "
                f"results={task.results_dir} | config_exists={exists}"
            )
        return summary

    # A global pool does not enforce each GPU's limit when task durations
    # differ. Each GPU needs its own queue. Threads only wait on subprocesses;
    # model execution and CUDA environments remain isolated in the children.
    with ExitStack() as stack:
        pools = {
            gpu: stack.enter_context(
                concurrent.futures.ThreadPoolExecutor(max_workers=jobs_per_gpu)
            ) for gpu in gpus
        }
        futures = {
            pools[task.gpu_id.split("/")[0]].submit(_launch_one_task, task, project_root): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            summary["results"].append(result)
            if result.get("returncode") == 0:
                summary["executed"] += 1
            else:
                summary["failed"] += 1
                print(
                    f"[FAIL] {result['task_id']} "
                    f"returncode={result.get('returncode')} "
                    f"stderr_tail={result.get('stderr_tail', '')!r}"
                )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point (wired into survot_rank.cli.cmd_schedule).
# ---------------------------------------------------------------------------


def cmd_schedule(args: argparse.Namespace) -> None:
    """Top-level entry for ``dct-reg schedule``."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = "/data1/dataset_csv"
    outer_split_root = "5fold_uni2h_outer_test_seed3"

    arms = list(args.arm) if args.arm else list(ARMS.keys())
    unknown_arms = set(arms) - set(ARMS.keys())
    if unknown_arms:
        raise SystemExit(
            f"Unknown arm(s): {sorted(unknown_arms)}; valid arms: "
            f"{sorted(ARMS.keys())}"
        )
    cancers = list(args.cancer) if args.cancer else list(DEFAULT_CANCERS)
    folds = list(args.fold) if args.fold else list(DEFAULT_FOLDS)
    seeds = list(args.seed) if args.seed else list(DEFAULT_SEEDS)
    if args.gpu is None:
        gpus = ["0"]
    else:
        # Accept repeated --gpu flags (each item can itself be a
        # comma-separated list) and flatten into a single ordered list.
        gpus = []
        for chunk in args.gpu:
            gpus.extend(g.strip() for g in chunk.split(",") if g.strip())
        if not gpus:
            gpus = ["0"]
    protocols = [args.protocol]

    tasks = build_task_plan(
        protocols=protocols,
        arms=arms,
        cancers=cancers,
        folds=folds,
        seeds=seeds,
        project_root=project_root,
        data_path=data_path,
        outer_split_root=outer_split_root,
        results_root=getattr(args, "results_root", None),
    )
    tasks = assign_gpus(tasks, list(dict.fromkeys(gpus)), args.jobs_per_gpu)

    # Always dump the task plan to a JSON file under results/schedule/ so
    # the dry-run leaves a paper-auditable artefact.
    plan_dir = os.path.join(project_root, "results", "v313_paper_v1", "schedule")
    os.makedirs(plan_dir, exist_ok=True)
    plan_path = os.path.join(
        plan_dir,
        f"plan_{args.protocol}_{','.join(cancers)}_folds{','.join(str(f) for f in folds)}_seeds{','.join(str(s) for s in seeds)}.json",
    )
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(
            [task.to_dict() for task in tasks],
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[schedule] {len(tasks)} tasks planned")
    print(f"[schedule] arms: {arms}")
    print(f"[schedule] cancers: {cancers}")
    print(f"[schedule] folds: {folds}")
    print(f"[schedule] seeds: {seeds}")
    print(f"[schedule] gpus: {gpus} (jobs_per_gpu={args.jobs_per_gpu})")
    print(f"[schedule] protocol: {args.protocol}")
    print(f"[schedule] task plan written to: {plan_path}")

    summary = launch_tasks(
        tasks=tasks,
        project_root=project_root,
        gpus=gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        execute=bool(args.execute),
    )
    print(
        f"[schedule] executed={summary['executed']} "
        f"failed={summary['failed']} total={summary['total']}"
    )
