"""Command line interface for the standalone DCT-Reg package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import apply_overrides, config_to_argv, load_config
from .project import PROJECT_ROOT, add_project_paths
from .research.methods.catalog import (
    CATALOG_UPDATED,
    METHOD_STATUSES,
    STATUS_LABELS,
    catalog_errors,
    iter_method_specs,
)
# Module-level import so unit tests can monkeypatch the runner without
# touching internal function-scope bindings inside cmd_train.
from .training.train_runner import run


def cmd_train(args: argparse.Namespace) -> None:
    add_project_paths()
    config = load_config(args.config)
    config = apply_overrides(config, args.set or [])
    user_keys = list(config.get("_dct_user_overrides", []))
    extra_args = args.extra_args or []
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    argv = config_to_argv(config) + extra_args

    from survot_rank.training.extended_args import process_args_extended
    parsed = process_args_extended(argv)
    # Forward the explicit `--set` overrides so model constructors can
    # distinguish "user-supplied" FROZEN_ARGUMENT keys (ablation studies).
    parsed._dct_user_overrides = set(user_keys)

    # Respect a CUDA_VISIBLE_DEVICES already set by the scheduler; only
    # fall back to parsed.gpu when the env was not pre-set.  Without this
    # guard, every CLI invocation would force GPU 0 regardless of the
    # scheduler's round-robin assignment (two CLI workers would collide on
    # the same physical GPU).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = parsed.gpu

    run(parsed)


def cmd_doctor(args: argparse.Namespace) -> None:
    add_project_paths()
    method_errors = catalog_errors(PROJECT_ROOT)
    checks = {
        "project_root": PROJECT_ROOT.exists(),
        "training": (PROJECT_ROOT / "survot_rank" / "training" / "train_runner.py").exists(),
        "dct_reg_method": (
            PROJECT_ROOT / "survot_rank" / "research" / "methods"
            / "dct_v310_directional_regularized_transport" / "model.py"
        ).exists(),
        "parent_model": (
            PROJECT_ROOT / "survot_rank" / "research" / "methods" / "ot_event_hazard_v2" / "model_v2.py"
        ).exists(),
        "compat_dataset": (
            PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset" / "dataset_survival.py"
        ).exists(),
        "compat_utils": (
            PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "utils" / "loss_func.py"
        ).exists(),
        "method_catalog": not method_errors,
    }
    for name, ok in checks.items():
        status = "OK" if ok else "MISSING"
        print(f"{status:8s} {name}")
    for error in method_errors:
        print(f"ERROR    {error}")


def cmd_methods(args: argparse.Namespace) -> None:
    """Print the executable method catalog without importing model code."""

    specs = list(iter_method_specs(args.status))
    if args.json:
        payload = [
            {
                "key": spec.key,
                "name": spec.display_name,
                "family": spec.family,
                "status": spec.status,
                "aliases": list(spec.aliases),
                "code": str(Path(spec.method_dir) / spec.model_file),
            }
            for spec in specs
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"DCT-Reg method catalog (updated {CATALOG_UPDATED})")
    for status in METHOD_STATUSES:
        status_specs = [spec for spec in specs if spec.status == status]
        if not status_specs:
            continue
        print(f"\n[{status}] {STATUS_LABELS[status]}")
        for spec in status_specs:
            aliases = ", ".join(spec.aliases) if spec.aliases else "-"
            print(f"  {spec.display_name}")
            print(f"    key: {spec.key}")
            print(f"    family: {spec.family}; aliases: {aliases}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dct-reg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Run training from a YAML config")
    train.add_argument("--config", required=True, help="Path to a YAML experiment config")
    train.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override one flat parameter, for example --set seed=5",
    )
    train.add_argument("extra_args", nargs=argparse.REMAINDER)
    train.set_defaults(func=cmd_train)

    doctor = subparsers.add_parser("doctor", help="Check expected project files")
    doctor.set_defaults(func=cmd_doctor)

    methods = subparsers.add_parser("methods", help="List registered methods and research roles")
    methods.add_argument("--status", choices=METHOD_STATUSES, help="Show one research role only")
    methods.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    methods.set_defaults(func=cmd_methods)

    # ------------------------------------------------------------------
    # §3.2: unified experiment scheduler.
    #
    # ``dct-reg schedule`` is the dry-run-by-default launcher for the
    # v3.13 experiment matrix.  Filters: protocol / arm / cancer /
    # fold / seed.  GPUs: --gpu 0,1 and --jobs-per-gpu N control
    # per-GPU subprocess fan-out.  --execute flips from dry-run
    # (default) to actually launching the subprocesses.
    # ------------------------------------------------------------------
    schedule = subparsers.add_parser(
        "schedule",
        help="§3.2 unified experiment scheduler (dry-run by default).",
    )
    schedule.add_argument(
        "--protocol",
        choices=["legacy_val", "outer_test"],
        default="legacy_val",
        help="Evaluation protocol (default legacy_val).",
    )
    schedule.add_argument(
        "--arm",
        action="append",
        default=None,
        help=(
            "Experiment arm(s) to schedule (default: all 11 v3.13 arms). "
            "Repeatable.  Examples: --arm exp6 --arm slotspe_native."
        ),
    )
    schedule.add_argument(
        "--cancer",
        action="append",
        default=None,
        help="Cancer / study cohort(s) (default: blca,kirc).  Repeatable.",
    )
    schedule.add_argument(
        "--fold",
        action="append",
        type=int,
        default=None,
        help="Fold index / indices to schedule (default: 0..4).  Repeatable.",
    )
    schedule.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Model seed(s) to schedule (default: 3).  Repeatable.",
    )
    schedule.add_argument(
        "--gpu",
        action="append",
        default=None,
        help=(
            "Physical GPU indices to assign to running subprocesses.  "
            "Default: '0'.  Accepts repeated flags or a comma-separated "
            "list: --gpu 0 --gpu 1  or  --gpu 0,1."
        ),
    )
    schedule.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Concurrent training subprocesses per GPU.  Default 1.",
    )
    schedule.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually launch the subprocesses.  Without this flag the "
            "scheduler prints the task list and exits (dry-run)."
        ),
    )
    schedule.set_defaults(func=cmd_schedule)

    return parser


def cmd_schedule(args: argparse.Namespace) -> None:
    """Delegate to ``survot_rank.training.scheduler.cmd_schedule``."""
    from survot_rank.training.scheduler import cmd_schedule as _impl

    _impl(args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
