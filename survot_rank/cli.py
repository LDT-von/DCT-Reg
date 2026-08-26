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


def cmd_train(args: argparse.Namespace) -> None:
    add_project_paths()
    config = load_config(args.config)
    config = apply_overrides(config, args.set or [])
    extra_args = args.extra_args or []
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    argv = config_to_argv(config) + extra_args

    # Set CUDA_VISIBLE_DEVICES before any torch import so that
    # torch.cuda.is_available() and device count reflect the correct GPU.
    from survot_rank.training.extended_args import process_args_extended
    parsed = process_args_extended(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = parsed.gpu

    from survot_rank.training.train_runner import run
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
        "legacy_dataset": (
            PROJECT_ROOT / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset" / "dataset_survival.py"
        ).exists(),
        "legacy_utils": (
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
