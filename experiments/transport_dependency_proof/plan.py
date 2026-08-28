#!/usr/bin/env python3
"""Plan and inspect the DCT-Reg transport-dependency proof suite.

This entry point does not train models by itself.  It delegates matched training
to the canonical launchers and generates held-out audit commands only for
checkpoints that already exist.  That separation prevents a plan command from
silently turning structural readiness into completed experimental evidence.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
MATRIX_PATH = PACKAGE_ROOT / "MATRIX.csv"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
CONFIG = REPO_ROOT / "configs" / "dct_v310_directional_regularized_transport.yaml"
KNOWN_CANCERS = ("blca", "ucec", "kirc", "hnsc", "skcm", "lusc")


@dataclass(frozen=True)
class Checkpoint:
    cancer: str
    fold: int
    path: Path


def proof_training_commands(python_bin: str = sys.executable) -> list[tuple[str, tuple[str, ...]]]:
    """Return the three canonical training queues in pre-registered order."""

    return [
        (
            "P1_objective_2x2",
            (
                python_bin,
                "scripts/run_dct_v310_experiments.py",
                "plan",
                "--cancers",
                "blca",
                "--folds",
                "0,1,2,3,4",
                "--variants",
                "nll_only,ipcw_only,direction_only,full",
            ),
        ),
        (
            "P2_mechanism_controls",
            (
                python_bin,
                "scripts/run_dct_v310_experiments.py",
                "plan",
                "--cancers",
                "blca,ucec,lusc",
                "--folds",
                "1,2,4",
                "--variants",
                "full,fixed_coupling,noisy_batch_mean_anchors,permuted_reference,stage_jitter",
            ),
        ),
    ]


def _infer_checkpoint(path: Path) -> Checkpoint | None:
    lowered_parts = [part.lower() for part in path.parts]
    cancer = next((item for item in KNOWN_CANCERS if item in lowered_parts), None)
    fold_match = re.search(r"fold[_-](\d+)", path.as_posix(), flags=re.IGNORECASE)
    if cancer is None or fold_match is None:
        return None
    return Checkpoint(cancer=cancer, fold=int(fold_match.group(1)), path=path.resolve())


def discover_full_checkpoints(results_root: Path) -> list[Checkpoint]:
    """Discover only full/final DCT-Reg checkpoints, excluding null controls."""

    controls = {
        "nll_only",
        "ipcw_only",
        "direction_only",
        "fixed_coupling",
        "noisy_batch_mean_anchors",
        "permuted_reference",
        "stage_jitter",
    }
    discovered: list[Checkpoint] = []
    if not results_root.exists():
        return discovered
    for path in results_root.rglob("checkpoint.pt"):
        parts = {part.lower() for part in path.parts}
        if parts & controls:
            continue
        if "full" not in parts and "final" not in parts:
            continue
        checkpoint = _infer_checkpoint(path)
        if checkpoint is not None:
            discovered.append(checkpoint)
    return sorted(discovered, key=lambda item: (item.cancer, item.fold, str(item.path)))


def _audit_base(checkpoint: Checkpoint, python_bin: str) -> list[str]:
    return [
        python_bin,
        "scripts/audit_dct_reg.py",
        "audit",
        "--config",
        CONFIG.relative_to(REPO_ROOT).as_posix(),
        "--checkpoint",
        str(checkpoint.path),
        "--fold",
        str(checkpoint.fold),
        "--epoch",
        "29",
        "--set",
        f"study={checkpoint.cancer}",
    ]


def checkpoint_audit_commands(
    checkpoint: Checkpoint, python_bin: str = sys.executable
) -> list[tuple[str, tuple[str, ...]]]:
    """Build factual, plan-control, anchor-swap and bidirectional dose audits."""

    evidence_root = checkpoint.path.parent
    proof_root = evidence_root / "proof_transport_dependency"
    commands: list[tuple[str, tuple[str, ...]]] = []
    for name, plan_control, anchor_mode in (
        ("factual", "none", "normal"),
        ("uniform_plan", "uniform", "normal"),
        ("shuffled_plan", "shuffled", "normal"),
        ("anchor_swap", "none", "swapped"),
    ):
        command = _audit_base(checkpoint, python_bin)
        command.extend(
            (
                "--output-dir",
                str(proof_root / name),
                "--plan-control",
                plan_control,
                "--anchor-mode",
                anchor_mode,
            )
        )
        commands.append((name, tuple(command)))

    sweep = [
        python_bin,
        "scripts/audit_dct_reg.py",
        "sweep",
        "--config",
        CONFIG.relative_to(REPO_ROOT).as_posix(),
        "--checkpoint",
        str(checkpoint.path),
        "--fold",
        str(checkpoint.fold),
        "--epoch",
        "29",
        "--alphas",
        "0.0,0.25,0.5,0.75,1.0",
        "--output-dir",
        str(proof_root / "dose_both_directions"),
        "--anchor-mode",
        "normal",
        "--set",
        f"study={checkpoint.cancer}",
    ]
    commands.append(("dose_both_directions", tuple(sweep)))
    return commands


def load_matrix() -> list[dict[str, str]]:
    with open(MATRIX_PATH, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_status(results_root: Path) -> None:
    print("DCT-REG TRANSPORT-DEPENDENCY PROOF STATUS")
    for row in load_matrix():
        print(
            f"{row['experiment_id']}: code={row['implementation_status']} "
            f"run={row['execution_status']} - {row['question']}"
        )
    checkpoints = discover_full_checkpoints(results_root)
    audit_metrics = list(results_root.rglob("proof_transport_dependency/*/audit_metrics.json")) \
        if results_root.exists() else []
    dose_metrics = list(results_root.rglob("proof_transport_dependency/*/dose_metrics.json")) \
        if results_root.exists() else []
    print(f"discovered_full_checkpoints={len(checkpoints)}")
    print(f"completed_mechanism_audits={len(audit_metrics)}")
    print(f"completed_dose_audits={len(dose_metrics)}")


def print_plan(results_root: Path, python_bin: str) -> None:
    print("DCT-REG TRANSPORT-DEPENDENCY PROOF PLAN")
    print("\nTRAINING QUEUES")
    for experiment_id, command in proof_training_commands(python_bin):
        print(f"[{experiment_id}]\n  {shlex.join(command)}")

    checkpoints = discover_full_checkpoints(results_root)
    print("\nHELD-OUT CHECKPOINT AUDITS")
    if not checkpoints:
        print(
            "No full/final formal checkpoint was found. Run the training queues first; "
            "audit commands are intentionally not fabricated without a real checkpoint."
        )
        return
    for checkpoint in checkpoints:
        print(f"\n[{checkpoint.cancer.upper()} fold{checkpoint.fold}] {checkpoint.path}")
        for name, command in checkpoint_audit_commands(checkpoint, python_bin):
            print(f"  {name}: {shlex.join(command)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dct-proof")
    parser.add_argument("mode", choices=("plan", "status"), nargs="?", default="plan")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "status":
        print_status(args.results_root.resolve())
    else:
        print_plan(args.results_root.resolve(), args.python_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
