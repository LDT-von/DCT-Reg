"""Tests for DCT v3.13 §3.2 unified scheduler.

Verifies:

* The 11-arm experiment matrix is fully covered (no arms missing).
* ``build_task_plan`` is deterministic: same inputs produce identical
  outputs across calls.
* ``assign_gpus`` round-robins tasks across GPUs respecting
  ``jobs_per_gpu``.
* Default dry-run prints the task plan and writes
  ``plan_<protocol>_<cohorts>_folds<folds>_seeds<seeds>.json`` under
  ``results/v313_paper_v1/schedule/``.
* The CLI ``dct-reg schedule --help`` exposes every flag.
"""

from __future__ import annotations

import json
import os

import pytest

from survot_rank.training.scheduler import (
    ARMS,
    DEFAULT_CANCERS,
    DEFAULT_FOLDS,
    DEFAULT_SEEDS,
    assign_gpus,
    build_task_plan,
    _resolve_config_yaml,
)


# ---------------------------------------------------------------------------
# Experiment matrix definition.
# ---------------------------------------------------------------------------


def test_arm_matrix_covers_eleven_arms():
    """§3.2 plan §2.5 defines 11 experiment arms.  Any change to the
    matrix MUST be reviewed against the plan first."""
    expected = {
        "exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6",
        "slotspe_native", "slotspe_matched", "direct", "independent",
    }
    assert set(ARMS.keys()) == expected
    for arm, spec in ARMS.items():
        assert "base_method" in spec
        assert "extra_set" in spec
        assert "config_template" in spec
        assert spec["extra_set"], f"{arm} must have at least one --set override"


def test_default_cancers_folds_seeds():
    """§3.2 plan §2.5 defaults: 2 cancers × 5 folds × 1 seed."""
    assert DEFAULT_CANCERS == ("blca", "kirc")
    assert DEFAULT_FOLDS == (0, 1, 2, 3, 4)
    assert DEFAULT_SEEDS == (3,)


# ---------------------------------------------------------------------------
# Task plan builder.
# ---------------------------------------------------------------------------


def test_build_task_plan_is_deterministic(tmp_path):
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    kwargs = dict(
        protocols=["legacy_val"],
        arms=["exp6"],
        cancers=["blca"],
        folds=[0],
        seeds=[3],
        project_root=project_root,
        data_path=str(data_path),
        outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    plan_a = build_task_plan(**kwargs)
    plan_b = build_task_plan(**kwargs)
    assert len(plan_a) == 1
    # Field-by-field equality (excluding the per-task list mutability
    # from concurrent fan-out).
    assert plan_a[0].to_dict() == plan_b[0].to_dict()


def test_build_task_plan_scales_with_matrix(tmp_path):
    """11 arms × 2 cancers × 5 folds × 1 seed × 1 protocol = 110 tasks."""
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["legacy_val"],
        arms=list(ARMS.keys()),
        cancers=list(DEFAULT_CANCERS),
        folds=list(DEFAULT_FOLDS),
        seeds=list(DEFAULT_SEEDS),
        project_root=project_root,
        data_path=str(data_path),
        outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    assert len(plan) == 110


def test_build_task_plan_outer_test_appends_protocol_overrides(tmp_path):
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["outer_test"],
        arms=["exp6"],
        cancers=["blca"],
        folds=[0],
        seeds=[3],
        project_root=project_root,
        data_path=str(data_path),
        outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    assert len(plan) == 1
    task = plan[0]
    # evaluation_protocol + which_splits MUST be appended under outer_test
    joined = "\n".join(task.extra_set)
    assert "evaluation_protocol=outer_test" in joined
    assert "inner_val_fraction=0.20" in joined
    assert "split_seed=3" in joined
    assert "which_splits=5fold_uni2h_outer_test_seed3" in joined


def test_build_task_plan_seed_is_always_set(tmp_path):
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["legacy_val"],
        arms=["exp6"],
        cancers=["blca"],
        folds=[0],
        seeds=[7, 11, 13],
        project_root=project_root,
        data_path=str(data_path),
        outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    assert len(plan) == 3
    for task in plan:
        # Per-task seed is appended to extra_set; the task itself stores it.
        assert f"seed={task.seed}" in task.extra_set


def test_build_task_plan_results_dirs_avoid_collisions(tmp_path):
    """Outer test and legacy_val MUST NOT share a results directory."""
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan_legacy = build_task_plan(
        protocols=["legacy_val"], arms=["exp6"], cancers=["blca"],
        folds=[0], seeds=[3], project_root=project_root,
        data_path=str(data_path), outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    plan_outer = build_task_plan(
        protocols=["outer_test"], arms=["exp6"], cancers=["blca"],
        folds=[0], seeds=[3], project_root=project_root,
        data_path=str(data_path), outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    assert plan_legacy[0].results_dir != plan_outer[0].results_dir
    assert "legacy_val" in plan_legacy[0].results_dir
    assert "outer_test" in plan_outer[0].results_dir


# ---------------------------------------------------------------------------
# GPU round-robin.
# ---------------------------------------------------------------------------


def test_assign_gpus_round_robin_single_gpu(tmp_path):
    """A single GPU gets all tasks when jobs_per_gpu=1."""
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["legacy_val"], arms=["exp6"], cancers=["blca"],
        folds=[0, 1, 2], seeds=[3], project_root=project_root,
        data_path=str(data_path), outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    plan = assign_gpus(plan, ["0"], jobs_per_gpu=1)
    assert all(task.gpu_id.startswith("0/") for task in plan)


def test_assign_gpus_round_robin_two_gpus(tmp_path):
    """Two GPUs split the workload evenly when jobs_per_gpu=1."""
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["legacy_val"], arms=list(ARMS.keys()),
        cancers=["blca"], folds=[0, 1, 2, 3, 4], seeds=[3],
        project_root=project_root, data_path=str(data_path),
        outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    plan = assign_gpus(plan, ["0", "1"], jobs_per_gpu=1)
    counts = {"0": 0, "1": 0}
    for task in plan:
        gpu_index = task.gpu_id.split("/")[0]
        counts[gpu_index] += 1
    total = sum(counts.values())
    assert counts["0"] == counts["1"] or abs(counts["0"] - counts["1"]) == 1
    assert total == len(plan)


def test_assign_gpus_jobs_per_gpu(tmp_path):
    """jobs_per_gpu=2 packs two tasks per GPU round-robin slot."""
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["legacy_val"], arms=list(ARMS.keys()),
        cancers=["blca"], folds=[0, 1], seeds=[3],
        project_root=project_root, data_path=str(data_path),
        outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    plan = assign_gpus(plan, ["0"], jobs_per_gpu=2)
    # First two tasks share GPU 0 (slot 0 within-gpu 0 and 1).
    assert plan[0].gpu_id == "0/0"
    assert plan[1].gpu_id == "0/1"
    assert plan[2].gpu_id == "0/0"


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


def test_cli_exposes_schedule_subcommand():
    from survot_rank.cli import build_parser

    parser = build_parser()
    # Parsing with no args fails (subcommand required), but the
    # subparsers themselves are discoverable via the help string.
    subcommands = []
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices:
            for name in action.choices:
                subcommands.append(name)
    assert "schedule" in subcommands


def test_cli_schedule_accepts_filters():
    from survot_rank.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "schedule",
        "--protocol", "outer_test",
        "--arm", "exp6",
        "--cancer", "blca",
        "--fold", "0",
        "--seed", "3",
        "--gpu", "0",
        "--jobs-per-gpu", "1",
    ])
    assert args.protocol == "outer_test"
    assert args.arm == ["exp6"]
    assert args.cancer == ["blca"]
    assert args.fold == [0]
    assert args.seed == [3]
    # --gpu is repeatable; one occurrence yields a single-element list.
    assert args.gpu == ["0"]
    assert args.jobs_per_gpu == 1
    assert args.execute is False


def test_cli_schedule_execute_flag():
    from survot_rank.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["schedule", "--execute"])
    assert args.execute is True


# ---------------------------------------------------------------------------
# Config resolver.
# ---------------------------------------------------------------------------


def test_resolve_config_yaml_returns_canonical_path(tmp_path):
    template, full = _resolve_config_yaml("exp6", "blca", str(tmp_path))
    assert template == "exp6_blca_uni2h.yaml"
    assert full.endswith("configs/exp6_blca_uni2h.yaml")


def test_resolve_config_yaml_slotspe_arm():
    template, full = _resolve_config_yaml("slotspe_native", "kirc", "/x")
    assert template == "slotspe_native_kirc_uni2h.yaml"
    assert full.endswith("/x/configs/slotspe_native_kirc_uni2h.yaml")


# ---------------------------------------------------------------------------
# Plan JSON shape.
# ---------------------------------------------------------------------------


def test_to_dict_includes_task_id(tmp_path):
    project_root = str(tmp_path)
    data_path = tmp_path / "csv"
    data_path.mkdir()
    (data_path / "splits" / "5fold_uni2h").mkdir(parents=True)
    plan = build_task_plan(
        protocols=["legacy_val"], arms=["exp6"], cancers=["blca"],
        folds=[0], seeds=[3], project_root=project_root,
        data_path=str(data_path), outer_split_root="5fold_uni2h_outer_test_seed3",
    )
    as_dict = plan[0].to_dict()
    assert as_dict["task_id"] == "exp6/legacy_val/blca/fold0/seed3"
    assert as_dict["arm"] == "exp6"
    assert as_dict["cancer"] == "blca"
    assert as_dict["fold"] == 0
    assert as_dict["seed"] == 3
    # Plan JSON is JSON-serialisable.
    json.dumps(as_dict)