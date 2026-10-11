"""CSV/plan/launch safety tests using temporary text files and mocked workers.

No dataset, model, CUDA, checkpoint, training or inference is used here.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import run_experiments as runner
from schedule_core import (
    ALL_ARMS, ARM_GROUPS, PlanError, build_tasks, check_split_unchanged,
    comma_values, integer_values, selected_arms, split_path, validate_split,
)


class ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="e048_schedule_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def csv(self, name: str = "split.csv", contents: str = "train,val\nP1,P2\nP3,\n") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def tasks(self, **overrides):
        self.csv("data/splits/fixed/blca/fold_0.csv")
        options = dict(
            arms=["full", "no_self"], studies=["blca"], folds=[0], seeds=[3], gpus=[0],
            batch_dir=self.root / "batch", data_path=self.root / "data", split_set="fixed",
            protocol="legacy_val", python_executable="python", worker_path=HERE / "worker.py",
            require_splits=True,
        )
        options.update(overrides)
        return build_tasks(**options)

    def arguments(self):
        return argparse.Namespace(data_path=None, data_root=None, split_set=None, max_epochs=None)

    def test_registered_groups_and_explicit_arm_parsing(self):
        self.assertEqual(len(ALL_ARMS), 17)
        self.assertEqual(len(ARM_GROUPS["first"]), 7)
        self.assertEqual(selected_arms("all", ["full,no_self", "full,no_cross"]),
                         ["full", "no_self", "no_cross"])
        self.assertEqual(comma_values(["BLCA", "KIRC,BLCA"], ()), ["BLCA", "KIRC"])
        with self.assertRaises(PlanError):
            selected_arms("first", ["typo"])
        with self.assertRaises(PlanError):
            integer_values(["5"], (), name="fold", maximum=4)
        with self.assertRaises(PlanError):
            integer_values(["-1"], (), name="seed")

    def test_split_hash_sizes_and_bom_are_read_from_identical_bytes(self):
        path = self.csv(contents="\ufeffUnnamed: 0,train,val\n0,P1,P2\n1,P3,\n")
        metadata = validate_split(path, "legacy_val")
        self.assertEqual(metadata["sizes"], {"train": 2, "val": 1})
        self.assertEqual(len(metadata["sha256"]), 64)
        path.write_text("train,val\nP1,P2\nP3,\n", encoding="utf-8")
        self.assertNotEqual(metadata["sha256"], validate_split(path, "legacy_val")["sha256"])

    def test_patient_leakage_detects_different_tcga_samples(self):
        path = self.csv(contents="train,val\nTCGA-AB-1234-01A,tcga-ab-1234-02A\n")
        with self.assertRaisesRegex(PlanError, "overlap"):
            validate_split(path, "legacy_val")

    def test_duplicate_and_empty_partition_are_rejected(self):
        for contents in ("train,val\nP1,P2\nP1,\n", "train,val\nP1,\n"):
            with self.subTest(contents=contents), self.assertRaises(PlanError):
                validate_split(self.csv(contents=contents), "legacy_val")

    def test_outer_test_requires_explicit_inner_and_outer_columns(self):
        path = self.csv(contents="train,val,test\nP1,P2,P3\n")
        with self.assertRaisesRegex(PlanError, "inner_train"):
            validate_split(path, "outer_test")
        path = self.csv(contents="inner_train,inner_val,outer_test\nP1,P2,P3\n")
        self.assertEqual(validate_split(path, "outer_test")["sizes"],
                         {"inner_train": 1, "inner_val": 1, "outer_test": 1})
        path = self.csv(contents="inner_train,inner_val,outer_test\nP1,P2,P1\n")
        with self.assertRaisesRegex(PlanError, "overlap"):
            validate_split(path, "outer_test")

    def test_legacy_test_partition_if_present_is_also_checked(self):
        path = self.csv(contents="train,val,test\nP1,P2,P1\n")
        with self.assertRaisesRegex(PlanError, "overlap"):
            validate_split(path, "legacy_val")

    def test_absolute_split_set_and_default_naming(self):
        self.assertEqual(split_path(self.root, "fixed", "blca", 2),
                         self.root / "splits" / "fixed" / "blca" / "fold_2.csv")
        self.assertEqual(split_path(self.root / "unused", str(self.root), "blca", 2),
                         self.root / "blca" / "fold_2.csv")

    def test_all_arms_share_split_and_unique_output(self):
        tasks = self.tasks(arms=list(ALL_ARMS), seeds=[3, 5], gpus=[2, 7])
        self.assertEqual(len(tasks), 34)
        self.assertEqual(len({task["split_sha256"] for task in tasks}), 1)
        self.assertEqual(len({task["output_dir"] for task in tasks}), 34)
        self.assertEqual([task["gpu_visible"] for task in tasks[:2]], ["2", "7"])
        self.assertFalse((self.root / "batch").exists())

    def test_dry_plan_reports_missing_splits_without_creating_files(self):
        tasks = self.tasks(studies=["missing"], require_splits=False)
        self.assertIsNone(tasks[0]["split_sha256"])
        self.assertIn("Cannot read", tasks[0]["split_error"])
        self.assertFalse((self.root / "batch").exists())
        with self.assertRaises(PlanError):
            self.tasks(studies=["missing"], require_splits=True)

    def test_split_mutation_prevents_launch(self):
        task = self.tasks()[0]
        Path(task["split_path"]).write_text("train,val\nP9,P8\n", encoding="utf-8")
        with self.assertRaisesRegex(PlanError, "changed"):
            check_split_unchanged(task)

    def test_task_config_changes_only_targeted_values(self):
        base = runner.read_config(HERE / "configs" / "base.yaml")
        task = self.tasks()[0]
        config = runner.task_config(base, task, self.arguments())
        self.assertEqual(config["train"]["seed"], 3)
        self.assertEqual(config["split"], {"k_start": 0, "k_end": 1})
        self.assertEqual(config["model"], base["model"])
        self.assertEqual(base["split"]["k_end"], 5)

    def test_existing_batch_cannot_be_executed_or_overwritten(self):
        tasks = self.tasks()
        batch = self.root / "batch"
        batch.mkdir()
        marker = batch / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        plan = {"batch_dir": str(batch), "tasks": tasks}
        with patch.object(runner.subprocess, "run") as run:
            with self.assertRaises(FileExistsError):
                runner.execute_plan(plan, runner.read_config(HERE / "configs" / "base.yaml"), self.arguments())
            run.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_nonzero_worker_stops_before_next_task(self):
        tasks = self.tasks()
        plan = {"batch_dir": str(self.root / "batch"), "tasks": tasks}
        with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = runner.execute_plan(plan, runner.read_config(HERE / "configs" / "base.yaml"), self.arguments())
        self.assertEqual(code, 7)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertFalse(Path(tasks[1]["output_dir"]).exists())
        self.assertFalse((self.root / "batch" / "completed.json").exists())

    def test_zero_exit_without_completion_is_failure(self):
        tasks = self.tasks()
        plan = {"batch_dir": str(self.root / "batch"), "tasks": tasks}
        with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = runner.execute_plan(plan, runner.read_config(HERE / "configs" / "base.yaml"), self.arguments())
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 1)

    def test_completion_requires_explicit_complete_status_and_training_flag(self):
        output = self.root / "completion_checks"
        output.mkdir()
        marker = output / "completion.json"
        invalid_values = [
            {}, {"status": "complete"}, {"training_complete": True},
            {"status": "failed", "training_complete": True},
            {"status": "complete", "training_complete": False},
            {"status": "complete", "training_complete": 1},
            {"status": "complete", "training_complete": "true"}, [],
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                marker.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(PlanError):
                    runner.check_completion(output)
        marker.write_text("{invalid", encoding="utf-8")
        with self.assertRaisesRegex(PlanError, "Invalid completion"):
            runner.check_completion(output)
        marker.write_text(json.dumps({"status": "complete", "training_complete": True}), encoding="utf-8")
        runner.check_completion(output)

    def test_incomplete_marker_stops_before_next_task(self):
        tasks = self.tasks()
        plan = {"batch_dir": str(self.root / "batch"), "tasks": tasks}

        def mocked_worker(command, **_kwargs):
            output = Path(command[command.index("--output-dir") + 1])
            (output / "completion.json").write_text(
                json.dumps({"status": "complete", "training_complete": False}), encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)

        with patch.object(runner.subprocess, "run", side_effect=mocked_worker) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = runner.execute_plan(plan, runner.read_config(HERE / "configs" / "base.yaml"), self.arguments())
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(Path(tasks[1]["output_dir"]).exists())
        self.assertFalse((self.root / "batch" / "completed.json").exists())

    def test_successful_mocked_worker_creates_fresh_complete_batch(self):
        tasks = self.tasks()
        plan = {"batch_dir": str(self.root / "batch"), "tasks": tasks}

        def mocked_worker(command, **_kwargs):
            output = Path(command[command.index("--output-dir") + 1])
            # Before the worker writes any metadata, only its input YAML exists.
            self.assertEqual([path.name for path in output.iterdir()], ["config.yaml"])
            (output / "completion.json").write_text(
                json.dumps({"status": "complete", "training_complete": True}), encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)

        with patch.object(runner.subprocess, "run", side_effect=mocked_worker) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                code = runner.execute_plan(plan, runner.read_config(HERE / "configs" / "base.yaml"), self.arguments())
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        completed = json.loads((self.root / "batch" / "completed.json").read_text(encoding="utf-8"))
        self.assertEqual(completed["tasks_completed"], 2)

    def test_plan_output_is_exclusive(self):
        path = self.root / "plan.json"
        runner.write_json_new(path, {"first": True})
        with self.assertRaises(FileExistsError):
            runner.write_json_new(path, {"second": True})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})

    def test_saved_plan_records_declared_grid_dimensions(self):
        destination = self.root / "plan.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = runner.main([
                "--arm", "full,no_self", "--cancer", "blca", "--fold", "0,1", "--seed", "3,5",
                "--data-path", str(self.root / "not_present"), "--results-root", str(self.root / "outputs"),
                "--batch-name", "planned", "--plan-out", str(destination),
            ])
        self.assertEqual(code, 0)
        plan = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(plan["studies"], ["blca"])
        self.assertEqual(plan["folds"], [0, 1])
        self.assertEqual(plan["seeds"], [3, 5])
        self.assertEqual(plan["selected_arms"], ["full", "no_self"])
        self.assertEqual(len(plan["tasks"]), 8)
        self.assertFalse((self.root / "outputs" / "planned").exists())


if __name__ == "__main__":
    unittest.main()
