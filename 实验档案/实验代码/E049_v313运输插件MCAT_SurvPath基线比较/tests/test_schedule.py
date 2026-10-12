"""Unit tests for E049 schedule_core."""
from __future__ import annotations

import pytest
from schedule_core import (
    ARM_NAMES, PlanError,
    comma_values, integer_values, selected_arms, patient_id,
    validate_split, split_path, build_tasks,
)
import csv
import io
import tempfile
from pathlib import Path


class TestCommaValues:
    def test_defaults(self):
        assert comma_values(None, ["a", "b"]) == ["a", "b"]

    def test_explicit_list(self):
        assert comma_values(["a", "b"], ["x"]) == ["a", "b"]

    def test_comma_separated(self):
        assert comma_values(["a,b,c"], []) == ["a", "b", "c"]

    def test_dedup(self):
        assert comma_values(["a", "a,b"], []) == ["a", "b"]

    def test_empty_rejected(self):
        with pytest.raises(PlanError):
            comma_values([""], ["x"])


class TestIntegerValues:
    def test_defaults(self):
        assert integer_values(None, [1, 2], name="x") == [1, 2]

    def test_explicit(self):
        assert integer_values(["3", "5"], [], name="x") == [3, 5]

    def test_comma_separated(self):
        assert integer_values(["1,2,3"], [], name="x") == [1, 2, 3]

    def test_negative_rejected(self):
        with pytest.raises(PlanError):
            integer_values(["-1"], [], name="x")


class TestSelectedArms:
    def test_all(self):
        assert selected_arms("all", None) == list(ARM_NAMES)

    def test_coupling(self):
        arms = selected_arms("coupling", None)
        assert "native" in arms
        assert "independent" in arms
        assert "learned" in arms
        assert "learned_three_decoders" not in arms

    def test_explicit_unknown_rejected(self):
        with pytest.raises(PlanError):
            selected_arms("all", ["unknown_arm"])

    def test_explicit_override(self):
        arms = selected_arms("all", ["native", "learned"])
        assert arms == ["native", "learned"]


class TestPatientId:
    def test_tcga_three_fields(self):
        assert patient_id("TCGA-AB-1234-01A") == "TCGA-AB-1234"

    def test_non_tcga(self):
        assert patient_id("PATIENT_001") == "PATIENT_001"


class TestValidateSplit:
    def test_outer_test_valid(self):
        content = "inner_train,inner_val,outer_test\nP1,P2,P3\nP4,P5,P6\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        try:
            result = validate_split(path, "outer_test")
            assert result["sha256"] is not None
            assert result["sizes"]["inner_train"] == 2
            assert result["sizes"]["inner_val"] == 2
            assert result["sizes"]["outer_test"] == 2
        finally:
            path.unlink()

    def test_legacy_val(self):
        content = "train,val\nP1,P2\nP3,P4\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        try:
            result = validate_split(path, "legacy_val")
            assert result["sizes"]["train"] == 2
            assert result["sizes"]["val"] == 2
        finally:
            path.unlink()

    def test_overlap_detected(self):
        content = "inner_train,inner_val,outer_test\nP1,P1,P3\nP4,P5,P6\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        try:
            with pytest.raises(PlanError, match="overlap"):
                validate_split(path, "outer_test")
        finally:
            path.unlink()


class TestBuildTasks:
    def test_task_count(self):
        arms = ["native", "learned"]
        baselines = ["mcat"]
        folds = [0, 1]
        seeds = [3]
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = build_tasks(
                arms=arms, baselines=baselines, folds=folds, seeds=seeds,
                gpus=[0],
                batch_dir=Path(tmpdir) / "batch",
                data_path=Path(tmpdir) / "data",
                split_set="outer_test",
                protocol="outer_test",
                python_executable="python",
                worker_path=Path("worker.py"),
                require_splits=False,
            )
        # 2 arms × 1 baseline × 2 folds × 1 seed = 4 tasks
        assert len(tasks) == 4
        for task in tasks:
            assert task["arm"] in arms
            assert task["baseline"] in baselines
            assert task["fold"] in folds
            assert task["seed"] in seeds
            assert "native" not in task["task_id"] or task["arm"] == "native"

    def test_gpu_round_robin(self):
        arms = ["native", "learned"]
        baselines = ["mcat"]
        folds = [0]
        seeds = [3]
        gpus = [0, 1, 2]
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = build_tasks(
                arms=arms, baselines=baselines, folds=folds, seeds=seeds,
                gpus=gpus,
                batch_dir=Path(tmpdir) / "batch",
                data_path=Path(tmpdir) / "data",
                split_set="outer_test",
                protocol="outer_test",
                python_executable="python",
                worker_path=Path("worker.py"),
                require_splits=False,
            )
        assert len(tasks) == 2
        # Should round-robin across gpus
        assert tasks[0]["gpu_visible"] == "0"
        assert tasks[1]["gpu_visible"] == "1"
