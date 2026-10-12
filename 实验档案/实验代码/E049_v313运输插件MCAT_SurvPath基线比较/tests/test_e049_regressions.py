"""Regression checks for the BLCA-only paired plan and summary keys."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_experiments import main
from summarize import fmt_comparison_table, read_batch
from worker import _mcat_omic_sizes_from_factory


def test_default_plan_is_blca_native_vs_full_plugin(tmp_path):
    batch = tmp_path / "batch"
    assert main(["--batch-dir", str(batch), "--skip-splits", "--dry-run"]) == 0
    tasks = json.loads((batch / "plan.json").read_text(encoding="utf-8"))["tasks"]
    assert len(tasks) == 20
    assert {task["study"] for task in tasks} == {"blca"}
    assert {task["arm"] for task in tasks} == {"native", "learned_three_decoders"}
    assert {task["baseline"] for task in tasks} == {"mcat", "survpath"}
    assert {task["fold"] for task in tasks} == set(range(5))
    assert all("/blca/fold_" in task["split_path"].replace("\\", "/") for task in tasks)


def test_summary_pairs_by_fold_even_when_rows_are_shuffled():
    rows = []
    for fold in range(5):
        rows.extend([
            dict(study="blca", protocol="outer_test", baseline="mcat", arm="native",
                 fold=fold, seed=3, score=0.60 + fold * 0.01),
            dict(study="blca", protocol="outer_test", baseline="mcat", arm="learned_three_decoders",
                 fold=fold, seed=3, score=0.65 + fold * 0.01),
        ])
    rows.reverse()
    table = fmt_comparison_table([("full", "native", "learned_three_decoders", "plugin")], rows, ["mcat"])
    assert "+0.0500" in table
    assert "blca/outer_test" in table
    incomplete = rows[:-1]
    assert "INCOMPLETE" in fmt_comparison_table(
        [("full", "native", "learned_three_decoders", "plugin")], incomplete, ["mcat"])


def test_mcat_rejects_fabricated_omic_group_sizes():
    assert _mcat_omic_sizes_from_factory(SimpleNamespace(omic_sizes=[3] * 6)) == [3] * 6
    with pytest.raises(ValueError, match="six"):
        _mcat_omic_sizes_from_factory(SimpleNamespace(omic_sizes=[3] * 7))


def test_summary_refuses_partial_planned_batch(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"tasks": [
        {"task_id": "outer_test/blca/mcat/native/fold0/seed3"},
    ]}), encoding="utf-8")
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed" / "completion.json").write_text(
        json.dumps({"task_id": "outer_test/blca/mcat/native/fold0/seed3", "status": "failed"}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        read_batch(tmp_path)
