"""Execution-boundary regression checks; no training or GPU work is started."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import os
import threading
import time

import pytest

from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.training import scheduler

ROOT = Path(__file__).resolve().parents[1]


def matrix(protocol="legacy_val", results_root=None):
    return scheduler.build_task_plan(
        protocols=[protocol], arms=["direct", "independent"],
        cancers=["kirc", "blca"], folds=range(5), seeds=[3],
        project_root=str(ROOT), data_path="/data1/dataset_csv",
        outer_split_root="5fold_uni2h_outer_test_seed3", results_root=results_root,
    )


def child_config(task):
    argv = scheduler._build_launch_command(task, str(ROOT))
    overrides = [argv[i + 1] for i, value in enumerate(argv) if value == "--set"]
    return config_to_argv(apply_overrides(load_config(task.config_yaml_path), overrides))


def value(argv, flag):
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize("protocol", ["legacy_val", "outer_test"])
def test_actual_child_arguments_isolate_all_twenty_tasks(protocol, tmp_path):
    tasks = matrix(protocol, str(tmp_path / "rerun"))
    output_dirs = set()
    folds_per_group = {}
    for task in tasks:
        argv = child_config(task)
        start, end = int(value(argv, "--k_start")), int(value(argv, "--k_end"))
        assert list(range(start, end)) == [task.fold]
        assert value(argv, "--results_dir") == task.results_dir
        assert value(argv, "--study") == task.cancer
        assert value(argv, "--survot_method") == scheduler.ARMS[task.arm]["base_method"]
        assert value(argv, "--evaluation_protocol") == protocol
        assert int(value(argv, "--seed")) == task.seed
        output_dirs.add(value(argv, "--results_dir"))
        folds_per_group.setdefault((task.arm, task.cancer), []).extend(range(start, end))
    assert len(output_dirs) == len(tasks) == 20
    assert all(folds == list(range(5)) for folds in folds_per_group.values())


def test_rerun_root_preserves_default_evidence_tree():
    original = matrix()
    rerun = matrix(results_root="results/v313_paper_v2")
    assert not ({t.results_dir for t in original} & {t.results_dir for t in rerun})
    assert all(os.path.isabs(t.results_dir) for t in rerun)


def test_dry_run_never_starts_a_child(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not start training")
    monkeypatch.setattr(scheduler, "_launch_one_task", forbidden)
    summary = scheduler.launch_tasks(matrix(), str(ROOT), gpus=["0", "1"], jobs_per_gpu=1, execute=False)
    assert summary == {"total": 20, "executed": 0, "failed": 0, "results": []}
    assert capsys.readouterr().out.count("[dry-run]") == 20


def test_launch_uses_task_fold_directory_gpu_and_project_cwd(monkeypatch):
    task = matrix()[3]
    task.gpu_id = "1/0"
    captured = {}
    def fake_subprocess(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(scheduler.subprocess, "run", fake_subprocess)
    outcome = scheduler._launch_one_task(task, str(ROOT))
    assert outcome["returncode"] == 0
    assert captured["cwd"] == str(ROOT)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["env"]["PYTHONPATH"] == str(ROOT)
    assert f"k_start={task.fold}" in captured["argv"]
    assert f"k_end={task.fold + 1}" in captured["argv"]
    assert f"results_dir={task.results_dir}" in captured["argv"]


@pytest.mark.parametrize("jobs", [1, 2])
def test_variable_duration_tasks_respect_each_gpu_limit(monkeypatch, jobs):
    active = Counter()
    peak = Counter()
    lock = threading.Lock()
    def fake_launch(task, project_root):
        gpu = task.gpu_id.split("/")[0]
        with lock:
            active[gpu] += 1
            peak[gpu] = max(peak[gpu], active[gpu])
        time.sleep(.003 if gpu == "0" else .025)
        with lock:
            active[gpu] -= 1
        return {"task_id": task.task_id(), "returncode": 0}
    monkeypatch.setattr(scheduler, "_launch_one_task", fake_launch)
    summary = scheduler.launch_tasks(matrix(), str(ROOT), gpus=["0", "1"], jobs_per_gpu=jobs, execute=True)
    assert summary["executed"] == 20
    assert set(peak) == {"0", "1"}
    assert all(count <= jobs for count in peak.values())


def test_written_plan_contains_gpu_and_executable_fold_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler, "__file__", str(tmp_path / "survot_rank/training/scheduler.py"))
    args = SimpleNamespace(arm=["direct", "independent"], cancer=["kirc", "blca"],
                           fold=None, seed=None, gpu=["0", "1"], protocol="legacy_val",
                           jobs_per_gpu=1, execute=False, results_root="results/v313_paper_v2")
    scheduler.cmd_schedule(args)
    plan_file = next((tmp_path / "results/v313_paper_v1/schedule").glob("*.json"))
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    assert len(plan) == 20
    assert Counter(t["gpu_id"] for t in plan) == {"0/0": 10, "1/0": 10}
    for task in plan:
        assert f"k_start={task['fold']}" in task["extra_set"]
        assert f"k_end={task['fold']+1}" in task["extra_set"]
        assert f"results_dir={task['results_dir']}" in task["extra_set"]
