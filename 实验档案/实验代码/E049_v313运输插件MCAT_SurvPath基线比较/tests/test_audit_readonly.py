"""Synthetic receipts and feature metadata only; no patient data or model run."""
import csv
import json
import pickle

import h5py
import torch

from audit_readonly import _padding, _sha256, _task, audit


def _csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_padding_audit_counts_only_short_patient_bags(tmp_path):
    data = tmp_path / "data"
    features = tmp_path / "features" / "blca" / "uni2-h" / "pt_files"
    features.mkdir(parents=True)
    _csv(data / "clinical" / "all" / "blca.csv", ["case id", "wsi"],
         [{"case id": name, "wsi": name + ".svs"} for name in ("p1", "p2", "p3")])
    split = data / "splits" / "fold_0.csv"
    _csv(split, ["inner_train", "inner_val", "outer_test"],
         [{"inner_train": "p1", "inner_val": "p2", "outer_test": "p3"}])
    for name, count in (("p1", 2), ("p2", 4), ("p3", 5)):
        with h5py.File(features / (name + ".h5"), "w") as handle:
            handle.create_dataset("features", shape=(count, 3), dtype="f4")
    task = {"fold": 0, "study": "blca", "split_file": {"path": str(split),
                                                "sha256": _sha256(split)},
            "recipe": {"data_path": str(data), "data_root_dir": str(tmp_path / "features"),
                       "wsi_encoder": "uni2-h", "num_patches": 4}}
    result = _padding([task])
    assert result["unique_patients"] == 3
    assert result["unique_padded_patients"] == 1
    assert result["folds"][0]["inner_train"]["padded_rows"] == 2
    assert result["folds"][0]["outer_test"]["padded_patients"] == 0


def test_receipt_audit_recomputes_score_and_reads_selected_checkpoint(tmp_path):
    split = tmp_path / "fold_0.csv"
    _csv(split, ["inner_train", "inner_val", "outer_test"], [
        {"inner_train": "p3", "inner_val": "p4", "outer_test": "p1"},
        {"inner_train": "", "inner_val": "", "outer_test": "p2"},
    ])
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    task_id = "outer_test/blca/mcat/learned_three_decoders/fold0/seed3"
    _json(task_dir / "task.json", {
        "task_id": task_id, "baseline": "mcat", "arm": "learned_three_decoders",
        "study": "blca", "protocol": "outer_test", "fold": 0, "seed": 3,
        "split_file": {"path": str(split), "sha256": _sha256(split)},
    })
    predictions = {"p1": {"risk": 1.0, "time": 1.0, "censor": 0.0},
                   "p2": {"risk": 0.0, "time": 2.0, "censor": 1.0}}
    with (task_dir / "predictions.pkl").open("wb") as stream:
        pickle.dump(predictions, stream)
    _json(task_dir / "labels.json", {"partition": "outer_test", "patients": {
        "p1": {"time": 1.0, "censor": 0.0}, "p2": {"time": 2.0, "censor": 1.0},
    }})
    _csv(task_dir / "epoch_curve_fold0.csv",
         ["epoch", "val_cindex", "val_cindex_ipcw", "train_self", "train_cross",
          "train_wsi", "train_weighted_total", "train_ramp"], [
              {"epoch": 0, "val_cindex": 0.4, "val_cindex_ipcw": 0.3},
              {"epoch": 1, "val_cindex": 0.5, "val_cindex_ipcw": 0.4,
               "train_self": 0.1, "train_cross": 0.2, "train_wsi": 0.3,
               "train_weighted_total": 0.03, "train_ramp": 1.0},
          ])
    torch.save({"adapter.plugin.wsi_gain": torch.tensor(0.2),
                "adapter.plugin.omics_gain": torch.tensor(-0.1)}, task_dir / "model.pth")
    _json(task_dir / "model_audit.json", {"baseline": "mcat",
         "arm": "learned_three_decoders", "enabled": True, "reconstruction": True})
    def artifact(name):
        return {"path": name, "sha256": _sha256(task_dir / name)}
    _json(task_dir / "completion.json", {
        "task_id": task_id, "status": "complete", "training_complete": True,
        "baseline": "mcat", "arm": "learned_three_decoders",
        "metric": "cindex", "protocol": "outer_test",
        "evaluation_partition": "outer_test", "score": 1.0,
        "prediction_patient_count": 2, "completed_epochs": 2,
        "prediction_artifact": artifact("predictions.pkl"),
        "label_audit_artifact": artifact("labels.json"),
        "checkpoint_artifact": artifact("model.pth"),
        "artifacts": [artifact("epoch_curve_fold0.csv"), artifact("model_audit.json")],
    })
    row, _ = _task(task_dir)
    assert row["outer_test_cindex"] == 1.0
    assert row["best_val_epoch"] == 1
    assert row["aux_at_best"]["train_cross"] == 0.2
    assert row["gains"]["wsi_gain"]["effective_tanh"] > 0


def test_receipt_rejects_wrong_cancer_even_with_valid_prediction_artifacts(tmp_path):
    directory = tmp_path / "task"
    directory.mkdir()
    identity = "outer_test/brca/mcat/native/fold0/seed3"
    _json(directory / "task.json", {"task_id": identity, "baseline": "mcat",
         "arm": "native", "study": "brca", "protocol": "outer_test",
         "fold": 0, "seed": 3})
    _json(directory / "completion.json", {"task_id": identity, "baseline": "mcat",
         "arm": "native", "status": "complete"})
    import pytest
    with pytest.raises(ValueError, match="BLCA"):
        _task(directory)


def _synthetic_batch(tmp_path, monkeypatch, baselines=("mcat", "survpath"), seeds=(3,)):
    import audit_readonly

    tasks = {}
    for baseline in baselines:
        for seed in seeds:
            for fold in range(5):
                for arm in ("native", "learned_three_decoders"):
                    directory = tmp_path / baseline / arm / f"fold{fold}" / f"seed{seed}"
                    directory.mkdir(parents=True)
                    (directory / "completion.json").write_text("{}", encoding="utf-8")
                    tasks[directory] = (
                        {"task_id": str(directory.relative_to(tmp_path)),
                         "baseline": baseline, "arm": arm, "fold": fold, "seed": seed,
                         "outer_test_cindex": 0.61 if arm != "native" else 0.60},
                        {"recipe": {"e049_arm": arm},
                         "split_file": {"sha256": f"split-{fold}"},
                         "config_file": {"sha256": "config"},
                         "source_files": [{"path": "/repo/worker.py", "sha256": "source"}],
                         "training_label_audit": {"fold": fold}},
                    )
    monkeypatch.setattr(audit_readonly, "_task", lambda directory: tasks[directory])
    monkeypatch.setattr(audit_readonly, "_padding", lambda _: {})
    return tasks


def test_task_grid_requires_both_baselines_even_when_20_tasks_exist(tmp_path, monkeypatch):
    _synthetic_batch(tmp_path, monkeypatch, baselines=("mcat",), seeds=(3, 4))
    result = audit(tmp_path)
    assert len(result["tasks"]) == 20
    assert any(error["error"] == "e049_task_grid_mismatch" for error in result["errors"])


def test_pair_rejects_different_source_snapshot(tmp_path, monkeypatch):
    tasks = _synthetic_batch(tmp_path, monkeypatch)
    for path, (_, task) in tasks.items():
        if path.parts[-4:-2] == ("mcat", "learned_three_decoders") and path.parts[-2] == "fold0":
            task["source_files"][0]["sha256"] = "changed-source"
    result = audit(tmp_path)
    assert len(result["paired_deltas"]) == 9
    assert any("Paired recipe/split mismatch" in error["error"] for error in result["errors"])


def test_task_errors_do_not_echo_patient_ids(tmp_path, monkeypatch):
    import audit_readonly

    (tmp_path / "completion.json").write_text("{}", encoding="utf-8")
    def fail(_):
        raise ValueError("patient-secret")
    monkeypatch.setattr(audit_readonly, "_task", fail)
    result = audit(tmp_path)
    assert "patient-secret" not in json.dumps(result)
