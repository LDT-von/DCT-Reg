"""Read-only BLCA plugin audit for an isolated E049 server batch.

Prints one compact JSON object. It never trains a model or writes an artifact.
The normal summary contains no patient IDs or feature values. Run with the
repository's DCT Python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import evidence


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _artifact(task_dir: Path, record: dict) -> Path:
    path = (task_dir / record["path"]).resolve()
    path.relative_to(task_dir.resolve())
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Artifact SHA-256 mismatch: {record['path']}")
    return path


def _curve(task_dir: Path, complete: dict) -> dict:
    records = [item for item in complete["artifacts"]
               if Path(item["path"]).name.startswith("epoch_curve_fold")
               and Path(item["path"]).suffix == ".csv"]
    if len(records) != 1:
        raise ValueError("Expected exactly one recorded epoch curve")
    with _artifact(task_dir, records[0]).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != complete["completed_epochs"]:
        raise ValueError("Epoch curve length disagrees with completion")
    finite = [row for row in rows if math.isfinite(float(row["val_cindex"]))]
    if not finite:
        raise ValueError("No finite inner-validation C-index")
    best = max(finite, key=lambda row: float(row["val_cindex"]))
    names = ("train_self", "train_cross", "train_wsi", "train_weighted_total", "train_ramp")
    return {
        "best_val_epoch": int(best["epoch"]),
        "best_val_cindex": float(best["val_cindex"]),
        "best_val_cindex_ipcw": _finite_or_none(best.get("val_cindex_ipcw")),
        "aux_at_best": {name: _finite_or_none(best.get(name)) for name in names
                        if best.get(name) is not None},
    }


def _gains(checkpoint: Path) -> dict:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = {}
    for name in ("wsi_gain", "omics_gain"):
        key = "adapter.plugin." + name
        if key not in state or state[key].numel() != 1:
            raise ValueError(f"Checkpoint lacks scalar {key}")
        raw = float(state[key].item())
        if not math.isfinite(raw):
            raise ValueError(f"Checkpoint has non-finite {key}")
        result[name] = {"raw": raw, "effective_tanh": math.tanh(raw)}
    return result


def _task(task_dir: Path) -> tuple[dict, dict]:
    task, complete = _json(task_dir / "task.json"), _json(task_dir / "completion.json")
    identity = (f"outer_test/blca/{task['baseline']}/{task['arm']}/"
                f"fold{int(task['fold'])}/seed{int(task['seed'])}")
    if (task.get("study") != "blca" or task.get("protocol") != "outer_test"
            or task["task_id"] != identity
            or complete.get("baseline") != task["baseline"]
            or complete.get("arm") != task["arm"]):
        raise ValueError("Task is not a consistent E049 BLCA outer-test task")
    if (task["task_id"] != complete["task_id"] or complete["status"] != "complete"
            or complete.get("training_complete") is not True or complete.get("metric") != "cindex"):
        raise ValueError("Task/completion identity or status mismatch")
    if complete["protocol"] != "outer_test" or complete["evaluation_partition"] != "outer_test":
        raise ValueError("Expected outer_test protocol and evaluation")
    split = task["split_file"]
    if _sha256(Path(split["path"])) != split["sha256"]:
        raise ValueError("Split SHA-256 mismatch")
    predictions = evidence.read_predictions(_artifact(task_dir, complete["prediction_artifact"]))
    label_audit = _json(_artifact(task_dir, complete["label_audit_artifact"]))
    evidence.verify_prediction_labels(predictions, label_audit, split["path"], "outer_test")
    score = evidence.compute_cindex(predictions)
    recorded = float(complete["score"])
    if (not math.isfinite(recorded) or abs(score - recorded) > 1e-8
            or len(predictions) != complete["prediction_patient_count"]):
        raise ValueError("Recomputed C-index or patient count differs from completion")
    curve = _curve(task_dir, complete)
    checkpoint = _artifact(task_dir, complete["checkpoint_artifact"])
    model_records = [item for item in complete["artifacts"]
                     if Path(item["path"]).name == "model_audit.json"]
    if len(model_records) != 1:
        raise ValueError("Expected exactly one recorded model audit")
    model = _json(_artifact(task_dir, model_records[0]))
    expected_enabled = task["arm"] == "learned_three_decoders"
    if (model.get("baseline") != task["baseline"] or model.get("arm") != task["arm"]
            or model.get("enabled") != expected_enabled
            or model.get("reconstruction") != expected_enabled):
        raise ValueError("Model audit disagrees with E049 arm identity")
    row = {
        "task_id": task["task_id"], "baseline": task["baseline"],
        "arm": task["arm"], "fold": int(task["fold"]), "seed": int(task["seed"]),
        "outer_test_cindex": score, "patients": len(predictions),
        **curve, "gains": _gains(checkpoint),
    }
    return row, task


def _csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _feature_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"WSI feature root is missing: {root}")
    result = {}
    for suffix in ("*.pt", "*.h5", "*.hdf5"):
        for path in root.rglob(suffix):
            result.setdefault(path.stem, path)
    return result


def _feature_length(path: Path) -> int:
    if path.suffix in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as handle:
            shape = handle["features"].shape
    elif path.suffix == ".pt":
        import torch
        try:
            value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError):
            value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value["features"]
        shape = value.shape
    else:
        raise ValueError("Unsupported WSI feature suffix")
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 2 or shape[0] < 1:
        raise ValueError("Invalid WSI feature shape")
    return int(shape[0])


def _padding(tasks: list[dict]) -> dict:
    by_fold = {}
    for task in tasks:
        fold = int(task["fold"])
        previous = by_fold.setdefault(fold, task)
        keys = ("data_path", "data_root_dir", "wsi_encoder", "num_patches")
        if (previous["split_file"]["sha256"] != task["split_file"]["sha256"]
                or any(previous["recipe"].get(key) != task["recipe"].get(key) for key in keys)):
            raise ValueError(f"WSI data/split configuration differs within fold{fold}")
    exemplar = next(iter(by_fold.values()))
    recipe = exemplar["recipe"]
    keys = ("data_path", "data_root_dir", "wsi_encoder", "num_patches")
    if any(any(task["recipe"].get(key) != recipe.get(key) for key in keys)
           for task in by_fold.values()):
        raise ValueError("WSI data configuration differs across folds")
    data_path = Path(recipe["data_path"])
    study = exemplar["study"]
    limit = int(recipe["num_patches"])
    root = Path(recipe["data_root_dir"]) / study / recipe.get("wsi_encoder", "uni") / "pt_files"
    if root.is_dir() and not (root / "TCGA-1.pt").exists():
        alternative = root / study / "uni" / "pt_files"
        if alternative.is_dir():
            root = alternative
    index = _feature_index(root)
    clinical_path = data_path / "clinical" / "all" / f"{study}.csv"
    clinical = {}
    for row in _csv_rows(clinical_path):
        clinical.setdefault(row["case id"], row.get("wsi", ""))
    cached_lengths = {}
    patient_lengths = {}
    def patient_length(patient: str) -> int:
        if patient in patient_lengths:
            return patient_lengths[patient]
        slides = clinical.get(patient)
        if not slides or slides.lower() == "nan":
            raise ValueError("A split patient has no WSI slide reference")
        total = 0
        for slide in slides.split(", "):
            stem = slide.removesuffix(".svs")
            path = next((root / f"{stem}{suffix}" for suffix in (".pt", ".h5", ".hdf5")
                         if (root / f"{stem}{suffix}").is_file()), None)
            path = path or index.get(stem)
            if path is not None:
                if path not in cached_lengths:
                    cached_lengths[path] = _feature_length(path)
                total += cached_lengths[path]
        if total < 1:
            raise ValueError("A split patient has no available WSI feature")
        patient_lengths[patient] = total
        return total

    fold_rows = []
    unique_patients = set()
    for fold, task in sorted(by_fold.items()):
        split_rows = _csv_rows(Path(task["split_file"]["path"]))
        summary = {"fold": fold}
        for partition in ("inner_train", "inner_val", "outer_test"):
            patients = {row[partition] for row in split_rows
                        if row.get(partition) and row[partition].lower() != "nan"}
            unique_patients.update(patients)
            lengths = [patient_length(patient) for patient in patients]
            summary[partition] = {
                "patients": len(patients),
                "padded_patients": sum(length < limit for length in lengths),
                "padded_rows": sum(max(0, limit - length) for length in lengths),
            }
        fold_rows.append(summary)
    return {
        "patch_limit": limit,
        "unique_patients": len(unique_patients),
        "unique_padded_patients": sum(patient_length(patient) < limit for patient in unique_patients),
        "folds": fold_rows,
    }


def _source_manifest(task: dict) -> dict[str, str]:
    """Compare repository-relative source names without printing server paths."""
    manifest = {}
    for entry in task["source_files"]:
        parts = Path(entry["path"]).parts
        for marker in ("survot_rank", "实验档案"):
            if marker in parts:
                name = "/".join(parts[parts.index(marker):])
                break
        else:
            name = Path(entry["path"]).name
        if name in manifest:
            raise ValueError("Duplicate source name in task receipt")
        manifest[name] = entry["sha256"]
    return manifest


def _pair_differences(native: dict, plugin: dict) -> dict:
    """Name mismatched receipt fields; never include patient labels or values."""
    differences = {}
    for field in ("split_file", "config_file"):
        if native[field]["sha256"] != plugin[field]["sha256"]:
            differences[field + "_sha256"] = True
    if native.get("training_label_audit") != plugin.get("training_label_audit"):
        differences["training_label_audit"] = True
    native_recipe, plugin_recipe = dict(native["recipe"]), dict(plugin["recipe"])
    native_recipe.pop("e049_arm", None)
    plugin_recipe.pop("e049_arm", None)
    recipe_keys = sorted(key for key in native_recipe.keys() | plugin_recipe.keys()
                         if native_recipe.get(key) != plugin_recipe.get(key))
    if recipe_keys:
        differences["recipe_keys"] = recipe_keys
    native_sources, plugin_sources = _source_manifest(native), _source_manifest(plugin)
    source_names = sorted(name for name in native_sources.keys() | plugin_sources.keys()
                          if native_sources.get(name) != plugin_sources.get(name))
    if source_names:
        differences["source_files"] = {"count": len(source_names),
                                       "examples": source_names[:20]}
    return differences


def audit(batch_dir: Path) -> dict:
    records, tasks, errors = [], [], []
    for path in sorted(batch_dir.rglob("completion.json")):
        try:
            row, task = _task(path.parent)
            records.append(row)
            tasks.append(task)
        except Exception as error:
            errors.append({"task": str(path.parent.relative_to(batch_dir)),
                           "error": "task_artifact_or_label_audit_failed",
                           "error_type": type(error).__name__})
    if not records:
        errors.append({"error": "No complete, verifiable E049 task found"})
    expected = {(baseline, arm, fold, 3)
                for baseline in ("mcat", "survpath")
                for arm in ("native", "learned_three_decoders")
                for fold in range(5)}
    actual = [(row["baseline"], row["arm"], row["fold"], row["seed"])
              for row in records]
    if set(actual) != expected or len(actual) != len(expected):
        errors.append({"error": "e049_task_grid_mismatch",
                       "missing": len(expected - set(actual)),
                       "unexpected": len(set(actual) - expected),
                       "duplicates": len(actual) - len(set(actual))})
    pairs, unverified_pairs = [], []
    grouped = {}
    for row, task in zip(records, tasks):
        arms = grouped.setdefault((row["baseline"], row["fold"], row["seed"]), {})
        if row["arm"] in arms:
            errors.append({"error": f"Duplicate arm: {row['baseline']} fold{row['fold']} seed{row['seed']} {row['arm']}"})
        arms[row["arm"]] = (row, task)
    for (baseline, fold, seed), arms in sorted(grouped.items()):
        if {"native", "learned_three_decoders"} <= arms.keys():
            native, native_task = arms["native"]
            plugin, plugin_task = arms["learned_three_decoders"]
            delta = plugin["outer_test_cindex"] - native["outer_test_cindex"]
            differences = _pair_differences(native_task, plugin_task)
            if differences:
                errors.append({"error": f"Paired recipe/split mismatch: {baseline} fold{fold} seed{seed}",
                               "differences": differences})
                unverified_pairs.append({"baseline": baseline, "fold": fold, "seed": seed,
                                         "raw_delta": delta, "differences": differences})
                continue
            pairs.append({"baseline": baseline, "fold": fold, "seed": seed,
                          "delta": delta})
    if len(pairs) != 10:
        errors.append({"error": f"Expected 10 native/full pairs; found {len(pairs)}"})
    mean_deltas = {}
    for baseline in sorted({pair["baseline"] for pair in pairs}):
        deltas = [pair["delta"] for pair in pairs if pair["baseline"] == baseline]
        mean_deltas[baseline] = {"mean": sum(deltas) / len(deltas),
                                 "positive_folds": sum(delta > 0 for delta in deltas),
                                 "folds": len(deltas)}
    try:
        padding = _padding(tasks) if tasks else None
    except Exception as error:
        padding = None
        errors.append({"error": "padding_metadata_audit_failed",
                       "error_type": type(error).__name__})
    return {"schema": "e049_plugin_readonly_audit_v1",
            "metric": "Harrell C-index (censor=0 denotes event)", "tasks": records,
            "paired_deltas": pairs, "unverified_pair_deltas": unverified_pairs,
            "mean_deltas": mean_deltas,
            "padding": padding, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.batch_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
