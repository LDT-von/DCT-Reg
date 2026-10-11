"""Verify E048 evidence before reporting matched, five-fold C-index results.

This command never trains a model or performs inference. Only locally generated
prediction pickles declared in a complete, hashed worker receipt are accepted.
Incomplete/invalid batches produce an audit report and a non-zero exit status;
their incomplete groups have no published mean or standard deviation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
PROTOCOL_PARTITION = {"legacy_val": "val", "outer_test": "outer_test"}
SCORE_TOLERANCE = 1e-7


class EvidenceError(ValueError):
    """An artifact cannot support a verified performance claim."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"Cannot read JSON {path}: {exc}") from exc


def _number(value: Any, label: str) -> float:
    # Trainer predictions contain NumPy scalar/one-element array values.
    if hasattr(value, "size") and int(value.size) != 1:
        raise EvidenceError(f"{label} must be a scalar")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


class _PredictionUnpickler(pickle.Unpickler):
    """Accept NumPy prediction primitives, not arbitrary pickle code."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"),
            ("numpy._core.numeric", "_frombuffer"),
        }
        if (module, name) not in allowed:
            raise EvidenceError(f"Unsupported prediction-pickle global: {module}.{name}")
        return super().find_class(module, name)


def read_predictions(path: Path | str) -> dict[str, dict[str, float]]:
    """Read the train_runner patient -> risk/censor/time prediction contract."""
    try:
        with Path(path).open("rb") as stream:
            payload = _PredictionUnpickler(stream).load()
    except (OSError, pickle.PickleError, ImportError, ValueError, EOFError) as exc:
        raise EvidenceError(f"Cannot read predictions {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise EvidenceError("Prediction pickle must contain a non-empty patient dictionary")
    predictions: dict[str, dict[str, float]] = {}
    for patient, row in payload.items():
        if not isinstance(patient, str) or not patient.strip() or patient != patient.strip():
            raise EvidenceError("Prediction patient IDs must be non-empty, unmodified strings")
        if not isinstance(row, Mapping):
            raise EvidenceError(f"Invalid prediction row for {patient}")
        if not {"risk", "censor", "time"}.issubset(row):
            raise EvidenceError(f"Prediction row {patient} lacks risk/censor/time")
        values = {key: _number(row[key], f"{patient}.{key}") for key in ("risk", "time", "censor")}
        if values["time"] < 0 or values["censor"] not in (0.0, 1.0):
            raise EvidenceError(f"Invalid survival time/censorship for {patient}")
        predictions[patient] = values
    return predictions


def compute_cindex(predictions: Mapping[str, Mapping[str, Any]], tied_tol: float = 1e-8) -> float:
    """Harrell C-index matching sksurv concordance_index_censored.

    Censor=0 denotes an observed event. An event can be compared with later
    observations and with censored observations at the same time. Two events
    at the same time are not comparable. Higher risk predicts an earlier event.
    Risk ties within tied_tol contribute one half; no comparable pairs is an
    error, never an invented zero score.
    """
    if not math.isfinite(tied_tol) or tied_tol < 0:
        raise EvidenceError("tied_tol must be finite and non-negative")
    rows = []
    for patient, row in predictions.items():
        time = _number(row["time"], f"{patient}.time")
        censor = _number(row["censor"], f"{patient}.censor")
        risk = _number(row["risk"], f"{patient}.risk")
        if time < 0 or censor not in (0.0, 1.0):
            raise EvidenceError(f"Invalid survival labels for {patient}")
        rows.append((time, censor, risk))
    concordant = tied = comparable = 0
    for time, censor, risk in rows:
        if censor != 0.0:
            continue
        for other_time, other_censor, other_risk in rows:
            if other_time <= time and not (other_time == time and other_censor == 1.0):
                continue
            comparable += 1
            difference = risk - other_risk
            if abs(difference) <= tied_tol:
                tied += 1
            elif difference > 0:
                concordant += 1
    if comparable == 0:
        raise EvidenceError("No comparable patient pairs; C-index is undefined")
    return (concordant + 0.5 * tied) / comparable


def _split_members(split_path: Path | str) -> dict[str, set[str]]:
    try:
        with Path(split_path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise EvidenceError("Split CSV has no columns")
            values: dict[str, list[str]] = {name: [] for name in reader.fieldnames if name}
            for row in reader:
                for column in values:
                    value = row.get(column)
                    if value is not None and value.strip():
                        if value != value.strip():
                            raise EvidenceError("Split patient IDs contain surrounding whitespace")
                        values[column].append(value)
    except OSError as exc:
        raise EvidenceError(f"Cannot read split {split_path}: {exc}") from exc
    for column, ids in values.items():
        if len(ids) != len(set(ids)):
            raise EvidenceError(f"Duplicate patient IDs in split column {column}")
    return {column: set(ids) for column, ids in values.items()}


def verify_prediction_labels(
    predictions: Mapping[str, Mapping[str, Any]],
    label_audit: Mapping[str, Any],
    split_path: Path | str,
    partition: str,
) -> None:
    """Check the actual loader labels, exact split membership and disjointness."""
    if not isinstance(label_audit, Mapping):
        raise EvidenceError("Actual-loader label audit must be a JSON object")
    if label_audit.get("partition") != partition:
        raise EvidenceError("Actual-loader label audit partition differs from completion")
    patients = label_audit.get("patients")
    if not isinstance(patients, Mapping) or set(patients) != set(predictions):
        raise EvidenceError("Prediction patient IDs differ from actual-loader label audit")
    split = _split_members(split_path)
    if partition not in split or not split[partition]:
        raise EvidenceError(f"Split lacks non-empty {partition} partition")
    if set(predictions) != split[partition]:
        raise EvidenceError("Predictions do not exactly cover the declared split partition")
    required = ("train", "val") if partition == "val" else ("inner_train", "inner_val", "outer_test")
    if any(name not in split or not split[name] for name in required):
        raise EvidenceError(f"Split lacks required non-empty columns: {required}")
    for index, name in enumerate(required):
        for other_name in required[index + 1:]:
            if split[name] & split[other_name]:
                raise EvidenceError(f"Patient overlap between split columns {name} and {other_name}")
    for patient, prediction in predictions.items():
        row = patients[patient]
        if not isinstance(row, Mapping):
            raise EvidenceError(f"Invalid actual-loader label row for {patient}")
        for key in ("time", "censor"):
            if key not in row or _number(row[key], f"audit.{patient}.{key}") != prediction[key]:
                raise EvidenceError(f"Prediction {key} differs from actual loader for {patient}")


def _resolve(path: str, base: Path, confined: bool = False) -> Path:
    if not isinstance(path, str) or not path:
        raise EvidenceError("Artifact path must be a non-empty string")
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if confined and not candidate.is_relative_to(base.resolve()):
        raise EvidenceError(f"Artifact escapes task directory: {candidate}")
    return candidate


def _verify_file(reference: Any, base: Path, *, confined: bool = False) -> Path:
    if not isinstance(reference, Mapping):
        raise EvidenceError("File reference must contain path and sha256")
    expected = reference.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise EvidenceError("File reference has no valid SHA-256")
    path = _resolve(reference.get("path"), base, confined)
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise EvidenceError(f"File absent or SHA-256 changed: {path}")
    return path


def _identity(task: Mapping[str, Any]) -> tuple[str, int, str, int, str]:
    if not isinstance(task, Mapping):
        raise EvidenceError("Each planned/final task must be a JSON object")
    study, arm, protocol = (task.get(key) for key in ("study", "arm", "protocol"))
    fold, seed = task.get("fold"), task.get("seed")
    if not all(isinstance(value, str) and value for value in (study, arm, protocol)):
        raise EvidenceError("Task lacks study/arm/protocol")
    if protocol not in PROTOCOL_PARTITION:
        raise EvidenceError(f"Unsupported protocol: {protocol}")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in EXPECTED_FOLDS:
        raise EvidenceError(f"Task fold must be one of {EXPECTED_FOLDS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvidenceError("Task seed must be a non-negative integer")
    return study, seed, protocol, fold, arm


def verify_task(task_dir: Path, planned: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on a single task, returning only independently verified data."""
    task = _json(task_dir / "task.json")
    completion = _json(task_dir / "completion.json")
    if not isinstance(task, Mapping) or not isinstance(completion, Mapping):
        raise EvidenceError("Task and completion receipts must be JSON objects")
    identity = _identity(task)
    if _identity(planned) != identity or task.get("task_id") != planned.get("task_id"):
        raise EvidenceError("Final task identity differs from batch plan")
    if completion.get("status") != "complete" or completion.get("training_complete") is not True:
        raise EvidenceError("Worker has no complete training receipt")
    for key in ("task_id", "protocol"):
        if completion.get(key) != task.get(key):
            raise EvidenceError(f"Completion {key} differs from task")
    if completion.get("metric") != "cindex":
        raise EvidenceError("Completion metric is not cindex")
    completed_epochs, max_epochs = completion.get("completed_epochs"), completion.get("max_epochs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (completed_epochs, max_epochs)):
        raise EvidenceError("Completion lacks positive integer epoch counts")
    if completed_epochs < max_epochs:
        if completion.get("stop_reason") != "early_stop":
            raise EvidenceError("Training ended before max_epochs without an early-stop receipt")
        _verify_file(completion.get("stopping_evidence_artifact"), task_dir, confined=True)
    partition = PROTOCOL_PARTITION[task["protocol"]]
    if completion.get("evaluation_partition") != partition:
        raise EvidenceError("Completion evaluation partition is incompatible with protocol")
    config_path = _verify_file(task.get("config_file"), task_dir)
    split_path = _verify_file(task.get("split_file"), task_dir)
    sources = task.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise EvidenceError("Task lacks hashed source files")
    for reference in sources:
        _verify_file(reference, task_dir)
    recipe = task.get("recipe")
    if not isinstance(recipe, Mapping) or not recipe:
        raise EvidenceError("Task lacks a common data/training recipe")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvidenceError("Completion has no hashed artifacts")
    verified_artifacts = {}
    for artifact in artifacts:
        path = _verify_file(artifact, task_dir, confined=True)
        if path in verified_artifacts:
            raise EvidenceError(f"Duplicate artifact path: {path}")
        verified_artifacts[path] = artifact["sha256"].lower()
    prediction_reference = completion.get("prediction_artifact")
    prediction_path = _verify_file(prediction_reference, task_dir, confined=True)
    required_role = "best_val" if partition == "val" else "outer_test"
    if prediction_reference.get("role") != required_role:
        raise EvidenceError("Prediction receipt role is not the protocol's final evaluation")
    if partition == "val" and prediction_path.name != f"split_{task['fold']}_results.pkl":
        raise EvidenceError("Legacy validation must use the actual best-epoch prediction file")
    audit_path = _verify_file(completion.get("label_audit_artifact"), task_dir, confined=True)
    checkpoint_path = _verify_file(completion.get("checkpoint_artifact"), task_dir, confined=True)
    for path in (prediction_path, audit_path, checkpoint_path):
        if path not in verified_artifacts:
            raise EvidenceError(f"Required artifact missing from artifacts list: {path}")
    if checkpoint_path.stat().st_size == 0:
        raise EvidenceError("Checkpoint artifact is empty")
    predictions = read_predictions(prediction_path)
    if completion.get("prediction_patient_count") != len(predictions):
        raise EvidenceError("Completion patient count differs from predictions")
    verify_prediction_labels(predictions, _json(audit_path), split_path, partition)
    score = compute_cindex(predictions)
    recorded_score = _number(completion.get("score"), "completion.score")
    if abs(recorded_score - score) > SCORE_TOLERANCE:
        raise EvidenceError(f"Recorded C-index {recorded_score} differs from recomputed {score}")
    # Model flags differ across arms; recipe records the shared data/train settings.
    fairness = json.dumps({
        "recipe": recipe,
        "split_sha256": task["split_file"]["sha256"].lower(),
        "source_files": sorted((str(_resolve(item["path"], task_dir)), item["sha256"].lower()) for item in sources),
    }, sort_keys=True, ensure_ascii=False)
    labels = {patient: (row["time"], row["censor"]) for patient, row in predictions.items()}
    return {
        "task_id": task["task_id"], "study": identity[0], "seed": identity[1],
        "protocol": identity[2], "fold": identity[3], "arm": identity[4],
        "cindex": score, "n_patients": len(predictions),
        "n_events": sum(row["censor"] == 0 for row in predictions.values()),
        "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
        "config_path": str(config_path), "config_sha256": task["config_file"]["sha256"],
        "split_sha256": task["split_file"]["sha256"],
        "_labels": labels, "_fairness": fairness,
    }


def summarize_batch(batch_dir: Path | str) -> dict[str, Any]:
    batch_dir = Path(batch_dir).resolve()
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = batch_dir / "plan.json"
    manifest = _json(manifest_path)
    tasks = manifest.get("tasks") if isinstance(manifest, Mapping) else None
    if not isinstance(tasks, list) or not tasks:
        raise EvidenceError("Batch manifest/plan requires a non-empty tasks list")
    declared_folds = manifest.get("folds", list(EXPECTED_FOLDS))
    if not isinstance(declared_folds, list) or not declared_folds or any(isinstance(fold, bool) or not isinstance(fold, int) or fold not in EXPECTED_FOLDS for fold in declared_folds) or len(set(declared_folds)) != len(declared_folds):
        raise EvidenceError("Declared folds must be unique members of 0,1,2,3,4")
    if manifest.get("expected_folds", list(EXPECTED_FOLDS)) != list(EXPECTED_FOLDS):
        raise EvidenceError("E048 complete summaries require exactly five expected folds")
    identities, ids = set(), set()
    task_dirs: dict[str, Path] = {}
    for task in tasks:
        identity = _identity(task)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise EvidenceError("Planned task lacks task_id")
        if identity in identities or task_id in ids:
            raise EvidenceError("Duplicate planned task identity/task_id")
        identities.add(identity)
        ids.add(task_id)
        location = task.get("output_dir", task.get("task_dir", task_id))
        task_dirs[task_id] = _resolve(location, batch_dir, confined=True)
    planned_groups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for study, seed, protocol, fold, arm in identities:
        planned_groups[(study, seed, protocol)].add(arm)
    # A plan may deliberately select a subset of arms, but its declared arms are
    # authoritative. Every planned group still needs Full and all five folds.
    declared_arms = manifest.get("arms", manifest.get("expected_arms", manifest.get("selected_arms")))
    if isinstance(declared_arms, list) and declared_arms and all(isinstance(arm, str) for arm in declared_arms):
        for arms in planned_groups.values():
            arms.update(declared_arms)
        declared_studies = manifest.get("studies", sorted({task["study"] for task in tasks}))
        declared_seeds = manifest.get("seeds", sorted({task["seed"] for task in tasks}))
        declared_protocols = manifest.get("protocols", sorted({task["protocol"] for task in tasks}))
        if isinstance(declared_studies, list) and isinstance(declared_seeds, list) and isinstance(declared_protocols, list):
            for study in declared_studies:
                for seed in declared_seeds:
                    for protocol in declared_protocols:
                        _identity({"study": study, "seed": seed, "protocol": protocol, "fold": 0, "arm": declared_arms[0]})
                        planned_groups[(study, seed, protocol)].update(declared_arms)
    errors, records = [], []
    for task in tasks:
        try:
            records.append(verify_task(task_dirs[task["task_id"]], task))
        except (EvidenceError, OSError, KeyError, TypeError) as exc:
            errors.append({"task_id": task["task_id"], "error": str(exc)})
    # Compare all controls within each study/seed/fold, not only whichever Full
    # happens to have finished first. A mismatch invalidates the entire cohort.
    by_cohort: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_cohort[(row["study"], row["seed"], row["protocol"], row["fold"])].append(row)
    invalid_cohorts = set()
    for cohort, rows in by_cohort.items():
        first = rows[0]
        for row in rows[1:]:
            if row["_labels"] != first["_labels"] or row["_fairness"] != first["_fairness"]:
                invalid_cohorts.add(cohort)
                errors.append({"cohort": list(cohort), "error": "Controls differ in patients, outcomes, split or common data/training/source recipe"})
                break
    # Five copies of the same split are not five-fold cross-validation.
    by_arm: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_arm[(row["study"], row["seed"], row["protocol"], row["arm"])].append(row)
    for arm_group, rows in by_arm.items():
        seen: set[str] = set()
        overlapping = False
        for row in sorted(rows, key=lambda row: row["fold"]):
            if seen.intersection(row["_labels"]):
                overlapping = True
            seen.update(row["_labels"])
        if overlapping:
            for row in rows:
                invalid_cohorts.add((row["study"], row["seed"], row["protocol"], row["fold"]))
            errors.append({"group": list(arm_group), "error": "Evaluation patients overlap across folds; this is not disjoint five-fold evidence"})
    verified = [row for row in records if (row["study"], row["seed"], row["protocol"], row["fold"]) not in invalid_cohorts]
    by_identity = {(row["study"], row["seed"], row["protocol"], row["fold"], row["arm"]): row for row in verified}
    summaries, differences, paired_summaries = [], [], []
    for group, arms in sorted(planned_groups.items()):
        study, seed, protocol = group
        full_arms = [arm for arm in arms if arm.lower() == "full"]
        full_arm = full_arms[0] if len(full_arms) == 1 else None
        if full_arm is None:
            errors.append({"group": list(group), "error": "Exactly one Full arm is required for matched ablation reporting"})
        group_complete = full_arm is not None and all((*group, fold, arm) in by_identity for arm in arms for fold in EXPECTED_FOLDS)
        for arm in sorted(arms):
            rows = [by_identity.get((*group, fold, arm)) for fold in EXPECTED_FOLDS]
            missing_folds = [fold for fold, row in zip(EXPECTED_FOLDS, rows) if row is None]
            complete = not missing_folds and group_complete
            scores = [row["cindex"] for row in rows if row is not None]
            summaries.append({
                "study": study, "seed": seed, "protocol": protocol, "arm": arm,
                "status": "complete" if complete else "incomplete", "n_verified_folds": len(scores),
                "missing_folds": missing_folds, "cindex_mean": statistics.mean(scores) if complete else None,
                "cindex_sample_std": statistics.stdev(scores) if complete else None,
            })
            if missing_folds:
                errors.append({"group": [*group, arm], "error": f"Missing/invalid folds: {missing_folds}"})
            if full_arm is None or arm == full_arm:
                continue
            pairs = []
            for fold in EXPECTED_FOLDS:
                full = by_identity.get((*group, fold, full_arm))
                control = by_identity.get((*group, fold, arm))
                if full is None or control is None:
                    continue
                delta = full["cindex"] - control["cindex"]
                pairs.append(delta)
                differences.append({
                    "study": study, "seed": seed, "protocol": protocol, "fold": fold,
                    "full_arm": full_arm, "control_arm": arm, "full_cindex": full["cindex"],
                    "control_cindex": control["cindex"], "full_minus_control": delta,
                })
            pair_complete = len(pairs) == 5 and group_complete
            paired_summaries.append({
                "study": study, "seed": seed, "protocol": protocol,
                "full_arm": full_arm, "control_arm": arm,
                "status": "complete" if pair_complete else "incomplete", "n_paired_folds": len(pairs),
                "mean_full_minus_control": statistics.mean(pairs) if pair_complete else None,
                "sample_std_full_minus_control": statistics.stdev(pairs) if pair_complete else None,
            })
    return {
        "schema_version": SCHEMA_VERSION, "experiment": "E048",
        "batch_dir": str(batch_dir), "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "status": "complete" if not errors else "incomplete_or_invalid", "errors": errors,
        "fold_results": [{key: value for key, value in row.items() if not key.startswith("_")} for row in verified],
        "five_fold_summary": summaries, "paired_differences": differences,
        "paired_five_fold_summary": paired_summaries,
        "evidence_boundary": "Real prediction evidence only; legacy_val and outer_test are separate. Five-fold SD is descriptive; seed x fold is not an independent significance sample. No p-values are produced.",
    }


def _csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_report(report: Mapping[str, Any], output_dir: Path | str) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _csv(output_dir / "fold_results.csv", report["fold_results"], ["task_id", "study", "seed", "protocol", "fold", "arm", "cindex", "n_patients", "n_events", "prediction_path", "prediction_sha256", "config_path", "config_sha256", "split_sha256"])
    _csv(output_dir / "five_fold_summary.csv", report["five_fold_summary"], ["study", "seed", "protocol", "arm", "status", "n_verified_folds", "missing_folds", "cindex_mean", "cindex_sample_std"])
    _csv(output_dir / "paired_differences.csv", report["paired_differences"], ["study", "seed", "protocol", "fold", "full_arm", "control_arm", "full_cindex", "control_cindex", "full_minus_control"])
    _csv(output_dir / "paired_five_fold_summary.csv", report["paired_five_fold_summary"], ["study", "seed", "protocol", "full_arm", "control_arm", "status", "n_paired_folds", "mean_full_minus_control", "sample_std_full_minus_control"])
    with (output_dir / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(
            "# E048 结果核验\n\n"
            f"核验状态：`{report['status']}`。错误/缺失条目：{len(report['errors'])}。\n\n"
            "只读取完成训练且包含哈希的最佳预测、checkpoint、实际评估 loader 标签；"
            "逐患者核对划分与结局，从预测重算 Harrell C-index，再核对 worker 成绩。\n\n"
            "各癌种、随机种子、评估协议分别统计五折均值及样本标准差（ddof=1）。"
            "不足五折的组不发布均值或标准差；具体缺失见 report.json。"
            "Full 减对照是同一折的成对差值。种子与折不当作独立样本，未计算显著性或 p 值。\n\n"
            "legacy_val 是按验证集选 checkpoint 的验证成绩；outer_test 才是选定"
            "checkpoint 后一次评估的外层测试成绩，二者不混合。"
            "结果汇总不证明机制因果关系，也不自动证明每个模块有稳定收益。\n\n"
            "代码测试使用临时合成 fixture，只验证核验与统计逻辑，不是模型实验成绩。"
            "本命令不训练、不推理、不生成缺失分数，不覆盖旧汇总目录。\n"
        )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    try:
        report = summarize_batch(args.batch_dir)
        output = write_report(report, args.output_dir)
    except (EvidenceError, OSError) as exc:
        print(f"E048 evidence rejected: {exc}")
        return 2
    print(f"E048 audit: {report['status']}; report: {output}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
