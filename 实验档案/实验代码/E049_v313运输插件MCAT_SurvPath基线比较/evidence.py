"""Prediction and split verification reused from E048; no training or inference."""
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
