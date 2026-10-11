"""Pure scheduling and patient-split checks for the E048 experiment batch.

This module imports neither torch nor the dataset/trainer. Importing it cannot
start a model or read feature bags. Split validation reads CSV files only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from itertools import product
from pathlib import Path
from typing import Any, Iterable


ARM_GROUPS: dict[str, tuple[str, ...]] = {
    "first": (
        "full", "wsi_only", "omics_only", "concat", "no_self", "no_cross",
        "no_reconstruction",
    ),
    "structure": (
        "full", "direct", "independent", "no_prototypes", "shared_stage_plan",
        "single_geometry", "no_event_encoder", "pooled_transport",
    ),
    "loss": ("full", "no_rank", "no_slot_nll", "no_diversity"),
}
ALL_ARMS = tuple(dict.fromkeys(arm for arms in ARM_GROUPS.values() for arm in arms))
ARM_GROUPS["all"] = ALL_ARMS
PROTOCOLS = ("legacy_val", "outer_test")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PlanError(ValueError):
    """An invalid plan or split that must prevent experiment execution."""


def ordered_unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def comma_values(values: list[str] | None, defaults: Iterable[str]) -> list[str]:
    """Accept repeated options and comma-separated entries without duplicates."""
    if values is None:
        return ordered_unique(defaults)
    parsed = [item.strip() for value in values for item in value.split(",")]
    if not parsed or any(not item for item in parsed):
        raise PlanError("Options must contain nonempty comma-separated values.")
    return ordered_unique(parsed)


def integer_values(
    values: list[str] | None, defaults: Iterable[int], *, name: str, maximum: int | None = None,
) -> list[int]:
    parsed = comma_values(values, (str(item) for item in defaults))
    try:
        numbers = ordered_unique(int(item) for item in parsed)
    except ValueError as exc:
        raise PlanError(f"{name} must contain integers.") from exc
    if any(item < 0 or (maximum is not None and item > maximum) for item in numbers):
        limit = f" between 0 and {maximum}" if maximum is not None else " >= 0"
        raise PlanError(f"{name} values must be{limit}.")
    return numbers


def selected_arms(group: str, explicit: list[str] | None) -> list[str]:
    arms = comma_values(explicit, ARM_GROUPS[group])
    unknown = [arm for arm in arms if arm not in ALL_ARMS]
    if unknown:
        raise PlanError(f"Unknown arms: {', '.join(unknown)}.")
    return arms


def safe_name(value: str, *, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise PlanError(f"{label} must contain only letters, numbers, dots, underscores or hyphens.")
    # A colon, separator or traversal is forbidden above; also reject device names
    # so the same plan can be materialized on Windows and Linux.
    stem = value.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", stem):
        raise PlanError(f"{label} is a reserved Windows device name.")
    return value


def patient_id(value: str) -> str:
    """Collapse TCGA sample/slide barcodes to their patient barcode."""
    normalized = value.strip()
    if normalized.upper().startswith("TCGA-"):
        fields = normalized.upper().split("-")
        if len(fields) < 3 or not fields[1] or not fields[2]:
            raise PlanError(f"Malformed TCGA identifier: {value!r}.")
        return "-".join(fields[:3])
    return normalized


def validate_split(path: Path, protocol: str) -> dict[str, Any]:
    """Hash and validate the same CSV bytes; never infer a held-out test split."""
    if protocol not in PROTOCOLS:
        raise PlanError(f"Unknown protocol: {protocol}.")
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise PlanError(f"Cannot read split CSV {path}: {exc}") from exc
    try:
        reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig")))
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise PlanError(f"Duplicate CSV column names in {path}.")
        required = {"inner_train", "inner_val", "outer_test"} if protocol == "outer_test" else {"train", "val"}
        if not required.issubset(headers):
            raise PlanError(f"Split {path} requires columns {sorted(required)}; got {headers}.")
        allowed = ("inner_train", "inner_val", "outer_test") if protocol == "outer_test" else ("train", "val", "test")
        partitions = {column: set() for column in allowed if column in headers}
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise PlanError(f"Extra CSV fields in {path}, line {line_number}.")
            for column, patients in partitions.items():
                value = row.get(column)
                if value is None:
                    raise PlanError(f"Missing CSV fields in {path}, line {line_number}.")
                stripped = value.strip()
                if not stripped or stripped.lower() in {"nan", "none", "null", "na"}:
                    continue
                identifier = patient_id(stripped)
                if identifier in patients:
                    raise PlanError(f"Duplicate patient {identifier} in {column} of {path}.")
                patients.add(identifier)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise PlanError(f"Invalid UTF-8 split CSV {path}: {exc}") from exc
    for column in required:
        if not partitions[column]:
            raise PlanError(f"Split partition {column} is empty in {path}.")
    columns = list(partitions)
    for index, first in enumerate(columns):
        for second in columns[index + 1:]:
            overlap = partitions[first] & partitions[second]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise PlanError(f"Patient overlap between {first}/{second} in {path}: {sample}.")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "sizes": {column: len(patients) for column, patients in partitions.items()},
    }


def split_path(data_path: Path, split_set: str, study: str, fold: int) -> Path:
    root = Path(split_set).expanduser()
    if not root.is_absolute():
        root = data_path / "splits" / root
    return root / study / f"fold_{fold}.csv"


def build_tasks(
    *, arms: list[str], studies: list[str], folds: list[int], seeds: list[int],
    gpus: list[int], batch_dir: Path, data_path: Path, split_set: str,
    protocol: str, python_executable: str, worker_path: Path, require_splits: bool,
) -> list[dict[str, Any]]:
    """Build the complete grid before creating directories or launching workers."""
    if protocol not in PROTOCOLS:
        raise PlanError(f"Unknown protocol: {protocol}.")
    if not all((arms, studies, folds, seeds, gpus)):
        raise PlanError("Every task dimension must contain at least one value.")
    if len(arms) != len(set(arms)) or any(arm not in ALL_ARMS for arm in arms):
        raise PlanError("Arms must be unique and belong to the registered E048 arms.")
    if len(studies) != len(set(studies)) or len(folds) != len(set(folds)) or len(seeds) != len(set(seeds)):
        raise PlanError("Task dimensions must not contain duplicate values.")
    for study in studies:
        safe_name(study, label="cancer")
    fingerprints: dict[tuple[str, int], dict[str, Any]] = {}
    for study, fold in product(studies, folds):
        source = split_path(data_path, split_set, study, fold)
        try:
            fingerprints[(study, fold)] = validate_split(source, protocol)
        except PlanError as exc:
            if require_splits:
                raise
            fingerprints[(study, fold)] = {
                "path": str(source.resolve()), "sha256": None, "sizes": None,
                "error": str(exc),
            }
    tasks = []
    # For each cohort/fold/seed, arms are adjacent and use one cached split hash.
    for index, (study, fold, seed, arm) in enumerate(product(studies, folds, seeds, arms)):
        task_id = f"{protocol}/{arm}/{study}/fold{fold}/seed{seed}"
        task_dir = batch_dir / task_id
        fingerprint = fingerprints[(study, fold)]
        config_path = task_dir / "config.yaml"
        tasks.append({
            "task_id": task_id, "arm": arm, "study": study, "fold": fold, "seed": seed,
            "gpu_visible": str(gpus[index % len(gpus)]), "protocol": protocol,
            "output_dir": str(task_dir), "config_path": str(config_path),
            "split_path": fingerprint["path"], "split_sha256": fingerprint["sha256"],
            "split_sizes": fingerprint["sizes"], "split_error": fingerprint.get("error"),
            "command": [
                python_executable, str(worker_path), "--arm", arm, "--config", str(config_path),
                "--output-dir", str(task_dir), "--protocol", protocol,
                "--fold", str(fold), "--seed", str(seed), "--gpu", "0",
            ],
        })
    return tasks


def check_split_unchanged(task: dict[str, Any]) -> None:
    """Catch split edits between batch planning and each worker launch."""
    actual = validate_split(Path(task["split_path"]), task["protocol"])
    if actual["sha256"] != task["split_sha256"]:
        raise PlanError(f"Split changed after planning: {task['split_path']}.")
