"""Reproducible per-fold evidence packages for formal DCT-Reg runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args), check=False, capture_output=True, text=True
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(_jsonable(payload), file_obj, ensure_ascii=False, indent=2)


def _normalized_ids(series: pd.Series) -> list[str]:
    return sorted(
        {
            str(value).strip().upper()
            for value in series.dropna().tolist()
            if str(value).strip()
        }
    )


def prepare_fold_evidence(args, fold: int, split_path: str | Path) -> Path:
    """Create immutable config, split, environment and initial run metadata."""

    git_dirty = bool(_git_value("status", "--porcelain"))
    if git_dirty and bool(getattr(args, "formal_require_clean_git", False)):
        raise RuntimeError(
            "formal evidence requires a clean Git worktree; commit the exact "
            "training source or disable formal_require_clean_git for a non-formal run"
        )
    evidence_dir = Path(args.results_dir) / "evidence" / f"fold_{fold}"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        key: _jsonable(value)
        for key, value in sorted(vars(args).items())
        if not key.startswith("_")
    }
    with open(evidence_dir / "resolved_config.yaml", "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(resolved, file_obj, allow_unicode=True, sort_keys=True)

    split_path = Path(split_path)
    split_df = pd.read_csv(split_path)
    columns = {
        column: _normalized_ids(split_df[column]) for column in split_df.columns
    }
    overlaps: dict[str, list[str]] = {}
    names = list(columns)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            shared = sorted(set(columns[left]) & set(columns[right]))
            if shared:
                overlaps[f"{left}__{right}"] = shared
    _write_json(
        evidence_dir / "split_manifest.json",
        {
            "fold": fold,
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "columns": {name: {"count": len(ids), "patient_ids": ids} for name, ids in columns.items()},
            "overlaps": overlaps,
        },
    )

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }
    _write_json(evidence_dir / "environment.json", environment)

    _write_json(
        evidence_dir / "run_manifest.json",
        {
            "status": "running",
            "fold": fold,
            "command": sys.argv,
            "working_directory": os.getcwd(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("symbolic-ref", "--short", "HEAD"),
            "git_dirty": git_dirty,
            "seed": getattr(args, "seed", None),
            "evaluation_protocol": (
                "fixed_epoch_outer_once"
                if getattr(args, "outer_eval_only", False)
                else "legacy_best_validation"
            ),
            "artifacts": {},
        },
    )
    return evidence_dir


def write_predictions_csv(results: dict[str, dict[str, Any]], output_path: str | Path) -> None:
    records = []
    for case_id, values in results.items():
        record = {"case_id": str(case_id)}
        for key, value in values.items():
            array = np.asarray(value)
            record[key] = (
                array.item()
                if array.ndim == 0
                else json.dumps(array.tolist(), separators=(",", ":"))
            )
        records.append(record)
    pd.DataFrame(records).to_csv(output_path, index=False)


def finalize_fold_evidence(
    evidence_dir: str | Path,
    *,
    metrics: dict[str, Any] | None,
    status: str = "complete",
) -> None:
    evidence_dir = Path(evidence_dir)
    manifest_path = evidence_dir / "run_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    artifacts = {}
    for path in sorted(evidence_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            artifact_name = path.relative_to(evidence_dir).as_posix()
            artifacts[artifact_name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    merged_metrics = dict(manifest.get("metrics") or {})
    if metrics:
        merged_metrics.update(_jsonable(metrics))
    manifest.update({"status": status, "metrics": merged_metrics, "artifacts": artifacts})
    _write_json(manifest_path, manifest)
