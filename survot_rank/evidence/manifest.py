"""Explicit provenance and fail-closed auditing of best-validation results."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("runs"):
        raise ValueError("Expected schema_version=1 and a nonempty runs list")
    for run in manifest["runs"]:
        for key in ("config", "checkpoint", "curve", "predictions", "split_csv"):
            if run.get(key):
                run[key] = str((path.parent / run[key]).resolve())
    return manifest


def cindex(times, censor, risk, tied_tol=1e-8):
    """Harrell C, including event/censor ties; no fallback score on failure."""
    times, censor, risk = [np.asarray(x, dtype=float).reshape(-1) for x in (times, censor, risk)]
    if not (times.size == censor.size == risk.size) or not times.size:
        raise ValueError("Outcome and risk lengths differ or are empty")
    if not all(np.isfinite(x).all() for x in (times, censor, risk)) or not np.isin(censor, [0, 1]).all():
        raise ValueError("Invalid outcomes or nonfinite risks")
    score, count = 0.0, 0
    for i in np.flatnonzero(censor == 0):
        comparable = (times > times[i]) | ((times == times[i]) & (censor == 1))
        difference = risk[i] - risk[comparable]
        score += float((difference > tied_tol).sum() + 0.5 * (np.abs(difference) <= tied_tol).sum())
        count += difference.size
    if not count:
        raise ValueError("No comparable survival pairs")
    return score / count


def best_epoch(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty epoch curve")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError("Epochs must be unique and increasing")
    values = np.asarray([float(row["val_cindex"]) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite validation C-index in epoch curve")
    index = int(values.argmax())
    return epochs[index], float(values[index]), rows


def predictions(path):
    # Read only trusted experiment artifacts supplied by the researcher.
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not data:
        raise ValueError("Predictions must be a nonempty patient-keyed dictionary")
    ids = sorted(data)
    if any(not isinstance(cid, str) for cid in ids):
        raise ValueError("Patient IDs must be strings")
    vectors = {}
    for key in ("risk", "time", "censor"):
        values = [np.asarray(data[cid][key]).reshape(-1) for cid in ids]
        if any(value.size != 1 for value in values):
            raise ValueError(f"Each patient must have exactly one {key}")
        vectors[key] = np.array([float(value[0]) for value in values])
    vectors["case_ids"] = ids
    vectors["cindex"] = cindex(vectors["time"], vectors["censor"], vectors["risk"])
    return vectors


def split_ids(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {key: [row.get(key, "").strip() for row in rows if row.get(key, "").strip()] for key in ("train", "val")}
    if not all(result.values()) or any(len(set(ids)) != len(ids) for ids in result.values()):
        raise ValueError("Missing or duplicated train/val patient IDs")
    if set(result["train"]) & set(result["val"]):
        raise ValueError("Training and validation patients overlap")
    return result


def audit(manifest):
    report = {"schema_version": 1, "protocol": "legacy_val", "runs": [], "errors": []}
    identities, hashes = {}, {}
    for run in manifest["runs"]:
        row = {key: run.get(key) for key in ("id", "arm", "cancer", "fold", "seed", "protocol", "source_commit")}
        row["errors"] = []
        try:
            identity = tuple(run[key] for key in ("arm", "cancer", "fold", "seed", "protocol"))
            if run["protocol"] != "legacy_val":
                raise ValueError("This toolkit evaluates legacy_val only")
            if run["id"] in identities or identity in identities.values():
                raise ValueError("Duplicate run ID or experimental identity")
            identities[run["id"]] = identity
            if not run.get("source_commit"):
                raise ValueError("Supply the actual training source_commit; do not guess current HEAD")
            if not 0 <= int(run["fold"]) <= 4:
                raise ValueError("Fold must be 0..4")
            row["hashes"] = {key: sha256(run[key]) for key in ("config", "curve", "predictions", "split_csv")}
            if run.get("checkpoint"):
                row["hashes"]["checkpoint"] = sha256(run["checkpoint"])
            epoch, value, _ = best_epoch(run["curve"])
            pred, ids = predictions(run["predictions"]), split_ids(run["split_csv"])
            expected = run.get("expected_val_ids", ids["val"])
            if not expected or len(expected) != len(set(expected)) or not set(expected) <= set(ids["val"]):
                raise ValueError("expected_val_ids must be unique validation IDs from the split")
            if set(expected) != set(ids["val"]) and not run.get("filter_reason"):
                raise ValueError("A filtered validation subset requires an explicit filter_reason")
            if set(pred["case_ids"]) != set(expected):
                raise ValueError("Prediction patient IDs differ from validation split (no silent sample dropping)")
            if abs(pred["cindex"] - value) > 5e-5:
                raise ValueError(f"Best predictions C={pred['cindex']:.8f} differs from curve C={value:.8f}; wrong epoch/final pickle?")
            row.update(best_epoch=epoch, cindex=pred["cindex"], n=len(pred["case_ids"]),
                       case_ids=pred["case_ids"], time=pred["time"].tolist(), censor=pred["censor"].tolist(),
                       train_ids=ids["train"], raw_val_ids=ids["val"])
            for key in ("curve", "predictions", "checkpoint"):
                digest = row["hashes"].get(key)
                if digest:
                    hashes.setdefault((key, digest), []).append(row)
        except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            row["errors"].append(str(exc))
        report["runs"].append(row)
    for (kind, _), rows in hashes.items():
        if len(rows) > 1:
            for row in rows:
                row["errors"].append(f"Byte-identical {kind} reused by runs: {', '.join(r['id'] for r in rows)}")
    groups = {}
    for row in report["runs"]:
        if "case_ids" in row:
            groups.setdefault(tuple(row[key] for key in ("arm", "cancer", "seed", "protocol")), []).append(row)
    for rows in groups.values():
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                if set(left["case_ids"]) & set(right["case_ids"]):
                    left["errors"].append(f"Validation patients overlap across folds with {right['id']}")
                    right["errors"].append(f"Validation patients overlap across folds with {left['id']}")
        if len(rows) == 5 and {row["fold"] for row in rows} == set(range(5)):
            universes = [set(row["train_ids"]) | set(row["raw_val_ids"]) for row in rows]
            raw_val = [cid for row in rows for cid in row["raw_val_ids"]]
            if any(universe != universes[0] for universe in universes) or set(raw_val) != universes[0] or len(raw_val) != len(set(raw_val)):
                for row in rows:
                    row["errors"].append("Five fold CSVs do not partition one common patient cohort")
    for row in report["runs"]:
        row["status"] = "failed" if row["errors"] else "passed"
        report["errors"].extend(f"{row['id']}: {error}" for error in row["errors"])
    report["passed"] = not report["errors"]
    return report


def discover(root, config, arm, cancer, seed, source_commit, data_path, overrides=()):
    """Explicit group labels; never infer cancer/arm from an ambiguous folder."""
    from survot_rank.config import apply_overrides, flatten_config, load_config
    config = Path(config).resolve()
    flat = flatten_config(apply_overrides(load_config(config), list(overrides)))
    runs = []
    for curve in sorted(Path(root).resolve().rglob("epoch_curve_fold*.csv")):
        fold = int(curve.stem.removeprefix("epoch_curve_fold"))
        parent = curve.parent
        checkpoint = parent / f"model_best_s{fold}.pth"
        runs.append(dict(id=f"{arm}_{cancer}_f{fold}_s{seed}", arm=arm, cancer=cancer, fold=fold,
                         seed=seed, protocol="legacy_val", source_commit=source_commit,
                         config=str(config), overrides=list(overrides), curve=str(curve),
                         predictions=str(parent / f"split_{fold}_results.pkl"),
                         checkpoint=str(checkpoint) if checkpoint.exists() else None,
                         split_csv=str(Path(data_path).resolve() / "splits" / flat.get("which_splits", "5fold_uni2h") / cancer / f"fold_{fold}.csv")))
    if not runs:
        raise ValueError(f"No epoch_curve_fold*.csv under {root}; specify explicit artifact paths for other trainers")
    return runs
