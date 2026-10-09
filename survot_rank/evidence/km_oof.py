"""Pool validation risk *groups* across five folds without pooling risk scales."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .manifest import sha256, split_ids


def canonical_arm(arm):
    return "exp6" if arm in ("full", "exp6") else arm


def load_fold_export(path):
    """Read an audited export with a recorded training-risk median; no inference."""
    path = Path(path).resolve()
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise ValueError(f"Unsupported export schema: {path}")
    run = meta["run"]
    for key in ("id", "arm", "cancer", "seed", "fold", "protocol", "source_commit", "split_csv"):
        if key not in run:
            raise ValueError(f"Missing run field {key}: {path}")
    if run["protocol"] != "legacy_val":
        raise ValueError("This figure describes legacy_val five-fold development predictions")
    threshold = meta.get("km_train_median")
    if threshold is None or not np.isfinite(float(threshold)):
        raise ValueError(f"Missing finite km_train_median: {path}; export this run with --km")
    split_path = Path(run["split_csv"])
    if not split_path.is_absolute():
        split_path = path.parent / split_path
    expected_hash = meta.get("hashes", {}).get("split_csv")
    if not expected_hash or sha256(split_path) != expected_hash:
        raise ValueError(f"Missing or mismatched audited split hash: {path}")
    split = split_ids(split_path)
    patients_path = path.with_name("patients.npz")
    with np.load(patients_path, allow_pickle=False) as data:
        ids = np.asarray(data["case_ids"], dtype=str).reshape(-1)
        arrays = {key: np.asarray(data[key], dtype=float).reshape(-1) for key in ("risk", "time", "censor")}
    n = len(ids)
    if not n or len(set(ids)) != n or any(not cid.strip() for cid in ids):
        raise ValueError(f"Empty or duplicate patient IDs: {path}")
    if any(len(value) != n or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Outcome/risk length or finite-value mismatch: {path}")
    if (arrays["time"] < 0).any() or not np.isin(arrays["censor"], [0, 1]).all():
        raise ValueError(f"Invalid survival outcomes: {path}")
    if set(ids) != set(split["val"]):
        raise ValueError(f"Patient IDs do not cover this complete validation fold: {path}")
    return dict(run=run, threshold=float(threshold), case_ids=ids, **arrays,
                train_ids=split["train"], val_ids=split["val"], source=dict(
                    export=str(path), export_sha256=sha256(path),
                    patients=str(patients_path), patients_sha256=sha256(patients_path),
                    artifact_hashes=meta.get("hashes", {}), best_epoch=meta.get("best_epoch")))


def pool_folds(records, *, cancer, arm="exp6", seed=3):
    """Require one complete partition; assign high risk within each fold first."""
    if len(records) != 5 or sorted(int(r["run"]["fold"]) for r in records) != list(range(5)):
        raise ValueError(f"{cancer}/{arm}: require exactly folds 0..4, without duplicates")
    rows, seen, universe, fold_info = [], set(), None, []
    for record in sorted(records, key=lambda r: int(r["run"]["fold"])):
        run = record["run"]
        if (run["cancer"] != cancer or canonical_arm(run["arm"]) != canonical_arm(arm)
                or int(run["seed"]) != seed or run["protocol"] != "legacy_val"):
            raise ValueError("Cannot pool different cancers, models, seeds, or protocols")
        current = set(record["train_ids"]) | set(record["val_ids"])
        if universe is None:
            universe = current
        elif universe != current:
            raise ValueError("The five splits do not refer to one common patient cohort")
        ids = record["case_ids"]
        if seen & set(ids):
            raise ValueError("A validation patient appears in more than one fold")
        seen.update(ids)
        # DCT risk is -sum(survival): larger / less negative means higher risk.
        high = record["risk"] >= record["threshold"]
        for i, cid in enumerate(ids):
            rows.append(dict(case_id=str(cid), fold=int(run["fold"]),
                risk=float(record["risk"][i]), train_median=record["threshold"],
                group="high" if high[i] else "low", time=float(record["time"][i]),
                censor=int(record["censor"][i]), event=int(record["censor"][i] == 0)))
        fold_info.append(dict(fold=int(run["fold"]), run_id=run["id"],
            n=len(ids), high_n=int(high.sum()), low_n=int((~high).sum()),
            train_median=record["threshold"], source=record["source"]))
    if seen != universe:
        raise ValueError("Validation folds do not partition the whole shared patient cohort")
    if {row["group"] for row in rows} != {"high", "low"}:
        raise ValueError("Pooled cohort must contain both risk groups")
    rows.sort(key=lambda row: row["case_id"])
    return dict(schema_version=1, cancer=cancer, arm=canonical_arm(arm), seed=seed,
        protocol="legacy_val", grouping="within-fold training-risk median, then pool labels",
        risk_direction="higher_is_worse", threshold_ties="high", n=len(rows),
        high_n=sum(row["group"] == "high" for row in rows),
        low_n=sum(row["group"] == "low" for row in rows), folds=fold_info, patients=rows,
        note="Five-fold out-of-fold grouping at validation-selected checkpoints; descriptive development evidence. "
             "Raw risks are not ranked across folds. Pooled log-rank/CI ignore CV model dependence.")


def collect_cohorts(export_root, cancers=("blca", "kirc"), *, arm="exp6", seed=3):
    """Select exports by recorded identity, never by directory name or first match."""
    selected = {cancer: [] for cancer in cancers}
    for path in sorted(Path(export_root).resolve().rglob("export.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        run = meta.get("run", {})
        cancer = run.get("cancer")
        if (cancer in selected and canonical_arm(run.get("arm")) == canonical_arm(arm)
                and run.get("seed") == seed):
            selected[cancer].append(load_fold_export(path))
    return {c: pool_folds(rs, cancer=c, arm=arm, seed=seed) for c, rs in selected.items()}
