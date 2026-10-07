import csv
import json

import numpy as np
import pytest

from survot_rank.evidence.km_oof import collect_cohorts, load_fold_export, pool_folds
from survot_rank.evidence.manifest import sha256


@pytest.fixture
def exported(tmp_path):
    ids = [f"patient{i}" for i in range(10)]
    paths = []
    for fold in range(5):
        val = ids[2 * fold:2 * fold + 2]
        train = [cid for cid in ids if cid not in val]
        split = tmp_path / f"fold_{fold}.csv"
        with split.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["train", "val"])
            for i, cid in enumerate(train):
                writer.writerow([cid, val[i] if i < 2 else ""])
        directory = tmp_path / "exports" / f"run{fold}"
        directory.mkdir(parents=True)
        # Distinct fold-specific scales expose an incorrect pooled raw-risk median.
        threshold = -10.0 * (fold + 1)
        np.savez(directory / "patients.npz", case_ids=val,
            risk=[threshold - 1, threshold + 1], time=[20, 10], censor=[1, 0])
        meta = dict(schema_version=1, km_train_median=threshold, best_epoch=3,
            hashes={"split_csv": sha256(split)}, run=dict(id=f"run{fold}", arm="exp6",
                cancer="blca", fold=fold, seed=3, protocol="legacy_val", source_commit="fixture",
                split_csv=str(split)))
        path = directory / "export.json"
        path.write_text(json.dumps(meta))
        paths.append(path)
    return tmp_path / "exports", paths


def test_pool_labels_before_combining_different_risk_scales(exported):
    root, _ = exported
    result = collect_cohorts(root, ["blca"])["blca"]
    assert result["n"] == 10 and result["high_n"] == result["low_n"] == 5
    for row in result["patients"]:
        assert row["group"] == ("high" if int(row["case_id"].removeprefix("patient")) % 2 else "low")
        assert row["event"] == 1 - row["censor"]


def test_zero_threshold_is_valid_and_ties_are_high(exported):
    _, paths = exported
    records = [load_fold_export(p) for p in paths]
    records[0]["threshold"] = 0.0
    records[0]["risk"] = np.array([0.0, -1.0])
    result = pool_folds(records, cancer="blca")
    assert result["patients"][0]["group"] == "high"
    assert result["patients"][1]["group"] == "low"


def test_missing_fold_rejected(exported):
    root, paths = exported
    paths[4].unlink()
    with pytest.raises(ValueError, match="exactly folds"):
        collect_cohorts(root, ["blca"])


def test_duplicate_export_rejected(exported):
    _, paths = exported
    records = [load_fold_export(p) for p in paths]
    with pytest.raises(ValueError, match="exactly folds"):
        pool_folds(records + [records[0]], cancer="blca")


def test_training_threshold_required_without_validation_fallback(exported):
    _, paths = exported
    meta = json.loads(paths[0].read_text())
    meta["km_train_median"] = None
    paths[0].write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="km_train_median"):
        load_fold_export(paths[0])


def test_patient_fold_identity_checked(exported):
    _, paths = exported
    patients = paths[0].with_name("patients.npz")
    np.savez(patients, case_ids=["patient8", "patient9"], risk=[-2, -1], time=[20, 10], censor=[1, 0])
    with pytest.raises(ValueError, match="complete validation fold"):
        load_fold_export(paths[0])


def test_repeated_validation_patient_rejected(exported):
    _, paths = exported
    records = [load_fold_export(p) for p in paths]
    records[1]["case_ids"][0] = records[0]["case_ids"][0]
    with pytest.raises(ValueError, match="more than one fold"):
        pool_folds(records, cancer="blca")


def test_mixed_seed_rejected(exported):
    _, paths = exported
    records = [load_fold_export(p) for p in paths]
    records[2]["run"]["seed"] = 4
    with pytest.raises(ValueError, match="different cancers"):
        pool_folds(records, cancer="blca")


def test_changed_split_hash_rejected(exported):
    _, paths = exported
    meta = json.loads(paths[0].read_text())
    with open(meta["run"]["split_csv"], "a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="split hash"):
        load_fold_export(paths[0])


def test_incomplete_shared_cohort_rejected(exported):
    _, paths = exported
    records = [load_fold_export(p) for p in paths]
    records[3]["train_ids"] = records[3]["train_ids"] + ["unexpected_patient"]
    with pytest.raises(ValueError, match="common patient cohort"):
        pool_folds(records, cancer="blca")
