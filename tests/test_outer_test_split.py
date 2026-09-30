"""Tests for DCT v3.13 §3.1 #5 outer_test split generator.

``paper/V313_IMPLEMENTATION_PLAN.md`` §3.3 requires:

* The original five-fold ``val`` becomes the outer test set.
* The original ``train`` is split 80/20 into ``inner_train`` and
  ``inner_val``, stratified on the censorship label.
* The split is reproducible across model seeds (fixed ``split_seed``).
* Patient IDs are patient-level (no overlap between splits).

These tests verify the generator end-to-end on synthetic clinical data.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from survot_rank.training.outer_test_split import (
    DEFAULT_INNER_TRAIN,
    DEFAULT_INNER_VAL,
    DEFAULT_OUTER_TEST,
    DEFAULT_INNER_VAL_FRACTION,
    DEFAULT_SPLIT_SEED,
    _fingerprint_outer_split,
    _make_outer_test_frame,
    build_outer_test_frame_with_clinical_df,
    generate_outer_test_splits,
    stratified_inner_split,
)


# ---------------------------------------------------------------------------
# Stratified inner split.
# ---------------------------------------------------------------------------


def test_stratified_inner_split_returns_two_disjoint_lists():
    train_ids = [f"TCGA-{i:04d}" for i in range(100)]
    # 30 events, 70 censored.
    event_mask = [1] * 30 + [0] * 70
    inner_train, inner_val = stratified_inner_split(
        train_ids=train_ids,
        event_mask=event_mask,
        inner_val_fraction=0.20,
        seed=3,
    )
    assert len(inner_train) == 80
    assert len(inner_val) == 20
    assert set(inner_train).isdisjoint(inner_val)
    assert set(inner_train) | set(inner_val) == set(train_ids)


def test_stratified_inner_split_preserves_event_rate():
    train_ids = [f"TCGA-{i:04d}" for i in range(100)]
    event_mask = [1] * 30 + [0] * 70
    inner_train, inner_val = stratified_inner_split(
        train_ids=train_ids,
        event_mask=event_mask,
        inner_val_fraction=0.20,
        seed=3,
    )
    # DCT clinical convention: censorship=1 means censored, 0 means event.
    # We must verify the event rate is close to the parent (30%).
    def _event_rate(ids):
        id_set = set(ids)
        rate = sum(e for pid, e in zip(train_ids, event_mask) if pid in id_set) / len(ids)
        return rate
    parent_rate = 30 / 100
    assert abs(_event_rate(inner_train) - parent_rate) < 0.05
    assert abs(_event_rate(inner_val) - parent_rate) < 0.05


def test_stratified_inner_split_is_reproducible_across_seeds():
    train_ids = [f"TCGA-{i:04d}" for i in range(100)]
    event_mask = [1] * 30 + [0] * 70
    inner_train_a, inner_val_a = stratified_inner_split(
        train_ids=train_ids, event_mask=event_mask, inner_val_fraction=0.20, seed=7,
    )
    inner_train_b, inner_val_b = stratified_inner_split(
        train_ids=train_ids, event_mask=event_mask, inner_val_fraction=0.20, seed=7,
    )
    assert inner_train_a == inner_train_b
    assert inner_val_a == inner_val_b


def test_stratified_inner_split_differs_across_seeds():
    train_ids = [f"TCGA-{i:04d}" for i in range(100)]
    event_mask = [1] * 30 + [0] * 70
    inner_train_a, inner_val_a = stratified_inner_split(
        train_ids=train_ids, event_mask=event_mask, inner_val_fraction=0.20, seed=3,
    )
    inner_train_b, inner_val_b = stratified_inner_split(
        train_ids=train_ids, event_mask=event_mask, inner_val_fraction=0.20, seed=11,
    )
    # The 5/95 fractile of difference probability is high for N=100.
    assert inner_train_a != inner_train_b


def test_stratified_inner_split_rejects_invalid_fraction():
    train_ids = ["a", "b"]
    event_mask = [1, 0]
    with pytest.raises(ValueError, match="inner_val_fraction"):
        stratified_inner_split(
            train_ids=train_ids, event_mask=event_mask, inner_val_fraction=0.0, seed=3,
        )
    with pytest.raises(ValueError, match="inner_val_fraction"):
        stratified_inner_split(
            train_ids=train_ids, event_mask=event_mask, inner_val_fraction=1.0, seed=3,
        )


def test_stratified_inner_split_handles_tiny_per_stratum_size():
    """When a stratum has 1 patient we must still send it to inner_train
    (not inner_val), so inner_train is never empty for a stratum."""
    train_ids = ["a", "b", "c", "d", "e"]
    event_mask = [1, 0, 0, 0, 0]
    inner_train, inner_val = stratified_inner_split(
        train_ids=train_ids,
        event_mask=event_mask,
        inner_val_fraction=0.20,
        seed=3,
    )
    # The single event patient must NOT be in inner_val.
    assert "a" in inner_train
    assert "a" not in inner_val


# ---------------------------------------------------------------------------
# Frame assembly.
# ---------------------------------------------------------------------------


def test_make_outer_test_frame_has_three_columns_with_correct_ids():
    inner_train = ["a", "b", "c"]
    inner_val = ["d", "e"]
    outer_test = ["f", "g", "h", "i", "j"]
    frame = _make_outer_test_frame(
        inner_train_ids=inner_train,
        inner_val_ids=inner_val,
        outer_test_ids=outer_test,
        inner_train_col=DEFAULT_INNER_TRAIN,
        inner_val_col=DEFAULT_INNER_VAL,
        outer_test_col=DEFAULT_OUTER_TEST,
    )
    assert list(frame.columns) == [
        DEFAULT_INNER_TRAIN,
        DEFAULT_INNER_VAL,
        DEFAULT_OUTER_TEST,
    ]
    # Padded to the longest column.
    assert len(frame) == 5
    assert set(frame[DEFAULT_INNER_TRAIN].dropna()) == {"a", "b", "c"}
    assert set(frame[DEFAULT_INNER_VAL].dropna()) == {"d", "e"}
    assert set(frame[DEFAULT_OUTER_TEST].dropna()) == {"f", "g", "h", "i", "j"}


def test_fingerprint_is_stable_for_identical_frames():
    inner_train = ["a", "b"]
    inner_val = ["c"]
    outer_test = ["d", "e", "f"]
    f1 = _make_outer_test_frame(inner_train, inner_val, outer_test,
                                  DEFAULT_INNER_TRAIN, DEFAULT_INNER_VAL, DEFAULT_OUTER_TEST)
    f2 = _make_outer_test_frame(inner_train, inner_val, outer_test,
                                  DEFAULT_INNER_TRAIN, DEFAULT_INNER_VAL, DEFAULT_OUTER_TEST)
    assert _fingerprint_outer_split(f1) == _fingerprint_outer_split(f2)


def test_fingerprint_changes_when_split_changes():
    f1 = _make_outer_test_frame(["a", "b"], ["c"], ["d", "e", "f"],
                                 DEFAULT_INNER_TRAIN, DEFAULT_INNER_VAL, DEFAULT_OUTER_TEST)
    f2 = _make_outer_test_frame(["b", "a"], ["c"], ["d", "e", "f"],
                                 DEFAULT_INNER_TRAIN, DEFAULT_INNER_VAL, DEFAULT_OUTER_TEST)
    assert _fingerprint_outer_split(f1) != _fingerprint_outer_split(f2)


# ---------------------------------------------------------------------------
# End-to-end with clinical DF (stratification).
# ---------------------------------------------------------------------------


def _synthetic_clinical_df(n_events: int, n_censored: int) -> pd.DataFrame:
    rows = []
    for i in range(n_events):
        rows.append({"case id": f"E{i:04d}", "censorship": 0})
    for i in range(n_censored):
        rows.append({"case id": f"C{i:04d}", "censorship": 1})
    return pd.DataFrame(rows)


def _synthetic_parent_split(train_ids, val_ids) -> pd.DataFrame:
    n = max(len(train_ids), len(val_ids))
    train_col = list(train_ids) + [None] * (n - len(train_ids))
    val_col = list(val_ids) + [None] * (n - len(val_ids))
    return pd.DataFrame({"train": train_col, "val": val_col})


def test_build_outer_test_frame_stratifies_event_patients():
    clinical = _synthetic_clinical_df(n_events=50, n_censored=150)
    train_ids = [f"E{i:04d}" for i in range(50)] + [f"C{i:04d}" for i in range(150)]
    val_ids = [f"V{i:04d}" for i in range(50)]  # outer test (no clinical row needed)
    parent = _synthetic_parent_split(train_ids, val_ids)
    frame = build_outer_test_frame_with_clinical_df(
        fold_df=parent,
        clinical_df=clinical,
        inner_val_fraction=0.20,
        seed=3,
    )
    inner_train = set(frame[DEFAULT_INNER_TRAIN].dropna())
    inner_val = set(frame[DEFAULT_INNER_VAL].dropna())
    outer_test = set(frame[DEFAULT_OUTER_TEST].dropna())
    # No patient overlap between subsets.
    assert inner_train.isdisjoint(inner_val)
    assert inner_train.isdisjoint(outer_test)
    assert inner_val.isdisjoint(outer_test)
    # inner_train ∪ inner_val == parent train_ids (every parent train patient
    # is routed exactly once).
    assert inner_train | inner_val == set(train_ids)
    # outer_test matches the parent val_ids.
    assert outer_test == set(val_ids)
    # Sizes: 200 train → 160 inner_train + 40 inner_val; val 50.
    assert len(inner_train) == 160
    assert len(inner_val) == 40
    assert len(outer_test) == 50


def test_build_outer_test_frame_raises_when_clinical_lacks_train_id():
    clinical = _synthetic_clinical_df(n_events=2, n_censored=2)
    train_ids = ["E0000", "E0001", "MISSING"]
    val_ids = ["V0"]
    parent = _synthetic_parent_split(train_ids, val_ids)
    with pytest.raises(ValueError, match="missing"):
        build_outer_test_frame_with_clinical_df(
            fold_df=parent,
            clinical_df=clinical,
            inner_val_fraction=0.20,
            seed=3,
        )


# ---------------------------------------------------------------------------
# generate_outer_test_splits end-to-end (on disk).
# ---------------------------------------------------------------------------


def test_generate_outer_test_splits_writes_csvs_and_returns_fingerprints(tmp_path):
    data_path = tmp_path
    study = "blca"
    folds = [0, 1, 2]

    # Stage the parent split directory.
    parent_dir = data_path / "splits" / "5fold" / study
    parent_dir.mkdir(parents=True)

    # Stage the clinical CSV.
    clinical_dir = data_path / study
    clinical_dir.mkdir()
    clinical = _synthetic_clinical_df(n_events=20, n_censored=80)
    clinical.to_csv(clinical_dir / "clinical.csv", index=False)

    # Stage three parent folds.
    train_ids = [f"E{i:04d}" for i in range(20)] + [f"C{i:04d}" for i in range(80)]
    val_ids = [f"V{i:04d}" for i in range(20)]
    parent_df = _synthetic_parent_split(train_ids, val_ids)
    for fold in folds:
        parent_df.to_csv(parent_dir / f"fold_{fold}.csv", index=False)

    fingerprints = generate_outer_test_splits(
        data_path=str(data_path),
        which_splits="5fold",
        outer_which_splits="5fold_outer_test_seed3",
        studies=[study],
        folds=folds,
        inner_val_fraction=0.20,
        seed=3,
    )
    assert study in fingerprints
    assert set(fingerprints[study].keys()) == set(folds)

    # CSVs exist on disk and have the expected schema.
    for fold in folds:
        out_csv = data_path / "splits" / "5fold_outer_test_seed3" / study / f"fold_{fold}.csv"
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert set(df.columns) == {
            DEFAULT_INNER_TRAIN,
            DEFAULT_INNER_VAL,
            DEFAULT_OUTER_TEST,
        }


def test_generate_outer_test_splits_is_idempotent(tmp_path):
    """Re-running with the same seed produces identical fingerprints and
    identical on-disk CSV contents."""
    data_path = tmp_path
    study = "blca"
    folds = [0, 1]
    parent_dir = data_path / "splits" / "5fold" / study
    parent_dir.mkdir(parents=True)
    clinical_dir = data_path / study
    clinical_dir.mkdir()
    clinical = _synthetic_clinical_df(n_events=20, n_censored=80)
    clinical.to_csv(clinical_dir / "clinical.csv", index=False)
    train_ids = [f"E{i:04d}" for i in range(20)] + [f"C{i:04d}" for i in range(80)]
    val_ids = [f"V{i:04d}" for i in range(20)]
    parent_df = _synthetic_parent_split(train_ids, val_ids)
    for fold in folds:
        parent_df.to_csv(parent_dir / f"fold_{fold}.csv", index=False)

    args = dict(
        data_path=str(data_path),
        which_splits="5fold",
        outer_which_splits="5fold_outer_test_seed3",
        studies=[study],
        folds=folds,
        inner_val_fraction=0.20,
        seed=DEFAULT_SPLIT_SEED,
    )
    fingerprints_a = generate_outer_test_splits(**args)
    snapshots_a = {
        fold: (data_path / "splits" / "5fold_outer_test_seed3" / study / f"fold_{fold}.csv").read_bytes()
        for fold in folds
    }
    fingerprints_b = generate_outer_test_splits(**args)
    snapshots_b = {
        fold: (data_path / "splits" / "5fold_outer_test_seed3" / study / f"fold_{fold}.csv").read_bytes()
        for fold in folds
    }
    assert fingerprints_a == fingerprints_b
    for fold in folds:
        assert snapshots_a[fold] == snapshots_b[fold]


def test_generate_outer_test_splits_raises_when_parent_missing(tmp_path):
    data_path = tmp_path
    study = "blca"
    (data_path / "splits" / "5fold" / study).mkdir(parents=True)
    (data_path / study).mkdir()
    pd.DataFrame({"case id": ["E0"], "censorship": [0]}).to_csv(
        data_path / study / "clinical.csv", index=False
    )
    with pytest.raises(FileNotFoundError, match="parent split not found"):
        generate_outer_test_splits(
            data_path=str(data_path),
            which_splits="5fold",
            outer_which_splits="5fold_outer_test_seed3",
            studies=[study],
            folds=[0],
            inner_val_fraction=0.20,
            seed=3,
        )
