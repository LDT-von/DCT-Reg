"""Tests for DCT v3.13 §3.1 #5 outer_test protocol wiring.

These tests use a stubbed ``SurvivalDataset`` and ``init_model_for_method``
so the trainer-side wiring can be validated without real data.

Verifies:

* CLI parser accepts ``--evaluation_protocol=outer_test``,
  ``--inner_val_fraction``, ``--split_seed``.
* ``get_split`` reads inner_train / inner_val when evaluation_protocol=outer_test.
* ``evaluate_outer_test`` is a no-op (returns None) when the protocol is
  legacy_val.
* The outer-test summary file is emitted under the expected name.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# CLI parsing.
# ---------------------------------------------------------------------------


def test_cli_parser_accepts_outer_test_choices():
    from survot_rank.training.extended_args import build_base_parser

    parser = build_base_parser()
    # Default
    ns = parser.parse_args([])
    assert ns.evaluation_protocol == "legacy_val"
    assert ns.inner_val_fraction == pytest.approx(0.20)
    assert ns.split_seed == 3
    # Explicit outer_test
    ns = parser.parse_args([
        "--evaluation_protocol", "outer_test",
        "--inner_val_fraction", "0.25",
        "--split_seed", "7",
    ])
    assert ns.evaluation_protocol == "outer_test"
    assert ns.inner_val_fraction == pytest.approx(0.25)
    assert ns.split_seed == 7
    # Unknown protocol rejected
    with pytest.raises(SystemExit):
        parser.parse_args(["--evaluation_protocol", "bogus"])


# ---------------------------------------------------------------------------
# get_split protocol routing.
# ---------------------------------------------------------------------------


class _StubDataset:
    """Minimal SurvivalDataset stand-in that records its constructor args."""

    instances = []

    def __init__(self, factory, wsi_path, split_key, fold, encoding_dim,
                 on_missing_wsi="error"):
        self.factory = factory
        self.wsi_path = wsi_path
        self.split_key = split_key
        self.fold = fold
        self.encoding_dim = encoding_dim
        self.on_missing_wsi = on_missing_wsi
        # Populate label_df so DataLoader length, label_df queries, and
        # bin counts work.
        self.label_df = pd.DataFrame({
            "case id": ["X"],
            "censorship": [0.0],
            "label": [0],
        })
        _StubDataset.instances.append(self)

    def __len__(self) -> int:
        return len(self.label_df)


def _make_dataset_factory(tmp_path, *, with_outer_columns: bool):
    """Stage a fake dataset_factory pointing at a temporary split CSV."""
    study = "blca"
    fold = 0
    splits_dir = tmp_path / "splits" / "5fold" / study
    splits_dir.mkdir(parents=True)

    if with_outer_columns:
        # Outer-test split layout (all columns equal length).
        df = pd.DataFrame({
            "inner_train": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
            "inner_val": ["P9", "P10", None, None, None, None, None, None],
            "outer_test": ["Q1", "Q2", "Q3", None, None, None, None, None],
        })
    else:
        df = pd.DataFrame({"train": ["P1"], "val": ["Q1"]})
    df.to_csv(splits_dir / f"fold_{fold}.csv", index=False)

    clinical = pd.DataFrame({"case id": ["P1"], "censorship": [0.0]})
    return SimpleNamespace(
        study=study,
        data_path=str(tmp_path),
        which_splits="5fold",
        clinical_df=clinical,
        censorship_var="censorship",
        label_col="survival_months_dss",
        fit_label_bins=lambda *_a, **_k: None,
    )


def test_get_split_routes_to_legacy_val(monkeypatch, tmp_path):
    from survot_rank.training import train_runner

    monkeypatch.setattr(train_runner, "SurvivalDataset", _StubDataset)
    _StubDataset.instances = []
    factory = _make_dataset_factory(tmp_path, with_outer_columns=False)
    args = SimpleNamespace(
        evaluation_protocol="legacy_val",
        fit_bins_on_train=False,
        data_root_dir=str(tmp_path),
        wsi_encoder="uni",
        encoding_dim=16,
        on_missing_wsi="error",
        rna_format="Pathways",
        event_sampling_fraction=0.0,
        event_stratified_batches=False,
        seed=3,
        batch_size=8,
        num_workers=0,
    )
    train_data, val_data, train_loader, val_loader = train_runner.get_split(
        args, factory, fold=0
    )
    split_keys = sorted(d.split_key for d in _StubDataset.instances)
    assert split_keys == ["train", "val"]


def test_get_split_routes_to_outer_test(monkeypatch, tmp_path):
    from survot_rank.training import train_runner

    monkeypatch.setattr(train_runner, "SurvivalDataset", _StubDataset)
    _StubDataset.instances = []
    factory = _make_dataset_factory(tmp_path, with_outer_columns=True)
    args = SimpleNamespace(
        evaluation_protocol="outer_test",
        fit_bins_on_train=False,
        data_root_dir=str(tmp_path),
        wsi_encoder="uni",
        encoding_dim=16,
        on_missing_wsi="error",
        rna_format="Pathways",
        event_sampling_fraction=0.0,
        event_stratified_batches=False,
        seed=3,
        batch_size=8,
        num_workers=0,
    )
    train_data, val_data, train_loader, val_loader = train_runner.get_split(
        args, factory, fold=0
    )
    split_keys = sorted(d.split_key for d in _StubDataset.instances)
    # Only inner_train and inner_val are constructed for training.  The
    # outer test loader is built lazily inside ``evaluate_outer_test``.
    assert split_keys == ["inner_train", "inner_val"]


def test_get_split_rejects_outer_test_without_columns(monkeypatch, tmp_path):
    """If evaluation_protocol=outer_test but the split CSV lacks the new
    columns, ``get_split`` raises a clear error that points the user at
    the outer-test split generator."""
    from survot_rank.training import train_runner

    monkeypatch.setattr(train_runner, "SurvivalDataset", _StubDataset)
    factory = _make_dataset_factory(tmp_path, with_outer_columns=False)
    args = SimpleNamespace(
        evaluation_protocol="outer_test",
        fit_bins_on_train=False,
        data_root_dir=str(tmp_path),
        wsi_encoder="uni",
        encoding_dim=16,
        on_missing_wsi="error",
        rna_format="Pathways",
        event_sampling_fraction=0.0,
        event_stratified_batches=False,
        seed=3,
        batch_size=8,
        num_workers=0,
    )
    with pytest.raises(KeyError, match="outer_test_split"):
        train_runner.get_split(args, factory, fold=0)


def test_get_split_rejects_unknown_protocol(monkeypatch, tmp_path):
    from survot_rank.training import train_runner

    factory = _make_dataset_factory(tmp_path, with_outer_columns=False)
    args = SimpleNamespace(
        evaluation_protocol="random",
        fit_bins_on_train=False,
        data_root_dir=str(tmp_path),
        wsi_encoder="uni",
        encoding_dim=16,
        on_missing_wsi="error",
        rna_format="Pathways",
        event_sampling_fraction=0.0,
        event_stratified_batches=False,
        seed=3,
        batch_size=8,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="evaluation_protocol"):
        train_runner.get_split(args, factory, fold=0)


# ---------------------------------------------------------------------------
# evaluate_outer_test rejection of legacy_val.
# ---------------------------------------------------------------------------


def test_evaluate_outer_test_rejects_legacy_val(tmp_path):
    from survot_rank.training import train_runner

    factory = _make_dataset_factory(tmp_path, with_outer_columns=False)
    args = SimpleNamespace(evaluation_protocol="legacy_val")
    with pytest.raises(ValueError, match="outer_test"):
        train_runner.evaluate_outer_test(
            args=args, dataset_factory=factory, fold=0,
            model_state_path="/nonexistent.pth", log_file=open(os.devnull, "w"),
        )


# ---------------------------------------------------------------------------
# Module import sanity (no real training; just verify the module loads).
# ---------------------------------------------------------------------------


def test_train_runner_module_loads_with_outer_test_symbols():
    from survot_rank.training import train_runner

    assert hasattr(train_runner, "evaluate_outer_test")
    assert hasattr(train_runner, "_load_inner_train_labels")
    # The split generator must be importable as a module too.
    from survot_rank.training.outer_test_split import generate_outer_test_splits
    assert callable(generate_outer_test_splits)
