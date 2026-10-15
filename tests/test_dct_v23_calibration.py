"""Synthetic teacher, provenance and patient-ID checks; no patient jobs."""
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from survot_rank.research.methods.dct_v23.calibration import (
    global_gene_ids, training_reference, load_calibration, array_digest,
    PatientIdentifiedDataset, collate_identified_pathways, process_identified_batch,
)
from survot_rank.research.methods.dct_v23.crossfit import (
    balanced_global_masks, fit_cross_patient_gains, fixed_projection, save_gain_table,
)


PATHWAYS = [["a", "b"], ["b", "c", "d"]]


def synthetic_payload():
    rng = np.random.default_rng(9)
    wsi = rng.normal(size=(60, 3))
    raw = np.stack([3 * wsi[:, 0] + rng.normal(0, .01, 60), rng.normal(size=60),
                    2 * wsi[:, 1] + rng.normal(0, .01, 60), rng.normal(size=60)], axis=1)
    return fit_cross_patient_gains(raw, wsi, patient_ids=[f"p{i}" for i in range(60)],
        pathway_gene_ids=PATHWAYS, inner_folds=3, mask_ratio=.5, mask_rounds=6,
        native_dim=2, wsi_dim=3, ridge_alpha=.5, seed=7), raw


def write_payload(tmp_path, payload):
    path = tmp_path / "gain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_payload(path, payload):
    return load_calibration(path, patient_ids=payload["provenance"]["train_patient_ids"],
        excluded_ids=["val", "test"], pathway_gene_ids=PATHWAYS,
        raw_expression_sha256=payload["provenance"]["raw_expression_sha256"])


def test_global_coordinates_keep_shared_genes_once():
    assert global_gene_ids(PATHWAYS) == ["a", "b", "c", "d"]


def test_masks_cover_every_global_gene_and_leave_visible_values():
    masks = balanced_global_masks(15, .3, 8, 3)
    assert masks.shape == (8, 15)
    assert masks.any(0).all() and (~masks).any(1).all()
    assert np.array_equal(masks, balanced_global_masks(15, .3, 8, 3))


@pytest.mark.parametrize("ratio,rounds", [(0, 8), (1, 8), (.1, 1)])
def test_invalid_or_incomplete_masks_fail(ratio, rounds):
    with pytest.raises(ValueError):
        balanced_global_masks(15, ratio, rounds, 3)


def test_projection_uses_fixed_gene_identity():
    first = fixed_projection(["a", "b"], 4, 3)
    permuted = fixed_projection(["b", "a"], 4, 3)
    assert np.array_equal(first, permuted[::-1])


def test_teachers_never_fit_their_scoring_patients():
    payload, raw = synthetic_payload()
    seen = []
    ids = payload["provenance"]["train_patient_ids"]
    for teacher in payload["provenance"]["teachers"]:
        fit, query = teacher["fit_patient_ids"], teacher["score_patient_ids"]
        assert not set(fit) & set(query)
        assert set(fit) | set(query) == set(ids)
        assert teacher["normalization_fit_patient_ids"] == fit
        expected = raw[[ids.index(x) for x in fit]].mean(0)
        assert np.allclose(teacher["gene_mean"], expected)
        seen.extend(query)
    assert len(seen) == len(set(seen)) == len(ids)
    assert payload["provenance"]["no_survival_labels"] is True


def test_conditional_teacher_recovers_real_heldout_expression():
    payload, _ = synthetic_payload()
    gain = np.asarray(list(payload["gains"].values()))
    assert gain.mean() > .2
    assert all(min(count) > 0 for count in payload["scored_gene_counts"].values())
    for patient in payload["gains"]:
        assert np.allclose(payload["gains"][patient],
            np.subtract(payload["native_mse"][patient], payload["conditional_mse"][patient]))


def test_query_hidden_values_never_enter_teacher_feature_projection(monkeypatch):
    from survot_rank.research.methods.dct_v23 import crossfit
    rng = np.random.default_rng(18)
    raw, wsi = rng.normal(size=(12, 4)), rng.normal(size=(12, 3))
    calls = []
    real_predict = crossfit._ridge_predict
    def capture(fit_x, fit_y, score_x, alpha):
        calls.append((fit_x.copy(), score_x.copy()))
        return real_predict(fit_x, fit_y, score_x, alpha)
    monkeypatch.setattr(crossfit, "_ridge_predict", capture)
    settings = dict(patient_ids=[f"p{i}" for i in range(12)], pathway_gene_ids=PATHWAYS,
        inner_folds=3, mask_ratio=.5, mask_rounds=2, native_dim=2, wsi_dim=2, seed=3)
    payload = crossfit.fit_cross_patient_gains(raw, wsi, **settings)
    original_native, original_conditional = calls[0], calls[1]
    patient = int(payload["provenance"]["teachers"][0]["score_patient_ids"][0][1:])
    gene = np.flatnonzero(payload["provenance"]["global_masks"][0])[0]
    changed = raw.copy()
    changed[patient, gene] += 1e6
    calls.clear()
    crossfit.fit_cross_patient_gains(changed, wsi, **settings)
    for original, altered in ((original_native, calls[0]), (original_conditional, calls[1])):
        assert np.array_equal(original[0], altered[0])
        assert np.array_equal(original[1], altered[1])


def test_signed_negative_gains_are_not_clamped(tmp_path):
    payload, _ = synthetic_payload()
    payload["gains"]["p0"][0] = -.75
    payload["native_mse"]["p0"][0] = .25
    payload["conditional_mse"]["p0"][0] = 1.
    gains, _ = load_payload(write_payload(tmp_path, payload), payload)
    assert gains["p0"][0] == -.75


def test_exclusive_save_does_not_replace_artifacts(tmp_path):
    payload, _ = synthetic_payload()
    path = save_gain_table(payload, tmp_path / "gain.json")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_gain_table({}, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("defect", ["in_sample", "wrong_norm", "extra_patient", "missing_patient", "nan_gain", "wrong_gene", "flag", "same_query_twice"])
def test_invalid_provenance_is_rejected(tmp_path, defect):
    payload, _ = synthetic_payload()
    p = payload["provenance"]
    if defect == "in_sample":
        p["teachers"][0]["fit_patient_ids"].append(p["teachers"][0]["score_patient_ids"][0])
    elif defect == "wrong_norm":
        p["teachers"][0]["normalization_fit_patient_ids"].append("val")
    elif defect == "extra_patient":
        payload["gains"]["val"] = [0., 0.]
    elif defect == "missing_patient":
        payload["gains"].pop("p0")
    elif defect == "nan_gain":
        payload["gains"]["p0"] = [float("nan"), 0]
    elif defect == "wrong_gene":
        p["pathway_gene_ids"][0] = ["unknown", "b"]
    elif defect == "flag":
        p["held_out_gain"] = False
    else:
        p["teachers"].append(p["teachers"][0])
    with pytest.raises(ValueError):
        load_payload(write_payload(tmp_path, payload), payload)


def test_stale_raw_expression_source_is_rejected(tmp_path):
    payload, _ = synthetic_payload()
    path = write_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="RNA source"):
        load_calibration(path, patient_ids=payload["provenance"]["train_patient_ids"], excluded_ids=[],
            pathway_gene_ids=PATHWAYS, raw_expression_sha256="stale")


def test_teacher_statistics_are_verified_against_actual_fit_patients(tmp_path):
    payload, raw = synthetic_payload()
    payload["provenance"]["teachers"][0]["gene_mean"][0] += 1.
    path = write_payload(tmp_path, payload)
    with pytest.raises(ValueError, match="does not match"):
        load_calibration(path, patient_ids=payload["provenance"]["train_patient_ids"], excluded_ids=[],
            pathway_gene_ids=PATHWAYS, raw_expression_sha256=array_digest(raw), raw_expression=raw)


def test_gain_cannot_be_modified_independently_of_scored_errors(tmp_path):
    payload, _ = synthetic_payload()
    payload["gains"]["p0"][0] += 1.
    with pytest.raises(ValueError, match="native-minus-conditional"):
        load_payload(write_payload(tmp_path, payload), payload)


def test_outer_train_normalizer_excludes_evaluation_outlier():
    factory = SimpleNamespace(omic_names=PATHWAYS,
        gene_data_df=pd.DataFrame({"a0": [0., 2., 4., 6.], "a1": [2., 4., 6., 8.],
                                  "val": [1e6] * 4}, index=["a", "b", "c", "d"]))
    ref = training_reference(factory, ["a0", "a1"])
    assert np.allclose(ref["raw_gene_mean"], [1, 3, 5, 7])
    assert np.allclose(ref["raw_gene_std"], 1.)


class SyntheticDataset:
    def __init__(self):
        self.label_df = pd.DataFrame({"case id": ["p0", "p1", "p2"]})
        self.split_key = "train"

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 1:
            return None
        return (torch.ones(2, 3) * index, [torch.tensor([float(index), 1.]), torch.ones(3)],
                torch.tensor(0), torch.tensor(2.), torch.tensor(0.))


def test_ids_survive_shuffled_sampling_and_corrupt_sample_drop():
    dataset = PatientIdentifiedDataset(SyntheticDataset())
    batch = collate_identified_pathways([dataset[i] for i in (2, 1, 0)])
    assert batch["patient_ids"] == ["p2", "p0"]
    assert batch["data"][0][0, 0, 0] == 2
    assert collate_identified_pathways([None]) is None


def test_forward_uses_ids_only_for_training_auxiliary():
    dataset = PatientIdentifiedDataset(SyntheticDataset())
    batch = collate_identified_pathways([dataset[2], dataset[0]])
    received = []
    def model(**kwargs):
        received.append(kwargs)
        return torch.zeros(2, 4), torch.tensor(0.)
    args = SimpleNamespace(cur_epoch=4)
    process_identified_batch(args, model, batch, "cpu")
    assert received[-1]["patient_ids"] == ["p2", "p0"]
    assert received[-1]["x_omic1"].shape == (2, 2)
    process_identified_batch(args, model, batch, "cpu", test=True)
    assert "patient_ids" not in received[-1]
    assert received[-1]["y"] is None and received[-1]["c"] is None


def test_legacy_payload_cannot_guess_training_ids():
    with pytest.raises(ValueError, match="explicit patient IDs"):
        process_identified_batch(SimpleNamespace(), lambda **kw: None, [torch.zeros(1)], "cpu")


def test_plain_mode_requires_no_gain_table_and_fits_only_training_stats(tmp_path):
    from survot_rank.training import train_runner as runner
    factory = SimpleNamespace(omic_names=PATHWAYS, gene_data_df=pd.DataFrame({
        "p0": [0., 1., 2., 3.], "p1": [2., 3., 4., 5.], "val": [1e6] * 4}, index=["a", "b", "c", "d"]))
    data = SimpleNamespace(label_df=pd.DataFrame({"case id": ["p0", "p1"]}), fold=0)
    args = SimpleNamespace(survot_method="dct_v23", dct_v23_reconstruction_mode="masked_plain")
    split = pd.DataFrame({"train": ["p0", "p1"], "val": ["val", None]})
    runner._configure_v23_training_data(args, factory, data, split, tmp_path / "no_read.csv")
    assert args._dct_v23_gain_payload is None
    assert np.allclose(args.dct_v23_raw_gene_mean, [1, 2, 3, 4])


def test_calibrated_mode_requires_prepared_gain_artifact(tmp_path):
    from survot_rank.training import train_runner as runner
    factory = SimpleNamespace(omic_names=PATHWAYS, gene_data_df=pd.DataFrame({
        "p0": [0., 1., 2., 3.], "p1": [2., 3., 4., 5.]}, index=["a", "b", "c", "d"]))
    data = SimpleNamespace(label_df=pd.DataFrame({"case id": ["p0", "p1"]}), fold=0)
    args = SimpleNamespace(dct_v23_reconstruction_mode="gain_calibrated")
    split = pd.DataFrame({"train": ["p0", "p1"], "val": ["val", None]})
    with pytest.raises(ValueError, match="prepared calibration"):
        runner._configure_v23_training_data(args, factory, data, split, tmp_path / "no_read.csv")


def test_training_snapshot_preserves_original_random_patch_policy():
    from survot_rank.training import train_runner as runner
    wrapper = PatientIdentifiedDataset(SyntheticDataset())
    loader = runner._build_v22_train_evaluation_loader(SimpleNamespace(
        survot_method="dct_v23", rna_format="Pathways", num_workers=0, seed=3), wrapper)
    assert wrapper.dataset.split_key == "train"
    assert loader.dataset.dataset.split_key == "val"
    assert next(iter(loader))["patient_ids"] == ["p0"]


def test_complete_preparation_launcher_reads_only_declared_training_patients(monkeypatch, tmp_path):
    from scripts.prepare_dct_v23_calibration import prepare_one_fold
    from survot_rank.research.legacy.slotspe_runtime.dataset import dataset_survival as source
    ids = [f"p{i}" for i in range(8)]
    rng = np.random.default_rng(19)
    genes = pd.DataFrame(rng.normal(size=(4, 8)), index=["a", "b", "c", "d"], columns=ids)
    data_dir = tmp_path / "data"
    splits = data_dir / "splits" / "synthetic" / "blca"
    splits.mkdir(parents=True)
    pd.DataFrame({"train": ids, "val": ["v0", "v1"] + [None] * 6}).to_csv(splits / "fold_0.csv", index=False)
    feature_path = tmp_path / "synthetic_features.pt"
    feature_path.write_bytes(b"synthetic fixture, not patient data")
    loaded = []
    class Factory:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.omic_names, self.pathway_names = PATHWAYS, ["path0", "path1"]
            self.gene_data_df = genes
            self.clinical_df = pd.DataFrame({"case id": ids + ["v0", "v1"], "wsi": ids + ["v0", "v1"]})
    class Dataset:
        def __init__(self, *a, **kw):
            pass
        def load_wsi(self, patient):
            assert patient in ids  # val/test features must never be queried.
            loaded.append(patient)
            return torch.tensor([[float(ids.index(patient)), 1., 2.], [1., 2., 3.]])
        def _resolve_wsi_feature_path(self, slide):
            return feature_path
    monkeypatch.setattr(source, "SurvivalDatasetFactory", Factory)
    monkeypatch.setattr(source, "SurvivalDataset", Dataset)
    config = dict(data=dict(study="blca", data_path=str(data_dir), data_root_dir=str(tmp_path),
        rna_format="Pathways", signature="combine", label_col="survival_months_dss", encoding_dim=3),
        train=dict(evaluation_protocol="legacy_val"), model=dict(wsi_encoder="synthetic"),
        split_setup=dict(which_splits="synthetic"))
    path = prepare_one_fold(config, 0, tmp_path / "output", dict(inner_folds=2, mask_ratio=.5,
        mask_rounds=2, native_dim=2, wsi_dim=2, ridge_alpha=1., seed=3))
    assert set(loaded) == set(ids) and len(loaded) == len(ids)
    payload = json.loads(path.read_text(encoding="utf-8"))
    load_calibration(path, patient_ids=ids, excluded_ids=["v0", "v1"], pathway_gene_ids=PATHWAYS,
        raw_expression_sha256=array_digest(genes.to_numpy().T), raw_expression=genes.to_numpy().T,
        encoding_dim=3, wsi_encoder="synthetic")
    assert payload["provenance"]["excluded_patient_ids"] == ["v0", "v1"]


def test_generated_crossfit_provenance_binds_to_real_candidate_model():
    from survot_rank.research.methods.dct_v23 import DCTV23, DCTV23Config
    payload, raw = synthetic_payload()
    cfg = DCTV23Config(omic_sizes=(2, 3), dim=8, wsi_input_dim=3, wsi_hidden_dim=4,
        num_classes=3, wsi_slots=2, omics_slots=2, stages=2, heads=2,
        event_layers=1, slot_iters=1, slot_heads=2, slot_dim_head=4,
        slot_hidden_dim=16, recon_hidden_dim=5, reconstruction_mode="gain_calibrated")
    model = DCTV23(cfg)
    model.configure_reconstruction_reference(PATHWAYS, raw.mean(0), raw.std(0),
        payload["provenance"]["train_patient_ids"])
    model.bind_gain_targets(payload["gains"], payload["provenance"])
    assert model._gain_targets["p0"] == pytest.approx(payload["gains"]["p0"])


def test_real_candidate_training_and_evaluation_use_identified_batches(monkeypatch, tmp_path):
    from survot_rank.training import train_runner as runner
    from survot_rank.research.methods.dct_v23 import DCTV23, DCTV23Config
    payload, raw = synthetic_payload()
    cfg = DCTV23Config(omic_sizes=(2, 3), dim=8, wsi_input_dim=3, wsi_hidden_dim=4,
        num_classes=3, wsi_slots=2, omics_slots=2, stages=2, heads=2, event_layers=1,
        slot_iters=1, slot_heads=2, slot_dim_head=4, slot_hidden_dim=16, recon_hidden_dim=5,
        dropout=0., wsi_dropout=0., ipcw_weight=0., relation_weight=0.,
        reconstruction_mode="gain_calibrated")
    model = DCTV23(cfg)
    model.configure_reconstruction_reference(PATHWAYS, raw.mean(0), raw.std(0),
        payload["provenance"]["train_patient_ids"])
    model.bind_gain_targets(payload["gains"], payload["provenance"])
    model.configure_train_reference(np.arange(1, len(raw) + 1, dtype=float), np.zeros(len(raw)))
    class Dataset(SyntheticDataset):
        def __getitem__(self, index):
            if index == 1:
                return None
            genes = torch.tensor(raw[index], dtype=torch.float32)
            return (torch.full((5, 3), float(index + 1)), [genes[:2], genes[1:]],
                    torch.tensor(index % 3), torch.tensor(float(index + 1)), torch.tensor(0.))
    wrapper = PatientIdentifiedDataset(Dataset())
    args = SimpleNamespace(survot_method="dct_v23", rna_format="Pathways", cur_fold=0,
        cur_epoch=5, max_epochs=1, batch_size=2, max_smoke_batches=0, grad_accum_steps=1,
        grad_clip_norm=0., bag_loss="nll_surv", alpha_surv=0., results_dir=str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "_calculate_metrics", lambda *a, **kw: (.7, .7, .2, .2, .7))
    train_loader = torch.utils.data.DataLoader(wrapper, batch_size=2, collate_fn=collate_identified_pathways)
    loss_fn = runner.init_loss_function(args)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        diagnostics = runner.train_one_epoch(args, 5, model, train_loader,
            torch.optim.SGD(model.parameters(), lr=0.), SimpleNamespace(step=lambda: None), loss_fn, io.StringIO())
        assert diagnostics["optimization_patient_count"] == 2
        assert np.isfinite(diagnostics["optimization_main_nll"])
        assert np.isfinite(diagnostics["optimization_auxiliary_total"])
        # Evaluation succeeds after clearing all gain targets. No patient-ID
        # cache is required to perform held-out survival inference.
        model._gain_targets.clear()
        evaluate_loader = torch.utils.data.DataLoader(wrapper, batch_size=1, collate_fn=collate_identified_pathways)
        result = runner.evaluate(args, SimpleNamespace(), model, evaluate_loader, loss_fn)
        assert set(result[0]) == {"p0", "p2"}
        assert model.last_evaluation_monitoring_metrics["patient_count"] == 2
        assert np.isfinite(result[-1])
    finally:
        torch.set_num_threads(original_threads)
