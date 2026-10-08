"""Synthetic scientific-contract tests, not real-data performance evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from survot_rank.evidence import additional as a
from survot_rank.evidence.additional_plots import make_additional_figures, read_exports
from test_dct_v313_transport_reconstruction import batch, make_args, train_reference
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import DCTV313TransportReconstruction


@pytest.fixture
def model(monkeypatch):
    import survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model as module
    monkeypatch.setattr(module, "_debug_log", lambda **kwargs: None)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(44)
    model = DCTV313TransportReconstruction(make_args()).eval()
    model.configure_train_reference(*train_reference())
    yield model
    torch.set_num_threads(previous_threads)


def payload():
    original = batch()
    result = {k: v[:1] for k, v in original.items()}
    result.update(y=None, c=None, event_time=None, wsi_missing=False, omic_missing=False)
    return result


def test_donor_controls_are_bijections_without_self_pairs():
    for n in (2, 3, 17):
        ids = [f"p{i}" for i in range(n)]
        donors = a.donor_indices(ids, 123, 0)
        assert sorted(donors) == list(range(n))
        assert np.all(donors != np.arange(n))
        assert np.array_equal(donors, a.donor_indices(ids, 123, 0))
    with pytest.raises(ValueError):
        a.donor_indices(["only"], 123, 0)


def test_retrieval_detects_real_matches_and_decoder_prior_collapse():
    target = np.eye(4).reshape(4, 1, 4)
    mean = np.zeros((1, 4))
    exact = a.retrieval_metrics(target, target, mean)
    assert exact["top1"] == exact["mrr"] == 1
    wrong = a.retrieval_metrics(np.roll(target, 1, axis=0), target, mean)
    assert wrong["top1"] == 0 and np.all(wrong["ranks"] > 1)
    constant = a.retrieval_metrics(np.zeros_like(target), target, mean)
    assert constant["top1"] == 0 and constant["mrr"] == .25
    assert constant["zero_query_count"] == 4


def test_patch_sampling_preserves_ids_and_uses_common_nested_subsets():
    valid = np.asarray([1, 1, 1, 1, 0, 0], dtype=bool)
    scores = np.asarray([.2, .8, .1, .4, 99, 99])
    top, bottom, random = a.patch_orders(scores, valid, 3, "patient", 0)
    assert top.tolist() == [1, 3, 0, 2] and bottom.tolist() == [2, 0, 3, 1]
    assert not set(random) & {4, 5}
    assert np.array_equal(random, a.patch_orders(scores, valid, 3, "patient", 0)[2])
    assert set(a.keep_rows(random, .5, deletion=False)) <= set(a.keep_rows(random, .25, deletion=False))
    assert len(a.keep_rows(random, .99, deletion=True)) == 1
    with pytest.raises(ValueError):
        a.validate_options("all", (0, 1), 2)


def test_actual_model_replay_controls_and_patch_recomputation(model):
    p = payload()
    before = {k: v.clone() for k, v in model.named_buffers()}
    sw, so, target = a.encode_patient(model, p)
    with torch.no_grad():
        standard = float(model._risk(model(**p)[0]).item())
    assert np.isclose(a.risk_for_rows(model, p, np.arange(6)), standard, atol=1e-6)
    subset = np.asarray([1, 3, 4])
    with torch.no_grad():
        subset_standard = float(model._risk(model(**dict(p, x_wsi=p["x_wsi"][:, subset]))[0]).item())
    assert np.isclose(a.risk_for_rows(model, p, subset), subset_standard, atol=1e-6)
    errors, decoded, shuffled_risk = a.decode_controls(model, sw, so, target, sw.flip(1), target[0])
    assert np.isfinite(shuffled_risk)
    plans, _, gate = a.solve(model, sw, so)
    _, native_error, _, _ = model.reconstruction_losses(x_omics=target, slots_wsi=sw, slots_omic=so,
        factual_plans=plans, factual_gate=gate, available=torch.ones(1, dtype=torch.bool))
    assert np.isclose(errors["native"].mean(), float(native_error.detach()), atol=1e-6)
    assert decoded["native"].shape == (5, 16)
    model._cross_mode = "direct"
    direct_errors, _, _ = a.decode_controls(model, sw, so, target, sw * 1.5, target[0])
    with torch.no_grad():
        expected_direct = a.pathway_error(model.pathway_reconstruction_decoder(sw * 1.5), target).numpy()[0]
    assert np.allclose(direct_errors["shuffled"], expected_direct, atol=1e-6)
    model._cross_mode = "transport"
    result = a.patch_experiment(model, p, np.ones((3, 6)) / 6, np.ones(6, dtype=bool), "synthetic",
                                (0, .5), 2, 123)
    assert np.allclose(result["deletion_risk"][0], standard, atol=1e-6)
    assert np.allclose(result["budget_risk"][0], standard, atol=1e-6)
    assert np.isfinite(result["budget_risk"]).all()
    assert np.array_equal(result["deletion_risk"][:, 0, :2], result["deletion_risk"][:, 1, :2])
    a._assert_buffers(model, before)
    for key in ("y", "c", "event_time"):
        with pytest.raises(ValueError, match="survival-label"):
            a.encode_patient(model, dict(p, **{key: torch.tensor([0])}))


def synthetic_export(tmp_path, monkeypatch, model):
    source = batch()
    samples = []
    for i in range(3):
        samples.append({k: v[i:i+1] for k, v in source.items() if k not in ("y", "c")})
        samples[-1].update(y=None, c=None, wsi_missing=False, omic_missing=False)
    ids = [f"synthetic{i}" for i in range(3)]
    dataset = SimpleNamespace(label_df=pd.DataFrame({"case id": ids}), split_key="val")
    class Training:
        split_key = "train"
        def __len__(self):
            return 3
    feature = tmp_path / "synthetic_feature.bin"
    feature.write_bytes(b"synthetic unit-test feature identity")
    def sample(ds, i):
        p = samples[i]
        # Mirror the real sample tuple shape; no real dataset is loaded.
        item = (p, None, None, torch.tensor(float(i + 1)), torch.tensor(0.))
        return item, [dict(feature_path=str(feature), patch_count=6)], np.zeros(6, dtype=int), np.arange(6)
    monkeypatch.setattr(a, "indexed_sample", sample)
    monkeypatch.setattr(a, "payload_for", lambda item, args, device: item[0])
    monkeypatch.setattr(a, "load_run", lambda run, device: (model, model.args,
        SimpleNamespace(pathway_names=[f"path{i}" for i in range(5)]), Training(), dataset))
    monkeypatch.setattr(a, "audit", lambda m: dict(passed=True, runs=[dict(hashes={})]))
    with torch.no_grad():
        risks = [float(model._risk(model(**p)[0]).item()) for p in samples]
    monkeypatch.setattr(a, "predictions", lambda path: dict(case_ids=ids, risk=risks, time=[1, 2, 3], censor=[0, 0, 0]))
    run = dict(id="synthetic_only", arm="exp6", cancer="blca", fold=0, seed=3,
               protocol="legacy_val", checkpoint="synthetic", predictions="synthetic", source_commit="synthetic")
    summary = a.export_additional(run, tmp_path / "exports", device="cpu", repeats=2, fractions=(0, .5))
    return summary, tmp_path / "exports"


def test_synthetic_full_pipeline_and_corrupt_export_rejection(tmp_path, monkeypatch, model):
    summary, exports = synthetic_export(tmp_path, monkeypatch, model)
    assert summary["training_reference_patient_count"] == 3
    assert summary["reconstruction"]["train_mean"]["zero_query_count"] == 3
    assert len(read_exports(exports)) == 1
    index = make_additional_figures(exports, tmp_path / "figures")
    assert len(index) == 5
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 5
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
    with (tmp_path / "figures/fold_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    recon = next(r for r in rows if r["experiment"] == "reconstruction" and r["condition"] == "native")
    paired = next(r for r in rows if r["experiment"] == "pairing" and r["condition"] == "native")
    assert recon["metric"] == "normalized_latent_error" and recon["direction"] == "lower"
    assert float(recon["value"]) == summary["reconstruction"]["native"]["mean_error"]
    assert paired["metric"] == "cindex" and paired["direction"] == "higher"
    assert float(paired["value"]) == summary["pairing"]["native_cindex"]
    assert any(r["experiment"] == "budget" and r["metric"] == "cindex" for r in rows)
    with np.load(exports / "synthetic_only" / "additional.npz") as arrays:
        assert arrays["reconstruction_shuffled_all_error"].shape == (3, 2, 5)
    with pytest.raises(ValueError, match="fresh"):
        make_additional_figures(exports, tmp_path / "figures")
    meta_path = exports / "synthetic_only" / "additional.json"
    original = meta_path.read_text(encoding="utf-8")
    altered = json.loads(original)
    altered["patches"]["unpadded_cindex"] = -9
    meta_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="summary disagrees"):
        read_exports(exports)
    meta_path.write_text(original, encoding="utf-8")
    arrays = exports / "synthetic_only" / "additional.npz"
    arrays.write_bytes(arrays.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_exports(exports)


def test_cli_default_does_not_load_checkpoint_or_create_outputs(tmp_path, capsys):
    from scripts.run_v313_additional_evidence import main
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, runs=[dict(id="p0", arm="exp6", cancer="blca", fold=0)])))
    output = tmp_path / "never_created"
    main(["run", "--manifest", str(manifest), "--output", str(output)])
    assert "no checkpoint/data loaded" in capsys.readouterr().out
    assert not output.exists()


def test_training_mean_retrieval_control_keeps_the_arithmetic_mean(model):
    p = payload()
    sw, so, target = a.encode_patient(model, p)
    # A real arithmetic mean of normalized tokens need not have unit variance.
    center = torch.nn.functional.layer_norm(target[0], (target.size(-1),)) * .25
    _, decoded, _ = a.decode_controls(model, sw, so, target, sw, center)
    assert np.allclose(decoded["train_mean"], center.cpu().numpy(), atol=1e-7)


def test_export_retrieval_center_is_the_unmodified_training_mean(tmp_path, monkeypatch, model):
    original_mean, original_metric = a.training_mean, a.retrieval_metrics
    reference = {}
    def capture_mean(*args, **kwargs):
        value = original_mean(*args, **kwargs)
        reference["mean"] = value.cpu().numpy()
        return value
    def verify_metric(decoded, targets, center):
        assert np.allclose(center, reference["mean"], atol=1e-7)
        return original_metric(decoded, targets, center)
    monkeypatch.setattr(a, "training_mean", capture_mean)
    monkeypatch.setattr(a, "retrieval_metrics", verify_metric)
    synthetic_export(tmp_path, monkeypatch, model)
