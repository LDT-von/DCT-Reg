"""Synthetic contract tests; no real-data training, extraction or inference."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from survot_rank.evidence.manifest import audit, best_epoch, cindex, load_manifest, save_json
from survot_rank.evidence.v313 import independent_plan, indexed_sample, load_checkpoint_strict, mix_plans, pathway_error, replay
from survot_rank.evidence.plots import clinical_figures, km_curve, make_figures, spatial_figures
from test_dct_v313_transport_reconstruction import make_args, batch, train_reference
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import DCTV313TransportReconstruction


def artifact(tmp_path, fold=0, arm="exp6"):
    directory = tmp_path / f"{arm}_{fold}"
    directory.mkdir()
    ids = [f"patient_{fold}_{j}" for j in range(3)]
    records = {cid: dict(risk=3 - j + (0 if arm == "exp6" else .1), time=float(j + 1), censor=0.) for j, cid in enumerate(ids)}
    pred = directory / "pred.pkl"
    pred.write_bytes(pickle.dumps(records))
    curve = directory / "curve.csv"
    curve.write_text(f"epoch,val_cindex\n0,{.1 + fold * .01 + (0 if arm == 'exp6' else .001)}\n1,1.0\n2,1.0\n", encoding="utf-8")
    split = tmp_path / f"split_{fold}.csv"
    train_ids = [f"patient_{f}_{j}" for f in range(5) if f != fold for j in range(3)]
    split.write_text("train,val\n" + "\n".join(f"{cid},{ids[j] if j < len(ids) else ''}" for j, cid in enumerate(train_ids)) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("data:\n  study: blca\n  wsi_encoder: uni2-h\ntrain:\n  seed: 3\n", encoding="utf-8")
    return dict(id=f"{arm}_f{fold}", arm=arm, cancer="blca", fold=fold, seed=3, protocol="legacy_val", source_commit="synthetic-only",
                predictions=str(pred), curve=str(curve), checkpoint=None, config=str(config), split_csv=str(split), overrides=[])


def test_cindex_event_censor_ties_and_no_fake_fallback():
    assert cindex([1, 1, 2], [0, 1, 0], [3, 2, 1]) == 1
    assert cindex([1, 2, 3], [0, 0, 0], [1, 1, 1]) == .5
    assert cindex([1, 2, 3], [0, 0, 0], [1, 2, 3]) == 0
    with pytest.raises(ValueError, match="No comparable"):
        cindex([1, 2], [1, 1], [3, 4])


def test_cindex_matches_training_metric_with_censor_and_risk_ties():
    from sksurv.metrics import concordance_index_censored
    rng = np.random.default_rng(23)
    for _ in range(10):
        times = rng.integers(1, 8, 30).astype(float)
        censor = rng.integers(0, 2, 30)
        risk = rng.integers(0, 5, 30).astype(float)
        assert cindex(times, censor, risk) == concordance_index_censored(censor == 0, times, risk)[0]


def test_audit_checks_cross_fold_patient_isolation(tmp_path):
    left, right = artifact(tmp_path, fold=0), artifact(tmp_path, fold=1)
    # Different bytes and scores alone cannot establish distinct folds.
    data = pickle.loads(Path(right["predictions"]).read_bytes())
    data["patient_0_0"] = data.pop("patient_1_0")
    Path(right["predictions"]).write_bytes(pickle.dumps(data))
    text = Path(right["split_csv"]).read_text().replace("patient_1_0", "patient_0_0").replace("patient_0_0,", "replacement_train,")
    Path(right["split_csv"]).write_text(text)
    report = audit({"runs": [left, right]})
    assert not report["passed"]
    assert any("overlap across folds" in error for error in report["errors"])


def test_audit_valid_earliest_max(tmp_path):
    run = artifact(tmp_path)
    assert best_epoch(run["curve"])[0] == 1
    report = audit({"runs": [run]})
    assert report["passed"] and report["runs"][0]["cindex"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "ids", "score", "split", "source"])
def test_audit_refuses_corrupted_evidence(tmp_path, problem):
    run = artifact(tmp_path)
    runs = [run]
    if problem == "duplicate":
        runs.append(dict(run, id="fake_fold1", fold=1))
    elif problem == "ids":
        Path(run["predictions"]).write_bytes(pickle.dumps({"wrong": dict(risk=1, time=1, censor=0), "also_wrong": dict(risk=0, time=2, censor=0)}))
    elif problem == "score":
        Path(run["curve"]).write_text("epoch,val_cindex\n0,0.7\n", encoding="utf-8")
    elif problem == "split":
        Path(run["split_csv"]).write_text("train,val\npatient_0_0,patient_0_0\n", encoding="utf-8")
    else:
        run["source_commit"] = None
    assert not audit({"runs": runs})["passed"]


def test_manifest_relative_paths(tmp_path):
    run = artifact(tmp_path)
    run["curve"] = str(Path(run["curve"]).relative_to(tmp_path))
    save_json(tmp_path / "manifest.json", dict(schema_version=1, runs=[run]))
    assert Path(load_manifest(tmp_path / "manifest.json")["runs"][0]["curve"]).is_absolute()


def test_intervention_preserves_actual_marginals_and_alpha_zero():
    plan = torch.rand(2, 3, 4, generator=torch.Generator().manual_seed(8))
    independent = independent_plan(plan)
    assert torch.allclose(independent.sum(-1), plan.sum(-1))
    assert torch.allclose(independent.sum(-2), plan.sum(-2))
    assert torch.equal(mix_plans([(plan, plan)], 0)[0][0], plan)
    for alpha in (.25, .5, 1):
        changed = mix_plans([(plan, plan)], alpha)[0][0]
        assert torch.allclose(changed.sum(-1), plan.sum(-1))
        assert torch.allclose(changed.sum(-2), plan.sum(-2))
    with pytest.raises(ValueError):
        mix_plans([(plan,)], 1.1)


@pytest.mark.parametrize("corruption", ["missing", "shape", "reference"])
def test_checkpoint_loading_never_skips_parameters_or_wrong_fold_reference(tmp_path, corruption):
    model = torch.nn.Linear(3, 2)
    model.register_buffer("dct_stage_edges", torch.tensor([1., 2., 3.]))
    state = model.state_dict()
    if corruption == "missing":
        del state["weight"]
    elif corruption == "shape":
        state["weight"] = torch.zeros(5, 3)
    else:
        state["dct_stage_edges"] = torch.tensor([1., 2., 4.])
    path = tmp_path / "wrong_checkpoint.pth"
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint_strict(model, path)


def test_real_v313_eval_replay_all_geometries_and_no_buffer_updates(monkeypatch):
    monkeypatch.setenv("DCT_DEBUG_LOG", str(Path(__file__).parent / "unused.log"))
    # Disable incidental legacy debug logging rather than writing outside the test tempdir.
    import survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model as module
    monkeypatch.setattr(module, "_debug_log", lambda **kwargs: None)
    torch.set_num_threads(1)
    model = DCTV313TransportReconstruction(make_args()).eval()
    model.configure_train_reference(*train_reference())
    payload = batch()
    payload.update(y=None, c=None)
    before = {name: value.clone() for name, value in model.named_buffers()}
    with torch.no_grad():
        logits = model(**payload)[0]
    values = replay(model, payload)
    assert np.allclose(values["logits"], logits.numpy(), atol=1e-6)
    assert values["plans"].shape == (8, 4, 3, 3, 3)
    assert values["attention_wsi"].shape == (8, 3, 6)
    assert values["attention_omic"].shape == (8, 3, 5)
    assert np.allclose(values["attention_wsi"].sum(-1), 1, atol=1e-5)
    assert np.allclose(values["sweep_risk"][:, 0], values["risk"], atol=1e-6)
    assert values["cross_pathway_error"].shape == (8, 5, 5)
    assert values["sweep_row_residual"].max() < 1e-6
    assert values["sweep_col_residual"].max() < 1e-6
    for name, value in model.named_buffers():
        assert torch.equal(value, before[name]), name
    target, prediction = torch.randn(8, 5, 16), torch.randn(8, 5, 16)
    assert torch.allclose(pathway_error(prediction, target).mean(), model._reconstruction_distance(prediction, target, torch.ones(8, dtype=torch.bool)))
    assert not model.slot_attention_wsi.to_q._forward_hooks


@pytest.mark.parametrize("num_patches", [5, 12])
def test_multislide_patch_identity_and_padding(num_patches):
    class Dataset:
        dataset_factory = SimpleNamespace(num_patches=num_patches)
        def _load_wsi_feature(self, path):
            return torch.arange(8 if path == "slideA.pt" else 12).reshape(-1, 2).float()
        def load_wsi(self, slides):
            return torch.cat([self._load_wsi_feature("slideA.pt"), self._load_wsi_feature("slideB.pt")])
        def __getitem__(self, index):
            raw = self.load_wsi("unused")
            n = min(num_patches, len(raw))
            selected = np.floor(np.arange(n) * len(raw) / n).astype(np.int64)
            output = raw[selected]
            if n < num_patches:
                output = torch.cat([output, torch.zeros(num_patches - n, 2)])
            return output, [torch.ones(2)], torch.tensor(0), torch.tensor(1.), torch.tensor(0.)
    _, files, slide, patch = indexed_sample(Dataset(), 0)
    assert [f["patch_count"] for f in files] == [4, 6]
    if num_patches == 5:
        assert slide.tolist() == [0, 0, 1, 1, 1]
        assert patch.tolist() == [0, 2, 0, 2, 4]
    else:
        assert slide[-2:].tolist() == [-1, -1]
        assert patch[-2:].tolist() == [-1, -1]


def test_km_observed_events_and_censoring():
    x, s, low, high = km_curve([1, 2, 3], [0, 1, 0])
    assert np.isclose(s[1], 2 / 3) and s[-1] == 0
    assert np.all(low <= s) and np.all(s <= high)


def test_supported_calibration_and_strict_coordinate_mapping(tmp_path):
    import matplotlib.pyplot as plt
    from survot_rank.evidence.manifest import sha256
    index, figures = {"skipped": []}, []
    def save(fig, name, note):
        figures.append(name)
        plt.close(fig)
    n = 60
    data = dict(train_time=np.arange(1, 101, dtype=float), train_censor=np.zeros(100),
                time=np.arange(1, n + 1, dtype=float), censor=np.zeros(n), bins=np.array([0, 10, 20, 30, 110]),
                survival=np.tile(np.linspace(.1, .9, n).reshape(-1, 1), (1, 4)))
    clinical_figures(data, "synthetic", plt, save, index)
    assert "brier_synthetic" in figures and "calibration_synthetic" in figures
    assert not index["skipped"]
    coords = tmp_path / "coords.npy"
    np.save(coords, np.arange(12).reshape(6, 2))
    case = dict(case_id="synthetic_patient", features=[dict(sha256="feature_hash", patch_count=6)])
    bag = dict(slide_index=np.array([0, 0, 0, -1]), patch_index=np.array([0, 2, 4, -1]),
               attention_wsi=np.ones((1, 3, 4)) / 4)
    mapping = {"features": [dict(feature_sha256="feature_hash", coords="coords.npy", coords_sha256=sha256(coords))]}
    spatial_figures(case, bag, mapping, tmp_path, "synthetic_case", plt, save, index)
    assert "synthetic_case_spatial_0" in figures
    np.save(coords, np.zeros((5, 2)))
    with pytest.raises(ValueError, match="hash mismatch"):
        spatial_figures(case, bag, mapping, tmp_path, "synthetic_case", plt, save, index)


def test_plot_synthetic_paired_folds_and_explicit_missing_exports(tmp_path):
    runs = [artifact(tmp_path, fold=f, arm=arm) for arm in ("exp6", "direct") for f in range(5)]
    index = make_figures({"runs": runs}, tmp_path / "figures", exports=tmp_path / "missing_exports")
    assert (tmp_path / "figures" / "paired_blca_direct_s3.pdf").exists()
    assert (tmp_path / "figures" / "scores.csv").exists()
    assert any("No completed exports" in message for message in index["skipped"])
    with pytest.raises(ValueError, match="empty"):
        make_figures({"runs": runs}, tmp_path / "figures")


def test_synthetic_checkpoint_export_to_figures(monkeypatch, tmp_path):
    import pandas as pd
    import survot_rank.evidence.v313 as exporter
    import survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model as module
    monkeypatch.setattr(module, "_debug_log", lambda **kwargs: None)
    torch.set_num_threads(1)
    model = DCTV313TransportReconstruction(make_args(dct_slot_init_mode="deterministic")).eval()
    model.configure_train_reference(*train_reference())
    payload = batch()
    payload.update(y=None, c=None)
    ids = [f"patient_{i}" for i in range(8)]
    values = []
    for i in range(8):
        one = {key: value[i:i + 1] if torch.is_tensor(value) else value for key, value in payload.items()}
        values.append(replay(model, one)["risk"][0])
    feature = tmp_path / "features.pt"
    feature.write_bytes(b"synthetic feature provenance only")
    class Dataset:
        dataset_factory = SimpleNamespace(num_patches=6)
        split_key = "val"
        label_df = pd.DataFrame({"case id": ids, "time": train_reference()[0].numpy(), "censor": train_reference()[1].numpy()})
        current_index = 0
        def __len__(self):
            return len(ids)
        def _load_wsi_feature(self, path):
            return payload["x_wsi"][self.current_index]
        def load_wsi(self, slides):
            return self._load_wsi_feature(str(feature))
        def __getitem__(self, i):
            self.current_index = i
            return (self.load_wsi("unused"), [payload[f"x_omic{j}"][i] for j in range(1, 6)],
                    torch.tensor(i // 2), train_reference()[0][i], train_reference()[1][i])
    args = make_args(dct_slot_init_mode="deterministic")
    monkeypatch.setattr(exporter, "load_run", lambda run, device: (model, args,
        SimpleNamespace(pathway_names=[f"Pathway {i}" for i in range(5)], bins=np.array([0, 3, 7, 11, 18]), label_col="time", censorship_var="censor"), Dataset(), Dataset()))
    # Exercise real export bookkeeping and plotting with synthetic inputs only.
    monkeypatch.setattr(exporter, "payload_for", lambda sample, args, device: dict(x_wsi=sample[0].unsqueeze(0),
        event_time=sample[3].reshape(1), y=None, c=None, **{f"x_omic{i}": x.unsqueeze(0) for i, x in enumerate(sample[1], 1)}))
    run = artifact(tmp_path)
    Path(run["predictions"]).write_bytes(pickle.dumps({cid: dict(risk=values[i], time=float(train_reference()[0][i]), censor=float(train_reference()[1][i])) for i, cid in enumerate(ids)}))
    Path(run["curve"]).write_text(f"epoch,val_cindex\n7,{cindex(train_reference()[0], train_reference()[1], values)}\n", encoding="utf-8")
    Path(run["split_csv"]).write_text("train,val\n" + "\n".join(f"train_{i},{cid}" for i, cid in enumerate(ids)), encoding="utf-8")
    checkpoint = tmp_path / "best.pth"
    torch.save(model.state_dict(), checkpoint)
    run["checkpoint"] = str(checkpoint)
    summary = exporter.export_run(run, tmp_path / "exports", device="cpu", case_ids=[ids[0]], km=True)
    assert summary["best_epoch"] == 7
    assert summary["km_train_median"] is not None
    assert len(summary["cases"]) == 1
    output = tmp_path / "exports" / run["id"]
    assert (output / "patients.npz").exists()
    index = make_figures({"runs": [run]}, tmp_path / "export_figures", exports=tmp_path / "exports")
    names = [figure["name"] for figure in index["figures"]]
    assert f"plan_sweep_{run['id']}" in names
    assert f"km_{run['id']}" in names
    assert any(name.endswith("_plans") for name in names)
    assert any(name.endswith("_pathways") for name in names)
    assert any("Calibration" in reason or "calibration" in reason for reason in index["skipped"])
    # Changing a source artifact invalidates existing export lineage.
    Path(run["curve"]).write_text(Path(run["curve"]).read_text() + "8,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hashes differ"):
        make_figures({"runs": [run]}, tmp_path / "changed_source_figures", exports=tmp_path / "exports")


def test_plan_and_bind_actual_fold_isolation(monkeypatch, tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("evidence_cli", root / "scripts" / "prepare_v313_evidence.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    data = tmp_path / "data"
    for cancer in ("blca", "kirc"):
        split = data / "splits" / "5fold_uni2h" / cancer
        split.mkdir(parents=True)
        for fold in range(5):
            (split / f"fold_{fold}.csv").write_text(f"train,val\ntrain,{cancer}_{fold}\n", encoding="utf-8")
    output = tmp_path / "plan.json"
    # Dry-run only; launch_tasks would raise if this test accidentally launched.
    import survot_rank.training.scheduler as scheduler
    monkeypatch.setattr(scheduler, "launch_tasks", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no launch authorized")))
    cli.main(["plan", "--group", "controls", "--output", str(output), "--results-root", str(tmp_path / "training"),
              "--data-path", str(data), "--feature-root", str(tmp_path / "features")])
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert len(plan["tasks"]) == 20
    assert len({task["results_dir"] for task in plan["tasks"]}) == 20
    assert {task["gpu_id"] for task in plan["tasks"]} == {"0/0", "1/0"}
    for task in plan["tasks"]:
        assert f"k_start={task['fold']}" in task["extra_set"]
        assert f"k_end={task['fold'] + 1}" in task["extra_set"]
        actual = Path(task["results_dir"]) / "actual_hash"
        actual.mkdir(parents=True)
        (actual / f"epoch_curve_fold{task['fold']}.csv").write_text("epoch,val_cindex\n0,0.5\n")
    cli.main(["bind", "--plan", str(output), "--output", str(tmp_path / "bound.json")])
    bound = load_manifest(tmp_path / "bound.json")
    assert len(bound["runs"]) == 20
    assert all("actual_hash" in run["checkpoint"] for run in bound["runs"])
    assert all(run["protocol"] == "legacy_val" for run in bound["runs"])
    # KIRC matched reconstruction falls back to its Full YAML, preserving actual ablation overrides.
    cli.main(["plan", "--group", "matched-recon", "--output", str(tmp_path / "matched.json"), "--results-root", str(tmp_path / "matched_training"),
              "--data-path", str(data), "--feature-root", str(tmp_path / "features")])
    matched = json.loads((tmp_path / "matched.json").read_text(encoding="utf-8"))
    assert len(matched["tasks"]) == 20
    assert all(Path(task["config_yaml_path"]).exists() for task in matched["tasks"])
