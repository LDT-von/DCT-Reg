"""Data-free workflow failures and prediction-path isolation, no patient jobs."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts import run_dct_v23_workflow as workflow
from survot_rank.research.methods.dct_v23 import DCTV23, DCTV23Config
from survot_rank.config import config_to_argv
from survot_rank.training.extended_args import process_args_extended


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny(path="combined"):
    return DCTV23Config(omic_sizes=(2, 3), dim=8, wsi_input_dim=3, wsi_hidden_dim=4,
        num_classes=3, wsi_slots=2, omics_slots=2, stages=2, heads=2,
        event_layers=1, slot_iters=1, slot_heads=2, slot_dim_head=4,
        slot_hidden_dim=16, dropout=0., wsi_dropout=0., reconstruction_mode="off",
        prediction_path=path, relation_weight=0., native_weight=0.)


def instances():
    torch.manual_seed(7)
    models = [DCTV23(tiny(p)).eval() for p in ("combined", "native_only", "transport_only")]
    weights = models[0].state_dict()
    for model in models[1:]:
        own = deepcopy(weights)
        own["_extra_state"]["config"]["prediction_path"] = model.config.prediction_path
        model.load_state_dict(own)
    x = dict(x_wsi=torch.randn(2, 4, 3), x_omic1=torch.randn(2, 2), x_omic2=torch.randn(2, 3))
    return models, x


def test_paths_keep_identical_parameters_and_decompose_shared_linear_head():
    models, x = instances()
    outputs = [m(**x)[0] for m in models]
    keys = [set(dict(m.named_parameters())) for m in models]
    assert keys[0] == keys[1] == keys[2]
    for name in keys[0]:
        assert torch.equal(dict(models[0].named_parameters())[name], dict(models[2].named_parameters())[name])
    assert torch.allclose(outputs[0], outputs[1] + outputs[2] - models[0].event_hazard.bias, atol=2e-6)
    assert torch.equal(models[1]._predict(models[0].last_state)[0], models[1].event_hazard(
        models[1]._native_feature(models[0].last_state)))


def test_native_prediction_ignores_exchange_and_has_no_main_cost_gradient():
    models, x = instances()
    m = models[1]
    before = m(**x)[0]
    state = m.last_state
    changed = replace(state, event_tokens=state.event_tokens + 100,
                      plans=state.plans * 10, stage_weights=state.stage_weights.flip(-1))
    assert torch.equal(before, m._predict(changed)[0])
    before.square().sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in m.core.stage_pair_cost.parameters())
    assert m.event_hazard.weight.grad.abs().sum() > 0


def test_transport_path_has_no_native_summary_and_missing_pair_is_bias_only():
    models, x = instances()
    m = models[2]
    m(**x)
    assert not m._native_feature(m.last_state).any()
    logits, _ = m(**x, omic_missing=True)
    assert torch.allclose(logits, m.event_hazard.bias.expand_as(logits))


@pytest.mark.parametrize("field,value", [("native_weight", .1), ("relation_weight", .1), ("reconstruction_mode", "masked_plain")])
def test_path_controls_reject_confounded_auxiliary_recipe(field, value):
    with pytest.raises(ValueError, match="attribution"):
        replace(tiny("native_only"), **{field: value})


def test_core_plan_is_data_free_exclusive_and_all_arms_have_matched_controls(tmp_path):
    root = tmp_path / "new"
    plan = workflow.make_plan(root, stage="core", folds=[0])
    assert len(plan["jobs"]) == 4
    assert plan["objective_recipe"] == workflow.OBJECTIVE_RECIPE
    assert not (root / "runs").exists() and not (root / "calibration").exists()
    _, _, jobs = workflow.read_plan(root)
    for job, cfg in jobs:
        args = process_args_extended(config_to_argv(cfg))
        args.omic_sizes = (2, 3)
        recipe = DCTV23Config.from_args(args)
        assert recipe.reconstruction_mode == "off"
        assert recipe.relation_weight == recipe.native_weight == 0
        assert recipe.local_readout == "none"
        assert set(DCTV23(recipe).get_extra_state()["losses"]) == {"global_nll", "slot_diversity"}
        assert cfg["model"]["dct_v2_lambda_ipcw_rank"] == args.dct_v2_lambda_ipcw_rank == 0.
        assert cfg["model"]["dct_v2_lambda_slot_nll"] == args.dct_v2_lambda_slot_nll == 0.
        assert job["objective_recipe"] == plan["objective_recipe"]
        assert args.seed == 3 and args.max_epochs == 30
        assert recipe.plan_mode == ("independent" if job["arm"] == "independent" else "learned")
    with pytest.raises(FileExistsError):
        workflow.make_plan(root, stage="core", folds=[0])


def test_recovery_plan_uses_simplified_core_and_keeps_recovery_controls(tmp_path):
    plan = workflow.make_plan(tmp_path / "new", stage="recovery", folds=[0, 2])
    _, _, jobs = workflow.read_plan(tmp_path / "new")
    assert len(jobs) == 6
    for job, cfg in jobs:
        args = process_args_extended(config_to_argv(cfg))
        args.omic_sizes = (2, 3)
        recipe = DCTV23Config.from_args(args)
        assert args.survot_method == "dct_v23"
        assert cfg["model"]["dct_v2_lambda_ipcw_rank"] == args.dct_v2_lambda_ipcw_rank == 0.
        assert cfg["model"]["dct_v2_lambda_slot_nll"] == args.dct_v2_lambda_slot_nll == 0.
        assert job["objective_recipe"] == plan["objective_recipe"] == workflow.OBJECTIVE_RECIPE
        assert cfg["model"]["dct_v2_lambda_relation"] == recipe.relation_weight == 0.
        assert cfg["model"]["dct_v2_lambda_native"] == recipe.native_weight == 0.
        assert cfg["model"]["dct_v21_local_readout"] == recipe.local_readout == "none"
        assert cfg["model"]["dct_v2_lambda_diversity"] == .1
        assert cfg["reconstruction"]["dct_v23_reconstruction_mode"] == job["arm"]
        assert cfg["reconstruction"]["dct_v23_lambda_reconstruction"] == .05
        assert cfg["reconstruction"]["dct_v23_lambda_calibration"] == .02
        assert cfg["reconstruction"]["dct_v23_calibration_dir"] == str((tmp_path / "new/calibration").resolve())
        assert args.seed == 3 and args.max_epochs == 30
        assert cfg["train"]["bag_loss"] == "nll_surv"
        assert cfg["train"]["lr"] == .0001 and cfg["train"]["reg"] == .005
        active = DCTV23(recipe).get_extra_state()["losses"]
        expected = {"global_nll", "slot_diversity"}
        if job["arm"] != "off":
            expected.add("reconstruction")
        if job["arm"] == "gain_calibrated":
            expected.add("gain_calibration")
        assert set(active) == expected
        assert len(active) == {"off": 2, "masked_plain": 3, "gain_calibrated": 4}[job["arm"]]


def test_canonical_recipe_preserves_historical_config_and_all_other_conditions():
    historical = yaml.safe_load((workflow.ROOT / "configs/dct_v23_blca_legacy_val_gain_calibrated.yaml").read_text(encoding="utf-8"))
    canonical = yaml.safe_load((workflow.ROOT / "configs/dct_rg_blca_legacy_val.yaml").read_text(encoding="utf-8"))
    assert historical["model"]["dct_v2_lambda_ipcw_rank"] == .1
    assert historical["model"]["dct_v2_lambda_slot_nll"] == .05
    assert historical["model"]["dct_v2_lambda_relation"] == .05
    assert historical["model"]["dct_v2_lambda_native"] == .05
    assert historical["model"]["dct_v21_local_readout"] == "centered_pair"
    expected = deepcopy(historical)
    expected["model"].update(dct_v2_lambda_ipcw_rank=0., dct_v2_lambda_slot_nll=0.,
        dct_v2_lambda_relation=0., dct_v2_lambda_native=0., dct_v21_local_readout="none")
    for field in ("name", "description"):
        expected[field] = canonical[field]
    for field in ("specific_simple", "results_dir"):
        expected["train"][field] = canonical["train"][field]
    expected["reconstruction"]["dct_v23_calibration_dir"] = canonical["reconstruction"]["dct_v23_calibration_dir"]
    assert canonical == expected
    assert canonical["train"]["results_dir"] != historical["train"]["results_dir"]
    assert canonical["reconstruction"]["dct_v23_calibration_dir"] != historical["reconstruction"]["dct_v23_calibration_dir"]


def test_edited_config_or_source_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "new"
    workflow.make_plan(root, stage="core", folds=[0])
    monkeypatch.setattr(workflow, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="Source"):
        workflow.read_plan(root)
    monkeypatch.undo()
    cfg = root / "configs/combined/fold_0.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config changed"):
        workflow.read_plan(root)


def test_missing_gain_table_prevents_any_training_subprocess(tmp_path, monkeypatch):
    root = tmp_path / "new"
    workflow.make_plan(root, stage="recovery", folds=[0])
    monkeypatch.setattr(workflow, "bind_inputs", lambda *a: None)
    def absent(*a):
        raise FileNotFoundError("synthetic missing gain table")
    monkeypatch.setattr(workflow, "audit_calibration", absent)
    monkeypatch.setattr(workflow.subprocess, "run", lambda *a, **kw: pytest.fail("Must not launch"))
    with pytest.raises(FileNotFoundError):
        workflow.run_jobs(root)


@pytest.mark.parametrize("returncode", [0, 2])
def test_exit_code_alone_cannot_mark_success_and_failure_is_recorded(tmp_path, monkeypatch, returncode):
    root = tmp_path / "new"
    workflow.make_plan(root, stage="core", folds=[0])
    monkeypatch.setattr(workflow, "bind_inputs", lambda *a: None)
    monkeypatch.setattr(workflow.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=returncode))
    def no_artifact(*a):
        raise FileNotFoundError("synthetic absent checkpoint")
    monkeypatch.setattr(workflow, "audit_job", no_artifact)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        workflow.run_jobs(root, ["combined"])
    assert list((root / "reports").glob("failure_*.json"))
    assert not list((root / "reports").glob("completed_*.json"))


def saved_fixture(tmp_path, calibrated=False):
    root = tmp_path / "batch"
    workflow.make_plan(root, stage="core", folds=[0], data_path=tmp_path / "data")
    _, _, jobs = workflow.read_plan(root, ["combined"])
    cfg = jobs[0][1]
    cfg["train"]["max_epochs"] = 2
    train_ids = ["p0", "p1", "p2", "p3"] if calibrated else ["p0", "p1"]
    if calibrated:
        cfg["reconstruction"]["dct_v23_reconstruction_mode"] = "gain_calibrated"
    data = Path(cfg["data"]["data_path"])
    splits = data / "splits/5fold_uni2h/blca"
    splits.mkdir(parents=True)
    pd.DataFrame({"train": train_ids, "val": ["v0", "v1"] + [None] * (len(train_ids)-2)}).to_csv(splits / "fold_0.csv", index=False)
    clinical = data / "clinical/all"
    clinical.mkdir(parents=True)
    pd.DataFrame({"case id": ["v0", "v1"], "survival_months_dss": [1., 2.], "censorship_dss": [0., 0.]}).to_csv(clinical / "blca.csv", index=False)
    args = process_args_extended(config_to_argv(cfg))
    args.omic_sizes = (2, 3)
    model = DCTV23(DCTV23Config.from_args(args), omic_names=(("a", "b"), ("b", "c", "d")))
    model.configure_reconstruction_reference((("a", "b"), ("b", "c", "d")), [0.] * 4, [1.] * 4, train_ids)
    if calibrated:
        from survot_rank.research.methods.dct_v23.crossfit import fit_cross_patient_gains, save_gain_table
        rng = np.random.default_rng(11)
        table = fit_cross_patient_gains(rng.normal(size=(4, 4)), rng.normal(size=(4, 1536)),
            patient_ids=train_ids, pathway_gene_ids=(("a", "b"), ("b", "c", "d")),
            inner_folds=2, mask_rounds=2, native_dim=2, wsi_dim=2)
        model.bind_gain_targets(table["gains"], table["provenance"])
        save_gain_table(table, Path(cfg["reconstruction"]["dct_v23_calibration_dir"]) / "fold_0.json")
    directory = Path(cfg["train"]["results_dir"]) / "blca/SurvOTRank_dct_v23/generated_parameters"
    directory.mkdir(parents=True)
    torch.save(model.state_dict(), directory / "model_best_s0.pth")
    pd.DataFrame({"epoch": [0, 1], "val_cindex": [1., 1.], "val_loss": [1., 2.]}).to_csv(directory / "epoch_curve_fold0.csv", index=False)
    pred = {}
    for cid, logit, time in (("v0", 1., 1.), ("v1", -1., 2.)):
        logits = np.full(4, logit)
        risk = -np.cumprod(1 - 1 / (1 + np.exp(-logits))).sum()
        pred[cid] = dict(risk=risk, logits=logits, time=time, censor=0.)
    with (directory / "split_0_results_final.pkl").open("wb") as handle:
        pickle.dump(pred, handle)
    return cfg, directory, pred


def test_saved_checkpoint_predictions_and_actual_budget_are_accepted(tmp_path):
    config, _, _ = saved_fixture(tmp_path)
    report = workflow.audit_job(config, 0)
    assert report["actual_epochs"] == 2 and report["best_epoch"] == 0
    assert report["val_cindex"] == 1.


def test_gain_checkpoint_matches_double_precision_table_after_float32_binding(tmp_path):
    config, _, _ = saved_fixture(tmp_path, calibrated=True)
    assert workflow.audit_job(config, 0)["val_cindex"] == 1.
    table_path = Path(config["reconstruction"]["dct_v23_calibration_dir"]) / "fold_0.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    table["gains"]["p0"][0] += 1.
    table_path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError, match="different recovery gains"):
        workflow.audit_job(config, 0)


def test_accepted_teacher_file_cannot_be_silently_replaced(tmp_path):
    path = tmp_path / "calibration/fold_0.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")
    workflow.dump_new(tmp_path / "calibration_binding.json", {"calibration/fold_0.json": workflow.sha(path)})
    workflow.check_calibration_binding(tmp_path, 0)
    path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="calibration content changed"):
        workflow.check_calibration_binding(tmp_path, 0)


@pytest.mark.parametrize("fault", ["short_curve", "missing_patient", "nan_prediction", "wrong_risk", "wrong_endpoint", "checkpoint_recipe"])
def test_incomplete_or_mismatched_artifacts_cannot_pass(tmp_path, fault):
    config, directory, pred = saved_fixture(tmp_path)
    if fault == "short_curve":
        pd.DataFrame({"epoch": [0], "val_cindex": [1.], "val_loss": [1.]}).to_csv(directory / "epoch_curve_fold0.csv", index=False)
    elif fault == "checkpoint_recipe":
        path = directory / "model_best_s0.pth"
        state = torch.load(path, weights_only=False)
        state["_extra_state"]["config"]["native_weight"] = .12
        torch.save(state, path)
    else:
        if fault == "missing_patient":
            pred.pop("v0")
        elif fault == "nan_prediction":
            pred["v0"]["risk"] = np.nan
        elif fault == "wrong_risk":
            pred["v0"]["risk"] += .1
        else:
            pred["v0"]["time"] += 1.
        with (directory / "split_0_results_final.pkl").open("wb") as handle:
            pickle.dump(pred, handle)
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.audit_job(config, 0)


def test_input_binding_rejects_csv_or_stored_feature_drift(tmp_path):
    cfg, _, _ = saved_fixture(tmp_path)
    data = Path(cfg["data"]["data_path"])
    for name in ("raw_rna_data_inter/blca_rna_inter.csv", "signatures/combine_signatures.csv"):
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n", encoding="utf-8")
    cfg["data"]["data_root_dir"] = str(tmp_path / "features")
    features = Path(cfg["data"]["data_root_dir"]) / "blca/uni2-h/pt_files"
    features.mkdir(parents=True)
    (features / "synthetic.pt").write_bytes(b"synthetic fixture")
    jobs = [(dict(fold=0), cfg)]
    workflow.bind_inputs(tmp_path / "batch", jobs)
    workflow.bind_inputs(tmp_path / "batch", jobs)
    (features / "synthetic.pt").write_bytes(b"changed fixture size")
    with pytest.raises(ValueError, match="changed"):
        workflow.bind_inputs(tmp_path / "batch", jobs)


def test_existing_outputs_audit_never_promotes_historical_source_to_verified(tmp_path):
    cfg, _, _ = saved_fixture(tmp_path)
    path = tmp_path / "original.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rows = workflow.audit_existing(path, cfg["train"]["results_dir"], [0], tmp_path / "historical_audit")
    assert rows[0]["status"] == "artifacts_accepted_source_pending"
