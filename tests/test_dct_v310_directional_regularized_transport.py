from __future__ import annotations

from types import SimpleNamespace
import json
import io

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.transport_dependency_proof.plan import (
    Checkpoint,
    checkpoint_audit_commands,
    proof_training_commands,
)
from scripts.audit_dct_reg import (
    apply_anchor_mode,
    controlled_plan_risk,
    dose_monotonicity,
    plan_total_variation,
)
from scripts import run_dct_v310_experiments as experiments
from scripts import run_dct_v310_final_cross_cancer as final
from survot_rank.research.methods.catalog import METHOD_ALIASES, PRIMARY_METHOD
from survot_rank.research.methods.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)
from survot_rank.research.methods.dct_v310_directional_regularized_transport import (
    DCTV310DirectionalRegularizedTransport,
)
from survot_rank.training.model_factory import get_model
from survot_rank.training.train_runner import compose_batch_objective, init_loss_function
from survot_rank.training import train_runner
from survot_rank.training.evidence import (
    finalize_fold_evidence,
    prepare_fold_evidence,
    write_predictions_csv,
)


def make_args(**overrides):
    values = dict(
        bag_loss="nll_surv",
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        alpha_surv=0.15,
        slot_num_wsi=3,
        slot_num_omics=3,
        slot_iters=2,
        otehv2_eps=0.05,
        otehv2_iter=20,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.99,
        dct_ipcw_rank_margin=0.99,
        dct_ipcw_rank_temperature=0.99,
        dct_ipcw_max_weight=99.0,
        dct_ipcw_rank_memory_size=1,
        dct_lambda_etar=0.99,
        dct_lambda_listwise=0.99,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.99,
        dct_evidence_mass_floor=0.99,
        dct_evidence_marginal_strength=0.0,
        dct_geometry_reliability_strength=0.99,
        dct_geometry_reliability_temperature=0.25,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.25,
        dct_v38_lambda_direction=0.99,
        dct_v38_lambda_dose=0.99,
        dct_v38_lambda_reconfiguration=0.99,
        dct_v38_direction_margin=0.99,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.99,
        dct_v38_alpha_mid=0.25,
        dct_v38_alpha_full=0.75,
        dct_v38_warmup_epochs=5,
        dct_v38_ramp_epochs=10,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.99,
        dct_v382_adaptive_aux_weights=True,
        dct_fixed_coupling=True,
        dct_random_anchors=True,
        dct_perm_labels_seed=7,
        dct_stage_jitter_fraction=0.3,
        dct_freeze_source_prototype="",
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        dct_slot_init_mode="gaussian",
        dct_slot_eval_seed=91,
        cur_epoch=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reference():
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return times, censorship


def batch(size=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    times, censorship = reference()
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return {
        "x_wsi": torch.randn(size, 6, 16, generator=generator),
        "x_omics": torch.randn(size, 5, 20, generator=generator),
        "y": y[:size],
        "event_time": times[:size],
        "c": censorship[:size],
    }


def override(job, key):
    command = list(job.command)
    prefix = f"{key}="
    for index, item in enumerate(command[:-1]):
        if item == "--set" and command[index + 1].startswith(prefix):
            return command[index + 1][len(prefix):]
    return None


def test_v310_is_primary_registered_method_and_factory_alias():
    key = "dct_v310_directional_regularized_transport"
    assert PRIMARY_METHOD == key
    assert METHOD_ALIASES["dct_v310"] == key
    assert METHOD_ALIASES["dct_reg"] == key
    model = get_model("dct_reg", make_args(), omic_input_dim=20)
    # The factory intentionally loads model.py under an isolated module name,
    # so class identity differs from the package import even though the public
    # implementation is the same class.
    assert type(model).__name__ == "DCTV310DirectionalRegularizedTransport"


def test_v310_hostile_overrides_cannot_change_frozen_recipe():
    args = make_args()
    model = DCTV310DirectionalRegularizedTransport(args, omic_input_dim=20)

    assert model.objective_weights() == {
        "nll": 1.0,
        "ipcw_rank": 0.10,
        "direction": 0.05,
    }
    assert model.dct_lambda_ipcw_rank == 0.10
    assert model.dct_v38_lambda_direction == 0.05
    assert model.dct_lambda_etar == 0.0
    assert model.dct_lambda_listwise == 0.0
    assert model.dct_v38_lambda_dose == 0.0
    assert model.dct_v38_lambda_reconfiguration == 0.0
    assert model.dct_v382_lambda_mgptr == 0.0
    assert model.dct_v382_adaptive_aux_weights is False
    assert model.dct_v38_warmup_epochs == 0
    assert model.dct_v38_ramp_epochs == 0
    assert model.dct_fixed_coupling is False
    assert model.dct_random_anchors is False
    assert model.dct_perm_labels_seed == 0
    assert model.dct_stage_jitter_fraction == 0.0
    assert model.dct_evidence_cost_weight == 0.0
    assert model.dct_geometry_reliability_strength == 0.0
    assert model.dct_mix_ratio == 1.0
    assert args.dct_slot_init_mode == "deterministic"


def test_v310_rejects_non_nll_primary_loss():
    with pytest.raises(ValueError, match="bag_loss='nll_surv'"):
        DCTV310DirectionalRegularizedTransport(
            make_args(bag_loss="cox_surv"), omic_input_dim=20
        )


def test_v310_forward_auxiliary_loss_is_exact_two_term_sum():
    torch.manual_seed(7)
    model = DCTV310DirectionalRegularizedTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    logits, aux_loss = model(**batch())
    diagnostics = model.last_training_losses
    expected = 0.10 * diagnostics["ipcw_rank"] + 0.05 * diagnostics["v38_direction"]

    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)
    assert diagnostics["ipcw_rank"] > 0
    assert diagnostics["v38_direction"] > 0
    assert float(aux_loss.detach()) == pytest.approx(
        float(expected), rel=1e-6, abs=1e-7
    )
    assert diagnostics["v38_total"] == pytest.approx(
        0.05 * float(diagnostics["v38_direction"]), rel=1e-6
    )
    assert model.dct_v38_lambda_dose * diagnostics["v38_dose"] == 0
    assert model.dct_v38_lambda_reconfiguration * diagnostics["v38_reconfiguration"] == 0
    assert "v382_mgptr_weighted" not in diagnostics


def test_v310_shared_trainer_objective_is_exact_paper_formula():
    torch.manual_seed(19)
    args = make_args()
    model = DCTV310DirectionalRegularizedTransport(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    payload = batch()
    logits, auxiliary_loss = model(**payload)
    loss_fn = init_loss_function(args)
    raw_nll = loss_fn(
        logits,
        payload["y"],
        payload["event_time"],
        payload["c"],
    )
    total = compose_batch_objective(raw_nll, auxiliary_loss, payload["y"].shape[0])
    diagnostics = model.last_training_losses
    expected = (
        raw_nll / payload["y"].shape[0]
        + 0.10 * diagnostics["ipcw_rank"]
        + 0.05 * diagnostics["v38_direction"]
    )

    torch.testing.assert_close(total, expected)


def test_shared_trainer_objective_rejects_empty_batch():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        compose_batch_objective(torch.tensor(1.0), torch.tensor(2.0), 0)


def test_v310_final_launcher_is_six_cancers_by_five_folds_and_exact_objective():
    args = final.build_parser().parse_args(["plan", "--python", "python"])
    jobs = final.build_jobs(args)

    assert len(jobs) == 30
    assert {job.cancer for job in jobs} == {
        "blca", "skcm", "hnsc", "lusc", "kirc", "ucec"
    }
    for job in jobs:
        assert job.config.as_posix() == (
            "configs/dct_v310_directional_regularized_transport.yaml"
        )
        assert override(job, "study") == job.cancer
        assert override(job, "survot_method") == (
            "dct_v310_directional_regularized_transport"
        )
        assert override(job, "bag_loss") == "nll_surv"
        assert override(job, "dct_lambda_ipcw_rank") == "0.1"
        assert override(job, "dct_v38_lambda_direction") == "0.05"
        assert override(job, "dct_v38_lambda_dose") == "0.0"
        assert override(job, "dct_v38_lambda_reconfiguration") == "0.0"
        assert override(job, "dct_v382_lambda_mgptr") == "0.0"
        assert override(job, "outer_eval_only") == "true"
        assert override(job, "formal_evidence_package") == "true"
        assert override(job, "formal_require_clean_git") == "true"


def test_v310_default_experiment_queue_is_matched_two_by_two_ablation():
    args = experiments.build_parser().parse_args(["plan", "--python", "python"])
    jobs = experiments.build_jobs(args)
    assert len(jobs) == 20
    assert {job.variant for job in jobs} == set(experiments.DEFAULT_VARIANTS)
    assert {job.cancer for job in jobs} == {"blca"}

    expected = {
        "nll_only": ("0.0", "0.0"),
        "ipcw_only": ("0.1", "0.0"),
        "direction_only": ("0.0", "0.05"),
        "full": ("0.1", "0.05"),
    }
    for job in jobs:
        assert (
            override(job, "dct_lambda_ipcw_rank"),
            override(job, "dct_v38_lambda_direction"),
        ) == expected[job.variant]
        if job.variant == "full":
            assert override(job, "survot_method") == (
                "dct_v310_directional_regularized_transport"
            )
        else:
            assert override(job, "survot_method") == experiments.PARENT_METHOD


def test_v310_mechanism_control_queue_matches_documented_subset():
    args = experiments.build_parser().parse_args(
        [
            "plan",
            "--python",
            "python",
            "--cancers",
            "blca,ucec,lusc",
            "--folds",
            "1,2,4",
            "--variants",
            (
                "fixed_coupling,noisy_batch_mean_anchors,"
                "permuted_reference"
            ),
        ]
    )
    jobs = experiments.build_jobs(args)

    assert len(jobs) == 27
    assert {job.cancer for job in jobs} == {"blca", "ucec", "lusc"}
    assert {job.fold for job in jobs} == {1, 2, 4}
    assert {job.variant for job in jobs} == {
        "fixed_coupling",
        "noisy_batch_mean_anchors",
        "permuted_reference",
    }


def test_fixed_coupling_replays_current_batch_for_two_batch_sizes():
    args = make_args(
        dct_lambda_ipcw_rank=0.10,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        dct_fixed_coupling=True,
        dct_random_anchors=False,
        dct_perm_labels_seed=0,
        dct_stage_jitter_fraction=0.0,
        dct_v38_warmup_epochs=0,
        dct_v38_ramp_epochs=0,
        dct_slot_init_mode="deterministic",
    )
    model = DCTTransportInterventionConsistency(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    logits_large, loss_large = model(**batch(size=8, seed=3))
    logits_small, loss_small = model(**batch(size=3, seed=4))

    assert logits_large.shape == (8, 4)
    assert logits_small.shape == (3, 4)
    assert torch.isfinite(loss_large)
    assert torch.isfinite(loss_small)
    assert model._factual_plan_cache[0][0].shape[0] == 3


def test_fixed_coupling_keeps_current_factual_sinkhorn_unchanged():
    common = dict(
        dct_lambda_ipcw_rank=0.10,
        dct_v38_lambda_direction=0.05,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        dct_random_anchors=False,
        dct_perm_labels_seed=0,
        dct_stage_jitter_fraction=0.0,
        dct_v38_warmup_epochs=0,
        dct_v38_ramp_epochs=0,
        dct_slot_init_mode="deterministic",
    )
    torch.manual_seed(13)
    fresh = DCTTransportInterventionConsistency(
        make_args(dct_fixed_coupling=False, **common), omic_input_dim=20
    )
    fixed = DCTTransportInterventionConsistency(
        make_args(dct_fixed_coupling=True, **common), omic_input_dim=20
    )
    fixed.load_state_dict(fresh.state_dict())
    fresh.configure_train_reference(*reference())
    fixed.configure_train_reference(*reference())
    fresh.eval()
    fixed.eval()

    payload = batch(seed=17)
    fresh(**payload)
    fixed(**payload)

    torch.testing.assert_close(
        fixed.last_explanations["stage_slot_pair_evidence"],
        fresh.last_explanations["stage_slot_pair_evidence"],
    )


def test_eval_exports_true_plans_and_feasible_uniform_control():
    torch.manual_seed(31)
    model = DCTV310DirectionalRegularizedTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()
    model(**batch(seed=32))
    model.eval()
    model(**batch(seed=33))

    explanations = model.last_explanations
    factual = explanations["factual_transport_plans"]
    low = explanations["low_transport_plans"]
    high = explanations["high_transport_plans"]
    assert factual.shape == (8, 4, 3, 3, 3)
    summary = plan_total_variation(
        factual.detach().numpy(), low.detach().numpy(), high.detach().numpy()
    )
    assert np.isfinite(summary["mean_tv"])

    controlled_risk, controlled = controlled_plan_risk(model, "uniform")
    assert controlled.shape == factual.shape
    assert torch.isfinite(controlled_risk).all()
    rows = controlled.sum(dim=-1)
    cols = controlled.sum(dim=-2)
    torch.testing.assert_close(
        rows, model._last_factual_rows.unsqueeze(2).expand_as(rows), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        cols, model._last_factual_cols.unsqueeze(2).expand_as(cols), atol=1e-6, rtol=1e-6
    )


def test_v310_repeated_eval_is_deterministic_and_plan_path_has_gradients():
    torch.manual_seed(41)
    model = DCTV310DirectionalRegularizedTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()
    logits, auxiliary = model(**batch(seed=42))
    (logits.sum() + auxiliary).backward()
    for module in (
        model.shared_wsi_prototypes,
        model.shared_omic_prototypes,
        model.stage_pair_cost,
        model.evidence_gate,
        model.fusion,
        model.event_hazard,
    ):
        parameters = [module] if isinstance(module, torch.nn.Parameter) else list(module.parameters())
        assert sum(float(item.grad.abs().sum()) for item in parameters if item.grad is not None) > 0

    model.eval()
    fixed_batch = batch(seed=43)
    with torch.no_grad():
        first = model(**fixed_batch)[0]
        second = model(**fixed_batch)[0]
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_anchor_swap_and_bidirectional_dose_controls():
    model = DCTV310DirectionalRegularizedTransport(make_args(), omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()
    model(**batch(seed=44))
    anchors = model.risk_anchor_costs.clone()
    seen = model.risk_anchor_seen.clone()
    apply_anchor_mode(model, "swapped")
    torch.testing.assert_close(model.risk_anchor_costs[:, 0], anchors[:, 1])
    torch.testing.assert_close(model.risk_anchor_costs[:, 1], anchors[:, 0])
    torch.testing.assert_close(model.risk_anchor_seen[:, 0], seen[:, 1])
    torch.testing.assert_close(model.risk_anchor_seen[:, 1], seen[:, 0])

    alphas = np.array([0.0, 0.5, 1.0])
    high = np.array([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]])
    low = np.array([[0.0, -0.2, -0.4], [0.1, -0.1, -0.3]])
    assert dose_monotonicity(alphas, high, increasing=True)["monotone_rate"] == 1.0
    assert dose_monotonicity(alphas, low, increasing=False)["monotone_rate"] == 1.0


def test_transport_dependency_proof_plan_is_centralized_and_checkpoint_bound(tmp_path):
    training = proof_training_commands("python")
    assert [item[0] for item in training] == [
        "P1_objective_2x2",
        "P2_mechanism_controls",
        "P6_cross_cancer_prediction",
    ]
    checkpoint = Checkpoint("blca", 2, tmp_path / "evidence" / "fold_2" / "checkpoint.pt")
    audits = checkpoint_audit_commands(checkpoint, "python")
    assert {name for name, _ in audits} == {
        "factual",
        "uniform_plan",
        "shuffled_plan",
        "anchor_swap",
        "dose_both_directions",
    }
    for _, command in audits:
        assert str(checkpoint.path) in command
        assert "study=blca" in command


def test_formal_evidence_package_contains_hashes_and_split_audit(tmp_path):
    split_path = tmp_path / "fold_0.csv"
    pd.DataFrame({"train": ["p1", "p2"], "val": ["p3", None]}).to_csv(
        split_path, index=False
    )
    args = SimpleNamespace(
        results_dir=str(tmp_path / "results"),
        seed=3,
        outer_eval_only=True,
        study="blca",
        max_epochs=50,
    )
    evidence_dir = prepare_fold_evidence(args, 0, split_path)
    write_predictions_csv(
        {"p3": {"risk": np.array(0.2), "logits": np.array([0.1, 0.2])}},
        evidence_dir / "predictions.csv",
    )
    (evidence_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    nested = evidence_dir / "proof_transport_dependency" / "uniform_plan"
    nested.mkdir(parents=True)
    (nested / "audit_metrics.json").write_text("{}", encoding="utf-8")
    finalize_fold_evidence(evidence_dir, metrics={"outer_cindex": 0.7})

    manifest = json.loads((evidence_dir / "run_manifest.json").read_text(encoding="utf-8"))
    split_manifest = json.loads(
        (evidence_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["evaluation_protocol"] == "fixed_epoch_outer_once"
    assert "checkpoint.pt" in manifest["artifacts"]
    assert "proof_transport_dependency/uniform_plan/audit_metrics.json" in manifest["artifacts"]
    assert len(manifest["artifacts"]["checkpoint.pt"]["sha256"]) == 64
    assert split_manifest["columns"]["train"]["count"] == 2
    assert split_manifest["overlaps"] == {}


def test_outer_eval_only_never_evaluates_during_training(monkeypatch, tmp_path):
    labels = pd.DataFrame(
        {"case id": ["p1", "p2"], "time": [1.0, 2.0], "c": [0.0, 1.0], "label": [0, 1]}
    )
    data = SimpleNamespace(label_df=labels)
    loader = SimpleNamespace(dataset=data)
    factory = SimpleNamespace(censorship_var="c", label_col="time")
    model = torch.nn.Linear(1, 1)
    model.configure_train_reference = lambda *_: None
    calls = {"train": 0, "evaluate": 0}

    monkeypatch.setattr(train_runner, "get_split", lambda *_: (data, data, loader, loader))
    monkeypatch.setattr(train_runner, "init_model_for_method", lambda *_: model)
    monkeypatch.setattr(train_runner, "init_loss_function", lambda *_: object())
    monkeypatch.setattr(
        train_runner, "init_optimizer", lambda *_: torch.optim.SGD(model.parameters(), lr=0.1)
    )
    monkeypatch.setattr(train_runner, "init_scheduler", lambda *_: object())
    monkeypatch.setattr(train_runner, "ensure_min_free_space", lambda *_: 100.0)
    monkeypatch.setattr(train_runner, "_extract_survival_metadata", lambda *_: object())

    def fake_train(*_):
        calls["train"] += 1
        assert calls["evaluate"] == 0
        return {"loss": float(calls["train"])}

    def fake_evaluate(*_):
        calls["evaluate"] += 1
        assert calls["train"] == 3
        return ({"p2": {"risk": 0.2, "censor": 1.0, "time": 2.0, "logits": [0.1]}},
                0.6, 0.61, np.array([0.1]), 0.2, 0.62, 0.3)

    monkeypatch.setattr(train_runner, "train_one_epoch", fake_train)
    monkeypatch.setattr(train_runner, "evaluate", fake_evaluate)
    args = SimpleNamespace(
        results_dir=str(tmp_path), min_free_space_gb=0.0, outer_eval_only=True,
        formal_evidence_package=False, early_stop_patience=0, max_epochs=3,
        binning_mode="global_qcut", fit_bins_on_train=True,
        event_sampling_fraction=0.0, seed=3,
    )
    _, metrics = train_runner.train_one_fold(args, factory, 0, io.StringIO())
    assert calls == {"train": 3, "evaluate": 1}
    assert metrics[0] == pytest.approx(0.6)
