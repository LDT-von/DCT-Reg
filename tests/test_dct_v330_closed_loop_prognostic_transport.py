from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import run_dct_v330_experiments as queue
from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.research.components.confidence_aware_transport_reaggregation import (
    ConfidenceAwareTransportReaggregation,
)
from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.research.methods.dct_v330_closed_loop_prognostic_transport import (
    DCTV330ClosedLoopPrognosticTransport,
)
from survot_rank.training.extended_args import build_base_parser
from survot_rank.training.model_factory import get_model


@pytest.fixture(autouse=True)
def reproducibility():
    torch.manual_seed(123)


def component_inputs(batch=3):
    return (
        torch.randn(batch, 3, 16),
        torch.randn(batch, 4, 16),
        torch.randn(batch, 7, 16),
        torch.randn(batch, 5, 16),
    )


def model_args(**overrides):
    values = dict(
        bag_loss="nll_surv",
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=3,
        slot_num_omics=4,
        slot_iters=2,
        alpha_surv=0.15,
        otehv2_eps=0.1,
        otehv2_iter=20,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_coupling_projection_iters=30,
        dct_coupling_projection_tol=1e-4,
        dct_evidence_marginal_strength=0.5,
        dct_slot_init_mode="deterministic",
        cur_epoch=6,
        dct_v330_feedback_iters=30,
        dct_v330_final_eps_anneal_epochs=12,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def payload(batch=6):
    return dict(
        x_wsi=torch.randn(batch, 7, 16),
        x_omics=torch.randn(batch, 5, 20),
        y=torch.arange(batch) % 4,
        event_time=torch.arange(1, batch + 1).float(),
        c=torch.tensor(([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])[:batch]),
    )


@pytest.mark.parametrize("variant", ConfidenceAwareTransportReaggregation.VARIANTS)
def test_component_variants_are_finite_and_shape_preserving(variant):
    module = ConfidenceAwareTransportReaggregation(
        16, variant=variant, sinkhorn_iters=100
    )
    inputs = component_inputs()
    wsi, omic = module(*inputs, epoch=4)
    assert wsi.shape == inputs[0].shape
    assert omic.shape == inputs[1].shape
    assert torch.isfinite(wsi).all() and torch.isfinite(omic).all()
    assert module.last_diagnostics
    if variant == "baseline":
        assert wsi is inputs[0] and omic is inputs[1]
    elif variant == "self_update":
        assert module.last_plan is None
    else:
        assert module.last_plan.shape == (3, 3, 4)
        assert module.last_gates["wsi"].shape == (3, 3, 1)
        assert module.last_gates["omics"].shape == (3, 4, 1)
        assert module.last_diagnostics["marginal_error"] < 1e-4


def test_confidence_feedback_has_plan_gate_marginal_and_token_gradients():
    module = ConfidenceAwareTransportReaggregation(
        16, variant="confidence_gate", sinkhorn_iters=100
    )
    inputs = tuple(item.requires_grad_() for item in component_inputs())
    wsi, omic = module(*inputs, epoch=4)
    (wsi.square().mean() + omic.square().mean()).backward()
    assert module.wsi_confidence_gate[-1].weight.grad.abs().sum() > 0
    assert module.omic_confidence_gate[-1].weight.grad.abs().sum() > 0
    assert module.wsi_marginal_score[-1].weight.grad.abs().sum() > 0
    assert module.omic_marginal_score[-1].weight.grad.abs().sum() > 0
    for tensor in inputs:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_gate_uses_no_survival_labels_and_epsilon_anneals():
    module = ConfidenceAwareTransportReaggregation(
        16,
        variant="confidence_gate",
        adaptive_epsilon_start=0.5,
        adaptive_epsilon_end=0.1,
        adaptive_epsilon_anneal_epochs=10,
    ).eval()
    data = component_inputs()
    module(*data, epoch=0)
    first = {key: value.clone() for key, value in module.last_gates.items()}
    module(*data, epoch=0)
    for key in first:
        torch.testing.assert_close(first[key], module.last_gates[key], rtol=0, atol=0)
    assert module.epsilon_ratio(0) == pytest.approx(0.5)
    assert module.epsilon_ratio(5) == pytest.approx(0.3)
    assert module.epsilon_ratio(20) == pytest.approx(0.1)


@pytest.mark.parametrize("variant", ConfidenceAwareTransportReaggregation.VARIANTS)
def test_model_forward_objective_and_transport_only_representation(variant):
    args = model_args(dct_v330_variant=variant)
    model = get_model("dct_v330", args, omic_input_dim=20)
    reference_times = torch.arange(1, 13).float()
    reference_c = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]).float()
    model.configure_train_reference(reference_times, reference_c)
    model.train()
    data = payload()
    logits, auxiliary = model(**data)
    assert logits.shape == (6, 4)
    assert torch.isfinite(logits).all() and torch.isfinite(auxiliary)
    assert model._last_transport_representation.shape == (6, 16)
    assert not model.risk_anchor_seen.any()
    weights = model.objective_weights()
    assert weights["direction"] == 0.0
    if variant == "prognostic_rank":
        assert weights["transport_representation_rank"] > 0
        assert auxiliary > 0
        assert model.last_training_losses["v330_representation_rank_samples"] > 0
    else:
        assert weights["transport_representation_rank"] == 0.0
        assert auxiliary == 0
    (logits.square().mean() + auxiliary).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_inference_ignores_labels_and_exposes_feedback_diagnostics():
    model = DCTV330ClosedLoopPrognosticTransport(
        model_args(dct_v330_variant="prognostic_rank"), omic_input_dim=20
    ).eval()
    data = payload()
    with torch.no_grad():
        labelled, auxiliary = model(**data)
        unlabelled, auxiliary_without_labels = model(
            x_wsi=data["x_wsi"], x_omics=data["x_omics"]
        )
    torch.testing.assert_close(labelled, unlabelled, rtol=0, atol=0)
    assert auxiliary == auxiliary_without_labels == 0
    for key in (
        "v330_plan_entropy",
        "v330_plan_uncertainty",
        "v330_dependence_tv",
        "v330_marginal_error",
        "v330_feedback_gate_mean",
        "v330_feedback_gate_std",
        "v330_final_plan_entropy",
        "v330_final_marginal_error",
        "v330_final_geometry_plan_tv",
    ):
        assert key in model.last_explanations
    assert model.last_explanations["feedback_plan"].shape == (6, 3, 4)


def test_training_reference_uses_observed_training_times_only():
    model = DCTV330ClosedLoopPrognosticTransport(model_args(), omic_input_dim=20)
    times = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0, 200.0])
    censoring = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    model.configure_train_reference(times, censoring)
    expected = torch.quantile(
        torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0.4, 0.6])
    )
    torch.testing.assert_close(model.dct_v330_risk_thresholds.cpu(), expected)


def test_final_risk_changes_when_transport_plan_changes():
    model = DCTV330ClosedLoopPrognosticTransport(
        model_args(dct_v330_variant="baseline"), omic_input_dim=20
    ).eval()
    wsi = torch.randn(2, 3, 16)
    omic = torch.randn(2, 4, 16)
    uniform = torch.full((2, 3, 4), 1.0 / 12.0)
    concentrated = torch.full((2, 3, 4), 1e-4)
    concentrated[:, 0, 0] = 1.0 - 11e-4
    uniform_plans = [tuple(uniform for _ in range(3)) for _ in range(4)]
    concentrated_plans = [tuple(concentrated for _ in range(3)) for _ in range(4)]
    with torch.no_grad():
        first, _ = model._encode_logits_from_plans(wsi, omic, uniform_plans)
        second, _ = model._encode_logits_from_plans(wsi, omic, concentrated_plans)
    assert not torch.allclose(first, second)


def test_catalog_config_and_twenty_five_matched_jobs():
    key = METHOD_ALIASES["dct_v330"]
    assert key == queue.METHOD
    assert METHOD_CATALOG[key].status == "candidate"
    args = queue.build_parser().parse_args(["plan", "--python", "python"])
    jobs = queue.build_jobs(args)
    assert len(jobs) == 25
    assert len({job.result_dir for job in jobs}) == 25
    for job in jobs:
        overrides = [
            job.command[index + 1]
            for index, item in enumerate(job.command)
            if item == "--set"
        ]
        config = apply_overrides(load_config(job.config), overrides)
        parsed = build_base_parser().parse_args(config_to_argv(config))
        assert parsed.survot_method == queue.METHOD
        assert parsed.dct_v330_variant == job.variant
        assert parsed.dct_lambda_ipcw_rank == 0
        assert parsed.dct_v38_lambda_direction == 0
        assert parsed.max_epochs == 50 and parsed.seed == 3
        assert parsed.k_start == job.fold and parsed.k_end == job.fold + 1
    assert len(queue.build_jobs(args, smoke=True)) == 5


@pytest.mark.parametrize(
    "kwargs",
    (
        {"variant": "bad"},
        {"rounds": 0},
        {"fixed_strength": 2.0},
        {"fixed_epsilon": 0.0},
        {"adaptive_epsilon_start": 0.05, "adaptive_epsilon_end": 0.1},
        {"marginal_strength": -0.1},
        {"gate_hidden_dim": 1},
    ),
)
def test_invalid_component_settings_fail(kwargs):
    with pytest.raises(ValueError):
        ConfidenceAwareTransportReaggregation(16, **kwargs)


def test_checkpoint_rejects_a_different_v330_variant():
    source = DCTV330ClosedLoopPrognosticTransport(
        model_args(dct_v330_variant="confidence_gate"), omic_input_dim=20
    )
    target = DCTV330ClosedLoopPrognosticTransport(
        model_args(dct_v330_variant="prognostic_rank"), omic_input_dim=20
    )
    with pytest.raises(RuntimeError, match="configuration"):
        target.load_state_dict(source.state_dict())
