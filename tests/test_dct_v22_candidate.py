"""Capacity controls and diagnostic witnesses, using synthetic tensors only."""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch
from torch import nn

from survot_rank.research.methods.dct_v21 import DCTV21, DCTV21Config
from survot_rank.research.methods.dct_v22 import DCTV22, DCTV22Config
from survot_rank.research.methods.dct_v22.diagnostics import (
    plan_monitoring_metrics, relation_gap_metrics, slot_monitoring_metrics, state_monitoring_metrics,
)


@pytest.fixture(autouse=True)
def single_thread_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**changes):
    return replace(DCTV22Config(
        dim=8, wsi_input_dim=6, wsi_hidden_dim=4, wsi_dropout=.30,
        omic_sizes=(3, 4), num_classes=3, wsi_slots=2, omics_slots=2,
        stages=2, heads=2, event_layers=1, slot_iters=1, slot_heads=2,
        slot_dim_head=4, slot_hidden_dim=16, dropout=.30,
        relation_warmup_epochs=0, relation_ramp_epochs=1,
    ), **changes)


def _v21_config(config):
    return DCTV21Config(**{name: value for name, value in config.to_dict().items()
                          if name in DCTV21Config.__dataclass_fields__})


def _inputs(batch=3):
    generator = torch.Generator().manual_seed(918)
    return dict(x_wsi=torch.randn(batch, 5, 6, generator=generator),
                x_omic1=torch.randn(batch, 3, generator=generator),
                x_omic2=torch.randn(batch, 4, generator=generator),
                wsi_mask=torch.ones(batch, 5, dtype=torch.bool),
                omics_mask=torch.ones(batch, 2, dtype=torch.bool),
                y=torch.arange(batch) % 3, c=(torch.arange(batch) % 2).float(),
                event_time=torch.arange(1, batch + 1).float(), epoch=10)


def _training_model(config=None):
    torch.manual_seed(715)
    model = DCTV22(config or _config())
    model.set_training_reference(torch.arange(1, 7).float(), torch.tensor([0., 1., 0., 1., 0., 0.]))
    return model.train()


def test_default_recipe_has_real_capacity_reduction_in_wide_projection():
    old = DCTV21(DCTV21Config(omic_sizes=(3, 4)))
    new = DCTV22(DCTV22Config(omic_sizes=(3, 4)))
    old_projection = sum(value.numel() for value in old.wsi_mlp.parameters())
    new_projection = sum(value.numel() for value in new.wsi_mlp.parameters())
    assert old_projection == 1536 * 1536 + 1536 + 1536 * 256 + 256
    assert new_projection == 1536 * 256 + 256 + 256 * 128 + 128
    assert new_projection < .2 * old_projection
    assert sum(value.numel() for value in new.parameters()) < sum(value.numel() for value in old.parameters())
    assert new.config.dim == 128 and new.config.dropout == .30
    assert new.get_extra_state()["losses"] == old.get_extra_state()["losses"]
    assert not any("recon" in name for name, _ in new.named_parameters())


def test_configured_dropout_is_after_activation_and_eval_is_deterministic():
    model = DCTV22(_config(wsi_dropout=.45))
    assert isinstance(model.wsi_mlp[0], nn.Linear)
    assert (model.wsi_mlp[0].in_features, model.wsi_mlp[0].out_features) == (6, 4)
    assert isinstance(model.wsi_mlp[1], nn.ReLU)
    assert isinstance(model.wsi_mlp[2], nn.Dropout)
    assert model.wsi_mlp[2].p == .45
    assert (model.wsi_mlp[3].in_features, model.wsi_mlp[3].out_features) == (4, 8)
    assert model.sig_networks[0][0][2].p == .30
    observed = []
    handle = model.wsi_mlp[2].register_forward_hook(lambda module, args, result: observed.append(result))
    model.eval()
    with torch.no_grad():
        first, _ = model(**_inputs())
        second, _ = model(**_inputs())
    handle.remove()
    assert len(observed) == 2
    torch.testing.assert_close(first, second, atol=0., rtol=0.)


def test_original_width_zero_dropout_control_keeps_seed_weights_predictions_and_rng():
    config = _config(wsi_hidden_dim=6, wsi_dropout=0., dropout=.15)
    torch.manual_seed(1327)
    old = DCTV21(_v21_config(config))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(1327)
    new = DCTV22(config)
    assert torch.equal(old_rng, torch.get_rng_state())
    old_values, new_values = dict(old.named_parameters()), dict(new.named_parameters())
    assert old_values.keys() == new_values.keys()
    for name in old_values:
        torch.testing.assert_close(old_values[name], new_values[name], atol=0., rtol=0.)
    assert len(new.wsi_mlp) == 3
    old.eval()
    new.eval()
    with torch.no_grad():
        older, _ = old(**_inputs())
        rng_before_monitoring = torch.get_rng_state().clone()
        newer, _ = new(**_inputs())
        assert torch.equal(rng_before_monitoring, torch.get_rng_state())
    torch.testing.assert_close(older, newer, atol=0., rtol=0.)
    torch.testing.assert_close(old.last_state.plans, new.last_state.plans, atol=0., rtol=0.)
    for model in (old, new):
        model.set_training_reference(torch.arange(1, 7).float(), torch.tensor([0., 1., 0., 1., 0., 0.]))
        model.train()
    torch.manual_seed(820)
    old_logits, old_auxiliary = old(**_inputs())
    old_rng_after_forward = torch.get_rng_state().clone()
    torch.manual_seed(820)
    new_logits, new_auxiliary = new(**_inputs())
    assert torch.equal(old_rng_after_forward, torch.get_rng_state())
    torch.testing.assert_close(new_logits, old_logits, atol=0., rtol=0.)
    torch.testing.assert_close(new_auxiliary, old_auxiliary, atol=0., rtol=0.)


def test_checkpoint_roundtrip_and_recipe_schema_boundaries():
    model = _training_model().eval()
    with torch.no_grad():
        expected, _ = model(**_inputs())
    saved = model.state_dict()
    restored = DCTV22(_config()).eval()
    restored.load_state_dict(saved)
    with torch.no_grad():
        actual, _ = restored(**_inputs())
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    torch.testing.assert_close(restored.censor_times, model.censor_times)
    with pytest.raises(ValueError, match="exact saved recipe"):
        DCTV22(_config(wsi_dropout=.4)).load_state_dict(saved)
    with pytest.raises(ValueError, match="schema"):
        DCTV22(_config()).load_state_dict(DCTV21(_v21_config(_config())).state_dict(), strict=False)
    with pytest.raises(ValueError, match="schema"):
        DCTV21(_v21_config(_config())).load_state_dict(saved, strict=False)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_six_term_sum_finite_backward_and_observation_only_monitoring(mixed_precision):
    model = _training_model()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        output = model.forward_with_losses(**_inputs(2))
    terms = output["terms"]
    names = ["ipcw_rank", "per_slot_nll", "slot_diversity", "relation", "native"]
    auxiliary = sum(terms[name + "_weighted"] for name in names)
    torch.testing.assert_close(output["auxiliary"], auxiliary)
    torch.testing.assert_close(output["loss"], terms["global_nll"] + auxiliary)
    assert not any("recon" in name for name in terms)
    assert model.last_monitoring_metrics["global_nll"] == pytest.approx(float(terms["global_nll"]), rel=.01)
    assert model.last_monitoring_metrics["auxiliary_total"] == pytest.approx(float(auxiliary))
    assert all(isinstance(value, float) and math.isfinite(value) for value in model.last_monitoring_metrics.values())
    output["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    # Monitoring adds no trainable modules or buffers.
    assert not any("monitor" in name for name in model.state_dict())


def test_feature_separation_and_identical_risk_responses_are_distinct_from_feature_collapse():
    separated = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    collapsed = torch.tensor([[[1., 0.], [1., 0.]]], requires_grad=True)
    same_response = torch.zeros(1, 2, 2, requires_grad=True)
    available = torch.tensor([True])
    healthy_features = slot_monitoring_metrics(separated, same_response, available, "wsi")
    equal_features = slot_monitoring_metrics(collapsed, same_response, available, "wsi")
    assert healthy_features["wsi_slot_pair_dist_mean"] == pytest.approx(math.sqrt(2))
    assert healthy_features["wsi_slot_near_duplicate_fraction"] == 0.
    assert healthy_features["wsi_slot_cosine_mean"] == 0.
    assert equal_features["wsi_slot_pair_dist_mean"] == 0.
    assert equal_features["wsi_slot_near_duplicate_fraction"] == 1.
    assert equal_features["wsi_slot_cosine_mean"] == 1.
    assert healthy_features["wsi_risk_response_variance"] == equal_features["wsi_risk_response_variance"] == 0.
    assert separated.grad is collapsed.grad is same_response.grad is None


def test_patientwise_slot_statistics_masks_and_saturation_are_explicit():
    slots = torch.tensor([[[0., 0.], [3., 4.]], [[0., 0.], [0., 0.]], [[1., 0.], [2., 0.]]])
    risk = torch.tensor([[[-20.], [20.]], [[0.], [0.]], [[0.], [0.]]])
    result = slot_monitoring_metrics(slots, risk, torch.tensor([True, False, True]), "omics")
    assert result["omics_monitor_patient_count"] == 2.
    assert result["omics_slot_pair_dist_mean"] == result["omics_slot_pair_dist_min"] == 3.
    assert result["omics_slot_near_duplicate_fraction"] == 0.
    assert result["omics_risk_response_variance"] == .5
    assert result["omics_tanh_saturation_fraction"] == .5
    absent = slot_monitoring_metrics(slots, risk, torch.zeros(3, dtype=torch.bool), "omics")
    assert absent == {"omics_monitor_patient_count": 0.}


@pytest.mark.parametrize("scale", [1., 1e-250, 0.])
def test_independence_diagnostic_distinguishes_selective_plan_and_handles_tiny_zero_mass(scale):
    diagonal = torch.tensor([[.5, 0.], [0., .5]], dtype=torch.float64)[None, None, None] * scale
    independent = torch.full_like(diagonal, .25 * scale)
    selective = plan_monitoring_metrics(diagonal, torch.tensor([True]))
    baseline = plan_monitoring_metrics(independent, torch.tensor([True]))
    assert selective["plan_independence_l1_normalized"] == (1. if scale else 0.)
    assert baseline["plan_independence_l1_normalized"] == 0.
    assert selective["transport_mass_mean"] == scale
    assert selective["transport_zero_mass_fraction"] == (0. if scale else 1.)
    assert all(math.isfinite(value) for value in selective.values())


def test_pair_masks_exclude_absent_modality_and_local_readout_from_aggregation():
    plan = torch.tensor([[.5, 0.], [0., .5]])[None, None, None].repeat(2, 1, 1, 1, 1)
    slots = torch.tensor([[[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
    state = SimpleNamespace(wsi_slots=slots, omics_slots=slots,
                            wsi_mask=torch.tensor([[True], [False]]),
                            omics_mask=torch.tensor([[True], [True]]),
                            plans=plan, pair_available=torch.tensor([True, False]))
    local = torch.tensor([[[3., 4.]], [[1e6, 1e6]]])
    result = state_monitoring_metrics(state, torch.zeros(2, 2, 1), torch.zeros(2, 2, 1), local, torch.tensor(.1))
    assert result["wsi_monitor_patient_count"] == result["transport_monitor_patient_count"] == 1.
    assert result["omics_monitor_patient_count"] == 2.
    assert result["local_readout_norm"] == 5.
    assert result["residual_gain"] == pytest.approx(math.tanh(.1))
    assert result["wsi_slot_near_duplicate_fraction"] == 0.
    assert result["omics_slot_near_duplicate_fraction"] == .5


def test_forward_exports_mask_specific_epoch_weights_and_no_eval_label_losses():
    model = DCTV22(_config()).eval()
    inputs = _inputs()
    inputs["wsi_mask"][0] = False
    inputs["omics_mask"][2] = False
    inputs["c"][:] = float("nan")
    with torch.no_grad():
        model(**inputs)
    weights = model.last_monitoring_metric_weights
    assert weights.keys() == model.last_monitoring_metrics.keys()
    assert weights["wsi_slot_pair_dist_mean"] == 2.
    assert weights["omics_risk_response_variance"] == 2.
    assert weights["plan_independence_l1_normalized"] == weights["local_readout_norm"] == 1.
    assert weights["residual_gain"] == 3.
    assert model.last_monitoring_metrics["wsi_monitor_patient_count"] == 2.
    assert model.last_monitoring_metrics["omics_monitor_patient_count"] == 2.
    assert model.last_monitoring_metrics["transport_monitor_patient_count"] == 1.
    assert "global_nll" not in model.last_monitoring_metrics


def test_args_defaults_roundtrip_and_explicit_version_controls():
    defaults = DCTV22Config.from_args(SimpleNamespace(omic_sizes=(3, 4)))
    assert defaults.dim == 128 and defaults.dropout == .30
    assert defaults.euclidean_scale == pytest.approx(math.sqrt(128))
    assert DCTV22Config.from_dict(defaults.to_dict()) == defaults
    explicit = DCTV22Config.from_args(SimpleNamespace(omic_sizes=(3, 4), wsi_projection_dim=256,
                otehv2_dropout=.15, dct_v22_wsi_hidden_dim=1536, dct_v22_wsi_dropout=0.))
    assert explicit.dim == 256 and explicit.dropout == .15
    assert explicit.wsi_hidden_dim == 1536 and explicit.wsi_dropout == 0.
    sampled = DCTV22Config.from_args(SimpleNamespace(omic_sizes=(3, 4), dct_v22_relation_gradient_interval=10))
    assert sampled.relation_gradient_interval == 10


def test_absolute_patient_relation_gap_reveals_signed_cancellation():
    gap = torch.tensor([-1., 1.], requires_grad=True)
    metrics = relation_gap_metrics(gap)
    assert gap.mean() == 0.
    assert metrics["probe_nll_gap_abs"] == 1.
    assert metrics["probe_worse_fraction"] == .5
    assert all(not value.requires_grad for value in metrics.values())


def test_sampled_raw_relation_gradient_preserves_grad_buffers_and_final_backward():
    model = _training_model(_config(relation_gradient_interval=2))
    first = model.forward_with_losses(**_inputs(2))
    assert model.last_monitoring_metrics["relation_cost_gradient_sample_count"] == 0.
    assert "relation_cost_gradient_norm_raw" not in model.last_monitoring_metrics
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    output = model.forward_with_losses(**_inputs(2))
    assert model.last_monitoring_metrics["relation_cost_gradient_sample_count"] == 1.
    assert model.last_monitoring_metric_weights["relation_cost_gradient_norm_raw"] == 1.
    observed = model.last_monitoring_metrics["relation_cost_gradient_norm_raw"]
    assert math.isfinite(observed) and observed >= 0.
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, before[name], atol=0., rtol=0.)
    # An independent evaluation of the retained v2.1 raw relation loss must
    # yield the same cost gradient norm; neither call applies lambda or AMP.
    raw = DCTV21.relation_loss(model, model.last_state, _inputs(2)["y"], _inputs(2)["c"], 10)
    parameters = tuple(model.core.stage_pair_cost.parameters())
    gradients = torch.autograd.grad(raw, parameters, allow_unused=True)
    expected = sum(value.double().square().sum() for value in gradients if value is not None).sqrt()
    assert observed == pytest.approx(float(expected), rel=1e-5, abs=1e-12)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, before[name], atol=0., rtol=0.)
    output["loss"].backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert first["terms"]["relation_probe_nll_gap_abs"] >= first["terms"]["relation_probe_nll_gap"].abs()


def test_raw_gradient_monitor_includes_active_evidence_cost_without_readout_parameters():
    model = _training_model(_config(relation_gradient_interval=1, evidence_cost_weight=.2))
    inputs = _inputs(2)
    model(**dict(inputs, compute_aux=False))
    generator = torch.Generator().manual_seed(382)
    state = replace(model.last_state,
                    wsi_slots=torch.randn(model.last_state.wsi_slots.shape, generator=generator),
                    omics_slots=torch.randn(model.last_state.omics_slots.shape, generator=generator))
    model._monitoring_training_forward_count = 1
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    parameters = model._cost_monitoring_parameters()
    expected_ids = {id(parameter) for module in (model.core.stage_pair_cost, model.core.evidence_gate)
                    for parameter in module.parameters() if parameter.requires_grad}
    assert {id(parameter) for parameter in parameters} == expected_ids
    assert len(parameters) == len(expected_ids)
    assert not ({id(parameter) for parameter in model.event_hazard.parameters()} & expected_ids)
    assert not ({id(parameter) for parameter in model.core.event_encoder.parameters()} & expected_ids)
    model.relation_loss(state, inputs["y"], inputs["c"], 10)
    observed = model.last_relation_diagnostics["cost_gradient_norm_raw"]
    raw = DCTV21.relation_loss(model, state, inputs["y"], inputs["c"], 10)
    gradients = torch.autograd.grad(raw, parameters, allow_unused=True)
    reference = sum(value.double().square().sum() for value in gradients if value is not None).sqrt()
    evidence_ids = {id(parameter) for parameter in model.core.evidence_gate.parameters()}
    assert any(value is not None and value.abs().sum() > 1e-9
               for parameter, value in zip(parameters, gradients) if id(parameter) in evidence_ids)
    torch.testing.assert_close(observed.double(), reference, rtol=1e-6, atol=1e-12)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, before[name], atol=0., rtol=0.)


@pytest.mark.parametrize("skip", ["warmup", "missing_pair", "eval"])
def test_relation_gradient_monitoring_skips_inactive_or_eval_and_clears_stale_values(skip):
    model = _training_model(_config(relation_gradient_interval=1))
    model(**_inputs(2))
    assert "relation_cost_gradient_norm_raw" in model.last_monitoring_metrics
    inputs = _inputs(2)
    if skip == "warmup":
        model.config = replace(model.config, relation_warmup_epochs=20)
    elif skip == "missing_pair":
        inputs["wsi_mask"][:] = False
    else:
        model.eval()
    with torch.set_grad_enabled(skip != "eval"):
        model(**inputs)
    assert "relation_cost_gradient_norm_raw" not in model.last_monitoring_metrics
    assert "cost_gradient_norm_raw" not in model.last_relation_diagnostics
    if skip != "eval":
        assert model.last_monitoring_metrics["relation_cost_gradient_sample_count"] == 0.


@pytest.mark.parametrize("change", [{"wsi_hidden_dim":0}, {"wsi_hidden_dim":True},
                                    {"wsi_dropout":1.}, {"wsi_dropout":float("nan")},
                                    {"wsi_dropout":True}, {"relation_gradient_interval":-1},
                                    {"relation_gradient_interval":True}])
def test_invalid_capacity_controls_rejected(change):
    with pytest.raises(ValueError):
        _config(**change)
