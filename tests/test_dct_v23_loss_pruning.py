"""Synthetic objective-pruning contracts and structural counterexamples.

These checks establish implementation behavior and possible ambiguities; they
do not measure patient benefit, observed gradient conflict, or generalization.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from survot_rank.research.methods.dct_v2 import model as v2_model
from survot_rank.research.methods.dct_v2.losses import logit_nll
from survot_rank.research.methods.dct_v21.model import DCTV21
from survot_rank.research.methods.dct_v23 import DCTV23, DCTV23Config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(**overrides):
    values = dict(omic_sizes=(2, 3), dim=8, wsi_input_dim=3,
        wsi_hidden_dim=4, num_classes=3, wsi_slots=2, omics_slots=2,
        stages=2, heads=2, event_layers=1, slot_iters=1, slot_heads=2,
        slot_dim_head=4, slot_hidden_dim=16, dropout=0., wsi_dropout=0.,
        reconstruction_mode="off")
    values.update(overrides)
    return DCTV23Config(**values)


def batch():
    generator = torch.Generator().manual_seed(29)
    return dict(x_wsi=torch.randn(2, 4, 3, generator=generator),
        x_omic1=torch.randn(2, 2, generator=generator),
        x_omic2=torch.randn(2, 3, generator=generator),
        y=torch.tensor([0, 2]), event_time=torch.tensor([1., 3.]),
        c=torch.tensor([0., 1.]), epoch=20)


@pytest.mark.parametrize("mode,current_count,legacy_count", [
    ("off", 2, 6), ("masked_plain", 3, 7), ("gain_calibrated", 4, 8)])
def test_active_terms_reflect_current_defaults_and_explicit_historical_recipe(mode, current_count, legacy_count):
    current = DCTV23(tiny(reconstruction_mode=mode)).get_extra_state()
    historical = DCTV23(tiny(reconstruction_mode=mode,
        ipcw_weight=.1, slot_weight=.05, relation_weight=.05, native_weight=.05,
        local_readout="centered_pair")).get_extra_state()
    assert len(current["losses"]) == current_count
    assert len(historical["losses"]) == legacy_count
    assert current["losses"] == current["loss_terms"]
    assert set(historical["losses"]) - set(current["losses"]) == {"ipcw_rank", "per_slot_nll", "relation", "native"}
    assert current["configured_loss_weights"]["ipcw_rank"] == 0.
    assert current["configured_loss_weights"]["per_slot_nll"] == 0.
    assert current["configured_loss_weights"]["relation"] == 0.
    assert current["configured_loss_weights"]["native"] == 0.
    assert current["config"]["local_readout"] == "none"
    assert historical["configured_loss_weights"]["ipcw_rank"] == .1
    assert historical["configured_loss_weights"]["per_slot_nll"] == .05


def test_absent_argument_weights_default_to_zero_but_explicit_history_is_preserved():
    absent = DCTV23Config.from_args(SimpleNamespace(omic_sizes=(2, 3)))
    explicit = DCTV23Config.from_args(SimpleNamespace(omic_sizes=(2, 3),
        dct_v2_lambda_ipcw_rank=.1, dct_v2_lambda_slot_nll=.05,
        dct_v2_lambda_relation=.05, dct_v2_lambda_native=.05,
        dct_v21_local_readout="centered_pair"))
    assert absent.ipcw_weight == absent.slot_weight == absent.relation_weight == absent.native_weight == 0.
    assert absent.local_readout == "none"
    assert explicit.ipcw_weight == .1 and explicit.slot_weight == .05
    assert explicit.relation_weight == explicit.native_weight == .05
    assert explicit.local_readout == "centered_pair"
    assert DCTV23Config.from_dict(explicit.to_dict()) == explicit


def test_real_forward_skips_inherited_slot_nll_and_never_fills_rank_memory(monkeypatch):
    model = DCTV23(tiny(relation_weight=0., native_weight=0.)).train()
    def forbidden(*args, **kwargs):
        pytest.fail("disabled ranking or inherited slot NLL was evaluated")
    monkeypatch.setattr(DCTV21, "_slot_objectives", forbidden)
    monkeypatch.setattr(v2_model, "inverse_censor_weight", forbidden)
    monkeypatch.setattr(v2_model, "ipcw_pairwise_ranking_loss", forbidden)
    assert model._rank_memory is None
    for epoch in (0, 1):
        inputs = dict(batch(), epoch=epoch)
        logits, auxiliary = model(**inputs)
        (logit_nll(logits, inputs["y"], inputs["c"]) + auxiliary).backward()
        assert torch.isfinite(logits).all() and torch.isfinite(auxiliary)
        assert model.last_training_losses["per_slot_nll"] == 0.
        assert model.last_training_losses["ipcw_rank"] == 0.
        assert model._rank_memory is None


@pytest.mark.parametrize("objective", ["collision_distance", "legacy_squared"])
@pytest.mark.parametrize("missing", [None, "wsi", "omics"])
def test_retained_diversity_value_and_gradients_match_historical_path(objective, missing):
    torch.manual_seed(17)
    model = DCTV23(tiny(diversity_objective=objective))
    state = SimpleNamespace(wsi_slots=(torch.randn(2, 2, 8)*.1).requires_grad_(),
        omics_slots=(torch.randn(2, 2, 8)*.1).requires_grad_(),
        wsi_mask=torch.ones(2, 4, dtype=torch.bool),
        omics_mask=torch.ones(2, 2, dtype=torch.bool))
    if missing is not None:
        getattr(state, missing + "_mask").fill_(False)
    labels, censors, ipcw = torch.tensor([0, 2]), torch.tensor([0., 1.]), torch.ones(2)
    zero, current = model._slot_objectives(state, labels, censors, ipcw)
    historical_nll, historical = DCTV21._slot_objectives(model, state, labels, censors, ipcw)
    assert zero == 0. and historical_nll > 0.
    assert torch.allclose(current, historical, rtol=1e-6, atol=1e-7)
    leaves = (state.wsi_slots, state.omics_slots,
        model.per_slot_hazard_wsi.weight, model.per_slot_hazard_wsi.bias,
        model.per_slot_hazard_omic.weight, model.per_slot_hazard_omic.bias)
    current_grad = torch.autograd.grad(current, leaves, retain_graph=True, allow_unused=True)
    historical_grad = torch.autograd.grad(historical, leaves, allow_unused=True)
    for actual, expected in zip(current_grad, historical_grad):
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-7)
    assert any(value is not None and value.abs().sum() > 0 for value in current_grad)


@pytest.mark.parametrize("mode", ["off", "masked_plain", "gain_calibrated"])
def test_both_slot_heads_can_flip_sign_without_changing_predictions_or_remaining_objectives(mode):
    torch.manual_seed(31)
    # Preserve the earlier structural audit of the explicit historical readout.
    model = DCTV23(tiny(reconstruction_mode=mode, local_readout="centered_pair",
                       relation_weight=.05, native_weight=.05)).train()
    inputs = batch()
    if mode != "off":
        model.configure_reconstruction_reference((("a", "b"), ("c", "d", "e")),
            [0.]*5, [1.]*5, ["synthetic_0", "synthetic_1"])
        inputs.update(patient_ids=["synthetic_0", "synthetic_1"],
            reconstruction_global_mask=torch.tensor([[True, False, True, False, False],
                [False, True, False, False, True]]))
    if mode == "gain_calibrated":
        # Direct synthetic labels isolate the objective algebra; provenance and
        # patient-cohort acceptance are tested by the separate workflow suite.
        model._gain_targets = {"synthetic_0": [.1, -.1], "synthetic_1": [-.2, .2]}
    # Couple stochastic slot initialization/dropout across the two forwards.
    torch.manual_seed(73)
    before_logits, before_auxiliary = model(**inputs)
    before_terms = {key: value.clone() for key, value in model.last_training_losses.items()
        if key in model.get_extra_state()["losses"]}
    with torch.no_grad():
        for head in (model.per_slot_hazard_wsi, model.per_slot_hazard_omic):
            head.weight.neg_()
            head.bias.neg_()
    torch.manual_seed(73)
    after_logits, after_auxiliary = model(**inputs)
    assert torch.allclose(after_logits, before_logits, rtol=1e-5, atol=1e-6)
    assert torch.allclose(after_auxiliary, before_auxiliary, rtol=1e-5, atol=1e-6)
    for key, before in before_terms.items():
        assert torch.allclose(model.last_training_losses[key], before, rtol=1e-5, atol=1e-6), key


def test_zero_gain_calibration_penalizes_donor_stage_variation_despite_correct_receiver_means():
    model = DCTV23(tiny(reconstruction_mode="gain_calibrated"))
    model._gain_targets = {"synthetic_0": [0., 0.]}
    probability = torch.tensor([[[[.2, .8], [.2, .8]], [[.8, .2], [.8, .2]]]])
    class FixedCosts(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.logit(probability))
        def forward(self, contents):
            assert contents.shape[:3] == (1, 2, 2)
            return self.logits
    model.core.stage_pair_cost = FixedCosts()
    state = SimpleNamespace(wsi_slots=torch.zeros(1, 2, 8),
        omics_slots=torch.zeros(1, 2, 8), omics_mask=torch.ones(1, 2, dtype=torch.bool),
        pair_available=torch.ones(1, dtype=torch.bool))
    reconstruction = dict(state=state, assignment=torch.eye(2).unsqueeze(0))
    loss = model.gain_calibration_loss(reconstruction, ["synthetic_0"])
    predicted = model.last_reconstruction["calibration_predicted_cost"]
    goal = model.last_reconstruction["calibration_cost_goal"]
    assert torch.allclose(predicted.mean((1, 2)), goal, atol=1e-7)
    assert torch.allclose(goal, torch.full_like(goal, .5))
    assert torch.allclose(loss, torch.tensor(.09), atol=1e-7)
    loss.backward()
    gradient = model.core.stage_pair_cost.logits.grad
    assert torch.all(gradient[probability < .5] < 0)
    assert torch.all(gradient[probability > .5] > 0)


def test_native_and_main_nll_can_push_the_same_prediction_weight_in_opposite_directions():
    model = DCTV23(tiny(local_readout="none", native_weight=.05))
    slots = torch.zeros(1, 2, 8)
    slots[..., 0] = 1.
    events = torch.zeros(1, 2, 8)
    events[..., 0] = -16.
    state = SimpleNamespace(wsi_slots=slots, omics_slots=slots.clone(),
        wsi_mask=torch.ones(1, 4, dtype=torch.bool),
        omics_mask=torch.ones(1, 2, dtype=torch.bool),
        plans=torch.full((1, 2, 3, 2, 2), .125),
        row_marginals=torch.full((1, 2, 2), .5),
        col_marginals=torch.full((1, 2, 2), .5),
        stage_weights=torch.full((1, 2), .5),
        event_tokens=events, pair_available=torch.ones(1, dtype=torch.bool))
    with torch.no_grad():
        model.event_hazard.weight.zero_()
        model.event_hazard.weight[0, 0] = .3
        model.event_hazard.bias.zero_()
        model.residual_gain.copy_(torch.atanh(torch.tensor(.5)))
    main_logits, native_logits, _ = model._predict(state)
    assert torch.allclose(main_logits[:, 0], torch.tensor([-.9]), atol=1e-6)
    assert torch.allclose(native_logits[:, 0], torch.tensor([.3]), atol=1e-6)
    deficit = model.reference_deficit(state)
    assert torch.allclose(deficit, torch.tensor([.5]))
    y, c = torch.tensor([0]), torch.tensor([0.])
    main = logit_nll(main_logits, y, c)
    native = model.config.native_weight*(deficit*logit_nll(native_logits, y, c, reduction="none")).mean()
    main_gradient = torch.autograd.grad(main, model.event_hazard.weight, retain_graph=True)[0]
    native_gradient = torch.autograd.grad(native, model.event_hazard.weight)[0]
    assert main_gradient[0, 0] > 0 and native_gradient[0, 0] < 0
    assert (main_gradient*native_gradient).sum() < 0
