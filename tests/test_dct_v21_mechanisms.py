"""Independent mechanism checks for DCT v2.1; no patient performance claims.

The references below are direct covariance arithmetic, ordinary Euclidean
distances, finite differences, and explicit counterexamples to the v2 rules.
"""
from dataclasses import replace
from io import BytesIO
import math

import pytest
import torch
from torch.nn import functional as F

from survot_rank.research.methods.dct_v2 import DCTV2, DCTV2Config
from survot_rank.research.methods.dct_v2.losses import logit_nll
from survot_rank.research.methods.dct_v2.transport import (
    fixed_realized_marginal_plan, realized_independent_plan, semantic_slots,
)
from survot_rank.research.methods.dct_v21 import DCTV21, DCTV21Config
from survot_rank.research.methods.dct_v21.objectives import (
    centered_pair_coactivation, collision_safe_separation,
)


@pytest.fixture(autouse=True)
def single_thread_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**changes):
    return replace(DCTV21Config(
        dim=8, wsi_input_dim=6, omic_sizes=(3, 4), num_classes=3,
        wsi_slots=2, omics_slots=2, stages=2, heads=2,
        event_layers=1, slot_iters=1, slot_heads=2, slot_dim_head=4,
        slot_hidden_dim=16, dropout=0., relation_warmup_epochs=0,
        relation_ramp_epochs=1,
    ), **changes)


def _inputs(batch=3):
    generator = torch.Generator().manual_seed(918)
    return dict(
        x_wsi=torch.randn(batch, 5, 6, generator=generator),
        x_omic1=torch.randn(batch, 3, generator=generator),
        x_omic2=torch.randn(batch, 4, generator=generator),
        wsi_mask=torch.ones(batch, 5, dtype=torch.bool),
        omics_mask=torch.ones(batch, 2, dtype=torch.bool),
        y=torch.arange(batch) % 3,
        c=(torch.arange(batch) % 2).float(),
        event_time=torch.arange(1, batch + 1, dtype=torch.float32), epoch=10,
    )


def _model(config=None, training=True):
    torch.manual_seed(715)
    model = DCTV21(config or _config())
    model.set_training_reference(torch.arange(1, 7).float(),
                                 torch.tensor([0., 1., 0., 1., 0., 0.]))
    return model.train(training)


def _assert_finite_gradients(model):
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    assert gradients
    for name, value in gradients:
        assert torch.isfinite(value).all(), name


def _four_dim_plan(matrix, dtype=torch.float64):
    return torch.tensor(matrix, dtype=dtype)[None, None, None]


@pytest.mark.parametrize("matrix, sign", [
    ([[.5, 0.], [0., .5]], 1),
    ([[0., .5], [.5, 0.]], -1),
    ([[.25, .25], [.25, .25]], 0),
])
def test_uniform_budget_diagonal_antidiagonal_and_independent_are_distinguishable(matrix, sign):
    p = _four_dim_plan(matrix)
    risk = torch.tensor([[[-1.], [1.]]], dtype=p.dtype)
    actual = centered_pair_coactivation(p, risk, risk)
    expected = sign * math.tanh(1.) ** 2
    torch.testing.assert_close(actual, actual.new_full((1, 1, 1), expected), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(p.sum(-1), p.new_full((1, 1, 1, 2), .5))
    torch.testing.assert_close(p.sum(-2), p.new_full((1, 1, 1, 2), .5))


def test_centered_pair_equals_direct_mass_weighted_covariance_with_nonuniform_budgets():
    generator = torch.Generator().manual_seed(818)
    p = torch.rand(2, 2, 3, 3, 4, generator=generator, dtype=torch.float64) * .07
    wl = torch.randn(2, 3, 2, generator=generator, dtype=p.dtype)
    ol = torch.randn(2, 4, 2, generator=generator, dtype=p.dtype)
    expected = torch.zeros(2, 2, 2, dtype=p.dtype)
    for b in range(2):
        for s in range(2):
            for g in range(3):
                mass = p[b, s, g].sum()
                probability = p[b, s, g] / mass
                wf, of = wl[b].tanh(), ol[b].tanh()
                mean_w = (probability.sum(1)[:, None] * wf).sum(0)
                mean_o = (probability.sum(0)[:, None] * of).sum(0)
                for w in range(3):
                    for o in range(4):
                        expected[b, s] += mass * probability[w, o] * (wf[w] - mean_w) * (of[o] - mean_o) / 3
    torch.testing.assert_close(centered_pair_coactivation(p, wl, ol), expected, atol=2e-16, rtol=1e-12)


@pytest.mark.parametrize("mass_aware", [True, False])
def test_plan_quality_scales_centered_pair_once_or_is_explicitly_normalized(mass_aware):
    p = _four_dim_plan([[.5, 0.], [0., .5]])
    risk = torch.tensor([[[-.9], [.7]]], dtype=p.dtype)
    base = centered_pair_coactivation(p, risk, risk, mass_aware=mass_aware)
    scaled = centered_pair_coactivation(.2 * p, risk, risk, mass_aware=mass_aware)
    torch.testing.assert_close(scaled, (.2 if mass_aware else 1.) * base, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("boundary", ["zero", "single_wsi", "single_omics", "constant_wsi", "constant_omics"])
def test_no_pair_information_yields_zero_centered_readout(boundary):
    p = _four_dim_plan([[.1, .3], [.2, .4]])
    wl = torch.tensor([[[-1.], [2.]]], dtype=p.dtype)
    ol = torch.tensor([[[2.], [-.5]]], dtype=p.dtype)
    if boundary == "zero":
        p.zero_()
    elif boundary == "single_wsi":
        p = p[..., :1, :]
        wl = wl[:, :1]
    elif boundary == "single_omics":
        p = p[..., :1]
        ol = ol[:, :1]
    elif boundary == "constant_wsi":
        wl[:] = .4
    else:
        ol[:] = .4
    torch.testing.assert_close(centered_pair_coactivation(p, wl, ol), p.new_zeros(1, 1, 1), atol=1e-15, rtol=0.)


@pytest.mark.parametrize("dtype,mass", [(torch.float16, 1e-4), (torch.bfloat16, 1e-30),
                                       (torch.float32, 1e-30), (torch.float64, 1e-200)])
def test_tiny_and_zero_quality_keep_readout_and_gradients_finite(dtype, mass):
    p = _four_dim_plan([[.5, .1], [.1, .5]], dtype=dtype).mul(mass).requires_grad_()
    wl = torch.tensor([[[-.5], [.5]]], dtype=dtype, requires_grad=True)
    ol = wl.detach().clone().requires_grad_()
    result = centered_pair_coactivation(p, wl, ol)
    assert torch.isfinite(result).all()
    result.sum().backward()
    for gradient in (p.grad, wl.grad, ol.grad):
        assert torch.isfinite(gradient).all()
    zero = centered_pair_coactivation(torch.zeros_like(p), wl, ol)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_collision_separation_value_is_actual_l2_hinge_not_squared_distance():
    slots = torch.tensor([[[0., 0.], [.3, .4], [2., 0.]]], dtype=torch.float64)
    expected = sum(max(0., 1. - math.dist(slots[0, i], slots[0, j]))
                   for i, j in ((0, 1), (0, 2), (1, 2))) / 3
    torch.testing.assert_close(collision_safe_separation(slots), slots.new_tensor(expected), atol=1e-14, rtol=1e-14)
    assert collision_safe_separation(slots, margin=0.) == 0.


@pytest.mark.parametrize("count,dim", [(2, 2), (3, 4), (7, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_collision_legal_direction_is_finite_balanced_and_lowers_true_loss(count, dim, dtype):
    slots = torch.zeros(1, count, dim, dtype=dtype, requires_grad=True)
    before = collision_safe_separation(slots)
    before.backward()
    gradient = slots.grad
    assert gradient.abs().sum() > 0.
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient.float().sum(1), torch.zeros(1, dim), atol=.002, rtol=0.)
    # A selected generalized gradient of -||x|| is legal at x=0. A small
    # descent step must lower the *forward* L2 hinge, including K>D cases.
    after = collision_safe_separation(slots.detach() - .05 * gradient)
    assert after < before
    old_slots = slots.detach().clone().requires_grad_()
    differences = old_slots[:, :, None] - old_slots[:, None, :]
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    old_loss = F.relu(1. - differences.square().sum(-1)[:, off_diagonal]).mean()
    old_loss.backward()
    assert old_loss > 0.
    assert torch.equal(old_slots.grad, torch.zeros_like(old_slots))


def test_noncollision_gradient_matches_finite_differences_and_exact_one_slot_has_zero_constraint():
    slots = torch.tensor([[[.12, .01], [.43, .17], [.19, .61]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s: collision_safe_separation(s, .9), (slots,), eps=1e-6, atol=1e-6)
    single = torch.randn(2, 1, 3, requires_grad=True)
    collision_safe_separation(single).backward()
    assert torch.equal(single.grad, torch.zeros_like(single))


def test_identical_prototypes_can_receive_separating_gradient_with_nondegenerate_tokens():
    tokens = torch.tensor([[[1., 0., .2], [0., 2., -.1], [-1., .2, .5]]], dtype=torch.float64)
    prototypes = torch.tensor([[.2, .5, -.3], [.2, .5, -.3]], dtype=torch.float64, requires_grad=True)
    slots = semantic_slots(tokens, prototypes, .3)
    torch.testing.assert_close(slots[:, 0], slots[:, 1], atol=0., rtol=0.)
    collision_safe_separation(slots).backward()
    assert torch.isfinite(prototypes.grad).all()
    assert prototypes.grad.abs().sum() > 1e-6
    assert not torch.equal(prototypes.grad[0], prototypes.grad[1])


def test_identical_token_counterexample_prevents_claiming_universal_collapse_escape():
    # If every token is identical, changing semantic assignments cannot change
    # the weighted mean. The objective cannot repair an upstream zero Jacobian.
    tokens = torch.tensor([[[1., .5, -.2], [1., .5, -.2]]], dtype=torch.float64)
    prototypes = torch.tensor([[.2, .5, -.3], [.2, .5, -.3]], dtype=torch.float64, requires_grad=True)
    slots = semantic_slots(tokens, prototypes, .3)
    collision_safe_separation(slots).backward()
    assert prototypes.grad.abs().max() < 1e-12


def test_contrast_is_active_at_independent_equality_with_zero_deployment_gain():
    # Zero main gain would kill the old auxiliary plan-to-loss gradient. The
    # new probe uses fixed positive gain, and softplus has slope 1/2 at gap=0.
    costs = torch.zeros(1, 1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    budgets = costs.new_full((1, 1, 1, 2), .5)
    p = fixed_realized_marginal_plan(costs, budgets, budgets, epsilon=.2, tolerance=1e-12)
    q = costs.new_full((1, 1, 1, 2, 2), .25)
    torch.testing.assert_close(p, q, atol=1e-12, rtol=0.)
    risk = costs.new_tensor([[[-1.], [1.]]])
    main_gain, probe_gain, temperature = 0., .1, .05
    main = .2 + main_gain * centered_pair_coactivation(p, risk, risk)
    assert torch.equal(main, main.new_full(main.shape, .2))
    learned = (.2 + probe_gain * centered_pair_coactivation(p, risk, risk)).squeeze(1)
    independent = (.2 + probe_gain * centered_pair_coactivation(q, risk, risk)).squeeze(1)
    y, c = torch.tensor([0]), torch.tensor([0.], dtype=costs.dtype)
    gap = logit_nll(learned, y, c, reduction="none") - logit_nll(independent, y, c, reduction="none").detach()
    objective = temperature * F.softplus(gap / temperature).mean()
    torch.testing.assert_close(objective, costs.new_tensor(temperature * math.log(2)), atol=1e-12, rtol=0.)
    objective.backward()
    assert costs.grad.abs().sum() > 1e-6
    assert torch.isfinite(costs.grad).all()
    assert budgets.grad is None


def test_v21_adds_no_parameters_and_has_its_own_serialized_recipe():
    new = _model()
    values = {k: v for k, v in new.config.to_dict().items() if k in DCTV2Config.__dataclass_fields__}
    old = DCTV2(DCTV2Config(**values))
    assert [(n, p.shape) for n, p in new.named_parameters()] == [(n, p.shape) for n, p in old.named_parameters()]
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters())
    assert DCTV21Config.from_dict(new.config.to_dict()) == new.config
    assert new.get_extra_state()["schema"] != old.get_extra_state()["schema"]
    with pytest.raises(ValueError, match="schema"):
        new.load_state_dict(old.state_dict())
    with pytest.raises(ValueError, match="schema"):
        old.load_state_dict(new.state_dict())


def test_checkpoint_roundtrip_retains_predictions_reference_and_rejects_recipe_change():
    model = _model(training=False)
    inputs = _inputs()
    with torch.no_grad():
        expected, _ = model(**inputs)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=False)
    restored = DCTV21(_config()).eval()
    restored.load_state_dict(state)
    with torch.no_grad():
        actual, _ = restored(**inputs)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    torch.testing.assert_close(restored.censor_survival, model.censor_survival)
    with pytest.raises(ValueError, match="exact saved recipe"):
        DCTV21(_config(relation_probe_gain=.2)).load_state_dict(state)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_full_six_term_sum_and_finite_backward(mixed_precision):
    model = _model()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        output = model.forward_with_losses(**_inputs(2))
    terms = output["terms"]
    auxiliary_names = ["ipcw_rank", "per_slot_nll", "slot_diversity", "relation", "native"]
    expected_auxiliary = sum(terms[name + "_weighted"] for name in auxiliary_names)
    torch.testing.assert_close(output["auxiliary"], expected_auxiliary)
    torch.testing.assert_close(output["loss"], terms["global_nll"] + expected_auxiliary)
    assert not any("recon" in name for name in terms)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["logits"]).all()
    output["loss"].backward()
    _assert_finite_gradients(model)


def test_relation_probe_cost_gradient_survives_zero_main_gain_and_freezes_all_other_parameters():
    model = _model(_config(residual_init=0.))
    model.sig_networks[0].eval()
    modes = {name: module.training for name, module in model.named_modules()}
    with torch.no_grad():
        model(**{**_inputs(), "compute_aux": False})
    state = model.last_state
    # Distinct slot evidence makes the test independent of accidentally near-
    # identical randomly initialized semantic prototypes.
    generator = torch.Generator().manual_seed(382)
    state = replace(state,
                    wsi_slots=torch.randn(state.wsi_slots.shape, generator=generator),
                    omics_slots=torch.randn(state.omics_slots.shape, generator=generator))
    model.zero_grad(set_to_none=True)
    relation = model.relation_loss(state, _inputs()["y"], _inputs()["c"], 10)
    assert relation > 0.
    relation.backward()
    allowed = "core.stage_pair_cost."
    assert any(p.grad is not None and p.grad.abs().sum() > 1e-9
               for name, p in model.named_parameters() if name.startswith(allowed))
    for name, p in model.named_parameters():
        if not name.startswith(allowed):
            assert p.grad is None or not p.grad.any(), name
    diagnostics = model.last_relation_diagnostics
    assert diagnostics["probe_gain"] == pytest.approx(.1)
    assert diagnostics["row_error"] < 1e-6
    assert diagnostics["col_error"] < 1e-6
    assert modes == {name: module.training for name, module in model.named_modules()}
    _assert_finite_gradients(model)


def test_independent_control_does_not_train_hidden_topology():
    model = _model(_config(plan_mode="independent"))
    output = model.forward_with_losses(**_inputs())
    assert output["terms"]["relation"] == 0.
    torch.testing.assert_close(model.last_state.plans, realized_independent_plan(model.last_state.plans))
    assert torch.all(model.last_local_readout.abs() < 1e-6)


def test_zero_actual_transport_is_exact_native_bypass_even_with_biased_readout(monkeypatch):
    model = _model(training=False)
    with torch.no_grad():
        model.event_hazard.bias.fill_(3.)
        model.per_slot_hazard_wsi.bias.fill_(1.)
    monkeypatch.setattr(model.core, "plans_from_costs", lambda costs, rows, cols, epoch:
                        (torch.zeros_like(costs), rows, cols))
    with torch.no_grad():
        logits, auxiliary = model(**_inputs())
    assert auxiliary == 0.
    assert not model.last_state.plans.any()
    assert not model.last_local_readout.any()
    torch.testing.assert_close(logits.sigmoid(), model.last_native_hazards, atol=0., rtol=0.)


def test_masks_eval_and_training_only_reference_protect_information_boundary():
    model = DCTV21(_config())
    with pytest.raises(RuntimeError, match="training-fold censor reference"):
        model(**_inputs())
    model.eval()
    inputs = _inputs()
    inputs["wsi_mask"][:, -2:] = False
    inputs["omics_mask"][:, -1] = False
    before = {name: value.clone() for name, value in model.named_buffers()}
    with torch.no_grad():
        first, auxiliary = model(**inputs)
        modified = dict(inputs)
        modified["x_wsi"] = inputs["x_wsi"].clone()
        modified["x_wsi"][~inputs["wsi_mask"]] = 1e6
        modified["x_omic2"] = torch.full_like(inputs["x_omic2"], 1e6)
        modified["c"] = torch.full_like(inputs["c"], float("nan"))
        modified["event_time"] = torch.full_like(inputs["event_time"], float("nan"))
        second, second_auxiliary = model(**modified)
    assert auxiliary == 0. and second_auxiliary == 0.
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    assert model._rank_memory is None
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_explicit_legacy_choices_reproduce_v2_for_matched_controls():
    new = _model(_config(local_readout="legacy_mean", relation_objective="legacy_hinge",
                         diversity_objective="legacy_squared"), training=False)
    values = {k: v for k, v in new.config.to_dict().items() if k in DCTV2Config.__dataclass_fields__}
    old = DCTV2(DCTV2Config(**values)).eval()
    state = new.state_dict()
    state["_extra_state"] = old.get_extra_state()
    old.load_state_dict(state)
    inputs = _inputs()
    with torch.no_grad():
        newer, _ = new(**inputs)
        older, _ = old(**inputs)
    torch.testing.assert_close(newer, older, atol=0., rtol=0.)
    torch.testing.assert_close(new.last_state.plans, old.last_state.plans, atol=0., rtol=0.)
    ipcw = torch.ones(3)
    slot_new, diversity_new = new._slot_objectives(new.last_state, inputs["y"], inputs["c"], ipcw)
    slot_old, diversity_old = old._slot_objectives(old.last_state, inputs["y"], inputs["c"], ipcw)
    torch.testing.assert_close(slot_new, slot_old, atol=0., rtol=0.)
    torch.testing.assert_close(diversity_new, diversity_old, atol=0., rtol=0.)
    torch.testing.assert_close(new.relation_loss(new.last_state, inputs["y"], inputs["c"], 10),
                               old.relation_loss(old.last_state, inputs["y"], inputs["c"], 10), atol=0., rtol=0.)


@pytest.mark.parametrize("support", ["zero", "single_wsi", "single_omics", "tiny"])
def test_relation_has_no_constant_pressure_without_matching_freedom(support):
    model = _model()
    inputs = _inputs()
    model(**dict(inputs, compute_aux=False))
    state = model.last_state
    plans = state.plans.clone()
    if support == "zero":
        plans = torch.zeros_like(plans)
    elif support == "single_wsi":
        plans[..., 1:, :] = 0.
    elif support == "single_omics":
        plans[..., :, 1:] = 0.
    else:
        plans = plans*1e-14
    boundary = replace(state, plans=plans)
    modes = {name: module.training for name, module in model.named_modules()}
    loss = model.relation_loss(boundary, inputs["y"], inputs["c"], 10)
    assert loss.item() == 0.
    assert model.last_relation_diagnostics["effective_mass"].item() == 0.
    assert modes == {name: module.training for name, module in model.named_modules()}


def test_relation_restores_mixed_module_modes_on_projection_failure(monkeypatch):
    import survot_rank.research.methods.dct_v21.model as module
    model = _model()
    inputs = _inputs()
    model(**dict(inputs, compute_aux=False))
    model.core.fusion.eval()
    modes = {name: item.training for name, item in model.named_modules()}
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate projection failure")
    monkeypatch.setattr(module, "fixed_realized_marginal_plan", fail)
    with pytest.raises(RuntimeError, match="deliberate projection failure"):
        model.relation_loss(model.last_state, inputs["y"], inputs["c"], 10)
    assert modes == {name: item.training for name, item in model.named_modules()}
