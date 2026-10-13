"""Synthetic math and integration checks for the independent DCT v2 model.

These checks verify objectives, masks, gradients and saved model state. They do
not estimate patient-level survival performance.
"""
from dataclasses import replace
from io import BytesIO
import math

import pytest
import torch

from survot_rank.research.methods.dct_v2 import DCTV2, DCTV2Config
from survot_rank.research.methods.dct_v2.losses import (
    StableNLLSurvLoss, censor_km, discrete_nll, inverse_censor_weight,
    ipcw_pairwise_ranking_loss, logit_nll,
)
from survot_rank.research.methods.dct_v2.transport import fixed_realized_marginal_plan
import survot_rank.research.methods.dct_v2.model as model_module


@pytest.fixture(autouse=True)
def single_thread_synthetic_models():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def _config(**changes):
    config = DCTV2Config(
        dim=8, wsi_input_dim=6, omic_sizes=(3, 4), num_classes=3,
        wsi_slots=2, omics_slots=2, stages=2, heads=2,
        event_layers=1, slot_iters=1, slot_heads=2, slot_dim_head=4,
        slot_hidden_dim=16, dropout=0., residual_init=.25,
        relation_warmup_epochs=0, relation_ramp_epochs=1,
    )
    return replace(config, **changes)


def _inputs(batch=3, padded=False):
    gen = torch.Generator().manual_seed(913)
    x_wsi = torch.randn(batch, 5, 6, generator=gen)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    if padded:
        mask[:, -2:] = False
    return {
        "x_wsi": x_wsi,
        "x_omic1": torch.randn(batch, 3, generator=gen),
        "x_omic2": torch.randn(batch, 4, generator=gen),
        "wsi_mask": mask,
        "omics_mask": torch.ones(batch, 2, dtype=torch.bool),
        "y": torch.arange(batch) % 3,
        "event_time": torch.arange(1, batch + 1, dtype=torch.float32),
        "c": (torch.arange(batch) % 2).float(),
        "epoch": 10,
    }


def _model(config=None, training=True):
    torch.manual_seed(712)
    model = DCTV2(config or _config())
    model.set_training_reference(
        torch.tensor([1., 2., 3., 4., 5., 6.]),
        torch.tensor([0., 1., 0., 1., 0., 0.]),
    )
    return model.train(training)


def _finite_gradients(model):
    grads = [(name, parameter.grad) for name, parameter in model.named_parameters()
             if parameter.grad is not None]
    assert grads, "a training objective must reach model parameters"
    for name, gradient in grads:
        assert torch.isfinite(gradient).all(), name


def test_config_roundtrip_preserves_architecture_and_six_term_recipe():
    config = _config()
    assert DCTV2Config.from_dict(config.to_dict()) == config
    assert config.ipcw_weight == .1
    assert config.slot_weight == .05
    assert config.diversity_weight == .1
    assert config.relation_weight == .05
    assert config.native_weight == .05


@pytest.mark.parametrize("changes", [
    {"dim": 7}, {"omic_sizes": ()}, {"wsi_input_dim": 0},
    {"relation_weight": float("nan")}, {"native_weight": -.1},
    {"ipcw_weight": True}, {"transport_mode": "plugin"},
])
def test_config_rejects_undefined_architecture_or_loss_recipe(changes):
    with pytest.raises((ValueError, TypeError)):
        _config(**changes)


@pytest.mark.parametrize("epoch,expected", [(2, 0.), (3, .01), (4, .02), (7, .05), (8, .05)])
def test_relation_warmup_is_zero_based_and_bounded(epoch, expected):
    config = _config(relation_warmup_epochs=3, relation_ramp_epochs=5)
    assert config.relation_effective_weight(epoch) == pytest.approx(expected)


@pytest.mark.parametrize("c,expected", [
    (0., -math.log(.8 * .3)),
    (1., -math.log(.8 * .7)),
])
def test_discrete_nll_uses_cumulative_survival_to_selected_bin(c, expected):
    hazards = torch.tensor([[.2, .3, .4]], dtype=torch.float64, requires_grad=True)
    actual = discrete_nll(hazards, torch.tensor([1]), torch.tensor([c]), alpha=0.)
    torch.testing.assert_close(actual, hazards.new_tensor(expected), atol=1e-12, rtol=1e-12)
    actual.backward()
    assert torch.isfinite(hazards.grad).all()
    assert hazards.grad[0, 2] == 0., "later bins must not enter the selected likelihood"


def test_discrete_nll_reductions_and_first_bin_event():
    hazards = torch.tensor([[.2, .3, .4], [.2, .3, .4]], dtype=torch.float64)
    labels, censor = torch.tensor([0, 2]), torch.tensor([0., 1.])
    expected = hazards.new_tensor([-math.log(.2), -math.log(.8 * .7 * .6)])
    torch.testing.assert_close(discrete_nll(hazards, labels, censor, reduction="none"), expected)
    torch.testing.assert_close(discrete_nll(hazards, labels, censor, reduction="mean"), expected.mean())
    torch.testing.assert_close(discrete_nll(hazards, labels, censor, reduction="sum"), expected.sum())


def test_probability_and_stable_logit_nll_agree_and_trainer_divides_batch_once():
    hazards = torch.tensor([[.2, .3, .4], [.7, .1, .2]], dtype=torch.float64)
    logits = hazards.logit()
    labels, censor = torch.tensor([1, 2]), torch.tensor([0., 1.])
    expected = discrete_nll(hazards, labels, censor)
    torch.testing.assert_close(logit_nll(logits, labels, censor), expected)
    criterion = StableNLLSurvLoss(alpha=0.)
    torch.testing.assert_close(criterion(logits, labels, None, censor) / 2, expected)


def test_slot_nll_uses_each_slot_and_patient_cumulative_survival():
    hazards = torch.tensor([[[.2, .3, .4], [.6, .1, .2]],
                            [[.7, .2, .1], [.3, .5, .4]]], dtype=torch.float64)
    labels, censor = torch.tensor([1, 2]), torch.tensor([0., 1.])
    expected = hazards.new_tensor([
        [-math.log(.8 * .3), -math.log(.4 * .1)],
        [-math.log(.3 * .8 * .9), -math.log(.7 * .5 * .6)],
    ])
    torch.testing.assert_close(discrete_nll(hazards, labels, censor, reduction="none"), expected)
    torch.testing.assert_close(logit_nll(hazards.logit(), labels, censor, reduction="none"), expected)


def test_censor_weights_use_training_km_left_limit_at_tied_times():
    times, survival = censor_km(torch.tensor([1., 1., 2., 3.]), torch.tensor([0., 1., 0., 1.]))
    torch.testing.assert_close(times, torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(survival, torch.tensor([.75, .75, 0.]))
    queries = torch.tensor([1., 2., 3., 4.])
    weights = inverse_censor_weight(queries, times, survival, power=2., max_weight=10.)
    torch.testing.assert_close(weights, torch.tensor([1., 16/9, 16/9, 10.]))


def test_ipcw_ranking_excludes_ties_and_censored_early_patients():
    logits = torch.tensor([[.1, .2, .3], [.5, .4, .2], [.7, .8, .9]], requires_grad=True)
    loss, count = ipcw_pairwise_ranking_loss(
        logits, torch.tensor([1., 1., 2.]), torch.tensor([1., 0., 0.]), torch.ones(3),
    )
    assert count == 1
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[0].any(), "the earlier censored patient does not define an event pair"
    assert logits.grad[1:].abs().sum() > 0.


def test_ipcw_ranking_never_trains_on_memory_memory_pairs():
    logits = torch.tensor([[.1, .2, .3]], requires_grad=True)
    memory = (torch.tensor([-.9, -.5]), torch.tensor([1., 2.]),
              torch.zeros(2), torch.ones(2))
    loss, count = ipcw_pairwise_ranking_loss(
        logits, torch.tensor([0.]), torch.ones(1), torch.ones(1), memory=memory,
    )
    assert count == 0 and loss == 0.
    loss.backward()
    assert not logits.grad.any()


@pytest.mark.parametrize("invalid", ["negative_bin", "too_large_bin", "fractional_bin", "censor", "hazard"])
def test_discrete_nll_rejects_invalid_likelihood_inputs(invalid):
    hazards = torch.tensor([[.2, .3, .4]])
    labels, censor = torch.tensor([1]), torch.tensor([0.])
    if invalid == "negative_bin":
        labels = torch.tensor([-1])
    elif invalid == "too_large_bin":
        labels = torch.tensor([3])
    elif invalid == "fractional_bin":
        labels = torch.tensor([1.5])
    elif invalid == "censor":
        censor = torch.tensor([.5])
    else:
        hazards[0, 0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        discrete_nll(hazards, labels, censor)


def test_fixed_margin_relation_projection_preserves_budget_and_cost_gradient_constraints():
    costs = torch.tensor([[[[[.2, .8, .3], [.7, .1, .5]]]]],
                         dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([[[[.18, .42]]]], dtype=torch.float64, requires_grad=True)
    cols = torch.tensor([[[[.25, .20, .15]]]], dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([[[[[.9, -.4, .2], [-.3, .6, -.8]]]]], dtype=torch.float64)
    plan = fixed_realized_marginal_plan(costs, rows, cols, epsilon=.3, tolerance=1e-10)
    torch.testing.assert_close(plan.sum(-1), rows.detach(), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(plan.sum(-2), cols.detach(), atol=1e-9, rtol=1e-9)
    (plan * coefficients).sum().backward()
    assert rows.grad is None and cols.grad is None
    assert costs.grad.abs().max() > 1e-3
    torch.testing.assert_close(costs.grad.sum(-1), torch.zeros_like(rows), atol=1e-8, rtol=0.)
    torch.testing.assert_close(costs.grad.sum(-2), torch.zeros_like(cols), atol=1e-8, rtol=0.)


def test_fixed_margin_projection_retains_zero_rows_and_zero_mass():
    costs = torch.rand(2, 1, 1, 3, 3, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([[[[.15, 0., .35]]], [[[0., 0., 0.]]]], dtype=torch.float64)
    cols = torch.tensor([[[[0., .2, .3]]], [[[0., 0., 0.]]]], dtype=torch.float64)
    plan = fixed_realized_marginal_plan(costs, rows, cols, epsilon=.3, tolerance=1e-10)
    torch.testing.assert_close(plan.sum(-1), rows, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(plan.sum(-2), cols, atol=1e-9, rtol=1e-9)
    assert not plan[0, 0, 0, 1].any()
    assert not plan[0, 0, 0, :, 0].any()
    assert not plan[1].any()
    (plan * torch.rand_like(plan)).sum().backward()
    assert torch.isfinite(costs.grad).all()
    assert not costs.grad[1].any()


@pytest.mark.parametrize("rows,cols", [((.2, .3), (.2, .4)), ((1e-12, 0.), (2e-12, 0.))])
def test_fixed_margin_projection_rejects_inconsistent_mass(rows, cols):
    costs = torch.zeros(1, 1, 1, 2, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        fixed_realized_marginal_plan(
            costs, torch.tensor([[[rows]]], dtype=torch.float64),
            torch.tensor([[[cols]]], dtype=torch.float64), epsilon=.2,
        )


def test_standalone_model_contains_no_external_host_plugin_or_reconstruction_decoder():
    model = _model()
    modules = [module.__class__.__module__ for module in model.modules()]
    forbidden = ("survot_rank.plugins", "mcat", "survpath")
    assert not any(any(fragment in name.lower() for fragment in forbidden) for name in modules)
    assert not any("decoder" in name.lower() or "recon" in name.lower()
                   for name, _ in model.named_parameters())
    assert not hasattr(model, "plugin")


@pytest.mark.parametrize("batch", [2, 3])
def test_full_training_objective_has_exact_six_term_sum_and_finite_backward(batch):
    model = _model()
    result = model.forward_with_losses(**_inputs(batch))
    terms = result["terms"]
    weighted_keys = ["ipcw_rank_weighted", "per_slot_nll_weighted", "slot_diversity_weighted",
                     "relation_weighted", "native_weighted"]
    expected = terms["global_nll"] + sum(terms[key] for key in weighted_keys)
    torch.testing.assert_close(result["loss"], expected)
    torch.testing.assert_close(result["auxiliary"], sum(terms[key] for key in weighted_keys))
    assert result["hazards"].shape == (batch, 3)
    assert ((result["hazards"] > 0.) & (result["hazards"] < 1.)).all()
    assert all(key in terms for key in ["ipcw_rank", "per_slot_nll", "slot_diversity", "relation", "native"])
    assert not any("recon" in key for key in terms)
    result["loss"].backward()
    _finite_gradients(model)


def test_full_objective_is_finite_under_cpu_bfloat16_autocast():
    model = _model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model.forward_with_losses(**_inputs(batch=2))
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["hazards"]).all()
    result["loss"].backward()
    _finite_gradients(model)


def test_zero_auxiliary_weights_leave_only_main_nll():
    model = _model(_config(ipcw_weight=0., slot_weight=0., diversity_weight=0.,
                           relation_weight=0., native_weight=0.))
    result = model.forward_with_losses(**_inputs())
    torch.testing.assert_close(result["loss"], result["terms"]["global_nll"])
    assert result["auxiliary"] == 0.
    assert all(result["terms"][name] == 0. for name in (
        "ipcw_rank_weighted", "per_slot_nll_weighted", "slot_diversity_weighted",
        "relation_weighted", "native_weighted"))


def test_eval_has_no_training_auxiliary_and_does_not_change_buffers():
    model = _model(training=False)
    before = {name: value.clone() for name, value in model.named_buffers()}
    inputs = _inputs()
    with torch.no_grad():
        first, auxiliary = model(**inputs)
        second, auxiliary_again = model(**{key: value for key, value in inputs.items()
                                         if key not in {"y", "event_time", "c"}})
    assert auxiliary == 0. and auxiliary_again == 0.
    torch.testing.assert_close(first, second)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], msg=name)


def test_zero_transport_mass_keeps_exact_native_prediction(monkeypatch):
    model = _model(training=False)

    def zero_plans(costs, rows, cols, epoch):
        return torch.zeros_like(costs), rows, cols

    monkeypatch.setattr(model.core, "plans_from_costs", zero_plans)
    with torch.no_grad():
        logits, auxiliary = model(**_inputs())
    assert not model.last_state.plans.any()
    assert not model.last_state.writeback_confidence.any()
    assert auxiliary == 0.
    torch.testing.assert_close(logits.sigmoid(), model.last_native_hazards, atol=0., rtol=0.)


def test_padding_patch_values_cannot_change_prediction_or_transport():
    model = _model(training=False)
    inputs = _inputs(padded=True)
    with torch.no_grad():
        first, _ = model(**inputs)
        plan = model.last_state.plans.clone()
        changed = dict(inputs)
        changed["x_wsi"] = inputs["x_wsi"].clone()
        changed["x_wsi"][~inputs["wsi_mask"]] = 1e6
        second, _ = model(**changed)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(plan, model.last_state.plans, atol=1e-6, rtol=1e-6)


def test_masked_omics_values_cannot_change_prediction():
    model = _model(training=False)
    inputs = _inputs()
    inputs["omics_mask"][:, 1] = False
    with torch.no_grad():
        first, _ = model(**inputs)
        changed = dict(inputs)
        changed["x_omic2"] = torch.full_like(inputs["x_omic2"], 1e6)
        second, _ = model(**changed)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_state_dict_roundtrip_retains_reference_and_prediction():
    model = _model(training=False)
    inputs = _inputs()
    with torch.no_grad():
        expected, _ = model(**inputs)
    archive = BytesIO()
    torch.save(model.state_dict(), archive)
    archive.seek(0)
    state = torch.load(archive, weights_only=False)
    restored = DCTV2(_config()).eval()
    restored.load_state_dict(state)
    with torch.no_grad():
        actual, _ = restored(**inputs)
    torch.testing.assert_close(actual, expected)
    for name, value in model.named_buffers():
        torch.testing.assert_close(dict(restored.named_buffers())[name], value)


def _no_gradient(parameter):
    return parameter.grad is None or not parameter.grad.any()


def test_relation_gradient_only_reaches_cost_network_and_restores_modes(monkeypatch):
    model = _model()
    model.sig_networks[0].eval()  # preserve a deliberately mixed train/eval state
    original_modes = {name: m.training for name, m in model.named_modules()}
    with torch.no_grad():
        model(**{**_inputs(), "compute_aux": False})
    criterion = model_module.logit_nll

    def forced_active(logits, *args, **kwargs):
        value = criterion(logits, *args, **kwargs)
        return value + (1. if logits.requires_grad else 0.)

    monkeypatch.setattr(model_module, "logit_nll", forced_active)
    loss = model.relation_loss(model.last_state, _inputs()["y"], _inputs()["c"], 10)
    assert loss > 0.
    loss.backward()
    allowed = "core.stage_pair_cost."
    assert any(p.grad is not None and p.grad.abs().sum() > 0. for name,p in model.named_parameters() if name.startswith(allowed))
    for name, p in model.named_parameters():
        if not name.startswith(allowed):
            assert _no_gradient(p), name
    assert original_modes == {name: m.training for name,m in model.named_modules()}
    _finite_gradients(model)
    assert model.last_relation_diagnostics["row_error"] < 1e-5
    assert model.last_relation_diagnostics["col_error"] < 1e-5


def test_relation_restores_all_modes_on_criterion_failure(monkeypatch):
    model = _model()
    with torch.no_grad():
        model(**{**_inputs(), "compute_aux": False})
    modes = {name:m.training for name,m in model.named_modules()}
    def fail(*args, **kwargs):
        raise RuntimeError("controlled criterion failure")
    monkeypatch.setattr(model_module, "logit_nll", fail)
    with pytest.raises(RuntimeError, match="controlled criterion failure"):
        model.relation_loss(model.last_state, _inputs()["y"], _inputs()["c"], 10)
    assert modes == {name:m.training for name,m in model.named_modules()}


def test_native_auxiliary_updates_original_slots_and_shared_head_only():
    model = _model(_config(ipcw_weight=0., slot_weight=0., diversity_weight=0., relation_weight=0.))
    _, auxiliary = model(**_inputs())
    assert model.last_training_losses["native"] > 0.
    auxiliary.backward()
    assert model.event_hazard.weight.grad.abs().sum() > 0.
    assert model.wsi_mlp[0].weight.grad.abs().sum() > 0.
    assert model.core.shared_wsi_prototypes.grad.abs().sum() > 0.
    for name,p in model.named_parameters():
        if name.startswith(("core.stage_pair_cost.", "core.evidence_gate.", "core.fusion.", "core.event_encoder.", "per_slot_hazard_")) or name == "residual_gain":
            assert _no_gradient(p), name
    _finite_gradients(model)


def test_independent_control_has_zero_relation_without_hidden_learned_topology():
    model = _model(_config(plan_mode="independent"))
    result = model.forward_with_losses(**_inputs())
    assert result["terms"]["relation"] == 0.
    plan = model.last_state.plans
    from survot_rank.research.methods.dct_v2.transport import realized_independent_plan
    torch.testing.assert_close(plan, realized_independent_plan(plan))


def test_training_requires_fold_reference_and_eval_never_consumes_heldout_labels():
    model = DCTV2(_config())
    with pytest.raises(RuntimeError, match="training-fold censor reference"):
        model(**_inputs())
    model.eval()
    with torch.no_grad():
        first,_ = model(**_inputs())
        changed = {**_inputs(), "c": torch.full((3,), float("nan")), "event_time":torch.full((3,), float("nan"))}
        second,_ = model(**changed)
    torch.testing.assert_close(first, second)
    assert model._rank_memory is None


def test_rank_memory_resets_each_epoch_and_survives_eval_unchanged():
    model = _model()
    model(**_inputs(2))
    assert model._rank_memory[0].numel() == 2
    previous = tuple(v.clone() for v in model._rank_memory)
    model.eval()
    with torch.no_grad():
        model(**{**_inputs(3), "epoch":11})
    for old,new in zip(previous,model._rank_memory):
        torch.testing.assert_close(old,new)
    model.train()
    model(**{**_inputs(3), "epoch":11})
    assert model._rank_memory[0].numel() == 3


def test_checkpoint_rejects_different_recipe_and_unrelated_version():
    model = _model()
    different = DCTV2(_config(native_weight=0.))
    with pytest.raises(ValueError, match="exact saved recipe"):
        different.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="standalone checkpoint schema"):
        model.load_state_dict({"_extra_state":{"schema":"dct_transport_plugin_v2"}})


def test_factory_and_shared_trainer_use_logits_and_exact_batch_mean():
    from survot_rank.training.model_factory import get_model
    from survot_rank.training.extended_args import build_base_parser
    from survot_rank.training.train_runner import init_loss_function, compose_batch_objective
    args = build_base_parser().parse_args([
        "--survot_method","dct_v2", "--encoding_dim","6", "--wsi_projection_dim","8",
        "--n_classes","3", "--slot_num_wsi","2", "--slot_num_omics","2", "--slot_iters","1",
        "--dct_num_stages","2", "--otehv2_heads","2", "--otehv2_layers","1", "--otehv2_dropout","0",
        "--alpha_surv","0", "--dct_v2_relation_warmup_epochs","0", "--dct_v2_relation_ramp_epochs","1"])
    args.omic_sizes = [3,4]
    args.cur_epoch = 10
    model = get_model("dct_v2", args)
    model.configure_train_reference(torch.tensor([1.,2.,3.,4.]), torch.tensor([0.,1.,0.,0.]))
    inputs = _inputs(2)
    logits, auxiliary = model(**inputs)
    loss_fn = init_loss_function(args)
    assert isinstance(loss_fn, StableNLLSurvLoss)
    combined = compose_batch_objective(loss_fn(logits, inputs["y"], inputs["event_time"], inputs["c"]), auxiliary, 2)
    torch.testing.assert_close(combined, logit_nll(logits, inputs["y"], inputs["c"])+auxiliary)
    combined.backward()
    _finite_gradients(model)


def test_absolute_cost_offset_reduces_realized_uot_mass():
    model = _model(training=False)
    with torch.no_grad():
        model(**_inputs(2))
        state = model.last_state
        costs, rows, cols = model.core.costs_and_marginals(state.wsi_slots, state.omics_slots)
        base, _, _ = model.core.plans_from_costs(costs, rows, cols, 10)
        shifted, _, _ = model.core.plans_from_costs(costs+1., rows, cols, 10)
    factor = math.exp(-1/(model.core.epsilon(10)+model.config.tau_wsi+model.config.tau_omics))
    assert torch.all(shifted.sum((-2,-1)) < base.sum((-2,-1)))
    torch.testing.assert_close(shifted, base*factor, atol=2e-6, rtol=2e-5)


def test_diversity_is_independent_of_repeating_patients_in_batch():
    model = _model(training=False)
    with torch.no_grad():
        model(**_inputs(2))
        state = model.last_state
        _, before = model._slot_objectives(state, _inputs(2)["y"], _inputs(2)["c"], torch.ones(2))
        from dataclasses import fields
        repeated = type(state)(**{f.name:torch.cat((getattr(state,f.name),getattr(state,f.name)),0) for f in fields(state)})
        _, after = model._slot_objectives(repeated, _inputs(2)["y"].repeat(2), _inputs(2)["c"].repeat(2), torch.ones(4))
    torch.testing.assert_close(before,after)


@pytest.mark.parametrize("missing", ["wsi_missing", "omic_missing"])
def test_single_modality_inference_uses_native_information_without_exchange(missing):
    model = _model(training=False)
    with torch.no_grad():
        logits, aux = model(**{**_inputs(),missing:True})
    assert torch.isfinite(logits).all() and aux == 0.
    assert not model.last_state.plans.any()
    torch.testing.assert_close(logits.sigmoid(), model.last_native_hazards, atol=0.,rtol=0.)


def test_native_reference_deficit_can_be_positive_even_at_unit_total_mass():
    model = _model(training=False)
    with torch.no_grad():
        model(**_inputs(2))
    state = model.last_state
    plan = torch.diag(torch.tensor([.7,.3])).expand_as(state.plans).clone().requires_grad_()
    rows = torch.full_like(state.row_marginals,.5,requires_grad=True)
    cols = torch.full_like(state.col_marginals,.5,requires_grad=True)
    example = replace(state,plans=plan,row_marginals=rows,col_marginals=cols)
    deficit = model.reference_deficit(example)
    torch.testing.assert_close(deficit,torch.full((2,),.2))
    assert not deficit.requires_grad
