"""Synthetic tensor contracts only; no patient training or inference."""
from __future__ import annotations

from dataclasses import replace
import copy
import subprocess
import sys

import pytest
import torch
from torch import nn

from survot_rank.plugins.transport_only import DCTTransportMechanism, TransportOnlyConfig


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def recipe(**kwargs):
    values = dict(host_dim=8, slot_budget=4, stages=2, eps=.2, eps_start=.2,
                  geometries=("cosine", "euclidean"), projection_iters=1000,
                  uot_iters=400)
    values.update(kwargs)
    return TransportOnlyConfig(**values)


def test_disabled_is_object_identity_without_rng_and_has_no_objective():
    mechanism = DCTTransportMechanism(recipe())
    w, o = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    state = torch.random.get_rng_state().clone()
    result = mechanism(w, o, enabled=False)
    assert type(result) is tuple and len(result) == 2
    assert result[0] is w and result[1] is o
    assert torch.equal(state, torch.random.get_rng_state())
    assert mechanism.last_plans is None
    assert not any(isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoder,
                                      nn.TransformerEncoderLayer))
                   for module in mechanism.modules())
    for forbidden in ("slot_attention_wsi", "fusion", "event_encoder", "event_gate",
                      "reconstruction", "survival_head", "aux_loss"):
        assert not hasattr(mechanism, forbidden)


def test_native_slots_are_direct_and_host_objective_reaches_transport_immediately():
    mechanism = DCTTransportMechanism(recipe())
    w = torch.randn(2, 3, 8, requires_grad=True)
    o = torch.randn(2, 4, 8, requires_grad=True)
    out_w, out_o = mechanism(w, o)
    assert out_w.shape == w.shape and out_o.shape == o.shape
    assert mechanism.last_diagnostics["wsi_compressed"] == [False, False]
    assert mechanism.last_diagnostics["omics_compressed"] == [False, False]
    assert mechanism.last_plans.shape == (2, 2, 2, 3, 4)
    (out_w.square().mean() + out_o.square().mean()).backward()
    assert mechanism.stage_pair_cost[1].weight.grad.norm() > 0
    assert mechanism.evidence_gate[1].weight.grad.norm() > 0
    assert mechanism.wsi_prototypes.grad is None
    assert mechanism.omics_prototypes.grad is None
    assert w.grad.norm() > 0 and o.grad.norm() > 0
    assert not mechanism.last_plans.requires_grad


def test_balanced_both_marginals_and_independent_control():
    mechanism = DCTTransportMechanism(recipe())
    w, o = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    mechanism(w, o)
    p = mechanism.last_plans
    rows, cols = mechanism.last_reference_rows, mechanism.last_reference_cols
    assert torch.allclose(p.sum(-1), rows[:, :, None].expand_as(p.sum(-1)), atol=1e-4)
    assert torch.allclose(p.sum(-2), cols[:, :, None].expand_as(p.sum(-2)), atol=1e-4)
    mechanism.set_plan_mode("independent")(w, o)
    expected = (rows[..., :, None] * cols[..., None, :]).unsqueeze(2)
    assert torch.allclose(mechanism.last_plans, expected.expand_as(p), atol=1e-6)


def test_uot_independent_keeps_factual_realized_mass_and_both_marginals():
    mechanism = DCTTransportMechanism(recipe(transport_mode="uot"))
    w, o = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    mechanism(w, o)
    learned = mechanism.last_plans.clone()
    mechanism.set_plan_mode("independent")(w, o)
    independent = mechanism.last_plans
    assert torch.allclose(independent.sum(-1), learned.sum(-1), atol=1e-7)
    assert torch.allclose(independent.sum(-2), learned.sum(-2), atol=1e-7)
    assert torch.allclose(independent.sum((-2, -1)), learned.sum((-2, -1)), atol=1e-7)
    assert all(report["actual_marginal_control"]
               for report in mechanism.last_diagnostics["solver"])


def test_uot_writeback_retains_mass_instead_of_unit_normalizing_it(monkeypatch):
    mechanism = DCTTransportMechanism(recipe(stages=1, geometries=("cosine",),
                                             transport_mode="uot"))
    w, o = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    mass = .25

    def controlled_solver(costs, rows, cols, epoch):
        return torch.full_like(costs, mass), rows, cols

    monkeypatch.setattr(mechanism, "plans_from_costs", controlled_solver)
    out_w, out_o = mechanism(w, o)
    assert torch.allclose(out_w - w, torch.tanh(mechanism.wsi_gain) * mass * o, atol=1e-7)
    assert torch.allclose(out_o - o, torch.tanh(mechanism.omics_gain) * mass * w, atol=1e-7)


def test_long_sequences_pool_and_lift_without_changing_host_shapes():
    mechanism = DCTTransportMechanism(recipe(slot_budget=3))
    w = torch.randn(2, 19, 8, requires_grad=True)
    o = torch.randn(2, 11, 8, requires_grad=True)
    out_w, out_o = mechanism(w, o)
    assert out_w.shape == w.shape and out_o.shape == o.shape
    assert mechanism.last_plans.shape == (2, 2, 2, 3, 3)
    assert mechanism.last_diagnostics["wsi_compressed"] == [True, True]
    (out_w.square().mean() + out_o.square().mean()).backward()
    assert mechanism.wsi_prototypes.grad.norm() > 0
    assert mechanism.omics_prototypes.grad.norm() > 0


def test_masked_padding_and_missing_modality_patients_remain_exactly_unchanged():
    mechanism = DCTTransportMechanism(recipe(slot_budget=3))
    w, o = torch.randn(3, 6, 8), torch.randn(3, 5, 8)
    wm = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    om = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    w[0, 2:] = float("nan")
    out_w, out_o = mechanism(w, o, wsi_mask=wm, omics_mask=om)
    assert torch.equal(out_w[0, :2], out_w[0, :2])
    assert torch.isnan(out_w[0, 2:]).all()
    assert torch.equal(out_o[~om], o[~om])
    assert torch.equal(out_w[1:], w[1:])
    assert torch.equal(out_o[1:], o[1:])
    assert mechanism.last_diagnostics["paired_patients"] == 1
    assert mechanism.last_diagnostics["transport_shapes"] == [(2, 2), (0, 0), (0, 0)]
    assert mechanism.last_plans[1:].count_nonzero() == 0


def test_explicit_zero_residual_is_supported_and_epoch_setter_is_used():
    mechanism = DCTTransportMechanism(recipe(residual_init=0., eps_start=.4, eps=.1))
    w, o = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    mechanism.set_epoch(6)
    out_w, out_o = mechanism(w, o)
    assert torch.equal(out_w, w) and torch.equal(out_o, o)
    assert mechanism.last_diagnostics["epoch"] == 6
    assert mechanism.last_diagnostics["epsilon"] == pytest.approx(.25)
    mechanism(w, o, epoch=0)
    assert mechanism.last_diagnostics["epsilon"] == pytest.approx(.4)


def test_checkpoint_requires_exact_config_and_restores_runtime_switches():
    cfg = recipe()
    mechanism = DCTTransportMechanism(cfg).set_enabled(False).set_epoch(7).set_plan_mode("independent")
    state = copy.deepcopy(mechanism.state_dict())
    restored = DCTTransportMechanism(cfg)
    restored.load_state_dict(state)
    assert restored.enabled is False and restored.current_epoch == 7
    assert restored.plan_mode == "independent"
    with pytest.raises(ValueError, match="exact checkpoint configuration"):
        DCTTransportMechanism(replace(cfg, eps=.3)).load_state_dict(state)
    with pytest.raises(ValueError, match="metadata"):
        restored.set_extra_state({"schema": "old_plugin_with_losses"})


@pytest.mark.parametrize("values", [dict(slot_budget=0), dict(host_dim=True), dict(eps=0),
                                         dict(geometries=()), dict(plan_mode="uniform"),
                                         dict(transport_mode="uot", cost_mode="legacy")])
def test_invalid_recipes_fail(values):
    with pytest.raises((ValueError, TypeError)):
        recipe(**values)


def test_fresh_import_does_not_load_old_plugin_task_losses():
    code = ("import sys; import survot_rank.plugins.transport_only; "
            "assert 'survot_rank.plugins.transport_v2' not in sys.modules; assert 'survot_rank.plugins.transport_v2.losses' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
