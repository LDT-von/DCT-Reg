"""Synthetic contract checks; no patient evaluation or training experiment."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from survot_rank.plugins.transport import (DCTTransportPlugin, TransportCore,
    TransportPluginConfig, TransportReconstructionHeads)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(**kwargs):
    values = dict(wsi_dim=8, omics_dim=6, dim=16, wsi_slots=3, omics_slots=2,
                  stages=2, heads=2, event_layers=1, slot_iters=2, slot_heads=2,
                  slot_dim_head=8, dropout=0., projection_iters=1000)
    values.update(kwargs)
    return TransportPluginConfig(**values)


def features(batch=2):
    generator = torch.Generator().manual_seed(29)
    return (torch.randn(batch, 7, 8, generator=generator),
            torch.randn(batch, 4, 6, generator=generator))


def active_plugin(**kwargs):
    torch.manual_seed(23)
    net = DCTTransportPlugin(config(**kwargs)).eval()
    with torch.no_grad():
        net.wsi_gain.fill_(.25)
        net.omics_gain.fill_(.25)
    return net


def test_disabled_is_exact_bypass_and_calls_no_feature_or_aux_module(monkeypatch):
    net = active_plugin()
    def fail(*a, **k):
        raise AssertionError("disabled module called")
    monkeypatch.setattr(net.core, "forward", fail)
    wsi, omics = features()
    result = net(wsi, omics, enabled=False)
    assert result.wsi is wsi and result.omics is omics
    assert result.transport is None and result.aux_loss.item() == 0


def test_zero_residual_initialization_preserves_original_features():
    net = DCTTransportPlugin(config()).eval()
    wsi, omics = features()
    result = net(wsi, omics)
    torch.testing.assert_close(result.wsi, wsi, rtol=0, atol=0)
    torch.testing.assert_close(result.omics, omics, rtol=0, atol=0)
    assert result.wsi.shape == wsi.shape and result.omics.shape == omics.shape


def test_nonzero_residual_carries_host_gradient_through_gate_cost_and_prototypes():
    net = active_plugin().train()
    wsi, omics = features()
    result = net(wsi, omics)
    (result.wsi.square().mean() + result.omics.square().mean()).backward()
    for name, module in (("stage_gate", net.core.event_gate),
                         ("stage_cost", net.core.stage_pair_cost),
                         ("evidence_gate", net.core.evidence_gate)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), name
    assert net.core.shared_wsi_prototypes.grad.abs().sum() > 0
    assert net.core.shared_omic_prototypes.grad.abs().sum() > 0


def test_zero_gate_has_opening_gradient_then_core_receives_host_gradient():
    net = DCTTransportPlugin(config()).train()
    opt = torch.optim.SGD(net.parameters(), lr=.1)
    wsi, omics = features()
    torch.manual_seed(47)
    out = net(wsi, omics)
    loss = out.wsi.square().mean() + out.omics.square().mean()
    loss.backward()
    assert net.wsi_gain.grad.abs() > 0 and net.omics_gain.grad.abs() > 0
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.manual_seed(47)
    out = net(wsi, omics)
    (out.wsi.square().mean() + out.omics.square().mean()).backward()
    assert net.core.shared_wsi_prototypes.grad.abs().sum() > 0


def test_masked_nan_padding_does_not_change_valid_features_or_plans():
    net = active_plugin()
    wsi, omics = features()
    expected = net(wsi, omics)
    h = torch.cat((wsi, torch.full((2, 3, 8), float("nan"))), 1)
    g = torch.cat((omics, torch.full((2, 2, 6), float("nan"))), 1)
    wm = torch.arange(10)[None].expand(2, -1) < 7
    om = torch.arange(6)[None].expand(2, -1) < 4
    actual = net(h, g, wsi_mask=wm, omics_mask=om)
    torch.testing.assert_close(actual.wsi[:, :7], expected.wsi, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual.omics[:, :4], expected.omics, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual.transport.plans, expected.transport.plans, rtol=2e-5, atol=2e-5)
    assert torch.isfinite(actual.wsi).all() and torch.isfinite(actual.omics).all()
    assert not actual.wsi[:, 7:].any() and not actual.omics[:, 4:].any()


def test_missing_modality_bypasses_cross_updates_and_all_missing_aux_still_backward():
    net = active_plugin().train()
    net.reconstruction = TransportReconstructionHeads(16, 4, heads=2)
    wsi, omics = features()
    wm = torch.zeros(2, 7, dtype=torch.bool)
    om = torch.ones(2, 4, dtype=torch.bool)
    out = net(wsi, omics, wsi_mask=wm, omics_mask=om, epoch=7)
    torch.testing.assert_close(out.omics, omics, rtol=0, atol=0)
    assert not out.transport.plans.any() and out.aux_losses["cross"].item() == 0
    out = net(wsi, omics, wsi_mask=wm, omics_mask=torch.zeros_like(om), epoch=7)
    assert out.aux_loss.requires_grad and out.aux_loss.item() == 0
    out.aux_loss.backward()
    assert torch.isfinite(net.wsi_gain.grad)


def test_eval_never_calls_auxiliary_heads_even_when_requested(monkeypatch):
    net = active_plugin()
    net.reconstruction = TransportReconstructionHeads(16, 4, heads=2)
    def fail(*a, **k):
        raise AssertionError("eval called reconstruction")
    monkeypatch.setattr(net.reconstruction, "forward", fail)
    assert net(*features(), compute_aux=True, epoch=7).aux_loss.item() == 0


@pytest.mark.parametrize("mode", ["learned", "independent", "uniform"])
def test_couplings_have_declared_marginals_and_nonnegative_finite_mass(mode):
    state = active_plugin(plan_mode=mode)(*features()).transport
    assert torch.isfinite(state.plans).all() and bool((state.plans >= 0).all())
    rows = state.row_marginals.unsqueeze(2).expand(-1, -1, 3, -1)
    cols = state.col_marginals.unsqueeze(2).expand(-1, -1, 3, -1)
    torch.testing.assert_close(state.plans.sum(-1), rows, rtol=0, atol=1e-4)
    torch.testing.assert_close(state.plans.sum(-2), cols, rtol=0, atol=1e-4)
    if mode == "independent":
        expected = state.row_marginals[..., :, None] * state.col_marginals[..., None, :]
        torch.testing.assert_close(state.plans[:, :, 0], expected)


def test_independent_preserves_learned_marginals_and_parameter_count():
    learned = active_plugin()
    independent = copy.deepcopy(learned).set_plan_mode("independent")
    one, two = learned(*features()).transport, independent(*features()).transport
    torch.testing.assert_close(one.row_marginals, two.row_marginals)
    torch.testing.assert_close(one.col_marginals, two.col_marginals)
    assert not torch.allclose(one.plans, two.plans)
    assert sum(p.numel() for p in learned.parameters()) == sum(p.numel() for p in independent.parameters())


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_coincident_slots_have_finite_cost_and_plan_gradients(value):
    core = active_plugin().core
    h = torch.full((2, 3, 16), value, requires_grad=True)
    g = torch.full((2, 2, 16), value, requires_grad=True)
    cost, rows, cols = core.costs_and_marginals(h, g)
    plans, _, _ = core.plans_from_costs(cost, rows, cols, epoch=7)
    (cost.square().sum() + plans.square().sum()).backward()
    assert torch.isfinite(h.grad).all() and torch.isfinite(g.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)


def test_patient_features_are_isolated_at_eval():
    net = active_plugin()
    h, g = features()
    expected = net(h, g)
    h2, g2 = h.clone(), g.clone()
    h2[1].mul_(10); g2[1].add_(100)
    actual = net(h2, g2)
    torch.testing.assert_close(actual.wsi[0], expected.wsi[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual.omics[0], expected.omics[0], atol=1e-5, rtol=1e-5)


def test_single_geometry_keeps_fixed_event_readout_capacity():
    multiple = active_plugin()
    single = active_plugin(geometries=("cosine",))
    assert sum(p.numel() for p in multiple.parameters()) == sum(p.numel() for p in single.parameters())
    out = single(*features())
    assert out.transport.plans.shape[2] == 1
    assert out.wsi.shape == features()[0].shape


def test_plan_mode_and_disabled_flag_survive_checkpoint_roundtrip():
    net = active_plugin().set_plan_mode("independent").set_enabled(False)
    saved = copy.deepcopy(net.state_dict())
    assert net.config.to_dict()["plan_mode"] == "independent"
    restored = DCTTransportPlugin(net.config).eval()
    restored.load_state_dict(saved)
    assert not restored.enabled and restored.core.plan_mode == "independent"
    torch.testing.assert_close(restored(*features()).wsi, features()[0], rtol=0, atol=0)


def test_checkpoint_refuses_silent_recipe_change():
    net = active_plugin()
    wrong = DCTTransportPlugin(replace(net.config, coordinate_temperature=.4))
    with pytest.raises(ValueError, match="checkpoint config"):
        wrong.load_state_dict(net.state_dict())


@pytest.mark.parametrize("name,value", [("stages", 0), ("heads", 3), ("eps", 0),
    ("plan_mode", "bad"), ("geometries", ()), ("projection_iters", 0),
    ("coordinate_temperature", float("nan"))])
def test_invalid_configs_fail_before_model_creation(name, value):
    with pytest.raises(ValueError):
        config(**{name: value})


def test_valid_nan_rejected_and_mask_must_be_boolean():
    net = active_plugin()
    h, g = features(); h[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        net(h, g)
    with pytest.raises(ValueError, match="bool"):
        net(*features(), wsi_mask=torch.ones(2, 7))


@pytest.mark.parametrize("device,dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_autocast_with_fp32_parameters_keeps_finite_output_and_gradients(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    net = active_plugin().to(device).train()
    h, g = (x.to(device) for x in features())
    wm = torch.ones(2, 7, dtype=torch.bool, device=device); wm[1] = False
    with torch.autocast(device_type=device, dtype=dtype):
        out = net(h, g, wsi_mask=wm)
        loss = out.wsi.square().mean() + out.omics.square().mean()
    assert torch.isfinite(out.wsi).all() and torch.isfinite(out.omics).all()
    assert torch.isfinite(out.transport.transported_wsi).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@pytest.mark.parametrize("device,dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_autocast_three_auxiliary_heads_with_missing_patient_remains_finite(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    net = active_plugin().train()
    net.reconstruction = TransportReconstructionHeads(16, 4, heads=2)
    net.to(device)
    h, g = (x.to(device) for x in features())
    wm = torch.ones(2, 7, dtype=torch.bool, device=device)
    om = torch.ones(2, 4, dtype=torch.bool, device=device)
    wm[1] = False; om[1] = False
    with torch.autocast(device_type=device, dtype=dtype):
        out = net(h, g, wsi_mask=wm, omics_mask=om, epoch=7)
        loss = out.wsi.square().mean() + out.omics.square().mean() + out.aux_loss
    assert all(torch.isfinite(value).all() for value in out.aux_losses.values())
    assert out.aux_loss.item() > 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_default_projection_budget_handles_sharp_bfloat16_plan_and_backpropagates(monkeypatch):
    torch.manual_seed(1)
    cfg = TransportPluginConfig(wsi_dim=8, omics_dim=8, dim=8, stages=2,
        heads=2, wsi_slots=2, omics_slots=2, slot_heads=2, slot_dim_head=4,
        dropout=0, residual_init=.2)
    # Preserve the observed construction and RNG order of the edge case.
    net = DCTTransportPlugin(cfg, TransportReconstructionHeads(8, 2, heads=2)).train()
    h, g = torch.randn(2, 3, 8), torch.randn(2, 2, 8)
    captured = {}
    original = net.core.plans_from_costs
    def capture(cost, rows, cols, epoch):
        captured.update(cost=cost.detach(), rows=rows.detach(), cols=cols.detach())
        return original(cost, rows, cols, epoch)
    monkeypatch.setattr(net.core, "plans_from_costs", capture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(h, g, epoch=9)
        loss = out.wsi.square().mean() + out.omics.square().mean() + out.aux_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    net.core.config = replace(cfg, projection_iters=1000)
    with pytest.raises(RuntimeError, match="marginal tolerance"):
        original(captured["cost"], captured["rows"], captured["cols"], epoch=9)


def test_from_dct_exports_equivalent_features_plans_and_keeps_frozen_prototypes():
    from test_dct_v313_transport_reconstruction import make_args
    from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import DCTV313TransportReconstruction
    torch.manual_seed(43)
    model = DCTV313TransportReconstruction(make_args(dct_v313_revision="repaired",
        dct_slot_init_mode="deterministic")).eval()
    model.shared_wsi_prototypes.requires_grad_(False)
    before = torch.random.get_rng_state()
    core = TransportCore.from_dct(model)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not core.shared_wsi_prototypes.requires_grad
    assert core.shared_wsi_prototypes.data_ptr() != model.shared_wsi_prototypes.data_ptr()
    h, g = torch.randn(2, 7, 16), torch.randn(2, 5, 16)
    masks = torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool)
    sw, so, *_ = model._encode_transport_slots(h, g, {"x_wsi": h, "wsi_patch_mask": masks[0]})
    costs, rows, cols, _ = model._cost_tensor(sw, so)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)
    expected_logits, expected_gate = model._encode_logits_from_plans(sw, so, plans)
    out = core(h, g, wsi_mask=masks[0], omics_mask=masks[1], epoch=7)
    torch.testing.assert_close(out.wsi_slots, sw, rtol=0, atol=0)
    torch.testing.assert_close(out.omics_slots, so, rtol=0, atol=0)
    expected_plans = torch.stack([torch.stack(stage, 1) for stage in plans], 1)
    torch.testing.assert_close(out.plans, expected_plans, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(out.stage_weights, expected_gate, rtol=2e-5, atol=2e-5)
    actual_logits = torch.einsum("bs,bsc->bc", out.stage_weights, model.event_hazard(out.event_tokens))
    torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-5, atol=2e-5)
    assert not any("hazard" in name or "reconstruction" in name for name in core.state_dict())
    restored = TransportCore(core.config).eval()
    restored.load_state_dict(core.state_dict(), strict=True)
    reloaded = restored(h, g, wsi_mask=masks[0], omics_mask=masks[1], epoch=7)
    torch.testing.assert_close(reloaded.plans, out.plans, rtol=0, atol=0)
