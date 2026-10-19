"""Current readout simplification; synthetic inputs, no patient jobs."""
from copy import deepcopy

import pytest
import torch

from survot_rank.research.methods.dct_v21 import model as v21_model
from survot_rank.research.methods.dct_v23 import DCTV23, DCTV23Config


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def candidate(mode="off", **overrides):
    options = dict(omic_sizes=(2, 3), dim=8, wsi_input_dim=3, wsi_hidden_dim=4,
        num_classes=3, wsi_slots=2, omics_slots=2, stages=2, heads=2, event_layers=1,
        slot_iters=1, slot_heads=2, slot_dim_head=4, slot_hidden_dim=16,
        dropout=0., wsi_dropout=0., reconstruction_mode=mode)
    options.update(overrides)
    torch.manual_seed(47)
    model = DCTV23(DCTV23Config(**options))
    if mode != "off":
        model.configure_reconstruction_reference((("a", "b"), ("c", "d", "e")),
            [0.] * 5, [1.] * 5, ["s0", "s1"])
    if mode == "gain_calibrated":
        model._gain_targets = {"s0": [.2, -.1], "s1": [-.1, .2]}
    return model


def inputs():
    generator = torch.Generator().manual_seed(13)
    return dict(x_wsi=torch.randn(2, 4, 3, generator=generator),
        x_omic1=torch.randn(2, 2, generator=generator),
        x_omic2=torch.randn(2, 3, generator=generator),
        y=torch.tensor([0, 2]), c=torch.tensor([0., 1.]),
        event_time=torch.tensor([1., 3.]), epoch=20, patient_ids=["s0", "s1"],
        reconstruction_global_mask=torch.tensor([[True, False, True, False, False],
                                                [False, True, False, True, False]]))


@pytest.mark.parametrize("mode,count", [("off", 2), ("masked_plain", 3), ("gain_calibrated", 4)])
def test_training_skips_relation_and_risk_coactivation_but_preserves_recovery(monkeypatch, mode, count):
    model = candidate(mode).train()
    def forbidden(*a, **kw):
        pytest.fail("Removed relation/coactivation path was executed")
    monkeypatch.setattr(model, "relation_loss", forbidden)
    monkeypatch.setattr(v21_model, "centered_pair_coactivation", forbidden)
    batch = inputs()
    output = model.forward_with_losses(**batch)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    for name in ("ipcw_rank", "per_slot_nll", "relation", "native"):
        assert model.last_training_losses[name + "_effective_weight"] == 0
        assert model.last_training_losses[name] == 0
    assert not model.last_local_readout.any()
    assert len(model.get_extra_state()["losses"]) == count
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if mode != "off":
        assert model.last_reconstruction_state is not None
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.reconstruction_decoder.parameters())
    if mode == "gain_calibrated":
        assert model.last_training_losses["gain_calibration_effective_weight"] > 0
        assert model.last_reconstruction["calibration_cost_goal"].shape == (2, 2)


def test_prediction_is_shared_head_on_native_plus_mass_weighted_event_residual():
    model = candidate().eval()
    logits, _ = model(**inputs())
    state = model.last_state
    native = model._native_feature(state)
    mass = model._quality(state)[0]
    features = native[:, None] + torch.tanh(model.residual_gain) * mass.clamp(0, 1)[..., None] * state.event_tokens
    expected = torch.einsum("bs,bsc->bc", model._stage_weights(state), model.event_hazard(features))
    assert torch.allclose(logits, expected, atol=1e-6)
    gradients = torch.autograd.grad(logits.square().sum(), tuple(model.core.event_encoder.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)


def test_slot_heads_no_longer_affect_prediction_or_receive_main_loss_gradient():
    model = candidate().eval()
    before, _ = model(**inputs())
    parameters = tuple(model.per_slot_hazard_wsi.parameters()) + tuple(model.per_slot_hazard_omic.parameters())
    gradients = torch.autograd.grad(before.square().sum(), parameters, allow_unused=True)
    assert all(g is None or not g.any() for g in gradients)
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(100.)
    after, _ = model(**inputs())
    assert torch.equal(before, after)


def test_native_information_and_exact_missing_pair_fallback_remain():
    model = candidate().eval()
    logits, _ = model(**inputs(), omic_missing=True)
    assert torch.equal(logits, model.event_hazard(model._native_feature(model.last_state)))
    assert model._native_feature(model.last_state).abs().sum() > 0


def test_explicit_historical_recipe_keeps_original_readout_and_checkpoint_boundary(monkeypatch):
    historical = candidate(local_readout="centered_pair", ipcw_weight=.1,
        slot_weight=.05, relation_weight=.05, native_weight=.05).eval()
    calls = []
    original = v21_model.centered_pair_coactivation
    def observed(*a, **kw):
        calls.append(True)
        return original(*a, **kw)
    monkeypatch.setattr(v21_model, "centered_pair_coactivation", observed)
    historical(**inputs())
    assert calls
    saved = deepcopy(historical.state_dict())
    restored = candidate(local_readout="centered_pair", ipcw_weight=.1,
        slot_weight=.05, relation_weight=.05, native_weight=.05).eval()
    restored.load_state_dict(saved, strict=True)
    assert torch.equal(historical(**inputs())[0], restored(**inputs())[0])
    with pytest.raises(ValueError, match="exact saved recipe"):
        candidate().load_state_dict(saved, strict=True)
