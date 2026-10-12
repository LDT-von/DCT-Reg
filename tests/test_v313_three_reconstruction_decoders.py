"""Independent auxiliary heads; synthetic tests establish no survival gain."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from test_dct_v313_transport_reconstruction import make_args, batch, train_reference
from survot_rank.config import apply_overrides, config_to_argv
from survot_rank.training.extended_args import process_args_extended
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import DCTV313TransportReconstruction
from scripts.prepare_v313_repaired_ablation import build_configs


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model(mode="separate", **overrides):
    values = dict(dct_v313_revision="repaired", dct_v313_omics_decoder_mode=mode,
                  dct_v313_lambda_wsi_reconstruction=0.05, dct_slot_init_mode="deterministic")
    values.update(overrides)
    torch.manual_seed(41)
    net = DCTV313TransportReconstruction(make_args(**values))
    net.configure_train_reference(*train_reference())
    return net


def test_three_heads_have_independent_parameter_storage_and_equal_omics_initialization():
    net = model()
    assert net.reconstruction_decoder_layout()["decoder_count"] == 3
    self_parameters = dict(net.pathway_reconstruction_decoder.named_parameters())
    cross_parameters = dict(net.cross_pathway_reconstruction_decoder.named_parameters())
    assert self_parameters.keys() == cross_parameters.keys()
    for name, parameter in self_parameters.items():
        other = cross_parameters[name]
        assert parameter is not other
        assert parameter.data_ptr() != other.data_ptr()
        assert torch.equal(parameter, other)


def test_switch_keeps_rng_all_existing_weights_and_initial_training_outputs_identical():
    shared = model("shared").train()
    rng = torch.random.get_rng_state()
    separate = model().train()
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in shared.state_dict().items():
        assert torch.equal(value, separate.state_dict()[name]), name
    torch.manual_seed(53)
    left_logits, left_aux = shared(**batch())
    torch.manual_seed(53)
    right_logits, right_aux = separate(**batch())
    torch.testing.assert_close(left_logits, right_logits, rtol=0, atol=0)
    torch.testing.assert_close(left_aux, right_aux, rtol=0, atol=0)
    assert shared.objective_weights_effective() == separate.objective_weights_effective()
    extra = sum(p.numel() for p in separate.cross_pathway_reconstruction_decoder.parameters())
    assert sum(p.numel() for p in separate.parameters()) == sum(p.numel() for p in shared.parameters()) + extra


@pytest.mark.parametrize("cross_mode", ["transport", "direct"])
def test_self_and_cross_losses_update_their_own_decoder_and_backbone_inputs(cross_mode):
    net = model(dct_v313_cross_mode=cross_mode).train()
    sw = torch.randn(2, 3, 16, requires_grad=True)
    so = torch.randn(2, 3, 16, requires_grad=True)
    target = torch.randn(2, 5, 16, requires_grad=True)
    costs, rows, cols, _ = net._cost_tensor(sw, so)
    plans, _ = net._plans_from_cost_tensor(costs, rows, cols, epoch=7)
    _, gate = net._encode_logits_from_plans(sw, so, plans)
    self_loss, cross_loss, _, _ = net.reconstruction_losses(
        x_omics=target, slots_wsi=sw, slots_omic=so, factual_plans=plans,
        factual_gate=gate, available=torch.ones(2, dtype=torch.bool),
    )
    self_loss.backward()
    assert net.pathway_reconstruction_decoder.pathway_queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.cross_pathway_reconstruction_decoder.parameters())
    assert so.grad.abs().sum() > 0
    assert target.grad is None
    net.zero_grad(set_to_none=True)
    sw.grad = so.grad = None
    cross_loss.backward()
    assert net.cross_pathway_reconstruction_decoder.pathway_queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.pathway_reconstruction_decoder.parameters())
    assert sw.grad.abs().sum() > 0
    if cross_mode == "transport":
        assert so.grad.abs().sum() > 0
    else:
        assert so.grad is None
    assert target.grad is None


def test_auxiliary_losses_really_backpropagate_into_main_model_parameters():
    net = model().train()
    _, aux = net(**batch())
    aux.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.wsi_mlp.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.sig_networks.parameters())
    assert net.stage_pair_cost[-1].weight.grad.abs().sum() > 0
    for decoder in (net.pathway_reconstruction_decoder, net.cross_pathway_reconstruction_decoder, net.wsi_reconstruction_decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters() if p.requires_grad)


def test_prediction_loss_does_not_update_any_auxiliary_decoder():
    net = model().train()
    logits, _ = net(**batch())
    logits.square().mean().backward()
    for decoder in (net.pathway_reconstruction_decoder, net.cross_pathway_reconstruction_decoder, net.wsi_reconstruction_decoder):
        assert all(p.grad is None for p in decoder.parameters())


@pytest.mark.parametrize("padding", [0.0, float("nan")])
def test_three_head_training_keeps_padding_outputs_and_gradients_invariant(padding):
    net = model().train()
    other = copy.deepcopy(net)
    payload = batch()
    padded = {key: value.clone() for key, value in payload.items()}
    padded["x_wsi"] = torch.cat([payload["x_wsi"], torch.full((8, 2, 16), padding)], dim=1).requires_grad_()
    padded["wsi_patch_mask"] = torch.tensor([[True]*6 + [False]*2]*8)
    torch.manual_seed(59)
    left_logits, left_aux = net(**payload)
    torch.manual_seed(59)
    right_logits, right_aux = other(**padded)
    torch.testing.assert_close(left_logits, right_logits, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(left_aux, right_aux, atol=3e-6, rtol=1e-5)
    left_aux.backward()
    right_aux.backward()
    for (name, parameter), (_, other_parameter) in zip(net.named_parameters(), other.named_parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(other_parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=1e-5, rtol=1e-4, msg=name)
    assert padded["x_wsi"].grad[:, 6:].abs().sum() == 0


def test_survival_inference_never_calls_any_reconstruction_decoder():
    shared = model("shared").eval()
    separate = model().eval()
    def forbidden(*args):
        raise AssertionError("reconstruction decoder entered survival inference")
    handles = []
    for net in (shared, separate):
        for decoder in (net.pathway_reconstruction_decoder, net.cross_pathway_reconstruction_decoder, net.wsi_reconstruction_decoder):
            if decoder is not None:
                handles.append(decoder.register_forward_pre_hook(forbidden))
    try:
        with torch.no_grad():
            left, left_aux = shared(**batch())
            right, right_aux = separate(**batch())
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left_aux == right_aux == 0.0
    finally:
        for handle in handles:
            handle.remove()


def test_checkpoint_layout_is_explicit_and_strict_loading_catches_wrong_mode():
    net = model()
    restored = model()
    restored.load_state_dict(net.state_dict(), strict=True)
    assert net.implementation_audit()["reconstruction_decoders"]["cross"] == "cross_pathway_reconstruction_decoder"
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model("shared").load_state_dict(net.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        net.load_state_dict(model("shared").state_dict(), strict=True)


@pytest.mark.parametrize("disabled,expected", [("self", "cross"), ("cross", "self")])
def test_disabled_branch_does_not_train_its_decoder(disabled, expected):
    overrides = {"dct_v313_disable_" + disabled + "_reconstruction": True}
    net = model(**overrides).train()
    _, aux = net(**batch())
    aux.backward()
    inactive = net.pathway_reconstruction_decoder if disabled == "self" else net.cross_pathway_reconstruction_decoder
    active = net.pathway_reconstruction_decoder if expected == "self" else net.cross_pathway_reconstruction_decoder
    assert all(p.grad is None for p in inactive.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in active.parameters())


def test_three_decoder_preset_and_shared_control_differ_only_in_head_sharing():
    base = yaml.safe_load(Path("configs/dct_v313_full_wsi_three_decoders.yaml").read_text(encoding="utf-8"))
    configs = build_configs(base, "fresh_runs")
    full, shared = configs["full"], configs["shared_decoder"]
    assert full["model"]["dct_v313_omics_decoder_mode"] == "separate"
    assert shared["model"]["dct_v313_omics_decoder_mode"] == "shared"
    left = copy.deepcopy(full)
    right = copy.deepcopy(shared)
    for item in (left, right):
        item.pop("name")
        item["train"].pop("specific_simple")
        item["train"].pop("results_dir")
        item["model"].pop("dct_v313_omics_decoder_mode")
    assert left == right
    config = apply_overrides(base, [])
    args = process_args_extended(config_to_argv(config))
    assert args.dct_v313_omics_decoder_mode == "separate"
    assert args.dct_v313_lambda_wsi_reconstruction == 0.05


def test_invalid_or_legacy_separate_mode_fails_early_and_shared_base_rejects_duplicate_control():
    with pytest.raises(ValueError, match="shared or separate"):
        model("invalid")
    with pytest.raises(ValueError, match="require.*repaired"):
        model(dct_v313_revision="legacy")
    base = yaml.safe_load(Path("configs/dct_v313_full_wsi_repaired.yaml").read_text(encoding="utf-8"))
    assert "shared_decoder" not in build_configs(base, "fresh_runs")
    with pytest.raises(ValueError, match="separate-decoder Full"):
        build_configs(base, "fresh_runs", ["shared_decoder"])
