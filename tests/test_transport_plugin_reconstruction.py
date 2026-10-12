"""Auxiliary head properties without loading a survival model or patient data."""

from __future__ import annotations

import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from survot_rank.plugins.transport.reconstruction import TransportReconstructionHeads


@pytest.fixture(autouse=True)
def _small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(previous)


def _state(batch=3, patches=5, pathways=4, dim=8, grad=False):
    return SimpleNamespace(
        wsi_tokens=torch.randn(batch, patches, dim, requires_grad=grad),
        omics_tokens=torch.randn(batch, pathways, dim, requires_grad=grad),
        wsi_mask=torch.ones(batch, patches, dtype=torch.bool),
        omics_mask=torch.ones(batch, pathways, dtype=torch.bool),
        wsi_slots=torch.randn(batch, 2, dim, requires_grad=grad),
        omics_slots=torch.randn(batch, 3, dim, requires_grad=grad),
        transported_wsi=torch.randn(batch, 3, dim, requires_grad=grad),
    )


def _heads(**kwargs):
    return TransportReconstructionHeads(8, 4, heads=2, **kwargs)


def _patient(state, index):
    return SimpleNamespace(**{
        name: value[index:index + 1] for name, value in vars(state).items()
    })


def test_separate_decoders_start_equal_without_shared_storage():
    module = _heads()
    self_parameters = dict(module.self_decoder.named_parameters())
    cross_parameters = dict(module.cross_decoder.named_parameters())
    assert self_parameters.keys() == cross_parameters.keys()
    for name, parameter in self_parameters.items():
        assert torch.equal(parameter, cross_parameters[name])
        assert parameter.data_ptr() != cross_parameters[name].data_ptr()
    wsi_ptrs = {parameter.data_ptr() for parameter in module.wsi_decoder.parameters()}
    assert wsi_ptrs.isdisjoint(parameter.data_ptr() for parameter in self_parameters.values())
    assert wsi_ptrs.isdisjoint(parameter.data_ptr() for parameter in cross_parameters.values())
    assert not module.wsi_decoder.query_projection.weight.requires_grad
    with torch.no_grad():
        module.self_decoder.queries.add_(1.0)
    assert not torch.equal(module.self_decoder.queries, module.cross_decoder.queries)


def test_shared_control_only_shares_omics_decoders():
    module = _heads(mode="shared")
    assert module.self_decoder is module.cross_decoder
    assert module.wsi_decoder is not module.self_decoder
    total, losses = module(_state(grad=True), epoch=7)
    total.backward()
    assert losses["self"].item() > 0 and losses["cross"].item() > 0
    assert module.self_decoder.queries.grad is not None


def test_detached_targets_but_slot_memories_receive_gradients():
    module = _heads()
    state = _state(grad=True)
    total, _ = module(state, epoch=7)
    total.backward()
    assert state.wsi_tokens.grad is None
    assert state.omics_tokens.grad is None
    for name in ("wsi_slots", "omics_slots", "transported_wsi"):
        gradient = getattr(state, name).grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert module.wsi_decoder.query_projection.weight.grad is None


def test_padding_nan_and_unavailable_memory_do_not_pollute_loss():
    module = _heads()
    state = _state()
    state.wsi_mask[0, 2:] = False
    state.omics_mask[0, 1:] = False
    state.wsi_mask[1] = False
    state.omics_mask[2] = False
    reference, reference_losses = module(state, epoch=7)
    corrupted = copy.deepcopy(state)
    corrupted.wsi_tokens.masked_fill_(~state.wsi_mask.unsqueeze(-1), float("nan"))
    corrupted.omics_tokens.masked_fill_(~state.omics_mask.unsqueeze(-1), float("nan"))
    corrupted.wsi_slots[1] = float("nan")
    corrupted.omics_slots[2] = float("nan")
    corrupted.transported_wsi[1:] = float("nan")
    actual, actual_losses = module(corrupted, epoch=7)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, reference)
    for name in ("self", "cross", "wsi"):
        torch.testing.assert_close(actual_losses[name], reference_losses[name])


def test_per_patient_average_respects_each_branch_availability():
    module = _heads()
    state = _state()
    state.wsi_mask[0, 2:] = False
    state.omics_mask[0, 1:] = False
    state.wsi_mask[1] = False
    state.omics_mask[2] = False
    _, batched = module(state, epoch=7)
    singles = [module(_patient(state, index), epoch=7)[1] for index in range(3)]
    for branch, indices in (("self", [0, 1]), ("cross", [0]), ("wsi", [0, 2])):
        expected = torch.stack([singles[index][branch] for index in indices]).mean()
        torch.testing.assert_close(batched[branch], expected)


@pytest.mark.parametrize("patches,pathways", [(0, 0), (5, 4)])
def test_empty_valid_sets_return_finite_graph_connected_zero(patches, pathways):
    module = _heads()
    state = _state(patches=patches, pathways=pathways, grad=True)
    state.wsi_mask[:] = False
    state.omics_mask[:] = False
    with torch.no_grad():
        state.wsi_tokens[:] = float("nan")
        state.omics_tokens[:] = float("nan")
        state.wsi_slots[:] = float("nan")
        state.omics_slots[:] = float("nan")
        state.transported_wsi[:] = float("nan")
    total, losses = module(state, epoch=7)
    assert total.requires_grad and total.item() == 0
    total.backward()
    for name in ("wsi_slots", "omics_slots", "transported_wsi"):
        assert torch.isfinite(getattr(state, name).grad).all()
        assert torch.count_nonzero(getattr(state, name).grad) == 0
    assert all(losses[name].item() == 0 for name in ("self", "cross", "wsi"))


@pytest.mark.parametrize("epoch,expected", [(0, 0), (2, 0), (3, 0.2), (6, 0.8), (7, 1), (20, 1)])
def test_ramp_and_explicit_branch_coefficients(epoch, expected):
    module = _heads(self_weight=0.02, cross_weight=0.03, wsi_weight=0.04)
    total, losses = module(_state(), epoch=epoch)
    assert losses["ramp"].item() == pytest.approx(expected)
    expected_total = expected * (
        0.02 * losses["self"] + 0.03 * losses["cross"] + 0.04 * losses["wsi"]
    )
    torch.testing.assert_close(total, expected_total)


def test_ramp_zero_skips_every_decoder():
    module = _heads()
    with patch.object(module.self_decoder, "forward", side_effect=AssertionError), \
         patch.object(module.cross_decoder, "forward", side_effect=AssertionError), \
         patch.object(module.wsi_decoder, "forward", side_effect=AssertionError):
        total, _ = module(_state(), epoch=2)
    assert total.item() == 0


@pytest.mark.parametrize("branch", ["self", "cross", "wsi"])
def test_zero_branch_weight_skips_its_decoder(branch):
    module = _heads(**{branch + "_weight": 0.0})
    decoder = getattr(module, branch + "_decoder")
    with patch.object(decoder, "forward", side_effect=AssertionError):
        total, losses = module(_state(), epoch=7)
    assert torch.isfinite(total)
    assert losses[branch].item() == 0


def test_wsi_chunk_size_preserves_loss_and_memory_gradient():
    full = _heads(wsi_chunk_size=100)
    # This deliberately changes only workspace chunking after an explicit
    # in-memory copy; a checkpoint load under another config is rejected.
    chunked = copy.deepcopy(full)
    chunked.wsi_chunk_size = 2
    state = _state(patches=7, grad=True)
    state.wsi_mask[0, 4:] = False
    state_copy = copy.deepcopy(state)
    full_total, full_losses = full(state, epoch=7)
    chunked_total, chunked_losses = chunked(state_copy, epoch=7)
    torch.testing.assert_close(full_total, chunked_total)
    torch.testing.assert_close(full_losses["wsi"], chunked_losses["wsi"])
    full_total.backward()
    chunked_total.backward()
    torch.testing.assert_close(state.wsi_slots.grad, state_copy.wsi_slots.grad, atol=1e-7, rtol=1e-5)


def test_shorter_omics_inputs_and_configured_maximum():
    module = _heads()
    total, _ = module(_state(pathways=2), epoch=7)
    assert torch.isfinite(total)
    with pytest.raises(ValueError, match="exceeding configured num_omics_tokens=4"):
        module(_state(pathways=5), epoch=7)


@pytest.mark.parametrize("mode", ["separate", "shared"])
def test_state_dict_roundtrip_preserves_predictions_and_alias_policy(mode):
    module = _heads(mode=mode)
    restored = _heads(mode=mode)
    restored.load_state_dict(copy.deepcopy(module.state_dict()))
    state = _state()
    total, losses = module(state, epoch=7)
    restored_total, restored_losses = restored(state, epoch=7)
    torch.testing.assert_close(total, restored_total)
    for name in losses:
        torch.testing.assert_close(losses[name], restored_losses[name])
    assert (restored.self_decoder is restored.cross_decoder) == (mode == "shared")


def test_wsi_has_no_target_residual_when_slot_values_are_zero():
    module = _heads()
    decoder = module.wsi_decoder
    # Attention values, attention output and MLP biases are zero here. Output
    # must remain zero despite nonzero target-conditioned queries.
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            if "bias" in name:
                parameter.zero_()
    prediction = decoder(torch.zeros(1, 2, 8), torch.randn(1, 5, 8))
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))


class _NativePathwayDecoder(nn.Module):
    """The native public head naming and full-pathway forward contract."""

    def __init__(self, template):
        super().__init__()
        self.pathway_queries = copy.deepcopy(template.queries)
        for name, module in template.named_children():
            self.add_module(name, copy.deepcopy(module))

    def forward(self, memory):
        queries = self.pathway_queries.expand(memory.size(0), -1, -1)
        keys = self.memory_norm(memory)
        decoded, _ = self.cross_attention(self.query_norm(queries), keys, keys, need_weights=False)
        decoded = queries + decoded
        return decoded + self.output_mlp(self.output_norm(decoded))


class _NativeHost(nn.Module):
    def __init__(self, mode="separate", wsi=True):
        super().__init__()
        template = _heads()
        self.pathway_reconstruction_decoder = _NativePathwayDecoder(template.self_decoder)
        self.cross_pathway_reconstruction_decoder = (
            _NativePathwayDecoder(template.cross_decoder) if mode == "separate" else None
        )
        self.wsi_reconstruction_decoder = copy.deepcopy(template.wsi_decoder) if wsi else None
        if self.wsi_reconstruction_decoder is not None:
            self.wsi_reconstruction_decoder.chunk_size = 2
        self._omics_decoder_mode = mode
        self.dct_v313_reconstruction_ramp_start = 5
        self.dct_v313_reconstruction_ramp_epochs = 3
        self.coefficients = {
            "self": 0.07, "cross": 0.02, "wsi": 0.01 if wsi else 0.0,
            "omics_decoder_mode": mode, "cross_mode": "transport",
        }
        if self.cross_pathway_reconstruction_decoder is not None:
            with torch.no_grad():
                self.cross_pathway_reconstruction_decoder.pathway_queries.add_(0.3)

    def effective_recon_coefficients(self):
        return dict(self.coefficients)


@pytest.mark.parametrize("mode", ["separate", "shared"])
def test_copy_native_heads_weights_effective_budget_and_ramp_without_host_alias(mode):
    host = _NativeHost(mode=mode)
    rng_before = torch.get_rng_state()
    module = TransportReconstructionHeads.from_dct(host)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert module.self_weight == 0.07 and module.cross_weight == 0.02 and module.wsi_weight == 0.01
    assert module.ramp_start == 5 and module.ramp_epochs == 3 and module.wsi_chunk_size == 2
    assert (module.self_decoder is module.cross_decoder) == (mode == "shared")
    assert all(child is not host for child in module.modules())
    assert {p.data_ptr() for p in module.parameters()}.isdisjoint(p.data_ptr() for p in host.parameters())
    memory = torch.randn(2, 3, 8)
    torch.testing.assert_close(module.self_decoder(memory, 4), host.pathway_reconstruction_decoder(memory))
    native_cross = host.cross_pathway_reconstruction_decoder or host.pathway_reconstruction_decoder
    torch.testing.assert_close(module.cross_decoder(memory, 4), native_cross(memory))
    for name, value in host.wsi_reconstruction_decoder.state_dict().items():
        torch.testing.assert_close(module.wsi_decoder.state_dict()[name], value)
    assert not module.wsi_decoder.query_projection.weight.requires_grad


def test_copy_native_off_wsi_retains_no_random_wsi_decoder():
    host = _NativeHost(mode="shared", wsi=False)
    module = TransportReconstructionHeads.from_dct(host)
    assert module.wsi_weight == 0 and module.wsi_decoder is None
    total, losses = module(_state(), epoch=8)
    assert torch.isfinite(total) and losses["wsi"].item() == 0


@pytest.mark.parametrize("case,message", [
    ("all_off", "all native reconstruction branches are disabled"),
    ("direct", "native direct control"),
    ("missing_cross", "separate cross_pathway_reconstruction_decoder is missing"),
    ("missing_wsi", "WSI coefficient is active"),
])
def test_copy_native_rejects_unsupported_or_inconsistent_states(case, message):
    host = _NativeHost()
    if case == "all_off":
        host.coefficients.update(self=0.0, cross=0.0, wsi=0.0)
    elif case == "direct":
        host.coefficients["cross_mode"] = "direct"
    elif case == "missing_cross":
        host.cross_pathway_reconstruction_decoder = None
    else:
        host.wsi_reconstruction_decoder = None
    with pytest.raises(ValueError, match=message):
        TransportReconstructionHeads.from_dct(host)


def test_copy_native_requires_explicit_effective_coefficient_api():
    with pytest.raises(ValueError, match="effective_recon_coefficients"):
        TransportReconstructionHeads.from_dct(nn.Linear(8, 8))


@pytest.mark.parametrize("mode", ["separate", "shared"])
def test_configuration_manifest_and_serialized_checkpoint_roundtrip(mode):
    source = _heads(mode=mode, self_weight=0.02, cross_weight=0.03,
                    wsi_weight=0.04, ramp_start=4, ramp_epochs=3, wsi_chunk_size=2)
    manifest = json.loads(json.dumps(source.configuration()))
    restored = TransportReconstructionHeads.from_config(manifest)
    checkpoint = io.BytesIO()
    torch.save(source.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert restored.configuration() == manifest
    state = _state()
    torch.testing.assert_close(source(state, epoch=7)[0], restored(state, epoch=7)[0])
    assert (restored.self_decoder is restored.cross_decoder) == (mode == "shared")


@pytest.mark.parametrize("overrides,field", [
    ({"self_weight": 0.04}, "self_weight"),
    ({"cross_weight": 0.04}, "cross_weight"),
    ({"wsi_weight": 0.04}, "wsi_weight"),
    ({"ramp_start": 3}, "ramp_start"),
    ({"ramp_epochs": 6}, "ramp_epochs"),
    ({"mode": "shared"}, "mode"),
    ({"wsi_chunk_size": 3}, "wsi_chunk_size"),
])
def test_checkpoint_rejects_same_shape_weights_under_a_different_objective(overrides, field):
    source = _heads()
    incompatible = _heads(**overrides)
    with pytest.raises(ValueError, match=field):
        incompatible.load_state_dict(source.state_dict())


def test_checkpoint_cannot_bypass_missing_objective_metadata_with_non_strict_loading():
    source = _heads()
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint.pop("_extra_state")
    with pytest.raises(RuntimeError, match="reconstruction checkpoint metadata is missing"):
        _heads().load_state_dict(checkpoint, strict=False)


def test_native_disabled_wsi_layout_manifest_and_checkpoint_roundtrip():
    source = TransportReconstructionHeads.from_dct(_NativeHost(wsi=False))
    manifest = json.loads(json.dumps(source.configuration()))
    assert manifest["wsi_head_enabled"] is False
    restored = TransportReconstructionHeads.from_config(manifest)
    assert restored.wsi_decoder is None
    restored.load_state_dict(source.state_dict())
    state = _state()
    torch.testing.assert_close(source(state, epoch=8)[0], restored(state, epoch=8)[0])
    extra_unused_wsi = _heads(self_weight=0.07, cross_weight=0.02, wsi_weight=0.0,
                             ramp_start=5, ramp_epochs=3)
    with pytest.raises(ValueError, match="wsi_head_enabled"):
        extra_unused_wsi.load_state_dict(source.state_dict())


def test_manifest_rejects_enabled_weight_for_a_missing_wsi_head():
    manifest = _heads().configuration()
    manifest["wsi_head_enabled"] = False
    with pytest.raises(ValueError, match="disabled WSI head requires wsi_weight=0"):
        TransportReconstructionHeads.from_config(manifest)


def test_checkpoint_metadata_validates_dimensions_heads_and_pathway_limit():
    source = _heads()
    saved = copy.deepcopy(source.get_extra_state())
    for name, different in (("dim", 16), ("heads", 4), ("num_omics_tokens", 5)):
        changed = copy.deepcopy(saved)
        changed["configuration"][name] = different
        with pytest.raises(ValueError, match=name):
            source.set_extra_state(changed)
