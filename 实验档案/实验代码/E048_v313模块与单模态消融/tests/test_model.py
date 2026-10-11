"""Synthetic numerical checks only; these do not measure patient performance."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("e048_model_tested", HERE / "model.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


@pytest.fixture(autouse=True)
def quiet_debug(tmp_path, monkeypatch):
    monkeypatch.setenv("DCT_DEBUG_LOG", str(tmp_path / "constructors.jsonl"))
    parent = sys.modules[controls.DCTV313TransportReconstruction.__module__]
    monkeypatch.setattr(parent, "_DEBUG_LOG_PATH", tmp_path / "constructors.jsonl")
    torch.set_num_threads(1)


def args():
    return SimpleNamespace(
        bag_loss="nll_surv", omic_sizes=[3, 4, 5], n_classes=4,
        encoding_dim=16, wsi_projection_dim=16, rna_format="Pathways",
        alpha_surv=.15, slot_num_wsi=3, slot_num_omics=3, slot_iters=2,
        otehv2_eps=.05, otehv2_iter=15, otehv2_heads=2,
        otehv2_layers=1, otehv2_dropout=0., dct_num_stages=4,
        dct_slot_init_mode="deterministic", dct_slot_eval_seed=91,
        dct_v313_recon_weighting="per_branch", cur_epoch=8,
    )


def payload():
    generator = torch.Generator().manual_seed(13)
    data = dict(x_wsi=torch.randn(4, 6, 16, generator=generator),
                y=torch.tensor([0, 1, 2, 3]), event_time=torch.tensor([1., 4., 10., 14.]),
                c=torch.tensor([0., 0., 1., 1.]))
    data.update({f"x_omic{i}": torch.randn(4, width, generator=generator)
                 for i, width in enumerate([3, 4, 5], start=1)})
    return data


def model(arm):
    result = controls.build_model(args(), arm)
    result.configure_train_reference(torch.tensor([1., 2., 4., 8., 10., 12., 14., 16.]),
                               torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]))
    return result


@pytest.mark.parametrize("arm", controls.ARM_NAMES)
def test_all_arms_train_backward_eval_and_reload(arm):
    torch.manual_seed(17)
    candidate = model(arm)
    candidate.train()
    logits, aux = candidate(**payload())
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all() and torch.isfinite(aux)
    (logits.square().mean() + aux).backward()
    gradients = [p.grad for p in candidate.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is None for p in candidate.parameters() if not p.requires_grad)
    candidate.eval()
    with torch.no_grad():
        expected = candidate(**payload())[0]
    restored = model(arm)
    restored.load_state_dict(copy.deepcopy(candidate.state_dict()))
    restored.eval()
    with torch.no_grad():
        actual = restored(**payload())[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_full_exactly_matches_parent_under_same_resolved_recipe():
    torch.manual_seed(31)
    reference = controls.DCTV313TransportReconstruction(controls.configure_arm(args(), "full"))
    torch.manual_seed(31)
    candidate = model("full")
    reference.configure_train_reference(torch.tensor([1., 2., 4., 8., 10., 12., 14., 16.]),
                                        torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]))
    reference.train()
    candidate.train()
    torch.manual_seed(45)
    expected_logits, expected_aux = reference(**payload())
    torch.manual_seed(45)
    logits, aux = candidate(**payload())
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
    assert set(reference.state_dict()) == set(candidate.state_dict())


@pytest.mark.parametrize("arm", ["wsi_only", "omics_only", "concat"])
def test_modalities_and_concat_bypass_transport(arm, monkeypatch):
    candidate = model(arm)
    def forbidden(*values, **kwargs):
        raise AssertionError("bypassed computation was called")
    monkeypatch.setattr(candidate, "_cost_tensor", forbidden)
    monkeypatch.setattr(candidate, "_plans_from_cost_tensor", forbidden)
    data = payload()
    if arm == "wsi_only":
        data = {key: value for key, value in data.items() if not key.startswith("x_omic")}
        monkeypatch.setattr(candidate, "_encode_omics", forbidden)
        assert not any(p.requires_grad for p in candidate.sig_networks.parameters())
    if arm == "omics_only":
        data.pop("x_wsi")
        monkeypatch.setattr(candidate.wsi_mlp, "forward", forbidden)
        assert not any(p.requires_grad for p in candidate.wsi_mlp.parameters())
    candidate.train()
    logits, aux = candidate(**data)
    (logits.square().mean() + aux).backward()
    candidate.eval()
    assert torch.isfinite(candidate(**data)[0]).all()


@pytest.mark.parametrize("arm,self_coef,cross_coef", [
    ("full", .05, .05), ("no_self", 0., .05), ("no_cross", .05, 0.),
    ("no_reconstruction", 0., 0.), ("wsi_only", 0., 0.), ("omics_only", .05, 0.),
])
def test_reconstruction_coefficients_do_not_renormalize(arm, self_coef, cross_coef):
    audit = model(arm).audit()
    assert audit["self"] == self_coef and audit["cross"] == cross_coef
    assert audit["wsi_reconstruction"] == 0


@pytest.mark.parametrize("arm,key", [("no_rank", "rank"), ("no_slot_nll", "slot_nll"),
                                     ("no_diversity", "diversity")])
def test_removed_auxiliary_has_zero_effective_coefficient(arm, key):
    assert model(arm).audit()[key] == 0.


@pytest.mark.parametrize("arm", ["shared_stage_plan", "single_geometry"])
def test_structural_cost_ties_are_real_in_both_modes(arm):
    candidate = model(arm)
    for training in (True, False):
        candidate.train(training)
        data = payload()
        sw, so, _, _ = candidate._encode_transport_slots(candidate.wsi_mlp(data["x_wsi"]),
                                                         candidate._encode_omics(data), data)
        costs, rows, cols, gates = candidate._cost_tensor(sw, so)
        if arm == "shared_stage_plan":
            for tensor in (costs, rows, cols, gates):
                torch.testing.assert_close(tensor, tensor[:, :1].expand_as(tensor))
        else:
            torch.testing.assert_close(costs, costs[:, :, :1].expand_as(costs))
        assert candidate.spt_num_stages == candidate.num_classes == 4


def test_no_prototypes_preserves_local_slots_and_valid_identity_diagnostics():
    candidate = model("no_prototypes")
    candidate.eval()
    data = payload()
    wsi, omics = candidate.wsi_mlp(data["x_wsi"]), candidate._encode_omics(data)
    sw, so, aw, ao = candidate._encode_transport_slots(wsi, omics, data)
    torch.testing.assert_close(sw, candidate.slot_attention_wsi(wsi))
    torch.testing.assert_close(so, candidate.slot_attention_omic(omics))
    for assignment in (aw, ao):
        torch.testing.assert_close(assignment, torch.eye(3).expand(4, -1, -1))
    assert torch.isfinite(candidate(**data)[0]).all()


def test_no_event_encoder_never_calls_transformer_train_or_eval(monkeypatch):
    candidate = model("no_event_encoder")
    def forbidden(*values, **kwargs):
        raise AssertionError("event encoder should be bypassed")
    monkeypatch.setattr(candidate.event_encoder, "forward", forbidden)
    for training in (True, False):
        candidate.train(training)
        assert torch.isfinite(candidate(**payload())[0]).all()


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="Unknown E048 arm"):
        model("typo")
