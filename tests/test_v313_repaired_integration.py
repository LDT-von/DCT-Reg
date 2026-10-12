"""Synthetic integration checks; these establish no survival-performance gain."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_dct_v313_transport_reconstruction import make_args, batch, train_reference
from survot_rank.config import load_config, apply_overrides, config_to_argv
from survot_rank.training.extended_args import process_args_extended
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import DCTV313TransportReconstruction
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.provenance import write_implementation_audit, claim_repaired_results_root


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def repaired(**changes):
    torch.manual_seed(19)
    values = dict(dct_v313_revision="repaired", dct_v313_lambda_wsi_reconstruction=0.05,
                  dct_slot_init_mode="deterministic")
    values.update(changes)
    net = DCTV313TransportReconstruction(make_args(**values))
    net.configure_train_reference(*train_reference())
    return net


def test_repaired_default_and_legacy_are_explicitly_distinct():
    net = repaired()
    assert net.objective_weights_effective()["slot_diversity"] == 0.10
    c = net.effective_recon_coefficients()
    assert [c[key] for key in ("self", "cross", "wsi", "total")] == pytest.approx([0.05, 0.05, 0.05, 0.15])
    assert net.implementation_audit()["input_padding_mask"] is True
    assert net.implementation_audit()["missing_omics_recovery_proven"] is False
    legacy = DCTV313TransportReconstruction(make_args())
    assert legacy._v313_revision == "legacy"
    assert legacy.objective_weights_effective()["slot_diversity"] == 0.02
    assert legacy.wsi_reconstruction_decoder is None
    assert "revision" not in legacy.effective_recon_coefficients()


def test_repaired_yaml_values_reach_the_actual_constructor():
    config = load_config("configs/dct_v313_full_wsi_repaired.yaml")
    config = apply_overrides(config, ["dct_v311_lambda_slot_diversity=0.07"])
    args = process_args_extended(config_to_argv(config))
    args._dct_user_overrides = set(config["_dct_user_overrides"])
    args.omic_sizes = [3, 4, 5, 6, 7]
    args.encoding_dim = args.wsi_projection_dim = 16
    args.slot_num_wsi = args.slot_num_omics = 3
    args.slot_iters = 2
    args.otehv2_heads = 2
    net = DCTV313TransportReconstruction(args)
    assert net._v313_revision == "repaired"
    assert net.objective_weights_effective()["slot_diversity"] == 0.07
    assert net.dct_v313_reconstruction_wsi_coef == 0.05
    assert net.dct_v313_reconstruction_self_coef == 0.05


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("padding_value", [0.0, 1000.0, float("nan")])
def test_padding_preserves_complete_forward_and_parameter_gradients(training, padding_value):
    net = repaired()
    net.train(training)
    padded_net = copy.deepcopy(net)
    payload = batch()
    padded = {key: value.clone() for key, value in payload.items()}
    extra = torch.full((8, 3, 16), padding_value)
    padded["x_wsi"] = torch.cat([payload["x_wsi"], extra], dim=1).requires_grad_()
    padded["wsi_patch_mask"] = torch.tensor([[True]*6 + [False]*3]*8)
    torch.manual_seed(23)
    logits, aux = net(**payload)
    torch.manual_seed(23)
    padded_logits, padded_aux = padded_net(**padded)
    torch.testing.assert_close(logits, padded_logits, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(torch.as_tensor(aux), torch.as_tensor(padded_aux), atol=3e-6, rtol=1e-5)
    (logits.square().mean() + aux).backward()
    (padded_logits.square().mean() + padded_aux).backward()
    for (name, parameter), (other_name, other) in zip(net.named_parameters(), padded_net.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            assert other.grad is not None, name
            assert torch.isfinite(other.grad).all(), name
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-5, rtol=1e-4, msg=name)
    assert torch.equal(padded["x_wsi"].grad[:, 6:], torch.zeros_like(extra))


def test_repaired_auxiliary_is_the_logged_three_branch_sum():
    net = repaired().train()
    _, auxiliary = net(**batch())
    losses = net.last_training_losses
    expected = (0.10 * losses["ipcw_rank"] + 0.05 * losses["v311_per_slot_nll"]
                + 0.10 * losses["v311_slot_diversity"]
                + 0.05 * losses["v313_reconstruction_self"]
                + 0.05 * losses["v313_reconstruction_cross"]
                + 0.05 * losses["v313_reconstruction_wsi"])
    torch.testing.assert_close(auxiliary, expected)
    assert "v313_diversity_wsi_duplicate_pair_fraction" in losses
    assert "v313_diversity_omic_full_collapse_patient_fraction" in losses
    auxiliary.backward()
    assert net.wsi_reconstruction_decoder.cross_attention.in_proj_weight.grad.abs().sum() > 0
    assert net.pathway_reconstruction_decoder.pathway_queries.grad.abs().sum() > 0


def test_explicit_patch_mask_is_authoritative_and_missing_patients_remain_empty():
    net = repaired()
    raw = torch.zeros(2, 4, 16)
    explicit = torch.tensor([[True, False, False, False], [True]*4])
    valid = net._wsi_valid_patch_mask(dict(x_wsi=raw, wsi_patch_mask=explicit, wsi_available=[True, False]))
    assert torch.equal(valid, torch.tensor([[True, False, False, False], [False]*4]))
    with pytest.raises(ValueError, match="boolean"):
        net._wsi_valid_patch_mask(dict(x_wsi=raw, wsi_patch_mask=explicit.float()))


def test_all_invalid_wsi_forward_is_finite_and_loss_ignores_patches():
    net = repaired().train()
    payload = batch()
    payload["x_wsi"].zero_()
    logits, auxiliary = net(**payload)
    assert torch.isfinite(logits).all() and torch.isfinite(auxiliary)
    assert net.last_training_losses["v313_reconstruction_wsi"].item() == 0
    assert net.last_training_losses["v313_valid_wsi_patch_mean"].item() == 0


def test_audit_records_sources_weights_and_refuses_overwrite(tmp_path):
    net = repaired()
    args = SimpleNamespace(results_dir=str(tmp_path), cur_fold=2, seed=3,
                           _dct_user_overrides={"dct_v313_revision"})
    path = write_implementation_audit(net, args)
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["fold"] == 2
    assert content["implementation"]["reconstruction"]["wsi"] == 0.05
    assert content["implementation"]["objective_weights"]["slot_diversity"] == 0.10
    assert "survot_rank/research/components/slot_attention.py" in content["source_sha256"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_implementation_audit(net, args)
    assert path.read_bytes() == before


def test_trainer_writes_fold_audit_before_training(tmp_path, monkeypatch):
    from survot_rank.training import train_runner
    net = repaired()
    factory = SimpleNamespace(omic_sizes=[3, 4, 5, 6, 7], omic_names=[], pathway_names=[])
    args = SimpleNamespace(rna_format="Pathways", survot_method="dct_v313", results_dir=str(tmp_path), cur_fold=1)
    monkeypatch.setattr(train_runner, "get_model", lambda **kwargs: net)
    monkeypatch.setattr(train_runner, "_print_network", lambda *args: None)
    monkeypatch.setattr(train_runner.torch.cuda, "is_available", lambda: False)
    assert train_runner.init_model_for_method(args, factory) is net
    assert (tmp_path / "v313_implementation_fold1.json").is_file()
    before = (tmp_path / "v313_implementation_fold1.json").read_bytes()
    assert train_runner.init_model_for_method(args, factory, write_audit=False) is net
    assert (tmp_path / "v313_implementation_fold1.json").read_bytes() == before


def test_explicit_reconstruction_coefficients_and_ramp_are_consumed():
    values = dict(dct_v313_lambda_reconstruction=0.20,
                  dct_v313_reconstruction_self_fraction=0.25,
                  dct_v313_reconstruction_cross_fraction=0.75,
                  dct_v313_reconstruction_ramp_start=1,
                  dct_v313_reconstruction_ramp_epochs=2)
    net = repaired(**values, _dct_user_overrides=set(values))
    assert net.dct_v313_reconstruction_self_coef == pytest.approx(0.05)
    assert net.dct_v313_reconstruction_cross_coef == pytest.approx(0.15)
    assert net.dct_v313_reconstruction_wsi_coef == pytest.approx(0.05)
    assert net._reconstruction_ramp(2) == 0.5
    assert net.implementation_audit()["reconstruction_ramp"] == {"start": 1, "epochs": 2}


def test_repaired_rejects_legacy_reconstruction_weighting():
    with pytest.raises(ValueError, match="requires.*per_branch"):
        repaired(dct_v313_recon_weighting="legacy", dct_v313_lambda_wsi_reconstruction=0.0)


def test_label_free_training_forward_has_no_stale_diversity_diagnostics():
    net = repaired().train()
    net(**batch())
    payload = {key: value for key, value in batch().items() if key not in {"y", "c", "event_time"}}
    logits, auxiliary = net(**payload)
    assert torch.isfinite(logits).all() and torch.isfinite(auxiliary)
    assert net.last_diversity_diagnostics == {}
    assert net._last_slot_variance == 0.0
    assert not any(key.startswith("v313_diversity_") for key in net.last_training_losses)


def test_constructor_defaults_are_not_mislabeled_as_explicit_config(tmp_path):
    net = repaired()
    path = write_implementation_audit(net, SimpleNamespace(results_dir=tmp_path, cur_fold=0))
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert "dct_v311_lambda_slot_diversity" not in audit["explicit_config_keys"]


def test_repaired_preflight_preserves_old_results_before_any_runner_work(tmp_path, monkeypatch):
    from survot_rank.training import train_runner
    old = tmp_path / "old"
    old.mkdir()
    log = old / "log_start_0_end_5.txt"
    log.write_text("historical evidence", encoding="utf-8")
    args = SimpleNamespace(results_dir=str(old), dct_v313_revision="repaired", only_test=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("runner work must not start before collision detection")
    monkeypatch.setattr(train_runner, "set_global_seed", forbidden)
    with pytest.raises(FileExistsError, match="new results_dir"):
        train_runner.run(args)
    assert log.read_text(encoding="utf-8") == "historical evidence"
    args.results_dir = str(tmp_path / "new")
    assert claim_repaired_results_root(args).is_dir()


def test_optional_none_availability_flags_use_defaults():
    net = repaired()
    payload = batch()
    expected = net._wsi_valid_patch_mask(payload)
    payload.update(wsi_available=None, wsi_missing=None)
    assert torch.equal(net._wsi_valid_patch_mask(payload), expected)


def test_e048_cannot_silently_disable_wsi_in_a_repaired_full():
    path = Path("实验档案/实验代码/E048_v313模块与单模态消融/model.py")
    spec = importlib.util.spec_from_file_location("e048_repaired_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="legacy two-reconstruction"):
        module.E048Control(make_args(dct_v313_revision="repaired"), "full")
