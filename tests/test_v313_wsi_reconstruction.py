from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from test_dct_v313_transport_reconstruction import make_args, batch, train_reference
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import DCTV313TransportReconstruction
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.wsi_reconstruction import WSISlotReconstructionDecoder


def model(weight=0.05, budget="additive", **kwargs):
    return DCTV313TransportReconstruction(make_args(
        dct_v313_lambda_wsi_reconstruction=weight,
        dct_v313_wsi_reconstruction_budget=budget, **kwargs,
    ))


def test_disabled_preserves_checkpoint_rng_and_shared_initialization():
    torch.manual_seed(123)
    old = DCTV313TransportReconstruction(make_args())
    continuation = torch.random.get_rng_state()
    torch.manual_seed(123)
    off = model(0)
    assert torch.equal(continuation, torch.random.get_rng_state())
    assert off.wsi_reconstruction_decoder is None
    off.load_state_dict(old.state_dict(), strict=True)
    assert off.effective_recon_coefficients() == old.effective_recon_coefficients()
    torch.manual_seed(123)
    on = model()
    assert torch.equal(continuation, torch.random.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, on.state_dict()[name]), name
    assert "reconstruction_wsi" not in off.objective_weights_effective()
    with pytest.raises(RuntimeError):
        off.load_state_dict(on.state_dict(), strict=True)
    model().load_state_dict(on.state_dict(), strict=True)


@pytest.mark.parametrize("budget,expected", [
    ("additive", [0.05, 0.05, 0.05, 0.15]),
    ("fixed_total", [1/30, 1/30, 1/30, 0.10]),
])
def test_effective_coefficients_and_audit(budget, expected):
    net = model(budget=budget)
    coeffs = net.effective_recon_coefficients()
    assert [coeffs[k] for k in ("self", "cross", "wsi", "total")] == pytest.approx(expected)
    assert coeffs["wsi_requested"] == 0.05
    assert coeffs["wsi_budget"] == budget
    assert net.objective_weights_effective()["reconstruction_wsi"] == coeffs["wsi"]


@pytest.mark.parametrize("weight,budget,extra", [
    (-0.1, "additive", {}), (float("nan"), "additive", {}),
    (float("inf"), "additive", {}), (0.1, "invalid", {}),
    (0.1, "additive", {"dct_v313_wsi_reconstruction_chunk_size": 0}),
    (0.1, "additive", {"dct_v313_recon_weighting": "legacy"}),
    (0.1, "fixed_total", {"dct_v313_disable_self_reconstruction": True, "dct_v313_disable_cross_reconstruction": True}),
])
def test_invalid_settings_fail_early(weight, budget, extra):
    with pytest.raises(ValueError):
        model(weight, budget, **extra)


def test_fixed_total_uses_active_budget():
    net = model(budget="fixed_total", dct_v313_disable_self_reconstruction=True)
    c = net.effective_recon_coefficients()
    assert c["self"] == 0
    assert c["cross"] == pytest.approx(0.025)
    assert c["wsi"] == pytest.approx(0.025)
    assert c["total"] == pytest.approx(0.05)


def test_decoder_slot_gradients_detached_targets_and_chunk_equivalence():
    torch.manual_seed(7)
    decoder = WSISlotReconstructionDecoder(16, 2, 2)
    whole = copy.deepcopy(decoder)
    whole.chunk_size = 100
    memory = torch.randn(3, 3, 16, requires_grad=True)
    target = torch.randn(3, 7, 16, requires_grad=True)
    valid = torch.tensor([[1]*7, [1, 1, 0, 0, 0, 0, 0], [0]*7], dtype=torch.bool)
    loss = decoder.reconstruction_loss(memory, target, valid)
    reference_memory = memory.detach().clone().requires_grad_()
    reference = whole.reconstruction_loss(reference_memory, target, valid)
    assert torch.allclose(loss, reference, atol=1e-6)
    loss.backward()
    reference.backward()
    assert torch.allclose(memory.grad, reference_memory.grad, atol=1e-6)
    assert memory.grad[:2].abs().sum() > 0
    assert memory.grad[2].abs().sum() == 0
    assert target.grad is None
    assert decoder.query_projection.weight.grad is None
    assert not decoder.query_projection.weight.requires_grad
    assert decoder.cross_attention.in_proj_weight.grad.abs().sum() > 0
    changed = target.detach().clone()
    changed[~valid] = float("nan")
    assert torch.allclose(decoder.reconstruction_loss(memory.detach(), changed, valid), loss.detach())
    # Explicit equal patient weighting, independent of each patient's patch count.
    per_patient = [
        decoder.reconstruction_loss(memory[i:i+1].detach(), target[i:i+1], valid[i:i+1])
        for i in (0, 1)
    ]
    assert torch.allclose(loss.detach(), torch.stack(per_patient).mean(), atol=1e-6)


def test_all_missing_returns_graph_connected_zero():
    decoder = WSISlotReconstructionDecoder(8, 2)
    memory = torch.randn(2, 3, 8, requires_grad=True)
    target = torch.randn(2, 4, 8, requires_grad=True)
    loss = decoder.reconstruction_loss(memory, target, torch.zeros(2, 4, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(memory.grad, torch.zeros_like(memory))
    assert target.grad is None


def test_model_excludes_zero_padding_explicit_mask_and_missing_patient():
    net = model()
    raw = torch.randn(3, 5, 16)
    raw[0, 3:] = 0
    projected = torch.randn_like(raw, requires_grad=True)
    slots = torch.randn(3, 3, 16, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0, 1] = False
    kwargs = dict(x_wsi=raw, wsi_patch_mask=mask,
                  wsi_available=torch.tensor([True, True, False]),
                  wsi_missing=torch.tensor([False, True, False]))
    loss = net.wsi_reconstruction_loss(slots, projected, kwargs)
    valid = torch.zeros(3, 5, dtype=torch.bool)
    valid[0, [0, 2]] = True
    expected = net.wsi_reconstruction_decoder.reconstruction_loss(slots, projected, valid)
    assert torch.equal(loss, expected)
    loss.backward()
    assert projected.grad is None
    assert slots.grad[0].abs().sum() > 0
    assert slots.grad[1:].abs().sum() == 0
    with pytest.raises(ValueError, match="wsi_patch_mask"):
        net.wsi_reconstruction_loss(slots, projected, dict(x_wsi=raw, wsi_patch_mask=torch.ones(3)))


@pytest.mark.parametrize("budget", ["additive", "fixed_total"])
@pytest.mark.parametrize("epoch", [0, 4, 7])
def test_training_applies_coefficients_once_and_uses_existing_ramp(budget, epoch):
    net = model(budget=budget)
    net.train()
    net.args.cur_epoch = epoch
    net.configure_train_reference(*train_reference())
    _, aux = net(**batch())
    log = net.last_training_losses
    reconstruction = (
        net.dct_v313_reconstruction_self_coef * log["v313_reconstruction_self"]
        + net.dct_v313_reconstruction_cross_coef * log["v313_reconstruction_cross"]
        + net.dct_v313_reconstruction_wsi_coef * log["v313_reconstruction_wsi"]
    )
    expected = (
        net.dct_lambda_ipcw_rank * log["ipcw_rank"]
        + net.dct_v313_lambda_slot_nll_effective * log["v311_per_slot_nll"]
        + net.dct_v313_lambda_slot_diversity_effective * log["v311_slot_diversity"]
        + net._reconstruction_ramp(epoch) * reconstruction
    )
    assert torch.allclose(log["v313_reconstruction_total"], reconstruction)
    assert torch.allclose(aux, expected)
    assert log["v313_reconstruction_wsi_coef"].item() == pytest.approx(net.dct_v313_reconstruction_wsi_coef)
    aux.backward()
    if epoch > 2:
        assert net.wsi_reconstruction_decoder.cross_attention.in_proj_weight.grad.abs().sum() > 0


def test_enabled_branch_preserves_prediction_and_never_runs_at_eval():
    off = model(0)
    on = model()
    on.load_state_dict(off.state_dict(), strict=False)
    payload = batch()
    for net in (off, on):
        net.train()
        net.configure_train_reference(*train_reference())
    torch.manual_seed(55)
    logits_off, _ = off(**payload)
    torch.manual_seed(55)
    logits_on, _ = on(**payload)
    assert torch.equal(logits_off, logits_on)
    on.eval()
    off.eval()
    hook = on.wsi_reconstruction_decoder.register_forward_hook(
        lambda *args: pytest.fail("WSI reconstruction ran during inference")
    )
    with torch.no_grad():
        assert torch.equal(off(**payload)[0], on(**payload)[0])
    hook.remove()


def load_launcher():
    path = Path("实验档案/实验代码/E046_WSI自重建消融/run_wsi_reconstruction_ablation.py")
    spec = importlib.util.spec_from_file_location("e046_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_only_no_side_effects_and_correct_fold_protocol(tmp_path):
    launch = load_launcher()
    root = tmp_path / "fresh"
    args = launch.parser().parse_args(["--results-root", str(root), "--cancers", "all", "--seeds", "3", "13"])
    plan = launch.build_plan(args)
    assert len(plan["tasks"]) == 3 * 10 * 5 * 2
    assert len({t["output"] for t in plan["tasks"]}) == len(plan["tasks"])
    assert not root.exists()
    assert all(t["overrides"]["evaluation_protocol"] == "outer_test" for t in plan["tasks"])
    for t in plan["tasks"]:
        overrides = t["overrides"]
        assert overrides["k_start"] == t["fold"]
        assert overrides["k_end"] == t["fold"] + 1
        assert overrides["seed"] == t["seed"]
        assert overrides["gpu"] == 0
        assert overrides["dct_v313_recon_weighting"] == "per_branch"
        assert overrides["split_seed"] == 3
        assert f"results_dir={t['output']}" in t["command"]


def test_execution_preflight_refuses_overwrite_and_missing_splits(tmp_path, monkeypatch):
    launch = load_launcher()
    root = tmp_path / "fresh"
    args = launch.parser().parse_args(["--results-root", str(root), "--data-path", str(tmp_path / "absent")])
    plan = launch.build_plan(args)
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **kw: plan["source_revision"])
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("training must not start"))
    with pytest.raises(ValueError, match="split CSVs"):
        launch.execute_plan(plan)
    assert not root.exists()
    root.mkdir()
    (root / "prior.log").write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        launch.execute_plan(plan)
    assert (root / "prior.log").read_text() == "preserve"


def test_wsi_only_objective_reaches_wsi_encoder_and_slots():
    net = model()
    net.train()
    payload = batch()
    projected = net.wsi_mlp(payload["x_wsi"])
    omics = net._encode_omics(payload)
    slots, _, _, _ = net._encode_transport_slots(projected, omics, payload)
    slots.retain_grad()
    loss = net.wsi_reconstruction_loss(slots, projected, payload)
    loss.backward()
    assert slots.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.wsi_mlp.parameters())


def test_new_flags_reach_cli_parser_and_model_factory():
    from survot_rank.training.extended_args import build_base_parser
    from survot_rank.training.model_factory import get_model
    args = build_base_parser().parse_args([
        "--dct_v313_lambda_wsi_reconstruction", "0.05",
        "--dct_v313_wsi_reconstruction_budget", "fixed_total",
        "--dct_v313_wsi_reconstruction_chunk_size", "128",
    ])
    assert args.dct_v313_lambda_wsi_reconstruction == 0.05
    assert args.dct_v313_wsi_reconstruction_budget == "fixed_total"
    assert args.dct_v313_wsi_reconstruction_chunk_size == 128
    net = get_model("dct_v313", make_args(
        dct_v313_lambda_wsi_reconstruction=args.dct_v313_lambda_wsi_reconstruction,
        dct_v313_wsi_reconstruction_budget=args.dct_v313_wsi_reconstruction_budget,
        dct_v313_wsi_reconstruction_chunk_size=args.dct_v313_wsi_reconstruction_chunk_size,
    ))
    assert net.wsi_reconstruction_decoder is not None
    assert net.wsi_reconstruction_decoder.chunk_size == 128


def test_execute_uses_config_snapshot_and_gpu_mapping(tmp_path, monkeypatch):
    launch = load_launcher()
    data = tmp_path / "data"
    features = tmp_path / "features"
    features.mkdir()
    split = data / "splits/5fold_uni2h/blca/fold_0.csv"
    split.parent.mkdir(parents=True)
    split.write_text("train,val\np1,p2\n")
    root = tmp_path / "fresh"
    args = launch.parser().parse_args([
        "--results-root", str(root), "--cancers", "blca", "--folds", "0",
        "--arms", "wsi_fixed_total", "--protocol", "legacy_val", "--gpu", "2",
        "--data-path", str(data), "--data-root-dir", str(features),
    ])
    plan = launch.build_plan(args)
    calls = []
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **kw: plan["source_revision"])
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    launch.execute_plan(plan)
    assert len(calls) == 1
    cmd, kw = calls[0]
    assert cmd[cmd.index("--config") + 1] == str(root / "base_config.yaml")
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert "gpu=0" in cmd
    assert (root / "plan.json").is_file()
    assert launch.sha256(root / "base_config.yaml") == plan["config_sha256"]


def test_source_drift_is_rejected_before_training(tmp_path):
    launch = load_launcher()
    args = launch.parser().parse_args(["--results-root", str(tmp_path / "fresh")])
    plan = launch.build_plan(args)
    first = next(iter(plan["source_sha256"]))
    plan["source_sha256"][first] = "incorrect"
    with pytest.raises(ValueError, match="Source changed"):
        launch.execute_plan(plan)
    assert not Path(plan["results_root"]).exists()
