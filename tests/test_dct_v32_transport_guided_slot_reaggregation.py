from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import run_dct_v32_experiments as queue
from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.research.components.transport_guided_slot_reaggregation import TransportGuidedSlotReaggregation
from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG, PRIMARY_METHOD
from survot_rank.research.methods.dct_v32_transport_guided_slot_reaggregation import DCTV32TransportGuidedSlotReaggregation
from survot_rank.training.extended_args import build_base_parser
from survot_rank.training.model_factory import get_model
from survot_rank.training.train_runner import compose_batch_objective, init_loss_function


@pytest.fixture(autouse=True)
def reproducibility():
    torch.manual_seed(123)


def inputs(batch=2):
    return (torch.randn(batch, 3, 16), torch.randn(batch, 4, 16),
            torch.randn(batch, 7, 16), torch.randn(batch, 5, 16))


def model_args(**overrides):
    values = dict(bag_loss="nll_surv", omic_sizes=None, n_classes=4,
                  encoding_dim=16, wsi_projection_dim=16, rna_format="RNASeq",
                  slot_num_wsi=3, slot_num_omics=4, slot_iters=2, alpha_surv=0.15,
                  otehv2_eps=0.1, otehv2_iter=20, otehv2_heads=2,
                  otehv2_layers=1, otehv2_dropout=0.0, dct_num_stages=4,
                  dct_coupling_projection_iters=30, dct_coupling_projection_tol=1e-4,
                  cur_epoch=12, dct_v32_feedback_iters=30)
    values.update(overrides)
    return SimpleNamespace(**values)


def payload(batch=3):
    return dict(x_wsi=torch.randn(batch, 7, 16), x_omics=torch.randn(batch, 5, 20),
                y=torch.arange(batch) % 4, event_time=torch.arange(1, batch + 1).float(),
                c=torch.arange(batch).float() % 2)


@pytest.mark.parametrize("mode", TransportGuidedSlotReaggregation.MODES)
def test_component_shapes_determinism_and_both_modalities(mode):
    module = TransportGuidedSlotReaggregation(16, mode=mode, sinkhorn_iters=100).eval()
    data = inputs()
    a, b = module(*data)
    a2, b2 = module(*data)
    assert a.shape == data[0].shape and b.shape == data[1].shape
    torch.testing.assert_close(a, a2, rtol=0, atol=0)
    torch.testing.assert_close(b, b2, rtol=0, atol=0)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if mode == "none":
        assert a is data[0] and b is data[1]
    if mode == "ot":
        assert module.last_plan.shape == (2, 3, 4)
        assert module.last_diagnostics["feedback_marginal_error"] < 1e-4


def test_ot_context_changes_original_token_assignments():
    module = TransportGuidedSlotReaggregation(16, strength=0.8)
    module.capture_attention = True
    data = inputs()
    module(*data)
    original = module.last_attention["wsi_assignment"].clone()
    module(data[0], torch.randn_like(data[1]), data[2], data[3])
    assert not torch.allclose(original, module.last_attention["wsi_assignment"])
    assert original.shape[-1] == data[2].shape[1]  # raw token count, NOT slot count
    torch.testing.assert_close(original.sum(1), torch.ones_like(original.sum(1)))
    for key in ("wsi_pooling", "omic_pooling"):
        weights = module.last_attention[key]
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        assert not weights.requires_grad


def test_zero_feedback_equals_self_update_with_identical_weights():
    ot = TransportGuidedSlotReaggregation(16, strength=0)
    control = TransportGuidedSlotReaggregation(16, mode="self")
    control.load_state_dict(ot.state_dict())
    data = inputs()
    for a, b in zip(ot(*data), control(*data)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_self_update_is_independent_of_other_modality():
    module = TransportGuidedSlotReaggregation(16, mode="self")
    data = inputs()
    first, _ = module(*data)
    changed, _ = module(data[0], torch.randn_like(data[1]), data[2], torch.randn_like(data[3]))
    torch.testing.assert_close(first, changed, rtol=0, atol=0)


def test_raw_tokens_and_cross_modal_context_have_gradients():
    module = TransportGuidedSlotReaggregation(16)
    data = tuple(x.requires_grad_() for x in inputs())
    wsi, _ = module(*data)
    wsi.square().mean().backward()
    for idx in (0, 1, 2):
        assert torch.isfinite(data[idx].grad).all()
        assert data[idx].grad.abs().sum() > 0
    assert module.wsi_reader.context_query.weight.grad.abs().sum() > 0


def test_token_permutation_invariance_and_masked_padding():
    module = TransportGuidedSlotReaggregation(16).eval()
    data = inputs()
    original = module(*data)
    permuted = module(data[0], data[1], data[2].flip(1), data[3].flip(1))
    for a, b in zip(original, permuted):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    padded_wsi = torch.cat([data[2], torch.full((2, 2, 16), float("nan"))], 1)
    mask = torch.arange(9).expand(2, -1) < 7
    padded = module(data[0], data[1], padded_wsi, data[3], wsi_mask=mask)
    for a, b in zip(original, padded):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="valid token"):
        module(*data, wsi_mask=torch.zeros((2, 7), dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        module(data[0], data[1], padded_wsi, data[3])


@pytest.mark.parametrize("kwargs", [{"mode": "bad"}, {"rounds": 0}, {"rounds": True},
                                    {"epsilon": 0}, {"epsilon": float("nan")},
                                    {"strength": 2}, {"sinkhorn_iters": 1001}])
def test_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        TransportGuidedSlotReaggregation(16, **kwargs)


@pytest.mark.parametrize("mode", TransportGuidedSlotReaggregation.MODES)
def test_model_nll_backward_no_anchors_and_eval_roundtrip(mode, tmp_path):
    args = model_args(dct_v32_feedback=mode, dct_lambda_ipcw_rank=99,
                      dct_v38_lambda_direction=99, dct_random_anchors=True)
    model = get_model("dct_v32", args, omic_input_dim=20)
    model.train()
    data = payload()
    logits, aux = model(**data)
    assert logits.shape == (3, 4) and aux == 0
    assert model.objective_weights() == {"nll": 1.0, "ipcw_rank": 0.0, "direction": 0.0}
    assert not model.risk_anchor_seen.any()
    raw = init_loss_function(args)(logits, data["y"], data["event_time"], data["c"])
    loss = compose_batch_objective(raw, aux, 3)
    torch.testing.assert_close(loss, raw / 3)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if mode in {"ot", "attention"}:
        assert model.reaggregation.wsi_reader.context_query.weight.grad.abs().sum() > 0
        assert model.reaggregation.omic_reader.context_query.weight.grad.abs().sum() > 0
    torch.optim.Adam(model.parameters(), lr=1e-4).step()
    model.eval()
    with torch.no_grad():
        first, _ = model(**data)
        # No labels are needed and labels cannot alter inference.
        second, _ = model(x_wsi=data["x_wsi"], x_omics=data["x_omics"])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert "low_risk_counterfactual" not in model.last_explanations
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = DCTV32TransportGuidedSlotReaggregation(model_args(dct_v32_feedback=mode), omic_input_dim=20).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        third, _ = restored(**data)
    torch.testing.assert_close(first, third, rtol=0, atol=0)
    assert model(**payload(batch=1))[0].shape == (1, 4)


def test_pathway_inputs_use_original_pathway_tokens():
    model = DCTV32TransportGuidedSlotReaggregation(
        model_args(rna_format="Pathways", omic_sizes=[3, 5, 7]), omic_input_dim=[3, 5, 7]).eval()
    model.reaggregation.capture_attention = True
    logits, aux = model(x_wsi=torch.randn(2, 9, 16), x_omic1=torch.randn(2, 3),
                        x_omic2=torch.randn(2, 5), x_omic3=torch.randn(2, 7))
    assert logits.shape == (2, 4) and aux == 0
    assert model.reaggregation.last_attention["omic_assignment"].shape == (2, 4, 3)


def test_catalog_config_and_twenty_matched_jobs():
    assert PRIMARY_METHOD == "dct_v310_directional_regularized_transport"
    assert METHOD_CATALOG[METHOD_ALIASES["dct_v32"]].status == "candidate"
    args = queue.build_parser().parse_args(["plan", "--python", "python"])
    jobs = queue.build_jobs(args)
    assert len(jobs) == 20
    assert len({job.result_dir for job in jobs}) == 20
    for job in jobs:
        overrides = [job.command[i + 1] for i, x in enumerate(job.command) if x == "--set"]
        config = apply_overrides(load_config(job.config), overrides)
        parsed = build_base_parser().parse_args(config_to_argv(config))
        assert parsed.survot_method == queue.METHOD
        assert parsed.dct_v32_feedback == queue.VARIANTS[job.variant]
        assert parsed.dct_lambda_ipcw_rank == parsed.dct_v38_lambda_direction == 0
        assert parsed.max_epochs == 30 and parsed.seed == 3
        assert parsed.k_start == job.fold and parsed.k_end == job.fold + 1
    args.seed = 11
    assert jobs[0].result_dir != queue.build_jobs(args)[0].result_dir
    args.seed = 3
    args.rounds = 2
    assert jobs[0].result_dir != queue.build_jobs(args)[0].result_dir
    assert len(queue.build_jobs(args, smoke=True)) == 4


def test_old_checkpoint_cannot_silently_become_v32():
    from survot_rank.research.methods.dct_v310_directional_regularized_transport import DCTV310DirectionalRegularizedTransport
    old = DCTV310DirectionalRegularizedTransport(model_args(), omic_input_dim=20)
    new = DCTV32TransportGuidedSlotReaggregation(model_args(), omic_input_dim=20)
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(old.state_dict())


def test_checkpoint_rejects_a_different_feedback_variant():
    ot = DCTV32TransportGuidedSlotReaggregation(model_args(), omic_input_dim=20)
    baseline = DCTV32TransportGuidedSlotReaggregation(model_args(dct_v32_feedback="none"), omic_input_dim=20)
    with pytest.raises(RuntimeError, match="configuration"):
        baseline.load_state_dict(ot.state_dict())


def test_baseline_preserves_old_factual_backbone():
    from survot_rank.research.methods.dct_v310_directional_regularized_transport import DCTV310DirectionalRegularizedTransport
    torch.manual_seed(7)
    old = DCTV310DirectionalRegularizedTransport(model_args(), omic_input_dim=20).eval()
    torch.manual_seed(7)
    baseline = DCTV32TransportGuidedSlotReaggregation(model_args(dct_v32_feedback="none"), omic_input_dim=20).eval()
    # All common tensors start identically. The candidate adds new parameters
    # AFTER constructing the old backbone, including for the no-feedback arm.
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
    data = payload()
    with torch.no_grad():
        # Reuse just the old factual path; old forward also needs references
        # for its post-hoc audit, irrelevant to this equality check.
        xw = old.wsi_mlp(data["x_wsi"])
        xo = old._encode_omics(data)
        sw, so, _, _ = old._encode_transport_slots(xw, xo, data)
        cost, rows, cols, _ = old._cost_tensor(sw, so)
        plans, _ = old._plans_from_cost_tensor(cost, rows, cols, old.args.cur_epoch)
        expected, _ = old._encode_logits_from_plans(sw, so, plans)
        actual, _ = baseline(**data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multiple_rounds_fresh_batch_and_clear_capture():
    module = TransportGuidedSlotReaggregation(16, rounds=2)
    module.capture_attention = True
    module(*inputs(batch=3))
    assert module.last_plan.shape[0] == 3
    module.capture_attention = False
    module(*inputs(batch=1))
    assert module.last_plan.shape[0] == 1
    assert module.last_attention is None


def test_component_cpu_bfloat16_autocast():
    module = TransportGuidedSlotReaggregation(16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(*inputs())
        loss = sum(x.float().square().mean() for x in out)
    loss.backward()
    assert all(torch.isfinite(x).all() for x in out)
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_launcher_runs_as_a_file_outside_repo(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(queue.REPO_ROOT / "scripts/run_dct_v32_experiments.py"),
                             "plan", "--variants", "ot_feedback", "--folds", "0"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CANDIDATE v3.2 TGSR" in result.stdout
    assert "dct_v32_feedback=ot" in result.stdout


def test_shared_collation_forward_and_loss_contract():
    from survot_rank.training.train_runner import _collate_pathways, _process_data_and_forward, _calculate_risk
    args = model_args(rna_format="Pathways", omic_sizes=[3, 5, 7], omic_missing=False)
    model = DCTV32TransportGuidedSlotReaggregation(args, omic_input_dim=[3, 5, 7])
    samples = [(torch.randn(9, 16), [torch.randn(n) for n in [3, 5, 7]],
                torch.tensor(i % 4), float(i + 1), float(i % 2)) for i in range(3)]
    data = _collate_pathways(samples)
    model.configure_train_reference(data[3], data[4])
    assert model.dct_stage_edges.numel() == model.dct_censor_times.numel() == 0
    out, y, times, censor = _process_data_and_forward(args, model, data, torch.device("cpu"))
    loss = compose_batch_objective(init_loss_function(args)(out[0], y, times, censor), out[1], len(y))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.numel() == 1 for value in model.last_training_losses.values())
    model.eval()
    with torch.no_grad():
        out, _, _, _ = _process_data_and_forward(args, model, data, torch.device("cpu"), test=True)
    risk, _ = _calculate_risk(out[0])
    torch.testing.assert_close(torch.from_numpy(risk), model.last_explanations["factual_risk"])
    with pytest.raises(ValueError, match="requires both"):
        model(omic_missing=True)
