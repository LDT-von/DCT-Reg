from __future__ import annotations

import math
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.research.methods.legacy.experimental.dct_v311_slot_interpretable.model import (
    DCTV311SlotInterpretable,
)
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import (
    DCTV313TransportReconstruction,
)
from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction.model import (
    DCTV314MaskedTransportReconstruction,
)
from survot_rank.training.model_factory import get_model
from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction.losses import discrete_nll, StableNLLSurvLoss
from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction.backbone import PrototypeSlotAttention


def make_args(**overrides):
    values = dict(
        bag_loss="nll_surv",
        omic_sizes=[3, 4, 5, 6, 7],
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="Pathways",
        alpha_surv=0.15,
        slot_num_wsi=3,
        slot_num_omics=3,
        slot_iters=2,
        otehv2_eps=0.05,
        otehv2_iter=15,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.99,
        dct_ipcw_rank_margin=0.99,
        dct_ipcw_rank_temperature=0.99,
        dct_ipcw_max_weight=99.0,
        dct_ipcw_rank_memory_size=1,
        dct_lambda_etar=0.99,
        dct_lambda_listwise=0.99,
        dct_anchor_momentum=0.0,
        dct_evidence_cost_weight=0.99,
        dct_evidence_mass_floor=0.99,
        dct_evidence_marginal_strength=0.0,
        dct_geometry_reliability_strength=0.99,
        dct_geometry_reliability_temperature=0.25,
        dct_coupling_projection_iters=20,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=0.25,
        dct_v38_lambda_direction=0.99,
        dct_v38_lambda_dose=0.99,
        dct_v38_lambda_reconfiguration=0.99,
        dct_v38_direction_margin=0.99,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.99,
        dct_v38_alpha_mid=0.25,
        dct_v38_alpha_full=0.75,
        dct_v38_warmup_epochs=5,
        dct_v38_ramp_epochs=10,
        dct_v38_dose_every=1,
        dct_v382_lambda_mgptr=0.99,
        dct_v382_adaptive_aux_weights=True,
        dct_fixed_coupling=True,
        dct_random_anchors=True,
        dct_perm_labels_seed=7,
        dct_stage_jitter_fraction=0.3,
        dct_freeze_source_prototype="",
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.2,
        spt_lambda_ot=0.0,
        spt_lambda_rank=0.0,
        spt_lambda_stage=0.0,
        spt_stage_margin=0.25,
        rg_eps_start=0.1,
        rg_eps_anneal=12,
        dct_slot_init_mode="gaussian",
        dct_slot_eval_seed=91,
        cur_epoch=7,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def train_reference():
    times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    censorship = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return times, censorship


def batch(seed=13):
    generator = torch.Generator().manual_seed(seed)
    payload = {
        "x_wsi": torch.randn(8, 6, 16, generator=generator),
        "y": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "event_time": train_reference()[0],
        "c": train_reference()[1],
    }
    for index, width in enumerate([3, 4, 5, 6, 7], start=1):
        payload[f"x_omic{index}"] = torch.randn(8, width, generator=generator)
    return payload


def test_v314_is_version_local_and_registered():
    assert DCTV311SlotInterpretable not in DCTV314MaskedTransportReconstruction.mro()
    assert DCTV313TransportReconstruction not in DCTV314MaskedTransportReconstruction.mro()
    key = "dct_v314_masked_transport_reconstruction"
    assert METHOD_ALIASES["dct_v314"] == key
    assert METHOD_CATALOG[key].status == "candidate"
    assert type(get_model("dct_v314", make_args())).__name__ == (
        "DCTV314MaskedTransportReconstruction"
    )

    config = yaml.safe_load(
        Path("configs/dct_v314_blca_uni.yaml").read_text(encoding="utf-8-sig")
    )
    assert config["train"]["survot_method"] == key
    assert config["data"]["rna_format"] == "Pathways"

    from survot_rank.config import load_config, config_to_argv, apply_overrides
    from survot_rank.training.extended_args import build_base_parser
    from survot_rank.training.train_runner import init_loss_function
    args = build_base_parser().parse_args(config_to_argv(apply_overrides(
        load_config('configs/dct_v314_blca_uni.yaml'),
        ['dct_v314_lambda_mtr=0.05', 'dct_v314_reconstruction_mode=hybrid'])))
    assert args.dct_v314_lambda_mtr == .05 and args.dct_v314_reconstruction_mode == 'hybrid'
    assert isinstance(init_loss_function(args), StableNLLSurvLoss)


def test_v314_default_objective_and_schedule():
    model = DCTV314MaskedTransportReconstruction(make_args())
    assert model.objective_weights() == {
        "nll": 1.0,
        "ipcw_rank": 0.10,
        "per_slot_nll": 0.05,
        "slot_diversity": 0.02,
        "masked_transport_reconstruction": 0.10,
    }
    assert model.dct_v38_lambda_direction == 0.0
    assert model.dct_v314_mask_ratio == 0.20
    assert model._mtr_ramp(0) == 0.0
    assert model._mtr_ramp(2) == 0.0
    assert model._mtr_ramp(4) == pytest.approx(0.4)
    assert model._mtr_ramp(7) == 1.0
    assert model.args.dct_ipcw_rank_margin == model.dct_ipcw_rank_margin == .02
    assert model.args.dct_fixed_coupling is False


@pytest.mark.parametrize("event", [True, False], ids=["event", "censored"])
def test_v314_per_slot_nll_uses_cumulative_survival(event):
    hazard = torch.full((1, 1, 4), 0.5, dtype=torch.float64, requires_grad=True)
    labels = F.one_hot(torch.tensor([3]), num_classes=4).to(torch.float64)
    event_mask = torch.tensor([[float(event)]], dtype=torch.float64)
    loss = DCTV314MaskedTransportReconstruction._nll_surv_per_slot(
        None,
        hazard,
        labels,
        event_mask,
        1.0 - event_mask,
        torch.ones(1, 1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        loss,
        torch.tensor([[4.0 * math.log(2.0)]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    gradient = torch.autograd.grad(loss.sum(), hazard)[0]
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient[..., :3] > 0)


def test_pathway_mask_has_exact_count_respects_availability_and_blocks_input_leakage():
    torch.manual_seed(3)
    model = DCTV314MaskedTransportReconstruction(make_args())
    target = torch.randn(3, 5, 16)
    available = torch.tensor([True, False, True])
    mask = model._sample_pathway_mask(target, available)
    assert mask.sum(dim=1).tolist() == [1, 0, 1]

    changed = target.clone()
    changed[mask] += 1000.0
    masked_original = model._apply_pathway_mask(target, mask)
    masked_changed = model._apply_pathway_mask(changed, mask)
    torch.testing.assert_close(masked_original, masked_changed)


def test_masked_transport_reconstruction_reaches_all_intended_gradients():
    torch.manual_seed(5)
    model = DCTV314MaskedTransportReconstruction(make_args())
    model.train()
    slots_wsi = torch.randn(3, 3, 16, requires_grad=True)
    full_slots_omic = torch.randn(3, 3, 16)
    target = torch.randn(3, 5, 16, requires_grad=True)
    full_costs, rows, cols, _ = model._cost_tensor(slots_wsi, full_slots_omic)
    full_plans, _ = model._plans_from_cost_tensor(full_costs, rows, cols, epoch=7)
    _, factual_gate = model._encode_logits_from_plans(
        slots_wsi, full_slots_omic, full_plans
    )
    explicit_mask = torch.tensor(
        [[True, False, False, False, False]] * 3, dtype=torch.bool
    )

    self_loss, cross_loss, total, transported, mask, _ = (
        model.masked_transport_reconstruction_losses(
            x_omics=target,
            slots_wsi=slots_wsi,
            factual_gate=factual_gate,
            available=torch.ones(3, dtype=torch.bool),
            epoch=7,
            pathway_mask=explicit_mask,
        )
    )
    total.backward()

    assert transported.shape == full_slots_omic.shape
    assert torch.equal(mask, explicit_mask)
    assert self_loss > 0 and cross_loss > 0
    assert slots_wsi.grad is not None and slots_wsi.grad.abs().sum() > 0
    assert target.grad is not None and target.grad.abs().sum() > 0
    assert model.pathway_mask_token.grad is not None
    assert model.pathway_mask_token.grad.abs().sum() > 0
    decoder_grad = model.pathway_reconstruction_decoder.pathway_queries.grad
    assert decoder_grad is not None and decoder_grad.abs().sum() > 0
    omic_gradients = [
        parameter.grad
        for parameter in model.slot_attention_omic.parameters()
        if parameter.grad is not None
    ]
    assert omic_gradients
    assert sum(gradient.abs().sum() for gradient in omic_gradients) > 0
    sinkhorn_grad = model.stage_pair_cost[-1].weight.grad
    assert sinkhorn_grad is not None and sinkhorn_grad.abs().sum() > 0


def test_v314_training_forward_is_exact_declared_auxiliary_sum():
    torch.manual_seed(17)
    model = DCTV314MaskedTransportReconstruction(make_args(cur_epoch=7))
    model.configure_train_reference(*train_reference())
    model.train()
    logits, auxiliary = model(**batch())
    diagnostics = model.last_training_losses
    expected = (
        0.10 * diagnostics["ipcw_rank"]
        + 0.05 * diagnostics["v314_per_slot_nll"]
        + 0.02 * diagnostics["v314_slot_diversity"]
        + 0.10 * diagnostics["v314_mtr_total"]
    )
    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all() and torch.isfinite(auxiliary)
    torch.testing.assert_close(auxiliary, expected)
    assert diagnostics["v314_mtr_self"] > 0
    assert diagnostics["v314_mtr_cross"] > 0
    assert diagnostics["v314_masked_pathway_fraction"] == pytest.approx(0.2)
    auxiliary.backward()
    assert model.pathway_mask_token.grad is not None
    assert model.pathway_mask_token.grad.abs().sum() > 0


def test_v314_unavailable_omics_zeroes_only_mtr_terms():
    torch.manual_seed(19)
    model = DCTV314MaskedTransportReconstruction(make_args(cur_epoch=7))
    model.configure_train_reference(*train_reference())
    model.train()
    payload = batch(seed=21)
    payload["omic_available"] = torch.zeros(8, dtype=torch.bool)
    _, auxiliary = model(**payload)
    diagnostics = model.last_training_losses
    assert torch.isfinite(auxiliary)
    assert diagnostics["v314_mtr_total"] == 0.0
    assert diagnostics["v314_masked_pathway_count"] == 0.0
    assert diagnostics["v314_omics_available_fraction"] == 0.0


def test_v314_training_without_survival_labels_keeps_mtr_finite():
    torch.manual_seed(22)
    model = DCTV314MaskedTransportReconstruction(make_args(cur_epoch=7))
    model.train()
    payload = batch(seed=22)
    payload.pop("y")
    payload.pop("event_time")
    payload.pop("c")
    logits, auxiliary = model(**payload)
    assert logits.shape == (8, 4)
    assert torch.isfinite(auxiliary)
    assert model.last_training_losses["v314_per_slot_nll"] == 0.0
    # Representation diversity is unsupervised and remains active without labels.
    assert torch.isfinite(model.last_training_losses["v314_slot_diversity"])
    assert model.last_training_losses["v314_mtr_total"] > 0


def test_v314_eval_preserves_factual_audit_and_per_slot_outputs():
    torch.manual_seed(23)
    model = DCTV314MaskedTransportReconstruction(make_args())
    model.configure_train_reference(*train_reference())
    model.eval()
    with torch.no_grad():
        logits, auxiliary = model(**batch(seed=29))
    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert auxiliary == 0.0
    assert model.last_explanations["event_gate"].shape == (8, 4)
    assert model.last_explanations["per_slot_hazard_wsi"].shape == (8, 3, 4)
    assert model.last_explanations["per_slot_hazard_omic"].shape == (8, 3, 4)


def test_v314_rejects_non_pathway_omics():
    with pytest.raises(ValueError, match="requires rna_format='Pathways'"):
        DCTV314MaskedTransportReconstruction(
            make_args(rna_format="RNASeq"), omic_input_dim=20
        )


def test_no_previous_method_runtime_dependency():
    program = '''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("survot_rank.research.methods.") and not name.startswith("survot_rank.research.methods.legacy"):
        raise AssertionError("older method imported: " + name)
    if "dct_v311" in name or "dct_v313" in name:
        raise AssertionError("older experiment imported: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction import DCTV314MaskedTransportReconstruction
assert all("v310" not in c.__name__ for c in DCTV314MaskedTransportReconstruction.mro())
'''
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('c', [0., 1.])
@pytest.mark.parametrize('time_bin', range(4))
def test_nll_matches_hand_calculation_all_bins_and_censoring(c, time_bin):
    logits = torch.zeros(2, 3, 4, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([time_bin, time_bin])
    censorship = torch.full((2,), c, dtype=torch.float64)
    result = discrete_nll(logits, y, censorship, alpha=.15)
    expected = (time_bin+1)*math.log(2)*(1-.15*c)
    torch.testing.assert_close(result, torch.full((2, 3), expected, dtype=torch.float64))
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_extreme_logits_have_finite_corrective_gradients():
    logits = torch.tensor([[100., -100., 100., -100.]], requires_grad=True)
    result = discrete_nll(logits, torch.tensor([3]), torch.tensor([0.]))
    result.sum().backward()
    assert torch.isfinite(result).all() and result.item() > 200
    assert logits.grad[0, 0] > .9 and logits.grad[0, 3] < -.9


def test_masked_branch_has_no_information_or_gradient_from_hidden_tokens_and_full_gate():
    torch.manual_seed(71)
    model = DCTV314MaskedTransportReconstruction(make_args()).eval()
    x = torch.randn(3, 5, 16, requires_grad=True)
    w = torch.randn(3, 3, 16, requires_grad=True)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 2] = True
    gate = torch.randn(3, 4, requires_grad=True)
    common = dict(slots_wsi=w, available=torch.ones(3, dtype=torch.bool), epoch=7, pathway_mask=mask)
    first = model.masked_transport_reconstruction_losses(x_omics=x, factual_gate=gate, **common)
    changed = x.detach().clone()
    changed[mask] = torch.randn_like(changed[mask])*20
    second = model.masked_transport_reconstruction_losses(x_omics=changed, factual_gate=gate*100, **common)
    # Transported memory AND masked omics slots cannot depend on hidden targets.
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
    torch.testing.assert_close(first[5], second[5], rtol=0, atol=0)
    first[2].backward()
    assert gate.grad is None
    assert torch.count_nonzero(x.grad[mask]) == 0
    assert x.grad[~mask].abs().sum() > 0


def test_reconstruction_target_has_no_dropout_and_restores_training_modes():
    model = DCTV314MaskedTransportReconstruction(make_args()).train()
    data = batch()
    a, b = model._clean_reconstruction_target(data), model._clean_reconstruction_target(data)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not a.requires_grad
    assert all(module.training for module in model.sig_networks.modules())


def test_masked_distance_ignores_unselected_and_unavailable_nan_targets():
    pred = torch.randn(2, 5, 16, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 0] = True
    availability = torch.tensor([True, False])
    original = DCTV314MaskedTransportReconstruction._masked_reconstruction_distance(pred, target, mask, availability)
    target[:, 1:] = float('nan')
    target[1] = float('nan')
    updated = DCTV314MaskedTransportReconstruction._masked_reconstruction_distance(pred, target, mask, availability)
    torch.testing.assert_close(original, updated)
    updated.backward()
    assert torch.isfinite(pred.grad).all()


def test_warmup_skips_reconstruction_and_explicit_epoch_takes_precedence(monkeypatch):
    model = DCTV314MaskedTransportReconstruction(make_args(cur_epoch=7)).train()
    def forbidden(**kwargs):
        raise AssertionError('inactive reconstruction executed')
    monkeypatch.setattr(model, 'masked_transport_reconstruction_losses', forbidden)
    logits, aux = model(**batch(), cur_epoch=0)
    assert model.last_training_losses['v314_mtr_total'] == 0
    assert model.last_training_losses['v314_mtr_weight'] == 0
    (logits.square().mean()+aux).backward()
    assert model.pathway_mask_token.grad is None


def test_checkpoint_roundtrip_restores_references_recipe_and_predictions():
    model = DCTV314MaskedTransportReconstruction(make_args()).train()
    model.configure_train_reference(*train_reference())
    model(**batch(), cur_epoch=9)
    model.eval()
    with torch.no_grad():
        before = model(**batch())[0]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = DCTV314MaskedTransportReconstruction(make_args(cur_epoch=0)).eval()
    restored.load_state_dict(state)
    with torch.no_grad():
        after = restored(**batch())[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert restored.has_train_reference
    assert restored.transport_epoch.item() == 9
    assert restored.args.cur_epoch == 9
    wrong = DCTV314MaskedTransportReconstruction(make_args(dct_v314_lambda_mtr=.05))
    with pytest.raises(RuntimeError, match='recipe'):
        wrong.load_state_dict(state)


def test_ipcw_uses_left_limit_at_tied_event_and_censor_time():
    model = DCTV314MaskedTransportReconstruction(make_args())
    model.configure_train_reference(torch.tensor([1., 1., 2., 3., 4., 5.]),
                                    torch.tensor([0., 1., 0., 0., 0., 1.]))
    assert model._ipcw(torch.tensor([1.]), before=True).item() == 1.
    assert model._ipcw(torch.tensor([1.])).item() == pytest.approx(5/4)
    assert model._ipcw(torch.tensor([0.]), before=True).item() == 1.
    assert model._ipcw(torch.tensor([1])).item() == pytest.approx(5/4)


def test_censoring_reference_matches_reverse_kaplan_meier_with_ties():
    from sksurv.nonparametric import kaplan_meier_estimator
    times = torch.tensor([1., 1., 2., 2., 3., 4., 5., 5.])
    censorship = torch.tensor([0., 1., 0., 1., 0., 0., 0., 1.])
    expected_times, expected_survival = kaplan_meier_estimator(
        (censorship == 0).numpy(), times.numpy(), reverse=True)
    model = DCTV314MaskedTransportReconstruction(make_args())
    model.configure_train_reference(times, censorship)
    torch.testing.assert_close(model.dct_censor_times, torch.as_tensor(expected_times).float())
    torch.testing.assert_close(model.dct_censor_survival, torch.as_tensor(expected_survival).float().clamp_min(.05))


def test_integer_survival_metadata_is_converted_without_truncating_weights():
    model = DCTV314MaskedTransportReconstruction(make_args())
    data = batch()
    data['event_time'] = data['event_time'].long().tolist()
    data['c'] = data['c'].long().tolist()
    converted, _, _ = model._validated_inputs(data)
    assert converted['event_time'].dtype == torch.float32
    assert converted['c'].dtype == torch.float32
    data['event_time'][0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        model(**data)


def test_content_distance_hinge_pushes_nearly_collapsed_slots_apart():
    model = DCTV314MaskedTransportReconstruction(make_args())
    torch.manual_seed(18)
    slots = (torch.randn(2, 1, 16) + .0001*torch.randn(2, 3, 16)).requires_grad_()
    loss = model.slot_diversity_loss(slots, slots)
    loss.backward()
    assert torch.isfinite(slots.grad).all() and slots.grad.norm() > .1
    before = loss.item()
    after = model.slot_diversity_loss(slots.detach()-.01*slots.grad, slots.detach()-.01*slots.grad)
    assert after.item() < before


def test_invalid_transport_hyperparameters_are_rejected():
    for overrides in ({'otehv2_eps': float('nan')}, {'otehv2_layers': 0},
                      {'spt_prog_cost': float('inf')}, {'n_classes': 0}):
        with pytest.raises(ValueError):
            DCTV314MaskedTransportReconstruction(make_args(**overrides))


def test_invalid_inputs_and_weights_fail_explicitly():
    for override in ({'dct_v314_lambda_mtr': float('nan')},
                     {'dct_v314_mask_ratio': 1.}, {'dct_v314_lambda_slot_nll': -.1},
                     {'bag_loss': 'cox_surv'}):
        with pytest.raises(ValueError):
            DCTV314MaskedTransportReconstruction(make_args(**override))
    model = DCTV314MaskedTransportReconstruction(make_args())
    with pytest.raises(ValueError, match='at least one'):
        model(**batch(), wsi_missing=True, omic_missing=True)
    with pytest.raises(ValueError, match='availability mask'):
        model(**batch(), omic_available=[True, False])
    with pytest.raises(ValueError, match='time bin'):
        discrete_nll(torch.zeros(2, 4), torch.tensor([.5, 1.]), torch.zeros(2))


def test_batch_one_unequal_slots_and_one_pathway_backward():
    model = DCTV314MaskedTransportReconstruction(make_args(omic_sizes=[3], slot_num_wsi=2,
                slot_num_omics=4, cur_epoch=7)).train()
    data = {'x_wsi': torch.randn(1, 6, 16), 'x_omic1': torch.randn(1, 3),
            'y': torch.tensor([1]), 'event_time': torch.tensor([2.]), 'c': torch.tensor([1.])}
    logits, aux = model(**data)
    loss = StableNLLSurvLoss(.15)(logits, data['y'], c=data['c'])+aux
    loss.backward()
    assert torch.isfinite(loss)
    assert model._last_pathway_mask.sum() == 1
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_reconstruction_changes_neither_factual_logits_nor_inference():
    model = DCTV314MaskedTransportReconstruction(make_args()).train()
    off = DCTV314MaskedTransportReconstruction(make_args()).train()
    off.load_state_dict(model.state_dict())
    off.reconstruction_mode = 'off'
    data = batch()
    torch.manual_seed(90)
    before = model(**data)[0]
    torch.manual_seed(90)
    after = off(**data)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_duplicate_slot_costs_backward_is_finite():
    model = DCTV314MaskedTransportReconstruction(make_args())
    slots = torch.zeros(2, 3, 16, requires_grad=True)
    costs, rows, cols, _ = model._cost_tensor(slots, slots)
    plans, distance = model._plans_from_cost_tensor(costs, rows, cols, 7)
    logits, _ = model._encode_logits_from_plans(slots, slots, plans)
    (logits.square().mean()+distance).backward()
    assert torch.isfinite(slots.grad).all()
    assert model._marginal_error(plans, rows, cols).max() < 1e-4


def test_slot_identity_cannot_manufacture_content_diversity():
    attention = PrototypeSlotAttention(16, 4, 3)
    repeated = torch.randn(2, 1, 16).expand(-1, 10, -1)
    result, pooling = attention(repeated, torch.randn(4, 16))
    torch.testing.assert_close(result, repeated[:, :1].expand(-1, 4, -1))
    torch.testing.assert_close(pooling.sum(-1), torch.ones(2, 4))


@pytest.mark.parametrize('mode', ['masked', 'full', 'hybrid', 'off'])
def test_reconstruction_modes_use_single_budget(mode):
    torch.manual_seed(76)
    model = DCTV314MaskedTransportReconstruction(make_args(dct_v314_reconstruction_mode=mode)).train()
    logits, aux = model(**batch())
    d = model.last_training_losses
    expected = .10*d['ipcw_rank'] + .05*d['v314_per_slot_nll'] + .02*d['v314_slot_diversity'] + d['v314_mtr_weight']*d['v314_mtr_total']
    torch.testing.assert_close(aux, expected)
    assert float(d['v314_mtr_weight']) == pytest.approx(0 if mode=='off' else .1)
    (logits.square().mean()+aux).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_missing_omics_changes_predictions_and_drops_invalid_values():
    model = DCTV314MaskedTransportReconstruction(make_args()).eval()
    data = batch()
    with torch.no_grad():
        full = model(**data)[0]
        missing = model(**data, omic_missing=True)[0]
        for name in data:
            if name.startswith('x_omic'):
                data[name][:] = float('nan')
        missing_nan = model(**data, omic_missing=True)[0]
    assert not torch.allclose(full, missing)
    torch.testing.assert_close(missing, missing_nan)
    with pytest.raises(ValueError, match='nonfinite'):
        model(**data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('amp', [False, True])
def test_cuda_forward_backward_and_optimizer_step(amp):
    torch.manual_seed(81)
    model = DCTV314MaskedTransportReconstruction(make_args()).cuda().train()
    model.configure_train_reference(*train_reference())
    data = {k:v.cuda() for k,v in batch().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    with torch.autocast('cuda', enabled=amp, dtype=torch.float16):
        logits, aux = model(**data)
        nll = StableNLLSurvLoss(alpha=.15)(logits, data['y'], c=data['c'])/len(logits)
        loss = nll+aux
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_shared_training_loop_and_batch_adapter_checkpoint_compatibility():
    from survot_rank.training.train_runner import train_one_epoch, init_loss_function, _process_data_and_forward
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = make_args(survot_method='dct_v314', batch_size=8, max_epochs=30, cur_fold=0,
                     grad_accum_steps=1, grad_clip_norm=1., max_smoke_batches=0)
    model = DCTV314MaskedTransportReconstruction(args).to(device)
    model.configure_train_reference(*train_reference())
    data = batch()
    pathways = [[data[f'x_omic{p}'][i] for p in range(1, 6)] for i in range(8)]
    packed = (data['x_wsi'], pathways, data['y'], data['event_time'], data['c'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    diagnostics = train_one_epoch(args, 7, model, [packed, packed], optimizer,
                                  scheduler, init_loss_function(args), io.StringIO())
    assert all(math.isfinite(float(value)) for value in diagnostics.values())
    assert diagnostics['v314_mtr_weight'] == pytest.approx(.1)
    model.eval()
    restored = DCTV314MaskedTransportReconstruction(make_args(cur_epoch=0)).to(device).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        before = _process_data_and_forward(args, model, packed, device, test=True)[0][0]
        after = _process_data_and_forward(restored.args, restored, packed, device, test=True)[0][0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
