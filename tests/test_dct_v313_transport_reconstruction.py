from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.research.methods.legacy.experimental.dct_v313_transport_reconstruction.model import (
    DCTV313TransportReconstruction,
)
from survot_rank.training.model_factory import get_model


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
        dct_v311_lambda_slot_nll=0.99,
        dct_v311_lambda_slot_diversity=0.99,
        dct_v313_lambda_reconstruction=0.99,
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


def test_v313_catalog_alias_factory_and_config():
    key = "dct_v313_transport_reconstruction"
    assert METHOD_ALIASES["dct_v313"] == key
    assert METHOD_CATALOG[key].status == "candidate"
    model = get_model("dct_v313", make_args())
    assert type(model).__name__ == "DCTV313TransportReconstruction"

    config = yaml.safe_load(
        Path("configs/dct_v313_blca_uni.yaml").read_text(encoding="utf-8-sig")
    )
    assert config["train"]["survot_method"] == key
    assert config["model"]["wsi_encoder"] == "uni"
    assert config["data"]["rna_format"] == "Pathways"


def test_v313_frozen_objective_and_reconstruction_schedule():
    model = DCTV313TransportReconstruction(make_args())
    assert model.objective_weights() == {
        "nll": 1.0,
        "ipcw_rank": 0.10,
        "per_slot_nll": 0.05,
        "slot_diversity": 0.10,
        "reconstruction": 0.10,
    }
    assert model.dct_v38_lambda_direction == 0.0
    assert model.dct_v313_lambda_reconstruction == 0.10
    assert model.dct_v313_reconstruction_self_fraction == 0.50
    assert model.dct_v313_reconstruction_cross_fraction == 0.50
    assert model._reconstruction_ramp(0) == 0.0
    assert model._reconstruction_ramp(2) == 0.0
    assert model._reconstruction_ramp(4) == pytest.approx(0.4)
    assert model._reconstruction_ramp(7) == 1.0
    assert model._reconstruction_ramp(30) == 1.0


@pytest.mark.parametrize("event", [True, False], ids=["event", "censored"])
def test_v313_per_slot_nll_uses_cumulative_survival(event):
    hazard = torch.full((1, 1, 4), 0.5, dtype=torch.float64, requires_grad=True)
    labels = F.one_hot(torch.tensor([3]), num_classes=4).to(torch.float64)
    event_mask = torch.tensor([[float(event)]], dtype=torch.float64)
    loss = DCTV313TransportReconstruction._nll_surv_per_slot(
        None,
        hazard,
        labels,
        event_mask,
        1.0 - event_mask,
        torch.ones(1, 1, dtype=torch.float64),
    )
    expected = torch.tensor([[4.0 * math.log(2.0)]], dtype=torch.float64)
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(loss.sum(), hazard)[0]
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient[..., :3] > 0)


def test_transport_reconstruction_has_wsi_omic_and_sinkhorn_gradients():
    torch.manual_seed(5)
    model = DCTV313TransportReconstruction(make_args())
    model.train()
    slots_wsi = torch.randn(3, 3, 16, requires_grad=True)
    slots_omic = torch.randn(3, 3, 16, requires_grad=True)
    target = torch.randn(3, 5, 16)
    costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, epoch=7)
    _, gate = model._encode_logits_from_plans(slots_wsi, slots_omic, plans)

    self_loss, cross_loss, total, transported = model.reconstruction_losses(
        x_omics=target,
        slots_wsi=slots_wsi,
        slots_omic=slots_omic,
        factual_plans=plans,
        factual_gate=gate,
        available=torch.ones(3, dtype=torch.bool),
    )
    total.backward()

    assert transported.shape == slots_omic.shape
    assert self_loss > 0
    assert cross_loss > 0
    assert slots_wsi.grad is not None and slots_wsi.grad.abs().sum() > 0
    assert slots_omic.grad is not None and slots_omic.grad.abs().sum() > 0
    assert model.pathway_reconstruction_decoder.pathway_queries.grad is not None
    assert model.pathway_reconstruction_decoder.pathway_queries.grad.abs().sum() > 0
    assert model.stage_pair_cost[-1].weight.grad is not None
    assert model.stage_pair_cost[-1].weight.grad.abs().sum() > 0


def test_v313_training_forward_is_exact_declared_auxiliary_sum():
    torch.manual_seed(17)
    model = DCTV313TransportReconstruction(make_args(cur_epoch=7))
    model.configure_train_reference(*train_reference())
    model.train()

    logits, auxiliary = model(**batch())
    diagnostics = model.last_training_losses
    expected = (
        0.10 * diagnostics["ipcw_rank"]
        + 0.05 * diagnostics["v311_per_slot_nll"]
        + 0.10 * diagnostics["v311_slot_diversity"]
        + 0.10 * diagnostics["v313_reconstruction_total"]
    )

    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary)
    torch.testing.assert_close(auxiliary, expected)
    assert diagnostics["v313_reconstruction_self"] > 0
    assert diagnostics["v313_reconstruction_cross"] > 0
    assert diagnostics["v313_reconstruction_ramp"] == 1.0
    assert diagnostics["v313_reconstruction_weight"] == pytest.approx(0.10)

    auxiliary.backward()
    decoder_grad = model.pathway_reconstruction_decoder.pathway_queries.grad
    assert decoder_grad is not None and decoder_grad.abs().sum() > 0


def test_v313_eval_preserves_factual_prediction_and_audit_outputs():
    torch.manual_seed(23)
    model = DCTV313TransportReconstruction(make_args())
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


def test_v313_rejects_non_pathway_omics():
    with pytest.raises(ValueError, match="requires rna_format='Pathways'"):
        DCTV313TransportReconstruction(make_args(rna_format="RNASeq"), omic_input_dim=20)
