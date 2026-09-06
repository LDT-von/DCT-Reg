from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import run_dct_v32_tgsr_optimization as queue
from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.research.components.transport_guided_slot_reaggregation import (
    TransportGuidedSlotReaggregation,
)
from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.research.methods.dct_v32_tgsr_objective_study import (
    DCTV32TGSRObjectiveStudy,
)
from survot_rank.research.methods.dct_v32_transport_guided_slot_reaggregation import (
    DCTV32TransportGuidedSlotReaggregation,
)
from survot_rank.training.extended_args import build_base_parser


def model_args(**overrides):
    values = dict(
        bag_loss="nll_surv",
        omic_sizes=None,
        n_classes=4,
        encoding_dim=16,
        wsi_projection_dim=16,
        rna_format="RNASeq",
        slot_num_wsi=3,
        slot_num_omics=4,
        slot_iters=2,
        alpha_surv=0.15,
        otehv2_eps=0.1,
        otehv2_iter=20,
        otehv2_heads=2,
        otehv2_layers=1,
        otehv2_dropout=0.0,
        dct_num_stages=4,
        dct_coupling_projection_iters=30,
        dct_coupling_projection_tol=1e-4,
        cur_epoch=12,
        dct_v32_feedback="ot",
        dct_v32_feedback_iters=30,
        dct_v32_objective="full",
        dct_v32_learnable_feedback_strength=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def payload(batch=4):
    return dict(
        x_wsi=torch.randn(batch, 7, 16),
        x_omics=torch.randn(batch, 5, 20),
        event_time=torch.tensor([1.0, 3.0, 5.0, 8.0])[:batch],
        c=torch.tensor([0.0, 0.0, 0.0, 1.0])[:batch],
    )


def test_learnable_feedback_strength_is_bounded_and_trainable():
    module = TransportGuidedSlotReaggregation(
        16, mode="ot", strength=0.25, learnable_strength=True
    )
    wsi_slots = torch.randn(2, 3, 16)
    omic_slots = torch.randn(2, 4, 16)
    wsi_tokens = torch.randn(2, 7, 16)
    omic_tokens = torch.randn(2, 5, 16)
    output = module(wsi_slots, omic_slots, wsi_tokens, omic_tokens)
    torch.testing.assert_close(
        module.last_diagnostics["feedback_strength"], torch.tensor(0.25)
    )
    sum(item.square().mean() for item in output).backward()
    gradient = module.feedback_strength_logit.grad
    assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0
    assert 0 < module.effective_strength(wsi_slots) < 1


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("nll", {"nll": 1.0, "ipcw_rank": 0.0, "direction": 0.0}),
        ("ipcw", {"nll": 1.0, "ipcw_rank": 0.10, "direction": 0.0}),
        ("direction", {"nll": 1.0, "ipcw_rank": 0.0, "direction": 0.05}),
        ("full", {"nll": 1.0, "ipcw_rank": 0.10, "direction": 0.05}),
    ],
)
def test_objective_study_activates_only_selected_terms(objective, expected):
    model = DCTV32TGSRObjectiveStudy(
        model_args(dct_v32_objective=objective), omic_input_dim=20
    )
    model.configure_train_reference(torch.arange(1.0, 9.0), torch.zeros(8))
    logits, auxiliary = model(**payload())
    assert logits.shape == (4, 4)
    assert torch.isfinite(auxiliary)
    assert model.objective_weights() == expected
    assert model.dct_lambda_ipcw_rank == expected["ipcw_rank"]
    assert model.dct_v38_lambda_direction == expected["direction"]
    assert model.last_training_losses["tgsr_feedback_strength"] == pytest.approx(0.25)
    if objective == "nll":
        assert auxiliary == 0
    else:
        assert auxiliary > 0


def test_nll_study_preserves_existing_tgsr_factual_path():
    torch.manual_seed(7)
    original = DCTV32TransportGuidedSlotReaggregation(
        model_args(dct_v32_objective="nll"), omic_input_dim=20
    ).eval()
    torch.manual_seed(7)
    study = DCTV32TGSRObjectiveStudy(
        model_args(dct_v32_objective="nll"), omic_input_dim=20
    ).eval()
    assert set(original.state_dict()) == set(study.state_dict())
    inputs = payload()
    with torch.no_grad():
        expected, _ = original(**inputs)
        actual, _ = study(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_catalog_and_matched_optimization_queue():
    key = METHOD_ALIASES["tgsr_optimized"]
    assert key == queue.STUDY_METHOD
    assert METHOD_CATALOG[key].status == "candidate"
    args = queue.build_parser().parse_args(["plan", "--python", "python"])
    jobs = queue.build_jobs(args)
    assert len(jobs) == len(queue.VARIANTS) * 5
    assert len({job.result_dir for job in jobs}) == len(jobs)
    for job in jobs:
        overrides = [
            job.command[index + 1]
            for index, item in enumerate(job.command[:-1])
            if item == "--set"
        ]
        parsed = build_base_parser().parse_args(
            config_to_argv(apply_overrides(load_config(job.config), overrides))
        )
        expected = queue.VARIANTS[job.variant]
        assert parsed.survot_method == expected["survot_method"]
        assert parsed.dct_v32_objective == expected["dct_v32_objective"]
        assert (
            parsed.dct_v32_learnable_feedback_strength
            is expected["dct_v32_learnable_feedback_strength"]
        )
        assert parsed.k_start == job.fold and parsed.k_end == job.fold + 1
        assert parsed.max_epochs == 50 and parsed.seed == 3
    assert len(queue.build_jobs(args, smoke=True)) == len(queue.VARIANTS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learnable_strength": "yes"},
        {"learnable_strength": True, "strength": 0.0},
        {"learnable_strength": True, "strength": 1.0},
    ],
)
def test_invalid_learnable_strength_settings_fail(kwargs):
    with pytest.raises(ValueError):
        TransportGuidedSlotReaggregation(16, **kwargs)
